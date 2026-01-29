#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_genre_hpc_true_rag_ollama_client_v2.py

Fixes added vs prior version:
1) Determinism: sets Ollama chat options (temperature=0, top_p=1; optional seed).
2) Enforced contract:
   - status must be one of {PROPOSED, REVIEW, ERROR}
   - updated_term must be EXACTLY one of retrieved_terms or ""
   - otherwise: force status=REVIEW, updated_term="", confidence=0.0 and set error.
3) Better normalization for retrieval:
   - strips trailing punctuation/codes
   - applies a small UK->US spelling normalization (catalogue(s)->catalog(s), etc.)
   - injects exact normalized matches to the front of retrieved_terms when available.
4) Defaults match your desired HPC settings:
   - batch size default 25
   - worker threads default 4

Keeps:
- True vector retrieval via ollama.embeddings() (no hardcoded host)
- Local vector cache: rbms_vectors.npy + rbms_texts.json
- Resume via genre_llm_results.csv
- Output merge into final CSV
- JSON extraction/parsing workflow
"""

import os
import csv
import json
import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ollama import chat
try:
    from ollama import embeddings as ollama_embeddings  # type: ignore
except Exception:
    ollama_embeddings = None


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:32b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", MODEL_NAME)

# Determinism controls
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0"))
TOP_P = float(os.environ.get("OLLAMA_TOP_P", "1"))
SEED = os.environ.get("OLLAMA_SEED", "").strip()
SEED = int(SEED) if SEED.isdigit() else None

# Files
OLD_FILE = "RBMSgenre_forHenry All Columns-Sample-700.csv"
RESULTS_FILE = os.environ.get("GENRE_RESULTS_FILE", "genre_llm_results.csv")
FINAL_OUTPUT = os.environ.get("GENRE_FINAL_OUTPUT", "RBMSgenre_with_updated_terms.csv")

RBMS_COL = os.environ.get("GENRE_RBMS_COL", "655 - Local Param 04")
UPDATED_COL = os.environ.get("GENRE_UPDATED_COL", "Updated RBMS Genre Term")
EXTRANEOUS_COL = os.environ.get("GENRE_EXTRANEOUS_COL", "RBMS Extraneous Text")

RBMS_TERMS_FILE = os.environ.get("RBMS_TERMS_FILE", "rbms_terms.json")
VEC_CACHE_NPY = os.environ.get("RBMS_VEC_CACHE_NPY", "rbms_vectors.npy")
TXT_CACHE_JSON = os.environ.get("RBMS_TXT_CACHE_JSON", "rbms_texts.json")

TOP_K = int(os.environ.get("RAG_TOP_K", "25"))
BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "25"))
MAX_WORKERS = int(os.environ.get("OLLAMA_MAX_WORKERS", "4"))

SLEEP_BETWEEN_BATCHES_SECONDS = float(os.environ.get("SLEEP_BETWEEN_BATCHES_SECONDS", "0.0"))
PROGRESS_EVERY_BATCHES = int(os.environ.get("PROGRESS_EVERY_BATCHES", "1"))

ALLOWED_STATUS = {"PROPOSED", "REVIEW", "ERROR"}

START_TIME = time.perf_counter()


# -----------------------------
# NORMALIZATION
# -----------------------------
_CODE_TAIL_RE = re.compile(r"(?:\s+|\.|,)*(rbmscv|rbgenr|rbpap|rbprov|aat|lcgft|fast)\b.*$", re.IGNORECASE)

# tiny UK->US spelling helpers (add more if you see patterns)
_UK_US = {
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "cataloguing": "cataloging",
    "catalogued": "cataloged",
    "theatre": "theater",
    "colour": "color",
    "favourite": "favorite",
    "honour": "honor",
}

def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()

def clean_for_retrieval(s: str) -> str:
    """Make the query closer to RBMS term surface forms before embedding."""
    t = norm_term(s)

    # Remove trailing code tails like ". rbgenr" etc.
    t = _CODE_TAIL_RE.sub("", t).strip()

    # Remove trailing punctuation
    t = t.rstrip(" .;,")

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Apply UK->US word-level replacements (case-insensitive, whole-word)
    def _swap_word(m):
        w = m.group(0)
        key = w.lower()
        rep = _UK_US.get(key)
        return rep if rep is not None else w

    if t:
        pattern = r"\b(" + "|".join(re.escape(k) for k in _UK_US.keys()) + r")\b"
        t = re.sub(pattern, _swap_word, t, flags=re.IGNORECASE)

    return t

def normalize_key_for_exact_match(s: str) -> str:
    t = clean_for_retrieval(s).lower()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# JSON parsing helpers
# -----------------------------
def _parse_all_json_objects(text: str):
    if not text:
        raise ValueError("Empty response text")

    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    if s.startswith("["):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("Expected JSON array")
        return [x for x in arr if isinstance(x, dict)]

    objs = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(s):
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(s[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + consumed
        except Exception:
            i = j + 1

    if not objs:
        raise ValueError(f"No JSON objects found | snippet: {s[:200]}")
    return objs


def _extract_text_from_ollama_response(raw):
    if isinstance(raw, dict):
        msg = raw.get("message")
        if isinstance(msg, dict):
            return norm_term(msg.get("content", ""))
        return norm_term(raw.get("response", ""))

    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return norm_term(obj.get("response", "")) or norm_term(obj.get("message", {}).get("content", ""))
            except Exception:
                return s
        return s

    return ""


# -----------------------------
# Ollama calls (client handles host)
# -----------------------------
def post_hpc(prompt: str):
    options = {"temperature": TEMPERATURE, "top_p": TOP_P}
    if SEED is not None:
        options["seed"] = SEED
    return chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=options,
    )


def ollama_embed(text: str) -> np.ndarray:
    if ollama_embeddings is None:
        raise RuntimeError(
            "The installed 'ollama' Python package in this environment does not expose embeddings(). "
            "If your environment supports /api/embeddings, upgrade the ollama python package, or ask me to add an HTTP fallback."
        )

    resp = ollama_embeddings(model=EMBED_MODEL, prompt=text)
    emb = resp.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError(f"Unexpected embeddings() response (no embedding). Keys: {list(resp.keys())}")
    return np.asarray(emb, dtype=np.float32)


# -----------------------------
# Vector index (RBMS terms)
# -----------------------------
def _l2_normalize_rows(V: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def load_or_build_rbms_vector_index(rbms_terms):
    if os.path.exists(VEC_CACHE_NPY) and os.path.exists(TXT_CACHE_JSON):
        try:
            V = np.load(VEC_CACHE_NPY)
            with open(TXT_CACHE_JSON, "r", encoding="utf-8") as f:
                T = json.load(f)
            if isinstance(T, list) and len(T) == V.shape[0]:
                return T, V
        except Exception:
            pass

    T = [norm_term(t) for t in rbms_terms if norm_term(t)]
    if not T:
        raise SystemExit("No RBMS terms loaded; cannot build retrieval index.")

    first = ollama_embed(T[0])
    dim = int(first.shape[0])

    V = np.zeros((len(T), dim), dtype=np.float32)
    V[0] = first

    for i in range(1, len(T)):
        V[i] = ollama_embed(T[i])
        if (i + 1) % 200 == 0 or (i + 1) == len(T):
            print(f"Embedded {i+1}/{len(T)} RBMS terms for vector index")

    V = _l2_normalize_rows(V)
    np.save(VEC_CACHE_NPY, V)
    with open(TXT_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(T, f, ensure_ascii=False)

    return T, V


def retrieve_top_k(query_raw: str, texts, V: np.ndarray, k: int = TOP_K, exact_map=None):
    q_clean = clean_for_retrieval(query_raw)
    if not q_clean:
        return []

    qv = ollama_embed(q_clean)
    qn = qv / (np.linalg.norm(qv) or 1.0)

    scores = V @ qn
    k_eff = min(k, scores.shape[0])
    if k_eff <= 0:
        return []

    idx = np.argpartition(scores, -k_eff)[-k_eff:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    retrieved = [texts[i] for i in idx]

    if exact_map is not None:
        key = normalize_key_for_exact_match(query_raw)
        canonical = exact_map.get(key)
        if canonical and canonical not in retrieved:
            retrieved.insert(0, canonical)
        elif canonical and retrieved and retrieved[0] != canonical:
            retrieved = [canonical] + [t for t in retrieved if t != canonical]

    return retrieved[:k_eff]


# -----------------------------
# LOAD RBMS TERMS + BUILD/LOAD INDEX
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE}")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)
RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
print(f"Loaded {len(RBMS_TERMS)} RBMS terms")

EXACT_MAP = {}
for t in RBMS_TERMS:
    k = normalize_key_for_exact_match(t)
    EXACT_MAP.setdefault(k, t)

print(f"Building/loading vector index (embed_model={EMBED_MODEL})")
RAG_TERMS, RAG_VECS = load_or_build_rbms_vector_index(RBMS_TERMS)
print(f"Vector index ready: {len(RAG_TERMS)} terms, dim={int(RAG_VECS.shape[1])}")


# -----------------------------
# LOAD INPUT CSV
# -----------------------------
print(f"Loading input CSV: {OLD_FILE}")

with open(OLD_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise SystemExit(f"No headers found in {OLD_FILE}")

    headers = list(reader.fieldnames)
    if RBMS_COL not in headers:
        raise SystemExit(f"Column '{RBMS_COL}' not found in {OLD_FILE}")

    old_rows = []
    all_terms = []
    for row in reader:
        for h in headers:
            if h not in row or row[h] is None:
                row[h] = ""
        term = norm_term(row.get(RBMS_COL, ""))
        row[RBMS_COL] = term
        old_rows.append(row)
        if term:
            all_terms.append(term)

unique_terms = sorted(set(all_terms))
print(f"Total rows: {len(old_rows)} | Non-empty terms: {len(all_terms)} | Unique terms: {len(unique_terms)}")


# -----------------------------
# RESUME SUPPORT (RESULTS_FILE)
# -----------------------------
existing_results = {}
file_exists = os.path.exists(RESULTS_FILE)

if file_exists:
    print(f"Found existing results file '{RESULTS_FILE}', loading for resume.")
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        df_existing = csv.DictReader(f)
        for row in df_existing:
            key = norm_term(row.get("original"))
            existing_results[key] = {
                "original": key,
                "status": row.get("status", ""),
                "updated_term": row.get("updated_term", ""),
                "extraneous_text": row.get("extraneous_text", ""),
                "confidence": float(row.get("confidence", 0.0)) if row.get("confidence") else 0.0,
                "error": row.get("error", ""),
            }
else:
    print("No existing results file. A new one will be created.")

processed_terms = set(existing_results.keys())
print(f"Already have {len(processed_terms)} terms in {RESULTS_FILE}")

csv_is_new = (not file_exists) or (os.path.getsize(RESULTS_FILE) == 0)

results_fh = open(RESULTS_FILE, "a", newline="", encoding="utf-8")
fieldnames = ["original", "status", "updated_term", "extraneous_text", "confidence", "error"]
writer = csv.DictWriter(results_fh, fieldnames=fieldnames)
if csv_is_new:
    writer.writeheader()
    results_fh.flush()


# -----------------------------
# BATCHED LLM CALL (TRUE RAG + batching + enforcement)
# -----------------------------
def ask_llm_batch(terms_batch):
    items = []
    retrieved_by_original = {}

    for term in terms_batch:
        orig = norm_term(term)
        retrieved = retrieve_top_k(orig, RAG_TERMS, RAG_VECS, k=TOP_K, exact_map=EXACT_MAP)
        retrieved_by_original[orig] = retrieved
        items.append({"original": orig, "retrieved_terms": retrieved})

    prompt = f"""
You are an expert rare-books cataloger specializing in RBMS Controlled Vocabularies.

For EACH item below:
- Choose the single best RBMS term from retrieved_terms (or empty if none fits).
- updated_term MUST be EXACTLY one of retrieved_terms or "".
- Put everything else from the input string into extraneous_text (codes like rbmscv/rbpap/aat/lcgft, places, dates, etc).
- If none fits, set status="REVIEW", updated_term="", confidence=0.0.

STRICT OUTPUT RULES:
- Return JSON LINES ONLY (one JSON object per line).
- No prose. No markdown.
- status must be EXACTLY one of: PROPOSED, REVIEW, ERROR

Each JSON object must contain:
original, status, updated_term, extraneous_text, confidence, error

ITEMS (JSON):
{json.dumps(items, ensure_ascii=False)}
""".strip()

    base = {}
    for t in terms_batch:
        tn = norm_term(t)
        base[tn] = {"original": tn, "status": "ERROR", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": ""}

    try:
        response = post_hpc(prompt)
        txt = _extract_text_from_ollama_response(response["message"]["content"])
        objs = _parse_all_json_objects(txt)
    except Exception as e:
        err = f"Batch error: {e}"
        for tn in base:
            base[tn]["error"] = err
        return list(base.values())

    out_map = {}
    for o in objs:
        if isinstance(o, dict):
            orig = norm_term(o.get("original"))
            if orig:
                out_map[orig] = o

    normalized = []
    for t in terms_batch:
        tn = norm_term(t)
        raw = out_map.get(tn)

        if not raw:
            r = base[tn]
            r["status"] = "ERROR"
            r["error"] = "Missing result from batch response"
            normalized.append(r)
            continue

        status = str(raw.get("status", "PROPOSED")).strip().upper() or "PROPOSED"
        updated = norm_term(raw.get("updated_term", ""))
        extr = norm_term(raw.get("extraneous_text", ""))
        err = norm_term(raw.get("error", ""))
        try:
            conf = float(raw.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        allowed_retrieved = set(retrieved_by_original.get(tn, []))
        problems = []

        if status not in ALLOWED_STATUS:
            problems.append(f"Invalid status '{status}'")
            status = "REVIEW"

        if updated and updated not in allowed_retrieved:
            problems.append("updated_term not in retrieved_terms")
            updated = ""
            conf = 0.0
            status = "REVIEW"

        if status == "PROPOSED" and not updated:
            status = "REVIEW"
            if not err:
                err = "No updated_term provided"

        if problems:
            err = (err + " | " if err else "") + "; ".join(problems)

        normalized.append(
            {
                "original": tn,
                "status": status,
                "updated_term": updated,
                "extraneous_text": extr,
                "confidence": conf,
                "error": err,
            }
        )

    return normalized


# -----------------------------
# PROCESS TERMS
# -----------------------------
terms_to_process = [t for t in unique_terms if norm_term(t) not in processed_terms]
print(f"Unique terms needing LLM after resume: {len(terms_to_process)}")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

batches = list(chunked(terms_to_process, BATCH_SIZE))
print(f"Batches to process: {len(batches)} | batch_size={BATCH_SIZE} | max_workers={MAX_WORKERS} | top_k={TOP_K}")

if batches:
    completed_batches = 0
    completed_terms = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(ask_llm_batch, batch): batch for batch in batches}

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                results = future.result()
            except Exception as e:
                results = [
                    {"original": norm_term(t), "status": "ERROR", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": str(e)}
                    for t in batch
                ]

            for result in results:
                term_norm = norm_term(result.get("original"))
                existing_results[term_norm] = result

                writer.writerow(
                    {
                        "original": term_norm,
                        "status": result.get("status", ""),
                        "updated_term": result.get("updated_term", ""),
                        "extraneous_text": result.get("extraneous_text", ""),
                        "confidence": float(result.get("confidence", 0.0)),
                        "error": result.get("error", ""),
                    }
                )
                completed_terms += 1

            results_fh.flush()
            completed_batches += 1

            if (completed_batches % PROGRESS_EVERY_BATCHES) == 0 or completed_batches == len(batches):
                print(f"Completed {completed_batches}/{len(batches)} batches | wrote {completed_terms} term results")

            if SLEEP_BETWEEN_BATCHES_SECONDS:
                time.sleep(SLEEP_BETWEEN_BATCHES_SECONDS)

else:
    print("No new terms left to process.")

results_fh.close()
print(f"All term-level results stored in '{RESULTS_FILE}'.")


# -----------------------------
# BUILD FINAL OUTPUT
# -----------------------------
print("Building final output CSV.")

term_to_updated = {}
term_to_extraneous = {}

with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = norm_term(row.get("original"))
        term_to_updated[t] = norm_term(row.get("updated_term"))
        term_to_extraneous[t] = norm_term(row.get("extraneous_text"))

out_headers = list(headers)
if UPDATED_COL not in out_headers:
    out_headers.append(UPDATED_COL)
if EXTRANEOUS_COL not in out_headers:
    out_headers.append(EXTRANEOUS_COL)

with open(FINAL_OUTPUT, "w", newline="", encoding="utf-8") as out_f:
    writer_out = csv.DictWriter(out_f, fieldnames=out_headers)
    writer_out.writeheader()

    for row in old_rows:
        term_norm = norm_term(row.get(RBMS_COL, ""))
        row[UPDATED_COL] = term_to_updated.get(term_norm, "")
        row[EXTRANEOUS_COL] = term_to_extraneous.get(term_norm, "")
        writer_out.writerow(row)

elapsed = time.perf_counter() - START_TIME
print(f"Final output written to '{FINAL_OUTPUT}'")
print(f"Done. Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
