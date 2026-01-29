#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_genre_hpc_true_rag_ollama_client_v3.py

Why you were seeing "Missing result from batch response":
- With batch_size=25 and top_k=25, the prompt can get VERY large (25 items x 25 candidates).
  Models commonly truncate or skip some items under token pressure.

This version fixes that by:
1) Sending fewer candidates per item in the PROMPT (defaults to 10) while still retrieving TOP_K for enforcement.
2) Adding automatic retry for any missing items (2nd pass) using smaller batches and fewer candidates.
3) Optionally increasing generation budget via options['num_predict'] (if your Ollama build honors it).

Keeps:
- true vector RAG (ollama.embeddings) with local cache
- deterministic options (temperature=0, top_p=1, optional seed)
- strict enforcement: updated_term must be exactly in candidates
- batching + 4 workers by default
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

TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0"))
TOP_P = float(os.environ.get("OLLAMA_TOP_P", "1"))
SEED = os.environ.get("OLLAMA_SEED", "").strip()
SEED = int(SEED) if SEED.isdigit() else None

# Generation budget (may be ignored by some builds, but harmless)
NUM_PREDICT = os.environ.get("OLLAMA_NUM_PREDICT", "").strip()
NUM_PREDICT = int(NUM_PREDICT) if NUM_PREDICT.isdigit() else None

OLD_FILE = "RBMSgenre_forHenry All Columns-Sample-700.csv"
RESULTS_FILE = os.environ.get("GENRE_RESULTS_FILE", "genre_llm_results.csv")
FINAL_OUTPUT = os.environ.get("GENRE_FINAL_OUTPUT", "RBMSgenre_with_updated_terms.csv")

RBMS_COL = os.environ.get("GENRE_RBMS_COL", "655 - Local Param 04")
UPDATED_COL = os.environ.get("GENRE_UPDATED_COL", "Updated RBMS Genre Term")
EXTRANEOUS_COL = os.environ.get("GENRE_EXTRANEOUS_COL", "RBMS Extraneous Text")

RBMS_TERMS_FILE = os.environ.get("RBMS_TERMS_FILE", "rbms_terms.json")
VEC_CACHE_NPY = os.environ.get("RBMS_VEC_CACHE_NPY", "rbms_vectors.npy")
TXT_CACHE_JSON = os.environ.get("RBMS_TXT_CACHE_JSON", "rbms_texts.json")

# Retrieval size for enforcement (can stay 25)
TOP_K = int(os.environ.get("RAG_TOP_K", "25"))

# How many candidates to actually send to the model per item (reduce prompt size)
TOP_K_FOR_PROMPT = int(os.environ.get("RAG_TOP_K_FOR_PROMPT", "10"))

BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "25"))
MAX_WORKERS = int(os.environ.get("OLLAMA_MAX_WORKERS", "4"))

# Retry behavior
RETRY_MISSING = os.environ.get("RETRY_MISSING", "1").strip().lower() not in {"0", "false", "no"}
RETRY_BATCH_SIZE = int(os.environ.get("RETRY_BATCH_SIZE", "8"))
RETRY_TOP_K_FOR_PROMPT = int(os.environ.get("RETRY_TOP_K_FOR_PROMPT", "8"))
RETRY_PASSES = int(os.environ.get("RETRY_PASSES", "1"))

SLEEP_BETWEEN_BATCHES_SECONDS = float(os.environ.get("SLEEP_BETWEEN_BATCHES_SECONDS", "0.0"))
PROGRESS_EVERY_BATCHES = int(os.environ.get("PROGRESS_EVERY_BATCHES", "1"))

ALLOWED_STATUS = {"PROPOSED", "REVIEW", "ERROR"}
START_TIME = time.perf_counter()


# -----------------------------
# NORMALIZATION
# -----------------------------
_CODE_TAIL_RE = re.compile(r"(?:\s+|\.|,)*(rbmscv|rbgenr|rbpap|rbprov|aat|lcgft|fast|rbbin)\b.*$", re.IGNORECASE)
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
    t = norm_term(s)
    t = _CODE_TAIL_RE.sub("", t).strip()
    t = t.rstrip(" .;,")
    t = re.sub(r"\s+", " ", t).strip()

    def _swap_word(m):
        w = m.group(0)
        rep = _UK_US.get(w.lower())
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
# JSON helpers
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
# Ollama calls
# -----------------------------
def post_hpc(prompt: str):
    options = {"temperature": TEMPERATURE, "top_p": TOP_P}
    if SEED is not None:
        options["seed"] = SEED
    if NUM_PREDICT is not None:
        options["num_predict"] = NUM_PREDICT

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
# Vector index
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

def retrieve_top_k(query_raw: str, texts, V: np.ndarray, k: int, exact_map):
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

    key = normalize_key_for_exact_match(query_raw)
    canonical = exact_map.get(key)
    if canonical:
        if canonical not in retrieved:
            retrieved.insert(0, canonical)
        elif retrieved and retrieved[0] != canonical:
            retrieved = [canonical] + [t for t in retrieved if t != canonical]

    return retrieved[:k_eff]


# -----------------------------
# LOAD RBMS TERMS + INDEX
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE}")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)
RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
print(f"Loaded {len(RBMS_TERMS)} RBMS terms")

EXACT_MAP = {}
for t in RBMS_TERMS:
    EXACT_MAP.setdefault(normalize_key_for_exact_match(t), t)

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
# RESUME SUPPORT
# -----------------------------
existing_results = {}
file_exists = os.path.exists(RESULTS_FILE)

if file_exists:
    print(f"Found existing results file '{RESULTS_FILE}', loading for resume.")
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        df_existing = csv.DictReader(f)
        for row in df_existing:
            key = norm_term(row.get("original"))
            if key:
                existing_results[key] = row

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
# BATCHED LLM CALL
# -----------------------------
def _single_pass(terms_batch, top_k_for_prompt: int):
    items = []
    retrieved_prompt = {}

    for term in terms_batch:
        orig = norm_term(term)
        full = retrieve_top_k(orig, RAG_TERMS, RAG_VECS, k=TOP_K, exact_map=EXACT_MAP)
        prompt_list = full[: max(1, min(top_k_for_prompt, len(full)))]
        retrieved_prompt[orig] = prompt_list
        items.append({"original": orig, "retrieved_terms": prompt_list})

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

    base = {norm_term(t): {"original": norm_term(t), "status": "ERROR", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": ""} for t in terms_batch}

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
            r["confidence"] = 0.0
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

        allowed = set(retrieved_prompt.get(tn, []))
        problems = []

        if status not in ALLOWED_STATUS:
            problems.append(f"Invalid status '{status}'")
            status = "REVIEW"

        if updated and updated not in allowed:
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


def ask_llm_batch(terms_batch):
    results = _single_pass(terms_batch, top_k_for_prompt=TOP_K_FOR_PROMPT)

    if not RETRY_MISSING:
        return results

    missing = [r["original"] for r in results if r.get("error") == "Missing result from batch response"]
    if not missing:
        return results

    for _ in range(RETRY_PASSES):
        still_missing = []
        replacements = {}

        for i in range(0, len(missing), RETRY_BATCH_SIZE):
            sub = missing[i:i+RETRY_BATCH_SIZE]
            sub_res = _single_pass(sub, top_k_for_prompt=RETRY_TOP_K_FOR_PROMPT)
            for sr in sub_res:
                if sr.get("error") == "Missing result from batch response":
                    still_missing.append(sr["original"])
                else:
                    replacements[sr["original"]] = sr

        results = [replacements.get(r["original"], r) for r in results]
        missing = still_missing
        if not missing:
            break

    return results


# -----------------------------
# PROCESS TERMS
# -----------------------------
terms_to_process = [t for t in unique_terms if norm_term(t) not in processed_terms]
print(f"Unique terms needing LLM after resume: {len(terms_to_process)}")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

batches = list(chunked(terms_to_process, BATCH_SIZE))
print(f"Batches to process: {len(batches)} | batch_size={BATCH_SIZE} | max_workers={MAX_WORKERS} | top_k={TOP_K} | top_k_for_prompt={TOP_K_FOR_PROMPT}")

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
                writer.writerow(
                    {
                        "original": term_norm,
                        "status": result.get("status", ""),
                        "updated_term": result.get("updated_term", ""),
                        "extraneous_text": result.get("extraneous_text", ""),
                        "confidence": float(result.get("confidence", 0.0) or 0.0),
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
