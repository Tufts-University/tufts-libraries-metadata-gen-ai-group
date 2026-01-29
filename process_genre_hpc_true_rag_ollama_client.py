#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_genre_hpc.py (true RAG, Ollama-python client style)

- Batches N terms per Ollama chat request (default 25).
- Uses TRUE vector retrieval ("RAG") WITHOUT pulling new models:
    * Embeddings are fetched via the installed `ollama` Python package (embeddings()).
    * The ollama package handles host/transport (e.g., via environment / your HPC wrapper).
- Builds/loads a local vector cache for RBMS terms:
    * rbms_vectors.npy (float32, L2-normalized, shape [num_terms, dim])
    * rbms_texts.json (list[str], aligned with vectors)
- Uses 4 worker threads (configurable) to run multiple batches concurrently.
- Keeps the SAME Ollama chat response workflow:
    response = post_hpc(prompt)
    txt = _extract_text_from_ollama_response(response["message"]["content"])
    ...parse JSON...
- No tqdm.
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
    # Preferred: use the ollama python client's embeddings() (host handled by the install)
    from ollama import embeddings as ollama_embeddings  # type: ignore
except Exception:
    ollama_embeddings = None


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:32b")

# Embedding model used by ollama.embeddings(). In your environment, embeddings worked with qwen3:32b.
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", MODEL_NAME)

# Usage:
#   python process_genre_hpc.py input.csv
OLD_FILE = "RBMSgenre_forHenry All Columns-Sample-700.csv"

# Intermediate results file (per-term mappings)
RESULTS_FILE = os.environ.get("GENRE_RESULTS_FILE", "genre_llm_results.csv")

# Final output file (same shape as OLD_FILE + updated columns)
FINAL_OUTPUT = os.environ.get("GENRE_FINAL_OUTPUT", "RBMSgenre_with_updated_terms.csv")

# Name of the RBMS term column in OLD_FILE
RBMS_COL = os.environ.get("GENRE_RBMS_COL", "655 - Local Param 04")

# Names of the new columns to add to the final output
UPDATED_COL = os.environ.get("GENRE_UPDATED_COL", "Updated RBMS Genre Term")
EXTRANEOUS_COL = os.environ.get("GENRE_EXTRANEOUS_COL", "RBMS Extraneous Text")

# RBMS terms JSON (list[str])
RBMS_TERMS_FILE = os.environ.get("RBMS_TERMS_FILE", "rbms_terms.json")

# Vector cache (local files you can write)
VEC_CACHE_NPY = os.environ.get("RBMS_VEC_CACHE_NPY", "rbms_vectors.npy")
TXT_CACHE_JSON = os.environ.get("RBMS_TXT_CACHE_JSON", "rbms_texts.json")

# Retrieval
TOP_K = int(os.environ.get("RAG_TOP_K", "25"))

# Batching
BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "1"))

# Concurrency (batch-level)
MAX_WORKERS = int(os.environ.get("OLLAMA_MAX_WORKERS", "10"))

# Optional pacing (tiny sleep after each completed batch write)
SLEEP_BETWEEN_BATCHES_SECONDS = float(os.environ.get("SLEEP_BETWEEN_BATCHES_SECONDS", "0.0"))

# Progress printing
PROGRESS_EVERY_BATCHES = int(os.environ.get("PROGRESS_EVERY_BATCHES", "1"))

START_TIME = time.perf_counter()


# -----------------------------
# HELPERS
# -----------------------------
def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()


def clean_for_retrieval(s: str) -> str:
    s = norm_term(s)
    # remove trailing codes like "rbprov MMeT", "rbmscv", etc.
    s = re.sub(r"\s+\b(rb\w+|aat|lcgft)\b.*$", "", s, flags=re.IGNORECASE).strip()
    # remove parenthetical qualifiers
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    # cut after first period (often where geographic/date stuff starts)
    s = s.split(".", 1)[0].strip()
    return s

def _parse_all_json_objects(text: str):
    """
    Parse MANY JSON objects out of one response.
    Supports:
      - JSON array: [ {...}, {...} ]
      - JSON Lines: {...}\n{...}\n...
      - Extra prose (we scan for objects starting at '{')
    """
    if not text:
        raise ValueError("Empty response text")

    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    # JSON array
    if s.startswith("["):
        arr = json.loads(s)
        if not isinstance(arr, list):
            raise ValueError("Expected a JSON array")
        return [x for x in arr if isinstance(x, dict)]

    # Scan for multiple objects
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
    """
    Tolerant extractor:
    - Some setups return dicts (typical: {"message":{"content":"..."}})
    - Others might return a JSON string
    """
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


def post_hpc(prompt):
    return chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )


# -----------------------------
# TRUE RAG: embeddings via ollama python client + numpy cosine retrieval
# -----------------------------
def ollama_embed(text: str) -> np.ndarray:
    """
    Fetch embedding using the installed `ollama` Python package.

    IMPORTANT:
    - We do NOT hardcode any host URL here.
    - The ollama package (and your HPC wrapper) handles the transport/host.
    """
    if ollama_embeddings is None:
        raise RuntimeError(
            "The installed 'ollama' Python package in this environment does not expose embeddings(). "
            "If your environment supports /api/embeddings, upgrade the ollama python package, or switch back to HTTP embeddings."
        )

    resp = ollama_embeddings(model=EMBED_MODEL, prompt=text)
    emb = resp.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError(f"Unexpected embeddings() response (no embedding). Keys: {list(resp.keys())}")
    return np.asarray(emb, dtype=np.float32)


def _l2_normalize_rows(V: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V / norms


def load_or_build_rbms_vector_index(rbms_terms):
    """
    Create/load vector index:
      - TXT_CACHE_JSON: list[str] (aligned with vectors)
      - VEC_CACHE_NPY: float32 matrix, L2-normalized
    """
    if os.path.exists(VEC_CACHE_NPY) and os.path.exists(TXT_CACHE_JSON):
        try:
            V = np.load(VEC_CACHE_NPY)
            with open(TXT_CACHE_JSON, "r", encoding="utf-8") as f:
                T = json.load(f)
            if isinstance(T, list) and len(T) == V.shape[0]:
                return T, V
        except Exception:
            pass  # rebuild

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


def retrieve_top_k(query: str, texts, V: np.ndarray, k: int = TOP_K):
    """
    Cosine similarity retrieval using normalized vectors:
      score = V @ q
    """
    q = norm_term(query)
    if not q:
        return []
    qv = ollama_embed(q)
    qn = qv / (np.linalg.norm(qv) or 1.0)

    scores = V @ qn
    k_eff = min(k, scores.shape[0])
    if k_eff <= 0:
        return []

    idx = np.argpartition(scores, -k_eff)[-k_eff:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [texts[i] for i in idx]


# -----------------------------
# LOAD RBMS TERMS + BUILD/LOAD INDEX
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE}")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)
RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
print(f"Loaded {len(RBMS_TERMS)} RBMS terms")

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
# BATCHED LLM CALL (TRUE RAG + batching)
# -----------------------------
def ask_llm_batch(terms_batch):
    """
    Build one prompt for <= BATCH_SIZE terms.
    Uses TRUE retrieval: top-k RBMS terms by vector similarity.
    Returns list of normalized result dicts (one per input term).
    """
    items = []
    for term in terms_batch:
        term_norm = norm_term(term)
        q = clean_for_retrieval(term_norm)
        retrieved = retrieve_top_k(q, RAG_TERMS, RAG_VECS, k=TOP_K)

        # lexical injection: guarantee obvious RBMS hits show up
        # (use the cleaned query as the headword)
        if q in RBMS_TERMS:
            retrieved = [q] + [t for t in retrieved if t != q]
        items.append({"original": term_norm, "retrieved_terms": retrieved})

    prompt = f"""
You are an expert rare-books cataloger specializing in RBMS Controlled Vocabularies.

For EACH item below:
- Choose the single best RBMS term from retrieved_terms (or empty if none fits).
- updated_term MUST be EXACTLY one of retrieved_terms or "".
- Put everything else from the input string into extraneous_text (codes like rbmscv/rbpap/aat/lcgft, places, dates, etc).
- If none fits, set status="REVIEW", updated_term="", confidence=0.0.

RETURN FORMAT:
JSON LINES ONLY (one JSON object per line). No prose. No markdown fences.

Each JSON object must contain:
original, status, updated_term, extraneous_text, confidence, error

ITEMS (JSON):
{json.dumps(items, ensure_ascii=False)}

Remember:
- updated_term must be a verbatim value from retrieved_terms (or "").
""".strip()

    base_by_term = {}
    for t in terms_batch:
        tn = norm_term(t)
        base_by_term[tn] = {
            "original": tn,
            "status": "ERROR",
            "updated_term": "",
            "extraneous_text": "",
            "confidence": 0.0,
            "error": "",
        }

    try:
        response = post_hpc(prompt)
        txt = _extract_text_from_ollama_response(response["message"]["content"])
        objs = _parse_all_json_objects(txt)
    except Exception as e:
        err = f"Batch error: {e}"
        for tn in base_by_term:
            base_by_term[tn]["error"] = err
        return list(base_by_term.values())

    out_map = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        orig = norm_term(o.get("original"))
        if not orig:
            continue
        out_map[orig] = o

    normalized = []
    for t in terms_batch:
        tn = norm_term(t)
        raw = out_map.get(tn)

        if not raw:
            r = base_by_term[tn]
            r["error"] = "Missing result from batch response"
            normalized.append(r)
            continue

        try:
            conf = float(raw.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        normalized.append(
            {
                "original": tn,
                "status": str(raw.get("status", "PROPOSED")).strip().upper() or "PROPOSED",
                "updated_term": norm_term(raw.get("updated_term", "")),
                "extraneous_text": norm_term(raw.get("extraneous_text", "")),
                "confidence": conf,
                "error": norm_term(raw.get("error", "")),
            }
        )

    return normalized


# -----------------------------
# PROCESS TERMS (BATCHED + 4 WORKERS)
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
                    {
                        "original": norm_term(t),
                        "status": "ERROR",
                        "updated_term": "",
                        "extraneous_text": "",
                        "confidence": 0.0,
                        "error": str(e),
                    }
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
# BUILD FINAL OUTPUT (CSV)
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
