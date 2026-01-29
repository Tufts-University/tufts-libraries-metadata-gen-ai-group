#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_genre_hpc.py

HPC-oriented version:
- Batches N terms per Ollama chat request (default 25).
- Uses a real retrieval step (vector similarity over RBMS terms) using Ollama embeddings,
  cached locally to a JSONL file.
- Uses 4 worker threads (configurable) to run multiple batches concurrently.
- Keeps the same Ollama response workflow you already use:
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
import math
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"
os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text"

# -----------------------------
# OLLAMA IMPORTS (tolerant)
# -----------------------------
# Your original script uses: from ollama import chat
# Some installs also expose embeddings() at top-level; support both.
try:
    from ollama import chat  # type: ignore
except Exception as e:
    raise SystemExit(f"Could not import ollama.chat. Ensure the 'ollama' Python package is installed. Error: {e}")

try:
    from ollama import embeddings as ollama_embeddings  # type: ignore
except Exception:
    ollama_embeddings = None


# -----------------------------
# CONFIG
# -----------------------------
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:32b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Usage:
#   python process_genre_hpc.py input.csv
OLD_FILE = sys.argv[1] if len(sys.argv) > 1 else "RBMSgenre_forHenry All Columns-Sample-700.csv"

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

# Cached embeddings for RBMS terms (JSONL: {"term":..., "embedding":[...], "model":...})
EMBED_CACHE_FILE = os.environ.get("RBMS_EMBED_CACHE_FILE", "rbms_terms_embeddings.jsonl")

# Retrieval
TOP_K = int(os.environ.get("RAG_TOP_K", "25"))

# Batching
BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "25"))

# Concurrency (batch-level)
MAX_WORKERS = int(os.environ.get("OLLAMA_MAX_WORKERS", "4"))

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


def _tokenize(text):
    text = norm_term(text).lower()
    return [w for w in re.split(r"\W+", text) if w]


def _parse_first_json_object(text: str):
    """
    Extract and parse the FIRST JSON object from a string.
    Ignores any trailing text after the first complete object.
    (Kept for compatibility, but batching uses _parse_all_json_objects.)
    """
    if not text:
        raise ValueError("Empty response text")

    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    start = s.find("{")
    if start == -1:
        raise ValueError(f"No '{{' found in response: {s[:200]}")

    decoder = json.JSONDecoder()
    obj, _idx = decoder.raw_decode(s[start:])
    if not isinstance(obj, dict):
        raise ValueError("First JSON value was not an object")
    return obj


def _parse_all_json_objects(text: str):
    """
    Parse MANY JSON objects out of one response.
    Supports:
      - JSON array: [ {...}, {...} ]
      - JSON Lines: {...}\\n{...}\\n...
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
    Your existing tolerant extractor:
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


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(x * x for x in a))


def _cosine(a, b):
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# -----------------------------
# OLLAMA CALLS (keep workflow)
# -----------------------------
def post_hpc(prompt: str):
    return chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )


def embed_text(text: str):
    """
    Get embeddings from Ollama.
    Tries ollama.embeddings() if available; otherwise tries ollama.Embeddings-ish shapes via chat won't work.
    """
    if ollama_embeddings is None:
        raise RuntimeError(
            "Ollama embeddings function not available in this Python install. "
            "Install/upgrade the 'ollama' Python package that exposes embeddings(), "
            "or provide your own embedding client."
        )

    resp = ollama_embeddings(model=EMBED_MODEL, prompt=text)
    # Typical response: {"embedding":[...]}
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise ValueError("Unexpected embeddings() response shape (no 'embedding' list).")
    return [float(x) for x in emb]


# -----------------------------
# LOAD RBMS TERMS + BUILD/LOAD VECTOR INDEX (RAG)
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE}")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)

RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
print(f"Loaded {len(RBMS_TERMS)} RBMS terms")


def load_embedding_cache(path: str):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                term = norm_term(obj.get("term"))
                model = norm_term(obj.get("model"))
                emb = obj.get("embedding")
                if term and model == EMBED_MODEL and isinstance(emb, list):
                    cache[term] = [float(x) for x in emb]
            except Exception:
                continue
    return cache


def append_embedding_cache(path: str, term: str, emb):
    rec = {"term": term, "model": EMBED_MODEL, "embedding": emb}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_term_embeddings():
    """
    Ensure we have embeddings for all RBMS_TERMS in EMBED_CACHE_FILE.
    Returns:
      terms: list[str]
      vectors: list[list[float]] aligned with terms
    """
    cache = load_embedding_cache(EMBED_CACHE_FILE)
    missing = [t for t in RBMS_TERMS if t not in cache]

    if missing:
        print(f"Embedding cache '{EMBED_CACHE_FILE}' has {len(cache)} terms; embedding {len(missing)} missing terms with model '{EMBED_MODEL}'")
        # Embed sequentially here (one-time cost). On HPC this is usually fast enough.
        # If you want, you can batch embeddings too, but Ollama embeddings API often expects one prompt.
        for i, term in enumerate(missing, 1):
            emb = embed_text(term)
            cache[term] = emb
            append_embedding_cache(EMBED_CACHE_FILE, term, emb)
            if i % 200 == 0 or i == len(missing):
                print(f"Embedded {i}/{len(missing)} missing terms")
    else:
        print(f"Embedding cache '{EMBED_CACHE_FILE}' already complete for model '{EMBED_MODEL}'")

    terms = RBMS_TERMS[:]  # keep original order
    vectors = [cache[t] for t in terms]
    return terms, vectors


RAG_TERMS, RAG_VECS = ensure_term_embeddings()


def retrieve_top_k(query: str, k: int = TOP_K):
    """
    Real retrieval: embed query, compute cosine similarity against term vectors, return top-k terms.
    """
    q = norm_term(query)
    if not q:
        return []
    qv = embed_text(q)

    # Keep a small heap of (score, term)
    heap = []
    for term, tv in zip(RAG_TERMS, RAG_VECS):
        score = _cosine(qv, tv)
        if len(heap) < k:
            heapq.heappush(heap, (score, term))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, term))

    heap.sort(reverse=True)
    return [t for score, t in heap if score > 0.0]


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
# BATCHED LLM CALL (RAG + batching)
# -----------------------------
def ask_llm_batch(terms_batch):
    """
    Build one prompt for <= BATCH_SIZE terms.
    Uses real retrieval: top-k RBMS terms by embedding similarity.
    Returns list of normalized result dicts (one per input term).
    """

    items = []
    for term in terms_batch:
        term_norm = norm_term(term)
        retrieved = retrieve_top_k(term_norm, k=TOP_K)
        items.append(
            {
                "original": term_norm,
                "retrieved_terms": retrieved,
            }
        )

    # IMPORTANT: require JSON lines for easy parsing
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
        # whole-batch failure: mark each term
        err = f"Batch error: {e}"
        for tn in base_by_term:
            base_by_term[tn]["error"] = err
        return list(base_by_term.values())

    # Index outputs by original
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

            # write results sequentially (main thread)
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

print(f"Final output written to '{FINAL_OUTPUT}'")
print(f"Done. Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

elapsed = time.perf_counter() - START_TIME
