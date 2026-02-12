#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_genre_hpc.py

What this does
--------------
- Reads a CSV with a legacy RBMS genre heading column.
- Calls an Ollama model (ollama.chat) to propose an RBMS controlled-vocabulary term.
- DOES NOT do upfront candidate shortlist selection.
- Enforces: updated_term must be an EXACT RBMS preferred term found in rbms_terms.json
  (with normalization/canonicalization and a couple safe transforms).
- Handles common model output issues:
  - "X, Y" inversion -> "Y X" (e.g., "Dissertations, Academic" -> "Academic dissertations")
  - Trailing parenthetical qualifier -> strip if base term exists, push qualifier to extraneous
    (e.g., "Accordion fold format (Binding)" -> updated="Accordion fold format", extraneous+="(Binding)")
- Adds robust Ollama retry/backoff for:
  - Exceptions (500s, timeouts, connection resets)
  - EMPTY responses (content missing/blank)
- Resume support using RESULTS_FILE keyed by original term.

Outputs
-------
- genre_llm_results.csv : term-level results
- RBMSgenre_with_updated_terms.csv : original rows + Updated/Extraneous columns

Usage
-----
python process_genre_hpc.py input.csv
(If no arg given, uses OLD_FILE constant.)

Notes
-----
- For large models like qwen3:32b, concurrency is a frequent cause of empty responses/500s.
  Start with MAX_WORKERS=1 or 2.
"""

import os
import csv
import json
import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ollama import chat

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "qwen3:32b"

# Usage: python process_genre_hpc.py input.csv
OLD_FILE = "test_inference.csv"

# Intermediate results file (per-term mappings)
RESULTS_FILE = "genre_llm_results4.csv"

# Final output file (same shape as OLD_FILE + updated columns)
FINAL_OUTPUT = "RBMSgenre_with_updated_terms.csv"

# Name of the RBMS term column in OLD_FILE
RBMS_COL = "655 - Local Param 04"

# Names of the new columns to add to the final output
UPDATED_COL = "Updated RBMS Genre Term"
EXTRANEOUS_COL = "RBMS Extraneous Text"

# RBMS terms JSON (authoritative list)
RBMS_TERMS_FILE = "rbms_terms.json"

# Concurrency (qwen3:32b is heavy; start low)
MAX_WORKERS = 1

# Pacing between completed futures
SLEEP_BETWEEN_CALLS_SECONDS = 0.5

# Logging
PROGRESS_EVERY_N = 10

# Ollama retry behavior
OLLAMA_MAX_RETRIES = 8
OLLAMA_BACKOFF_START_SECONDS = 1.0
OLLAMA_BACKOFF_MAX_SECONDS = 30.0

# Enable to print raw empty responses for debugging
DEBUG_PRINT_EMPTY_RESPONSES = False


# -----------------------------
# HELPER: normalize term
# -----------------------------
def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()


START_TIME = time.perf_counter()

# -----------------------------
# CLEANING / NORMALIZATION
# -----------------------------
_CODE_TAIL_RE = re.compile(
    r"(?:\s+|\.|,)*(rbmscv|rbgenr|rbpap|rbprov|aat|lcgft|fast|rbbin)\b.*$",
    re.IGNORECASE,
)

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

_INVERTED_RE = re.compile(r"^\s*([^,]+),\s*(.+?)\s*$")
_PAREN_TRAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def clean_for_retrieval(s: str) -> str:
    """
    Reduce noise:
    - Strip trailing code tails like ". rbgenr"
    - Remove trailing punctuation
    - Collapse whitespace
    - Normalize a small UK->US spelling set
    """
    t = norm_term(s)
    if not t:
        return ""

    # Strip trailing code tails like ". rbgenr"
    t = _CODE_TAIL_RE.sub("", t).strip()

    # Remove trailing punctuation
    t = t.rstrip(" .;,")

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # UK->US spelling normalization (whole-word, case-insensitive)
    def _swap_word(m):
        w = m.group(0)
        rep = _UK_US.get(w.lower())
        return rep if rep is not None else w

    pattern = r"\b(" + "|".join(re.escape(k) for k in _UK_US.keys()) + r")\b"
    t = re.sub(pattern, _swap_word, t, flags=re.IGNORECASE)

    return t

def normalize_key(s: str) -> str:
    """
    Canonical key for matching model output to RBMS terms:
    - Clean code tails, punctuation, whitespace
    - Lowercase
    """
    t = clean_for_retrieval(s).lower()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonicalize_rbms_term(s: str, rbms_exact_map: dict) -> str:
    """Return the exact RBMS term if s matches one after normalization; else ''."""
    return rbms_exact_map.get(normalize_key(s), "")

def try_deinvert(term: str) -> str:
    """
    Convert "X, Y" -> "Y X"
    e.g. "Dissertations, Academic" -> "Academic Dissertations"
    """
    t = norm_term(term)
    m = _INVERTED_RE.match(t)
    if not m:
        return t
    a, b = m.group(1).strip(), m.group(2).strip()
    return f"{b} {a}".strip()

def strip_trailing_paren(term: str):
    """
    If term ends with ' (...)', return (base, paren_suffix) else (term, '').
    """
    t = norm_term(term)
    if not _PAREN_TRAIL_RE.search(t):
        return t, ""
    base = _PAREN_TRAIL_RE.sub("", t).strip()
    suffix = t[len(base):].strip()
    return base, suffix

def token_set_for_guard(s: str) -> set:
    """
    Tokenize a string for a mild semantic-leap guard.
    Uses normalize_key so codes/punct drop out.
    """
    t = normalize_key(s)
    return set(_WORD_RE.findall(t.lower()))

def overlap_guard(input_term: str, proposed_term: str) -> bool:
    """
    Mild guard against wild semantic leaps.
    Returns True if proposal seems plausibly related.

    Default rule:
    - If both sides have tokens:
      - Require at least ONE overlapping token.
    This still allows synonymy where at least one anchor word overlaps.
    If you want stricter behavior, change this function.
    """
    inp = token_set_for_guard(input_term)
    out = token_set_for_guard(proposed_term)
    if not inp or not out:
        return True
    return bool(inp & out)


# -----------------------------
# LOAD RBMS TERMS
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE}.")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)

RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
RBMS_TERMS_SET = set(RBMS_TERMS)
RBMS_EXACT_MAP = {normalize_key(t): t for t in RBMS_TERMS}

print(f"Loaded RBMS terms: {len(RBMS_TERMS)}")




# -----------------------------
# LOAD INPUT CSV (NO PANDAS)
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

print(f"Columns in {OLD_FILE}: {headers}")
print(f"Total rows in old file: {len(old_rows)}")
print(f"Rows non-empty in {RBMS_COL}: {sum(1 for t in all_terms if t)}")
print(f"Unique non-empty terms in {RBMS_COL}: {len(unique_terms)}")


# -----------------------------
# LOAD/INIT RESULTS FILE (RESUME SUPPORT)
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


def _extract_text_from_ollama_response(raw):
    """
    Defensive extraction of model text across:
    - dict responses
    - pydantic-style objects with attributes (raw.message.content)
    - objects that support model_dump()/dict()/json()
    - plain strings
    """
    if raw is None:
        return ""

    # Plain string
    if isinstance(raw, str):
        return raw.strip()

    # Pydantic / object response: try attribute access first
    try:
        msg = getattr(raw, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content is not None:
                return str(content).strip()
    except Exception:
        pass

    # If it can convert itself to dict-like, do that
    for meth in ("model_dump", "dict"):
        try:
            fn = getattr(raw, meth, None)
            if callable(fn):
                raw = fn()
                break
        except Exception:
            pass

    # Now handle dict shapes
    if isinstance(raw, dict):
        msg = raw.get("message")
        if isinstance(msg, dict):
            content = msg.get("content", "")
            return ("" if content is None else str(content)).strip()

        # Some proxies
        resp = raw.get("response")
        if resp is not None:
            return str(resp).strip()

        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            m = choices[0].get("message", {})
            if isinstance(m, dict):
                return str(m.get("content", "")).strip()

        return ""

    # Last resort: string repr (not ideal, but better than empty)
    try:
        s = str(raw).strip()
        return s
    except Exception:
        return ""


def post_hpc(prompt: str):
    """
    Single-call Ollama invocation. No backoff, no retries.
    BUT correctly handles object responses and only errors if content truly missing.
    """
    resp = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    txt = _extract_text_from_ollama_response(resp)

    # IMPORTANT: if str(resp) fallback was used, it might include lots of metadata.
    # We still want to fail if we cannot find a JSON object start.
    if not txt or "{" not in txt:
        raise RuntimeError(f"Ollama response did not contain usable JSON text. Extracted snippet: {txt[:300]}")

    return resp

def _parse_first_json_object(text: str):
    """
    Extract and parse the FIRST JSON object from a string.
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
    obj, _ = decoder.raw_decode(s[start:])
    if not isinstance(obj, dict):
        raise ValueError("First JSON value was not an object")

    return obj


# -----------------------------
# LLM ASK / CANONICALIZATION
# -----------------------------
def ask_llm(term: str):
    term_norm = norm_term(term)

    prompt = f"""
You are an expert rare-books cataloger specializing in RBMS Controlled Vocabularies.


IMPORTANT CONSTRAINTS:
- before comparing terms, analyze the meaning of the supplied term and see if any of the RBMS terms match conceptually
- "updated_term" MUST be an EXACT RBMS preferred term (verbatim).
- Do NOT invent new terms.
- Do NOT invert word order (do not output strings like "X, Y").
- If the input begins with an RBMS term and then has extra text (parentheses, codes, places, dates),
  you MUST select that RBMS term and put the rest into "extraneous_text".
- Codes like rbmscv/rbgenr/rbbin/aat/lcgft/fast and dates/places MUST NOT appear in updated_term.
- If you cannot supply an exact RBMS term confidently, use REVIEW with updated_term "" and confidence 0.0.

INPUT STRING:
"{term_norm}"

Return ONE strict JSON object only:

{{
  "original": "{term_norm}",
  "status": "PROPOSED" or "REVIEW",
  "updated_term": "",
  "extraneous_text": "",
  "confidence": 0.0,
  "error": ""
}}
""".strip()

    base_result = {
        "original": term_norm,
        "status": "ERROR",
        "updated_term": "",
        "extraneous_text": "",
        "confidence": 0.0,
        "error": "",
    }

    try:
        response = post_hpc(prompt)
        txt = _extract_text_from_ollama_response(response)

        try:
            obj = _parse_first_json_object(txt)
        except Exception as e:
            base_result["error"] = f"JSON parse error: {e} | snippet: {txt[:200]}"
            return base_result

        original = norm_term(obj.get("original", term_norm))
        status = str(obj.get("status", "REVIEW")).strip().upper() or "REVIEW"
        updated_raw = norm_term(obj.get("updated_term", ""))
        extraneous = norm_term(obj.get("extraneous_text", ""))
        conf_raw = obj.get("confidence", 0.0)

        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = 0.0



        # -----------------------------
        # PREP: normalize model updated_term before canonicalization
        # -----------------------------
        updated_work = updated_raw

        

        # Trim whitespace
        updated_work = norm_term(updated_work)

        updated = ""   # ALWAYS initialize

        if updated_work:
            canon = canonicalize_rbms_term(updated_work, RBMS_EXACT_MAP)
            if canon:
                updated = canon
        if updated_work:
            canon = canonicalize_rbms_term(updated_work, RBMS_EXACT_MAP)
            if canon:
                updated_work = canon

        # (2) Trailing parenthetical -> strip + canonicalize base, push suffix to extraneous
        if updated_work and not updated:
            base, paren_suffix = strip_trailing_paren(updated_work)
            if base and paren_suffix:
                canon_base = canonicalize_rbms_term(base, RBMS_EXACT_MAP)
                if canon_base:
                    updated = canon_base
                    extraneous = (paren_suffix + " " + extraneous).strip() if extraneous else paren_suffix

        # If model provided something but we can't map it to RBMS, REVIEW
        if updated_work and not updated:
            return {
                "original": original,
                "status": "REVIEW",
                "updated_term": "",
                "extraneous_text": extraneous,
                "confidence": 0.0,
                "error": f"updated_term not in rbms_terms.json: {updated_raw}",
            }

        # If model says PROPOSED but didn't give a valid RBMS term, force REVIEW
        if status == "PROPOSED" and not updated:
            status = "REVIEW"
            confidence = 0.0

        # Mild semantic-leap guard (tune overlap_guard() if needed)
        if updated and not overlap_guard(term_norm, updated):
            return {
                "original": original,
                "status": "REVIEW",
                "updated_term": "",
                "extraneous_text": extraneous,
                "confidence": 0.0,
                "error": f"Proposed term seems unrelated to input: {updated}",
            }

        return {
            "original": original,
            "status": status,
            "updated_term": updated,
            "extraneous_text": extraneous,
            "confidence": confidence,
            "error": "",
        }

    except Exception as e:
        base_result["error"] = str(e)
        return base_result


# -----------------------------
# PROCESS TERMS
# -----------------------------
terms_to_process = [t for t in unique_terms if norm_term(t) not in processed_terms]
print(f"Unique terms needing LLM after resume: {len(terms_to_process)}")

if terms_to_process:
    print(f"Processing terms with ThreadPoolExecutor (max_workers={MAX_WORKERS}).")
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_term = {executor.submit(ask_llm, term): term for term in terms_to_process}

        for future in as_completed(future_to_term):
            term = future_to_term[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "original": norm_term(term),
                    "status": "ERROR",
                    "updated_term": "",
                    "extraneous_text": "",
                    "confidence": 0.0,
                    "error": str(e),
                }

            term_key = norm_term(result.get("original"))
            existing_results[term_key] = result

            writer.writerow(
                {
                    "original": term_key,
                    "status": result.get("status", ""),
                    "updated_term": result.get("updated_term", ""),
                    "extraneous_text": result.get("extraneous_text", ""),
                    "confidence": float(result.get("confidence", 0.0)),
                    "error": result.get("error", ""),
                }
            )
            results_fh.flush()

            completed += 1
            if (completed % PROGRESS_EVERY_N) == 0 or completed == len(terms_to_process):
                print(f"Processed {completed}/{len(terms_to_process)} terms")

            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)
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
print("Done.")

END_TIME = time.perf_counter()
elapsed = END_TIME - START_TIME
print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
