#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
OLD_FILE = "RBMSgenre_forHenry All Columns-Sample-700.csv"

# Intermediate results file (per-term mappings)
RESULTS_FILE = "genre_llm_results.csv"

# Final output file (same shape as OLD_FILE + updated columns)
FINAL_OUTPUT = "RBMSgenre_with_updated_terms.csv"

# Name of the RBMS term column in OLD_FILE
RBMS_COL = "655 - Local Param 04"

# Names of the new columns to add to the final output
UPDATED_COL = "Updated RBMS Genre Term"
EXTRANEOUS_COL = "RBMS Extraneous Text"

# RBMS terms JSON for lightweight candidate retrieval
RBMS_TERMS_FILE = "rbms_terms.json"

# Concurrency
MAX_WORKERS = 8

# Pacing
SLEEP_BETWEEN_CALLS_SECONDS = 0.01
PROGRESS_EVERY_N = 10


# -----------------------------
# HELPER: normalize term
# -----------------------------
def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()


def _tokenize(text):
    text = norm_term(text).lower()
    return [w for w in re.split(r"\W+", text) if w]


def _jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


START_TIME = time.perf_counter()

# -----------------------------
# LOAD RBMS TERMS
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE} for candidate selection.")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)

RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
RBMS_TERMS_LOWER = [t.lower() for t in RBMS_TERMS]


def get_candidate_terms(input_term, max_candidates=25):
    """
    Candidate retrieval WITHOUT difflib:
    - Token overlap (Jaccard) between input and RBMS term
    - Bonus if one string contains the other
    """
    text = norm_term(input_term)
    if not text:
        return []

    text_lower = text.lower()
    in_tokens = set(_tokenize(text_lower))

    scored = []
    for rb_term, rb_lower in zip(RBMS_TERMS, RBMS_TERMS_LOWER):
        rb_tokens = set(_tokenize(rb_lower))

        j = _jaccard(in_tokens, rb_tokens)

        sub = 0.0
        if rb_lower and (rb_lower in text_lower or text_lower in rb_lower):
            sub = 1.0

        score = (0.8 * j) + (0.2 * sub)
        scored.append((score, rb_term))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = [term for score, term in scored[:max_candidates] if score >= 0.15]

    seen = set()
    candidates = []
    for t in top:
        if t not in seen:
            candidates.append(t)
            seen.add(t)

    return candidates


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
        # Normalize missing keys to "" for safety
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


# -----------------------------
# OLLAMA CALL  (UNCHANGED WORKFLOW)
# -----------------------------
def post_hpc(prompt):
    return chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )


def _parse_first_json_object(text: str):
    """
    Extract and parse the FIRST JSON object from a string.
    Ignores any trailing text after the first complete object.
    """
    if not text:
        raise ValueError("Empty response text")

    # If the model wrapped JSON in ```json fences, strip them
    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    start = s.find("{")
    if start == -1:
        raise ValueError(f"No '{{' found in response: {s[:200]}")

    decoder = json.JSONDecoder()
    obj, _idx = decoder.raw_decode(s[start:])  # parses first object only
    if not isinstance(obj, dict):
        raise ValueError("First JSON value was not an object")
    return obj


def _extract_text_from_ollama_response(raw):
    """
    Make this tolerant:
    - Some setups return dicts (typical: {"message":{"content":"..."}})
    - Others might return a JSON string
    """
    if isinstance(raw, dict):
        msg = raw.get("message")
        if isinstance(msg, dict):
            return norm_term(msg.get("content", ""))
        # fallback if custom server shape
        return norm_term(raw.get("response", ""))

    # If it's a string, try parse as JSON, else return as-is
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


def ask_llm(term):
    term_norm = norm_term(term)

    candidates = get_candidate_terms(term_norm, max_candidates=25)
    candidates_block = "\n".join(f"- {c}" for c in candidates) if candidates else "(none)"

    prompt = f"""
You are an expert rare-books cataloger specializing in RBMS Controlled Vocabularies.

You will be given:
- A noisy input string representing a legacy genre heading.
- A SHORTLIST of RBMS candidate terms that were selected by a string-matching algorithm.

YOUR JOB:
1. Choose the single best RBMS term from the SHORTLIST, if any is appropriate.
2. Everything in the input that is NOT part of that RBMS term MUST go into "extraneous_text".
3. Under NO circumstances may you output any term as "updated_term" that is not in the SHORTLIST.

IMPORTANT:
- Never include codes like "aat", "lcgft", "rbpap", or URLs in "updated_term".
- Those MUST ALWAYS go to "extraneous_text".
- If NO candidate fits with high confidence, use:
    "status": "REVIEW"
    "updated_term": ""
    "confidence": 0.0

INPUT STRING:
"{term_norm}"

RBMS CANDIDATE SHORTLIST:
{candidates_block}

OUTPUT FORMAT (strict JSON, one object only):

{{
  "original": "{term_norm}",
  "status": "<CHANGED|PROPOSED|DELETED|REVIEW|ERROR>",
  "updated_term": "<one of the candidate terms above, or empty>",
  "extraneous_text": "<everything not part of the RBMS term, or empty>",
  "confidence": 0.0,
  "error": ""
}}

RULES:
- If the input clearly matches one candidate term inside the string, set:
    "status": "PROPOSED"
    "updated_term": "<that candidate exactly>"
    "extraneous_text": "<everything else, such as 'aat', 'lcgft', dates, places, URLs>"
- If the model's internal RBMS changed/deleted knowledge indicates
  CHANGED or DELETED, you may override "status" accordingly,
  but "updated_term" must still be an RBMS term from the SHORTLIST
  or empty.
- No extra text before or after the JSON.
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

        # Get the model text (ollama.chat returns a dict with message.content)
        txt = _extract_text_from_ollama_response(response["message"]["content"])

        try:
            obj = _parse_first_json_object(txt)
        except Exception as e:
            base_result["error"] = f"JSON parse error: {e} | snippet: {txt[:200]}"
            return base_result

        original = norm_term(obj.get("original", term_norm))
        status = str(obj.get("status", "PROPOSED")).strip().upper() or "PROPOSED"
        updated = norm_term(obj.get("updated_term", ""))
        extraneous = norm_term(obj.get("extraneous_text", ""))
        conf_raw = obj.get("confidence", 0.0)

        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = 0.0

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
# PROCESS TERMS (4-WORKER THREADPOOL, NO TQDM)
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
