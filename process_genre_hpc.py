#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import re



import pandas as pd
from ollama import chat

# -----------------------------
# CONFIG
# -----------------------------
# OLLAMA_URL = "https://library-gen-ai-metadata.library.tufts.edu/api/generate"
MODEL_NAME = "qwen3"

# Input RBMS genre Excel file (NO file picker; command line only)
# Usage: python process_genre_fast_single.py input.xlsx
OLD_FILE = "RBMSgenre_forHenry All Columns-Sample-10.xlsx"

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

# Networking / pacing
HTTP_TIMEOUT_SECONDS = 300
SLEEP_BETWEEN_CALLS_SECONDS = 0.01
PROGRESS_EVERY_N = 10


# -----------------------------
# HELPER: normalize term
# -----------------------------
def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()


# -----------------------------
# LOAD RBMS TERMS
# -----------------------------
print(f"Loading RBMS terms from {RBMS_TERMS_FILE} for candidate selection.")
with open(RBMS_TERMS_FILE, "r", encoding="utf-8") as f:
    RBMS_TERMS = json.load(f)

RBMS_TERMS = [norm_term(t) for t in RBMS_TERMS if norm_term(t)]
RBMS_TERMS_LOWER = [t.lower() for t in RBMS_TERMS]


def _tokenize(text):
    text = norm_term(text).lower()
    return [w for w in re.split(r"\W+", text) if w]


def _jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


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

        # Substring bonus (simple, fast)
        sub = 0.0
        if rb_lower and (rb_lower in text_lower or text_lower in rb_lower):
            sub = 1.0

        score = (0.8 * j) + (0.2 * sub)
        scored.append((score, rb_term))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Drop very low scores to keep shortlist tight (tweak if needed)
    top = [term for score, term in scored[:max_candidates] if score >= 0.15]

    # Ensure uniqueness, preserve order
    seen = set()
    candidates = []
    for t in top:
        if t not in seen:
            candidates.append(t)
            seen.add(t)

    return candidates


# -----------------------------
# LOAD EXCEL WITH PANDAS
# -----------------------------
print("Loading input Excel with pandas.")
# NOTE: We do NOT honor hidden/filtered rows here because that required openpyxl imports in the original.
old_df = pd.read_excel(OLD_FILE, dtype={RBMS_COL: "str"}).fillna("")

if RBMS_COL not in old_df.columns:
    raise SystemExit(f"Column '{RBMS_COL}' not found in {OLD_FILE}")

print(f"Columns in {OLD_FILE}: {list(old_df.columns)}")

# Normalize the RBMS column
old_df[RBMS_COL] = old_df[RBMS_COL].apply(norm_term)

# Only consider rows with non-empty RBMS_COL
mask_nonempty = old_df[RBMS_COL].astype(str).str.strip() != ""
all_terms = old_df.loc[mask_nonempty, RBMS_COL].tolist()
unique_terms = sorted(set(all_terms))

print(f"Total rows in old file: {len(old_df)}")
print(f"Rows non-empty in {RBMS_COL}: {mask_nonempty.sum()}")
print(f"Unique non-empty terms in {RBMS_COL}: {len(unique_terms)}")


# -----------------------------
# LOAD/INIT RESULTS FILE (RESUME SUPPORT)
# -----------------------------
existing_results = {}
file_exists = os.path.exists(RESULTS_FILE)

if file_exists:
    print(f"Found existing results file '{RESULTS_FILE}', loading for resume.")
    df_existing = pd.read_csv(RESULTS_FILE, dtype=str).fillna("")
    for _, row in df_existing.iterrows():
        key = norm_term(row.get("original"))
        existing_results[key] = {
            "original": key,
            "status": row.get("status", ""),
            "updated_term": row.get("updated_term", ""),
            "extraneous_text": row.get("extraneous_text", ""),
            "confidence": float(row.get("confidence", 0.0)) if row.get("confidence") else 0.0,
            "error": row.get("error", "")
        }
else:
    print("No existing results file. A new one will be created.")

processed_terms = set(existing_results.keys())
print(f"Already have {len(processed_terms)} terms in {RESULTS_FILE}")

# Prepare CSV writer (append mode, write header if new)
csv_is_new = (not file_exists) or (os.path.getsize(RESULTS_FILE) == 0)

results_fh = open(RESULTS_FILE, "a", newline="", encoding="utf-8")
fieldnames = ["original", "status", "updated_term", "extraneous_text", "confidence", "error"]
writer = csv.DictWriter(results_fh, fieldnames=fieldnames)
if csv_is_new:
    writer.writeheader()
    results_fh.flush()


# -----------------------------
# HTTP CALL TO OLLAMA API (ONE TERM) - stdlib urllib
# -----------------------------
# def _http_post_json(url, payload, timeout):
#     data = json.dumps(payload).encode("utf-8")
#     req = urllib.request.Request(
#         url=url,
#         data=data,
#         headers={"Content-Type": "application/json", "Accept": "application/json"},
#         method="POST",
#     )
#     with urllib.request.urlopen(req, timeout=timeout) as resp:
#         body = resp.read().decode("utf-8", errors="replace")
#         return body

def post_hpc(prompt):
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}]
    )

    return response

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

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    base_result = {
        "original": term_norm,
        "status": "ERROR",
        "updated_term": "",
        "extraneous_text": "",
        "confidence": 0.0,
        "error": ""
    }

    try:
        raw = post_hpc(prompt)
        data = json.loads(raw)

        txt = norm_term(data.get("response", ""))

        # Extract JSON from response (in case model adds stray text)
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = txt[start:end + 1]
        else:
            base_result["error"] = f"No JSON object in response: {txt[:200]}"
            return base_result

        try:
            obj = json.loads(json_str)
        except Exception as e:
            base_result["error"] = f"JSON parse error: {e} | snippet: {json_str[:200]}"
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
            "error": ""
        }

    # except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
    #     base_result["error"] = str(e)
    #     return base_result
    except Exception as e:
        base_result["error"] = str(e)
        return base_result


# -----------------------------
# PROCESS TERMS (SINGLE-THREADED)
# -----------------------------
terms_to_process = [t for t in unique_terms if norm_term(t) not in processed_terms]
print(f"Unique terms needing LLM after resume: {len(terms_to_process)}")

if terms_to_process:
    print("Processing terms single-threaded (no workers).")

    for i, term in enumerate(terms_to_process, start=1):
        result = ask_llm(term)
        term_norm = norm_term(result.get("original"))

        existing_results[term_norm] = result

        writer.writerow({
            "original": term_norm,
            "status": result.get("status", ""),
            "updated_term": result.get("updated_term", ""),
            "extraneous_text": result.get("extraneous_text", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "error": result.get("error", "")
        })
        results_fh.flush()

        if (i % PROGRESS_EVERY_N) == 0 or i == len(terms_to_process):
            print(f"Processed {i}/{len(terms_to_process)} terms")

        time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)
else:
    print("No new terms left to process.")

results_fh.close()
print(f"All term-level results stored in '{RESULTS_FILE}'.")


# -----------------------------
# BUILD FINAL OUTPUT
# -----------------------------
print("Building final output DataFrame.")

df_results = pd.read_csv(RESULTS_FILE, dtype=str).fillna("")
term_to_updated = {}
term_to_extraneous = {}

for _, row in df_results.iterrows():
    t = norm_term(row.get("original"))
    term_to_updated[t] = norm_term(row.get("updated_term"))
    term_to_extraneous[t] = norm_term(row.get("extraneous_text"))

updated_terms_for_rows = []
extraneous_for_rows = []

for _, row in old_df.iterrows():
    term_norm = norm_term(row[RBMS_COL])
    updated_terms_for_rows.append(term_to_updated.get(term_norm, ""))
    extraneous_for_rows.append(term_to_extraneous.get(term_norm, ""))

old_df[UPDATED_COL] = updated_terms_for_rows
old_df[EXTRANEOUS_COL] = extraneous_for_rows

old_df.to_csv(FINAL_OUTPUT, index=False)
print(f"Final output written to '{FINAL_OUTPUT}'")
print("Done.")
