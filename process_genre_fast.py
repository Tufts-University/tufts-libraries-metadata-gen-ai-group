#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
import openpyxl

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"

# This must match the name you used when you ran:
#   ollama create rbmscv -f rbms-crosswalk-3b.modelfile
MODEL_NAME = "rbmscv"

# Input RBMS genre Excel file
OLD_FILE = "RBMSgenre-old_postOR.xlsx"

# Intermediate results file (per-term mappings)
RESULTS_FILE = "genre_llm_results.csv"

# Final output file (same shape as OLD_FILE + updated columns)
FINAL_OUTPUT = "RBMSgenre_with_updated_terms.csv"

# Name of the RBMS term column in OLD_FILE
RBMS_COL = "655 - Local Param 04"

# Names of the new columns to add to the final output
UPDATED_COL = "Updated RBMS Genre Term"
EXTRANEOUS_COL = "RBMS Extraneous Text"

# Max parallel HTTP calls to Ollama
MAX_WORKERS = 4


# -----------------------------
# HELPER: normalize term
# -----------------------------
def norm_term(s):
    if s is None:
        return ""
    return str(s).strip()


# -----------------------------
# LOAD EXCEL, HONOR FILTERED/HIDDEN ROWS
# -----------------------------
print("Loading input Excel and honoring filtered/hidden rows.")

# Load with openpyxl to see row visibility (filters create hidden rows)
wb = openpyxl.load_workbook(OLD_FILE, data_only=True)
ws = wb.active  # adjust if you need a specific sheet

# Build a visibility map: row index (1-based) -> True (visible) / False (hidden)
row_visible = {}
for i in range(1, ws.max_row + 1):
    dim = ws.row_dimensions.get(i)
    hidden = bool(dim.hidden) if dim is not None and dim.hidden is not None else False
    row_visible[i] = not hidden

# Now load with pandas for easier column handling
old_df = pd.read_excel(OLD_FILE, engine="openpyxl").fillna("")

if RBMS_COL not in old_df.columns:
    raise SystemExit(f"Column '{RBMS_COL}' not found in {OLD_FILE}")

print(f"Columns in {OLD_FILE}: {list(old_df.columns)}")

# Normalize the RBMS column
old_df[RBMS_COL] = old_df[RBMS_COL].apply(norm_term)

# Build a mask for "visible" data rows.
# We assume row 1 is the header row in Excel, so DataFrame index 0 corresponds to Excel row 2.
visible_mask = []
for idx in old_df.index:
    excel_row_num = idx + 2  # shift by 1 for zero-based, +1 for header -> 2
    visible_mask.append(row_visible.get(excel_row_num, True))

visible_mask = pd.Series(visible_mask, index=old_df.index)

# Only consider rows that are both visible and have non-empty RBMS_COL
mask_nonempty = old_df[RBMS_COL].astype(str).str.strip() != ""
effective_mask = visible_mask & mask_nonempty

all_terms = old_df.loc[effective_mask, RBMS_COL].tolist()
unique_terms = sorted(set(all_terms))

print(f"Total rows in old file: {len(old_df)}")
print(f"Rows visible & non-empty in {RBMS_COL}: {effective_mask.sum()}")
print(f"Unique visible, non-empty terms in {RBMS_COL}: {len(unique_terms)}")


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
csv_is_new = not file_exists or os.path.getsize(RESULTS_FILE) == 0

results_fh = open(RESULTS_FILE, "a", newline="", encoding="utf-8")
fieldnames = ["original", "status", "updated_term", "extraneous_text", "confidence", "error"]
writer = csv.DictWriter(results_fh, fieldnames=fieldnames)
if csv_is_new:
    writer.writeheader()
    results_fh.flush()


# -----------------------------
# HTTP CALL TO OLLAMA API (ONE TERM)
# -----------------------------
def ask_llm(term):
    """
    Call Ollama with a *small* prompt (one term).
    Returns a dict with keys:
      original, status, updated_term, extraneous_text, confidence, error
    """
    term_norm = norm_term(term)

    prompt = f"""
You are an expert rare-books cataloger specializing in RBMS Controlled Vocabularies.

Your task is to interpret and, when possible, update the following legacy RBMS genre term:

TERM:
"{term}"

RULES (summarized):

1. Authoritative vocabulary only
   - You may ONLY propose updated terms that are valid RBMS Genre/Form terms
     from the embedded RBMS datasets in this model.
   - Do NOT invent or hallucinate new headings.

2. Find the RBMS term *inside* the string
   - Scan the input string and look for the longest phrase that matches
     an authorized RBMS term (case-insensitive).
   - That phrase should be used as "updated_term".
   - Everything else (codes like 'aat', 'lcgft', 'rbpap', dates, places,
     URLs, etc.) must go into "extraneous_text".

   Example:
     Input: "Abstracts. aat"
     updated_term    = "Abstracts"
     extraneous_text = "aat"

     Input: "Ballads -- England -- 1760s"
     updated_term    = "Ballads"
     extraneous_text = "England; 1760s"

3. Changed/deleted terms
   - If this term matches a known changed or deleted term in the embedded
     RBMS data, obey the model's system instructions for CHANGED/DELETED.

4. REVIEW
   - If you cannot confidently map to a single RBMS heading inside the string,
     return:
       status = "REVIEW"
       updated_term = ""
       extraneous_text = the original string or useful notes
       confidence = 0.0

5. Output format (strict JSON)
   Respond ONLY with a single JSON object, no prose, exactly like:

   {{
     "original": "{term_norm}",
     "status": "<CHANGED|PROPOSED|DELETED|REVIEW|ERROR>",
     "updated_term": "<RBMS term or empty>",
     "extraneous_text": "<removed noise or empty>",
     "confidence": 0.0,
     "error": ""
   }}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    base_result = {
        "original": term_norm,
        "status": "ERROR",
        "updated_term": "",
        "extraneous_text": "",
        "confidence": 0.0,
        "error": ""
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        txt = data.get("response", "").strip()

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

    except Exception as e:
        base_result["error"] = str(e)
        return base_result


# -----------------------------
# PROCESS TERMS WITH THREADPOOL
# -----------------------------
terms_to_process = [t for t in unique_terms if norm_term(t) not in processed_terms]
print(f"Unique terms needing LLM after resume: {len(terms_to_process)}")

if terms_to_process:
    print(f"Starting ThreadPool with {MAX_WORKERS} workers.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_term = {}
        for term in terms_to_process:
            future = executor.submit(ask_llm, term)
            future_to_term[future] = term

        for future in tqdm(as_completed(future_to_term),
                           total=len(future_to_term),
                           desc="Processing terms"):
            result = future.result()
            term_norm = norm_term(result.get("original"))

            # Update in-memory
            existing_results[term_norm] = result

            # Write incrementally
            writer.writerow({
                "original": term_norm,
                "status": result.get("status", ""),
                "updated_term": result.get("updated_term", ""),
                "extraneous_text": result.get("extraneous_text", ""),
                "confidence": float(result.get("confidence", 0.0)),
                "error": result.get("error", "")
            })
            results_fh.flush()

            # Short pause to avoid hammering Ollama
            time.sleep(0.01)
else:
    print("No new terms left to process.")

# Close the results file
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
    term_norm = norm_term(row.get("original"))
    term_to_updated[term_norm] = norm_term(row.get("updated_term"))
    term_to_extraneous[term_norm] = norm_term(row.get("extraneous_text"))

# Map over all rows in old_df; we preserve all rows, including filtered ones
updated_terms_for_rows = []
extraneous_for_rows = []

for _, row in old_df.iterrows():
    term_norm = norm_term(row[RBMS_COL])
    updated = term_to_updated.get(term_norm, "")
    extra = term_to_extraneous.get(term_norm, "")
    updated_terms_for_rows.append(updated)
    extraneous_for_rows.append(extra)

old_df[UPDATED_COL] = updated_terms_for_rows
old_df[EXTRANEOUS_COL] = extraneous_for_rows

old_df.to_csv(FINAL_OUTPUT, index=False)
print(f"Final output written to '{FINAL_OUTPUT}'")
print("Done.")
