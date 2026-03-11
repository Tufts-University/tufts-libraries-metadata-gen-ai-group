#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from difflib import get_close_matches

import openai

# Configuration
MODEL_NAME = "gpt-5"         # per user's request; change if your endpoint uses a different name
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0         # exponential backoff base seconds
CLOSE_MATCHES = 6           # number of fuzzy candidates to present to the LLM
TEMPERATURE = 0.0           # deterministic

def load_rbms_terms(path):
    with open(path, "r", encoding="utf-8") as fh:
        terms = json.load(fh)
    # keep original order but also prepare a lowered map for fast checks
    terms_set = set(terms)
    terms_lower = {t.lower(): t for t in terms}
    # for longest-first matching
    terms_sorted_by_len = sorted(terms, key=lambda s: len(s), reverse=True)
    return terms, terms_set, terms_lower, terms_sorted_by_len

def extract_field_value(line):
    # Expect "655 - Local Param 04: <value>" or bare value
    parts = line.split(":", 1)
    if len(parts) == 2 and "655" in parts[0]:
        return parts[1].strip()
    return line.strip()

def find_prefix_matches(value, rbms_terms_sorted):
    vlow = value.lower().lstrip()
    matches = []
    for term in rbms_terms_sorted:
        if vlow.startswith(term.lower()):
            matches.append(term)
    return matches

def fuzzy_candidates(value, rbms_terms):
    # Use difflib to get close textual matches
    candidates = get_close_matches(value, rbms_terms, n=CLOSE_MATCHES, cutoff=0.5)
    return candidates

def call_openai_chat(messages, model, max_retries=MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=800,
            )
            return resp
        except Exception as e:
            if attempt == max_retries:
                raise
            backoff = RETRY_BACKOFF ** attempt
            time.sleep(backoff)
    raise RuntimeError("OpenAI call failed after retries")

def extract_first_json(text):
    # Attempt to find first {...} block and parse it as JSON
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i+1])
                    return obj
                except json.JSONDecodeError:
                    return None
    return None

def prepare_prompt(original_value, candidates, rbms_terms):
    # System message enforces constraints
    sys_msg = (
        "You are a strict cataloging assistant. The authoritative list of allowed RBMS preferred "
        "terms is provided. For the given 'original' string, choose EXACTLY one updated_term from "
        "the provided candidate list OR return an empty string for updated_term (indicating REVIEW). "
        "Constraints (must be followed):\n"
        "- updated_term MUST be an EXACT verbatim string from the provided candidate list and from "
        "the RBMS list; do NOT invent new terms or modify terms (no punctuation changes, no pluralization changes).\n"
        "- Do NOT invert word order; do NOT return terms in the form 'X, Y'.\n"
        "- Codes like rbmscv/rbgenr/rbbin/aat/lcgft/fast and dates/places MUST NOT appear in updated_term.\n"
        "- If the input begins with an RBMS term and then has extra text (parentheses, codes, places, dates), "
        "select that RBMS term as updated_term and put the remainder (exactly as it appears) into extraneous_text.\n"
        "- If you cannot supply an exact RBMS term confidently, set updated_term to an empty string.\n"
        "Output requirement: Respond with a single JSON object only, with keys exactly:\n"
        '  \"original\", \"status\", \"updated_term\", \"extraneous_text\", \"confidence\", \"error\"\n'
        '- status must be \"PROPOSED\" when you choose a term, otherwise \"REVIEW\".\n'
        '- confidence must be a decimal between 0.0 and 1.0 (use 1.0 for confident matches, 0.0 for REVIEW).\n'
        '- error may be an empty string on success or a short explanation if needed.\n'
        "Do not include any extra text outside the JSON object.\n"
    )

    cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) if candidates else "(no candidates)"
    user_msg = (
        f"Original: {original_value}\n\n"
        f"Candidate RBMS terms (choose only from these exact strings, or return an empty updated_term):\n"
        f"{cand_text}\n\n"
        "Instruction: If the beginning of the Original matches one of the candidates exactly (case-insensitive), "
        "prefer that candidate and put any following characters in extraneous_text. If none of the candidates fits "
        "confidently, return updated_term as an empty string. Respect all constraints above.\n"
        "Return the JSON object now."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

def main():
    parser = argparse.ArgumentParser(description="Map CSV 655 Local Param 04 terms to RBMS preferred terms using GPT-5.")
    parser.add_argument("input", help="Input file (one line per record; lines containing '655 - Local Param 04: ...' are supported)")
    parser.add_argument("output", help="Output CSV file (original,status,updated_term,extraneous_text,confidence,error)")
    parser.add_argument("--rbms", default="rbms_terms.json", help="RBMS terms JSON file (default: rbms_terms.json)")
    parser.add_argument("--model", default=MODEL_NAME, help="LLM model name to call (default: gpt-5)")
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: Set OPENAI_API_KEY in your environment before running.")
        return
    openai.api_key = openai_api_key

    rbms_terms, rbms_set, rbms_lower_map, rbms_sorted = load_rbms_terms(args.rbms)

    # Read input lines
    with open(args.input, "r", encoding="utf-8", errors="ignore") as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    results = []
    total = len(raw_lines)
    for idx, line in enumerate(raw_lines, start=1):
        original_value = extract_field_value(line)
        # Build candidate list: prefix matches first, then fuzzy matches
        candidates = []
        prefix_matches = find_prefix_matches(original_value, rbms_sorted)
        seen = set()
        for t in prefix_matches:
            if t not in seen:
                candidates.append(t); seen.add(t)
        if len(candidates) < CLOSE_MATCHES:
            fuzzy = fuzzy_candidates(original_value, rbms_terms)
            for t in fuzzy:
                if t not in seen:
                    candidates.append(t); seen.add(t)
        if not candidates:
            candidates = rbms_terms[:min(5, len(rbms_terms))]

        messages = prepare_prompt(original_value, candidates, rbms_terms)
        # Call LLM
        try:
            resp = call_openai_chat(messages, model=args.model)
            content = resp["choices"][0]["message"]["content"]
            parsed = extract_first_json(content)
            if parsed is None:
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
            if parsed and isinstance(parsed, dict):
                updated_term = parsed.get("updated_term", "") or ""
                if updated_term and updated_term not in rbms_set:
                    parsed["status"] = "REVIEW"
                    parsed["updated_term"] = ""
                    parsed["confidence"] = 0.0
                    parsed["error"] = "LLM suggested term not in rbms_terms.json; rejected."
                else:
                    if updated_term:
                        m = re.match(r'\s*' + re.escape(updated_term), original_value, flags=re.IGNORECASE)
                        if m:
                            remainder = original_value[m.end():].strip()
                            parsed["extraneous_text"] = remainder
                        else:
                            parsed["extraneous_text"] = str(parsed.get("extraneous_text", "")).strip()
                        parsed["status"] = "PROPOSED"
                        parsed["confidence"] = float(parsed.get("confidence", 1.0))
                        parsed["error"] = parsed.get("error", "") or ""
                    else:
                        parsed["status"] = "REVIEW"
                        parsed["confidence"] = 0.0
                        parsed["error"] = parsed.get("error", "") or ""
                out_row = {
                    "original": original_value,
                    "status": parsed.get("status", "REVIEW"),
                    "updated_term": parsed.get("updated_term", ""),
                    "extraneous_text": parsed.get("extraneous_text", ""),
                    "confidence": parsed.get("confidence", 0.0),
                    "error": parsed.get("error", "")
                }
            else:
                out_row = {
                    "original": original_value,
                    "status": "REVIEW",
                    "updated_term": "",
                    "extraneous_text": "",
                    "confidence": 0.0,
                    "error": "Could not parse LLM response as JSON"
                }
        except Exception as e:
            out_row = {
                "original": original_value,
                "status": "REVIEW",
                "updated_term": "",
                "extraneous_text": "",
                "confidence": 0.0,
                "error": f"OpenAI error: {e}"
            }

        results.append(out_row)

        if idx % 50 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as outfh:
        writer = csv.DictWriter(outfh, fieldnames=["original", "status", "updated_term", "extraneous_text", "confidence", "error"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Wrote {len(results)} rows to {args.output}")

if __name__ == "__main__":
    main()