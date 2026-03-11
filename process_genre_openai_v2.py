#!/usr/bin/env python3
"""
Robust resume script for mapping '655 - Local Param 04' values to RBMS terms.

Key fixes:
- More robust extraction of the field value (supports ":" and "," separators).
- Deterministic prefix match attempted before calling the LLM (reduces API calls & failures).
- When LLM output cannot be parsed as JSON, the raw LLM text is appended to llm_debug.log for inspection.
- Compatible with openai>=1.0.0 client (uses OpenAI().chat.completions.create with max_completion_tokens).
- Writes progress immediately to CSV and periodic XLSX snapshots.

Usage:
  pip install --upgrade "openai>=1.0.0" xlsxwriter
  export OPENAI_API_KEY="sk-..."
  python3 map_with_gpt5_resume.py RBMSgenre_forHenry.csv output.csv output.xlsx
  For quick deterministic test: add --no-llm
"""
import argparse
import csv
import json
import os
import re
import time
import tempfile
from difflib import get_close_matches

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import xlsxwriter
except Exception:
    xlsxwriter = None

DEFAULT_MODEL = "gpt-5"
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
CLOSE_MATCHES = 6
MAX_COMPLETION_TOKENS = 800

CSV_FIELDNAMES = ["original", "status", "updated_term", "extraneous_text", "confidence", "error"]
DEBUG_LOG = "llm_debug.log"

def normalize_input_value(v):
    """Apply small typo and singular→plural corrections and collapse whitespace."""
    if not v:
        return v
    s = v.strip()
    for wrong, right in SIMPLE_CORRECTIONS.items():
        s = re.sub(re.escape(wrong), right, s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)
    return s
def load_rbms_terms(path):
    with open(path, "r", encoding="utf-8") as fh:
        terms = json.load(fh)
    terms_set = set(terms)
    terms_sorted = sorted(terms, key=lambda s: len(s), reverse=True)
    return terms, terms_set, terms_sorted

def extract_field_value(line):
    """
    Try common separators after the label.
    Accept formats like:
      "655 - Local Param 04: Abaca fibers (Paper) rbpap"
      "655 - Local Param 04, Abaca fibers (Paper) rbpap"
      or just "Abaca fibers (Paper) rbpap"
    Returns the trimmed value or the original line if nothing found.
    """
    if not line:
        return ""
    # If the line already looks like just the value, return it
    stripped = line.strip()
    # If it contains '655' label then attempt to split
    if stripped.lower().startswith("655"):
        # try colon first
        parts = stripped.split(":", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        # try comma
        parts = stripped.split(",", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        # try tab
        parts = stripped.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        # nothing after label -> return empty so caller can decide
        return ""
    # otherwise assume the whole line is the value
    return stripped

def find_prefix_matches(value, rbms_terms_sorted):
    vlow = value.lower().lstrip()
    matches = []
    for term in rbms_terms_sorted:
        if vlow.startswith(term.lower()):
            matches.append(term)
    return matches

def fuzzy_candidates(value, rbms_terms):
    return get_close_matches(value, rbms_terms, n=CLOSE_MATCHES, cutoff=0.5)

def call_openai_chat(messages, model, max_retries=MAX_RETRIES):
    if OpenAI is None:
        raise RuntimeError("OpenAI client not installed (pip install openai>=1.0.0)")
    client = OpenAI()
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            return resp
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError("OpenAI call failed after retries")

def get_chat_response_text(resp):
    try:
        if isinstance(resp, dict):
            choices = resp.get("choices")
            if choices and len(choices) > 0:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list) and content:
                        c0 = content[0]
                        if isinstance(c0, str):
                            return c0
                        if isinstance(c0, dict):
                            for k in ("text", "content"):
                                if k in c0 and isinstance(c0[k], str):
                                    return c0[k]
                text = first.get("text")
                if isinstance(text, str):
                    return text
        choices = getattr(resp, "choices", None)
        if choices:
            choice0 = choices[0]
            msg = getattr(choice0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, str):
                        return first
                    if isinstance(first, dict):
                        for k in ("text", "content"):
                            if k in first:
                                return first[k]
                return str(msg)
            text = getattr(choice0, "text", None)
            if text:
                return text
    except Exception:
        pass
    return str(resp)

def extract_first_json(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None

def prepare_prompt(original_value, candidates):
    sys_msg = (
        "You are a strict cataloging assistant. The authoritative list of allowed RBMS preferred "
        "terms is provided. For the given 'original' string, choose EXACTLY one updated_term from "
        "the provided candidate list OR return an empty string for updated_term (indicating REVIEW). "
        "Constraints (must be followed):\n"
        "- updated_term MUST be an EXACT verbatim string from the provided candidate list and from "
        "the RBMS list; do NOT invent new terms or modify terms.\n"
        "- Do NOT invert word order; do NOT return 'X, Y'.\n"
        "- Codes like rbmscv/rbgenr/rbbin/aat/lcgft/fast and dates/places MUST NOT appear in updated_term.\n"
        "- If the input begins with an RBMS term and then has extra text, select that RBMS term as updated_term and put the remainder into extraneous_text.\n"
        "Output a single JSON object only with keys: original, status, updated_term, extraneous_text, confidence, error.\n"
    )
    cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) if candidates else "(no candidates)"
    user_msg = f"Original: {original_value}\n\nCandidates:\n{cand_text}\n\nReturn the JSON object now."
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

def write_xlsx_from_csv(csv_path, xlsx_path):
    if xlsxwriter is None:
        raise RuntimeError("xlsxwriter not installed (pip install xlsxwriter)")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(tmp_fd)
    try:
        workbook = xlsxwriter.Workbook(tmp_path)
        ws = workbook.add_worksheet("results")
        header_fmt = workbook.add_format({"bold": True})
        for c, col in enumerate(CSV_FIELDNAMES):
            ws.write(0, c, col, header_fmt)
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for r_idx, row in enumerate(reader, start=1):
                for c, col in enumerate(CSV_FIELDNAMES):
                    ws.write(r_idx, c, row.get(col, ""))
        workbook.close()
        os.replace(tmp_path, xlsx_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def read_processed_set_from_csv(csv_path):
    processed = set()
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            orig = row.get("original", "").strip()
            if orig:
                processed.add(orig)
    return processed

def append_row_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass

def deterministic_match(orig, rbms_sorted_local):
    olow = orig.strip().lower()
    for term in rbms_sorted_local:
        if olow.startswith(term.lower()):
            m = re.match(r'\s*' + re.escape(term), orig, flags=re.IGNORECASE)
            extr = orig[m.end():].strip() if m else orig[len(term):].strip()
            return {"original": orig, "status": "PROPOSED", "updated_term": term, "extraneous_text": extr, "confidence": 1.0, "error": ""}
    return None

def log_llm_debug(original_value, content, resp):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"--- ORIGINAL: {original_value}\n")
            f.write(content + "\n\n")
            f.write(f"--- RESPONSE: {resp}\n\n")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_csv")
    parser.add_argument("output_xlsx")
    parser.add_argument("--rbms", default="rbms_terms.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    if not args.no_llm and OpenAI is None:
        print("ERROR: openai>=1.0.0 client not installed. Install: pip install --upgrade openai")
        return
    if not args.no_llm and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in environment or use --no-llm")
        return
    if not os.path.exists(args.rbms):
        print(f"ERROR: RBMS terms file not found: {args.rbms}")
        return

    rbms_terms, rbms_set, rbms_sorted = load_rbms_terms(args.rbms)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    processed_set = read_processed_set_from_csv(args.output_csv)
    print(f"Found {len(processed_set)} already-processed rows in {args.output_csv}")

    total = len(raw_lines)
    to_process = []
    for line in raw_lines:
        val = extract_field_value(line)
        if val == "":
            continue
        to_process.append(val)

    print(f"Total input lines: {total}. Pending new rows to process: {len([v for v in to_process if v not in processed_set])}")

    new_rows_count = 0
    snapshot_interval = max(1, args.snapshot_interval)

    for original_value in to_process:
        if original_value in processed_set:
            continue

        # deterministic first (use normalized for matching)
        dm = deterministic_match(original_value, rbms_sorted)
        if dm:
            # compute extraneous relative to original, not the normalized form
            m = re.match(r'\s*' + re.escape(dm["updated_term"]), original_value, flags=re.IGNORECASE)
            extr = original_value[m.end():].strip() if m else dm["extraneous_text"]
            out_row = {"original": original_value, "status": "PROPOSED", "updated_term": dm["updated_term"], "extraneous_text": extr, "confidence": 1.0, "error": ""}
        elif args.no_llm:
            out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": "No deterministic match"}
        else:
            candidates = []
            seen = set()
            for t in find_prefix_matches(original_value, rbms_sorted):
                if t not in seen:
                    candidates.append(t); seen.add(t)
            if len(candidates) < CLOSE_MATCHES:
                for t in fuzzy_candidates(original_value, rbms_terms):
                    if t not in seen:
                        candidates.append(t); seen.add(t)
            if not candidates:
                candidates = rbms_terms[:min(5, len(rbms_terms))]

            messages = prepare_prompt(original_value, candidates)
            try:
                resp = call_openai_chat(messages, model=args.model)
                content = get_chat_response_text(resp)
                if not content or not content.strip():
                    log_llm_debug(original_value, content, resp)
                    out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": "Empty LLM content; raw saved to llm_debug.log"}
                else:
                    parsed = extract_first_json(content)
                    if parsed is None:
                        log_llm_debug(original_value, content, resp)
                        # fallback: try deterministic match on normalized value again
                        dm2 = deterministic_match(original_value, rbms_sorted)
                        if dm2:
                            m = re.match(r'\s*' + re.escape(dm2["updated_term"]), original_value, flags=re.IGNORECASE)
                            extr2 = original_value[m.end():].strip() if m else dm2["extraneous_text"]
                            out_row = {"original": original_value, "status": "PROPOSED", "updated_term": dm2["updated_term"], "extraneous_text": extr2, "confidence": 1.0, "error": "Could not parse LLM JSON; raw saved to llm_debug.log"}
                        else:
                            out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": "Could not parse LLM response as JSON; raw saved to llm_debug.log"}
                    else:
                        updated_term = parsed.get("updated_term", "") or ""
                        if updated_term and updated_term not in rbms_set:
                            out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": "LLM suggested term not in rbms_terms.json; rejected."}
                        elif updated_term:
                            m = re.match(r'\s*' + re.escape(updated_term), original_value, flags=re.IGNORECASE)
                            extr = original_value[m.end():].strip() if m else str(parsed.get("extraneous_text", "")).strip()
                            out_row = {"original": original_value, "status": "PROPOSED", "updated_term": updated_term, "extraneous_text": extr, "confidence": float(parsed.get("confidence", 1.0)), "error": parsed.get("error", "") or ""}
                        else:
                            out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": parsed.get("error", "") or "LLM returned empty updated_term"}
            except Exception as e:
                out_row = {"original": original_value, "status": "REVIEW", "updated_term": "", "extraneous_text": "", "confidence": 0.0, "error": f"OpenAI error: {e}"}

        append_row_to_csv(args.output_csv, out_row)
        processed_set.add(original_value)
        new_rows_count += 1

        if new_rows_count % snapshot_interval == 0:
            try:
                write_xlsx_from_csv(args.output_csv, args.output_xlsx)
                print(f"Snapshot written to {args.output_xlsx} after {new_rows_count} new rows.")
            except Exception as e:
                print(f"Warning: failed to write XLSX snapshot: {e}")

        if new_rows_count % 10 == 0:
            print(f"Processed {new_rows_count}/{len([v for v in to_process if v not in processed_set])} new rows...")

    if new_rows_count > 0:
        try:
            write_xlsx_from_csv(args.output_csv, args.output_xlsx)
            print(f"Final snapshot written to {args.output_xlsx}.")
        except Exception as e:
            print(f"Warning: failed to write final XLSX snapshot: {e}")
    else:
        print("No new rows processed; no snapshot created.")

    print("Done.")

if __name__ == "__main__":    
    main()