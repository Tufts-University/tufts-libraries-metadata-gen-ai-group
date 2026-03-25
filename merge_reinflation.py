import io
import csv
import argparse
import pandas as pd
import numpy as np  

def main():
    parser = argparse.ArgumentParser(description="Left-merge reinflation workbook with All Processed.txt")
    parser.add_argument("--left-xlsx", required=True, help="Path to the Excel workbook")
    parser.add_argument("--right-txt", required=True, help="Path to the tab-delimited text file")
    parser.add_argument("--sheet", default="Sheet1", help="Left sheet name (default: Sheet1)")
    parser.add_argument("--left-key", default="655 - Local Param 04", help="Left join column")
    parser.add_argument("--right-key", default="original", help="Right join column")
    parser.add_argument("--merged-out", default="RBMSgenreMMSIDs_forReinflation_left_merged.xlsx")
    parser.add_argument("--unmatched-rows-out", default="RBMSgenreMMSIDs_forReinflation_unmatched_rows.xlsx")
    parser.add_argument("--unmatched-unique-out", default="RBMSgenreMMSIDs_forReinflation_unmatched_655_unique.csv")
    args = parser.parse_args()

    left = pd.read_excel(args.left_xlsx, sheet_name=args.sheet, dtype=str)
    left = left.where(pd.notna(left), "")

    # Read only the first tab-delimited block.
    # The provided file also contains an appended CSV copy later in the file,
    # so we stop before that repeated CSV header.
    with open(args.right_txt, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    csv_header = "id,original,status,updated_term,extraneous_text,confidence,error"
    cut_idx = next(
        (i for i, line in enumerate(lines[1:], start=1) if line.startswith(csv_header)),
        len(lines)
    )
    right = pd.read_csv(
        io.StringIO("".join(lines[:cut_idx])), 
        sep="\t", 
        dtype=str,
        quoting=csv.QUOTE_NONNUMERIC,
        quotechar='"'
    )
    right = right.where(pd.notna(right), "")

    # Normalize keys by trimming outer whitespace.
    left["_merge_key"] = left[args.left_key].astype(str).str.strip()
    right["_merge_key"] = right[args.right_key].astype(str).str.strip()

    # Deduplicate right side by key, keeping the first row,
    # so the left merge does not multiply rows.
    right_nonblank = right[right["_merge_key"] != ""].copy()
    right_dedup = right_nonblank.drop_duplicates(subset=["_merge_key"], keep="first").copy()

    merged = left.merge(right_dedup, how="left", on="_merge_key", suffixes=("", "_right"))
    match_mask = merged[args.right_key].notna() & (merged[args.right_key] != "")
    merged["match_found"] = np.where(match_mask, "Y", "N")

    right_cols = [c for c in right_dedup.columns if c != "_merge_key"]
    merged_output = merged[left.columns.tolist()[:-1] + ["match_found"] + right_cols]

    unmatched_rows = merged_output[merged_output["match_found"] == "N"][[args.left_key, "MMS Id"]].copy()

    unique_unmatched = unmatched_rows[[args.left_key]].copy()
    unique_unmatched[args.left_key] = unique_unmatched[args.left_key].astype(str).str.strip()
    unique_unmatched = (
        unique_unmatched[unique_unmatched[args.left_key] != ""]
        .drop_duplicates()
        .sort_values(args.left_key)
    )

    with pd.ExcelWriter(args.merged_out, engine="openpyxl") as writer:
        merged_output.to_excel(writer, index=False, sheet_name="merged")

    with pd.ExcelWriter(args.unmatched_rows_out, engine="openpyxl") as writer:
        unmatched_rows.to_excel(writer, index=False, sheet_name="unmatched_rows")

    unique_unmatched.to_csv(args.unmatched_unique_out, index=False, encoding="utf-8")

    print(f"Left rows: {len(left):,}")
    print(f"Right rows (tab block): {len(right):,}")
    print(f"Right unique keys kept: {len(right_dedup):,}")
    print(f"Matched left rows: {match_mask.sum():,}")
    print(f"Unmatched left rows: {len(unmatched_rows):,}")
    print(f"Unique unmatched 655 values: {len(unique_unmatched):,}")

if __name__ == "__main__":
    main()
