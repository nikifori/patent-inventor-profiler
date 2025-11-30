'''
@File    :   post_processing.py
@Time    :   11/2025
@Author  :   nikifori
@Version :   -
'''
from pathlib import Path
import pandas as pd
import sys

# --- Hardcoded inputs ---
CSV_PATH = Path("/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-llm-filtered.csv")
ENSEMBLE_COL = "Ensemble_Final_Label"  # exact column name
CSV_DELIM = ","

# --- Output files (same folder as input) ---
OUT_GREEN = CSV_PATH.with_name(CSV_PATH.stem + "-GREEN.csv")
OUT_NOT_GREEN = CSV_PATH.with_name(CSV_PATH.stem + "-NOT_GREEN.csv")

def pct(n, total):
    return f"{(100.0*n/total):.2f}%" if total else "0.00%"

def main():
    if not CSV_PATH.exists():
        print(f"Error: file not found -> {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # ---- Duplicate check ----
    dupl_criterion_col = 'Lens ID'
    if dupl_criterion_col not in df.columns:
        print(f"Error: column '{dupl_criterion_col}' not found in the CSV.", file=sys.stderr)
        print(f"Columns available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    dup_mask_rows = df[dupl_criterion_col].duplicated(keep=False)
    dup_rows_count = int(dup_mask_rows.sum())

    # number of distinct criterion that are duplicated (appear >= 2 times)
    dup_count = int(df.loc[dup_mask_rows, dupl_criterion_col].nunique())

    print(f"=== Duplicate '{dupl_criterion_col}' Report ===")
    print(f"File: {CSV_PATH}")
    print(f"Rows with duplicated {dupl_criterion_col}:    {dup_rows_count}")
    print(f"Distinct duplicated {dupl_criterion_col}:    {dup_count}\n")

    # ---- Classification stats & split ----
    if ENSEMBLE_COL not in df.columns:
        print(f"Error: column '{ENSEMBLE_COL}' not found.", file=sys.stderr)
        print(f"Columns available: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    total = len(df)
    labels = df[ENSEMBLE_COL].astype(str).str.strip()

    green_mask = labels.eq("Green Patent")
    not_green_mask = labels.eq("Not Green Patent")
    other_mask = ~(green_mask | not_green_mask)

    green_cnt = int(green_mask.sum())
    not_green_cnt = int(not_green_mask.sum())
    other_cnt = int(other_mask.sum())

    # --- Print stats ---
    print("=== Green Patent Classification Stats ===")
    print(f"Total rows:           {total}")
    print(f"Green Patent:         {green_cnt}  ({pct(green_cnt, total)})")
    print(f"Not Green Patent:     {not_green_cnt}  ({pct(not_green_cnt, total)})")
    print(f"Unlabeled/Other:      {other_cnt}  ({pct(other_cnt, total)})")

    # --- Save splits ---
    df[green_mask].to_csv(OUT_GREEN, index=False, sep=CSV_DELIM)
    df[not_green_mask].to_csv(OUT_NOT_GREEN, index=False, sep=CSV_DELIM)

    print("\nSaved:")
    print(f"  GREEN → {OUT_GREEN}  ({green_cnt} rows)")
    print(f"  NOT GREEN → {OUT_NOT_GREEN}  ({not_green_cnt} rows)")

if __name__ == "__main__":
    main()