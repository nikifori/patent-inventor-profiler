'''
@File    :   post_processing.py
@Time    :   11/2025
@Author  :   nikifori
@Version :   -
'''
from pathlib import Path
import pandas as pd
import sys
import re
import unicodedata

# --- Hardcoded inputs ---
CSV_PATH = Path("/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-llm-filtered.csv")
ENSEMBLE_COL = "Ensemble_Final_Label"  # exact column name
CSV_DELIM = ","

# --- Dedup config ---
DEDUP_TITLE_COL = "Title"
DEDUP_ABSTRACT_COL = "Abstract"
DEDUP_FAMILY_COL = "Simple Family Members"
PUB_DATE_COL = "Publication Date"
DOC_TYPE_COL = "Document Type"
FULLTEXT_COL = "Has Full Text"

# --- Output files (same folder as input) ---
OUT_GREEN = CSV_PATH.with_name(CSV_PATH.stem + "-GREEN_deduped.csv")
OUT_NOT_GREEN = CSV_PATH.with_name(CSV_PATH.stem + "-NOT_GREEN_deduped.csv")
OUT_DEDUPED = CSV_PATH.with_name(CSV_PATH.stem + "-DEDUPED.csv")  # optional: full deduped dataset

def pct(n, total):
    return f"{(100.0*n/total):.2f}%" if total else "0.00%"

_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    """
    Deterministic normalization for Title/Abstract.
    - Unicode normalize (NFKC)
    - lower
    - strip punctuation
    - collapse whitespace
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = _punct_re.sub(" ", s)      # remove punctuation/symbol-ish chars
    s = _ws_re.sub(" ", s).strip() # normalize whitespace
    return s

def normalize_family_members(s: str) -> str:
    """
    Normalize the 'Simple Family Members' field:
    - split by ';;'
    - trim
    - sort
    - join back
    This makes the key order-insensitive.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    parts = [p.strip() for p in str(s).split(";;") if p.strip()]
    parts.sort()
    return ";;".join(parts)

def check_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        print(f"Columns available: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

def duplicate_report(df, key_cols, label=""):
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_rows = int(dup_mask.sum())
    dup_keys = int(df.loc[dup_mask, key_cols].drop_duplicates().shape[0]) if dup_rows else 0
    print(f"=== Duplicate Report {label}===")
    print(f"Rows with duplicated keys: {dup_rows}")
    print(f"Distinct duplicated keys:  {dup_keys}\n")

def main():
    if not CSV_PATH.exists():
        print(f"Error: file not found -> {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Required columns
    check_cols(df, [ENSEMBLE_COL, DEDUP_TITLE_COL, DEDUP_ABSTRACT_COL, DEDUP_FAMILY_COL, PUB_DATE_COL])

    # ---- Build normalized key columns ----
    df["_norm_title"] = df[DEDUP_TITLE_COL].map(normalize_text)
    df["_norm_abstract"] = df[DEDUP_ABSTRACT_COL].map(normalize_text)
    df["_norm_family"] = df[DEDUP_FAMILY_COL].map(normalize_family_members)

    dedup_key = ["_norm_family", "_norm_title", "_norm_abstract"]

    # Report duplicates before dedup
    duplicate_report(df, dedup_key, label="(before dedup)")

    # ---- Prepare sorting to keep the latest ----
    df["_pub_date"] = pd.to_datetime(df[PUB_DATE_COL], errors="coerce")

    # Tie-breakers:
    # prefer Granted Patent over others, and full text "yes" over "no"
    if DOC_TYPE_COL in df.columns:
        df["_is_granted"] = df[DOC_TYPE_COL].astype(str).str.contains("Granted", case=False, na=False).astype(int)
    else:
        df["_is_granted"] = 0

    if FULLTEXT_COL in df.columns:
        df["_has_fulltext"] = df[FULLTEXT_COL].astype(str).str.strip().str.lower().eq("yes").astype(int)
    else:
        df["_has_fulltext"] = 0

    # Sort so the "best/latest" is last in each duplicate group, then keep='last'
    df = df.sort_values(
        by=["_pub_date", "_is_granted", "_has_fulltext"],
        ascending=[True, True, True],
        kind="mergesort"  # stable
    )

    # ---- Deduplicate (keep latest) ----
    before = len(df)
    df_dedup = df.drop_duplicates(subset=dedup_key, keep="last").copy()
    after = len(df_dedup)
    removed = before - after

    print("=== Deduplication Result ===")
    print(f"Rows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Removed:     {removed}  ({pct(removed, before)})\n")

    duplicate_report(df_dedup, dedup_key, label="(after dedup)")

    # ---- Classification stats & split ----
    if ENSEMBLE_COL not in df_dedup.columns:
        print(f"Error: column '{ENSEMBLE_COL}' not found.", file=sys.stderr)
        print(f"Columns available: {list(df_dedup.columns)}", file=sys.stderr)
        sys.exit(3)

    labels = df_dedup[ENSEMBLE_COL].astype(str).str.strip()

    green_mask = labels.eq("Green Patent")
    not_green_mask = labels.eq("Not Green Patent")
    other_mask = ~(green_mask | not_green_mask)

    total = len(df_dedup)
    green_cnt = int(green_mask.sum())
    not_green_cnt = int(not_green_mask.sum())
    other_cnt = int(other_mask.sum())

    print("=== Green Patent Classification Stats (after dedup) ===")
    print(f"Total rows:           {total}")
    print(f"Green Patent:         {green_cnt}  ({pct(green_cnt, total)})")
    print(f"Not Green Patent:     {not_green_cnt}  ({pct(not_green_cnt, total)})")
    print(f"Unlabeled/Other:      {other_cnt}  ({pct(other_cnt, total)})")

    # ---- Drop internal helper columns from outputs ----
    helper_cols = ["_norm_title", "_norm_abstract", "_norm_family", "_pub_date", "_is_granted", "_has_fulltext"]
    df_out = df_dedup.drop(columns=helper_cols, errors="ignore")

    # --- Save outputs ---
    # Optional: full deduped dataset
    df_out.to_csv(OUT_DEDUPED, index=False, sep=CSV_DELIM)

    df_out.loc[green_mask].to_csv(OUT_GREEN, index=False, sep=CSV_DELIM)
    df_out.loc[not_green_mask].to_csv(OUT_NOT_GREEN, index=False, sep=CSV_DELIM)

    print("\nSaved:")
    print(f"  DEDUPED   → {OUT_DEDUPED}  ({total} rows)")
    print(f"  GREEN     → {OUT_GREEN}  ({green_cnt} rows)")
    print(f"  NOT GREEN → {OUT_NOT_GREEN}  ({not_green_cnt} rows)")

if __name__ == "__main__":
    main()
