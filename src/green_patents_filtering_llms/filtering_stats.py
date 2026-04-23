'''
@File    :   filtering_stats.py
@Time    :   04/2026
@Author  :   nikifori
@Version :   -
'''
from pathlib import Path
import pandas as pd

FILE1 = Path("/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-llm-filtered.csv")
FILE2 = Path("/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-openai-batch-filtered-5_4_nano_minimal.csv")
# FILE2 = Path("/home/nikifori/Desktop/thesis/repo/data/patent-query-over-2017-openai-batch-filtered_5_nano_minimal.csv")

FILE1_LABEL = "Ensemble_Final_Label"
# FILE1_LABEL = "Final_Label"
FILE2_LABEL = "Final_Label"
KEY = "Lens ID"

GREEN = "Green Patent"
NOT_GREEN = "Not Green Patent"

WRITE_OUTPUT = True
MERGED_OUTPUT = Path("/home/nikifori/Desktop/thesis/repo/data/patent-classification-comparison.csv")
DISAGREEMENTS_OUTPUT = Path("/home/nikifori/Desktop/thesis/repo/data/patent-classification-disagreements.csv")


def clean_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def check_columns(df: pd.DataFrame, needed: list[str], name: str) -> None:
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def get_basic_stats(df: pd.DataFrame, label_col: str) -> dict:
    total = len(df)
    green = (df[label_col] == GREEN).sum()
    not_green = (df[label_col] == NOT_GREEN).sum()
    other = total - green - not_green

    return {
        "rows": total,
        "green_count": green,
        "green_pct": 100 * green / total if total else 0.0,
        "not_green_count": not_green,
        "not_green_pct": 100 * not_green / total if total else 0.0,
        "other_count": other,
        "other_pct": 100 * other / total if total else 0.0,
    }


def print_basic_stats(title: str, stats: dict) -> None:
    print(f"\n{title}")
    print(f"Rows: {stats['rows']:,}")
    print(f"Green Patent: {stats['green_count']:,} ({stats['green_pct']:.2f}%)")
    print(f"Not Green Patent: {stats['not_green_count']:,} ({stats['not_green_pct']:.2f}%)")
    if stats["other_count"] > 0:
        print(f"Other/Invalid: {stats['other_count']:,} ({stats['other_pct']:.2f}%)")


def main() -> None:
    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)

    check_columns(df1, [KEY, FILE1_LABEL], "File 1")
    check_columns(df2, [KEY, FILE2_LABEL], "File 2")

    df1[KEY] = df1[KEY].astype(str).str.strip()
    df2[KEY] = df2[KEY].astype(str).str.strip()
    df1[FILE1_LABEL] = clean_label(df1[FILE1_LABEL])
    df2[FILE2_LABEL] = clean_label(df2[FILE2_LABEL])

    dup1 = df1[KEY].duplicated().sum()
    dup2 = df2[KEY].duplicated().sum()

    print("Duplicate Lens ID rows before deduplication:")
    print(f"File 1: {dup1:,}")
    print(f"File 2: {dup2:,}")

    stats1 = get_basic_stats(df1, FILE1_LABEL)
    stats2 = get_basic_stats(df2, FILE2_LABEL)

    print_basic_stats("File 1 stats", stats1)
    print_basic_stats("File 2 stats", stats2)

    df1 = df1.drop_duplicates(subset=[KEY], keep="first").copy()
    df2 = df2.drop_duplicates(subset=[KEY], keep="first").copy()

    df1 = df1.rename(columns={FILE1_LABEL: "Label_File1"})
    df2 = df2.rename(columns={FILE2_LABEL: "Label_File2"})

    merged = pd.merge(
        df1[[KEY, "Label_File1"]],
        df2[[KEY, "Label_File2"]],
        on=KEY,
        how="inner",
        validate="one_to_one",
    )

    print(f"\nCommon patents in both files: {len(merged):,}")

    if merged.empty:
        print("No common Lens IDs found.")
        return

    both_green = ((merged["Label_File1"] == GREEN) & (merged["Label_File2"] == GREEN)).sum()
    both_not_green = ((merged["Label_File1"] == NOT_GREEN) & (merged["Label_File2"] == NOT_GREEN)).sum()
    file1_green_file2_not = ((merged["Label_File1"] == GREEN) & (merged["Label_File2"] == NOT_GREEN)).sum()
    file1_not_file2_green = ((merged["Label_File1"] == NOT_GREEN) & (merged["Label_File2"] == GREEN)).sum()

    file1_green_total = (merged["Label_File1"] == GREEN).sum()
    file2_green_total = (merged["Label_File2"] == GREEN).sum()

    pct_file1_green_also_green = 100 * both_green / file1_green_total if file1_green_total else 0.0
    pct_file2_green_also_green = 100 * both_green / file2_green_total if file2_green_total else 0.0
    agreement = 100 * (both_green + both_not_green) / len(merged)

    print("\nCombined stats")
    print(f"Both Green Patent: {both_green:,}")
    print(f"Both Not Green Patent: {both_not_green:,}")
    print(f"File 1 Green / File 2 Not Green: {file1_green_file2_not:,}")
    print(f"File 1 Not Green / File 2 Green: {file1_not_file2_green:,}")
    print(f"Overall agreement: {agreement:.2f}%")

    print("\nGreen overlap")
    print(
        f"From File 1 green patents, also green in File 2: "
        f"{both_green:,} / {file1_green_total:,} ({pct_file1_green_also_green:.2f}%)"
    )
    print(
        f"From File 2 green patents, also green in File 1: "
        f"{both_green:,} / {file2_green_total:,} ({pct_file2_green_also_green:.2f}%)"
    )

    print("\nConfusion matrix")
    confusion = pd.crosstab(merged["Label_File1"], merged["Label_File2"])
    print(confusion)

    file1_only = len(set(df1[KEY]) - set(df2[KEY]))
    file2_only = len(set(df2[KEY]) - set(df1[KEY]))

    print("\nCoverage differences")
    print(f"Lens IDs only in File 1: {file1_only:,}")
    print(f"Lens IDs only in File 2: {file2_only:,}")

    green1 = set(df1.loc[df1["Label_File1"] == GREEN, KEY])
    green2 = set(df2.loc[df2["Label_File2"] == GREEN, KEY])
    green_intersection = len(green1 & green2)
    green_union = len(green1 | green2)
    green_jaccard = 100 * green_intersection / green_union if green_union else 0.0

    print(f"\nGreen-set Jaccard similarity: {green_jaccard:.2f}%")

    merged["Same_Label"] = merged["Label_File1"] == merged["Label_File2"]
    merged["Both_Green"] = (merged["Label_File1"] == GREEN) & (merged["Label_File2"] == GREEN)

    # both_green_ids = set(merged.loc[merged["Both_Green"], KEY])
    # file1_both_green = df1[df1[KEY].isin(both_green_ids)].copy()

    # output_green_both = Path("./data/patent-query-over-2017-llm-filtered-GREEN_deduped_v2.csv")
    # file1_both_green.to_csv(output_green_both, index=False)

    # print(f"Saved File 1 patents that are Green in both files to: {output_green_both}")
    # print(f"Rows saved: {len(file1_both_green):,}")

    disagreements = merged[merged["Label_File1"] != merged["Label_File2"]].copy()

    if WRITE_OUTPUT:
        # merged.to_csv(MERGED_OUTPUT, index=False)
        disagreements.to_csv(DISAGREEMENTS_OUTPUT, index=False)
        # print(f"\nSaved merged comparison to: {MERGED_OUTPUT}")
        print(f"Saved disagreements to: {DISAGREEMENTS_OUTPUT}")


if __name__ == "__main__":
    main()