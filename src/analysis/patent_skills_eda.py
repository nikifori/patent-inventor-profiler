#!/usr/bin/env python3
"""
EDA for patent-skill data (patent X skills).

Default input:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v4/patent_X_skills.csv

Default output:
  /home/nikifori/Desktop/thesis/repo/output/patent_skills_eda/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.facecolor": "#eef3f8",
    "axes.facecolor": "#f8fbff",
}
plt.rcParams.update(PLOT_STYLE)

PALETTE = [
    "#1d3557",
    "#457b9d",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#7b2cbf",
    "#3a86ff",
    "#ff006e",
    "#8338ec",
]

DEFAULT_INPUT = Path(
    "/home/nikifori/Desktop/thesis/repo/output/all_data_deduped_gpt_filtered_tfidf/patent_X_skills.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/nikifori/Desktop/thesis/repo/output/patent_skills_eda/v2"
)


def _resolve_patent_id(df: pd.DataFrame) -> pd.Series:
    for col in ["Lens ID", "Display Key", "#", "Application Number"]:
        if col in df.columns:
            return df[col].astype(str).str.strip()
    return pd.Series([f"pat_{i}" for i in range(len(df))], index=df.index)


def _resolve_year(df: pd.DataFrame) -> pd.Series:
    if "Publication Year" in df.columns:
        year = pd.to_numeric(df["Publication Year"], errors="coerce")
        return year.astype("Int64")
    if "Publication Date" in df.columns:
        year = pd.to_datetime(df["Publication Date"], errors="coerce").dt.year
        return year.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def _save_hist(
    series: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 40,
    show_median: bool = True,
) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.hist(values, bins=bins, color="#457b9d", edgecolor="#ffffff", linewidth=0.9, alpha=0.9)
    if show_median:
        median_val = values.median()
        ax.axvline(median_val, color="#e63946", linewidth=2.3, linestyle="--", label=f"Median: {median_val:.3f}")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#c8d3e1", linestyle="--", linewidth=0.8, alpha=0.65)
    if show_median:
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _save_top_bar(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(plot_df))]
    bars = ax.barh(plot_df[y_col], plot_df[x_col], color=colors, edgecolor="#ffffff", linewidth=0.9)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="x", color="#c8d3e1", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_max = float(plot_df[x_col].max()) if len(plot_df) else 0.0
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + (0.01 * x_max if x_max > 0 else 0.1),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.0f}",
            va="center",
            fontsize=11,
            color="#243447",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_eda(input_csv: Path, output_dir: Optional[Path], top_n: int) -> Path:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    out_dir = output_dir if output_dir is not None else input_csv.parent / "eda_patent_skills"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cleanup stale files from older script versions.
    for stale_name in ["top_patents_by_rows.csv", "hist_rows_per_patent.png"]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    df = pd.read_csv(input_csv)

    required_cols = {"skill", "score"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["patent_id"] = _resolve_patent_id(df)
    df["skill"] = df["skill"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["year"] = _resolve_year(df)
    if "Jurisdiction" in df.columns:
        df["jurisdiction"] = df["Jurisdiction"].astype(str).str.strip()
    else:
        df["jurisdiction"] = "NA"

    df = df[df["patent_id"].ne("") & df["skill"].ne("")].copy()
    df = df[df["score"].notna()].copy()

    patent_skill_unique = df.drop_duplicates(subset=["patent_id", "skill"])

    patent_stats = df.groupby("patent_id", dropna=False).agg(
        row_count=("skill", "size"),
        unique_skills=("skill", "nunique"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    )
    if "year" in df.columns:
        year_span = df.dropna(subset=["year"]).groupby("patent_id").agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        patent_stats = patent_stats.join(year_span, how="left")
    if "jurisdiction" in df.columns:
        patent_j = (
            df.groupby("patent_id")["jurisdiction"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .rename("jurisdiction")
        )
        patent_stats = patent_stats.join(patent_j, how="left")
    patent_stats = patent_stats.sort_values(["unique_skills", "row_count"], ascending=[False, False])
    patent_stats.to_csv(out_dir / "patent_stats.csv")

    skill_stats = df.groupby("skill", dropna=False).agg(
        row_count=("patent_id", "size"),
        unique_patents=("patent_id", "nunique"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    )
    skill_years = df.dropna(subset=["year"]).groupby("skill").agg(unique_years=("year", "nunique"))
    skill_stats = skill_stats.join(skill_years, how="left")
    skill_stats = skill_stats.sort_values(["row_count", "unique_patents"], ascending=[False, False])
    skill_stats.to_csv(out_dir / "skill_stats.csv")

    top_patents_by_unique_skills = patent_stats.head(top_n)
    top_patents_by_unique_skills.to_csv(out_dir / "top_patents_by_unique_skills.csv")

    top_skills = skill_stats.head(top_n)
    top_skills.to_csv(out_dir / "top_skills_overall.csv")

    yearly_overview = (
        df.dropna(subset=["year"])
        .groupby("year", as_index=False)
        .agg(
            row_count=("skill", "size"),
            unique_patents=("patent_id", "nunique"),
            unique_skills=("skill", "nunique"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .sort_values("year")
    )
    yearly_overview.to_csv(out_dir / "yearly_overview.csv", index=False)

    top_skills_per_year = (
        df.dropna(subset=["year"])
        .groupby(["year", "skill"], as_index=False)
        .agg(
            row_count=("patent_id", "size"),
            unique_patents=("patent_id", "nunique"),
            mean_score=("score", "mean"),
        )
    )
    top_skills_per_year = top_skills_per_year.sort_values(["year", "row_count"], ascending=[True, False])
    top_skills_per_year["rank_within_year"] = top_skills_per_year.groupby("year")["row_count"].rank(
        method="first", ascending=False
    )
    top_skills_per_year = top_skills_per_year[top_skills_per_year["rank_within_year"] <= top_n].copy()
    top_skills_per_year.to_csv(out_dir / "top_skills_per_year.csv", index=False)

    jurisdiction_stats = (
        df.groupby("jurisdiction", as_index=False)
        .agg(
            row_count=("skill", "size"),
            unique_patents=("patent_id", "nunique"),
            unique_skills=("skill", "nunique"),
            mean_score=("score", "mean"),
        )
        .sort_values("row_count", ascending=False)
    )
    jurisdiction_stats.to_csv(out_dir / "jurisdiction_stats.csv", index=False)
    top_jurisdictions = jurisdiction_stats.head(top_n)
    top_jurisdictions.to_csv(out_dir / "top_jurisdictions.csv", index=False)

    summary = {
        "input_csv": str(input_csv),
        "rows_after_cleaning": int(len(df)),
        "unique_patents": int(df["patent_id"].nunique()),
        "unique_skills": int(df["skill"].nunique()),
        "avg_skills_per_patent": float(patent_skill_unique.groupby("patent_id")["skill"].nunique().mean()),
        "median_skills_per_patent": float(patent_skill_unique.groupby("patent_id")["skill"].nunique().median()),
        "score_mean": float(df["score"].mean()),
        "score_median": float(df["score"].median()),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "year_min": int(df["year"].dropna().min()) if df["year"].notna().any() else None,
        "year_max": int(df["year"].dropna().max()) if df["year"].notna().any() else None,
        "unique_jurisdictions": int(df["jurisdiction"].nunique()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=True)

    _save_hist(
        patent_stats["unique_skills"],
        out_dir / "hist_unique_skills_per_patent.png",
        "Distribution of Unique Skills per Patent",
        "Unique skills",
        show_median=False,
    )
    _save_hist(
        df["score"],
        out_dir / "hist_skill_score.png",
        "Distribution of Skill Similarity Scores",
        "Score",
        show_median=True,
    )

    _save_top_bar(
        top_skills.reset_index(),
        x_col="row_count",
        y_col="skill",
        out_path=out_dir / "top_skills_overall_bar.png",
        title=f"Top {top_n} Skills by Row Count",
    )
    _save_top_bar(
        top_patents_by_unique_skills.reset_index().rename(columns={"patent_id": "patent"}),
        x_col="unique_skills",
        y_col="patent",
        out_path=out_dir / "top_patents_by_unique_skills_bar.png",
        title=f"Top {top_n} Patents by Unique Skills",
    )
    _save_top_bar(
        top_jurisdictions,
        x_col="row_count",
        y_col="jurisdiction",
        out_path=out_dir / "top_jurisdictions_bar.png",
        title=f"Top {top_n} Jurisdictions by Skill Rows",
    )

    top_skills_for_trend = top_skills.reset_index()["skill"].head(min(10, top_n)).tolist()
    trend_df = (
        df.dropna(subset=["year"])
        .groupby(["year", "skill"], as_index=False)
        .agg(row_count=("patent_id", "size"))
    )
    trend_df = trend_df[trend_df["skill"].isin(top_skills_for_trend)].copy()

    if not trend_df.empty:
        pivot = trend_df.pivot(index="year", columns="skill", values="row_count").fillna(0).sort_index()
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, col in enumerate(pivot.columns):
            color = PALETTE[i % len(PALETTE)]
            ax.plot(
                pivot.index,
                pivot[col],
                label=col,
                color=color,
                linewidth=2.4,
                marker="o",
                markersize=4.5,
                alpha=0.95,
            )
        ax.set_title("Yearly Trend of Top Skills (row_count)", pad=12, fontweight="bold")
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Rows")
        ax.grid(color="#c8d3e1", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
        fig.tight_layout()
        fig.savefig(out_dir / "top_skills_yearly_trend.png", dpi=170)
        plt.close(fig)

    if not yearly_overview.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            yearly_overview["year"],
            yearly_overview["unique_patents"],
            color="#1d3557",
            linewidth=2.8,
            marker="o",
            markersize=5.0,
            label="Unique patents",
        )
        ax.plot(
            yearly_overview["year"],
            yearly_overview["unique_skills"],
            color="#2a9d8f",
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            label="Unique skills",
        )
        ax.set_title("Yearly Patent and Skill Coverage", pad=12, fontweight="bold")
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Count")
        ax.grid(color="#c8d3e1", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
        fig.tight_layout()
        fig.savefig(out_dir / "yearly_patent_skill_coverage.png", dpi=170)
        plt.close(fig)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for patent-skill CSV.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    out_dir = run_eda(args.input_csv.resolve(), args.output_dir.resolve() if args.output_dir else None, args.top_n)
    print(f"[INFO] EDA outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
