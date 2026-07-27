#!/usr/bin/env python3
"""
Straightforward EDA for inventor-skill data.

Default input:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v4/patent_X_inventor_skills.csv

Outputs (CSV + PNG) are written under:
  <input_parent>/eda_inventor_skills
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
    "legend.fontsize": 12,
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
    "/home/nikifori/Desktop/thesis/repo/output/all_data_deduped_gpt_filtered_tfidf/patent_X_inventor_skills.csv"
)

DEFAULT_OUTPUT = Path(
    "/home/nikifori/Desktop/thesis/repo/output/inventor_skills_eda/v3"
)


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

    out_dir = output_dir if output_dir is not None else input_csv.parent / "eda_inventor_skills"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = {"inventor", "skill", "score"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["inventor"] = df["inventor"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["year"] = _resolve_year(df)

    df = df[df["inventor"].ne("") & df["skill"].ne("")].copy()
    df = df[df["score"].notna()].copy()

    patent_col = "Lens ID" if "Lens ID" in df.columns else None

    inventor_group = df.groupby("inventor", dropna=False)
    inventor_stats = inventor_group.agg(
        row_count=("skill", "size"),
        unique_skills=("skill", "nunique"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    )
    if patent_col is not None:
        inventor_stats["unique_patents"] = inventor_group[patent_col].nunique()
    inventor_year_span = (
        df.dropna(subset=["year"])
        .groupby("inventor")
        .agg(first_year=("year", "min"), last_year=("year", "max"), active_years=("year", "nunique"))
    )
    inventor_stats = inventor_stats.join(inventor_year_span, how="left").sort_values(
        ["unique_skills", "row_count"], ascending=[False, False]
    )
    inventor_stats.to_csv(out_dir / "inventor_stats.csv")

    skill_group = df.groupby("skill", dropna=False)
    skill_stats = skill_group.agg(
        row_count=("inventor", "size"),
        unique_inventors=("inventor", "nunique"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
    )
    if patent_col is not None:
        skill_stats["unique_patents"] = skill_group[patent_col].nunique()
    skill_stats = skill_stats.sort_values(["row_count", "unique_inventors"], ascending=[False, False])
    skill_stats.to_csv(out_dir / "skill_stats.csv")

    top_inventors_unique_skills = inventor_stats.sort_values(
        ["unique_skills", "row_count"], ascending=[False, False]
    ).head(top_n)
    top_inventors_unique_skills.to_csv(out_dir / "top_inventors_by_unique_skills.csv")

    top_inventors_rows = inventor_stats.sort_values("row_count", ascending=False).head(top_n)
    top_inventors_rows.to_csv(out_dir / "top_inventors_by_rows.csv")

    top_skills = skill_stats.head(top_n)
    top_skills.to_csv(out_dir / "top_skills_overall.csv")

    yearly_skill_counts = (
        df.dropna(subset=["year"])
        .groupby(["year", "skill"], as_index=False)
        .agg(
            row_count=("inventor", "size"),
            unique_inventors=("inventor", "nunique"),
            mean_score=("score", "mean"),
        )
    )
    yearly_skill_counts = yearly_skill_counts.sort_values(["year", "row_count"], ascending=[True, False])
    yearly_skill_counts["rank_within_year"] = yearly_skill_counts.groupby("year")["row_count"].rank(
        method="first", ascending=False
    )
    top_skills_per_year = yearly_skill_counts[yearly_skill_counts["rank_within_year"] <= top_n].copy()
    top_skills_per_year.to_csv(out_dir / "top_skills_per_year.csv", index=False)

    score_below_06_mask = df["score"] < 0.6

    summary = {
        "input_csv": str(input_csv),
        "rows_after_cleaning": int(len(df)),
        "unique_inventors": int(df["inventor"].nunique()),
        "unique_skills": int(df["skill"].nunique()),
        "score_mean": float(df["score"].mean()),
        "score_median": float(df["score"].median()),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "score_below_0.6_share": float(score_below_06_mask.mean()),
        "score_below_0.6_count": int(score_below_06_mask.sum()),
        "year_min": int(df["year"].dropna().min()) if df["year"].notna().any() else None,
        "year_max": int(df["year"].dropna().max()) if df["year"].notna().any() else None,
    }

    if patent_col is not None:
        patents_per_inventor = inventor_stats["unique_patents"].dropna()
        summary["patents_per_inventor_median"] = float(patents_per_inventor.median())
        summary["patents_per_inventor_mean"] = float(patents_per_inventor.mean())
        summary["single_patent_inventor_share"] = float((patents_per_inventor == 1).mean())
        summary["single_patent_inventor_count"] = int((patents_per_inventor == 1).sum())
    else:
        summary["patents_per_inventor_median"] = None
        summary["patents_per_inventor_mean"] = None
        summary["single_patent_inventor_share"] = None
        summary["single_patent_inventor_count"] = None

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=True)

    _save_hist(
        inventor_stats["unique_skills"],
        out_dir / "hist_unique_skills_per_inventor.png",
        "Distribution of Unique Skills per Inventor",
        "Unique skills",
        show_median=False,
    )
    _save_hist(
        inventor_stats["row_count"],
        out_dir / "hist_rows_per_inventor.png",
        "Distribution of Rows per Inventor",
        "Rows",
        show_median=False,
    )
    _save_hist(
        df["score"],
        out_dir / "hist_skill_score.png",
        "Distribution of Skill Similarity Scores",
        "Score",
    )
    _save_top_bar(
        top_skills.reset_index(),
        x_col="row_count",
        y_col="skill",
        out_path=out_dir / "top_skills_overall_bar.png",
        title=f"Top {top_n} Skills by Row Count",
    )
    _save_top_bar(
        top_inventors_unique_skills.reset_index(),
        x_col="unique_skills",
        y_col="inventor",
        out_path=out_dir / "top_inventors_by_unique_skills_bar.png",
        title=f"Top {top_n} Inventors by Unique Skills",
    )

    top_skills_for_trend = top_skills.reset_index()["skill"].head(min(10, top_n)).tolist()
    trend_df = yearly_skill_counts[yearly_skill_counts["skill"].isin(top_skills_for_trend)].copy()
    if not trend_df.empty:
        pivot = trend_df.pivot(index="year", columns="skill", values="row_count").fillna(0).sort_index()
        fig, ax = plt.subplots(figsize=(14, 8))
        x_years = pivot.index.tolist()
        for i, col in enumerate(pivot.columns):
            color = PALETTE[i % len(PALETTE)]
            y = pivot[col]
            ax.plot(
                x_years,
                y,
                label=col,
                color=color,
                linewidth=2.5,
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

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for patent-inventor-skill CSV.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    out_dir = run_eda(args.input_csv.resolve(), args.output_dir.resolve() if args.output_dir else None, args.top_n)
    print(f"[INFO] EDA outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()