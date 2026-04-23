"""
Optical analysis for archetypes from the best run in:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v3

Outputs are written to:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v3/archetypes_plots

Main visuals:
- t-SNE 2D of inventor skill-input vectors, colored by dominant archetype
- t-SNE 3D of inventor skill-input vectors
- Dominant archetype counts bar chart
- Archetype cosine-similarity heatmap

Extra sparse-aware visuals added:
- UMAP 2D/3D on raw inventor input matrix
- TruncatedSVD(100) + UMAP 2D/3D
- TruncatedSVD(100) + t-SNE 2D/3D

The plotted "X" markers are the actual archetypes, embedded in the same reduced
space as the inventor input vectors. They are not cluster centers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, TSNE
from archetypes.visualization.simplex import simplex
import umap

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "#eef3f8",
    "axes.facecolor": "#f8fbff",
}
plt.rcParams.update(PLOT_STYLE)

DEFAULT_BASE_DIR = Path("/home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v3")
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "archetypes_plots_final_final_final_final_final"


def _with_k_suffix(path: Path, selected_k: int) -> Path:
    return path.with_name(f"{path.stem}_k{selected_k}{path.suffix}")


def _load_latest_auto_run(index_jsonl: Path) -> Dict:
    if not index_jsonl.exists():
        raise FileNotFoundError(f"Missing run index: {index_jsonl}")
    with index_jsonl.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in reversed(rows):
        if row.get("run_mode") == "auto":
            return row
    raise RuntimeError(f"No auto run found in {index_jsonl}")


def _load_run_artifacts(
    run_dir: Path,
    selected_k: int,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.Index, list[str]]:
    model_path = run_dir / "models" / f"k_{selected_k}.npz"
    input_matrix_path = run_dir / "inventor_skill_matrix.csv.gz"
    memberships_path = run_dir / "memberships.csv.gz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not input_matrix_path.exists():
        raise FileNotFoundError(f"Missing inventor matrix artifact: {input_matrix_path}")
    if not memberships_path.exists():
        raise FileNotFoundError(f"Missing memberships artifact: {memberships_path}")

    model = np.load(model_path)
    archetypes = np.asarray(model["archetypes"], dtype=float)

    inventor_input_df = pd.read_csv(input_matrix_path, index_col=0)
    inventor_input_df.index = inventor_input_df.index.astype(str)
    inventor_input_df.columns = [str(col) for col in inventor_input_df.columns]

    memberships_df = pd.read_csv(memberships_path, index_col=0)
    memberships_df.index = memberships_df.index.astype(str)
    memberships_df.columns = [str(col) for col in memberships_df.columns]

    if memberships_df.index.tolist() != inventor_input_df.index.tolist():
        raise RuntimeError(
            "Mismatch between inventor order in inventor_skill_matrix.csv.gz and memberships.csv.gz."
        )

    feature_names = inventor_input_df.columns.tolist()
    inventor_index = pd.Index(inventor_input_df.index)

    inventor_input = sparse.csr_matrix(inventor_input_df.to_numpy(dtype=float, copy=False))
    memberships = memberships_df.to_numpy(dtype=float, copy=False)

    if memberships.shape[0] != len(inventor_index):
        raise RuntimeError(
            "Mismatch between memberships rows and inventor index rows: "
            f"{memberships.shape[0]} vs {len(inventor_index)}"
        )
    if archetypes.shape[1] != len(feature_names):
        raise RuntimeError(
            "Mismatch between archetype features and matrix columns: "
            f"{archetypes.shape[1]} vs {len(feature_names)}"
        )
    if memberships.shape[1] != selected_k:
        raise RuntimeError(
            "Mismatch between membership columns and selected_k: "
            f"{memberships.shape[1]} vs {selected_k}"
        )

    return inventor_input, memberships, archetypes, inventor_index, feature_names


def _plot_membership_simplex(
    memberships: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    max_points: int = 4000,
    random_state: int = 42,
) -> None:
    memberships = np.asarray(memberships, dtype=float)

    if np.any(memberships < 0):
        raise ValueError("memberships must be non-negative for simplex plotting")

    row_sums = memberships.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    memberships = memberships / safe_row_sums

    dominant = np.argmax(memberships, axis=1)
    purity = np.max(memberships, axis=1)

    n = memberships.shape[0]
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        memberships_plot = memberships[idx]
        dominant_plot = dominant[idx]
        purity_plot = purity[idx]
    else:
        memberships_plot = memberships
        dominant_plot = dominant
        purity_plot = purity

    cmap = plt.get_cmap("tab20", k)
    point_colors = [cmap(i) for i in dominant_plot]

    fig, ax = plt.subplots(figsize=(12, 12))

    simplex(
        memberships_plot,
        c=point_colors,
        s=(purity_plot ** 2) * 100.0,
        alpha=0.6,
        ax=ax,
        show_axis=True,
        show_vertices=False,
        axis_params={
            "color": "#4a4a4a",
            "linewidth": 1.0,
            "linestyle": "-",
            "zorder": 0,
        },
    )

    theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
    for i, angle in enumerate(theta):
        x = np.cos(angle)
        y = np.sin(angle)

        # place a large colored bullet slightly outside the simplex vertex
        # ax.text(
        #     1.08 * x,
        #     1.08 * y,
        #     "●",
        #     ha="center",
        #     va="center",
        #     fontsize=22,
        #     fontweight="bold",
        #     color=cmap(i),
        #     alpha=0.55,
        #     zorder=5,
        # )

        # place the archetype label a bit further out, in the same color
        ax.text(
            1.08 * x,
            1.08 * y,
            f"A{i+1}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=cmap(i),
            zorder=6,
        )

    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_membership_simplex_vertices_csv(out_path: Path, k: int) -> None:
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
    vertices_df = pd.DataFrame(
        {
            "archetype": [f"A{i}" for i in range(1, k + 1)],
            "x": np.cos(theta),
            "y": np.sin(theta),
        }
    )
    vertices_df.to_csv(out_path, index=False)


def _cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    return normalized @ normalized.T


def _sample_indices_by_group(groups: np.ndarray, max_points: int, random_state: int) -> np.ndarray:
    n = len(groups)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(random_state)
    sampled = []
    group_ids, group_counts = np.unique(groups, return_counts=True)
    proportions = group_counts / group_counts.sum()

    for gid, p in zip(group_ids, proportions):
        idx = np.flatnonzero(groups == gid)
        take = max(1, int(round(max_points * p)))
        take = min(take, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        sampled.append(chosen)

    sampled_idx = np.unique(np.concatenate(sampled))
    if len(sampled_idx) > max_points:
        sampled_idx = rng.choice(sampled_idx, size=max_points, replace=False)
    elif len(sampled_idx) < max_points:
        remaining = np.setdiff1d(np.arange(n), sampled_idx)
        need = min(max_points - len(sampled_idx), len(remaining))
        if need > 0:
            extra = rng.choice(remaining, size=need, replace=False)
            sampled_idx = np.concatenate([sampled_idx, extra])
    return np.sort(sampled_idx)


def _split_joint_embedding(embedding: np.ndarray, n_inventors: int) -> Tuple[np.ndarray, np.ndarray]:
    return embedding[:n_inventors], embedding[n_inventors:]


def _stack_sparse_dense_for_joint_embedding(
    inventor_sparse: sparse.csr_matrix,
    archetypes: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.vstack(
        [
            inventor_sparse,
            sparse.csr_matrix(np.asarray(archetypes, dtype=float)),
        ],
        format="csr",
    )


def _plot_embedding_2d(
    inventor_embedding: np.ndarray,
    archetype_embedding: np.ndarray,
    dominant: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    cmap = plt.get_cmap("tab20", k)
    fig, ax = plt.subplots(figsize=(12, 9))
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if np.any(mask):
            ax.scatter(
                inventor_embedding[mask, 0],
                inventor_embedding[mask, 1],
                s=14,
                alpha=0.58,
                color=cmap(arch_id - 1),
                label=f"A{arch_id}",
            )

        arch_point = archetype_embedding[arch_id - 1]
        ax.scatter(
            arch_point[0],
            arch_point[1],
            marker="X",
            s=180,
            color=cmap(arch_id - 1),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.8,
            zorder=5,
        )
        ax.text(
            arch_point[0],
            arch_point[1],
            f"A{arch_id}",
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_embedding_3d(
    inventor_embedding: np.ndarray,
    archetype_embedding: np.ndarray,
    dominant: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = plt.get_cmap("tab20", k)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if np.any(mask):
            ax.scatter(
                inventor_embedding[mask, 0],
                inventor_embedding[mask, 1],
                inventor_embedding[mask, 2],
                s=10,
                alpha=0.45,
                color=cmap(arch_id - 1),
                label=f"A{arch_id}",
            )

        arch_point = archetype_embedding[arch_id - 1]
        ax.scatter(
            arch_point[0],
            arch_point[1],
            arch_point[2],
            marker="X",
            alpha=0.8,
            s=220,
            color=cmap(arch_id - 1),
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
        )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend(loc="upper left", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_archetype_counts(dominant: np.ndarray, out_path: Path, k: int) -> None:
    counts = pd.Series(dominant).value_counts().reindex(range(1, k + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(
        [f"A{i}" for i in counts.index],
        counts.values,
        color="#457b9d",
        edgecolor="#ffffff",
        linewidth=0.9,
    )
    ax.set_title("Inventors per Dominant Archetype", pad=12, fontweight="bold")
    ax.set_xlabel("Archetype")
    ax.set_ylabel("Inventor count")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_archetype_similarity_heatmap(similarity: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Archetype Cosine Similarity (skill-space)", pad=12, fontweight="bold")
    ticks = np.arange(similarity.shape[0])
    labels = [f"A{i+1}" for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            val = similarity[i, j]
            txt_color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_top_features_csv(archetypes: np.ndarray, feature_names: list[str], out_path: Path, top_n: int = 20) -> None:
    rows = []
    for i in range(archetypes.shape[0]):
        arch_label = f"Archetype_{i+1}"
        s = pd.Series(archetypes[i], index=feature_names)
        top = s.sort_values(ascending=False).head(top_n)
        for rank, (feat, val) in enumerate(top.items(), start=1):
            rows.append(
                {
                    "archetype": arch_label,
                    "rank": rank,
                    "feature": feat,
                    "weight": float(val),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _fit_umap(matrix, n_components: int, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.10,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def _safe_svd_components(n_rows: int, n_cols: int, target: int = 100) -> int:
    return max(2, min(int(target), n_rows - 1, n_cols - 1))


def _compute_purity(memberships: np.ndarray) -> np.ndarray:
    memberships = np.asarray(memberships, dtype=float)
    row_sums = memberships.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    memberships = memberships / safe_row_sums
    return np.max(memberships, axis=1)


def _purity_stats_dict(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
            "iqr": np.nan,
        }

    q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.quantile(
        values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "var": float(np.var(values, ddof=0)),
        "min": float(np.min(values)),
        "q01": float(q01),
        "q05": float(q05),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "q95": float(q95),
        "q99": float(q99),
        "max": float(np.max(values)),
        "iqr": float(q75 - q25),
    }


def _save_purity_stats_overall(purity: np.ndarray, out_path: Path) -> None:
    stats = _purity_stats_dict(purity)
    pd.DataFrame([stats]).to_csv(out_path, index=False)


def _save_purity_stats_by_archetype(
    purity: np.ndarray,
    dominant: np.ndarray,
    k: int,
    out_path: Path,
) -> None:
    rows = []
    for arch_id in range(1, k + 1):
        vals = purity[dominant == arch_id]
        stats = _purity_stats_dict(vals)
        stats["archetype"] = f"A{arch_id}"
        rows.append(stats)

    cols = [
        "archetype",
        "count",
        "mean",
        "std",
        "var",
        "min",
        "q01",
        "q05",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "q95",
        "q99",
        "max",
        "iqr",
    ]
    pd.DataFrame(rows)[cols].to_csv(out_path, index=False)


def _plot_purity_histogram(
    purity: np.ndarray,
    out_path: Path,
    title: str,
    bins: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    counts, bin_edges, patches = ax.hist(
        purity,
        bins=bins,
        range=(0.0, 1.0),
        color="#457b9d",
        edgecolor="#ffffff",
        linewidth=0.9,
    )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Purity (max membership)")
    ax.set_ylabel("Inventor count")
    ax.grid(axis="y", alpha=0.25)

    mean_val = float(np.mean(purity)) if len(purity) > 0 else np.nan
    median_val = float(np.median(purity)) if len(purity) > 0 else np.nan
    if np.isfinite(mean_val):
        ax.axvline(mean_val, color="#d62828", linestyle="--", linewidth=1.6, label=f"Mean = {mean_val:.3f}")
    if np.isfinite(median_val):
        ax.axvline(median_val, color="#2a9d8f", linestyle=":", linewidth=1.8, label=f"Median = {median_val:.3f}")

    ymax = max(counts) if len(counts) > 0 else 0
    if ymax > 0:
        for c, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
            if c > 0:
                ax.text(
                    (left + right) / 2.0,
                    c,
                    f"{int(c)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_purity_histogram_by_archetype(
    purity: np.ndarray,
    dominant: np.ndarray,
    k: int,
    out_dir: Path,
    bins: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20", k)

    for arch_id in range(1, k + 1):
        vals = purity[dominant == arch_id]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        counts, bin_edges, patches = ax.hist(
            vals,
            bins=bins,
            range=(0.0, 1.0),
            color=cmap(arch_id - 1),
            edgecolor="#ffffff",
            linewidth=0.9,
        )

        if len(vals) > 0:
            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            ax.axvline(mean_val, color="#d62828", linestyle="--", linewidth=1.6, label=f"Mean = {mean_val:.3f}")
            ax.axvline(median_val, color="#2a9d8f", linestyle=":", linewidth=1.8, label=f"Median = {median_val:.3f}")

        ax.set_title(f"Purity Histogram for Dominant Archetype A{arch_id}", pad=12, fontweight="bold")
        ax.set_xlabel("Purity (max membership)")
        ax.set_ylabel("Inventor count")
        ax.grid(axis="y", alpha=0.25)

        for c, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
            if c > 0:
                ax.text(
                    (left + right) / 2.0,
                    c,
                    f"{int(c)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
        fig.tight_layout()

        out_path = out_dir / f"purity_histogram_A{arch_id}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def run_analysis(base_dir: Path, output_dir: Path, max_inventors_tsne: int, random_state: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = _load_latest_auto_run(base_dir / "archetype_runs" / "index.jsonl")
    run_dir = Path(run["run_dir"])
    selected_k = int(run["selected_k"])

    inventor_input, memberships, archetypes, inventor_index, feature_names = _load_run_artifacts(
        run_dir=run_dir,
        selected_k=selected_k,
    )

    dominant = np.argmax(memberships, axis=1) + 1
    sample_idx = _sample_indices_by_group(dominant, max_inventors_tsne, random_state)

    purity = _compute_purity(memberships)

    _plot_purity_histogram(
        purity=purity,
        out_path=_with_k_suffix(output_dir / "purity_histogram_overall.png", selected_k),
        title="Purity Histogram (all inventors)",
        bins=30,
    )

    _plot_purity_histogram_by_archetype(
        purity=purity,
        dominant=dominant,
        k=selected_k,
        out_dir=_with_k_suffix(output_dir / "purity_histograms_by_archetype.png", selected_k),
        bins=20,
    )

    _save_purity_stats_overall(
        purity=purity,
        out_path=_with_k_suffix(output_dir / "purity_stats_overall.csv", selected_k),
    )

    _save_purity_stats_by_archetype(
        purity=purity,
        dominant=dominant,
        k=selected_k,
        out_path=_with_k_suffix(output_dir / "purity_stats_by_archetype.csv", selected_k),
    )

    _plot_membership_simplex(
        memberships=memberships,
        out_path=_with_k_suffix(output_dir / "membership_simplex_plot.png", selected_k),
        k=selected_k,
        title="Simplex: Colored by Dominant Archetype",
        max_points=4000,
        random_state=random_state,
    )

    _save_membership_simplex_vertices_csv(
        _with_k_suffix(output_dir / "membership_simplex_vertices.csv", selected_k),
        selected_k,
    )

    input_sample = inventor_input[sample_idx]
    dominant_sample = dominant[sample_idx]

    # Dense joint matrix for the original code paths
    joint_matrix = np.vstack([input_sample.toarray(), archetypes])
    n_sample = input_sample.shape[0]
    n_joint = joint_matrix.shape[0]

    perplexity = min(35, max(5, n_joint // 40))
    if perplexity >= n_joint:
        perplexity = max(1, n_joint - 1)

    # -----------------------------
    # Existing plots
    # -----------------------------
    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb2_all = tsne_2d.fit_transform(joint_matrix)
    emb2, emb2_arch = _split_joint_embedding(emb2_all, n_sample)
    _plot_embedding_2d(
        emb2,
        emb2_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "tsne_2d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in t-SNE 2D (colored by dominant archetype)",
        "t-SNE component 1",
        "t-SNE component 2",
    )

    pca_2d = PCA(n_components=2, random_state=random_state)
    emb_pca2_all = pca_2d.fit_transform(joint_matrix)
    emb_pca2, emb_pca2_arch = _split_joint_embedding(emb_pca2_all, n_sample)
    _plot_embedding_2d(
        emb_pca2,
        emb_pca2_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "pca_2d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in PCA 2D (colored by dominant archetype)",
        "PCA component 1",
        "PCA component 2",
    )

    pca_3d = PCA(n_components=3, random_state=random_state)
    emb_pca3_all = pca_3d.fit_transform(joint_matrix)
    emb_pca3, emb_pca3_arch = _split_joint_embedding(emb_pca3_all, n_sample)
    _plot_embedding_3d(
        emb_pca3,
        emb_pca3_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "pca_3d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in PCA 3D (colored by dominant archetype)",
        "PCA c1",
        "PCA c2",
        "PCA c3",
    )

    n_neighbors = min(15, max(2, n_joint - 1))
    isomap_2d = Isomap(n_components=2, n_neighbors=n_neighbors)
    emb_iso2_all = isomap_2d.fit_transform(joint_matrix)
    emb_iso2, emb_iso2_arch = _split_joint_embedding(emb_iso2_all, n_sample)
    _plot_embedding_2d(
        emb_iso2,
        emb_iso2_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "isomap_2d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in Isomap 2D (colored by dominant archetype)",
        "Isomap component 1",
        "Isomap component 2",
    )

    isomap_3d = Isomap(n_components=3, n_neighbors=n_neighbors)
    emb_iso3_all = isomap_3d.fit_transform(joint_matrix)
    emb_iso3, emb_iso3_arch = _split_joint_embedding(emb_iso3_all, n_sample)
    _plot_embedding_3d(
        emb_iso3,
        emb_iso3_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "isomap_3d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in Isomap 3D (colored by dominant archetype)",
        "Isomap c1",
        "Isomap c2",
        "Isomap c3",
    )

    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb3_all = tsne_3d.fit_transform(joint_matrix)
    emb3, emb3_arch = _split_joint_embedding(emb3_all, n_sample)
    _plot_embedding_3d(
        emb3,
        emb3_arch,
        dominant_sample,
        _with_k_suffix(output_dir / "tsne_3d_inventor_skills.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in t-SNE 3D (colored by dominant archetype)",
        "t-SNE c1",
        "t-SNE c2",
        "t-SNE c3",
    )

    # -----------------------------
    # NEW PLOTS
    # -----------------------------
    joint_sparse = _stack_sparse_dense_for_joint_embedding(input_sample, archetypes)

    # 1) UMAP on raw sparse input + archetypes
    emb_umap_raw_all_2d = _fit_umap(joint_sparse, n_components=2, random_state=random_state)
    emb_umap_raw_2d, emb_umap_raw_arch_2d = _split_joint_embedding(emb_umap_raw_all_2d, n_sample)
    _plot_embedding_2d(
        emb_umap_raw_2d,
        emb_umap_raw_arch_2d,
        dominant_sample,
        _with_k_suffix(output_dir / "umap_2d_raw_input.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in UMAP 2D (raw sparse TF-IDF, colored by dominant archetype)",
        "UMAP component 1",
        "UMAP component 2",
    )

    emb_umap_raw_all_3d = _fit_umap(joint_sparse, n_components=3, random_state=random_state)
    emb_umap_raw_3d, emb_umap_raw_arch_3d = _split_joint_embedding(emb_umap_raw_all_3d, n_sample)
    _plot_embedding_3d(
        emb_umap_raw_3d,
        emb_umap_raw_arch_3d,
        dominant_sample,
        _with_k_suffix(output_dir / "umap_3d_raw_input.png", selected_k),
        selected_k,
        "Inventor Skill Input Space in UMAP 3D (raw sparse TF-IDF, colored by dominant archetype)",
        "UMAP c1",
        "UMAP c2",
        "UMAP c3",
    )

    # 2) SVD(100) + UMAP
    svd_n_components = _safe_svd_components(
        n_rows=n_joint,
        n_cols=joint_sparse.shape[1],
        target=100,
    )
    svd_100 = TruncatedSVD(n_components=svd_n_components, random_state=random_state)
    joint_svd = svd_100.fit_transform(joint_sparse)

    emb_umap_svd_all_2d = _fit_umap(joint_svd, n_components=2, random_state=random_state)
    emb_umap_svd_2d, emb_umap_svd_arch_2d = _split_joint_embedding(emb_umap_svd_all_2d, n_sample)
    _plot_embedding_2d(
        emb_umap_svd_2d,
        emb_umap_svd_arch_2d,
        dominant_sample,
        _with_k_suffix(output_dir / "svd100_umap_2d_input.png", selected_k),
        selected_k,
        f"Inventor Skill Input Space in SVD({svd_n_components}) + UMAP 2D (colored by dominant archetype)",
        "UMAP component 1",
        "UMAP component 2",
    )

    emb_umap_svd_all_3d = _fit_umap(joint_svd, n_components=3, random_state=random_state)
    emb_umap_svd_3d, emb_umap_svd_arch_3d = _split_joint_embedding(emb_umap_svd_all_3d, n_sample)
    _plot_embedding_3d(
        emb_umap_svd_3d,
        emb_umap_svd_arch_3d,
        dominant_sample,
        _with_k_suffix(output_dir / "svd100_umap_3d_input.png", selected_k),
        selected_k,
        f"Inventor Skill Input Space in SVD({svd_n_components}) + UMAP 3D (colored by dominant archetype)",
        "UMAP c1",
        "UMAP c2",
        "UMAP c3",
    )

    # 3) SVD(100) + t-SNE
    perplexity_svd_tsne = min(35, max(5, n_joint // 40))
    if perplexity_svd_tsne >= n_joint:
        perplexity_svd_tsne = max(1, n_joint - 1)

    tsne_svd_2d = TSNE(
        n_components=2,
        perplexity=perplexity_svd_tsne,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb_svd_tsne_all_2d = tsne_svd_2d.fit_transform(joint_svd)
    emb_svd_tsne_2d, emb_svd_tsne_arch_2d = _split_joint_embedding(emb_svd_tsne_all_2d, n_sample)
    _plot_embedding_2d(
        emb_svd_tsne_2d,
        emb_svd_tsne_arch_2d,
        dominant_sample,
        _with_k_suffix(output_dir / "svd100_tsne_2d_input.png", selected_k),
        selected_k,
        f"Inventor Skill Input Space in SVD({svd_n_components}) + t-SNE 2D (colored by dominant archetype)",
        "t-SNE component 1",
        "t-SNE component 2",
    )

    tsne_svd_3d = TSNE(
        n_components=3,
        perplexity=perplexity_svd_tsne,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb_svd_tsne_all_3d = tsne_svd_3d.fit_transform(joint_svd)
    emb_svd_tsne_3d, emb_svd_tsne_arch_3d = _split_joint_embedding(emb_svd_tsne_all_3d, n_sample)
    _plot_embedding_3d(
        emb_svd_tsne_3d,
        emb_svd_tsne_arch_3d,
        dominant_sample,
        _with_k_suffix(output_dir / "svd100_tsne_3d_input.png", selected_k),
        selected_k,
        f"Inventor Skill Input Space in SVD({svd_n_components}) + t-SNE 3D (colored by dominant archetype)",
        "t-SNE c1",
        "t-SNE c2",
        "t-SNE c3",
    )

    # -----------------------------
    # Summary plots / exports
    # -----------------------------
    _plot_archetype_counts(
        dominant,
        _with_k_suffix(output_dir / "dominant_archetype_counts.png", selected_k),
        selected_k,
    )

    sim = _cosine_similarity_rows(archetypes)
    _plot_archetype_similarity_heatmap(
        sim,
        _with_k_suffix(output_dir / "archetype_cosine_similarity_heatmap.png", selected_k),
    )

    _save_top_features_csv(
        archetypes=archetypes,
        feature_names=feature_names,
        out_path=_with_k_suffix(output_dir / "archetype_top_features.csv", selected_k),
        top_n=25,
    )

    run_meta = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "run_id": run.get("run_id"),
        "run_dir": str(run_dir),
        "selected_k": selected_k,
        "inventors_total": int(memberships.shape[0]),
        "features_total": int(archetypes.shape[1]),
        "inventors_used_for_embedding": int(n_sample),
        "joint_embedding_points": int(n_joint),
        "embedding_input_file": str(run_dir / "inventor_skill_matrix.csv.gz"),
        "embedding_basis": "run-specific inventor input matrix",
        "dominant_coloring_file": str(run_dir / "memberships.csv.gz"),
        "dominant_coloring_basis": "AA memberships argmax (equivalent to normalized coefficients argmax)",
        "archetype_marker_basis": "AA archetype vectors jointly embedded with inventor inputs",
        "tsne_perplexity": float(perplexity),
        "pca_explained_variance_ratio_2d": [float(v) for v in pca_2d.explained_variance_ratio_],
        "pca_explained_variance_ratio_3d": [float(v) for v in pca_3d.explained_variance_ratio_],
        "isomap_n_neighbors": int(n_neighbors),
        "umap_metric": "cosine",
        "umap_n_neighbors": 30,
        "umap_min_dist": 0.10,
        "svd_n_components_for_sparse_plots": int(svd_n_components),
        "svd_explained_variance_ratio_sum": float(np.sum(svd_100.explained_variance_ratio_)),
        "random_state": int(random_state),
        "membership_simplex_basis": "AA memberships projected with archetypes.visualization.simplex.simplex",
        "purity_definition": "max normalized AA membership per inventor",
        "purity_stats_files": {
            "overall": str(_with_k_suffix(output_dir / "purity_stats_overall.csv", selected_k)),
            "by_archetype": str(_with_k_suffix(output_dir / "purity_stats_by_archetype.csv", selected_k)),
        },
    }
    with _with_k_suffix(output_dir / "analysis_metadata.json", selected_k).open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True, ensure_ascii=True)

    sampled_inventors = pd.DataFrame(
        {
            "inventor": inventor_index[sample_idx].astype(str),
            "dominant_archetype": dominant_sample,
            "tsne2_x": emb2[:, 0],
            "tsne2_y": emb2[:, 1],
            "tsne3_x": emb3[:, 0],
            "tsne3_y": emb3[:, 1],
            "tsne3_z": emb3[:, 2],
            "pca2_x": emb_pca2[:, 0],
            "pca2_y": emb_pca2[:, 1],
            "pca3_x": emb_pca3[:, 0],
            "pca3_y": emb_pca3[:, 1],
            "pca3_z": emb_pca3[:, 2],
            "isomap2_x": emb_iso2[:, 0],
            "isomap2_y": emb_iso2[:, 1],
            "isomap3_x": emb_iso3[:, 0],
            "isomap3_y": emb_iso3[:, 1],
            "isomap3_z": emb_iso3[:, 2],
            "umap_raw_2d_x": emb_umap_raw_2d[:, 0],
            "umap_raw_2d_y": emb_umap_raw_2d[:, 1],
            "umap_raw_3d_x": emb_umap_raw_3d[:, 0],
            "umap_raw_3d_y": emb_umap_raw_3d[:, 1],
            "umap_raw_3d_z": emb_umap_raw_3d[:, 2],
            "svd_umap_2d_x": emb_umap_svd_2d[:, 0],
            "svd_umap_2d_y": emb_umap_svd_2d[:, 1],
            "svd_umap_3d_x": emb_umap_svd_3d[:, 0],
            "svd_umap_3d_y": emb_umap_svd_3d[:, 1],
            "svd_umap_3d_z": emb_umap_svd_3d[:, 2],
            "svd_tsne_2d_x": emb_svd_tsne_2d[:, 0],
            "svd_tsne_2d_y": emb_svd_tsne_2d[:, 1],
            "svd_tsne_3d_x": emb_svd_tsne_3d[:, 0],
            "svd_tsne_3d_y": emb_svd_tsne_3d[:, 1],
            "svd_tsne_3d_z": emb_svd_tsne_3d[:, 2],
        }
    )
    sampled_inventors.to_csv(
        _with_k_suffix(output_dir / "inventor_embedding_coordinates_sampled.csv", selected_k),
        index=False,
    )

    archetype_points = pd.DataFrame(
        {
            "archetype": [f"A{i}" for i in range(1, selected_k + 1)],
            "tsne2_x": emb2_arch[:, 0],
            "tsne2_y": emb2_arch[:, 1],
            "tsne3_x": emb3_arch[:, 0],
            "tsne3_y": emb3_arch[:, 1],
            "tsne3_z": emb3_arch[:, 2],
            "pca2_x": emb_pca2_arch[:, 0],
            "pca2_y": emb_pca2_arch[:, 1],
            "pca3_x": emb_pca3_arch[:, 0],
            "pca3_y": emb_pca3_arch[:, 1],
            "pca3_z": emb_pca3_arch[:, 2],
            "isomap2_x": emb_iso2_arch[:, 0],
            "isomap2_y": emb_iso2_arch[:, 1],
            "isomap3_x": emb_iso3_arch[:, 0],
            "isomap3_y": emb_iso3_arch[:, 1],
            "isomap3_z": emb_iso3_arch[:, 2],
            "umap_raw_2d_x": emb_umap_raw_arch_2d[:, 0],
            "umap_raw_2d_y": emb_umap_raw_arch_2d[:, 1],
            "umap_raw_3d_x": emb_umap_raw_arch_3d[:, 0],
            "umap_raw_3d_y": emb_umap_raw_arch_3d[:, 1],
            "umap_raw_3d_z": emb_umap_raw_arch_3d[:, 2],
            "svd_umap_2d_x": emb_umap_svd_arch_2d[:, 0],
            "svd_umap_2d_y": emb_umap_svd_arch_2d[:, 1],
            "svd_umap_3d_x": emb_umap_svd_arch_3d[:, 0],
            "svd_umap_3d_y": emb_umap_svd_arch_3d[:, 1],
            "svd_umap_3d_z": emb_umap_svd_arch_3d[:, 2],
            "svd_tsne_2d_x": emb_svd_tsne_arch_2d[:, 0],
            "svd_tsne_2d_y": emb_svd_tsne_arch_2d[:, 1],
            "svd_tsne_3d_x": emb_svd_tsne_arch_3d[:, 0],
            "svd_tsne_3d_y": emb_svd_tsne_arch_3d[:, 1],
            "svd_tsne_3d_z": emb_svd_tsne_arch_3d[:, 2],
        }
    )
    archetype_points.to_csv(
        _with_k_suffix(output_dir / "archetype_embedding_coordinates.csv", selected_k),
        index=False,
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical analysis for archetype run outputs.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_inventors_tsne", type=int, default=6000)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out = run_analysis(
        base_dir=args.base_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        max_inventors_tsne=int(args.max_inventors_tsne),
        random_state=int(args.random_state),
    )
    print(f"[INFO] Archetype optical analysis outputs: {out}")


if __name__ == "__main__":
    main()