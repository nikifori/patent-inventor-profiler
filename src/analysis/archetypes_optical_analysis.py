#!/usr/bin/env python3
"""
Optical analysis for archetypes from the best run in:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v3

Outputs are written to:
  /home/nikifori/Desktop/thesis/repo/output/all_data_dedupled_tfidf_v3/archetypes_plots

Main visuals:
- t-SNE 2D of inventor coefficients (colored by dominant archetype)
- t-SNE 3D of inventor coefficients
- Dominant archetype counts bar chart
- Archetype cosine-similarity heatmap

Also writes:
- top features per archetype CSV
- run metadata JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

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
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "archetypes_plots"


def _load_latest_auto_run(index_jsonl: Path) -> Dict:
    if not index_jsonl.exists():
        raise FileNotFoundError(f"Missing run index: {index_jsonl}")
    with index_jsonl.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in reversed(rows):
        if row.get("run_mode") == "auto":
            return row
    raise RuntimeError(f"No auto run found in {index_jsonl}")


def _load_run_artifacts(run_dir: Path, selected_k: int) -> Tuple[np.ndarray, np.ndarray, pd.Index, list[str]]:
    model_path = run_dir / "models" / f"k_{selected_k}.npz"
    matrix_path = run_dir / "inventor_skill_matrix.csv.gz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing inventor matrix artifact: {matrix_path}")

    model = np.load(model_path)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    archetypes = np.asarray(model["archetypes"], dtype=float)

    header = pd.read_csv(matrix_path, nrows=0).columns.tolist()
    feature_names = header[1:]
    inventor_index = pd.read_csv(matrix_path, usecols=[0]).iloc[:, 0]

    if coefficients.shape[0] != len(inventor_index):
        raise RuntimeError(
            "Mismatch between coefficient rows and inventor index rows: "
            f"{coefficients.shape[0]} vs {len(inventor_index)}"
        )
    if archetypes.shape[1] != len(feature_names):
        raise RuntimeError(
            "Mismatch between archetype features and matrix columns: "
            f"{archetypes.shape[1]} vs {len(feature_names)}"
        )

    return coefficients, archetypes, inventor_index, feature_names


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


def _plot_tsne_2d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    cmap = plt.get_cmap("tab20", k)
    fig, ax = plt.subplots(figsize=(12, 9))
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.58,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(center[0], center[1], marker="X", s=180, color=cmap(arch_id - 1), edgecolor="black", linewidth=0.8)
        ax.text(center[0], center[1], f"A{arch_id}", fontsize=10, fontweight="bold", va="bottom")

    ax.set_title("Inventor Space in t-SNE 2D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pca_2d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    cmap = plt.get_cmap("tab20", k)
    fig, ax = plt.subplots(figsize=(12, 9))
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.58,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(center[0], center[1], marker="X", s=180, color=cmap(arch_id - 1), edgecolor="black", linewidth=0.8)
        ax.text(center[0], center[1], f"A{arch_id}", fontsize=10, fontweight="bold", va="bottom")

    ax.set_title("Inventor Space in PCA 2D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_tsne_3d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = plt.get_cmap("tab20", k)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=10,
            alpha=0.45,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(
            center[0],
            center[1],
            center[2],
            marker="X",
            s=220,
            color=cmap(arch_id - 1),
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_title("Inventor Space in t-SNE 3D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("t-SNE c1")
    ax.set_ylabel("t-SNE c2")
    ax.set_zlabel("t-SNE c3")
    ax.legend(loc="upper left", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pca_3d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = plt.get_cmap("tab20", k)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=10,
            alpha=0.45,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(
            center[0],
            center[1],
            center[2],
            marker="X",
            s=220,
            color=cmap(arch_id - 1),
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_title("Inventor Space in PCA 3D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("PCA c1")
    ax.set_ylabel("PCA c2")
    ax.set_zlabel("PCA c3")
    ax.legend(loc="upper left", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_isomap_2d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    cmap = plt.get_cmap("tab20", k)
    fig, ax = plt.subplots(figsize=(12, 9))
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            alpha=0.58,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(center[0], center[1], marker="X", s=180, color=cmap(arch_id - 1), edgecolor="black", linewidth=0.8)
        ax.text(center[0], center[1], f"A{arch_id}", fontsize=10, fontweight="bold", va="bottom")

    ax.set_title("Inventor Space in Isomap 2D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("Isomap component 1")
    ax.set_ylabel("Isomap component 2")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_isomap_3d(embedding: np.ndarray, dominant: np.ndarray, out_path: Path, k: int) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = plt.get_cmap("tab20", k)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for arch_id in range(1, k + 1):
        mask = dominant == arch_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=10,
            alpha=0.45,
            color=cmap(arch_id - 1),
            label=f"A{arch_id}",
        )
        center = embedding[mask].mean(axis=0)
        ax.scatter(
            center[0],
            center[1],
            center[2],
            marker="X",
            s=220,
            color=cmap(arch_id - 1),
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_title("Inventor Space in Isomap 3D (by dominant archetype)", pad=12, fontweight="bold")
    ax.set_xlabel("Isomap c1")
    ax.set_ylabel("Isomap c2")
    ax.set_zlabel("Isomap c3")
    ax.legend(loc="upper left", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_archetype_counts(dominant: np.ndarray, out_path: Path, k: int) -> None:
    counts = pd.Series(dominant).value_counts().reindex(range(1, k + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar([f"A{i}" for i in counts.index], counts.values, color="#457b9d", edgecolor="#ffffff", linewidth=0.9)
    ax.set_title("Inventors per Dominant Archetype", pad=12, fontweight="bold")
    ax.set_xlabel("Archetype")
    ax.set_ylabel("Inventor count")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{int(val)}", ha="center", va="bottom", fontsize=10)
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


def run_analysis(base_dir: Path, output_dir: Path, max_inventors_tsne: int, random_state: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run = _load_latest_auto_run(base_dir / "archetype_runs" / "index.jsonl")
    run_dir = Path(run["run_dir"])
    selected_k = int(run["selected_k"])

    coefficients, archetypes, inventor_index, feature_names = _load_run_artifacts(run_dir, selected_k)

    dominant = np.argmax(coefficients, axis=1) + 1
    sample_idx = _sample_indices_by_group(dominant, max_inventors_tsne, random_state)
    coeff_sample = coefficients[sample_idx]
    dominant_sample = dominant[sample_idx]

    n_sample = len(sample_idx)
    perplexity = min(35, max(5, n_sample // 40))
    if perplexity >= n_sample:
        perplexity = max(2, n_sample - 1)

    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb2 = tsne_2d.fit_transform(coeff_sample)
    _plot_tsne_2d(emb2, dominant_sample, output_dir / "tsne_2d_inventor_coefficients.png", selected_k)

    pca_2d = PCA(n_components=2, random_state=random_state)
    emb_pca2 = pca_2d.fit_transform(coeff_sample)
    _plot_pca_2d(emb_pca2, dominant_sample, output_dir / "pca_2d_inventor_coefficients.png", selected_k)

    pca_3d = PCA(n_components=3, random_state=random_state)
    emb_pca3 = pca_3d.fit_transform(coeff_sample)
    _plot_pca_3d(emb_pca3, dominant_sample, output_dir / "pca_3d_inventor_coefficients.png", selected_k)

    n_neighbors = min(15, max(2, n_sample - 1))
    isomap_2d = Isomap(n_components=2, n_neighbors=n_neighbors)
    emb_iso2 = isomap_2d.fit_transform(coeff_sample)
    _plot_isomap_2d(emb_iso2, dominant_sample, output_dir / "isomap_2d_inventor_coefficients.png", selected_k)

    isomap_3d = Isomap(n_components=3, n_neighbors=n_neighbors)
    emb_iso3 = isomap_3d.fit_transform(coeff_sample)
    _plot_isomap_3d(emb_iso3, dominant_sample, output_dir / "isomap_3d_inventor_coefficients.png", selected_k)

    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb3 = tsne_3d.fit_transform(coeff_sample)
    _plot_tsne_3d(emb3, dominant_sample, output_dir / "tsne_3d_inventor_coefficients.png", selected_k)

    _plot_archetype_counts(dominant, output_dir / "dominant_archetype_counts.png", selected_k)

    sim = _cosine_similarity_rows(archetypes)
    _plot_archetype_similarity_heatmap(sim, output_dir / "archetype_cosine_similarity_heatmap.png")

    _save_top_features_csv(
        archetypes=archetypes,
        feature_names=feature_names,
        out_path=output_dir / "archetype_top_features.csv",
        top_n=25,
    )

    run_meta = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "run_id": run.get("run_id"),
        "run_dir": str(run_dir),
        "selected_k": selected_k,
        "inventors_total": int(coefficients.shape[0]),
        "features_total": int(archetypes.shape[1]),
        "inventors_used_for_tsne": int(n_sample),
        "tsne_perplexity": float(perplexity),
        "pca_explained_variance_ratio_2d": [float(v) for v in pca_2d.explained_variance_ratio_],
        "pca_explained_variance_ratio_3d": [float(v) for v in pca_3d.explained_variance_ratio_],
        "isomap_n_neighbors": int(n_neighbors),
        "random_state": int(random_state),
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True, ensure_ascii=True)

    sampled_inventors = pd.DataFrame(
        {
            "inventor": inventor_index.iloc[sample_idx].astype(str).values,
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
        }
    )
    sampled_inventors.to_csv(output_dir / "inventor_tsne_coordinates_sampled.csv", index=False)

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
