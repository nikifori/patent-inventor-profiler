'''
@File    :   pipeline_utils.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
'''
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional, Sequence
import json
import os
import numpy as np
import platform
import hashlib
import sys
import time
import types
import importlib
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import sparse
from archetypes.visualization.simplex import simplex
import umap


class Link2Skill_Mapping:
    """
    Class to handle mapping between links and skills.
    """
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = os.path.abspath(__file__) + '/../../../resources/ESCO_link2skill_mapping.csv'

        csv_path = Path(csv_path).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        df = pd.read_csv(csv_path, delimiter=';')
        df["conceptUri"] = df["conceptUri"].str.strip()
        df["preferredLabel"] = df["preferredLabel"].str.strip()
        self._link2skill: Dict[str, str] = dict(df[["conceptUri", "preferredLabel"]].itertuples(index=False, name=None))
    
    def link2skill(self, link: str = None) -> str:
        """
        Return the skill label for a concept URI.

        Raises
        ------
        KeyError: if link is not in the mapping.
        """
        if not link:
            raise ValueError("Link must be a non-empty string")
        try:
            return self._link2skill[link]
        except KeyError:
            raise KeyError(f"Link not found in mapping: {link}")


def load_data(data_path: str = None) -> Dict[str, Any]:
    """
    Load data from a .csv or .jsonl file and return it as a dictionary.

    Parameters:
    ----------
    data_path : str
        Path to the input file. Must end with .csv or .jsonl.

    Returns:
    -------
    Dict[str, Any]
        Dictionary representation of the data (as returned by pandas.DataFrame.to_dict()).

    Raises:
    ------
    ValueError
        If data_path is not provided or the file extension is unsupported.
    """
    if data_path is None:
        raise ValueError("data_path must be provided")

    path = Path(data_path)
    suffix = path.suffix.lower()

    match suffix:
        case '.csv':
            # Read CSV with comma delimiter
            df = pd.read_csv(path, delimiter=',')
            return df.to_dict(orient='records')
        case '.jsonl':
            # Read JSON Lines and return list of objects, handling UTF-8 BOM
            records = []
            with path.open('r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        case _:
            raise ValueError(f"Unsupported file extension: {suffix}")


def mean_or_zero(vals: Optional[List[float]]) -> float:
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


def _json_default_serializer(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=_json_default_serializer)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _safe_package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _collect_environment_versions() -> Dict[str, Optional[str]]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "archetypes": _safe_package_version("archetypes"),
        "matplotlib": _safe_package_version("matplotlib"),
        "kneed": _safe_package_version("kneed"),
        "scikit-learn": _safe_package_version("scikit-learn"),
        "torch": _safe_package_version("torch"),
        "jax": _safe_package_version("jax"),
    }


def _to_numpy_array(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if value is None:
        arr = np.array([])
    else:
        obj = value
        if hasattr(obj, "detach"):
            obj = obj.detach()
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "numpy"):
            try:
                obj = obj.numpy()
            except Exception:
                pass
        arr = np.asarray(obj)
    if dtype is not None:
        try:
            arr = arr.astype(dtype, copy=False)
        except Exception:
            pass
    return arr


def _load_installed_torch_archetypal_analysis():
    """
    Load `archetypal_analysis` from installed editable `archetypes` package.
    Returns (callable, module_path).
    """
    try:
        torch_mod = importlib.import_module("archetypes.torch")
        fn = getattr(torch_mod, "archetypal_analysis", None)
        if fn is not None:
            return fn, "archetypes.torch"
    except Exception:
        pass

    for module_name in ("archetypes.torch._AA", "archetypes.torch._aa"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, "archetypal_analysis", None)
        if fn is not None:
            return fn, module_name
    raise ImportError(
        "Could not import `archetypal_analysis` from installed archetypes package. "
        "Tried: archetypes.torch._AA, archetypes.torch._aa"
    )


def _fingerprint_inventor_skill_df(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"shape:{df.shape}".encode("utf-8"))
    hasher.update(f"dtypes:{','.join(map(str, df.dtypes.tolist()))}".encode("utf-8"))
    for idx_val in df.index:
        hasher.update(str(idx_val).encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    hasher.update(b"__COLUMNS__\n")
    for col in df.columns:
        hasher.update(str(col).encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    values = _to_numpy_array(df.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    hasher.update(values.tobytes(order="C"))
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default_serializer)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(_stable_json_dumps(payload))
        f.write("\n")


def _save_model_npz(path: Path, model: Any, rss: float, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=_to_numpy_array(getattr(model, "coefficients_", None), dtype=np.float64),
        archetypes=_to_numpy_array(getattr(model, "archetypes_", None), dtype=np.float64),
        arch_coefficients=_to_numpy_array(getattr(model, "arch_coefficients_", None), dtype=np.float64),
        labels=_to_numpy_array(getattr(model, "labels_", None)),
        loss=_to_numpy_array(getattr(model, "loss_", None), dtype=np.float64),
        rss=np.array([float(rss)], dtype=np.float64),
        n_iter=np.array([int(getattr(model, "n_iter_", -1))], dtype=np.int64),
        k=np.array([int(k)], dtype=np.int64),
    )


def _save_elbow_plot(fig_path: Path, candidate_ks: List[int], mean_rss_per_k: List[float], best_k: Optional[int]) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(candidate_ks, mean_rss_per_k, marker="o")
    plt.xlabel("Number of archetypes (k)")
    plt.ylabel("Mean RSS")
    plt.title("Elbow plot for AA (k vs mean RSS)")
    if best_k is not None and best_k in candidate_ks:
        best_idx = candidate_ks.index(best_k)
        plt.scatter(
            [candidate_ks[best_idx]],
            [mean_rss_per_k[best_idx]],
            marker="x",
            s=100,
        )
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


_ARCHETYPE_PLOT_STYLE = {
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


def _aa_with_k_suffix(path: Path, selected_k: int) -> Path:
    return path.with_name(f"{path.stem}_k{selected_k}{path.suffix}")


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


def _pad_embedding(embedding: np.ndarray, target_dims: int) -> np.ndarray:
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embedding.shape}")
    if embedding.shape[1] >= target_dims:
        return embedding[:, :target_dims]
    pad_width = target_dims - embedding.shape[1]
    return np.pad(embedding, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def _compute_tsne_embedding(
    coefficients: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    n_sample = coefficients.shape[0]
    if n_sample < 2:
        return None, None

    perplexity_floor = max(5, n_sample // 40)
    perplexity = float(min(35, max(1, n_sample - 1), perplexity_floor))
    embedding = TSNE(
        n_components=int(n_components),
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=int(random_state),
    ).fit_transform(coefficients)
    return embedding, perplexity


def _compute_pca_embedding(
    coefficients: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, List[float]]:
    max_components = max(1, min(int(n_components), coefficients.shape[0], coefficients.shape[1]))
    pca = PCA(n_components=max_components, random_state=int(random_state))
    embedding = pca.fit_transform(coefficients)
    explained = [float(v) for v in pca.explained_variance_ratio_]
    if len(explained) < n_components:
        explained.extend([0.0] * (n_components - len(explained)))
    return _pad_embedding(embedding, n_components), explained[:n_components]


def _compute_isomap_embedding(
    coefficients: np.ndarray,
    n_components: int,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    n_sample = coefficients.shape[0]
    if n_sample < 2:
        return None, None

    n_neighbors = max(1, min(15, n_sample - 1))
    max_components = max(1, min(int(n_components), n_sample - 1, coefficients.shape[1]))
    embedding = Isomap(n_components=max_components, n_neighbors=n_neighbors).fit_transform(coefficients)
    return _pad_embedding(embedding, n_components), n_neighbors


def _plot_embedding_2d(
    embedding: np.ndarray,
    dominant: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
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
            ax.scatter(
                center[0],
                center[1],
                marker="X",
                s=180,
                color=cmap(arch_id - 1),
                edgecolor="black",
                linewidth=0.8,
            )
            ax.text(center[0], center[1], f"A{arch_id}", fontsize=10, fontweight="bold", va="bottom")

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
    embedding: np.ndarray,
    dominant: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
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

        ax.set_title(title, pad=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.legend(loc="upper left", ncols=2, frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _plot_archetype_counts(dominant: np.ndarray, out_path: Path, k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = pd.Series(dominant).value_counts().reindex(range(1, k + 1), fill_value=0)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
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


def _save_top_features_csv(
    archetypes: np.ndarray,
    feature_names: List[str],
    out_path: Path,
    top_n: int = 25,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(archetypes.shape[0]):
        arch_label = f"Archetype_{i+1}"
        scores = pd.Series(archetypes[i], index=feature_names)
        top = scores.sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(top.items(), start=1):
            rows.append(
                {
                    "archetype": arch_label,
                    "rank": rank,
                    "feature": feature,
                    "weight": float(value),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _draft_archetype_category(top_feature: str) -> str:
    top_feature_l = top_feature.lower()
    if "wind" in top_feature_l:
        return "Wind"
    if "photovoltaic" in top_feature_l or "solar" in top_feature_l:
        return "Solar"
    if "vehicle" in top_feature_l or "battery" in top_feature_l:
        return "EV / Vehicle / Storage"
    if "system" in top_feature_l or "control" in top_feature_l:
        return "Systems / Control"
    return "Other"


def _run_archetype_optical_analysis(
    *,
    coefficients: np.ndarray,
    archetypes: np.ndarray,
    inventor_index: pd.Index,
    feature_names: List[str],
    selected_k: int,
    output_dir: Path,
    random_state: int,
    max_inventors_tsne: int,
    metadata: Dict[str, Any],
) -> Path:
    if coefficients.ndim != 2 or archetypes.ndim != 2:
        raise ValueError("Expected 2D coefficient and archetype matrices for optical analysis.")
    if coefficients.shape[0] != len(inventor_index):
        raise ValueError(
            "Mismatch between coefficient rows and inventor index rows: "
            f"{coefficients.shape[0]} vs {len(inventor_index)}"
        )
    if archetypes.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between archetype features and inventor skill columns: "
            f"{archetypes.shape[1]} vs {len(feature_names)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dominant = np.argmax(coefficients, axis=1) + 1
    sample_idx = _sample_indices_by_group(dominant, max_inventors_tsne, random_state)
    coeff_sample = coefficients[sample_idx]
    dominant_sample = dominant[sample_idx]

    tsne_2d = None
    tsne_3d = None
    tsne_perplexity = None
    try:
        tsne_2d, tsne_perplexity = _compute_tsne_embedding(coeff_sample, n_components=2, random_state=random_state)
        if tsne_2d is not None:
            _plot_embedding_2d(
                tsne_2d,
                dominant_sample,
                output_dir / "tsne_2d_inventor_coefficients.png",
                selected_k,
                "Inventor Space in t-SNE 2D (by dominant archetype)",
                "t-SNE component 1",
                "t-SNE component 2",
            )
    except Exception as exc:
        print(f"[WARN] Failed to generate t-SNE 2D archetype plot: {exc}")

    try:
        tsne_3d, tsne_perplexity_3d = _compute_tsne_embedding(coeff_sample, n_components=3, random_state=random_state)
        if tsne_3d is not None:
            _plot_embedding_3d(
                tsne_3d,
                dominant_sample,
                output_dir / "tsne_3d_inventor_coefficients.png",
                selected_k,
                "Inventor Space in t-SNE 3D (by dominant archetype)",
                "t-SNE c1",
                "t-SNE c2",
                "t-SNE c3",
            )
            if tsne_perplexity is None:
                tsne_perplexity = tsne_perplexity_3d
    except Exception as exc:
        print(f"[WARN] Failed to generate t-SNE 3D archetype plot: {exc}")

    pca_2d, pca_2d_evr = _compute_pca_embedding(coeff_sample, n_components=2, random_state=random_state)
    _plot_embedding_2d(
        pca_2d,
        dominant_sample,
        output_dir / "pca_2d_inventor_coefficients.png",
        selected_k,
        "Inventor Space in PCA 2D (by dominant archetype)",
        "PCA component 1",
        "PCA component 2",
    )

    pca_3d, pca_3d_evr = _compute_pca_embedding(coeff_sample, n_components=3, random_state=random_state)
    _plot_embedding_3d(
        pca_3d,
        dominant_sample,
        output_dir / "pca_3d_inventor_coefficients.png",
        selected_k,
        "Inventor Space in PCA 3D (by dominant archetype)",
        "PCA c1",
        "PCA c2",
        "PCA c3",
    )

    isomap_2d = None
    isomap_3d = None
    isomap_n_neighbors = None
    try:
        isomap_2d, isomap_n_neighbors = _compute_isomap_embedding(coeff_sample, n_components=2)
        if isomap_2d is not None:
            _plot_embedding_2d(
                isomap_2d,
                dominant_sample,
                output_dir / "isomap_2d_inventor_coefficients.png",
                selected_k,
                "Inventor Space in Isomap 2D (by dominant archetype)",
                "Isomap component 1",
                "Isomap component 2",
            )
    except Exception as exc:
        print(f"[WARN] Failed to generate Isomap 2D archetype plot: {exc}")

    try:
        isomap_3d, isomap_n_neighbors_3d = _compute_isomap_embedding(coeff_sample, n_components=3)
        if isomap_3d is not None:
            _plot_embedding_3d(
                isomap_3d,
                dominant_sample,
                output_dir / "isomap_3d_inventor_coefficients.png",
                selected_k,
                "Inventor Space in Isomap 3D (by dominant archetype)",
                "Isomap c1",
                "Isomap c2",
                "Isomap c3",
            )
            if isomap_n_neighbors is None:
                isomap_n_neighbors = isomap_n_neighbors_3d
    except Exception as exc:
        print(f"[WARN] Failed to generate Isomap 3D archetype plot: {exc}")

    _plot_archetype_counts(dominant, output_dir / "dominant_archetype_counts.png", selected_k)
    _plot_archetype_similarity_heatmap(
        _cosine_similarity_rows(archetypes),
        output_dir / "archetype_cosine_similarity_heatmap.png",
    )
    _save_top_features_csv(
        archetypes=archetypes,
        feature_names=feature_names,
        out_path=output_dir / "archetype_top_features.csv",
        top_n=25,
    )

    sampled_inventors = pd.DataFrame(
        {
            "inventor": inventor_index[sample_idx].astype(str),
            "dominant_archetype": dominant_sample,
            "tsne2_x": tsne_2d[:, 0] if tsne_2d is not None else np.nan,
            "tsne2_y": tsne_2d[:, 1] if tsne_2d is not None else np.nan,
            "tsne3_x": tsne_3d[:, 0] if tsne_3d is not None else np.nan,
            "tsne3_y": tsne_3d[:, 1] if tsne_3d is not None else np.nan,
            "tsne3_z": tsne_3d[:, 2] if tsne_3d is not None else np.nan,
            "pca2_x": pca_2d[:, 0],
            "pca2_y": pca_2d[:, 1],
            "pca3_x": pca_3d[:, 0],
            "pca3_y": pca_3d[:, 1],
            "pca3_z": pca_3d[:, 2],
            "isomap2_x": isomap_2d[:, 0] if isomap_2d is not None else np.nan,
            "isomap2_y": isomap_2d[:, 1] if isomap_2d is not None else np.nan,
            "isomap3_x": isomap_3d[:, 0] if isomap_3d is not None else np.nan,
            "isomap3_y": isomap_3d[:, 1] if isomap_3d is not None else np.nan,
            "isomap3_z": isomap_3d[:, 2] if isomap_3d is not None else np.nan,
        }
    )
    sampled_inventors.to_csv(output_dir / "inventor_tsne_coordinates_sampled.csv", index=False)

    analysis_metadata = {
        **metadata,
        "output_dir": str(output_dir),
        "selected_k": int(selected_k),
        "inventors_total": int(coefficients.shape[0]),
        "features_total": int(archetypes.shape[1]),
        "inventors_used_for_embeddings": int(len(sample_idx)),
        "tsne_perplexity": float(tsne_perplexity) if tsne_perplexity is not None else None,
        "pca_explained_variance_ratio_2d": pca_2d_evr,
        "pca_explained_variance_ratio_3d": pca_3d_evr,
        "isomap_n_neighbors": int(isomap_n_neighbors) if isomap_n_neighbors is not None else None,
        "random_state": int(random_state),
        "max_inventors_tsne": int(max_inventors_tsne),
    }
    _write_json(output_dir / "analysis_metadata.json", analysis_metadata)
    return output_dir


def _aa_split_joint_embedding(embedding: np.ndarray, n_inventors: int) -> Tuple[np.ndarray, np.ndarray]:
    return embedding[:n_inventors], embedding[n_inventors:]


def _aa_stack_sparse_dense_for_joint_embedding(inventor_sparse: Any, archetypes: np.ndarray) -> Any:
    return sparse.vstack(
        [
            inventor_sparse,
            sparse.csr_matrix(np.asarray(archetypes, dtype=float)),
        ],
        format="csr",
    )


def _aa_plot_membership_simplex(
    memberships: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    max_points: int = 40000,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
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
            ax.text(
                1.08 * np.cos(angle),
                1.08 * np.sin(angle),
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


def _aa_save_membership_simplex_vertices_csv(out_path: Path, k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pd.DataFrame(
        {
            "archetype": [f"A{i}" for i in range(1, k + 1)],
            "x": np.cos(theta),
            "y": np.sin(theta),
        }
    ).to_csv(out_path, index=False)


def _aa_fit_umap(matrix: Any, n_components: int, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.10,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def _aa_safe_svd_components(n_rows: int, n_cols: int, target: int = 100) -> int:
    return max(2, min(int(target), n_rows - 1, n_cols - 1))


def _aa_compute_purity(memberships: np.ndarray) -> np.ndarray:
    memberships = np.asarray(memberships, dtype=float)
    row_sums = memberships.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    memberships = memberships / safe_row_sums
    return np.max(memberships, axis=1)


def _aa_purity_stats_dict(values: np.ndarray) -> Dict[str, Any]:
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
        values,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
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


def _aa_save_purity_stats_overall(purity: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_aa_purity_stats_dict(purity)]).to_csv(out_path, index=False)


def _aa_save_purity_stats_by_archetype(
    purity: np.ndarray,
    dominant: np.ndarray,
    k: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch_id in range(1, k + 1):
        stats = _aa_purity_stats_dict(purity[dominant == arch_id])
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


def _aa_plot_purity_histogram(
    purity: np.ndarray,
    out_path: Path,
    title: str,
    bins: int = 30,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        counts, bin_edges, _ = ax.hist(
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

        if len(counts) > 0 and max(counts) > 0:
            for c, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
                if c > 0:
                    ax.text((left + right) / 2.0, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

        ax.set_xlim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _aa_plot_purity_histogram_by_archetype(
    purity: np.ndarray,
    dominant: np.ndarray,
    k: int,
    out_dir: Path,
    bins: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
        for arch_id in range(1, k + 1):
            vals = purity[dominant == arch_id]
            fig, ax = plt.subplots(figsize=(11, 6.5))
            counts, bin_edges, _ = ax.hist(
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
                    ax.text((left + right) / 2.0, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)

            ax.set_xlim(0.0, 1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best", frameon=True, facecolor="#ffffff", edgecolor="#dbe3ee")
            fig.tight_layout()
            fig.savefig(out_dir / f"purity_histogram_A{arch_id}.png", dpi=180)
            plt.close(fig)


def _aa_plot_embedding_2d(
    inventor_embedding: np.ndarray,
    archetype_embedding: np.ndarray,
    dominant: np.ndarray,
    out_path: Path,
    k: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
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


def _aa_plot_embedding_3d(
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_ARCHETYPE_PLOT_STYLE):
        cmap = plt.get_cmap("tab20", max(k, 1))
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
                s=220,
                alpha=0.8,
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


def _aa_embedding_column(embedding: Optional[np.ndarray], axis: int, length: int) -> np.ndarray:
    if embedding is None:
        return np.full(length, np.nan)
    return embedding[:, axis]


def _run_archetype_optical_analysis_current(
    *,
    inventor_skill_df: pd.DataFrame,
    memberships_df: pd.DataFrame,
    archetypes: np.ndarray,
    inventor_index: pd.Index,
    feature_names: List[str],
    selected_k: int,
    output_dir: Path,
    random_state: int,
    max_inventors_tsne: int,
    metadata: Dict[str, Any],
) -> Path:
    memberships = memberships_df.to_numpy(dtype=float, copy=False)
    if memberships.ndim != 2 or archetypes.ndim != 2:
        raise ValueError("Expected 2D membership and archetype matrices for optical analysis.")
    if memberships.shape[0] != len(inventor_index):
        raise ValueError(
            "Mismatch between membership rows and inventor index rows: "
            f"{memberships.shape[0]} vs {len(inventor_index)}"
        )
    if inventor_skill_df.shape[0] != len(inventor_index):
        raise ValueError(
            "Mismatch between inventor skill rows and inventor index rows: "
            f"{inventor_skill_df.shape[0]} vs {len(inventor_index)}"
        )
    if archetypes.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between archetype features and inventor skill columns: "
            f"{archetypes.shape[1]} vs {len(feature_names)}"
        )
    if inventor_skill_df.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between inventor skill columns and feature names: "
            f"{inventor_skill_df.shape[1]} vs {len(feature_names)}"
        )
    if memberships.shape[1] != selected_k:
        raise ValueError(
            "Mismatch between membership columns and selected_k: "
            f"{memberships.shape[1]} vs {selected_k}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    inventor_input = sparse.csr_matrix(inventor_skill_df.to_numpy(dtype=float, copy=False))
    dominant = np.argmax(memberships, axis=1) + 1
    sample_idx = _sample_indices_by_group(dominant, max_inventors_tsne, random_state)
    dominant_sample = dominant[sample_idx]
    purity = _aa_compute_purity(memberships)
    input_sample = inventor_input[sample_idx]
    joint_matrix = np.vstack([input_sample.toarray(), archetypes])
    joint_sparse = _aa_stack_sparse_dense_for_joint_embedding(input_sample, archetypes)
    n_sample = input_sample.shape[0]
    n_joint = joint_matrix.shape[0]

    perplexity = min(35, max(5, n_joint // 40))
    if perplexity >= n_joint:
        perplexity = max(1, n_joint - 1)
    n_neighbors = min(15, max(2, n_joint - 1))

    _aa_plot_purity_histogram(
        purity=purity,
        out_path=_aa_with_k_suffix(output_dir / "purity_histogram_overall.png", selected_k),
        title="Purity Histogram (all inventors)",
        bins=30,
    )
    _aa_plot_purity_histogram_by_archetype(
        purity=purity,
        dominant=dominant,
        k=selected_k,
        out_dir=_aa_with_k_suffix(output_dir / "purity_histograms_by_archetype.png", selected_k),
        bins=20,
    )
    _aa_save_purity_stats_overall(
        purity=purity,
        out_path=_aa_with_k_suffix(output_dir / "purity_stats_overall.csv", selected_k),
    )
    _aa_save_purity_stats_by_archetype(
        purity=purity,
        dominant=dominant,
        k=selected_k,
        out_path=_aa_with_k_suffix(output_dir / "purity_stats_by_archetype.csv", selected_k),
    )
    try:
        _aa_plot_membership_simplex(
            memberships=memberships,
            out_path=_aa_with_k_suffix(output_dir / "membership_simplex_plot.png", selected_k),
            k=selected_k,
            title="Simplex: Colored by Dominant Archetype",
            max_points=40000,
            random_state=random_state,
        )
        _aa_save_membership_simplex_vertices_csv(
            _aa_with_k_suffix(output_dir / "membership_simplex_vertices.csv", selected_k),
            selected_k,
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate membership simplex outputs: {exc}")

    emb2 = None
    emb2_arch = None
    emb3 = None
    emb3_arch = None
    emb_pca2 = None
    emb_pca2_arch = None
    emb_pca3 = None
    emb_pca3_arch = None
    emb_iso2 = None
    emb_iso2_arch = None
    emb_iso3 = None
    emb_iso3_arch = None
    emb_umap_raw_2d = None
    emb_umap_raw_arch_2d = None
    emb_umap_raw_3d = None
    emb_umap_raw_arch_3d = None
    emb_umap_svd_2d = None
    emb_umap_svd_arch_2d = None
    emb_umap_svd_3d = None
    emb_umap_svd_arch_3d = None
    emb_svd_tsne_2d = None
    emb_svd_tsne_arch_2d = None
    emb_svd_tsne_3d = None
    emb_svd_tsne_arch_3d = None
    pca_2d_evr = None
    pca_3d_evr = None
    svd_n_components = None
    svd_explained_variance_ratio_sum = None

    try:
        emb2_all = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        ).fit_transform(joint_matrix)
        emb2, emb2_arch = _aa_split_joint_embedding(emb2_all, n_sample)
        _aa_plot_embedding_2d(
            emb2,
            emb2_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "tsne_2d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in t-SNE 2D (colored by dominant archetype)",
            "t-SNE component 1",
            "t-SNE component 2",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype t-SNE 2D plot: {exc}")

    try:
        emb3_all = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        ).fit_transform(joint_matrix)
        emb3, emb3_arch = _aa_split_joint_embedding(emb3_all, n_sample)
        _aa_plot_embedding_3d(
            emb3,
            emb3_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "tsne_3d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in t-SNE 3D (colored by dominant archetype)",
            "t-SNE c1",
            "t-SNE c2",
            "t-SNE c3",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype t-SNE 3D plot: {exc}")

    try:
        pca_2d = PCA(n_components=2, random_state=random_state)
        emb_pca2_all = pca_2d.fit_transform(joint_matrix)
        emb_pca2, emb_pca2_arch = _aa_split_joint_embedding(emb_pca2_all, n_sample)
        pca_2d_evr = [float(v) for v in pca_2d.explained_variance_ratio_]
        _aa_plot_embedding_2d(
            emb_pca2,
            emb_pca2_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "pca_2d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in PCA 2D (colored by dominant archetype)",
            "PCA component 1",
            "PCA component 2",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype PCA 2D plot: {exc}")

    try:
        pca_3d = PCA(n_components=3, random_state=random_state)
        emb_pca3_all = pca_3d.fit_transform(joint_matrix)
        emb_pca3, emb_pca3_arch = _aa_split_joint_embedding(emb_pca3_all, n_sample)
        pca_3d_evr = [float(v) for v in pca_3d.explained_variance_ratio_]
        _aa_plot_embedding_3d(
            emb_pca3,
            emb_pca3_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "pca_3d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in PCA 3D (colored by dominant archetype)",
            "PCA c1",
            "PCA c2",
            "PCA c3",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype PCA 3D plot: {exc}")

    try:
        emb_iso2_all = Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(joint_matrix)
        emb_iso2, emb_iso2_arch = _aa_split_joint_embedding(emb_iso2_all, n_sample)
        _aa_plot_embedding_2d(
            emb_iso2,
            emb_iso2_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "isomap_2d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in Isomap 2D (colored by dominant archetype)",
            "Isomap component 1",
            "Isomap component 2",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype Isomap 2D plot: {exc}")

    try:
        emb_iso3_all = Isomap(n_components=3, n_neighbors=n_neighbors).fit_transform(joint_matrix)
        emb_iso3, emb_iso3_arch = _aa_split_joint_embedding(emb_iso3_all, n_sample)
        _aa_plot_embedding_3d(
            emb_iso3,
            emb_iso3_arch,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "isomap_3d_inventor_skills.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in Isomap 3D (colored by dominant archetype)",
            "Isomap c1",
            "Isomap c2",
            "Isomap c3",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype Isomap 3D plot: {exc}")

    try:
        emb_umap_raw_all_2d = _aa_fit_umap(joint_sparse, n_components=2, random_state=random_state)
        emb_umap_raw_2d, emb_umap_raw_arch_2d = _aa_split_joint_embedding(emb_umap_raw_all_2d, n_sample)
        _aa_plot_embedding_2d(
            emb_umap_raw_2d,
            emb_umap_raw_arch_2d,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "umap_2d_raw_input.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in UMAP 2D (raw sparse TF-IDF, colored by dominant archetype)",
            "UMAP component 1",
            "UMAP component 2",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate raw-input UMAP 2D plot: {exc}")

    try:
        emb_umap_raw_all_3d = _aa_fit_umap(joint_sparse, n_components=3, random_state=random_state)
        emb_umap_raw_3d, emb_umap_raw_arch_3d = _aa_split_joint_embedding(emb_umap_raw_all_3d, n_sample)
        _aa_plot_embedding_3d(
            emb_umap_raw_3d,
            emb_umap_raw_arch_3d,
            dominant_sample,
            _aa_with_k_suffix(output_dir / "umap_3d_raw_input.png", selected_k),
            selected_k,
            "Inventor Skill Input Space in UMAP 3D (raw sparse TF-IDF, colored by dominant archetype)",
            "UMAP c1",
            "UMAP c2",
            "UMAP c3",
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate raw-input UMAP 3D plot: {exc}")

    joint_svd = None
    perplexity_svd_tsne = None
    try:
        svd_n_components = _aa_safe_svd_components(n_rows=n_joint, n_cols=joint_sparse.shape[1], target=125)
        svd_100 = TruncatedSVD(n_components=svd_n_components, random_state=random_state)
        joint_svd = svd_100.fit_transform(joint_sparse)
        svd_explained_variance_ratio_sum = float(np.sum(svd_100.explained_variance_ratio_))
    except Exception as exc:
        print(f"[WARN] Failed to compute SVD embedding basis: {exc}")

    if joint_svd is not None:
        try:
            emb_umap_svd_all_2d = _aa_fit_umap(joint_svd, n_components=2, random_state=random_state)
            emb_umap_svd_2d, emb_umap_svd_arch_2d = _aa_split_joint_embedding(emb_umap_svd_all_2d, n_sample)
            _aa_plot_embedding_2d(
                emb_umap_svd_2d,
                emb_umap_svd_arch_2d,
                dominant_sample,
                _aa_with_k_suffix(output_dir / "svd100_umap_2d_input.png", selected_k),
                selected_k,
                f"Inventor Skill Input Space in SVD({svd_n_components}) + UMAP 2D (colored by dominant archetype)",
                "UMAP component 1",
                "UMAP component 2",
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate SVD+UMAP 2D plot: {exc}")

        try:
            emb_umap_svd_all_3d = _aa_fit_umap(joint_svd, n_components=3, random_state=random_state)
            emb_umap_svd_3d, emb_umap_svd_arch_3d = _aa_split_joint_embedding(emb_umap_svd_all_3d, n_sample)
            _aa_plot_embedding_3d(
                emb_umap_svd_3d,
                emb_umap_svd_arch_3d,
                dominant_sample,
                _aa_with_k_suffix(output_dir / "svd100_umap_3d_input.png", selected_k),
                selected_k,
                f"Inventor Skill Input Space in SVD({svd_n_components}) + UMAP 3D (colored by dominant archetype)",
                "UMAP c1",
                "UMAP c2",
                "UMAP c3",
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate SVD+UMAP 3D plot: {exc}")

        perplexity_svd_tsne = min(100, max(5, n_joint // 40))
        if perplexity_svd_tsne >= n_joint:
            perplexity_svd_tsne = max(1, n_joint - 1)

        try:
            emb_svd_tsne_all_2d = TSNE(
                n_components=2,
                perplexity=perplexity_svd_tsne,
                learning_rate="auto",
                init="pca",
                random_state=random_state,
            ).fit_transform(joint_svd)
            emb_svd_tsne_2d, emb_svd_tsne_arch_2d = _aa_split_joint_embedding(emb_svd_tsne_all_2d, n_sample)
            _aa_plot_embedding_2d(
                emb_svd_tsne_2d,
                emb_svd_tsne_arch_2d,
                dominant_sample,
                _aa_with_k_suffix(output_dir / "svd100_tsne_2d_input.png", selected_k),
                selected_k,
                f"Inventor Skill Input Space in SVD({svd_n_components}) + t-SNE 2D (colored by dominant archetype)",
                "t-SNE component 1",
                "t-SNE component 2",
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate SVD+t-SNE 2D plot: {exc}")

        try:
            emb_svd_tsne_all_3d = TSNE(
                n_components=3,
                perplexity=perplexity_svd_tsne,
                learning_rate="auto",
                init="pca",
                random_state=random_state,
            ).fit_transform(joint_svd)
            emb_svd_tsne_3d, emb_svd_tsne_arch_3d = _aa_split_joint_embedding(emb_svd_tsne_all_3d, n_sample)
            _aa_plot_embedding_3d(
                emb_svd_tsne_3d,
                emb_svd_tsne_arch_3d,
                dominant_sample,
                _aa_with_k_suffix(output_dir / "svd100_tsne_3d_input.png", selected_k),
                selected_k,
                f"Inventor Skill Input Space in SVD({svd_n_components}) + t-SNE 3D (colored by dominant archetype)",
                "t-SNE c1",
                "t-SNE c2",
                "t-SNE c3",
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate SVD+t-SNE 3D plot: {exc}")

    try:
        _plot_archetype_counts(
            dominant,
            _aa_with_k_suffix(output_dir / "dominant_archetype_counts.png", selected_k),
            selected_k,
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate dominant archetype counts plot: {exc}")

    try:
        _plot_archetype_similarity_heatmap(
            _cosine_similarity_rows(archetypes),
            _aa_with_k_suffix(output_dir / "archetype_cosine_similarity_heatmap.png", selected_k),
        )
    except Exception as exc:
        print(f"[WARN] Failed to generate archetype similarity heatmap: {exc}")

    try:
        _save_top_features_csv(
            archetypes=archetypes,
            feature_names=feature_names,
            out_path=_aa_with_k_suffix(output_dir / "archetype_top_features.csv", selected_k),
            top_n=25,
        )
    except Exception as exc:
        print(f"[WARN] Failed to export archetype top features: {exc}")

    run_dir = metadata.get("run_dir")
    embedding_input_file = str(Path(run_dir) / "inventor_skill_matrix.csv.gz") if run_dir else None
    dominant_coloring_file = str(Path(run_dir) / "memberships.csv.gz") if run_dir else None

    _write_json(
        _aa_with_k_suffix(output_dir / "analysis_metadata.json", selected_k),
        {
            **metadata,
            "output_dir": str(output_dir),
            "selected_k": int(selected_k),
            "inventors_total": int(memberships.shape[0]),
            "features_total": int(archetypes.shape[1]),
            "inventors_used_for_embedding": int(n_sample),
            "joint_embedding_points": int(n_joint),
            "embedding_input_file": embedding_input_file,
            "embedding_basis": "run-specific inventor input matrix",
            "dominant_coloring_file": dominant_coloring_file,
            "dominant_coloring_basis": "AA memberships argmax (equivalent to normalized coefficients argmax)",
            "archetype_marker_basis": "AA archetype vectors jointly embedded with inventor inputs",
            "tsne_perplexity": float(perplexity),
            "pca_explained_variance_ratio_2d": pca_2d_evr,
            "pca_explained_variance_ratio_3d": pca_3d_evr,
            "isomap_n_neighbors": int(n_neighbors),
            "umap_metric": "cosine",
            "umap_n_neighbors": 30,
            "umap_min_dist": 0.10,
            "svd_n_components_for_sparse_plots": int(svd_n_components) if svd_n_components is not None else None,
            "svd_explained_variance_ratio_sum": svd_explained_variance_ratio_sum,
            "random_state": int(random_state),
            "membership_simplex_basis": "AA memberships projected with archetypes.visualization.simplex.simplex",
            "purity_definition": "max normalized AA membership per inventor",
            "purity_stats_files": {
                "overall": str(_aa_with_k_suffix(output_dir / "purity_stats_overall.csv", selected_k)),
                "by_archetype": str(_aa_with_k_suffix(output_dir / "purity_stats_by_archetype.csv", selected_k)),
            },
            "max_inventors_tsne": int(max_inventors_tsne),
        },
    )

    pd.DataFrame(
        {
            "inventor": inventor_index[sample_idx].astype(str),
            "dominant_archetype": dominant_sample,
            "tsne2_x": _aa_embedding_column(emb2, 0, n_sample),
            "tsne2_y": _aa_embedding_column(emb2, 1, n_sample),
            "tsne3_x": _aa_embedding_column(emb3, 0, n_sample),
            "tsne3_y": _aa_embedding_column(emb3, 1, n_sample),
            "tsne3_z": _aa_embedding_column(emb3, 2, n_sample),
            "pca2_x": _aa_embedding_column(emb_pca2, 0, n_sample),
            "pca2_y": _aa_embedding_column(emb_pca2, 1, n_sample),
            "pca3_x": _aa_embedding_column(emb_pca3, 0, n_sample),
            "pca3_y": _aa_embedding_column(emb_pca3, 1, n_sample),
            "pca3_z": _aa_embedding_column(emb_pca3, 2, n_sample),
            "isomap2_x": _aa_embedding_column(emb_iso2, 0, n_sample),
            "isomap2_y": _aa_embedding_column(emb_iso2, 1, n_sample),
            "isomap3_x": _aa_embedding_column(emb_iso3, 0, n_sample),
            "isomap3_y": _aa_embedding_column(emb_iso3, 1, n_sample),
            "isomap3_z": _aa_embedding_column(emb_iso3, 2, n_sample),
            "umap_raw_2d_x": _aa_embedding_column(emb_umap_raw_2d, 0, n_sample),
            "umap_raw_2d_y": _aa_embedding_column(emb_umap_raw_2d, 1, n_sample),
            "umap_raw_3d_x": _aa_embedding_column(emb_umap_raw_3d, 0, n_sample),
            "umap_raw_3d_y": _aa_embedding_column(emb_umap_raw_3d, 1, n_sample),
            "umap_raw_3d_z": _aa_embedding_column(emb_umap_raw_3d, 2, n_sample),
            "svd_umap_2d_x": _aa_embedding_column(emb_umap_svd_2d, 0, n_sample),
            "svd_umap_2d_y": _aa_embedding_column(emb_umap_svd_2d, 1, n_sample),
            "svd_umap_3d_x": _aa_embedding_column(emb_umap_svd_3d, 0, n_sample),
            "svd_umap_3d_y": _aa_embedding_column(emb_umap_svd_3d, 1, n_sample),
            "svd_umap_3d_z": _aa_embedding_column(emb_umap_svd_3d, 2, n_sample),
            "svd_tsne_2d_x": _aa_embedding_column(emb_svd_tsne_2d, 0, n_sample),
            "svd_tsne_2d_y": _aa_embedding_column(emb_svd_tsne_2d, 1, n_sample),
            "svd_tsne_3d_x": _aa_embedding_column(emb_svd_tsne_3d, 0, n_sample),
            "svd_tsne_3d_y": _aa_embedding_column(emb_svd_tsne_3d, 1, n_sample),
            "svd_tsne_3d_z": _aa_embedding_column(emb_svd_tsne_3d, 2, n_sample),
        }
    ).to_csv(
        _aa_with_k_suffix(output_dir / "inventor_embedding_coordinates_sampled.csv", selected_k),
        index=False,
    )

    pd.DataFrame(
        {
            "archetype": [f"A{i}" for i in range(1, selected_k + 1)],
            "tsne2_x": _aa_embedding_column(emb2_arch, 0, selected_k),
            "tsne2_y": _aa_embedding_column(emb2_arch, 1, selected_k),
            "tsne3_x": _aa_embedding_column(emb3_arch, 0, selected_k),
            "tsne3_y": _aa_embedding_column(emb3_arch, 1, selected_k),
            "tsne3_z": _aa_embedding_column(emb3_arch, 2, selected_k),
            "pca2_x": _aa_embedding_column(emb_pca2_arch, 0, selected_k),
            "pca2_y": _aa_embedding_column(emb_pca2_arch, 1, selected_k),
            "pca3_x": _aa_embedding_column(emb_pca3_arch, 0, selected_k),
            "pca3_y": _aa_embedding_column(emb_pca3_arch, 1, selected_k),
            "pca3_z": _aa_embedding_column(emb_pca3_arch, 2, selected_k),
            "isomap2_x": _aa_embedding_column(emb_iso2_arch, 0, selected_k),
            "isomap2_y": _aa_embedding_column(emb_iso2_arch, 1, selected_k),
            "isomap3_x": _aa_embedding_column(emb_iso3_arch, 0, selected_k),
            "isomap3_y": _aa_embedding_column(emb_iso3_arch, 1, selected_k),
            "isomap3_z": _aa_embedding_column(emb_iso3_arch, 2, selected_k),
            "umap_raw_2d_x": _aa_embedding_column(emb_umap_raw_arch_2d, 0, selected_k),
            "umap_raw_2d_y": _aa_embedding_column(emb_umap_raw_arch_2d, 1, selected_k),
            "umap_raw_3d_x": _aa_embedding_column(emb_umap_raw_arch_3d, 0, selected_k),
            "umap_raw_3d_y": _aa_embedding_column(emb_umap_raw_arch_3d, 1, selected_k),
            "umap_raw_3d_z": _aa_embedding_column(emb_umap_raw_arch_3d, 2, selected_k),
            "svd_umap_2d_x": _aa_embedding_column(emb_umap_svd_arch_2d, 0, selected_k),
            "svd_umap_2d_y": _aa_embedding_column(emb_umap_svd_arch_2d, 1, selected_k),
            "svd_umap_3d_x": _aa_embedding_column(emb_umap_svd_arch_3d, 0, selected_k),
            "svd_umap_3d_y": _aa_embedding_column(emb_umap_svd_arch_3d, 1, selected_k),
            "svd_umap_3d_z": _aa_embedding_column(emb_umap_svd_arch_3d, 2, selected_k),
            "svd_tsne_2d_x": _aa_embedding_column(emb_svd_tsne_arch_2d, 0, selected_k),
            "svd_tsne_2d_y": _aa_embedding_column(emb_svd_tsne_arch_2d, 1, selected_k),
            "svd_tsne_3d_x": _aa_embedding_column(emb_svd_tsne_arch_3d, 0, selected_k),
            "svd_tsne_3d_y": _aa_embedding_column(emb_svd_tsne_arch_3d, 1, selected_k),
            "svd_tsne_3d_z": _aa_embedding_column(emb_svd_tsne_arch_3d, 2, selected_k),
        }
    ).to_csv(
        _aa_with_k_suffix(output_dir / "archetype_embedding_coordinates.csv", selected_k),
        index=False,
    )

    return output_dir


def _run_archetype_interpretation(
    *,
    archetypes: np.ndarray,
    feature_names: List[str],
    selected_k: int,
    output_dir: Path,
    metadata: Dict[str, Any],
    top_n: int = 10,
) -> Path:
    if archetypes.ndim != 2:
        raise ValueError("Expected 2D archetype matrix for interpretation outputs.")
    if archetypes.shape[1] != len(feature_names):
        raise ValueError(
            "Mismatch between archetype features and inventor skill columns: "
            f"{archetypes.shape[1]} vs {len(feature_names)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    arch_df = pd.DataFrame(
        archetypes,
        columns=feature_names,
        index=[f"Archetype_{i+1}" for i in range(archetypes.shape[0])],
    )
    arch_df.to_csv(output_dir / f"archetype_k{selected_k}_matrix.csv.gz", compression="gzip")

    mean_feature_scores = arch_df.mean(axis=0)
    rows = []
    for archetype_name in arch_df.index:
        scores = arch_df.loc[archetype_name]

        top_values = scores.sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(top_values.items(), start=1):
            rows.append(
                {
                    "archetype": archetype_name,
                    "rank": rank,
                    "feature": feature,
                    "score": float(value),
                    "kind": "top_value",
                }
            )

        top_delta_vs_mean = (scores - mean_feature_scores).sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(top_delta_vs_mean.items(), start=1):
            rows.append(
                {
                    "archetype": archetype_name,
                    "rank": rank,
                    "feature": feature,
                    "score": float(value),
                    "kind": "top_delta_vs_mean",
                }
            )

        competing_archetypes = arch_df.drop(index=archetype_name)
        if competing_archetypes.empty:
            second_best = pd.Series(0.0, index=arch_df.columns)
        else:
            second_best = competing_archetypes.max(axis=0)
        top_gap_vs_second_best = (scores - second_best).sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(top_gap_vs_second_best.items(), start=1):
            rows.append(
                {
                    "archetype": archetype_name,
                    "rank": rank,
                    "feature": feature,
                    "score": float(value),
                    "kind": "top_gap_vs_2nd_best",
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / f"archetype_k{selected_k}_top_features_and_deltas.csv", index=False)

    draft_rows = []
    for archetype_name in arch_df.index:
        top_feature = arch_df.loc[archetype_name].sort_values(ascending=False).index[0]
        draft_rows.append(
            {
                "archetype": archetype_name,
                "top_feature": top_feature,
                "draft_category": _draft_archetype_category(top_feature),
            }
        )
    pd.DataFrame(draft_rows).to_csv(output_dir / f"archetype_k{selected_k}_draft_categories.csv", index=False)

    _write_json(
        output_dir / "interpretation_metadata.json",
        {
            **metadata,
            "output_dir": str(output_dir),
            "selected_k": int(selected_k),
            "top_n": int(top_n),
            "n_features": int(len(feature_names)),
            "n_archetypes": int(archetypes.shape[0]),
        },
    )
    return output_dir


def _coefficients_to_memberships_df(coefficients: np.ndarray, index: pd.Index) -> pd.DataFrame:
    row_sums = coefficients.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    probs = coefficients / safe_row_sums
    perc = probs * 100.0
    k = coefficients.shape[1]
    columns = [f"Archetype_{i+1}" for i in range(k)]
    return pd.DataFrame(perc, index=index, columns=columns)


def build_inventor_skill_df(
    data: List[Dict],
    mode: str = "hard",
    inventor_field: str = "Inventors",
    skill_field: str = "skill_labels",
    inventor_sep: str = ";;",
    drop_empty_patents: bool = True,
) -> pd.DataFrame:
    """
    Build an Inventor x Skill matrix.

    Parameters
    ----------
    data : list of dict
        Patent records (already enriched with 'skill_labels': [(label, score), ...]).
    mode : {"soft","hard","binary-hard","tfidf","tf-idf"}
        - "soft": average similarity score per (inventor, skill)
        - "hard": count presence per patent (0/1 within a patent) and sum per inventor
        - "binary-hard": binary skill presence per inventor from the "hard" count matrix
        - "tfidf": TF-IDF transform of the "hard" count matrix
    inventor_field : str
        Field containing ';;'-separated inventor names.
    skill_field : str
        Field containing list of (skill_label, score) tuples.
    inventor_sep : str
        Separator used inside inventor_field.
    drop_empty_patents : bool
        If True, patents with empty skill list are skipped.

    Returns
    -------
    pd.DataFrame
        Rows = inventors, Columns = skills.
        - "soft": float (avg scores, 0.0 if absent)
        - "hard": int counts (sum of per-patent presence)
        - "binary-hard": int binary indicators (1 if inventor has the skill at least once)
        - "tfidf": float TF-IDF weights
    """
    mode = str(mode).strip().lower()
    match mode:
        case "soft" | "hard" | "binary-hard" | "tfidf" | "tf-idf":
            mode = "tfidf" if mode == "tf-idf" else mode
        case _:
            raise ValueError('mode must be one of: "soft", "hard", "binary-hard", "tfidf", "tf-idf"')

    # collect inventors + skills per patent
    # normalize inventors; optionally skip patents with no skills
    def iter_inventor_patent_entries() -> Iterable[Tuple[str, List[Tuple[str, float]]]]:
        for pat in data:
            skills = pat.get(skill_field, [])
            if drop_empty_patents and not skills:
                continue
            inv_raw = (pat.get(inventor_field) or "").split(inventor_sep)
            inventors = [s.strip() for s in inv_raw if s and s.strip()]
            if not inventors:
                continue
            yield from ((inv, skills) for inv in inventors)

    if mode in {"hard", "binary-hard", "tfidf"}:
        counts = defaultdict(lambda: defaultdict(int))  # inventor -> skill -> count
        all_skills = set()

        for inventor, skills in iter_inventor_patent_entries():
            # ensure per-patent 0/1 presence (avoid multiple adds from same patent)
            present = {label for (label, _score) in skills if label}
            if not present:
                continue
            all_skills.update(present)
            for skill in present:
                counts[inventor][skill] += 1

        if not counts:
            return pd.DataFrame()

        skill_list = sorted(all_skills)
        inventor_list = sorted(counts.keys())
        rows = [[counts[inv].get(sk, 0) for sk in skill_list] for inv in inventor_list]
        df_count = pd.DataFrame(rows, index=inventor_list, columns=skill_list).astype(int)

        match mode:
            case "hard":
                # return df_count
                df_hard = np.log1p(df_count.astype(float))
                return df_hard
            case "binary-hard":
                return (df_count > 0).astype(int)
            case "tfidf":
                try:
                    from sklearn.feature_extraction.text import TfidfTransformer
                    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
                    values = tfidf.fit_transform(df_count.values).toarray()
                    df_tfidf = pd.DataFrame(values, index=df_count.index, columns=df_count.columns)
                    return df_tfidf
                except Exception as e:
                    raise RuntimeError(f"Failed to compute TF-IDF: {e}")

    score_lists: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_skills = set()

    for inventor, skills in iter_inventor_patent_entries():
        for label, score in skills:
            if not label:
                continue
            all_skills.add(label)
            score_lists[inventor][label].append(float(score))

    if not score_lists:
        return pd.DataFrame()

    skill_list = sorted(all_skills)
    inventor_list = sorted(score_lists.keys())

    rows = [[mean_or_zero(score_lists[inv].get(sk)) for sk in skill_list] for inv in inventor_list]
    df_soft = pd.DataFrame(rows, index=inventor_list, columns=skill_list)
    return df_soft


def inventor_archetype_memberships(
    inventor_skill_df: pd.DataFrame,
    n_archetypes: int,
    max_k: int = 10,
    n_init: int = 1,
    random_state: Optional[int] = 42,
    alternative_random_seeds: Optional[List[int]] = None,
    iter_per_num_archetypes: int = 1,
    method="nnls",
    backend: str = "numpy",     # one of: "numpy", "jax", "torch"
    init: str = "uniform",      # see archetypes docs: 'uniform', 'furthest_sum', 'furthest_first', 'aa_plus_plus'
    max_iter: int = 500,
    tol: float = 1e-4,
    output_dir: str = "./output",
    experiment_metadata: Optional[Dict[str, Any]] = None,
    save_repro_bundle: bool = True,
    repro_subdir: str = "archetype_runs",
) -> pd.DataFrame:
    """
    Fit Archetypal Analysis on an inventor×skill matrix and return per-inventor
    archetype percentages (rows sum to 100).

    Parameters
    ----------
    inventor_skill_df : pd.DataFrame
        Rows = inventors, columns = skills. Values can be counts, TF-IDF, or soft scores.
    n_archetypes : int
        Number of archetypes to learn. If set to -1, an optimal number in [2, 15]
        (bounded additionally by n_samples) is selected automatically using an
        elbow rule on the (mean) residual sum of squares (RSS) curve. The elbow
        plot is saved as a PNG in `output_dir` for debugging.
    random_state : Optional[int]
        For reproducibility. Used for the first run per candidate k.
    alternative_random_seeds : Optional[List[int]]
        Seeds used for additional runs when `iter_per_num_archetypes > 1`.
        They are consumed in order for the 2nd, 3rd, ... runs per k.
        If there are more iterations than seeds, the remaining runs use `None`.
    iter_per_num_archetypes : int
        Number of AA runs (with different random_state) per candidate k
        when `n_archetypes == -1`. If 1, only `random_state` is used.

    backend : {"numpy","jax","torch"}
        Backend selector.
        - "numpy"/"jax": uses `AA(...).fit(X)` estimator path.
        - "torch": uses installed `archetypes` torch module
          `archetypes.torch._AA` or `archetypes.torch._aa` and calls
          `archetypal_analysis`.
    init : str
        Initialization method for AA.
    max_iter : int
        Max iterations.
    tol : float
        Convergence tolerance.
    output_dir : str
        Directory where the elbow plot will be saved when `n_archetypes == -1`.
    experiment_metadata : Optional[Dict[str, Any]]
        Optional metadata snapshot from the caller (config path, effective config,
        CLI overrides, etc.) to persist for reproducibility.
    save_repro_bundle : bool
        If True, persist a non-overwriting run bundle under
        `<output_dir>/<repro_subdir>/...`.
    repro_subdir : str
        Subdirectory inside output_dir where reproducibility bundles are stored.

    Returns
    -------
    pd.DataFrame
        Rows = inventors (index copied from input),
        Cols = Archetype_1..Archetype_k,
        Values = percentages (float, 0..100) of each archetype per inventor.
    """
    if inventor_skill_df is None or inventor_skill_df.empty:
        return pd.DataFrame()

    output_dir_path = Path(output_dir).resolve()
    run_start_utc = datetime.now(timezone.utc)
    run_start_perf = time.perf_counter()

    # Select backend
    AA_cls = None
    torch_archetypal_analysis_fn = None
    torch_backend_impl = None
    if backend == "numpy":
        from archetypes import AA as AA_cls
    elif backend == "jax":
        from archetypes.jax import AA as AA_cls
    elif backend == "torch":
        torch_archetypal_analysis_fn, torch_backend_impl = _load_installed_torch_archetypal_analysis()
    else:
        raise ValueError("backend must be one of: 'numpy', 'jax', 'torch'")

    X = inventor_skill_df.values.astype(float, copy=False)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if alternative_random_seeds is None:
        alternative_random_seeds = [7, 21, 35, 84, 45, 43, 100]

    aa_params = {
        "n_archetypes_requested": int(n_archetypes),
        "max_k": int(max_k),
        "n_init": int(n_init),
        "torch_backend_impl": torch_backend_impl if backend == "torch" else None,
        "torch_n_runs": int(n_init) if backend == "torch" else None,
        "torch_epochs": int(max_iter) if backend == "torch" else None,
        "torch_batch_size": 1 if backend == "torch" else None,
        "torch_dtype": "float32" if backend == "torch" else None,
        "random_state": random_state,
        "alternative_random_seeds": list(alternative_random_seeds),
        "iter_per_num_archetypes": int(iter_per_num_archetypes),
        "method": method,
        "backend": backend,
        "init": init,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "output_dir": str(output_dir_path),
        "save_repro_bundle": bool(save_repro_bundle),
        "repro_subdir": repro_subdir,
    }
    aa_core_params = {
        "backend": backend,
        "torch_backend_impl": torch_backend_impl if backend == "torch" else None,
        "torch_n_runs": int(n_init) if backend == "torch" else None,
        "torch_epochs": int(max_iter) if backend == "torch" else None,
        "torch_batch_size": 1 if backend == "torch" else None,
        "torch_dtype": "float32" if backend == "torch" else None,
        "method": method,
        "init": init,
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "iter_per_num_archetypes": int(iter_per_num_archetypes),
        "random_state": random_state,
        "alternative_random_seeds": list(alternative_random_seeds),
        "max_k": int(max_k),
        "n_archetypes_requested": int(n_archetypes),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }
    inventor_skill_df_fingerprint = _fingerprint_inventor_skill_df(inventor_skill_df)
    aa_core_params_fingerprint = _stable_hash(aa_core_params)

    run_mode = "auto" if n_archetypes == -1 else "fixed"
    run_timestamp = run_start_utc.strftime("%Y%m%dT%H%M%SZ")
    short_hash = hashlib.sha256(
        f"{run_timestamp}|{time.time_ns()}|{inventor_skill_df_fingerprint}|{aa_core_params_fingerprint}".encode("utf-8")
    ).hexdigest()[:10]
    run_id = f"{run_timestamp}__{run_mode}__{short_hash}"

    # Helper to compute RSS (residual sum of squares) robustly
    def _compute_rss(model, data: np.ndarray) -> float:
        rss_attr = getattr(model, "rss_", None)
        if rss_attr is not None:
            return float(rss_attr)

        recon_err = getattr(model, "reconstruction_error_", None)
        if recon_err is not None:
            return float(recon_err)

        coeff = _to_numpy_array(getattr(model, "coefficients_", None), dtype=np.float64)
        archetypes_mat = _to_numpy_array(getattr(model, "archetypes_", None), dtype=np.float64)
        X_hat = coeff @ archetypes_mat
        err_norm = np.linalg.norm(data - X_hat, ord="fro")
        return float(err_norm ** 2)

    # Simple wrapper to fit a model with given k and seed and return fit diagnostics.
    def _fit_aa_with_rss(k: int, seed: Optional[int]):
        if backend in ["numpy", "jax"]:
            aa = AA_cls(
                n_archetypes=k,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                init=init,
                random_state=seed,
                method=method,
                method_params={
                    "max_iter_optimizer": 5000,
                },
            )
            fit_start = time.perf_counter()
            aa.fit(X)
            fit_seconds = float(time.perf_counter() - fit_start)
            rss_val = _compute_rss(aa, X)
            n_iter = int(getattr(aa, "n_iter_", -1))
            return aa, rss_val, fit_seconds, n_iter
        elif backend == "torch":
            if torch_archetypal_analysis_fn is None:
                raise RuntimeError("Torch backend requested but installed archetypal_analysis function is not available.")
            try:
                import torch
            except Exception as exc:
                raise RuntimeError(f"Torch backend requested, but torch is unavailable: {exc}") from exc

            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_epochs = max(1, int(max_iter))
            torch_runs = max(1, int(n_init))

            fit_start = time.perf_counter()
            archetypes_t, proportions_t, mse_loss = torch_archetypal_analysis_fn(
                X=X,
                n_archetypes=int(k),
                n_runs=int(torch_runs),
                verbose=False,
                random_state=seed,
                epochs=int(torch_epochs),
                batch_size=10000,
                device=torch_device,
                dtype=torch.float32,
            )
            fit_seconds = float(time.perf_counter() - fit_start)

            coeff_np_raw = _to_numpy_array(proportions_t, dtype=np.float64)
            if coeff_np_raw.shape == (int(k), int(n_samples)):
                coeff_np = coeff_np_raw.T
            elif coeff_np_raw.shape == (int(n_samples), int(k)):
                coeff_np = coeff_np_raw
            else:
                raise RuntimeError(
                    "Unexpected torch proportions shape. "
                    f"Got {coeff_np_raw.shape}, expected {(int(n_samples), int(k))} "
                    f"or {(int(k), int(n_samples))}."
                )

            archetypes_np_raw = _to_numpy_array(archetypes_t, dtype=np.float64)
            if archetypes_np_raw.shape == (int(k), int(n_features)):
                archetypes_np = archetypes_np_raw
            elif archetypes_np_raw.shape == (int(n_features), int(k)):
                archetypes_np = archetypes_np_raw.T
            else:
                raise RuntimeError(
                    "Unexpected torch archetypes shape. "
                    f"Got {archetypes_np_raw.shape}, expected {(int(k), int(n_features))} "
                    f"or {(int(n_features), int(k))}."
                )
            x_hat = coeff_np @ archetypes_np
            residual = X - x_hat
            rss_val = float(np.sum(residual * residual))
            n_iter = int(torch_epochs)

            aa = types.SimpleNamespace(
                coefficients_=coeff_np,
                archetypes_=archetypes_np,
                arch_coefficients_=np.array([], dtype=np.float64),
                labels_=np.argmax(coeff_np, axis=1).astype(int) if coeff_np.size else np.array([], dtype=int),
                loss_=np.array([float(mse_loss)], dtype=np.float64),
                rss_=rss_val,
                n_iter_=n_iter,
                backend_impl_=torch_backend_impl,
            )
            return aa, rss_val, fit_seconds, n_iter
        else:
            raise ValueError("backend must be one of: 'numpy', 'jax', 'torch'")

    memberships_df: pd.DataFrame
    final_model: Optional[Any] = None
    selected_k: Optional[int] = None
    elbow_method = "not_applicable"
    candidate_ks: List[int] = []
    mean_rss_per_k: List[float] = []
    best_model_per_k: Dict[int, Any] = {}
    best_rss_per_k: Dict[int, float] = {}
    plot_points_records: List[Dict[str, Any]] = []
    k_sweep_run_records: List[Dict[str, Any]] = []
    k_sweep_summary_records: List[Dict[str, Any]] = []
    fixed_run_fit_seconds: Optional[float] = None
    fixed_run_n_iter: Optional[int] = None
    repro_root: Optional[Path] = None
    run_dir: Optional[Path] = None

    # Case 1: user specified a fixed number of archetypes
    if n_archetypes != -1:
        aa, fixed_rss, fixed_run_fit_seconds, fixed_run_n_iter = _fit_aa_with_rss(n_archetypes, random_state)
        coeff = _to_numpy_array(getattr(aa, "coefficients_", None), dtype=np.float64)
        memberships_df = _coefficients_to_memberships_df(coeff, inventor_skill_df.index)

        selected_k = int(coeff.shape[1])
        final_model = aa
        best_model_per_k[selected_k] = aa
        best_rss_per_k[selected_k] = float(fixed_rss)

        plot_points_records.append(
            {
                "k": int(selected_k),
                "rss": float(fixed_rss),
                "rss_kind": "fixed_single",
                "run_idx": 0,
                "seed": random_state,
                "fit_seconds": fixed_run_fit_seconds,
                "n_iter": fixed_run_n_iter,
            }
        )
    else:
        # Case 2: automatic selection of n_archetypes via elbow on RSS curve
        k_min = 2
        k_max = min(max_k, max(2, n_samples))
        n_runs = max(1, int(iter_per_num_archetypes))
        base_seeds = [random_state] + list(alternative_random_seeds)

        if n_samples < 2:
            aa, rss_1, fit_seconds_1, n_iter_1 = _fit_aa_with_rss(1, random_state)
            coeff = _to_numpy_array(getattr(aa, "coefficients_", None), dtype=np.float64)
            memberships_df = _coefficients_to_memberships_df(coeff, inventor_skill_df.index)
            selected_k = int(coeff.shape[1])
            final_model = aa
            elbow_method = "degenerate_n_samples_lt_2"
            candidate_ks = [selected_k]
            mean_rss_per_k = [float(rss_1)]
            best_model_per_k[selected_k] = aa
            best_rss_per_k[selected_k] = float(rss_1)

            k_sweep_run_records.append(
                {
                    "k": selected_k,
                    "run_idx": 0,
                    "seed": random_state,
                    "rss": float(rss_1),
                    "fit_seconds": fit_seconds_1,
                    "n_iter": int(n_iter_1),
                    "is_best_for_k": True,
                }
            )
            k_sweep_summary_records.append(
                {
                    "k": selected_k,
                    "mean_rss": float(rss_1),
                    "best_rss": float(rss_1),
                    "best_run_idx": 0,
                    "best_seed": random_state,
                    "best_n_iter": int(n_iter_1),
                    "best_fit_seconds": fit_seconds_1,
                }
            )
            plot_points_records.append(
                {
                    "k": int(selected_k),
                    "rss": float(rss_1),
                    "rss_kind": "auto_mean",
                    "run_idx": None,
                    "seed": None,
                    "fit_seconds": None,
                    "n_iter": None,
                }
            )
        else:
            candidate_ks = list(range(k_min, k_max + 1))
            # candidate_ks = list(range(26, 30 + 1))

            for k in tqdm(candidate_ks):
                rss_values: List[float] = []
                best_model_k = None
                best_rss_k = np.inf
                best_seed_k = None
                best_run_idx_k = None
                best_n_iter_k = None
                best_fit_seconds_k = None
                best_record_idx = None

                for run_idx in range(n_runs):
                    seed = base_seeds[run_idx] if run_idx < len(base_seeds) else None
                    aa_k, rss_k, fit_seconds_k, n_iter_k = _fit_aa_with_rss(k, seed)
                    rss_values.append(rss_k)

                    k_sweep_run_records.append(
                        {
                            "k": int(k),
                            "run_idx": int(run_idx),
                            "seed": seed,
                            "rss": float(rss_k),
                            "fit_seconds": fit_seconds_k,
                            "n_iter": int(n_iter_k),
                            "is_best_for_k": False,
                        }
                    )
                    curr_record_idx = len(k_sweep_run_records) - 1

                    if rss_k < best_rss_k:
                        best_rss_k = rss_k
                        best_model_k = aa_k
                        best_seed_k = seed
                        best_run_idx_k = run_idx
                        best_n_iter_k = int(n_iter_k)
                        best_fit_seconds_k = fit_seconds_k
                        best_record_idx = curr_record_idx

                if best_record_idx is not None:
                    k_sweep_run_records[best_record_idx]["is_best_for_k"] = True

                mean_rss = float(sum(rss_values) / len(rss_values))
                mean_rss_per_k.append(mean_rss)
                best_model_per_k[k] = best_model_k
                best_rss_per_k[k] = float(best_rss_k)
                k_sweep_summary_records.append(
                    {
                        "k": int(k),
                        "mean_rss": mean_rss,
                        "best_rss": float(best_rss_k),
                        "best_run_idx": best_run_idx_k,
                        "best_seed": best_seed_k,
                        "best_n_iter": best_n_iter_k,
                        "best_fit_seconds": best_fit_seconds_k,
                    }
                )
                plot_points_records.append(
                    {
                        "k": int(k),
                        "rss": mean_rss,
                        "rss_kind": "auto_mean",
                        "run_idx": None,
                        "seed": None,
                        "fit_seconds": None,
                        "n_iter": None,
                    }
                )

            # Try kneed for elbow detection first.
            best_k = None
            try:
                from kneed import KneeLocator  # type: ignore

                x = np.array(candidate_ks, dtype=float)
                y = np.array(mean_rss_per_k, dtype=float)

                kneedle = KneeLocator(
                    x,
                    y,
                    curve="convex",
                    direction="decreasing",
                )
                knee_x = kneedle.elbow or kneedle.knee

                if knee_x is not None:
                    knee_rounded = int(round(float(knee_x)))
                    knee_rounded = max(k_min, min(k_max, knee_rounded))
                    if knee_rounded in best_model_per_k:
                        best_k = knee_rounded
                        elbow_method = "kneed"
            except Exception:
                best_k = None

            if best_k is None:
                # Fallback: farthest point from the first-last chord.
                x = np.array(candidate_ks, dtype=float)
                y = np.array(mean_rss_per_k, dtype=float)

                x0, y0 = x[0], y[0]
                x1, y1 = x[-1], y[-1]
                vx, vy = x1 - x0, y1 - y0
                denom = (vx ** 2 + vy ** 2) ** 0.5 or 1.0

                distances = []
                for xi, yi in zip(x[1:-1], y[1:-1]):
                    px, py = xi - x0, yi - y0
                    cross = abs(px * vy - py * vx)
                    distances.append(cross / denom)

                if distances:
                    max_idx = int(np.argmax(distances))
                    best_k = int(x[1 + max_idx])
                else:
                    best_k = int(candidate_ks[0])
                elbow_method = "chord_farthest_point"

            selected_k = int(best_k)
            final_model = best_model_per_k[selected_k]
            coeff = _to_numpy_array(getattr(final_model, "coefficients_", None), dtype=np.float64)
            memberships_df = _coefficients_to_memberships_df(coeff, inventor_skill_df.index)

        # Keep legacy top-level elbow plot behavior for auto mode.
        try:
            if candidate_ks and mean_rss_per_k:
                _save_elbow_plot(
                    fig_path=output_dir_path / "elbow_n_archetypes.png",
                    candidate_ks=candidate_ks,
                    mean_rss_per_k=mean_rss_per_k,
                    best_k=selected_k,
                )
        except Exception:
            pass

    if final_model is None or selected_k is None:
        raise RuntimeError("Failed to fit Archetypal Analysis model.")

    for rec in plot_points_records:
        rec["source_run_id"] = run_id
        rec["source_mode"] = run_mode
        rec["n_archetypes_requested"] = int(n_archetypes)
        rec["selected_k"] = int(selected_k)
        rec["max_k"] = int(max_k)
        rec["iter_per_num_archetypes"] = int(iter_per_num_archetypes)
        rec["random_state"] = random_state
        rec["backend"] = backend
        rec["method"] = method
        rec["init"] = init
        rec["n_init"] = int(n_init)
        rec["max_iter"] = int(max_iter)
        rec["tol"] = float(tol)
        rec["inventor_skill_df_fingerprint"] = inventor_skill_df_fingerprint
        rec["aa_core_params_fingerprint"] = aa_core_params_fingerprint

    if save_repro_bundle:
        try:
            repro_root = output_dir_path / repro_subdir
            repro_root.mkdir(parents=True, exist_ok=True)
            run_dir = repro_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            inventor_skill_df.to_csv(run_dir / "inventor_skill_matrix.csv.gz", compression="gzip")
            memberships_df.to_csv(run_dir / "memberships.csv.gz", compression="gzip")

            _write_json(
                run_dir / "fingerprints.json",
                {
                    "inventor_skill_df_fingerprint": inventor_skill_df_fingerprint,
                    "aa_core_params_fingerprint": aa_core_params_fingerprint,
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                },
            )

            plot_points_columns = [
                "k",
                "rss",
                "rss_kind",
                "source_run_id",
                "source_mode",
                "n_archetypes_requested",
                "selected_k",
                "max_k",
                "iter_per_num_archetypes",
                "run_idx",
                "seed",
                "fit_seconds",
                "n_iter",
                "random_state",
                "backend",
                "method",
                "init",
                "n_init",
                "max_iter",
                "tol",
                "inventor_skill_df_fingerprint",
                "aa_core_params_fingerprint",
            ]
            pd.DataFrame(plot_points_records, columns=plot_points_columns).to_csv(
                run_dir / "plot_points.csv",
                index=False,
            )

            if run_mode == "auto":
                pd.DataFrame(k_sweep_run_records).to_csv(run_dir / "k_sweep_runs.csv", index=False)
                pd.DataFrame(k_sweep_summary_records).to_csv(run_dir / "k_sweep_summary.csv", index=False)
                _write_json(
                    run_dir / "elbow_payload.json",
                    {
                        "candidate_ks": [int(k) for k in candidate_ks],
                        "mean_rss_per_k": [float(v) for v in mean_rss_per_k],
                        "best_k": int(selected_k),
                        "elbow_method": elbow_method,
                    },
                )
                if candidate_ks and mean_rss_per_k:
                    _save_elbow_plot(
                        fig_path=run_dir / "elbow_n_archetypes.png",
                        candidate_ks=candidate_ks,
                        mean_rss_per_k=mean_rss_per_k,
                        best_k=selected_k,
                    )

            if run_mode == "fixed":
                fixed_rss_val = float(best_rss_per_k[selected_k])
                _save_model_npz(
                    path=models_dir / f"k_{int(selected_k)}.npz",
                    model=final_model,
                    rss=fixed_rss_val,
                    k=int(selected_k),
                )
            else:
                for k_model, model_obj in best_model_per_k.items():
                    if model_obj is None:
                        continue
                    _save_model_npz(
                        path=models_dir / f"k_{int(k_model)}.npz",
                        model=model_obj,
                        rss=float(best_rss_per_k.get(k_model, np.nan)),
                        k=int(k_model),
                    )

            run_end_utc = datetime.now(timezone.utc)
            run_duration_seconds = float(time.perf_counter() - run_start_perf)
            matrix_values = _to_numpy_array(inventor_skill_df.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
            matrix_nonzero = int(np.count_nonzero(matrix_values))

            run_manifest = {
                "run_id": run_id,
                "run_mode": run_mode,
                "started_at_utc": run_start_utc.isoformat(),
                "finished_at_utc": run_end_utc.isoformat(),
                "duration_seconds": run_duration_seconds,
                "paths": {
                    "output_dir": str(output_dir_path),
                    "repro_root": str(repro_root),
                    "run_dir": str(run_dir),
                },
                "parameters": aa_params,
                "selection": {
                    "selected_k": int(selected_k),
                    "target_k": int(n_archetypes) if n_archetypes != -1 else None,
                    "elbow_method": elbow_method,
                },
                "data_summary": {
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "nonzero_entries": matrix_nonzero,
                },
                "fingerprints": {
                    "inventor_skill_df_fingerprint": inventor_skill_df_fingerprint,
                    "aa_core_params_fingerprint": aa_core_params_fingerprint,
                },
                "environment": _collect_environment_versions(),
                "experiment_metadata": experiment_metadata or {},
                "fit_summary": {
                    "fixed_fit_seconds": fixed_run_fit_seconds,
                    "fixed_n_iter": fixed_run_n_iter,
                    "k_sweep_candidates": [int(k) for k in candidate_ks],
                },
            }
            _write_json(run_dir / "run_manifest.json", run_manifest)

            _append_jsonl(
                repro_root / "index.jsonl",
                {
                    "run_id": run_id,
                    "run_mode": run_mode,
                    "run_dir": str(run_dir),
                    "created_at_utc": run_end_utc.isoformat(),
                    "n_archetypes_requested": int(n_archetypes),
                    "selected_k": int(selected_k),
                    "target_k": int(n_archetypes) if n_archetypes != -1 else None,
                    "inventor_skill_df_fingerprint": inventor_skill_df_fingerprint,
                    "aa_core_params_fingerprint": aa_core_params_fingerprint,
                },
            )
        except Exception as exc:
            print(f"[WARN] Failed to save reproducibility bundle: {exc}")

    postprocess_random_state = int(random_state if random_state is not None else 42)
    effective_config = (experiment_metadata or {}).get("effective_config", {}) if experiment_metadata else {}
    max_inventors_tsne_value = effective_config.get(
        "max_inventors_tsne",
        effective_config.get("archetypes_plots_max_inventors_tsne", 40000),
    )
    max_inventors_tsne = int(max_inventors_tsne_value) if max_inventors_tsne_value is not None else 40000

    coefficients = _to_numpy_array(getattr(final_model, "coefficients_", None), dtype=np.float64)
    archetypes = _to_numpy_array(getattr(final_model, "archetypes_", None), dtype=np.float64)
    feature_names = [str(col) for col in inventor_skill_df.columns.tolist()]
    inventor_index = pd.Index(inventor_skill_df.index)

    if coefficients.ndim == 2 and archetypes.ndim == 2:
        postprocess_metadata = {
            "base_dir": str(output_dir_path),
            "run_id": run_id,
            "run_mode": run_mode,
            "run_dir": str(run_dir) if run_dir is not None else None,
            "repro_root": str(repro_root) if repro_root is not None else None,
            "selected_k": int(selected_k),
            "elbow_method": elbow_method,
        }

        try:
            _run_archetype_optical_analysis_current(
                inventor_skill_df=inventor_skill_df,
                memberships_df=memberships_df,
                archetypes=archetypes,
                inventor_index=inventor_index,
                feature_names=feature_names,
                selected_k=int(selected_k),
                output_dir=output_dir_path / "archetypes_plots",
                random_state=postprocess_random_state,
                max_inventors_tsne=max_inventors_tsne,
                metadata=postprocess_metadata,
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate archetypes_plots outputs: {exc}")

        try:
            _run_archetype_interpretation(
                archetypes=archetypes,
                feature_names=feature_names,
                selected_k=int(selected_k),
                output_dir=output_dir_path / "archetypes_interpretation",
                metadata=postprocess_metadata,
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate archetypes_interpretation outputs: {exc}")
    else:
        print("[WARN] Skipping archetype post-processing because fitted model outputs are not 2D matrices.")

    return memberships_df


def patents_to_long_with_all_cols_explode(
    data,
    keep_link: bool = False,
    drop_llm_cols: bool = True,
    explode_inventors: bool = False,
    inventor_field: str = "Inventors",
    inventor_sep: str = ";;",
    drop_empty_patents: bool = True,
):
    df = pd.DataFrame(data)

    if drop_llm_cols:
        df = df.drop(columns=["llama3_1_8b_instruct_q8_0", "qwen2_5_7b_instruct", "mistral_instruct"], errors="ignore")

    if drop_empty_patents:
        df = df[df["skill_labels"].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)].copy()

    if explode_inventors:
        # Same inventor preprocessing used in iter_inventor_patent_entries:
        # split by separator, trim, drop empty names, then explode.
        def _parse_inventors(v):
            inv_raw = (v or "").split(inventor_sep)
            return [s.strip() for s in inv_raw if s and s.strip()]

        df["_inventor_names"] = df[inventor_field].apply(_parse_inventors)
        df = df[df["_inventor_names"].map(len) > 0].copy()

    # explode skills
    df = df.explode("skill_labels", ignore_index=True)

    # split (skill, score)
    df[["skill", "score"]] = pd.DataFrame(df["skill_labels"].tolist(), index=df.index)
    df["score"] = df["score"].astype(float)

    if explode_inventors:
        df = df.explode("_inventor_names", ignore_index=True)
        df["inventor"] = df["_inventor_names"]
        df = df.drop(columns=["_inventor_names"])

    if keep_link:
        df = df.explode("skill_links", ignore_index=True)
        # This ONLY works if skill_labels and skill_links explode in the same order and same length per patent.
        # If that's always true in your pipeline, it's fine.
        df["skill_link"] = df["skill_links"].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and x else None)

    return df


def main():
    pass


if __name__ == '__main__':
    main()
