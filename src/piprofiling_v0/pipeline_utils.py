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
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from tqdm import tqdm


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
    import matplotlib.pyplot as plt

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
    mode : {"soft","hard","tfidf"}
        - "soft": average similarity score per (inventor, skill)
        - "hard": count presence per patent (0/1 within a patent) and sum per inventor
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
        - "tfidf": float TF-IDF weights
    """
    if mode not in {"soft", "hard", "tfidf"}:
        raise ValueError('mode must be one of: "soft", "hard", "tfidf"')

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

    if mode in {"hard", "tfidf"}:
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

        if mode == "hard":
            return df_count

        # mode == "tfidf"
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
        Which backend to use from the `archetypes` package.
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

    run_start_utc = datetime.now(timezone.utc)
    run_start_perf = time.perf_counter()

    # Select backend
    if backend == "numpy":
        from archetypes import AA as AA_cls
    elif backend == "jax":
        from archetypes.jax import AA as AA_cls
    elif backend == "torch":
        from archetypes.torch import AA as AA_cls
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
        "random_state": random_state,
        "alternative_random_seeds": list(alternative_random_seeds),
        "iter_per_num_archetypes": int(iter_per_num_archetypes),
        "method": method,
        "backend": backend,
        "init": init,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "output_dir": str(Path(output_dir).resolve()),
        "save_repro_bundle": bool(save_repro_bundle),
        "repro_subdir": repro_subdir,
    }
    aa_core_params = {
        "backend": backend,
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
                method_kwargs={
                    "max_iter_optimizer": 5000,
                },
            )
        elif backend == "torch":
            aa = AA_cls(
                n_archetypes=k,
                max_iter=max_iter,
                tol=tol,
                init=init,
                device='cuda' if backend == "torch" else None,
                method_kwargs={
                    "max_iter_optimizer": 5000,
                },
            )
        else:
            raise ValueError("backend must be one of: 'numpy', 'jax', 'torch'")

        fit_start = time.perf_counter()
        aa.fit(X)
        fit_seconds = float(time.perf_counter() - fit_start)
        rss_val = _compute_rss(aa, X)
        n_iter = int(getattr(aa, "n_iter_", -1))
        return aa, rss_val, fit_seconds, n_iter

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
            out_dir_path = Path(output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            if candidate_ks and mean_rss_per_k:
                _save_elbow_plot(
                    fig_path=out_dir_path / "elbow_n_archetypes.png",
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
            output_dir_path = Path(output_dir).resolve()
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
