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
    random_state: Optional[int] = 42,
    alternative_random_seeds: Optional[List[int]] = None,
    iter_per_num_archetypes: int = 1,
    backend: str = "numpy",     # one of: "numpy", "jax", "torch"
    init: str = "uniform",      # see archetypes docs: 'uniform', 'furthest_sum', 'furthest_first', 'aa_plus_plus'
    max_iter: int = 300,
    tol: float = 1e-4,
    output_dir: str = "./output",
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

    Returns
    -------
    pd.DataFrame
        Rows = inventors (index copied from input),
        Cols = Archetype_1..Archetype_k,
        Values = percentages (float, 0..100) of each archetype per inventor.
    """
    if inventor_skill_df is None or inventor_skill_df.empty:
        return pd.DataFrame()

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

    # Helper to compute RSS (residual sum of squares) robustly
    def _compute_rss(model, data: np.ndarray) -> float:
        rss_attr = getattr(model, "rss_", None)
        if rss_attr is not None:
            return float(rss_attr)

        recon_err = getattr(model, "reconstruction_error_", None)
        if recon_err is not None:
            return float(recon_err)

        coeff = np.asarray(model.coefficients_, dtype=float)
        archetypes_mat = np.asarray(model.archetypes_, dtype=float)
        X_hat = coeff @ archetypes_mat
        err_norm = np.linalg.norm(data - X_hat, ord="fro")
        return float(err_norm ** 2)

    # Simple wrapper to fit a model with given k and seed and return (model, rss)
    def _fit_aa_with_rss(k: int, seed: Optional[int]):
        aa = AA_cls(
            n_archetypes=k,
            max_iter=max_iter,
            tol=tol,
            init=init,
            random_state=seed,
            method_kwargs={
                "max_iter_optimizer": 5000,
            },
        )
        aa.fit(X)
        rss_val = _compute_rss(aa, X)
        return aa, rss_val

    # Case 1: user specified a fixed number of archetypes → original behaviour
    if n_archetypes != -1:
        aa, _ = _fit_aa_with_rss(n_archetypes, random_state)

        coeff = np.asarray(aa.coefficients_, dtype=float)  # (n_samples, k)
        row_sums = coeff.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        probs = coeff / safe_row_sums
        perc = probs * 100.0

        k = coeff.shape[1]
        columns = [f"Archetype_{i+1}" for i in range(k)]
        return pd.DataFrame(perc, index=inventor_skill_df.index, columns=columns)

    # Case 2: automatic selection of n_archetypes via elbow on RSS curve
    # ------------------------------------------------------------------
    # Range proposal: 2..15 is standard for elbow methods; we also bound by n_samples.
    k_min = 2
    k_max = min(45, max(2, n_samples))  # ensure at least 2 if n_samples >= 2

    if n_samples < 2:
        aa, _ = _fit_aa_with_rss(1, random_state)
        coeff = np.asarray(aa.coefficients_, dtype=float)
        row_sums = coeff.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        probs = coeff / safe_row_sums
        perc = probs * 100.0
        columns = ["Archetype_1"]
        return pd.DataFrame(perc, index=inventor_skill_df.index, columns=columns)

    candidate_ks = list(range(k_min, k_max + 1))

    if alternative_random_seeds is None:
        alternative_random_seeds = [7, 21, 35, 84, 45, 43, 100]

    base_seeds = [random_state] + list(alternative_random_seeds)
    n_runs = max(1, int(iter_per_num_archetypes))

    mean_rss_per_k: List[float] = []
    best_model_per_k: Dict[int, Any] = {}

    for k in tqdm(candidate_ks):
        rss_values: List[float] = []
        best_model_k = None
        best_rss_k = np.inf

        for run_idx in range(n_runs):
            seed = base_seeds[run_idx] if run_idx < len(base_seeds) else None
            aa_k, rss_k = _fit_aa_with_rss(k, seed)
            rss_values.append(rss_k)

            if rss_k < best_rss_k:
                best_rss_k = rss_k
                best_model_k = aa_k

        mean_rss = float(sum(rss_values) / len(rss_values))
        mean_rss_per_k.append(mean_rss)
        best_model_per_k[k] = best_model_k

    # Try to use kneed.KneeLocator if available for elbow detection
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
            if knee_rounded < k_min:
                knee_rounded = k_min
            if knee_rounded > k_max:
                knee_rounded = k_max
            if knee_rounded in best_model_per_k:
                best_k = knee_rounded
    except Exception:
        best_k = None

    if best_k is None:
        # Fallback elbow detection: "farthest point from the chord"
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
            best_k = candidate_ks[0]

    # --- Save elbow plot for debugging ---
    try:
        from pathlib import Path as _Path
        import matplotlib.pyplot as plt

        out_dir_path = _Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir_path / "elbow_n_archetypes.png"

        plt.figure()
        plt.plot(candidate_ks, mean_rss_per_k, marker="o")
        plt.xlabel("Number of archetypes (k)")
        plt.ylabel("Mean RSS")
        plt.title("Elbow plot for AA (k vs mean RSS)")

        # Mark chosen k
        if best_k is not None:
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
    except Exception:
        # Plotting is only for debugging; ignore any failures.
        pass

    # Use the best model we already fitted for best_k
    final_model = best_model_per_k[best_k]
    coeff = np.asarray(final_model.coefficients_, dtype=float)
    row_sums = coeff.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    probs = coeff / safe_row_sums
    perc = probs * 100.0

    k = coeff.shape[1]
    columns = [f"Archetype_{i+1}" for i in range(k)]
    memberships_df = pd.DataFrame(perc, index=inventor_skill_df.index, columns=columns)

    return memberships_df


def main():
    pass


if __name__ == '__main__':
    main()
