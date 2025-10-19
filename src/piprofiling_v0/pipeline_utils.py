'''
@File    :   pipeline_utils.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
'''
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Optional
import json
import os
import numpy as np


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
    backend: str = "numpy",     # one of: "numpy", "jax", "torch"
    init: str = "uniform",      # see archetypes docs: 'uniform', 'furthest_sum', 'furthest_first', 'aa_plus_plus'
    max_iter: int = 300,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """
    Fit Archetypal Analysis on an inventor×skill matrix and return per-inventor
    archetype percentages (rows sum to 100).

    Parameters
    ----------
    inventor_skill_df : pd.DataFrame
        Rows = inventors, columns = skills. Values can be counts, TF-IDF, or soft scores.
    n_archetypes : int
        Number of archetypes to learn.
    random_state : Optional[int]
        For reproducibility.
    backend : {"numpy","jax","torch"}
        Which backend to use from the `archetypes` package.
    init : str
        Initialization method for AA.
    max_iter : int
        Max iterations.
    tol : float
        Convergence tolerance.

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
        from archetypes import AA
    elif backend == "jax":
        from archetypes.jax import AA
    elif backend == "torch":
        from archetypes.torch import AA
    else:
        raise ValueError("backend must be one of: 'numpy', 'jax', 'torch'")

    # Prepare data
    X = inventor_skill_df.values.astype(float, copy=False)

    # Fit AA
    aa = AA(
        n_archetypes=n_archetypes,
        max_iter=max_iter,
        tol=tol,
        init=init,
        random_state=random_state,
    )
    aa.fit(X)

    # The estimator exposes per-sample coefficients in `coefficients_` (shape: n_samples×n_archetypes).
    # We row-normalize them to sum to 1 (then scale to 0..100).
    coeff = np.asarray(aa.coefficients_, dtype=float)  # (n_samples, k)
    row_sums = coeff.sum(axis=1, keepdims=True)
    # Avoid division by zero: if a row sums to 0, keep it as zeros
    safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    probs = coeff / safe_row_sums
    perc = probs * 100.0

    # Build a tidy DataFrame
    k = coeff.shape[1]
    columns = [f"Archetype_{i+1}" for i in range(k)]
    memberships_df = pd.DataFrame(perc, index=inventor_skill_df.index, columns=columns)

    return memberships_df


def main():
    pass


if __name__ == '__main__':
    main()