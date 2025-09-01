'''
@File    :   pipeline_utils.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
'''
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json
import os


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


def main():
    pass


if __name__ == '__main__':
    main()