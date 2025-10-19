'''
@File    :   pipeline.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
'''
import argparse
from pathlib import Path
import pandas as pd
import torch

from utils import load_config, override_config_values
from pipeline_utils import (
    load_data, 
    Link2Skill_Mapping,
    build_inventor_skill_df,
    inventor_archetype_memberships
)

from esco_skill_extractor import SkillExtractor


def main():
    parser = argparse.ArgumentParser(description="Patent Inventors Profiling Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="./src/piprofiling_v0/configs/piprofiling.yaml",
        help="Full path to the YAML configuration file",
    )
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default=None,
        help="Path to the CSV file containing input data.",
    )
    
    args = parser.parse_args()

    # Load initial config from file
    yaml_file_path = Path(args.config).resolve()
    config_values = load_config(yaml_file_path)

    override_config_values(args, config_values)

    data = load_data(config_values.get("data_csv_path", None))

    # Initializations ---------------------------------------------------------
    skill_extractor = SkillExtractor(
        model=config_values.get("model", "all-MiniLM-L6-v2"),
        skills_threshold=config_values.get("model_skill_threshold", 0.6),
        device = "cuda" if config_values.get("device", None)=="cuda" and torch.cuda.is_available() else "cpu",
    )

    link2skill_mapper = Link2Skill_Mapping(csv_path=rf'{config_values.get("link2skill_mapping_file", None)}')
    # -------------------------------------------------------------------------

    batch_size = config_values.get("batch_size", 32)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_text = [
            patent['Title'] + " " + patent['Abstract'] 
            for patent in batch
        ]

        batch_skills = skill_extractor.get_skills(batch_text)

        for j in range(len(batch)):
            batch[j]['skill_links'] = batch_skills[j]
            batch[j]['skill_labels'] = [
                (link2skill_mapper.link2skill(link[0]), link[1]) for link in batch_skills[j]
            ]
    
    # filter out skills with similarity < onfig_values.get("actual_skill_threshold", None)
    if config_values.get("actual_skill_threshold", None) is not None:
        for patent in data:
            valid_idx = [k for k, (_, score) in enumerate(patent['skill_links']) if score > config_values.get("actual_skill_threshold", None)]
            patent['skill_links'] = [patent['skill_links'][k] for k in valid_idx]
            patent['skill_labels'] = [patent['skill_labels'][k] for k in valid_idx]

    inventor_skill_df = build_inventor_skill_df(
        data = data,
        mode = config_values.get("inventor_vector_type", "soft")
    )

    num_archetypes = int(config_values.get("num_archetypes", 3))

    inventor_arche_df = inventor_archetype_memberships(
        inventor_skill_df,
        n_archetypes=num_archetypes,
        random_state=config_values.get("random_seed", 42),
        backend="numpy",
        init="uniform",
    )

    print(1)

    # TODO build the table Inventors X Skills


if __name__ == '__main__':
    main()