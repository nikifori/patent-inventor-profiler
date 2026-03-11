'''
@File    :   pipeline.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
'''
import argparse
import copy
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from utils import load_config, override_config_values
from pipeline_utils import (
    load_data, 
    Link2Skill_Mapping,
    build_inventor_skill_df,
    inventor_archetype_memberships,
    patents_to_long_with_all_cols_explode
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

    # Checks ---------------------------------------------------------
    num_archetypes = int(config_values.get("num_archetypes", 3))
    iter_per_num_archetypes = config_values.get("iter_per_num_archetypes", 1)
    alternative_random_seeds = config_values.get("alternative_random_seeds", [])

    if num_archetypes == -1 and iter_per_num_archetypes > 1:
        assert len(alternative_random_seeds) >=  iter_per_num_archetypes - 1, "Not enough alternative random seeds provided."

    # Initializations ---------------------------------------------------------
    skill_extractor = SkillExtractor(
        model=config_values.get("model", "all-MiniLM-L6-v2"),
        skills_threshold=config_values.get("model_skill_threshold", 0.6),
        device = "cuda" if config_values.get("device", None)=="cuda" and torch.cuda.is_available() else "cpu",
    )

    link2skill_mapper = Link2Skill_Mapping(csv_path=rf'{config_values.get("link2skill_mapping_file", None)}')
    # -------------------------------------------------------------------------

    batch_size = config_values.get("batch_size", 32)
    for i in tqdm(range(0, len(data), batch_size)):
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
    
    # filter out skills with similarity < config_values.get("actual_skill_threshold", None)
    if config_values.get("actual_skill_threshold", None) is not None:
        for patent in data:
            valid_idx = [k for k, (_, score) in enumerate(patent['skill_links']) if score > config_values.get("actual_skill_threshold", None)]
            patent['skill_links'] = [patent['skill_links'][k] for k in valid_idx]
            patent['skill_labels'] = [patent['skill_labels'][k] for k in valid_idx]

    # optionally save patent_X_skills_csv
    if config_values.get("save_patent_X_skills_csv", False):
        patent_X_skills_folder = Path(config_values.get("output_dir", "./output")).resolve()
        patent_X_skills_folder.mkdir(parents=True, exist_ok=True)

        patent_X_skills_file = patent_X_skills_folder / "patent_X_skills.csv"
        patent_X_inventor_skills_file = patent_X_skills_folder / "patent_X_inventor_skills.csv"

        # build long df (1 row per (patent, skill) with all patent cols duplicated)
        df_long_patent_skill = patents_to_long_with_all_cols_explode(
            data,
            keep_link=False,
            drop_llm_cols=True,
            explode_inventors=False,
            inventor_field="Inventors",
            inventor_sep=";;",
            drop_empty_patents=True,
        )
        df_long_patent_skill.to_csv(patent_X_skills_file, index=False)

        # build long df (1 row per (patent, inventor, skill) with all patent cols duplicated)
        df_long_patent_inventor_skill = patents_to_long_with_all_cols_explode(
            data,
            keep_link=False,
            drop_llm_cols=True,
            explode_inventors=True,
            inventor_field="Inventors",
            inventor_sep=";;",
            drop_empty_patents=True,
        )
        df_long_patent_inventor_skill.to_csv(patent_X_inventor_skills_file, index=False)

        print(f"[INFO] Saved long Patent x Skill CSV to: {patent_X_skills_file}")
        print(f"[INFO] Saved long Patent x Inventor x Skill CSV to: {patent_X_inventor_skills_file}")

    # build inventor skill df
    inventor_skill_df = build_inventor_skill_df(
        data = data,
        mode = config_values.get("inventor_vector_type", "soft")
    )

    cli_overrides = {
        key: val
        for key, val in vars(args).items()
        if key != "config" and val is not None
    }
    experiment_metadata = {
        "config_path": str(yaml_file_path),
        "effective_config": copy.deepcopy(config_values),
        "cli_overrides": cli_overrides,
        "input_data": {
            "data_csv_path": config_values.get("data_csv_path", None),
            "num_patent_records": len(data),
            "inventor_matrix_shape": list(inventor_skill_df.shape),
        },
    }

    inventor_arche_df = inventor_archetype_memberships(
        inventor_skill_df,
        n_archetypes=num_archetypes,
        random_state=config_values.get("random_seed", 42),
        max_k=config_values.get("max_k", 20),
        n_init=config_values.get("n_init", 1),
        alternative_random_seeds=alternative_random_seeds,
        iter_per_num_archetypes=iter_per_num_archetypes,
        method=config_values.get("method", "nnls"),
        backend=config_values.get("backend", "numpy"),
        init=config_values.get("init", "uniform"),
        max_iter=config_values.get("max_iter", 500),
        tol=config_values.get("tol", 1e-4),
        output_dir=config_values.get("output_dir", "./output"),
        experiment_metadata=experiment_metadata,
        save_repro_bundle=config_values.get("save_repro_bundle", True),
        repro_subdir=config_values.get("repro_subdir", "archetype_runs"),
    )

    print(1)


if __name__ == '__main__':
    main()
