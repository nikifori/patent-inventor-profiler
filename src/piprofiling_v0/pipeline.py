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
from pipeline_utils import load_data, Link2Skill_Mapping

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
        skills_threshold=config_values.get("skills_threshold", 0.6),
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
                link2skill_mapper.link2skill(link) for link in batch_skills[j]
            ]
    
    print(1)

    # TODO build the table Inventors X Skills


if __name__ == '__main__':
    main()