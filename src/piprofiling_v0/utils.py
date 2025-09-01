"""
@File    :   utils.py
@Time    :   07/2025
@Author  :   nikifori
@Version :   -
"""

import yaml
import argparse
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    import yaml

    if config_path is None:
        raise ValueError("Configuration path must be provided.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def override_config_values(
    args: argparse.Namespace = None, config_values: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Override configuration values with command line arguments.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        config_values (dict): Existing configuration values.
    Returns:
        dict: Updated configuration values.
    """
    if args is None or config_values is None:
        raise ValueError("Both args and config_values must be provided.")

    for key, val in vars(args).items():
        if key == "config" or val is None:
            continue
        config_values[key] = val


def main():
    pass


if __name__ == "__main__":
    main()
