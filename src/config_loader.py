import os
import yaml


def load_config(config_filename: str):
    """Load a yml config file from the config directory.
    #TODO: Add yml validation.

    Args:
        config_filename (str): name of the config file

    Returns:
        dict: config file as a dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", config_filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
