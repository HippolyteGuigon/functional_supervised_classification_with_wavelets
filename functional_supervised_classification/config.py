import os
from pathlib import Path
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def config_path() -> Path:
    """Path to the active config file: ``$FSC_CONFIG`` if set, else configs/config.yaml."""
    return Path(os.environ.get("FSC_CONFIG", _DEFAULT_CONFIG_PATH))


def load_config() -> dict:
    """
    Load the project configuration.

    Reads ``configs/config.yaml`` by default, or the file pointed to by the
    ``FSC_CONFIG`` environment variable when set (used by the tests to run the
    pipeline on a throw-away configuration).

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    with open(config_path()) as f:
        return yaml.safe_load(f)
