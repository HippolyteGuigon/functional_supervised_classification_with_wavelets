from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def load_config() -> dict:
    """
    Load the project configuration from configs/config.yaml.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
