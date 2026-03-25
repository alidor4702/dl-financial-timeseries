import yaml
from pathlib import Path


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_device(preference: str = "auto"):
    import torch
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
