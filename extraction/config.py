"""Simple YAML config with dot-access. Replaces Hydra."""

import yaml
from pathlib import Path


class Config(dict):
    """Dict with attribute access: cfg.key instead of cfg['key']."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            if isinstance(val, dict):
                self[key] = Config(val)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, val):
        self[key] = val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_config(path: str = None) -> Config:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
