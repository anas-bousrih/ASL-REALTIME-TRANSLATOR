from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class AppConfig:
    raw: Dict[str, Any]


def load_config(path: str = "configs/models.yaml") -> AppConfig:
    cfg_path = Path(path).resolve()
    repo_root = cfg_path.parent.parent

    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", {})
    for model_cfg in models.values():
        if not isinstance(model_cfg, dict):
            continue
        for key in ("checkpoint", "train_csv"):
            value = model_cfg.get(key)
            if not value:
                continue
            p = Path(value).expanduser()
            if not p.is_absolute():
                model_cfg[key] = str((repo_root / p).resolve())

    return AppConfig(raw=data)
