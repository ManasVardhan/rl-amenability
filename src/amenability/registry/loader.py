from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "models.yaml"


class RegistryError(Exception):
    """Raised when the model roster is malformed."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    params_b: float
    license: str
    role: str


def load_registry(path: Path | None = None) -> dict[str, ModelSpec]:
    path = path or DEFAULT_PATH
    raw = yaml.safe_load(path.read_text())
    out: dict[str, ModelSpec] = {}
    for entry in raw["models"]:
        spec = ModelSpec(**entry)
        if spec.key in out:
            raise RegistryError(f"duplicate model key: {spec.key}")
        out[spec.key] = spec
    return out
