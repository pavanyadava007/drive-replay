"""Resolves the stack configuration for a run: defaults, then vehicle overrides, then command-line --set."""
from __future__ import annotations

import json

from tools.replay import paths


def resolve(vehicle: str, overrides: list[str] | None = None) -> tuple[dict, list[str]]:
    layers = ["configs/fcw_default.json"]
    cfg = {k: v for k, v in json.loads((paths.CONFIGS / "fcw_default.json").read_text()).items() if not k.startswith("_")}
    vfile = paths.CONFIGS / "vehicles" / f"{vehicle}.json"
    if vfile.exists():
        cfg.update({k: v for k, v in json.loads(vfile.read_text()).items() if not k.startswith("_")})
        layers.append(f"configs/vehicles/{vehicle}.json")
    for o in overrides or []:
        key, _, raw = o.partition("=")
        if not raw:
            raise ValueError(f"--set expects key=value, got '{o}'")
        cfg[key] = json.loads(raw)
        layers.append(f"--set {o}")
    return cfg, layers
