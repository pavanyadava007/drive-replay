"""Where things live. Everything is relative to the repository root unless an environment variable says otherwise."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    # bazel run sets BUILD_WORKSPACE_DIRECTORY; plain python -m runs from the checkout
    env = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
CATALOG = ROOT / "recordings" / "catalog.json"
CONFIGS = ROOT / "configs"
BUGS = ROOT / "bugs"
GOLDEN = ROOT / "ci" / "golden"
SUITE = ROOT / "ci" / "suite.toml"
DATA = Path(os.environ.get("DRIVE_REPLAY_DATA", ROOT / "data" / "recordings"))
DROPS = Path(os.environ.get("DRIVE_REPLAY_DROPS", ROOT / ".drops"))
RUNS = Path(os.environ.get("DRIVE_REPLAY_RUNS", ROOT / "runs"))
