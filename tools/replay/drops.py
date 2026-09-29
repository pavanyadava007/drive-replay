"""Build drops: versioned, stamped builds of the stack that runs are pinned to.

A drop is a directory holding replay_main plus drop.json (version, commit, sha256, build time). CI produces one
per build; locally `replay drop build <git-ref>` builds any tag or commit in a throwaway git worktree with the
same Bazel flags, so an old drop can be rebuilt when a bug report names it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tools.replay import paths
from tools.replay.catalog import ResolveError, sha256_file


@dataclass(frozen=True)
class Drop:
    name: str
    path: Path
    version: str
    commit: str
    sha256: str

    @property
    def binary(self) -> Path:
        return self.path / "replay_main"


def _load(d: Path) -> Drop:
    m = json.loads((d / "drop.json").read_text())
    return Drop(d.name, d, m["version"], m["commit"], m["sha256"])


def list_drops() -> list[Drop]:
    if not paths.DROPS.exists():
        return []
    ds = [_load(d) for d in paths.DROPS.iterdir() if (d / "drop.json").exists()]
    return sorted(ds, key=lambda d: json.loads((d.path / "drop.json").read_text())["built_at"])


def import_binary(binary: Path, dest_root: Path | None = None) -> Drop:
    """Registers an already built replay_main (for example a CI artifact) as a drop."""
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout.split()
    version, commit = out[0], out[1]
    root = dest_root or paths.DROPS
    d = root / f"v{version}-{commit}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, d / "replay_main")
    os.chmod(d / "replay_main", 0o755)
    meta = {"version": version, "commit": commit, "sha256": sha256_file(d / "replay_main"),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "drop.json").write_text(json.dumps(meta, indent=1) + "\n")
    return _load(d)


def build(ref: str) -> Drop:
    """Builds a git ref with `bazel build --config=drop` in a temporary worktree and registers it."""
    root = paths.ROOT
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, capture_output=True,
                            text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="drop-") as tmp:
        wt = Path(tmp) / "src"
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=root, check=True,
                       capture_output=True)
        try:
            subprocess.run(["bazel", "build", "--config=drop", "//stack:replay_main"], cwd=wt, check=True)
            binary = Path(subprocess.run(["bazel", "cquery", "--config=drop", "--output=files", "//stack:replay_main"],
                                         cwd=wt, capture_output=True, text=True, check=True).stdout.split()[0])
            drop = import_binary(wt / binary)
            subprocess.run(["bazel", "shutdown"], cwd=wt, capture_output=True, check=False)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=root, capture_output=True, check=False)
    return drop


def resolve(spec: str | None) -> Drop:
    """spec: None/'latest' (newest drop), a version ('0.2.0' or 'v0.2.0'), a commit prefix, or a drop directory."""
    if spec and Path(spec).is_dir() and (Path(spec) / "drop.json").exists():
        return _load(Path(spec))
    drops = list_drops()
    if not drops:
        raise ResolveError("no build drops yet. Build one with: replay drop build HEAD")
    if spec in (None, "latest"):
        return drops[-1]
    v = spec.removeprefix("v")
    hits = [d for d in drops if d.version == v or d.commit.startswith(spec) or d.name == spec]
    if not hits:
        raise ResolveError(f"no drop matches '{spec}'. Have: {', '.join(d.name for d in drops)}. "
                           f"Build it with: replay drop build <git-ref>")
    return hits[-1]
