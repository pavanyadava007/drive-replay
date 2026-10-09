"""Writes the data of the interactive AR-HUD demo page (site/hud.html) to site/data/hud/.

    python scripts/export_mpred_demo.py [--scenes ~/workspace/drive-replay-data/can_scenes_v2] [--out site/data/hud]

Runs the fitted predictors of configs/mpred/predictors.json (the benchmark code path, tools/mpred/predictors.py) on
the whole test split at noise levels 0, 1 and 2 (the seeded noise of the report's noise sweep), picks 10 scenes by
the objective rules in tools/mpred/demo_export.py (RULES) and writes one JSON file per scene plus index.json.
When runs/mpred/results.pkl (the cache of scripts/reproduce_mpred.py) exists, every exported per-scene mean is
compared with the benchmark's and the export stops on any difference above 1e-9. Deterministic: the same inputs
give byte-identical files. About 30 s.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mpred import data as D  # noqa: E402
from tools.mpred import demo_export as DE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=Path, default=D.DEFAULT_SCENES)
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "data" / "hud")
    ap.add_argument("--results", type=Path, default=ROOT / "runs" / "mpred" / "results.pkl")
    args = ap.parse_args()
    t0 = time.monotonic()
    index = DE.write_all(args.out, args.scenes, args.results)
    size = sum(f.stat().st_size for f in args.out.glob("*.json"))
    print(f"wrote {len(index['scenes'])} scenes to {args.out} ({size / 1e6:.1f} MB) in {time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
