"""Loads the parts of a recording the tooling needs: metadata, ego states and ground truth."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tools.replay.mcaplite import iter_messages


@dataclass
class RecordingView:
    meta: dict
    ego: list[dict] = field(default_factory=list)
    gt: list[dict] = field(default_factory=list)

    @property
    def start_us(self) -> int:
        return self.meta["t_us"]

    @property
    def end_us(self) -> int:
        return max(self.ego[-1]["t_us"], self.gt[-1]["t_us"] if self.gt else 0)


def load(path: str | Path) -> RecordingView:
    view = None
    ego, gt = [], []
    for topic, _, data in iter_messages(path, {"/meta", "/ego/state", "/gt/objects"}):
        m = json.loads(data)
        if topic == "/meta":
            view = m
        elif topic == "/ego/state":
            ego.append(m)
        else:
            gt.append(m)
    if view is None:
        raise ValueError(f"{path}: no /meta message")
    return RecordingView(view, ego, gt)
