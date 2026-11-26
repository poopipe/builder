"""load kit width catalogue written by tools.build_road_kit_widths"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def road_kit_widths_path() -> Path:
    """package path for road_kit_widths.json"""
    return Path(__file__).with_name("road_kit_widths.json")


def load_kit_widths(path: Path | None = None) -> dict[str, float]:
    """kit name → width_m; skips kits with no measured width"""
    json_path: Path = path if path is not None else road_kit_widths_path()
    if not json_path.is_file():
        raise FileNotFoundError(
            f"road kit widths catalogue missing: {json_path} "
            "(run python -m tools.build_road_kit_widths)"
        )
    raw: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("road_kit_widths.json root must be an object")
    kits_raw: Any = raw.get("kits")
    if not isinstance(kits_raw, list):
        raise ValueError("road_kit_widths.json missing kits list")
    widths: dict[str, float] = {}
    entry: Any
    for entry in kits_raw:
        if not isinstance(entry, dict):
            continue
        name: Any = entry.get("kit")
        width: Any = entry.get("width_m")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(width, (int, float)):
            continue
        widths[name] = float(width)
    return widths
