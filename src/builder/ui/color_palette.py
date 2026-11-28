"""load and resolve the shared color palette from json"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pyray import Color


def color_palette_path() -> Path:
    """default palette json next to other application assets"""
    return Path(__file__).resolve().parent.parent / "assets" / "color_palette.json"


def load_color_palette(path: Path | None = None) -> tuple[Color, ...]:
    """load rgb triples from json; raises on invalid data"""
    file_path: Path = color_palette_path() if path is None else path
    raw: Any = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("color palette must be a non-empty array")
    colors: list[Color] = []
    entry: Any
    for entry in raw:
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or not all(
                isinstance(channel, int) and not isinstance(channel, bool)
                for channel in entry
            )
        ):
            raise ValueError("color palette entries must be [r, g, b] ints")
        r: int
        g: int
        b: int
        r, g, b = entry
        if min(r, g, b) < 0 or max(r, g, b) > 255:
            raise ValueError("color palette channels must be 0..255")
        colors.append(Color(r, g, b, 255))
    return tuple(colors)


@lru_cache(maxsize=1)
def default_color_palette() -> tuple[Color, ...]:
    """cached default palette from assets"""
    return load_color_palette()


def palette_color(
    index: int | None,
    palette: tuple[Color, ...] | None = None,
) -> Color | None:
    """resolve a palette index, or none when unset/out of range"""
    if index is None:
        return None
    colors: tuple[Color, ...] = (
        default_color_palette() if palette is None else palette
    )
    if index < 0 or index >= len(colors):
        return None
    return colors[index]
