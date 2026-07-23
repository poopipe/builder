"""Ground plane grid helpers."""

from __future__ import annotations

from pyray import draw_grid


def draw_ground_grid(slices: int = 20, spacing: float = 1.0) -> None:
    """Draw an XZ ground grid centred on the origin."""
    draw_grid(slices, spacing)
