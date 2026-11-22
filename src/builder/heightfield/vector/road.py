"""road ribbon height / half-width queries from path points"""

from __future__ import annotations

from collections.abc import Callable
from math import hypot

SurfaceAtXz = Callable[[float, float], float]


def nearest_segment_lerp_y(
    x: float,
    z: float,
    points_xzy: list[tuple[float, float, float]],
    closed: bool,
) -> float:
    """lerp third component at xz by projection onto the nearest path segment"""
    if len(points_xzy) == 0:
        return 0.0
    if len(points_xzy) == 1:
        return points_xzy[0][2]
    best_dist: float = 1e9
    best_y: float = points_xzy[0][2]
    segment_count: int = len(points_xzy) if closed else len(points_xzy) - 1
    i: int
    for i in range(segment_count):
        ax: float
        az: float
        ay: float
        bx: float
        bz: float
        by: float
        ax, az, ay = points_xzy[i]
        bx, bz, by = points_xzy[(i + 1) % len(points_xzy)]
        dx: float = bx - ax
        dz: float = bz - az
        length_sq: float = dx * dx + dz * dz
        t: float = 0.0
        if length_sq > 1e-12:
            t = max(
                0.0,
                min(1.0, ((x - ax) * dx + (z - az) * dz) / length_sq),
            )
        px: float = ax + dx * t
        pz: float = az + dz * t
        dist: float = hypot(x - px, z - pz)
        if dist < best_dist:
            best_dist = dist
            best_y = ay + (by - ay) * t
    return best_y


def ribbon_height_polyline(
    x: float,
    z: float,
    points_abs_xzy: list[tuple[float, float, float]],
    closed: bool,
) -> float:
    """absolute ribbon height for a polyline (y already surface+offset)"""
    return nearest_segment_lerp_y(x, z, points_abs_xzy, closed)


def ribbon_height_circle(
    _x: float,
    _z: float,
    points_abs_xzy: list[tuple[float, float, float]],
) -> float:
    """absolute ribbon height for a circle (constant from center point)"""
    if len(points_abs_xzy) == 0:
        return 0.0
    return points_abs_xzy[0][2]


def ribbon_height_rectangle(
    x: float,
    z: float,
    points_abs_xzy: list[tuple[float, float, float]],
) -> float:
    """absolute ribbon height for a rectangle from two opposite corners"""
    if len(points_abs_xzy) == 0:
        return 0.0
    if len(points_abs_xzy) == 1:
        return points_abs_xzy[0][2]
    ax: float
    az: float
    ay: float
    bx: float
    bz: float
    by: float
    ax, az, ay = points_abs_xzy[0]
    bx, bz, by = points_abs_xzy[1]
    span_x: float = bx - ax
    span_z: float = bz - az
    tx: float = 0.5
    tz: float = 0.5
    if abs(span_x) > 1e-8:
        tx = max(0.0, min(1.0, (x - ax) / span_x))
    if abs(span_z) > 1e-8:
        tz = max(0.0, min(1.0, (z - az) / span_z))
    t: float = 0.5 * (tx + tz)
    return ay + (by - ay) * t


def ribbon_half_width_polyline(
    x: float,
    z: float,
    points_xzw: list[tuple[float, float, float]],
    closed: bool,
) -> float:
    """half-width metres along a polyline (third component is half-width)"""
    return nearest_segment_lerp_y(x, z, points_xzw, closed)


def ribbon_half_width_circle(
    _x: float,
    _z: float,
    points_xzw: list[tuple[float, float, float]],
) -> float:
    """half-width metres for a circle (constant from center point)"""
    if len(points_xzw) == 0:
        return 0.0
    return points_xzw[0][2]


def ribbon_half_width_rectangle(
    x: float,
    z: float,
    points_xzw: list[tuple[float, float, float]],
) -> float:
    """half-width metres for a rectangle from two opposite corners"""
    return ribbon_height_rectangle(x, z, points_xzw)


def absolute_road_points(
    points_xzy: list[tuple[float, float, float]],
    surface_at_xz: SurfaceAtXz,
) -> list[tuple[float, float, float]]:
    """convert surface-relative offsets to absolute heights at each point"""
    result: list[tuple[float, float, float]] = []
    point: tuple[float, float, float]
    for point in points_xzy:
        x: float
        z: float
        offset: float
        x, z, offset = point
        result.append((x, z, float(surface_at_xz(x, z)) + offset))
    return result


def road_half_width_points(
    points_xzyw: list[tuple[float, float, float, float]],
    base_radius: float,
) -> list[tuple[float, float, float]]:
    """(x, z, half_width) from shape radius × per-point width_scale"""
    result: list[tuple[float, float, float]] = []
    point: tuple[float, float, float, float]
    for point in points_xzyw:
        x: float
        z: float
        _offset: float
        scale: float
        x, z, _offset, scale = point
        result.append((x, z, max(0.05, float(base_radius) * float(scale))))
    return result