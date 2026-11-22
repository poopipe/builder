"""cubic bezier evaluation and arc-length densify"""

from __future__ import annotations

from math import hypot

from pyray import Vector3, vector3_add, vector3_length, vector3_scale, vector3_subtract

default_probe_count: int = 64
default_densify_step_metres: float = 1.0


def bezier_point(
    p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float
) -> Vector3:
    """evaluate a cubic bezier at t in [0, 1]"""
    u: float = 1.0 - t
    uu: float = u * u
    tt: float = t * t
    return vector3_add(
        vector3_add(
            vector3_scale(p0, uu * u),
            vector3_scale(p1, 3.0 * uu * t),
        ),
        vector3_add(
            vector3_scale(p2, 3.0 * u * tt),
            vector3_scale(p3, tt * t),
        ),
    )


def bezier_tangent(
    p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float
) -> Vector3:
    """evaluate cubic bezier derivative at t"""
    u: float = 1.0 - t
    return vector3_add(
        vector3_add(
            vector3_scale(vector3_subtract(p1, p0), 3.0 * u * u),
            vector3_scale(vector3_subtract(p2, p1), 6.0 * u * t),
        ),
        vector3_scale(vector3_subtract(p3, p2), 3.0 * t * t),
    )


def t_at_arc_length(
    cumulative: list[float],
    probe_ts: list[float],
    target: float,
) -> float:
    """return parameter t for a distance along the probed polyline"""
    total: float = cumulative[-1]
    if total <= 1e-12:
        return 0.0
    clamped: float = max(0.0, min(total, target))
    index: int
    for index in range(1, len(cumulative)):
        if cumulative[index] >= clamped:
            span: float = cumulative[index] - cumulative[index - 1]
            if span < 1e-12:
                return probe_ts[index]
            alpha: float = (clamped - cumulative[index - 1]) / span
            return (
                probe_ts[index - 1]
                + (probe_ts[index] - probe_ts[index - 1]) * alpha
            )
    return probe_ts[-1]


def probe_segment(
    p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, probe_count: int
) -> tuple[list[float], list[Vector3], list[float]]:
    """equal-t probes, positions, and cumulative arc lengths"""
    probe_ts: list[float] = [
        float(i) / float(probe_count) for i in range(probe_count + 1)
    ]
    probe: list[Vector3] = [
        bezier_point(p0, p1, p2, p3, t) for t in probe_ts
    ]
    cumulative: list[float] = [0.0]
    index: int
    for index in range(probe_count):
        cumulative.append(
            cumulative[-1]
            + vector3_length(vector3_subtract(probe[index + 1], probe[index]))
        )
    return (probe_ts, probe, cumulative)


def densify_segment_positions(
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    p3: Vector3,
    step_metres: float,
    *,
    include_start: bool,
    include_end: bool,
    probe_count: int = default_probe_count,
) -> list[Vector3]:
    """sample one cubic segment by arc length"""
    probe_ts: list[float]
    _probe: list[Vector3]
    cumulative: list[float]
    probe_ts, _probe, cumulative = probe_segment(
        p0, p1, p2, p3, probe_count
    )
    length: float = cumulative[-1]
    spacing: float = max(1e-6, step_metres)
    divisions: int = max(1, int(round(length / spacing)))
    samples: list[Vector3] = []
    first: int = 0 if include_start else 1
    last: int = divisions if include_end else divisions - 1
    sample: int
    for sample in range(first, last + 1):
        if sample == divisions:
            samples.append(p3)
            continue
        target: float = length * float(sample) / float(divisions)
        t: float = t_at_arc_length(cumulative, probe_ts, target)
        samples.append(bezier_point(p0, p1, p2, p3, t))
    return samples


def densify_segment_xz_attrs(
    p0: Vector3,
    p1: Vector3,
    p2: Vector3,
    p3: Vector3,
    scale0: float,
    scale1: float,
    step_metres: float,
    *,
    include_start: bool,
    include_end: bool,
    probe_count: int = default_probe_count,
) -> list[tuple[float, float, float, float]]:
    """sample (x, z, y, width_scale) along one cubic; y is Vector3.y"""
    probe_ts: list[float]
    _probe: list[Vector3]
    cumulative: list[float]
    probe_ts, _probe, cumulative = probe_segment(
        p0, p1, p2, p3, probe_count
    )
    length: float = cumulative[-1]
    spacing: float = max(1e-6, step_metres)
    divisions: int = max(1, int(round(length / spacing)))
    samples: list[tuple[float, float, float, float]] = []
    first: int = 0 if include_start else 1
    last: int = divisions if include_end else divisions - 1
    sample: int
    for sample in range(first, last + 1):
        t: float
        if sample == divisions:
            t = 1.0
            pos: Vector3 = p3
        else:
            target: float = length * float(sample) / float(divisions)
            t = t_at_arc_length(cumulative, probe_ts, target)
            pos = bezier_point(p0, p1, p2, p3, t)
        scale: float = scale0 + (scale1 - scale0) * t
        samples.append(
            (float(pos.x), float(pos.z), float(pos.y), float(scale))
        )
    return samples


def xz_distance(a: Vector3, b: Vector3) -> float:
    """horizontal distance between two points"""
    return hypot(float(b.x) - float(a.x), float(b.z) - float(a.z))
