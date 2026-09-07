"""2d value-noise displacement along a chosen axis"""

from __future__ import annotations

from math import floor

from pyray import Transform, Vector3, vector3_add

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    ParamField,
    ParamMap,
    Axis,
)
from builder.modifiers.modifier_types import ModifierSpec


def hash2(ix: int, iy: int, seed: int) -> float:
    """deterministic hash of a lattice point to [0, 1)"""
    n: int = (ix * 374761393 + iy * 668265263 + seed * 982451653) & 0x7FFFFFFF
    n = (n ^ (n >> 13)) * 1274126177
    return float(n & 0x7FFFFFFF) / float(0x7FFFFFFF)


def fade(t: float) -> float:
    """smoothstep fade for value-noise interpolation"""
    return t * t * (3.0 - 2.0 * t)


def value_noise_2d(x: float, y: float, seed: int) -> float:
    """2d value noise in roughly [-1, 1]"""
    x0: int = int(floor(x))
    y0: int = int(floor(y))
    fx: float = x - float(x0)
    fy: float = y - float(y0)
    u: float = fade(fx)
    v: float = fade(fy)
    n00: float = hash2(x0, y0, seed)
    n10: float = hash2(x0 + 1, y0, seed)
    n01: float = hash2(x0, y0 + 1, seed)
    n11: float = hash2(x0 + 1, y0 + 1, seed)
    nx0: float = n00 + (n10 - n00) * u
    nx1: float = n01 + (n11 - n01) * u
    # map [0,1] -> [-1,1]
    return (nx0 + (nx1 - nx0) * v) * 2.0 - 1.0


def sample_axes(displace_axis: int) -> tuple[int, int]:
    """return the two plane axes used to sample noise for a displace axis"""
    if displace_axis == 0:
        return 1, 2
    if displace_axis == 1:
        return 0, 2
    return 0, 1


def component(position: Vector3, axis: int) -> float:
    """return one axis component of a position"""
    if axis == 0:
        return float(position.x)
    if axis == 1:
        return float(position.y)
    return float(position.z)


def axis_offset(axis: int, amount: float) -> Vector3:
    """return a displacement vector along axis"""
    if axis == 0:
        return Vector3(amount, 0.0, 0.0)
    if axis == 1:
        return Vector3(0.0, amount, 0.0)
    return Vector3(0.0, 0.0, amount)


def noise_displace_apply(
    slots: list[GeneratedSlot],
    params: ParamMap,
    _: BuildContext,
) -> list[GeneratedSlot]:
    """displace each slot along axis by amplitude * value noise"""
    axis: int = max(0, min(2, int(params.get("axis", 1))))
    amplitude: float = float(params.get("amplitude", 0.5))
    scale: float = max(0.1, float(params.get("scale", 2.0)))
    seed: int = int(params.get("seed", 0))
    ax0: int
    ax1: int
    ax0, ax1 = sample_axes(axis)
    result: list[GeneratedSlot] = []
    slot: GeneratedSlot
    for slot in slots:
        position: Vector3 = slot.transform.translation
        sample_x: float = component(position, ax0) / scale
        sample_y: float = component(position, ax1) / scale
        noise: float = value_noise_2d(sample_x, sample_y, seed)
        offset: Vector3 = axis_offset(axis, amplitude * noise)
        result.append(
            GeneratedSlot(
                Transform(
                    vector3_add(position, offset),
                    slot.transform.rotation,
                    slot.transform.scale,
                ),
                slot.role,
            )
        )
    return result


noise_displace_defaults: ParamMap = {
    "axis": 1,
    "amplitude": 0.5,
    "scale": 2.0,
    "seed": 0,
}

noise_displace_spec: ModifierSpec = ModifierSpec(
    kind="noise_displace",
    label="Noise displace",
    fields=(
        ParamField("axis", "Axis", "enum", 1.0, options=Axis),
        ParamField(
            "amplitude", "Amplitude", "float", 0.1, minimum=-50.0, maximum=50.0
        ),
        ParamField("scale", "Scale", "float", 0.25, minimum=0.1, maximum=100.0),
        ParamField("seed", "Seed", "int", 1.0, minimum=0.0, maximum=1_000_000.0),
    ),
    defaults=noise_displace_defaults,
    apply=noise_displace_apply,
)
