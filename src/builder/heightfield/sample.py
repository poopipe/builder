"""sample height from one or more heightmap layers"""

from __future__ import annotations

from builder.heightfield.heightfield_types import HeightmapLayer
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    ensure_heightmap_buffer,
)


def sample_buffer(buffer: HeightmapBuffer, u: float, v: float) -> float:
    """bilinear sample signed float32 height at uv in 0..1 (clamped)"""
    if buffer.width <= 0 or buffer.height <= 0:
        return 0.0
    uu: float = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    vv: float = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    # map onto pixel centers spanning the image
    x: float = uu * float(buffer.width - 1)
    y: float = vv * float(buffer.height - 1)
    x0: int = int(x)
    y0: int = int(y)
    x1: int = x0 + 1 if x0 + 1 < buffer.width else x0
    y1: int = y0 + 1 if y0 + 1 < buffer.height else y0
    tx: float = x - float(x0)
    ty: float = y - float(y0)
    i00: int = y0 * buffer.width + x0
    i10: int = y0 * buffer.width + x1
    i01: int = y1 * buffer.width + x0
    i11: int = y1 * buffer.width + x1
    s00: float = buffer.samples[i00]
    s10: float = buffer.samples[i10]
    s01: float = buffer.samples[i01]
    s11: float = buffer.samples[i11]
    s0: float = s00 * (1.0 - tx) + s10 * tx
    s1: float = s01 * (1.0 - tx) + s11 * tx
    return s0 * (1.0 - ty) + s1 * ty


def sample_height(
    catalog: HeightmapCatalog,
    layers: tuple[HeightmapLayer, ...],
    u: float,
    v: float,
) -> float:
    """sum layer contributions at field uv; missing buffers contribute 0

    value may be signed; world y = offset + amplitude * value per layer
    """
    total: float = 0.0
    layer: HeightmapLayer
    for layer in layers:
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
            catalog, layer.heightmap_id
        )
        value: float = 0.0 if buffer is None else sample_buffer(buffer, u, v)
        total += layer.offset + layer.amplitude * value
    return total


def layer_height_bounds(
    catalog: HeightmapCatalog, layers: tuple[HeightmapLayer, ...]
) -> tuple[float, float]:
    """estimate world-y min/max from buffer bounds and layer amp/offset"""
    if not layers:
        return (-0.5, 0.5)
    height_min: float = 0.0
    height_max: float = 0.0
    first: bool = True
    layer: HeightmapLayer
    for layer in layers:
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
            catalog, layer.heightmap_id
        )
        if buffer is None:
            lo: float = layer.offset
            hi: float = layer.offset
        else:
            a: float = layer.amplitude
            s0: float = buffer.sample_min
            s1: float = buffer.sample_max
            # amp may be negative; take both corners
            c0: float = layer.offset + a * s0
            c1: float = layer.offset + a * s1
            lo = c0 if c0 < c1 else c1
            hi = c1 if c0 < c1 else c0
        if first:
            height_min = lo
            height_max = hi
            first = False
        else:
            height_min += lo
            height_max += hi
    if first or abs(height_max - height_min) < 1e-6:
        return (height_min - 0.5, height_max + 0.5)
    return (height_min, height_max)
