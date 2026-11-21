"""sample height from base + masked edit layers"""

from __future__ import annotations

from math import floor

from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    HeightLayerSource,
    Heightfield,
    PaintHeightSource,
    ProceduralHeightSource,
    ProceduralHeightType,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
    ensure_heightmap_buffer,
)


def wrap01(value: float) -> float:
    """wrap into [0, 1)"""
    return value - floor(value)


def clamp01(value: float) -> float:
    """clamp to 0..1"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def sample_buffer(
    buffer: HeightmapBuffer, u: float, v: float, *, wrap: bool = False
) -> float:
    """bilinear sample signed float32 height at uv (clamp or wrap)"""
    if buffer.width <= 0 or buffer.height <= 0:
        return 0.0
    width: int = buffer.width
    height: int = buffer.height
    x0: int
    y0: int
    x1: int
    y1: int
    tx: float
    ty: float
    if wrap:
        uu: float = wrap01(u)
        vv: float = wrap01(v)
        # texel space [0, width) / [0, height); neighbors wrap
        x: float = uu * float(width)
        y: float = vv * float(height)
        x0 = int(floor(x)) % width
        y0 = int(floor(y)) % height
        x1 = (x0 + 1) % width
        y1 = (y0 + 1) % height
        tx = x - floor(x)
        ty = y - floor(y)
    else:
        uu = clamp01(u)
        vv = clamp01(v)
        # map onto pixel centers spanning the image
        x = uu * float(width - 1)
        y = vv * float(height - 1)
        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1 if x0 + 1 < width else x0
        y1 = y0 + 1 if y0 + 1 < height else y0
        tx = x - float(x0)
        ty = y - float(y0)
    i00: int = y0 * width + x0
    i10: int = y0 * width + x1
    i01: int = y1 * width + x0
    i11: int = y1 * width + x1
    s00: float = buffer.samples[i00]
    s10: float = buffer.samples[i10]
    s01: float = buffer.samples[i01]
    s11: float = buffer.samples[i11]
    s0: float = s00 * (1.0 - tx) + s10 * tx
    s1: float = s01 * (1.0 - tx) + s11 * tx
    return s0 * (1.0 - ty) + s1 * ty


def sample_source(
    catalog: HeightmapCatalog,
    source: HeightLayerSource | None,
    u: float,
    v: float,
    *,
    missing: float = 0.0,
    wrap: bool = False,
) -> float:
    """sample a height/mask source at uv"""
    match source:
        case None:
            return missing
        case CatalogHeightSource(heightmap_id=heightmap_id) | PaintHeightSource(
            heightmap_id=heightmap_id
        ):
            buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
                catalog, heightmap_id
            )
            if buffer is None:
                return missing
            return sample_buffer(buffer, u, v, wrap=wrap)
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            match procedural_type:
                case ProceduralHeightType.constant:
                    return float(constant_value)


def blend_height(
    accum: float,
    layer_value: float,
    mask: float,
    blend: HeightBlendMode,
) -> float:
    """combine accum with a masked layer contribution"""
    m: float = clamp01(mask)
    match blend:
        case HeightBlendMode.add:
            return accum + layer_value * m
        case HeightBlendMode.subtract:
            return accum - layer_value * m
        case HeightBlendMode.multiply:
            return accum * ((1.0 - m) + layer_value * m)
        case HeightBlendMode.max:
            blended: float = accum if accum > layer_value else layer_value
            return accum * (1.0 - m) + blended * m
        case HeightBlendMode.min:
            blended_min: float = accum if accum < layer_value else layer_value
            return accum * (1.0 - m) + blended_min * m


def sample_height(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    u: float,
    v: float,
) -> float:
    """world-space height at field uv from base + enabled edit layers"""
    base_sample: float = 0.0
    if heightfield.base is not None:
        base_id: HeightmapId = heightfield.base
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(catalog, base_id)
        if buffer is not None:
            base_sample = sample_buffer(buffer, u, v)
    accum: float = (
        float(heightfield.base_offset.value)
        + float(heightfield.base_amplitude.value) * base_sample
    )
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        if not layer.enabled:
            continue
        layer_u: float = u * float(layer.uv_scale_u) + float(layer.uv_offset_u)
        layer_v: float = v * float(layer.uv_scale_v) + float(layer.uv_offset_v)
        raw: float = sample_source(
            catalog, layer.height, layer_u, layer_v, missing=0.0, wrap=True
        )
        layer_value: float = float(layer.offset) + float(layer.amplitude) * raw
        mask: float
        if layer.mask is None:
            mask = 1.0
        else:
            mask = sample_source(
                catalog, layer.mask, layer_u, layer_v, missing=0.0, wrap=True
            )
        opacity: float = float(layer.opacity.value)
        accum = blend_height(
            accum, layer_value, mask * opacity, layer.blend
        )
    return accum


def source_sample_bounds(
    catalog: HeightmapCatalog, source: HeightLayerSource | None
) -> tuple[float, float]:
    """min/max sample range for a source (missing → 0)"""
    match source:
        case None:
            return (0.0, 0.0)
        case CatalogHeightSource(heightmap_id=heightmap_id) | PaintHeightSource(
            heightmap_id=heightmap_id
        ):
            buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
                catalog, heightmap_id
            )
            if buffer is None:
                return (0.0, 0.0)
            return (buffer.sample_min, buffer.sample_max)
        case ProceduralHeightSource(
            procedural_type=procedural_type, constant_value=constant_value
        ):
            match procedural_type:
                case ProceduralHeightType.constant:
                    value: float = float(constant_value)
                    return (value, value)


def scaled_bounds(
    sample_min: float, sample_max: float, amplitude: float, offset: float
) -> tuple[float, float]:
    """world bounds from sample range after amp/offset"""
    c0: float = offset + amplitude * sample_min
    c1: float = offset + amplitude * sample_max
    lo: float = c0 if c0 < c1 else c1
    hi: float = c1 if c0 < c1 else c0
    return (lo, hi)


def layer_height_bounds(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[float, float]:
    """estimate world-y min/max from base and edit-layer sample bounds"""
    base_lo: float
    base_hi: float
    if heightfield.base is None:
        base_lo, base_hi = (0.0, 0.0)
    else:
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
            catalog, heightfield.base
        )
        if buffer is None:
            base_lo, base_hi = (0.0, 0.0)
        else:
            base_lo, base_hi = (buffer.sample_min, buffer.sample_max)
    height_min: float
    height_max: float
    height_min, height_max = scaled_bounds(
        base_lo,
        base_hi,
        float(heightfield.base_amplitude.value),
        float(heightfield.base_offset.value),
    )
    layer: HeightEditLayer
    for layer in heightfield.edit_layers:
        if not layer.enabled:
            continue
        s0: float
        s1: float
        s0, s1 = source_sample_bounds(catalog, layer.height)
        lo: float
        hi: float
        lo, hi = scaled_bounds(s0, s1, float(layer.amplitude), float(layer.offset))
        opacity: float = clamp01(float(layer.opacity.value))
        match layer.blend:
            case HeightBlendMode.add:
                height_min += lo * opacity
                height_max += hi * opacity
            case HeightBlendMode.subtract:
                height_min -= hi * opacity
                height_max -= lo * opacity
            case HeightBlendMode.multiply:
                # conservative envelope over masked multiply
                candidates: list[float] = [
                    height_min,
                    height_max,
                    height_min * lo,
                    height_min * hi,
                    height_max * lo,
                    height_max * hi,
                ]
                height_min = min(candidates)
                height_max = max(candidates)
            case HeightBlendMode.max:
                height_min = height_min if height_min > lo else lo
                height_max = height_max if height_max > hi else hi
            case HeightBlendMode.min:
                height_min = height_min if height_min < lo else lo
                height_max = height_max if height_max < hi else hi
    if abs(height_max - height_min) < 1e-6:
        return (height_min - 0.5, height_max + 0.5)
    return (height_min, height_max)
