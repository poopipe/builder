"""create and stamp sculpt correction layers (metres over stack below)"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from builder.heightfield.brushes.brush_types import (
    brush_falloff,
    brush_falloff_flatten,
)
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    SculptHeightMode,
    SculptHeightSource,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapAsset,
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
    register_heightmap_asset,
)
from builder.heightfield.paint import (
    paint_resolution_for_heightfield,
    replace_heightmap_buffer,
    touch_heightmap_buffer,
    unique_paint_id,
)
from builder.heightfield.sample import sample_height_below

# ignore near-zero deltas when baking absolute coverage
sculpt_delta_coverage_epsilon: float = 1e-6


def roi_neighborhood_average(
    samples: list[float], width: int, height: int, x: int, y: int
) -> float:
    """3x3 mean over a dense roi scratch; out-of-bounds neighbors skipped"""
    total: float = 0.0
    count: int = 0
    dy: int
    for dy in (-1, 0, 1):
        dx: int
        for dx in (-1, 0, 1):
            nx: int = x + dx
            ny: int = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            total += samples[ny * width + nx]
            count += 1
    if count == 0:
        return 0.0
    return total / float(count)


def create_sculpt_heightmap(
    catalog: HeightmapCatalog,
    *,
    label: str,
    width: int,
    height: int,
    cache_dir: Path,
    stem: str = "sculpt",
    fill: float = 0.0,
) -> HeightmapId:
    """register a blank f32 sculpt raster and warm a filled buffer"""
    heightmap_id: HeightmapId = unique_paint_id(catalog, stem)
    cache_path: Path = (cache_dir / f"{heightmap_id.name}.hraw").resolve()
    source_path: Path = cache_path
    from builder.heightfield.heightmap_catalog import heightmap_buffer_empty
    from builder.heightfield.sparse_height import set_sample

    buffer: HeightmapBuffer
    if abs(fill) <= 1e-12:
        buffer = heightmap_buffer_empty(width, height)
    else:
        buffer = heightmap_buffer_from_samples(
            width, height, [fill] * (width * height)
        )
        # keep bounds honest when every texel is a constant fill
        y: int
        for y in range(min(1, height)):
            x: int
            for x in range(min(1, width)):
                set_sample(buffer, x, y, fill)
    asset: HeightmapAsset = HeightmapAsset(
        heightmap_id=heightmap_id,
        label=label,
        source_path=str(source_path),
        cache_path=str(cache_path),
        sample_type=HeightSampleType.f32,
        width=width,
        height=height,
    )
    register_heightmap_asset(catalog, asset)
    catalog.buffers[heightmap_id] = buffer
    catalog.paint_revisions[heightmap_id] = 0
    return heightmap_id


def append_sculpt_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    *,
    cache_dir: Path,
    label: str = "Sculpt",
    mode: SculptHeightMode = SculptHeightMode.delta,
) -> Heightfield:
    """append a new blank sculpt correction layer"""
    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    heightmap_id: HeightmapId = create_sculpt_heightmap(
        catalog,
        label=label,
        width=width,
        height=height,
        cache_dir=cache_dir,
        stem="sculpt",
        fill=0.0,
    )
    coverage_id: HeightmapId | None = None
    if mode is SculptHeightMode.absolute:
        coverage_id = create_sculpt_heightmap(
            catalog,
            label=f"{label} Coverage",
            width=width,
            height=height,
            cache_dir=cache_dir,
            stem="sculpt_cov",
            fill=0.0,
        )
    layer: HeightEditLayer = HeightEditLayer(
        name=label,
        height=SculptHeightSource(
            heightmap_id=heightmap_id,
            mode=mode,
            coverage_id=coverage_id,
        ),
        amplitude=1.0,
        offset=0.0,
    )
    return replace(
        heightfield,
        edit_layers=heightfield.edit_layers + (layer,),
    )


def convert_sculpt_buffers(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layer_index: int,
    source: SculptHeightSource,
    mode: SculptHeightMode,
    *,
    cache_dir: Path,
) -> SculptHeightSource:
    """rewrite sculpt buffers so composite height is preserved under the new mode

    delta → absolute: height = below + delta; coverage 1 only where |delta| > eps
    absolute → delta: delta = coverage * (height - below)
    """
    from builder.heightfield.heightmap_catalog import materialize_heightmap_samples

    height_buf: HeightmapBuffer | None = catalog.buffers.get(source.heightmap_id)
    width: int
    height: int
    if height_buf is not None:
        width = height_buf.width
        height = height_buf.height
        old_samples: list[float] = materialize_heightmap_samples(height_buf)
    else:
        width, height = paint_resolution_for_heightfield(catalog, heightfield)
        old_samples = [0.0] * (width * height)
    new_heights: list[float] = [0.0] * (width * height)
    coverage_id: HeightmapId | None = source.coverage_id
    y: int
    x: int
    match mode:
        case SculptHeightMode.absolute:
            if coverage_id is None:
                coverage_id = create_sculpt_heightmap(
                    catalog,
                    label=f"{heightfield.edit_layers[layer_index].name} Coverage",
                    width=width,
                    height=height,
                    cache_dir=cache_dir,
                    stem="sculpt_cov",
                    fill=0.0,
                )
            new_coverage: list[float] = [0.0] * (width * height)
            for y in range(height):
                for x in range(width):
                    u: float = (float(x) + 0.5) / float(width)
                    v: float = (float(y) + 0.5) / float(height)
                    below: float = sample_height_below(
                        catalog, heightfield, layer_index, u, v
                    )
                    index: int = y * width + x
                    delta: float = old_samples[index]
                    new_heights[index] = below + delta
                    # untouched deltas stay live pass-through
                    if abs(delta) > sculpt_delta_coverage_epsilon:
                        new_coverage[index] = 1.0
            replace_heightmap_buffer(
                catalog, source.heightmap_id, width, height, new_heights
            )
            replace_heightmap_buffer(
                catalog, coverage_id, width, height, new_coverage
            )
            return SculptHeightSource(
                heightmap_id=source.heightmap_id,
                mode=SculptHeightMode.absolute,
                coverage_id=coverage_id,
            )
        case SculptHeightMode.delta:
            old_coverage: list[float]
            if (
                coverage_id is not None
                and catalog.buffers.get(coverage_id) is not None
            ):
                cov_buf: HeightmapBuffer = catalog.buffers[coverage_id]
                if cov_buf.width == width and cov_buf.height == height:
                    old_coverage = materialize_heightmap_samples(cov_buf)
                else:
                    old_coverage = [0.0] * (width * height)
            else:
                old_coverage = [0.0] * (width * height)
            for y in range(height):
                for x in range(width):
                    u = (float(x) + 0.5) / float(width)
                    v = (float(y) + 0.5) / float(height)
                    below = sample_height_below(
                        catalog, heightfield, layer_index, u, v
                    )
                    index = y * width + x
                    cov: float = max(0.0, min(1.0, old_coverage[index]))
                    new_heights[index] = cov * (old_samples[index] - below)
            replace_heightmap_buffer(
                catalog, source.heightmap_id, width, height, new_heights
            )
            return SculptHeightSource(
                heightmap_id=source.heightmap_id,
                mode=SculptHeightMode.delta,
                coverage_id=None,
            )


def set_sculpt_layer_mode(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layer_index: int,
    mode: SculptHeightMode,
    *,
    cache_dir: Path,
) -> Heightfield:
    """switch sculpt mode; convert buffers so the composited surface is preserved"""
    if layer_index < 0 or layer_index >= len(heightfield.edit_layers):
        return heightfield
    layer: HeightEditLayer = heightfield.edit_layers[layer_index]
    match layer.height:
        case SculptHeightSource() as source:
            pass
        case _:
            return heightfield
    if source.mode is mode:
        return heightfield
    converted: SculptHeightSource = convert_sculpt_buffers(
        catalog,
        heightfield,
        layer_index,
        source,
        mode,
        cache_dir=cache_dir,
    )
    layers: list[HeightEditLayer] = list(heightfield.edit_layers)
    layers[layer_index] = replace(layer, height=converted)
    return replace(heightfield, edit_layers=tuple(layers))


def stamp_sculpt_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layer_index: int,
    brush_kind: str,
    center_u: float,
    center_v: float,
    radius_u: float,
    radius_v: float,
    intensity_metres: float,
    flatten_target: float | None,
    *,
    cache_dir: Path | None = None,
) -> tuple[bool, Heightfield]:
    """apply one metre brush stamp; flatten promotes delta layers to absolute+coverage"""
    if layer_index < 0 or layer_index >= len(heightfield.edit_layers):
        return (False, heightfield)
    layer: HeightEditLayer = heightfield.edit_layers[layer_index]
    match layer.height:
        case SculptHeightSource() as source:
            pass
        case _:
            return (False, heightfield)
    is_flatten_kind: bool = brush_kind in (
        "flatten",
        "flatten_up",
        "flatten_down",
        "flatten_absolute",
    )
    # flatten replaces the live stack below; absolute+coverage avoids baking
    # an inverse of soft features underneath (eg. vector sculpt falloff)
    if is_flatten_kind and source.mode is SculptHeightMode.delta:
        resolve_cache: Path = (
            Path.cwd() / "heightmaps" if cache_dir is None else cache_dir
        )
        heightfield = set_sculpt_layer_mode(
            catalog,
            heightfield,
            layer_index,
            SculptHeightMode.absolute,
            cache_dir=resolve_cache,
        )
        layer = heightfield.edit_layers[layer_index]
        match layer.height:
            case SculptHeightSource() as source:
                pass
            case _:
                return (False, heightfield)
    from builder.heightfield.brushes.brush_types import BrushStrokeContext
    from builder.heightfield.sparse_height import get_sample, set_sample

    height_buf: HeightmapBuffer | None = catalog.buffers.get(source.heightmap_id)
    if height_buf is None:
        return (False, heightfield)
    width: int = height_buf.width
    height: int = height_buf.height
    cov_buf: HeightmapBuffer | None = None
    if (
        source.mode is SculptHeightMode.absolute
        and source.coverage_id is not None
    ):
        cov_buf = catalog.buffers.get(source.coverage_id)
        if cov_buf is None or cov_buf.width != width or cov_buf.height != height:
            return (False, heightfield)

    context: BrushStrokeContext = BrushStrokeContext(
        center_u=center_u,
        center_v=center_v,
        radius_u=radius_u,
        radius_v=radius_v,
        intensity=intensity_metres,
        flatten_target=flatten_target,
    )
    radius_px_u: float = context.radius_u * float(width)
    radius_px_v: float = context.radius_v * float(height)
    cx: float = context.center_u * float(width)
    cy: float = context.center_v * float(height)
    x0: int = max(0, int(cx - radius_px_u) - 1)
    x1: int = min(width - 1, int(cx + radius_px_u) + 1)
    y0: int = max(0, int(cy - radius_px_v) - 1)
    y1: int = min(height - 1, int(cy + radius_px_v) + 1)
    strength: float = max(0.0, min(1.0, abs(float(intensity_metres))))
    if is_flatten_kind:
        if flatten_target is None:
            return (False, heightfield)
        strength = max(0.0, min(1.0, float(intensity_metres)))
    if brush_kind == "smooth":
        strength = max(0.0, min(1.0, float(intensity_metres)))

    needs_below: bool = (
        source.mode is SculptHeightMode.absolute
        or is_flatten_kind
        or brush_kind == "smooth"
    )
    # smooth needs a 1-texel halo so 3x3 averages stay inside the roi
    snap_pad: int = 1 if brush_kind == "smooth" else 0
    snap_x0: int = max(0, x0 - snap_pad)
    snap_x1: int = min(width - 1, x1 + snap_pad)
    snap_y0: int = max(0, y0 - snap_pad)
    snap_y1: int = min(height - 1, y1 + snap_pad)
    roi_w: int = snap_x1 - snap_x0 + 1
    roi_h: int = snap_y1 - snap_y0 + 1
    below_roi: list[float] | None = None
    visible_roi: list[float] | None = None
    if needs_below:
        below_roi = [0.0] * (roi_w * roi_h)
        if brush_kind == "smooth":
            visible_roi = [0.0] * (roi_w * roi_h)
        sy: int
        for sy in range(snap_y0, snap_y1 + 1):
            sx: int
            for sx in range(snap_x0, snap_x1 + 1):
                su: float = (float(sx) + 0.5) / float(width)
                sv: float = (float(sy) + 0.5) / float(height)
                sbelow: float = sample_height_below(
                    catalog, heightfield, layer_index, su, sv
                )
                roi_index: int = (sy - snap_y0) * roi_w + (sx - snap_x0)
                below_roi[roi_index] = sbelow
                if visible_roi is not None:
                    stored: float = get_sample(height_buf, sx, sy)
                    match source.mode:
                        case SculptHeightMode.delta:
                            visible_roi[roi_index] = sbelow + stored
                        case SculptHeightMode.absolute:
                            assert cov_buf is not None
                            scov: float = max(
                                0.0, min(1.0, get_sample(cov_buf, sx, sy))
                            )
                            visible_roi[roi_index] = (
                                sbelow * (1.0 - scov) + stored * scov
                            )

    y: int
    for y in range(y0, y1 + 1):
        x: int
        for x in range(x0, x1 + 1):
            u: float = (float(x) + 0.5) / float(width)
            v: float = (float(y) + 0.5) / float(height)
            w: float = (
                brush_falloff_flatten(u, v, context)
                if is_flatten_kind
                else brush_falloff(u, v, context)
            )
            if w <= 0.0:
                continue
            roi_index = (y - snap_y0) * roi_w + (x - snap_x0)
            below: float = 0.0
            if below_roi is not None:
                below = below_roi[roi_index]
            match source.mode:
                case SculptHeightMode.delta:
                    current: float = get_sample(height_buf, x, y)
                    if brush_kind == "add":
                        set_sample(
                            height_buf, x, y, current + intensity_metres * w
                        )
                    elif brush_kind == "subtract":
                        set_sample(
                            height_buf, x, y, current - intensity_metres * w
                        )
                    elif is_flatten_kind:
                        assert flatten_target is not None
                        visible: float = below + current
                        if brush_kind == "flatten_up" and visible >= flatten_target:
                            continue
                        if brush_kind == "flatten_down" and visible <= flatten_target:
                            continue
                        desired: float = flatten_target - below
                        mix: float = w * strength
                        set_sample(
                            height_buf,
                            x,
                            y,
                            current + (desired - current) * mix,
                        )
                    elif brush_kind == "smooth":
                        assert visible_roi is not None
                        avg: float = roi_neighborhood_average(
                            visible_roi,
                            roi_w,
                            roi_h,
                            x - snap_x0,
                            y - snap_y0,
                        )
                        mix = w * strength
                        target_delta: float = avg - below
                        set_sample(
                            height_buf,
                            x,
                            y,
                            current + (target_delta - current) * mix,
                        )
                case SculptHeightMode.absolute:
                    assert cov_buf is not None
                    cov: float = max(0.0, min(1.0, get_sample(cov_buf, x, y)))
                    stored = get_sample(height_buf, x, y)
                    ref: float = below * (1.0 - cov) + stored * cov
                    if brush_kind == "add":
                        set_sample(height_buf, x, y, ref + intensity_metres * w)
                        set_sample(cov_buf, x, y, cov + (1.0 - cov) * w)
                    elif brush_kind == "subtract":
                        set_sample(height_buf, x, y, ref - intensity_metres * w)
                        set_sample(cov_buf, x, y, cov + (1.0 - cov) * w)
                    elif is_flatten_kind:
                        assert flatten_target is not None
                        if brush_kind == "flatten_up" and ref >= flatten_target:
                            continue
                        if brush_kind == "flatten_down" and ref <= flatten_target:
                            continue
                        mix = w * strength
                        set_sample(
                            height_buf,
                            x,
                            y,
                            ref + (flatten_target - ref) * mix,
                        )
                        set_sample(cov_buf, x, y, cov + (1.0 - cov) * mix)
                    elif brush_kind == "smooth":
                        assert visible_roi is not None
                        avg = roi_neighborhood_average(
                            visible_roi,
                            roi_w,
                            roi_h,
                            x - snap_x0,
                            y - snap_y0,
                        )
                        mix = w * strength
                        set_sample(height_buf, x, y, ref + (avg - ref) * mix)
                        set_sample(cov_buf, x, y, cov + (1.0 - cov) * mix)

    touch_heightmap_buffer(catalog, source.heightmap_id)
    if cov_buf is not None and source.coverage_id is not None:
        touch_heightmap_buffer(catalog, source.coverage_id)
    return (True, heightfield)
