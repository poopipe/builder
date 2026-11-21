"""create and stamp paint heightmap layers"""

from __future__ import annotations

import struct
from pathlib import Path

from builder.heightfield.brushes.brush_types import BrushStrokeContext
from builder.heightfield.brushes.registry import get_brush_spec
from builder.heightfield.height_image_types import DecodedHeightImage, HeightSampleType
from builder.heightfield.heightmap_catalog import (
    HeightmapAsset,
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
    register_heightmap_asset,
)
from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    Heightfield,
    PaintHeightSource,
)


default_paint_resolution: int = 1024


def paint_revision(catalog: HeightmapCatalog, heightmap_id: HeightmapId) -> int:
    """content revision for a paint (or any) heightmap buffer"""
    return int(catalog.paint_revisions.get(heightmap_id, 0))


def bump_paint_revision(catalog: HeightmapCatalog, heightmap_id: HeightmapId) -> int:
    """increment and return the content revision"""
    next_rev: int = paint_revision(catalog, heightmap_id) + 1
    catalog.paint_revisions[heightmap_id] = next_rev
    return next_rev


def replace_heightmap_buffer(
    catalog: HeightmapCatalog,
    heightmap_id: HeightmapId,
    width: int,
    height: int,
    samples: list[float],
) -> HeightmapBuffer:
    """replace the cpu buffer and bump paint revision"""
    buffer: HeightmapBuffer = heightmap_buffer_from_samples(width, height, samples)
    catalog.buffers[heightmap_id] = buffer
    bump_paint_revision(catalog, heightmap_id)
    asset: HeightmapAsset | None = catalog.entries.get(heightmap_id)
    if asset is not None and (
        asset.width != width
        or asset.height != height
        or asset.sample_type is not HeightSampleType.f32
    ):
        catalog.entries[heightmap_id] = HeightmapAsset(
            heightmap_id=heightmap_id,
            label=asset.label,
            source_path=asset.source_path,
            cache_path=asset.cache_path,
            sample_type=HeightSampleType.f32,
            width=width,
            height=height,
        )
    return buffer


def touch_heightmap_buffer(
    catalog: HeightmapCatalog, heightmap_id: HeightmapId
) -> int:
    """bump content revision after in-place sample edits (no buffer rebuild)"""
    return bump_paint_revision(catalog, heightmap_id)


def decoded_f32_from_buffer(buffer: HeightmapBuffer) -> DecodedHeightImage:
    """pack float samples for .hraw write"""
    data: bytes = struct.pack(f"<{len(buffer.samples)}f", *buffer.samples)
    return DecodedHeightImage(
        width=buffer.width,
        height=buffer.height,
        sample_type=HeightSampleType.f32,
        data=data,
    )


def paint_resolution_for_heightfield(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[int, int]:
    """paint raster size: match base map when present, else default square"""
    if heightfield.base is not None:
        buffer: HeightmapBuffer | None = catalog.buffers.get(heightfield.base)
        if buffer is not None:
            return (buffer.width, buffer.height)
        asset: HeightmapAsset | None = catalog.entries.get(heightfield.base)
        if asset is not None:
            return (asset.width, asset.height)
    return (default_paint_resolution, default_paint_resolution)


def unique_paint_id(catalog: HeightmapCatalog, stem: str) -> HeightmapId:
    """allocate an unused paint heightmap id"""
    candidate: HeightmapId = HeightmapId(stem)
    if candidate not in catalog.entries:
        return candidate
    index: int = 1
    while True:
        candidate = HeightmapId(f"{stem}_{index}")
        if candidate not in catalog.entries:
            return candidate
        index += 1


def create_paint_heightmap(
    catalog: HeightmapCatalog,
    *,
    label: str,
    width: int,
    height: int,
    cache_dir: Path,
) -> HeightmapId:
    """register a blank f32 paint raster and warm a zero buffer"""
    heightmap_id: HeightmapId = unique_paint_id(catalog, "paint")
    cache_path: Path = (cache_dir / f"{heightmap_id.name}.hraw").resolve()
    # source_path mirrors cache for paint-only assets (no imported image)
    source_path: Path = cache_path
    samples: list[float] = [0.0] * (width * height)
    buffer: HeightmapBuffer = heightmap_buffer_from_samples(width, height, samples)
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


def append_paint_edit_layer(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    *,
    cache_dir: Path,
    label: str = "Paint",
) -> Heightfield:
    """append a new blank paint edit layer to the heightfield"""
    from dataclasses import replace

    width: int
    height: int
    width, height = paint_resolution_for_heightfield(catalog, heightfield)
    heightmap_id: HeightmapId = create_paint_heightmap(
        catalog, label=label, width=width, height=height, cache_dir=cache_dir
    )
    layer: HeightEditLayer = HeightEditLayer(
        name=label,
        height=PaintHeightSource(heightmap_id=heightmap_id),
    )
    return replace(
        heightfield,
        edit_layers=heightfield.edit_layers + (layer,),
    )


def stamp_paint_layer(
    catalog: HeightmapCatalog,
    heightmap_id: HeightmapId,
    brush_kind: str,
    brush_params: object,
    context: BrushStrokeContext,
) -> bool:
    """apply one brush stamp to a paint buffer; return true if stamped"""
    buffer: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if buffer is None:
        return False
    spec = get_brush_spec(brush_kind)
    spec.stamp(
        buffer.samples, buffer.width, buffer.height, context, brush_params
    )
    touch_heightmap_buffer(catalog, heightmap_id)
    return True
