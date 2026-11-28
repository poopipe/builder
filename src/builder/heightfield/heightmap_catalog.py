"""session catalog of heightmap images for heightfields"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from collections.abc import Sequence

from builder.heightfield.height_image_decode import (
    DecodeHeightImage,
    default_decode_height_image,
    f32_array_from_bytes,
    float_samples_from_decoded,
)
from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
)
from builder.heightfield.height_raw_cache import (
    default_height_raw_cache_path,
    read_height_raw_cache,
    write_height_raw_cache,
)
from builder.heightfield.sparse_height import (
    SparseHeightRaster,
    dense_from_sparse,
    sparse_from_dense,
)


@dataclass(frozen=True)
class HeightmapId:
    """stable handle into the heightmap catalog"""

    name: str


@dataclass(frozen=True)
class HeightmapAsset:
    """metadata for a heightmap available in the session"""

    heightmap_id: HeightmapId
    label: str
    source_path: str
    cache_path: str
    sample_type: HeightSampleType
    width: int
    height: int


# live height storage is sparse tiles (not a dense WxH list)
HeightmapBuffer = SparseHeightRaster


@dataclass
class HeightmapCatalog:
    """heightmaps available for heightfield layers"""

    entries: dict[HeightmapId, HeightmapAsset] = field(default_factory=dict)
    buffers: dict[HeightmapId, HeightmapBuffer] = field(default_factory=dict)
    # content revision for paint (and any mutated) buffers — gpu cache invalidation
    paint_revisions: dict[HeightmapId, int] = field(default_factory=dict)
    # pending gpu upload region (x0,x1,y0,y1); None value means full buffer
    paint_dirty_rects: dict[HeightmapId, tuple[int, int, int, int] | None] = field(
        default_factory=dict
    )
    # heightmaps awaiting a deferred paint_revisions bump (cpu already edited)
    deferred_paint_bumps: set[HeightmapId] = field(default_factory=set)
    # frames to wait before flushing deferred bumps (1 = next frame)
    gpu_flush_hold_frames: int = 0
    decode: DecodeHeightImage = field(default=default_decode_height_image)


def register_heightmap_asset(catalog: HeightmapCatalog, asset: HeightmapAsset) -> None:
    """add or replace heightmap metadata by id"""
    catalog.entries[asset.heightmap_id] = asset


def unload_heightmap_buffer(catalog: HeightmapCatalog, heightmap_id: HeightmapId) -> None:
    """drop a cached cpu buffer if present"""
    catalog.buffers.pop(heightmap_id, None)


def clear_heightmap_catalog(catalog: HeightmapCatalog) -> None:
    """remove every heightmap asset and buffer"""
    catalog.entries.clear()
    catalog.buffers.clear()
    catalog.paint_revisions.clear()
    catalog.paint_dirty_rects.clear()
    catalog.deferred_paint_bumps.clear()
    catalog.gpu_flush_hold_frames = 0


def heightmap_id_for_path(path: Path) -> HeightmapId:
    """derive a stable id from a file stem"""
    stem: str = path.stem.strip()
    if stem == "":
        stem = "heightmap"
    return HeightmapId(stem)


def heightmap_buffer_from_samples(
    width: int, height: int, samples: Sequence[float]
) -> HeightmapBuffer:
    """page dense samples into a sparse raster (import / bootstrap)"""
    return sparse_from_dense(width, height, samples)


def heightmap_buffer_empty(width: int, height: int) -> HeightmapBuffer:
    """empty sparse raster at logical resolution"""
    if width <= 0 or height <= 0:
        raise ValueError("heightmap has no pixels")
    return SparseHeightRaster(width=width, height=height)


def heightmap_buffer_from_decoded(image: DecodedHeightImage) -> HeightmapBuffer:
    """build a sparse raster from packed decoded samples"""
    if image.sample_type is HeightSampleType.f32:
        return heightmap_buffer_from_samples(
            image.width, image.height, f32_array_from_bytes(image.data)
        )
    return heightmap_buffer_from_samples(
        image.width, image.height, float_samples_from_decoded(image)
    )


def materialize_heightmap_samples(buffer: HeightmapBuffer) -> list[float]:
    """dense row-major samples for export only"""
    return dense_from_sparse(buffer)


def load_or_build_height_raw(
    source_path: Path,
    cache_path: Path,
    decode: DecodeHeightImage,
) -> DecodedHeightImage:
    """load .hraw when present, otherwise decode source and write the cache"""
    if cache_path.is_file():
        return read_height_raw_cache(cache_path)
    decoded: DecodedHeightImage = decode(source_path)
    write_height_raw_cache(cache_path, decoded)
    return decoded


def import_heightmap_image(
    catalog: HeightmapCatalog,
    source_path: Path,
    heightmap_id: HeightmapId,
    label: str,
    *,
    cache_path: Path | None = None,
) -> HeightmapAsset:
    """decode source, write .hraw cache, register asset, and warm the float buffer"""
    source: Path = source_path.resolve()
    cache: Path = (
        cache_path.resolve()
        if cache_path is not None
        else default_height_raw_cache_path(source)
    )
    decoded: DecodedHeightImage = catalog.decode(source)
    write_height_raw_cache(cache, decoded)
    asset: HeightmapAsset = HeightmapAsset(
        heightmap_id=heightmap_id,
        label=label,
        source_path=str(source),
        cache_path=str(cache),
        sample_type=decoded.sample_type,
        width=decoded.width,
        height=decoded.height,
    )
    register_heightmap_asset(catalog, asset)
    catalog.buffers[heightmap_id] = heightmap_buffer_from_decoded(decoded)
    return asset


def ensure_heightmap_buffer(
    catalog: HeightmapCatalog, heightmap_id: HeightmapId
) -> HeightmapBuffer | None:
    """return a cached sparse raster, loading via .hraw (rebuilding if needed)"""
    existing: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if existing is not None:
        return existing
    asset: HeightmapAsset | None = catalog.entries.get(heightmap_id)
    if asset is None:
        return None
    decoded: DecodedHeightImage = load_or_build_height_raw(
        Path(asset.source_path),
        Path(asset.cache_path),
        catalog.decode,
    )
    buffer: HeightmapBuffer = heightmap_buffer_from_decoded(decoded)
    catalog.buffers[heightmap_id] = buffer
    return buffer
