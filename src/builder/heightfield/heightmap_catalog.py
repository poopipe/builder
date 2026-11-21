"""session catalog of heightmap images for heightfields"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from builder.heightfield.height_image_decode import (
    DecodeHeightImage,
    default_decode_height_image,
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


@dataclass(frozen=True)
class HeightmapBuffer:
    """cpu height samples in row-major order as signed float32 values"""

    width: int
    height: int
    samples: tuple[float, ...]
    sample_min: float
    sample_max: float


@dataclass
class HeightmapCatalog:
    """heightmaps available for heightfield layers"""

    entries: dict[HeightmapId, HeightmapAsset] = field(default_factory=dict)
    buffers: dict[HeightmapId, HeightmapBuffer] = field(default_factory=dict)
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


def heightmap_id_for_path(path: Path) -> HeightmapId:
    """derive a stable id from a file stem"""
    stem: str = path.stem.strip()
    if stem == "":
        stem = "heightmap"
    return HeightmapId(stem)


def heightmap_buffer_from_samples(
    width: int, height: int, samples: tuple[float, ...]
) -> HeightmapBuffer:
    """build a buffer with precomputed sample bounds"""
    if width <= 0 or height <= 0:
        raise ValueError("heightmap has no pixels")
    if len(samples) != width * height:
        raise ValueError("heightmap sample count does not match width*height")
    sample_min: float = min(samples) if samples else 0.0
    sample_max: float = max(samples) if samples else 0.0
    return HeightmapBuffer(
        width=width,
        height=height,
        samples=samples,
        sample_min=sample_min,
        sample_max=sample_max,
    )


def heightmap_buffer_from_decoded(image: DecodedHeightImage) -> HeightmapBuffer:
    """build a float buffer from packed decoded samples"""
    samples: list[float] = float_samples_from_decoded(image)
    return heightmap_buffer_from_samples(
        image.width, image.height, tuple(samples)
    )


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
    """return a cached float buffer, loading via .hraw (rebuilding if needed)"""
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
