"""session catalog of heightmap images for heightfields"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyray import (
    PixelFormat,
    get_image_color,
    image_format,
    load_image,
    unload_image,
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


@dataclass(frozen=True)
class HeightmapBuffer:
    """cpu luminance samples in row-major order, values in 0..1"""

    width: int
    height: int
    samples: tuple[float, ...]


@dataclass
class HeightmapCatalog:
    """heightmaps available for heightfield layers"""

    entries: dict[HeightmapId, HeightmapAsset] = field(default_factory=dict)
    buffers: dict[HeightmapId, HeightmapBuffer] = field(default_factory=dict)


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


def load_heightmap_buffer(path: Path) -> HeightmapBuffer:
    """load an image and convert to luminance samples"""
    image = load_image(str(path))
    try:
        image_format(image, PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
        width: int = int(image.width)
        height: int = int(image.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"heightmap has no pixels: {path}")
        samples: list[float] = []
        y: int
        x: int
        for y in range(height):
            for x in range(width):
                color = get_image_color(image, x, y)
                r: float = float(color.r) / 255.0
                g: float = float(color.g) / 255.0
                b: float = float(color.b) / 255.0
                samples.append((r + g + b) / 3.0)
        return HeightmapBuffer(width=width, height=height, samples=tuple(samples))
    finally:
        unload_image(image)


def ensure_heightmap_buffer(
    catalog: HeightmapCatalog, heightmap_id: HeightmapId
) -> HeightmapBuffer | None:
    """return a cached buffer, loading from source_path when needed"""
    existing: HeightmapBuffer | None = catalog.buffers.get(heightmap_id)
    if existing is not None:
        return existing
    asset: HeightmapAsset | None = catalog.entries.get(heightmap_id)
    if asset is None:
        return None
    buffer: HeightmapBuffer = load_heightmap_buffer(Path(asset.source_path))
    catalog.buffers[heightmap_id] = buffer
    return buffer
