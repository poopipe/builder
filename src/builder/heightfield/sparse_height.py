"""sparse tiled height raster — live working set for all height maps"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import floor

# logical texels per tile edge; edits allocate only touched tiles
height_tile_size: int = 256


def page_count(extent: int, tile_size: int = height_tile_size) -> int:
    """number of pages covering extent texels"""
    if extent < 1:
        return 0
    return (extent + tile_size - 1) // tile_size


@dataclass
class SparseHeightRaster:
    """height samples stored only in allocated tiles

    empty (unallocated) texels read as 0
    """

    width: int
    height: int
    tile_size: int = height_tile_size
    # (page_x, page_y) -> row-major tile_size*tile_size floats
    tiles: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    sample_min: float = 0.0
    sample_max: float = 0.0

    @property
    def pages_x(self) -> int:
        """page count along x"""
        return page_count(self.width, self.tile_size)

    @property
    def pages_y(self) -> int:
        """page count along y"""
        return page_count(self.height, self.tile_size)


def empty_tile(tile_size: int = height_tile_size) -> list[float]:
    """zeroed tile storage"""
    return [0.0] * (tile_size * tile_size)


def ensure_tile(
    raster: SparseHeightRaster, page_x: int, page_y: int
) -> list[float]:
    """allocate a tile if missing; return its samples"""
    key: tuple[int, int] = (page_x, page_y)
    existing: list[float] | None = raster.tiles.get(key)
    if existing is not None:
        return existing
    created: list[float] = empty_tile(raster.tile_size)
    raster.tiles[key] = created
    return created


def drop_empty_tile(
    raster: SparseHeightRaster, page_x: int, page_y: int
) -> None:
    """remove a tile if every sample is ~0"""
    key: tuple[int, int] = (page_x, page_y)
    tile: list[float] | None = raster.tiles.get(key)
    if tile is None:
        return
    value: float
    for value in tile:
        if abs(value) > 1e-12:
            return
    del raster.tiles[key]


def get_sample(raster: SparseHeightRaster, x: int, y: int) -> float:
    """read one texel; unallocated → 0"""
    if x < 0 or y < 0 or x >= raster.width or y >= raster.height:
        return 0.0
    ts: int = raster.tile_size
    page_x: int = x // ts
    page_y: int = y // ts
    tile: list[float] | None = raster.tiles.get((page_x, page_y))
    if tile is None:
        return 0.0
    lx: int = x - page_x * ts
    ly: int = y - page_y * ts
    return tile[ly * ts + lx]


def set_sample(raster: SparseHeightRaster, x: int, y: int, value: float) -> None:
    """write one texel; allocates the tile"""
    if x < 0 or y < 0 or x >= raster.width or y >= raster.height:
        return
    ts: int = raster.tile_size
    page_x: int = x // ts
    page_y: int = y // ts
    was_empty: bool = not raster.tiles
    tile: list[float] = ensure_tile(raster, page_x, page_y)
    lx: int = x - page_x * ts
    ly: int = y - page_y * ts
    tile[ly * ts + lx] = value
    if was_empty:
        raster.sample_min = value
        raster.sample_max = value
        return
    if value < raster.sample_min:
        raster.sample_min = value
    if value > raster.sample_max:
        raster.sample_max = value


def add_sample(raster: SparseHeightRaster, x: int, y: int, delta: float) -> None:
    """add to one texel"""
    set_sample(raster, x, y, get_sample(raster, x, y) + delta)


def clear_raster(raster: SparseHeightRaster) -> None:
    """drop every allocated tile"""
    raster.tiles.clear()
    raster.sample_min = 0.0
    raster.sample_max = 0.0


def clear_raster_region(
    raster: SparseHeightRaster, x0: int, x1: int, y0: int, y1: int
) -> None:
    """zero samples in an inclusive-exclusive texel rectangle"""
    rx0: int = max(0, x0)
    rx1: int = min(raster.width, x1)
    ry0: int = max(0, y0)
    ry1: int = min(raster.height, y1)
    if rx0 >= rx1 or ry0 >= ry1:
        return
    ts: int = raster.tile_size
    page_x0: int = rx0 // ts
    page_x1: int = (rx1 - 1) // ts
    page_y0: int = ry0 // ts
    page_y1: int = (ry1 - 1) // ts
    py: int
    for py in range(page_y0, page_y1 + 1):
        px: int
        for px in range(page_x0, page_x1 + 1):
            tile: list[float] | None = raster.tiles.get((px, py))
            if tile is None:
                continue
            ox: int = px * ts
            oy: int = py * ts
            local_y0: int = max(ry0, oy) - oy
            local_y1: int = min(ry1, oy + ts) - oy
            local_x0: int = max(rx0, ox) - ox
            local_x1: int = min(rx1, ox + ts) - ox
            ly: int
            for ly in range(local_y0, local_y1):
                row: int = ly * ts
                lx: int
                for lx in range(local_x0, local_x1):
                    tile[row + lx] = 0.0
            drop_empty_tile(raster, px, py)


def pages_overlapping_texels(
    raster: SparseHeightRaster, x0: int, x1: int, y0: int, y1: int
) -> list[tuple[int, int]]:
    """page keys intersecting an inclusive-exclusive texel rect"""
    rx0: int = max(0, x0)
    rx1: int = min(raster.width, x1)
    ry0: int = max(0, y0)
    ry1: int = min(raster.height, y1)
    if rx0 >= rx1 or ry0 >= ry1:
        return []
    ts: int = raster.tile_size
    out: list[tuple[int, int]] = []
    py: int
    for py in range(ry0 // ts, (ry1 - 1) // ts + 1):
        px: int
        for px in range(rx0 // ts, (rx1 - 1) // ts + 1):
            out.append((px, py))
    return out


def recompute_sample_bounds(raster: SparseHeightRaster) -> None:
    """scan allocated tiles for min/max (logical extent only)"""
    if not raster.tiles:
        raster.sample_min = 0.0
        raster.sample_max = 0.0
        return
    sample_min: float = 0.0
    sample_max: float = 0.0
    first: bool = True
    ts: int = raster.tile_size
    key: tuple[int, int]
    tile: list[float]
    for key, tile in raster.tiles.items():
        px: int
        py: int
        px, py = key
        ox: int = px * ts
        oy: int = py * ts
        ly: int
        for ly in range(ts):
            gy: int = oy + ly
            if gy >= raster.height:
                break
            row: int = ly * ts
            lx: int
            for lx in range(ts):
                gx: int = ox + lx
                if gx >= raster.width:
                    break
                value: float = tile[row + lx]
                if first:
                    sample_min = value
                    sample_max = value
                    first = False
                elif value < sample_min:
                    sample_min = value
                elif value > sample_max:
                    sample_max = value
    raster.sample_min = sample_min
    raster.sample_max = sample_max


def sparse_from_dense(
    width: int, height: int, samples: Sequence[float]
) -> SparseHeightRaster:
    """page a dense row-major buffer into sparse tiles (import path)

    accepts list or array.array('f'); copies tile rows via slice assign;
    skips all-zero pages
    """
    if width < 1 or height < 1:
        raise ValueError("heightmap has no pixels")
    if len(samples) != width * height:
        raise ValueError("heightmap sample count does not match width*height")
    raster: SparseHeightRaster = SparseHeightRaster(width=width, height=height)
    ts: int = raster.tile_size
    sample_min: float = 0.0
    sample_max: float = 0.0
    first: bool = True
    py: int
    for py in range(page_count(height, ts)):
        oy: int = py * ts
        tile_h: int = min(ts, height - oy)
        px: int
        for px in range(page_count(width, ts)):
            ox: int = px * ts
            tile_w: int = min(ts, width - ox)
            any_nz: bool = False
            ly: int
            for ly in range(tile_h):
                src: int = (oy + ly) * width + ox
                value: float
                for value in samples[src : src + tile_w]:
                    if abs(value) > 1e-12:
                        any_nz = True
                        break
                if any_nz:
                    break
            if not any_nz:
                continue
            tile: list[float] = empty_tile(ts)
            for ly in range(tile_h):
                src = (oy + ly) * width + ox
                dst: int = ly * ts
                row_vals = samples[src : src + tile_w]
                tile[dst : dst + tile_w] = row_vals
                for value in row_vals:
                    if first:
                        sample_min = value
                        sample_max = value
                        first = False
                    elif value < sample_min:
                        sample_min = value
                    elif value > sample_max:
                        sample_max = value
            raster.tiles[(px, py)] = tile
    if not first:
        raster.sample_min = sample_min
        raster.sample_max = sample_max
    return raster


def dense_from_sparse(raster: SparseHeightRaster) -> list[float]:
    """materialize a full dense buffer for export only"""
    samples: list[float] = [0.0] * (raster.width * raster.height)
    key: tuple[int, int]
    tile: list[float]
    ts: int = raster.tile_size
    for key, tile in raster.tiles.items():
        px: int
        py: int
        px, py = key
        ox: int = px * ts
        oy: int = py * ts
        tile_h: int = min(ts, raster.height - oy)
        tile_w: int = min(ts, raster.width - ox)
        ly: int
        for ly in range(tile_h):
            src: int = ly * ts
            dst: int = (oy + ly) * raster.width + ox
            samples[dst : dst + tile_w] = tile[src : src + tile_w]
    return samples


def dense_region_from_sparse(
    raster: SparseHeightRaster, x0: int, x1: int, y0: int, y1: int
) -> list[float]:
    """row-major samples for an inclusive-exclusive texel rectangle"""
    rx0: int = max(0, x0)
    rx1: int = min(raster.width, x1)
    ry0: int = max(0, y0)
    ry1: int = min(raster.height, y1)
    if rx0 >= rx1 or ry0 >= ry1:
        return []
    region_w: int = rx1 - rx0
    out: list[float] = [0.0] * (region_w * (ry1 - ry0))
    y: int
    for y in range(ry0, ry1):
        row: int = (y - ry0) * region_w
        x: int
        for x in range(rx0, rx1):
            out[row + (x - rx0)] = get_sample(raster, x, y)
    return out


def sample_bilinear(raster: SparseHeightRaster, u: float, v: float) -> float:
    """bilinear sample at field uv"""
    if raster.width < 1 or raster.height < 1:
        return 0.0
    x: float = u * float(raster.width) - 0.5
    y: float = v * float(raster.height) - 0.5
    x0: int = int(floor(x))
    y0: int = int(floor(y))
    tx: float = x - float(x0)
    ty: float = y - float(y0)
    s00: float = get_sample(raster, x0, y0)
    s10: float = get_sample(raster, x0 + 1, y0)
    s01: float = get_sample(raster, x0, y0 + 1)
    s11: float = get_sample(raster, x0 + 1, y0 + 1)
    a: float = s00 + (s10 - s00) * tx
    b: float = s01 + (s11 - s01) * tx
    return a + (b - a) * ty
