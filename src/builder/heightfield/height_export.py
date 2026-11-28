"""export composited heightfield heights to .asc / u16 .png"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from builder.heightfield.heightfield_types import (
    Heightfield,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    ensure_heightmap_buffer,
    heightmap_buffer_from_samples,
)
from builder.heightfield.sample import sample_height


def export_resolution(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> tuple[int, int]:
    """prefer base heightmap size; else mesh vertex grid"""
    if heightfield.base is not None:
        buffer: HeightmapBuffer | None = ensure_heightmap_buffer(
            catalog, heightfield.base
        )
        if buffer is not None and buffer.width > 0 and buffer.height > 0:
            return (buffer.width, buffer.height)
    layout = heightfield_layout(heightfield)
    return (layout.res_x, layout.res_z)


def composite_height_buffer(
    catalog: HeightmapCatalog, heightfield: Heightfield
) -> HeightmapBuffer:
    """rasterize final world heights (enabled layers only); row 0 = south"""
    width: int
    height: int
    width, height = export_resolution(catalog, heightfield)
    if width < 1 or height < 1:
        raise ValueError("heightfield export resolution is empty")
    samples: list[float] = []
    y: int
    x: int
    for y in range(height):
        v: float = float(y) / float(height - 1) if height > 1 else 0.5
        for x in range(width):
            u: float = float(x) / float(width - 1) if width > 1 else 0.5
            samples.append(sample_height(catalog, heightfield, u, v))
    return heightmap_buffer_from_samples(width, height, samples)


def north_first_rows(buffer: HeightmapBuffer) -> list[float]:
    """flip south-first / east-first buffer to ASC north-first / west-first"""
    from builder.heightfield.heightmap_catalog import materialize_heightmap_samples

    width: int = buffer.width
    height: int = buffer.height
    samples: list[float] = materialize_heightmap_samples(buffer)
    flipped: list[float] = []
    row: int
    for row in range(height - 1, -1, -1):
        start: int = row * width
        row_vals: list[float] = samples[start : start + width]
        flipped.extend(reversed(row_vals))
    return flipped


def write_height_export_meta(
    path: Path,
    *,
    width: int,
    height: int,
    cell_size_m: float,
    size_x: float,
    size_z: float,
    height_min: float,
    height_max: float,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
) -> None:
    """write {stem}_meta.json beside an exported height file"""
    meta_path: Path = path.parent / f"{path.stem}_meta.json"
    payload: dict[str, object] = {
        "grid": {
            "width": width,
            "height": height,
            "cell_size_m": cell_size_m,
            "world_size_m_x": size_x,
            "world_size_m_z": size_z,
            "origin_x": origin_x,
            "origin_z": origin_z,
        },
        "elevation_m": {"min": height_min, "max": height_max},
        "coordinates": {
            "png_asc_origin": (
                "row 0 = north (max z), column 0 = west "
                f"(x = {origin_x}, z = {origin_z})"
            ),
        },
    }
    meta_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_height_asc(
    path: Path,
    buffer: HeightmapBuffer,
    *,
    size_x: float,
    size_z: float,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
) -> None:
    """write ESRI ASCII grid (metres, north-first) and sidecar meta"""
    if buffer.width < 1 or buffer.height < 1:
        raise ValueError("cannot export empty height buffer")
    cellsize: float = size_x / float(buffer.width)
    north_first: list[float] = north_first_rows(buffer)
    lines: list[str] = [
        f"ncols         {buffer.width}",
        f"nrows         {buffer.height}",
        f"xllcorner     {origin_x:.6f}",
        f"yllcorner     {origin_z:.6f}",
        f"cellsize      {cellsize}",
        "NODATA_value  -9999.0",
    ]
    row: int
    for row in range(buffer.height):
        start: int = row * buffer.width
        row_vals: list[float] = north_first[start : start + buffer.width]
        lines.append(" ".join(f"{value:.6f}" for value in row_vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_height_export_meta(
        path,
        width=buffer.width,
        height=buffer.height,
        cell_size_m=cellsize,
        size_x=size_x,
        size_z=size_z,
        height_min=buffer.sample_min,
        height_max=buffer.sample_max,
        origin_x=origin_x,
        origin_z=origin_z,
    )


def pack_u16_heights(samples: list[float], height_min: float, height_max: float) -> list[int]:
    """normalize metres to uint16 spanning height_min..height_max"""
    span: float = height_max - height_min
    if abs(span) < 1e-8:
        return [32768 for _ in samples]
    scale: float = 65535.0 / span
    packed: list[int] = []
    value: float
    for value in samples:
        coded: float = (value - height_min) * scale
        if coded < 0.0:
            coded = 0.0
        elif coded > 65535.0:
            coded = 65535.0
        packed.append(int(round(coded)))
    return packed


def write_height_png_u16(
    path: Path,
    buffer: HeightmapBuffer,
    *,
    size_x: float,
    size_z: float,
    origin_x: float = 0.0,
    origin_z: float = 0.0,
) -> None:
    """write 16-bit gray png (north-first) and sidecar meta with elevation bounds"""
    if buffer.width < 1 or buffer.height < 1:
        raise ValueError("cannot export empty height buffer")
    north_first: list[float] = north_first_rows(buffer)
    height_min: float = buffer.sample_min
    height_max: float = buffer.sample_max
    codes: list[int] = pack_u16_heights(north_first, height_min, height_max)
    image: Image.Image = Image.new("I;16", (buffer.width, buffer.height))
    # pillow putdata accepts sequence of ints for I;16
    image.putdata(codes)
    image.save(path)
    cellsize: float = size_x / float(buffer.width)
    write_height_export_meta(
        path,
        width=buffer.width,
        height=buffer.height,
        cell_size_m=cellsize,
        size_x=size_x,
        size_z=size_z,
        height_min=height_min,
        height_max=height_max,
        origin_x=origin_x,
        origin_z=origin_z,
    )


def export_composited_height(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    path: Path,
) -> HeightmapBuffer:
    """composite enabled layers and write .asc or .png by path suffix"""
    buffer: HeightmapBuffer = composite_height_buffer(catalog, heightfield)
    size_x: float = float(heightfield.size_x.value)
    size_z: float = float(heightfield.size_z.value)
    origin_x: float = float(heightfield.origin_x.value)
    origin_z: float = float(heightfield.origin_z.value)
    suffix: str = path.suffix.lower()
    match suffix:
        case ".asc":
            write_height_asc(
                path,
                buffer,
                size_x=size_x,
                size_z=size_z,
                origin_x=origin_x,
                origin_z=origin_z,
            )
        case ".png":
            write_height_png_u16(
                path,
                buffer,
                size_x=size_x,
                size_z=size_z,
                origin_x=origin_x,
                origin_z=origin_z,
            )
        case _:
            raise ValueError(f"unsupported height export suffix {suffix!r}")
    return buffer
