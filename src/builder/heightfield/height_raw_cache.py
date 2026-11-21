"""self-describing raw height cache (.hraw) beside imported images"""

from __future__ import annotations

import struct
from pathlib import Path

from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
    bytes_per_sample,
)


hraw_magic: bytes = b"HRAW"
hraw_version: int = 1
# magic, version u16, sample_type u16, width u32, height u32 → 16 bytes
hraw_header_struct: str = "<4sHHII"
hraw_header_size: int = struct.calcsize(hraw_header_struct)
hraw_suffix: str = ".hraw"


def default_height_raw_cache_path(source: Path) -> Path:
    """cache path beside the source image: stem.hraw"""
    return source.with_name(f"{source.stem}{hraw_suffix}")


def write_height_raw_cache(path: Path, image: DecodedHeightImage) -> None:
    """write a .hraw file from decoded packed samples"""
    expected: int = image.width * image.height * bytes_per_sample(image.sample_type)
    if len(image.data) != expected:
        raise ValueError(
            f"height raw payload size mismatch: got {len(image.data)}, "
            f"expected {expected}"
        )
    header: bytes = struct.pack(
        hraw_header_struct,
        hraw_magic,
        hraw_version,
        int(image.sample_type),
        image.width,
        image.height,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + image.data)


def read_height_raw_cache(path: Path) -> DecodedHeightImage:
    """read a .hraw file into packed samples"""
    blob: bytes = path.read_bytes()
    if len(blob) < hraw_header_size:
        raise ValueError(f"height raw file too small: {path}")
    magic: bytes
    version: int
    sample_raw: int
    width: int
    height: int
    magic, version, sample_raw, width, height = struct.unpack(
        hraw_header_struct, blob[:hraw_header_size]
    )
    if magic != hraw_magic:
        raise ValueError(f"height raw magic mismatch: {path}")
    if version != hraw_version:
        raise ValueError(
            f"unsupported height raw version {version}; expected {hraw_version}"
        )
    try:
        sample_type: HeightSampleType = HeightSampleType(sample_raw)
    except ValueError as exc:
        raise ValueError(f"unknown height sample type {sample_raw}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"height raw has no pixels: {path}")
    payload: bytes = blob[hraw_header_size:]
    expected: int = width * height * bytes_per_sample(sample_type)
    if len(payload) != expected:
        raise ValueError(
            f"height raw payload size mismatch: got {len(payload)}, "
            f"expected {expected}"
        )
    return DecodedHeightImage(
        width=width,
        height=height,
        sample_type=sample_type,
        data=payload,
    )
