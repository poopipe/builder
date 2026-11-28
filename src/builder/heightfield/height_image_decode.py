"""height-image decode surface; swap the callable to replace pillow later"""

from __future__ import annotations

import struct
from array import array
from collections.abc import Callable
from pathlib import Path

from builder.heightfield.asc_height_decode import asc_decode_height_image
from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
)
from builder.heightfield.pillow_height_decode import pillow_decode_height_image


type DecodeHeightImage = Callable[[Path], DecodedHeightImage]


def default_decode_height_image(path: Path) -> DecodedHeightImage:
    """dispatch by suffix: .asc → ESRI grid, else pillow image"""
    if path.suffix.lower() == ".asc":
        return asc_decode_height_image(path)
    return pillow_decode_height_image(path)


def samples_from_u8(raw: bytes) -> list[float]:
    """normalize packed 8-bit samples to 0..1"""
    return [float(value) / 255.0 for value in raw]


def samples_from_u16(raw: bytes) -> list[float]:
    """normalize packed little-endian uint16 samples to 0..1"""
    if len(raw) % 2 != 0:
        raise ValueError("uint16 payload length must be even")
    count: int = len(raw) // 2
    values: tuple[int, ...] = struct.unpack(f"<{count}H", raw)
    return [float(value) / 65535.0 for value in values]


def samples_from_i32(raw: bytes) -> list[float]:
    """signed little-endian int32 samples as float32 values (no normalize)"""
    if len(raw) % 4 != 0:
        raise ValueError("int32 payload length must be a multiple of 4")
    count: int = len(raw) // 4
    values: tuple[int, ...] = struct.unpack(f"<{count}i", raw)
    return [float(value) for value in values]


def f32_array_from_bytes(raw: bytes) -> array[float]:
    """packed little-endian float32 → contiguous array (no per-element python float)"""
    if len(raw) % 4 != 0:
        raise ValueError("float32 payload length must be a multiple of 4")
    samples: array[float] = array("f")
    samples.frombytes(raw)
    return samples


def samples_from_f32(raw: bytes) -> list[float]:
    """signed little-endian float32 samples as a python list"""
    return list(f32_array_from_bytes(raw))


def float_samples_from_decoded(image: DecodedHeightImage) -> list[float]:
    """convert packed decoded samples to signed float heights"""
    match image.sample_type:
        case HeightSampleType.u8:
            return samples_from_u8(image.data)
        case HeightSampleType.u16:
            return samples_from_u16(image.data)
        case HeightSampleType.i32:
            return samples_from_i32(image.data)
        case HeightSampleType.f32:
            return samples_from_f32(image.data)
