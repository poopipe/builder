"""pillow-backed height image decode (swappable via DecodeHeightImage)"""

from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image

from builder.heightfield.height_image_types import (
    DecodedHeightImage,
    HeightSampleType,
)


def pack_u16_le(values: list[int]) -> bytes:
    """pack unsigned 16-bit samples as little-endian bytes"""
    return struct.pack(f"<{len(values)}H", *values)


def pillow_decode_height_image(path: Path) -> DecodedHeightImage:
    """decode a height image with pillow; 16-bit gray png is the supported path

    8-bit gray is accepted so additional formats can share this decoder later
    """
    with Image.open(path) as image:
        image.load()
        width: int = int(image.size[0])
        height: int = int(image.size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"heightmap has no pixels: {path}")
        mode: str = image.mode
        count: int = width * height
        if mode in ("I;16", "I;16L"):
            raw: bytes = image.tobytes("raw", mode)
            if len(raw) != count * 2:
                raise ValueError(f"unexpected 16-bit payload size for {path}")
            values: list[int] = list(struct.unpack(f"<{count}H", raw))
            return DecodedHeightImage(
                width=width,
                height=height,
                sample_type=HeightSampleType.u16,
                data=pack_u16_le(values),
            )
        if mode == "I;16B":
            raw_be: bytes = image.tobytes("raw", "I;16B")
            if len(raw_be) != count * 2:
                raise ValueError(f"unexpected 16-bit payload size for {path}")
            values_be: list[int] = list(struct.unpack(f">{count}H", raw_be))
            return DecodedHeightImage(
                width=width,
                height=height,
                sample_type=HeightSampleType.u16,
                data=pack_u16_le(values_be),
            )
        if mode == "I":
            # pillow often promotes 16-bit gray png to 32-bit int pixels
            pixels: list[int] = [int(value) for value in image.getdata()]
            if len(pixels) != count:
                raise ValueError(f"unexpected pixel count for {path}")
            max_value: int = max(pixels) if pixels else 0
            min_value: int = min(pixels) if pixels else 0
            if min_value < 0 or max_value > 65535:
                raise ValueError(
                    f"32-bit int heightmap out of uint16 range "
                    f"[{min_value}, {max_value}]: {path}"
                )
            return DecodedHeightImage(
                width=width,
                height=height,
                sample_type=HeightSampleType.u16,
                data=pack_u16_le(pixels),
            )
        if mode == "L":
            raw_u8: bytes = image.tobytes("raw", "L")
            if len(raw_u8) != count:
                raise ValueError(f"unexpected 8-bit payload size for {path}")
            return DecodedHeightImage(
                width=width,
                height=height,
                sample_type=HeightSampleType.u8,
                data=raw_u8,
            )
        raise ValueError(
            f"unsupported height image mode {mode!r} for {path}; "
            f"expected 16-bit or 8-bit grayscale"
        )
