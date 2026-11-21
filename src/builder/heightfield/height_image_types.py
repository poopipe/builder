"""decoded height-image sample layouts (independent of raylib pixel formats)"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class HeightSampleType(IntEnum):
    """packed sample type in a decoded height image or .hraw cache"""

    u8 = 1
    u16 = 2
    i32 = 3
    f32 = 4


def bytes_per_sample(sample_type: HeightSampleType) -> int:
    """payload bytes for one sample of the given type"""
    match sample_type:
        case HeightSampleType.u8:
            return 1
        case HeightSampleType.u16:
            return 2
        case HeightSampleType.i32 | HeightSampleType.f32:
            return 4


@dataclass(frozen=True)
class DecodedHeightImage:
    """row-major packed samples ready to cache or convert to float heights"""

    width: int
    height: int
    sample_type: HeightSampleType
    data: bytes
