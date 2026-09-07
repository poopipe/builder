"""typed generator param wrappers with ui metadata"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class IntParam:
    """integer generator param"""

    value: int
    label: str = ""
    step: int = 1
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class FloatParam:
    """float generator param"""

    value: float
    label: str = ""
    step: float = 0.25
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class BoolParam:
    """boolean generator param"""

    value: bool = False
    label: str = ""


@dataclass(frozen=True)
class EnumParam:
    """int-enum generator param"""

    value: IntEnum
    label: str = ""
    options: tuple[IntEnum, ...] | None = None

    def members(self) -> tuple[IntEnum, ...]:
        """allowed enum members for ui; defaults to the value's enum type"""
        if self.options is not None:
            return self.options
        enum_type: type[IntEnum] = type(self.value)
        return tuple(enum_type)


@dataclass(frozen=True)
class Int3Param:
    """three related integer components"""

    x: int = 0
    y: int = 0
    z: int = 0
    label: str = ""
    label_x: str = "X"
    label_y: str = "Y"
    label_z: str = "Z"
    step: int = 1
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True)
class Float3Param:
    """three related float components"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    label: str = ""
    label_x: str = "X"
    label_y: str = "Y"
    label_z: str = "Z"
    step: float = 0.25
    minimum: float | None = None
    maximum: float | None = None


def clamp_range[T: (int, float)](
    value: T, minimum: T | None, maximum: T | None
) -> T:
    """clamp a number to an optional inclusive range"""
    number: T = value
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number
