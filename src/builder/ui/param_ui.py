"""typed inspector controls reflected from params dataclasses"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import Any, get_type_hints

from pyray import Rectangle

from builder.generators.param_types import (
    BoolParam,
    EnumParam,
    Float3Param,
    FloatParam,
    Int3Param,
    IntParam,
    clamp_range,
)
from builder.ui.theme import (
    ui_button_gap,
    ui_button_height,
    ui_font_size,
    ui_pad,
    ui_stepper_width,
)


@dataclass(frozen=True)
class BoolControl:
    name: str


@dataclass(frozen=True)
class IntControl:
    name: str


@dataclass(frozen=True)
class FloatControl:
    name: str


@dataclass(frozen=True)
class EnumControl:
    name: str


@dataclass(frozen=True)
class Float3ComponentControl:
    name: str
    component: str


@dataclass(frozen=True)
class Int3ComponentControl:
    name: str
    component: str


type ParamControl = (
    BoolControl
    | IntControl
    | FloatControl
    | EnumControl
    | Float3ComponentControl
    | Int3ComponentControl
)


def field_label(name: str) -> str:
    """turn a snake_case field name into a short ui label"""
    return " ".join(part.capitalize() for part in name.split("_") if part != "")


def control_key(control: ParamControl) -> str:
    """stable focus id for one inspector control"""
    match control:
        case Float3ComponentControl(name=name, component=component) | Int3ComponentControl(
            name=name, component=component
        ):
            return f"{name}.{component}"
        case BoolControl(name=name) | IntControl(name=name) | FloatControl(
            name=name
        ) | EnumControl(name=name):
            return name


def controls_from_params_type(params_type: type) -> tuple[ParamControl, ...]:
    """reflect a params dataclass into typed inspector controls"""
    hints: dict[str, Any] = get_type_hints(params_type)
    out: list[ParamControl] = []
    for field in fields(params_type):
        annotation: Any = hints[field.name]
        match annotation:
            case _ if annotation is Float3Param:
                component: str
                for component in ("x", "y", "z"):
                    out.append(Float3ComponentControl(field.name, component))
            case _ if annotation is Int3Param:
                for component in ("x", "y", "z"):
                    out.append(Int3ComponentControl(field.name, component))
            case _ if annotation is BoolParam:
                out.append(BoolControl(field.name))
            case _ if annotation is EnumParam:
                out.append(EnumControl(field.name))
            case _ if annotation is IntParam:
                out.append(IntControl(field.name))
            case _ if annotation is FloatParam:
                out.append(FloatControl(field.name))
    return tuple(out)


def control_by_key(
    controls: tuple[ParamControl, ...], key: str
) -> ParamControl:
    """return the control whose focus key matches"""
    control: ParamControl
    for control in controls:
        if control_key(control) == key:
            return control
    raise KeyError(f"unknown inspector control {key!r}")


def vec3_component_label(param: Float3Param | Int3Param, component: str) -> str:
    """component label from a vec3 wrapper"""
    match component:
        case "x":
            return param.label_x
        case "y":
            return param.label_y
        case "z":
            return param.label_z
        case _:
            raise KeyError(component)


def control_label(params: Any, control: ParamControl) -> str:
    """label for a control from its param wrapper"""
    match control:
        case BoolControl(name=name):
            param: BoolParam = getattr(params, name)
            return param.label or field_label(name)
        case IntControl(name=name):
            int_param: IntParam = getattr(params, name)
            return int_param.label or field_label(name)
        case FloatControl(name=name):
            float_param: FloatParam = getattr(params, name)
            return float_param.label or field_label(name)
        case EnumControl(name=name):
            enum_param: EnumParam = getattr(params, name)
            return enum_param.label or field_label(name)
        case Float3ComponentControl(name=name, component=component):
            float3: Float3Param = getattr(params, name)
            base: str = float3.label or field_label(name)
            return f"{base} {vec3_component_label(float3, component)}"
        case Int3ComponentControl(name=name, component=component):
            int3: Int3Param = getattr(params, name)
            int_base: str = int3.label or field_label(name)
            return f"{int_base} {vec3_component_label(int3, component)}"


def control_step(params: Any, control: ParamControl) -> float:
    """step size for a numeric control"""
    match control:
        case IntControl(name=name):
            return float(getattr(params, name).step)
        case FloatControl(name=name):
            return float(getattr(params, name).step)
        case Float3ComponentControl(name=name) | Int3ComponentControl(name=name):
            return float(getattr(params, name).step)
        case _:
            return 1.0


def read_control(params: Any, control: ParamControl) -> Any:
    """read the live value for one control from params"""
    match control:
        case BoolControl(name=name):
            return bool(getattr(params, name).value)
        case IntControl(name=name):
            return int(getattr(params, name).value)
        case FloatControl(name=name):
            return float(getattr(params, name).value)
        case EnumControl(name=name):
            return getattr(params, name).value
        case Float3ComponentControl(name=name, component=component):
            return float(getattr(getattr(params, name), component))
        case Int3ComponentControl(name=name, component=component):
            return int(getattr(getattr(params, name), component))


def write_control(params: Any, control: ParamControl, value: Any) -> Any:
    """return params with one control written"""
    match control:
        case BoolControl(name=name):
            current_bool: BoolParam = getattr(params, name)
            return replace(
                params, **{name: replace(current_bool, value=bool(value))}
            )
        case IntControl(name=name):
            current_int: IntParam = getattr(params, name)
            return replace(
                params,
                **{
                    name: replace(
                        current_int,
                        value=clamp_range(
                            int(round(float(value))),
                            current_int.minimum,
                            current_int.maximum,
                        ),
                    )
                },
            )
        case FloatControl(name=name):
            current_float: FloatParam = getattr(params, name)
            return replace(
                params,
                **{
                    name: replace(
                        current_float,
                        value=clamp_range(
                            float(value),
                            current_float.minimum,
                            current_float.maximum,
                        ),
                    )
                },
            )
        case EnumControl(name=name):
            current_enum: EnumParam = getattr(params, name)
            enum_type: type[IntEnum] = type(current_enum.value)
            return replace(
                params,
                **{name: replace(current_enum, value=enum_type(int(value)))},
            )
        case Float3ComponentControl(name=name, component=component):
            current_float3: Float3Param = getattr(params, name)
            clamped_f: float = clamp_range(
                float(value), current_float3.minimum, current_float3.maximum
            )
            updated_f: Float3Param = replace(
                current_float3, **{component: clamped_f}
            )
            return replace(params, **{name: updated_f})
        case Int3ComponentControl(name=name, component=component):
            current_int3: Int3Param = getattr(params, name)
            clamped_i: int = clamp_range(
                int(round(float(value))),
                current_int3.minimum,
                current_int3.maximum,
            )
            updated_i: Int3Param = replace(
                current_int3, **{component: clamped_i}
            )
            return replace(params, **{name: updated_i})


def format_control(control: ParamControl, value: Any) -> str:
    """format a control value for the inspector text field"""
    match control:
        case BoolControl():
            return "on" if value else "off"
        case IntControl() | Int3ComponentControl():
            return str(int(value))
        case FloatControl() | Float3ComponentControl():
            return f"{float(value):.4g}"
        case EnumControl():
            member: IntEnum = value
            return member.name.capitalize()


def parse_control(control: ParamControl, text: str) -> Any | None:
    """parse typed inspector text into a control value, or None if invalid"""
    stripped: str = text.strip()
    if stripped == "":
        return None
    match control:
        case BoolControl() | EnumControl():
            return None
        case IntControl() | Int3ComponentControl():
            try:
                return int(stripped)
            except ValueError:
                return None
        case FloatControl() | Float3ComponentControl():
            try:
                return float(stripped)
            except ValueError:
                return None


def enum_members(params: Any, control: EnumControl) -> tuple[IntEnum, ...]:
    """allowed enum members for an enum control"""
    param: EnumParam = getattr(params, control.name)
    return param.members()


@dataclass(frozen=True)
class ParamRowRects:
    """hit targets for one reflected param row"""

    key: str
    label: Rectangle
    minus: Rectangle
    value: Rectangle
    plus: Rectangle
    option_rects: tuple[Rectangle, ...] = ()


def layout_control_row(
    control: ParamControl,
    params: Any,
    x: float,
    y: float,
    inner_w: float,
) -> tuple[ParamRowRects, float]:
    """place one param row and return it with the y below it"""
    empty: Rectangle = Rectangle(0.0, 0.0, 0.0, 0.0)
    row_key: str = control_key(control)
    match control:
        case BoolControl():
            hit: Rectangle = Rectangle(x, y, inner_w, float(ui_button_height))
            row: ParamRowRects = ParamRowRects(
                key=row_key,
                label=hit,
                minus=empty,
                value=hit,
                plus=empty,
            )
            return row, y + float(ui_button_height) + ui_pad
        case EnumControl() as enum_control:
            options: tuple[IntEnum, ...] = enum_members(params, enum_control)
            enum_label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
            controls_y: float = y + float(ui_font_size) + 4.0
            option_count: int = max(1, len(options))
            option_w: float = (
                inner_w - ui_button_gap * (option_count - 1)
            ) / option_count
            option_rects: tuple[Rectangle, ...] = tuple(
                Rectangle(
                    x + (option_w + ui_button_gap) * index,
                    controls_y,
                    option_w,
                    float(ui_button_height),
                )
                for index in range(len(options))
            )
            row = ParamRowRects(
                key=row_key,
                label=enum_label,
                minus=empty,
                value=empty,
                plus=empty,
                option_rects=option_rects,
            )
            return row, controls_y + float(ui_button_height) + ui_pad
        case (
            IntControl()
            | FloatControl()
            | Float3ComponentControl()
            | Int3ComponentControl()
        ):
            label: Rectangle = Rectangle(x, y, inner_w, float(ui_font_size))
            controls_y = y + float(ui_font_size) + 4.0
            minus: Rectangle = Rectangle(
                x, controls_y, float(ui_stepper_width), float(ui_button_height)
            )
            plus: Rectangle = Rectangle(
                x + inner_w - float(ui_stepper_width),
                controls_y,
                float(ui_stepper_width),
                float(ui_button_height),
            )
            value: Rectangle = Rectangle(
                minus.x + minus.width + ui_button_gap,
                controls_y,
                plus.x - (minus.x + minus.width + ui_button_gap * 2),
                float(ui_button_height),
            )
            row = ParamRowRects(
                key=row_key,
                label=label,
                minus=minus,
                value=value,
                plus=plus,
            )
            return row, controls_y + float(ui_button_height) + ui_pad
