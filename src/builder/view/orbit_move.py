"""fps-style keyboard fly deltas for the orbit camera"""

from __future__ import annotations

from math import cos, radians, sin

from pyray import Vector3, vector3_length, vector3_scale

# metres per second at distance=1 and move_speed=1
orbit_fly_speed_factor: float = 1.2
# metres per wheel notch at distance=1 and zoom_speed=1
orbit_dolly_speed_factor: float = 1.5
move_speed_step: float = 0.25
move_speed_minimum: float = 0.1
move_speed_maximum: float = 100.0

view_distance_default: float = 4096.0
view_distance_step: float = 64.0
view_distance_minimum: float = 64.0
view_distance_maximum: float = 131072.0
view_distance_near: float = 0.05


def clamp_move_speed(value: float) -> float:
    """keep move speed in a usable range"""
    return max(move_speed_minimum, min(move_speed_maximum, value))


def nudged_move_speed(current: float, direction: int) -> float:
    """step move speed up or down by move_speed_step"""
    next_value: float = current + move_speed_step * float(direction)
    next_value = round(next_value / move_speed_step) * move_speed_step
    return clamp_move_speed(next_value)


def clamp_view_distance(value: float) -> float:
    """keep far clip distance in a usable range"""
    return max(view_distance_minimum, min(view_distance_maximum, value))


def nudged_view_distance(current: float, direction: int) -> float:
    """step view distance up or down by view_distance_step"""
    next_value: float = current + view_distance_step * float(direction)
    next_value = round(next_value / view_distance_step) * view_distance_step
    return clamp_view_distance(next_value)


def orbit_yaw_forward(yaw_degrees: float) -> Vector3:
    """horizontal look direction on xz from orbit yaw"""
    yaw_r: float = radians(yaw_degrees)
    return Vector3(-sin(yaw_r), 0.0, -cos(yaw_r))


def orbit_yaw_right(yaw_degrees: float) -> Vector3:
    """horizontal strafe right on xz from orbit yaw"""
    yaw_r: float = radians(yaw_degrees)
    return Vector3(cos(yaw_r), 0.0, -sin(yaw_r))


def fly_move_delta(
    yaw_degrees: float,
    move_forward: float,
    move_right: float,
    move_up: float,
    distance: float,
    move_speed: float,
    dt: float,
) -> Vector3:
    """world-space translation for one frame of wasd+qe input

    move_* are signed axis sums; the result is normalized then scaled
    """
    if dt <= 0.0:
        return Vector3(0.0, 0.0, 0.0)
    forward: Vector3 = orbit_yaw_forward(yaw_degrees)
    right: Vector3 = orbit_yaw_right(yaw_degrees)
    raw: Vector3 = Vector3(
        forward.x * move_forward + right.x * move_right,
        move_up,
        forward.z * move_forward + right.z * move_right,
    )
    length: float = vector3_length(raw)
    if length < 1e-8:
        return Vector3(0.0, 0.0, 0.0)
    direction: Vector3 = vector3_scale(raw, 1.0 / length)
    speed: float = max(distance, 0.5) * orbit_fly_speed_factor * max(move_speed, 0.1)
    return vector3_scale(direction, speed * dt)


def dolly_move_delta(
    look_forward: Vector3,
    wheel: float,
    distance: float,
    zoom_speed: float,
) -> Vector3:
    """translate along full look direction for one wheel step (includes pitch)"""
    if abs(wheel) < 1e-8:
        return Vector3(0.0, 0.0, 0.0)
    length: float = vector3_length(look_forward)
    if length < 1e-8:
        return Vector3(0.0, 0.0, 0.0)
    direction: Vector3 = vector3_scale(look_forward, 1.0 / length)
    step: float = (
        max(distance, 0.5) * orbit_dolly_speed_factor * max(zoom_speed, 0.1)
    )
    return vector3_scale(direction, wheel * step)
