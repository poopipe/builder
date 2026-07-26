"""grid distribution functions — each returns a list of transforms"""

from __future__ import annotations

from math import acos, cos, pi, radians, sin

from pyray import (
    Matrix,
    Transform,
    Vector3,
    Vector4,
    quaternion_from_axis_angle,
    quaternion_from_matrix,
    quaternion_identity,
    vector3_add,
    vector3_cross_product,
    vector3_dot_product,
    vector3_length,
    vector3_normalize,
    vector3_scale,
    vector3_subtract,
)

from builder.scene.scene_types import transform_at
from builder.generators.generator_types import GeneratedSlot, SlotRole

type AxisIndex = int


def clamp_axis(axis: AxisIndex) -> AxisIndex:
    """clamp an axis index to 0..2 (x/y/z)"""
    return max(0, min(2, int(axis)))


def axis_vector(axis: AxisIndex, value: float) -> Vector3:
    """return a vector with value along the given axis"""
    if axis == 0:
        return Vector3(value, 0.0, 0.0)
    if axis == 1:
        return Vector3(0.0, value, 0.0)
    return Vector3(0.0, 0.0, value)


def axis_component(position: Vector3, axis: AxisIndex) -> float:
    """return the position component on the given axis"""
    if axis == 0:
        return float(position.x)
    if axis == 1:
        return float(position.y)
    return float(position.z)


def ring_direction(axis: AxisIndex, angle_rad: float) -> Vector3:
    """unit direction in the plane perpendicular to the cylinder axis"""
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)
    if axis == 0:
        return Vector3(0.0, c, s)
    if axis == 1:
        return Vector3(c, 0.0, s)
    return Vector3(c, s, 0.0)


def aspect_scale_axis(cylinder: AxisIndex) -> AxisIndex:
    """world axis stretched by aspect for a polygon about the cylinder axis

    matches the cosine basis of ring_direction: Y-cylinder -> X, X-cylinder -> Y,
    Z-cylinder -> X
    """
    if cylinder == 0:
        return 1
    return 0


def rotation_from_z_to(direction: Vector3, up_hint: Vector3) -> Vector4:
    """quaternion rotating +Z onto direction

    when direction is -Z the 180° axis is taken from up_hint projected into the
    XY plane, so a Y-up radial ring keeps +Y upright instead of flipping about X
    """
    target: Vector3 = vector3_normalize(direction)
    z: Vector3 = Vector3(0.0, 0.0, 1.0)
    dot: float = vector3_dot_product(z, target)
    if dot > 0.999999:
        return quaternion_identity()
    if dot < -0.999999:
        axis: Vector3 = Vector3(float(up_hint.x), float(up_hint.y), 0.0)
        if vector3_length(axis) < 1e-6:
            axis = Vector3(1.0, 0.0, 0.0)
        else:
            axis = vector3_normalize(axis)
        return quaternion_from_axis_angle(axis, pi)
    axis = vector3_normalize(vector3_cross_product(z, target))
    angle: float = acos(max(-1.0, min(1.0, dot)))
    return quaternion_from_axis_angle(axis, angle)


def rotation_looking_along(direction: Vector3, up: Vector3) -> Vector4:
    """quaternion with +Z along direction and +Y toward up"""
    forward: Vector3 = vector3_normalize(direction)
    up_hint: Vector3 = vector3_normalize(up)
    right: Vector3 = vector3_cross_product(up_hint, forward)
    if vector3_length(right) < 1e-6:
        # forward parallel to up: pick any axis not aligned with forward
        fallback: Vector3 = Vector3(0.0, 1.0, 0.0)
        if abs(vector3_dot_product(forward, fallback)) > 0.999:
            fallback = Vector3(1.0, 0.0, 0.0)
        right = vector3_cross_product(fallback, forward)
    right = vector3_normalize(right)
    upward: Vector3 = vector3_cross_product(forward, right)
    # columns are local X/Y/Z in world space (m[row + 4*col])
    matrix: Matrix = Matrix()
    matrix.m0 = float(right.x)
    matrix.m1 = float(right.y)
    matrix.m2 = float(right.z)
    matrix.m3 = 0.0
    matrix.m4 = float(upward.x)
    matrix.m5 = float(upward.y)
    matrix.m6 = float(upward.z)
    matrix.m7 = 0.0
    matrix.m8 = float(forward.x)
    matrix.m9 = float(forward.y)
    matrix.m10 = float(forward.z)
    matrix.m11 = 0.0
    matrix.m12 = 0.0
    matrix.m13 = 0.0
    matrix.m14 = 0.0
    matrix.m15 = 1.0
    return quaternion_from_matrix(matrix)


def align_build_from(
    transforms: list[Transform],
    origin: Vector3,
    build_from: AxisIndex,
) -> list[Transform]:
    """shift so the build-from axis min of positions equals origin on that axis"""
    if not transforms:
        return transforms
    axis: AxisIndex = clamp_axis(build_from)
    min_val: float = min(
        axis_component(transform.translation, axis) for transform in transforms
    )
    delta: float = axis_component(origin, axis) - min_val
    if abs(delta) < 1e-12:
        return transforms
    offset: Vector3 = axis_vector(axis, delta)
    aligned: list[Transform] = []
    transform: Transform
    for transform in transforms:
        aligned.append(
            Transform(
                vector3_add(transform.translation, offset),
                transform.rotation,
                transform.scale,
            )
        )
    return aligned


def align_slots(
    slots: list[GeneratedSlot],
    origin: Vector3,
    build_from: AxisIndex,
) -> list[GeneratedSlot]:
    """align slot translations with build-from, preserving roles"""
    if not slots:
        return slots
    aligned: list[Transform] = align_build_from(
        [slot.transform for slot in slots], origin, build_from
    )
    return [
        GeneratedSlot(transform, slot.role)
        for slot, transform in zip(slots, aligned, strict=True)
    ]


def grid_transforms(
    origin: Vector3,
    count_x: int = 3,
    count_y: int = 1,
    count_z: int = 3,
    spacing_x: float = 2.0,
    spacing_y: float = 2.0,
    spacing_z: float = 2.0,
    build_from: AxisIndex = 1,
) -> list[Transform]:
    """return transforms for a regular XYZ lattice

    centered on the non-build axes; flush to origin on build_from (default Y)
    """
    step_x: Vector3 = Vector3(spacing_x, 0.0, 0.0)
    step_y: Vector3 = Vector3(0.0, spacing_y, 0.0)
    step_z: Vector3 = Vector3(0.0, 0.0, spacing_z)
    start: Vector3 = vector3_subtract(
        origin,
        vector3_add(
            vector3_add(
                vector3_scale(step_x, (count_x - 1) * 0.5),
                vector3_scale(step_y, (count_y - 1) * 0.5),
            ),
            vector3_scale(step_z, (count_z - 1) * 0.5),
        ),
    )
    transforms: list[Transform] = []
    ix: int
    iy: int
    iz: int
    for iz in range(count_z):
        for iy in range(count_y):
            for ix in range(count_x):
                position: Vector3 = vector3_add(
                    start,
                    vector3_add(
                        vector3_add(
                            vector3_scale(step_x, float(ix)),
                            vector3_scale(step_y, float(iy)),
                        ),
                        vector3_scale(step_z, float(iz)),
                    ),
                )
                transforms.append(transform_at(position))
    return align_build_from(transforms, origin, build_from)


def radial_grid_transforms(
    origin: Vector3,
    radius: float,
    spacing: float,
    face_center: bool = False,
    axis: AxisIndex = 1,
    count_height: int = 1,
    spacing_height: float = 2.0,
    count_radius: int = 1,
    spacing_radius: float = 2.0,
    build_from: AxisIndex = 1,
) -> list[Transform]:
    """return transforms for stacked concentric rings around a cylinder axis

    spacing is the angular step in degrees.
    radius is the outer ring; count_radius rings step inward by spacing_radius.
    count_height stacks copies along axis.
    when face_center is set each transform's +Z points toward the cylinder axis.
    """
    cylinder: AxisIndex = clamp_axis(axis)
    height_count: int = max(1, int(count_height))
    radius_count: int = max(1, int(count_radius))
    transforms: list[Transform] = []
    ih: int
    ir: int
    for ih in range(height_count):
        height: float = (float(ih) - (height_count - 1) * 0.5) * spacing_height
        height_offset: Vector3 = axis_vector(cylinder, height)
        for ir in range(radius_count):
            ring_radius: float = radius - float(ir) * spacing_radius
            if ring_radius <= 0.0:
                continue
            angle_deg: float = 0.0
            while angle_deg < 360.0:
                angle_rad: float = radians(angle_deg)
                direction: Vector3 = ring_direction(cylinder, angle_rad)
                position: Vector3 = vector3_add(
                    origin,
                    vector3_add(
                        vector3_scale(direction, ring_radius), height_offset
                    ),
                )
                if face_center:
                    # +Z toward the cylinder axis = opposite the ring direction
                    inward: Vector3 = vector3_scale(direction, -1.0)
                    transforms.append(
                        Transform(
                            position,
                            rotation_from_z_to(
                                inward, axis_vector(cylinder, 1.0)
                            ),
                            Vector3(1.0, 1.0, 1.0),
                        )
                    )
                else:
                    transforms.append(transform_at(position))
                angle_deg += spacing
    return align_build_from(transforms, origin, build_from)


def ngon_point(
    origin: Vector3,
    cylinder: AxisIndex,
    ring_radius: float,
    height_offset: Vector3,
    angle_rad: float,
) -> Vector3:
    """return a point on a regular polygon ring"""
    direction: Vector3 = ring_direction(cylinder, angle_rad)
    return vector3_add(
        origin,
        vector3_add(vector3_scale(direction, ring_radius), height_offset),
    )


def scale_aspect(
    position: Vector3,
    center: Vector3,
    aspect: float,
    scale_axis: AxisIndex,
) -> Vector3:
    """scale position away from center along the chosen world axis"""
    if abs(aspect - 1.0) < 1e-12:
        return position
    offset: Vector3 = vector3_subtract(position, center)
    if scale_axis == 0:
        offset = Vector3(float(offset.x) * aspect, float(offset.y), float(offset.z))
    elif scale_axis == 1:
        offset = Vector3(float(offset.x), float(offset.y) * aspect, float(offset.z))
    else:
        offset = Vector3(float(offset.x), float(offset.y), float(offset.z) * aspect)
    return vector3_add(center, offset)


def append_oriented_slot(
    slots: list[GeneratedSlot],
    position: Vector3,
    center: Vector3,
    edge_dir: Vector3,
    cylinder: AxisIndex,
    facing: int,
    role: SlotRole,
) -> None:
    """append a slot with optional facing toward center or along the edge"""
    aim: Vector3 | None = None
    if facing == 1:
        aim = vector3_subtract(center, position)
    elif facing == 2:
        aim = edge_dir
    if aim is None or vector3_length(aim) < 1e-6:
        slots.append(GeneratedSlot(transform_at(position), role))
        return
    slots.append(
        GeneratedSlot(
            Transform(
                position,
                rotation_looking_along(aim, axis_vector(cylinder, 1.0)),
                Vector3(1.0, 1.0, 1.0),
            ),
            role,
        )
    )


def ngon_grid_transforms(
    origin: Vector3,
    radius: float,
    sides: int,
    spacing: float,
    facing: int = 0,
    point_facing: int = 0,
    include_points: bool = True,
    aspect: float = 1.0,
    axis: AxisIndex = 1,
    count_height: int = 1,
    spacing_height: float = 2.0,
    count_radius: int = 1,
    spacing_radius: float = 2.0,
    build_from: AxisIndex = 1,
) -> list[GeneratedSlot]:
    """return slots for stacked concentric regular polygons

    spacing is the target linear step along each edge in meters.
    each edge includes its start vertex (when include_points) and evenly spaced
    intermediates; the endpoint is omitted so shared vertices appear once.
    radius is the outer circumradius; count_radius rings step inward.
    aspect scales offsets from the ring center along the in-plane axis that
    matches ring_direction's cosine basis (Y-cylinder -> X, etc).
    facing / point_facing: 0 = none, 1 = +Z toward center, 2 = +Z along the edge.
    """
    cylinder: AxisIndex = clamp_axis(axis)
    side_count: int = max(3, int(sides))
    height_count: int = max(1, int(count_height))
    radius_count: int = max(1, int(count_radius))
    step: float = max(1e-6, float(spacing))
    aspect_scale: float = max(1e-6, float(aspect))
    stretch_axis: AxisIndex = aspect_scale_axis(cylinder)
    slots: list[GeneratedSlot] = []
    ih: int
    ir: int
    for ih in range(height_count):
        height: float = (float(ih) - (height_count - 1) * 0.5) * spacing_height
        height_offset: Vector3 = axis_vector(cylinder, height)
        for ir in range(radius_count):
            ring_radius: float = radius - float(ir) * spacing_radius
            if ring_radius <= 0.0:
                continue
            center: Vector3 = vector3_add(origin, height_offset)
            edge: int
            for edge in range(side_count):
                a0: float = (2.0 * pi * float(edge)) / float(side_count)
                a1: float = (2.0 * pi * float(edge + 1)) / float(side_count)
                p0: Vector3 = scale_aspect(
                    ngon_point(origin, cylinder, ring_radius, height_offset, a0),
                    center,
                    aspect_scale,
                    stretch_axis,
                )
                p1: Vector3 = scale_aspect(
                    ngon_point(origin, cylinder, ring_radius, height_offset, a1),
                    center,
                    aspect_scale,
                    stretch_axis,
                )
                edge_dir: Vector3 = vector3_subtract(p1, p0)
                divisions: int = max(
                    1, int(round(vector3_length(edge_dir) / step))
                )
                first: int = 0 if include_points else 1
                sample: int
                for sample in range(first, divisions):
                    t: float = float(sample) / float(divisions)
                    position: Vector3 = vector3_add(
                        p0, vector3_scale(edge_dir, t)
                    )
                    role: SlotRole = "point" if sample == 0 else "edge"
                    slot_facing: int = point_facing if role == "point" else facing
                    append_oriented_slot(
                        slots,
                        position,
                        center,
                        edge_dir,
                        cylinder,
                        slot_facing,
                        role,
                    )
    return align_slots(slots, origin, build_from)
