"""viewport overlays for live vector shapes on their work plane"""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, hypot, pi, sin

from pyray import (
    BlendMode,
    Camera3D,
    Color,
    Matrix,
    Vector2,
    Vector3,
    begin_blend_mode,
    draw_cube,
    draw_plane,
    end_blend_mode,
    ffi,
    matrix_to_float_v,
    rl_disable_depth_mask,
    rl_enable_depth_mask,
    rl_mult_matrixf,
    rl_pop_matrix,
    rl_push_matrix,
    vector3_distance,
)

from builder.heightfield.heightfield_types import Heightfield, heightfield_layout
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height, sample_height_below
from builder.heightfield.vector.circle import CircleShapeParams, circle_rim_local_xz
from builder.heightfield.vector.coords import (
    ordered_child_nodes,
    vector_layer_is_linked,
    vector_layer_is_road,
    vector_layer_plane_y,
    vector_point_entries,
    vector_point_entries_xzy,
    vector_point_xz,
    vector_road_layer_index,
)
from builder.heightfield.vector.overlay_scale import (
    overlay_line_pixels,
    overlay_radius_for_camera,
    point_cube_size_for_line_radius,
    shadow_line_pixels,
)
from builder.view.line_shader import (
    LineBatch,
    LineShader,
    add_line_segment,
    clear_line_batch,
    flush_line_batch,
    line_shadow_depth_bias,
)
from builder.heightfield.vector.polyline import PolylineShapeParams
from builder.heightfield.vector.rasterize import (
    find_vector_layer_ancestor,
    find_vector_shape_ancestor,
)
from builder.heightfield.vector.rectangle import rectangle_corners
from builder.path.points import densify_path_positions
from builder.heightfield.vector.vector_types import (
    VectorShapeRecipe,
    world_xz_to_uv,
)
from builder.scene.scene_types import Node
from builder.scene.transforms import matrix_translation, world_matrix


overlay_color: Color = Color(80, 200, 255, 220)
shadow_color: Color = Color(30, 50, 70, 200)
shadow_handle_color: Color = Color(180, 220, 255, 255)
stem_color: Color = Color(120, 180, 220, 200)
point_color: Color = Color(255, 220, 80, 255)
point_selected_color: Color = Color(255, 140, 40, 255)
plane_color: Color = Color(80, 180, 255, 36)
# subdivide long segments so the shadow drapes over hills
overlay_step_metres: float = 2.0
_min_segment_length: float = 1e-6


def surface_y(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    x: float,
    z: float,
    size_x: float,
    size_z: float,
    lift: float,
    *,
    below_layer_index: int | None = None,
) -> float:
    """composited height at local xz, plus overlay lift

    when below_layer_index is set, sample only the stack under that edit layer
    """
    u: float
    v: float
    u, v = world_xz_to_uv(x, z, size_x, size_z)
    if below_layer_index is None:
        return sample_height(catalog, heightfield, u, v) + lift
    return (
        sample_height_below(
            catalog, heightfield, below_layer_index, u, v
        )
        + lift
    )


def draw_segment_flat(
    batch: LineBatch,
    ax: float,
    az: float,
    bx: float,
    bz: float,
    plane_y: float,
    thickness_px: float,
    color: Color,
) -> None:
    """queue one xz segment on the work plane"""
    add_line_segment(
        batch,
        ax,
        plane_y,
        az,
        bx,
        plane_y,
        bz,
        color,
        thickness_px,
    )


def draw_segment_draped(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    ax: float,
    az: float,
    bx: float,
    bz: float,
    size_x: float,
    size_z: float,
    thickness_px: float,
    color: Color,
    lift: float,
    *,
    below_layer_index: int | None = None,
) -> None:
    """queue one xz segment draped onto the terrain"""
    dx: float = bx - ax
    dz: float = bz - az
    length: float = hypot(dx, dz)
    if length < _min_segment_length:
        return
    steps: int = max(1, int(length / overlay_step_metres))
    s: int
    for s in range(steps):
        t0: float = float(s) / float(steps)
        t1: float = float(s + 1) / float(steps)
        x0: float = ax + dx * t0
        z0: float = az + dz * t0
        x1: float = ax + dx * t1
        z1: float = az + dz * t1
        y0: float = surface_y(
            catalog,
            heightfield,
            x0,
            z0,
            size_x,
            size_z,
            lift,
            below_layer_index=below_layer_index,
        )
        y1: float = surface_y(
            catalog,
            heightfield,
            x1,
            z1,
            size_x,
            size_z,
            lift,
            below_layer_index=below_layer_index,
        )
        add_line_segment(
            batch, x0, y0, z0, x1, y1, z1, color, thickness_px
        )


def shape_polyline_points(
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
) -> tuple[list[tuple[float, float]], bool] | None:
    """xz polyline and closed flag for a shape, or none if nothing to draw"""
    recipe: VectorShapeRecipe | None = shape.vector_shape
    if recipe is None or not recipe.enabled:
        return None
    points: list[tuple[float, float]] = vector_point_xz(
        nodes, shape.id, heightfield_group_id
    )
    match recipe.kind:
        case "polyline":
            closed: bool = False
            match recipe.params:
                case PolylineShapeParams(closed=closed_param):
                    closed = bool(closed_param.value)
                case _:
                    pass
            densify_step: float = max(
                overlay_step_metres,
                float(recipe.radius.value) * 0.25,
            )
            densified = densify_path_positions(
                nodes,
                shape.id,
                heightfield_group_id,
                closed=closed,
                step_metres=densify_step,
            )
            curve: list[tuple[float, float]] = [
                (float(p.x), float(p.z)) for p in densified
            ]
            return curve, closed
        case "circle":
            if len(points) == 0:
                return None
            radius: float = 8.0
            match recipe.params:
                case CircleShapeParams(radius=radius_param):
                    radius = float(radius_param.value)
                case _:
                    pass
            cx: float = points[0][0]
            cz: float = points[0][1]
            circumference: float = 2.0 * pi * max(radius, 0.1)
            count: int = max(32, int(circumference / overlay_step_metres))
            rim: list[tuple[float, float]] = []
            i: int
            for i in range(count):
                angle: float = (2.0 * pi * float(i)) / float(count)
                rim.append((cx + cos(angle) * radius, cz + sin(angle) * radius))
            return rim, True
        case "rectangle":
            corners = rectangle_corners(points)
            if corners is None:
                return None
            return corners, True
        case _:
            return points, False


def draw_polyline_shadow(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    points: list[tuple[float, float]],
    closed: bool,
    size_x: float,
    size_z: float,
    thickness_px: float,
    lift: float,
    *,
    below_layer_index: int | None = None,
) -> None:
    """drape polyline segments onto the terrain"""
    if len(points) < 2:
        return
    i: int
    for i in range(len(points) if closed else len(points) - 1):
        a = points[i]
        b = points[(i + 1) % len(points)]
        draw_segment_draped(
            batch,
            catalog,
            heightfield,
            a[0],
            a[1],
            b[0],
            b[1],
            size_x,
            size_z,
            thickness_px,
            shadow_color,
            lift,
            below_layer_index=below_layer_index,
        )


def draw_polyline_flat(
    batch: LineBatch,
    points: list[tuple[float, float]],
    closed: bool,
    plane_y: float,
    thickness_px: float,
    color: Color,
) -> None:
    """queue polyline segments on the work plane"""
    if len(points) < 2:
        return
    i: int
    for i in range(len(points) if closed else len(points) - 1):
        a = points[i]
        b = points[(i + 1) % len(points)]
        draw_segment_flat(
            batch,
            a[0],
            a[1],
            b[0],
            b[1],
            plane_y,
            thickness_px,
            color,
        )


def draw_shape_shadow(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
    size_x: float,
    size_z: float,
    thickness_px: float,
    lift: float,
    *,
    below_layer_index: int | None = None,
) -> None:
    """terrain shadow for one vector shape"""
    polyline = shape_polyline_points(nodes, shape, heightfield_group_id)
    if polyline is None:
        return
    points: list[tuple[float, float]]
    closed: bool
    points, closed = polyline
    draw_polyline_shadow(
        batch,
        catalog,
        heightfield,
        points,
        closed,
        size_x,
        size_z,
        thickness_px,
        lift,
        below_layer_index=below_layer_index,
    )


def draw_road_point_stems(
    batch: LineBatch,
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
    layer_id: str,
    size_x: float,
    size_z: float,
    selected_ids: set[str],
    stem_thickness_px: float,
    cube_size: float,
    lift: float,
) -> None:
    """shadow cubes, stems, and ribbon-height markers for road points"""
    road_index: int | None = vector_road_layer_index(heightfield, layer_id)
    if road_index is None:
        return
    entry: tuple[str, float, float, float]
    for entry in vector_point_entries_xzy(
        nodes, shape.id, heightfield_group_id
    ):
        point_id: str
        px: float
        pz: float
        offset_y: float
        point_id, px, pz, offset_y = entry
        shadow: float = surface_y(
            catalog,
            heightfield,
            px,
            pz,
            size_x,
            size_z,
            lift,
            below_layer_index=road_index,
        )
        u: float
        v: float
        u, v = world_xz_to_uv(px, pz, size_x, size_z)
        # offset is relative to terrain under the road layer
        ribbon: float = (
            sample_height_below(catalog, heightfield, road_index, u, v)
            + offset_y
        )
        handle_color: Color = (
            point_selected_color
            if point_id in selected_ids
            else shadow_handle_color
        )
        draw_cube(
            Vector3(px, shadow, pz),
            cube_size,
            cube_size,
            cube_size,
            handle_color,
        )
        add_line_segment(
            batch,
            px,
            shadow,
            pz,
            px,
            ribbon,
            pz,
            stem_color,
            stem_thickness_px,
        )
        draw_cube(
            Vector3(px, ribbon, pz),
            cube_size * 0.85,
            cube_size * 0.85,
            cube_size * 0.85,
            handle_color,
        )


def draw_shape_flat(
    batch: LineBatch,
    nodes: dict[str, Node],
    shape: Node,
    heightfield_group_id: str,
    plane_y: float,
    selected_ids: set[str],
    thickness_px: float,
    cube_size: float,
) -> None:
    """work-plane lines and point cubes for one vector shape"""
    recipe: VectorShapeRecipe | None = shape.vector_shape
    polyline = shape_polyline_points(nodes, shape, heightfield_group_id)
    if polyline is not None:
        points: list[tuple[float, float]]
        closed: bool
        points, closed = polyline
        draw_polyline_flat(
            batch, points, closed, plane_y, thickness_px, overlay_color
        )
    entry: tuple[str, float, float]
    for entry in vector_point_entries(
        nodes, shape.id, heightfield_group_id
    ):
        point_id: str
        px: float
        pz: float
        point_id, px, pz = entry
        color: Color = (
            point_selected_color if point_id in selected_ids else point_color
        )
        draw_cube(
            Vector3(px, plane_y, pz),
            cube_size,
            cube_size,
            cube_size,
            color,
        )
    if (
        recipe is not None
        and recipe.kind == "circle"
        and shape.id in selected_ids
    ):
        points_xz: list[tuple[float, float]] = vector_point_xz(
            nodes, shape.id, heightfield_group_id
        )
        if len(points_xz) > 0:
            shape_radius: float = 8.0
            match recipe.params:
                case CircleShapeParams(radius=radius_param):
                    shape_radius = float(radius_param.value)
                case _:
                    pass
            rim_x: float
            rim_z: float
            rim_x, rim_z = circle_rim_local_xz(
                points_xz[0][0], points_xz[0][1], shape_radius
            )
            draw_cube(
                Vector3(rim_x, plane_y, rim_z),
                cube_size,
                cube_size,
                cube_size,
                point_selected_color,
            )


def draw_work_plane(
    size_x: float, size_z: float, plane_y: float
) -> None:
    """cheap translucent xz quad at the layer work-plane height"""
    begin_blend_mode(BlendMode.BLEND_ALPHA)
    rl_disable_depth_mask()
    draw_plane(
        Vector3(0.0, plane_y, 0.0),
        Vector2(size_x, size_z),
        plane_color,
    )
    rl_enable_depth_mask()
    end_blend_mode()


def selected_vector_layer_id(
    nodes: dict[str, Node],
    heightfield_group_id: str,
    selected_ids: Sequence[str],
) -> str | None:
    """vector layer under this heightfield touched by the selection, if any"""
    selected_id: str
    for selected_id in selected_ids:
        layer_id: str | None = find_vector_layer_ancestor(nodes, selected_id)
        if layer_id is None:
            continue
        layer: Node | None = nodes.get(layer_id)
        if layer is None or layer.parent_id != heightfield_group_id:
            continue
        return layer_id
    return None


def selected_vector_shape_id(
    nodes: dict[str, Node],
    heightfield_group_id: str,
    selected_ids: Sequence[str],
) -> str | None:
    """vector shape under this heightfield touched by the selection, if any"""
    selected_id: str
    for selected_id in selected_ids:
        shape_id: str | None = find_vector_shape_ancestor(nodes, selected_id)
        if shape_id is None:
            continue
        shape: Node | None = nodes.get(shape_id)
        if shape is None or shape.parent_id is None:
            continue
        layer: Node | None = nodes.get(shape.parent_id)
        if layer is None or layer.parent_id != heightfield_group_id:
            continue
        if layer.vector_layer is None:
            continue
        return shape_id
    return None


def draw_vector_overlays(
    catalog: HeightmapCatalog,
    nodes: dict[str, Node],
    heightfield_group: Node,
    camera: Camera3D,
    viewport_height_px: float,
    line_shader: LineShader,
    line_batch: LineBatch,
    selected_ids: Sequence[str] = (),
) -> None:
    """draw work-plane shapes, terrain shadows, and point cubes for one heightfield"""
    heightfield: Heightfield | None = heightfield_group.heightfield
    if heightfield is None:
        return
    selected_layer_id: str | None = selected_vector_layer_id(
        nodes, heightfield_group.id, selected_ids
    )
    if selected_layer_id is None:
        return
    layer: Node | None = nodes.get(selected_layer_id)
    if (
        layer is None
        or layer.vector_layer is None
        or not vector_layer_is_linked(heightfield, layer.id)
    ):
        return
    selected_shape_id: str | None = selected_vector_shape_id(
        nodes, heightfield_group.id, selected_ids
    )
    layout = heightfield_layout(heightfield)
    size_x: float = layout.size_x
    size_z: float = layout.size_z
    selected_set: set[str] = set(selected_ids)
    hf_world: Matrix = world_matrix(nodes, heightfield_group.id)
    distance: float = vector3_distance(
        camera.position, matrix_translation(hf_world)
    )
    overlay_radius: float = overlay_radius_for_camera(
        camera, distance, viewport_height_px, overlay_line_pixels
    )
    cube_size: float = point_cube_size_for_line_radius(overlay_radius)
    clear_line_batch(line_batch)
    # pyray's rl_mult_matrixf wants float*; matrix_to_float_v returns float16
    rl_push_matrix()
    rl_mult_matrixf(ffi.cast("float *", matrix_to_float_v(hf_world).v))
    plane_y: float = vector_layer_plane_y(
        nodes, layer.id, heightfield_group.id
    )
    shapes: list[Node] = [
        shape
        for shape in ordered_child_nodes(nodes, layer.id)
        if shape.vector_shape is not None
    ]
    is_road: bool = vector_layer_is_road(heightfield, layer.id)
    road_index: int | None = (
        vector_road_layer_index(heightfield, layer.id) if is_road else None
    )
    # shadow only for the selected shape, before the translucent plane
    if selected_shape_id is not None:
        shape: Node
        for shape in shapes:
            if shape.id != selected_shape_id:
                continue
            draw_shape_shadow(
                line_batch,
                catalog,
                heightfield,
                nodes,
                shape,
                heightfield_group.id,
                size_x,
                size_z,
                shadow_line_pixels,
                0.0,
                below_layer_index=road_index,
            )
            if is_road:
                draw_road_point_stems(
                    line_batch,
                    catalog,
                    heightfield,
                    nodes,
                    shape,
                    heightfield_group.id,
                    layer.id,
                    size_x,
                    size_z,
                    selected_set,
                    shadow_line_pixels * 0.6,
                    cube_size,
                    0.0,
                )
            break
        flush_line_batch(
            line_batch, line_shader, depth_bias=line_shadow_depth_bias
        )
    draw_work_plane(size_x, size_z, plane_y)
    shape_flat: Node
    for shape_flat in shapes:
        draw_shape_flat(
            line_batch,
            nodes,
            shape_flat,
            heightfield_group.id,
            plane_y,
            selected_set,
            overlay_line_pixels,
            cube_size,
        )
    flush_line_batch(line_batch, line_shader)
    rl_pop_matrix()
