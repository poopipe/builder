"""heightfield-local positions for vector layer nodes"""

from __future__ import annotations

from pyray import Matrix, Vector3, matrix_invert, matrix_multiply

from builder.heightfield.heightfield_types import (
    Heightfield,
    VectorHeightSource,
    VectorRoadHeightSource,
    VectorSculptHeightSource,
)
from builder.scene.scene_types import Node, NodeRole
from builder.scene.transforms import matrix_translation, world_matrix


def ordered_child_nodes(
    nodes: dict[str, Node], parent_id: str
) -> list[Node]:
    """direct children in scene insertion order"""
    return [node for node in nodes.values() if node.parent_id == parent_id]


def node_hf_local_matrix(
    nodes: dict[str, Node], node_id: str, heightfield_group_id: str
) -> Matrix:
    """node world pose expressed in the heightfield group's local frame"""
    node_world: Matrix = world_matrix(nodes, node_id)
    hf_world: Matrix = world_matrix(nodes, heightfield_group_id)
    # pyray multiply(a, b) => math b@a; want inv(hf) @ node
    return matrix_multiply(node_world, matrix_invert(hf_world))


def node_hf_local_position(
    nodes: dict[str, Node], node_id: str, heightfield_group_id: str
) -> Vector3:
    """heightfield-local origin of a node"""
    return matrix_translation(
        node_hf_local_matrix(nodes, node_id, heightfield_group_id)
    )


def vector_layer_plane_y(
    nodes: dict[str, Node], layer_id: str, heightfield_group_id: str
) -> float:
    """work-plane height from the vector layer node in heightfield-local space"""
    return float(
        node_hf_local_position(nodes, layer_id, heightfield_group_id).y
    )


def vector_point_xz(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[float, float]]:
    """ordered vector_point children as heightfield-local xz"""
    points: list[tuple[float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.vector_point:
            continue
        local: Vector3 = node_hf_local_position(
            nodes, child.id, heightfield_group_id
        )
        points.append((float(local.x), float(local.z)))
    return points


def vector_point_xzy(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[float, float, float]]:
    """ordered vector_point children as (hf-local x, z, parent-local y offset)"""
    return [
        (x, z, y)
        for x, z, y, _scale in vector_point_xzyw(
            nodes, shape_id, heightfield_group_id
        )
    ]


def vector_point_width_scale(child: Node) -> float:
    """width scale from a vector_point recipe; default 1 when missing"""
    if child.vector_point is None:
        return 1.0
    return max(0.05, float(child.vector_point.width_scale.value))


def vector_point_xzyw(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[float, float, float, float]]:
    """ordered (hf-local x, z, parent-local y offset, width_scale)"""
    points: list[tuple[float, float, float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.vector_point:
            continue
        local: Vector3 = node_hf_local_position(
            nodes, child.id, heightfield_group_id
        )
        # y offset is parent-local so the workplane does not inflate it
        offset_y: float = float(child.transform.translation.y)
        points.append(
            (
                float(local.x),
                float(local.z),
                offset_y,
                vector_point_width_scale(child),
            )
        )
    return points


def vector_point_entries(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[str, float, float]]:
    """ordered (point_id, x, z) in heightfield-local space"""
    entries: list[tuple[str, float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.vector_point:
            continue
        local: Vector3 = node_hf_local_position(
            nodes, child.id, heightfield_group_id
        )
        entries.append((child.id, float(local.x), float(local.z)))
    return entries


def vector_point_entries_xzy(
    nodes: dict[str, Node],
    shape_id: str,
    heightfield_group_id: str,
) -> list[tuple[str, float, float, float]]:
    """ordered (point_id, hf-local x, z, parent-local y offset)"""
    entries: list[tuple[str, float, float, float]] = []
    child: Node
    for child in ordered_child_nodes(nodes, shape_id):
        if child.role is not NodeRole.vector_point:
            continue
        local: Vector3 = node_hf_local_position(
            nodes, child.id, heightfield_group_id
        )
        offset_y: float = float(child.transform.translation.y)
        entries.append(
            (child.id, float(local.x), float(local.z), offset_y)
        )
    return entries


def vector_layer_is_linked(
    heightfield: Heightfield, layer_node_id: str
) -> bool:
    """true if an edit layer source references this vector layer node"""
    edit = None
    for edit in heightfield.edit_layers:
        match edit.height:
            case VectorHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return True
            case VectorSculptHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return True
            case VectorRoadHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return True
            case _:
                pass
    return False


def vector_layer_is_road(
    heightfield: Heightfield, layer_node_id: str
) -> bool:
    """true if this vector layer is driven by a VectorRoadHeightSource"""
    return vector_road_layer_index(heightfield, layer_node_id) is not None


def vector_road_layer_index(
    heightfield: Heightfield, layer_node_id: str
) -> int | None:
    """edit-layer index of the VectorRoadHeightSource for this layer, if any"""
    layer_index: int
    edit = None
    for layer_index, edit in enumerate(heightfield.edit_layers):
        match edit.height:
            case VectorRoadHeightSource(layer_node_id=source_id) if (
                source_id == layer_node_id
            ):
                return layer_index
            case _:
                pass
    return None


def vector_point_belongs_to_road(
    nodes: dict[str, Node], point_id: str
) -> bool:
    """true when a vector_point sits under a road-linked vector layer"""
    layer_id: str | None = None
    hf_id: str | None = None
    current_id: str | None = point_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return False
        if layer_id is None and node.vector_layer is not None:
            layer_id = current_id
        if node.heightfield is not None:
            hf_id = current_id
            break
        current_id = node.parent_id
    if layer_id is None or hf_id is None:
        return False
    hf_node: Node | None = nodes.get(hf_id)
    if hf_node is None or hf_node.heightfield is None:
        return False
    return vector_layer_is_road(hf_node.heightfield, layer_id)
