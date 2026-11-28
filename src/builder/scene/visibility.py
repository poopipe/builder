"""effective visibility for scene nodes"""

from __future__ import annotations

from builder.scene.scene_types import Node


def node_effectively_visible(nodes: dict[str, Node], node_id: str) -> bool:
    """true when the node and every ancestor has visible=True"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return False
        if not node.visible:
            return False
        current_id = node.parent_id
    return True
