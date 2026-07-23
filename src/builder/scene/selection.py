"""Selection helpers for group-only editing."""

from __future__ import annotations

from builder.scene.scene_types import Node


def root_group_id(nodes: dict[str, Node], node_id: str) -> str:
    """ return id of the root ancestor for a node """
    current_id: str = node_id
    while True:
        node: Node = nodes[current_id]
        if node.parent_id is None:
            return current_id
        current_id = node.parent_id
