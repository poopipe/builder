"""selection helpers for group-only editing"""

from __future__ import annotations

from collections.abc import Iterable

from builder.scene.scene_types import Node


def root_group_id(nodes: dict[str, Node], node_id: str) -> str:
    """return id of the root ancestor for a node"""
    current_id: str = node_id
    while True:
        node: Node = nodes[current_id]
        if node.parent_id is None:
            return current_id
        current_id = node.parent_id


def subtree_ids(nodes: dict[str, Node], root_ids: Iterable[str]) -> list[str]:
    """return each root id plus all descendant ids"""
    children: dict[str, list[str]] = {}
    node: Node
    for node in nodes.values():
        if node.parent_id is None:
            continue
        children.setdefault(node.parent_id, []).append(node.id)
    result: list[str] = []
    seen: set[str] = set()
    stack: list[str] = list(root_ids)
    while stack:
        node_id: str = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        result.append(node_id)
        stack.extend(children.get(node_id, []))
    return result
