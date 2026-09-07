"""copy node subtrees with fresh ids"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from pyray import Transform

from builder.generators.generator_types import Generator, Modifier
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node
from builder.scene.selection import subtree_ids


def copy_transform(transform: Transform) -> Transform:
    """return a transform owning its own storage

    Transform is a ctypes struct, so a copy keeps the clone independent
    """
    return Transform(transform.translation, transform.rotation, transform.scale)


def clone_modifier(modifier: Modifier) -> Modifier:
    """copy a modifier; frozen params dataclasses are shared safely"""
    return replace(modifier)


def clone_generator(generator: Generator | None) -> Generator | None:
    """copy a generator; frozen params dataclasses are shared safely"""
    if generator is None:
        return None
    return replace(
        generator,
        modifiers=tuple(clone_modifier(modifier) for modifier in generator.modifiers),
    )


def clone_subtrees(
    nodes: dict[str, Node],
    root_ids: Sequence[str],
) -> tuple[list[Node], list[str]]:
    """return copies of each root and its descendants, plus the new root ids

    descendants are re-parented onto their copied parent; roots keep theirs
    """
    ordered: list[str] = []
    root_id: str
    for root_id in root_ids:
        ordered.extend(subtree_ids(nodes, [root_id]))
    id_map: dict[str, str] = {node_id: new_node_id() for node_id in ordered}
    clones: list[Node] = []
    node_id: str
    for node_id in ordered:
        node: Node = nodes[node_id]
        parent_id: str | None = node.parent_id
        if parent_id is not None and parent_id in id_map:
            parent_id = id_map[parent_id]
        clones.append(
            replace(
                node,
                id=id_map[node_id],
                parent_id=parent_id,
                transform=copy_transform(node.transform),
                generator=clone_generator(node.generator),
            )
        )
    new_roots: list[str] = [id_map[root_id] for root_id in root_ids]
    return clones, new_roots
