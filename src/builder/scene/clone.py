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
    """copy a modifier, detaching its param map from the original"""
    return replace(modifier, params=dict(modifier.params))


def clone_generator(generator: Generator | None) -> Generator | None:
    """copy a generator, detaching params and modifiers from the original"""
    if generator is None:
        return None
    return replace(
        generator,
        params=dict(generator.params),
        modifiers=tuple(clone_modifier(modifier) for modifier in generator.modifiers),
    )


def clone_subtrees(
    nodes: dict[str, Node],
    root_ids: Sequence[str],
) -> tuple[list[Node], list[str]]:
    """return copies of each root and its descendants, plus the new root ids

    descendants are re-parented onto their copied parent; roots keep theirs
    """
    ordered: list[str] = subtree_ids(nodes, root_ids)
    id_map: dict[str, str] = {node_id: new_node_id() for node_id in ordered}
    roots: set[str] = set(root_ids)
    clones: list[Node] = []
    new_roots: list[str] = []
    node_id: str
    for node_id in ordered:
        node: Node | None = nodes.get(node_id)
        if node is None:
            continue
        parent_id: str | None = node.parent_id
        new_parent_id: str | None = (
            id_map.get(parent_id, parent_id) if parent_id is not None else None
        )
        clones.append(
            replace(
                node,
                id=id_map[node_id],
                parent_id=new_parent_id,
                transform=copy_transform(node.transform),
                generator=clone_generator(node.generator),
            )
        )
        if node_id in roots:
            new_roots.append(id_map[node_id])
    return clones, new_roots
