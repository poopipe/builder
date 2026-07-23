"""CPU scene: a collection of nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from builder.scene.scene_types import Node


@dataclass
class Scene:
    """Mutable node collection. Commands mutate this; the view only reads it."""

    nodes: dict[str, Node] = field(default_factory=dict)

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """Insert or replace nodes by id."""
        node: Node
        for node in nodes:
            self.nodes[node.id] = node

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        """Remove nodes by id; missing ids are ignored."""
        node_id: str
        for node_id in node_ids:
            self.nodes.pop(node_id, None)

    def clear_nodes(self) -> None:
        """Remove every node."""
        self.nodes.clear()

    def all_nodes(self) -> list[Node]:
        """Return nodes in insertion order (dict order)."""
        return list(self.nodes.values())
