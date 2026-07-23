"""CPU scene: a collection of nodes and group selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from builder.scene.scene_types import Node


@dataclass
class Scene:
    """Mutable node collection. Commands mutate this; the view only reads it."""

    nodes: dict[str, Node] = field(default_factory=dict)
    selected_ids: set[str] = field(default_factory=set)

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
            self.selected_ids.discard(node_id)

    def clear_nodes(self) -> None:
        """Remove every node and clear selection."""
        self.nodes.clear()
        self.selected_ids.clear()

    def all_nodes(self) -> list[Node]:
        """Return nodes in insertion order (dict order)."""
        return list(self.nodes.values())

    def set_selection(self, group_ids: Iterable[str]) -> None:
        """Replace the selection with the given group ids."""
        self.selected_ids = {group_id for group_id in group_ids}

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self.selected_ids.clear()

    def toggle_selection(self, group_id: str) -> None:
        """Add or remove a group id from the selection."""
        if group_id in self.selected_ids:
            self.selected_ids.discard(group_id)
        else:
            self.selected_ids.add(group_id)
