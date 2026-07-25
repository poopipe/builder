"""CPU scene: a collection of nodes and group selection"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from builder.scene.scene_types import MeshId, Node


@dataclass(frozen=True)
class RemovedNode:
    """snapshot of a node removed from the scene, for cache invalidation"""

    id: str
    parent_id: str | None
    mesh_id: MeshId | None


@dataclass
class Scene:
    """mutable node collection. Commands mutate this; the view only reads it"""

    nodes: dict[str, Node] = field(default_factory=dict)
    # ordered: the first entry is the target of the parent command
    selected_ids: list[str] = field(default_factory=list)
    revision: int = 0
    structure_revision: int = 0
    dirty_ids: set[str] = field(default_factory=set)
    added_ids: set[str] = field(default_factory=set)
    removed: list[RemovedNode] = field(default_factory=list)

    def bump_revision(self) -> None:
        """invalidate cached draw state for this scene"""
        self.revision += 1

    def mark_structure_changed(self) -> None:
        """force a full draw-cache rebuild on the next sync"""
        self.structure_revision += 1
        self.dirty_ids.clear()
        self.added_ids.clear()
        self.removed.clear()
        self.bump_revision()

    def set_node(self, node: Node) -> None:
        """insert or replace one node; transform-only edits stay incremental"""
        old: Node | None = self.nodes.get(node.id)
        self.nodes[node.id] = node
        if old is None:
            self.added_ids.add(node.id)
            self.bump_revision()
            return
        if old.parent_id != node.parent_id or old.mesh_id != node.mesh_id:
            self.removed.append(
                RemovedNode(id=old.id, parent_id=old.parent_id, mesh_id=old.mesh_id)
            )
            self.added_ids.add(node.id)
            self.bump_revision()
            return
        self.dirty_ids.add(node.id)
        self.bump_revision()

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """insert or replace nodes by id"""
        if not nodes:
            return
        node: Node
        for node in nodes:
            old: Node | None = self.nodes.get(node.id)
            self.nodes[node.id] = node
            if old is None:
                self.added_ids.add(node.id)
                continue
            if old.parent_id != node.parent_id or old.mesh_id != node.mesh_id:
                self.removed.append(
                    RemovedNode(
                        id=old.id, parent_id=old.parent_id, mesh_id=old.mesh_id
                    )
                )
                self.added_ids.add(node.id)
            else:
                self.dirty_ids.add(node.id)
        self.bump_revision()

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        """remove nodes by id; missing ids are ignored"""
        removed_any: bool = False
        node_id: str
        for node_id in node_ids:
            node: Node | None = self.nodes.pop(node_id, None)
            if node is None:
                continue
            if node_id in self.selected_ids:
                self.selected_ids.remove(node_id)
            self.added_ids.discard(node_id)
            self.dirty_ids.discard(node_id)
            self.removed.append(
                RemovedNode(id=node.id, parent_id=node.parent_id, mesh_id=node.mesh_id)
            )
            removed_any = True
        if removed_any:
            self.bump_revision()

    def clear_nodes(self) -> None:
        """remove every node and clear selection"""
        if not self.nodes and not self.selected_ids:
            return
        self.nodes.clear()
        self.selected_ids.clear()
        self.mark_structure_changed()

    def all_nodes(self) -> list[Node]:
        """return nodes in insertion order (dict order)"""
        return list(self.nodes.values())

    def set_selection(self, group_ids: Iterable[str]) -> None:
        """replace the selection, preserving order and dropping duplicates"""
        ordered: list[str] = []
        group_id: str
        for group_id in group_ids:
            if group_id not in ordered:
                ordered.append(group_id)
        self.selected_ids = ordered

    def clear_selection(self) -> None:
        """clear the current selection"""
        self.selected_ids.clear()

    def toggle_selection(self, group_id: str) -> None:
        """add a group id to the end of the selection, or remove it"""
        if group_id in self.selected_ids:
            self.selected_ids.remove(group_id)
        else:
            self.selected_ids.append(group_id)

    def first_selected(self) -> str | None:
        """return the id that parenting treats as the target, or none"""
        return self.selected_ids[0] if self.selected_ids else None
