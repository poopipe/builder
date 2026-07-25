"""CPU-side matrix / instance-buffer cache for the draw path.

Instance buffers hold LOCAL transforms. Each batch binds its parent's world
matrix as a shader uniform, so moving a group costs one uniform upload instead
of rewriting every child transform.

Structure edits (add/remove) update only the affected batches; existing groups
keep their cached locals and buffers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pyray import Matrix, ffi, matrix_multiply

from builder.scene.scene import RemovedNode, Scene
from builder.scene.scene_types import MeshId, Node
from builder.scene.transforms import transform_matrix

type BatchKey = tuple[MeshId, str | None]


@dataclass
class MeshInstanceBatch:
    """ local instance transforms for one (mesh, parent) draw call """

    transforms: Any
    count: int
    mesh_id: MeshId
    parent_id: str | None
    index_by_id: dict[str, int] = field(default_factory=dict)


@dataclass
class DrawCache:
    """Cached local matrices, parent world matrices, and instance buffers."""

    revision: int = -1
    structure_revision: int = -1
    local: dict[str, Matrix] = field(default_factory=dict)
    world: dict[str, Matrix] = field(default_factory=dict)
    batches: dict[BatchKey, MeshInstanceBatch] = field(default_factory=dict)
    batch_parent_ids: set[str] = field(default_factory=set)
    # parent_id -> child node ids (None key = root-level nodes)
    children: dict[str | None, set[str]] = field(default_factory=lambda: defaultdict(set))


def clear_draw_cache(cache: DrawCache) -> None:
    """ drop all cached GPU/CPU draw state (used on teardown) """
    cache.local.clear()
    cache.world.clear()
    cache.batches.clear()
    cache.batch_parent_ids.clear()
    cache.children.clear()
    cache.revision = -1
    cache.structure_revision = -1


def world_matrix_for(scene: Scene, cache: DrawCache, node_id: str) -> Matrix:
    """ return a node's world matrix, composing from the cached parent world """
    cached: Matrix | None = cache.world.get(node_id)
    if cached is not None:
        return cached
    node: Node = scene.nodes[node_id]
    local: Matrix = cache.local[node_id]
    if node.parent_id is None:
        return local
    parent_world: Matrix = world_matrix_for(scene, cache, node.parent_id)
    return matrix_multiply(local, parent_world)


def sync_draw_cache(scene: Scene, cache: DrawCache) -> None:
    """ rebuild or incrementally update caches when the scene revision changed """
    if cache.revision == scene.revision:
        return
    if cache.revision < 0 or cache.structure_revision != scene.structure_revision:
        rebuild_all(scene, cache)
        cache.structure_revision = scene.structure_revision
    else:
        apply_structure_delta(scene, cache)
        apply_incremental_updates(scene, cache)
    scene.dirty_ids.clear()
    scene.added_ids.clear()
    scene.removed.clear()
    cache.revision = scene.revision


def rebuild_all(scene: Scene, cache: DrawCache) -> None:
    """ recompute locals, child index, batches, and parent worlds from scratch """
    cache.local.clear()
    cache.world.clear()
    cache.children = defaultdict(set)
    node_id: str
    node: Node
    for node_id, node in scene.nodes.items():
        cache.local[node_id] = transform_matrix(node.transform)
        cache.children[node.parent_id].add(node_id)
    rebuild_instance_batches(scene, cache)
    refresh_batch_parent_ids(cache)
    parent_id: str
    for parent_id in cache.batch_parent_ids:
        fill_world_matrix(scene, cache, parent_id)


def refresh_batch_parent_ids(cache: DrawCache) -> None:
    """ refresh the set of parents referenced by instance batches """
    cache.batch_parent_ids = {
        batch.parent_id
        for batch in cache.batches.values()
        if batch.parent_id is not None
    }


def fill_world_matrix(scene: Scene, cache: DrawCache, node_id: str) -> Matrix:
    """ memoized world matrix built from already-cached local matrices """
    cached: Matrix | None = cache.world.get(node_id)
    if cached is not None:
        return cached
    node: Node = scene.nodes[node_id]
    local: Matrix = cache.local[node_id]
    if node.parent_id is None:
        cache.world[node_id] = local
        return local
    parent: Matrix = fill_world_matrix(scene, cache, node.parent_id)
    world: Matrix = matrix_multiply(local, parent)
    cache.world[node_id] = world
    return world


def has_dirty_ancestor(scene: Scene, node_id: str, dirty: set[str]) -> bool:
    """ true if the node or any ancestor has a changed transform """
    current_id: str | None = node_id
    while current_id is not None:
        if current_id in dirty:
            return True
        node: Node | None = scene.nodes.get(current_id)
        if node is None:
            return False
        current_id = node.parent_id
    return False


def apply_structure_delta(scene: Scene, cache: DrawCache) -> None:
    """ apply pending add/remove without touching unrelated groups """
    if not scene.added_ids and not scene.removed:
        return
    affected_keys: set[BatchKey] = set()
    removed: RemovedNode
    for removed in scene.removed:
        cache.local.pop(removed.id, None)
        cache.world.pop(removed.id, None)
        siblings: set[str] | None = cache.children.get(removed.parent_id)
        if siblings is not None:
            siblings.discard(removed.id)
        if removed.mesh_id is not None:
            affected_keys.add((removed.mesh_id, removed.parent_id))
        if removed.id in cache.batch_parent_ids:
            key: BatchKey
            for key in list(cache.batches):
                if key[1] == removed.id:
                    affected_keys.add(key)

    node_id: str
    for node_id in scene.added_ids:
        node: Node | None = scene.nodes.get(node_id)
        if node is None:
            continue
        cache.local[node_id] = transform_matrix(node.transform)
        cache.children[node.parent_id].add(node_id)
        if node.mesh_id is not None:
            affected_keys.add((node.mesh_id, node.parent_id))

    rebuild_batches(scene, cache, affected_keys)
    refresh_batch_parent_ids(cache)
    parent_id: str
    for parent_id in cache.batch_parent_ids:
        if parent_id not in cache.world and parent_id in scene.nodes:
            fill_world_matrix(scene, cache, parent_id)


def rebuild_batches(
    scene: Scene,
    cache: DrawCache,
    keys: Iterable[BatchKey],
) -> None:
    """ rebuild only the named batches from the child index """
    key: BatchKey
    for key in keys:
        mesh_id: MeshId
        parent_id: str | None
        mesh_id, parent_id = key
        member_ids: set[str] = cache.children.get(parent_id, set())
        members: list[Node] = []
        child_id: str
        for child_id in member_ids:
            node: Node | None = scene.nodes.get(child_id)
            if node is not None and node.mesh_id == mesh_id:
                members.append(node)
        if not members:
            cache.batches.pop(key, None)
            continue
        fill_batch(cache, key, members)


def fill_batch(cache: DrawCache, key: BatchKey, members: list[Node]) -> None:
    """ allocate/reuse a Matrix buffer and fill with member locals """
    count: int = len(members)
    existing: MeshInstanceBatch | None = cache.batches.get(key)
    transforms: Any
    if existing is not None and existing.count == count:
        transforms = existing.transforms
    else:
        transforms = ffi.new("Matrix[]", count)
    index_by_id: dict[str, int] = {}
    index: int
    node: Node
    for index, node in enumerate(members):
        transforms[index] = cache.local[node.id]
        index_by_id[node.id] = index
    cache.batches[key] = MeshInstanceBatch(
        transforms=transforms,
        count=count,
        mesh_id=key[0],
        parent_id=key[1],
        index_by_id=index_by_id,
    )


def apply_incremental_updates(scene: Scene, cache: DrawCache) -> None:
    """ refresh only the transforms invalidated by this frame's edits """
    dirty: set[str] = {
        node_id for node_id in scene.dirty_ids if node_id in scene.nodes
    }
    if not dirty:
        return
    node_id: str
    for node_id in dirty:
        cache.local[node_id] = transform_matrix(scene.nodes[node_id].transform)

    stale: list[str] = [
        cached_id
        for cached_id in cache.world
        if cached_id not in scene.nodes
        or has_dirty_ancestor(scene, cached_id, dirty)
    ]
    for node_id in stale:
        cache.world.pop(node_id, None)
    parent_id: str
    for parent_id in cache.batch_parent_ids:
        if parent_id in scene.nodes:
            fill_world_matrix(scene, cache, parent_id)

    for node_id in dirty:
        write_local_instance_slot(scene, cache, node_id)


def write_local_instance_slot(scene: Scene, cache: DrawCache, node_id: str) -> None:
    """ patch one node's local matrix into its instance buffer slot """
    node: Node | None = scene.nodes.get(node_id)
    if node is None or node.mesh_id is None:
        return
    batch: MeshInstanceBatch | None = cache.batches.get((node.mesh_id, node.parent_id))
    if batch is None:
        return
    index: int | None = batch.index_by_id.get(node_id)
    if index is None:
        return
    batch.transforms[index] = cache.local[node_id]


def rebuild_instance_batches(scene: Scene, cache: DrawCache) -> None:
    """ rebuild every batch from the child index (full sync only) """
    keys: set[BatchKey] = set()
    parent_id: str | None
    child_ids: set[str]
    for parent_id, child_ids in cache.children.items():
        child_id: str
        for child_id in child_ids:
            node: Node | None = scene.nodes.get(child_id)
            if node is not None and node.mesh_id is not None:
                keys.add((node.mesh_id, parent_id))
    cache.batches.clear()
    rebuild_batches(scene, cache, keys)
