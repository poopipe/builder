"""pure helpers for sequencing a MeshPattern across generated slots"""

from __future__ import annotations

from dataclasses import replace
from random import Random

from builder.generators.generator_types import MeshPattern, MeshSequenceMode
from builder.scene.scene_types import builtin_cube, MeshId

mesh_sequence_modes: tuple[MeshSequenceMode, ...] = ("repeat", "pingpong", "random")


def mesh_for_index(pattern: MeshPattern, index: int) -> MeshId:
    """return the mesh id for the child at position index"""
    count: int = len(pattern.mesh_ids)
    if count == 0:
        return builtin_cube
    if count == 1:
        return pattern.mesh_ids[0]
    if pattern.mode == "repeat":
        return pattern.mesh_ids[index % count]
    if pattern.mode == "pingpong":
        # reflect without repeating the endpoints: A B C B A B C ...
        period: int = 2 * count - 2
        position: int = index % period
        if position >= count:
            position = period - position
        return pattern.mesh_ids[position]
    # random, but deterministic per (seed, index) so regen/reload is stable
    mixed: int = (pattern.seed * 2654435761 + index * 40503) & 0xFFFFFFFF
    return pattern.mesh_ids[Random(mixed).randrange(count)]


def next_mode(mode: MeshSequenceMode) -> MeshSequenceMode:
    """return the following sequence mode, wrapping around"""
    index: int = mesh_sequence_modes.index(mode)
    return mesh_sequence_modes[(index + 1) % len(mesh_sequence_modes)]


def set_slot(pattern: MeshPattern, slot: int, mesh_id: MeshId) -> MeshPattern:
    """replace the mesh at slot; out-of-range slots are ignored"""
    if slot < 0 or slot >= len(pattern.mesh_ids):
        return pattern
    ids: list[MeshId] = list(pattern.mesh_ids)
    ids[slot] = mesh_id
    return replace(pattern, mesh_ids=tuple(ids))


def remove_slot(pattern: MeshPattern, slot: int) -> MeshPattern:
    """drop the mesh at slot; out-of-range slots are ignored"""
    if slot < 0 or slot >= len(pattern.mesh_ids):
        return pattern
    ids: list[MeshId] = list(pattern.mesh_ids)
    del ids[slot]
    return replace(pattern, mesh_ids=tuple(ids))


def remove_mesh(pattern: MeshPattern, mesh_id: MeshId) -> MeshPattern:
    """drop every slot referencing mesh_id (for a future delete-mesh command)"""
    remaining: tuple[MeshId, ...] = tuple(
        slot for slot in pattern.mesh_ids if slot != mesh_id
    )
    return replace(pattern, mesh_ids=remaining)
