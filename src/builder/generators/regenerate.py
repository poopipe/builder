"""regenerate and bake parametric group children"""

from __future__ import annotations

from dataclasses import replace

from builder.generators.generator_types import (
    BuildContext,
    GeneratedSlot,
    Generator,
    GeneratorSpec,
    MeshPattern,
    Modifier,
    ParamField,
    ParamMap,
    ParamValue,
)
from builder.generators.mesh_pattern import mesh_for_index
from builder.generators.orient import apply_orient_to_slots
from builder.generators.registry import (
    clamp_param,
    field_for,
    get_spec,
    params_with_defaults,
)
from builder.modifiers.apply import apply_modifier_stack
from builder.modifiers.registry import (
    modifier_field_for,
    modifier_params_with_defaults,
)
from builder.scene.ids import new_node_id
from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node


def mesh_child_ids(nodes: dict[str, Node], group_id: str) -> list[str]:
    """return ids of direct meshed children; nested groups are left alone"""
    return [
        node.id
        for node in nodes.values()
        if node.parent_id == group_id and node.mesh_id is not None
    ]


def selected_parametric_group(scene: Scene) -> Node | None:
    """return the sole selected group if it still owns a generator"""
    if len(scene.selected_ids) != 1:
        return None
    group_id: str = scene.selected_ids[0]
    node: Node | None = scene.nodes.get(group_id)
    if node is None or node.generator is None:
        return None
    return node


def pattern_for_slot(generator: Generator, slot: GeneratedSlot) -> MeshPattern:
    """return the mesh pattern for a slot role"""
    if slot.role == "point" and generator.point_meshes is not None:
        return generator.point_meshes
    return generator.meshes


def regenerate_group(scene: Scene, group_id: str) -> None:
    """replace meshed children from the group's generator recipe

    nested group children are preserved so hierarchy survives regenerate/load
    """
    group: Node = scene.nodes[group_id]
    generator: Generator | None = group.generator
    if generator is None:
        raise ValueError(f"group {group_id} has no generator")
    spec: GeneratorSpec = get_spec(generator.kind)
    params: ParamMap = params_with_defaults(generator.kind, generator.params)
    if params != generator.params:
        scene.add_nodes(
            [replace(group, generator=replace(generator, params=params))]
        )
        group = scene.nodes[group_id]
        generator = group.generator
        assert generator is not None
    context: BuildContext = BuildContext(nodes=scene.nodes, group_id=group_id)
    slots: list[GeneratedSlot] = apply_orient_to_slots(
        spec.build_transforms(params, context), params
    )
    slots = apply_modifier_stack(slots, generator.modifiers, context)
    scene.remove_nodes(mesh_child_ids(scene.nodes, group_id))
    children: list[Node] = []
    index: int
    slot: GeneratedSlot
    role_index: dict[str, int] = {"default": 0, "point": 0, "edge": 0}
    for slot in slots:
        index = role_index[slot.role]
        role_index[slot.role] = index + 1
        pattern: MeshPattern = pattern_for_slot(generator, slot)
        mesh_id: MeshId = mesh_for_index(pattern, index)
        children.append(
            Node(
                id=new_node_id(),
                parent_id=group_id,
                transform=slot.transform,
                mesh_id=mesh_id,
                generator=None,
            )
        )
    scene.add_nodes(children)


def bake_group(scene: Scene, group_id: str) -> None:
    """drop the generator recipe; children remain as a static group"""
    group: Node = scene.nodes[group_id]
    if group.generator is None:
        return
    scene.add_nodes([replace(group, generator=None)])


def set_generator_param(
    scene: Scene,
    group_id: str,
    key: str,
    value: ParamValue,
) -> None:
    """update one generator param and regenerate children"""
    group: Node = scene.nodes[group_id]
    generator: Generator | None = group.generator
    if generator is None:
        raise ValueError(f"group {group_id} has no generator")
    spec: GeneratorSpec = get_spec(generator.kind)
    field: ParamField = field_for(spec, key)
    clamped: ParamValue = clamp_param(field, value)
    params: ParamMap = params_with_defaults(generator.kind, generator.params)
    params[key] = clamped
    scene.add_nodes([replace(group, generator=replace(generator, params=params))])
    regenerate_group(scene, group_id)


def step_generator_param(
    scene: Scene,
    group_id: str,
    key: str,
    direction: int,
) -> None:
    """nudge one generator param by its field step and regenerate"""
    group: Node = scene.nodes[group_id]
    generator: Generator | None = group.generator
    if generator is None:
        raise ValueError(f"group {group_id} has no generator")
    spec: GeneratorSpec = get_spec(generator.kind)
    field: ParamField = field_for(spec, key)
    params: ParamMap = params_with_defaults(generator.kind, generator.params)
    current: ParamValue = params[key]
    stepped: float = float(current) + field.step * float(direction)
    set_generator_param(scene, group_id, key, stepped)


def require_generator(scene: Scene, group_id: str) -> Generator:
    """return the group's generator or raise if it has none"""
    generator: Generator | None = scene.nodes[group_id].generator
    if generator is None:
        raise ValueError(f"group {group_id} has no generator")
    return generator


def set_generator_pattern(
    scene: Scene,
    group_id: str,
    pattern: MeshPattern,
    *,
    points: bool = False,
) -> None:
    """replace the edge or point mesh pattern and regenerate children"""
    group: Node = scene.nodes[group_id]
    generator: Generator = require_generator(scene, group_id)
    if points:
        scene.add_nodes(
            [replace(group, generator=replace(generator, point_meshes=pattern))]
        )
    else:
        scene.add_nodes(
            [replace(group, generator=replace(generator, meshes=pattern))]
        )
    regenerate_group(scene, group_id)


def set_generator_modifiers(
    scene: Scene,
    group_id: str,
    modifiers: tuple[Modifier, ...],
) -> None:
    """replace the modifier stack and regenerate children"""
    group: Node = scene.nodes[group_id]
    generator: Generator = require_generator(scene, group_id)
    scene.add_nodes(
        [replace(group, generator=replace(generator, modifiers=modifiers))]
    )
    regenerate_group(scene, group_id)


def set_modifier_param(
    scene: Scene,
    group_id: str,
    index: int,
    key: str,
    value: ParamValue,
) -> None:
    """update one param on one modifier and regenerate"""
    group: Node = scene.nodes[group_id]
    generator: Generator = require_generator(scene, group_id)
    if index < 0 or index >= len(generator.modifiers):
        raise IndexError(f"modifier index {index} out of range")
    modifier: Modifier = generator.modifiers[index]
    field = modifier_field_for(modifier.kind, key)
    clamped: ParamValue = clamp_param(field, value)
    params: ParamMap = modifier_params_with_defaults(modifier.kind, modifier.params)
    params[key] = clamped
    updated: list[Modifier] = list(generator.modifiers)
    updated[index] = replace(modifier, params=params)
    set_generator_modifiers(scene, group_id, tuple(updated))


def step_modifier_param(
    scene: Scene,
    group_id: str,
    index: int,
    key: str,
    direction: int,
) -> None:
    """nudge one modifier param by its field step and regenerate"""
    group: Node = scene.nodes[group_id]
    generator: Generator = require_generator(scene, group_id)
    if index < 0 or index >= len(generator.modifiers):
        raise IndexError(f"modifier index {index} out of range")
    modifier: Modifier = generator.modifiers[index]
    field = modifier_field_for(modifier.kind, key)
    params: ParamMap = modifier_params_with_defaults(modifier.kind, modifier.params)
    current: ParamValue = params[key]
    stepped: float = float(current) + field.step * float(direction)
    set_modifier_param(scene, group_id, index, key, stepped)


def spline_generator_id(nodes: dict[str, Node], node_id: str) -> str | None:
    """return ancestor id that owns a spline generator, if any"""
    current_id: str | None = node_id
    while current_id is not None:
        node: Node | None = nodes.get(current_id)
        if node is None:
            return None
        if node.generator is not None and node.generator.kind == "spline":
            return current_id
        current_id = node.parent_id
    return None


def regenerate_splines_touching(
    scene: Scene, node_ids: list[str] | set[str]
) -> None:
    """regenerate any spline generator affected by edits to node_ids"""
    seen: set[str] = set()
    node_id: str
    for node_id in node_ids:
        spline_id: str | None = spline_generator_id(scene.nodes, node_id)
        if spline_id is None or spline_id in seen:
            continue
        seen.add(spline_id)
        regenerate_group(scene, spline_id)
