"""regenerate and bake parametric group children"""

from __future__ import annotations

from dataclasses import replace

from pyray import Transform

from builder.generators.generator_types import (
    Generator,
    GeneratorSpec,
    ParamField,
    ParamMap,
    ParamValue,
)
from builder.generators.registry import clamp_param, field_for, get_spec
from builder.scene.ids import new_node_id
from builder.scene.scene import Scene
from builder.scene.scene_types import Node


def child_ids(nodes: dict[str, Node], group_id: str) -> list[str]:
    """return ids of direct children of a group"""
    return [node.id for node in nodes.values() if node.parent_id == group_id]


def selected_parametric_group(scene: Scene) -> Node | None:
    """return the sole selected group if it still owns a generator"""
    if len(scene.selected_ids) != 1:
        return None
    group_id: str = next(iter(scene.selected_ids))
    node: Node | None = scene.nodes.get(group_id)
    if node is None or node.generator is None:
        return None
    return node


def regenerate_group(scene: Scene, group_id: str) -> None:
    """replace group children from the group's generator recipe"""
    group: Node = scene.nodes[group_id]
    generator: Generator | None = group.generator
    if generator is None:
        raise ValueError(f"group {group_id} has no generator")
    spec: GeneratorSpec = get_spec(generator.kind)
    locals_: list[Transform] = spec.build_transforms(generator.params)
    scene.remove_nodes(child_ids(scene.nodes, group_id))
    children: list[Node] = []
    local: Transform
    for local in locals_:
        children.append(
            Node(
                id=new_node_id(),
                parent_id=group_id,
                transform=local,
                mesh_id=generator.mesh_id,
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
    params: ParamMap = dict(generator.params)
    params[key] = clamped
    scene.add_nodes(
        [replace(group, generator=replace(generator, params=params))]
    )
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
    current: ParamValue = generator.params[key]
    stepped: float = float(current) + field.step * float(direction)
    set_generator_param(scene, group_id, key, stepped)
