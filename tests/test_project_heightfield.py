"""project heightfield modifier"""

from __future__ import annotations

import unittest

from pyray import (
    Transform,
    Vector3,
    quaternion_identity,
    quaternion_to_matrix,
    vector3_dot_product,
)

from builder.generators.generator_types import BuildContext, GeneratedSlot
from builder.generators.param_types import BoolParam, FloatParam, NameParam
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
)
from builder.modifiers.project_heightfield import (
    ProjectHeightfieldParams,
    project_heightfield_apply,
    resolve_heightfield_node,
    sample_heightfield_normal,
)
from builder.scene.scene_types import Node, transform_at


def flat_heightfield(height: float) -> Heightfield:
    """heightfield with no base map; samples at constant height"""
    return Heightfield(
        base=None,
        base_height_min=FloatParam(height, label="Height Min"),
        base_height_max=FloatParam(height, label="Height Max"),
    )


def slot_at(x: float, y: float, z: float) -> GeneratedSlot:
    """one default-role slot at a translation"""
    return GeneratedSlot(
        Transform(
            Vector3(x, y, z),
            quaternion_identity(),
            Vector3(1.0, 1.0, 1.0),
        )
    )


class TestProjectHeightfield(unittest.TestCase):
    def test_projects_world_y_with_offset(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        hf: Node = Node(
            id="hf",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            heightfield=flat_heightfield(7.0),
        )
        group: Node = Node(
            id="gen",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        nodes: dict[str, Node] = {hf.id: hf, group.id: group}
        context: BuildContext = BuildContext(
            nodes=nodes, group_id=group.id, heightmap_catalog=catalog
        )
        params: ProjectHeightfieldParams = ProjectHeightfieldParams(
            offset=FloatParam(1.5, label="Offset"),
            heightfield_node_id=NameParam("", label="Heightfield"),
        )
        out: list[GeneratedSlot] = project_heightfield_apply(
            [slot_at(3.0, 99.0, -2.0)], params, context
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].transform.translation.x, 3.0)
        self.assertAlmostEqual(out[0].transform.translation.y, 8.5)
        self.assertAlmostEqual(out[0].transform.translation.z, -2.0)

    def test_preserves_relative_generator_y(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        hf: Node = Node(
            id="hf",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            heightfield=flat_heightfield(7.0),
        )
        group: Node = Node(
            id="gen",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        nodes: dict[str, Node] = {hf.id: hf, group.id: group}
        context: BuildContext = BuildContext(
            nodes=nodes, group_id=group.id, heightmap_catalog=catalog
        )
        out: list[GeneratedSlot] = project_heightfield_apply(
            [slot_at(0.0, 0.0, 0.0), slot_at(0.0, 3.0, 0.0)],
            ProjectHeightfieldParams(),
            context,
        )
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].transform.translation.y, 7.0)
        self.assertAlmostEqual(out[1].transform.translation.y, 10.0)

    def test_empty_id_with_one_heightfield(self) -> None:
        nodes: dict[str, Node] = {
            "hf": Node(
                id="hf",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(2.0),
            ),
        }
        picked: Node | None = resolve_heightfield_node(nodes, "")
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.id, "hf")

    def test_empty_id_with_two_heightfields_is_none(self) -> None:
        nodes: dict[str, Node] = {
            "b": Node(
                id="b",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(1.0),
            ),
            "a": Node(
                id="a",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(2.0),
            ),
        }
        self.assertIsNone(resolve_heightfield_node(nodes, ""))

    def test_explicit_id_picks_that_heightfield(self) -> None:
        nodes: dict[str, Node] = {
            "b": Node(
                id="b",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(1.0),
            ),
            "a": Node(
                id="a",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(2.0),
            ),
        }
        picked: Node | None = resolve_heightfield_node(nodes, "b")
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.id, "b")

    def test_empty_id_two_heightfields_apply_is_noop(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        nodes: dict[str, Node] = {
            "a": Node(
                id="a",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(7.0),
            ),
            "b": Node(
                id="b",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
                heightfield=flat_heightfield(9.0),
            ),
            "gen": Node(
                id="gen",
                parent_id=None,
                transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                mesh_id=None,
            ),
        }
        context: BuildContext = BuildContext(
            nodes=nodes, group_id="gen", heightmap_catalog=catalog
        )
        out: list[GeneratedSlot] = project_heightfield_apply(
            [slot_at(0.0, 4.0, 0.0)], ProjectHeightfieldParams(), context
        )
        self.assertAlmostEqual(out[0].transform.translation.y, 4.0)

    def test_missing_catalog_is_noop(self) -> None:
        hf: Node = Node(
            id="hf",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            heightfield=flat_heightfield(7.0),
        )
        group: Node = Node(
            id="gen",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        nodes: dict[str, Node] = {hf.id: hf, group.id: group}
        context: BuildContext = BuildContext(nodes=nodes, group_id=group.id)
        slots: list[GeneratedSlot] = [slot_at(0.0, 4.0, 0.0)]
        out: list[GeneratedSlot] = project_heightfield_apply(
            slots, ProjectHeightfieldParams(), context
        )
        self.assertAlmostEqual(out[0].transform.translation.y, 4.0)

    def test_align_to_normal_matches_slope(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        ramp_id: HeightmapId = HeightmapId("ramp")
        catalog.buffers[ramp_id] = HeightmapBuffer(
            width=3,
            height=3,
            samples=[0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0],
            sample_min=0.0,
            sample_max=10.0,
        )
        heightfield: Heightfield = Heightfield(
            size_x=FloatParam(10.0, label="Size X"),
            size_z=FloatParam(10.0, label="Size Z"),
            base=None,
            base_height_min=FloatParam(0.0, label="Height Min"),
            base_height_max=FloatParam(0.0, label="Height Max"),
            edit_layers=(
                HeightEditLayer(
                    name="ramp",
                    height=CatalogHeightSource(ramp_id),
                    blend=HeightBlendMode.add,
                    opacity=FloatParam(1.0, label="Opacity"),
                    amplitude=1.0,
                    offset=0.0,
                ),
            ),
        )
        hf: Node = Node(
            id="hf",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
            heightfield=heightfield,
        )
        group: Node = Node(
            id="gen",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        nodes: dict[str, Node] = {hf.id: hf, group.id: group}
        context: BuildContext = BuildContext(
            nodes=nodes, group_id=group.id, heightmap_catalog=catalog
        )
        params: ProjectHeightfieldParams = ProjectHeightfieldParams(
            align_to_normal=BoolParam(True, label="Align to normal"),
        )
        out: list[GeneratedSlot] = project_heightfield_apply(
            [slot_at(0.0, 0.0, 0.0)], params, context
        )
        expected: Vector3 = sample_heightfield_normal(
            catalog, heightfield, 0.5, 0.5
        )
        matrix = quaternion_to_matrix(out[0].transform.rotation)
        up: Vector3 = Vector3(matrix.m4, matrix.m5, matrix.m6)
        self.assertGreater(vector3_dot_product(up, expected), 0.99)


if __name__ == "__main__":
    unittest.main()
