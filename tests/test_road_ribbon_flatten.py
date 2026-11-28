"""road-sections ribbon flatten: surface_y (shadow height) + stroke densify"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pyray import Vector3

from builder.heightfield.heightfield_types import (
    HeightEditLayer,
    ProceduralHeightSource,
    ProceduralHeightType,
    RoadSectionsHeightSource,
    default_heightfield,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.paint import create_paint_heightmap
from builder.heightfield.road_sections.rasterize import (
    rasterize_road_sections_layer,
    sparse_centerline_xz,
    sparse_grade_knots,
)
from builder.heightfield.road_sections.road_sections_types import (
    RoadPolyline,
    RoadSectionsLayerRecipe,
    RoadShpRecipe,
    RoadUsageGroupRecipe,
)
from builder.heightfield.vector.overlay import surface_y
from builder.scene.ids import new_node_id
from builder.scene.scene_types import Node, transform_at


class TestSparseCenterline(unittest.TestCase):
    def test_densifies_on_coarse_spacing(self) -> None:
        # 100 m east with dense verts; 32 m step → ~4 knots
        points: list[tuple[float, float]] = [
            (float(i), 0.0) for i in range(0, 101)
        ]
        knots = sparse_centerline_xz(points, 32.0)
        self.assertGreaterEqual(len(knots), 4)
        self.assertLessEqual(len(knots), 6)
        self.assertAlmostEqual(knots[0][0], 0.0)
        self.assertAlmostEqual(knots[-1][0], 100.0)
        xs: list[float] = [k[0] for k in knots[1:-1]]
        self.assertTrue(
            all(abs(x - round(x / 32.0) * 32.0) < 1e-6 for x in xs)
        )

    def test_grade_knots_sample_surface(self) -> None:
        points: list[tuple[float, float]] = [(0.0, 0.0), (100.0, 0.0)]

        def surface(x: float, _z: float) -> float:
            return x * 0.1

        knots = sparse_grade_knots(points, 32.0, surface)
        self.assertAlmostEqual(knots[0][2], 0.0)
        self.assertAlmostEqual(knots[-1][2], 10.0)


class TestRibbonRasterize(unittest.TestCase):
    def test_flatten_writes_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            heightmap_id = create_paint_heightmap(
                catalog,
                label="rs",
                width=64,
                height=64,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="rs_cov",
                width=64,
                height=64,
                cache_dir=cache_dir,
            )
            layer_id: str = new_node_id()
            shp_id: str = new_node_id()
            usage_id: str = new_node_id()
            # dense verts across the field centerline
            noisy: list[tuple[float, float]] = [
                (float(i) - 48.0, 0.0) for i in range(0, 97)
            ]
            group = RoadUsageGroupRecipe(
                usage="primary",
                kit="asf1",
                polylines=(RoadPolyline(id="p0", points=tuple(noisy)),),
                ribbon_flatten=True,
            )
            heightfield = default_heightfield()
            updated = replace(
                heightfield,
                edit_layers=(
                    HeightEditLayer(
                        name="Road Sections",
                        height=RoadSectionsHeightSource(
                            layer_node_id=layer_id,
                            heightmap_id=heightmap_id,
                            coverage_id=coverage_id,
                        ),
                    ),
                ),
            )
            nodes: dict[str, Node] = {
                "hf1": Node(
                    id="hf1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                ),
                layer_id: Node(
                    id=layer_id,
                    parent_id="hf1",
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_sections_layer=RoadSectionsLayerRecipe(),
                    name="Road Sections",
                ),
                shp_id: Node(
                    id=shp_id,
                    parent_id=layer_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_shp=RoadShpRecipe(source_path="x.shp"),
                    name="roads",
                ),
                usage_id: Node(
                    id=usage_id,
                    parent_id=shp_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_usage_group=group,
                    name="primary",
                ),
            }
            ok = rasterize_road_sections_layer(
                catalog,
                updated,
                nodes,
                layer_id,
                heightmap_id,
                coverage_id,
                layer_index=0,
            )
            self.assertTrue(ok)
            cov = catalog.buffers[coverage_id]
            self.assertGreater(max(cov.samples), 0.5)

    def test_no_flatten_leaves_coverage_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            heightmap_id = create_paint_heightmap(
                catalog,
                label="rs",
                width=32,
                height=32,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="rs_cov",
                width=32,
                height=32,
                cache_dir=cache_dir,
            )
            layer_id: str = new_node_id()
            heightfield = default_heightfield()
            updated = replace(
                heightfield,
                edit_layers=(
                    HeightEditLayer(
                        name="Road Sections",
                        height=RoadSectionsHeightSource(
                            layer_node_id=layer_id,
                            heightmap_id=heightmap_id,
                            coverage_id=coverage_id,
                        ),
                    ),
                ),
            )
            nodes: dict[str, Node] = {
                "hf1": Node(
                    id="hf1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                ),
                layer_id: Node(
                    id=layer_id,
                    parent_id="hf1",
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_sections_layer=RoadSectionsLayerRecipe(),
                    name="Road Sections",
                ),
            }
            catalog.buffers[coverage_id].samples[0] = 1.0
            ok = rasterize_road_sections_layer(
                catalog,
                updated,
                nodes,
                layer_id,
                heightmap_id,
                coverage_id,
                layer_index=0,
            )
            self.assertTrue(ok)
            self.assertEqual(max(catalog.buffers[coverage_id].samples), 0.0)

    def test_flatten_absolute_matches_shadow_surface_y(self) -> None:
        """ribbon stores the same height shadow draping would sample"""
        with tempfile.TemporaryDirectory() as tmp:
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            heightmap_id = create_paint_heightmap(
                catalog,
                label="rs",
                width=64,
                height=64,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="rs_cov",
                width=64,
                height=64,
                cache_dir=cache_dir,
            )
            layer_id: str = new_node_id()
            shp_id: str = new_node_id()
            usage_id: str = new_node_id()
            underlay_y: float = 42.5
            group = RoadUsageGroupRecipe(
                usage="primary",
                kit="asf1",
                polylines=(
                    RoadPolyline(
                        id="p0",
                        points=((-40.0, 0.0), (40.0, 0.0)),
                    ),
                ),
                ribbon_flatten=True,
            )
            heightfield = default_heightfield()
            updated = replace(
                heightfield,
                edit_layers=(
                    HeightEditLayer(
                        name="Underlay",
                        height=ProceduralHeightSource(
                            ProceduralHeightType.constant, underlay_y
                        ),
                    ),
                    HeightEditLayer(
                        name="Road Sections",
                        height=RoadSectionsHeightSource(
                            layer_node_id=layer_id,
                            heightmap_id=heightmap_id,
                            coverage_id=coverage_id,
                        ),
                    ),
                ),
            )
            nodes: dict[str, Node] = {
                "hf1": Node(
                    id="hf1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                ),
                layer_id: Node(
                    id=layer_id,
                    parent_id="hf1",
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_sections_layer=RoadSectionsLayerRecipe(),
                    name="Road Sections",
                ),
                shp_id: Node(
                    id=shp_id,
                    parent_id=layer_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_shp=RoadShpRecipe(source_path="x.shp"),
                    name="roads",
                ),
                usage_id: Node(
                    id=usage_id,
                    parent_id=shp_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_usage_group=group,
                    name="primary",
                ),
            }
            ok = rasterize_road_sections_layer(
                catalog,
                updated,
                nodes,
                layer_id,
                heightmap_id,
                coverage_id,
                layer_index=1,
            )
            self.assertTrue(ok)
            layout = heightfield_layout(updated)
            expected: float = surface_y(
                catalog,
                updated,
                0.0,
                0.0,
                layout.size_x,
                layout.size_z,
                0.0,
                below_layer_index=1,
            )
            self.assertAlmostEqual(expected, underlay_y, places=4)
            cov = catalog.buffers[coverage_id].samples
            abs_h = catalog.buffers[heightmap_id].samples
            matched: int = 0
            index: int
            for index, coverage in enumerate(cov):
                if coverage < 0.9:
                    continue
                # buffer stores premultiplied height (h*cov)
                self.assertAlmostEqual(
                    abs_h[index] / coverage, expected, places=4
                )
                matched += 1
            self.assertGreater(matched, 10)

    def test_premult_ribbon_no_trench_on_upsample(self) -> None:
        """bilinear sample between road and empty exterior must not pull toward 0"""
        with tempfile.TemporaryDirectory() as tmp:
            catalog = HeightmapCatalog()
            cache_dir = Path(tmp) / "heightmaps"
            cache_dir.mkdir()
            heightmap_id = create_paint_heightmap(
                catalog,
                label="rs",
                width=32,
                height=32,
                cache_dir=cache_dir,
            )
            coverage_id = create_paint_heightmap(
                catalog,
                label="rs_cov",
                width=32,
                height=32,
                cache_dir=cache_dir,
            )
            layer_id: str = new_node_id()
            shp_id: str = new_node_id()
            usage_id: str = new_node_id()
            underlay_y: float = 100.0
            group = RoadUsageGroupRecipe(
                usage="primary",
                kit="asf1",
                polylines=(
                    RoadPolyline(
                        id="p0",
                        points=((-40.0, 0.0), (40.0, 0.0)),
                    ),
                ),
                ribbon_flatten=True,
            )
            heightfield = default_heightfield()
            updated = replace(
                heightfield,
                edit_layers=(
                    HeightEditLayer(
                        name="Underlay",
                        height=ProceduralHeightSource(
                            ProceduralHeightType.constant, underlay_y
                        ),
                    ),
                    HeightEditLayer(
                        name="Road Sections",
                        height=RoadSectionsHeightSource(
                            layer_node_id=layer_id,
                            heightmap_id=heightmap_id,
                            coverage_id=coverage_id,
                        ),
                    ),
                ),
            )
            nodes: dict[str, Node] = {
                "hf1": Node(
                    id="hf1",
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated,
                    name="Heightfield",
                ),
                layer_id: Node(
                    id=layer_id,
                    parent_id="hf1",
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_sections_layer=RoadSectionsLayerRecipe(),
                    name="Road Sections",
                ),
                shp_id: Node(
                    id=shp_id,
                    parent_id=layer_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_shp=RoadShpRecipe(source_path="x.shp"),
                    name="roads",
                ),
                usage_id: Node(
                    id=usage_id,
                    parent_id=shp_id,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    road_usage_group=group,
                    name="primary",
                ),
            }
            ok = rasterize_road_sections_layer(
                catalog,
                updated,
                nodes,
                layer_id,
                heightmap_id,
                coverage_id,
                layer_index=1,
            )
            self.assertTrue(ok)
            from builder.heightfield.sample import sample_height

            # dense uv scan across the centerline; no sample below underlay
            samples: list[float] = []
            i: int
            for i in range(64):
                u: float = float(i) / 63.0
                samples.append(sample_height(catalog, updated, u, 0.5))
            self.assertGreaterEqual(min(samples), underlay_y - 0.05)


if __name__ == "__main__":
    unittest.main()
