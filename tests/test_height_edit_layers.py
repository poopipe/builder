"""height edit layer blend and scene encode"""

from __future__ import annotations

import unittest

from builder.generators.param_types import FloatParam
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
    ProceduralHeightSource,
    ProceduralHeightType,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapBuffer,
    HeightmapCatalog,
    HeightmapId,
)
from builder.heightfield.sample import blend_height, sample_buffer, sample_height
from builder.io.scene_format import heightfield_from_json, heightfield_to_json


class TestBlendHeight(unittest.TestCase):
    def test_add_subtract_multiply_max_min(self) -> None:
        self.assertAlmostEqual(
            blend_height(10.0, 3.0, 1.0, HeightBlendMode.add), 13.0
        )
        self.assertAlmostEqual(
            blend_height(10.0, 3.0, 0.5, HeightBlendMode.add), 11.5
        )
        self.assertAlmostEqual(
            blend_height(10.0, 3.0, 1.0, HeightBlendMode.subtract), 7.0
        )
        self.assertAlmostEqual(
            blend_height(10.0, 2.0, 1.0, HeightBlendMode.multiply), 20.0
        )
        self.assertAlmostEqual(
            blend_height(10.0, 2.0, 0.0, HeightBlendMode.multiply), 10.0
        )
        self.assertAlmostEqual(
            blend_height(4.0, 9.0, 1.0, HeightBlendMode.max), 9.0
        )
        self.assertAlmostEqual(
            blend_height(4.0, 9.0, 1.0, HeightBlendMode.min), 4.0
        )


class TestSampleEditStack(unittest.TestCase):
    def test_procedural_add_layer(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightfield: Heightfield = Heightfield(
            base=None,
            base_amplitude=FloatParam(1.0, label="Base Amplitude"),
            base_offset=FloatParam(2.0, label="Base Offset"),
            edit_layers=(
                HeightEditLayer(
                    name="bump",
                    height=ProceduralHeightSource(
                        ProceduralHeightType.constant, 5.0
                    ),
                    blend=HeightBlendMode.add,
                    opacity=FloatParam(1.0, label="Opacity"),
                    amplitude=1.0,
                    offset=0.0,
                ),
            ),
        )
        self.assertAlmostEqual(sample_height(catalog, heightfield, 0.5, 0.5), 7.0)

    def test_uv_scale_wraps_edit_layer(self) -> None:
        """uv_scale_u=2 tiles; field u=0.5 maps to layer u=1 → wraps to left texel"""
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightmap_id: HeightmapId = HeightmapId("stripe")
        # 2x1: left=1, right=0
        catalog.buffers[heightmap_id] = HeightmapBuffer(
            width=2,
            height=1,
            samples=(1.0, 0.0),
            sample_min=0.0,
            sample_max=1.0,
        )
        heightfield: Heightfield = Heightfield(
            base=None,
            base_amplitude=FloatParam(1.0, label="Base Amplitude"),
            base_offset=FloatParam(0.0, label="Base Offset"),
            edit_layers=(
                HeightEditLayer(
                    name="stripe",
                    height=CatalogHeightSource(heightmap_id),
                    blend=HeightBlendMode.add,
                    amplitude=1.0,
                    offset=0.0,
                    uv_scale_u=2.0,
                    uv_scale_v=1.0,
                ),
            ),
        )
        # field 0 and 0.5 → layer u 0 and 1; wrap makes both hit the left texel (1)
        at_start: float = sample_height(catalog, heightfield, 0.0, 0.0)
        at_half: float = sample_height(catalog, heightfield, 0.5, 0.0)
        self.assertAlmostEqual(at_start, 1.0)
        self.assertAlmostEqual(at_half, 1.0)
        # same phase half a tile later
        at_quarter: float = sample_height(catalog, heightfield, 0.25, 0.0)
        at_three_quarter: float = sample_height(catalog, heightfield, 0.75, 0.0)
        self.assertAlmostEqual(at_quarter, at_three_quarter)
        # clamp would stick at the right edge (0) for layer u=1
        clamped_at_one: float = sample_buffer(
            catalog.buffers[heightmap_id], 1.0, 0.0, wrap=False
        )
        self.assertAlmostEqual(clamped_at_one, 0.0)
        self.assertNotAlmostEqual(at_half, clamped_at_one)


class TestHeightfieldSceneJson(unittest.TestCase):
    def test_round_trip(self) -> None:
        heightfield: Heightfield = Heightfield(
            base=HeightmapId("terrain"),
            edit_layers=(
                HeightEditLayer(
                    name="detail",
                    height=CatalogHeightSource(HeightmapId("noise")),
                    blend=HeightBlendMode.multiply,
                    mask=ProceduralHeightSource(
                        ProceduralHeightType.constant, 1.0
                    ),
                    amplitude=0.5,
                    offset=-1.0,
                    uv_scale_u=2.0,
                    uv_scale_v=0.5,
                    uv_offset_u=0.1,
                    uv_offset_v=-0.25,
                ),
            ),
        )
        encoded = heightfield_to_json(heightfield)
        decoded: Heightfield = heightfield_from_json(encoded)
        self.assertEqual(decoded.base, HeightmapId("terrain"))
        self.assertEqual(len(decoded.edit_layers), 1)
        layer: HeightEditLayer = decoded.edit_layers[0]
        self.assertEqual(layer.name, "detail")
        self.assertEqual(layer.blend, HeightBlendMode.multiply)
        self.assertAlmostEqual(layer.amplitude, 0.5)
        self.assertAlmostEqual(layer.offset, -1.0)
        self.assertAlmostEqual(layer.uv_scale_u, 2.0)
        self.assertAlmostEqual(layer.uv_scale_v, 0.5)
        self.assertAlmostEqual(layer.uv_offset_u, 0.1)
        self.assertAlmostEqual(layer.uv_offset_v, -0.25)
        self.assertIn("uv_scale_u", encoded["edit_layers"][0])


if __name__ == "__main__":
    unittest.main()
