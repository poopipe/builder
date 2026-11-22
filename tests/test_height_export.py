"""composited height export to .asc / u16 .png"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from builder.heightfield.asc_height_decode import asc_decode_height_image
from builder.heightfield.height_export import (
    composite_height_buffer,
    export_composited_height,
    pack_u16_heights,
)
from builder.heightfield.height_image_decode import samples_from_f32
from builder.heightfield.height_image_types import HeightSampleType
from builder.heightfield.heightfield_types import (
    CatalogHeightSource,
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
    default_heightfield,
)
from builder.heightfield.heightmap_catalog import (
    HeightmapAsset,
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
    register_heightmap_asset,
)
from builder.heightfield.pillow_height_decode import pillow_decode_height_image


def register_f32_map(
    catalog: HeightmapCatalog,
    name: str,
    width: int,
    height: int,
    samples: list[float],
) -> HeightmapId:
    heightmap_id: HeightmapId = HeightmapId(name)
    register_heightmap_asset(
        catalog,
        HeightmapAsset(
            heightmap_id=heightmap_id,
            label=name,
            source_path=f"{name}.f32",
            cache_path=f"{name}.hraw",
            sample_type=HeightSampleType.f32,
            width=width,
            height=height,
        ),
    )
    catalog.buffers[heightmap_id] = heightmap_buffer_from_samples(
        width, height, samples
    )
    return heightmap_id


class TestCompositeExport(unittest.TestCase):
    def test_disabled_layer_skipped(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        base_id: HeightmapId = register_f32_map(
            catalog, "base", 2, 2, [10.0, 10.0, 10.0, 10.0]
        )
        bump_id: HeightmapId = register_f32_map(
            catalog, "bump", 2, 2, [5.0, 5.0, 5.0, 5.0]
        )
        heightfield: Heightfield = replace(
            default_heightfield(),
            base=base_id,
            edit_layers=(
                HeightEditLayer(
                    name="bump",
                    height=CatalogHeightSource(heightmap_id=bump_id),
                    enabled=False,
                    blend=HeightBlendMode.add,
                    amplitude=1.0,
                    offset=0.0,
                ),
            ),
        )
        buffer = composite_height_buffer(catalog, heightfield)
        self.assertEqual(buffer.samples, [10.0, 10.0, 10.0, 10.0])

    def test_enabled_add_layer(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        base_id: HeightmapId = register_f32_map(
            catalog, "base", 2, 2, [10.0, 10.0, 10.0, 10.0]
        )
        bump_id: HeightmapId = register_f32_map(
            catalog, "bump", 2, 2, [5.0, 5.0, 5.0, 5.0]
        )
        heightfield: Heightfield = replace(
            default_heightfield(),
            base=base_id,
            edit_layers=(
                HeightEditLayer(
                    name="bump",
                    height=CatalogHeightSource(heightmap_id=bump_id),
                    enabled=True,
                    blend=HeightBlendMode.add,
                    amplitude=1.0,
                    offset=0.0,
                ),
            ),
        )
        buffer = composite_height_buffer(catalog, heightfield)
        self.assertEqual(buffer.samples, [15.0, 15.0, 15.0, 15.0])

    def test_export_asc_north_first_and_meta(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        # south row [1,2], north row [3,4] in buffer order
        base_id: HeightmapId = register_f32_map(
            catalog, "base", 2, 2, [1.0, 2.0, 3.0, 4.0]
        )
        heightfield: Heightfield = replace(
            default_heightfield(),
            base=base_id,
            size_x=replace(default_heightfield().size_x, value=256.0),
            size_z=replace(default_heightfield().size_z, value=256.0),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            asc_path: Path = Path(temp_dir) / "out.asc"
            export_composited_height(catalog, heightfield, asc_path)
            self.assertTrue(asc_path.is_file())
            meta_path: Path = Path(temp_dir) / "out_meta.json"
            self.assertTrue(meta_path.is_file())
            meta: object = json.loads(meta_path.read_text(encoding="utf-8"))
            assert isinstance(meta, dict)
            elevation: object = meta["elevation_m"]
            assert isinstance(elevation, dict)
            self.assertEqual(elevation["min"], 1.0)
            self.assertEqual(elevation["max"], 4.0)
            # re-decode flips north→south, so buffer matches original
            decoded = asc_decode_height_image(asc_path)
            samples: list[float] = samples_from_f32(decoded.data)
            self.assertEqual(samples, [1.0, 2.0, 3.0, 4.0])

    def test_export_png_u16_and_meta(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        base_id: HeightmapId = register_f32_map(
            catalog, "base", 2, 2, [0.0, 50.0, 100.0, 100.0]
        )
        heightfield: Heightfield = replace(default_heightfield(), base=base_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path: Path = Path(temp_dir) / "out.png"
            export_composited_height(catalog, heightfield, png_path)
            self.assertTrue(png_path.is_file())
            self.assertTrue((Path(temp_dir) / "out_meta.json").is_file())
            decoded = pillow_decode_height_image(png_path)
            self.assertEqual(decoded.sample_type, HeightSampleType.u16)
            # north-first on disk: max z row first → [100,100] then [0,50]
            # after pillow load row0 is file row0
            from builder.heightfield.height_image_decode import samples_from_u16

            norms: list[float] = samples_from_u16(decoded.data)
            self.assertAlmostEqual(norms[0], 1.0, places=4)
            self.assertAlmostEqual(norms[1], 1.0, places=4)
            self.assertAlmostEqual(norms[2], 0.0, places=4)
            self.assertAlmostEqual(norms[3], 0.5, places=4)

    def test_pack_u16_endpoints(self) -> None:
        packed: list[int] = pack_u16_heights([10.0, 20.0], 10.0, 20.0)
        self.assertEqual(packed, [0, 65535])


if __name__ == "__main__":
    unittest.main()
