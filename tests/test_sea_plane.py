"""sea plane tile visibility / cull rule"""

from __future__ import annotations

import unittest
from dataclasses import replace

from builder.generators.param_types import FloatParam
from builder.heightfield.heightfield_types import (
    Heightfield,
    default_heightfield,
    heightfield_layout,
)
from builder.heightfield.heightmap_catalog import (
    HeightSampleType,
    HeightmapAsset,
    HeightmapCatalog,
    HeightmapId,
    heightmap_buffer_from_samples,
    register_heightmap_asset,
)
from builder.heightfield.sea_plane import (
    build_sea_visibility,
    refresh_sea_visibility,
    sea_level_y,
    sea_tile_visible,
    tile_index_range_for_aabb,
)


def flat_heightfield(height_min: float) -> Heightfield:
    return replace(
        default_heightfield(),
        base=None,
        base_height_min=FloatParam(height_min, label="Height Min"),
        base_height_max=FloatParam(height_min, label="Height Max"),
        size_x=FloatParam(256.0, label="Size X"),
        size_z=FloatParam(256.0, label="Size Z"),
    )


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


class TestSeaTileVisible(unittest.TestCase):
    def test_below_sea_level_is_visible(self) -> None:
        self.assertTrue(sea_tile_visible(sea_level_y - 1.0))

    def test_at_sea_level_is_hidden(self) -> None:
        self.assertFalse(sea_tile_visible(sea_level_y))

    def test_above_sea_level_is_hidden(self) -> None:
        self.assertFalse(sea_tile_visible(sea_level_y + 5.0))


class TestBuildSeaVisibility(unittest.TestCase):
    def test_all_above_hides_every_tile(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightfield: Heightfield = flat_heightfield(10.0)
        layout = heightfield_layout(heightfield)
        visible: list[bool] = build_sea_visibility(catalog, heightfield, layout)
        self.assertEqual(len(visible), layout.tiles_x * layout.tiles_z)
        self.assertTrue(all(not flag for flag in visible))

    def test_all_below_shows_every_tile(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightfield: Heightfield = flat_heightfield(-2.0)
        layout = heightfield_layout(heightfield)
        visible: list[bool] = build_sea_visibility(catalog, heightfield, layout)
        self.assertEqual(len(visible), layout.tiles_x * layout.tiles_z)
        self.assertTrue(all(visible))

    def test_mixed_tile_stays_visible(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        # 2x2 field: three land samples, one underwater — min < 0 → show
        base_id: HeightmapId = register_f32_map(
            catalog, "mixed_sea", 2, 2, [5.0, 5.0, 5.0, -3.0]
        )
        heightfield: Heightfield = replace(
            flat_heightfield(0.0),
            base=base_id,
            base_height_min=FloatParam(0.0, label="Height Min"),
            base_height_max=FloatParam(1.0, label="Height Max"),
        )
        layout = heightfield_layout(heightfield)
        visible: list[bool] = build_sea_visibility(catalog, heightfield, layout)
        self.assertEqual(len(visible), layout.tiles_x * layout.tiles_z)
        self.assertTrue(any(visible))


class TestRefreshSeaVisibility(unittest.TestCase):
    def test_tile_index_range_covers_corner(self) -> None:
        heightfield: Heightfield = flat_heightfield(10.0)
        layout = heightfield_layout(heightfield)
        self.assertGreaterEqual(layout.tiles_x, 2)
        self.assertGreaterEqual(layout.tiles_z, 2)
        origin_x: float = -0.5 * layout.size_x
        origin_z: float = -0.5 * layout.size_z
        tile_world: float = float(layout.tile_verts - 1) * layout.spacing
        # only the +x / +z corner tile
        min_x: float = origin_x + tile_world * float(layout.tiles_x - 1) + 1.0
        max_x: float = origin_x + tile_world * float(layout.tiles_x) - 1.0
        min_z: float = origin_z + tile_world * float(layout.tiles_z - 1) + 1.0
        max_z: float = origin_z + tile_world * float(layout.tiles_z) - 1.0
        tile_range = tile_index_range_for_aabb(
            layout, min_x, max_x, min_z, max_z
        )
        self.assertIsNotNone(tile_range)
        assert tile_range is not None
        self.assertEqual(
            tile_range,
            (
                layout.tiles_x - 1,
                layout.tiles_x - 1,
                layout.tiles_z - 1,
                layout.tiles_z - 1,
            ),
        )

    def test_dirty_refresh_preserves_untouched_flags(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        heightfield: Heightfield = flat_heightfield(10.0)
        layout = heightfield_layout(heightfield)
        visible: list[bool] = build_sea_visibility(catalog, heightfield, layout)
        self.assertTrue(all(not flag for flag in visible))
        # corrupt one flag, refresh a distant empty dirty band that misses it
        visible[0] = True
        origin_x: float = -0.5 * layout.size_x
        origin_z: float = -0.5 * layout.size_z
        tile_world: float = float(layout.tile_verts - 1) * layout.spacing
        dirty = (
            origin_x + tile_world * float(layout.tiles_x - 1) + 1.0,
            origin_x + tile_world * float(layout.tiles_x) - 1.0,
            origin_z + tile_world * float(layout.tiles_z - 1) + 1.0,
            origin_z + tile_world * float(layout.tiles_z) - 1.0,
        )
        refreshed: list[bool] = refresh_sea_visibility(
            catalog, heightfield, layout, visible, dirty
        )
        self.assertTrue(refreshed[0])
        corner: int = (layout.tiles_z - 1) * layout.tiles_x + (
            layout.tiles_x - 1
        )
        self.assertFalse(refreshed[corner])


if __name__ == "__main__":
    unittest.main()
