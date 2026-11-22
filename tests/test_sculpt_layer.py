"""sculpt correction layer sample and stamp"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from builder.generators.param_types import FloatParam
from builder.heightfield.heightfield_types import (
    HeightBlendMode,
    HeightEditLayer,
    Heightfield,
    ProceduralHeightSource,
    ProceduralHeightType,
    SculptHeightMode,
    SculptHeightSource,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height, sample_height_below
from builder.heightfield.sculpt import (
    append_sculpt_edit_layer,
    set_sculpt_layer_mode,
    stamp_sculpt_layer,
)


def stamp(
    catalog: HeightmapCatalog,
    heightfield: Heightfield,
    layer_index: int,
    brush_kind: str,
    center_u: float,
    center_v: float,
    radius_u: float,
    radius_v: float,
    intensity: float,
    flatten_target: float | None,
    cache_dir: Path,
) -> tuple[bool, Heightfield]:
    """stamp helper that threads cache_dir and returns updated heightfield"""
    return stamp_sculpt_layer(
        catalog,
        heightfield,
        layer_index,
        brush_kind,
        center_u,
        center_v,
        radius_u,
        radius_v,
        intensity,
        flatten_target,
        cache_dir=cache_dir,
    )


class TestSculptLayerComposite(unittest.TestCase):
    def test_delta_adds_metres_over_below(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield: Heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(10.0, label="Height Min"),
                base_height_max=FloatParam(10.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.delta,
            )
            below: float = sample_height_below(catalog, heightfield, 0, 0.5, 0.5)
            self.assertAlmostEqual(below, 10.0)
            stamped: bool
            stamped, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "add",
                0.5,
                0.5,
                0.5,
                0.5,
                3.0,
                None,
                Path(tmp),
            )
            self.assertTrue(stamped)
            at_center: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertGreater(at_center, 12.0)

    def test_delta_flatten_uses_below_target(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield: Heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(5.0, label="Height Min"),
                base_height_max=FloatParam(5.0, label="Height Max"),
                edit_layers=(
                    HeightEditLayer(
                        name="bump",
                        height=ProceduralHeightSource(
                            ProceduralHeightType.constant, 4.0
                        ),
                        blend=HeightBlendMode.add,
                        amplitude=1.0,
                        offset=0.0,
                    ),
                ),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.delta,
            )
            # stack below sculpt = 5 + 4 = 9
            below = sample_height_below(catalog, heightfield, 1, 0.5, 0.5)
            self.assertAlmostEqual(below, 9.0)
            target: float = 9.0
            stamped, heightfield = stamp(
                catalog,
                heightfield,
                1,
                "flatten",
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                target,
                Path(tmp),
            )
            self.assertTrue(stamped)
            match heightfield.edit_layers[1].height:
                case SculptHeightSource(mode=mode):
                    self.assertIs(mode, SculptHeightMode.absolute)
                case _:
                    self.fail("expected sculpt source")
            # absolute flatten to below keeps visible height ~9
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), 9.0, places=2
            )

    def test_absolute_pass_through_when_coverage_zero(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(7.0, label="Height Min"),
                base_height_max=FloatParam(7.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.absolute,
            )
            match heightfield.edit_layers[0].height:
                case SculptHeightSource(coverage_id=coverage_id):
                    self.assertIsNotNone(coverage_id)
                case _:
                    self.fail("expected sculpt source")
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), 7.0
            )

    def test_mode_switch_preserves_composited_height(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir: Path = Path(tmp)
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(10.0, label="Height Min"),
                base_height_max=FloatParam(10.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=cache_dir,
                mode=SculptHeightMode.delta,
            )
            ok: bool
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "add",
                0.5,
                0.5,
                0.4,
                0.4,
                2.5,
                None,
                cache_dir,
            )
            self.assertTrue(ok)
            before: float = sample_height(catalog, heightfield, 0.5, 0.5)
            heightfield = set_sculpt_layer_mode(
                catalog,
                heightfield,
                0,
                SculptHeightMode.absolute,
                cache_dir=cache_dir,
            )
            match heightfield.edit_layers[0].height:
                case SculptHeightSource(mode=mode, coverage_id=cov):
                    self.assertEqual(mode, SculptHeightMode.absolute)
                    self.assertIsNotNone(cov)
                case _:
                    self.fail("expected absolute sculpt source")
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), before, places=4
            )
            match heightfield.edit_layers[0].height:
                case SculptHeightSource(coverage_id=cov_id):
                    assert cov_id is not None
                    cov_buf = catalog.buffers[cov_id]
                    # corner far from stamp should stay uncovered
                    self.assertAlmostEqual(cov_buf.samples[0], 0.0)
                    # center of stamp should be covered
                    mid: int = (cov_buf.height // 2) * cov_buf.width + (
                        cov_buf.width // 2
                    )
                    self.assertGreater(cov_buf.samples[mid], 0.5)
                case _:
                    self.fail("expected absolute sculpt source")
            heightfield = set_sculpt_layer_mode(
                catalog,
                heightfield,
                0,
                SculptHeightMode.delta,
                cache_dir=cache_dir,
            )
            match heightfield.edit_layers[0].height:
                case SculptHeightSource(mode=mode2, coverage_id=cov2):
                    self.assertEqual(mode2, SculptHeightMode.delta)
                    self.assertIsNone(cov2)
                case _:
                    self.fail("expected delta sculpt source")
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5), before, places=4
            )

    def test_delta_flatten_up_only_raises(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(5.0, label="Height Min"),
                base_height_max=FloatParam(5.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.delta,
            )
            # dip below the flatten target
            ok: bool
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "subtract",
                0.5,
                0.5,
                0.3,
                0.3,
                2.0,
                None,
                Path(tmp),
            )
            self.assertTrue(ok)
            dipped: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertLess(dipped, 5.0)
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "flatten_up",
                0.5,
                0.5,
                0.4,
                0.4,
                1.0,
                5.0,
                Path(tmp),
            )
            self.assertTrue(ok)
            raised: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertGreater(raised, dipped)
            self.assertLessEqual(raised, 5.0 + 1e-4)

    def test_delta_smooth_lowers_add_bump(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(0.0, label="Height Min"),
                base_height_max=FloatParam(0.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.delta,
            )
            ok: bool
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "add",
                0.5,
                0.5,
                0.15,
                0.15,
                5.0,
                None,
                Path(tmp),
            )
            self.assertTrue(ok)
            raised: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertGreater(raised, 2.0)
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "smooth",
                0.5,
                0.5,
                0.4,
                0.4,
                1.0,
                None,
                Path(tmp),
            )
            self.assertTrue(ok)
            smoothed: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertLess(smoothed, raised)

    def test_flatten_after_mode_switch_stays_on_plane(self) -> None:
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir: Path = Path(tmp)
            heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(10.0, label="Height Min"),
                base_height_max=FloatParam(10.0, label="Height Max"),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=cache_dir,
                mode=SculptHeightMode.delta,
            )
            ok: bool
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "add",
                0.5,
                0.5,
                0.35,
                0.35,
                4.0,
                None,
                cache_dir,
            )
            self.assertTrue(ok)
            raised: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertGreater(raised, 11.0)
            # flatten to the visible surface height at press (raised)
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "flatten",
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                raised,
                cache_dir,
            )
            self.assertTrue(ok)
            flat: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertAlmostEqual(flat, raised, places=2)
            match heightfield.edit_layers[0].height:
                case SculptHeightSource(mode=flat_mode):
                    self.assertIs(flat_mode, SculptHeightMode.absolute)
                case _:
                    self.fail("flatten should promote to absolute")
            after_convert: float = sample_height(catalog, heightfield, 0.5, 0.5)
            self.assertAlmostEqual(after_convert, flat, places=4)
            ok, heightfield = stamp(
                catalog,
                heightfield,
                0,
                "flatten",
                0.5,
                0.5,
                0.5,
                0.5,
                1.0,
                after_convert,
                cache_dir,
            )
            self.assertTrue(ok)
            self.assertAlmostEqual(
                sample_height(catalog, heightfield, 0.5, 0.5),
                after_convert,
                places=3,
            )

    def test_delta_flatten_plateau_hits_target_mid_radius(self) -> None:
        """flatten interior (incl. 0.5x radius) reaches target; outer rim stays soft"""
        catalog: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield: Heightfield = Heightfield(
                base=None,
                base_height_min=FloatParam(0.0, label="Height Min"),
                base_height_max=FloatParam(0.0, label="Height Max"),
                edit_layers=(
                    HeightEditLayer(
                        name="bump",
                        height=ProceduralHeightSource(
                            ProceduralHeightType.constant, 10.0
                        ),
                        blend=HeightBlendMode.add,
                        amplitude=1.0,
                        offset=0.0,
                    ),
                ),
            )
            heightfield = append_sculpt_edit_layer(
                catalog,
                heightfield,
                cache_dir=Path(tmp),
                mode=SculptHeightMode.delta,
            )
            radius_u: float = 0.25
            radius_v: float = 0.25
            target: float = 0.0
            stamped: bool
            stamped, heightfield = stamp(
                catalog,
                heightfield,
                1,
                "flatten",
                0.5,
                0.5,
                radius_u,
                radius_v,
                1.0,
                target,
                Path(tmp),
            )
            self.assertTrue(stamped)
            match heightfield.edit_layers[1].height:
                case SculptHeightSource(mode=promoted):
                    self.assertIs(promoted, SculptHeightMode.absolute)
                case _:
                    self.fail("expected absolute after flatten")
            center: float = sample_height(catalog, heightfield, 0.5, 0.5)
            mid_u: float = 0.5 + 0.5 * radius_u
            mid: float = sample_height(catalog, heightfield, mid_u, 0.5)
            rim_u: float = 0.5 + 0.92 * radius_u
            rim: float = sample_height(catalog, heightfield, rim_u, 0.5)
            # absolute replace: mid-radius matches center at the flatten target
            self.assertAlmostEqual(center, target, places=2)
            self.assertAlmostEqual(mid, target, places=2)
            # outer rim still shows the below bump (not fully replaced)
            self.assertGreater(rim, 5.0)


if __name__ == "__main__":
    unittest.main()
