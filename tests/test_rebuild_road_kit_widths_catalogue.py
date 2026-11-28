"""live catalogue build coverage when dayz parts are on P:"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from builder.heightfield.road_sections.part_catalog import default_parts_roots
from tools.build_road_kit_widths import build_catalogue

_PARTS: Path = Path(r"P:\DZ\structures\roads\parts")


@unittest.skipUnless(_PARTS.is_dir(), "dayz road parts not on P:")
class TestBuildRoadKitWidthsCatalogue(unittest.TestCase):
    def test_catalogue_includes_snap_points(self) -> None:
        catalogue: dict[str, Any] = build_catalogue(default_parts_roots)
        pieces: list[dict[str, Any]] = catalogue["pieces"]
        with_snaps: list[dict[str, Any]] = [
            p for p in pieces if p.get("snap_points")
        ]
        self.assertGreater(len(with_snaps), 10)
        asf1_25: dict[str, Any] | None = next(
            (p for p in pieces if p.get("stem") == "asf1_25"), None
        )
        self.assertIsNotNone(asf1_25)
        assert asf1_25 is not None
        snaps: dict[str, Any] | None = asf1_25.get("snap_points")
        self.assertIsNotNone(snaps)
        assert snaps is not None
        self.assertAlmostEqual(float(snaps["lb"][0]), -6.0, places=3)
        self.assertAlmostEqual(float(snaps["lb"][2]), -12.5, places=3)


if __name__ == "__main__":
    unittest.main()
