"""null implausible snap_points in road_kit_widths.json (no p3d rescan)

usage (from repo root, venv active)::

    python -m tools.scrub_road_kit_snap_points
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tools.build_road_kit_widths import snap_mapping_plausible


def main() -> int:
    root: Path = Path(__file__).resolve().parents[1]
    path: Path = (
        root
        / "src"
        / "builder"
        / "heightfield"
        / "road_sections"
        / "road_kit_widths.json"
    )
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    pieces: Any = raw.get("pieces")
    if not isinstance(pieces, list):
        print("no pieces list", file=sys.stderr)
        return 1
    cleared: int = 0
    entry: Any
    for entry in pieces:
        if not isinstance(entry, dict):
            continue
        snaps: Any = entry.get("snap_points")
        if not isinstance(snaps, dict):
            continue
        mapped: dict[str, tuple[float, float, float]] = {}
        name: str
        ok: bool = True
        for name in ("lb", "le", "pb", "pe"):
            pt: Any = snaps.get(name)
            if not isinstance(pt, list) or len(pt) != 3:
                ok = False
                break
            mapped[name] = (float(pt[0]), float(pt[1]), float(pt[2]))
        if not ok or not snap_mapping_plausible(mapped):
            entry["snap_points"] = None
            cleared += 1
            stem: Any = entry.get("stem")
            print(f"cleared {stem}")
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print(f"cleared {cleared} snap set(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
