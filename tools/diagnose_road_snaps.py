"""diagnose Memory lb/le/pb/pe recovery for sample road p3ds

usage (repo root, venv)::

    python -m tools.diagnose_road_snaps
"""

from __future__ import annotations

from pathlib import Path

from tools.build_road_kit_widths import (
    memory_lod_vertex_table_blob,
    parse_snap_selection_indices,
    recover_snap_points_from_indices,
)

_SAMPLES: tuple[tuple[str, float], ...] = (
    (r"P:\DZ\structures\roads\parts\asf1_22 50.p3d", 12.0),
    (r"P:\DZ\structures\roads\parts\asf3_22 50.p3d", 7.0),
    (r"P:\DZ\structures\roads\parts\asf3_15 75.p3d", 7.0),
    (r"P:\DZ\structures\roads\parts\asf3_7 100.p3d", 7.0),
    (r"P:\DZ\structures\roads\parts\asf3_30 25.p3d", 7.0),
    (r"P:\DZ\structures\roads\parts\asf3_6konec.p3d", 7.0),
)


def main() -> int:
    path_s: str
    width: float
    for path_s, width in _SAMPLES:
        path: Path = Path(path_s)
        print(f"--- {path.name}  width_hint={width}")
        if not path.is_file():
            print("  missing file")
            continue
        data: bytes = path.read_bytes()
        indices = parse_snap_selection_indices(data)
        print(f"  indices: {indices}")
        if indices is None:
            print("  snaps: (no selection block)")
            continue
        blob = memory_lod_vertex_table_blob(data)
        print(
            f"  memory VT blob: "
            f"{'none' if blob is None else f'{len(blob)} bytes'}"
        )
        snaps = recover_snap_points_from_indices(
            data, indices, width_m=width
        )
        print(f"  snaps: {snaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
