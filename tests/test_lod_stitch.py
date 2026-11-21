"""pure-index checks for heightfield lod edge stitching"""

from __future__ import annotations

import unittest

from builder.heightfield.gpu.grid_mesh import (
    build_stitched_indices,
    vertex_active_on_edges,
)
from builder.heightfield.gpu.lod import (
    edge_east,
    edge_north,
    edge_south,
    edge_west,
    heightfield_lod_max,
    lod_tile_verts,
)


def decode_xz(tile_verts: int, index: int) -> tuple[int, int]:
    return (index % tile_verts, index // tile_verts)


def edge_segments(
    indices: list[int], tile_verts: int, which: str
) -> list[tuple[int, int]]:
    """unique undirected boundary edges on the named tile side"""
    step: int = tile_verts - 1
    segs: set[tuple[int, int]] = set()
    i: int
    for i in range(0, len(indices), 3):
        tri: tuple[int, int, int] = (
            indices[i],
            indices[i + 1],
            indices[i + 2],
        )
        pair: tuple[int, int]
        for pair in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ax: int
            az: int
            bx: int
            bz: int
            ax, az = decode_xz(tile_verts, pair[0])
            bx, bz = decode_xz(tile_verts, pair[1])
            on_edge: bool = False
            if which == "north" and az == step and bz == step:
                on_edge = True
            elif which == "south" and az == 0 and bz == 0:
                on_edge = True
            elif which == "west" and ax == 0 and bx == 0:
                on_edge = True
            elif which == "east" and ax == step and bx == step:
                on_edge = True
            if not on_edge:
                continue
            lo: int
            hi: int
            lo, hi = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
            segs.add((lo, hi))
    return sorted(segs)


class TestLodStitch(unittest.TestCase):
    def test_stitched_indices_skip_inactive_edge_verts(self) -> None:
        lod: int
        edge_mask: int
        for lod in range(heightfield_lod_max + 1):
            tile_verts: int = lod_tile_verts(lod)
            step: int = tile_verts - 1
            for edge_mask in range(16):
                indices: list[int] = build_stitched_indices(
                    tile_verts, edge_mask
                )
                self.assertTrue(indices)
                self.assertEqual(len(indices) % 3, 0)
                index: int
                for index in indices:
                    x: int
                    z: int
                    x, z = decode_xz(tile_verts, index)
                    self.assertTrue(
                        vertex_active_on_edges(x, z, step, edge_mask),
                        f"inactive vert ({x},{z}) mask={edge_mask} lod={lod}",
                    )

    def test_collapsed_boundary_spans_two_steps(self) -> None:
        sides: list[tuple[int, str]] = [
            (edge_north, "north"),
            (edge_south, "south"),
            (edge_west, "west"),
            (edge_east, "east"),
        ]
        lod: int
        for lod in range(heightfield_lod_max + 1):
            tile_verts: int = lod_tile_verts(lod)
            step: int = tile_verts - 1
            if step < 2:
                continue
            bit: int
            which: str
            for bit, which in sides:
                other: int
                for other in (0, 15 & ~bit):
                    edge_mask: int = bit | other
                    indices: list[int] = build_stitched_indices(
                        tile_verts, edge_mask
                    )
                    segs: list[tuple[int, int]] = edge_segments(
                        indices, tile_verts, which
                    )
                    self.assertTrue(
                        segs, f"no {which} edge for mask={edge_mask}"
                    )
                    lo: int
                    hi: int
                    for lo, hi in segs:
                        ax: int
                        az: int
                        bx: int
                        bz: int
                        ax, az = decode_xz(tile_verts, lo)
                        bx, bz = decode_xz(tile_verts, hi)
                        if which in ("north", "south"):
                            self.assertEqual(az, bz)
                            self.assertEqual(
                                abs(ax - bx),
                                2,
                                f"{which} span {abs(ax - bx)} mask={edge_mask}",
                            )
                        else:
                            self.assertEqual(ax, bx)
                            self.assertEqual(
                                abs(az - bz),
                                2,
                                f"{which} span {abs(az - bz)} mask={edge_mask}",
                            )


if __name__ == "__main__":
    unittest.main()
