"""draw-cache rebuild must stay near-linear for large generator regenerates"""

from __future__ import annotations

import time
import unittest

from pyray import Vector3

from builder.scene.scene import Scene
from builder.scene.scene_types import MeshId, Node, transform_at
from builder.view.draw_cache import DrawCache, sync_draw_cache


class TestDrawCacheScale(unittest.TestCase):
    def test_incremental_add_many_siblings(self) -> None:
        scene = Scene()
        parent = Node(
            id="p",
            parent_id=None,
            transform=transform_at(Vector3(0.0, 0.0, 0.0)),
            mesh_id=None,
        )
        scene.add_nodes([parent])
        cache = DrawCache()
        sync_draw_cache(scene, cache)

        mesh = MeshId("cube")
        children = [
            Node(
                id=f"c{i}",
                parent_id="p",
                transform=transform_at(Vector3(float(i), 0.0, 0.0)),
                mesh_id=mesh,
            )
            for i in range(20000)
        ]
        t0 = time.perf_counter()
        scene.add_nodes(children)
        sync_draw_cache(scene, cache)
        elapsed = time.perf_counter() - t0
        print(f"\nincremental_20k={elapsed:.3f}s batches={len(cache.batches)}")
        self.assertEqual(len(cache.batches), 1)
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
