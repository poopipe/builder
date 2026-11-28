"""gpu vector sculpt stamp vs cpu reference on a small field"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyray import (
    ConfigFlags,
    Vector3,
    close_window,
    init_window,
    set_config_flags,
    set_target_fps,
)

from builder.heightfield.gpu.heightfield_gpu import (
    create_heightfield_gpu_state,
    unload_heightfield_gpu,
)
from builder.heightfield.heightfield_types import (
    VectorSculptHeightSource,
    default_heightfield,
)
from builder.heightfield.heightmap_catalog import HeightmapCatalog
from builder.heightfield.sample import sample_height
from builder.heightfield.vector.edit import append_vector_sculpt_edit_layer
from builder.heightfield.vector.rasterize import rasterize_vector_layer
from builder.scene.scene_types import Node, transform_at


def try_init_gl() -> bool:
    """open a hidden window for gpu tests; false when gl is unavailable"""
    try:
        set_config_flags(
            ConfigFlags.FLAG_WINDOW_HIDDEN | ConfigFlags.FLAG_WINDOW_UNDECORATED
        )
        init_window(64, 64, "gpu-sculpt-test")
        set_target_fps(30)
        return True
    except Exception:
        return False


class TestGpuSculptStamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gl_ok = try_init_gl()

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "gl_ok", False):
            close_window()

    def test_gpu_circle_add_matches_cpu(self) -> None:
        if not self.gl_ok:
            self.skipTest("opengl context unavailable")
        catalog_cpu: HeightmapCatalog = HeightmapCatalog()
        catalog_gpu: HeightmapCatalog = HeightmapCatalog()
        with tempfile.TemporaryDirectory() as tmp:
            heightfield = default_heightfield()
            hf_id = "hf1"
            updated_cpu, new_nodes = append_vector_sculpt_edit_layer(
                catalog_cpu,
                heightfield,
                hf_id,
                cache_dir=Path(tmp) / "cpu",
                default_shape_kind="circle",
            )
            nodes_cpu: dict[str, Node] = {
                hf_id: Node(
                    id=hf_id,
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated_cpu,
                )
            }
            for node in new_nodes:
                nodes_cpu[node.id] = node
            layer = next(n for n in new_nodes if n.vector_layer is not None)
            source_cpu = updated_cpu.edit_layers[0].height
            assert isinstance(source_cpu, VectorSculptHeightSource)
            rasterize_vector_layer(
                catalog_cpu,
                updated_cpu,
                nodes_cpu,
                layer.id,
                source_cpu.heightmap_id,
                sculpt_metres=True,
                layer_index=0,
            )
            mid_cpu = sample_height(catalog_cpu, updated_cpu, 0.5, 0.5)

            updated_gpu, new_nodes_gpu = append_vector_sculpt_edit_layer(
                catalog_gpu,
                heightfield,
                hf_id,
                cache_dir=Path(tmp) / "gpu",
                default_shape_kind="circle",
            )
            nodes_gpu: dict[str, Node] = {
                hf_id: Node(
                    id=hf_id,
                    parent_id=None,
                    transform=transform_at(Vector3(0.0, 0.0, 0.0)),
                    mesh_id=None,
                    heightfield=updated_gpu,
                )
            }
            for node in new_nodes_gpu:
                nodes_gpu[node.id] = node
            layer_gpu = next(
                n for n in new_nodes_gpu if n.vector_layer is not None
            )
            source_gpu = updated_gpu.edit_layers[0].height
            assert isinstance(source_gpu, VectorSculptHeightSource)
            gpu = create_heightfield_gpu_state()
            try:
                rasterize_vector_layer(
                    catalog_gpu,
                    updated_gpu,
                    nodes_gpu,
                    layer_gpu.id,
                    source_gpu.heightmap_id,
                    sculpt_metres=True,
                    layer_index=0,
                    heightfield_gpu=gpu,
                )
                mid_gpu = sample_height(catalog_gpu, updated_gpu, 0.5, 0.5)
            finally:
                unload_heightfield_gpu(gpu)
            self.assertAlmostEqual(mid_gpu, mid_cpu, places=2)
            self.assertNotAlmostEqual(mid_cpu, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
