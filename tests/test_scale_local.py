"""scale local modifier"""

from __future__ import annotations

import unittest

from pyray import Transform, Vector3, quaternion_identity

from builder.generators.generator_types import BuildContext, GeneratedSlot
from builder.generators.param_types import Float3Param
from builder.modifiers.scale_local import ScaleLocalParams, scale_local_apply
from builder.scene.scene_types import transform_at


class TestScaleLocal(unittest.TestCase):
    def test_multiplies_each_axis(self) -> None:
        slot = GeneratedSlot(
            Transform(
                Vector3(1.0, 2.0, 3.0),
                quaternion_identity(),
                Vector3(2.0, 3.0, 4.0),
            )
        )
        params = ScaleLocalParams(
            scale=Float3Param(x=0.5, y=2.0, z=1.5, label="Scale")
        )
        context = BuildContext(nodes={}, group_id="g")
        out = scale_local_apply([slot], params, context)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].transform.scale.x, 1.0)
        self.assertAlmostEqual(out[0].transform.scale.y, 6.0)
        self.assertAlmostEqual(out[0].transform.scale.z, 6.0)
        self.assertAlmostEqual(out[0].transform.translation.x, 1.0)

    def test_default_is_identity(self) -> None:
        slot = GeneratedSlot(transform_at(Vector3(0.0, 0.0, 0.0)))
        out = scale_local_apply([slot], ScaleLocalParams(), BuildContext({}, "g"))
        self.assertAlmostEqual(out[0].transform.scale.x, 1.0)
        self.assertAlmostEqual(out[0].transform.scale.y, 1.0)
        self.assertAlmostEqual(out[0].transform.scale.z, 1.0)


if __name__ == "__main__":
    unittest.main()
