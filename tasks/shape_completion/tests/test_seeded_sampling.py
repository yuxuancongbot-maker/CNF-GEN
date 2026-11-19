"""CPU regressions for the evaluator's fixed surface-sampling randomness.

No GPU, no official weights, no data: only a tiny in-memory tetrahedron and
the sampling helpers from the evaluation module.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcnf_shape_completion import evaluate  # noqa: E402

_HAS_TRIMESH = importlib.util.find_spec("trimesh") is not None

_TETRA = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64),
)


def _rng_state() -> tuple:
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


class SeededSurfaceSamplingTests(unittest.TestCase):
    def test_explicit_offsets_cover_all_six_methods(self):
        self.assertEqual(set(evaluate.SURFACE_SAMPLING_OFFSETS), set(evaluate.METHODS))
        self.assertEqual(set(evaluate.CORRECTED_SAMPLING_OFFSETS), {"post", "cnf", "plain_post"})
        # Matched randomness for the paired CNF vs plain_post comparison.
        self.assertEqual(
            evaluate.CORRECTED_SAMPLING_OFFSETS["cnf"],
            evaluate.CORRECTED_SAMPLING_OFFSETS["plain_post"],
        )
        self.assertGreater(evaluate.CORRECTED_SAMPLING_OFFSETS["cnf"], 0)

    @unittest.skipUnless(_HAS_TRIMESH, "trimesh is needed for mesh sampling")
    def test_sampling_is_reproducible_and_does_not_touch_global_rng(self):
        verts, faces = _TETRA
        np.random.seed(20261119)
        before = _rng_state()
        first = evaluate.sample_surface_points(verts, faces, 64, seed=501)
        after_first = _rng_state()
        self.assertEqual(before, after_first)

        # A different global state must not change the explicitly seeded draw.
        np.random.seed(1)
        again = evaluate.sample_surface_points(verts, faces, 64, seed=501)
        self.assertTrue(np.array_equal(first, again))

        # A different explicit seed must produce a different sample.
        other = evaluate.sample_surface_points(verts, faces, 64, seed=907)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(first.shape, (64, 3))


if __name__ == "__main__":
    unittest.main()
