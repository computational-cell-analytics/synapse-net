"""Offline regression tests for the edge filters migrated from vigra to bioimage-cpp.

These exercise the three vigra -> bioimage_cpp.filters replacements in
``synapse_net.ground_truth.shape_refinement.edge_filter`` ("laplace", "ggm",
"structure-tensor"). They use small synthetic images and assert meaningful
properties of the filter response, so a regression in the migration fails loudly.
"""
import unittest

import numpy as np


# The migrated filter methods (vigra -> bioimage_cpp.filters).
MIGRATED_METHODS = ("laplace", "ggm", "structure-tensor")


class TestShapeRefinement(unittest.TestCase):
    def test_edge_filter_methods(self):
        from synapse_net.ground_truth.shape_refinement import edge_filter

        # Image with a sharp vertical step edge at column 32.
        step = np.zeros((64, 64), dtype="float32")
        step[:, 32:] = 1.0

        # Image with a bright square (has corners), used for the structure tensor
        # whose second eigenvalue is ~0 along an ideal straight edge.
        square = np.zeros((64, 64), dtype="float32")
        square[20:44, 20:44] = 1.0

        for method in MIGRATED_METHODS:
            data = square if method == "structure-tensor" else step
            edge_map = edge_filter(data, sigma=2.0, method=method)

            # Basic invariants for every migrated filter.
            self.assertEqual(edge_map.shape, data.shape)
            self.assertTrue(np.issubdtype(edge_map.dtype, np.floating))
            self.assertTrue(np.all(np.isfinite(edge_map)))
            # The filter must produce a non-trivial response.
            self.assertGreater(float(edge_map.std()), 0.0)

            # For laplace and ggm the response must be stronger at the edge than
            # in a flat region (a real correctness check, not just a shape check).
            if method in ("laplace", "ggm"):
                edge_response = np.abs(edge_map[:, 28:37]).max()
                flat_response = np.abs(edge_map[:, 2:10]).max()
                self.assertGreater(edge_response, flat_response)

    def test_edge_filter_3d_per_slice(self):
        from synapse_net.ground_truth.shape_refinement import edge_filter

        data = np.zeros((3, 32, 32), dtype="float32")
        data[:, :, 16:] = 1.0

        edge_map = edge_filter(data, sigma=2.0, method="ggm", per_slice=True, n_threads=1)
        self.assertEqual(edge_map.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(edge_map)))


if __name__ == "__main__":
    unittest.main()
