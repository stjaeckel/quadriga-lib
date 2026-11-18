# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.RTtools.subdivide_triangles.
# Scope: argument and output handling, material indices, error paths.
# Physics validation lives in the Catch2 suite (test_quadriga_tools.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def area(tri):
    return 0.5 * np.linalg.norm(np.cross(tri[:, 3:6] - tri[:, 0:3], tri[:, 6:9] - tri[:, 0:3]), axis=1)


class test_case(unittest.TestCase):

    def setUp(self):
        self.tri = np.array([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 3.0]])
        self.mtl = np.array([5, 2], dtype=np.uint64)

    def test_basic_call(self):
        out = quadriga_lib.RTtools.subdivide_triangles(3, self.tri, self.mtl)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        tri_out, mtl_out = out
        self.assertEqual(tri_out.shape, (18, 9))
        self.assertEqual(tri_out.dtype, np.float64)
        self.assertEqual(mtl_out.shape, (18,))
        self.assertEqual(mtl_out.dtype, np.int64)
        npt.assert_array_equal(mtl_out, [5] * 9 + [2] * 9)  # passed through, not shifted
        npt.assert_allclose([area(tri_out[:9]).sum(), area(tri_out[9:]).sum()], area(self.tri), atol=1e-12, rtol=0)
        tri_1, _ = quadriga_lib.RTtools.subdivide_triangles(1, self.tri)
        npt.assert_array_equal(tri_1, self.tri)

    def test_mtl_ind(self):
        tri_out, mtl_out = quadriga_lib.RTtools.subdivide_triangles(2, self.tri)
        self.assertEqual(mtl_out.shape, (0,))
        self.assertEqual(mtl_out.dtype, np.int64)
        tri_n, mtl_n = quadriga_lib.RTtools.subdivide_triangles(2, self.tri, None)
        npt.assert_array_equal(tri_n, tri_out)
        self.assertEqual(mtl_n.shape, (0,))
        tri_m, mtl_m = quadriga_lib.RTtools.subdivide_triangles(2, self.tri, [0, 7])  # list, 0 = no material
        npt.assert_array_equal(tri_m, tri_out)
        npt.assert_array_equal(mtl_m, [0] * 4 + [7] * 4)

    def test_keyword_arguments(self):
        out = quadriga_lib.RTtools.subdivide_triangles(2, self.tri, self.mtl)
        out_k = quadriga_lib.RTtools.subdivide_triangles(n_div=2, triangles_in=self.tri, mtl_ind=self.mtl)
        for a, b in zip(out_k, out):
            npt.assert_array_equal(a, b)

    def test_array_layout_and_dtype(self):
        tri0, mtl0 = self.tri.copy(), self.mtl.copy()
        tri_out, _ = quadriga_lib.RTtools.subdivide_triangles(2, self.tri, self.mtl)
        npt.assert_array_equal(self.tri, tri0)
        npt.assert_array_equal(self.mtl, mtl0)
        npt.assert_array_equal(quadriga_lib.RTtools.subdivide_triangles(2, np.asfortranarray(self.tri))[0], tri_out)
        big = np.zeros((4, 9))
        big[::2] = self.tri
        npt.assert_array_equal(quadriga_lib.RTtools.subdivide_triangles(2, big[::2])[0], tri_out)
        npt.assert_array_equal(quadriga_lib.RTtools.subdivide_triangles(2, self.tri.astype(np.int32))[0], tri_out)

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.subdivide_triangles(0, self.tri)
        self.assertIn("n_div", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.subdivide_triangles(2, self.tri[:, :8])
        self.assertIn("9 columns", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.subdivide_triangles(2, self.tri, [1, 2, 3])
        self.assertIn("mtl_ind", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.subdivide_triangles(2**40, self.tri)  # output beyond 64 bits
        self.assertIn("too large", str(ctx.exception))
        with self.assertRaises(TypeError):
            quadriga_lib.RTtools.subdivide_triangles(2)
        with self.assertRaises(TypeError):
            quadriga_lib.RTtools.subdivide_triangles(2, self.tri, mtl=self.mtl)


if __name__ == "__main__":
    unittest.main()
