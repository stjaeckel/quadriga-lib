# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.interp.
# Scope: argument and output handling, 1D / 2D selection, ranks, error paths.
# Physics validation lives in the Catch2 suite (test_interp.cpp, test_quadriga_tools.cpp).

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


class test_case(unittest.TestCase):

    def setUp(self):
        # Linear data: value = 1 + 2 x + 10 y + 100 e, so interpolation is exact inside the grid
        self.x = np.array([0.0, 1.0, 3.0, 4.0])
        self.y = np.array([0.0, 2.0, 5.0])
        e = np.arange(2)
        self.data = 1 + 2 * self.x[None, :, None] + 10 * self.y[:, None, None] + 100 * e[None, None, :]  # (3, 4, 2)
        self.x_new = np.array([0.5, 2.0, 3.5])
        self.y_new = np.array([1.0, 4.0])

    def test_2d(self):
        out = quadriga_lib.tools.interp(self.x, self.y, self.data, self.x_new, self.y_new)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 3, 2))
        self.assertEqual(out.dtype, np.float64)
        exp = 1 + 2 * self.x_new[None, :, None] + 10 * self.y_new[:, None, None] + 100 * np.arange(2)[None, None, :]
        npt.assert_allclose(out, exp, atol=1e-12, rtol=0)
        out2 = quadriga_lib.tools.interp(self.x, self.y, self.data[:, :, 1], self.x_new, self.y_new)  # 2D input
        self.assertEqual(out2.shape, (2, 3))
        npt.assert_array_equal(out2, out[:, :, 1])
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, self.y, self.data, self.x, self.y), self.data)

    def test_1d(self):
        d1 = self.data[0]  # (nx, ne) = (4, 2)
        out = quadriga_lib.tools.interp(self.x, None, d1, self.x_new)
        self.assertEqual(out.shape, (3, 2))
        npt.assert_allclose(out, 1 + 2 * self.x_new[:, None] + 100 * np.arange(2)[None, :], atol=1e-12, rtol=0)
        out1 = quadriga_lib.tools.interp(self.x, None, d1[:, 1], self.x_new)  # 1D input
        self.assertEqual(out1.shape, (3,))
        npt.assert_array_equal(out1, out[:, 1])
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, [], d1, self.x_new), out)  # empty y = 1D
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, None, d1, self.x_new, None), out)
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, None, d1, self.x), d1)
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, None, d1, [-1.0, 9.0]), d1[[0, 3]])  # constant extrapolation

    def test_keyword_arguments(self):
        out = quadriga_lib.tools.interp(self.x, self.y, self.data, self.x_new, self.y_new)
        out_k = quadriga_lib.tools.interp(x=self.x, y=self.y, input=self.data, x_new=self.x_new, y_new=self.y_new)
        npt.assert_array_equal(out_k, out)

    def test_array_layout_and_dtype(self):
        d0 = self.data.copy()
        out = quadriga_lib.tools.interp(self.x, self.y, self.data, self.x_new, self.y_new)
        npt.assert_array_equal(self.data, d0)
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, self.y, np.asfortranarray(self.data),
                                                         self.x_new, self.y_new), out)
        big = np.zeros((3, 8, 2))
        big[:, ::2, :] = self.data
        npt.assert_array_equal(quadriga_lib.tools.interp(self.x, self.y, big[:, ::2, :], self.x_new, self.y_new), out)
        npt.assert_allclose(quadriga_lib.tools.interp(self.x.astype(np.float32), list(self.y),
                                                      self.data.astype(np.float32), self.x_new, self.y_new),
                            out, atol=1e-4, rtol=0)  # float32 data rounding

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.interp(self.x, self.y, self.data, self.x_new)
        self.assertIn("y_new", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.interp(self.x, None, self.data[0], self.x_new, self.y_new)  # y forgotten
        self.assertIn("y_new", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.interp(self.x, None, self.data, self.x_new)  # 3D data in 1D
        self.assertIn("1D", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.interp(self.x, self.y, self.x, self.x_new, self.y_new)  # 1D data in 2D
        self.assertIn("2D", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.interp(self.x[:3], self.y, self.data, self.x_new, self.y_new)
        self.assertIn("Data dimensions", str(ctx.exception))
        with self.assertRaises(TypeError):
            quadriga_lib.tools.interp(self.x, self.y, self.data)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.interp(self.x, None, self.data[0], self.x_new, z_new=1)


if __name__ == "__main__":
    unittest.main()
