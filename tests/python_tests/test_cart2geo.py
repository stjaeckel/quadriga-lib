# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

# Now you can import your package
import quadriga_lib

v = 1.0 / np.sqrt(2.0)


class test_case(unittest.TestCase):

    # --- Combined input, default (combine=True) ---

    def test_combined_default(self):
        e = np.random.rand(3, 6, 2)
        geo = quadriga_lib.tools.cart2geo(e)
        self.assertEqual(geo.shape, (3, 6, 2))
        self.assertEqual(geo.dtype, np.float64)

    # combine=True rows must equal the combine=False tuple
    def test_combine_flag_consistency(self):
        e = np.random.rand(3, 4, 3)
        geo = quadriga_lib.tools.cart2geo(e, combine=True)
        az, el, length = quadriga_lib.tools.cart2geo(e, combine=False)

        self.assertEqual(az.shape, (4, 3))
        self.assertEqual(el.shape, (4, 3))
        self.assertEqual(length.shape, (4, 3))

        np.testing.assert_allclose(geo[0], az, atol=1e-12)
        np.testing.assert_allclose(geo[1], el, atol=1e-12)
        np.testing.assert_allclose(geo[2], length, atol=1e-12)

    # --- Known values via combined cube ---

    def test_known_values(self):
        # point 0: (-v, -v, 0) -> az=-3pi/4, el=0,    len=1
        # point 1: ( 0, 2v, 2v) -> az= pi/2, el=pi/4, len=2
        cart = np.zeros((3, 2, 1))
        cart[:, 0, 0] = [-v, -v, 0.0]
        cart[:, 1, 0] = [0.0, 2 * v, 2 * v]

        az, el, length = quadriga_lib.tools.cart2geo(cart, combine=False)
        self.assertEqual(az.shape, (2, 1))

        np.testing.assert_allclose(az.ravel(),     [-3 * np.pi / 4, np.pi / 2], atol=1e-5)
        np.testing.assert_allclose(el.ravel(),     [0.0,            np.pi / 4], atol=1e-5)
        np.testing.assert_allclose(length.ravel(), [1.0,            2.0],       atol=1e-5)

    # 2D (3, N) input is treated as a (3, N, 1) cube -> outputs (N, 1)
    def test_2d_input_as_cube(self):
        # columns are points: (0,1,1) and (v,v,-1)
        cart = np.array([[0.0, v],
                         [1.0, v],
                         [1.0, -1.0]])
        az, el, length = quadriga_lib.tools.cart2geo(cart, combine=False)

        self.assertEqual(az.shape, (2, 1))
        np.testing.assert_allclose(az.ravel(),     [np.pi / 2, np.pi / 4],   atol=1e-5)
        np.testing.assert_allclose(el.ravel(),     [np.pi / 4, -np.pi / 4],  atol=1e-5)
        np.testing.assert_allclose(length.ravel(), [np.sqrt(2), np.sqrt(2)], atol=1e-5)

    # --- Separate x, y, z input path ---

    def test_separate_input_values(self):
        x = np.array([[0.0], [v]])
        y = np.array([[1.0], [v]])
        z = np.array([[1.0], [-1.0]])

        az, el, length = quadriga_lib.tools.cart2geo(x, y, z, combine=False)
        self.assertEqual(az.shape, (2, 1))

        np.testing.assert_allclose(az.ravel(),     [np.pi / 2, np.pi / 4],   atol=1e-5)
        np.testing.assert_allclose(el.ravel(),     [np.pi / 4, -np.pi / 4],  atol=1e-5)
        np.testing.assert_allclose(length.ravel(), [np.sqrt(2), np.sqrt(2)], atol=1e-5)

    # Separate path must match the equivalent cube path
    def test_separate_matches_combined(self):
        x = np.array([[0.0], [v]])
        y = np.array([[1.0], [v]])
        z = np.array([[1.0], [-1.0]])
        cart = np.stack([x, y, z], axis=0)  # (3, 2, 1)

        geo_cube = quadriga_lib.tools.cart2geo(cart)
        geo_sep = quadriga_lib.tools.cart2geo(x, y, z)
        np.testing.assert_allclose(geo_cube, geo_sep, atol=1e-12)

    # Separate (n, m) matrices preserve their shape
    def test_separate_shape_preserved(self):
        xm = np.random.rand(3, 4)
        ym = np.random.rand(3, 4)
        zm = np.random.rand(3, 4)

        az, el, length = quadriga_lib.tools.cart2geo(xm, ym, zm, combine=False)
        self.assertEqual(az.shape, (3, 4))
        self.assertEqual(el.shape, (3, 4))
        self.assertEqual(length.shape, (3, 4))

        geo = quadriga_lib.tools.cart2geo(xm, ym, zm)
        self.assertEqual(geo.shape, (3, 3, 4))

    # --- use_kernel variants ---

    def test_use_kernel(self):
        np.random.seed(0)
        e = np.random.rand(3, 6, 2)

        g_auto = quadriga_lib.tools.cart2geo(e, use_kernel=0)  # auto
        g_gen = quadriga_lib.tools.cart2geo(e, use_kernel=1)   # GENERIC (double)
        np.testing.assert_allclose(g_auto, g_gen, atol=1e-5)

        # AVX2 path: matches GENERIC when available (single precision internally),
        # otherwise raises -- accept either for portability.
        try:
            g_avx2 = quadriga_lib.tools.cart2geo(e, use_kernel=2)
            np.testing.assert_allclose(g_avx2, g_gen, atol=1e-4)
        except (ValueError, RuntimeError):
            pass

    # --- Error paths ---

    def test_errors(self):
        # 4D input: adapter rejects before the wrapper checks shape
        e4 = np.random.rand(3, 6, 5, 2)
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.cart2geo(e4)
        self.assertEqual(str(ctx.exception), "Expected 1D, 2D or 3D array, got 4D")

        # Combined input without 3 rows
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.cart2geo(np.random.rand(2, 4))
        self.assertIn("(3, n, m)", str(ctx.exception))

        # Separate inputs with size mismatch
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.cart2geo(np.random.rand(4, 1),
                                        np.random.rand(5, 1),
                                        np.random.rand(4, 1))
        self.assertIn("equal in size", str(ctx.exception))

        # Only one of y / z provided
        with self.assertRaises(ValueError):
            quadriga_lib.tools.cart2geo(np.random.rand(4, 1), None, np.random.rand(4, 1))


if __name__ == '__main__':
    unittest.main()