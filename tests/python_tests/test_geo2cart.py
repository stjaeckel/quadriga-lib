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

r2 = np.sqrt(2.0)


class test_case(unittest.TestCase):

    # --- Separate input, default (combine=True) ---

    def test_separate_default(self):
        az = (np.random.rand(2, 6) - 0.5) * 2 * np.pi
        el = (np.random.rand(2, 6) - 0.5) * np.pi
        length = np.random.rand(2, 6) + 0.5

        cart = quadriga_lib.tools.geo2cart(az, el, length)
        self.assertEqual(cart.shape, (3, 2, 6))
        self.assertEqual(cart.dtype, np.float64)

    # combine=True rows must equal the combine=False tuple
    def test_combine_flag_consistency(self):
        az = (np.random.rand(4, 3) - 0.5) * 2 * np.pi
        el = (np.random.rand(4, 3) - 0.5) * np.pi
        length = np.random.rand(4, 3) + 0.5

        cart = quadriga_lib.tools.geo2cart(az, el, length, combine=True)
        x, y, z = quadriga_lib.tools.geo2cart(az, el, length, combine=False)

        self.assertEqual(x.shape, (4, 3))
        np.testing.assert_allclose(cart[0], x, atol=1e-12)
        np.testing.assert_allclose(cart[1], y, atol=1e-12)
        np.testing.assert_allclose(cart[2], z, atol=1e-12)

    # --- Known values ---

    def test_known_values(self):
        # (0,    0,    1)  -> (1, 0, 0)
        # (pi/4, 0,    r2) -> (1, 1, 0)
        # (0,    pi/4, r2) -> (1, 0, 1)
        az     = np.array([[0.0], [np.pi / 4], [0.0]])
        el     = np.array([[0.0], [0.0],       [np.pi / 4]])
        length = np.array([[1.0], [r2],        [r2]])

        x, y, z = quadriga_lib.tools.geo2cart(az, el, length, combine=False)
        np.testing.assert_allclose(x.ravel(), [1.0, 1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(y.ravel(), [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(z.ravel(), [0.0, 0.0, 1.0], atol=1e-5)

    def test_known_values_two_points(self):
        # (pi,    pi/4, r2) -> (-1,  0, 1)
        # (-pi/2, pi/4, r2) -> ( 0, -1, 1)
        az     = np.array([[np.pi], [-np.pi / 2]])
        el     = np.array([[np.pi / 4], [np.pi / 4]])
        length = np.array([[r2], [r2]])

        cart = quadriga_lib.tools.geo2cart(az, el, length)
        self.assertEqual(cart.shape, (3, 2, 1))
        np.testing.assert_allclose(cart[:, 0, 0], [-1.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(cart[:, 1, 0], [0.0, -1.0, 1.0], atol=1e-5)

    # Omitted len defaults to unit length
    def test_default_unit_length(self):
        az = (np.random.rand(2, 3) - 0.5) * 2 * np.pi
        el = (np.random.rand(2, 3) - 0.5) * np.pi

        x, y, z = quadriga_lib.tools.geo2cart(az, el, combine=False)
        mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        np.testing.assert_allclose(mag, np.ones((2, 3)), atol=1e-5)

    # Row vs column orientation: a transposed input transposes the spatial axes
    def test_orientation(self):
        az_c     = np.array([[np.pi], [-np.pi / 2]])
        el_c     = np.array([[np.pi / 4], [np.pi / 4]])
        length_c = np.array([[r2], [r2]])
        c = quadriga_lib.tools.geo2cart(az_c, el_c, length_c)         # (3, 2, 1)

        az_r     = np.array([[np.pi, -np.pi / 2]])
        el_r     = np.array([[np.pi / 4, np.pi / 4]])
        length_r = np.array([[r2, r2]])
        d = quadriga_lib.tools.geo2cart(az_r, el_r, length_r)         # (3, 1, 2)

        self.assertEqual(c.shape, (3, 2, 1))
        self.assertEqual(d.shape, (3, 1, 2))
        np.testing.assert_allclose(np.transpose(c, (0, 2, 1)), d, atol=1e-5)

    # --- Combined geo_coords input ---

    def test_combined_input_matches_separate(self):
        geo = np.random.rand(3, 4, 2)
        cart_comb = quadriga_lib.tools.geo2cart(geo)
        cart_sep = quadriga_lib.tools.geo2cart(geo[0], geo[1], geo[2])
        self.assertEqual(cart_comb.shape, (3, 4, 2))
        np.testing.assert_allclose(cart_comb, cart_sep, atol=1e-12)

    # --- Round trips against cart2geo ---

    def test_roundtrip_combined(self):
        cart = np.random.rand(3, 5, 4)
        geo = quadriga_lib.tools.cart2geo(cart)
        cart_rt = quadriga_lib.tools.geo2cart(geo)
        self.assertEqual(cart_rt.shape, (3, 5, 4))
        np.testing.assert_allclose(cart_rt, cart, atol=1e-4)

    def test_roundtrip_separate(self):
        x0 = np.random.rand(4, 3)
        y0 = np.random.rand(4, 3)
        z0 = np.random.rand(4, 3)
        az, el, length = quadriga_lib.tools.cart2geo(x0, y0, z0, combine=False)
        x1, y1, z1 = quadriga_lib.tools.geo2cart(az, el, length, combine=False)
        np.testing.assert_allclose(x1, x0, atol=1e-4)
        np.testing.assert_allclose(y1, y0, atol=1e-4)
        np.testing.assert_allclose(z1, z0, atol=1e-4)

    # --- dtype handling ---

    def test_dtype_conversion(self):
        e = ((np.random.rand(2, 6) - 0.5) * 2 * np.pi).astype(np.float32)
        cart = quadriga_lib.tools.geo2cart(e, e, e)
        self.assertEqual(cart.shape, (3, 2, 6))
        self.assertEqual(cart.dtype, np.float64)

    # --- use_kernel variants ---

    def test_use_kernel(self):
        np.random.seed(0)
        az = (np.random.rand(3, 4) - 0.5) * 2 * np.pi
        el = (np.random.rand(3, 4) - 0.5) * np.pi
        length = np.random.rand(3, 4) + 0.5

        g_auto = quadriga_lib.tools.geo2cart(az, el, length, use_kernel=0)
        g_gen = quadriga_lib.tools.geo2cart(az, el, length, use_kernel=1)
        np.testing.assert_allclose(g_auto, g_gen, atol=1e-5)

        try:
            g_avx2 = quadriga_lib.tools.geo2cart(az, el, length, use_kernel=2)
            np.testing.assert_allclose(g_avx2, g_gen, atol=1e-4)
        except (ValueError, RuntimeError):
            pass

    # --- Error paths ---

    def test_errors(self):
        e = (np.random.rand(2, 6) - 0.5) * 2 * np.pi

        # Empty az (separate mode, el provided)
        with self.assertRaises(ValueError):
            quadriga_lib.tools.geo2cart(np.array([]), e)

        # Empty el
        with self.assertRaises(ValueError):
            quadriga_lib.tools.geo2cart(e, np.array([]))

        # az / el shape mismatch
        with self.assertRaises(ValueError):
            quadriga_lib.tools.geo2cart(e[0:1, :], e)
        with self.assertRaises(ValueError):
            quadriga_lib.tools.geo2cart(e, e[0:1, :])

        # len shape mismatch
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.geo2cart(e, e, e[0:1, :])
        self.assertIn("len", str(ctx.exception))

        # Combined input without 3 rows
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.geo2cart(np.random.rand(2, 4))
        self.assertIn("(3, n, m)", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()