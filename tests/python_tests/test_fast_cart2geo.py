# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Tests of quadriga_lib.tools.fast_cart2geo: shapes, values, kernels, error paths.

import sys
import os
import unittest
import numpy as np

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

v = 1.0 / np.sqrt(2.0)


class test_case(unittest.TestCase):

    # Separate (n, m) inputs give a tuple of three (n, m) float64 arrays
    def test_output_shape(self):
        x, y, z = np.random.rand(3, 6, 2)
        out = quadriga_lib.tools.fast_cart2geo(x, y, z)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        for a in out:
            self.assertEqual(a.shape, (6, 2))
            self.assertEqual(a.dtype, np.float64)

    def test_known_values(self):
        # point 0: (-v, -v, 0) -> az=-3pi/4, el=0,    len=1
        # point 1: ( 0, 2v, 2v) -> az= pi/2, el=pi/4, len=2
        x = np.array([[-v], [0.0]])
        y = np.array([[-v], [2 * v]])
        z = np.array([[0.0], [2 * v]])

        az, el, length = quadriga_lib.tools.fast_cart2geo(x, y, z)
        self.assertEqual(az.shape, (2, 1))

        np.testing.assert_allclose(az.ravel(),     [-3 * np.pi / 4, np.pi / 2], atol=1e-5)
        np.testing.assert_allclose(el.ravel(),     [0.0,            np.pi / 4], atol=1e-5)
        np.testing.assert_allclose(length.ravel(), [1.0,            2.0],       atol=1e-5)

    # Outputs have the shape of the inputs (0D to 3D), with the values of the flattened call
    def test_shape_preserved(self):
        x, y, z = np.random.default_rng(0).random((3, 2, 3, 4))
        ref = quadriga_lib.tools.fast_cart2geo(x.ravel(), y.ravel(), z.ravel(), use_kernel=1)
        for shape in ((24,), (4, 6), (2, 3, 4)):
            out = quadriga_lib.tools.fast_cart2geo(x.reshape(shape), y.reshape(shape), z.reshape(shape), use_kernel=1)
            for a, r in zip(out, ref):
                self.assertEqual(a.shape, shape)
                np.testing.assert_array_equal(a, r.reshape(shape))
        out = quadriga_lib.tools.fast_cart2geo(np.array(1.0), np.array(1.0), np.array(0.0))
        self.assertEqual(out[0].shape, ())
        np.testing.assert_allclose(out[0], np.pi / 4, atol=1e-6)

    # Empty input gives empty outputs, as in C++
    def test_empty_input(self):
        e = np.zeros((0, 3))
        az, el, length = quadriga_lib.tools.fast_cart2geo(e, e, e)
        self.assertEqual(az.shape, (0, 3))
        self.assertEqual(el.shape, (0, 3))
        self.assertEqual(length.shape, (0, 3))

    # Outputs equal the reference formulas element-wise, for Fortran-ordered inputs (aliased by the wrapper),
    # C-ordered and float32 inputs (copied), and small and large n
    def test_values_equal_reference(self):
        rng = np.random.default_rng(1)
        for n, m in ((5, 1), (100, 3), (100000, 1)):
            x, y, z = rng.uniform(-3, 3, (3, n, m))
            ln_ref = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            ref = (np.arctan2(y, x), np.arcsin(np.clip(z / ln_ref, -1, 1)), ln_ref)
            for order in ("F", "C"):
                xo, yo, zo = (np.asarray(a, order=order) for a in (x, y, z))
                out = quadriga_lib.tools.fast_cart2geo(xo, yo, zo, use_kernel=1)
                for a, r, name in zip(out, ref, ("az", "el", "length")):
                    np.testing.assert_allclose(a, r, rtol=0, atol=1e-12, err_msg="%s, n=%d, order=%s" % (name, n, order))
        out = quadriga_lib.tools.fast_cart2geo(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32),
                                               use_kernel=1)
        np.testing.assert_allclose(out[2], ln_ref, rtol=1e-6)

    def test_use_kernel(self):
        np.random.seed(0)
        x, y, z = np.random.rand(3, 6, 2)

        g_auto = quadriga_lib.tools.fast_cart2geo(x, y, z, use_kernel=0)  # auto
        g_gen = quadriga_lib.tools.fast_cart2geo(x, y, z, use_kernel=1)   # GENERIC (double)
        np.testing.assert_allclose(g_auto, g_gen, atol=1e-5)
        np.testing.assert_array_equal(quadriga_lib.tools.fast_cart2geo(x, y, z), g_auto)  # default kernel: 0 (auto)

        # AVX2 path: matches GENERIC when available (single precision internally),
        # otherwise raises -- accept either for portability.
        try:
            g_avx2 = quadriga_lib.tools.fast_cart2geo(x, y, z, use_kernel=2)
        except ValueError as err:
            self.assertIn("AVX2", str(err))
        else:
            np.testing.assert_allclose(g_avx2, g_gen, atol=1e-4, rtol=0)

    # Keyword names as in the Usage line; a non-contiguous view gives the same result
    def test_keyword_arguments(self):
        x, y, z = np.random.default_rng(0).random((3, 4, 2))
        ref = quadriga_lib.tools.fast_cart2geo(x, y, z, use_kernel=1)
        np.testing.assert_array_equal(quadriga_lib.tools.fast_cart2geo(x=x, y=y, z=z, use_kernel=1), ref)
        xs, ys, zs = (np.repeat(a, 2, axis=0)[::2] for a in (x, y, z))
        np.testing.assert_array_equal(quadriga_lib.tools.fast_cart2geo(xs, ys, zs, use_kernel=1), ref)

    # Kernel values outside 0, 1, 2 are rejected, not run with the auto kernel
    def test_invalid_use_kernel(self):
        x, y, z = np.random.rand(3, 2, 2)
        for k in (3, 99, -1):
            with self.assertRaises(ValueError, msg="use_kernel=%d" % k):
                quadriga_lib.tools.fast_cart2geo(x, y, z, use_kernel=k)

    def test_errors(self):
        x = np.random.rand(4, 1)

        # 4D input: adapter rejects before the wrapper checks shape
        e4 = np.random.rand(3, 6, 5, 2)
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_cart2geo(e4, e4, e4)
        self.assertIn("4D", str(ctx.exception))

        # Shape mismatch, also with an equal number of elements
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_cart2geo(x, np.random.rand(5, 1), x)
        self.assertIn("'y'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_cart2geo(x, x, np.random.rand(1, 4))
        self.assertIn("'z'", str(ctx.exception))
        x3 = np.random.rand(2, 3, 2)
        with self.assertRaises(ValueError):
            quadriga_lib.tools.fast_cart2geo(x3, x3.reshape(2, 6), x3)

        # y and z are mandatory; None is rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_cart2geo(x)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_cart2geo(x, None, x)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_cart2geo(None, x, x)

    # There is no combined (3, n, m) form: no combine flag, and a stacked array is not a single input
    def test_combined_form_removed(self):
        cart = np.random.rand(3, 4, 2)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_cart2geo(cart)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_cart2geo(cart[0], cart[1], cart[2], combine=True)


if __name__ == '__main__':
    unittest.main()
