# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.fast_geo2cart.

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

r2 = np.sqrt(2.0)


class test_case(unittest.TestCase):

    # Separate (n, m) inputs give a tuple of three (n, m) float64 arrays
    def test_output_shape(self):
        az = (np.random.rand(2, 6) - 0.5) * 2 * np.pi
        el = (np.random.rand(2, 6) - 0.5) * np.pi
        length = np.random.rand(2, 6) + 0.5

        out = quadriga_lib.tools.fast_geo2cart(az, el, length)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        for a in out:
            self.assertEqual(a.shape, (2, 6))
            self.assertEqual(a.dtype, np.float64)

    # ---- Known values ----

    def test_known_values(self):
        # (0,    0,    1)  -> (1, 0, 0)
        # (pi/4, 0,    r2) -> (1, 1, 0)
        # (0,    pi/4, r2) -> (1, 0, 1)
        az     = np.array([[0.0], [np.pi / 4], [0.0]])
        el     = np.array([[0.0], [0.0],       [np.pi / 4]])
        length = np.array([[1.0], [r2],        [r2]])

        x, y, z = quadriga_lib.tools.fast_geo2cart(az, el, length)
        np.testing.assert_allclose(x.ravel(), [1.0, 1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(y.ravel(), [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(z.ravel(), [0.0, 0.0, 1.0], atol=1e-5)

    # Omitted len and len=None give unit length
    def test_default_unit_length(self):
        az = (np.random.rand(2, 3) - 0.5) * 2 * np.pi
        el = (np.random.rand(2, 3) - 0.5) * np.pi

        x, y, z = quadriga_lib.tools.fast_geo2cart(az, el)
        mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        np.testing.assert_allclose(mag, np.ones((2, 3)), atol=1e-5)
        np.testing.assert_array_equal(quadriga_lib.tools.fast_geo2cart(az, el, None), (x, y, z))
        np.testing.assert_array_equal(quadriga_lib.tools.fast_geo2cart(az, el, len=np.ones((2, 3))), (x, y, z))

    # Row vs column orientation: a transposed input transposes the outputs
    def test_orientation(self):
        az     = np.array([[np.pi], [-np.pi / 2]])
        el     = np.array([[np.pi / 4], [np.pi / 4]])
        length = np.array([[r2], [r2]])
        c = quadriga_lib.tools.fast_geo2cart(az, el, length)
        d = quadriga_lib.tools.fast_geo2cart(az.T, el.T, length.T)
        for a, b in zip(c, d):
            self.assertEqual(a.shape, (2, 1))
            self.assertEqual(b.shape, (1, 2))
            np.testing.assert_array_equal(a.T, b)

    # Outputs have the shape of the inputs (1D to 3D), with the values of the flattened call
    def test_shape_preserved(self):
        az, el, length = np.random.default_rng(0).random((3, 2, 3, 4))
        ref = quadriga_lib.tools.fast_geo2cart(az.ravel(), el.ravel(), length.ravel(), use_kernel=1)
        for shape in ((24,), (4, 6), (2, 3, 4)):
            out = quadriga_lib.tools.fast_geo2cart(az.reshape(shape), el.reshape(shape), length.reshape(shape),
                                                   use_kernel=1)
            for a, r in zip(out, ref):
                self.assertEqual(a.shape, shape)
                np.testing.assert_array_equal(a, r.reshape(shape))
        with self.assertRaises(ValueError):
            quadriga_lib.tools.fast_geo2cart(az, el.reshape(2, 12))
        with self.assertRaises(ValueError):
            quadriga_lib.tools.fast_geo2cart(az, el, length.reshape(6, 4))

    # Empty input gives empty outputs, as in C++
    def test_empty_input(self):
        e = np.zeros((0, 3))
        for out in (quadriga_lib.tools.fast_geo2cart(e, e), quadriga_lib.tools.fast_geo2cart(e, e, e)):
            for a in out:
                self.assertEqual(a.shape, (0, 3))

    # Outputs equal the reference formulas element-wise, for Fortran-ordered inputs (aliased by the wrapper),
    # C-ordered and float32 inputs (copied), and for n > 16 up to large n
    def test_values_equal_reference(self):
        rng = np.random.default_rng(0)
        for n, m in ((5, 1), (100, 3), (100000, 1)):
            az = rng.uniform(-3, 3, (n, m))
            el = rng.uniform(-1.5, 1.5, (n, m))
            ln = rng.uniform(1, 10, (n, m))
            ref = (np.cos(el) * np.cos(az) * ln, np.cos(el) * np.sin(az) * ln, np.sin(el) * ln)
            for order in ("F", "C"):
                a, e, l = (np.asarray(t, order=order) for t in (az, el, ln))
                out = quadriga_lib.tools.fast_geo2cart(a, e, l, use_kernel=1)
                for o, r, name in zip(out, ref, ("x", "y", "z")):
                    np.testing.assert_allclose(o, r, rtol=0, atol=1e-12, err_msg="%s, n=%d, order=%s" % (name, n, order))
        out = quadriga_lib.tools.fast_geo2cart(az.astype(np.float32), el.astype(np.float32), ln.astype(np.float32),
                                               use_kernel=1)
        np.testing.assert_allclose(out[2], ref[2], atol=1e-5)

    # ---- Round trip against fast_cart2geo ----

    def test_roundtrip(self):
        x0, y0, z0 = np.random.rand(3, 4, 3)
        az, el, length = quadriga_lib.tools.fast_cart2geo(x0, y0, z0)
        x1, y1, z1 = quadriga_lib.tools.fast_geo2cart(az, el, length)
        np.testing.assert_allclose(x1, x0, atol=1e-4)
        np.testing.assert_allclose(y1, y0, atol=1e-4)
        np.testing.assert_allclose(z1, z0, atol=1e-4)

    # ---- use_kernel variants ----

    def test_use_kernel(self):
        np.random.seed(0)
        az = (np.random.rand(3, 4) - 0.5) * 2 * np.pi
        el = (np.random.rand(3, 4) - 0.5) * np.pi
        length = np.random.rand(3, 4) + 0.5

        g_auto = quadriga_lib.tools.fast_geo2cart(az, el, length, use_kernel=0)
        g_gen = quadriga_lib.tools.fast_geo2cart(az, el, length, use_kernel=1)
        np.testing.assert_allclose(g_auto, g_gen, atol=1e-5)
        np.testing.assert_array_equal(quadriga_lib.tools.fast_geo2cart(az, el, length), g_auto)  # default kernel: 0

        try:
            g_avx2 = quadriga_lib.tools.fast_geo2cart(az, el, length, use_kernel=2)
        except ValueError as err:
            self.assertIn("AVX2", str(err))
        else:
            np.testing.assert_allclose(g_avx2, g_gen, atol=1e-4, rtol=0)

    # Keyword names as in the Usage line; a non-contiguous view gives the same result
    def test_keyword_arguments(self):
        az, el, length = np.random.default_rng(0).random((3, 4, 2))
        ref = quadriga_lib.tools.fast_geo2cart(az, el, length, use_kernel=1)
        np.testing.assert_array_equal(quadriga_lib.tools.fast_geo2cart(az=az, el=el, len=length, use_kernel=1), ref)
        a, e, l = (np.repeat(t, 2, axis=0)[::2] for t in (az, el, length))
        np.testing.assert_array_equal(quadriga_lib.tools.fast_geo2cart(a, e, l, use_kernel=1), ref)

    # Kernel values outside 0, 1, 2 are rejected
    def test_invalid_use_kernel(self):
        e = np.random.rand(2, 2)
        for k in (3, 99, -1):
            with self.assertRaises(ValueError, msg="use_kernel=%d" % k):
                quadriga_lib.tools.fast_geo2cart(e, e, use_kernel=k)

    # ---- Error paths ----

    def test_errors(self):
        e = (np.random.rand(2, 6) - 0.5) * 2 * np.pi

        # az / el shape mismatch, also with an equal number of elements
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_geo2cart(e[0:1, :], e)
        self.assertIn("'el'", str(ctx.exception))
        with self.assertRaises(ValueError):
            quadriga_lib.tools.fast_geo2cart(e, e.reshape(6, 2))
        with self.assertRaises(ValueError):
            quadriga_lib.tools.fast_geo2cart(e, np.array([]))

        # len shape mismatch
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_geo2cart(e, e, e[0:1, :])
        self.assertIn("'len'", str(ctx.exception))

        # az and el are mandatory; None is rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_geo2cart(e)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_geo2cart(None, e)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_geo2cart(e, None)

    # There is no combined (3, n, m) form: no combine flag, and a stacked array is not a single input
    def test_combined_form_removed(self):
        geo = np.random.rand(3, 4, 2)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_geo2cart(geo)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_geo2cart(geo[0], geo[1], geo[2], combine=True)


if __name__ == '__main__':
    unittest.main()
