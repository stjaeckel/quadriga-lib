# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.fast_sincos.
# Scope: argument and output handling, kernel selection, error paths.
# Physics validation lives in the Catch2 suite (test_fast_sincos.cpp).

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
        self.x = np.linspace(-20.0, 30.0, 37)  # 37: not a multiple of the AVX2 width

    def test_basic_call(self):
        out = quadriga_lib.tools.fast_sincos(self.x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        sin_x, cos_x = out
        for a in out:
            self.assertEqual(a.shape, (37,))
            self.assertEqual(a.dtype, np.float32)
        npt.assert_allclose(sin_x, np.sin(self.x), atol=2e-7, rtol=0)  # documented error 1.3e-7
        npt.assert_allclose(cos_x, np.cos(self.x), atol=2e-7, rtol=0)

    def test_use_kernel(self):
        s0, c0 = quadriga_lib.tools.fast_sincos(self.x)
        s1, c1 = quadriga_lib.tools.fast_sincos(x=self.x, use_kernel=1)
        npt.assert_allclose(s1, s0, atol=3e-7, rtol=0)  # GENERIC and AVX2: each within 1.3e-7 of the exact value
        npt.assert_allclose(c1, c0, atol=3e-7, rtol=0)
        npt.assert_array_equal(quadriga_lib.tools.fast_sincos(self.x, use_kernel=0)[0], s0)

    def test_array_layout_and_dtype(self):
        x0 = self.x.copy()
        s, c = quadriga_lib.tools.fast_sincos(self.x)
        npt.assert_array_equal(self.x, x0)
        big = np.zeros(74)
        big[::2] = self.x
        npt.assert_array_equal(quadriga_lib.tools.fast_sincos(big[::2])[0], s)
        x32 = self.x.astype(np.float32)
        npt.assert_allclose(quadriga_lib.tools.fast_sincos(x32)[1], np.cos(x32.astype(np.float64)),
                            atol=2e-7, rtol=0)  # documented error 1.3e-7
        npt.assert_array_equal(quadriga_lib.tools.fast_sincos(list(self.x))[1], c)
        e = quadriga_lib.tools.fast_sincos(np.zeros(0))
        self.assertEqual(e[0].shape, (0,))

    def test_shape(self):
        # Elementwise: outputs keep the shape of x, also for C-ordered and 0-d inputs
        x2 = self.x[:36].reshape(4, 9)
        s2, c2 = quadriga_lib.tools.fast_sincos(x2)
        self.assertEqual(s2.shape, (4, 9))
        self.assertEqual(c2.shape, (4, 9))
        npt.assert_allclose(s2, np.sin(x2), atol=2e-7, rtol=0)
        npt.assert_allclose(c2, np.cos(x2), atol=2e-7, rtol=0)
        x3 = self.x[:36].reshape(2, 3, 6)
        npt.assert_allclose(quadriga_lib.tools.fast_sincos(x3)[0], np.sin(x3), atol=2e-7, rtol=0)
        s0, c0 = quadriga_lib.tools.fast_sincos(0.5)
        self.assertEqual(s0.shape, ())
        npt.assert_allclose(c0, np.cos(0.5), atol=2e-7, rtol=0)

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.fast_sincos(self.x, use_kernel=5)
        self.assertIn("use_kernel", str(ctx.exception))
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_sincos()
        with self.assertRaises(TypeError):
            quadriga_lib.tools.fast_sincos(self.x, cos_only=True)


if __name__ == "__main__":
    unittest.main()
