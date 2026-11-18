# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.RTtools.generate_diffraction_paths.
# Scope: argument and output handling, level of detail, scalar mode, error paths.
# Physics validation lives in the Catch2 suite (test_calc_diffraction_gain.cpp).

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
        self.orig = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        self.dest = np.array([[20.0, 0.0, 1.0], [5.0, 30.0, 2.0], [0.0, 3.0, 40.0]])
        self.out = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 2)

    def test_basic_call(self):
        out = self.out
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        for a in out[:3]:
            self.assertEqual(a.shape, (3, 19, 2))  # lod 2: n_path 19, n_seg 3
        self.assertEqual(out[3].shape, (3, 19, 3))
        for a in out:
            self.assertEqual(a.dtype, np.float64)
        npt.assert_allclose(np.sum(np.prod(out[3], axis=2), axis=1), np.ones(3), atol=1e-12, rtol=0)
        # lod 7: direct line in 10 segments; waypoints on the line from orig to dest
        rx, ry, rz, w = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 7)
        self.assertEqual(rx.shape, (3, 1, 9))
        t = np.arange(1, 10) / 10
        for i in range(3):
            exp = self.orig[i][:, None] + (self.dest[i] - self.orig[i])[:, None] * t[None, :]
            npt.assert_allclose(np.stack([rx[i, 0], ry[i, 0], rz[i, 0]]), exp, atol=1e-12, rtol=0)

    def test_scalar_mode(self):
        out_n = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 2, scalar_mode=None)
        out_f = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 2, scalar_mode=False)
        for a, b, c in zip(out_n, out_f, self.out):
            npt.assert_array_equal(a, c)
            npt.assert_array_equal(b, c)
        # Same wavelength: acoustic 343 / f_a equals EM 299792458 / 3e9
        f_a = 343.0 * 3.0e9 / 299792458.0
        out_s = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, f_a, 2, scalar_mode=True)
        for a, c in zip(out_s, self.out):
            npt.assert_allclose(a, c, atol=1e-9, rtol=0)
        out_s2 = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 2, scalar_mode=True)
        self.assertGreater(np.max(np.abs(out_s2[2] - self.out[2])), 1e-6)  # different wavelength, other ellipsoid

    def test_keyword_arguments(self):
        out_k = quadriga_lib.RTtools.generate_diffraction_paths(orig=self.orig, dest=self.dest, center_freq=3.0e9,
                                                                lod=2, scalar_mode=False)
        for a, c in zip(out_k, self.out):
            npt.assert_array_equal(a, c)

    def test_array_layout_and_dtype(self):
        o0, d0 = self.orig.copy(), self.dest.copy()
        out_f = quadriga_lib.RTtools.generate_diffraction_paths(np.asfortranarray(self.orig), self.dest, 3.0e9, 2)
        npt.assert_array_equal(self.orig, o0)
        npt.assert_array_equal(self.dest, d0)
        big = np.zeros((6, 3))
        big[::2] = self.dest
        out_v = quadriga_lib.RTtools.generate_diffraction_paths(self.orig, big[::2], 3.0e9, 2)
        out_i = quadriga_lib.RTtools.generate_diffraction_paths(self.orig.astype(np.int32), self.dest.tolist(), 3e9, 2)
        for a, b, c, r in zip(out_f, out_v, out_i, self.out):
            npt.assert_array_equal(a, r)
            npt.assert_array_equal(b, r)
            npt.assert_array_equal(c, r)

    def test_errors(self):
        for lod in (0, 8):
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, lod)
            self.assertIn("lod", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest[:2], 3.0e9, 2)
        self.assertIn("dest", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.generate_diffraction_paths(self.orig[:, :2], self.dest, 3.0e9, 2)
        self.assertIn("3 columns", str(ctx.exception))
        for f in (0.0, np.nan, np.inf):
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, f, 2)
            self.assertIn("center_freq", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9, 2, scalar_mode=1)
        self.assertIn("scalar_mode", str(ctx.exception))
        with self.assertRaises(TypeError):
            quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, 3.0e9)
        with self.assertRaises(TypeError):
            quadriga_lib.RTtools.generate_diffraction_paths(self.orig, self.dest, lod=2, center_frequency=3.0e9)


if __name__ == "__main__":
    unittest.main()
