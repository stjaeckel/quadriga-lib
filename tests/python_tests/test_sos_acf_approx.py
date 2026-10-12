# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.acf_approx.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_acf_approx.cpp).

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
        # One sinusoid with three different components; 4 samples with 2 m spacing -> 13 grid points
        self.freq = np.array([[0.03, 0.05, -0.02]])
        self.dist = np.array([0.0, 2.0, 4.0, 6.0])
        self.ax = np.arange(-12.0, 13.0, 2.0)

    def test_basic_call(self):
        out = quadriga_lib.sos.acf_approx(self.freq, self.dist)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        acf_approx, dist_2d = out

        self.assertEqual(acf_approx.shape, (13, 13, 3))
        self.assertEqual(dist_2d.shape, (13,))
        self.assertEqual(acf_approx.dtype, np.float64)
        self.assertEqual(dist_2d.dtype, np.float64)
        npt.assert_allclose(dist_2d, self.ax, atol=1e-13, rtol=0)

        # Planes x-y, x-z, y-z on the last axis; column = first axis of the plane, row = second axis
        c, r = np.meshgrid(self.ax, self.ax)
        npt.assert_allclose(acf_approx[:, :, 0], np.cos(2 * np.pi * (0.03 * c + 0.05 * r)), atol=1e-12, rtol=0)
        npt.assert_allclose(acf_approx[:, :, 1], np.cos(2 * np.pi * (0.03 * c - 0.02 * r)), atol=1e-12, rtol=0)
        npt.assert_allclose(acf_approx[:, :, 2], np.cos(2 * np.pi * (0.05 * c - 0.02 * r)), atol=1e-12, rtol=0)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.acf_approx(self.freq, self.dist)
        out = quadriga_lib.sos.acf_approx(freq=self.freq, dist=self.dist)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_dimensions(self):
        ref, _ = quadriga_lib.sos.acf_approx(self.freq, self.dist)

        # 2-D and 1-D: the output stays 3-dimensional
        a2, _ = quadriga_lib.sos.acf_approx(self.freq[:, :2], self.dist)
        self.assertEqual(a2.shape, (13, 13, 1))
        npt.assert_allclose(a2[:, :, 0], ref[:, :, 0], atol=1e-12, rtol=0)

        a1, d1 = quadriga_lib.sos.acf_approx(self.freq[:, :1], self.dist)
        self.assertEqual(a1.shape, (13, 1, 1))
        self.assertEqual(d1.shape, (13,))
        npt.assert_allclose(a1[:, 0, 0], np.cos(2 * np.pi * 0.03 * self.ax), atol=1e-12, rtol=0)

        # Several sinusoids (rows of freq) are averaged
        a4, _ = quadriga_lib.sos.acf_approx([[0.03], [0.07]], self.dist)
        npt.assert_allclose(a4[:, 0, 0], 0.5 * (np.cos(2 * np.pi * 0.03 * self.ax) + np.cos(2 * np.pi * 0.07 * self.ax)),
                            atol=1e-12, rtol=0)

    def test_array_layout_and_dtype(self):
        freq = np.array([[0.03, 0.05, -0.02], [-0.07, 0.01, 0.04]])
        ref = quadriga_lib.sos.acf_approx(freq, self.dist)
        variants = [
            (np.ascontiguousarray(freq), self.dist),                                # C order
            (np.asfortranarray(freq), self.dist.reshape(-1, 1)),                    # Fortran order, column
            (np.repeat(freq, 2, axis=0)[::2, :], np.repeat(self.dist, 2)[::2]),     # non-contiguous views
            (freq.astype(np.float32), self.dist.astype(np.int32)),                  # other dtypes
            (freq.tolist(), self.dist.tolist()),                                    # plain lists
        ]
        for f, d in variants:
            out = quadriga_lib.sos.acf_approx(f, d)
            for a, b in zip(out, ref):
                self.assertEqual(a.dtype, np.float64)
                npt.assert_allclose(a, b, atol=1e-5, rtol=0)  # float32 frequencies

    def test_inputs_not_modified(self):
        freq, dist = self.freq.copy(), self.dist.copy()
        quadriga_lib.sos.acf_approx(freq, dist)
        npt.assert_array_equal(freq, self.freq)
        npt.assert_array_equal(dist, self.dist)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_approx(np.ones((2, 4)), self.dist)
        self.assertIn("freq", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_approx(self.freq, [0.0])
        self.assertIn("dist", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_approx(self.freq, [0.0, 1.0, 3.0])
        self.assertIn("equally spaced", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_approx(self.freq)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_approx(self.freq, self.dist, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
