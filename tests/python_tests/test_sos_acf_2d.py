# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.acf_2d.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_acf_2d.cpp).

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
        # 3 samples with 1 m spacing -> grid of 9 points from -4 m to 4 m, center at index 4
        self.acf = np.array([1.0, 0.5, 0.25])
        self.dist = np.array([0.0, 1.0, 2.0])

    def test_basic_call(self):
        out = quadriga_lib.sos.acf_2d(self.acf, self.dist)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        acf_2d, dist_2d = out

        self.assertEqual(acf_2d.shape, (9, 9))
        self.assertEqual(dist_2d.shape, (9,))
        self.assertEqual(acf_2d.dtype, np.float64)
        self.assertEqual(dist_2d.dtype, np.float64)

        npt.assert_allclose(dist_2d, np.arange(-4.0, 5.0), atol=1e-14, rtol=0)
        npt.assert_allclose(acf_2d[4, 4:8], [1.0, 0.5, 0.25, 0.0], atol=1e-14, rtol=0)
        npt.assert_allclose(acf_2d[5, 5], 0.5 - (np.sqrt(2.0) - 1.0) * 0.25, atol=1e-14, rtol=0)
        npt.assert_allclose(acf_2d, acf_2d.T, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.acf_2d(self.acf, self.dist)
        out = quadriga_lib.sos.acf_2d(acf=self.acf, dist=self.dist)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_arguments_not_swapped(self):
        # Another spacing changes the axis only
        acf_2d, dist_2d = quadriga_lib.sos.acf_2d(self.acf, self.dist * 2.5)
        npt.assert_allclose(dist_2d, 2.5 * np.arange(-4.0, 5.0), atol=1e-13, rtol=0)
        npt.assert_allclose(acf_2d, quadriga_lib.sos.acf_2d(self.acf, self.dist)[0], atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.sos.acf_2d(self.acf, self.dist)
        variants = [
            lambda a: np.repeat(a, 2)[::2],     # non-contiguous view
            lambda a: a.reshape(-1, 1),         # column
            lambda a: a.astype(np.float32),     # other dtype
            lambda a: a.tolist(),               # plain list
        ]
        for fn in variants:
            out = quadriga_lib.sos.acf_2d(fn(self.acf), fn(self.dist))
            for a, b in zip(out, ref):
                self.assertEqual(a.dtype, np.float64)
                npt.assert_allclose(a, b, atol=1e-14, rtol=0)  # the values are exact in float32

    def test_inputs_not_modified(self):
        acf, dist = self.acf.copy(), self.dist.copy()
        quadriga_lib.sos.acf_2d(acf, dist)
        npt.assert_array_equal(acf, self.acf)
        npt.assert_array_equal(dist, self.dist)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_2d([1.0], [0.0])
        self.assertIn("acf", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_2d(self.acf, [0.0, 1.0, 3.0])
        self.assertIn("equally spaced", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_2d(self.acf)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_2d(self.acf, self.dist, no_such_argument=1)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.acf, self.dist]
        quadriga_lib.sos.acf_2d(*args)
        for i in range(2):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.acf_2d(*bad)


if __name__ == "__main__":
    unittest.main()
