# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.calc_mse.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_calc_mse.cpp).

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
        # 1-D, two sinusoids; the target is their own ACF, so there is no error inside the range
        self.freq = np.array([[0.04], [0.13]])
        self.dist = np.arange(0.0, 20.5, 0.5)
        self.acf = 0.5 * (np.cos(2 * np.pi * 0.04 * self.dist) + np.cos(2 * np.pi * 0.13 * self.dist))

        # Another target and a 3-D set of sinusoids
        self.other = np.exp(-self.dist / 5.0)
        self.f3 = np.array([[0.03, 0.05, -0.02], [-0.07, 0.01, 0.04]])

    def test_basic_call(self):
        out = quadriga_lib.sos.calc_mse(self.freq, self.acf, self.dist)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)
        mse, mse_core, mse_all, acf_tail, test_dir = out

        for x in (mse, mse_core, mse_all):
            self.assertIsInstance(x, float)
        self.assertGreater(mse, 250.0)
        self.assertEqual(mse_core, mse)  # 1-D: both are evaluated on the sample points
        self.assertTrue(0.0 < mse_all < 30.0)

        self.assertEqual(acf_tail.shape, (4,))
        self.assertEqual(acf_tail.dtype, np.float64)
        d = 20.0 * np.array([2.0, 4.0, 10.0, 20.0])
        npt.assert_allclose(acf_tail, 0.5 * (np.cos(2 * np.pi * 0.04 * d) + np.cos(2 * np.pi * 0.13 * d)),
                            atol=1e-11, rtol=0)

        self.assertIsInstance(test_dir, np.ndarray)
        self.assertEqual(test_dir.shape, (0, 0))  # no test directions in 1-D
        self.assertEqual(test_dir.dtype, np.float64)

        # A wrong target: the error follows from the definition
        mse2 = quadriga_lib.sos.calc_mse(self.freq, self.other, self.dist)[0]
        npt.assert_allclose(mse2, -10.0 * np.log10(np.mean(np.square(self.other - self.acf))), atol=1e-9, rtol=0)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.calc_mse(self.f3, self.other, self.dist, 7)
        out = quadriga_lib.sos.calc_mse(freq=self.f3, acf=self.other, dist=self.dist, n_dir=7)
        self.assertEqual(out[:3], ref[:3])
        npt.assert_array_equal(out[3], ref[3])
        npt.assert_array_equal(out[4], ref[4])

    def test_optional_inputs(self):
        # 3-D, default n_dir = 10: test directions (azimuth, elevation)
        m3, c3, a3, t3, dir3 = quadriga_lib.sos.calc_mse(self.f3, self.other, self.dist)
        self.assertEqual(dir3.shape, (10, 2))
        npt.assert_allclose(np.sin(dir3[:, 1]), (np.arange(10) + 0.5) / 10.0, atol=1e-13, rtol=0)
        self.assertTrue(c3 > 0.0 and a3 > 0.0 and c3 != m3 and a3 != c3)
        self.assertEqual(t3.shape, (4,))

        m4, _, _, _, dir4 = quadriga_lib.sos.calc_mse(self.f3, self.other, self.dist, 10)
        self.assertEqual(m4, m3)
        npt.assert_array_equal(dir4, dir3)

        m5, _, _, _, dir5 = quadriga_lib.sos.calc_mse(self.f3, self.other, self.dist, 7)
        self.assertEqual(dir5.shape, (7, 2))
        self.assertGreater(abs(m5 - m3), 1e-6)

        # 2-D: one column of azimuth angles
        dir6 = quadriga_lib.sos.calc_mse(self.f3[:, :2], self.other, self.dist, np.int32(4))[4]
        self.assertEqual(dir6.shape, (4, 1))
        npt.assert_allclose(dir6[:, 0], np.pi * np.arange(4) / 4.0, atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.sos.calc_mse(self.f3, self.other, self.dist)
        variants = [
            (np.ascontiguousarray(self.f3), self.other, self.dist),                                     # C order
            (np.asfortranarray(self.f3), self.other.reshape(-1, 1), self.dist.reshape(-1, 1)),          # Fortran, columns
            (np.repeat(self.f3, 2, axis=0)[::2, :], np.repeat(self.other, 2)[::2], np.repeat(self.dist, 2)[::2]),
            (self.f3.tolist(), self.other.tolist(), self.dist.tolist()),                                # plain lists
        ]
        for f, a, d in variants:
            out = quadriga_lib.sos.calc_mse(f, a, d)
            self.assertEqual(out[:3], ref[:3])
            npt.assert_array_equal(out[3], ref[3])
            npt.assert_array_equal(out[4], ref[4])

        # float32 input: values in dB change slightly
        out = quadriga_lib.sos.calc_mse(self.f3.astype(np.float32), self.other.astype(np.float32),
                                        self.dist.astype(np.float32))
        self.assertIsInstance(out[0], float)
        npt.assert_allclose(out[:3], ref[:3], atol=1e-3, rtol=0)
        self.assertEqual(out[3].dtype, np.float64)

    def test_inputs_not_modified(self):
        args = [self.f3, self.other, self.dist]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.calc_mse(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.calc_mse(self.freq, self.acf, self.dist[:5])
        self.assertIn("dist", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.calc_mse(self.freq, self.acf, self.dist, 0)
        self.assertIn("n_dir", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.calc_mse(self.freq, self.acf)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.calc_mse(self.freq, self.acf, self.dist, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
