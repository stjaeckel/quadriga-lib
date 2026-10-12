# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.acf_estimate.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_acf_estimate.cpp).

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
        self.freq, self.acf, self.dist = quadriga_lib.sos.preset("Gauss256")
        self.phase = quadriga_lib.sos.init_phases(256, 1, 3)[:, 0]
        self.bins = np.array([2.0, 5.0, 10.0, 15.0, 22.0, 30.0])

    def test_basic_call(self):
        # Without the ACF table: single-device estimate only, the third output is None
        out = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, seed=1)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        acf_est, dist_est, acf_est_dual = out

        self.assertEqual(acf_est.shape, (6,))
        self.assertEqual(dist_est.shape, (6,))
        self.assertEqual(acf_est.dtype, np.float64)
        self.assertEqual(dist_est.dtype, np.float64)
        self.assertIsNone(acf_est_dual)

        self.assertTrue(np.all(np.isfinite(acf_est)) and np.all(np.abs(acf_est) <= 1.0))
        self.assertGreater(acf_est[0], 0.7)       # 2 m with a Gaussian ACF of 10 m
        self.assertLess(abs(acf_est[5]), 0.4)     # 30 m

        # The mean distances lie inside their bins, so the outputs are not swapped
        lower = np.array([0.0, 3.5, 7.5, 12.5, 18.5, 26.0])
        upper = np.array([3.5, 7.5, 12.5, 18.5, 26.0, 32.8])
        self.assertTrue(np.all(dist_est > lower) and np.all(dist_est <= upper))

    def test_dual_output(self):
        ref = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, seed=1)

        # With the ACF table all three outputs are arrays; the first two do not change
        acf_est, dist_est, acf_est_dual = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150,
                                                                        self.acf, self.dist, 1)
        npt.assert_array_equal(acf_est, ref[0])
        npt.assert_array_equal(dist_est, ref[1])
        self.assertIsInstance(acf_est_dual, np.ndarray)
        self.assertEqual(acf_est_dual.shape, (6,))
        self.assertEqual(acf_est_dual.dtype, np.float64)

        # Bins with fewer than 6 pairs are NaN, which happens for the smallest distances with two devices
        valid = np.isfinite(acf_est_dual)
        self.assertGreaterEqual(np.sum(valid), 4)
        self.assertTrue(np.all(np.abs(acf_est_dual[valid]) <= 1.0))
        self.assertGreater(np.max(np.abs(acf_est_dual[valid] - acf_est[valid])), 1e-4)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, self.acf, self.dist, 1)
        out = quadriga_lib.sos.acf_estimate(freq=self.freq, phase=self.phase, bins=self.bins, n_pos=150,
                                            acf=self.acf, dist=self.dist, seed=1)
        self.assertEqual(len(out), 3)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, seed=1)

        # None and empty arrays are the same as omitted
        for out in (quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, None, None, 1),
                    quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, [], np.zeros(0), 1)):
            npt.assert_array_equal(out[0], ref[0])
            npt.assert_array_equal(out[1], ref[1])
            self.assertIsNone(out[2])

        # Another seed: other positions
        a3 = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, seed=2)[0]
        self.assertGreater(np.max(np.abs(a3 - ref[0])), 1e-4)

        # seed = -1 (default)
        a4 = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150)[0]
        self.assertEqual(a4.shape, (6,))

        # n_pos reaches the library
        a5 = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 20, seed=1)[0]
        self.assertTrue(np.max(np.abs(a5 - ref[0])) > 1e-4 or np.any(np.isnan(a5)))

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, self.acf, self.dist, 1)
        mats = [
            lambda a: np.ascontiguousarray(a),            # C order
            lambda a: np.asfortranarray(a),               # Fortran order
            lambda a: np.repeat(a, 2, axis=0)[::2, :],    # non-contiguous view
            lambda a: a.tolist(),                         # plain lists
        ]
        vecs = [
            lambda a: a,
            lambda a: a.reshape(-1, 1),                   # column, as returned by sos.init_phases
            lambda a: np.repeat(a, 2)[::2],               # non-contiguous view
            lambda a: a.tolist(),
        ]
        for fm, fv in zip(mats, vecs):
            out = quadriga_lib.sos.acf_estimate(fm(self.freq), fv(self.phase), fv(self.bins), np.uint32(150),
                                                fv(self.acf), fv(self.dist), np.int32(1))
            for a, b in zip(out, ref):
                self.assertEqual(a.dtype, np.float64)
                npt.assert_array_equal(a, b)

        # float32 frequencies and phases, integer bins
        out = quadriga_lib.sos.acf_estimate(self.freq.astype(np.float32), self.phase.astype(np.float32),
                                            self.bins.astype(np.int32), 150, seed=1)
        self.assertEqual(out[0].dtype, np.float64)
        npt.assert_allclose(out[0], ref[0], atol=1e-3, rtol=0)

    def test_inputs_not_modified(self):
        args = [self.freq, self.phase, self.bins, self.acf, self.dist]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, self.acf, self.dist, 1)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_estimate(self.freq[:, :2], self.phase, self.bins, 150)
        self.assertIn("freq", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 1)
        self.assertIn("n_pos", str(ctx.exception))

        # Only one of acf and dist: the two-device estimate is requested and C++ rejects the call
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, self.acf)
        self.assertIn("acf_est_dual", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, dist=self.dist)
        self.assertIn("acf_est_dual", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acf_estimate(self.freq, self.phase, self.bins, 150, no_such_argument=1)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.freq, self.phase, self.bins, 20]
        quadriga_lib.sos.acf_estimate(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.acf_estimate(*bad)


if __name__ == "__main__":
    unittest.main()
