# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.generate.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_generate.cpp).

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
        # 1-D target made of two sinusoids, which the optimiser must recover
        self.dist = np.arange(0.0, 50.5, 0.5)
        self.acf = 0.5 * (np.cos(2 * np.pi * 0.04 * self.dist) + np.cos(2 * np.pi * 0.13 * self.dist))

        # Exponential target for the 2-D and 3-D cases
        self.dist_e = np.arange(0.0, 41.0)
        self.expo = np.exp(-self.dist_e / 10.0)

    def test_basic_call(self):
        out = quadriga_lib.sos.generate(self.acf, self.dist, 2, 1, True, 10, True, 0.0, 0.0, 3)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        freq, ase, test_dir = out

        self.assertEqual(freq.shape, (2, 1))  # (n_coeff, n_dim)
        self.assertEqual(freq.dtype, np.float64)
        npt.assert_allclose(np.sort(np.abs(freq[:, 0])), [0.04, 0.13], atol=1e-5, rtol=0)  # limited by the search step

        self.assertEqual(ase.ndim, 1)
        self.assertGreaterEqual(ase.size, 1)
        self.assertEqual(ase.dtype, np.float64)
        self.assertLess(ase[-1], 1e-6)

        self.assertIsInstance(test_dir, np.ndarray)
        self.assertEqual(test_dir.shape, (0, 0))  # no test directions in 1-D

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, False, 7, False, 0.03, 0.2, 5)
        out = quadriga_lib.sos.generate(acf=self.expo, dist=self.dist_e, n_coeff=12, n_dim=2, uniform_smp=False,
                                        n_dir=7, random_init=False, freq_min=0.03, freq_max=0.2, seed=5)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_defaults(self):
        # n_dim = 3, n_dir = 10, uniform_smp = True, random seed
        freq, ase, test_dir = quadriga_lib.sos.generate(self.expo, self.dist_e, 12)
        self.assertEqual(freq.shape, (12, 3))
        self.assertEqual(test_dir.shape, (10, 2))
        self.assertEqual(test_dir.dtype, np.float64)

        # The remaining defaults, with a seed: random_init = True, freq_min = 0, freq_max = 0
        ref = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, seed=5)
        out = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 3, True, 10, True, 0.0, 0.0, 5)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        # n_dim and n_dir
        f4, a4, d4 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, True, 7, True, 0.0, 0.0, 5)
        self.assertEqual(f4.shape, (12, 2))
        self.assertEqual(d4.shape, (7, 1))
        npt.assert_allclose(d4[:, 0], np.pi * np.arange(7) / 7.0, atol=1e-14, rtol=0)
        self.assertTrue(np.all(np.diff(a4) <= 0.0))

        # uniform_smp = True: regular directions of the sinusoids; False: random directions
        az = np.pi * np.arange(12) / 12.0
        npt.assert_allclose(f4[:, 1] * np.cos(az) - f4[:, 0] * np.sin(az), np.zeros(12), atol=1e-14, rtol=0)
        f5 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, uniform_smp=False, n_dir=7, seed=5)[0]
        self.assertGreater(np.max(np.abs(f5[:, 1] * np.cos(az) - f5[:, 0] * np.sin(az))), 1e-3)

        # random_init
        f6 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, n_dir=7, random_init=False, seed=5)[0]
        self.assertGreater(np.max(np.abs(f6 - f4)), 1e-6)

        # freq_min and freq_max limit the magnitude
        f7 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, n_dir=7, freq_min=0.03, freq_max=0.2, seed=5)[0]
        mag = np.sqrt(np.sum(np.square(f7), axis=1))
        self.assertTrue(np.all(mag >= 0.03 - 1e-12) and np.all(mag <= 0.2 + 1e-12))
        self.assertLess(np.min(np.sqrt(np.sum(np.square(f4), axis=1))), 0.03)

        # freq_max alone
        f8 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, n_dir=7, freq_max=0.05, seed=5)[0]
        self.assertLessEqual(np.max(np.sqrt(np.sum(np.square(f8), axis=1))), 0.05 + 1e-12)
        self.assertGreater(np.max(np.sqrt(np.sum(np.square(f4), axis=1))), 0.05)

        # seed: repeatable, and another seed gives another table
        f9 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, n_dir=7, seed=5)[0]
        npt.assert_array_equal(f9, f4)
        f10 = quadriga_lib.sos.generate(self.expo, self.dist_e, 12, 2, n_dir=7, seed=6)[0]
        self.assertGreater(np.max(np.abs(f10 - f4)), 1e-6)

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.sos.generate(self.acf, self.dist, 2, 1, seed=3)
        variants = [
            lambda a: np.repeat(a, 2)[::2],     # non-contiguous view
            lambda a: a.reshape(-1, 1),         # column
            lambda a: a.tolist(),               # plain list
        ]
        for fn in variants:
            out = quadriga_lib.sos.generate(fn(self.acf), fn(self.dist), np.uint32(2), np.int32(1), np.bool_(True),
                                            np.int64(10), 1, 0, 0, np.int32(3))
            for a, b in zip(out, ref):
                self.assertEqual(a.dtype, np.float64)
                npt.assert_array_equal(a, b)

        # float32 target: the same two frequencies within the search step
        freq = quadriga_lib.sos.generate(self.acf.astype(np.float32), self.dist.astype(np.float32), 2, 1, seed=3)[0]
        self.assertEqual(freq.dtype, np.float64)
        npt.assert_allclose(np.sort(np.abs(freq[:, 0])), [0.04, 0.13], atol=1e-5, rtol=0)

    def test_inputs_not_modified(self):
        acf, dist = self.acf.copy(), self.dist.copy()
        quadriga_lib.sos.generate(acf, dist, 2, 1, seed=3)
        npt.assert_array_equal(acf, self.acf)
        npt.assert_array_equal(dist, self.dist)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.generate(self.acf, self.dist, 0)
        self.assertIn("n_coeff", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.generate(self.acf, self.dist, 2, 4)
        self.assertIn("n_dim", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.generate(self.acf, self.dist[:5], 2)
        self.assertIn("dist", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.generate(self.acf, self.dist, 2, seed=-2)
        self.assertIn("seed", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.generate(self.acf, self.dist)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.generate(self.acf, self.dist, 2, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
