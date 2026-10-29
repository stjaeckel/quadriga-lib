# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_init_phases.
# Scope: argument and output handling, seed rule, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_ssf_phases.cpp).

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

CONF_DIR = os.path.join(current_dir, "../../data/config")


class test_case(unittest.TestCase):

    def setUp(self):
        # UMi LOS file, N = 12 NLOS clusters
        self.sp = quadriga_lib.gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
        self.N = int(self.sp["NumClusters"]) - 1
        blk = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
        self.mirror = (blk[:, None] * self.N + np.arange(self.N)[None, :]).ravel()
        self.phase = quadriga_lib.gbsm.ssf_init_phases(self.sp, 42)

    def test_basic_call(self):
        phase = self.phase
        self.assertIsInstance(phase, np.ndarray)
        self.assertEqual(phase.shape, (256, 9 * self.N, 2))
        self.assertEqual(phase.dtype, np.float64)
        self.assertTrue(np.all((phase >= -np.pi) & (phase < np.pi)))
        npt.assert_array_equal(phase[:, :, 1], phase[:, self.mirror, 0])  # slice 1 mirrors slice 0
        npt.assert_array_equal(quadriga_lib.gbsm.ssf_init_phases(self.sp, 42), phase)

    def test_seed_rule(self):
        # Seeds of the sets: SplitMix64 of the large-scale seed and the set tag (test vectors of the seed rule)
        f = quadriga_lib.gbsm.ssf_init_phases
        npt.assert_array_equal(self.phase[:, :, 0],
                               quadriga_lib.sos.init_phases(256, 9 * self.N, 8766244108781592946))
        npt.assert_array_equal(f(self.sp, 42, 0)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, self.N, 4119046106699552547))
        npt.assert_array_equal(f(self.sp, 42, 1)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, self.N, 9018399064009245349))
        npt.assert_array_equal(f(self.sp, 0)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, 9 * self.N, 313702574736366215))

    def test_keyword_arguments(self):
        pk = quadriga_lib.gbsm.ssf_init_phases(scenpar=self.sp, seed=42, set_id=-1, preset="Comb256")
        npt.assert_array_equal(pk, self.phase)

    def test_optional_inputs(self):
        f = quadriga_lib.gbsm.ssf_init_phases
        pr = f(self.sp)  # seed -1: system random device
        self.assertEqual(pr.shape, self.phase.shape)
        self.assertTrue(np.max(np.abs(pr - self.phase)) > 1)
        npt.assert_array_equal(f(self.sp, 42, -1), self.phase)
        f0 = f(self.sp, 42, 0)
        self.assertEqual(f0.shape, (256, self.N, 2))
        npt.assert_array_equal(f0[:, :, 1], f0[:, :, 0])  # shadowing mirrors itself
        self.assertTrue(np.max(np.abs(f(self.sp, 42, 1) - f0)) > 1)
        pc = f(self.sp, 42, -1, "Comb1024")
        self.assertEqual(pc.shape, (1024, 9 * self.N, 2))

    def test_scenpar(self):
        f = quadriga_lib.gbsm.ssf_init_phases
        self.assertEqual(f({"NumClusters": 5, "NumClusters_min": 1}, 42).shape, (256, 36, 2))
        p0 = f({"NumClusters": 1, "NumClusters_min": 1}, 42, 0)  # N = 0: empty set
        self.assertEqual(p0.shape, (256, 0, 2))
        self.assertEqual(f(None, 42).shape[1] % 9, 0)  # all defaults

    def test_errors(self):
        f = quadriga_lib.gbsm.ssf_init_phases
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, -2)
        self.assertIn("Input 'seed' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, 42, -2)
        self.assertIn("Input 'set_id' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, 42, -1, "Unknown")
        self.assertIn("preset", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f({"NumClusters": 0}, 42)
        self.assertIn("NumClusters", str(ctx.exception))
        with self.assertRaises(TypeError):
            f()
        with self.assertRaises(TypeError):
            f(self.sp, seed=42, unknown=1)
        with self.assertRaises(TypeError):
            f(self.sp, 1.5)


if __name__ == "__main__":
    unittest.main()
