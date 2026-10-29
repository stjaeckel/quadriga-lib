# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_draw.
# Scope: argument and output handling, set type, floor IDs, flags, error paths.
# Physics validation lives in the Catch2 suite (test_ssf_phases.cpp).

import sys
import os
import unittest
from math import erfc, sqrt
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
        # UMi LOS file (N = 12, SC_lambda = 12 m), shared and per-frequency phases, 5 links
        self.sp = quadriga_lib.gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
        self.N = int(self.sp["NumClusters"]) - 1
        self.phase = quadriga_lib.gbsm.ssf_init_phases(self.sp, 7)
        self.phase_f = quadriga_lib.gbsm.ssf_init_phases(self.sp, 7, 0)
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5],
                                [3.0, 4.0, 10.0]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.rnd = quadriga_lib.gbsm.ssf_draw(self.sp, self.phase, self.tx_pos, self.rx_pos)
        self.fl = np.array([0, 1, 0, 2, 1])

    def test_basic_call(self):
        rnd, N = self.rnd, self.N
        self.assertIsInstance(rnd, np.ndarray)
        self.assertEqual(rnd.shape, (5, 9 * N))
        self.assertEqual(rnd.dtype, np.float64)
        self.assertTrue(np.all((rnd[:, :N] > 0) & (rnd[:, :N] < 1)))  # Uniform delay block
        self.assertTrue(np.max(np.abs(rnd[:, N:])) < 6 and np.std(rnd) > 0.1)

        # Column N (sign of AOD, cluster 0) is the field of slice 0 at the Rx positions projected on the x-y plane,
        # column 0 its uniform map; ssf_draw may evaluate sin/cos in single precision (AVX2 kernel)
        freq = quadriga_lib.sos.preset("Comb256", self.sp["SC_lambda"])[0]
        pos_xy = np.column_stack([self.rx_pos[:, :2], np.zeros(5)])
        ref = quadriga_lib.sos.val(freq, self.phase[:, N, 0], pos_xy).ravel()
        npt.assert_allclose(rnd[:, N], ref, atol=1e-5, rtol=0)
        ref = quadriga_lib.sos.val(freq, self.phase[:, 0, 0], pos_xy).ravel()
        npt.assert_allclose(rnd[:, 0], [0.5 * erfc(-x / sqrt(2)) for x in ref], atol=1e-5, rtol=0)

        # Single mobility: the values do not depend on tx_pos
        npt.assert_array_equal(quadriga_lib.gbsm.ssf_draw(self.sp, self.phase, np.zeros((1, 3)), self.rx_pos), rnd)

    def test_keyword_arguments(self):
        rk = quadriga_lib.gbsm.ssf_draw(scenpar=self.sp, phase=self.phase_f, tx_pos=self.tx_pos, rx_pos=self.rx_pos,
                                        set_id=0, floor_id=self.fl, dual_mobility=False, preset="Comb256",
                                        horizontal=True)
        rp = quadriga_lib.gbsm.ssf_draw(self.sp, self.phase_f, self.tx_pos, self.rx_pos, 0, self.fl, False,
                                        "Comb256", True)
        npt.assert_array_equal(rk, rp)
        self.assertEqual(rk.shape, (5, self.N))

    def test_set_id(self):
        f = quadriga_lib.gbsm.ssf_draw
        args = (self.sp, self.phase_f, self.tx_pos, self.rx_pos)
        rf = f(*args, 0)
        self.assertEqual(rf.shape, (5, self.N))
        self.assertTrue(np.any(rf < 0))  # Normal shadowing values
        npt.assert_array_equal(f(*args, 3), rf)  # same set type
        npt.assert_array_equal(f(self.sp, self.phase, self.tx_pos, self.rx_pos, -1), self.rnd)

    def test_floor_id(self):
        f = quadriga_lib.gbsm.ssf_draw
        args = (self.sp, self.phase, self.tx_pos, self.rx_pos, -1)
        npt.assert_array_equal(f(*args, None), self.rnd)
        npt.assert_array_equal(f(*args, np.zeros(5)), self.rnd)
        rl = f(*args, self.fl)
        npt.assert_array_equal(rl[[0, 2]], self.rnd[[0, 2]])  # floor 0 unchanged
        self.assertTrue(np.all(np.max(np.abs(rl[[1, 3, 4]] - self.rnd[[1, 3, 4]]), axis=1) > 1e-3))
        npt.assert_array_equal(f(*args, self.fl.astype(np.float32)), rl)
        npt.assert_array_equal(f(*args, list(self.fl)), rl)

    def test_flags(self):
        f = quadriga_lib.gbsm.ssf_draw
        tx_d = np.array([[5.0, 5.0, 1.5], [30.0, 0.0, 1.5], [40.0, -20.0, 1.5], [60.0, 10.0, 1.5], [0.0, 9.0, 1.5]])
        rd = f(self.sp, self.phase, tx_d, self.rx_pos, -1, None, True)
        self.assertTrue(np.max(np.abs(rd - self.rnd)) > 1e-3)
        blk = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
        swap = (blk[:, None] * self.N + np.arange(self.N)[None, :]).ravel()
        npt.assert_array_equal(f(self.sp, self.phase, self.rx_pos, tx_d, -1, None, True), rd[:, swap])
        npt.assert_array_equal(f(self.sp, self.phase, tx_d, self.rx_pos, -1, None, None), self.rnd)
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] += np.array([0.0, 5.0, 10.0, 2.0, 7.0])
        npt.assert_array_equal(f(self.sp, self.phase, self.tx_pos, rx_z), self.rnd)  # 2D: z not used
        r3 = f(self.sp, self.phase, self.tx_pos, self.rx_pos, horizontal=False)
        r3z = f(self.sp, self.phase, self.tx_pos, rx_z, horizontal=False)
        self.assertTrue(np.max(np.abs(r3z - r3)) > 1e-3)
        npt.assert_array_equal(f(self.sp, self.phase, self.tx_pos, self.rx_pos, horizontal=None), self.rnd)
        with self.assertRaises(TypeError) as ctx:
            f(self.sp, self.phase, self.tx_pos, self.rx_pos, dual_mobility=2.5)
        self.assertIn("dual_mobility", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.sp, self.phase, self.tx_pos, self.rx_pos, horizontal=1)

    def test_preset_and_scenpar(self):
        f = quadriga_lib.gbsm.ssf_draw
        pc = quadriga_lib.gbsm.ssf_init_phases(self.sp, 7, -1, "Comb1024")
        rc = f(self.sp, pc, self.tx_pos, self.rx_pos, preset="Comb1024")
        self.assertEqual(rc.shape, (5, 9 * self.N))
        self.assertTrue(np.max(np.abs(rc - self.rnd)) > 1e-3)
        sp0 = {"NumClusters": 1, "NumClusters_min": 1}  # N = 0: empty set, any empty phase
        self.assertEqual(f(sp0, np.zeros((0, 0, 0)), self.tx_pos, self.rx_pos).shape, (5, 0))

    def test_zero_links(self):
        r0 = quadriga_lib.gbsm.ssf_draw(self.sp, self.phase, self.tx_pos, np.zeros((0, 3)))
        self.assertEqual(r0.shape, (0, 9 * self.N))

    def test_memory_layout_and_dtype(self):
        f = quadriga_lib.gbsm.ssf_draw
        npt.assert_array_equal(f(self.sp, np.asfortranarray(self.phase), np.asfortranarray(self.tx_pos),
                                 np.asfortranarray(self.rx_pos)), self.rnd)
        big = np.zeros((10, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.sp, self.phase, self.tx_pos, big[::2, ::2]), self.rnd)  # strided view
        npt.assert_array_equal(f(self.sp, self.phase, self.tx_pos.astype(np.float32),
                                 self.rx_pos.astype(np.float32)), self.rnd)  # positions exact in float32
        rx0, ph0 = self.rx_pos.copy(), self.phase.copy()
        f(self.sp, self.phase, self.tx_pos, self.rx_pos, -1, self.fl)
        npt.assert_array_equal(self.rx_pos, rx0)
        npt.assert_array_equal(self.phase, ph0)

    def test_errors(self):
        f = quadriga_lib.gbsm.ssf_draw
        args = (self.sp, self.phase, self.tx_pos, self.rx_pos)
        with self.assertRaises(ValueError) as ctx:
            f(*args, -1, np.array([0, -1, 0, 0, 0]))
        self.assertIn("Entries in 'floor_id' must be non-negative integers below 2^53.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(*args, -2)
        self.assertIn("Input 'set_id' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, self.phase_f, self.tx_pos, self.rx_pos)
        self.assertIn("Input 'phase' must have size [256, 108, 2] for preset 'Comb256' and this set.",
                      str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, self.phase, self.tx_pos[:, :2], self.rx_pos)
        self.assertIn("Input 'tx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.sp, self.phase, self.tx_pos)
        with self.assertRaises(TypeError):
            f(*args, unknown=1)
        with self.assertRaises(TypeError):
            f(self.sp, None, self.tx_pos, self.rx_pos)


if __name__ == "__main__":
    unittest.main()
