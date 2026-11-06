# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.draw.
# Scope: argument and output handling, set types and set index, floor IDs, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_phases_lsf.cpp, _cluster.cpp, _ray.cpp).

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
SWAP = [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10]  # "lsf": ASD <-> ASA, ESD <-> ESA

class test_lsf(unittest.TestCase):

    def setUp(self):
        # Phases of a scenario with distinct decorrelation distances, 5 links
        self.scenpar = {"DS_lambda": 10.0, "AS_D_lambda": 18.0, "AS_A_lambda": 15.0,
                        "ES_D_lambda": 12.0, "ES_A_lambda": 8.0}
        self.phase = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7)
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5],
                                [3.0, 4.0, 10.0]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.rnd = quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos)
        self.fl = np.array([0, 1, 0, 2, 1])
        self.rf = quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos, floor_id=self.fl)

    def test_basic_call(self):
        rnd = self.rnd
        self.assertIsInstance(rnd, np.ndarray)
        self.assertEqual(rnd.shape, (5, 11))
        self.assertEqual(rnd.dtype, np.float64)
        self.assertTrue(np.all((rnd[:, 9:] > 0) & (rnd[:, 9:] < 1)))  # Uniform columns
        self.assertTrue(np.max(np.abs(rnd[:, :9])) < 6 and np.std(rnd) > 0.1)

        # Column 0 is the DS field (10 m) of slice 0 at the Rx positions projected on the x-y plane;
        # gbsm.draw may evaluate sin/cos in single precision (AVX2 kernel)
        freq = quadriga_lib.sos.preset("Comb256", 10.0)[0]
        pos_xy = np.column_stack([self.rx_pos[:, :2], np.zeros(5)])
        ref = quadriga_lib.sos.val(freq, self.phase[:, 0, 0], pos_xy)
        npt.assert_allclose(rnd[:, 0], ref.ravel(), atol=1e-5, rtol=0)

        # Single mobility: the values do not depend on tx_pos; one link alone gives its row
        npt.assert_array_equal(quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, np.zeros((1, 3)), self.rx_pos),
                               rnd)
        r3 = quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos[2:3])
        npt.assert_allclose(r3, rnd[2:3], atol=1e-12, rtol=0)

    def test_keyword_arguments(self):
        rk = quadriga_lib.gbsm.draw(scenpar=self.scenpar, set_type="lsf", phase=self.phase, tx_pos=self.tx_pos,
                                    rx_pos=self.rx_pos, preset="Comb256", floor_id=self.fl, dual_mobility=False,
                                    horizontal=True)
        npt.assert_array_equal(rk, self.rf)
        rp = quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos, preset="Comb256",
                                    floor_id=self.fl, dual_mobility=False, horizontal=True)
        npt.assert_array_equal(rp, self.rf)

    def test_floor_id(self):
        f = quadriga_lib.gbsm.draw
        args = (self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(*args, floor_id=None), self.rnd)
        npt.assert_array_equal(f(*args, floor_id=np.zeros(5)), self.rnd)
        npt.assert_array_equal(f(*args, floor_id=[]), self.rnd)  # empty = all on floor 0
        rf = self.rf
        npt.assert_array_equal(rf[[0, 2]], self.rnd[[0, 2]])  # floor 0 unchanged
        self.assertTrue(np.all(np.max(np.abs(rf[[1, 3, 4]] - self.rnd[[1, 3, 4]]), axis=1) > 1e-3))

        # Floor IDs start at 0 (no index shift): floor 1 adds the offsets 2·pi·n/256 to the phases
        phase_1 = self.phase + (2 * np.pi * np.arange(256) / 256)[:, None, None]
        r1 = f(self.scenpar, "lsf", phase_1, self.tx_pos, self.rx_pos[[1, 4]])
        npt.assert_allclose(r1, rf[[1, 4]], atol=1e-10, rtol=0)

        # Integer and float types of floor_id
        npt.assert_array_equal(f(*args, floor_id=self.fl.astype(np.float64)), rf)
        npt.assert_array_equal(f(*args, floor_id=self.fl.astype(np.uint8)), rf)
        npt.assert_array_equal(f(*args, floor_id=self.fl.astype(np.uint64)), rf)
        npt.assert_array_equal(f(*args, floor_id=self.fl.astype(np.float32)), rf)
        npt.assert_array_equal(f(*args, floor_id=self.fl.tolist()), rf)
        npt.assert_array_equal(f(*args, floor_id=self.fl.reshape(1, 5)), rf)

        # The largest ID is 2^32 - 1 (32-bit labels)
        big = np.array([2**32 - 1, 1, 0, 2, 1], dtype=np.int64)
        self.assertEqual(f(*args, floor_id=big).shape, (5, 11))

    def test_dual_mobility(self):
        f = quadriga_lib.gbsm.draw
        tx_d = np.array([[5.0, 5.0, 1.5], [30.0, 0.0, 1.5], [40.0, -20.0, 1.5], [60.0, 10.0, 1.5], [0.0, 9.0, 1.5]])
        rd = f(self.scenpar, "lsf", self.phase, tx_d, self.rx_pos, floor_id=None, dual_mobility=True)
        self.assertGreater(np.max(np.abs(rd - self.rnd)), 1e-3)
        rs = f(self.scenpar, "lsf", self.phase, self.rx_pos, tx_d, floor_id=None, dual_mobility=True)
        npt.assert_array_equal(rs, rd[:, SWAP])  # columns 3/4 and 5/6 swapped
        npt.assert_array_equal(f(self.scenpar, "lsf", self.phase, tx_d, self.rx_pos, dual_mobility=False), self.rnd)

    def test_flags(self):
        f = quadriga_lib.gbsm.draw
        args = (self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(*args, floor_id=None, dual_mobility=None, preset="Comb256", horizontal=None),
                               self.rnd)  # None selects the default
        npt.assert_array_equal(f(*args, dual_mobility=np.bool_(False), horizontal=np.bool_(True)), self.rnd)
        for name in ("dual_mobility", "horizontal"):
            for bad in (2.5, 0, "no"):
                with self.assertRaises(TypeError) as ctx:
                    f(*args, **{name: bad})
                self.assertIn(f"Input '{name}' must be a bool or None.", str(ctx.exception))

    def test_horizontal_and_preset(self):
        f = quadriga_lib.gbsm.draw
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] += [0.0, 5.0, 10.0, 2.0, 7.0]
        npt.assert_array_equal(f(self.scenpar, "lsf", self.phase, self.tx_pos, rx_z), self.rnd)  # 2D: z not used
        r3 = f(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos, horizontal=False)
        r3z = f(self.scenpar, "lsf", self.phase, self.tx_pos, rx_z, horizontal=False)
        self.assertGreater(np.max(np.abs(r3z - r3)), 1e-3)  # 3D: z is used
        phase_c = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7, preset="Comb1024")
        rc = f(self.scenpar, "lsf", phase_c, self.tx_pos, self.rx_pos, preset="Comb1024")
        self.assertEqual(rc.shape, (5, 11))
        self.assertGreater(np.max(np.abs(rc - self.rnd)), 1e-3)

    def test_scenpar(self):
        f = quadriga_lib.gbsm.draw
        sp2 = dict(self.scenpar)
        sp2["DS_lambda"] = 50.0
        r2 = f(sp2, "lsf", self.phase, self.tx_pos, self.rx_pos)
        self.assertGreater(np.max(np.abs(r2[:, 0] - self.rnd[:, 0])), 1e-3)
        npt.assert_array_equal(r2[:, 1:], self.rnd[:, 1:])
        sv = quadriga_lib.gbsm.scenpar_validate(self.scenpar)[0]  # complete dict
        npt.assert_array_equal(f(sv, "lsf", self.phase, self.tx_pos, self.rx_pos), self.rnd)
        rn = f(None, "lsf", self.phase, self.tx_pos, self.rx_pos)  # all defaults
        self.assertGreater(np.max(np.abs(rn[:, 0] - self.rnd[:, 0])), 1e-3)

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.draw
        npt.assert_array_equal(f(self.scenpar, "lsf", self.phase, self.tx_pos, np.asfortranarray(self.rx_pos)),
                               self.rnd)
        npt.assert_array_equal(f(self.scenpar, "lsf", self.phase, self.tx_pos, np.ascontiguousarray(self.rx_pos)),
                               self.rnd)
        big = np.zeros((10, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.scenpar, "lsf", self.phase, self.tx_pos, big[::2, ::2]), self.rnd)
        npt.assert_array_equal(f(self.scenpar, "lsf", np.ascontiguousarray(self.phase), self.tx_pos, self.rx_pos),
                               self.rnd)
        rt = f(self.scenpar, "lsf", self.phase, self.tx_pos.astype(np.float32), self.rx_pos.astype(np.float32))
        npt.assert_array_equal(rt, self.rnd)  # positions exact in float32
        rp = f(self.scenpar, "lsf", self.phase.astype(np.float32), self.tx_pos, self.rx_pos)
        npt.assert_allclose(rp, self.rnd, atol=1e-4, rtol=0)  # phases rounded to float32

    def test_zero_links(self):
        r0 = quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, np.zeros((0, 3)))
        self.assertEqual(r0.shape, (0, 11))
        self.assertEqual(r0.dtype, np.float64)

    def test_inputs_not_modified(self):
        rx, ph, fl = self.rx_pos.copy(), self.phase.copy(), self.fl.copy()
        quadriga_lib.gbsm.draw(self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos, floor_id=self.fl,
                               dual_mobility=True)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(self.phase, ph)
        npt.assert_array_equal(self.fl, fl)

    def test_errors(self):
        f = quadriga_lib.gbsm.draw
        args = (self.scenpar, "lsf", self.phase, self.tx_pos, self.rx_pos)
        for bad in ([0, -1, 0, 0, 0], [0, 1.5, 0, 0, 0], [0, np.nan, 0, 0, 0], [0, 2.0**32, 0, 0, 0],
                    np.array([0, 2**32, 0, 0, 0], dtype=np.int64)):  # wrapper
            with self.assertRaises(ValueError) as ctx:
                f(*args, floor_id=bad)
            self.assertIn("Entries in 'floor_id' must be non-negative integers below 2^32.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", self.phase[:, :10, :], self.tx_pos, self.rx_pos)
        self.assertIn("Input 'phase' must have size [256, 11, 2] for preset 'Comb256'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(*args, preset="Comb1024")
        self.assertIn("Input 'phase' must have size [1024, 11, 2] for preset 'Comb1024'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(*args, floor_id=[0, 1])
        self.assertIn("Number of elements in 'floor_id' does not match", str(ctx.exception))
        msg3d = "Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors."
        for fl in (self.fl, np.zeros(5)):  # C++: 3D mode, also with all IDs 0
            with self.assertRaises(ValueError) as ctx:
                f(*args, floor_id=fl, horizontal=False)
            self.assertIn(msg3d, str(ctx.exception))
        self.assertEqual(f(*args, floor_id=None, horizontal=False).shape, (5, 11))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", self.phase, self.tx_pos[:, :2], self.rx_pos)
        self.assertIn("Input 'tx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.scenpar, "lsf", self.phase, self.tx_pos)
        with self.assertRaises(TypeError):
            f(*args, foo=1)
        with self.assertRaises(TypeError):
            f(self.scenpar, "lsf", None, self.tx_pos, self.rx_pos)  # mandatory array as None


class test_cluster(unittest.TestCase):

    def setUp(self):
        # UMi LOS file (N = 12, SC_lambda = 12 m), shared and per-frequency phases, 5 links
        self.sp = quadriga_lib.gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
        self.N = int(self.sp["NumClusters"]) - 1
        self.phase = quadriga_lib.gbsm.init_phases(self.sp, "cluster", 7)
        self.phase_f = quadriga_lib.gbsm.init_phases(self.sp, "cluster", 7, 0)
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5],
                                [3.0, 4.0, 10.0]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.rnd = quadriga_lib.gbsm.draw(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos)
        self.fl = np.array([0, 1, 0, 2, 1])

    def test_basic_call(self):
        rnd, N = self.rnd, self.N
        self.assertIsInstance(rnd, np.ndarray)
        self.assertEqual(rnd.shape, (5, 9 * N))
        self.assertEqual(rnd.dtype, np.float64)
        self.assertTrue(np.all((rnd[:, :N] > 0) & (rnd[:, :N] < 1)))  # Uniform delay block
        self.assertTrue(np.max(np.abs(rnd[:, N:])) < 6 and np.std(rnd) > 0.1)

        # Column N (sign of AOD, cluster 0) is the field of slice 0 at the Rx positions projected on the x-y plane,
        # column 0 its uniform map; gbsm.draw may evaluate sin/cos in single precision (AVX2 kernel)
        freq = quadriga_lib.sos.preset("Comb256", self.sp["SC_lambda"])[0]
        pos_xy = np.column_stack([self.rx_pos[:, :2], np.zeros(5)])
        ref = quadriga_lib.sos.val(freq, self.phase[:, N, 0], pos_xy).ravel()
        npt.assert_allclose(rnd[:, N], ref, atol=1e-5, rtol=0)
        ref = quadriga_lib.sos.val(freq, self.phase[:, 0, 0], pos_xy).ravel()
        npt.assert_allclose(rnd[:, 0], [0.5 * erfc(-x / sqrt(2)) for x in ref], atol=1e-5, rtol=0)

        # Single mobility: the values do not depend on tx_pos
        npt.assert_array_equal(quadriga_lib.gbsm.draw(self.sp, "cluster", self.phase, np.zeros((1, 3)), self.rx_pos),
                               rnd)

    def test_keyword_arguments(self):
        rk = quadriga_lib.gbsm.draw(scenpar=self.sp, set_type="cluster", phase=self.phase_f, tx_pos=self.tx_pos,
                                    rx_pos=self.rx_pos, set_id=0, preset="Comb256", floor_id=self.fl,
                                    dual_mobility=False, horizontal=True)
        rp = quadriga_lib.gbsm.draw(self.sp, "cluster", self.phase_f, self.tx_pos, self.rx_pos, 0, "Comb256", self.fl,
                                    dual_mobility=False, horizontal=True)
        npt.assert_array_equal(rk, rp)
        self.assertEqual(rk.shape, (5, self.N))

    def test_set_id(self):
        f = quadriga_lib.gbsm.draw
        args = (self.sp, "cluster", self.phase_f, self.tx_pos, self.rx_pos)
        rf = f(*args, set_id=0)
        self.assertEqual(rf.shape, (5, self.N))
        self.assertTrue(np.any(rf < 0))  # Normal shadowing values
        npt.assert_array_equal(f(*args, set_id=3), rf)  # same set type
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, -1), self.rnd)

    def test_floor_id(self):
        f = quadriga_lib.gbsm.draw
        args = (self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, -1)
        npt.assert_array_equal(f(*args, floor_id=None), self.rnd)
        npt.assert_array_equal(f(*args, floor_id=np.zeros(5)), self.rnd)
        rl = f(*args, floor_id=self.fl)
        npt.assert_array_equal(rl[[0, 2]], self.rnd[[0, 2]])  # floor 0 unchanged
        self.assertTrue(np.all(np.max(np.abs(rl[[1, 3, 4]] - self.rnd[[1, 3, 4]]), axis=1) > 1e-3))
        npt.assert_array_equal(f(*args, floor_id=self.fl.astype(np.float32)), rl)
        npt.assert_array_equal(f(*args, floor_id=list(self.fl)), rl)

    def test_flags(self):
        f = quadriga_lib.gbsm.draw
        tx_d = np.array([[5.0, 5.0, 1.5], [30.0, 0.0, 1.5], [40.0, -20.0, 1.5], [60.0, 10.0, 1.5], [0.0, 9.0, 1.5]])
        rd = f(self.sp, "cluster", self.phase, tx_d, self.rx_pos, -1, floor_id=None, dual_mobility=True)
        self.assertTrue(np.max(np.abs(rd - self.rnd)) > 1e-3)
        blk = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
        swap = (blk[:, None] * self.N + np.arange(self.N)[None, :]).ravel()
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.rx_pos, tx_d, -1, floor_id=None,
                                 dual_mobility=True), rd[:, swap])
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, tx_d, self.rx_pos, -1, floor_id=None,
                                 dual_mobility=None), self.rnd)
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] += np.array([0.0, 5.0, 10.0, 2.0, 7.0])
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.tx_pos, rx_z), self.rnd)  # 2D: z not used
        r3 = f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, horizontal=False)
        r3z = f(self.sp, "cluster", self.phase, self.tx_pos, rx_z, horizontal=False)
        self.assertTrue(np.max(np.abs(r3z - r3)) > 1e-3)
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, horizontal=None), self.rnd)
        with self.assertRaises(TypeError) as ctx:
            f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, dual_mobility=2.5)
        self.assertIn("dual_mobility", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, horizontal=1)

    def test_preset_and_scenpar(self):
        f = quadriga_lib.gbsm.draw
        pc = quadriga_lib.gbsm.init_phases(self.sp, "cluster", 7, -1, "Comb1024")
        rc = f(self.sp, "cluster", pc, self.tx_pos, self.rx_pos, preset="Comb1024")
        self.assertEqual(rc.shape, (5, 9 * self.N))
        self.assertTrue(np.max(np.abs(rc - self.rnd)) > 1e-3)
        sp0 = {"NumClusters": 1, "NumClusters_min": 1}  # N = 0: empty set, any empty phase
        self.assertEqual(f(sp0, "cluster", np.zeros((0, 0, 0)), self.tx_pos, self.rx_pos).shape, (5, 0))

    def test_zero_links(self):
        r0 = quadriga_lib.gbsm.draw(self.sp, "cluster", self.phase, self.tx_pos, np.zeros((0, 3)))
        self.assertEqual(r0.shape, (0, 9 * self.N))

    def test_memory_layout_and_dtype(self):
        f = quadriga_lib.gbsm.draw
        npt.assert_array_equal(f(self.sp, "cluster", np.asfortranarray(self.phase), np.asfortranarray(self.tx_pos),
                                 np.asfortranarray(self.rx_pos)), self.rnd)
        big = np.zeros((10, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.tx_pos, big[::2, ::2]), self.rnd)  # strided view
        npt.assert_array_equal(f(self.sp, "cluster", self.phase, self.tx_pos.astype(np.float32),
                                 self.rx_pos.astype(np.float32)), self.rnd)  # positions exact in float32
        rx0, ph0 = self.rx_pos.copy(), self.phase.copy()
        f(self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos, -1, floor_id=self.fl)
        npt.assert_array_equal(self.rx_pos, rx0)
        npt.assert_array_equal(self.phase, ph0)

    def test_errors(self):
        f = quadriga_lib.gbsm.draw
        args = (self.sp, "cluster", self.phase, self.tx_pos, self.rx_pos)
        with self.assertRaises(ValueError) as ctx:
            f(*args, set_id=-1, floor_id=np.array([0, -1, 0, 0, 0]))
        self.assertIn("Entries in 'floor_id' must be non-negative integers below 2^32.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(*args, set_id=-2)
        self.assertIn("Input 'set_id' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", self.phase_f, self.tx_pos, self.rx_pos)
        self.assertIn("Input 'phase' must have size [256, 108, 2] for preset 'Comb256' and this set.",
                      str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", self.phase, self.tx_pos[:, :2], self.rx_pos)
        self.assertIn("Input 'tx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.sp, "cluster", self.phase, self.tx_pos)
        with self.assertRaises(TypeError):
            f(*args, unknown=1)
        with self.assertRaises(TypeError):
            f(self.sp, "cluster", None, self.tx_pos, self.rx_pos)


class test_set_types(unittest.TestCase):

    def setUp(self):
        self.sp = {"DS_lambda": 10.0}
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])

    def test_set_type_set_id_and_center_freq(self):
        f = quadriga_lib.gbsm.draw
        ph = quadriga_lib.gbsm.init_phases(self.sp, "lsf", 7)
        r = f(self.sp, "lsf", ph, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(self.sp, "lsf", ph, self.tx_pos, self.rx_pos, 5), r)  # set_id is ignored for "lsf"
        npt.assert_array_equal(f(self.sp, "lsf", ph, self.tx_pos, self.rx_pos, center_freq=[-1.0, np.nan]), r)
        pc = quadriga_lib.gbsm.init_phases(self.sp, "cluster", 7)
        rc = f(self.sp, "cluster", pc, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(self.sp, "cluster", pc, self.tx_pos, self.rx_pos, center_freq=np.array([3.5e9])), rc)
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "rays", ph, self.tx_pos, self.rx_pos)
        self.assertIn("Input 'set_type' must be 'lsf', 'cluster' or 'ray'.", str(ctx.exception))

    def test_ray(self):
        # (n_link, 14·N·M + N); center_freq is required by the shared set (set_id = -1) only
        f = quadriga_lib.gbsm.draw
        sr = {"NumClusters": 3, "NumClusters_min": 1, "NumSubPaths": 4, "SC_lambda": 15.0, "PerClusterDS": 11.0,
              "PerClusterDS_gamma": -2.0, "PerClusterAS_A": 22.0}  # N = 2, M = 4
        n_link = self.rx_pos.shape[0]
        pr = quadriga_lib.gbsm.init_phases(sr, "ray", 7)
        rr = f(sr, "ray", pr, self.tx_pos, self.rx_pos, center_freq=[3.5e9])
        self.assertEqual(rr.shape, (n_link, 114))
        self.assertTrue(np.all((rr[:, 106:] > 0) & (rr[:, 106:] < 1)))  # block 14, Uniform
        self.assertFalse(np.array_equal(f(sr, "ray", pr, self.tx_pos, self.rx_pos, center_freq=[28e9]), rr))
        with self.assertRaises(ValueError) as ctx:
            f(sr, "ray", pr, self.tx_pos, self.rx_pos)
        self.assertIn("Input 'center_frequency' is required for the shared 'ray' set.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(sr, "ray", pr, self.tx_pos, self.rx_pos, center_freq=[np.nan])
        self.assertIn("Input 'center_frequency' must be finite and > 0.", str(ctx.exception))
        pk = quadriga_lib.gbsm.init_phases(sr, "ray", 7, 1)
        rk = f(sr, "ray", pk, self.tx_pos, self.rx_pos, 1)  # k >= 0
        self.assertEqual(rk.shape, (n_link, 114))
        npt.assert_array_equal(f(sr, "ray", pk, self.tx_pos, self.rx_pos, 1, center_freq=[-1.0, np.nan]), rk)


if __name__ == "__main__":
    unittest.main()
