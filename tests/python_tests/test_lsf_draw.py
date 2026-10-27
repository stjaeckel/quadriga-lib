# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.lsf_draw.
# Scope: argument and output handling, floor IDs, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_phases.cpp).

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

SWAP = [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10]  # ASD <-> ASA, ESD <-> ESA


class test_case(unittest.TestCase):

    def setUp(self):
        # Phases of a scenario with distinct decorrelation distances, 5 links
        self.scenpar = {"DS_lambda": 10.0, "AS_D_lambda": 18.0, "AS_A_lambda": 15.0,
                        "ES_D_lambda": 12.0, "ES_A_lambda": 8.0}
        self.phase = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7)
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5],
                                [3.0, 4.0, 10.0]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.rnd = quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, self.rx_pos)
        self.fl = np.array([0, 1, 0, 2, 1])
        self.rf = quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, self.rx_pos, self.fl)

    def test_basic_call(self):
        rnd = self.rnd
        self.assertIsInstance(rnd, np.ndarray)
        self.assertEqual(rnd.shape, (5, 11))
        self.assertEqual(rnd.dtype, np.float64)
        self.assertTrue(np.all((rnd[:, 9:] > 0) & (rnd[:, 9:] < 1)))  # Uniform columns
        self.assertTrue(np.max(np.abs(rnd[:, :9])) < 6 and np.std(rnd) > 0.1)

        # Column 0 is the DS field (10 m) of slice 0 at the Rx positions projected on the x-y plane;
        # lsf_draw may evaluate sin/cos in single precision (AVX2 kernel)
        freq = quadriga_lib.sos.preset("Comb256", 10.0)[0]
        pos_xy = np.column_stack([self.rx_pos[:, :2], np.zeros(5)])
        ref = quadriga_lib.sos.val(freq, self.phase[:, 0, 0], pos_xy)
        npt.assert_allclose(rnd[:, 0], ref.ravel(), atol=1e-5, rtol=0)

        # Single mobility: the values do not depend on tx_pos; one link alone gives its row
        npt.assert_array_equal(quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, np.zeros((1, 3)),
                                                          self.rx_pos), rnd)
        r3 = quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, self.rx_pos[2:3])
        npt.assert_allclose(r3, rnd[2:3], atol=1e-12, rtol=0)

    def test_keyword_arguments(self):
        rk = quadriga_lib.gbsm.lsf_draw(scenpar=self.scenpar, phase=self.phase, tx_pos=self.tx_pos,
                                        rx_pos=self.rx_pos, floor_id=self.fl, dual_mobility=False,
                                        preset="Comb256", horizontal=True)
        npt.assert_array_equal(rk, self.rf)
        rp = quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, self.rx_pos, self.fl, False,
                                        "Comb256", True)
        npt.assert_array_equal(rp, self.rf)

    def test_floor_id(self):
        f = quadriga_lib.gbsm.lsf_draw
        args = (self.scenpar, self.phase, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(*args, None), self.rnd)
        npt.assert_array_equal(f(*args, np.zeros(5)), self.rnd)
        npt.assert_array_equal(f(*args, []), self.rnd)  # empty = all on floor 0
        rf = self.rf
        npt.assert_array_equal(rf[[0, 2]], self.rnd[[0, 2]])  # floor 0 unchanged
        self.assertTrue(np.all(np.max(np.abs(rf[[1, 3, 4]] - self.rnd[[1, 3, 4]]), axis=1) > 1e-3))

        # Floor IDs start at 0 (no index shift): floor 1 adds the offsets 2·pi·n/256 to the phases
        phase_1 = self.phase + (2 * np.pi * np.arange(256) / 256)[:, None, None]
        r1 = f(self.scenpar, phase_1, self.tx_pos, self.rx_pos[[1, 4]])
        npt.assert_allclose(r1, rf[[1, 4]], atol=1e-10, rtol=0)

        # Integer and float types of floor_id
        npt.assert_array_equal(f(*args, self.fl.astype(np.float64)), rf)
        npt.assert_array_equal(f(*args, self.fl.astype(np.uint8)), rf)
        npt.assert_array_equal(f(*args, self.fl.astype(np.uint64)), rf)
        npt.assert_array_equal(f(*args, self.fl.astype(np.float32)), rf)
        npt.assert_array_equal(f(*args, self.fl.tolist()), rf)
        npt.assert_array_equal(f(*args, self.fl.reshape(1, 5)), rf)

        # The largest exact ID is 2^53 - 1; larger values would be rounded when read as float64
        big = np.array([2**53 - 1, 1, 0, 2, 1], dtype=np.int64)
        self.assertEqual(f(*args, big).shape, (5, 11))

    def test_dual_mobility(self):
        f = quadriga_lib.gbsm.lsf_draw
        tx_d = np.array([[5.0, 5.0, 1.5], [30.0, 0.0, 1.5], [40.0, -20.0, 1.5], [60.0, 10.0, 1.5], [0.0, 9.0, 1.5]])
        rd = f(self.scenpar, self.phase, tx_d, self.rx_pos, None, True)
        self.assertGreater(np.max(np.abs(rd - self.rnd)), 1e-3)
        rs = f(self.scenpar, self.phase, self.rx_pos, tx_d, None, True)
        npt.assert_array_equal(rs, rd[:, SWAP])  # columns 3/4 and 5/6 swapped
        npt.assert_array_equal(f(self.scenpar, self.phase, tx_d, self.rx_pos, dual_mobility=False), self.rnd)

    def test_flags(self):
        f = quadriga_lib.gbsm.lsf_draw
        args = (self.scenpar, self.phase, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(*args, None, None, "Comb256", None), self.rnd)  # None selects the default
        npt.assert_array_equal(f(*args, dual_mobility=np.bool_(False), horizontal=np.bool_(True)), self.rnd)
        for name in ("dual_mobility", "horizontal"):
            for bad in (2.5, 0, "no"):
                with self.assertRaises(TypeError) as ctx:
                    f(*args, **{name: bad})
                self.assertIn(f"Input '{name}' must be a bool or None.", str(ctx.exception))

    def test_horizontal_and_preset(self):
        f = quadriga_lib.gbsm.lsf_draw
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] += [0.0, 5.0, 10.0, 2.0, 7.0]
        npt.assert_array_equal(f(self.scenpar, self.phase, self.tx_pos, rx_z), self.rnd)  # 2D: z not used
        r3 = f(self.scenpar, self.phase, self.tx_pos, self.rx_pos, horizontal=False)
        r3z = f(self.scenpar, self.phase, self.tx_pos, rx_z, horizontal=False)
        self.assertGreater(np.max(np.abs(r3z - r3)), 1e-3)  # 3D: z is used
        phase_c = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7, preset="Comb1024")
        rc = f(self.scenpar, phase_c, self.tx_pos, self.rx_pos, preset="Comb1024")
        self.assertEqual(rc.shape, (5, 11))
        self.assertGreater(np.max(np.abs(rc - self.rnd)), 1e-3)

    def test_scenpar(self):
        f = quadriga_lib.gbsm.lsf_draw
        sp2 = dict(self.scenpar)
        sp2["DS_lambda"] = 50.0
        r2 = f(sp2, self.phase, self.tx_pos, self.rx_pos)
        self.assertGreater(np.max(np.abs(r2[:, 0] - self.rnd[:, 0])), 1e-3)
        npt.assert_array_equal(r2[:, 1:], self.rnd[:, 1:])
        sv = quadriga_lib.gbsm.scenpar_validate(self.scenpar)[0]  # with 'SubpathMethod'
        npt.assert_array_equal(f(sv, self.phase, self.tx_pos, self.rx_pos), self.rnd)
        rn = f(None, self.phase, self.tx_pos, self.rx_pos)  # all defaults
        self.assertGreater(np.max(np.abs(rn[:, 0] - self.rnd[:, 0])), 1e-3)

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.lsf_draw
        npt.assert_array_equal(f(self.scenpar, self.phase, self.tx_pos, np.asfortranarray(self.rx_pos)), self.rnd)
        npt.assert_array_equal(f(self.scenpar, self.phase, self.tx_pos, np.ascontiguousarray(self.rx_pos)),
                               self.rnd)
        big = np.zeros((10, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.scenpar, self.phase, self.tx_pos, big[::2, ::2]), self.rnd)
        npt.assert_array_equal(f(self.scenpar, np.ascontiguousarray(self.phase), self.tx_pos, self.rx_pos),
                               self.rnd)
        rt = f(self.scenpar, self.phase, self.tx_pos.astype(np.float32), self.rx_pos.astype(np.float32))
        npt.assert_array_equal(rt, self.rnd)  # positions exact in float32
        rp = f(self.scenpar, self.phase.astype(np.float32), self.tx_pos, self.rx_pos)
        npt.assert_allclose(rp, self.rnd, atol=1e-4, rtol=0)  # phases rounded to float32

    def test_zero_links(self):
        r0 = quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, np.zeros((0, 3)))
        self.assertEqual(r0.shape, (0, 11))
        self.assertEqual(r0.dtype, np.float64)

    def test_inputs_not_modified(self):
        rx, ph, fl = self.rx_pos.copy(), self.phase.copy(), self.fl.copy()
        quadriga_lib.gbsm.lsf_draw(self.scenpar, self.phase, self.tx_pos, self.rx_pos, self.fl, True)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(self.phase, ph)
        npt.assert_array_equal(self.fl, fl)

    def test_errors(self):
        f = quadriga_lib.gbsm.lsf_draw
        args = (self.scenpar, self.phase, self.tx_pos, self.rx_pos)
        for bad in ([0, -1, 0, 0, 0], [0, 1.5, 0, 0, 0], [0, np.nan, 0, 0, 0], [0, 2.0**53, 0, 0, 0],
                    np.array([0, 2**53 + 1, 0, 0, 0], dtype=np.int64)):  # wrapper
            with self.assertRaises(ValueError) as ctx:
                f(*args, bad)
            self.assertIn("Entries in 'floor_id' must be non-negative integers below 2^53.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, self.phase[:, :10, :], self.tx_pos, self.rx_pos)
        self.assertIn("Input 'phase' must have size [256, 11, 2] for preset 'Comb256'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(*args, preset="Comb1024")
        self.assertIn("Input 'phase' must have size [1024, 11, 2] for preset 'Comb1024'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(*args, [0, 1])
        self.assertIn("Number of elements in 'floor_id' does not match", str(ctx.exception))
        msg3d = "Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors."
        for fl in (self.fl, np.zeros(5)):  # C++: 3D mode, also with all IDs 0
            with self.assertRaises(ValueError) as ctx:
                f(*args, fl, horizontal=False)
            self.assertIn(msg3d, str(ctx.exception))
        self.assertEqual(f(*args, None, horizontal=False).shape, (5, 11))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, self.phase, self.tx_pos[:, :2], self.rx_pos)
        self.assertIn("Input 'tx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.scenpar, self.phase, self.tx_pos)
        with self.assertRaises(TypeError):
            f(*args, foo=1)
        with self.assertRaises(TypeError):
            f(self.scenpar, None, self.tx_pos, self.rx_pos)  # mandatory array as None


if __name__ == "__main__":
    unittest.main()
