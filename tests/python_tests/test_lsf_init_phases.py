# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.lsf_init_phases.
# Scope: argument and output handling, defaults, error paths.
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

MIRROR = [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10]  # ASD <-> ASA, ESD <-> ESA


class test_case(unittest.TestCase):

    def setUp(self):
        # Distinct decorrelation distances, 4 Rx positions, one Tx
        self.scenpar = {"DS_lambda": 10.0, "AS_D_lambda": 18.0, "AS_A_lambda": 15.0}
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.phase = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7)
        self.pb = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7, None, self.rx_pos)

    def test_basic_call(self):
        phase = self.phase
        self.assertIsInstance(phase, np.ndarray)
        self.assertEqual(phase.shape, (256, 11, 2))
        self.assertEqual(phase.dtype, np.float64)
        npt.assert_array_equal(phase[:, :, 0], quadriga_lib.sos.init_phases(256, 11, 7))  # raw phases
        npt.assert_array_equal(phase[:, :, 1], phase[:, MIRROR, 0])  # slice 1 mirrors slice 0
        self.assertTrue(np.all((phase >= -np.pi) & (phase < np.pi)))

    def test_keyword_arguments(self):
        pk = quadriga_lib.gbsm.lsf_init_phases(scenpar=self.scenpar, seed=7, tx_pos=self.tx_pos,
                                               rx_pos=self.rx_pos, dual_mobility=False, remove_bias=True,
                                               preset="Comb256", horizontal=True)
        npt.assert_array_equal(pk, self.pb)
        pp = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7, self.tx_pos, self.rx_pos, False, True,
                                               "Comb256", True)
        npt.assert_array_equal(pp, self.pb)

    def test_seed(self):
        npt.assert_array_equal(quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7), self.phase)
        npt.assert_array_equal(quadriga_lib.gbsm.lsf_init_phases(self.scenpar, np.int64(7)), self.phase)
        p1 = quadriga_lib.gbsm.lsf_init_phases(self.scenpar)  # default -1: system random device
        p2 = quadriga_lib.gbsm.lsf_init_phases(self.scenpar, -1)
        self.assertEqual(p1.shape, (256, 11, 2))
        self.assertGreater(np.max(np.abs(p1 - p2)), 0.1)
        npt.assert_array_equal(p1[:, :, 1], p1[:, MIRROR, 0])

    def test_positions_and_bias_removal(self):
        pb = self.pb
        npt.assert_array_equal(pb[:, :, 1], pb[:, MIRROR, 0])
        self.assertTrue(np.all(np.max(np.abs(pb[:, :9, 0] - self.phase[:, :9, 0]), axis=0) > 1e-6))
        npt.assert_array_equal(pb[:, 9:, :], self.phase[:, 9:, :])  # Uniform generators unchanged

        # The corrected fields have a zero mean over the positions (lsf_draw may use single-precision sin/cos)
        rnd = quadriga_lib.gbsm.lsf_draw(self.scenpar, pb, self.tx_pos, self.rx_pos)
        npt.assert_allclose(np.mean(rnd[:, :9], axis=0), np.zeros(9), atol=1e-4, rtol=0)

        # Omitted, None and non-default values
        f = quadriga_lib.gbsm.lsf_init_phases
        npt.assert_array_equal(f(self.scenpar, 7, None, None), self.phase)
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos[:2]), self.phase)  # 2 positions: none
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=np.zeros((0, 3))), self.phase)  # empty: none
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos, remove_bias=False), self.phase)
        npt.assert_array_equal(f(self.scenpar, 7, self.tx_pos, self.rx_pos), pb)  # tx_pos unused

        # Dual mobility: over the Rx and Tx positions; one Tx row equals n_link equal rows
        pd = f(self.scenpar, 7, self.tx_pos, self.rx_pos, True)
        self.assertGreater(np.max(np.abs(pd - pb)), 1e-6)
        npt.assert_array_equal(pd[:, :, 1], pd[:, MIRROR, 0])
        npt.assert_array_equal(f(self.scenpar, 7, np.tile(self.tx_pos, (4, 1)), self.rx_pos, True), pd)

    def test_horizontal_and_preset(self):
        f = quadriga_lib.gbsm.lsf_init_phases
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] = [0.0, 7.0, 2.0, 11.0]
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=rx_z), self.pb)  # 2D: z not used
        ph = f(self.scenpar, 7, rx_pos=self.rx_pos, horizontal=False)
        self.assertGreater(np.max(np.abs(ph - self.pb)), 1e-6)
        pc = f(self.scenpar, 7, preset="Comb1024")
        self.assertEqual(pc.shape, (1024, 11, 2))
        npt.assert_array_equal(pc[:, :, 0], quadriga_lib.sos.init_phases(1024, 11, 7))

    def test_flags(self):
        f = quadriga_lib.gbsm.lsf_init_phases
        # None selects the default of each flag
        npt.assert_array_equal(f(self.scenpar, 7, None, self.rx_pos, None, None, "Comb256", None), self.pb)
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos, remove_bias=None), self.pb)
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos, horizontal=None), self.pb)
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos, remove_bias=np.bool_(False)), self.phase)
        # Any other type raises TypeError
        for name in ("dual_mobility", "remove_bias", "horizontal"):
            for bad in (2.5, 1, "yes", np.array([True])):
                with self.assertRaises(TypeError) as ctx:
                    f(self.scenpar, 7, rx_pos=self.rx_pos, **{name: bad})
                self.assertIn(f"Input '{name}' must be a bool or None.", str(ctx.exception))

    def test_scenpar(self):
        f = quadriga_lib.gbsm.lsf_init_phases
        npt.assert_array_equal(f(None, 7), self.phase)  # no positions: phases do not depend on scenpar
        npt.assert_array_equal(f({}, 7), self.phase)
        pbd = f(None, 7, rx_pos=self.rx_pos)
        self.assertGreater(np.max(np.abs(pbd[:, 0, 0] - self.pb[:, 0, 0])), 1e-6)  # DS_lambda differs
        npt.assert_array_equal(pbd[:, 1, 0], self.pb[:, 1, 0])  # KF_lambda is the default in both
        sv = quadriga_lib.gbsm.scenpar_validate(self.scenpar)[0]  # complete dict incl. 'SubpathMethod'
        npt.assert_array_equal(f(sv, 7, rx_pos=self.rx_pos), self.pb)
        sp_int = {"DS_lambda": 10, "AS_D_lambda": np.float32(18), "AS_A_lambda": np.array([15])}
        npt.assert_array_equal(f(sp_int, 7, rx_pos=self.rx_pos), self.pb)

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.lsf_init_phases
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=np.ascontiguousarray(self.rx_pos)), self.pb)
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=np.asfortranarray(self.rx_pos)), self.pb)
        big = np.zeros((8, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=big[::2, ::2]), self.pb)  # strided view
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos.astype(np.float32)), self.pb)  # exact
        npt.assert_array_equal(f(self.scenpar, 7, rx_pos=self.rx_pos.tolist()), self.pb)

    def test_inputs_not_modified(self):
        rx = self.rx_pos.copy()
        tx = self.tx_pos.copy()
        sp = dict(self.scenpar)
        quadriga_lib.gbsm.lsf_init_phases(self.scenpar, 7, self.tx_pos, self.rx_pos, True)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(self.tx_pos, tx)
        self.assertEqual(self.scenpar, sp)

    def test_errors(self):
        f = quadriga_lib.gbsm.lsf_init_phases
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, 7, preset="Comb999")
        self.assertIn("Input 'preset' is not a known preset.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, 7, None, self.rx_pos, True)
        self.assertIn("Input 'tx_pos' is required for the bias removal with dual mobility.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, 7, rx_pos=self.rx_pos[:, :2])
        self.assertIn("Input 'rx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, -2)
        self.assertIn("Input 'seed' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++ (scenpar_validate)
            f({"Foo": 1.0}, 7)
        self.assertIn("Unknown key 'Foo' in 'scenpar'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++ (scenpar_validate)
            f({"DS_lambda": -1.0}, 7)
        self.assertIn("DS_lambda", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # scenpar helper
            f(5, 7)
        self.assertIn("Input 'scenpar' must be a dict or None.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f()
        with self.assertRaises(TypeError):
            f(self.scenpar, foo=1)


if __name__ == "__main__":
    unittest.main()
