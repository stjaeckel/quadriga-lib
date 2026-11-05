# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.init_phases.
# Scope: argument and output handling, set types and set index, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_phases_lsf.cpp, test_gbsm_phases_cluster.cpp).

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
MIRROR = [0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10]  # "lsf": ASD <-> ASA, ESD <-> ESA

class test_lsf(unittest.TestCase):

    def setUp(self):
        # Distinct decorrelation distances, 4 Rx positions, one Tx
        self.scenpar = {"DS_lambda": 10.0, "AS_D_lambda": 18.0, "AS_A_lambda": 15.0}
        self.rx_pos = np.array([[0.0, 0.0, 1.5], [20.0, 5.0, 1.5], [45.0, -10.0, 3.0], [80.0, 30.0, 1.5]])
        self.tx_pos = np.array([[10.0, 10.0, 25.0]])
        self.phase = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7)
        self.pb = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7, tx_pos=None, rx_pos=self.rx_pos)

    def test_basic_call(self):
        phase = self.phase
        self.assertIsInstance(phase, np.ndarray)
        self.assertEqual(phase.shape, (256, 11, 2))
        self.assertEqual(phase.dtype, np.float64)
        npt.assert_array_equal(phase[:, :, 0], quadriga_lib.sos.init_phases(256, 11, 7))  # raw phases
        npt.assert_array_equal(phase[:, :, 1], phase[:, MIRROR, 0])  # slice 1 mirrors slice 0
        self.assertTrue(np.all((phase >= -np.pi) & (phase < np.pi)))

    def test_keyword_arguments(self):
        pk = quadriga_lib.gbsm.init_phases(scenpar=self.scenpar, set_type="lsf", seed=7, preset="Comb256",
                                           tx_pos=self.tx_pos, rx_pos=self.rx_pos, dual_mobility=False,
                                           remove_bias=True, horizontal=True)
        npt.assert_array_equal(pk, self.pb)
        pp = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7, preset="Comb256", tx_pos=self.tx_pos,
                                           rx_pos=self.rx_pos, dual_mobility=False, remove_bias=True, horizontal=True)
        npt.assert_array_equal(pp, self.pb)

    def test_seed(self):
        npt.assert_array_equal(quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7), self.phase)
        npt.assert_array_equal(quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", np.int64(7)), self.phase)
        p1 = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf")  # default -1: system random device
        p2 = quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", -1)
        self.assertEqual(p1.shape, (256, 11, 2))
        self.assertGreater(np.max(np.abs(p1 - p2)), 0.1)
        npt.assert_array_equal(p1[:, :, 1], p1[:, MIRROR, 0])

    def test_positions_and_bias_removal(self):
        pb = self.pb
        npt.assert_array_equal(pb[:, :, 1], pb[:, MIRROR, 0])
        self.assertTrue(np.all(np.max(np.abs(pb[:, :9, 0] - self.phase[:, :9, 0]), axis=0) > 1e-6))
        npt.assert_array_equal(pb[:, 9:, :], self.phase[:, 9:, :])  # Uniform generators unchanged

        # The corrected fields have a zero mean over the positions (gbsm.draw may use single-precision sin/cos)
        rnd = quadriga_lib.gbsm.draw(self.scenpar, "lsf", pb, self.tx_pos, self.rx_pos)
        npt.assert_allclose(np.mean(rnd[:, :9], axis=0), np.zeros(9), atol=1e-4, rtol=0)

        # Omitted, None and non-default values
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, tx_pos=None, rx_pos=None), self.phase)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos[:2]), self.phase)  # 2 positions: none
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=np.zeros((0, 3))), self.phase)  # empty: none
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, remove_bias=False), self.phase)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, tx_pos=self.tx_pos, rx_pos=self.rx_pos), pb)  # tx_pos unused

        # Dual mobility: over the Rx and Tx positions; one Tx row equals n_link equal rows
        pd = f(self.scenpar, "lsf", 7, tx_pos=self.tx_pos, rx_pos=self.rx_pos, dual_mobility=True)
        self.assertGreater(np.max(np.abs(pd - pb)), 1e-6)
        npt.assert_array_equal(pd[:, :, 1], pd[:, MIRROR, 0])
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, tx_pos=np.tile(self.tx_pos, (4, 1)), rx_pos=self.rx_pos,
                                 dual_mobility=True), pd)

    def test_horizontal_and_preset(self):
        f = quadriga_lib.gbsm.init_phases
        rx_z = self.rx_pos.copy()
        rx_z[:, 2] = [0.0, 7.0, 2.0, 11.0]
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=rx_z), self.pb)  # 2D: z not used
        ph = f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, horizontal=False)
        self.assertGreater(np.max(np.abs(ph - self.pb)), 1e-6)
        pc = f(self.scenpar, "lsf", 7, preset="Comb1024")
        self.assertEqual(pc.shape, (1024, 11, 2))
        npt.assert_array_equal(pc[:, :, 0], quadriga_lib.sos.init_phases(1024, 11, 7))

    def test_flags(self):
        f = quadriga_lib.gbsm.init_phases
        # None selects the default of each flag
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, preset="Comb256", tx_pos=None, rx_pos=self.rx_pos,
                                 dual_mobility=None, remove_bias=None, horizontal=None), self.pb)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, remove_bias=None), self.pb)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, horizontal=None), self.pb)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, remove_bias=np.bool_(False)), self.phase)
        # Any other type raises TypeError
        for name in ("dual_mobility", "remove_bias", "horizontal"):
            for bad in (2.5, 1, "yes", np.array([True])):
                with self.assertRaises(TypeError) as ctx:
                    f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos, **{name: bad})
                self.assertIn(f"Input '{name}' must be a bool or None.", str(ctx.exception))

    def test_scenpar(self):
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(f(None, "lsf", 7), self.phase)  # no positions: phases do not depend on scenpar
        npt.assert_array_equal(f({}, "lsf", 7), self.phase)
        pbd = f(None, "lsf", 7, rx_pos=self.rx_pos)
        self.assertGreater(np.max(np.abs(pbd[:, 0, 0] - self.pb[:, 0, 0])), 1e-6)  # DS_lambda differs
        npt.assert_array_equal(pbd[:, 1, 0], self.pb[:, 1, 0])  # KF_lambda is the default in both
        sv = quadriga_lib.gbsm.scenpar_validate(self.scenpar)[0]  # complete dict
        npt.assert_array_equal(f(sv, "lsf", 7, rx_pos=self.rx_pos), self.pb)
        sp_int = {"DS_lambda": 10, "AS_D_lambda": np.float32(18), "AS_A_lambda": np.array([15])}
        npt.assert_array_equal(f(sp_int, "lsf", 7, rx_pos=self.rx_pos), self.pb)

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=np.ascontiguousarray(self.rx_pos)), self.pb)
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=np.asfortranarray(self.rx_pos)), self.pb)
        big = np.zeros((8, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=big[::2, ::2]), self.pb)  # strided view
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos.astype(np.float32)), self.pb)  # exact
        npt.assert_array_equal(f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos.tolist()), self.pb)

    def test_inputs_not_modified(self):
        rx = self.rx_pos.copy()
        tx = self.tx_pos.copy()
        sp = dict(self.scenpar)
        quadriga_lib.gbsm.init_phases(self.scenpar, "lsf", 7, tx_pos=self.tx_pos, rx_pos=self.rx_pos,
                                      dual_mobility=True)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(self.tx_pos, tx)
        self.assertEqual(self.scenpar, sp)

    def test_errors(self):
        f = quadriga_lib.gbsm.init_phases
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", 7, preset="Comb999")
        self.assertIn("Input 'preset' is not a known preset.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", 7, tx_pos=None, rx_pos=self.rx_pos, dual_mobility=True)
        self.assertIn("Input 'tx_pos' is required for the bias removal with dual mobility.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", 7, rx_pos=self.rx_pos[:, :2])
        self.assertIn("Input 'rx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.scenpar, "lsf", -2)
        self.assertIn("Input 'seed' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++ (scenpar_validate)
            f({"Foo": 1.0}, "lsf", 7)
        self.assertIn("Unknown key 'Foo' in 'scenpar'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++ (scenpar_validate)
            f({"DS_lambda": -1.0}, "lsf", 7)
        self.assertIn("DS_lambda", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # scenpar helper
            f(5, "lsf", 7)
        self.assertIn("Input 'scenpar' must be a dict or None.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f()
        with self.assertRaises(TypeError):
            f(self.scenpar, "lsf", foo=1)


class test_cluster(unittest.TestCase):

    def setUp(self):
        # UMi LOS file, N = 12 NLOS clusters
        self.sp = quadriga_lib.gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
        self.N = int(self.sp["NumClusters"]) - 1
        blk = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
        self.mirror = (blk[:, None] * self.N + np.arange(self.N)[None, :]).ravel()
        self.phase = quadriga_lib.gbsm.init_phases(self.sp, "cluster", 42)

    def test_basic_call(self):
        phase = self.phase
        self.assertIsInstance(phase, np.ndarray)
        self.assertEqual(phase.shape, (256, 9 * self.N, 2))
        self.assertEqual(phase.dtype, np.float64)
        self.assertTrue(np.all((phase >= -np.pi) & (phase < np.pi)))
        npt.assert_array_equal(phase[:, :, 1], phase[:, self.mirror, 0])  # slice 1 mirrors slice 0
        npt.assert_array_equal(quadriga_lib.gbsm.init_phases(self.sp, "cluster", 42), phase)

    def test_seed_rule(self):
        # Seeds of the sets: SplitMix64 of the large-scale seed and the set tag (test vectors of the seed rule)
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(self.phase[:, :, 0],
                               quadriga_lib.sos.init_phases(256, 9 * self.N, 8766244108781592946))
        npt.assert_array_equal(f(self.sp, "cluster", 42, 0)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, self.N, 4119046106699552547))
        npt.assert_array_equal(f(self.sp, "cluster", 42, 1)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, self.N, 9018399064009245349))
        npt.assert_array_equal(f(self.sp, "cluster", 0)[:, :, 0],
                               quadriga_lib.sos.init_phases(256, 9 * self.N, 313702574736366215))

    def test_keyword_arguments(self):
        pk = quadriga_lib.gbsm.init_phases(scenpar=self.sp, set_type="cluster", seed=42, set_id=-1, preset="Comb256")
        npt.assert_array_equal(pk, self.phase)

    def test_optional_inputs(self):
        f = quadriga_lib.gbsm.init_phases
        pr = f(self.sp, "cluster")  # seed -1: system random device
        self.assertEqual(pr.shape, self.phase.shape)
        self.assertTrue(np.max(np.abs(pr - self.phase)) > 1)
        npt.assert_array_equal(f(self.sp, "cluster", 42, -1), self.phase)
        f0 = f(self.sp, "cluster", 42, 0)
        self.assertEqual(f0.shape, (256, self.N, 2))
        npt.assert_array_equal(f0[:, :, 1], f0[:, :, 0])  # shadowing mirrors itself
        self.assertTrue(np.max(np.abs(f(self.sp, "cluster", 42, 1) - f0)) > 1)
        pc = f(self.sp, "cluster", 42, -1, "Comb1024")
        self.assertEqual(pc.shape, (1024, 9 * self.N, 2))

    def test_scenpar(self):
        f = quadriga_lib.gbsm.init_phases
        self.assertEqual(f({"NumClusters": 5, "NumClusters_min": 1}, "cluster", 42).shape, (256, 36, 2))
        p0 = f({"NumClusters": 1, "NumClusters_min": 1}, "cluster", 42, 0)  # N = 0: empty set
        self.assertEqual(p0.shape, (256, 0, 2))
        self.assertEqual(f(None, "cluster", 42).shape[1] % 9, 0)  # all defaults

    def test_errors(self):
        f = quadriga_lib.gbsm.init_phases
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", -2)
        self.assertIn("Input 'seed' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", 42, -2)
        self.assertIn("Input 'set_id' must be >= -1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", 42, -1, "Unknown")
        self.assertIn("preset", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f({"NumClusters": 0}, "cluster", 42)
        self.assertIn("NumClusters", str(ctx.exception))
        with self.assertRaises(TypeError):
            f()
        with self.assertRaises(TypeError):
            f(self.sp, "cluster", seed=42, unknown=1)
        with self.assertRaises(TypeError):
            f(self.sp, "cluster", 1.5)


class test_set_types(unittest.TestCase):

    def setUp(self):
        self.sp = {"DS_lambda": 10.0}
        self.phase = quadriga_lib.gbsm.init_phases(self.sp, "lsf", 7)

    def test_set_type_and_set_id(self):
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(f(self.sp, "lsf", 7, 5), self.phase)  # set_id is ignored for "lsf"
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "LSF", 7)
        self.assertIn("Input 'set_type' must be 'lsf', 'cluster' or 'ray'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "ray", 7)
        self.assertIn("Set type 'ray' is not available yet.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, "cluster", 7, rx_pos=np.zeros((3, 3)))
        self.assertIn("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only.", str(ctx.exception))

    def test_scenario_dict(self):
        # 'SubpathMethod' = 'legacy' is accepted and ignored, other values and types raise; outputs have no such key
        f = quadriga_lib.gbsm.init_phases
        npt.assert_array_equal(f({"DS_lambda": 10.0, "SubpathMethod": "legacy"}, "lsf", 7), self.phase)
        with self.assertRaises(ValueError) as ctx:
            f({"DS_lambda": 10.0, "SubpathMethod": "mmMAGIC"}, "lsf", 7)
        self.assertIn("Value 'mmMAGIC' of 'SubpathMethod' is not supported; only 'legacy' is accepted.",
                      str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f({"DS_lambda": 10.0, "SubpathMethod": 5}, "lsf", 7)
        self.assertIn("Entry 'SubpathMethod' of 'scenpar' must be a str.", str(ctx.exception))
        self.assertNotIn("SubpathMethod", quadriga_lib.gbsm.scenpar_validate(self.sp)[0])


if __name__ == "__main__":
    unittest.main()
