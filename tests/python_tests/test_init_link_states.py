# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.init_link_states.
# Scope: argument and output handling, index base, defaults, seed, config_dirs, in/out split, error paths.
# The link states are tested in the Catch2 suite (test_init_link_states.cpp).
# Physics validation lives in the Catch2 suite.

import sys
import os
import pathlib
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def write_text(fn, text):
    with open(fn, "w") as f:
        f.write(text)


def same(a, b):
    # Deep equality of dicts, lists and arrays; NaN equals NaN, dtypes and shapes must match
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


class test_case(unittest.TestCase):

    def setUp(self):
        # 2 Tx sites and 2 Rx terminals (U1 moves north at 2 m/s), two bands
        self.tmp = tempfile.TemporaryDirectory()
        self.fn_tx, self.fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(self.fn_tx, "name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n")
        write_text(self.fn_rx, "name,x,y,z,indoor,speed,direction_az\nU1,100,0,1.5,0,2,90\nU2,900,0,1.5,0,,\n")
        self.L = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9])[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L = self.L
        self.assertEqual(L["links"]["i_tx"].size, 0)
        L2 = quadriga_lib.layout.init_link_states(L, seed=5)
        k = L2["links"]
        self.assertEqual(L2["seed"], 5)
        self.assertEqual(k["seed"], 5)
        npt.assert_array_equal(k["i_tx"], [0, 1, 0, 1])  # one link per site and terminal, terminal-major, 0-based
        npt.assert_array_equal(k["i_rx"], [0, 0, 1, 1])
        npt.assert_array_equal(k["i_segment"], [0, 0, 0, 0])
        npt.assert_array_equal(k["image"], [0, 0, 0, 0])
        self.assertEqual(k["conf_names"], ["UMa_LOS", "UMa_NLOS", "UMa_LOS_O2I", "UMa_NLOS_O2I"])
        self.assertTrue(np.all(k["i_conf"] < 4))  # 0-based index into conf_names
        for l in range(4):  # the LOS state selects the conf; indoor = 0: no O2I conf
            name = k["conf_names"][k["i_conf"][l]]
            self.assertEqual(name.endswith("_LOS"), bool(k["los"][l]))
            self.assertNotIn("O2I", name)
        npt.assert_array_equal(k["indoor"], np.zeros(4))
        npt.assert_array_equal(k["floor_id"], np.zeros(4))
        npt.assert_array_equal(k["d2d_in"], np.zeros(4))
        self.assertEqual(k["o2i_rnd"].shape, (2, 4))  # one row per terminal
        self.assertEqual(k["o2i_model"].shape, (2,))
        self.assertEqual((k["i_tx"].dtype, k["los"].dtype, k["indoor"].dtype, k["gbsm_seed"].dtype),
                         (np.uint32, np.uint32, np.int32, np.int64))
        self.assertEqual(k["method"], "38.901")
        self.assertEqual(k["sos_preset"], "Exp256")
        self.assertEqual(k["anchor_freq"], 2e9)
        self.assertIs(k["pol_variability"], False)
        self.assertEqual(len(k["conf_files"]), 4)
        self.assertTrue(os.path.isfile(k["conf_files"][1]))  # the shipped folder was found
        self.assertIsInstance(k["layout_hash"], int)
        self.assertNotEqual(k["layout_hash"], 0)
        self.assertTrue(same(L2["tx"], L["tx"]))  # the rest is unchanged
        self.assertEqual(L["links"]["i_tx"].size, 0)  # the input stays unchanged

    def test_seed(self):
        L = self.L
        L2 = quadriga_lib.layout.init_link_states(L, seed=5)
        self.assertTrue(same(quadriga_lib.layout.init_link_states(L, seed=5), L2))
        L4 = quadriga_lib.layout.init_link_states(L)  # a drawn seed is stored
        self.assertGreaterEqual(L4["seed"], 0)
        self.assertEqual(L4["links"]["seed"], L4["seed"])
        L5 = quadriga_lib.layout.init_link_states(L, seed=L4["seed"])
        self.assertTrue(same(L5["links"], L4["links"]))
        L6 = quadriga_lib.layout.init_link_states(L, seed=6)
        self.assertFalse(np.array_equal(L6["links"]["gbsm_seed"], L2["links"]["gbsm_seed"]))

    def test_optional_inputs(self):
        L = self.L
        L7 = quadriga_lib.layout.init_link_states(L, method="38.901_MF", o2i_low=0.5, pol_variability=True, anchor_freq=3e9,
                                                  sos_preset="Comb256", seed=5)
        self.assertEqual((L7["links"]["method"], L7["links"]["anchor_freq"], L7["links"]["sos_preset"]), ("38.901_MF", 3e9, "Comb256"))
        self.assertIs(L7["links"]["pol_variability"], True)
        L8 = quadriga_lib.layout.init_link_states(L, method="quadriga", seed=5)
        self.assertEqual((L8["links"]["method"], L8["links"]["sos_preset"]), ("quadriga", "Comb256"))  # the default of this method
        L9 = quadriga_lib.layout.init_link_states(L, method="38.901", o2i_low=None, pol_variability=None, anchor_freq=np.float32(2e9),
                                                  sos_preset="", seed=np.uint8(5), config_dirs=None)
        self.assertTrue(same(L9, quadriga_lib.layout.init_link_states(L, seed=5)))
        self.assertTrue(same(quadriga_lib.layout.init_link_states(layout_in=L, seed=5, config_dirs=[]), L9))  # the shipped folder is added

    def test_config_dirs(self):
        # A user conf file is found in config_dirs (str, list or tuple); the wrapper adds the shipped folder
        L = self.L
        k = quadriga_lib.layout.init_link_states(L, seed=5)["links"]
        folder = os.path.join(self.tmp.name, "conf")
        os.mkdir(folder)
        with open(k["conf_files"][1]) as f:
            write_text(os.path.join(folder, "MyUMa.conf"), f.read())
        Lc = dict(L, tx=dict(L["tx"], scenario=["MyUMa", "MyUMa"]))
        Lc2 = quadriga_lib.layout.init_link_states(Lc, seed=5, config_dirs=folder)
        self.assertEqual(Lc2["links"]["conf_names"], ["MyUMa"])
        self.assertEqual(len(Lc2["links"]["conf_files"]), 1)
        self.assertTrue(Lc2["links"]["conf_files"][0].endswith("MyUMa.conf"))
        Lc3 = quadriga_lib.layout.init_link_states(Lc, seed=5, config_dirs=["does_not_exist_folder", folder])
        self.assertTrue(same(Lc3, Lc2))
        self.assertTrue(same(quadriga_lib.layout.init_link_states(Lc, seed=5, config_dirs=(folder,)), Lc2))

    def test_errors(self):
        L = self.L
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # method is keyword-only
            quadriga_lib.layout.init_link_states(L, "38.901")
        with self.assertRaises(TypeError):
            quadriga_lib.layout.init_link_states(1)
        with self.assertRaisesRegex(TypeError, "Input 'pol_variability' must be a bool or None"):
            quadriga_lib.layout.init_link_states(L, pol_variability=1)
        with self.assertRaises(ValueError):  # config_dirs must be str or a list of str
            quadriga_lib.layout.init_link_states(L, config_dirs=5)
        with self.assertRaisesRegex(ValueError, "method"):  # C++
            quadriga_lib.layout.init_link_states(L, method="foo")
        with self.assertRaisesRegex(ValueError, "o2i_low"):  # C++
            quadriga_lib.layout.init_link_states(L, o2i_low=2)
        Lc = dict(L, tx=dict(L["tx"], scenario=["MyUMa", "MyUMa"]))
        with self.assertRaisesRegex(ValueError, "scenario 'MyUMa' is neither a scenario family nor a conf file"):  # C++
            quadriga_lib.layout.init_link_states(Lc, seed=5)
        L0 = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx)[0]
        with self.assertRaisesRegex(ValueError, "center_freq"):  # C++: no frequencies
            quadriga_lib.layout.init_link_states(L0)


if __name__ == "__main__":
    unittest.main()
