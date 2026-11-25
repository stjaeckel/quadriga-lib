# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.set_antenna.
# Scope: argument and output handling, index base, defaults, in/out split, error paths.
# The antenna lists are tested in the Catch2 suite (test_qd_layout.cpp).
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
        # 2 Tx sites and 2 Rx terminals without antennas, two bands
        self.tmp = tempfile.TemporaryDirectory()
        fn_tx, fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_tx, "name,x,y,z\nA,0,0,25\nB,500,0,25\n")
        write_text(fn_rx, "name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n")
        self.L = quadriga_lib.layout.csv_read(fn_tx, fn_rx, center_freq=[2e9, 3.5e9])[0]
        self.panel = {"panel": [2, 2, 1, 2]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L = self.L
        L2 = quadriga_lib.layout.set_antenna(L, "3GPP", self.panel)  # a panel of 2 x 2 x 1 x 2 elements for all Tx nodes
        npt.assert_array_equal(L2["tx"]["i_antenna"], [1, 1])  # one shared entry; 1-based, 0 = generic antenna
        npt.assert_array_equal(L2["rx"]["i_antenna"], [0, 0])
        self.assertEqual(L2["antenna_cols"], ["antenna", "panel"])
        self.assertEqual(L2["antenna_args"], [['3GPP,"2,2,1,2"']])
        self.assertEqual([len(a) for a in L2["antenna"]], [2])  # an array spaced in wavelengths: one dict per band
        self.assertEqual(L2["antenna"][0][1]["element_pos"].shape, (3, 8))
        npt.assert_array_equal(L["tx"]["i_antenna"], [0, 0])  # the input stays unchanged
        self.assertEqual(L["antenna"], [])

    def test_optional_inputs(self):
        L, panel = self.L, self.panel
        L3 = quadriga_lib.layout.set_antenna(L, "3GPP", panel, rx_side=True, i_node=1)  # 0-based node 1 of the Rx side
        npt.assert_array_equal(L3["rx"]["i_antenna"], [0, 1])
        npt.assert_array_equal(L3["tx"]["i_antenna"], [0, 0])
        L4 = quadriga_lib.layout.set_antenna(L, "dipole", center_freq=2e9)  # one band: a line per band
        self.assertEqual(L4["antenna_args"], [["2000000000,dipole", "3500000000,"]])
        L4 = quadriga_lib.layout.set_antenna(L4, "3GPP", panel, center_freq=3.5e9)
        self.assertEqual(L4["antenna_args"], [["2000000000,dipole,", '3500000000,3GPP,"2,2,1,2"']])
        self.assertEqual(L4["antenna"][0][1]["element_pos"].shape, (3, 8))
        L5 = quadriga_lib.layout.set_antenna(L, "3GPP", panel, evaluate=False)  # no evaluation
        self.assertEqual(L5["antenna"], [])
        npt.assert_array_equal(L5["tx"]["i_antenna"], [1, 1])
        L6 = quadriga_lib.layout.set_antenna(self.L, "3GPP", panel)
        L6 = quadriga_lib.layout.set_antenna(L6, "none", i_node=0)  # no antenna for the first node
        npt.assert_array_equal(L6["tx"]["i_antenna"], [0, 1])
        L7 = quadriga_lib.layout.set_antenna(L, "3GPP", {"panel": np.array([2, 2, 1, 2], dtype=np.float32), "center_freq": 3.5e9})
        self.assertEqual(L7["antenna_args"][0][1], '3500000000,3GPP,"2,2,1,2"')  # the band in args
        L8 = quadriga_lib.layout.set_antenna(L, "ula", {"panel": [4, 1, 1, 1], "steer": [10, 5], "apply_weights": True}, evaluate=False)
        self.assertEqual(L8["antenna_args"], [['ula,"4,1,1,1","10,5",1']])  # a 1D steer is one row; bool as 1
        # None, other types and keywords
        L9 = quadriga_lib.layout.set_antenna(L, "3GPP", args=panel, rx_side=None, i_node=np.array([0, 1], dtype=np.uint8),
                                             center_freq=np.nan, evaluate=None)
        self.assertTrue(same(L9, quadriga_lib.layout.set_antenna(L, "3GPP", panel)))
        self.assertTrue(same(quadriga_lib.layout.set_antenna(L, "dipole", None), quadriga_lib.layout.set_antenna(L, "dipole")))

    def test_errors(self):
        L, panel = self.L, self.panel
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # rx_side is keyword-only
            quadriga_lib.layout.set_antenna(L, "dipole", None, True)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna(L)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna(1, "dipole")
        with self.assertRaisesRegex(ValueError, "Input 'args' must be a dict or None"):
            quadriga_lib.layout.set_antenna(L, "dipole", 5)
        with self.assertRaisesRegex(ValueError, "Keys of 'args' must be str"):
            quadriga_lib.layout.set_antenna(L, "dipole", {1: 2})
        with self.assertRaisesRegex(TypeError, "Input 'evaluate' must be a bool or None"):
            quadriga_lib.layout.set_antenna(L, "dipole", evaluate=1)
        with self.assertRaisesRegex(ValueError, "Input 'i_node'"):  # negative index
            quadriga_lib.layout.set_antenna(L, "dipole", i_node=-1)
        with self.assertRaisesRegex(ValueError, "i_node"):  # C++: out of range
            quadriga_lib.layout.set_antenna(L, "dipole", i_node=2)
        with self.assertRaisesRegex(ValueError, "unknown argument 'foo'"):  # C++
            quadriga_lib.layout.set_antenna(L, "3GPP", {"foo": 1})
        with self.assertRaisesRegex(ValueError, "is not used by the antenna type 'dipole'"):  # C++
            quadriga_lib.layout.set_antenna(L, "dipole", panel)
        with self.assertRaisesRegex(ValueError, "Input 'type' must be an antenna type"):  # C++
            quadriga_lib.layout.set_antenna(L, "nope")


if __name__ == "__main__":
    unittest.main()
