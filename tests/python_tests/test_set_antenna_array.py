# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.set_antenna_array.
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
        # 2 Tx sites and 2 Rx terminals without antennas, two bands; an omni-directional array
        self.tmp = tempfile.TemporaryDirectory()
        fn_tx, fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_tx, "name,x,y,z\nA,0,0,25\nB,500,0,25\n")
        write_text(fn_rx, "name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n")
        self.L = quadriga_lib.layout.csv_read(fn_tx, fn_rx, center_freq=[2e9, 3.5e9])[0]
        self.ant = quadriga_lib.arrayant.generate_antenna("omni", 10)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L, ant = self.L, self.ant
        L2 = quadriga_lib.layout.set_antenna_array(L, ant)  # stored for all bands of the Tx nodes
        npt.assert_array_equal(L2["tx"]["i_antenna"], [1, 1])  # 1-based, 0 = generic antenna
        npt.assert_array_equal(L2["rx"]["i_antenna"], [0, 0])
        self.assertEqual([len(a) for a in L2["antenna"]], [1])  # one array for all bands
        npt.assert_array_equal(L2["antenna"][0][0]["e_theta_re"], ant["e_theta_re"])
        npt.assert_array_equal(L2["antenna"][0][0]["azimuth_grid"], ant["azimuth_grid"])
        self.assertEqual(L2["antenna_args"], [[]])  # an evaluated array without description
        self.assertEqual(L["antenna"], [])  # the input stays unchanged

    def test_optional_inputs(self):
        L, ant = self.L, self.ant
        L3 = quadriga_lib.layout.set_antenna_array(L, ant, rx_side=True, i_node=1)  # 0-based
        npt.assert_array_equal(L3["rx"]["i_antenna"], [0, 1])
        npt.assert_array_equal(L3["tx"]["i_antenna"], [0, 0])
        L4 = quadriga_lib.layout.set_antenna_array(L, ant, center_freq=3.5e9)  # one band: the other is not set yet
        self.assertEqual(len(L4["antenna"][0]), 2)
        self.assertEqual(L4["antenna"][0][0]["e_theta_re"].size, 0)  # band 0 is not evaluated
        npt.assert_array_equal(L4["antenna"][0][1]["e_theta_re"], ant["e_theta_re"])
        L4b = quadriga_lib.layout.set_antenna(L4, "dipole", rx_side=True)  # arrays and unset bands pass through unchanged
        self.assertTrue(same(L4b["antenna"][0], L4["antenna"][0]))
        L5 = quadriga_lib.layout.set_antenna_array(L, ant, rx_side=None, i_node=np.array([0, 1], dtype=np.uint8), center_freq=np.nan)
        self.assertTrue(same(L5, quadriga_lib.layout.set_antenna_array(L, ant)))
        L6 = quadriga_lib.layout.set_antenna_array(layout_in=L, array=ant, rx_side=False, i_node=[0, 1])
        self.assertTrue(same(L6, L5))

    def test_errors(self):
        L, ant = self.L, self.ant
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # rx_side is keyword-only
            quadriga_lib.layout.set_antenna_array(L, ant, True)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna_array(L)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna_array(1, ant)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna_array(L, 1)
        with self.assertRaisesRegex(ValueError, "Input 'i_node'"):  # negative index
            quadriga_lib.layout.set_antenna_array(L, ant, i_node=-1)
        with self.assertRaisesRegex(ValueError, "i_node"):  # C++: out of range
            quadriga_lib.layout.set_antenna_array(L, ant, i_node=2)
        with self.assertRaisesRegex(ValueError, "Entry 'array' is not an arrayant dict: the key 'e_theta_re' is missing"):
            quadriga_lib.layout.set_antenna_array(L, {})
        bad = dict(ant, e_theta_re=ant["e_theta_re"][:, :, [0, 0]])
        with self.assertRaisesRegex(ValueError, "Input 'array' is not valid"):  # C++
            quadriga_lib.layout.set_antenna_array(L, bad)
        multi = dict(ant, e_theta_re=np.stack([ant["e_theta_re"]] * 2, axis=3))
        with self.assertRaisesRegex(ValueError, "single-frequency"):  # a 4D array
            quadriga_lib.layout.set_antenna_array(L, multi)


if __name__ == "__main__":
    unittest.main()
