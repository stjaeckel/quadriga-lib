# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.set_antenna_qdant.
# Scope: argument and output handling, index base, defaults, in/out split, error paths.
# The antenna lists, the bands and the CSV round trip are tested in the Catch2 suite (test_set_antenna.cpp).

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


class test_case(unittest.TestCase):

    def setUp(self):
        # 2 Tx sites and 2 Rx terminals without antennas, two bands; a QDANT file with two entries
        self.tmp = tempfile.TemporaryDirectory()
        fn_tx, fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_tx, "name,x,y,z\nA,0,0,25\nB,500,0,25\n")
        write_text(fn_rx, "name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n")
        self.L = quadriga_lib.layout.csv_read(fn_tx, fn_rx, center_freq=[2e9, 3.5e9])[0]
        self.fn = os.path.join(self.tmp.name, "meas.qdant")
        quadriga_lib.arrayant.qdant_write(self.fn, quadriga_lib.arrayant.generate_antenna("omni", 30), 1)
        quadriga_lib.arrayant.qdant_write(self.fn, quadriga_lib.arrayant.generate_antenna("dipole", 30), 2)
        self.F = pathlib.Path(os.path.abspath(self.fn)).as_posix()

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L, F = self.L, self.F
        L2 = quadriga_lib.layout.set_antenna_qdant(L, self.fn)  # entry 1 for all bands of the Tx nodes
        npt.assert_array_equal(L2["tx"]["i_antenna"], [1, 1])  # 1-based, 0 = generic antenna
        npt.assert_array_equal(L2["rx"]["i_antenna"], [0, 0])
        self.assertEqual(L2["antenna_cols"], ["antenna", "qdant_file", "qdant_id"])
        self.assertEqual(L2["antenna_args"], [["qdant," + F + ",1"]])
        self.assertEqual(len(L2["antenna"][0]), 1)  # one array for all bands
        self.assertEqual(L["antenna"], [])  # the input stays unchanged
        L3 = quadriga_lib.layout.set_antenna_qdant(L, pathlib.Path(self.fn))  # a path-like file name
        self.assertEqual(L3["antenna_args"], L2["antenna_args"])

    def test_optional_inputs(self):
        L, F = self.L, self.F
        ref, _ = quadriga_lib.arrayant.qdant_read(self.fn, 2)
        L3 = quadriga_lib.layout.set_antenna_qdant(L, self.fn, qdant_id=2, rx=True, i_node=1)  # 0-based
        npt.assert_array_equal(L3["rx"]["i_antenna"], [0, 1])
        npt.assert_array_equal(L3["tx"]["i_antenna"], [0, 0])
        self.assertEqual(L3["antenna_args"], [["qdant," + F + ",2"]])
        npt.assert_array_equal(L3["antenna"][0][0]["e_theta_re"], ref["e_theta_re"])
        L4 = quadriga_lib.layout.set_antenna_qdant(L, self.fn, 2, center_freq=3.5e9)  # qdant_id by position; the other band is not set yet
        self.assertEqual(L4["antenna_args"], [["2000000000,,,", "3500000000,qdant," + F + ",2"]])
        L5 = quadriga_lib.layout.set_antenna_qdant(L, self.fn, evaluate=False)
        self.assertEqual(L5["antenna_args"], [["qdant," + F + ",1"]])
        self.assertEqual(L5["antenna"], [])  # not evaluated
        L6 = quadriga_lib.layout.set_antenna_qdant(L, self.fn, qdant_id=1, rx=None, i_node=None, evaluate=None)
        self.assertEqual(L6["antenna_args"], [["qdant," + F + ",1"]])

    def test_errors(self):
        L = self.L
        with self.assertRaisesRegex(RuntimeError, "QDANT file .* does not exist or is not a file"):
            quadriga_lib.layout.set_antenna_qdant(L, os.path.join(self.tmp.name, "none.qdant"))
        with self.assertRaisesRegex(ValueError, "Input 'qdant_id' must be between 1"):
            quadriga_lib.layout.set_antenna_qdant(L, self.fn, qdant_id=0)
        with self.assertRaisesRegex(ValueError, "QDANT file"):
            quadriga_lib.layout.set_antenna_qdant(L, self.fn, qdant_id=3)
        with self.assertRaisesRegex(ValueError, "Input 'i_node' has an index outside"):
            quadriga_lib.layout.set_antenna_qdant(L, self.fn, i_node=2)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna_qdant(L, self.fn, qdant_id=-1)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_antenna_qdant(L, self.fn, 2, True)  # rx is keyword-only


if __name__ == "__main__":
    unittest.main()
