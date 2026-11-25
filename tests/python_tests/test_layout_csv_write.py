# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.csv_write.
# Scope: argument handling, the files written, defaults, error paths.
# The format and the rules of the columns are tested in the Catch2 suite (test_layout_csv.cpp).
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


def same(a, b, rtol=0.0):
    # Deep equality of dicts, lists and arrays; NaN equals NaN, dtypes and shapes must match; float arrays within rtol
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        if not (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype):
            return False
        if a.dtype.kind == "f":
            return bool(np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True))
        return np.array_equal(a, b)
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b

# The file holds numbers with 15 significant digits (16 or 17 where 15 do not read back) and angles in degrees
RTOL = 1e-14


class test_case(unittest.TestCase):

    def setUp(self):
        # A layout read from files: 3 Tx nodes, 2 Rx terminals, 2 frequencies
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.fn_tx, self.fn_rx = os.path.join(d, "tx.csv"), os.path.join(d, "rx.csv")
        self.fn_q = os.path.join(d, "rx_1.qdant")
        write_text(self.fn_tx, "name,sector,x,y,z,heading,tilt,antenna,panel,note\n"
                   "BS1,0,0,0,25,30,10,3gpp,\"2,2,1,2\",a\n"
                   "BS1,1,,,,150,,,,\n"
                   "BS2,0,500,0,25,90,0,dipole,,b\n")
        write_text(self.fn_rx, "name,x,y,z,heading,speed,direction_az,scenario,pairing\n"
                   "UT1,10,20,1.5,45,0.833,120,UMa,BS1\n"
                   "UT2,0,0,1.5,0,,,,\n"
                   "UT2,1,0,1.5,,,,,\n"
                   "UT2,2,0,1.5,90,,,,\n"
                   "UT2,2,1,1.5,,,,UMa_NLOS,\n")
        self.L, _, _ = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9])
        os.remove(self.fn_tx)
        os.remove(self.fn_rx)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        return quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx)[0]

    def test_basic_call(self):
        self.assertIsNone(quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L))
        self.assertTrue(same(self.read(), self.L, RTOL))
        with open(self.fn_rx) as f:
            self.assertIn("UT1,10,20,1.5,45,120,0.833,UMa,BS1", f.read())  # the units of the file: degrees

    def test_compact_and_one_side(self):
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L)
        size_compact = os.path.getsize(self.fn_rx)
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L, compact=False)  # every cell: a longer file, the same content
        self.assertGreater(os.path.getsize(self.fn_rx), size_compact)
        self.assertTrue(same(self.read(), self.L, RTOL))
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L, compact=None)  # None = the default
        self.assertEqual(os.path.getsize(self.fn_rx), size_compact)
        os.remove(self.fn_tx)
        os.remove(self.fn_rx)
        quadriga_lib.layout.csv_write("", self.fn_rx, self.L)  # the other file is not written
        self.assertTrue(os.path.exists(self.fn_rx) and not os.path.exists(self.fn_tx))
        quadriga_lib.layout.csv_write(pathlib.Path(self.fn_tx), self.fn_rx.encode(), self.L)  # path-like and bytes
        self.assertTrue(os.path.exists(self.fn_tx))

    def test_qdant_export(self):
        # An array without a line is exported to a QDANT file next to the CSV file
        ant = quadriga_lib.arrayant.generate_antenna("omni", 10)
        Lq = quadriga_lib.layout.set_antenna_array(self.L, ant, rx=True, i_node=1)
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, Lq)
        self.assertTrue(os.path.exists(self.fn_q))
        Lb = self.read()
        npt.assert_array_equal(Lb["rx"]["i_antenna"], [0, 3])
        npt.assert_array_equal(Lb["antenna"][2][0]["e_theta_re"], ant["e_theta_re"])
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L)  # no longer referenced: removed
        self.assertFalse(os.path.exists(self.fn_q))

    def test_input_is_not_modified(self):
        L = self.L
        copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in L["tx"].items()}
        quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, L)
        self.assertTrue(same(L["tx"], copy))

    def test_errors(self):
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # compact is keyword-only
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L, True)
        with self.assertRaisesRegex(TypeError, "Input 'fn_tx' must be a str or path-like object"):
            quadriga_lib.layout.csv_write(1, self.fn_rx, self.L)
        with self.assertRaisesRegex(TypeError, "Input 'compact' must be a bool or None"):
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, self.L, compact=1)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, 1)
        Lx = dict(self.L)
        Lx["foo"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'layout'."):
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, Lx)
        Lx = dict(self.L)
        Lx["tx"] = dict(Lx["tx"], bar=1)
        with self.assertRaisesRegex(ValueError, "Unknown key 'bar' in 'layout.tx'."):
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, Lx)
        Lx = dict(self.L)
        Lx["tx"] = dict(Lx["tx"], name=["a,b", "BS1", "BS2"])
        with self.assertRaisesRegex(ValueError, "not valid"):  # C++
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_rx, Lx)
        with self.assertRaisesRegex(ValueError, "different"):  # C++
            quadriga_lib.layout.csv_write(self.fn_tx, self.fn_tx, self.L)


if __name__ == "__main__":
    unittest.main()
