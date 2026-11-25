# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.csv_read.
# Scope: argument and output handling, dict members, index base, defaults, error paths.
# The format, the rules of the columns and the antenna composition are tested in the Catch2 suite (test_layout_csv.cpp).
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


LAYOUT_KEYS = ["name", "tx", "rx", "links", "seed", "center_freq", "antenna", "antenna_cols", "antenna_args", "wrap", "pairing"]


class test_case(unittest.TestCase):

    def setUp(self):
        # 3 Tx nodes (2 sites, one with an extra sector, a 3GPP and a dipole antenna), 2 Rx terminals
        self.tmp = tempfile.TemporaryDirectory()
        self.fn_tx = os.path.join(self.tmp.name, "tx.csv")
        self.fn_rx = os.path.join(self.tmp.name, "rx.csv")
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
        self.L, self.extra_tx, self.extra_rx = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9])

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L, d2r = self.L, np.pi / 180
        self.assertEqual(list(L.keys()), LAYOUT_KEYS)
        self.assertEqual(L["name"], "")
        self.assertEqual(L["seed"], -1)
        npt.assert_array_equal(L["center_freq"], [2e9, 3.5e9])
        T, R = L["tx"], L["rx"]
        self.assertEqual(T["name"], ["BS1", "BS1", "BS2"])
        npt.assert_array_equal(T["sector"], [0, 1, 0])
        self.assertEqual(T["sector"].dtype, np.uint32)
        npt.assert_array_equal(T["pos"], [[0, 0, 500], [0, 0, 0], [25, 25, 25]])
        npt.assert_allclose(T["orientation"], np.array([[0, 0, 0], [10, 10, 0], [30, 150, 90]]) * d2r, atol=1e-14, rtol=0)
        self.assertEqual(T["speed"].shape, (0,))  # optional members stay empty
        self.assertEqual(T["extra_names"], ["note"])
        self.assertEqual(T["extra"], [["a", "", "b"]])
        self.assertEqual(T["scenpar_override"], {})
        self.assertEqual(R["name"], ["UT1", "UT2"])
        self.assertEqual(R["pos"].shape, (3, 5))
        npt.assert_array_equal(R["pos"][:, 4], [2, 1, 1.5])
        npt.assert_array_equal(R["speed"], [0.833, 0, 0, 0, 0])
        self.assertEqual(R["direction"].shape, (2, 5))
        npt.assert_allclose(R["direction"][:, 0], [120 * d2r, 0], atol=1e-14, rtol=0)
        self.assertTrue(np.all(np.isnan(R["direction"][0, 1:])))
        self.assertEqual(R["scenario"], ["UMa", "", "", "", "UMa_NLOS"])
        self.assertEqual(L["links"]["i_tx"].size, 0)
        self.assertEqual(L["links"]["method"], "")

    def test_antennas_and_indices(self):
        L, T, R = self.L, self.L["tx"], self.L["rx"]
        self.assertEqual(L["antenna_cols"], ["antenna", "panel"])
        self.assertEqual(L["antenna_args"], [['3gpp,"2,2,1,2"'], ["dipole,"]])
        npt.assert_array_equal(T["i_antenna"], [1, 1, 2])  # 1-based in all APIs, 0 = generic antenna
        npt.assert_array_equal(R["i_antenna"], [0, 0])
        self.assertEqual([len(a) for a in L["antenna"]], [2, 1])  # spaced array: one dict per band; co-located: all bands
        self.assertEqual(L["antenna"][0][1]["element_pos"].shape, (3, 8))  # panel 2 x 2 x 1 x 2
        self.assertEqual(L["antenna"][0][1]["center_freq"], 3.5e9)
        # Indices are 0-based
        npt.assert_array_equal(L["pairing"], [[0, 0, 1], [0, 1, 1]])
        self.assertEqual(L["pairing"].dtype, np.int64)
        npt.assert_array_equal(T["offset"], [0, 1, 2])
        npt.assert_array_equal(R["offset"], [0, 1])
        self.assertEqual(T["offset"].dtype, np.int64)

    def test_optional_inputs(self):
        L = self.L
        L3, _, _ = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9], eval_antennas=False)
        self.assertEqual(L3["antenna"], [])  # lines and indices are kept, no arrays
        self.assertEqual(L3["antenna_args"], L["antenna_args"])
        npt.assert_array_equal(L3["tx"]["i_antenna"], L["tx"]["i_antenna"])
        for center_freq in (None, np.array([2e9, 3.5e9], dtype=np.float32), (2e9, 3.5e9)):  # other types, None = from the files
            if center_freq is None:
                with self.assertRaisesRegex(RuntimeError, "frequenc"):  # an array spaced in wavelengths needs frequencies
                    quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=center_freq)
            else:
                Lt, _, _ = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=center_freq)
                npt.assert_array_equal(Lt["center_freq"], [2e9, 3.5e9])
        write_text(self.fn_tx, "name,x,y,z,center_freq\nA,0,0,10,1e9\nA,,,,2e9\n")  # frequencies from the file
        Lf, _, _ = quadriga_lib.layout.csv_read(self.fn_tx, "", eval_antennas=None)
        npt.assert_array_equal(Lf["center_freq"], [1e9, 2e9])

    def test_outputs_and_paths(self):
        L, extra_tx, extra_rx = quadriga_lib.layout.csv_read(pathlib.Path(self.fn_tx), self.fn_rx.encode(), center_freq=[2e9, 3.5e9])
        self.assertTrue(same(L, self.L))
        self.assertEqual(extra_tx, ["note"])
        self.assertEqual(extra_rx, [])
        fn_rx2 = os.path.join(self.tmp.name, "rx2.csv")
        write_text(fn_rx2, "name,x,y,z\nUT1,1,2,1.5\nUT2,3,4,1.5\n")
        Lr, _, _ = quadriga_lib.layout.csv_read("", fn_rx2)  # one side only
        self.assertEqual(Lr["tx"]["name"], [])
        self.assertEqual(Lr["rx"]["name"], ["UT1", "UT2"])
        self.assertEqual(Lr["pairing"].size, 0)  # all pairs

    def test_errors(self):
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # center_freq is keyword-only
            quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, [2e9, 3.5e9])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.csv_read(self.fn_tx)
        with self.assertRaisesRegex(TypeError, "Input 'fn_tx' must be a str or path-like object"):
            quadriga_lib.layout.csv_read(1, self.fn_rx)
        with self.assertRaisesRegex(TypeError, "Input 'eval_antennas' must be a bool or None"):
            quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9], eval_antennas=1)
        with self.assertRaisesRegex(ValueError, "center_freq"):  # C++: not ascending
            quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[3.5e9, 2e9])
        with self.assertRaisesRegex(RuntimeError, "missing.csv"):  # C++: the file name in the message
            quadriga_lib.layout.csv_read(os.path.join(self.tmp.name, "missing.csv"), "")


if __name__ == "__main__":
    unittest.main()
