# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.eval_antennas.
# Scope: argument and output handling, defaults, in/out split, the round trip of every member of a layout, error paths.
# The evaluation rules are tested in the Catch2 suite (test_qd_layout.cpp).
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
        # 3GPP panel of 2 x 2 x 1 x 2 elements for the Tx nodes (not evaluated), a dipole for Rx node 0
        self.tmp = tempfile.TemporaryDirectory()
        self.fn_tx, self.fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(self.fn_tx, "name,x,y,z,antenna,panel\nA,0,0,25,3gpp,\"2,2,1,2\"\nB,500,0,25,3gpp,\"2,2,1,2\"\n")
        write_text(self.fn_rx, "name,x,y,z,antenna\nU1,10,20,1.5,dipole\nU2,30,40,1.5,\n")
        self.L = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, eval_antennas=False)[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L = self.L
        self.assertEqual(L["antenna"], [])
        self.assertEqual(L["center_freq"].size, 0)
        Le = quadriga_lib.layout.eval_antennas(L, center_freq=[2e9, 3.5e9])  # the arrays at the given frequencies
        npt.assert_array_equal(Le["center_freq"], [2e9, 3.5e9])
        self.assertEqual([len(a) for a in Le["antenna"]], [2, 1])  # spaced array: one per band; one dipole for all bands
        self.assertEqual(Le["antenna"][0][0]["element_pos"].shape, (3, 8))
        spread = [np.ptp(b["element_pos"][1]) for b in Le["antenna"][0]]
        npt.assert_allclose(spread[0] / (299792458 / 2e9), spread[1] / (299792458 / 3.5e9), atol=1e-12, rtol=0)  # in wavelengths
        npt.assert_array_equal(Le["tx"]["i_antenna"], [1, 1])
        npt.assert_array_equal(Le["rx"]["i_antenna"], [2, 0])
        self.assertEqual(L["center_freq"].size, 0)  # the input stays unchanged
        self.assertEqual(L["antenna"], [])

    def test_optional_inputs(self):
        Le = quadriga_lib.layout.eval_antennas(self.L, center_freq=[2e9, 3.5e9])
        self.assertTrue(same(quadriga_lib.layout.eval_antennas(Le), Le))  # omitted and None keep the frequencies
        self.assertTrue(same(quadriga_lib.layout.eval_antennas(Le, center_freq=None), Le))
        Le4 = quadriga_lib.layout.eval_antennas(Le, center_freq=np.array([1e9, 3.5e9], dtype=np.float32))  # a new list regenerates all lines
        npt.assert_array_equal(Le4["center_freq"], [1e9, 3.5e9])
        self.assertEqual(Le4["antenna"][0][0]["center_freq"], 1e9)
        self.assertTrue(same(quadriga_lib.layout.eval_antennas(layout_in=self.L, center_freq=(2e9, 3.5e9)), Le))

    def test_round_trip_of_all_members(self):
        # A layout with every column, antennas and links: the conversion to C++ and back loses no member
        write_text(self.fn_tx, "name,sector,x,y,z,heading,tilt,antenna,panel,seed,layout_seed,wrap,d_clutter,r_clutter,h_clutter,"
                   "scenpar.DS_VS,scenario,note\n"
                   "A,0,0,0,25,30,10,3gpp,\"2,2,1,2\",7,42,\"1000,0;0,1000\",5,0.5,3,0.2,UMa,x\n"
                   "A,1,,,,150,,,,,,,,,,,,\n"
                   "B,0,1000,0,25,90,0,omni,,,,,,,,,UMa,y\n")
        write_text(self.fn_rx, "name,x,y,z,heading,speed,direction_az,direction_el,time,indoor,floor,d2d_in,relative,pairing\n"
                   "U1,100,0,1.5,0,,,,0,0,,,1,\n"
                   "U1,110,0,1.5,,,,,1,,,,,\n"
                   "U2,900,0,1.5,90,2,45,10,,0,,,,A\n")
        Lr = quadriga_lib.layout.csv_read(self.fn_tx, self.fn_rx, center_freq=[2e9, 3.5e9])[0]
        Lr = quadriga_lib.layout.init_link_states(Lr, seed=5)
        Lr["name"] = "rich"
        npt.assert_array_equal(Lr["tx"]["seed"], [7, 7, -1])
        npt.assert_array_equal(Lr["tx"]["clutter"], np.array([[5, 5, np.nan], [0.5, 0.5, np.nan], [3, 3, np.nan]]))
        self.assertEqual(list(Lr["tx"]["scenpar_override"].keys()), ["DS_VS"])
        npt.assert_array_equal(Lr["tx"]["scenpar_override"]["DS_VS"], [0.2, 0.2, np.nan])
        npt.assert_array_equal(Lr["rx"]["relative"], [1, 0])
        self.assertEqual(Lr["seed"], 5)
        npt.assert_array_equal(Lr["wrap"], [[1000, 0], [0, 1000]])
        npt.assert_array_equal(Lr["pairing"], [[0, 1, 0], [0, 0, 1]])  # (site, terminal), 0-based: U1 with A and B, U2 with A
        npt.assert_array_equal(Lr["rx"]["time"], [0, 1, np.nan])
        self.assertEqual(Lr["tx"]["extra"], [["x", "", "y"]])
        self.assertGreater(Lr["links"]["i_tx"].size, 0)
        self.assertGreater(len(Lr["antenna"]), 0)
        self.assertTrue(same(quadriga_lib.layout.eval_antennas(Lr), Lr))
        # 64-bit members as Python lists and ints (e.g. from JSON) are read exactly, also above 2^53
        self.assertGreater(Lr["links"]["gbsm_seed"].max(), 2**53)
        links = dict(Lr["links"], gbsm_seed=Lr["links"]["gbsm_seed"].tolist(), seed=int(Lr["links"]["seed"]))
        Lj = dict(Lr, tx=dict(Lr["tx"], seed=Lr["tx"]["seed"].tolist()), links=links)
        self.assertTrue(same(quadriga_lib.layout.eval_antennas(Lj), Lr))
        # The links can be discarded (None or an empty dict); missing and empty members are empty
        for links in ({}, None):
            Ld = quadriga_lib.layout.eval_antennas(dict(Lr, links=links))
            self.assertEqual(Ld["links"]["i_tx"].size, 0)
            self.assertEqual(Ld["links"]["method"], "")
            self.assertEqual(Ld["links"]["conf_names"], [])
            self.assertEqual(Ld["links"]["seed"], -1)
            self.assertEqual(Ld["links"]["anchor_freq"], 2e9)
            self.assertTrue(same(Ld["tx"], Lr["tx"]))
        Lm = quadriga_lib.layout.eval_antennas({"center_freq": np.array([2e9])})  # every other key is missing
        npt.assert_array_equal(Lm["center_freq"], [2e9])
        self.assertEqual(Lm["tx"]["name"], [])
        self.assertTrue(Lm["rx"]["pos"].size == 0 and Lm["tx"]["offset"].size == 0 and Lm["pairing"].size == 0 and Lm["antenna"] == [])
        self.assertEqual(Lm["seed"], -1)
        Lz = dict(Lr, tx=dict(Lr["tx"], clutter=None, scenpar_override=None), rx=dict(Lr["rx"], time=np.array([])))
        Lz = quadriga_lib.layout.eval_antennas(Lz)
        self.assertTrue(Lz["tx"]["clutter"].size == 0 and Lz["rx"]["time"].size == 0 and Lz["tx"]["scenpar_override"] == {})
        for part in ("tx", "rx", "links"):  # unknown keys of the nested dicts name their place
            Lu = dict(Lr)
            Lu[part] = dict(Lr[part], foo=1)
            with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'layout_in.%s'." % part):
                quadriga_lib.layout.eval_antennas(Lu)

    def test_errors(self):
        L = self.L
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # center_freq is keyword-only
            quadriga_lib.layout.eval_antennas(L, [2e9, 3.5e9])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.eval_antennas(1)
        for key, value in (("anchor_freq", "a"), ("layout_hash", -1), ("layout_hash", 2**64), ("layout_hash", 1.5), ("seed", 2**63)):
            with self.assertRaisesRegex(ValueError, "Entry 'layout_in.links.%s'" % key):  # wrong types and values in links
                quadriga_lib.layout.eval_antennas(dict(L, links={key: value}))
        for band in ({}, {"center_freq": 3.5e9}):  # an antenna band without the pattern keys
            with self.assertRaisesRegex(ValueError, r"Entry 'layout_in\.antenna\[0\]\[1\]' is not an arrayant dict: the key 'e_theta_re' is missing"):
                quadriga_lib.layout.eval_antennas(dict(L, antenna=[[quadriga_lib.arrayant.generate_antenna("omni"), band]]))
        with self.assertRaisesRegex(ValueError, "Entry 'layout_in.antenna\\[0\\]' must be a list of arrayant dicts"):
            quadriga_lib.layout.eval_antennas(dict(L, antenna=[5]))
        with self.assertRaisesRegex(ValueError, "Entry 'antenna' of 'layout_in' must be a list"):
            quadriga_lib.layout.eval_antennas(dict(L, antenna=5))
        with self.assertRaisesRegex(ValueError, "Entry 'pairing' of 'layout_in' must have 2 rows"):
            quadriga_lib.layout.eval_antennas(dict(L, pairing=np.array([0, 1, 2])))
        with self.assertRaisesRegex(ValueError, "Entry 'layout_in.seed' must be -1 or a non-negative integer"):
            quadriga_lib.layout.eval_antennas(dict(L, seed=-2))
        with self.assertRaisesRegex(ValueError, r"Input 'layout_in\.tx\.offset' must contain integers from 0 to"):  # a negative index
            quadriga_lib.layout.eval_antennas(dict(L, tx=dict(L["tx"], offset=np.array([-1, 0]))))
        with self.assertRaisesRegex(ValueError, "wavelengths"):  # C++: an array spaced in wavelengths needs a frequency
            quadriga_lib.layout.eval_antennas(L)
        with self.assertRaisesRegex(ValueError, "center_freq"):  # C++: not ascending
            quadriga_lib.layout.eval_antennas(L, center_freq=[3.5e9, 2e9])


if __name__ == "__main__":
    unittest.main()
