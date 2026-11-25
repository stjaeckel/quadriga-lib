# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.get_channels_layout.
# Scope: argument and output handling, index base, defaults, stack, seed, config_dirs, error paths.
# The channels are tested in the Catch2 suite (test_get_channels_layout.cpp).
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
        self.L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L = self.L
        ch, index = quadriga_lib.layout.get_channels_layout(L, seed=5)  # the link states are drawn for this call only
        self.assertEqual([c["name"] for c in ch], ["A_U1", "B_U1", "A_U2", "B_U2"] * 2)  # Tx node fastest, then Rx node, then frequency
        npt.assert_array_equal(index, [[0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])  # 0-based
        self.assertEqual((index.shape, index.dtype), ((3, 8), np.int64))
        npt.assert_array_equal(ch[0]["center_freq"], [2e9])
        npt.assert_array_equal(ch[4]["center_freq"], [3.5e9])
        npt.assert_array_equal(ch[1]["tx_pos"][:, 0], [1000, 0, 25])
        self.assertEqual(ch[0]["coeff"][0].shape[:2], (2, 2))  # generic antennas: two ports each
        self.assertNotIn("path_gain", ch[0])
        self.assertEqual(L["links"]["i_tx"].size, 0)  # the layout is unchanged
        ch2, _ = quadriga_lib.layout.get_channels_layout(self.L2)  # the same with filled links
        self.assertTrue(same(ch2, ch))

    def test_optional_inputs(self):
        L, L2 = self.L, self.L2
        ch, _ = quadriga_lib.layout.get_channels_layout(L2, times=[0, 1, 2])
        self.assertEqual(len(ch[0]["coeff"]), 3)
        npt.assert_allclose(ch[0]["rx_pos"], [[100] * 3, [0, 2, 4], [1.5] * 3], atol=1e-12, rtol=0)  # U1 moves 2 m/s north
        chs, _ = quadriga_lib.layout.get_channels_layout(L2, times=np.array([0, 1, 2], dtype=np.float32), stack=True)
        self.assertEqual(chs[0]["coeff"].shape[3], 3)
        ch, _ = quadriga_lib.layout.get_channels_layout(L2, output="rays")
        self.assertIn("path_gain", ch[0])
        chp, _ = quadriga_lib.layout.get_channels_layout(L2, times=[0, 1], method="38.901", mobility="phase", overlap=0.2)
        self.assertEqual(len(chp[0]["coeff"]), 2)
        chq, _ = quadriga_lib.layout.get_channels_layout(L, method="quadriga", seed=5)  # another method needs empty links
        self.assertEqual(len(chq), 8)
        ch5, _ = quadriga_lib.layout.get_channels_layout(L, seed=5)
        ch6, _ = quadriga_lib.layout.get_channels_layout(L, seed=6)
        self.assertFalse(np.array_equal(ch5[0]["coeff"][0], ch6[0]["coeff"][0]))  # another seed, other channels
        chd, _ = quadriga_lib.layout.get_channels_layout(L, seed=5, config_dirs=[])
        self.assertTrue(same(chd, ch5))
        chn, _ = quadriga_lib.layout.get_channels_layout(L, times=None, output="taps", method="", mobility="auto", overlap=0.5,
                                                         seed=np.int8(5), config_dirs=None, stack=None)  # the defaults
        self.assertTrue(same(chn, ch5))
        Lo = quadriga_lib.layout.set_antenna(L, "omni")  # an omni-directional Tx antenna has one port
        cho, _ = quadriga_lib.layout.get_channels_layout(Lo, seed=5)
        self.assertEqual(cho[0]["coeff"][0].shape[:2], (2, 1))
        Lp = dict(L2, pairing=np.array([[1, 0], [0, 1]]))  # pairs (B, U1) and (A, U2), 0-based
        ch, index = quadriga_lib.layout.get_channels_layout(Lp)
        self.assertEqual([c["name"] for c in ch], ["B_U1", "A_U2", "B_U1", "A_U2"])
        npt.assert_array_equal(index, [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])

    def test_memory_order(self):
        # Arrays in Fortran order and strided views give the same channels as contiguous ones
        L2 = self.L2
        ch, _ = quadriga_lib.layout.get_channels_layout(L2)
        wide = np.zeros((3, 2 * L2["rx"]["pos"].shape[1]))
        wide[:, ::2] = L2["rx"]["pos"]
        Lf = dict(L2, tx=dict(L2["tx"], pos=np.asfortranarray(L2["tx"]["pos"])), rx=dict(L2["rx"], pos=wide[:, ::2]))
        self.assertFalse(Lf["rx"]["pos"].flags.f_contiguous or Lf["rx"]["pos"].flags.c_contiguous)
        chf, _ = quadriga_lib.layout.get_channels_layout(Lf)
        self.assertTrue(same(chf, ch))

    def test_config_dirs(self):
        # A user conf file is found in config_dirs; the wrapper adds the shipped folder
        L, L2 = self.L, self.L2
        folder = os.path.join(self.tmp.name, "conf")
        os.mkdir(folder)
        with open(L2["links"]["conf_files"][1]) as f:
            write_text(os.path.join(folder, "MyUMa.conf"), f.read())
        Lc = dict(L, tx=dict(L["tx"], scenario=["MyUMa", "MyUMa"]))
        chc, _ = quadriga_lib.layout.get_channels_layout(Lc, seed=5, config_dirs=folder)
        self.assertEqual(len(chc), 8)
        with self.assertRaisesRegex(ValueError, "scenario 'MyUMa' is neither a scenario family nor a conf file"):
            quadriga_lib.layout.get_channels_layout(Lc, seed=5)

    def test_errors(self):
        L, L2 = self.L, self.L2
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # times is keyword-only
            quadriga_lib.layout.get_channels_layout(L, [0, 1])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.get_channels_layout(1)
        with self.assertRaisesRegex(TypeError, "Input 'stack' must be a bool or None"):
            quadriga_lib.layout.get_channels_layout(L, stack=1)
        with self.assertRaises(ValueError):  # config_dirs must be str or a list of str
            quadriga_lib.layout.get_channels_layout(L, config_dirs=5)
        with self.assertRaisesRegex(ValueError, "output"):  # C++
            quadriga_lib.layout.get_channels_layout(L2, output="foo")
        with self.assertRaisesRegex(ValueError, "38.901"):  # C++: another method than that of the links
            quadriga_lib.layout.get_channels_layout(L2, method="quadriga")
        with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'layout.links'"):
            quadriga_lib.layout.get_channels_layout(dict(L2, links=dict(L2["links"], foo=1)))


if __name__ == "__main__":
    unittest.main()
