# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.channel.tr38901.
# Scope: argument and output handling, index base, defaults, keywords, stack, error paths.
# The channels are tested in the Catch2 suite (test_get_channels_tr38901.cpp).

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


class test_case(unittest.TestCase):

    def setUp(self):
        self.bs = quadriga_lib.arrayant.generate_antenna("3gpp", 10.0, center_freq=3.5e9, panel=[2, 1, 1, 1])
        self.ut = quadriga_lib.arrayant.generate_antenna("xpol", 10.0)

    def test_basic(self):
        # One site with 3 sectors, 2 UTs per cell: 6 channels, UT fastest; 0-based index
        chan, gain_dB, layout, index = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", 3.5e9, n_ut=2, seed=7, grid=[0])
        self.assertEqual(len(chan), 6)
        self.assertEqual(gain_dB.shape, (6, 3))
        self.assertEqual(index.shape, (3, 6))
        self.assertEqual(index.dtype, np.int64)
        npt.assert_array_equal(index[1, :], np.arange(6))
        npt.assert_array_equal(index[2, :], np.zeros(6))
        npt.assert_array_equal(index[0, :], np.argmax(gain_dB, axis=1))
        self.assertEqual(chan[0]["coeff"][0].shape[:2], (2, 2))    # 2 UT ports (xpol), 2 BS ports
        self.assertTrue(np.iscomplexobj(chan[0]["coeff"][0]))
        self.assertEqual(layout["links"]["method"], "38.901")
        self.assertEqual(layout["pairing"].shape, (2, 6))

    def test_same_as_layout_pipeline(self):
        # The returned layout gives the same channels through get_channels_layout (serving sectors selected)
        chan, gain_dB, layout, index = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, seed=3, grid=[0])
        ch2, idx2 = quadriga_lib.layout.get_channels_layout(layout)
        keep = [c for c in range(idx2.shape[1]) if idx2[0, c] == index[0, idx2[1, c]]]
        self.assertEqual(len(keep), len(chan))
        for a, c in zip(chan, keep):
            npt.assert_array_equal(a["coeff"][0], ch2[c]["coeff"][0])

    def test_tuples(self):
        # The layout tuples reach generate_layout
        chan, gain_dB, layout, index = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", n_ut=3, seed=5, spacing=[200],
                                                                    grid=[0, np.nan, 1], bs=[30], ut=[1.8], indoor=[0.0])
        L = quadriga_lib.layout.generate_layout("38.901_UMa", 3, [3.5e9], spacing=[200], grid=[0, np.nan, 1], bs=[30], ut=[1.8],
                                                indoor=[0.0], seed=5)
        self.assertEqual(gain_dB.shape, (3, 1))
        npt.assert_array_equal(layout["tx"]["pos"], L["tx"]["pos"])
        npt.assert_array_equal(layout["rx"]["pos"], L["rx"]["pos"])
        self.assertTrue(np.all(layout["rx"]["indoor"] == 0))
        with self.assertRaisesRegex(ValueError, "'indoor'"):
            quadriga_lib.channel.tr38901(self.bs, self.ut, "InH_Open", indoor=[0.5])

    def test_optional_inputs(self):
        kw = dict(n_ut=1, seed=5, grid=[0])
        a = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", [3.5e9, 28e9], **kw)[0]
        self.assertEqual(len(a), 6)
        self.assertEqual(a[3]["center_freq"], 28e9)
        b = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, time_grid=[0.01, 5e-3], ut=[np.nan, np.nan, 10.0], **kw)[0]
        self.assertEqual(len(b[0]["coeff"]), 3)
        npt.assert_allclose(np.linalg.norm(b[0]["rx_pos"][:, 2] - b[0]["rx_pos"][:, 0]), 0.1, atol=1e-9)
        self.assertEqual(len(quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", time_grid=[0.002], **kw)[0][0]["coeff"]), 3)
        c = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, los="nlos", output="rays", link_gain=False, **kw)
        self.assertTrue(np.all(c[2]["links"]["los"] == 0))
        self.assertIn("path_gain", c[0][0])
        c_on = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, los="nlos", output="rays", **kw)[0]
        self.assertGreater(np.sum(np.abs(c[0][0]["coeff"][0]) ** 2), 1e4 * np.sum(np.abs(c_on[0]["coeff"][0]) ** 2))  # no path loss
        d = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, indoor=[0.0], attach="rsrp", **kw)
        d_ls = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, indoor=[0.0], **kw)
        self.assertFalse(np.allclose(d[1], d_ls[1]))  # RSRP and coupling gain differ
        o0 = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", 3.5e9, indoor=[1.0], o2i_low=0.0, **kw)[1]
        o1 = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", 3.5e9, indoor=[1.0], o2i_low=1.0, **kw)[1]
        self.assertTrue(np.all(o1 > o0))  # the low-loss building model at all UTs
        f = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, uplink=True, config_dirs=None, **kw)[0]
        self.assertEqual(f[0]["coeff"][0].shape[:2], (2, 2))
        npt.assert_array_equal(f[0]["tx_pos"], quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, **kw)[0][0]["rx_pos"])
        g = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, stack=True, **kw)[0]
        self.assertEqual(g[0]["coeff"].ndim, 4)
        s, _, _, i_s = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, links="site", **kw)
        self.assertEqual(len(s), 9)  # the 3 sectors of the serving site per UT
        npt.assert_array_equal(i_s[1, :], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        a7, g7, L7, i7 = quadriga_lib.channel.tr38901(self.bs, self.ut, "UMi", 3.5e9, n_ut=1, seed=5, grid=[1], links="all")
        self.assertEqual(len(a7), 21 * 21)  # 7 sites x 3 sectors to 21 UTs
        self.assertEqual(L7["pairing"].shape[1], 0)

    def test_defaults_and_keywords(self):
        # Positional center_freq; the default is 3.5 GHz; other optional arguments are keyword-only; None = default
        chan = quadriga_lib.channel.tr38901(self.bs, self.ut, "InH_Open", None, seed=1, spacing=None, grid=None, area=None, bs=None, ut=None,
                                            indoor=None, time_grid=None, config_dirs=None)[0]
        self.assertEqual(len(chan), 12)
        self.assertEqual(chan[0]["center_freq"], 3.5e9)
        with self.assertRaises(TypeError):
            quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", 3.5e9, 2)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "Input 'time_grid' must have at most 2 entries"):
            quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", grid=[0], time_grid=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Input 'los' must be 'auto', 'los' or 'nlos'."):
            quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", los="LOS")
        with self.assertRaisesRegex(TypeError, "Input 'link_gain' must be a bool or None."):
            quadriga_lib.channel.tr38901(self.bs, self.ut, "UMa", link_gain="yes")


if __name__ == "__main__":
    unittest.main()
