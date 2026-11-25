# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.get_channels_from_rays.
# Scope: argument and output handling, index base, defaults, stack, rays of several forms, error paths.
# The channels are tested in the Catch2 suite (test_get_channels_layout.cpp).
# Physics validation lives in the Catch2 suite.

import sys
import os
import copy
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
        self.rays = quadriga_lib.layout.generate_rays(self.L2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        L2, rays = self.L2, self.rays
        out = quadriga_lib.layout.get_channels_from_rays(L2, rays)
        self.assertIsInstance(out, tuple)
        ch, index = out
        self.assertEqual(len(ch), 8)  # Tx node fastest, then Rx node, then frequency
        self.assertEqual([c["name"] for c in ch], ["A_U1", "B_U1", "A_U2", "B_U2"] * 2)
        npt.assert_array_equal(index, [[0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])  # 0-based
        self.assertEqual((index.shape, index.dtype), ((3, 8), np.int64))
        npt.assert_array_equal(ch[0]["center_freq"], [2e9])
        npt.assert_array_equal(ch[4]["center_freq"], [3.5e9])
        npt.assert_array_equal(ch[1]["tx_pos"][:, 0], [1000, 0, 25])
        npt.assert_array_equal(ch[2]["rx_pos"][:, 0], [900, 0, 1.5])
        self.assertEqual(len(ch[0]["coeff"]), 1)  # one snapshot
        n_path = ch[0]["coeff"][0].shape[2]
        self.assertEqual(ch[0]["coeff"][0].shape, (2, 2, n_path))  # the generic antenna has two ports
        self.assertEqual(ch[0]["coeff"][0].dtype, np.complex128)
        self.assertNotIn("path_gain", ch[0])  # taps
        self.assertEqual(ch[0]["delay"][0][0, 0, 0], 0)  # the LOS tap is the delay reference
        self.assertTrue(np.all(np.isfinite(ch[0]["coeff"][0])))
        ch2, index2 = quadriga_lib.layout.get_channels_layout(L2)  # the stages in one call
        self.assertTrue(same(ch2, ch))
        npt.assert_array_equal(index2, index)

    def test_optional_inputs(self):
        L2, rays = self.L2, self.rays
        ch, index = quadriga_lib.layout.get_channels_from_rays(L2, rays[:2], i_link=[0, 1])  # terminal U1 with both sites
        self.assertEqual(len(ch), 4)
        npt.assert_array_equal(index, [[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1]])
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, [rays[2], rays[0]], i_link=[2, 0])  # another order: the channels follow the nodes
        self.assertEqual([c["name"] for c in ch], ["A_U1", "A_U2", "A_U1", "A_U2"])
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, times=[0, 1, 2])  # U1 moves 2 m/s north
        self.assertEqual(len(ch[0]["coeff"]), 3)
        npt.assert_allclose(ch[0]["rx_pos"], [[100] * 3, [0, 2, 4], [1.5] * 3], atol=1e-12, rtol=0)
        npt.assert_array_equal(ch[2]["rx_pos"], np.tile([[900], [0], [1.5]], (1, 3)))  # a static terminal is repeated
        chs, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, times=[0, 1, 2], stack=True)  # one array with the snapshot last
        self.assertEqual(chs[0]["coeff"].shape, ch[0]["coeff"][0].shape + (3,))
        npt.assert_array_equal(chs[0]["coeff"][..., 1], ch[0]["coeff"][1])
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, output="rays")
        self.assertIn("path_gain", ch[0])
        self.assertEqual(ch[0]["path_gain"][0].shape[0], ch[0]["coeff"][0].shape[2])
        rs = quadriga_lib.layout.generate_rays(L2, scatterers=True)
        chd, _ = quadriga_lib.layout.get_channels_from_rays(L2, rs, times=[0, 1, 2], mobility="drift")  # needs the scatterers
        chp, _ = quadriga_lib.layout.get_channels_from_rays(L2, rs, times=[0, 1, 2], mobility="phase")
        self.assertEqual(chp[0]["coeff"][0].shape, chd[0]["coeff"][0].shape)
        cho, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, oxygen=True)
        self.assertEqual(len(cho), 8)
        cha, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, use_absolute_delays=True)  # the LOS delay in all delays
        npt.assert_allclose(cha[0]["delay"][0][0, 0, 0], np.hypot(100, 23.5) / 299792458, atol=1e-15, rtol=0)
        chv, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, overlap=0.2)
        self.assertEqual(len(chv), 8)
        Lp = dict(L2, pairing=np.array([[1], [0]]))  # only the pair (site B, terminal U1), 0-based
        ch, index = quadriga_lib.layout.get_channels_from_rays(Lp, rays)
        self.assertEqual([c["name"] for c in ch], ["B_U1", "B_U1"])
        npt.assert_array_equal(index, [[1, 1], [0, 0], [0, 1]])
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, rays, i_link=None, times=None, output="taps", mobility="auto", oxygen=None,
                                                      use_absolute_delays=None, overlap=0.5, stack=None)  # the defaults
        self.assertEqual(len(ch), 8)

    def test_rays_forms_and_edits(self):
        L2, rays = self.L2, self.rays
        rays_before, L2_before = copy.deepcopy(rays), copy.deepcopy(L2)
        re = [[dict(r) for r in link] for link in rays]
        re[0][0]["power"] = np.zeros_like(re[0][0]["power"])  # an edited ray set: no power in band 0 of link 0
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, re, output="rays")
        npt.assert_array_equal(ch[0]["path_gain"][0], 0)
        self.assertTrue(np.any(ch[2]["path_gain"][0] > 0))
        self.assertTrue(same(rays, rays_before))  # the rays and the layout of the caller stay unchanged
        self.assertTrue(same(L2, L2_before))
        # Ray dicts with other dtypes and tuples of rays; one ray dict alone for a link with one object ('quadriga')
        rs = [tuple(dict(r, power=r["power"].astype(np.float32)) for r in link) for link in rays]
        ch, _ = quadriga_lib.layout.get_channels_from_rays(L2, tuple(rs))
        self.assertEqual(len(ch), 8)
        Lq = quadriga_lib.layout.init_link_states(self.L, method="quadriga", seed=5)
        rq = quadriga_lib.layout.generate_rays(Lq)
        chq, _ = quadriga_lib.layout.get_channels_from_rays(Lq, rq)
        chd, _ = quadriga_lib.layout.get_channels_from_rays(Lq, [link[0] for link in rq])
        self.assertTrue(same(chd, chq))

    def test_errors(self):
        L2, rays = self.L2, self.rays
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # i_link is keyword-only
            quadriga_lib.layout.get_channels_from_rays(L2, rays, [0])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.get_channels_from_rays(L2)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.get_channels_from_rays(L2, None)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.get_channels_from_rays(1, rays)
        with self.assertRaisesRegex(ValueError, "Input 'rays' must be a list"):
            quadriga_lib.layout.get_channels_from_rays(L2, 5)
        with self.assertRaisesRegex(ValueError, r"Input 'rays\[0\]' must be a ray dict or a list of ray dicts"):
            quadriga_lib.layout.get_channels_from_rays(L2, [1])
        bad = [[{k: v for k, v in r.items() if k != "delay"} for r in link] for link in rays]
        with self.assertRaisesRegex(ValueError, r"Entry 'delay' of 'rays\[0\]\[0\]' is missing"):
            quadriga_lib.layout.get_channels_from_rays(L2, bad)
        with self.assertRaisesRegex(ValueError, "Input 'i_link'"):  # negative index
            quadriga_lib.layout.get_channels_from_rays(L2, rays, i_link=-1)
        with self.assertRaisesRegex(TypeError, "Input 'oxygen' must be a bool or None"):
            quadriga_lib.layout.get_channels_from_rays(L2, rays, oxygen=1)
        with self.assertRaisesRegex(ValueError, "rays"):  # C++: one entry per link of i_link
            quadriga_lib.layout.get_channels_from_rays(L2, rays, i_link=[1])
        with self.assertRaisesRegex(ValueError, "output"):  # C++
            quadriga_lib.layout.get_channels_from_rays(L2, rays, output="foo")
        with self.assertRaisesRegex(ValueError, "drift"):  # C++: no scatterers
            quadriga_lib.layout.get_channels_from_rays(L2, rays, mobility="drift")
        with self.assertRaisesRegex(ValueError, "init_link_states"):  # C++: no links
            quadriga_lib.layout.get_channels_from_rays(self.L, rays)


if __name__ == "__main__":
    unittest.main()
