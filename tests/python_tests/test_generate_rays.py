# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.generate_rays.
# Scope: argument and output handling, index base, defaults, ray dicts, error paths.
# The ray generation is tested in the Catch2 suite (test_init_link_states.cpp).
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


RAY_KEYS = ["method", "tx_pos", "rx_pos", "center_freq", "delay", "power", "aod", "eod", "aoa", "eoa", "polmat", "path_type", "tap",
            "gr_eps", "fbs_pos", "lbs_pos", "fallback"]


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
        L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)
        rays = quadriga_lib.layout.generate_rays(L2)
        self.assertIsInstance(rays, list)
        self.assertEqual(len(rays), 4)  # one list per link, one ray dict per band (TR methods)
        self.assertEqual([len(r) for r in rays], [2] * 4)
        self.assertEqual(list(rays[0][0].keys()), RAY_KEYS)
        r = rays[0][1]  # link 0 (site A, terminal U1), band 3.5 GHz
        n = r["delay"].size
        self.assertGreater(n, 1)
        self.assertEqual(r["method"], "38.901")
        npt.assert_array_equal(r["tx_pos"], [0, 0, 25])
        npt.assert_array_equal(r["rx_pos"], [100, 0, 1.5])
        npt.assert_array_equal(r["center_freq"], [3.5e9])
        self.assertEqual(r["power"].shape, (n, 1))
        self.assertEqual(r["polmat"].shape, (8, n, 1))
        self.assertEqual((r["path_type"].dtype, r["tap"].dtype), (np.uint32, np.uint32))
        self.assertEqual(r["path_type"][0], 0)  # the LOS ray comes first, its delay is the reference
        self.assertEqual(r["delay"][0], 0)
        self.assertTrue(np.all(r["path_type"] <= 2) and np.all(r["power"] >= 0))
        self.assertEqual(r["fbs_pos"].size + r["lbs_pos"].size + r["fallback"].size, 0)  # no scatterers requested
        npt.assert_array_equal(rays[1][0]["tx_pos"], [1000, 0, 25])  # link 1: site B, terminal U1
        npt.assert_array_equal(rays[2][0]["rx_pos"], [900, 0, 1.5])  # link 2: site A, terminal U2
        npt.assert_array_equal(rays[0][0]["center_freq"], [2e9])
        npt.assert_array_equal(rays[0][0]["path_type"], r["path_type"])  # the bands share the geometry of the clusters

    def test_optional_inputs(self):
        L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)
        rays = quadriga_lib.layout.generate_rays(L2)
        rs = quadriga_lib.layout.generate_rays(L2, i_link=[2, 1])  # 0-based links in the given order
        self.assertEqual(len(rs), 2)
        self.assertTrue(same(rs[0], rays[2]) and same(rs[1], rays[1]))  # the values of a link do not depend on the other links
        rs = quadriga_lib.layout.generate_rays(L2, scatterers=True)
        n = rs[0][0]["delay"].size
        self.assertEqual(rs[0][0]["fbs_pos"].shape, (3, n))
        self.assertEqual(rs[0][0]["lbs_pos"].shape, (3, n))
        npt.assert_array_equal(rs[0][0]["delay"], rays[0][0]["delay"])
        rs = quadriga_lib.layout.generate_rays(L2, i_link=1, scatterers=True)
        self.assertEqual(len(rs), 1)
        npt.assert_array_equal(rs[0][1]["tx_pos"], [1000, 0, 25])
        self.assertTrue(same(quadriga_lib.layout.generate_rays(L2, i_link=None, scatterers=None), rays))
        rs = quadriga_lib.layout.generate_rays(layout=L2, i_link=np.array([2, 1], dtype=np.int32))
        self.assertTrue(same(rs[0], rays[2]))

    def test_input_is_not_modified(self):
        L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)
        before = copy.deepcopy(L2)
        quadriga_lib.layout.generate_rays(L2, scatterers=True)
        self.assertTrue(same(L2, before))

    def test_quadriga_method(self):
        Lq = quadriga_lib.layout.init_link_states(self.L, method="quadriga", seed=5)
        rq = quadriga_lib.layout.generate_rays(Lq, i_link=0)
        self.assertEqual((len(rq), len(rq[0])), (1, 1))  # one object with all frequencies
        r = rq[0][0]
        self.assertEqual(r["method"], "quadriga")
        npt.assert_array_equal(r["center_freq"], [2e9, 3.5e9])
        self.assertEqual(r["power"].shape, (r["delay"].size, 2))
        self.assertEqual(r["fbs_pos"].shape, (3, r["delay"].size))  # always with scatterers

    def test_errors(self):
        L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # i_link is keyword-only
            quadriga_lib.layout.generate_rays(L2, [0])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.generate_rays(1)
        with self.assertRaisesRegex(ValueError, "Input 'i_link'"):  # negative index
            quadriga_lib.layout.generate_rays(L2, i_link=-1)
        with self.assertRaisesRegex(TypeError, "Input 'scatterers' must be a bool or None"):
            quadriga_lib.layout.generate_rays(L2, scatterers=1)
        with self.assertRaisesRegex(ValueError, "i_link"):  # C++: out of range
            quadriga_lib.layout.generate_rays(L2, i_link=4)
        with self.assertRaisesRegex(ValueError, "i_link"):  # C++: repeated
            quadriga_lib.layout.generate_rays(L2, i_link=[1, 1])
        with self.assertRaisesRegex(ValueError, "init_link_states"):  # C++: no links
            quadriga_lib.layout.generate_rays(self.L)
        Lm = dict(L2, tx=dict(L2["tx"], pos=L2["tx"]["pos"] + 5))
        with self.assertRaisesRegex(ValueError, "The layout has changed since 'init_link_states'"):  # C++
            quadriga_lib.layout.generate_rays(Lm)


if __name__ == "__main__":
    unittest.main()
