# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_scatterers.
# Scope: argument and output handling, dict sharing rules, validity scales, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_rays_solve_scatterers.cpp, test_gbsm_rays.cpp).

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

gbsm = quadriga_lib.gbsm
CONF_DIR = os.path.join(current_dir, "../../data/config")
TX = np.array([[0.0, 0.0, 10.0]])
RX = np.array([[60.0, 10.0, 1.5], [-30.0, 80.0, 1.5]])
FC = np.array([3.5e9, 28e9])
KEYS = ["method", "tx_pos", "rx_pos", "center_frequency", "delay", "power", "aod", "eod", "aoa", "eoa", "M",
        "path_type", "tap", "gr_eps", "fbs_pos", "lbs_pos", "fallback"]


def make_rays():
    # "quadriga" rays of two links in UMi LOS
    sp = gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
    phase = gbsm.init_phases(sp, "lsf", 3)
    lsp, _, _, _, mu, sig = gbsm.lsp_generate(sp, FC, TX, RX, gbsm.draw(sp, "lsf", phase, TX, RX))
    rnd = gbsm.draw(sp, "cluster", gbsm.init_phases(sp, "cluster", 3), TX, RX)
    cl = gbsm.ssf_clusters(sp, TX, RX, lsp, rnd, "quadriga")
    rr = gbsm.draw(sp, "ray", gbsm.init_phases(sp, "ray", 3), TX, RX, center_freq=FC)
    return gbsm.ssf_rays(sp, TX, RX, FC, lsp, mu, sig, *cl[:6], rr, "quadriga")


class test_case(unittest.TestCase):

    def setUp(self):
        self.rays = make_rays()
        self.n_ray = self.rays[0]["delay"].size
        self.out = gbsm.ssf_scatterers(self.rays[0])

    def test_basic_call(self):
        self.assertIsInstance(self.out, tuple)
        self.assertEqual(len(self.out), 3)
        r, b_med, b_eff = self.out
        n_ray = self.n_ray
        self.assertIsInstance(r, dict)
        self.assertEqual(list(r.keys()), KEYS)
        self.assertEqual(r["fbs_pos"].shape, (3, n_ray))
        self.assertEqual(r["lbs_pos"].shape, (3, n_ray))
        self.assertEqual(r["fallback"].shape, (n_ray,))
        self.assertEqual(r["fallback"].dtype, np.uint32)
        self.assertTrue(np.all(r["fallback"] <= 1))
        npt.assert_allclose(r["fbs_pos"][:, 0], (TX[0] + RX[0]) / 2, atol=1e-12, rtol=0)  # LOS midpoint
        self.assertEqual(self.rays[0]["fbs_pos"].size, 0)  # input unchanged
        for k in KEYS[:14]:
            self.assertIs(r[k], self.rays[0][k])  # unchanged arrays shared, no copy
        self.assertTrue(np.shares_memory(r["delay"], self.rays[0]["delay"]))
        self.assertEqual(b_med.shape, (2, 2))
        self.assertEqual(b_eff.shape, (2, 2))
        self.assertEqual(b_med.dtype, np.float64)
        self.assertTrue(np.all(b_med > 0) and np.all(b_eff > 0))

    def test_validity_scales(self):
        r, b_med, b_eff = self.out
        ant = quadriga_lib.arrayant.generate("omni")
        for k in range(2):
            par = gbsm.single_link_channel(ant, ant, r, np.array([0.0]), "drift", "taps", k)[1]
            npt.assert_array_equal(b_med[:, k], par["b_med"])
            npt.assert_array_equal(b_eff[:, k], par["b_eff"])

    def test_d_min_and_solved_input(self):
        r, b_med, _ = self.out
        r2, m2, _ = gbsm.ssf_scatterers(self.rays[0], 2.0)
        npt.assert_array_equal(r2["fbs_pos"], r["fbs_pos"])
        npt.assert_array_equal(m2, b_med)
        r2 = gbsm.ssf_scatterers(rays=self.rays[0], d_min=2)[0]
        npt.assert_array_equal(r2["lbs_pos"], r["lbs_pos"])
        fbs0, lbs0, fb0 = r["fbs_pos"].copy(), r["lbs_pos"].copy(), r["fallback"].copy()
        r5 = gbsm.ssf_scatterers(r, 5.0)[0]  # 2 rays fall back at 5 m, none at 2 m
        npt.assert_array_equal(r["fbs_pos"], fbs0)
        npt.assert_array_equal(r["lbs_pos"], lbs0)
        npt.assert_array_equal(r["fallback"], fb0)
        self.assertGreater(np.max(np.abs(r5["fbs_pos"] - fbs0)), 1e-3)
        self.assertGreater(np.sum(r5["fallback"]), np.sum(fb0))
        self.assertFalse(np.shares_memory(r5["fbs_pos"], r["fbs_pos"]))
        self.assertTrue(np.shares_memory(r5["delay"], r["delay"]))

    def test_dict_handling(self):
        d = {k: v for k, v in self.rays[0].items() if k not in ["gr_eps", "fbs_pos", "lbs_pos", "fallback"]}
        d["path_type"] = d["path_type"].astype(np.int64)
        d["tap"] = list(d["tap"].astype(float))
        d["delay"] = np.ascontiguousarray(d["delay"][::-1])[::-1]  # non-contiguous view
        d["M"] = np.ascontiguousarray(d["M"])
        r, b_med, _ = gbsm.ssf_scatterers(d)
        self.assertEqual(list(r.keys()), KEYS)  # all keys on output
        self.assertEqual(r["gr_eps"].size, 0)
        self.assertEqual(r["gr_eps"].dtype, np.complex128)
        npt.assert_array_equal(r["fbs_pos"], self.out[0]["fbs_pos"])
        npt.assert_array_equal(b_med, self.out[1])
        d["gr_eps"] = None
        npt.assert_array_equal(gbsm.ssf_scatterers(d)[0]["lbs_pos"], self.out[0]["lbs_pos"])

    def test_errors(self):
        r = self.rays[0]
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers({k: v for k, v in r.items() if k != "delay"})
        self.assertIn("Entry 'delay' of 'rays' is missing.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(dict(r, foo=1))
        self.assertIn("Entry 'foo' of 'rays' is unknown.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(dict(r, method=1))
        self.assertIn("Entry 'method' of 'rays' must be a str.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(dict(r, tap=-np.ones(self.n_ray)))
        self.assertIn("Entries in 'rays.tap' must be non-negative integers", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(dict(r, delay=np.full(self.n_ray, np.nan)))
        self.assertIn("Input 'rays.delay' must be finite.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(dict(r, gr_eps=np.ones(3)))
        self.assertIn("'rays.gr_eps_re' and 'rays.gr_eps_im'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_scatterers(r, 0.0)
        self.assertIn("Input 'd_min' must be finite and > 0.", str(ctx.exception))
        with self.assertRaises(TypeError):
            gbsm.ssf_scatterers(self.rays)
        with self.assertRaises(TypeError):
            gbsm.ssf_scatterers()
        with self.assertRaises(TypeError):
            gbsm.ssf_scatterers(r, foo=1)


if __name__ == "__main__":
    unittest.main()
