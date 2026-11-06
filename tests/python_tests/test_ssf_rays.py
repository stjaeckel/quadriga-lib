# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_rays.
# Scope: argument and output handling, ray dict keys, defaults, flags, error paths.
# Physics validation lives in the Catch2 suite (test_ssf_rays.cpp, test_ssf_rays_quadriga.cpp).

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
RX = np.array([[60.0, 10.0, 1.5], [-30.0, 80.0, 1.5], [120.0, -40.0, 7.5]])
FC = np.array([6e9, 30e9])
KEYS = ["method", "tx_pos", "rx_pos", "center_frequency", "delay", "power", "aod", "eod", "aoa", "eoa", "M",
        "path_type", "tap", "gr_eps", "fbs_pos", "lbs_pos", "fallback"]


def lsps(sp, fc):
    # LSPs, ground permittivity, means and std at the links
    phase = gbsm.init_phases(sp, "lsf", 7)
    lsp, _, gr_eps, _, mu, sig = gbsm.lsp_generate(sp, fc, TX, RX, gbsm.draw(sp, "lsf", phase, TX, RX))
    return lsp, gr_eps, mu, sig


def cset(sp, set_type, k):
    # Per-frequency random set k (0-based set id)
    return gbsm.draw(sp, set_type, gbsm.init_phases(sp, set_type, 7, k), TX, RX, k)


class test_case(unittest.TestCase):

    def setUp(self):
        sp = gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
        sp["GR_enabled"] = 1.0
        self.sp = sp
        self.lsp, self.gr_eps, self.mu, self.sig = lsps(sp, FC)
        rnd = gbsm.draw(sp, "cluster", gbsm.init_phases(sp, "cluster", 7), TX, RX)
        self.n_ray = 2 + (int(sp["NumClusters"]) - 1) * int(sp["NumSubPaths"])
        # "quadriga": clusters of all frequencies, the shared "ray" set
        self.cl = gbsm.ssf_clusters(sp, TX, RX, self.lsp, rnd, "quadriga", gr_eps=self.gr_eps)
        self.rq = gbsm.draw(sp, "ray", gbsm.init_phases(sp, "ray", 7), TX, RX, center_freq=FC)
        self.args = (sp, TX, RX, FC, self.lsp, self.mu, self.sig, *self.cl[:6], self.rq)
        # "38.901_MF": per-frequency cluster and "ray" sets, anchor LSPs at 2 GHz
        rf = np.stack([cset(sp, "cluster", k) for k in range(2)], 2)
        self.rr = np.stack([cset(sp, "ray", k) for k in range(2)], 2)
        la = lsps(sp, np.array([2e9]))[0]
        self.cm = gbsm.ssf_clusters(sp, TX, RX, self.lsp, rnd, "38.901_MF", rf, la, gr_eps=self.gr_eps)
        self.rays = gbsm.ssf_rays(*self.args, "quadriga", gr_eps=self.gr_eps)

    def mf(self, **kw):
        return gbsm.ssf_rays(self.sp, TX, RX, FC, self.lsp, self.mu, self.sig, *self.cm[:6], self.rr, "38.901_MF",
                             gr_eps=self.gr_eps, **kw)

    def test_basic_call(self):
        rays, n_ray = self.rays, self.n_ray
        self.assertIsInstance(rays, list)
        self.assertEqual(len(rays), 3)
        for l, r in enumerate(rays):
            self.assertIsInstance(r, dict)
            self.assertEqual(list(r.keys()), KEYS)
            self.assertEqual(r["method"], "quadriga")
            npt.assert_array_equal(r["tx_pos"], TX[0])
            npt.assert_array_equal(r["rx_pos"], RX[l])
            npt.assert_array_equal(r["center_frequency"], FC)
            self.assertEqual(r["delay"].shape, (n_ray,))
            self.assertEqual(r["power"].shape, (n_ray, 2))
            self.assertEqual(r["eoa"].shape, (n_ray,))
            self.assertEqual(r["M"].shape, (8, n_ray, 2))
            self.assertEqual(r["M"].dtype, np.float64)
            for k in ["path_type", "tap", "fallback"]:
                self.assertEqual(r[k].dtype, np.uint32)
            npt.assert_array_equal(r["path_type"][:3], [0, 1, 2])  # codes
            npt.assert_array_equal(r["tap"][:3], [0, 3, 6])
            self.assertEqual(r["gr_eps"].dtype, np.complex128)
            npt.assert_array_equal(r["gr_eps"], self.gr_eps[l])
            for k in ["fbs_pos", "lbs_pos", "fallback"]:
                self.assertEqual(r[k].size, 0)
            npt.assert_allclose(r["power"][0], self.cl[1][l, 0], atol=1e-14, rtol=0)

    def test_no_ground_reflection(self):
        sn = dict(self.sp, GR_enabled=0.0)
        rnd = gbsm.draw(sn, "cluster", gbsm.init_phases(sn, "cluster", 7), TX, RX)
        cl = gbsm.ssf_clusters(sn, TX, RX, self.lsp, rnd, "quadriga")
        r = gbsm.ssf_rays(sn, TX, RX, FC, self.lsp, self.mu, self.sig, *cl[:6], self.rq, "quadriga")[0]
        self.assertEqual(list(r.keys()), KEYS)
        self.assertEqual(r["delay"].shape, (self.n_ray - 1,))
        self.assertEqual(r["gr_eps"].size, 0)
        self.assertEqual(r["gr_eps"].dtype, np.complex128)

    def test_keywords_and_i_freq(self):
        r0 = self.mf()
        npt.assert_array_equal(r0[0]["center_frequency"], [FC[0]])
        r1 = self.mf(i_freq=1)
        npt.assert_array_equal(r1[0]["center_frequency"], [FC[1]])  # 0-based
        npt.assert_array_equal(r1[1]["gr_eps"], [self.gr_eps[1, 1]])
        self.assertEqual(r1[0]["power"].shape, (self.n_ray, 1))
        self.assertGreater(np.max(np.abs(r1[0]["delay"] - r0[0]["delay"])), 0)
        cm = self.cm
        r2 = gbsm.ssf_rays(scenpar=self.sp, tx_pos=TX, rx_pos=RX, center_freq=FC, lsp=self.lsp, lsp_mu=self.mu,
                           lsp_sigma=self.sig, delay=cm[0], power=cm[1], aod=cm[2], eod=cm[3], aoa=cm[4], eoa=cm[5],
                           rnd_ray=self.rr, method="38.901_MF", i_freq=1, angle_model="additive",
                           pol_model="38.901", gr_eps=self.gr_eps, pol_variability=False, random_phase=True)
        npt.assert_array_equal(r2[2]["M"], r1[2]["M"])

    def test_optional_inputs(self):
        r0 = self.mf()
        r1 = self.mf(pol_variability=None, random_phase=None)
        npt.assert_array_equal(r1[2]["M"], r0[2]["M"])
        r = self.mf(angle_model="rotation")
        self.assertGreater(np.max(np.abs(r[2]["eod"] - r0[2]["eod"])), 1e-6)
        r = self.mf(pol_model="quadriga")
        self.assertGreater(np.max(np.abs(r[2]["M"] - r0[2]["M"])), 1e-6)
        r = self.mf(pol_variability=True)
        self.assertGreater(np.max(np.abs(r[2]["M"] - r0[2]["M"])), 1e-6)
        npt.assert_array_equal(r[2]["delay"], r0[2]["delay"])
        r = self.mf(random_phase=np.bool_(False))
        npt.assert_array_equal(r[2]["M"][1::2, 2:], 0.0)  # initial phases 0: real matrices
        # Default method "38.901" at one frequency
        cl = self.cm
        rd = gbsm.ssf_rays(self.sp, TX, RX, FC[:1], self.lsp[:, :1], self.mu[:, :1], self.sig[:, :1],
                           *[c[:, :, :1] for c in cl[:6]], self.rr[:, :, :1], gr_eps=self.gr_eps[:, :1])
        self.assertEqual(rd[0]["method"], "38.901")
        npt.assert_array_equal(rd[0]["center_frequency"], FC[:1])

    def test_array_handling(self):
        tx = np.asfortranarray(TX.astype(np.float32))
        args = list(self.args)
        args[1], args[3] = tx, list(FC)
        args[4] = np.ascontiguousarray(self.lsp)
        r = gbsm.ssf_rays(*args, "quadriga", gr_eps=self.gr_eps)
        npt.assert_array_equal(r[1]["delay"], self.rays[1]["delay"])
        npt.assert_array_equal(r[1]["M"], self.rays[1]["M"])
        r = gbsm.ssf_rays(*self.args, "quadriga", gr_eps=self.gr_eps.real)  # real array: imaginary part 0
        self.assertEqual(r[0]["gr_eps"].dtype, np.complex128)
        npt.assert_array_equal(r[0]["gr_eps"].imag, 0.0)
        lsp0 = self.lsp.copy()
        gbsm.ssf_rays(*self.args, "quadriga", gr_eps=self.gr_eps)
        npt.assert_array_equal(self.lsp, lsp0)

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.mf(i_freq=2)
        self.assertIn("Input 'i_freq' is out of range for 2 frequencies.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_rays(*self.args, "quadriga")
        self.assertIn("Inputs 'gr_eps_re' and 'gr_eps_im' are required", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_rays(*self.args, "foo")
        self.assertIn("Input 'method' must be", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_rays(*self.args, "quadriga", gr_eps=self.gr_eps[:, :, None, None])
        self.assertIn("Input 'gr_eps' must have size (n_link, n_freq).", str(ctx.exception))
        with self.assertRaises(TypeError):
            gbsm.ssf_rays(*self.args, "quadriga", gr_eps=self.gr_eps, random_phase=2.5)
        with self.assertRaises(TypeError):
            gbsm.ssf_rays(*self.args[:13])
        with self.assertRaises(TypeError):
            gbsm.ssf_rays(*self.args, foo=1)


if __name__ == "__main__":
    unittest.main()
