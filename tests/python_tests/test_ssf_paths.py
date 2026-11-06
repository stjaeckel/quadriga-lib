# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_paths.
# Scope: argument and output handling, defaults, flags, error paths.
# Physics validation lives in the Catch2 suite (test_ssf_paths.cpp).

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
RX = np.array([[60.0, 10.0, 1.5]])
FC = np.array([60e9])


def make_rays():
    # "38.901" rays of one link in UMi LOS with ground reflection at 60 GHz, solved
    sp = gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
    sp["GR_enabled"] = 1.0
    phase = gbsm.init_phases(sp, "lsf", 5)
    lsp, _, gr_eps, _, mu, sig = gbsm.lsp_generate(sp, FC, TX, RX, gbsm.draw(sp, "lsf", phase, TX, RX))
    rnd = gbsm.draw(sp, "cluster", gbsm.init_phases(sp, "cluster", 5), TX, RX)
    rf = gbsm.draw(sp, "cluster", gbsm.init_phases(sp, "cluster", 5, 0), TX, RX, 0)
    rr = gbsm.draw(sp, "ray", gbsm.init_phases(sp, "ray", 5, 0), TX, RX, 0)
    cl = gbsm.ssf_clusters(sp, TX, RX, lsp, rnd, "38.901", rf, gr_eps=gr_eps)
    rays = gbsm.ssf_rays(sp, TX, RX, FC, lsp, mu, sig, *cl[:6], rr[:, :, None], gr_eps=gr_eps)
    return gbsm.ssf_scatterers(rays[0])[0]


class test_case(unittest.TestCase):

    def setUp(self):
        self.rays = make_rays()
        r = self.rays
        self.kept = np.nonzero(np.any(r["power"] > 0, 1) | (r["path_type"] < 2))[0]
        self.out = gbsm.ssf_paths(r, np.array([-90.0]))

    def test_basic_call(self):
        out, r, kept = self.out, self.rays, self.kept
        n_path = kept.size
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 11)
        pg, pl, M, aod, eod, aoa, eoa, fbs, lbs, pt, ri = out
        self.assertEqual(pg.shape, (n_path, 1))
        self.assertEqual(pl.shape, (n_path,))
        self.assertEqual(M.shape, (8, n_path, 1))
        self.assertEqual(aod.shape, (n_path,))
        self.assertEqual(fbs.shape, (3, 0))  # "phase"
        self.assertEqual(lbs.shape, (3, 0))
        self.assertEqual(pg.dtype, np.float64)
        self.assertEqual(pt.dtype, np.int64)
        self.assertEqual(ri.dtype, np.int64)
        npt.assert_array_equal(ri, kept)  # 0-based
        npt.assert_array_equal(pt, r["path_type"][ri])
        npt.assert_array_equal(pt[:3], [0, 1, 2])
        npt.assert_array_equal(r["tap"][ri[:2]], [0, 3])
        npt.assert_allclose(pg[:, 0], r["power"][kept, 0] * 1e-9, atol=1e-20, rtol=0)
        npt.assert_allclose(pl, np.linalg.norm(RX - TX) + 299792458 * r["delay"][kept], atol=1e-9, rtol=0)
        npt.assert_array_equal(M, r["M"][:, kept])
        npt.assert_array_equal(eoa, r["eoa"][kept])

    def test_keywords(self):
        out = gbsm.ssf_paths(rays=self.rays, link_gain_dB=np.array([-90.0]), mobility="auto", tx_pos=None,
                             rx_pos=None, kf_ratio=None, oxygen=False)
        for a, b in zip(out, self.out):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        r, pg, pl, M = self.rays, *self.out[:3]
        out = gbsm.ssf_paths(r, [-90.0], "drift")
        npt.assert_array_equal(out[7], r["fbs_pos"][:, out[10]])
        npt.assert_array_equal(out[8], r["lbs_pos"][:, out[10]])
        self.assertGreater(np.max(np.abs(out[1] - pl)), 1e-6)  # fallback rays: longer polyline
        rx = RX[0] + [0.01, 0.0, 0.0]
        out = gbsm.ssf_paths(r, [-90.0], "phase", TX[0], rx)
        npt.assert_array_equal(out[0], pg)
        self.assertGreater(np.max(np.abs(out[2] - M)), 1e-3)  # displacement phase
        out2 = gbsm.ssf_paths(r, [-90.0], "phase", rx_pos=list(rx))
        npt.assert_array_equal(out2[2], out[2])
        g = gbsm.ssf_paths(r, [-90.0], kf_ratio=np.zeros(1))[0]
        npt.assert_array_equal(g[:2, 0], 0.0)  # kf_ratio 0: no LOS and GR
        g = gbsm.ssf_paths(r, [-90.0], oxygen=True)[0]
        self.assertTrue(np.all(g < pg))  # oxygen at 60 GHz
        npt.assert_array_equal(gbsm.ssf_paths(r, [-90.0], oxygen=np.bool_(True))[0], g)
        npt.assert_array_equal(gbsm.ssf_paths(r, [-90.0], oxygen=None)[0], pg)
        npt.assert_allclose(gbsm.ssf_paths(r, np.array([-80], dtype=np.int32))[0], pg * 10, atol=1e-18, rtol=0)

    def test_dict_handling(self):
        d = {k: v for k, v in self.rays.items() if k not in ["gr_eps", "fallback"]}
        d["path_type"] = d["path_type"].astype(np.float32)
        d["M"] = np.ascontiguousarray(d["M"])
        out = gbsm.ssf_paths(d, [-90.0])
        npt.assert_array_equal(out[0], self.out[0])
        npt.assert_array_equal(out[9], self.out[9])
        m0 = self.rays["M"].copy()
        gbsm.ssf_paths(self.rays, [-90.0], "phase", rx_pos=RX[0] + 1.0)
        npt.assert_array_equal(self.rays["M"], m0)  # input not modified

    def test_errors(self):
        r = self.rays
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_paths(r, [0.0], "foo")
        self.assertIn("Input 'mobility' must be 'phase', 'drift' or 'auto'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_paths(r, [0.0, 0.0])
        self.assertIn("Input 'link_gain_dB' must have n_freq elements.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_paths(dict(r, fbs_pos=None, lbs_pos=None), [0.0], "drift")
        self.assertIn("Mobility 'drift' needs 'rays.fbs_pos'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.ssf_paths({k: v for k, v in r.items() if k != "tap"}, [0.0])
        self.assertIn("Entry 'tap' of 'rays' is missing.", str(ctx.exception))
        with self.assertRaises(TypeError):
            gbsm.ssf_paths(r, [0.0], oxygen=2.5)
        with self.assertRaises(TypeError):
            gbsm.ssf_paths(r)
        with self.assertRaises(TypeError):
            gbsm.ssf_paths(r, None)
        with self.assertRaises(TypeError):
            gbsm.ssf_paths(r, [0.0], foo=1)


if __name__ == "__main__":
    unittest.main()
