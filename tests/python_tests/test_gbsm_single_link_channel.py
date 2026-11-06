# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.single_link_channel.
# Scope: argument and output handling, defaults, flags, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_single_link_channel.cpp).

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
FC = np.array([3.5e9, 28e9])
PAR = ["tap", "n_ray_tap", "fbs_pos", "lbs_pos", "gr_pos", "link_gain_dB", "b_med", "b_eff", "limit", "ratio",
       "ratio_total"]


def make_rays():
    # "quadriga" rays of one link in UMi LOS with ground reflection at two frequencies, solved
    sp = gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_UMi_LOS.conf"))[0]
    sp["GR_enabled"] = 1.0
    phase = gbsm.init_phases(sp, "lsf", 9)
    lsp, _, gr_eps, _, mu, sig = gbsm.lsp_generate(sp, FC, TX, RX, gbsm.draw(sp, "lsf", phase, TX, RX))
    rnd = gbsm.draw(sp, "cluster", gbsm.init_phases(sp, "cluster", 9), TX, RX)
    cl = gbsm.ssf_clusters(sp, TX, RX, lsp, rnd, "quadriga", gr_eps=gr_eps)
    rr = gbsm.draw(sp, "ray", gbsm.init_phases(sp, "ray", 9), TX, RX, center_freq=FC)
    rays = gbsm.ssf_rays(sp, TX, RX, FC, lsp, mu, sig, *cl[:6], rr, "quadriga", gr_eps=gr_eps)
    return gbsm.ssf_scatterers(rays[0])[0]


class test_case(unittest.TestCase):

    def setUp(self):
        self.rays = make_rays()
        self.tx = quadriga_lib.arrayant.generate("omni")
        self.rx = quadriga_lib.arrayant.generate("xpol")
        self.taps = np.unique(self.rays["tap"])
        self.n_path = int(np.sum(np.any(self.rays["power"] > 0, 1) | (self.rays["path_type"] < 2)))
        self.out = gbsm.single_link_channel(self.tx, self.rx, self.rays, np.array([-90.0]))

    def call(self, *args, **kw):
        return gbsm.single_link_channel(self.tx, self.rx, self.rays, np.array([-90.0]), *args, **kw)

    def test_basic_call(self):
        self.assertIsInstance(self.out, tuple)
        self.assertEqual(len(self.out), 2)
        chan, par = self.out
        n_tap = self.taps.size
        self.assertIsInstance(chan, dict)  # one dict, not a list
        self.assertIsInstance(chan["coeff"], list)  # stack=False: one array per snapshot
        self.assertEqual(chan["coeff"][0].shape, (2, 1, n_tap))
        self.assertEqual(chan["coeff"][0].dtype, np.complex128)
        self.assertEqual(chan["delay"][0].shape, (2, 1, n_tap))
        npt.assert_array_equal(chan["center_frequency"], FC[0])
        npt.assert_array_equal(chan["rx_position"], RX.T)
        self.assertIsInstance(par, dict)
        self.assertEqual(list(par.keys()), PAR)
        self.assertEqual(par["tap"].dtype, np.uint32)
        npt.assert_array_equal(par["tap"], self.taps)  # codes
        npt.assert_array_equal(par["tap"][:2], [0, 3])
        npt.assert_array_equal(par["ratio"], [0.0, 0.0])
        self.assertEqual(par["fbs_pos"].shape, (3, self.n_path))
        kept = np.any(self.rays["power"] > 0, 1) | (self.rays["path_type"] < 2)
        for k in ["fbs_pos", "lbs_pos"]:  # float32 copies
            self.assertEqual(par[k].dtype, np.float32)
            npt.assert_array_equal(par[k], self.rays[k][:, kept].astype(np.float32))
        self.assertEqual(par["gr_pos"].dtype, np.float32)
        self.assertEqual(par["gr_pos"].shape, (3, 1))

    def test_keywords(self):
        chan, par = gbsm.single_link_channel(tx_array=self.tx, rx_array=self.rx, rays=self.rays,
                                             link_gain_dB=np.array([-90.0]), mobility="auto", output="taps", i_freq=0,
                                             tx_pos=None, rx_pos=None, tx_orientation=None, rx_orientation=None,
                                             kf_ratio=None, oxygen=None, use_absolute_delays=None, stack=None)
        npt.assert_array_equal(chan["coeff"][0], self.out[0]["coeff"][0])
        npt.assert_array_equal(par["b_eff"], self.out[1]["b_eff"])

    def test_optional_inputs(self):
        c0 = self.out[0]["coeff"][0]
        chan, par = self.call("phase")
        self.assertNotIn("fbs_pos", par)
        chan, par = self.call("drift", "rays")
        self.assertEqual(chan["coeff"][0].shape, (2, 1, self.n_path))
        self.assertEqual(chan["path_gain"][0].shape, (self.n_path,))
        self.assertEqual(chan["fbs_pos"][0].shape, (3, self.n_path))
        self.assertEqual(par["tap"].size, self.n_path)
        self.assertNotIn("n_ray_tap", par)
        chan = self.call(i_freq=1)[0]
        npt.assert_array_equal(chan["center_frequency"], FC[1])
        self.assertGreater(np.max(np.abs(chan["coeff"][0] - c0)), 1e-9)
        rx2 = np.vstack([RX, RX + [1.0, 0.0, 0.0]])
        chan = self.call(tx_pos=TX, rx_pos=rx2)[0]
        self.assertEqual(len(chan["coeff"]), 2)
        npt.assert_allclose(chan["coeff"][0], c0, atol=1e-12, rtol=0)
        par = gbsm.single_link_channel(self.tx, self.rx, self.rays, [-90.0, -80.0], rx_pos=rx2)[1]
        npt.assert_array_equal(par["link_gain_dB"], [-90.0, -80.0])
        self.assertTrue(par["ratio"][1] > 0 and par["ratio"][0] == 0)
        chan = self.call(tx_orientation=np.array([[0.0, 0.0, 1.0]]), rx_orientation=[[0.0, 0.5, 0.0]])[0]
        npt.assert_array_equal(chan["tx_orientation"], [[0.0], [0.0], [1.0]])
        self.assertGreater(np.max(np.abs(chan["coeff"][0] - c0)), 1e-9)
        chan = self.call(kf_ratio=[0.0])[0]
        self.assertGreater(np.max(np.abs(chan["coeff"][0] - c0)), 1e-9)
        chan = self.call(use_absolute_delays=True)[0]
        npt.assert_allclose(chan["delay"][0] - self.out[0]["delay"][0], np.linalg.norm(RX - TX) / 299792458,
                            atol=1e-15, rtol=0)
        r60 = dict(self.rays, center_frequency=np.array([3.5e9, 60e9]))  # oxygen absorption needs > 52 GHz
        g0 = gbsm.single_link_channel(self.tx, self.rx, r60, [-90.0], output="rays", i_freq=1)[0]["path_gain"][0]
        g1 = gbsm.single_link_channel(self.tx, self.rx, r60, [-90.0], output="rays", i_freq=1, oxygen=True)[0]
        self.assertTrue(np.all(g1["path_gain"][0] < g0))

    def test_stack(self):
        rx2 = np.vstack([RX, RX + [1.0, 0.0, 0.0]])
        c0 = self.call(rx_pos=rx2)[0]
        c1 = self.call(rx_pos=rx2, stack=True)[0]
        self.assertIsInstance(c1["coeff"], np.ndarray)
        self.assertEqual(c1["coeff"].shape, (2, 1, self.taps.size, 2))
        npt.assert_array_equal(c1["coeff"][..., 1], c0["coeff"][1])
        npt.assert_array_equal(c1["delay"][..., 0], c0["delay"][0])

    def test_array_handling(self):
        d = dict(self.rays, M=np.ascontiguousarray(self.rays["M"]), tap=self.rays["tap"].astype(np.int32))
        chan = gbsm.single_link_channel(self.tx, self.rx, d, np.array([-90], dtype=np.float32),
                                        tx_pos=np.asfortranarray(TX), rx_pos=RX.astype(np.float32))[0]
        npt.assert_array_equal(chan["coeff"][0], self.out[0]["coeff"][0])

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            self.call(i_freq=2)
        self.assertIn("Input 'i_freq' is out of range for 2 frequencies.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.call(rx_pos=RX[0])
        self.assertIn("Input 'rx_pos' must have 3 columns.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.call(output="foo")
        self.assertIn("Input 'output' must be 'taps' or 'rays'.", str(ctx.exception))
        bad = dict(self.tx, e_theta_re=self.tx["e_theta_re"][:, :2])
        with self.assertRaises(ValueError) as ctx:
            gbsm.single_link_channel(bad, self.rx, self.rays, [0.0])
        self.assertIn("Transmit antenna: ", str(ctx.exception))
        with self.assertRaises(TypeError):
            self.call(stack=2.5)
        with self.assertRaises(TypeError):
            gbsm.single_link_channel(self.tx, self.rx, self.rays)
        with self.assertRaises(TypeError):
            self.call(foo=1)


if __name__ == "__main__":
    unittest.main()
