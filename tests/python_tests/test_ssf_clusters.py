# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.ssf_clusters.
# Scope: argument and output handling, optional inputs, slot layout, error paths.
# Physics validation lives in the Catch2 suite (test_ssf_clusters.cpp, test_ssf_clusters_mf.cpp,
# test_ssf_clusters_quadriga.cpp).

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
RX = np.array([[60.0, 10.0, 1.5], [-30.0, 80.0, 1.5], [120.0, -40.0, 7.5], [15.0, 25.0, 1.5]])


def read(name):
    return gbsm.conf_file_read(os.path.join(CONF_DIR, "3GPP_38.901_v19_" + name + ".conf"))[0]


def draw(sp, fc, seed):
    # LSPs, ground permittivity, EoD offset, absolute TOA and the random values of one scenario at the links
    phase = gbsm.lsf_init_phases(sp, seed, remove_bias=False)
    lsp, abs_toa, gr_eps, eod, _, _ = gbsm.lsp_generate(sp, np.array(fc), TX, RX, gbsm.lsf_draw(sp, phase, TX, RX))
    rnd = gbsm.ssf_draw(sp, gbsm.ssf_init_phases(sp, seed), TX, RX)
    rnd_f = np.stack([gbsm.ssf_draw(sp, gbsm.ssf_init_phases(sp, seed, k), TX, RX, k) for k in range(len(fc))], 2)
    return lsp, rnd, rnd_f, abs_toa, gr_eps, eod


class test_case(unittest.TestCase):

    def setUp(self):
        self.sp, self.sn = read("UMi_LOS"), read("UMi_NLOS")
        self.N = int(self.sp["NumClusters"]) - 1  # 12
        self.lsp, self.rnd, self.rnd_f, _, self.gr_eps, _ = draw(self.sp, [6e9], 11)
        self.lsp_n, self.rnd_n, self.rnd_fn, self.abs_toa_n, _, self.eod_n = draw(self.sn, [6e9], 12)
        self.out = gbsm.ssf_clusters(self.sp, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f)
        v = RX - TX
        self.aod_l = np.arctan2(v[:, 1], v[:, 0])
        self.eod_l = np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))

    def test_basic_call(self):
        out, N = self.out, self.N
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 7)
        delay, power, aod, eod, aoa, eoa, n_active = out
        for x in out[:6]:
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, (4, N + 1, 1))
            self.assertEqual(x.dtype, np.float64)
        self.assertEqual(n_active.shape, (4,))
        self.assertEqual(n_active.dtype, np.uint64)
        npt.assert_array_equal(delay[:, 0, 0], np.zeros(4))
        npt.assert_allclose(aod[:, 0, 0], self.aod_l, atol=1e-14, rtol=0)
        npt.assert_allclose(eod[:, 0, 0], self.eod_l, atol=1e-14, rtol=0)
        npt.assert_allclose(aoa[:, 0, 0], np.angle(-np.exp(1j * self.aod_l)), atol=1e-14, rtol=0)
        npt.assert_allclose(eoa[:, 0, 0], -self.eod_l, atol=1e-14, rtol=0)
        K = self.lsp[:, 0, 1]
        npt.assert_allclose(power[:, 0, 0], K / (K + 1), atol=1e-14, rtol=0)  # LOS slot
        self.assertTrue(np.all(power.sum(axis=1) <= 1 + 1e-12))
        npt.assert_array_equal(n_active, np.sum(power[:, 1:, 0] > 0, axis=1))

        # NLOS scenario: zero LOS power
        pn = gbsm.ssf_clusters(self.sn, TX, RX, self.lsp_n, self.rnd_n, "38.901", self.rnd_fn)[1]
        self.assertEqual(pn.shape, (4, 20, 1))
        npt.assert_array_equal(pn[:, 0, 0], np.zeros(4))

    def test_keyword_arguments(self):
        ok = gbsm.ssf_clusters(scenpar=self.sn, tx_pos=TX, rx_pos=RX, lsp=self.lsp_n, rnd=self.rnd_n,
                               method="38.901", rnd_freq=self.rnd_fn, lsp_anchor=None, eod_offset_deg=self.eod_n,
                               abs_toa=self.abs_toa_n, gr_eps=None, o2i=np.array([0, 1, 0, 0]),
                               n_cluster=np.array([19, 6, 9, 19]))
        op = gbsm.ssf_clusters(self.sn, TX, RX, self.lsp_n, self.rnd_n, "38.901", self.rnd_fn, None, self.eod_n,
                               self.abs_toa_n, None, np.array([0, 1, 0, 0]), np.array([19, 6, 9, 19]))
        for a, b in zip(ok, op):
            npt.assert_array_equal(a, b)
        self.assertTrue(np.all(ok[6] <= [19, 6, 9, 19]))

    def test_method(self):
        f = gbsm.ssf_clusters
        dq, pq, _, _, _, _, nq = f(self.sp, TX, RX, self.lsp, self.rnd, "quadriga")
        self.assertEqual(dq.shape, (4, self.N + 1, 1))
        npt.assert_allclose(pq.sum(axis=1)[:, 0], np.ones(4), atol=1e-12, rtol=0)
        npt.assert_array_equal(nq, np.full(4, self.N))
        npt.assert_array_equal(f(self.sp, TX, RX, self.lsp, self.rnd, rnd_freq=self.rnd_f)[0], self.out[0])

        # 38.901_MF with two frequencies and anchor LSPs: delays and angles per frequency
        lsp2, _, rnd_f2, _, _, _ = draw(self.sp, [6e9, 30e9], 11)
        lsp_a = draw(self.sp, [2e9], 11)[0]
        out = f(self.sp, TX, RX, lsp2, self.rnd, "38.901_MF", rnd_f2, lsp_a)
        for x in out[:6]:
            self.assertEqual(x.shape, (4, self.N + 1, 2))
        npt.assert_array_equal(rnd_f2[:, :, :1], self.rnd_f)

    def test_eod_offset_and_abs_toa(self):
        f = gbsm.ssf_clusters
        args = (self.sn, TX, RX, self.lsp_n, self.rnd_n, "38.901", self.rnd_fn)
        d0, p0, _, e0 = f(*args)[:4]
        e1 = f(*args, eod_offset_deg=self.eod_n)[3]
        self.assertTrue(np.max(np.abs(e1[:, 1:] - e0[:, 1:])) > 1e-3)
        npt.assert_array_equal(e1[:, 0], e0[:, 0])
        npt.assert_array_equal(f(*args, eod_offset_deg=None)[3], e0)
        dt, pt = f(*args, abs_toa=self.abs_toa_n)[:2]
        self.assertTrue(np.all(self.abs_toa_n > 0))
        npt.assert_allclose(dt[:, 1:, 0] - d0[:, 1:, 0], np.tile(self.abs_toa_n[:, None], (1, 19)), atol=1e-15,
                            rtol=0)
        npt.assert_array_equal(pt, p0)
        npt.assert_array_equal(f(*args, abs_toa=list(self.abs_toa_n))[0], dt)
        # "quadriga" in an NLOS scenario with absolute TOA: abs_toa required, the first NLOS cluster at abs_toa
        with self.assertRaises(ValueError) as ctx:
            f(self.sn, TX, RX, self.lsp_n, self.rnd_n, "quadriga")
        self.assertIn("Method 'quadriga' requires 'abs_toa'", str(ctx.exception))
        dq = f(self.sn, TX, RX, self.lsp_n, self.rnd_n, "quadriga", abs_toa=self.abs_toa_n)[0]
        npt.assert_array_equal(dq[:, 1:, 0].min(axis=1), self.abs_toa_n)

    def test_ground_reflection(self):
        f = gbsm.ssf_clusters
        sg = dict(self.sp)
        sg["GR_enabled"] = 1.0
        self.assertEqual(self.gr_eps.dtype, np.complex128)
        dg, pg, ag, eg, _, eag, _ = f(sg, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f, gr_eps=self.gr_eps)
        self.assertEqual(dg.shape, (4, self.N + 2, 1))
        npt.assert_array_equal(pg[:, [0] + list(range(2, self.N + 2))], self.out[1])  # other slots unchanged
        self.assertTrue(np.all(pg[:, 1, 0] > 0))
        npt.assert_allclose(ag[:, 1, 0], self.aod_l, atol=1e-14, rtol=0)
        e_gr = -np.arctan2(TX[0, 2] + RX[:, 2], np.hypot(RX[:, 0] - TX[0, 0], RX[:, 1] - TX[0, 1]))
        npt.assert_allclose(eg[:, 1, 0], e_gr, atol=1e-14, rtol=0)
        npt.assert_allclose(eag[:, 1, 0], e_gr, atol=1e-14, rtol=0)

        # A real array is accepted (imaginary part 0); the outputs differ
        pr = f(sg, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f, gr_eps=self.gr_eps.real.copy())[1]
        self.assertTrue(np.max(np.abs(pr[:, 1] - pg[:, 1])) > 1e-6)
        npt.assert_array_equal(pr[:, 2:], pg[:, 2:])
        p32 = f(sg, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f, gr_eps=self.gr_eps.astype(np.complex64))[1]
        npt.assert_allclose(p32, pg, atol=1e-6, rtol=0)  # permittivity rounded to single precision
        with self.assertRaises(ValueError) as ctx:
            f(sg, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f, gr_eps=np.conj(self.gr_eps))
        self.assertIn("'gr_eps_im' >= 0", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # a 3-D array is not flattened
            f(sg, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f, gr_eps=self.gr_eps[:, :, np.newaxis])
        self.assertIn("Input 'gr_eps' must have size (n_link, n_freq).", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(sg, TX, RX, self.lsp, self.rnd, "quadriga")
        self.assertIn("are required when 'GR_enabled' is set", str(ctx.exception))

    def test_o2i_and_n_cluster(self):
        f = gbsm.ssf_clusters
        args = (self.sn, TX, RX, self.lsp_n, self.rnd_n, "38.901", self.rnd_fn)
        ea = f(*args)[5]
        eo = f(*args, o2i=np.array([False, True, False, False]))[5]
        npt.assert_array_equal(eo[[0, 2, 3]], ea[[0, 2, 3]])
        self.assertTrue(np.max(np.abs(eo[1, 1:] - ea[1, 1:])) > 1e-3)
        npt.assert_array_equal(f(*args, o2i=[0, 2.5, 0, 0])[5], eo)  # numeric, nonzero = True

        nc = np.array([12, 6, 9, 12])
        dc, pc, ac, _, _, _, nac = f(self.sp, TX, RX, self.lsp, self.rnd, "quadriga", n_cluster=nc)
        npt.assert_array_equal(pc[1, 7:, 0], np.zeros(6))  # slots 7 ... 12 inactive
        npt.assert_array_equal(dc[1, 7:, 0], np.zeros(6))
        npt.assert_allclose(ac[1, 7:, 0], np.full(6, self.aod_l[1]), atol=1e-14, rtol=0)
        npt.assert_array_equal(nac, nc)
        npt.assert_array_equal(f(self.sp, TX, RX, self.lsp, self.rnd, "quadriga", n_cluster=nc.astype(float))[0], dc)
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, TX, RX, self.lsp, self.rnd, "quadriga", n_cluster=[12, 6.5, 9, 12])
        self.assertIn("Entries in 'n_cluster' must be non-negative integers below 2^53.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, TX, RX, self.lsp, self.rnd, "quadriga", n_cluster=[12, 3, 9, 12])
        self.assertIn("Entries of 'n_cluster' must be in [6, 12].", str(ctx.exception))

    def test_special_sizes(self):
        f = gbsm.ssf_clusters
        # Zero links
        out = f(self.sp, TX, np.zeros((0, 3)), np.zeros((0, 1, 8)), np.zeros((0, 9 * self.N)), "38.901",
                np.zeros((0, self.N, 1)))
        self.assertEqual(out[0].shape, (0, self.N + 1, 1))
        self.assertEqual(out[6].shape, (0,))
        # NumClusters = 1: LOS path only
        s1 = dict(self.sp)
        s1["NumClusters"], s1["NumClusters_min"] = 1.0, 1.0
        d1, p1, _, _, _, _, n1 = f(s1, TX, RX, self.lsp, np.zeros((4, 0)))
        self.assertEqual(d1.shape, (4, 1, 1))
        npt.assert_array_equal(p1[:, 0, 0], np.ones(4))
        npt.assert_array_equal(n1, np.zeros(4))

    def test_memory_layout_and_dtype(self):
        f = gbsm.ssf_clusters
        args = (self.sp, TX, RX, self.lsp, self.rnd, "38.901", self.rnd_f)
        of = f(self.sp, np.asfortranarray(TX), np.asfortranarray(RX), np.asfortranarray(self.lsp),
               np.asfortranarray(self.rnd), "38.901", np.asfortranarray(self.rnd_f))
        big = np.zeros((8, 2 * 9 * self.N))
        big[::2, ::2] = self.rnd
        os_ = f(self.sp, TX, RX, self.lsp, big[::2, ::2], "38.901", self.rnd_f)  # strided view
        o32 = f(self.sp, TX.astype(np.float32), RX.astype(np.float32), *args[3:])  # positions exact in float32
        for a, b, c, d in zip(self.out, of, os_, o32):
            npt.assert_array_equal(b, a)
            npt.assert_array_equal(c, a)
            npt.assert_array_equal(d, a)
        lsp0, rnd0 = self.lsp.copy(), self.rnd.copy()
        f(*args)
        npt.assert_array_equal(self.lsp, lsp0)
        npt.assert_array_equal(self.rnd, rnd0)

    def test_errors(self):
        f = gbsm.ssf_clusters
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, TX, RX, self.lsp, self.rnd)
        self.assertIn("Input 'rnd_freq' is required for method '38.901'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, TX, RX, self.lsp, self.rnd, "other")
        self.assertIn("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            f(self.sp, TX, RX, self.lsp, self.rnd[:, 1:], "quadriga")
        self.assertIn("Input 'rnd' must have size [n_link, 108].", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.sp, TX, RX, self.lsp)
        with self.assertRaises(TypeError):
            f(self.sp, TX, RX, self.lsp, self.rnd, unknown=1)
        with self.assertRaises(TypeError):
            f(self.sp, TX, RX, None, self.rnd)


if __name__ == "__main__":
    unittest.main()
