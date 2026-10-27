# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.lsp_generate.
# Scope: argument and output handling, complex flag, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_lsp_generate.cpp).

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

config_dir = os.path.join(current_dir, "../../data/config")


def read_conf(name):
    return quadriga_lib.gbsm.conf_file_read(os.path.join(config_dir, name))[0]


class test_case(unittest.TestCase):

    def setUp(self):
        # UMa files, 3 links, 2 frequencies
        self.sp = read_conf("3GPP_38.901_v19_UMa_LOS.conf")
        self.sp_nlos = read_conf("3GPP_38.901_v19_UMa_NLOS.conf")
        self.sp_inf = read_conf("3GPP_38.901_v19_InF_NLOS_SL.conf")
        self.tx_pos = np.array([[0.0, 0.0, 25.0]])
        self.rx_pos = np.array([[100.0, 0.0, 1.5], [0.0, 200.0, 1.5], [-50.0, 50.0, 22.5]])
        self.fc = np.array([2e9, 28e9])
        self.rnd = np.zeros((3, 11))
        self.rnd[:, 9] = 0.5  # medium ground
        self.out = quadriga_lib.gbsm.lsp_generate(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd)

    def call(self, *args, **kwargs):
        return quadriga_lib.gbsm.lsp_generate(*args, **kwargs)

    def test_basic_call(self):
        out = self.out
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 6)
        lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = out
        shapes = [(3, 2, 8), (3,), (3, 2), (3, 2), (3, 2, 8), (3, 2, 8)]
        for x, s in zip(out, shapes):
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, s)
        for x in (lsp, abs_toa, eod_offset_deg, lsp_mu, lsp_sigma):
            self.assertEqual(x.dtype, np.float64)
        self.assertEqual(gr_eps.dtype, np.complex128)

        # rnd = 0 gives the median LSPs: 10^mu for DS, 10^(0.1·mu) for KF
        npt.assert_allclose(lsp[:, :, 0], 10.0 ** lsp_mu[:, :, 0], rtol=1e-12, atol=0)
        npt.assert_allclose(lsp[:, :, 1], 10.0 ** (0.1 * lsp_mu[:, :, 1]), rtol=1e-12, atol=0)
        npt.assert_array_equal(lsp[:, :, 2], np.ones((3, 2)))  # SF: mean 0 dB
        self.assertTrue(np.all(lsp_sigma >= 0))
        self.assertAlmostEqual(lsp_sigma[0, 0, 2], 4.0, delta=1e-12)  # SF_sigma of the file

        # Column 9 of rnd (0.5) is the medium ground: 30.4·f^-0.47 + j·0.18·f^1.05, f in GHz
        f = np.tile(self.fc / 1e9, (3, 1))
        npt.assert_allclose(gr_eps.real, 30.4 * f ** -0.47, rtol=1e-12, atol=0)
        npt.assert_allclose(gr_eps.imag, 0.18 * f ** 1.05, rtol=1e-12, atol=0)

        # Column 8 of rnd is the absolute TOA (UMa NLOS: absTOA_mu = -7.4, no bound); LOS: disabled
        npt.assert_array_equal(abs_toa, np.zeros(3))
        rnd_t = self.rnd.copy()
        rnd_t[:, 8] = [0.0, 1.0, -1.0]
        toa = self.call(self.sp_nlos, self.fc, self.tx_pos, self.rx_pos, rnd_t)[1]
        npt.assert_allclose(toa, 10.0 ** (-7.4 + self.sp_nlos["absTOA_sigma"] * rnd_t[:, 8]), rtol=1e-12, atol=0)

    def test_keyword_arguments(self):
        ok = self.call(scenpar=self.sp, fc=self.fc, tx_pos=self.tx_pos, rx_pos=self.rx_pos, rnd=self.rnd,
                       sf_sigma=None, cholesky=True, clip_spreads=True, complex=True)
        op = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, None, True, True, True)
        for a, b, c in zip(ok, op, self.out):
            npt.assert_array_equal(a, c)
            npt.assert_array_equal(b, c)

    def test_complex_flag(self):
        os_ = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, complex=False)
        self.assertEqual(len(os_), 7)
        lsp, abs_toa, gr_eps_re, gr_eps_im, eod, mu, sigma = os_
        for x in (gr_eps_re, gr_eps_im):
            self.assertEqual(x.shape, (3, 2))
            self.assertEqual(x.dtype, np.float64)
        npt.assert_array_equal(self.out[2].real, gr_eps_re)
        npt.assert_array_equal(self.out[2].imag, gr_eps_im)
        for a, b in zip((lsp, abs_toa, eod, mu, sigma), (self.out[0], self.out[1]) + self.out[3:]):
            npt.assert_array_equal(a, b)

    def test_sf_sigma(self):
        lsp_sigma = self.out[5]
        o2 = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, None)
        npt.assert_array_equal(o2[0], self.out[0])
        npt.assert_array_equal(o2[5], lsp_sigma)
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, np.zeros((0, 2)))[5],
                               lsp_sigma)  # empty = scenario value
        sf_in = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rnd_s = self.rnd.copy()
        rnd_s[:, 2] = 1.0  # SF value passed through
        lsp3, _, _, _, _, sig3 = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_s, sf_in)
        npt.assert_array_equal(sig3[:, :, 2], sf_in)  # replaces slice 2
        npt.assert_array_equal(sig3[:, :, [0, 1, 3, 4, 5, 6, 7]], lsp_sigma[:, :, [0, 1, 3, 4, 5, 6, 7]])
        npt.assert_allclose(lsp3[:, :, 2], 10.0 ** (0.1 * sf_in), rtol=1e-12, atol=0)
        sig4 = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_s, sf_in.astype(np.int32))[5]
        npt.assert_array_equal(sig4, sig3)

    def test_cholesky_and_clip_spreads(self):
        # TR order: the SF value is not mixed with KF; symmetric root: it is
        rnd_k = self.rnd.copy()
        rnd_k[:, 1] = 1.0  # KF value only
        lsp_c = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_k)[0]
        npt.assert_array_equal(lsp_c[:, :, 2], np.ones((3, 2)))
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_k, cholesky=True)[0], lsp_c)
        lsp_q = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_k, cholesky=False)[0]
        self.assertTrue(np.all(np.abs(lsp_q[:, 0, 2] - 1) > 1e-3))

        rnd_a = self.rnd.copy()
        rnd_a[:, 3] = 6.0  # large ASD
        lsp_a = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_a)[0]
        self.assertEqual(np.max(lsp_a[:, :, 3]), 104.0)
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_a, clip_spreads=True)[0],
                               lsp_a)
        lsp_u = self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_a, clip_spreads=False)[0]
        self.assertGreater(np.max(lsp_u[:, :, 3]), 104.0)

    def test_flags(self):
        args = (self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd)
        out = self.call(*args, None, None, None, None)  # None selects the default of each flag
        self.assertEqual(len(out), 6)
        for a, b in zip(out, self.out):
            npt.assert_array_equal(a, b)
        self.assertEqual(len(self.call(*args, complex=np.bool_(False))), 7)
        for name in ("cholesky", "clip_spreads", "complex"):
            for bad in (2.5, 1, "yes"):
                with self.assertRaises(TypeError) as ctx:
                    self.call(*args, **{name: bad})
                self.assertIn(f"Input '{name}' must be a bool or None.", str(ctx.exception))

    def test_scenpar(self):
        # DS_VS changes the InF delay spread by lg(a·V/S + b) (DS_VS_a = 30, DS_VS_b = 32 in the InF files)
        mu_4 = self.call(self.sp_inf, self.fc, self.tx_pos, self.rx_pos, self.rnd)[4]
        sp = dict(self.sp_inf)
        sp["DS_VS"] = 10.0
        lsp_10, _, _, _, mu_10, _ = self.call(sp, self.fc, self.tx_pos, self.rx_pos, self.rnd)
        npt.assert_allclose(mu_10[:, :, 0] - mu_4[:, :, 0], np.full((3, 2), np.log10((30 * 10 + 32) / (30 * 4 + 32))),
                            atol=1e-12, rtol=0)
        npt.assert_allclose(lsp_10[:, :, 0], 10.0 ** mu_10[:, :, 0], rtol=1e-12, atol=0)
        self.assertEqual(self.call(None, self.fc, self.tx_pos, self.rx_pos, self.rnd)[0].shape, (3, 2, 8))

    def test_conf_file_chain(self):
        # conf_file_read -> path_loss (sf_sigma) -> lsp_generate
        sp, plpar = quadriga_lib.gbsm.conf_file_read(os.path.join(config_dir, "3GPP_38.901_v19_UMa_NLOS.conf"))[:2]
        pl, sf = quadriga_lib.gbsm.path_loss(plpar, sp, self.fc, self.tx_pos, self.rx_pos)
        lsp, _, _, _, _, sig = self.call(sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, sf)
        npt.assert_array_equal(sig[:, :, 2], sf)
        self.assertEqual(lsp.shape, (3, 2, 8))

    def test_array_layout_and_dtype(self):
        lsp = self.out[0]
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, np.asfortranarray(self.rx_pos), self.rnd)[0],
                               lsp)
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, np.asfortranarray(self.rnd))[0],
                               lsp)
        big = np.zeros((6, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(self.call(self.sp, self.fc, self.tx_pos, big[::2, ::2], self.rnd)[0], lsp)
        lsp_t = self.call(self.sp, self.fc.astype(np.float32), self.tx_pos.astype(np.float32),
                          self.rx_pos.astype(np.float32), self.rnd.astype(np.float32))[0]
        npt.assert_allclose(lsp_t, lsp, rtol=1e-12, atol=0)  # inputs exact in float32
        npt.assert_array_equal(self.call(self.sp, self.fc, np.tile(self.tx_pos, (3, 1)), self.rx_pos, self.rnd)[0],
                               lsp)

    def test_zero_links(self):
        out = self.call(self.sp, self.fc, self.tx_pos, np.zeros((0, 3)), np.zeros((0, 11)))
        self.assertEqual([x.shape for x in out], [(0, 2, 8), (0,), (0, 2), (0, 2), (0, 2, 8), (0, 2, 8)])

    def test_inputs_not_modified(self):
        rnd, rx = self.rnd.copy(), self.rx_pos.copy()
        sf = np.ones((3, 2))
        self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, sf)
        npt.assert_array_equal(self.rnd, rnd)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(sf, np.ones((3, 2)))

    def test_errors(self):
        rnd_b = self.rnd.copy()
        rnd_b[:, 9] = 1.5
        for cplx in (True, False):  # the permittivity is always computed, so column 9 is always checked
            with self.assertRaises(ValueError) as ctx:  # C++
                self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, rnd_b, complex=cplx)
            self.assertIn("Column 9 of 'rnd' must be in the range from 0 to 1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd[:, :10])
        self.assertIn("Input 'rnd' must have size [n_link, 11].", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, -np.ones((3, 2)))
        self.assertIn("Input 'sf_sigma' must be finite and >= 0.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            self.call({"ds_kf": 0.9, "ds_sf": 0.9, "sf_kf": -0.9}, self.fc, self.tx_pos, self.rx_pos, self.rnd)
        self.assertIn("LSP cross-correlation matrix in 'scenpar' is not positive definite.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            self.call(self.sp, np.zeros(0), self.tx_pos, self.rx_pos, self.rnd)
        self.assertIn("Input 'fc' cannot be empty.", str(ctx.exception))
        with self.assertRaises(TypeError):
            self.call(self.sp, self.fc, self.tx_pos, self.rx_pos)
        with self.assertRaises(TypeError):
            self.call(self.sp, self.fc, self.tx_pos, self.rx_pos, self.rnd, foo=1)


if __name__ == "__main__":
    unittest.main()
