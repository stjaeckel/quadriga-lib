# SPDX-License-Identifier: Apache-2.0
#
# quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import channel

# ==========================================================================================
# Helpers
# ==========================================================================================

def bake_delay_phase(coeff_4d, delay_3d, freq_in):
    """Bake exp(-j*2*pi*freq_in[f]*delay) into 4D coefficient array."""
    n_rx, n_tx, n_path, n_freq = coeff_4d.shape
    out = coeff_4d.copy()
    for f in range(n_freq):
        for p in range(n_path):
            if delay_3d.shape[0] == 1 and delay_3d.shape[1] == 1:
                dl = delay_3d[0, 0, p]
                out[:, :, p, f] *= np.exp(-1j * 2.0 * np.pi * freq_in[f] * dl)
            else:
                for r in range(n_rx):
                    for t in range(n_tx):
                        dl = delay_3d[r, t, p]
                        out[r, t, p, f] *= np.exp(-1j * 2.0 * np.pi * freq_in[f] * dl)
    return out


class test_case(unittest.TestCase):

    # ==================================================================================
    # Backward-compatibility test (original test file)
    # ==================================================================================

    def test_backward_compat(self):
        coeff = np.zeros((4, 3, 2), dtype=complex)
        coeff[:, 0, 0] = np.arange(0.25, 1.25, 0.25)
        coeff[:, 1, 0] = np.arange(1, 5)
        coeff[:, 2, 0] = 1j * np.arange(1, 5)
        coeff[:, :, 1] = -coeff[:, :, 0]

        coeff_list = [coeff, 2 * coeff, 3 * coeff]
        fc = 299792458.0
        delay = np.zeros((1, 1, 2), dtype=float)
        delay[0, 0, 0] = 1 / fc
        delay[0, 0, 1] = 1.5 / fc
        delay_list = [delay, delay, delay]
        pilots = np.arange(0, 2.1, 0.1)

        hmat = channel.baseband_freq_response(coeff_list, delay_list, fc, carriers=11)
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal(hmat_re[:, :, 0, 0], T, decimal=5)
        npt.assert_almost_equal(hmat_im[:, :, 0, 0], T, decimal=5)
        npt.assert_almost_equal(hmat_re[:, 0, :, :] * 4.0, hmat_re[:, 1, :, :], decimal=6)

        T = np.array([[0.5, 2, 0], [1, 4, 0], [1.5, 6, 0], [2, 8, 0]])
        npt.assert_almost_equal(hmat_re[:, :, 10, 0], T, decimal=6)

        npt.assert_almost_equal(hmat_re[:, :, :, 0] * 2.0, hmat_re[:, :, :, 1], decimal=6)
        npt.assert_almost_equal(hmat_re[:, :, :, 0] * 3.0, hmat_re[:, :, :, 2], decimal=5)

        hmat = channel.baseband_freq_response(coeff_list, delay_list, fc, pilot_grid=pilots, snap=[1, 2, 1, 0])
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal(hmat_re[:, :, 20, 3], T, decimal=5)
        npt.assert_almost_equal(hmat_im[:, :, 20, 3], T, decimal=5)

        T = np.array([[0.25, 1, -1], [0.5, 2, -2], [0.75, 3, -3], [1, 4, -4]])
        npt.assert_almost_equal(hmat_im[:, :, 15, 3], T, decimal=5)

        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 2.0, hmat_re[:, :, :, 0], decimal=5)
        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 3.0, hmat_re[:, :, :, 1], decimal=5)
        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 2.0, hmat_re[:, :, :, 2], decimal=5)

    # ==================================================================================
    # Error handling
    # ==================================================================================

    def test_error_no_coeff(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(delay=[np.zeros((1, 1, 1))], bandwidth=1e6)

    def test_error_coeff_and_coeff_re(self):
        c = [np.zeros((1, 1, 1), dtype=complex)]
        cr = [np.zeros((1, 1, 1))]
        ci = [np.zeros((1, 1, 1))]
        dl = [np.zeros((1, 1, 1))]
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=c, delay=dl, bandwidth=1e6, coeff_re=cr, coeff_im=ci)

    def test_error_coeff_re_without_im(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(delay=[np.zeros((1, 1, 1))], bandwidth=1e6,
                                           coeff_re=[np.zeros((1, 1, 1))])

    def test_error_coeff_im_without_re(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(delay=[np.zeros((1, 1, 1))], bandwidth=1e6,
                                           coeff_im=[np.zeros((1, 1, 1))])

    def test_error_no_delay(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1), dtype=complex)], bandwidth=1e6)

    def test_error_delay_not_list(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1), dtype=complex)],
                                           delay=np.zeros((1, 1, 1)), bandwidth=1e6)

    def test_error_coeff_not_list(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=np.zeros((1, 1, 1), dtype=complex),
                                           delay=[np.zeros((1, 1, 1))], bandwidth=1e6)

    def test_error_delay_length_mismatch(self):
        c = [np.zeros((1, 1, 1), dtype=complex)] * 2
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=c, delay=[np.zeros((1, 1, 1))], bandwidth=1e6)

    def test_error_cim_length_mismatch(self):
        cr = [np.zeros((1, 1, 1))] * 2
        ci = [np.zeros((1, 1, 1))]
        dl = [np.zeros((1, 1, 1))] * 2
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff_re=cr, coeff_im=ci, delay=dl, bandwidth=1e6)

    def test_error_2d_coeff(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((2, 3), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))], bandwidth=1e6)

    def test_error_mixed_ndim(self):
        c = [np.zeros((1, 1, 1), dtype=complex), np.zeros((1, 1, 1, 2), dtype=complex)]
        dl = [np.zeros((1, 1, 1))] * 2
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=c, delay=dl, bandwidth=1e6)

    def test_error_no_freq_grid(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))])

    def test_error_bandwidth_and_freq_out(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))],
                                           bandwidth=1e6, freq_out=np.array([1e9]))

    def test_error_4d_no_freq_in(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1, 2), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))],
                                           freq_out=np.array([1e9, 2e9]))

    def test_error_4d_no_freq_out(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1, 2), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))],
                                           freq_in=np.array([1e9, 2e9]))

    def test_error_4d_with_bandwidth(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1, 2), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))], bandwidth=1e6,
                                           freq_in=np.array([1e9, 2e9]), freq_out=np.array([1e9]))

    def test_error_4d_with_pilot_grid(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1, 2), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))],
                                           pilot_grid=np.array([0.0, 1.0]),
                                           freq_in=np.array([1e9, 2e9]), freq_out=np.array([1e9]))

    def test_error_n_rx_mismatch(self):
        c = [np.zeros((2, 1, 1), dtype=complex), np.zeros((3, 1, 1), dtype=complex)]
        dl = [np.zeros((1, 1, 1))] * 2
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=c, delay=dl, bandwidth=1e6)

    def test_error_delay_shape_mismatch(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 3), dtype=complex)],
                                           delay=[np.zeros((1, 1, 2))], bandwidth=1e6)

    def test_error_snap_out_of_range(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))], bandwidth=1e6,
                                           snap=np.array([5]))

    def test_error_4d_freq_in_dim_mismatch(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff=[np.zeros((1, 1, 1, 3), dtype=complex)],
                                           delay=[np.zeros((1, 1, 1))],
                                           freq_in=np.array([1e9, 2e9]),
                                           freq_out=np.array([1e9]))

    def test_error_re_im_shape_mismatch(self):
        with self.assertRaises(Exception):
            channel.baseband_freq_response(coeff_re=[np.zeros((1, 1, 2))],
                                           coeff_im=[np.zeros((1, 1, 3))],
                                           delay=[np.zeros((1, 1, 2))], bandwidth=1e6)

    def test_empty_list(self):
        hmat = channel.baseband_freq_response(coeff=[], delay=[], bandwidth=1e6)
        self.assertEqual(hmat.size, 0)

    # ==================================================================================
    # Single-freq: coeff_re + coeff_im
    # ==================================================================================

    def test_single_freq_re_im(self):
        coeff = np.array([[[1.0 + 0.5j, 0.5], [0.3 - 0.2j, -0.7j]]])  # (1,2,2)
        delay = np.zeros((1, 1, 2)); delay[0, 0, 0] = 5e-9; delay[0, 0, 1] = 15e-9
        cl = [coeff, 2.0 * coeff]; dl = [delay, delay]

        hmat_cx = channel.baseband_freq_response(coeff=cl, delay=dl, bandwidth=100e6)
        hmat_ri = channel.baseband_freq_response(coeff_re=[np.real(c) for c in cl],
                                                  coeff_im=[np.imag(c) for c in cl],
                                                  delay=dl, bandwidth=100e6)
        npt.assert_allclose(hmat_cx, hmat_ri, atol=1e-12, rtol=0)

    # ==================================================================================
    # Single-freq: freq_out instead of bandwidth
    # ==================================================================================

    def test_single_freq_with_freq_out(self):
        coeff = np.ones((1, 1, 1), dtype=complex)
        delay = np.zeros((1, 1, 1)); delay[0, 0, 0] = 10e-9
        bw = 100e6; n_c = 16; pilot = np.linspace(0, 1, n_c)
        hmat_ref = channel.baseband_freq_response(coeff=[coeff], delay=[delay], bandwidth=bw, pilot_grid=pilot)

        freq_out = np.linspace(0.0, bw, n_c)
        hmat_test = channel.baseband_freq_response(coeff=[coeff], delay=[delay], freq_out=freq_out)
        npt.assert_allclose(hmat_ref, hmat_test, atol=1e-10, rtol=0)

    # ==================================================================================
    # Single-freq: snap indices and varying n_path
    # ==================================================================================

    def test_single_freq_snap(self):
        c = [i * np.ones((1, 1, 1), dtype=complex) for i in [1, 2, 3]]
        dl = [np.zeros((1, 1, 1))] * 3
        hmat = channel.baseband_freq_response(coeff=c, delay=dl, bandwidth=1e6, carriers=4,
                                               snap=np.array([2, 0]))
        self.assertEqual(hmat.shape, (1, 1, 4, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 3.0, atol=1e-10, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 1.0, atol=1e-10, rtol=0)

    def test_single_freq_varying_n_path(self):
        c0 = np.ones((1, 1, 2), dtype=complex)
        c1 = np.ones((1, 1, 5), dtype=complex)
        hmat = channel.baseband_freq_response(coeff=[c0, c1],
                                               delay=[np.zeros((1, 1, 2)), np.zeros((1, 1, 5))],
                                               bandwidth=1e6, carriers=4)
        self.assertEqual(hmat.shape, (1, 1, 4, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 2.0, atol=1e-10, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 5.0, atol=1e-10, rtol=0)

    # ==================================================================================
    # Multi-freq core tests
    # ==================================================================================

    def test_mf_single_path_zero_delay(self):
        c = np.full((1, 1, 1, 1), 0.7 + 0j)
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=np.array([1e9]), freq_out=np.array([0.5e9, 1e9, 2e9]),
                                               remove_delay_phase=False)
        self.assertEqual(hmat.shape, (1, 1, 3, 1))
        for k in range(3):
            npt.assert_allclose(hmat[0, 0, k, 0], 0.7, atol=1e-12, rtol=0)

    def test_mf_delay_phase_rotation(self):
        tau = 10e-9; fout = np.array([1e9, 1.5e9, 2e9])
        dl = np.zeros((1, 1, 1)); dl[0, 0, 0] = tau
        hmat = channel.baseband_freq_response(coeff=[np.ones((1, 1, 1, 1), dtype=complex)],
                                               delay=[dl], freq_in=np.array([1e9]), freq_out=fout,
                                               remove_delay_phase=False)
        for k in range(3):
            npt.assert_allclose(hmat[0, 0, k, 0], np.exp(-1j * 2 * np.pi * fout[k] * tau), atol=1e-10, rtol=0)

    def test_mf_two_path(self):
        A1, A2, tau1, tau2 = 1.0, 0.5, 5e-9, 20e-9
        fout = np.array([1e9, 1.25e9, 1.5e9, 2e9])
        c = np.zeros((1, 1, 2, 1), dtype=complex); c[0, 0, 0, 0] = A1; c[0, 0, 1, 0] = A2
        dl = np.zeros((1, 1, 2)); dl[0, 0, 0] = tau1; dl[0, 0, 1] = tau2
        hmat = channel.baseband_freq_response(coeff=[c], delay=[dl], freq_in=np.array([1e9]),
                                               freq_out=fout, remove_delay_phase=False)
        for k in range(4):
            expected = A1 * np.exp(-1j * 2 * np.pi * fout[k] * tau1) + A2 * np.exp(-1j * 2 * np.pi * fout[k] * tau2)
            npt.assert_allclose(hmat[0, 0, k, 0], expected, atol=1e-10, rtol=0)

    def test_mf_slerp_magnitude(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1e9, 1.25e9, 1.5e9, 1.75e9, 2e9])
        c = np.zeros((1, 1, 1, 2), dtype=complex); c[0, 0, 0, 0] = 1.0; c[0, 0, 0, 1] = 3.0
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=fin, freq_out=fout, remove_delay_phase=False)
        for k in range(5):
            npt.assert_allclose(hmat[0, 0, k, 0].real, 1.0 + (fout[k] - 1e9) / 1e9 * 2.0, atol=1e-10, rtol=0)

    def test_mf_extrapolation(self):
        fin = np.array([1e9, 2e9]); fout = np.array([0.5e9, 1e9, 1.5e9, 2e9, 3e9])
        c = np.zeros((1, 1, 1, 2), dtype=complex); c[0, 0, 0, 0] = 1.0; c[0, 0, 0, 1] = 2.0
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=fin, freq_out=fout, remove_delay_phase=False)
        expected = [1.0, 1.0, 1.5, 2.0, 2.0]
        for k in range(5):
            npt.assert_allclose(hmat[0, 0, k, 0].real, expected[k], atol=1e-10, rtol=0)

    def test_mf_remove_delay_phase(self):
        fin = np.array([1e9, 2e9, 3e9]); fout = np.linspace(1e9, 3e9, 32); tau = 50e-9
        env = np.full((1, 1, 1, 3), 0.8 * np.exp(1j * 0.5))
        dl = np.zeros((1, 1, 1)); dl[0, 0, 0] = tau
        hmat_ref = channel.baseband_freq_response(coeff=[env], delay=[dl], freq_in=fin, freq_out=fout,
                                                   remove_delay_phase=False)
        hmat_test = channel.baseband_freq_response(coeff=[bake_delay_phase(env, dl, fin)], delay=[dl],
                                                    freq_in=fin, freq_out=fout, remove_delay_phase=True)
        npt.assert_allclose(hmat_test, hmat_ref, atol=1e-10, rtol=0)

    def test_mf_remove_delay_phase_multipath(self):
        fin = np.array([1e9, 1.5e9, 2e9]); fout = np.linspace(1e9, 2e9, 16)
        amps = [1.0, 0.5, 0.3]; phis = [0.0, 0.7, -0.4]; taus = [10e-9, 30e-9, 80e-9]
        env = np.zeros((1, 1, 3, 3), dtype=complex)
        for p in range(3): env[0, 0, p, :] = amps[p] * np.exp(1j * phis[p])
        dl = np.zeros((1, 1, 3)); dl[0, 0, :] = taus
        hmat_ref = channel.baseband_freq_response(coeff=[env], delay=[dl], freq_in=fin, freq_out=fout,
                                                   remove_delay_phase=False)
        hmat_test = channel.baseband_freq_response(coeff=[bake_delay_phase(env, dl, fin)], delay=[dl],
                                                    freq_in=fin, freq_out=fout, remove_delay_phase=True)
        npt.assert_allclose(hmat_test, hmat_ref, atol=1e-9, rtol=0)

    def test_mf_envelope_ramp_large_delay(self):
        fin = np.array([1e9, 2e9, 3e9]); fout = np.linspace(1e9, 3e9, 64); tau = 100e-9
        env = np.zeros((1, 1, 1, 3), dtype=complex)
        for f in range(3): env[0, 0, 0, f] = [1.0, 2.0, 3.0][f]
        dl = np.zeros((1, 1, 1)); dl[0, 0, 0] = tau
        hmat_ref = channel.baseband_freq_response(coeff=[env], delay=[dl], freq_in=fin, freq_out=fout,
                                                   remove_delay_phase=False)
        hmat_test = channel.baseband_freq_response(coeff=[bake_delay_phase(env, dl, fin)], delay=[dl],
                                                    freq_in=fin, freq_out=fout, remove_delay_phase=True)
        npt.assert_allclose(np.abs(hmat_test) ** 2, np.abs(hmat_ref) ** 2, atol=1e-10, rtol=0)
        npt.assert_allclose(hmat_test, hmat_ref, atol=1e-9, rtol=0)

    def test_mf_re_im(self):
        fin = np.array([1e9, 2e9]); fout = np.linspace(1e9, 2e9, 8)
        c = np.zeros((1, 1, 2, 2), dtype=complex)
        c[0, 0, 0, 0] = 1.0 + 0.2j; c[0, 0, 1, 0] = 0.5 - 0.3j
        c[0, 0, 0, 1] = 0.8 + 0.4j; c[0, 0, 1, 1] = 0.6 - 0.1j
        dl = np.zeros((1, 1, 2)); dl[0, 0, 0] = 10e-9; dl[0, 0, 1] = 30e-9

        hmat_cx = channel.baseband_freq_response(coeff=[c], delay=[dl], freq_in=fin, freq_out=fout,
                                                  remove_delay_phase=False)
        hmat_ri = channel.baseband_freq_response(coeff_re=[np.real(c)], coeff_im=[np.imag(c)],
                                                  delay=[dl], freq_in=fin, freq_out=fout,
                                                  remove_delay_phase=False)
        npt.assert_allclose(hmat_cx, hmat_ri, atol=1e-12, rtol=0)

    def test_mf_mimo(self):
        c = np.zeros((2, 2, 1, 1), dtype=complex)
        c[0, 0, 0, 0] = 1.0; c[1, 0, 0, 0] = 2.0; c[0, 1, 0, 0] = 3.0; c[1, 1, 0, 0] = 4.0
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=np.array([1e9]), freq_out=np.array([1e9]),
                                               remove_delay_phase=False)
        self.assertEqual(hmat.shape, (2, 2, 1, 1))
        npt.assert_allclose(hmat[0, 0, 0, 0].real, 1.0, atol=1e-12, rtol=0)
        npt.assert_allclose(hmat[1, 0, 0, 0].real, 2.0, atol=1e-12, rtol=0)
        npt.assert_allclose(hmat[0, 1, 0, 0].real, 3.0, atol=1e-12, rtol=0)
        npt.assert_allclose(hmat[1, 1, 0, 0].real, 4.0, atol=1e-12, rtol=0)

    def test_mf_spherical_delays(self):
        fout = np.array([1e9, 2e9]); tau0, tau1 = 5e-9, 15e-9
        dl = np.zeros((2, 1, 1)); dl[0, 0, 0] = tau0; dl[1, 0, 0] = tau1
        hmat = channel.baseband_freq_response(coeff=[np.ones((2, 1, 1, 1), dtype=complex)],
                                               delay=[dl], freq_in=np.array([1e9]), freq_out=fout,
                                               remove_delay_phase=False)
        for k in range(2):
            npt.assert_allclose(hmat[0, 0, k, 0], np.exp(-1j * 2 * np.pi * fout[k] * tau0), atol=1e-10, rtol=0)
            npt.assert_allclose(hmat[1, 0, k, 0], np.exp(-1j * 2 * np.pi * fout[k] * tau1), atol=1e-10, rtol=0)

    def test_mf_snapshots_varying_n_path(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1.5e9])
        c0 = np.ones((1, 1, 1, 2), dtype=complex)
        c1 = np.ones((1, 1, 3, 2), dtype=complex)
        hmat = channel.baseband_freq_response(coeff=[c0, c1],
                                               delay=[np.zeros((1, 1, 1)), np.zeros((1, 1, 3))],
                                               freq_in=fin, freq_out=fout, remove_delay_phase=False)
        self.assertEqual(hmat.shape, (1, 1, 1, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 1.0, atol=1e-10, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 3.0, atol=1e-10, rtol=0)

    def test_mf_snap_indices(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1.5e9])
        cs = [np.ones((1, 1, 1, 2), dtype=complex) * v for v in [1, 2, 3]]
        dl = [np.zeros((1, 1, 1))] * 3
        hmat = channel.baseband_freq_response(coeff=cs, delay=dl, freq_in=fin, freq_out=fout,
                                               snap=np.array([2, 0]), remove_delay_phase=False)
        self.assertEqual(hmat.shape, (1, 1, 1, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 3.0, atol=1e-10, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 1.0, atol=1e-10, rtol=0)

    def test_mf_3d_delay_broadcast(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1e9]); tau = 10e-9
        c = np.ones((1, 1, 1, 2), dtype=complex)
        dl3 = np.zeros((1, 1, 1)); dl3[0, 0, 0] = tau
        dl4 = np.zeros((1, 1, 1, 2)); dl4[0, 0, 0, :] = tau
        h3 = channel.baseband_freq_response(coeff=[c], delay=[dl3], freq_in=fin, freq_out=fout,
                                             remove_delay_phase=False)
        h4 = channel.baseband_freq_response(coeff=[c], delay=[dl4], freq_in=fin, freq_out=fout,
                                             remove_delay_phase=False)
        npt.assert_allclose(h3, h4, atol=1e-12, rtol=0)

    def test_mf_three_segments(self):
        fin = np.array([1e9, 2e9, 4e9]); fout = np.array([1.5e9, 2e9, 3e9])
        c = np.zeros((1, 1, 1, 3), dtype=complex); c[0, 0, 0, :] = [1.0, 3.0, 7.0]
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=fin, freq_out=fout, remove_delay_phase=False)
        npt.assert_allclose(hmat[0, 0, 0, 0].real, 2.0, atol=1e-10, rtol=0)
        npt.assert_allclose(hmat[0, 0, 1, 0].real, 3.0, atol=1e-10, rtol=0)
        npt.assert_allclose(hmat[0, 0, 2, 0].real, 5.0, atol=1e-10, rtol=0)

    def test_mf_acoustic(self):
        tau = 10.0 / 343.0
        fin = np.array([500.0, 1000.0, 2000.0]); fout = np.array([500.0, 750.0, 1000.0, 1500.0, 2000.0])
        dl = np.zeros((1, 1, 1)); dl[0, 0, 0] = tau
        hmat = channel.baseband_freq_response(coeff=[np.ones((1, 1, 1, 3), dtype=complex)],
                                               delay=[dl], freq_in=fin, freq_out=fout,
                                               remove_delay_phase=False)
        npt.assert_allclose(np.abs(hmat[0, 0, :, 0]) ** 2, np.ones(5), atol=1e-10, rtol=0)
        npt.assert_allclose(hmat[0, 0, 2, 0], np.exp(-1j * 2 * np.pi * 1000.0 * tau), atol=1e-10, rtol=0)

    def test_mf_remove_phase_spherical(self):
        fin = np.array([1e9, 2e9]); fout = np.linspace(1e9, 2e9, 8)
        env = np.ones((2, 1, 1, 2), dtype=complex)
        dl = np.zeros((2, 1, 1)); dl[0, 0, 0] = 10e-9; dl[1, 0, 0] = 40e-9
        hmat_ref = channel.baseband_freq_response(coeff=[env], delay=[dl], freq_in=fin, freq_out=fout,
                                                   remove_delay_phase=False)
        hmat_test = channel.baseband_freq_response(coeff=[bake_delay_phase(env, dl, fin)], delay=[dl],
                                                    freq_in=fin, freq_out=fout, remove_delay_phase=True)
        npt.assert_allclose(hmat_test, hmat_ref, atol=1e-9, rtol=0)

    def test_mf_phase_unwrap(self):
        c = np.zeros((1, 1, 1, 2), dtype=complex)
        c[0, 0, 0, 0] = np.exp(1j * 2.8); c[0, 0, 0, 1] = np.exp(-1j * 2.8)
        hmat = channel.baseband_freq_response(coeff=[c], delay=[np.zeros((1, 1, 1))],
                                               freq_in=np.array([1e9, 2e9]), freq_out=np.array([1.5e9]),
                                               remove_delay_phase=False)
        self.assertLess(hmat[0, 0, 0, 0].real, -0.99)

    # ==================================================================================
    # Output shape checks
    # ==================================================================================

    def test_output_shape_single_freq(self):
        hmat = channel.baseband_freq_response(coeff=[np.ones((3, 2, 4), dtype=complex)],
                                               delay=[np.zeros((1, 1, 4))], bandwidth=1e6, carriers=16)
        self.assertEqual(hmat.shape, (3, 2, 16, 1))

    def test_output_shape_multi_freq(self):
        hmat = channel.baseband_freq_response(coeff=[np.ones((2, 3, 5, 2), dtype=complex)],
                                               delay=[np.zeros((1, 1, 5))],
                                               freq_in=np.array([1e9, 2e9]),
                                               freq_out=np.linspace(1e9, 2e9, 64))
        self.assertEqual(hmat.shape, (2, 3, 64, 1))


if __name__ == '__main__':
    unittest.main()