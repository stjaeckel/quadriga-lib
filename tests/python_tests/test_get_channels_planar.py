# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import math
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import arrayant

C = 299792458.0
PI = math.pi


def omni():
    return arrayant.generate('omni')


def xpol():
    return arrayant.generate('xpol')


def make_M(n_path, vv=1.0, hh=-1.0):
    M = np.zeros((8, n_path), order='F')
    M[0, :] = vv
    M[6, :] = hh
    return M


def amp2(cr, ci):
    return np.sqrt(cr ** 2 + ci ** 2)


def minimal_ant():
    # 2-element omni; element 1 has twice the field amplitude, elements offset in y
    ant = omni()
    ant = arrayant.copy_element(ant, 0, 1)
    ant['e_theta_re'][:, :, 1] = 2.0
    ant['e_theta_im'][:, :, 1] = 0.0
    ant['e_phi_re'][:, :, 1] = 0.0
    ant['e_phi_im'][:, :, 1] = 0.0
    ant['element_pos'] = np.array([[0, 0], [1, -1], [0, 0]], order='F', dtype=float)
    ant['coupling_re'] = np.eye(2, order='F')
    ant['coupling_im'] = np.zeros((2, 2), order='F')
    return ant


def minimal_paths():
    aod = np.radians([0, 90])
    eod = np.radians([0, 45])
    aoa = np.radians([180, 180 - math.degrees(math.atan(10 / 20))])
    eoa = np.radians([0, math.degrees(math.atan(10 / math.sqrt(10 ** 2 + 20 ** 2)))])
    path_gain = np.array([1.0, 0.25])
    path_length = np.array([20.0, math.hypot(10, 10) + math.sqrt(10 ** 2 + 20 ** 2 + 10 ** 2)])
    M = np.array([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-1, -1], [0, 0]],
                 order='F', dtype=float)
    return aod, eod, aoa, eoa, path_gain, path_length, M


class TestGetChannelsPlanar(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Minimal test - amplitude, delay, Doppler (mirrors C++ minimal test)
    # -------------------------------------------------------------------------

    def test_minimal(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, path_gain, path_length, M = minimal_paths()

        coeff_re, coeff_im, delay, rx_Doppler = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)

        # Power = |coeff|^2
        amp = coeff_re ** 2 + coeff_im ** 2
        npt.assert_almost_equal(amp[:, :, 0], np.array([[1, 4], [4, 16]]), decimal=13)
        npt.assert_almost_equal(amp[:, :, 1], np.array([[0.25, 1], [1, 4]]), decimal=13)

        d0 = 20.0
        e0 = math.sqrt(9 ** 2 + 10 ** 2) + math.sqrt(9 ** 2 + 20 ** 2 + 10 ** 2)
        e1 = math.sqrt(9 ** 2 + 10 ** 2) + math.sqrt(11 ** 2 + 20 ** 2 + 10 ** 2)
        e2 = math.sqrt(11 ** 2 + 10 ** 2) + math.sqrt(9 ** 2 + 20 ** 2 + 10 ** 2)
        e3 = math.sqrt(11 ** 2 + 10 ** 2) + math.sqrt(11 ** 2 + 20 ** 2 + 10 ** 2)
        npt.assert_almost_equal(delay[:, :, 0], np.array([[d0, d0], [d0, d0]]) / C, decimal=13)
        npt.assert_almost_equal(delay[:, :, 1], np.array([[e0, e2], [e1, e3]]) / C, decimal=10)

        doppler = math.cos(aoa[1]) * math.cos(eoa[1])
        npt.assert_almost_equal(rx_Doppler, np.array([-1.0, doppler]), decimal=13)
        self.assertEqual(rx_Doppler.shape, (2,))

    # -------------------------------------------------------------------------
    # Fake LOS path (mirrors C++ "Fake LOS" test)
    # -------------------------------------------------------------------------

    def test_fake_los_path(self):
        ant = omni()

        # LOS and NLOS swapped vs minimal; the real LOS is path index 1
        aod = np.radians([90, 0])
        eod = np.radians([45, 0])
        aoa = np.array([PI - math.atan(10 / 20), PI])
        eoa = np.array([math.atan(10 / math.sqrt(10 ** 2 + 20 ** 2)), 0.0])
        path_gain = np.array([0.25, 1.0])
        path_length = np.array([math.hypot(10, 10) + math.sqrt(10 ** 2 + 20 ** 2 + 10 ** 2), 20.0])
        M = make_M(2)

        coeff_re, coeff_im, delay, rx_Doppler = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True, True)

        self.assertEqual(coeff_re.shape[2], 3)
        self.assertEqual(coeff_im.shape[2], 3)
        self.assertEqual(delay.shape[2], 3)
        self.assertEqual(rx_Doppler.shape, (3,))

        amp = amp2(coeff_re, coeff_im)
        npt.assert_almost_equal(amp[0, 0, 0], 1.0, decimal=6)   # LOS
        npt.assert_almost_equal(amp[0, 0, 1], 0.5, decimal=6)   # NLOS
        npt.assert_almost_equal(amp[0, 0, 2], 0.0, decimal=6)   # fake (zero power)

        d0 = 20.0
        e0 = math.hypot(10, 10) + math.sqrt(10 ** 2 + 20 ** 2 + 10 ** 2)
        npt.assert_almost_equal(delay[0, 0, 0], d0 / C, decimal=13)
        npt.assert_almost_equal(delay[0, 0, 1], e0 / C, decimal=12)
        npt.assert_almost_equal(delay[0, 0, 2], d0 / C, decimal=13)

    # -------------------------------------------------------------------------
    # Coupling (mirrors C++ "Coupling" test, 3 sub-cases)
    # -------------------------------------------------------------------------

    def test_coupling(self):
        # 3-element array. Only elements 0 and 2 are copies of the omni source;
        # index 1 is left as a zero pattern (copy_element 0 -> 2 only), so it
        # contributes no field. Elements 0 and 2 are offset in y by +/-1.
        def base_ant():
            a = omni()
            a = arrayant.copy_element(a, 0, 2)
            a['element_pos'] = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], order='F', dtype=float)
            return a

        aod = np.zeros(1)
        eod = np.zeros(1)
        aoa = np.array([PI])
        eoa = np.zeros(1)
        path_gain = np.array([1.0])
        path_length = np.array([20.0])
        M = make_M(1)

        # Case 1: identity coupling (from copy_element), relative delays, fake LOS appended
        ant = base_ant()
        cr, ci, delay, _ = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 0.0, False, True)

        self.assertEqual(cr.shape, (3, 3, 2))
        expected = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=float)
        npt.assert_almost_equal(cr[:, :, 0], expected, decimal=13)
        npt.assert_almost_equal(ci[:, :, 0], np.zeros((3, 3)), decimal=13)
        npt.assert_almost_equal(cr[:, :, 1], np.zeros((3, 3)), decimal=13)     # fake LOS
        npt.assert_almost_equal(delay[:, :, 0], np.zeros((3, 3)), decimal=13)  # relative, all == LOS

        # Case 2: all elements summed into a single port (weight 1)
        ant = base_ant()
        ant['coupling_re'] = np.ones((3, 1), order='F')
        ant.pop('coupling_im', None)
        cr, ci, delay, _ = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 0.0, True, False)

        self.assertEqual(cr.shape, (1, 1, 1))
        npt.assert_almost_equal(cr[0, 0, 0], 4.0, decimal=13)   # (1 + 0 + 1)^2
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=13)
        npt.assert_almost_equal(delay[0, 0, 0], 20.0 / C, decimal=13)

        # Case 3: single port, coupling weight 2
        ant = base_ant()
        ant['coupling_re'] = np.ones((3, 1), order='F') * 2.0
        ant.pop('coupling_im', None)
        cr, ci, delay, _ = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 0.0, True, False)

        self.assertEqual(cr.size, 1)
        npt.assert_almost_equal(cr[0, 0, 0], 16.0, decimal=13)  # (2*(1 + 0 + 1))^2
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=13)
        npt.assert_almost_equal(delay[0, 0, 0], 20.0 / C, decimal=13)

    # -------------------------------------------------------------------------
    # complex=True output matches the Re/Im split (new feature)
    # -------------------------------------------------------------------------

    def test_complex_output_matches_split(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, path_gain, path_length, M = minimal_paths()
        args = (ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
                [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0])
        kwargs = dict(center_freq=2997924580.0, use_absolute_delays=True)

        cr, ci, delay_r, dop_r = arrayant.get_channels_planar(*args, **kwargs)
        coeff, delay_c, dop_c = arrayant.get_channels_planar(*args, complex=True, **kwargs)

        self.assertEqual(len(arrayant.get_channels_planar(*args, **kwargs)), 4)
        self.assertEqual(len(arrayant.get_channels_planar(*args, complex=True, **kwargs)), 3)

        self.assertTrue(np.iscomplexobj(coeff))
        npt.assert_array_equal(coeff.real, cr)
        npt.assert_array_equal(coeff.imag, ci)
        npt.assert_array_equal(delay_c, delay_r)
        npt.assert_array_equal(dop_c, dop_r)

    def test_complex_with_fake_los(self):
        ant = omni()
        aod = np.radians([90, 0])
        eod = np.radians([45, 0])
        aoa = np.array([PI - math.atan(10 / 20), PI])
        eoa = np.array([math.atan(10 / math.sqrt(10 ** 2 + 20 ** 2)), 0.0])
        path_gain = np.array([0.25, 1.0])
        path_length = np.array([math.hypot(10, 10) + math.sqrt(10 ** 2 + 20 ** 2 + 10 ** 2), 20.0])
        M = make_M(2)
        args = (ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
                [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0])
        kwargs = dict(center_freq=2997924580.0, use_absolute_delays=True, add_fake_los_path=True)

        cr, ci, _, _ = arrayant.get_channels_planar(*args, **kwargs)
        coeff, _, _ = arrayant.get_channels_planar(*args, complex=True, **kwargs)

        self.assertEqual(coeff.shape[2], 3)
        self.assertTrue(np.iscomplexobj(coeff))
        npt.assert_array_equal(coeff.real, cr)
        npt.assert_array_equal(coeff.imag, ci)

    # -------------------------------------------------------------------------
    # rx_Doppler equals cos(aoa)*cos(eoa)
    # -------------------------------------------------------------------------

    def test_doppler_directions(self):
        ant = omni()
        path_gain = np.array([1.0])
        path_length = np.array([100.0])
        M = make_M(1)
        for aoa_v, eoa_v in [(0.0, 0.0), (PI / 2, 0.0), (PI, 0.0), (0.0, PI / 2)]:
            _, _, _, dop = arrayant.get_channels_planar(
                ant, ant, np.zeros(1), np.zeros(1), np.array([aoa_v]), np.array([eoa_v]),
                path_gain, path_length, M,
                [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True)
            npt.assert_almost_equal(dop[0], math.cos(aoa_v) * math.cos(eoa_v), decimal=12)

    # -------------------------------------------------------------------------
    # center_freq = 0 disables phase
    # -------------------------------------------------------------------------

    def test_center_freq_zero_no_phase(self):
        ant = omni()
        cr, ci, _, _ = arrayant.get_channels_planar(
            ant, ant, np.zeros(1), np.zeros(1), np.array([PI]), np.zeros(1),
            np.array([1.0]), np.array([100.0]), make_M(1),
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True)
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=6)
        npt.assert_almost_equal(amp2(cr, ci)[0, 0, 0], 1.0, decimal=5)

    # -------------------------------------------------------------------------
    # Absolute vs relative delays
    # -------------------------------------------------------------------------

    def test_absolute_vs_relative_delays(self):
        ant = omni()
        dist = 50.0
        common = (ant, ant, np.zeros(1), np.zeros(1), np.array([PI]), np.zeros(1),
                  np.array([1.0]), np.array([dist]), make_M(1),
                  [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0])
        cr_a, ci_a, d_a, _ = arrayant.get_channels_planar(*common, center_freq=0.0, use_absolute_delays=True)
        cr_r, ci_r, d_r, _ = arrayant.get_channels_planar(*common, center_freq=0.0, use_absolute_delays=False)

        npt.assert_almost_equal(d_a[0, 0, 0], dist / C, decimal=12)
        npt.assert_almost_equal(d_r[0, 0, 0], 0.0, decimal=12)
        npt.assert_almost_equal(cr_a[0, 0, 0], cr_r[0, 0, 0], decimal=12)
        npt.assert_almost_equal(ci_a[0, 0, 0], ci_r[0, 0, 0], decimal=12)

    # -------------------------------------------------------------------------
    # Phase introduced by a non-zero center frequency
    # -------------------------------------------------------------------------

    def test_phase_with_center_frequency(self):
        ant = omni()
        dist = 100.0
        fc = 1.0e9
        common = (ant, ant, np.zeros(1), np.zeros(1), np.array([PI]), np.zeros(1),
                  np.array([1.0]), np.array([dist]), make_M(1),
                  [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0])
        cr_f, ci_f, d_f, _ = arrayant.get_channels_planar(*common, center_freq=fc, use_absolute_delays=True)
        cr_0, ci_0, d_0, _ = arrayant.get_channels_planar(*common, center_freq=0.0, use_absolute_delays=True)

        npt.assert_almost_equal(amp2(cr_f, ci_f)[0, 0, 0], amp2(cr_0, ci_0)[0, 0, 0], decimal=8)
        npt.assert_almost_equal(d_f[0, 0, 0], d_0[0, 0, 0], decimal=12)
        rotated = (abs(cr_f[0, 0, 0] - cr_0[0, 0, 0]) > 1e-6) or (abs(ci_f[0, 0, 0] - ci_0[0, 0, 0]) > 1e-6)
        self.assertTrue(rotated)

    # -------------------------------------------------------------------------
    # Complex polarization matrix introduces phase
    # -------------------------------------------------------------------------

    def test_complex_M_introduces_phase(self):
        ant = omni()
        s2 = math.sqrt(2.0) / 2.0
        M = np.zeros((8, 1), order='F')
        M[0, 0] = s2; M[1, 0] = s2     # VV = exp(j*pi/4)
        M[6, 0] = -s2; M[7, 0] = -s2   # HH = -exp(j*pi/4)

        cr, ci, _, _ = arrayant.get_channels_planar(
            ant, ant, np.zeros(1), np.zeros(1), np.array([PI]), np.zeros(1),
            np.array([1.0]), np.array([100.0]), M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True)

        npt.assert_almost_equal(amp2(cr, ci)[0, 0, 0], 1.0, decimal=5)
        self.assertGreater(abs(ci[0, 0, 0]), 1e-3)

    # -------------------------------------------------------------------------
    # Cross-polarization via M
    # -------------------------------------------------------------------------

    def test_xpol_via_M(self):
        probe = xpol()
        z = np.zeros(1)
        pos = ([0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0])

        def run(M):
            return arrayant.get_channels_planar(
                probe, probe, z, z, np.array([PI]), z,
                np.array([1.0]), np.array([100.0]), M, *pos, 0.0, True)

        M_vh = np.zeros((8, 1), order='F'); M_vh[2, 0] = 1.0
        M_vv = np.zeros((8, 1), order='F'); M_vv[0, 0] = 1.0
        M_z = np.zeros((8, 1), order='F')

        cr_vh, ci_vh, *_ = run(M_vh)
        cr_vv, ci_vv, *_ = run(M_vv)
        cr_z, ci_z, *_ = run(M_z)

        self.assertGreater(np.sum(amp2(cr_vh, ci_vh) ** 2), 1e-6)
        self.assertGreater(np.sum(amp2(cr_vv, ci_vv) ** 2), 1e-6)
        self.assertLess(np.sum(amp2(cr_z, ci_z) ** 2), 1e-12)
        self.assertFalse(np.allclose(cr_vh, cr_vv, atol=1e-6))

    # -------------------------------------------------------------------------
    # RX bank rotation swaps xpol amplitudes
    # -------------------------------------------------------------------------

    def test_rx_rotation_swaps_xpol(self):
        tx = omni()
        rx = xpol()
        z = np.zeros(1)
        common = (tx, rx, z, z, np.array([PI]), z, np.array([1.0]), np.array([100.0]), make_M(1))

        cr0, ci0, _, _ = arrayant.get_channels_planar(
            *common, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, False)
        cr_r, ci_r, _, _ = arrayant.get_channels_planar(
            *common, [0, 0, 0], [0, 0, 0], [100, 0, 0], [PI / 2, 0, 0], 0.0, False)

        a_v0 = amp2(cr0, ci0)[0, 0, 0]
        a_h0 = amp2(cr0, ci0)[1, 0, 0]
        a_vr = amp2(cr_r, ci_r)[0, 0, 0]
        a_hr = amp2(cr_r, ci_r)[1, 0, 0]
        npt.assert_almost_equal(a_v0, a_hr, decimal=4)
        npt.assert_almost_equal(a_h0, a_vr, decimal=4)

    # -------------------------------------------------------------------------
    # Output shapes incl. fake-LOS n_path+1 (regression for the wrapper fix)
    # -------------------------------------------------------------------------

    def test_output_shape_mimo(self):
        tx = omni()
        tx = arrayant.copy_element(tx, 0, 1)
        tx = arrayant.copy_element(tx, 0, 2)
        tx['element_pos'] = np.array([[0, 0, 0], [0.5, 0.0, -0.5], [0, 0, 0]], order='F', dtype=float)
        tx['coupling_re'] = np.eye(3, order='F')
        tx['coupling_im'] = np.zeros((3, 3), order='F')

        rx = omni()
        rx = arrayant.copy_element(rx, 0, 1)
        rx['element_pos'] = np.array([[0, 0], [0.5, -0.5], [0, 0]], order='F', dtype=float)
        rx['coupling_re'] = np.eye(2, order='F')
        rx['coupling_im'] = np.zeros((2, 2), order='F')

        n_path = 4
        aod = np.array([0.0, 0.1, -0.1, 0.2])
        eod = np.array([0.0, 0.05, 0.1, 0.0])
        aoa = np.array([PI, PI - 0.1, PI + 0.1, PI])
        eoa = np.array([0.0, 0.05, -0.05, 0.1])
        path_gain = np.ones(n_path)
        path_length = np.array([100.0, 105.0, 110.0, 102.0])
        M = make_M(n_path)

        cr, ci, delay, dop = arrayant.get_channels_planar(
            tx, rx, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)
        for arr in (cr, ci, delay):
            self.assertEqual(arr.shape, (2, 3, n_path))
        self.assertEqual(dop.shape, (n_path,))

        cr2, ci2, delay2, dop2 = arrayant.get_channels_planar(
            tx, rx, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, True)
        for arr in (cr2, ci2, delay2):
            self.assertEqual(arr.shape, (2, 3, n_path + 1))
        self.assertEqual(dop2.shape, (n_path + 1,))

    # -------------------------------------------------------------------------
    # Zero path gain produces a zero coefficient
    # -------------------------------------------------------------------------

    def test_zero_path_gain(self):
        ant = omni()
        aod = np.zeros(2)
        eod = np.zeros(2)
        aoa = np.array([PI, PI])
        eoa = np.zeros(2)
        path_gain = np.array([1.0, 0.0])
        path_length = np.array([100.0, 100.0])
        M = make_M(2)

        cr, ci, _, _ = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True)

        self.assertGreater(amp2(cr, ci)[0, 0, 0], 0.0)
        npt.assert_almost_equal(cr[0, 0, 1], 0.0, decimal=12)
        npt.assert_almost_equal(ci[0, 0, 1], 0.0, decimal=12)

    # -------------------------------------------------------------------------
    # Error handling (mirrors MEX exception tests)
    # -------------------------------------------------------------------------

    def test_element_pos_optional(self):
        ant = minimal_ant()
        ant.pop('element_pos', None)
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        result = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, pg, pl, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(len(result), 4)

    def test_error_coupling_im_without_re(self):
        ant = minimal_ant()
        ant.pop('element_pos', None)
        ant.pop('coupling_re', None)   # leaves coupling_im only
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            arrayant.get_channels_planar(
                ant, ant, aod, eod, aoa, eoa, pg, pl, M,
                [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(str(ctx.exception),
                         "Transmit antenna: Imaginary part of coupling matrix (phase component) "
                         "defined without real part (absolute component)")

    def test_no_coupling_keys_accepted(self):
        ant = minimal_ant()
        ant.pop('element_pos', None)
        ant.pop('coupling_re', None)
        ant.pop('coupling_im', None)
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        result = arrayant.get_channels_planar(
            ant, ant, aod, eod, aoa, eoa, pg, pl, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(len(result), 4)

    def test_error_mismatched_n_path(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            arrayant.get_channels_planar(
                ant, ant, aod[[0, 1, 1]], eod, aoa, eoa, pg, pl, M,
                [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(str(ctx.exception),
                         "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' "
                         "must have the same number of columns (n_paths).")

    def test_error_tx_pos_wrong_size(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            arrayant.get_channels_planar(
                ant, ant, aod, eod, aoa, eoa, pg, pl, M,
                [0, 0, 1, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(str(ctx.exception), "Input 'tx_pos' has incorrect number of elements.")


if __name__ == '__main__':
    unittest.main()