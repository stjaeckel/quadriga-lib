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


class TestGetChannelsSpherical(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Minimal test — angles, amplitude, delay, phase (mirrors C++ minimal test)
    # -------------------------------------------------------------------------

    def test_minimal(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant['e_theta_re'][:, :, 1] = 2.0
        ant['e_theta_im'][:, :, 1] = 0.0
        ant['e_phi_re'][:, :, 1] = 0.0
        ant['e_phi_im'][:, :, 1] = 0.0
        ant['element_pos'] = np.array([[0, 0], [1, -1], [0, 0]], order='F')
        ant['coupling_re'] = np.eye(2, order='F')
        ant['coupling_im'] = np.zeros((2, 2), order='F')

        fbs_pos = np.array([[10, 0], [0, 10], [1, 11]], order='F')
        path_gain = np.array([1.0, 0.25])
        path_length = np.zeros(2)
        M = np.array([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-1, -1], [0, 0]], order='F', dtype=float)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True, False, angles=True)

        aod_d = np.degrees(aod)
        eod_d = np.degrees(eod)
        aoa_d = np.degrees(aoa)
        eoa_d = np.degrees(eoa)

        # AOD
        alpha = np.degrees(np.arctan(2.0 / 20.0))
        npt.assert_almost_equal(aod_d[:, :, 0], [[0, alpha], [-alpha, 0]], decimal=12)
        npt.assert_almost_equal(aod_d[:, :, 1], [[90, 90], [90, 90]], decimal=12)

        # AOA
        alpha_aoa = 180.0 - alpha
        npt.assert_almost_equal(np.cos(np.radians(aoa_d[:, :, 0])),
                                np.cos(np.radians([[180, -alpha_aoa], [alpha_aoa, 180]])), decimal=11)
        alpha2 = 180.0 - np.degrees(np.arctan(9.0 / 20.0))
        beta2 = 180.0 - np.degrees(np.arctan(11.0 / 20.0))
        npt.assert_almost_equal(aoa_d[:, :, 1], [[alpha2, alpha2], [beta2, beta2]], decimal=11)

        # EOD
        alpha = np.degrees(np.arctan(10.0 / 11.0))
        beta = np.degrees(np.arctan(10.0 / 9.0))
        npt.assert_almost_equal(eod_d[:, :, 0], [[0, 0], [0, 0]], decimal=12)
        npt.assert_almost_equal(eod_d[:, :, 1], [[beta, alpha], [beta, alpha]], decimal=11)

        # EOA
        alpha = np.degrees(np.arctan(10.0 / math.sqrt(9.0 ** 2 + 20.0 ** 2)))
        beta = np.degrees(np.arctan(10.0 / math.sqrt(11.0 ** 2 + 20.0 ** 2)))
        npt.assert_almost_equal(eoa_d[:, :, 0], [[0, 0], [0, 0]], decimal=12)
        npt.assert_almost_equal(eoa_d[:, :, 1], [[alpha, alpha], [beta, beta]], decimal=11)

        # Amplitude
        amp = cr ** 2 + ci ** 2
        npt.assert_almost_equal(amp[:, :, 0], [[1, 4], [4, 16]], decimal=11)
        npt.assert_almost_equal(amp[:, :, 1], [[0.25, 1], [1, 4]], decimal=11)

        # Delays
        d0 = 20.0
        d1 = math.hypot(20.0, 2.0)
        e0 = math.hypot(9.0, 10.0) + math.sqrt(9.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
        e1 = math.hypot(9.0, 10.0) + math.sqrt(11.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
        e2 = math.hypot(11.0, 10.0) + math.sqrt(9.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
        e3 = math.hypot(11.0, 10.0) + math.sqrt(11.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
        npt.assert_almost_equal(delay[:, :, 0], np.array([[d0, d1], [d1, d0]]) / C, decimal=11)
        npt.assert_almost_equal(delay[:, :, 1], np.array([[e0, e2], [e1, e3]]) / C, decimal=11)

        # Phase (use_absolute_delays=True so delay is not subtracted)
        fc = 2997924580.0
        wavelength = C / fc
        wk = 2.0 * PI / wavelength

        def expected_phase(dist):
            return math.fmod(wk * math.fmod(dist, wavelength) + 2 * PI, 2 * PI)

        def actual_phase(r, i):
            p = math.atan2(-i, r)
            return math.fmod(p + 2 * PI, 2 * PI)

        npt.assert_almost_equal(actual_phase(cr[0, 0, 0], ci[0, 0, 0]), expected_phase(d0), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[1, 0, 0], ci[1, 0, 0]), expected_phase(d1), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[0, 1, 0], ci[0, 1, 0]), expected_phase(d1), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[1, 1, 0], ci[1, 1, 0]), expected_phase(d0), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[0, 0, 1], ci[0, 0, 1]), expected_phase(e0), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[1, 0, 1], ci[1, 0, 1]), expected_phase(e1), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[0, 1, 1], ci[0, 1, 1]), expected_phase(e2), decimal=3)
        npt.assert_almost_equal(actual_phase(cr[1, 1, 1], ci[1, 1, 1]), expected_phase(e3), decimal=3)

    # -------------------------------------------------------------------------
    # complex=True output matches Re/Im split
    # -------------------------------------------------------------------------

    def test_complex_output_matches_split(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant['e_theta_re'][:, :, 1] = 2.0
        ant['element_pos'] = np.array([[0, 0], [1, -1], [0, 0]], order='F')
        ant['coupling_re'] = np.eye(2, order='F')
        ant['coupling_im'] = np.zeros((2, 2), order='F')

        fbs_pos = np.array([[10, 0], [0, 10], [1, 11]], order='F')
        path_gain = np.array([1.0, 0.25])
        path_length = np.zeros(2)
        M = np.array([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-1, -1], [0, 0]], order='F', dtype=float)
        kwargs = dict(center_freq=2997924580.0, use_absolute_delays=True)

        cr, ci, delay_r = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], **kwargs)

        coeff, delay_c = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], complex=True, **kwargs)

        self.assertTrue(np.iscomplexobj(coeff))
        npt.assert_array_equal(coeff.real, cr)
        npt.assert_array_equal(coeff.imag, ci)
        npt.assert_array_equal(delay_c, delay_r)

    def test_complex_output_with_angles(self):
        ant = omni()
        fbs_pos = np.array([[50.0], [0.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([100.0])
        M = make_M(1)

        cr, ci, delay_r, aod_r, eod_r, aoa_r, eoa_r = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False, angles=True)

        coeff, delay_c, aod_c, eod_c, aoa_c, eoa_c = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False,
            angles=True, complex=True)

        self.assertTrue(np.iscomplexobj(coeff))
        npt.assert_array_equal(coeff.real, cr)
        npt.assert_array_equal(coeff.imag, ci)
        npt.assert_array_equal(delay_c, delay_r)
        npt.assert_array_equal(aod_c, aod_r)
        npt.assert_array_equal(eod_c, eod_r)
        npt.assert_array_equal(aoa_c, aoa_r)
        npt.assert_array_equal(eoa_c, eoa_r)

    # -------------------------------------------------------------------------
    # TX rotation
    # -------------------------------------------------------------------------

    def test_tx_rotation_bank(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant['e_theta_re'][:, :, 1] = 2.0
        ant['element_pos'] = np.array([[0, 0], [15, -15], [0, 0]], order='F')
        ant['coupling_re'] = np.eye(2, order='F')
        ant['coupling_im'] = np.zeros((2, 2), order='F')
        probe = xpol()

        fbs_pos = np.array([[10.0], [0.0], [1.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        d0 = (math.sqrt(20.0 ** 2 + 15.0 ** 2) - 20.0) / C

        # Bank = -pi/2: elements should be vertical, delay symmetric
        cr, ci, delay = arrayant.get_channels_spherical(
            ant, probe, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [-PI / 2, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, False)

        npt.assert_almost_equal(delay[:, :, 0], [[d0, d0], [d0, d0]], decimal=11)
        npt.assert_almost_equal(cr[1, 0, 0], 1.0, decimal=5)
        npt.assert_almost_equal(cr[1, 1, 0], 2.0, decimal=5)
        npt.assert_almost_equal(ci[:, :, 0], np.zeros((2, 2)), decimal=5)

        # Bank = -pi/2, tilt = pi/2: delays flip sign
        cr2, ci2, delay2 = arrayant.get_channels_spherical(
            ant, probe, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [-PI / 2, PI / 2, 0], [20, 0, 1], [PI / 2, 0, 0], 2997924580.0, False)

        # Bank = -pi/2, tilt = pi/2: elements shift along z → delays ±15/C
        d1 = 15.0 / C
        npt.assert_almost_equal(delay2[0, 0, 0], -d1, decimal=14)
        npt.assert_almost_equal(delay2[1, 0, 0], -d1, decimal=14)
        npt.assert_almost_equal(delay2[0, 1, 0],  d1, decimal=14)
        npt.assert_almost_equal(delay2[1, 1, 0],  d1, decimal=14)
        
        npt.assert_almost_equal(cr2[0, 0, 0], -1.0, decimal=5)
        npt.assert_almost_equal(cr2[0, 1, 0], -2.0, decimal=5)

    # -------------------------------------------------------------------------
    # Fake LOS path
    # -------------------------------------------------------------------------

    def test_fake_los_path(self):
        ant = omni()

        # LOS path first, NLOS second (reversed vs minimal test)
        fbs_pos = np.array([[0, 10], [10, 0], [11, 1]], order='F')
        path_gain = np.array([0.25, 1.0])
        path_length = np.zeros(2)
        M = make_M(2)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True, True,
            angles=True)

        self.assertEqual(cr.shape[2], 3)
        self.assertEqual(delay.shape[2], 3)
        self.assertEqual(aod.shape[2], 3)

        aod_d = np.degrees(aod)
        eod_d = np.degrees(eod)
        aoa_d = np.degrees(aoa)
        eoa_d = np.degrees(eoa)

        # Slice 0: LOS path (prepended fake → real LOS is now first real path)
        npt.assert_almost_equal(aod_d[0, 0, 0], 0.0, decimal=4)
        npt.assert_almost_equal(aod_d[0, 0, 1], 90.0, decimal=4)
        npt.assert_almost_equal(aod_d[0, 0, 2], 0.0, decimal=4)

        alpha = 180.0 - np.degrees(np.arctan(10.0 / 20.0))
        self.assertAlmostEqual(abs(aoa_d[0, 0, 0]), 180.0, places=2)
        npt.assert_almost_equal(aoa_d[0, 0, 1], alpha, decimal=2)
        self.assertAlmostEqual(abs(aoa_d[0, 0, 2]), 180.0, places=2)

        npt.assert_almost_equal(eod_d[0, 0, 0], 0.0, decimal=2)
        npt.assert_almost_equal(eod_d[0, 0, 1], 45.0, decimal=2)
        npt.assert_almost_equal(eod_d[0, 0, 2], 0.0, decimal=2)

        alpha_eoa = np.degrees(np.arctan(10.0 / math.sqrt(10.0 ** 2 + 20.0 ** 2)))
        npt.assert_almost_equal(eoa_d[0, 0, 0], 0.0, decimal=2)
        npt.assert_almost_equal(eoa_d[0, 0, 1], alpha_eoa, decimal=2)
        npt.assert_almost_equal(eoa_d[0, 0, 2], 0.0, decimal=2)

        amp = amp2(cr, ci)
        npt.assert_almost_equal(amp[0, 0, 0], 1.0, decimal=5)
        npt.assert_almost_equal(amp[0, 0, 1], 0.5, decimal=5)
        npt.assert_almost_equal(amp[0, 0, 2], 0.0, decimal=5)

        d0 = 20.0
        e0 = math.hypot(10.0, 10.0) + math.sqrt(10.0 ** 2 + 20.0 ** 2 + 10.0 ** 2)
        npt.assert_almost_equal(delay[0, 0, 0], d0 / C, decimal=11)
        npt.assert_almost_equal(delay[0, 0, 1], e0 / C, decimal=11)
        npt.assert_almost_equal(delay[0, 0, 2], d0 / C, decimal=11)

    def test_fake_los_single_path_zero_power(self):
        ant = omni()
        fbs_pos = np.array([[50.0], [30.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, True)

        self.assertEqual(cr.shape[2], 2)
        # Fake LOS slice 0 must be zero
        self.assertAlmostEqual(abs(cr[0, 0, 0]), 0.0, places=12)
        self.assertAlmostEqual(abs(ci[0, 0, 0]), 0.0, places=12)
        # Real path slice 1 must be non-zero
        self.assertGreater(amp2(cr, ci)[0, 0, 1], 0.0)

    # -------------------------------------------------------------------------
    # Coupling
    # -------------------------------------------------------------------------

    def test_coupling_identity(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant = arrayant.copy_element(ant, 0, 2)
        ant['element_pos'] = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], order='F')
        ant['coupling_re'] = np.eye(3, order='F')
        ant['coupling_im'] = np.zeros((3, 3), order='F')

        fbs_pos = np.array([[10.0], [0.0], [1.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 0.0, False, True)

        self.assertEqual(cr.shape[2], 2)
        npt.assert_almost_equal(ci[:, :, 0], np.zeros((3, 3)), decimal=5)
        npt.assert_almost_equal(ci[:, :, 1], np.zeros((3, 3)), decimal=5)
        npt.assert_almost_equal(cr[:, :, 1], np.zeros((3, 3)), decimal=5)

        # Only elements 0 and 2 (same gain, symmetric) should have non-zero re
        expected = np.zeros((3, 3))
        expected[0, 0] = 1.0; expected[0, 2] = 1.0
        expected[2, 0] = 1.0; expected[2, 2] = 1.0
        npt.assert_almost_equal(cr[:, :, 0], np.ones((3, 3)), decimal=5)

    def test_coupling_beamforming(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant = arrayant.copy_element(ant, 0, 2)
        ant['element_pos'] = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], order='F')
        # All elements summed into one port
        ant['coupling_re'] = np.ones((3, 1), order='F') * 2.0
        ant['coupling_im'] = np.zeros((3, 1), order='F')

        fbs_pos = np.array([[10.0], [0.0], [1.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 0.0, True, False,
            angles=True)[:6]

        self.assertEqual(cr.shape, (1, 1, 1))
        npt.assert_almost_equal(cr[0, 0, 0], 36.0, decimal=4)
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=4)
        npt.assert_almost_equal(aod[0, 0, 0], 0.0, decimal=12)
        npt.assert_almost_equal(eod[0, 0, 0], 0.0, decimal=12)
        npt.assert_almost_equal(math.cos(aoa[0, 0, 0]), -1.0, decimal=12)

    # -------------------------------------------------------------------------
    # SISO LOS scenarios
    # -------------------------------------------------------------------------

    def test_siso_los_along_x(self):
        tx = omni(); rx = omni()
        dist = 100.0
        fbs_pos = np.array([[dist / 2], [0], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            tx, rx, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertEqual(cr.shape, (1, 1, 1))
        self.assertAlmostEqual(ci[0, 0, 0], 0.0, places=12)
        self.assertAlmostEqual(amp2(cr, ci)[0, 0, 0], 1.0, places=5)
        self.assertAlmostEqual(delay[0, 0, 0], dist / C, places=12)
        self.assertAlmostEqual(aod[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(eod[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(math.cos(aoa[0, 0, 0]) + 1.0, 0.0, places=5)
        self.assertAlmostEqual(eoa[0, 0, 0], 0.0, places=5)

    def test_siso_los_along_y(self):
        ant = omni()
        dist = 80.0
        fbs_pos = np.array([[0], [dist / 2], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [0, dist, 0], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertAlmostEqual(aod[0, 0, 0], PI / 2, places=5)
        self.assertAlmostEqual(aoa[0, 0, 0] + PI / 2, 0.0, places=5)
        self.assertAlmostEqual(eod[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(eoa[0, 0, 0], 0.0, places=5)

    def test_siso_los_vertical(self):
        ant = omni()
        height = 50.0
        fbs_pos = np.array([[0], [0], [height / 2]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([height])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [0, 0, height], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertAlmostEqual(eod[0, 0, 0], PI / 2, places=5)
        self.assertAlmostEqual(eoa[0, 0, 0] + PI / 2, 0.0, places=5)
        self.assertAlmostEqual(delay[0, 0, 0], height / C, places=12)

    def test_siso_los_diagonal(self):
        ant = omni()
        d = 100.0
        fbs_pos = np.array([[d / 2], [d / 2], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([math.sqrt(2.0) * d])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [d, d, 0], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertAlmostEqual(aod[0, 0, 0], PI / 4, places=5)
        self.assertAlmostEqual(eod[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(aoa[0, 0, 0], math.atan2(-d, -d), places=5)
        dist = math.sqrt(d ** 2 + d ** 2)
        self.assertAlmostEqual(delay[0, 0, 0], dist / C, places=12)

    def test_siso_los_negative_x(self):
        ant = omni()
        dist = 50.0
        fbs_pos = np.array([[-dist / 2], [0], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [-dist, 0, 0], [0, 0, 0], 0.0, True, False, angles=True)

        # AOD toward -x: cos(aod) == -1
        self.assertAlmostEqual(math.cos(aod[0, 0, 0]) + 1.0, 0.0, places=5)
        # AOA from +x: 0
        self.assertAlmostEqual(aoa[0, 0, 0], 0.0, places=5)

    def test_offset_tx_rx_positions(self):
        ant = omni()
        Tx, Ty, Tz = 10.0, 20.0, 5.0
        Rx, Ry, Rz = 110.0, 20.0, 5.0
        dist = 100.0
        fbs_pos = np.array([[(Tx + Rx) / 2], [Ty], [Tz]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [Tx, Ty, Tz], [0, 0, 0], [Rx, Ry, Rz], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertAlmostEqual(delay[0, 0, 0], dist / C, places=12)
        self.assertAlmostEqual(aod[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(eod[0, 0, 0], 0.0, places=5)

    # -------------------------------------------------------------------------
    # FBS != LBS (two-bounce)
    # -------------------------------------------------------------------------

    def test_fbs_lbs_different(self):
        ant = omni()
        fbs_pos = np.array([[20.0], [30.0], [0.0]], order='F')
        lbs_pos = np.array([[80.0], [30.0], [0.0]], order='F')
        d_tx_fbs = math.sqrt(20.0 ** 2 + 30.0 ** 2)
        d_fbs_lbs = 60.0
        d_lbs_rx = math.sqrt(20.0 ** 2 + 30.0 ** 2)
        d_total = d_tx_fbs + d_fbs_lbs + d_lbs_rx

        path_gain = np.array([1.0])
        path_length = np.array([d_total + 10.0])
        M = make_M(1)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, lbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False, angles=True)

        self.assertAlmostEqual(delay[0, 0, 0], d_total / C, places=12)
        self.assertAlmostEqual(aod[0, 0, 0], math.atan2(30.0, 20.0), places=5)
        self.assertAlmostEqual(aoa[0, 0, 0], math.atan2(30.0, -20.0), places=5)

    # -------------------------------------------------------------------------
    # Zero center frequency disables phase
    # -------------------------------------------------------------------------

    def test_zero_center_freq_no_phase(self):
        ant = omni()
        fbs_pos = np.array([[50.0], [0.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([100.0])
        M = make_M(1)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        self.assertAlmostEqual(ci[0, 0, 0], 0.0, places=5)
        self.assertGreater(cr[0, 0, 0], 0.0)
        self.assertAlmostEqual(cr[0, 0, 0], 1.0, places=5)

    # -------------------------------------------------------------------------
    # Absolute vs relative delays
    # -------------------------------------------------------------------------

    def test_absolute_vs_relative_delays(self):
        ant = omni()
        dist = 50.0
        fbs_pos = np.array([[dist / 2], [0], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)
        kwargs = dict(center_freq=0.0)

        cr_abs, ci_abs, d_abs = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], use_absolute_delays=True, **kwargs)

        cr_rel, ci_rel, d_rel = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], use_absolute_delays=False, **kwargs)

        self.assertAlmostEqual(d_abs[0, 0, 0], dist / C, places=12)
        self.assertAlmostEqual(d_rel[0, 0, 0], 0.0, places=12)
        self.assertAlmostEqual(cr_abs[0, 0, 0], cr_rel[0, 0, 0], places=12)
        self.assertAlmostEqual(ci_abs[0, 0, 0], ci_rel[0, 0, 0], places=12)

    # -------------------------------------------------------------------------
    # Phase with center frequency
    # -------------------------------------------------------------------------

    def test_phase_with_center_frequency(self):
        ant = omni()
        dist = 100.0
        fc = 1.0e9
        wavelength = C / fc
        fbs_pos = np.array([[dist / 2], [0], [0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([dist])
        M = make_M(1)

        cr_f, ci_f, d_f = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], fc, True, False)

        cr_0, ci_0, d_0 = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], 0.0, True, False)

        # Amplitudes match
        self.assertAlmostEqual(amp2(cr_f, ci_f)[0, 0, 0], amp2(cr_0, ci_0)[0, 0, 0], places=8)
        # Delays match
        self.assertAlmostEqual(d_f[0, 0, 0], d_0[0, 0, 0], places=12)
        # Phase matches wavenumber * fmod(dist, wavelength)
        wk = 2.0 * PI / wavelength
        expected = math.fmod(wk * math.fmod(dist, wavelength) + 2 * PI, 2 * PI)
        actual = math.fmod(math.atan2(-ci_f[0, 0, 0], cr_f[0, 0, 0]) + 2 * PI, 2 * PI)
        diff = math.fmod(abs(actual - expected) + 2 * PI, 2 * PI)
        if diff > PI:
            diff = 2 * PI - diff
        self.assertLess(diff, 1e-4)

    # -------------------------------------------------------------------------
    # Complex M matrix introduces phase
    # -------------------------------------------------------------------------

    def test_complex_M_introduces_phase(self):
        ant = omni()
        fbs_pos = np.array([[50.0], [0.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.array([100.0])
        s2 = math.sqrt(2.0) / 2.0
        M = np.zeros((8, 1), order='F')
        M[0, 0] = s2; M[1, 0] = s2    # VV = exp(j*pi/4)
        M[6, 0] = -s2; M[7, 0] = -s2  # HH = -exp(j*pi/4)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        # Amplitude = 1 (|M_VV| = 1)
        self.assertAlmostEqual(amp2(cr, ci)[0, 0, 0], 1.0, places=5)
        # Imaginary part must be non-zero (phase from complex M)
        self.assertGreater(abs(ci[0, 0, 0]), 1e-3)

    # -------------------------------------------------------------------------
    # Cross-polarization via M
    # -------------------------------------------------------------------------

    def test_xpol_M_matrix(self):
        probe = xpol()
        fbs_pos = np.array([[50.0], [50.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)

        M_vh = np.zeros((8, 1), order='F'); M_vh[2, 0] = 1.0  # Re(VH)
        M_vv = np.zeros((8, 1), order='F'); M_vv[0, 0] = 1.0  # Re(VV)
        M_z  = np.zeros((8, 1), order='F')

        cr_vh, ci_vh, _ = arrayant.get_channels_spherical(
            probe, probe, fbs_pos, fbs_pos, path_gain, path_length, M_vh,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        cr_vv, ci_vv, _ = arrayant.get_channels_spherical(
            probe, probe, fbs_pos, fbs_pos, path_gain, path_length, M_vv,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        cr_z, ci_z, _ = arrayant.get_channels_spherical(
            probe, probe, fbs_pos, fbs_pos, path_gain, path_length, M_z,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        self.assertGreater(np.sum(amp2(cr_vh, ci_vh) ** 2), 1e-6)
        self.assertGreater(np.sum(amp2(cr_vv, ci_vv) ** 2), 1e-6)
        self.assertLess(np.sum(amp2(cr_z, ci_z) ** 2), 1e-12)
        # VH and VV must produce different channel matrices
        self.assertFalse(np.allclose(cr_vh, cr_vv, atol=1e-6))

    # -------------------------------------------------------------------------
    # RX rotation swaps xpol amplitudes
    # -------------------------------------------------------------------------

    def test_rx_rotation_swaps_xpol(self):
        ant = omni(); probe = xpol()
        fbs_pos = np.array([[50.0], [0.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        cr0, ci0, d0 = arrayant.get_channels_spherical(
            ant, probe, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, False, False)

        cr_r, ci_r, d_r = arrayant.get_channels_spherical(
            ant, probe, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [PI / 2, 0, 0], 0.0, False, False)

        # Delays unchanged (point antenna, no element offset)
        self.assertAlmostEqual(d0[0, 0, 0], d_r[0, 0, 0], places=12)
        self.assertAlmostEqual(d0[1, 0, 0], d_r[1, 0, 0], places=12)

        a_v0 = amp2(cr0, ci0)[0, 0, 0]
        a_h0 = amp2(cr0, ci0)[1, 0, 0]
        a_vr = amp2(cr_r, ci_r)[0, 0, 0]
        a_hr = amp2(cr_r, ci_r)[1, 0, 0]

        self.assertAlmostEqual(a_v0, a_hr, places=4)
        self.assertAlmostEqual(a_h0, a_vr, places=4)

    # -------------------------------------------------------------------------
    # Output shapes
    # -------------------------------------------------------------------------

    def test_output_shape_mimo(self):
        tx = omni()
        tx = arrayant.copy_element(tx, 0, 1)
        tx = arrayant.copy_element(tx, 0, 2)
        tx['element_pos'] = np.array([[0, 0, 0], [0.5, 0.0, -0.5], [0, 0, 0]], order='F')
        tx['coupling_re'] = np.eye(3, order='F')
        tx['coupling_im'] = np.zeros((3, 3), order='F')

        rx = omni()
        rx = arrayant.copy_element(rx, 0, 1)
        rx['element_pos'] = np.array([[0, 0], [0.5, -0.5], [0, 0]], order='F')
        rx['coupling_re'] = np.eye(2, order='F')
        rx['coupling_im'] = np.zeros((2, 2), order='F')

        n_path = 4
        fbs_pos = np.zeros((3, n_path), order='F')
        fbs_pos[0, :] = [50.0, 40.0, 60.0, 30.0]
        fbs_pos[1, 1] = 20.0; fbs_pos[2, 2] = 15.0; fbs_pos[1, 3] = -10.0
        path_gain = np.ones(n_path)
        path_length = np.zeros(n_path)
        M = make_M(n_path)

        cr, ci, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical(
            tx, rx, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False, angles=True)

        for arr in [cr, ci, delay, aod, eod, aoa, eoa]:
            self.assertEqual(arr.shape, (2, 3, n_path))

        # With fake LOS: n_path + 1 slices
        cr2, ci2, delay2, aod2, _, _, _ = arrayant.get_channels_spherical(
            tx, rx, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, True, angles=True)

        self.assertEqual(cr2.shape, (2, 3, n_path + 1))
        self.assertEqual(aod2.shape, (2, 3, n_path + 1))

    def test_asymmetric_tx_rx(self):
        tx = omni()
        tx = arrayant.copy_element(tx, 0, 1)
        tx['element_pos'] = np.array([[0, 0], [0.5, -0.5], [0, 0]], order='F')
        tx['coupling_re'] = np.eye(2, order='F')
        tx['coupling_im'] = np.zeros((2, 2), order='F')
        rx = omni()

        fbs_pos = np.array([[50.0], [0.0], [0.0]], order='F')
        path_gain = np.array([1.0])
        path_length = np.zeros(1)
        M = make_M(1)

        cr, ci, delay = arrayant.get_channels_spherical(
            tx, rx, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        self.assertEqual(cr.shape, (1, 2, 1))
        self.assertEqual(delay.shape, (1, 2, 1))

    # -------------------------------------------------------------------------
    # Multiple NLOS paths, relative delays
    # -------------------------------------------------------------------------

    def test_multiple_paths_relative_delays(self):
        ant = omni()
        dist = 100.0
        fbs_pos = np.zeros((3, 3), order='F')
        fbs_pos[0, 0] = dist / 2
        fbs_pos[0, 1] = dist / 2; fbs_pos[2, 1] = 30.0
        fbs_pos[0, 2] = dist / 2; fbs_pos[1, 2] = 40.0

        path_gain = np.array([1.0, 0.5, 0.25])
        path_length = np.array([dist, 0.0, 0.0])
        M = make_M(3)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], 0.0, False, False)

        self.assertEqual(cr.shape[2], 3)
        self.assertAlmostEqual(delay[0, 0, 0], 0.0, places=12)
        self.assertGreater(delay[0, 0, 1], 0.0)
        self.assertGreater(delay[0, 0, 2], 0.0)

        d1 = 2 * math.sqrt(50.0 ** 2 + 30.0 ** 2)
        d2 = 2 * math.sqrt(50.0 ** 2 + 40.0 ** 2)
        self.assertAlmostEqual(delay[0, 0, 1], (d1 - dist) / C, places=12)
        self.assertAlmostEqual(delay[0, 0, 2], (d2 - dist) / C, places=12)

        amp = amp2(cr, ci)
        self.assertAlmostEqual(amp[0, 0, 0], 1.0, places=5)
        self.assertAlmostEqual(amp[0, 0, 1], math.sqrt(0.5), places=5)
        self.assertAlmostEqual(amp[0, 0, 2], 0.5, places=5)

    # -------------------------------------------------------------------------
    # Zero path gain
    # -------------------------------------------------------------------------

    def test_zero_path_gain(self):
        ant = omni()
        fbs_pos = np.zeros((3, 2), order='F')
        fbs_pos[0, 0] = 50.0; fbs_pos[0, 1] = 50.0; fbs_pos[1, 1] = 20.0
        path_gain = np.array([1.0, 0.0])
        path_length = np.zeros(2)
        M = make_M(2)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        self.assertGreater(amp2(cr, ci)[0, 0, 0], 0.0)
        self.assertAlmostEqual(cr[0, 0, 1], 0.0, places=12)
        self.assertAlmostEqual(ci[0, 0, 1], 0.0, places=12)

    # -------------------------------------------------------------------------
    # Large number of paths (stress test)
    # -------------------------------------------------------------------------

    def test_large_number_of_paths(self):
        ant = omni()
        n_path = 100
        fbs_pos = np.zeros((3, n_path), order='F')
        for p in range(n_path):
            angle = 2.0 * PI * p / n_path
            fbs_pos[0, p] = 50.0 + 30.0 * math.cos(angle)
            fbs_pos[1, p] = 30.0 * math.sin(angle)

        path_gain = np.array([1.0 / (p + 1) for p in range(n_path)])
        path_length = np.zeros(n_path)
        M = make_M(n_path)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 1.0e9, True, False)

        self.assertEqual(cr.shape[2], n_path)
        self.assertTrue(np.all(delay > 0.0))

    # -------------------------------------------------------------------------
    # Physical consistency: absolute delays all positive, LOS shortest
    # -------------------------------------------------------------------------

    def test_delay_physical_consistency(self):
        ant = omni()
        fbs_pos = np.zeros((3, 3), order='F')
        fbs_pos[0, 0] = 50.0
        fbs_pos[0, 1] = 50.0; fbs_pos[1, 1] = 30.0
        fbs_pos[0, 2] = 50.0; fbs_pos[2, 2] = 40.0

        path_gain = np.ones(3)
        path_length = np.zeros(3)
        M = make_M(3)

        cr, ci, delay = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], 0.0, True, False)

        self.assertTrue(np.all(delay > 0.0))
        self.assertLessEqual(delay[0, 0, 0], delay[0, 0, 1])
        self.assertLessEqual(delay[0, 0, 0], delay[0, 0, 2])

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    def test_error_mismatched_n_path(self):
        ant = omni()
        fbs_pos = np.zeros((3, 2), order='F'); fbs_pos[0, :] = [10, 5]
        path_gain = np.ones(3)  # wrong: 3 vs 2 paths
        path_length = np.zeros(2)
        M = make_M(2)
        with self.assertRaises(ValueError):
            arrayant.get_channels_spherical(
                ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
                [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0])

    def test_error_lbs_col_mismatch(self):
        ant = omni()
        fbs_pos = np.zeros((3, 2), order='F'); fbs_pos[0, :] = [10, 5]
        lbs_pos = np.zeros((3, 1), order='F'); lbs_pos[0, 0] = 10
        path_gain = np.ones(2)
        path_length = np.zeros(2)
        M = make_M(2)
        with self.assertRaises(ValueError):
            arrayant.get_channels_spherical(
                ant, ant, fbs_pos, lbs_pos, path_gain, path_length, M,
                [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0])

    def test_error_M_wrong_rows(self):
        ant = omni()
        fbs_pos = np.zeros((3, 1), order='F'); fbs_pos[0, 0] = 10
        path_gain = np.ones(1)
        path_length = np.zeros(1)
        M_bad = np.zeros((4, 1), order='F')  # neither 8 nor 2 rows
        with self.assertRaises(ValueError):
            arrayant.get_channels_spherical(
                ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M_bad,
                [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0])

    def test_error_tx_pos_wrong_size(self):
        ant = omni()
        fbs_pos = np.zeros((3, 2), order='F'); fbs_pos[0, :] = [10, 5]
        path_gain = np.ones(2)
        path_length = np.zeros(2)
        M = make_M(2)
        with self.assertRaises(ValueError):
            arrayant.get_channels_spherical(
                ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
                [0, 0, 1, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0])

    def test_error_coupling_im_without_re(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant['coupling_im'] = np.zeros((2, 2), order='F')
        ant.pop('coupling_re', None)

        fbs_pos = np.zeros((3, 2), order='F'); fbs_pos[0, :] = [10, 0]; fbs_pos[1, 1] = 10
        path_gain = np.array([1.0, 0.25])
        path_length = np.zeros(2)
        M = make_M(2)

        with self.assertRaises(ValueError) as ctx:
            arrayant.get_channels_spherical(
                ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
                [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertIn("Imaginary part of coupling matrix", str(ctx.exception))

    def test_no_coupling_keys_accepted(self):
        ant = omni()
        ant = arrayant.copy_element(ant, 0, 1)
        ant.pop('coupling_re', None)
        ant.pop('coupling_im', None)
        ant.pop('element_pos', None)

        fbs_pos = np.zeros((3, 2), order='F'); fbs_pos[0, :] = [10, 0]; fbs_pos[1, 1] = 10
        path_gain = np.array([1.0, 0.25])
        path_length = np.zeros(2)
        M = make_M(2)

        result = arrayant.get_channels_spherical(
            ant, ant, fbs_pos, fbs_pos, path_gain, path_length, M,
            [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], 2997924580.0, True)
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()