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

from quadriga_lib import arrayant, channel


def re_im(out, at=0, n=1):
    """Output tuple with the complex output (array or list of arrays) from index 'at' replaced by real and imaginary parts"""
    def split(c):
        return ([x.real for x in c], [x.imag for x in c]) if isinstance(c, list) else (c.real, c.imag)
    return tuple(out[:at]) + sum((split(c) for c in out[at:at + n]), ()) + tuple(out[at + n:])

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

        coeff_re, coeff_im, delay, rx_Doppler = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            2997924580.0, use_absolute_delays=True))

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

        coeff_re, coeff_im, delay, rx_Doppler = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            2997924580.0, use_absolute_delays=True, add_fake_los_path=True))

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
        cr, ci, delay, _ = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=False, add_fake_los_path=True))

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
        cr, ci, delay, _ = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=True, add_fake_los_path=False))

        self.assertEqual(cr.shape, (1, 1, 1))
        npt.assert_almost_equal(cr[0, 0, 0], 4.0, decimal=13)   # (1 + 0 + 1)^2
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=13)
        npt.assert_almost_equal(delay[0, 0, 0], 20.0 / C, decimal=13)

        # Case 3: single port, coupling weight 2
        ant = base_ant()
        ant['coupling_re'] = np.ones((3, 1), order='F') * 2.0
        ant.pop('coupling_im', None)
        cr, ci, delay, _ = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=True, add_fake_los_path=False))

        self.assertEqual(cr.size, 1)
        npt.assert_almost_equal(cr[0, 0, 0], 16.0, decimal=13)  # (2*(1 + 0 + 1))^2
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=13)
        npt.assert_almost_equal(delay[0, 0, 0], 20.0 / C, decimal=13)

    # -------------------------------------------------------------------------
    # rx_Doppler equals cos(aoa)*cos(eoa)
    # -------------------------------------------------------------------------

    def test_doppler_directions(self):
        ant = omni()
        path_gain = np.array([1.0])
        path_length = np.array([100.0])
        M = make_M(1)
        for aoa_v, eoa_v in [(0.0, 0.0), (PI / 2, 0.0), (PI, 0.0), (0.0, PI / 2)]:
            _, _, _, dop = re_im(channel.planar(
                ant, ant, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], np.zeros(1), np.zeros(1), np.array([aoa_v]),
                np.array([eoa_v]), path_gain, path_length, M, 0.0, use_absolute_delays=True))
            npt.assert_almost_equal(dop[0], math.cos(aoa_v) * math.cos(eoa_v), decimal=12)

    # -------------------------------------------------------------------------
    # center_freq = 0 disables phase
    # -------------------------------------------------------------------------

    def test_center_freq_zero_no_phase(self):
        ant = omni()
        cr, ci, _, _ = re_im(channel.planar(
            ant, ant, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], np.zeros(1), np.zeros(1), np.array([PI]),
            np.zeros(1), np.array([1.0]), np.array([100.0]), make_M(1), 0.0, use_absolute_delays=True))
        npt.assert_almost_equal(ci[0, 0, 0], 0.0, decimal=6)
        npt.assert_almost_equal(amp2(cr, ci)[0, 0, 0], 1.0, decimal=5)

    # -------------------------------------------------------------------------
    # Absolute vs relative delays
    # -------------------------------------------------------------------------

    def test_absolute_vs_relative_delays(self):
        ant = omni()
        dist = 50.0
        common = (ant, ant, [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], np.zeros(1), np.zeros(1), np.array([PI]),
                  np.zeros(1), np.array([1.0]), np.array([dist]), make_M(1))
        cr_a, ci_a, d_a, _ = re_im(channel.planar(*common, center_freq=0.0, use_absolute_delays=True))
        cr_r, ci_r, d_r, _ = re_im(channel.planar(*common, center_freq=0.0, use_absolute_delays=False))

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
        common = (ant, ant, [0, 0, 0], [0, 0, 0], [dist, 0, 0], [0, 0, 0], np.zeros(1), np.zeros(1), np.array([PI]),
                  np.zeros(1), np.array([1.0]), np.array([dist]), make_M(1))
        cr_f, ci_f, d_f, _ = re_im(channel.planar(*common, center_freq=fc, use_absolute_delays=True))
        cr_0, ci_0, d_0, _ = re_im(channel.planar(*common, center_freq=0.0, use_absolute_delays=True))

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

        cr, ci, _, _ = re_im(channel.planar(
            ant, ant, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], np.zeros(1), np.zeros(1), np.array([PI]),
            np.zeros(1), np.array([1.0]), np.array([100.0]), M, 0.0, use_absolute_delays=True))

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
            return re_im(channel.planar(
                probe, probe, *pos, z, z, np.array([PI]), z,
                np.array([1.0]), np.array([100.0]), M, 0.0, use_absolute_delays=True))

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
        paths = (z, z, np.array([PI]), z, np.array([1.0]), np.array([100.0]), make_M(1))

        cr0, ci0, _, _ = re_im(channel.planar(
            tx, rx, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], *paths, 0.0, use_absolute_delays=False))
        cr_r, ci_r, _, _ = re_im(channel.planar(
            tx, rx, [0, 0, 0], [0, 0, 0], [100, 0, 0], [PI / 2, 0, 0], *paths, 0.0, use_absolute_delays=False))

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

        cr, ci, delay, dop = re_im(channel.planar(
            tx, rx, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=True, add_fake_los_path=False))
        for arr in (cr, ci, delay):
            self.assertEqual(arr.shape, (2, 3, n_path))
        self.assertEqual(dop.shape, (n_path,))

        cr2, ci2, delay2, dop2 = re_im(channel.planar(
            tx, rx, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=True, add_fake_los_path=True))
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

        cr, ci, _, _ = re_im(channel.planar(
            ant, ant, [0, 0, 0], [0, 0, 0], [100, 0, 0], [0, 0, 0], aod, eod, aoa, eoa, path_gain, path_length, M,
            0.0, use_absolute_delays=True))

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
        result = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, pg, pl, M, 2997924580.0,
            use_absolute_delays=True))
        self.assertEqual(len(result), 4)

    def test_error_coupling_im_without_re(self):
        ant = minimal_ant()
        ant.pop('element_pos', None)
        ant.pop('coupling_re', None)   # leaves coupling_im only
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            channel.planar(
                ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, pg, pl, M, 2997924580.0,
                use_absolute_delays=True)
        self.assertEqual(str(ctx.exception),
                         "Transmit antenna: Imaginary part of coupling matrix (phase component) "
                         "defined without real part (absolute component)")

    def test_no_coupling_keys_accepted(self):
        ant = minimal_ant()
        ant.pop('element_pos', None)
        ant.pop('coupling_re', None)
        ant.pop('coupling_im', None)
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        result = re_im(channel.planar(
            ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, pg, pl, M, 2997924580.0,
            use_absolute_delays=True))
        self.assertEqual(len(result), 4)

    def test_error_mismatched_n_path(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            channel.planar(
                ant, ant, [0, 0, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod[[0, 1, 1]], eod, aoa, eoa, pg, pl, M,
                2997924580.0, use_absolute_delays=True)
        self.assertEqual(str(ctx.exception),
                         "Snapshot 0: Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'polmat' "
                         "must have the same number of columns (n_paths).")

    def test_error_tx_pos_wrong_size(self):
        ant = minimal_ant()
        aod, eod, aoa, eoa, pg, pl, M = minimal_paths()
        with self.assertRaises(ValueError) as ctx:
            channel.planar(
                ant, ant, [0, 0, 1, 1], [0, 0, 0], [20, 0, 1], [0, 0, 0], aod, eod, aoa, eoa, pg, pl, M,
                2997924580.0, use_absolute_delays=True)
        self.assertEqual(str(ctx.exception), "Input 'tx_pos' has incorrect number of elements.")


    # -------------------------------------------------------------------------
    # Input checks
    # -------------------------------------------------------------------------

    def _planar_args(self):
        return (np.zeros(3), np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3),
                np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1) * 10,
                np.array([[1.0, 0, 0, 0, 0, 0, 1, 0]]).T)

    def test_none_and_flag_types(self):
        o = arrayant.generate('omni')
        args = (o, o) + self._planar_args()
        ref = channel.planar(*args, 1e9)
        for name in ("use_absolute_delays", "add_fake_los_path"):
            out = channel.planar(*args, 1e9, **{name: None})
            for a, b in zip(out, ref):
                npt.assert_array_equal(a, b)
            with self.assertRaises(TypeError) as ctx:
                channel.planar(*args, 1e9, **{name: 2.5})
            self.assertIn("'" + name + "'", str(ctx.exception))

        # Mandatory arrays refuse None
        for i in range(2, len(args)):
            with self.assertRaises(TypeError):
                channel.planar(*args[:i], None, *args[i + 1:])

        # 4D (multi-frequency) arrayants are refused
        ant4 = {k: (np.stack([v, v], axis=3) if k.startswith('e_') else v) for k, v in o.items()}
        ant4['center_freq'] = np.array([1e9, 2e9])
        with self.assertRaises(ValueError) as ctx:
            channel.planar(ant4, *args[1:])
        self.assertIn("has 4D (multi-frequency) patterns", str(ctx.exception))

    def test_position_with_more_than_3_elements_raises(self):
        o = arrayant.generate('omni')
        args = list(self._planar_args())
        for i in (0, 1, 2, 3):  # tx_pos, tx_orientation, rx_pos, rx_orientation
            bad = list(args)
            bad[i] = np.array([[0.0, 50.0], [0.0, 0.0], [10.0, 10.0]])  # (3, 2), documented (3,)
            with self.assertRaises(ValueError):
                channel.planar(o, o, *bad)
        args[0] = np.array([[0.0], [0.0], [0.0]])  # (3, 1) has 3 elements and is accepted
        channel.planar(o, o, *args)

    def test_center_freq_array_single_frequency(self):
        import warnings
        o = arrayant.generate('omni')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            o['center_freq'] = 2e9
            ref = re_im(channel.planar(o, o, *self._planar_args()))
            for cf in (np.array([2e9]), np.array(2e9), np.float32(2e9), [2e9], 2000000000):
                o['center_freq'] = cf
                r = re_im(channel.planar(o, o, *self._planar_args()))
                for k in range(len(ref)):
                    npt.assert_array_equal(r[k], ref[k])
        for cf in (np.array([2e9, 3e9]), np.array([]), None):
            o['center_freq'] = cf
            with self.assertRaises(ValueError):
                channel.planar(o, o, *self._planar_args())


class test_zero_paths_fake_los(unittest.TestCase):

    # 0 paths with add_fake_los_path: one zero-power LOS path with the LOS delay (as in C++ and MATLAB)
    def test(self):
        ant = arrayant.generate('omni')
        e = np.zeros(0)
        cr, ci, dl, dop = re_im(channel.planar(ant, ant, [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0], e, e, e, e, e, e,
                                         np.zeros((8, 0)), 1e9, use_absolute_delays=True, add_fake_los_path=True))
        self.assertEqual(dl.shape, (1, 1, 1))
        npt.assert_array_equal(np.concatenate([cr.ravel(), ci.ravel(), np.ravel(dop)]), [0, 0, 0])
        npt.assert_allclose(dl.ravel(), [10 / 299792458.0], rtol=1e-15)
        cr, ci, dl, dop = re_im(channel.planar(ant, ant, [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0], e, e, e, e, e, e,
                                         np.zeros((8, 0)), 1e9, use_absolute_delays=False, add_fake_los_path=True))
        npt.assert_array_equal(dl.ravel(), [0])

    # 0 paths without add_fake_los_path: empty outputs (n_rx_ports, n_tx_ports, 0), as in C++ and MATLAB
    def test_without_fake_los(self):
        ant = arrayant.generate('xpol')
        e = np.zeros(0)
        cr, ci, dl, dop = re_im(channel.planar(ant, ant, [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0], e, e, e, e, e, e,
                                         np.zeros((8, 0)), 1e9, use_absolute_delays=True, add_fake_los_path=False))
        for x in (cr, ci, dl):
            self.assertEqual(x.shape, (2, 2, 0))
        self.assertEqual(np.shape(dop), (0,))
        c, dl, dop = channel.planar(ant, ant, [0, 0, 0], [0, 0, 0], [10, 0, 0], [0, 0, 0], e, e, e, e, e, e,
                                    np.zeros((8, 0)), 1e9, use_absolute_delays=True, add_fake_los_path=False)
        self.assertEqual(c.shape, (2, 2, 0))
        self.assertTrue(np.iscomplexobj(c))


class test_use_kernel(unittest.TestCase):
    # use_kernel (before i_freq) reaches C++: 0 gives the default result, 1 agrees within single precision, 3 raises from C++
    def test(self):
        ang = np.array([0.1, 0.5])
        args = (xpol(), xpol(), [0, 0, 10], [0, 0, 0], [50, 0, 1.5], [0, 0, 0], ang, ang, -ang, ang, np.ones(2),
                np.array([60.0, 70.0]), make_M(2), 3e9)
        ref = re_im(channel.planar(*args))
        for a, b in zip(re_im(channel.planar(*args, use_kernel=0)), ref):
            npt.assert_array_equal(a, b)  # 0 is the default
        for a, b in zip(re_im(channel.planar(*args, use_kernel=1)), ref):
            npt.assert_allclose(a, b, rtol=0, atol=1e-5 * np.abs(b).max())  # GENERIC: within single precision
        with self.assertRaises(ValueError) as ctx:
            channel.planar(*args, use_kernel=3)
        self.assertIn("'use_kernel' must be 0, 1 or 2", str(ctx.exception))



class test_batch(unittest.TestCase):
    # Lists of path arrays: one call for all snapshots, each equal to the single-snapshot call

    def setUp(self):
        rng = np.random.default_rng(3)
        self.ant = xpol()
        self.n_snap = 5
        self.paths = []
        for s in range(self.n_snap):
            n = 0 if s == 2 else 1 + s  # snapshot 2 has no paths
            M = rng.standard_normal((8, n, 2))
            self.paths.append((rng.uniform(-3, 3, n), rng.uniform(-1, 1, n), rng.uniform(-3, 3, n),
                               rng.uniform(-1, 1, n), rng.uniform(0.1, 1, (n, 2)), rng.uniform(30, 90, n), M))
        self.tx = rng.uniform(-5, 5, (3, self.n_snap))
        self.rx = rng.uniform(20, 30, (3, self.n_snap))
        self.ori = np.array([0.0, 0.1, 0.3])

    def lists(self):
        return [[p[k] for p in self.paths] for k in range(7)]

    def test_single_snapshot_list_inputs(self):
        # A single snapshot takes Python lists as arrays (converted), they do not select the batch mode
        p = self.paths[4]
        ref = channel.planar(self.ant, self.ant, self.tx[:, 0], self.ori, self.rx[:, 0], self.ori, *p, i_freq=1)
        out = channel.planar(self.ant, self.ant, self.tx[:, 0], self.ori, self.rx[:, 0], self.ori,
                             *[x.tolist() for x in p], i_freq=1)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_equals_single_calls(self):
        for fake_los in (False, True):
            out = channel.planar(self.ant, self.ant, self.tx, self.ori, self.rx, self.ori, *self.lists(),
                                 [1e9, 2e9], use_absolute_delays=True, add_fake_los_path=fake_los, i_freq=1)
            self.assertEqual(len(out), 3)
            for x in out:
                self.assertIsInstance(x, list)
                self.assertEqual(len(x), self.n_snap)
            for s in range(self.n_snap):
                ref = channel.planar(self.ant, self.ant, self.tx[:, s], self.ori, self.rx[:, s], self.ori,
                                     *self.paths[s], [1e9, 2e9], use_absolute_delays=True, add_fake_los_path=fake_los,
                                     i_freq=1)
                self.assertEqual(out[0][s].dtype, np.complex128)
                for a, b in zip(out, ref):
                    npt.assert_array_equal(a[s], b)
            self.assertEqual(out[0][2].shape, (2, 2, 1 if fake_los else 0))

    def test_empty_batch(self):
        # An empty list selects the batch mode with 0 snapshots: every output is an empty list
        out = channel.planar(self.ant, self.ant, self.tx[:, :1], self.ori, self.rx[:, :1], self.ori, *[[]] * 7)
        self.assertEqual(out, ([], [], []))

    def test_shared_position_and_entry_layouts(self):
        # A (3, 1) position applies to all snapshots; C-order and float32 list entries are converted
        ref = channel.planar(self.ant, self.ant, self.tx[:, :1], self.ori, self.rx, self.ori, *self.lists(), [1e9, 2e9])
        for conv in (np.ascontiguousarray, lambda x: x.astype(np.float32)):
            lists = [[conv(x) for x in lst] for lst in self.lists()]
            out = channel.planar(self.ant, self.ant, self.tx[:, :1], self.ori, self.rx, self.ori, *lists, [1e9, 2e9])
            for s in range(self.n_snap):
                single = channel.planar(self.ant, self.ant, self.tx[:, 0], self.ori, self.rx[:, s], self.ori,
                                        *[conv(x) for x in self.paths[s]], [1e9, 2e9])
                for a, b, r in zip(out, single, ref):
                    npt.assert_array_equal(a[s], b)
                    if conv is np.ascontiguousarray:
                        npt.assert_array_equal(a[s], r[s])

    def test_errors(self):
        lists = self.lists()
        args = (self.ant, self.ant, self.tx, self.ori, self.rx, self.ori)
        with self.assertRaises(ValueError) as ctx:
            channel.planar(*args, *lists[:6], self.paths[0][6])
        self.assertIn("must all be lists", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            channel.planar(*args, *lists[:5], lists[5][:-1], lists[6])
        self.assertIn("must have the same length", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            channel.planar(self.ant, self.ant, self.tx[:, :2], self.ori, self.rx, self.ori, *lists)
        self.assertIn("1 or 'n_snap' columns", str(ctx.exception))
        # The lowest failing snapshot is named, also when run in parallel
        n = 40
        one = (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.full(1, 40.0), make_M(1))
        lists = [[one[k]] * n for k in range(7)]
        lists[6][23] = lists[6][5] = np.ones((7, 1))
        with self.assertRaisesRegex(ValueError, "Snapshot 5: "):
            channel.planar(self.ant, self.ant, [0, 0, 1], self.ori, [20, 0, 1], self.ori, *lists)


if __name__ == '__main__':
    unittest.main()