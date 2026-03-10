# SPDX-License-Identifier: Apache-2.0
#
# quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
# Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
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
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def gen_omni():
    """Generate an omnidirectional antenna dictionary (single-frequency, 3D patterns)."""
    return quadriga_lib.arrayant.generate('omni')


def gen_xpol():
    """Generate a cross-polarized antenna dictionary."""
    return quadriga_lib.arrayant.generate('xpol')


def gen_custom(az_3dB=90.0, el_3dB=90.0, rear_gain=0.0):
    """Generate a custom antenna dictionary."""
    return quadriga_lib.arrayant.generate('custom', 1, az_3dB, el_3dB, rear_gain)


def copy_element(ant, src):
    """Copy element 'src' and append it as a new element. Returns modified dict."""
    ant = dict(ant)  # shallow copy
    for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im']:
        ant[key] = np.concatenate([ant[key], ant[key][:, :, src:src + 1]], axis=2)
    ant['element_pos'] = np.concatenate([ant['element_pos'], ant['element_pos'][:, src:src + 1]], axis=1)
    n_elem = ant['e_theta_re'].shape[2]
    ant['coupling_re'] = np.eye(n_elem, order='F')
    ant['coupling_im'] = np.zeros((n_elem, n_elem), order='F')
    return ant


def amp(cr, ci, i, j, k, f=None):
    """Compute amplitude from real and imaginary arrays at given index.
    If f is not None, uses 4D indexing [i,j,k,f]; otherwise 3D [i,j,k]."""
    if f is not None:
        return math.sqrt(cr[i, j, k, f] ** 2 + ci[i, j, k, f] ** 2)
    else:
        return math.sqrt(cr[i, j, k] ** 2 + ci[i, j, k] ** 2)


class TestGetChannelsMultifreq(unittest.TestCase):

    def test_documentation_example(self):
        """Run the exact example from the documentation."""
        # Build a 2-way speaker as TX (source)
        tx_woofer = quadriga_lib.arrayant.generate_speaker(
            driver_type='piston', radius=0.083,
            lower_cutoff=50.0, upper_cutoff=3000.0,
            lower_rolloff_slope=12.0, upper_rolloff_slope=24.0, sensitivity=87.0,
            radiation_type='hemisphere', baffle_width=0.20, baffle_height=0.30,
            frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]),
            angular_resolution=10.0)

        # Omnidirectional microphone as RX (single-frequency, clamped for all output frequencies)
        rx = quadriga_lib.arrayant.generate('omni')

        # Simple LOS path setup
        fbs_pos = np.array([[0.5], [0.0], [0.0]])
        lbs_pos = np.array([[0.5], [0.0], [0.0]])
        path_length = np.array([1.0])

        # Frequency-flat path gain and scalar Jones matrix at two input frequencies
        freq_in = np.array([100.0, 10000.0])
        path_gain = np.ones((1, 2))
        M = np.zeros((2, 1, 2))
        M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0

        # Compute channel at 3 output frequencies using speed of sound
        freq_out = np.array([200.0, 1000.0, 5000.0])
        coeff_re, coeff_im, delays = quadriga_lib.arrayant.get_channels_multifreq(
            tx_woofer, rx, fbs_pos, lbs_pos, path_gain, path_length, M,
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3),
            freq_in, freq_out, propagation_speed=343.0)

        # Shape: (n_rx_ports, n_tx_ports, n_path, n_freq_out)
        self.assertEqual(coeff_re.ndim, 4)
        self.assertEqual(coeff_re.shape[0], 1)   # 1 RX port (omni mic)
        self.assertEqual(coeff_re.shape[2], 1)   # 1 path
        self.assertEqual(coeff_re.shape[3], 3)   # 3 output frequencies
        self.assertTrue(np.all(np.isfinite(coeff_re)))
        self.assertTrue(np.all(np.isfinite(coeff_im)))
        self.assertTrue(np.all(np.isfinite(delays)))

        # Delays should all be zero (relative mode, single LOS path)
        npt.assert_allclose(delays, 0.0, atol=1e-10)

        # All frequencies should produce non-zero coefficients
        for f in range(3):
            a = amp(coeff_re, coeff_im, 0, 0, 0, f)
            self.assertGreater(a, 0.0)

    # ============================================================================
    # SECTION 1: Input validation
    # ============================================================================

    def test_empty_tx_array_throws(self):
        """Empty TX antenna pattern fields should raise an error."""
        rx = gen_omni()
        ant_tx = dict(rx)
        ant_tx['e_theta_re'] = np.zeros((0, 0, 0), order='F')
        ant_tx['e_theta_im'] = np.zeros((0, 0, 0), order='F')
        ant_tx['e_phi_re'] = np.zeros((0, 0, 0), order='F')
        ant_tx['e_phi_im'] = np.zeros((0, 0, 0), order='F')
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        pg = np.ones((1, 1), order='F'); pl = np.array([1.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                ant_tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo)

    def test_empty_freq_in_throws(self):
        """Empty freq_in should raise an error."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        pg = np.ones((1, 1), order='F'); pl = np.array([1.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([]); fo = np.array([1e9])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo)

    def test_empty_freq_out_throws(self):
        """Empty freq_out should raise an error."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        pg = np.ones((1, 1), order='F'); pl = np.array([1.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo)

    def test_mismatched_path_gain_throws(self):
        """path_gain with wrong number of rows should raise an error."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 2), order='F'); lbs = np.zeros((3, 2), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        fbs[0, 1] = 10.0; lbs[0, 1] = 10.0
        pg = np.ones((3, 1), order='F')  # Wrong: 3 rows but only 2 paths
        pl = np.array([10.0, 10.0])
        M = np.zeros((8, 2, 1), order='F'); M[0, 0, 0] = 1.0; M[0, 1, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo)

    def test_M_wrong_row_count_throws(self):
        """M with neither 8 nor 2 rows should raise an error."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((4, 1, 1), order='F')  # Wrong: neither 8 nor 2
        fi = np.array([1e9]); fo = np.array([1e9])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo)

    def test_non_positive_propagation_speed_throws(self):
        """Non-positive propagation speed should raise an error."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.get_channels_multifreq(
                tx, rx, fbs, lbs, pg, pl, M,
                np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
                fi, fo, propagation_speed=-1.0)

    # ============================================================================
    # SECTION 2: Output sizing
    # ============================================================================

    def test_output_dimensions(self):
        """Verify 4D output shape: (n_rx, n_tx, n_path, n_freq_out)."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 3), order='F'); lbs = np.zeros((3, 3), order='F')
        fbs[0, :] = 10.0; lbs[0, :] = 10.0
        pg = np.ones((3, 2), order='F'); pl = np.array([10.0, 10.0, 10.0])
        M = np.zeros((8, 3, 2), order='F')
        M[0, 0, 0] = 1.0; M[0, 1, 0] = 1.0; M[0, 2, 0] = 1.0
        M[0, 0, 1] = 1.0; M[0, 1, 1] = 1.0; M[0, 2, 1] = 1.0
        fi = np.array([1e9, 2e9]); fo = np.array([1e9, 1.5e9, 2e9, 3e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.ndim, 4)
        self.assertEqual(cr.shape, (1, 1, 3, 4))  # (n_rx, n_tx, n_path, n_freq_out)
        self.assertEqual(ci.shape, (1, 1, 3, 4))
        self.assertEqual(dl.shape, (1, 1, 3, 4))

    def test_fake_los_adds_extra_path(self):
        """add_fake_los_path should increase path count by 1."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 5.0; fbs[1, 0] = 5.0  # NLOS path
        lbs[0, 0] = 5.0; lbs[1, 0] = 5.0
        pg = np.ones((1, 1), order='F'); pl = np.array([15.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, add_fake_los_path=True)

        self.assertEqual(cr.shape[2], 2)  # 1 path + 1 fake LOS

    # ============================================================================
    # SECTION 3: Match single-frequency get_channels_spherical
    # ============================================================================

    def test_matches_single_freq_spherical(self):
        """Multi-freq with single entry should match get_channels_spherical."""
        C = 299792458.0
        fc = 2997924580.0  # 10 * C

        ant = gen_omni()
        ant = copy_element(ant, 0)
        ant['element_pos'][1, 0] = 1.0
        ant['element_pos'][1, 1] = -1.0
        ant['e_theta_re'][:, :, 1] *= 2.0

        fbs = np.zeros((3, 2), order='F'); lbs = np.zeros((3, 2), order='F')
        fbs[0, 0] = 10.0; fbs[2, 0] = 1.0
        fbs[0, 1] = 20.0; fbs[2, 1] = 1.0
        lbs[:] = fbs

        pg_vec = np.array([1.0, 0.25])
        pl_vec = np.array([0.0, 0.0])
        pg_mat = pg_vec.reshape(2, 1, order='F')

        M_single = np.zeros((8, 2), order='F')
        M_single[0, 0] = 1.0; M_single[6, 0] = -1.0
        M_single[0, 1] = 1.0; M_single[6, 1] = -1.0
        M_multi = M_single.reshape(8, 2, 1, order='F')

        fi = np.array([fc]); fo = np.array([fc])
        tx_pos = np.array([0.0, 0.0, 1.0])
        rx_pos = np.array([20.0, 0.0, 1.0])
        ori = np.zeros(3)

        # Single-frequency reference (returns 3D arrays)
        cr_ref, ci_ref, dl_ref = quadriga_lib.arrayant.get_channels_spherical(
            ant, ant, fbs, lbs, pg_vec, pl_vec, M_single,
            tx_pos, ori, rx_pos, ori, fc, True, False)

        # Multi-frequency version (returns 4D arrays)
        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            ant, ant, fbs, lbs, pg_mat, pl_vec, M_multi,
            tx_pos, ori, rx_pos, ori, fi, fo,
            use_absolute_delays=True)

        self.assertEqual(cr.shape[3], 1)  # single freq_out
        npt.assert_allclose(cr[:, :, :, 0], cr_ref, atol=1e-10, rtol=0)
        npt.assert_allclose(ci[:, :, :, 0], ci_ref, atol=1e-10, rtol=0)
        npt.assert_allclose(dl[:, :, :, 0], dl_ref, atol=1e-14, rtol=0)

    # ============================================================================
    # SECTION 4: Delay calculation with custom propagation speed
    # ============================================================================

    def test_acoustic_delays(self):
        """Speed of sound should produce correct acoustic delays."""
        c_sound = 343.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 5.0; lbs[0, 0] = 5.0
        pg = np.ones((1, 1), order='F'); pl = np.array([5.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1000.0]); fo = np.array([1000.0])

        # Absolute delays
        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([5.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True, propagation_speed=c_sound)

        npt.assert_allclose(dl[0, 0, 0, 0], 5.0 / c_sound, atol=1e-12, rtol=0)

        # Relative delays
        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([5.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=False, propagation_speed=c_sound)

        self.assertAlmostEqual(dl[0, 0, 0, 0], 0.0, places=14)

    def test_radio_vs_acoustic_delay_ratio(self):
        """Radio and acoustic delays should differ by the speed ratio."""
        c_radio = 299792458.0; c_sound = 343.0; dist = 10.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = dist; lbs[0, 0] = dist
        pg = np.ones((1, 1), order='F'); pl = np.array([dist])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        _, _, dl_r = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([dist, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True, propagation_speed=c_radio)

        _, _, dl_a = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([dist, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True, propagation_speed=c_sound)

        ratio = dl_a[0, 0, 0, 0] / dl_r[0, 0, 0, 0]
        self.assertAlmostEqual(ratio, c_radio / c_sound, delta=1e-6)

    # ============================================================================
    # SECTION 5: Scalar pressure Jones matrix (M with 2 rows)
    # ============================================================================

    def test_scalar_M_matches_full_M(self):
        """Scalar M (2 rows) should give same VV result as full M (8 rows)."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        fi = np.array([1e9]); fo = np.array([1e9])

        M_full = np.zeros((8, 1, 1), order='F')
        M_full[0, 0, 0] = 0.8; M_full[1, 0, 0] = 0.3

        M_scalar = np.zeros((2, 1, 1), order='F')
        M_scalar[0, 0, 0] = 0.8; M_scalar[1, 0, 0] = 0.3

        cr_f, ci_f, dl_f = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M_full,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        cr_s, ci_s, dl_s = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M_scalar,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        npt.assert_allclose(cr_f, cr_s, atol=1e-12, rtol=0)
        npt.assert_allclose(ci_f, ci_s, atol=1e-12, rtol=0)
        npt.assert_allclose(dl_f, dl_s, atol=1e-14, rtol=0)

    # ============================================================================
    # SECTION 6: Frequency interpolation of path_gain and M
    # ============================================================================

    def test_path_gain_linear_interpolation(self):
        """Path gain should be linearly interpolated across frequency."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pl = np.array([10.0])
        pg = np.array([[1.0, 4.0]], order='F')
        M = np.zeros((2, 1, 2), order='F'); M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0
        fi = np.array([1e9, 2e9]); fo = np.array([1.5e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        a = amp(cr, ci, 0, 0, 0, 0)
        expected = math.sqrt(2.5)
        self.assertAlmostEqual(a, expected, delta=0.01)

    def test_M_slerp_interpolation(self):
        """M should be interpolated via SLERP (phase rotation)."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pl = np.array([10.0])
        pg = np.ones((1, 2), order='F')
        M = np.zeros((2, 1, 2), order='F')
        M[0, 0, 0] = 1.0  # ReVV = 1 at freq_in[0]
        M[1, 0, 1] = 1.0  # ImVV = 1 at freq_in[1] (90 deg phase)
        fi = np.array([1e9, 2e9]); fo = np.array([1.5e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        a = amp(cr, ci, 0, 0, 0, 0)
        self.assertGreater(a, 0.5)
        self.assertLess(a, 1.5)

    # ============================================================================
    # SECTION 7: Extrapolation (clamping)
    # ============================================================================

    def test_extrapolation_clamps(self):
        """Querying below freq_in range should clamp to first entry."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pl = np.array([10.0])
        pg = np.array([[1.0, 4.0]], order='F')
        M = np.zeros((2, 1, 2), order='F'); M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0
        fi = np.array([1e9, 2e9])

        cr_lo, ci_lo, _ = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, np.array([100.0]))

        cr_ex, ci_ex, _ = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, np.array([1e9]))

        amp_lo = amp(cr_lo, ci_lo, 0, 0, 0, 0)
        amp_ex = amp(cr_ex, ci_ex, 0, 0, 0, 0)
        self.assertAlmostEqual(amp_lo, amp_ex, delta=1e-10)

    # ============================================================================
    # SECTION 8: Multiple output frequencies
    # ============================================================================

    def test_multiple_output_frequencies(self):
        """Multiple output frequencies should produce correct shape and finite values."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.linspace(500e6, 3e9, 20)

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[3], 20)
        self.assertTrue(np.all(np.isfinite(cr)))
        self.assertTrue(np.all(np.isfinite(ci)))
        self.assertTrue(np.all(np.isfinite(dl)))

    # ============================================================================
    # SECTION 9: GHz antenna with frequency-dependent pattern
    # ============================================================================

    def test_freq_dependent_pattern(self):
        """Frequency-dependent pattern (scaled by frequency) should increase amplitude."""
        ghz_freqs = [1.0e9, 2.0e9, 3.0e9]
        base = gen_custom(90.0, 90.0, 0.0)
        n_el, n_az, n_elem = base['e_theta_re'].shape
        etr = np.zeros((n_el, n_az, n_elem, 3), order='F')
        eti = np.zeros((n_el, n_az, n_elem, 3), order='F')
        epr = np.zeros((n_el, n_az, n_elem, 3), order='F')
        epi = np.zeros((n_el, n_az, n_elem, 3), order='F')
        for i in range(3):
            scale = 1.0 + 0.5 * i
            etr[:, :, :, i] = base['e_theta_re'] * scale
            eti[:, :, :, i] = base['e_theta_im'] * scale
            epr[:, :, :, i] = base['e_phi_re'] * scale
            epi[:, :, :, i] = base['e_phi_im'] * scale

        tx = dict(base)
        tx['e_theta_re'] = etr; tx['e_theta_im'] = eti
        tx['e_phi_re'] = epr; tx['e_phi_im'] = epi
        tx['center_freq'] = np.array(ghz_freqs)

        rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0; M[6, 0, 0] = -1.0
        fi = np.array([1e9]); fo = np.array([1e9, 2e9, 3e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[3], 3)
        a0 = amp(cr, ci, 0, 0, 0, 0)
        a1 = amp(cr, ci, 0, 0, 0, 1)
        a2 = amp(cr, ci, 0, 0, 0, 2)
        self.assertGreater(a1, a0)
        self.assertGreater(a2, a1)

    # ============================================================================
    # SECTION 10: Acoustic speaker simulation
    # ============================================================================

    def test_acoustic_speaker_simulation(self):
        """Acoustic speaker: verify output shape and finite values."""
        tx_woofer = quadriga_lib.arrayant.generate_speaker(
            driver_type='piston', radius=0.083, lower_cutoff=50.0, upper_cutoff=3000.0,
            lower_rolloff_slope=12.0, upper_rolloff_slope=24.0, sensitivity=87.0,
            radiation_type='hemisphere', baffle_width=0.20, baffle_height=0.30,
            frequencies=np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0]),
            angular_resolution=10.0)

        # Note: arrayant_concat_multi is not yet available in Python, so we test
        # each driver individually to confirm the API works with generate_speaker output
        rx = gen_omni()
        fbs = np.array([[0.5], [0.0], [0.0]], order='F')
        lbs = np.array([[0.5], [0.0], [0.0]], order='F')
        pl = np.array([1.0])
        fi = np.array([100.0, 10000.0])
        pg = np.ones((1, 2), order='F')
        M = np.zeros((2, 1, 2), order='F'); M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0
        fo = np.array([200.0, 1000.0, 5000.0])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx_woofer, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, propagation_speed=343.0)

        self.assertEqual(cr.shape[3], 3)  # 3 output frequencies
        self.assertTrue(np.all(np.isfinite(cr)))
        self.assertTrue(np.all(np.isfinite(dl)))

    # ============================================================================
    # SECTION 11: Full polarimetric Jones matrix
    # ============================================================================

    def test_full_polarimetric_jones(self):
        """Full polarimetric Jones with cross-pol should produce non-zero H response."""
        probe = gen_xpol()
        tx = gen_omni(); rx = probe
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((8, 1, 1), order='F')
        M[0, 0, 0] = 1.0   # ReVV
        M[2, 0, 0] = 0.5   # ReVH
        M[6, 0, 0] = -1.0  # ReHH
        fi = np.array([1e9]); fo = np.array([1e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[0], 2)  # xpol has 2 elements
        self.assertEqual(cr.shape[1], 1)
        amp_v = amp(cr, ci, 0, 0, 0, 0)
        amp_h = amp(cr, ci, 1, 0, 0, 0)
        self.assertGreater(amp_v, 0.1)
        self.assertGreater(amp_h, 0.1)

    # ============================================================================
    # SECTION 12: TX and RX orientation
    # ============================================================================

    def test_tx_orientation(self):
        """TX orientation should rotate the pattern."""
        ant = gen_omni()
        ant = copy_element(ant, 0)
        ant['element_pos'][1, 0] = 15.0
        ant['element_pos'][1, 1] = -15.0
        ant['e_theta_re'][:, :, 1] *= 2.0

        probe = gen_xpol()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; fbs[2, 0] = 1.0; lbs[:] = fbs
        pg = np.ones((1, 1), order='F'); pl = np.zeros(1)
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0; M[6, 0, 0] = -1.0
        fi = np.array([2997924580.0]); fo = np.array([2997924580.0])
        pi = math.pi

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            ant, probe, fbs, lbs, pg, pl, M,
            np.array([0.0, 0.0, 1.0]), np.array([-pi / 2.0, 0.0, 0.0]),
            np.array([20.0, 0.0, 1.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[0], 2)  # xpol probe has 2 elements
        self.assertEqual(cr.shape[1], 2)  # TX has 2 elements
        self.assertEqual(cr.shape[3], 1)  # single freq
        self.assertTrue(np.all(np.isfinite(cr)))

    # ============================================================================
    # SECTION 13: Frequency-dependent phase from wave number
    # ============================================================================

    def test_different_frequencies_different_phases(self):
        """Different frequencies should produce different phases but same amplitude."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1.0e9, 1.5e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        a0 = amp(cr, ci, 0, 0, 0, 0)
        a1 = amp(cr, ci, 0, 0, 0, 1)
        npt.assert_allclose(a0, a1, atol=1e-10, rtol=0)

        p0 = math.atan2(-ci[0, 0, 0, 0], cr[0, 0, 0, 0])
        p1 = math.atan2(-ci[0, 0, 0, 1], cr[0, 0, 0, 1])
        self.assertGreater(abs(p0 - p1), 1e-6)

        # Delays should be the same (same geometry)
        npt.assert_allclose(dl[0, 0, 0, 0], dl[0, 0, 0, 1], atol=1e-14, rtol=0)

    # ============================================================================
    # SECTION 14: NLOS paths
    # ============================================================================

    def test_los_and_nlos_delays(self):
        """LOS and NLOS paths should have different delays."""
        C = 299792458.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 2), order='F'); lbs = np.zeros((3, 2), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0        # LOS path
        fbs[0, 1] = 5.0;  fbs[1, 1] = 5.0          # NLOS FBS
        lbs[0, 1] = 5.0;  lbs[1, 1] = 5.0

        d_nlos = math.sqrt(25.0 + 25.0) + math.sqrt(25.0 + 25.0)
        pl = np.array([10.0, d_nlos])
        pg = np.ones((2, 1), order='F')
        M = np.zeros((8, 2, 1), order='F')
        M[0, 0, 0] = 1.0; M[0, 1, 0] = 1.0
        M[6, 0, 0] = -1.0; M[6, 1, 0] = -1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        self.assertLess(dl[0, 0, 0, 0], dl[0, 0, 1, 0])
        npt.assert_allclose(dl[0, 0, 0, 0], 10.0 / C, atol=1e-12, rtol=0)
        npt.assert_allclose(dl[0, 0, 1, 0], d_nlos / C, atol=1e-10, rtol=0)

    # ============================================================================
    # SECTION 15: Multi-element MIMO antenna
    # ============================================================================

    def test_2x2_mimo(self):
        """2x2 MIMO: correct dimensions and element scaling."""
        tx_ant = gen_omni()
        tx_ant = copy_element(tx_ant, 0)
        tx_ant['element_pos'][1, 0] = 0.5
        tx_ant['element_pos'][1, 1] = -0.5
        tx_ant['e_theta_re'][:, :, 1] *= 2.0

        rx_ant = gen_omni()
        rx_ant = copy_element(rx_ant, 0)

        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 20.0; lbs[0, 0] = 20.0
        pg = np.ones((1, 1), order='F'); pl = np.array([20.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0; M[6, 0, 0] = -1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx_ant, rx_ant, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([20.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[0], 2)  # 2 RX elements
        self.assertEqual(cr.shape[1], 2)  # 2 TX elements
        self.assertEqual(cr.shape[2], 1)  # 1 path

        # TX element 1 has 2x gain -> column 1 should have ~2x amplitude
        a00 = amp(cr, ci, 0, 0, 0, 0)
        a01 = amp(cr, ci, 0, 1, 0, 0)
        self.assertAlmostEqual(a01 / a00, 2.0, delta=0.1)

    # ============================================================================
    # SECTION 17: Zero center_frequency (phase disabled)
    # ============================================================================

    def test_zero_frequency_disables_phase(self):
        """Zero output frequency should disable phase rotation."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([0.0])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        self.assertAlmostEqual(abs(ci[0, 0, 0, 0]), 0.0, places=14)
        self.assertAlmostEqual(cr[0, 0, 0, 0], 1.0, delta=1e-10)

    # ============================================================================
    # SECTION 18: Multiple freq_in entries
    # ============================================================================

    def test_multiple_freq_in_varying_gain(self):
        """Multiple freq_in entries with varying path gain should interpolate correctly."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pl = np.array([10.0])
        fi = np.array([1e9, 2e9, 3e9])
        pg = np.array([[1.0, 4.0, 9.0]], order='F')
        M = np.zeros((2, 1, 3), order='F')
        M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0; M[0, 0, 2] = 1.0
        fo = np.array([1e9, 2e9, 3e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        # At exact freq_in entries: amplitude = sqrt(gain)
        a0 = amp(cr, ci, 0, 0, 0, 0)
        a1 = amp(cr, ci, 0, 0, 0, 1)
        a2 = amp(cr, ci, 0, 0, 0, 2)
        self.assertAlmostEqual(a0, 1.0, delta=0.01)
        self.assertAlmostEqual(a1, 2.0, delta=0.01)
        self.assertAlmostEqual(a2, 3.0, delta=0.01)

    # ============================================================================
    # SECTION 19: Absolute vs relative delays
    # ============================================================================

    def test_absolute_vs_relative_delays(self):
        """Absolute delay includes LOS, relative subtracts it."""
        C = 299792458.0; dist = 20.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = dist; lbs[0, 0] = dist
        pg = np.ones((1, 1), order='F'); pl = np.array([dist])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        _, _, dl_a = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([dist, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        _, _, dl_r = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([dist, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=False)

        npt.assert_allclose(dl_a[0, 0, 0, 0], dist / C, atol=1e-12, rtol=0)
        self.assertAlmostEqual(abs(dl_r[0, 0, 0, 0]), 0.0, places=14)

    # ============================================================================
    # SECTION 20: Amplitude stable, phase varies with frequency
    # ============================================================================

    def test_sweep_amplitude_stable_phase_varies(self):
        """Across a frequency sweep, amplitude should be stable, phase should vary."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.linspace(1e9, 3e9, 10)

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        amp_ref = amp(cr, ci, 0, 0, 0, 0)
        for f in range(1, 10):
            a = amp(cr, ci, 0, 0, 0, f)
            self.assertAlmostEqual(a, amp_ref, delta=1e-10)

    # ============================================================================
    # SECTION 22: Acoustic single-bounce reflection
    # ============================================================================

    def test_acoustic_single_bounce(self):
        """Acoustic simulation: NLOS delay should be greater than LOS delay."""
        c_sound = 343.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 2), order='F'); lbs = np.zeros((3, 2), order='F')
        fbs[0, 0] = 3.0; lbs[0, 0] = 3.0                    # LOS
        fbs[0, 1] = 1.5; fbs[1, 1] = 2.0                    # Wall reflection
        lbs[0, 1] = 1.5; lbs[1, 1] = 2.0

        d_los = 3.0
        d_nlos = math.sqrt(1.5 * 1.5 + 4.0) + math.sqrt(1.5 * 1.5 + 4.0)
        pl = np.array([d_los, d_nlos])
        pg = np.zeros((2, 1), order='F'); pg[0, 0] = 1.0; pg[1, 0] = 0.5
        M = np.zeros((2, 2, 1), order='F')
        M[0, 0, 0] = 1.0; M[0, 1, 0] = 1.0
        fi = np.array([500.0]); fo = np.array([500.0])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([3.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True, propagation_speed=c_sound)

        self.assertLess(dl[0, 0, 0, 0], dl[0, 0, 1, 0])
        npt.assert_allclose(dl[0, 0, 0, 0], d_los / c_sound, atol=1e-10, rtol=0)
        npt.assert_allclose(dl[0, 0, 1, 0], d_nlos / c_sound, atol=1e-10, rtol=0)

    # ============================================================================
    # SECTION 23: Frequency-dependent path gain with acoustic simulation
    # ============================================================================

    def test_air_absorption_increases_with_frequency(self):
        """Higher frequency should reduce amplitude (air absorption model)."""
        c_sound = 343.0
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 5.0; lbs[0, 0] = 5.0
        pl = np.array([5.0])
        fi = np.array([200.0, 1000.0, 5000.0, 10000.0])
        pg = np.array([[1.0, 0.8, 0.4, 0.1]], order='F')
        M = np.zeros((2, 1, 4), order='F')
        for s in range(4):
            M[0, 0, s] = 1.0

        fo = np.array([200.0, 5000.0, 10000.0])
        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([5.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True, propagation_speed=c_sound)

        a_lo = amp(cr, ci, 0, 0, 0, 0)
        a_mid = amp(cr, ci, 0, 0, 0, 1)
        a_hi = amp(cr, ci, 0, 0, 0, 2)
        self.assertGreater(a_lo, a_mid)
        self.assertGreater(a_mid, a_hi)

    # ============================================================================
    # SECTION 24: Coupling interpolation across frequency
    # ============================================================================

    def test_freq_dependent_coupling(self):
        """Frequency-dependent coupling should produce different coefficients."""
        base = gen_omni()
        base = copy_element(base, 0)

        n_el, n_az, n_elem = base['e_theta_re'].shape
        etr = np.zeros((n_el, n_az, n_elem, 2), order='F')
        eti = np.zeros((n_el, n_az, n_elem, 2), order='F')
        epr = np.zeros((n_el, n_az, n_elem, 2), order='F')
        epi = np.zeros((n_el, n_az, n_elem, 2), order='F')
        for i in range(2):
            etr[:, :, :, i] = base['e_theta_re']
            eti[:, :, :, i] = base['e_theta_im']
            epr[:, :, :, i] = base['e_phi_re']
            epi[:, :, :, i] = base['e_phi_im']

        tx = dict(base)
        tx['e_theta_re'] = etr; tx['e_theta_im'] = eti
        tx['e_phi_re'] = epr; tx['e_phi_im'] = epi
        tx['center_freq'] = np.array([1e9, 2e9])

        # Coupling: 3D (n_elem, n_ports, n_freq)
        cpl_re = np.zeros((2, 2, 2), order='F')
        cpl_re[0, 0, 0] = 1.0; cpl_re[1, 0, 0] = 0.5
        cpl_re[0, 1, 0] = 0.5; cpl_re[1, 1, 0] = 1.0
        cpl_re[0, 0, 1] = 1.0; cpl_re[1, 1, 1] = 1.0
        tx['coupling_re'] = cpl_re
        tx['coupling_im'] = np.zeros((2, 2, 2), order='F')

        rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((8, 1, 1), order='F')
        M[0, 0, 0] = 1.0; M[6, 0, 0] = -1.0
        fi = np.array([1e9]); fo = np.array([1e9, 2e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape[3], 2)
        # The two frequency slices should differ (different coupling)
        self.assertFalse(np.allclose(cr[:, :, :, 0], cr[:, :, :, 1], atol=1e-6))

    # ============================================================================
    # SECTION 25: Reversed freq_out order
    # ============================================================================

    def test_reversed_freq_out_order(self):
        """Reversed freq_out order should give consistent results."""
        tx = gen_omni(); rx = gen_omni()
        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.array([[1.0, 4.0]], order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 2), order='F'); M[0, 0, 0] = 1.0; M[0, 0, 1] = 1.0
        fi = np.array([1e9, 2e9])
        fo_fwd = np.array([1.2e9, 1.8e9])
        fo_rev = np.array([1.8e9, 1.2e9])

        cr_f, ci_f, dl_f = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo_fwd, use_absolute_delays=True)

        cr_r, ci_r, dl_r = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo_rev, use_absolute_delays=True)

        # Forward[:,:,:,0] == Reverse[:,:,:,1] and vice versa
        npt.assert_allclose(cr_f[:, :, :, 0], cr_r[:, :, :, 1], atol=1e-12, rtol=0)
        npt.assert_allclose(cr_f[:, :, :, 1], cr_r[:, :, :, 0], atol=1e-12, rtol=0)
        npt.assert_allclose(dl_f[:, :, :, 0], dl_r[:, :, :, 1], atol=1e-14, rtol=0)
        npt.assert_allclose(dl_f[:, :, :, 1], dl_r[:, :, :, 0], atol=1e-14, rtol=0)

    # ============================================================================
    # Edge case: 3D (single-freq) antenna input
    # ============================================================================

    def test_single_freq_3d_antenna_input(self):
        """Standard 3D-pattern antenna dict (no 4th freq dim) should work correctly."""
        tx = gen_omni(); rx = gen_omni()

        # Verify input is indeed 3D
        self.assertEqual(tx['e_theta_re'].ndim, 3)

        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((2, 1, 1), order='F'); M[0, 0, 0] = 1.0
        fi = np.array([1e9]); fo = np.array([1e9, 2e9])

        cr, ci, dl = quadriga_lib.arrayant.get_channels_multifreq(
            tx, rx, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo)

        self.assertEqual(cr.shape, (1, 1, 1, 2))
        self.assertTrue(np.all(np.isfinite(cr)))
        self.assertTrue(np.all(np.isfinite(ci)))
        self.assertTrue(np.all(np.isfinite(dl)))

        # Amplitudes should be identical (same antenna clamped for both frequencies)
        a0 = amp(cr, ci, 0, 0, 0, 0)
        a1 = amp(cr, ci, 0, 0, 0, 1)
        npt.assert_allclose(a0, a1, atol=1e-10, rtol=0)

    def test_single_freq_3d_matches_4d_single_slice(self):
        """3D input should produce same results as equivalent 4D with 1 freq slice."""
        tx_3d = gen_omni(); rx_3d = gen_omni()

        # Create 4D version by adding a frequency dimension
        tx_4d = dict(tx_3d)
        for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im']:
            tx_4d[key] = tx_3d[key][:, :, :, np.newaxis].copy(order='F')
        tx_4d['center_freq'] = np.array([299792458.0])

        rx_4d = dict(rx_3d)
        for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im']:
            rx_4d[key] = rx_3d[key][:, :, :, np.newaxis].copy(order='F')
        rx_4d['center_freq'] = np.array([299792458.0])

        fbs = np.zeros((3, 1), order='F'); lbs = np.zeros((3, 1), order='F')
        fbs[0, 0] = 10.0; lbs[0, 0] = 10.0
        pg = np.ones((1, 1), order='F'); pl = np.array([10.0])
        M = np.zeros((8, 1, 1), order='F'); M[0, 0, 0] = 1.0; M[6, 0, 0] = -1.0
        fi = np.array([1e9]); fo = np.array([1e9])

        cr_3d, ci_3d, dl_3d = quadriga_lib.arrayant.get_channels_multifreq(
            tx_3d, rx_3d, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        cr_4d, ci_4d, dl_4d = quadriga_lib.arrayant.get_channels_multifreq(
            tx_4d, rx_4d, fbs, lbs, pg, pl, M,
            np.zeros(3), np.zeros(3), np.array([10.0, 0.0, 0.0]), np.zeros(3),
            fi, fo, use_absolute_delays=True)

        npt.assert_allclose(cr_3d, cr_4d, atol=1e-12, rtol=0)
        npt.assert_allclose(ci_3d, ci_4d, atol=1e-12, rtol=0)
        npt.assert_allclose(dl_3d, dl_4d, atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
