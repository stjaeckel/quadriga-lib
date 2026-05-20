# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def _stack_multi_freq(ant, n_freq):
    """Build a multi-frequency arrayant dict by stacking the pattern fields along a new 4th axis."""
    return {
        'e_theta_re':     np.stack([ant['e_theta_re']] * n_freq, axis=-1),
        'e_theta_im':     np.stack([ant['e_theta_im']] * n_freq, axis=-1),
        'e_phi_re':       np.stack([ant['e_phi_re']]   * n_freq, axis=-1),
        'e_phi_im':       np.stack([ant['e_phi_im']]   * n_freq, axis=-1),
        'azimuth_grid':   ant['azimuth_grid'],
        'elevation_grid': ant['elevation_grid'],
    }


class test_arrayant_calc_beamwidth(unittest.TestCase):

    def test_dipole_default(self):
        # Short dipole: 360° az (omni in azimuth), ≈90° el (cos² shape), broadside pointing
        ant = quadriga_lib.arrayant.generate('dipole')
        bw_az, bw_el, _, el_pt = quadriga_lib.arrayant.calc_beamwidth(ant)

        # Single-frequency input → 1D outputs
        self.assertEqual(bw_az.ndim, 1)
        self.assertEqual(bw_az.shape, (1,))
        npt.assert_allclose(bw_az, [360.0], atol=0.05, rtol=0)
        npt.assert_allclose(bw_el, [90.0],  atol=0.5,  rtol=0)
        npt.assert_allclose(el_pt, [0.0],   atol=0.5,  rtol=0)

    def test_custom_recovers_input_beamwidth(self):
        # Custom(30°, 20°, 0) should recover its design beamwidth and point at (0, 0)
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0, rear_gain_lin=0.0)
        bw_az, bw_el, az_pt, el_pt = quadriga_lib.arrayant.calc_beamwidth(ant_c)

        npt.assert_allclose(bw_az, [30.0], atol=0.5, rtol=0)
        npt.assert_allclose(bw_el, [20.0], atol=0.5, rtol=0)
        npt.assert_allclose(az_pt, [0.0],  atol=0.1, rtol=0)
        npt.assert_allclose(el_pt, [0.0],  atol=0.1, rtol=0)

    def test_threshold_widens_beamwidth(self):
        # 6 dB threshold should yield a beamwidth at least 5° wider than 3 dB
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        bw3, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_c, threshold_dB=3.0)
        bw6, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_c, threshold_dB=6.0)

        self.assertGreater(bw6[0], bw3[0] + 5.0)

    def test_threshold_default(self):
        # Omitting threshold_dB uses the default value of 3.0
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        bw_omit, _, _, _     = quadriga_lib.arrayant.calc_beamwidth(ant_c)
        bw_explicit, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_c, threshold_dB=3.0)

        npt.assert_allclose(bw_omit, bw_explicit, atol=1e-9, rtol=0)

    def test_xpol_full_grid(self):
        # xpol: 2 cross-polarized isotropic elements -> full grid for both elements
        ant_xp = quadriga_lib.arrayant.generate('xpol')
        bw_az, bw_el, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_xp)

        self.assertEqual(bw_az.shape, (2,))
        npt.assert_allclose(bw_az, [360.0, 360.0], atol=0.05, rtol=0)
        npt.assert_allclose(bw_el, [180.0, 180.0], atol=0.05, rtol=0)

    def test_subset_selection(self):
        # 0-based index 1 picks only the second element
        ant_xp = quadriga_lib.arrayant.generate('xpol')
        bw_az, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_xp, np.array([1]))

        self.assertEqual(bw_az.shape, (1,))
        npt.assert_allclose(bw_az, [360.0], atol=0.05, rtol=0)

    def test_element_default_variants(self):
        # Omitted, None, and empty array should all behave identically (all elements)
        ant_xp = quadriga_lib.arrayant.generate('xpol')
        expected_az = np.array([360.0, 360.0])

        bw_omit, _, _, _  = quadriga_lib.arrayant.calc_beamwidth(ant_xp)
        bw_none, _, _, _  = quadriga_lib.arrayant.calc_beamwidth(ant_xp, None)
        bw_empty, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_xp, np.array([], dtype=np.uint64))

        for bw in (bw_omit, bw_none, bw_empty):
            self.assertEqual(bw.shape, (2,))
            npt.assert_allclose(bw, expected_az, atol=0.05, rtol=0)

    def test_element_kwarg(self):
        # Keyword-argument form
        ant_xp = quadriga_lib.arrayant.generate('xpol')
        bw_az, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_xp, element=np.array([1, 0]))

        self.assertEqual(bw_az.shape, (2,))
        npt.assert_allclose(bw_az, [360.0, 360.0], atol=0.05, rtol=0)

    def test_multi_freq_basic(self):
        # Multi-frequency: stack a custom (30°, 20°) pattern across 2 frequencies
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        ant_mf = _stack_multi_freq(ant_c, 2)
        bw_az, bw_el, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_mf)

        # Multi-freq input → 2D output (n_out, n_freq)
        self.assertEqual(bw_az.ndim, 2)
        self.assertEqual(bw_az.shape, (1, 2))
        npt.assert_allclose(bw_az, [[30.0, 30.0]], atol=0.5, rtol=0)
        npt.assert_allclose(bw_el, [[20.0, 20.0]], atol=0.5, rtol=0)

    def test_multi_freq_with_subset(self):
        # Multi-frequency with element selection → shape (2, 2)
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        ant_mf = _stack_multi_freq(ant_c, 2)
        bw_az, _, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_mf, np.array([0, 0]))

        self.assertEqual(bw_az.shape, (2, 2))
        npt.assert_allclose(bw_az,
                            [[30.0, 30.0],
                             [30.0, 30.0]],
                            atol=0.5, rtol=0)

    def test_multi_freq_xpol(self):
        # Multi-frequency xpol with 3 frequencies, 2 elements → shape (2, 3)
        ant_xp = quadriga_lib.arrayant.generate('xpol')
        ant_mf = _stack_multi_freq(ant_xp, 3)
        bw_az, bw_el, _, _ = quadriga_lib.arrayant.calc_beamwidth(ant_mf)

        self.assertEqual(bw_az.shape, (2, 3))
        npt.assert_allclose(bw_az, np.full((2, 3), 360.0), atol=0.05, rtol=0)
        npt.assert_allclose(bw_el, np.full((2, 3), 180.0), atol=0.05, rtol=0)

    def test_output_is_4_tuple(self):
        # Function should return a 4-tuple of numpy arrays
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        result = quadriga_lib.arrayant.calc_beamwidth(ant_c)

        self.assertEqual(len(result), 4)
        for arr in result:
            self.assertIsInstance(arr, np.ndarray)

    def test_error_element_out_of_bounds(self):
        # Custom antenna has 1 element; index 1 is out of bounds
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.calc_beamwidth(ant_c, np.array([1]))

    def test_error_missing_field(self):
        # arrayant dict missing a required pattern field
        ant_c = quadriga_lib.arrayant.generate('custom', az_3dB=30.0, el_3dB=20.0)
        ant_bad = {k: v for k, v in ant_c.items() if k != 'e_theta_re'}
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.calc_beamwidth(ant_bad)


if __name__ == '__main__':
    unittest.main()
