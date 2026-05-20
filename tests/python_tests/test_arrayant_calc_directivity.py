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


class test_arrayant_calc_directivity(unittest.TestCase):

    def test_dipole_default(self):
        # Short dipole: known directivity ≈ 1.76 dBi
        ant = quadriga_lib.arrayant.generate('dipole')
        directivity = quadriga_lib.arrayant.calc_directivity(ant)

        # Single-frequency input → 1D output
        self.assertEqual(directivity.ndim, 1)
        self.assertEqual(directivity.shape, (1,))
        npt.assert_allclose(directivity, [1.760964], atol=1e-6, rtol=0)

    def test_dipole_element_repeat(self):
        # Selecting the same element twice should return the same value twice
        ant = quadriga_lib.arrayant.generate('dipole')
        directivity = quadriga_lib.arrayant.calc_directivity(ant, np.array([0, 0]))

        self.assertEqual(directivity.shape, (2,))
        npt.assert_allclose(directivity, [1.760964, 1.760964], atol=1e-6, rtol=0)

    def test_xpol_all_elements(self):
        # xpol: 2 cross-polarized isotropic elements → 0 dBi each
        ant = quadriga_lib.arrayant.generate('xpol')
        directivity = quadriga_lib.arrayant.calc_directivity(ant)

        self.assertEqual(directivity.shape, (2,))
        npt.assert_allclose(directivity, [0.0, 0.0], atol=1e-6, rtol=0)

    def test_xpol_subset(self):
        # Subset selection: 0-based index 1 picks the second element
        ant = quadriga_lib.arrayant.generate('xpol')
        directivity = quadriga_lib.arrayant.calc_directivity(ant, np.array([1]))

        self.assertEqual(directivity.shape, (1,))
        npt.assert_allclose(directivity, [0.0], atol=1e-6, rtol=0)

    def test_element_default_variants(self):
        # Omitted, None, and empty array should all behave identically (all elements)
        ant = quadriga_lib.arrayant.generate('xpol')
        expected = np.array([0.0, 0.0])

        d_omit  = quadriga_lib.arrayant.calc_directivity(ant)
        d_none  = quadriga_lib.arrayant.calc_directivity(ant, None)
        d_empty = quadriga_lib.arrayant.calc_directivity(ant, np.array([], dtype=np.uint64))

        for d in (d_omit, d_none, d_empty):
            self.assertEqual(d.shape, (2,))
            npt.assert_allclose(d, expected, atol=1e-6, rtol=0)

    def test_element_kwarg(self):
        # Keyword-argument form
        ant = quadriga_lib.arrayant.generate('xpol')
        directivity = quadriga_lib.arrayant.calc_directivity(ant, element=np.array([1, 0]))

        self.assertEqual(directivity.shape, (2,))
        npt.assert_allclose(directivity, [0.0, 0.0], atol=1e-6, rtol=0)

    def test_multi_freq_dipole(self):
        # Multi-frequency: stack the dipole pattern across 2 frequencies
        ant = quadriga_lib.arrayant.generate('dipole')
        ant_mf = _stack_multi_freq(ant, 2)
        directivity = quadriga_lib.arrayant.calc_directivity(ant_mf)

        # Multi-freq input → 2D output (n_out, n_freq)
        self.assertEqual(directivity.ndim, 2)
        self.assertEqual(directivity.shape, (1, 2))
        npt.assert_allclose(directivity, [[1.760964, 1.760964]], atol=1e-6, rtol=0)

    def test_multi_freq_with_subset(self):
        # Multi-frequency with element selection
        ant = quadriga_lib.arrayant.generate('dipole')
        ant_mf = _stack_multi_freq(ant, 2)
        directivity = quadriga_lib.arrayant.calc_directivity(ant_mf, np.array([0, 0]))

        self.assertEqual(directivity.shape, (2, 2))
        npt.assert_allclose(directivity,
                            [[1.760964, 1.760964],
                             [1.760964, 1.760964]],
                            atol=1e-6, rtol=0)

    def test_multi_freq_xpol(self):
        # Multi-frequency xpol with 3 frequencies, 2 elements → shape (2, 3)
        ant = quadriga_lib.arrayant.generate('xpol')
        ant_mf = _stack_multi_freq(ant, 3)
        directivity = quadriga_lib.arrayant.calc_directivity(ant_mf)

        self.assertEqual(directivity.shape, (2, 3))
        npt.assert_allclose(directivity, np.zeros((2, 3)), atol=1e-6, rtol=0)

    def test_error_element_out_of_bounds(self):
        # dipole has 1 element; index 1 is out of bounds
        ant = quadriga_lib.arrayant.generate('dipole')
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.calc_directivity(ant, np.array([1]))

    def test_error_missing_field(self):
        # arrayant dict missing a required pattern field
        ant = quadriga_lib.arrayant.generate('dipole')
        ant_bad = {k: v for k, v in ant.items() if k != 'e_theta_re'}
        with self.assertRaises(Exception):
            quadriga_lib.arrayant.calc_directivity(ant_bad)


if __name__ == '__main__':
    unittest.main()