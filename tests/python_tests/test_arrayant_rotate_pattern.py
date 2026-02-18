import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import arrayant


def make_single_freq_ant(n_el=19, n_az=37, n_elem=1, freq=1e9, res=10.0):
    """Helper: create a single-frequency arrayant via generate('omni')."""
    ant = arrayant.generate('omni', res, freq=freq)
    if n_elem > 1:
        for i in range(1, n_elem):
            ant = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                        dest_element=np.array([i], dtype=np.uint64))
    return ant


def make_multi_freq_ant(n_elem=1, n_freq=3, res=10.0):
    """Helper: create a multi-frequency arrayant dict with 4D patterns from omni patterns."""
    n_el = int(180.0 / res) + 1
    n_az = int(360.0 / res) + 1
    el_grid = np.linspace(-np.pi / 2, np.pi / 2, n_el)
    az_grid = np.linspace(-np.pi, np.pi, n_az)
    freqs = np.array([1000.0 * (i + 1) for i in range(n_freq)])

    # Omni pattern: e_theta_re = 1.0 everywhere
    e_theta_re = np.ones((n_el, n_az, n_elem, n_freq), order='F')
    return {
        'e_theta_re': e_theta_re,
        'e_theta_im': np.zeros((n_el, n_az, n_elem, n_freq), order='F'),
        'e_phi_re': np.zeros((n_el, n_az, n_elem, n_freq), order='F'),
        'e_phi_im': np.zeros((n_el, n_az, n_elem, n_freq), order='F'),
        'azimuth_grid': az_grid,
        'elevation_grid': el_grid,
        'element_pos': np.zeros((3, n_elem), order='F'),
        'coupling_re': np.eye(n_elem, order='F'),
        'coupling_im': np.zeros((n_elem, n_elem), order='F'),
        'center_freq': freqs,
        'name': 'test_multi',
    }


def make_dipole_single(res=10.0, freq=1e9):
    """Helper: create a single-frequency dipole antenna."""
    return arrayant.generate('dipole', res, freq=freq)


def make_dipole_multi(n_freq=3, res=10.0):
    """Helper: create a multi-frequency dipole-like antenna (directional in e_theta)."""
    ant = arrayant.generate('dipole', res)
    n_el, n_az, n_elem = ant['e_theta_re'].shape
    freqs = np.array([1000.0 * (i + 1) for i in range(n_freq)])

    # Stack same dipole pattern across frequencies
    e_theta_re = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_theta_im = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_phi_re = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_phi_im = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    for f in range(n_freq):
        e_theta_re[:, :, :, f] = ant['e_theta_re']
        e_theta_im[:, :, :, f] = ant['e_theta_im']
        e_phi_re[:, :, :, f] = ant['e_phi_re']
        e_phi_im[:, :, :, f] = ant['e_phi_im']

    return {
        'e_theta_re': e_theta_re,
        'e_theta_im': e_theta_im,
        'e_phi_re': e_phi_re,
        'e_phi_im': e_phi_im,
        'azimuth_grid': ant['azimuth_grid'].copy(),
        'elevation_grid': ant['elevation_grid'].copy(),
        'element_pos': ant['element_pos'].copy(),
        'coupling_re': ant['coupling_re'].copy(),
        'coupling_im': ant['coupling_im'].copy(),
        'center_freq': freqs,
        'name': 'dipole_multi',
    }


class TestRotatePatternSingleFreq(unittest.TestCase):
    """Tests for rotate_pattern with single-frequency (3D) arrayants."""

    # ================================================================================
    # Basic rotation
    # ================================================================================

    def test_zero_rotation_is_identity(self):
        """Zero rotation should return an identical pattern."""
        ant = make_dipole_single()
        out = arrayant.rotate_pattern(ant, x_deg=0.0, y_deg=0.0, z_deg=0.0, usage=3)
        npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=10)
        npt.assert_array_almost_equal(out['e_phi_re'], ant['e_phi_re'], decimal=10)

    def test_360_rotation_is_identity(self):
        """360 degree rotation around any axis should return the original pattern."""
        ant = make_dipole_single()
        out = arrayant.rotate_pattern(ant, z_deg=360.0, usage=3)
        npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)

    def test_y_rotation_shifts_pattern(self):
        """Rotation around y-axis (tilt) should shift the dipole pattern."""
        ant = make_dipole_single(res=10.0)
        out = arrayant.rotate_pattern(ant, y_deg=90.0, usage=3)
        # Pattern should differ from original (dipole is symmetric around z, not y)
        self.assertFalse(np.allclose(out['e_theta_re'], ant['e_theta_re'], atol=0.01),
                         "90 deg y-rotation on dipole should change pattern")

    def test_x_rotation_bank(self):
        """Bank rotation (x-axis) should modify the pattern."""
        ant = make_dipole_single(res=10.0)
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=3)
        self.assertFalse(np.allclose(out['e_theta_re'], ant['e_theta_re'], atol=0.01))

    def test_y_rotation_tilt(self):
        """Tilt rotation (y-axis) should modify the pattern."""
        ant = make_dipole_single(res=10.0)
        out = arrayant.rotate_pattern(ant, y_deg=30.0, usage=3)
        self.assertFalse(np.allclose(out['e_theta_re'], ant['e_theta_re'], atol=0.01))

    # ================================================================================
    # Usage modes
    # ================================================================================

    def test_usage_2_polarization_only_on_omni(self):
        """Usage 2 (polarization only) on omni should change e_phi but keep e_theta power."""
        ant = make_single_freq_ant(res=10.0)
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=2)
        # Omni has all power in e_theta. After polarization rotation, some power goes to e_phi.
        self.assertGreater(np.max(np.abs(out['e_phi_re'])), 0.1,
                           "Polarization rotation should move power to e_phi")

    def test_usage_3_no_grid_adjust(self):
        """Usage 3 should not change the azimuth/elevation grids."""
        ant = make_dipole_single(res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=45.0, usage=3)
        npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'])
        npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'])

    # ================================================================================
    # Element selection
    # ================================================================================

    def test_rotate_specific_element_only(self):
        """Rotating only element 0 should leave element 1 unchanged."""
        ant = arrayant.generate('xpol', 10.0)  # 2 elements
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=2, element=np.array([0], dtype=np.uint32))
        # Element 0 was rotated, element 1 should be unchanged
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 1], ant['e_theta_re'][:, :, 1], decimal=10)
        # Element 0 should have changed
        self.assertFalse(np.allclose(out['e_theta_re'][:, :, 0], ant['e_theta_re'][:, :, 0], atol=0.01))

    def test_rotate_all_elements_default(self):
        """Empty element array should rotate all elements."""
        ant = arrayant.generate('xpol', 10.0)
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=2)
        # Both elements should have changed
        self.assertFalse(np.allclose(out['e_theta_re'][:, :, 0], ant['e_theta_re'][:, :, 0], atol=0.01))
        self.assertFalse(np.allclose(out['e_phi_re'][:, :, 1], ant['e_phi_re'][:, :, 1], atol=0.01))

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_3d(self):
        """Single-freq input should produce 3D output."""
        ant = make_single_freq_ant(res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=10.0, usage=3)
        self.assertEqual(out['e_theta_re'].ndim, 3)

    def test_output_has_all_keys(self):
        """Output should contain all expected keys."""
        ant = make_single_freq_ant(res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=10.0, usage=3)
        for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im',
                     'azimuth_grid', 'elevation_grid', 'element_pos',
                     'coupling_re', 'coupling_im', 'center_freq', 'name']:
            self.assertIn(key, out)

    def test_input_not_modified(self):
        """Input dict should not be modified."""
        ant = make_dipole_single(res=10.0)
        original = ant['e_theta_re'].copy()
        _ = arrayant.rotate_pattern(ant, z_deg=90.0, usage=3)
        npt.assert_array_equal(ant['e_theta_re'], original)

    # ================================================================================
    # Energy conservation
    # ================================================================================

    def test_total_power_conserved(self):
        """Total power (sum of squared magnitudes) should be conserved under polarization rotation."""
        ant = make_dipole_single(res=5.0)
        out = arrayant.rotate_pattern(ant, x_deg=30.0, y_deg=20.0, z_deg=45.0, usage=2)
        power_in = (np.sum(ant['e_theta_re'] ** 2 + ant['e_theta_im'] ** 2 +
                           ant['e_phi_re'] ** 2 + ant['e_phi_im'] ** 2))
        power_out = (np.sum(out['e_theta_re'] ** 2 + out['e_theta_im'] ** 2 +
                            out['e_phi_re'] ** 2 + out['e_phi_im'] ** 2))
        npt.assert_almost_equal(power_out, power_in, decimal=10)

    # ================================================================================
    # Inverse rotation
    # ================================================================================

    def test_inverse_rotation_restores_pattern(self):
        """Rotating by +angle then -angle (usage 3, no grid adjust) should restore pattern."""
        ant = make_dipole_single(res=1.0)
        fwd = arrayant.rotate_pattern(ant, y_deg=90.0, usage=3)
        back = arrayant.rotate_pattern(fwd, y_deg=-90.0, usage=3)
        npt.assert_array_almost_equal(back['e_theta_re'], ant['e_theta_re'], decimal=3)


class TestRotatePatternMultiFreq(unittest.TestCase):
    """Tests for rotate_pattern with multi-frequency (4D) arrayants."""

    # ================================================================================
    # Basic rotation
    # ================================================================================

    def test_zero_rotation_is_identity(self):
        """Zero rotation should return an identical pattern at all frequencies."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        out = arrayant.rotate_pattern(ant, x_deg=0.0, y_deg=0.0, z_deg=0.0, usage=3)
        npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=10)

    def test_y_rotation_modifies_pattern(self):
        """Y-axis rotation should change the directional pattern at all frequencies."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        out = arrayant.rotate_pattern(ant, y_deg=90.0, usage=3)
        for f in range(3):
            self.assertFalse(np.allclose(out['e_theta_re'][:, :, 0, f],
                                         ant['e_theta_re'][:, :, 0, f], atol=0.01),
                             f"Pattern at freq index {f} should change after 90 deg y-rotation")

    def test_rotation_consistent_across_frequencies(self):
        """For identical patterns, rotation result should be the same at all frequencies."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        out = arrayant.rotate_pattern(ant, y_deg=45.0, usage=3)
        for f in range(1, 3):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0, f],
                                          out['e_theta_re'][:, :, 0, 0], decimal=10)

    def test_360_rotation_is_identity(self):
        """360 deg rotation should return the original pattern."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=360.0, usage=3)
        npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)


    # ================================================================================
    # Usage modes (multi-freq: grid adjust disabled automatically)
    # ================================================================================

    def test_usage_0_no_grid_adjust_multi(self):
        """Usage 0 on multi-freq should not adjust grids (mapped to usage 3 internally)."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=45.0, usage=0)
        npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'])
        npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'])

    def test_usage_2_polarization_only(self):
        """Usage 2 on multi-freq omni should move power to e_phi."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=2)
        self.assertGreater(np.max(np.abs(out['e_phi_re'])), 0.1)

    # ================================================================================
    # Element selection
    # ================================================================================

    def test_rotate_specific_element_multi(self):
        """Rotating element 0 should leave element 1 unchanged at all frequencies."""
        ant = make_multi_freq_ant(n_elem=2, n_freq=3, res=10.0)
        # Set element 1 to a different value for identification
        ant['e_theta_re'][:, :, 1, :] = 2.0
        out = arrayant.rotate_pattern(ant, x_deg=45.0, usage=2,
                                      element=np.array([0], dtype=np.uint32))
        # Element 1 (0-based) should be unchanged
        for f in range(3):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 1, f],
                                          ant['e_theta_re'][:, :, 1, f], decimal=10)

    def test_rotate_all_elements_multi(self):
        """Empty element array should rotate all elements at all frequencies."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, y_deg=90.0, usage=3)
        for f in range(2):
            self.assertFalse(np.allclose(out['e_theta_re'][:, :, 0, f],
                                         ant['e_theta_re'][:, :, 0, f], atol=0.01))

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_4d(self):
        """Multi-freq input should produce 4D output."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=10.0, usage=3)
        self.assertEqual(out['e_theta_re'].ndim, 4)
        self.assertEqual(out['e_theta_re'].shape[3], 3)

    def test_center_freq_preserved(self):
        """center_freq should be preserved."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=10.0, usage=3)
        npt.assert_array_almost_equal(out['center_freq'], ant['center_freq'])

    def test_grids_unchanged(self):
        """Grids should be unchanged for multi-freq (no grid adjust)."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=45.0, usage=3)
        npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'])
        npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'])

    def test_name_preserved(self):
        """Name should be preserved."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=2, res=10.0)
        out = arrayant.rotate_pattern(ant, z_deg=10.0)
        self.assertEqual(out['name'], 'test_multi')

    def test_input_not_modified(self):
        """Input should not be modified."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        original = ant['e_theta_re'].copy()
        _ = arrayant.rotate_pattern(ant, z_deg=90.0, usage=3)
        npt.assert_array_equal(ant['e_theta_re'], original)

    # ================================================================================
    # Energy conservation
    # ================================================================================

    def test_total_power_conserved_multi(self):
        """Total power should be conserved under polarization rotation at each frequency."""
        ant = make_dipole_multi(n_freq=3, res=5.0)
        out = arrayant.rotate_pattern(ant, x_deg=30.0, y_deg=20.0, z_deg=45.0, usage=2)
        for f in range(3):
            power_in = (np.sum(ant['e_theta_re'][:, :, :, f] ** 2 +
                               ant['e_theta_im'][:, :, :, f] ** 2 +
                               ant['e_phi_re'][:, :, :, f] ** 2 +
                               ant['e_phi_im'][:, :, :, f] ** 2))
            power_out = (np.sum(out['e_theta_re'][:, :, :, f] ** 2 +
                                out['e_theta_im'][:, :, :, f] ** 2 +
                                out['e_phi_re'][:, :, :, f] ** 2 +
                                out['e_phi_im'][:, :, :, f] ** 2))
            npt.assert_almost_equal(power_out, power_in, decimal=10)

    # ================================================================================
    # Inverse rotation
    # ================================================================================

    def test_inverse_rotation_restores_pattern_multi(self):
        """Rotate +angle then -angle should restore pattern at all frequencies."""
        ant = make_dipole_multi(n_freq=2, res=3.0)
        fwd = arrayant.rotate_pattern(ant, y_deg=90.0, usage=3)
        back = arrayant.rotate_pattern(fwd, y_deg=-90.0, usage=3)
        npt.assert_array_almost_equal(back['e_theta_re'], ant['e_theta_re'], decimal=3)

    # ================================================================================
    # Integration with generate_speaker
    # ================================================================================

    def test_rotate_generated_speaker(self):
        """Rotate a generated speaker and verify output structure."""
        speaker = arrayant.generate_speaker(frequencies=np.array([500.0, 2000.0]),
                                            angular_resolution=10.0)
        out = arrayant.rotate_pattern(speaker, y_deg=15.0, usage=3)
        self.assertEqual(out['e_theta_re'].ndim, 4)
        self.assertEqual(out['e_theta_re'].shape, speaker['e_theta_re'].shape)
        npt.assert_array_almost_equal(out['center_freq'], speaker['center_freq'])

    # ================================================================================
    # Combined axes
    # ================================================================================

    def test_combined_rotation_multi(self):
        """Combined x+y+z rotation should produce a different pattern than single-axis y rotation."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        out_y = arrayant.rotate_pattern(ant, y_deg=45.0, usage=3)
        out_xyz = arrayant.rotate_pattern(ant, x_deg=20.0, y_deg=45.0, z_deg=30.0, usage=3)
        self.assertFalse(np.allclose(out_y['e_theta_re'], out_xyz['e_theta_re'], atol=0.01),
                         "Combined rotation should differ from single-axis rotation")


if __name__ == '__main__':
    unittest.main()
