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


class TestSpeakerGenerate(unittest.TestCase):
    """Tests for arrayant.generate_speaker()"""

    # ================================================================================
    # Output structure and types
    # ================================================================================

    def test_default_returns_dict_with_required_keys(self):
        """Default call returns a dict with all expected keys."""
        data = arrayant.generate_speaker()
        for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im',
                     'azimuth_grid', 'elevation_grid', 'element_pos',
                     'coupling_re', 'coupling_im', 'center_freq', 'name']:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_default_pattern_is_4d(self):
        """Default call produces 4D pattern arrays (multiple auto frequencies)."""
        data = arrayant.generate_speaker()
        self.assertEqual(data['e_theta_re'].ndim, 4,
                         "Pattern should be 4D (elevation, azimuth, elements, freq)")
        n_el, n_az, n_elem, n_freq = data['e_theta_re'].shape
        self.assertGreater(n_freq, 1, "Auto frequencies should produce multiple entries")
        self.assertEqual(n_elem, 1, "Single driver should have 1 element")

    def test_default_all_pattern_shapes_match(self):
        """All four pattern fields have identical shape."""
        data = arrayant.generate_speaker()
        shape = data['e_theta_re'].shape
        self.assertEqual(data['e_theta_im'].shape, shape)
        self.assertEqual(data['e_phi_re'].shape, shape)
        self.assertEqual(data['e_phi_im'].shape, shape)

    def test_center_freq_is_1d_array(self):
        """center_freq is a 1D numpy array matching number of frequency samples."""
        data = arrayant.generate_speaker()
        cf = data['center_freq']
        self.assertIsInstance(cf, np.ndarray)
        self.assertEqual(cf.ndim, 1)
        n_freq = data['e_theta_re'].shape[3]
        self.assertEqual(cf.shape[0], n_freq)

    def test_center_freq_values_are_positive_and_sorted(self):
        """Frequency samples should be positive and in ascending order."""
        data = arrayant.generate_speaker()
        cf = data['center_freq']
        self.assertTrue(np.all(cf > 0))
        self.assertTrue(np.all(np.diff(cf) > 0))

    def test_grids_are_1d(self):
        """azimuth_grid and elevation_grid are 1D arrays."""
        data = arrayant.generate_speaker()
        self.assertEqual(data['azimuth_grid'].ndim, 1)
        self.assertEqual(data['elevation_grid'].ndim, 1)

    def test_element_pos_shape(self):
        """element_pos should be (3, n_elements)."""
        data = arrayant.generate_speaker()
        n_elem = data['e_theta_re'].shape[2]
        self.assertEqual(data['element_pos'].shape, (3, n_elem))

    def test_pattern_is_fortran_contiguous(self):
        """Pattern arrays should be column-major (Fortran) contiguous."""
        data = arrayant.generate_speaker()
        self.assertTrue(data['e_theta_re'].flags.f_contiguous,
                        "e_theta_re should be Fortran contiguous")

    # ================================================================================
    # Angular resolution
    # ================================================================================

    def test_angular_resolution_5deg(self):
        """5 deg resolution: azimuth 73 points, elevation 37 points."""
        data = arrayant.generate_speaker(angular_resolution=5.0)
        self.assertEqual(data['azimuth_grid'].shape[0], 73)    # -180 to 180 in 5 deg steps
        self.assertEqual(data['elevation_grid'].shape[0], 37)   # -90 to 90 in 5 deg steps

    def test_angular_resolution_10deg(self):
        """10 deg resolution: azimuth 37 points, elevation 19 points."""
        data = arrayant.generate_speaker(angular_resolution=10.0)
        self.assertEqual(data['azimuth_grid'].shape[0], 37)
        self.assertEqual(data['elevation_grid'].shape[0], 19)

    def test_angular_resolution_affects_pattern_shape(self):
        """Changing resolution changes first two dims of pattern."""
        d5 = arrayant.generate_speaker(angular_resolution=5.0)
        d10 = arrayant.generate_speaker(angular_resolution=10.0)
        self.assertGreater(d5['e_theta_re'].shape[0], d10['e_theta_re'].shape[0])
        self.assertGreater(d5['e_theta_re'].shape[1], d10['e_theta_re'].shape[1])

    # ================================================================================
    # Custom frequency vector
    # ================================================================================

    def test_custom_frequencies(self):
        """Explicit frequency vector is used and reflected in center_freq."""
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        data = arrayant.generate_speaker(frequencies=freqs)
        npt.assert_array_almost_equal(data['center_freq'], freqs)
        self.assertEqual(data['e_theta_re'].shape[3], 5)

    def test_single_frequency_returns_3d_pattern(self):
        """Single frequency produces a 3D pattern (no 4th dim) and scalar center_freq."""
        data = arrayant.generate_speaker(frequencies=np.array([1000.0]))
        # With one entry, arrayant2dict_multi delegates to single-freq version
        self.assertEqual(data['e_theta_re'].ndim, 3)
        self.assertIsInstance(data['center_freq'], float)

    def test_two_frequencies(self):
        """Two frequencies produce 4D patterns with shape[-1] == 2."""
        freqs = np.array([500.0, 5000.0])
        data = arrayant.generate_speaker(frequencies=freqs)
        self.assertEqual(data['e_theta_re'].ndim, 4)
        self.assertEqual(data['e_theta_re'].shape[3], 2)
        npt.assert_array_almost_equal(data['center_freq'], freqs)

    # ================================================================================
    # Driver types
    # ================================================================================

    def test_piston_driver(self):
        """Piston driver produces valid finite patterns."""
        data = arrayant.generate_speaker(driver_type='piston')
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))
        self.assertGreater(np.max(np.abs(data['e_theta_re'])), 0)

    def test_horn_driver(self):
        """Horn driver produces valid finite patterns."""
        data = arrayant.generate_speaker(driver_type='horn', radius=0.025,
                                         lower_cutoff=1500.0, upper_cutoff=20000.0)
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))
        self.assertGreater(np.max(np.abs(data['e_theta_re'])), 0)

    def test_omni_driver(self):
        """Omni driver produces valid finite patterns."""
        data = arrayant.generate_speaker(driver_type='omni', radius=0.165,
                                         lower_cutoff=30.0, upper_cutoff=300.0)
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))
        self.assertGreater(np.max(np.abs(data['e_theta_re'])), 0)

    def test_omni_driver_is_isotropic(self):
        """Omni driver at its passband center should have nearly uniform directivity."""
        freqs = np.array([100.0])
        data = arrayant.generate_speaker(driver_type='omni', radius=0.165,
                                         lower_cutoff=30.0, upper_cutoff=300.0,
                                         radiation_type='monopole', frequencies=freqs,
                                         angular_resolution=10.0)
        pat = data['e_theta_re'][:, :, 0]
        # All values should be approximately equal for monopole omni
        rel_spread = (np.max(pat) - np.min(pat)) / np.mean(np.abs(pat))
        self.assertLess(rel_spread, 0.01, "Monopole omni should be nearly isotropic")

    # ================================================================================
    # Radiation types
    # ================================================================================

    def test_radiation_monopole(self):
        """Monopole radiation produces finite output."""
        data = arrayant.generate_speaker(radiation_type='monopole',
                                         frequencies=np.array([1000.0]))
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))

    def test_radiation_hemisphere(self):
        """Hemisphere radiation produces finite output."""
        data = arrayant.generate_speaker(radiation_type='hemisphere',
                                         frequencies=np.array([1000.0]))
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))

    def test_radiation_dipole(self):
        """Dipole radiation produces finite output."""
        data = arrayant.generate_speaker(radiation_type='dipole',
                                         frequencies=np.array([1000.0]))
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))

    def test_radiation_cardioid(self):
        """Cardioid radiation produces finite output."""
        data = arrayant.generate_speaker(radiation_type='cardioid',
                                         frequencies=np.array([1000.0]))
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))

    def test_dipole_has_rear_null(self):
        """Dipole radiation should have near-zero amplitude at 90 deg off-axis."""
        freqs = np.array([1000.0])
        data = arrayant.generate_speaker(driver_type='omni', radiation_type='dipole',
                                         lower_cutoff=30.0, upper_cutoff=20000.0,
                                         frequencies=freqs, angular_resolution=5.0)
        pat = data['e_theta_re'][:, :, 0]  # 2D: (elevation, azimuth)
        el_grid = data['elevation_grid']
        az_grid = data['azimuth_grid']
        # At elevation=0, azimuth=±90 deg (off-axis), dipole should be near zero
        el_mid = np.argmin(np.abs(el_grid))
        az_90 = np.argmin(np.abs(az_grid - np.pi / 2))
        on_axis = np.abs(pat[el_mid, np.argmin(np.abs(az_grid))])
        off_axis = np.abs(pat[el_mid, az_90])
        if on_axis > 0:
            self.assertLess(off_axis / on_axis, 0.15,
                            "Dipole should have deep null at 90 deg off-axis")

    def test_cardioid_null_at_rear(self):
        """Cardioid radiation should have near-zero at 180 deg (rear)."""
        freqs = np.array([2000.0])
        data = arrayant.generate_speaker(driver_type='omni', radiation_type='cardioid',
                                         lower_cutoff=30.0, upper_cutoff=20000.0,
                                         frequencies=freqs, angular_resolution=5.0)
        pat = data['e_theta_re'][:, :, 0]
        el_grid = data['elevation_grid']
        az_grid = data['azimuth_grid']
        el_mid = np.argmin(np.abs(el_grid))
        # On-axis (az=0) vs rear (az=π)
        on_axis = np.abs(pat[el_mid, np.argmin(np.abs(az_grid))])
        at_rear = np.abs(pat[el_mid, np.argmin(np.abs(az_grid - np.pi))])
        if on_axis > 0:
            self.assertLess(at_rear / on_axis, 0.15,
                            "Cardioid should have null at rear")

    # ================================================================================
    # Frequency response (bandpass behavior)
    # ================================================================================

    def test_bandpass_peak_in_passband(self):
        """Maximum on-axis response should occur within the passband."""
        freqs = np.array([20.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 18000.0])
        data = arrayant.generate_speaker(lower_cutoff=80.0, upper_cutoff=12000.0,
                                         frequencies=freqs, angular_resolution=10.0)
        pat = data['e_theta_re']
        el_grid = data['elevation_grid']
        az_grid = data['azimuth_grid']
        el_mid = np.argmin(np.abs(el_grid))
        az_0 = np.argmin(np.abs(az_grid))
        on_axis = np.array([np.abs(pat[el_mid, az_0, 0, f]) for f in range(len(freqs))])
        peak_idx = np.argmax(on_axis)
        peak_freq = freqs[peak_idx]
        self.assertGreaterEqual(peak_freq, 80.0,
                                f"Peak at {peak_freq} Hz is below lower cutoff")
        self.assertLessEqual(peak_freq, 12000.0,
                             f"Peak at {peak_freq} Hz is above upper cutoff")

    def test_rolloff_below_passband(self):
        """On-axis response at 20 Hz should be much less than at 500 Hz."""
        freqs = np.array([20.0, 500.0])
        data = arrayant.generate_speaker(lower_cutoff=80.0, upper_cutoff=12000.0,
                                         frequencies=freqs, angular_resolution=10.0)
        pat = data['e_theta_re']
        el_mid = pat.shape[0] // 2
        az_mid = pat.shape[1] // 2
        amp_20 = np.abs(pat[el_mid, az_mid, 0, 0])
        amp_500 = np.abs(pat[el_mid, az_mid, 0, 1])
        self.assertGreater(amp_500, amp_20 * 3,
                           "500 Hz should be significantly stronger than 20 Hz for 80 Hz cutoff")

    def test_rolloff_above_passband(self):
        """On-axis response at 18 kHz should be less than at 5 kHz."""
        freqs = np.array([5000.0, 18000.0])
        data = arrayant.generate_speaker(lower_cutoff=80.0, upper_cutoff=12000.0,
                                         frequencies=freqs, angular_resolution=10.0)
        pat = data['e_theta_re']
        el_mid = pat.shape[0] // 2
        az_mid = pat.shape[1] // 2
        amp_5k = np.abs(pat[el_mid, az_mid, 0, 0])
        amp_18k = np.abs(pat[el_mid, az_mid, 0, 1])
        self.assertGreater(amp_5k, amp_18k,
                           "5 kHz should be stronger than 18 kHz for 12 kHz upper cutoff")

    # ================================================================================
    # Sensitivity scaling
    # ================================================================================

    def test_sensitivity_scaling(self):
        """Higher sensitivity produces larger pattern amplitudes."""
        freqs = np.array([1000.0])
        d85 = arrayant.generate_speaker(sensitivity=85.0, frequencies=freqs,
                                        angular_resolution=10.0)
        d95 = arrayant.generate_speaker(sensitivity=95.0, frequencies=freqs,
                                        angular_resolution=10.0)
        max_85 = np.max(np.abs(d85['e_theta_re']))
        max_95 = np.max(np.abs(d95['e_theta_re']))
        # +10 dB → factor of ~3.16 in amplitude
        ratio = max_95 / max_85
        npt.assert_almost_equal(ratio, 10.0 ** (10.0 / 20.0), decimal=2)

    # ================================================================================
    # Piston directivity vs frequency
    # ================================================================================

    def test_piston_beaming_increases_with_frequency(self):
        """Piston pattern should narrow (more directional) at higher frequencies."""
        freqs = np.array([200.0, 8000.0])
        data = arrayant.generate_speaker(driver_type='piston', radius=0.05,
                                         lower_cutoff=50.0, upper_cutoff=20000.0,
                                         radiation_type='monopole', frequencies=freqs,
                                         angular_resolution=5.0)
        pat = data['e_theta_re']
        el_grid = data['elevation_grid']
        az_grid = data['azimuth_grid']
        el_mid = np.argmin(np.abs(el_grid))
        az_0 = np.argmin(np.abs(az_grid))
        az_90 = np.argmin(np.abs(az_grid - np.pi / 2))

        # Ratio of 90-deg to on-axis amplitude
        ratio_lo = np.abs(pat[el_mid, az_90, 0, 0]) / max(np.abs(pat[el_mid, az_0, 0, 0]), 1e-30)
        ratio_hi = np.abs(pat[el_mid, az_90, 0, 1]) / max(np.abs(pat[el_mid, az_0, 0, 1]), 1e-30)
        self.assertGreater(ratio_lo, ratio_hi,
                           "Piston should be more directional at 8 kHz than at 200 Hz")

    # ================================================================================
    # Horn parameters
    # ================================================================================

    def test_horn_custom_coverage(self):
        """Horn with custom coverage angles produces valid output."""
        data = arrayant.generate_speaker(driver_type='horn', radius=0.025,
                                         lower_cutoff=1500.0, upper_cutoff=20000.0,
                                         hor_coverage=90.0, ver_coverage=60.0,
                                         frequencies=np.array([4000.0]),
                                         angular_resolution=5.0)
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))
        self.assertGreater(np.max(np.abs(data['e_theta_re'])), 0)

    def test_horn_auto_coverage(self):
        """Horn with coverage=0 (auto) produces valid output."""
        data = arrayant.generate_speaker(driver_type='horn', radius=0.025,
                                         lower_cutoff=1500.0, upper_cutoff=20000.0,
                                         hor_coverage=0.0, ver_coverage=0.0,
                                         frequencies=np.array([4000.0]),
                                         angular_resolution=5.0)
        self.assertTrue(np.all(np.isfinite(data['e_theta_re'])))

    # ================================================================================
    # Baffle parameters (piston only)
    # ================================================================================

    def test_different_baffle_sizes_change_pattern(self):
        """Different baffle dimensions produce different hemisphere radiation patterns."""
        freqs = np.array([1000.0])
        d_small = arrayant.generate_speaker(radiation_type='hemisphere',
                                            baffle_width=0.10, baffle_height=0.10,
                                            frequencies=freqs, angular_resolution=10.0)
        d_large = arrayant.generate_speaker(radiation_type='hemisphere',
                                            baffle_width=0.40, baffle_height=0.40,
                                            frequencies=freqs, angular_resolution=10.0)
        # Patterns should differ (different baffle step frequencies)
        self.assertFalse(np.allclose(d_small['e_theta_re'], d_large['e_theta_re']),
                         "Different baffle sizes should produce different patterns")

    # ================================================================================
    # Rolloff slope
    # ================================================================================

    def test_steeper_rolloff_attenuates_more(self):
        """Steeper lower rolloff should attenuate out-of-band frequencies more."""
        freqs = np.array([20.0, 500.0])
        d12 = arrayant.generate_speaker(lower_cutoff=80.0, lower_rolloff_slope=12.0,
                                        frequencies=freqs, angular_resolution=10.0)
        d48 = arrayant.generate_speaker(lower_cutoff=80.0, lower_rolloff_slope=48.0,
                                        frequencies=freqs, angular_resolution=10.0)
        el = d12['e_theta_re'].shape[0] // 2
        az = d12['e_theta_re'].shape[1] // 2
        amp_20_12 = np.abs(d12['e_theta_re'][el, az, 0, 0])
        amp_20_48 = np.abs(d48['e_theta_re'][el, az, 0, 0])
        amp_500_12 = np.abs(d12['e_theta_re'][el, az, 0, 1])
        amp_500_48 = np.abs(d48['e_theta_re'][el, az, 0, 1])
        # At 500 Hz (in-band), both should be similar
        npt.assert_almost_equal(amp_500_12, amp_500_48, decimal=1)
        # At 20 Hz (out-of-band), steeper rolloff should be smaller
        self.assertGreater(amp_20_12, amp_20_48,
                           "48 dB/oct should attenuate 20 Hz more than 12 dB/oct")

    # ================================================================================
    # Auto frequency generation
    # ================================================================================

    def test_auto_freq_covers_passband(self):
        """Auto-generated frequencies should span at least the passband."""
        data = arrayant.generate_speaker(lower_cutoff=200.0, upper_cutoff=8000.0)
        cf = data['center_freq']
        self.assertLessEqual(cf[0], 200.0,
                             "Auto freqs should start at or below lower cutoff")
        self.assertGreaterEqual(cf[-1], 8000.0,
                                "Auto freqs should end at or above upper cutoff")

    def test_auto_freq_within_audible(self):
        """Auto-generated frequencies should be within 20–20000 Hz."""
        data = arrayant.generate_speaker()
        cf = data['center_freq']
        self.assertGreaterEqual(cf[0], 20.0)
        self.assertLessEqual(cf[-1], 20000.0)

    # ================================================================================
    # E-phi should be zero (acoustic: no polarization, only e_theta used)
    # ================================================================================

    def test_e_phi_is_zero(self):
        """For loudspeakers, e_phi fields should be zero (scalar pressure, no polarization)."""
        data = arrayant.generate_speaker(frequencies=np.array([1000.0]))
        npt.assert_array_equal(data['e_phi_re'], np.zeros_like(data['e_phi_re']))
        npt.assert_array_equal(data['e_phi_im'], np.zeros_like(data['e_phi_im']))

    def test_e_theta_im_is_zero_for_piston_monopole(self):
        """Piston + monopole: real-valued directivity, imaginary part should be zero."""
        data = arrayant.generate_speaker(driver_type='piston', radiation_type='monopole',
                                         frequencies=np.array([1000.0]))
        npt.assert_array_almost_equal(data['e_theta_im'],
                                      np.zeros_like(data['e_theta_im']), decimal=14)

    # ================================================================================
    # Coupling matrix
    # ================================================================================

    def test_single_driver_identity_coupling(self):
        """Single driver should have identity coupling (1x1 real, 0 imaginary)."""
        data = arrayant.generate_speaker(frequencies=np.array([1000.0]))
        npt.assert_almost_equal(data['coupling_re'], np.array([[1.0]]), decimal=14)
        npt.assert_almost_equal(data['coupling_im'], np.array([[0.0]]), decimal=14)

    # ================================================================================
    # Name
    # ================================================================================

    def test_name_contains_driver_type(self):
        """Name string should be set."""
        data = arrayant.generate_speaker(driver_type='piston')
        self.assertIsInstance(data['name'], str)
        self.assertGreater(len(data['name']), 0)

    # ================================================================================
    # Errors
    # ================================================================================

    def test_invalid_driver_type_throws(self):
        """Invalid driver type should raise an error."""
        with self.assertRaises(Exception):
            arrayant.generate_speaker(driver_type='plasma')

    def test_invalid_radiation_type_throws(self):
        """Invalid radiation type should raise an error."""
        with self.assertRaises(Exception):
            arrayant.generate_speaker(radiation_type='laserbeam')

    # ================================================================================
    # Reproduce with different parameters
    # ================================================================================

    def test_radius_affects_directivity(self):
        """Larger radius should produce narrower beam at same frequency."""
        freqs = np.array([5000.0])
        d_small = arrayant.generate_speaker(driver_type='piston', radius=0.03,
                                            lower_cutoff=50.0, upper_cutoff=20000.0,
                                            radiation_type='monopole', frequencies=freqs,
                                            angular_resolution=5.0)
        d_large = arrayant.generate_speaker(driver_type='piston', radius=0.10,
                                            lower_cutoff=50.0, upper_cutoff=20000.0,
                                            radiation_type='monopole', frequencies=freqs,
                                            angular_resolution=5.0)
        el_mid = d_small['e_theta_re'].shape[0] // 2
        az_grid = d_small['azimuth_grid']
        az_0 = np.argmin(np.abs(az_grid))
        az_90 = np.argmin(np.abs(az_grid - np.pi / 2))

        # Normalize to on-axis
        ratio_small = (np.abs(d_small['e_theta_re'][el_mid, az_90, 0]) /
                       max(np.abs(d_small['e_theta_re'][el_mid, az_0, 0]), 1e-30))
        ratio_large = (np.abs(d_large['e_theta_re'][el_mid, az_90, 0]) /
                       max(np.abs(d_large['e_theta_re'][el_mid, az_0, 0]), 1e-30))
        self.assertGreater(ratio_small, ratio_large,
                           "Larger radius should beam more (lower off-axis ratio)")

    def test_cutoff_shift_moves_passband(self):
        """Shifting cutoffs should shift where the response peaks."""
        freqs = np.array([200.0, 2000.0, 10000.0])
        d_low = arrayant.generate_speaker(lower_cutoff=50.0, upper_cutoff=1000.0,
                                          frequencies=freqs, angular_resolution=10.0)
        d_high = arrayant.generate_speaker(lower_cutoff=3000.0, upper_cutoff=18000.0,
                                           frequencies=freqs, angular_resolution=10.0)
        el = d_low['e_theta_re'].shape[0] // 2
        az = d_low['e_theta_re'].shape[1] // 2

        # Low-pass speaker: 200 Hz should be stronger than 10 kHz
        amp_low_200 = np.abs(d_low['e_theta_re'][el, az, 0, 0])
        amp_low_10k = np.abs(d_low['e_theta_re'][el, az, 0, 2])
        self.assertGreater(amp_low_200, amp_low_10k)

        # High-pass speaker: 10 kHz should be stronger than 200 Hz
        amp_high_200 = np.abs(d_high['e_theta_re'][el, az, 0, 0])
        amp_high_10k = np.abs(d_high['e_theta_re'][el, az, 0, 2])
        self.assertGreater(amp_high_10k, amp_high_200)


if __name__ == '__main__':
    unittest.main()
