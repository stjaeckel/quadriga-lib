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


def make_single_freq_ant(n_el=5, n_az=9, n_elem=1, freq=1e9):
    """Helper: create a minimal single-frequency arrayant dict."""
    el_grid = np.linspace(-np.pi / 2, np.pi / 2, n_el)
    az_grid = np.linspace(-np.pi, np.pi, n_az)

    # Each element gets a distinct constant pattern for easy identification
    e_theta_re = np.zeros((n_el, n_az, n_elem), order='F')
    for i in range(n_elem):
        e_theta_re[:, :, i] = float(i + 1)

    return {
        'e_theta_re': e_theta_re,
        'e_theta_im': np.zeros((n_el, n_az, n_elem), order='F'),
        'e_phi_re': np.zeros((n_el, n_az, n_elem), order='F'),
        'e_phi_im': np.zeros((n_el, n_az, n_elem), order='F'),
        'azimuth_grid': az_grid,
        'elevation_grid': el_grid,
        'element_pos': np.zeros((3, n_elem), order='F'),
        'coupling_re': np.eye(n_elem, order='F'),
        'coupling_im': np.zeros((n_elem, n_elem), order='F'),
        'center_freq': freq,
        'name': 'test_ant',
    }


def make_multi_freq_ant(n_el=5, n_az=9, n_elem=1, n_freq=3):
    """Helper: create a minimal multi-frequency arrayant dict with 4D patterns."""
    el_grid = np.linspace(-np.pi / 2, np.pi / 2, n_el)
    az_grid = np.linspace(-np.pi, np.pi, n_az)
    freqs = np.array([1000.0 * (i + 1) for i in range(n_freq)])

    # Each element and frequency gets a distinct constant pattern
    e_theta_re = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    for f in range(n_freq):
        for i in range(n_elem):
            e_theta_re[:, :, i, f] = float((f + 1) * 10 + (i + 1))

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


class TestCopyElementSingleFreq(unittest.TestCase):
    """Tests for copy_element with single-frequency (3D) arrayants."""

    # ================================================================================
    # Basic copy operations
    # ================================================================================

    def test_copy_single_to_new_index(self):
        """Copy element 0 to index 1 — should expand from 1 to 2 elements."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 2)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0], out['e_theta_re'][:, :, 1])

    def test_copy_preserves_pattern_values(self):
        """Copied element should have identical pattern to source."""
        ant = make_single_freq_ant(n_elem=2)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([2], dtype=np.uint64))
        # Element 2 should be a copy of element 0 (value = 1.0)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 2],
                                      np.ones((5, 9)), decimal=14)
        # Element 1 should be unchanged (value = 2.0)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 1],
                                      np.full((5, 9), 2.0), decimal=14)

    def test_copy_to_existing_index_overwrites(self):
        """Copy element 0 to existing element 1 — overwrites element 1."""
        ant = make_single_freq_ant(n_elem=2)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        # Both should now have value 1.0 (element 0's pattern)
        self.assertEqual(out['e_theta_re'].shape[2], 2)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0],
                                      out['e_theta_re'][:, :, 1])

    def test_copy_one_source_to_multiple_destinations(self):
        """Copy one source to multiple destinations."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1, 2, 3], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 4)
        for i in range(4):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, i],
                                          np.ones((5, 9)), decimal=14)

    def test_copy_multiple_sources_to_multiple_destinations(self):
        """Copy [0,1] to [2,3] — paired copy."""
        ant = make_single_freq_ant(n_elem=2)
        out = arrayant.copy_element(ant, source_element=np.array([0, 1], dtype=np.uint64),
                                    dest_element=np.array([2, 3], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 4)
        # Element 2 = copy of 0 (value 1.0), element 3 = copy of 1 (value 2.0)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 2],
                                      np.ones((5, 9)), decimal=14)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 3],
                                      np.full((5, 9), 2.0), decimal=14)

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_3d(self):
        """Single-freq input produces 3D output patterns."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].ndim, 3)

    def test_output_has_all_keys(self):
        """Output dict contains all expected keys."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im',
                     'azimuth_grid', 'elevation_grid', 'element_pos',
                     'coupling_re', 'coupling_im', 'center_freq', 'name']:
            self.assertIn(key, out)

    def test_grids_unchanged(self):
        """Azimuth and elevation grids should be unchanged after copy."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'])
        npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'])

    def test_element_pos_expanded(self):
        """element_pos should expand to match new element count."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['element_pos'].shape, (3, 2))

    def test_coupling_expanded(self):
        """Coupling matrix should expand to match new element count."""
        ant = make_single_freq_ant(n_elem=1)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['coupling_re'].shape[0], 2)

    def test_input_not_modified(self):
        """Input dict should not be modified (copy semantics)."""
        ant = make_single_freq_ant(n_elem=1)
        original_shape = ant['e_theta_re'].shape
        _ = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                  dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(ant['e_theta_re'].shape, original_shape)

    # ================================================================================
    # All pattern fields copied
    # ================================================================================

    def test_all_pattern_fields_copied(self):
        """All four pattern fields should have the copied element."""
        ant = make_single_freq_ant(n_elem=1)
        # Set distinct values in all fields
        ant['e_theta_re'][:, :, 0] = 1.0
        ant['e_theta_im'][:, :, 0] = 2.0
        ant['e_phi_re'][:, :, 0] = 3.0
        ant['e_phi_im'][:, :, 0] = 4.0
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 1], 1.0)
        npt.assert_array_almost_equal(out['e_theta_im'][:, :, 1], 2.0)
        npt.assert_array_almost_equal(out['e_phi_re'][:, :, 1], 3.0)
        npt.assert_array_almost_equal(out['e_phi_im'][:, :, 1], 4.0)

    # ================================================================================
    # Error handling
    # ================================================================================

    def test_mismatched_source_dest_length_throws(self):
        """source and dest vectors of different lengths should throw."""
        ant = make_single_freq_ant(n_elem=3)
        with self.assertRaises(Exception):
            arrayant.copy_element(ant, source_element=np.array([0, 1], dtype=np.uint64),
                                  dest_element=np.array([2, 3, 4], dtype=np.uint64))

    def test_source_out_of_bounds_throws(self):
        """Source index beyond element count should throw."""
        ant = make_single_freq_ant(n_elem=1)
        with self.assertRaises(Exception):
            arrayant.copy_element(ant, source_element=np.array([5], dtype=np.uint64),
                                  dest_element=np.array([1], dtype=np.uint64))

    # ================================================================================
    # Integration with generate
    # ================================================================================

    def test_copy_on_generated_antenna(self):
        """Copy element on a generated omni antenna."""
        ant = arrayant.generate('omni', 10)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1, 2], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 3)
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0],
                                      out['e_theta_re'][:, :, 1])
        npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0],
                                      out['e_theta_re'][:, :, 2])


class TestCopyElementMultiFreq(unittest.TestCase):
    """Tests for copy_element with multi-frequency (4D) arrayants."""

    # ================================================================================
    # Basic copy operations
    # ================================================================================

    def test_copy_single_to_new_index(self):
        """Copy element 0 to index 1 — should expand from 1 to 2 elements across all freqs."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].ndim, 4)
        self.assertEqual(out['e_theta_re'].shape[2], 2)
        self.assertEqual(out['e_theta_re'].shape[3], 3)

    def test_copy_preserves_pattern_per_frequency(self):
        """Copied element should have correct per-frequency values."""
        ant = make_multi_freq_ant(n_elem=2, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([2], dtype=np.uint64))
        # Element 0 at freq 0 has value 11.0, freq 1 has 21.0, freq 2 has 31.0
        for f in range(3):
            expected = float((f + 1) * 10 + 1)
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 2, f],
                                          np.full((5, 9), expected), decimal=14)

    def test_copy_to_existing_overwrites_all_freqs(self):
        """Copy element 0 to element 1 — overwrites at all frequencies."""
        ant = make_multi_freq_ant(n_elem=2, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 2)
        for f in range(3):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0, f],
                                          out['e_theta_re'][:, :, 1, f])

    def test_copy_one_source_to_multiple_destinations(self):
        """Copy one source to multiple destinations across all frequencies."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=2)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1, 2], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 3)
        for f in range(2):
            for i in range(3):
                npt.assert_array_almost_equal(out['e_theta_re'][:, :, i, f],
                                              out['e_theta_re'][:, :, 0, f])

    def test_copy_multiple_sources_to_multiple_destinations(self):
        """Copy [0,1] to [2,3] — paired copy across all frequencies."""
        ant = make_multi_freq_ant(n_elem=2, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0, 1], dtype=np.uint64),
                                    dest_element=np.array([2, 3], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], 4)
        for f in range(3):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 2, f],
                                          out['e_theta_re'][:, :, 0, f])
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 3, f],
                                          out['e_theta_re'][:, :, 1, f])

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_4d(self):
        """Multi-freq input produces 4D output patterns."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].ndim, 4)

    def test_center_freq_preserved(self):
        """center_freq should be preserved as a 1D array."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        npt.assert_array_almost_equal(out['center_freq'], ant['center_freq'])

    def test_grids_unchanged(self):
        """Azimuth and elevation grids should be unchanged."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'])
        npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'])

    def test_element_pos_expanded(self):
        """element_pos should expand to match new element count."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['element_pos'].shape, (3, 2))

    def test_coupling_expanded(self):
        """Coupling matrix should expand for new elements."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        # Coupling is shared (2D) since all freq entries are identical
        self.assertGreaterEqual(out['coupling_re'].shape[0], 2)

    def test_name_preserved(self):
        """Name string should be preserved."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(out['name'], 'test_multi')

    def test_input_not_modified(self):
        """Input dict should not be modified (copy semantics)."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=3)
        original_shape = ant['e_theta_re'].shape
        _ = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                  dest_element=np.array([1], dtype=np.uint64))
        self.assertEqual(ant['e_theta_re'].shape, original_shape)

    # ================================================================================
    # All pattern fields copied
    # ================================================================================

    def test_all_pattern_fields_copied(self):
        """All four pattern fields should have the copied element at all frequencies."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=2)
        ant['e_theta_re'][:, :, 0, :] = 1.0
        ant['e_theta_im'][:, :, 0, :] = 2.0
        ant['e_phi_re'][:, :, 0, :] = 3.0
        ant['e_phi_im'][:, :, 0, :] = 4.0
        out = arrayant.copy_element(ant, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([1], dtype=np.uint64))
        for f in range(2):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 1, f], 1.0)
            npt.assert_array_almost_equal(out['e_theta_im'][:, :, 1, f], 2.0)
            npt.assert_array_almost_equal(out['e_phi_re'][:, :, 1, f], 3.0)
            npt.assert_array_almost_equal(out['e_phi_im'][:, :, 1, f], 4.0)

    # ================================================================================
    # Error handling
    # ================================================================================

    def test_mismatched_source_dest_length_throws(self):
        """source and dest vectors of different lengths should throw."""
        ant = make_multi_freq_ant(n_elem=3, n_freq=2)
        with self.assertRaises(Exception):
            arrayant.copy_element(ant, source_element=np.array([0, 1], dtype=np.uint64),
                                  dest_element=np.array([2, 3, 4], dtype=np.uint64))

    def test_source_out_of_bounds_throws(self):
        """Source index beyond element count should throw."""
        ant = make_multi_freq_ant(n_elem=1, n_freq=2)
        with self.assertRaises(Exception):
            arrayant.copy_element(ant, source_element=np.array([5], dtype=np.uint64),
                                  dest_element=np.array([1], dtype=np.uint64))

    # ================================================================================
    # Integration with generate_speaker
    # ================================================================================

    def test_copy_on_generated_speaker(self):
        """Copy element on a generated speaker (multi-freq)."""
        speaker = arrayant.generate_speaker(frequencies=np.array([500.0, 2000.0, 8000.0]),
                                            angular_resolution=10.0)
        n_elem_before = speaker['e_theta_re'].shape[2]
        out = arrayant.copy_element(speaker, source_element=np.array([0], dtype=np.uint64),
                                    dest_element=np.array([n_elem_before], dtype=np.uint64))
        self.assertEqual(out['e_theta_re'].shape[2], n_elem_before + 1)
        # Copied element should match source at all frequencies
        for f in range(3):
            npt.assert_array_almost_equal(out['e_theta_re'][:, :, 0, f],
                                          out['e_theta_re'][:, :, n_elem_before, f])


if __name__ == '__main__':
    unittest.main()
