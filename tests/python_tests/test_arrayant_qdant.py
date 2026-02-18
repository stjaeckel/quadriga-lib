import sys
import os
import unittest
import tempfile
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import arrayant


def _temp_qdant():
    """Return a temporary .qdant file path that does NOT yet exist on disk.
    qdant_write expects either a non-existent file (to create) or a valid QDANT XML file.
    An empty file left by NamedTemporaryFile would cause 'No document element found'."""
    fd, fn = tempfile.mkstemp(suffix='.qdant')
    os.close(fd)
    if os.path.exists(fn): os.unlink(fn)  # Remove the empty file; qdant_write will create it fresh
    return fn


def make_dipole_single(res=10.0, freq=1e9):
    """Helper: create a single-frequency dipole antenna."""
    return arrayant.generate('dipole', res, freq=freq)


def make_custom_single(az=30.0, el=30.0, res=10.0, freq=2.4e9):
    """Helper: create a single-frequency custom beam antenna."""
    return arrayant.generate('custom', res, az_3dB=az, el_3dB=el, freq=freq)


def make_xpol_single(res=10.0, freq=1e9):
    """Helper: create a single-frequency cross-polarized antenna (2 elements)."""
    return arrayant.generate('xpol', res, freq=freq)


def make_dipole_multi(n_freq=3, res=10.0):
    """Helper: create a multi-frequency dipole-like antenna with 4D patterns."""
    ant = arrayant.generate('dipole', res)
    n_el, n_az, n_elem = ant['e_theta_re'].shape
    freqs = np.array([1000.0 * (i + 1) for i in range(n_freq)])

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


def make_multi_freq_xpol(n_freq=2, res=10.0):
    """Helper: create a multi-frequency xpol antenna (2 elements) with 4D patterns."""
    ant = arrayant.generate('xpol', res)
    n_el, n_az, n_elem = ant['e_theta_re'].shape
    freqs = np.array([500.0 * (i + 1) for i in range(n_freq)])

    e_theta_re = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_theta_im = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_phi_re = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    e_phi_im = np.zeros((n_el, n_az, n_elem, n_freq), order='F')
    for f in range(n_freq):
        # Scale pattern slightly per frequency so entries are distinguishable
        scale = 1.0 + 0.1 * f
        e_theta_re[:, :, :, f] = ant['e_theta_re'] * scale
        e_theta_im[:, :, :, f] = ant['e_theta_im'] * scale
        e_phi_re[:, :, :, f] = ant['e_phi_re'] * scale
        e_phi_im[:, :, :, f] = ant['e_phi_im'] * scale

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
        'name': 'xpol_multi',
    }


class TestQdantWriteReadSingleFreq(unittest.TestCase):
    """Tests for qdant_write and qdant_read with single-frequency (3D) arrayants."""

    # ================================================================================
    # Basic round-trip
    # ================================================================================

    def test_dipole_round_trip(self):
        """Write and read a dipole antenna; pattern data should survive the round-trip."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_theta_im'], ant['e_theta_im'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_re'], ant['e_phi_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_im'], ant['e_phi_im'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_custom_beam_round_trip(self):
        """Write and read a custom beam antenna; verify pattern shape and values."""
        ant = make_custom_single(az=30.0, el=60.0, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=4)
            self.assertEqual(out['e_theta_re'].shape, ant['e_theta_re'].shape)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_xpol_round_trip(self):
        """Write and read a 2-element xpol antenna; both elements should round-trip."""
        ant = make_xpol_single(res=10.0)
        self.assertEqual(ant['e_theta_re'].shape[2], 2)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_re'], ant['e_phi_re'], decimal=5)
            self.assertEqual(out['e_theta_re'].shape[2], 2)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Grid and metadata preservation
    # ================================================================================

    def test_grids_preserved(self):
        """Azimuth and elevation grids should survive the round-trip."""
        ant = make_dipole_single(res=5.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'], decimal=10)
            npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_center_freq_preserved(self):
        """Center frequency should survive the round-trip."""
        ant = make_dipole_single(freq=3.5e9)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_almost_equal(out['center_freq'], 3.5e9, decimal=0)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_element_pos_preserved(self):
        """Element positions should survive the round-trip."""
        ant = make_xpol_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['element_pos'], ant['element_pos'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_coupling_preserved(self):
        """Coupling matrices should survive the round-trip."""
        ant = make_xpol_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['coupling_re'], ant['coupling_re'], decimal=10)
            npt.assert_array_almost_equal(out['coupling_im'], ant['coupling_im'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_name_preserved(self):
        """Antenna name should survive the round-trip."""
        ant = make_dipole_single()
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            self.assertEqual(out['name'], ant['name'])
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_3d(self):
        """Single-freq read should produce 3D pattern arrays."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            self.assertEqual(out['e_theta_re'].ndim, 3)
            self.assertEqual(out['e_phi_re'].ndim, 3)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_output_has_all_keys(self):
        """Output should contain all expected keys."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im',
                        'azimuth_grid', 'elevation_grid', 'element_pos',
                        'coupling_re', 'coupling_im', 'center_freq', 'name', 'layout']:
                self.assertIn(key, out)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_layout_returned(self):
        """Layout should be returned as a numpy array."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            self.assertIsInstance(out['layout'], np.ndarray)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # ID parameter
    # ================================================================================

    def test_write_returns_id(self):
        """qdant_write should return the ID of the written entry."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            id_out = arrayant.qdant_write(fn, ant)
            self.assertIsInstance(id_out, int)
            self.assertGreaterEqual(id_out, 1)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_write_with_explicit_id(self):
        """Writing with an explicit ID and reading back with that ID should work."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant, id=3)
            out = arrayant.qdant_read(fn, id=3)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_default_id_reads_first(self):
        """Default read (id=1) should return the first entry."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant, id=1)
            out = arrayant.qdant_read(fn)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Fortran contiguity
    # ================================================================================

    def test_output_is_fortran_contiguous(self):
        """Pattern arrays from qdant_read should be Fortran-contiguous."""
        ant = make_dipole_single(res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn)
            self.assertTrue(out['e_theta_re'].flags.f_contiguous)
        finally:
            if os.path.exists(fn): os.unlink(fn)


class TestQdantWriteReadMultiFreq(unittest.TestCase):
    """Tests for qdant_write and qdant_read with multi-frequency (4D) arrayants."""

    # ================================================================================
    # Basic round-trip
    # ================================================================================

    def test_dipole_multi_round_trip(self):
        """Write and read a multi-freq dipole; 4D pattern data should survive the round-trip."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_theta_im'], ant['e_theta_im'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_re'], ant['e_phi_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_im'], ant['e_phi_im'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_xpol_multi_round_trip(self):
        """Write and read a multi-freq 2-element xpol; scaled patterns should round-trip."""
        ant = make_multi_freq_xpol(n_freq=4, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
            npt.assert_array_almost_equal(out['e_phi_re'], ant['e_phi_re'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_single_freq_as_multi_round_trip(self):
        """A single-freq pattern stored as 4D (n_freq=1) should round-trip correctly."""
        ant = make_dipole_multi(n_freq=1, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            # Read back as single (id=1) since n_freq=1 writes one entry
            out = arrayant.qdant_read(fn, id=1)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'][:, :, :, 0], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Output structure
    # ================================================================================

    def test_output_is_4d(self):
        """Multi-freq read (id=0) should produce 4D pattern arrays."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['e_theta_re'].ndim, 4)
            self.assertEqual(out['e_phi_re'].ndim, 4)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_output_shape_matches_input(self):
        """Output shape should match input shape for all pattern fields."""
        ant = make_multi_freq_xpol(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['e_theta_re'].shape, ant['e_theta_re'].shape)
            self.assertEqual(out['e_theta_im'].shape, ant['e_theta_im'].shape)
            self.assertEqual(out['e_phi_re'].shape, ant['e_phi_re'].shape)
            self.assertEqual(out['e_phi_im'].shape, ant['e_phi_im'].shape)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_n_freq_dimension(self):
        """4th dimension should equal the number of frequency entries."""
        n_freq = 5
        ant = make_dipole_multi(n_freq=n_freq, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['e_theta_re'].shape[3], n_freq)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_output_has_all_keys(self):
        """Output should contain all expected keys."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            for key in ['e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im',
                        'azimuth_grid', 'elevation_grid', 'element_pos',
                        'coupling_re', 'coupling_im', 'center_freq', 'name', 'layout']:
                self.assertIn(key, out)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Grid and metadata preservation
    # ================================================================================

    def test_grids_preserved(self):
        """Azimuth and elevation grids should survive the multi-freq round-trip."""
        ant = make_dipole_multi(n_freq=3, res=5.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            npt.assert_array_almost_equal(out['azimuth_grid'], ant['azimuth_grid'], decimal=10)
            npt.assert_array_almost_equal(out['elevation_grid'], ant['elevation_grid'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_center_freq_preserved(self):
        """Center frequencies (1D array) should survive the round-trip."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            npt.assert_array_almost_equal(out['center_freq'], ant['center_freq'], decimal=0)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_center_freq_is_1d_array(self):
        """For multi-freq output, center_freq should be a 1D array."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['center_freq'].ndim, 1)
            self.assertEqual(len(out['center_freq']), 3)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_element_pos_preserved(self):
        """Element positions should survive the multi-freq round-trip."""
        ant = make_multi_freq_xpol(n_freq=2, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            npt.assert_array_almost_equal(out['element_pos'], ant['element_pos'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_name_preserved(self):
        """Antenna name should survive the multi-freq round-trip."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['name'], 'dipole_multi')
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_layout_returned(self):
        """Layout should be returned for multi-freq reads."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertIn('layout', out)
            self.assertIsInstance(out['layout'], np.ndarray)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Write return value
    # ================================================================================

    def test_write_returns_zero_for_multi(self):
        """qdant_write with 4D input should return 0."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        fn = _temp_qdant()
        try:
            id_out = arrayant.qdant_write(fn, ant)
            self.assertEqual(id_out, 0)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Per-frequency data integrity
    # ================================================================================

    def test_per_frequency_data_distinct(self):
        """Scaled per-frequency patterns should remain distinct after round-trip."""
        ant = make_multi_freq_xpol(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            # Freq 0 and freq 2 have different scaling (1.0 vs 1.2), should not be equal
            self.assertFalse(np.allclose(out['e_theta_re'][:, :, 0, 0],
                                         out['e_theta_re'][:, :, 0, 2], atol=1e-6))
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_individual_freq_matches_single_read(self):
        """Each freq entry read with id=0 should match reading that entry individually."""
        ant = make_multi_freq_xpol(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out_multi = arrayant.qdant_read(fn, id=0)
            for f_idx in range(3):
                out_single = arrayant.qdant_read(fn, id=f_idx + 1)
                npt.assert_array_almost_equal(
                    out_multi['e_theta_re'][:, :, :, f_idx],
                    out_single['e_theta_re'], decimal=10)
                npt.assert_array_almost_equal(
                    out_multi['e_phi_re'][:, :, :, f_idx],
                    out_single['e_phi_re'], decimal=10)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Cross-path compatibility
    # ================================================================================

    def test_multi_write_single_read(self):
        """Entries written via multi-freq path should be individually readable."""
        ant = make_dipole_multi(n_freq=3, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            for f_idx in range(3):
                out = arrayant.qdant_read(fn, id=f_idx + 1)
                self.assertEqual(out['e_theta_re'].ndim, 3)
                npt.assert_array_almost_equal(
                    out['e_theta_re'], ant['e_theta_re'][:, :, :, f_idx], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Integration with generate_speaker
    # ================================================================================

    def test_speaker_round_trip(self):
        """A generated speaker (multi-freq) should survive write/read round-trip."""
        speaker = arrayant.generate_speaker(frequencies=np.array([500.0, 2000.0]),
                                            angular_resolution=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, speaker)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['e_theta_re'].shape, speaker['e_theta_re'].shape)
            npt.assert_array_almost_equal(out['e_theta_re'], speaker['e_theta_re'], decimal=5)
            npt.assert_array_almost_equal(out['center_freq'], speaker['center_freq'], decimal=0)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    # ================================================================================
    # Edge cases
    # ================================================================================

    def test_two_frequencies(self):
        """Minimal multi-freq case (n_freq=2) should work."""
        ant = make_dipole_multi(n_freq=2, res=10.0)
        fn = _temp_qdant()
        try:
            arrayant.qdant_write(fn, ant)
            out = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out['e_theta_re'].shape[3], 2)
            npt.assert_array_almost_equal(out['e_theta_re'], ant['e_theta_re'], decimal=5)
        finally:
            if os.path.exists(fn): os.unlink(fn)

    def test_overwrite_existing_file(self):
        """Writing multi-freq to an existing file should overwrite it."""
        ant1 = make_dipole_multi(n_freq=2, res=10.0)
        ant2 = make_dipole_multi(n_freq=4, res=10.0)
        fn = _temp_qdant()
        try:
            # Write first version
            arrayant.qdant_write(fn, ant1)
            out1 = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out1['e_theta_re'].shape[3], 2)

            # Overwrite with second version (different n_freq)
            arrayant.qdant_write(fn, ant2)
            out2 = arrayant.qdant_read(fn, id=0)
            self.assertEqual(out2['e_theta_re'].shape[3], 4)
        finally:
            if os.path.exists(fn): os.unlink(fn)


if __name__ == '__main__':
    unittest.main()