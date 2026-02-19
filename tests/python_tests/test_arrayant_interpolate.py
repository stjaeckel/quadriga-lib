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


def make_single_freq_ant(scale=1.0, n_az=2, polarimetric=False):
    """Build a simple single-frequency arrayant dict with 1 element, 1 elevation sample."""
    az_grid = np.linspace(0, np.pi, n_az)
    values = np.array([scale * (1.0 - 2.0 * i / max(n_az - 1, 1)) for i in range(n_az)])
    ant = {
        "e_theta_re": values.reshape(1, n_az, 1),                # (n_el=1, n_az, n_elem=1)
        "e_theta_im": (0.5 * values).reshape(1, n_az, 1),
        "e_phi_re": np.zeros((1, n_az, 1)),
        "e_phi_im": np.zeros((1, n_az, 1)),
        "azimuth_grid": az_grid,
        "elevation_grid": np.array([0.0]),
    }
    if polarimetric:
        ant["e_phi_re"] = (-0.5 * values).reshape(1, n_az, 1)
        ant["e_phi_im"] = (0.25 * values).reshape(1, n_az, 1)
    return ant


def make_multi_freq_ant(center_freqs, polarimetric=False):
    """Build a multi-frequency arrayant dict (4D patterns). Scale grows with frequency index."""
    n_freq = len(center_freqs)
    # Each frequency entry has pattern scaled by (index+1)
    e_theta_re = np.zeros((1, 2, 1, n_freq))
    e_theta_im = np.zeros((1, 2, 1, n_freq))
    e_phi_re = np.zeros((1, 2, 1, n_freq))
    e_phi_im = np.zeros((1, 2, 1, n_freq))
    for i in range(n_freq):
        scale = float(i + 1)
        e_theta_re[0, :, 0, i] = [scale, -scale]
        e_theta_im[0, :, 0, i] = [0.5 * scale, -0.5 * scale]
        if polarimetric:
            e_phi_re[0, :, 0, i] = [-0.5 * scale, 0.5 * scale]
            e_phi_im[0, :, 0, i] = [0.25 * scale, -0.25 * scale]
    ant = {
        "e_theta_re": e_theta_re,
        "e_theta_im": e_theta_im,
        "e_phi_re": e_phi_re,
        "e_phi_im": e_phi_im,
        "azimuth_grid": np.array([0.0, np.pi]),
        "elevation_grid": np.array([0.0]),
        "center_freq": np.array(center_freqs, dtype=np.float64),
    }
    return ant


# ======================================================================================
# Single-frequency tests (backward compatibility with existing API)
# ======================================================================================

class TestSingleFreqBasic(unittest.TestCase):
    """Original single-frequency tests — must not regress."""

    def test_az_interpolation(self):
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        el = [[-0.5, 0, 0, 0.5]]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el)
        npt.assert_almost_equal(vr, [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(vi, [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(hr, [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(hi, [[6, 5, 4, 3]], decimal=14)

    def test_el_interpolation(self):
        ant = {"e_theta_re": [-2, 2], "e_theta_im": [-1, 1],
               "e_phi_re": [3, 1], "e_phi_im": [6, 2],
               "azimuth_grid": 0, "elevation_grid": [0, np.pi / 2]}
        az = [[-0.5, 0, 0, 0.5]]
        el = [[0, np.pi / 8, np.pi / 4, 3 * np.pi / 8]]
        res = arrayant.interpolate(ant, az, el, dist=1, local_angles=1)
        npt.assert_almost_equal(res[0], [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(res[1], [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(res[2], [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(res[3], [[6, 5, 4, 3]], decimal=14)
        npt.assert_almost_equal(res[4], [[0, 0, 0, 0]], decimal=14)
        npt.assert_almost_equal(res[5], az, decimal=14)
        npt.assert_almost_equal(res[6], el, decimal=14)

    def test_complex_output(self):
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        el = [[-0.5, 0, 0, 0.5]]
        v, h = arrayant.interpolate(ant, az, el, complex=1)
        npt.assert_almost_equal(v.real, [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(v.imag, [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(h.real, [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(h.imag, [[6, 5, 4, 3]], decimal=14)

    def test_polarization_rotation(self):
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[1, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        ori = [[np.pi / 4, -np.pi / 4], [0, 0], [0, 0]]
        vr, vi, hr, hi = arrayant.interpolate(ant, 0, 0, [0, 0], ori)
        rs2 = 1 / np.sqrt(2)
        npt.assert_almost_equal(vr[:, 0], [rs2, rs2], decimal=14)
        npt.assert_almost_equal(vi[:, 0], [rs2, rs2], decimal=14)
        npt.assert_almost_equal(hr[:, 0], [rs2, -rs2], decimal=14)
        npt.assert_almost_equal(hi[:, 0], [rs2, -rs2], decimal=14)

    def test_projected_distance(self):
        ant = {"e_theta_re": 1, "e_theta_im": 0,
               "e_phi_re": 0, "e_phi_im": 0,
               "azimuth_grid": 0, "elevation_grid": 0}
        _, _, _, _, ds = arrayant.interpolate(ant, 0, 0, [0, 0, 0], element_pos=np.eye(3), dist=1)
        npt.assert_almost_equal(ds[:, 0], [-1, 0, 0], decimal=14)

    def test_dist_and_local_angles_output_count(self):
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4]]
        el = [[0, 0]]
        # 4 field + 1 dist + 3 local = 8
        res = arrayant.interpolate(ant, az, el, dist=1, local_angles=1)
        self.assertEqual(len(res), 8)
        # Complex: 2 field + 1 dist = 3
        res = arrayant.interpolate(ant, az, el, complex=1, dist=1)
        self.assertEqual(len(res), 3)

    def test_fast_access(self):
        ant = arrayant.generate("3gpp", az_3dB=90, el_3dB=90, res=10, N=2)
        arrayant.interpolate(ant, 0, 0, fast_access=1)


# ======================================================================================
# Single-frequency + frequency duplication
# ======================================================================================

class TestSingleFreqWithFrequency(unittest.TestCase):
    """Single-frequency (3D) arrayant with `frequency` parameter → duplicated 3D output."""

    def test_single_angle_single_freq(self):
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        vr, vi, hr, hi = arrayant.interpolate(ant, [[0]], [[0]], frequency=[1000])
        self.assertEqual(vr.ndim, 3)
        self.assertEqual(vr.shape, (1, 1, 1))
        npt.assert_almost_equal(vr[0, 0, 0], -2, decimal=14)

    def test_multiple_freqs_duplicate(self):
        """Output must be identical across all frequency slices."""
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        el = [[-0.5, 0, 0, 0.5]]
        freqs = [100, 500, 1000, 5000, 20000]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)
        self.assertEqual(vr.shape, (1, 4, 5))
        for f in range(len(freqs)):
            npt.assert_almost_equal(vr[:, :, f], [[-2, -1, 0, 1]], decimal=14)
            npt.assert_almost_equal(vi[:, :, f], [[-1, -0.5, 0, 0.5]], decimal=14)
            npt.assert_almost_equal(hr[:, :, f], [[3, 2.5, 2, 1.5]], decimal=14)
            npt.assert_almost_equal(hi[:, :, f], [[6, 5, 4, 3]], decimal=14)

    def test_complex_output_with_freq(self):
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 2]]
        el = [[0, 0]]
        v, h = arrayant.interpolate(ant, az, el, frequency=[1000, 2000], complex=1)
        self.assertEqual(v.ndim, 3)
        self.assertEqual(v.shape, (1, 2, 2))
        npt.assert_almost_equal(v[:, :, 0], v[:, :, 1])
        npt.assert_almost_equal(h[:, :, 0], h[:, :, 1])

    def test_output_count_with_freq(self):
        """With frequency: always 4 (real) or 2 (complex) outputs, no dist/local_angles."""
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        res = arrayant.interpolate(ant, 0, 0, frequency=[1000])
        self.assertEqual(len(res), 4)
        res = arrayant.interpolate(ant, 0, 0, frequency=[1000], complex=1)
        self.assertEqual(len(res), 2)

    def test_dist_with_freq_throws(self):
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, 0, 0, frequency=[1000], dist=1)

    def test_local_angles_with_freq_throws(self):
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, 0, 0, frequency=[1000], local_angles=1)

    def test_multi_element_duplication(self):
        """Multiple elements with orientation, duplicated across freqs."""
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[1, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        ori = [[np.pi / 4, -np.pi / 4], [0, 0], [0, 0]]
        vr, vi, hr, hi = arrayant.interpolate(ant, 0, 0, [0, 0], ori, frequency=[100, 200, 300])
        self.assertEqual(vr.shape, (2, 1, 3))
        # All freq slices identical
        for f in range(3):
            npt.assert_almost_equal(vr[:, :, f], vr[:, :, 0], decimal=14)
            npt.assert_almost_equal(hr[:, :, f], hr[:, :, 0], decimal=14)

    def test_acoustic_freqs_duplication(self):
        """Acoustic frequency range: pattern duplicated identically."""
        ant = {"e_theta_re": [[1, -1]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        freqs = [20, 100, 500, 1000, 4000, 16000, 20000]  # Full audible range
        vr, vi, hr, hi = arrayant.interpolate(ant, [[0, np.pi / 2]], [[0, 0]], frequency=freqs)
        self.assertEqual(vr.shape, (1, 2, 7))
        for f in range(len(freqs)):
            npt.assert_almost_equal(vr[0, 0, f], 1.0, decimal=14)
            npt.assert_almost_equal(vr[0, 1, f], 0.0, decimal=14)


# ======================================================================================
# Multi-frequency tests (4D patterns)
# ======================================================================================

class TestMultiFreqBasic(unittest.TestCase):
    """Multi-frequency arrayant (4D patterns) with frequency interpolation."""

    def test_output_dimensions(self):
        ant = make_multi_freq_ant([500, 1000, 2000])
        az = np.zeros((1, 7))
        el = np.zeros((1, 7))
        freqs = [600, 1500, 1800, 3000]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)
        self.assertEqual(vr.shape, (1, 7, 4))
        self.assertEqual(vi.shape, (1, 7, 4))
        self.assertEqual(hr.shape, (1, 7, 4))
        self.assertEqual(hi.shape, (1, 7, 4))

    def test_complex_output_dimensions(self):
        ant = make_multi_freq_ant([500, 1000, 2000])
        v, h = arrayant.interpolate(ant, [[0]], [[0]], frequency=[600, 1800], complex=1)
        self.assertEqual(v.ndim, 3)
        self.assertEqual(v.shape, (1, 1, 2))
        self.assertTrue(np.iscomplexobj(v))
        self.assertTrue(np.iscomplexobj(h))

    def test_exact_frequency_match(self):
        """Querying at exact center frequencies must recover single-freq interpolation."""
        freqs = [500, 1000, 2000]
        ant = make_multi_freq_ant(freqs, polarimetric=True)
        az = [[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]]
        el = np.zeros((1, 4))

        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)

        # Compare each slice against single-freq extraction
        for f_idx, cf in enumerate(freqs):
            scale = float(f_idx + 1)
            single = {
                "e_theta_re": np.array([scale, -scale]).reshape(1, 2, 1),
                "e_theta_im": np.array([0.5 * scale, -0.5 * scale]).reshape(1, 2, 1),
                "e_phi_re": np.array([-0.5 * scale, 0.5 * scale]).reshape(1, 2, 1),
                "e_phi_im": np.array([0.25 * scale, -0.25 * scale]).reshape(1, 2, 1),
                "azimuth_grid": np.array([0.0, np.pi]),
                "elevation_grid": np.array([0.0]),
            }
            vr_s, vi_s, hr_s, hi_s = arrayant.interpolate(single, az, el)
            npt.assert_almost_equal(vr[:, :, f_idx], vr_s, decimal=12)
            npt.assert_almost_equal(vi[:, :, f_idx], vi_s, decimal=12)
            npt.assert_almost_equal(hr[:, :, f_idx], hr_s, decimal=12)
            npt.assert_almost_equal(hi[:, :, f_idx], hi_s, decimal=12)

    def test_extrapolation_below_clamps(self):
        ant = make_multi_freq_ant([500, 1000, 2000])
        vr_low, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[10])
        vr_exact, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[500])
        npt.assert_almost_equal(vr_low[0, 0, 0], vr_exact[0, 0, 0], decimal=12)

    def test_extrapolation_above_clamps(self):
        ant = make_multi_freq_ant([500, 1000, 2000])
        vr_high, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[99999])
        vr_exact, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[2000])
        npt.assert_almost_equal(vr_high[0, 0, 0], vr_exact[0, 0, 0], decimal=12)

    def test_single_entry_same_for_all_freqs(self):
        """Single-entry multi-freq vector always returns same result."""
        ant = make_multi_freq_ant([1000])
        freqs = [100, 1000, 50000]
        vr, vi, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=freqs)
        for s in range(3):
            npt.assert_almost_equal(vr[0, 0, s], 1.0, decimal=12)
            npt.assert_almost_equal(vi[0, 0, s], 0.5, decimal=12)

    def test_midpoint_in_phase_linear_blend(self):
        """Two in-phase entries → slerp = linear amplitude blend at midpoint."""
        e_theta_re = np.zeros((1, 2, 1, 2))
        e_theta_re[0, :, 0, 0] = 1.0   # freq 0: value 1.0 at both az
        e_theta_re[0, :, 0, 1] = 3.0   # freq 1: value 3.0 at both az
        ant = {
            "e_theta_re": e_theta_re,
            "e_theta_im": np.zeros((1, 2, 1, 2)),
            "e_phi_re": np.zeros((1, 2, 1, 2)),
            "e_phi_im": np.zeros((1, 2, 1, 2)),
            "azimuth_grid": np.array([0.0, np.pi]),
            "elevation_grid": np.array([0.0]),
            "center_freq": np.array([1000.0, 2000.0]),
        }
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[1500])
        npt.assert_almost_equal(vr[0, 0, 0], 2.0, decimal=10)

    def test_slerp_90deg_phase_shift(self):
        """Entry 0: (1,0), Entry 1: (0,1) → midpoint slerp → (cos(pi/4), sin(pi/4))."""
        ant = {
            "e_theta_re": np.array([[[[1.0, 0.0]]]]),   # (1,1,1,2)
            "e_theta_im": np.array([[[[0.0, 1.0]]]]),
            "e_phi_re": np.zeros((1, 1, 1, 2)),
            "e_phi_im": np.zeros((1, 1, 1, 2)),
            "azimuth_grid": np.array([0.0]),
            "elevation_grid": np.array([0.0]),
            "center_freq": np.array([1000.0, 2000.0]),
        }
        v, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[1500], complex=1)
        rs2 = 1 / np.sqrt(2)
        npt.assert_almost_equal(v[0, 0, 0].real, rs2, decimal=10)
        npt.assert_almost_equal(v[0, 0, 0].imag, rs2, decimal=10)

    def test_reversed_query_order(self):
        ant = make_multi_freq_ant([1000, 2000, 3000])
        fwd = [1200, 1800, 2500]
        rev = [2500, 1800, 1200]
        vr_f, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=fwd)
        vr_r, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=rev)
        npt.assert_almost_equal(vr_f[0, 0, 0], vr_r[0, 0, 2], decimal=12)
        npt.assert_almost_equal(vr_f[0, 0, 1], vr_r[0, 0, 1], decimal=12)
        npt.assert_almost_equal(vr_f[0, 0, 2], vr_r[0, 0, 0], decimal=12)

    def test_single_query_frequency(self):
        ant = make_multi_freq_ant([1000, 2000, 3000])
        vr, _, _, _ = arrayant.interpolate(ant, [[0, 1, 2]], np.zeros((1, 3)), frequency=[1500])
        self.assertEqual(vr.shape[2], 1)
        self.assertEqual(vr.shape[1], 3)


# ======================================================================================
# Multi-frequency with orientation and element selection
# ======================================================================================

class TestMultiFreqAdvanced(unittest.TestCase):

    def test_element_selection(self):
        """Duplicate element via i_element for multi-freq path."""
        ant = make_multi_freq_ant([1000])
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], element=[0, 0, 0], frequency=[1000])
        self.assertEqual(vr.shape[0], 3)
        npt.assert_almost_equal(vr[0, 0, 0], vr[1, 0, 0], decimal=14)
        npt.assert_almost_equal(vr[0, 0, 0], vr[2, 0, 0], decimal=14)

    def test_orientation_applied(self):
        """Orientation parameter should affect output for multi-freq path."""
        ant = make_multi_freq_ant([1000])
        ori_zero = np.array([[0], [0], [0]], dtype=np.float64)
        ori_rot = np.array([[0], [0], [np.pi / 4]], dtype=np.float64)
        vr0, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], orientation=ori_zero, frequency=[1000])
        vr1, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], orientation=ori_rot, frequency=[1000])
        # With rotation the interpolated value at az=0 must change
        self.assertFalse(np.allclose(vr0, vr1))

    def test_all_outputs_finite(self):
        ant = make_multi_freq_ant([500, 1000, 2000], polarimetric=True)
        freqs = np.linspace(400, 2100, 20)
        az = np.linspace(-np.pi, np.pi, 36).reshape(1, -1)
        el = np.zeros_like(az)
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)
        self.assertTrue(np.all(np.isfinite(vr)))
        self.assertTrue(np.all(np.isfinite(vi)))
        self.assertTrue(np.all(np.isfinite(hr)))
        self.assertTrue(np.all(np.isfinite(hi)))


# ======================================================================================
# Acoustic frequency scenarios
# ======================================================================================

class TestAcousticFrequencies(unittest.TestCase):
    """Test with acoustic frequency ranges (20 Hz – 20 kHz)."""

    def _make_acoustic_multi(self):
        """Build a simple multi-freq arrayant at octave-band center frequencies."""
        freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        n_freq = len(freqs)
        n_az = 36
        az_grid = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
        # Simple dipole-like pattern: cos(az) * scale
        e_theta_re = np.zeros((1, n_az, 1, n_freq))
        for i in range(n_freq):
            scale = 1.0 + 0.1 * i
            e_theta_re[0, :, 0, i] = scale * np.cos(az_grid)
        ant = {
            "e_theta_re": e_theta_re,
            "e_theta_im": np.zeros_like(e_theta_re),
            "e_phi_re": np.zeros_like(e_theta_re),
            "e_phi_im": np.zeros_like(e_theta_re),
            "azimuth_grid": az_grid,
            "elevation_grid": np.array([0.0]),
            "center_freq": np.array(freqs, dtype=np.float64),
        }
        return ant, freqs

    def test_octave_band_interpolation(self):
        ant, freqs = self._make_acoustic_multi()
        query_freqs = [100, 350, 750, 1500, 3000, 6000, 12000]
        vr, vi, hr, hi = arrayant.interpolate(ant, [[0]], [[0]], frequency=query_freqs)
        self.assertEqual(vr.shape[2], len(query_freqs))
        self.assertTrue(np.all(np.isfinite(vr)))

    def test_exact_octave_match(self):
        ant, freqs = self._make_acoustic_multi()
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=freqs)
        # On-axis (az=0) value should increase with frequency index
        for i in range(1, len(freqs)):
            self.assertGreater(vr[0, 0, i], vr[0, 0, i - 1])

    def test_subwoofer_extrapolation(self):
        """Below-range query clamps to lowest entry."""
        ant, freqs = self._make_acoustic_multi()
        vr_low, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[10])
        vr_first, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[freqs[0]])
        npt.assert_almost_equal(vr_low[0, 0, 0], vr_first[0, 0, 0], decimal=12)

    def test_ultrasonic_extrapolation(self):
        """Above-range query clamps to highest entry."""
        ant, freqs = self._make_acoustic_multi()
        vr_high, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[40000])
        vr_last, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[freqs[-1]])
        npt.assert_almost_equal(vr_high[0, 0, 0], vr_last[0, 0, 0], decimal=12)

    def test_single_freq_ant_acoustic_duplication(self):
        """Single-freq omnidirectional speaker duplicated across 1/3-octave bands."""
        ant = {"e_theta_re": np.ones((1, 1, 1)), "e_theta_im": np.zeros((1, 1, 1)),
               "e_phi_re": np.zeros((1, 1, 1)), "e_phi_im": np.zeros((1, 1, 1)),
               "azimuth_grid": np.array([0.0]), "elevation_grid": np.array([0.0])}
        third_octave = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                        3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=third_octave)
        self.assertEqual(vr.shape, (1, 1, len(third_octave)))
        # All slices must be 1.0
        npt.assert_almost_equal(vr.ravel(), np.ones(len(third_octave)), decimal=14)


# ======================================================================================
# Radio frequency scenarios
# ======================================================================================

class TestRadioFrequencies(unittest.TestCase):
    """Test with radio frequency ranges (GHz)."""

    def test_wifi_band_interpolation(self):
        """Multi-freq antenna at WiFi 2.4 GHz band edges, query at channel centers."""
        freqs = [2.4e9, 2.45e9, 2.5e9]
        n_freq = len(freqs)
        n_az = 36
        az_grid = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
        e_theta_re = np.zeros((1, n_az, 1, n_freq))
        for i in range(n_freq):
            scale = 1.0 + 0.1 * i
            e_theta_re[0, :, 0, i] = scale * np.cos(az_grid)
        ant = {
            "e_theta_re": e_theta_re,
            "e_theta_im": np.zeros_like(e_theta_re),
            "e_phi_re": np.zeros_like(e_theta_re),
            "e_phi_im": np.zeros_like(e_theta_re),
            "azimuth_grid": az_grid,
            "elevation_grid": np.array([0.0]),
            "center_freq": np.array(freqs, dtype=np.float64),
        }
        wifi_channels = [2.412e9, 2.437e9, 2.462e9, 2.484e9]
        vr, vi, hr, hi = arrayant.interpolate(ant, [[0]], [[0]], frequency=wifi_channels)
        self.assertEqual(vr.shape[2], 4)
        self.assertTrue(np.all(np.isfinite(vr)))
        # On-axis gain should generally increase with frequency
        self.assertLessEqual(abs(vr[0, 0, 0]), abs(vr[0, 0, 3]) + 0.01)

    def test_5g_mmwave_band(self):
        """Multi-freq at 28 GHz mmWave band."""
        freqs = [27.5e9, 28.0e9, 28.5e9]
        ant = make_multi_freq_ant(freqs)
        query = np.linspace(27.0e9, 29.0e9, 10)
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=query)
        self.assertEqual(vr.shape[2], 10)
        self.assertTrue(np.all(np.isfinite(vr)))


# ======================================================================================
# Error handling
# ======================================================================================

class TestErrorHandling(unittest.TestCase):

    def test_no_arrayant_raises(self):
        with self.assertRaises(KeyError):
            arrayant.interpolate()

    def test_empty_azimuth_raises(self):
        ant = {"e_theta_re": np.random.random((5, 10, 3)),
               "e_theta_im": np.random.random((5, 10, 3)),
               "e_phi_re": np.random.random((5, 10, 3)),
               "e_phi_im": np.random.random((5, 10, 3)),
               "azimuth_grid": np.linspace(-np.pi, np.pi, 10),
               "elevation_grid": np.linspace(-np.pi / 2, np.pi / 2, 5)}
        with self.assertRaises(ValueError) as ctx:
            arrayant.interpolate(ant)
        self.assertEqual(str(ctx.exception), "Azimuth angles cannot be empty.")

    def test_mismatched_az_el_raises(self):
        ant = {"e_theta_re": np.random.random((5, 10, 3)),
               "e_theta_im": np.random.random((5, 10, 3)),
               "e_phi_re": np.random.random((5, 10, 3)),
               "e_phi_im": np.random.random((5, 10, 3)),
               "azimuth_grid": np.linspace(-np.pi, np.pi, 10),
               "elevation_grid": np.linspace(-np.pi / 2, np.pi / 2, 5)}
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, 0)

    def test_element_out_of_range_raises(self):
        ant = {"e_theta_re": np.random.random((2, 10, 3)),
               "e_theta_im": np.random.random((2, 10, 3)),
               "e_phi_re": np.random.random((2, 10, 3)),
               "e_phi_im": np.random.random((2, 10, 3)),
               "azimuth_grid": np.linspace(-np.pi, np.pi, 10),
               "elevation_grid": np.array([-np.pi / 4, np.pi / 4])}
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, 0, 0, element=5)

    def test_fast_access_non_contiguous_raises(self):
        ant = {"e_theta_re": np.random.random((2, 10, 3)),
               "e_theta_im": np.random.random((2, 10, 3)),
               "e_phi_re": np.random.random((2, 10, 3)),
               "e_phi_im": np.random.random((2, 10, 3)),
               "azimuth_grid": np.linspace(-np.pi, np.pi, 10),
               "elevation_grid": np.array([-np.pi / 4, np.pi / 4])}
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, 0, 0, fast_access=1)

    def test_multi_freq_missing_frequency_raises(self):
        """4D arrayant without frequency parameter must throw."""
        ant = make_multi_freq_ant([1000, 2000])
        with self.assertRaises(ValueError) as ctx:
            arrayant.interpolate(ant, [[0]], [[0]])
        self.assertIn("frequency", str(ctx.exception).lower())

    def test_multi_freq_dist_raises(self):
        ant = make_multi_freq_ant([1000, 2000])
        with self.assertRaises(ValueError) as ctx:
            arrayant.interpolate(ant, [[0]], [[0]], frequency=[1500], dist=1)
        self.assertIn("dist", str(ctx.exception).lower())

    def test_multi_freq_local_angles_raises(self):
        ant = make_multi_freq_ant([1000, 2000])
        with self.assertRaises(ValueError) as ctx:
            arrayant.interpolate(ant, [[0]], [[0]], frequency=[1500], local_angles=1)
        self.assertIn("local_angles", str(ctx.exception).lower())

    def test_multi_freq_inconsistent_grids_raises(self):
        """Corrupted multi-freq dict (mismatched grid) should fail validation."""
        ant = make_multi_freq_ant([1000, 2000])
        # Corrupt: make e_theta_re have wrong shape in 2nd azimuth dimension
        bad = ant.copy()
        shape = list(bad["e_theta_re"].shape)
        bad_data = np.zeros((shape[0], shape[1] + 1, shape[2], shape[3]))
        bad_data[:, :shape[1], :, :] = bad["e_theta_re"]
        bad["e_theta_re"] = bad_data
        with self.assertRaises(ValueError):
            arrayant.interpolate(bad, [[0]], [[0]], frequency=[1500])

    def test_multi_freq_element_out_of_range_raises(self):
        ant = make_multi_freq_ant([1000])
        with self.assertRaises(ValueError):
            arrayant.interpolate(ant, [[0]], [[0]], element=[5], frequency=[1000])


# ======================================================================================
# Edge cases
# ======================================================================================

class TestEdgeCases(unittest.TestCase):

    def test_single_angle_single_element(self):
        ant = {"e_theta_re": [[[1]]], "e_theta_im": [[[0]]],
               "e_phi_re": [[[0]]], "e_phi_im": [[[0]]],
               "azimuth_grid": [0], "elevation_grid": [0]}
        vr, vi, hr, hi = arrayant.interpolate(ant, 0, 0)
        self.assertEqual(vr.shape, (1, 1))
        npt.assert_almost_equal(vr[0, 0], 1.0, decimal=14)

    def test_many_angles(self):
        ant = arrayant.generate("omni")
        n = 10000
        az = np.random.uniform(-np.pi, np.pi, (1, n))
        el = np.random.uniform(-np.pi / 2, np.pi / 2, (1, n))
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el)
        self.assertEqual(vr.shape, (1, n))
        self.assertTrue(np.all(np.isfinite(vr)))

    def test_single_freq_no_frequency_gives_2d(self):
        """Without frequency kwarg, output is 2D (original behavior)."""
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        vr, vi, hr, hi = arrayant.interpolate(ant, 0, 0)
        self.assertEqual(vr.ndim, 2)

    def test_single_freq_with_frequency_gives_3d(self):
        """With frequency kwarg, output is 3D even for single freq."""
        ant = {"e_theta_re": [[1, 0]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        vr, vi, hr, hi = arrayant.interpolate(ant, 0, 0, frequency=[1000])
        self.assertEqual(vr.ndim, 3)

    def test_multi_freq_single_entry(self):
        """4D arrayant with only 1 frequency entry."""
        ant = make_multi_freq_ant([1000])
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[500, 1000, 2000])
        self.assertEqual(vr.shape[2], 3)
        # All slices identical since only one freq entry (clamped)
        npt.assert_almost_equal(vr[0, 0, 0], vr[0, 0, 1], decimal=12)
        npt.assert_almost_equal(vr[0, 0, 1], vr[0, 0, 2], decimal=12)

    def test_multi_freq_many_query_freqs(self):
        ant = make_multi_freq_ant([100, 1000, 10000])
        freqs = np.linspace(50, 15000, 200).tolist()
        vr, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=freqs)
        self.assertEqual(vr.shape[2], 200)
        self.assertTrue(np.all(np.isfinite(vr)))

    def test_near_exact_frequency_tolerance(self):
        """Query at center_freq + tiny offset should match exact."""
        ant = make_multi_freq_ant([1000, 2000])
        vr_near, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[1000.0000001])
        vr_exact, _, _, _ = arrayant.interpolate(ant, [[0]], [[0]], frequency=[1000.0])
        npt.assert_almost_equal(vr_near[0, 0, 0], vr_exact[0, 0, 0], decimal=10)

    def test_complex_vs_real_consistency(self):
        """Complex output Re/Im must match separate real outputs for both paths."""
        # Single-freq path
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4, np.pi / 2]]
        el = [[0, 0, 0]]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el)
        v, h = arrayant.interpolate(ant, az, el, complex=1)
        npt.assert_almost_equal(v.real, vr, decimal=14)
        npt.assert_almost_equal(v.imag, vi, decimal=14)
        npt.assert_almost_equal(h.real, hr, decimal=14)
        npt.assert_almost_equal(h.imag, hi, decimal=14)

    def test_complex_vs_real_consistency_with_freq(self):
        """Complex vs real consistency for single-freq + frequency duplication."""
        ant = {"e_theta_re": [[-2, 2]], "e_theta_im": [[-1, 1]],
               "e_phi_re": [[3, 1]], "e_phi_im": [[6, 2]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = [[0, np.pi / 4]]
        el = [[0, 0]]
        freqs = [100, 1000]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)
        v, h = arrayant.interpolate(ant, az, el, frequency=freqs, complex=1)
        npt.assert_almost_equal(v.real, vr, decimal=14)
        npt.assert_almost_equal(v.imag, vi, decimal=14)
        npt.assert_almost_equal(h.real, hr, decimal=14)
        npt.assert_almost_equal(h.imag, hi, decimal=14)

    def test_complex_vs_real_consistency_multi_freq(self):
        """Complex vs real consistency for multi-freq path."""
        ant = make_multi_freq_ant([500, 1000, 2000], polarimetric=True)
        az = [[0, np.pi / 4]]
        el = [[0, 0]]
        freqs = [750, 1500]
        vr, vi, hr, hi = arrayant.interpolate(ant, az, el, frequency=freqs)
        v, h = arrayant.interpolate(ant, az, el, frequency=freqs, complex=1)
        npt.assert_almost_equal(v.real, vr, decimal=12)
        npt.assert_almost_equal(v.imag, vi, decimal=12)
        npt.assert_almost_equal(h.real, hr, decimal=12)
        npt.assert_almost_equal(h.imag, hi, decimal=12)

    def test_spherical_wave_per_element_angles(self):
        """n_out angles (different az/el per element) with frequency duplication."""
        ant = {"e_theta_re": [[1, -1]], "e_theta_im": [[0, 0]],
               "e_phi_re": [[0, 0]], "e_phi_im": [[0, 0]],
               "azimuth_grid": [0, np.pi], "elevation_grid": 0}
        az = np.array([[0, np.pi / 4], [np.pi / 4, np.pi / 2]])  # (2, 2)
        el = np.zeros((2, 2))
        vr, _, _, _ = arrayant.interpolate(ant, az, el, element=[0, 0], frequency=[1000, 2000])
        self.assertEqual(vr.shape, (2, 2, 2))
        # Slices must be identical (single-freq duplication)
        npt.assert_almost_equal(vr[:, :, 0], vr[:, :, 1], decimal=14)


if __name__ == '__main__':
    unittest.main()