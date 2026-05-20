# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import arrayant

# pybind11 maps std::invalid_argument -> ValueError and std::runtime_error -> RuntimeError;
# a non-dict argument fails the py::dict conversion with TypeError. Accept any for error cases.
ANY_ERR = (ValueError, TypeError, RuntimeError)

PATTERN_KEYS = ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im")

# Shared sampling grids — every hand-built antenna uses these unless overridden.
AZ = np.array([-2.0, -0.5, 0.5, 2.0])      # 4 azimuth points, within [-pi, pi], sorted
EL = np.array([-0.8, 0.0, 0.8])            # 3 elevation points, within [-pi/2, pi/2], sorted


def _ant(values, name="ant", center_freq=2.0e9, az=None, el=None,
         element_pos=None, coupling_re=None, coupling_im=None):
    """Build a single-frequency arrayant dict.

    `values` has one entry per element; element k's four pattern fields are filled with
    distinct constants derived from values[k], so the origin of any output slice is checkable.
    """
    n_elem = len(values)
    az = AZ if az is None else np.asarray(az, dtype=float)
    el = EL if el is None else np.asarray(el, dtype=float)
    shape = (el.size, az.size, n_elem)

    etr, eti = np.zeros(shape), np.zeros(shape)
    epr, epi = np.zeros(shape), np.zeros(shape)
    for k, v in enumerate(values):
        etr[:, :, k] = float(v)
        eti[:, :, k] = float(v) + 0.1
        epr[:, :, k] = float(v) + 0.2
        epi[:, :, k] = float(v) + 0.3

    return {
        "e_theta_re": etr, "e_theta_im": eti,
        "e_phi_re": epr, "e_phi_im": epi,
        "azimuth_grid": az,
        "elevation_grid": el,
        "element_pos": (np.zeros((3, n_elem)) if element_pos is None
                        else np.asarray(element_pos, dtype=float)),
        "coupling_re": (np.eye(n_elem) if coupling_re is None
                        else np.asarray(coupling_re, dtype=float)),
        "coupling_im": (np.zeros((n_elem, n_elem)) if coupling_im is None
                        else np.asarray(coupling_im, dtype=float)),
        "center_freq": float(center_freq),
        "name": name,
    }


def _stack_mf(ants, center_freqs=None):
    """Stack single-frequency dicts into one multi-frequency dict (4D pattern fields)."""
    mf = {key: np.stack([a[key] for a in ants], axis=3) for key in PATTERN_KEYS}
    mf["azimuth_grid"] = ants[0]["azimuth_grid"]
    mf["elevation_grid"] = ants[0]["elevation_grid"]
    mf["element_pos"] = ants[0]["element_pos"]
    mf["coupling_re"] = ants[0]["coupling_re"]
    mf["coupling_im"] = ants[0]["coupling_im"]
    if center_freqs is None:
        center_freqs = [1.0e9 * (i + 1) for i in range(len(ants))]
    mf["center_freq"] = np.asarray(center_freqs, dtype=float)
    mf["name"] = ants[0]["name"]
    return mf


def _freq_slice(d, f):
    """Extract frequency index f of a 4D dict as a dict of 3D pattern fields."""
    return {key: d[key][:, :, :, f] for key in PATTERN_KEYS}


def _empty_ant():
    """A frequency-dependent dict with zero frequency entries (4th pattern dimension = 0)."""
    z = np.zeros((EL.size, AZ.size, 2, 0))
    return {
        "e_theta_re": z, "e_theta_im": z, "e_phi_re": z, "e_phi_im": z,
        "azimuth_grid": AZ, "elevation_grid": EL,
        "element_pos": np.zeros((3, 2)),
        "coupling_re": np.eye(2), "coupling_im": np.zeros((2, 2)),
        "center_freq": np.zeros(0),
        "name": "empty",
    }


class test_case(unittest.TestCase):

    def _check_concat(self, out, a, b):
        """Assert `out` is the element-wise concatenation of single-freq dicts `a` then `b`."""
        na = a["e_theta_re"].shape[2]
        nb = b["e_theta_re"].shape[2]
        for key in PATTERN_KEYS:
            self.assertEqual(out[key].shape[2], na + nb)
            for k in range(na):
                npt.assert_allclose(out[key][:, :, k], a[key][:, :, k], atol=1e-12, rtol=0)
            for k in range(nb):
                npt.assert_allclose(out[key][:, :, na + k], b[key][:, :, k], atol=1e-12, rtol=0)

    # ---- Single-frequency tests ----

    def test_single_basic(self):
        ant1 = _ant([1, 2])
        ant2 = _ant([3, 4])
        out = arrayant.concat(ant1, ant2)

        # 2 + 2 = 4 elements; first two slices from ant1, last two from ant2.
        self.assertEqual(out["e_theta_re"].shape[2], 4)
        self._check_concat(out, ant1, ant2)

        # Sampling grids are preserved.
        npt.assert_allclose(out["azimuth_grid"], ant1["azimuth_grid"], atol=1e-12, rtol=0)
        npt.assert_allclose(out["elevation_grid"], ant1["elevation_grid"], atol=1e-12, rtol=0)

    def test_coupling_block_diagonal(self):
        c1_re = np.array([[1.0, 2.0], [3.0, 4.0]])
        c1_im = np.array([[0.1, 0.2], [0.3, 0.4]])
        c2_re = np.array([[5.0, 6.0], [7.0, 8.0]])
        c2_im = np.array([[0.5, 0.6], [0.7, 0.8]])

        ant1 = _ant([1, 2], coupling_re=c1_re, coupling_im=c1_im)
        ant2 = _ant([3, 4], coupling_re=c2_re, coupling_im=c2_im)
        out = arrayant.concat(ant1, ant2)

        # Coupling is assembled block-diagonally: inputs on the diagonal, zeros off-diagonal.
        zero = np.zeros((2, 2))
        expected_re = np.block([[c1_re, zero], [zero, c2_re]])
        expected_im = np.block([[c1_im, zero], [zero, c2_im]])
        self.assertEqual(out["coupling_re"].shape, (4, 4))
        self.assertEqual(out["coupling_im"].shape, (4, 4))
        npt.assert_allclose(out["coupling_re"], expected_re, atol=1e-12, rtol=0)
        npt.assert_allclose(out["coupling_im"], expected_im, atol=1e-12, rtol=0)

    def test_element_pos(self):
        pos1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        pos2 = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        ant1 = _ant([1, 2], element_pos=pos1)
        ant2 = _ant([3, 4], element_pos=pos2)
        out = arrayant.concat(ant1, ant2)

        # element_pos matrices are joined horizontally.
        self.assertEqual(out["element_pos"].shape, (3, 4))
        npt.assert_allclose(out["element_pos"], np.hstack([pos1, pos2]), atol=1e-12, rtol=0)

    def test_inherited_freq_and_name(self):
        ant1 = _ant([1, 2], name="first", center_freq=2.4e9)
        ant2 = _ant([3, 4], name="second", center_freq=5.8e9)   # deliberately different
        out = arrayant.concat(ant1, ant2)

        # center_freq and name are inherited from arrayant_in1.
        npt.assert_allclose(float(out["center_freq"]), 2.4e9, rtol=1e-9)
        self.assertEqual(out["name"], "first")

    def test_asymmetric_element_counts(self):
        ant1 = _ant([10])             # 1 element
        ant2 = _ant([20, 21])         # 2 elements
        out = arrayant.concat(ant1, ant2)

        # 1 + 2 = 3 elements; coupling grows to 3x3 block-diagonal.
        self.assertEqual(out["e_theta_re"].shape[2], 3)
        self.assertEqual(out["coupling_re"].shape, (3, 3))
        self._check_concat(out, ant1, ant2)

    # ---- Multi-frequency tests ----

    def test_multifreq_basic(self):
        cf = [1.0e9, 2.0e9]
        ant1 = _ant([1, 2])
        ant2 = _ant([3, 4])
        mf1 = _stack_mf([ant1, ant1], center_freqs=cf)
        mf2 = _stack_mf([ant2, ant2], center_freqs=cf)
        out = arrayant.concat(mf1, mf2)

        # Multi-freq input -> 4D output, two frequency entries, 4 elements each.
        self.assertEqual(out["e_theta_re"].ndim, 4)
        self.assertEqual(out["e_theta_re"].shape[3], 2)
        for f in range(2):
            self._check_concat(_freq_slice(out, f), ant1, ant2)
        npt.assert_allclose(out["center_freq"], cf, rtol=1e-12)

    def test_multifreq_distinct(self):
        cf = [1.0e9, 2.0e9]
        # Distinct content per frequency entry.
        a0, a1 = _ant([1, 2]), _ant([3, 4])
        b0, b1 = _ant([10, 11]), _ant([20, 21])
        mf_a = _stack_mf([a0, a1], center_freqs=cf)
        mf_b = _stack_mf([b0, b1], center_freqs=cf)
        out = arrayant.concat(mf_a, mf_b)

        self.assertEqual(out["e_theta_re"].shape[3], 2)
        # Entry 0 = a0 ++ b0, entry 1 = a1 ++ b1.
        self._check_concat(_freq_slice(out, 0), a0, b0)
        self._check_concat(_freq_slice(out, 1), a1, b1)
        # The two output entries differ from each other.
        self.assertFalse(np.array_equal(out["e_theta_re"][:, :, :, 0],
                                        out["e_theta_re"][:, :, :, 1]))

    # ---- Error handling ----

    def test_errors(self):
        ant1 = _ant([1, 2])
        ant2 = _ant([3, 4])

        # Inputs must be dicts.
        with self.assertRaises(ANY_ERR):
            arrayant.concat(1.0, ant2)
        with self.assertRaises(ANY_ERR):
            arrayant.concat(ant1, 1.0)

        # Inputs must not be empty (frequency-dependent dict with zero entries).
        with self.assertRaises(ANY_ERR):
            arrayant.concat(_empty_ant(), ant2)

        # Mismatched number of frequency entries (single-freq vs 2-entry multi-freq).
        with self.assertRaises(ANY_ERR):
            arrayant.concat(ant1, _stack_mf([ant2, ant2]))

        # Single-frequency: azimuth grid size mismatch.
        with self.assertRaises(ANY_ERR):
            arrayant.concat(ant1, _ant([3, 4], az=np.array([-2.0, 0.0, 2.0])))

        # Single-frequency: elevation grid size mismatch.
        with self.assertRaises(ANY_ERR):
            arrayant.concat(ant1, _ant([3, 4], el=np.array([-0.8, 0.8])))

        # Single-frequency: azimuth grid value mismatch (same size, shifted values).
        with self.assertRaises(ANY_ERR):
            arrayant.concat(ant1, _ant([3, 4], az=AZ + 0.3))

        # Multi-frequency: azimuth grid size mismatch.
        short_az = _ant([3, 4], az=np.array([-2.0, 0.0, 2.0]))
        with self.assertRaises(ANY_ERR):
            arrayant.concat(_stack_mf([ant1, ant1]), _stack_mf([short_az, short_az]))

        # Multi-frequency: elevation grid size mismatch.
        short_el = _ant([3, 4], el=np.array([-0.8, 0.8]))
        with self.assertRaises(ANY_ERR):
            arrayant.concat(_stack_mf([ant1, ant1]), _stack_mf([short_el, short_el]))


if __name__ == '__main__':
    unittest.main()
