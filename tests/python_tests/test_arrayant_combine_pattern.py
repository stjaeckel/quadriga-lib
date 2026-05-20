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
# the numpy<->C++ boundary may also surface TypeError. Accept any of them for error cases.
ANY_ERR = (ValueError, TypeError, RuntimeError)


def _grids():
    """Small azimuth/elevation grids reused across the tests."""
    az = np.array([-1.5, 0.0, 1.5, 2.0]) * np.pi / 2.0     # 4 points, within [-pi, pi]
    el = np.array([-0.9, 0.0, 0.9]) * np.pi / 2.0          # 3 points, within [-pi/2, pi/2]
    return az, el


def _ant_sf(n_elem=2, et_re=1.0, et_im=2.0, ep_re=3.0, ep_im=4.0,
            center_freq=2.0e9, name="bla"):
    """Build a single-frequency arrayant dict with constant pattern fields."""
    az, el = _grids()
    shape = (el.size, az.size, n_elem)
    return {
        "e_theta_re": np.full(shape, float(et_re)),
        "e_theta_im": np.full(shape, float(et_im)),
        "e_phi_re": np.full(shape, float(ep_re)),
        "e_phi_im": np.full(shape, float(ep_im)),
        "azimuth_grid": az,
        "elevation_grid": el,
        "element_pos": np.zeros((3, n_elem)),
        "coupling_re": np.ones((n_elem, 1)),               # sum all elements into one port
        "coupling_im": np.zeros((n_elem, 1)),
        "center_freq": float(center_freq),
        "name": name,
    }


def _stack_mf(ants):
    """Stack a list of single-freq dicts into one multi-freq dict (4D pattern fields)."""
    mf = {}
    for key in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im"):
        mf[key] = np.stack([np.asarray(a[key], dtype=float) for a in ants], axis=3)
    mf["azimuth_grid"] = np.asarray(ants[0]["azimuth_grid"], dtype=float)
    mf["elevation_grid"] = np.asarray(ants[0]["elevation_grid"], dtype=float)
    mf["element_pos"] = np.asarray(ants[0]["element_pos"], dtype=float)
    mf["center_freq"] = np.array([a["center_freq"] for a in ants], dtype=float)
    if "coupling_re" in ants[0]:
        mf["coupling_re"] = np.asarray(ants[0]["coupling_re"], dtype=float)
    if "coupling_im" in ants[0]:
        mf["coupling_im"] = np.asarray(ants[0]["coupling_im"], dtype=float)
    return mf


def _slice_freq(mf, f):
    """Extract frequency index f of a multi-freq output dict as a single-freq dict."""
    sf = {key: mf[key][:, :, :, f] for key in
          ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im")}
    sf["azimuth_grid"] = mf["azimuth_grid"]
    sf["elevation_grid"] = mf["elevation_grid"]
    sf["element_pos"] = mf["element_pos"]
    sf["coupling_re"] = mf["coupling_re"]
    sf["coupling_im"] = mf["coupling_im"]
    sf["center_freq"] = float(mf["center_freq"][f])
    return sf


class test_case(unittest.TestCase):

    # ---- Single-frequency tests ----

    def test_single_frequency(self):
        # Two elements, each pattern field constant -> coupling [1;1] sums them into one port.
        ant = _ant_sf(n_elem=2)
        out = arrayant.combine_pattern(ant)

        npt.assert_allclose(out["e_theta_re"][:, :, 0], np.full((3, 4), 2.0), atol=1e-14, rtol=0)
        npt.assert_allclose(out["e_theta_im"][:, :, 0], np.full((3, 4), 4.0), atol=1e-14, rtol=0)
        npt.assert_allclose(out["e_phi_re"][:, :, 0], np.full((3, 4), 6.0), atol=1e-14, rtol=0)
        npt.assert_allclose(out["e_phi_im"][:, :, 0], np.full((3, 4), 8.0), atol=1e-14, rtol=0)

        # Output coupling is identity, element positions are zeroed.
        npt.assert_allclose(out["coupling_re"], np.eye(1), atol=1e-14, rtol=0)
        npt.assert_allclose(out["coupling_im"], np.zeros((1, 1)), atol=1e-14, rtol=0)
        npt.assert_allclose(out["element_pos"][:, 0], [0.0, 0.0, 0.0], atol=1e-14, rtol=0)

        # Regression: with no `freq` argument the input center_freq must be retained (not zeroed).
        self.assertEqual(float(out["center_freq"]), 2.0e9)
        self.assertEqual(out["name"], "bla")

    def test_grid_interpolation_and_directivity(self):
        ant = arrayant.generate("3gpp", res=10, N=2, spacing=0.0)
        ant["e_theta_re"][:, :, 1] = 2.0 * ant["e_theta_re"][:, :, 1]
        ant["coupling_re"] = [1, 1]
        ant["coupling_im"] = [0, 0]
        ant["name"] = "buy_bitcoin"

        # Element 0 + 2*element 0 -> 3x the base pattern.
        out = arrayant.combine_pattern(ant)
        npt.assert_allclose(out["e_theta_re"][:, :, 0], 3.0 * ant["e_theta_re"][:, :, 0],
                            atol=1e-14, rtol=0)
        npt.assert_allclose(out["coupling_re"], np.eye(1), atol=1e-14, rtol=0)
        npt.assert_allclose(out["element_pos"][:, 0], [0.0, 0.0, 0.0], atol=1e-14, rtol=0)

        # Denser azimuth grid -> every 2nd sample reproduces the original grid.
        az_grid = np.linspace(-np.pi, np.pi, 73)
        out = arrayant.combine_pattern(ant, 3.0e9, az_grid)
        npt.assert_allclose(out["e_theta_re"][:, ::2, 0], 3.0 * ant["e_theta_re"][:, :, 0],
                            atol=1e-14, rtol=0)
        self.assertEqual(float(out["center_freq"]), 3.0e9)

        # Denser elevation grid via keyword argument; name is propagated.
        el_grid = np.linspace(-np.pi / 2, np.pi / 2, 37)
        out = arrayant.combine_pattern(ant, elevation_grid=el_grid)
        npt.assert_allclose(out["e_theta_re"][::2, :, 0], 3.0 * ant["e_theta_re"][:, :, 0],
                            atol=1e-14, rtol=0)
        self.assertEqual(out["name"], "buy_bitcoin")

        # Directivity is preserved when combining co-located elements ...
        directivity_ant = arrayant.calc_directivity(ant, 0)
        directivity_out = arrayant.calc_directivity(out)
        npt.assert_almost_equal(directivity_ant, directivity_out, decimal=1)

        # ... and increases once the elements are physically separated.
        ant["element_pos"][1, :] = [-0.25, 0.25]
        out = arrayant.combine_pattern(ant)
        directivity_out = arrayant.calc_directivity(out)
        self.assertGreater(directivity_out, directivity_ant + 1.2)

    # ---- Multi-frequency tests ----

    def test_multifreq_default(self):
        # Three single-freq antennas, pattern scaled by 1/2/3 at 1/2/3 GHz.
        ants = [_ant_sf(n_elem=1, et_re=k, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=k * 1e9)
                for k in (1, 2, 3)]
        out = arrayant.combine_pattern(_stack_mf(ants))

        # Multi-freq input -> 4D output, one slice per frequency, input order preserved.
        self.assertEqual(out["e_theta_re"].ndim, 4)
        self.assertEqual(out["e_theta_re"].shape[3], 3)
        for k in range(3):
            npt.assert_allclose(out["e_theta_re"][:, :, 0, k], np.full((3, 4), k + 1.0),
                                atol=1e-12, rtol=0)
        npt.assert_allclose(out["center_freq"], [1e9, 2e9, 3e9], rtol=1e-12)

    def test_multifreq_freq_argument(self):
        ants = [_ant_sf(n_elem=1, et_re=k, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=k * 1e9)
                for k in (1, 2, 3)]
        mf = _stack_mf(ants)

        # Scalar freq selects a single frequency -> single (3D) dict, scalar center_freq.
        out = arrayant.combine_pattern(mf, 2e9)
        self.assertEqual(out["e_theta_re"].ndim, 3)
        self.assertEqual(np.ndim(out["center_freq"]), 0)
        npt.assert_allclose(out["e_theta_re"][:, :, 0], np.full((3, 4), 2.0), atol=1e-12, rtol=0)
        self.assertEqual(float(out["center_freq"]), 2e9)

        # Vector freq interpolates -> 4D output (linear blend of the in-phase reals).
        out = arrayant.combine_pattern(mf, [1.5e9, 2.5e9])
        self.assertEqual(out["e_theta_re"].shape[3], 2)
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 0], np.full((3, 4), 1.5), atol=1e-10, rtol=0)
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 1], np.full((3, 4), 2.5), atol=1e-10, rtol=0)
        npt.assert_allclose(out["center_freq"], [1.5e9, 2.5e9], rtol=1e-12)

        # Out-of-range frequencies clamp to the nearest entry.
        out = arrayant.combine_pattern(mf, [0.5e9, 10e9])
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 0], np.full((3, 4), 1.0), atol=1e-12, rtol=0)
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 1], np.full((3, 4), 3.0), atol=1e-12, rtol=0)

        # Single-freq input + vector freq -> 4D output, all clamped to that one entry.
        single = _ant_sf(n_elem=1, et_re=2.0, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=2e9)
        out = arrayant.combine_pattern(single, [1e9, 2e9, 3e9])
        self.assertEqual(out["e_theta_re"].shape[3], 3)
        for k in range(3):
            npt.assert_allclose(out["e_theta_re"][:, :, 0, k], np.full((3, 4), 2.0),
                                atol=1e-12, rtol=0)
        npt.assert_allclose(out["center_freq"], [1e9, 2e9, 3e9], rtol=1e-12)

    def test_multifreq_custom_grids(self):
        ants = [_ant_sf(n_elem=1, et_re=k, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=k * 1e9)
                for k in (1, 2, 3)]
        az_new = np.linspace(-np.pi, np.pi, 9)
        el_new = np.linspace(-np.pi / 2, np.pi / 2, 5)
        out = arrayant.combine_pattern(_stack_mf(ants), azimuth_grid=az_new, elevation_grid=el_new)

        self.assertEqual(out["e_theta_re"].shape, (5, 9, 1, 3))
        self.assertEqual(out["azimuth_grid"].size, 9)
        self.assertEqual(out["elevation_grid"].size, 5)

    def test_multifreq_coupling_and_polarization(self):
        az, el = _grids()

        # Per-frequency coupling supplied as a 3D array (n_elem, n_ports, n_freq).
        coupling_re = np.zeros((1, 1, 2))
        coupling_re[:, :, 0] = 1.0
        coupling_re[:, :, 1] = 3.0
        mf = {
            "e_theta_re": np.ones((3, 4, 1, 2)),
            "e_theta_im": np.zeros((3, 4, 1, 2)),
            "e_phi_re": np.zeros((3, 4, 1, 2)),
            "e_phi_im": np.zeros((3, 4, 1, 2)),
            "azimuth_grid": az,
            "elevation_grid": el,
            "element_pos": np.zeros((3, 1)),
            "coupling_re": coupling_re,
            "coupling_im": np.zeros((1, 1, 2)),
            "center_freq": np.array([1e9, 2e9]),
        }
        # Midpoint frequency -> coupling interpolated between 1 and 3 -> 2; pattern stays 1.
        out = arrayant.combine_pattern(mf, 1.5e9)
        npt.assert_allclose(out["e_theta_re"][:, :, 0], np.full((3, 4), 2.0), atol=1e-10, rtol=0)

        # Both polarisations carried through a multi-freq combine.
        ants = [_ant_sf(n_elem=1, et_re=k, et_im=0.0, ep_re=-k, ep_im=0.0, center_freq=k * 1e9)
                for k in (1, 2)]
        out = arrayant.combine_pattern(_stack_mf(ants))
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 0], np.full((3, 4), 1.0), atol=1e-12, rtol=0)
        npt.assert_allclose(out["e_phi_re"][:, :, 0, 0], np.full((3, 4), -1.0), atol=1e-12, rtol=0)
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 1], np.full((3, 4), 2.0), atol=1e-12, rtol=0)
        npt.assert_allclose(out["e_phi_re"][:, :, 0, 1], np.full((3, 4), -2.0), atol=1e-12, rtol=0)

        # Two physical elements summed by coupling [1;1] at each frequency.
        ants2 = [_ant_sf(n_elem=2, et_re=k, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=k * 1e9)
                 for k in (1, 2)]
        out = arrayant.combine_pattern(_stack_mf(ants2))
        self.assertEqual(out["e_theta_re"].shape, (3, 4, 1, 2))
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 0], np.full((3, 4), 2.0), atol=1e-12, rtol=0)
        npt.assert_allclose(out["e_theta_re"][:, :, 0, 1], np.full((3, 4), 4.0), atol=1e-12, rtol=0)

    def test_multifreq_directivity(self):
        ant = arrayant.generate("3gpp", res=10, N=2, spacing=0.0)
        ant["coupling_re"] = np.ones((2, 1))
        ant["coupling_im"] = np.zeros((2, 1))
        ant["element_pos"][1, :] = [-0.25, 0.25]

        d_single = arrayant.calc_directivity(ant, 0)

        # Replicate the same (element-separated) antenna at two frequencies.
        mf = {key: np.stack([ant[key], ant[key]], axis=3)
              for key in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im")}
        mf["azimuth_grid"] = ant["azimuth_grid"]
        mf["elevation_grid"] = ant["elevation_grid"]
        mf["element_pos"] = ant["element_pos"]
        mf["coupling_re"] = ant["coupling_re"]
        mf["coupling_im"] = ant["coupling_im"]
        mf["center_freq"] = np.array([2e9, 3e9])

        out = arrayant.combine_pattern(mf)
        self.assertEqual(out["e_theta_re"].shape[3], 2)
        for k in range(2):
            d_combined = arrayant.calc_directivity(_slice_freq(out, k), 0)
            self.assertGreater(d_combined, d_single + 1.2)

    # ---- freq input-conversion tests ----

    def test_freq_input_conversion(self):
        ant = _ant_sf(n_elem=1, et_re=1.0, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=2e9)

        # Python int and float scalars both override the center frequency.
        self.assertEqual(float(arrayant.combine_pattern(ant, int(3e9))["center_freq"]), 3e9)
        self.assertEqual(float(arrayant.combine_pattern(ant, 3.0e9)["center_freq"]), 3e9)

        # 0-D numpy array and numpy scalar are promoted to a 1-element vector.
        self.assertEqual(float(arrayant.combine_pattern(ant, np.array(2.5e9))["center_freq"]), 2.5e9)
        self.assertEqual(float(arrayant.combine_pattern(ant, np.float64(2.5e9))["center_freq"]), 2.5e9)

        # freq <= 0 and freq=None mean "not given" -> the input center_freq is kept.
        for noop in (0, 0.0, None):
            self.assertEqual(float(arrayant.combine_pattern(ant, noop)["center_freq"]), 2e9)

        # A Python list / numpy array of length > 1 triggers the multi-frequency path.
        out = arrayant.combine_pattern(ant, [1.5e9, 2.5e9])
        self.assertEqual(out["e_theta_re"].ndim, 4)
        self.assertEqual(out["e_theta_re"].shape[3], 2)
        npt.assert_allclose(out["center_freq"], [1.5e9, 2.5e9], rtol=1e-12)

        out = arrayant.combine_pattern(ant, np.array([1e9, 2e9, 3e9]))
        self.assertEqual(out["e_theta_re"].shape[3], 3)

    def test_freq_input_invalid(self):
        ant = _ant_sf(n_elem=1, et_re=1.0, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=2e9)

        # Non-numeric `freq` values must raise a clear error, not crash or pass silently.
        for bad_freq in ("not_a_number",
                         ["a", "b"],
                         np.array(["p", "q"]),
                         {"freq": 1e9},
                         object()):
            with self.assertRaises(ANY_ERR):
                arrayant.combine_pattern(ant, bad_freq)

    # ---- Error tests ----

    def test_input_errors(self):
        # Mismatched pattern field shapes.
        ant = _ant_sf(n_elem=2)
        ant["e_theta_im"] = np.ones((3, 3, 2))                # wrong azimuth count
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(ant)

        # azimuth_grid length inconsistent with the pattern.
        ant = _ant_sf(n_elem=2)
        ant["azimuth_grid"] = np.array([-1.0, 0.0, 1.0])      # 3 != 4
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(ant)

        # element_pos must be (3, n_elements).
        ant = _ant_sf(n_elem=2)
        ant["element_pos"] = np.zeros((3, 5))
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(ant)

        # coupling rows must match the number of elements.
        ant = _ant_sf(n_elem=2)
        ant["coupling_re"] = np.ones((5, 1))
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(ant)

        # Imaginary coupling without a real part is rejected.
        ant = _ant_sf(n_elem=2)
        del ant["coupling_re"]
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(ant)

    def test_multifreq_input_errors(self):
        ants = [_ant_sf(n_elem=1, et_re=k, et_im=0.0, ep_re=0.0, ep_im=0.0, center_freq=k * 1e9)
                for k in (1, 2, 3)]

        # Pattern fields disagree on the number of frequency slices.
        mf = _stack_mf(ants)
        mf["e_theta_im"] = mf["e_theta_im"][:, :, :, :2]      # 2 freqs vs 3
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(mf)

        # center_freq length does not match the 4D pattern.
        mf = _stack_mf(ants)
        mf["center_freq"] = np.array([1e9, 2e9])              # 2 vs 3
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(mf)

        # Pattern field with an unsupported number of dimensions.
        mf = _stack_mf(ants)
        mf["e_theta_re"] = np.ones((3, 4))                    # 2D, neither 3D nor 4D
        with self.assertRaises(ANY_ERR):
            arrayant.combine_pattern(mf)


if __name__ == '__main__':
    unittest.main()