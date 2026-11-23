// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# interpolate_pattern
Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the array and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation)
- Multi-frequency mode (4D pattern arrays, or target frequencies given): the two entries whose center frequencies
  bracket each target frequency are interpolated and blended

## Usage:
```
# Single-frequency mode
V, H = quadriga_lib.arrayant.interpolate_pattern( arrayant, azimuth, elevation, *, i_element, orientation, \
    element_pos, use_kernel )

# Single-frequency mode with the projected distance and the local angles
V, H, dist, azimuth_loc, elevation_loc, gamma = quadriga_lib.arrayant.interpolate_pattern( arrayant, azimuth, \
    elevation, ..., dist=True, local_angles=True )

# Multi-frequency mode
V, H = quadriga_lib.arrayant.interpolate_pattern( arrayant, azimuth, elevation, ..., freq_new=freqs )
```

## Inputs:
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array; entries need
  not be sorted by frequency
- **`azimuth`** — Azimuth angles in GCS in rad; one row: the same angles for all outputs (planar wave), `n_out` rows:
  one angle per output (spherical wave); `(1, n_ang)` or `(n_out, n_ang)`
- **`elevation`** — Elevation angles in GCS in rad; same shape as `azimuth`
- **`i_element`** — Element indices, 0-based, duplicates allowed; `(n_out,)`; default: `None` (all elements in order,
  `n_out = n_elements`)
- **`orientation`** — Euler angles (bank, tilt, heading) of the array, for all outputs and angles, per output, per
  angle, or both, applied at all frequencies; `(3, 1)`, `(3, n_out)`, `(3, 1, n_ang)` or `(3, n_out, n_ang)`; default:
  `None` (no rotation)
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; multi-frequency mode: checked for
  its shape only, since no output depends on the positions; `(3, n_out)`; default: `None` (the positions of the selected
  elements)
- **`freq_new`** — Target frequencies in Hz, finite; selects the multi-frequency mode, in which a 3D pattern is the same
  at every frequency; ascending or descending order is fastest; `(n_freq,)`; default: `None` (the `center_freq` of the
  entries in order)
- **`dist`** — Also return the projected distance, at extra time and memory; single-frequency mode only; default:
  `False`
- **`local_angles`** — Also return the local angles and the polarization rotation angle, at extra time and memory;
  single-frequency mode only; default: `False`
- **`fast_access`** — Require the pattern arrays to be float64 and Fortran-contiguous, so that they are read without a
  copy; raises otherwise. With `False`, such arrays are read without a copy as well and others are copied; default:
  `False`
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (raises if unavailable or if the pattern has more than
  $2^{31} - 1$ samples); other values raise; `int`; default: 0

## Outputs:
- **`V`** — e-theta (vertical) field, complex; `(n_out, n_ang)`, multi-frequency mode `(n_out, n_ang, n_freq)`
- **`H`** — e-phi (horizontal) field, complex; same shape as `V`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the array; negative for elements
  closer to the source; `(n_out, n_ang)`; returned when `dist` is `True`
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `(n_out, n_ang)`; returned
  when `local_angles` is `True`
- **`gamma`** — Polarization rotation angle from the global to the local frame; `(n_out, n_ang)`; returned when
  `local_angles` is `True`

## Notes:
- **Interpolation**: linear where neighboring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- **Frequency interpolation** (multi-frequency mode): spherical interpolation of the complex values over frequency,
  linear where the two values have nearly opposite phases or one of them is zero. Target frequencies outside the range
  of the center frequencies take the nearest entry (no extrapolation). Consecutive target frequencies with the same
  bracketing entries reuse the spatial results
- **Angles**: not range-checked; values outside $[-\pi, \pi]$ and $[-\pi/2, \pi/2]$ denote the same direction; an
  elevation beyond $\pm\pi/2$ crosses the pole, so e-theta and e-phi have the opposite sign of the same direction given
  in range. NaN or infinite angles give NaN in all outputs of that angle and output, without an error. `azimuth` of
  shape `(1, 0)` or `(n_out, 0)` gives empty outputs; `np.array([])` raises `ValueError`
- **Poles**: on a pole of the local frame (a direction along the z-axis of the oriented array) the local azimuth is
  chosen so that the local and the global theta unit vectors agree: the polarization is not rotated there
- **Kernels**: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the
  peak field value and 2e-6 rad in the angles, and near a pole of the local frame up to about 2e-7 / d rad at a distance
  d (in rad). Auto currently selects GENERIC

## See also:
- [[apply_port_weights]] (patterns per port instead of per element)
- [[rotate_pattern]] (applies an orientation to the stored patterns)
- [[join_arrays]], [[generate_speaker]] (build multi-element, multi-frequency arrays)
- [[channel.spherical]] (channel coefficients from the interpolated patterns)
MD!*/

py::tuple interpolate_pattern(const py::dict &arrayant,
                              const py::array_t<double> &azimuth,
                              const py::array_t<double> &elevation,
                              py::handle i_element,
                              py::handle orientation,
                              py::handle element_pos,
                              py::handle freq_new,
                              py::handle dist,
                              py::handle local_angles,
                              py::handle fast_access,
                              int use_kernel)
{
    const bool want_dist = qd_python_flag(dist, false, "dist");
    const bool want_local_angles = qd_python_flag(local_angles, false, "local_angles");
    const bool strict = qd_python_flag(fast_access, false, "fast_access");

    const auto azimuth_a = qd_python_numpy2arma_Mat(azimuth, true);
    const auto elevation_a = qd_python_numpy2arma_Mat(elevation, true);
    const auto i_element_a = qd_python_numpy2arma_Col<arma::uword>(i_element, true, false, "i_element");
    const auto orientation_a = qd_python_numpy2arma_Cube<double>(orientation, true);
    const auto element_pos_a = qd_python_numpy2arma_Mat<double>(element_pos, true);
    const auto freq_new_a = qd_python_numpy2arma_Col<double>(freq_new, true);


    const auto ant = qd_python_dict2arrayant_multi(arrayant, true, strict);
    const arma::uword n_out = i_element_a.is_empty() ? ant[0].n_elements() : i_element_a.n_elem;
    const arma::uword n_ang = azimuth_a.n_cols;

    if (ant.size() > 1 || !freq_new_a.is_empty()) // Multi-frequency mode
    {
        if (want_dist || want_local_angles)
            throw std::invalid_argument("Outputs 'dist' and 'local_angles' are not available in the multi-frequency mode.");

        const arma::uword n_freq = freq_new_a.is_empty() ? (arma::uword)ant.size() : freq_new_a.n_elem;
        arma::cube V_re, V_im, H_re, H_im; // Left empty, the complex outputs replace them
        arma::cx_cube V, H;
        auto V_py = qd_python_init_output(n_out, n_ang, n_freq, &V);
        auto H_py = qd_python_init_output(n_out, n_ang, n_freq, &H);

        quadriga_lib::interpolate_pattern<double>(ant, azimuth_a, elevation_a, V_re, V_im, H_re, H_im,
                                                         freq_new_a,
                                                         i_element_a, orientation_a, element_pos_a,
                                                         &V, &H, use_kernel);
        return py::make_tuple(V_py, H_py);
    }

    arma::mat V_re, V_im, H_re, H_im, dist_a, azimuth_loc, elevation_loc, gamma; // V_re ... H_im left empty
    arma::cx_mat V, H;
    py::array_t<double> dist_py, azimuth_loc_py, elevation_loc_py, gamma_py;
    auto V_py = qd_python_init_output(n_out, n_ang, &V);
    auto H_py = qd_python_init_output(n_out, n_ang, &H);
    if (want_dist)
        dist_py = qd_python_init_output(n_out, n_ang, &dist_a);
    if (want_local_angles)
    {
        azimuth_loc_py = qd_python_init_output(n_out, n_ang, &azimuth_loc);
        elevation_loc_py = qd_python_init_output(n_out, n_ang, &elevation_loc);
        gamma_py = qd_python_init_output(n_out, n_ang, &gamma);
    }

    quadriga_lib::interpolate_pattern<double>(ant[0], azimuth_a, elevation_a, V_re, V_im, H_re, H_im,
                                               i_element_a, orientation_a, element_pos_a,
                                               want_dist ? &dist_a : nullptr,
                                               want_local_angles ? &azimuth_loc : nullptr,
                                               want_local_angles ? &elevation_loc : nullptr,
                                               want_local_angles ? &gamma : nullptr,
                                               &V, &H, use_kernel);

    // Output tuple in the documented order
    py::list out;
    out.append(V_py);
    out.append(H_py);
    if (want_dist)
        out.append(dist_py);
    if (want_local_angles)
    {
        out.append(azimuth_loc_py);
        out.append(elevation_loc_py);
        out.append(gamma_py);
    }
    return py::tuple(out);
}