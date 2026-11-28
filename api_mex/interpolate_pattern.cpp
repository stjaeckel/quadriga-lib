// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# INTERPOLATE_PATTERN
Interpolate the polarimetric field pattern of an antenna array at given azimuth and elevation angles

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the array and the resulting polarization rotation applied
- Between the samples of the pattern, amplitude and phase are interpolated separately (spherical interpolation)
- Multi-frequency mode (a struct array, or target frequencies given): the two entries whose center frequencies bracket
  each target frequency are interpolated and blended

## Usage:
```
% Single-frequency mode
[ V, H, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.interpolate_pattern( ...
    arrayant, azimuth, elevation, i_element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V, H ] = quadriga_lib.interpolate_pattern( arrayant, azimuth, elevation, ...
    i_element, orientation, element_pos, freq_new, use_kernel );
```

## Inputs:
- **`arrayant`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array; entries need
  not be sorted by frequency
- **`azimuth`** — Azimuth angles in GCS in rad; one row: the same angles for all outputs (planar wave), `n_out` rows:
  one angle per output (spherical wave); `[1, n_ang]` or `[n_out, n_ang]`
- **`elevation`** — Elevation angles in GCS in rad; same size as `azimuth`
- **`i_element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements in order
  (`n_out = n_elements`)
- **`orientation`** — Euler angles (bank, tilt, heading) of the array, for all outputs and angles, per output, per
  angle, or both, applied at all frequencies; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]` or `[3, n_out, n_ang]`; default:
  no rotation
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; multi-frequency mode: checked for
  its size only, since no output depends on the positions; `[3, n_out]`; default: the positions of the selected elements
- **`freq_new`** — Target frequencies in Hz, finite; selects the multi-frequency mode; ascending or descending order is
  fastest; `[n_freq]`; default: the center frequencies of the entries in order
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (error if unavailable or if the pattern has more than
  2^31 - 1 samples); other values raise an error; default: 0

## Outputs:
- **`V`** — Complex e-theta (vertical) field; `[n_out, n_ang]`, multi-frequency mode `[n_out, n_ang, n_freq]`
- **`H`** — Complex e-phi (horizontal) field; same size as `V`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the array; negative for elements
  closer to the source; `[n_out, n_ang]`; single-frequency mode only
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `[n_out, n_ang]`;
  single-frequency mode only
- **`gamma`** — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`; single-frequency mode
  only

## Notes:
- **Interpolation**: linear where neighboring samples have nearly opposite phases; the azimuth grid wraps around,
  elevations outside the grid take the values of the first or last grid row
- **Frequency interpolation** (multi-frequency mode): spherical interpolation of the complex values over frequency,
  linear where the two values have nearly opposite phases or one of them is zero. Target frequencies outside the range
  of the center frequencies take the nearest entry (no extrapolation). Consecutive target frequencies with the same
  bracketing entries reuse the spatial results
- **Angles**: not range-checked; values outside [-π, π] and [-π/2, π/2] denote the same direction; an elevation beyond
  ±π/2 crosses the pole, so e-theta and e-phi have the opposite sign of the same direction given in range. NaN or
  infinite angles give NaN in all outputs of that angle and output, without an error. `azimuth` of size `[1, 0]` or
  `[n_out, 0]` gives empty outputs; `[]` raises an error
- **Poles**: on a pole of the local frame (a direction along the z-axis of the oriented array) the local azimuth is
  chosen so that the local and the global theta unit vectors agree: the polarization is not rotated there
- **Kernels**: GENERIC computes in double precision; AVX2 in single precision, with errors of up to about 2e-6 of the
  peak field value and 2e-6 rad in the angles, and near a pole of the local frame up to about 2e-7 / d rad at a distance
  d (in rad). Auto currently selects GENERIC

## See also:
- [[apply_port_weights]] (patterns per port instead of per element)
- [[rotate_pattern]] (applies an orientation to the stored patterns)
- [[join_arrays]], [[generate_speaker]] (build multi-element, multi-frequency arrays)
- [[get_channels_spherical]] (channel coefficients from the interpolated patterns)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto ant = qd_mex_struct2arrayant_multi(prhs[0], false, false, "arrayant");
    if (ant.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' cannot be empty.");
    const auto az = qd_mex_get_Mat<double>(prhs[1], false, "azimuth");
    const auto el = qd_mex_get_Mat<double>(prhs[2], false, "elevation");

    // Element indices: 1-based (MATLAB) -> 0-based (C++), empty = all elements
    const arma::uvec i_element = (nrhs < 4) ? arma::uvec() : qd_mex_get_index_Col(prhs[3], "i_element");

    const auto orientation = (nrhs < 5) ? arma::cube() : qd_mex_get_Cube<double>(prhs[4], false, "orientation");
    const auto element_pos = (nrhs < 6) ? arma::mat() : qd_mex_get_Mat<double>(prhs[5], false, "element_pos");
    const auto freq_new = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6], false, "freq_new");
    const int use_kernel = (nrhs < 8) ? 0 : qd_mex_get_scalar<int>(prhs[7], "use_kernel", 0);

    const arma::uword n_out = i_element.is_empty() ? ant[0].n_elements() : i_element.n_elem;
    const arma::uword n_ang = az.n_cols;

    if (ant.size() > 1 || !freq_new.is_empty()) // Multi-frequency mode
    {
        if (nlhs > 2)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency mode supports at most 2 outputs (V, H).");

        // C++ writes the complex fields into the MATLAB arrays
        const arma::uword n_freq = freq_new.is_empty() ? (arma::uword)ant.size() : freq_new.n_elem;
        arma::cube V_re, V_im, H_re, H_im;
        arma::cx_cube V, H;
        plhs[0] = qd_mex_init_output(&V, n_out, n_ang, n_freq);
        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&H, n_out, n_ang, n_freq);

        CALL_QD(quadriga_lib::interpolate_pattern<double>(ant, az, el, V_re, V_im, H_re, H_im,
                                                                 freq_new, i_element, orientation, element_pos,
                                                                 &V, (nlhs > 1) ? &H : nullptr, use_kernel));
        return;
    }

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    arma::cx_mat V, H;
    plhs[0] = qd_mex_init_output(&V, n_out, n_ang);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&H, n_out, n_ang);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&dist, n_out, n_ang);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&azimuth_loc, n_out, n_ang);
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&elevation_loc, n_out, n_ang);
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&gamma, n_out, n_ang);

    CALL_QD(quadriga_lib::interpolate_pattern<double>(ant[0], az, el, V_re, V_im, H_re, H_im, i_element, orientation, element_pos,
                                                       (nlhs > 2) ? &dist : nullptr, (nlhs > 3) ? &azimuth_loc : nullptr,
                                                       (nlhs > 4) ? &elevation_loc : nullptr, (nlhs > 5) ? &gamma : nullptr,
                                                       &V, (nlhs > 1) ? &H : nullptr, use_kernel));
}
