// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_INTERPOLATE
Interpolate polarimetric array antenna field patterns (single- and multi-frequency)

- Interpolates complex e-theta (V) and e-phi (H) field components at the requested
  azimuth / elevation angles.
- Single-frequency mode: pass a 1-element struct (or arrayant data via separate inputs);
  returns up to 8 outputs including the optional `dist`, `azimuth_loc`, `elevation_loc`,
  and `gamma` as matrices of size `[n_out, n_ang]`
- Multi-frequency mode is selected automatically when `arrayant` is a struct array with
  more than one element or when `freq` is non-empty; for each target frequency, the two
  bracketing `center_freq` entries are located and blended via SLERP.
- Separate arrayant inputs are accepted in single-frequency mode only

## Usage:
```
% Single-frequency, struct input
[V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
    quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, element, orientation, element_pos );

% Single-frequency, separate arrayant inputs
[V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
    quadriga_lib.arrayant_interpolate( [], azimuth, elevation, element, orientation, element_pos, [], ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid );

% Multi-frequency, struct array input
[V_re, V_im, H_re, H_im] = ...
    quadriga_lib.arrayant_interpolate( arrayant_multi, azimuth, elevation, element, orientation, element_pos, freq );
```

## Inputs:
- **`arrayant`** *(optional)* — Struct (single-frequency) or struct array (multi-frequency)
  containing the arrayant data; field layout as in [[generate_arrayant]]. Pass `[]` to provide
  the data via separate inputs (single-frequency only)
- **`azimuth`** — Azimuth angles in rad, in [-π, π]; single or double precision;
  `[1, n_ang]` for planar-wave mode (same angles for all elements) or `[n_out, n_ang]` for
  per-element angles (spherical-wave mode)
- **`elevation`** — Elevation angles in rad, in [-π/2, π/2]; single or double; shape must
  match `azimuth`
- **`element`** *(optional)* — 1-based element indices to interpolate; duplicates allowed;
  defaults to `[1:n_elements]` when empty; `[1, n_out]`, `[n_out, 1]`, or `[]`; uint32 or double
- **`orientation`** *(optional)* — Antenna orientation (bank, tilt, heading) in rad; East is
  the default broadside; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]`, `[3, n_out, n_ang]`, or `[]`
- **`element_pos`** *(optional)* — Override element positions in m; `[3, n_out]` or `[]`;
  falls back to `arrayant.element_pos` (or zeros) when empty
- **`freq`** *(optional, struct input only)* — Target frequencies in Hz; `[n_freq]`. When passing a
  struct array, `freq` may be omitted or `[]`, in which case the `center_freq` values of the struct
  array entries are used as target frequencies (no interpolation between bands, one output slice per entry).

## Inputs (separate arrayant data, required when `arrayant` is `[]`, single-frequency only):
- **`e_theta_re`** — e-theta real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth sample grid in rad, sorted, in [-π, π]; `[n_azimuth]`
- **`elevation_grid`** — Elevation sample grid in rad, sorted, in [-π/2, π/2]; `[n_elevation]`

## Derived sizes:
  `n_azimuth`      | Number of azimuth samples in the pattern
  `n_elevation`    | Number of elevation samples in the pattern
  `n_elements`     | Number of antenna elements in the pattern
  `n_ang`          | Number of interpolation angles
  `n_out`          | Number of output elements (`n_elements` if `element` is empty, else `numel(element)`)
  `n_freq`         | Number of target frequencies (multi-frequency mode only)

## Outputs:
- **`V_re`** — Real part of the interpolated e-theta (vertical) field component;
  `[n_out, n_ang]` in single-freq mode, `[n_out, n_ang, n_freq]` in multi-freq mode
- **`V_im`** — Imaginary part of the e-theta component; same size as `V_re`
- **`H_re`** — Real part of the interpolated e-phi (horizontal) field component; same size as `V_re`
- **`H_im`** — Imaginary part of the e-phi component; same size as `V_re`
- **`dist`** *(single-frequency only)* — Effective distances between the antenna elements
  projected onto the wavefront plane; used for phase computation; `[n_out, n_ang]`
- **`azimuth_loc`** *(single-frequency only)* — Azimuth angles in the local (rotated) element
  frame in rad; `[n_out, n_ang]`
- **`elevation_loc`** *(single-frequency only)* — Elevation angles in the local element frame in
  rad; `[n_out, n_ang]`
- **`gamma`** *(single-frequency only)* — Polarization rotation angles in rad; `[n_out, n_ang]`

## See also:
- [[qdant_read]] / [[qdant_write]] (load / save arrayant data)
- [[generate_arrayant]] (arrayant struct layout)
- [[generate_speaker]] (typical multi-frequency struct array source)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Accepted input count: struct mode = [3, 7], separate single-freq mode = 13
    if (nrhs < 3 || (nrhs > 7 && nrhs != 13))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (nlhs == 0)
        return;

    // Common inputs
    const auto az = qd_mex_get_Mat<double>(prhs[1]);
    const auto el = qd_mex_get_Mat<double>(prhs[2]);

    if (az.n_elem == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Azimuth angles cannot be empty.");

    const arma::uvec element_ind = (nrhs > 3) ? qd_mex_typecast_Col<arma::uword>(prhs[3]) - 1 : arma::uvec();
    const auto ori = (nrhs > 4) ? qd_mex_get_Cube<double>(prhs[4]) : arma::cube();
    const auto elpos = (nrhs > 5) ? qd_mex_get_Mat<double>(prhs[5]) : arma::mat();

    // Copy required: may be resized below when derived from center_frequency
    auto freq = (nrhs > 6) ? qd_mex_get_Col<double>(prhs[6], true) : arma::vec();

    // Dispatch logic
    const bool arrayant_is_struct = mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[0]) > 0;
    bool multifreq = false;
    if (arrayant_is_struct && !freq.empty())
        multifreq = true;
    else if (arrayant_is_struct && mxGetNumberOfElements(prhs[0]) > 1)
        multifreq = true;

    // Multi-freq branch
    if (multifreq)
    {
        if (nlhs > 4)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency mode supports at most 4 outputs (V_re, V_im, H_re, H_im).");

        const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], true);

        if (freq.empty())
        {
            freq.set_size(ant_multi.size());
            for (size_t i = 0; i < ant_multi.size(); ++i)
                freq[i] = ant_multi[i].center_frequency;
        }

        const arma::uword n_ang = az.n_cols;
        const arma::uword n_out = (element_ind.n_elem == 0) ? ant_multi[0].n_elements() : element_ind.n_elem;
        const arma::uword n_freq = freq.n_elem;

        arma::cube V_re, V_im, H_re, H_im;
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&V_re, n_out, n_ang, n_freq);
        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&V_im, n_out, n_ang, n_freq);
        if (nlhs > 2)
            plhs[2] = qd_mex_init_output(&H_re, n_out, n_ang, n_freq);
        if (nlhs > 3)
            plhs[3] = qd_mex_init_output(&H_im, n_out, n_ang, n_freq);

        // Validation already done in qd_mex_struct2arrayant_multi(..., true)
        CALL_QD(quadriga_lib::arrayant_interpolate_multi(ant_multi, &az, &el, &freq,
                                                         &V_re, &V_im, &H_re, &H_im,
                                                         element_ind, &ori, &elpos, false));
        return;
    }

    // Single-freq branch
    auto ant = quadriga_lib::arrayant<double>();
    if (arrayant_is_struct)
        ant = qd_mex_struct2arrayant(prhs[0]);
    else if (nrhs == 13)
    {
        ant.e_theta_re = qd_mex_get_Cube<double>(prhs[7]);
        ant.e_theta_im = qd_mex_get_Cube<double>(prhs[8]);
        ant.e_phi_re = qd_mex_get_Cube<double>(prhs[9]);
        ant.e_phi_im = qd_mex_get_Cube<double>(prhs[10]);
        ant.azimuth_grid = qd_mex_get_Col<double>(prhs[11]);
        ant.elevation_grid = qd_mex_get_Col<double>(prhs[12]);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Allocate outputs
    const arma::uword n_ang = az.n_cols;
    const arma::uword n_out = (element_ind.n_elem == 0) ? ant.n_elements() : element_ind.n_elem;

    arma::mat V_re, V_im, H_re, H_im, dist_proj, azimuth_loc, elevation_loc, gamma;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&V_re, n_out, n_ang);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&V_im, n_out, n_ang);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&H_re, n_out, n_ang);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&H_im, n_out, n_ang);
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&dist_proj, n_out, n_ang);
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&azimuth_loc, n_out, n_ang);
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&elevation_loc, n_out, n_ang);
    if (nlhs > 7)
        plhs[7] = qd_mex_init_output(&gamma, n_out, n_ang);

    // Interpolate data
    if (nlhs > 5)
        CALL_QD(ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj, &azimuth_loc, &elevation_loc, &gamma));
    else if (nlhs > 4)
        CALL_QD(ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj));
    else
        CALL_QD(ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos));
}