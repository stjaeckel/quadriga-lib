// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_INTERPOLATE
Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles

- Returns the complex e-theta (vertical) and e-phi (horizontal) field of the selected elements for directions in the
  global coordinate system (GCS), with the orientation of the antenna and the resulting polarization rotation applied
- Single-frequency mode (a 1x1 struct and no `frequency`): up to 8 outputs. Multi-frequency mode (a struct array with
  more than one entry, or `frequency` given): for each target frequency, the two entries whose center frequencies
  bracket it are interpolated and blended; frequencies outside the entries take the nearest entry; up to 4 outputs
- `azimuth` of size `[1, 0]` or `[n_out, 0]` gives empty outputs; `[]` raises an error
- NaN or infinite angles give NaN in all outputs of that angle and output, without an error
- Kernels: GENERIC computes in double precision; AVX2 in single precision, with errors of up to a few 1e-5 of the peak
  field value and about 1e-4 rad in the angles (more near the poles of the local frame); auto selects GENERIC in this
  version, which may change in a later version

## Usage:
```
% Single-frequency mode
[ V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma ] = quadriga_lib.arrayant_interpolate( ...
    arrayant, azimuth, elevation, element, orientation, element_pos, [], use_kernel );

% Multi-frequency mode
[ V_re, V_im, H_re, H_im ] = quadriga_lib.arrayant_interpolate( arrayant, azimuth, elevation, ...
    element, orientation, element_pos, frequency, use_kernel );
```

## Inputs:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]]; a struct
  array may contain a frequency-dependent model
- **`azimuth`** — Azimuth angles in GCS; `[1, n_ang]` (same angles for all outputs) or `[n_out, n_ang]` (one angle per
  output)
- **`elevation`** — Elevation angles in GCS; same size as `azimuth`
- **`element`** — Element indices, 1-based, duplicates allowed; `[n_out]`; default: all elements
- **`orientation`** — Euler angles (bank, tilt, heading) of the antenna; `[3, 1]`, `[3, n_out]`, `[3, 1, n_ang]` or
  `[3, n_out, n_ang]`; default: no rotation
- **`element_pos`** — Element positions in local antenna coordinates, used for `dist`; `[3, n_out]`; default: the
  positions of the arrayant
- **`frequency`** — Target frequencies in Hz; selects the multi-frequency mode; `[n_freq]`; default: the `center_freq`
  of the entries
- **`use_kernel`** — Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

## Outputs:
- **`V_re`** / **`V_im`** — e-theta (vertical) field, real / imaginary part; `[n_out, n_ang]`; multi-frequency mode:
  `[n_out, n_ang, n_freq]`
- **`H_re`** / **`H_im`** — e-phi (horizontal) field, real / imaginary part; same size as `V_re`
- **`dist`** — Distance of each element from the wavefront plane through the origin of the antenna; negative for
  elements closer to the source; `[n_out, n_ang]`; single-frequency mode only
- **`azimuth_loc`** / **`elevation_loc`** — Azimuth and elevation in the local antenna frame; `[n_out, n_ang]`;
  single-frequency mode only
- **`gamma`** — Polarization rotation angle from the global to the local frame; `[n_out, n_ang]`; single-frequency
  mode only

## See also:
- [[arrayant_generate]] (arrayant struct layout)
- [[arrayant_qdant_read]] (load arrayant data)
- [[generate_speaker]] (typical source of multi-frequency struct arrays)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto ant = qd_mex_struct2arrayant_multi(prhs[0]);
    if (ant.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' cannot be empty.");
    const auto az = qd_mex_get_Mat<double>(prhs[1], false, "azimuth");
    const auto el = qd_mex_get_Mat<double>(prhs[2], false, "elevation");

    // Element indices: 1-based (MATLAB) -> 0-based (C++), empty = all elements
    arma::uvec element = (nrhs < 4) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[3], true, "element");
    if (arma::any(element == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'element' cannot be 0 (1-based index).");
    element -= 1;

    const auto orientation = (nrhs < 5) ? arma::cube() : qd_mex_get_Cube<double>(prhs[4], false, "orientation");
    const auto element_pos = (nrhs < 6) ? arma::mat() : qd_mex_get_Mat<double>(prhs[5], false, "element_pos");
    const auto frequency = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6], false, "frequency");
    const int use_kernel = (nrhs < 8) ? 0 : qd_mex_get_scalar<int>(prhs[7], "use_kernel", 0);

    const arma::uvec *p_element = element.is_empty() ? nullptr : &element;
    const arma::cube *p_orientation = orientation.is_empty() ? nullptr : &orientation;
    const arma::mat *p_element_pos = element_pos.is_empty() ? nullptr : &element_pos;

    const arma::uword n_out = element.is_empty() ? ant[0].n_elements() : element.n_elem;
    const arma::uword n_ang = az.n_cols;

    if (ant.size() > 1 || !frequency.is_empty()) // Multi-frequency mode
    {
        if (nlhs > 4)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency mode supports at most 4 outputs (V_re, V_im, H_re, H_im).");

        const arma::uword n_freq = frequency.is_empty() ? (arma::uword)ant.size() : frequency.n_elem;
        arma::cube V_re, V_im, H_re, H_im;
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&V_re, n_out, n_ang, n_freq);
        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&V_im, n_out, n_ang, n_freq);
        if (nlhs > 2)
            plhs[2] = qd_mex_init_output(&H_re, n_out, n_ang, n_freq);
        if (nlhs > 3)
            plhs[3] = qd_mex_init_output(&H_im, n_out, n_ang, n_freq);

        CALL_QD(quadriga_lib::arrayant_interpolate_multi<double>(ant, az, el, V_re, V_im, H_re, H_im,
                                                                 frequency.is_empty() ? nullptr : &frequency,
                                                                 p_element, p_orientation, p_element_pos, use_kernel));
        return;
    }

    arma::mat V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&V_re, n_out, n_ang);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&V_im, n_out, n_ang);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&H_re, n_out, n_ang);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&H_im, n_out, n_ang);
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&dist, n_out, n_ang);
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&azimuth_loc, n_out, n_ang);
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&elevation_loc, n_out, n_ang);
    if (nlhs > 7)
        plhs[7] = qd_mex_init_output(&gamma, n_out, n_ang);

    CALL_QD(quadriga_lib::arrayant_interpolate<double>(ant[0], az, el, V_re, V_im, H_re, H_im, p_element, p_orientation, p_element_pos,
                                                       (nlhs > 4) ? &dist : nullptr, (nlhs > 5) ? &azimuth_loc : nullptr,
                                                       (nlhs > 6) ? &elevation_loc : nullptr, (nlhs > 7) ? &gamma : nullptr,
                                                       use_kernel));
}
