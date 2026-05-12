// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_ROTATE_PATTERN
Rotate antenna radiation patterns around the principal axes using Euler rotations

- Rotates pattern and/or polarization around x (bank), y (tilt), z (heading) axes
- Rotations applied in order x, y, z, composed as Rz·Ry·Rx (intrinsic Tait-Bryan)
- Adjusts the sampling grid for non-uniformly sampled antennas when `usage` is 0 or 1
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid
  spurious polarization effects

## Usage:
```
% Struct in / struct out
arrayant_out = quadriga_lib.arrayant_rotate_pattern(arrayant_in, x_deg, y_deg, z_deg, usage, i_element);

% Separate-field outputs (single-frequency results only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name] = quadriga_lib.arrayant_rotate_pattern( ...
    arrayant_in, x_deg, y_deg, z_deg, usage, element);

% Separate inputs (single-frequency only)
arrayant_out = quadriga_lib.arrayant_rotate_pattern([], x_deg, y_deg, z_deg, usage, element, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_freq, name);
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array represents a frequency-dependent model
- **`x_deg`** *(optional)* — Rotation around x-axis (bank) in degrees; default: 0
- **`y_deg`** *(optional)* — Rotation around y-axis (tilt) in degrees; default: 0
- **`z_deg`** *(optional)* — Rotation around z-axis (heading) in degrees; default: 0
- **`usage`** *(optional)* — Rotation mode; default: 0
   | Mode | Pattern | Polarization | Grid adj. |
   | ---- | ------- | ------------ | --------- |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |
  Multi-frequency input accepts `usage` in {0, 1, 2} and never adjusts the grid (internally maps
  0 → 3 and 1 → 4 for uniform-grid consistency across frequencies).
- **`i_element`** *(optional)* — 1-based element indices to rotate; defaults to all elements; `[n]`

## Outputs:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [[arrayant_generate]]. Single struct when input is a
  single struct; struct array of size `numel(arrayant_in)` when input is a struct array.
- **`e_theta_re`, ..., `name`** — Separate-field outputs; **only available for single-frequency
  results** (single-struct input).
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts: struct mode = 1..6, separate-input mode (single-freq only) = 12..17
    if (nrhs < 1 || (nrhs > 6 && nrhs < 12) || nrhs > 17)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse scalar rotation/usage arguments
    const double x_deg = (nrhs < 2) ? 0.0 : qd_mex_get_scalar<double>(prhs[1], "x_deg", 0.0);
    const double y_deg = (nrhs < 3) ? 0.0 : qd_mex_get_scalar<double>(prhs[2], "y_deg", 0.0);
    const double z_deg = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "z_deg", 0.0);
    const unsigned usage = (nrhs < 5) ? 0 : qd_mex_get_scalar<unsigned>(prhs[4], "usage", 0);

    // Dispatch
    const bool struct_input = mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[0]) > 0;
    const bool separate_inputs = !struct_input && nrhs >= 12;

    if (!struct_input && !separate_inputs)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "First input must be an arrayant struct/struct array, or [] with separate inputs (>=12 args).");

    const bool multifreq = struct_input && mxGetNumberOfElements(prhs[0]) > 1;

    // Element indices: 1-based MATLAB → 0-based C++ (empty = all elements)
    arma::uvec i_element = (nrhs < 6) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[5], true);

    if (arma::any(i_element == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
    else // Convert to 0-based
        i_element -= 1;

    // Multi-frequency branch
    if (multifreq)
    {
        if (nlhs > 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency output supports only struct output.");

        auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], true, true);
        CALL_QD(quadriga_lib::arrayant_rotate_pattern_multi(ant_multi, x_deg, y_deg, z_deg, usage, i_element));

        plhs[0] = qd_mex_arrayant2struct_multi(ant_multi);
        return;
    }

    // Single-frequency branch
    auto ant = quadriga_lib::arrayant<double>();
    if (struct_input)
        ant = qd_mex_struct2arrayant(prhs[0], true, true);
    else // separate_inputs (nrhs >= 12)
    {
        ant.e_theta_re = qd_mex_get_Cube<double>(prhs[6], true);
        ant.e_theta_im = qd_mex_get_Cube<double>(prhs[7], true);
        ant.e_phi_re = qd_mex_get_Cube<double>(prhs[8], true);
        ant.e_phi_im = qd_mex_get_Cube<double>(prhs[9], true);
        ant.azimuth_grid = qd_mex_get_Col<double>(prhs[10], true);
        ant.elevation_grid = qd_mex_get_Col<double>(prhs[11], true);
        if (nrhs > 12)
            ant.element_pos = qd_mex_get_Mat<double>(prhs[12], true);
        if (nrhs > 13)
            ant.coupling_re = qd_mex_get_Mat<double>(prhs[13], true);
        if (nrhs > 14)
            ant.coupling_im = qd_mex_get_Mat<double>(prhs[14], true);
        if (nrhs > 15)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[15], "center_freq", 299792458.0);
        if (nrhs > 16)
            ant.name = qd_mex_get_string(prhs[16]);
    }

    if (i_element.n_elem == 0)
        CALL_QD(ant.rotate_pattern(x_deg, y_deg, z_deg, usage));
    else
        for (auto el : i_element)
            CALL_QD(ant.rotate_pattern(x_deg, y_deg, z_deg, usage, (unsigned)el));

    // Return to MATLAB
    if (nlhs == 1)
        plhs[0] = qd_mex_arrayant2struct(ant);
    else if (nlhs == 11)
    {
        plhs[0] = qd_mex_copy2matlab(&ant.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&ant.e_theta_im);
        plhs[2] = qd_mex_copy2matlab(&ant.e_phi_re);
        plhs[3] = qd_mex_copy2matlab(&ant.e_phi_im);
        plhs[4] = qd_mex_copy2matlab(&ant.azimuth_grid, true);
        plhs[5] = qd_mex_copy2matlab(&ant.elevation_grid, true);
        plhs[6] = qd_mex_copy2matlab(&ant.element_pos);
        plhs[7] = qd_mex_copy2matlab(&ant.coupling_re);
        plhs[8] = qd_mex_copy2matlab(&ant.coupling_im);
        plhs[9] = qd_mex_copy2matlab(&ant.center_frequency);
        plhs[10] = mxCreateString(ant.name.c_str());
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}