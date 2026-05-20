// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COMBINE_PATTERN
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates the element field patterns, element positions, and coupling weights into one effective
  pattern per port (column of the coupling matrix)
- The result behaves as a virtual array with one element per port, zeroed element positions, and an
  identity coupling matrix
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning

## Usage:
```
% Input as struct (struct mode)
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in );
arrayant_out = quadriga_lib.arrayant_combine_pattern( arrayant_in, center_freq_new, azimuth_grid_new, elevation_grid_new );

% Separate outputs, struct input (single-freq only)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, ...
    coupling_im, center_freq, name] = quadriga_lib.arrayant_combine_pattern( arrayant_in );

% Separate inputs (split-mode, single-freq only)
arrayant_out = quadriga_lib.arrayant_combine_pattern( [], center_freq_new, azimuth_grid_new, elevation_grid_new, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name );
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`center_freq_new`** — Alternative frequency (grid) in Hz; if provided, the combined pattern is 
  recomputed (interpolated) at each requested frequency. If omitted or empty, the function uses each input 
   entry's `center_freq` unchanged.
- **`azimuth_grid_new`** — Alternative azimuth grid in rad; in [-pi, pi]; sorted; defaults to input grid
- **`elevation_grid_new`** — Alternative elevation grid in rad; in [-pi/2, pi/2]; sorted; defaults to input grid
- **`e_theta_re`** ... **`elevation_grid`** — Required inputs for slit-mode
- **`element_pos`** ... **`name`** — Optional inputs for slit-mode

## Outputs:
- **`arrayant_out`** — Arrayant struct (single-frequency result) or struct array (multi-frequency
  result); field layout as documented in [[arrayant_generate]]. Single struct when both input and
  `freq` describe a single frequency; struct array of size `numel(freq)` (or `numel(arrayant_in)`
  when `freq` is omitted) otherwise.
- **`e_theta_re`, ..., `name`** — Separate-field outputs; **only available for single-frequency
  results** (single-struct input with scalar/omitted `freq`, or separate-input mode).

## See also:
- [[arrayant_generate]] (for field layout in the arrayant struct)
- [[arrayant_rotate_pattern]] (for changing the orientation of elements before combining)
- [[arrayant_calc_beamwidth]] (calculates the beam width of array antennas)
- [[arrayant_calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || (nrhs > 4 && nrhs < 10) || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Common optional inputs
    const arma::vec freq = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
    const arma::vec az = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2]);
    const arma::vec el = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3]);

    // Dispatch
    const bool arrayant_is_struct = mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[0]) > 0;
    const bool separate_inputs = !arrayant_is_struct && nrhs >= 10;

    if (!arrayant_is_struct && !separate_inputs)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "First input must be an arrayant struct/struct array, or [] with separate inputs (>=10 args).");

    if (separate_inputs && freq.n_elem > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency mode (vector freq) requires struct input.");

    bool multifreq = false;
    if (arrayant_is_struct && freq.n_elem > 1)
        multifreq = true;
    else if (arrayant_is_struct && mxGetNumberOfElements(prhs[0]) > 1)
        multifreq = true;

    // Multi-frequency branch
    if (multifreq)
    {
        if (nlhs > 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Multi-frequency output supports only struct output (1 lhs).");

        // Helper validates internally; arrayant_combine_pattern_multi validates again — cheap.
        const auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0]);

        std::vector<quadriga_lib::arrayant<double>> out;
        CALL_QD(out = quadriga_lib::arrayant_combine_pattern_multi(ant_multi, &az, &el, &freq));

        if (out.size() == 1)
            plhs[0] = qd_mex_arrayant2struct(out[0]);
        else
            plhs[0] = qd_mex_arrayant2struct_multi(out);
        return;
    }

    // Single-frequency branch
    auto ant = quadriga_lib::arrayant<double>();
    if (arrayant_is_struct)
        ant = qd_mex_struct2arrayant(prhs[0]);
    else // separate_inputs (nrhs >= 10)
    {
        ant.e_theta_re = qd_mex_get_Cube<double>(prhs[4]);
        ant.e_theta_im = qd_mex_get_Cube<double>(prhs[5]);
        ant.e_phi_re = qd_mex_get_Cube<double>(prhs[6]);
        ant.e_phi_im = qd_mex_get_Cube<double>(prhs[7]);
        ant.azimuth_grid = qd_mex_get_Col<double>(prhs[8]);
        ant.elevation_grid = qd_mex_get_Col<double>(prhs[9]);
        if (nrhs > 10)
            ant.element_pos = qd_mex_get_Mat<double>(prhs[10]);
        if (nrhs > 11)
            ant.coupling_re = qd_mex_get_Mat<double>(prhs[11]);
        if (nrhs > 12)
            ant.coupling_im = qd_mex_get_Mat<double>(prhs[12]);
        if (nrhs > 13)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[13], "center_freq", 299792458.0);
        if (nrhs > 14)
            ant.name = qd_mex_get_string(prhs[14]);
    }

    // Optional scalar freq override (single-freq path only)
    if (freq.n_elem == 1)
        ant.center_frequency = freq[0];

    // Call member function
    auto arrayant_out = quadriga_lib::arrayant<double>();
    CALL_QD(arrayant_out = ant.combine_pattern(&az, &el));

    // Return to MATLAB
    if (nlhs == 1)
        plhs[0] = qd_mex_arrayant2struct(arrayant_out);
    else if (nlhs > 1)
    {
        plhs[0] = qd_mex_copy2matlab(&arrayant_out.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&arrayant_out.e_theta_im);
        if (nlhs > 2)
            plhs[2] = qd_mex_copy2matlab(&arrayant_out.e_phi_re);
        if (nlhs > 3)
            plhs[3] = qd_mex_copy2matlab(&arrayant_out.e_phi_im);
        if (nlhs > 4)
            plhs[4] = qd_mex_copy2matlab(&arrayant_out.azimuth_grid, true);
        if (nlhs > 5)
            plhs[5] = qd_mex_copy2matlab(&arrayant_out.elevation_grid, true);
        if (nlhs > 6)
            plhs[6] = qd_mex_copy2matlab(&arrayant_out.element_pos);
        if (nlhs > 7)
            plhs[7] = qd_mex_copy2matlab(&arrayant_out.coupling_re);
        if (nlhs > 8)
            plhs[8] = qd_mex_copy2matlab(&arrayant_out.coupling_im);
        if (nlhs > 9)
            plhs[9] = qd_mex_copy2matlab(&arrayant_out.center_frequency);
        if (nlhs > 10)
            plhs[10] = mxCreateString(arrayant_out.name.c_str());
    }
}
