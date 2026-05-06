// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_CALC_DIRECTIVITY
Calculates the directivity in dBi of array antenna elements

- Directivity = 10 log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling

## Usage:

```
% Input as struct (struct mode)
directivity = quadriga_lib.arrayant_calc_directivity(arrayant);
directivity = quadriga_lib.arrayant_calc_directivity(arrayant, i_element);

% Separate inputs (split mode)
directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid);

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, ...
    e_phi_im, azimuth_grid, elevation_grid, i_element);
```

## Inputs (struct mode):
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`i_element`** *(optional)* — Element index; 1-based; if not provided or empty, the directivity is
  calculated for all elements; uint64; `[n_out]` or empty

## Inputs (split mode):
- **`e_theta_re`** — e-theta field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, sorted; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, sorted; `[n_elevation]`
- **`i_element`** *(optional)* — Element index; 1-based; if not provided or empty, the directivity is
  calculated for all elements; uint64; `[n_out]` or empty

## Output Argument:
- **`directivity`** - Directivity of the antenna pattern in dBi; `[n_out, n_freq]`;
  with `n_out = n_elements` when `i_element` is omitted/empty.

## See also:
- [[arrayant_combine_pattern]] (to apply element coupling before calculating directivity)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (!(nrhs == 1 || nrhs == 2 || nrhs == 6 || nrhs == 7))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    bool struct_mode = mxIsStruct(prhs[0]);
    size_t n_freq = 1;

    if (struct_mode) // Struct input
    {
        if (nrhs > 2)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Cannot mix struct input with separate arrayant inputs.");

        n_freq = (size_t)mxGetNumberOfElements(prhs[0]);
        if (n_freq > 1)
            ant_multi = qd_mex_struct2arrayant_multi(prhs[0], true);
        else
            ant = qd_mex_struct2arrayant(prhs[0], true);
    }
    else if (nrhs >= 6) // Separate inputs
    {
        ant.e_theta_re = qd_mex_get_Cube<double>(prhs[0]);
        ant.e_theta_im = qd_mex_get_Cube<double>(prhs[1]);
        ant.e_phi_re = qd_mex_get_Cube<double>(prhs[2]);
        ant.e_phi_im = qd_mex_get_Cube<double>(prhs[3]);
        ant.azimuth_grid = qd_mex_get_Col<double>(prhs[4]);
        ant.elevation_grid = qd_mex_get_Col<double>(prhs[5]);
        ant.validate();
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input must be a struct.");

    // Read n_elements
    arma::uword n_elements = (n_freq > 1) ? ant_multi[0].e_theta_re.n_slices : ant.e_theta_re.n_slices;
    if (n_elements == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Array antenna has no elements.");

    // Read i_element (1-based index)
    arma::uvec element_ind;
    if (struct_mode)
        element_ind = (nrhs < 2) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[1], "i_element");
    else
        element_ind = (nrhs < 7) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[6], "i_element");

    if (element_ind.empty()) // Set defaults
        element_ind = arma::regspace<arma::uvec>(0, n_elements - 1);
    else if (arma::any(element_ind == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
    else // Convert to 0-based
        element_ind -= 1;

    arma::mat directivity;
    plhs[0] = qd_mex_init_output(&directivity, element_ind.n_elem, (arma::uword)n_freq);

    if (n_freq > 1)
    {
        for (size_t i_freq = 0; i_freq < n_freq; ++i_freq)
        {
            auto *p_directivity = directivity.colptr(i_freq);
            for (auto el : element_ind)
                CALL_QD(*p_directivity++ = ant_multi[i_freq].calc_directivity_dBi(el));
        }
    }
    else
    {
        auto *p_directivity = directivity.memptr();
        for (auto el : element_ind)
            CALL_QD(*p_directivity++ = ant.calc_directivity_dBi(el));
    }
}
