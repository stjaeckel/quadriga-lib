// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_CALC_BEAMWIDTH
Calculates the beam width of array antenna elements in degree

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern (≈100x finer grid in
  each direction than the antenna sampling grid)
- Calculated per element, not per port; ignores element coupling

## Usage:
```
% Input as struct (struct mode)
[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant );

[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = quadriga_lib.arrayant_calc_beamwidth( arrayant, i_element, threshold_dB );

% Separate inputs (split mode)
[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = ...
    quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid );

[ beamwidth_az, beamwidth_el, az_point_ang, el_point_ang ] = ...
    quadriga_lib.arrayant_calc_beamwidth( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, ...
    i_element, threshold_dB );
```

## Inputs (struct mode):
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`i_element`** — Element index; 1-based; if not provided or empty, all elements are used; uint64; `[n_out]` or empty
- **`threshold_dB`** — Threshold in dB; default: 3 (equivalent to FWHM)

## Inputs (split mode):
- **`e_theta_re`** — e-theta field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_theta_im`** — e-theta field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_re`** — e-phi field component, real part; `[n_elevation, n_azimuth, n_elements]`
- **`e_phi_im`** — e-phi field component, imaginary part; `[n_elevation, n_azimuth, n_elements]`
- **`azimuth_grid`** — Azimuth angles in rad, -π to π, sorted; `[n_azimuth]`
- **`elevation_grid`** — Elevation angles in rad, -π/2 to π/2, sorted; `[n_elevation]`
- **`i_element`** — Element index; 1-based; if not provided or empty, all elements are used; uint64; `[n_out]` or empty
- **`threshold_dB`** — Threshold in dB; default: 3 (equivalent to FWHM)

## Outputs:
- **`beamwidth_az`** — Azimuth beamwidth in degree; `[n_out, n_freq]`; with `n_out = n_elements` when `i_element` is omitted/empty
- **`beamwidth_el`** — Elevation beamwidth in degree; `[n_out, n_freq]`
- **`az_point_ang`** — Azimuth pointing angle of the main beam in degree; `[n_out, n_freq]`
- **`el_point_ang`** — Elevation pointing angle of the main beam in degree; `[n_out, n_freq]`

## See also:
- [[arrayant_combine_pattern]] (to apply element coupling before calculating directivity)
- [[arrayant_calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!(nrhs == 1 || nrhs == 2 || nrhs == 3 || nrhs == 6 || nrhs == 7 || nrhs == 8))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    bool struct_mode = mxIsStruct(prhs[0]);
    size_t n_freq = 1;

    if (struct_mode) // Struct input
    {
        if (nrhs > 3)
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

    // Read threshold_dB
    double threshold_dB;
    if (struct_mode)
        threshold_dB = (nrhs < 3) ? 3.0 : qd_mex_get_scalar<double>(prhs[2], "threshold_dB", 3.0);
    else
        threshold_dB = (nrhs < 8) ? 3.0 : qd_mex_get_scalar<double>(prhs[7], "threshold_dB", 3.0);

    // Allocate outputs
    arma::mat bw_az, bw_el, az_pt, el_pt;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&bw_az, element_ind.n_elem, (arma::uword)n_freq);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&bw_el, element_ind.n_elem, (arma::uword)n_freq);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&az_pt, element_ind.n_elem, (arma::uword)n_freq);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&el_pt, element_ind.n_elem, (arma::uword)n_freq);

    if (n_freq > 1)
    {
        for (size_t i_freq = 0; i_freq < n_freq; ++i_freq)
        {
            double *p_bw_az = (nlhs > 0) ? bw_az.colptr(i_freq) : nullptr;
            double *p_bw_el = (nlhs > 1) ? bw_el.colptr(i_freq) : nullptr;
            double *p_az_pt = (nlhs > 2) ? az_pt.colptr(i_freq) : nullptr;
            double *p_el_pt = (nlhs > 3) ? el_pt.colptr(i_freq) : nullptr;
            for (auto el : element_ind)
            {
                CALL_QD(ant_multi[i_freq].calc_beamwidth_deg(el, threshold_dB, p_bw_az, p_bw_el, p_az_pt, p_el_pt));
                if (p_bw_az)
                    ++p_bw_az;
                if (p_bw_el)
                    ++p_bw_el;
                if (p_az_pt)
                    ++p_az_pt;
                if (p_el_pt)
                    ++p_el_pt;
            }
        }
    }
    else
    {
        double *p_bw_az = (nlhs > 0) ? bw_az.memptr() : nullptr;
        double *p_bw_el = (nlhs > 1) ? bw_el.memptr() : nullptr;
        double *p_az_pt = (nlhs > 2) ? az_pt.memptr() : nullptr;
        double *p_el_pt = (nlhs > 3) ? el_pt.memptr() : nullptr;
        for (auto el : element_ind)
        {
            CALL_QD(ant.calc_beamwidth_deg(el, threshold_dB, p_bw_az, p_bw_el, p_az_pt, p_el_pt));
            if (p_bw_az)
                ++p_bw_az;
            if (p_bw_el)
                ++p_bw_el;
            if (p_az_pt)
                ++p_az_pt;
            if (p_el_pt)
                ++p_el_pt;
        }
    }
}