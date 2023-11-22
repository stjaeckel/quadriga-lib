// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_arrayant.hpp"
#include "mex_helper_functions.cpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COMBINE_PATTERN
Calculate effective radiation patterns for array antennas

## Description:
An array antenna consists of multiple individual elements. Each element occupies a specific position 
relative to the array's phase-center, its local origin. Elements can also be inter-coupled, 
represented by a coupling matrix. By integrating the element radiation patterns, their positions, 
and the coupling weights, one can determine an effective radiation pattern observable by a receiver 
in the antenna's far field. Leveraging these effective patterns is especially beneficial in antenna 
design, beamforming applications such as in 5G systems, and in planning wireless communication 
networks in complex environments like urban areas. This streamlined approach offers a significant 
boost in computation speed when calculating MIMO channel coefficients, as it reduces the number of 
necessary operations. The function `arrayant_combine_pattern` is designed to compute these effective 
radiation patterns.

## Usage:

```
[ e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c] = quadriga_lib.arrayant_combine_pattern( ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, 
    coupling_re, coupling_im, center_freq);
```

## Examples:

The following example creates a unified linear array of 4 dipoles, spaced at half-wavelength. The
elements are then coupled with each other (i.e., they receive the same signal). The effective pattern
is calculated using `arrayant_combine_pattern`.

```
% Generate dipole pattern
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos] = ...
    <a href="#arrayant_generate">quadriga_lib.arrayant_generate</a>('dipole');

% Duplicate 4 times
e_theta_re  = repmat(e_theta_re, [1,1,4]);
e_theta_im  = repmat(e_theta_im, [1,1,4]);
e_phi_re    = repmat(e_phi_re, [1,1,4]);
e_phi_im    = repmat(e_phi_im, [1,1,4]);
element_pos = repmat(element_pos, [1,4]);

% Set element positions and coupling matrix
element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);

% Calculate effective pattern
[ e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c] = quadriga_lib.arrayant_combine_pattern( ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re);

% Plot gain
plot( azimuth_grid*180/pi, [ 10*log10( e_theta_re(91,:,1).^2 ); 10*log10( e_theta_re_c(91,:).^2 ) ]);
axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')

```

## Input Arguments:
- **Antenna data:** (inputs 1-10, single or double)
  `e_theta_re`     | Real part of e-theta field component                  | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | Imaginary part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | Real part of e-phi field component                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | Imaginary part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements]` or `[]`
  `coupling_re`    | Real part of coupling matrix, optional                | Size: `[n_elements, n_ports]` or `[]`
  `coupling_im`    | Imaginary part of coupling matrix, optional           | Size: `[n_elements, n_ports]` or `[]`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar

## Output Arguments:
- **`e_theta_re_c`**<br>
  Real part of the e-theta component (vertical component) of the effective array antenna.<br>
  Size: `[n_elevation, n_azimuth, n_ports]`

- **`e_theta_im_c`**<br>
  Imaginary part of the e-theta component (vertical component) of the effective array antenna.<br>
  Size: `[n_elevation, n_azimuth, n_ports]`

- **`e_phi_re_c`**<br>
  Real part of the e-phi component (horizontal component) of the effective array antenna.<br>
  Size: `[n_elevation, n_azimuth, n_ports]`

- **`e_phi_im_c`**<br>
  Imaginary part of the e-phi component (horizontal component) of the effective array antenna.<br>
  Size: `[n_elevation, n_azimuth, n_ports]`

## Caveat:
The effective antenna has all elements at the phase center `[0,0,0]'` and has perfect isolation
between its elements. Hence, no outputs for the effective `element_pos` and `coupling` are needed.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_elements] or []
    //  7 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  8 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    //  9 - center_frequency   Center frequency in [Hz]                                     Scalar

    // Outputs:
    //  0 - e_theta_re_c    Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_ports]
    //  1 - e_theta_im_c    Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_ports]
    //  2 - e_phi_re_c      Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_ports]
    //  3 - e_phi_im_c      Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_ports]

    if (nrhs < 6)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", "Need at least 6 inputs.");

    if (nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", "Can have at most 10 inputs.");

    if (nlhs != 4)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", "Must have exactly 4 outputs.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 9; ++i)
        if (nrhs > i)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Create arrayant object from the input data
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
    {
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]);
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]);
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]);
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]);
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]);
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]);
        if (nrhs > 6)
            arrayant_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[6]);
        if (nrhs > 7)
            arrayant_single.coupling_re = qd_mex_reinterpret_Mat<float>(prhs[7]);
        if (nrhs > 8)
            arrayant_single.coupling_im = qd_mex_reinterpret_Mat<float>(prhs[8]);
        if (nrhs > 9)
            arrayant_single.center_frequency = qd_mex_get_scalar<float>(prhs[9], "center_frequency");
        arrayant_single.read_only = true;
    }
    else
    {
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]);
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]);
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]);
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]);
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]);
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]);
        if (nrhs > 6)
            arrayant_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[6]);
        if (nrhs > 7)
            arrayant_double.coupling_re = qd_mex_reinterpret_Mat<double>(prhs[7]);
        if (nrhs > 8)
            arrayant_double.coupling_im = qd_mex_reinterpret_Mat<double>(prhs[8]);
        if (nrhs > 9)
            arrayant_double.center_frequency = qd_mex_get_scalar<double>(prhs[9], "center_frequency");
        arrayant_double.read_only = true;
    }

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:IO_error", error_message.c_str());

    // Allocate memory for output data
    unsigned long long n_az = use_single ? arrayant_single.n_azimuth() : arrayant_double.n_azimuth();
    unsigned long long n_el = use_single ? arrayant_single.n_elevation() : arrayant_double.n_elevation();
    unsigned long long n_prt = use_single ? arrayant_single.n_ports() : arrayant_double.n_ports();

    quadriga_lib::arrayant<float> arrayant_single_out;
    quadriga_lib::arrayant<double> arrayant_double_out;
    if (use_single)
        plhs[0] = qd_mex_init_output(&arrayant_single_out.e_theta_re, n_el, n_az, n_prt),
        plhs[1] = qd_mex_init_output(&arrayant_single_out.e_theta_im, n_el, n_az, n_prt),
        plhs[2] = qd_mex_init_output(&arrayant_single_out.e_phi_re, n_el, n_az, n_prt),
        plhs[3] = qd_mex_init_output(&arrayant_single_out.e_phi_im, n_el, n_az, n_prt);
    else
        plhs[0] = qd_mex_init_output(&arrayant_double_out.e_theta_re, n_el, n_az, n_prt),
        plhs[1] = qd_mex_init_output(&arrayant_double_out.e_theta_im, n_el, n_az, n_prt),
        plhs[2] = qd_mex_init_output(&arrayant_double_out.e_phi_re, n_el, n_az, n_prt),
        plhs[3] = qd_mex_init_output(&arrayant_double_out.e_phi_im, n_el, n_az, n_prt);

    // Call library function
    try
    {
        if (use_single)
            arrayant_single.combine_pattern(&arrayant_single_out);
        else
            arrayant_double.combine_pattern(&arrayant_double_out);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_combine_pattern:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}