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
#include "mex_helper_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_GENERATE
Generates predefined array antenna models

## Description:
This functions can be used to generate a variety of pre-defined array antenna models, including 3GPP
array antennas used for 5G-NR simulations. The first argument is the array type. The following input
arguments are then specific to this type.

## Usage:

```
% Isotropic radiator, vertical polarization
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('omni', res);

% Short dipole radiating with vertical polarization
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('dipole', res);

% Half-wave dipole radiating with vertical polarization
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('half-wave-dipole', res);

% Cross-polarized isotropic radiator
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('xpol', res);

% An antenna with a custom 3dB beam with (in degree)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = ...
    quadriga_lib.arrayant_generate('custom', az_3dB, el_3db, rear_gain_lin, res );

% Antenna model for the 3GPP-NR channel model
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = ...
    quadriga_lib.arrayant_generate('3GPP', M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, res );

% Antenna model for the 3GPP-NR channel model with a custom pattern
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = ...
    quadriga_lib.arrayant_generate('3GPP', M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, res, ...
    e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c, azimuth_grid_c, elevation_grid_c );
```

## Optional input argument for types 'omni', 'dipole', 'half-wave-dipole', and 'xpol':
- **`res`**<br>
  Pattern resolution in [deg], scalar, Default = 1 deg

## Input arguments for type 'custom':
- **`az_3dB`**<br>
  3dB beam width in azimuth direction in [deg], scalar

- **`el_3db`**<br>
  3dB beam width in elevation direction in [deg], scalar

- **`rear_gain_lin`**<br>
  Isotropic gain (linear scale) at the back of the antenna, scalar

- **`res`** (optional)<br>
  Pattern resolution in [deg], scalar, Default = 1 deg

## Input arguments for type '3GPP':
- **`M`**<br>
  Number of vertically stacked elements, scalar, default = 1

- **`N`**<br>
  Number of horizontally stacked elements, scalar, default = 1

- **`center_freq`**<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

- **`pol`**<br>
  Polarization indicator to be applied for each of the M elements:<br>
  `pol = 1` | vertical polarization (default value)
  `pol = 2` | H/V polarized elements, results in 2NM elements
  `pol = 3` | +/-45° polarized elements, results in 2NM elements
  `pol = 4` | vertical polarization, combines elements in vertical direction, results in N elements
  `pol = 5` | H/V polarization, combines elements in vertical direction, results in 2N elements
  `pol = 6` | +/-45° polarization, combines elements in vertical direction, results in 2N elements
  Polarization indicator is ignored when a custom pattern is provided.

- **`tilt`**<br>
  The electric downtilt angle in [deg], Only relevant for `pol = 4/5/6`, scalar, default = 0

- **`spacing`**<br>
  Element spacing in [λ], scalar, default = 0.5

- **`Mg`**<br>
  Number of nested panels in a column, scalar, default = 1

- **`Ng`**<br>
  Number of nested panels in a row, scalar, default = 1

- **`dgv`**<br>
  Panel spacing in vertical direction in [λ], scalar, default = 0.5

- **`dgh`**<br>
  Panel spacing in horizontal direction in [λ], scalar, default = 0.5

- **`res`** (optional)<br>
  Pattern resolution in [deg], scalar, Default = 1 deg, Note: In case of a custom pattern, `res` is
  replaced by `e_theta_re_c` and the generated array antenna inherits the resolutions from the 
  custom pattern data.

- **Antenna data for custom pattern data:** (inputs 11-16, double precision, optional)
  `e_theta_re_c`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_theta_im_c`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_re_c`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_im_c`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `azimuth_grid_c`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid_c` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`

  If custom pattern data is not provided, the default 3GPP element pattern is used.

## Output Arguments:
- **Antenna data of the generated array antenna:** (outputs 1-11, double precision)
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
  `name`           | Name of the array antenna object                      | String
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - array_type      Array type (string)
    //  1-17                Additional parameters

    // Outputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - element_pos     Element positions                                               Size [3, n_elements]
    //  7 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  8 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    //  9 - center_frequency   Center frequency in [Hz]                                     Scalar
    // 10 - name            Name of the array antenna object, string

    // Number of in and outputs
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Arrayant type name missing.");

    if (nlhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of output arguments.");

    std::string array_type = qd_mex_get_string(prhs[0]);

    // Read resolution
    double res = 1.0;
    if (array_type == "omni" || array_type == "dipole" || array_type == "short-dipole" || array_type == "xpol")
        res = nrhs < 2 ? 1.0 : qd_mex_get_scalar<double>(prhs[1], "res", 1.0);

    // Returns double by default
    quadriga_lib::arrayant<double> arrayant_double;

    if (array_type == "omni")
        arrayant_double = quadriga_lib::generate_arrayant_omni<double>(res);
    else if (array_type == "dipole" || array_type == "short-dipole")
        arrayant_double = quadriga_lib::generate_arrayant_dipole<double>(res);
    else if (array_type == "half-wave-dipole")
        arrayant_double = quadriga_lib::generate_arrayant_half_wave_dipole<double>(res);
    else if (array_type == "xpol")
        arrayant_double = quadriga_lib::generate_arrayant_xpol<double>(res);
    else if (array_type == "custom")
    {
        res = nrhs < 5 ? 1.0 : qd_mex_get_scalar<double>(prhs[4], "res", 1.0);
        if (nrhs < 4)
            mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");
        else
            arrayant_double = quadriga_lib::generate_arrayant_custom<double>(qd_mex_get_scalar<double>(prhs[1], "az_3dB", 90.0),
                                                                             qd_mex_get_scalar<double>(prhs[2], "el_3db", 90.0),
                                                                             qd_mex_get_scalar<double>(prhs[3], "rear_gain_lin", 0.0), res);
    }
    else if (array_type == "3GPP" || array_type == "3gpp")
    {
        unsigned M = nrhs < 2 ? 1 : qd_mex_get_scalar<unsigned>(prhs[1], "M", 1);
        unsigned N = nrhs < 3 ? 1 : qd_mex_get_scalar<unsigned>(prhs[2], "N", 1);
        double center_freq = nrhs < 4 ? 299792458.0 : qd_mex_get_scalar<double>(prhs[3], "center_freq", 299792458.0);
        unsigned pol = nrhs < 5 ? 1 : qd_mex_get_scalar<unsigned>(prhs[4], "pol", 1);
        double tilt = nrhs < 6 ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "tilt", 0.0);
        double spacing = nrhs < 7 ? 0.5 : qd_mex_get_scalar<double>(prhs[6], "spacing", 0.5);
        unsigned Mg = nrhs < 8 ? 1 : qd_mex_get_scalar<unsigned>(prhs[7], "Mg", 1);
        unsigned Ng = nrhs < 9 ? 1 : qd_mex_get_scalar<unsigned>(prhs[8], "Ng", 1);
        double dgv = nrhs < 10 ? 0.5 : qd_mex_get_scalar<double>(prhs[9], "dgv", 0.5);
        double dgh = nrhs < 11 ? 0.5 : qd_mex_get_scalar<double>(prhs[10], "dgh", 0.5);
        res = nrhs < 12 ? 1.0 : qd_mex_get_scalar<double>(prhs[11], "res", 1.0);

        if (nrhs < 13) // Use default 3GPP pattern
            CALL_QD(arrayant_double = quadriga_lib::generate_arrayant_3GPP<double>(M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, nullptr, res));
        else if (nrhs < 17)
            mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");
        else if (mxIsDouble(prhs[11]) && mxIsDouble(prhs[12]) && mxIsDouble(prhs[13]) && mxIsDouble(prhs[14]) && mxIsDouble(prhs[15]) && mxIsDouble(prhs[16]))
        {
            quadriga_lib::arrayant<double> pattern;
            pattern.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[11]);
            pattern.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[12]);
            pattern.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[13]);
            pattern.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[14]);
            pattern.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[15]);
            pattern.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[16]);
            CALL_QD(arrayant_double = quadriga_lib::generate_arrayant_3GPP<double>(M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &pattern));
        }
        else
            mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Custom antenna pattern must be provided in double precision.");
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Array type not supported!");

    // Write to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&arrayant_double.e_theta_re);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&arrayant_double.e_theta_im);

    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&arrayant_double.e_phi_re);

    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&arrayant_double.e_phi_im);

    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&arrayant_double.azimuth_grid, true);

    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&arrayant_double.elevation_grid, true);

    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&arrayant_double.element_pos);

    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&arrayant_double.coupling_re);

    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&arrayant_double.coupling_im);

    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&arrayant_double.center_frequency);

    if (nlhs > 10)
        plhs[10] = mxCreateString(arrayant_double.name.c_str());
}