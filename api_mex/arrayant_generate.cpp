// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
% Simple antenna models, output as struct
ant = quadriga_lib.arrayant_generate('omni', res);               % Isotropic radiator, v-pol
ant = quadriga_lib.arrayant_generate('dipole', res);             % Short dipole, v-pol
ant = quadriga_lib.arrayant_generate('half-wave-dipole', res);   % Half-wave dipole, v-pol
ant = quadriga_lib.arrayant_generate('xpol', res);               % Cross-polarized isotropic radiator

% An antenna with a custom 3dB beam with (in degree)
ant = quadriga_lib.arrayant_generate('custom', res, freq, az_3dB, el_3db, rear_gain_lin);

% Antenna model for the 3GPP-NR channel model with 3GPP default pattern
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [],
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% Antenna model for the 3GPP-NR channel model with a custom beam width
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, az_3dB, el_3db, rear_gain_lin,
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% Antenna model for the 3GPP-NR channel model with a custom antenna pattern
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [],
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh, pattern);

% Optional for all types: output as separate variables, (must have exactly 11 outputs)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_generate( ... );
```

## Input Arguments:
- **`type`** [1]<br>
  Antenna model type, string

- **`res`** [2]<br>
  Pattern resolution in [deg], scalar, default = 1 deg

- **`freq`** [3]<br>
  The center frequency in [Hz], scalar, default = 299792458 Hz

## Input arguments for type 'custom' and '3GPP' (custom beam width):
- **`az_3dB`** [4]<br>
  3dB beam width in azimuth direction in [deg], scalar,
  default for `custom` = 90 deg, default for `3gpp` = 67 deg

- **`el_3db`** [5]<br>
  3dB beam width in elevation direction in [deg], scalar,
  default for `custom` = 90 deg, default for `3gpp` = 67 deg

- **`rear_gain_lin`** [6]<br>
  Isotropic gain (linear scale) at the back of the antenna, scalar, default = 0.0

## Input arguments for type '3GPP':
- **`M`** [7]<br>
  Number of vertically stacked elements, scalar, default = 1

- **`N`** [8]<br>
  Number of horizontally stacked elements, scalar, default = 1

- **`pol`** [9]<br>
  Polarization indicator to be applied for each of the M elements:<br>
  `pol = 1` | vertical polarization (default value)
  `pol = 2` | H/V polarized elements, results in 2NM elements
  `pol = 3` | +/-45° polarized elements, results in 2NM elements
  `pol = 4` | vertical polarization, combines elements in vertical direction, results in N elements
  `pol = 5` | H/V polarization, combines elements in vertical direction, results in 2N elements
  `pol = 6` | +/-45° polarization, combines elements in vertical direction, results in 2N elements
  Polarization indicator is ignored when a custom pattern is provided.

- **`tilt`** [10]<br>
  The electric downtilt angle in [deg], Only relevant for `pol = 4/5/6`, scalar, default = 0

- **`spacing`** [11]<br>
  Element spacing in [λ], scalar, default = 0.5

- **`Mg`** [12]<br>
  Number of nested panels in a column, scalar, default = 1

- **`Ng`** [13]<br>
  Number of nested panels in a row, scalar, default = 1

- **`dgv`** [14]<br>
  Panel spacing in vertical direction in [λ], scalar, default = 0.5

- **`dgh`** [15]<br>
  Panel spacing in horizontal direction in [λ], scalar, default = 0.5

- **`pattern`** [16]<br>
  Struct containing a custom pattern (default = empty) with at least the following fields:
  `e_theta_re_c`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_theta_im_c`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_re_c`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `e_phi_im_c`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements_c]`
  `azimuth_grid_c`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid_c` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`

  If custom pattern data is not provided, the pattern is generated internally (either with a custom
  beam width if `az_3dB` and `el_3db` are given or using the default 3GPP pattern).

## Output Arguments:
- **`ant`**<br>
  Struct containing the arrayant data with the following fields:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted             | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions                     | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part                            | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part                       | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], optional, default = 0.3 GHz | Scalar
  `name`           | Name of the array antenna object                      | String
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Arrayant type name missing.");

    if (nlhs == 0)
        return;

    std::string array_type = qd_mex_get_string(prhs[0]);
    double res = (nrhs < 2) ? 1.0 : qd_mex_get_scalar<double>(prhs[1], "res", 1.0);
    double freq = (nrhs < 3) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[2], "freq", 299792458.0);
    double az_3dB = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "az_3dB", 0.0);
    double el_3dB = (nrhs < 5) ? 0.0 : qd_mex_get_scalar<double>(prhs[4], "el_3dB", 0.0);
    double rear_gain_lin = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "rear_gain_lin", 0.0);
    unsigned M = (nrhs < 7) ? 1 : qd_mex_get_scalar<unsigned>(prhs[6], "M", 1);
    unsigned N = (nrhs < 8) ? 1 : qd_mex_get_scalar<unsigned>(prhs[7], "N", 1);
    unsigned pol = (nrhs < 9) ? 1 : qd_mex_get_scalar<unsigned>(prhs[8], "pol", 1);
    double tilt = (nrhs < 10) ? 0.0 : qd_mex_get_scalar<double>(prhs[9], "tilt", 0.0);
    double spacing = (nrhs < 11) ? 0.5 : qd_mex_get_scalar<double>(prhs[10], "spacing", 0.5);
    unsigned Mg = (nrhs < 12) ? 1 : qd_mex_get_scalar<unsigned>(prhs[11], "Mg", 1);
    unsigned Ng = (nrhs < 13) ? 1 : qd_mex_get_scalar<unsigned>(prhs[12], "Ng", 1);
    double dgv = (nrhs < 14) ? 0.5 : qd_mex_get_scalar<double>(prhs[13], "dgv", 0.5);
    double dgh = (nrhs < 15) ? 0.5 : qd_mex_get_scalar<double>(prhs[14], "dgh", 0.5);

    quadriga_lib::arrayant<double> arrayant;

    if (array_type == "omni")
        arrayant = quadriga_lib::generate_arrayant_omni<double>(res);
    else if (array_type == "dipole" || array_type == "short-dipole")
        arrayant = quadriga_lib::generate_arrayant_dipole<double>(res);
    else if (array_type == "half-wave-dipole")
        arrayant = quadriga_lib::generate_arrayant_half_wave_dipole<double>(res);
    else if (array_type == "xpol")
        arrayant = quadriga_lib::generate_arrayant_xpol<double>(res);
    else if (array_type == "custom")
        arrayant = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
    else if (array_type == "3GPP" || array_type == "3gpp")
    {
        if (nrhs > 15)
        {
            quadriga_lib::arrayant<double> custom_array;
            custom_array.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[15], "e_theta_re"));
            custom_array.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[15], "e_theta_im"));
            custom_array.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[15], "e_phi_re"));
            custom_array.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[15], "e_phi_im"));
            custom_array.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[15], "azimuth_grid"));
            custom_array.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[15], "elevation_grid"));
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array);
        }
        else if (az_3dB > 0.0 && el_3dB > 0.0) // Use custom beam width
        {
            auto custom_array = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
            if (pol == 2 || pol == 5)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(90.0, 0.0, 0.0, 2, 1);
            }
            else if (pol == 3 || pol == 6)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
                custom_array.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
            }
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array);
        }
        else // Use 3GPP default pattern
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, nullptr, res);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Array type not supported!");

    // Set center frequency for all types
    arrayant.center_frequency = freq;

    if (nlhs == 1) // Output as struct
    {
        std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                           "azimuth_grid", "elevation_grid", "element_pos",
                                           "coupling_re", "coupling_im", "center_freq", "name"};

        plhs[0] = qd_mex_make_struct(fields);
        qd_mex_set_field(plhs[0], fields[0], qd_mex_copy2matlab(&arrayant.e_theta_re));
        qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&arrayant.e_theta_im));
        qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&arrayant.e_phi_re));
        qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&arrayant.e_phi_im));
        qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&arrayant.azimuth_grid, true));
        qd_mex_set_field(plhs[0], fields[5], qd_mex_copy2matlab(&arrayant.elevation_grid, true));
        qd_mex_set_field(plhs[0], fields[6], qd_mex_copy2matlab(&arrayant.element_pos));
        qd_mex_set_field(plhs[0], fields[7], qd_mex_copy2matlab(&arrayant.coupling_re));
        qd_mex_set_field(plhs[0], fields[8], qd_mex_copy2matlab(&arrayant.coupling_im));
        qd_mex_set_field(plhs[0], fields[9], qd_mex_copy2matlab(&arrayant.center_frequency));
        qd_mex_set_field(plhs[0], fields[10], mxCreateString(arrayant.name.c_str()));
    }
    else if (nlhs == 11) // Separate outputs
    {
        plhs[0] = qd_mex_copy2matlab(&arrayant.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&arrayant.e_theta_im);
        plhs[2] = qd_mex_copy2matlab(&arrayant.e_phi_re);
        plhs[3] = qd_mex_copy2matlab(&arrayant.e_phi_im);
        plhs[4] = qd_mex_copy2matlab(&arrayant.azimuth_grid, true);
        plhs[5] = qd_mex_copy2matlab(&arrayant.elevation_grid, true);
        plhs[6] = qd_mex_copy2matlab(&arrayant.element_pos);
        plhs[7] = qd_mex_copy2matlab(&arrayant.coupling_re);
        plhs[8] = qd_mex_copy2matlab(&arrayant.coupling_im);
        plhs[9] = qd_mex_copy2matlab(&arrayant.center_frequency);
        plhs[10] = mxCreateString(arrayant.name.c_str());
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}