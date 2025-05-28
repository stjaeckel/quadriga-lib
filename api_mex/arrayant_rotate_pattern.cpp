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
# ARRAYANT_ROTATE_PATTERN
Rotates antenna patterns

## Description:
This MATLAB function transforms the radiation patterns of array antenna elements, allowing for
precise rotations around the three principal axes (x, y, z) of the local Cartesian coordinate system.
This is essential in antenna design and optimization, enabling engineers to tailor the radiation
pattern for enhanced performance. The function also adjusts the sampling grid for non-uniformly
sampled antennas, such as parabolic antennas with small apertures, ensuring accurate and efficient
computations. The 3 rotations are applies in the order: 1. rotation around the x-axis (bank angle);
2. rotation around the y-axis (tilt angle), 3. rotation around the z-axis (heading angle)

## Usage:

```
% Minimal example (input/output = struct)
arrayant_out = quadriga_lib.arrayant_rotate_pattern(arrayant_in, bank, tilt, head, usage, element);

% Separate outputs, struct input
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_rotate_pattern(arrayant_in, ...
    bank, tilt, head, usage, element);

% Separate inputs
arrayant_out = quadriga_lib.arrayant_rotate_pattern([], bank, tilt, head, usage, element, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name);
```

## Input Arguments:
- **`arrayant_in`** [1]<br>
  Struct containing the arrayant data with the following fields:
  `e_theta_re`     | e-theta field component, real part                    | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | e-theta field component, imaginary part               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | e-phi field component, real part                      | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | e-phi field component, imaginary part                 | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Size: `[n_elevation]`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Size: `[3, n_elements]`
  `coupling_re`    | Coupling matrix, real part, optional                  | Size: `[n_elements, n_ports]`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Size: `[n_elements, n_ports]`
  `center_freq`    | Center frequency in [Hz], optional                    | Scalar
  `name`           | Name of the array antenna object, optional            | String
  If an empty array is passed, array antenna data is provided as separate inputs (Inputs 7-17)<br><br>

- **`x_deg`** [2] (optional)<br>
  The rotation angle around x-axis (bank angle) in [degrees]

- **`y_deg`** [3] (optional)<br>
  The rotation angle around y-axis (tilt angle) in [degrees]

- **`z_deg`** [4] (optional)<br>
  The rotation angle around z-axis (heading angle) in [degrees]

- **`usage`** [5] (optional)<br>
  The optional parameter 'usage' can limit the rotation procedure either to the pattern or polarization.
  `usage = 0` | Rotate both, pattern and polarization, adjusts sampling grid (default)
  `usage = 1` | Rotate only pattern, adjusts sampling grid
  `usage = 2` | Rotate only polarization
  `usage = 3` | Rotate both, but do not adjust the sampling grid

- **`element`** [6] (optional)<br>
  The element indices for which the pattern should be transformed. Optional parameter. Values must
  be between 1 and n_elements. If this parameter is not provided (or an empty array is passed),
  all elements will be rotated by the same angles. Size: `[1, n_elements]` or empty `[]`

## Output Arguments:
- **`arrayant_out`**<br>
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
  `center_freq`    | Center frequency in [Hz], default = 0.3 GHz           | Scalar
  `name`           | Name of the array antenna object                      | String
  Can be returned as separate outputs.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs == 0)
        return;

    // Assemble array antenna object (copy input data)
    auto ant = quadriga_lib::arrayant<double>();
    if (nrhs <= 6) // Struct
    {
        ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"), true);
        ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"), true);
        ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"), true);
        ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"), true);
        ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"), true);
        ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"), true);
        if (qd_mex_has_field(prhs[0], "element_pos"))
            ant.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "element_pos"), true);
        if (qd_mex_has_field(prhs[0], "coupling_re"))
            ant.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_re"), true);
        if (qd_mex_has_field(prhs[0], "coupling_im"))
            ant.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_im"), true);
        if (qd_mex_has_field(prhs[0], "center_frequency"))
            ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(prhs[0], "center_frequency"));
        if (qd_mex_has_field(prhs[0], "name"))
            ant.name = qd_mex_get_string(qd_mex_get_field(prhs[0], "name"));
    }
    else if (nrhs >= 12) // Separate
    {
        ant.e_theta_re = qd_mex_get_double_Cube(prhs[6], true);
        ant.e_theta_im = qd_mex_get_double_Cube(prhs[7], true);
        ant.e_phi_re = qd_mex_get_double_Cube(prhs[8], true);
        ant.e_phi_im = qd_mex_get_double_Cube(prhs[9], true);
        ant.azimuth_grid = qd_mex_get_double_Col(prhs[10], true);
        ant.elevation_grid = qd_mex_get_double_Col(prhs[11], true);
        if (nrhs > 12)
            ant.element_pos = qd_mex_get_double_Mat(prhs[12], true);
        if (nrhs > 13)
            ant.coupling_re = qd_mex_get_double_Mat(prhs[13], true);
        if (nrhs > 14)
            ant.coupling_im = qd_mex_get_double_Mat(prhs[14], true);
        if (nrhs > 15)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[15]);
        if (nrhs > 16)
            ant.name = qd_mex_get_string(prhs[15]);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Parse other arguments
    double x_deg = (nrhs < 2) ? 0.0 : qd_mex_get_scalar<double>(prhs[1], "x_deg", 0.0);
    double y_deg = (nrhs < 3) ? 0.0 : qd_mex_get_scalar<double>(prhs[2], "y_deg", 0.0);
    double z_deg = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "z_deg", 0.0);
    unsigned usage = (nrhs < 5) ? 0 : qd_mex_get_scalar<unsigned>(prhs[4], "usage", 0);
    const arma::u32_vec element_ind = (nrhs < 6) ? arma::u32_vec() : qd_mex_typecast_Col<unsigned>(prhs[5]) - 1;

    // Call member function
    if (element_ind.n_elem == 0)
        CALL_QD(ant.rotate_pattern(x_deg, y_deg, z_deg, usage));
    else
        for (auto el : element_ind)
            CALL_QD(ant.rotate_pattern(x_deg, y_deg, z_deg, usage, el));

    // Return to MATLAB
    if (nlhs == 1) // Output as struct
    {
        std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                           "azimuth_grid", "elevation_grid", "element_pos",
                                           "coupling_re", "coupling_im", "center_freq", "name"};

        plhs[0] = qd_mex_make_struct(fields);
        qd_mex_set_field(plhs[0], fields[0], qd_mex_copy2matlab(&ant.e_theta_re));
        qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&ant.e_theta_im));
        qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&ant.e_phi_re));
        qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&ant.e_phi_im));
        qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&ant.azimuth_grid, true));
        qd_mex_set_field(plhs[0], fields[5], qd_mex_copy2matlab(&ant.elevation_grid, true));
        qd_mex_set_field(plhs[0], fields[6], qd_mex_copy2matlab(&ant.element_pos));
        qd_mex_set_field(plhs[0], fields[7], qd_mex_copy2matlab(&ant.coupling_re));
        qd_mex_set_field(plhs[0], fields[8], qd_mex_copy2matlab(&ant.coupling_im));
        qd_mex_set_field(plhs[0], fields[9], qd_mex_copy2matlab(&ant.center_frequency));
        qd_mex_set_field(plhs[0], fields[10], mxCreateString(ant.name.c_str()));
    }
    else if (nlhs == 11) // Separate outputs
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