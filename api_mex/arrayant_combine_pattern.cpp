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
% Minimal example (input/output = struct)
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in);

% Optional inputs: freq, azimuth_grid, elevation_grid
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in, freq, azimuth_grid, elevation_grid);

% Separate outputs, struct input
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_combine_pattern(arrayant_in);

% Separate inputs
arrayant_out = quadriga_lib.arrayant_combine_pattern([], freq, azimuth_grid, elevation_grid, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name);
```

## Examples:

The following example creates a unified linear array of 4 dipoles, spaced at half-wavelength. The
elements are then coupled with each other (i.e., they receive the same signal). The effective pattern
is calculated using `arrayant_combine_pattern`.

```
% Generate dipole pattern
ant = quadriga_lib.arrayant_generate('dipole');

% Duplicate 4 times
ant.e_theta_re  = repmat(ant.e_theta_re, [1,1,4]);
ant.e_theta_im  = repmat(ant.e_theta_im, [1,1,4]);
ant.e_phi_re    = repmat(ant.e_phi_re, [1,1,4]);
ant.e_phi_im    = repmat(ant.e_phi_im, [1,1,4]);
ant.element_pos = repmat(ant.element_pos, [1,4]);

% Set element positions and coupling matrix
ant.element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
ant.coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);
ant.coupling_im = [ 0 ; 0 ; 0 ; 0 ];

% Calculate effective pattern
ant_c = quadriga_lib.arrayant_combine_pattern( ant );

% Plot gain
plot( ant.azimuth_grid*180/pi, [ 10*log10( ant.e_theta_re(91,:,1).^2 ); 10*log10( ant_c.e_theta_re(91,:).^2 ) ]);
axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')
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
  If an empty array is passed, array antenna data is provided as separate inputs (Inputs 5-15)<br><br>

- **`freq`** [2] (optional)<br>
  An alternative value for the center frequency. Overwrites the value given in `arrayant_in`. If
  neither `freq` not `arrayant_in["center_freq"]` are given, an error is thrown.

- **`azimuth_grid`** [3] (optional)<br>
  Alternative azimuth angles for the output in [rad], -pi to pi, sorted, Size: `[n_azimuth_out]`,
  If not given, `arrayant_in.azimuth_grid` is used instead.

- **`elevation_grid`** [4] (optional)<br>
  Alternative elevation angles for the output in [rad], -pi/2 to pi/2, sorted, Size: `[n_elevation_out]`,
  If not given, `arrayant_in.elevation_grid` is used instead.

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

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    if (nrhs <= 4) // Struct
    {
        ant.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"));
        ant.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"));
        ant.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"));
        ant.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"));
        ant.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"));
        ant.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"));
        if (qd_mex_has_field(prhs[0], "element_pos"))
            ant.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "element_pos"));
        if (qd_mex_has_field(prhs[0], "coupling_re"))
            ant.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_re"));
        if (qd_mex_has_field(prhs[0], "coupling_im"))
            ant.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_im"));
        if (qd_mex_has_field(prhs[0], "center_frequency"))
            ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(prhs[0], "center_frequency"));
        if (qd_mex_has_field(prhs[0], "name"))
            ant.name = qd_mex_get_string(qd_mex_get_field(prhs[0], "name"));
    }
    else if (nrhs >= 10) // Separate
    {
        ant.e_theta_re = qd_mex_get_double_Cube(prhs[4]);
        ant.e_theta_im = qd_mex_get_double_Cube(prhs[5]);
        ant.e_phi_re = qd_mex_get_double_Cube(prhs[6]);
        ant.e_phi_im = qd_mex_get_double_Cube(prhs[7]);
        ant.azimuth_grid = qd_mex_get_double_Col(prhs[8]);
        ant.elevation_grid = qd_mex_get_double_Col(prhs[9]);
        if (nrhs > 10)
            ant.element_pos = qd_mex_get_double_Mat(prhs[10]);
        if (nrhs > 11)
            ant.coupling_re = qd_mex_get_double_Mat(prhs[11]);
        if (nrhs > 12)
            ant.coupling_im = qd_mex_get_double_Mat(prhs[12]);
        if (nrhs > 13)
            ant.center_frequency = qd_mex_get_scalar<double>(prhs[13]);
        if (nrhs > 14)
            ant.name = qd_mex_get_string(prhs[14]);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Frequency
    ant.center_frequency = (nrhs < 2) ? ant.center_frequency : qd_mex_get_scalar<double>(prhs[1], "freq", ant.center_frequency);

    // Parse grid
    arma::vec az, el;
    if (nrhs > 2)
        az = qd_mex_get_double_Col(prhs[2]);
    if (nrhs > 3)
        el = qd_mex_get_double_Col(prhs[3]);

    // Call member function
    auto arrayant_out = quadriga_lib::arrayant<double>();
    CALL_QD(arrayant_out = ant.combine_pattern(&az, &el));

    // Return to MATLAB
    if (nlhs == 1) // Output as struct
    {
        std::vector<std::string> fields = {"e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im",
                                           "azimuth_grid", "elevation_grid", "element_pos",
                                           "coupling_re", "coupling_im", "center_freq", "name"};

        plhs[0] = qd_mex_make_struct(fields);
        qd_mex_set_field(plhs[0], fields[0], qd_mex_copy2matlab(&arrayant_out.e_theta_re));
        qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&arrayant_out.e_theta_im));
        qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&arrayant_out.e_phi_re));
        qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&arrayant_out.e_phi_im));
        qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&arrayant_out.azimuth_grid, true));
        qd_mex_set_field(plhs[0], fields[5], qd_mex_copy2matlab(&arrayant_out.elevation_grid, true));
        qd_mex_set_field(plhs[0], fields[6], qd_mex_copy2matlab(&arrayant_out.element_pos));
        qd_mex_set_field(plhs[0], fields[7], qd_mex_copy2matlab(&arrayant_out.coupling_re));
        qd_mex_set_field(plhs[0], fields[8], qd_mex_copy2matlab(&arrayant_out.coupling_im));
        qd_mex_set_field(plhs[0], fields[9], qd_mex_copy2matlab(&arrayant_out.center_frequency));
        qd_mex_set_field(plhs[0], fields[10], mxCreateString(arrayant_out.name.c_str()));
    }
    else if (nlhs == 11) // Separate outputs
    {
        plhs[0] = qd_mex_copy2matlab(&arrayant_out.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&arrayant_out.e_theta_im);
        plhs[2] = qd_mex_copy2matlab(&arrayant_out.e_phi_re);
        plhs[3] = qd_mex_copy2matlab(&arrayant_out.e_phi_im);
        plhs[4] = qd_mex_copy2matlab(&arrayant_out.azimuth_grid, true);
        plhs[5] = qd_mex_copy2matlab(&arrayant_out.elevation_grid, true);
        plhs[6] = qd_mex_copy2matlab(&arrayant_out.element_pos);
        plhs[7] = qd_mex_copy2matlab(&arrayant_out.coupling_re);
        plhs[8] = qd_mex_copy2matlab(&arrayant_out.coupling_im);
        plhs[9] = qd_mex_copy2matlab(&arrayant_out.center_frequency);
        plhs[10] = mxCreateString(arrayant_out.name.c_str());
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}
