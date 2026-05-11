// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_arrayant.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_COMBINE_PATTERN
Combine element patterns, positions, and coupling weights into effective radiation patterns

- Integrates `e_theta_re/im`, `e_phi_re/im`, `element_pos`, and `coupling_re/im` to produce one output 
  element per port (column) of the coupling matrix
- Useful for beamforming and MIMO channel computation speedup

## Usage:
```
% Input as struct (struct mode)
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in);
arrayant_out = quadriga_lib.arrayant_combine_pattern(arrayant_in, freq, azimuth_grid, elevation_grid);

% Separate outputs, struct input
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name] = quadriga_lib.arrayant_combine_pattern(arrayant_in);

% Separate inputs
arrayant_out = quadriga_lib.arrayant_combine_pattern([], freq, azimuth_grid, elevation_grid, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, freq, name);
```

## Input Arguments:
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array may contain a frequency-dependent model
- **`freq`** [2] —  An alternative value for the center frequency. Overwrites the value given in `arrayant_in`. If
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
        ant.e_theta_re = qd_mex_get_Cube<double>(qd_mex_get_field(prhs[0], "e_theta_re"));
        ant.e_theta_im = qd_mex_get_Cube<double>(qd_mex_get_field(prhs[0], "e_theta_im"));
        ant.e_phi_re = qd_mex_get_Cube<double>(qd_mex_get_field(prhs[0], "e_phi_re"));
        ant.e_phi_im = qd_mex_get_Cube<double>(qd_mex_get_field(prhs[0], "e_phi_im"));
        ant.azimuth_grid = qd_mex_get_Col<double>(qd_mex_get_field(prhs[0], "azimuth_grid"));
        ant.elevation_grid = qd_mex_get_Col<double>(qd_mex_get_field(prhs[0], "elevation_grid"));
        if (qd_mex_has_field(prhs[0], "element_pos"))
            ant.element_pos = qd_mex_get_Mat<double>(qd_mex_get_field(prhs[0], "element_pos"));
        if (qd_mex_has_field(prhs[0], "coupling_re"))
            ant.coupling_re = qd_mex_get_Mat<double>(qd_mex_get_field(prhs[0], "coupling_re"));
        if (qd_mex_has_field(prhs[0], "coupling_im"))
            ant.coupling_im = qd_mex_get_Mat<double>(qd_mex_get_field(prhs[0], "coupling_im"));
        if (qd_mex_has_field(prhs[0], "center_freq"))
            ant.center_frequency = qd_mex_get_scalar<double>(qd_mex_get_field(prhs[0], "center_freq"));
        if (qd_mex_has_field(prhs[0], "name"))
            ant.name = qd_mex_get_string(qd_mex_get_field(prhs[0], "name"));
    }
    else if (nrhs >= 10) // Separate
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
        az = qd_mex_get_Col<double>(prhs[2]);
    if (nrhs > 3)
        el = qd_mex_get_Col<double>(prhs[3]);

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
