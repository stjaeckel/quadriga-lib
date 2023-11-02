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
[ e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r ] = ...
    quadriga_lib.arrayant_rotate_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, ...
    elevation_grid, element_pos, x_deg, y_deg, z_deg, usage )
```

## Input Arguments:
- **`e_theta_re`**<br>
  Real part of the e-theta component (vertical component) of the far field of each antenna element
  in the array antenna. Single or double precision, <br>Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_theta_im`**<br>
  Imaginary part of the e-theta component of the electric field.
  Single or double precision, <br>Size: [n_elevation, n_azimuth, n_elements]

- **`e_phi_re`**<br>
  Real part of the e-phi component (horizontal component) of the far field of each antenna element
  in the array antenna. Single or double precision, <br>Size: `[n_elevation, n_azimuth, n_elements]`

- **`e_phi_im`**<br>
  Imaginary part of the e-phi component of the electric field.
  Single or double precision, <br>Size: `[n_elevation, n_azimuth, n_elements]`

- **`azimuth_grid`**<br>
  Azimuth angles (theta) in [rad] were samples of the field patterns are provided. Values must be
  between -pi and pi, sorted in ascending order. Single or double precision, Size: `[n_azimuth]`

- **`elevation_grid`**<br>
  Elevation angles (phi) in [rad] where samples of the field patterns are provided. Values must be
  between -pi/2 and pi/2, sorted in ascending order. Single or double precision, <br>Size: `[n_elevation]`

- **`element_pos`**<br>
  Antenna element (x,y,z) positions relative to the array's phase-center in units of [m].<br>
  Size: `[3, n_elements]` or `[]`; empty input assumes position `[0;0;0]` for all elements

- **`x_deg`**<br>
  The rotation angle around x-axis (bank angle) in [degrees]

- **`y_deg`**<br>
  The rotation angle around y-axis (tilt angle) in [degrees]

- **`z_deg`**<br>
  The rotation angle around z-axis (heading angle) in [degrees]

- **`usage`**<br>
  The optional parameter 'usage' can limit the rotation procedure either to the pattern or polarization.
  `usage = 0` | Rotate both, pattern and polarization, adjusts sampling grid (default)
  `usage = 1` | Rotate only pattern, adjusts sampling grid
  `usage = 2` | Rotate only polarization
  `usage = 3` | Rotate both, but do not adjust the sampling grid

## Output Arguments:
- **`e_theta_re_r`**<br>
  Real part of the e-theta component (vertical component) of the rotated array antenna.<br>
  Size: `[n_elevation_r, n_azimuth_r, n_elements]`

- **`e_theta_im_r`**<br>
  Imaginary part of the e-theta component (vertical component) of the rotated array antenna.<br>
  Size: `[n_elevation_r, n_azimuth_r, n_elements]`

- **`e_phi_re_r`**<br>
  Real part of the e-phi component (horizontal component) of the rotated array antenna.<br>
  Size: `[n_elevation_r, n_azimuth_r, n_elements]`

- **`e_phi_im_r`**<br>
  Imaginary part of the e-phi component (horizontal component) of the rotated array antenna.<br>
  Size: `[n_elevation_r, n_azimuth_r, n_elements]`

- **`azimuth_grid_r`**<br>
  Azimuth angles (theta) in [rad] of the updated sampling grid, Size: `[n_azimuth_r]`

- **`elevation_grid_r`**<br>
  Elevation angles (phi) in [rad] of the updated sampling grid, Size: `[n_elevation_r]`

- **`element_pos_r`**<br>
  Antenna element (x,y,z) positions of the rotated array antenna
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
    //  7 - x_deg           The rotation angle around x-axis (bank angle) in [degrees]      Scalar
    //  8 - y_deg           The rotation angle around y-axis (tilt angle) in [degrees]      Scalar
    //  9 - z_deg           The rotation angle around z-axis (heading angle) in [degrees]   Scalar
    // 10 - usage           0: Rotate both (pattern+polarization), 1: Rotate only pattern, 2: Rotate only polarization, 3: as (0), but w/o grid adjusting

    // Outputs:
    //  0 - e_theta_re_r    Vertical component of the electric field, real part,            Size [n_elevation_r, n_azimuth_r, n_elements]
    //  1 - e_theta_im_r    Vertical component of the electric field, imaginary part,       Size [n_elevation_r, n_azimuth_r, n_elements]
    //  2 - e_phi_re_r      Horizontal component of the electric field, real part,          Size [n_elevation_r, n_azimuth_r, n_elements]
    //  3 - e_phi_im_r      Horizontal component of the electric field, imaginary part,     Size [n_elevation_r, n_azimuth_r, n_elements]
    //  4 - azimuth_grid_r    Azimuth angles in pattern (theta) in [rad], sorted,           Vector of length "n_azimuth_r"
    //  5 - elevation_grid_r  Elevation angles in pattern (phi) in [rad], sorted,           Vector of length "n_elevation_r"
    //  6 - element_pos_r     Element positions, optional, Default: [0,0,0]                 Size [3, n_elements]

    if (nrhs < 7)
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Need at least 7 inputs.");

    if (nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Can have at most 11 inputs.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Can have at most 7 outputs.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 7; ++i)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]),
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]),
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]),
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]),
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]),
        arrayant_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[6]),
        arrayant_single.read_only = true;
    else
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]),
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]),
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]),
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]),
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]),
        arrayant_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[6]),
        arrayant_double.read_only = true;

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", error_message.c_str());

    double x_deg = nrhs < 8 ? 0.0 : qd_mex_get_scalar<double>(prhs[7], "x_deg", 0.0);
    double y_deg = nrhs < 9 ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "y_deg", 0.0);
    double z_deg = nrhs < 10 ? 0.0 : qd_mex_get_scalar<double>(prhs[9], "z_deg", 0.0);
    unsigned usage = nrhs < 11 ? 0 : qd_mex_get_scalar<unsigned>(prhs[10], "usage", 0);

    quadriga_lib::arrayant<float> output_single;
    quadriga_lib::arrayant<double> output_double;

    // Call library function
    try
    {
        if (use_single)
            arrayant_single.rotate_pattern(float(x_deg), float(y_deg), float(z_deg), usage, -1, &output_single);
        else
            arrayant_double.rotate_pattern(x_deg, y_deg, z_deg, usage, -1, &output_double);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Copy output to MATLAB
    if (use_single)
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_copy2matlab(&output_single.e_theta_re);
        if (nlhs > 1)
            plhs[1] = qd_mex_copy2matlab(&output_single.e_theta_im);
        if (nlhs > 2)
            plhs[2] = qd_mex_copy2matlab(&output_single.e_phi_re);
        if (nlhs > 3)
            plhs[3] = qd_mex_copy2matlab(&output_single.e_phi_im);
        if (nlhs > 4)
            plhs[4] = qd_mex_copy2matlab(&output_single.azimuth_grid);
        if (nlhs > 5)
            plhs[5] = qd_mex_copy2matlab(&output_single.elevation_grid);
        if (nlhs > 6)
            plhs[6] = qd_mex_copy2matlab(&output_single.element_pos);
    }
    else
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_copy2matlab(&output_double.e_theta_re);
        if (nlhs > 1)
            plhs[1] = qd_mex_copy2matlab(&output_double.e_theta_im);
        if (nlhs > 2)
            plhs[2] = qd_mex_copy2matlab(&output_double.e_phi_re);
        if (nlhs > 3)
            plhs[3] = qd_mex_copy2matlab(&output_double.e_phi_im);
        if (nlhs > 4)
            plhs[4] = qd_mex_copy2matlab(&output_double.azimuth_grid);
        if (nlhs > 5)
            plhs[5] = qd_mex_copy2matlab(&output_double.elevation_grid);
        if (nlhs > 6)
            plhs[6] = qd_mex_copy2matlab(&output_double.element_pos);
    }
}