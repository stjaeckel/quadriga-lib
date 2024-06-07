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
#include "quadriga_tools.hpp"
#include "mex_helper_functions.cpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# CART2GEO
Transform Cartesian (x,y,z) coordinates to Geographic (az, el, length) coordinates

## Description:
This function transforms Cartesian (x,y,z) coordinates to Geographic (azimuth, elevation, length)
coordinates. A geographic coordinate system is a three-dimensional reference system that locates
points on the surface of a sphere. A point has three coordinate values: azimuth, elevation and length
where azimuth and elevation measure angles. In the geographic coordinate system, the elevation angle
θ = 90° points to the zenith and θ = 0° points to the horizon.

## Usage:

```
[ azimuth, elevation, length ] = arrayant_lib.cart2geo( cart )
```

## Input Argument:
- **`cart`**<br>
  Cartesian coordinates (x,y,z)
  Single or double precision, Size: `[3, n_row, n_col]`

## Output Arguments:
- **`azimuth`**<br>
  Azimuth angles in [rad], values between -pi and pi.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`elevation`**<br>
  Elevation angles in [rad], values between -pi/2 and pi/2.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`length`**<br>
  Vector length, i.e. the distance from the origin to the point defined by x,y,z.
  Single or double precision (same as input), Size `[n_row, n_col]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input:           cart            Cartesian coordinates,                  Size [3, n_row, n_col]
    // Outputs:         azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector, optional,         Size [n_row, n_col]

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_input", "Cartesian coordinates not given.");

    if (nrhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_input", "Too many input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_output", "Too many output arguments.");

    if (mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:empty", "Input cannot be empty.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:IO_error", "Input must be provided in 'single' or 'double' precision.");

    arma::fcube cart_single;
    arma::cube cart_double;
    arma::cube geo_double;

    try
    {
        if (use_single)
        {
            cart_single = qd_mex_reinterpret_Cube<float>(prhs[0]);
            geo_double = quadriga_lib::cart2geo(cart_single);
        }
        else
        {
            cart_double = qd_mex_reinterpret_Cube<double>(prhs[0]);
            geo_double = quadriga_lib::cart2geo(cart_double);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (nlhs > 0 && use_single)
    {
        auto tmp = arma::conv_to<arma::fmat>::from(geo_double.slice(0));
        plhs[0] = qd_mex_copy2matlab(&tmp);
    }
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&geo_double.slice(0));

    if (nlhs > 1 && use_single)
    {
        auto tmp = arma::conv_to<arma::fmat>::from(geo_double.slice(1));
        plhs[1] = qd_mex_copy2matlab(&tmp);
    }
    else if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&geo_double.slice(1));

    if (nlhs > 2 && use_single)
    {
        auto tmp = arma::conv_to<arma::fmat>::from(geo_double.slice(2));
        plhs[2] = qd_mex_copy2matlab(&tmp);
    }
    else if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&geo_double.slice(2));
}
