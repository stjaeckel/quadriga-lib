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
# GEO2CART
Transform Geographic (az, el, length) to Cartesian (x,y,z) coordinates coordinates

## Description:
This function transforms Geographic (azimuth, elevation, length) coordinates to Cartesian (x,y,z)
coordinates. A geographic coordinate system is a three-dimensional reference system that locates
points on the surface of a sphere. A point has three coordinate values: azimuth, elevation and length
where azimuth and elevation measure angles. In the geographic coordinate system, the elevation angle
θ = 90◦ points to the zenith and θ = 0◦ points to the horizon.

## Usage:

```
cart = arrayant_lib.geo2cart( azimuth, elevation, length )
```

## Input Arguments:

- **`azimuth`**<br>
  Azimuth angles in [rad], values between -pi and pi.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`elevation`**<br>
  Elevation angles in [rad], values between -pi/2 and pi/2.
  Single or double precision (same as input), Size `[n_row, n_col]`

- **`length`** (optional)<br>
  Vector length, i.e. the distance from the origin to the point defined by x,y,z.
  Single or double precision (same as input), Size `[n_row, n_col]` or empty `[]`

## Output Argument:

- **`cart`**<br>
  Cartesian coordinates (x,y,z)
  Single or double precision, Size: `[3, n_row, n_col]`

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:          azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector, optional,         Size [n_row, n_col]
    // Output:          cart            Cartesian coordinates,                  Size [3, n_row, n_col]

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:no_input", "Need at least one input argument.");

    if (nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:no_input", "Too many input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:no_output", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:IO_error", "Inputs must be provided in 'single' or 'double' precision.");

    if (use_single && nlhs > 0)
    {
        arma::fmat azimuth = qd_mex_reinterpret_Mat<float>(prhs[0]);
        arma::fmat elevation, length;

        if (nrhs > 1 && mxIsSingle(prhs[1]))
            elevation = qd_mex_reinterpret_Mat<float>(prhs[1]);
        else if (nrhs > 1)
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:IO_error", "Input 'elevation' must have same type as 'azimuth'.");
        else
            elevation = arma::fmat(azimuth.n_rows, azimuth.n_cols);

        if (nrhs > 2 && mxIsSingle(prhs[2]))
            length = qd_mex_reinterpret_Mat<float>(prhs[2]);
        else if (nrhs > 2)
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:IO_error", "Input 'length' must have same type as 'azimuth'.");
        else
            length = arma::fmat(azimuth.n_rows, azimuth.n_cols, arma::fill::ones);

        try
        {
            auto tmp = quadriga_lib::geo2cart(azimuth, elevation, length);
            auto cart = arma::conv_to<arma::fcube>::from(tmp);
            plhs[0] = qd_mex_copy2matlab(&cart);
        }
        catch (const std::invalid_argument &ex)
        {
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:unknown_error", ex.what());
        }
        catch (...)
        {
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:unknown_error", "Unknown failure occurred. Possible memory corruption!");
        }
    }
    else if (nlhs > 0) // double
    {
        arma::mat azimuth = qd_mex_reinterpret_Mat<double>(prhs[0]);
        arma::mat elevation, length;

        if (nrhs > 1 && mxIsDouble(prhs[1]))
            elevation = qd_mex_reinterpret_Mat<double>(prhs[1]);
        else if (nrhs > 1)
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:IO_error", "Input 'elevation' must have same type as 'azimuth'.");
        else
            elevation = arma::mat(azimuth.n_rows, azimuth.n_cols);

        if (nrhs > 2 && mxIsDouble(prhs[2]))
            length = qd_mex_reinterpret_Mat<double>(prhs[2]);
        else if (nrhs > 2)
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:IO_error", "Input 'length' must have same type as 'azimuth'.");
        else
            length = arma::mat(azimuth.n_rows, azimuth.n_cols, arma::fill::ones);
        try
        {
            auto cart = quadriga_lib::geo2cart(azimuth, elevation, length);
            plhs[0] = qd_mex_copy2matlab(&cart);
        }
        catch (const std::invalid_argument &ex)
        {
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:unknown_error", ex.what());
        }
        catch (...)
        {
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:unknown_error", "Unknown failure occurred. Possible memory corruption!");
        }
    }
}
