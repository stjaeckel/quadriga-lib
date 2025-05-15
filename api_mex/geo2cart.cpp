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
#include "mex_helper_functions.hpp"

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
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Need at least one input argument.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    if (mxIsSingle(prhs[0]) && nlhs > 0)
    {
        const arma::fmat azimuth = qd_mex_get_single_Mat(prhs[0]);
        const arma::fmat elevation = (nrhs > 1) ? qd_mex_get_single_Mat(prhs[1]) : arma::fmat(azimuth.n_rows, azimuth.n_cols);
        const arma::fmat length = (nrhs > 2) ? qd_mex_get_single_Mat(prhs[2]) : arma::fmat(azimuth.n_rows, azimuth.n_cols, arma::fill::ones);

        arma::fcube cart;
        CALL_QD(cart = quadriga_lib::geo2cart(azimuth, elevation, length));
        plhs[0] = qd_mex_copy2matlab(&cart);
    }
    else if (nlhs > 0)
    {
        const arma::mat azimuth = qd_mex_get_double_Mat(prhs[0]);
        const arma::mat elevation = (nrhs > 1) ? qd_mex_get_double_Mat(prhs[1]) : arma::mat(azimuth.n_rows, azimuth.n_cols);
        const arma::mat length = (nrhs > 2) ? qd_mex_get_double_Mat(prhs[2]) : arma::mat(azimuth.n_rows, azimuth.n_cols, arma::fill::ones);

        arma::cube cart;
        CALL_QD(cart = quadriga_lib::geo2cart(azimuth, elevation, length));
        plhs[0] = qd_mex_copy2matlab(&cart);
    }
}
