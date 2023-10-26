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
#include <cstring>

/*!MATLAB
%CART2GEO Transform Cartesian coordinates to geographic coordinates
%
% Description:
%   [azimuth, elevation, length] = arrayant_lib.cart2geo(cart) 
%   transforms corresponding elements of the Cartesian coordinate system (x, y, and z) to geographic 
%   coordinates azimuth, elevation, and length.
%
% Input:
%   cart
%   Cartesian coordinates (x,y,z)
%   Single or double precision, Size: [3, n_row, n_col]
%
% Outputs:
%   azimuth
%   Azimuth angles in [rad], values between -pi and pi.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%   elevation
%   Elevation angles in [rad], values between -pi/2 and pi/2.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%   length
%   Vector length, i.e. the distance from the origin to the point defined by x,y,z.
%   Single or double precision (same as input), Size [n_row, n_col]
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
% All rights reserved.
%
% e-mail: info@sjc-wireless.com
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
% in compliance with the License. You may obtain a copy of the License at
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software distributed under the License
% is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
% or implied. See the License for the specific language governing permissions and limitations under
% the License.
MATLAB!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input:           cart            Cartesian coordinates,                  Size [3, n_row, n_col]
    // Outputs:         azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector, optional,         Size [n_row, n_col]

    if (nrhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_input", "Incorrect number of input arguments.");

    if (nlhs < 2 || nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:no_output", "Incorrect number of output arguments.");

    if (mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:empty", "Input cannot be empty.");

    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(prhs[0]); // Number of dimensions in pattern
    const mwSize *dims = mxGetDimensions(prhs[0]);               // Read number of elements elements per dimension
    unsigned n_row = (unsigned)dims[1], n_col = 1;
    if (n_dim == 3)
        n_col = (unsigned)dims[2];
    else if (n_dim > 3)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:size_mismatch", "Input must have 2 or 3 dimensions.");

    if (dims[0] != 3)
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:size_mismatch", "Input must have 3 rows.");

    if (mxIsSingle(prhs[0]))
    {
        const arma::fcube cart = arma::fcube((float *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube geo_double = quadriga_lib::cart2geo(cart);           // double precision output
        arma::fcube geo = arma::conv_to<arma::fcube>::from(geo_double); // conversion to single
        const float *ptr = geo.memptr();

        plhs[0] = mxCreateNumericMatrix(n_row, n_col, mxSINGLE_CLASS, mxREAL); // Azimuth
        plhs[1] = mxCreateNumericMatrix(n_row, n_col, mxSINGLE_CLASS, mxREAL); // Elevation
        if (nlhs > 2)
            plhs[2] = mxCreateNumericMatrix(n_row, n_col, mxSINGLE_CLASS, mxREAL); // Length
        std::memcpy((float *)mxGetData(plhs[0]), ptr, sizeof(float) * n_row * n_col);
        std::memcpy((float *)mxGetData(plhs[1]), &ptr[n_row * n_col], sizeof(float) * n_row * n_col);
        if (nlhs > 2)
            std::memcpy((float *)mxGetData(plhs[2]), &ptr[2 * n_row * n_col], sizeof(float) * n_row * n_col);
    }
    else if (mxIsDouble(prhs[0]))
    {
        const arma::cube cart = arma::cube((double *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube geo = quadriga_lib::cart2geo(cart);
        const double *ptr = geo.memptr();

        plhs[0] = mxCreateNumericMatrix(n_row, n_col, mxDOUBLE_CLASS, mxREAL); // Azimuth
        plhs[1] = mxCreateNumericMatrix(n_row, n_col, mxDOUBLE_CLASS, mxREAL); // Elevation
        if (nlhs > 2)
            plhs[2] = mxCreateNumericMatrix(n_row, n_col, mxDOUBLE_CLASS, mxREAL); // Length
        std::memcpy((double *)mxGetData(plhs[0]), ptr, sizeof(double) * n_row * n_col);
        std::memcpy((double *)mxGetData(plhs[1]), &ptr[n_row * n_col], sizeof(double) * n_row * n_col);
        if (nlhs > 2)
            std::memcpy((double *)mxGetData(plhs[2]), &ptr[2 * n_row * n_col], sizeof(double) * n_row * n_col);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:cart2geo:wrong_type", "Input must be provided in 'single' or 'double' precision.");
}
