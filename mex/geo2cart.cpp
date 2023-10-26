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
%GEO2CART Transforms geographic coordinates to Cartesian coordinates
%
% Description:
%   cart = arrayant_lib.cart2geo(azimuth, elevation, length) 
%   transforms corresponding elements of the Cartesian coordinate system (x, y, and z) to geographic 
%   coordinates azimuth, elevation, and length.
%
% Inputs:
%   azimuth
%   Azimuth angles in [rad], values between -pi and pi.
%   Single or double precision, Size [n_row, n_col]
%
%   elevation
%   Elevation angles in [rad], values between -pi/2 and pi/2.
%   Single or double precision, Size [n_row, n_col]
%
%   length
%   Vector length, i.e. the distance from the origin to the point defined by x,y,z. Optional. 
%   Single or double precision (same as azimuth), Size [n_row, n_col] or empty []
%
% Output:
%   cart
%   Cartesian coordinates (x,y,z)
%   Single or double precision (same as input), Size: [3, n_row, n_col]
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
    // Inputs:          azimuth         Azimuth angles in [rad],                Size [n_row, n_col]
    //                  elevation       Elevation angles in [rad],              Size [n_row, n_col]
    //                  length          Length of the vector, optional,         Size [n_row, n_col]
    // Output:          cart            Cartesian coordinates,                  Size [3, n_row, n_col]

    if (nrhs < 2 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:no_input", "Incorrect number of input arguments.");

    if (nlhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:no_output", "Incorrect number of output arguments.");

    if (mxGetNumberOfElements(prhs[0]) == 0 || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:empty", "Inputs 'azimuth' and 'elevation' cannot be empty.");

    unsigned n_row = (unsigned)mxGetM(prhs[0]);
    unsigned n_col = (unsigned)mxGetN(prhs[0]);

    if ((unsigned)mxGetM(prhs[1]) != n_row || (unsigned)mxGetN(prhs[1]) != n_col)
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:size_mismatch", "Number of elements in 'elevation' does not match number of elements in 'azimuth'.");

    if (nrhs > 2 && ((unsigned)mxGetM(prhs[2]) != n_row || (unsigned)mxGetN(prhs[2]) != n_col))
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:size_mismatch", "Number of elements in 'length' does not match number of elements in 'azimuth'.");

    const mwSize n_dim = 3;
    mwSize dims[3] = {3, n_row, n_col};

    if (mxIsSingle(prhs[0]) && mxIsSingle(prhs[1]))
    {
        const arma::fmat azimuth = arma::fmat((float *)mxGetData(prhs[0]), n_row, n_col, false, true);
        const arma::fmat elevation = arma::fmat((float *)mxGetData(prhs[1]), n_row, n_col, false, true);
        arma::fmat length;
        if (nrhs < 3)
            length = arma::fmat(n_row, n_col, arma::fill::ones);
        else if (mxIsSingle(prhs[2]))
            length = arma::fmat((float *)mxGetData(prhs[2]), n_row, n_col, false, true);
        else
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:wrong_type", "Inputs 'azimuth', 'elevation' and 'length' must have same type.");

        arma::cube cart_double = quadriga_lib::geo2cart(azimuth, elevation, length); // double precision output
        arma::fcube cart = arma::conv_to<arma::fcube>::from(cart_double);            // conversion to single
        plhs[0] = mxCreateNumericArray(n_dim, dims, mxSINGLE_CLASS, mxREAL);
        std::memcpy((float *)mxGetData(plhs[0]), cart.memptr(), sizeof(float) * cart.n_elem);
    }
    else if (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1]))
    {
        const arma::mat azimuth = arma::mat((double *)mxGetData(prhs[0]), n_row, n_col, false, true);
        const arma::mat elevation = arma::mat((double *)mxGetData(prhs[1]), n_row, n_col, false, true);
        arma::mat length;
        if (nrhs < 3)
            length = arma::mat(n_row, n_col, arma::fill::ones);
        else if (mxIsDouble(prhs[2]))
            length = arma::mat((double *)mxGetData(prhs[2]), n_row, n_col, false, true);
        else
            mexErrMsgIdAndTxt("quadriga_lib:geo2cart:wrong_type", "Inputs 'azimuth', 'elevation' and 'length' must have same type.");

        arma::cube cart = quadriga_lib::geo2cart(azimuth, elevation, length);
        plhs[0] = mxCreateNumericArray(n_dim, dims, mxDOUBLE_CLASS, mxREAL);
        std::memcpy((double *)mxGetData(plhs[0]), cart.memptr(), sizeof(double) * cart.n_elem);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:geo2cart:wrong_type", "Inputs 'azimuth' or 'elevation' must have same type (single or double).");
}
