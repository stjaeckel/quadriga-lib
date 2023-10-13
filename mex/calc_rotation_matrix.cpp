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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input:       orientation      Orientation vectors (rows = bank, tilt, heading) in [rad], Size [3, n_row, n_col]
    //              invert_y_axis    If set to 1, the rotation around the y-axis is inverted
    //              transpose        Transpose the output
    // Output:      rotation         Rotation matrix, Size [9, n_row, n_col]

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:no_input", "Incorrect number of input arguments.");

    if (nlhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:no_output", "Incorrect number of output arguments.");

    if (mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:empty", "Input cannot be empty.");

    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(prhs[0]); // Number of dimensions in "orientation"
    const mwSize *dims = mxGetDimensions(prhs[0]);               // Read number of elements elements per dimension
    unsigned n_row = (unsigned)dims[1];                          // Number of rows
    unsigned n_col = n_dim < 3 ? 1 : (unsigned)dims[2];          // Number of columns

    if ((unsigned)dims[0] != 3)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:size_mismatch", "Input must have 3 elements on the first dimension.");

    // Read scalar variables
    bool invert_y_axis = nrhs < 2 ? false : qd_mex_get_scalar<bool>(prhs[1], "invert_y_axis");
    bool transpose = nrhs < 3 ? false : qd_mex_get_scalar<bool>(prhs[2], "transpose");

    mwSize dims_out[3] = {9, n_row, n_col};
    if (mxIsSingle(prhs[0]))
    {
        const arma::fcube orientation = arma::fcube((float *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube Rd = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose); // double precision output
        arma::fcube R = arma::conv_to<arma::fcube>::from(Rd);                                        // conversion to single
        plhs[0] = mxCreateNumericArray(3, dims_out, mxSINGLE_CLASS, mxREAL);
        std::memcpy((float *)mxGetData(plhs[0]), R.memptr(), sizeof(float) * R.n_elem);
    }
    else if (mxIsDouble(prhs[0]))
    {
        const arma::cube orientation = arma::cube((double *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube R = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose);
        plhs[0] = mxCreateNumericArray(3, dims_out, mxDOUBLE_CLASS, mxREAL);
        std::memcpy((double *)mxGetData(plhs[0]), R.memptr(), sizeof(double) * R.n_elem);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:wrong_type", "Input must be provided in 'single' or 'double' precision.");
}
