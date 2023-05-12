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
#include <cstring> // For memcopy

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    // x    Vector of x sample points; size [ 1, nx ] or [ nx, 1 ]
    // y    Vector of y sample points; size [ 1, ny ] or [ ny, 1 ]
    // z    The input data matrix; size [ ny, nx, ne ]; the 3rd dimension allows for interpolations of multiple data-sets; for one-dimensional interpolation, the size must be [ 1, nx, ne ]
    // xc   Vector of x sample points after interpolation; size [ 1, nxi ] or [ nxi , 1 ]
    // yc   Vector of y sample points after interpolation; size [ 1, nyi ] or [ nyi, 1 ]

    // Output:
    // zi   The interpolated data; size [ nyi, nxi, ne ]

    // Number of in and outputs
    if (nlhs != 1 || nrhs < 4)
        mexErrMsgIdAndTxt("quadriga_lib:interp:no_input", "Wrong number of input/output arguments.");

    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "Inputs must be provided in 'single' or 'double' precision.");

    if ((use_single && !mxIsSingle(prhs[2])) || (!use_single && !mxIsDouble(prhs[2])))
        mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "All inputs must have the same type: 'single' or 'double' precision");

    if ((use_single && !mxIsSingle(prhs[3])) || (!use_single && !mxIsDouble(prhs[3])))
        mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "All inputs must have the same type: 'single' or 'double' precision");

    if (mxGetNumberOfElements(prhs[0]) == 0 || mxGetNumberOfElements(prhs[2]) == 0 || mxGetNumberOfElements(prhs[3]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "Inputs 'x', 'z', 'xc' cannot be empty.");

    arma::fcube input_single, output_single;
    arma::fvec xi_single, yi_single(1), xo_single, yo_single(1);

    arma::cube input_double, output_double;
    arma::vec xi_double, yi_double(1), xo_double, yo_double(1);

    if (use_single)
        xi_single = qd_mex_reinterpret_Col<float>(prhs[0]),
        input_single = qd_mex_reinterpret_Cube<float>(prhs[2]),
        xo_single = qd_mex_reinterpret_Col<float>(prhs[3]);
    else
        xi_double = qd_mex_reinterpret_Col<double>(prhs[0]),
        input_double = qd_mex_reinterpret_Cube<double>(prhs[2]),
        xo_double = qd_mex_reinterpret_Col<double>(prhs[3]);

    if (mxGetNumberOfElements(prhs[1]) != 0)
        if (use_single && mxIsSingle(prhs[1]))
            yi_single = qd_mex_reinterpret_Col<float>(prhs[1]);
        else if (!use_single && mxIsDouble(prhs[1]))
            yi_double = qd_mex_reinterpret_Col<double>(prhs[1]);
        else
            mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "All inputs must have the same type: 'single' or 'double' precision");

    if (nrhs > 4 && mxGetNumberOfElements(prhs[4]) != 0)
        if (use_single && mxIsSingle(prhs[4]))
            yo_single = qd_mex_reinterpret_Col<float>(prhs[4]);
        else if (!use_single && mxIsDouble(prhs[4]))
            yo_double = qd_mex_reinterpret_Col<double>(prhs[4]);
        else
            mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "All inputs must have the same type: 'single' or 'double' precision");

    unsigned n_yo = use_single ? yo_single.n_elem : yo_double.n_elem;
    unsigned n_xo = use_single ? xo_single.n_elem : xo_double.n_elem;
    unsigned n_el = use_single ? input_single.n_slices : input_double.n_slices;

    mwSize dims[3] = {(mwSize)n_yo, (mwSize)n_xo, (mwSize)n_el};

    if (use_single)
        plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL),
        output_single = arma::fcube((float *)mxGetData(plhs[0]), n_yo, n_xo, n_el, false, true);
    else
        plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL),
        output_double = arma::cube((double *)mxGetData(plhs[0]), n_yo, n_xo, n_el, false, true);

    // Call 2D linear interpolation function
    std::string error_message;
    if (use_single)
        error_message = quadriga_tools::interp(&input_single, &xi_single, &yi_single, &xo_single, &yo_single, &output_single);
    else
        error_message = quadriga_tools::interp(&input_double, &xi_double, &yi_double, &xo_double, &yo_double, &output_double);

    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:interp:error", error_message.c_str());
}