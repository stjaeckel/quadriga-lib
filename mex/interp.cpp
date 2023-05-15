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
    if (mxIsSingle(prhs[2]) || mxIsDouble(prhs[2]))
        use_single = mxIsSingle(prhs[2]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:interp:wrong_type", "Inputs must be provided in 'single' or 'double' precision.");

    std::string error_message;
    if (use_single)
    {
        arma::fcube input = qd_mex_reinterpret_Cube<float>(prhs[2]);
        arma::fvec xi, yi(1), xo, yo(1);

        if (mxIsSingle(prhs[0]))
            xi = qd_mex_reinterpret_Col<float>(prhs[0]);
        else
            xi = qd_mex_typecast_Col<float>(prhs[0], "xi");

        if (mxGetNumberOfElements(prhs[1]) != 0)
            if (mxIsSingle(prhs[1]))
                yi = qd_mex_reinterpret_Col<float>(prhs[1]);
            else
                yi = qd_mex_typecast_Col<float>(prhs[1], "yi");

        if (mxIsSingle(prhs[3]))
            xo = qd_mex_reinterpret_Col<float>(prhs[3]);
        else
            xo = qd_mex_typecast_Col<float>(prhs[3], "xo");

        if (nrhs > 4 && mxGetNumberOfElements(prhs[4]) != 0)
            if (mxIsSingle(prhs[4]))
                yo = qd_mex_reinterpret_Col<float>(prhs[4]);
            else
                yo = qd_mex_typecast_Col<float>(prhs[4], "yo");

        unsigned n_yo = yo.n_elem, n_xo = xo.n_elem, n_el = input.n_slices;
        mwSize dims[3] = {(mwSize)n_yo, (mwSize)n_xo, (mwSize)n_el};

        plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
        arma::fcube output = arma::fcube((float *)mxGetData(plhs[0]), n_yo, n_xo, n_el, false, true);
        error_message = quadriga_tools::interp(&input, &xi, &yi, &xo, &yo, &output);
    }
    else
    {
        arma::cube input = qd_mex_reinterpret_Cube<double>(prhs[2]);
        arma::vec xi, yi(1), xo, yo(1);

        if (mxIsDouble(prhs[0]))
            xi = qd_mex_reinterpret_Col<double>(prhs[0]);
        else
            xi = qd_mex_typecast_Col<double>(prhs[0], "xi");

        if (mxGetNumberOfElements(prhs[1]) != 0)
            if (mxIsDouble(prhs[1]))
                yi = qd_mex_reinterpret_Col<double>(prhs[1]);
            else
                yi = qd_mex_typecast_Col<double>(prhs[1], "yi");

        if (mxIsDouble(prhs[3]))
            xo = qd_mex_reinterpret_Col<double>(prhs[3]);
        else
            xo = qd_mex_typecast_Col<double>(prhs[3], "xo");

        if (nrhs > 4 && mxGetNumberOfElements(prhs[4]) != 0)
            if (mxIsDouble(prhs[4]))
                yo = qd_mex_reinterpret_Col<double>(prhs[4]);
            else
                yo = qd_mex_typecast_Col<double>(prhs[4], "yo");

        unsigned n_yo = yo.n_elem, n_xo = xo.n_elem, n_el = input.n_slices;
        mwSize dims[3] = {(mwSize)n_yo, (mwSize)n_xo, (mwSize)n_el};

        plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        arma::cube output = arma::cube((double *)mxGetData(plhs[0]), n_yo, n_xo, n_el, false, true);

        error_message = quadriga_tools::interp(&input, &xi, &yi, &xo, &yo, &output);
    }

    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:interp:error", error_message.c_str());
}