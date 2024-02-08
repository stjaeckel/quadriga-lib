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
    // Inputs:
    //  0 - orig            Ray origin points in GCS, Size [ n_pos, 3 ]
    //  1 - dest            Ray destination points in GCS, Size [ n_pos, 3 ]
    //  2 - mesh            Faces of the triangular mesh, Size: [ n_mesh, 9 ]
    //  3 - mtl_prop        Material properties, Size: [ n_mesh, 5 ]
    //  4 - center_freq     Center frequency in [Hz]
    //  5 - lod             Level of detail, scalar value 0-6
    //  6 - verbose         Verbosity level 0-2, default = 0

    // Outputs:
    //  0 - gain            Diffraction gain; linear scale; Size: [ n_pos ]
    //  1 - coord           Approximate coordinates of the diffracted path; [ 3, n_seg-1, n_pos ]

    if (nrhs < 4)
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:IO_error", "Need at least 4 input arguments.");

    if (nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:IO_error", "Too many input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 4; ++i)
        if (nrhs > i)
            if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
                mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Read inputs
    arma::fmat orig_single, dest_single, mesh_single, mtl_prop_single;
    arma::mat orig_double, dest_double, mesh_double, mtl_prop_double;

    if (use_single)
    {
        orig_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
        dest_single = qd_mex_reinterpret_Mat<float>(prhs[1]);
        mesh_single = qd_mex_reinterpret_Mat<float>(prhs[2]);
        mtl_prop_single = qd_mex_reinterpret_Mat<float>(prhs[3]);
    }
    else
    {
        orig_double = qd_mex_reinterpret_Mat<double>(prhs[0]);
        dest_double = qd_mex_reinterpret_Mat<double>(prhs[1]);
        mesh_double = qd_mex_reinterpret_Mat<double>(prhs[2]);
        mtl_prop_double = qd_mex_reinterpret_Mat<double>(prhs[3]);
    }

    double center_freq = (nrhs < 5) ? 1.0e9 : qd_mex_get_scalar<double>(prhs[4], "center_frequency", 1.0e9);
    int lod = (nrhs < 6) ? 2 : qd_mex_get_scalar<int>(prhs[5], "lod", 2);
    int verbose = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "verbose", 0);

    unsigned long long n_pos = use_single ? orig_single.n_rows : orig_double.n_rows;
    unsigned long long n_seg = 0;
    if (lod == 1 || lod == 2)
        n_seg = 2;
    else if (lod == 3)
        n_seg = 3;
    else if (lod == 4)
        n_seg = 4;
    else if (lod == 5 || lod == 6)
        n_seg = 1;

    // Initialize output containers
    arma::fvec gain_single;
    arma::vec gain_double;
    arma::fcube coord_single;
    arma::cube coord_double;

    // Get pointers
    arma::fvec *p_gain_single = &gain_single;
    arma::vec *p_gain_double = &gain_double;
    arma::fcube *p_coord_single = &coord_single;
    arma::cube *p_coord_double = &coord_double;

    // Allocate memory
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_init_output(p_gain_single, n_pos);
    else if (nlhs > 0) // double
        plhs[0] = qd_mex_init_output(p_gain_double, n_pos);
    else
        p_gain_single = nullptr, p_gain_double = nullptr;

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_init_output(p_coord_single, 3, n_seg, n_pos);
    else if (nlhs > 1) // double
        plhs[1] = qd_mex_init_output(p_coord_double, 3, n_seg, n_pos);
    else
        p_coord_single = nullptr, p_coord_double = nullptr;

    // Call library function
    try
    {
        if (use_single)
        {
            quadriga_lib::calc_diffraction_gain<float>(&orig_single, &dest_single, &mesh_single, &mtl_prop_single,
                                                       (float)center_freq, lod, p_gain_single, p_coord_single, verbose);
        }
        else // double
        {
            quadriga_lib::calc_diffraction_gain<double>(&orig_double, &dest_double, &mesh_double, &mtl_prop_double,
                                                        center_freq, lod, p_gain_double, p_coord_double, verbose);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:calc_diffraction_gain:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}