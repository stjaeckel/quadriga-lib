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
    //  0 - fn              Filename of the OBJ file (including path); string
    //  1 - use_single      Switch for single precision, default = 0 (double precision)

    // Output:
    //  0 - mesh            Vertices of the triangular mesh, Size [ no_mesh, 9 ]
    //  1 - mtl_prop        Material properties of each mesh element; Size [ no_mesh, 5 ]
    //  2 - vert_list       List of vertices found in the OBJ file; Size [ no_vert, 3 ]
    //  3 - face_ind        Face indices (=entries in vert_list); uint32; 1-based; Size [ no_mesh, 3 ]
    //  4 - obj_ind         Object index; uint32; Size [ no_mesh, 1 ]
    //  5 - mtl_ind         Material index; uint32; Size [ no_mesh, 1 ]

    // Notes:
    // Material values are:
    //     * Real part of relative permittivity at f = 1 GHz (a)
    //     * Frequency dependence of rel. permittivity (b) such that ϵ = a · f^b
    //     * Conductivity at f = 1 GHz (c)
    //     * Frequency dependence of conductivity (d) such that σ = c· f^d
    //     * Fixed attenuation in dB

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Filename is missing.");

    if (nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Too many input arguments.");

    if (nlhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:io_error", "Too many output arguments.");

    // Read filename
    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read "use_single"
    bool use_single = nrhs < 2 ? false : qd_mex_get_scalar<bool>(prhs[1], "use_single", false);

    // Declare Armadillo variables
    arma::Mat<double> mesh_double, mtl_prop_double, vert_list_double;
    arma::Mat<float> mesh_single, mtl_prop_single, vert_list_single;
    arma::Mat<unsigned> face_ind;
    arma::Col<unsigned> obj_ind, mtl_ind;

    // Read data from file
    try
    {
        if (use_single)
            quadriga_lib::obj_file_read<float>(fn, &mesh_single, &mtl_prop_single, &vert_list_single, &face_ind, &obj_ind, &mtl_ind);
        else
            quadriga_lib::obj_file_read<double>(fn, &mesh_double, &mtl_prop_double, &vert_list_double, &face_ind, &obj_ind, &mtl_ind);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:obj_file_read:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Convert 0-based indexing to 1-based indexing
    face_ind = face_ind + 1;

    // Write to MATLAB / Octave
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_copy2matlab(&mesh_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&mesh_double);

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_single);
    else if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_double);

    if (nlhs > 2 && use_single)
        plhs[2] = qd_mex_copy2matlab(&vert_list_single);
    else if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&vert_list_double);

    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&face_ind);

    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&obj_ind);

    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&mtl_ind);
}
