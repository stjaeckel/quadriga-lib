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
#include "quadriga_lib.hpp"
#include "mex_helper_functions.cpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_out] or []
    //  7 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  8 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    //  9 - center_frequency   Center frequency in [Hz]                                     Scalar

    // Outputs:
    //  0 - e_theta_re_c    Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_ports]
    //  1 - e_theta_im_c    Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_ports]
    //  2 - e_phi_re_c      Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_ports]
    //  3 - e_phi_im_c      Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_ports]

    if (nlhs = !4 || nrhs < 9)
        mexErrMsgIdAndTxt("quadriga_lib:combine_pattern:IO_error", "Wrong number of input/output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:combine_pattern:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 9; i++)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0], true),
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1], true),
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2], true),
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3], true),
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]),
        arrayant_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[6], true),
        arrayant_single.coupling_re = qd_mex_reinterpret_Mat<float>(prhs[7], true),
        arrayant_single.coupling_im = qd_mex_reinterpret_Mat<float>(prhs[8], true),
        arrayant_single.center_frequency = qd_mex_get_scalar<float>(prhs[9], "center_frequency");
    else
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0], true),
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1], true),
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2], true),
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3], true),
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]),
        arrayant_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[6], true),
        arrayant_double.coupling_re = qd_mex_reinterpret_Mat<double>(prhs[7], true),
        arrayant_double.coupling_im = qd_mex_reinterpret_Mat<double>(prhs[8], true),
        arrayant_double.center_frequency = qd_mex_get_scalar<double>(prhs[9], "center_frequency");

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:combine_pattern:IO_error", error_message.c_str());

    // Call library function
    if (use_single)
        arrayant_single.combine_pattern();
    else
        arrayant_double.combine_pattern();

    // Write output
    if (use_single)
        plhs[0] = qd_mex_copy2matlab(&arrayant_single.e_theta_re),
        plhs[1] = qd_mex_copy2matlab(&arrayant_single.e_theta_im),
        plhs[2] = qd_mex_copy2matlab(&arrayant_single.e_phi_re),
        plhs[3] = qd_mex_copy2matlab(&arrayant_single.e_phi_im);
    else
        plhs[0] = qd_mex_copy2matlab(&arrayant_double.e_theta_re),
        plhs[1] = qd_mex_copy2matlab(&arrayant_double.e_theta_im),
        plhs[2] = qd_mex_copy2matlab(&arrayant_double.e_phi_re),
        plhs[3] = qd_mex_copy2matlab(&arrayant_double.e_phi_im);
}