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
#include <cstring> // For memcopy

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - array_type      Array type (string)
    //  1-10                Additional parameters

    // Outputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - element_pos     Element positions                                               Size [3, n_elements]
    //  7 - coupling_re     Coupling matrix, real part                                      Size [n_elements, n_ports]
    //  8 - coupling_im     Coupling matrix, imaginary part                                 Size [n_elements, n_ports]
    //  9 - center_frequency   Center frequency in [Hz]                                     Scalar
    // 10 - name            Name of the array antenna object, string

    // Number of in and outputs
    if (nlhs < 11 || nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Input 'array_type' must be a string");

    std::string array_type = mxArrayToString(prhs[0]);
    quadriga_lib::arrayant<double> arrayant_double;

    if (array_type == "omni")
        arrayant_double.generate_omni();
    else if (array_type == "dipole" || array_type == "short-dipole")
        arrayant_double.generate_dipole();
    else if (array_type == "half-wave-dipole")
        arrayant_double.generate_half_wave_dipole();
    // else if (array_type == "custom")
    //     if (nrhs < 4)
    //         mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");
    //     else
    //         arrayant_double.generate_custom(qd_mex_get_scalar<double>(prhs[1], "az_3dB"),
    //                                         qd_mex_get_scalar<double>(prhs[2], "el_3db"),
    //                                         qd_mex_get_scalar<double>(prhs[3], "rear_gain_lin"));
    else
        mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Array type not supported!");

    // Write to MATLAB
    plhs[0] = qd_mex_copy2matlab(&arrayant_double.e_theta_re);
    plhs[1] = qd_mex_copy2matlab(&arrayant_double.e_theta_im);
    plhs[2] = qd_mex_copy2matlab(&arrayant_double.e_phi_re);
    plhs[3] = qd_mex_copy2matlab(&arrayant_double.e_phi_im);
    plhs[4] = qd_mex_copy2matlab(&arrayant_double.azimuth_grid, true);
    plhs[5] = qd_mex_copy2matlab(&arrayant_double.elevation_grid, true);
    plhs[6] = qd_mex_copy2matlab(&arrayant_double.element_pos);
    plhs[7] = qd_mex_copy2matlab(&arrayant_double.coupling_re);
    plhs[8] = qd_mex_copy2matlab(&arrayant_double.coupling_im);
    plhs[9] = qd_mex_copy2matlab(&arrayant_double.center_frequency);
    plhs[10] = mxCreateString(arrayant_double.name.c_str());
}