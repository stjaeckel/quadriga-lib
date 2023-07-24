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
    //  6 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_elements] or []
    //  7 - x_deg           The rotation angle around x-axis (bank angle) in [degrees]      Scalar
    //  8 - y_deg           The rotation angle around y-axis (tilt angle) in [degrees]      Scalar
    //  9 - z_deg           The rotation angle around z-axis (heading angle) in [degrees]   Scalar
    // 10 - usage           0: Rotate both (pattern+polarization), 1: Rotate only pattern, 2: Rotate only polarization, 3: as (0), but w/o grid adjusting

    // Outputs:
    //  0 - e_theta_re_r    Vertical component of the electric field, real part,            Size [n_elevation_r, n_azimuth_r, n_elements]
    //  1 - e_theta_im_r    Vertical component of the electric field, imaginary part,       Size [n_elevation_r, n_azimuth_r, n_elements]
    //  2 - e_phi_re_r      Horizontal component of the electric field, real part,          Size [n_elevation_r, n_azimuth_r, n_elements]
    //  3 - e_phi_im_r      Horizontal component of the electric field, imaginary part,     Size [n_elevation_r, n_azimuth_r, n_elements]
    //  4 - azimuth_grid_r    Azimuth angles in pattern (theta) in [rad], sorted,           Vector of length "n_azimuth_r"
    //  5 - elevation_grid_r  Elevation angles in pattern (phi) in [rad], sorted,           Vector of length "n_elevation_r"
    //  6 - element_pos_r     Element positions, optional, Default: [0,0,0]                 Size [3, n_elements]

    if (nlhs < 7 || nrhs < 7)
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Wrong number of input/output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    for (int i = 1; i < 7; i++)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    if (use_single)
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]),
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]),
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]),
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]),
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]),
        arrayant_single.element_pos = qd_mex_reinterpret_Mat<float>(prhs[6]),
        arrayant_single.read_only = true;
    else
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]),
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]),
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]),
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]),
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]),
        arrayant_double.element_pos = qd_mex_reinterpret_Mat<double>(prhs[6]),
        arrayant_double.read_only = true;

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:IO_error", error_message.c_str());

    double x_deg = nrhs < 8 ? 0.0 : qd_mex_get_scalar<double>(prhs[7], "x_deg");
    double y_deg = nrhs < 9 ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "y_deg");
    double z_deg = nrhs < 10 ? 0.0 : qd_mex_get_scalar<double>(prhs[9], "z_deg");
    unsigned usage = nrhs < 11 ? 0 : qd_mex_get_scalar<double>(prhs[10], "usage");

    quadriga_lib::arrayant<float> output_single;
    quadriga_lib::arrayant<double> output_double;

    // Call library function
    try
    {
        if (use_single)
            arrayant_single.rotate_pattern(float(x_deg), float(y_deg), float(z_deg), usage, -1, &output_single);
        else
            arrayant_double.rotate_pattern(x_deg, y_deg, z_deg, usage, -1, &output_double);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Copy output to MATLAB
    if (use_single)
        plhs[0] = qd_mex_copy2matlab(&output_single.e_theta_re),
        plhs[1] = qd_mex_copy2matlab(&output_single.e_theta_im),
        plhs[2] = qd_mex_copy2matlab(&output_single.e_phi_re),
        plhs[3] = qd_mex_copy2matlab(&output_single.e_phi_im),
        plhs[4] = qd_mex_copy2matlab(&output_single.azimuth_grid),
        plhs[5] = qd_mex_copy2matlab(&output_single.elevation_grid),
        plhs[6] = qd_mex_copy2matlab(&output_single.element_pos);
    else
        plhs[0] = qd_mex_copy2matlab(&output_double.e_theta_re),
        plhs[1] = qd_mex_copy2matlab(&output_double.e_theta_im),
        plhs[2] = qd_mex_copy2matlab(&output_double.e_phi_re),
        plhs[3] = qd_mex_copy2matlab(&output_double.e_phi_im),
        plhs[4] = qd_mex_copy2matlab(&output_double.azimuth_grid),
        plhs[5] = qd_mex_copy2matlab(&output_double.elevation_grid),
        plhs[6] = qd_mex_copy2matlab(&output_double.element_pos);
}