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
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Input 'array_type' must be a string");

    std::string array_type = mxArrayToString(prhs[0]);
    quadriga_lib::arrayant<double> arrayant_double;

    if (array_type == "omni")
        arrayant_double = quadriga_lib::generate_arrayant_omni<double>();
    else if (array_type == "dipole" || array_type == "short-dipole")
        arrayant_double = quadriga_lib::generate_arrayant_dipole<double>();
    else if (array_type == "half-wave-dipole")
        arrayant_double = quadriga_lib::generate_arrayant_half_wave_dipole<double>();
    else if (array_type == "xpol")
        arrayant_double = quadriga_lib::generate_arrayant_xpol<double>();
    else if (array_type == "custom")
        if (nrhs < 4)
            mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");
        else
            arrayant_double = quadriga_lib::generate_arrayant_custom<double>(qd_mex_get_scalar<double>(prhs[1], "az_3dB", 90.0),
                                                                             qd_mex_get_scalar<double>(prhs[2], "el_3db", 90.0),
                                                                             qd_mex_get_scalar<double>(prhs[3], "rear_gain_lin", 0.0));
    else if (array_type == "3GPP" || array_type == "3gpp")
    {
        unsigned M = nrhs < 2 ? 1 : qd_mex_get_scalar<unsigned>(prhs[1], "M", 1); 
        unsigned N = nrhs < 3 ? 1 : qd_mex_get_scalar<unsigned>(prhs[2], "N", 1);
        double center_freq = nrhs < 4 ? 299792458.0 : qd_mex_get_scalar<double>(prhs[3], "center_freq", 299792458.0);
        unsigned pol = nrhs < 5 ? 1 : qd_mex_get_scalar<unsigned>(prhs[4], "pol", 1);
        double tilt = nrhs < 6 ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "tilt", 0.0);
        double spacing = nrhs < 7 ? 0.5 : qd_mex_get_scalar<double>(prhs[6], "spacing", 0.5);
        unsigned Mg = nrhs < 8 ? 1 : qd_mex_get_scalar<unsigned>(prhs[7], "Mg", 1);
        unsigned Ng = nrhs < 9 ? 1 : qd_mex_get_scalar<unsigned>(prhs[8], "Ng", 1);
        double dgv = nrhs < 10 ? 0.5 : qd_mex_get_scalar<double>(prhs[9], "dgv", 0.5);
        double dgh = nrhs < 11 ? 0.5 : qd_mex_get_scalar<double>(prhs[10], "dgh", 0.5);
        if (nrhs < 12)
            arrayant_double = quadriga_lib::generate_arrayant_3GPP<double>(M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh);
        else if (nrhs < 17)
            mexErrMsgIdAndTxt("quadriga_lib:generate:no_input", "Wrong number of input/output arguments.");
        else if (mxIsDouble(prhs[11]) && mxIsDouble(prhs[12]) && mxIsDouble(prhs[13]) &&
                 mxIsDouble(prhs[14]) && mxIsDouble(prhs[15]) && mxIsDouble(prhs[16]))
        {
            quadriga_lib::arrayant<double> pattern;
            pattern.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[11]);
            pattern.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[12]);
            pattern.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[13]);
            pattern.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[14]);
            pattern.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[15]);
            pattern.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[16]);

            try
            {
                arrayant_double = quadriga_lib::generate_arrayant_3GPP<double>(M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &pattern);
            }
            catch (const std::invalid_argument &ex)
            {
                mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", ex.what());
            }
            catch (...)
            {
                mexErrMsgIdAndTxt("quadriga_lib:rotate_pattern:unknown_error", "Unknown failure occurred. Possible memory corruption!");
            }
        }
        else
            mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Custom antenna pattern must be provided in double precision.");
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:generate:wrong_type", "Array type not supported!");

    // Write to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&arrayant_double.e_theta_re);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&arrayant_double.e_theta_im);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&arrayant_double.e_phi_re);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&arrayant_double.e_phi_im);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&arrayant_double.azimuth_grid, true);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&arrayant_double.elevation_grid, true);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&arrayant_double.element_pos);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&arrayant_double.coupling_re);
    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&arrayant_double.coupling_im);
    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&arrayant_double.center_frequency);
    if (nlhs > 10)
        plhs[10] = mxCreateString(arrayant_double.name.c_str());
}