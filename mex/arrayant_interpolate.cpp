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
#include "qd_arrayant_interpolate.hpp"

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - e_theta_re      Vertical component of the electric field, real part,            Size [n_elevation, n_azimuth, n_elements]
    //  1 - e_theta_im      Vertical component of the electric field, imaginary part,       Size [n_elevation, n_azimuth, n_elements]
    //  2 - e_phi_re        Horizontal component of the electric field, real part,          Size [n_elevation, n_azimuth, n_elements]
    //  3 - e_phi_im        Horizontal component of the electric field, imaginary part,     Size [n_elevation, n_azimuth, n_elements]
    //  4 - azimuth_grid    Azimuth angles in pattern (theta) in [rad], sorted,             Vector of length "n_azimuth"
    //  5 - elevation_grid  Elevation angles in pattern (phi) in [rad], sorted,             Vector of length "n_elevation"
    //  6 - azimuth         Azimuth angles for interpolation in [rad],                      Size [1, n_ang] or [n_out, n_ang]
    //  7 - elevation       Elevation angles for interpolation in [rad]                     Size [1, n_ang] or [n_out, n_ang]
    //  8 - i_element       Element indices, 1-based, optional, Default: 1:n_elements       Vector of length "n_out" or []
    //  9 - orientation     Orientation (bank, tilt, heading) in [rad], optional, Default: [0,0,0], Size [3, 1] or [3, n_out] or []
    // 10 - element_pos     Element positions, optional, Default: [0,0,0]                   Size [3, n_out] or []

    // Outputs:
    //  0 - V_re            Interpolated vertical field, real part,                         Size [n_out, n_ang]
    //  1 - V_im            Interpolated vertical field, imaginary part,                    Size [n_out, n_ang]
    //  2 - H_re            Interpolated horizontal field, real part,                       Size [n_out, n_ang]
    //  3 - H_im            Interpolated horizontal field, imaginary part,                  Size [n_out, n_ang]
    //  4 - dist            Effective distances, optional                                   Size [n_out, n_ang]
    //  5 - azimuth_loc     Azimuth angles [rad] in local antenna coordinates, optional,    Size [n_out, n_ang]
    //  6 - elevation_loc   Elevation angles [rad] in local antenna coordinates, optional,  Size [n_out, n_ang]

    if (nrhs < 8 || nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:no_input", "Incorrect number of input arguments.");

    if (nlhs < 4 || nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:no_output", "Incorrect number of output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Inputs must be provided in 'single' or 'double' precision.");

    // Create arrayant object and validate the input
    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    for (int i = 0; i < 6; i++)
    {
        unsigned n_dim = (unsigned)mxGetNumberOfDimensions(prhs[i]); // Number of dimensions - either 2 or 3
        const mwSize *dims = mxGetDimensions(prhs[i]);               // Read number of elements elements per dimension
        unsigned d1 = (unsigned)dims[0];                             // Number of elements on first dimension
        unsigned d2 = (unsigned)dims[1];                             // Number of elements on second dimension
        unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];             // Number of elements on third dimension
        bool not_empty = d1 * d2 * d3 > 0;

        if (not_empty && ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i]))))
            mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

        if (use_single)
        {
            if (not_empty && i == 0)
                arrayant_single.e_theta_re = arma::fcube((float *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 1)
                arrayant_single.e_theta_im = arma::fcube((float *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 2)
                arrayant_single.e_phi_re = arma::fcube((float *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 3)
                arrayant_single.e_phi_im = arma::fcube((float *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 4)
                arrayant_single.azimuth_grid = arma::fvec((float *)mxGetData(prhs[i]), d1 * d2 * d3, false, true);
            else if (not_empty && i == 5)
                arrayant_single.elevation_grid = arma::fvec((float *)mxGetData(prhs[i]), d1 * d2 * d3, false, true);
        }
        else // double
        {
            if (not_empty && i == 0)
                arrayant_double.e_theta_re = arma::cube((double *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 1)
                arrayant_double.e_theta_im = arma::cube((double *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 2)
                arrayant_double.e_phi_re = arma::cube((double *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 3)
                arrayant_double.e_phi_im = arma::cube((double *)mxGetData(prhs[i]), d1, d2, d3, false, true);
            else if (not_empty && i == 4)
                arrayant_double.azimuth_grid = arma::vec((double *)mxGetData(prhs[i]), d1 * d2 * d3, false, true);
            else if (not_empty && i == 5)
                arrayant_double.elevation_grid = arma::vec((double *)mxGetData(prhs[i]), d1 * d2 * d3, false, true);
        }
    }

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:import_error", error_message.c_str());

    if (mxGetNumberOfElements(prhs[6]) == 0 || mxGetNumberOfElements(prhs[7]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:import_error", "Inputs cannot be empty.");

    unsigned n_elevation = use_single ? arrayant_single.n_elevation() : arrayant_double.n_elevation();
    unsigned n_azimuth = use_single ? arrayant_single.n_azimuth() : arrayant_double.n_azimuth();
    unsigned n_elements = use_single ? arrayant_single.n_elements() : arrayant_double.n_elements();

    unsigned n_out = (unsigned)mxGetM(prhs[6]); // Number of rows in "azimuth"
    unsigned n_ang = (unsigned)mxGetN(prhs[6]); // Number of angles for interpolation

    if ((unsigned)mxGetM(prhs[7]) != n_out || (unsigned)mxGetN(prhs[7]) != n_ang)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Number of elements in 'elevation' does not match number of elements in 'azimuth'.");

    // Convert input arguments to armadillo objects (inputs from MATLAB are read-only)
    arma::fmat azimuth_single, elevation_single;
    arma::mat azimuth_double, elevation_double;
    if (use_single)
        azimuth_single = arma::fmat((float *)mxGetData(prhs[6]), n_out, n_ang, false, true),
        elevation_single = arma::fmat((float *)mxGetData(prhs[7]), n_out, n_ang, false, true);
    else
        azimuth_double = arma::mat((double *)mxGetData(prhs[6]), n_out, n_ang, false, true),
        elevation_double = arma::mat((double *)mxGetData(prhs[7]), n_out, n_ang, false, true);

    // Process optional input : i_element
    arma::Col<unsigned> i_element;
    if (nrhs < 9 || mxGetNumberOfElements(prhs[8]) == 0)
        i_element = arma::regspace<arma::Col<unsigned>>(1, n_elements);
    else if (mxIsClass(prhs[8], "uint32"))
        i_element = arma::Col<unsigned>((unsigned *)mxGetData(prhs[8]), mxGetNumberOfElements(prhs[8])); // Creates a copy
    else if (mxIsDouble(prhs[8]))
        i_element = arma::conv_to<arma::Col<unsigned>>::from(arma::vec((double *)mxGetData(prhs[8]), mxGetNumberOfElements(prhs[8]), false, true));
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Input 'i_element' must be given in double or uint32 precision.");

    if (n_out != 1 && n_out != i_element.n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Number of rows in 'azimuth' and 'elevation' must be 1 or match the number of elements.");

    // Check if values are valid (using lambda function)
    auto fnc_iel = [n_elements](unsigned &val)
    {
        if (val < 1 || val > n_elements)
            mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:out_of_bound", "Input 'i_element' must have values between 1 and 'n_elements'.");
    };
    i_element.for_each(fnc_iel);
    n_out = i_element.n_elem;

    // Process optional input : orientation
    arma::fcube orientation_single;
    arma::cube orientation_double;
    unsigned n_dim = 2, n_orientation2 = 0, n_orientation3 = 0;
    const mwSize *dims;
    if (nrhs >= 10)
        n_dim = (unsigned)mxGetNumberOfDimensions(prhs[9]),
        dims = mxGetDimensions(prhs[9]),
        n_orientation2 = (unsigned)dims[1],
        n_orientation3 = n_dim < 3 ? 1 : (unsigned)dims[2];

    if (nrhs < 10 || mxGetNumberOfElements(prhs[9]) == 0)
        if (use_single)
            orientation_single = arma::fcube(3, 1, 1, arma::fill::zeros);
        else
            orientation_double = arma::cube(3, 1, 1, arma::fill::zeros);
    else if ((unsigned)*dims != 3)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 3 elements on the first dimension.");
    else if (n_orientation2 != 1 && n_orientation2 != n_out)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.");
    else if (n_orientation3 != 1 && n_orientation3 != n_ang)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.");
    else if (use_single && mxIsSingle(prhs[9]))
        orientation_single = arma::fcube((float *)mxGetData(prhs[9]), 3, n_orientation2, n_orientation3, false, true);
    else if (!use_single && mxIsDouble(prhs[9]))
        orientation_double = arma::cube((double *)mxGetData(prhs[9]), 3, n_orientation2, n_orientation3, false, true);
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Input 'orientation' must be given in double or single precision.");

    // Process optional input : element_pos
    arma::fmat element_pos_single;
    arma::mat element_pos_double;
    if (nrhs < 11 || mxGetNumberOfElements(prhs[10]) == 0)
        if (use_single)
            element_pos_single = arma::fmat(3, n_out, arma::fill::zeros);
        else
            element_pos_double = arma::mat(3, n_out, arma::fill::zeros);
    else if (mxGetM(prhs[10]) != 3)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'element_pos' must have 3 rows.");
    else if (mxGetN(prhs[10]) != n_out)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'element_pos' must have 'n_elements' columns.");
    else if (use_single && mxIsSingle(prhs[10]))
        element_pos_single = arma::fmat((float *)mxGetData(prhs[10]), 3, (unsigned)mxGetN(prhs[10]));
    else if (!use_single && mxIsDouble(prhs[10]))
        element_pos_double = arma::mat((double *)mxGetData(prhs[10]), 3, (unsigned)mxGetN(prhs[10]));
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Input 'element_pos' must be given in double or single precision.");

    // Generate output variables (read and write access)
    if (use_single)
        plhs[0] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        plhs[1] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        plhs[2] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        plhs[3] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL);
    else
        plhs[0] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        plhs[1] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        plhs[2] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        plhs[3] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL);

    // Convert to armadillo objects
    arma::fmat V_re_single, V_im_single, H_re_single, H_im_single;
    arma::mat V_re_double, V_im_double, H_re_double, H_im_double;
    if (use_single)
        V_re_single = arma::fmat((float *)mxGetData(plhs[0]), n_out, n_ang, false, true),
        V_im_single = arma::fmat((float *)mxGetData(plhs[1]), n_out, n_ang, false, true),
        H_re_single = arma::fmat((float *)mxGetData(plhs[2]), n_out, n_ang, false, true),
        H_im_single = arma::fmat((float *)mxGetData(plhs[3]), n_out, n_ang, false, true);
    else
        V_re_double = arma::mat((double *)mxGetData(plhs[0]), n_out, n_ang, false, true),
        V_im_double = arma::mat((double *)mxGetData(plhs[1]), n_out, n_ang, false, true),
        H_re_double = arma::mat((double *)mxGetData(plhs[2]), n_out, n_ang, false, true),
        H_im_double = arma::mat((double *)mxGetData(plhs[3]), n_out, n_ang, false, true);

    // Optional output: dist
    arma::fmat dist_single;
    arma::mat dist_double;
    if (nlhs < 5 && use_single)
        dist_single = arma::fmat(0, 0);
    else if (nlhs < 5 && !use_single)
        dist_double = arma::mat(0, 0);
    else if (use_single)
        plhs[4] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        dist_single = arma::fmat((float *)mxGetData(plhs[4]), n_out, n_ang, false, true);
    else
        plhs[4] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        dist_double = arma::mat((double *)mxGetData(plhs[4]), n_out, n_ang, false, true);

    // Optional output "azimuth_loc"
    arma::fmat azimuth_loc_single;
    arma::mat azimuth_loc_double;
    if (nlhs < 6 && use_single)
        azimuth_loc_single = arma::fmat(0, 0);
    else if (nlhs < 6 && !use_single)
        azimuth_loc_double = arma::mat(0, 0);
    else if (use_single)
        plhs[5] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        azimuth_loc_single = arma::fmat((float *)mxGetData(plhs[5]), n_out, n_ang, false, true);
    else
        plhs[5] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        azimuth_loc_double = arma::mat((double *)mxGetData(plhs[5]), n_out, n_ang, false, true);

    // Optional output "elevation_loc"
    arma::fmat elevation_loc_single;
    arma::mat elevation_loc_double;
    if (nlhs < 7 && use_single)
        elevation_loc_single = arma::fmat(0, 0);
    else if (nlhs < 7 && !use_single)
        elevation_loc_double = arma::mat(0, 0);
    else if (use_single)
        plhs[6] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        elevation_loc_single = arma::fmat((float *)mxGetData(plhs[6]), n_out, n_ang, false, true);
    else
        plhs[6] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        elevation_loc_double = arma::mat((double *)mxGetData(plhs[6]), n_out, n_ang, false, true);

    // Call private library function
    if (use_single)
        qd_arrayant_interpolate(&arrayant_single,
                                &azimuth_single, &elevation_single,
                                &i_element, &orientation_single, &element_pos_single,
                                &V_re_single, &V_im_single, &H_re_single, &H_im_single, &dist_single,
                                &azimuth_loc_single, &elevation_loc_single);
    else
        qd_arrayant_interpolate(&arrayant_double,
                                &azimuth_double, &elevation_double,
                                &i_element, &orientation_double, &element_pos_double,
                                &V_re_double, &V_im_double, &H_re_double, &H_im_double, &dist_double,
                                &azimuth_loc_double, &elevation_loc_double);
}
