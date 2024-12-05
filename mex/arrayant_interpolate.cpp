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
#include "quadriga_arrayant.hpp"
#include "qd_arrayant_functions.hpp"
#include "mex_helper_functions.cpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_INTERPOLATE
Interpolate array antenna field patterns

## Description:
This function interpolates polarimetric antenna field patterns for a given set of azimuth and
elevation angles.

## Usage:

```
[V_re, V_im, H_re, H_im, dist, azimuth_loc, elevation_loc, gamma] = ...
    quadriga_lib.arrayant_interpolate( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, azimuth, elevation, i_element, orientation, element_pos )
```

## Input Arguments:
- **Antenna data:** (inputs 1-6, single or double precision)
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation, n_azimuth, n_elements]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation, n_azimuth, n_elements]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation]`
  
- **`azimuth`**<br>
  Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi and pi, single or double precision.
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`
  
- **`elevation`**<br>
  Elevation angles in [rad] for which the field pattern should be interpolated. Values must be
  between -pi/2 and pi/2, single or double precision.
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Size: `[1, n_ang]`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Size: `[n_out, n_ang]`
  
- **`i_element`**<br>
  The element indices for which the interpolation should be done. Optional parameter. Values must
  be between 1 and n_elements. It is possible to duplicate elements, i.e. by passing `[1,1,2]`.
  If this parameter is not provided (or an empty array is passed), `i_element` is initialized
  to `[1:n_elements]`. In this case, `n_out = n_elements`. Allowed types: uint32 or double.<br>
  Size: `[1, n_out]` or `[n_out, 1]` or empty `[]`
  
- **`orientation`**<br>
  This (optional) 3-element vector describes the orientation of the array antenna or of individual
  array elements. The The first value describes the ”bank angle”, the second value describes the
  ”tilt angle”, (positive values point upwards), the third value describes the bearing or ”heading
  angle”, in mathematic sense. Values must be given in [rad]. East corresponds to 0, and the
  angles increase counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. By
  default, the orientation is `[0,0,0]`', i.e. the broadside of the antenna points at the horizon
  towards the East. Single or double precision<br>
  Size: `[3, 1]` or `[3, n_out]` or `[3, 1, n_ang]` or `[3, n_out, n_ang]` or empty `[]`
  
- **`element_pos`**<br>
  Positions of the array antenna elements in local cartesian coordinates (using units of [m]).
  Optional parameter. If this parameter is not given, all elements are placed at the phase center
  of the array at coordinates `[0,0,0]'`. Otherwise, positions are given for the elements in the
  output of the interpolation function. For example, when duplicating the fist element by setting
  `i_element = [1,1]`, different element positions can be set for the two elements in the output.
  Single or double precision, <br>Size: `[3, n_out]` or empty `[]`
  
## Derived inputs:
  `n_azimuth`      | Number of azimuth angles in the filed pattern 
  `n_elevation`    | Number of elevation angles in the filed pattern 
  `n_elements`     | Number of antenna elements filed pattern of the array antenna
  `n_ang`          | Number of interpolation angles
  `n_out`          | Number of antenna elements in the generated output (may differ from n_elements)

## Output Arguments:
- **`V_re`**<br>
  Real part of the interpolated e-theta (vertical) field component.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`V_im`**<br>
  Imaginary part of the interpolated e-theta (vertical) field component.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`H_re`**<br>
  Real part of the interpolated e-phi (horizontal) field component.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`H_im`**<br>
  Imaginary part of the interpolated e-phi (horizontal) field component.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`dist`**<br>
  The effective distances between the antenna elements when seen from the direction of the
  incident path. The distance is calculated by an projection of the array positions on the normal
  plane of the incident path. This is needed for calculating the phase of the antenna response.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`azimuth_loc`**<br>
  The azimuth angles in [rad] for the local antenna coordinate system, i.e., after applying the
  'orientation'. If no orientation vector is given, these angles are identical to the input
  azimuth angles. Optional output.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`elevation_loc`**<br>
  The elevation angles in [rad] for the local antenna coordinate system, i.e., after applying the
  'orientation'. If no orientation vector is given, these angles are identical to the input
  elevation angles. Optional output.<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
  
- **`gamma`**<br>
  Polarization rotation angles in [rad]<br>
  Single or double precision (same as input), Size `[n_out, n_ang]`
MD!*/

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
    //  7 - gamma           Polarization rotation angles in [rad], optional,                Size [n_out, n_ang]

    if (nrhs < 8 || nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:no_input", "Incorrect number of input arguments.");

    if (nlhs < 4 || nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:no_output", "Incorrect number of output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Inputs must be provided in 'single' or 'double' precision.");

    for (int i = 1; i < 8; i++)
        if ((use_single && !mxIsSingle(prhs[i])) || (!use_single && !mxIsDouble(prhs[i])))
            mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    if (mxGetNumberOfElements(prhs[6]) == 0 || mxGetNumberOfElements(prhs[7]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:import_error", "Inputs cannot be empty.");

    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    arma::fmat azimuth_single, elevation_single;
    arma::mat azimuth_double, elevation_double;
    if (use_single)
        arrayant_single.e_theta_re = qd_mex_reinterpret_Cube<float>(prhs[0]),
        arrayant_single.e_theta_im = qd_mex_reinterpret_Cube<float>(prhs[1]),
        arrayant_single.e_phi_re = qd_mex_reinterpret_Cube<float>(prhs[2]),
        arrayant_single.e_phi_im = qd_mex_reinterpret_Cube<float>(prhs[3]),
        arrayant_single.azimuth_grid = qd_mex_reinterpret_Col<float>(prhs[4]),
        arrayant_single.elevation_grid = qd_mex_reinterpret_Col<float>(prhs[5]),
        azimuth_single = qd_mex_reinterpret_Mat<float>(prhs[6]),
        elevation_single = qd_mex_reinterpret_Mat<float>(prhs[7]);
    else
        arrayant_double.e_theta_re = qd_mex_reinterpret_Cube<double>(prhs[0]),
        arrayant_double.e_theta_im = qd_mex_reinterpret_Cube<double>(prhs[1]),
        arrayant_double.e_phi_re = qd_mex_reinterpret_Cube<double>(prhs[2]),
        arrayant_double.e_phi_im = qd_mex_reinterpret_Cube<double>(prhs[3]),
        arrayant_double.azimuth_grid = qd_mex_reinterpret_Col<double>(prhs[4]),
        arrayant_double.elevation_grid = qd_mex_reinterpret_Col<double>(prhs[5]),
        azimuth_double = qd_mex_reinterpret_Mat<double>(prhs[6]),
        elevation_double = qd_mex_reinterpret_Mat<double>(prhs[7]);

    // Validate the data integrity
    std::string error_message = use_single ? arrayant_single.validate() : arrayant_double.validate();
    if (!error_message.empty())
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:import_error", error_message.c_str());

    unsigned long long n_elements = use_single ? arrayant_single.n_elements() : arrayant_double.n_elements();
    unsigned long long n_out = (unsigned)mxGetM(prhs[6]); // Number of rows in "azimuth"
    unsigned long long n_ang = (unsigned)mxGetN(prhs[6]); // Number of angles for interpolation

    if ((unsigned)mxGetM(prhs[7]) != n_out || (unsigned)mxGetN(prhs[7]) != n_ang)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Number of elements in 'elevation' does not match number of elements in 'azimuth'.");

    // Process optional input : i_element
    arma::Col<unsigned> i_element;
    if (nrhs < 9 || mxGetNumberOfElements(prhs[8]) == 0)
        i_element = arma::regspace<arma::Col<unsigned>>(1, unsigned(n_elements));
    else
        i_element = qd_mex_typecast_Col<unsigned>(prhs[8], "i_element");

    if (n_out != 1 && n_out != i_element.n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Number of rows in 'azimuth' and 'elevation' must be 1 or match the number of elements.");

    // Check if values are valid (using lambda function)
    for (unsigned *val = i_element.begin(); val < i_element.end(); val++)
        if (*val < 1 || *val > n_elements)
            mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:out_of_bound", "Input 'i_element' must have values between 1 and 'n_elements'.");
    n_out = i_element.n_elem;

    // Process optional input : orientation
    arma::fcube orientation_single;
    arma::cube orientation_double;
    if (nrhs < 10 || mxGetNumberOfElements(prhs[9]) == 0)
        if (use_single)
            orientation_single = arma::fcube(3, 1, 1, arma::fill::zeros);
        else
            orientation_double = arma::cube(3, 1, 1, arma::fill::zeros);
    else if (use_single && mxIsSingle(prhs[9]))
        orientation_single = qd_mex_reinterpret_Cube<float>(prhs[9]);
    else if (!use_single && mxIsDouble(prhs[9]))
        orientation_double = qd_mex_reinterpret_Cube<double>(prhs[9]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:wrong_type", "Input 'orientation' must be given in double or single precision.");

    unsigned long long o1 = use_single ? orientation_single.n_rows : orientation_double.n_rows;
    unsigned long long o2 = use_single ? orientation_single.n_cols : orientation_double.n_cols;
    unsigned long long o3 = use_single ? orientation_single.n_slices : orientation_double.n_slices;

    if (o1 != 3ULL)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 3 elements on the first dimension.");
    else if (o2 != 1ULL && o2 != n_out)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 1 or 'n_elements' elements on the second dimension.");
    else if (o3 != 1ULL && o3 != n_ang)
        mexErrMsgIdAndTxt("quadriga_lib:arrayant_interpolate:size_mismatch", "Input 'orientation' must have 1 or 'n_ang' elements on the third dimension.");

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

    // Optional output "gamma"
    arma::fmat gamma_single;
    arma::mat gamma_double;
    if (nlhs < 8 && use_single)
        gamma_single = arma::fmat(0, 0);
    else if (nlhs < 8 && !use_single)
        gamma_double = arma::mat(0, 0);
    else if (use_single)
        plhs[7] = mxCreateNumericMatrix(n_out, n_ang, mxSINGLE_CLASS, mxREAL),
        gamma_single = arma::fmat((float *)mxGetData(plhs[7]), n_out, n_ang, false, true);
    else
        plhs[7] = mxCreateNumericMatrix(n_out, n_ang, mxDOUBLE_CLASS, mxREAL),
        gamma_double = arma::mat((double *)mxGetData(plhs[7]), n_out, n_ang, false, true);

    // Call private library function
    if (use_single)
        qd_arrayant_interpolate(&arrayant_single.e_theta_re, &arrayant_single.e_theta_im,
                                &arrayant_single.e_phi_re, &arrayant_single.e_phi_im,
                                &arrayant_single.azimuth_grid, &arrayant_single.elevation_grid,
                                &azimuth_single, &elevation_single,
                                &i_element, &orientation_single, &element_pos_single,
                                &V_re_single, &V_im_single, &H_re_single, &H_im_single, &dist_single,
                                &azimuth_loc_single, &elevation_loc_single, &gamma_single);
    else
        qd_arrayant_interpolate(&arrayant_double.e_theta_re, &arrayant_double.e_theta_im,
                                &arrayant_double.e_phi_re, &arrayant_double.e_phi_im,
                                &arrayant_double.azimuth_grid, &arrayant_double.elevation_grid,
                                &azimuth_double, &elevation_double,
                                &i_element, &orientation_double, &element_pos_double,
                                &V_re_double, &V_im_double, &H_re_double, &H_im_double, &dist_double,
                                &azimuth_loc_double, &elevation_loc_double, &gamma_double);
}
