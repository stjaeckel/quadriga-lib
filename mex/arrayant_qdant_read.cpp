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
#include <cstring> // For memcopy

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file
    //  1 - id              ID if the antenna to be read from the file (optional, default: 1)
    //  2 - use_single      Indicator if results should be returned in single precision (optional, default: 0, double)

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
    // 11 - layout          Layout of multiple array antennas (optional), uint32            Matrix

    // Number of in and outputs
    if (nlhs < 11 || nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:no_input", "Wrong number of input/output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:wrong_type", "Input 'fn' must be a string");

    std::string fn = mxArrayToString(prhs[0]);

    // Read id variable
    unsigned id;
    if (nrhs < 2)
        id = 1;
    else if (mxGetNumberOfElements(prhs[1]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:size_mismatch", "Input 'id' must be scalar.");
    else if (mxIsDouble(prhs[1]))
    {
        double *tmp = (double *)mxGetData(prhs[1]);
        id = (unsigned)tmp[0];
    }
    else if (mxIsSingle(prhs[1]))
    {
        float *tmp = (float *)mxGetData(prhs[1]);
        id = (unsigned)tmp[0];
    }
    else if (mxIsClass(prhs[1], "uint32") || mxIsClass(prhs[1], "int32"))
    {
        unsigned *tmp = (unsigned *)mxGetData(prhs[1]);
        id = tmp[0];
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:wrong_type", "Input 'id' must be either of type 'double', 'single' ur 'uint32'.");

    // Read "use_single" variable
    bool use_single;
    if (nrhs < 3 || mxGetNumberOfElements(prhs[2]) == 0)
        use_single = false;
    else if (mxGetNumberOfElements(prhs[2]) != 1)
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:size_mismatch", "Input 'use_single' must be scalar.");
    else if (mxIsDouble(prhs[2]))
    {
        double *tmp = (double *)mxGetData(prhs[2]);
        use_single = (bool)tmp[0];
    }
    else if (mxIsClass(prhs[2], "logical"))
    {
        bool *tmp = (bool *)mxGetData(prhs[2]);
        use_single = tmp[0];
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:qdant_read:wrong_type", "Input 'use_single' must be either of type 'double' or 'logical'.");

    quadriga_lib::arrayant<float> arrayant_single;
    quadriga_lib::arrayant<double> arrayant_double;
    arma::Mat<unsigned> layout;
    if (use_single)
        arrayant_single = quadriga_lib::arrayant<float>(fn, id, &layout);
    else
        arrayant_double = quadriga_lib::arrayant<double>(fn, id, &layout);

    unsigned n_azimuth = use_single ? arrayant_single.n_azimuth() : arrayant_double.n_azimuth();
    unsigned n_elevation = use_single ? arrayant_single.n_elevation() : arrayant_double.n_elevation();
    unsigned n_elements = use_single ? arrayant_single.n_elements() : arrayant_double.n_elements();
    unsigned n_ports = use_single ? arrayant_single.n_ports() : arrayant_double.n_ports();

    mwSize dims[3] = {n_elevation, n_azimuth, n_elements};

    if (use_single)
        plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL),
        plhs[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL),
        plhs[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL),
        plhs[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL),
        plhs[4] = mxCreateNumericMatrix(1, n_azimuth, mxSINGLE_CLASS, mxREAL),
        plhs[5] = mxCreateNumericMatrix(1, n_elevation, mxSINGLE_CLASS, mxREAL),
        plhs[6] = mxCreateNumericMatrix(3, n_elements, mxSINGLE_CLASS, mxREAL),
        plhs[7] = mxCreateNumericMatrix(n_elements, n_ports, mxSINGLE_CLASS, mxREAL),
        plhs[8] = mxCreateNumericMatrix(n_elements, n_ports, mxSINGLE_CLASS, mxREAL),
        plhs[9] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    else
        plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL),
        plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL),
        plhs[2] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL),
        plhs[3] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL),
        plhs[4] = mxCreateNumericMatrix(1, n_azimuth, mxDOUBLE_CLASS, mxREAL),
        plhs[5] = mxCreateNumericMatrix(1, n_elevation, mxDOUBLE_CLASS, mxREAL),
        plhs[6] = mxCreateNumericMatrix(3, n_elements, mxDOUBLE_CLASS, mxREAL),
        plhs[7] = mxCreateNumericMatrix(n_elements, n_ports, mxDOUBLE_CLASS, mxREAL),
        plhs[8] = mxCreateNumericMatrix(n_elements, n_ports, mxDOUBLE_CLASS, mxREAL),
        plhs[9] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);

    if (use_single)
    {
        std::memcpy((float *)mxGetData(plhs[0]), arrayant_single.e_theta_re.memptr(), sizeof(float) * arrayant_single.e_theta_re.n_elem);
        std::memcpy((float *)mxGetData(plhs[1]), arrayant_single.e_theta_im.memptr(), sizeof(float) * arrayant_single.e_theta_im.n_elem);
        std::memcpy((float *)mxGetData(plhs[2]), arrayant_single.e_phi_re.memptr(), sizeof(float) * arrayant_single.e_phi_re.n_elem);
        std::memcpy((float *)mxGetData(plhs[3]), arrayant_single.e_phi_im.memptr(), sizeof(float) * arrayant_single.e_phi_im.n_elem);
        std::memcpy((float *)mxGetData(plhs[4]), arrayant_single.azimuth_grid.memptr(), sizeof(float) * arrayant_single.azimuth_grid.n_elem);
        std::memcpy((float *)mxGetData(plhs[5]), arrayant_single.elevation_grid.memptr(), sizeof(float) * arrayant_single.elevation_grid.n_elem);
        std::memcpy((float *)mxGetData(plhs[6]), arrayant_single.element_pos.memptr(), sizeof(float) * arrayant_single.element_pos.n_elem);
        std::memcpy((float *)mxGetData(plhs[7]), arrayant_single.coupling_re.memptr(), sizeof(float) * arrayant_single.coupling_re.n_elem);
        std::memcpy((float *)mxGetData(plhs[8]), arrayant_single.coupling_im.memptr(), sizeof(float) * arrayant_single.coupling_im.n_elem);
        std::memcpy((float *)mxGetData(plhs[9]), &arrayant_single.center_frequency, sizeof(float));
    }
    else
    {
        std::memcpy((double *)mxGetData(plhs[0]), arrayant_double.e_theta_re.memptr(), sizeof(double) * arrayant_double.e_theta_re.n_elem);
        std::memcpy((double *)mxGetData(plhs[1]), arrayant_double.e_theta_im.memptr(), sizeof(double) * arrayant_double.e_theta_im.n_elem);
        std::memcpy((double *)mxGetData(plhs[2]), arrayant_double.e_phi_re.memptr(), sizeof(double) * arrayant_double.e_phi_re.n_elem);
        std::memcpy((double *)mxGetData(plhs[3]), arrayant_double.e_phi_im.memptr(), sizeof(double) * arrayant_double.e_phi_im.n_elem);
        std::memcpy((double *)mxGetData(plhs[4]), arrayant_double.azimuth_grid.memptr(), sizeof(double) * arrayant_double.azimuth_grid.n_elem);
        std::memcpy((double *)mxGetData(plhs[5]), arrayant_double.elevation_grid.memptr(), sizeof(double) * arrayant_double.elevation_grid.n_elem);
        std::memcpy((double *)mxGetData(plhs[6]), arrayant_double.element_pos.memptr(), sizeof(double) * arrayant_double.element_pos.n_elem);
        std::memcpy((double *)mxGetData(plhs[7]), arrayant_double.coupling_re.memptr(), sizeof(double) * arrayant_double.coupling_re.n_elem);
        std::memcpy((double *)mxGetData(plhs[8]), arrayant_double.coupling_im.memptr(), sizeof(double) * arrayant_double.coupling_im.n_elem);
        std::memcpy((double *)mxGetData(plhs[9]), &arrayant_double.center_frequency, sizeof(double));
    }
    plhs[10] = use_single ? mxCreateString(arrayant_single.name.c_str()) : mxCreateString(arrayant_double.name.c_str());

    if (nlhs == 12)
    {
        plhs[11] = mxCreateNumericMatrix(layout.n_rows, layout.n_cols, mxUINT32_CLASS, mxREAL);
        if (layout.n_elem != 0)
            std::memcpy((unsigned *)mxGetData(plhs[11]), layout.memptr(), sizeof(unsigned) * layout.n_elem);
    }
}
