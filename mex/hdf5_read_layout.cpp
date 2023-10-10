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
    //  0 - fn              Filename of the QDANT file

    // Output:
    //  0 - storage_dims    Dimensions of the storage layout in the file, 4-element vector
    //  1 - has_data        Array indicating if data was stored in that location

    // Read filename
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_layout:no_input", "Filename is missing.");

    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_layout:wrong_type", "Input 'fn' must be a string");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    arma::Col<unsigned> storage_space;
    arma::Col<unsigned> channelID;
    try
    {
        storage_space = quadriga_lib::hdf5_read_layout(fn, &channelID);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_layout:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_layout:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);

    // Determine if there is data
    if (nlhs > 1)
    {
        unsigned nx = storage_space.at(0),
                 ny = storage_space.at(1),
                 nz = storage_space.at(2),
                 nw = storage_space.at(3);

        unsigned n_data = nx * ny * nz * nw;
        if (channelID.n_elem != (uword)n_data)
            mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_layout:unknown_error", "Corrupted storage index.");

        mwSize dims[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)nw};
        plhs[1] = mxCreateNumericArray(4, dims, mxUINT32_CLASS, mxREAL);
        unsigned *ptrO = (unsigned *)mxGetData(plhs[1]);
        unsigned *ptrI = channelID.memptr();

        for (unsigned n = 0; n < n_data; n++)
            if (ptrI[n] != 0)
                ptrO[n] = 1;
    }
}
