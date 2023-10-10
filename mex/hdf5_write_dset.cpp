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
    //  0 - fn              Filename of the HDF5 file
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]
    //  2 - name            Names of the dataset, string
    //  3 - data            Dataset to be written

    // Output:
    //  0 - storage_dims    Dimensions of the storage layout in the file, 4-element vector

    // Notes:
    // - Throws an error if dataset already exists at this location
    // - Throws an error if file does not exist (use hdf5_create_file)
    // - Supported types: string, double, float, (u)int32, (u)int64
    // - Supported size: up to 3 dimensions
    // - Storage order is maintained

    // Number of in and outputs
    if (nrhs < 4)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:no_input", "Need at least 4 inputs (file name, location, dataset name and data).");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:no_output", "Incorrect number of output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:wrong_type", "Input 'fn' must be a string");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read location (1-based)
    arma::Col<unsigned> location = qd_mex_typecast_Col<unsigned>(prhs[1], "location");

    unsigned ix = location.empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1ULL ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2ULL ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3ULL ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:wrong_type", "Input 'location' cannot contain zeros (1-based indexing)");
    --ix, --iy, --iz, --iw;

    // Read dataset name
    if (!mxIsClass(prhs[2], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:wrong_type", "Input 'name' must be a string");

    auto mx_dn = mxArrayToString(prhs[2]);
    std::string name = std::string(mx_dn);
    mxFree(mx_dn);

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    arma::Col<unsigned> storage_space;
    try
    {
        storage_space = quadriga_lib::hdf5_read_layout(fn);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (storage_space.at(0) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:no_output", "File does not exist.");

    // Throw error if location exceeds storage space
    if (ix > storage_space.at(0) || iy > storage_space.at(1) || iz > storage_space.at(2) || iw > storage_space.at(3))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:wrong_type", "Location exceeds storage space in HDF file.");

    // Construct data
    std::any data;
    if (mxGetNumberOfElements(prhs[3]) != 0)
    {
        if (mxIsClass(prhs[3], "char"))
        {
            auto chr = mxArrayToString(prhs[3]);
            data = std::any(std::string(chr));
            mxFree(chr);
        }
        else
            data = qd_mex_anycast(prhs[3], "data", false);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:wrong_type", "Data cannot be empty.");

    // Write to file
    try
    {
        quadriga_lib::hdf5_write_dset(fn, name, &data, ix, iy, iz, iw);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_dset:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}