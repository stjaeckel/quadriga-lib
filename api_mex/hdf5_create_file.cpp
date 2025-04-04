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
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_CREATE_FILE
Create a new HDF5 channel file with a custom storage layout

## Description:
Quadriga-Lib offers an HDF5-based method for storing and managing channel data. A key feature of this
library is its ability to organize multiple channels within a single HDF5 file while enabling access
to individual data sets without the need to read the entire file. In this system, channels can be
structured in a multi-dimensional array. For instance, the first dimension might represent the Base
Station (BS), the second the User Equipment (UE), and the third the frequency. However, it is important
to note that the dimensions of the storage layout must be defined when the file is initially created
and cannot be altered thereafter. The function `quadriga_lib.hdf5_create_file` is used to create an
empty file with a predetermined custom storage layout.

## Usage:

```
quadriga_lib.hdf5_create_file( fn, storage_dims );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`storage_dims`** (optional)<br>
  Size of the dimensions of the storage space, vector with 1-4 elements, i.e. `[nx]`, `[nx, ny]`, 
  `[nx,ny,nz]` or `[nx,ny,nz,nw]`. By default, `nx = 65536, ny = 1, nz = 1, nw = 1`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the HDF5 file
    //  1 - storage_dims    Dimensions of the storage space, vector with 1-4 elements, i.e. [nx], [nx, ny], [nx,ny,nz] or [nx,ny,nz,nw]

    // Output:
    //  0 - storage_dims    Dimensions of the storage space in the file, 4-element vector

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:no_output", "Incorrect number of output arguments.");

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:no_input", "Filename is missing.");

    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:wrong_type", "Input 'fn' must be a string");

    // Read file name
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:IO_error", "Input 'fn' must be a string.");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read storage_dims
    arma::Col<unsigned> storage_dims;
    if (nrhs > 1)
        storage_dims = qd_mex_typecast_Col<unsigned>(prhs[1], "storage_dims");

    unsigned ix = storage_dims.empty() ? 65536U : storage_dims.at(0);
    unsigned iy = storage_dims.n_elem > 1ULL ? storage_dims.at(1) : 1U;
    unsigned iz = storage_dims.n_elem > 2ULL ? storage_dims.at(2) : 1U;
    unsigned iw = storage_dims.n_elem > 3ULL ? storage_dims.at(3) : 1U;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:wrong_type", "Input 'storage_dims' cannot contain zeros (1-based indexing)");

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    arma::Col<unsigned> storage_space(4);
    try
    {
        storage_space = quadriga_lib::hdf5_read_layout(fn);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (storage_space.at(0) != 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:files_exists", "File already exists.");

    // Create file
    try
    {
        quadriga_lib::hdf5_create(fn, ix, iy, iz, iw);
        storage_space.at(0) = ix;
        storage_space.at(1) = iy;
        storage_space.at(2) = iz;
        storage_space.at(3) = iw;
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:create_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_create_file:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}
