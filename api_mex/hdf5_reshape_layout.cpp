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
# HDF5_RESHAPE_LAYOUT
Reshapes the storage layout inside an existing HDF5 file

## Description:
Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A 
notable feature of this library is its capacity to manage multiple channels within a single HDF5 
file. In this framework, channels can be arranged in a multi-dimensional array format.
Once an HDF5 file has been created, the number of channels in the storage layout is fixed. 
However, it is possible to reshape the layout using `quadriga_lib.hdf5_reshape_layout`. 

## Usage:

```
quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`storage_dims`**<br>
  Size of the dimensions of the storage space, vector with 1-4 elements, i.e. `[nx]`, `[nx, ny]`, 
  `[nx,ny,nz]` or `[nx,ny,nz,nw]`. By default, `nx = 65536, ny = 1, nz = 1, nw = 1`<br>
  An error is thrown if the number of elements in the file is different from the given size.
MD!*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file
    //  1 - storage_dims    Dimensions of the storage layout, vector with 1-4 elements, i.e. [nx], [nx, ny], [nx,ny,nz] or [nx,ny,nz,nw]

    // Output:
    //  0 - storage_dims    Dimensions of the storage space in the file, 4-element vector

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:no_output", "Incorrect number of output arguments.");

    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:no_input", "Filename is missing.");

    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:wrong_type", "Input 'fn' must be a string");

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
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:wrong_type", "Input 'storage_dims' cannot contain zeros (1-based indexing)");

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    arma::Col<unsigned> storage_space;
    try
    {
        storage_space = quadriga_lib::hdf5_read_layout(fn);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Reshape storage dimensions
    try
    {
        quadriga_lib::hdf5_reshape_layout(fn, ix, iy, iz, iw);
        storage_space.at(0) = ix;
        storage_space.at(1) = iy;
        storage_space.at(2) = iz;
        storage_space.at(3) = iw;
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:create_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_reshape_layout:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}
