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
# HDF5_READ_DSET_NAMES
Read the names of unstructured data fields from an HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition 
to structured datasets, the library facilitates the inclusion of extra datasets of various types 
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis 
results. Users can add any number of such unstructured datasets, each identified by a unique 
dataset name. The function `quadriga_lib.hdf5_read_dset_names` retrieves the names of all these 
datasets, returning them as a cell array of strings.

## Usage:

```
names = quadriga_lib.hdf5_read_dset_names( fn, location );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`location`** (optional)<br>
  Storage location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`, 
  `[ix,iy,iz]` or `[ix,iy,iz,iw]`; Default: `ix = iy = iz = iw = 1`

## Output Argument:
- **`names`**<br>
  List of names of all these at the given location in the files; Cell array of strings
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]

    // Outputs:
    //  0 - names           Names of the unstructured datasets, Cell array of strings

    // Number of in and outputs
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:no_input", "Filename is missing.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:no_input", "Too many output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:wrong_type", "Input 'fn' must be a string");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read location (1-based)
    arma::Col<unsigned> location;
    if (nrhs > 1)
        location = qd_mex_typecast_Col<unsigned>(prhs[1], "location");

    unsigned ix = location.empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1ULL ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2ULL ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3ULL ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:wrong_type", "Input 'location' cannot contain zeros (1-based indexing)");
    --ix, --iy, --iz, --iw;

    // Read names of the unstructured data fields from file
    quadriga_lib::channel<float> channel;
    std::vector<std::string> par_names;
    unsigned long long n_par = 0ULL;
    try
    {
        n_par = quadriga_lib::hdf5_read_dset_names(fn, &par_names, ix, iy, iz, iw);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset_names:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (nlhs > 0 && n_par == 0ULL) // Nothing there
        plhs[0] = mxCreateCellMatrix(0, 0);
    else if (nlhs > 0)
    {
        // Create the cell array
         mxArray *cellArray = mxCreateCellMatrix((mwSize)n_par, 1);
         
        // Fill in the cell array
        for (size_t i = 0; i < par_names.size(); ++i)
        {
            mxArray *mxStr = mxCreateString(par_names[i].c_str());
            mxSetCell(cellArray, i, mxStr);
        }

        // Set the output
        plhs[0] = cellArray;
    }
}