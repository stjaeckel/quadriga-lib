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
# HDF5_READ_DSET
Read a single unstructured dataset from an HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition 
to structured datasets, the library facilitates the inclusion of extra datasets of various types 
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis 
results. The function `quadriga_lib.hdf5_read_dset` retrieves a single unstructured dataset. The
output type of the function is defined by the datatype in the file. An empty matrix is returned 
if the dataset does not exist in the file.

## Usage:

```
dset = quadriga_lib.hdf5_read_dset( fn, location, name );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`location`** (optional)<br>
  Storage location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`, 
  `[ix,iy,iz]` or `[ix,iy,iz,iw]`; Default: `ix = iy = iz = iw = 1`

- **`name`**<br>
  Name of the dataset; String

## Output Argument:
- **`dset`**<br>
  Output data. Type and size is defined by the dataspace in the file
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file, string
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]
    //  2 - name            Names of the dataset, string

    // Outputs:
    //  0 - data            Dataset

    // Notes:
    // - Returns empty double matrix if there is no data at this location

    // Number of in and outputs
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:no_input", "Need 3 inputs (file name, location and dataset name).");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:no_input", "Too many output arguments.");

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
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:wrong_type", "Input 'location' cannot contain zeros (1-based indexing)");
    --ix, --iy, --iz, --iw;

    // Read dataset name
    if (!mxIsClass(prhs[2], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:wrong_type", "Input 'name' must be a string");

    auto mx_dn = mxArrayToString(prhs[2]);
    std::string name = std::string(mx_dn);
    mxFree(mx_dn);

    // Read unstructured data field from HDF5 file
    std::any dset;
    try
    {
        dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_dset:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (nlhs > 0 && !dset.has_value())
        plhs[0] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
    else if (nlhs > 0)
    {
        unsigned long long dims[3];
        void *dataptr;
        int type_id = quadriga_lib::any_type_id(&dset, dims, &dataptr);

        if (type_id == 9) // Strings
        {
            auto data = std::any_cast<std::string>(dset);
            plhs[0] = mxCreateString(data.c_str());
        }

        // Scalars
        else if (type_id == 10)
            plhs[0] = qd_mex_copy2matlab((float *)dataptr);
        else if (type_id == 11)
            plhs[0] = qd_mex_copy2matlab((double *)dataptr);
        else if (type_id == 12)
            plhs[0] = qd_mex_copy2matlab((unsigned long long int *)dataptr);
        else if (type_id == 13)
            plhs[0] = qd_mex_copy2matlab((long long int *)dataptr);
        else if (type_id == 14)
            plhs[0] = qd_mex_copy2matlab((unsigned int *)dataptr);
        else if (type_id == 15)
            plhs[0] = qd_mex_copy2matlab((int *)dataptr);

        // Matrices
        else if (type_id == 20)
        {
            auto data = std::any_cast<arma::Mat<float>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 21)
        {
            auto data = std::any_cast<arma::Mat<double>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 22)
        {
            auto data = std::any_cast<arma::Mat<unsigned long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 23)
        {
            auto data = std::any_cast<arma::Mat<long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 24)
        {
            auto data = std::any_cast<arma::Mat<unsigned>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 25)
        {
            auto data = std::any_cast<arma::Mat<int>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }

        // Cubes
        else if (type_id == 30)
        {
            auto data = std::any_cast<arma::Cube<float>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 31)
        {
            auto data = std::any_cast<arma::Cube<double>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 32)
        {
            auto data = std::any_cast<arma::Cube<unsigned long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 33)
        {
            auto data = std::any_cast<arma::Cube<long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 34)
        {
            auto data = std::any_cast<arma::Cube<unsigned>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 35)
        {
            auto data = std::any_cast<arma::Cube<int>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }

        // Vectors (Columns only)
        else if (type_id == 40)
        {
            auto data = std::any_cast<arma::Col<float>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 41)
        {
            auto data = std::any_cast<arma::Col<double>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 42)
        {
            auto data = std::any_cast<arma::Col<unsigned long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 43)
        {
            auto data = std::any_cast<arma::Col<long long>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 44)
        {
            auto data = std::any_cast<arma::Col<unsigned>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
        else if (type_id == 45)
        {
            auto data = std::any_cast<arma::Col<int>>(dset);
            plhs[0] = qd_mex_copy2matlab(&data);
        }
    }
}