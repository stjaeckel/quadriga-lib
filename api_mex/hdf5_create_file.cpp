// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_CREATE_FILE
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists; delete it first to recreate it

## Usage:
```
storage_space = quadriga_lib.hdf5_create_file( fn, storage_dims );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`storage_dims`** — Size of the storage layout; vector with 1-4 elements, i.e. `[nx]`, `[nx, ny]`, 
  `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

## Output:
- **`storage_space`** — Actual storage dimensions used; `[4]`; uint32

## See also:
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_write_dset]] (for writing arbitrary unstructured data)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::u32_vec storage_dims = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1]);

    unsigned nx = storage_dims.is_empty() ? 65536 : storage_dims.at(0);
    unsigned ny = storage_dims.n_elem > 1 ? storage_dims.at(1) : 1;
    unsigned nz = storage_dims.n_elem > 2 ? storage_dims.at(2) : 1;
    unsigned nw = storage_dims.n_elem > 3 ? storage_dims.at(3) : 1;

    if (nx == 0 || ny == 0 || nz == 0 || nw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'storage_dims' cannot contain zeros.");

    // Fail if file already exists - hdf5_read_layout returns [0,0,0,0] if the file is missing
    arma::u32_vec existing;
    CALL_QD(existing = quadriga_lib::hdf5_read_layout(fn));
    if (existing.at(0) != 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File already exists.");

    // Create file
    CALL_QD(quadriga_lib::hdf5_create(fn, nx, ny, nz, nw));

    // Return storage space
    if (nlhs > 0)
    {
        arma::u32_vec storage_space = {nx, ny, nz, nw};
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
    }
}
