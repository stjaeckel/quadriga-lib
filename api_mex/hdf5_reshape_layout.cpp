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
# HDF5_RESHAPE_LAYOUT
Reshape the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout; otherwise an error is thrown
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a valid HDF5 file

## Usage:
```
storage_space = quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`storage_dims`** — New storage layout; vector with 1-4 elements,
  i.e. `[nx]`, `[nx, ny]`, `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

## Outputs:
- **`storage_space`** — New storage dimensions `[nx, ny, nz, nw]`; `[4]`; uint32

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_read_layout]] (for reading the exisiting layout)
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

    if (nx == 0|| ny == 0|| nz == 0|| nw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'storage_dims' cannot contain zeros.");

    // Reshape layout
    CALL_QD(quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw));

    // Return new storage space
    if (nlhs > 0)
    {
        arma::u32_vec storage_space = {nx, ny, nz, nw};
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
    }
}
