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
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (files
  written by quadriga-lib 0.12.4 and earlier are rejected; files modified with other HDF5 tools are
  not supported and may be rejected)
- One writer per file at a time; files are not locked

## Usage:
```
storage_space = quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`nx`** — New number of slots along dimension X
- **`ny`** — New number of slots along dimension Y; default: 1
- **`nz`** — New number of slots along dimension Z; default: 1
- **`nw`** — New number of slots along dimension W; default: 1

## Outputs:
- **`storage_space`** — New storage dimensions `[nx, ny, nz, nw]`; `[4]`; uint32

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_read_layout]] (for reading the existing layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const unsigned nx = qd_mex_get_scalar<unsigned>(prhs[1], "nx", 0U);
    const unsigned ny = (nrhs < 3) ? 1U : qd_mex_get_scalar<unsigned>(prhs[2], "ny", 1U);
    const unsigned nz = (nrhs < 4) ? 1U : qd_mex_get_scalar<unsigned>(prhs[3], "nz", 1U);
    const unsigned nw = (nrhs < 5) ? 1U : qd_mex_get_scalar<unsigned>(prhs[4], "nw", 1U);

    // Reshape layout
    CALL_QD(quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw));

    // Return new storage space
    arma::u32_vec storage_space = {nx, ny, nz, nw};
    plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}
