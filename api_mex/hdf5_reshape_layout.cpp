// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_RESHAPE_LAYOUT
Reshape the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file
- One writer per file at a time; files are not locked

## Usage:
```
quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`nx`** — New number of slots along dimension X
- **`ny`** — New number of slots along dimension Y; default: 1
- **`nz`** — New number of slots along dimension Z; default: 1
- **`nw`** — New number of slots along dimension W; default: 1

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_read_layout]] (for reading the existing layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs
    if (nrhs < 2 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const unsigned nx = qd_mex_get_scalar<unsigned>(prhs[1], "nx", 0U);
    const unsigned ny = (nrhs < 3) ? 1U : qd_mex_get_scalar<unsigned>(prhs[2], "ny", 1U);
    const unsigned nz = (nrhs < 4) ? 1U : qd_mex_get_scalar<unsigned>(prhs[3], "nz", 1U);
    const unsigned nw = (nrhs < 5) ? 1U : qd_mex_get_scalar<unsigned>(prhs[4], "nw", 1U);

    CALL_QD(quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw));
}
