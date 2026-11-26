// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# HDF5_RESHAPE_LAYOUT
Reshape the storage layout of an existing HDF5 channel file

- Sets new dimensions `[nx, ny, nz, nw]` with the slot count `nx · ny · nz · nw` of the file
- The data stays in place: in the column-major order of the slots (as in `has_data` of [[hdf5_read_layout]]), slot
  `k` keeps its channel, which gets the slot index of position `k` in the new layout

## Usage:
```
quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

## Inputs:
- **`fn`** — File name of an existing channel file
- **`nx`** — New number of slots along x
- **`ny`**, **`nz`**, **`nw`** — New number of slots along y, z and w; default: 1

## See also:
- [[hdf5_create_file]] (sets the layout of a new file)
- [[hdf5_read_layout]] (reads the layout)
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
