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
# HDF5_CREATE_FILE
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

## Usage:
```
quadriga_lib.hdf5_create_file( fn, nx, ny, nz, nw );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`nx`** — Number of slots along dimension X; default: 65536
- **`ny`** — Number of slots along dimension Y; default: 1
- **`nz`** — Number of slots along dimension Z; default: 1
- **`nw`** — Number of slots along dimension W; default: 1

## See also:
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_write_dset]] (for writing arbitrary unstructured data)
- [[hdf5_read_layout]] (for reading the layout)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const unsigned nx = (nrhs < 2) ? 65536U : qd_mex_get_scalar<unsigned>(prhs[1], "nx", 65536U);
    const unsigned ny = (nrhs < 3) ? 1U : qd_mex_get_scalar<unsigned>(prhs[2], "ny", 1U);
    const unsigned nz = (nrhs < 4) ? 1U : qd_mex_get_scalar<unsigned>(prhs[3], "nz", 1U);
    const unsigned nw = (nrhs < 5) ? 1U : qd_mex_get_scalar<unsigned>(prhs[4], "nw", 1U);

    // C++ throws if the file exists
    CALL_QD(quadriga_lib::hdf5_create_file(fn, nx, ny, nz, nw));
}
