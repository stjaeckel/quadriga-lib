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
# HDF5_READ_LAYOUT
Read the storage layout of an HDF5 channel file

- Returns the dimensions of the slot grid and, optionally, the slots that hold data
- A missing file gives `[0, 0, 0, 0]`, so the function also tests whether a file exists; a file that is not a channel
  file raises an error

## Usage:
```
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

## Input:
- **`fn`** — File name

## Outputs:
- **`storage_dims`** — Layout `[nx, ny, nz, nw]`; uint32; `[1, 4]`
- **`has_data`** — `true` for the slots that were written, also with an empty channel; logical; `[nx, ny, nz, nw]`, empty for a missing file

## See also:
- [[hdf5_create_file]] (creates a file with a given layout)
- [[hdf5_reshape_layout]] (changes the dimensions)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");

    arma::Col<unsigned> storage_space;
    arma::Col<unsigned> channel_id;
    arma::Col<unsigned> *p_channel_id = (nlhs > 1) ? &channel_id : nullptr;

    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn, p_channel_id));

    // Return storage space as 1x4 row vector
    plhs[0] = qd_mex_copy2matlab(&storage_space, true);

    // Build binary occupancy mask as 4D uint32 array (no Col/Mat/Cube helper for 4D)
    if (nlhs > 1)
    {
        unsigned nx = storage_space.at(0),
                 ny = storage_space.at(1),
                 nz = storage_space.at(2),
                 nw = storage_space.at(3);

        unsigned n_data = nx * ny * nz * nw;
        if (channel_id.n_elem != (arma::uword)n_data)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Corrupted storage index.");

        mwSize dims[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)nw};
        plhs[1] = mxCreateLogicalArray(4, dims);
        mxLogical *ptrO = mxGetLogicals(plhs[1]);
        const unsigned *ptrI = channel_id.memptr();

        for (unsigned n = 0; n < n_data; ++n)
            if (ptrI[n] != 0)
                ptrO[n] = 1;
    }
}
