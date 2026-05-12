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
# HDF5_READ_LAYOUT
Read the storage layout of channel data inside an HDF5 file

- Returns the dimensions of the 4D channel slot grid stored inside an HDF5 file
- Returns `[0, 0, 0, 0]` if the file does not exist; errors if the file exists but is not a valid HDF5 file
- Also reports which slots already contain data, so callers can locate free slots without scanning the file

## Usage:
```
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

## Input:
- **`fn`** — Filename of the HDF5 file; string

## Outputs:
- **`storage_dims`** *(optional)* — Size of the storage space `[nx, ny, nz, nw]`; `[4]`;  uint32
- **`has_data`** *(optional)* — Slot occupancy mask; `true` where data exists, `false` otherwise; 
  `[nx, ny, nz, nw]`; logical
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input
    const std::string fn = qd_mex_get_string(prhs[0]);

    // Read storage layout (and channelIDs when requested)
    arma::Col<unsigned> storage_space;
    arma::Col<unsigned> channelID;
    arma::Col<unsigned> *p_channelID = (nlhs > 1) ? &channelID : nullptr;

    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn, p_channelID));

    // Return storage space as 1x4 row vector
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);

    // Build binary occupancy mask as 4D uint32 array (no Col/Mat/Cube helper for 4D)
    if (nlhs > 1)
    {
        unsigned nx = storage_space.at(0),
                 ny = storage_space.at(1),
                 nz = storage_space.at(2),
                 nw = storage_space.at(3);

        unsigned n_data = nx * ny * nz * nw;
        if (channelID.n_elem != (arma::uword)n_data)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Corrupted storage index.");

        mwSize dims[4] = {(mwSize)nx, (mwSize)ny, (mwSize)nz, (mwSize)nw};
        plhs[1] = mxCreateLogicalArray(4, dims);
        mxLogical *ptrO = mxGetLogicals(plhs[1]);
        const unsigned *ptrI = channelID.memptr();

        for (unsigned n = 0; n < n_data; ++n)
            if (ptrI[n] != 0)
                ptrO[n] = 1;
    }
}
