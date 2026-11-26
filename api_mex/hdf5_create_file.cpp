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
# HDF5_CREATE_FILE
Create a new HDF5 channel file with a given storage layout

- Creates a channel file with `nx · ny · nz · nw` empty slots; the slot count is fixed, [[hdf5_reshape_layout]] changes
  only the dimensions
- [[hdf5_write_channel]] and [[hdf5_write_dset]] create a missing file themselves, with the layout that just holds their
  slots; this function sets any other one
- Standard HDF5 readers open the file, see [HDF5 channel file format](formats.html#hdf5_channel); change it only with
  the functions of this section

## Usage:
```
quadriga_lib.hdf5_create_file( fn, nx, ny, nz, nw );
```

## Inputs:
- **`fn`** — File name; the file must not exist
- **`nx`** — Number of slots along x, at least 1; default: 65536
- **`ny`**, **`nz`**, **`nw`** — Number of slots along y, z and w, each at least 1; at most 4294967295 (2^32 - 1)
  slots in total; default: 1

## See also:
- [[hdf5_write_channel]] (writes channels into slots)
- [[hdf5_read_layout]] (reads the layout and the occupied slots)
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
