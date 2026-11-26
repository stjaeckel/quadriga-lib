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
# HDF5_READ_DSET_NAMES
Read the names of the unstructured datasets of a slot of an HDF5 channel file

- Lists the datasets of slot `(ix, iy, iz, iw)` whose names start with `prefix`, without the prefix, in the order
  they were written

## Usage:
```
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — File name of an existing channel file
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; default: 1
- **`prefix`** — Prefix of the listed names; `''` lists all datasets with their full names; default: `'par_'`

## Output:
- **`names`** — Dataset names; cell array of strings; `[n_names, 1]`, empty if the slot is empty or outside the
  layout or has none

## See also:
- [[hdf5_read_dset]] (reads a dataset by name)
- [[hdf5_write_dset]] (writes a dataset)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const unsigned ix = (nrhs < 2) ? 0U : qd_mex_get_index<unsigned>(prhs[1], "ix"); // 1-based -> 0-based
    const unsigned iy = (nrhs < 3) ? 0U : qd_mex_get_index<unsigned>(prhs[2], "iy"); // 1-based -> 0-based
    const unsigned iz = (nrhs < 4) ? 0U : qd_mex_get_index<unsigned>(prhs[3], "iz"); // 1-based -> 0-based
    const unsigned iw = (nrhs < 5) ? 0U : qd_mex_get_index<unsigned>(prhs[4], "iw"); // 1-based -> 0-based
    const std::string prefix = (nrhs < 6) ? std::string("par_") : qd_mex_get_string_or_empty(prhs[5], "par_", "prefix");

    std::vector<std::string> names;
    CALL_QD(names = quadriga_lib::hdf5_read_dset_names(fn, ix, iy, iz, iw, prefix));

    plhs[0] = qd_mex_copy2matlab(&names);
}
