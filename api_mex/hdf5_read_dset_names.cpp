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
# HDF5_READ_DSET_NAMES
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`
- Returned names have the prefix stripped
- Returns an empty cell array if no matching datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [[hdf5_read_channel]])

## Usage:
```
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix used to identify unstructured datasets; string; default: `'par_'`

## Outputs:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings

## See also:
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const unsigned ix = (nrhs < 2) ? 0U : qd_mex_get_index<unsigned>(prhs[1], "ix"); // 1-based -> 0-based
    const unsigned iy = (nrhs < 3) ? 0U : qd_mex_get_index<unsigned>(prhs[2], "iy"); // 1-based -> 0-based
    const unsigned iz = (nrhs < 4) ? 0U : qd_mex_get_index<unsigned>(prhs[3], "iz"); // 1-based -> 0-based
    const unsigned iw = (nrhs < 5) ? 0U : qd_mex_get_index<unsigned>(prhs[4], "iw"); // 1-based -> 0-based
    const std::string prefix = (nrhs < 6) ? std::string("par_") : qd_mex_get_string(prhs[5], "par_", "prefix");

    // Read dataset names
    std::vector<std::string> names;
    CALL_QD(quadriga_lib::hdf5_read_dset_names(fn, &names, ix, iy, iz, iw, prefix));

    // Return cell array
    plhs[0] = qd_mex_copy2matlab(&names);
}
