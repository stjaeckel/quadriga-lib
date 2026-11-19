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
# HDF5_WRITE_DSET
Write a single unstructured dataset to an HDF5 file

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must not be empty and must contain only alphanumeric characters and underscores
- Creates the file with the layout `[ix, iy, iz, iw]` if it does not exist; use [[hdf5_create_file]] for another
  layout
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [[hdf5_write_channel]] to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Arrays with more than 3 dimensions, sparse and complex arrays are rejected
- Row vectors are stored as given (`[1, n]`)
- One writer per file at a time; files are not locked

## Usage:
```
quadriga_lib.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; `''` stores `name` without a prefix; default (omitted or `[]`): `'par_'`

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs; // No outputs
    if (nrhs < 3 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const std::string name = qd_mex_get_string(prhs[1], "", "name");
    const unsigned ix = (nrhs < 4) ? 0U : qd_mex_get_index<unsigned>(prhs[3], "ix"); // 1-based -> 0-based
    const unsigned iy = (nrhs < 5) ? 0U : qd_mex_get_index<unsigned>(prhs[4], "iy"); // 1-based -> 0-based
    const unsigned iz = (nrhs < 6) ? 0U : qd_mex_get_index<unsigned>(prhs[5], "iz"); // 1-based -> 0-based
    const unsigned iw = (nrhs < 7) ? 0U : qd_mex_get_index<unsigned>(prhs[6], "iw"); // 1-based -> 0-based
    const std::string prefix = (nrhs < 8) ? std::string("par_") : qd_mex_get_string_or_empty(prhs[7], "par_", "prefix");

    if (mxGetNumberOfElements(prhs[2]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'data' cannot be empty.");

    std::any data;
    if (mxIsClass(prhs[2], "char"))
        data = std::any(qd_mex_get_string(prhs[2], "", "data"));
    else
        data = qd_mex_anycast(prhs[2], "data");

    // C++ creates a missing file
    CALL_QD(quadriga_lib::hdf5_write_dset(fn, name, data, ix, iy, iz, iw, prefix));
}
