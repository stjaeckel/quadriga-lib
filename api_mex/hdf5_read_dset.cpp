// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_DSET
Read a single unstructured dataset from an HDF5 file

- Reads a user-defined dataset stored under `prefix + name` (e.g. `"par_carrier_frequency"`)
- Type and shape of the returned data are determined by the dataset's HDF5 dataspace
- Returns an empty `[]` matrix if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

## Usage:
```
dset = quadriga_lib.hdf5_read_dset( fn, name, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; `''` looks up `name` as given; default (omitted or `[]`): `'par_'`

## Output:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; empty `[]` if the dataset is missing

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const std::string name = qd_mex_get_string(prhs[1], "", "name");
    const unsigned ix = (nrhs < 3) ? 0U : qd_mex_get_index<unsigned>(prhs[2], "ix"); // 1-based -> 0-based
    const unsigned iy = (nrhs < 4) ? 0U : qd_mex_get_index<unsigned>(prhs[3], "iy"); // 1-based -> 0-based
    const unsigned iz = (nrhs < 5) ? 0U : qd_mex_get_index<unsigned>(prhs[4], "iz"); // 1-based -> 0-based
    const unsigned iw = (nrhs < 6) ? 0U : qd_mex_get_index<unsigned>(prhs[5], "iw"); // 1-based -> 0-based
    const std::string prefix = (nrhs < 7) ? std::string("par_") : qd_mex_get_string_or_empty(prhs[6], "par_", "prefix");

    std::any dset;
    CALL_QD(dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw, prefix));

    plhs[0] = qd_mex_any2matlab(dset);
}
