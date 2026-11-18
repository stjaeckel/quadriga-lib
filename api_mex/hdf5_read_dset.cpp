// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_DSET
Read a single unstructured dataset from an HDF5 file

- Reads a user-defined dataset stored under `prefix + name` (e.g. `"par_carrier_frequency"`)
- Type and shape of the returned data are determined by the dataset's HDF5 dataspace
- Returns an empty `[]` matrix if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [[hdf5_read_channel]])
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
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; default: `'par_'`

## Outputs:
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

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const std::string name = qd_mex_get_string(prhs[1], "", "name");
    unsigned ix = (nrhs < 3) ? 1U : qd_mex_get_scalar<unsigned>(prhs[2], "ix", 1U);
    unsigned iy = (nrhs < 4) ? 1U : qd_mex_get_scalar<unsigned>(prhs[3], "iy", 1U);
    unsigned iz = (nrhs < 5) ? 1U : qd_mex_get_scalar<unsigned>(prhs[4], "iz", 1U);
    unsigned iw = (nrhs < 6) ? 1U : qd_mex_get_scalar<unsigned>(prhs[5], "iw", 1U);
    if (ix == 0U || iy == 0U || iz == 0U || iw == 0U)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'ix', 'iy', 'iz' and 'iw' cannot be 0 (1-based index).");
    const std::string prefix = (nrhs < 7) ? std::string("par_") : qd_mex_get_string(prhs[6], "par_", "prefix");

    // Convert to 0-based indices for the C++ call
    --ix, --iy, --iz, --iw;

    // Read dataset
    std::any dset;
    CALL_QD(dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw, prefix));

    plhs[0] = qd_mex_any2matlab(dset);
}