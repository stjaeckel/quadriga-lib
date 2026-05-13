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
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

## Usage:
```
dset = quadriga_lib.hdf5_read_dset( fn, location, name, prefix );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`prefix`** *(optional)* — Prefix prepended to `name` when looking up the dataset; string; default: `'par_'`

## Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; empty `[]` if the dataset is missing

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::Col<unsigned> location = qd_mex_get_Col<unsigned>(prhs[1]);
    const std::string name = qd_mex_get_string(prhs[2]);
    const std::string prefix = (nrhs < 4) ? std::string("par_") : qd_mex_get_string(prhs[3], "par_");

    unsigned ix = location.is_empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1 ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2 ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3 ? location.at(3) : 1;

    if (ix == 0U || iy == 0U || iz == 0U || iw == 0U)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'location' cannot contain zeros (1-based indexing).");

    // Convert to 0-based indices for the C++ call
    --ix, --iy, --iz, --iw;

    // Read dataset
    std::any dset;
    CALL_QD(dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw, prefix));

    if (nlhs > 0)
        plhs[0] = qd_mex_any2matlab(dset);
}