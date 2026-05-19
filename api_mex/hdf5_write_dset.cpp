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
# HDF5_WRITE_DSET
Write a single unstructured dataset to an HDF5 file

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must contain only alphanumeric characters and underscores
- The file must already exist (use [[hdf5_create_file]] first)
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Row vectors are stored as column vectors

## Usage:
```
storage_dims = quadriga_lib.hdf5_write_dset( fn, location, name, data, prefix );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; default: `'par_'`

## Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::Col<unsigned> location = qd_mex_get_Col<unsigned>(prhs[1]);
    const std::string name = qd_mex_get_string(prhs[2]);
    const std::string prefix = (nrhs < 5) ? std::string("par_") : qd_mex_get_string(prhs[4], "par_");

    unsigned ix = location.is_empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1 ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2 ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3 ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'location' cannot contain zeros (1-based indexing).");

    // Convert to 0-based indices for the C++ call
    --ix, --iy, --iz, --iw;

    // Validate data is non-empty
    if (mxGetNumberOfElements(prhs[3]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'data' cannot be empty.");

    // Wrap data into std::any (strings need special handling)
    std::any data;
    if (mxIsClass(prhs[3], "char"))
        data = std::any(qd_mex_get_string(prhs[3]));
    else
        data = qd_mex_anycast(prhs[3], "data", false);

    // Read layout for "file exists" check and to populate storage_dims output
    arma::Col<unsigned> storage_space;
    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn));
    if (storage_space.at(0) == 0U)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File does not exist (use quadriga_lib.hdf5_create_file).");

    // Write dataset
    CALL_QD(quadriga_lib::hdf5_write_dset(fn, name, &data, ix, iy, iz, iw, prefix));

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}
