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

## Usage:
```
names = quadriga_lib.hdf5_read_dset_names( fn, location, prefix );
```

## Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`location`** *(optional)* — Slot location inside the file; 1-based; vector with 1-4
  elements, i.e. `[ix]`, `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; default:  `[1, 1, 1, 1]`
- **`prefix`** *(optional)* — Prefix used to identify unstructured datasets; string; default: `'par_'`

## Outputs:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings

## See also:
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::u32_vec location = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1]);
    const std::string prefix = (nrhs < 3) ? std::string("par_") : qd_mex_get_string(prhs[2], "par_");

    unsigned ix = location.is_empty() ? 1U : location.at(0);
    unsigned iy = location.n_elem > 1ULL ? location.at(1) : 1U;
    unsigned iz = location.n_elem > 2ULL ? location.at(2) : 1U;
    unsigned iw = location.n_elem > 3ULL ? location.at(3) : 1U;

    if (ix == 0U || iy == 0U || iz == 0U || iw == 0U)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'location' cannot contain zeros (1-based indexing).");

    // Convert to 0-based indices for the C++ call
    --ix, --iy, --iz, --iw;

    // Read dataset names
    std::vector<std::string> par_names;
    CALL_QD(quadriga_lib::hdf5_read_dset_names(fn, &par_names, ix, iy, iz, iw, prefix));

    // Return cell array
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&par_names);
}
