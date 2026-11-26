// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# HDF5_READ_DSET
Read an unstructured dataset from a slot of an HDF5 channel file

- Reads the dataset `prefix + name` of slot `(ix, iy, iz, iw)`, e.g. `par_carrier_frequency`
- The class and shape follow the stored dataset: char for a string; single, double, int32, uint32, int64 or uint64
  for a scalar, a column vector `[n, 1]` (a 1D dataset), a matrix or a 3D array; a row vector written by
  [[hdf5_write_dset]] comes back as `[1, n]`

## Usage:
```
dset = quadriga_lib.hdf5_read_dset( fn, name, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — File name of an existing channel file
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; default: 1
- **`prefix`** — Prefix of the dataset name; `''` reads `name` as given; default: `'par_'`

## Output:
- **`dset`** — The dataset; `[]` if the slot is empty or outside the layout or has no such dataset

## See also:
- [[hdf5_read_dset_names]] (lists the datasets of a slot)
- [[hdf5_write_dset]] (writes a dataset)
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
