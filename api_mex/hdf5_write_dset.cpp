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
# HDF5_WRITE_DSET
Write an unstructured dataset to a slot of an HDF5 channel file

- Adds the dataset `prefix + name` to slot `(ix, iy, iz, iw)`; everything else stored in the slot stays. An empty
  slot gets a channel without structured data
- A missing file is created with the layout `[ix, iy, iz, iw]`; [[hdf5_create_file]] sets another layout
- A dataset is written once: a dataset of the same name in the slot raises an error. A later [[hdf5_write_channel]] to
  the slot removes it with the rest of the slot
- Invalid names and types raise an error before the file is changed
- [[hdf5_read_channel]] returns the datasets with the prefix `par_` as the unstructured data `par`; [[hdf5_read_dset]]
  reads each one back

## Usage:
```
quadriga_lib.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, prefix );
```

## Inputs:
- **`fn`** — File name
- **`name`** — Dataset name without the prefix, not empty; `prefix + name` holds only letters, digits and `_`, at
  most 65000 characters
- **`data`** — Not empty: a char array (string), or a real, non-sparse single, double, int32, uint32, int64 or uint64
  array with at most 3 dimensions. A scalar is stored as a scalar, a column vector as a vector, a row vector as a
  `[1, n]` matrix
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 1-based; inside the layout of an
  existing file; default: 1
- **`prefix`** — Prefix of the dataset name; `''` stores `name` without a prefix; default: `'par_'`

## See also:
- [[hdf5_read_dset]] (reads a dataset by name)
- [[hdf5_read_dset_names]] (lists the datasets of a slot)
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
