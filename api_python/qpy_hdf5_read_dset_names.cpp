// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_read_dset_names
Read the names of unstructured datasets from an HDF5 file

- Lists all unstructured datasets stored at the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- Datasets are identified by the `par_` prefix; the returned names have that prefix stripped
- Returns an empty list if no unstructured datasets are present at the slot

## Usage:
```
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0

## Outputs:
- **`names`** — Names of all unstructured datasets at the slot, with the `par_` prefix stripped; list of str

## See also:
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

py::list hdf5_read_dset_names(const std::string &fn,
                              unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    std::vector<std::string> par_names;
    quadriga_lib::hdf5_read_dset_names(fn, &par_names, ix, iy, iz, iw);
    return qd_python_copy2list(par_names);
}