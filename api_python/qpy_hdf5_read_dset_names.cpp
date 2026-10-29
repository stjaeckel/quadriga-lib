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
- Lists the datasets whose names start with `prefix` (default `par_`; `''` lists all); the returned names have the
  prefix stripped
- Returns an empty list if no unstructured datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [[hdf5_read_channel]])

## Usage:
```
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix used to identify unstructured datasets; `''` lists all datasets; str; default: `'par_'`

## Outputs:
- **`names`** — Names of the unstructured datasets at the slot whose names start with `prefix`, with the prefix
  stripped, in creation order; list of str

## See also:
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

py::list hdf5_read_dset_names(const std::string &fn,
                              unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                              const std::string &prefix)
{
    std::vector<std::string> par_names;
    quadriga_lib::hdf5_read_dset_names(fn, &par_names, ix, iy, iz, iw, prefix);
    return qd_python_copy2list(par_names);
}