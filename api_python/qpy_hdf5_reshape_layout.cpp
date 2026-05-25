// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_reshape_layout
Reshapes the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a valid HDF5 file

## Usage:
```
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1
MD!*/

void hdf5_reshape_layout(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw);
}