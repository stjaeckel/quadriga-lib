// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- The total slot count is fixed at creation; the layout can be reshaped with [[hdf5_reshape_layout]]
- Raises an error if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

## Usage:
```
quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

## See also:
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_write_dset]] (for writing arbitrary unstructured data)
- [[hdf5_read_layout]] (for reading the layout)
MD!*/

void hdf5_create_file(const py::handle &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    // C++ throws if the file exists
    quadriga_lib::hdf5_create_file(qd_python_path(fn, "fn"), nx, ny, nz, nw);
}
