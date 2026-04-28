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

## Description:
Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A
notable feature of this library is its capacity to manage multiple channels within a single HDF5
file. In this framework, channels can be arranged in a multi-dimensional array format.
Once an HDF5 file has been created, the number of channels in the storage layout is fixed.
However, it is possible to reshape the layout using `quadriga_lib.channel.hdf5_reshape_layout`.

## Usage:
```
from quadriga_lib import channel
channel.hdf5_reshape_layout( fn, storage_dims );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`nx`** (optional)<br>
  Number of elements on the x-dimension, Default = 65536

- **`ny`** (optional)<br>
  Number of elements on the x-dimension, Default = 1

- **`nz`** (optional)<br>
  Number of elements on the x-dimension, Default = 1

- **`nw`** (optional)<br>
  Number of elements on the x-dimension, Default = 1
MD!*/

void hdf5_reshape_layout(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    quadriga_lib::hdf5_reshape_layout(fn, nx, ny, nz, nw);
}