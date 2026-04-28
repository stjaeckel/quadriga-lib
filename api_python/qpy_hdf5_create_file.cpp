// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

## Description:
Quadriga-Lib offers an HDF5-based method for storing and managing channel data. A key feature of this
library is its ability to organize multiple channels within a single HDF5 file while enabling access
to individual data sets without the need to read the entire file. In this system, channels can be
structured in a multi-dimensional array. For instance, the first dimension might represent the Base
Station (BS), the second the User Equipment (UE), and the third the frequency. However, it is important
to note that the dimensions of the storage layout must be defined when the file is initially created
and cannot be altered thereafter. The function `quadriga_lib.channel.hdf5_create_file` is used to create an
empty file with a predetermined custom storage layout.

## Usage:
```
from quadriga_lib import channel
channel.hdf5_create_file( fn, nx, ny, nz, nw )
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

void hdf5_create_file(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
  auto storage_space = quadriga_lib::hdf5_read_layout(fn);
  
  if (storage_space.at(0) != 0)
    throw std::invalid_argument("File already exists.");

  quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);
}