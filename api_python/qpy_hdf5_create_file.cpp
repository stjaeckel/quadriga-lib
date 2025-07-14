// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_CREATE_FILE
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