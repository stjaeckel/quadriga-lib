// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_python_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_RESHAPE_LAYOUT
Reshapes the storage layout inside an existing HDF5 file

## Description:
Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A
notable feature of this library is its capacity to manage multiple channels within a single HDF5
file. In this framework, channels can be arranged in a multi-dimensional array format.
Once an HDF5 file has been created, the number of channels in the storage layout is fixed.
However, it is possible to reshape the layout using `quadriga_lib.hdf5_reshape_layout`.

## Usage:

```
quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
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