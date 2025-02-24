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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "quadriga_lib.hpp"

#include "python_helpers.cpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_LAYOUT
Read the storage layout of channel data inside an HDF5 file

## Description:
Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A
notable feature of this library is its capacity to manage multiple channels within a single HDF5
file. In this framework, channels can be arranged in a multi-dimensional array format.

The function `quadriga_lib.hdf5_read_layout` is designed to read the storage layout from an
existing file. Furthermore, it also generates an array that marks the locations within the layout
where data already exists. This functionality aids in efficiently managing and accessing channel
data within the HDF5 file structure.

## Usage:

```
storage_dims, has_data = quadriga_lib.hdf5_read_layout( fn )
```

## Input Argument:
- **`fn`**<br>
  Filename of the HDF5 file, string

## Output Arguments:
- **`storage_dims`**<br>
  Size of the dimensions of the storage space, vector with 4 elements, i.e. `[nx,ny,nz,nw]`.

- **`has_data`**<br>
  Array indicating if data exists (value 1) or not (value 0); uint32; Size: `[nx,ny,nz,nw]`
MD!*/

pybind11::tuple hdf5_read_layout(std::string fn)
{
  arma::Col<unsigned> channelID;
  arma::Col<unsigned> storage_space = quadriga_lib::hdf5_read_layout(fn, &channelID);

  auto nx = (unsigned long long)storage_space.at(0);
  auto ny = (unsigned long long)storage_space.at(1);
  auto nz = (unsigned long long)storage_space.at(2);
  auto nw = (unsigned long long)storage_space.at(3);

  auto sz = (unsigned long long)sizeof(unsigned);
  size_t strides[4] = {sz, ny * sz, nx * ny * sz, nx * ny * nz * sz};
  auto has_data = pybind11::array_t<unsigned>({nx, ny, nz, nw}, strides, channelID.memptr());

  return pybind11::make_tuple(pybind11::array_t<unsigned>(4ULL, storage_space.memptr()), has_data);
}