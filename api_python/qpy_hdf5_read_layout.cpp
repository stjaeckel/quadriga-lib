// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

## Description:
Quadriga-Lib provides an HDF5-based solution for the storage and organization of channel data. A
notable feature of this library is its capacity to manage multiple channels within a single HDF5
file. In this framework, channels can be arranged in a multi-dimensional array format.

The function `quadriga_lib.channel.hdf5_read_layout` is designed to read the storage layout from an
existing file. Furthermore, it also generates an array that marks the locations within the layout
where data already exists. This functionality aids in efficiently managing and accessing channel
data within the HDF5 file structure.

## Usage:
```
from quadriga_lib import channel
storage_dims, has_data = channel.hdf5_read_layout( fn )
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

py::tuple hdf5_read_layout(std::string fn)
{
    arma::Col<unsigned> channelID;
    arma::Col<unsigned> storage_space = quadriga_lib::hdf5_read_layout(fn, &channelID);

    auto nx = (py::ssize_t)storage_space.at(0);
    auto ny = (py::ssize_t)storage_space.at(1);
    auto nz = (py::ssize_t)storage_space.at(2);
    auto nw = (py::ssize_t)storage_space.at(3);

    auto n_bytes = (py::ssize_t)sizeof(unsigned);
    py::ssize_t strides[4] = {n_bytes, ny * n_bytes, nx * ny * n_bytes, nx * ny * nz * n_bytes};
    auto has_data = py::array_t<unsigned>({nx, ny, nz, nw}, strides, channelID.memptr());

    return py::make_tuple(py::array_t<unsigned>(4ULL, storage_space.memptr()), has_data);
}