// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_write_dset
Writes unstructured data to a HDF5 file

## Description:
Quadriga-Lib offers a solution based on HDF5 for storing and organizing channel data. In addition
to structured datasets, the library facilitates the inclusion of extra datasets of various types
and shapes. This feature is particularly beneficial for integrating descriptive data or analysis
results. The function `quadriga_lib.channel.hdf5_write_dset` writes a single unstructured dataset.

## Usage:

```
from quadriga_lib import channel
channel.hdf5_write_dset( fn, ix, iy, iz, iw, name, data );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`ix`**<br>
  Storage index for x-dimension, Default = 0

- **`iy`**<br>
  Storage index for y-dimension, Default = 0

- **`iz`**<br>
  Storage index for z-dimension, Default = 0

- **`iw`**<br>
  Storage index for w-dimension, Default = 0

- **`name`**<br>
  Name of the dataset; String

- **`data`**<br>
  Data to be written

## Caveat:
- Throws an error if dataset already exists at this location
- Throws an error if file does not exist (use hdf5_create_file)
- Supported types: string, double, float, (u)int32, (u)int64
- Supported size: up to 3 dimensions
- Storage order is maintained
MD!*/

void hdf5_write_dset(const std::string fn,
                     unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                     const std::string name = "",
                     const py::handle data = py::none())
{

    // Check if data is None
    if (data.is_none())
        throw std::invalid_argument("Data not provided.");

    if (name.empty())
        throw std::invalid_argument("Dataset name not provided.");

    // Convert data to a std::any
    auto any_data = qd_python_anycast(data, name);

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    auto storage_space = quadriga_lib::hdf5_read_layout(fn);

    if (storage_space.at(0) == 0)
        throw std::invalid_argument("File does not exist.");

    if (ix > storage_space.at(0) || iy > storage_space.at(1) || iz > storage_space.at(2) || iw > storage_space.at(3))
        throw std::invalid_argument("Location exceeds storage space in HDF file.");

    // Write to file
    quadriga_lib::hdf5_write_dset(fn, name, &any_data, ix, iy, iz, iw);
}
