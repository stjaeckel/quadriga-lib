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
Write a single unstructured dataset to an HDF5 file

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must contain only alphanumeric characters and underscores
- The file must already exist (use [[hdf5_create_file]] first)
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [[hdf5_write_channel]] to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: str; Python float (stored as double), int and bool (stored as int64; a bool is stored as the
  integer 0 or 1 and reads back as an int); numpy scalars and 0-d arrays; numpy arrays with 1, 2 or 3 dimensions
  in any memory layout and byte order; numeric element types: float32, float64, int32, uint32, int64, uint64.
  Other types (e.g. int16, numpy bool, complex) raise an error
- 1D arrays are stored as vectors and read back with shape `(n,)`, also with a single element (not as a scalar);
  2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`

## Usage:
```
quadriga_lib.channel.hdf5_write_dset( fn, ix, iy, iz, iw, name, data )
```

## Input Arguments:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; str
- **`data`** — Data to be written; type must be supported (see above); cannot be empty

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
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
