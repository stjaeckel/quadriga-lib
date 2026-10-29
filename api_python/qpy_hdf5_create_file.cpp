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

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- The total slot count is fixed at creation; the layout can be reshaped with [[hdf5_reshape_layout]]
- Raises an error if the target file already exists; delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

## Usage:
```
storage_space = quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

## Outputs:
- **`storage_space`** — Storage dimensions used by the new file, `[nx, ny, nz, nw]`; `(4,)`; int64

## See also:
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_write_dset]] (for writing arbitrary unstructured data)
MD!*/

py::array_t<py::ssize_t> hdf5_create_file(const std::string &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    // Fail if the file exists: hdf5_read_layout returns zeros only for a missing file and throws for any other file
    // it cannot read (files of older versions, other files, unreadable files)
    bool exists = true;
    try
    {
        exists = quadriga_lib::hdf5_read_layout(fn).at(0) != 0;
    }
    catch (const std::exception &)
    {
    }
    if (exists)
        throw std::invalid_argument("File already exists.");

    // Create file with the requested storage layout
    quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);

    // Return the storage layout that was used
    arma::u32_vec storage_space = {nx, ny, nz, nw};
    return qd_python_copy2numpy<unsigned, py::ssize_t>(&storage_space);
}

// pybind11 declaration:
// m.def("hdf5_create_file", &hdf5_create_file,
//       py::arg("fn"),
//       py::arg("nx") = 65536,
//       py::arg("ny") = 1,
//       py::arg("nz") = 1,
//       py::arg("nw") = 1);