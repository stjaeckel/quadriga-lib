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

- Returns the dimensions of the 4D channel-slot grid stored inside an HDF5 file
- Also reports which slots already hold data, so free slots can be found without scanning the file
- Returns `(0, 0, 0, 0)` dimensions if the file does not exist
- Raises an error if the file exists but is not a valid HDF5 file

## Usage:
```
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str

## Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `(4,)`
- **`has_data`** — Slot occupancy mask; 1 where a slot holds data, 0 otherwise; `(nx, ny, nz, nw)`

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

py::tuple hdf5_read_layout(const std::string &fn)
{
    // Read the storage layout and the per-slot channel IDs
    arma::Col<unsigned> channelID;
    arma::Col<unsigned> storage_space = quadriga_lib::hdf5_read_layout(fn, &channelID);

    const unsigned nx = storage_space.at(0), ny = storage_space.at(1),
                   nz = storage_space.at(2), nw = storage_space.at(3);

    if (channelID.n_elem != (arma::uword)nx * ny * nz * nw)
        throw std::runtime_error("Corrupted storage index.");

    // storage_dims: 4-element vector [nx, ny, nz, nw]
    auto storage_dims = qd_python_copy2numpy(storage_space);

    // has_data: 4D occupancy mask (1 where a slot holds data, 0 otherwise)
    std::vector<arma::Cube<unsigned>> has_data_cubes;
    auto has_data = qd_python_init_output(nx, ny, nz, nw, &has_data_cubes);
    for (unsigned w = 0; w < nw; ++w)
    {
        unsigned *dst = has_data_cubes[w].memptr();
        const unsigned *src = channelID.memptr() + (size_t)w * nx * ny * nz;
        for (size_t i = 0, n = (size_t)nx * ny * nz; i < n; ++i)
            dst[i] = (src[i] != 0u) ? 1u : 0u;
    }

    return py::make_tuple(storage_dims, has_data);
}

// pybind11 declaration:
// m.def("hdf5_read_layout", &hdf5_read_layout,
//       py::arg("fn"));