// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

- Reads one user-defined dataset from the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- The dataset is looked up under `'par_' + name` — the `par_` prefix is prepended internally
- The returned type and shape are defined by the dataset's HDF5 dataspace
- Returns `None` if the dataset does not exist at the requested slot
- Supported types: str, scalar, vector, 2D array, and 3D array

## Usage:
```
dset = quadriga_lib.channel.hdf5_read_dset( fn, ix, iy, iz, iw, name )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without the `par_` prefix, e.g. `'carrier_frequency'`; str

## Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; `None` if the dataset is missing

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
- [[hdf5_read_channel]] (for reading structured channel data)
MD!*/

py::object hdf5_read_dset(const std::string &fn,
                          unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                          const std::string &name)
{
    // Read the dataset; the C++ side prepends the default 'par_' prefix to 'name'
    std::any dset = quadriga_lib::hdf5_read_dset(fn, name, ix, iy, iz, iw);

    // Convert to numpy array / scalar / str; None if the dataset is missing
    return qd_python_any2numpy(dset);
}

// pybind11 declaration:
// m.def("hdf5_read_dset", &hdf5_read_dset,
//       py::arg("fn"),
//       py::arg("ix") = 0,
//       py::arg("iy") = 0,
//       py::arg("iz") = 0,
//       py::arg("iw") = 0,
//       py::arg("name"));
