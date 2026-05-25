// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# qdant_read
Reads array antenna data from QDANT files

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- A QDANT file may hold one or several entries (e.g. a frequency-dependent antenna model)
- `id` reads a single entry by its 1-based ID; the result is a dict with 3D pattern fields
- `id = 0` reads every entry: a frequency-dependent dict with 4D pattern fields when the file
  holds multiple entries, or a plain single-entry dict (3D fields) when it holds exactly one
- Reading all entries is the inverse of `qdant_write` with a 4D-pattern dict

## Usage:
```
# Read a single entry (default: the first entry)
data = quadriga_lib.arrayant.qdant_read( fn )
data = quadriga_lib.arrayant.qdant_read( fn, id )

# Read every entry as a frequency-dependent arrayant
data = quadriga_lib.arrayant.qdant_read( fn, id=0 )
```

## Inputs:
- **`fn`** — path to the QDANT file; str; must not be empty
- **`id`** — 1-based ID of the entry to read; `0` reads all entries (see above); default: 1

## Outputs:
- **`data`** — Dict with the array antenna data; keys as in [[generate]], with these specifics:
  - `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` — Pattern fields;
    `(n_elevation, n_azimuth, n_elements)` for a single entry, or
    `(n_elevation, n_azimuth, n_elements, n_freq)` when `id = 0` reads multiple entries
  - `center_freq` — Center frequency in Hz; scalar for a single entry, `(n_freq,)` when reading multiple entries
  - `coupling_re`, `coupling_im` — Coupling matrices; `(n_elements, n_ports)`, or
    `(n_elements, n_ports, n_freq)` when reading multiple entries with per-entry coupling
  - `layout` — Matrix of element IDs describing how the entries are arranged in the file; uint32

## See also:
- [[qdant_write]] (for writing QDANT data)
- [[generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

py::dict arrayant_qdant_read(const std::string &fn,
                             unsigned id)
{
    arma::Mat<unsigned> layout;
    py::dict output;

    if (id == 0) // Read every entry as a (possibly frequency-dependent) arrayant
    {
        auto ant_vec = quadriga_lib::qdant_read_multi<double>(fn, &layout);
        if (ant_vec.empty())
            throw std::runtime_error("File does not contain any antenna data.");
        output = qd_python_arrayant2dict_multi(ant_vec);
    }
    else // Read a single entry by its 1-based ID
    {
        auto ant = quadriga_lib::qdant_read<double>(fn, id, &layout);
        output = qd_python_arrayant2dict(ant);
    }

    output["layout"] = qd_python_copy2numpy<unsigned, py::ssize_t>(&layout);
    return output;
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("qdant_read", &arrayant_qdant_read,
//       py::arg("fn"),
//       py::arg("id") = 1);