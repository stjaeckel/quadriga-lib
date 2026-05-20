// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"
#include <filesystem>

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# qdant_write
Writes array antenna data to QDANT files

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- Single-frequency input (3D pattern fields) writes one entry; `id` places it in the file, and
  several antennas can share one file via distinct IDs
- Frequency-dependent input (4D pattern fields) writes every frequency entry with sequential
  1-based IDs; `id` and `layout` do not apply, and the file must not already exist
- Returns the ID assigned to the written entry (`0` for a frequency-dependent write)

## Usage:
```
# Single-frequency write
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant )
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant, id )

# Frequency-dependent write (4D patterns) — all entries written sequentially
arrayant.quadriga_lib.qdant_write( fn, arrayant )
```

## Inputs:
- **`fn`** — Output QDANT filename; str; must not be empty
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`id`** — Target 1-based ID of the entry inside the file; `0` appends after the highest
  existing ID (or 1 if the file does not exist); ignored for 4D input; default: 0
- **`layout`** — uint32 matrix organizing multiple antenna IDs within the file; must reference
  only IDs present in the file; ignored for 4D input; default: `None`

## Outputs:
- **`id_in_file`** — ID assigned to the entry in the file after writing; `0` for a frequency-dependent (4D) write

## See also:
- [[qdant_read]] (for reading QDANT data)
- [[generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#6cab4884">QDANT</a>)
MD!*/

py::ssize_t arrayant_qdant_write(const std::string &fn,
                                 const py::dict &arrayant,
                                 unsigned id,
                                 py::handle layout)
{
    if (fn.empty())
        throw std::invalid_argument("File name cannot be empty.");

    // Parse the (possibly frequency-dependent) antenna via the unified multi-freq reader
    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, true, false, true);

    if (ant_vec.empty())
        throw std::invalid_argument("'arrayant' does not contain any antenna data.");

    // Frequency-dependent input: write every entry with sequential 1-based IDs.
    if (ant_vec.size() > 1)
    {
        if (std::filesystem::exists(fn))
            throw std::runtime_error("File exists. Writing a frequency-dependent arrayant to an existing file is not allowed.");
        quadriga_lib::qdant_write_multi(fn, ant_vec);
        return 0;
    }

    // Single-frequency input: write one entry at the requested ID.
    const auto layout_a = qd_python_numpy2arma_Mat<unsigned>(layout, true);
    return (py::ssize_t)ant_vec[0].qdant_write(fn, id, layout_a);
}

// pybind11 declaration (register under the `arrayant` submodule in python_main.cpp):
// m.def("qdant_write", &arrayant_qdant_write,
//       py::arg("fn"),
//       py::arg("arrayant"),
//       py::arg("id") = 0,
//       py::arg("layout") = py::none());