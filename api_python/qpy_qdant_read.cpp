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
- Without `id` (or with `id = None`), every entry is read: a frequency-dependent dict with 4D pattern fields
  when the file holds multiple entries, or a plain single-entry dict (3D fields) when it holds exactly one;
  multiple entries must form a frequency set (same number of elements, same azimuth and elevation grids, element
  positions and coupling size), otherwise an error asks to pass `id`
- Reading all entries is the inverse of `qdant_write` with a 4D-pattern dict
- An `id` that is not in the file raises an error; the first entry counts as ID 1 only if it has no
  `id` attribute
- Malformed files raise an error, also when they declare more elements (`NoElements`) than the file
  has data for; elements without pattern data read as zeros (at most 2^24 zero values per entry)

## Usage:
```
# Read every entry (a frequency-dependent arrayant when the file holds several)
ant, layout = quadriga_lib.arrayant.qdant_read( fn )

# Read a single entry
ant, layout = quadriga_lib.arrayant.qdant_read( fn, id )
```

## Inputs:
- **`fn`** — path to the QDANT file; must not be empty
- **`id`** — 1-based ID of the entry to read, >= 1; `None` reads all entries (see above); default: None

## Outputs:
- **`ant`** — Dict with the array antenna data; keys as in [[generate]], with these specifics:
  - `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` — Pattern fields;
    `(n_elevation, n_azimuth, n_elements)` for a single entry, or
    `(n_elevation, n_azimuth, n_elements, n_freq)` when multiple entries are read
  - `center_freq` — Center frequency in Hz; scalar for a single entry, `(n_freq,)` when reading multiple entries
  - `coupling_re`, `coupling_im` — Coupling matrices; `(n_elements, n_ports)`, or
    `(n_elements, n_ports, n_freq)` when reading multiple entries with per-entry coupling
- **`layout`** — Matrix of element IDs describing how the entries are arranged in the file; int64

## See also:
- [[qdant_write]] (for writing QDANT data)
- [[generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#qdant">QDANT</a>)
MD!*/

py::tuple qdant_read(const py::handle &fn,
                    py::handle id)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    arma::Mat<unsigned> layout;
    py::dict output;

    if (id.is_none()) // Read every entry as a (possibly frequency-dependent) arrayant
    {
        auto ant_vec = quadriga_lib::qdant_read_multi<double>(fn_s, &layout);
        if (ant_vec.empty())
            throw std::runtime_error("File does not contain any antenna data.");
        if (ant_vec.size() > 1) // The entries must form a frequency set
        {
            std::string err = quadriga_lib::arrayant_is_valid_multi(ant_vec);
            if (!err.empty())
                throw std::invalid_argument("Entries in file '" + fn_s + "' do not form a frequency set (" + err +
                                            "); pass 'id' to read single entries.");
        }
        output = qd_python_arrayant2dict_multi(ant_vec);
    }
    else // Read a single entry by its 1-based ID
    {
        long long i = 0;
        try
        {
            i = py::cast<long long>(id);
        }
        catch (const py::cast_error &)
        {
            throw py::type_error("Input 'id' must be an integer or None.");
        }
        if (i < 1 || i > (long long)std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Input 'id' must be >= 1 (None reads all entries).");
        auto ant = quadriga_lib::qdant_read<double>(fn_s, (unsigned)i, &layout);
        output = qd_python_arrayant2dict(ant);
    }

    return py::make_tuple(output, qd_python_copy2numpy<unsigned, py::ssize_t>(&layout));
}
