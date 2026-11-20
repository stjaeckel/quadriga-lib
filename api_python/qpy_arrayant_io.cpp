// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"


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
            throw std::runtime_error("File '" + fn_s + "' does not contain any antenna data.");
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

/*!MD
# qdant_write
Writes array antenna data to QDANT files

- QDANT is the QuaDRiGa array antenna exchange format, an XML format for storing antenna patterns
- Single-frequency input (3D pattern fields) writes one entry; `id` places it in the file, and
  several antennas can share one file via distinct IDs
- Frequency-dependent input (4D pattern fields) writes every frequency entry with sequential
  1-based IDs; `id` and `layout` do not apply, and the file must not already exist
- Returns the ID assigned to the written entry (`0` for a frequency-dependent write)
- Without `layout`, the layout in the file is replaced by a row of all IDs in the file in file order;
  a custom layout written before is lost (pass it again with every write that should keep it)
- The file has single-precision-equivalent accuracy relative to the element peak: pattern
  magnitudes (dB) and phases (degree) are rounded so that every sample is within 6e-8 of the peak
  amplitude of its element; samples far below the peak are less accurate relative to their own
  amplitude (about 6e-4 at -80 dB, 6e-2 at -120 dB below the peak); positions and coupling are
  written as float values (at most 9 significant digits); sampling grids are rounded to 0.001
  degree; all non-zero pattern samples are written at any level, exact zeros as `-inf` dB

## Usage:
```
# Single-frequency write
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant )
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant, id, *, layout )

# Frequency-dependent write (4D patterns) — all entries written sequentially
quadriga_lib.arrayant.qdant_write( fn, arrayant )
```

## Inputs:
- **`fn`** — Output QDANT filename; must not be empty
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`; 4D input
  requires `center_freq` with one value per frequency entry `(n_freq,)`
- **`id`** — Target 1-based ID of the entry inside the file; `0` appends after the highest
  existing ID (or 1 if the file does not exist); ignored for 4D input; default: 0
- **`layout`** — uint32 matrix organizing multiple antenna IDs within the file; must reference
  only IDs present in the file; ignored for 4D input; default: `None` (a row of all IDs in the file,
  which replaces a layout written before)

## Output:
- **`id_in_file`** — ID assigned to the entry in the file after writing; `0` for a frequency-dependent (4D) write

## See also:
- [[qdant_read]] (for reading QDANT data)
- [[generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#qdant">QDANT</a>)
MD!*/

py::ssize_t qdant_write(const py::handle &fn,
                                 const py::dict &arrayant,
                                 unsigned id,
                                 py::handle layout)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, true, false, true);

    if (ant_vec.empty())
        throw std::invalid_argument("Input 'arrayant' does not contain any antenna data.");

    // Frequency-dependent input: write every entry with sequential 1-based IDs.
    if (ant_vec.size() > 1)
    {
        quadriga_lib::qdant_write_multi(fn_s, ant_vec);
        return 0;
    }

    // Single-frequency input: write one entry at the requested ID.
    const auto layout_a = qd_python_numpy2arma_Mat<unsigned>(layout, true, false, "layout");
    return (py::ssize_t)quadriga_lib::qdant_write(fn_s, ant_vec[0], id, layout_a);
}

/*!MD
# export_obj_file
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- The pattern is mapped onto an icosphere; a higher `icosphere_n_div` gives a finer mesh
- The OBJ file is written to `fn`; the function returns nothing
- Accepts a frequency-dependent antenna (4D pattern fields); `i_freq` selects which frequency
  entry is exported

## Usage:
```
quadriga_lib.arrayant.export_obj_file( fn, arrayant, *, directivity_range_dB, colormap, object_radius, \
    icosphere_n_div, i_element, i_freq )
```

## Inputs:
- **`fn`** — Output OBJ filename; must not be empty and must end in `.obj`
- **`arrayant`** — Dict with the array antenna data; keys as in [[generate]]; pattern fields may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`directivity_range_dB`** — Dynamic range of the visualized directivity pattern in dB; default: 30.0
- **`colormap`** — Colormap name; default: jet; Available: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer
- **`object_radius`** — Radius of the exported object in meters; default: 1.0
- **`icosphere_n_div`** — Icosphere subdivision count; higher gives a finer mesh; see [[icosphere]]; default: 4
- **`i_element`** — Element indices to export; 1D list or array of int; `None` or empty exports all elements; default: `None`
- **`i_freq`** — Frequency index to export from a frequency-dependent antenna; must satisfy `0 <= i_freq < n_freq`; default: 0
MD!*/

void arrayant_export_obj_file(const py::handle &fn,
                              const py::dict &arrayant,
                              double directivity_range_dB,
                              const std::string &colormap,
                              double object_radius,
                              arma::uword icosphere_n_div,
                              py::handle i_element,
                              arma::uword i_freq)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    // Element indices (0-based); empty uvec = export all elements
    const arma::uvec i_element_a = qd_python_numpy2arma_Col<arma::uword>(i_element, true, false, "i_element");

    const auto ant = qd_python_dict2arrayant_multi(arrayant, true, false, true, false);

    if (i_freq >= ant.size())
        throw std::invalid_argument("Input 'i_freq' is out of range.");

    quadriga_lib::arrayant_export_obj_file(fn_s, ant[i_freq], directivity_range_dB, colormap, object_radius, icosphere_n_div, i_element_a);
}
