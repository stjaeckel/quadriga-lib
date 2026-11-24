// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"


/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# qdant_read
Read arrayant objects from a QDANT file

- Loads arrays from a file in the [QDANT format](formats.html#qdant). A file can hold several arrays, each with a
  numeric ID, e.g. one per frequency of a frequency-dependent model, where `center_freq` of each entry identifies the
  frequency

## Usage:
```
# Read the entries in the layout (a frequency-dependent arrayant when the file holds several)
ant, layout = quadriga_lib.arrayant.qdant_read( fn )

# Read a single entry
ant, layout = quadriga_lib.arrayant.qdant_read( fn, ids )

# Read several entries (list of dicts)
ant_list, layout = quadriga_lib.arrayant.qdant_read( fn, [3, 1, 3] )
```

## Inputs:
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** — 1-based ID of the entry to read (int), or a list or array of IDs, read in this order, duplicates allowed;
  an empty list returns an empty list; default: `None` (the entries listed in the layout, each once, in order of first
  appearance (column-major scan), or all entries if the file has no layout; several entries must then form a
  multi-frequency array, otherwise an error asks to pass `ids`)

## Outputs:
- **`ant`** — Arrayant dict ([[generate_antenna]]) with 3D pattern arrays for an int; a list of such dicts for a list or
  array; with `ids=None`, a 3D dict for one entry and a 4D multi-frequency dict for several
- **`layout`** — The file's layout matrix of entry IDs (0 = empty slot); int64

## Notes:
- **IDs**: the number used in the layout matrix and by [[qdant_write]]. A first entry without an `id` attribute (files
  from other tools) counts as ID 1
- **Errors**: an ID that is 0 or not in the file, and a missing, unreadable or malformed file, raise an error
- **Elements without data**: elements without pattern nodes read as zeros. A file whose element count is not backed by
  its data, or that exceeds the [reader limits](formats.html#qdant), raises an error

## See also:
- [[qdant_write]] (writes such files; reading all entries is the inverse of writing a 4D dict)
- [[generate_antenna]], [[generate_speaker]] (create arrays without a file)
MD!*/

py::tuple qdant_read(const py::handle &fn,
                     py::handle ids)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    arma::Mat<unsigned> layout;
    py::object output;

    if (ids.is_none()) // Read every entry as a (possibly frequency-dependent) arrayant
    {
        auto ant_vec = quadriga_lib::qdant_read<double>(fn_s, {}, &layout);
        if (ant_vec.empty())
            throw std::runtime_error("File '" + fn_s + "' does not contain any antenna data.");
        if (ant_vec.size() > 1) // The entries must form a frequency set
        {
            std::string err = quadriga_lib::is_valid(ant_vec);
            if (!err.empty())
                throw std::invalid_argument("Entries in file '" + fn_s + "' do not form a frequency set (" + err +
                                            "); pass 'ids' to read single entries.");
        }
        output = qd_python_arrayant2dict_multi(ant_vec);
    }
    else if (py::isinstance<py::list>(ids) || py::isinstance<py::tuple>(ids) ||
             (py::isinstance<py::array>(ids) && py::cast<py::array>(ids).ndim() > 0)) // Read the listed entries
    {
        const arma::uvec ids_a = qd_python_numpy2arma_Col<arma::uword>(ids, false, false, "ids");
        py::list ant_list;
        if (ids_a.n_elem == 0) // An empty list reads no entries; the layout is still returned
        {
            quadriga_lib::qdant_read<double>(fn_s, {}, &layout);
            return py::make_tuple(ant_list, qd_python_copy2numpy<unsigned, py::ssize_t>(&layout));
        }
        auto ant_vec = quadriga_lib::qdant_read<double>(fn_s, ids_a, &layout);
        for (const auto &ant : ant_vec)
            ant_list.append(qd_python_arrayant2dict(ant));
        output = ant_list;
    }
    else // Read a single entry by its 1-based ID
    {
        long long i = 0;
        try
        {
            i = py::cast<long long>(ids);
        }
        catch (const py::cast_error &)
        {
            throw py::type_error("Input 'ids' must be an integer, a list of integers or None.");
        }
        if (i < 1 || i > (long long)std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Input 'ids' must be >= 1 (None reads the entries in the layout).");
        auto ant_vec = quadriga_lib::qdant_read<double>(fn_s, i, &layout);
        output = qd_python_arrayant2dict(ant_vec[0]);
    }

    return py::make_tuple(output, qd_python_copy2numpy<unsigned, py::ssize_t>(&layout));
}

/*!MD
# qdant_write
Write arrayant data to a QDANT (XML) file

- Stores an array in a file in the [QDANT format](formats.html#qdant). A file can hold several arrays with distinct IDs
  and a layout matrix that arranges them
- 4D pattern arrays store a frequency-dependent model in a new file, each entry holding the pattern at its `center_freq`

## Usage:
```
# Single-frequency write
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant )
id_in_file = quadriga_lib.arrayant.qdant_write( fn, arrayant, id, *, layout )

# Frequency-dependent write (4D patterns) — all entries written sequentially
quadriga_lib.arrayant.qdant_write( fn, arrayant )
```

## Inputs:
- **`fn`** — Output QDANT file name, must not be empty; an existing file is updated; 4D input: the file must not exist
  (raises `RuntimeError` and leaves the file unchanged)
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array, written with
  the sequential IDs `1, 2, ..., n_freq` and an `(n_freq, 1)` layout of these IDs
- **`id`** — Target ID in the file; an entry with this ID is replaced; 0 appends with `max_existing_id + 1`, or 1 in a
  new file; ignored for 4D input; default: 0
- **`layout`** — Integer matrix of entry IDs that arranges the file; every value must be an ID in the file (0 is not
  accepted); ignored for 4D input; default: `None`, which replaces the layout in the file by a `(1, n)` row of all IDs
  in file order, so a custom layout must be passed again with every write that should keep it

## Output:
- **`id_in_file`** — ID assigned to the written antenna within the file; 0 for 4D input

## Notes:
- **Pattern accuracy**: the file has single-precision-equivalent accuracy relative to the element peak. Pattern
  magnitudes (in dB) and phases (in degrees) get the fewest decimals (0 to 6) that keep the error of each complex sample
  at or below 6e-8 times the peak amplitude of its element (over `e_theta` and `e_phi`). Samples near the peak are
  accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the peak (about 6e-4 at
  -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at a low level is
  quantized relative to its own peak
- **Other values**: element positions and port weights (magnitude, phase) are written as the shortest text that reads
  back as the same float (at most 9 significant digits); sampling grids are rounded to 0.001 degree
- **Zeros**: all non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element
  that is zero in both field components is written as an all `-inf` `e_theta_mag` node

## See also:
- [[qdant_read]] (reads the file back)
- [[generate_speaker]] (typical source of frequency-dependent arrays)
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
        quadriga_lib::qdant_write(fn_s, ant_vec);
        return 0;
    }

    // Single-frequency input: write one entry at the requested ID.
    const auto layout_a = qd_python_numpy2arma_Mat<unsigned>(layout, true, false, "layout");
    return (py::ssize_t)quadriga_lib::qdant_write(fn_s, ant_vec[0], id, layout_a);
}

/*!MD
# export_pattern_obj
Export antenna pattern geometry to a Wavefront OBJ file for 3D visualization

- Writes the power pattern of each element as a colored 3D surface, one object per element, placed at its position in
  `element_pos`
- The distance of the surface from the element is linear in the level in dB: `object_radius` at the peak over all
  exported elements, 0 at `directivity_range_dB` below it. The color shows the same level

## Usage:
```
quadriga_lib.arrayant.export_pattern_obj( fn, arrayant, *, directivity_range_dB, colormap, object_radius, \
    icosphere_n_div, i_element, i_freq )
```

## Inputs:
- **`fn`** — Output OBJ file name, must end in `.obj`; the colormap is written to a `.mtl` file of the same name;
  existing files are overwritten
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array; the exported
  elements must have finite field values and must not be zero in all directions
- **`directivity_range_dB`** — Range of the plot below the peak in dB, finite and not 0; the sign is ignored; default:
  30.0
- **`colormap`** — Colormap name: `jet`, `parula`, `winter`, `hot`, `turbo`, `copper`, `spring`, `cool`, `gray`,
  `autumn` or `summer`; default: `jet`
- **`object_radius`** — Radius of the surface at the peak in m, positive and finite; default: 1.0
- **`icosphere_n_div`** — Subdivisions of the [[icosphere]] that carries the surface; higher = finer mesh with $20 n^2$
  faces; default: 4
- **`i_element`** — Element indices to export, 0-based; list or array of int; default: `None` (all elements)
- **`i_freq`** — Frequency entry of a 4D input to export, 0-based, below `n_freq`; default: 0

## See also:
- [[calc_directivity]], [[calc_beamwidth]] (the same pattern as numbers)
- [[channel.export_obj_file]] (exports propagation paths to an OBJ file)
MD!*/

void export_pattern_obj(const py::handle &fn,
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

    quadriga_lib::export_pattern_obj(fn_s, ant[i_freq], directivity_range_dB, colormap, object_radius, icosphere_n_div, i_element_a);
}
