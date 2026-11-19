// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

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

    // Parse the (possibly frequency-dependent) input antenna
    const auto ant = qd_python_dict2arrayant_multi(arrayant, true, false, true, false);

    if (i_freq >= ant.size())
        throw std::invalid_argument("'i_freq' is out of bound.");

    quadriga_lib::arrayant_export_obj_file(fn_s, ant[i_freq], directivity_range_dB, colormap, object_radius, icosphere_n_div, i_element_a);
}
