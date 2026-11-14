// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# calc_directivity
Calculates the directivity in dBi of array antenna elements

- Directivity = 10·log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores element coupling
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
directivity_dBi = quadriga_lib.arrayant.calc_directivity( arrayant, i_element )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate]] for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)

## Outputs:
- **`directivity_dBi`** — Directivity in dBi; `(n_out,)` for a single-frequency input,
  `(n_out, n_freq)` for a multi-frequency input; `n_out = n_elements` when `i_element` is None

## See also:
- [[combine_pattern]] (apply element coupling before calculating directivity)
- [[calc_beamwidth]] (calculates the beam width of array antennas)
MD!*/

py::array_t<double> arrayant_calc_directivity(const py::dict &arrayant,
                                              py::handle i_element)
{
    // Pattern arrays as views; 3D single-frequency and 4D multi-frequency arrays; validated by the C++ function
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, false, false);
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true);

    if (ant_multi.size() == 1) // Single frequency: 1D output (n_out,)
    {
        arma::vec directivity_dBi;
        quadriga_lib::arrayant_calc_directivity(ant_multi[0], directivity_dBi, &i_el);
        return qd_python_copy2numpy(&directivity_dBi);
    }

    // Multi-frequency: 2D output (n_out, n_freq)
    arma::mat directivity_dBi;
    quadriga_lib::arrayant_calc_directivity_multi(ant_multi, directivity_dBi, &i_el);
    return qd_python_copy2numpy(&directivity_dBi);
}

// pybind11 declaration:
// m.def("calc_directivity", &arrayant_calc_directivity,
//       py::arg("arrayant"),
//       py::arg("i_element") = py::none());
