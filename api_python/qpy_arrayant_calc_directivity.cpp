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
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
directivity = quadriga_lib.arrayant.calc_directivity( arrayant )
directivity = quadriga_lib.arrayant.calc_directivity( arrayant, element )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate]] for the field layout
- **`element`** — Element indices; if None or empty, all elements are used; `(n_out,)` or None; default: None

## Outputs:
- **`directivity`** — Directivity in dBi; `(n_out,)` for a single-frequency input,
  `(n_out, n_freq)` for a multi-frequency input; `n_out = n_elements` when `element` is None

## See also:
- [[combine_pattern]] (apply element coupling before calculating directivity)
- [[calc_beamwidth]] (calculates the beam width of array antennas)
MD!*/

py::array_t<double> arrayant_calc_directivity(const py::dict &arrayant,
                                              py::handle element)
{
    // Parse arrayant (handles 3D single-freq and 4D multi-freq pattern arrays uniformly)
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, true);
    const size_t n_freq = ant_multi.size();
    const arma::uword n_elements = ant_multi[0].e_theta_re.n_slices;

    if (n_elements == 0)
        throw std::invalid_argument("Array antenna has no elements.");

    // Element indices; empty input → all elements
    const auto element_a = qd_python_numpy2arma_Col<arma::uword>(element, true);
    const arma::uvec element_ind = element_a.empty() ? arma::regspace<arma::uvec>(0, n_elements - 1) : element_a;

    if (n_freq == 1)
    {
        // Single frequency: 1D output (n_out,)
        arma::vec directivity;
        auto out_py = qd_python_init_output(element_ind.n_elem, &directivity);

        auto *p_directivity = directivity.memptr();
        for (auto el : element_ind)
            *p_directivity++ = ant_multi[0].calc_directivity_dBi(el);

        return out_py;
    }

    // Multi-frequency: 2D output (n_out, n_freq)
    arma::mat directivity;
    auto out_py = qd_python_init_output(element_ind.n_elem, (arma::uword)n_freq, &directivity);

    for (size_t i_freq = 0; i_freq < n_freq; ++i_freq)
    {
        auto *p_directivity = directivity.colptr(i_freq);
        for (auto el : element_ind)
            *p_directivity++ = ant_multi[i_freq].calc_directivity_dBi(el);
    }

    return out_py;
}

// pybind11 declaration:
// m.def("calc_directivity", &arrayant_calc_directivity,
//       py::arg("arrayant"),
//       py::arg("element") = py::none());
