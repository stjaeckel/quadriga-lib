// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# calc_beamwidth
Calculate the beamwidth and pointing angles of array antenna elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern
  (≈100x finer grid in each direction than the antenna sampling grid)
- Calculated per element, not per port; ignores element coupling
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
bw_az, bw_el, az_pt, el_pt = quadriga_lib.arrayant.calc_beamwidth( arrayant )
bw_az, bw_el, az_pt, el_pt = quadriga_lib.arrayant.calc_beamwidth( arrayant, element, threshold_dB )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate]] for the field layout
- **`element`** — Element indices (0-based); if None or empty, all elements are used; `(n_out,)` or None;
  default: None
- **`threshold_dB`** — Threshold in dB (3 = FWHM); default: 3.0

## Outputs:
- **`bw_az`** — Azimuth beamwidth in degrees; `(n_out,)` for single-frequency input,
  `(n_out, n_freq)` for multi-frequency input; `n_out = n_elements` when `element` is None
- **`bw_el`** — Elevation beamwidth in degrees; same shape as `bw_az`
- **`az_pt`** — Azimuth pointing angle of the main beam in degrees; same shape as `bw_az`
- **`el_pt`** — Elevation pointing angle of the main beam in degrees; same shape as `bw_az`

## See also:
- [[combine_pattern]] (apply element coupling before calculating beamwidth)
- [[calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

py::tuple arrayant_calc_beamwidth(const py::dict &arrayant,
                                  py::handle element,
                                  double threshold_dB)
{
    // Parse arrayant (handles 3D single-freq and 4D multi-freq pattern arrays uniformly)
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, true);
    const size_t n_freq = ant_multi.size();
    const arma::uword n_elements = ant_multi[0].e_theta_re.n_slices;

    if (n_elements == 0)
        throw std::invalid_argument("Array antenna has no elements.");

    // Element indices (0-based); empty input → all elements
    const auto element_a = qd_python_numpy2arma_Col<arma::uword>(element, true);
    const arma::uvec element_ind = element_a.empty() ? arma::regspace<arma::uvec>(0, n_elements - 1) : element_a;

    if (n_freq == 1)
    {
        // Single frequency: 1D outputs (n_out,)
        arma::vec bw_az, bw_el, az_pt, el_pt;
        auto bw_az_py = qd_python_init_output(element_ind.n_elem, &bw_az);
        auto bw_el_py = qd_python_init_output(element_ind.n_elem, &bw_el);
        auto az_pt_py = qd_python_init_output(element_ind.n_elem, &az_pt);
        auto el_pt_py = qd_python_init_output(element_ind.n_elem, &el_pt);

        double *p_bw_az = bw_az.memptr();
        double *p_bw_el = bw_el.memptr();
        double *p_az_pt = az_pt.memptr();
        double *p_el_pt = el_pt.memptr();
        for (auto el : element_ind)
            ant_multi[0].calc_beamwidth_deg(el, threshold_dB, p_bw_az++, p_bw_el++, p_az_pt++, p_el_pt++);

        return py::make_tuple(bw_az_py, bw_el_py, az_pt_py, el_pt_py);
    }

    // Multi-frequency: 2D outputs (n_out, n_freq)
    arma::mat bw_az, bw_el, az_pt, el_pt;
    auto bw_az_py = qd_python_init_output(element_ind.n_elem, (arma::uword)n_freq, &bw_az);
    auto bw_el_py = qd_python_init_output(element_ind.n_elem, (arma::uword)n_freq, &bw_el);
    auto az_pt_py = qd_python_init_output(element_ind.n_elem, (arma::uword)n_freq, &az_pt);
    auto el_pt_py = qd_python_init_output(element_ind.n_elem, (arma::uword)n_freq, &el_pt);

    for (size_t i_freq = 0; i_freq < n_freq; ++i_freq)
    {
        double *p_bw_az = bw_az.colptr(i_freq);
        double *p_bw_el = bw_el.colptr(i_freq);
        double *p_az_pt = az_pt.colptr(i_freq);
        double *p_el_pt = el_pt.colptr(i_freq);
        for (auto el : element_ind)
            ant_multi[i_freq].calc_beamwidth_deg(el, threshold_dB, p_bw_az++, p_bw_el++, p_az_pt++, p_el_pt++);
    }

    return py::make_tuple(bw_az_py, bw_el_py, az_pt_py, el_pt_py);
}

// pybind11 declaration:
// m.def("calc_beamwidth", &arrayant_calc_beamwidth,
//       py::arg("arrayant"),
//       py::arg("element") = py::none(),
//       py::arg("threshold_dB") = 3.0);