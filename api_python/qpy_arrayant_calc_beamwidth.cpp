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
- The width is that of the main lobe, searched outwards from the maximum; side lobes above the threshold
  are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full
  circle; 360° when the whole cut is above the threshold (e.g. a beam pointing to a pole)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles;
  at most 180°
- Calculated per element, not per port; ignores element coupling
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg = \
    quadriga_lib.arrayant.calc_beamwidth( arrayant, i_element, threshold_dB )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate]] for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)
- **`threshold_dB`** — Threshold in dB, positive and finite (3 = FWHM); default: 3.0

## Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `(n_out,)` for single-frequency input,
  `(n_out, n_freq)` for multi-frequency input; `n_out = n_elements` when `i_element` is None
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; same shape as `beamwidth_az_deg`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`

## See also:
- [[combine_pattern]] (apply element coupling before calculating beamwidth)
- [[calc_directivity]] (directivity in dBi of array antenna elements)
MD!*/

py::tuple arrayant_calc_beamwidth(const py::dict &arrayant,
                                  py::handle i_element,
                                  double threshold_dB)
{
    // Pattern arrays as views; 3D single-frequency and 4D multi-frequency arrays; validated by the C++ function
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, false, false);
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true);

    if (ant_multi.size() == 1) // Single frequency: 1D outputs (n_out,)
    {
        arma::vec bw_az, bw_el, az_pt, el_pt;
        quadriga_lib::arrayant_calc_beamwidth(ant_multi[0], &i_el, threshold_dB, &bw_az, &bw_el, &az_pt, &el_pt);
        return py::make_tuple(qd_python_copy2numpy(&bw_az), qd_python_copy2numpy(&bw_el),
                              qd_python_copy2numpy(&az_pt), qd_python_copy2numpy(&el_pt));
    }

    // Multi-frequency: 2D outputs (n_out, n_freq)
    arma::mat bw_az, bw_el, az_pt, el_pt;
    quadriga_lib::arrayant_calc_beamwidth_multi(ant_multi, &i_el, threshold_dB, &bw_az, &bw_el, &az_pt, &el_pt);
    return py::make_tuple(qd_python_copy2numpy(&bw_az), qd_python_copy2numpy(&bw_el),
                          qd_python_copy2numpy(&az_pt), qd_python_copy2numpy(&el_pt));
}

// pybind11 declaration:
// m.def("calc_beamwidth", &arrayant_calc_beamwidth,
//       py::arg("arrayant"),
//       py::arg("i_element") = py::none(),
//       py::arg("threshold_dB") = 3.0);
