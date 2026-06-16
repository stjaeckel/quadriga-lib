// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_angular_spread
Calculate azimuth and elevation angular spreads with spherical wrapping

- Computes RMS azimuth and elevation angular spreads from power-weighted angles
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- RMS spread uses the 3GPP TR 38.901 second-moment definition of wrapped deviations from the
  circular mean
- When `wrapping = True`, the power-weighted mean direction is computed in Cartesian coordinates
  and all paths are rotated so the centroid lies on the equator before computing spreads,
  avoiding pole singularity artifacts
- When `wrapping = False`, spreads are computed directly from raw angles; `orientation` is zero
  and `phi`/`theta` equal the input `az`/`el`
- When `calc_bank_angle = True`, an optimal bank angle maximizing azimuth spread is derived
  analytically from eigenvectors of the 2x2 power-weighted covariance matrix; only used when
  `wrapping = True`
- When `quantize > 0`, paths within that angular distance are grouped and their powers summed

## Usage:
```
azimuth_spread, elevation_spread, orientation, phi, theta = \
    quadriga_lib.tools.calc_angular_spread( az, el, powers, wrapping, calc_bank_angle, quantize )
```

## Inputs:
- **`az`** — Azimuth angles; range -pi to pi; list of length `n_cir`, each element a 1D array of length `n_path`
- **`el`** — Elevation angles; range -pi/2 to pi/2; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers; list of length `n_cir`, each element a 1D array of length `n_path`
- **`wrapping`** — If True, enables spherical rotation; default: False
- **`calc_bank_angle`** — If True, computes the optimal bank angle analytically; only used when `wrapping = True`; default: False
- **`quantize`** — Angular quantization step in [deg]; paths within this distance are grouped; default: 0.0 (no quantization)

## Outputs:
- **`azimuth_spread`** — RMS azimuth angular spread; `(n_cir,)`
- **`elevation_spread`** — RMS elevation angular spread; `(n_cir,)`
- **`orientation`** — Power-weighted mean orientation in Euler angles [bank; tilt; heading]; `(3, n_cir)`
- **`phi`** — Rotated azimuth angles; list of length `n_cir`
- **`theta`** — Rotated elevation angles; list of length `n_cir`
MD!*/

py::tuple calc_angular_spread(py::list az,
                              py::list el,
                              py::list powers,
                              bool wrapping,
                              bool calc_bank_angle,
                              double quantize)
{
    // Convert inputs to std::vector<arma::vec>
    std::vector<arma::vec> az_a = qd_python_list2vector_Col<double>(az);
    std::vector<arma::vec> el_a = qd_python_list2vector_Col<double>(el);
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);

    // Declare outputs
    arma::vec azimuth_spread, elevation_spread;
    arma::mat orientation;
    std::vector<arma::vec> phi, theta;

    // Call library function
    quadriga_lib::calc_angular_spreads_sphere<double>(az_a, el_a, powers_a,
                                                      &azimuth_spread, &elevation_spread, &orientation,
                                                      &phi, &theta, !wrapping, calc_bank_angle, quantize);

    // Convert to Python
    auto azimuth_spread_py = qd_python_copy2numpy(&azimuth_spread);
    auto elevation_spread_py = qd_python_copy2numpy(&elevation_spread);
    auto orientation_py = qd_python_copy2numpy(&orientation);
    auto phi_py = qd_python_copy2list(&phi);
    auto theta_py = qd_python_copy2list(&theta);

    return py::make_tuple(azimuth_spread_py, elevation_spread_py, orientation_py, phi_py, theta_py);
}

// pybind11 declaration:
// m.def("calc_angular_spread", &calc_angular_spread,
//       py::arg("az"),
//       py::arg("el"),
//       py::arg("powers"),
//       py::arg("wrapping") = false,
//       py::arg("calc_bank_angle") = false,
//       py::arg("quantize") = 0.0);