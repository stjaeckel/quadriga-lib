// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at
  random positions
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin,
  evaluates the generator there, and for each distance bin computes the correlation coefficient
  between the two values of all position pairs whose distance falls into the bin
- Positions: an array `(2·n_pos, 3)` is filled column by column (all x, then all y, then all z)
  with `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in
  [[sos_init_phases]]
- Bin `n` covers the distances from the midpoint between `bins[n-1]` and `bins[n]` (excluded) to
  the midpoint between `bins[n]` and `bins[n+1]` (included). The first bin starts at 1e-5 m, the
  last bin ends half a mean bin spacing above `bins[-1]`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the
  first device, the second `n_pos` positions the second device, and the value of a link is
  [[sos_val]] with both positions. Two links form a pair if the distance between their first
  devices falls into the bin and the distance between their second devices does not exceed the
  upper limit of the bin; requires `acf` and `dist`
- `acf_est_dual` is computed if `acf` or `dist` is given, and is `None` if both are omitted; giving
  only one of the two is an error
- The estimate describes one realisation; it scatters around the ACF of the sinusoids

## Usage:
```
acf_est, dist_est, acf_est_dual = quadriga_lib.sos.acf_estimate( freq, phase, bins, n_pos, \
    acf, dist, seed )
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `(n_coeff, 3)`
- **`phase`** — Phases of the generator in [rad]; `(n_coeff,)` or `(n_coeff, 1)`
- **`bins`** — Bin centers; must be >= 0 and increase strictly; at least 2 elements; `(n_bin,)`
- **`n_pos`** — Number of random positions per device; `int`; must be > 1
- **`acf`** — Target ACF; required for `acf_est_dual`; `(n_acf,)`; default: `None`
- **`dist`** — Sample points of the ACF; must start at 0 and increase strictly; `(n_acf,)`;
  default: `None`
- **`seed`** — Seed for the positions; `int`; must be >= -1; -1 seeds from the system random device;
  default: -1

## Outputs:
- **`acf_est`** — Estimated ACF; `(n_bin,)`
- **`dist_est`** — Mean distance of the pairs in each bin; `(n_bin,)`
- **`acf_est_dual`** — Estimated ACF for two devices on the same map; `(n_bin,)`; `None` if `acf`
  and `dist` were not given

## See also:
- [[sos_acf_approx]] (ACF of the sinusoids in closed form)
- [[sos_val]] (evaluation of the generator)
MD!*/

py::tuple sos_acf_estimate(const py::array_t<double> &freq,
                           const py::array_t<double> &phase,
                           const py::array_t<double> &bins,
                           arma::uword n_pos,
                           py::handle acf,
                           py::handle dist,
                           arma::sword seed)
{
    // Read inputs
    const auto freq_a = qd_python_numpy2arma_Mat<double>(freq, true);
    const auto phase_a = qd_python_numpy2arma_Col<double>(phase, true);
    const auto bins_a = qd_python_numpy2arma_Col<double>(bins, true);
    const auto acf_a = qd_python_numpy2arma_Col<double>(acf, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);

    // Wrap optional pointers
    const arma::vec *p_acf = acf_a.is_empty() ? nullptr : &acf_a;
    const arma::vec *p_dist = dist_a.is_empty() ? nullptr : &dist_a;

    // Output allocation; the two-device estimate is requested only with an ACF table
    arma::vec acf_est, dist_est, acf_est_dual;
    auto acf_est_py = qd_python_init_output(bins_a.n_elem, &acf_est);
    auto dist_est_py = qd_python_init_output(bins_a.n_elem, &dist_est);

    py::object acf_est_dual_py = py::none();
    arma::vec *p_acf_est_dual = nullptr;
    if (p_acf != nullptr || p_dist != nullptr)
    {
        acf_est_dual_py = qd_python_init_output(bins_a.n_elem, &acf_est_dual);
        p_acf_est_dual = &acf_est_dual;
    }

    // Call library function
    quadriga_lib::sos_acf_estimate<double>(freq_a, phase_a, bins_a, n_pos, acf_est, &dist_est, p_acf_est_dual, p_acf, p_dist, seed);

    return py::make_tuple(acf_est_py, dist_est_py, acf_est_dual_py);
}

// pybind11 declaration:
// m.def("acf_estimate", &sos_acf_estimate,
//       py::arg("freq"),
//       py::arg("phase"),
//       py::arg("bins"),
//       py::arg("n_pos"),
//       py::arg("acf") = py::none(),
//       py::arg("dist") = py::none(),
//       py::arg("seed") = -1);
