// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio
- Inputs are lists so each CIR can have a different number of paths
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis via the Jones matrix
  transform `M_circ = T · M_lin · T^-1`
- `M` is stored with interleaved real/imaginary parts, 8 rows per path:
  `[Re(M_vv); Im(M_vv); Re(M_vh); Im(M_vh); Re(M_hv); Im(M_hv); Re(M_hh); Im(M_hh)]`
- LOS paths are identified by comparing path length against the direct TX-RX distance `dTR`; paths
  with `path_length < dTR + window_size` are excluded by default
- Normalization of `M` does not affect XPR (cancels in the ratio) but does affect `pg`
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR
  is set to 0
- TX/RX positions may be fixed `(3, 1)` or mobile `(3, n_cir)`

## Usage:
```
xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio( powers, M, path_length, tx_pos, rx_pos, \
    include_los, window_size )
```

## Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`M`** — Polarization transfer matrices with interleaved real/imag parts per CIR; list of length
  `n_cir`, each element a 2D array of shape `(8, n_path)`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`include_los`** — If True, includes LOS and near-LOS paths in the XPR calculation; default: False
- **`window_size`** — LOS exclusion window; paths within `dTR + window_size` are excluded when
  `include_los = False`; default: 0.01

## Outputs:
- **`xpr`** — XPR on linear scale; `(n_cir, 6)`; columns (0-based):<br><br>
   | Col | Description                                                     |
   | :-: | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_hv)^2)                      |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_vh)^2)                      |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_RL)^2)                   |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_LR)^2)                   |
- **`pg`** — Total path gain summed over all paths (including LOS) as
  `0.5 · sum(powers · (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `(n_cir,)`
MD!*/

py::tuple calc_cross_polarization_ratio(py::list powers,
                                        py::list M,
                                        py::list path_length,
                                        const py::array_t<double> &tx_pos,
                                        const py::array_t<double> &rx_pos,
                                        bool include_los,
                                        double window_size)
{
    // Convert inputs
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);
    std::vector<arma::mat> M_a = qd_python_list2vector_Mat<double>(M);
    std::vector<arma::vec> path_length_a = qd_python_list2vector_Col<double>(path_length);
    arma::mat tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    arma::mat rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);

    // Declare outputs
    arma::mat xpr;
    arma::vec pg;

    // Call library
    quadriga_lib::calc_cross_polarization_ratio<double>(powers_a, M_a, path_length_a, tx_pos_a, rx_pos_a,
                                                        &xpr, &pg, include_los, window_size);

    // Convert outputs
    auto xpr_py = qd_python_copy2numpy(&xpr);
    auto pg_py = qd_python_copy2numpy(&pg);

    return py::make_tuple(xpr_py, pg_py);
}

// pybind11 declaration:
// m.def("calc_cross_polarization_ratio", &calc_cross_polarization_ratio,
//       py::arg("powers"),
//       py::arg("M"),
//       py::arg("path_length"),
//       py::arg("tx_pos"),
//       py::arg("rx_pos"),
//       py::arg("include_los") = false,
//       py::arg("window_size") = 0.01);
