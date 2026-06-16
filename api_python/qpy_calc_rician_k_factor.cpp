// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

- Computes the Rician K-Factor (KF): ratio of LOS power to NLOS power per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- LOS paths are those with length ≤ `dTR + window_size`, where `dTR` is the direct TX-RX distance;
  remaining paths are NLOS
- If total NLOS power is zero, KF is set to infinity; if total LOS power is zero, KF is set to 0
- TX/RX positions may be fixed `(3,)` / `(3, 1)` (reused for all CIRs) or mobile `(3, n_cir)`

## Usage:
```
kf, pg = quadriga_lib.tools.calc_rician_k_factor( powers, path_length, tx_pos, rx_pos, window_size )
```

## Inputs:
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`path_length`** — Absolute TX-to-RX path lengths per CIR; list of length `n_cir`, each element
  a 1D array of length `n_path` matching `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`rx_pos`** — Receiver position [x; y; z]; `(3,)` or `(3, 1)` (fixed) or `(3, n_cir)` (mobile)
- **`window_size`** — LOS window; paths with length ≤ `dTR + window_size` are treated as LOS;
  default: 0.01

## Outputs:
- **`kf`** — Rician K-Factor on linear scale; `(n_cir,)`
- **`pg`** — Total path gain (sum of all path powers) per CIR; `(n_cir,)`
MD!*/

py::tuple calc_rician_k_factor(py::list powers,
                               py::list path_length,
                               const py::array_t<double> &tx_pos,
                               const py::array_t<double> &rx_pos,
                               double window_size)
{
    // Convert Python inputs to C++ types
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);
    std::vector<arma::vec> path_length_a = qd_python_list2vector_Col<double>(path_length);
    arma::mat tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    arma::mat rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);

    // Declare outputs
    arma::vec kf, pg;

    // Call C++ library (always compute all outputs for Python)
    quadriga_lib::calc_rician_k_factor<double>(powers_a, path_length_a, tx_pos_a, rx_pos_a, &kf, &pg, window_size);

    auto kf_py = qd_python_copy2numpy(&kf);
    auto pg_py = qd_python_copy2numpy(&pg);

    return py::make_tuple(kf_py, pg_py);
}

// pybind11 declaration:
// m.def("calc_rician_k_factor", &calc_rician_k_factor,
//       py::arg("powers"),
//       py::arg("path_length"),
//       py::arg("tx_pos"),
//       py::arg("rx_pos"),
//       py::arg("window_size") = 0.01);