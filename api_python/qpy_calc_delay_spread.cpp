// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Computes RMS delay spread from a set of delays and corresponding linear-scale powers per CIR
- Inputs are lists of 1D arrays so each CIR can have a different number of paths
- Paths with power below `p_max / 10^(0.1 · threshold)` are excluded; the default threshold of
  100 dB effectively includes all paths
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their
  powers summed before the spread is computed

## Usage:
```
ds, mean_delay = quadriga_lib.tools.calc_delay_spread( delays, powers, threshold, granularity )
```

## Inputs:
- **`delays`** — Delays per CIR; list of length `n_cir`, each element a 1D array of length `n_path`
- **`powers`** — Path powers on linear scale per CIR; list of length `n_cir`, each element a 1D
  array of length `n_path`
- **`threshold`** — Power threshold in [dB] relative to the strongest path; paths below threshold
  are excluded; default: 100.0
- **`granularity`** — Bin width in [s] for grouping paths in the delay domain; default: 0.0 (no grouping)

## Outputs:
- **`ds`** — RMS delay spread per CIR; `(n_cir,)`
- **`mean_delay`** — Mean delay per CIR; `(n_cir,)`
MD!*/

py::tuple calc_delay_spread(py::list delays,
                            py::list powers,
                            double threshold,
                            double granularity)
{
    // Convert Python lists to std::vector<arma::vec>
    std::vector<arma::vec> delays_a = qd_python_list2vector_Col<double>(delays);
    std::vector<arma::vec> powers_a = qd_python_list2vector_Col<double>(powers);

    // Declare outputs
    arma::vec mean_delay;

    // ds is the return value, mean_delay is the out-pointer
    arma::vec ds = quadriga_lib::calc_delay_spread<double>(delays_a, powers_a, threshold, granularity, &mean_delay);

    auto ds_py = qd_python_copy2numpy(&ds);
    auto mean_delay_py = qd_python_copy2numpy(&mean_delay);

    return py::make_tuple(ds_py, mean_delay_py);
}

// pybind11 declaration:
// m.def("calc_delay_spread", &calc_delay_spread,
//       py::arg("delays"),
//       py::arg("powers"),
//       py::arg("threshold") = 100.0,
//       py::arg("granularity") = 0.0);