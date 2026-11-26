// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# acdf
Empirical cumulative distribution functions (CDF) of several data sets, their average and quantile statistics

- The CDF of each data set (column of `data`) at the bin centers: each finite value counts for the nearest bin (values
  outside the bins for the first or last one), and the CDF is the cumulative count divided by the number of finite
  values of the set
- Average `cdf_avg` in quantile space: for the probabilities $P = 0, 0.001, \dots, 0.999$, the bins at which the
  CDFs of the sets reach $P$ are averaged over the sets, and the averaged quantile function is mapped back to the
  bins; this averages the position of the distributions instead of their CDF values
- `mu` and `sig`: mean and standard deviation (normalized by $N - 1$) of the quantiles at
  $P = 0.1, 0.2, \dots, 0.9$ over the sets, each taken over the 11 probabilities within $\pm 0.005$; for one set its
  quantiles and 0
- Non-finite values are left out; a set without finite values gets a NaN CDF and is left out of `cdf_avg`, `mu` and
  `sig`, which are NaN if no set has finite values (possible only with given bins)
- Generated bins: `n_bins` equally spaced bins from the smallest to the largest finite value of `data` (±10 % of the
  value, or ±1 for 0, if all are equal); `data` without finite values then throws

## Usage:
```
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, *, n_bins )
```

## Inputs:
- **`data`** — Data sets, one per column, magnitudes below about 1e18; `(n_samples, n_sets)`
- **`bins`** — Bin centers, finite and strictly ascending; `(n_bins_in,)`; default: `None` (generated bins)
- **`n_bins`** — Number of generated bins, >= 2; not used with `bins`; default: 201

## Outputs:
- **`cdf_per_set`** — CDF of each set; `(n_bins_out, n_sets)`
- **`bins_out`** — Bin centers, `bins` or the generated ones; `(n_bins_out,)`
- **`cdf_avg`** — Average CDF; `(n_bins_out,)`
- **`mu`** — Mean of the quantiles at 0.1, ..., 0.9; `(9,)`
- **`sig`** — Standard deviation of these quantiles; `(9,)`

## See also:
- [[calc_delay_spread]] (delay spreads of many CIRs, a typical input)
MD!*/

py::tuple acdf(const py::array_t<double> &data,
               py::handle bins,
               arma::uword n_bins)
{
    const auto data_a = qd_python_numpy2arma_Mat<double>(data, true);
    const auto bins_in_a = qd_python_numpy2arma_Col<double>(bins, true);

    arma::uword n_bins_out = bins_in_a.empty() ? n_bins : bins_in_a.n_elem;
    arma::uword n_sets = data_a.n_cols;

    arma::mat cdf_per_set;
    arma::vec cdf_avg, bins_out, mu, sig;

    auto cdf_per_set_py = qd_python_init_output(n_bins_out, n_sets, &cdf_per_set);
    auto cdf_avg_py = qd_python_init_output(n_bins_out, &cdf_avg);
    auto mu_py = qd_python_init_output(9, &mu);
    auto sig_py = qd_python_init_output(9, &sig);

    if (!bins_in_a.empty())
        bins_out = bins_in_a;

    quadriga_lib::acdf<double>(data_a, &bins_out, &cdf_per_set, &cdf_avg, &mu, &sig, n_bins);

    auto bins_out_py = qd_python_copy2numpy(&bins_out);

    return py::make_tuple(cdf_per_set_py, bins_out_py, cdf_avg_py, mu_py, sig_py);
}
