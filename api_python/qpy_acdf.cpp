// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

## Description:
- Calculates the empirical CDF from the given data matrix, where each column represents an
  independent data set (e.g., repeated experiment runs).
- Individual CDFs are computed per column and an averaged CDF is obtained by interpolation in
  quantile space.
- `Inf` and `NaN` values in the data are excluded from the computation.
- If `bins` is `None` or empty, 201 equally spaced bins spanning the data range are generated.

## Usage:
```
import quadriga_lib
result = quadriga_lib.tools.acdf( data, bins, n_bins )
```

## Arguments:
- `np.ndarray **data**` (input)<br>
  Input data matrix. Shape `(n_samples, n_sets)`. Each column is one data set.

- `np.ndarray **bins** = None` (optional input)<br>
  Bin centers for the histogram. Shape `(n_bins,)`. If `None` or empty, bins are auto-generated.

- `int **n_bins** = 201` (input)<br>
  Number of bins to generate when bins are auto-generated. Must be at least 2.

## Returns:
- `dict` with keys:<br>
  `"bins"` — Bin centers, shape `(n_bins,)`.<br>
  `"Sh"` — Individual CDFs, shape `(n_bins, n_sets)`.<br>
  `"Sc"` — Averaged CDF, shape `(n_bins,)`.<br>
  `"mu"` — Mean quantiles at 0.1–0.9, shape `(9,)`.<br>
  `"sig"` — Std of quantiles at 0.1–0.9, shape `(9,)`.

## Example:
```
import numpy as np
import quadriga_lib

data = np.random.randn(10000, 5)
result = quadriga_lib.tools.acdf(data)
bins = result["bins"]   # 201 bin centers
Sh = result["Sh"]       # Individual CDFs, shape (201, 5)
Sc = result["Sc"]       # Averaged CDF, shape (201,)
mu = result["mu"]       # Mean quantiles, shape (9,)
sig = result["sig"]     # Std of quantiles, shape (9,)
```
MD!*/

py::dict acdf(const py::array_t<double> &data_py,
              py::object bins_py,
              arma::uword n_bins)
{
    // Convert input data
    arma::mat data = qd_python_numpy2arma_Mat<double>(data_py);

    // Handle bins input
    arma::vec bins;
    arma::vec *p_bins = &bins;

    if (!bins_py.is_none())
    {
        py::array_t<double> bins_arr = bins_py.cast<py::array_t<double>>();
        if (bins_arr.size() > 0)
            bins = qd_python_numpy2arma_Col<double>(bins_arr);
    }

    // Declare outputs
    arma::mat Sh;
    arma::vec Sc, mu, sig;

    // Call library
    quadriga_lib::acdf<double>(data, p_bins, &Sh, &Sc, &mu, &sig, n_bins);

    // Build output dict
    py::dict output;
    output["bins"] = qd_python_copy2numpy(bins);
    output["Sh"] = qd_python_copy2numpy(Sh);
    output["Sc"] = qd_python_copy2numpy(Sc);
    output["mu"] = qd_python_copy2numpy(mu);
    output["sig"] = qd_python_copy2numpy(sig);

    return output;
}

// pybind11 declaration:
// m.def("acdf", &acdf,
//       py::arg("data"),
//       py::arg("bins") = py::none(),
//       py::arg("n_bins") = 201);
