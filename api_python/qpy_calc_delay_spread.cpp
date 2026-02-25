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
# calc_delay_spread
Calculate the RMS delay spread in [s]

## Description:
- Computes the root-mean-square (RMS) delay spread from a given set of delays and corresponding
  linear-scale powers for each channel impulse response (CIR).
- An optional power threshold in [dB] relative to the strongest path can be applied. Paths with
  power below `p_max(dB) - threshold` are excluded from the calculation.
- An optional granularity parameter in [s] groups paths in the delay domain. Powers of paths
  falling into the same delay bin are summed before computing the delay spread.
- Optionally returns the mean delay for each CIR.

## Usage:
```
import quadriga_lib
ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers, threshold=100.0, granularity=0.0)
```

## Arguments:
- `list of np.ndarray **delays**` (input)<br>
  Delays in [s]. A list of length `n_cir`, where each element is a 1D numpy array of path delays.

- `list of np.ndarray **powers**` (input)<br>
  Path powers on a linear scale [W]. Same structure as `delays`.

- `float **threshold** = 100.0` (input)<br>
  Power threshold in [dB] relative to the strongest path. Default: 100 dB.

- `float **granularity** = 0.0` (input)<br>
  Window size in [s] for grouping paths in the delay domain. Default: 0 (no grouping).

## Returns:
- `np.ndarray **ds**` (output)<br>
  RMS delay spread in [s] for each CIR. Shape `(n_cir,)`.

- `np.ndarray **mean_delay**` (output)<br>
  Mean delay in [s] for each CIR. Shape `(n_cir,)`.

## Example:
```
import numpy as np
import quadriga_lib

delays = [np.array([0.0, 1e-6, 2e-6])]
powers = [np.array([1.0, 0.5, 0.25])]
ds, mean_delay = quadriga_lib.calc_delay_spread(delays, powers)
```
MD!*/

py::tuple calc_delay_spread(py::list delays_py, py::list powers_py,
                            double threshold, double granularity)
{
    // Convert Python lists to std::vector<arma::vec>
    std::vector<arma::vec> delays = qd_python_list2vector_Col<double>(delays_py);
    std::vector<arma::vec> powers = qd_python_list2vector_Col<double>(powers_py);

    // Declare outputs
    arma::vec mean_delay;

    // Call C++ library (always pass all output pointers)
    arma::vec ds = quadriga_lib::calc_delay_spread<double>(delays, powers, threshold, granularity, &mean_delay);

    // Convert to Python types
    auto ds_p = qd_python_copy2numpy(ds);
    auto mean_delay_p = qd_python_copy2numpy(mean_delay);

    return py::make_tuple(ds_p, mean_delay_p);
}
