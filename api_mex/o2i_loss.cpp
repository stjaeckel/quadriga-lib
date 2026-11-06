// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# O2I_LOSS
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier
  frequency; the caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table
  7.4.3-1), `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and
  `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 1)` below the sum of the weights up to `j`, else
  the last model with a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 2 and 3 of `rnd` times `d2d_in_max` (`low`, `high`,
  `low_A`, clause 7.4.3.1), column 2 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [[sos_rand]] and [[sos_randn]] at the correlation distances of TR 38.901 clause
  7.6.3.3: 50 m for column 1, 25 m for columns 2 and 3, 10 m for column 4
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`,
  SMa `low`, `high` and `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5)
  with different maximum indoor distances: call `o2i_loss` once per building type

## Usage:
```
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( center_freq, rnd, models, weights, d2d_in_max );
```

## Inputs:
- **`center_freq`** — Carrier frequencies in [Hz], finite and > 0 (messages name it
  `center_frequency`); `[n_freq]`
- **`rnd`** — Random values per UT: columns 1 to 3 Uniform in [0, 1] (model choice, two indoor
  distances), column 4 Normal(0,1), finite (deviation `sigma_P`, car loss); `[n_ut, 4]`, `n_ut = 0`
  allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; cell
  array of strings or one string; `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `[n_model]`
- **`d2d_in_max`** — Maximum indoor 2D distance in [m], finite and >= 0 (field `O2I_d2D_in_max`; TR:
  25 m for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings);
  default: 25

## Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** — Indoor 2D distance in [m], 0 for the car models; `[n_ut, 1]`
- **`model_index`** — Chosen model per UT, 1-based index into `models`; uint64; `[n_ut, 1]`

## See also:
- [[los_probability]] (uses the outdoor distance `d2D - d2d_in`)
- [[path_loss]] (basic path loss, legacy outdoor distance from `d2d_in`)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec center_freq = qd_mex_get_Col<double>(prhs[0]);
    const arma::mat rnd = qd_mex_get_Mat<double>(prhs[1]);
    const std::vector<std::string> models = qd_mex_get_strings(prhs[2], "models");
    const arma::vec weights = qd_mex_get_Col<double>(prhs[3]);
    const double d2d_in_max = (nrhs < 5) ? 25.0 : qd_mex_get_scalar<double>(prhs[4], "d2d_in_max", 25.0);

    // Output allocation
    arma::mat loss;
    arma::vec d2d_in;
    arma::uvec model_index;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&loss, rnd.n_rows, center_freq.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&d2d_in, rnd.n_rows);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&model_index, rnd.n_rows);

    // Wrap optional pointers
    arma::vec *p_d2d_in = (nlhs > 1) ? &d2d_in : nullptr;
    arma::uvec *p_model_index = (nlhs > 2) ? &model_index : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::o2i_loss(center_freq, rnd, models, weights, loss, d2d_in_max, p_d2d_in, p_model_index));

    // Model index: 0-based (C++) -> 1-based (MATLAB)
    if (nlhs > 2)
        model_index += 1;
}
