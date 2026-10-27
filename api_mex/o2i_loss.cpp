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
  frequency, from four normalized random values per UT; `f = fc / 1e9` in GHz, `lg = log10`
- Material losses of TR 38.901 v19.5.0 Table 7.4.3-1 in dB: `L_glass = 2 + 0.2 f`,
  `L_IRR = 25.4 + 0.11 f` (IRR glass, Release 19 value), `L_concrete = 5 + 4 f`,
  `L_plywood = 1.03 + 0.17 f`; external wall
  `tw(p1, L1, p2, L2) = 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))`
- Models, with `N = rnd(u, 4)` and the indoor 2D distance `d2din`:
  - `low` (low-loss, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_concrete) + 0.5·d2din + 4.4·N`
  - `high` (high-loss, Table 7.4.3-2): `tw(0.7, L_IRR, 0.3, L_concrete) + 0.5·d2din + 6.5·N`
  - `low_A` (low-loss A, Table 7.4.3-2): `tw(0.3, L_glass, 0.7, L_plywood) + 0.5·d2din + 4.4·N`
  - `legacy_6GHz` (backward compatible model for single-frequency simulations below 6 GHz,
    Table 7.4.3-3): `20 + 0.5·d2din`, no deviation; it goes with a shadow-fading std of 7 dB (the
    O2I conf files of UMa and UMi have `SF_sigma = 7`)
  - `car` (clause 7.4.3.2): `9 + 5·N`; `car_metal` (metallized car windows): `20 + 5·N`; no indoor
    distance
- Columns of `rnd`: 1 = Uniform(0,1) for the model choice, 2 and 3 = Uniform(0,1) for the indoor
  distance, 4 = Normal(0,1) for the deviation (sigma_P of Table 7.4.3-2, the car loss)
- Model choice per UT: `Ccum_j = (w_1 + ... + w_j)/sum(w)`; the model is the smallest `j` with
  `rnd(u, 1) < Ccum_j`, or, if there is none, the last `j` with `w_j > 0`. A model with weight 0 is
  never chosen
- Indoor distance: `low`, `high`, `low_A`: `d2din = min(rnd(u, 2), rnd(u, 3))·d2d_in_max`, the
  minimum of two uniforms of clause 7.4.3.1; `legacy_6GHz`: `d2din = rnd(u, 2)·d2d_in_max` (one
  uniform, Table 7.4.3-3); `car`, `car_metal`: 0
- The caller draws `rnd` with [[sos_rand]] and [[sos_randn]] at the correlation distances of
  TR 38.901 clause 7.6.3.3 (building type 50 m for column 1, d2D-in 25 m for columns 2 and 3,
  sigma_P 10 m for column 4) and adds `loss` to the path loss of the indoor UT. `d2d_in` gives the
  outdoor distance `d2D - d2D-in` for [[los_probability]] (call `o2i_loss` first) and is the input
  of the legacy outdoor distance of [[path_loss]]
- Applicability (Table 7.4.3-2): RMa uses `low` only, InF `high` only; UMa, UMi: `low` and `high`;
  SMa: `low`, `high` and `low_A`. TR maximum indoor distance: 25 m for UMa, UMi and SMa commercial
  buildings, 10 m for RMa and SMa residential buildings (conf-file field `O2I_d2D_in_max`). The TR
  SMa layout has 90 % residential and 10 % commercial buildings (Table 7.2-5): split the UTs by
  building type and call `o2i_loss` once per type
- Validation, in this order: `fc` non-empty, finite and > 0; `rnd` with 4 columns, columns 1 to 3 in
  [0, 1], column 4 finite; `models` non-empty, each a known name; `weights` with one finite element
  >= 0 per model and a sum of 1 within 1e-6; `d2d_in_max` finite and >= 0. The error messages count
  the columns of `rnd` from 0

## Usage:
```
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, d2d_in_max );
```

## Inputs:
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: Uniform(0,1) model choice, Uniform(0,1) d2D-in #1,
  Uniform(0,1) d2D-in #2, Normal(0,1) deviation; `[n_ut, 4]`
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; cell
  array of strings or one string; `[n_model]`
- **`weights`** — Probability of each model, >= 0, sum 1; `[n_model]`
- **`d2d_in_max`** — Maximum indoor 2D distance in [m], finite and >= 0 (conf-file field
  `O2I_d2D_in_max`); default: 25

## Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** — Indoor 2D distance in [m], 0 for the car models; `[n_ut, 1]`
- **`model_index`** — Chosen model per UT, 1-based index into `models`; uint64; `[n_ut, 1]`

## See also:
- [[los_probability]] (uses the outdoor distance `d2D - d2d_in`)
- [[path_loss]] (basic path loss, legacy outdoor distance from `d2d_in`)
- [[sos_rand]], [[sos_randn]] (spatially consistent random values for `rnd`)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::vec fc = qd_mex_get_Col<double>(prhs[0]);
    const arma::mat rnd = qd_mex_get_Mat<double>(prhs[1]);
    const std::vector<std::string> models = qd_mex_get_strings(prhs[2], "models");
    const arma::vec weights = qd_mex_get_Col<double>(prhs[3]);
    const double d2d_in_max = (nrhs < 5) ? 25.0 : qd_mex_get_scalar<double>(prhs[4], "d2d_in_max", 25.0);

    // Output allocation
    arma::mat loss;
    arma::vec d2d_in;
    arma::uvec model_index;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&loss, rnd.n_rows, fc.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&d2d_in, rnd.n_rows);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&model_index, rnd.n_rows);

    // Wrap optional pointers
    arma::vec *p_d2d_in = (nlhs > 1) ? &d2d_in : nullptr;
    arma::uvec *p_model_index = (nlhs > 2) ? &model_index : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::o2i_loss(fc, rnd, models, weights, loss, d2d_in_max, p_d2d_in, p_model_index));

    // Model index: 0-based (C++) -> 1-based (MATLAB)
    if (nlhs > 2)
        model_index += 1;
}
