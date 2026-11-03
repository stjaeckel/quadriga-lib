// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1), `legacy_6GHz`
  (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz; the O2I conf files of UMa and UMi have its `SF_sigma = 7`),
  `car` and `car_metal` (clause 7.4.3.2, metallized windows)
- Model of UT `u`: the first model `j` with `rnd(u, 0)` below the sum of the weights up to `j`, else the last model with
  a weight > 0; a model with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause 7.4.3.1),
  column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [[sos.rand]] and [[sos.randn]] at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m for
  column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa all three. SMa
  has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum indoor distances: call
  `o2i_loss` once per building type

## Usage:
```
loss, d2d_in, model_index = quadriga_lib.gbsm.o2i_loss( fc, rnd, models, weights, d2d_in_max )
```

## Inputs:
- **`fc`** — Carrier frequencies in Hz, finite and > 0; `(n_freq,)`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column 3
  Normal(0,1), finite (deviation `sigma_P`, car loss); `(n_ut, 4)`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; list of str or one str;
  `n_model` entries
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `(n_model,)`
- **`d2d_in_max`** — Maximum indoor 2D distance in m, finite and >= 0 (key `O2I_d2D_in_max`; TR: 25 m for UMa, UMi and
  SMa commercial buildings, 10 m for RMa and SMa residential buildings); default: 25.0

## Outputs:
- **`loss`** — Penetration loss in dB; `(n_ut, n_freq)`
- **`d2d_in`** — Indoor 2D distance in m, 0 for the car models; `(n_ut,)`
- **`model_index`** — Chosen model per UT, 0-based index into `models`; uint64; `(n_ut,)`

## See also:
- [[los_probability]] (uses the outdoor distance `d2D - d2d_in`)
- [[path_loss]] (basic path loss, legacy outdoor distance from `d2d_in`)
MD!*/

py::tuple o2i_loss(const py::array_t<double> &fc,
                   const py::array_t<double> &rnd,
                   py::handle models,
                   const py::array_t<double> &weights,
                   double d2d_in_max)
{
    // Read inputs
    const auto fc_a = qd_python_numpy2arma_Col<double>(fc, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto models_a = qd_python_list2vector_Strings(models);
    const auto weights_a = qd_python_numpy2arma_Col<double>(weights, true);

    // Output allocation
    arma::mat loss;
    arma::vec d2d_in;
    arma::uvec model_index;
    auto loss_py = qd_python_init_output(rnd_a.n_rows, fc_a.n_elem, &loss);
    auto d2d_in_py = qd_python_init_output(rnd_a.n_rows, &d2d_in);
    auto model_index_py = qd_python_init_output(rnd_a.n_rows, &model_index);

    // Call library function
    quadriga_lib::o2i_loss(fc_a, rnd_a, models_a, weights_a, loss, d2d_in_max, &d2d_in, &model_index);

    return py::make_tuple(loss_py, d2d_in_py, model_index_py);
}

// pybind11 declaration:
// m.def("o2i_loss", &o2i_loss,
//       py::arg("fc").none(false),
//       py::arg("rnd").none(false),
//       py::arg("models"),
//       py::arg("weights").none(false),
//       py::arg("d2d_in_max") = 25.0);
