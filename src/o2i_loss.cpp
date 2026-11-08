// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# o2i_loss
O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3

- Draws for each UT one of the given penetration-loss models and evaluates its loss at each carrier frequency; the
  caller adds `loss` to the path loss of the UT
- Models of TR 38.901 v19.5.0: `low`, `high`, `low_A` (Table 7.4.3-2 with the materials of Table 7.4.3-1),
  `legacy_6GHz` (Table 7.4.3-3, single-frequency UMa and UMi below 6 GHz), `car` and `car_metal` (clause 7.4.3.2,
  metallized windows)
- Model of UT `u`: the smallest `j` with `rnd(u, 0) < w_0 + ... + w_j`, else the last model with a weight > 0; a model
  with weight 0 is never chosen
- Indoor distance: the minimum of columns 1 and 2 of `rnd` times `d2d_in_max` (`low`, `high`, `low_A`, clause
  7.4.3.1), column 1 times `d2d_in_max` (`legacy_6GHz`), 0 for the car models
- Draw `rnd` with [[sos_rand]] and [[sos_randn]] at the correlation distances of TR 38.901 clause 7.6.3.3: 50 m
  for column 0, 25 m for columns 1 and 2, 10 m for column 3
- TR applicability (Table 7.4.3-2): RMa `low` only, InF `high` only, UMa and UMi `low` and `high`, SMa `low`,
  `high` and `low_A`. SMa has 90 % residential and 10 % commercial buildings (Table 7.2-5) with different maximum
  indoor distances: call `o2i_loss` once per building type

## Declaration:
```
void quadriga_lib::o2i_loss(
    const arma::vec &center_frequency,
    const arma::mat &rnd,
    const std::vector<std::string> &models,
    const arma::vec &weights,
    arma::mat &loss,
    double d2d_in_max = 25.0,
    arma::vec *d2d_in = nullptr,
    arma::uvec *model_index = nullptr);
```

## Inputs:
- **`center_frequency`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`rnd`** — Random values per UT: columns 0 to 2 Uniform in [0, 1] (model choice, two indoor distances), column
  3 Normal(0,1), finite (deviation `sigma_P`, car loss); `[n_ut, 4]`, `n_ut = 0` allowed
- **`models`** — Models to mix: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal`; `[n_model]`
- **`weights`** — Probability of each model, finite and >= 0, sum 1 within 1e-6; `[n_model]`
- **`d2d_in_max`** *(optional)* — Maximum indoor 2D distance in [m], finite and >= 0 (key `O2I_d2D_in_max`; TR: 25 m
  for UMa, UMi and SMa commercial buildings, 10 m for RMa and SMa residential buildings)

## Outputs:
- **`loss`** — Penetration loss in [dB]; `[n_ut, n_freq]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m], 0 for the car models; `[n_ut]`
- **`model_index`** *(optional)* — Chosen model per UT, 0-based index into `models`; `[n_ut]`

## See also:
- [[los_probability]] (uses the outdoor distance `d2D - d2d_in`)
- [[path_loss]] (basic path loss, legacy outdoor distance from `d2d_in`)
MD!*/

void quadriga_lib::o2i_loss(const arma::vec &center_frequency,
                            const arma::mat &rnd,
                            const std::vector<std::string> &models,
                            const arma::vec &weights,
                            arma::mat &loss,
                            double d2d_in_max,
                            arma::vec *d2d_in,
                            arma::uvec *model_index)
{
    // Carrier frequencies
    qd_gbsm_check_fc(center_frequency);

    // Random values
    if (rnd.n_cols != 4)
        throw std::invalid_argument("Input 'rnd' must have 4 columns.");
    const arma::uword n_ut = rnd.n_rows, n_freq = center_frequency.n_elem;
    const double *pr = rnd.memptr(); // column c of UT u at pr[c · n_ut + u]
    for (arma::uword u = 0; u < 3 * n_ut; ++u)
        if (!(pr[u] >= 0.0 && pr[u] <= 1.0))
            throw std::invalid_argument("Uniform columns of 'rnd' must be in the range from 0 to 1.");
    for (arma::uword u = 3 * n_ut; u < 4 * n_ut; ++u)
        if (!std::isfinite(pr[u]))
            throw std::invalid_argument("Column 3 of 'rnd' must be finite.");

    // Models: 0 low, 1 high, 2 low_A, 3 legacy_6GHz, 4 car, 5 car_metal
    const char *names[6] = {"low", "high", "low_A", "legacy_6GHz", "car", "car_metal"};
    const arma::uword n_model = (arma::uword)models.size();
    if (n_model == 0)
        throw std::invalid_argument("Input 'models' cannot be empty.");
    std::vector<int> type(n_model, -1);
    for (arma::uword j = 0; j < n_model; ++j)
    {
        for (int t = 0; t < 6; ++t)
            if (models[j] == names[t])
                type[j] = t;
        if (type[j] < 0)
            throw std::invalid_argument("Model '" + models[j] + "' is not supported.");
    }

    // Weights
    if (weights.n_elem != n_model)
        throw std::invalid_argument("Number of elements in 'weights' does not match number of models.");
    double w_sum = 0.0;
    for (arma::uword j = 0; j < n_model; ++j)
    {
        if (!std::isfinite(weights[j]) || weights[j] < 0.0)
            throw std::invalid_argument("Input 'weights' must be finite and >= 0.");
        w_sum += weights[j];
    }
    if (std::abs(w_sum - 1.0) > 1e-6)
        throw std::invalid_argument("Input 'weights' must sum to 1.");

    if (!std::isfinite(d2d_in_max) || d2d_in_max < 0.0)
        throw std::invalid_argument("Input 'd2d_in_max' must be finite and >= 0.");
    qd_check_alias(&loss, "loss", &center_frequency, &rnd, &weights);
    qd_check_alias(d2d_in, "d2d_in", &center_frequency, &rnd, &weights);
    qd_check_alias_outputs(&loss, "loss", d2d_in, "d2d_in");

    // Cumulative weights and the fallback model (last non-zero weight)
    std::vector<double> c_cum(n_model);
    double acc = 0.0;
    arma::uword j_last = 0;
    for (arma::uword j = 0; j < n_model; ++j)
    {
        acc += weights[j];
        c_cum[j] = acc / w_sum;
        if (weights[j] > 0.0)
            j_last = j;
    }

    // Wall losses per model type and frequency, TR 38.901 Tables 7.4.3-1 and 7.4.3-2
    auto tw = [](double p1, double L1, double p2, double L2)
    { return 5.0 - 10.0 * std::log10(p1 * std::pow(10.0, -0.1 * L1) + p2 * std::pow(10.0, -0.1 * L2)); };
    arma::mat pl_tw(3, n_freq, arma::fill::none);
    for (arma::uword k = 0; k < n_freq; ++k)
    {
        const double f = center_frequency[k] * 1.0e-9;
        const double L_glass = 2.0 + 0.2 * f, L_irr = 25.4 + 0.11 * f, L_concrete = 5.0 + 4.0 * f,
                     L_plywood = 1.03 + 0.17 * f;
        pl_tw(0, k) = tw(0.3, L_glass, 0.7, L_concrete);
        pl_tw(1, k) = tw(0.7, L_irr, 0.3, L_concrete);
        pl_tw(2, k) = tw(0.3, L_glass, 0.7, L_plywood);
    }
    const double sigma_P[3] = {4.4, 6.5, 4.4};

    // Outputs
    if (loss.n_rows != n_ut || loss.n_cols != n_freq)
        loss.set_size(n_ut, n_freq);
    if (d2d_in != nullptr && d2d_in->n_elem != n_ut)
        d2d_in->set_size(n_ut);
    if (model_index != nullptr && model_index->n_elem != n_ut)
        model_index->set_size(n_ut);

    for (arma::uword u = 0; u < n_ut; ++u)
    {
        const double U0 = rnd(u, 0), U1 = rnd(u, 1), U2 = rnd(u, 2), N = rnd(u, 3);

        arma::uword j = j_last;
        for (arma::uword i = 0; i < n_model; ++i)
            if (U0 < c_cum[i])
            {
                j = i;
                break;
            }

        const int t = type[j];
        double din = 0.0;
        if (t <= 2) // low, high, low_A
        {
            din = std::min(U1, U2) * d2d_in_max;
            const double dev = 0.5 * din + sigma_P[t] * N;
            for (arma::uword k = 0; k < n_freq; ++k)
                loss(u, k) = pl_tw(t, k) + dev;
        }
        else if (t == 3) // legacy_6GHz
        {
            din = U1 * d2d_in_max;
            for (arma::uword k = 0; k < n_freq; ++k)
                loss(u, k) = 20.0 + 0.5 * din;
        }
        else // car, car_metal
        {
            const double val = (t == 4 ? 9.0 : 20.0) + 5.0 * N;
            for (arma::uword k = 0; k < n_freq; ++k)
                loss(u, k) = val;
        }

        if (d2d_in != nullptr)
            d2d_in->at(u) = din;
        if (model_index != nullptr)
            model_index->at(u) = j;
    }
}
