// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
    // Model identifiers
    enum qd_los_model
    {
        QD_LOS_RMA,
        QD_LOS_UMI,
        QD_LOS_UMA,
        QD_LOS_INH_MIXED,
        QD_LOS_INH_OPEN,
        QD_LOS_INF_SL,
        QD_LOS_INF_DL,
        QD_LOS_INF_SH,
        QD_LOS_INF_DH,
        QD_LOS_INF_HH,
        QD_LOS_SMA,
        QD_LOS_DENSEURBAN,
        QD_LOS_URBAN,
        QD_LOS_SUBURBAN,
        QD_LOS_RURAL,
        QD_LOS_INDUSTRIAL,
        QD_LOS_LOS,
        QD_LOS_NLOS,
        QD_LOS_N_MODEL
    };

    const char *qd_los_names[QD_LOS_N_MODEL] = {
        "38.901_RMa", "38.901_UMi", "38.901_UMa", "38.901_InH_Mixed", "38.901_InH_Open", "38.901_InF_SL",
        "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH", "38.901_SMa", "38.811_DenseUrban",
        "38.811_Urban", "38.811_Suburban", "38.811_Rural", "QuaDRiGa_Industrial", "LOS", "NLOS"};

    // LOS probability of TR 38.811 Table 6.6.1-1 in percent over the elevation in degrees
    const double qd_los_811_el[11] = {-90.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};
    const double qd_los_811_dense_urban[11] = {0.0, 0.0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1};
    const double qd_los_811_urban[11] = {0.0, 0.0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2};
    const double qd_los_811_suburban_rural[11] = {0.0, 0.0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8};

    // Inverse of the standard normal CDF Phi(x) = 0.5·erfc(-x/sqrt(2))
    // - p <= 0: -Inf; p >= 1: +Inf; p > 0.5: -F(1 - p) (1 - p is exact for p >= 0.5)
    // - p in (0, 0.5]: Acklam's rational approximation, followed by one Halley step on the lower-tail CDF
    // - Accuracy: below 1e-10·max(1, abs(F)) for p in [1e-12, 1 - 1e-12]
    double qd_los_probit(double p)
    {
        if (p <= 0.0)
            return -std::numeric_limits<double>::infinity();
        if (p >= 1.0)
            return std::numeric_limits<double>::infinity();
        if (p > 0.5)
            return -qd_los_probit(1.0 - p);

        const double a0 = -3.969683028665376e+01, a1 = 2.209460984245205e+02, a2 = -2.759285104469687e+02,
                     a3 = 1.383577518672690e+02, a4 = -3.066479806614716e+01, a5 = 2.506628277459239e+00;
        const double b0 = -5.447609879822406e+01, b1 = 1.615858368580409e+02, b2 = -1.556989798598866e+02,
                     b3 = 6.680131188771972e+01, b4 = -1.328068155288572e+01;
        const double c0 = -7.784894002430293e-03, c1 = -3.223964580411365e-01, c2 = -2.400758277161838e+00,
                     c3 = -2.549732539343734e+00, c4 = 4.374664141464968e+00, c5 = 2.938163982698783e+00;
        const double d0 = 7.784695709041462e-03, d1 = 3.224671290700398e-01, d2 = 2.445134137142996e+00,
                     d3 = 3.754408661907416e+00;
        const double p_low = 0.02425;

        double x;
        if (p < p_low)
        {
            const double q = std::sqrt(-2.0 * std::log(p));
            x = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
        }
        else
        {
            const double q = p - 0.5, r = q * q;
            x = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q /
                (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0);
        }

        // One Halley step; skipped where exp(x²/2) overflows (p below about 1e-308)
        const double sqrt2 = 1.4142135623730951, sqrt2pi = 2.5066282746310002;
        const double e = 0.5 * std::erfc(-x / sqrt2) - p;
        const double u = e * sqrt2pi * std::exp(0.5 * x * x);
        if (std::isfinite(u))
            x = x - u / (1.0 + 0.5 * x * u);
        return x;
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# los_probability
LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models

- Evaluates the probability that a link is in line-of-sight (LOS) for each link, from its 2D distance and the BS and
  UT heights, and optionally the soft LOS state of TR 38.901 eq. (7.6-18)
- Notation: `d = d2d_out(l)`, `hB = h_bs(l)`, `hU = h_ut(l)` in m, `lg = log10`, `ln` the natural logarithm. All
  results are limited to [0, 1]
- TR 38.901 v19.5.0 Table 7.4.2-1:
  - `38.901_RMa`: 1 for `d <= 10`, else `exp(-(d - 10)/1000)`
  - `38.901_UMi`: 1 for `d <= 18`, else `18/d + exp(-d/36)·(1 - 18/d)`
  - `38.901_UMa`: 1 for `d <= 18`, else `(18/d + exp(-d/63)·(1 - 18/d))·(1 + C'·5/4·(d/100)³·exp(-d/150))`
    with `C' = 0` for `hU <= 13`, `((hU - 13)/10)^1.5` for `13 < hU <= 23` and 1 for `hU > 23`; for `hU > 13`
    the formula exceeds 1 slightly just above 18 m (e.g. up to 1.0060 for `hU = 22.5`), where the limit gives 1
  - `38.901_InH_Mixed` (mixed office): 1 for `d <= 1.2`, `exp(-(d - 1.2)/4.7)` for `1.2 < d < 6.5`,
    `exp(-(d - 6.5)/32.6)·0.32` for `d >= 6.5`
  - `38.901_InH_Open` (open office): 1 for `d <= 5`, `exp(-(d - 5)/70.8)` for `5 < d <= 49`,
    `exp(-(d - 49)/211.7)·0.54` for `d > 49`
  - `38.901_InF_SL`, `38.901_InF_DL`: `exp(-d·kinv)` with `kinv = -ln(1 - r)/d_clutter`
  - `38.901_InF_SH`, `38.901_InF_DH`: with `kinv = -ln(1 - r)/d_clutter`: 1 for `hU >= h_c` (UT above the
    clutter); `exp(-d·kinv·(h_c - hU)/(hB - hU))` for `hB > hU` (TR: `k = -d_clutter/ln(1 - r)·(hB - hU)/(h_c - hU)`);
    `exp(-d·kinv)` (the rule of SL and DL) for `hB <= hU < h_c`, where the TR formula has a BS above the UT and
    would give `k <= 0`. `r = 0` gives 1
  - `38.901_InF_HH`: 1
  - `38.901_SMa` (TR v19): 1 for `d <= 10` or `hB <= hU`, else `exp(-d·(k_com + k_res + k_veg))` with
    `k_x = -ln(1 - r_x)/d_clutter·max(h_x - hU, 0)/(hB - hU)`, heights `h_com = 20`, `h_res = 8`, `h_veg = 15` m,
    densities `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter`; the TR writes `max(., 0)` only for the
    residential term, which gives the same values for `hU < 15` m (the TR range)
- Defaults of the clutter arguments (NaN), TR 38.901 Table 7.2-4 and Table 7.8-7 for InF, Table 7.4.2-1 for SMa:
  - `InF_SL`: `r = 0.2`, `d_clutter = 10` m; `InF_DL`: `r = 0.6`, `d_clutter = 2` m
  - `InF_SH`: `r = 0.2`, `d_clutter = 10` m, `h_c = 2` m; `InF_DH`: `r = 0.6`, `d_clutter = 2` m, `h_c = 6` m
  - `SMa`: `d_clutter = 30` m, `r_veg = 0` (no vegetation; 0.1 sparse, 0.2 dense vegetation); `h_clutter` is not used
  - The other models ignore the clutter arguments (they are validated for every model)
- TR 38.811 (via QuaDRiGa), `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`: the LOS
  probability in percent over the elevation `atan2(hB, d)` in degrees (absolute BS height), linear interpolation
  over the grid -90, 0, 10, ..., 90 deg, times 0.01:
  - DenseUrban: 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1
  - Urban: 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2
  - Suburban and Rural: 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8
- QuaDRiGa, `QuaDRiGa_Industrial`: `min(1, exp(-(d + 20)/80)·exp(min(hB, 8)/5 - 1))`
- `LOS`: 1, `NLOS`: 0
- Soft LOS state (TR 38.901 eq. 7.6-18), computed when `los_soft` is requested:
  `los_soft(l, k) = 1/2 + atan(sqrt(20/lambda_k)·(rnd_g(l) + F(p_los(l))))/pi`, `lambda_k = c/fc(k)` with
  `c = 299792458` m/s and `F(p) = sqrt(2)·erfinv(2p - 1)` the inverse of the standard normal CDF (accurate to
  `1e-10·max(1, abs(F))` for `p` in [1e-12, 1 - 1e-12]). `p = 1` gives 1, `p = 0` gives 0. `los_soft > 0.5`
  exactly when `rnd_g > -F(p)`, which has the probability `p` for `rnd_g` from N(0,1)
- The LOS state itself (a draw against `p_los`) and its spatial correlation are the caller's: e.g. a spatially
  consistent uniform value from [[sos_rand]], or `rnd_g` from [[sos_randn]], at the decorrelation distance of the
  scenario key `LOS_state_lambda` (TR 38.901 Table 7.6.3.1-2). The `LOS_state_lambda` of the InF conf files is
  `d_clutter / 2` of the TR defaults; it does not follow a `d_clutter` argument, so a caller who changes
  `d_clutter` passes its own correlation distance
- O2I UTs: Table 7.4.2-1 uses the outdoor distance `d2d_out = d2D - d2D-in` with `d2D-in` from [[o2i_loss]], which
  therefore runs first; the caller clamps `d2d_out` at 0 (negative distances are rejected). Indoor scenarios
  (InH, InF) pass the 2D distance
- Outputs must not share memory with the inputs or with each other
- Validation, in this order: the model name; `d2d_out` finite and >= 0; `h_bs` and `h_ut` with 1 or `n_link`
  elements and finite; `d_clutter` NaN or finite and > 0, `r_clutter` NaN or in [0, 1), `h_clutter` NaN or finite
  and >= 0 (for every model); if `los_soft` is requested: `rnd_g` and `fc` present, `rnd_g` with `n_link`
  finite elements, `fc` non-empty, finite and > 0. Without `los_soft`, `rnd_g` and `fc` are ignored. All checks
  run before any output is written

## Declaration:
```
void quadriga_lib::los_probability(
    const std::string &model,
    const arma::vec &d2d_out,
    const arma::vec &h_bs,
    const arma::vec &h_ut,
    arma::vec &p_los,
    double d_clutter = NAN,
    double r_clutter = NAN,
    double h_clutter = NAN,
    const arma::vec *rnd_g = nullptr,
    const arma::vec *fc = nullptr,
    arma::mat *los_soft = nullptr);
```

## Inputs:
- **`model`** — Model name: `38.901_RMa`, `38.901_UMi`, `38.901_UMa`, `38.901_InH_Mixed`, `38.901_InH_Open`,
  `38.901_InF_SL`, `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_DH`, `38.901_InF_HH`, `38.901_SMa`,
  `38.811_DenseUrban`, `38.811_Urban`, `38.811_Suburban`, `38.811_Rural`, `QuaDRiGa_Industrial`, `LOS`, `NLOS`
- **`d2d_out`** — 2D outdoor distance in [m] (indoor scenarios: 2D distance), finite and >= 0; `[n_link]`, 0 allowed
- **`h_bs`** — BS height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`h_ut`** — UT height in [m]; one element is used for all links; `[n_link]` or `[1]`
- **`d_clutter`** *(optional)* — Clutter size in [m] of the InF and SMa models; NaN = TR default of the model
- **`r_clutter`** *(optional)* — Clutter density of the InF models, vegetation density `r_veg` of SMa, in [0, 1);
  NaN = default
- **`h_clutter`** *(optional)* — Clutter height in [m] of InF-SH and InF-DH; NaN = default
- **`rnd_g`** *(optional)* — Normal(0,1) value per link for the soft LOS state (G of eq. 7.6-18); required with
  `los_soft`; `[n_link]`
- **`fc`** *(optional)* — Carrier frequencies in [Hz] for the soft LOS state; required with `los_soft`; `[n_freq]`

## Outputs:
- **`p_los`** — LOS probability in [0, 1]; `[n_link]`
- **`los_soft`** *(optional)* — Soft LOS state in [0, 1]; `[n_link, n_freq]`

## See also:
- [[o2i_loss]] (indoor distance for `d2d_out`)
- [[sos_rand]], [[sos_randn]] (spatially consistent LOS-state draw and `rnd_g`)
- [[conf_file_read]] (key `LOS_state_lambda`)
MD!*/

void quadriga_lib::los_probability(const std::string &model,
                                   const arma::vec &d2d_out,
                                   const arma::vec &h_bs,
                                   const arma::vec &h_ut,
                                   arma::vec &p_los,
                                   double d_clutter,
                                   double r_clutter,
                                   double h_clutter,
                                   const arma::vec *rnd_g,
                                   const arma::vec *fc,
                                   arma::mat *los_soft)
{
    // Model name
    int m = -1;
    for (int i = 0; i < (int)QD_LOS_N_MODEL; ++i)
        if (model == qd_los_names[i])
            m = i;
    if (m < 0)
        throw std::invalid_argument("Model '" + model + "' is not supported.");

    // Distances and heights
    const arma::uword n_link = d2d_out.n_elem;
    for (arma::uword l = 0; l < n_link; ++l)
        if (!std::isfinite(d2d_out[l]) || d2d_out[l] < 0.0)
            throw std::invalid_argument("Input 'd2d_out' must be finite and >= 0.");
    if (h_bs.n_elem != 1 && h_bs.n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'h_bs' must be 1 or match 'd2d_out'.");
    if (!h_bs.is_finite())
        throw std::invalid_argument("Input 'h_bs' must be finite.");
    if (h_ut.n_elem != 1 && h_ut.n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'h_ut' must be 1 or match 'd2d_out'.");
    if (!h_ut.is_finite())
        throw std::invalid_argument("Input 'h_ut' must be finite.");

    // Clutter arguments
    if (!std::isnan(d_clutter) && !(std::isfinite(d_clutter) && d_clutter > 0.0))
        throw std::invalid_argument("Input 'd_clutter' must be NaN or > 0.");
    if (!std::isnan(r_clutter) && !(r_clutter >= 0.0 && r_clutter < 1.0))
        throw std::invalid_argument("Input 'r_clutter' must be NaN or in the range from 0 to 1, 1 excluded.");
    if (!std::isnan(h_clutter) && !(std::isfinite(h_clutter) && h_clutter >= 0.0))
        throw std::invalid_argument("Input 'h_clutter' must be NaN or >= 0.");

    // Soft LOS inputs
    if (los_soft != nullptr)
    {
        if (rnd_g == nullptr || fc == nullptr)
            throw std::invalid_argument("Output 'los_soft' requires 'rnd_g' and 'fc'.");
        if (rnd_g->n_elem != n_link)
            throw std::invalid_argument("Number of elements in 'rnd_g' does not match number of elements in 'd2d_out'.");
        if (!rnd_g->is_finite())
            throw std::invalid_argument("Input 'rnd_g' must be finite.");
        qd_gbsm_check_fc(*fc);
    }

    // Model parameters with their defaults
    double kinv0 = 0.0, hc = 0.0;    // InF: -ln(1 - r)/d_clutter, clutter height
    double ks[3] = {0.0, 0.0, 0.0};  // SMa: -ln(1 - r_x)/d_clutter for commercial, residential, vegetation
    const double hs[3] = {20.0, 8.0, 15.0};
    const double *tab = nullptr;     // 38.811 table
    if (m == QD_LOS_INF_SL || m == QD_LOS_INF_DL || m == QD_LOS_INF_SH || m == QD_LOS_INF_DH)
    {
        const bool low = (m == QD_LOS_INF_SL || m == QD_LOS_INF_SH);
        const double r = std::isnan(r_clutter) ? (low ? 0.2 : 0.6) : r_clutter;
        const double dc = std::isnan(d_clutter) ? (low ? 10.0 : 2.0) : d_clutter;
        hc = std::isnan(h_clutter) ? (low ? 2.0 : 6.0) : h_clutter;
        kinv0 = -std::log(1.0 - r) / dc;
    }
    else if (m == QD_LOS_SMA)
    {
        const double dc = std::isnan(d_clutter) ? 30.0 : d_clutter;
        const double r_veg = std::isnan(r_clutter) ? 0.0 : r_clutter;
        ks[0] = -std::log(1.0 - 0.02) / dc;
        ks[1] = -std::log(1.0 - 0.18) / dc;
        ks[2] = -std::log(1.0 - r_veg) / dc;
    }
    else if (m == QD_LOS_DENSEURBAN)
        tab = qd_los_811_dense_urban;
    else if (m == QD_LOS_URBAN)
        tab = qd_los_811_urban;
    else if (m == QD_LOS_SUBURBAN || m == QD_LOS_RURAL)
        tab = qd_los_811_suburban_rural;

    // Outputs
    if (p_los.n_elem != n_link)
        p_los.set_size(n_link);

    // Clutter attenuation exp(-d·k), 1 at d = 0 also for k = +Inf (a subnormal 'd_clutter')
    auto clutter = [](double d, double k)
    { return (d > 0.0) ? std::exp(-d * k) : 1.0; };

    const bool hb1 = h_bs.n_elem == 1, hu1 = h_ut.n_elem == 1;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const double d = d2d_out[l], hB = hb1 ? h_bs[0] : h_bs[l], hU = hu1 ? h_ut[0] : h_ut[l];
        double p = 1.0;
        switch (m)
        {
        case QD_LOS_RMA:
            if (d > 10.0)
                p = std::exp(-(d - 10.0) / 1000.0);
            break;
        case QD_LOS_UMI:
            if (d > 18.0)
                p = 18.0 / d + std::exp(-d / 36.0) * (1.0 - 18.0 / d);
            break;
        case QD_LOS_UMA:
            if (d > 18.0)
            {
                const double Cp = hU <= 13.0 ? 0.0 : (hU <= 23.0 ? std::pow((hU - 13.0) / 10.0, 1.5) : 1.0);
                const double x = d / 100.0;
                p = (18.0 / d + std::exp(-d / 63.0) * (1.0 - 18.0 / d)) *
                    (1.0 + Cp * 1.25 * x * x * x * std::exp(-d / 150.0));
            }
            break;
        case QD_LOS_INH_MIXED:
            if (d > 1.2 && d < 6.5)
                p = std::exp(-(d - 1.2) / 4.7);
            else if (d >= 6.5)
                p = std::exp(-(d - 6.5) / 32.6) * 0.32;
            break;
        case QD_LOS_INH_OPEN:
            if (d > 5.0 && d <= 49.0)
                p = std::exp(-(d - 5.0) / 70.8);
            else if (d > 49.0)
                p = std::exp(-(d - 49.0) / 211.7) * 0.54;
            break;
        case QD_LOS_INF_SL:
        case QD_LOS_INF_DL:
            p = clutter(d, kinv0);
            break;
        case QD_LOS_INF_SH:
        case QD_LOS_INF_DH:
            if (hU >= hc)
                p = 1.0;
            else if (hB > hU)
                p = clutter(d, kinv0 * (hc - hU) / (hB - hU));
            else
                p = clutter(d, kinv0);
            break;
        case QD_LOS_INF_HH:
            break;
        case QD_LOS_SMA:
            if (d > 10.0 && hB > hU)
            {
                double s = 0.0;
                for (int x = 0; x < 3; ++x)
                    if (hs[x] > hU) // a UT above the clutter type is not blocked by it
                        s += ks[x] * (hs[x] - hU) / (hB - hU);
                p = clutter(d, s);
            }
            break;
        case QD_LOS_DENSEURBAN:
        case QD_LOS_URBAN:
        case QD_LOS_SUBURBAN:
        case QD_LOS_RURAL:
        {
            const double el = std::atan2(hB, d) * 180.0 / arma::datum::pi;
            p = 0.01 * qd_interp_linear(tab, qd_los_811_el, 11, el);
            break;
        }
        case QD_LOS_INDUSTRIAL:
            p = std::exp(-(d + 20.0) / 80.0) * std::exp(std::min(hB, 8.0) / 5.0 - 1.0);
            break;
        case QD_LOS_LOS:
            break;
        default: // NLOS
            p = 0.0;
        }
        p_los[l] = p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
    }

    // Soft LOS state, TR 38.901 eq. (7.6-18)
    if (los_soft != nullptr)
    {
        const arma::uword n_freq = fc->n_elem;
        arma::vec G(n_link, arma::fill::none); // G + F(p) per link
        for (arma::uword l = 0; l < n_link; ++l)
            G[l] = rnd_g->at(l) + qd_los_probit(p_los[l]);

        if (los_soft->n_rows != n_link || los_soft->n_cols != n_freq)
            los_soft->set_size(n_link, n_freq);
        for (arma::uword k = 0; k < n_freq && n_link > 0; ++k)
        {
            const double kk = std::sqrt(20.0 * fc->at(k) / 299792458.0); // sqrt(20 / lambda)
            double *ps = los_soft->colptr(k);
            for (arma::uword l = 0; l < n_link; ++l)
                ps[l] = 0.5 + std::atan(kk * G[l]) / arma::datum::pi;
        }
    }
}
