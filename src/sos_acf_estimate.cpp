// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_acf_estimate
Monte-Carlo estimate of the ACF of one generator

- Estimates the autocorrelation function (ACF) of one realisation of a generator from its values at random
  positions, without using the frequencies other than through [[sos_val]]
- Draws `2·n_pos` positions, uniformly in a cube with half edge `0.4·max(bins)` around the origin, evaluates
  the generator there, and for each distance bin computes the correlation coefficient between the two values
  of all position pairs whose distance falls into the bin
- Positions: a matrix `[2·n_pos, 3]` is filled column by column (all x, then all y, then all z) with
  `( 2·u - 1 ) · 0.4·max(bins)`, where `u` comes from the random stream described in [[sos_init_phases]]
- Bin `n` covers the distances from the midpoint between `bins(n-1)` and `bins(n)` (excluded) to the midpoint
  between `bins(n)` and `bins(n+1)` (included). The first bin starts at 1e-5 m, the last bin ends half a mean
  bin spacing above `bins(end)`
- Bins with fewer than 6 pairs return NaN in all outputs
- `acf_est_dual`: the same for two devices on the same map. The first `n_pos` positions are the first device,
  the second `n_pos` positions the second device, and the value of a link is [[sos_val]] with both positions.
  Two links form a pair if the distance between their first devices falls into the bin and the distance
  between their second devices does not exceed the upper limit of the bin; requires `acf` and `dist`
- The estimate describes one realisation; it scatters around the ACF of the sinusoids

## Declaration:
```
void quadriga_lib::sos_acf_estimate(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &phase,
    const arma::Col<dtype> &bins,
    arma::uword n_pos,
    arma::Col<dtype> &acf_est,
    arma::Col<dtype> *dist_est = nullptr,
    arma::Col<dtype> *acf_est_dual = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::sword seed = -1);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `[n_coeff, 3]`
- **`phase`** — Phases of the generator in [rad]; `[n_coeff]`
- **`bins`** — Bin centers in [m]; must be >= 0 and increase strictly; at least 2 elements; `[n_bin]`
- **`n_pos`** — Number of random positions per device; must be > 1
- **`acf`** *(optional)* — Target ACF; required for `acf_est_dual`; `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0 and increase strictly; `[n_acf]`
- **`seed`** *(optional)* — Seed for the positions; must be >= -1; `-1` seeds from the system random device

## Outputs:
- **`acf_est`** — Estimated ACF; `[n_bin]`
- **`dist_est`** *(optional)* — Mean distance of the pairs in each bin in [m]; `[n_bin]`
- **`acf_est_dual`** *(optional)* — Estimated ACF for two devices on the same map; `[n_bin]`

## See also:
- [[sos_acf_approx]] (ACF of the sinusoids in closed form)
- [[sos_val]] (evaluation of the generator)
MD!*/

template <typename dtype>
void quadriga_lib::sos_acf_estimate(const arma::Mat<dtype> &freq,
                                    const arma::Col<dtype> &phase,
                                    const arma::Col<dtype> &bins,
                                    arma::uword n_pos,
                                    arma::Col<dtype> &acf_est,
                                    arma::Col<dtype> *dist_est,
                                    arma::Col<dtype> *acf_est_dual,
                                    const arma::Col<dtype> *acf,
                                    const arma::Col<dtype> *dist,
                                    arma::sword seed)
{
    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols != 3)
        throw std::invalid_argument("Input 'freq' must have 3 columns.");
    if (phase.n_elem != freq.n_rows)
        throw std::invalid_argument("Number of elements in 'phase' does not match number of rows in 'freq'.");
    if (bins.n_elem < 2)
        throw std::invalid_argument("Input 'bins' must have at least 2 elements.");
    if (!(bins.at(0) >= (dtype)0.0))
        throw std::invalid_argument("Input 'bins' must be >= 0.");
    for (arma::uword i = 1; i < bins.n_elem; ++i)
        if (!(bins.at(i) > bins.at(i - 1)))
            throw std::invalid_argument("Input 'bins' must increase strictly.");
    if (n_pos < 2)
        throw std::invalid_argument("Input 'n_pos' must be > 1.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");
    if (acf_est_dual != nullptr)
    {
        if (acf == nullptr || dist == nullptr || acf->is_empty() || dist->is_empty())
            throw std::invalid_argument("Inputs 'acf' and 'dist' are required for output 'acf_est_dual'.");
        qd_sos_check_acf(*acf, *dist);
    }

    const arma::uword n_bin = bins.n_elem;

    const arma::uword n_all = 2 * n_pos;
    const double half_edge = 0.4 * (double)bins.at(n_bin - 1);

    // Random positions: rows 0 ... n_pos-1 first device, rows n_pos ... 2·n_pos-1 second device
    std::mt19937_64 rng = qd_sos_rng(seed);
    arma::mat pos(n_all, 3, arma::fill::none);
    for (arma::uword i = 0; i < pos.n_elem; ++i)
        pos.at(i) = (2.0 * qd_sos_uniform(rng) - 1.0) * half_edge;

    // Upper limit of each bin; the lower limit is the upper limit of the previous bin
    std::vector<double> upper(n_bin);
    for (arma::uword b = 0; b + 1 < n_bin; ++b)
        upper[b] = 0.5 * ((double)bins.at(b) + (double)bins.at(b + 1));
    upper[n_bin - 1] = (double)bins.at(n_bin - 1) + 0.5 * ((double)bins.at(n_bin - 1) - (double)bins.at(0)) / (double)(n_bin - 1);
    const double lowest = 1.0e-5;

    // Index of the bin that holds distance d, n_bin if none
    auto find_bin = [&](double d) -> arma::uword
    {
        if (!(d > lowest) || d > upper[n_bin - 1])
            return n_bin;
        arma::uword lo = 0, hi = n_bin - 1;
        while (lo < hi)
        {
            arma::uword mid = (lo + hi) / 2;
            if (d <= upper[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    };

    auto distance = [&](arma::uword i, arma::uword j) -> double
    {
        double dx = pos.at(i, 0) - pos.at(j, 0), dy = pos.at(i, 1) - pos.at(j, 1), dz = pos.at(i, 2) - pos.at(j, 2);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Sums for the correlation coefficient of each bin: count, x, y, x^2, y^2, x·y, distance
    auto correlate = [&](const std::vector<double> &acc, arma::uword b, double &coeff, double &mean_dist)
    {
        const double *a = &acc[7 * b];
        coeff = std::numeric_limits<double>::quiet_NaN(), mean_dist = coeff;
        if (a[0] > 5.5)
        {
            double n = a[0], cov = a[5] / n - a[1] * a[2] / (n * n);
            double vx = a[3] / n - a[1] * a[1] / (n * n), vy = a[4] / n - a[2] * a[2] / (n * n);
            coeff = cov / std::sqrt(vx * vy), mean_dist = a[6] / n;
        }
    };

    // The GENERIC kernel is used: this is an offline tool and its result should not depend on the CPU
    const arma::mat freq_d = arma::conv_to<arma::mat>::from(freq), phase_d = arma::conv_to<arma::mat>::from(phase);
    arma::mat val;
    quadriga_lib::sos_val<double>(freq_d, phase_d, pos, val, nullptr, nullptr, nullptr, nullptr, 1.0, false, 1);

    std::vector<double> acc(7 * n_bin, 0.0);
    for (arma::uword i = 0; i < n_all; ++i)
        for (arma::uword j = i + 1; j < n_all; ++j)
        {
            double d = distance(i, j);
            arma::uword b = find_bin(d);
            if (b < n_bin)
            {
                double x = val.at(i, 0), y = val.at(j, 0), *a = &acc[7 * b];
                a[0] += 1.0, a[1] += x, a[2] += y, a[3] += x * x, a[4] += y * y, a[5] += x * y, a[6] += d;
            }
        }

    if (acf_est.n_elem != n_bin)
        acf_est.set_size(n_bin);
    if (dist_est != nullptr && dist_est->n_elem != n_bin)
        dist_est->set_size(n_bin);
    for (arma::uword b = 0; b < n_bin; ++b)
    {
        double coeff, mean_dist;
        correlate(acc, b, coeff, mean_dist);
        acf_est.at(b) = (dtype)coeff;
        if (dist_est != nullptr)
            dist_est->at(b) = (dtype)mean_dist;
    }

    if (acf_est_dual != nullptr)
    {
        const arma::vec acf_d = arma::conv_to<arma::vec>::from(*acf), dist_d = arma::conv_to<arma::vec>::from(*dist);
        const arma::mat pos_a = pos.rows(0, n_pos - 1), pos_b = pos.rows(n_pos, n_all - 1);
        quadriga_lib::sos_val<double>(freq_d, phase_d, pos_a, val, &pos_b, nullptr, &acf_d, &dist_d, 1.0, false, 1);

        std::fill(acc.begin(), acc.end(), 0.0);
        for (arma::uword i = 0; i < n_pos; ++i)
            for (arma::uword j = i + 1; j < n_pos; ++j)
            {
                double d = distance(i, j);
                arma::uword b = find_bin(d);
                if (b < n_bin && distance(n_pos + i, n_pos + j) <= upper[b])
                {
                    double x = val.at(i, 0), y = val.at(j, 0), *a = &acc[7 * b];
                    a[0] += 1.0, a[1] += x, a[2] += y, a[3] += x * x, a[4] += y * y, a[5] += x * y, a[6] += d;
                }
            }

        if (acf_est_dual->n_elem != n_bin)
            acf_est_dual->set_size(n_bin);
        for (arma::uword b = 0; b < n_bin; ++b)
        {
            double coeff, mean_dist;
            correlate(acc, b, coeff, mean_dist);
            acf_est_dual->at(b) = (dtype)coeff;
        }
    }
}

template void quadriga_lib::sos_acf_estimate(const arma::Mat<float> &freq, const arma::Col<float> &phase, const arma::Col<float> &bins, arma::uword n_pos,
                                             arma::Col<float> &acf_est, arma::Col<float> *dist_est, arma::Col<float> *acf_est_dual,
                                             const arma::Col<float> *acf, const arma::Col<float> *dist, arma::sword seed);
template void quadriga_lib::sos_acf_estimate(const arma::Mat<double> &freq, const arma::Col<double> &phase, const arma::Col<double> &bins, arma::uword n_pos,
                                             arma::Col<double> &acf_est, arma::Col<double> *dist_est, arma::Col<double> *acf_est_dual,
                                             const arma::Col<double> *acf, const arma::Col<double> *dist, arma::sword seed);
