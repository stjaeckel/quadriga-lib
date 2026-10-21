// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <algorithm>
#include <utility>
#include <vector>

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; default threshold of 100 dB effectively includes all paths.
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their powers summed before computing the spread; function recurses on the binned profile.
  The bins are placed on a grid common to all CIRs; only occupied bins are stored, so memory does not grow with the delay span.
  All delays must then be finite, and the global delay span divided by `granularity` must be below 2^53.

## Declaration:
```
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    dtype threshold = 100.0,
    dtype granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

## Inputs:
- **`delays`** — Delays in [s] per CIR; `[n_cir]` vector, each element a column vector of length `n_path`
- **`powers`** — Path powers in linear scale [W]; same structure as `delays`
- **`threshold`** *(optional)* — Power threshold in [dB] relative to strongest path; paths below threshold are excluded
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain; 0 disables grouping; must be finite
  and non-negative; when > 0, non-finite delays or a span / granularity of 2^53 or more throw `std::invalid_argument`

## Outputs:
- **`mean_delay`** *(optional)* — Mean delay in [s] per CIR; `[n_cir]`

## Returns:
- RMS delay spread in [s] for each CIR; `[n_cir]`

## See also:
- [[quantize_delays]] (for mapping delays to a fixed tap grid)
- [[calc_rician_k_factor]] (for calculating K-factor)
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_delay_spread(const std::vector<arma::Col<dtype>> &delays,
                                                 const std::vector<arma::Col<dtype>> &powers,
                                                 dtype threshold,
                                                 dtype granularity,
                                                 arma::Col<dtype> *mean_delay)
{
    // --- Input validation ---
    arma::uword n_cir = (arma::uword)delays.size();

    if (n_cir == 0)
        throw std::invalid_argument("Input 'delays' must not be empty.");

    if (powers.size() != n_cir)
        throw std::invalid_argument("Input 'powers' must have the same length as 'delays'.");

    for (arma::uword i = 0; i < n_cir; ++i)
    {
        if (delays[i].n_elem != powers[i].n_elem)
            throw std::invalid_argument("Delays and powers must have the same number of elements for each CIR.");
        if (delays[i].n_elem == 0)
            throw std::invalid_argument("Delay and power vectors must not be empty.");
    }

    if (threshold <= (dtype)0.0)
        throw std::invalid_argument("Threshold must be positive.");

    if (!(granularity >= (dtype)0.0) || !std::isfinite(granularity))
        throw std::invalid_argument("Granularity must be a finite, non-negative number.");

    // --- Granularity-based binning ---
    if (granularity > (dtype)0.0)
    {
        // Find global min and max delay across all CIRs; all delays must be finite
        dtype global_min = delays[0].at(0);
        dtype global_max = global_min;
        for (arma::uword i = 0; i < n_cir; ++i)
        {
            const dtype *p_tau = delays[i].memptr();
            for (arma::uword p = 0; p < delays[i].n_elem; ++p)
            {
                if (!std::isfinite(p_tau[p]))
                    throw std::invalid_argument("Delays must be finite when 'granularity' is used.");
                global_min = p_tau[p] < global_min ? p_tau[p] : global_min;
                global_max = p_tau[p] > global_max ? p_tau[p] : global_max;
            }
        }

        // Bin indices are computed in floating point; they must stay exact integers
        if (((double)global_max - (double)global_min) / (double)granularity >= 9007199254740992.0) // 2^53
            throw std::invalid_argument("Delay span / granularity must be below 2^53.");

        dtype min_delay_bin = std::floor(global_min / granularity) * granularity;
        if (!std::isfinite(min_delay_bin))
            throw std::invalid_argument("Delays / granularity exceed the floating-point range.");

        // Sparse binning per CIR: bin index round((tau - min_delay_bin) / granularity) on the global grid, powers of
        // a bin summed in path order, occupied bins passed on in ascending order (empty bins hold no power and do
        // not change the result of the direct computation)
        std::vector<arma::Col<dtype>> binned_delays(n_cir);
        std::vector<arma::Col<dtype>> binned_powers(n_cir);
        std::vector<std::pair<double, arma::uword>> key; // (bin index, path index)

        for (arma::uword i = 0; i < n_cir; ++i)
        {
            const dtype *p_tau = delays[i].memptr();
            const dtype *p_pow = powers[i].memptr();
            arma::uword n_path = delays[i].n_elem;

            key.resize(n_path);
            for (arma::uword p = 0; p < n_path; ++p) // "+ 0.0" turns a -0 bin into +0
                key[p] = {(double)std::round((p_tau[p] - min_delay_bin) / granularity) + 0.0, p};
            std::sort(key.begin(), key.end());

            arma::uword n_bins = 0;
            for (arma::uword p = 0; p < n_path; ++p)
                n_bins += (p == 0 || key[p].first != key[p - 1].first) ? 1 : 0;

            arma::Col<dtype> &bd = binned_delays[i], &bp = binned_powers[i];
            bd.set_size(n_bins), bp.zeros(n_bins);
            arma::uword b = 0;
            for (arma::uword p = 0; p < n_path; ++p)
            {
                if (p != 0 && key[p].first != key[p - 1].first)
                    ++b;
                bd[b] = min_delay_bin + (dtype)key[p].first * granularity;
                bp[b] += p_pow[key[p].second];
            }
        }

        // Recursive call without granularity
        return quadriga_lib::calc_delay_spread(binned_delays, binned_powers, threshold, (dtype)0.0, mean_delay);
    }

    // --- Direct computation (granularity == 0) ---
    arma::Col<dtype> ds(n_cir);
    dtype *p_ds = ds.memptr();

    bool calc_mean = (mean_delay != nullptr);
    if (calc_mean)
    {
        mean_delay->set_size(n_cir);
    }

    for (arma::uword i = 0; i < n_cir; ++i)
    {
        arma::uword n_path = delays[i].n_elem;
        const dtype *p_tau = delays[i].memptr();
        const dtype *p_pow = powers[i].memptr();

        // Copy powers for thresholding
        arma::Col<dtype> pow_work(n_path);
        dtype *p_pw = pow_work.memptr();
        std::memcpy(p_pw, p_pow, n_path * sizeof(dtype));

        // Apply threshold
        dtype max_pow = *std::max_element(p_pw, p_pw + n_path);
        dtype min_pow = max_pow / std::pow((dtype)10.0, (dtype)0.1 * threshold);
        for (arma::uword p = 0; p < n_path; ++p)
        {
            if (p_pw[p] < min_pow)
                p_pw[p] = (dtype)0.0;
        }

        // Normalize powers
        dtype pt = (dtype)0.0;
        for (arma::uword p = 0; p < n_path; ++p)
            pt += p_pw[p];

        if (pt <= (dtype)0.0)
        {
            p_ds[i] = (dtype)0.0;
            if (calc_mean)
                mean_delay->at(i) = (dtype)0.0;
            continue;
        }

        dtype inv_pt = (dtype)1.0 / pt;
        for (arma::uword p = 0; p < n_path; ++p)
            p_pw[p] *= inv_pt;

        // Mean delay
        dtype md = (dtype)0.0;
        for (arma::uword p = 0; p < n_path; ++p)
            md += p_pw[p] * p_tau[p];

        if (calc_mean)
            mean_delay->at(i) = md;

        // RMS delay spread: sqrt( E[tau^2] - E[tau]^2 )
        // Using the same formula as MATLAB: sqrt( sum(p*(tau-md)^2) - (sum(p*(tau-md)))^2 )
        dtype sum_pw_tmp2 = (dtype)0.0;
        dtype sum_pw_tmp = (dtype)0.0;
        for (arma::uword p = 0; p < n_path; ++p)
        {
            dtype tmp = p_tau[p] - md;
            sum_pw_tmp += p_pw[p] * tmp;
            sum_pw_tmp2 += p_pw[p] * tmp * tmp;
        }

        dtype ds_sq = sum_pw_tmp2 - sum_pw_tmp * sum_pw_tmp;
        p_ds[i] = (ds_sq > (dtype)0.0) ? std::sqrt(ds_sq) : (dtype)0.0;
    }

    return ds;
}

template arma::Col<float> quadriga_lib::calc_delay_spread(const std::vector<arma::Col<float>> &delays,
                                                          const std::vector<arma::Col<float>> &powers,
                                                          float threshold,
                                                          float granularity,
                                                          arma::Col<float> *mean_delay);

template arma::Col<double> quadriga_lib::calc_delay_spread(const std::vector<arma::Col<double>> &delays,
                                                           const std::vector<arma::Col<double>> &powers,
                                                           double threshold,
                                                           double granularity,
                                                           arma::Col<double> *mean_delay);
