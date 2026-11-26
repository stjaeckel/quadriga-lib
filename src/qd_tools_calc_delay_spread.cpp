// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# calc_delay_spread
Calculate the RMS delay spread of each CIR from its path delays and powers

- With the path powers $p_l$ normalized to sum 1 over the paths of a CIR, the mean delay and the RMS delay spread are

  $$\bar\tau = \sum_l p_l \, \tau_l, \qquad \sigma_\tau = \sqrt{\sum_l p_l \, (\tau_l - \bar\tau)^2}$$

- Paths more than `threshold_dB` below the strongest path of their CIR are left out
- With `granularity` > 0, the delays are first rounded to the nearest multiple of `granularity`, and the powers of
  the paths with the same rounded delay are summed
- A CIR without power gives a spread and a mean delay of 0

## Declaration:
```
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    double threshold_dB = 100.0,
    double granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

## Inputs:
- **`delays`** — Path delays in s; `[n_cir]` vectors `[n_path]`, at least one CIR and one path per CIR; `n_path` may
  differ between the CIRs
- **`powers`** — Path powers, linear; `[n_cir]` vectors `[n_path]`
- **`threshold_dB`** *(optional)* — Power range in dB below the strongest path that is kept, > 0
- **`granularity`** *(optional)* — Spacing of the delay grid in s, finite and >= 0; 0 = no grid; with a grid, all
  delays must be finite and their span must be below $2^{53}$ grid steps

## Output:
- **`mean_delay`** *(optional)* — Mean delay $\bar\tau$ in s; `[n_cir]`

## Returns:
- RMS delay spread $\sigma_\tau$ in s; `[n_cir]`

## See also:
- [[quantize_delays]] (delays on a tap grid with interpolation)
- [[calc_angular_spread]] (angular spreads)
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_delay_spread(const std::vector<arma::Col<dtype>> &delays,
                                                 const std::vector<arma::Col<dtype>> &powers,
                                                 double threshold_dB_in,
                                                 double granularity_in,
                                                 arma::Col<dtype> *mean_delay)
{
    const dtype threshold_dB = dtype(threshold_dB_in);
    const dtype granularity = dtype(granularity_in);
    // --- Input validation ---
    arma::uword n_cir = (arma::uword)delays.size();

    if (n_cir == 0)
        throw std::invalid_argument("Input 'delays' must not be empty.");

    if (powers.size() != n_cir)
        throw std::invalid_argument("Input 'powers' must have the same length as 'delays'.");

    for (arma::uword i = 0; i < n_cir; ++i)
    {
        if (delays[i].n_elem != powers[i].n_elem)
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Input 'powers' must have the same number of elements as 'delays'.");
        if (delays[i].n_elem == 0)
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Input 'delays' must not be empty.");
        qd_check_alias(mean_delay, "mean_delay", &delays[i], &powers[i]);
    }

    if (threshold_dB <= (dtype)0.0)
        throw std::invalid_argument("Input 'threshold_dB' must be > 0.");

    if (!(granularity >= (dtype)0.0) || !std::isfinite(granularity))
        throw std::invalid_argument("Input 'granularity' must be finite and non-negative.");

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
        return quadriga_lib::calc_delay_spread(binned_delays, binned_powers, threshold_dB, (dtype)0.0, mean_delay);
    }

    // --- Direct computation (granularity == 0) ---
    arma::Col<dtype> ds(n_cir, arma::fill::none);
    dtype *p_ds = ds.memptr();

    bool calc_mean = (mean_delay != nullptr);
    if (calc_mean)
        mean_delay->set_size(n_cir);

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
        dtype min_pow = max_pow / std::pow((dtype)10.0, (dtype)0.1 * threshold_dB);
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
                                                          double threshold_dB,
                                                          double granularity,
                                                          arma::Col<float> *mean_delay);

template arma::Col<double> quadriga_lib::calc_delay_spread(const std::vector<arma::Col<double>> &delays,
                                                           const std::vector<arma::Col<double>> &powers,
                                                           double threshold_dB,
                                                           double granularity,
                                                           arma::Col<double> *mean_delay);
