// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_delay_spread
Calculates RMS delay spread from per-CIR delays and linear-scale powers

- Paths with power below `p_max / 10^(0.1 * threshold)` are excluded; default threshold of 100 dB effectively includes all paths.
- When `granularity > 0`, paths falling into the same delay bin of width `granularity` have their powers summed before computing the spread; function recurses on the binned profile.

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
- **`granularity`** *(optional)* — Bin width in [s] for grouping paths in the delay domain; 0 disables grouping

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

    if (granularity < (dtype)0.0)
        throw std::invalid_argument("Granularity must be non-negative.");

    // --- Granularity-based binning ---
    if (granularity > (dtype)0.0)
    {
        // Find global min and max delay across all CIRs
        dtype global_min = delays[0].min();
        dtype global_max = delays[0].max();
        for (arma::uword i = 1; i < n_cir; ++i)
        {
            dtype cmin = delays[i].min();
            dtype cmax = delays[i].max();
            if (cmin < global_min)
                global_min = cmin;
            if (cmax > global_max)
                global_max = cmax;
        }

        dtype min_delay_bin = std::floor(global_min / granularity) * granularity;
        dtype max_delay_bin = std::ceil(global_max / granularity) * granularity;

        // Number of bins
        arma::uword n_bins = (arma::uword)std::round((max_delay_bin - min_delay_bin) / granularity) + 1;

        // Build the delay axis for the bins
        arma::Col<dtype> delay_axis(n_bins);
        dtype *p_delay_axis = delay_axis.memptr();
        for (arma::uword b = 0; b < n_bins; ++b)
            p_delay_axis[b] = min_delay_bin + (dtype)b * granularity;

        // Build binned power delay profiles
        std::vector<arma::Col<dtype>> binned_delays(n_cir);
        std::vector<arma::Col<dtype>> binned_powers(n_cir);

        for (arma::uword i = 0; i < n_cir; ++i)
        {
            arma::Col<dtype> pdp(n_bins, arma::fill::zeros);
            dtype *p_pdp = pdp.memptr();
            const dtype *p_tau = delays[i].memptr();
            const dtype *p_pow = powers[i].memptr();
            arma::uword n_path = delays[i].n_elem;

            for (arma::uword p = 0; p < n_path; ++p)
            {
                arma::uword bin_idx = (arma::uword)std::round((p_tau[p] - min_delay_bin) / granularity);
                if (bin_idx < n_bins)
                    p_pdp[bin_idx] += p_pow[p];
            }

            binned_delays[i] = delay_axis;
            binned_powers[i] = pdp;
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
