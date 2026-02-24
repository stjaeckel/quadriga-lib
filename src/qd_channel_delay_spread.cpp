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

#include "quadriga_tools.hpp"

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
  falling into the same delay bin are summed before computing the delay spread. This is useful
  when the system bandwidth limits the time resolution (e.g. 50 ns at 20 MHz bandwidth).
- When granularity is applied the function recursively calls itself on the binned power delay
  profile.
- Optionally returns the mean delay for each CIR.

## Declaration:
```
template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_delay_spread(
    const std::vector<arma::Col<dtype>> &delays,
    const std::vector<arma::Col<dtype>> &powers,
    dtype threshold = 100.0,
    dtype granularity = 0.0,
    arma::Col<dtype> *mean_delay = nullptr);
```

## Arguments:
- `const std::vector<arma::Col<dtype>> &**delays**` (input)<br>
  Delays in [s]. A vector of length `n_cir`, where each element is an Armadillo column vector
  of length `n_path` (the number of paths may differ per CIR).

- `const std::vector<arma::Col<dtype>> &**powers**` (input)<br>
  Path powers on a linear scale [W]. Same structure as `delays`.

- `dtype **threshold** = 100.0` (input)<br>
  Power threshold in [dB] relative to the strongest path. Paths with power below
  `max_power / 10^(0.1 * threshold)` are excluded. Default: 100 dB (effectively all paths).

- `dtype **granularity** = 0.0` (input)<br>
  Window size in [s] for grouping paths in the delay domain. Paths whose delays fall into the
  same bin of width `granularity` have their powers summed. Default: 0 (no grouping).

- `arma::Col<dtype> ***mean_delay** = nullptr` (optional output)<br>
  Mean delay in [s] for each CIR. Length `[n_cir]`.

## Returns:
- `arma::Col<dtype> **ds**` (output)<br>
  RMS delay spread in [s] for each CIR. Length `[n_cir]`.

## Example:
```
std::vector<arma::vec> delays = { {0.0, 1e-6, 2e-6} };
std::vector<arma::vec> powers = { {1.0, 0.5, 0.25} };
arma::vec mean_delay;
arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);
// ds(0) ≈ 0.6901e-6, mean_delay(0) ≈ 0.5714e-6
```
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

// --- Explicit template instantiation ---
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
