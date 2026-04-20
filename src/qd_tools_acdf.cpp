// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back; if non-empty, those bin centers are used; if `nullptr`, bins are auto-generated internally

## Declaration:
```
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
    arma::Col<dtype> *bins = nullptr,
    arma::Mat<dtype> *Sh = nullptr,
    arma::Col<dtype> *Sc = nullptr,
    arma::Col<dtype> *mu = nullptr,
    arma::Col<dtype> *sig = nullptr,
    arma::uword n_bins = 201);
```

## Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins`** *(optional)* — Bin centers; auto-generated and stored back if pointing to empty vector, used as-is if non-empty, ignored if `nullptr`; `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

## Outputs:
- **`Sh`** *(optional)* — Individual CDFs, one per column of data; `[n_bins, n_sets]`
- **`Sc`** *(optional)* — Averaged CDF via quantile-space averaging across data sets; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets, `[9]`
MD!*/

template <typename dtype>
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
                        arma::Col<dtype> *bins,
                        arma::Mat<dtype> *Sh,
                        arma::Col<dtype> *Sc,
                        arma::Col<dtype> *mu,
                        arma::Col<dtype> *sig,
                        arma::uword n_bins)
{
    // --- Input validation ---

    if (data.is_empty())
        throw std::invalid_argument("Input 'data' must not be empty.");

    if (n_bins < 2)
        throw std::invalid_argument("Input 'n_bins' must be at least 2.");

    arma::uword n_samples = data.n_rows;
    arma::uword n_sets = data.n_cols;

    // --- Determine bins ---

    bool bins_provided = (bins != nullptr && bins->n_elem > 0);
    arma::Col<dtype> bins_local;

    if (bins_provided)
    {
        bins_local = *bins;
    }
    else
    {
        // Find min and max of all finite data
        dtype mi = std::numeric_limits<dtype>::max();
        dtype ma = std::numeric_limits<dtype>::lowest();
        const dtype *p_data = data.memptr();
        arma::uword n_total = n_samples * n_sets;

        for (arma::uword i = 0; i < n_total; ++i)
        {
            dtype val = p_data[i];
            if (std::isfinite(val))
            {
                if (val < mi)
                    mi = val;
                if (val > ma)
                    ma = val;
            }
        }

        if (mi > ma)
            throw std::invalid_argument("Input 'data' contains no finite values.");

        if (mi == ma)
        {
            dtype offset = (mi == (dtype)0.0) ? (dtype)1.0 : std::abs(mi) * (dtype)0.1;
            mi -= offset;
            ma += offset;
        }

        bins_local.set_size(n_bins);
        dtype step = (ma - mi) / (dtype)(n_bins - 1);
        for (arma::uword i = 0; i < n_bins; ++i)
            bins_local(i) = mi + (dtype)i * step;

        if (bins != nullptr)
            *bins = bins_local;
    }

    arma::uword no_bins = bins_local.n_elem;
    const dtype *p_bins = bins_local.memptr();

    // --- Compute individual CDFs (Sh) ---

    // We need the individual CDFs for Sc/mu/sig computation regardless of whether Sh is requested
    arma::Mat<dtype> Sh_local(no_bins, n_sets, arma::fill::zeros);

    for (arma::uword s = 0; s < n_sets; ++s)
    {
        const dtype *p_col = data.colptr(s);
        dtype *p_sh = Sh_local.colptr(s);
        arma::uword n_valid = 0;

        // Compute histogram counts
        // MATLAB's hist assigns each value to the nearest bin center
        // Bin edges are midpoints between consecutive bin centers
        for (arma::uword i = 0; i < n_samples; ++i)
        {
            dtype val = p_col[i];
            if (!std::isfinite(val))
                continue;

            ++n_valid;

            // Clamp Inf-like values
            // Find the appropriate bin using edge midpoints
            if (val <= p_bins[0])
            {
                p_sh[0] += (dtype)1.0;
            }
            else if (val >= p_bins[no_bins - 1])
            {
                p_sh[no_bins - 1] += (dtype)1.0;
            }
            else
            {
                // Binary search for the nearest bin center
                arma::uword lo = 0, hi = no_bins - 1;
                while (hi - lo > 1)
                {
                    arma::uword mid = (lo + hi) / 2;
                    if (val < p_bins[mid])
                        hi = mid;
                    else
                        lo = mid;
                }
                // val is between p_bins[lo] and p_bins[hi]
                // Assign to nearest bin center
                dtype edge = (p_bins[lo] + p_bins[hi]) / (dtype)2.0;
                if (val < edge)
                    p_sh[lo] += (dtype)1.0;
                else
                    p_sh[hi] += (dtype)1.0;
            }
        }

        // Cumulative sum and normalize
        if (n_valid > 0)
        {
            dtype inv_n = (dtype)1.0 / (dtype)n_valid;
            dtype cumsum = (dtype)0.0;
            for (arma::uword b = 0; b < no_bins; ++b)
            {
                cumsum += p_sh[b];
                p_sh[b] = cumsum * inv_n;
            }
        }
    }

    if (Sh != nullptr)
        *Sh = Sh_local;

    // --- Compute averaged CDF, mu, sig ---

    if (Sc != nullptr || mu != nullptr || sig != nullptr)
    {
        if (n_sets == 1)
        {
            // No averaging needed
            if (Sc != nullptr)
                *Sc = Sh_local.col(0);

            if (mu != nullptr || sig != nullptr)
            {
                arma::Col<dtype> mu_local(9);
                arma::Col<dtype> sig_local(9, arma::fill::zeros);
                const dtype *p_cdf = Sh_local.memptr();
                for (arma::uword q = 0; q < 9; ++q)
                {
                    dtype level = (dtype)(q + 1) * (dtype)0.1;
                    arma::uword idx = 0;
                    for (arma::uword b = 0; b < no_bins; ++b)
                    {
                        if (p_cdf[b] >= level)
                        {
                            idx = b;
                            break;
                        }
                        idx = b;
                    }
                    mu_local(q) = p_bins[idx];
                }
                if (mu != nullptr)
                    *mu = mu_local;
                if (sig != nullptr)
                    *sig = sig_local;
            }
        }
        else
        {
            // Quantile-space averaging
            // Fine grid of probability levels
            arma::uword n_vals = 1000;
            arma::Col<dtype> vals(n_vals);
            for (arma::uword i = 0; i < n_vals; ++i)
                vals(i) = (dtype)i / (dtype)n_vals;

            // For each probability level, find x-value from each CDF and average
            arma::Col<dtype> Sc_q(n_vals); // averaged x-values in quantile space

            bool get_mu_sig = (mu != nullptr || sig != nullptr);
            arma::Mat<dtype> temp;
            if (get_mu_sig)
                temp.set_size(n_sets, n_vals);

            for (arma::uword v = 0; v < n_vals; ++v)
            {
                dtype level = vals(v);
                dtype sum_x = (dtype)0.0;

                for (arma::uword s = 0; s < n_sets; ++s)
                {
                    const dtype *p_cdf = Sh_local.colptr(s);
                    // Find first bin where CDF > level
                    arma::uword idx = no_bins - 1;
                    for (arma::uword b = 0; b < no_bins; ++b)
                    {
                        if (p_cdf[b] >= level)
                        {
                            idx = b;
                            break;
                        }
                    }
                    dtype x_val = p_bins[idx];
                    sum_x += x_val;
                    if (get_mu_sig)
                        temp(s, v) = x_val;
                }
                Sc_q(v) = sum_x / (dtype)n_sets;
            }

            // Compute mu and sig at quantile levels 0.1, 0.2, ..., 0.9
            if (get_mu_sig)
            {
                arma::Col<dtype> mu_local(9);
                arma::Col<dtype> sig_local(9);

                for (arma::uword q = 0; q < 9; ++q)
                {
                    // Quantile level (q+1)*0.1 corresponds to index (q+1)*100
                    // Use a window around it for robustness (matching MATLAB)
                    arma::uword center = (q + 1) * 100;
                    arma::uword i_start = (center >= 5) ? center - 5 : 0;
                    arma::uword i_end = (center + 5 < n_vals) ? center + 5 : n_vals - 1;

                    dtype sum_val = (dtype)0.0;
                    dtype sum_sq = (dtype)0.0;
                    arma::uword count = 0;

                    for (arma::uword v = i_start; v <= i_end; ++v)
                    {
                        for (arma::uword s = 0; s < n_sets; ++s)
                        {
                            dtype val = temp(s, v);
                            sum_val += val;
                            sum_sq += val * val;
                            ++count;
                        }
                    }

                    dtype mean_val = sum_val / (dtype)count;
                    mu_local(q) = mean_val;

                    // Use population std for consistency with MATLAB's std (which uses N-1),
                    // but we are computing over a window, so use sample std
                    dtype n_f = (dtype)count;
                    dtype sample_var = (sum_sq - sum_val * sum_val / n_f) / (n_f - (dtype)1.0);
                    sig_local(q) = (sample_var > (dtype)0.0) ? std::sqrt(sample_var) : (dtype)0.0;
                }

                if (mu != nullptr)
                    *mu = mu_local;
                if (sig != nullptr)
                    *sig = sig_local;
            }

            // Map averaged quantile function back to bin grid
            if (Sc != nullptr)
            {
                arma::Col<dtype> Sc_local(no_bins);
                for (arma::uword b = 0; b < no_bins; ++b)
                {
                    arma::uword count = 0;
                    for (arma::uword v = 0; v < n_vals; ++v)
                    {
                        if (Sc_q(v) < p_bins[b])
                            ++count;
                    }
                    Sc_local(b) = (dtype)count / (dtype)n_vals;
                }
                *Sc = Sc_local;
            }
        }
    }
}

template void quadriga_lib::acdf(const arma::Mat<float> &data,
                                 arma::Col<float> *bins,
                                 arma::Mat<float> *Sh,
                                 arma::Col<float> *Sc,
                                 arma::Col<float> *mu,
                                 arma::Col<float> *sig,
                                 arma::uword n_bins);

template void quadriga_lib::acdf(const arma::Mat<double> &data,
                                 arma::Col<double> *bins,
                                 arma::Mat<double> *Sh,
                                 arma::Col<double> *Sc,
                                 arma::Col<double> *mu,
                                 arma::Col<double> *sig,
                                 arma::uword n_bins);
