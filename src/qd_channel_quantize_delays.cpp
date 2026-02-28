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

#include "quadriga_channel.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <stdexcept>

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# quantize_delays
Fixes the path delays to a grid of delay bins

## Description:
- For channel emulation with finite delay resolution, path delays must be mapped to a fixed grid
  of delay bins (taps). Rounding delays to the nearest tap causes discontinuities in the frequency
  domain when a delay crosses a tap boundary (e.g. as a mobile terminal moves). This function
  instead approximates each path delay using two adjacent taps with power-weighted coefficients,
  producing smooth transitions.
- For a path at fractional offset &delta; between tap indices, two taps are created with complex
  coefficients scaled by (1&minus;&delta;)^&alpha; and &delta;^&alpha;, where &alpha; is the power
  exponent. The default &alpha;=1.0 (linear interpolation) is optimal for narrowband systems. Use
  &alpha;=0.5 to preserve wideband (incoherent) power.
- If input delays are already quantized (all fractional offsets below 0.01), the interpolation
  weight computation is skipped but the same delay-selection logic is used.
- The `fix_taps` parameter controls whether delay grids are shared across antenna pairs and/or
  snapshots, trading accuracy for a more compact representation.
- Input delays may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`. When
  shared and fix_taps is 0 or 3, delays are expanded internally and output delays are per-antenna.
  When shared and fix_taps is 1 or 2, output delays remain shared `[1, 1, n_taps]`.
- The number of antennas `n_rx` and `n_tx` must be the same across all snapshots, but the number
  of paths `n_path_s` may differ per snapshot.

## Declaration:
```
template <typename dtype>
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing = (dtype)5.0e-9,
    arma::uword max_no_taps = 48,
    dtype power_exponent = (dtype)1.0,
    int fix_taps = 0);
```

## Arguments:
- `const std::vector<arma::Cube<dtype>> ***coeff_re**` (input)<br>
  Channel coefficients, real part. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` where `n_path_s` may differ across snapshots.

- `const std::vector<arma::Cube<dtype>> ***coeff_im**` (input)<br>
  Channel coefficients, imaginary part. Same sizes as `coeff_re`.

- `const std::vector<arma::Cube<dtype>> ***delay**` (input)<br>
  Path delays in seconds. Vector of length `n_snap`, each cube of size
  `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`. The number of paths must match `coeff_re`.

- `std::vector<arma::Cube<dtype>> ***coeff_re_quant**` (output)<br>
  Output coefficients, real part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***coeff_im_quant**` (output)<br>
  Output coefficients, imaginary part. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]`.

- `std::vector<arma::Cube<dtype>> ***delay_quant**` (output)<br>
  Output delays in seconds. Vector of length `n_snap`, each cube of size `[n_rx, n_tx, n_taps]` or
  `[1, 1, n_taps]`.

- `dtype **tap_spacing** = 5.0e-9` (input)<br>
  Spacing of the delay bins in seconds. Default: 5 ns (200 MHz sampling rate).

- `arma::uword **max_no_taps** = 48` (input)<br>
  Maximum number of output taps. 0 means unlimited.

- `dtype **power_exponent** = 1.0` (input)<br>
  Interpolation exponent &alpha;. Use 1.0 for narrowband (linear) or 0.5 for wideband (power-preserving).

- `int **fix_taps** = 0` (input)<br>
  Delay sharing mode: 0 = per tx-rx pair and snapshot, 1 = single grid for all,
  2 = per snapshot, 3 = per tx-rx pair.

## Example:
```
// Create synthetic test data: 2 snapshots with different numbers of paths
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1, 1, 3); cim[0].set_size(1, 1, 3); dl[0].set_size(1, 1, 3);
cre[1].set_size(1, 1, 2); cim[1].set_size(1, 1, 2); dl[1].set_size(1, 1, 2);
cre[0](0,0,0) = 1.0; cre[0](0,0,1) = 0.5; cre[0](0,0,2) = 0.3;
cre[1](0,0,0) = 0.8; cre[1](0,0,1) = 0.4;
cim[0].zeros(); cim[1].zeros();
dl[0](0,0,0) = 0.0; dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9;
dl[1](0,0,0) = 0.0; dl[1](0,0,1) = 10.0e-9;

std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```
MD!*/

// --- Internal helper: find_optimal_delays ---
// Accumulates PDP, selects the strongest taps up to max_taps.
// dn/pn: rounded tap indices and their power (n_non entries)
// di/pi: interpolated tap indices and their power (n_int entries, may be 0)
// Returns sorted vector of selected tap indices.
static std::vector<unsigned> find_optimal_delays(
    const unsigned *dn, const double *pn, arma::uword n_non,
    const unsigned *di, const double *pi, arma::uword n_int,
    arma::uword max_taps)
{
    // Find maximum tap index across both arrays
    unsigned max_idx = 0;
    for (arma::uword i = 0; i < n_non; ++i)
        if (dn[i] > max_idx)
            max_idx = dn[i];
    for (arma::uword i = 0; i < n_int; ++i)
        if (di[i] > max_idx)
            max_idx = di[i];

    if (max_idx == 0 && n_non == 0 && n_int == 0)
        return {};

    // Accumulate PDP for non-interpolated taps
    std::vector<double> pdp_non(max_idx + 1, 0.0);
    for (arma::uword i = 0; i < n_non; ++i)
        pdp_non[dn[i]] += pn[i];

    // Collect non-zero non-interpolated taps
    std::vector<unsigned> dB;
    std::vector<double> dB_power;
    for (unsigned k = 0; k <= max_idx; ++k)
    {
        if (pdp_non[k] > 0.0)
        {
            dB.push_back(k);
            dB_power.push_back(pdp_non[k]);
        }
    }

    arma::uword eff_max = (max_taps == 0) ? (arma::uword)(max_idx + 1) : max_taps;

    // If non-interp taps fill or exceed budget, or no interp data — pick strongest non-interp
    if (dB.size() >= eff_max || n_int == 0)
    {
        if (dB.size() > eff_max)
        {
            std::vector<arma::uword> idx(dB.size());
            for (arma::uword i = 0; i < idx.size(); ++i)
                idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](arma::uword a, arma::uword b)
                      { return dB_power[a] > dB_power[b]; });
            idx.resize(eff_max);
            std::sort(idx.begin(), idx.end());
            std::vector<unsigned> result(eff_max);
            for (arma::uword i = 0; i < eff_max; ++i)
                result[i] = dB[idx[i]];
            return result;
        }
        return dB;
    }

    // Accumulate PDP for interpolated taps
    std::vector<double> pdp_int(max_idx + 1, 0.0);
    for (arma::uword i = 0; i < n_int; ++i)
        pdp_int[di[i]] += pi[i];

    // Collect all non-zero interpolated taps
    std::vector<unsigned> D;
    for (unsigned k = 0; k <= max_idx; ++k)
    {
        if (pdp_int[k] > 0.0)
            D.push_back(k);
    }

    if (D.size() <= eff_max)
        return D;

    // Too many — keep all non-interp, fill remainder with strongest interp-only
    std::vector<unsigned> interp_only;
    std::vector<double> interp_only_pwr;
    for (arma::uword i = 0; i < D.size(); ++i)
    {
        bool is_non = false;
        for (arma::uword j = 0; j < dB.size(); ++j)
            if (D[i] == dB[j])
            {
                is_non = true;
                break;
            }
        if (!is_non)
        {
            interp_only.push_back(D[i]);
            interp_only_pwr.push_back(pdp_int[D[i]]);
        }
    }

    arma::uword n_extra = eff_max - dB.size();
    if (n_extra > (arma::uword)interp_only.size())
        n_extra = (arma::uword)interp_only.size();

    std::vector<arma::uword> idx2(interp_only.size());
    for (arma::uword i = 0; i < (arma::uword)idx2.size(); ++i)
        idx2[i] = i;
    std::sort(idx2.begin(), idx2.end(), [&](arma::uword a, arma::uword b)
              { return interp_only_pwr[a] > interp_only_pwr[b]; });
    idx2.resize(n_extra);

    std::vector<unsigned> result = dB;
    for (arma::uword i = 0; i < n_extra; ++i)
        result.push_back(interp_only[idx2[i]]);
    std::sort(result.begin(), result.end());
    return result;
}

// --- Main implementation ---
template <typename dtype>
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing,
    arma::uword max_no_taps,
    dtype power_exponent,
    int fix_taps)
{
    // --- Input validation ---
    if (coeff_re == nullptr)
        throw std::invalid_argument("Input 'coeff_re' must not be NULL.");
    if (coeff_im == nullptr)
        throw std::invalid_argument("Input 'coeff_im' must not be NULL.");
    if (delay == nullptr)
        throw std::invalid_argument("Input 'delay' must not be NULL.");
    if (coeff_re_quant == nullptr)
        throw std::invalid_argument("Output 'coeff_re_quant' must not be NULL.");
    if (coeff_im_quant == nullptr)
        throw std::invalid_argument("Output 'coeff_im_quant' must not be NULL.");
    if (delay_quant == nullptr)
        throw std::invalid_argument("Output 'delay_quant' must not be NULL.");

    arma::uword n_snap = (arma::uword)coeff_re->size();
    if (n_snap == 0)
        throw std::invalid_argument("Input 'coeff_re' must not be empty.");
    if ((arma::uword)coeff_im->size() != n_snap)
        throw std::invalid_argument("Inputs 'coeff_re' and 'coeff_im' must have the same length.");
    if ((arma::uword)delay->size() != n_snap)
        throw std::invalid_argument("Inputs 'coeff_re' and 'delay' must have the same length.");
    if (tap_spacing <= (dtype)0)
        throw std::invalid_argument("Input 'tap_spacing' must be positive.");
    if (power_exponent <= (dtype)0)
        throw std::invalid_argument("Input 'power_exponent' must be positive.");
    if (fix_taps < 0 || fix_taps > 3)
        throw std::invalid_argument("Input 'fix_taps' must be 0, 1, 2, or 3.");

    arma::uword n_rx = coeff_re->at(0).n_rows;
    arma::uword n_tx = coeff_re->at(0).n_cols;
    arma::uword n_ant = n_rx * n_tx;

    if (n_rx == 0 || n_tx == 0)
        throw std::invalid_argument("Input coefficient cubes must not be empty.");

    // Per-snapshot path counts and cumulative offsets for flat indexing
    std::vector<arma::uword> n_path_vec(n_snap);
    std::vector<arma::uword> offset(n_snap + 1, 0);

    // Determine shared_delays from first snapshot with paths > 0
    bool shared_delays = false;
    bool shared_delays_set = false;

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        arma::uword n_p = coeff_re->at(s).n_slices;
        n_path_vec[s] = n_p;

        if (n_p > 0)
        {
            if (coeff_re->at(s).n_rows != n_rx || coeff_re->at(s).n_cols != n_tx)
                throw std::invalid_argument("All cubes in 'coeff_re' must have the same number of rows and columns.");
            if (coeff_im->at(s).n_rows != n_rx || coeff_im->at(s).n_cols != n_tx || coeff_im->at(s).n_slices != n_p)
                throw std::invalid_argument("All cubes in 'coeff_im' must have the same size as 'coeff_re'.");

            bool snap_shared = (delay->at(s).n_rows == 1 && delay->at(s).n_cols == 1);
            if (!shared_delays_set)
            {
                shared_delays = snap_shared;
                shared_delays_set = true;
            }
            if (snap_shared)
            {
                if (delay->at(s).n_rows != 1 || delay->at(s).n_cols != 1 || delay->at(s).n_slices != n_p)
                    throw std::invalid_argument("All cubes in 'delay' must have consistent size.");
            }
            else
            {
                if (delay->at(s).n_rows != n_rx || delay->at(s).n_cols != n_tx || delay->at(s).n_slices != n_p)
                    throw std::invalid_argument("All cubes in 'delay' must have the same size as 'coeff_re'.");
            }
        }

        offset[s + 1] = offset[s] + n_ant * n_p;
    }

    dtype inv_tap = (dtype)1.0 / tap_spacing;
    arma::uword n_total = offset[n_snap];

    // --- Step 1: Compute tap indices, fractional offsets, detect already-quantized ---
    std::vector<unsigned> Dn(n_total);    // Rounded (nearest) tap index
    std::vector<unsigned> Di_lo(n_total); // Floor tap index
    std::vector<unsigned> Di_hi(n_total); // Ceil tap index
    std::vector<dtype> frac_val(n_total); // Fractional offset [0,1)

    bool already_quantized = true;

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        const dtype *dl_ptr = delay->at(s).memptr();
        arma::uword n_p = n_path_vec[s];
        arma::uword base = offset[s];

        for (arma::uword t = 0; t < n_tx; ++t)
        {
            for (arma::uword r = 0; r < n_rx; ++r)
            {
                for (arma::uword p = 0; p < n_p; ++p)
                {
                    arma::uword ci = base + r + t * n_rx + p * n_ant;
                    arma::uword dli = shared_delays ? p : (r + t * n_rx + p * n_ant);
                    dtype val = dl_ptr[dli] * inv_tap;
                    dtype fl = std::floor(val);
                    dtype fr = val - fl;
                    unsigned fl_u = (unsigned)fl;

                    Dn[ci] = (fr >= (dtype)0.5) ? fl_u + 1 : fl_u;
                    Di_lo[ci] = fl_u;
                    Di_hi[ci] = fl_u + 1;
                    frac_val[ci] = fr;

                    if (fr > (dtype)0.01 && fr < (dtype)0.99)
                        already_quantized = false;
                }
            }
        }
    }

    // --- Step 2: Compute powers and interpolated weights ---
    std::vector<double> Pn(n_total);
    std::vector<double> Pi_lo_pwr, Pi_hi_pwr;
    std::vector<dtype> Ci_re_lo, Ci_im_lo, Ci_re_hi, Ci_im_hi;

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        const dtype *re = coeff_re->at(s).memptr();
        const dtype *im = coeff_im->at(s).memptr();
        arma::uword n_coeff_s = n_ant * n_path_vec[s];
        arma::uword base = offset[s];
        for (arma::uword i = 0; i < n_coeff_s; ++i)
            Pn[base + i] = (double)(re[i] * re[i] + im[i] * im[i]);
    }

    if (!already_quantized)
    {
        Pi_lo_pwr.resize(n_total);
        Pi_hi_pwr.resize(n_total);
        Ci_re_lo.resize(n_total);
        Ci_im_lo.resize(n_total);
        Ci_re_hi.resize(n_total);
        Ci_im_hi.resize(n_total);

        for (arma::uword s = 0; s < n_snap; ++s)
        {
            const dtype *re = coeff_re->at(s).memptr();
            const dtype *im = coeff_im->at(s).memptr();
            arma::uword n_coeff_s = n_ant * n_path_vec[s];
            arma::uword base = offset[s];

            for (arma::uword i = 0; i < n_coeff_s; ++i)
            {
                arma::uword gi = base + i;
                dtype fr = frac_val[gi];
                dtype w_lo = std::pow((dtype)1.0 - fr, power_exponent);
                dtype w_hi = std::pow(fr, power_exponent);

                Ci_re_lo[gi] = w_lo * re[i];
                Ci_im_lo[gi] = w_lo * im[i];
                Ci_re_hi[gi] = w_hi * re[i];
                Ci_im_hi[gi] = w_hi * im[i];

                double p = (double)(re[i] * re[i] + im[i] * im[i]);
                Pi_lo_pwr[gi] = (double)(w_lo * w_lo) * p;
                Pi_hi_pwr[gi] = (double)(w_hi * w_hi) * p;
            }
        }
    }

    // --- Step 3: Compute optimal delay grids ---
    // Lambda: collect data for a subset of (antenna, snapshot) pairs and call find_optimal_delays
    auto compute_grid = [&](const std::vector<std::pair<arma::uword, arma::uword>> &pairs) -> std::vector<unsigned>
    {
        // Count total entries (variable n_path per snapshot)
        arma::uword nn = 0;
        for (arma::uword k = 0; k < (arma::uword)pairs.size(); ++k)
            nn += n_path_vec[pairs[k].second];

        std::vector<unsigned> dn_sub(nn);
        std::vector<double> pn_sub(nn);
        std::vector<unsigned> di_sub;
        std::vector<double> pi_sub;

        if (!already_quantized)
        {
            di_sub.resize(2 * nn);
            pi_sub.resize(2 * nn);
        }

        arma::uword si = 0;
        for (arma::uword k = 0; k < (arma::uword)pairs.size(); ++k)
        {
            arma::uword a = pairs[k].first;  // linear antenna index
            arma::uword s = pairs[k].second; // snapshot index
            arma::uword r = a % n_rx;
            arma::uword t = a / n_rx;
            arma::uword n_p = n_path_vec[s];
            arma::uword base = offset[s];

            for (arma::uword p = 0; p < n_p; ++p)
            {
                arma::uword ci = base + r + t * n_rx + p * n_ant;

                dn_sub[si] = Dn[ci];
                pn_sub[si] = Pn[ci];

                if (!already_quantized)
                {
                    di_sub[si] = Di_lo[ci];      // floor
                    di_sub[nn + si] = Di_hi[ci]; // ceil
                    pi_sub[si] = Pi_lo_pwr[ci];
                    pi_sub[nn + si] = Pi_hi_pwr[ci];
                }
                si++;
            }
        }

        return find_optimal_delays(
            dn_sub.data(), pn_sub.data(), nn,
            already_quantized ? nullptr : di_sub.data(),
            already_quantized ? nullptr : pi_sub.data(),
            already_quantized ? 0 : 2 * nn,
            max_no_taps);
    };

    // D_grid[s][a] = sorted vector of tap indices
    std::vector<std::vector<std::vector<unsigned>>> D_grid(n_snap, std::vector<std::vector<unsigned>>(n_ant));

    if (fix_taps == 1)
    {
        // Single grid for all antenna pairs and snapshots
        std::vector<std::pair<arma::uword, arma::uword>> all_pairs;
        all_pairs.reserve(n_snap * n_ant);
        for (arma::uword s = 0; s < n_snap; ++s)
            for (arma::uword a = 0; a < n_ant; ++a)
                all_pairs.push_back({a, s});
        auto grid = compute_grid(all_pairs);
        for (arma::uword s = 0; s < n_snap; ++s)
            for (arma::uword a = 0; a < n_ant; ++a)
                D_grid[s][a] = grid;
    }
    else if (fix_taps == 2)
    {
        // One grid per snapshot, shared across antenna pairs
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            std::vector<std::pair<arma::uword, arma::uword>> pairs;
            pairs.reserve(n_ant);
            for (arma::uword a = 0; a < n_ant; ++a)
                pairs.push_back({a, s});
            auto grid = compute_grid(pairs);
            for (arma::uword a = 0; a < n_ant; ++a)
                D_grid[s][a] = grid;
        }
    }
    else if (fix_taps == 3)
    {
        // One grid per antenna pair, shared across snapshots
        for (arma::uword a = 0; a < n_ant; ++a)
        {
            std::vector<std::pair<arma::uword, arma::uword>> pairs;
            pairs.reserve(n_snap);
            for (arma::uword s = 0; s < n_snap; ++s)
                pairs.push_back({a, s});
            auto grid = compute_grid(pairs);
            for (arma::uword s = 0; s < n_snap; ++s)
                D_grid[s][a] = grid;
        }
    }
    else
    { // fix_taps == 0
        // Each (antenna, snapshot) gets its own grid
        for (arma::uword s = 0; s < n_snap; ++s)
            for (arma::uword a = 0; a < n_ant; ++a)
                D_grid[s][a] = compute_grid({{a, s}});
    }

    // --- Step 4: Determine global n_taps (max across all grids) ---
    arma::uword n_taps = 0;
    for (arma::uword s = 0; s < n_snap; ++s)
        for (arma::uword a = 0; a < n_ant; ++a)
            if ((arma::uword)D_grid[s][a].size() > n_taps)
                n_taps = (arma::uword)D_grid[s][a].size();
    if (n_taps == 0)
        n_taps = 1;

    // --- Step 5: Map coefficients to the delay grid ---
    // Output delays are shared [1,1,n_taps] only when input was shared AND fix_taps is 1 or 2
    bool output_shared = shared_delays && (fix_taps == 1 || fix_taps == 2);

    coeff_re_quant->resize(n_snap);
    coeff_im_quant->resize(n_snap);
    delay_quant->resize(n_snap);

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        arma::uword n_p = n_path_vec[s];

        coeff_re_quant->at(s).zeros(n_rx, n_tx, n_taps);
        coeff_im_quant->at(s).zeros(n_rx, n_tx, n_taps);
        if (output_shared)
            delay_quant->at(s).zeros(1, 1, n_taps);
        else
            delay_quant->at(s).zeros(n_rx, n_tx, n_taps);

        dtype *ore = coeff_re_quant->at(s).memptr();
        dtype *oim = coeff_im_quant->at(s).memptr();
        dtype *odl = delay_quant->at(s).memptr();
        const dtype *ire = coeff_re->at(s).memptr();
        const dtype *iim = coeff_im->at(s).memptr();
        arma::uword gbase = offset[s];

        // Write shared delay values (use antenna 0's grid, which is the same for all)
        if (output_shared)
        {
            const auto &grid = D_grid[s][0];
            for (arma::uword k = 0; k < (arma::uword)grid.size(); ++k)
                odl[k] = (dtype)grid[k] * tap_spacing;
        }

        for (arma::uword a = 0; a < n_ant; ++a)
        {
            arma::uword r = a % n_rx;
            arma::uword t = a / n_rx;
            const auto &grid = D_grid[s][a];
            arma::uword n_grid = (arma::uword)grid.size();

            // Write per-antenna delays
            if (!output_shared)
            {
                for (arma::uword k = 0; k < n_grid; ++k)
                    odl[r + t * n_rx + k * n_ant] = (dtype)grid[k] * tap_spacing;
            }

            // Build lookup: tap_index -> position in grid
            unsigned max_grid_val = 0;
            for (arma::uword k = 0; k < n_grid; ++k)
                if (grid[k] > max_grid_val)
                    max_grid_val = grid[k];

            std::vector<int> tap2pos(max_grid_val + 2, -1); // +2 for ceil safety
            for (arma::uword k = 0; k < n_grid; ++k)
                tap2pos[grid[k]] = (int)k;

            // Map each path to the grid
            for (arma::uword p = 0; p < n_p; ++p)
            {
                arma::uword ci = r + t * n_rx + p * n_ant; // Index in coeff cube
                arma::uword gi = gbase + ci;               // Global flat index

                if (already_quantized)
                {
                    // No interpolation — accumulate at rounded tap
                    unsigned dn_val = Dn[gi];
                    if (dn_val <= max_grid_val && tap2pos[dn_val] >= 0)
                    {
                        arma::uword oi = r + t * n_rx + (arma::uword)tap2pos[dn_val] * n_ant;
                        ore[oi] += ire[ci];
                        oim[oi] += iim[ci];
                    }
                }
                else
                {
                    unsigned lo = Di_lo[gi];
                    unsigned hi = Di_hi[gi];
                    bool lo_in = (lo <= max_grid_val && tap2pos[lo] >= 0);
                    bool hi_in = (hi < (unsigned)tap2pos.size() && tap2pos[hi] >= 0);

                    if (lo_in && hi_in)
                    {
                        // Both floor and ceil are in the grid — use interpolated coefficients
                        arma::uword oi_lo = r + t * n_rx + (arma::uword)tap2pos[lo] * n_ant;
                        arma::uword oi_hi = r + t * n_rx + (arma::uword)tap2pos[hi] * n_ant;
                        ore[oi_lo] += Ci_re_lo[gi];
                        oim[oi_lo] += Ci_im_lo[gi];
                        ore[oi_hi] += Ci_re_hi[gi];
                        oim[oi_hi] += Ci_im_hi[gi];
                    }
                    else
                    {
                        // Fallback — one or both interpolation taps missing; use rounded tap
                        unsigned dn_val = Dn[gi];
                        if (dn_val <= max_grid_val && tap2pos[dn_val] >= 0)
                        {
                            arma::uword oi = r + t * n_rx + (arma::uword)tap2pos[dn_val] * n_ant;
                            ore[oi] += ire[ci];
                            oim[oi] += iim[ci];
                        }
                    }
                }
            }
        }
    }

    // --- Step 6: Remove trailing all-zero taps ---
    arma::uword max_used = 0;
    for (arma::uword s = 0; s < n_snap; ++s)
    {
        const dtype *ore = coeff_re_quant->at(s).memptr();
        const dtype *oim = coeff_im_quant->at(s).memptr();
        for (arma::uword k = n_taps; k > 0; --k)
        {
            for (arma::uword a = 0; a < n_ant; ++a)
            {
                arma::uword oi = a + (k - 1) * n_ant;
                if (ore[oi] != (dtype)0 || oim[oi] != (dtype)0)
                {
                    if (k > max_used)
                        max_used = k;
                    goto next_snap; // Found a non-zero in this snapshot
                }
            }
        }
    next_snap:;
    }
    if (max_used == 0)
        max_used = 1;

    if (max_used < n_taps)
    {
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            arma::Cube<dtype> tmp_re(n_rx, n_tx, max_used);
            arma::Cube<dtype> tmp_im(n_rx, n_tx, max_used);
            std::memcpy(tmp_re.memptr(), coeff_re_quant->at(s).memptr(), n_ant * max_used * sizeof(dtype));
            std::memcpy(tmp_im.memptr(), coeff_im_quant->at(s).memptr(), n_ant * max_used * sizeof(dtype));
            coeff_re_quant->at(s) = std::move(tmp_re);
            coeff_im_quant->at(s) = std::move(tmp_im);

            if (output_shared)
            {
                arma::Cube<dtype> tmp_dl(1, 1, max_used);
                std::memcpy(tmp_dl.memptr(), delay_quant->at(s).memptr(), max_used * sizeof(dtype));
                delay_quant->at(s) = std::move(tmp_dl);
            }
            else
            {
                arma::Cube<dtype> tmp_dl(n_rx, n_tx, max_used);
                std::memcpy(tmp_dl.memptr(), delay_quant->at(s).memptr(), n_ant * max_used * sizeof(dtype));
                delay_quant->at(s) = std::move(tmp_dl);
            }
        }
    }
}

// --- Explicit template instantiation ---
template void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<float>> *coeff_re,
    const std::vector<arma::Cube<float>> *coeff_im,
    const std::vector<arma::Cube<float>> *delay,
    std::vector<arma::Cube<float>> *coeff_re_quant,
    std::vector<arma::Cube<float>> *coeff_im_quant,
    std::vector<arma::Cube<float>> *delay_quant,
    float tap_spacing,
    arma::uword max_no_taps,
    float power_exponent,
    int fix_taps);

template void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<double>> *coeff_re,
    const std::vector<arma::Cube<double>> *coeff_im,
    const std::vector<arma::Cube<double>> *delay,
    std::vector<arma::Cube<double>> *coeff_re_quant,
    std::vector<arma::Cube<double>> *coeff_im_quant,
    std::vector<arma::Cube<double>> *delay_quant,
    double tap_spacing,
    arma::uword max_no_taps,
    double power_exponent,
    int fix_taps);
