// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <climits>
#include <utility>
#include <vector>
#include <stdexcept>

// --- Internal helper: accumulate_pdp ---
// Sums the power per tap index, in the order of the entries, and returns the taps with non-zero power in ascending
// order together with their power. The work does not depend on the absolute delays: a dense vector is used only over
// the index range [d_min, d_max] and only when that range is small compared to the number of entries; otherwise the
// entries are sorted by tap index (sparse). Both give the same sums in the same order.
static void accumulate_pdp(const unsigned *d, const double *p, arma::uword n, unsigned d_min, unsigned d_max,
                           std::vector<unsigned> &taps, std::vector<double> &power)
{
    taps.clear();
    power.clear();
    if (n == 0)
        return;

    arma::uword range = (arma::uword)(d_max - d_min) + 1;
    if (range <= 4 * n + 64)
    {
        std::vector<double> pdp(range, 0.0);
        for (arma::uword i = 0; i < n; ++i)
            pdp[d[i] - d_min] += p[i];
        for (arma::uword k = 0; k < range; ++k)
            if (pdp[k] > 0.0)
            {
                taps.push_back(d_min + (unsigned)k);
                power.push_back(pdp[k]);
            }
    }
    else
    {
        std::vector<std::pair<unsigned, arma::uword>> order(n); // (tap index, entry), sorted: entry order per tap
        for (arma::uword i = 0; i < n; ++i)
            order[i] = {d[i], i};
        std::sort(order.begin(), order.end());
        for (arma::uword i = 0; i < n;)
        {
            unsigned k = order[i].first;
            double sum = 0.0;
            for (; i < n && order[i].first == k; ++i)
                sum += p[order[i].second];
            if (sum > 0.0)
            {
                taps.push_back(k);
                power.push_back(sum);
            }
        }
    }
}

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
    if (n_non == 0 && n_int == 0)
        return {};

    // Index ranges of both arrays
    unsigned min_non = UINT_MAX, max_non = 0, min_int = UINT_MAX, max_int = 0;
    for (arma::uword i = 0; i < n_non; ++i)
    {
        min_non = (dn[i] < min_non) ? dn[i] : min_non;
        max_non = (dn[i] > max_non) ? dn[i] : max_non;
    }
    for (arma::uword i = 0; i < n_int; ++i)
    {
        min_int = (di[i] < min_int) ? di[i] : min_int;
        max_int = (di[i] > max_int) ? di[i] : max_int;
    }
    unsigned max_idx = (max_int > max_non) ? max_int : max_non;

    // Accumulate PDP for non-interpolated taps, collect non-zero taps
    std::vector<unsigned> dB;
    std::vector<double> dB_power;
    accumulate_pdp(dn, pn, n_non, min_non, max_non, dB, dB_power);

    arma::uword eff_max = (max_taps == 0) ? (arma::uword)max_idx + 1 : max_taps;

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

    // Accumulate PDP for interpolated taps, collect all non-zero interpolated taps
    std::vector<unsigned> D;
    std::vector<double> D_power;
    accumulate_pdp(di, pi, n_int, min_int, max_int, D, D_power);

    if (D.size() <= eff_max)
        return D;

    // Too many — keep all non-interp, fill remainder with strongest interp-only (D and dB are both sorted)
    std::vector<unsigned> interp_only;
    std::vector<double> interp_only_pwr;
    for (arma::uword i = 0, j = 0; i < D.size(); ++i)
    {
        while (j < dB.size() && dB[j] < D[i])
            ++j;
        if (j == dB.size() || dB[j] != D[i])
        {
            interp_only.push_back(D[i]);
            interp_only_pwr.push_back(D_power[i]);
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

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1-delta)^alpha and delta^alpha,
  where delta is the fractional offset within the bin and alpha is `power_exponent`; this avoids discontinuities when 
  delays cross tap boundaries
- Use `power_exponent=1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional offsets are below 0.01 or above 0.99, weight computation is skipped but tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`; shared delays are expanded 
  internally when `fix_taps` is 0 or 3, and output delays remain shared `[1, 1, n_taps]` when `fix_taps` is 1 or 2;
  per-antenna input always gives `[n_rx, n_tx, n_taps]` output delays (identical for all pairs when `fix_taps` is 1 or 2)
- `n_rx` and `n_tx` must be identical across all snapshots; `n_path_s` may differ per snapshot
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0 (there is no tap below 0 to
  interpolate with; the error is below `tap_spacing / 2`, as for nearest-tap rounding). Delays at or below
  `-tap_spacing / 2` throw `std::invalid_argument`, as do NaN, Inf and delays of 2^31 taps or more
  (`delay / tap_spacing < 2^31`, e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [[get_channels_spherical]], [[get_channels_multifreq]] and
  [[get_channels_planar]]) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values
- Outputs may alias inputs, also across arguments (e.g. `delay_quant == coeff_re`); all inputs are read before the
  outputs are written. The three outputs must be different objects.

## Declaration:
```
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

## Inputs:
- **`coeff_re`** — Channel coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]`
- **`coeff_im`** — Channel coefficients, imaginary part; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`;
  non-negative with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** *(optional)* — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate
- **`max_no_taps`** *(optional)* — Maximum number of output taps; 0 means unlimited
- **`power_exponent`** *(optional)* — Interpolation exponent alpha; 1.0 = narrowband, 0.5 = wideband power-preserving
- **`fix_taps`** *(optional)* — Delay grid sharing mode:<br><br>
  | Value | Meaning                                                                                                                               |
  | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps]`                                                                     |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), identical for every snapshot |
  | 2     | Per snapshot; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), but each snapshot has its own independent tap grid — taps do not align across snapshots |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps]`                                                             |

## Outputs:
- **`coeff_re_quant`** — Output coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_taps]`
- **`coeff_im_quant`** — Output coefficients, imaginary part; same layout as `coeff_re_quant`
- **`delay_quant`** — Output delays in seconds; each cube `[n_rx, n_tx, n_taps]`, or `[1, 1, n_taps]` for shared delay input with
  `fix_taps` 1 or 2

## Example:
```
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1,1,3); cim[0].set_size(1,1,3); dl[0].set_size(1,1,3);
cre[1].set_size(1,1,2); cim[1].set_size(1,1,2); dl[1].set_size(1,1,2);
dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9; dl[1](0,0,1) = 10.0e-9;
std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```
MD!*/

template <typename dtype>
void quadriga_lib::quantize_delays(const std::vector<arma::Cube<dtype>> *coeff_re,
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
    if (coeff_re_quant == coeff_im_quant || coeff_re_quant == delay_quant || coeff_im_quant == delay_quant)
        throw std::invalid_argument("Outputs 'coeff_re_quant', 'coeff_im_quant' and 'delay_quant' must be different objects.");

    arma::uword n_snap = (arma::uword)coeff_re->size();
    if (n_snap == 0)
        throw std::invalid_argument("Input 'coeff_re' must not be empty.");
    if ((arma::uword)coeff_im->size() != n_snap)
        throw std::invalid_argument("Inputs 'coeff_re' and 'coeff_im' must have the same length.");
    if ((arma::uword)delay->size() != n_snap)
        throw std::invalid_argument("Inputs 'coeff_re' and 'delay' must have the same length.");
    if (!(tap_spacing > (dtype)0) || !std::isfinite(tap_spacing) || !std::isfinite((dtype)1.0 / tap_spacing))
        throw std::invalid_argument("Input 'tap_spacing' must be positive and finite.");
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
            else if (snap_shared != shared_delays)
                throw std::invalid_argument("All snapshots must consistently use shared [1,1,n_path] or per-antenna [n_rx,n_tx,n_path] delays.");

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

    // Valid delays: above -tap_spacing / 2 and below 2^31 taps; this keeps every tap index (and index + 1) within
    // the range of 'unsigned'. Small negative delays (relative delays of elements closer than the phase centre) are
    // mapped to tap 0.
    const dtype delay_min = (dtype)-0.5 * tap_spacing;
    const dtype tap_limit = (dtype)2147483648.0; // 2^31

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
                    dtype dl_val = dl_ptr[dli];
                    dtype val = dl_val * inv_tap;
                    if (!(dl_val > delay_min && val < tap_limit)) // also catches NaN and Inf
                    {
                        if (!std::isfinite(dl_val))
                            throw std::invalid_argument("Input 'delay' contains NaN or Inf values.");
                        if (dl_val <= delay_min)
                            throw std::invalid_argument("Input 'delay' contains a value at or below -tap_spacing / 2; relative delays of large or acoustic arrays can be that negative, use absolute delays (use_absolute_delays = true).");
                        throw std::invalid_argument("Input 'delay' contains a value of 2^31 taps or more (delay / tap_spacing must be below 2^31).");
                    }
                    val = (val < (dtype)0) ? (dtype)0 : val; // -tap_spacing / 2 < delay < 0: tap 0
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

    // The outputs are built in local vectors and moved to the output arguments at the end, after the last read of
    // the inputs, so any output may alias any input
    std::vector<arma::Cube<dtype>> out_re(n_snap), out_im(n_snap), out_dl(n_snap);
    std::vector<int> tap2pos; // Lookup tap index - grid[0] -> position in grid, reused for all grids

    for (arma::uword s = 0; s < n_snap; ++s)
    {
        arma::uword n_p = n_path_vec[s];

        out_re[s].zeros(n_rx, n_tx, n_taps);
        out_im[s].zeros(n_rx, n_tx, n_taps);
        if (output_shared)
            out_dl[s].zeros(1, 1, n_taps);
        else
            out_dl[s].zeros(n_rx, n_tx, n_taps);

        dtype *ore = out_re[s].memptr();
        dtype *oim = out_im[s].memptr();
        dtype *odl = out_dl[s].memptr();
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

            if (n_grid == 0)
                continue;

            // Position of a tap index in the (sorted) grid, -1 if not in the grid: dense table over the grid's index
            // range when that range is small, binary search otherwise
            const unsigned g_min = grid.front();
            const arma::uword g_range = (arma::uword)(grid.back() - g_min) + 1;
            const bool use_table = g_range <= 4 * n_grid + 1024;
            if (use_table)
            {
                tap2pos.assign(g_range, -1);
                for (arma::uword k = 0; k < n_grid; ++k)
                    tap2pos[grid[k] - g_min] = (int)k;
            }
            auto pos_of = [&](unsigned tap) -> int
            {
                if (use_table)
                {
                    if (tap < g_min || (arma::uword)(tap - g_min) >= g_range)
                        return -1;
                    return tap2pos[tap - g_min];
                }
                auto it = std::lower_bound(grid.begin(), grid.end(), tap);
                return (it != grid.end() && *it == tap) ? (int)(it - grid.begin()) : -1;
            };

            // Map each path to the grid
            for (arma::uword p = 0; p < n_p; ++p)
            {
                arma::uword ci = r + t * n_rx + p * n_ant; // Index in coeff cube
                arma::uword gi = gbase + ci;               // Global flat index

                if (already_quantized)
                {
                    // No interpolation — accumulate at rounded tap
                    int pos = pos_of(Dn[gi]);
                    if (pos >= 0)
                    {
                        arma::uword oi = r + t * n_rx + (arma::uword)pos * n_ant;
                        ore[oi] += ire[ci];
                        oim[oi] += iim[ci];
                    }
                }
                else
                {
                    int pos_lo = pos_of(Di_lo[gi]);
                    int pos_hi = pos_of(Di_hi[gi]);

                    if (pos_lo >= 0 && pos_hi >= 0)
                    {
                        // Both floor and ceil are in the grid — use interpolated coefficients
                        arma::uword oi_lo = r + t * n_rx + (arma::uword)pos_lo * n_ant;
                        arma::uword oi_hi = r + t * n_rx + (arma::uword)pos_hi * n_ant;
                        ore[oi_lo] += Ci_re_lo[gi];
                        oim[oi_lo] += Ci_im_lo[gi];
                        ore[oi_hi] += Ci_re_hi[gi];
                        oim[oi_hi] += Ci_im_hi[gi];
                    }
                    else
                    {
                        // Fallback — one or both interpolation taps missing; use rounded tap
                        int pos = pos_of(Dn[gi]);
                        if (pos >= 0)
                        {
                            arma::uword oi = r + t * n_rx + (arma::uword)pos * n_ant;
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
        const dtype *ore = out_re[s].memptr();
        const dtype *oim = out_im[s].memptr();
        for (arma::uword k = n_taps; k > max_used; --k) // upper bound tightens as max_used grows
        {
            for (arma::uword a = 0; a < n_ant; ++a)
            {
                arma::uword oi = a + (k - 1) * n_ant;
                if (ore[oi] != (dtype)0 || oim[oi] != (dtype)0)
                {
                    max_used = k; // k > max_used guaranteed by outer loop condition
                    break;        // no larger k will be checked — outer bound handles it
                }
            }
        }
    }
    if (max_used == 0)
        max_used = 1;

    if (max_used < n_taps)
    {
        for (arma::uword s = 0; s < n_snap; ++s)
        {
            arma::Cube<dtype> tmp_re(n_rx, n_tx, max_used);
            arma::Cube<dtype> tmp_im(n_rx, n_tx, max_used);
            std::memcpy(tmp_re.memptr(), out_re[s].memptr(), n_ant * max_used * sizeof(dtype));
            std::memcpy(tmp_im.memptr(), out_im[s].memptr(), n_ant * max_used * sizeof(dtype));
            out_re[s] = std::move(tmp_re);
            out_im[s] = std::move(tmp_im);

            if (output_shared)
            {
                arma::Cube<dtype> tmp_dl(1, 1, max_used);
                std::memcpy(tmp_dl.memptr(), out_dl[s].memptr(), max_used * sizeof(dtype));
                out_dl[s] = std::move(tmp_dl);
            }
            else
            {
                arma::Cube<dtype> tmp_dl(n_rx, n_tx, max_used);
                std::memcpy(tmp_dl.memptr(), out_dl[s].memptr(), n_ant * max_used * sizeof(dtype));
                out_dl[s] = std::move(tmp_dl);
            }
        }
    }

    // All inputs have been read: assign the outputs
    *coeff_re_quant = std::move(out_re);
    *coeff_im_quant = std::move(out_im);
    *delay_quant = std::move(out_dl);
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
