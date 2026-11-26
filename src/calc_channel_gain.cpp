// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# calc_channel_gain
Calculate the channel gain of each snapshot, the power summed over all paths and antenna pairs

- With coefficients: $G = \sum_{r,t,l} \lvert a_{r,t,l} \rvert^2$ over all receive and transmit antennas and paths
- Without coefficients, from `polmat` for ideal dual-polarized antennas (V and H, unit gain) at both ends:
  $G = \sum_l g_l \sum_{i} \lvert M_{l,i} \rvert^2$ over the four entries of the polarization transfer matrix of each
  path; $g_l$ is `path_gain`, else the free-space gain
  $10^{-(32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d_l / \mathrm{m})) / 10}$ from `path_length` $d_l$ at
  `center_freq` $f$ (per snapshot if it has `n_snap` entries, 1 GHz if empty), else 1

## Declaration:
```
arma::Col<dtype> quadriga_lib::calc_channel_gain(const channel<dtype> &chan);
```

## Input:
- **`chan`** — Channel, valid (see [[channel]]), with `coeff_re` / `coeff_im` or `polmat`; a channel without snapshots
  gives an empty result

## Returns:
- Channel gain, linear, 0 for a snapshot without paths; `[n_snap]`

## See also:
- [[calc_xpr]] (path gain and XPR from `polmat`)
- [[calc_rician_k_factor]] (gain split into LOS and NLOS power)
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::calc_channel_gain(const channel<dtype> &chan)
{
    arma::uword n_snap_arma = chan.n_snap();
    size_t n_snap_vector = (size_t)n_snap_arma;

    if (n_snap_arma == 0ULL)
        return arma::Col<dtype>();

    if (chan.coeff_re.empty() && chan.polmat.empty())
        throw std::invalid_argument("Input 'chan' must contain 'coeff_re' or 'polmat'.");

    std::string error_message = chan.is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument("Input 'chan' is not valid: " + error_message);

    bool use_coeff = !chan.coeff_re.empty();
    bool has_path_gain = !chan.path_gain.empty();
    bool has_path_length = !chan.path_length.empty();
    bool has_multi_freq = chan.center_freq.n_elem == n_snap_arma;

    dtype fGHz_PG = !chan.center_freq.is_empty() ? chan.center_freq.at(0) * (dtype)1.0e-9 : (dtype)1.0;
    fGHz_PG = -(dtype)32.45 - (dtype)20.0 * std::log10(fGHz_PG);

    auto n_paths = chan.n_path();
    auto p_path = n_paths.memptr();
    arma::Col<dtype> PG(n_snap_arma);
    auto pPG = PG.memptr();

    for (size_t iS = 0; iS < n_snap_vector; ++iS)
        if (p_path[iS] != 0)
        {
            dtype p = (dtype)0.0;
            if (use_coeff)
            {
                for (auto &a : chan.coeff_re[iS])
                    p += a * a;
                for (auto &a : chan.coeff_im[iS])
                    p += a * a;
            }
            else // use polarization
            {
                const dtype *p_pol = chan.polmat[iS].memptr();
                const dtype *p_gain = has_path_gain ? chan.path_gain[iS].memptr() : nullptr;
                const dtype *p_length = has_path_length ? chan.path_length[iS].memptr() : nullptr;

                if (has_multi_freq)
                    fGHz_PG = -(dtype)32.45 - (dtype)20.0 * std::log10(chan.center_freq.at(iS) * (dtype)1.0e-9);

                for (arma::uword iP = 0ULL; iP < p_path[iS]; ++iP)
                {
                    dtype gain = (dtype)1.0;
                    if (has_path_gain) // Use provided gain
                        gain = std::sqrt(p_gain[iP]);
                    else if (has_path_length) // Calculate FSPL
                    {
                        gain = fGHz_PG - (dtype)20.0 * std::log10(p_length[iP]);
                        gain = std::pow((dtype)10.0, (dtype)0.1 * gain);
                        gain = std::sqrt(gain);
                    }
                    for (unsigned j = 0; j < 8; ++j)
                    {
                        dtype v = gain * p_pol[8 * iP + j];
                        p += v * v;
                    }
                }
            }
            pPG[iS] = p;
        }

    return PG;
}

template arma::Col<float> quadriga_lib::calc_channel_gain(const channel<float> &chan);
template arma::Col<double> quadriga_lib::calc_channel_gain(const channel<double> &chan);
