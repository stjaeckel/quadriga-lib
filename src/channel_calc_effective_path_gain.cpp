// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_channel.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# channel_calc_effective_path_gain
Calculate the effective path gain per snapshot in linear scale

- Sums power over all paths and TX/RX antenna pairs to produce one gain value per snapshot
- Uses `coeff_re`/`coeff_im` if available; falls back to `polmat` assuming ideal XPOL antennas
- Throws if neither coefficients nor polarization data are present, or with the `is_valid()` message of an invalid
  channel

## Declaration:
```
arma::Col<dtype> quadriga_lib::channel_calc_effective_path_gain(const channel<dtype> &chan);
```

## Input:
- **`chan`** — Channel object, see [[channel]]

## Returns:
- Effective path gains in linear scale, one entry per snapshot; `[n_snap]`
MD!*/

template <typename dtype>
arma::Col<dtype> quadriga_lib::channel_calc_effective_path_gain(const channel<dtype> &chan)
{
    arma::uword n_snap_arma = chan.n_snap();
    size_t n_snap_vector = (size_t)n_snap_arma;

    if (n_snap_arma == 0ULL)
        return arma::Col<dtype>();

    if (chan.coeff_re.empty() && chan.polmat.empty())
        throw std::invalid_argument("Neither coefficients nor polarization are provided.");

    std::string error_message = chan.is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

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

template arma::Col<float> quadriga_lib::channel_calc_effective_path_gain(const channel<float> &chan);
template arma::Col<double> quadriga_lib::channel_calc_effective_path_gain(const channel<double> &chan);
