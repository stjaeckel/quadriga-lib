// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_sos.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_remove_bias
Remove the sample mean of sum-of-sinusoids generators over a set of positions

- One realisation of a spatially correlated random field has a non-zero mean over a finite set of
  positions. Its variance over the realisations is the mean of the autocorrelation function (ACF) over all
  pairs of positions, which is much larger than `1 / n_pos` when the positions lie within a few
  decorrelation distances of each other. This is a property of any correlated field, not an error of the
  generator
- This function adds one offset to all phases of a generator such that the mean of the Normal distributed
  values of [[sos_val]] over the given positions becomes zero
- Closed form, no iteration: with `z_n = mean over the positions of exp( j·2·pi · f_n·x )` and
  `Z = sum_n z_n · exp( j·phi_n )`, the mean is `sqrt(2/n_coeff) · Re( exp( j·offset ) · Z )`, which
  vanishes for `offset = pi/2 - arg(Z)` and for that value minus pi; the one with the smaller magnitude is
  applied, so the offset lies between -pi/2 and pi/2
- One offset per generator; with two devices (`pos_b` given) it zeroes the mean of the two-device value of
  [[sos_val]] with the same `spatial_freq_scale_b` and `reciprocal`
- Two reciprocal devices at a position where the normalization of [[sos_val]] is singular (`c = -1`) throw
- Effect on the statistics: the result is the field conditioned on a zero mean over the positions. The
  variance at a position drops from 1 to `1 - c^2 / v`, where `v` is the variance of the sample mean and
  `c` the mean ACF between that position and all positions; on average over the positions the loss is at
  least `v`. It is small only if the positions spread over many decorrelation distances. For positions
  within about one decorrelation distance most of the variance is removed
- Rule of thumb, measured for 200 positions in a square with the edge `L` and the decorrelation distance
  `d` (`Comb256`): the variance drops by 74 % for `L = d`, 44 % for `L = 2·d`, 14 % for `L = 5·d`, 5 % for
  `L = 10·d`, 2.5 % for `L = 20·d` and less than 1 % from `L = 50·d` on
- The zero mean holds for exactly these positions and frequencies. It is lost when the phases are used
  with other frequencies (another decorrelation distance) or positions
- The mean of the Uniform output of [[sos_val]] is not exactly 0.5 afterwards
- Phases are returned in the interval from -pi (included) to pi (excluded). A generator whose `abs(Z)` is
  below `1e-14 · n_coeff` already has a zero mean for every offset and is left unchanged, also if its
  phases lie outside that interval
- The sums over the positions are computed in parallel over the sinusoids; the result does not depend
  on the number of threads

## Declaration:
```
void quadriga_lib::sos_remove_bias(
    const arma::Mat<dtype> &spatial_freq,
    arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    const arma::Mat<dtype> &pos_b = arma::Mat<dtype>(),
    double spatial_freq_scale_b = 1.0,
    bool reciprocal = true,
    arma::Mat<dtype> *offset = nullptr,
    arma::Mat<dtype> *mean_before = nullptr);
```

## Inputs:
- **`spatial_freq`** — Spatial frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; updated in-place; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `spatial_freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos, 1]`, `[n_pos, 2]` or `[n_pos, 3]` as for `pos`
- **`spatial_freq_scale_b`** *(optional)* — Spatial-frequency scale of the second device as in [[sos_val]]; must be > 0;
  other than 1 requires `pos_b` and `reciprocal = false`
- **`reciprocal`** *(optional)* — Two devices: the mode of [[sos_val]]

## Outputs:
- **`offset`** *(optional)* — Offsets added to the phases in [rad]; `[n_gen, 1]`
- **`mean_before`** *(optional)* — Sample mean of the values of [[sos_val]] for the given arguments before the
  correction; `[n_gen, 1]`

## See also:
- [[sos_val]] (the values whose mean is removed)
MD!*/

template <typename dtype>
void quadriga_lib::sos_remove_bias(const arma::Mat<dtype> &spatial_freq,
                                   arma::Mat<dtype> &phase,
                                   const arma::Mat<dtype> &pos,
                                   const arma::Mat<dtype> &pos_b,
                                   double spatial_freq_scale_b_in,
                                   bool reciprocal,
                                   arma::Mat<dtype> *offset,
                                   arma::Mat<dtype> *mean_before)
{
    const dtype spatial_freq_scale_b = dtype(spatial_freq_scale_b_in);
    const bool two_devices = !pos_b.is_empty();

    if (spatial_freq.n_elem == 0)
        throw std::invalid_argument("Input 'spatial_freq' cannot be empty.");
    if (spatial_freq.n_cols > 3)
        throw std::invalid_argument("Input 'spatial_freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != spatial_freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'spatial_freq'.");
    if (pos.n_elem == 0)
        throw std::invalid_argument("Input 'pos' cannot be empty.");
    if (pos.n_cols > spatial_freq.n_cols)
        throw std::invalid_argument("Input 'pos' cannot have more columns than 'spatial_freq'.");
    if (two_devices)
    {
        if (pos_b.n_cols > spatial_freq.n_cols)
            throw std::invalid_argument("Input 'pos_b' cannot have more columns than 'spatial_freq'.");
        if (pos_b.n_rows != pos.n_rows)
            throw std::invalid_argument("Number of rows in 'pos_b' does not match number of rows in 'pos'.");
    }
    if (!(spatial_freq_scale_b > (dtype)0.0) || !std::isfinite((double)spatial_freq_scale_b))
        throw std::invalid_argument("Input 'spatial_freq_scale_b' must be finite and > 0.");
    if (spatial_freq_scale_b != (dtype)1.0 && !two_devices)
        throw std::invalid_argument("Input 'spatial_freq_scale_b' other than 1 requires 'pos_b'.");
    if (spatial_freq_scale_b != (dtype)1.0 && reciprocal)
        throw std::invalid_argument("Input 'spatial_freq_scale_b' other than 1 requires 'reciprocal = false'.");

    // 'phase' is updated in place, 'offset' and 'mean_before' are written
    qd_check_alias(&phase, "phase", &spatial_freq, &pos, &pos_b);
    qd_check_alias(offset, "offset", &spatial_freq, &phase, &pos, &pos_b);
    qd_check_alias(mean_before, "mean_before", &spatial_freq, &phase, &pos, &pos_b);
    qd_check_alias_outputs(offset, "offset", mean_before, "mean_before");

    const arma::uword n_gen = phase.n_cols, n_coeff = spatial_freq.n_rows, n_pos = pos.n_rows;
    const double two_pi = 6.283185307179586, pi = 3.141592653589793, amp = std::sqrt(2.0 / (double)n_coeff);

    // Positions as [n_pos, 3] in double, missing columns 0
    auto pos3 = [&](const arma::Mat<dtype> &p)
    {
        arma::mat q(n_pos, 3, arma::fill::zeros);
        for (arma::uword c = 0; c < p.n_cols; ++c)
            for (arma::uword i = 0; i < n_pos; ++i)
                q.at(i, c) = (double)p.at(i, c);
        return q;
    };

    std::vector<double> fx, fy, fz, gx, gy, gz;
    // Frequencies {x, y, z} in double: f of the first device; with two devices g = b·f_p(n) of the second (as sos_val)
    qd_sos_split_freq(spatial_freq, 1.0, fx, fy, fz);
    const arma::mat A = pos3(pos), B = two_devices ? pos3(pos_b) : arma::mat();
    std::vector<arma::uword> perm;
    std::vector<double> w(n_pos, 1.0 / (double)n_pos);
    if (two_devices)
    {
        perm = qd_sos_permutation(n_coeff);
        qd_sos_split_freq(spatial_freq, (double)spatial_freq_scale_b, gx, gy, gz, &perm);
        if (reciprocal)
        {
            std::vector<double> dfx(n_coeff), dfy(n_coeff), dfz(n_coeff);
            for (arma::uword n = 0; n < n_coeff; ++n)
                dfx[n] = fx[n] - gx[n], dfy[n] = fy[n] - gy[n], dfz[n] = fz[n] - gz[n];
            for (arma::uword i = 0; i < n_pos; ++i)
            {
                w[i] = qd_sos_pair_weight(dfx.data(), dfy.data(), dfz.data(), n_coeff, A.at(i, 0) - B.at(i, 0),
                                          A.at(i, 1) - B.at(i, 1), A.at(i, 2) - B.at(i, 2)) / (double)n_pos;
                if (std::isnan(w[i]))
                    throw std::invalid_argument("Inputs 'pos' and 'pos_b' give a singular normalization (c = -1) for 'reciprocal = true'; use more sinusoids.");
            }
        }
    }

    // z_n = sum over the positions of w · exp( j·2·pi·t_n ), t_n = f_n·a (+ g_n·b, + the exchanged term for reciprocal)
    // Parallel over the sinusoids: each z_n is summed by one thread in the order of the positions
    std::vector<double> zr(n_coeff, 0.0), zi(n_coeff, 0.0);
    const long long n_coeff_ll = (long long)n_coeff;
    const bool recip = two_devices && reciprocal;
#pragma omp parallel for if (n_coeff * n_pos > 4096)
    for (long long i_coeff = 0; i_coeff < n_coeff_ll; ++i_coeff)
    {
        const arma::uword n = (arma::uword)i_coeff;
        double re = 0.0, im = 0.0;
        for (arma::uword i = 0; i < n_pos; ++i)
        {
            const double ta = fx[n] * A.at(i, 0) + fy[n] * A.at(i, 1) + fz[n] * A.at(i, 2);
            double t = ta;
            if (two_devices)
                t += gx[n] * B.at(i, 0) + gy[n] * B.at(i, 1) + gz[n] * B.at(i, 2);
            t -= std::floor(t + 0.5);
            re += w[i] * std::cos(two_pi * t), im += w[i] * std::sin(two_pi * t);
            if (recip) // s(b, a): f_p(n) at a, f_n at b
            {
                double u = gx[n] * A.at(i, 0) + gy[n] * A.at(i, 1) + gz[n] * A.at(i, 2) + fx[n] * B.at(i, 0) + fy[n] * B.at(i, 1) + fz[n] * B.at(i, 2);
                u -= std::floor(u + 0.5);
                re += w[i] * std::cos(two_pi * u), im += w[i] * std::sin(two_pi * u);
            }
        }
        zr[n] = re, zi[n] = im;
    }

    // Offset for each generator from (zr, zi), applied to the phases
    arma::vec off(n_gen), mean(n_gen);
    const long long n_gen_ll = (long long)n_gen;
#pragma omp parallel for if (n_coeff * n_gen > 4096)
    for (long long i_gen = 0; i_gen < n_gen_ll; ++i_gen)
    {
        const arma::uword g = (arma::uword)i_gen;
        dtype *p = phase.colptr(g);
        double Zr = 0.0, Zi = 0.0;
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            double c = std::cos((double)p[n]), s = std::sin((double)p[n]);
            Zr += zr[n] * c - zi[n] * s, Zi += zr[n] * s + zi[n] * c;
        }
        mean.at(g) = amp * Zr;

        double delta = 0.0;
        if (std::sqrt(Zr * Zr + Zi * Zi) >= 1.0e-14 * (double)n_coeff)
        {
            // Zero of Re( exp( j·delta ) · Z ) with the smaller magnitude
            delta = std::remainder(0.5 * pi - std::atan2(Zi, Zr), pi);
            for (arma::uword n = 0; n < n_coeff; ++n)
            {
                dtype v = (dtype)std::remainder((double)p[n] + delta, two_pi);
                p[n] = (v >= (dtype)pi) ? -(dtype)pi : v;
            }
        }
        off.at(g) = delta;
    }

    if (offset != nullptr)
        *offset = arma::conv_to<arma::Mat<dtype>>::from(arma::mat(off));
    if (mean_before != nullptr)
        *mean_before = arma::conv_to<arma::Mat<dtype>>::from(arma::mat(mean));
}

template void quadriga_lib::sos_remove_bias(const arma::Mat<float> &spatial_freq, arma::Mat<float> &phase, const arma::Mat<float> &pos, const arma::Mat<float> &pos_b,
                                            double spatial_freq_scale_b, bool reciprocal, arma::Mat<float> *offset, arma::Mat<float> *mean_before);
template void quadriga_lib::sos_remove_bias(const arma::Mat<double> &spatial_freq, arma::Mat<double> &phase, const arma::Mat<double> &pos, const arma::Mat<double> &pos_b,
                                            double spatial_freq_scale_b, bool reciprocal, arma::Mat<double> *offset, arma::Mat<double> *mean_before);
