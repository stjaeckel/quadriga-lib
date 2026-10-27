// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

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
- One device (`pos_b` omitted): one offset per generator
- Two devices on the same map (`pos_b` given, `phase_b` omitted): one offset per generator that zeroes the
  mean of the combined, normalised value; `acf` and `dist` are required
- Two devices on different maps (`pos_b` and `phase_b` given): each device is treated on its own, `phase`
  with `pos` and `phase_b` with `pos_b` and the frequencies `freq · freq_scale_b`; two offsets per generator.
  The mean of the combined value is then zero as well. `phase_b` cannot share memory with `phase`: one
  matrix cannot hold two corrections
- The arguments that are written, `phase`, `phase_b`, `offset` and `mean_before`, cannot share memory with
  another argument, not even in part, and `offset` and `mean_before` cannot be the same object. Such a call
  throws and leaves all arguments unchanged
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
- Two devices on the same map need an ACF above -1 at all sample points, as in [[sos_val]]
- The sums over the positions are computed in parallel over the sinusoids; the result does not depend
  on the number of threads

## Declaration:
```
void quadriga_lib::sos_remove_bias(
    const arma::Mat<dtype> &freq,
    arma::Mat<dtype> &phase,
    const arma::Mat<dtype> &pos,
    const arma::Mat<dtype> *pos_b = nullptr,
    arma::Mat<dtype> *phase_b = nullptr,
    const arma::Col<dtype> *acf = nullptr,
    const arma::Col<dtype> *dist = nullptr,
    arma::Mat<dtype> *offset = nullptr,
    arma::Mat<dtype> *mean_before = nullptr,
    dtype freq_scale_b = 1.0);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; updated in-place; `[n_coeff, n_gen]`
- **`pos`** — Positions of the first device in [m]; one row per position; the number of columns sets the
  number of dimensions: `[n_pos, 1]` = `{x}`, `[n_pos, 2]` = `{x, y}` or `[n_pos, 3]` = `{x, y, z}`;
  coordinates that are not given are 0; not more columns than `freq`
- **`pos_b`** *(optional)* — Positions of the second device in [m]; omitted or empty = one device;
  `[n_pos_b, 1]`, `[n_pos_b, 2]` or `[n_pos_b, 3]` as for `pos`; `n_pos_b = n_pos` for the same map, any
  number for different maps
- **`phase_b`** *(optional)* — Phases of the second device in [rad]; requires `pos_b`; omitted or empty =
  same map; updated in-place; no memory shared with `phase`; `[n_coeff, n_gen]`
- **`acf`** *(optional)* — Target ACF; required for two devices on the same map, must then be > -1;
  `[n_acf]`
- **`dist`** *(optional)* — Sample points of the ACF in [m]; must start at 0, increase strictly and be
  finite; `[n_acf]`
- **`freq_scale_b`** *(optional)* — Frequency scale of the second device, i.e. decorrelation distance of
  the first device divided by that of the second; must be > 0; other than 1 requires `phase_b`

## Outputs:
- **`offset`** *(optional)* — Offsets added to the phases in [rad]; `[n_gen, 1]`, or `[n_gen, 2]` with
  `phase_b` (column 0: `phase`, column 1: `phase_b`)
- **`mean_before`** *(optional)* — Sample mean before the correction; `[n_gen, 1]`: mean of the values of
  [[sos_val]] for the given arguments; with `phase_b` `[n_gen, 2]`: mean of the field of each device on its
  own positions, the mean of the combined value being their sum divided by `sqrt(2)`

## See also:
- [[sos_val]] (the values whose mean is removed)
MD!*/

template <typename dtype>
void quadriga_lib::sos_remove_bias(const arma::Mat<dtype> &freq,
                                   arma::Mat<dtype> &phase,
                                   const arma::Mat<dtype> &pos,
                                   const arma::Mat<dtype> *pos_b,
                                   arma::Mat<dtype> *phase_b,
                                   const arma::Col<dtype> *acf,
                                   const arma::Col<dtype> *dist,
                                   arma::Mat<dtype> *offset,
                                   arma::Mat<dtype> *mean_before,
                                   dtype freq_scale_b)
{
    const bool two_devices = pos_b != nullptr && !pos_b->is_empty();
    const bool two_maps = phase_b != nullptr && !phase_b->is_empty();

    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols > 3)
        throw std::invalid_argument("Input 'freq' must have 1, 2 or 3 columns.");
    if (phase.n_elem == 0)
        throw std::invalid_argument("Input 'phase' cannot be empty.");
    if (phase.n_rows != freq.n_rows)
        throw std::invalid_argument("Number of rows in 'phase' does not match number of rows in 'freq'.");
    if (pos.n_elem == 0)
        throw std::invalid_argument("Input 'pos' cannot be empty.");
    if (pos.n_cols > freq.n_cols)
        throw std::invalid_argument("Input 'pos' cannot have more columns than 'freq'.");
    if (two_devices && pos_b->n_cols > freq.n_cols)
        throw std::invalid_argument("Input 'pos_b' cannot have more columns than 'freq'.");
    if (two_maps)
    {
        if (!two_devices)
            throw std::invalid_argument("Input 'phase_b' requires 'pos_b'.");
        if (phase_b->n_rows != phase.n_rows || phase_b->n_cols != phase.n_cols)
            throw std::invalid_argument("Input 'phase_b' must have the same size as 'phase'.");
    }
    if (!(freq_scale_b > (dtype)0.0) || !std::isfinite((double)freq_scale_b))
        throw std::invalid_argument("Input 'freq_scale_b' must be > 0.");
    if (freq_scale_b != (dtype)1.0 && !two_maps)
        throw std::invalid_argument("Input 'freq_scale_b' other than 1 requires 'phase_b'.");
    if (two_devices && !two_maps)
    {
        if (pos_b->n_rows != pos.n_rows)
            throw std::invalid_argument("Number of rows in 'pos_b' does not match number of rows in 'pos'.");
        if (acf == nullptr || dist == nullptr || acf->is_empty() || dist->is_empty())
            throw std::invalid_argument("Inputs 'acf' and 'dist' are required for two devices on the same map.");
        qd_sos_check_acf_same_map(*acf, *dist);
    }

    // 'phase' and 'phase_b' are updated and 'offset' and 'mean_before' written while the other arguments are
    // still read: none of them can share memory with another argument, not even in part
    const arma::Mat<dtype> *args[9] = {&freq, &phase, &pos, pos_b, phase_b, acf, dist, offset, mean_before};
    const char *names[9] = {"freq", "phase", "pos", "pos_b", "phase_b", "acf", "dist", "offset", "mean_before"};
    const bool written[9] = {false, true, false, false, true, false, false, true, true};
    for (int i = 1; i < 9; ++i)
        for (int j = 0; j < i; ++j)
            if ((written[i] || written[j]) && args[i] != nullptr && args[j] != nullptr && qd_sos_share_memory(*args[i], *args[j]))
                throw std::invalid_argument(std::string(i < 7 ? "Input '" : "Output '") + names[i] + "' cannot share memory with '" + names[j] + "'.");
    if (offset != nullptr && offset == mean_before)
        throw std::invalid_argument("Outputs 'offset' and 'mean_before' cannot be the same object.");

    const arma::uword n_gen = phase.n_cols, n_col = two_maps ? 2 : 1;

    const arma::uword n_coeff = freq.n_rows;
    const double two_pi = 6.283185307179586, pi = 3.141592653589793, amp = std::sqrt(2.0 / (double)n_coeff);

    // z_n = sum over the positions of weight · exp( j·2·pi · scale · f_n·x ), added to (zr, zi)
    // Parallel over the sinusoids: each z_n is summed by one thread in the order of the positions
    const long long n_coeff_ll = (long long)n_coeff;
    auto accumulate = [&](const arma::Mat<dtype> &p, double scale, const double *weight, std::vector<double> &zr, std::vector<double> &zi)
    {
        const arma::uword n_rows = p.n_rows, n_cols = p.n_cols;
        const dtype *p0 = p.colptr(0), *p1 = (n_cols > 1) ? p.colptr(1) : nullptr, *p2 = (n_cols > 2) ? p.colptr(2) : nullptr;
        double *pzr = zr.data(), *pzi = zi.data();
#pragma omp parallel for if (n_coeff * n_rows > 4096)
        for (long long i_coeff = 0; i_coeff < n_coeff_ll; ++i_coeff)
        {
            const arma::uword n = (arma::uword)i_coeff;
            double fx = scale * (double)freq.at(n, 0), fy = (freq.n_cols > 1) ? scale * (double)freq.at(n, 1) : 0.0;
            double fz = (freq.n_cols > 2) ? scale * (double)freq.at(n, 2) : 0.0, re = 0.0, im = 0.0;
            for (arma::uword i = 0; i < n_rows; ++i)
            {
                double t = fx * (double)p0[i];
                if (p1 != nullptr)
                    t += fy * (double)p1[i];
                if (p2 != nullptr)
                    t += fz * (double)p2[i];
                t -= std::floor(t + 0.5);
                re += weight[i] * std::cos(two_pi * t), im += weight[i] * std::sin(two_pi * t);
            }
            pzr[n] += re, pzi[n] += im;
        }
    };

    // Offset for each generator of 'ph' from (zr, zi); applies it and reports offset and mean
    arma::mat off(n_gen, n_col), mean(n_gen, n_col);
    const long long n_gen_ll = (long long)n_gen;
    auto correct = [&](arma::Mat<dtype> &ph, const std::vector<double> &zr, const std::vector<double> &zi, arma::uword col)
    {
#pragma omp parallel for if (n_coeff * n_gen > 4096)
        for (long long i_gen = 0; i_gen < n_gen_ll; ++i_gen)
        {
            const arma::uword g = (arma::uword)i_gen;
            dtype *p = ph.colptr(g);
            double Zr = 0.0, Zi = 0.0;
            for (arma::uword n = 0; n < n_coeff; ++n)
            {
                double c = std::cos((double)p[n]), s = std::sin((double)p[n]);
                Zr += zr[n] * c - zi[n] * s, Zi += zr[n] * s + zi[n] * c;
            }
            mean.at(g, col) = amp * Zr;

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
            off.at(g, col) = delta;
        }
    };

    std::vector<double> zr(n_coeff, 0.0), zi(n_coeff, 0.0);
    if (two_devices && !two_maps)
    {
        // Same map: both devices enter with the normalisation of the pair
        const arma::vec acf_d = arma::conv_to<arma::vec>::from(*acf), dist_d = arma::conv_to<arma::vec>::from(*dist);
        std::vector<double> w(pos.n_rows);
        for (arma::uword i = 0; i < pos.n_rows; ++i)
        {
            double d2 = 0.0;
            for (arma::uword c = 0; c < 3; ++c)
            {
                double a = (c < pos.n_cols) ? (double)pos.at(i, c) : 0.0, b = (c < pos_b->n_cols) ? (double)pos_b->at(i, c) : 0.0;
                d2 += (a - b) * (a - b);
            }
            double A = qd_interp_linear(acf_d.memptr(), dist_d.memptr(), acf_d.n_elem, std::sqrt(d2));
            w[i] = 1.0 / (std::sqrt(2.0 + 2.0 * A) * (double)pos.n_rows);
        }
        accumulate(pos, 1.0, w.data(), zr, zi);
        accumulate(*pos_b, 1.0, w.data(), zr, zi);
        correct(phase, zr, zi, 0);
    }
    else
    {
        std::vector<double> w(pos.n_rows, 1.0 / (double)pos.n_rows);
        accumulate(pos, 1.0, w.data(), zr, zi);
        correct(phase, zr, zi, 0);
        if (two_maps)
        {
            std::fill(zr.begin(), zr.end(), 0.0), std::fill(zi.begin(), zi.end(), 0.0);
            w.assign(pos_b->n_rows, 1.0 / (double)pos_b->n_rows);
            accumulate(*pos_b, (double)freq_scale_b, w.data(), zr, zi);
            correct(*phase_b, zr, zi, 1);
        }
    }

    if (offset != nullptr)
    {
        if (offset->n_rows != n_gen || offset->n_cols != n_col)
            offset->set_size(n_gen, n_col);
        for (arma::uword i = 0; i < n_gen * n_col; ++i)
            offset->at(i) = (dtype)off.at(i);
    }
    if (mean_before != nullptr)
    {
        if (mean_before->n_rows != n_gen || mean_before->n_cols != n_col)
            mean_before->set_size(n_gen, n_col);
        for (arma::uword i = 0; i < n_gen * n_col; ++i)
            mean_before->at(i) = (dtype)mean.at(i);
    }
}

template void quadriga_lib::sos_remove_bias(const arma::Mat<float> &freq, arma::Mat<float> &phase, const arma::Mat<float> &pos, const arma::Mat<float> *pos_b,
                                            arma::Mat<float> *phase_b, const arma::Col<float> *acf, const arma::Col<float> *dist, arma::Mat<float> *offset,
                                            arma::Mat<float> *mean_before, float freq_scale_b);
template void quadriga_lib::sos_remove_bias(const arma::Mat<double> &freq, arma::Mat<double> &phase, const arma::Mat<double> &pos, const arma::Mat<double> *pos_b,
                                            arma::Mat<double> *phase_b, const arma::Col<double> *acf, const arma::Col<double> *dist, arma::Mat<double> *offset,
                                            arma::Mat<double> *mean_before, double freq_scale_b);
