// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the sources of the sum-of-sinusoids functions (sos_*.cpp)

#ifndef qd_helpers_sos_hpp
#define qd_helpers_sos_hpp

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Validate an ACF table given as (acf, dist), throws std::invalid_argument
template <typename dtype>
static inline void qd_sos_check_acf(const arma::Col<dtype> &acf, const arma::Col<dtype> &dist)
{
    if (acf.n_elem < 2)
        throw std::invalid_argument("Input 'acf' must have at least 2 elements.");
    if (dist.n_elem != acf.n_elem)
        throw std::invalid_argument("Number of elements in 'dist' does not match number of elements in 'acf'.");
    const dtype *p = dist.memptr();
    if (p[0] != (dtype)0.0)
        throw std::invalid_argument("Input 'dist' must start at 0.");
    for (arma::uword i = 1; i < dist.n_elem; ++i)
        if (!(p[i] > p[i - 1]))
            throw std::invalid_argument("Input 'dist' must increase strictly.");
    if (!std::isfinite((double)p[dist.n_elem - 1]))
        throw std::invalid_argument("Input 'dist' must be finite.");
    for (arma::uword i = 0; i < acf.n_elem; ++i)
        if (!std::isfinite((double)acf.at(i)))
            throw std::invalid_argument("Input 'acf' must be finite.");
}

// Frequencies as separate x, y, z vectors in double, scaled; missing dimensions are 0; with 'perm', row perm[n] goes
// to element n
template <typename dtype>
static inline void qd_sos_split_freq(const arma::Mat<dtype> &spatial_freq, double scale,
                              std::vector<double> &fx, std::vector<double> &fy, std::vector<double> &fz,
                              const std::vector<arma::uword> *perm = nullptr)
{
    arma::uword n_coeff = spatial_freq.n_rows;
    fx.assign(n_coeff, 0.0), fy.assign(n_coeff, 0.0), fz.assign(n_coeff, 0.0);
    for (arma::uword n = 0; n < n_coeff; ++n)
    {
        const arma::uword r = perm ? (*perm)[n] : n;
        fx[n] = scale * (double)spatial_freq.at(r, 0);
        if (spatial_freq.n_cols > 1)
            fy[n] = scale * (double)spatial_freq.at(r, 1);
        if (spatial_freq.n_cols > 2)
            fz[n] = scale * (double)spatial_freq.at(r, 2);
    }
}

// Fixed permutation p of the sinusoids for two devices of sos_val, sos_remove_bias and gbsm_draw: the second device uses
// frequency p(n) in sinusoid n; one cycle through all sinusoids (no fixed point for n_coeff >= 2), a function of n_coeff
// only (defined in sos_val.cpp)
std::vector<arma::uword> qd_sos_permutation(arma::uword n_coeff);

// Normalization of the reciprocal two-device value in double precision, as used by sos_remove_bias (sos_val computes c
// with its kernel): 1 / sqrt( 2 + 2·c ), c = (1/N) · sum_n cos( 2·pi·df_n·d ), df_n = f_n - f_p(n), d = (dx, dy, dz) =
// first minus second position; exactly even in d; NaN where 2 + 2·c <= 1e-6 (c = -1, tables with few sinusoids), as
// in sos_val
static inline double qd_sos_pair_weight(const double *dfx, const double *dfy, const double *dfz, size_t n_coeff,
                                        double dx, double dy, double dz)
{
    double c = 0.0;
    for (size_t n = 0; n < n_coeff; ++n)
    {
        const double t = std::remainder(dfx[n] * dx + dfy[n] * dy + dfz[n] * dz, 1.0);
        c += std::cos(6.283185307179586 * std::abs(t));
    }
    const double v = 2.0 + 2.0 * c / (double)n_coeff;
    return (v > 1.0e-6) ? 1.0 / std::sqrt(v) : std::numeric_limits<double>::quiet_NaN();
}

// Type and number of sinusoids of a preset of sos_preset (defined in sos_preset.cpp)
// type: 0 = Exp, 1 = Gauss, 2 = Comb; returns false for a name that is not a preset
bool qd_sos_preset_lookup(const std::string &name, int &type, arma::uword &n_coeff);

// Check that the sample points of an ACF table are equally spaced
// Tolerance: 1e-4 of the step, but not less than 4 spacings of dtype at the last sample point, which is
// what the rounding of the sample points themselves can amount to in a long single-precision table
template <typename dtype>
static inline bool qd_sos_equally_spaced(const arma::Col<dtype> &dist)
{
    double step = (double)dist.at(dist.n_elem - 1) / (double)(dist.n_elem - 1);
    double tol = std::max(1.0e-4 * step, 4.0 * (double)std::numeric_limits<dtype>::epsilon() * (double)dist.at(dist.n_elem - 1));
    for (arma::uword i = 0; i < dist.n_elem; ++i)
        if (std::abs((double)dist.at(i) - (double)i * step) > tol)
            return false;
    return true;
}

// Regular directions for the sum-of-sinusoids functions, unit vectors {ux, uy, uz}
// n_dim = 3: spiral (Fibonacci) lattice on the upper hemisphere, uz = (i + 0.5) / n
// n_dim = 2: equally spaced on the half circle, azimuth = pi · i / n, uz = 0
// n_dim = 1: ux = 1
// Optional outputs az, el: azimuth and elevation angles in [rad]
static inline void qd_sos_directions(size_t n, size_t n_dim, double *ux, double *uy, double *uz,
                                     double *az = nullptr, double *el = nullptr)
{
    const double pi = 3.141592653589793;
    const double golden_angle = pi * (3.0 - std::sqrt(5.0));
    for (size_t i = 0; i < n; ++i)
    {
        double a = 0.0, e = 0.0;
        if (n_dim == 3)
        {
            e = std::asin(((double)i + 0.5) / (double)n);
            a = std::remainder((double)i * golden_angle, 2.0 * pi);
        }
        else if (n_dim == 2)
            a = pi * (double)i / (double)n;
        ux[i] = std::cos(a) * std::cos(e), uy[i] = std::sin(a) * std::cos(e), uz[i] = std::sin(e);
        if (az != nullptr)
            az[i] = a;
        if (el != nullptr)
            el[i] = e;
    }
}

#endif
