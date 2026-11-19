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

// Validate an ACF table for two devices on the same map, throws std::invalid_argument
// The combined value is divided by sqrt( 2 + 2·A ), so the ACF must stay above -1
template <typename dtype>
static inline void qd_sos_check_acf_same_map(const arma::Col<dtype> &acf, const arma::Col<dtype> &dist)
{
    qd_sos_check_acf(acf, dist);
    for (arma::uword i = 0; i < acf.n_elem; ++i)
        if (!(acf.at(i) > (dtype)-1.0))
            throw std::invalid_argument("Input 'acf' must be > -1 for two devices on the same map.");
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
