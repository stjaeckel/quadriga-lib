// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# sos_calc_mse
Approximation error of the ACF of a set of sinusoids

- Compares the autocorrelation function (ACF) produced by the sinusoids,
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`, with the target ACF
- All errors are returned as `-10·log10( mean squared error )`; larger is better
- Return value: error along the test directions on the sample points `dist`. `n_dim = 1`: one direction;
  `n_dim = 2`: `n_dir` directions equally spaced on the half circle, azimuth `pi · i / n_dir`;
  `n_dim = 3`: `n_dir` directions on a spiral (Fibonacci) lattice on the upper hemisphere,
  `sin(elevation) = ( i + 0.5 ) / n_dir`, azimuth `i · pi · ( 3 - sqrt(5) )`. A direction and its opposite
  give the same ACF
- `mse_core`: error on the grid of [[sos_acf_approx]] against [[sos_acf_2d]], for the grid points within a
  radius of `n_acf` grid steps around the center; for `n_dim = 3` the mean squared error is averaged over
  the three planes
- `mse_all`: same on the complete grid, which extends to `2·Dmax` and has a target of 0 beyond `Dmax`
- For `n_dim = 1`, `mse_core` is evaluated on the sample points `dist`, `mse_all` on the axis from
  `-2·Dmax` to `2·Dmax`, and the return value equals `mse_core`
- `acf_tail`: approximated ACF, averaged over the test directions, at 2, 4, 10 and 20 times `Dmax`. The
  target is 0 there; sinusoids with a frequency near zero show up as a plateau of height
  `n_low / n_coeff`, which adds a constant offset to all values of a large area
- `mse_core` and `mse_all` need the 2-D grids and are only computed when requested
- A perfect fit has a mean squared error of 0 and is returned as +Inf dB; this happens for a target that
  the sinusoids reproduce exactly at all sample points, e.g. sinusoids at zero frequency and a target of 1
- The test directions are evaluated in parallel; the result does not depend on the number of threads
- Equally spaced has the meaning given in [[sos_acf_2d]]
- `acf_tail` and `test_dir` may be the same objects as inputs

## Declaration:
```
dtype quadriga_lib::sos_calc_mse(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &acf,
    const arma::Col<dtype> &dist,
    dtype *mse_core = nullptr,
    dtype *mse_all = nullptr,
    arma::Col<dtype> *acf_tail = nullptr,
    arma::Mat<dtype> *test_dir = nullptr,
    arma::uword n_dir = 10);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; `[n_acf]`
- **`n_dir`** *(optional)* — Number of test directions; must be > 0; ignored for `n_dim = 1`

## Outputs:
- **`mse_core`** *(optional)* — Error within the core radius in [dB]
- **`mse_all`** *(optional)* — Error on the complete grid in [dB]
- **`acf_tail`** *(optional)* — Approximated ACF beyond the fitted range; `[4]`
- **`test_dir`** *(optional)* — Test directions in [rad]; `[n_dir, 2]` with columns `{azimuth, elevation}`
  for `n_dim = 3`; `[n_dir, 1]` with the azimuth for `n_dim = 2`; empty for `n_dim = 1`

## Returns:
- Error along the test directions in [dB]

## See also:
- [[sos_generate]] (finds the frequencies; minimises the error returned here)
- [[sos_acf_approx]], [[sos_acf_2d]] (the two grids that are compared)
MD!*/

template <typename dtype>
dtype quadriga_lib::sos_calc_mse(const arma::Mat<dtype> &freq,
                                 const arma::Col<dtype> &acf,
                                 const arma::Col<dtype> &dist,
                                 dtype *mse_core,
                                 dtype *mse_all,
                                 arma::Col<dtype> *acf_tail,
                                 arma::Mat<dtype> *test_dir,
                                 arma::uword n_dir)
{
    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols > 3)
        throw std::invalid_argument("Input 'freq' must have 1, 2 or 3 columns.");
    qd_sos_check_acf(acf, dist);
    if (!qd_sos_equally_spaced(dist))
        throw std::invalid_argument("Input 'dist' must be equally spaced.");
    if (n_dir == 0)
        throw std::invalid_argument("Input 'n_dir' cannot be 0.");

    const arma::uword n_dim = freq.n_cols, n_test = (n_dim == 1) ? 1 : n_dir;

    const arma::uword n_coeff = freq.n_rows, n_acf = acf.n_elem;
    const double two_pi = 6.283185307179586, d_max = (double)dist.at(n_acf - 1);

    // Test directions and the frequency of each sinusoid along each of them
    std::vector<double> ux(n_test), uy(n_test), uz(n_test), az(n_test), el(n_test);
    qd_sos_directions(n_test, n_dim, ux.data(), uy.data(), uz.data(), az.data(), el.data());

    arma::mat ft(n_coeff, n_test);
    for (arma::uword t = 0; t < n_test; ++t)
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            double f = (double)freq.at(n, 0) * ux[t];
            if (n_dim > 1)
                f += (double)freq.at(n, 1) * uy[t];
            if (n_dim > 2)
                f += (double)freq.at(n, 2) * uz[t];
            ft.at(n, t) = f;
        }

    // ACF of the sinusoids along direction t at distance d
    auto acf_dir = [&](arma::uword t, double d) -> double
    {
        const double *f = ft.colptr(t);
        double acc = 0.0;
        for (arma::uword n = 0; n < n_coeff; ++n)
            acc += std::cos(two_pi * f[n] * d);
        return acc / (double)n_coeff;
    };

    // Squared errors of all test directions in parallel; they are then added in a fixed order, so that the
    // result does not depend on the number of threads
    arma::mat sq(n_acf, n_test, arma::fill::none);
    const long long n_test_ll = (long long)n_test;
#pragma omp parallel for if (n_test > 1)
    for (long long i_test = 0; i_test < n_test_ll; ++i_test)
    {
        const arma::uword t = (arma::uword)i_test;
        double *p = sq.colptr(t);
        for (arma::uword s = 0; s < n_acf; ++s)
        {
            double e = (double)acf.at(s) - acf_dir(t, (double)dist.at(s));
            p[s] = e * e;
        }
    }
    double err = 0.0;
    for (arma::uword i = 0; i < sq.n_elem; ++i)
        err += sq.at(i);
    const double mse = -10.0 * std::log10(err / (double)(n_test * n_acf));

    if (mse_core != nullptr || mse_all != nullptr)
    {
        const arma::mat freq_d = arma::conv_to<arma::mat>::from(freq);
        const arma::vec acf_d = arma::conv_to<arma::vec>::from(acf), dist_d = arma::conv_to<arma::vec>::from(dist);
        const long long S = (long long)n_acf, half = 2 * S - 2;

        arma::cube approx;
        quadriga_lib::sos_acf_approx<double>(freq_d, dist_d, approx);

        double core = 0.0, all = 0.0;
        if (n_dim == 1)
        {
            // Target on the axis: the table mirrored around the center, 0 beyond the last sample point
            for (long long k = -half; k <= half; ++k)
            {
                long long a = (k < 0) ? -k : k;
                double target = (a < S) ? acf_d.at((arma::uword)a) : 0.0;
                double e = approx.at((arma::uword)(k + half), 0, 0) - target;
                all += e * e;
            }
            all /= (double)(2 * half + 1);
            core = err / (double)n_acf;
        }
        else
        {
            arma::mat target;
            quadriga_lib::sos_acf_2d<double>(acf_d, dist_d, target);
            for (arma::uword pl = 0; pl < approx.n_slices; ++pl)
            {
                double sum_core = 0.0, sum_all = 0.0, n_core = 0.0;
                for (long long ic = -half; ic <= half; ++ic)
                    for (long long ir = -half; ir <= half; ++ir)
                    {
                        double e = approx.at((arma::uword)(ir + half), (arma::uword)(ic + half), pl) - target.at((arma::uword)(ir + half), (arma::uword)(ic + half));
                        sum_all += e * e;
                        if (ic * ic + ir * ir <= S * S)
                            sum_core += e * e, n_core += 1.0;
                    }
                core += sum_core / n_core / (double)approx.n_slices;
                all += sum_all / (double)((2 * half + 1) * (2 * half + 1)) / (double)approx.n_slices;
            }
        }
        if (mse_core != nullptr)
            *mse_core = (n_dim == 1) ? (dtype)mse : (dtype)(-10.0 * std::log10(core));
        if (mse_all != nullptr)
            *mse_all = (dtype)(-10.0 * std::log10(all));
    }

    if (acf_tail != nullptr)
    {
        if (acf_tail->n_elem != 4)
            acf_tail->set_size(4);
        const double mult[4] = {2.0, 4.0, 10.0, 20.0};
        arma::mat tail(n_test, 4, arma::fill::none);
#pragma omp parallel for if (n_test > 1)
        for (long long i_test = 0; i_test < n_test_ll; ++i_test)
            for (arma::uword k = 0; k < 4; ++k)
                tail.at((arma::uword)i_test, k) = acf_dir((arma::uword)i_test, mult[k] * d_max);
        for (arma::uword k = 0; k < 4; ++k)
        {
            double acc = 0.0;
            for (arma::uword t = 0; t < n_test; ++t)
                acc += tail.at(t, k);
            acf_tail->at(k) = (dtype)(acc / (double)n_test);
        }
    }

    if (test_dir != nullptr)
    {
        arma::uword nr = (n_dim == 1) ? 0 : n_test, nc = (n_dim == 1) ? 0 : n_dim - 1;
        if (test_dir->n_rows != nr || test_dir->n_cols != nc)
            test_dir->set_size(nr, nc);
        for (arma::uword t = 0; t < nr; ++t)
        {
            test_dir->at(t, 0) = (dtype)az[t];
            if (nc > 1)
                test_dir->at(t, 1) = (dtype)el[t];
        }
    }

    return (dtype)mse;
}

template float quadriga_lib::sos_calc_mse(const arma::Mat<float> &freq, const arma::Col<float> &acf, const arma::Col<float> &dist, float *mse_core, float *mse_all,
                                          arma::Col<float> *acf_tail, arma::Mat<float> *test_dir, arma::uword n_dir);
template double quadriga_lib::sos_calc_mse(const arma::Mat<double> &freq, const arma::Col<double> &acf, const arma::Col<double> &dist, double *mse_core, double *mse_all,
                                           arma::Col<double> *acf_tail, arma::Mat<double> *test_dir, arma::uword n_dir);
