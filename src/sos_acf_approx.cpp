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
# sos_acf_approx
ACF produced by a set of sinusoids

- Autocorrelation function (ACF) of the values of [[sos_val]] over uniformly random phases:
  `R(dx) = (1/n_coeff) · sum_n cos( 2·pi · f_n·dx )`
- Evaluated on the grid axis of [[sos_acf_2d]]: from `-2·Dmax` to `2·Dmax` with the spacing of `dist`,
  `n_grid = 4·n_acf - 3`
- `n_dim = 1`: along the axis; `[n_grid, 1, 1]`
- `n_dim = 2`: on the x-y plane; `[n_grid, n_grid, 1]`; element `(iy, ix)`
- `n_dim = 3`: on the x-y, x-z and y-z plane (slices 0, 1, 2); `[n_grid, n_grid, 3]`; the row index belongs
  to the second axis of the plane, the column index to the first
- Only the sample points `dist` are used, not the ACF values
- `R(-dx) = R(dx)`: each plane is point-symmetric about its center. One half is computed and the other
  half is its mirror image, so the symmetry is exact
- Equally spaced has the meaning given in [[sos_acf_2d]]
- `dist_2d` may be the same object as `dist`

## Declaration:
```
void quadriga_lib::sos_acf_approx(
    const arma::Mat<dtype> &freq,
    const arma::Col<dtype> &dist,
    arma::Cube<dtype> &acf_approx,
    arma::Col<dtype> *dist_2d = nullptr);
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`dist`** — Sample points of the ACF in [m]; must start at 0, be equally spaced and finite; at least
  2 elements; `[n_acf]`

## Outputs:
- **`acf_approx`** — Approximated ACF; size depends on `n_dim`, see above
- **`dist_2d`** *(optional)* — Grid axis in [m]; `[n_grid]`

## See also:
- [[sos_acf_2d]] (target ACF on the same grid)
- [[sos_calc_mse]] (difference between the two)
MD!*/

template <typename dtype>
void quadriga_lib::sos_acf_approx(const arma::Mat<dtype> &freq,
                                  const arma::Col<dtype> &dist,
                                  arma::Cube<dtype> &acf_approx,
                                  arma::Col<dtype> *dist_2d)
{
    if (freq.n_elem == 0)
        throw std::invalid_argument("Input 'freq' cannot be empty.");
    if (freq.n_cols > 3)
        throw std::invalid_argument("Input 'freq' must have 1, 2 or 3 columns.");
    if (dist.n_elem < 2)
        throw std::invalid_argument("Input 'dist' must have at least 2 elements.");
    if (dist.at(0) != (dtype)0.0)
        throw std::invalid_argument("Input 'dist' must start at 0.");
    for (arma::uword i = 0; i < dist.n_elem; ++i)
        if (!std::isfinite((double)dist.at(i)))
            throw std::invalid_argument("Input 'dist' must be finite.");
    if (!(dist.at(dist.n_elem - 1) > (dtype)0.0) || !qd_sos_equally_spaced(dist))
        throw std::invalid_argument("Input 'dist' must be equally spaced.");

    const arma::uword n_dim = freq.n_cols, n_grid = 4 * dist.n_elem - 3;
    const arma::uword n_rows = n_grid, n_cols = (n_dim == 1) ? 1 : n_grid, n_slices = (n_dim == 3) ? 3 : 1;

    const arma::uword n_coeff = freq.n_rows;
    const double two_pi = 6.283185307179586, step = (double)dist.at(dist.n_elem - 1) / (double)(dist.n_elem - 1);
    const long long half = 2 * (long long)dist.n_elem - 2;

    if (acf_approx.n_rows != n_rows || acf_approx.n_cols != n_cols || acf_approx.n_slices != n_slices)
        acf_approx.set_size(n_rows, n_cols, n_slices);

    // Cosine and sine of each sinusoid at each grid point of each axis, Size [n_coeff, n_grid] per axis.
    // With cos(a + b) = cos(a)·cos(b) - sin(a)·sin(b) a plane is the difference of two matrix products.
    std::vector<std::vector<double>> C(n_dim), S(n_dim);
    for (arma::uword d = 0; d < n_dim; ++d)
    {
        C[d].resize(n_coeff * n_grid), S[d].resize(n_coeff * n_grid);
        double *pc = C[d].data(), *ps = S[d].data();
        const dtype *pf = freq.colptr(d);
        const long long n_grid_ll = (long long)n_grid;
#pragma omp parallel for if (n_coeff * n_grid > 4096)
        for (long long i_grid = 0; i_grid < n_grid_ll; ++i_grid)
        {
            const arma::uword k = (arma::uword)i_grid;
            double xk = (double)(i_grid - half) * step;
            for (arma::uword n = 0; n < n_coeff; ++n)
            {
                double t = (double)pf[n] * xk;
                t -= std::floor(t + 0.5);
                pc[k * n_coeff + n] = std::cos(two_pi * t), ps[k * n_coeff + n] = std::sin(two_pi * t);
            }
        }
    }

    const double scale = 1.0 / (double)n_coeff;
    if (n_dim == 1)
    {
        dtype *out = acf_approx.memptr();
        for (arma::uword k = 0; k < n_grid; ++k)
        {
            double acc = 0.0;
            for (arma::uword n = 0; n < n_coeff; ++n)
                acc += C[0][k * n_coeff + n];
            out[k] = (dtype)(acc * scale);
        }
    }
    else
    {
        // Slice 0: x-y, slice 1: x-z, slice 2: y-z; the first axis runs along the columns
        const arma::uword first[3] = {0, 0, 1}, second[3] = {1, 2, 2};
        // R(-dx) = R(dx): the columns up to the center are computed, the others are their mirror image
        const long long n_half_ll = half + 1;
        for (arma::uword s = 0; s < n_slices; ++s)
        {
            const double *Ca = C[first[s]].data(), *Sa = S[first[s]].data();
            const double *Cb = C[second[s]].data(), *Sb = S[second[s]].data();
            dtype *out = acf_approx.slice_memptr(s);

#pragma omp parallel for
            for (long long i_col = 0; i_col < n_half_ll; ++i_col)
            {
                const arma::uword c = (arma::uword)i_col;
                const double *ca = &Ca[c * n_coeff], *sa = &Sa[c * n_coeff];
                for (arma::uword r = 0; r < n_grid; ++r)
                {
                    const double *cb = &Cb[r * n_coeff], *sb = &Sb[r * n_coeff];
                    double acc = 0.0;
                    for (arma::uword n = 0; n < n_coeff; ++n)
                        acc += ca[n] * cb[n] - sa[n] * sb[n];
                    out[c * n_grid + r] = (dtype)(acc * scale);
                }
            }
            for (arma::uword c = (arma::uword)n_half_ll; c < n_grid; ++c)
                for (arma::uword r = 0; r < n_grid; ++r)
                    out[c * n_grid + r] = out[(n_grid - 1 - c) * n_grid + (n_grid - 1 - r)];
        }
    }

    if (dist_2d != nullptr)
    {
        if (dist_2d->n_elem != n_grid)
            dist_2d->set_size(n_grid);
        for (long long i = -half; i <= half; ++i)
            dist_2d->at((arma::uword)(i + half)) = (dtype)((double)i * step);
    }
}

template void quadriga_lib::sos_acf_approx(const arma::Mat<float> &freq, const arma::Col<float> &dist, arma::Cube<float> &acf_approx, arma::Col<float> *dist_2d);
template void quadriga_lib::sos_acf_approx(const arma::Mat<double> &freq, const arma::Col<double> &dist, arma::Cube<double> &acf_approx, arma::Col<double> *dist_2d);
