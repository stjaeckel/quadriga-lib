// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_math.hpp"
#include "qd_helpers_shared.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

/*!SECTION
Math functions
SECTION!*/

/*!MD
# interp_2D
Perform linear interpolation (1D or 2D) on multiple data sets

- Interpolates given input data at specified output points.
- Outside the range of the input sample points, the nearest end value is returned (constant extrapolation);
  output sample points that are NaN give NaN.
- The input sample points need not be sorted (they are sorted internally); repeated values are allowed;
  they must be finite.
- `interp_1D` interpolates the columns of a matrix; a single data set is a matrix with one column.
- Data types (`dtype`): `float` or `double`

## Declarations:
```
void quadriga_lib::interp_2D(
    const arma::Col<dtype> &x,
    const arma::Col<dtype> &y,
    const arma::Cube<dtype> &input,
    const arma::Col<dtype> &x_new,
    const arma::Col<dtype> &y_new,
    arma::Cube<dtype> &output);

void quadriga_lib::interp_1D(
    const arma::Col<dtype> &x,
    const arma::Mat<dtype> &input,
    const arma::Col<dtype> &x_new,
    arma::Mat<dtype> &output);
```

## Inputs:
- **`x`** — Input x-axis sample points; `[nx]`
- **`y`** — Input y-axis sample points (2D only); `[ny]`
- **`input`** — Input data; 2D: `[ny, nx, ne]`; 1D: `[nx, ne]`; `ne` = number of data sets
- **`x_new`** — Output x-axis sample points; `[nx_new]`
- **`y_new`** — Output y-axis sample points (2D only); `[ny_new]`

## Output:
- **`output`** — Interpolated data; 2D: `[ny_new, nx_new, ne]`; 1D: `[nx_new, ne]`

## Examples:
- 2D interpolation example:
```
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec x = arma::linspace(0, 4, 5);
arma::vec y = arma::linspace(0, 4, 5);
arma::vec x_new = arma::linspace(0, 4, 10);
arma::vec y_new = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(x, y, input, x_new, y_new, output);
```
- 1D interpolation example:
```
arma::vec input = arma::linspace(0, 1, 5);
arma::vec x = arma::linspace(0, 4, 5);
arma::vec x_new = arma::linspace(0, 4, 10);

arma::mat output;
quadriga_lib::interp_1D(x, input, x_new, output);
```
MD!*/

// 2D linear interpolation
template <typename dtype>
void quadriga_lib::interp_2D(const arma::Col<dtype> &x, const arma::Col<dtype> &y,
                             const arma::Cube<dtype> &input,
                             const arma::Col<dtype> &x_new, const arma::Col<dtype> &y_new,
                             arma::Cube<dtype> &output)
{
    const arma::uword nx = input.n_cols;
    const arma::uword ny = input.n_rows;
    const arma::uword ne = input.n_slices;
    const arma::uword mx = x_new.n_elem;
    const arma::uword my = y_new.n_elem;

    if (input.n_elem == 0 || x.n_elem != nx || y.n_elem != ny || ne == 0)
        throw std::invalid_argument("Data dimensions must match the given number of sample points.");
    if (!x.is_finite())
        throw std::invalid_argument("Input 'x' must be finite.");
    if (!y.is_finite())
        throw std::invalid_argument("Input 'y' must be finite.");
    qd_check_alias(&output, "output", &input);

    if (output.n_rows != my || output.n_cols != mx || output.n_slices != ne)
        output.set_size(my, mx, ne);

    if (mx == 0 || my == 0)
        return;

    // Lambda to calculate the indices
    // - i_prev, i_next, d_prev, d_next must be allocated to length samples_out.n_elem
    auto calc_indices = [](const arma::Col<dtype> &samples_in, const arma::Col<dtype> &samples_out,
                           arma::uword *i_prev, arma::uword *i_next, dtype *d_prev, dtype *d_next)
    {
        const arma::uword n = samples_in.n_elem;
        const arma::uword m = samples_out.n_elem;

        bool input_samples_sorted = samples_in.is_sorted();
        arma::uvec ind = input_samples_sorted ? arma::regspace<arma::uvec>(0, n - 1) : arma::sort_index(samples_in);

        arma::uword *p_ind = ind.memptr();
        const dtype *p_samples_in = samples_in.memptr();

        dtype *p_grid_srt = new dtype[n];
        if (!input_samples_sorted)
            for (arma::uword i = 0ULL; i < n; ++i)
                p_grid_srt[i] = p_samples_in[p_ind[i]];

        const dtype *p_grid = input_samples_sorted ? p_samples_in : p_grid_srt;

        dtype *p_diff = new dtype[n];
        *p_diff = dtype(1.0);
        for (arma::uword i = 1ULL; i < n; ++i)
            p_diff[i] = dtype(1.0) / (p_grid[i] - p_grid[i - 1]);

        // Index search: ip = index of the first grid value > val (n if there is none), by a linear scan for
        // short grids and a binary search otherwise; both give the same index, also with repeated grid values
        const dtype *p_samples_out = samples_out.memptr();
        for (arma::uword i = 0ULL; i < m; ++i)
        {
            arma::uword ip = 0ULL, in = 0ULL;                               // Indices for reading the input data
            dtype val = p_samples_out[i], wp = dtype(1.0), wn = dtype(0.0); // Relative weights for interpolation
            if (n <= 16ULL)
                while (ip < n && p_grid[ip] <= val)
                    ++ip;
            else
                ip = (arma::uword)(std::upper_bound(p_grid, p_grid + n, val) - p_grid);

            // A NaN query ends at ip == 0 (scan) or ip == n (binary search): NaN weights make the output NaN
            if ((ip == 0ULL || ip == n) && std::isnan(val))
            {
                i_prev[i] = p_ind[0], i_next[i] = p_ind[0];
                d_prev[i] = std::numeric_limits<dtype>::quiet_NaN(), d_next[i] = std::numeric_limits<dtype>::quiet_NaN();
                continue;
            }
            if (ip == n)
                in = --ip;
            else if (ip != 0ULL)
            {
                in = ip--;
                wp = (p_grid[in] - val) * p_diff[in];
                wp = (wp > dtype(1.0)) ? dtype(1.0) : wp;
                wp = (wp < dtype(0.0)) ? dtype(0.0) : wp;
                wn = dtype(1.0) - wp;
            }
            i_prev[i] = p_ind[ip], i_next[i] = p_ind[in], d_prev[i] = wp, d_next[i] = wn;
        }
        delete[] p_diff;
        delete[] p_grid_srt;
    };

    // Calculate the x-interpolation parameters
    arma::uword *i_xp_vec = new arma::uword[mx];
    arma::uword *i_xn_vec = new arma::uword[mx];
    dtype *xp_vec = new dtype[mx];
    dtype *xn_vec = new dtype[mx];
    calc_indices(x, x_new, i_xp_vec, i_xn_vec, xp_vec, xn_vec);

    // Calculate the y-interpolation parameters
    arma::uword *i_yp_vec = new arma::uword[my];
    arma::uword *i_yn_vec = new arma::uword[my];
    dtype *yp_vec = new dtype[my];
    dtype *yn_vec = new dtype[my];
    calc_indices(y, y_new, i_yp_vec, i_yn_vec, yp_vec, yn_vec);

    // Interpolate the input data and write to output memory
    for (arma::uword ie = 0; ie < ne; ++ie)
    {
        const dtype *p_input = input.slice_memptr(ie);
        dtype *p_output = output.slice_memptr(ie);

        for (arma::uword ix = 0ULL; ix < mx; ++ix)
        {
            for (arma::uword iy = 0ULL; iy < my; ++iy)
            {
                arma::uword iA = i_xp_vec[ix] * ny + i_yp_vec[iy];
                arma::uword iB = i_xn_vec[ix] * ny + i_yp_vec[iy];
                arma::uword iC = i_xp_vec[ix] * ny + i_yn_vec[iy];
                arma::uword iD = i_xn_vec[ix] * ny + i_yn_vec[iy];

                dtype wA = xp_vec[ix] * yp_vec[iy];
                dtype wB = xn_vec[ix] * yp_vec[iy];
                dtype wC = xp_vec[ix] * yn_vec[iy];
                dtype wD = xn_vec[ix] * yn_vec[iy];

                *p_output++ = wA * p_input[iA] + wB * p_input[iB] + wC * p_input[iC] + wD * p_input[iD];
            }
        }
    }

    delete[] i_xp_vec;
    delete[] i_xn_vec;
    delete[] i_yp_vec;
    delete[] i_yn_vec;
    delete[] xp_vec;
    delete[] xn_vec;
    delete[] yp_vec;
    delete[] yn_vec;
}

template <typename dtype>
void quadriga_lib::interp_1D(const arma::Col<dtype> &x, const arma::Mat<dtype> &input,
                             const arma::Col<dtype> &x_new, arma::Mat<dtype> &output)
{
    // Validated here, before 'output' is resized; the alias check needs the matrix objects
    if (input.n_elem == 0 || x.n_elem != input.n_rows)
        throw std::invalid_argument("Data dimensions must match the given number of sample points.");
    if (!x.is_finite())
        throw std::invalid_argument("Input 'x' must be finite.");
    qd_check_alias(&output, "output", &input);

    if (output.n_rows != x_new.n_elem || output.n_cols != input.n_cols)
        output.set_size(x_new.n_elem, input.n_cols);

    // The 1D case is the 2D case with one y sample: input [1, nx, ne], output [1, nx_new, ne]
    const arma::Col<dtype> y(1, arma::fill::zeros);
    arma::Cube<dtype> output_cube(output.memptr(), 1ULL, x_new.n_elem, input.n_cols, false, true);
    const arma::Cube<dtype> input_cube(const_cast<dtype *>(input.memptr()), 1ULL, input.n_rows, input.n_cols, false, true);
    quadriga_lib::interp_2D(x, y, input_cube, x_new, y, output_cube);
}

template void quadriga_lib::interp_2D(const arma::Col<float> &x, const arma::Col<float> &y,
                                      const arma::Cube<float> &input,
                                      const arma::Col<float> &x_new, const arma::Col<float> &y_new,
                                      arma::Cube<float> &output);

template void quadriga_lib::interp_2D(const arma::Col<double> &x, const arma::Col<double> &y,
                                      const arma::Cube<double> &input,
                                      const arma::Col<double> &x_new, const arma::Col<double> &y_new,
                                      arma::Cube<double> &output);

template void quadriga_lib::interp_1D(const arma::Col<float> &x, const arma::Mat<float> &input,
                                      const arma::Col<float> &x_new, arma::Mat<float> &output);

template void quadriga_lib::interp_1D(const arma::Col<double> &x, const arma::Mat<double> &input,
                                      const arma::Col<double> &x_new, arma::Mat<double> &output);
