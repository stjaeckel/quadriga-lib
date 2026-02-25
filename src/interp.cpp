// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_tools.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# interp_1D / interp_2D
Perform linear interpolation (1D or 2D) on single or multiple data sets.

## Description:
- Interpolates given input data at specified output points.
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
- Data types (`dtype`): `float` or `double`

## Declarations:
```
void interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
               const arma::Col<dtype> &xo, const arma::Col<dtype> &yo, arma::Cube<dtype> &output);

arma::Cube<dtype> interp_2D(const arma::Cube<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                            const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_2D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                           const arma::Col<dtype> &xo, const arma::Col<dtype> &yo);

arma::Mat<dtype> interp_1D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);

arma::Col<dtype> interp_1D(const arma::Col<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo);
```

## Arguments:
- `input`: Input data array/matrix (size details below)
- `xi`: Input x-axis sampling points, vector of length `nx`
- `yi`: Input y-axis sampling points (for 2D only), vector of length `ny`
- `xo`: Output x-axis sampling points, vector of length `mx`
- `yo`: Output y-axis sampling points (for 2D only), vector of length `my`
- `output`: Interpolated data cube (modified in-place for one variant)

## Input / Output size details:
- 2D interpolation of multiple datasets (`arma::Cube`):<br>
  Input size: `[ny, nx, ne]`, Output size: `[my, mx, ne]`

- 2D interpolation of single dataset (`arma::Mat`):<br>
  Input size: `[ny, nx]`, Output size: `[my, mx]`

- 1D interpolation of multiple datasets (`arma::Mat`):<br>
  Input size: `[nx, ne]`, Output size: `[mx, ne]`

- 1D interpolation of single dataset (`arma::Col`):<br>
  Input length: `[nx]`, Output length: `[mx]`

## Examples:
- 2D interpolation example:
```
arma::cube input(5, 5, 2, arma::fill::randu); // example input data
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec yi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);
arma::vec yo = arma::linspace(0, 4, 10);

arma::cube output;
quadriga_lib::interp_2D(input, xi, yi, xo, yo, output);
```
- 1D interpolation example:
```
arma::vec input = arma::linspace(0, 1, 5);
arma::vec xi = arma::linspace(0, 4, 5);
arma::vec xo = arma::linspace(0, 4, 10);

auto output = quadriga_lib::interp_1D(input, xi, xo);
```
MD!*/

// 2D linear interpolation
template <typename dtype>
void quadriga_lib::interp_2D(const arma::Cube<dtype> &input,
                             const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                             const arma::Col<dtype> &xo, const arma::Col<dtype> &yo,
                             arma::Cube<dtype> &output)
{
    const arma::uword nx = input.n_cols;
    const arma::uword ny = input.n_rows;
    const arma::uword ne = input.n_slices;
    const arma::uword mx = xo.n_elem;
    const arma::uword my = yo.n_elem;

    if (input.n_elem == 0 || xi.n_elem != nx || yi.n_elem != ny || ne == 0)
        throw std::invalid_argument("Data dimensions must match the given number of sample points.");

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

        const dtype *p_samples_out = samples_out.memptr();
        for (arma::uword i = 0ULL; i < m; ++i)
        {
            arma::uword ip = 0ULL, in = 0ULL;                               // Indices for reading the input data
            dtype val = p_samples_out[i], wp = dtype(1.0), wn = dtype(0.0); // Relative weights for interpolation
            while (ip < n && p_grid[ip] <= val)
                ++ip;
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
    calc_indices(xi, xo, i_xp_vec, i_xn_vec, xp_vec, xn_vec);

    // Calculate the y-interpolation parameters
    arma::uword *i_yp_vec = new arma::uword[my];
    arma::uword *i_yn_vec = new arma::uword[my];
    dtype *yp_vec = new dtype[my];
    dtype *yn_vec = new dtype[my];
    calc_indices(yi, yo, i_yp_vec, i_yn_vec, yp_vec, yn_vec);

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
arma::Cube<dtype> quadriga_lib::interp_2D(const arma::Cube<dtype> &input,
                                          const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                                          const arma::Col<dtype> &xo, const arma::Col<dtype> &yo)
{
    arma::Cube<dtype> output;
    quadriga_lib::interp_2D(input, xi, yi, xo, yo, output);
    return output;
}

template <typename dtype>
arma::Mat<dtype> quadriga_lib::interp_2D(const arma::Mat<dtype> &input,
                                         const arma::Col<dtype> &xi, const arma::Col<dtype> &yi,
                                         const arma::Col<dtype> &xo, const arma::Col<dtype> &yo)
{
    arma::Mat<dtype> output(yo.n_elem, xo.n_elem, arma::fill::none);
    arma::Cube<dtype> output_cube(output.memptr(), yo.n_elem, xo.n_elem, 1ULL, false, true);
    const arma::Cube<dtype> input_cube(const_cast<dtype *>(input.memptr()), input.n_rows, input.n_cols, 1ULL, false, true);
    quadriga_lib::interp_2D(input_cube, xi, yi, xo, yo, output_cube);
    return output;
}

template <typename dtype>
arma::Mat<dtype> quadriga_lib::interp_1D(const arma::Mat<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo)
{
    arma::Col<dtype> y(1);
    arma::Mat<dtype> output(xo.n_elem, input.n_cols, arma::fill::none);
    arma::Cube<dtype> output_cube(output.memptr(), 1ULL, xo.n_elem, input.n_cols, false, true);
    const arma::Cube<dtype> input_cube(const_cast<dtype *>(input.memptr()), 1ULL, input.n_rows, input.n_cols, false, true);
    quadriga_lib::interp_2D(input_cube, xi, y, xo, y, output_cube);
    return output;
}

template <typename dtype>
arma::Col<dtype> quadriga_lib::interp_1D(const arma::Col<dtype> &input, const arma::Col<dtype> &xi, const arma::Col<dtype> &xo)
{
    arma::Col<dtype> y(1);
    arma::Col<dtype> output(xo.n_elem, arma::fill::none);
    arma::Cube<dtype> output_cube(output.memptr(), 1ULL, xo.n_elem, 1ULL, false, true);
    const arma::Cube<dtype> input_cube(const_cast<dtype *>(input.memptr()), 1ULL, input.n_rows, 1ULL, false, true);
    quadriga_lib::interp_2D(input_cube, xi, y, xo, y, output_cube);
    return output;
}

template void quadriga_lib::interp_2D(const arma::Cube<float> &input,
                                      const arma::Col<float> &xi, const arma::Col<float> &yi,
                                      const arma::Col<float> &xo, const arma::Col<float> &yo,
                                      arma::Cube<float> &output);

template void quadriga_lib::interp_2D(const arma::Cube<double> &input,
                                      const arma::Col<double> &xi, const arma::Col<double> &yi,
                                      const arma::Col<double> &xo, const arma::Col<double> &yo,
                                      arma::Cube<double> &output);

template arma::Cube<float> quadriga_lib::interp_2D(const arma::Cube<float> &input,
                                                   const arma::Col<float> &xi, const arma::Col<float> &yi,
                                                   const arma::Col<float> &xo, const arma::Col<float> &yo);

template arma::Cube<double> quadriga_lib::interp_2D(const arma::Cube<double> &input,
                                                    const arma::Col<double> &xi, const arma::Col<double> &yi,
                                                    const arma::Col<double> &xo, const arma::Col<double> &yo);

template arma::Mat<float> quadriga_lib::interp_2D(const arma::Mat<float> &input,
                                                  const arma::Col<float> &xi, const arma::Col<float> &yi,
                                                  const arma::Col<float> &xo, const arma::Col<float> &yo);

template arma::Mat<double> quadriga_lib::interp_2D(const arma::Mat<double> &input,
                                                   const arma::Col<double> &xi, const arma::Col<double> &yi,
                                                   const arma::Col<double> &xo, const arma::Col<double> &yo);

template arma::Mat<float> quadriga_lib::interp_1D(const arma::Mat<float> &input,
                                                  const arma::Col<float> &xi,
                                                  const arma::Col<float> &xo);

template arma::Mat<double> quadriga_lib::interp_1D(const arma::Mat<double> &input,
                                                   const arma::Col<double> &xi,
                                                   const arma::Col<double> &xo);

template arma::Col<float> quadriga_lib::interp_1D(const arma::Col<float> &input,
                                                  const arma::Col<float> &xi,
                                                  const arma::Col<float> &xo);

template arma::Col<double> quadriga_lib::interp_1D(const arma::Col<double> &input,
                                                   const arma::Col<double> &xi,
                                                   const arma::Col<double> &xo);
