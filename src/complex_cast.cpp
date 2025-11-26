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

template <typename dtype>
void quadriga_lib::complex_cast(const arma::Mat<dtype> &real, const arma::Mat<dtype> &imag, arma::cx_mat &complex)
{
    arma::uword n_rows = real.n_rows, n_cols = real.n_cols;

    if (imag.n_rows != n_rows || imag.n_cols != n_cols)
        throw std::invalid_argument("Sizes of real and imaginary parts dont match");

    if (complex.n_rows != n_rows || complex.n_cols != n_cols)
        complex.set_size(n_rows, n_cols);

    auto *p_complex = complex.memptr();
    const dtype *p_real = real.memptr(), *p_imag = imag.memptr();

    for (arma::uword i = 0; i < n_rows * n_cols; ++i)
        p_complex[i] = {(double)p_real[i], (double)p_imag[i]};
}

template <typename dtype>
void quadriga_lib::complex_cast(const arma::Cube<dtype> &real, const arma::Cube<dtype> &imag, arma::cx_cube &complex)
{
    arma::uword n_rows = real.n_rows, n_cols = real.n_cols, n_slices = real.n_slices;

    if (imag.n_rows != n_rows || imag.n_cols != n_cols || imag.n_slices != n_slices)
        throw std::invalid_argument("Sizes of real and imaginary parts dont match");

    if (complex.n_rows != n_rows || complex.n_cols != n_cols || complex.n_slices != n_slices)
        complex.set_size(n_rows, n_cols, n_slices);

    auto *p_complex = complex.memptr();
    const dtype *p_real = real.memptr(), *p_imag = imag.memptr();

    for (arma::uword i = 0; i < n_rows * n_cols * n_slices; ++i)
        p_complex[i] = {(double)p_real[i], (double)p_imag[i]};
}

template <typename dtype>
void quadriga_lib::complex_cast(const arma::cx_mat &complex, arma::Mat<dtype> &real, arma::Mat<dtype> &imag)
{
    arma::uword n_rows = complex.n_rows, n_cols = complex.n_cols;

    if (real.n_rows != n_rows || real.n_cols != n_cols)
        real.set_size(n_rows, n_cols);

    if (imag.n_rows != n_rows || imag.n_cols != n_cols)
        imag.set_size(n_rows, n_cols);

    const auto *p_complex = complex.memptr();
    dtype *p_real = real.memptr(), *p_imag = imag.memptr();

    for (arma::uword i = 0; i < n_rows * n_cols; ++i)
    {
        p_real[i] = (dtype)p_complex[i].real();
        p_imag[i] = (dtype)p_complex[i].imag();
    }
}

template <typename dtype>
void quadriga_lib::complex_cast(const arma::cx_cube &complex, arma::Cube<dtype> &real, arma::Cube<dtype> &imag)
{
    arma::uword n_rows = complex.n_rows, n_cols = complex.n_cols, n_slices = complex.n_slices;

    if (real.n_rows != n_rows || real.n_cols != n_cols || real.n_slices != n_slices)
        real.set_size(n_rows, n_cols, n_slices);

    if (imag.n_rows != n_rows || imag.n_cols != n_cols || imag.n_slices != n_slices)
        imag.set_size(n_rows, n_cols, n_slices);

    const auto *p_complex = complex.memptr();
    dtype *p_real = real.memptr(), *p_imag = imag.memptr();

    for (arma::uword i = 0; i < n_rows * n_cols * n_slices; ++i)
    {
        p_real[i] = (dtype)p_complex[i].real();
        p_imag[i] = (dtype)p_complex[i].imag();
    }
}

// Template instantiation
template void quadriga_lib::complex_cast(const arma::Mat<float> &real, const arma::Mat<float> &imag, arma::cx_mat &complex);
template void quadriga_lib::complex_cast(const arma::Mat<double> &real, const arma::Mat<double> &imag, arma::cx_mat &complex);

template void quadriga_lib::complex_cast(const arma::Cube<float> &real, const arma::Cube<float> &imag, arma::cx_cube &complex);
template void quadriga_lib::complex_cast(const arma::Cube<double> &real, const arma::Cube<double> &imag, arma::cx_cube &complex);

template void quadriga_lib::complex_cast(const arma::cx_mat &complex, arma::Mat<float> &real, arma::Mat<float> &imag);
template void quadriga_lib::complex_cast(const arma::cx_mat &complex, arma::Mat<double> &real, arma::Mat<double> &imag);

template void quadriga_lib::complex_cast(const arma::cx_cube &complex, arma::Cube<float> &real, arma::Cube<float> &imag);
template void quadriga_lib::complex_cast(const arma::cx_cube &complex, arma::Cube<double> &real, arma::Cube<double> &imag);
