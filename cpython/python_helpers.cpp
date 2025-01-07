// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#ifndef quadriga_python_helpers_H
#define quadriga_python_helpers_H

#include <any>
#include <armadillo>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Get the dimensions of a buffer and check if storage order if column-major
inline bool get_dims(const pybind11::buffer_info *buf, size_t *d1, size_t *d2, size_t *d3, size_t *d4)
{
    size_t n_dim = (size_t)buf->ndim;              // Number of dimensions
    *d1 = (n_dim < 1) ? 1 : (size_t)buf->shape[0]; // Number of elements on first dimension
    *d2 = (n_dim < 2) ? 1 : (size_t)buf->shape[1]; // Number of elements on second dimension
    *d3 = (n_dim < 3) ? 1 : (size_t)buf->shape[2]; // Number of elements on third dimension
    *d4 = (n_dim < 4) ? 1 : (size_t)buf->shape[3]; // Number of elements on fourth dimension

    if (n_dim < 2) // 0D and 1D can be considered column-major trivially
        return true;

    for (size_t i = 1; i < n_dim; ++i)
        if (buf->strides[i - 1] > buf->strides[i])
            return false;

    return true;
}

// Conversion function from row to columns-major order
template <typename dtype>
inline void convert_row2col_major(const dtype *ptr_in, dtype *ptr_out,
                                  size_t d1 = 1, size_t d2 = 1, size_t d3 = 1, size_t d4 = 1)
{
    for (size_t i = 0; i < d1; ++i)
        for (size_t j = 0; j < d2; ++j)
            for (size_t k = 0; k < d3; ++k)
                for (size_t l = 0; l < d4; ++l)
                {
                    size_t row_major_index = ((i * d2 + j) * d3 + k) * d4 + l;
                    size_t col_major_index = i + d1 * (j + d2 * (k + d3 * l));
                    ptr_out[col_major_index] = ptr_in[row_major_index];
                }
}

template <typename dtype>
inline arma::Col<dtype> qd_python_NPArray_to_Col(const pybind11::array_t<dtype> *input)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return arma::Col<dtype>();

    if (is_col_major)
        return arma::Col<dtype>(reinterpret_cast<dtype *>(buf.ptr), d1 * d2 * d3 * d4, false, true);

    arma::Col<dtype> output = arma::Col<dtype>(d1 * d2 * d3 * d4, arma::fill::none);
    convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), output.memptr(), d1, d2, d3, d4);
    return output;
}

template <typename dtype>
inline arma::Mat<dtype> qd_python_NPArray_to_Mat(const pybind11::array_t<dtype> *input)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return arma::Mat<dtype>();

    if (is_col_major)
        return arma::Mat<dtype>(reinterpret_cast<dtype *>(buf.ptr), d1, d2 * d3 * d4, false, true);

    arma::Mat<dtype> output = arma::Mat<dtype>(d1, d2 * d3 * d4, arma::fill::none);
    convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), output.memptr(), d1, d2, d3, d4);
    return output;
}

template <typename dtype>
inline arma::Cube<dtype> qd_python_NPArray_to_Cube(const pybind11::array_t<dtype> *input)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return arma::Cube<dtype>();

    if (is_col_major)
        return arma::Cube<dtype>(reinterpret_cast<dtype *>(buf.ptr), d1, d2, d3 * d4, false, true);

    arma::Cube<dtype> output = arma::Cube<dtype>(d1, d2, d3 * d4, arma::fill::none);
    convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), output.memptr(), d1, d2, d3, d4);
    return output;
}

template <typename dtype>
inline std::vector<arma::Col<dtype>> qd_python_NPArray_to_vectorCol(const pybind11::array_t<dtype> *input, size_t vec_dim)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return std::vector<arma::Col<dtype>>();

    dtype *ptr_reorg = new dtype[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const dtype *ptr_i = is_col_major ? reinterpret_cast<dtype *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    auto output = std::vector<arma::Col<dtype>>();
    if (vec_dim == 0)
        for (size_t n = 0; n < d1; ++n)
        {
            auto tmp = arma::Col<dtype>(d2 * d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d2 * d3 * d4; ++m)
                ptr[m] = ptr_i[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (size_t n = 0; n < d2; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();

            for (size_t m34 = 0; m34 < d3 * d4; ++m34)
                for (size_t m1 = 0; m1 < d1; ++m1)
                    ptr[m34 * d1 + m1] = ptr_i[m34 * d2 * d1 + n * d1 + m1];

            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (size_t n = 0; n < d3; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d2 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m4 = 0; m4 < d4; ++m4)
                for (size_t m12 = 0; m12 < d1 * d2; ++m12)
                    ptr[m4 * d2 * d1 + m12] = ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (size_t n = 0; n < d4; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d2 * d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d1 * d2 * d3; ++m)
                ptr[m] = ptr_i[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        throw std::invalid_argument("Armadillo object dimensions must be 0,1,2 or 3");

    delete[] ptr_reorg;
    return output;
}

template <typename dtype>
inline std::vector<arma::Mat<dtype>> qd_python_NPArray_to_vectorMat(const pybind11::array_t<dtype> *input, size_t vec_dim)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return std::vector<arma::Mat<dtype>>();

    dtype *ptr_reorg = new dtype[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const dtype *ptr_i = is_col_major ? reinterpret_cast<dtype *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    auto output = std::vector<arma::Mat<dtype>>();
    if (vec_dim == 0)
        for (size_t n = 0; n < d1; ++n)
        {
            auto tmp = arma::Mat<dtype>(d2, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d2 * d3 * d4; ++m)
                ptr[m] = ptr_i[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (size_t n = 0; n < d2; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m34 = 0; m34 < d3 * d4; ++m34)
                for (size_t m1 = 0; m1 < d1; ++m1)
                    ptr[m34 * d1 + m1] = ptr_i[m34 * d2 * d1 + n * d1 + m1];
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (size_t n = 0; n < d3; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d2 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m4 = 0; m4 < d4; ++m4)
                for (size_t m12 = 0; m12 < d1 * d2; ++m12)
                    ptr[m4 * d2 * d1 + m12] = ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (size_t n = 0; n < d4; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d2 * d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d1 * d2 * d3; ++m)
                ptr[m] = ptr_i[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        throw std::invalid_argument("Armadillo object dimensions must be 0,1,2 or 3");

    delete[] ptr_reorg;
    return output;
}

template <typename dtype>
inline std::vector<arma::Cube<dtype>> qd_python_NPArray_to_vectorCube(const pybind11::array_t<dtype> *input, size_t vec_dim)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return std::vector<arma::Cube<dtype>>();

    dtype *ptr_reorg = new dtype[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<dtype *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const dtype *ptr_i = is_col_major ? reinterpret_cast<dtype *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    auto output = std::vector<arma::Cube<dtype>>();
    if (vec_dim == 0)
        for (size_t n = 0; n < d1; ++n)
        {
            auto tmp = arma::Cube<dtype>(d2, d3, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d2 * d3 * d4; ++m)
                ptr[m] = ptr_i[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (size_t n = 0; n < d2; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d3, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m34 = 0; m34 < d3 * d4; ++m34)
                for (size_t m1 = 0; m1 < d1; ++m1)
                    ptr[m34 * d1 + m1] = ptr_i[m34 * d2 * d1 + n * d1 + m1];
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (size_t n = 0; n < d3; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d2, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m4 = 0; m4 < d4; ++m4)
                for (size_t m12 = 0; m12 < d1 * d2; ++m12)
                    ptr[m4 * d2 * d1 + m12] = ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (size_t n = 0; n < d4; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d2, d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d1 * d2 * d3; ++m)
                ptr[m] = ptr_i[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        throw std::invalid_argument("Armadillo object dimensions must be 0,1,2 or 3");

    delete[] ptr_reorg;
    return output;
}

template <typename dtype>
inline void qd_python_complexNPArray_to_2Mat(const pybind11::array_t<std::complex<dtype>> *input,
                                             arma::Mat<dtype> *output_re,
                                             arma::Mat<dtype> *output_im)
{
    if (output_re == nullptr || output_im == nullptr)
        throw std::invalid_argument("Outputs cannot be NULL");

    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
    {
        output_re->reset();
        output_im->reset();
        return;
    }

    std::complex<dtype> *ptr_reorg = new std::complex<dtype>[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<std::complex<dtype> *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const std::complex<dtype> *ptr_i = is_col_major ? reinterpret_cast<std::complex<dtype> *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    output_re->set_size(d1, d2 * d3 * d4);
    output_im->set_size(d1, d2 * d3 * d4);
    dtype *ptr_re = output_re->memptr();
    dtype *ptr_im = output_im->memptr();
    for (size_t m = 0; m < d1 * d2 * d3 * d4; ++m)
    {
        std::complex<dtype> value = ptr_i[m];
        ptr_re[m] = value.real();
        ptr_im[m] = value.imag();
    }

    delete[] ptr_reorg;
    return;
}

template <typename dtype>
inline void qd_python_complexNPArray_to_2Cubes(const pybind11::array_t<std::complex<dtype>> *input,
                                               arma::Cube<dtype> *output_re,
                                               arma::Cube<dtype> *output_im)
{
    if (output_re == nullptr || output_im == nullptr)
        throw std::invalid_argument("Outputs cannot be NULL");

    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
    {
        output_re->reset();
        output_im->reset();
        return;
    }

    std::complex<dtype> *ptr_reorg = new std::complex<dtype>[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<std::complex<dtype> *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const std::complex<dtype> *ptr_i = is_col_major ? reinterpret_cast<std::complex<dtype> *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    output_re->set_size(d1, d2, d3 * d4);
    output_im->set_size(d1, d2, d3 * d4);
    dtype *ptr_re = output_re->memptr();
    dtype *ptr_im = output_im->memptr();
    for (size_t m = 0; m < d1 * d2 * d3 * d4; ++m)
    {
        std::complex<dtype> value = ptr_i[m];
        ptr_re[m] = value.real();
        ptr_im[m] = value.imag();
    }

    delete[] ptr_reorg;
    return;
}

// Convert a complex NParray to an vector of Matrices with interleaved real / imaginary parts
template <typename dtype>
inline std::vector<arma::Mat<dtype>> qd_python_complexNPArray_to_vectorMat(const pybind11::array_t<std::complex<dtype>> *input, size_t vec_dim)
{
    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return std::vector<arma::Mat<dtype>>();

    std::complex<dtype> *ptr_reorg = new std::complex<dtype>[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<std::complex<dtype> *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const std::complex<dtype> *ptr_i = is_col_major ? reinterpret_cast<std::complex<dtype> *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    auto output = std::vector<arma::Mat<dtype>>();
    if (vec_dim == 0)
        for (size_t n = 0; n < d1; ++n)
        {
            auto tmp = arma::Mat<dtype>(2 * d2, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d2 * d3 * d4; ++m)
            {
                std::complex<dtype> value = ptr_i[m * d1 + n];
                ptr[2 * m] = value.real();
                ptr[2 * m + 1] = value.imag();
            }
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (size_t n = 0; n < d2; ++n)
        {
            auto tmp = arma::Mat<dtype>(2 * d1, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m34 = 0; m34 < d3 * d4; ++m34)
                for (size_t m1 = 0; m1 < d1; ++m1)
                {
                    size_t i = m34 * d1 + m1;
                    std::complex<dtype> value = ptr_i[m34 * d2 * d1 + n * d1 + m1];
                    ptr[2 * i] = value.real();
                    ptr[2 * i + 1] = value.imag();
                }
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (size_t n = 0; n < d3; ++n)
        {
            auto tmp = arma::Mat<dtype>(2 * d1, d2 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m4 = 0; m4 < d4; ++m4)
                for (size_t m12 = 0; m12 < d1 * d2; ++m12)
                {
                    size_t i = m4 * d2 * d1 + m12;
                    std::complex<dtype> value = ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
                    ptr[2 * i] = value.real();
                    ptr[2 * i + 1] = value.imag();
                }
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (size_t n = 0; n < d4; ++n)
        {
            auto tmp = arma::Mat<dtype>(2 * d1, d2 * d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            for (size_t m = 0; m < d1 * d2 * d3; ++m)
            {
                std::complex<dtype> value = ptr_i[n * d3 * d2 * d1 + m];
                ptr[2 * m] = value.real();
                ptr[2 * m + 1] = value.imag();
            }
            output.push_back(tmp);
        }
    else
        throw std::invalid_argument("Armadillo object dimensions must be 0,1,2 or 3");

    delete[] ptr_reorg;
    return output;
}

template <typename dtype>
inline void qd_python_complexNPArray_to_2vectorCube(const pybind11::array_t<std::complex<dtype>> *input, size_t vec_dim,
                                                    std::vector<arma::Cube<dtype>> *output_re,
                                                    std::vector<arma::Cube<dtype>> *output_im)
{
    if (output_re == nullptr || output_im == nullptr)
        throw std::invalid_argument("Outputs cannot be NULL");

    output_re->clear();
    output_im->clear();

    pybind11::buffer_info buf = input->request();
    size_t d1, d2, d3, d4;
    bool is_col_major = get_dims(&buf, &d1, &d2, &d3, &d4);
    size_t n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return;

    std::complex<dtype> *ptr_reorg = new std::complex<dtype>[n_data];
    if (!is_col_major)
        convert_row2col_major(reinterpret_cast<std::complex<dtype> *>(buf.ptr), ptr_reorg, d1, d2, d3, d4);
    const std::complex<dtype> *ptr_i = is_col_major ? reinterpret_cast<std::complex<dtype> *>(buf.ptr) : ptr_reorg;

    // Convert data to armadillo output
    if (vec_dim == 0)
        for (size_t n = 0; n < d1; ++n)
        {
            auto tmp_re = arma::Cube<dtype>(d2, d3, d4, arma::fill::none);
            auto tmp_im = arma::Cube<dtype>(d2, d3, d4, arma::fill::none);
            dtype *ptr_re = tmp_re.memptr();
            dtype *ptr_im = tmp_im.memptr();
            for (size_t m = 0; m < d2 * d3 * d4; ++m)
            {
                std::complex<dtype> value = ptr_i[m * d1 + n];
                ptr_re[m] = value.real();
                ptr_im[m] = value.imag();
            }
            output_re->push_back(tmp_re);
            output_im->push_back(tmp_im);
        }
    else if (vec_dim == 1)
        for (size_t n = 0; n < d2; ++n)
        {
            auto tmp_re = arma::Cube<dtype>(d1, d3, d4, arma::fill::none);
            auto tmp_im = arma::Cube<dtype>(d1, d3, d4, arma::fill::none);
            dtype *ptr_re = tmp_re.memptr();
            dtype *ptr_im = tmp_im.memptr();
            for (size_t m34 = 0; m34 < d3 * d4; ++m34)
                for (size_t m1 = 0; m1 < d1; ++m1)
                {
                    std::complex<dtype> value = ptr_i[m34 * d2 * d1 + n * d1 + m1];
                    ptr_re[m34 * d1 + m1] = value.real();
                    ptr_im[m34 * d1 + m1] = value.imag();
                }
            output_re->push_back(tmp_re);
            output_im->push_back(tmp_im);
        }
    else if (vec_dim == 2)
        for (size_t n = 0; n < d3; ++n)
        {
            auto tmp_re = arma::Cube<dtype>(d1, d2, d4, arma::fill::none);
            auto tmp_im = arma::Cube<dtype>(d1, d2, d4, arma::fill::none);
            dtype *ptr_re = tmp_re.memptr();
            dtype *ptr_im = tmp_im.memptr();
            for (size_t m4 = 0; m4 < d4; ++m4)
                for (size_t m12 = 0; m12 < d1 * d2; ++m12)
                {
                    std::complex<dtype> value = ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
                    ptr_re[m4 * d2 * d1 + m12] = value.real();
                    ptr_im[m4 * d2 * d1 + m12] = value.imag();
                }
            output_re->push_back(tmp_re);
            output_im->push_back(tmp_im);
        }
    else if (vec_dim == 3)
        for (size_t n = 0; n < d4; ++n)
        {
            auto tmp_re = arma::Cube<dtype>(d1, d2, d3, arma::fill::none);
            auto tmp_im = arma::Cube<dtype>(d1, d2, d3, arma::fill::none);
            dtype *ptr_re = tmp_re.memptr();
            dtype *ptr_im = tmp_im.memptr();
            for (size_t m = 0; m < d1 * d2 * d3; ++m)
            {
                std::complex<dtype> value = ptr_i[n * d3 * d2 * d1 + m];
                ptr_re[m] = value.real();
                ptr_im[m] = value.imag();
            }
            output_re->push_back(tmp_re);
            output_im->push_back(tmp_im);
        }
    else
        throw std::invalid_argument("Armadillo object dimensions must be 0,1,2 or 3");

    delete[] ptr_reorg;
    return;
}

// Copy: Vector
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_Col_to_NPArray(const arma::Col<dtype> *input,   // Column Vector
                                                         arma::uword ns = 0,              // Number of elements in output
                                                         const arma::uword *is = nullptr, // List of elements to copy, 0-based
                                                         bool transpose = false)          // Transpose output
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    ns = (ns == 0) ? input->n_elem : ns;

    auto output = transpose ? pybind11::array_t<dtype>({(arma::uword)1, ns})
                            : pybind11::array_t<dtype>(ns);

    auto buf = output.request();
    dtype *ptr_o = static_cast<dtype *>(buf.ptr);
    const dtype *ptr_i = input->memptr();

    if (is == nullptr) // Copy all
        std::memcpy(ptr_o, ptr_i, sizeof(dtype) * ns);
    else // Copy selected
        for (arma::uword i = 0; i < ns; ++i)
            ptr_o[i] = (is[i] >= input->n_elem) ? *ptr_i : ptr_i[is[i]];

    return output;
}

// Copy: Matrix
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_Mat_to_NPArray(const arma::Mat<dtype> *input,   // Matrix
                                                         arma::uword ns = 0,              // Number of elements in output
                                                         const arma::uword *is = nullptr) // List of elements to copy, 0-based
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    arma::uword m = input->n_rows;       // Rows
    ns = (ns == 0) ? input->n_cols : ns; // Output columns

    size_t strides[2] = {sizeof(dtype), m * sizeof(dtype)};
    auto output = pybind11::array_t<dtype>({m, ns}, strides);
    auto buf = output.request();
    dtype *ptr_o = static_cast<dtype *>(buf.ptr);

    if (is == nullptr) // Copy all
        std::memcpy(ptr_o, input->memptr(), sizeof(dtype) * input->n_elem);
    else // Copy selected
        for (arma::uword i = 0; i < ns; ++i)
        {
            arma::uword k = (is[i] >= input->n_cols) ? 0 : is[i];
            std::memcpy(&ptr_o[i * m], input->colptr(k), sizeof(dtype) * m);
        }

    return output;
}

// Copy: Cube
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_Cube_to_NPArray(const arma::Cube<dtype> *input,  // Cube
                                                          arma::uword ns = 0,              // Number of elements in output
                                                          const arma::uword *is = nullptr) // List of elements to copy, 0-based
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    arma::uword m = input->n_rows * input->n_cols; // Rows and columns
    ns = (ns == 0) ? input->n_slices : ns;         // Slices

    size_t strides[3] = {sizeof(dtype), input->n_rows * sizeof(dtype), m * sizeof(dtype)};
    auto output = pybind11::array_t<dtype>({input->n_rows, input->n_cols, ns}, strides);
    auto buf = output.request();
    dtype *ptr_o = static_cast<dtype *>(buf.ptr);

    if (is == nullptr) // Copy all
        std::memcpy(ptr_o, input->memptr(), sizeof(dtype) * input->n_elem);
    else // Copy selected
        for (arma::uword i = 0; i < ns; ++i)
        {
            arma::uword k = (is[i] >= input->n_slices) ? 0 : is[i];
            std::memcpy(&ptr_o[i * m], input->slice_memptr(k), sizeof(dtype) * m);
        }

    return output;
}

// Copy: Complex Matrix
template <typename dtype>
inline pybind11::array_t<std::complex<dtype>> qd_python_2Mat_to_complexNPArray(const arma::Mat<dtype> *input_re, // Matrix, Real part
                                                                               const arma::Mat<dtype> *input_im, // Matrix, Imaginary Part
                                                                               arma::uword n_col_out = 0,        // Number of elements in output
                                                                               const arma::uword *is = nullptr)  // List of elements to copy, 0-based
{
    if (input_re == nullptr || input_im == nullptr)
        throw std::invalid_argument("Input data cannot be NULL");

    if (input_re->empty())
        return pybind11::array_t<std::complex<dtype>>();

    if (input_re->n_rows != input_im->n_rows || input_re->n_cols != input_im->n_cols)
        throw std::invalid_argument("Sizes of real and imaginary parts don't match.");

    arma::uword n_row = input_re->n_rows;                        // Rows
    n_col_out = (n_col_out == 0) ? input_re->n_cols : n_col_out; // Output columns

    size_t strides[2] = {sizeof(std::complex<dtype>), n_row * sizeof(std::complex<dtype>)};
    auto output = pybind11::array_t<std::complex<dtype>>({n_row, n_col_out}, strides);
    auto buf = output.request();
    std::complex<dtype> *ptr_o = static_cast<std::complex<dtype> *>(buf.ptr);

    for (arma::uword i_col_out = 0; i_col_out < n_col_out; ++i_col_out) // Col loop
    {
        arma::uword i_col_in = i_col_out;
        if (is != nullptr)
            i_col_in = (is[i_col_out] >= input_re->n_cols) ? 0 : is[i_col_out];

        const dtype *ptr_i_re = input_re->colptr(i_col_in);
        const dtype *ptr_i_im = input_im->colptr(i_col_in);

        for (arma::uword i_row = 0; i_row < n_row; ++i_row) // Row loop
        {
            arma::uword offset = i_col_out * n_row + i_row;
            ptr_o[offset].real(ptr_i_re[i_row]);
            ptr_o[offset].imag(ptr_i_im[i_row]);
        }
    }

    return output;
}

// Copy: Complex Cube
template <typename dtype>
inline pybind11::array_t<std::complex<dtype>> qd_python_2Cubes_to_complexNPArray(const arma::Cube<dtype> *input_re, // Cube, Real part
                                                                                 const arma::Cube<dtype> *input_im, // Cube, Imaginary Part
                                                                                 arma::uword n_slices_out = 0,      // Number of elements in output
                                                                                 const arma::uword *is = nullptr)   // List of elements to copy, 0-based
{
    if (input_re == nullptr || input_im == nullptr)
        throw std::invalid_argument("Input data cannot be NULL");

    if (input_re->empty())
        return pybind11::array_t<std::complex<dtype>>();

    if (input_re->n_rows != input_im->n_rows || input_re->n_cols != input_im->n_cols || input_re->n_slices != input_im->n_slices)
        throw std::invalid_argument("Sizes of real and imaginary parts don't match.");

    arma::uword n_elem_per_slice = input_re->n_rows * input_re->n_cols;     // No elements per slice
    n_slices_out = (n_slices_out == 0) ? input_re->n_slices : n_slices_out; // Slices

    size_t strides[3] = {sizeof(std::complex<dtype>), input_re->n_rows * sizeof(std::complex<dtype>), n_elem_per_slice * sizeof(std::complex<dtype>)};
    auto output = pybind11::array_t<std::complex<dtype>>({input_re->n_rows, input_re->n_cols, n_slices_out}, strides);
    auto buf = output.request();
    std::complex<dtype> *ptr_o = static_cast<std::complex<dtype> *>(buf.ptr);

    for (arma::uword i_slice_out = 0; i_slice_out < n_slices_out; ++i_slice_out)
    {
        arma::uword i_slice_in = i_slice_out;
        if (is != nullptr)
            i_slice_in = (is[i_slice_out] >= input_re->n_slices) ? 0 : is[i_slice_out];

        const dtype *ptr_i_re = input_re->slice_memptr(i_slice_in);
        const dtype *ptr_i_im = input_im->slice_memptr(i_slice_in);

        for (arma::uword i_elem = 0; i_elem < n_elem_per_slice; ++i_elem)
        {
            arma::uword offset = i_slice_out * n_elem_per_slice + i_elem;
            ptr_o[offset].real(ptr_i_re[i_elem]);
            ptr_o[offset].imag(ptr_i_im[i_elem]);
        }
    }

    return output;
}

// Copy: Vector of Vectors
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_vectorCol_to_NPArray(const std::vector<arma::Col<dtype>> *input,
                                                               arma::uword ns = 0,              // Number of elements in output
                                                               const arma::uword *is = nullptr, // List of elements to copy, 0-based
                                                               dtype padding = (dtype)0)        // Data used as padding
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    // Get maximum input data dimensions
    arma::uword m = 0;
    for (auto &v : *input)
        m = (v.n_elem > m) ? v.n_elem : m;
    ns = (ns == 0 || is == nullptr) ? (arma::uword)input->size() : ns;

    size_t strides[2] = {sizeof(dtype), m * sizeof(dtype)};
    auto output = pybind11::array_t<dtype>({m, ns}, strides);
    auto buf = output.request();
    dtype *ptr = static_cast<dtype *>(buf.ptr);

    // Get snapshot range
    arma::uword *js = new arma::uword[ns];
    if (is == nullptr)
        for (arma::uword i = 0; i < ns; ++i)
            js[i] = i;
    else
        std::memcpy(js, is, ns * sizeof(arma::uword));

    // Copy data
    for (arma::uword i = 0; i < ns; ++i)
    {
        arma::uword k = (js[i] >= input->size()) ? 0 : js[i];
        arma::uword r = input->at(k).n_elem;

        if (r != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r);
    }

    delete[] js;
    return output;
}

// Copy: Vector of Matrices
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_vectorMat_to_NPArray(const std::vector<arma::Mat<dtype>> *input,
                                                               arma::uword ns = 0,              // Number of elements in output
                                                               const arma::uword *is = nullptr, // List of elements to copy, 0-based
                                                               dtype padding = (dtype)0)        // Data used as padding
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    // Get maximum input data dimensions
    arma::uword n_rows = 0, n_cols = 0;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols;

    arma::uword m = n_rows * n_cols;
    ns = (ns == 0 || is == nullptr) ? (arma::uword)input->size() : ns;

    size_t strides[3] = {sizeof(dtype), n_rows * sizeof(dtype), m * sizeof(dtype)};
    auto output = pybind11::array_t<dtype>({n_rows, n_cols, ns}, strides);
    auto buf = output.request();
    dtype *ptr = static_cast<dtype *>(buf.ptr);

    // Get snapshot range
    arma::uword *js = new arma::uword[ns];
    if (is == nullptr)
        for (arma::uword i = 0; i < ns; ++i)
            js[i] = i;
    else
        std::memcpy(js, is, ns * sizeof(arma::uword));

    // Copy data
    for (arma::uword i = 0; i < ns; ++i)
    {
        arma::uword k = (js[i] >= (arma::uword)input->size()) ? 0 : js[i];
        arma::uword r = input->at(k).n_rows, c = input->at(k).n_cols;

        if (r * c != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        if (r == n_rows)
            std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r * c);
        else // Copy column by column
            for (arma::uword ic = 0; ic < c; ++ic)
                std::memcpy(&ptr[i * m + ic * n_rows],
                            input->at(k).colptr(ic), sizeof(dtype) * r);
    }

    delete[] js;
    return output;
}

// Copy: Vector of Matrices (convert to Complex)
template <typename dtype>
inline pybind11::array_t<std::complex<dtype>> qd_python_vectorMat_to_complexNPArray(const std::vector<arma::Mat<dtype>> *input,
                                                                                    arma::uword ns = 0,              // Number of elements in output
                                                                                    const arma::uword *is = nullptr) // List of elements to copy, 0-based
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<std::complex<dtype>>();

    // Get maximum input data dimensions
    arma::uword n_rows = 0, n_cols = 0;
    for (auto &v : *input)
    {
        if (v.n_rows % 2 == 1)
            throw std::invalid_argument("Number of rows must be a multiple of 2 for real to complex conversion.");
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows;
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols;
    }

    n_rows /= 2;
    arma::uword m = n_rows * n_cols;
    ns = (ns == 0 || is == nullptr) ? (arma::uword)input->size() : ns;

    size_t strides[3] = {sizeof(std::complex<dtype>), n_rows * sizeof(std::complex<dtype>), m * sizeof(std::complex<dtype>)};
    auto output = pybind11::array_t<std::complex<dtype>>({n_rows, n_cols, ns}, strides);
    auto buf = output.request();
    std::complex<dtype> *ptr = static_cast<std::complex<dtype> *>(buf.ptr);

    // Get snapshot range
    arma::uword *js = new arma::uword[ns];
    if (is == nullptr)
        for (arma::uword i = 0; i < ns; ++i)
            js[i] = i;
    else
        std::memcpy(js, is, ns * sizeof(arma::uword));

    // Copy data
    for (arma::uword i = 0; i < ns; ++i)
    {
        arma::uword k = (js[i] >= (arma::uword)input->size()) ? 0 : js[i];
        arma::uword r = input->at(k).n_rows / 2, c = input->at(k).n_cols;
        const dtype *ptr_i = input->at(k).memptr();

        if (r * c != m)
            for (std::complex<dtype> *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                (*p).real(0.0), (*p).imag(0.0);

        for (arma::uword ic = 0; ic < c; ++ic)
            for (arma::uword ir = 0; ir < r; ++ir)
            {
                arma::uword offset = i * m + ic * n_rows + ir;
                ptr[offset].real(ptr_i[2 * ic * n_rows + 2 * ir]);
                ptr[offset].imag(ptr_i[2 * ic * n_rows + 2 * ir + 1]);
            }
    }

    delete[] js;
    return output;
}

// Copy: Vector of Cubes
template <typename dtype>
inline pybind11::array_t<dtype> qd_python_vectorCube_to_NPArray(const std::vector<arma::Cube<dtype>> *input,
                                                                arma::uword ns = 0,              // Number of elements in output
                                                                const arma::uword *is = nullptr, // List of elements to copy, 0-based
                                                                dtype padding = (dtype)0)        // Data used as padding
{
    if (input == nullptr || input->empty())
        return pybind11::array_t<dtype>();

    // Get maximum input data dimensions
    arma::uword n_rows = 0, n_cols = 0, n_slices = 0;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols,
        n_slices = v.n_slices > n_slices ? v.n_slices : n_slices;

    arma::uword m = n_rows * n_cols * n_slices;
    ns = (ns == 0 || is == nullptr) ? (arma::uword)input->size() : ns;

    size_t dims[4] = {(size_t)n_rows, (size_t)n_cols, (size_t)n_slices, (size_t)ns};
    size_t strides[4] = {sizeof(dtype), n_rows * sizeof(dtype), n_rows * n_cols * sizeof(dtype), m * sizeof(dtype)};

    auto output = pybind11::array_t<dtype>(dims, strides);
    auto buf = output.request();
    dtype *ptr = static_cast<dtype *>(buf.ptr);

    // Get snapshot range
    arma::uword *js = new arma::uword[ns];
    if (is == nullptr)
        for (arma::uword i = 0; i < ns; ++i)
            js[i] = i;
    else
        std::memcpy(js, is, ns * sizeof(arma::uword));

    // Copy data
    for (arma::uword i = 0; i < ns; ++i)
    {
        arma::uword k = (js[i] >= input->size()) ? 0 : js[i];
        arma::uword r = input->at(k).n_rows, c = input->at(k).n_cols, s = input->at(k).n_slices;

        if (r * c * s != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        if (r == n_rows && c == n_cols)
            std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r * c * s);
        else // Copy column by column
            for (arma::uword is = 0; is < s; ++is)
                for (arma::uword ic = 0; ic < c; ++ic)
                    std::memcpy(&ptr[i * m + is * n_rows * n_cols + ic * n_rows],
                                input->at(k).slice_colptr(is, ic), sizeof(dtype) * r);
    }

    delete[] js;
    return output;
}

// Copy: Vector of Complex Cubes
template <typename dtype>
inline pybind11::array_t<std::complex<dtype>> qd_python_2vectorCubes_to_complexNPArray(const std::vector<arma::Cube<dtype>> *input_re,
                                                                                       const std::vector<arma::Cube<dtype>> *input_im,
                                                                                       arma::uword ns = 0,              // Number of elements in output
                                                                                       const arma::uword *is = nullptr) // List of elements to copy, 0-based
{
    if (input_re == nullptr || input_im == nullptr)
        throw std::invalid_argument("Input data cannot be NULL");

    if (input_re->empty())
        return pybind11::array_t<std::complex<dtype>>();

    if (input_re->size() != input_im->size())
        throw std::invalid_argument("Number of snapshots in real and imaginary parts don't match.");

    for (size_t i = 0; i < input_re->size(); ++i)
        if (input_re->at(i).n_rows != input_im->at(i).n_rows ||
            input_re->at(i).n_cols != input_im->at(i).n_cols ||
            input_re->at(i).n_slices != input_im->at(i).n_slices)
            throw std::invalid_argument("Sizes of real and imaginary parts don't match.");

    // Get maximum input data dimensions
    arma::uword n_rows = 0, n_cols = 0, n_slices = 0;
    for (auto &v : *input_re)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols,
        n_slices = v.n_slices > n_slices ? v.n_slices : n_slices;

    arma::uword m = n_rows * n_cols * n_slices;
    ns = (ns == 0 || is == nullptr) ? (arma::uword)input_re->size() : ns;

    size_t dims[4] = {(size_t)n_rows, (size_t)n_cols, (size_t)n_slices, (size_t)ns};
    size_t strides[4] = {sizeof(std::complex<dtype>), n_rows * sizeof(std::complex<dtype>),
                         n_rows * n_cols * sizeof(std::complex<dtype>), m * sizeof(std::complex<dtype>)};

    auto output = pybind11::array_t<std::complex<dtype>>(dims, strides);
    auto buf = output.request();
    std::complex<dtype> *ptr = static_cast<std::complex<dtype> *>(buf.ptr);

    // Get snapshot range
    arma::uword *js = new arma::uword[ns];
    if (is == nullptr)
        for (arma::uword i = 0; i < ns; ++i)
            js[i] = i;
    else
        std::memcpy(js, is, ns * sizeof(arma::uword));

    // Copy data
    for (arma::uword i = 0; i < ns; ++i)
    {
        arma::uword k = (js[i] >= (arma::uword)input_re->size()) ? 0 : js[i];
        arma::uword r = input_re->at(k).n_rows, c = input_re->at(k).n_cols, s = input_re->at(k).n_slices;

        if (r * c * s != m)
            for (std::complex<dtype> *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                (*p).real(0.0), (*p).imag(0.0);

        const dtype *ptr_i_re = input_re->at(k).memptr();
        const dtype *ptr_i_im = input_im->at(k).memptr();

        size_t j = 0;
        for (arma::uword is = 0; is < s; ++is)
            for (arma::uword ic = 0; ic < c; ++ic)
                for (arma::uword ir = 0; ir < r; ++ir)
                {
                    arma::uword offset = i * m + is * n_rows * n_cols + ic * n_rows + ir;
                    ptr[offset].real(ptr_i_re[j]);
                    ptr[offset].imag(ptr_i_im[j++]);
                }
    }

    delete[] js;
    return output;
}

// Convert to std::any
inline std::any qd_python_anycast(pybind11::handle obj, std::string var_name = "")
{

    if (pybind11::isinstance<pybind11::str>(obj))
        return pybind11::cast<std::string>(obj);

    if (pybind11::isinstance<pybind11::float_>(obj))
        return pybind11::cast<double>(obj);

    if (pybind11::isinstance<pybind11::int_>(obj))
        return pybind11::cast<long long>(obj);

    if (pybind11::isinstance<pybind11::array>(obj))
    {
        pybind11::array arr = pybind11::cast<pybind11::array>(obj);
        pybind11::buffer_info buf = arr.request();

        if (arr.dtype().is(pybind11::dtype::of<double>()))
        {
            auto arr_t = pybind11::array_t<double>(pybind11::reinterpret_borrow<pybind11::array_t<double>>(arr));
            if (buf.ndim == 1)
                return arma::Row<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<double>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<double>(&arr_t);
        }

        if (arr.dtype().is(pybind11::dtype::of<float>()))
        {
            auto arr_t = pybind11::array_t<float>(pybind11::reinterpret_borrow<pybind11::array_t<float>>(arr));
            if (buf.ndim == 1)
                return arma::Row<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<float>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<float>(&arr_t);
        }

        if (arr.dtype().is(pybind11::dtype::of<unsigned>()))
        {
            auto arr_t = pybind11::array_t<unsigned>(pybind11::reinterpret_borrow<pybind11::array_t<unsigned>>(arr));
            if (buf.ndim == 1)
                return arma::Row<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<unsigned>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<unsigned>(&arr_t);
        }

        if (arr.dtype().is(pybind11::dtype::of<int>()))
        {
            auto arr_t = pybind11::array_t<int>(pybind11::reinterpret_borrow<pybind11::array_t<int>>(arr));
            if (buf.ndim == 1)
                return arma::Row<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<int>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<int>(&arr_t);
        }

        if (arr.dtype().is(pybind11::dtype::of<unsigned long long>()))
        {
            auto arr_t = pybind11::array_t<unsigned long long>(pybind11::reinterpret_borrow<pybind11::array_t<unsigned long long>>(arr));
            if (buf.ndim == 1)
                return arma::Row<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<unsigned long long>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<unsigned long long>(&arr_t);
        }

        if (arr.dtype().is(pybind11::dtype::of<long long>()) || arr.dtype().is(pybind11::dtype::of<int64_t>()))
        {
            auto arr_t = pybind11::array_t<long long>(pybind11::reinterpret_borrow<pybind11::array_t<long long>>(arr));
            if (buf.ndim == 1)
                return arma::Row<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_NPArray_to_Mat<long long>(&arr_t);
            else if (buf.ndim == 3)
                return qd_python_NPArray_to_Cube<long long>(&arr_t);
        }
    }

    std::string error_message = "Input '" + var_name + "' has an unsupported type.";
    throw std::invalid_argument(error_message);

    return std::any();
}

#endif
