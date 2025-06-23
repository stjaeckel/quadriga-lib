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

#ifndef quadriga_python_arma_adapter_H
#define quadriga_python_arma_adapter_H

#include <any>
#include <armadillo>
#include <string>
#include <cstring>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*
Cheat sheet:

- Copy from Armadillo types to Numpy array (auto = py::array_t<dtype>)

auto pyarray = qd_python_copy2numpy(Col, transpose, i_elem);        // Copy column vector
auto pyarray = qd_python_copy2numpy(Mat, i_col);                    // Copy matrix
auto pyarray = qd_python_copy2numpy(Cube, i_slice);                 // Copy cube

auto pyarray = qd_python_copy2numpy(MatRe, MatIm, i_col);           // Copy Re/Im matrices to complex
auto pyarray = qd_python_copy2numpy(CubeRe, CubeIm, i_slice);       // Copy Re/Im cubes to complex

auto pyarray = qd_python_copy2numpy<unsigned, ssize_t>(Col);        // Cast column vector from u32 to int
auto pyarray = qd_python_copy2numpy<unsigned, ssize_t>(Mat);        // Cast matrix from u32 to int

- Copy std::vector of Armadillo types to py::list of Numpy arrays

auto pylist = qd_python_copy2numpy(vecCol/Mat/Cube, i_vec);         // Copy vector of Col, Mat or Cubes
auto pylist = qd_python_copy2numpy(vecMatRe, vecMatIm, i_vec);      // Copy vector of Re/Im matrices to list of complex
auto pylist = qd_python_copy2numpy(vecCubeRe, vecCubeIm, i_vec);    // Copy vector of Re/Im cubes to list of complex
auto pylist = qd_python_copy2python(vecStrings, i_vec);             // Copy vector of strings

- Reserve memory in Python and map to Armadillo (Arma can write directly to Python, no copy needed)

auto pyarray = qd_python_init_output(n_rows, Col);                          // Init 1D and map to Col
auto pyarray = qd_python_init_output(n_rows, n_cols, Mat);                  // Init 2D and map to Mat
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, Cube);       // Init 3D and map to Cube
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, n_frames);   // Init 4D

- Parse shape of a Numpy array or list of Numpy arrays
  0: n_row, 1: n_cols, 2: n_slices, 3-5: Strides in number of elements, 6: Fortran-contiguous, 7: n_elem, 8: n_bytes

std::array<size_t, 9> shape = qd_python_get_shape(pyarray);         // For Numpy arrays

std::vector<dtype *> pointers;                                      // Direct memory pointers
std::vector<py::array_t<dtype>> owned_arrays;                       // Needed for type conversion
std::array<size_t, 9> shape = qd_python_get_list_shape(pylist, pointers, owned_arrays); // Lists

- Copy Numpy arrays to Armadillo Cubes (you can use a wrapper to get Mat/Col from Cubes)

qd_python_copy2arma(pointer, shape, outCube);                       // Copy from Python memory to Cube
qd_python_copy2arma(pointer, shape, outCubeRe, outCubeIm);          // Copy from Python memory (complex) to Cube Re/Im

qd_python_copy2arma(pyarray, outCube);                              // Copy from Python array to Cube
qd_python_copy2arma(pyarrayComplex, outCubeRe, outCubeIm);          // Copy from complex Python array to Cube Re/Im

- Convert Numpy arrays to Armadillo types
  view : (0, default) Copy data, (1) Get memory view if strides match, otherwise copy
  strict : Only applies if view = 1 : (0, default) Copy if strides don't match, (1) Error is strides don't match
  Data type (double, int, etc.) is obtained from pyarray and mapped to corresponding Arma type

auto Col = qd_python_numpy2arma_Col(pyarray, view, strict);         // Map to Col
auto Mat = qd_python_numpy2arma_Mat(pyarray, view, strict);         // Map to Mat
auto Cube = qd_python_numpy2arma_Cube(pyarray, view, strict);       // Map to Mat

- Copy py::list of Numpy arrays to std::vector of Armadillo types

auto vecCol = qd_python_list2vector_Col<dtype>(pylist);             // Copy py::list to std::vector<arma::Col<dtype>>
auto vecMat = qd_python_list2vector_Mat<dtype>(pylist);             // Copy py::list to std::vector<arma::Mat<dtype>>
auto vecCube = qd_python_list2vector_Cube<dtype>(pylist);           // Copy py::list to std::vector<arma::Cube<dtype>>
auto vecStrings = qd_python_list2vector_Strings(pylist);            // Copy py::list to std::vector<std::string>

qd_python_list2vector_Cube_Cplx<dtype>(pylistComplex, vecCubeRe, vecCubeIm);      // For Complex -> Re/Im conversion

- Convert Complex to Interleaved (and back)

auto Mat = qd_python_Complex2Interleaved(Mat_Complex);              // Convert arma::Mat<std::complex<dtype>> to arma::Mat<dtype>
auto vecMat = qd_python_Complex2Interleaved(vecMat_Complex);        // same for std::vector<arma::Mat<std::complex<dtype>>>

auto Mat_Complex = qd_python_Interleaved2Complex(Mat);              // Convert arma::Mat<dtype> to arma::Mat<std::complex<dtype>>
auto vecMat_Complex = qd_python_Interleaved2Complex(vecMat);        // same for std::vector<arma::Mat<dtype>>
*/

// -------------------------------- qd_python_copy2numpy --------------------------------

template <typename dtype>
py::array_t<dtype> qd_python_copy2numpy(const arma::Col<dtype> &input,  // Column Vector
                                        bool transpose = false,         // Transpose output
                                        const arma::uvec &indices = {}) // Optional indices to copy
{
    const ssize_t n_bytes = sizeof(dtype);
    const ssize_t n_elements = indices.empty() ? (ssize_t)input.n_elem : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape, strides;
    if (transpose)
        shape = {1, n_elements}, strides = {n_elements * n_bytes, n_bytes};
    else
        shape = {n_elements}, strides = {n_bytes};

    py::array_t<dtype> output(shape, strides);
    dtype *p_out = output.mutable_data();
    const dtype *p_in = input.memptr();

    if (indices.empty())
        std::memcpy(output.mutable_data(), input.memptr(), n_elements * n_bytes);
    else
        for (ssize_t i_el = 0; i_el < n_elements; ++i_el)
        {
            if (indices[i_el] >= input.n_elem)
                throw std::out_of_range("Index out of bound.");
            p_out[i_el] = p_in[indices[i_el]];
        }

    return output;
}

template <typename dtype_arma, typename dtype_numpy>
py::array_t<dtype_numpy> qd_python_copy2numpy(const arma::Col<dtype_arma> &input, // Column Vector
                                              bool transpose = false,             // Transpose output
                                              const arma::uvec &indices = {})     // Optional indices to copy
{
    const ssize_t n_bytes = sizeof(dtype_numpy);
    const ssize_t n_elements = indices.empty() ? (ssize_t)input.n_elem : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape, strides;
    if (transpose)
        shape = {1, n_elements}, strides = {n_elements * n_bytes, n_bytes};
    else
        shape = {n_elements}, strides = {n_bytes};

    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();
    const dtype_arma *p_in = input.memptr();

    if (indices.empty())
        for (ssize_t i_el = 0; i_el < n_elements; ++i_el)
            p_out[i_el] = (dtype_numpy)p_in[i_el];
    else
        for (ssize_t i_el = 0; i_el < n_elements; ++i_el)
        {
            if (indices[i_el] >= input.n_elem)
                throw std::out_of_range("Index out of bound.");
            p_out[i_el] = (dtype_numpy)p_in[indices[i_el]];
        }

    return output;
}

template <typename dtype>
py::array_t<dtype> qd_python_copy2numpy(const arma::Mat<dtype> &input,  // Matrix
                                        const arma::uvec &indices = {}) // Optional columns to copy
{
    const ssize_t n_bytes = sizeof(dtype);
    const ssize_t n_rows = (ssize_t)input.n_rows;
    const ssize_t n_cols = indices.empty() ? (ssize_t)input.n_cols : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape = {n_rows, n_cols}, strides = {n_bytes, n_rows * n_bytes};
    py::array_t<dtype> output(shape, strides);
    dtype *p_out = output.mutable_data();

    if (indices.empty())
        std::memcpy(p_out, input.memptr(), n_rows * n_cols * n_bytes);
    else
        for (ssize_t i_col = 0; i_col < n_cols; ++i_col)
        {
            if (indices[i_col] >= input.n_cols)
                throw std::out_of_range("Index out of bound.");
            std::memcpy(&p_out[i_col * n_rows], input.colptr(indices[i_col]), n_rows * n_bytes);
        }

    return output;
}

template <typename dtype_arma, typename dtype_numpy>
py::array_t<dtype_numpy> qd_python_copy2numpy(const arma::Mat<dtype_arma> &input, // Matrix
                                              const arma::uvec &indices = {})     // Optional columns to copy
{
    const ssize_t n_bytes = sizeof(dtype_numpy);
    const ssize_t n_rows = (ssize_t)input.n_rows;
    const ssize_t n_cols = indices.empty() ? (ssize_t)input.n_cols : (ssize_t)indices.n_elem;
    const ssize_t n_elements = n_rows * n_cols;

    std::vector<ssize_t> shape = {n_rows, n_cols}, strides = {n_bytes, n_rows * n_bytes};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();
    const dtype_arma *p_in = input.memptr();

    if (indices.empty())
        for (ssize_t i_el = 0; i_el < n_elements; ++i_el)
            p_out[i_el] = (dtype_numpy)p_in[i_el];
    else
        for (ssize_t i_col = 0; i_col < n_cols; ++i_col)
        {
            if (indices[i_col] >= input.n_cols)
                throw std::out_of_range("Index out of bound.");
            for (ssize_t i_row = 0; i_row < n_rows; ++i_row)
                p_out[i_col * n_rows + i_row] = (dtype_numpy)p_in[indices[i_col] * n_rows + i_row];
        }

    return output;
}

template <typename dtype>
py::array_t<std::complex<dtype>> qd_python_copy2numpy(const arma::Mat<dtype> &real,   // Matrix (real part)
                                                      const arma::Mat<dtype> &imag,   // Matrix (imaginary part)
                                                      const arma::uvec &indices = {}) // Optional columns to copy
{
    if (imag.n_rows != real.n_rows || imag.n_cols != real.n_cols)
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    const ssize_t n_bytes = sizeof(std::complex<dtype>);
    const ssize_t n_rows = (ssize_t)real.n_rows;
    const ssize_t n_cols = indices.empty() ? (ssize_t)real.n_cols : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape = {n_rows, n_cols}, strides = {n_bytes, n_rows * n_bytes};
    py::array_t<std::complex<dtype>> output(shape, strides);
    std::complex<dtype> *p_out = output.mutable_data();

    if (indices.empty())
    {
        const dtype *p_real = real.memptr(), *p_imag = imag.memptr();
        for (ssize_t i = 0; i < n_rows * n_cols; ++i)
            p_out[i] = {p_real[i], p_imag[i]};
    }
    else

        for (ssize_t i_col = 0; i_col < n_cols; ++i_col)
        {
            if (indices[i_col] >= real.n_cols)
                throw std::out_of_range("Index out of bound.");

            std::complex<dtype> *p_col = &p_out[i_col * n_rows];
            const dtype *p_real = real.colptr(indices[i_col]), *p_imag = imag.colptr(indices[i_col]);

            for (ssize_t i_row = 0; i_row < n_rows; ++i_row)
                p_col[i_row] = {p_real[i_row], p_imag[i_row]};
        }

    return output;
}

template <typename dtype>
py::array_t<dtype> qd_python_copy2numpy(const arma::Cube<dtype> &input, // Cube
                                        const arma::uvec &indices = {}) // Optional slices to copy
{
    const ssize_t n_bytes = sizeof(dtype);
    const ssize_t n_rows = (ssize_t)input.n_rows;
    const ssize_t n_cols = (ssize_t)input.n_cols;
    const ssize_t n_slices = indices.empty() ? (ssize_t)input.n_slices : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape = {n_rows, n_cols, n_slices}, strides = {n_bytes, n_rows * n_bytes, n_rows * n_cols * n_bytes};
    py::array_t<dtype> output(shape, strides);
    dtype *p_out = output.mutable_data();

    if (indices.empty())
        std::memcpy(p_out, input.memptr(), n_rows * n_cols * n_slices * n_bytes);
    else
        for (ssize_t i = 0; i < n_slices; ++i)
        {
            if (indices[i] >= input.n_slices)
                throw std::out_of_range("Index out of bound.");
            std::memcpy(&p_out[i * n_rows * n_cols], input.slice_memptr(indices[i]), n_rows * n_cols * n_bytes);
        }

    return output;
}

template <typename dtype>
py::array_t<std::complex<dtype>> qd_python_copy2numpy(const arma::Cube<dtype> &real,  // Cube (real part)
                                                      const arma::Cube<dtype> &imag,  // Cube (imaginary part)
                                                      const arma::uvec &indices = {}) // Optional slices to copy
{
    if (imag.n_rows != real.n_rows || imag.n_cols != real.n_cols || imag.n_slices != real.n_slices)
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    const ssize_t n_bytes = sizeof(std::complex<dtype>);
    const ssize_t n_rows = (ssize_t)real.n_rows;
    const ssize_t n_cols = (ssize_t)real.n_cols;
    const ssize_t n_slices = indices.empty() ? (ssize_t)real.n_slices : (ssize_t)indices.n_elem;

    std::vector<ssize_t> shape = {n_rows, n_cols, n_slices}, strides = {n_bytes, n_rows * n_bytes, n_rows * n_cols * n_bytes};
    py::array_t<std::complex<dtype>> output(shape, strides);
    std::complex<dtype> *p_out = output.mutable_data();

    if (indices.empty())
    {
        const dtype *p_real = real.memptr(), *p_imag = imag.memptr();
        for (ssize_t i = 0; i < n_rows * n_cols * n_slices; ++i)
            p_out[i] = {p_real[i], p_imag[i]};
    }
    else
        for (ssize_t i_slice = 0; i_slice < n_slices; ++i_slice)
        {
            if (indices[i_slice] >= real.n_slices)
                throw std::out_of_range("Index out of bound.");

            std::complex<dtype> *p_slice = &p_out[i_slice * n_rows * n_cols];
            const dtype *p_real = real.slice_memptr(indices[i_slice]), *p_imag = imag.slice_memptr(indices[i_slice]);

            for (ssize_t i_mat = 0; i_mat < n_rows * n_cols; ++i_mat)
                p_slice[i_mat] = {p_real[i_mat], p_imag[i_mat]};
        }

    return output;
}

template <typename dtype>
py::list qd_python_copy2numpy(const std::vector<dtype> &input, // Vector of Stuff
                              const arma::uvec &indices = {})  // Optional elements to copy
{
    py::list output;
    if (indices.empty())
        for (const auto &element : input)
            output.append(qd_python_copy2numpy(element));
    else
        for (auto ind : indices)
        {
            if (ind >= input.size())
                throw std::out_of_range("Index out of bound.");
            output.append(qd_python_copy2numpy(input.at(ind)));
        }
    return output;
}

template <typename dtype>
py::list qd_python_copy2numpy(const std::vector<dtype> &real, // Vector of Real Parts
                              const std::vector<dtype> &imag, // Vector of Imaginary Parts
                              const arma::uvec &indices = {}) // Optional elements to copy
{
    if (real.size() != imag.size())
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    py::list output;
    if (indices.empty())
        for (size_t ind = 0; ind < real.size(); ++ind)
            output.append(qd_python_copy2numpy(real.at(ind), imag.at(ind)));
    else
        for (auto ind : indices)
        {
            if (ind >= real.size())
                throw std::out_of_range("Index out of bound.");
            output.append(qd_python_copy2numpy(real.at(ind), imag.at(ind)));
        }
    return output;
}

py::list qd_python_copy2python(const std::vector<std::string> &input, // Vector of strings
                               const arma::uvec &indices = {})        // Optional elements to copy
{
    py::list output;
    if (indices.empty())
        for (const auto &element : input)
            output.append(element);
    else
        for (auto ind : indices)
        {
            if (ind >= input.size())
                throw std::out_of_range("Index out of bound.");
            output.append(input.at(ind));
        }
    return output;
}

// -------------------------------- qd_python_init_output --------------------------------

template <typename dtype>
py::array_t<dtype> qd_python_init_output(arma::uword n_elem,
                                         arma::Col<dtype> *wrapper = nullptr)
{
    const ssize_t n_bytes = (ssize_t)sizeof(dtype);

    std::array<ssize_t, 1> shape = {(ssize_t)n_elem};
    std::array<ssize_t, 1> strides = {n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
        *wrapper = arma::Col<dtype>((dtype *)output.data(), n_elem, false, true);

    return output;
}

template <typename dtype>
py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols,
                                         arma::Mat<dtype> *wrapper = nullptr)
{
    const ssize_t n_bytes = (ssize_t)sizeof(dtype);

    std::array<ssize_t, 2> shape = {(ssize_t)n_rows, (ssize_t)n_cols};
    std::array<ssize_t, 2> strides = {n_bytes, (ssize_t)n_rows * n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
        *wrapper = arma::Mat<dtype>((dtype *)output.data(), n_rows, n_cols, false, true);

    return output;
}

template <typename dtype>
py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols, arma::uword n_slices,
                                         arma::Cube<dtype> *wrapper = nullptr)
{
    const ssize_t n_bytes = (ssize_t)sizeof(dtype);

    std::array<ssize_t, 3> shape = {(ssize_t)n_rows, (ssize_t)n_cols, (ssize_t)n_slices};
    std::array<ssize_t, 3> strides = {n_bytes, (ssize_t)n_rows * n_bytes, ssize_t(n_rows * n_cols) * n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
        *wrapper = arma::Cube<dtype>((dtype *)output.data(), n_rows, n_cols, n_slices, false, true);

    return output;
}

template <typename dtype>
py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols, arma::uword n_slices, arma::uword n_frames)
{
    const ssize_t n_bytes = (ssize_t)sizeof(dtype);

    std::array<ssize_t, 4> shape = {(ssize_t)n_rows, (ssize_t)n_cols, (ssize_t)n_slices, (ssize_t)n_frames};
    std::array<ssize_t, 4> strides = {n_bytes, (ssize_t)n_rows * n_bytes, ssize_t(n_rows * n_cols) * n_bytes, ssize_t(n_rows * n_cols * n_slices) * n_bytes};
    py::array_t<dtype> output(shape, strides);

    return output;
}

// -------------------------------- qd_python_get_shape --------------------------------

// 0-2: n_row, n_cols, n_slices
// 3-5: Strides in number of elements (not bytes!)
//   6: Fortran-contiguous (1) or C-contiguous (0)
//   7: Total number of elements
//   8: Total number of bytes
template <typename dtype>
static inline std::array<size_t, 9> qd_python_get_shape(const py::array_t<dtype> &input, bool shape_only = false)
{
    auto buf = input.request();
    int nd = (int)buf.ndim;
    const size_t n_bytes = sizeof(dtype);

    if (nd > 3)
        throw std::invalid_argument("Expected 1D, 2D or 3D array, got " + std::to_string(nd) + "D");

    if (nd == 0)
        return {1, 1, 1, 1, 1, 1, 1, 1, n_bytes};

    std::array<size_t, 9> shape;
    shape[0] = (size_t)buf.shape[0];
    shape[1] = (nd >= 2) ? (size_t)buf.shape[1] : 1;
    shape[2] = (nd == 3) ? (size_t)buf.shape[2] : 1;
    nd = (shape[1] == 1 && shape[2] == 1) ? 1 : (shape[2] == 1 ? 2 : 3);

    if (shape_only)
        return shape;

    if (buf.itemsize != (ssize_t)n_bytes)
        throw std::invalid_argument("Dtype size mismatch");

    shape[3] = (size_t)buf.strides[0] / n_bytes;
    shape[4] = (nd >= 2) ? (size_t)buf.strides[1] / n_bytes : shape[0];
    shape[5] = (nd == 3) ? (size_t)buf.strides[2] / n_bytes : shape[0] * shape[1];

    // Check if input is Fortran-contiguous
    shape[6] = (shape[3] == 1) && (shape[4] == shape[0]) && (shape[5] == shape[0] * shape[1]);

    shape[7] = shape[0] * shape[1] * shape[2];           // Total number of elements
    shape[8] = shape[0] * shape[1] * shape[2] * n_bytes; // Total number of bytes

    return shape;
}

// 0-2: n_row, n_cols, n_slices
// 3-5: Strides in number of elements (not bytes!)
//   6: Fortran-contiguous (1) or C-contiguous (0)
//   7: Total number of elements
//   8: Total number of bytes
template <typename dtype>
std::vector<std::array<size_t, 9>> qd_python_get_list_shape(const py::list &input,
                                                            std::vector<dtype *> &buffer_pointers,
                                                            std::vector<py::array_t<dtype>> &owned_arrays)
{
    size_t n_input = input.size();

    std::vector<std::array<size_t, 9>> output;
    buffer_pointers.clear();
    owned_arrays.clear();

    if (n_input == 0)
        return output;

    output.resize(n_input);
    buffer_pointers.resize(n_input);
    owned_arrays.reserve(n_input);

    for (size_t i = 0; i < n_input; ++i)
    {
        py::array array = py::cast<py::array>(input[i]);

        py::array_t<dtype> typed_array;
        if (!py::detail::npy_format_descriptor<dtype>::dtype().is(array.dtype()))
            typed_array = py::array_t<dtype>(array); // implicit cast creates new array
        else
            typed_array = array.cast<py::array_t<dtype>>();

        output[i] = qd_python_get_shape(typed_array);
        buffer_pointers[i] = const_cast<dtype *>(typed_array.data());
        owned_arrays.push_back(std::move(typed_array)); // store to keep alive
    }
    return output;
}

// -------------------------------- qd_python_copy2arma --------------------------------

template <typename dtype>
static inline void qd_python_copy2arma(const dtype *src, const std::array<size_t, 9> &shape,
                                       arma::Cube<dtype> &output)
{
    const size_t nr = shape[0];
    const size_t nc = shape[1];
    const size_t ns = shape[2];

    // Allocate the Armadillo cube
    if (output.n_rows != nr || output.n_cols != nc || output.n_slices != ns)
        output.set_size(nr, nc, ns);

    // Copy data
    dtype *dst = output.memptr();

    if (shape[6] == 1)
        std::memcpy(dst, src, shape[8]);
    else
        for (size_t is = 0; is < ns; ++is)
            for (size_t ic = 0; ic < nc; ++ic)
            {
                size_t dst_offset = is * (nr * nc) + ic * nr;
                size_t src_offset = is * shape[5] + ic * shape[4];
                for (size_t ir = 0; ir < nr; ++ir)
                    dst[dst_offset + ir] = src[src_offset + ir * shape[3]];
            }
}

template <typename dtype>
void qd_python_copy2arma(const py::array_t<dtype> &input, arma::Cube<dtype> &output)
{
    auto shape = qd_python_get_shape(input);
    const dtype *src = input.data();
    qd_python_copy2arma(src, shape, output);
}

template <typename dtype>
static inline void qd_python_copy2arma(const std::complex<dtype> *src, const std::array<size_t, 9> &shape,
                                       arma::Cube<dtype> &real, arma::Cube<dtype> &imag)
{
    const size_t nr = shape[0];
    const size_t nc = shape[1];
    const size_t ns = shape[2];

    // Allocate the Armadillo cube
    if (real.n_rows != nr || real.n_cols != nc || real.n_slices != ns)
        real.set_size(nr, nc, ns);
    if (imag.n_rows != nr || imag.n_cols != nc || imag.n_slices != ns)
        imag.set_size(nr, nc, ns);

    // Copy data
    dtype *dst_re = real.memptr();
    dtype *dst_im = imag.memptr();

    for (size_t is = 0; is < ns; ++is)
        for (size_t ic = 0; ic < nc; ++ic)
        {
            size_t dst_offset = is * (nr * nc) + ic * nr;
            size_t src_offset = is * shape[5] + ic * shape[4];
            for (size_t ir = 0; ir < nr; ++ir)
            {
                auto src_data = src[src_offset + ir * shape[3]];
                dst_re[dst_offset + ir] = src_data.real();
                dst_im[dst_offset + ir] = src_data.imag();
            }
        }
}

template <typename dtype>
void qd_python_copy2arma(const py::array_t<std::complex<dtype>> &input, arma::Cube<dtype> &real, arma::Cube<dtype> &imag)
{
    auto shape = qd_python_get_shape(input);
    const std::complex<dtype> *src = input.data();
    qd_python_copy2arma(src, shape, real, imag);
}

// -------------------------------- qd_python_numpy2arma --------------------------------

template <typename dtype>
arma::Col<dtype> qd_python_numpy2arma_Col(const py::array_t<dtype> &input,
                                          bool view = false, bool strict = false,
                                          std::string var_name = "", arma::uword n_elem = 0)
{
    auto shape = qd_python_get_shape(input, !view);
    if (n_elem != 0 && shape[0] != n_elem)
    {
        if (var_name.empty())
            throw std::invalid_argument("Incorrect number of elements.");
        else
            throw std::invalid_argument("Input '" + var_name + "' has incorrect number of elements.");
    }
    if (view && shape[6])
        return arma::Col<dtype>(const_cast<dtype *>(input.data()), shape[0] * shape[1] * shape[2], true, false);
    if (view && strict)
        throw std::invalid_argument("Could not obtain memory view, possibly due to mismatching strides.");

    arma::Col<dtype> output(shape[0] * shape[1] * shape[2], arma::fill::none);
    arma::Cube<dtype> wrapper(output.memptr(), shape[0], shape[1], shape[2], false, true);
    qd_python_copy2arma(input, wrapper);
    return output;
}

template <typename dtype>
arma::Mat<dtype> qd_python_numpy2arma_Mat(const py::array_t<dtype> &input, bool view = false, bool strict = false)
{
    auto shape = qd_python_get_shape(input, !view);
    if (view && shape[6])
        return arma::Mat<dtype>(const_cast<dtype *>(input.data()), shape[0], shape[1] * shape[2], true, false);
    if (view && strict)
        throw std::invalid_argument("Could not obtain memory view, possibly due to mismatching strides.");

    arma::Mat<dtype> output(shape[0], shape[1] * shape[2], arma::fill::none);
    arma::Cube<dtype> wrapper(output.memptr(), shape[0], shape[1], shape[2], false, true);
    qd_python_copy2arma(input, wrapper);
    return output;
}

template <typename dtype>
arma::Cube<dtype> qd_python_numpy2arma_Cube(const py::array_t<dtype> &input, bool view = false, bool strict = false)
{
    auto shape = qd_python_get_shape(input, !view);
    if (view && shape[6])
        return arma::Cube<dtype>(const_cast<dtype *>(input.data()), shape[0], shape[1], shape[2], true, false);
    if (view && strict)
        throw std::invalid_argument("Could not obtain memory view, possibly due to mismatching strides.");

    arma::Cube<dtype> output(shape[0], shape[1], shape[2], arma::fill::none);
    qd_python_copy2arma(input, output);
    return output;
}

template <typename dtype>
arma::Col<dtype> qd_python_numpy2arma_Col(const py::handle &obj, bool view = false, bool strict = false)
{
    py::array_t<dtype> pyarray;
    if (py::isinstance<py::array_t<dtype>>(obj))
        pyarray = py::reinterpret_borrow<py::array_t<dtype>>(obj);
    else
    {
        if (view && strict)
            throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
        pyarray = py::cast<py::array_t<dtype>>(obj);
        view = false;
    }
    return qd_python_numpy2arma_Col(pyarray, view, strict);
}

template <typename dtype>
arma::Mat<dtype> qd_python_numpy2arma_Mat(const py::handle &obj, bool view = false, bool strict = false)
{
    py::array_t<dtype> pyarray;
    if (py::isinstance<py::array_t<dtype>>(obj))
        pyarray = py::reinterpret_borrow<py::array_t<dtype>>(obj);
    else
    {
        if (view && strict)
            throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
        pyarray = py::cast<py::array_t<dtype>>(obj);
        view = false;
    }
    return qd_python_numpy2arma_Mat(pyarray, view, strict);
}

template <typename dtype>
arma::Cube<dtype> qd_python_numpy2arma_Cube(const py::handle &obj, bool view = false, bool strict = false)
{
    py::array_t<dtype> pyarray;
    if (py::isinstance<py::array_t<dtype>>(obj))
        pyarray = py::reinterpret_borrow<py::array_t<dtype>>(obj);
    else
    {
        if (view && strict)
            throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
        pyarray = py::cast<py::array_t<dtype>>(obj);
        view = false;
    }
    return qd_python_numpy2arma_Cube(pyarray, view, strict);
}

// -------------------------------- qd_python_list2vector --------------------------------

template <typename dtype>
std::vector<arma::Col<dtype>> qd_python_list2vector_Col(const py::list &input)
{
    size_t n_input = input.size();
    std::vector<arma::Col<dtype>> output(n_input);

    std::vector<dtype *> pointers;
    std::vector<py::array_t<dtype>> arrays;
    auto shape = qd_python_get_list_shape(input, pointers, arrays);

    for (size_t i = 0; i < n_input; ++i)
    {
        output[i].set_size(shape[i][7]);
        arma::Cube<dtype> wrapper(output[i].memptr(), shape[i][0], shape[i][1], shape[i][2], false, true);
        qd_python_copy2arma(pointers[i], shape[i], wrapper);
    }
    return output;
}

template <typename dtype>
std::vector<arma::Mat<dtype>> qd_python_list2vector_Mat(const py::list &input)
{
    size_t n_input = input.size();
    std::vector<arma::Mat<dtype>> output(n_input);

    std::vector<dtype *> pointers;
    std::vector<py::array_t<dtype>> arrays;
    auto shape = qd_python_get_list_shape(input, pointers, arrays);

    for (size_t i = 0; i < n_input; ++i)
    {
        output[i].set_size(shape[i][0], shape[i][1] * shape[i][2]);
        arma::Cube<dtype> wrapper(output[i].memptr(), shape[i][0], shape[i][1], shape[i][2], false, true);
        qd_python_copy2arma(pointers[i], shape[i], wrapper);
    }
    return output;
}

template <typename dtype>
std::vector<arma::Cube<dtype>> qd_python_list2vector_Cube(const py::list &input)
{
    size_t n_input = input.size();
    std::vector<arma::Cube<dtype>> output(n_input);

    std::vector<dtype *> pointers;
    std::vector<py::array_t<dtype>> arrays;
    auto shape = qd_python_get_list_shape(input, pointers, arrays);

    for (size_t i = 0; i < n_input; ++i)
        qd_python_copy2arma(pointers[i], shape[i], output[i]);

    return output;
}

template <typename dtype>
void qd_python_list2vector_Cube_Cplx(const py::list &input,
                                     std::vector<arma::Cube<dtype>> &real,
                                     std::vector<arma::Cube<dtype>> &imag)
{
    size_t n_input = input.size();

    std::vector<std::complex<dtype> *> pointers;
    std::vector<py::array_t<std::complex<dtype>>> arrays;
    auto shape = qd_python_get_list_shape(input, pointers, arrays);

    real.clear();
    real.resize(n_input);
    imag.clear();
    imag.resize(n_input);

    for (size_t i = 0; i < n_input; ++i)
        qd_python_copy2arma(pointers[i], shape[i], real[i], imag[i]);
}

std::vector<std::string> qd_python_list2vector_Strings(const py::list &input)
{
    std::vector<std::string> output;
    output.reserve(py::len(input));

    for (py::handle obj : input)
    {
        output.emplace_back(py::cast<std::string>(obj));
    }

    return output;
}

// -------------------------------- Imterleave complex --------------------------------

template <typename dtype>
arma::Mat<dtype> qd_python_Complex2Interleaved(const arma::Mat<std::complex<dtype>> &input)
{
    auto output = arma::Mat<dtype>(2 * input.n_rows, input.n_cols, arma::fill::none);
    dtype *p_out = output.memptr();
    const std::complex<dtype> *p_in = input.memptr();

    for (arma::uword i = 0; i < input.n_elem; ++i)
    {
        p_out[2 * i] = p_in[i].real();
        p_out[2 * i + 1] = p_in[i].imag();
    }
    return output;
}

template <typename dtype>
std::vector<arma::Mat<dtype>> qd_python_Complex2Interleaved(const std::vector<arma::Mat<std::complex<dtype>>> &input)
{
    size_t n_input = input.size();
    std::vector<arma::Mat<dtype>> output(n_input);
    for (size_t i = 0; i < n_input; ++i)
        output[i] = qd_python_Complex2Interleaved(input[i]);
    return output;
}

template <typename dtype>
arma::Mat<std::complex<dtype>> qd_python_Interleaved2Complex(const arma::Mat<dtype> &input)
{
    if (input.n_rows % 2 != 0)
        throw std::invalid_argument("Input must have an even number of rows.");

    auto output = arma::Mat<std::complex<dtype>>(input.n_rows / 2, input.n_cols, arma::fill::none);
    std::complex<dtype> *p_out = output.memptr();
    const dtype *p_in = input.memptr();

    for (arma::uword i = 0; i < output.n_elem; ++i)
        p_out[i] = {p_in[2 * i], p_in[2 * i + 1]};
    return output;
}

template <typename dtype>
std::vector<arma::Mat<std::complex<dtype>>> qd_python_Interleaved2Complex(const std::vector<arma::Mat<dtype>> &input)
{
    size_t n_input = input.size();
    std::vector<arma::Mat<std::complex<dtype>>> output(n_input);
    for (size_t i = 0; i < n_input; ++i)
        output[i] = qd_python_Interleaved2Complex(input[i]);
    return output;
}

// Convert to std::any
inline std::any qd_python_anycast(py::handle obj, std::string var_name = "")
{

    if (py::isinstance<py::str>(obj))
        return py::cast<std::string>(obj);

    if (py::isinstance<py::float_>(obj))
        return py::cast<double>(obj);

    if (py::isinstance<py::int_>(obj))
        return py::cast<long long>(obj);

    if (py::isinstance<py::array>(obj))
    {
        py::array arr = py::cast<py::array>(obj);
        py::buffer_info buf = arr.request();

        if (arr.dtype().is(py::dtype::of<double>()))
        {
            auto arr_t = py::array_t<double>(py::reinterpret_borrow<py::array_t<double>>(arr));
            if (buf.ndim == 1)
                return arma::Row<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<double>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<double>(arr_t);
        }

        if (arr.dtype().is(py::dtype::of<float>()))
        {
            auto arr_t = py::array_t<float>(py::reinterpret_borrow<py::array_t<float>>(arr));
            if (buf.ndim == 1)
                return arma::Row<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<float>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<float>(arr_t);
        }

        if (arr.dtype().is(py::dtype::of<unsigned>()))
        {
            auto arr_t = py::array_t<unsigned>(py::reinterpret_borrow<py::array_t<unsigned>>(arr));
            if (buf.ndim == 1)
                return arma::Row<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<unsigned>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<unsigned>(arr_t);
        }

        if (arr.dtype().is(py::dtype::of<int>()))
        {
            auto arr_t = py::array_t<int>(py::reinterpret_borrow<py::array_t<int>>(arr));
            if (buf.ndim == 1)
                return arma::Row<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<int>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<int>(arr_t);
        }

        if (arr.dtype().is(py::dtype::of<unsigned long long>()))
        {
            auto arr_t = py::array_t<unsigned long long>(py::reinterpret_borrow<py::array_t<unsigned long long>>(arr));
            if (buf.ndim == 1)
                return arma::Row<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<unsigned long long>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<unsigned long long>(arr_t);
        }

        if (arr.dtype().is(py::dtype::of<long long>()) || arr.dtype().is(py::dtype::of<int64_t>()))
        {
            auto arr_t = py::array_t<long long>(py::reinterpret_borrow<py::array_t<long long>>(arr));
            if (buf.ndim == 1)
                return arma::Row<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return qd_python_numpy2arma_Mat<long long>(arr_t);
            else if (buf.ndim == 3)
                return qd_python_numpy2arma_Cube<long long>(arr_t);
        }
    }

    std::string error_message = "Input '" + var_name + "' has an unsupported type.";
    throw std::invalid_argument(error_message);

    return std::any();
}

#endif
