// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_python_arma_adapter_H
#define quadriga_python_arma_adapter_H

#include <any>
#include <armadillo>
#include <string>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>

namespace py = pybind11;

template <typename T>
struct qd_is_complex : std::false_type
{
};
template <typename T>
struct qd_is_complex<std::complex<T>> : std::true_type
{
};

// -------------------------------- qd_python_copy2numpy --------------------------------

// Copy an arma::Col into a Numpy array with dtype to dtype_numpy conversion and optional element selection.
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_copy2numpy(const arma::Col<dtype> *re,
                                                     const arma::Col<dtype> *im = nullptr,
                                                     const arma::uvec &i_elem = {},
                                                     bool transpose = false)
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr)
        throw std::invalid_argument("Input 're' must not be null.");

    if (im != nullptr && im->n_elem != re->n_elem)
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    const bool select = !i_elem.empty();

    // Complex output without an explicit imag part: re is interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);
    if (interleaved_re && !select && (re->n_elem % 2 != 0))
        throw std::invalid_argument("Interleaved-complex input requires an even element count.");

    const size_t n_elements = select ? i_elem.n_elem : (interleaved_re ? re->n_elem / 2 : re->n_elem);
    const size_t idx_limit = interleaved_re ? re->n_elem / 2 : re->n_elem;

    // Interleaved real case doubles the element count
    size_t out_elem = n_elements;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_elem = 2 * n_elements;

    const py::ssize_t n_bytes = sizeof(dtype_numpy);
    const py::ssize_t oe = (py::ssize_t)out_elem;
    std::vector<py::ssize_t> shape, strides;
    if (transpose)
        shape = {1, oe}, strides = {oe * n_bytes, n_bytes};
    else
        shape = {oe}, strides = {n_bytes};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    const dtype *p_re = re->memptr();
    const dtype *p_im = (im != nullptr) ? im->memptr() : nullptr;

    for (size_t k = 0; k < n_elements; ++k)
    {
        const size_t idx = select ? i_elem[k] : k;
        if (select && idx >= idx_limit)
            throw std::out_of_range("Index out of bound.");

        if constexpr (qd_is_complex<dtype_numpy>::value) // Case 3: complex output
        {
            using real_t = typename dtype_numpy::value_type;
            p_out[k] = (im != nullptr)
                           ? dtype_numpy((real_t)p_re[idx], (real_t)p_im[idx])              // separate re / im
                           : dtype_numpy((real_t)p_re[2 * idx], (real_t)p_re[2 * idx + 1]); // interleaved re
        }
        else if (im == nullptr)                // Case 1: real copy with cast
            p_out[k] = (dtype_numpy)p_re[idx]; // Case 2: interleaved re/im (out_elem == 2 * n_elements)
        else
        {
            p_out[2 * k] = (dtype_numpy)p_re[idx];
            p_out[2 * k + 1] = (dtype_numpy)p_im[idx];
        }
    }
    return output;
}

// Copy an arma::Mat into a Numpy array with dtype to dtype_numpy conversion and optional column selection
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_copy2numpy(const arma::Mat<dtype> *re,
                                                     const arma::Mat<dtype> *im = nullptr,
                                                     const arma::uvec &i_col = {})
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr)
        throw std::invalid_argument("Input 're' must not be null.");

    if (im != nullptr && (im->n_rows != re->n_rows || im->n_cols != re->n_cols))
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    const bool select = !i_col.empty();
    const size_t n_cols = select ? i_col.n_elem : re->n_cols;

    // Complex output without an explicit imag part: re's rows are interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);
    if (interleaved_re && (re->n_rows % 2 != 0))
        throw std::invalid_argument("Interleaved-complex input requires an even row count.");
    const size_t n_rows = interleaved_re ? re->n_rows / 2 : re->n_rows;

    // Interleaved real case doubles the row count
    size_t out_rows = n_rows;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_rows = 2 * n_rows;

    const size_t n_bytes = sizeof(dtype_numpy);
    std::array<py::ssize_t, 2> shape = {(py::ssize_t)out_rows, (py::ssize_t)n_cols};
    std::array<py::ssize_t, 2> strides = {(py::ssize_t)n_bytes, py::ssize_t(out_rows * n_bytes)};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    for (size_t c = 0; c < n_cols; ++c)
    {
        const size_t src_col = select ? i_col[c] : c;
        if (select && src_col >= re->n_cols)
            throw std::out_of_range("Index out of bound.");

        const dtype *p_re = re->colptr(src_col);
        const dtype *p_im = (im != nullptr) ? im->colptr(src_col) : nullptr;

        if constexpr (qd_is_complex<dtype_numpy>::value) // Case 3: complex output (imag = 0 if im == nullptr)
        {
            using real_t = typename dtype_numpy::value_type;
            dtype_numpy *dc = p_out + c * out_rows;
            if (im == nullptr) // im interleaved within re
                for (size_t i = 0; i < n_rows; ++i)
                    dc[i] = dtype_numpy((real_t)p_re[2 * i], (real_t)p_re[2 * i + 1]);
            else
                for (size_t i = 0; i < n_rows; ++i)
                    dc[i] = dtype_numpy((real_t)p_re[i], (real_t)p_im[i]);
        }
        else
        {
            dtype_numpy *dc = p_out + c * out_rows;
            if (im == nullptr) // Case 1: real copy with cast
                for (size_t i = 0; i < n_rows; ++i)
                    dc[i] = (dtype_numpy)p_re[i];
            else // Case 2: interleaved re/im along rows (out_rows == 2 * n_rows)
                for (size_t i = 0; i < n_rows; ++i)
                {
                    dc[2 * i] = (dtype_numpy)p_re[i];
                    dc[2 * i + 1] = (dtype_numpy)p_im[i];
                }
        }
    }
    return output;
}

// Copy an arma::Cube into a Numpy array with dtype to dtype_numpy conversion and optional slice selection
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_copy2numpy(const arma::Cube<dtype> *re,
                                                     const arma::Cube<dtype> *im = nullptr,
                                                     const arma::uvec &i_slice = {})
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr)
        throw std::invalid_argument("Input 're' must not be null.");

    if (im != nullptr && (im->n_rows != re->n_rows || im->n_cols != re->n_cols || im->n_slices != re->n_slices))
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    const bool select = !i_slice.empty();
    const size_t n_cols = re->n_cols;
    const size_t n_slices = select ? i_slice.n_elem : re->n_slices;

    // Complex output without an explicit imag part: re's rows are interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);
    if (interleaved_re && (re->n_rows % 2 != 0))
        throw std::invalid_argument("Interleaved-complex input requires an even row count.");
    const size_t n_rows = interleaved_re ? re->n_rows / 2 : re->n_rows;
    const size_t slice_size = n_rows * n_cols;

    // Interleaved real case doubles the row count
    size_t out_rows = n_rows;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_rows = 2 * n_rows;

    const size_t n_bytes = sizeof(dtype_numpy);
    const size_t out_slice = out_rows * n_cols;
    std::array<py::ssize_t, 3> shape = {(py::ssize_t)out_rows, (py::ssize_t)n_cols, (py::ssize_t)n_slices};
    std::array<py::ssize_t, 3> strides = {(py::ssize_t)n_bytes, py::ssize_t(out_rows * n_bytes), py::ssize_t(out_slice * n_bytes)};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    for (size_t s = 0; s < n_slices; ++s)
    {
        const size_t src_slice = select ? i_slice[s] : s;
        if (select && src_slice >= re->n_slices)
            throw std::out_of_range("Index out of bound.");

        const dtype *p_re = re->slice_memptr(src_slice);
        const dtype *p_im = (im != nullptr) ? im->slice_memptr(src_slice) : nullptr;

        if constexpr (qd_is_complex<dtype_numpy>::value) // Case 3: complex output (imag = 0 if im == nullptr)
        {
            using real_t = typename dtype_numpy::value_type;
            dtype_numpy *ds = p_out + s * out_slice;
            if (im == nullptr) // im interleaved within re's rows
                for (size_t j = 0; j < n_cols; ++j)
                {
                    dtype_numpy *dc = ds + j * n_rows;
                    const dtype *sc = p_re + j * re->n_rows;
                    for (size_t i = 0; i < n_rows; ++i)
                        dc[i] = dtype_numpy((real_t)sc[2 * i], (real_t)sc[2 * i + 1]);
                }
            else
                for (size_t e = 0; e < slice_size; ++e)
                    ds[e] = dtype_numpy((real_t)p_re[e], (real_t)p_im[e]);
        }
        else
        {
            dtype_numpy *ds = p_out + s * out_slice;
            if (im == nullptr) // Case 1: real copy with cast
                for (size_t e = 0; e < slice_size; ++e)
                    ds[e] = (dtype_numpy)p_re[e];
            else // Case 2: interleaved re/im along rows (out_rows == 2 * n_rows)
                for (size_t j = 0; j < n_cols; ++j)
                {
                    dtype_numpy *dc = ds + j * out_rows;
                    const dtype *sr = p_re + j * n_rows;
                    const dtype *si = p_im + j * n_rows;
                    for (size_t i = 0; i < n_rows; ++i)
                    {
                        dc[2 * i] = (dtype_numpy)sr[i];
                        dc[2 * i + 1] = (dtype_numpy)si[i];
                    }
                }
        }
    }
    return output;
}

// Copy a std::vector of arma::Col/Mat/Cube into a Python list, optional element selection via i_vec.
// Each element is forwarded to qd_python_copy2numpy; dtype_numpy picks real / interleaved / complex output.
template <typename dtype_arma, typename dtype_numpy = typename dtype_arma::elem_type>
static py::list qd_python_copy2list(const std::vector<dtype_arma> *re,
                                    const std::vector<dtype_arma> *im = nullptr,
                                    const arma::uvec &i_vec = {})
{
    if (re == nullptr)
        throw std::invalid_argument("Input 're' must not be null.");

    if (im != nullptr && im->size() != re->size())
        throw std::invalid_argument("Sizes of real and imaginary parts dont match.");

    using dtype = typename dtype_arma::elem_type;

    auto convert = [&](size_t ind)
    {
        return (im != nullptr)
                   ? qd_python_copy2numpy<dtype, dtype_numpy>(&re->at(ind), &im->at(ind))
                   : qd_python_copy2numpy<dtype, dtype_numpy>(&re->at(ind));
    };

    py::list output;
    if (i_vec.empty())
        for (size_t ind = 0; ind < re->size(); ++ind)
            output.append(convert(ind));
    else
        for (auto ind : i_vec)
        {
            if (ind >= re->size())
                throw std::out_of_range("Index out of bound.");
            output.append(convert(ind));
        }
    return output;
}

static py::list qd_python_copy2list(const std::vector<std::string> &input, // Vector of strings
                                    const arma::uvec &i_vec = {})          // Optional elements to copy
{
    py::list output;
    if (i_vec.empty())
        for (const auto &element : input)
            output.append(element);
    else
        for (auto ind : i_vec)
        {
            if (ind >= input.size())
                throw std::out_of_range("Index out of bound.");
            output.append(input.at(ind));
        }
    return output;
}

// --- Stack std::vector types to additional dimension ---

// Stack vector(s) of Cols into a Numpy array (frames = vector index, ragged + zero-padded),
// with dtype -> dtype_numpy conversion and optional frame selection via i_vec.
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_stack2numpy(const std::vector<arma::Col<dtype>> *re,
                                                      const std::vector<arma::Col<dtype>> *im = nullptr,
                                                      const arma::uvec &i_vec = {})
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr || re->empty())
        return py::array_t<dtype_numpy>();

    const bool select = !i_vec.empty();
    const size_t n_frames = select ? i_vec.n_elem : re->size();

    // Validate the frame selector up front
    if (select)
        for (size_t f = 0; f < n_frames; ++f)
            if (i_vec[f] >= re->size())
                throw std::out_of_range("Index out of bound.");

    // Complex output without an explicit imag part: each re frame is interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);

    // Ragged scan over the selected frames (re and, if present, im) -> rows = global max length
    bool ragged = false;
    size_t n_rows = (*re)[select ? i_vec[0] : 0].n_elem;
    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? i_vec[f] : f;
        const size_t r = (*re)[src_f].n_elem;
        if (interleaved_re && (r % 2 != 0))
            throw std::invalid_argument("Interleaved-complex input requires an even element count.");
        if (r != n_rows)
        {
            ragged = true;
            n_rows = std::max(n_rows, r);
        }
        if (im != nullptr)
        {
            if (src_f < im->size())
            {
                const size_t ri = (*im)[src_f].n_elem;
                if (ri != n_rows)
                {
                    ragged = true;
                    n_rows = std::max(n_rows, ri);
                }
            }
            else
                ragged = true; // selected frame has no imaginary counterpart -> gap
        }
    }

    if (interleaved_re)
        n_rows /= 2; // re frames hold interleaved pairs -> half as many complex rows

    // Interleaved real case doubles the row count; complex/real cases keep n_rows
    size_t out_rows = n_rows;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_rows = 2 * n_rows;

    const size_t n_bytes = sizeof(dtype_numpy);
    const size_t n_total = out_rows * n_frames;
    std::array<py::ssize_t, 2> shape = {(py::ssize_t)out_rows, (py::ssize_t)n_frames};
    std::array<py::ssize_t, 2> strides = {(py::ssize_t)n_bytes, py::ssize_t(out_rows * n_bytes)};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    if (ragged)
        std::fill_n(p_out, n_total, dtype_numpy{});

    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? (size_t)i_vec[f] : f;
        const arma::Col<dtype> &cr = (*re)[src_f];
        const size_t r_re = cr.n_elem;
        const dtype *p_re = cr.memptr();

        const bool has_im = (im != nullptr) && (src_f < im->size());
        const size_t r_im = has_im ? (*im)[src_f].n_elem : 0;
        const dtype *p_im = has_im ? (*im)[src_f].memptr() : nullptr;

        if constexpr (!qd_is_complex<dtype_numpy>::value)
        {
            dtype_numpy *dst = p_out + f * out_rows;
            if (im == nullptr) // Case 1: real output with cast
                for (size_t i = 0; i < r_re; ++i)
                    dst[i] = (dtype_numpy)p_re[i];
            else // Case 2: interleaved re/im along rows (out_rows == 2 * n_rows)
            {
                for (size_t i = 0; i < r_re; ++i)
                    dst[2 * i] = (dtype_numpy)p_re[i];
                for (size_t i = 0; i < r_im; ++i)
                    dst[2 * i + 1] = (dtype_numpy)p_im[i];
            }
        }
        else // Case 3: complex output (imag = 0 where im is absent or short)
        {
            using real_t = typename dtype_numpy::value_type;
            dtype_numpy *dst = p_out + f * out_rows;
            if (has_im)
            {
                const size_t r = std::max(r_re, r_im);
                for (size_t i = 0; i < r; ++i)
                    dst[i] = dtype_numpy(i < r_re ? (real_t)p_re[i] : real_t(0),
                                         i < r_im ? (real_t)p_im[i] : real_t(0));
            }
            else // im interleaved within re
                for (size_t i = 0; i < r_re / 2; ++i)
                    dst[i] = dtype_numpy((real_t)p_re[2 * i], (real_t)p_re[2 * i + 1]);
        }
    }
    return output;
}

// Stack vector(s) of Mats into a Numpy array (frames = vector index, ragged + zero-padded),
// with dtype -> dtype_numpy conversion and optional frame selection via i_vec.
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_stack2numpy(const std::vector<arma::Mat<dtype>> *re,
                                                      const std::vector<arma::Mat<dtype>> *im = nullptr,
                                                      const arma::uvec &i_vec = {})
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr || re->empty())
        return py::array_t<dtype_numpy>();

    const bool select = !i_vec.empty();
    const size_t n_frames = select ? (size_t)i_vec.n_elem : re->size();

    // Validate the frame selector up front
    if (select)
        for (size_t f = 0; f < n_frames; ++f)
            if (i_vec[f] >= re->size())
                throw std::out_of_range("Index out of bound.");

    // Complex output without an explicit imag part: each re frame's rows are interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);

    // Ragged scan over the selected frames (re and, if present, im)
    bool ragged = false;
    size_t n_rows = (*re)[select ? (size_t)i_vec[0] : 0].n_rows;
    size_t n_cols = (*re)[select ? (size_t)i_vec[0] : 0].n_cols;
    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? (size_t)i_vec[f] : f;
        const arma::Mat<dtype> &c = (*re)[src_f];
        if (interleaved_re && (c.n_rows % 2 != 0))
            throw std::invalid_argument("Interleaved-complex input requires an even row count.");

        if (c.n_rows != n_rows || c.n_cols != n_cols)
        {
            ragged = true;
            n_rows = std::max(n_rows, (size_t)c.n_rows);
            n_cols = std::max(n_cols, (size_t)c.n_cols);
        }
        if (im != nullptr)
        {
            if (src_f < im->size())
            {
                const arma::Mat<dtype> &ci = (*im)[src_f];
                if (ci.n_rows != n_rows || ci.n_cols != n_cols)
                {
                    ragged = true;
                    n_rows = std::max(n_rows, (size_t)ci.n_rows);
                    n_cols = std::max(n_cols, (size_t)ci.n_cols);
                }
            }
            else
                ragged = true; // selected frame has no imaginary counterpart -> gap
        }
    }

    if (interleaved_re)
        n_rows /= 2; // re frames hold interleaved pairs -> half as many complex rows

    // Interleaved real case doubles the row count
    size_t out_rows = n_rows;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_rows = 2 * n_rows;

    const size_t n_bytes = sizeof(dtype_numpy);
    const size_t out_frame = out_rows * n_cols;
    const size_t n_total = out_frame * n_frames;
    std::array<py::ssize_t, 3> shape = {(py::ssize_t)out_rows, (py::ssize_t)n_cols, (py::ssize_t)n_frames};
    std::array<py::ssize_t, 3> strides = {(py::ssize_t)n_bytes, py::ssize_t(out_rows * n_bytes), py::ssize_t(out_frame * n_bytes)};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    if (ragged)
        std::fill_n(p_out, n_total, dtype_numpy{});

    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? (size_t)i_vec[f] : f;
        const arma::Mat<dtype> &mr = (*re)[src_f];
        const size_t r_re = mr.n_rows, c_re = mr.n_cols;
        const dtype *p_re = mr.memptr();

        const bool has_im = (im != nullptr) && (src_f < im->size());
        const size_t r_im = has_im ? (*im)[src_f].n_rows : 0;
        const size_t c_im = has_im ? (*im)[src_f].n_cols : 0;
        const dtype *p_im = has_im ? (*im)[src_f].memptr() : nullptr;

        if constexpr (!qd_is_complex<dtype_numpy>::value)
        {
            dtype_numpy *dst = p_out + f * out_frame;
            if (im == nullptr) // Case 1: real output with cast (per-column)
                for (size_t j = 0; j < c_re; ++j)
                {
                    dtype_numpy *dc = dst + j * n_rows;
                    const dtype *sc = p_re + j * r_re;
                    for (size_t i = 0; i < r_re; ++i)
                        dc[i] = (dtype_numpy)sc[i];
                }
            else // Case 2: interleaved re/im along rows (out_rows == 2 * n_rows)
            {
                for (size_t j = 0; j < c_re; ++j)
                {
                    dtype_numpy *dc = dst + j * out_rows;
                    const dtype *sc = p_re + j * r_re;
                    for (size_t i = 0; i < r_re; ++i)
                        dc[2 * i] = (dtype_numpy)sc[i];
                }
                for (size_t j = 0; j < c_im; ++j)
                {
                    dtype_numpy *dc = dst + j * out_rows;
                    const dtype *sc = p_im + j * r_im;
                    for (size_t i = 0; i < r_im; ++i)
                        dc[2 * i + 1] = (dtype_numpy)sc[i];
                }
            }
        }
        else // Case 3: complex output (imag = 0 where im is absent or short)
        {
            using real_t = typename dtype_numpy::value_type;
            dtype_numpy *dst = p_out + f * out_frame;
            if (has_im)
            {
                const size_t rr = std::max(r_re, r_im);
                const size_t cc = std::max(c_re, c_im);
                for (size_t j = 0; j < cc; ++j)
                {
                    dtype_numpy *dc = dst + j * n_rows;
                    for (size_t i = 0; i < rr; ++i)
                    {
                        const real_t vr = (i < r_re && j < c_re) ? (real_t)p_re[j * r_re + i] : real_t(0);
                        const real_t vi = (i < r_im && j < c_im) ? (real_t)p_im[j * r_im + i] : real_t(0);
                        dc[i] = dtype_numpy(vr, vi);
                    }
                }
            }
            else // im interleaved within re's rows
                for (size_t j = 0; j < c_re; ++j)
                {
                    dtype_numpy *dc = dst + j * n_rows;
                    const dtype *sc = p_re + j * r_re;
                    for (size_t i = 0; i < r_re / 2; ++i)
                        dc[i] = dtype_numpy((real_t)sc[2 * i], (real_t)sc[2 * i + 1]);
                }
        }
    }
    return output;
}

// Stack vector(s) of Cubes into a Numpy array (frames = vector index, ragged + zero-padded),
// with dtype -> dtype_numpy conversion and optional frame selection via i_vec.
template <typename dtype, typename dtype_numpy = dtype>
static py::array_t<dtype_numpy> qd_python_stack2numpy(const std::vector<arma::Cube<dtype>> *re,
                                                      const std::vector<arma::Cube<dtype>> *im = nullptr,
                                                      const arma::uvec &i_vec = {})
{
    static_assert(std::is_arithmetic_v<dtype_numpy> || qd_is_complex<dtype_numpy>::value, "dtype_numpy must be an arithmetic type or std::complex<...>");

    if (re == nullptr || re->empty())
        return py::array_t<dtype_numpy>();

    const bool select = !i_vec.empty();
    const size_t n_frames = select ? (size_t)i_vec.n_elem : re->size();

    // Validate the frame selector up front
    if (select)
        for (size_t f = 0; f < n_frames; ++f)
            if (i_vec[f] >= re->size())
                throw std::out_of_range("Index out of bound.");

    // Complex output without an explicit imag part: each re frame's rows are interpreted as interleaved pairs [re0, im0, re1, im1, ...]
    const bool interleaved_re = qd_is_complex<dtype_numpy>::value && (im == nullptr);

    // Ragged scan over the selected frames (re and, if present, im)
    bool ragged = false;
    size_t n_rows = (*re)[select ? (size_t)i_vec[0] : 0].n_rows;
    size_t n_cols = (*re)[select ? (size_t)i_vec[0] : 0].n_cols;
    size_t n_slices = (*re)[select ? (size_t)i_vec[0] : 0].n_slices;
    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? (size_t)i_vec[f] : f;
        const arma::Cube<dtype> &c = (*re)[src_f];
        if (interleaved_re && (c.n_rows % 2 != 0))
            throw std::invalid_argument("Interleaved-complex input requires an even row count.");
        if (c.n_rows != n_rows || c.n_cols != n_cols || c.n_slices != n_slices)
        {
            ragged = true;
            n_rows = std::max(n_rows, (size_t)c.n_rows);
            n_cols = std::max(n_cols, (size_t)c.n_cols);
            n_slices = std::max(n_slices, (size_t)c.n_slices);
        }
        if (im != nullptr)
        {
            if (src_f < im->size())
            {
                const arma::Cube<dtype> &ci = (*im)[src_f];
                if (ci.n_rows != n_rows || ci.n_cols != n_cols || ci.n_slices != n_slices)
                {
                    ragged = true;
                    n_rows = std::max(n_rows, (size_t)ci.n_rows);
                    n_cols = std::max(n_cols, (size_t)ci.n_cols);
                    n_slices = std::max(n_slices, (size_t)ci.n_slices);
                }
            }
            else
                ragged = true; // selected frame has no imaginary counterpart -> gap
        }
    }

    if (interleaved_re)
        n_rows /= 2; // re frames hold interleaved pairs -> half as many complex rows

    // Interleaved real case doubles the row count
    size_t out_rows = n_rows;
    if constexpr (!qd_is_complex<dtype_numpy>::value)
        if (im != nullptr)
            out_rows = 2 * n_rows;

    const size_t n_bytes = sizeof(dtype_numpy);
    const size_t out_slice = out_rows * n_cols;
    const size_t out_frame = out_slice * n_slices;
    const size_t n_total = out_frame * n_frames;
    std::array<py::ssize_t, 4> shape = {(py::ssize_t)out_rows, (py::ssize_t)n_cols, (py::ssize_t)n_slices, (py::ssize_t)n_frames};
    std::array<py::ssize_t, 4> strides = {(py::ssize_t)n_bytes, py::ssize_t(out_rows * n_bytes), py::ssize_t(out_slice * n_bytes), py::ssize_t(out_frame * n_bytes)};
    py::array_t<dtype_numpy> output(shape, strides);
    dtype_numpy *p_out = output.mutable_data();

    if (ragged)
        std::fill_n(p_out, n_total, dtype_numpy{});

    for (size_t f = 0; f < n_frames; ++f)
    {
        const size_t src_f = select ? (size_t)i_vec[f] : f;
        const arma::Cube<dtype> &cr = (*re)[src_f];
        const size_t r_re = cr.n_rows, c_re = cr.n_cols, s_re = cr.n_slices;
        const dtype *p_re = cr.memptr();
        const size_t sl_re = r_re * c_re;

        const bool has_im = (im != nullptr) && (src_f < im->size());
        const size_t r_im = has_im ? (*im)[src_f].n_rows : 0;
        const size_t c_im = has_im ? (*im)[src_f].n_cols : 0;
        const size_t s_im = has_im ? (*im)[src_f].n_slices : 0;
        const dtype *p_im = has_im ? (*im)[src_f].memptr() : nullptr;
        const size_t sl_im = r_im * c_im;

        if constexpr (!qd_is_complex<dtype_numpy>::value)
        {
            dtype_numpy *dst = p_out + f * out_frame;
            if (im == nullptr) // Case 1: real output with cast (per-column)
                for (size_t k = 0; k < s_re; ++k)
                {
                    dtype_numpy *dk = dst + k * out_slice;
                    const dtype *sk = p_re + k * sl_re;
                    for (size_t j = 0; j < c_re; ++j)
                    {
                        dtype_numpy *dc = dk + j * n_rows;
                        const dtype *sc = sk + j * r_re;
                        for (size_t i = 0; i < r_re; ++i)
                            dc[i] = (dtype_numpy)sc[i];
                    }
                }
            else // Case 2: interleaved re/im along rows (out_rows == 2 * n_rows)
            {
                for (size_t k = 0; k < s_re; ++k)
                {
                    dtype_numpy *dk = dst + k * out_slice;
                    const dtype *sk = p_re + k * sl_re;
                    for (size_t j = 0; j < c_re; ++j)
                    {
                        dtype_numpy *dc = dk + j * out_rows;
                        const dtype *sc = sk + j * r_re;
                        for (size_t i = 0; i < r_re; ++i)
                            dc[2 * i] = (dtype_numpy)sc[i];
                    }
                }
                for (size_t k = 0; k < s_im; ++k)
                {
                    dtype_numpy *dk = dst + k * out_slice;
                    const dtype *sk = p_im + k * sl_im;
                    for (size_t j = 0; j < c_im; ++j)
                    {
                        dtype_numpy *dc = dk + j * out_rows;
                        const dtype *sc = sk + j * r_im;
                        for (size_t i = 0; i < r_im; ++i)
                            dc[2 * i + 1] = (dtype_numpy)sc[i];
                    }
                }
            }
        }
        else // Case 3: complex output (imag = 0 where im is absent or short)
        {
            using real_t = typename dtype_numpy::value_type;
            dtype_numpy *dst = p_out + f * out_frame;
            if (has_im)
            {
                const size_t rr = std::max(r_re, r_im);
                const size_t cc = std::max(c_re, c_im);
                const size_t ss = std::max(s_re, s_im);
                for (size_t k = 0; k < ss; ++k)
                {
                    dtype_numpy *dk = dst + k * out_slice;
                    for (size_t j = 0; j < cc; ++j)
                    {
                        dtype_numpy *dc = dk + j * n_rows;
                        for (size_t i = 0; i < rr; ++i)
                        {
                            const real_t vr = (i < r_re && j < c_re && k < s_re)
                                                  ? (real_t)p_re[k * sl_re + j * r_re + i]
                                                  : real_t(0);
                            const real_t vi = (i < r_im && j < c_im && k < s_im)
                                                  ? (real_t)p_im[k * sl_im + j * r_im + i]
                                                  : real_t(0);
                            dc[i] = dtype_numpy(vr, vi);
                        }
                    }
                }
            }
            else // im interleaved within re's rows
                for (size_t k = 0; k < s_re; ++k)
                {
                    dtype_numpy *dk = dst + k * out_slice;
                    const dtype *sk = p_re + k * sl_re;
                    for (size_t j = 0; j < c_re; ++j)
                    {
                        dtype_numpy *dc = dk + j * n_rows;
                        const dtype *sc = sk + j * r_re;
                        for (size_t i = 0; i < r_re / 2; ++i)
                            dc[i] = dtype_numpy((real_t)sc[2 * i], (real_t)sc[2 * i + 1]);
                    }
                }
        }
    }
    return output;
}

// -------------------------------- qd_python_init_output --------------------------------

template <typename dtype>
static py::array_t<dtype> qd_python_init_output(arma::uword n_elem,
                                                arma::Col<dtype> *wrapper = nullptr)
{
    const py::ssize_t n_bytes = (py::ssize_t)sizeof(dtype);

    std::array<py::ssize_t, 1> shape = {(py::ssize_t)n_elem};
    std::array<py::ssize_t, 1> strides = {n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
    {
        std::destroy_at(wrapper);
        ::new (wrapper) arma::Col<dtype>((dtype *)output.data(), n_elem, false, true);
    }

    return output;
}

template <typename dtype>
static py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols,
                                                arma::Mat<dtype> *wrapper = nullptr)
{
    const py::ssize_t n_bytes = (py::ssize_t)sizeof(dtype);

    std::array<py::ssize_t, 2> shape = {(py::ssize_t)n_rows, (py::ssize_t)n_cols};
    std::array<py::ssize_t, 2> strides = {n_bytes, (py::ssize_t)n_rows * n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
    {
        std::destroy_at(wrapper);
        ::new (wrapper) arma::Mat<dtype>((dtype *)output.data(), n_rows, n_cols, false, true);
    }

    return output;
}

template <typename dtype>
static py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols, arma::uword n_slices,
                                                arma::Cube<dtype> *wrapper = nullptr)
{
    const py::ssize_t n_bytes = (py::ssize_t)sizeof(dtype);

    std::array<py::ssize_t, 3> shape = {(py::ssize_t)n_rows, (py::ssize_t)n_cols, (py::ssize_t)n_slices};
    std::array<py::ssize_t, 3> strides = {n_bytes, (py::ssize_t)n_rows * n_bytes, py::ssize_t(n_rows * n_cols) * n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (wrapper != nullptr)
    {
        std::destroy_at(wrapper);
        ::new (wrapper) arma::Cube<dtype>((dtype *)output.data(), n_rows, n_cols, n_slices, false, true);
    }

    return output;
}

template <typename dtype>
static py::array_t<dtype> qd_python_init_output(arma::uword n_rows, arma::uword n_cols, arma::uword n_slices, arma::uword n_frames,
                                                std::vector<arma::Cube<dtype>> *cubes = nullptr)
{
    const py::ssize_t n_bytes = (py::ssize_t)sizeof(dtype);

    std::array<py::ssize_t, 4> shape = {(py::ssize_t)n_rows, (py::ssize_t)n_cols, (py::ssize_t)n_slices, (py::ssize_t)n_frames};
    std::array<py::ssize_t, 4> strides = {n_bytes, (py::ssize_t)n_rows * n_bytes, py::ssize_t(n_rows * n_cols) * n_bytes, py::ssize_t(n_rows * n_cols * n_slices) * n_bytes};
    py::array_t<dtype> output(shape, strides);

    if (cubes != nullptr)
    {
        cubes->resize(n_frames);
        dtype *base = (dtype *)output.mutable_data();
        size_t frame_stride = (size_t)n_rows * (size_t)n_cols * (size_t)n_slices;
        for (arma::uword i = 0; i < n_frames; ++i)
        {
            std::destroy_at(&(*cubes)[i]);
            ::new (&(*cubes)[i]) arma::Cube<dtype>(base + i * frame_stride, n_rows, n_cols, n_slices, false, true);
        }
    }

    return output;
}

// -------------------------------- qd_python_element_strided --------------------------------

// True if the array can be read with element strides: the data pointer is aligned for dtype, and every dimension
// with more than one entry has a non-negative stride that is a multiple of the item size. Not true for negative
// strides (x[::-1]), fields of packed structured arrays (byte strides like 12 for a float64 field) and unaligned
// buffers.
template <typename dtype>
static bool qd_python_has_element_strides(const py::array &input)
{
    if (reinterpret_cast<std::uintptr_t>(input.data()) % alignof(dtype) != 0)
        return false;
    for (py::ssize_t d = 0; d < input.ndim(); ++d)
        if (input.shape(d) > 1 && (input.strides(d) < 0 || input.strides(d) % (py::ssize_t)sizeof(dtype) != 0))
            return false;
    return true;
}

// Returns the input itself if it can be read with element strides, otherwise an aligned Fortran-contiguous copy made
// by numpy. Use before qd_python_get_shape on arrays that come from the user.
template <typename dtype>
static py::array_t<dtype> qd_python_element_strided(const py::array_t<dtype> &input)
{
    if (qd_python_has_element_strides<dtype>(input))
        return input;
    constexpr int flags = py::array::f_style | py::array::forcecast | py::detail::npy_api::NPY_ARRAY_ALIGNED_;
    auto output = py::array_t<dtype, flags>::ensure(input);
    if (!output)
        throw std::invalid_argument("Could not convert the input to a contiguous array.");
    return py::array_t<dtype>(output);
}

// -------------------------------- qd_python_get_shape --------------------------------

// 0-2: n_row, n_cols, n_slices
// 3-5: Strides in number of elements (not bytes!)
//   6: Fortran-contiguous (1) or not (0), as numpy's F_CONTIGUOUS flag (including the 4th dim if present)
//   7: Total number of elements
//   8: Total number of bytes
// Optional: n_frames (4th dim, 1 if input is 1D-3D), stride_frames (stride of 4th dim in elements)
// With shape_only = false, the array must have element strides (see qd_python_element_strided).
template <typename dtype>
static std::array<size_t, 9> qd_python_get_shape(const py::array_t<dtype> &input, bool shape_only = false,
                                                 size_t *n_frames = nullptr, size_t *stride_frames = nullptr)
{
    auto buf = input.request();
    int nd = (int)buf.ndim;
    const size_t n_bytes = sizeof(dtype);

    if (n_frames == nullptr && nd > 3)
        throw std::invalid_argument("Expected 1D, 2D or 3D array, got " + std::to_string(nd) + "D");

    if (nd > 4)
        throw std::invalid_argument("Expected 1D, 2D, 3D or 4D array, got " + std::to_string(nd) + "D");

    if (nd == 0)
    {
        if (n_frames != nullptr)
            *n_frames = 1;
        if (stride_frames != nullptr)
            *stride_frames = 1;
        return {1, 1, 1, 1, 1, 1, 1, 1, n_bytes};
    }

    std::array<size_t, 9> shape;
    shape[0] = (size_t)buf.shape[0];
    shape[1] = (nd >= 2) ? (size_t)buf.shape[1] : 1;
    shape[2] = (nd >= 3) ? (size_t)buf.shape[2] : 1;

    size_t nf = (nd == 4) ? (size_t)buf.shape[3] : 1;
    if (n_frames != nullptr)
        *n_frames = nf;

    nd = (shape[1] == 1 && shape[2] == 1) ? 1 : (shape[2] == 1 ? 2 : 3);

    if (shape_only)
    {
        if (stride_frames != nullptr)
            *stride_frames = shape[0] * shape[1] * shape[2];
        return shape;
    }

    if (buf.itemsize != (py::ssize_t)n_bytes)
        throw std::invalid_argument("Dtype size mismatch");

    if (!qd_python_has_element_strides<dtype>(input))
        throw std::invalid_argument("Internal error: array strides are not a non-negative multiple of the item size.");

    shape[3] = (size_t)buf.strides[0] / n_bytes;
    shape[4] = (nd >= 2) ? (size_t)buf.strides[1] / n_bytes : shape[0];
    shape[5] = (nd == 3) ? (size_t)buf.strides[2] / n_bytes : shape[0] * shape[1];

    size_t sf = (nf > 1) ? (size_t)buf.strides[3] / n_bytes : shape[0] * shape[1] * shape[2];
    if (stride_frames != nullptr)
        *stride_frames = sf;

    // Fortran-contiguous (including 4th dim if present); numpy's flag also covers the arbitrary strides of
    // dimensions with a single entry, e.g. a C-ordered (n, 1) array
    shape[6] = (input.flags() & py::array::f_style) ? 1 : 0;

    shape[7] = shape[0] * shape[1] * shape[2];           // Total number of elements (per frame)
    shape[8] = shape[0] * shape[1] * shape[2] * n_bytes; // Total number of bytes (per frame)

    return shape;
}

// 0-2: n_row, n_cols, n_slices
// 3-5: Strides in number of elements (not bytes!)
//   6: Fortran-contiguous (1) or C-contiguous (0)
//   7: Total number of elements
//   8: Total number of bytes
template <typename dtype>
static std::vector<std::array<size_t, 9>> qd_python_get_list_shape(const py::list &input,
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
        typed_array = qd_python_element_strided(typed_array);

        output[i] = qd_python_get_shape(typed_array);
        buffer_pointers[i] = const_cast<dtype *>(typed_array.data());
        owned_arrays.push_back(std::move(typed_array)); // store to keep alive
    }
    return output;
}

// -------------------------------- qd_python_copy2arma --------------------------------

template <typename dtype>
static void qd_python_copy2arma(const dtype *src, const std::array<size_t, 9> &shape,
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
    {
        if (shape[8]) // No memcpy for empty arrays (null data pointer)
            std::memcpy(dst, src, shape[8]);
    }
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
static void qd_python_copy2arma(const py::array_t<dtype> &input, arma::Cube<dtype> &output)
{
    const auto array = qd_python_element_strided(input);
    auto shape = qd_python_get_shape(array);
    const dtype *src = array.data();
    qd_python_copy2arma(src, shape, output);
}

template <typename dtype>
static void qd_python_copy2arma(const std::complex<dtype> *src, const std::array<size_t, 9> &shape,
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
static void qd_python_copy2arma(const py::array_t<std::complex<dtype>> &input, arma::Cube<dtype> &real, arma::Cube<dtype> &imag)
{
    const auto array = qd_python_element_strided(input);
    auto shape = qd_python_get_shape(array);
    const std::complex<dtype> *src = array.data();
    qd_python_copy2arma(src, shape, real, imag);
}

// -------------------------------- qd_python_numpy2arma --------------------------------
//
// Convert a numpy array to an Armadillo object.
// - view = false: the data is copied.
// - view = true: the Armadillo object aliases the numpy memory (copy_aux_mem = false, strict = true) when the array
//   is Fortran-contiguous and aligned; otherwise the data is copied. view = true, strict = true throws instead of
//   copying. Empty arrays give an empty object (nothing to alias).
// - What may be aliased, by overload:
//   - `const py::array_t<dtype> &` (an lvalue): the array the call site holds, e.g. a wrapper parameter. If pybind11
//     had to convert the caller's object (other dtype, list, scalar), this is the converted array, which pybind11
//     keeps in its argument caster until the wrapped function returns.
//   - `py::array_t<dtype> &&` (a temporary): never aliased, always copied, since the temporary is released before
//     the result is used.
//   - `const py::handle &`: aliased only when the object already is an ndarray of dtype (no conversion, the caller's
//     own array); everything else is converted into a temporary and copied.
// - Rules for code that uses view = true: an aliased object is read-only (never written, resized or assigned to),
//   it must not outlive the wrapper call or be stored in a returned object, and its source array must stay
//   referenced while the GIL is released (other threads may remove list entries or dict values). Read-only numpy
//   arrays (flags.writeable = False) are aliased for reading as well.

// True if an Armadillo object may alias the memory of 'input'
template <typename dtype>
static bool qd_python_can_alias(const py::array_t<dtype> &input)
{
    const int flags = input.flags();
    return (flags & py::array::f_style) && (flags & py::detail::npy_api::NPY_ARRAY_ALIGNED_);
}

static void qd_python_no_view_error()
{
    throw std::invalid_argument("Could not obtain memory view, possibly due to mismatching strides.");
}

template <typename dtype>
static arma::Col<dtype> qd_python_numpy2arma_Col(const py::array_t<dtype> &input,
                                                 bool view = false, bool strict = false,
                                                 std::string var_name = "", arma::uword n_elem = 0)
{
    const auto shape = qd_python_get_shape(input, true);
    const arma::uword n = (arma::uword)(shape[0] * shape[1] * shape[2]);
    if (n_elem != 0 && n != n_elem)
    {
        if (var_name.empty())
            throw std::invalid_argument("Incorrect number of elements.");
        else
            throw std::invalid_argument("Input '" + var_name + "' has incorrect number of elements.");
    }
    if (n == 0)
        return arma::Col<dtype>();
    if (view && qd_python_can_alias(input))
        return arma::Col<dtype>(const_cast<dtype *>(input.data()), n, false, true);
    if (view && strict)
        qd_python_no_view_error();

    arma::Col<dtype> output(n, arma::fill::none);
    arma::Cube<dtype> wrapper(output.memptr(), shape[0], shape[1], shape[2], false, true);
    qd_python_copy2arma(input, wrapper);
    return output;
}

template <typename dtype>
static arma::Col<dtype> qd_python_numpy2arma_Col(py::array_t<dtype> &&input,
                                                 bool view = false, bool strict = false,
                                                 std::string var_name = "", arma::uword n_elem = 0)
{
    const py::array_t<dtype> &array = input;
    if (view && strict && array.size() != 0)
        qd_python_no_view_error();
    return qd_python_numpy2arma_Col(array, false, false, var_name, n_elem);
}

template <typename dtype>
static arma::Mat<dtype> qd_python_numpy2arma_Mat(const py::array_t<dtype> &input, bool view = false, bool strict = false)
{
    const auto shape = qd_python_get_shape(input, true);
    const arma::uword n_rows = (arma::uword)shape[0], n_cols = (arma::uword)(shape[1] * shape[2]);
    if (n_rows * n_cols == 0)
        return arma::Mat<dtype>(n_rows, n_cols);
    if (view && qd_python_can_alias(input))
        return arma::Mat<dtype>(const_cast<dtype *>(input.data()), n_rows, n_cols, false, true);
    if (view && strict)
        qd_python_no_view_error();

    arma::Mat<dtype> output(n_rows, n_cols, arma::fill::none);
    arma::Cube<dtype> wrapper(output.memptr(), shape[0], shape[1], shape[2], false, true);
    qd_python_copy2arma(input, wrapper);
    return output;
}

template <typename dtype>
static arma::Mat<dtype> qd_python_numpy2arma_Mat(py::array_t<dtype> &&input, bool view = false, bool strict = false)
{
    const py::array_t<dtype> &array = input;
    if (view && strict && array.size() != 0)
        qd_python_no_view_error();
    return qd_python_numpy2arma_Mat(array, false, false);
}

template <typename dtype>
static arma::Cube<dtype> qd_python_numpy2arma_Cube(const py::array_t<dtype> &input, bool view = false, bool strict = false)
{
    const auto shape = qd_python_get_shape(input, true);
    if (shape[0] * shape[1] * shape[2] == 0)
        return arma::Cube<dtype>(shape[0], shape[1], shape[2]);
    if (view && qd_python_can_alias(input))
        return arma::Cube<dtype>(const_cast<dtype *>(input.data()), shape[0], shape[1], shape[2], false, true);
    if (view && strict)
        qd_python_no_view_error();

    arma::Cube<dtype> output(shape[0], shape[1], shape[2], arma::fill::none);
    qd_python_copy2arma(input, output);
    return output;
}

template <typename dtype>
static arma::Cube<dtype> qd_python_numpy2arma_Cube(py::array_t<dtype> &&input, bool view = false, bool strict = false)
{
    const py::array_t<dtype> &array = input;
    if (view && strict && array.size() != 0)
        qd_python_no_view_error();
    return qd_python_numpy2arma_Cube(array, false, false);
}

// py::handle overloads: None gives an empty object; only an ndarray of dtype (the caller's own array) can be aliased
template <typename dtype>
static arma::Col<dtype> qd_python_numpy2arma_Col(const py::handle &obj, bool view = false, bool strict = false)
{
    if (obj.is_none())
        return arma::Col<dtype>();
    if (py::isinstance<py::array_t<dtype>>(obj))
    {
        const auto array = py::reinterpret_borrow<py::array_t<dtype>>(obj);
        return qd_python_numpy2arma_Col(array, view, strict);
    }
    if (view && strict)
        throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
    return qd_python_numpy2arma_Col(py::cast<py::array_t<dtype>>(obj));
}

template <typename dtype>
static arma::Mat<dtype> qd_python_numpy2arma_Mat(const py::handle &obj, bool view = false, bool strict = false)
{
    if (obj.is_none())
        return arma::Mat<dtype>();
    if (py::isinstance<py::array_t<dtype>>(obj))
    {
        const auto array = py::reinterpret_borrow<py::array_t<dtype>>(obj);
        return qd_python_numpy2arma_Mat(array, view, strict);
    }
    if (view && strict)
        throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
    return qd_python_numpy2arma_Mat(py::cast<py::array_t<dtype>>(obj));
}

template <typename dtype>
static arma::Cube<dtype> qd_python_numpy2arma_Cube(const py::handle &obj, bool view = false, bool strict = false)
{
    if (obj.is_none())
        return arma::Cube<dtype>();
    if (py::isinstance<py::array_t<dtype>>(obj))
    {
        const auto array = py::reinterpret_borrow<py::array_t<dtype>>(obj);
        return qd_python_numpy2arma_Cube(array, view, strict);
    }
    if (view && strict)
        throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
    return qd_python_numpy2arma_Cube(py::cast<py::array_t<dtype>>(obj));
}

// Convert a 3D or 4D Numpy array to std::vector<arma::Cube<dtype>>
// 3D input → vector with 1 entry, 4D input → vector with n_frames entries
// Aliasing rules as for qd_python_numpy2arma_Cube; a Fortran-contiguous array gives one alias per frame.
template <typename dtype>
static std::vector<arma::Cube<dtype>> qd_python_numpy2arma_vecCube(const py::array_t<dtype> &input,
                                                                   bool view = false, bool strict = false)
{
    size_t n_frames, stride_frames;
    auto shape = qd_python_get_shape(input, true, &n_frames, &stride_frames);
    const bool empty = shape[0] * shape[1] * shape[2] == 0;

    if (view && !empty && qd_python_can_alias(input)) // Fully Fortran-contiguous: direct memory views
    {
        std::vector<arma::Cube<dtype>> output(n_frames);
        dtype *base = const_cast<dtype *>(input.data());
        for (size_t i = 0; i < n_frames; ++i)
            output[i] = arma::Cube<dtype>(base + i * stride_frames, shape[0], shape[1], shape[2], false, true);
        return output;
    }

    if (view && strict && !empty && n_frames != 0)
        qd_python_no_view_error();

    // Copy path: offset source pointer per frame, delegate to existing copy2arma
    const auto array = qd_python_element_strided(input);
    shape = qd_python_get_shape(array, false, &n_frames, &stride_frames);
    std::vector<arma::Cube<dtype>> output(n_frames);
    const dtype *src = array.data();
    for (size_t i = 0; i < n_frames; ++i)
    {
        output[i].set_size(shape[0], shape[1], shape[2]);
        qd_python_copy2arma(src + i * stride_frames, shape, output[i]);
    }
    return output;
}

template <typename dtype>
static std::vector<arma::Cube<dtype>> qd_python_numpy2arma_vecCube(py::array_t<dtype> &&input,
                                                                   bool view = false, bool strict = false)
{
    const py::array_t<dtype> &array = input;
    if (view && strict && array.size() != 0)
        qd_python_no_view_error();
    return qd_python_numpy2arma_vecCube(array, false, false);
}

// py::handle overload for qd_python_numpy2arma_vecCube
template <typename dtype>
static std::vector<arma::Cube<dtype>> qd_python_numpy2arma_vecCube(const py::handle &obj,
                                                                   bool view = false, bool strict = false)
{
    if (py::isinstance<py::array_t<dtype>>(obj))
    {
        const auto array = py::reinterpret_borrow<py::array_t<dtype>>(obj);
        return qd_python_numpy2arma_vecCube(array, view, strict);
    }
    if (view && strict)
        throw std::invalid_argument("Expected a numpy.ndarray, but got something else.");
    return qd_python_numpy2arma_vecCube(py::cast<py::array_t<dtype>>(obj));
}

template <typename dtype>
static void qd_python_numpy2arma_vecCube_Cplx(const py::array_t<std::complex<dtype>> &input,
                                              std::vector<arma::Cube<dtype>> &real,
                                              std::vector<arma::Cube<dtype>> &imag)
{
    size_t n_frames, stride_frames;
    const auto array = qd_python_element_strided(input);
    auto shape = qd_python_get_shape(array, false, &n_frames, &stride_frames);

    real.clear();
    real.resize(n_frames);
    imag.clear();
    imag.resize(n_frames);

    const std::complex<dtype> *src = array.data();
    for (size_t i = 0; i < n_frames; ++i)
        qd_python_copy2arma(src + i * stride_frames, shape, real[i], imag[i]);
}

// py::handle overload
template <typename dtype>
static void qd_python_numpy2arma_vecCube_Cplx(const py::handle &obj,
                                              std::vector<arma::Cube<dtype>> &real,
                                              std::vector<arma::Cube<dtype>> &imag)
{
    auto pyarray = py::cast<py::array_t<std::complex<dtype>>>(obj);
    qd_python_numpy2arma_vecCube_Cplx(pyarray, real, imag);
}

// Convert a 1D or 2D Numpy array to std::vector<arma::Col<dtype>>
// 1D input → vector with 1 entry; 2D input → vector with n_cols entries (one Col per column).
// Always copies (column extraction cannot alias). None → empty vector.
template <typename dtype>
static std::vector<arma::Col<dtype>> qd_python_numpy2arma_vecCol(const py::handle &obj)
{
    if (obj.is_none())
        return {};
    const auto mat = qd_python_numpy2arma_Mat<dtype>(obj, true);
    std::vector<arma::Col<dtype>> output(mat.n_cols);
    for (arma::uword c = 0; c < mat.n_cols; ++c)
        output[c] = mat.col(c);
    return output;
}

// Convert a 2D or 3D Numpy array to std::vector<arma::Mat<dtype>>
// 2D input → vector with 1 entry; 3D input → vector with n_slices entries (one Mat per slice).
// Always copies (slice extraction cannot alias). None → empty vector.
// Works for real dtypes and for std::complex<...>.
template <typename dtype>
static std::vector<arma::Mat<dtype>> qd_python_numpy2arma_vecMat(const py::handle &obj)
{
    if (obj.is_none())
        return {};
    const auto cube = qd_python_numpy2arma_Cube<dtype>(obj, true);
    std::vector<arma::Mat<dtype>> output(cube.n_slices);
    for (arma::uword s = 0; s < cube.n_slices; ++s)
        output[s] = cube.slice(s);
    return output;
}

// -------------------------------- qd_python_list2vector --------------------------------

template <typename dtype>
static std::vector<arma::Col<dtype>> qd_python_list2vector_Col(const py::list &input)
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
static std::vector<arma::Mat<dtype>> qd_python_list2vector_Mat(const py::list &input)
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
static std::vector<arma::Cube<dtype>> qd_python_list2vector_Cube(const py::list &input)
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
static void qd_python_list2vector_Cube_Cplx(const py::list &input,
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

static std::vector<std::string> qd_python_list2vector_Strings(const py::handle &input)
{
    std::vector<std::string> output;

    // None (or a null handle) -> empty vector
    if (!input || input.is_none())
        return output;

    // A bare string -> single-element vector
    if (py::isinstance<py::str>(input))
    {
        output.emplace_back(py::cast<std::string>(input));
        return output;
    }

    // Otherwise expect a list or tuple of strings; reject anything else
    if (!py::isinstance<py::list>(input) && !py::isinstance<py::tuple>(input))
        throw std::invalid_argument("Expected a string, a list of strings, or None.");

    auto seq = py::reinterpret_borrow<py::sequence>(input);
    output.reserve(py::len(seq));
    for (py::handle obj : seq)
    {
        if (!py::isinstance<py::str>(obj))
            throw std::invalid_argument("Expected a string, a list of strings, or None.");
        output.emplace_back(py::cast<std::string>(obj));
    }

    return output;
}

// -------------------------------- Imterleave complex --------------------------------

template <typename dtype>
static arma::Mat<dtype> qd_python_Complex2Interleaved(const arma::Mat<std::complex<dtype>> &input)
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
static std::vector<arma::Mat<dtype>> qd_python_Complex2Interleaved(const std::vector<arma::Mat<std::complex<dtype>>> &input)
{
    size_t n_input = input.size();
    std::vector<arma::Mat<dtype>> output(n_input);
    for (size_t i = 0; i < n_input; ++i)
        output[i] = qd_python_Complex2Interleaved(input[i]);
    return output;
}

// -------------------------------- std::any --------------------------------

// Convert a numpy array (or 0-d array) of a supported element type to a std::any; any strides and byte order.
// The caller's own Fortran-contiguous array is aliased (rules of qd_python_numpy2arma: the std::any is read-only and
// must not outlive the wrapper call); converted arrays (other byte order) and other layouts are copied.
template <typename dtype>
static std::any qd_python_anycast_array(const py::array &obj, const std::string &var_name)
{
    const auto arr = py::array_t<dtype>::ensure(obj); // Native byte order; a copy only if the byte order differs
    if (!arr)
        throw std::invalid_argument("Input '" + var_name + "' could not be converted.");
    const bool view = arr.ptr() == obj.ptr(); // No conversion: 'arr' is the caller's array

    switch (arr.ndim())
    {
    case 0:
        return *arr.data();
    case 1: // Stored as a column vector
    {
        const arma::uword n = (arma::uword)arr.shape(0);
        if (view && n != 0 && qd_python_can_alias(arr))
            return arma::Row<dtype>(const_cast<dtype *>(arr.data()), n, false, true);
        arma::Row<dtype> output(n, arma::fill::none);
        arma::Cube<dtype> wrapper(output.memptr(), n, 1, 1, false, true);
        qd_python_copy2arma(arr, wrapper);
        return output;
    }
    case 2: // Also (n, 1) and (1, 1), so that the shape is kept
        return qd_python_numpy2arma_Mat<dtype>(arr, view);
    case 3:
        return qd_python_numpy2arma_Cube<dtype>(arr, view);
    default:
        throw std::invalid_argument("Input '" + var_name + "' has more than 3 dimensions.");
    }
}

// Convert a Python object to a std::any (unstructured data of the HDF5 writers)
// - str -> std::string, float -> double, int and bool -> long long (a bool is stored as the integer 0 or 1)
// - numpy scalars and 0-d arrays -> scalar of their element type
// - numpy arrays: 1D -> arma::Row, 2D -> arma::Mat, 3D -> arma::Cube; Fortran-contiguous arrays are aliased, all
//   others copied (any strides, byte order)
// - Element types: float32, float64, int32, uint32, int64, uint64; any other type throws
static std::any qd_python_anycast(py::handle obj, std::string var_name = "")
{
    if (py::isinstance<py::str>(obj))
        return py::cast<std::string>(obj);

    if (py::isinstance<py::float_>(obj))
        return py::cast<double>(obj);

    if (py::isinstance<py::int_>(obj))
        return py::cast<long long>(obj);

    py::object array_obj; // Stays null for unsupported types (a default py::array would be an empty array)
    if (py::isinstance<py::array>(obj))
        array_obj = py::reinterpret_borrow<py::object>(obj);
    else if (py::isinstance(obj, py::module_::import("numpy").attr("generic"))) // numpy scalar -> 0-d array
        array_obj = py::array::ensure(obj);

    if (array_obj)
    {
        const auto arr = py::reinterpret_borrow<py::array>(array_obj);
        const char kind = arr.dtype().kind();
        const py::ssize_t size = arr.dtype().itemsize();

        if (kind == 'f' && size == 8)
            return qd_python_anycast_array<double>(arr, var_name);
        if (kind == 'f' && size == 4)
            return qd_python_anycast_array<float>(arr, var_name);
        if (kind == 'u' && size == 4)
            return qd_python_anycast_array<unsigned>(arr, var_name);
        if (kind == 'i' && size == 4)
            return qd_python_anycast_array<int>(arr, var_name);
        if (kind == 'u' && size == 8)
            return qd_python_anycast_array<unsigned long long>(arr, var_name);
        if (kind == 'i' && size == 8)
            return qd_python_anycast_array<long long>(arr, var_name);
    }

    std::string error_message = "Input '" + var_name + "' has an unsupported type.";
    throw std::invalid_argument(error_message);

    return std::any();
}

// -------------------------------- qd_python_map2dict / dict2map --------------------------------

// Convert std::unordered_map<std::string, std::vector<dtype>> to a Python dict.
// Each key becomes a dict entry; each value vector becomes a 1D numpy array of length n_mtl.
// 'key_order' (optional) fixes the insertion order of the keys present in the map; keys not in
// the map are skipped, and any map key not listed is appended afterwards in iteration order.
template <typename dtype, typename dtype_numpy = dtype>
static py::dict qd_python_map2dict(const std::unordered_map<std::string, std::vector<dtype>> &map,
                                   const std::vector<std::string> &key_order = {})
{
    py::dict output;

    auto emit = [&](const std::string &key)
    {
        const std::vector<dtype> &vec = map.at(key);
        arma::Col<dtype> col(const_cast<dtype *>(vec.data()), vec.size(), false, true); // no-copy alias
        output[py::str(key)] = qd_python_copy2numpy<dtype, dtype_numpy>(&col);
    };

    std::unordered_map<std::string, bool> done;
    for (const auto &key : key_order)
        if (map.find(key) != map.end() && !done[key])
            emit(key), done[key] = true;

    for (const auto &kv : map)
        if (!done[kv.first])
            emit(kv.first), done[kv.first] = true;

    return output;
}

// Convert a Python dict (or None) to std::unordered_map<std::string, std::vector<dtype>>.
// Each dict entry's value is read as a 1D array and cast to dtype. All values must have the same
// length (n_mtl); a mismatch throws. None or an empty dict yields an empty map.
template <typename dtype>
static std::unordered_map<std::string, std::vector<dtype>> qd_python_dict2map(const py::handle &obj)
{
    std::unordered_map<std::string, std::vector<dtype>> map;
    if (obj.is_none())
        return map;
    if (!py::isinstance<py::dict>(obj))
        throw std::invalid_argument("Expected a dict mapping column names to 1D arrays.");

    py::dict d = py::reinterpret_borrow<py::dict>(obj);

    bool n_set = false;
    arma::uword n_mtl = 0;
    for (auto item : d)
    {
        const std::string key = py::str(item.first);
        arma::Col<dtype> col = qd_python_numpy2arma_Col<dtype>(item.second, false, false);

        if (!n_set)
            n_mtl = col.n_elem, n_set = true;
        else if (col.n_elem != n_mtl)
            throw std::invalid_argument("Material property column '" + key + "' has " +
                                        std::to_string(col.n_elem) + " elements, expected " +
                                        std::to_string(n_mtl) + " (all columns must have the same length).");

        map[key] = std::vector<dtype>(col.begin(), col.end());
    }
    return map;
}

#endif
