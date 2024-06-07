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
            if (buf.ndim == 1)
                return arma::Row<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<double>(reinterpret_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }

        if (arr.dtype().is(pybind11::dtype::of<float>()))
        {
            if (buf.ndim == 1)
                return arma::Row<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<float>(reinterpret_cast<float *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }

        if (arr.dtype().is(pybind11::dtype::of<unsigned>()))
        {
            if (buf.ndim == 1)
                return arma::Row<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<unsigned>(reinterpret_cast<unsigned *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }

        if (arr.dtype().is(pybind11::dtype::of<int>()))
        {
            if (buf.ndim == 1)
                return arma::Row<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<int>(reinterpret_cast<int *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }

        if (arr.dtype().is(pybind11::dtype::of<unsigned long long>()))
        {
            if (buf.ndim == 1)
                return arma::Row<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<unsigned long long>(reinterpret_cast<unsigned long long *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }

        if (arr.dtype().is(pybind11::dtype::of<long long>()) || arr.dtype().is(pybind11::dtype::of<int64_t>()))
        {
            if (buf.ndim == 1)
                return arma::Row<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2 && buf.shape[1] == 1)
                return arma::Col<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], false, true);
            else if (buf.ndim == 2)
                return arma::Mat<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
            else if (buf.ndim == 3)
                return arma::Cube<long long>(reinterpret_cast<long long *>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], false, true);
        }
    }

    std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
    throw std::invalid_argument(error_message);

    return std::any();
}

#endif
