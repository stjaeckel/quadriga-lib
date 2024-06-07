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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "quadriga_lib.hpp"

#include "python_helpers.cpp" // qd_python_anycast

pybind11::array_t<unsigned> hdf5_write_channel(const std::string fn,
                                               unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                               const pybind11::dict par,
                                               const pybind11::array_t<double> rx_pos,
                                               const pybind11::array_t<double> tx_pos,
                                               const pybind11::array_t<std::complex<double>> coeff,
                                               const pybind11::array_t<double> delay,
                                               const pybind11::array_t<double> center_frequency,
                                               const std::string name,
                                               const int initial_position,
                                               const pybind11::array_t<double> path_gain,
                                               const pybind11::array_t<double> path_length,
                                               const pybind11::array_t<std::complex<double>> path_polarization,
                                               const pybind11::array_t<double> path_angles,
                                               const pybind11::array_t<double> path_fbs_pos,
                                               const pybind11::array_t<double> path_lbs_pos,
                                               const pybind11::array_t<unsigned> no_interact,
                                               const pybind11::array_t<double> interact_coord,
                                               const pybind11::array_t<double> rx_orientation,
                                               const pybind11::array_t<double> tx_orientation)
{
    // Construct channel object from input data
    auto c = quadriga_lib::channel<double>();
    c.initial_position = initial_position;
    c.name = name;

    // Process the unstructured data
    for (auto item : par)
    {
        std::string fieldName = pybind11::str(item.first);
        std::string fieldString = "par." + fieldName;
        c.par_names.push_back(std::string(fieldName));
        c.par_data.push_back(qd_python_anycast(item.second, fieldName));
    }

    if (rx_pos.size() != 0)
    {
        pybind11::buffer_info buf = rx_pos.request();
        if (buf.ndim != 2)
            throw std::invalid_argument("'rx_pos' must be a Matrix (2 dimensions).");
        c.rx_pos = arma::mat(reinterpret_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
    }

    if (tx_pos.size() != 0)
    {
        pybind11::buffer_info buf = tx_pos.request();
        if (buf.ndim != 2)
            throw std::invalid_argument("'tx_pos' must be a Matrix (2 dimensions).");
        c.tx_pos = arma::mat(reinterpret_cast<double *>(buf.ptr), buf.shape[0], buf.shape[1], false, true);
    }

    if (coeff.size() != 0)
    {
        pybind11::buffer_info buf = coeff.request();
        std::complex<double> *ptr_python = reinterpret_cast<std::complex<double> *>(buf.ptr);

        size_t n_rx = buf.shape[0];
        size_t n_tx = (buf.ndim > 1) ? buf.shape[1] : 1;
        size_t n_path = (buf.ndim > 2) ? buf.shape[2] : 1;
        size_t n_snap = (buf.ndim > 3) ? buf.shape[3] : 1;

        auto data_real = std::vector<arma::cube>();
        auto data_imag = std::vector<arma::cube>();

        for (size_t i_snap = 0; i_snap < n_snap; ++i_snap)
        {
            auto tmp_real = arma::cube(n_rx, n_tx, n_path, arma::fill::none);
            auto tmp_imag = arma::cube(n_rx, n_tx, n_path, arma::fill::none);
            double *ptr_real = tmp_real.memptr();
            double *ptr_imag = tmp_imag.memptr();

            for (size_t i = 0; i < n_rx * n_tx * n_path; ++i)
            {
                std::complex<double> value = ptr_python[i];
                ptr_real[i] = value.real();
                ptr_imag[i] = value.imag();
            }

            data_real.push_back(tmp_real);
            data_imag.push_back(tmp_imag);
        }

        c.coeff_re = data_real;
        c.coeff_im = data_imag;
    }

    if (center_frequency.size() != 0)
    {
        pybind11::buffer_info buf = center_frequency.request();
        c.center_frequency = arma::vec(reinterpret_cast<double *>(buf.ptr), buf.size, false, true);
    }

    // Create HDF File if it dies not already exist
    auto storage_space = quadriga_lib::hdf5_read_layout(fn);
    if (storage_space.at(0) == 0) // File does not exist
    {
        unsigned nx = 1, ny = 1, nz = 1, nw = 1;
        if (iy == 0 && iz == 0 && iw == 0)
            nx = ix > 65536U ? ix : 65536U;
        else if (iz == 0 && iw == 0)
            nx = ix > 1024U ? ix : 1024U,
            ny = iy > 64U ? iy : 64U;
        else if (iw == 0)
            nx = ix > 256U ? ix : 256U,
            ny = iy > 16U ? iy : 16U,
            nz = iz > 16U ? iz : 16U;
        else
            nx = ix > 128U ? ix : 128U,
            ny = iy > 8U ? iy : 8U,
            nz = iz > 8U ? iz : 8U,
            nw = iz > 8U ? iw : 8U;

        quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);
        storage_space.at(0) = nx;
        storage_space.at(1) = ny;
        storage_space.at(2) = nz;
        storage_space.at(3) = nw;
    }

    // Throw error if location exceeds storage space
    if (ix > storage_space.at(0) || iy > storage_space.at(1) || iz > storage_space.at(2) || iw > storage_space.at(3))
        throw std::invalid_argument("Location exceeds storage space in HDF file");

    // Write data to file
    c.hdf5_write(fn, ix, iy, iz, iw);

    return pybind11::array_t<unsigned>(4ULL, storage_space.memptr());
}