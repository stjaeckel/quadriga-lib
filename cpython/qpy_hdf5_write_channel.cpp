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
                                               const pybind11::array_t<double> rx_position,
                                               const pybind11::array_t<double> tx_position,
                                               const pybind11::array_t<std::complex<double>> coeff,
                                               const pybind11::array_t<double> delay,
                                               const double center_frequency,
                                               const std::string name,
                                               const unsigned initial_position,
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

    // Process the unstructured data
    for (auto item : par)
    {
        std::string fieldName = pybind11::str(item.first);
        std::string fieldString = "par." + fieldName;
        c.par_names.push_back(std::string(fieldName));
        c.par_data.push_back(qd_python_anycast(item.second, fieldName));
    }

    // Write to HDF File
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

    c.hdf5_write(fn, ix, iy, iz, iw);

    return pybind11::array_t<unsigned>(4ULL, storage_space.memptr());
}