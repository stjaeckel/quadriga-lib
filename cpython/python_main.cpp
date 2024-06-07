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

namespace py = pybind11;

// Include parts
#include "qpy_cart2geo.cpp"
#include "qpy_icosphere.cpp"
#include "qpy_hdf5_create_file.cpp"
#include "qpy_hdf5_read_layout.cpp"
#include "qpy_hdf5_reshape_layout.cpp"
#include "qpy_hdf5_write_channel.cpp"

PYBIND11_MODULE(quadriga_lib, m)
{
    m.def("cart2geo", &cart2geo, py::arg("cart"));

    m.def("icosphere", &icosphere, py::arg("n_div") = 1,
          py::arg("radius") = 1.0, py::arg("direction_xyz") = false);

    m.def("hdf5_create_file", &hdf5_create_file, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_read_layout", &hdf5_read_layout, py::arg("fn"));

    m.def("hdf5_reshape_layout", &hdf5_reshape_layout, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_write_channel", &hdf5_write_channel, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::arg("par") = py::dict(),
          py::arg("rx_position") = py::array_t<double>(),
          py::arg("tx_position") = py::array_t<double>(),
          py::arg("coeff") = pybind11::array_t<std::complex<double>>(),
          py::arg("delay") = py::array_t<double>(),
          py::arg("center_frequency") = 1.0,
          py::arg("name") = "New channel",
          py::arg("initial_position") = 1,
          py::arg("path_gain") = py::array_t<double>(),
          py::arg("path_length") = py::array_t<double>(),
          py::arg("path_polarization") = py::array_t<std::complex<double>>(),
          py::arg("path_angles") = py::array_t<double>(),
          py::arg("path_fbs_pos") = py::array_t<double>(),
          py::arg("path_lbs_pos") = py::array_t<double>(),
          py::arg("no_interact") = py::array_t<unsigned>(),
          py::arg("interact_coord") = py::array_t<double>(),
          py::arg("rx_orientation") = py::array_t<double>(),
          py::arg("tx_orientation") = py::array_t<double>());
}
