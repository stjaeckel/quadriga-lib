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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_baseband_freq_response.cpp"
#include "qpy_channel_export_obj_file.cpp"
#include "qpy_hdf5_create_file.cpp"
#include "qpy_hdf5_read_channel.cpp"
#include "qpy_hdf5_read_dset_names.cpp"
#include "qpy_hdf5_read_layout.cpp"
#include "qpy_hdf5_reshape_layout.cpp"
#include "qpy_hdf5_write_channel.cpp"
#include "qpy_hdf5_write_dset.cpp"

void quadriga_lib_channel(py::module_ &m)
{
     m.def("baseband_freq_response", &baseband_freq_response,
          py::arg("coeff") = py::list(),
          py::arg("delay") = py::list(),
          py::arg("bandwidth"),
          py::arg("carriers") = 128,
          py::arg("pilot_grid") = py::array_t<double>(),
          py::arg("snap") = py::array_t<arma::uword>());

    m.def("channel_export_obj_file", &channel_export_obj_file,
          py::arg("fn"),
          py::arg("max_no_paths") = 0,
          py::arg("gain_max") = -60.0,
          py::arg("gain_min") = -140.0,
          py::arg("colormap") = "jet",
          py::arg("radius_max") = 0.05,
          py::arg("radius_min") = 0.01,
          py::arg("n_edges") = 5,
          py::arg("rx_pos") = py::array_t<double>(),
          py::arg("tx_pos") = py::array_t<double>(),
          py::arg("no_interact") = py::list(),
          py::arg("interact_coord") = py::list(),
          py::arg("center_freq") = py::array_t<double>(),
          py::arg("coeff") = py::list(),
          py::arg("i_snap") = py::array_t<arma::uword>());

    m.def("hdf5_create_file", &hdf5_create_file, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_read_channel", &hdf5_read_channel, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::arg("snap") = py::array_t<arma::uword>());

    m.def("hdf5_read_dset_names", &hdf5_read_dset_names, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0);

    m.def("hdf5_read_layout", &hdf5_read_layout, py::arg("fn"));

    m.def("hdf5_reshape_layout", &hdf5_reshape_layout, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_write_channel", &hdf5_write_channel, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::arg("par") = py::dict(),
          py::arg("rx_pos") = py::array_t<double>(),
          py::arg("tx_pos") = py::array_t<double>(),
          py::arg("coeff") = py::list(),
          py::arg("delay") = py::list(),
          py::arg("center_frequency") = py::array_t<double>(),
          py::arg("name") = "New channel",
          py::arg("initial_position") = 0,
          py::arg("path_gain") = py::list(),
          py::arg("path_length") = py::list(),
          py::arg("path_polarization") = py::list(),
          py::arg("path_angles") = py::list(),
          py::arg("path_fbs_pos") = py::list(),
          py::arg("path_lbs_pos") = py::list(),
          py::arg("no_interact") = py::list(),
          py::arg("interact_coord") = py::list(),
          py::arg("rx_orientation") = py::array_t<double>(),
          py::arg("tx_orientation") = py::array_t<double>());

    m.def("hdf5_write_dset", &hdf5_write_dset, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::arg("name"),
          py::arg("data") = py::none());
}
