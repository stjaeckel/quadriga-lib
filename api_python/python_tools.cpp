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
#include "qpy_cart2geo.cpp"
#include "qpy_write_png.cpp"
#include "qpy_calc_delay_spread.cpp"
#include "qpy_acdf.cpp"
#include "qpy_calc_angular_spreads_sphere.cpp"

void quadriga_lib_tools(py::module_ &m)
{
    m.def("acdf", &acdf,
          py::arg("data"),
          py::arg("bins") = py::none(),
          py::arg("n_bins") = 201);

    m.def("cart2geo", &cart2geo, py::arg("cart"));

    m.def("write_png", &write_png,
          py::arg("fn"),
          py::arg("data") = py::array_t<double>(),
          py::arg("colormap") = "jet",
          py::arg("min_val") = NAN,
          py::arg("max_val") = NAN,
          py::arg("log_transform") = false);

    m.def("calc_angular_spreads_sphere", &calc_angular_spreads_sphere,
          py::arg("az"),
          py::arg("el"),
          py::arg("pw"),
          py::arg("calc_bank_angle") = true,
          py::arg("quantize") = 0.0);

    m.def("calc_delay_spread", &calc_delay_spread,
          py::arg("delays"),
          py::arg("powers"),
          py::arg("threshold") = 100.0,
          py::arg("granularity") = 0.0);
}