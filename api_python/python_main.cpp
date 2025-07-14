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

// Declare submodules
void quadriga_lib_arrayant(py::module_ &m);
void quadriga_lib_channel(py::module_ &m);
void quadriga_lib_RTtools(py::module_ &m);

// Include parts
#include "qpy_components.cpp"
#include "qpy_version.cpp"

PYBIND11_MODULE(quadriga_lib, m)
{
    py::module_ arrayant = m.def_submodule("arrayant", "Array antenna functions");
    quadriga_lib_arrayant(arrayant);

    py::module_ channel = m.def_submodule("channel", "Channel functions");
    quadriga_lib_channel(channel);

    py::module_ RTtools = m.def_submodule("RTtools", "Site-Specific Simulation Tools");
    quadriga_lib_RTtools(RTtools);
  
    m.def("components", &components);

    m.def("version", &version);
}
