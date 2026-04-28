// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Declare submodules
void quadriga_lib_arrayant(py::module_ &m);
void quadriga_lib_channel(py::module_ &m);
void quadriga_lib_RTtools(py::module_ &m);
void quadriga_lib_tools(py::module_ &m);

// Include parts
#include "qpy_components.cpp"
#include "qpy_version.cpp"

PYBIND11_MODULE(quadriga_lib, m)
{
    py::module_ arrayant = m.def_submodule("arrayant", "Array antenna functions");
    quadriga_lib_arrayant(arrayant);

    py::module_ channel = m.def_submodule("channel", "Channel functions");
    quadriga_lib_channel(channel);

    py::module_ tools = m.def_submodule("tools", "Miscellaneous / Tools");
    quadriga_lib_tools(tools);

    py::module_ RTtools = m.def_submodule("RTtools", "Site-Specific Simulation Tools");
    quadriga_lib_RTtools(RTtools);
  
    m.def("components", &components);

    m.def("version", &version);
}
