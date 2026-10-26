// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_conf_file_read.cpp"
#include "qpy_conf_file_write.cpp"
#include "qpy_scenpar_validate.cpp"
#include "qpy_conf_file_list.cpp"

void quadriga_lib_gbsm(py::module_ &m)
{
    m.def("conf_file_read", &conf_file_read,
          py::arg("fn"),
          py::arg("validate") = true);

    m.def("conf_file_write", &conf_file_write,
          py::arg("fn"),
          py::arg("scenpar"),
          py::arg("plpar") = py::none(),
          py::arg("short_names") = py::none(),
          py::arg("ref") = py::none(),
          py::arg("ref_sec") = py::none(),
          py::arg("valid_range") = py::none(),
          py::arg("write_defaults") = false);

    m.def("scenpar_validate", &scenpar_validate,
          py::arg("scenpar_in") = py::none(),
          py::arg("validate") = true);

    m.def("conf_file_list", &conf_file_list,
          py::arg("dirs"),
          py::arg("parse_shortnames") = true);
}