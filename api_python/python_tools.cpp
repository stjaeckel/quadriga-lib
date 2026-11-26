// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "qpy_fastmath.cpp"
#include "qpy_write_png.cpp"
#include "qpy_acdf.cpp"
#include "qpy_calc_rotation_matrix.cpp"
#include "qpy_interp.cpp"

void quadriga_lib_tools(py::module_ &m)
{
    m.def("acdf", &acdf,
          py::arg("data").none(false),
          py::arg("bins") = py::none(),
          py::kw_only(),
          py::arg("n_bins") = 201);

    m.def("fast_cart2geo", &fast_cart2geo,
          py::arg("x").none(false),
          py::arg("y").none(false),
          py::arg("z").none(false),
          py::kw_only(),
          py::arg("use_kernel") = 0);

    m.def("write_png", &write_png,
          py::arg("fn"),
          py::arg("data").none(false),
          py::kw_only(),
          py::arg("colormap") = "jet",
          py::arg("min_val") = NAN,
          py::arg("max_val") = NAN,
          py::arg("log_transform") = false);

    m.def("fast_geo2cart", &fast_geo2cart,
          py::arg("az").none(false),
          py::arg("el").none(false),
          py::arg("len") = py::none(),
          py::kw_only(),
          py::arg("use_kernel") = 0);

    m.def("calc_rotation_matrix", &calc_rotation_matrix,
          py::arg("orientation").none(false),
          py::kw_only(),
          py::arg("transposed") = false);

    m.def("fast_sincos", &fast_sincos,
          py::arg("x").none(false),
          py::kw_only(),
          py::arg("use_kernel") = 0);

    m.def("interp", &interp,
          py::arg("x").none(false),
          py::arg("y"),
          py::arg("input").none(false),
          py::arg("x_new").none(false),
          py::arg("y_new") = py::none());
}
