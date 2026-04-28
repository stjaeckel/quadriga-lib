// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_cart2geo.cpp"
#include "qpy_write_png.cpp"
#include "qpy_calc_delay_spread.cpp"
#include "qpy_acdf.cpp"
#include "qpy_calc_angular_spread.cpp"
#include "qpy_calc_rician_k_factor.cpp"
#include "qpy_calc_cross_polarization_ratio.cpp"

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

    m.def("calc_angular_spread", &calc_angular_spread,
          py::arg("az"),
          py::arg("el"),
          py::arg("powers"),
          py::arg("wrapping") = false,
          py::arg("calc_bank_angle") = false,
          py::arg("quantize") = 0.0);

    m.def("calc_delay_spread", &calc_delay_spread,
          py::arg("delays"),
          py::arg("powers"),
          py::arg("threshold") = 100.0,
          py::arg("granularity") = 0.0);

    m.def("calc_rician_k_factor", &calc_rician_k_factor,
          py::arg("powers"),
          py::arg("path_length"),
          py::arg("tx_pos"),
          py::arg("rx_pos"),
          py::arg("window_size") = 0.01);

    m.def("calc_cross_polarization_ratio", &calc_cross_polarization_ratio,
          py::arg("powers"),
          py::arg("M"),
          py::arg("path_length"),
          py::arg("tx_pos"),
          py::arg("rx_pos"),
          py::arg("include_los") = false,
          py::arg("window_size") = 0.01);
}