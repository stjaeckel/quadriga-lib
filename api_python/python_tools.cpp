// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_fast_cart2geo.cpp"
#include "qpy_write_png.cpp"
#include "qpy_calc_delay_spread.cpp"
#include "qpy_acdf.cpp"
#include "qpy_calc_angular_spread.cpp"
#include "qpy_calc_rician_k_factor.cpp"
#include "qpy_calc_cross_polarization_ratio.cpp"
#include "qpy_fast_geo2cart.cpp"

void quadriga_lib_tools(py::module_ &m)
{
    m.def("acdf", &acdf,
          py::arg("data").none(false),
          py::arg("bins") = py::none(),
          py::arg("n_bins") = 201);

    m.def("fast_cart2geo", &fast_cart2geo,
          py::arg("cart").none(false),
          py::arg("y") = py::none(),
          py::arg("z") = py::none(),
          py::arg("combine") = true,
          py::arg("use_kernel") = 1);

    m.def("write_png", &write_png,
          py::arg("fn"),
          py::arg("data").none(false),
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
          py::arg("quantize_deg") = 0.0);

    m.def("calc_delay_spread", &calc_delay_spread,
          py::arg("delays"),
          py::arg("powers"),
          py::arg("threshold_dB") = 100.0,
          py::arg("granularity") = 0.0);

    m.def("calc_rician_k_factor", &calc_rician_k_factor,
          py::arg("powers"),
          py::arg("path_length"),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("window_size") = 0.01);

    m.def("calc_cross_polarization_ratio", &calc_cross_polarization_ratio,
          py::arg("powers"),
          py::arg("M"),
          py::arg("path_length"),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("include_los") = false,
          py::arg("window_size") = 0.01);

    m.def("fast_geo2cart", &fast_geo2cart,
          py::arg("az").none(false),
          py::arg("el") = py::none(),
          py::arg("len") = py::none(),
          py::arg("combine") = true,
          py::arg("use_kernel") = 1);
}