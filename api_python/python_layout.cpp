// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <limits>

namespace py = pybind11;

#include "qpy_layout_types.cpp"
#include "qpy_layout_csv.cpp"
#include "qpy_layout_generate.cpp"
#include "qpy_layout_mobility.cpp"
#include "qpy_layout_nodes.cpp"
#include "qpy_layout_pipeline.cpp"

// Mandatory arrays are declared with .none(false): without it pybind11 converts None to a 0-dimensional
// array holding NaN.
void quadriga_lib_layout(py::module_ &m)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    m.def("csv_read", &layout_csv_read,
          py::arg("fn_tx"),
          py::arg("fn_rx"),
          py::kw_only(),
          py::arg("center_freq") = py::none(),
          py::arg("eval_antennas") = true);

    m.def("csv_write", &layout_csv_write,
          py::arg("fn_tx"),
          py::arg("fn_rx"),
          py::arg("layout"),
          py::kw_only(),
          py::arg("compact") = true);

    m.def("generate_layout", &generate_layout,
          py::arg("type"),
          py::arg("n_ut") = 0,
          py::kw_only(),
          py::arg("spacing") = py::none(),
          py::arg("grid") = py::none(),
          py::arg("area") = py::none(),
          py::arg("bs") = py::none(),
          py::arg("ut") = py::none(),
          py::arg("indoor") = py::none(),
          py::arg("options") = py::none(),
          py::arg("per_cell") = true,
          py::arg("seed") = -1);

    m.def("drop_terminals", &drop_terminals,
          py::arg("layout_in"),
          py::arg("n_ut"),
          py::kw_only(),
          py::arg("bounds") = py::none(),
          py::arg("ut") = py::none(),
          py::arg("indoor") = py::none(),
          py::arg("options") = py::none(),
          py::arg("per_cell") = true,
          py::arg("seed") = -1);

    m.def("generate_track", &generate_track,
          py::arg("nodes_in"),
          py::arg("i_node").none(false),
          py::arg("type"),
          py::kw_only(),
          py::arg("length") = 0.0,
          py::arg("speed") = nan,
          py::arg("direction") = py::none(),
          py::arg("spacing") = nan,
          py::arg("options") = py::none(),
          py::arg("seed") = -1);

    m.def("sample_track", &sample_track,
          py::arg("nodes"),
          py::arg("i_node"),
          py::arg("times").none(false));

    m.def("set_orientation", &set_orientation,
          py::arg("nodes_in"),
          py::kw_only(),
          py::arg("orientation") = py::none(),
          py::arg("i_node") = py::none(),
          py::arg("force") = false);

    m.def("set_antenna", &set_antenna,
          py::arg("layout_in"),
          py::arg("type"),
          py::arg("args") = py::none(),
          py::kw_only(),
          py::arg("rx") = false,
          py::arg("i_node") = py::none(),
          py::arg("center_freq") = nan,
          py::arg("evaluate") = true);

    m.def("set_antenna_array", &set_antenna_array,
          py::arg("layout_in"),
          py::arg("array"),
          py::kw_only(),
          py::arg("rx") = false,
          py::arg("i_node") = py::none(),
          py::arg("center_freq") = nan);

    m.def("eval_antennas", &eval_antennas,
          py::arg("layout_in"),
          py::kw_only(),
          py::arg("center_freq") = py::none());

    m.def("get_channels_layout", &get_channels_layout,
          py::arg("layout"),
          py::kw_only(),
          py::arg("times") = py::none(),
          py::arg("output") = "taps",
          py::arg("method") = "",
          py::arg("mobility") = "auto",
          py::arg("overlap") = 0.5,
          py::arg("seed") = -1,
          py::arg("config_dirs") = py::none(),
          py::arg("stack") = false);

    m.def("init_link_states", &init_link_states,
          py::arg("layout_in"),
          py::kw_only(),
          py::arg("method") = "38.901",
          py::arg("o2i_low") = py::none(),
          py::arg("pol_variability") = false,
          py::arg("anchor_freq") = 2.0e9,
          py::arg("sos_preset") = "",
          py::arg("seed") = -1,
          py::arg("config_dirs") = py::none());

    m.def("generate_rays", &generate_rays,
          py::arg("layout"),
          py::kw_only(),
          py::arg("i_link") = py::none(),
          py::arg("scatterers") = false);

    m.def("get_gbsm_channels", &get_gbsm_channels,
          py::arg("layout"),
          py::arg("rays").none(false),
          py::kw_only(),
          py::arg("i_link") = py::none(),
          py::arg("times") = py::none(),
          py::arg("output") = "taps",
          py::arg("mobility") = "auto",
          py::arg("oxygen") = false,
          py::arg("use_absolute_delays") = false,
          py::arg("overlap") = 0.5,
          py::arg("stack") = false);

    m.def("attach_terminals", &attach_terminals,
          py::arg("layout"),
          py::kw_only(),
          py::arg("rule") = "large_scale",
          py::arg("rays") = py::none(),
          py::arg("i_freq") = 0,
          py::arg("tx_port") = 0);

    m.def("export_rays_qrt", &export_rays_qrt,
          py::arg("fn"),
          py::arg("layout"),
          py::kw_only(),
          py::arg("rays") = py::none());
}
