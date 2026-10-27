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
#include "qpy_lsf_init_phases.cpp"
#include "qpy_lsf_draw.cpp"
#include "qpy_path_loss.cpp"
#include "qpy_lsp_generate.cpp"
#include "qpy_los_probability.cpp"
#include "qpy_o2i_loss.cpp"

// Mandatory arrays are declared with .none(false): without it pybind11 converts None to a 0-dimensional
// array holding NaN.
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

    m.def("lsf_init_phases", &lsf_init_phases,
          py::arg("scenpar"),
          py::arg("seed") = -1,
          py::arg("tx_pos") = py::none(),
          py::arg("rx_pos") = py::none(),
          py::arg("dual_mobility") = false,
          py::arg("remove_bias") = true,
          py::arg("preset") = "Comb256",
          py::arg("horizontal") = true);

    m.def("lsf_draw", &lsf_draw,
          py::arg("scenpar"),
          py::arg("phase").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("floor_id") = py::none(),
          py::arg("dual_mobility") = false,
          py::arg("preset") = "Comb256",
          py::arg("horizontal") = true);

    m.def("path_loss", &path_loss,
          py::arg("plpar"),
          py::arg("scenpar"),
          py::arg("fc").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("d2d_in") = py::none(),
          py::arg("rnd_he") = py::none());

    m.def("lsp_generate", &lsp_generate,
          py::arg("scenpar"),
          py::arg("fc").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("rnd").none(false),
          py::arg("sf_sigma") = py::none(),
          py::arg("cholesky") = true,
          py::arg("clip_spreads") = true,
          py::arg("complex") = true);

    m.def("los_probability", &los_probability,
          py::arg("model"),
          py::arg("d2d_out").none(false),
          py::arg("h_bs").none(false),
          py::arg("h_ut").none(false),
          py::arg("d_clutter") = py::none(),
          py::arg("r_clutter") = py::none(),
          py::arg("h_clutter") = py::none(),
          py::arg("rnd_g") = py::none(),
          py::arg("fc") = py::none());

    m.def("o2i_loss", &o2i_loss,
          py::arg("fc").none(false),
          py::arg("rnd").none(false),
          py::arg("models"),
          py::arg("weights").none(false),
          py::arg("d2d_in_max") = 25.0);
}
