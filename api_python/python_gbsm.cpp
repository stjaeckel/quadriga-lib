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
#include "qpy_gbsm_init_phases.cpp"
#include "qpy_gbsm_draw.cpp"
#include "qpy_path_loss.cpp"
#include "qpy_lsp_generate.cpp"
#include "qpy_los_probability.cpp"
#include "qpy_o2i_loss.cpp"
#include "qpy_ssf_clusters.cpp"
#include "qpy_ssf_rays.cpp"
#include "qpy_ssf_scatterers.cpp"
#include "qpy_ssf_paths.cpp"
#include "qpy_gbsm_rays_from_paths.cpp"
#include "qpy_gbsm_single_link_channel.cpp"

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

    m.def("init_phases", &gbsm_init_phases,
          py::arg("scenpar"),
          py::arg("set_type"),
          py::arg("seed") = -1,
          py::arg("set_id") = -1,
          py::arg("preset") = "Comb256",
          py::arg("tx_pos") = py::none(),
          py::arg("rx_pos") = py::none(),
          py::arg("dual_mobility") = false,
          py::arg("remove_bias") = true,
          py::arg("horizontal") = true);

    m.def("draw", &gbsm_draw,
          py::arg("scenpar"),
          py::arg("set_type"),
          py::arg("phase").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("set_id") = -1,
          py::arg("preset") = "Comb256",
          py::arg("floor_id") = py::none(),
          py::arg("center_freq") = py::none(),
          py::arg("dual_mobility") = false,
          py::arg("horizontal") = true);

    m.def("path_loss", &path_loss,
          py::arg("plpar"),
          py::arg("scenpar"),
          py::arg("center_freq").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("d2d_in") = py::none(),
          py::arg("rnd_he") = py::none());

    m.def("lsp_generate", &lsp_generate,
          py::arg("scenpar"),
          py::arg("center_freq").none(false),
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
          py::arg("center_freq") = py::none());

    m.def("o2i_loss", &o2i_loss,
          py::arg("center_freq").none(false),
          py::arg("rnd").none(false),
          py::arg("models"),
          py::arg("weights").none(false),
          py::arg("d2d_in_max") = 25.0);

    m.def("ssf_clusters", &ssf_clusters,
          py::arg("scenpar"),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("lsp").none(false),
          py::arg("rnd").none(false),
          py::arg("method") = "38.901",
          py::arg("rnd_freq") = py::none(),
          py::arg("lsp_anchor") = py::none(),
          py::arg("eod_offset_deg") = py::none(),
          py::arg("abs_toa") = py::none(),
          py::arg("gr_eps") = py::none(),
          py::arg("o2i") = py::none(),
          py::arg("n_cluster") = py::none());
    m.def("ssf_rays", &ssf_rays,
          py::arg("scenpar"),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("center_freq").none(false),
          py::arg("lsp").none(false),
          py::arg("lsp_mu").none(false),
          py::arg("lsp_sigma").none(false),
          py::arg("delay").none(false),
          py::arg("power").none(false),
          py::arg("aod").none(false),
          py::arg("eod").none(false),
          py::arg("aoa").none(false),
          py::arg("eoa").none(false),
          py::arg("rnd_ray").none(false),
          py::arg("method") = "38.901",
          py::arg("i_freq") = 0,
          py::arg("angle_model") = "additive",
          py::arg("pol_model") = "38.901",
          py::arg("gr_eps") = py::none(),
          py::arg("pol_variability") = false,
          py::arg("random_phase") = true);
    m.def("ssf_scatterers", &ssf_scatterers,
          py::arg("rays"),
          py::arg("d_min") = 2.0);
    m.def("ssf_paths", &ssf_paths,
          py::arg("rays"),
          py::arg("link_gain_dB").none(false),
          py::arg("mobility") = "auto",
          py::arg("tx_pos") = py::none(),
          py::arg("rx_pos") = py::none(),
          py::arg("kf_ratio") = py::none(),
          py::arg("oxygen") = false);
    m.def("rays_from_paths", &gbsm_rays_from_paths,
          py::arg("center_freq").none(false),
          py::arg("tx_pos").none(false),
          py::arg("rx_pos").none(false),
          py::arg("fbs_pos").none(false),
          py::arg("lbs_pos").none(false),
          py::arg("path_gain").none(false),
          py::arg("path_length").none(false),
          py::arg("M").none(false),
          py::arg("method") = "quadriga",
          py::arg("path_type") = py::none(),
          py::arg("tap") = py::none(),
          py::arg("gr_eps") = py::none());
    m.def("single_link_channel", &gbsm_single_link_channel,
          py::arg("tx_array"),
          py::arg("rx_array"),
          py::arg("rays"),
          py::arg("link_gain_dB").none(false),
          py::arg("mobility") = "auto",
          py::arg("output") = "taps",
          py::arg("i_freq") = 0,
          py::arg("tx_pos") = py::none(),
          py::arg("rx_pos") = py::none(),
          py::arg("tx_orientation") = py::none(),
          py::arg("rx_orientation") = py::none(),
          py::arg("kf_ratio") = py::none(),
          py::arg("oxygen") = false,
          py::arg("use_absolute_delays") = false,
          py::arg("stack") = false);
}
