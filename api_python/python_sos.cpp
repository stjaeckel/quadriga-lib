// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_sos_preset.cpp"
#include "qpy_sos_init_phases.cpp"
#include "qpy_sos_acfi.cpp"
#include "qpy_sos_val.cpp"
#include "qpy_sos_map.cpp"
#include "qpy_sos_randn.cpp"
#include "qpy_sos_rand.cpp"
#include "qpy_sos_randi.cpp"
#include "qpy_sos_remove_bias.cpp"
#include "qpy_sos_acf_2d.cpp"
#include "qpy_sos_acf_approx.cpp"
#include "qpy_sos_calc_mse.cpp"
#include "qpy_sos_acf_estimate.cpp"
#include "qpy_sos_generate.cpp"

// Mandatory arrays are declared with .none(false): without it pybind11 converts None to a 0-dimensional
// array holding NaN, which the library would evaluate as one position at NaN.
void quadriga_lib_sos(py::module_ &m)
{
    m.def("preset", &sos_preset,
          py::arg("name"),
          py::arg("dist_decorr") = 10.0);

    m.def("init_phases", &sos_init_phases,
          py::arg("n_coeff"),
          py::arg("n_gen") = 1,
          py::arg("seed") = -1);

    m.def("acfi", &sos_acfi,
          py::arg("acf").none(false),
          py::arg("dist").none(false),
          py::arg("d").none(false));

    m.def("val", &sos_val,
          py::arg("freq").none(false),
          py::arg("phase").none(false),
          py::arg("pos").none(false),
          py::arg("pos_b") = py::none(),
          py::arg("phase_b") = py::none(),
          py::arg("acf") = py::none(),
          py::arg("dist") = py::none(),
          py::arg("freq_scale_b") = 1.0,
          py::arg("uniform") = false,
          py::arg("use_kernel") = 0);

    m.def("map", &sos_map,
          py::arg("freq").none(false),
          py::arg("phase").none(false),
          py::arg("x").none(false),
          py::arg("y") = py::none(),
          py::arg("z") = py::none(),
          py::arg("uniform") = false,
          py::arg("use_kernel") = 0);

    m.def("randn", &sos_randn,
          py::arg("dist_decorr").none(false),
          py::arg("pos").none(false),
          py::arg("pos_b") = py::none(),
          py::arg("dist_decorr_b") = py::none(),
          py::arg("preset") = "Comb256",
          py::arg("seed") = -1);

    m.def("rand", &sos_rand,
          py::arg("dist_decorr").none(false),
          py::arg("pos").none(false),
          py::arg("pos_b") = py::none(),
          py::arg("dist_decorr_b") = py::none(),
          py::arg("preset") = "Comb256",
          py::arg("seed") = -1);

    m.def("randi", &sos_randi,
          py::arg("dist_decorr").none(false),
          py::arg("pos").none(false),
          py::arg("imax"),
          py::arg("pos_b") = py::none(),
          py::arg("dist_decorr_b") = py::none(),
          py::arg("preset") = "Comb256",
          py::arg("seed") = -1);

    m.def("remove_bias", &sos_remove_bias,
          py::arg("freq").none(false),
          py::arg("phase_in").none(false),
          py::arg("pos").none(false),
          py::arg("pos_b") = py::none(),
          py::arg("phase_b_in") = py::none(),
          py::arg("acf") = py::none(),
          py::arg("dist") = py::none(),
          py::arg("freq_scale_b") = 1.0);

    m.def("acf_2d", &sos_acf_2d,
          py::arg("acf").none(false),
          py::arg("dist").none(false));

    m.def("acf_approx", &sos_acf_approx,
          py::arg("freq").none(false),
          py::arg("dist").none(false));

    m.def("calc_mse", &sos_calc_mse,
          py::arg("freq").none(false),
          py::arg("acf").none(false),
          py::arg("dist").none(false),
          py::arg("n_dir") = 10);

    m.def("acf_estimate", &sos_acf_estimate,
          py::arg("freq").none(false),
          py::arg("phase").none(false),
          py::arg("bins").none(false),
          py::arg("n_pos"),
          py::arg("acf") = py::none(),
          py::arg("dist") = py::none(),
          py::arg("seed") = -1);

    m.def("generate", &sos_generate,
          py::arg("acf").none(false),
          py::arg("dist").none(false),
          py::arg("n_coeff"),
          py::arg("n_dim") = 3,
          py::arg("uniform_smp") = true,
          py::arg("n_dir") = 10,
          py::arg("random_init") = true,
          py::arg("freq_min") = 0.0,
          py::arg("freq_max") = 0.0,
          py::arg("seed") = -1);
}
