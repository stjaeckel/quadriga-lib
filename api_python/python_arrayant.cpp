// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "qpy_arrayant_ops.cpp"
#include "qpy_arrayant_io.cpp"
#include "qpy_arrayant_generate.cpp"
#include "qpy_arrayant_interpolate.cpp"

void quadriga_lib_arrayant(py::module_ &m)
{
    m.def("calc_beamwidth", &arrayant_calc_beamwidth,
          py::arg("arrayant"),
          py::kw_only(),
          py::arg("i_element") = py::none(),
          py::arg("threshold_dB") = 3.0);

    m.def("calc_directivity", &arrayant_calc_directivity,
          py::arg("arrayant"),
          py::kw_only(),
          py::arg("i_element") = py::none());

    m.def("combine_pattern", &arrayant_combine_pattern,
          py::arg("arrayant"),
          py::kw_only(),
          py::arg("freq_new") = py::none(),
          py::arg("azimuth_grid") = py::none(),
          py::arg("elevation_grid") = py::none(),
          py::arg("propagation_speed") = 299792458.0);

    m.def("concat", &arrayant_concat,
          py::arg("arrayant1"),
          py::arg("arrayant2"));

    m.def("copy_element", &arrayant_copy_element,
          py::arg("arrayant"),
          py::arg("i_source").none(false),
          py::arg("i_target").none(false));

    m.def("export_obj_file", &arrayant_export_obj_file,
          py::arg("fn"),
          py::arg("arrayant"),
          py::kw_only(),
          py::arg("directivity_range_dB") = 30.0,
          py::arg("colormap") = "jet",
          py::arg("object_radius") = 1.0,
          py::arg("icosphere_n_div") = 4,
          py::arg("i_element") = py::none(),
          py::arg("i_freq") = 0);

    m.def("generate", &arrayant_generate,
          py::arg("type"),
          py::arg("res") = NAN,
          py::kw_only(),
          py::arg("center_freq") = 299792458.0,
          py::arg("panel") = py::none(),
          py::arg("spacing") = py::none(),
          py::arg("ports") = py::none(),
          py::arg("pol") = py::none(),
          py::arg("steer") = py::none(),
          py::arg("element") = py::none(),
          py::arg("pattern") = py::none(),
          py::arg("beams") = py::none(),
          py::arg("separate_beams") = false,
          py::arg("apply_weights") = false);

    m.def("generate_speaker", &generate_speaker,
          py::arg("driver_type") = "piston",
          py::kw_only(),
          py::arg("radius") = 0.05,
          py::arg("lower_cutoff") = 80.0,
          py::arg("upper_cutoff") = 12000.0,
          py::arg("lower_rolloff_slope_dB") = 12.0,
          py::arg("upper_rolloff_slope_dB") = 12.0,
          py::arg("sensitivity_dB") = 85.0,
          py::arg("radiation_type") = "hemisphere",
          py::arg("hor_coverage_deg") = NAN,
          py::arg("ver_coverage_deg") = NAN,
          py::arg("horn_control_freq") = NAN,
          py::arg("baffle_width") = 0.15,
          py::arg("baffle_height") = 0.25,
          py::arg("center_freq") = py::none(),
          py::arg("angular_resolution_deg") = 5.0);

    m.def("interpolate", &arrayant_interpolate,
          py::arg("arrayant").none(false),
          py::arg("azimuth").none(false),
          py::arg("elevation").none(false),
          py::kw_only(),
          py::arg("i_element") = py::none(),
          py::arg("orientation") = py::none(),
          py::arg("element_pos") = py::none(),
          py::arg("freq_new") = py::none(),
          py::arg("dist") = false,
          py::arg("local_angles") = false,
          py::arg("fast_access") = false,
          py::arg("use_kernel") = 0);

    m.def("qdant_read", &qdant_read, py::arg("fn"), py::arg("id") = py::none());

    m.def("qdant_write", &qdant_write,
          py::arg("fn"),
          py::arg("arrayant"),
          py::arg("id") = 0,
          py::kw_only(),
          py::arg("layout") = py::none());

    m.def("rotate_pattern", &arrayant_rotate_pattern,
          py::arg("arrayant"),
          py::arg("orientation").none(false),
          py::kw_only(),
          py::arg("usage") = 0,
          py::arg("i_element") = py::none());
}
