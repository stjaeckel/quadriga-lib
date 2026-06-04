// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_arrayant_calc_beamwidth.cpp"
#include "qpy_arrayant_calc_directivity.cpp"
#include "qpy_arrayant_combine_pattern.cpp"
#include "qpy_arrayant_concat.cpp"
#include "qpy_arrayant_copy_element.cpp"
#include "qpy_arrayant_export_obj_file.cpp"
#include "qpy_arrayant_generate.cpp"
#include "qpy_arrayant_interpolate.cpp"
#include "qpy_arrayant_qdant_read.cpp"
#include "qpy_arrayant_qdant_write.cpp"
#include "qpy_arrayant_rotate_pattern.cpp"
#include "qpy_generate_speaker.cpp"
#include "qpy_get_channels_planar.cpp"
#include "qpy_get_channels_spherical.cpp"
#include "qpy_get_channels_multifreq.cpp"

void quadriga_lib_arrayant(py::module_ &m)
{
    m.def("calc_beamwidth", &arrayant_calc_beamwidth,
          py::arg("arrayant"),
          py::arg("element") = py::none(),
          py::arg("threshold_dB") = 3.0);

    m.def("calc_directivity", &arrayant_calc_directivity,
          py::arg("arrayant"),
          py::arg("element") = py::none());

    m.def("combine_pattern", &arrayant_combine_pattern,
          py::arg("arrayant"),
          py::arg("freq") = py::none(),
          py::arg("azimuth_grid") = py::none(),
          py::arg("elevation_grid") = py::none());

    m.def("concat", &arrayant_concat,
          py::arg("arrayant1"),
          py::arg("arrayant2"));

    m.def("copy_element", &arrayant_copy_element,
          py::arg("arrayant"),
          py::arg("source_element"),
          py::arg("dest_element"));

    m.def("export_obj_file", &arrayant_export_obj_file,
          py::arg("fn"),
          py::arg("arrayant"),
          py::arg("directivity_range") = 30.0,
          py::arg("colormap") = "jet",
          py::arg("object_radius") = 1.0,
          py::arg("icosphere_n_div") = 4,
          py::arg("element") = py::none(),
          py::arg("freq_ind") = 0);

    m.def("generate", &arrayant_generate,
          py::arg("type"),
          py::arg("res") = 0.0,
          py::arg("freq") = 299792458.0,
          py::arg("az_3dB") = 0.0,
          py::arg("el_3dB") = 0.0,
          py::arg("rear_gain_lin") = 0.0,
          py::arg("M") = 1,
          py::arg("N") = 1,
          py::arg("pol") = 1,
          py::arg("tilt") = 0.0,
          py::arg("spacing") = 0.5,
          py::arg("Mg") = 1,
          py::arg("Ng") = 1,
          py::arg("dgv") = 0.5,
          py::arg("dgh") = 0.5,
          py::arg("beam_az") = py::none(),
          py::arg("beam_el") = py::none(),
          py::arg("beam_weight") = py::none(),
          py::arg("separate_beams") = false,
          py::arg("apply_weights") = false,
          py::arg("pattern") = py::dict());

    m.def("generate_speaker", &arrayant_generate_speaker,
          py::arg("driver_type") = "piston",
          py::arg("radius") = 0.05,
          py::arg("lower_cutoff") = 80.0,
          py::arg("upper_cutoff") = 12000.0,
          py::arg("lower_rolloff_slope") = 12.0,
          py::arg("upper_rolloff_slope") = 12.0,
          py::arg("sensitivity") = 85.0,
          py::arg("radiation_type") = "hemisphere",
          py::arg("hor_coverage") = 0.0,
          py::arg("ver_coverage") = 0.0,
          py::arg("horn_control_freq") = 0.0,
          py::arg("baffle_width") = 0.15,
          py::arg("baffle_height") = 0.25,
          py::arg("frequencies") = py::none(),
          py::arg("angular_resolution") = 5.0);

    m.def("interpolate", &arrayant_interpolate,
          py::arg("arrayant"),
          py::arg("azimuth") = py::none(),
          py::arg("elevation") = py::none(),
          py::arg("element") = py::none(),
          py::arg("orientation") = py::none(),
          py::arg("element_pos") = py::none(),
          py::arg("frequency") = py::none(),
          py::arg("complex") = false,
          py::arg("dist") = false,
          py::arg("local_angles") = false,
          py::arg("fast_access") = false);

    m.def("qdant_read", &arrayant_qdant_read, py::arg("fn"), py::arg("id") = 1);

    m.def("qdant_write", &arrayant_qdant_write,
          py::arg("fn"),
          py::arg("arrayant"),
          py::arg("id") = 0,
          py::arg("layout") = py::none());

    m.def("rotate_pattern", &arrayant_rotate_pattern,
          py::arg("arrayant"),
          py::arg("x_deg") = 0.0,
          py::arg("y_deg") = 0.0,
          py::arg("z_deg") = 0.0,
          py::arg("usage") = 0,
          py::arg("element") = py::none());

    m.def("get_channels_planar", &get_channels_planar,
          py::arg("ant_tx") = py::dict(),
          py::arg("ant_rx") = py::dict(),
          py::arg("aod") = py::array_t<double>(),
          py::arg("eod") = py::array_t<double>(),
          py::arg("aoa") = py::array_t<double>(),
          py::arg("eoa") = py::array_t<double>(),
          py::arg("path_gain") = py::array_t<double>(),
          py::arg("path_length") = py::array_t<double>(),
          py::arg("M") = py::array_t<double>(),
          py::arg("tx_pos") = py::array_t<double>(),
          py::arg("tx_orientation") = py::array_t<double>(),
          py::arg("rx_pos") = py::array_t<double>(),
          py::arg("rx_orientation") = py::array_t<double>(),
          py::arg("center_freq") = 0.0,
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false);

    m.def("get_channels_spherical", &get_channels_spherical,
          py::arg("ant_tx") = py::dict(),
          py::arg("ant_rx") = py::dict(),
          py::arg("fbs_pos") = py::array_t<double>(),
          py::arg("lbs_pos") = py::array_t<double>(),
          py::arg("path_gain") = py::array_t<double>(),
          py::arg("path_length") = py::array_t<double>(),
          py::arg("M") = py::array_t<double>(),
          py::arg("tx_pos") = py::array_t<double>(),
          py::arg("tx_orientation") = py::array_t<double>(),
          py::arg("rx_pos") = py::array_t<double>(),
          py::arg("rx_orientation") = py::array_t<double>(),
          py::arg("center_freq") = 0.0,
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false,
          py::arg("angles") = false);

    m.def("get_channels_multifreq", &get_channels_multifreq,
          py::arg("ant_tx"),
          py::arg("ant_rx"),
          py::arg("fbs_pos"),
          py::arg("lbs_pos"),
          py::arg("path_gain"),
          py::arg("path_length"),
          py::arg("M"),
          py::arg("tx_pos"),
          py::arg("tx_orientation"),
          py::arg("rx_pos"),
          py::arg("rx_orientation"),
          py::arg("freq_in"),
          py::arg("freq_out"),
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false,
          py::arg("propagation_speed") = 299792458.0,
          py::arg("complex") = false);
}