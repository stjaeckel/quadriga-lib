// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "qpy_baseband_freq_response.cpp"
#include "qpy_channel_export_obj_file.cpp"
#include "qpy_hdf5.cpp"
#include "qpy_get_channels.cpp"
#include "qpy_qrt_file.cpp"
#include "qpy_quantize_delays.cpp"

void quadriga_lib_channel(py::module_ &m)
{
    m.def("baseband_freq_response", &baseband_freq_response,
          py::arg("coeff").none(false),
          py::arg("delay").none(false),
          py::arg("pilot_grid").none(false),
          py::arg("bandwidth"),
          py::kw_only(),
          py::arg("i_snap") = py::none(),
          py::arg("use_kernel") = 0);

    m.def("baseband_freq_response_multi", &baseband_freq_response_multi,
          py::arg("coeff").none(false),
          py::arg("delay").none(false),
          py::arg("freq").none(false),
          py::arg("freq_new").none(false),
          py::kw_only(),
          py::arg("remove_delay_phase") = true,
          py::arg("i_snap") = py::none(),
          py::arg("use_kernel") = 0);

    m.def("export_obj_file", &channel_export_obj_file,
          py::arg("fn"),
          py::arg("chan").none(false),
          py::kw_only(),
          py::arg("max_paths") = 0,
          py::arg("gain_max_dB") = -60.0,
          py::arg("gain_min_dB") = -140.0,
          py::arg("colormap") = "jet",
          py::arg("i_snap") = py::none(),
          py::arg("radius_max") = 0.05,
          py::arg("radius_min") = 0.01,
          py::arg("n_edges") = 5);

    m.def("hdf5_create_file", &hdf5_create_file, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_read_channel", &hdf5_read_channel,
          py::arg("fn"),
          py::arg("ix") = py::none(),
          py::arg("iy") = py::none(),
          py::arg("iz") = py::none(),
          py::arg("iw") = py::none(),
          py::kw_only(),
          py::arg("i_snap") = py::none(),
          py::arg("stack") = false);

    m.def("hdf5_read_dset_names", &hdf5_read_dset_names, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::kw_only(),
          py::arg("prefix") = "par_");

    m.def("hdf5_read_dset", &hdf5_read_dset,
          py::arg("fn"),
          py::arg("name"),
          py::arg("ix") = 0,
          py::arg("iy") = 0,
          py::arg("iz") = 0,
          py::arg("iw") = 0,
          py::kw_only(),
          py::arg("prefix") = "par_");

    m.def("hdf5_read_layout", &hdf5_read_layout, py::arg("fn"));

    m.def("hdf5_reshape_layout", &hdf5_reshape_layout, py::arg("fn"),
          py::arg("nx"), py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_write_channel", &hdf5_write_channel,
          py::arg("fn"),
          py::arg("chan").none(false),
          py::arg("par") = py::none(),
          py::arg("ix") = py::none(),
          py::arg("iy") = py::none(),
          py::arg("iz") = py::none(),
          py::arg("iw") = py::none());

    m.def("hdf5_write_dset", &hdf5_write_dset, py::arg("fn"),
          py::arg("name"),
          py::arg("data").none(false),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::kw_only(),
          py::arg("prefix") = "par_");

    m.def("ieee_indoor", &get_channels_ieee_indoor,
          py::arg("ap_array"),
          py::arg("sta_array"),
          py::arg("channel_type"),
          py::arg("center_freq") = 5.25e9,
          py::kw_only(),
          py::arg("tap_spacing") = 10.0e-9,
          py::arg("n_users") = 1,
          py::arg("observation_time") = 0.0,
          py::arg("update_interval") = 1.0e-3,
          py::arg("speed_station_kmh") = 0.0,
          py::arg("speed_env_kmh") = 1.2,
          py::arg("dist") = py::none(),
          py::arg("n_floors") = py::none(),
          py::arg("uplink") = false,
          py::arg("offset_angles_deg") = py::none(),
          py::arg("n_subpath") = 20,
          py::arg("doppler_effect") = 50.0,
          py::arg("seed") = -1,
          py::arg("kf") = NAN,
          py::arg("xpr_nlos") = NAN,
          py::arg("sf_std_los_dB") = NAN,
          py::arg("sf_std_nlos_dB") = NAN,
          py::arg("dist_breakpoint") = NAN,
          py::arg("n_walls") = py::none(),
          py::arg("wall_loss_dB") = 5.0,
          py::arg("stack") = false);

    m.def("irs", &get_channels_irs,
          py::arg("tx_array").none(false),
          py::arg("rx_array").none(false),
          py::arg("irs_array").none(false),
          py::arg("tx_pos").none(false),
          py::arg("tx_orientation").none(false),
          py::arg("rx_pos").none(false),
          py::arg("rx_orientation").none(false),
          py::arg("irs_pos").none(false),
          py::arg("irs_orientation").none(false),
          py::arg("fbs_pos_1").none(false),
          py::arg("lbs_pos_1").none(false),
          py::arg("path_gain_1").none(false),
          py::arg("path_length_1").none(false),
          py::arg("polmat_1").none(false),
          py::arg("fbs_pos_2").none(false),
          py::arg("lbs_pos_2").none(false),
          py::arg("path_gain_2").none(false),
          py::arg("path_length_2").none(false),
          py::arg("polmat_2").none(false),
          py::arg("center_freq") = 0.0,
          py::kw_only(),
          py::arg("i_irs") = 0,
          py::arg("threshold_dB") = -140.0,
          py::arg("use_absolute_delays") = false,
          py::arg("angles") = false,
          py::arg("irs_array_2") = py::none(),
          py::arg("active_path") = py::none(),
          py::arg("use_kernel") = 0);

    m.def("planar", &get_channels_planar,
          py::arg("tx_array"),
          py::arg("rx_array"),
          py::arg("tx_pos").none(false),
          py::arg("tx_orientation").none(false),
          py::arg("rx_pos").none(false),
          py::arg("rx_orientation").none(false),
          py::arg("aod").none(false),
          py::arg("eod").none(false),
          py::arg("aoa").none(false),
          py::arg("eoa").none(false),
          py::arg("path_gain").none(false),
          py::arg("path_length").none(false),
          py::arg("polmat").none(false),
          py::arg("center_freq") = py::none(),
          py::kw_only(),
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false,
          py::arg("use_kernel") = 0,
          py::arg("i_freq") = 0);

    m.def("spherical", &get_channels_spherical,
          py::arg("tx_array"),
          py::arg("rx_array"),
          py::arg("tx_pos").none(false),
          py::arg("tx_orientation").none(false),
          py::arg("rx_pos").none(false),
          py::arg("rx_orientation").none(false),
          py::arg("fbs_pos").none(false),
          py::arg("lbs_pos").none(false),
          py::arg("path_gain").none(false),
          py::arg("path_length").none(false),
          py::arg("polmat").none(false),
          py::arg("center_freq") = py::none(),
          py::kw_only(),
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false,
          py::arg("angles") = false,
          py::arg("use_kernel") = 0,
          py::arg("i_freq") = 0);

    m.def("multifreq", &get_channels_multifreq,
          py::arg("tx_array"),
          py::arg("rx_array"),
          py::arg("tx_pos").none(false),
          py::arg("tx_orientation").none(false),
          py::arg("rx_pos").none(false),
          py::arg("rx_orientation").none(false),
          py::arg("fbs_pos").none(false),
          py::arg("lbs_pos").none(false),
          py::arg("path_gain").none(false),
          py::arg("path_length").none(false),
          py::arg("polmat").none(false),
          py::arg("freq").none(false),
          py::arg("freq_new").none(false),
          py::kw_only(),
          py::arg("use_absolute_delays") = false,
          py::arg("add_fake_los_path") = false,
          py::arg("propagation_speed") = 299792458.0,
          py::arg("use_kernel") = 0);

    m.def("quantize_delays", &quantize_delays,
          py::arg("coeff").none(false),
          py::arg("delay").none(false),
          py::kw_only(),
          py::arg("tap_spacing") = 5e-9,
          py::arg("max_taps") = 48,
          py::arg("power_exponent") = 1.0,
          py::arg("fix_taps") = 0,
          py::arg("stack") = false);

    m.def("qrt_file_parse", &qrt_file_parse, py::arg("fn"));

    m.def("qrt_file_read", &qrt_file_read,
          py::arg("fn"),
          py::arg("i_cir") = py::none(),
          py::arg("i_orig") = 0,
          py::kw_only(),
          py::arg("uplink") = false,
          py::arg("normalize_polmat") = 1);
}
