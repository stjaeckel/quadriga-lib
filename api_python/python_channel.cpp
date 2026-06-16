// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include parts
#include "qpy_baseband_freq_response.cpp"
#include "qpy_channel_export_obj_file.cpp"
#include "qpy_hdf5_create_file.cpp"
#include "qpy_hdf5_read_channel.cpp"
#include "qpy_hdf5_read_dset_names.cpp"
#include "qpy_hdf5_read_dset.cpp"
#include "qpy_hdf5_read_layout.cpp"
#include "qpy_hdf5_reshape_layout.cpp"
#include "qpy_hdf5_write_channel.cpp"
#include "qpy_hdf5_write_dset.cpp"
#include "qpy_get_channels_ieee_indoor.cpp"
#include "qpy_qrt_file_parse.cpp"
#include "qpy_qrt_file_read.cpp"
#include "qpy_quantize_delays.cpp"

void quadriga_lib_channel(py::module_ &m)
{
    m.def("baseband_freq_response", &baseband_freq_response,
          py::arg("coeff") = py::none(),
          py::arg("delay") = py::none(),
          py::arg("bandwidth") = 0.0,
          py::arg("carriers") = 128,
          py::arg("pilot_grid") = py::none(),
          py::arg("snap") = py::none(),
          py::arg("coeff_re") = py::none(),
          py::arg("coeff_im") = py::none(),
          py::arg("freq_in") = py::none(),
          py::arg("freq_out") = py::none(),
          py::arg("remove_delay_phase") = true);

    m.def("channel_export_obj_file", &channel_export_obj_file,
          py::arg("fn"),
          py::arg("max_no_paths") = 0,
          py::arg("gain_max") = -60.0,
          py::arg("gain_min") = -140.0,
          py::arg("colormap") = "jet",
          py::arg("radius_max") = 0.05,
          py::arg("radius_min") = 0.01,
          py::arg("n_edges") = 5,
          py::arg("rx_pos") = py::none(),
          py::arg("tx_pos") = py::none(),
          py::arg("no_interact") = py::list(),
          py::arg("interact_coord") = py::list(),
          py::arg("center_freq") = py::none(),
          py::arg("coeff") = py::none(),
          py::arg("coeff_re") = py::none(),
          py::arg("coeff_im") = py::none(),
          py::arg("i_snap") = py::array_t<arma::uword>());

    m.def("hdf5_create_file", &hdf5_create_file, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_read_channel", &hdf5_read_channel,
          py::arg("fn"),
          py::arg("ix") = py::none(),
          py::arg("iy") = py::none(),
          py::arg("iz") = py::none(),
          py::arg("iw") = py::none(),
          py::arg("snap") = py::none(),
          py::arg("stack") = false);

    m.def("hdf5_read_dset_names", &hdf5_read_dset_names, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0);

    m.def("hdf5_read_dset", &hdf5_read_dset,
          py::arg("fn"),
          py::arg("ix") = 0,
          py::arg("iy") = 0,
          py::arg("iz") = 0,
          py::arg("iw") = 0,
          py::arg("name"));

    m.def("hdf5_read_layout", &hdf5_read_layout, py::arg("fn"));

    m.def("hdf5_reshape_layout", &hdf5_reshape_layout, py::arg("fn"),
          py::arg("nx") = 65536, py::arg("ny") = 1, py::arg("nz") = 1, py::arg("nw") = 1);

    m.def("hdf5_write_channel", &hdf5_write_channel,
          py::arg("fn"),
          py::arg("chan"),
          py::arg("par") = py::none(),
          py::arg("ix") = py::none(),
          py::arg("iy") = py::none(),
          py::arg("iz") = py::none(),
          py::arg("iw") = py::none());

    m.def("hdf5_write_dset", &hdf5_write_dset, py::arg("fn"),
          py::arg("ix") = 0, py::arg("iy") = 0, py::arg("iz") = 0, py::arg("iw") = 0,
          py::arg("name"),
          py::arg("data") = py::none());

    m.def("get_ieee_indoor", &get_channels_ieee_indoor,
          py::arg("ap_array") = py::dict(),
          py::arg("sta_array") = py::dict(),
          py::arg("ChannelType"),
          py::arg("CarrierFreq_Hz") = 5.25e9,
          py::arg("tap_spacing_s") = 10.0e-9,
          py::arg("n_users") = 1,
          py::arg("observation_time") = 0.0,
          py::arg("update_rate") = 1.0e-3,
          py::arg("speed_station_kmh") = 0.0,
          py::arg("speed_env_kmh") = 1.2,
          py::arg("Dist_m") = py::array_t<double>(),
          py::arg("n_floors") = py::array_t<arma::uword>(),
          py::arg("uplink") = false,
          py::arg("offset_angles") = py::array_t<double>(),
          py::arg("n_subpath") = 20,
          py::arg("Doppler_effect") = 50.0,
          py::arg("seed") = -1,
          py::arg("KF_linear") = NAN,
          py::arg("XPR_NLOS_linear") = NAN,
          py::arg("SF_std_dB_LOS") = NAN,
          py::arg("SF_std_dB_NLOS") = NAN,
          py::arg("dBP_m") = NAN,
          py::arg("n_walls") = py::array_t<arma::uword>(),
          py::arg("wall_loss") = 5.0,
          py::arg("stack") = false);

    m.def("quantize_delays", &quantize_delays,
          py::arg("coeff_re") = py::none(),
          py::arg("coeff_im") = py::none(),
          py::arg("delay") = py::none(),
          py::arg("tap_spacing") = 5e-9,
          py::arg("max_no_taps") = 48,
          py::arg("power_exponent") = 1.0,
          py::arg("fix_taps") = 0,
          py::arg("stack") = false,
          py::arg("complex") = false,
          py::arg("coeff") = py::none());

    m.def("qrt_file_parse", &qrt_file_parse, py::arg("fn"));

    m.def("qrt_file_read", &qrt_file_read,
          py::arg("fn"),
          py::arg("i_cir") = py::none(),
          py::arg("i_orig") = 0,
          py::arg("downlink") = true,
          py::arg("normalize_M") = 1);
}
