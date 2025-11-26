// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "mex_helper_functions.hpp"
#include "quadriga_lib.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - tx_array            Transmit antenna array
    //  1 - rx_array            Receive antenna array
    //  2 - ChannelType         Channel Model Type (A, B, C, D, E, F) as defined by TGn
    //  3 - CarrierFreq_GHz     Carrier frequency in GHz, Default = 5.25
    //  4 - tap_spacing_ns      Taps spacing in nanoseconds, must be equal to 10 ns divided by a power of 2, Default = 10
    //  5 - n_users             Number of user (only for TGac, TGn = 1), Default = 1
    //  6 - observation_time    Channel observation time in seconds (0.0 = static channel)
    //  7 - update_rate         Channel update interval in seconds, default = 1e-3
    //  8 - speed_station_kmh   Movement speed of the station in km/h (Jakes Doppler), Default = 0
    //  9 - speed_env_kmh       Movement speed of the environment in km/h (Bell-Doppler), Default = 0.089
    // 10 - Dist_m              Distance between TX and RX in meters, length n_users or length 1 (if same for all users), Default = 4.99
    // 11 - n_floors            Number of floors for the TGah model, adjusted for each user, up to 4 floors, length n_users or length 1, Default = 0
    // 12 - uplink              Default channel direction is downlink, set uplink to true to get reverse direction
    // 13 - offset_angles       Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
    // 15 - n_subpath           Number of sub-paths per path and cluster for Laplacian AS mapping, Default = 20
    // 16 - seed                Random seed for repetitive simulations

    // Outputs:
    //  chan                Struct array of length n_user containing the channel data

    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Need at least TX and RX antennas and channel type.");

    if (nrhs > 16)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    // Assemble TX array antenna object
    auto ant_tx = quadriga_lib::arrayant<double>();
    ant_tx.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_re"));
    ant_tx.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_theta_im"));
    ant_tx.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_re"));
    ant_tx.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[0], "e_phi_im"));
    ant_tx.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "azimuth_grid"));
    ant_tx.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[0], "elevation_grid"));
    if (qd_mex_has_field(prhs[0], "element_pos"))
        ant_tx.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "element_pos"));
    if (qd_mex_has_field(prhs[0], "coupling_re"))
        ant_tx.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_re"));
    if (qd_mex_has_field(prhs[0], "coupling_im"))
        ant_tx.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[0], "coupling_im"));

    // Assemble RX array antenna object
    auto ant_rx = quadriga_lib::arrayant<double>();
    ant_rx.e_theta_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_re"));
    ant_rx.e_theta_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_theta_im"));
    ant_rx.e_phi_re = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_re"));
    ant_rx.e_phi_im = qd_mex_get_double_Cube(qd_mex_get_field(prhs[1], "e_phi_im"));
    ant_rx.azimuth_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "azimuth_grid"));
    ant_rx.elevation_grid = qd_mex_get_double_Col(qd_mex_get_field(prhs[1], "elevation_grid"));
    if (qd_mex_has_field(prhs[1], "element_pos"))
        ant_rx.element_pos = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "element_pos"));
    if (qd_mex_has_field(prhs[1], "coupling_re"))
        ant_rx.coupling_re = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_re"));
    if (qd_mex_has_field(prhs[1], "coupling_im"))
        ant_rx.coupling_im = qd_mex_get_double_Mat(qd_mex_get_field(prhs[1], "coupling_im"));

    // Read model parameters
    std::string ChannelType = qd_mex_get_string(prhs[2]);
    double CarrierFreq = (nrhs < 4) ? 5.25 : qd_mex_get_scalar<double>(prhs[3], "CarrierFreq_GHz", 5.25);
    double tap_spacing = (nrhs < 5) ? 10.0 : qd_mex_get_scalar<double>(prhs[4], "tap_spacing_ns", 10.0);
    arma::uword n_users = (nrhs < 6) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[5], "n_users", 1);
    double observation_time = (nrhs < 7) ? 0.0 : qd_mex_get_scalar<double>(prhs[6], "observation_time", 0.0);
    double update_rate = (nrhs < 8) ? 1.0e-3 : qd_mex_get_scalar<double>(prhs[7], "update_rate", 1.0e-3);
    double speed_station_kmh = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "speed_station_kmh", 0.0);
    double speed_env_kmh = (nrhs < 10) ? 0.089 : qd_mex_get_scalar<double>(prhs[9], "speed_env_kmh", 0.089);
    arma::vec Dist = (nrhs < 11) ? arma::vec{4.99} : qd_mex_get_double_Col(prhs[10]);
    arma::uvec n_floors = (nrhs < 12) ? arma::uvec{0} : qd_mex_typecast_Col<arma::uword>(prhs[11]);
    bool uplink = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "uplink", false);
    arma::mat offset_angles = (nrhs < 14) ? arma::mat{} : qd_mex_get_double_Mat(prhs[13]);
    arma::uword n_subpath = (nrhs < 15) ? 20 : qd_mex_get_scalar<arma::uword>(prhs[14], "n_subpath", 20);
    arma::uword seed = (nrhs < 16) ? -1 : qd_mex_get_scalar<arma::uword>(prhs[15], "seed", -1);

    CarrierFreq *= 1e9;  // GHz to Hz
    tap_spacing *= 1e-9; // ns to seconds

    // Declare outputs
    std::vector<quadriga_lib::channel<double>> chan;

    // Call library function
    CALL_QD(chan = quadriga_lib::get_channels_ieee_indoor(ant_tx,
                                                          ant_rx,
                                                          ChannelType,
                                                          CarrierFreq,
                                                          tap_spacing,
                                                          n_users,
                                                          observation_time,
                                                          update_rate,
                                                          speed_station_kmh,
                                                          Dist,
                                                          n_floors,
                                                          uplink,
                                                          offset_angles,
                                                          n_subpath,
                                                          seed));

    if (nlhs > 0)
    {
        std::vector<std::string> fields = {"name", "tx_pos", "rx_pos", "tx_orientation",
                                           "rx_orientation", "coeff_re", "coeff_im",
                                           "delay", "path_gain"};

        plhs[0] = qd_mex_make_struct(fields, n_users);

        for (arma::uword i_user = 0; i_user < n_users; ++i_user)
        {
            qd_mex_set_field(plhs[0], fields[0], mxCreateString(chan[i_user].name.c_str()), i_user);
            qd_mex_set_field(plhs[0], fields[1], qd_mex_copy2matlab(&chan[i_user].tx_pos), i_user);
            qd_mex_set_field(plhs[0], fields[2], qd_mex_copy2matlab(&chan[i_user].rx_pos), i_user);
            qd_mex_set_field(plhs[0], fields[3], qd_mex_copy2matlab(&chan[i_user].tx_orientation), i_user);
            qd_mex_set_field(plhs[0], fields[4], qd_mex_copy2matlab(&chan[i_user].rx_orientation), i_user);
            qd_mex_set_field(plhs[0], fields[5], qd_mex_vector2matlab(&chan[i_user].coeff_re), i_user);
            qd_mex_set_field(plhs[0], fields[6], qd_mex_vector2matlab(&chan[i_user].coeff_im), i_user);
            qd_mex_set_field(plhs[0], fields[7], qd_mex_vector2matlab(&chan[i_user].delay), i_user);
            qd_mex_set_field(plhs[0], fields[8], qd_mex_vector2matlab(&chan[i_user].path_gain), i_user);
        }
    }
}