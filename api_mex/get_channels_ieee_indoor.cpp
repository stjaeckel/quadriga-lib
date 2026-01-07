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

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_IEEE_INDOOR
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

## Description:
- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions.
- 2D model: no elevation angles are used; azimuth angles and planar motion are considered.
- For 3D antenna models (default models from [[arrayant_generate]]), only the azimuth cut at `elevation_grid = 0` is used
- Supports channel model types `A, B, C, D, E, F` (as defined by TGn) via `ChannelType`.
- Can generate MU-MIMO channels (`n_users > 1`) with per-user distances/floors and optional angle
  offsets according to TGac.
- Optional time evolution via `observation_time`, `update_rate`, and mobility parameters.

## Declaration:
```
chan = quadriga_lib.get_channels_ieee_indoor(ap_array, sta_array, ChannelType, CarrierFreq_Hz, ...
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, ...
   Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed, ...
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m);
```

## ap_array:
- **`ap_array`** [1]<br>
  Struct containing the access point array antenna with `n_tx` elements (= ports after element coupling)
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation_ap, n_azimuth_ap, n_elements_ap]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation_ap, n_azimuth_ap, n_elements_ap]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation_ap, n_azimuth_ap, n_elements_ap]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation_ap, n_azimuth_ap, n_elements_ap]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth_ap]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation_ap]`

- **`sta_array`** [2]<br>
  Struct containing the mobile station array antenna with `n_rx` elements (= ports after element coupling)
  `e_theta_re`     | Real part of e-theta field component             | Size: `[n_elevation_sta, n_azimuth_sta, n_elements_sta]`
  `e_theta_im`     | Imaginary part of e-theta field component        | Size: `[n_elevation_sta, n_azimuth_sta, n_elements_sta]`
  `e_phi_re`       | Real part of e-phi field component               | Size: `[n_elevation_sta, n_azimuth_sta, n_elements_sta]`
  `e_phi_im`       | Imaginary part of e-phi field component          | Size: `[n_elevation_sta, n_azimuth_sta, n_elements_sta]`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Size: `[n_azimuth_sta]`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Size: `[n_elevation_sta]`

- `**ChannelType**` [3]<br>
  Channel model type as defined by TGn. String. Supported: `A, B, C, D, E, F`.

- `**CarrierFreq_Hz** = 5.25e9` [4] (optional)<br>
  Carrier frequency in Hz.

- `**tap_spacing_s** = 10e-9` [5] (optional)<br>
  Tap spacing in seconds. Must be equal to `10 ns / 2^k` (TGn default = `10e-9`).

- `**n_users** = 1` [6] (optional)<br>
  Number of users (only for TGac, TGah). Output struct array length equals `n_users`.

- `**observation_time** = 0` [7] (optional)<br>
  Channel observation time in seconds. `0` creates a static channel.

- `**update_rate** = 1e-3` [8] (optional)<br>
  Channel update interval in seconds (only relevant when `observation_time > 0`).

- `**speed_station_kmh** = 0` [9] (optional)<br>
  Station movement speed in km/h. Movement direction is `AoA_offset`. Only relevant when `observation_time > 0`.

- `**speed_env_kmh** = 1.2` [10] (optional)<br>
  Environment movement speed in km/h. Default `1.2` for TGn, use `0.089` for TGac. Only relevant when `observation_time > 0`.

- `vector **Dist_m** = [4.99]` [11] (optional)<br>
  TX-to-RX distance(s) in meters. Length `n_users` or length `1` (same distance for all users).

- `vector **n_floors** = [0]` [12] (optional)<br>
  Number of floors for TGah model (per user), up to 4 floors. Length `n_users` or length `1`.

- `**uplink** = false` [13] (optional)<br>
  Channel direction flag. Default is downlink; set to `true` to generate reverse (uplink) direction.

- `**offset_angles** = []` [14] (optional)<br>
  Offset angles in degree for MU-MIMO channels. Empty uses model defaults (TGac auto for `n_users > 1`).
  Size `[4, n_users]` with rows: `AoD LOS, AoD NLOS, AoA LOS, AoA NLOS`.

- `**n_subpath** = 20` [15] (optional)<br>
  Number of sub-paths per path/cluster used for Laplacian angular spread mapping.

- `**Doppler_effect** = 50` [16] (optional)<br>
  Special Doppler effects: models `D, E` (fluorescent lights, value = mains freq.) and `F` (moving vehicle speed in km/h).
  Use `0` to disable.

- `**seed** = -1` [17] (optional)<br>
  Numeric seed for repeatability. `-1` disables the fixed seed and uses the system random device.

- `**KF_linear** = []` [18] (optional)<br>
  Overwrites the model-specific KF-value. If this parameter is empty (default), NAN or negative, model defaults are used:
  A/B/C (KF = 1 for d < dBP, 0 otherwise); D (KF = 2 for d < dBP, 0 otherwise); E/F (KF = 4 for d < dBP, 0 otherwise).
  KF is applied to the first tap only. Breakpoint distance is ignored for `KF_linear >= 0`.

- `**XPR_NLOS_linear** = []` [19] (optional)<br>
  Overwrites the model-specific Cross-polarization ratio. If this parameter is empty (default), NAN or negative,
  the model default of 2 (3 dB) is used. XPR is applied to all NLOS taps.

- `**SF_std_dB_LOS** = []` [20] (optional)<br>
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is empty (default) or NAN,
  the model default of 3 dB is used. `SF_std_dB_LOS` is applied to all LOS channels, where the
  AP-STA distance d < dBP.

- `**SF_std_dB_NLOS** = []` [21] (optional)<br>
  Overwrites the model-specific shadow fading for LOS channels. If this parameter is empty (default) or NAN,
  the model defaults are A/B: 4 dB, C/D: 5 dB, E/F: 6 dB. `SF_std_dB_NLOS` is applied to all NLOS channels,
  where the AP-STA distance d >= dBP.

- `**dBP_m** = []` [22] (optional)<br>
  Overwrites the model-specific breakpoint distance. If this parameter is empty (default), NAN or negative,
  the model defaults are A/B/C: 5 m, D: 10 m, E: 20 m, F: 30 m.

## Returns:
- **`chan`**<br>
  Struct array of length `n_users` containing the channel data with the following fields.
  `name`           | Channel name                                                             | String
  `tx_position`    | Transmitter positions (AP for downlink, STA for uplink)                  | Size: `[3, 1]` or `[3, n_snap]`
  `rx_position`    | Receiver positions (STA for downlink, AP for uplink)                     | Size: `[3, 1]` or `[3, n_snap]`
  `tx_orientation` | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | Size: `[3, 1]` or `[3, n_snap]`
  `rx_orientation` | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: `[3, 1]` or `[3, n_snap]`
  `coeff_re`       | Channel coefficients, real part                                          | Size: `[n_rx, n_tx, n_path, n_snap]`
  `coeff_im`       | Channel coefficients, imaginary part                                     | Size: `[n_rx, n_tx, n_path, n_snap]`
  `delay`          | Propagation delays in seconds                                            | Size: `[n_rx, n_tx, n_path, n_snap]`
  `path_gain`      | Path gain before antenna, linear scale                                   | Size: `[n_path, n_snap]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Need at least TX and RX antennas and channel type.");

    if (nrhs > 22)
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
    double CarrierFreq_Hz = (nrhs < 4) ? 5.25e9 : qd_mex_get_scalar<double>(prhs[3], "CarrierFreq_GHz", 5.25e9);
    double tap_spacing_s = (nrhs < 5) ? 10.0e-9 : qd_mex_get_scalar<double>(prhs[4], "tap_spacing_ns", 10.0e-9);
    arma::uword n_users = (nrhs < 6) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[5], "n_users", 1);
    double observation_time = (nrhs < 7) ? 0.0 : qd_mex_get_scalar<double>(prhs[6], "observation_time", 0.0);
    double update_rate = (nrhs < 8) ? 1.0e-3 : qd_mex_get_scalar<double>(prhs[7], "update_rate", 1.0e-3);
    double speed_station_kmh = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "speed_station_kmh", 0.0);
    double speed_env_kmh = (nrhs < 10) ? 1.2 : qd_mex_get_scalar<double>(prhs[9], "speed_env_kmh", 1.2);
    arma::vec Dist = (nrhs < 11) ? arma::vec{4.99} : qd_mex_get_double_Col(prhs[10]);
    arma::uvec n_floors = (nrhs < 12) ? arma::uvec{0} : qd_mex_typecast_Col<arma::uword>(prhs[11]);
    bool uplink = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "uplink", false);
    arma::mat offset_angles = (nrhs < 14) ? arma::mat{} : qd_mex_get_double_Mat(prhs[13]);
    arma::uword n_subpath = (nrhs < 15) ? 20 : qd_mex_get_scalar<arma::uword>(prhs[14], "n_subpath", 20);
    double Doppler_effect = (nrhs < 16) ? 50.0 : qd_mex_get_scalar<double>(prhs[15], "Doppler_effect", 50.0);
    arma::sword seed = (nrhs < 17) ? -1 : qd_mex_get_scalar<arma::sword>(prhs[16], "seed", -1);
    double KF_linear = (nrhs < 18) ? NAN : qd_mex_get_scalar<double>(prhs[17], "KF_linear", NAN);
    double XPR_NLOS_linear = (nrhs < 19) ? NAN : qd_mex_get_scalar<double>(prhs[18], "XPR_NLOS_linear", NAN);
    double SF_std_dB_LOS = (nrhs < 20) ? NAN : qd_mex_get_scalar<double>(prhs[19], "SF_std_dB_LOS", NAN);
    double SF_std_dB_NLOS = (nrhs < 21) ? NAN : qd_mex_get_scalar<double>(prhs[20], "SF_std_dB_NLOS", NAN);
    double dBP_m = (nrhs < 22) ? NAN : qd_mex_get_scalar<double>(prhs[21], "dBP_m", NAN);

    // Declare outputs
    std::vector<quadriga_lib::channel<double>> chan;

    // Call library function
    CALL_QD(chan = quadriga_lib::get_channels_ieee_indoor(ant_tx,
                                                          ant_rx,
                                                          ChannelType,
                                                          CarrierFreq_Hz,
                                                          tap_spacing_s,
                                                          n_users,
                                                          observation_time,
                                                          update_rate,
                                                          speed_station_kmh,
                                                          speed_env_kmh,
                                                          Dist,
                                                          n_floors,
                                                          uplink,
                                                          offset_angles,
                                                          n_subpath,
                                                          Doppler_effect,
                                                          seed,
                                                          KF_linear,
                                                          XPR_NLOS_linear,
                                                          SF_std_dB_LOS,
                                                          SF_std_dB_NLOS,
                                                          dBP_m));

    if (nlhs > 0)
    {
        std::vector<std::string> fields = {"name", "tx_position", "rx_position", "tx_orientation",
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