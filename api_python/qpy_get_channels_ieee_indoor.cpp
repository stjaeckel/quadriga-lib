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

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# get_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

## Description:
- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions.
- 2D model: no elevation angles are used; azimuth angles and planar motion are considered.
- For 3D antenna models (default models from arrayant [[generate]]), only the azimuth cut at `elevation_grid = 0` is used
- Supports channel model types `A, B, C, D, E, F` (as defined by TGn) via `ChannelType`.
- Can generate MU-MIMO channels (`n_users > 1`) with per-user distances/floors and optional angle
  offsets according to TGac.
- Optional time evolution via `observation_time`, `update_rate`, and mobility parameters.

## Declaration:
```
from quadriga_lib import channel

chan = channel.get_ieee_indoor(ap_array, sta_array, ChannelType, CarrierFreq_Hz, ...
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, ...
   Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed);
```

## ap_array:
- **`ap_array`** [1]<br>
  Dictionary containing the access point array antenna with `n_tx` elements (= ports after element coupling)
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation_ap, n_azimuth_ap, n_elements_ap)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_elevation_ap, n_azimuth_ap, n_elements_ap)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_elevation_ap, n_azimuth_ap, n_elements_ap)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_elevation_ap, n_azimuth_ap, n_elements_ap)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth_ap)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation_ap)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements_ap)`
  `coupling_re`    | Coupling matrix, real part, optional                  | Shape: `(n_elements_ap, n_ports_ap)`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Shape: `(n_elements_ap, n_ports_ap)`

- **`sta_array`** [2]<br>
  Dictionary containing the mobile station array antenna with `n_rx` elements (= ports after element coupling)
  `e_theta_re`     | e-theta field component, real part                    | Shape: `(n_elevation_sta, n_azimuth_sta, n_elements_sta)`
  `e_theta_im`     | e-theta field component, imaginary part               | Shape: `(n_elevation_sta, n_azimuth_sta, n_elements_sta)`
  `e_phi_re`       | e-phi field component, real part                      | Shape: `(n_elevation_sta, n_azimuth_sta, n_elements_sta)`
  `e_phi_im`       | e-phi field component, imaginary part                 | Shape: `(n_elevation_sta, n_azimuth_sta, n_elements_sta)`
  `azimuth_grid`   | Azimuth angles in [rad], -pi to pi, sorted            | Shape: `(n_azimuth_sta)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted      | Shape: `(n_elevation_sta)`
  `element_pos`    | Antenna element (x,y,z) positions, optional           | Shape: `(3, n_elements_sta)`
  `coupling_re`    | Coupling matrix, real part, optional                  | Shape: `(n_elements_sta, n_ports_sta)`
  `coupling_im`    | Coupling matrix, imaginary part, optional             | Shape: `(n_elements_sta, n_ports_sta)`

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
  Shape `[4, n_users]` with rows: `AoD LOS, AoD NLOS, AoA LOS, AoA NLOS`.

- `**n_subpath** = 20` [15] (optional)<br>
  Number of sub-paths per path/cluster used for Laplacian angular spread mapping.

- `**Doppler_effect** = 50` [16] (optional)<br>
  Special Doppler effects: models `D, E` (fluorescent lights, value = mains freq.) and `F` (moving vehicle speed in km/h).
  Use `0` to disable.

- `**seed** = -1` [17] (optional)<br>
  Numeric seed for repeatability. `-1` disables the fixed seed and uses the system random device.

## Returns:
- **`chan`**<br>
  List of length `n_users` containing dictionaries of channel data with the following keys.
  `name`           | Channel name                                                             | String
  `tx_position`    | Transmitter positions (AP for downlink, STA for uplink)                  | Shape: `(3, 1)` or `(3, n_snap)`
  `rx_position`    | Receiver positions (STA for downlink, AP for uplink)                     | Shape: `(3, 1)` or `(3, n_snap)`
  `tx_orientation` | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | Shape: `(3, 1)` or `(3, n_snap)`
  `rx_orientation` | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Shape: `(3, 1)` or `(3, n_snap)`
  `coeff`          | Channel coefficients, complex valued                                     | list of `[n_rx, n_tx, n_path_s]`
  `delay`          | Propagation delays in seconds                                            | list of `[n_rx, n_tx, n_path_s]`
  `path_gain`      | Path gain before antenna, linear scale                                   | list of `[n_path_s]`
MD!*/

py::list get_channels_ieee_indoor(const py::dict &ap_array,                 // Access point array antenna with 'n_tx' elements (= ports after element coupling)
                                  const py::dict &sta_array,                // Mobile station array antenna with 'n_rx' elements (= ports after element coupling)
                                  const std::string &ChannelType,           // Channel Model Type (A, B, C, D, E, F) as defined by TGn
                                  const double CarrierFreq_Hz,              // Carrier frequency in Hz
                                  const double tap_spacing_s,               // Taps spacing in seconds, must be equal to 10 ns divided by a power of 2, TGn = 10e-9
                                  const arma::uword n_users,                // Number of user (only for TGac, TGah)
                                  const double observation_time,            // Channel observation time in seconds (0.0 = static channel)
                                  const double update_rate,                 // Channel update interval in seconds
                                  const double speed_station_kmh,           // Movement speed of the station in km/h (optional feature, default = 0), movement direction = AoA_offset
                                  const double speed_env_kmh,               // Movement speed of the environment in km/h (default = 1.2 for TGn) use 0.089 for TGac
                                  const py::array_t<double> &Dist_m,        // Distance between TX and TX in meters, length n_users or length 1 (if same for all users)
                                  const py::array_t<arma::uword> &n_floors, // Number of floors for the TGah model, adjusted for each user, up to 4 floors, length n_users or length 1 (if same for all users)
                                  const bool uplink,                        // Default channel direction is downlink, set uplink to true to get reverse direction
                                  const py::array_t<double> &offset_angles, // Offset angles in degree for MU-MIMO channels, empty (TGac auto for n_users > 1), Size: [4, n_users] with rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS
                                  const arma::uword n_subpath,              // Number of sub-paths per path and cluster for Laplacian AS mapping
                                  const double Doppler_effect,              // Special Doppler effects in models D, E (fluorescent lights, value = mains freq.) and F (moving vehicle speed in kmh), use 0.0 to disable
                                  const arma::sword seed)                   // Numeric seed, optional, value -1 disabled seed and uses system random device
{
    // Parse input arguments
    const auto ant_tx_a = qd_python_dict2arrayant(ap_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(sta_array, true);
    const auto Dist_m_a = qd_python_numpy2arma_Col(Dist_m, true);
    const auto n_floors_a = qd_python_numpy2arma_Col(n_floors, true);
    const auto offset_angles_a = qd_python_numpy2arma_Mat(offset_angles, true);

    // Call CPP implementation
    std::vector<quadriga_lib::channel<double>> chan;
    chan = quadriga_lib::get_channels_ieee_indoor(ant_tx_a,
                                                  ant_rx_a,
                                                  ChannelType,
                                                  CarrierFreq_Hz,
                                                  tap_spacing_s,
                                                  n_users,
                                                  observation_time,
                                                  update_rate,
                                                  speed_station_kmh,
                                                  speed_env_kmh,
                                                  Dist_m_a,
                                                  n_floors_a,
                                                  uplink,
                                                  offset_angles_a,
                                                  n_subpath,
                                                  Doppler_effect,
                                                  seed);

    // Copy results to Python
    py::list list;
    for (const auto &channel : chan)
    {
        py::dict item;
        item["name"] = channel.name;
        item["tx_position"] = qd_python_copy2numpy(channel.tx_pos);
        item["rx_position"] = qd_python_copy2numpy(channel.rx_pos);
        item["tx_orientation"] = qd_python_copy2numpy(channel.tx_orientation);
        item["rx_orientation"] = qd_python_copy2numpy(channel.rx_orientation);
        item["coeff"] = qd_python_copy2numpy(channel.coeff_re, channel.coeff_im);
        item["delay"] = qd_python_copy2numpy(channel.delay);
        item["path_gain"] = qd_python_copy2numpy(channel.path_gain);
        list.append(item);
    }
    return list;
}