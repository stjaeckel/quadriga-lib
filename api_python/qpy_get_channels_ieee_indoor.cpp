// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# get_ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_rate`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Default KF (linear): A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- Default XPR NLOS: 2 (3 dB); default SF LOS: 3 dB; default SF NLOS: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB
- Default breakpoint distance: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- Floor floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz
- NAN or negative value for any override parameter restores the model default
- Per-snapshot data (`coeff`, `delay`, `path_gain`) is returned as a list of per-snapshot arrays, 
  or stacked into one array along an appended snapshot axis when `stack = True`

## Usage:
```
chan = quadriga_lib.channel.get_ieee_indoor( ap_array, sta_array, ChannelType, CarrierFreq_Hz,
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh,
   Dist_m, n_floors, uplink, offset_angles, n_subpath, Doppler_effect, seed,
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m, n_walls, wall_loss, stack )
```

## Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see arrayant-[[generate]]
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see arrayant-[[generate]]
- **`ChannelType`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`CarrierFreq_Hz`** *(optional)* — Carrier frequency; default: 5.25e9
- **`tap_spacing_s`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; default: 0
- **`update_rate`** *(optional)* — Channel update interval in seconds; relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h; movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h; use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`Dist_m`** *(optional)* — TX-to-RX distance(s); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models; `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `true` to generate uplink (reverse) direction; default: false
- **`offset_angles`** *(optional)* — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `(4, n_users)`; default: [] (auto-generate)
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`Doppler_effect`** *(optional)* — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** *(optional)* — Overrides model KF (linear scale); NAN or negative restores model default; default: NAN
- **`XPR_NLOS_linear`** *(optional)* — Overrides NLOS cross-polarization ratio (linear scale); NAN or negative restores model default; default: NAN
- **`SF_std_dB_LOS`** *(optional)* — Overrides LOS shadow fading std in dB (applied when d < dBP); NAN restores model default; default: NAN
- **`SF_std_dB_NLOS`** *(optional)* — Overrides NLOS shadow fading std in dB (applied when d >= dBP); NAN restores model default; default: NAN
- **`dBP_m`** *(optional)* — Overrides breakpoint distance; NAN or negative restores model default; default: NAN
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `(n_users,)` or `(1,)`; default: 0
- **`wall_loss`** *(optional)* — Penetration loss for a single wall; TGax defines 5 or 7; default: 5
- **`stack`** *(optional)* — If `True`, per-snapshot data is stacked into a single array along an appended 
  snapshot axis instead of being returned as a list of per-snapshot arrays; default: False


## Output:
- **`chan`** — List of length `n_users` containing dictionaries of channel data. Only the keys
  listed below are populated; all other channel keys are omitted by this generator.<br><br>
  | Key                | Description                                                              | Shape `stack = False`            | Shape `stack = True`           |
  | ------------------ | ------------------------------------------------------------------------ | -------------------------------- | ------------------------------ |
  | `name`             | Channel name                                                             | str                              | str                            |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `coeff`            | Channel coefficients, complex valued                                     | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`            | `(n_path, n_snap)`             |
  | `center_frequency` | Center Frequency in Hz                                                   | scalar                           | scalar                         |

See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
- [[hdf5_write_channel]] (for writing channel data to a HDF5 file)
- [[hdf5_read_channel]] (for reading channel data to a HDF5 file)
MD!*/

py::list get_channels_ieee_indoor(const py::dict &ap_array,
                                  const py::dict &sta_array,
                                  const std::string &ChannelType,
                                  const double CarrierFreq_Hz,
                                  const double tap_spacing_s,
                                  const arma::uword n_users,
                                  const double observation_time,
                                  const double update_rate,
                                  const double speed_station_kmh,
                                  const double speed_env_kmh,
                                  const py::array_t<double> &Dist_m,
                                  const py::array_t<arma::uword> &n_floors,
                                  const bool uplink,
                                  const py::array_t<double> &offset_angles,
                                  const arma::uword n_subpath,
                                  const double Doppler_effect,
                                  const arma::sword seed,
                                  const double KF_linear,
                                  const double XPR_NLOS_linear,
                                  const double SF_std_dB_LOS,
                                  const double SF_std_dB_NLOS,
                                  const double dBP_m,
                                  const py::array_t<arma::uword> &n_walls,
                                  const double wall_loss,
                                  const bool stack)
{
    // Parse input arguments
    const auto ant_tx_a = qd_python_dict2arrayant(ap_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(sta_array, true);
    const auto Dist_m_a = qd_python_numpy2arma_Col(Dist_m, true);
    const auto n_floors_a = qd_python_numpy2arma_Col(n_floors, true);
    const auto offset_angles_a = qd_python_numpy2arma_Mat(offset_angles, true);
    const auto n_walls_a = qd_python_numpy2arma_Col(n_walls, true);

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
                                                  seed,
                                                  KF_linear,
                                                  XPR_NLOS_linear,
                                                  SF_std_dB_LOS,
                                                  SF_std_dB_NLOS,
                                                  dBP_m,
                                                  n_walls_a,
                                                  wall_loss);

    // Copy results to Python
    return qd_python_channel2list(chan, false, stack);
}