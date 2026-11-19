// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# channel.ieee_indoor
Generate indoor MIMO channel realizations for IEEE TGn/TGac/TGax/TGah models

- Generates one or multiple indoor channel realizations based on IEEE TGn/TGac/TGax/TGah model definitions
- 2D model: azimuth angles and planar motion only, no elevation
- Supported channel types: `A, B, C, D, E, F` (TGn definitions)
- MU-MIMO supported (`n_users > 1`) with per-user distances/floors and optional angle offsets per TGac
- Time-evolving channels via `observation_time`, `update_interval`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Floor penetration loss according to TGah for `center_freq` < 1 GHz and TGax for above 1 GHz; floors add attenuation
  (and the TGah shadow fading below 1 GHz) but no delay; the station is placed 3 m per floor above the AP
- Taps with zero power in the model tables are removed; the number of paths does not depend on distance, floors or walls
- Time evolution: each sub-path is rotated by `exp(+j 2 pi f_D t)`, i.e. a station moving towards the AP or a scatterer
  sees a positive Doppler shift `f_D`, as the channel of [[channel.spherical]] along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- Override parameters (`kf`, `xpr_nlos`, `sf_std_los_dB`, `sf_std_nlos_dB`, `dist_breakpoint`): `NaN` selects the model default
- Per-snapshot data (`coeff`, `delay`, `path_gain`) is returned as a list of per-snapshot arrays, 
  or stacked into one array along an appended snapshot axis when `stack = True`

## Usage:
```
chan = quadriga_lib.channel.ieee_indoor( ap_array, sta_array, channel_type, center_freq, *, \
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, \
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, \
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB, stack )
```

## Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling, see arrayant-[[generate]]
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling, see arrayant-[[generate]]
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** *(optional)* — Carrier frequency in Hz (finite, above 100 MHz); default: 5.25e9
- **`tap_spacing`** *(optional)* — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** *(optional)* — Number of users (TGac/TGah/TGax only); output list length equals `n_users`; default: 1
- **`observation_time`** *(optional)* — Channel observation time in seconds; must be finite; default: 0
- **`update_interval`** *(optional)* — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** *(optional)* — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** *(optional)* — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist`** *(optional)* — TX-to-RX distance(s) in meters (finite, > 0); `(n_users,)` or `(1,)`; default: 4.99
- **`n_floors`** *(optional)* — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `(n_users,)` or `(1,)`; default: 0
- **`uplink`** *(optional)* — Set `True` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: False
- **`offset_angles_deg`** *(optional)* — Azimuth offset angles in degrees (finite); rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; `(4, n_users)`;
  default: TGac auto-defaults for `n_users > 1`
- **`n_subpath`** *(optional)* — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** *(optional)* — Special Doppler: models D/E use the mains frequency in Hz, model F the vehicle speed in km/h (finite, >= 0); 0 disables; default: 50
- **`seed`** *(optional)* — RNG seed; any other value than `-1` gives the same channels on all platforms; `-1` uses the system random device; default: -1
- **`kf`** *(optional)* — Overrides the model KF (linear scale), finite and >= 0; default: A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0;
  applied to first tap only; with `kf` given, the LOS path is kept also beyond the breakpoint and on other floors
- **`xpr_nlos`** *(optional)* — Overrides the NLOS cross-polarization ratio (linear scale), finite and > 0; default: 2 (3 dB)
- **`sf_std_los_dB`** *(optional)* — Overrides the LOS shadow fading std in dB (applied when d < breakpoint), finite and >= 0; default: 3 dB, 2 dB below 1 GHz (TGah)
- **`sf_std_nlos_dB`** *(optional)* — Overrides the NLOS shadow fading std in dB (applied when d >= breakpoint), finite and >= 0;
  default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB, each 1 dB lower below 1 GHz (TGah)
- **`dist_breakpoint`** *(optional)* — Overrides the breakpoint distance in meters, finite and > 0; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** *(optional)* — Number of walls per user TGax models; `(n_users,)` or `(1,)`; default: 0
- **`wall_loss_dB`** *(optional)* — Penetration loss for a single wall in dB (finite); TGax defines 5 or 7; default: 5
- **`stack`** *(optional)* — If `True`, per-snapshot data is stacked into a single array along an appended 
  snapshot axis instead of being returned as a list of per-snapshot arrays; default: False


## Output:
- **`chan`** — List of length `n_users` containing dictionaries of channel data. Only the keys
  listed below are populated; all other channel keys are omitted by this generator.<br><br>
  | Key                | Description                                                              | Shape `stack = False`            | Shape `stack = True`           |
  | ------------------ | ------------------------------------------------------------------------ | -------------------------------- | ------------------------------ |
  | `name`             | Channel name                                                             | str                              | str                            |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, 1)` or `(3, n_snap)`        | `(3, 1)` or `(3, n_snap)`      |
  | `coeff`            | Channel coefficients, complex valued                                     | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`            | `(n_path, n_snap)`             |
  | `center_freq` | Center Frequency in Hz                                                   | `(1,)`                           | `(1,)`                         |
  | `initial_position` | 0-based index of the reference snapshot                                  | int                              | int                            |

## See also:
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
                                  const std::string &channel_type,
                                  const double center_freq,
                                  const double tap_spacing,
                                  const arma::uword n_users,
                                  const double observation_time,
                                  const double update_interval,
                                  const double speed_station_kmh,
                                  const double speed_env_kmh,
                                  const py::handle &dist,
                                  const py::handle &n_floors,
                                  const py::handle &uplink,
                                  const py::handle &offset_angles_deg,
                                  const arma::uword n_subpath,
                                  const double doppler_effect,
                                  const arma::sword seed,
                                  const double kf,
                                  const double xpr_nlos,
                                  const double sf_std_los_dB,
                                  const double sf_std_nlos_dB,
                                  const double dist_breakpoint,
                                  const py::handle &n_walls,
                                  const double wall_loss_dB,
                                  const py::handle &stack)
{
    // Parse input arguments (None gives an empty container, which selects the default)
    const auto ant_tx_a = qd_python_dict2arrayant(ap_array, true);
    const auto ant_rx_a = qd_python_dict2arrayant(sta_array, true);
    const auto dist_a = qd_python_numpy2arma_Col<double>(dist, true);
    const auto n_floors_a = qd_python_numpy2arma_Col<arma::uword>(n_floors, true, false, "n_floors");
    const bool uplink_b = qd_python_flag(uplink, false, "uplink");
    const auto offset_angles_deg_a = qd_python_numpy2arma_Mat<double>(offset_angles_deg, true);
    const auto n_walls_a = qd_python_numpy2arma_Col<arma::uword>(n_walls, true, false, "n_walls");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // Call CPP implementation
    const auto chan = quadriga_lib::get_channels_ieee_indoor(ant_tx_a, ant_rx_a, channel_type, center_freq, tap_spacing, n_users,
                                                             observation_time, update_interval, speed_station_kmh, speed_env_kmh,
                                                             dist_a, n_floors_a, uplink_b, offset_angles_deg_a, n_subpath,
                                                             doppler_effect, seed, kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB,
                                                             dist_breakpoint, n_walls_a, wall_loss_dB);

    // Copy results to Python
    return qd_python_channel2list(chan, false, stack_b);
}
