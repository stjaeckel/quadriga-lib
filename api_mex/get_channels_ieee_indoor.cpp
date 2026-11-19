// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_IEEE_INDOOR
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
  sees a positive Doppler shift `f_D`, as the channel of [[get_channels_spherical]] along the returned station positions
- Uplink: the coefficient and delay matrices are the transposes of the downlink ones (reciprocal channel, no conjugation),
  so both directions have the same Doppler shift; positions and orientations are swapped (the moving station is the transmitter)
- Override parameters (`kf`, `xpr_nlos`, `sf_std_los_dB`, `sf_std_nlos_dB`, `dist_breakpoint`): `NaN` or `[]` selects the model default

## Usage:
```
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
   tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, ...
   dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, ...
   kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB );
```

## Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling; see [[arrayant_generate]]
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling; see [[arrayant_generate]]
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** — Carrier frequency in Hz (finite, above 100 MHz); default: 5.25e9
- **`tap_spacing`** — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** — Channel observation time in seconds; must be finite; default: 0
- **`update_interval`** — Channel update interval in seconds (finite, > 0); relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h (finite); movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** — Environment speed in km/h (finite); use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist`** — TX-to-RX distance(s) in meters (finite, > 0); `[n_users]` or `[1]`; default: 4.99
- **`n_floors`** — Number of floors per user for TGah or TGax models (0 to 4, attenuation only, no delay); `[n_users]` or `[1]`; default: 0
- **`uplink`** — Set `true` to generate uplink (reverse) direction; coefficients are the transpose of the downlink; default: false
- **`offset_angles_deg`** — Azimuth offset angles in degrees (finite); rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS; `[4, n_users]`;
  default: `[]` (TGac auto-defaults for `n_users > 1`)
- **`n_subpath`** — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** — Special Doppler: models D/E use the mains frequency in Hz, model F the vehicle speed in km/h (finite, >= 0); 0 disables; default: 50
- **`seed`** — RNG seed; any other value than `-1` gives the same channels on all platforms; `-1` uses the system random device; default: -1
- **`kf`** — Overrides the model KF (linear scale), finite and >= 0; default: A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0;
  applied to first tap only; with `kf` given, the LOS path is kept also beyond the breakpoint and on other floors
- **`xpr_nlos`** — Overrides the NLOS cross-polarization ratio (linear scale), finite and > 0;
  default: 2 (3 dB)
- **`sf_std_los_dB`** — Overrides the LOS shadow fading std in dB (applied when d < breakpoint), finite and >= 0; default: 3 dB, 2 dB below 1 GHz (TGah)
- **`sf_std_nlos_dB`** — Overrides the NLOS shadow fading std in dB (applied when d >= breakpoint), finite and >= 0;
  default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB, each 1 dB lower below 1 GHz (TGah)
- **`dist_breakpoint`** — Overrides the breakpoint distance in meters, finite and > 0; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** — Number of walls per user TGax models; `[n_users]` or `[1]`; default: 0
- **`wall_loss_dB`** — Penetration loss for a single wall in dB (finite); TGax defines 5 or 7; default: 5

## Output:
- **`chan`**<br>
  Struct array of length `n_users` containing the channel data with the following fields:<br><br>
  | Field              | Description                                                              | Type                                  |
  | ------------------ | ------------------------------------------------------------------------ | ------------------------------------- |
  | `name`             | Channel name                                                             | String                                |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `tx_orientation`   |  Transmitter orientation, Euler angles (AP for downlink, STA for uplink) | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `coeff_re`         | Channel coefficients, real part                                          | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `delay`            | Propagation delays in seconds                                            | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `path_gain`        | Path gain before antenna, linear scale                                   | Size: `[n_path, n_snap]`              |
  | `center_freq` | Center Frequency in Hz                                                   | Scalar                                |

## See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
- [[hdf5_write_channel]] (for writing channel data to a HDF5 file)
- [[hdf5_read_channel]] (for reading channel data to a HDF5 file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 24)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
    auto ant_tx = qd_mex_struct2arrayant(prhs[0], false, false, "ap_array");
    auto ant_rx = qd_mex_struct2arrayant(prhs[1], false, false, "sta_array");

    // Read model parameters
    std::string channel_type = qd_mex_get_string(prhs[2], "", "channel_type");
    double center_freq = (nrhs < 4) ? 5.25e9 : qd_mex_get_scalar<double>(prhs[3], "center_freq", 5.25e9);
    double tap_spacing = (nrhs < 5) ? 10.0e-9 : qd_mex_get_scalar<double>(prhs[4], "tap_spacing", 10.0e-9);
    arma::uword n_users = (nrhs < 6) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[5], "n_users", 1);
    double observation_time = (nrhs < 7) ? 0.0 : qd_mex_get_scalar<double>(prhs[6], "observation_time", 0.0);
    double update_interval = (nrhs < 8) ? 1.0e-3 : qd_mex_get_scalar<double>(prhs[7], "update_interval", 1.0e-3);
    double speed_station_kmh = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "speed_station_kmh", 0.0);
    double speed_env_kmh = (nrhs < 10) ? 1.2 : qd_mex_get_scalar<double>(prhs[9], "speed_env_kmh", 1.2);
    arma::vec dist = (nrhs < 11) ? arma::vec() : qd_mex_get_Col<double>(prhs[10], false, "dist");
    arma::uvec n_floors = (nrhs < 12) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[11], false, "n_floors");
    bool uplink = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "uplink", false);
    arma::mat offset_angles_deg = (nrhs < 14) ? arma::mat() : qd_mex_get_Mat<double>(prhs[13], false, "offset_angles_deg");
    arma::uword n_subpath = (nrhs < 15) ? 20 : qd_mex_get_scalar<arma::uword>(prhs[14], "n_subpath", 20);
    double doppler_effect = (nrhs < 16) ? 50.0 : qd_mex_get_scalar<double>(prhs[15], "doppler_effect", 50.0);
    const arma::sword seed = (nrhs < 17) ? -1 : (arma::sword)qd_mex_get_seed(prhs[16]);
    double kf = (nrhs < 18) ? NAN : qd_mex_get_scalar<double>(prhs[17], "kf", NAN);
    double xpr_nlos = (nrhs < 19) ? NAN : qd_mex_get_scalar<double>(prhs[18], "xpr_nlos", NAN);
    double sf_std_los_dB = (nrhs < 20) ? NAN : qd_mex_get_scalar<double>(prhs[19], "sf_std_los_dB", NAN);
    double sf_std_nlos_dB = (nrhs < 21) ? NAN : qd_mex_get_scalar<double>(prhs[20], "sf_std_nlos_dB", NAN);
    double dist_breakpoint = (nrhs < 22) ? NAN : qd_mex_get_scalar<double>(prhs[21], "dist_breakpoint", NAN);
    arma::uvec n_walls = (nrhs < 23) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[22], false, "n_walls");
    double wall_loss_dB = (nrhs < 24) ? 5.0 : qd_mex_get_scalar<double>(prhs[23], "wall_loss_dB", 5.0);

    // Declare outputs
    std::vector<quadriga_lib::channel<double>> chan;

    // Call library function (empty inputs select the defaults)
    CALL_QD(chan = quadriga_lib::get_channels_ieee_indoor(ant_tx, ant_rx, channel_type, center_freq, tap_spacing, n_users,
                                                          observation_time, update_interval, speed_station_kmh, speed_env_kmh,
                                                          dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect,
                                                          seed, kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint,
                                                          n_walls, wall_loss_dB));

    plhs[0] = qd_mex_channel2struct(chan);
}