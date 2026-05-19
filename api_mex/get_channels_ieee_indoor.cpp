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
- Time-evolving channels via `observation_time`, `update_rate`, and mobility parameters; `observation_time = 0.0` yields a static channel
- Floor penetration loss according to TGah for CarrierFreq < 1 GHz and TGax for above 1 GHz
- NAN or negative value for any override parameter restores the model default

## Usage:
```
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
   tap_spacing_s, n_users, observation_time, update_rate, speed_station_kmh, speed_env_kmh, ...
   dist_m, n_floors, uplink, offset_angles, n_subpath, doppler_effect, seed, ...
   KF_linear, XPR_NLOS_linear, SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m, n_walls, wall_loss );
```

## Inputs:
- **`ap_array`** — Access point array antenna; `n_tx` = number of ports after element coupling; see [[arrayant_generate]]
- **`sta_array`** — Mobile station array antenna; `n_rx` = number of ports after element coupling; see [[arrayant_generate]]
- **`channel_type`** — Model type string; one of `"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`
- **`center_freq`** — Carrier frequency; default: 5.25e9
- **`tap_spacing_s`** — Tap spacing in seconds; must equal `10 ns / 2^k`; default: 10e-9
- **`n_users`** — Number of users (TGac/TGah/TGax only); output vector length equals `n_users`; default: 1
- **`observation_time`** — Channel observation time in seconds; default: 0
- **`update_rate`** — Channel update interval in seconds; relevant only when `observation_time > 0`; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h; movement direction is `AoA_offset`; relevant only when `observation_time > 0`; default: 0
- **`speed_env_kmh`** — Environment speed in km/h; use `0.089` for TGac; relevant only when `observation_time > 0`; default: 1.2 (TGn)
- **`dist_m`** — TX-to-RX distance(s); `[n_users]` or `[1]`; default: 4.99
- **`n_floors`** — Number of floors per user for TGah or TGax models; `[n_users]` or `[1]`; default: 0
- **`uplink`** — Set `true` to generate uplink (reverse) direction; default: false
- **`offset_angles`** — Azimuth offset angles in degrees; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  empty uses TGac auto-defaults for `n_users > 1`; `[4, n_users]`; default: [] (auto-generate)
- **`n_subpath`** — Sub-paths per cluster for Laplacian angular spread mapping; default: 20
- **`doppler_effect`** — Special Doppler: models D/E use mains frequency (Hz), model F uses vehicle speed (km/h); 0 disables; default: 50
- **`seed`** — RNG seed for repeatability; `-1` uses the system random device; default: -1
- **`KF_linear`** — Overrides model KF (linear scale); default:  A/B/C → 1 (LOS) / 0 (NLOS), D → 2/0, E/F → 4/0; 
  applied to first tap only; breakpoint ignored when `KF_linear >= 0`
- **`XPR_NLOS_linear`** — Overrides NLOS cross-polarization ratio (linear scale); default: XPR NLOS: 2 (3 dB)
- **`SF_std_dB_LOS`** — Overrides LOS shadow fading std in dB (applied when d < dBP); default: 3 dB
- **`SF_std_dB_NLOS`** — Overrides NLOS shadow fading std in dB (applied when d >= dBP); default: A/B → 4 dB, C/D → 5 dB, E/F → 6 dB
- **`dBP_m`** — Overrides breakpoint distance; default: A/B/C → 5 m, D → 10 m, E → 20 m, F → 30 m
- **`n_walls`** — Number of walls per user TGax models; `[n_users]` or `[1]`; default: 0
- **`wall_loss`** — Penetration loss for a single wall; TGax defines 5 or 7; default: 5

## Output:
- **`chan`**<br>
  Struct array of length `n_users` containing the channel data with the following fields:<br><br>
  | Field              | Description                                                              | Type                                  |
  | ------------------ | ------------------------------------------------------------------------ | ------------------------------------- |
  | `name`             | Channel name                                                             | String                                |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `tx_orientation`   |  Transmitter orientation, Euler angles (AP for downlink, STA for uplink) | Size: `[3, 1]` or `[3, n_snap]`       |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | Size: `[3, 1]` or `[3, n_snap]`       |
  | `coeff_re`         | Channel coefficients, real part                                          | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `delay`            | Propagation delays in seconds                                            | Size: `[n_rx, n_tx, n_path, n_snap]`  |
  | `path_gain`        | Path gain before antenna, linear scale                                   | Size: `[n_path, n_snap]`              |
  | `center_frequency` | Center Frequency in Hz                                                   | Scalar                                |

## See also:
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 24)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
    auto ant_tx = qd_mex_struct2arrayant(prhs[0]);
    auto ant_rx = qd_mex_struct2arrayant(prhs[1]);

    // Read model parameters
    std::string ChannelType = qd_mex_get_string(prhs[2]);
    double CarrierFreq_Hz = (nrhs < 4) ? 5.25e9 : qd_mex_get_scalar<double>(prhs[3], "CarrierFreq_GHz", 5.25e9);
    double tap_spacing_s = (nrhs < 5) ? 10.0e-9 : qd_mex_get_scalar<double>(prhs[4], "tap_spacing_ns", 10.0e-9);
    arma::uword n_users = (nrhs < 6) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[5], "n_users", 1);
    double observation_time = (nrhs < 7) ? 0.0 : qd_mex_get_scalar<double>(prhs[6], "observation_time", 0.0);
    double update_rate = (nrhs < 8) ? 1.0e-3 : qd_mex_get_scalar<double>(prhs[7], "update_rate", 1.0e-3);
    double speed_station_kmh = (nrhs < 9) ? 0.0 : qd_mex_get_scalar<double>(prhs[8], "speed_station_kmh", 0.0);
    double speed_env_kmh = (nrhs < 10) ? 1.2 : qd_mex_get_scalar<double>(prhs[9], "speed_env_kmh", 1.2);
    arma::vec Dist = (nrhs < 11) ? arma::vec{4.99} : qd_mex_get_Col<double>(prhs[10]);
    arma::uvec n_floors = (nrhs < 12) ? arma::uvec{0} : qd_mex_get_Col<arma::uword>(prhs[11]);
    bool uplink = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "uplink", false);
    arma::mat offset_angles = (nrhs < 14) ? arma::mat{} : qd_mex_get_Mat<double>(prhs[13]);
    arma::uword n_subpath = (nrhs < 15) ? 20 : qd_mex_get_scalar<arma::uword>(prhs[14], "n_subpath", 20);
    double Doppler_effect = (nrhs < 16) ? 50.0 : qd_mex_get_scalar<double>(prhs[15], "Doppler_effect", 50.0);
    arma::sword seed = (nrhs < 17) ? -1 : qd_mex_get_scalar<arma::sword>(prhs[16], "seed", -1);
    double KF_linear = (nrhs < 18) ? NAN : qd_mex_get_scalar<double>(prhs[17], "KF_linear", NAN);
    double XPR_NLOS_linear = (nrhs < 19) ? NAN : qd_mex_get_scalar<double>(prhs[18], "XPR_NLOS_linear", NAN);
    double SF_std_dB_LOS = (nrhs < 20) ? NAN : qd_mex_get_scalar<double>(prhs[19], "SF_std_dB_LOS", NAN);
    double SF_std_dB_NLOS = (nrhs < 21) ? NAN : qd_mex_get_scalar<double>(prhs[20], "SF_std_dB_NLOS", NAN);
    double dBP_m = (nrhs < 22) ? NAN : qd_mex_get_scalar<double>(prhs[21], "dBP_m", NAN);
    arma::uvec n_walls = (nrhs < 23) ? arma::uvec{0} : qd_mex_get_Col<arma::uword>(prhs[22]);
    double wall_loss = (nrhs < 24) ? 5.0 : qd_mex_get_scalar<double>(prhs[23], "wall_loss", 5.0);

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
                                                          dBP_m,
                                                          n_walls,
                                                          wall_loss));

    if (nlhs > 0)
        plhs[0] = qd_mex_channel2struct(chan);
}