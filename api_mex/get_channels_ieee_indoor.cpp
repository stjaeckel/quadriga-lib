// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# GET_CHANNELS_IEEE_INDOOR
Generate indoor MIMO channels of the IEEE TGn, TGac, TGah and TGax models

- Channels of the IEEE 802.11 indoor models A to F (TGn), with the TGac extensions (several users with offset angles,
  tap interpolation for shorter tap spacings), the floor loss of TGah below 1 GHz and the floor and wall loss of TGax
  above; one channel per user
- 2D model: azimuth angles only and motion in the horizontal plane; [[get_channels_planar]] computes the coefficients
  with the AP and station arrays
- Static channels (`observation_time = 0`, one snapshot) or time-evolving channels with the Doppler shifts of the
  station and the environment

## Usage:
```
chan = quadriga_lib.get_channels_ieee_indoor( ap_array, sta_array, channel_type, center_freq, ...
    tap_spacing, n_users, observation_time, update_interval, speed_station_kmh, speed_env_kmh, ...
    dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect, seed, ...
    kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint, n_walls, wall_loss_dB );
```

## Inputs:
- **`ap_array`**, **`sta_array`** — AP and station array, struct, fields in [[generate_antenna]]; `n_ap` and `n_sta`
  ports
- **`channel_type`** — Model: `'A'`, `'B'`, `'C'`, `'D'`, `'E'` or `'F'`
- **`center_freq`** — Carrier frequency in Hz, finite, above 100 MHz; default: 5.25e9
- **`tap_spacing`** — Tap spacing in s, 10 ns divided by a power of 2; default: 10e-9
- **`n_users`** — Number of users, >= 1; default: 1
- **`observation_time`** — Observation time in s, finite, >= 0; 0: one snapshot; default: 0
- **`update_interval`** — Time between snapshots in s, finite, > 0; default: 1e-3
- **`speed_station_kmh`** — Station speed in km/h, finite; default: 0
- **`speed_env_kmh`** — Speed of the environment in km/h, finite; TGn 1.2, TGac 0.089; default: 1.2
- **`dist`** — Horizontal AP-station distance in m per user, finite, > 0; one value for all users or `[n_users, 1]`;
  default: 4.99
- **`n_floors`** — Floors between AP and station per user, 0 to 4; one value or `[n_users, 1]`; default: 0
- **`uplink`** — Return the uplink, with the station as Tx; default: `false`
- **`offset_angles_deg`** — Offset angles in degrees per user, finite; rows: AoD LOS, AoD NLOS, AoA LOS, AoA NLOS;
  `[4, n_users]`; default: `[]` (0 for one user, the pseudo-random offsets of TGac for several)
- **`n_subpath`** — Sub-paths per tap, >= 1; default: 20
- **`doppler_effect`** — Special Doppler (Notes): the mains frequency in Hz for models D and E, the vehicle speed in
  km/h for model F; finite, >= 0; 0 disables it; default: 50
- **`seed`** — Seed of the random numbers; -1: the system random device; other values give the same channels on all
  platforms; default: -1
- **`kf`** — Ricean K-factor of the first tap, linear, finite, >= 0; default: `NaN` (model default, Notes)
- **`xpr_nlos`** — Cross-polarization ratio of the NLOS paths, linear, finite, > 0; default: `NaN` (model default)
- **`sf_std_los_dB`**, **`sf_std_nlos_dB`** — Standard deviation of the shadow fading in dB below and above the
  breakpoint distance, finite, >= 0; default: `NaN` (model default)
- **`dist_breakpoint`** — Breakpoint distance in m, finite, > 0; default: `NaN` (model default)
- **`n_walls`** — Walls between AP and station per user (TGax); one value or `[n_users, 1]`; default: 0
- **`wall_loss_dB`** — Loss of one wall in dB, finite; TGax: 5 or 7; default: 5

## Output:
- **`chan`** — Channels, struct array `[1, n_users]`, fields in [[hdf5_read_channel]]: `name` (`'user_<i>'`, 0-based),
  `center_freq`, the positions and orientations (station positions `[3, n_snap]`), `coeff_re`, `coeff_im`, `delay`
  `[n_rx, n_tx, n_path, n_snap]` (downlink `[n_sta, n_ap, n_path, n_snap]`) and `path_gain` `[n_path, n_snap]`: the
  path powers without antennas, with path loss and shadow fading, whose mean over the snapshots is the power of the
  model

## Notes:
- **Model defaults** (`NaN` in the override inputs), with the LOS shadow fading 3 dB and the NLOS XPR 2 (3 dB) for all
  models; below 1 GHz (TGah) both shadow fading values are 1 dB lower:

  | Model | A | B | C | D | E | F |
  | --- | --- | --- | --- | --- | --- | --- |
  | KF, linear | 1 | 1 | 1 | 2 | 4 | 4 |
  | Breakpoint distance in m | 5 | 5 | 5 | 10 | 20 | 30 |
  | NLOS shadow fading in dB | 4 | 4 | 5 | 5 | 6 | 6 |

- **LOS**: a user has a LOS path below the breakpoint distance on the floor of the AP, and always when `kf` is given;
  the KF splits the power of the first tap between the LOS path and the first NLOS cluster
- **Paths**: one per tap of the model tables with non-zero power, plus the LOS path; each tap sums `n_subpath` sub-paths
  with Laplacian angular spreads. Below 10 ns `tap_spacing`, the taps are interpolated (TGac)
- **Path loss** in dB at the distance $d$ = `dist`: $20 \lg(4\pi f_c / c) + 20 \lg d$ below the breakpoint distance
  $d_\mathrm{BP}$, $20 \lg(4\pi f_c / c) + 20 \lg d_\mathrm{BP} + 35 \lg(d / d_\mathrm{BP})$ beyond, plus the floor
  loss (TGah below 1 GHz, TGax above) and `n_walls` times `wall_loss_dB`. The shadow fading uses the LOS value below
  the breakpoint, else the NLOS value; below 1 GHz on another floor, the TGah floor values replace it
- **Geometry**: the AP at the origin, the station at the distance `dist` in the direction of the AoD LOS offset and
  3 m higher per floor; floors add loss but no delay. The delays are relative to the LOS delay
- **Time evolution**: `1 + floor(observation_time / update_interval)` snapshots, `update_interval` apart; each sub-path
  turns by $e^{j 2\pi f_D t}$ with the Doppler shift $f_D$ of the station speed plus a random environment Doppler
  (bell-shaped spectrum, TGn). The station moves along its heading, toward the AP for an AoA LOS offset of 0; moving
  toward the AP or a scatterer gives a positive $f_D$, as [[get_channels_spherical]] along the returned station
  positions
- **Special Doppler** (`doppler_effect` > 0): models D and E modulate three taps with fluorescent light at the
  harmonics of the mains frequency, model F adds the reflection of a moving vehicle to one tap (TGn)
- **Uplink**: the coefficients and delays are the transposed downlink ones (reciprocal channel, no conjugation, the same
  Doppler shift); the positions and orientations are swapped, the station is the Tx

## See also:
- [[get_channels_planar]] (computes the coefficients)
- [[hdf5_write_channel]] (store the channels)
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc">IEEE 802.11-03/940r4 - TGn Channel Models</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc">IEEE 802.11-09/0308r12 - TGac Channel Model Addendum</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx">IEEE 802.11-11/0968r4 - TGah Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx">IEEE 802.11-14/0882r4 - IEEE 802.11ax Channel Model</a>
- <a target="_blank" rel="noopener noreferrer" href="publications/11-25-2318-00-0ucm-a-modern-cpp-framework-for-the-ieee-indoor-channel-models.pdf">S. Jaeckel; "A modern C++ framework for the IEEE indoor channel models"; IEEE 802.11-25/2318r0; Tech. Rep., 2025</a>
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 24)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    auto ant_tx = qd_mex_struct2arrayant(prhs[0], false, false, "ap_array");
    auto ant_rx = qd_mex_struct2arrayant(prhs[1], false, false, "sta_array");

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

    std::vector<quadriga_lib::channel<double>> chan;

    // Empty inputs select the C++ defaults
    CALL_QD(chan = quadriga_lib::get_channels_ieee_indoor(ant_tx, ant_rx, channel_type, center_freq, tap_spacing, n_users,
                                                          observation_time, update_interval, speed_station_kmh, speed_env_kmh,
                                                          dist, n_floors, uplink, offset_angles_deg, n_subpath, doppler_effect,
                                                          seed, kf, xpr_nlos, sf_std_los_dB, sf_std_nlos_dB, dist_breakpoint,
                                                          n_walls, wall_loss_dB));

    plhs[0] = qd_mex_channel2struct(chan);
}
