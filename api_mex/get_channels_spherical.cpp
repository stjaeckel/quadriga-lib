// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_SPHERICAL
Calculate MIMO channel coefficients and delays for spherical wave propagation

- Computes complex channel coefficients and propagation delays for all TX/RX element pairs and paths,
  using spherical wave assumption with per-element phase and delay.
- Interpolates antenna patterns for both arrays, accounting for element positions and array orientation
  (bank/tilt/heading Euler angles).
- Polarization coupling is applied via the 8-row transfer matrix `M` (interleaved Re/Im for VV, VH, HV, HH components).
- If `center_frequency == 0`, phase calculation is disabled and only delays are computed.
- If `use_absolute_delays == false`, the minimum delay (LOS delay) is subtracted from all paths.
- If `add_fake_los_path == true`, a zero-power LOS path is prepended when no LOS path is detected.

## Usage:
```
[ coeff_re, coeff_im, delays, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( tx_array, rx_array, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    center_freq, use_absolute_delays, add_fake_los_path, use_avx2 );
```

## Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [[arrayant_generate]]
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [[arrayant_generate]]
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, interleaved (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** *(optional)* — Center frequency; set to `0` or skip/leave empty to skip phase computation
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** *(optional)* — If `true`, prepends a zero-power LOS path when none is present; Default: `false`
- **`use_avx2`** *(optional)* — If `true`, use AVX2 for antenna interpolation; faster, but less accurate;
  ignored when not supported; Default: `false`

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path]`

## See also:
- [[get_channels_planar]] (planar wave variant)
- [[get_channels_irs]] (for IRS-assisted communication)
- [[arrayant_generate]] (antenna array geneartor)
- [[baseband_freq_response]] (for calculating the frequency response)
- [[quantize_delays]] (for mapping delays to a fixed grid)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 11 || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
    auto ant_tx = qd_mex_struct2arrayant(prhs[0]);
    auto ant_rx = qd_mex_struct2arrayant(prhs[1]);

    // Parse other arguments
    const auto fbs_pos = qd_mex_get_Mat<double>(prhs[2]);
    const auto lbs_pos = qd_mex_get_Mat<double>(prhs[3]);
    const auto path_gain = qd_mex_get_Col<double>(prhs[4]);
    const auto path_length = qd_mex_get_Col<double>(prhs[5]);
    const auto M = qd_mex_get_Mat<double>(prhs[6]);
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[7], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[8], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[9], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[10], "rx_orientation", 3);
    double center_freq = (nrhs < 12) ? 0.0 : qd_mex_get_scalar<double>(prhs[11], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "add_fake_los_path", false);
    bool use_avx2 = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "use_avx2", false);

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx.n_ports();
    arma::uword n_ports_rx = ant_rx.n_ports();
    arma::uword n_path = fbs_pos.n_cols;

    // Initialize output memory
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = nullptr, *p_eod = nullptr, *p_aoa = nullptr, *p_eoa = nullptr;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&coeff_re, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coeff_im, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&aod, n_ports_rx, n_ports_tx, n_path), p_aod = &aod;
    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&eod, n_ports_rx, n_ports_tx, n_path), p_eod = &eod;
    if (nlhs > 5)
        plhs[5] = qd_mex_init_output(&aoa, n_ports_rx, n_ports_tx, n_path), p_aoa = &aoa;
    if (nlhs > 6)
        plhs[6] = qd_mex_init_output(&eoa, n_ports_rx, n_ports_tx, n_path), p_eoa = &eoa;

    // Call member function
    if (n_path)
        CALL_QD(quadriga_lib::get_channels_spherical<double>(&ant_tx, &ant_rx,
                                                             Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                             &fbs_pos, &lbs_pos, &path_gain, &path_length, &M,
                                                             &coeff_re, &coeff_im, &delay,
                                                             center_freq, use_absolute_delays, add_fake_los_path,
                                                             p_aod, p_eod, p_aoa, p_eoa, use_avx2));
}
