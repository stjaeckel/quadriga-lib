// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_PLANAR
Calculate MIMO channel coefficients for planar wave paths

- Computes complex channel coefficients and delays for all TX/RX element pairs across `n_path` propagation paths.
- Interpolates antenna patterns for both arrays, accounting for element positions, orientation, and polarization.
- LOS path detection is distance-based (angles ignored).
- Polarization transfer matrix `M` must be normalized; it is the 2x2 Jones matrix (rows = RX polarisation V/H,
  columns = TX polarisation V/H), stored column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx),
  rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx), rows 7-8 = (H rx, H tx).
- `path_length` is used as given for the delay and phase of the phase centres (no geometry check).
- If `add_fake_los_path` is true, one output path is always added (output size `n_path+1`): slot 1 holds the LOS path,
  with zero power if no LOS path was found. LOS = the strongest path whose `path_length` is within 0.1 mm of the TX-RX
  distance; it is moved to slot 1 and its original slot (input index + 1) holds a zero-power path. All other paths
  keep their order, shifted by one slot.
- With 0 paths (all path inputs empty), the outputs are empty, `[n_rx, n_tx, 0]`; with `add_fake_los_path = true` they
  hold only this zero-power LOS path, with the LOS delays per antenna pair.
- Setting `center_freq = 0` disables phase calculation (delays still computed).
- `use_absolute_delays = false` subtracts the straight-line TX↔RX distance (between the phase centres) from all path
  lengths before converting to delay; per-element delays of pairs that are closer to each other than the phase
  centres are then negative, down to about -(half the TX array size + half the RX array size) / speed of light; [[quantize_delays]] rejects
  delays at or below `-tap_spacing / 2` (at 5 ns taps: half sizes adding up to more than about 0.75 m), use
  `use_absolute_delays = true` in that case.

## Usage:
```
[ coeff_re, coeff_im, delay, rx_Doppler ] = quadriga_lib.get_channels_planar( tx_array, rx_array, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    center_freq, use_absolute_delays, add_fake_los_path );
```

## Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [[arrayant_generate]]
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [[arrayant_generate]]
- **`aod`** — Departure azimuth angles; rad; `[n_path, 1]`
- **`eod`** — Departure elevation angles; rad; `[n_path, 1]`
- **`aoa`** — Arrival azimuth angles; rad; `[n_path, 1]`
- **`eoa`** — Arrival elevation angles; rad; `[n_path, 1]`
- **`path_gain`** — Path gains in linear scale; `[n_path, 1]`
- **`path_length`** — Total path lengths from TX to RX phase center; `[n_path, 1]`
- **`M`** — Polarization transfer matrix, Jones row order as above; `[8, n_path]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`center_freq`** — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: 0
- **`use_absolute_delays`** — If `true`, delays include the LOS component; Default: `false`
- **`add_fake_los_path`** — If `true`, always adds one output path; slot 1 holds the LOS path (zero power if none was
  found); a LOS path found in the input leaves a zero-power path in its original slot; Default: `false`


## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path(+1)]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path(+1)]`
- **`rx_Doppler`** — Doppler weights for moving RX; positive = moving toward path, negative = away; `[1, n_path(+1)]`

## See also:
- [[get_channels_spherical]] (spherical wave variant accounting for per-element angle differences)
- [[get_channels_ieee_indoor]] (for generating IEEE compliant channels using `get_channels_planar` internally)
- [[arrayant_generate]] (antenna array generator)
- [[baseband_freq_response]] (for calculating the frequency response)
- [[quantize_delays]] (for mapping delays to a fixed grid)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 13 || nrhs > 16)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
    auto ant_tx = qd_mex_struct2arrayant(prhs[0]);
    auto ant_rx = qd_mex_struct2arrayant(prhs[1]);

    // Parse other arguments
    const auto aod = qd_mex_get_Col<double>(prhs[2]);
    const auto eod = qd_mex_get_Col<double>(prhs[3]);
    const auto aoa = qd_mex_get_Col<double>(prhs[4]);
    const auto eoa = qd_mex_get_Col<double>(prhs[5]);
    const auto path_gain = qd_mex_get_Col<double>(prhs[6]);
    const auto path_length = qd_mex_get_Col<double>(prhs[7]);
    const auto M = qd_mex_get_Mat<double>(prhs[8]);
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[9], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[10], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[11], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[12], "rx_orientation", 3);
    double center_freq = (nrhs < 14) ? 0.0 : qd_mex_get_scalar<double>(prhs[13], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 16) ? false : qd_mex_get_scalar<bool>(prhs[15], "add_fake_los_path", false);

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);

    // Derived inputs
    arma::uword n_ports_tx = ant_tx.n_ports();
    arma::uword n_ports_rx = ant_rx.n_ports();
    arma::uword n_path = add_fake_los_path ? aod.n_elem + 1 : aod.n_elem;

    // Initialize output memory
    arma::cube coeff_re, coeff_im, delay;
    arma::vec rx_Doppler;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&coeff_re, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coeff_im, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path);
    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&rx_Doppler, n_path, true);

    // Call the C++ function
    // 0 paths: C++ returns the zero-power LOS path with add_fake_los_path, else empty outputs
    CALL_QD(quadriga_lib::get_channels_planar<double>(&ant_tx, &ant_rx,
                                                      Tx, Ty, Tz, Tb, Tt, Th, Rx, Ry, Rz, Rb, Rt, Rh,
                                                      &aod, &eod, &aoa, &eoa, &path_gain, &path_length, &M,
                                                      &coeff_re, &coeff_im, &delay,
                                                      center_freq, use_absolute_delays, add_fake_los_path,
                                                      &rx_Doppler));
}