// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_MULTIFREQ
Compute channel coefficients for spherical waves across multiple frequencies

- Multi-frequency extension of [[get_channels_spherical]] with frequency-dependent antenna patterns, 
  path gains, and Jones matrices
- Geometry (angles, element delays, LOS detection) is computed once and reused across all output frequencies
- Aligns four frequency grids: TX array (from each `tx_array.center_freq`), RX array, input
  samples (`freq_in`), and output (`freq_out`)
- TX/RX patterns are interpolated per output frequency via SLERP with linear fallback
- `path_gain` is interpolated linearly; `M` is interpolated via SLERP per complex entry pair to preserve phase
- Extrapolation clamps to the nearest frequency entry on all four grids
- `propagation_speed` supports EM (speed of light, default) and acoustic (~343 m/s) simulations
- `M` accepts 8 rows (full polarimetric: ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH) or 2 rows
  (scalar pressure: ReVV, ImVV only)
- Coupling matrices are interpolated across frequencies (SLERP for complex pairs), identical to
  antenna pattern handling
- `n_path_out = n_path + 1` if `add_fake_los_path` else `n_path`

## Usage:
```
[ coeff_re, coeff_im, delay ] = quadriga_lib.get_channels_multifreq( tx_array, rx_array, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, ...
    freq_in, freq_out, use_absolute_delays, add_fake_los_path, propagation_speed );
```

## Inputs:
- **`tx_array`** — Multi-frequency TX arrayant struct array; one entry per input frequency, see [[arrayant_generate]]
- **`rx_array`** — Multi-frequency RX arrayant struct array; one entry per input frequency, see [[arrayant_generate]]
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Linear-scale path gains per input frequency; `[n_path, n_freq_in]`
- **`path_length`** — Absolute TX-to-RX path lengths; `[n_path, 1]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq_in]` (full pol, interleaved Re/Im for VV, VH, HV, HH) or `[2, n_path, n_freq_in]` (scalar pressure: ReVV, ImVV only)
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`freq_in`** — Input sample frequencies for `path_gain` and `M`; `[n_freq_in, 1]`
- **`freq_out`** — Target output frequencies; `[n_freq_out, 1]`
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component; default: `false`
- **`add_fake_los_path`** *(optional)* — If `true`, prepends a zero-power LOS path when none is present; default: `false`
- **`propagation_speed`** *(optional)* — Wave speed in m/s; use ~343.0 for acoustics; default: `299792458.0`

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_out]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_out, n_freq_out]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_out, n_freq_out]`

## See also:
- [[get_channels_spherical]] (single-frequency equivalent)
- [[generate_speaker]] (acoustic source construction)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 13 || nrhs > 16)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse multi-frequency antenna struct arrays
    auto ant_tx = qd_mex_struct2arrayant_multi(prhs[0]);
    auto ant_rx = qd_mex_struct2arrayant_multi(prhs[1]);

    if (ant_tx.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "tx_array must not be empty.");
    if (ant_rx.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "rx_array must not be empty.");

    // Parse other arguments
    const auto fbs_pos = qd_mex_get_Mat<double>(prhs[2]);
    const auto lbs_pos = qd_mex_get_Mat<double>(prhs[3]);
    const auto path_gain = qd_mex_get_Mat<double>(prhs[4]);
    const auto path_length = qd_mex_get_Col<double>(prhs[5]);
    const auto M = qd_mex_get_Cube<double>(prhs[6]);
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[7], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[8], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[9], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[10], "rx_orientation", 3);
    const auto freq_in = qd_mex_get_Col<double>(prhs[11]);
    const auto freq_out = qd_mex_get_Col<double>(prhs[12]);
    bool use_absolute_delays = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "add_fake_los_path", false);
    double propagation_speed = (nrhs < 16) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[15], "propagation_speed", 299792458.0);

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);

    // Derived sizes
    arma::uword n_ports_tx = ant_tx[0].n_ports();
    arma::uword n_ports_rx = ant_rx[0].n_ports();
    arma::uword n_path = add_fake_los_path ? fbs_pos.n_cols + 1 : fbs_pos.n_cols;
    arma::uword n_freq_out = freq_out.n_elem;

    // Initialize 4D output memory
    std::vector<arma::cube> coeff_re, coeff_im, delay;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&coeff_re, n_ports_rx, n_ports_tx, n_path, n_freq_out);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coeff_im, n_ports_rx, n_ports_tx, n_path, n_freq_out);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path, n_freq_out);

    // Call C++ function
    if ((add_fake_los_path && n_path > 1) || (!add_fake_los_path && n_path > 0))
        CALL_QD(quadriga_lib::get_channels_multifreq<double>(ant_tx, ant_rx,
                                                             Tx, Ty, Tz, Tb, Tt, Th,
                                                             Rx, Ry, Rz, Rb, Rt, Rh,
                                                             fbs_pos, lbs_pos, path_gain, path_length, M,
                                                             freq_in, freq_out,
                                                             coeff_re, coeff_im, delay,
                                                             use_absolute_delays, add_fake_los_path,
                                                             propagation_speed));
}