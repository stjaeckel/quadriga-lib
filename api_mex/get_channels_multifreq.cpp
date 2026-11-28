// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# GET_CHANNELS_MULTIFREQ
Calculate MIMO channel coefficients and delays for spherical wave propagation at several frequencies

- As [[get_channels_spherical]], at the output frequencies `freq_new`, with multi-frequency arrays (frequency-dependent
  patterns and port weights), path gains and Jones matrices; for radio waves and, with `propagation_speed` = 343 m/s
  and scalar `polmat`, for sound
- The geometry (element-pair path lengths, angles, LOS path) is computed once; at each output frequency, the patterns
  and port weights of the arrays and `path_gain` and `polmat` are interpolated between their two neighbouring entries
- The coefficients follow the formula of [[get_channels_spherical]] with $f_c$ = `freq_new(k)` and $c$ =
  `propagation_speed`; the frequency is the 4th dimension of the outputs, a multi-frequency channel

## Usage:
```
[ coeff, delay ] = quadriga_lib.get_channels_multifreq( tx_array, rx_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, path_length, polmat, ...
    freq, freq_new, use_absolute_delays, add_fake_los_path, propagation_speed, use_kernel );
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx multi-frequency array, struct array `[1, n]` with one entry per frequency
  in any order, fields in [[generate_antenna]], valid, at least one entry; `n_tx` and `n_rx` ports
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m, finite; 3 elements
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientation, Euler angles (bank, tilt, heading) in rad,
  finite; 3 elements
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m; `[3, n_path]`
- **`path_gain`** — Path gains at the frequencies `freq`, linear power; `[n_path, n_freq]`
- **`path_length`** — Path lengths in m between the Tx and Rx phase centres, as in [[get_channels_spherical]];
  `[n_path, 1]`
- **`polmat`** — Jones matrices at the frequencies `freq` (section description): `[8, n_path, n_freq]`, or
  `[2, n_path, n_freq]` for a scalar coefficient (real and imaginary part, sound)
- **`freq`** — Sample frequencies of `path_gain` and `polmat` in Hz, in any order; `[n_freq, 1]`, at least one
- **`freq_new`** — Output frequencies in Hz; `[n_freq_new, 1]`, at least one
- **`use_absolute_delays`** — Keep the delay of the Tx-Rx distance in all delays; `false`: delays relative to it;
  default: `false`
- **`add_fake_los_path`** — Put the LOS path into an added slot 1, with zero power if there is none; default: `false`
- **`propagation_speed`** — Wave speed in m/s, > 0; default: 299792458
- **`use_kernel`** — Kernel of the antenna interpolation: 0 = auto (GENERIC in this version), 1 = GENERIC, 2 = AVX2
  (single-precision arithmetic; error if unavailable); default: 0

## Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_out, n_freq_new]`, `n_out = n_path`, or `n_path + 1` with
  `add_fake_los_path`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_out, n_freq_new]`

## Notes:
- **Interpolation**: `path_gain` linear; patterns, port weights and `polmat` per complex value, the amplitude linear and
  the phase along the shorter arc (linear for nearly opposite phases); constant outside the range of each grid
- **Scalar `polmat`**: only the $F_\theta$ patterns are used,
  $a_l = \sqrt{g_l} \, F_{\theta,r} \, m_l \, F_{\theta,t} \, e^{-j 2\pi f d / c}$ with the complex scalar $m_l$
- **Paths**: NLOS path length, LOS path, `add_fake_los_path` and 0 paths as in [[get_channels_spherical]]
- **Relative delays**: negative delays as in [[get_channels_spherical]], with $c$ = `propagation_speed`; for sound, this
  affects practically every array, so use absolute delays for [[quantize_delays]]
- **Delay phase**: the coefficients carry the phase of the absolute path length also with relative delays;
  [[freq_response_mf]] with `remove_delay_phase = true` therefore needs `use_absolute_delays = true`

## See also:
- [[get_channels_spherical]] (one frequency)
- [[freq_response_mf]] (frequency response of a multi-frequency channel)
- [[join_arrays]] (multi-frequency arrays)
- [[generate_speaker]] (loudspeaker arrays)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 13 || nrhs > 17)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    auto ant_tx = qd_mex_struct2arrayant_multi(prhs[0], false, false, "tx_array");
    auto ant_rx = qd_mex_struct2arrayant_multi(prhs[1], false, false, "rx_array");

    if (ant_tx.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'tx_array' cannot be empty.");
    if (ant_rx.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'rx_array' cannot be empty.");

    const auto fbs_pos = qd_mex_get_Mat<double>(prhs[6], false, "fbs_pos");
    const auto lbs_pos = qd_mex_get_Mat<double>(prhs[7], false, "lbs_pos");
    const auto path_gain = qd_mex_get_Mat<double>(prhs[8], false, "path_gain");
    const auto path_length = qd_mex_get_Col<double>(prhs[9], false, "path_length");
    const auto polmat = qd_mex_get_Cube<double>(prhs[10], false, "polmat");
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[2], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[3], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[4], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[5], "rx_orientation", 3);
    const auto freq = qd_mex_get_Col<double>(prhs[11], false, "freq");
    const auto freq_new = qd_mex_get_Col<double>(prhs[12], false, "freq_new");
    bool use_absolute_delays = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "use_absolute_delays", false);
    bool add_fake_los_path = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "add_fake_los_path", false);
    double propagation_speed = (nrhs < 16) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[15], "propagation_speed", 299792458.0);
    const int use_kernel = (nrhs < 17) ? 0 : qd_mex_get_scalar<int>(prhs[16], "use_kernel", 0);

    arma::uword n_ports_tx = ant_tx[0].n_ports();
    arma::uword n_ports_rx = ant_rx[0].n_ports();
    arma::uword n_path = add_fake_los_path ? fbs_pos.n_cols + 1 : fbs_pos.n_cols;
    arma::uword n_freq_new = freq_new.n_elem;

    // 4D outputs, one cube per entry of freq_new; C++ writes the complex coefficients into the MATLAB array
    std::vector<arma::cx_cube> coeff;
    std::vector<arma::cube> coeff_re, coeff_im, delay;

    plhs[0] = qd_mex_init_output(&coeff, n_ports_rx, n_ports_tx, n_path, n_freq_new);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&delay, n_ports_rx, n_ports_tx, n_path, n_freq_new);

    // 0 paths: C++ returns the zero-power LOS path with add_fake_los_path, else empty outputs
    CALL_QD(quadriga_lib::get_channels_multifreq<double>(ant_tx, ant_rx,
                                                         tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                         fbs_pos, lbs_pos, path_gain, path_length, polmat,
                                                         freq, freq_new,
                                                         coeff_re, coeff_im, delay,
                                                         use_absolute_delays, add_fake_los_path,
                                                         propagation_speed, &coeff, use_kernel));
}
