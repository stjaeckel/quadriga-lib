// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"
#include "qd_helpers_shared.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# FREQ_RESPONSE_MF
Frequency response of a MIMO channel from coefficients sampled at several frequencies

- The frequency response of one snapshot of a multi-frequency channel at the absolute carrier frequencies
  $f_k$ = `freq_new(k)`: the coefficients of each path are interpolated from the input frequencies `freq` to $f_k$
  and the delay phase is applied there, $H_k = \sum_l \tilde a_l(f_k) \, e^{-j 2\pi f_k \tau_l}$
- Interpolation between the two neighbouring input frequencies: amplitude linear, phase along the shorter arc
  (SLERP); constant outside the range of `freq`
- `remove_delay_phase = true` interpolates the envelope $a_l(f_i) \, e^{j 2\pi f_i \tau_l}$ instead of $a_l(f_i)$, so
  that the delay phase the channel generators include is not interpolated
- One snapshot is a 4D array `[n_rx, n_tx, n_path, n_freq]`; several snapshots are cell arrays with one array per
  snapshot (`coeff` and `delay` of the same length), computed in parallel; the number of paths may differ
  between them

## Usage:
```
hmat = quadriga_lib.freq_response_mf( coeff, delay, freq, freq_new, remove_delay_phase, i_snap, use_kernel );
```

## Inputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path, n_freq]`, or a cell array of length `n_snap` with one
  such array per snapshot
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` (a 4th dimension of length `n_freq` is
  allowed, only the first frequency is used), or a cell array as `coeff`
- **`freq`** — Input frequencies in Hz, finite, strictly ascending; `[n_freq, 1]`
- **`freq_new`** — Output carrier frequencies in Hz, finite; `[n_carrier, 1]`
- **`remove_delay_phase`** — Remove the delay phase from the coefficients before the interpolation: `true` for the
  coefficients of [[get_channels_multifreq]] and [[get_channels_spherical]], `false` for coefficients without delay
  phase; default: `true`
- **`i_snap`** — 1-based indices of the snapshots to compute; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

## Outputs:
- **`hmat`** — Complex frequency response; `[n_rx, n_tx, n_carrier, n_out]`; the selected snapshots must have equal
  `n_rx` and `n_tx`

## Notes:
- With `remove_delay_phase = true`, generate the channels with `use_absolute_delays = true`: their coefficients carry
  the phase of the absolute path length also when `delay` holds relative delays. With relative delays the phase of
  the Tx-Rx distance stays in the coefficients and is interpolated across `freq`, which gives wrong phases (the
  magnitudes stay right) unless the spacing of `freq` is far below $c$ / (Tx-Rx distance)
- One input frequency (3D arrays): the coefficients apply at all carriers; with `remove_delay_phase = true` the
  result is that of [[freq_response]] with `bandwidth` · `pilot_grid` = `freq_new - freq`
- Precision: the delay phase is reduced to one period in double precision (and removed in double precision with
  `remove_delay_phase`); sin / cos of the output phase and the interpolation are single
  precision, the sums double precision

## See also:
- [[freq_response]] (single-frequency coefficients, carriers relative to the bandwidth)
- [[get_channels_multifreq]] (multi-frequency channel)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto freq = qd_mex_get_Col<double>(prhs[2], false, "freq");
    const auto freq_new = qd_mex_get_Col<double>(prhs[3], false, "freq_new");
    const bool remove_delay_phase = (nrhs < 5) ? true : qd_mex_get_scalar<bool>(prhs[4], "remove_delay_phase", true);
    const arma::uvec i_snap = (nrhs < 6) ? arma::uvec() : qd_mex_get_index_Col(prhs[5], "i_snap");
    const int use_kernel = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "use_kernel", 0);

    // One snapshot as a 4D array (one cube per input frequency) or a batch as cell arrays; the coefficients are split
    // into real and imaginary part (copy), delays of class double are not copied. Only the first frequency of 'delay'
    // is used: C++ reads only delay[0] of the n_freq entries.
    std::vector<std::vector<arma::cube>> coeff_re, coeff_im, delay;
    auto get_coeff = [&](const mxArray *a, const char *name)
    {
        coeff_re.emplace_back(), coeff_im.emplace_back();
        if (a != nullptr)
            qd_mex_get_complex_CubeVec<double>(a, name, coeff_re.back(), coeff_im.back());
    };
    auto get_Delay = [&](const mxArray *a, const char *name)
    {
        auto d = qd_mex_get_CubeVec<double>(a, false, name);
        if (d.size() > 1 && d.size() != freq.n_elem)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'delay' must have a 4th dimension of 1 or the length of 'freq'.");
        d.resize(freq.n_elem);
        return d;
    };
    if (mxIsCell(prhs[0]) || mxIsCell(prhs[1]))
    {
        const char *error = "Inputs 'coeff' and 'delay' must both be cell arrays.";
        if (!mxIsCell(prhs[0]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error);
        for (size_t i = 0, n = mxGetNumberOfElements(prhs[0]); i < n; ++i)
            get_coeff(mxGetCell(prhs[0], (mwIndex)i), "coeff");
        delay = qd_mex_read_cell<std::vector<arma::cube>>(prhs[1], "delay", error, get_Delay);
    }
    else
    {
        get_coeff(prhs[0], "coeff");
        delay.push_back(get_Delay(prhs[1], "delay"));
    }

    // Output [n_rx, n_tx, n_carrier, n_out]: the selected snapshots must share n_rx and n_tx; C++ checks the rest
    arma::uvec snap;
    CALL_QD(snap = qd_snapshot_list(i_snap, coeff_re.size()));
    const arma::uword n_rx = (snap.is_empty() || coeff_re[snap[0]].empty()) ? 0 : coeff_re[snap[0]][0].n_rows;
    const arma::uword n_tx = (snap.is_empty() || coeff_re[snap[0]].empty()) ? 0 : coeff_re[snap[0]][0].n_cols;
    for (arma::uword s : snap)
        if (!coeff_re[s].empty() && (coeff_re[s][0].n_rows != n_rx || coeff_re[s][0].n_cols != n_tx))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'coeff' must have the n_rx and n_tx of the first selected snapshot.");

    // C++ writes the complex result into the MATLAB array
    std::vector<arma::cx_cube> hmat;
    plhs[0] = qd_mex_init_output(&hmat, n_rx, n_tx, freq_new.n_elem, snap.n_elem);

    CALL_QD(quadriga_lib::freq_response_mf<double>(coeff_re, coeff_im, delay, freq, freq_new, nullptr, nullptr, &hmat,
                                                   remove_delay_phase, i_snap, use_kernel));
}
