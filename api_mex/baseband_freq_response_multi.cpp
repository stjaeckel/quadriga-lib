// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"
#include "qd_helpers_shared.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# BASEBAND_FREQ_RESPONSE_MULTI
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

- Interpolates the complex channel coefficients from the input frequencies `freq` to the output carrier
  frequencies `freq_new` using SLERP (magnitude and unwrapped phase each linear), then applies the delay-induced
  phase `exp(-j·2π·freq_new·delay)` per output carrier in double precision
- One snapshot is a 4D array `[n_rx, n_tx, n_path, n_freq]`; a batch of snapshots is a cell array of such arrays
  (`coeff_re`, `coeff_im` and `delay` cell arrays of the same length), processed in parallel; the number of paths may
  differ between snapshots
- Single-frequency coefficients with absolute output carriers: `n_freq = 1` (a 3D array), `freq` the frequency the
  coefficients were generated at; with `remove_delay_phase = true` (the default) and coefficients that carry the delay
  phase at `freq`, as the channel generators return them, this gives the result of [[baseband_freq_response]] with
  `pilot_grid · bandwidth = freq_new - freq`
- `delay` is frequency-independent: `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`; of a 4D `delay` (as returned by
  [[get_channels_multifreq]]) only the first frequency is used
- With `remove_delay_phase`, the channels from [[get_channels_multifreq]] / [[get_channels_spherical]] must be
  generated with `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`,
  which gives wrong phases (magnitudes stay right)
- Output frequencies outside the range of `freq` use constant extrapolation from the nearest endpoint

## Usage:
```
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response_multi( coeff_re, coeff_im, delay, freq, freq_new, ...
    remove_delay_phase, i_snap, use_kernel );
```

## Inputs:
- **`coeff_re`** — Real part of the channel coefficients; `[n_rx, n_tx, n_path, n_freq]`, or a cell array of length
  `n_snap` with one such array per snapshot
- **`coeff_im`** — Imaginary part of the channel coefficients; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; an array `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` (a 4th dimension of length
  `n_freq` is allowed, only the first frequency is used), or a cell array of length `n_snap` with one such array per
  snapshot
- **`freq`** — Input sample frequencies in Hz, sorted ascending; `[n_freq, 1]`
- **`freq_new`** — Output carrier frequencies in Hz; `[n_carrier, 1]`
- **`remove_delay_phase`** — Remove the delay phase `exp(-j·2π·freq·delay)` from the coefficients before the
  interpolation and apply it again at `freq_new`; `true` for the output of the channel generators, `false` for pure
  envelope coefficients; default: `true`
- **`i_snap`** — Snapshots to process, 1-based; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

## Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`

## See also:
- [[baseband_freq_response]] (single-frequency coefficients, relative carrier positions)
- [[get_channels_multifreq]] (multi-frequency channel generator)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto freq = qd_mex_get_Col<double>(prhs[3], false, "freq");
    const auto freq_new = qd_mex_get_Col<double>(prhs[4], false, "freq_new");
    const bool remove_delay_phase = (nrhs < 6) ? true : qd_mex_get_scalar<bool>(prhs[5], "remove_delay_phase", true);
    const arma::uvec i_snap = (nrhs < 7) ? arma::uvec() : qd_mex_get_index_Col(prhs[6], "i_snap");
    const int use_kernel = (nrhs < 8) ? 0 : qd_mex_get_scalar<int>(prhs[7], "use_kernel", 0);

    // One snapshot as a 4D array (one cube per input frequency) or a batch as cell arrays; data of class double is
    // not copied. Only the first frequency of 'delay' is used: C++ reads only delay[0] of the n_freq entries.
    auto get_Coeff = [](const mxArray *a, const char *name)
    { return qd_mex_get_CubeVec<double>(a, false, name); };
    auto get_Delay = [&](const mxArray *a, const char *name)
    {
        auto d = qd_mex_get_CubeVec<double>(a, false, name);
        if (d.size() > 1 && d.size() != freq.n_elem)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'delay' must have a 4th dimension of 1 or the length of 'freq'.");
        d.resize(freq.n_elem);
        return d;
    };
    std::vector<std::vector<arma::cube>> coeff_re, coeff_im, delay;
    if (mxIsCell(prhs[0]) || mxIsCell(prhs[1]) || mxIsCell(prhs[2]))
    {
        const char *error = "Inputs 'coeff_re', 'coeff_im' and 'delay' must all be cell arrays.";
        coeff_re = qd_mex_read_cell<std::vector<arma::cube>>(prhs[0], "coeff_re", error, get_Coeff);
        coeff_im = qd_mex_read_cell<std::vector<arma::cube>>(prhs[1], "coeff_im", error, get_Coeff);
        delay = qd_mex_read_cell<std::vector<arma::cube>>(prhs[2], "delay", error, get_Delay);
    }
    else
    {
        coeff_re.push_back(get_Coeff(prhs[0], "coeff_re"));
        coeff_im.push_back(get_Coeff(prhs[1], "coeff_im"));
        delay.push_back(get_Delay(prhs[2], "delay"));
    }

    // Output [n_rx, n_tx, n_carrier, n_out]: the selected snapshots must share n_rx and n_tx; C++ checks the rest
    arma::uvec snap;
    CALL_QD(snap = qd_snapshot_list(i_snap, coeff_re.size()));
    const arma::uword n_rx = (snap.is_empty() || coeff_re[snap[0]].empty()) ? 0 : coeff_re[snap[0]][0].n_rows;
    const arma::uword n_tx = (snap.is_empty() || coeff_re[snap[0]].empty()) ? 0 : coeff_re[snap[0]][0].n_cols;
    for (arma::uword s : snap)
        if (!coeff_re[s].empty() && (coeff_re[s][0].n_rows != n_rx || coeff_re[s][0].n_cols != n_tx))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'coeff_re' must have the n_rx and n_tx of the first selected snapshot.");

    std::vector<arma::cube> hmat_re, hmat_im;
    plhs[0] = qd_mex_init_output(&hmat_re, n_rx, n_tx, freq_new.n_elem, snap.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&hmat_im, n_rx, n_tx, freq_new.n_elem, snap.n_elem);

    CALL_QD(quadriga_lib::baseband_freq_response_multi_vec<double>(coeff_re, coeff_im, delay, freq, freq_new,
                                                                   &hmat_re, (nlhs > 1) ? &hmat_im : nullptr, nullptr,
                                                                   remove_delay_phase, i_snap, use_kernel));
}
