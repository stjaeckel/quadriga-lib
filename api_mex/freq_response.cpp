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
# FREQ_RESPONSE
Frequency response of a MIMO channel at the given subcarriers

- The frequency response of one snapshot at the subcarriers $k$: $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$ with the
  offset $f_k$ = `bandwidth` · `pilot_grid(k)` from the carrier
- One snapshot is a 3D array; several snapshots are cell arrays with one array per snapshot (`coeff_re`,
  `coeff_im` and `delay` of the same length), computed in parallel; the number of paths may differ between them
- A channel struct of [[hdf5_read_channel]] holds the snapshots on the 4th dimension, zero-padded;
  `num2cell(ch.coeff_re, [1 2 3])`, and the same for `coeff_im` and `delay`, gives the cell arrays (the zero-padded
  paths do not change the result)
- For a multi-frequency channel and absolute carrier frequencies, use [[freq_response_mf]]
- Precision: the phase is reduced to one period in double precision; sin / cos, coefficients and sums are single
  precision

## Usage:
```
[ hmat_re, hmat_im ] = quadriga_lib.freq_response( coeff_re, coeff_im, delay, pilot_grid, bandwidth, ...
    i_snap, use_kernel );
```

## Inputs:
- **`coeff_re`** / **`coeff_im`** — Real and imaginary part of the coefficients; `[n_rx, n_tx, n_path]`, or cell arrays
  of length `n_snap` with one such array per snapshot
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`, or a cell array as `coeff_re`
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `[n_carrier, 1]`
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** — 1-based indices of the snapshots to compute; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable);
  default: 0

## Outputs:
- **`hmat_re`** / **`hmat_im`** — Real and imaginary part of the frequency response; `[n_rx, n_tx, n_carrier, n_out]`;
  the selected snapshots must have equal `n_rx` and `n_tx`

## See also:
- [[freq_response_mf]] (multi-frequency channel, absolute carrier frequencies)
- [[get_channels_spherical]] (coefficients and delays)
- [[quantize_delays]] (delays on a tap grid, for time-domain processing)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // One snapshot as a 3D array or a batch as cell arrays; data of class double is not copied
    auto get_Cube = [](const mxArray *a, const char *name)
    {
        qd_mex_check_input(a, name, 3);
        return qd_mex_get_Cube<double>(a, false, name);
    };
    std::vector<arma::cube> coeff_re, coeff_im, delay;
    if (mxIsCell(prhs[0]) || mxIsCell(prhs[1]) || mxIsCell(prhs[2]))
    {
        const char *error = "Inputs 'coeff_re', 'coeff_im' and 'delay' must all be cell arrays.";
        coeff_re = qd_mex_read_cell<arma::cube>(prhs[0], "coeff_re", error, get_Cube);
        coeff_im = qd_mex_read_cell<arma::cube>(prhs[1], "coeff_im", error, get_Cube);
        delay = qd_mex_read_cell<arma::cube>(prhs[2], "delay", error, get_Cube);
    }
    else
    {
        coeff_re.push_back(get_Cube(prhs[0], "coeff_re"));
        coeff_im.push_back(get_Cube(prhs[1], "coeff_im"));
        delay.push_back(get_Cube(prhs[2], "delay"));
    }

    const auto pilot_grid = qd_mex_get_Col<double>(prhs[3], false, "pilot_grid");
    const double bandwidth = qd_mex_get_scalar<double>(prhs[4], "bandwidth");
    const arma::uvec i_snap = (nrhs < 6) ? arma::uvec() : qd_mex_get_index_Col(prhs[5], "i_snap");
    const int use_kernel = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "use_kernel", 0);

    // Output [n_rx, n_tx, n_carrier, n_out]: the selected snapshots must share n_rx and n_tx; C++ checks the rest
    arma::uvec snap;
    CALL_QD(snap = qd_snapshot_list(i_snap, coeff_re.size()));
    const arma::uword n_rx = snap.is_empty() ? 0 : coeff_re[snap[0]].n_rows;
    const arma::uword n_tx = snap.is_empty() ? 0 : coeff_re[snap[0]].n_cols;
    for (arma::uword s : snap)
        if (coeff_re[s].n_rows != n_rx || coeff_re[s].n_cols != n_tx)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "All snapshots of 'coeff_re' must have the same number of rows and columns.");

    std::vector<arma::cube> hmat_re, hmat_im;
    plhs[0] = qd_mex_init_output(&hmat_re, n_rx, n_tx, pilot_grid.n_elem, snap.n_elem);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&hmat_im, n_rx, n_tx, pilot_grid.n_elem, snap.n_elem);

    CALL_QD(quadriga_lib::freq_response<double>(coeff_re, coeff_im, delay, pilot_grid, bandwidth,
                                                &hmat_re, (nlhs > 1) ? &hmat_im : nullptr, nullptr,
                                                i_snap, use_kernel));
}
