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
- One snapshot is a 3D array; several snapshots are cell arrays with one array per snapshot (`coeff` and `delay`
  of the same length), computed in parallel; the number of paths may differ between them
- A channel struct of [[hdf5_read_channel]] holds the snapshots on the 4th dimension, zero-padded;
  `num2cell(ch.coeff, [1 2 3])`, and the same for `delay`, gives the cell arrays (the zero-padded paths do not change
  the result)
- For a multi-frequency channel and absolute carrier frequencies, use [[freq_response_mf]]
- Precision: the phase is reduced to one period in double precision; sin / cos, coefficients and sums are single
  precision

## Usage:
```
hmat = quadriga_lib.freq_response( coeff, delay, pilot_grid, bandwidth, i_snap, use_kernel );
```

## Inputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path]`, or a cell array of length `n_snap` with one such array
  per snapshot
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`, or a cell array as `coeff`
- **`pilot_grid`** — Subcarrier positions relative to the bandwidth, finite: 0 = carrier, 1 = carrier + `bandwidth`;
  `[n_carrier, 1]`
- **`bandwidth`** — Bandwidth in Hz, finite and >= 0
- **`i_snap`** — 1-based indices of the snapshots to compute; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable);
  default: 0

## Outputs:
- **`hmat`** — Complex frequency response; `[n_rx, n_tx, n_carrier, n_out]`; the selected snapshots must have equal
  `n_rx` and `n_tx`

## See also:
- [[freq_response_mf]] (multi-frequency channel, absolute carrier frequencies)
- [[get_channels_spherical]] (coefficients and delays)
- [[quantize_delays]] (delays on a tap grid, for time-domain processing)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // One snapshot as a 3D array or a batch as cell arrays; the coefficients are split into real and imaginary part
    // (copy), delays of class double are not copied
    std::vector<arma::cube> coeff_re, coeff_im, delay;
    auto get_coeff = [&](const mxArray *a, const char *name)
    {
        coeff_re.emplace_back(), coeff_im.emplace_back();
        if (a == nullptr)
            return;
        qd_mex_check_input(a, name, 3, true);
        qd_mex_get_complex_Cube<double>(a, name, coeff_re.back(), coeff_im.back());
    };
    auto get_delay = [](const mxArray *a, const char *name)
    {
        qd_mex_check_input(a, name, 3);
        return qd_mex_get_Cube<double>(a, false, name);
    };
    if (mxIsCell(prhs[0]) || mxIsCell(prhs[1]))
    {
        const char *error = "Inputs 'coeff' and 'delay' must both be cell arrays.";
        if (!mxIsCell(prhs[0]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error);
        for (size_t i = 0, n = mxGetNumberOfElements(prhs[0]); i < n; ++i)
            get_coeff(mxGetCell(prhs[0], (mwIndex)i), "coeff");
        delay = qd_mex_read_cell<arma::cube>(prhs[1], "delay", error, get_delay);
    }
    else
    {
        get_coeff(prhs[0], "coeff");
        delay.push_back(get_delay(prhs[1], "delay"));
    }

    const auto pilot_grid = qd_mex_get_Col<double>(prhs[2], false, "pilot_grid");
    const double bandwidth = qd_mex_get_scalar<double>(prhs[3], "bandwidth");
    const arma::uvec i_snap = (nrhs < 5) ? arma::uvec() : qd_mex_get_index_Col(prhs[4], "i_snap");
    const int use_kernel = (nrhs < 6) ? 0 : qd_mex_get_scalar<int>(prhs[5], "use_kernel", 0);

    // Output [n_rx, n_tx, n_carrier, n_out]: the selected snapshots must share n_rx and n_tx; C++ checks the rest
    arma::uvec snap;
    CALL_QD(snap = qd_snapshot_list(i_snap, coeff_re.size()));
    const arma::uword n_rx = snap.is_empty() ? 0 : coeff_re[snap[0]].n_rows;
    const arma::uword n_tx = snap.is_empty() ? 0 : coeff_re[snap[0]].n_cols;
    for (arma::uword s : snap)
        if (coeff_re[s].n_rows != n_rx || coeff_re[s].n_cols != n_tx)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "All snapshots of 'coeff' must have the same number of rows and columns.");

    // C++ writes the complex result into the MATLAB array
    std::vector<arma::cx_cube> hmat;
    plhs[0] = qd_mex_init_output(&hmat, n_rx, n_tx, pilot_grid.n_elem, snap.n_elem);

    CALL_QD(quadriga_lib::freq_response<double>(coeff_re, coeff_im, delay, pilot_grid, bandwidth,
                                                nullptr, nullptr, &hmat, i_snap, use_kernel));
}
