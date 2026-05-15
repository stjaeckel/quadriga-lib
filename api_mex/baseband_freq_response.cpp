// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# BASEBAND_FREQ_RESPONSE
Compute the baseband frequency response of a MIMO channel

- Three dispatch modes depending on the dimensionality of coeff_re / coeff_im / delay  and the
  presence of `i_snap` — see the table below
- Single-frequency mode: computes the frequency-domain channel matrix at the carrier positions via
  DFT over time-domain path coefficients and delays
- Multi-frequency mode: interpolates complex channel coefficients from the input frequency grid
  `center_freq` to the output grid using SLERP, then applies delay-induced phase rotation per output
  carrier; only the first frequency slab of `delay` is used
- Multi-snapshot mode: applies the single-frequency backend in parallel across snapshots via OpenMP;
  `i_snap` (1-based MATLAB indices) optionally selects a subset
- Carrier positions are specified one of two ways: as a normalized `pilot_grid` paired with `bandwidth`
  (where `0.0` corresponds to `center_freq(1)` and `1.0` to `center_freq(1) + bandwidth`), or as absolute
  frequencies via `carrier_freq`; supplying both pairs is an error
- Multi-frequency inputs always require `center_freq`; if `carrier_freq` is omitted it is derived as
  `center_freq(1) + pilot_grid · bandwidth`
- Multi-snapshot inputs require `center_freq` to be omitted or scalar (no combined multi-freq + multi-snap)
- Single-frequency inputs ignore `center_freq` when `pilot_grid` + `bandwidth` are given
- `delay` supports broadcasting: shape `[1, 1, n_path, ...]` applies the same delays to all RX/TX pairs
- Internal arithmetic is single-precision; uses AVX2 where supported; double inputs are narrowed to
  float internally, results widened back

## Usage:
```
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, ...
    pilot_grid, bandwidth, center_freq, carrier_freq, i_snap );
```

## Dispatch modes:
| Mode             | Triggered by                             | `coeff_re` shape                  | Output shape                       |
|------------------|------------------------------------------|-----------------------------------|------------------------------------|
| Single-frequency | 3D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path]`            | `[n_rx, n_tx, n_carrier]`          |
| Multi-frequency  | 4D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path, n_freq]`    | `[n_rx, n_tx, n_carrier]`          |
| Multi-snapshot   | `i_snap` supplied (may be 0 for all)     | `[n_rx, n_tx, n_path, n_snap]`    | `[n_rx, n_tx, n_carrier, n_out]`   |

## Inputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path]` (single-freq) or
  `[n_rx, n_tx, n_path, n_freq]` (multi-freq) or `[n_rx, n_tx, n_path, n_snap]` (multi-snap)
- **`coeff_im`** — Imaginary part of channel coefficients; same shape as `coeff_re`
- **`delay`** — Path delays in seconds; same shape as `coeff_re`, optionally broadcast over RX/TX with
  shape `[1, 1, n_path]` or `[1, 1, n_path, ...]`
- **`pilot_grid`** *(optional)* — Normalized sub-carrier positions; `0.0` = center,
  `1.0` = center + bandwidth; must be paired with `bandwidth`; `[n_carrier, 1]`
- **`bandwidth`** *(optional)* — Total baseband bandwidth in Hz; must be paired with `pilot_grid`
- **`center_freq`** *(optional)* — Input sample frequencies; required for multi-frequency inputs;
  length must equal the 4th dimension of `coeff_re`; for multi-snap must be omitted or scalar; `[n_freq, 1]`
- **`carrier_freq`** *(optional)* — Absolute output carrier frequencies in Hz; cannot be combined
  with `pilot_grid` + `bandwidth`; `[n_carrier, 1]`
- **`i_snap`** *(optional)* — Triggers multi-snap mode. Scalar `0` processes all snapshots;
  a positive vector of 1-based indices processes the selected subset. Omitting this argument
  or passing `[]` keeps the function in single/multi-frequency mode.

## Outputs:
- **`hmat_re`** *(optional)* — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
  (single/multi-freq) or `[n_rx, n_tx, n_carrier, n_out]` (multi-snap)
- **`hmat_im`** *(optional)* — Imaginary part of the frequency-domain channel matrix; same shape as `hmat_re`

## See also:
- [[get_channels_spherical]] (single-frequency channel generator)
- [[get_channels_multifreq]] (multi-frequency channel generator)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs (3D -> vec of length 1, 4D -> vec of length n_freq)
    const auto coeff_re = qd_mex_get_CubeVec<double>(prhs[0]);
    const auto coeff_im = qd_mex_get_CubeVec<double>(prhs[1]);
    const auto delay = qd_mex_get_CubeVec<double>(prhs[2]);
    const auto pilot_grid_in = qd_mex_get_Col<double>(prhs[3]);
    const double bandwidth = qd_mex_get_scalar<double>(prhs[4], "bandwidth", 0.0);
    const auto center_freq = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5]);
    const auto carrier_freq_in = (nrhs < 7) ? arma::vec() : qd_mex_get_Col<double>(prhs[6]);

// i_snap signals multi-snap mode. Scalar 0 = all snapshots; non-empty
// positive vector = subset (1-based). Omitted or [] -> single/multi-freq mode.
arma::uvec i_snap;
bool has_isnap = false;
if (nrhs >= 8 && !mxIsEmpty(prhs[7]))
{
    has_isnap = true;
    arma::uvec idx = qd_mex_get_Col<arma::uword>(prhs[7]);
    if (!(idx.n_elem == 1 && idx[0] == 0))
    {
        if (idx.min() < 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror",
                "Snapshot indices must be 1-based (>=1) or scalar 0 for all snapshots.");
        i_snap = idx - 1; // 1-based MATLAB -> 0-based C++
    }
 }

    // Identify which carrier source(s) the user supplied
    bool has_cf = !center_freq.is_empty();
    bool has_xf = !carrier_freq_in.is_empty();
    bool has_grid_pair = !pilot_grid_in.is_empty() && bandwidth > 0.0;

    if (coeff_re.empty())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "coeff_re must not be empty.");
    if (has_grid_pair && has_xf)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Specify either pilot_grid+bandwidth or carrier_freq, not both.");

    arma::uword n_slab = coeff_re.size(); // 4th dim if 4D, else 1

    if (has_isnap)
    {
        // Multi-snapshot mode: 4th dim is snapshots
        if (has_cf && center_freq.n_elem != 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "In multi-snap mode, center_freq must be omitted or scalar.");
        if (!has_grid_pair && !has_xf)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Provide pilot_grid+bandwidth or carrier_freq.");
        if (has_xf && carrier_freq_in.n_elem == 1 && !has_cf)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Single-element carrier_freq requires center_freq as reference.");
    }
    else if (n_slab == 1)
    {
        // Single-frequency mode
        bool has_freq_pair = has_cf && has_xf;
        if (!has_grid_pair && !has_freq_pair)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Provide pilot_grid+bandwidth or center_freq+carrier_freq.");
    }
    else
    {
        // Multi-frequency mode
        if (!has_cf)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "center_freq is required for multi-frequency input.");
        if (center_freq.n_elem != n_slab)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Length of center_freq must match the 4th dimension of coeff_re.");
        if (!has_grid_pair && !has_xf)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Provide pilot_grid+bandwidth or carrier_freq.");
    }

    // Output dimensions
    arma::uword n_rx = coeff_re[0].n_rows;
    arma::uword n_tx = coeff_re[0].n_cols;
    arma::uword n_carrier = has_xf ? carrier_freq_in.n_elem : pilot_grid_in.n_elem;

    // 3D outputs for single/multi-freq, 4D (cube vector) for multi-snap
    arma::cube hmat_re, hmat_im;
    std::vector<arma::cube> hmat_re_v, hmat_im_v;

    if (has_isnap)
    {
        arma::uword n_out = i_snap.is_empty() ? n_slab : i_snap.n_elem;
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&hmat_re_v, n_rx, n_tx, n_carrier, n_out);
        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&hmat_im_v, n_rx, n_tx, n_carrier, n_out);
    }
    else
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&hmat_re, n_rx, n_tx, n_carrier);
        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&hmat_im, n_rx, n_tx, n_carrier);
    }

    arma::cube *p_hmat_re = (nlhs > 0) ? &hmat_re : nullptr;
    arma::cube *p_hmat_im = (nlhs > 1) ? &hmat_im : nullptr;
    std::vector<arma::cube> *p_hmat_re_v = (nlhs > 0) ? &hmat_re_v : nullptr;
    std::vector<arma::cube> *p_hmat_im_v = (nlhs > 1) ? &hmat_im_v : nullptr;

    // Dispatch
    if (has_isnap) // Multi-snapshot backend
    {
        arma::vec pilot_grid;
        double bw;

        if (has_grid_pair)
        {
            bw = bandwidth;
            pilot_grid = pilot_grid_in;
        }
        else // has_xf
        {
            arma::uword n_c = carrier_freq_in.n_elem;
            if (n_c == 1)
            {
                bw = std::abs(carrier_freq_in[0] - center_freq[0]);
                pilot_grid.ones(1);
                if (carrier_freq_in[0] < center_freq[0])
                    pilot_grid[0] = -1.0;
            }
            else
            {
                bw = carrier_freq_in[n_c - 1] - carrier_freq_in[0];
                pilot_grid = (carrier_freq_in - carrier_freq_in[0]) / bw;
            }
        }

        CALL_QD(quadriga_lib::baseband_freq_response_vec<double>(&coeff_re, &coeff_im, &delay,
                                                                 &pilot_grid, bw, p_hmat_re_v, p_hmat_im_v,
                                                                 i_snap.is_empty() ? nullptr : &i_snap));
    }
    else if (n_slab == 1) // Single-frequency backend; center_freq is ignored
    {
        arma::vec pilot_grid;
        double bw;

        if (has_grid_pair)
        {
            bw = bandwidth;
            pilot_grid = pilot_grid_in;
        }
        else // has_freq_pair
        {
            if (n_carrier == 1)
            {
                bw = std::abs(carrier_freq_in[0] - center_freq[0]);
                pilot_grid.ones(1);
                if (carrier_freq_in[0] < center_freq[0])
                    pilot_grid[0] = -1.0;
            }
            else
            {
                bw = carrier_freq_in[n_carrier - 1] - carrier_freq_in[0];
                pilot_grid = (carrier_freq_in - carrier_freq_in[0]) / bw;
            }
        }

        CALL_QD(quadriga_lib::baseband_freq_response<double>(&coeff_re[0], &coeff_im[0], &delay[0],
                                                             &pilot_grid, bw, p_hmat_re, p_hmat_im));
    }
    else // Multi-frequency backend
    {
        arma::vec carrier_freq;

        if (has_xf)
            carrier_freq = carrier_freq_in;
        else // has_grid_pair
            carrier_freq = center_freq.at(0) + pilot_grid_in * bandwidth;

        CALL_QD(quadriga_lib::baseband_freq_response_multi<double>(coeff_re, coeff_im, delay,
                                                                   center_freq, carrier_freq, p_hmat_re, p_hmat_im));
    }
}