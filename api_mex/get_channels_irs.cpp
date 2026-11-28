// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

// IRS array struct: the port weights hold the IRS weights; without one, C++ throws instead of using identity
static inline quadriga_lib::arrayant<double> qd_irs_struct2arrayant(const mxArray *input, const std::string &var_name)
{
    auto ant = qd_mex_struct2arrayant(input, false, false, var_name);
    auto given = [&](const char *name)
    { return qd_mex_has_field(input, name) && !mxIsEmpty(qd_mex_get_field(input, name)); };
    if (!given("port_weights_re") && !given("port_weights_im"))
        ant.port_weights_re.reset(), ant.port_weights_im.reset();
    return ant;
}

/*!SECTION
Channel generators
SECTION!*/

/*!MD
# GET_CHANNELS_IRS
Calculate MIMO channel coefficients and delays for paths over an intelligent reflecting surface

- Paths of one link over an intelligent reflecting surface (IRS) with `n_irs` elements: segment 1 from the Tx to the
  IRS and segment 2 from the IRS to the Rx, each computed by [[get_channels_spherical]] with the IRS elements as Rx and
  Tx elements; each output path combines one path of each segment
- Each IRS element $k$ reflects once with its weight $w_k$, column `i_irs` of the IRS port weights (a codebook entry):
  per path pair, $\mathbf{H} = \mathbf{B} \, \mathrm{diag}(\mathbf{w}) \, \mathbf{A}$ with the element channels
  $\mathbf{A}$ of segment 1 `[n_irs, n_tx]` and $\mathbf{B}$ of segment 2 `[n_rx, n_irs]`
- The delay of a pair is the sum of its segment delays, each the $|w_k|^2$-weighted mean over the IRS elements; the
  departure angles come from segment 1, the arrival angles from segment 2, as $|w_k|^2$-weighted mean directions
- Kept are the path pairs whose strongest coefficient has a power above `threshold_dB`, or those set in `active_path`;
  the pair of segment-1 path $i_1$ and segment-2 path $i_2$ (1-based) has the index $(i_1 - 1) n_{\mathrm{path},2} + i_2$
  in `active_path` and `active_path_out`, and the output keeps this order
- The outputs come in another order than in C++: `active_path_out` comes before the angles, so that the angles can be
  left off at the end

## Usage:
```
[ coeff, delay, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    tx_array, rx_array, irs_array, tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, irs_array_2, active_path, use_kernel );
```

## Inputs:
- **`tx_array`**, **`rx_array`** — Tx and Rx array, struct, fields in [[generate_antenna]]; `n_tx` and `n_rx` ports
- **`irs_array`** — IRS array with `n_irs` elements, struct, valid, with port weights `[n_irs, n_codebook]` that hold
  the reflection weights; its patterns serve both segments unless `irs_array_2` is given
- **`tx_pos`**, **`rx_pos`**, **`irs_pos`** — Tx, Rx and IRS position in m, finite; 3 elements
- **`tx_orientation`**, **`rx_orientation`**, **`irs_orientation`** — Orientations, Euler angles (bank, tilt, heading)
  in rad, finite; 3 elements
- **`fbs_pos_1`**, **`lbs_pos_1`** — First- and last-bounce scatterer positions of the segment-1 paths in m;
  `[3, n_path_1]`
- **`path_gain_1`** — Path gains of segment 1, linear power; `[n_path_1, 1]`
- **`path_length_1`** — Path lengths of segment 1 in m between the Tx and IRS phase centres, as in
  [[get_channels_spherical]]; `[n_path_1, 1]`
- **`polmat_1`** — Jones matrices of segment 1, the IRS receiving (section description); `[8, n_path_1]`
- **`fbs_pos_2`**, **`lbs_pos_2`**, **`path_gain_2`**, **`path_length_2`**, **`polmat_2`** — The same for segment 2
  from the IRS to the Rx, the IRS transmitting; `n_path_2` paths
- **`i_irs`** — Codebook entry, 1-based column of the IRS port weights, at most `n_codebook`; default: 1
- **`threshold_dB`** — Minimum power in dB of the strongest coefficient of a kept path pair; not used with
  `active_path`; default: -140
- **`center_freq`** — Center frequency in Hz, finite and >= 0; 0 gives the coefficients at f = 0, as in
  [[get_channels_spherical]]; default: 0
- **`use_absolute_delays`** — Keep the delay of the direct Tx-Rx distance in all delays; `false`: delays relative to it
  (Notes); default: `false`
- **`irs_array_2`** — Patterns of the IRS for segment 2 (asymmetric IRS), struct, valid, `n_irs` elements; its port
  weights and element positions are not used; default: `[]` (`irs_array`)
- **`active_path`** — Path pairs to keep, replacing `threshold_dB`; logical or numeric (nonzero = keep);
  `[n_path_1 * n_path_2, 1]`; default: `[]` (`threshold_dB`)
- **`use_kernel`** — Kernel of both segments, as in [[get_channels_spherical]]; default: 0

## Outputs:
- **`coeff`** — Complex coefficients; `[n_rx, n_tx, n_path_irs]` with the number of kept path pairs `n_path_irs`
- **`delay`** — Delays in s; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** — Kept path pairs, for `active_path` of a later call; logical; `[n_path_1 * n_path_2, 1]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure and arrival in rad, GCS;
  `[n_rx, n_tx, n_path_irs]`

## Notes:
- **IRS port weights**: an `irs_array` without `port_weights_re` / `port_weights_im` gives an error here, unlike in the
  other functions, where they mean identity; the IRS elements are not combined into ports
- **Segments**: path lengths, LOS paths and the per-element geometry as in [[get_channels_spherical]]; a segment
  without paths (all its path inputs empty) gives the outputs `[n_rx, n_tx, 0]`
- **Relative delays**: the delay of the direct Tx-Rx distance (not via the IRS) is subtracted; delays can be slightly
  negative when the IRS is close to the Tx-Rx line

## See also:
- [[get_channels_spherical]] (the channel of each segment)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 19 || nrhs > 26)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const auto tx_array = qd_mex_struct2arrayant(prhs[0], false, false, "tx_array");
    const auto rx_array = qd_mex_struct2arrayant(prhs[1], false, false, "rx_array");
    const auto irs_array = qd_irs_struct2arrayant(prhs[2], "irs_array");

    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[3], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[4], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[5], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[6], "rx_orientation", 3);
    const auto irs_pos = qd_mex_typecast_Col<double>(prhs[7], "irs_pos", 3);
    const auto irs_orientation = qd_mex_typecast_Col<double>(prhs[8], "irs_orientation", 3);

    // Segment TX → IRS
    const auto fbs_pos_1 = qd_mex_get_Mat<double>(prhs[9], false, "fbs_pos_1");
    const auto lbs_pos_1 = qd_mex_get_Mat<double>(prhs[10], false, "lbs_pos_1");
    const auto path_gain_1 = qd_mex_get_Col<double>(prhs[11], false, "path_gain_1");
    const auto path_length_1 = qd_mex_get_Col<double>(prhs[12], false, "path_length_1");
    const auto polmat_1 = qd_mex_get_Mat<double>(prhs[13], false, "polmat_1");

    // Segment IRS → RX
    const auto fbs_pos_2 = qd_mex_get_Mat<double>(prhs[14], false, "fbs_pos_2");
    const auto lbs_pos_2 = qd_mex_get_Mat<double>(prhs[15], false, "lbs_pos_2");
    const auto path_gain_2 = qd_mex_get_Col<double>(prhs[16], false, "path_gain_2");
    const auto path_length_2 = qd_mex_get_Col<double>(prhs[17], false, "path_length_2");
    const auto polmat_2 = qd_mex_get_Mat<double>(prhs[18], false, "polmat_2");

    const arma::uword i_irs = (nrhs < 20) ? 0 : qd_mex_get_index(prhs[19], "i_irs"); // 1-based -> 0-based
    double threshold_dB = (nrhs < 21) ? -140.0 : qd_mex_get_scalar<double>(prhs[20], "threshold_dB", -140.0);
    double center_freq = (nrhs < 22) ? 0.0 : qd_mex_get_scalar<double>(prhs[21], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 23) ? false : qd_mex_get_scalar<bool>(prhs[22], "use_absolute_delays", false);

    // Optional second IRS array (RX-facing side)
    quadriga_lib::arrayant<double> irs_array_2;
    if (nrhs > 23 && !mxIsEmpty(prhs[23]))
        irs_array_2 = qd_irs_struct2arrayant(prhs[23], "irs_array_2");

    const auto active_path = (nrhs < 25 || mxIsEmpty(prhs[24])) ? std::vector<bool>() : qd_mex_matlab2vector_Bool(prhs[24], "active_path");
    const int use_kernel = (nrhs < 26) ? 0 : qd_mex_get_scalar<int>(prhs[25], "use_kernel", 0);

    // Output sizes depend on threshold_dB and active_path and are set by C++
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = (nlhs > 3) ? &aod : nullptr;
    arma::cube *p_eod = (nlhs > 4) ? &eod : nullptr;
    arma::cube *p_aoa = (nlhs > 5) ? &aoa : nullptr;
    arma::cube *p_eoa = (nlhs > 6) ? &eoa : nullptr;

    std::vector<bool> active_path_out;
    std::vector<bool> *p_active_path_out = (nlhs > 2) ? &active_path_out : nullptr;
    CALL_QD(quadriga_lib::get_channels_irs<double>(tx_array, rx_array, irs_array,
                                                   tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                   irs_pos, irs_orientation,
                                                   fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1,
                                                   fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2,
                                                   coeff_re, coeff_im, delay,
                                                   i_irs, threshold_dB, center_freq, use_absolute_delays,
                                                   p_aod, p_eod, p_aoa, p_eoa, irs_array_2, active_path,
                                                   p_active_path_out, use_kernel));

    // Write to MATLAB; empty results keep their [n_rx, n_tx, 0] shape
    plhs[0] = qd_mex_copy2matlab(&coeff_re, &coeff_im, true);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab_keep_shape(delay);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&active_path_out);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab_keep_shape(aod);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab_keep_shape(eod);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab_keep_shape(aoa);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab_keep_shape(eoa);
}
