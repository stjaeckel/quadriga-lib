// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

// IRS array struct: the coupling matrix holds the IRS weights; without one, C++ throws instead of using identity
static inline quadriga_lib::arrayant<double> qd_irs_struct2arrayant(const mxArray *input, const std::string &var_name)
{
    auto ant = qd_mex_struct2arrayant(input, false, false, var_name);
    auto given = [&](const char *name)
    { return qd_mex_has_field(input, name) && !mxIsEmpty(qd_mex_get_field(input, name)); };
    if (!given("coupling_re") && !given("coupling_im"))
        ant.coupling_re.reset(), ant.coupling_im.reset();
    return ant;
}

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_IRS
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- The IRS coupling matrix holds the IRS weights and must be given: an `irs_array` or `irs_array_2` struct without
  `coupling_re` / `coupling_im` throws, unlike in the other functions, where it stands for identity
- Polarization coupling is applied via the 8-row transfer matrices `polmat_1`, `polmat_2`: 2x2 Jones matrices (rows = receive
  polarisation V/H, columns = transmit polarisation V/H; the IRS receives in segment 1 and transmits in segment 2),
  column-major with Re/Im interleaved: rows 1-2 = (V rx, V tx), rows 3-4 = (H rx, V tx), rows 5-6 = (V rx, H tx),
  rows 7-8 = (H rx, H tx)
- Each segment is computed like [[get_channels_spherical]]: NLOS path lengths are `max(path_length, d_poly)` with the
  polyline length `d_poly` of the segment; the delay of a combined path is the sum of both segment delays
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths exceeding `threshold_dB`; the pair of
  segment-1 path `i_1` and segment-2 path `i_2` (1-based) has index `(i_1 - 1) · n_path_2 + i_2` in `active_path` and
  `active_path_out`, and the output keeps this order
- A segment without paths (all its path inputs empty) gives empty outputs `[n_rx, n_tx, 0]`
- If `active_path` is provided, it overrides `threshold_dB` for path selection
- Optional `irs_array_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- `center_freq = 0` gives the coefficients at f = 0: the delay phase exp(-j·2π·f·τ) is 1, the antenna and polarization phases remain; delays are computed as usual
- If `use_absolute_delays == false`, the LOS delay between the TX and RX phase centres (direct distance, not via the IRS)
  is subtracted from all paths; per-element delays can be slightly negative when the IRS is close to the TX-RX line

## Usage:
```
[ coeff_re, coeff_im, delay, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    tx_array, rx_array, irs_array, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, irs_array_2, active_path, use_kernel );
```

## Inputs:
- **`tx_array`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [[arrayant_generate]]
- **`rx_array`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [[arrayant_generate]]
- **`irs_array`** — IRS antenna array (TX-facing side); `n_irs` = number of ports
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`irs_pos`** — IRS position; `[3, 1]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1, 1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center for TX → IRS paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_1, 1]`
- **`polmat_1`** — Polarization transfer matrix for TX → IRS paths, row order as above; `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2, 1]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center for IRS → RX paths; values shorter than the
  polyline (e.g. 0) use the polyline length; `[n_path_2, 1]`
- **`polmat_2`** — Polarization transfer matrix for IRS → RX paths, row order as above; `[8, n_path_2]`
- **`i_irs`** — IRS codebook port index, 1-based (one port per column of the IRS coupling matrix); default: `1`
- **`threshold_dB`** — Gain threshold in dB; path combinations below this are discarded; dB; default: `-140`
- **`center_freq`** — Center frequency in Hz; 0 (or empty) gives the coefficients at f = 0 (see above); default: `0`
- **`use_absolute_delays`** — If `true`, delays include the LOS component; default: `false`
- **`irs_array_2`** — Second IRS antenna array for the RX-facing side, with the same number of elements and the same
  coupling size as `irs_array`; enables asymmetric IRS patterns
- **`active_path`** — Bitmask selecting active path pairs; overrides `threshold_dB` when non-empty; logical; `[n_path_1 · n_path_2, 1]`
- **`use_kernel`** — Kernel of both segments, as in [[get_channels_spherical]]: 0 = auto, 1 = GENERIC, 2 = AVX2 (single
  precision; error if unavailable); auto selects GENERIC in this version; default: 0

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delay`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** — Bitmask indicating which path combinations were included in the output; logical; `[n_path_1 · n_path_2, 1]`
- **`aod`** — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`

## See also:
- [[get_channels_spherical]] (single-segment spherical-wave channel)
- [[get_channels_planar]] (single-segment planar-wave channel)
- [[arrayant_generate]] (antenna array generator)
MD!*/

// Cube output; an empty result keeps its [n_rx, n_tx, 0] shape (the generic copy returns 0x0)
static mxArray *copy_cube(const arma::cube &c)
{
    if (c.n_elem != 0)
        return qd_mex_copy2matlab(&c);
    mwSize dims[3] = {(mwSize)c.n_rows, (mwSize)c.n_cols, (mwSize)c.n_slices};
    return mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 19 || nrhs > 26)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Antenna arrays
    const auto tx_array = qd_mex_struct2arrayant(prhs[0], false, false, "tx_array");
    const auto rx_array = qd_mex_struct2arrayant(prhs[1], false, false, "rx_array");
    const auto irs_array = qd_irs_struct2arrayant(prhs[2], "irs_array");

    // Positions and orientations
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

    // Configuration parameters
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

    // Codebook port index: 1-based (MATLAB) -> 0-based (C++)
    const arma::uword n_codebook = irs_array.coupling_re.n_cols; // 0 = missing coupling, reported by C++
    if (n_codebook != 0 && i_irs >= n_codebook)
    {
        std::string msg = "Input 'i_irs' must be from 1 to " + std::to_string(n_codebook == 0 ? 1 : n_codebook) +
                          " (1-based codebook port index, one per column of the IRS coupling matrix).";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }

    // Output containers — size is runtime-dependent (filtered by threshold_dB / active_path)
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = (nlhs > 4) ? &aod : nullptr;
    arma::cube *p_eod = (nlhs > 5) ? &eod : nullptr;
    arma::cube *p_aoa = (nlhs > 6) ? &aoa : nullptr;
    arma::cube *p_eoa = (nlhs > 7) ? &eoa : nullptr;

    // Call library function
    std::vector<bool> active_path_out;
    std::vector<bool> *p_active_path_out = (nlhs > 3) ? &active_path_out : nullptr;
    CALL_QD(quadriga_lib::get_channels_irs<double>(tx_array, rx_array, irs_array,
                                                   tx_pos, tx_orientation, rx_pos, rx_orientation,
                                                   irs_pos, irs_orientation,
                                                   fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, polmat_1,
                                                   fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, polmat_2,
                                                   coeff_re, coeff_im, delay,
                                                   i_irs, threshold_dB, center_freq, use_absolute_delays,
                                                   p_aod, p_eod, p_aoa, p_eoa, irs_array_2, active_path,
                                                   p_active_path_out, use_kernel));

    // Write to MATLAB
    plhs[0] = copy_cube(coeff_re);
    if (nlhs > 1)
        plhs[1] = copy_cube(coeff_im);
    if (nlhs > 2)
        plhs[2] = copy_cube(delay);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&active_path_out);
    if (nlhs > 4)
        plhs[4] = copy_cube(aod);
    if (nlhs > 5)
        plhs[5] = copy_cube(eod);
    if (nlhs > 6)
        plhs[6] = copy_cube(aoa);
    if (nlhs > 7)
        plhs[7] = copy_cube(eoa);
}