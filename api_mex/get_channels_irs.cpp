// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# GET_CHANNELS_IRS
Calculate MIMO channel coefficients for IRS-assisted communication

- Computes channel coefficients and delays from two path segments: TX → IRS and IRS → RX
- IRS is modeled as a passive array; phase shifts are defined via its coupling matrix; codebook entry selected by `i_irs`
- Polarization coupling is applied via the 8-row transfer matrices `M_1`, `M_2` (interleaved Re/Im for VV, VH, HV, HH components)
- Output paths `n_path_irs` are all combinations of segment 1 and segment 2 paths exceeding `threshold_dB`
- If `active_path_in` is provided, it overrides `threshold_dB` for path selection
- Optional `ant_irs_2` provides a separate IRS antenna pattern for the RX-facing side (asymmetric IRS)
- If `center_freq == 0`, phase calculation is disabled and only delays are computed
- If `use_absolute_delays == false`, the minimum delay (LOS delay) is subtracted from all paths

## Usage:
```
[ coeff_re, coeff_im, delays, active_path_out, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_irs( ...
    ant_tx, ant_rx, ant_irs, ...
    fbs_pos_1, lbs_pos_1, path_gain_1, path_length_1, M_1, ...
    fbs_pos_2, lbs_pos_2, path_gain_2, path_length_2, M_2, ...
    tx_pos, tx_orientation, rx_pos, rx_orientation, irs_pos, irs_orientation, ...
    i_irs, threshold_dB, center_freq, use_absolute_delays, active_path_in, ant_irs_2 );
```

## Inputs:
- **`ant_tx`** — Transmit antenna array; `n_tx` = number of ports after element coupling, see [[arrayant_generate]]
- **`ant_rx`** — Receive antenna array; `n_rx` = number of ports after element coupling, see [[arrayant_generate]]
- **`ant_irs`** — IRS antenna array (TX-facing side); `n_irs` = number of ports
- **`fbs_pos_1`** — First-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`lbs_pos_1`** — Last-bounce scatterer positions for TX → IRS paths; `[3, n_path_1]`
- **`path_gain_1`** — Path gains in linear scale for TX → IRS paths; `[n_path_1, 1]`
- **`path_length_1`** — Total path lengths from TX to IRS phase center for TX → IRS paths; `[n_path_1, 1]`
- **`M_1`** — Polarization transfer matrix for TX → IRS paths, interleaved
  (ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH); `[8, n_path_1]`
- **`fbs_pos_2`** — First-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`lbs_pos_2`** — Last-bounce scatterer positions for IRS → RX paths; `[3, n_path_2]`
- **`path_gain_2`** — Path gains in linear scale for IRS → RX paths; `[n_path_2, 1]`
- **`path_length_2`** — Total path lengths from IRS to RX phase center for IRS → RX paths; `[n_path_2, 1]`
- **`M_2`** — Polarization transfer matrix for IRS → RX paths, interleaved complex; `[8, n_path_2]`
- **`tx_pos`** — Transmitter position; `[3, 1]`
- **`tx_orientation`** — Transmitter orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`rx_pos`** — Receiver position; `[3, 1]`
- **`rx_orientation`** — Receiver orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`irs_pos`** — IRS position; `[3, 1]`
- **`irs_orientation`** — IRS orientation as Euler angles (bank, tilt, heading); `[3, 1]`
- **`i_irs`** *(optional)* — IRS codebook port index; default: `0`
- **`threshold_dB`** *(optional)* — Gain threshold in dB; path combinations below this are discarded; dB; default: `-140`
- **`center_freq`** *(optional)* — Center frequency; set to `0` or skip/leave empty to skip phase computation; default: `0`
- **`use_absolute_delays`** *(optional)* — If `true`, delays include the LOS component; default: `false`
- **`active_path_in`** *(optional)* — Bitmask selecting active path pairs; overrides `threshold_dB`
  when non-empty; logical; `[n_path_1 · n_path_2, 1]`
- **`ant_irs_2`** *(optional)* — Second IRS antenna array for the RX-facing side; enables asymmetric IRS patterns

## Outputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`coeff_im`** — Imaginary part of channel coefficients; `[n_rx, n_tx, n_path_irs]`
- **`delays`** — Propagation delays in seconds; `[n_rx, n_tx, n_path_irs]`
- **`active_path_out`** *(optional)* — Bitmask indicating which path combinations were included in
  the output; logical; `[n_path_1 · n_path_2, 1]`
- **`aod`** *(optional)* — Azimuth of departure; `[n_rx, n_tx, n_path_irs]`
- **`eod`** *(optional)* — Elevation of departure; `[n_rx, n_tx, n_path_irs]`
- **`aoa`** *(optional)* — Azimuth of arrival; `[n_rx, n_tx, n_path_irs]`
- **`eoa`** *(optional)* — Elevation of arrival; `[n_rx, n_tx, n_path_irs]`

## See also:
- [[get_channels_spherical]] (single-segment spherical-wave channel)
- [[get_channels_planar]] (single-segment planar-wave channel)
- [[arrayant_generate]] (antenna array generator)
- [[combine_irs_coord]] (coordinate setup for IRS geometry)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 19 || nrhs > 25)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Parse antenna objects
    auto ant_tx = qd_mex_struct2arrayant(prhs[0]);
    auto ant_rx = qd_mex_struct2arrayant(prhs[1]);
    auto ant_irs = qd_mex_struct2arrayant(prhs[2]);

    // Segment TX → IRS
    const auto fbs_pos_1 = qd_mex_get_Mat<double>(prhs[3]);
    const auto lbs_pos_1 = qd_mex_get_Mat<double>(prhs[4]);
    const auto path_gain_1 = qd_mex_get_Col<double>(prhs[5]);
    const auto path_length_1 = qd_mex_get_Col<double>(prhs[6]);
    const auto M_1 = qd_mex_get_Mat<double>(prhs[7]);

    // Segment IRS → RX
    const auto fbs_pos_2 = qd_mex_get_Mat<double>(prhs[8]);
    const auto lbs_pos_2 = qd_mex_get_Mat<double>(prhs[9]);
    const auto path_gain_2 = qd_mex_get_Col<double>(prhs[10]);
    const auto path_length_2 = qd_mex_get_Col<double>(prhs[11]);
    const auto M_2 = qd_mex_get_Mat<double>(prhs[12]);

    // Positions and orientations
    const auto tx_pos = qd_mex_typecast_Col<double>(prhs[13], "tx_pos", 3);
    const auto tx_orientation = qd_mex_typecast_Col<double>(prhs[14], "tx_orientation", 3);
    const auto rx_pos = qd_mex_typecast_Col<double>(prhs[15], "rx_pos", 3);
    const auto rx_orientation = qd_mex_typecast_Col<double>(prhs[16], "rx_orientation", 3);
    const auto irs_pos = qd_mex_typecast_Col<double>(prhs[17], "irs_pos", 3);
    const auto irs_orientation = qd_mex_typecast_Col<double>(prhs[18], "irs_orientation", 3);

    // Configuration parameters
    arma::uword i_irs = (nrhs < 20) ? 0 : qd_mex_get_scalar<arma::uword>(prhs[19], "i_irs", 0);
    double threshold_dB = (nrhs < 21) ? -140.0 : qd_mex_get_scalar<double>(prhs[20], "threshold_dB", -140.0);
    double center_freq = (nrhs < 22) ? 0.0 : qd_mex_get_scalar<double>(prhs[21], "center_freq", 0.0);
    bool use_absolute_delays = (nrhs < 23) ? false : qd_mex_get_scalar<bool>(prhs[22], "use_absolute_delays", false);
    const auto active_path_in = (nrhs < 24 || mxIsEmpty(prhs[23])) ? std::vector<bool>() : qd_mex_matlab2vector_Bool(prhs[23]);

    // Optional second IRS array (RX-facing side)
    quadriga_lib::arrayant<double> ant_irs_2;
    bool use_alternative_irs = (nrhs > 24) && !mxIsEmpty(prhs[24]);
    if (use_alternative_irs)
        ant_irs_2 = qd_mex_struct2arrayant(prhs[24]);

    // Extract scalar values
    double Tx = tx_pos.at(0), Ty = tx_pos.at(1), Tz = tx_pos.at(2);
    double Tb = tx_orientation.at(0), Tt = tx_orientation.at(1), Th = tx_orientation.at(2);
    double Rx = rx_pos.at(0), Ry = rx_pos.at(1), Rz = rx_pos.at(2);
    double Rb = rx_orientation.at(0), Rt = rx_orientation.at(1), Rh = rx_orientation.at(2);
    double Ix = irs_pos.at(0), Iy = irs_pos.at(1), Iz = irs_pos.at(2);
    double Ib = irs_orientation.at(0), It = irs_orientation.at(1), Ih = irs_orientation.at(2);

    // Output containers — size is runtime-dependent (filtered by threshold_dB / active_path_in)
    arma::cube coeff_re, coeff_im, delay, aod, eod, aoa, eoa;
    arma::cube *p_aod = (nlhs > 4) ? &aod : nullptr;
    arma::cube *p_eod = (nlhs > 5) ? &eod : nullptr;
    arma::cube *p_aoa = (nlhs > 6) ? &aoa : nullptr;
    arma::cube *p_eoa = (nlhs > 7) ? &eoa : nullptr;
    const quadriga_lib::arrayant<double> *p_irs_array_2 = use_alternative_irs ? &ant_irs_2 : nullptr;
    const std::vector<bool> *p_active_path = active_path_in.empty() ? nullptr : &active_path_in;

    // Call library function
    std::vector<bool> active_path_out;
    CALL_QD(active_path_out = quadriga_lib::get_channels_irs<double>(&ant_tx, &ant_rx, &ant_irs,
                                                                     Tx, Ty, Tz, Tb, Tt, Th,
                                                                     Rx, Ry, Rz, Rb, Rt, Rh,
                                                                     Ix, Iy, Iz, Ib, It, Ih,
                                                                     &fbs_pos_1, &lbs_pos_1, &path_gain_1, &path_length_1, &M_1,
                                                                     &fbs_pos_2, &lbs_pos_2, &path_gain_2, &path_length_2, &M_2,
                                                                     &coeff_re, &coeff_im, &delay,
                                                                     i_irs, threshold_dB, center_freq, use_absolute_delays,
                                                                     p_aod, p_eod, p_aoa, p_eoa, p_irs_array_2, p_active_path));

    // Write to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&coeff_re);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&coeff_im);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&delay);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&active_path_out);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&aod);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&eod);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&aoa);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&eoa);
}