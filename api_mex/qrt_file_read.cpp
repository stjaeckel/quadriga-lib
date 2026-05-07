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
# QRT_FILE_READ
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped

## Usage:
```
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

## Input Arguments:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** *(optional)* — Snapshot indices; 1-based; uint64; `[n_out]` or empty; default: read all
- **`i_orig`** *(optional)* — Origin index; 1-based; uint64; scalar; default: 1
- **`downlink`** *(optional)* — If `true`, origin=TX, destination=RX; if `false`, roles are
  swapped; logical scalar; default: `true`
- **`normalize_M`** *(optional)* — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>
  | `normalize_M` | `M`                   | `path_gain`                      |
  | ------------- | --------------------- | -------------------------------- |
  | 0             | As stored in QRT file | -PL                              |
  | 1             | Max column power = 1  | -PL minus material losses        |

## Output Arguments:
- **`center_freq`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions;  Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain on linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path]`
- **`M`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(no_int)]`

## See also:
- [[arrayant_generate]] (for generating antenna arrays)
- [[get_channels_planar]] (for embedding antennas using departure and arrival angles)
- [[get_channels_spherical]] (for embedding antennas using FBS/LBS positions)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 17)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    arma::uvec i_cir = (nrhs < 2) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[1]);
    arma::uword i_orig = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "i_orig", 1);
    const bool downlink = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "downlink", true);
    const int normalize_M = (nrhs < 5) ? 1 : qd_mex_get_scalar<int>(prhs[4], "normalize_M", 1);

    // Convert 1-based (MATLAB) to 0-based (C++); guard against unsigned underflow
    if (i_orig == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "'i_orig' must be >= 1 (1-based indices).");
    i_orig -= 1;

    // Initialize cache and file access
    std::ifstream stream(fn, std::ios::in | std::ios::binary);
    quadriga_lib::qrt_read_cache cache;
    CALL_QD(cache = quadriga_lib::qrt_read_cache_init(fn, &stream));

    if (i_cir.empty()) // Default: read all CIRs
        i_cir = arma::regspace<arma::uvec>(0, arma::uword(cache.no_cir - 1));
    else // Convert 1-based (MATLAB) to 0-based (C++); guard against unsigned underflow
    {
        if (arma::any(i_cir == 0))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_cir' must be >= 1 (1-based indices).");
        i_cir -= 1;

        if (arma::any(i_cir >= (arma::uword)cache.no_cir))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "CIR index exceeds number of CIRs in file.");
    }

    // Flag to enable reading mutiple entries at once
    arma::uword no_out = i_cir.n_elem;

    // Return frequencies in Hz
    if (nlhs > 0)
    {
        arma::vec center_freq;
        plhs[0] = qd_mex_init_output(&center_freq, cache.no_freq);
        for (unsigned i_freq = 0; i_freq < cache.no_freq; ++i_freq)
            if (cache.version == 6) // Scalar mode, stored in Hz
                center_freq[i_freq] = (double)cache.freq[i_freq];
            else // EM Mode, stored in GHz
                center_freq[i_freq] = 1.0e9 * (double)cache.freq[i_freq];
    }

    // Initialize output data structures
    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&tx_pos, 3, no_out);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&tx_orientation, 3, no_out);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&rx_pos, 3, no_out);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&rx_orientation, 3, no_out);

    if (nlhs > 5)
        plhs[5] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 6)
        plhs[6] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 7)
        plhs[7] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 8)
        plhs[8] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 9)
        plhs[9] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 10)
        plhs[10] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 11)
        plhs[11] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 12)
        plhs[12] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 13)
        plhs[13] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 14)
        plhs[14] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 15)
        plhs[15] = mxCreateCellMatrix(no_out, 1);

    if (nlhs > 16)
        plhs[16] = mxCreateCellMatrix(no_out, 1);

    // Temporary read buffers
    arma::vec tx_pos_buff, tx_orientation_buff, rx_pos_buff, rx_orientation_buff;
    arma::mat fbs_pos, lbs_pos, path_gain;
    arma::vec path_length, aod, eod, aoa, eoa;
    arma::cube M;
    std::vector<arma::mat> path_coord;
    arma::u32_vec no_int;
    arma::fmat coord;

    // Wrap optional output pointers based on nlhs
    arma::vec *p_tx_pos = (nlhs > 1) ? &tx_pos_buff : nullptr;
    arma::vec *p_tx_orientation = (nlhs > 2) ? &tx_orientation_buff : nullptr;
    arma::vec *p_rx_pos = (nlhs > 3) ? &rx_pos_buff : nullptr;
    arma::vec *p_rx_orientation = (nlhs > 4) ? &rx_orientation_buff : nullptr;
    arma::mat *p_fbs_pos = (nlhs > 5) ? &fbs_pos : nullptr;
    arma::mat *p_lbs_pos = (nlhs > 6) ? &lbs_pos : nullptr;
    arma::mat *p_path_gain = (nlhs > 7) ? &path_gain : nullptr;
    arma::vec *p_path_length = (nlhs > 8) ? &path_length : nullptr;
    arma::cube *p_M = (nlhs > 9) ? &M : nullptr;
    arma::vec *p_aod = (nlhs > 10) ? &aod : nullptr;
    arma::vec *p_eod = (nlhs > 11) ? &eod : nullptr;
    arma::vec *p_aoa = (nlhs > 12) ? &aoa : nullptr;
    arma::vec *p_eoa = (nlhs > 13) ? &eoa : nullptr;
    std::vector<arma::mat> *p_path_coord = (nlhs > 14) ? &path_coord : nullptr;
    arma::u32_vec *p_no_int = (nlhs > 15) ? &no_int : nullptr;
    arma::fmat *p_coord = (nlhs > 16) ? &coord : nullptr;

    // Iterate over all CIRs
    for (arma::uword i_out = 0; i_out < no_out; ++i_out)
    {
        // Call library function
        CALL_QD(quadriga_lib::qrt_file_read<double>(fn, i_cir[i_out], i_orig, downlink, nullptr,
                                                    p_tx_pos, p_tx_orientation, p_rx_pos, p_rx_orientation,
                                                    p_fbs_pos, p_lbs_pos, p_path_gain, p_path_length, p_M,
                                                    p_aod, p_eod, p_aoa, p_eoa,
                                                    p_path_coord, normalize_M,
                                                    p_no_int, p_coord, &stream, &cache));

        // Copy to MATLAB
        if (nlhs > 1)
            tx_pos.col(i_out) = tx_pos_buff;
        if (nlhs > 2)
            tx_orientation.col(i_out) = tx_orientation_buff;
        if (nlhs > 3)
            rx_pos.col(i_out) = rx_pos_buff;
        if (nlhs > 4)
            rx_orientation.col(i_out) = rx_orientation_buff;
        if (nlhs > 5)
            mxSetCell(plhs[5], i_out, qd_mex_copy2matlab(&fbs_pos));
        if (nlhs > 6)
            mxSetCell(plhs[6], i_out, qd_mex_copy2matlab(&lbs_pos));
        if (nlhs > 7)
            mxSetCell(plhs[7], i_out, qd_mex_copy2matlab(&path_gain));
        if (nlhs > 8)
            mxSetCell(plhs[8], i_out, qd_mex_copy2matlab(&path_length));
        if (nlhs > 9)
            mxSetCell(plhs[9], i_out, qd_mex_copy2matlab(&M));
        if (nlhs > 10)
            mxSetCell(plhs[10], i_out, qd_mex_copy2matlab(&aod));
        if (nlhs > 11)
            mxSetCell(plhs[11], i_out, qd_mex_copy2matlab(&eod));
        if (nlhs > 12)
            mxSetCell(plhs[12], i_out, qd_mex_copy2matlab(&aoa));
        if (nlhs > 13)
            mxSetCell(plhs[13], i_out, qd_mex_copy2matlab(&eoa));
        if (nlhs > 14)
        {
            // Variable column count per path - cell array preserves per-path structure
            const mwSize n_path = (mwSize)path_coord.size();
            auto out = mxCreateCellMatrix(n_path, 1);
            for (mwSize i = 0; i < n_path; ++i)
                mxSetCell(out, i, qd_mex_copy2matlab(&path_coord[i]));
            mxSetCell(plhs[14], i_out, out);
        }
        if (nlhs > 15)
            mxSetCell(plhs[15], i_out, qd_mex_copy2matlab(&no_int));
        if (nlhs > 16)
            mxSetCell(plhs[16], i_out, qd_mex_copy2matlab(&coord));
    }

    // Close stream
    stream.close();
}