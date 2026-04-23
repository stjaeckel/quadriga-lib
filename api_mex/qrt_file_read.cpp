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

- Reads channel impulse response data for a specific snapshot index and origin point
- All output arguments are optional; MATLAB only computes outputs that are requested
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped

## Usage:
```
[ center_frequency, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

## Input Arguments:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** *(optional)* — Snapshot index, 1-based; default: 1
- **`i_orig`** *(optional)* — Origin index, 1-based; default: 1
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
- **`center_frequency`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3]`
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gain on linear scale; `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_coord`** — Interaction coordinates per path; cell array of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; uint32; `[n_path]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single; `[3, sum(no_int)]`
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
    arma::uword i_cir = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "i_cir", 1);
    arma::uword i_orig = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "i_orig", 1);
    const bool downlink = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "downlink", true);
    const int normalize_M = (nrhs < 5) ? 1 : qd_mex_get_scalar<int>(prhs[4], "normalize_M", 1);

    // Convert 1-based (MATLAB) to 0-based (C++); guard against unsigned underflow
    if (i_cir == 0 || i_orig == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "i_cir and i_orig must be >= 1 (1-based).");
    i_cir -= 1;
    i_orig -= 1;

    // Declare outputs
    arma::vec center_frequency, tx_pos, tx_orientation, rx_pos, rx_orientation;
    arma::mat fbs_pos, lbs_pos, path_gain;
    arma::vec path_length;
    arma::cube M;
    arma::vec aod, eod, aoa, eoa;
    std::vector<arma::mat> path_coord;
    arma::u32_vec no_int;
    arma::fmat coord;

    // Wrap optional output pointers based on nlhs
    arma::vec *p_center_frequency = (nlhs > 0) ? &center_frequency : nullptr;
    arma::vec *p_tx_pos = (nlhs > 1) ? &tx_pos : nullptr;
    arma::vec *p_tx_orientation = (nlhs > 2) ? &tx_orientation : nullptr;
    arma::vec *p_rx_pos = (nlhs > 3) ? &rx_pos : nullptr;
    arma::vec *p_rx_orientation = (nlhs > 4) ? &rx_orientation : nullptr;
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

    // Call library function
    CALL_QD(quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, downlink,
                                                p_center_frequency, p_tx_pos, p_tx_orientation,
                                                p_rx_pos, p_rx_orientation,
                                                p_fbs_pos, p_lbs_pos, p_path_gain, p_path_length, p_M,
                                                p_aod, p_eod, p_aoa, p_eoa,
                                                p_path_coord, normalize_M,
                                                p_no_int, p_coord));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&center_frequency);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&tx_pos);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&tx_orientation);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&rx_pos);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&rx_orientation);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&fbs_pos);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&lbs_pos);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&path_gain);
    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&path_length);
    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&M);
    if (nlhs > 10)
        plhs[10] = qd_mex_copy2matlab(&aod);
    if (nlhs > 11)
        plhs[11] = qd_mex_copy2matlab(&eod);
    if (nlhs > 12)
        plhs[12] = qd_mex_copy2matlab(&aoa);
    if (nlhs > 13)
        plhs[13] = qd_mex_copy2matlab(&eoa);
    if (nlhs > 14)
    {
        // Variable column count per path - cell array preserves per-path structure
        const mwSize n_path = static_cast<mwSize>(path_coord.size());
        plhs[14] = mxCreateCellMatrix(1, n_path);
        for (mwSize i = 0; i < n_path; ++i)
            mxSetCell(plhs[14], i, qd_mex_copy2matlab(&path_coord[i]));
    }
    if (nlhs > 15)
        plhs[15] = qd_mex_copy2matlab(&no_int);
    if (nlhs > 16)
        plhs[16] = qd_mex_copy2matlab(&coord);
}