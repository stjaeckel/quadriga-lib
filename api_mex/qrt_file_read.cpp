// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"
#include "quadriga_lib_helper_functions.hpp"
#include "qd_qrt_chunk_reader.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# QRT_FILE_READ
Read ray-tracing CIR data from a QRT file

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- Reads in chunks of 16 snapshots; besides the output, at most two chunks are held in memory
- If `uplink = false`, origin is TX and destination is RX; if `true`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal

## Usage:
```
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, uplink, normalize_polmat );
```

## Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; uint64; `[n_out]` or empty; default: `[]` (read all; for a file
  without CIRs, all outputs are empty with `n_out = 0`)
- **`i_orig`** — Origin index; 1-based; uint64; scalar; default: 1
- **`uplink`** — If `false`, origin=TX, destination=RX; if `true`, roles are swapped (origin=RX, destination=TX); logical scalar; default: `false`
- **`normalize_polmat`** — Controls `polmat` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm<br><br>
    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | :-----------: | :---------------------------------------------: | :------------------------------: |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |<br><br>

    Other values are rejected with an error.

## Outputs:
- **`center_freq`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions;  Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain in linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path]`
- **`polmat`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`n_interact`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(n_interact)]`
- **`interact_type`** — Interaction type codes (flat, concatenated across paths, matching `coord`); uint8;
  Cell of length `n_out`; elements `[sum(n_interact)]`. Empty for v4 legacy files. GBSM paths written with the C++ API
  carry 129 (GR) and 193 (NLOS); LOS and GR paths are not marked as GBSM.

## See also:
- [[arrayant_generate]] (for generating antenna arrays)
- [[get_channels_planar]] (for embedding antennas using departure and arrival angles)
- [[get_channels_spherical]] (for embedding antennas using FBS/LBS positions)
- [[get_channels_multifreq]] (for multi-frequency antenna embedding)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 18)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const arma::uvec i_cir = (nrhs < 2) ? arma::uvec() : qd_mex_get_index_Col(prhs[1], "i_cir"); // 1-based -> 0-based
    const arma::uword i_orig = (nrhs < 3) ? 0 : qd_mex_get_index(prhs[2], "i_orig"); // 1-based -> 0-based
    const bool uplink = (nrhs < 4) ? false : qd_mex_get_scalar<bool>(prhs[3], "uplink", false);
    const int normalize_polmat = (nrhs < 5) ? 1 : qd_mex_get_scalar<int>(prhs[4], "normalize_polmat", 1);

    // Reads in chunks: a reader thread reads the next chunk while this thread copies the previous one to MATLAB
    std::vector<arma::mat *> pos(4);
    arma::mat tx_pos, tx_orientation, rx_pos, rx_orientation;
    pos[0] = &tx_pos, pos[1] = &tx_orientation, pos[2] = &rx_pos, pos[3] = &rx_orientation;

    auto init = [&](const arma::vec &center_freq, arma::uword n_out)
    {
        plhs[0] = qd_mex_copy2matlab(&center_freq);
        for (int i = 1; i < 5 && i < nlhs; ++i)
            plhs[i] = qd_mex_init_output(pos[i - 1], 3, n_out);
        for (int i = 5; i < nlhs; ++i)
            plhs[i] = mxCreateCellMatrix(n_out, 1);
    };

    auto consume = [&](const qd_qrt_chunk &b, arma::uword offset)
    {
        const arma::uword n = b.i_cir.n_elem;
        for (int i = 1; i < 5 && i < nlhs; ++i)
        {
            const arma::mat &src = (i == 1) ? b.tx_pos : (i == 2) ? b.tx_orientation : (i == 3) ? b.rx_pos : b.rx_orientation;
            pos[i - 1]->cols(offset, offset + n - 1) = src;
        }
        for (arma::uword j = 0; j < n; ++j)
        {
            const mwIndex o = (mwIndex)(offset + j);
            if (nlhs > 5)
                mxSetCell(plhs[5], o, qd_mex_copy2matlab(&b.fbs_pos[j]));
            if (nlhs > 6)
                mxSetCell(plhs[6], o, qd_mex_copy2matlab(&b.lbs_pos[j]));
            if (nlhs > 7)
                mxSetCell(plhs[7], o, qd_mex_copy2matlab(&b.path_gain[j]));
            if (nlhs > 8)
                mxSetCell(plhs[8], o, qd_mex_copy2matlab(&b.path_length[j]));
            if (nlhs > 9)
                mxSetCell(plhs[9], o, qd_mex_copy2matlab(&b.polmat[j]));
            if (nlhs > 10)
                mxSetCell(plhs[10], o, qd_mex_copy2matlab(&b.aod[j]));
            if (nlhs > 11)
                mxSetCell(plhs[11], o, qd_mex_copy2matlab(&b.eod[j]));
            if (nlhs > 12)
                mxSetCell(plhs[12], o, qd_mex_copy2matlab(&b.aoa[j]));
            if (nlhs > 13)
                mxSetCell(plhs[13], o, qd_mex_copy2matlab(&b.eoa[j]));
            if (nlhs > 14)
            {
                // Variable column count per path - cell array preserves per-path structure
                const std::vector<arma::mat> &pc = b.path_coord[j];
                mxArray *out = mxCreateCellMatrix((mwSize)pc.size(), 1);
                for (size_t i = 0; i < pc.size(); ++i)
                    mxSetCell(out, (mwIndex)i, qd_mex_copy2matlab(&pc[i]));
                mxSetCell(plhs[14], o, out);
            }
            if (nlhs > 15)
                mxSetCell(plhs[15], o, qd_mex_copy2matlab(&b.n_interact[j]));
            if (nlhs > 16)
                mxSetCell(plhs[16], o, qd_mex_copy2matlab(&b.coord[j]));
            if (nlhs > 17)
            {
                const std::vector<uint8_t> &it = b.interact_type[j];
                mxArray *it_out = mxCreateNumericMatrix((mwSize)it.size(), 1, mxUINT8_CLASS, mxREAL);
                if (!it.empty())
                    std::memcpy(mxGetData(it_out), it.data(), it.size() * sizeof(uint8_t));
                mxSetCell(plhs[17], o, it_out);
            }
        }
    };

    CALL_QD(qd_qrt_read_chunked<qd_qrt_no_unlock>(fn, i_cir, i_orig, uplink, normalize_polmat, nlhs < 1 ? 1 : nlhs, init, consume));
}
