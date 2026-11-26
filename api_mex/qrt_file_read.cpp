// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"
#include "qd_helpers_wrappers.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# QRT_FILE_READ
Read the paths of CIRs from a QRT file

- Reads the paths of a list of CIRs of one origin, with the positions and orientations of both ends: the inputs of
  [[get_channels_spherical]] (`fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `polmat`) or
  [[get_channels_planar]] (the angles)
- Only the requested outputs are computed; with at most 5 outputs, no path data is read
- Reads in chunks of 16 CIRs: besides the outputs, at most two chunks are held in memory
- `uplink` swaps the ends: the origin becomes the receiver (see Notes)

## Usage:
```
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, uplink, normalize_polmat );
```

## Inputs:
- **`fn`** — File name
- **`i_cir`** — CIR indices, 1-based, at most `n_cir`; `[n_out, 1]`; default: `[]` (all CIRs of the file)
- **`i_orig`** — Origin index, 1-based, at most `n_orig`; default: 1
- **`uplink`** — `false`: origin = Tx, destination = Rx; `true`: origin = Rx, destination = Tx; default: `false`
- **`normalize_polmat`** — Scaling of `polmat` and `path_gain`, 0 or 1 (see Notes); default: 1

## Outputs:
- **`center_freq`** — Frequencies in Hz; `[n_freq, 1]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, n_out]`
- **`tx_orientation`**, **`rx_orientation`** — Tx and Rx orientations (bank, tilt, heading) in rad; `[3, n_out]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in m: the first and the last interaction
  point, the midpoint of Tx and Rx for a LOS path; cell array `[n_out, 1]` of `[3, n_path]`
- **`path_gain`** — Path gain, linear (see Notes); cell array `[n_out, 1]` of `[n_path, n_freq]`
- **`path_length`** — Length of the polyline from the Tx through the interaction points to the Rx in m; cell array
  `[n_out, 1]` of `[n_path, 1]`
- **`polmat`** — Polarization transfer matrix; cell array `[n_out, 1]` of `[8, n_path, n_freq]`; version 6: the
  complex scalar coefficient, `[2, n_path, n_freq]`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Azimuth and elevation of departure (at the Tx) and arrival (at the
  Rx) in rad; cell array `[n_out, 1]` of `[n_path, 1]`
- **`path_coord`** — Points of each path from the Tx through the interaction points to the Rx; cell array
  `[n_out, 1]` of cell arrays `[n_path, 1]` of `[3, n_interact + 2]`
- **`n_interact`** — Number of interaction points of each path, 0 = LOS; cell array `[n_out, 1]` of uint32
  `[n_path, 1]`
- **`coord`** — Interaction points of all paths, concatenated in path order; cell array `[n_out, 1]` of single
  `[3, sum(n_interact)]`
- **`interact_type`** — Interaction type codes (see [QRT file format: stored values](formats.html#qrt-stored-values)),
  concatenated like `coord`; cell array `[n_out, 1]` of uint8 `[sum(n_interact), 1]`

## Notes:
- **Path gain**: the file stores per path the coefficients with the interaction (material) losses; the reader adds the
  propagation loss of the path length $d$, for EM files (version 5) the free-space loss
  $\mathrm{PL} = 32.45 + 20 \log_{10}(f / \mathrm{GHz}) + 20 \log_{10}(d / \mathrm{m})$ dB, for scalar files (version 6)
  the spherical spreading and the air absorption $\alpha(f)$ of ISO 9613-1 at 20 °C, 50 % humidity and 1 atm,
  $\mathrm{PL} = 20 \log_{10}(d / \mathrm{m}) + \alpha(f) \, d$ dB. With $p$ the larger column power of the stored
  matrix (version 6: the power of the coefficient):<br><br>
  | `normalize_polmat` | `polmat` | `path_gain` |
  | ------------------ | -------- | ----------- |
  | 0 | As stored | $10^{-\mathrm{PL}/10}$ |
  | 1 | Stored matrix divided by $\sqrt{p}$: largest column power 1 (uplink: row power) | $p \cdot 10^{-\mathrm{PL}/10}$ |

- **Uplink**: `polmat` is the transpose of the downlink matrix (reciprocal channel, no conjugation) with the same
  `path_gain`; positions, orientations, angles, `fbs_pos`, `lbs_pos` and `path_coord` swap their ends.
  `n_interact`, `coord` and `interact_type` keep the stored order from the origin to the destination, and the type
  codes describe that traversal
- An origin slot that is not written yet gives zero positions and orientations and no paths
- **Errors**: an index out of range, or a file that cannot be opened, is not a [QRT file](formats.html#qrt) of version 5
  or 6, or is truncated or corrupt (counts or offsets beyond the file, path blocks of different CIRs that overlap). A
  message about one requested CIR starts with `Snapshot <k>: `, with `k` its position in `i_cir`

## See also:
- [[qrt_file_parse]] (metadata without path data)
- [[get_channels_spherical]] (coefficients from the paths)
- [[get_channels_planar]] (coefficients from the angles)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 18)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

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
