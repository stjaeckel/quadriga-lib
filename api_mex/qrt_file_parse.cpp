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
# QRT_FILE_PARSE
Read metadata from a QRT file

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count,
  CIR offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

## Usage:
```
[ no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
     cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

## Input:
- **`fn`** — Path to the QRT file; string

## Outputs:
- **`no_cir`** — Number of channel snapshots per origin point; uint64 scalar
- **`no_orig`** — Number of origin points (TX); uint64 scalar
- **`no_dest`** — Number of destination points (RX); uint64 scalar
- **`no_freq`** — Number of frequency bands; uint64 scalar
- **`cir_offset`** — CIR offset per destination; uint64; `[no_dest]`
- **`orig_names`** — Names of origin points; cell array of strings; `[no_orig]`
- **`dest_names`** — Names of destination points; cell array of strings; `[no_dest]`
- **`version`** — QRT file version number; int32 scalar
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); single; `[no_freq]`
- **`cir_pos`** — CIR positions in Cartesian coordinates; single; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles; single; `[no_cir, 3]`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; single; `[no_orig, 3]`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; single; `[no_orig, 3]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read filename
    const std::string fn = qd_mex_get_string(prhs[0]);

    // Declare outputs
    arma::uword no_cir, no_orig, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;
    arma::fvec center_freq;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    // Wrap optional output pointers based on nlhs
    arma::uword *p_no_cir = (nlhs > 0) ? &no_cir : nullptr;
    arma::uword *p_no_orig = (nlhs > 1) ? &no_orig : nullptr;
    arma::uword *p_no_dest = (nlhs > 2) ? &no_dest : nullptr;
    arma::uword *p_no_freq = (nlhs > 3) ? &no_freq : nullptr;
    arma::uvec *p_cir_offset = (nlhs > 4) ? &cir_offset : nullptr;
    std::vector<std::string> *p_orig_names = (nlhs > 5) ? &orig_names : nullptr;
    std::vector<std::string> *p_dest_names = (nlhs > 6) ? &dest_names : nullptr;
    int *p_version = (nlhs > 7) ? &version : nullptr;
    arma::fvec *p_center_freq = (nlhs > 8) ? &center_freq : nullptr;
    arma::fmat *p_cir_pos = (nlhs > 9) ? &cir_pos : nullptr;
    arma::fmat *p_cir_orientation = (nlhs > 10) ? &cir_orientation : nullptr;
    arma::fmat *p_orig_pos = (nlhs > 11) ? &orig_pos : nullptr;
    arma::fmat *p_orig_orientation = (nlhs > 12) ? &orig_orientation : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::qrt_file_parse(fn, p_no_cir, p_no_orig, p_no_dest, p_no_freq,
                                         p_cir_offset, p_orig_names, p_dest_names, p_version,
                                         p_center_freq, p_cir_pos, p_cir_orientation, p_orig_pos,
                                         p_orig_orientation));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&no_cir);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&no_orig);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&no_dest);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&no_freq);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&cir_offset);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&orig_names);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&dest_names);
    if (nlhs > 7)
        plhs[7] = qd_mex_copy2matlab(&version);
    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&center_freq);
    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&cir_pos);
    if (nlhs > 10)
        plhs[10] = qd_mex_copy2matlab(&cir_orientation);
    if (nlhs > 11)
        plhs[11] = qd_mex_copy2matlab(&orig_pos);
    if (nlhs > 12)
        plhs[12] = qd_mex_copy2matlab(&orig_orientation);
}