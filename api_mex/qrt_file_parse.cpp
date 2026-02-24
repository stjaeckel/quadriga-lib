// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# QRT_FILE_PARSE
Read metadata from a QRT file

## Description:
- Parses a QRT file and extracts metadata such as the number of snapshots, origins, destinations, and frequencies.
- All output arguments are optional; MATLAB only computes outputs that are requested.
- Can also retrieve CIR offsets per destination, human-readable names for origins and destinations, and the file version.

## Usage:
```
[ no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, ...
  fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation ] = qrt_file_parse( fn );
```

## Input Argument:
- **`fn`**<br>
  Path to the QRT file, string.

## Output Arguments:
- **`no_cir`**<br>
  Number of channel snapshots per origin point, scalar.

- **`no_orig`**<br>
  Number of origin points (TX), scalar.

- **`no_dest`**<br>
  Number of destinations (RX), scalar.

- **`no_freq`**<br>
  Number of frequency bands, scalar.

- **`cir_offset`**<br>
  CIR offset for each destination, uint64 vector of size `[no_dest, 1]`.

- **`orig_names`**<br>
  Names of the origin points (TXs), cell array of strings with `no_orig` entries.

- **`dest_names`**<br>
  Names of the destination points (RXs), cell array of strings with `no_dest` entries.
  
- **`version`**<br>
  QRT file version number, scalar integer.

- **`fGHz`**<br>
  Center frequency in GHz, float vector of size `[no_freq, 1]`.

- **`cir_pos`**<br>
  CIR positions in Cartesian coordinates, float matrix of size `[no_cir, 3]`.

- **`cir_orientation`**<br>
  CIR orientation in Euler angles in rad, float matrix of size `[no_cir, 3]`.

- **`orig_pos`**<br>
  Origin (TX) positions in Cartesian coordinates, float matrix of size `[no_orig, 3]`.

- **`orig_orientation`**<br>
  Origin (TX) orientations in Euler angles in rad, float matrix of size `[no_orig, 3]`.

## Example:

```
[no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, ...
    fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation] = ...
    qrt_file_parse('scene.qrt');
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read filename
    std::string fn = qd_mex_get_string(prhs[0]);

    // Declare outputs
    arma::uword no_cir, no_orig, no_dest, no_freq;
    arma::uvec cir_offset;
    std::vector<std::string> orig_names, dest_names;
    int version;
    arma::fvec fGHz;
    arma::fmat cir_pos, cir_orientation, orig_pos, orig_orientation;

    // Set up optional output pointers based on nlhs
    arma::uword *p_no_cir = (nlhs > 0) ? &no_cir : nullptr;
    arma::uword *p_no_orig = (nlhs > 1) ? &no_orig : nullptr;
    arma::uword *p_no_dest = (nlhs > 2) ? &no_dest : nullptr;
    arma::uword *p_no_freq = (nlhs > 3) ? &no_freq : nullptr;
    arma::uvec *p_cir_offset = (nlhs > 4) ? &cir_offset : nullptr;
    std::vector<std::string> *p_orig_names = (nlhs > 5) ? &orig_names : nullptr;
    std::vector<std::string> *p_dest_names = (nlhs > 6) ? &dest_names : nullptr;
    int *p_version = (nlhs > 7) ? &version : nullptr;
    arma::fvec *p_fGHz = (nlhs > 8) ? &fGHz : nullptr;
    arma::fmat *p_cir_pos = (nlhs > 9) ? &cir_pos : nullptr;
    arma::fmat *p_cir_orientation = (nlhs > 10) ? &cir_orientation : nullptr;
    arma::fmat *p_orig_pos = (nlhs > 11) ? &orig_pos : nullptr;
    arma::fmat *p_orig_orientation = (nlhs > 12) ? &orig_orientation : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::qrt_file_parse(fn, p_no_cir, p_no_orig, p_no_dest, p_no_freq,
                                         p_cir_offset, p_orig_names, p_dest_names, p_version,
                                         p_fGHz, p_cir_pos, p_cir_orientation, p_orig_pos, p_orig_orientation));

    // Write to MATLAB
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
        plhs[8] = qd_mex_copy2matlab(&fGHz);
    if (nlhs > 9)
        plhs[9] = qd_mex_copy2matlab(&cir_pos);
    if (nlhs > 10)
        plhs[10] = qd_mex_copy2matlab(&cir_orientation);
    if (nlhs > 11)
        plhs[11] = qd_mex_copy2matlab(&orig_pos);
    if (nlhs > 12)
        plhs[12] = qd_mex_copy2matlab(&orig_orientation);
}