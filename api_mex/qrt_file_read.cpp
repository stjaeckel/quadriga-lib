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
# QRT_FILE_READ
Read ray-tracing data from a QRT file

## Description:
- Reads channel impulse response (CIR) data from a QRT file for a specific snapshot and origin point.
- Supports both uplink and downlink directions by swapping TX/RX roles accordingly.
- All output arguments are optional; MATLAB only computes outputs that are requested.
- The `normalize_M` parameter controls how the polarization transfer matrix `M` and path gains are returned.

## Usage:
```
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
  path_gain, path_length, M, aod, eod, aoa, eoa, path_coord ] = ...
    qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

## Input Arguments:
- **`fn`**<br>
  Path to the QRT file, string.

- **`i_cir`** (optional)<br>
  Snapshot index (1-based), scalar. Default: 1

- **`i_orig`** (optional)<br>
  Origin index (1-based). For downlink, origin corresponds to the transmitter, scalar. Default: 1

- **`downlink`** (optional)<br>
  If `true`, origin is TX and destination is RX (downlink). If `false`, roles are swapped (uplink), 
  logical scalar. Default: `true`

- **`normalize_M`** (optional)<br>
  Normalization option for the polarization transfer matrix, scalar integer. Default: 1
   0 | `M` as stored in QRT file, `path_gain` is -FSPL
   1 | `M` has sum-column power of 2, `path_gain` is -FSPL minus material losses

## Output Arguments:
- **`center_freq`**<br>
  Center frequency in Hz, double vector of size `[n_freq, 1]`.

- **`tx_pos`**<br>
  Transmitter position in Cartesian coordinates, double vector of size `[3, 1]`.

- **`tx_orientation`**<br>
  Transmitter orientation (bank, tilt, heading) in radians, double vector of size `[3, 1]`.

- **`rx_pos`**<br>
  Receiver position in Cartesian coordinates, double vector of size `[3, 1]`.

- **`rx_orientation`**<br>
  Receiver orientation (bank, tilt, heading) in radians, double vector of size `[3, 1]`.

- **`fbs_pos`**<br>
  First-bounce scatterer positions, double matrix of size `[3, n_path]`.

- **`lbs_pos`**<br>
  Last-bounce scatterer positions, double matrix of size `[3, n_path]`.

- **`path_gain`**<br>
  Path gain on linear scale, double matrix of size `[n_path, n_freq]`.

- **`path_length`**<br>
  Absolute path length from TX to RX phase center, double vector of size `[n_path, 1]`.

- **`M`**<br>
  Polarization transfer matrix, double array of size `[8, n_path, n_freq]` or `[2, n_path, n_freq]` 
  for v6 files.

- **`aod`**<br>
  Departure azimuth angles in radians, double vector of size `[n_path, 1]`.

- **`eod`**<br>
  Departure elevation angles in radians, double vector of size `[n_path, 1]`.

- **`aoa`**<br>
  Arrival azimuth angles in radians, double vector of size `[n_path, 1]`.

- **`eoa`**<br>
  Arrival elevation angles in radians, double vector of size `[n_path, 1]`.

- **`path_coord`**<br>
  Interaction coordinates per path, cell array of length `n_path`. Each cell contains a double matrix 
  of size `[3, n_interact + 2]`.

## Example:
```
[center_freq, tx_pos, ~, rx_pos, ~, fbs_pos, lbs_pos, path_gain, path_length, M, ...
    aod, eod, aoa, eoa] = qrt_file_read('scene.qrt', 0, 0, true, 1);
```
MD!*/


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    std::string fn = qd_mex_get_string(prhs[0]);
    arma::uword i_cir = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "i_cir", 1);
    arma::uword i_orig = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "i_orig", 1);
    bool downlink = (nrhs < 4) ? true : qd_mex_get_scalar<bool>(prhs[3], "downlink", true);
    int normalize_M = (nrhs < 5) ? 1 : qd_mex_get_scalar<int>(prhs[4], "normalize_M", 1);

    i_cir -= 1;
    i_orig -=1;

    // Declare outputs
    arma::vec center_frequency, tx_pos, tx_orientation, rx_pos, rx_orientation;
    arma::mat fbs_pos, lbs_pos, path_gain;
    arma::vec path_length;
    arma::cube M;
    arma::vec aod, eod, aoa, eoa;
    std::vector<arma::mat> path_coord;

    // Set up optional output pointers based on nlhs
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

    // Call library function
    CALL_QD(quadriga_lib::qrt_file_read<double>(fn, i_cir, i_orig, downlink,
                                                p_center_frequency, p_tx_pos, p_tx_orientation,
                                                p_rx_pos, p_rx_orientation,
                                                p_fbs_pos, p_lbs_pos, p_path_gain, p_path_length, p_M,
                                                p_aod, p_eod, p_aoa, p_eoa,
                                                p_path_coord, normalize_M));

    // Write to MATLAB
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
        // path_coord is a vector of matrices - output as cell array
        mwSize n_path = path_coord.size();
        plhs[14] = mxCreateCellMatrix(1, n_path);
        for (mwSize i = 0; i < n_path; i++)
            mxSetCell(plhs[14], i, qd_mex_copy2matlab(&path_coord[i]));
    }
}