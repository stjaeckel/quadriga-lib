// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
# BASEBAND_FREQ_RESPONSE
Transforms the channel into frequency domain and returns the frequency response

## Usage:

```
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, pilot_grid, bandwidth, i_snap );
```

## Input Arguments:
- **`coeff_re`**<br>
  Channel coefficients, real part, Size: `[ n_rx, n_tx, n_path, n_snap ]`

- **`coeff_im`**<br>
  Channel coefficients, imaginary part, Size: `[ n_rx, n_tx, n_path, n_snap ]`

- **`delays`**<br>
  Propagation delay in seconds, Size: `[ n_rx, n_tx, n_path, n_snap ]` or `[ 1, 1, n_path, n_snap ]` 
  or `[ n_path, n_snap ]`

- **`pilot_grid`**<br>
  Sub-carrier positions relative to the bandwidth. The carrier positions are given relative to the
  bandwidth where '0' is the begin of the spectrum (i.e., the center frequency f0) and '1' is
  equal to f0+bandwidth. To obtain the channel frequency response centered around f0, the
  input variable 'pilot_grid' must be set to '(-N/2:N/2)/N', where N is the number of sub-
  carriers. Vector of length: `[ n_carriers ]`

- **`bandwidth`**<br>
  The baseband bandwidth in [Hz], scalar

- **`i_snap`** (optional)<br>
  Snapshot indices for which the frequency response should be generated (1-based index). If this
  variable is not given, all snapshots are processed. Length: `[ n_out ]`

## Output Argument:
- **`hmat_re`**<br>
  Freq. domain channel matrices (H), real part, Size `[ n_rx, n_tx, n_carriers, n_out ]`

- **`hmat_im`**<br>
  Freq. domain channel matrices (H), imaginary part, Size `[ n_rx, n_tx, n_carriers, n_out ]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - coeff_re        Channel coefficients, real part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  1 - coeff_im        Channel coefficients, imaginary part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  2 - delay           Path delays in seconds, size [n_rx, n_tx, n_path, n_snap] or [n_path, n_snap], zero-padded
    //  3 - pilot_grid      Sub-carrier positions, relative to the bandwidth, 0.0 = fc, 1.0 = fc+bandwidth, Size: [ n_carriers ]
    //  4 - bandwidth       The baseband bandwidth in [Hz], scalar
    //  5 - i_snap          Snapshot indices, 1-based, optional, vector of length "n_out"

    // Outputs:
    //  0 - hmat_re         Channel matrices (H), real part, Size [n_rx, n_tx, n_carriers, n_out]
    //  1 - hmat_im         Channel matrices (H), imaginary part, Size [n_rx, n_tx, n_carriers, n_out]

    // Number of in and outputs
    if (nrhs < 5 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:baseband_freq_response:no_input", "Incorrect number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:baseband_freq_response:no_output", "Incorrect number of output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:baseband_freq_response:IO_error", "Inputs must be provided in 'single' or 'double' precision.");

    // Read inputs
    std::vector<arma::fcube> coeff_re_single, coeff_im_single, delay_single;
    arma::fvec pilot_grid_single;

    std::vector<arma::cube> coeff_re_double, coeff_im_double, delay_double;
    arma::vec pilot_grid_double;

    if (use_single)
    {
        coeff_re_single = qd_mex_matlab2vector_Cube<float>(prhs[0], 3);
        coeff_im_single = qd_mex_matlab2vector_Cube<float>(prhs[1], 3);
        pilot_grid_single = qd_mex_typecast_Col<float>(prhs[3], "pilot_grid");

        size_t n_snap = coeff_re_single.size();
        size_t n_dim = (size_t)mxGetNumberOfDimensions(prhs[2]);
        size_t n_cols = (size_t)mxGetN(prhs[2]);
        if (n_dim == 2 && n_cols == n_snap)
        {
            auto tmp = qd_mex_matlab2vector_Cube<float>(prhs[2], 1);
            for (auto &d : tmp)
                delay_single.push_back(arma::Cube<float>(d.memptr(), 1, 1, d.n_elem, true));
        }
        else
            delay_single = qd_mex_matlab2vector_Cube<float>(prhs[2], 3);
    }
    else
    {
        coeff_re_double = qd_mex_matlab2vector_Cube<double>(prhs[0], 3);
        coeff_im_double = qd_mex_matlab2vector_Cube<double>(prhs[1], 3);
        pilot_grid_double = qd_mex_typecast_Col<double>(prhs[3], "pilot_grid");

        size_t n_snap = coeff_re_double.size();
        size_t n_dim = (size_t)mxGetNumberOfDimensions(prhs[2]);
        size_t n_cols = (size_t)mxGetN(prhs[2]);
        if (n_dim == 2 && n_cols == n_snap)
        {
            auto tmp = qd_mex_matlab2vector_Cube<double>(prhs[2], 1);
            for (auto &d : tmp)
                delay_double.push_back(arma::Cube<double>(d.memptr(), 1, 1, d.n_elem, true));
        }
        else
            delay_double = qd_mex_matlab2vector_Cube<double>(prhs[2], 3);
    }

    double bandwidth = qd_mex_get_scalar<double>(prhs[4], "bandwidth");

    arma::u32_vec i_snap;
    if (nrhs > 5)
    {
        i_snap = qd_mex_typecast_Col<unsigned>(prhs[5], "i_snap");
        i_snap = i_snap - 1;
    }

    std::vector<arma::fcube> hmat_re_single, hmat_im_single;
    std::vector<arma::cube> hmat_re_double, hmat_im_double;

    try
    {
        if (use_single)
        {
            quadriga_lib::baseband_freq_response_vec<float>(&coeff_re_single, &coeff_im_single, &delay_single, &pilot_grid_single,
                                                            bandwidth, &hmat_re_single, &hmat_im_single, &i_snap);
        }
        else // double
        {
            quadriga_lib::baseband_freq_response_vec<double>(&coeff_re_double, &coeff_im_double, &delay_double, &pilot_grid_double,
                                                             bandwidth, &hmat_re_double, &hmat_im_double, &i_snap);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:baseband_freq_response:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:baseband_freq_response:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_vector2matlab(&hmat_re_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_vector2matlab(&hmat_re_double);

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_vector2matlab(&hmat_im_single);
    else if (nlhs > 1)
        plhs[1] = qd_mex_vector2matlab(&hmat_im_double);
}