// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# FAST_SINCOS
Fast, approximate sine/cosine for MATLAB numeric arrays

## Description:
Computes elementwise sine and/or cosine for input angles in radians.
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision
- Results are approximate and may differ from MATLAB `sin` / `cos`
- For x in [-pi, pi], the maximum absolute error is 2^(-22.1), and larger otherwise
- For x in [-500, 500], the maximum absolute error is 2^(-16.0)
- Request one or two outputs to control which results are returned
- With one output, set the optional `cosineOnly` flag to `true` to return cosine instead of sine

## Usage:

```
[s, c] = arrayant_lib.fast_sincos(x);
s = arrayant_lib.fast_sincos(x);
c = arrayant_lib.fast_sincos(x, true);
```

## Input Arguments:
- `**x**` (input)<br>
  Numeric array of angles in radians. Any size/shape.

- `**cosineOnly** = false` (optional input)<br>
  Logical scalar. When requesting a single output, set to `true` to return `cos(x)`; otherwise returns
  `sin(x)`.

## Output Arguments:
- `**s**`<br>
  Single-precision `sin(x)`. Same size as `x`.

- `**c**`<br>
  Single-precision `cos(x)`. Same size as `x`.

## Examples:

```
% Input as single for best performance
x = single(linspace(0, 2*pi, 1000));

% Compute sine and cosine
[s, c] = arrayant_lib.fast_sincos(x);

% Compute only sine (single output)
s = arrayant_lib.fast_sincos(x);

% Compute only cosine (single output with flag)
c = arrayant_lib.fast_sincos(x, true);

% Double input is accepted; outputs remain single
xd = linspace(0, 2*pi, 8);
s_only = arrayant_lib.fast_sincos(xd);        % class(s_only) == 'single'
c_only = arrayant_lib.fast_sincos(xd, true);  % class(c_only) == 'single'
```
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    if (nlhs == 0)
        return;

    // Read first input as cube, convert to single precision if needed
    arma::fcube Xs;
    arma::cube Xd;
    bool use_double_input = mxIsDouble(prhs[0]);

    if (use_double_input)
        Xd = qd_mex_reinterpret_Cube<double>(prhs[0]); // No copy
    else if (mxIsSingle(prhs[0]))
        Xs = qd_mex_reinterpret_Cube<float>(prhs[0]); // No copy
    else
        Xs = qd_mex_typecast_Cube<float>(prhs[0]);

    arma::uword n_rows = use_double_input ? Xd.n_rows : Xs.n_rows;
    arma::uword n_cols = use_double_input ? Xd.n_cols : Xs.n_cols;
    arma::uword n_slices = use_double_input ? Xd.n_slices : Xs.n_slices;
    arma::uword n_elem = use_double_input ? Xd.n_elem : Xs.n_elem;

    // Check if we want to get only the cosine
    bool cos_only = (nrhs < 2) ? false : qd_mex_get_scalar<bool>(prhs[1], "cosineOnly");

    // Allocate memory for the outputs
    arma::fcube P, Q;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&P, n_rows, n_cols, n_slices);

    if (nlhs > 1 && !cos_only)
        plhs[1] = qd_mex_init_output(&Q, n_rows, n_cols, n_slices);
    else if (nlhs > 1 && cos_only)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Cannot have 2 outputs in cosine-only mode.");

    // Empty input
    if (n_elem == 0)
        return;

    // Serialize cubes to vectors (map memory)
    arma::fvec xs;
    arma::vec xd;
    if (use_double_input)
        xd = arma::vec(Xd.memptr(), n_elem, false, true);
    else
        xs = arma::fvec(Xs.memptr(), n_elem, false, true);

    arma::fvec p = arma::fvec(P.memptr(), n_elem, false, true);

    arma::fvec q;
    if (n_elem != 0)
        q = arma::fvec(Q.memptr(), n_elem, false, true);

    // Call Quadriga-Lib C++ API
    if (use_double_input)
    {
        if (cos_only)
            CALL_QD(quadriga_lib::fast_sincos(xd, nullptr, &p));
        else if (nlhs > 1)
            CALL_QD(quadriga_lib::fast_sincos(xd, &p, &q));
        else
            CALL_QD(quadriga_lib::fast_sincos(xd, &p));
    }
    else
    {
        if (cos_only)
            CALL_QD(quadriga_lib::fast_sincos(xs, nullptr, &p));
        else if (nlhs > 1)
            CALL_QD(quadriga_lib::fast_sincos(xs, &p, &q));
        else
            CALL_QD(quadriga_lib::fast_sincos(xs, &p));
    }
}