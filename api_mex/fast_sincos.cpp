// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_math.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# FAST_SINCOS
Compute elementwise approximate sine and/or cosine of a vector

- AVX2-optimized (8 floats/lane); scalar fallback without AVX2
- For x in [-pi, pi]: max absolute error = 2^(-22.1); for x in [-500, 500]: 2^(-16.0)
- Either `s` or `c` may be `nullptr` to skip that computation
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision
- Request one or two outputs to control which results are returned
- With one output, set the optional `cos_only` flag to `true` to return cosine instead of sine

## Usage:
```
[s, c] = quadriga_lib.fast_sincos(x);
s = quadriga_lib.fast_sincos(x);
c = quadriga_lib.fast_sincos(x, true);
```

## Inputs:
- **`x`** (input) — Input angles; radians; `[n_elem]`
- **`cos_only`** — Forsingle output: `true` returns `cos(x)`; false returns `sin(x)`; default: false

## Outputs:
- **`s`** — sin(x); `[n_elem]`
- **`c`** — cos(x); `[n_elem]`
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
    {
        Xd.~Cube<double>();
        ::new (&Xd) arma::cube(qd_mex_reinterpret_Cube<double>(prhs[0]));
    }
    else if (mxIsSingle(prhs[0]))
    {
        Xs.~Cube<float>();
        ::new (&Xs) arma::fcube(qd_mex_reinterpret_Cube<float>(prhs[0]));
    }
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
    {
        xd.~Col<double>();
        ::new (&xd) arma::vec(Xd.memptr(), n_elem, false, true);
    }
    else
    {
        xs.~Col<float>();
        ::new (&xs) arma::fvec(Xs.memptr(), n_elem, false, true);
    }

    arma::fvec p = arma::fvec(P.memptr(), n_elem, false, true);

    arma::fvec q;
    if (n_elem != 0)
    {
        q.~Col<float>();
        ::new (&q) arma::fvec(Q.memptr(), n_elem, false, true);
    }

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