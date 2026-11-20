// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_math.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# FAST_SINCOS
Compute elementwise approximate sine and/or cosine of a vector

- Max absolute error about 1.3e-7 (2^-22.9) against the exact sine / cosine of the input value for
  abs(x) <= 1e10 (the argument is reduced in double precision where single precision is not enough)
- Results stay in [-1, 1] for all finite x; Inf and NaN give NaN
- Works on vectors, matrices, and 3-D arrays
- Accepts any numeric input class; best performance with single precision
- Outputs are always single precision, also for double input
- Request one or two outputs to control which results are returned
- With one output, set the optional `cos_only` flag to `true` to return cosine instead of sine

## Usage:
```
[sin_x, cos_x] = quadriga_lib.fast_sincos(x);
sin_x = quadriga_lib.fast_sincos(x);
cos_x = quadriga_lib.fast_sincos(x, true);
[sin_x, cos_x] = quadriga_lib.fast_sincos(x, [], use_kernel);
```

## Inputs:
- **`x`** — Input angles; radians; vector, matrix or 3-D array
- **`cos_only`** — For a single output: `true` returns `cos(x)`; `false` or `[]` returns `sin(x)`; default: false
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

## Outputs:
- **`sin_x`** — sin(x); single; size of `x`
- **`cos_x`** — cos(x); single; size of `x`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Double and single inputs are aliased, other classes are converted to single
    arma::fcube Xs;
    arma::cube Xd;
    bool use_double_input = mxIsDouble(prhs[0]);

    if (use_double_input)
    {
        Xd.~Cube<double>();
        ::new (&Xd) arma::cube(qd_mex_reinterpret_Cube<double>(prhs[0], "x"));
    }
    else if (mxIsSingle(prhs[0]))
    {
        Xs.~Cube<float>();
        ::new (&Xs) arma::fcube(qd_mex_reinterpret_Cube<float>(prhs[0], "x"));
    }
    else
        Xs = qd_mex_typecast_Cube<float>(prhs[0], "x");

    arma::uword n_rows = use_double_input ? Xd.n_rows : Xs.n_rows;
    arma::uword n_cols = use_double_input ? Xd.n_cols : Xs.n_cols;
    arma::uword n_slices = use_double_input ? Xd.n_slices : Xs.n_slices;
    arma::uword n_elem = use_double_input ? Xd.n_elem : Xs.n_elem;

    bool cos_only = (nrhs < 2) ? false : qd_mex_get_scalar<bool>(prhs[1], "cos_only", false);
    const int use_kernel = (nrhs < 3) ? 0 : qd_mex_get_scalar<int>(prhs[2], "use_kernel", 0);

    arma::fcube P, Q;
    plhs[0] = qd_mex_init_output(&P, n_rows, n_cols, n_slices);

    if (nlhs > 1 && !cos_only)
        plhs[1] = qd_mex_init_output(&Q, n_rows, n_cols, n_slices);
    else if (nlhs > 1 && cos_only)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'cos_only' must be false when two outputs are requested.");

    // Flat vector views on the cube memory
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
    if (Q.n_elem != 0)
    {
        q.~Col<float>();
        ::new (&q) arma::fvec(Q.memptr(), n_elem, false, true);
    }

    if (use_double_input)
    {
        if (cos_only)
            CALL_QD(quadriga_lib::fast_sincos(xd, nullptr, &p, use_kernel));
        else if (nlhs > 1)
            CALL_QD(quadriga_lib::fast_sincos(xd, &p, &q, use_kernel));
        else
            CALL_QD(quadriga_lib::fast_sincos(xd, &p, nullptr, use_kernel));
    }
    else
    {
        if (cos_only)
            CALL_QD(quadriga_lib::fast_sincos(xs, nullptr, &p, use_kernel));
        else if (nlhs > 1)
            CALL_QD(quadriga_lib::fast_sincos(xs, &p, &q, use_kernel));
        else
            CALL_QD(quadriga_lib::fast_sincos(xs, &p, nullptr, use_kernel));
    }
}
