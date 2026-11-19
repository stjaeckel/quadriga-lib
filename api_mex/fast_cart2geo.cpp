// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# FAST_CART2GEO
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: `len = sqrt(x² + y² + z²)`, `az = atan2(y, x)`, `el = asin(clamp(z / len, -1, 1))`
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- `z/len` is clamped to [-1, 1] before `asin` to guard against `len == 0` and rounding artifacts
  pushing `abs(z/len)` slightly above 1
- Azimuth in [-pi, pi], with the special values of `atan2` (e.g. `atan2(0, -0) = pi`); NaN input gives NaN
  in `az`, `el` and `len`
- The AVX2 kernel computes the angles in single precision; `len` and `z/len` in double precision
- `x`, `y` and `z` have the same size; the outputs have that size; empty input gives empty outputs

## Usage:
```
[ az, el, len ] = quadriga_lib.fast_cart2geo( x, y, z, use_kernel );
```

## Inputs:
- **`x`** — X-coordinates; up to 4 dimensions
- **`y`** — Y-coordinates; size of `x`
- **`z`** — Z-coordinates; size of `x`
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 
  (throws if AVX2 unavailable); other values throw; default: 0

## Outputs:
- **`az`** — Azimuth angles in radians; size of `x`
- **`el`** — Elevation angles in radians; size of `x`
- **`len`** — Euclidean vector length `sqrt(x² + y² + z²)`; size of `x`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs as flat column vectors (column-major order)
    const arma::vec x = qd_mex_get_Col<double>(prhs[0], false, "x");
    const arma::vec y = qd_mex_get_Col<double>(prhs[1], false, "y");
    const arma::vec z = qd_mex_get_Col<double>(prhs[2], false, "z");
    const int use_kernel = (nrhs < 4) ? 0 : qd_mex_get_scalar<int>(prhs[3], "use_kernel", 0);

    if (!qd_mex_same_dims(prhs[1], prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'y' must have the same size as 'x'.");
    if (!qd_mex_same_dims(prhs[2], prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'z' must have the same size as 'x'.");

    // Outputs in the size of x
    arma::vec az, el, len;
    plhs[0] = qd_mex_init_output_like(&az, prhs[0]);
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output_like(&el, prhs[0]);
    if (nlhs > 2)
        plhs[2] = qd_mex_init_output_like(&len, prhs[0]);

    // Call library function; it sizes el when it is not an output
    CALL_QD(quadriga_lib::fast_cart2geo(x, y, z, az, el, (nlhs > 2) ? &len : nullptr, use_kernel));
}
