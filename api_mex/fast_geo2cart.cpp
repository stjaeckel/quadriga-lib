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
# FAST_GEO2CART
Convert elementwise azimuth/elevation angles to Cartesian coordinates

- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sin_az`, `cos_az`, `sin_el`, `cos_el` return intermediate sin/cos values; omit from the
  output list to skip their computation
- The AVX2 kernel (selected by `use_kernel=0` where available, forced by `use_kernel=2`) computes sine and
  cosine in single precision (angles reduced in double precision first; max. error about 1.3e-7 for angles up
  to 1e10), results widened back; `use_kernel=1` keeps full double precision
- `el` and `len` have the same size as `az`; the outputs have that size; empty input gives empty outputs

## Usage:
```
[ x, y, z, sin_az, cos_az, sin_el, cos_el ] = quadriga_lib.fast_geo2cart( az, el, len, use_kernel );
```

## Inputs:
- **`az`** — Azimuth angles in radians; up to 4 dimensions
- **`el`** — Elevation angles in radians; size of `az`
- **`len`** — Euclidean vector length sqrt(x^2 + y^2 + z^2); size of `az`; default: 1
- **`use_kernel`** — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); other values throw; default: 0

## Outputs:
- **`x`** — X-coordinates; size of `az`
- **`y`** — Y-coordinates; size of `az`
- **`z`** — Z-coordinates; size of `az`
- **`sin_az`** — sin(az); size of `az`
- **`cos_az`** — cos(az); size of `az`
- **`sin_el`** — sin(el); size of `az`
- **`cos_el`** — cos(el); size of `az`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs as flat column vectors (column-major order); len empty = unit length
    const arma::vec az = qd_mex_get_Col<double>(prhs[0], false, "az");
    const arma::vec el = qd_mex_get_Col<double>(prhs[1], false, "el");
    const arma::vec len = (nrhs < 3) ? arma::vec() : qd_mex_get_Col<double>(prhs[2], false, "len");
    const int use_kernel = (nrhs < 4) ? 0 : qd_mex_get_scalar<int>(prhs[3], "use_kernel", 0);

    if (!qd_mex_same_dims(prhs[1], prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'el' must have the same size as 'az'.");
    if (!len.empty() && !qd_mex_same_dims(prhs[2], prhs[0]))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'len' must have the same size as 'az'.");

    // Outputs in the size of az; C++ sizes x, y, z when they are not requested, sin / cos are computed only on request
    arma::vec out[7];
    for (int i = 0; i < std::max(nlhs, 1); ++i)
        plhs[i] = qd_mex_init_output_like(&out[i], prhs[0]);

    CALL_QD(quadriga_lib::fast_geo2cart<double>(az, el, out[0], out[1], out[2],
                                                (nlhs > 3) ? &out[3] : nullptr, (nlhs > 4) ? &out[4] : nullptr,
                                                (nlhs > 5) ? &out[5] : nullptr, (nlhs > 6) ? &out[6] : nullptr,
                                                len.empty() ? nullptr : &len, use_kernel));
}
