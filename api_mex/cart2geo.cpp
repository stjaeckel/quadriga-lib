// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# CART2GEO
Convert elementwise Cartesian coordinates to azimuth/elevation angles and vector length

- Computes: `len = sqrt(x² + y² + z²)`, `az = atan2(y, x)`, `el = asin(clamp(z / len, -1, 1))`
- Inputs are arbitrary 3D vectors (not required to be unit-length); `len` returns the Euclidean norm
- `z/len` is clamped to [-1, 1] before `asin` to guard against `len == 0` and rounding artifacts
  pushing `abs(z/len)` slightly above 1
- Option to provide a single `[3, n, m]` cube or separate x, y, z `[n, m]` inputs

## Usage:
```
[ az, el, len ] = quadriga_lib.cart2geo( x, y, z, use_kernel );
```

## Inputs:
- **`x`** — X-coordinates or combined input; `[n, m]` or `[3, n, m]`
- **`y`** — Y-coordinates; `[n, m]` or empty
- **`z`** — Z-coordinates; `[n, m]` or empty
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); default: 1

## Outputs:
- **`az`** — Azimuth angles in radians; `[n, m]`
- **`el`** — Elevation angles in radians; `[n, m]`
- **`len`** *(optional)* — Euclidean vector length `sqrt(x² + y² + z²)`; `[n, m]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    arma::cube x = qd_mex_get_Cube<double>(prhs[0]);
    arma::mat y = (nrhs < 2) ? arma::mat() : qd_mex_get_Mat<double>(prhs[1]);
    arma::mat z = (nrhs < 3) ? arma::mat() : qd_mex_get_Mat<double>(prhs[2]);
    const int use_kernel = (nrhs < 4) ? 1 : qd_mex_get_scalar<int>(prhs[3], "use_kernel", 1);

    // Split data
    arma::uword nm = x.n_elem, n = x.n_cols, m = x.n_slices;
    arma::vec x_vec, y_vec, z_vec;
    if (x.n_rows == 3 && y.empty() && z.empty())
    {
        nm = n * m;
        x_vec.set_size(nm), y_vec.set_size(nm), z_vec.set_size(nm);
        const double *pd = x.memptr();
        double *px = x_vec.memptr(), *py = y_vec.memptr(), *pz = z_vec.memptr();
        for (arma::uword i = 0; i < nm; ++i)
            px[i] = pd[3 * i], py[i] = pd[3 * i + 1], pz[i] = pd[3 * i + 2];
    }
    else if (y.n_elem == nm && z.n_elem == nm)
    {
        x_vec = arma::vec(x.memptr(), nm, false, true);
        y_vec = arma::vec(y.memptr(), nm, false, true);
        z_vec = arma::vec(z.memptr(), nm, false, true);
        n = y.n_rows, m = y.n_cols;
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong input argument format.");

    // Output allocation
    arma::mat az, el, len;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&az, n, m);
    else
        az.set_size(n, m);

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&el, n, m);
    else
        el.set_size(n, m);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&len, n, m);

    // Map to vectors
    arma::vec az_vec(az.memptr(), nm, false, true);
    arma::vec el_vec(el.memptr(), nm, false, true);
    arma::vec len_vec = len.empty() ? arma::vec() : arma::vec(len.memptr(), nm, false, true);

    arma::vec *p_len_vec = (nlhs > 2) ? &len_vec : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::fast_cart2geo(x_vec, y_vec, z_vec, az_vec, el_vec, p_len_vec, use_kernel));
}