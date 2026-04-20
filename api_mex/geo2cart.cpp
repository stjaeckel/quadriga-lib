// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_math.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# GEO2CART
Convert elementwise azimuth/elevation angles to Cartesian coordinates

## Description:
- Conversion: `x = cos(el) cos(az) len`, `y = cos(el) sin(az) len`, `z = sin(el) len`
- Optional outputs `sAZ`, `cAZ`, `sEL`, `cEL` return intermediate sin/cos values; omit from the
  output list to skip their computation
- Defaults to the GENERIC kernel (`use_kernel=1`) to preserve full double precision, matching
  MATLAB's default numeric type
- Set `use_kernel=0` for auto-selection or `use_kernel=2` to force AVX2; the AVX2 kernel
  computes in single precision internally (inputs narrowed to float, results widened back)

## Usage:
```
split = true;
[ x, y, z, sAZ, cAZ, sEL, cEL ] = quadriga_lib.fast_geo2cart( az, el, len, use_kernel, split );

split = false;
cart = quadriga_lib.fast_geo2cart( az, el, len, use_kernel, split );
```

## Input Arguments:
- **`az`** — Azimuth angles in radians; `[n, m]`
- **`el`** — Elevation angles in radians; `[n, m]`
- **`len`** *(optional)* — Euclidean vector length sqrt(x^2 + y^2 + z^2); `[n, m]`; default: 1
- **`use_kernel`** *(optional)* — Kernel selection: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (throws if AVX2 unavailable); default: 1
- **`split`** *(optional)* — If true, return x/y/z and optional sin/cos as separate `[n, m]`
  matrices. If false, return a single combined `[3, n, m]` cube; sin/cos outputs unavailable
  in this mode; default: false

## Output Arguments:
- **`x_or_cart`** — If `split=true`: X-coordinates `[n, m]`. If `split=false`: combined cube
  with components along the first dim, `[3, n, m]`
- **`y`** — Y-coordinates; `[n, m]` or empty
- **`z`** — Z-coordinates; `[n, m]` or empty
- **`sAZ`** *(optional)* — sin(az); `[n, m]` or empty
- **`cAZ`** *(optional)* — cos(az); `[n, m]` or empty
- **`sEL`** *(optional)* — sin(el); `[n, m]` or empty
- **`cEL`** *(optional)* — cos(el); `[n, m]` or empty
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    // Read input data
    arma::mat az = qd_mex_get_Mat<double>(prhs[0]);
    arma::mat el = qd_mex_get_Mat<double>(prhs[1]);
    arma::mat len = (nrhs < 3) ? arma::mat() : qd_mex_get_Mat<double>(prhs[2]);
    const int use_kernel = (nrhs < 4) ? 1 : qd_mex_get_scalar<int>(prhs[3], "use_kernel", 1);
    const bool split = (nrhs < 5) ? false : qd_mex_get_scalar<bool>(prhs[4], "split", false);

    if (split && nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    else if (!split && nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::uword nm = az.n_elem, n = az.n_rows, m = az.n_cols;

    if (el.n_rows != n || el.n_cols != m)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "az and el must have the same shape.");
    if (!len.empty() && (len.n_rows != n || len.n_cols != m))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "len must have the same shape as az.");

    // Output allocation
    arma::cube cart;
    arma::mat x, y, z, sAZ, cAZ, sEL, cEL;

    if (split)
    {
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&x, n, m);
        else
            x.set_size(n, m);

        if (nlhs > 1)
            plhs[1] = qd_mex_init_output(&y, n, m);
        else
            y.set_size(n, m);

        if (nlhs > 2)
            plhs[2] = qd_mex_init_output(&z, n, m);
        else
            z.set_size(n, m);

        if (nlhs > 3)
            plhs[3] = qd_mex_init_output(&sAZ, n, m);
        if (nlhs > 4)
            plhs[4] = qd_mex_init_output(&cAZ, n, m);
        if (nlhs > 5)
            plhs[5] = qd_mex_init_output(&sEL, n, m);
        if (nlhs > 6)
            plhs[6] = qd_mex_init_output(&cEL, n, m);
    }
    else
    {
        x.set_size(n, m);
        y.set_size(n, m);
        z.set_size(n, m);
        if (nlhs > 0)
            plhs[0] = qd_mex_init_output(&cart, 3, n, m);
    }

    // Map to vectors
    const arma::vec az_vec(az.memptr(), nm, false, true);
    const arma::vec el_vec(el.memptr(), nm, false, true);
    const arma::vec len_vec = len.empty() ? arma::vec() : arma::vec(len.memptr(), nm, false, true);
    arma::vec x_vec(x.memptr(), nm, false, true);
    arma::vec y_vec(y.memptr(), nm, false, true);
    arma::vec z_vec(z.memptr(), nm, false, true);
    arma::vec sAZ_vec = sAZ.empty() ? arma::vec() : arma::vec(sAZ.memptr(), nm, false, true);
    arma::vec cAZ_vec = cAZ.empty() ? arma::vec() : arma::vec(cAZ.memptr(), nm, false, true);
    arma::vec sEL_vec = sEL.empty() ? arma::vec() : arma::vec(sEL.memptr(), nm, false, true);
    arma::vec cEL_vec = cEL.empty() ? arma::vec() : arma::vec(cEL.memptr(), nm, false, true);

    // Wrap optional pointers
    const arma::vec *p_len = len_vec.empty() ? nullptr : &len_vec;
    arma::vec *p_sAZ = sAZ_vec.empty() ? nullptr : &sAZ_vec;
    arma::vec *p_cAZ = cAZ_vec.empty() ? nullptr : &cAZ_vec;
    arma::vec *p_sEL = sEL_vec.empty() ? nullptr : &sEL_vec;
    arma::vec *p_cEL = cEL_vec.empty() ? nullptr : &cEL_vec;

    // Call library function
    CALL_QD(quadriga_lib::fast_geo2cart<double>(az_vec, el_vec, x_vec, y_vec, z_vec,
                                                p_sAZ, p_cAZ, p_sEL, p_cEL, p_len, use_kernel));

    // Combine outputs
    if (!split && nlhs > 0)
    {
        double *pd = cart.memptr();
        const double *px = x.memptr(), *py = y.memptr(), *pz = z.memptr();
        for (arma::uword i = 0; i < nm; ++i)
        {
            pd[3 * i] = px[i];
            pd[3 * i + 1] = py[i];
            pd[3 * i + 2] = pz[i];
        }
    }
}