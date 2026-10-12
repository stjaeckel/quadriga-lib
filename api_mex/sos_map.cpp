// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_MAP
Spatially correlated random values on a regular grid

- Evaluates the generators of [[sos_val]] for one device at all combinations of the grid vectors
- Output in meshgrid order: element `(iy, ix, iz, g)` belongs to the position
  `{x(ix), y(iy), z(iz)}` and to generator `g`, i.e. column `g` of `phase`
- The number of grid vectors given must not exceed the number of columns of `freq`; an omitted or
  empty `y` or `z` means a single sample at 0 on that axis
- `z` can be given without `y` if `freq` has 3 columns

## Usage:
```
map = quadriga_lib.sos_map( freq, phase, x, y, z, uniform, use_kernel );
```

## Inputs:
- **`freq`** — Frequencies in [1/m]; `n_dim` = 1, 2 or 3; `[n_coeff, n_dim]`
- **`phase`** — Phases in [rad]; one column per generator; `[n_coeff, n_gen]`
- **`x`** — Grid points on the x-axis; `[nx]`
- **`y`** *(optional)* — Grid points on the y-axis; requires `n_dim >= 2`; omitted or empty = one
  sample at 0, `ny = 1`; `[ny]`
- **`z`** *(optional)* — Grid points on the z-axis; requires `n_dim = 3`; omitted or empty = one
  sample at 0, `nz = 1`; `[nz]`
- **`uniform`** — Output distribution: false = Normal(0,1), true = Uniform between 0 and 1;
  default: false
- **`use_kernel`** — Compute kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2;
  default: 0

## Output:
- **`map`** — Values; `[ny, nx, nz, n_gen]`

## See also:
- [[sos_val]] (evaluation at arbitrary positions, two devices)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 3 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const arma::mat freq = qd_mex_get_Mat<double>(prhs[0]);
    const arma::mat phase = qd_mex_get_Mat<double>(prhs[1]);
    const arma::vec x = qd_mex_get_Col<double>(prhs[2]);
    const arma::vec y = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3]);
    const arma::vec z = (nrhs < 5) ? arma::vec() : qd_mex_get_Col<double>(prhs[4]);
    const bool uniform = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "uniform", false);
    const int use_kernel = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "use_kernel", 0);

    const arma::uword nx = x.n_elem, ny = y.empty() ? 1 : y.n_elem, nz = z.empty() ? 1 : z.n_elem, n_gen = phase.n_cols;

    // Output allocation (a zero size is left to the library to reject)
    std::vector<arma::cube> map;
    if (nlhs > 0 && nx != 0 && n_gen != 0)
        plhs[0] = qd_mex_init_output(&map, ny, nx, nz, n_gen);

    // Wrap optional pointers
    const arma::vec *p_y = y.empty() ? nullptr : &y;
    const arma::vec *p_z = z.empty() ? nullptr : &z;

    // Call library function
    CALL_QD(quadriga_lib::sos_map<double>(freq, phase, x, map, p_y, p_z, uniform, use_kernel));
}
