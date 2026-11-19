// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# GENERATE_DIFFRACTION_PATHS
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by [[calc_diffraction_gain]]: elliptic-arc paths sampling
  the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the
  semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  sum(prod(weights,3),2) = 1

## Usage:
```
[ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_freq, lod, scalar_mode );
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_freq`** — Center frequency in Hz; in scalar mode the acoustic frequency
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:<br><br>
   | `lod` | `n_path` | `n_seg` | Note  |
   | :---: | -------: | ------: | ----: |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |
   | 7     | 1        | 10      | direct line in 10 segments |
- **`scalar_mode`** *(optional)* — `true` = acoustic: the Fresnel ellipsoids use the wavelength `343 / center_freq`;
  default: `false` (EM, `299792458 / center_freq`)

## Outputs:
- **`rays`** — Coordinates of path waypoints (x, y, z stacked along the 4th dimension, endpoints
  excluded); `[n_pos, n_path, n_seg-1, 3]`
- **`weights`** — Per-segment weights; `[n_pos, n_path, n_seg]`

## See also:
- [[calc_diffraction_gain]] (consumes the output of this function)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::mat orig = qd_mex_get_Mat<double>(prhs[0], false, "orig");
    const arma::mat dest = qd_mex_get_Mat<double>(prhs[1], false, "dest");
    const double center_freq = qd_mex_get_scalar<double>(prhs[2], "center_freq", 0.0);
    if (mxGetNumberOfElements(prhs[3]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'lod' cannot be empty.");
    const int lod = qd_mex_get_scalar<int>(prhs[3], "lod", 1);
    const bool scalar_mode = (nrhs < 5) ? false : qd_mex_get_scalar<bool>(prhs[4], "scalar_mode", false);

    // Output sizes depend on lod and are set by C++
    arma::cube ray_x, ray_y, ray_z, weight;
    CALL_QD(quadriga_lib::generate_diffraction_paths<double>(orig, dest, center_freq, lod,
                                                             ray_x, ray_y, ray_z, weight, scalar_mode));

    // rays [n_pos, n_path, n_seg-1, 3]: x, y, z stacked along the 4th dimension
    mwSize dims[4] = {(mwSize)ray_x.n_rows, (mwSize)ray_x.n_cols, (mwSize)ray_x.n_slices, 3};
    plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
    double *data = (double *)mxGetData(plhs[0]);
    const arma::uword offset = ray_x.n_elem;
    if (offset != 0)
    {
        std::memcpy(data, ray_x.memptr(), offset * sizeof(double));
        std::memcpy(&data[offset], ray_y.memptr(), offset * sizeof(double));
        std::memcpy(&data[2 * offset], ray_z.memptr(), offset * sizeof(double));
    }

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&weight);
}
