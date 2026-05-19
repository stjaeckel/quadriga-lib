// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# GENERATE_DIFFRACTION_PATHS
Generate elliptic propagation paths and weights for diffraction gain estimation

- Generates inputs required by quadriga_lib.[[calc_diffraction_gain]]: elliptic-arc paths sampling
  the Fresnel ellipsoid volume between each TX-RX pair, plus per-segment weights
- Each ellipsoid has `n_path` paths, each with `n_seg` segments; `orig` and `dest` lie on the
  semi-major axis
- Weights are derived from the knife-edge diffraction model; initial weights normalized so
  sum(prod(weights,3),2) = 1

## Usage:
```
[ rays, weights ] = quadriga_lib.generate_diffraction_paths( orig, dest, center_frequency, lod );
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`center_frequency`** — Center frequency in Hz
- **`lod`** — Level of detail; controls `n_path` and `n_seg`:<br><br>
   | `lod` | `n_path` | `n_seg` | Note  |
   | :---: | -------: | ------: | ----: |
   | 1     | 7        | 3       | -     |
   | 2     | 19       | 3       | -     |
   | 3     | 37       | 4       | -     |
   | 4     | 61       | 5       | -     |
   | 5     | 1        | 2       | debug |
   | 6     | 2        | 2       | debug |

## Outputs:
- **`rays`** — Coordinates of path waypoints (x, y, z stacked along the 4th dimension, endpoints
  excluded); `[n_pos, n_path, n_seg-1, 3]`
- **`weights`** — Per-segment weights; `[n_pos, n_path, n_seg]`

## See also:
- [[calc_diffraction_gain]] (consumes the output of this function)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs != 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat orig = qd_mex_get_Mat<double>(prhs[0]);
    const arma::mat dest = qd_mex_get_Mat<double>(prhs[1]);
    const double center_frequency = qd_mex_get_scalar<double>(prhs[2], "center_frequency", 0.0);
    const int lod = qd_mex_get_scalar<int>(prhs[3], "lod", 1);

    // Determine output dimensions from lod
    arma::uword n_path = 0, n_seg = 0;
    switch (lod)
    {
    case 1:
        n_path = 7;
        n_seg = 3;
        break;
    case 2:
        n_path = 19;
        n_seg = 3;
        break;
    case 3:
        n_path = 37;
        n_seg = 4;
        break;
    case 4:
        n_path = 61;
        n_seg = 5;
        break;
    case 5:
        n_path = 1;
        n_seg = 2;
        break;
    case 6:
        n_path = 2;
        n_seg = 2;
        break;
    default:
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'lod' must be in range 1..6.");
    }

    const arma::uword n_pos = orig.n_rows;
    const arma::uword n_wp = n_seg - 1; // waypoint count (endpoints excluded)

    // Allocate the rays output as a 4D MATLAB array [n_pos, n_path, n_seg-1, 3]
    // and wrap each (x, y, z) slab as a strict Armadillo cube so the C++ function
    // writes directly into MATLAB-owned memory without an extra copy.
    arma::cube ray_x, ray_y, ray_z;
    if (nlhs > 0)
    {
        mwSize dims[4] = {(mwSize)n_pos, (mwSize)n_path, (mwSize)n_wp, 3};
        plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
        double *data = (double *)mxGetData(plhs[0]);
        const arma::uword offset = n_pos * n_path * n_wp;
        ray_x = arma::cube(data, n_pos, n_path, n_wp, false, true);
        ray_y = arma::cube(&data[offset], n_pos, n_path, n_wp, false, true);
        ray_z = arma::cube(&data[2 * offset], n_pos, n_path, n_wp, false, true);
    }

    // Allocate weights output [n_pos, n_path, n_seg]
    arma::cube weight;
    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&weight, n_pos, n_path, n_seg);

    // Call library function
    CALL_QD(quadriga_lib::generate_diffraction_paths<double>(&orig, &dest, center_frequency, lod,
                                                             &ray_x, &ray_y, &ray_z, &weight));
}