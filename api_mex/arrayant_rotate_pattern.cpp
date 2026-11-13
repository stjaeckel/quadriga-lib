// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_ROTATE_PATTERN
Rotate antenna radiation patterns using Euler rotations

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [[arrayant_interpolate]] returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); coupling, `center_freq` and `name` are kept
- Usage 0 and 1 adjust the sampling grid for non-uniformly sampled antennas; this needs all elements, so a subset
  in `i_element` is an error when the grid must be adjusted
- For scalar acoustic fields (pressure stored in `e_theta_re` only), use `usage = 1` to avoid
  spurious polarization effects

## Usage:
```
arrayant_out = quadriga_lib.arrayant_rotate_pattern(arrayant_in, orientation, usage, i_element);
```

## Inputs:
- **`arrayant_in`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array represents a frequency-dependent model
- **`orientation`** — Euler angles (bank, tilt, heading) in rad; 3-element vector
- **`usage`** — Rotation mode; default: 0<br><br>
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

  Multi-frequency input never adjusts the grid (uniform-grid consistency across frequencies): 0 and 3
  rotate pattern and polarization, 1 and 4 the pattern only, 2 the polarization only (as in Python).
- **`i_element`** — Element indices to rotate, 1-based, no duplicates; default: all elements; `[n]`

## Outputs:
- **`arrayant_out`** — Arrayant struct, or struct array of size `numel(arrayant_in)` for multi-frequency input;
  field layout as documented in [[arrayant_generate]]
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in' must be an arrayant struct or struct array.");

    const arma::vec orientation = qd_mex_get_Col<double>(prhs[1], true, "orientation");
    if (orientation.n_elem != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'orientation' must have 3 elements.");
    const unsigned usage = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "usage", 0);

    // Element indices: 1-based MATLAB -> 0-based C++ (empty = all elements)
    arma::uvec i_element = (nrhs < 4) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[3], true, "i_element");
    if (arma::any(i_element == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
    i_element -= 1;

    if (mxGetNumberOfElements(prhs[0]) > 1) // Multi-frequency
    {
        // Usage 3 and 4 (no grid adjustment) are what the multi-frequency function always does for 0 and 1
        const unsigned usage_multi = (usage == 3) ? 0 : ((usage == 4) ? 1 : usage);
        auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], false, true); // Validated by the C++ function
        CALL_QD(quadriga_lib::arrayant_rotate_pattern_multi(ant_multi, orientation, usage_multi, &i_element));
        plhs[0] = qd_mex_arrayant2struct_multi(ant_multi);
        return;
    }

    auto ant = qd_mex_struct2arrayant(prhs[0], false, true); // Validated by the C++ function
    CALL_QD(quadriga_lib::arrayant_rotate_pattern(ant, orientation, usage, &i_element));
    plhs[0] = qd_mex_arrayant2struct(ant); // Also without output argument: the struct goes to ans
}
