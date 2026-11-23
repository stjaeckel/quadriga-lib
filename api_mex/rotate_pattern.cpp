// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# ROTATE_PATTERN
Rotate the patterns of an antenna array using Euler rotations

- Turns the patterns, the polarization and the element positions of an array, e.g. for a mechanical downtilt. The
  rotated pattern equals, at the grid points, what [[interpolate_pattern]] returns for the unrotated pattern with the
  same `orientation`
- Port weights, `center_freq` and `name` are kept

## Usage:
```
arrayant_out = quadriga_lib.rotate_pattern(arrayant_in, orientation, usage, i_element);
```

## Inputs:
- **`arrayant_in`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array for a multi-frequency array
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; a positive tilt turns the main beam down, as in
  3GPP TR 38.901; 3 elements
- **`usage`** — Rotation mode; default: 0. Modes 0 and 1 adjust the sampling grid of non-uniformly sampled patterns,
  which needs all elements: for such a pattern, a subset in `i_element` raises an error. A struct array never adjusts
  the grid, so that all entries keep the same grid: there 0 equals 3, and 1 equals 4. Element positions turn in all
  modes except 2. For scalar acoustic fields (pressure in `e_theta_re` only), use 1 to avoid spurious polarization
  effects:<br><br>
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** — Element indices to rotate, 1-based, no duplicates; `[n]`; default: all elements

## Output:
- **`arrayant_out`** — Rotated arrayant, in the form of `arrayant_in`

## See also:
- [[interpolate_pattern]] (evaluates a pattern for an orientation without changing it)
- [[calc_rotation_matrix]] (rotation matrix of the orientation)
- [[generate_antenna]] (electrical tilt through the port weights)
- [[join_arrays]] (joins arrays that are then rotated together)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
    if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant_in' must be an arrayant struct or struct array.");

    const arma::vec orientation = qd_mex_get_Col<double>(prhs[1], false, "orientation"); // 3 elements, checked by C++
    const unsigned usage = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "usage", 0);

    // Element indices: 1-based MATLAB -> 0-based C++ (empty = all elements)
    const arma::uvec i_element = (nrhs < 4) ? arma::uvec() : qd_mex_get_index_Col(prhs[3], "i_element");

    if (mxGetNumberOfElements(prhs[0]) > 1) // Multi-frequency
    {
        // Usage 3 and 4 (no grid adjustment) are what the multi-frequency function always does for 0 and 1
        const unsigned usage_multi = (usage == 3) ? 0 : ((usage == 4) ? 1 : usage);
        auto ant_multi = qd_mex_struct2arrayant_multi(prhs[0], false, true, "arrayant_in"); // Validated by the C++ function
        CALL_QD(quadriga_lib::rotate_pattern(ant_multi, orientation, usage_multi, i_element));
        plhs[0] = qd_mex_arrayant2struct_multi(ant_multi);
        return;
    }

    auto ant = qd_mex_struct2arrayant(prhs[0], false, true, "arrayant_in"); // Validated by the C++ function
    CALL_QD(quadriga_lib::rotate_pattern(ant, orientation, usage, i_element));
    plhs[0] = qd_mex_arrayant2struct(ant);
}
