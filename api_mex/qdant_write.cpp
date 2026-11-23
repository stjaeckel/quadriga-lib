// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# QDANT_WRITE
Write arrayant data to a QDANT (XML) file

- Stores an array in a file in the [QDANT format](formats.html#qdant). A file can hold several arrays with distinct IDs
  and a layout matrix that arranges them
- A struct array stores a frequency-dependent model in a new file, each entry holding the pattern at its `center_freq`

## Usage:
```
id_in_file = quadriga_lib.qdant_write( fn, arrayant, id, layout );
```

## Inputs:
- **`fn`** — Output QDANT file name, must not be empty; an existing file is updated; struct array: the file must not
  exist (raises an error and leaves the file unchanged)
- **`arrayant`** — Arrayant struct ([[generate_antenna]]), or 1xN struct array, written with the sequential IDs
  `1, 2, ..., N` and an `[N, 1]` layout of these IDs
- **`id`** — Target ID in the file; an entry with this ID is replaced; 0 appends with `max_existing_id + 1`, or 1 in a
  new file; struct array: must be 0 or omitted; default: 0
- **`layout`** — Matrix of entry IDs that arranges the file; every value must be an ID in the file (0 is not accepted);
  struct array: must be empty; default: `[]`, which replaces the layout in the file by a `[1, n]` row of all IDs in file
  order, so a custom layout must be passed again with every write that should keep it

## Output:
- **`id_in_file`** — ID assigned to the written antenna within the file; 0 for a struct array; uint32

## Notes:
- **Pattern accuracy**: the file has single-precision-equivalent accuracy relative to the element peak. Pattern
  magnitudes (in dB) and phases (in degrees) get the fewest decimals (0 to 6) that keep the error of each complex sample
  at or below 6e-8 times the peak amplitude of its element (over `e_theta` and `e_phi`). Samples near the peak are
  accurate to about 6e-8 of their own amplitude; the relative error grows by 10 per 20 dB below the peak (about 6e-4 at
  -80 dB, 6e-2 at -120 dB), while the absolute error stays below 6e-8 of the peak. An element at a low level is
  quantized relative to its own peak
- **Other values**: element positions and port weights (magnitude, phase) are written as the shortest text that reads
  back as the same float (at most 9 significant digits); sampling grids are rounded to 0.001 degree
- **Zeros**: all non-zero pattern samples are written, at any level; exact zeros are written as `-inf` dB. An element
  that is zero in both field components is written as an all `-inf` `e_theta_mag` node

## See also:
- [[qdant_read]] (reads the file back)
- [[generate_speaker]] (typical source of frequency-dependent arrays)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    // File IDs in 'id' and 'layout' are 1-based in C++ too; no shift intended
    unsigned id = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "id", 0);
    arma::u32_mat layout = (nrhs < 4) ? arma::u32_mat() : qd_mex_typecast_Mat<unsigned>(prhs[3], "layout");

    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' must be an arrayant struct or struct array.");

    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    const bool is_multi = mxGetNumberOfElements(prhs[1]) > 1;
    if (is_multi)
    {
        ant_multi = qd_mex_struct2arrayant_multi(prhs[1], true, false, "arrayant");
        if (id != 0 || !layout.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'id' and 'layout' are not allowed for multi-frequency arrays.");
    }
    else
        ant = qd_mex_struct2arrayant(prhs[1], true, false, "arrayant");

    if (is_multi)
        CALL_QD(quadriga_lib::qdant_write(fn, ant_multi));
    else
        CALL_QD(id = quadriga_lib::qdant_write(fn, ant, id, layout));

    plhs[0] = qd_mex_copy2matlab(&id);
}
