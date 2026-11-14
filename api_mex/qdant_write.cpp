// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# QDANT_WRITE
Writes array antenna data to QDANT files

- The QuaDRiGa array antenna exchange format (QDANT) is an XML format for storing antenna pattern data
- Multiple array antennas can be stored in the same file using distinct `id` values
- If writing to an existing file without specifying an `id`, the data is appended at the end and
  the returned `id_in_file` identifies its location in the file
- An optional `layout` can be provided to organize the data inside the file
- Without `layout`, the layout in the file is replaced by a row of all IDs in the file in file order;
  a custom layout written before is lost (pass it again with every write that should keep it)
- The file has single-precision-equivalent accuracy relative to the element peak: pattern
  magnitudes (dB) and phases (degree) are rounded so that every sample is within 6e-8 of the peak
  amplitude of its element; samples far below the peak are less accurate relative to their own
  amplitude (about 6e-4 at -80 dB, 6e-2 at -120 dB below the peak); positions and coupling are
  written as float values (at most 9 significant digits); sampling grids are rounded to 0.001
  degree; all non-zero pattern samples are written at any level, exact zeros as `-inf` dB
- Passing a struct array (multiple elements) writes a frequency-dependent model with sequential
  1-based IDs; in this mode appending to an exisiting file is not allowed and will cause an error

## Usage:
```
id_in_file = quadriga_lib.qdant_write( fn, arrayant, id, layout );
```

## Inputs:
- **`fn`** — Output QDANT filename; string; must not be empty
- **`arrayant`** — Struct containing the arrayant data; field layout as documented in [[arrayant_generate]];
  a struct array writes a frequency-dependent model and requires `id` and `layout` to be omitted
- **`id`** — Target ID of the antenna inside the file; default: max-ID in existing file + 1 (or 1 if the file does not exist);
  must be omitted for a multi-frequency model
- **`layout`** — Matrix organizing multiple antenna IDs within the file; must only reference IDs present in the file; uint32;
  default: a row of all IDs in the file (replaces a layout written before)

## Outputs:
- **`id_in_file`** — ID assigned to the antenna in the file after writing; set to 0 in multi-frequency (struct array) mode

## See also:
- [[qdant_read]] (for reading QDANT data)
- [[arrayant_generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#qdant">QDANT</a>)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0]);
    unsigned id = (nrhs < 3) ? 0 : qd_mex_get_scalar<unsigned>(prhs[2], "id", 0);
    arma::u32_mat layout = (nrhs < 4) ? arma::u32_mat() : qd_mex_typecast_Mat<unsigned>(prhs[3], "layout");

    if (fn.empty())
         mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File name cannot be empty.");
    if (!mxIsStruct(prhs[1]) || mxGetNumberOfElements(prhs[1]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'arrayant' must be an arrayant struct or struct array.");

    // Assemble array antenna object
    auto ant = quadriga_lib::arrayant<double>();
    auto ant_multi = std::vector<quadriga_lib::arrayant<double>>();
    const bool is_multi = mxGetNumberOfElements(prhs[1]) > 1;
    if (is_multi)
    {
        ant_multi = qd_mex_struct2arrayant_multi(prhs[1], true);
        if (id != 0 || !layout.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Inputs 'id' and 'layout' are not allowed for multi-frequency arrays.");
    }
    else
        ant = qd_mex_struct2arrayant(prhs[1], true);

    // Check if files exists in multi-freq mode and throw error if yes
    if (is_multi && std::filesystem::exists(fn))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File exists. Appending a multi-frequency array to an existing file is not allowed.");

    // Write to file
    if (is_multi)
        CALL_QD(qdant_write_multi(fn, ant_multi));
    else
        CALL_QD(id = quadriga_lib::qdant_write(fn, ant, id, &layout));

    plhs[0] = qd_mex_copy2matlab(&id); // Also without output argument: the ID goes to ans
}
