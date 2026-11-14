// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# QDANT_READ
Reads array antenna data from QDANT files

- The QuaDRiGa array antenna exchange format (QDANT) is an XML format for storing antenna pattern data
- Without `id`, all entries are read: returns a struct array when the file has multiple entries (frequency-dependent model) 
  or a single struct when it has exactly one entry; multiple entries must form a frequency set (same number of elements,
  same azimuth and elevation grids, element positions and coupling size), otherwise
  an error asks to pass `id`
- Differs from Python, where `qdant_read(fn)` reads ID 1 only
- With `id`, a single entry is read; useful for picking one frequency from a multi-frequency file
- An `id` that is not in the file is an error; the first entry counts as ID 1 only if it has no `id` attribute
- Malformed files are an error, also when they declare more elements (`NoElements`) than the file has data for;
  elements without pattern data read as zeros (at most 2^24 zero values per entry)

## Usage:
```
% Multi-frequency read (struct array, all entries)
[ ant, layout ] = quadriga_lib.qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.qdant_read( fn, id );
```

## Inputs:
- **`fn`** — Path to the QDANT file; string; must not be empty
- **`id`** — 1-based ID of the antenna entry to read; pass `[]` or omit to read every entry in the file

## Outputs:
- **`ant`** — Arrayant struct (single entry) or struct array (multiple entries); field layout as documented in [[arrayant_generate]]
- **`layout`** — Matrix of element IDs describing how entries are arranged in the file; datatype: uint32

## See also:
- [[qdant_write]] (for writing QDANT data)
- [[arrayant_generate]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#qdant">QDANT</a>)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // No output argument behaves as one (the struct goes to "ans")
    const int n_out = (nlhs == 0) ? 1 : nlhs;

    // Read inputs
    std::string fn = qd_mex_get_string(prhs[0]);
    bool read_all = (nrhs < 2) || mxIsEmpty(prhs[1]);
    unsigned id = read_all ? 1 : qd_mex_get_scalar<unsigned>(prhs[1], "id", 1);

    std::vector<quadriga_lib::arrayant<double>> arrayant;
    arma::u32_mat layout;

    // No 'id' given -> read every entry
    if (read_all)
    {
        CALL_QD(arrayant = quadriga_lib::qdant_read_multi<double>(fn, &layout));

        if (arrayant.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "File does not contain any data.");
        else if (arrayant.size() > 1) // Multi-freq mode
        {
            // The entries must form a frequency set (same element count, grids, positions and coupling size)
            std::string err = quadriga_lib::arrayant_is_valid_multi(arrayant);
            if (!err.empty())
            {
                err = "Entries in file '" + fn + "' do not form a frequency set (" + err +
                      "); pass 'id' to read single entries.";
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", err.c_str());
            }

            plhs[0] = qd_mex_arrayant2struct_multi(arrayant);
            if (n_out == 2)
                plhs[1] = qd_mex_copy2matlab(&layout);
            return;
        }
    }
    else // Single-freq mode
    {
        arrayant.resize(1);
        CALL_QD(arrayant[0] = quadriga_lib::qdant_read<double>(fn, id, &layout));
    }

    plhs[0] = qd_mex_arrayant2struct(arrayant[0]);
    if (n_out == 2)
        plhs[1] = qd_mex_copy2matlab(&layout);
}
