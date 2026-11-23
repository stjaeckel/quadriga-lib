// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# QDANT_READ
Reads antenna array data from QDANT files

- The QuaDRiGa Array Antenna Exchange Format (QDANT) is an XML format for storing antenna pattern data
- Without `ids`, the entries listed in the layout are read (all entries if the file has no layout): returns a struct array when the file has multiple entries (frequency-dependent model) 
  or a single struct when it has exactly one entry; multiple entries must form a frequency set (same number of elements,
  same azimuth and elevation grids, element positions and port-weights size), otherwise
  an error asks to pass `ids`
- With a scalar `ids`, a single entry is read as a struct; useful for picking one frequency from a multi-frequency file
- With a vector `ids`, the entries are read in the given order (duplicates allowed) and returned as a struct array
- An ID that is 0 or not in the file is an error; the first entry counts as ID 1 only if it has no `id` attribute
- Malformed files are an error, also when they declare more elements (`NoElements`) than the file has data for;
  elements without pattern data read as zeros (at most 2^24 zero values per entry)

## Usage:
```
% Multi-frequency read (struct array, entries in the layout)
[ ant, layout ] = quadriga_lib.qdant_read( fn );

% Single-frequency read (struct output)
[ ant, layout ] = quadriga_lib.qdant_read( fn, ids );

% Several entries (struct array)
[ ant, layout ] = quadriga_lib.qdant_read( fn, [3,1,3] );
```

## Inputs:
- **`fn`** — Path to the QDANT file; string; must not be empty
- **`ids`** — 1-based ID of the antenna entry to read (scalar) or vector of IDs; pass `[]` or omit to read the entries in the layout

## Outputs:
- **`ant`** — Arrayant struct (single entry) or struct array (multiple entries); field layout as documented in [[generate_antenna]]
- **`layout`** — Matrix of element IDs describing how entries are arranged in the file; datatype: uint32

## See also:
- [[qdant_write]] (for writing QDANT data)
- [[generate_antenna]] (for the arrayant struct layout)
- QuaDRiGa Array Antenna Exchange Format (<a href="formats.html#qdant">QDANT</a>)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    std::string fn = qd_mex_get_string(prhs[0], "", "fn");
    const bool read_all = (nrhs < 2) || mxIsEmpty(prhs[1]);
    const arma::uvec ids = read_all ? arma::uvec() : qd_mex_typecast_Col<arma::uword>(prhs[1], "ids"); // 1-based file IDs; no shift intended

    std::vector<quadriga_lib::arrayant<double>> arrayant;
    arma::u32_mat layout;
    CALL_QD(arrayant = quadriga_lib::qdant_read<double>(fn, ids, &layout));

    if (arrayant.empty())
    {
        const std::string msg = "File '" + fn + "' does not contain any antenna data.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }

    // All entries of a file with multiple entries must form a frequency set
    if (read_all && arrayant.size() > 1)
    {
        std::string err = quadriga_lib::is_valid(arrayant); // no CALL_QD: returns the error
        if (!err.empty())
        {
            err = "Entries in file '" + fn + "' do not form a frequency set (" + err +
                  "); pass 'ids' to read single entries.";
            qd_mex_cpp_error(err);
        }
    }

    // A scalar 'ids' or a single entry gives a struct, a vector of IDs or multiple entries a struct array
    plhs[0] = (arrayant.size() == 1) ? qd_mex_arrayant2struct(arrayant[0]) : qd_mex_arrayant2struct_multi(arrayant);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&layout);
}
