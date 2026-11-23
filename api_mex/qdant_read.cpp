// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# QDANT_READ
Read arrayant objects from a QDANT file

- Loads arrays from a file in the [QDANT format](formats.html#qdant). A file can hold several arrays, each with a
  numeric ID, e.g. one per frequency of a frequency-dependent model, where `center_freq` of each entry identifies the
  frequency

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
- **`fn`** — Path to the QDANT file; must not be empty
- **`ids`** — 1-based IDs of the entries to read, in this order, duplicates allowed; scalar or vector; default: the
  entries listed in the layout, each once, in order of first appearance (column-major scan), or all entries if the file
  has no layout; several entries must then form a multi-frequency array, otherwise an error asks to pass `ids`

## Outputs:
- **`ant`** — Arrayant struct ([[generate_antenna]]) for one entry, 1xN struct array for several
- **`layout`** — The file's layout matrix of entry IDs (0 = empty slot); uint32

## Notes:
- **IDs**: the number used in the layout matrix and by [[qdant_write]]. A first entry without an `id` attribute (files
  from other tools) counts as ID 1
- **Errors**: an ID that is 0 or not in the file, and a missing, unreadable or malformed file, raise an error
- **Elements without data**: elements without pattern nodes read as zeros. A file whose element count is not backed by
  its data, or that exceeds the [reader limits](formats.html#qdant), raises an error

## See also:
- [[qdant_write]] (writes such files)
- [[generate_antenna]], [[generate_speaker]] (create arrays without a file)
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
