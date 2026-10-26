// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# CONF_FILE_LIST
List the scenarios defined by the conf files in a set of directories

- Lists the regular files and symbolic links to regular files whose name ends in `.conf`
  (case-sensitive) and is longer than `.conf`, not recursive. The scenario name is the file name
  without this extension, whatever characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order
  of their names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first
  directory wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = true` each file is read and one entry is added right after the file for
  every line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte
  order mark at the start of the file is skipped.
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`,
  `_`, or if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped
- An empty string `''` is the current directory

## Usage:
```
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( dirs, parse_shortnames );
```

## Inputs:
- **`dirs`** — Directories, searched in this order; one string or a cell array of strings; an empty
  cell array lists nothing
- **`parse_shortnames`** — `true`: read the files and add their short names; `false`: list the files
  only, which is faster; default: true

## Outputs:
- **`scenarios`** — Scenario names and short names; cell array of strings; `[n_scen, 1]`
- **`file_names`** — File name including `.conf` for each entry; cell array of strings;
  `[n_scen, 1]`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`; cell array of
  strings; `[n_scen, 1]`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; a char array is one directory, also when it is empty (= current directory)
    std::vector<std::string> dirs;
    if (mxIsChar(prhs[0]))
        dirs.push_back(qd_mex_get_string(prhs[0]));
    else if (mxIsCell(prhs[0]))
        dirs = qd_mex_get_strings(prhs[0], "dirs");
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'dirs' must be a string or a cell array of strings.");
    const bool parse_shortnames = (nrhs < 2) ? true : qd_mex_get_scalar<bool>(prhs[1], "parse_shortnames", true);

    // Output allocation
    std::vector<std::string> scenarios, file_names, file_dirs;

    // Wrap optional pointers
    std::vector<std::string> *p_file_names = (nlhs > 1) ? &file_names : nullptr;
    std::vector<std::string> *p_file_dirs = (nlhs > 2) ? &file_dirs : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::conf_file_list(dirs, scenarios, p_file_names, p_file_dirs, parse_shortnames));

    // Copy to MATLAB
    plhs[0] = qd_mex_copy2matlab(&scenarios);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&file_names);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&file_dirs);
}
