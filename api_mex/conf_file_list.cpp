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

- Lists the regular files and symbolic links to regular files named `<scenario>.conf`
  (case-sensitive, non-empty scenario name) in each directory, not recursive: the directories in the
  given order, the files of one directory in byte-wise sorted order of their names
- A file name found in an earlier directory is skipped, so a local folder can override the files of
  a shared one
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry
  right after its file; other lines are not examined
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if a `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty
  string `''` is the current directory

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
