// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"
#include "qd_helpers_gbsm.hpp"

// Folder of the shipped conf files, found next to the package folder that 'which' reports, so that a moved
// installation finds it: +quadriga_lib/../config (installed), +quadriga_lib/../data/config (source tree)
static std::string qd_mex_conf_dir()
{
    mxArray *name = mxCreateString("quadriga_lib.conf_file_list"), *path = nullptr;
    std::string file;
    if (mexCallMATLAB(1, &path, 1, &name, "which") == 0 && path != nullptr)
    {
        if (char *s = mxArrayToString(path))
            file = s, mxFree(s);
        mxDestroyArray(path);
    }
    mxDestroyArray(name);
    return qd_conf_shipped_dir(file, {"../config", "../data/config"});
}

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
- With `include_shipped = true` the folder of the conf files that come with the library is searched
  last, by the same rules, so a file in `dirs` overrides the shipped file of the same name
- With `parse_shortnames = true` every line `ShortName = name` that is not a comment adds an entry
  right after its file; other lines are not examined
- An error is raised if a short name equals an earlier entry, if a file name equals an earlier short
  name, if a `ShortName` value is not one name, or if a listed file cannot be opened
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty
  string `''` is the current directory

## Usage:
```
[ scenarios, file_names, file_dirs ] = quadriga_lib.conf_file_list( dirs, parse_shortnames, include_shipped );
```

## Inputs:
- **`dirs`** — Directories, searched in this order; one string or a cell array of strings; can be
  an empty cell array
- **`parse_shortnames`** — `true`: read the files and add their short names; `false`: list the files
  only, which is faster; default: true
- **`include_shipped`** — `true`: search the folder of the shipped conf files after `dirs`; `false`:
  only `dirs`; default: true

## Outputs:
- **`scenarios`** — Scenario names and short names; cell array of strings; `[n_scen, 1]`
- **`file_names`** — File name including `.conf` for each entry; cell array of strings;
  `[n_scen, 1]`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`, or the absolute
  path of the shipped folder; cell array of strings; `[n_scen, 1]`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; a char array is one directory, also when it is empty (= current directory)
    std::vector<std::string> dirs;
    if (mxIsChar(prhs[0]))
        dirs.push_back(qd_mex_get_string(prhs[0], "", "dirs"));
    else if (mxIsCell(prhs[0]))
        dirs = qd_mex_get_strings(prhs[0], "dirs");
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'dirs' must be a string or a cell array of strings.");
    const bool parse_shortnames = (nrhs < 2) ? true : qd_mex_get_scalar<bool>(prhs[1], "parse_shortnames", true);
    const bool include_shipped = (nrhs < 3) ? true : qd_mex_get_scalar<bool>(prhs[2], "include_shipped", true);

    std::vector<std::string> scenarios, file_names, file_dirs;

    std::vector<std::string> *p_file_names = (nlhs > 1) ? &file_names : nullptr;
    std::vector<std::string> *p_file_dirs = (nlhs > 2) ? &file_dirs : nullptr;

    const std::string shipped_dir = include_shipped ? qd_mex_conf_dir() : std::string();
    if (!shipped_dir.empty())
        dirs.push_back(shipped_dir);

    CALL_QD(scenarios = quadriga_lib::conf_file_list(dirs, p_file_names, p_file_dirs, parse_shortnames, false));

    plhs[0] = qd_mex_copy2matlab(&scenarios);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&file_names);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&file_dirs);
}
