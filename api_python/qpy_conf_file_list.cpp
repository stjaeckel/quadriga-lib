// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the files whose name ends in `.conf` (case-sensitive) in each directory, not recursive. The
  scenario name is the file name without this extension.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order
  of their names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first
  directory wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = True` each file is read and one entry is added right after the file for
  every line `ShortName = name` that is not a comment. Other lines are not examined.
- Raises `ValueError` if a short name equals an earlier entry, if a file name equals an earlier short
  name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Raises `RuntimeError` if a listed file cannot be opened for reading the short names
- A directory that does not exist or is not a directory is skipped
- An empty str `""` is the current directory

## Usage:
```
scenarios, file_names, file_dirs = quadriga_lib.tools.conf_file_list( dirs, parse_shortnames )
```

## Inputs:
- **`dirs`** — Directories, searched in this order; one str or a list of str; an empty list lists
  nothing
- **`parse_shortnames`** — `True`: read the files and add their short names; `False`: list the files
  only, which is faster; default: True

## Outputs:
- **`scenarios`** — Scenario names and short names; list of str; length `n_scen`
- **`file_names`** — File name including `.conf` for each entry; list of str; length `n_scen`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs`; list of str;
  length `n_scen`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

py::tuple conf_file_list(py::handle dirs,
                         bool parse_shortnames)
{
    // Read inputs; a str is one directory, also when it is empty (= current directory)
    const auto dirs_v = qd_python_list2vector_Strings(dirs);

    // Output allocation
    std::vector<std::string> scenarios, file_names, file_dirs;

    // Call library function
    quadriga_lib::conf_file_list(dirs_v, scenarios, &file_names, &file_dirs, parse_shortnames);

    // Copy to python
    auto scenarios_py = qd_python_copy2list(scenarios);
    auto file_names_py = qd_python_copy2list(file_names);
    auto file_dirs_py = qd_python_copy2list(file_dirs);

    return py::make_tuple(scenarios_py, file_names_py, file_dirs_py);
}

// pybind11 declaration:
// m.def("conf_file_list", &conf_file_list,
//       py::arg("dirs"),
//       py::arg("parse_shortnames") = true);
