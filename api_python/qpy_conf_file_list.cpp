// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the regular files and symbolic links to regular files whose name ends in `.conf` (case-sensitive)
  and is longer than `.conf`, not recursive. The scenario name is the file name without this extension,
  whatever characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order
  of their names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first
  directory wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = True` each file is read and one entry is added right after the file for
  every line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte order
  mark at the start of the file is skipped.
- Raises `ValueError` if a short name equals an earlier entry, if a file name equals an earlier short
  name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Raises `RuntimeError` if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped
- An empty str `""` is the current directory

## Usage:
```
scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list( dirs, parse_shortnames )
```

## Inputs:
- **`dirs`** — Directories, searched in this order; one path or a list or tuple of paths, each a str,
  bytes or path-like object (e.g. `pathlib.Path`); an empty list or `None` lists nothing
- **`parse_shortnames`** — `True`: read the files and add their short names; `False`: list the files
  only, which is faster; bool or `numpy.bool_`; default: True; `None` selects the default

## Outputs:
- **`scenarios`** — Scenario names and short names; list of str; length `n_scen`
- **`file_names`** — File name including `.conf` for each entry; list of str; length `n_scen`
- **`file_dirs`** — Directory of the file for each entry, exactly as given in `dirs` (a path-like object
  converted with `os.fspath`); list of str; length `n_scen`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

py::tuple conf_file_list(py::handle dirs,
                         py::handle parse_shortnames)
{
    // Read inputs; one path (str, bytes, os.PathLike) is one directory, also when it is empty (= current directory)
    std::vector<std::string> dirs_v;
    std::string path;
    if (qd_python_try_path(dirs, path))
        dirs_v.push_back(path);
    else if (py::isinstance<py::list>(dirs) || py::isinstance<py::tuple>(dirs))
        for (py::handle entry : py::reinterpret_borrow<py::sequence>(dirs))
        {
            if (!qd_python_try_path(entry, path))
                throw std::invalid_argument("Expected a string, a list of strings, or None.");
            dirs_v.push_back(path);
        }
    else if (!dirs.is_none())
        throw std::invalid_argument("Expected a string, a list of strings, or None.");
    const bool parse_shortnames_b = qd_python_flag(parse_shortnames, true, "parse_shortnames");

    // Output allocation
    std::vector<std::string> scenarios, file_names, file_dirs;

    // Call library function
    quadriga_lib::conf_file_list(dirs_v, scenarios, &file_names, &file_dirs, parse_shortnames_b);

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
