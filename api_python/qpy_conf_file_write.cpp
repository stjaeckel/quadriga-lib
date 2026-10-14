// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  shipped with QuaDRiGa, so that [[conf_file_read]] returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`,
  then the reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Index in `ref_sec` and write order:
  - 0: Large scale distributions, with an overview formula per LSP; 1: Large-Scale fading
    decorrelation distances; 2: Inter-parameter correlations, with the correlation matrix as
    comment; 3: Model parameters; 8: EoD offset; 4: Decorrelation distance for the small-scale
    fading spatial consistency; 5: Parameters for the ground reflection extension; 6: Absolute
    time of arrival offset; 9: Optional modelling components; 7: Path-loss model
  - The indices 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and elevation
  terms; with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the ESD line also has the
  terms of `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`, the lower limit of the mean
- Each line is `name = value % comment`, aligned in columns; the comment states the unit
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`,
  `tripple_slope`), the line `PL_model`, then the entries of `plpar` in dict order with the prefix
  `PL_`; several values of one entry are written on one line
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to
  the same double in [[conf_file_read]]; `Inf` and `-Inf` are written as such
- With `write_defaults = False` a key is written only if its value differs from the default;
  `SubpathMethod` only if it is not `legacy`; a section without a written key is left out
- With `write_defaults = True` all 165 numeric keys and `SubpathMethod` are written
- The inputs are checked with the rules of [[scenpar_validate]] before the file is opened: unknown
  keys (including `GR_epsilon`), `NaN`, values outside their range and a wrong `SubpathMethod`
  are errors. Further errors: a line break in the file name, a `model` that is not one name, an empty or
  `NaN` path-loss value, short names that are not one name or not distinct, line breaks in `ref` or
  `ref_sec`, more than 10 entries in `ref_sec`, a `valid_range` that is not empty and not `(4, 2)`.
- Raises `ValueError` for invalid input and `RuntimeError` if the file cannot be opened or written
- The text is written to a temporary file in the same directory, which must be writable, and then renamed
  to `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is replaced
  and keeps its permissions
- Every file written passes [[conf_file_read]] with `validate = True` and without ignored keys

## Usage:
```
quadriga_lib.tools.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, write_defaults )
```

## Inputs:
- **`fn`** — Path of the conf file to write; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`scenpar`** — Scenario parameters; dict with any subset of the 166 keys returned by
  [[conf_file_read]]; numeric values are real scalars (int, float, bool or a numpy value with one
  element); `SubpathMethod` is a str; missing keys have their default; `None` or `{}` writes the
  default set
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str) and one real scalar, list or
  1D array per further entry, named without the prefix `PL_`, as returned by [[conf_file_read]];
  written in dict order; `None` or `{}`: no path-loss section; default: None
- **`short_names`** — Short names of the scenario; list of str or one str; each one name of the
  characters `A-Z`, `a-z`, `0-9`, `_`; distinct; default: None
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; str; default: None (no reference)
- **`ref_sec`** — Reference per section, indexed as listed above; list of up to 10 str; an empty or
  missing entry writes no reference; default: None
- **`valid_range`** — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `(4, 2)`; default: None
- **`write_defaults`** — `True`: write all keys; `False`: write only keys that differ from their
  default; bool or `numpy.bool_`; default: False; `None` selects the default

## See also:
- [[conf_file_read]] (read the file back)
- [[scenpar_validate]] (value checks and defaults)
MD!*/

void conf_file_write(py::object fn,
                     py::handle scenpar,
                     py::handle plpar,
                     py::handle short_names,
                     py::handle ref,
                     py::handle ref_sec,
                     py::handle valid_range,
                     py::handle write_defaults)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    // Scenario parameters: dict of real scalars and the str entry 'SubpathMethod'
    std::string subpath_method;
    const auto scenpar_m = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");

    // Path-loss parameters: 'model' and the numeric entries; the dict order is the write order
    std::string pl_model;
    std::vector<std::string> pl_order;
    const auto plpar_m = qd_python_dict2plpar(plpar, pl_model, pl_order, "plpar");

    const auto short_names_v = qd_python_list2vector_Strings(short_names);

    if (!ref.is_none() && !py::isinstance<py::str>(ref))
        throw std::invalid_argument("Input 'ref' must be a str or None.");
    const std::string ref_s = ref.is_none() ? std::string() : py::cast<std::string>(ref);

    // A single str would silently become the reference of section 0
    if (py::isinstance<py::str>(ref_sec))
        throw std::invalid_argument("Input 'ref_sec' must be a list of str or None.");
    const auto ref_sec_v = qd_python_list2vector_Strings(ref_sec);

    const auto valid_range_a = qd_python_numpy2arma_Mat<double>(valid_range, true);
    const bool write_defaults_b = qd_python_flag(write_defaults, false, "write_defaults");

    // Call library function
    quadriga_lib::conf_file_write(fn_s, scenpar_m, subpath_method, pl_model, &plpar_m, &pl_order, &short_names_v,
                                  ref_s, &ref_sec_v, &valid_range_a, write_defaults_b);
}

// pybind11 declaration:
// m.def("conf_file_write", &conf_file_write,
//       py::arg("fn"),
//       py::arg("scenpar"),
//       py::arg("plpar") = py::none(),
//       py::arg("short_names") = py::none(),
//       py::arg("ref") = py::none(),
//       py::arg("ref_sec") = py::none(),
//       py::arg("valid_range") = py::none(),
//       py::arg("write_defaults") = false);
