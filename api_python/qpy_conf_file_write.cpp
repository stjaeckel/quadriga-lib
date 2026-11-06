// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files of QuaDRiGa; every
  written file passes [[conf_file_read]] with `validate = True`, without ignored keys and with the same values bit for
  bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and the short names. Up to
  10 sections follow, each with an optional line `% See: ref; ref_sec` and, for the LSPs and the path-loss model, the
  formulas as comments; the `plpar` keys are written in dict order. Sections by entry of `ref_sec`: 0 large-scale
  distributions, 1 large-scale decorrelation distances, 2 inter-parameter correlations, 3 model parameters, 4
  small-scale decorrelation distance, 5 ground reflection, 6 absolute TOA, 7 path-loss model, 8 EoD offset, 9 optional
  modelling components; 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- Numbers have up to 15 significant digits, or 16 or 17 where 15 do not read back to the same double, independent of the
  locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = False`: only keys whose value differs from the default (bit for bit); a section without a written
  key is left out
- All inputs are validated before the file is opened: the checks of [[scenpar_validate]]; `model`, the other keys of
  `plpar` and the short names are names, the short names distinct; every `plpar` key has at least one value and no
  `NaN`; `fn`, `ref` and `ref_sec` have no line break
- The text is written to a temporary file in the same directory, which must be writable, and renamed to `fn`: a failed
  call creates no file and leaves an existing file unchanged; an existing file keeps its permissions, a symbolic link
  `fn` is written through

## Usage:
```
quadriga_lib.gbsm.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, write_defaults )
```

## Inputs:
- **`fn`** — Path of the conf file to write; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`scenpar`** — Scenario parameters; dict with any subset of the 171 keys returned by
  [[conf_file_read]]; real scalars (int, float, bool or a numpy value with one element); missing keys
  have their default; `None` or `{}` writes the default set
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
  height, UT height in m; columns: lower and upper end, one value is written if both are equal; a
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

    // Scenario parameters: dict of real scalars
    const auto scenpar_m = qd_python_dict2scenpar(scenpar, "scenpar");

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
    quadriga_lib::conf_file_write(fn_s, scenpar_m, pl_model, &plpar_m, &pl_order, &short_names_v,
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
