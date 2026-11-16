# General usage notes

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-general-usage-notes

- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- The conventions and shared quantities of the C++ [Glossary](../cpp/general-usage-notes.md#glossary) apply; the points below are specific to Python.
- `import quadriga_lib`; all functions live in its submodules `arrayant`, `channel`, `tools`, `RTtools`, `sos` and `gbsm`, except `version` and `components`, which are at the top level. A name drops the prefix that would repeat its submodule (`arrayant_`, `channel_` and `get_channels_`, `sos_`, `gbsm_`): C++ `get_channels_spherical` is `quadriga_lib.channel.spherical`. The Usage line shows the full call, e.g. `quadriga_lib.tools.acdf(...)` or `quadriga_lib.version()`.
- Array shapes follow in backticks as numpy shapes, e.g. `(n_rx, n_tx, n_path)`; `(n,)` is a 1D array.
- Array inputs take numpy arrays in any memory order; lists, scalars and other numeric dtypes are converted (copied). Outputs are numpy arrays: float64 for real values, complex128 for complex values; integer outputs have the dtype stated at the output.
- Complex data: many functions take or return complex arrays where C++ uses separate `_re` / `_im` arrays or interleaved rows; some select the form with a `complex` flag. Each function states its form.
- Data structures (array antennas, channels, scenario parameters, ray data) are dicts; the function that defines one lists its keys. Each call builds a new arrayant from an arrayant dict, so its data is checked in full on every call.
- Optional inputs state their default and can be omitted; an optional array can also be passed as `None`. Several outputs are returned as a tuple in the order of the Usage line.
- Errors: invalid inputs (shape/domain) raise `ValueError`, I/O failures `RuntimeError`, an index out of range `IndexError`; a call that does not match the signature (missing required argument, unknown keyword, `None` for a required array, a value of the wrong type) raises `TypeError`. Messages name the argument.
