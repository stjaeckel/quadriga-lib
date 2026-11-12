+++
title = "Python API"
description = "%%GAP: Python API reference for Quadriga-Lib: array antennas, channel generation, statistics, the 3GPP stochastic model and site-specific simulation tools.%%"
lead = "%%GAP: Reference for the Python package `quadriga_lib`, version {{ version }}.%%"
+++

- Each function has a 1-line short description, optional detailed notes, a Usage block, and Inputs/Outputs sections.
- `import quadriga_lib`; all functions live in its submodules `arrayant`, `channel`, `tools`, `RTtools`, `sos` and `gbsm`. The Usage line shows the full call, e.g. `quadriga_lib.tools.acdf(...)`.
- Array shapes follow in backticks as numpy shapes, e.g. `(n_rx, n_tx, n_path)`; `(n,)` is a 1D array.
- Array inputs take numpy arrays in any memory order; lists, scalars and other numeric dtypes are converted (copied). Outputs are numpy arrays: float64 for real values, complex128 for complex values; integer outputs have the dtype stated at the output.
- Data structures (array antennas, channels, scenario parameters, ray data) are dicts; the function that defines one lists its keys. Each call builds a new arrayant from an arrayant dict, so its data is checked in full on every call.
- Optional inputs state their default and can be omitted; an optional array can also be passed as `None`. Several outputs are returned as a tuple in the order of the Usage line.
- Index conventions: 0-based unless the field is explicitly called "1-based".
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Errors: invalid inputs (shape/domain) raise `ValueError`, I/O failures `RuntimeError`, an index out of range `IndexError`; a call that does not match the signature (missing required argument, unknown keyword, `None` for a required array, a value of the wrong type) raises `TypeError`. Messages name the argument, in some functions by its C++ name (e.g. `center_frequency` for `center_freq`).
