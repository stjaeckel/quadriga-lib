You are implementing a lightweight Python API wrapper for pybind11 (v3) to call a function of my C++ radio channel modelling library (quadriga-lib). You will receive the documentation block of the C++ function (or its name) and/or the MATLAB MEX wrapper that implements the same function for MATLAB. Your task is to add the wrapper (a regular C++ function exposed via pybind11) in `api_python/`, register it, and add a test in `tests/python_tests/`, following the workflow below. The wrapper converts arguments, allocates outputs and calls the C++ function; it should be user-friendly for Python users while keeping the performance of the underlying C++ library. The C++ side is covered by `md/SPEC_CPP_API.md`, the MEX wrappers by `md/SPEC_MATLAB_API.md`.

# CPP/MEX API DOCUMENTATION BLOCK FORMAT
- Each C++ function has a 1-line summary, a description as bullet list, a `## Declaration:` block, and `## Inputs:` / `## Outputs:` / `## Returns:` sections (singular when there is exactly one entry); see `md/SPEC_CPP_API.md` for the full format. MEX wrappers have a `## Usage:` block instead of the Declaration.
- C++ arguments with a default value or accepting `nullptr` are marked `*(optional)*`; the default is visible in the Declaration.
- Mandatory C++ arguments are passed by reference, optional ones by pointer (default `nullptr`). Older functions pass mandatory arguments by pointer as well.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated.
- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Pointer arguments: `nullptr` skips optional outputs (they are neither allocated nor computed); required inputs passed by pointer throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`
- Index conventions: 0-based unless the field is explicitly called "1-based"
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).

# REPOSITORY LAYOUT AND BUILD
- `api_python/qpy_<function_name>.cpp` — Python wrappers, one file per Python function.
- The Python module is built from 7 dispatcher files: `python_main.cpp` (top-level module, `version`, `components`) and one file per submodule: `python_arrayant.cpp`, `python_channel.cpp`, `python_tools.cpp`, `python_RTtools.cpp`, `python_sos.cpp`, `python_gbsm.cpp`. Each dispatcher `#include`s its `qpy_*.cpp` files and registers the functions with `m.def(...)`.
  - A new wrapper is NOT picked up automatically: it must be included and registered in the dispatcher of its submodule (see "pybind11 registration").
  - All wrappers of one submodule are compiled into a single translation unit. Names of wrapper functions and of `static` helper functions must therefore be unique within the submodule.
- `api_python/python_arma_adapter.hpp` (numpy / list / dict conversions) and `api_python/python_quadriga_adapter.hpp` (arrayant / channel dicts, `std::any` datasets) — helper functions. Read them when a signature is unclear; the lists in this document are a summary.
- `include/*.hpp` — C++ declarations; `src/*.cpp` — C++ implementations with their documentation blocks (`/*!MD ... MD!*/`); `api_mex/*.cpp` — MEX wrappers.
- `tests/python_tests/test_<function_name>.py` — tests, one file per wrapper; pytest collects all `test_*.py` files in the folder, so a new test file needs no registration.
- `lib/` — the compiled Python module (`quadriga_lib.cpython-*.so`) is installed here by the build; the tests import it from there.
- Generated overviews (read-only, produced by `make documentation`, never edit them; they may lag behind the sources until the documentation is rebuilt):
  - `html_docu/quadriga_lib_api_cpp.md` — all C++ functions
  - `html_docu/quadriga_lib_api_python.md` — all Python functions
  - `html_docu/quadriga_lib_api_mex.md` — all MATLAB functions
  - Each starts with general usage notes and a "Function Index" table with the columns Function, Section and Line. The Line column is the line number at which the entry starts: read the index first, then extract only the entries you need by line range instead of reading the whole file.
- The text in the `/*!MD ... MD!*/` block of a wrapper is merged into the Markdown overview and compiled to HTML. It also serves as the reference when reading the wrapper source. The delimiters `/*!MD`, `MD!*/`, `/*!SECTION`, `SECTION!*/` are matched literally, each on its own line.
- Commands (Linux, from the repository root):
  - `make python` — build the library and the Python module into `lib/`
  - `make python_test` — run the complete Python test suite (`python3 -m pytest tests/python_tests -x -s`)
  - `python3 -m pytest tests/python_tests/test_<function_name>.py -x -s` — run one test file; this is the normal way to iterate on one wrapper

# WORKFLOW
The order differs from the C++ workflow: the wrapper is written first, the test second. The physics is already verified by the Catch2 suite of the C++ function; the Python test exists to exercise the wrapper. The work normally happens in one session.

There are two review points at which you stop, report, and wait for my answer before continuing:
- **Review point A** (end of Phase 1) — only if there are open questions about the Python interface.
- **Review point B** (end of Phase 4) — always: the final result.

## Phase 0 — What you receive
- The documentation block of the C++ function, or its name (the block is then in `src/*.cpp` and in `html_docu/quadriga_lib_api_cpp.md`), and/or the MEX wrapper of the same function.
- Optionally the submodule and remarks on the Python interface (argument names, order, flags, what to leave out).

## Phase 1 — Read the contract and fix the Python interface
- Read the C++ documentation block and the declaration in `include/`, and the MEX wrapper if one is given.
- Read one or two existing wrappers that use the same pattern and follow them, e.g.:
  - `qpy_acdf.cpp` — array inputs, optional array input, outputs of known and unknown size
  - `qpy_obj_file_read.cpp` — outputs of runtime size, integer index outputs, string lists, map-to-dict
  - `qpy_obj_file_write.cpp` — many optional inputs passed to C++ as `nullptr` when absent
  - `qpy_calc_delay_spread.cpp` — ragged inputs as lists of arrays
  - `qpy_arrayant_rotate_pattern.cpp` — arrayant dict in / dict out, in-place modification
  - `qpy_get_channels_multifreq.cpp` — arrayant dicts, 3-element position vectors, 4D outputs, `complex` flag
  - `qpy_quantize_delays.cpp` — `complex` and `stack` flags for lists of arrays
  - `qpy_hdf5_write_channel.cpp`, `qpy_hdf5_read_channel.cpp` — channel dicts, unstructured data
- Settle before writing code: the Python name and the submodule; the usage line (order and kwarg names of the inputs, order of the returned tuple, in/out splits); which inputs are optional and their defaults; whether the `complex` and `stack` flags apply; the type of every output (array dtype, list, dict, scalar); the size of every output and whether the wrapper can know it before the call; the helper for every argument; the section name.
- **Review point A:** if anything in this list cannot be settled from the C++ documentation, the MEX wrapper and this spec (the submodule, a C++ type without helper, an output order that is not obvious, a flag beyond `complex` / `stack`, logic that would have to live in the wrapper), list the open questions with your proposed answer for each, and wait. If everything is clear, state the usage line and continue.

## Phase 2 — Wrapper and registration
- Write `api_python/qpy_<function_name>.cpp` following the "IMPLEMENTATION GUIDELINES".
- Register the wrapper in the dispatcher of its submodule AND add the identical registration as a comment block at the end of the wrapper file.
- Build. The wrapper must compile without warnings.

## Phase 3 — Test
- Write `tests/python_tests/test_<function_name>.py` (see "TESTS"). The test exercises the wrapper: every argument reaches the right place under its documented keyword name, optional arguments and defaults work, the returned objects have the right order, type, shape and dtype, errors arrive as Python exceptions.
- Run the test. It is expected to pass, since the C++ function is already verified. If it fails, the cause is in the wrapper, the registration or the test's expectation; find out which before changing anything. If the failure points to the C++ function itself, stop and report it — do not work around it in the wrapper.

## Phase 4 — Finish
- The new test passes; the complete Python test suite passes.
- The documentation block of the wrapper is complete and matches what the wrapper does; the usage line, the wrapper signature, the dispatcher entry and the comment block name the same arguments in the same order.
- **Review point B:** report the final state: the Python usage line and the submodule, the test results with the actual output of the run, compiler warnings if any, every deviation from the C++ interface or the MEX wrapper and why, and anything that was not tested. Do not report a test as passing that was not run.
- The MEX wrapper is a separate task and is only written on request.

# IMPLEMENTATION GUIDELINES

## File and function naming
- All Python wrapper source files use the `qpy_` prefix and match the C++/MEX function name, so `quadriga_lib::function_name` is wrapped in `qpy_function_name.cpp`.
- By default, the wrapper function itself and the name exposed to Python are both the name of the C++ function it wraps (e.g. `acdf`). The function name is kept as-is — no case conversion (lowercase stays lowercase).
- Deviations are decided case by case. Example: the existing array antenna wrappers are C++ functions named `arrayant_<name>` that are exposed as `quadriga_lib.arrayant.<name>` (the prefix that repeats the submodule is dropped). Do not introduce such a renaming for a new wrapper unless requested.
- The heading of the documentation block and the Usage line always use the name exposed to Python.

## Submodules
- The dispatcher registers each wrapper with one of the submodules (`arrayant`, `channel`, `tools`, `RTtools`, `sos`, `gbsm`). From Python, the user calls e.g. `quadriga_lib.tools.acdf(...)`.
- The submodule usually follows from the section of the C++ documentation block:

| Section in the C++ documentation | Submodule |
| --- | --- |
| Array antenna functions | `arrayant` |
| Channel functions | `channel` |
| Channel statistics | `tools` |
| Math functions | `tools` |
| Miscellaneous / Tools | `tools` |
| Site-specific simulation tools | `RTtools` |
| Sum-of-sinusoids random number generator | `sos` |
| Geometry-based stochastic model | `gbsm` |
| Channel generation functions | not clear (`arrayant` or `channel`): ask |

- If the submodule is not clear from the C++ documentation, ask.
- Submodule `sos`: the wrapper of `quadriga_lib::sos_<name>` is the C++ function `sos_<name>` in `qpy_sos_<name>.cpp`, exposed as `quadriga_lib.sos.<name>` (prefix dropped, as for `arrayant`). The heading of its documentation block and the cross-references follow the rule for prefixed names under "Documentation block" below (`# sos.<name>`, linked as `[[sos.<name>]]`); the Usage line shows the call without the prefix.

## Index and dtype conventions
- C++ and Python both use 0-based indices, so no conversion is performed in either direction. Indices documented as "1-based" in the C++ documentation remain 1-based in Python (the user provides 1-based and gets 1-based back).
- C++ functions may offer dtype = float / double specialization. The Python wrapper only uses the double version, so hardcode dtype to `double` in the call to the C++ function.
- Armadillo types map to their C++ counterparts, e.g. `arma::uword = unsigned long long` (enforced at compile time by static_asserts).
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two Python names with `_in` / `_out` suffixes (or `bins` / `bins_out` style where the input name is short).
- Integer array inputs (indices, counts) are read with the integer type the C++ function expects, e.g. `qd_python_numpy2arma_Col<arma::uword>(element, true)`; the helper casts from whatever the user passes.
- Integer array outputs (face/object/material indices, sub-mesh offsets, counts): prefer zero-copy over casting.
  - If the output size is known and the output is pre-allocated with `qd_python_init_output`, keep the C++ integer type (e.g. `arma::u32_vec` is returned as a `uint32` array). No cast, no copy.
  - If the output must be copied anyway (`qd_python_copy2numpy`, unknown size), cast to signed 64-bit, since Python's native `int` and numpy's default integer are signed 64-bit:
    `qd_python_copy2numpy<unsigned, py::ssize_t>(&u32vec)` or `qd_python_copy2numpy<arma::uword, py::ssize_t>(&uvec)`.
  - The declared return type of the wrapper must match the numpy element type that is actually produced (e.g. `py::array_t<py::ssize_t>`), otherwise pybind11 silently converts the array again.
- Float outputs are unaffected.

## License statement
- At the top of the file, include the following license statement as a comment block:
```
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.
```

## Headers
- You most likely will only need these headers:
```
#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"
```
- `python_arma_adapter.hpp` pulls in pybind11, numpy, and Armadillo. `py` is aliased to `pybind11`.
- If the wrapper converts arrayant dicts, channel dicts or `std::any` datasets (see below), include only
```
#include "python_quadriga_adapter.hpp"
```
  instead. It includes `python_arma_adapter.hpp` and `quadriga_lib.hpp`, so only this one header is needed.

## Helper function signatures
Quadriga-Lib uses Armadillo for data structures. Conversion between numpy arrays / Python objects and Armadillo types is done via the helpers in `python_arma_adapter.hpp`.

The `copy2numpy`, `stack2numpy` and `copy2list` converters take their Armadillo inputs **by pointer** (pass `&obj`, and `nullptr` for an absent optional input) and are templated as `<dtype, dtype_numpy = dtype>`: `dtype` is the Armadillo scalar (deduced from the argument) and `dtype_numpy` is the numpy element type. The combination of `dtype_numpy` and the `im` argument selects the output form:

| `dtype_numpy` | `im` | Result |
| --- | --- | --- |
| real (default) | `nullptr` | plain copy of `re`, cast to `dtype_numpy` if it differs from `dtype` |
| real | given | real array with real and imaginary parts **interleaved along the rows**; the row count doubles |
| `std::complex<...>` | given | genuine complex array built from `re` and `im` |
| `std::complex<...>` | `nullptr` | `re` is interpreted as **interleaved** pairs `[re0, im0, re1, im1, ...]` along the rows; the row count halves (used e.g. for the polarization matrix `M`, 8 rows -> 4 complex rows). It is NOT "imaginary part = 0". |

```cpp
// ---------- Outputs: Armadillo -> numpy (copy) ----------

// Copy a single Armadillo object to a Numpy array (auto = py::array_t<dtype_numpy>)
auto pyarray = qd_python_copy2numpy(&Col, &ColIm, i_elem, transpose);  // Col; im/i_elem/transpose optional
auto pyarray = qd_python_copy2numpy(&Mat, &MatIm, i_col);              // Mat; im/i_col optional
auto pyarray = qd_python_copy2numpy(&Cube, &CubeIm, i_slice);          // Cube; im/i_slice optional
auto pyarray = qd_python_copy2numpy(&Col);                             // simplest form: plain real copy
auto pyarray = qd_python_copy2numpy<unsigned, py::ssize_t>(&Mat);      // integer output: cast to signed 64-bit
auto pyarray = qd_python_copy2numpy<double, std::complex<double>>(&MatRe, &MatIm);  // Re/Im -> complex

// Stack a std::vector of Armadillo objects into one higher-dim Numpy array
// (ragged inputs are zero-padded to the per-axis maximum; frame = last axis; i_vec selects frames)
auto pyarray = qd_python_stack2numpy(&vecCol, &vecColIm, i_vec);       // vector<Col>  -> 2D
auto pyarray = qd_python_stack2numpy(&vecMat, &vecMatIm, i_vec);       // vector<Mat>  -> 3D
auto pyarray = qd_python_stack2numpy(&vecCube, &vecCubeIm, i_vec);     // vector<Cube> -> 4D
auto pyarray = qd_python_stack2numpy<double, std::complex<double>>(&vecCubeRe, &vecCubeIm);  // -> complex 4D

// Copy a std::vector of Armadillo objects to a py::list of Numpy arrays (each entry keeps its own shape)
auto pylist = qd_python_copy2list(&vec, &vecImag, i_vec);              // vec = vector<Col|Mat|Cube>; im/i_vec optional
auto pylist = qd_python_copy2list<arma::Cube<double>, std::complex<double>>(&vecCubeRe, &vecCubeIm);  // -> list of complex
auto pylist = qd_python_copy2list(vecStrings, i_vec);                  // vector of strings, passed BY REFERENCE

// std::unordered_map<std::string, std::vector<dtype>> -> dict of 1D arrays; key_order fixes the key order
auto pydict = qd_python_map2dict(map, key_order);

// ---------- Outputs: allocate in numpy, alias in Armadillo (zero-copy writes) ----------

auto pyarray = qd_python_init_output(n_elem, &Col);                            // 1D
auto pyarray = qd_python_init_output(n_rows, n_cols, &Mat);                    // 2D
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, &Cube);         // 3D
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, n_frames, &vecCubes); // 4D

// ---------- Inputs: numpy -> Armadillo ----------

// Overloads for `const py::array_t<dtype> &` (mandatory inputs)
auto Col     = qd_python_numpy2arma_Col(pyarray, view, strict, var_name, n_elem); // var_name, n_elem optional
auto Mat     = qd_python_numpy2arma_Mat(pyarray, view, strict);
auto Cube    = qd_python_numpy2arma_Cube(pyarray, view, strict);
auto vecCube = qd_python_numpy2arma_vecCube(pyarray, view, strict);   // 3D -> 1 Cube, 4D -> vector<Cube>

// Overloads for `py::handle` (optional inputs); dtype must be given explicitly
auto Col     = qd_python_numpy2arma_Col<dtype>(handle, view, strict);  // no var_name / n_elem here
auto Mat     = qd_python_numpy2arma_Mat<dtype>(handle, view, strict);
auto Cube    = qd_python_numpy2arma_Cube<dtype>(handle, view, strict);
auto vecCube = qd_python_numpy2arma_vecCube<dtype>(handle, view, strict);

// Regular array -> std::vector, split along the last axis (always copies; None -> empty vector)
auto vecCol  = qd_python_numpy2arma_vecCol<dtype>(handle);  // 1D -> 1 Col;  2D -> one Col per column
auto vecMat  = qd_python_numpy2arma_vecMat<dtype>(handle);  // 2D -> 1 Mat;  3D -> one Mat per slice

// Complex numpy -> separate Re/Im Armadillo objects (copy)
qd_python_copy2arma(pyarrayComplex, CubeRe, CubeIm);                       // py::array_t<std::complex<dtype>>
qd_python_numpy2arma_vecCube_Cplx<dtype>(handle_or_pyarray, vecCubeRe, vecCubeIm); // 3D / 4D complex

// py::list of numpy arrays -> std::vector of Armadillo types (ragged, always copies)
auto vecCol     = qd_python_list2vector_Col<dtype>(pylist);
auto vecMat     = qd_python_list2vector_Mat<dtype>(pylist);
auto vecCube    = qd_python_list2vector_Cube<dtype>(pylist);
qd_python_list2vector_Cube_Cplx<dtype>(pylistComplex, vecCubeRe, vecCubeIm);   // Complex -> Re/Im
auto vecStrings = qd_python_list2vector_Strings(handle);  // None -> empty; a single str -> 1 entry; list / tuple of str

// dict of 1D arrays (or None) -> std::unordered_map<std::string, std::vector<dtype>>; all values equal length
auto map = qd_python_dict2map<dtype>(handle);

// Python str / float / int / numpy array -> std::any (type preserved; unsupported type throws)
std::any data = qd_python_anycast(handle, var_name);

// ---------- python_quadriga_adapter.hpp ----------

quadriga_lib::arrayant<double> qd_python_dict2arrayant(const py::dict &arrayant,
        bool view = false, bool strict = false, bool validate = false);
std::vector<quadriga_lib::arrayant<double>> qd_python_dict2arrayant_multi(const py::dict &arrayant,
        bool view = false, bool strict = false, bool validate = false);
py::dict qd_python_arrayant2dict(const quadriga_lib::arrayant<double> &ant);
py::dict qd_python_arrayant2dict_multi(const std::vector<quadriga_lib::arrayant<double>> &ant);

quadriga_lib::channel<double> qd_python_dict2channel(const py::dict &dict, bool validate = false);
std::vector<quadriga_lib::channel<double>> qd_python_list2channel(const py::handle &obj, bool validate = false);
py::dict qd_python_channel2dict(const quadriga_lib::channel<double> &channel,
        const arma::uvec &snap = arma::uvec(), bool validate = false, bool stack = false);
py::list qd_python_channel2list(const std::vector<quadriga_lib::channel<double>> &chan,
        bool validate = false, bool stack = false);

// std::any -> numpy array / Python number / str (type and shape preserved); no value -> None
py::object qd_python_any2numpy(const std::any &dset);
```
- `dtype` (Armadillo scalar): `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`. `dtype_numpy` (numpy element type, defaults to `dtype`): any of those, or `std::complex<float>` / `std::complex<double>`.
- `dtype_numpy` cannot be deduced; to set it you must also spell `dtype`, since it is the second template parameter (e.g. `<double, std::complex<double>>`). For `qd_python_copy2list` the first template parameter is the Armadillo type (e.g. `arma::Cube<double>`).
- Armadillo inputs `re`/`im` are passed by pointer; `im` defaults to `nullptr`. Selectors `i_elem`/`i_col`/`i_slice`/`i_vec` are `arma::uvec` passed by value and default to empty (= use all); `transpose` defaults to `false` (a `Col` with `transpose = true` is emitted as a `(1, n)` row).
- `i_vec` selects which `std::vector` elements (frames) are used; `i_elem`/`i_col`/`i_slice` select elements/columns/slices within a single object. Out-of-range indices throw `std::out_of_range`.
- `copy2numpy` and `copy2list` throw `std::invalid_argument` on a null `re`; `stack2numpy` returns an empty array when `re` is null or empty.
- The `py::handle` overloads of `qd_python_numpy2arma_*` accept `py::none()` and return an empty Armadillo object — used for optional inputs. They also accept anything numpy can convert (lists, scalars, arrays of another dtype); these are cast and copied.
- `view = true` requests a view of the numpy memory when the array is Fortran-contiguous with matching dtype, and falls back to a copy otherwise; `strict = true` throws instead of falling back.
- Lower-level helpers `qd_python_get_shape` and `qd_python_get_list_shape` (shape, strides and data pointers of numpy arrays) exist for special cases; do not use them unless the standard converters cannot do the job.

## Type mapping
| C++ type | Python type | Input | Output |
| --- | --- | --- | --- |
| scalar (`double`, `int`, `bool`, `arma::uword`, ...) | `float`, `int`, `bool` | native type in the wrapper signature | return directly or put into `py::make_tuple` |
| `std::string` | `str` | `const std::string &` in the signature | return `std::string` directly |
| `std::vector<std::string>` | `list` of `str` | `qd_python_list2vector_Strings` | `qd_python_copy2list(vec)` |
| `arma::Col` | 1D array `(n,)` | `qd_python_numpy2arma_Col` | `qd_python_init_output` / `qd_python_copy2numpy` |
| `arma::Mat` | 2D array | `qd_python_numpy2arma_Mat` | `qd_python_init_output` / `qd_python_copy2numpy` |
| `arma::Cube` | 3D array | `qd_python_numpy2arma_Cube` | `qd_python_init_output` / `qd_python_copy2numpy` |
| `std::vector<arma::Cube>`, regular shape | 4D array | `qd_python_numpy2arma_vecCube` | `qd_python_init_output` / `qd_python_stack2numpy` |
| `std::vector<arma::Col/Mat/Cube>`, ragged | `list` of arrays (or stacked array with `stack=True`) | `qd_python_list2vector_*` | `qd_python_copy2list` / `qd_python_stack2numpy` |
| separate `_re` / `_im` Armadillo objects | one complex array | `qd_python_copy2arma`, `qd_python_numpy2arma_vecCube_Cplx`, `qd_python_list2vector_Cube_Cplx` | `qd_python_copy2numpy<double, std::complex<double>>(&re, &im)` and the matching `stack2numpy` / `copy2list` forms |
| `std::unordered_map<std::string, std::vector<dtype>>` | `dict` of 1D arrays | `qd_python_dict2map` | `qd_python_map2dict` |
| `std::any` | `str`, `float`, `int` or array (type preserved) | `qd_python_anycast` | `qd_python_any2numpy` |
| `quadriga_lib::arrayant<dtype>`, also `std::vector` of it | arrayant `dict` | `qd_python_dict2arrayant` / `_multi` | `qd_python_arrayant2dict` / `_multi` |
| `quadriga_lib::channel<dtype>` | channel `dict` | `qd_python_dict2channel` | `qd_python_channel2dict` |
| `std::vector<quadriga_lib::channel<dtype>>` | `list` of channel dicts | `qd_python_list2channel` | `qd_python_channel2list` |

- There is no helper for C++ types that are not listed; ask before adding raw pybind11 / buffer code for them.

## Arrayant dict
- A `quadriga_lib::arrayant<double>` is represented in Python as a `dict` with the keys below. Requires `python_quadriga_adapter.hpp`.

| Key | C++ member | Shape | Required as input |
| --- | --- | --- | --- |
| `e_theta_re` | `e_theta_re` | `(n_elevation, n_azimuth, n_elements)` | yes |
| `e_theta_im` | `e_theta_im` | `(n_elevation, n_azimuth, n_elements)` | yes |
| `e_phi_re` | `e_phi_re` | `(n_elevation, n_azimuth, n_elements)` | yes |
| `e_phi_im` | `e_phi_im` | `(n_elevation, n_azimuth, n_elements)` | yes |
| `azimuth_grid` | `azimuth_grid` | `(n_azimuth,)` | yes |
| `elevation_grid` | `elevation_grid` | `(n_elevation,)` | yes |
| `element_pos` | `element_pos` | `(3, n_elements)` | no; default: zeros |
| `coupling_re` | `coupling_re` | `(n_elements, n_ports)` | no; default: identity |
| `coupling_im` | `coupling_im` | `(n_elements, n_ports)` | no; default: zeros |
| `center_freq` | `center_frequency` | scalar | no; default: 299792458 |
| `name` | `name` | `str` | no |

- Note the name difference: the dict key is `center_freq`, the C++ member is `center_frequency`.
- The field patterns stay split into `_re` / `_im` arrays (not complex), in both directions.
- A frequency-dependent (multi-frequency) arrayant is ONE dict (not a list of dicts, unlike the MATLAB struct array): the four pattern arrays are 4D `(n_elevation, n_azimuth, n_elements, n_freq)`, `center_freq` is a 1D array `(n_freq,)`, the coupling matrices are 2D (shared by all frequencies) or 3D `(n_elements, n_ports, n_freq)`; grids, `element_pos` and `name` are shared. It maps to `std::vector<quadriga_lib::arrayant<double>>`.
- Reading:
  - `qd_python_dict2arrayant_multi(arrayant, view, strict, validate)` accepts both layouts and always returns a `std::vector` (length 1 for a single-frequency dict). Use it for all wrappers that support frequency-dependent antennas.
  - `qd_python_dict2arrayant(arrayant, view, strict, validate)` reads a single-frequency dict only.
  - The argument order is `(dict, view, strict, validate)` — NOT the MEX order `(input, validate, copy)`. A call `qd_python_dict2arrayant(ant, true)` sets `view`, not `validate`.
  - `validate = true` runs the arrayant validation and throws on inconsistent data. Use `false` when the C++ function validates its inputs anyway.
  - Use `view = true` by default; use `view = false` whenever the arrayant object is modified after reading (e.g. in-place member functions such as `rotate_pattern`, `copy_element`), so that the user's arrays are never changed.
- Writing: `return qd_python_arrayant2dict(ant);` or `return qd_python_arrayant2dict_multi(ant_vec);` (the latter returns the single-frequency layout for a vector of length 1).
- The wrapper takes the dict as `const py::dict &arrayant` and returns `py::dict`.
- In the documentation, describe arrayant arguments as: "Dict containing the arrayant data; layout as documented in [[generate]]" and add a note where frequency-dependent input is supported. Do not repeat the key table.

## Channel dict
- A `quadriga_lib::channel<double>` is represented in Python as a `dict` with the keys below; a `std::vector` of channels is a `list` of such dicts. Requires `python_quadriga_adapter.hpp`.
- Members that are a `std::vector` over snapshots in C++ exist in two layouts:
  - "list" model (`stack = false`, default): a `list` with one array per snapshot; the number of paths may differ between snapshots.
  - "stack" model (`stack = true`): one array with the snapshot index on the last axis, zero-padded.

| Key | C++ member | Shape per snapshot (list model) | Type |
| --- | --- | --- | --- |
| `name` | `name` | — | `str` |
| `tx_position` | `tx_pos` | `(3, 1)` or `(3, n_snap)`; plain array in both models | float |
| `rx_position` | `rx_pos` | `(3, 1)` or `(3, n_snap)`; plain array in both models | float |
| `tx_orientation` | `tx_orientation` | `(3, 1)` or `(3, n_snap)`; plain array in both models | float |
| `rx_orientation` | `rx_orientation` | `(3, 1)` or `(3, n_snap)`; plain array in both models | float |
| `coeff` | `coeff_re`, `coeff_im` | `(n_rx, n_tx, n_path)` | complex |
| `delay` | `delay` | `(n_rx, n_tx, n_path)` | float |
| `path_gain` | `path_gain` | `(n_path,)` | float |
| `path_length` | `path_length` | `(n_path,)` | float |
| `path_polarization` | `path_polarization` | `(4, n_path)`; stored interleaved with 8 rows in C++ | complex |
| `path_angles` | `path_angles` | `(n_path, 4)` | float |
| `fbs_pos` | `path_fbs_pos` | `(3, n_path)` | float |
| `lbs_pos` | `path_lbs_pos` | `(3, n_path)` | float |
| `no_interact` | `no_interact` | `(n_path,)` | int |
| `interact_coord` | `interact_coord` | `(3, sum(no_interact))` | float |
| `center_frequency` | `center_frequency` | scalar or `(n_snap,)`; plain array in both models | float |
| `initial_position` | `initial_position` | — | `int` |

- Note the name differences: `tx_position` / `rx_position` and `fbs_pos` / `lbs_pos` in Python vs. `tx_pos` / `rx_pos` and `path_fbs_pos` / `path_lbs_pos` in C++; the coefficients are one complex key `coeff`.
- Reading: `qd_python_list2channel(handle, validate)` accepts a single dict or a list of dicts and always returns a `std::vector`; `qd_python_dict2channel(dict, validate)` reads one dict. All keys are optional; the list and stack layout is detected per key; `coeff_re` / `coeff_im` are accepted as an alternative to `coeff`.
- Writing: `qd_python_channel2list(chan_vec, validate, stack)` or `qd_python_channel2dict(channel, snap, validate, stack)`; `snap` selects snapshots (0-based, empty = all). Only populated members become keys.
- Unstructured data (`par_names` / `par_data`, a list of `std::any`) is NOT part of the channel dict. It is exchanged through a separate `par` dict, converted with `qd_python_anycast` / `qd_python_any2numpy`; see `qpy_hdf5_write_channel.cpp` and `qpy_hdf5_read_channel.cpp`.
- In the documentation, refer to [[hdf5_read_channel]] for the layout instead of repeating the table.

## Exception handling
- No `CALL_QD` macro and no try/catch is needed. pybind11 automatically translates C++ exceptions into Python exceptions, so the wrapper simply calls the C++ function directly: `std::invalid_argument` -> `ValueError`, `std::out_of_range` -> `IndexError`, `std::runtime_error` and other `std::exception` -> `RuntimeError`.
- Errors detected in the wrapper itself are raised the same way, e.g. `throw std::invalid_argument("Input 'x' cannot be empty.");`

## Section statement
- Functions are grouped into sections, marked by a comment block. This is used for grouping in the generated documentation:
```
/*!SECTION
Section name from C++ documentation block
SECTION!*/
```
- Section names currently in use: "Array antenna functions", "Channel functions", "Channel generation functions", "Channel statistics", "Geometry-based stochastic model", "Math functions", "Miscellaneous / Tools", "Site-specific simulation tools", "Sum-of-sinusoids random number generator". The name must match exactly to end up in the same group.
- The section name is not the Python submodule (see "Submodules").

## Documentation block
- Included as a comment block immediately before the wrapper function definition.
- Block starts with `/*!MD` and ends with `MD!*/`.
- First line after `/*!MD` is the function name as exposed to Python (kept as-is, lowercase), followed by a 1-line summary on the next line.
  - If the C++ name carries a prefix that the Python submodule replaces, the heading is the submodule and the name joined by a dot: `sos_val` is exposed as `quadriga_lib.sos.val` and headed `# sos.val`; cross-references are written `[[sos.val]]`. This keeps the heading equal to what the user types and avoids clashes between bare names of different sections (e.g. `generate`).
- The bullet-point description follows after one empty line — there is no `## Description:` heading.
- The 1-line summary should be identical to the C++/MEX documentation block.
- Each bullet contains one concise fact.
- Never use `*` or `|` in formulas or any other text — they are reserved for markdown formatting. Use `·` for multiplication (e.g. `Rz·Ry·Rx`) and `abs()` for absolute values.
- Lines should not exceed 120 characters (soft limit). The block is only compiled to HTML and Markdown, so nothing breaks on longer lines; but it is also read directly in the wrapper source, and it should be readable there without horizontal scrolling. Split longer lines and indent continuation lines by 2 spaces. Table rows may exceed the limit.
- Omit any dtype specifications — they are not relevant for the Python user (the wrapper always uses double). Exception: state the type of integer and complex arguments.
- Use Python types and syntax in descriptions: tuples `(n, m)` for shapes, `list`, `dict`, `str`, `None`, etc.
- Use a `## Usage:` section with the function signature as it should be called from Python:
- Never specify any import statements, e.g. "from quadriga_lib import channel" (these are the same for all API files); only place the function call with the "quadriga_lib" included:
```
## Usage:
```
out1, out2 = quadriga_lib.tools.function_name( in1, in2, optional_in3 )
```
```
- The kwarg names in the Usage line **must match** the pybind11 declaration (i.e. they are the names the user types as keyword arguments).
- Split the usage block across multiple lines if it exceeds 120 characters, indenting continuation lines by 4 spaces and ending the continued line with ` \`.
- If flags change the returned tuple (see "Standard flags"), show one usage line per variant, each preceded by a `#` comment line; later variants may abbreviate the unchanged arguments as `...`.
- Use `## Inputs:` and `## Outputs:` sections with the same one-liner style as in the C++ documentation block. Use the singular `## Input:` / `## Output:` only when there is exactly one input / output. Omit the section if there are none.
- Argument format: ``- **`name`** — Description; further facts separated by semicolons; `(shape)`; default: X``
- Optional inputs are simply marked with their default value (e.g. `default: 201` or `default: None`); the user can omit them or pass `None` for arrays. The default implies that the input is optional; do NOT add "(optional)". Only if an optional input has no default value that can be stated, mark it with `*(optional)*` directly after the argument name.
- Default values must be added to the argument description as they are not obvious from the signature.
- Units must only be specified when not SI (meters, radians, Hz, linear scale).
- C++ returns are merged into the `## Outputs:` section.
- Break argument descriptions longer than 120 characters into multiple lines, indenting continuation lines by 2 spaces.
- An optional `## See also:` section at the end lists related functions as `- [[function_name]] (short reason)`; `[[...]]` becomes a link in the HTML documentation.

## Function body structure
- The wrapper is a regular C++ free function. Its return type depends on the number of outputs:
  - **No output**: `void`.
  - **Single output**: return the object directly (e.g. `py::array_t<double>`, `arma::uword`, `std::string`, `py::list`, `py::dict`).
  - **Multiple outputs**: return `py::tuple` and build it via `py::make_tuple(out1_py, out2_py, ...)`.
- All outputs are always computed and returned — there is no `nargout` equivalent in pybind11. Use this as the default. The only standard exceptions are the `complex` and `stack` flags (see "Standard flags"). Do not add other skip or mode flags unless requested; if the C++ side has prohibitively expensive optional computation, ask.
- Naming convention inside the wrapper:
  - Input numpy arg → Armadillo view with suffix `_a` (e.g. `data` → `data_a`).
  - For in/out splits, the input Armadillo gets suffix `_a` (e.g. `bins` → `bins_a`).
  - Output Armadillo objects use the same name as in the Outputs section of the doc.
  - The corresponding numpy variable returned to Python uses the suffix `_py` (e.g. `cdf_per_set` → `cdf_per_set_py`).
- Keep the wrapper thin: argument conversion, output allocation and the C++ call. Anything that changes numerical results belongs in the C++ library; if a wrapper seems to need it, ask first.
- Performance tweaks in the wrapper — releasing the GIL (`py::gil_scoped_release`), OpenMP, worker threads, batch modes — are NOT part of a new wrapper. They are only added on request, after benchmarking, case by case. (Some existing wrappers contain such optimizations; do not copy them into new wrappers.)

## Input conversion

### Required numeric arrays
- Mandatory array inputs use `const py::array_t<dtype>&` in the wrapper signature:
```
const auto data_a = qd_python_numpy2arma_Mat<double>(data, true);
```
- The `true` second argument means `view=true, strict=false` — try to get a view, fall back to a copy if strides do not match.
- Use this default unless the input is modified later (e.g. the wrapper writes back into it). In that case use `false` (force copy) to avoid mutating the user's numpy buffer.
- Helpers: `qd_python_numpy2arma_Col`, `qd_python_numpy2arma_Mat`, `qd_python_numpy2arma_Cube`.
- dtype can be: `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`. Use `double` to match the C++ call.
- Fixed-length vectors (e.g. positions, orientations) are read with a length check (only available in the `py::array_t` overload of `qd_python_numpy2arma_Col`):
```
const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
```
- In the pybind11 declaration, mandatory inputs have no default: `py::arg("data")`.

### Optional numeric arrays
- Optional array inputs use `py::handle` (NOT `py::array_t<>`) with default `py::none()`:
```
py::handle bins  // in the function signature
const auto bins_in_a = qd_python_numpy2arma_Col<double>(bins, true);
```
- The handle overloads of `qd_python_numpy2arma_*` return an empty Armadillo object when the handle is `None`, which can then be checked via `.empty()`.
- In the pybind11 declaration: `py::arg("bins") = py::none()`.
- Do not use `py::array_t<...>` with an empty-array default (`= py::array_t<double>()`) or `py::object` for new wrappers (some older wrappers do).

### Optional inputs and `nullptr`
- If the C++ function takes an optional input by pointer and distinguishes `nullptr` from an empty object, pass `nullptr` when the user did not provide the input:
```
const arma::uvec *p_obj_ind = obj_ind_a.is_empty() ? nullptr : &obj_ind_a;
```
- If the C++ documentation states that an empty object and `nullptr` behave the same, pass the address of the (possibly empty) object directly.

### 4D numeric arrays as std::vector<arma::Cube>
- Strict 4D inputs `[d1, d2, d3, d4]` map to `std::vector<arma::Cube<dtype>>` of length `d4`, each cube of shape `(d1, d2, d3)`:
```
const auto data = qd_python_numpy2arma_vecCube<dtype>(input, true);
```
- A 3D input yields a vector of length 1.
- Use this — NOT `qd_python_list2vector_Cube` — when the C++ documentation specifies a regular 4D shape like `[d1, d2, d3, d4]`. `qd_python_list2vector_Cube` is for ragged/padded lists of Cubes.

### Lists of Armadillo types (variable-length)
- The C++ function may expect a `std::vector<arma::Mat<dtype>>` (or Col / Cube), used for ragged lists of arrays. The Python user passes a `py::list` of numpy arrays:
```
const auto vec = qd_python_list2vector_Col<dtype>(pylist);
const auto vec = qd_python_list2vector_Mat<dtype>(pylist);
const auto vec = qd_python_list2vector_Cube<dtype>(pylist);
```
- For lists of strings: `qd_python_list2vector_Strings(handle)` (accepts `None`, a single `str`, or a list / tuple of `str`).

### Strings
- Strings are taken as `const std::string&` in the wrapper signature; pybind11 handles conversion automatically. No helper needed.
- Optional strings: `py::arg("name") = ""` in the declaration.

### Scalars
- Scalars use native C++ types directly in the wrapper signature: `arma::uword n_bins`, `double power`, `int count`, `bool flag`, etc.
- Defaults are set in the pybind11 declaration only: `py::arg("n_bins") = 201`. Do not repeat the default in the C++ function signature.
- Allowed types: `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`, `arma::uword`, `bool`.

### Complex arrays
- The C++ library represents complex data as separate `arma::Mat<dtype>` (or `arma::Cube<dtype>`) for the real and imaginary parts — `arma::cx_*` types are not used. This is to avoid copy overhead in the C++ pipeline.
- In Python, **complex numpy arrays are the default**: a C++ `X_re` / `X_im` pair is presented to the Python user as one complex array `X`, for inputs and for outputs.
- Complex input → Re/Im Armadillo objects (copy):
  - single array: `qd_python_copy2arma(complex_pyarr, real_cube, imag_cube);` with the argument declared as `const py::array_t<std::complex<double>> &`
  - 3D / 4D array to `std::vector<arma::Cube>`: `qd_python_numpy2arma_vecCube_Cplx<double>(handle, vec_re, vec_im);`
  - list of arrays: `qd_python_list2vector_Cube_Cplx<double>(pylist, vec_re, vec_im);`
- Re/Im Armadillo objects → complex output (copy):
  - `auto X_py = qd_python_copy2numpy<double, std::complex<double>>(&X_re, &X_im);`
  - `auto X_py = qd_python_stack2numpy<double, std::complex<double>>(&vec_re, &vec_im);`
  - `auto X_py = qd_python_copy2list<arma::Cube<double>, std::complex<double>>(&vec_re, &vec_im);`
- Complex outputs always need one merge copy; the split form can be written zero-copy via `qd_python_init_output`. Where this matters, offer both through the `complex` flag (see "Standard flags").
- Whether a wrapper should additionally accept split `X_re` / `X_im` inputs next to the complex `X` (as some existing wrappers do) is decided case by case; ask.
- Exception: the arrayant dict keeps its split `_re` / `_im` keys.

## Output allocation
- Declare empty Armadillo variables for all output arguments, e.g.:
```
arma::mat cdf_per_set;
arma::vec cdf_avg, bins_out, mu, sig;
```

There are 2 ways to allocate outputs:

### Known output size
- Use the `qd_python_init_output` helper to allocate numpy memory upfront and map the Armadillo object onto it (zero-copy writes from C++):
```
auto cdf_per_set_py = qd_python_init_output(n_bins_out, n_sets, &cdf_per_set);
auto cdf_avg_py     = qd_python_init_output(n_bins_out, &cdf_avg);
```
- Overloads: 1D `(n_elem, &Col)`, 2D `(n_rows, n_cols, &Mat)`, 3D `(n_rows, n_cols, n_slices, &Cube)`, 4D `(n_rows, n_cols, n_slices, n_frames, &vector<Cube>)`.
- Allowed dtypes: `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`.
- The size arguments are `arma::uword`; cast literals and other integer types where the overload would be ambiguous, e.g. `(arma::uword)3`.
- The output size must be known before the C++ call — typically derived from the input dimensions stated in the C++ documentation.
- The size must match exactly what the C++ function produces: if C++ resizes the container, it detaches from the numpy buffer and the returned array is never written.
- The memory is NOT zero-initialized (unlike the MEX helper). If the C++ call is skipped or does not write every element, fill the output explicitly (e.g. `.zeros()`).
- For 4D outputs, each Cube in the vector aliases its slab of the numpy-owned 4D buffer (zero-copy). Do not resize the vector or any of its cubes after this call.

### Unknown output size
- Do not pre-allocate. Pass the empty Armadillo object to the C++ function, which will resize and populate it. After the call, copy to numpy:
```
arma::vec bins_out;
// ... C++ call writes into bins_out ...
auto bins_out_py = qd_python_copy2numpy(&bins_out);
```
- Use this for outputs whose size depends on runtime behavior of the C++ call (e.g. reading from a file, runtime-generated bin counts), and for data that the C++ function returns by value.
- `qd_python_copy2numpy` supports Col, Mat and Cube. Scalars and strings are returned as native C++ types.

### List of Armadillo types (output)
- If the C++ function returns a `std::vector<arma::Col/Mat/Cube>` with ragged sizes, return a list:
```
auto pylist = qd_python_copy2list(&vec_of_mats);
```
- Returns a `py::list` of numpy arrays, one per vector entry. Each entry preserves its own shape (ragged supported).
- To return one zero-padded array with the vector index on the last axis instead, use `qd_python_stack2numpy(&vec_of_mats)`; see the `stack` flag.
- For a regular 4D output with known dimensions, prefer the `qd_python_init_output(..., &vector<Cube>)` overload — it is zero-copy.

## Optional outputs / always-allocate policy
- Because all outputs are always returned, there is **no nullptr guard on output pointers** in the wrapper. The output Armadillo objects are always allocated (via `init_output`) or always passed by address (for unknown-size outputs), and the C++ function always receives a valid non-null pointer.
- Drop the MATLAB-style `var.empty() ? nullptr : &var` pattern for outputs. (It is still used for optional inputs, see "Optional inputs and `nullptr`".)
- Exception: the in/out param pattern. When a C++ argument is both input and output (e.g. `bins`), the wrapper:
  1. Reads the input arma object (empty if user passed `None`).
  2. Declares a separate output arma object (empty by default).
  3. If the input arma is non-empty, copies it into the output arma so the C++ side uses it as-is.
  4. Passes the address of the output arma to C++. The C++ side either keeps the existing content or resizes/generates new content.
  5. Copies the output arma to numpy after the call.

## Standard flags
- Two keyword flags are standard and are added wherever they apply. They are the last arguments of the wrapper, in this order, and each gets one line in the `## Inputs:` section.
- **`complex`** (`bool`, default: `true`) — applies when the C++ function outputs `X_re` / `X_im` pairs.
  - `complex=True`: each pair is returned as one complex array `X`.
  - `complex=False`: the pair is returned as two real arrays `X_re`, `X_im`, in place of `X` in the tuple; use `qd_python_init_output` for zero-copy where the size is known.
  - Note: the existing wrappers that already have this flag default to `false`; do not change them unless requested.
- **`stack`** (`bool`, default: `false`) — applies when the C++ function outputs a `std::vector` of Armadillo objects with possibly ragged sizes (typically one entry per snapshot).
  - `stack=False`: return a `list` of arrays (`qd_python_copy2list`).
  - `stack=True`: return one zero-padded array with the vector index on the last axis (`qd_python_stack2numpy`).
- Outputs whose Python type depends on a flag are held in a `py::object` (or the wrapper returns from separate branches).
- Any other flag that changes the returned tuple (as e.g. `angles`, `dist`, `combine` in existing wrappers) is not standard: ask before adding one.

## Call the C++ function
- Call the underlying C++ function directly with the converted inputs and output pointers:
```
quadriga_lib::function_name<double>(arg1, arg2, &optional_arg3, &optional_output4);
```
- Hardcode `<double>` for templated functions.
- No try/catch (see "Exception handling").
- After the C++ call, copy any unknown-size outputs to numpy via `qd_python_copy2numpy`.

## Return value
- Single output: return the numpy / scalar / string / list object directly.
```
return out_py;
```
- Multiple outputs: build a tuple and return.
```
return py::make_tuple(out1_py, out2_py, out3_py, out4_py);
```
- The order in the tuple must match the order in the `## Usage:` line of the documentation.

## pybind11 registration
- Register the wrapper in the dispatcher file of its submodule (`python_arrayant.cpp`, `python_channel.cpp`, `python_tools.cpp`, `python_RTtools.cpp`, `python_sos.cpp`, `python_gbsm.cpp`; `python_main.cpp` for top-level functions):
  1. Add `#include "qpy_function_name.cpp"` to the include list at the top of the dispatcher.
  2. Add the `m.def(...)` call inside the `quadriga_lib_<submodule>(py::module_ &m)` function.
- In addition, repeat the same registration as a comment block at the end of the wrapper file, so that the wrapper file documents its own Python signature:
```
// pybind11 declaration:
// m.def("function_name", &function_name,
//       py::arg("input1"),
//       py::arg("input2") = py::none(),
//       py::arg("n_bins") = 201);
```
- The comment block and the dispatcher entry must be identical. When changing the arguments of an existing wrapper, update the function signature, the dispatcher, the comment block and the Usage line together.
- The kwarg names in `py::arg(...)` must match the C++ parameter names of the wrapper AND the names used in the `## Usage:` line of the documentation. These are what the user types as keyword arguments from Python.
- One `py::arg(...)` line per parameter, in the order of the wrapper signature. Include defaults for all optional parameters; mandatory parameters have no default and come first.

# OPTIONAL: REFERENCE FROM MEX WRAPPER
- If a MEX wrapper for the same C++ function is provided as additional context, treat it as the authoritative reference for the new wrapper: argument order, in/out splits, default values, descriptive argument renaming, and which outputs are computed.
- Diverge from the MEX wrapper only where Python idiom or this spec demands it. Specifically:
  - **Indices**: stay 0-based; no `+1` / `-1` conversions and no zero-checks.
  - **Outputs**: always-allocate and always-return; drop all `nlhs > X` gates and all `var.empty() ? nullptr : &var` patterns for outputs.
  - **Function name in MD heading**: no case conversion.
  - **Lists**: MATLAB cell arrays → `py::list`. Zero-padded MATLAB arrays that stand for a ragged `std::vector` → `py::list` of arrays (plus the `stack` flag).
  - **Complex**: MATLAB split Re/Im → one complex numpy array by default (plus the `complex` flag).
  - **Structs**: MATLAB struct → `dict`; arrayant struct array → one arrayant dict with 4D patterns; channel struct array → list of channel dicts.
  - **Optional inputs**: MATLAB `[]` → `None`.
  - **Argument names in the Usage line**: must match the pybind11 kwarg names exactly, not the MATLAB names.
  - **No `CALL_QD` / try-catch**: pybind11 handles exceptions automatically.
  - **No nargout-style gates**: every output is always computed and returned.
  - **Legacy MEX features** (separate arrayant field inputs / outputs, dummy return values of void functions) are not ported.
- The MEX wrapper is optional context. When it is not provided, generate the Python wrapper from the C++ documentation block alone. Existing Python wrappers may differ from their MEX counterparts; do not align them unless requested.

# TESTS

## Scope
- The Python test verifies the **wrapper**, not the physics. The numerical behavior of the C++ function is pinned by its Catch2 suite and is not re-derived here: no oracles for the physics, no sweeps over the parameter space.
- What the test must show:
  - **Data lands where it should**: every input reaches the C++ argument it belongs to, positionally and by keyword; every output comes back in the documented position of the tuple, with the documented type, shape and dtype.
  - **Optional arguments and defaults work**: omitted, passed as `None`, and passed explicitly.
  - **Array handling works**: the result does not depend on the memory layout or dtype of the input arrays, and inputs are not modified.
  - **Flags work**: `complex` and `stack` change the form of the output, not its values.
  - **Errors are passed correctly**: exceptions thrown by the C++ function and by the wrapper arrive as the right Python exception type with the right message.
- For this, a few small cases with known results are enough: values that can be written down by hand, or that follow from the input by construction. Use distinctive inputs (different values per element, non-square shapes, non-symmetric index lists) so that a swapped argument, a transposed array or a wrong axis order changes the result visibly.
- Keep it lean: no duplicated cases, no helper layers. The test is as long as it needs to be to cover the checklist below.

## File and structure
- One file per wrapper: `tests/python_tests/test_<function_name>.py`, built on `unittest` and run by pytest. Layout:
```
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.<submodule>.<function_name>.
# Scope: argument and output handling, defaults, flags, error paths.
# Physics validation lives in the Catch2 suite (test_<function_name>.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def setUp(self):
        ...

    def test_basic_call(self):
        ...

    def test_errors(self):
        ...


if __name__ == "__main__":
    unittest.main()
```
- One class `test_case(unittest.TestCase)` per file; one method per aspect of the checklist below, named `test_<what_is_checked>`. Shared inputs are built in `setUp` or in small module-level helper functions.
- Call the function through its full path, `quadriga_lib.<submodule>.<function_name>(...)`, as in the Usage line of the documentation.
- Use other library functions for supporting tasks (`quadriga_lib.RTtools.cube` for a mesh, `quadriga_lib.arrayant.generate` for an antenna, `quadriga_lib.RTtools.ray_triangle_intersect` for hit indices, ...) instead of building such data by hand; `html_docu/quadriga_lib_api_python.md` lists what is available.
- Tests that write files use `tempfile` (or distinct file names) and remove the files at the end.
- The random generator, where used, is seeded: `rng = np.random.default_rng(0)`.

## Assertions
- `self.assertEqual(a, b)` — shapes, dtypes, lengths, strings, scalars that must match exactly, e.g. `self.assertEqual(out.shape, (201, 2))`, `self.assertEqual(out.dtype, np.float64)`
- `self.assertIsInstance(out, tuple)` / `list` / `dict` / `np.ndarray` — types of returned objects
- `npt.assert_array_equal(a, b)` — integer, boolean and index arrays, and values that must pass through unchanged
- `npt.assert_allclose(a, b, atol=tol, rtol=0)` — floating-point and complex arrays; state `atol` and `rtol` explicitly
- Tolerances: `1e-14` to `1e-12` for values that are exact by construction in double, around `1e-6` where single precision is involved. The wrapper does not change values, so a tight tolerance is the norm; a loose one needs a comment.
- Errors: assert the exception type and a characteristic substring of the message:
```
with self.assertRaises(ValueError) as ctx:
    quadriga_lib.tools.function_name(bad_input)
self.assertIn("n_bins", str(ctx.exception))
```
  - `ValueError` for `std::invalid_argument`, `IndexError` for `std::out_of_range`, `RuntimeError` for `std::runtime_error`, `TypeError` when pybind11 rejects the call itself (missing mandatory argument, unknown keyword, an object that cannot be converted to the declared type).
  - Assert the specific type. Do not use a catch-all tuple of exception types (some older tests do); if the type is not what this list predicts, that is a finding to report.

## What every test covers
- **Basic call**: all mandatory inputs; check the type of the returned object (tuple and its length for several outputs), and the type, shape, dtype and values of every output.
- **Keyword arguments**: one call that passes every argument by keyword, using the names from the Usage line, gives the same result as the positional call. This is the check that the Usage line, the wrapper signature and the registration agree; a wrong name raises `TypeError`.
- **Each optional input**: omitted, passed as `None` where the default is `None` (must give the same result as omitted), and passed with a non-default value that visibly changes the result. This also shows that the default in the registration matches the documentation.
- **Array memory layout**: the same data as C-ordered array, as Fortran-ordered array (`np.asfortranarray`), and as a non-contiguous view (e.g. a slice with a step) gives the same result. This exercises both the view and the copy path of the input helpers.
- **Array dtype**: at least one call with `float32` or integer arrays (and a plain Python list for an optional input) gives the same result as the `float64` call.
- **Inputs are not modified**: keep a copy of the inputs before the call and compare afterwards — mandatory for wrappers that work in place on a copy (arrayant dicts, index arrays), cheap everywhere else.
- **Output dtypes**: `float64` for real outputs, `complex128` for complex outputs; integer outputs as documented (the C++ integer type for pre-allocated outputs, `int64` for copied ones).
- **Indices** (if the function has index arguments): they are 0-based in and out (the smallest valid value is 0), except those documented as 1-based, which pass through unchanged. Use an input in which the index matters and is not symmetric.
- **`complex` flag** (if present): `complex=True` returns one complex array per pair; `complex=False` returns the two real arrays in its place; real and imaginary part of the complex result equal the split result. Check the length of the tuple in both cases.
- **`stack` flag** (if present): `stack=False` returns a list with one array per entry; `stack=True` returns one array with the entry index on the last axis; the entries agree (up to zero padding for ragged data).
- **Dicts** (arrayant, channel, material properties): a dict with only the mandatory keys is accepted; an output dict has the documented keys and shapes; a round trip (output of one call as input of the next) works; for arrayants, the single-frequency (3D) and the frequency-dependent (4D) layout are both exercised where supported.
- **Errors from C++**: at least one case in which the C++ function throws (e.g. a wrong number of columns, an out-of-range scalar), to show that the exception type and message come through.
- **Errors from the wrapper**: every `throw` in the wrapper itself is reached by one case.
- **Errors from pybind11**: a missing mandatory argument and an unknown keyword raise `TypeError`.
- Light sanity of the values is welcome where it is free (a known closed-form result for the simplest input); anything beyond that belongs in the Catch2 suite.

# REFERENCE EXAMPLE

C++ documentation block for the function `acdf`:

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each column CDF are averaged,
  then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back;
  if non-empty, those bin centers are used; if `nullptr`, bins are auto-generated internally

## Declaration:
```
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
    arma::Col<dtype> *bins = nullptr,
    arma::Mat<dtype> *cdf_per_set = nullptr,
    arma::Col<dtype> *cdf_avg = nullptr,
    arma::Col<dtype> *mu = nullptr,
    arma::Col<dtype> *sig = nullptr,
    arma::uword n_bins = 201);
```

## Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins`** *(optional)* — Bin centers; auto-generated and stored back if pointing to empty vector,
  used as-is if non-empty, ignored if `nullptr`; `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

## Outputs:
- **`cdf_per_set`** *(optional)* — Individual CDFs, one per column of data; `[n_bins, n_sets]`
- **`cdf_avg`** *(optional)* — Averaged CDF via quantile-space averaging across data sets; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets, `[9]`
MD!*/

Complete Python wrapper implementation for the `acdf` function based on the above documentation block, following the implementation guidelines provided (file `api_python/qpy_acdf.cpp`):

```
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each
  column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` is empty, equally spaced bins spanning the data range are generated

## Usage:
```
cdf_per_set, bins_out, cdf_avg, mu, sig = quadriga_lib.tools.acdf( data, bins, n_bins )
```

## Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `(n_samples, n_sets)`
- **`bins`** — Bin centers; used as-is if non-empty; if `None` or empty, equally spaced bins spanning
  the data range are auto-generated; `(n_bins_in,)` or `None`; default: `None`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty `bins`
  are provided; default: 201

## Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data; `(n_bins_out, n_sets)`
- **`bins_out`** — Auto-generated bins; copy of `bins` when non-empty `bins` are provided;
  `n_bins_out = n_bins` or `n_bins_out = n_bins_in`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across data sets; `(n_bins_out,)`
- **`mu`** — Mean of the 0.1–0.9 quantiles across data sets; `(9,)`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across data sets; `(9,)`
MD!*/

py::tuple acdf(const py::array_t<double> &data,
               py::handle bins,
               arma::uword n_bins)
{
    // Read input data
    const auto data_a = qd_python_numpy2arma_Mat<double>(data, true);
    const auto bins_in_a = qd_python_numpy2arma_Col<double>(bins, true);

    arma::uword n_bins_out = bins_in_a.empty() ? n_bins : bins_in_a.n_elem;
    arma::uword n_sets = data_a.n_cols;

    // Output allocation
    arma::mat cdf_per_set;
    arma::vec cdf_avg, bins_out, mu, sig;

    auto cdf_per_set_py = qd_python_init_output(n_bins_out, n_sets, &cdf_per_set);
    auto cdf_avg_py     = qd_python_init_output(n_bins_out, &cdf_avg);
    auto mu_py          = qd_python_init_output(9, &mu);
    auto sig_py         = qd_python_init_output(9, &sig);

    // Special case for bins
    if (!bins_in_a.empty())
        bins_out = bins_in_a;

    // Call library function
    quadriga_lib::acdf<double>(data_a, &bins_out, &cdf_per_set, &cdf_avg, &mu, &sig, n_bins);

    // Copy to python
    auto bins_out_py = qd_python_copy2numpy(&bins_out);

    // Return tuple
    return py::make_tuple(cdf_per_set_py, bins_out_py, cdf_avg_py, mu_py, sig_py);
}

// pybind11 declaration:
// m.def("acdf", &acdf,
//       py::arg("data"),
//       py::arg("bins") = py::none(),
//       py::arg("n_bins") = 201);
```

Registration in the dispatcher `api_python/python_tools.cpp` (identical to the comment block above):

```
// Include parts
#include "qpy_acdf.cpp"

void quadriga_lib_tools(py::module_ &m)
{
    m.def("acdf", &acdf,
          py::arg("data"),
          py::arg("bins") = py::none(),
          py::arg("n_bins") = 201);
}
```

Test for the `acdf` wrapper, `tests/python_tests/test_acdf.py`:

```
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.acdf.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_acdf.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def setUp(self):
        # Two data sets with different, known content; shape (100, 2)
        self.data = np.column_stack((np.arange(1.0, 101.0), np.arange(101.0, 201.0)))

    def test_basic_call(self):
        out = quadriga_lib.tools.acdf(self.data)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)
        cdf_per_set, bins_out, cdf_avg, mu, sig = out

        self.assertEqual(cdf_per_set.shape, (201, 2))
        self.assertEqual(bins_out.shape, (201,))
        self.assertEqual(cdf_avg.shape, (201,))
        self.assertEqual(mu.shape, (9,))
        self.assertEqual(sig.shape, (9,))
        for x in out:
            self.assertEqual(x.dtype, np.float64)

        npt.assert_allclose(bins_out[[0, -1]], [1.0, 200.0], atol=1e-12, rtol=0)  # bins span the data
        npt.assert_allclose(cdf_per_set[-1, :], [1.0, 1.0], atol=1e-12, rtol=0)   # each CDF ends at 1
        self.assertGreater(cdf_per_set[100, 0], 0.99)                             # columns are not swapped
        self.assertLess(cdf_per_set[99, 1], 0.01)

    def test_keyword_arguments(self):
        ref = quadriga_lib.tools.acdf(self.data, None, 11)
        out = quadriga_lib.tools.acdf(data=self.data, bins=None, n_bins=11)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        ref = quadriga_lib.tools.acdf(self.data)

        # bins = None is the same as omitted
        out = quadriga_lib.tools.acdf(self.data, None)
        npt.assert_array_equal(out[0], ref[0])

        # n_bins changes the output size
        cdf_per_set, bins_out, _, _, _ = quadriga_lib.tools.acdf(self.data, n_bins=11)
        self.assertEqual(cdf_per_set.shape, (11, 2))
        self.assertEqual(bins_out.shape, (11,))

        # Explicit bins are used as-is and returned; n_bins is then ignored
        bins = np.linspace(0.0, 250.0, 26)
        cdf_per_set, bins_out, _, _, _ = quadriga_lib.tools.acdf(self.data, bins, 5)
        self.assertEqual(cdf_per_set.shape, (26, 2))
        npt.assert_allclose(bins_out, bins, atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.tools.acdf(self.data)

        variants = [
            np.ascontiguousarray(self.data),              # C order
            np.asfortranarray(self.data),                 # Fortran order
            np.repeat(self.data, 2, axis=0)[::2, :],      # non-contiguous view
            self.data.astype(np.float32),                 # other dtype
        ]
        for data in variants:
            out = quadriga_lib.tools.acdf(data)
            self.assertEqual(out[0].dtype, np.float64)
            npt.assert_allclose(out[0], ref[0], atol=1e-6, rtol=0)

        # bins given as a plain list
        out = quadriga_lib.tools.acdf(self.data, [0.0, 100.0, 200.0])
        self.assertEqual(out[0].shape, (3, 2))

    def test_inputs_not_modified(self):
        data = self.data.copy()
        bins = np.linspace(0.0, 250.0, 26)
        bins_copy = bins.copy()
        quadriga_lib.tools.acdf(data, bins)
        npt.assert_array_equal(data, self.data)
        npt.assert_array_equal(bins, bins_copy)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.acdf(self.data, n_bins=1)
        self.assertIn("n_bins", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.acdf(np.zeros((0, 0)))
        self.assertIn("data", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.acdf()
        with self.assertRaises(TypeError):
            quadriga_lib.tools.acdf(self.data, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
```
