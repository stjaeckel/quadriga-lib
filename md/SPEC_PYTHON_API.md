quadriga-lib is a C++ radio channel modelling library with MATLAB/Octave and Python APIs. This document gives the conventions for the Python wrapper (pybind11 v3) of a C++ function: the wrapper in `api_python/`, its registration, and its test in `tests/python_tests/`. The wrapper converts arguments, allocates outputs and calls the C++ function; it should be user-friendly for Python users while keeping the performance of the C++ library. Like `md/SPEC_CPP_API.md`, it is persistent and used at every stage of the work; the Python interface is worked out in the plan or the spec, and the workflow (reviews, stops) is set in the plan. Where this document says "ask", "on request" or "unless requested", the point belongs in the plan or the spec; an implementer who finds it unsettled stops and reports it. The MEX wrappers are covered by `md/SPEC_MATLAB_API.md`.

# C++ CONVENTIONS
- `md/SPEC_CPP_API.md` applies to wrappers and tests as well and is not repeated here: "GENERAL CONVENTIONS" (types, shapes, index bases, units, citations, lean code, MSVC), "Input validation" (exceptions), "File-local helpers" (helper scoping), "License statement" (`#` comments in Python files) and "Documentation block".

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
  - Each starts with a "Function Index"; use it as described in `md/SPEC_CPP_API.md` ("Fixtures and oracles").
- The text in the `/*!MD ... MD!*/` block of a wrapper is merged into the Markdown overview and compiled to HTML. It also serves as the reference when reading the wrapper source. The delimiters `/*!MD`, `MD!*/`, `/*!SECTION`, `SECTION!*/` are matched literally, each on its own line.
- Commands (Linux, from the repository root):
  - `make python` — build the library and the Python module into `lib/`
  - `make python_test` — run the complete Python test suite (`python3 -m pytest tests/python_tests -x -s`)
  - `python3 -m pytest tests/python_tests/test_<function_name>.py -x -s` — run one test file; this is the normal way to iterate on one wrapper

# INTERFACE: WHAT THE PLAN OR SPEC SETTLES
The interface is derived from the C++ declaration and documentation block, existing or settled in the same plan or spec, and from the MEX wrapper where one exists (see "REFERENCE FROM THE MEX WRAPPER"). The plan or the spec settles the Python interface:
- the Python name and the submodule
- the usage line: order and kwarg names of the inputs, order of the returned tuple, in/out splits
- which inputs are optional and their defaults
- whether the `complex` and `stack` flags apply
- the type of every output (array dtype, list, dict, scalar), its size, and whether the wrapper can know it before the call
- the helper for every argument, and the section name
- anything that does not fit this document: an unclear submodule, a C++ type without helper, an output order that is not obvious, a flag beyond `complex` / `stack`, logic that would have to live in the wrapper

Existing wrappers that show the patterns:
- `qpy_o2i_loss.cpp` — a complete recent example: array and string-list inputs, a default, outputs of known size, an integer index output; registered in `python_gbsm.cpp`, test in `tests/python_tests/test_o2i_loss.py` (its documentation block predates the lean rules)
- `qpy_acdf.cpp` — array inputs, optional array input, outputs of known and unknown size
- `qpy_obj_file_read.cpp` — outputs of runtime size, integer index outputs, string lists, map-to-dict
- `qpy_obj_file_write.cpp` — many optional inputs passed to C++ as `nullptr` when absent
- `qpy_calc_delay_spread.cpp` — ragged inputs as lists of arrays
- `qpy_arrayant_rotate_pattern.cpp` — arrayant dict in / dict out, in-place modification
- `qpy_get_channels_multifreq.cpp` — arrayant dicts, 3-element position vectors, 4D outputs, `complex` flag
- `qpy_quantize_delays.cpp` — `complex` and `stack` flags for lists of arrays
- `qpy_hdf5_write_channel.cpp`, `qpy_hdf5_read_channel.cpp` — channel dicts, unstructured data

# IMPLEMENTATION ORDER
Unlike the C++ work, the wrapper is written first and the test second: the physics is already verified by the Catch2 suite, and the Python test exercises the wrapper.
1. **Wrapper and registration.** Write `api_python/qpy_<function_name>.cpp` following the "IMPLEMENTATION GUIDELINES". Register it in the dispatcher of its submodule AND add the identical registration as a comment block at the end of the wrapper file. It must compile without warnings.
2. **Test.** Write `tests/python_tests/test_<function_name>.py` (see "TESTS") and run it. It is expected to pass. If it fails, the cause is in the wrapper, the registration or the test's expectation; find out which before changing anything. If the failure points to the C++ function itself, stop and report it; do not work around it in the wrapper.
3. **Finish.** The new test passes; the complete Python test suite passes. The documentation block of the wrapper is complete and matches what the wrapper does; the usage line, the wrapper signature, the dispatcher entry and the comment block name the same arguments in the same order.

# HAND-OVER REPORT
- the Python usage line and the submodule
- the test results with the actual output of the run; do not report a test as passing that was not run
- compiler warnings, if any
- every deviation from the C++ interface or the MEX wrapper and why, and anything that was not tested

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
- Submodule `sos`: the wrapper of `quadriga_lib::sos_<name>` is the C++ function `sos_<name>` in `qpy_sos_<name>.cpp`, exposed as `quadriga_lib.sos.<name>` (prefix dropped, as for `arrayant`). Its documentation block is headed `# sos.<name>` and linked as `[[sos.<name>]]`, so that the heading equals what the user types and bare names of different sections (e.g. `generate`) do not clash; the Usage line shows the call `quadriga_lib.sos.<name>(...)`. The arrayant wrappers keep the bare name (`# generate`).

## Index and dtype conventions
- C++ and Python both use 0-based indices, so no conversion is performed in either direction. Indices documented as "1-based" in the C++ documentation remain 1-based in Python (the user provides 1-based and gets 1-based back).
- Templated C++ functions are called with `double`. Functions with fixed types (e.g. the float-only ray-tracing pipeline) are called with their type, and their outputs keep it.
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two Python names with `_in` / `_out` suffixes (or `bins` / `bins_out` style where the input name is short).
- Integer array inputs (indices, counts) are read with the integer type the C++ function expects, e.g. `qd_python_numpy2arma_Col<arma::uword>(element, true)`; the helper casts from whatever the user passes.
- Integer array outputs (face/object/material indices, sub-mesh offsets, counts): prefer zero-copy over casting.
  - If the output size is known and the output is pre-allocated with `qd_python_init_output`, keep the C++ integer type (e.g. `arma::u32_vec` is returned as a `uint32` array). No cast, no copy.
  - If the output must be copied anyway (`qd_python_copy2numpy`, unknown size), cast to signed 64-bit, since Python's native `int` and numpy's default integer are signed 64-bit:
    `qd_python_copy2numpy<unsigned, py::ssize_t>(&u32vec)` or `qd_python_copy2numpy<arma::uword, py::ssize_t>(&uvec)`.
  - The declared return type of the wrapper must match the numpy element type that is actually produced (e.g. `py::array_t<py::ssize_t>`), otherwise pybind11 silently converts the array again.
- Float outputs are unaffected.

## Headers
```
#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"
```
- `python_arma_adapter.hpp` pulls in pybind11, numpy, and Armadillo. `py` is aliased to `pybind11`.
- If the wrapper converts arrayant dicts, channel dicts or `std::any` datasets, or reads a flag (`qd_python_flag`), include only `#include "python_quadriga_adapter.hpp"` instead; it includes the two headers above.

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
- Place the section block of the C++ function before the documentation block; the name must match one of the names in `md/SPEC_CPP_API.md` ("Section statement") exactly. The section name is not the Python submodule (see "Submodules").
```
/*!SECTION
Section name from C++ documentation block
SECTION!*/
```

## Documentation block
- The format and the lean rules of the C++ block apply (`md/SPEC_CPP_API.md`, "Documentation block"): same one-line summary, description as bullets without a heading, singular `## Input:` / `## Output:` for one entry, `## See also:`, no `*` or `|` in text, lines up to 120 characters with continuation lines indented by 2. Do not expand what is brief in the C++ block. The differences:
- The heading is the name exposed to Python, without case conversion (`sos` wrappers: see "Submodules").
- Omit dtype specifications (except for integer and complex arguments). Use Python types and syntax: tuples `(n, m)` for shapes, `list`, `dict`, `str`, `None`.
- A `## Usage:` section replaces the Declaration. It shows the call with its full path and no import statements; split it at 120 characters with continuation lines indented by 4 and ending in ` \`. If flags change the returned tuple (see "Standard flags"), show one usage line per variant, each preceded by a `#` comment line; later variants may abbreviate unchanged arguments as `...`.
```
## Usage:
```
out1, out2 = quadriga_lib.tools.function_name( in1, in2, optional_in3 )
```
```
- Argument format: ``- **`name`** — Description; further facts separated by semicolons; `(shape)`; default: X``
- Optional inputs state their default (`default: 201`, `default: None`), which is not visible in the Usage line; the user can omit them or pass `None` for arrays. Mark an input `*(optional)*` only if no default can be stated.
- State the unit only when not SI. C++ returns are listed under `## Outputs:`.

## Function body structure
- The wrapper is a regular C++ free function. Its return type depends on the number of outputs:
  - **No output**: `void`.
  - **Single output**: return the object directly (e.g. `py::array_t<double>`, `arma::uword`, `std::string`, `py::list`, `py::dict`).
  - **Multiple outputs**: return `py::tuple` and build it via `py::make_tuple(out1_py, out2_py, ...)`.
- All outputs are always computed and returned: there is no `nargout` equivalent in pybind11, and the C++ function always gets a valid pointer for every output. The only standard exceptions are the `complex` and `stack` flags (see "Standard flags"). Do not add other skip or mode flags unless requested; if the C++ side has prohibitively expensive optional computation, ask.
- In/out arguments (e.g. `bins` / `bins_out`): read the input (empty for `None`); declare a separate output object and copy a non-empty input into it; pass the output's address to C++, which keeps or regenerates the content; copy it to numpy after the call.
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
- Fixed-length vectors (e.g. positions, orientations) are read with a length check (only available in the `py::array_t` overload of `qd_python_numpy2arma_Col`):
```
const auto tx_pos_a = qd_python_numpy2arma_Col(tx_pos, true, false, "tx_pos", 3);
```
- In the pybind11 declaration, mandatory array inputs have no default and refuse `None`: `py::arg("data").none(false)`. Without it, `None` is converted into an array instead of raising `TypeError`.

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

### 4D arrays and lists of arrays
- A regular 4D shape `[d1, d2, d3, d4]` is read with `qd_python_numpy2arma_vecCube<dtype>(input, true)` (a 3D input yields a vector of length 1); `qd_python_list2vector_*` is only for ragged lists of arrays, which the user passes as a `list`.
- Lists of strings: `qd_python_list2vector_Strings(handle)`.

### Strings
- Strings are taken as `const std::string&` in the wrapper signature; pybind11 handles conversion automatically. No helper needed.
- Optional strings: `py::arg("name") = ""` in the declaration.

### Scalars
- Scalars use native C++ types directly in the wrapper signature: `arma::uword n_bins`, `double power`, `int count`, etc. Flags (bool options) follow "Flags" below.
- Defaults are set in the pybind11 declaration only: `py::arg("n_bins") = 201`. Do not repeat the default in the C++ function signature.

### Flags
- A flag is taken as `py::handle` and read with `qd_python_flag(handle, default_value, "name")` (`python_quadriga_adapter.hpp`): `None` selects the default, a Python `bool` or `numpy.bool_` gives its value, any other type (float, int, str, array) raises `TypeError` with `Input '<name>' must be a bool or None.`. A plain `bool` parameter is not used: pybind11 would turn `None` into `False` (silently not the default) and accept other types.
- The default stays in the pybind11 declaration (`py::arg("flag") = true`) and is passed again as `default_value`, so that `None` and an omitted argument behave the same.
- The `## Inputs:` line states the default; one bullet of the description states the rule once for all flags of the function ("take a bool or `numpy.bool_`; `None` selects the default, any other type raises `TypeError`").
- Tests: `None` equals the default, a non-bool value (e.g. `2.5`) raises `TypeError`.
- The `gbsm` submodule follows this rule for all its flags; older wrappers with plain `bool` flags predate it.

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
- A `std::vector<arma::Col/Mat/Cube>` with ragged sizes is returned as a `list` (`qd_python_copy2list`), or as one zero-padded array with the vector index on the last axis (`qd_python_stack2numpy`, see the `stack` flag). A regular 4D output with known dimensions uses the zero-copy `qd_python_init_output(..., &vector<Cube>)` overload.

## Standard flags
- Two keyword flags are standard and are added wherever they apply. They are the last arguments of the wrapper, in this order, and each gets one line in the `## Inputs:` section.
- Both are flags in the sense of "Flags" (read with `qd_python_flag`).
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
- The kwarg names in `py::arg(...)` must match the C++ parameter names of the wrapper AND the names in the `## Usage:` line. These are what the user types as keyword arguments from Python.
- One `py::arg(...)` line per parameter, in the order of the wrapper signature. Include defaults for all optional parameters; mandatory parameters have no default and come first.

# REFERENCE FROM THE MEX WRAPPER
- Where a MEX wrapper exists, the plan or spec takes its argument order, in/out splits, default values, argument names and computed outputs as the reference. The Python wrapper differs only in:
  - **Indices**: stay 0-based; no `+1` / `-1` conversions and no zero-checks.
  - **Outputs**: always computed and returned; no `nlhs` gates or `nullptr` patterns for outputs.
  - **Heading**: no case conversion.
  - **Lists**: MATLAB cell arrays and zero-padded arrays that stand for a ragged `std::vector` become a `list` (plus the `stack` flag).
  - **Complex**: MATLAB split Re/Im becomes one complex array by default (plus the `complex` flag).
  - **Structs**: MATLAB struct → `dict`; arrayant struct array → one arrayant dict with 4D patterns; channel struct array → list of channel dicts.
  - **Optional inputs**: MATLAB `[]` → `None`.
  - **Exceptions**: no `CALL_QD`; pybind11 translates them (see "Exception handling").
  - **Legacy MEX features** (separate arrayant field inputs / outputs, dummy return values of void functions) are not ported.

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
- Keep it lean: no duplicated cases, no helper layers. The test is as long as it needs to be to cover the checklist below. It must also run fast: the complete suite runs in every future session (see "TESTS" in `md/SPEC_CPP_API.md`).

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
  - The exception types follow "Exception handling"; `TypeError` is raised when pybind11 rejects the call itself (missing mandatory argument, unknown keyword, `None` for a mandatory array, an object that cannot be converted to the declared type).
  - Assert the specific type. Do not use a catch-all tuple of exception types (some older tests do); if the type is not what this list predicts, that is a finding to report.

## Default coverage
The plan or spec may narrow or extend this list.
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
