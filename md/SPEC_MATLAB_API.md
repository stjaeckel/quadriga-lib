You are implementing a lightweight API wrapper for MATLAB / Octave MEX to call a function of my C++ radio channel modelling library (quadriga-lib). You will receive the documentation block of the C++ function (or its name). Your task is to add the MEX wrapper in `api_mex/` and a test in `tests/mex_tests/`, following the workflow below. The wrapper converts arguments, shifts indices, allocates outputs and calls the C++ function; it should be user-friendly for MATLAB users while keeping the performance of the underlying C++ library. The C++ side is covered by `md/SPEC_CPP_API.md`, the Python wrappers by `md/SPEC_PYTHON_API.md`.

# CPP API DOCUMENTATION BLOCK FORMAT
- Each function has a 1-line summary, a description as bullet list, a `## Declaration:` block, and `## Inputs:` / `## Outputs:` / `## Returns:` sections (singular when there is exactly one entry); see `md/SPEC_CPP_API.md` for the full format.
- Arguments with a default value or accepting `nullptr` are marked `*(optional)*`; the default is visible in the Declaration.
- Mandatory arguments are passed by reference, optional ones by pointer (default `nullptr`). Older functions pass mandatory arguments by pointer as well.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated.
- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`; I/O failures a `std::runtime_error`.
- Index conventions: 0-based unless the field is explicitly called "1-based" (which applies to `obj_ind`, `mtl_ind`, `fbs_ind`, `sbs_ind`, and QDANT `id`).
- Units: angles in radians (degrees only where stated, e.g. `*_deg`, `*_3dB`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).

# REPOSITORY LAYOUT AND BUILD
- `api_mex/<function_name>.cpp` — MEX wrappers, one file per MATLAB function. CMake globs `api_mex/*.cpp`, so a new file needs no registration in the build system.
- `src/mex_helper_functions.hpp` (generic conversions) and `src/mex_quadriga_lib_functions.hpp` (arrayant / channel structs, `std::any` datasets) — helper functions. Read them when a signature is unclear; the lists in this document are a summary.
- `include/*.hpp` — C++ declarations; `src/*.cpp` — C++ implementations with their documentation blocks (`/*!MD ... MD!*/`).
- `tests/mex_tests/test_<function_name>.m` — MOxUnit tests, one file per wrapper; `tests/quadriga_lib_mex_tests.m` collects and runs all `tests/mex_tests/test*.m` files, so a new test file needs no registration.
- `+quadriga_lib/` — the MATLAB package populated by the build: the compiled MEX files and one `.m` help file per function.
- Generated overviews (read-only, produced by `make documentation`, never edit them; they may lag behind the sources until the documentation is rebuilt):
  - `html_docu/quadriga_lib_api_cpp.md` — all C++ functions
  - `html_docu/quadriga_lib_api_mex.md` — all MATLAB functions
  - Both start with general usage notes and a "Function Index" table with the columns Function, Section and Line. The Line column is the line number at which the entry starts: read the index first, then extract only the entries you need by line range instead of reading the whole file.
- The text in the `/*!MD ... MD!*/` block of a wrapper is reused for 3 purposes: (1) merged into the Markdown overview, (2) compiled to HTML, (3) converted into the `.m` help file of the function that provides the MATLAB `help` text (`tools/extract_matlab_comments.py`, which strips backticks and asterisks and prefixes each line with `%`). The delimiters `/*!MD`, `MD!*/`, `/*!SECTION`, `SECTION!*/` are matched literally, each on its own line.
- Commands (Linux, from the repository root):
  - `make matlab=ON` (or `octave=ON`) — build the library and the MEX files into `+quadriga_lib/`
  - `make test matlab=ON` (or `octave=ON`) — additionally run the Catch2 suite and the complete MEX test suite
  - A single test file is run from MATLAB / Octave by putting the repository root (for `+quadriga_lib`), `tests/mex_tests` and MOxUnit (`external/MOxUnit-master/MOxUnit`, then `moxunit_set_path`) on the path and calling the test function by name, e.g. `test_icosphere`. It returns silently on success and raises an error on the first failed assertion.
- Wrappers and tests must work in both MATLAB and Octave.

# WORKFLOW
The order differs from the C++ workflow: the wrapper is written first, the test second. The physics is already verified by the Catch2 suite of the C++ function; the MEX test exists to exercise the wrapper. The work normally happens in one session.

There are two review points at which you stop, report, and wait for my answer before continuing:
- **Review point A** (end of Phase 1) — only if there are open questions about the MATLAB interface.
- **Review point B** (end of Phase 4) — always: the final result.

## Phase 0 — What you receive
- The documentation block of the C++ function, or its name (the block is then in `src/*.cpp` and in `html_docu/quadriga_lib_api_cpp.md`).
- Optionally the section name and remarks on the MATLAB interface (argument names, order, what to leave out).

## Phase 1 — Read the contract and fix the MATLAB interface
- Read the C++ documentation block and the declaration in `include/`.
- Read one or two existing wrappers that use the same pattern and follow them, e.g.:
  - `icosphere.cpp` — scalar inputs, outputs of known size, optional output pointers
  - `obj_file_read.cpp` — outputs of runtime size, index shifts, cell arrays, map-to-struct
  - `triangle_mesh_aabb.cpp` — optional 1-based index input converted in place
  - `get_channels_spherical.cpp` — arrayant struct inputs, 3-element position vectors
  - `get_channels_multifreq.cpp` — multi-frequency arrayant struct arrays, 4D outputs
  - `arrayant_rotate_pattern.cpp` — arrayant struct in / struct out, in-place modification
  - `hdf5_write_channel.cpp`, `hdf5_read_channel.cpp` — channel structs, unstructured data
- Settle before writing code: the MATLAB usage line (order and names of inputs and outputs, in/out splits); which inputs are optional and their defaults; which arguments are indices and their base on the C++ side; the size of every output and whether the wrapper can know it before the call; the helper for every argument; the section name.
- **Review point A:** if anything in this list cannot be settled from the C++ documentation and this spec (a C++ type without helper, an unclear index base, an output order that is not obvious, logic that would have to live in the wrapper), list the open questions with your proposed answer for each, and wait. If everything is clear, state the usage line and continue.

## Phase 2 — Wrapper
- Write `api_mex/<function_name>.cpp` following the "IMPLEMENTATION GUIDELINES".
- Build. The wrapper must compile without warnings.

## Phase 3 — Test
- Write `tests/mex_tests/test_<function_name>.m` (see "TESTS"). The test exercises the wrapper: every argument reaches the right place, indices are shifted correctly, optional arguments and defaults work, outputs have the right size and class, errors arrive in MATLAB.
- Run the test. It is expected to pass, since the C++ function is already verified. If it fails, the cause is in the wrapper or in the test's expectation; find out which before changing either. If the failure points to the C++ function itself, stop and report it — do not work around it in the wrapper.

## Phase 4 — Finish
- The new test passes; the complete MEX test suite passes.
- The documentation block of the wrapper is complete and matches what the wrapper does.
- **Review point B:** report the final state: the MATLAB usage line, the test results with the actual output of the run (state which of MATLAB / Octave was used), compiler warnings if any, every deviation from the C++ interface and why, and anything that was not tested. Do not report a test as passing that was not run.
- The Python wrapper is a separate task and is only written on request.

# IMPLEMENTATION GUIDELINES
- The mex function should be named according to the C++ function it wraps, so quadriga_lib::function_name should be wrapped by a mex function named "function_name.cpp". Exceptions (e.g. wrapping a class member function, or one MATLAB function dispatching to several C++ functions) are decided case by case.
- All wrappers are called from the +quadriga_lib package which will be populated by the build system, so the user calls quadriga_lib.function_name from MATLAB, which maps to function_name.cpp in the source code. 
- The mexFunction in function_name.cpp should call the underlying C++ function quadriga_lib::function_name.
- Keep the wrapper thin: argument conversion, index shifts, output allocation and the C++ call. Anything that changes numerical results (unit conversions, defaults that differ from C++, loops over several C++ calls, mode dispatch) belongs in the C++ library; if a wrapper seems to need it, ask first.
- All errors map to "quadriga_lib:CPPerror", `mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Error text.");`
- C++ functions may offer dtype = float / double specialization. The MATLAB mex wrapper only uses the double version, so you can hardcode the dtype to double in the call to the C++ function. (`fast_sincos`, `interp` and `qrt_file_parse` are existing exceptions; do not copy this for new wrappers.)
- Integer outputs keep their C++ type (e.g. `arma::u32_vec` is returned as uint32, `arma::uvec` as uint64); state the type in the documentation.
- Armadillo types map to their c++ counterparts, e.g. arma::uword = unsigned long long (this is enforced at compile time by static_asserts).
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two MATLAB names with _in / _out suffixes.

## Index conversion
- C++ indices are 0-based unless stated otherwise in the documentation, while MATLAB indices are 1-based. The mex wrapper should handle this conversion when passing indices between MATLAB and C++.
  - If CPP expects 0-based indices, the MATLAB user should provide 1-based indices, and the mex wrapper should convert them to 0-based before passing to C++.
  - If CPP expects 1-based indices, the MATLAB user should provide 1-based indices (no conversion needed).
  - If CPP outputs 0-based indices, the mex wrapper should convert them to 1-based before returning to MATLAB.
  - If CPP outputs 1-based indices, the mex wrapper should return them as-is to MATLAB.
- Index inputs that get converted MUST be read with `copy = true`. Without the copy, the Armadillo object may borrow the memory of the MATLAB input, and editing it in place modifies the caller's variable (undefined behavior).
- Always check for zeros before subtracting, otherwise unsigned indices wrap around:
```
// Vector index input, 1-based (MATLAB) -> 0-based (C++), empty = C++ default
arma::uvec i_element = (nrhs < 3) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[2], true);
if (arma::any(i_element == 0))
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'i_element' cannot be 0 (1-based index).");
i_element -= 1;

// Scalar index input
arma::uword i_orig = (nrhs < 4) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[3], "i_orig", 1);
if (i_orig == 0)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'i_orig' cannot be 0 (1-based index).");
i_orig -= 1;
```
- Index outputs are shifted after the C++ call and before they are handed to MATLAB:
```
if (nlhs > 2)
{
    face_ind += 1; // 0-based (C++) -> 1-based (MATLAB)
    plhs[2] = qd_mex_copy2matlab(&face_ind);
}
```
- Indices that are 1-based in C++ pass through unchanged in both directions; they often use 0 as "none", so no zero-check applies. Add a short comment in the code stating that no shift is intended.
- State "1-based" in the MATLAB documentation of every index argument.

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
#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"
```
- `#include "mex_quadriga_lib_functions.hpp"` is only needed in special cases, i.e. when the wrapper converts arrayant structs, channel structs or `std::any` datasets (see below). It pulls in the three headers above.

## Helper functions signatures
- Quadriga-Lib uses Armadillo for data structures, Armadillo headers are pulled in by quadriga_lib.hpp, so you can use Armadillo types directly in the mex wrapper
- Conversion from MEX to Armadillo can be done by calling the helper functions in mex_helper_functions.hpp. 
- `dtype` is one of: float, double, int, unsigned, long long, unsigned long long (plus bool for scalars)

```
// ---------- Inputs: MATLAB -> C++ ----------

// Scalar, casts to dtype; returns default_value for empty input "[]"; var_name is used in error messages
dtype qd_mex_get_scalar(const mxArray *input, std::string var_name = "", dtype default_value = dtype(NAN));

// Char array -> std::string; returns default_value for empty input
std::string qd_mex_get_string(const mxArray *input, std::string default_value = "");

// Cell array of strings (or a single char array) -> vector of strings; empty input -> empty vector
std::vector<std::string> qd_mex_get_strings(const mxArray *input);

// Numeric arrays. Borrow the MATLAB memory (zero-copy, read-only!) when the MATLAB class matches
// dtype and copy == false; otherwise allocate and type-cast. Empty input -> empty object.
// NOTE: the second argument is `bool copy`, NOT a variable name. Passing a string literal compiles
//       and silently means copy = true.
arma::Col<dtype> qd_mex_get_Col(const mxArray *input, bool copy = false);  // 1D, any shape is vectorized
arma::Mat<dtype> qd_mex_get_Mat(const mxArray *input, bool copy = false);  // 2D
arma::Cube<dtype> qd_mex_get_Cube(const mxArray *input, bool copy = false); // 3D
std::vector<arma::Cube<dtype>> qd_mex_get_CubeVec(const mxArray *input, bool copy = false); // 4D

// Always copy + type-cast; var_name is used in error messages
// n_elem != 0 enforces the number of elements (e.g. 3 for a position vector)
arma::Col<dtype> qd_mex_typecast_Col(const mxArray *input, std::string var_name = "", size_t n_elem = 0);
arma::Mat<dtype> qd_mex_typecast_Mat(const mxArray *input, std::string var_name = "");
arma::Cube<dtype> qd_mex_typecast_Cube(const mxArray *input, std::string var_name = "");
std::vector<arma::Cube<dtype>> qd_mex_typecast_CubeVec(const mxArray *input, std::string var_name = "");

// Regular ND array -> std::vector of Armadillo objects, split along dimension vec_dim (0-based)
// Always copies and casts to dtype; empty input -> empty vector
std::vector<arma::Col<dtype>> qd_mex_matlab2vector_Col(const mxArray *input, size_t vec_dim);
std::vector<arma::Mat<dtype>> qd_mex_matlab2vector_Mat(const mxArray *input, size_t vec_dim);
std::vector<arma::Cube<dtype>> qd_mex_matlab2vector_Cube(const mxArray *input, size_t vec_dim);

// Numeric or logical array -> std::vector<bool> (nonzero = true)
std::vector<bool> qd_mex_matlab2vector_Bool(const mxArray *input);

// 1x1 struct of numeric vectors -> map (one key per field, all fields must have equal length)
// Non-struct or empty input -> empty map
std::unordered_map<std::string, std::vector<dtype>> qd_mex_struct2map(const mxArray *strct);

// Any numeric MATLAB array -> std::any holding a scalar / arma::Col / Mat / Cube of the MATLAB class
// (no cast, borrows memory unless create_copy). Strings are not handled, use qd_mex_get_string.
std::any qd_mex_anycast(const mxArray *input, std::string var_name = "", bool create_copy = false);

// ---------- Outputs: C++ -> MATLAB (copy) ----------

mxArray *qd_mex_copy2matlab(const dtype *input); // Scalar
mxArray *qd_mex_copy2matlab(const arma::Row<dtype> *input); // Row Vector

mxArray *qd_mex_copy2matlab(const arma::Col<dtype> *input, // Column Vector
                            bool transpose = false,        // Transpose output (return a row vector)
                            size_t ns = 0,                 // Number of elements in output
                            const size_t *is = nullptr);   // List of elements to copy, 0-based

mxArray *qd_mex_copy2matlab(const arma::Mat<dtype> *input, // Matrix
                            size_t ns = 0,                 // Number of columns in output
                            const size_t *is = nullptr);   // List of columns to copy, 0-based

mxArray *qd_mex_copy2matlab(const arma::Cube<dtype> *input, // Cube
                            size_t ns = 0,                  // Number of columns in output
                            const size_t *is = nullptr);    // List of columns to copy, 0-based

mxArray *qd_mex_copy2matlab(const std::vector<std::string> *strings); // Cell array of strings [n, 1]
mxArray *qd_mex_copy2matlab(const std::vector<bool> *bools);          // Logical column vector

// std::vector of Armadillo objects -> ND array with one extra dimension, zero-padded to the largest
// element; empty vector -> empty matrix. Ignore ns, is, padding.
mxArray *qd_mex_vector2matlab(const std::vector<arma::Col<dtype>> *input,
                              size_t ns = 0, const size_t *is = nullptr, dtype padding = (dtype)0);
mxArray *qd_mex_vector2matlab(const std::vector<arma::Mat<dtype>> *input,
                              size_t ns = 0, const size_t *is = nullptr, dtype padding = (dtype)0);
mxArray *qd_mex_vector2matlab(const std::vector<arma::Cube<dtype>> *input,
                              size_t ns = 0, const size_t *is = nullptr, dtype padding = (dtype)0);

// Map -> 1x1 struct, one column-vector field per key; field_order fixes the order of the fields
mxArray *qd_mex_map2struct(const std::unordered_map<std::string, std::vector<dtype>> &map,
                           const std::vector<std::string> &field_order = {});

// ---------- Outputs: allocate in MATLAB, alias in Armadillo (zero-copy) ----------

mxArray *qd_mex_init_output(arma::Row<dtype> *input, size_t n_elem); // 1D row
mxArray *qd_mex_init_output(arma::Col<dtype> *input, size_t n_elem, bool transpose = false); // 1D col 
mxArray *qd_mex_init_output(arma::Mat<dtype> *input, size_t n_rows, size_t n_cols); // 2D
mxArray *qd_mex_init_output(arma::Cube<dtype> *input, size_t n_rows, size_t n_cols, size_t n_slices);  // 3D
mxArray *qd_mex_init_output(std::vector<arma::Cube<dtype>> *input, size_t n_rows, size_t n_cols, size_t n_slices, size_t n_cubes); // 4D

// ---------- Structs ----------

mxArray *qd_mex_make_struct(const std::vector<std::string> &fields, size_t N = 1); // 1xN struct array; no fields -> 0x0 struct
void qd_mex_set_field(mxArray *strct, const std::string &field, mxArray *data, size_t n = 0); // n = element index, 0-based
bool qd_mex_has_field(const mxArray *strct, const std::string &field);
mxArray *qd_mex_get_field(const mxArray *strct, const std::string &field, size_t n = 0); // errors if field is missing

// ---------- mex_quadriga_lib_functions.hpp ----------

quadriga_lib::arrayant<double> qd_mex_struct2arrayant(const mxArray *input, bool validate = false, bool copy = false);
std::vector<quadriga_lib::arrayant<double>> qd_mex_struct2arrayant_multi(const mxArray *input, bool validate = false, bool copy = false);
mxArray *qd_mex_arrayant2struct(const quadriga_lib::arrayant<double> &ant);
mxArray *qd_mex_arrayant2struct_multi(const std::vector<quadriga_lib::arrayant<double>> &ant);

std::vector<quadriga_lib::channel<double>> qd_mex_struct2channel(const mxArray *input, bool validate = false, bool copy = false);
mxArray *qd_mex_channel2struct(const std::vector<quadriga_lib::channel<double>> &chan, bool validate = false);

// std::any (string, scalar, arma::Col / Row / Mat / Cube) -> MATLAB array of the same type; no value -> []
mxArray *qd_mex_any2matlab(const std::any &dset);
```

## Type mapping
| C++ type | MATLAB type | Input helper | Output helper |
| --- | --- | --- | --- |
| scalar (`dtype`, `bool`, `arma::uword`, ...) | scalar | `qd_mex_get_scalar` | `qd_mex_copy2matlab` |
| `std::string` | char array | `qd_mex_get_string` | `mxCreateString(str.c_str())` |
| `std::vector<std::string>` | cell array of char arrays, `[n, 1]` | `qd_mex_get_strings` | `qd_mex_copy2matlab` |
| `arma::Col` | column vector (any shape accepted as input) | `qd_mex_get_Col` | `qd_mex_init_output` / `qd_mex_copy2matlab` |
| `arma::Row` | row vector | `qd_mex_get_Col` + conversion | `qd_mex_init_output` / `qd_mex_copy2matlab` |
| `arma::Mat` | 2D array | `qd_mex_get_Mat` | `qd_mex_init_output` / `qd_mex_copy2matlab` |
| `arma::Cube` | 3D array | `qd_mex_get_Cube` | `qd_mex_init_output` / `qd_mex_copy2matlab` |
| `std::vector<arma::Cube>`, regular shape | 4D array | `qd_mex_get_CubeVec` | `qd_mex_init_output` |
| `std::vector<arma::Col/Mat/Cube>`, ragged | ND array, zero-padded, vector on last dim | `qd_mex_matlab2vector_*` | `qd_mex_vector2matlab` |
| `std::vector<bool>` | logical column vector | `qd_mex_matlab2vector_Bool` | `qd_mex_copy2matlab` |
| `std::unordered_map<std::string, std::vector<dtype>>` | 1x1 struct of column vectors | `qd_mex_struct2map` | `qd_mex_map2struct` |
| `std::any` | numeric array or char array, type preserved | `qd_mex_anycast` (`qd_mex_get_string` for char) | `qd_mex_any2matlab` |
| `quadriga_lib::arrayant<dtype>` | 1x1 struct | `qd_mex_struct2arrayant` | `qd_mex_arrayant2struct` |
| `std::vector<quadriga_lib::arrayant<dtype>>` | 1xN struct array (one entry per frequency) | `qd_mex_struct2arrayant_multi` | `qd_mex_arrayant2struct_multi` |
| `std::vector<quadriga_lib::channel<dtype>>` | 1xN struct array | `qd_mex_struct2channel` | `qd_mex_channel2struct` |
| `arma::cx_mat`, `arma::cx_vec`, ... | not supported | ask before wrapping | ask before wrapping |

- Complex data is always passed as separate real / imaginary arrays with `_re` / `_im` suffixes.
- There is no helper for C++ types that are not listed (e.g. `std::vector<uint8_t>`, nested vectors); ask before adding raw `mx*` code for them.

## Arrayant struct
- A `quadriga_lib::arrayant<double>` is represented in MATLAB as a struct with the fields below. Requires `mex_quadriga_lib_functions.hpp`.

| MATLAB field | C++ member | Size | Required as input |
| --- | --- | --- | --- |
| `e_theta_re` | `e_theta_re` | `[n_elevation, n_azimuth, n_elements]` | yes |
| `e_theta_im` | `e_theta_im` | `[n_elevation, n_azimuth, n_elements]` | yes |
| `e_phi_re` | `e_phi_re` | `[n_elevation, n_azimuth, n_elements]` | yes |
| `e_phi_im` | `e_phi_im` | `[n_elevation, n_azimuth, n_elements]` | yes |
| `azimuth_grid` | `azimuth_grid` | `[n_azimuth]`; returned as row vector | yes |
| `elevation_grid` | `elevation_grid` | `[n_elevation]`; returned as row vector | yes |
| `element_pos` | `element_pos` | `[3, n_elements]` | no; default: zeros |
| `coupling_re` | `coupling_re` | `[n_elements, n_ports]` | no; default: identity |
| `coupling_im` | `coupling_im` | `[n_elements, n_ports]` | no; default: zeros |
| `center_freq` | `center_frequency` | scalar | no; default: 299792458 |
| `name` | `name` | string | no |

- Note the name difference: the MATLAB field is `center_freq`, the C++ member is `center_frequency`.
- A 1x1 struct is a single-frequency arrayant; a 1xN struct array is a frequency-dependent (multi-frequency) model and maps to `std::vector<quadriga_lib::arrayant<double>>`.
- Reading: `qd_mex_struct2arrayant(prhs[i], validate, copy)` or `qd_mex_struct2arrayant_multi(...)`.
  - `validate = true` runs the arrayant validation and throws on inconsistent data. Use `false` when the C++ function validates its inputs anyway.
  - `copy = false` borrows the MATLAB memory (read-only). Use `copy = true` whenever the arrayant object is modified after reading (e.g. in-place member functions such as `rotate_pattern`, `copy_element`).
- Writing: `plhs[0] = qd_mex_arrayant2struct(ant);` or `qd_mex_arrayant2struct_multi(ant_vec);`
- C++ functions that take `const arrayant<dtype> *` get the address of the local object, e.g. `&ant_tx`.
- In the documentation, describe arrayant arguments as: "Struct containing the arrayant data; field layout as documented in [[arrayant_generate]]" and add "; a struct array may contain a frequency-dependent model" where multi-frequency input is supported. Do not repeat the field table.
- Some existing arrayant wrappers additionally accept the fields as separate inputs or return them as separate outputs. This is legacy behavior; new wrappers use the struct only, unless requested otherwise.

## Channel struct
- A `quadriga_lib::channel<double>` is represented in MATLAB as a struct with the fields below; a `std::vector` of channels is a 1xN struct array. Requires `mex_quadriga_lib_functions.hpp`.
- Members that are a `std::vector` over snapshots in C++ are regular arrays in MATLAB with the snapshot on the last dimension (zero-padded when the number of paths differs between snapshots).

| MATLAB field | C++ member | MATLAB size |
| --- | --- | --- |
| `name` | `name` | string |
| `tx_position` | `tx_pos` | `[3, 1]` or `[3, n_snap]` |
| `rx_position` | `rx_pos` | `[3, 1]` or `[3, n_snap]` |
| `tx_orientation` | `tx_orientation` | `[3, 1]` or `[3, n_snap]` |
| `rx_orientation` | `rx_orientation` | `[3, 1]` or `[3, n_snap]` |
| `coeff_re`, `coeff_im`, `delay` | same | `[n_rx, n_tx, n_path, n_snap]` |
| `path_gain`, `path_length` | same | `[n_path, n_snap]` |
| `path_polarization` | `path_polarization` | `[8, n_path, n_snap]` |
| `path_angles` | `path_angles` | `[n_path, 4, n_snap]` |
| `fbs_pos` | `path_fbs_pos` | `[3, n_path, n_snap]` |
| `lbs_pos` | `path_lbs_pos` | `[3, n_path, n_snap]` |
| `no_interact` | `no_interact` | `[n_path, n_snap]`; uint32 |
| `interact_coord` | `interact_coord` | `[3, max(sum(no_interact)), n_snap]` |
| `center_frequency` | `center_frequency` | scalar or `[n_snap]` |
| `initial_position` | `initial_position` | scalar; int32 |

- Note the name differences: `tx_position` / `rx_position` and `fbs_pos` / `lbs_pos` in MATLAB vs. `tx_pos` / `rx_pos` and `path_fbs_pos` / `path_lbs_pos` in C++.
- Reading: `qd_mex_struct2channel(prhs[i], validate, copy)` always returns a `std::vector`; all fields are optional, missing fields leave the member empty. Wrap the call in `CALL_QD` when `validate = true`.
- Writing: `plhs[0] = qd_mex_channel2struct(chan_vec);` Only populated members become fields (decided by the first channel); an empty vector returns a `0x0` struct.
- Unstructured data (`par_names` / `par_data`, a list of `std::any`) is NOT part of the channel struct. It is exchanged through a separate `par` struct, converted with `qd_mex_anycast` / `qd_mex_any2matlab`; see `hdf5_write_channel.cpp` and `hdf5_read_channel.cpp`.
- In the documentation, refer to [[hdf5_read_channel]] for the field layout instead of repeating the table.

## CALL_QD Macro
- Use the CALL_QD macro to call the C++ function and catch any exceptions

#define CALL_QD(expr)                                              \
    do                                                             \
    {                                                              \
        try                                                        \
        {                                                          \
            expr;                                                  \
        }                                                          \
        catch (const std::exception &ex)                           \
        {                                                          \
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", ex.what()); \
        }                                                          \
    } while (0)

## Section statement
- Functions are grouped into sections, which are marked by a comment block like:
- This should be provided with the documentation block, but if not, ask for it. 
- It is important to include it in the mex file, as it is used for grouping functions in the generated documentation.
- Section names currently in use: "Array antenna functions", "Channel functions", "Channel generation functions", "Channel statistics", "Geometry-based stochastic model", "Math functions", "Miscellaneous / Tools", "Site-specific simulation tools", "Sum-of-sinusoids random number generator". The name must match exactly to end up in the same group.
```
/*!SECTION
Section name from C++ documentation block
SECTION!*/
```

## Documentation block
- The documentation block should be included as a comment block immediately before the function definition.
- Block starts with `/*!MD` and ends with `MD!*/`
- First line after `/*!MD` is the function name, followed by a one-line summary. 
- The function name must be converted to UPPERCASE, e.g. `acdf` becomes `ACDF`.
- The rest of the block is the detailed documentation, which must be adapted to the MATLAB context
- It should closely match the C++ documentation block, with MATLAB-specific adjustments
- 1-line summary should be identical to the C++ documentation block
- The description follows the summary after one empty line as a bullet list, WITHOUT a "## Description:" heading
- Description should be adapted to MATLAB users, e.g. by removing C++-specific details and adding MATLAB-specific usage notes if needed. 
- It should be in bullet point format, with one concise fact per bullet
- Never use "*" or "|" in formulas or any other text. These are reserved for markdown formatting. Instead use "·" for multiplication, e.g. "Rz·Ry·Rx" instead of "Rz*Ry*Rx" and abs() for absolute values.
- Lines should not exceed 100 characters (soft limit): the text is also converted to the `.m` help file of the function, and the limit keeps the MATLAB `help` output readable. Split longer lines and indent the continued lines by 2 spaces. Table rows and usage lines may exceed the limit when splitting would break them.
- You can omit any dtype specifications in the documentation block, as they are not relevant for the MATLAB user. All C++ functions that have a dtype specialization are wrapped with the double version in MATLAB.
- Instead of the "## Declaration:" section, use a "## Usage:" section that includes the function signature as it should be called from MATLAB, such as:

    ## Usage:
    ```
    [ output1, output2 ] = quadriga_lib.function_name( input1, input2, optional_input3 );
    ```

- Split the usage block "..." into multiple lines if it exceeds 100 characters, and indent the continued lines by 4 spaces.
- Inputs and outputs should be described in separate sections "## Inputs:" and "## Outputs:", with the same one-liner style as in the C++ documentation block. Use the singular "## Input:" / "## Output:" only when there is exactly one input / output. Omit the section if there are none.
- Argument format: ``- **`name`** — Description; further facts separated by semicolons; `[size]`; default: X``
- Optional inputs (C++: default value or `const type *name = nullptr`) are also optional in MATLAB: the user can omit them or pass an empty array `[]` to get the default behavior.
  - If the input has a default value, add "; default: X" to the description. This implies that the input is optional; do NOT add "(optional)".
  - Only if an optional input has no default value that can be stated, mark it with `*(optional)*` directly after the argument name.
  - Outputs are never marked as optional (MATLAB outputs can always be omitted).
- Datatypes must only be specified when not `double`
- Unit must only be specified when not SI (meters, radians, Hz, linear scale), e.g. "angles in degrees", "power in dB"
- C++ returns get merged into the output arguments section
- An optional "## See also:" section at the end lists related functions as `- [[function_name]] (short reason)`; `[[...]]` becomes a link in the HTML documentation.

## Function body structure
- Alway use this function signature for the mex function: 
  `void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])`

## Validate argument counts
- The first step is to validate the number of input and output arguments against the expected counts like:
```
// Validate argument counts
if (nrhs < 1 || nrhs > 2)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
if (nlhs > 3)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
```

## Input conversion
- the header mex_helper_functions.hpp provides various conversion functions to convert between MATLAB mxArray and C++ types. Use these functions to convert the input arguments from MATLAB to the appropriate C++ types before calling the underlying C++ function:

### numeric array types
- Required numeric array types are converted by one of these functions:
```
const auto data = qd_mex_get_Col<dtype>(prhs[0]);
const auto data = qd_mex_get_Mat<dtype>(prhs[0]);
const auto data = qd_mex_get_Cube<dtype>(prhs[0]);
```
- Always assume `copy = false` for input arguments and add const qualifier. The only exception are inputs that the wrapper modifies (e.g. index conversion, see above): these use `copy = true` and are not const.
- Adjust the name to match the c++ variable name, and the dtype to match the expected type in the C++ function (e.g. double)
- dtype can be one of: float, double, int, unsigned, long long, unsigned long long
- Fixed-length vectors (e.g. positions, orientations) are read with a length check:
```
const auto tx_pos = qd_mex_typecast_Col<double>(prhs[7], "tx_pos", 3);
```

### 4D numeric arrays as std::vector<arma::Cube>
- Strict 4D inputs `[d1, d2, d3, d4]` map to `std::vector<arma::Cube<dtype>>` of length `d4`,
  each cube of shape `(d1, d2, d3)`. Use:
```
const auto data = qd_mex_get_CubeVec<dtype>(prhs[0]);
```
- Supports zero-copy aliasing when the MATLAB type matches `dtype` and `copy = false` (default);
  otherwise type-casts into freshly allocated cubes.
- dtype can be one of: float, double, int, unsigned, long long, unsigned long long.
- Use this — NOT `qd_mex_matlab2vector_Cube` — when the C++ documentation specifies a regular
  4D shape like `[d1, d2, d3, d4]`. `qd_mex_matlab2vector_Cube` is for ragged/padded vectors
  with a runtime `vec_dim` and always copies.

### std::vector of Armadillo types
- The C++ function may expect a std::vector of Armadillo types (used for variable-length lists of arrays), e.g. `std::vector<arma::Mat<dtype>>`
- There are specific helper functions for these types, e.g.:
```
const auto data = qd_mex_matlab2vector_Col<dtype>(prhs[0], 1);
const auto data = qd_mex_matlab2vector_Mat<dtype>(prhs[0], 2);
const auto data = qd_mex_matlab2vector_Cube<dtype>(prhs[0], 3);
```
- dtype can be one of: float, double, int, unsigned, long long, unsigned long long
- The second argument indicates the dimension used for std::vector, 0-based; by convention this is the last dimension of the MATLAB array
- The input should be a mxArray, missing valued get zero-padded, and the function returns a std::vector of Armadillo types

### Optional array types
- Optional array types should be wrapped and mapped to an empty Armadillo type if the input is empty, e.g.:
```
const auto bins = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
```
- Adjust this according to the expected type Col/Mat/Cube and dtype

### Strings
- Strings are converted by:
```
const auto name = (nrhs < 3) ? std::string("name") : qd_mex_get_string(prhs[2], "name");
```
- The second argument is the default value (returned for empty input), not a variable name
- Same wrapping rules apply
- Lists of strings (cell arrays): `qd_mex_get_strings(prhs[4])`

### Scalars:
- Scalars are converted by:
```
const auto n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);
```
- Always apply the guard (nrhs < X) before reading prhs[X-1]
- The default values must be set in two places: once when the prhs is not given and once passed to conversion function, as the conversion function also catches empty inputs "[]"
- Always pass a default value to avoid getting NaN when the user provides an empty array for a scalar input, e.g. `[]`
- The name of the scalar must be passed to the conversion function for error messages, e.g. "n_bins" for error messages
- Allowed types for scalars are: float, double, int, unsigned, long long, unsigned long long, bool
    
### Complex arrays
- arma::cx_mat / arma::cx_vec not supported, ask before wrapping


## Output allocation
- declare empty Armadillo variables for all output arguments, e.g.:
```
arma::mat az, el, len;
arma::uvec cluster_idx;
arma::cube path_data;
```
There are 2 ways to allocate output arguments:

### Known output size
- If the output has a known size use the qd_mex_init_output helper
```
if (nlhs > 0)
    plhs[0] = qd_mex_init_output(&az, n_rows, n_cols);
```
- it detects the array and data type from the Armadillo object, calls mxCreateNumericMatrix to allocate MATLAB-owned memory, and sets the pointer of the Armadillo object to the allocated memory. This way, when the C++ function writes to the Armadillo object, it directly writes to the MATLAB output array.
- qd_mex_init_output has overloads for: Col, Row, Mat, Cube, std::vector<Cube> (4D output)
- Allowed data types: float, double, int, unsigned, long long, unsigned long long
- For qd_mex_init_output, the wrapper must know the output size before the C++ call - these should be clear from the C++ documentation block, e.g. if the output is described as [n_bins, n_sets], the wrapper can determine n_bins and n_sets from the input data dimensions and allocate accordingly.
- The memory is zero-initialized and the size must match exactly what the C++ function produces: if C++ resizes the container, the result is written to new memory and the MATLAB output stays zero.
- For 4D outputs `[n_rows, n_cols, n_slices, n_cubes]` use the std::vector<Cube> overload:
```
std::vector<arma::cube> paths;
if (nlhs > 0)
    plhs[0] = qd_mex_init_output(&paths, n_rows, n_cols, n_slices, n_cubes);
```
- Each cube in the vector aliases its slab of the MATLAB-owned 4D buffer (zero-copy). Do not resize the vector or any of its cubes after this call.

### Unknown output size
- Do not pre-allocate the output variable, but store data locally, usually it is sufficient to pass the empty Armadillo object to the C++ function, which will resize and populate it as needed
- Also use for cases where c++ function passed data as a return value.
- After the C++ function call, copy the results to MATLAB memory using the qd_mex_copy2matlab helper, e.g.:
```    
if (nlhs > 0)
    plhs[0] = qd_mex_copy2matlab(&az);
```
- Allowed data types: float, double, int, unsigned, long long, unsigned long long
- Supports armadillo types Col, Row, Mat, Cube, scalar values
- Supports std::vector<std::string> *strings - returned as cell array of strings in MATLAB
- Supports std::vector<bool> *bools - returned as logical array (Col-vector) in MATLAB
- prefer init_output when size is known by the wrapper, fall back to copy2matlab when size depends on the C++ call's runtime behavior

### std::vector of Armadillo types
- If the C++ function returns a std::vector of Armadillo types, use the following helper functions to copy them to MATLAB:
```
if (nlhs > 0)
    plhs[0] = qd_mex_vector2matlab(&std_vector_or_arma_object);
```
- Supported are: std::vector of arma::Col, arma::Mat, arma::Cube
- Optional inputs ns, is should be ignored
- Data is zero-padded to the maximum size in the vector, so that the output is a regular ND array e.g. 3D for std::vector<arma::Mat> 
- Use this for unknown/ragged sizes determined by the C++ call. For a regular 4D output with known dimensions, prefer the `qd_mex_init_output(std::vector<arma::Cube>*, ...)` overload — it's zero-copy.

### Functions without outputs
- If the C++ function returns nothing, the wrapper has no outputs: check `nlhs > 0` as error and omit the "## Outputs:" section.

## Pointer wrapper
- The c++ function might accept `nullptr` for optional inputs and outputs
- Generally, this avoids unnecessary allocations and copying when the user does not need the output or wants to use default values for the input. Also it might skip unnecessary computations in the C++ function when the output is not needed.
- Wrap all OPTIONAL c++ inputs and outputs to nullptr if the user does not provide them, e.g.:
```
arma::uvec *p_cluster_idx = (nlhs > 2) ? &cluster_idx : nullptr; // OR
arma::uvec *p_cluster_idx = cluster_idx.empty() ? nullptr : &cluster_idx;
```
- The cluster_idx.empty() variant is preferred, but it MUST NOT be used if the C++ expects an empty argument and resizes it depending on runtime conditions (e.g. reading from a file), as this would lead to unintended nullptr being passed. In this case, the (nlhs > X) guard should be used instead, which relies on the user explicitly providing an output argument when they want to receive the output.
- In short: outputs allocated by `qd_mex_init_output` use the `.empty()` variant; outputs returned by `qd_mex_copy2matlab` use the `(nlhs > X)` variant.
- C++ arguments that are passed by reference (mandatory inputs and outputs in the current C++ convention) get the Armadillo object itself, not its address. For a mandatory output by reference, pre-allocate with `qd_mex_init_output` only if the user requested it; otherwise pass the local, empty object.
- If c++ does not has the nullptr option for optional arguments or arguments are passed as reference (= c++ mandatory outputs), then simply pass the empty Armadillo object, C++ will populate it as needed, and the wrapper can discard it if the user did not request the output (i.e. simply skip the qd_mex_init_output step for that output argument)

## Call the C++ function
- Now call the underlying C++ function with the converted input arguments and pointer-wrapped optional arguments
- Use the CALL_QD macro to call the function and catch any exceptions, e.g.:
```
CALL_QD(quadriga_lib::function_name(arg1, arg2, optional_arg3, optional_output4));
```
- After the C++ function call, copy all non-preallocated results to MATLAB memory as stated above

# TESTS

## Scope
- The MEX test verifies the **wrapper**, not the physics. The numerical behavior of the C++ function is pinned by its Catch2 suite and is not re-derived here: no oracles for the physics, no sweeps over the parameter space.
- What the test must show:
  - **Data lands where it should**: every input reaches the C++ argument it belongs to, every output comes back in the documented position, with the documented size, orientation (row / column) and class.
  - **Indices are converted correctly**: 1-based in, 1-based out, pass-through indices unchanged.
  - **Optional arguments and defaults work**: omitted, passed as `[]`, and passed explicitly.
  - **Output handling works**: 0 outputs up to the maximum number of outputs.
  - **Errors are passed correctly**: wrong argument counts, the wrapper's own checks, and exceptions thrown by the C++ function arrive as MATLAB errors with the right message.
- For this, a few small cases with known results are enough: values that can be written down by hand, or that follow from the input by construction. Use distinctive inputs (different values per element, non-square sizes, non-symmetric index lists) so that a swapped argument, a transposed array or an index off by one changes the result visibly.
- Keep it lean: no duplicated cases, no helper layers. The test is as long as it needs to be to cover the checklist below.

## File and structure
- One file per wrapper: `tests/mex_tests/test_<function_name>.m`. It contains one main function with the same name as the file and no arguments; MOxUnit calls it as one test case:
```
function test_function_name
% MEX interface test for quadriga_lib.function_name.
% Scope: argument and output handling, index conversion, error paths.
% Physics validation lives in the Catch2 suite (test_function_name.cpp).

...

end
```
- Local helper functions go after the `end` of the main function; every function in the file is then closed with `end`.
- Structure the body with `%%` section comments in this order: setup, basic call and values, optional inputs, outputs, type handling, indices, errors.
- Use other library functions for supporting tasks (`quadriga_lib.cube` for a mesh, `quadriga_lib.arrayant_generate` for an antenna, `quadriga_lib.ray_triangle_intersect` for hit indices, ...) instead of building such data by hand; `html_docu/quadriga_lib_api_mex.md` lists what is available.
- Tests that write files use distinct file names in the working directory, delete leftovers at the start and the files at the end.
- The code must run in MATLAB and Octave: use `strfind` (not `contains`), char arrays in single quotes (not `"strings"`), no `arguments` blocks, no functions that exist in only one of the two.

## Assertions
- MOxUnit assertion functions:
  - `assertEqual( a, b )` — exact comparison including size and class; use it for sizes, indices, integer and logical outputs, strings, and for values that must pass through unchanged
  - `assertElementsAlmostEqual( a, b, 'absolute', tol )` — floating-point values
  - `assertTrue( cond )` — everything else, e.g. `assertTrue( isa(out, 'uint32') )`
- Tolerances: `1e-14` to `1e-12` for values that are exact by construction in double, around `1e-6` where single precision is involved. The wrapper does not change values, so a tight tolerance is the norm; a loose one needs a comment.
- Errors are checked by message substring. Define this helper once per test file as a local function:
```
% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
```
  and call it with an anonymous function: `assert_throws( @() quadriga_lib.function_name( a, b ), 'Wrong number of input arguments.' );`
  - An anonymous function requests at most one output. To trigger the "too many outputs" check, add a small local function that requests one output more than the wrapper allows and pass a handle to it:
```
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.function_name( varargin{:} ); %#ok<ASGLU>
end
```
  - Older tests spell the `try` / `catch` block out for every case; use the helper in new tests.

## What every test covers
- **Basic call**: all mandatory inputs, all outputs; check size, class and values of every output.
- **Each optional input**: omitted, passed as `[]` (must give the same result as omitted), and passed with a non-default value that visibly changes the result. This also shows that the default in the wrapper matches the documentation.
- **Outputs**: a call with 0 outputs must not fail; calls with fewer outputs than the maximum give the same values as the full call for the outputs that are requested (the wrapper passes `nullptr` for the others).
- **Argument counts**: one input too many, too few mandatory inputs, one output too many → `'Wrong number of input arguments.'` / `'Wrong number of output arguments.'`.
- **Type handling**: at least one call with non-double inputs (`single(...)`, an integer class for an index or count) gives the same result; outputs have the documented class (double unless stated; integer class for index outputs).
- **Orientation and shape**: a vector input given as row and as column is accepted; vector outputs have the documented orientation; empty optional arrays are accepted.
- **Indices** (if the function has index arguments): use an input in which the index matters and is not symmetric, check the result against the 1-based expectation; index outputs are 1-based (the smallest valid value is 1, not 0); a 0 in a converted index input raises the wrapper's error; pass-through indices (1-based on the C++ side, 0 = none) come back unchanged.
- **Structs** (arrayant, channel, material properties): a struct with only the mandatory fields is accepted; an output struct has the documented fields; a round trip (output of one call as input of the next) works; a non-struct input raises an error.
- **Errors from the wrapper**: every `mexErrMsgIdAndTxt` in the wrapper is reached by one case.
- **Errors from C++**: at least one case in which the C++ function throws (e.g. a wrong number of columns), to show that `CALL_QD` passes the message through; match a characteristic substring of the C++ message.
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

Complete MEX wrapper implementation for the `acdf` function based on the above documentation block, following the implementation guidelines provided:

// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# ACDF
Calculate the empirical averaged cumulative distribution function (CDF)

- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative
  sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from
  each column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins_in` is empty, equally spaced bins spanning the data range are generated

## Usage:
```
[ cdf_per_set, bins_out, cdf_avg, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

## Inputs:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins_in`** *(optional)* — Bin centers; used as-is if non-empty; if omitted or empty, equally
  spaced bins spanning the data range are generated; `[n_bins_in]`
- **`n_bins`** — Number of bins when auto-generating; must be >= 2; ignored when non-empty
  `bins_in` are provided; default: 201

## Outputs:
- **`cdf_per_set`** — Individual CDFs; one per column of data; `[n_bins_out, n_sets]`
- **`bins_out`** — Auto-generated bins; copy of `bins_in` when non-empty `bins_in` are provided;
  `[n_bins_out = n_bins]` or `[n_bins_out = n_bins_in]`
- **`cdf_avg`** — Averaged CDF via quantile-space averaging across data sets; `[n_bins_out]`
- **`mu`** — Mean of the 0.1–0.9 quantiles across data sets; `[9]`
- **`sig`** — Standard deviation of the 0.1–0.9 quantiles across data sets; `[9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat data = qd_mex_get_Mat<double>(prhs[0]);
    const arma::vec bins_in = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1]);
    const arma::uword n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);

    arma::uword n_bins_out = bins_in.empty() ? n_bins : bins_in.n_elem;
    arma::uword n_sets = data.n_cols;

    // Output allocation
    arma::mat cdf_per_set;
    arma::vec cdf_avg, bins_out, mu, sig;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&cdf_per_set, n_bins_out, n_sets);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&cdf_avg, n_bins_out);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&mu, 9);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sig, 9);

    // Special case for bins
    if (!bins_in.empty())
        bins_out = bins_in;

    // Wrap optional pointers
    arma::mat *p_cdf_per_set = cdf_per_set.empty() ? nullptr : &cdf_per_set;
    arma::vec *p_cdf_avg = cdf_avg.empty() ? nullptr : &cdf_avg;
    arma::vec *p_mu = mu.empty() ? nullptr : &mu;
    arma::vec *p_sig = sig.empty() ? nullptr : &sig;

    // Call library function
    CALL_QD(quadriga_lib::acdf<double>(data, &bins_out, p_cdf_per_set, p_cdf_avg, p_mu, p_sig, n_bins));

    // Copy to MATLAB
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&bins_out);
}

Test for the `acdf` wrapper, `tests/mex_tests/test_acdf.m`:

```
function test_acdf
% MEX interface test for quadriga_lib.acdf.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_acdf.cpp).

%% Setup: two data sets with different, known content
data = [ (1:100)', (101:200)' ];

%% Basic call, all outputs
[ cdf_per_set, bins, cdf_avg, mu, sig ] = quadriga_lib.acdf( data );
assertEqual( size(cdf_per_set), [201, 2] );
assertEqual( size(bins), [201, 1] );
assertEqual( size(cdf_avg), [201, 1] );
assertEqual( size(mu), [9, 1] );
assertEqual( size(sig), [9, 1] );
assertTrue( isa(cdf_per_set, 'double') );
assertElementsAlmostEqual( bins([1, end]), [1; 200], 'absolute', 1e-12 );   % bins span the data
assertElementsAlmostEqual( cdf_per_set(end, :), [1, 1], 'absolute', 1e-12 );         % each CDF ends at 1
assertTrue( cdf_per_set(101, 1) > 0.99 && cdf_per_set(100, 2) < 0.01 );                       % columns are not swapped

%% Optional inputs: omitted, empty, explicit
[ cdf_per_set2, bins2 ] = quadriga_lib.acdf( data, [], [] );
assertEqual( cdf_per_set2, cdf_per_set );
assertEqual( bins2, bins );

[ cdf_per_set3, bins3 ] = quadriga_lib.acdf( data, [], 11 );                         % n_bins
assertEqual( size(cdf_per_set3), [11, 2] );
assertEqual( size(bins3), [11, 1] );

bins_in = linspace( 0, 250, 26 );                                           % row vector is accepted
[ cdf_per_set4, bins4 ] = quadriga_lib.acdf( data, bins_in, 5 );                     % n_bins is ignored
assertEqual( size(cdf_per_set4), [26, 2] );
assertElementsAlmostEqual( bins4, bins_in', 'absolute', 1e-14 );            % returned as column

%% Outputs: none, and fewer than the maximum
quadriga_lib.acdf( data );
cdf_per_set5 = quadriga_lib.acdf( data );
assertEqual( cdf_per_set5, cdf_per_set );
[ ~, ~, cdf_avg6 ] = quadriga_lib.acdf( data );
assertEqual( cdf_avg6, cdf_avg );

%% Type handling
cdf_per_set7 = quadriga_lib.acdf( single(data), [], uint32(201) );
assertTrue( isa(cdf_per_set7, 'double') );
assertElementsAlmostEqual( cdf_per_set7, cdf_per_set, 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.acdf(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.acdf( data, [], 201, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out6( data ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.acdf( data, [], 1 ), 'n_bins' );             % thrown by C++

end

% Request 6 outputs so that the nlhs > 5 guard fires
function out6(varargin)
[~, ~, ~, ~, ~, ~] = quadriga_lib.acdf( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
```
