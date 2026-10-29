quadriga-lib is a C++ radio channel modelling library with MATLAB/Octave and Python APIs. This document gives the conventions for the MATLAB / Octave MEX wrapper of a C++ function: the wrapper in `api_mex/` and its test in `tests/mex_tests/`. The wrapper converts arguments, shifts indices, allocates outputs and calls the C++ function; it should be user-friendly for MATLAB users while keeping the performance of the C++ library. Like `md/SPEC_CPP_API.md`, it is persistent and used at every stage of the work; the MATLAB interface is worked out in the plan or the spec, and the workflow (reviews, stops) is set in the plan. Where this document says "ask", "on request" or "unless requested", the point belongs in the plan or the spec; an implementer who finds it unsettled stops and reports it. The Python wrappers are covered by `md/SPEC_PYTHON_API.md`.

# C++ CONVENTIONS
- `md/SPEC_CPP_API.md` applies to wrappers and tests as well and is not repeated here: "GENERAL CONVENTIONS" (types, shapes, index bases, units, citations, lean code, MSVC), "Input validation" (exceptions), "File-local helpers" (helper scoping), "License statement" and "Documentation block".

# REPOSITORY LAYOUT AND BUILD
- `api_mex/<function_name>.cpp` — MEX wrappers, one file per MATLAB function. CMake globs `api_mex/*.cpp`, so a new file needs no registration in the build system.
- `src/mex_helper_functions.hpp` (generic conversions) and `src/mex_quadriga_lib_functions.hpp` (arrayant / channel structs, `std::any` datasets) — helper functions. Read them when a signature is unclear; the lists in this document are a summary.
- `include/*.hpp` — C++ declarations; `src/*.cpp` — C++ implementations with their documentation blocks (`/*!MD ... MD!*/`).
- `tests/mex_tests/test_<function_name>.m` — MOxUnit tests, one file per wrapper; `tests/quadriga_lib_mex_tests.m` collects and runs all `tests/mex_tests/test*.m` files, so a new test file needs no registration.
- `+quadriga_lib/` — the MATLAB package populated by the build: the compiled MEX files and one `.m` help file per function.
- Generated overviews (read-only, produced by `make documentation`, never edit them; they may lag behind the sources until the documentation is rebuilt):
  - `html_docu/quadriga_lib_api_cpp.md` — all C++ functions
  - `html_docu/quadriga_lib_api_mex.md` — all MATLAB functions
  - Both start with a "Function Index"; use it as described in `md/SPEC_CPP_API.md` ("Fixtures and oracles").
- The text in the `/*!MD ... MD!*/` block of a wrapper is reused for 3 purposes: (1) merged into the Markdown overview, (2) compiled to HTML, (3) converted into the `.m` help file of the function that provides the MATLAB `help` text (`tools/extract_matlab_comments.py`, which strips backticks and asterisks and prefixes each line with `%`). The delimiters `/*!MD`, `MD!*/`, `/*!SECTION`, `SECTION!*/` are matched literally, each on its own line.
- Commands (Linux, from the repository root):
  - `make` — build the library and install the MEX files into `+quadriga_lib/` (`matlab` and `octave` default to ON); run it after every wrapper change
  - `make test` — build and run the Catch2 suite and the complete MEX test suites; it does not install, so run `make` first
  - A single test file is run from MATLAB / Octave by putting the repository root (for `+quadriga_lib`), `tests/mex_tests` and MOxUnit (`external/MOxUnit-master/MOxUnit`, unpacked by `make moxunit-lib`; then `moxunit_set_path`) on the path and calling the test function by name, e.g. `test_icosphere`. It returns silently on success and raises an error on the first failed assertion.
- Wrappers and tests must work in both MATLAB and Octave.

# INTERFACE: WHAT THE PLAN OR SPEC SETTLES
The interface is derived from the C++ declaration and documentation block, existing or settled in the same plan or spec. The plan or the spec settles the MATLAB interface:
- the MATLAB usage line: order and names of inputs and outputs, in/out splits
- which inputs are optional and their defaults
- which arguments are indices and their base on the C++ side
- the size of every output and whether the wrapper can know it before the call
- the helper for every argument, and the section name
- anything that does not fit this document: a C++ type without helper, an unclear index base, an output order that is not obvious, logic that would have to live in the wrapper

Existing wrappers that show the patterns:
- `o2i_loss.cpp` — a complete recent example: array and string-list inputs, a default, outputs of known size, a 1-based index output; test in `tests/mex_tests/test_o2i_loss.m` (its documentation block predates the lean rules)
- `icosphere.cpp` — scalar inputs, outputs of known size, optional output pointers
- `obj_file_read.cpp` — outputs of runtime size, index shifts, cell arrays, map-to-struct
- `triangle_mesh_aabb.cpp` — optional 1-based index input converted in place
- `get_channels_spherical.cpp` — arrayant struct inputs, 3-element position vectors
- `get_channels_multifreq.cpp` — multi-frequency arrayant struct arrays, 4D outputs
- `arrayant_rotate_pattern.cpp` — arrayant struct in / struct out, in-place modification
- `hdf5_write_channel.cpp`, `hdf5_read_channel.cpp` — channel structs, unstructured data

# IMPLEMENTATION ORDER
Unlike the C++ work, the wrapper is written first and the test second: the physics is already verified by the Catch2 suite, and the MEX test exercises the wrapper.
1. **Wrapper.** Write `api_mex/<function_name>.cpp` following the "IMPLEMENTATION GUIDELINES". It must compile without warnings.
2. **Test.** Write `tests/mex_tests/test_<function_name>.m` (see "TESTS") and run it. It is expected to pass. If it fails, the cause is in the wrapper or in the test's expectation; find out which before changing either. If the failure points to the C++ function itself, stop and report it; do not work around it in the wrapper.
3. **Finish.** The new test passes; the complete MEX test suite passes. The documentation block of the wrapper is complete and matches what the wrapper does.

# HAND-OVER REPORT
- the MATLAB usage line
- the test results with the actual output of the run, and which of MATLAB / Octave was used; do not report a test as passing that was not run
- compiler warnings, if any
- every deviation from the C++ interface and why, and anything that was not tested

# IMPLEMENTATION GUIDELINES
- `quadriga_lib::function_name` is wrapped in `api_mex/function_name.cpp`, whose `mexFunction` calls it; MATLAB users call it as `quadriga_lib.function_name` from the `+quadriga_lib` package populated by the build. Exceptions (a class member function, one MATLAB function dispatching to several C++ functions) are settled in the plan or spec.
- Keep the wrapper thin: argument conversion, index shifts, output allocation and the C++ call. Anything that changes numerical results (unit conversions, defaults that differ from C++, loops over several C++ calls, mode dispatch) belongs in the C++ library; if a wrapper seems to need it, ask first.
- All errors map to "quadriga_lib:CPPerror": `mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Error text.");`
- Templated C++ functions are called with `double`. Functions with fixed types (e.g. the float-only ray-tracing pipeline) are called with their type, and their outputs keep it.
- Integer outputs keep their C++ type (e.g. `arma::u32_vec` is returned as uint32, `arma::uvec` as uint64); state the type in the documentation.
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two MATLAB names with `_in` / `_out` suffixes.

## Index conversion
- MATLAB users give and get 1-based indices. Indices that are 0-based in C++ are shifted by the wrapper (-1 on input, +1 on output); indices that are 1-based in C++ pass through unchanged.
- Index inputs that get converted MUST be read with `copy = true`. Without the copy, the Armadillo object may borrow the memory of the MATLAB input, and editing it in place modifies the caller's variable (undefined behavior).
- Always check for zeros before subtracting, otherwise unsigned indices wrap around:
```
// Vector index input, 1-based (MATLAB) -> 0-based (C++), empty = C++ default
arma::uvec i_element = (nrhs < 3) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[2], true, "i_element");
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

## Headers
```
#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"
```
- `#include "mex_quadriga_lib_functions.hpp"` instead when the wrapper converts arrayant structs, channel structs or `std::any` datasets; it pulls in the three headers above.

## Helper function signatures
- Armadillo is pulled in by `quadriga_lib.hpp`. `dtype` in the helpers is one of float, double, int, unsigned, long long, unsigned long long (plus bool for scalars).
- Pass the argument name as `var_name` wherever a helper takes it, so that error messages name the argument.

```
// ---------- Inputs: MATLAB -> C++ ----------

// Scalar, casts to dtype; returns default_value for empty input "[]". Integer types need an integer value in range.
dtype qd_mex_get_scalar(const mxArray *input, std::string var_name, dtype default_value);
dtype qd_mex_get_scalar(const mxArray *input, std::string var_name = ""); // floating point only; NaN for "[]"

// Seed of a random number generator: -1 (random) or 0 ... 2^63-1; -1 for empty input
long long qd_mex_get_seed(const mxArray *input, const std::string &var_name = "seed");

// Char array -> std::string; returns default_value for empty input (the second argument is NOT a name)
std::string qd_mex_get_string(const mxArray *input, std::string default_value = "");

// Cell array of strings (or a single char array) -> vector of strings; empty input -> empty vector
std::vector<std::string> qd_mex_get_strings(const mxArray *input, const std::string &var_name = "");

// Numeric arrays. Borrow the MATLAB memory (zero-copy, read-only!) when the MATLAB class matches
// dtype and copy == false; otherwise allocate and type-cast. Empty input -> empty object.
// NOTE: the second argument is `bool copy`. Passing a string literal there compiles and means copy = true.
arma::Col<dtype> qd_mex_get_Col(const mxArray *input, bool copy = false, const std::string &var_name = "");  // 1D
arma::Mat<dtype> qd_mex_get_Mat(const mxArray *input, bool copy = false, const std::string &var_name = "");  // 2D
arma::Cube<dtype> qd_mex_get_Cube(const mxArray *input, bool copy = false, const std::string &var_name = ""); // 3D
std::vector<arma::Cube<dtype>> qd_mex_get_CubeVec(const mxArray *input, bool copy = false,
                                                  const std::string &var_name = ""); // regular 4D

// Always copy + type-cast; n_elem != 0 enforces the number of elements (e.g. 3 for a position vector)
arma::Col<dtype> qd_mex_typecast_Col(const mxArray *input, std::string var_name = "", size_t n_elem = 0);
arma::Mat<dtype> qd_mex_typecast_Mat(const mxArray *input, std::string var_name = "");
arma::Cube<dtype> qd_mex_typecast_Cube(const mxArray *input, std::string var_name = "");
std::vector<arma::Cube<dtype>> qd_mex_typecast_CubeVec(const mxArray *input, std::string var_name = "");

// Ragged lists: ND array -> std::vector of Armadillo objects, split along dimension vec_dim (0-based,
// by convention the last MATLAB dimension); always copies; empty input -> empty vector
std::vector<arma::Col<dtype>> qd_mex_matlab2vector_Col(const mxArray *input, size_t vec_dim, const std::string &var_name = "");
std::vector<arma::Mat<dtype>> qd_mex_matlab2vector_Mat(const mxArray *input, size_t vec_dim, const std::string &var_name = "");
std::vector<arma::Cube<dtype>> qd_mex_matlab2vector_Cube(const mxArray *input, size_t vec_dim, const std::string &var_name = "");
std::vector<bool> qd_mex_matlab2vector_Bool(const mxArray *input, const std::string &var_name = ""); // nonzero = true

// 1x1 struct of numeric vectors -> map (one key per field, equal lengths); non-struct or empty -> empty map
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
| random seed (`long long`, -1 = random) | scalar | `qd_mex_get_seed` | - |
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
- Some existing arrayant wrappers additionally accept the fields as separate inputs or return them as separate outputs. This is legacy behavior; new wrappers use the struct only.

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

## Section statement
- Place the section block of the C++ function before the documentation block; it groups the functions in the generated documentation. The name must match one of the names in `md/SPEC_CPP_API.md` ("Section statement") exactly.
```
/*!SECTION
Section name from C++ documentation block
SECTION!*/
```

## Documentation block
- The format and the lean rules of the C++ block apply (`md/SPEC_CPP_API.md`, "Documentation block"): same one-line summary, description as bullets without a heading, singular `## Input:` / `## Output:` for one entry, `## See also:`, no `*` or `|` in text. Do not expand what is brief in the C++ block. The differences:
- The heading is the function name in UPPERCASE, e.g. `acdf` becomes `ACDF`.
- The description is adapted to MATLAB users: C++-specific details (dtype, pointers) are dropped, MATLAB-specific notes added where needed.
- Lines should not exceed 100 characters (soft limit), since the block also becomes the `.m` help file. Indent continuation lines by 2 spaces; table rows and usage lines may exceed the limit.
- A `## Usage:` section replaces the Declaration; split it at 100 characters with continuation lines indented by 4 spaces:

    ## Usage:
    ```
    [ output1, output2 ] = quadriga_lib.function_name( input1, input2, optional_input3 );
    ```

- Argument format: ``- **`name`** — Description; further facts separated by semicolons; `[size]`; default: X``
- Optional inputs (C++: default value or `nullptr`) can be omitted or passed as `[]`. A stated default implies that the input is optional; mark it `*(optional)*` only if no default can be stated. Outputs are never marked optional.
- State the type only when not `double`, the unit only when not SI. C++ returns are listed under `## Outputs:`.

## Function body
- Signature: `void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])`
- Validate the argument counts first:
```
if (nrhs < 1 || nrhs > 2)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
if (nlhs > 3)
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
```
- A C++ function without results gives a wrapper without outputs: `nlhs > 0` is an error, and the `## Outputs:` section is omitted.

## Input conversion
- Read every optional input behind a guard, `(nrhs < k) ? default : ...(prhs[k - 1])`, before touching `prhs`.
- Numeric arrays are `const` and read with `copy = false`, with the dtype the C++ function expects; only inputs that the wrapper modifies (index conversion) use `copy = true`. An omitted optional array is an empty object:
```
const auto data = qd_mex_get_Mat<double>(prhs[0], false, "data");
const auto bins = (nrhs < 2) ? arma::vec() : qd_mex_get_Col<double>(prhs[1], false, "bins");
const auto tx_pos = qd_mex_typecast_Col<double>(prhs[7], "tx_pos", 3); // fixed length
```
- A regular 4D shape `[d1, d2, d3, d4]` is read with `qd_mex_get_CubeVec` (zero-copy where possible); `qd_mex_matlab2vector_*` is only for ragged, zero-padded lists.
- Strings: `(nrhs < 3) ? std::string("name") : qd_mex_get_string(prhs[2], "name")` (the second argument is the default); lists of strings: `qd_mex_get_strings(prhs[4], "models")`.
- Scalars state their default twice, once for an omitted input and once for `[]`; integer and bool scalars must pass a default:
```
const auto n_bins = (nrhs < 3) ? 201 : qd_mex_get_scalar<arma::uword>(prhs[2], "n_bins", 201);
```
- Seeds are read with `qd_mex_get_seed`.

## Output allocation
- Declare an empty Armadillo variable for every output.
- **Known size**: allocate in MATLAB memory before the call with `qd_mex_init_output`; the C++ function then writes into the MATLAB array directly (zero-copy). The memory is zero-initialized and the size must match exactly what the C++ function produces: if it resizes the container, the result goes to new memory and the MATLAB output stays zero. For a 4D output use the `std::vector<arma::Cube>` overload and do not resize the vector or its cubes afterwards.
```
if (nlhs > 0)
    plhs[0] = qd_mex_init_output(&az, n_rows, n_cols);
```
- **Unknown size** (determined by the C++ call, or a return value): pass the empty object and copy after the call with `qd_mex_copy2matlab`, or `qd_mex_vector2matlab` for a `std::vector` of Armadillo objects (zero-padded to a regular ND array). Prefer `qd_mex_init_output` whenever the wrapper knows the size.

## Optional and mandatory C++ arguments
- Optional C++ outputs (pointers) get `(nlhs > k) ? &obj : nullptr`, so that an output nobody requested is neither allocated nor computed.
- Optional C++ inputs (pointers) get `nullptr` when the input is omitted or empty.
- Mandatory C++ arguments (by reference) get the object itself. A mandatory output is pre-allocated with `qd_mex_init_output` only when it is requested; otherwise the local empty object is passed and discarded.

## Call the C++ function
- `CALL_QD(expr)` (defined in `mex_helper_functions.hpp`) wraps the call and turns any `std::exception` into `mexErrMsgIdAndTxt("quadriga_lib:CPPerror", ex.what())`:
```
CALL_QD(quadriga_lib::function_name(arg1, arg2, p_optional_arg3, p_output4));
```
- After the call, copy all results that were not pre-allocated to MATLAB.

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
- Keep it lean: no duplicated cases, no helper layers. The test is as long as it needs to be to cover the checklist below. It must also run fast: the complete suite runs in every future session (see "TESTS" in `md/SPEC_CPP_API.md`).

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

## Default coverage
The plan or spec may narrow or extend this list.
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
