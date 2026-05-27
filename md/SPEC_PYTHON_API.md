You are implementing a lightweight Python API wrapper for pybind11 (v3) to call my C++ radio channel modelling library (quadriga-lib). You will receive either a documentation block for the C++ API and/or a MATLAB MEX wrapper that implements the same function for MATLAB. Your task is to implement the wrapper function (a regular C++ function exposed via pybind11) that correctly calls the C++ function, handles input and output arguments, and ensures proper memory management. The wrapper should be designed to be user-friendly for Python users while maintaining the performance benefits of the underlying C++ library.

# CPP/MEX API DOCUMENTATION BLOCK FORMAT
- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.
- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.
- All functions and classes live in the `quadriga_lib` namespace.
- Default include: `#include "quadriga_lib.hpp"`.
- Template parameter `dtype` is `float` or `double` unless stated.
- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.
- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.
- Invalid inputs (shape/domain) cause a `std::invalid_argument`
- Index conventions: 0-based unless the field is explicitly called "1-based"
- Units: angles in radians (degrees only where stated, e.g. `*_deg`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).

# IMPLEMENTATION GUIDELINES

## File and function naming
- All Python wrapper source files use the `qpy_` prefix and match the C++/MEX function name, so `quadriga_lib::function_name` is wrapped in `qpy_function_name.cpp`.
- The wrapper function itself is named after the C++ function it wraps (e.g. `acdf`), and is exposed to Python via pybind11. The function name is kept as-is — no case conversion (lowercase stays lowercase).
- The dispatcher in `python_main.cpp` registers each wrapper with one of the submodules (`arrayant`, `channel`, `tools`, `RTtools`). From Python, the user calls e.g. `quadriga_lib.tools.acdf(...)`.

## Index and dtype conventions
- C++ and Python both use 0-based indices, so no conversion is performed in either direction. Indices documented as "1-based" in the C++ documentation remain 1-based in Python (the user provides 1-based and gets 1-based back).
- C++ functions may offer dtype = float / double specialization. The Python wrapper only uses the double version, so hardcode dtype to `double` in the call to the C++ function.
- Armadillo types map to their C++ counterparts, e.g. `arma::uword = unsigned long long` (enforced at compile time by static_asserts).
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two Python names with `_in` / `_out` suffixes (or `bins` / `bins_out` style where the input name is short).

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

## Helper function signatures
Quadriga-Lib uses Armadillo for data structures. Conversion between numpy arrays / Python objects and Armadillo types is done via the helpers in `python_arma_adapter.hpp`.

The `copy2numpy`, `stack2numpy` and `copy2list` converters take their Armadillo inputs **by pointer** (pass `&obj`, and `nullptr` for an absent optional input) and are templated as `<dtype, dtype_numpy = dtype>`: `dtype` is the Armadillo scalar (deduced from the argument) and `dtype_numpy` is the numpy element type. `dtype_numpy` may be the same as `dtype`, a different real type (cast on copy), or a `std::complex<...>` (complex output). When a *real* `dtype_numpy` is combined with an imaginary input, the real and imaginary parts are **interleaved along the rows** and the row count doubles; pass a `std::complex<...>` `dtype_numpy` to get a genuine complex array instead.

```cpp
// --- Copy a single Armadillo object to a Numpy array (auto = py::array_t<dtype_numpy>) ---
auto pyarray = qd_python_copy2numpy(&Col, &ColIm, i_elem, transpose);  // Col; im/i_elem/transpose optional
auto pyarray = qd_python_copy2numpy(&Mat, &MatIm, i_col);              // Mat; im/i_col optional
auto pyarray = qd_python_copy2numpy(&Cube, &CubeIm, i_slice);          // Cube; im/i_slice optional
auto pyarray = qd_python_copy2numpy(&Col);                             // simplest form: plain real copy
auto pyarray = qd_python_copy2numpy<unsigned, py::ssize_t>(&Mat);      // cast on copy
auto pyarray = qd_python_copy2numpy<float, std::complex<double>>(&MatRe, &MatIm, i_col);  // Re/Im -> complex
auto pyarray = qd_python_copy2numpy<float, std::complex<double>>(&CubeRe);                // Re only -> complex (imag = 0)

// --- Stack a std::vector of Armadillo objects into one higher-dim Numpy array ---
// (ragged inputs are zero-padded to the per-axis maximum; frame = last axis; i_vec selects frames)
auto pyarray = qd_python_stack2numpy(&vecCol, &vecColIm, i_vec);       // vector<Col>  -> 2D
auto pyarray = qd_python_stack2numpy(&vecMat, &vecMatIm, i_vec);       // vector<Mat>  -> 3D
auto pyarray = qd_python_stack2numpy(&vecCube, &vecCubeIm, i_vec);     // vector<Cube> -> 4D
auto pyarray = qd_python_stack2numpy<float, std::complex<double>>(&vecCubeRe, &vecCubeIm);  // -> complex 4D

// --- Copy a std::vector of Armadillo objects to a py::list of Numpy arrays ---
auto pylist = qd_python_copy2list(&vec, &vecImag, i_vec);              // vec = vector<Col|Mat|Cube>; im/i_vec optional
auto pylist = qd_python_copy2list<arma::Mat<float>, std::complex<double>>(&vecMatRe, &vecMatIm);  // -> list of complex
auto pylist = qd_python_copy2list(vecStrings, i_vec);                // vector of strings

// --- Reserve memory in Python and map to Armadillo (zero-copy writes) ---
auto pyarray = qd_python_init_output(n_elem, &Col);                            // 1D
auto pyarray = qd_python_init_output(n_rows, n_cols, &Mat);                    // 2D
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, &Cube);         // 3D
auto pyarray = qd_python_init_output(n_rows, n_cols, n_slices, n_frames, &vecCubes); // 4D

// --- Convert numpy -> Armadillo (view=true means zero-copy view if strides match, else copy) ---
auto Col     = qd_python_numpy2arma_Col(pyarray_or_handle, view, strict);
auto Mat     = qd_python_numpy2arma_Mat(pyarray_or_handle, view, strict);
auto Cube    = qd_python_numpy2arma_Cube(pyarray_or_handle, view, strict);
auto vecCube = qd_python_numpy2arma_vecCube(pyarray_or_handle, view, strict);  // 4D -> vector<Cube>

// --- Copy py::list of numpy arrays to std::vector of Armadillo types ---
auto vecCol     = qd_python_list2vector_Col<dtype>(pylist);
auto vecMat     = qd_python_list2vector_Mat<dtype>(pylist);
auto vecCube    = qd_python_list2vector_Cube<dtype>(pylist);
auto vecStrings = qd_python_list2vector_Strings(pylist);
qd_python_list2vector_Cube_Cplx<dtype>(pylistComplex, vecCubeRe, vecCubeIm);   // Complex -> Re/Im
```
- `dtype` (deduced, Armadillo scalar): `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`. `dtype_numpy` (numpy element type, defaults to `dtype`): any of those, or `std::complex<float>` / `std::complex<double>`.
- `dtype_numpy` cannot be deduced; to set it you must also spell `dtype`, since it is the second template parameter (e.g. `<float, std::complex<double>>`).
- Armadillo inputs `re`/`im` are passed by pointer; `im` defaults to `nullptr`. Selectors `i_elem`/`i_col`/`i_slice`/`i_vec` are `arma::uvec` passed by value and default to empty (= use all); `transpose` defaults to `false` (a `Col` with `transpose = true` is emitted as a `(1, n)` row).
- `i_vec` selects which `std::vector` elements (frames) are used; `i_elem`/`i_col`/`i_slice` select elements/columns/slices within a single object. Out-of-range indices throw `std::out_of_range`.
- `copy2numpy` and `copy2list` throw `std::invalid_argument` on a null `re`; `stack2numpy` returns an empty array when `re` is null or empty.
- The handle overloads of `qd_python_numpy2arma_*` accept `py::none()` and return an empty Armadillo object — used for optional inputs.


## Exception handling
- No `CALL_QD` macro is needed. pybind11 automatically translates `std::exception` (including `std::invalid_argument` and `std::runtime_error`) into Python exceptions, so the wrapper simply calls the C++ function directly.

## Section statement
- Functions are grouped into sections, marked by a comment block. This is used for grouping in the generated documentation:
```
/*!SECTION
Section name from C++ documentation block
SECTION!*/
```
- The section name does not correspond to the Python submodule — the submodule attachment is handled manually in the dispatcher.

## Documentation block
- Included as a comment block immediately before the wrapper function definition.
- Block starts with `/*!MD` and ends with `MD!*/`.
- First line after `/*!MD` is the function name (kept as-is, lowercase), followed by a 1-line summary on the next line.
- The bullet-point description follows directly after the 1-line summary — there is no `## Description:` heading.
- The 1-line summary should be identical to the C++/MEX documentation block.
- Each bullet contains one concise fact.
- Never use `*` or `|` in formulas or any other text — they are reserved for markdown formatting. Use `·` for multiplication (e.g. `Rz·Ry·Rx`) and `abs()` for absolute values.
- Lines longer than 120 characters need to be split, indenting continuation lines by 2 spaces.
- Omit any dtype specifications — they are not relevant for the Python user (the wrapper always uses double and integer types are converted automatically).
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
- Split the usage block across multiple lines if it exceeds 120 characters, indenting continuation lines by 4 spaces.
- Use `## Inputs:` and `## Outputs:` sections with the same one-liner style as in the C++ documentation block.
- Optional inputs are simply marked with their default value (e.g. `default: 201` or `default: None`); the user can omit them or pass `None` for arrays / `[]` for empty effect.
- Default values must be added to the argument description as they are not obvious from the signature.
- Units must only be specified when not SI (meters, radians, Hz, linear scale).
- C++ returns are merged into the `## Outputs:` section.
- Break argument descriptions longer than 120 characters into multiple lines, indenting continuation lines by 2 spaces.

## Function body structure
- The wrapper is a regular C++ free function. Its return type depends on the number of outputs:
  - **Single output**: return the object directly (e.g. `py::array_t<double>`, `arma::uword`, `std::string`, `py::list`).
  - **Multiple outputs**: return `py::tuple` and build it via `py::make_tuple(out1_py, out2_py, ...)`.
- All outputs are always computed and returned — there is no `nargout` equivalent in pybind11. Use this as the default. Do not add ad-hoc skip flags unless the C++ side has prohibitively expensive optional computation that the user has explicitly asked to gate.
- Naming convention inside the wrapper:
  - Input numpy arg → Armadillo view with suffix `_a` (e.g. `data` → `data_a`).
  - For in/out splits, the input Armadillo gets suffix `_a` (e.g. `bins` → `bins_a`).
  - Output Armadillo objects use the same name as in the Outputs section of the doc.
  - The corresponding numpy variable returned to Python uses the suffix `_py` (e.g. `cdf_per_set` → `cdf_per_set_py`).

## Input conversion

### Required numeric arrays
- Required array inputs use `const py::array_t<dtype>&`:
```
const auto data_a = qd_python_numpy2arma_Mat<double>(data, true);
```
- The `true` second argument means `view=true, strict=false` — try to get a zero-copy view, fall back to a copy if strides do not match.
- Use this default unless the input is modified later (e.g. the wrapper writes back into it). In that case use `false` (force copy) to avoid mutating the user's numpy buffer.
- Helpers: `qd_python_numpy2arma_Col`, `qd_python_numpy2arma_Mat`, `qd_python_numpy2arma_Cube`.
- dtype can be: `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`. Use `double` to match the C++ call.

### Optional numeric arrays
- Optional array inputs use `py::handle` (NOT `py::array_t<>`) with default `py::none()`:
```
py::handle bins  // in the function signature
const auto bins_in_a = qd_python_numpy2arma_Col<double>(bins, true);
```
- The handle overloads of `qd_python_numpy2arma_*` return an empty Armadillo object when the handle is `None`, which can then be checked via `.empty()`.
- In the pybind11 declaration comment: `py::arg("bins") = py::none()`.

### 4D numeric arrays as std::vector<arma::Cube>
- Strict 4D inputs `[d1, d2, d3, d4]` map to `std::vector<arma::Cube<dtype>>` of length `d4`, each cube of shape `(d1, d2, d3)`:
```
const auto data = qd_python_numpy2arma_vecCube<dtype>(input, true);
```
- Supports zero-copy aliasing when the numpy array is Fortran-contiguous with matching dtype; otherwise copies into freshly allocated cubes.
- Use this — NOT `qd_python_list2vector_Cube` — when the C++ documentation specifies a regular 4D shape like `[d1, d2, d3, d4]`. `qd_python_list2vector_Cube` is for ragged/padded lists of Cubes.

### Lists of Armadillo types (variable-length)
- The C++ function may expect a `std::vector<arma::Mat<dtype>>` (or Col / Cube), used for ragged lists of arrays. The Python user passes a `py::list` of numpy arrays:
```
const auto vec = qd_python_list2vector_Col<dtype>(pylist);
const auto vec = qd_python_list2vector_Mat<dtype>(pylist);
const auto vec = qd_python_list2vector_Cube<dtype>(pylist);
```
- For lists of strings: `qd_python_list2vector_Strings(pylist)`.

### Strings
- Strings are taken as `const std::string&` in the wrapper signature; pybind11 handles conversion automatically. No helper needed.
- Optional strings: `const std::string& name = ""` with `py::arg("name") = ""` in the declaration.

### Scalars
- Scalars use native C++ types directly in the wrapper signature: `arma::uword n_bins`, `double power`, `int count`, `bool flag`, etc.
- Defaults are set in the pybind11 declaration only: `py::arg("n_bins") = 201`. pybind11 also accepts the default in the function signature, but keep it in the declaration for visibility.
- Allowed types: `float`, `double`, `int`, `unsigned`, `long long`, `unsigned long long`, `arma::uword`, `bool`.

### Complex arrays
- The C++ library represents complex data as separate `arma::Mat<dtype>` (or `arma::Cube<dtype>`) for the real and imaginary parts — `arma::cx_*` types are not used. This is to avoid copy overhead in the C++ pipeline.
- By default, the Python wrapper presents these as **two separate real-valued numpy arrays** (`X_re`, `X_im`). This matches the C++ shape and avoids an unnecessary interleave/deinterleave copy.
- If the C++ documentation explicitly specifies that the Python wrapper should accept/return a single complex numpy array, use the complex helpers:
  - Input complex numpy → Re/Im arma:  
    `qd_python_copy2arma(complex_pyarr, real_cube, imag_cube);`
  - Re/Im arma → complex numpy output:  
    `auto out_py = qd_python_copy2numpy(real_mat, imag_mat);`
- For lists of complex arrays, use `qd_python_list2vector_Cube_Cplx` (input) and the `qd_python_copy2numpy(vecRe, vecIm)` overload (output).

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
- The output size must be known before the C++ call — typically derived from the input dimensions stated in the C++ documentation.
- For 4D outputs, each Cube in the vector aliases its slab of the numpy-owned 4D buffer (zero-copy). Do not resize the vector or any of its cubes after this call.

### Unknown output size
- Do not pre-allocate. Pass the empty Armadillo object to the C++ function, which will resize and populate it. After the call, copy to numpy:
```
arma::vec bins_out;
// ... C++ call writes into bins_out ...
auto bins_out_py = qd_python_copy2numpy(bins_out);
```
- Use this for outputs whose size depends on runtime behavior of the C++ call (e.g. reading from a file, runtime-generated bin counts).
- `qd_python_copy2numpy` supports: scalar, Col, Row, Mat, Cube, and (via the `_4d` variant) `std::vector<arma::Cube>` for ragged → padded 4D output.

### List of Armadillo types (output)
- If the C++ function returns a `std::vector<arma::Col/Mat/Cube>`, use the list overload of `qd_python_copy2numpy`:
```
auto pylist = qd_python_copy2numpy(vec_of_mats);
```
- Returns a `py::list` of numpy arrays, one per vector entry. Each entry preserves its own shape (ragged supported).

## Optional outputs / always-allocate policy
- Because all outputs are always returned, there is **no nullptr guard** on output pointers in the wrapper. The output Armadillo objects are always allocated (via `init_output`) or always passed by address (for unknown-size outputs), and the C++ function always receives a valid non-null pointer.
- Drop the MATLAB-style `var.empty() ? nullptr : &var` pattern entirely.
- Exception: the in/out param pattern. When a C++ argument is both input and output (e.g. `bins`), the wrapper:
  1. Reads the input arma object (empty if user passed `None`).
  2. Declares a separate output arma object (empty by default).
  3. If the input arma is non-empty, copies it into the output arma so the C++ side uses it as-is.
  4. Passes the address of the output arma to C++. The C++ side either keeps the existing content or resizes/generates new content.
  5. Copies the output arma to numpy after the call.

## Call the C++ function
- Call the underlying C++ function directly with the converted inputs and output pointers:
```
quadriga_lib::function_name<double>(arg1, arg2, &optional_arg3, &optional_output4);
```
- Hardcode `<double>` for templated functions.
- No try/catch — pybind11 catches `std::exception` automatically and re-raises as `RuntimeError` in Python.
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

## pybind11 declaration comment
- At the end of the file, include the pybind11 registration line as a comment block. The user copies this manually into the appropriate dispatcher function (e.g. `quadriga_lib_tools(m)` in `python_main.cpp`).
- Default the submodule handle to `m` (which corresponds to `tools` by convention; the user adjusts to `arrayant`, `channel`, or `RTtools` as needed):
```
// pybind11 declaration:
// m.def("function_name", &function_name,
//       py::arg("input1"),
//       py::arg("input2") = py::none(),
//       py::arg("n_bins") = 201);
```
- The kwarg names in `py::arg(...)` must match the C++ parameter names AND the names used in the `## Usage:` line of the documentation. These are what the user types as keyword arguments from Python.
- One `py::arg(...)` line per parameter. Include defaults for all optional parameters.

# OPTIONAL: REFERENCE FROM MEX WRAPPER
- If a MEX wrapper for the same C++ function already exists, it may be provided as additional context. In that case, treat it as the authoritative reference for: argument order, in/out splits, default values, descriptive argument renaming, and which outputs are computed.
- Diverge from the MEX wrapper only where Python idiom or this spec demands it. Specifically:
  - **Indices**: stay 0-based; no `+1` / `-1` conversions.
  - **Outputs**: always-allocate and always-return; drop all `nlhs > X` gates and all `var.empty() ? nullptr : &var` patterns.
  - **Function name in MD heading**: no case conversion.
  - **Lists**: MATLAB cell arrays → `py::list`.
  - **Complex**: MATLAB split Re/Im → keep split Re/Im by default in Python too (do not auto-merge into complex numpy unless the C++ doc explicitly requests it).
  - **Argument names in the Usage line**: must match the pybind11 kwarg names exactly, not the MATLAB names.
  - **No `CALL_QD` / try-catch**: pybind11 handles exceptions automatically.
  - **No nargout-style gates**: every output is always computed and returned.
- The MEX wrapper is optional context. When it is not provided, generate the Python wrapper from the C++ documentation block alone.

# REFERENCE EXAMPLE

C++ documentation block for the function `acdf`:

/*!MD
# acdf
Calculate the empirical averaged cumulative distribution function (CDF)

## Description:
- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from each column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` points to an empty vector, equally spaced bins spanning the data range are generated and stored back; if non-empty, those bin centers are used; if `nullptr`, bins are auto-generated internally

## Declaration:
```
void quadriga_lib::acdf(const arma::Mat<dtype> &data,
    arma::Col<dtype> *bins = nullptr,
    arma::Mat<dtype> *Sh = nullptr,
    arma::Col<dtype> *Sc = nullptr,
    arma::Col<dtype> *mu = nullptr,
    arma::Col<dtype> *sig = nullptr,
    arma::uword n_bins = 201);
```

## Input Arguments:
- **`data`** — Input data matrix; each column is one independent data set, `[n_samples, n_sets]`
- **`bins`** *(optional)* — Bin centers; auto-generated and stored back if pointing to empty vector, used as-is if non-empty, ignored if `nullptr`, `[n_bins]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when non-empty bins are provided

## Output Arguments:
- **`Sh`** *(optional)* — Individual CDFs, one per column of data, `[n_bins, n_sets]`
- **`Sc`** *(optional)* — Averaged CDF via quantile-space averaging across data sets, `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets, `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets, `[9]`
MD!*/

Complete Python wrapper implementation for the `acdf` function based on the above documentation block, following the implementation guidelines provided:

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
    auto bins_out_py = qd_python_copy2numpy(bins_out);

    // Return tuple
    return py::make_tuple(cdf_per_set_py, bins_out_py, cdf_avg_py, mu_py, sig_py);
}

// pybind11 declaration:
// m.def("acdf", &acdf,
//       py::arg("data"),
//       py::arg("bins") = py::none(),
//       py::arg("n_bins") = 201);
```
