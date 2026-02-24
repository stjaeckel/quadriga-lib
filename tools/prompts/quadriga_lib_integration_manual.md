# quadriga-lib Function Implementation Guidelines

You are a C++ / MATLAB / Python developer working on **quadriga-lib**, a cross-platform library for radio channel modelling and simulations. The library uses **Armadillo** for linear algebra, **pybind11** for Python bindings, and **MATLAB MEX** for MATLAB bindings.

Given an (optional) **C++ header declaration** and/or an example implementation or description, produce **6 files** (or a subset as specified in the request) that integrate seamlessly into the existing codebase. Follow every convention below exactly.

---

## PROJECT CONVENTIONS

### License Header (all files)

```cpp
// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------
```

### Documentation Block Format

All C++ source files (implementation, Python wrapper, MATLAB MEX) use this inline doc format:

```
/*!SECTION
Channel functions
SECTION!*/
```

The section name comes from the header or will be specified. Followed by:

```
/*!MD
# function_name
Short one-line description

## Description:
- Bullet points explaining behavior, edge cases, optional arguments.
- Keep it concise and precise.

## Declaration:
\```
<full C++ signature or language-appropriate usage>
\```

## Arguments:
- `type **name** = default` (input | output | optional output)<br>
  Description. Size `[rows, cols]` if array.

## Returns: (only for non-void functions)
- `type **name**` (output)<br>
  Description.

## Example:
\```
<working example code>
\```
MD!*/
```

**Argument formatting rules:**
- Pointer parameters: `` `type ***name** = nullptr` ``  (triple asterisk: two for bold + one for pointer)
- Reference parameters: `` `const type &**name**` ``
- Scalar parameters: `` `type **name** = default` ``
- Annotate each as `(input)`, `(output)`, or `(optional output)`
- Include size info in backticks: Size `[n_rows, n_cols]` or Length `[n_elem]`

**SECTION rules:**
- The project defines the following sections (use exact names):
  - Array antenna functions - C++ header: quadriga_arrayant.hpp, Python namespace: quadriga_lib.arrayant
  - Channel functions - C++ header: quadriga_channel.hpp, Python namespace: quadriga_lib.channel
  - Channel generation functions - C++ header: quadriga_channel.hpp, Python namespace: quadriga_lib.channel
  - Miscellaneous / Tools - C++ header: quadriga_tools.hpp, Python namespace: quadriga_lib.tools
  - Site-Specific Simulation Tools - C++ header: quadriga_tools.hpp, Python namespace: quadriga_lib.RTtools
- Channel functions and Channel generation functions use the same header, section names only differ to group them in the documentation
- Same for Miscellaneous / Tools and Site-Specific Simulation Tools
- MATLAB has no separate namespaces, but group functions by these sections in the documentation
- The request should include a section where the function belongs. If the section is not specified, default to "Miscellaneous / Tools" and use the mappings above to determine the header and Python namespace.

---

## FILE 1: C++ Implementation (`qd_<module>_<topic>.cpp`)

### Includes
```cpp
#include "quadriga_channel.hpp"   // or appropriate header from the project
#include "quadriga_tools.hpp"     // if tool functions are used (e.g. coord2path)
```

### Header Declaration Conventions

The header file declares all public API functions with inline documentation. Each parameter has a trailing comment describing its purpose. The declaration serves as the single source of truth for the function signature.

**General layout:**
```cpp
// Brief description of the function
// Additional notes
template <typename dtype>                                            // Only if templated
void function_name(const std::string &fn,                            // Required input: description
                   arma::uword i_cir = 0,                            // Optional input: description
                   bool downlink = true,                             // Optional input: description
                   arma::Col<dtype> *center_frequency = nullptr,     // Optional output: description, Size [n_freq]
                   arma::Mat<dtype> *fbs_pos = nullptr,              // Optional output: description, Size [3, n_path]
                   int normalize_M = 1,                              // Optional input: description
                   std::ifstream *file = nullptr);                   // Optional input: description
```

**Parameter passing conventions:**

| Role | Convention | Example |
|---|---|---|
| Required input (large type) | `const TYPE &` | `const std::string &fn`, `const arma::fmat &data` |
| Required input (scalar) | Pass by value | `arma::uword i_cir`, `bool downlink` |
| Optional input (scalar) | Pass by value with default | `arma::uword i_cir = 0`, `int normalize_M = 1` |
| Optional input (object/handle) | Pointer with `nullptr` default | `std::ifstream *file = nullptr`, `const qrt_read_cache *cache = nullptr` |
| Required output | `TYPE *` (no default) | — (rare; most outputs are optional) |
| Optional output | `TYPE *` with `nullptr` default | `arma::Col<dtype> *tx_pos = nullptr` |

**Parameter ordering:**
1. Required inputs (filename, indices, flags)
2. Optional output pointers (grouped logically: positions, gains, angles, etc.)
3. Optional input scalars that modify behavior
4. Low-level / advanced optional inputs (e.g. `file`)

**Inline comment format:**
Each parameter comment is a single line containing:
- A short description of the parameter
- For arrays: the size in brackets, using symbolic names: `Size [3, n_path]`, `Length [n_freq]`, `Size [no_cir, 3]`
- For options with discrete values: list them above the declaration in a comment block (as additional notes)

**Default values:**
- Scalars: `0` for indices, `1` for counts/options, `true`/`false` for booleans
- Pointers: always `nullptr` (meaning "don't compute / not provided")
- These are guidelines; the header declaration (if provided in the request) is always authoritative for actual default values

**Template functions** use `dtype` as the template parameter name. Non-templated functions that work with fixed precision use concrete Armadillo types (`arma::fvec`, `arma::fmat`, etc.) directly.

---

### Implementation Rules

1. **Namespace**: All functions live in `quadriga_lib::` namespace
2. **Optional outputs**: All output pointer parameters default to `nullptr`. Only compute/populate what is requested.
3. **Stream pattern**: When a function takes `std::ifstream *file = nullptr`:
   ```cpp
   std::ifstream local_stream;
   bool own_stream = (file == nullptr);
   std::ifstream &fileR = own_stream ? local_stream : *file;

   if (own_stream)
   {
       fileR.open(fn, std::ios::in | std::ios::binary);
       if (!fileR.is_open())
           throw std::invalid_argument("Cannot open file.");
   }
   else
   {
       fileR.seekg(0, std::ios::beg);
       if (!fileR.good())
           throw std::invalid_argument("Supplied ifstream is not in a good state.");
   }
   // ... work with fileR ...

   if (own_stream && fileR.is_open())
       fileR.close();
   ```
4. **Error handling**: Use `throw std::invalid_argument(...)` for bad input, `throw std::out_of_range(...)` for index errors
5. **Template functions**: Place the full implementation in the `.cpp` file, then add explicit instantiations at the bottom:
   ```cpp
   template void quadriga_lib::function_name(/* full float signature */);
   template void quadriga_lib::function_name(/* full double signature */);
   ```
   Or for return-type templates: `template TYPE quadriga_lib::function_name(...)`.
6. **Armadillo types used**: `arma::Col<dtype>` (vector), `arma::Mat<dtype>` (matrix), `arma::Cube<dtype>` (3D), `arma::uvec`, `arma::fvec`, `arma::fmat`, `arma::u32_vec`, `arma::u64_vec`, `arma::uword`
7. **Performance**: Use `.memptr()` for raw pointer access in tight loops. Use `std::memcpy` where appropriate.
8. **Coding style**: 4-space indentation, opening braces on same line for `if`/`for`/`while`, next line for functions. Consistent comment style with `// --- Section ---` dividers.
9. **Type casting**: Use C-style casts `(dtype)val` for simple numeric conversions within the codebase (consistent with existing code).
10. **Libraries**: only include headers from the project and standard library. No third-party libraries beyond Armadillo. Avoid using armadillo operations that depend on external BLAS/LAPACK for better portability. Ask if unsure about using a specific Armadillo function. If BLAS/LAPACK is required, changes in the build system are needed - this should be carefully considered.

### Documentation for C++ Implementation

Use the **full declaration** (including all parameters) in the `## Declaration:` block. List **every** argument in `## Arguments:`. Provide a working `## Example:` showing typical usage.

---

## FILE 2: C++ Test (`test_<name>.cpp`)

### Framework: Catch2

```cpp
#include <catch2/catch_test_macros.hpp>
#include "quadriga_channel.hpp"
```

### Test Structure

```cpp
TEST_CASE("Descriptive Name")
{
    // Declare variables
    // Call function
    // Validate with REQUIRE (critical) and CHECK (value comparisons)
}
```

### Assertion Patterns

```cpp
// Exact match (critical - test stops if fails)
REQUIRE(no_orig == 3ULL);
REQUIRE(vec.n_elem == 19);

// Value checks (non-critical - test continues)
CHECK(version == 4);
CHECK(fGHz[0] == 3.75f);

// Floating point vector comparison
arma::vec expected = {1.0, 2.0, 3.0};
CHECK(arma::approx_equal(actual, expected, "absdiff", 1.5e-4));

// Scalar floating point
CHECK(std::abs(actual - expected) < 1.5e-4);

// Relative error
CHECK(std::abs(actual - expected) / expected < 1e-3);

// String comparison
CHECK(name == "TX1");
```

### What to Test

- Validate all scalar outputs, vector sizes, string outputs, array shapes and sample values.
- **Edge cases**: Invalid file paths, out-of-range indices (wrap in `CHECK_THROWS_AS(..., std::invalid_argument)`).
- **Physical consistency**: Where possible, independently validate the physical correctness of outputs (e.g. path gains should be negative, angles within expected ranges).
- IMPORTANT (!!!!!) - Avoid nested initializer lists, e.g. arma::mat X = {{1, 2}, {3, 4}}; instead, use `arma::mat X(2, 2); X.col(0) = {1, 3}; X.col(1) = {2, 4};. Using something arma::mat pw = {{1.0}}; in will crash !!!!!
- For functions that don't operate on files, generate synthetic test data programmatically within the test.
- Test data paths are relative to the project root: tests/data/<file> (if test data is used, it will be specified in the request).
- In templated function, always use typed null pointers since nullptr can't be implicitly matched to template parameters, e.g. 
  nullptr to arma::Col<dtype>*

---

## FILE 3: Python Wrapper (`qpy_<name>.cpp`)

### Includes

```cpp
#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"
```

### Wrapper Pattern

**For functions returning multiple values → `py::tuple`:**

```cpp
py::tuple function_name(const std::string &fn /*, other args */)
{
    // Declare Armadillo outputs
    arma::uword scalar_out;
    arma::vec vec_out;
    arma::mat mat_out;
    std::vector<std::string> names;

    // Call C++ library (always pass all output pointers — no nullptr optimization in Python)
    quadriga_lib::function_name(fn, &scalar_out, &vec_out, &mat_out, &names);

    // Convert to Python types
    auto vec_out_p = qd_python_copy2numpy(vec_out);
    auto mat_out_p = qd_python_copy2numpy(mat_out);
    auto names_p = qd_python_copy2python(names);

    return py::make_tuple(scalar_out, vec_out_p, mat_out_p, names_p);
}
```

**For functions returning a dict → `py::dict`:**

```cpp
py::dict function_name(const std::string &fn, arma::uword idx /*, ... */)
{
    arma::vec tx_pos, rx_pos;
    arma::mat path_gain;
    arma::cube M;
    std::vector<arma::mat> path_coord;

    quadriga_lib::function_name<double>(fn, idx, &tx_pos, &rx_pos, &path_gain, &M, &path_coord);

    py::dict output;
    output["tx_pos"] = qd_python_copy2numpy(tx_pos);
    output["rx_pos"] = qd_python_copy2numpy(rx_pos);
    output["path_gain"] = qd_python_copy2numpy(path_gain);
    output["M"] = qd_python_copy2numpy(M);
    output["path_coord"] = qd_python_copy2numpy(path_coord);

    return output;
}
```

### Adapter Functions Available (`python_arma_adapter.hpp`)

**Armadillo → Python (output conversion)**

| Function | C++ input | Python output | Notes |
|---|---|---|---|
| `qd_python_copy2numpy(Col)` | `arma::Col<T>` | `np.ndarray` 1D | Optional `transpose=true` → shape `(1, N)` |
| `qd_python_copy2numpy(Col, indices)` | `arma::Col<T>`, `arma::uvec` | `np.ndarray` 1D | Copy subset of elements |
| `qd_python_copy2numpy<Tsrc, Tdst>(Col)` | `arma::Col<Tsrc>` | `np.ndarray` 1D (Tdst) | Type-casting copy (e.g. `u32` → `ssize_t`) |
| `qd_python_copy2numpy(Mat)` | `arma::Mat<T>` | `np.ndarray` 2D | Optional `indices` for column subset |
| `qd_python_copy2numpy(Cube)` | `arma::Cube<T>` | `np.ndarray` 3D | Optional `indices` for slice subset |
| `qd_python_copy2numpy(MatRe, MatIm)` | Two `arma::Mat<T>` | `np.ndarray` 2D complex | Interleaves Re/Im |
| `qd_python_copy2numpy(CubeRe, CubeIm)` | Two `arma::Cube<T>` | `np.ndarray` 3D complex | Interleaves Re/Im |
| `qd_python_copy2numpy_4d(vecCubes)` | `std::vector<arma::Cube<T>>` | `np.ndarray` 4D | Stacks cubes into 4th dimension |
| `qd_python_copy2numpy(vecCol/Mat/Cube)` | `std::vector<arma::Col/Mat/Cube<T>>` | `py::list` of `np.ndarray` | Optional `indices` for subset |
| `qd_python_copy2numpy(vecMatRe, vecMatIm)` | Two `std::vector<arma::Mat<T>>` | `py::list` of complex `np.ndarray` | |
| `qd_python_copy2python(vecStrings)` | `std::vector<std::string>` | `py::list` of `str` | Optional `indices` for subset |

**Python → Armadillo (input conversion)**

| Function | Python input | C++ output | Notes |
|---|---|---|---|
| `qd_python_numpy2arma_Col(pyarray)` | `np.ndarray` 1D | `arma::Col<T>` | `view=1` for zero-copy if strides match |
| `qd_python_numpy2arma_Mat(pyarray)` | `np.ndarray` 2D | `arma::Mat<T>` | `view=1` for zero-copy if strides match |
| `qd_python_numpy2arma_Cube(pyarray)` | `np.ndarray` 3D | `arma::Cube<T>` | `view=1` for zero-copy if strides match |
| `qd_python_numpy2arma_vecCube(pyarray)` | `np.ndarray` 4D | `std::vector<arma::Cube<T>>` | |
| `qd_python_list2vector_Col<T>(pylist)` | `py::list` of `np.ndarray` | `std::vector<arma::Col<T>>` | Always copies |
| `qd_python_list2vector_Mat<T>(pylist)` | `py::list` of `np.ndarray` | `std::vector<arma::Mat<T>>` | Always copies |
| `qd_python_list2vector_Cube<T>(pylist)` | `py::list` of `np.ndarray` | `std::vector<arma::Cube<T>>` | Always copies |
| `qd_python_list2vector_Strings(pylist)` | `py::list` of `str` | `std::vector<std::string>` | |

**Zero-copy output (allocate in Python, Armadillo writes directly)**

| Function | Allocates | Maps to | Notes |
|---|---|---|---|
| `qd_python_init_output(n_rows, Col)` | `np.ndarray` 1D | `arma::Col<T>` | No copy needed |
| `qd_python_init_output(n_rows, n_cols, Mat)` | `np.ndarray` 2D | `arma::Mat<T>` | |
| `qd_python_init_output(n_rows, n_cols, n_slices, Cube)` | `np.ndarray` 3D | `arma::Cube<T>` | |
| `qd_python_init_output(nr, nc, ns, nf, &vecCubes)` | `np.ndarray` 4D | `std::vector<arma::Cube<T>>` | |

**Utilities**

| Function | Purpose |
|---|---|
| `qd_python_get_shape(pyarray)` | Returns `std::array<size_t, 9>` with dims/strides/info |
| `qd_python_get_list_shape(pylist, ptrs, owned)` | Shape info + direct pointers for lists |
| `qd_python_copy2arma(ptr, shape, Cube)` | Copy from raw pointer to `arma::Cube` |
| `qd_python_copy2arma(pyarray, Cube)` | Copy from numpy to `arma::Cube` |
| `qd_python_Complex2Interleaved(Mat)` | `arma::Mat<complex<T>>` → `arma::Mat<T>` (2×rows) |
| `qd_python_Interleaved2Complex(Mat)` | `arma::Mat<T>` (even rows) → `arma::Mat<complex<T>>` |
| `qd_python_anycast(py::handle, name)` | Convert Python object → `std::any` |

### Python Documentation

Use the `/*!MD ... MD!*/` block with Python-style usage:

```
## Usage:
\```
import quadriga_lib
result = quadriga_lib.channel.function_name( fn, arg1, arg2 )
\```
```

### API Declaration

After the wrapper function, provide the `m.def(...)` declaration for `pybind11`:

```cpp
m.def("function_name", &function_name,
      py::arg("fn"),
      py::arg("param1") = 0,
      py::arg("param2") = true);
```

### Rules
- **Double precision only** — always instantiate templates with `<double>`
- Index parameters are **0-based** (same as C++)
- Always pass all output pointers (no `nullptr` optimization) since all outputs are returned to Python
- `int` and `bool` parameters are passed directly (no conversion needed)
- Use py::tuple when outputs are a fixed, small set (less than 7 values) of scalars/arrays with a natural ordering. Use py::dict when outputs are numerous (more than 6) or the user would typically access them by name. If unsure, default to py::dict for better readability and maintainability.

---

## FILE 4: Python Test (`test_<name>.py`)

### Framework: `unittest`

```python
import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def test_v4(self):
        fn = os.path.join(current_dir, '../data/test.qrt')
        # ... tests ...

    def test_v5(self):
        fn = os.path.join(current_dir, '../data/test_v5.qrt')
        # ... tests ...


if __name__ == '__main__':
    unittest.main()
```

### Assertion Patterns

```python
# Exact match
self.assertEqual(no_orig, 3)
self.assertEqual(len(names), 3)
self.assertEqual(names[0], "TX1")

# Shape check
self.assertEqual(cir_pos.shape, (no_cir, 3))

# Floating point (absolute tolerance)
npt.assert_allclose(actual, expected, atol=1.5e-4, rtol=0)

# Single value
npt.assert_allclose(fGHz[0], 3.75, atol=1e-6, rtol=0)

# Array comparison
T = np.array([-12.9607, 59.6906, 2.0])
npt.assert_allclose(np.asarray(data["tx_pos"]), T, atol=1.5e-4, rtol=0)
```

### What to Test
- Mirror the C++ test structure closely
- Import the appropriate submodule based on the function's section.
- Data paths relative to `current_dir`: `os.path.join(current_dir, '../data/<file>')`
- Indices are **0-based** in Python
- For functions that don't operate on files, generate synthetic test data programmatically within the test.

---

## FILE 5: MATLAB MEX Wrapper (`<name>.cpp`)

### Includes

```cpp
#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"
```

### MEX Function Pattern

```cpp
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > N)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    std::string fn = qd_mex_get_string(prhs[0]);
    arma::uword idx = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "param_name", 1);
    bool flag = (nrhs < 3) ? true : qd_mex_get_scalar<bool>(prhs[2], "flag_name", true);
    int option = (nrhs < 4) ? 1 : qd_mex_get_scalar<int>(prhs[3], "option_name", 1);

    // IMPORTANT: Convert 1-based MATLAB indices to 0-based C++ indices
    idx -= 1;

    // Declare output variables
    arma::mat out_mat_fixed;
    arma::vec out_vec;
    // ...

    // Alternative zero-copy: if output size is known, and output is requested
    // plhs[1] = qd_mex_init_output(&out_vec, n);

    // Set up optional output pointers based on nlhs
    arma::vec *p_out_vec = (nlhs > 0) ? &out_vec : nullptr;

    // For known output size, allocate zero-copy output arrays
    size_t r = 10, c = 5; // Example length, replace with actual expected size
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&out_mat_fixed, r, c);

    // Call library function (double precision only)
    CALL_QD(quadriga_lib::function_name<double>(fn, idx, flag, &out_mat_fixed, p_out_vec, option));

    // Write to MATLAB
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&out_vec);
}
```

### Adapter Functions Available (`mex_helper_functions.hpp`)

**Macro**

| Macro | Purpose |
|---|---|
| `CALL_QD(expr)` | Wraps expression in try/catch, forwards C++ exceptions to MATLAB via `mexErrMsgIdAndTxt` |

**MATLAB → C++ (scalar / string input)**

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `qd_mex_get_scalar<T>` | `(const mxArray*, var_name="", default=NAN)` | `T` | Casts any MATLAB numeric to `<T>`. Returns default for empty. |
| `qd_mex_get_string` | `(const mxArray*, default="")` | `std::string` | Reads MATLAB char array |

**MATLAB → Armadillo (zero-copy reinterpret, MATLAB type must match)**

| Function | Returns | Notes |
|---|---|---|
| `qd_mex_reinterpret_Col<T>(input, copy=false)` | `arma::Col<T>` | Zero-copy if `copy=false` |
| `qd_mex_reinterpret_Mat<T>(input, copy=false)` | `arma::Mat<T>` | |
| `qd_mex_reinterpret_Cube<T>(input, copy=false)` | `arma::Cube<T>` | 4D → Cube (dims 3+4 merged) |

**MATLAB → Armadillo (type-casting copy, any numeric input)**

| Function | Returns | Notes |
|---|---|---|
| `qd_mex_typecast_Col<T>(input, name="", n_elem=0)` | `arma::Col<T>` | Optional element count validation |
| `qd_mex_typecast_Mat<T>(input, name="")` | `arma::Mat<T>` | |
| `qd_mex_typecast_Cube<T>(input, name="")` | `arma::Cube<T>` | |

**MATLAB → Armadillo (convenience shortcuts, auto zero-copy or typecast)**

| Function | Returns | Notes |
|---|---|---|
| `qd_mex_get_double_Col(input, copy=false)` | `arma::vec` | Zero-copy if input is double |
| `qd_mex_get_single_Col(input, copy=false)` | `arma::fvec` | Zero-copy if input is single |
| `qd_mex_get_double_Mat(input, copy=false)` | `arma::mat` | |
| `qd_mex_get_single_Mat(input, copy=false)` | `arma::fmat` | |
| `qd_mex_get_double_Cube(input, copy=false)` | `arma::cube` | |
| `qd_mex_get_single_Cube(input, copy=false)` | `arma::fcube` | |

**MATLAB → std::vector (splitting along a dimension)**

| Function | Returns | Notes |
|---|---|---|
| `qd_mex_matlab2vector_Col<T>(input, vec_dim)` | `std::vector<arma::Col<T>>` | Split along dim `vec_dim` (0-based) |
| `qd_mex_matlab2vector_Mat<T>(input, vec_dim)` | `std::vector<arma::Mat<T>>` | |
| `qd_mex_matlab2vector_Cube<T>(input, vec_dim)` | `std::vector<arma::Cube<T>>` | |
| `qd_mex_matlab2vector_Bool(input)` | `std::vector<bool>` | Any numeric/logical → bool |

**MATLAB → std::any (generic)**

| Function | Returns | Notes |
|---|---|---|
| `qd_mex_anycast(input, name="", copy=false)` | `std::any` | scalar→native, col→`arma::Col`, row→`arma::Mat(1,N)`, 2D→`arma::Mat`, 3D/4D→`arma::Cube` |

**Armadillo / C++ → MATLAB (copy to mxArray)**

| Function | C++ input | MATLAB output | Notes |
|---|---|---|---|
| `qd_mex_copy2matlab(const T*)` | Scalar pointer (`double`, `float`, `int`, `unsigned`, `unsigned long long`, `arma::uword`) | `mxArray*` scalar | |
| `qd_mex_copy2matlab(const arma::Row<T>*)` | Row vector | `mxArray*` `[1, N]` | |
| `qd_mex_copy2matlab(const arma::Col<T>*, transpose, ns, is)` | Column vector | `mxArray*` `[N, 1]` or `[1, N]` | Optional column subset via `ns`/`is` |
| `qd_mex_copy2matlab(const arma::Mat<T>*, ns, is)` | Matrix | `mxArray*` 2D | Optional column subset |
| `qd_mex_copy2matlab(arma::Cube<T>*, ns, is)` | Cube | `mxArray*` 3D | Optional slice subset |
| `qd_mex_copy2matlab(const std::vector<std::string>*)` | String vector | `mxArray*` cell array of char | |
| `qd_mex_copy2matlab(const std::vector<bool>*)` | Bool vector | `mxArray*` logical | |

Supported types for `T`: `double`, `float`, `unsigned`, `int`, `unsigned long long`, `long long`

**std::vector<Armadillo> → MATLAB (stack into higher-dim array)**

| Function | C++ input | MATLAB output | Notes |
|---|---|---|---|
| `qd_mex_vector2matlab(const std::vector<arma::Col<T>>*, ns, is, padding)` | Vector of cols | `mxArray*` 2D | Extra dim = vector index |
| `qd_mex_vector2matlab(const std::vector<arma::Mat<T>>*, ns, is, padding)` | Vector of mats | `mxArray*` 3D | Padded to largest element |
| `qd_mex_vector2matlab(const std::vector<arma::Cube<T>>*, ns, is, padding)` | Vector of cubes | `mxArray*` 4D | |

**Armadillo ← MATLAB (zero-copy output initialization)**

| Function | Allocates | Maps to | Supported types |
|---|---|---|---|
| `qd_mex_init_output(&Row, n_elem)` | `mxArray*` `[1, N]` | `arma::Row<T>` | `float`, `double`, `unsigned` |
| `qd_mex_init_output(&Col, n_elem, transpose=false)` | `mxArray*` `[N, 1]` | `arma::Col<T>` | `float`, `double`, `unsigned` |
| `qd_mex_init_output(&Mat, n_rows, n_cols)` | `mxArray*` 2D | `arma::Mat<T>` | `float`, `double`, `unsigned` |
| `qd_mex_init_output(&Cube, n_rows, n_cols, n_slices)` | `mxArray*` 3D | `arma::Cube<T>` | `float`, `double`, `unsigned` |

Usage: `plhs[0] = qd_mex_init_output(&my_mat, nr, nc);` — writing to `my_mat` writes directly to MATLAB memory.

**MATLAB Struct Helpers**

| Function | Purpose |
|---|---|
| `qd_mex_make_struct(fields, N=1)` | Create empty `1×N` struct with field names |
| `qd_mex_set_field(strct, field, data, n=0)` | Set field of struct element |
| `qd_mex_has_field(strct, field) → bool` | Check if field exists |
| `qd_mex_get_field(strct, field) → mxArray*` | Get field (throws if missing) |

### Cell Array Output Pattern (for `std::vector<arma::mat>`)

```cpp
if (nlhs > N)
{
    mwSize n_items = vec_of_mat.size();
    plhs[N] = mxCreateCellMatrix(1, n_items);
    for (mwSize i = 0; i < n_items; i++)
        mxSetCell(plhs[N], i, qd_mex_copy2matlab(&vec_of_mat[i]));
}
```

### MATLAB Documentation

Use `/*!MD ... MD!*/` with MATLAB-style usage and 1-based indexing:

```
## Usage:
\```
[ out1, out2, out3 ] = quadriga_lib.function_name( fn, idx, flag );
\```
```

**CRITICAL: MATLAB indices are 1-based.** Document parameters as 1-based. In the `mexFunction`, subtract 1 before calling the library.

### Rules
- **Double precision only** — instantiate templates with `<double>`
- Use qd_mex_get_double_Col, qd_mex_get_double_Mat, qd_mex_get_double_Cube to automatically manage reinterpret or typecast based on input type (e.g. single → double)
- Indices are **1-based** in the MATLAB interface, converted to 0-based in `mexFunction`
- Use `CALL_QD(...)` macro to wrap the library call (handles exceptions)
- All optional outputs governed by `nlhs`
- All optional inputs governed by `nrhs` with defaults
- If output size is known before calling the library, prefer `qd_mex_init_output` to allocate zero-copy output arrays and pass pointers to the library for direct writing
- In the documentation, distinguish between "Input Arguments" and "Output Arguments" sections, and clearly specify the expected MATLAB types and shapes for each parameter.
- Only define input data types when they are fixed (i.e. other types would throw an error, e.g. strings). qd_mex_get_scalar always casts to the requested type, so the library function should be robust to receiving different numeric types (e.g. int, double). The "qd_mex_get_double_*" function variants handle varying input types gracefully, so the library should be robust to receiving any type as long as it can be cast to double. 
- Always state the output types and shapes in the documentation

---

## FILE 6: MATLAB Test (`test_<name>.m`)

### Framework: MOxUnit-style function

```matlab
function test_function_name()
% MOxUnit tests for quadriga_lib.function_name

fn = 'data/test.qrt';

% --- Test Section Name ---
[out1, out2, out3] = quadriga_lib.function_name(fn);

assertEqual(out1, uint64(3));           % Exact match (arma::uword → uint64)
assertEqual(out2, int32(4));            % Exact match (int → int32)
assertEqual(numel(cell_out), 3);        % Count elements

assertEqual(cell_out{1}, 'TX1');        % String in cell array

% Floating point comparisons
assertElementsAlmostEqual(actual, single(3.75), 'absolute', 1e-6);   % Single precision
assertElementsAlmostEqual(actual, expected_double, 'absolute', 1.5e-4); % Double precision

% Shape checks
assertEqual(size(mat_out), [double(no_rows), 3]);

end
```

### Type Mapping (C++ → MATLAB)

| C++ type | MATLAB type |
|---|---|
| `arma::uword` | `uint64` |
| `int` | `int32` |
| `arma::fvec`, `arma::fmat` | `single` |
| `arma::vec`, `arma::mat`, `arma::cube` | `double` |
| `arma::uvec` | `uint64` |
| `std::vector<std::string>` | cell array of `char` |

### Rules
- Indices are **1-based**
- `assertEqual` for exact matches (use correct MATLAB type)
- `assertElementsAlmostEqual` with `'absolute'` tolerance for floats
- Column vectors in MATLAB: TX pos is `[3, 1]` (compare to `[-12.9607; 59.6906; 2.0]`)
- Test data path is relative: `'data/test.qrt'` (if test data is needed, should be provided in the request)
- Test all relevant outputs, including scalars, vectors, matrices, and strings. Validate shapes and sample values.
- Test failure cases (e.g. invalid file paths, out-of-range indices) - they should throw errors that can be caught with `assertError` or `assertErrorThrown`.
- For functions that don't operate on files, generate synthetic test data programmatically within the test.

---

## IMPORTANT CROSS-CUTTING RULES

### Index Convention Summary

| Interface | Index base | Example: first element |
|---|---|---|
| C++ | 0-based | `i_cir = 0` |
| Python | 0-based | `cir=0` |
| MATLAB | 1-based | `i_cir = 1` (converted to 0 inside MEX) |

### Precision Summary

| Interface | Template type | Notes |
|---|---|---|
| C++ implementation | `<dtype>` (float + double) | Explicit instantiation for both |
| Python wrapper | `<double>` only | All outputs are double numpy arrays |
| MATLAB MEX | `<double>` only | MATLAB default is double |

### Vector Orientation

| Context | Orientation | Example |
|---|---|---|
| C++ Armadillo | Column vectors | `arma::vec pos = {x, y, z}` → size `[3, 1]` |
| Python numpy | 1D arrays | `pos.shape = (3,)` |
| MATLAB | Column vectors | `pos = [x; y; z]` → size `[3, 1]` |

For array sizes, always follow the header declaration.

---

## WHAT YOU RECEIVE AS INPUT

1. Optional **C++ header declaration(s)** — the function signature(s) with comments describing each parameter
2. **Optionally**: an example implementation, a description of the algorithm, or test data values
3. A section name to derive the header files and Python submodule from

If only a reference implementation is provided, you must reverse-engineer the declaration and documentation from it. The reference function may already contain documentation that you can reuse. If this is not possible, ask for a header declaration with parameter descriptions before proceeding.

## WHAT YOU PRODUCE

Six files (or a subset specified in the request), each complete and ready to drop into the codebase:

| # | File | Purpose |
|---|---|---|
| 1 | `qd_<module>_<topic>.cpp` | C++ implementation with documentation |
| 2 | `test_<name>.cpp` | Catch2 tests |
| 3 | `qpy_<name>.cpp` | pybind11 Python wrapper with documentation |
| 4 | `test_<name>.py` | Python unittest tests |
| 5 | `<name>.cpp` (MEX) | MATLAB MEX wrapper with documentation |
| 6 | `test_<name>.m` | MATLAB MOxUnit tests |

Plus: the `m.def(...)` pybind11 declaration and an updated header declaration (if documentation can be improved).

Plus: C++ header declaration with comments for each parameter (if not provided in the request or if the provided declaration is incomplete/unclear). Try to improve the documentation in the header as much as possible, since this is the primary reference for users and developers. 

---

## EXAMPLE: Complete Input → Output Mapping

**Input (header):**
```cpp
void qrt_file_parse(const std::string &fn,                          // Path to the QRT file
                    arma::uword *no_cir = nullptr,                  // Number of channel snapshots per origin point
                    ...);
```

**Output file 1 (C++ impl)** contains:
- License → `/*!SECTION` → `/*!MD` docs with full declaration + arguments + example → implementation → (template instantiation if templated)

**Output file 2 (C++ test)** contains:
- License → `#include <catch2/catch_test_macros.hpp>` → `TEST_CASE` blocks testing parse + read + cache + normalization + uplink/downlink

**Output file 3 (Python wrapper)** contains:
- License → `#include "python_arma_adapter.hpp"` → `/*!MD` Python docs → wrapper function → `m.def(...)` declaration

**Output file 4 (Python test)** contains:
- License → unittest class → test methods mirroring C++ tests with 0-based indices

**Output file 5 (MEX wrapper)** contains:
- License → `#include "mex.h"` → `/*!MD` MATLAB docs → `mexFunction` with `nlhs`/`nrhs` handling, 1-based index conversion

**Output file 6 (MATLAB test)** contains:
- Function with `assertEqual` / `assertElementsAlmostEqual`, 1-based indices, correct MATLAB types

---

## CHECKLIST BEFORE DELIVERING

- [ ] All 6 files have the correct license header
- [ ] C++ implementation has `/*!MD` documentation with Declaration, Arguments (all params), Example
- [ ] Template functions have explicit float + double instantiation at end of file
- [ ] Python wrapper uses `<double>` only, returns `py::tuple` or `py::dict`
- [ ] Python wrapper includes `m.def(...)` declaration
- [ ] MATLAB MEX converts 1-based indices to 0-based
- [ ] MATLAB MEX uses `CALL_QD(...)` macro
- [ ] MATLAB test uses correct types (`uint64`, `int32`, `single`, `double`)
- [ ] All tests cover: basic functionality, edge cases, uplink/downlink if applicable, multi-frequency if applicable
- [ ] Stream/cache patterns followed if function takes `std::ifstream*`
- [ ] No `nullptr` optimization in Python wrapper (all outputs always computed)
