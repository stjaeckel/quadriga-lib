You are implementing a lightweight API wrapper for MATLAB mex to call my C++ radio channel modelling library (quadriga-lib). You will receive a documentation block for the C++ API. Your task is to implement the mex wrapper function in MATLAB that correctly calls the C++ function, handles input and output arguments, and ensures proper memory management. The mex function should be designed to be user-friendly for MATLAB users while maintaining the performance benefits of the underlying C++ library.

# CPP API DOCUMENTATION BLOCK FORMAT
- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.
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

# IMPLEMENTATION GUIDELINES
- The mex function should be named according to the C++ function it wraps, so quadriga_lib::function_name should be wrapped by a mex function named "function_name.cpp".
- All wrappers are called from the +quadriga_lib package which will be populated by the build system, so the user calls quadriga_lib.function_name from MATLAB, which maps to function_name.cpp in the source code. 
- The mexFunction in function_name.cpp should call the underlying C++ function quadriga_lib::function_name.
- All errors map to "quadriga_lib:CPPerror", `mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Error text.");`
- C++ functions may offer dtype = float / double specialization. The MATLAB mex wrapper only uses the double version, so you can hardcode the dtype to double in the call to the C++ function. 
- C++ indices are 0-based unless stated otherwise in the documentation, while MATLAB indices are 1-based. The mex wrapper should handle this conversion when passing indices between MATLAB and C++.
  - If CPP expects 0-based indices, the MATLAB user should provide 1-based indices, and the mex wrapper should convert them to 0-based before passing to C++.
  - If CPP expects 1-based indices, the MATLAB user should provide 1-based indices (no conversion needed).
  - If CPP outputs 0-based indices, the mex wrapper should convert them to 1-based before returning to MATLAB.
  - If CPP outputs 1-based indices, the mex wrapper should return them as-is to MATLAB.
- Armadillo types map to their c++ counterparts, e.g. arma::uword = unsigned long long (this is enforced at compile time by static_asserts).
- When a C++ parameter is both input and output (in-out pointer pattern), split it into two MATLAB names with _in / _out suffixes.

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

## Helper functions signatures
- Quadriga-Lib uses Armadillo for data structures, Armadillo headers are pulled in by quadriga_lib.hpp, so you can use Armadillo types directly in the mex wrapper
- Conversion from MEX to Armadillo can be done by calling the helper functions in mex_helper_functions.hpp. 

```
dtype qd_mex_get_scalar(const mxArray *input, std::string var_name = "", dtype default_value = dtype(NAN));
std::string qd_mex_get_string(const mxArray *input, std::string default_value = "");

arma::Col<dtype> qd_mex_get_Col(const mxArray *input, bool copy = false);
arma::Mat<dtype> qd_mex_get_Mat(const mxArray *input, bool copy = false);
arma::Cube<dtype> qd_mex_get_Cube(const mxArray *input, bool copy = false);

std::vector<bool> qd_mex_matlab2vector_Bool(const mxArray *input);

std::vector<arma::Col<dtype>> qd_mex_matlab2vector_Col(const mxArray *input, size_t vec_dim);
std::vector<arma::Mat<dtype>> qd_mex_matlab2vector_Mat(const mxArray *input, size_t vec_dim);
std::vector<arma::Cube<dtype>> qd_mex_matlab2vector_Cube(const mxArray *input, size_t vec_dim);

mxArray *qd_mex_copy2matlab(const dtype *input); // Scalar
mxArray *qd_mex_copy2matlab(const arma::Row<dtype> *input); // Row Vector

mxArray *qd_mex_copy2matlab(const arma::Col<dtype> *input, // Column Vector
                                   bool transpose = false,        // Transpose output
                                   size_t ns = 0,                 // Number of elements in output
                                   const size_t *is = nullptr);    // List of elements to copy, 0-based

mxArray *qd_mex_copy2matlab(const arma::Mat<dtype> *input, // Matrix
                                   size_t ns = 0,                 // Number of columns in output
                                   const size_t *is = nullptr);    // List of columns to copy, 0-based

mxArray *qd_mex_copy2matlab(arma::Cube<dtype> *input,   // Cube
                                   size_t ns = 0,              // Number of columns in output
                                   const size_t *is = nullptr); // List of columns to copy, 0-based

mxArray *qd_mex_copy2matlab(const std::vector<std::string> *strings); // Cell array of strings

mxArray *qd_mex_copy2matlab(const std::vector<bool> *bools); // Logical array (column vector)

mxArray *qd_mex_vector2matlab(const std::vector<arma::Col<dtype>> *input, 
        size_t ns = 0, const size_t *is = nullptr,d dtype padding = (dtype)0);

mxArray *qd_mex_vector2matlab(const std::vector<arma::Mat<dtype>> *input, 
        size_t ns = 0, const size_t *is = nullptr,d dtype padding = (dtype)0);

mxArray *qd_mex_vector2matlab(const std::vector<arma::Cube<dtype>> *input, 
        size_t ns = 0, const size_t *is = nullptr,d dtype padding = (dtype)0)

mxArray *qd_mex_init_output(arma::Row<dtype> *input, size_t n_elem);
mxArray *qd_mex_init_output(arma::Col<dtype> *input, size_t n_elem, bool transpose = false); 
mxArray *qd_mex_init_output(arma::Mat<dtype> *input, size_t n_rows, size_t n_cols);
mxArray *qd_mex_init_output(arma::Cube<dtype> *input, size_t n_rows, size_t n_cols, size_t n_slices); 
```

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
- Description should be adapted to MATLAB users, e.g. by removing C++-specific details and adding MATLAB-specific usage notes if needed. 
- It should be in bullet point format, with one concise fact per bullet
- Lines longer than 100 characters need to be split into multiple lines for better readability, and indent the continued lines by 2 spaces. 
- You can omit any dtype specifications in the documentation block, as they are not relevant for the MATLAB user. All C++ functions that have a dtype specialization are wrapped with the double version in MATLAB.
- Instead of the "## Declaration:" section, use a "## Usage:" section that includes the function signature as it should be called from MATLAB, such as:

    ## Usage:
    ```
    [ output1, output2 ] = function_name( input1, input2, optional_input3 );
    ```

- Split the usage block "..." into multiple lines if it exceeds 100 characters, and indent the continued lines by 4 spaces.
- Inputs and outputs should be described in separate sections "## Inputs:" and "## Outputs:", with the same one-liner style as in the C++ documentation block.
- Optional input arguments (C++: const type *name = nullptr) should be marked with "(optional)" directly after the argument name, they are also optional in MATLAB, and the user can simply omit them when calling the function, or provide them as empty arrays (e.g. `[]`) if they want to use the default behavior.
- Default input values must be added to the argument description as they are not obvious from the declaration, e.g. "; default: 0"
- Datatypes must only be specified when not `double`
- Unit must only be specified when not SI (meters, radians, Hz, linear scale), e.g. "angles in degrees", "power in dB"
- C++ returns get merged into the output arguments section
- Break lines (in arguments) longer than 100 characters into multiple lines for better readability, and indent the continued lines by 2 spaces.

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
- Always assume `copy = false` for input arguments and add const qualifier
- Adjust the name to match the c++ variable name, and the dtype to match the expected type in the C++ function (e.g. double)
- dtype can be one of: float, double, int, unsigned, long long, unsigned long long

### std::vector of Armadillo types
- The C++ function may expect a std::vector of Armadillo types (used for variable-length lists of arrays), e.g. `std::vector<arma::Mat<dtype>>`
- There are specific helper functions for these types, e.g.:
```
const auto data = qd_mex_matlab2vector_Col<dtype>(prhs[0], 1);
const auto data = qd_mex_matlab2vector_Mat<dtype>(prhs[0], 1);
const auto data = qd_mex_matlab2vector_Cube<dtype>(prhs[0], 1);
```
- dtype can be one of: float, double, int, unsigned, long long, unsigned long long
- The second argument indicates the dimension used for std::vector, 0-based
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
const auto name = qd_mex_get_string(prhs[2], "name");
```
- Same wrapping rules apply

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
- qd_mex_init_output has overloads for: Col, Row, Mat, Cube 
- Allowed data types: float, double, int, unsigned, long long, unsigned long long
- For qd_mex_init_output, the wrapper must know the output size before the C++ call - these should be clear from the C++ documentation block, e.g. if the output is described as [n_bins, n_sets], the wrapper can determine n_bins and n_sets from the input data dimensions and allocate accordingly.

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

## Pointer wrapper
- The c++ function might accept `nullptr` for optional inputs and outputs
- Generally, this avoids unnecessary allocations and copying when the user does not need the output or wants to use default values for the input. Also it might skip unnecessary computations in the C++ function when the output is not needed.
- Wrap all OPTIONAL c++ inputs and outputs to nullptr if the user does not provide them, e.g.:
```
arma::uvec *p_cluster_idx = (nlhs > 2) ? &cluster_idx : nullptr; // OR
arma::uvec *p_cluster_idx = cluster_idx.empty() ? nullptr : &cluster_idx;
```
- The cluster_idx.empty() variant is preferred, but it MUST NOT be used if the C++ expects an empty argument and resizes it depending on runtime conditions (e.g. reading from a file), as this would lead to unintended nullptr being passed. In this case, the (nlhs > X) guard should be used instead, which relies on the user explicitly providing an output argument when they want to receive the output.
- 
- If c++ does not has the nullptr option for optional arguments or arguments are passed as reference (= c++ mandatory outputs), then simply pass the empty Armadillo object, C++ will populate it as needed, and the wrapper can discard it if the user did not request the output (i.e. simply skip the qd_mex_init_output step for that output argument)

## Call the C++ function
- Now call the underlying C++ function with the converted input arguments and pointer-wrapped optional arguments
- Use the CALL_QD macro to call the function and catch any exceptions, e.g.:
```
CALL_QD(quadriga_lib::function_name(arg1, arg2, optional_arg3, optional_output4));
```
- After the C++ function call, copy all non-preallocated results to MATLAB memory as stated above

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

Complete MEX wrapper implementation for the `acdf` function based on the above documentation block, following the implementation guidelines provided:

// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# ACDF
Calculate the empirical averaged cumulative distribution function (CDF)

## Description:
- Computes per-column empirical CDFs by histogramming into bins and taking the normalized cumulative
  sum
- Averaged CDF is obtained by quantile-space averaging: for a fine probability grid, x-values from
  each column CDF are averaged, then mapped back to the bin grid
- Quantile statistics (mean and std) are reported at the 0.1, 0.2, ..., 0.9 probability levels
- `Inf` and `NaN` values are excluded from computation
- If `bins` is empty, equally spaced bins spanning the data range are generated

## Usage:
```
[ Sh, bins_out, Sc, mu, sig ] = quadriga_lib.acdf( data, bins_in, n_bins );
```

## Input Arguments:
- **`data`** — Input data matrix; each column is one independent data set; `[n_samples, n_sets]`
- **`bins_in`** *(optional)* — Bin centers; used as-is if non-empty; `[n_bins_in]`
- **`n_bins`** *(optional)* — Number of bins when auto-generating; must be >= 2; ignored when
  non-empty `bins_in` are provided

## Output Arguments:
- **`Sh`** *(optional)* — Individual CDFs; one per column of data; `[n_bins_out, n_sets]`
- **`bins_out`** *(optional)* — Auto-generated bins; copy of `bins_in` when
  non-empty `bins_in` are provided; `[n_bins_out = n_bins]` or `[n_bins_out = n_bins_in]`
- **`Sc`** *(optional)* — Averaged CDF via quantile-space averaging across data sets; `[n_bins]`
- **`mu`** *(optional)* — Mean of the 0.1–0.9 quantiles across data sets; `[9]`
- **`sig`** *(optional)* — Standard deviation of the 0.1–0.9 quantiles across data sets; `[9]`
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
    arma::mat Sh;
    arma::vec Sc, bins_out, mu, sig;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&Sh, n_bins_out, n_sets);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&Sc, n_bins_out);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&mu, 9);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sig, 9);

    // Special case for bins
    if (!bins_in.empty())
        bins_out = bins_in;

    // Wrap optional pointers
    arma::mat *p_Sh = Sh.empty() ? nullptr : &Sh;
    arma::vec *p_Sc = Sc.empty() ? nullptr : &Sc;
    arma::vec *p_mu = mu.empty() ? nullptr : &mu;
    arma::vec *p_sig = sig.empty() ? nullptr : &sig;

    // Call library function
    CALL_QD(quadriga_lib::acdf<double>(data, &bins_out, p_Sh, p_Sc, p_mu, p_sig, n_bins));

    // Copy to MATLAB
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&bins_out);
}