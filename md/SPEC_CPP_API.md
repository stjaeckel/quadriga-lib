You are implementing a function of my C++ radio channel modelling library (quadriga-lib). You will receive the contract of the function as a declaration in one of the public headers in `include/`, together with a reference implementation (e.g. in MATLAB) and/or a verbose description of what the function should do. Your task is to add the implementation in `src/` and a Catch2 test in `tests/catch2_tests/`, following the test-first workflow below. The library is the numerical core behind the MATLAB/Octave and Python APIs, so correctness, a stable interface and performance all matter.

# GENERAL CONVENTIONS
- All functions and classes live in the `quadriga_lib` namespace. Language standard is C++17.
- Armadillo is used for all array data. Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.
- Template parameter `dtype` is `float` or `double` unless stated. Templates are explicitly instantiated in the source file; nothing is implemented in the public headers.
- `arma::uword`, `size_t` and `unsigned long long` are the same 64-bit type (enforced by `static_assert` in the headers) and can be used interchangeably.
- Complex data is passed as separate real and imaginary arrays (`_re` / `_im`), or interleaved along the rows where documented (e.g. the polarization transfer matrix with 8 rows). `arma::cx_*` types are not used in interfaces.
- Index conventions: 0-based unless the field is explicitly called "1-based" (typically indices where 0 means "none", e.g. `mtl_ind`, `fbs_ind`, `sbs_ind`, QDANT `id`).
- Units: angles in radians (degrees only where stated, e.g. `*_deg`, `*_3dB`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.
- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.
- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).
- The project has grown over several years. Older files deviate from this document in naming, documentation layout and style. For new code follow this document and the recent files named as examples; do not copy older conventions, and do not restyle existing files unless asked.

# REPOSITORY LAYOUT AND BUILD
- `include/` — public headers, the contract:
  - `quadriga_lib.hpp` — version functions, channel generation, the float-only ray-tracing pipeline (`ray_init`, `ray_progress`, `ray_commit`, ...); includes the four headers below
  - `quadriga_tools.hpp` — site-specific simulation tools, channel statistics, miscellaneous tools
  - `quadriga_math.hpp` — math functions
  - `quadriga_arrayant.hpp` — `arrayant` class and array antenna functions
  - `quadriga_channel.hpp` — `channel` and `path` classes, channel functions
- `src/` — implementations (`*.cpp`) and internal headers (`*.hpp`, `*.h`); internal headers are not installed and may be included by tests that test internals.
- `tests/catch2_tests/` — Catch2 (v3) tests, one file per function or topic; `tests/quadriga_lib_catch2_tests.cpp` holds `main`.
- `tests/data/` — input files for tests.
- Build system: CMake globs `src/*.cpp` and `tests/catch2_tests/*.cpp` (with `CONFIGURE_DEPENDS`, not recursive). A new source or test file needs no registration. Files in subfolders of `tests/catch2_tests/` are not compiled.
- File name suffixes select special compilation: `src/*_avx2.cpp` is compiled with `-mavx2 -mfma` (only if AVX2 is enabled), `src/*_cuda.cu` with nvcc (only if CUDA is enabled). Do not use these suffixes for anything else.
- Compile definitions for the library: `BUILD_WITH_AVX2` and `BUILD_WITH_CUDA` (0 or 1), `ARMA_DONT_USE_BLAS`, `ARMA_DONT_USE_LAPACK`, `_USE_MATH_DEFINES`.
  - Armadillo functions that need LAPACK (`inv`, `solve`, `eig_*`, `svd`, `det`, `pinv`, ...) are NOT available. Matrix products work but are not BLAS-accelerated. Write small linear algebra by hand.
- The library is compiled with `-O3 -Wall -Wextra -Wpedantic`; new code must compile without warnings. The tests are compiled with `-w`.
- The code must also compile with MSVC: no GCC-only extensions, OpenMP 2.0 only (see "OpenMP").
- Commands (Linux, from the repository root):
  - `make test` — configures with tests enabled, builds, and runs the complete Catch2 suite (`build_linux/test_bin`)
  - `cmake --build build_linux --parallel` — rebuild after the tests have been enabled once
  - `build_linux/test_bin "my_function*"` — run only the test cases whose name matches; this is the normal way to iterate on one function
  - Tests are run from the repository root; relative paths such as `tests/data/x.obj` depend on it.

# WORKFLOW
The work is test-first and proceeds in the phases below, normally within one session: write the tests first and expect them to fail, then implement, then run the tests until they are green, iterating if issues arise. Do not skip ahead: in particular, do not start the implementation before the tests exist and their oracles have been evaluated.

There are three review points at which you stop, report, and wait for my answer before continuing:
- **Review point A** (end of Phase 1) — only if there are open questions about the contract.
- **Review point B** (end of Phase 3) — always: the tests and the oracle evaluation, before any implementation.
- **Review point C** (end of Phase 5) — always: the final result.

## Phase 0 — What you receive
- The declaration in a public header in `include/`, with a comment per argument. This is the contract.
- A reference implementation (e.g. MATLAB) and/or a description of the intended behavior.

## Phase 1 — Read the contract
- Read the declaration, the reference and the neighbouring functions it builds on. Their documentation is collected in `html_docu/quadriga_lib_api_cpp.md` (read-only, generated; see "Fixtures and oracles" for how to use its line index); the sources of the documentation blocks are in `src/*.cpp`.
- Do not change the declaration (names, order, types, defaults). If the contract is ambiguous, contradicts the reference, or cannot be implemented as written, stop and ask.
- Settle before writing code: sizes of all outputs as a function of the inputs; which arguments are optional and what omitting them means; what is validated and what is undefined; behavior for empty inputs; units and index bases.
- **Review point A:** if anything in this list cannot be settled from the contract and the reference, list the open questions with your proposed answer for each, and wait. If everything is clear, state the decisions briefly and continue.

## Phase 2 — Stub that compiles
- Create `src/<function_name>.cpp` with: license statement, includes, section statement, a first version of the documentation block, the function definition, and the explicit template instantiations.
- The stub body does no computation. It only touches every argument (so that there are no unused-parameter warnings), sizes every requested output to its final shape and fills it with zeros, and returns a default value.
- Build the library. The stub must compile and link without warnings. This proves the interface, the includes and the instantiations before any logic exists.

## Phase 3 — Tests from first principles
- Create `tests/catch2_tests/test_<function_name>.cpp` (see "TESTS").
- Derive the expected values from first principles: closed-form results, hand-computed small cases, conservation laws, symmetries, limiting cases, and oracles that are implemented independently in the test file. Use the reference implementation as a source of expected numbers where it is trusted, not as the only oracle.
- Do NOT derive expectations from the implementation, and do not call library internals to build an oracle for the function under test: a shared bug would hide.
- Add diagnostic output (see "Diagnostics") so that the numbers behind the assertions can be inspected.
- Build and run only the new test cases against the stub. Then evaluate the oracles before implementing:
  - Checks on values are expected to fail against the stub. Confirm that each one fails for the right reason (zeros from the stub), not because of a mistake in the test.
  - Check the oracles against each other and against the reference: do the independent expectations agree, do limiting cases reduce to known results, are tolerances realistic for `float` and `double`?
  - Identify which expectations are uncertain or conflict with the contract or the reference.
- **Review point B:** stop and report before implementing. The report contains:
  - the list of test cases with one line each on what is asserted and which oracle is used
  - the result of the run against the stub: which checks fail as expected, and any that pass or fail unexpectedly
  - the diagnostic tables that matter for judging the oracles
  - the tolerances chosen and why
  - open points: uncertain expectations, conflicts between oracles, contract or reference, and behavior the tests do not cover
- Wait for my review. Changes I request to the tests are made and re-run against the stub before Phase 4 starts.

## Phase 4 — Implementation
- Replace the stub body with the implementation (see "IMPLEMENTATION GUIDELINES"). Input validation first, then the computation.
- Iterate with `build_linux/test_bin "my_function*"` until the new tests pass. If a test fails, first decide whether the implementation or the expectation is wrong; never adjust an expectation or a tolerance just to make a test pass without understanding the difference.

## Phase 5 — Go live
- All new tests pass, for `float` and `double`.
- The complete suite passes (`make test`); the library compiles without warnings.
- Diagnostic output is still enabled (see "Diagnostics"); I switch it off myself after the review.
- The documentation block describes what the function actually does, including the decisions made in Phases 1 to 4.
- **Review point C:** report the final state: test results (new cases and complete suite, with the actual output of the run), compiler warnings if any, every expectation or tolerance that was changed after review point B and why, deviations from the contract or the reference, and known limitations. Do not report a test as passing that was not run.
- MEX and Python wrappers are separate tasks (`md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`) and are only written on request.

# PUBLIC HEADER (THE CONTRACT)
You normally receive the declaration; these are the conventions it follows, for reading it correctly and for the cases where you are asked to propose one.
- Declarations are grouped under a comment line such as `// ---- Channel statistics ----` in the header that fits the topic.
- A comment block above the declaration gives a one-line description and, as `// - ...` lines, the facts a caller needs.
- Each argument is on its own line with a trailing comment: meaning, unit, index base, and size, e.g. `// Ray origins in GCS, [n_ray, 3]`. In/out and optional arguments are marked in the comment.
- Argument order: required inputs, optional inputs, outputs, then scalar options; later extensions are appended at the end with a default so that existing callers keep working.
- Argument passing — the rule for new declarations:
  - **Mandatory** inputs and outputs (and in-place updated data) are passed by **reference**: `const arma::Mat<dtype> &input`, `arma::Mat<dtype> &output`. No `nullptr` check is needed or possible.
  - **Optional** inputs and outputs are passed by **pointer**, default-initialized to `nullptr`: `const arma::Mat<dtype> *input = nullptr`, `arma::Mat<dtype> *output = nullptr`. A `nullptr` input means "not provided"; a `nullptr` output is not allocated and not computed.
  - Many existing declarations (most of `quadriga_tools.hpp`) pass mandatory arguments by pointer as well. That is the older style: follow a given declaration as it is, but if a mandatory argument of a *new* contract is declared as a pointer, point it out at review point A.
- Results are returned through output arguments; the return value is used for a single natural result (a count, a mesh, a status vector) and documented under "Returns".
- Default values exist only in the header, never in the definition.
- Functions that are not templates use fixed types (e.g. the `float`-only ray-tracing pipeline).

# IMPLEMENTATION GUIDELINES

## Source file
- One public function per file, named after the function: `quadriga_lib::function_name` is implemented in `src/function_name.cpp`. Closely related variants of one function (e.g. `_multi`, `_vec`) may share the file. Member functions of the `arrayant`, `channel` and `path` classes go into the existing `qd_arrayant*.cpp`, `qd_channel*.cpp`, `qd_path.cpp` files.
- File layout, in this order:
  1. License statement
  2. Includes: the public header that declares the function first, then internal headers, then standard headers
  3. File-local helpers (see below)
  4. Section statement
  5. Documentation block
  6. Function definition
  7. Explicit template instantiations

## License statement
- At the top of every source, header and test file:
```
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.
```

## Includes
- Include the narrowest public header that declares the function: usually `#include "quadriga_tools.hpp"`; `#include "quadriga_lib.hpp"` if the function is declared there or several public headers are needed.
- Include every standard header whose names are used (`<stdexcept>`, `<cmath>`, `<algorithm>`, `<limits>`, `<vector>`, ...); do not rely on transitive includes.
- Internal headers in `src/` with reusable helpers — read them before writing a helper of your own:
  - `quadriga_lib_helper_functions.hpp` — small numeric helpers (`qd_calc_length`, `crossp`, `dotp`, `qd_rotation_matrix`, `qd_in_range`, ...)
  - `quadriga_lib_material_helpers.hpp` — material property map access and the material model (`mtl_validate`, `mtl_col`, `mtl_val`, ...)
  - `qd_atmospheric_absorption.hpp`, `bits.hpp`, `slerp.h`
  - `quadriga_lib_generic_functions.hpp`, `quadriga_lib_avx2_functions.hpp`, `quadriga_lib_cuda_functions.hpp` — declarations of the compute kernels

## Function definition
- Define with the qualified name, without reopening the namespace and without `using namespace`:
```
template <typename dtype>
void quadriga_lib::function_name(const arma::Mat<dtype> *input, arma::Mat<dtype> *output, arma::uword n)
{
    ...
}
```
- No default arguments in the definition.
- End the file with explicit instantiations for `float` and `double` (one per overload), spelling out the full argument list:
```
template void quadriga_lib::function_name(const arma::Mat<float> *input, arma::Mat<float> *output, arma::uword n);
template void quadriga_lib::function_name(const arma::Mat<double> *input, arma::Mat<double> *output, arma::uword n);
```

## File-local helpers
- Helpers used only in this file are `static` (or in an anonymous namespace) and placed before the documentation block of the public function. Prefix them with `qd_` to avoid clashes with other headers.
- A helper that is needed by more than one source file goes into one of the internal headers as `static inline`; ask before adding a new internal header or a new public function.

## Input validation
- Validate first, compute second: all checks run before any output is resized or written, so that a failed call leaves the outputs untouched.
- Throw `std::invalid_argument` for invalid input (null, empty, wrong shape, out-of-range value, inconsistent sizes). Use `std::runtime_error` for I/O and other runtime failures. `std::out_of_range` is used for index arguments in a few places; prefer `std::invalid_argument` unless the contract says otherwise.
- Message style: a full sentence naming the argument in single quotes, e.g.
  - `"Input 'orig' cannot be NULL."`
  - `"Input 'mesh' cannot be empty."`
  - `"Input 'mesh' must have 9 columns."`
  - `"Input 'dest' must have size [n_ray, 3]."`
  - `"Number of elements in 'fbs_ind' does not match number of rows in 'orig'."`
  - `"Output 'center' cannot be NULL."`
- Check in this order per argument: not empty, shape, value range; then the consistency between arguments. Mandatory arguments passed by reference need no null check; a mandatory argument that an existing declaration passes by pointer is checked for `nullptr` first.
- Validation must reject input that would otherwise cause undefined behavior (out-of-bounds indices into the mesh or the material table, zero where a division follows). Do not validate what is cheap to tolerate and expensive to check, unless the contract asks for it.
- Parameters that are unused in a build configuration are silenced with `(void)name;` inside the matching `#if`.

## Outputs
- Mandatory outputs are references and are always computed. Optional outputs are pointers: when one is `nullptr`, skip both its allocation and the computational steps that only serve that output.
- If a quantity belonging to an omitted optional output is needed internally (as an intermediate for another output), compute it in a variable that is local to the function; never allocate or write through the caller's pointer for it, and do not compute it at all when nothing needs it.
- The same holds for optional inputs: a `nullptr` input selects the documented default behavior; a default object that is needed internally is local to the function.
- Resize an output only if its size differs from the required size, then write into it in place:
```
if (out != nullptr && (out->n_rows != n_ray || out->n_cols != 3))
    out->set_size(n_ray, 3);
```
  - Reason: the MEX and Python wrappers pre-allocate outputs in MATLAB / numpy memory and map Armadillo objects onto that memory. An unconditional `set_size`, `zeros(n, m)`, or assignment of a differently sized object would detach the output from that memory or throw. With the size check, a correctly pre-sized output is written zero-copy.
  - Consequence: every element of an output must be written by the function (it may hold arbitrary data on entry), and the output size must be derivable from the inputs and documented.
- Do not read from an output before writing it, and do not assume that distinct arguments point to distinct objects unless the contract says so; if aliasing an input with an output is not supported, say so in the documentation.
- In/out arguments (documented as "updated in-place") keep their size; validate it like an input.

## Numerics and types
- Access array data through raw pointers (`memptr()`, `colptr(i)`, `slice_memptr(i)`) in hot loops; Armadillo element access `(i, j)` and expression templates are fine for setup code. Remember the column-major layout when striding.
- Sizes and counters are `arma::uword`. Indices stored in data use the type from the contract (`unsigned` / `arma::u32_vec`, `short`, `uint8_t`). Cast explicitly when mixing types; the compiler warnings are enabled.
- For `dtype = float`, do intermediate math in `double` where cancellation or accumulation matters and cast the result once: `dtype(x)`. Write literals so that no unintended double-precision math enters a float kernel (`(dtype)0.5`, `1.0f`).
- Never compare floating-point results for equality in the logic unless the contract defines an exact sentinel (e.g. 0 = disabled).
- No global or static mutable state; functions must be re-entrant. Random numbers only through an explicit seed argument.
- No output to `std::cout` from library code, except where the contract has a `verbose` argument.

## OpenMP
- Parallelize only loops that dominate the run time and are independent per iteration. Guard nothing with `#ifdef _OPENMP` unless OpenMP runtime functions are called; a plain `#pragma omp parallel for` is ignored by a compiler without OpenMP.
- The loop variable of a parallel `for` must be a signed integer type (`int` or `long long`) for MSVC; cast to `arma::uword` inside the loop.
- Exceptions must not leave a parallel region. Wrap throwing calls with the `OMP_SAFE_CALL` macro from `quadriga_lib.hpp` and rethrow after the region:
```
bool omp_failed = false;
std::string omp_error;
#pragma omp parallel for
for (long long i = 0; i < n_iter; ++i)
{
    OMP_SAFE_CALL(omp_failed, omp_error, do_work((arma::uword)i));
}
if (omp_failed)
    throw std::runtime_error(omp_error);
```
- The result must not depend on the number of threads (same values, same order). If a parallel and a serial path exist, a test compares them.

## AVX2 and CUDA kernels
- New functions are implemented in portable C++ first. Vectorized or GPU kernels are reserved for the most performance-critical sections and are only added on request, after benchmarking.
- Where they exist, the layout is: public function in `function_name.cpp` validates, converts and dispatches; kernels are free functions `qd_XXX_GENERIC` / `qd_XXX_AVX2` / `qd_XXX_CUDA` in `function_name_generic.cpp` / `function_name_avx2.cpp` / `function_name_cuda.cu`, declared in the `quadriga_lib_*_functions.hpp` headers; calls are guarded by `#if BUILD_WITH_AVX2` / `#if BUILD_WITH_CUDA` and a runtime check; a `use_kernel` argument selects 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA and throws if the requested kernel is unavailable. See `ray_triangle_intersect.cpp`. All kernels must produce the same results within the documented tolerance, and a test compares them.

## Section statement
- Functions are grouped into sections for the generated documentation. Place the block before the documentation block:
```
/*!SECTION
Site-specific simulation tools
SECTION!*/
```
- Section names in use: "Array antenna class", "Array antenna functions", "Channel class", "Channel functions", "Channel generation functions", "Channel statistics", "Math functions", "Miscellaneous / Tools", "Site-specific simulation tools". The name must match exactly to end up in the same group. If the section is not clear from the header, ask.

## Documentation block
- The block is placed immediately before the function definition. It starts with `/*!MD` and ends with `MD!*/`, each on its own line (the delimiters are matched literally by the documentation tools).
- It is the reference documentation of the function: it is compiled to HTML and Markdown (`make documentation`), and it is the input from which the MEX and Python wrappers and their documentation are derived. Sizes, units, index bases, defaults and optionality must therefore be complete and exact.
- Layout:
```
/*!MD
# function_name
One-line summary

- Description as a bullet list, one fact per bullet
- ...

## Declaration:
```
return_type quadriga_lib::function_name(
    const arma::Mat<dtype> *input,
    arma::Mat<dtype> *output = nullptr,
    arma::uword n = 1);
```

## Inputs:
- **`input`** — Description; further facts; `[n_rows, n_cols]`
- **`n`** *(optional)* — Description

## Outputs:
- **`output`** *(optional)* — Description; `[n_rows, 3]`

## Returns:
- Description of the return value; `[size]`

## See also:
- [[other_function]] (short reason)
MD!*/
```
- First line: `# ` and the function name as declared (member functions of a class: `# .method_name`). Second line: the one-line summary, which is reused verbatim by the wrappers.
- The description follows after one empty line as a bullet list, without a `## Description:` heading. Say what the function does and what a caller must know: algorithm in one or two bullets, conventions, edge cases, what is and is not validated, what happens for empty or omitted arguments. Do not narrate the implementation.
- `## Declaration:` repeats the declaration from the header including the default values, without the per-argument comments and without the `template` line.
- `## Inputs:`, `## Outputs:`, `## Returns:`; use the singular (`## Input:`, `## Output:`) when there is exactly one; omit a section that would be empty. In/out arguments are listed under Inputs and marked "updated in-place" (or under Outputs when they are primarily results).
- Argument format: ``- **`name`** *(optional)* — Description; facts separated by semicolons; `[size]` ``
  - Mark every argument that has a default value or accepts `nullptr` with `*(optional)*` (the default itself is visible in the Declaration) and state what omitting it means.
  - Give the size in backticks, with the symbolic dimension names used consistently across all arguments; define each dimension name where it first appears.
  - State the unit when it is not SI, the index base for every index argument, and the meaning of sentinel values (0, NaN, negative).
- `## See also:` lists related functions as `[[name]]` with a short reason; `[[...]]` becomes a link.
- Never use `*` or `|` in formulas or running text — they are markdown syntax. Use `·` for multiplication and `abs()` for absolute values.
- Lines should not exceed 120 characters (soft limit); indent continuation lines by 2 spaces.
- Example of a recent, complete block: `src/ray_commit.cpp`; a small one: `src/cube.cpp`.

# TESTS

## File and structure
- One file per function: `tests/catch2_tests/test_<function_name>.cpp`. Tests of internal kernels that include internal headers go into `test_<function_name>_internal.cpp`.
- File layout: license statement; a header comment (see below); `#include <catch2/catch_test_macros.hpp>`; the public header; standard headers; fixtures and oracles; test cases.
- Header comment: a short statement of what this suite pins down and what it leaves to other suites, where the oracles come from, and the conventions of the fixtures (geometry, materials, units). See `test_ray_commit.cpp`.
- All test files are linked into one executable. Every helper function, struct and constant in a test file must be `static` or inside an anonymous namespace, and test case names must be unique across the whole suite.
- Test case names: `"function_name - what is asserted"`, written as a statement about behavior, e.g. `"ray_commit - a receiver outside the beam tube is not committed"`. Tags are not used.
- One behavior per `TEST_CASE`. Use `SECTION`s for variations that share a setup (e.g. one section per argument in the validation test).

## Fixtures and oracles
- Keep it lean. Separate two kinds of code in a test file:
  - **Oracles** for the test object: these you build yourself, in the test file, independently of the library.
  - **Supporting tasks** (scene geometry, intersections that feed the function under test, antenna patterns, coordinate conversions, file I/O, ...): use the library. It is fine and preferred to call library functions that are not related to the current test objective, e.g. `quadriga_lib::cube` to generate geometry or `quadriga_lib::ray_triangle_intersect` to produce the hit indices for a function that consumes them. They have their own tests; do not re-implement them in the test file.
- A complete overview of all available C++ functions is in `html_docu/quadriga_lib_api_cpp.md`. Use it to find helpers before writing one.
  - The file is generated by `make documentation` from the documentation blocks in `src/`; never edit it. It may lag behind the sources until the documentation is rebuilt.
  - It starts with general usage notes and a "Function Index" table with the columns Function, Section and Line. The Line column is the line number at which the entry starts, so read the index first (about the first 150 lines) and then extract only the entries you need by line range instead of reading the whole file.
- Build the smallest input that isolates the behavior: one ray, one face, one path. Exact-by-construction geometry (axis-aligned, unit distances) gives expectations that need no tolerance discussion.
- Put shared setup into a few small helper functions or a fixture struct at the top of the file. Write a helper only when it is used more than once or makes a test case readable; no helper layers, no generality that no test uses.
- Oracles: closed-form formulas, direct transcription of the governing equations in scalar `double` math, brute-force evaluation of what the function computes cleverly. Comment each oracle with the formula and its range of validity. An oracle must not call the function under test or the internals it is built on.
- Prefer checks that are independent of conventions and hard to satisfy by accident: energy conservation, reciprocity, symmetry, invariance under subdivision / reordering / partitioning, agreement of two independent code paths, known limiting cases.
- Use distinctive inputs (non-trivial values, distinct values per element, permuted index maps) so that a swapped argument, an off-by-one or cross-talk between elements changes the result visibly.

## Assertions
- `CHECK` for the assertions under test (the case continues and reports all failures); `REQUIRE` only for preconditions without which the rest of the case would crash or be meaningless (output sizes before indexing).
- Arrays: `CHECK(arma::approx_equal(A, B, "absdiff", tol));` (or `"reldiff"`). Scalars: `CHECK(std::abs(a - b) < tol);`. Integers, indices and flags are compared exactly. The Catch2 `Approx` class is not used in new tests.
- Choose tolerances from the arithmetic, not from the observed error: near machine precision for exact-by-construction `double` results (1e-12 to 1e-14), a few `float` epsilons scaled by the magnitude for `float`, and the documented model accuracy where an approximation is tested. State the reason in a comment when a tolerance is loose.
- Inside loops, add `INFO("angle " << a << ", type " << t);` before the checks so that a failure reports the case.
- When `nullptr` is passed for a templated pointer argument and `dtype` cannot be deduced from another argument, cast it: `(arma::mat *)nullptr`.

## What every suite covers
- The values, against the oracles, for the normal cases and for every branch the contract distinguishes.
- Output sizes and types for all outputs.
- Optional arguments: omitted versus provided; each optional output requested alone and together with others gives the same values.
- Edge cases named in the contract: empty optional inputs, a single element, sentinel values, degenerate geometry.
- Pre-sized outputs: an output that already has the correct size is filled completely; an output with a wrong size or stale content is resized and overwritten.
- `float` and `double`: both instantiations run; a parity case compares them within `float` accuracy.
- Determinism: repeated calls give identical results; parallel and serial paths agree; where the function processes data in blocks, one case crosses the block size.
- Input validation, in one test case named `"function_name - input validation"`: a valid baseline with `CHECK_NOTHROW`, then one section per argument in which exactly one thing is made invalid and `CHECK_THROWS_AS(call, std::invalid_argument)` is asserted. Pairing every negative case with the passing baseline shows that the guard is not over-eager.
- Tests that write files use distinct file names in the working directory and remove them at the end of the case; input files come from `tests/data/`.

## Diagnostics
- Diagnostic output makes the numbers behind the assertions visible while the oracles and the implementation are being developed. It is controlled by one switch at the top of the test file:
```
// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif
```
- Print tables in CSV form with a `# diag:` title line and a header line, one row per case, with the computed value, the oracle value and their difference:
```
QD_DIAG("\n# diag: thin sheet composite vs analytic slab, 1 GHz\n");
QD_DIAG("material,angle,sheet_dB,ana_dB,d_dB\n");
...
QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f\n", name, angle, a, b, a - b);
```
- Two kinds of test cases print diagnostics:
  - Regular cases print the compared quantities next to their `CHECK`s.
  - Pure diagnostic cases, named `"Diagnostic - ..."`, exist to produce numbers for a question that is still open. They assert only sanity (finite, non-negative, energy not created), so they stay green while the divergence they measure is being investigated. Explain in a comment what the table shows and how to read it.
- New test files are written with the switch enabled, and it stays enabled through all phases including the hand-over at review point C. I switch it off myself once I am satisfied that the tests pass; do not comment it out unless asked. The diagnostic cases and the `QD_DIAG` lines stay in the file for later investigations.
- The code must compile and pass in both states of the switch: nothing but printing may depend on it.
- See `tests/catch2_tests/test_ray_mesh_interact.cpp` for the pattern.

# REFERENCE SKELETONS

Stub of Phase 2 for a function declared in `quadriga_tools.hpp` as

```
// Short description
template <typename dtype>
arma::uword my_function(const arma::Mat<dtype> &points,     // Points in GCS, [n_point, 3]
                        arma::Col<dtype> *dist = nullptr,   // Optional output: Distance to the origin in [m], [n_point]
                        dtype scale = 1.0);                 // Scaling factor
```

`src/my_function.cpp`:

```
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <stdexcept>

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# my_function
Short description

- First draft of the description; completed in Phase 5

## Declaration:
```
arma::uword quadriga_lib::my_function(
    const arma::Mat<dtype> &points,
    arma::Col<dtype> *dist = nullptr,
    dtype scale = 1.0);
```

## Inputs:
- **`points`** — Points in GCS; `[n_point, 3]`
- **`scale`** *(optional)* — Scaling factor

## Output:
- **`dist`** *(optional)* — Distance to the origin in [m]; `[n_point]`

## Returns:
- Number of points, `n_point`
MD!*/

template <typename dtype>
arma::uword quadriga_lib::my_function(const arma::Mat<dtype> &points,
                                      arma::Col<dtype> *dist,
                                      dtype scale)
{
    // STUB: no computation yet
    (void)scale;

    if (points.n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns.");

    arma::uword n_point = points.n_rows;

    if (dist != nullptr)
    {
        if (dist->n_elem != n_point)
            dist->set_size(n_point);
        dist->zeros();
    }

    return n_point;
}

template arma::uword quadriga_lib::my_function(const arma::Mat<float> &points, arma::Col<float> *dist, float scale);
template arma::uword quadriga_lib::my_function(const arma::Mat<double> &points, arma::Col<double> *dist, double scale);
```

`tests/catch2_tests/test_my_function.cpp`:

```
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::my_function.
//
// Pins down: the distance values, the scaling, the output size, the optional output.
// Oracle: Euclidean norm computed in scalar double math in this file.
// Conventions: points are rows {x, y, z} in meters.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>
#include <stdexcept>

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    // Oracle: scale · sqrt(x² + y² + z²)
    double ref_dist(double x, double y, double z, double scale)
    {
        return scale * std::sqrt(x * x + y * y + z * z);
    }
}

TEST_CASE("my_function - distances match the Euclidean norm")
{
    arma::mat points = {{3.0, 4.0, 0.0}, {0.0, 0.0, 2.0}, {1.0, 2.0, 2.0}};
    arma::vec dist;

    arma::uword n = quadriga_lib::my_function(points, &dist, 2.0);

    CHECK(n == 3);
    REQUIRE(dist.n_elem == 3);

    QD_DIAG("\n# diag: my_function distances\n");
    QD_DIAG("row,dist,ref,diff\n");
    for (arma::uword i = 0; i < 3; ++i)
    {
        double ref = ref_dist(points(i, 0), points(i, 1), points(i, 2), 2.0);
        QD_DIAG("%llu,%.12f,%.12f,%.3e\n", (unsigned long long)i, dist(i), ref, dist(i) - ref);
        INFO("row " << i);
        CHECK(std::abs(dist(i) - ref) < 1e-14);
    }
}

TEST_CASE("my_function - input validation")
{
    arma::mat points(2, 3, arma::fill::ones);
    arma::vec dist;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::my_function(points, &dist));
        CHECK_NOTHROW(quadriga_lib::my_function(points)); // optional output omitted
    }

    SECTION("points")
    {
        arma::mat bad(2, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::my_function(bad, &dist), std::invalid_argument);
    }
}
```
