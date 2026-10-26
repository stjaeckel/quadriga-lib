# Task: critical review of the implementation spec, round 1

You are an independent reviewer. Working directory `/home/stephan/sjc/quadriga-lib-fixes` (branch `fixes`).
Review `md/gbsm_lsf_spec.md` (draft r0) and find everything that is wrong, missing, inconsistent, bloated or not
implementable autonomously. You did not write it; be critical. Read the writer's report
`md/reports/spec_r0_writer.md` (40 writer choices W-1..W-40, 8 issues I-1..I-8) and `md/gbsm_lsf_findings.md`.

Sources to check the spec against: TR 38.901 v19.5.0 (`/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`),
the QuaDRiGa-NG MATLAB code (`/home/stephan/sjc/QuaDRiGa-NG/quadriga_src`, read-only), the quadriga-lib code
(`include/`, `src/qd_scenpar.cpp`, `src/sos_*.cpp`, `api_mex/`, `api_python/`, `tests/`), and the convention
files `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.

## Review guidance from the user (binding for this review)

1. **Decisions are not binding.** The D-decisions of the plan were partly made on a hunch. If a decision is a
   blocker or a clearly better solution exists, raise it as a finding with the alternative and its trade-off.
2. **Review every writer choice W-1..W-40 critically**: is it sensible, why was it made, what is the alternative
   and its trade-off (accuracy / performance / correctness / code size)? Give a verdict per choice: keep / change
   (to what) / escalate to the user (only if it cannot be resolved cleanly). Also judge I-1..I-8 and propose a
   resolution for each.
3. **QuaDRiGa-NG is not authoritative.** It was never scanned for bugs. Where it conflicts with 38.901, the spec
   should follow 38.901, unless the QuaDRiGa choice produces correct statistics and has a good reason (then it may
   stay as an option). Use judgement.
4. **36.873 is deprecated** (early version of 38.901, not updated since 2018): compatibility has low priority;
   flag any spec effort spent on it that is not worth it.
5. **Precision:** the functions are `double`; the SOS evaluation may run in single precision to use the AVX2
   kernel (`sos_val` float instantiation); the loss of accuracy is acceptable. Check that the spec allows this
   where it pays off and states the resulting test tolerances.
6. **No references to `md/` documents** in any produced source, test, wrapper or conf file. They must stand on
   their own and may cite 38.901 or other published material only. Internal tracking IDs (D-, W-, I-numbers) stay
   in the spec and internal docs. Flag every place where the spec would put them into code.
7. **Lean code, no bloat.** Reuse existing helpers (e.g. the Cholesky in `src/qd_scenpar.cpp`, the SOS functions,
   the MEX/Python helpers, `quadriga_lib_helper_functions.hpp`). Clean structure, readable for humans and AI: given
   only a source file, its purpose and working must be clear. Flag duplication and over-engineering.
8. **Lean public API** that fits the existing code base: only what a caller needs; implementation details stay
   out unless needed to use the API correctly. Flag every argument, output or model that is not needed or could be
   simpler (e.g. many bool flags, rarely used outputs, model zoo of `o2i_loss`/`los_probability`).
9. **Naming and scope:** clear names. A helper used once is a lambda inside the caller; helpers reused within one
   cpp are `static inline` or in an anonymous namespace; only helpers used across several files go into a private
   header, which then needs its own documentation. Flag violations, incl. the planned
   `src/quadriga_gbsm_helpers.hpp` — is it justified, and what exactly belongs in it?
10. **Cholesky / sqrtm / BLAS:** BLAS/LAPACK is disabled today. The user would consider enabling it if it saves
    hundreds of lines. Assess: how many lines the hand-written versions really need (8x8 Cholesky, symmetric
    matrix square root, inverse normal CDF, anything else), which BLAS/LAPACK routines would replace them
    (names), what enabling LAPACK costs (linking on Linux/Windows/MSVC, MEX against MATLAB's LAPACK, Python
    wheels, CI), and alternatives (C++ standard library, header-only libraries, existing code in the repo, an
    algorithm that avoids sqrtm entirely). Also question whether the `sqrtm` legacy option is needed at all.
11. **MSVC:** the code must compile with MSVC (not available here); flag anything in the spec that would not
    (GCC-only constructs, VLAs, OpenMP > 2.0, `M_PI` without `_USE_MATH_DEFINES`, designated initializers, ...).
12. **Conf keys:** every new key must be fully integrated into the reader and the writer (key table entry,
    default, validation class, writer section and comment, written only when it differs from the default),
    with tests. Check the spec specifies all of this exactly.
13. **SSF readiness:** check section "Requirements from the later SSF port" against
    `@qd_builder/gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`, `add_sdc.m` and TR 7.5 steps 5-12,
    7.6.2, 7.6.3.1, 7.6.13-7.6.16. The LSF module must provide everything the SSF port needs (SSF design itself is
    out of scope). Flag what is missing.
14. **Autonomous implementability:** every function must be specified without open ends (sizes, validation,
    edge cases, error messages, defaults); tests must have independent oracles and justified tolerances; the run
    split must fit one agent context per run.
15. **Correctness against the TR:** spot-check the formulas, constants and per-file conf values against the TR
    tables (path loss, LOS probability, O2I, LSP tables, ZSD/ZOD tables, abs TOA). Report every mismatch.

## Output

- Append your findings to the table in `md/gbsm_lsf_findings.md`: ID `R1-nn`, round 1, severity
  (blocker / major / minor / nit), spec section, finding with the proposed resolution; leave the resolution column
  empty (the reviser fills it). Mark findings that need the user's decision with `[escalate]` in the finding text.
- Write your full report to `md/reports/spec_r1_review.md`: the per-W verdict table, the I-1..I-8 resolutions, the
  BLAS assessment (point 10) with line estimates, SSF gaps, and a short overall judgement.
- Do not edit the spec, the plan, or any code. Do not build.
- Final answer to the coordinator: at most 25 lines — counts per severity, the blockers in one line each, the
  escalations in one line each, and the BLAS recommendation in two lines.
