# Task: critical review of the implementation spec, round 2

You are an independent reviewer; you did not write the spec and did not do round 1. Working directory
`/home/stephan/sjc/quadriga-lib-fixes` (branch `fixes`). Review `md/gbsm_lsf_spec.md` (r1.1) and find everything
that is wrong, missing, inconsistent, bloated or not implementable autonomously.

Read first `md/gbsm_lsf_findings.md` (round-1 findings and their resolutions). Do not re-raise a resolved finding
unless its resolution is wrong or incomplete; then say so with the finding ID. Decisions marked `user:` in the
resolution column are the user's and are not reopened.

Sources: TR 38.901 v19.5.0 (`/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`), QuaDRiGa-NG
(`/home/stephan/sjc/QuaDRiGa-NG/quadriga_src`, read-only), the quadriga-lib code (`include/`, `src/`, `api_mex/`,
`api_python/`, `tests/`) and `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.

## Review guidance from the user (binding)

1. The spec must **stand on its own**: an implementer needs only the spec, the source code and TR 38.901. Flag any
   reference to other internal documents or earlier revisions in normative text, and any internal ID that would
   end up in code, headers, tests, wrappers or conf files.
2. Where QuaDRiGa-NG conflicts with the TR, follow the TR unless the QuaDRiGa choice gives correct statistics for
   a good reason. 36.873 and mmMAGIC are deprecated.
3. Functions are double; the SOS evaluation may run in single precision / AVX2 with matching tolerances.
4. Lean code and lean public API that fits the existing code base; reuse existing helpers; helpers used once are
   lambdas, reused in one file `static inline` / anonymous namespace, a private header only for cross-file helpers
   (documented). Clear names. MSVC-compatible. BLAS/LAPACK stays disabled.
5. New conf keys fully integrated into reader and writer (default, class, section, comment, not written at the
   default) with tests.
6. **SSF readiness**: the LSF module must provide everything the later small-scale port needs (check against
   `@qd_builder/gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`, `add_sdc.m`, TR 7.5 steps 5-12, 7.6.2,
   7.6.3.1, 7.6.13-7.6.16). Pay special attention here: round 1 found no API gaps — verify that claim
   independently, including per-frequency handling, the extensibility of the generator machinery for SSF
   generators, and what the builder must store.
7. **Autonomous implementability of each run** (R1, R2a, R2b, R3a, R3b, R4a, R4b): self-contained, fits one agent
   context, unambiguous sizes/validation/edge cases/errors, independent test oracles with justified tolerances,
   clear acceptance criteria.
8. **Correctness**: check the formulas and constants against the TR independently of round 1, especially the
   parts changed in r1/r1.1 (the reciprocity scheme of `lsf_draw`, bias removal, the sqrtm algorithm, the reduced
   `o2i_loss`, the `d2d_in` input of `path_loss`, LOS probability models, InF V/S). Recompute numbers where the
   spec states them.

## Output

- Append findings `R2-nn` (round 2, severity blocker / major / minor / nit, section, finding with proposed
  resolution, resolution column empty) to `md/gbsm_lsf_findings.md`; mark user decisions with `[escalate]`.
- Full report to `md/reports/spec_r2_review.md`: findings rationale, the SSF verification, a per-run
  implementability verdict, and an overall judgement (ready / ready after fixes / not ready).
- Do not edit the spec or code; do not build.
- Final answer: at most 20 lines — counts per severity, blockers and escalations one line each, overall verdict.
