# Task: write the implementation spec for the QuaDRiGa large-scale model port

You write `md/gbsm_lsf_spec.md` in `/home/stephan/sjc/quadriga-lib-fixes` (branch `fixes`). The spec must be
complete enough that implementation agents can build everything from it **autonomously, without asking
questions**: C++ core, Catch2 tests, MEX wrappers and tests, Python wrappers and tests, conf-file changes,
documentation. It will be reviewed critically by independent reviewer agents in three rounds, and then implemented.

## Inputs (read in this order)

1. `md/gbsm_lsf_plan.md` — the approved plan: decisions D1 to D32 (binding), gap analysis, API draft (section 6),
   work packages, integration hand-over. The spec replaces this plan as the single source of truth; carry every
   decision over (cite the D-numbers), do not reopen them.
2. `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` — binding conventions and workflow.
   Note one change of process: the review points A/B/C of these SPEC files are NOT stops for a human. The
   implementation agent decides within the spec and records every decision and deviation in its report.
3. Existing code to reuse: `include/quadriga_tools.hpp` (scenario-parameter declarations), `src/qd_scenpar.cpp`
   (key table, parser, writer), `include/quadriga_sos.hpp` and `src/sos_*.cpp`, the MEX and Python wrappers and
   tests of the conf functions and of the SOS functions (patterns to follow), `data/config/3GPP_38.901_v19_*.conf`.
4. The MATLAB reference (read-only): `/home/stephan/sjc/QuaDRiGa-NG/quadriga_src/@qd_builder/` (`qd_builder.m`,
   `init_sos.m`, `gen_lsf_parameters.m`, `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`, `get_channels.m`),
   `@qd_layout/set_scenario.m`, `@qd_layout/gen_o2i_loss.m`, `@qd_sos/qd_sos.m`.
5. TR 38.901 v19.5.0 (read-only): `/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`, clauses 7.2,
   7.4, 7.5 steps 1-4 and Tables 7.5-6 to 7.5-12, 7.6.3, 7.6.5, 7.6.8, 7.6.9, 7.8.4.

## What the spec must contain

1. **Scope and decisions**: goal, the D-table (carried over), out-of-scope list.
2. **Public API**: the exact declarations for `include/quadriga_gbsm.hpp` (the four moved functions and the six new
   ones), with per-argument comments as in SPEC_CPP_API. Resolve every point the plan leaves to "review point A"
   (section 7 of the plan: `rnd` columns and output shape of `o2i_loss` incl. mmMAGIC, 38.811 table handling,
   floor separation method). Nothing may remain open.
3. **Normative behaviour per function**: every formula (from the TR and from the MATLAB code, cited by clause /
   file and line), every clamp, default, edge case, validation rule and error message pattern, output sizes,
   behaviour of `nullptr` optionals, units. Write the formulas in a form that can be transcribed into code. Where
   the TR and the MATLAB code differ, state both and which flag selects which (D4).
4. **Conf-key changes**: the new numeric keys (table entries with class, writer section, comment) and the new
   `PL_` keys, and the exact values to put into each of the 23 v19 files.
5. **Tests**: per function a list of test cases (name, what is asserted, the oracle, tolerance and why), following
   SPEC_CPP_API. Oracles are TR formulas typed into the tests, independent of the scenpar mapping. Include the
   statistical tests for the random fields with tolerances derived from sample size.
6. **Wrappers**: MEX usage lines and Python signatures (submodule `gbsm`, D29) for all ten functions (four moved,
   six new), with their tests, per the wrapper SPEC files.
7. **Implementation runs**: split the work into 3 or 4 sequential, self-contained runs, each small enough for one
   agent context (suggested: R1 header move + new keys + Python submodule; R2 `lsf_init_phases`, `lsf_draw`,
   `lsp_generate`; R3 `path_loss`, `los_probability`, `o2i_loss`; R4 MEX + Python wrappers + documentation sweep).
   Per run: files touched, order of steps, acceptance criteria (which suites must pass), and the report it writes.
8. **SSF readiness**: a section "Requirements from the later SSF port". Read
   `@qd_builder/gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`, `add_sdc.m` and TR 7.5 steps 5-12 plus
   7.6.2, 7.6.3.1, 7.6.13 to 7.6.16, and list every quantity the small-scale step needs from the LSF module (e.g.
   per-link LSP means, KF in dB, EoD offset, soft LOS, cluster-number range, XPR mean/std, per-frequency
   handling, absolute TOA, ground permittivity, whether the LSF phase set must be extensible for SSF generators).
   For each: which spec function/output provides it, or what must be added. Add what is missing to the API.
9. **Constraints** (into every run): double only; C++17; no warnings; MSVC; OpenMP 2.0; no BLAS/LAPACK (no
   `arma::chol`, `eig_sym`, `sqrtmat`, `norm`, `dot`, `solve` — write the 8x8 Cholesky and Jacobi by hand); no
   `std::to_chars` for floating point (MATLAB R2024a, objdump gate on the MEX files: nothing above GLIBCXX_3.4.28);
   at most 12 threads, never two builds or test suites at once, check `/proc/loadavg`; quadriga-lib must not
   depend on QuaDRiGa-NG paths or data; NG and the TR read-only; agents do not commit (the user commits after
   each step). MATLAB R2024a, R2026a and Octave can be run from the shell for the MEX tests.
10. **Integration hand-over**: carry over section 10 of the plan (out of scope for the implementation runs).

## Rules for you

- Write only `md/gbsm_lsf_spec.md`, `md/gbsm_lsf_findings.md` (create it with a header and an empty findings table:
  ID, round, severity, section, finding, resolution) and your report. Do not change source code, tests, conf files
  or the plan. Do not build anything.
- Where you must make a choice the plan does not cover, make it, mark it in the spec as `[writer choice]` with a
  one-line reason, and list it in your report so the reviewer and the user can check it.
- If a decision of the plan turns out to be wrong or infeasible, do not silently change it: keep it, mark the
  problem as `[issue]` in the spec and list it in the report.
- Write your full report to `md/reports/spec_r0_writer.md`: sections written, every `[writer choice]` and `[issue]`,
  SSF requirements found, anything you could not verify.
- Your final answer to the coordinator: at most 25 lines — spec size, the run split, the count and one-line list of
  writer choices and issues, the top SSF requirements that changed the API. Nothing else.
