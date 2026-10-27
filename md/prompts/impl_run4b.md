# Task: implementation Run 4b of the large-scale model port

Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes`. You implement **Run 4b** of the spec
`md/gbsm_lsf_spec.md` (status: final for implementation).

Read first: the spec sections 1, 2, 3 (the functions the Run 4b entry names), 6 (the Python parts and general rules), 5 (the parts for Run 4b), 8,
7 (the Run 4b entry: files, steps, acceptance, report), 9; then `md/SPEC_CPP_API.md`, and `md/SPEC_MATLAB_API.md`
and `md/SPEC_PYTHON_API.md` for the wrapper parts. The spec wins where it differs from the SPEC files (spec 1.4).

Rules:
- Do exactly what the spec's Run 4b entry lists, in its order, test-first. The review points A/B/C of the SPEC files
  are not stops: decide within the spec and record each decision in your report.
- If the spec contradicts itself, the code base, or cannot be implemented as written: do not guess. Implement
  nothing further in the affected part, describe the problem precisely in your report, and finish the rest.
- Constraints of spec section 9 apply in full (no internal references in produced files, MSVC-compatible code,
  no BLAS/LAPACK, no `std::to_chars` for floating point, objdump gate on the MEX files: nothing above
  GLIBCXX_3.4.28).
- Machine: shared. Check `cat /proc/loadavg` before builds; at most 12 threads (`--parallel 12`,
  `OMP_NUM_THREADS<=12`); never two builds or two test suites at once. No CUDA.
- MATLAB R2024a, R2026a and Octave can be run from the shell for the MEX tests; run the MEX suite in Octave, R2026a and
  R2024a at least.
- Do not commit. Do not touch QuaDRiGa-NG or the TR (read-only). Do not edit the spec or the findings log.
- Write the run report `md/reports/r4b_report.md` with the sections required by spec section 7, including the
  actual test output summaries (Catch2, MEX, Python), compiler warnings, the objdump result, every decision and
  deviation, and a proposed commit split (files per commit and a commit message per commit, without internal IDs
  or `md/` references in the messages).
- Final answer to the coordinator: at most 20 lines — done / not done per step, suite results (pass counts),
  spec problems found, files changed count.
- The helper header and the three shared helpers (floor offset, field evaluation, bias removal) must follow the
  fixed signatures of spec 3.0.5: they are reused by the later small-scale port.
- Runs 1 to 4a are committed (HEAD); start from a clean tree. Conf values follow the precision rule of spec 1.2 (source precision, derived values at most 4 significant digits unless the stated accuracy limits need more). Note: the Catch2 test binary links no BLAS; do not use Armadillo matrix products above 4x4 in tests.
- User decisions on three points from Run 4a (apply them; they are not yet in the spec, so update the spec text
  in 6.1 accordingly as part of this run):
  1. MEX floor IDs: the spec's test point that `uint8` floor IDs are accepted is dropped; the accepted types are
     those the existing MEX helpers read (double, single, int32/uint32, int64/uint64). Edit spec 6.1.
  2. Zero links from MATLAB: the MEX helpers turn `[0, 3]` into 0x0, so the wrappers reject zero links; this is
     accepted and noted in spec 6.1 (C++ supports zero links). Python: support zero links if the Python helpers
     keep the shape, otherwise document the same restriction.
  3. The C++ documentation block of `path_loss` uses `[X|v]` notation, which SPEC_CPP_API forbids; reword it as
     the MEX block does.
- Add to `md/SPEC_CPP_API.md` (tests section) a rule: the Catch2 test binary links no BLAS, so tests must not use
  Armadillo matrix products above 4x4 (write small loops instead).
- Python: build the module (`make python` or as the spec says) and run the full Python suite.
