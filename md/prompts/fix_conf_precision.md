# Task: readable numbers in the conf files (fix to Run 1, spec update)

Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes`, uncommitted Run 1 changes in the tree.

User feedback: values like `DS_VS = 4.2727272727272725` in the v19 conf files are unreadable. Bit-identical
results are not required; reasonable accuracy is. Readability wins over 1e-7 dB accuracy.

Do:
1. **Rule in the spec** (`md/gbsm_lsf_spec.md`): add a binding requirement in 1.2: values in conf files are
   written with the precision of their source (as the TR prints them), and derived values with at most 4
   significant digits, unless more digits change a result by more than 0.01 dB (path loss, SF) or 0.1 % (LSP
   means, distances); state the reason (readability, accuracy far below model uncertainty). Apply the rule in
   section 4 (all listed conf values, incl. the values planned for Run 3a such as `40/3` / `20π/3` break-point
   factors and any other full-precision constants), and adjust any test tolerance in section 5 that depended on
   the exact value (derive the new tolerance from the rounding, with a comment). Remove the statements that
   called for full precision.
2. **Conf files**: round the values Run 1 wrote (`DS_VS` and any other long numbers added in this run) per the
   rule, e.g. 47/11 → 4.27, 50/11 → 4.55; keep the comments readable (they may say "47/11 for the 120 x 60 x 10 m
   hall").
3. **Tests**: update the Catch2 / MEX / Python tests that compare these values or derived quantities; tolerances
   from the rounding, not from observed errors.
4. Run the affected Catch2 tests and then the full Catch2 suite (resource rules: `cat /proc/loadavg`, at most 12
   threads, one build at a time), and the conf-related Python and MEX (Octave) tests.

Rules: no internal IDs or `md/` references in produced files; do not commit; append a short section "Precision fix"
to `md/reports/r1_report.md` (what changed, test results). Final answer: at most 10 lines.
