# Task: apply the user's decisions on spec r2

Working directory `/home/stephan/sjc/quadriga-lib-fixes`. Edit `md/gbsm_lsf_spec.md` (r2 → r2.1). The options are
prepared in place (`[pending user decision R2-nn]` notes); see also `md/reports/spec_r2_revise.md`.

Decisions (binding):
1. **R2-11: `Comb256` stays the default preset.** State the reason in the spec requirements: the exponential ACF of
   TR 7.4.4 is given "with sufficient accuracy" as a fit; its kink at zero distance makes the values evolve roughly
   along mobility tracks, while `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance) gives
   smooth evolution. `Exp256` gives the literal TR ACF and is one argument away. Resolution: `user: Comb256 default`.
2. **R2-12: option A.** One sentence: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` are
   permanent coding-rule files of the repository (not working documents) and are binding for the implementation.
   Resolution: `user: SPEC files allowed`.
3. **R2-16: option A, drop `vs_ratio`** from the C++, MEX and Python APIs, its validation and tests; the caller sets
   `DS_VS` in its scenpar copy. Resolution: `user: drop vs_ratio`.

Remove the three pending notes. Write only the spec, the three resolution cells and a short report
`md/reports/spec_r2_user_apply.md`. No code, no build. Final answer: at most 8 lines.
