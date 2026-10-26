# Report: user decisions applied to spec r2 (r2 -> r2.1)

Task: `md/prompts/spec_apply_user_r2.md`. Changed `md/gbsm_lsf_spec.md` (2039 -> about 1990 lines) and the
resolution cells of R2-11, R2-12, R2-16 in `md/gbsm_lsf_findings.md`. No code, no build.

## Decisions applied

- **R2-11, `Comb256` stays the default.** 1.2 bullet "ACF of the LSF generators" now carries the reason: the TR
  gives the exponential ACF as a fit "with sufficient accuracy"; its kink at zero distance makes the values evolve
  roughly along tracks; `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance) evolves
  smoothly; `"Exp256"` gives the literal TR ACF. Defaults in 2.1, 6.1 and 6.2 unchanged.
- **R2-12, option A.** One sentence in the header after "Where this spec and a SPEC file disagree ...": the three
  SPEC files are permanent coding-rule files of the repository, not working documents, and binding for the
  implementation. No other edit.
- **R2-16, option A, `vs_ratio` dropped.** Edits as prepared: 1.2 bullet (caller sets `DS_VS` in its scenpar copy,
  reason: an overriding argument would duplicate the key); 2.1 declaration; 3.0.4 V/S term from `DS_VS` only; 3.0.5
  `qd_gbsm_lsp_mu_sigma` signature; 3.3.3; 3.4.1 item 3 and the function-specific message removed; 3.4.2 step 3;
  3.4.4; 4.4.1 comment line 3 ("set DS_VS for another hall."); 5.4 V/S case, InF TR spot check and validation row;
  6.1 usage line, defaults, nrhs 5..8 and MEX test point; 6.2 Python signature.

## Beyond the prepared edit lists

- 3.4.1 items 4 to 6 renumbered 3 to 5; the reference "3.4.1 item 5" in 3.4.3 changed to "item 4".
- Header: status r2.1, the paragraph on open user decisions replaced by "No user decision is open."; both
  `[pending user decision]` pointers under 2.1 removed with the three notes.
- Section 12: new row r2.1.
