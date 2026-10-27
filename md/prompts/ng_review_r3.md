# Task: QuaDRiGa-NG integration readiness review (final review round)

You are an independent reviewer. Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes`.
The large-scale model of QuaDRiGa-NG (MATLAB, read-only: `/home/stephan/sjc/QuaDRiGa-NG/quadriga_src`) will be
replaced by the new quadriga-lib functions after the merge. Check whether **anything in the library blocks or
complicates that integration**, and whether the library provides everything the later small-scale (SSF) port
needs.

Library side: `include/quadriga_gbsm.hpp`, the MEX wrappers in `api_mex/` (`lsf_init_phases`, `lsf_draw`,
`path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`, `conf_file_*`, `scenpar_validate`), their docs and
tests; the integration hand-over in `md/gbsm_lsf_spec.md` section 10 and the SSF requirements in section 8.

QuaDRiGa-NG side (read all callers): `@qd_builder/` (`qd_builder.m` incl. `lsp_vals`, `init_sos.m`,
`gen_lsf_parameters.m`, `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`, `get_channels.m`, `gen_ssf_parameters.m`,
`gen_fbs_lbs.m`, `add_sdc.m`, `add_paths.m`, `copy.m`, `split_rx.m`, `split_multi_freq.m`, `check_los.m`,
`gen_lsf_from_ssf.m`), `@qd_layout/` (`set_scenario.m`, `gen_o2i_loss.m`, `power_map.m`, `get_channels.m`),
`@qd_simulation_parameters/`, `@qd_track` (`par.o2i_loss`, `o2i_d3din`, segments), the tutorials using these, and
the unit tests in `xunit_new/`.

For every place where NG uses the old LSF code, answer: which library call replaces it, with which arguments; is
the data available in the builder/layout at that point (positions as rows vs columns, frequencies, tracks with
segments, dual-mobility flags, multi-frequency builders, object arrays); what state the builder must store (phase
cube, preset, flags); which behaviour changes (3GPP defaults vs legacy flags; is every legacy behaviour reachable
by a flag for the parity test); performance concerns (call granularity per builder, per track, per snapshot; MEX
overhead for many small calls, e.g. `get_sf_profile` along long tracks); MATLAB/Octave specifics (types, 1-based
indices, empty inputs, zero links being rejected by the MEX wrappers).

Also check the SSF readiness: list what `gen_ssf_parameters.m`, `gen_fbs_lbs.m` and `get_channels.m` need from the
LSF step and confirm the library provides it (spec section 8 claims no API gap).

Rules: read-only; do not build (another reviewer owns the builds). You may run light probes in Octave or MATLAB
R2024a using the already built MEX files (`+quadriga_lib` in the repo) from your scratchpad; never write into
QuaDRiGa-NG. Write the full report to `md/reports/ng_r3_review.md` with a findings table (ID `N3-nn`, severity
blocker / major / minor / nit, file:line on either side, finding, proposed fix in the library or note for the
integration; `[escalate]` for user decisions) and a mapping table (NG function → library calls → status:
ready / ready with note / blocked). Final answer: at most 15 lines — counts per severity, blockers and majors one
line each, verdict on integration readiness.
