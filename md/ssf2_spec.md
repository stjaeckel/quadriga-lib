# SSF series 2: path generator (spec)

Implementation spec for the second small-scale fading series: from the clusters of `ssf_clusters` to the paths of `get_channels_planar`, `get_channels_spherical`, `get_channels_multifreq` and the QRT writer, plus a single-link channel function and the changes to existing GBSM functions. Working document; deleted before the merge, its outcome goes to `md/ssf_notes.md`.

## 1. Scope and status

Status: spec v1, written against plan v6 (`md/ssf2_plan.md`, commit 664e116) and the maintainer decisions S1–S6 on the plan's open items. The plan keeps the reasoning and the measurements (findings 1–18, decisions P1–P26); this spec supersedes the plan for every implementation detail. Where the spec gives a number, the source is named: a finding (F), a decision (P, S), a plan section, a code line, or a measurement made for this spec (section 6.1, "spec scratch").

The series delivers, in quadriga-lib only (no QuaDRiGa-NG code):

- the merged random-set functions `gbsm_init_phases` and `gbsm_draw` with the new set type "ray" (P22);
- the ray level of TR 38.901 v19.5.0 clause 7.5 steps 7–11 (antenna-free part), 7.6.1 (oxygen), 7.6.16 (polarisation power variability), the QuaDRiGa ray delays and rotation polarisation model, in `ssf_rays`, which writes one `gbsm_rays` object per link;
- the scatterer solver, the member `gbsm_rays::solve_scatterers` (wrappers: `ssf_scatterers`), the path assembly `ssf_paths` (mobility `phase` and `drift`), the QRT pair `qrt_paths_from_scatterers` / `gbsm_rays_from_paths`, and `gbsm_single_link_channel`;
- the changes to existing functions: merge of `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw`; `fc` → `center_frequency`; `SubpathMethod` removed from the API; zero-copy outputs in all GBSM wrappers; `ssf_clusters` doc sizes; `PerClusterDS_fc_min = 6` in the two v19 UMa files;
- MEX and Python wrappers of all of these except `qrt_paths_from_scatterers`, and their documentation.

Out of scope (readiness only, section 10): multi-frequency sample grids, TR Procedure A (7.6.3.2), TR 7.6.2.2 as a simulation option, the mobile-scatterer term of 7.6.10, a library-only TR 7.8.2 calibration, 7.6.4, 7.6.13, 7.6.14, tracks, segments and layouts.

## 2. Conventions

### 2.1 Placement, notation, units, codes (plan 5.1)

- **Placement.** All new functions and the class in `include/quadriga_gbsm.hpp`, doc section "Geometry-based stochastic model": `gbsm_init_phases`, `gbsm_draw` under a new header group `// ---- Random sets ----` before "Large-scale fading"; `gbsm_rays` (with its member `solve_scatterers`), `ssf_rays`, `ssf_paths`, `gbsm_rays_from_paths` in "Small-scale fading" after `ssf_clusters`. Exceptions: `qrt_paths_from_scatterers` in `include/quadriga_channel.hpp`, doc section "Channel functions"; `gbsm_single_link_channel` in `include/quadriga_lib.hpp` next to `get_channels_ieee_indoor`, doc section "Geometry-based stochastic model".
- **Notation.** N = `NumClusters` − 1 NLOS clusters, M = `NumSubPaths`, G = 1 if `GR_enabled` > 0 else 0. Sizes: `n_link` links, `n_slot` = `NumClusters` + G slots of `ssf_clusters`, `n_ray` rays of one link (1 + G + N·M from `ssf_rays`, incl. LOS, GR and zero-power padding; any count from other sources), `n_path` paths after zero-power rays are dropped, `n_freq` frequencies, `n_snap` snapshots, `n_tap` taps, `n_rx`, `n_tx` antenna ports, `n_gen` generators of a set, `n_coeff` sinusoids of a preset. No `n_fd` (P26).
- **Units and layouts.** Hz, s, m, rad, GCS, elevation angles; linear powers unless the name ends in `_dB`; c = 299792458 m/s. Batch positions are rows (`[n_link, 3]`, `[1, 3]`), one position is `[3]`, per-snapshot positions `[n_snap, 3]`; the `channel` class stores `[3, n_snap]`. Scatterer positions are columns `[3, n_ray]` / `[3, n_path]`. Delays are relative to the LOS delay |rx_pos − tx_pos|/c of the positions the ray data refer to, unless stated. Jones matrices: the 8-row layout of `get_channels_*`: rows 0–1 Re/Im θθ (Rx θ, Tx θ), 2–3 φθ (Rx φ, Tx θ), 4–5 θφ (Rx θ, Tx φ), 6–7 φφ; written J = [[J_θθ, J_θφ], [J_φθ, J_φφ]] with rows = Rx in this spec.
- **Codes.** Path type 0 = LOS, 1 = GR, 2 = NLOS. Tap index `tap` = 3·slot + i, i.e. LOS 0, GR 3, NLOS cluster n: 3·(1 + G + n) + i with sub-cluster i ∈ {0, 1, 2} for split clusters, i = 0 otherwise (always 0 in `quadriga`); taps are listed in ascending index. Both codes are defined once, in the section description; the wrappers never shift them (codes, not indices).
- **Argument order.** Required inputs, mandatory outputs, then all arguments with a default: options, optional inputs, optional outputs, flags; extensions are appended (`md/SPEC_CPP_API.md`).
- **Types and outputs.** Double only, no templates. Mandatory inputs and outputs by reference, optional ones by pointer with `nullptr` = not given / not computed. Outputs are resized only when their size differs, so pre-sized outputs (also the entries of `std::vector<gbsm_rays>`) are written in place; every element of an output is written. An attribute that a function sets empty is cleared only if it is not empty (`if (!x.empty()) x.reset();`), so an aliased wrapper attribute is never re-created.
- **Errors.** Invalid input throws `std::invalid_argument` with a full sentence that names the argument in single quotes (`"Input 'd_min' must be finite and > 0."`); class attributes are named `'rays.<attribute>'`. All checks run before any output is touched. The consumers of `gbsm_rays` call `is_valid()` first and throw its message unchanged. The messages of `scenpar_validate`, `arrayant::is_valid` and `qd_gbsm_check_links` pass through unchanged. The message tables in section 3 are exact; the runs keep the wording.
- **Wrappers.** MEX and Python for all functions of section 3 except `qrt_paths_from_scatterers`; the frequency argument is `center_freq` in all wrappers; Python names drop the repeated `gbsm_` prefix (`gbsm.init_phases`, `gbsm.draw`, `gbsm.rays_from_paths`, `gbsm.single_link_channel`), MATLAB keeps the full names (P23). MEX index rule: `set_id` 0 = shared set, k ≥ 1 = set of frequency k; `i_freq` and `ray_ind` 1-based; Python keeps the C++ 0-based values (`set_id` −1 = shared). Details: section 8.

### 2.2 Code rules for every run

- `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` apply in full; doc blocks follow the block grammar and writing pitfalls of `docs_builder/README.md`; `make docs` regenerates `docs/` (run by the manager once per merge).
- Produced code, tests, wrappers and conf files cite only published material (TR clauses and equations, papers, quadriga-lib.org pages) and the `md/SPEC_*.md` files; never this spec, the plan, `md/ssf_notes.md`, decision or finding ids (P…, S…, F…, R…), or run names.
- Lean code: a helper used once is a lambda; reuse library functions (`scenpar_validate`, `sos_*`, `get_channels_*`, `qrt_file_*`, `path` members, `quadriga_lib_helper_functions.hpp`); no BLAS/LAPACK calls (`inv`, `solve`, `norm`, `dot`, `accu(a % b)`, products above 4×4); MSVC must compile it (no VLAs, OpenMP 2.0, signed loop variables in parallel loops). Prefer fewer lines and wider tolerances over bit-identity where results may move within reasonable bounds; say why in a comment.
- Tests: `OMP_WAIT_POLICY=passive <build>/test_bin "<pattern>"` for targeted cases, from the run's own checkout (tests write files with fixed names); the full `make test` once at the end of a run. Supporting tasks in tests use library calls; oracles are built in the test file (or the shared test header) without the code under test.
- Shared machine: check `cat /proc/loadavg` before builds and tests; keep the load plus the run's threads near 32; numpy workers with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; timings only under `flock /tmp/bench.lock`, with the load before and after, provisional above load 4. MEX builds need g++-12 (`md/SPEC_MATLAB_API.md`).

### 2.3 Terms (P26, plan section 10)

The "Terms" paragraph of the GBSM section description defines these terms once (TR names where they exist, QuaDRiGa names marked); the API uses no "geometry set", "group" or "CIR":

| Term | Meaning |
|---|---|
| scenario | type of environment with its statistics: numeric keys `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx–Rx pair of a batch (row l of `rx_pos`) |
| drop | one realisation of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 7.5 steps 5–7) |
| slot | index in the `ssf_clusters` outputs: 0 LOS, 1 GR (if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (TR 7.5 step 11, Table 7.5-5) |
| ray | one of the M rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| path | a propagation path as `get_channels_*` and QRT take it; `ssf_paths` makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in the `taps` output: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`quadriga`); index `tap` |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (TR phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting) |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (`gbsm_rays.power`): the coefficient powers of TR 7.5-30 split over the rays, GR as `ssf_clusters` sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss (`ssf_paths`, `get_channels_*`, QRT) |
| link gain | −PL − O2I + SF in dB of a link at a position (`link_gain_dB`) |
| random set | the random values of one `set_type` and `set_id`, drawn from one phase array; never changes size |
| generator | one spatially consistent random field; one column of `rnd` |
| block | the generators of one random quantity in a set (N or N·M columns) |

## 3. Functions

Each subsection gives the declaration (plan section 5 verbatim; S1 adds `gbsm_rays::b_eff` and changes the validity lines of the `ssf_paths` and `gbsm_single_link_channel` header comments), the behaviour, the validation in the order of the checks with the exact messages, the edge cases, OpenMP and cost. Algorithms are in section 4, the set layouts in section 5.

### 3.1 `gbsm_init_phases` (replaces `lsf_init_phases`, `ssf_init_phases`; `src/qd_gbsm_phases.cpp`)

```
// Draw the random phases of one generator set of a BS site and scenario
// - Set types: "lsf" large-scale generators, "cluster" cluster random values, "ray" ray random values; generators
//   and columns: see gbsm_draw
// - The phases are a fixed function of seed, set_type and set_id, equal on all platforms; the sets are independent
// - "lsf": bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
void gbsm_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances; NumClusters; NumSubPaths for "ray")
                      const std::string &set_type,                            // "lsf", "cluster" or "ray"
                      arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, n_gen, 2]; slice 0 = Rx side, slice 1 = Tx side
                      arma::sword seed = -1,                                  // Seed of the site and scenario, the same for all set types; -1 = system random device
                      arma::sword set_id = -1,                                // "cluster", "ray": -1 = shared set, k >= 0 = set of frequency k (0-based); ignored for "lsf"
                      const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                      const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions for the bias removal, [1, 3] or [n_link, 3]; "lsf" only
                      const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions for the bias removal (2D: one floor), [n_link, 3]; "lsf" only
                      bool dual_mobility = false,                             // "lsf": Tx and Rx both move, the Tx side is a second device
                      bool remove_bias = true,                                // "lsf": remove the sample mean over the positions
                      bool horizontal = true);                                // "lsf": 2D bias removal in the x-y plane (TR); false: 3D (legacy)
```

Behaviour:

- `phase` gets `[n_coeff, n_gen, 2]` (resized only if different); `n_coeff` from `preset`, `n_gen` from the set table (5.1): "lsf" 11, "cluster" −1: 9·N, "cluster" k: N, "ray" (any `set_id`): 14·N·M + N. `NumClusters = 1` (N = 0) gives `[n_coeff, 0, 2]` for "cluster" and "ray" without a sum-of-sinusoids call.
- "lsf": bit for bit the former `lsf_init_phases` (raw phases `sos_init_phases(n_coeff, 11, P0, seed)`, bias removal of columns 0–8 as before, slice 1 = mirror columns). "cluster": bit for bit the former `ssf_init_phases` (derived seed with tag 1 for −1 and 2 + k for k, no bias removal). "ray": derived seed with tag 2^63 + 2^32 + set_id + 2 (algorithm 4.1), no bias removal, slice 1 = mirror columns of the ray mirror table (5.3).
- Seed −1 passes on (system random device) for all set types.
- `dual_mobility`, `remove_bias`, `horizontal` are ignored for "cluster" and "ray" (documented, not checked); `set_id` is ignored for "lsf" (not checked).
- Migration: `lsf_init_phases(sp, ph, seed, &tx, &rx, dual, rb, preset, hor)` → `gbsm_init_phases(sp, "lsf", ph, seed, -1, preset, &tx, &rx, dual, rb, hor)`; `ssf_init_phases(sp, ph, seed, id, preset)` → `gbsm_init_phases(sp, "cluster", ph, seed, id, preset)`.

Validation (in this order; nothing is written before all pass):

| # | Condition | Message |
|---|---|---|
| 1 | `set_type` not one of the three | `Input 'set_type' must be 'lsf', 'cluster' or 'ray'.` |
| 2 | `seed < -1` | `Input 'seed' must be >= -1.` |
| 3 | "cluster"/"ray" and `set_id < -1` | `Input 'set_id' must be >= -1.` |
| 4 | unknown preset | `Input 'preset' is not a known preset.` |
| 5 | `scenpar` invalid | message of `scenpar_validate` |
| 6 | "cluster"/"ray" and `tx_pos` or `rx_pos` not `nullptr` | `Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only.` |
| 7 | "lsf": positions | messages of `qd_gbsm_check_links` |
| 8 | "lsf": bias removal with dual mobility without `tx_pos` | `Input 'tx_pos' is required for the bias removal with dual mobility.` (existing) |

Edge cases: N = 0 (empty "cluster"/"ray" sets); M = 1 ("ray" 15·N generators, layout as 5.3 with M = 1); "lsf" with fewer than 3 positions (no bias removal, existing rule). No OpenMP of its own. Cost: one `sos_init_phases` call; at N = 19, M = 20 a "ray" set has 5339 generators and phases `[256, 5339, 2]` ≈ 22 MB (plan 5.3).

### 3.2 `gbsm_draw` (replaces `lsf_draw`, `ssf_draw`; `src/qd_gbsm_phases.cpp`)

```
// Spatially correlated normalized random values of one generator set at a set of links
// - Columns, distributions and decorrelation distances of each set type: see the set table in the documentation
// - Dual mobility: the Tx side uses the mirror generator; swapping 'tx_pos' and 'rx_pos' exchanges the mirror pairs
// - Links with different floor IDs are uncorrelated at equal horizontal positions (2D mode only)
void gbsm_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances, NumClusters, NumSubPaths, PerCluster* for "ray")
               const std::string &set_type,                             // "lsf", "cluster" or "ray"; must match gbsm_init_phases
               const arma::cube &phase,                                 // Phases from gbsm_init_phases, [n_coeff, n_gen, 2]
               const arma::mat &tx_pos,                                 // Tx positions in [m], [1, 3] or [n_link, 3]
               const arma::mat &rx_pos,                                 // Rx positions in [m], [n_link, 3]
               arma::mat &rnd,                                          // Output: Random values, [n_link, n_gen]
               arma::sword set_id = -1,                                 // As in gbsm_init_phases; ignored for "lsf"
               const std::string &preset = "Comb256",                   // Must match gbsm_init_phases
               const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor; 2D only
               const arma::vec *center_frequency = nullptr,             // Optional: Carrier frequencies in [Hz], [n_freq]; required by the shared "ray" set, same list as for ssf_rays
               bool dual_mobility = false,                              // Tx and Rx both move: the Tx side is a second device
               bool horizontal = true);                                 // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)
```

Behaviour:

- `rnd` gets `[n_link, n_gen]` (resized only if different). "lsf" and "cluster" keep the generators, distributions, decorrelation distances, mirror rules, floor rules and evaluation of the former `lsf_draw` / `ssf_draw` unchanged (same machinery in `qd_gbsm_eval`), so stored phases and seeds reproduce series-1 results.
- "ray": generator table of 5.3; every generator uses `SC_lambda` (0 → 0.1 m) except block 14 of the shared set (`set_id = -1`), which uses λ_τ of algorithm 4.2, also when `SC_lambda = 0`. λ_τ is evaluated only when N > 0. Block 14 of a per-frequency set uses `SC_lambda`.
- `center_frequency` is read only for "ray" with `set_id = -1` (required there, also for N = 0); for every other set it is ignored and not checked. The docs say: pass the same list as to `ssf_rays` (nothing can check it); another list gives other `quadriga` rays, as it gives other `quadriga` clusters (series 1: delays move 13–51 ns at p50, plan P21).
- Dual mobility: block 14 is self-mirrored with one λ_τ from c_ASA, so a Tx move changes the relative ray lengths 2–8× faster than plane waves at the Tx would (c_ASA/c_ASD; documented, P21).
- A set never changes size (P22): the series-1 sentence "later versions append blocks" of `ssf_draw` is replaced by "a set never changes size; new random quantities come in new sets (a new `set_type` or `set_id`) read through appended optional inputs of their consumers". Changing `NumSubPaths` changes the "ray" layout as `NumClusters` changes all SSF layouts.
- Migration: `lsf_draw(sp, ph, tx, rx, rnd, &fl, dual, preset, hor)` → `gbsm_draw(sp, "lsf", ph, tx, rx, rnd, -1, preset, &fl, nullptr, dual, hor)`; `ssf_draw(sp, ph, tx, rx, rnd, id, &fl, dual, preset, hor)` → `gbsm_draw(sp, "cluster", ph, tx, rx, rnd, id, preset, &fl, nullptr, dual, hor)`. The hand migration that keeps the old order, `gbsm_draw(sp, "cluster", ph, tx, rx, rnd, id, nullptr)`, compiles (`nullptr` binds to `preset`; constructing `std::string` from a null pointer is undefined behaviour, libstdc++ throws `std::logic_error`); the release notes name it; no test exercises it.

Validation (in this order):

| # | Condition | Message |
|---|---|---|
| 1 | `set_type` not one of the three | `Input 'set_type' must be 'lsf', 'cluster' or 'ray'.` |
| 2 | positions | messages of `qd_gbsm_check_links` (columns, rows of `tx_pos`, finite) |
| 3 | "cluster"/"ray" and `set_id < -1` | `Input 'set_id' must be >= -1.` |
| 4 | unknown preset | `Input 'preset' is not a known preset.` |
| 5 | `scenpar` invalid | message of `scenpar_validate` |
| 6 | `phase` size (an empty set accepts any empty cube) | `Input 'phase' must have size [<n_coeff>, <n_gen>, 2] for preset '<preset>' and this set.` (existing; "lsf" keeps its existing wording without "and this set") |
| 7 | `phase` not finite | `Input 'phase' must be finite.` |
| 8 | `floor_id` size | `Number of elements in 'floor_id' does not match number of rows in 'rx_pos'.` |
| 9 | `floor_id` with `horizontal = false` | `Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors.` |
| 10 | "ray", `set_id = -1`, `center_frequency == nullptr` | `Input 'center_frequency' is required for set type 'ray' with set_id = -1.` |
| 11 | "ray", `set_id = -1`: empty, non-finite or ≤ 0 entries | `Input 'center_frequency' cannot be empty.` / `Input 'center_frequency' must be finite and > 0.` (`qd_gbsm_check_fc`, renamed message, 3.3) |

Edge cases: `n_link = 0` → `[0, n_gen]`; N = 0 → `[n_link, 0]`; `SC_lambda = 0` (0.1 m except shared block 14); c_ASA = 0 (λ_τ = 10 km by the guard); c̄_DS = 0 (λ_τ = 0.1 m). OpenMP: none of its own (`sos_val` as before). Cost: about 33 ns per value and link, about 0.18 ms per link for a "ray" set at N = 19, M = 20 (F8, provisional).

### 3.3 Changes to existing functions (plan 5.4; R1)

`ssf_clusters` (header comments and docs only; signature and values unchanged): the comment lines of plan 5.4 (`ssf_draw` → `gbsm_draw`, sizes `[n_link, n_slot, n_freq]` for "38.901_MF" else `[n_link, n_slot, 1]`, `rnd_freq` from `gbsm_draw ("cluster", set_id = k)`); `n_fd` leaves the doc block; links `[[ssf_draw]]` → `[[gbsm_draw]]`.

`conf_file_read`, `scenpar_validate`, `conf_file_write`: the argument `subpath_method` is deleted (declarations of plan 5.4). `conf_file_read` accepts the line `SubpathMethod = legacy` and throws on any other value: `<fn>, line <n>: Value '<value>' of 'SubpathMethod' is not supported; only 'legacy' is accepted.` (the prefix of the existing `where` lambda of `qd_scenpar_check`, `src/qd_scenpar.cpp:277`). `conf_file_write` never writes the key (also with `write_defaults = true`); a written file reads back bit for bit. `scenpar_validate` has no string argument any more. The `key_order` doc drops its remark on the legacy position of `SubpathMethod`. The line `SubpathMethod = legacy` is removed from the 23 shipped conf files.

`los_probability`, `lsp_generate`, `o2i_loss`, `path_loss`: `fc` → `center_frequency` (declarations of plan 5.4; values unchanged); doc text that names the argument follows; `qd_gbsm_check_fc` throws `Input 'center_frequency' cannot be empty.` / `Input 'center_frequency' must be finite and > 0.`; comments naming `lsf_draw` say `gbsm_draw ("lsf")`.

Migration hazard (plan 5.4): old calls shift silently into arguments of the same type: `conf_file_read(fn, sp, &subpath_method)` fills it with `PL_model`; `nullptr`-padded calls move `key_order`, `pl_order`, `short_names`, `ignored_keys` one slot down (the last into `bool validate`); `conf_file_write(fn, sp, "legacy")` writes `PL_model = legacy`. R1 finds every caller by the rename-to-find procedure (section 9, R1).

Conf files: `3GPP_38.901_v19_UMa_LOS.conf` and `3GPP_38.901_v19_UMa_NLOS.conf` get `PerClusterDS_fc_min = 6` after `PerClusterDS_min` (TR Table 7.5-6 note 6: c_DS 3.91 ns instead of 4.71 ns at 3.5 GHz; spec scratch: 3.9099 ns vs 4.7078 ns).

Section description (`src/qd_scenpar.cpp`): the workflow paragraph names `gbsm_init_phases` / `gbsm_draw` (R1) and the ray level (R10); the large-scale generator table moves into the set table of section 5 (R1 for "lsf" and "cluster", R3 adds "ray"); the "Terms" paragraph and the codes are added (R10).

Wrappers: MEX and Python of `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw` are replaced by `gbsm_init_phases`, `gbsm_draw` (section 8); the scenario struct / dict rule of P24 (section 8.4); `fc` → `center_freq`; zero-copy outputs in all existing GBSM wrappers where the size is known in advance (P25, section 8.3).

### 3.4 `gbsm_rays` (new class; `include/quadriga_gbsm.hpp`, members in `src/qd_gbsm_rays.cpp`)

```
// Ray data of one link (geometry-based stochastic model)
// - Output of ssf_rays and gbsm_rays_from_paths; solve_scatterers adds the scatterer positions
// - Input of ssf_paths and gbsm_single_link_channel, which call is_valid() first
// - Set the attributes by name: later versions append attributes
class gbsm_rays
{
public:
    std::string method = "38.901"; // Method of the ray data: "38.901", "38.901_MF" or "quadriga"; sets the GR power law and the default mobility
    arma::vec tx_pos;               // Tx position the ray data refer to in [m], [3]
    arma::vec rx_pos;               // Rx position the ray data refer to in [m], [3]
    arma::vec center_frequency;     // Carrier frequencies in [Hz], [n_freq]
    arma::vec delay;                // Delays in [s] relative to the LOS delay |rx_pos - tx_pos| / c, [n_ray]
    arma::mat power;                // Coefficient powers (linear), without link gain; data from paths: path gain, [n_ray, n_freq]
    arma::vec aod;                  // Azimuth of departure in [rad], GCS, [n_ray]
    arma::vec eod;                  // Elevation of departure in [rad], GCS, [n_ray]
    arma::vec aoa;                  // Azimuth of arrival in [rad], GCS, [n_ray]
    arma::vec eoa;                  // Elevation of arrival in [rad], GCS, [n_ray]
    arma::cube M;                   // Polarization transfer matrices incl. initial phases, row order of get_channels_*, [8, n_ray, n_freq]
    arma::uvec path_type;           // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_ray]
    arma::uvec tap;                 // Tap index, rays with equal index form one tap, [n_ray]
    arma::vec gr_eps_re;            // Ground permittivity, real part, [n_freq] or empty; needed where a GR ray is recomputed
    arma::vec gr_eps_im;            // Ground permittivity, imaginary part, [n_freq] or empty; needed where a GR ray is recomputed
    arma::mat fbs_pos;              // First-bounce scatterer positions in [m], [3, n_ray]; empty until solve_scatterers
    arma::mat lbs_pos;              // Last-bounce scatterer positions in [m], [3, n_ray]; empty until solve_scatterers
    arma::uvec fallback;            // 1 = lengthened by the solver fallback, [n_ray]; empty until solve_scatterers
    gbsm_rays() {};                 // Default constructor

    arma::uword n_ray() const;                     // Number of rays, length of 'delay'
    arma::uword n_freq() const;                    // Number of frequencies, length of 'center_frequency'
    bool empty() const;                            // true if the object holds no rays
    std::string is_valid() const;                  // Empty string if the object is valid, else an error message
    arma::vec b_med(arma::uword i_freq = 0) const; // b_med of the Tx (to FBS) and the Rx (to LBS) in [m], [2]; NaN without scatterers or NLOS power
    arma::vec b_eff(arma::uword i_freq = 0) const; // b_eff of the Tx and the Rx in [m], [2]; NaN without scatterers
    void solve_scatterers(double d_min = 2.0);     // Two-bounce solver: writes fbs_pos, lbs_pos and fallback; d_min in [m], > 0 (see 3.6)
};
```

Attributes as in plan 5.5 (doc table: attribute, size, meaning, set by). `gr_eps_re/_im` hold `gr_eps_re - j·gr_eps_im` (the conjugate convention of `lsp_generate`); required only where a GR ray's geometry is recomputed (`drift` away from `tx_pos`/`rx_pos`), else ignored. No cluster delay and no τ_Δ attribute (P1). Data are plain; the class does not normalise or reorder. Syntax of the declaration with `b_eff` checked in the spec scratch (`g++ -std=c++17 -fsyntax-only -Wall -Wextra -Wpedantic`, no warnings).

Members:

| Member | Returns |
|---|---|
| `n_ray()` | `delay.n_elem` |
| `n_freq()` | `center_frequency.n_elem` |
| `empty()` | `n_ray() == 0` |
| `is_valid()` | empty string or the first failed check of the table below |
| `b_med(i_freq)` | `[2]` = (Tx, Rx) in m: lower weighted median of the end-to-scatterer distances dist(FBS, tx_pos) (Tx) and dist(LBS, rx_pos) (Rx) over the NLOS rays (`path_type` 2) with `power(r, i_freq) > 0`, fallback rays included, weights `power(:, i_freq)`; NaN without scatterers or without NLOS power (algorithm 4.12) |
| `b_eff(i_freq)` | `[2]` = (Tx, Rx) in m: (Σ P / Σ (P / b²))^½ over the kept rays at `i_freq` (power > 0, plus LOS and GR), b = dist(FBS, tx_pos) / dist(LBS, rx_pos) for NLOS, d3D for LOS, d_GR for GR (at `tx_pos`, `rx_pos`); NaN without scatterers or with Σ P = 0 (S1, algorithm 4.12) |

`b_med` and `b_eff` throw the `is_valid()` message on an invalid object and `Input 'i_freq' must be smaller than the number of frequencies.` for `i_freq >= n_freq()`.

`is_valid()` checks, in this order (n_ray = `delay.n_elem`, n_freq = `center_frequency.n_elem`; messages exact):

| # | Check | Message |
|---|---|---|
| 1 | `method` | `Input 'rays.method' must be '38.901', '38.901_MF' or 'quadriga'.` |
| 2 | `tx_pos`, `rx_pos` 3 elements, finite | `Input 'rays.tx_pos' must have 3 elements.` / `Input 'rays.tx_pos' must be finite.` (same for `rx_pos`) |
| 3 | `center_frequency` not empty, finite, > 0 | `Input 'rays.center_frequency' cannot be empty.` / `Input 'rays.center_frequency' must be finite and > 0.` |
| 4 | `delay` finite | `Input 'rays.delay' must be finite.` |
| 5 | `power` size and values | `Input 'rays.power' must have size [n_ray, n_freq].` / `Input 'rays.power' must be finite and >= 0.` |
| 6 | `aod`, `eod`, `aoa`, `eoa` | `Input 'rays.aod' must have n_ray elements.` / `Input 'rays.aod' must be finite.` (same per angle) |
| 7 | `M` | `Input 'rays.M' must have size [8, n_ray, n_freq].` / `Input 'rays.M' must be finite.` |
| 8 | `path_type` | `Input 'rays.path_type' must have n_ray elements.` / `Values of 'rays.path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).` / `Input 'rays.path_type' can hold at most one LOS and one GR ray.` |
| 9 | `tap` size | `Input 'rays.tap' must have n_ray elements.` |
| 10 | permittivity | `Inputs 'rays.gr_eps_re' and 'rays.gr_eps_im' must both be empty or both have n_freq elements.` / `Input 'rays.gr_eps_re' must be finite and >= 1.` / `Input 'rays.gr_eps_im' must be finite and >= 0.` |
| 11 | scatterers | `Inputs 'rays.fbs_pos' and 'rays.lbs_pos' must both be empty or both have size [3, n_ray].` / `Input 'rays.fbs_pos' must be finite.` (same for `lbs_pos`) |
| 12 | `fallback` | `Input 'rays.fallback' must be empty or have n_ray elements.` / `Values of 'rays.fallback' must be 0 or 1.` |

Notes: n_ray = 0 is valid (every per-ray attribute empty with the stated outer sizes, e.g. `power` `[0, n_freq]`, `M` `[8, 0, n_freq]`). Tap values are not checked (free labels, P20). Delays are not required ≥ 0: data from paths can carry rounding (float32 QRT lengths) below d3D.

Rules for rays from several sources in one object (P20, documented in the class block): at most one LOS and one GR ray; LOS keeps tap index 0 and GR 3 so that `taps` lists them first; other sources take any other tap indices (not dense); one `method` for all rays; `solve_scatterers` overwrites given scatterers, so rays with given scatterers are solved in a separate object and the attributes are concatenated.

### 3.5 `ssf_rays` (new; `src/ssf_rays.cpp`)

```
// Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)
// - One geometry per call: "38.901" and "38.901_MF" compute frequency 'i_freq' (with slice i_freq of 'rnd_ray');
//   "quadriga" all frequencies; all other inputs as ssf_clusters and lsp_generate return them
// - A pure function of the clusters, the LSPs and the random values: no seed, batch-independent
void ssf_rays(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (NumClusters, NumSubPaths, PerCluster*, GR_enabled, PolVar_sigma)
              const arma::mat &tx_pos,                                // Tx positions in [m], [1, 3] or [n_link, 3]
              const arma::mat &rx_pos,                                // Rx positions in [m], [n_link, 3]
              const arma::vec &center_frequency,                      // Carrier frequencies in [Hz], [n_freq]
              const arma::cube &lsp,                                  // LSPs from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &lsp_mu,                               // Log-domain means from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &lsp_sigma,                            // Log-domain std from lsp_generate, [n_link, n_freq, 8]
              const arma::cube &delay,                                // Cluster delays from ssf_clusters, [n_link, n_slot, n_freq] ("38.901_MF") or [n_link, n_slot, 1]
              const arma::cube &power,                                // Cluster powers from ssf_clusters, [n_link, n_slot, n_freq]
              const arma::cube &aod,                                  // Cluster AOD from ssf_clusters, size as 'delay'
              const arma::cube &eod,                                  // Cluster EOD from ssf_clusters, size as 'delay'
              const arma::cube &aoa,                                  // Cluster AOA from ssf_clusters, size as 'delay'
              const arma::cube &eoa,                                  // Cluster EOA from ssf_clusters, size as 'delay'
              const arma::cube &rnd_ray,                              // "ray" sets from gbsm_draw, [n_link, n_gen, n_freq] (TR methods: slice k = set k) or [n_link, n_gen, 1] (quadriga: set -1)
              std::vector<gbsm_rays> &rays,                           // Output: Ray data, n_link entries; fbs_pos, lbs_pos, fallback empty
              const std::string &method = "38.901",                   // Method of ssf_clusters: "38.901", "38.901_MF" or "quadriga"
              arma::uword i_freq = 0,                                 // TR methods: index into center_frequency and slice of rnd_ray, 0-based; quadriga: must be 0
              const std::string &angle_model = "additive",            // "additive" (TR, zenith fold) or "rotation" (3-D rotation of the offsets)
              const std::string &pol_model = "38.901",                // "38.901" (kappa, four phases) or "quadriga" (rotation model)
              const arma::mat *gr_eps_re = nullptr,                   // Optional: Ground permittivity, real part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
              const arma::mat *gr_eps_im = nullptr,                   // Optional: Ground permittivity, imaginary part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
              bool pol_variability = false,                           // TR 7.6.16 polarisation power variability; pol_model "38.901" only
              bool random_phase = true);                              // false: all initial phases 0
```

Behaviour:

- `rays` gets `n_link` entries (resized only if the count differs); entries whose attributes already have the right sizes are written in place. Entry l: `method`; `tx_pos` = row l of `tx_pos` (row 0 for a single Tx), `rx_pos` = row l of `rx_pos`, both as `[3]`; `center_frequency` = `[center_frequency(i_freq)]` for the TR methods, all of `center_frequency` for `quadriga` (n_freq_out = 1 or n_freq); `delay`, `power` `[n_ray, n_freq_out]`, angles, `M` `[8, n_ray, n_freq_out]`, `path_type`, `tap`; `gr_eps_re/_im` = element (l, `i_freq`) (TR methods) or row l (`quadriga`) of the permittivity when G = 1, empty when G = 0 (the inputs are then ignored, as `ssf_clusters`); `fbs_pos`, `lbs_pos`, `fallback` empty.
- Ray layout, fixed per scenario: index 0 LOS ray, index 1 GR ray (G = 1), then ray m of NLOS cluster n (slot 1 + G + n) at index 1 + G + n·M + m. The rays of a split cluster are listed by sub-cluster (positions m = 0–9 sub-cluster 1, 10–15 sub-cluster 2, 16–19 sub-cluster 3; P4), so every tap is a contiguous block. Clusters at power 0 (removed, padding) keep their M rays, computed as any other ray, at power 0.
- `path_type`: 0, 1 (G = 1), 2 for all NLOS rays. `tap`: 0, 3, 3·(1 + G + n) + i (algorithm 4.5).
- LOS and GR rays: delay, angles of slots 0 and 1; power = slot power per frequency; Jones LOS diag(1, −1), GR diag(R∥, −R⊥)/R per frequency (algorithm 4.9). NLOS rays: power P_n/M per frequency (P_n = slot power at that frequency); delays, angles, Jones by section 4.
- TR methods compute frequency `i_freq` only: slice `i_freq` of `delay`/angles for "38.901_MF" (the only slice otherwise), `power(:, :, i_freq)`, `lsp(:, i_freq, :)`, `lsp_mu`, `lsp_sigma` of `i_freq`, slice `i_freq` of `rnd_ray`, c_DS and c_ZSD of that frequency, sub-clusters by the powers of that frequency. A caller with one frequency per call (NG's split builders) passes `center_frequency` with one entry and the set of its global frequency as the one slice. `quadriga`: one geometry for all frequencies (coupling, ray delays, angles, initial phases, η shared; P5), per-frequency powers, XPR mean/std, Jones.
- `pol_model = "quadriga"`: Jones of NLOS rays by the rotation model with the owner-end rule (algorithm 4.8); `pol_variability` throws there (P1).
- `random_phase = false`: all NLOS initial phases 0 (both polarisation models).
- `i_freq` indexes the input frequency list here and `rays.center_frequency` in `gbsm_single_link_channel` (one entry for the TR methods, so 0 there); both docs say so.
- Swap reciprocity: with mirrored inputs (dual mobility in `gbsm_draw`, mirrored LSPs and clusters, a scenario with `PerClusterAS_D = PerClusterAS_A` and `PerClusterES_D = PerClusterES_A`, `PerClusterES_D_scale = 0`), swapping `tx_pos` and `rx_pos` exchanges departure and arrival per ray and transposes M exactly (TR model by the mirror pairs, rotation model by the owner end; P19).

Validation (in this order):

| # | Condition | Message |
|---|---|---|
| 1 | `method` | `Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.` |
| 2 | `angle_model` | `Input 'angle_model' must be 'additive' or 'rotation'.` |
| 3 | `pol_model` | `Input 'pol_model' must be '38.901' or 'quadriga'.` |
| 4 | `pol_variability` with `pol_model = "quadriga"` | `Input 'pol_variability' needs pol_model '38.901'.` |
| 5 | `scenpar` | message of `scenpar_validate` |
| 6 | positions | messages of `qd_gbsm_check_links` |
| 7 | `center_frequency` | messages of `qd_gbsm_check_fc` (`'center_frequency'`) |
| 8 | "38.901" with n_freq > 1 | `Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several frequencies.` (series-1 wording) |
| 9 | `i_freq` | `Input 'i_freq' must be smaller than the number of frequencies.` / `Input 'i_freq' must be 0 for method 'quadriga'.` |
| 10 | `lsp`, `lsp_mu`, `lsp_sigma` | `Input 'lsp' must have size [n_link, n_freq, 8].` (same wording per name) |
| 11 | `delay`, `aod`, `eod`, `aoa`, `eoa` | `Input 'delay' must have size [n_link, n_slot, n_freq] for method '38.901_MF', else [n_link, n_slot, 1].` (same wording per name; n_slot = `NumClusters` + G) |
| 12 | `power` | `Input 'power' must have size [n_link, n_slot, n_freq].` |
| 13 | `rnd_ray` | `Input 'rnd_ray' must have size [n_link, 14·N·M + N, n_freq] for the TR methods or [n_link, 14·N·M + N, 1] for 'quadriga'.` |
| 14 | permittivity with G = 1 (series-1 wording) | `Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set.` / `Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq].` / `Inputs 'gr_eps_re' and 'gr_eps_im' must be finite.` / `Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.` |
| 15 | values used | `Input 'delay' must be finite.` (same for the angles); `Input 'power' must be finite and >= 0.`; `Input 'lsp_mu' must be finite.` (slices 5, 7 used); `Input 'lsp_sigma' must be finite and >= 0.` (slice 7); `Input 'lsp' must be finite and > 0.` (slice 7, rotation model only); `Input 'rnd_ray' must be finite.`; `Values of block 14 of 'rnd_ray' must be in [0, 1].` (`quadriga` only) |

As `ssf_clusters`: only sizes show that the inputs belong together. `n_link = 0` gives an empty vector.

Edge cases: N = 0 (`rnd_ray` `[n_link, 0, ·]`; rays LOS (+ GR) only); M = 1 (one ray per cluster, offset 0, `quadriga` delay offset 0); odd M (zero-offset ray first, algorithm 4.3); M ≠ 20 (no sub-clusters); one active cluster (only that one split); zero-power clusters (not split, rays at power 0); NLOS scenario (LOS and GR rays at power 0, kept); TR LOS (first NLOS cluster at τ = 0: its first sub-cluster shares delay 0 with the LOS ray, P10); `SC_lambda = 0` (no effect here: the values come in); c_ASA = 0 (all AOA offsets 0); GR disabled (no GR ray, `gr_eps_*` empty); single link; `quadriga` clamp of negative ray delays (algorithm 4.5).

OpenMP: one parallel loop over the links, `if (n_link >= L_rays)` with L_rays measured by R4 (rule of `md/SPEC_CPP_API.md`; start from the `ssf_clusters` value 8); exceptions inside via `OMP_SAFE_CALL`; results independent of the thread count. Cost target: below the draw of its "ray" set (≈ 0.18 ms per link at N = 19, M = 20, F8, provisional); R4 reports its measured cost per link (provisional under load).

### 3.6 `gbsm_rays::solve_scatterers` (new member; `src/qd_gbsm_rays.cpp`; wrappers `ssf_scatterers`)

The solver is a member of `gbsm_rays` (it reads and writes only the object, as `arrayant::rotate_pattern`); the MEX and Python wrappers keep the free-function name `ssf_scatterers(rays, d_min)` (8.1, 8.2), one name pair in `api_matching.toml` (7.4). Doc block `# .solve_scatterers` (member block). Declared in the class (3.4); its doc comment in the class declaration is the one line there, the full text below goes into the doc block:

```
// First- and last-bounce scatterers of the rays of one link (two-bounce solver)
// - NLOS rays: exact departure and arrival angles where a solution exists, equal distances from both ends;
//   otherwise the shortest polyline along the ray's own directions with both scatterers at least d_min away
// - LOS ray on the direct line, GR ray at the specular point on z = 0, rays with zero power at all frequencies at
//   the midpoint of the direct line
void gbsm_rays::solve_scatterers(double d_min = 2.0); // Minimum distance of the scatterers to both ends in [m], > 0; writes fbs_pos, lbs_pos and fallback
```

Behaviour:

- Reads `tx_pos`, `rx_pos`, `delay`, `power`, the angles and `path_type`; writes `fbs_pos`, `lbs_pos` `[3, n_ray]` and `fallback` `[n_ray]` for all rays (resized only if different), overwriting given scatterers (P20); nothing else changes (`delay` keeps the drop value; a fallback ray's polyline is longer).
- LOS ray: FBS = LBS = midpoint of `tx_pos`, `rx_pos` (the convention of `coord2path` for paths without interactions). GR ray: FBS = LBS = specular point on z = 0 (algorithm 4.10). NLOS ray with power 0 at all frequencies: FBS = LBS = midpoint, `fallback` 0. NLOS ray with power > 0 at some frequency: the balanced closed form of algorithm 4.10 with the length L = |rx_pos − tx_pos| + c·delay; `fallback` = 1 when it lengthens the ray. LOS and GR: `fallback` 0.
- Afterwards `rays.b_med()` gives the validity scale of `drift` and `rays.b_eff()` that of `phase` (P15, S1).
- Measured on library clusters (F9, F14): with d_min = 2 m the fallback applies to 5.5–9.6 % of the rays in the TR methods without absolute TOA and ≤ 2.3 % otherwise, adds a median delay of 0.2–4.6 ns (p99 ≤ 21.7 ns) and changes the per-link DS by ≤ 4.1 % at p99; documented.

Validation: `is_valid()` (message passed through; the object is left unchanged on any error); `Input 'd_min' must be finite and > 0.` Edge cases: n_ray = 0 (empty `[3, 0]`, `[0]` outputs); d3D = 0 (the closed form stays defined, algorithm 4.10); h_t + h_r = 0 with a GR ray (GR point = midpoint); rays with zero excess length (TR LOS first cluster, clamped `quadriga` rays: fallback); the 0.1 mm LOS branch of `get_channels_spherical`: about 2·10⁻⁶ of track samples in TR LOS come within 0.1 mm of the LOS length (P13, documented, not handled). Per link, no batch, no OpenMP region; batch callers loop over the entries of `ssf_rays`. Cost ≈ 12.5 ns per ray (F14, provisional).

### 3.7 `ssf_paths` (new; `src/ssf_paths.cpp`)

```
// Antenna-free paths of one link at one position
// - "phase": use the outputs with get_channels_planar and the arrays at the positions of 'rays' (the displacement
//   phase is in M); "drift": inputs of get_channels_spherical / get_channels_multifreq and qrt_paths_from_scatterers
// - Rays with zero power at all frequencies are dropped, LOS and GR rays are kept
// - Validity: "phase" up to about sqrt(lambda * b_eff / 8) per moving end (see gbsm_rays::b_eff),
//   "drift" up to about 0.2 * b_med per moving end
void ssf_paths(const gbsm_rays &rays,                // Ray data of the link
               const arma::vec &link_gain_dB,        // Link gain -PL - O2I + SF at the position in [dB], [n_freq]; 0 for data from paths
               arma::mat &path_gain,                 // Output: Path gains (linear), [n_path, n_freq]
               arma::vec &path_length,               // Output: Path lengths in [m], [n_path]
               arma::cube &M,                        // Output: Polarization transfer matrices, [8, n_path, n_freq]
               const std::string &mobility = "auto", // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
               const arma::vec *tx_pos = nullptr,    // Optional: Tx position in [m], [3]; nullptr = rays.tx_pos
               const arma::vec *rx_pos = nullptr,    // Optional: Rx position in [m], [3]; nullptr = rays.rx_pos
               const arma::vec *kf_ratio = nullptr,  // Optional: K(position) / K(drop), [n_freq]; nullptr = 1
               arma::vec *aod = nullptr,             // Optional output: Azimuth of departure in [rad], [n_path]
               arma::vec *eod = nullptr,             // Optional output: Elevation of departure in [rad], [n_path]
               arma::vec *aoa = nullptr,             // Optional output: Azimuth of arrival in [rad], [n_path]
               arma::vec *eoa = nullptr,             // Optional output: Elevation of arrival in [rad], [n_path]
               arma::mat *fbs_pos = nullptr,         // Optional output: First-bounce scatterer positions in [m], [3, n_path]; "phase": [3, 0]
               arma::mat *lbs_pos = nullptr,         // Optional output: Last-bounce scatterer positions in [m], [3, n_path]; "phase": [3, 0]
               arma::uvec *path_type = nullptr,      // Optional output: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
               arma::uvec *ray_ind = nullptr,        // Optional output: Index of the ray in 'rays', 0-based, [n_path]
               bool oxygen = false);                 // TR 7.6.1 oxygen absorption from each path's own length
```

Behaviour:

- Kept rays: LOS and GR rays always; NLOS rays with power > 0 at some frequency. Paths in ray order; `ray_ind` maps each path to its ray (`rays.tap(ray_ind)`, `rays.fallback(ray_ind)`). For one frequency f, `path_gain.col(f)` and `M.slice(f)` are the `[n_path]` and `[8, n_path]` inputs of `get_channels_planar` / `get_channels_spherical`; all outputs as they are feed `get_channels_multifreq` and `qrt_paths_from_scatterers`.
- `mobility = "auto"`: "phase" for `rays.method` "38.901" and "38.901_MF", "drift" for "quadriga".
- Path gain per frequency (algorithm 4.11): coefficient power (GR: recomputed in `drift`), `kf_ratio` rescale of LOS/GR against NLOS, × 10^(`link_gain_dB`/10), × 10^(−OL/10) with `oxygen` (OL from `path_length`, 7.6-1 without τ_Δ, P1).
- "phase": `path_length` = |rays.rx_pos − rays.tx_pos| + c·delay; angles of the ray data; M = ray M × exp(j·2π·f/c·(r̂_rx·Δr_rx + r̂_tx·Δr_tx)) per frequency with Δr = position − ray-data position (algorithm 4.11). `fbs_pos`, `lbs_pos` are returned as `[3, 0]`. The docs lead with the use: `get_channels_planar` with the arrays at `rays.tx_pos` / `rays.rx_pos` (at the new positions the LOS path is no longer identified and the displacement phase counts twice, silently).
- "drift": scatterers fixed, ends at the given positions; polyline lengths and angles from the geometry; LOS recomputed (midpoint of the new direct line, length d3D(t)); GR point, Fresnel coefficients and the GR power law of `rays.method` recomputed when an end differs from the ray-data position (exact comparison of the three coordinates); M of NLOS rays unchanged (algorithm 4.11).
- `link_gain_dB` and `kf_ratio` have `n_freq` entries; `is_valid()` first. The docs state both validity limits (F17, P7, P15) and point to `b_eff` / `b_med` of `gbsm_rays`.

Validation (in this order):

| # | Condition | Message |
|---|---|---|
| 1 | `rays` | message of `is_valid()` |
| 2 | `mobility` | `Input 'mobility' must be 'phase', 'drift' or 'auto'.` |
| 3 | `link_gain_dB` | `Input 'link_gain_dB' must have n_freq elements.` / `Input 'link_gain_dB' must be finite.` |
| 4 | `tx_pos`, `rx_pos` given | `Input 'tx_pos' must have 3 elements.` / `Input 'tx_pos' must be finite.` (same for `rx_pos`) |
| 5 | `kf_ratio` given | `Input 'kf_ratio' must have n_freq elements.` / `Input 'kf_ratio' must be finite and >= 0.` |
| 6 | "drift" without scatterers | `Mobility 'drift' needs 'rays.fbs_pos' and 'rays.lbs_pos'; solve the scatterers first.` |
| 7 | "drift", GR ray, an end moved, permittivity empty | `Input 'rays.gr_eps_re' is required to recompute the ground reflection away from the positions of 'rays'.` |
| 8 | "drift", GR ray, an end moved below z = 0 | `Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.` |

Edge cases: n_ray = 0 (outputs `[0, n_freq]`, `[0]`, `[8, 0, n_freq]`); positions equal to the ray-data positions (phase: no rotation; drift: exactly the drop geometry, fallback rays with their longer polyline); NLOS scenario (LOS/GR kept at gain 0); `kf_ratio` = 0 (LOS and GR gain 0); data from paths (`link_gain_dB = 0`; `drift` keeps the path gain of the ray-data positions, so distance changes of the gain come through `link_gain_dB`). No OpenMP (one position per call). Cost: O(n_ray·n_freq), no target.

### 3.8 `qrt_paths_from_scatterers` (new; `include/quadriga_channel.hpp`, "Channel functions", `src/qd_channel_qrt.cpp`)

```
// Path objects of one CIR from spherical path data, for qrt_file_append
// - The path gain and the inverse free-space loss of the polyline length are folded into the stored Jones matrices
// - LOS: 0 interactions; GR: 1 (reflection); NLOS: 2 (scattering)
// - Use local coordinates (QRT stores float32): shift x and y only, the ground stays at z = 0
std::vector<path> qrt_paths_from_scatterers(const arma::vec &center_frequency, // Frequencies in [Hz], [n_freq], 1 to 127 entries
                                            const arma::vec &tx_pos,           // Tx (origin) position in [m], [3]
                                            const arma::vec &rx_pos,           // Rx position of the CIR in [m], [3]
                                            const arma::mat &fbs_pos,          // First-bounce scatterer positions in [m], [3, n_path]
                                            const arma::mat &lbs_pos,          // Last-bounce scatterer positions in [m], [3, n_path]
                                            const arma::mat &path_gain,        // Path gains (linear), [n_path, n_freq]
                                            const arma::cube &M,               // Polarization transfer matrices, [8, n_path, n_freq]
                                            const arma::uvec &path_type,       // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
                                            arma::uword i_cir = 0);            // CIR index, stored in path::iC, < 2^32
```

Behaviour:

- Returns n_path objects, EM layout, `n_freq` frequencies, `iC = i_cir`, in input order; the caller appends them to the `path_data` of `qrt_file_append` (one call per origin; `qrt_file_init` takes GHz, documented).
- Interactions (S4): LOS none (`nREF` = `nSCT` = 0); GR one at `lbs_pos` with code 129 (0x81, front reflection, outside; existing code), `nREF` = 1; NLOS two, at `fbs_pos` then `lbs_pos`, both with code 193 (0xC1, the reserved scattering form), `nSCT` = 2. The `length` member is set to the polyline Tx → interactions (the origin form of `calc_length`, float).
- Jones per path p and frequency f (algorithm 4.13): J = M(:, p, f) · sqrt(path_gain(p, f)) · L_p · 10^((32.45 + 20·lg(f/10⁹))/20), L_p = polyline length in double; so that `qrt_file_read` (either `normalize_M`) returns path_gain·|M|² within float32 (F2: 4.7e-8), the polyline as `path_length` and `no_int` = `path_type` (as `arma::u32_vec`).
- Double inputs; the objects are float32 by construction. Local coordinates (F2: path length error 2.7 µm / 0.31 mm / 1.0 mm for coordinates up to 150 m / 5 km / 20 km, phase error 0.003 / 0.39 / 1.29 rad at 60 GHz).
- Writer memory, documented with its scaling in the snapshot count: about 80–105 B per path object (3–4 GB for 1000 UTs × 100 snapshots × 381 rays at one frequency); at NG's track sampling (58 snapshots per metre at 3.5 GHz, 467 at 28 GHz) 381 rays give 2.2·10⁴ objects per metre and UT, 18–23 GB for 100 UTs × 100 m at 3.5 GHz (estimate, plan 5.9).

Validation (in this order): `Input 'center_frequency' must have 1 to 127 elements.` / `Input 'center_frequency' must be finite and > 0.`; `Input 'tx_pos' must have 3 elements.` / `... must be finite.` (same for `rx_pos`); `Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].` / `... must be finite.`; `Input 'path_gain' must have size [n_path, n_freq].` / `Input 'path_gain' must be finite and >= 0.`; `Input 'M' must have size [8, n_path, n_freq].` / `Input 'M' must be finite.`; `Input 'path_type' must have n_path elements.` / `Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).`; `Input 'i_cir' must be < 2^32.` (n_path is the column count of `fbs_pos`.) Edge cases: n_path = 0 (empty vector); LOS with d3D = 0 (Jones 0). No OpenMP; cost O(n_path·n_freq).

### 3.9 `gbsm_rays_from_paths` (new; `include/quadriga_gbsm.hpp`, GBSM section, `src/gbsm_rays_from_paths.cpp`)

```
// Ray data of one link from spherical path data (e.g. read with qrt_file_read)
// - Delays from the path lengths, angles from the scatterer positions, power = path gain
// - Fills a pre-initialised object in place
void gbsm_rays_from_paths(const arma::vec &center_frequency,      // Frequencies in [Hz], [n_freq]
                          const arma::vec &tx_pos,                // Tx position in [m], [3]
                          const arma::vec &rx_pos,                // Rx position in [m], [3]
                          const arma::mat &fbs_pos,               // First-bounce scatterer positions in [m], [3, n_path]
                          const arma::mat &lbs_pos,               // Last-bounce scatterer positions in [m], [3, n_path]
                          const arma::mat &path_gain,             // Path gains (linear), [n_path, n_freq]
                          const arma::vec &path_length,           // Path lengths in [m], [n_path]
                          const arma::cube &M,                    // Polarization transfer matrices, [8, n_path, n_freq]
                          gbsm_rays &rays,                        // Output: Ray data, n_ray = n_path
                          const std::string &method = "quadriga", // Value of rays.method: GR power law and default mobility
                          const arma::uvec *path_type = nullptr,  // Optional: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]; nullptr = LOS where FBS = LBS on the direct line, else NLOS
                          const arma::uvec *tap = nullptr,        // Optional: Tap index, [n_path]; nullptr = one tap per path
                          const arma::vec *gr_eps_re = nullptr,   // Optional: Ground permittivity, real part, [n_freq]; needed where a GR path is recomputed
                          const arma::vec *gr_eps_im = nullptr);  // Optional: Ground permittivity, imaginary part, [n_freq]; needed where a GR path is recomputed
```

Behaviour:

- Inputs carry the output names and the order of `qrt_file_read`. `method`, `center_frequency`, `tx_pos`, `rx_pos` copied; `delay` = (`path_length` − |rx_pos − tx_pos|)/c; departure angles from `tx_pos` towards the first point among (`fbs_pos`, `lbs_pos`, `rx_pos`) that differs from `tx_pos`, arrival angles from `rx_pos` towards the first point among (`lbs_pos`, `fbs_pos`, `tx_pos`) that differs from `rx_pos` (LOS with FBS = LBS = midpoint gives the direct line); `power` = `path_gain`; `M`, `fbs_pos`, `lbs_pos` copied; `fallback` empty (cleared only if not empty); `gr_eps_re/_im` copied or empty. All attributes are written in place when their sizes match.
- `path_type = nullptr`: a path is LOS when dist(FBS, LBS) ≤ 0.1 mm and `path_length` − |rx_pos − tx_pos| ≤ 0.1 mm (the LOS tolerance of `get_channels_*`; open point 11.1), else NLOS; more than one such path throws. QRT stores no GR type: a GR path read from a file needs `path_type` (for files of `qrt_paths_from_scatterers` the `no_int` of `qrt_file_read`, converted to `arma::uvec`).
- `tap = nullptr`: tap index = path index (the LOS path then has its path index, not 0; documented).
- A GR path whose interaction point (`lbs_pos`) has |z| > 1 mm throws (local coordinates shifted in z, P12; 1 mm is the float32 length error at 20 km of F2).
- Used with `link_gain_dB = 0` in `ssf_paths` / `gbsm_single_link_channel`.

Validation (in this order): `Input 'center_frequency' cannot be empty.` / `... must be finite and > 0.`; `tx_pos`, `rx_pos` as 3.8; `Inputs 'fbs_pos' and 'lbs_pos' must have size [3, n_path].` / `... must be finite.`; `Input 'path_gain' must have size [n_path, n_freq].` / `... must be finite and >= 0.`; `Input 'path_length' must have n_path elements.` / `Input 'path_length' must be finite and >= 0.`; `Input 'M' must have size [8, n_path, n_freq].` / `... must be finite.`; `Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.`; `Input 'path_type' must have n_path elements.` / `Values of 'path_type' must be 0 (LOS), 1 (GR) or 2 (NLOS).` / `Input 'path_type' can hold at most one LOS and one GR path.` / (nullptr) `More than one path matches the LOS geometry; pass 'path_type'.`; `Input 'tap' must have n_path elements.`; `Inputs 'gr_eps_re' and 'gr_eps_im' must both be given or both be omitted.` / `... must have n_freq elements.` / `Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.`; `The interaction point of the GR path must be on the ground (z = 0); shift local coordinates in x and y only.` Edge cases: n_path = 0 (empty object with the given frequencies); a file LOS path (0 interactions, FBS = LBS = midpoint). No OpenMP.

### 3.10 `gbsm_single_link_channel` (new; `include/quadriga_lib.hpp`, GBSM section, `src/gbsm_single_link_channel.cpp`)

```
// Channel coefficients of one link at one frequency over one segment (geometry-based stochastic model)
// - Per snapshot get_channels_planar ("phase", arrays at the positions of 'rays', computed once per distinct
//   orientation) or get_channels_spherical ("drift") with the paths of ssf_paths
// - "taps": rays summed per tap, LOS and GR taps always present, tap count fixed per call; "rays": one path per ray
// - Validity: see ssf_paths; the channel's 'par' reports the largest displacement relative to the limit
//   ("phase": sqrt(lambda * b_eff / 8); "drift": 0.2 * b_med)
channel<double> gbsm_single_link_channel(const arrayant<double> &tx_array,          // Tx array antenna, n_tx ports
                                         const arrayant<double> &rx_array,          // Rx array antenna, n_rx ports
                                         const gbsm_rays &rays,                     // Ray data of the link
                                         const arma::vec &link_gain_dB,             // Link gain -PL - O2I + SF at rays.center_frequency(i_freq) in [dB], [n_snap] or [1]
                                         const std::string &mobility = "auto",      // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
                                         const std::string &output = "taps",        // "taps" or "rays"
                                         arma::uword i_freq = 0,                    // Frequency index in rays.center_frequency, 0-based; TR methods: 0 (one entry)
                                         const arma::mat *tx_pos = nullptr,         // Optional: Tx positions in [m], [n_snap, 3] or [1, 3]; nullptr = rays.tx_pos
                                         const arma::mat *rx_pos = nullptr,         // Optional: Rx positions in [m], [n_snap, 3] or [1, 3]; nullptr = rays.rx_pos
                                         const arma::mat *tx_orientation = nullptr, // Optional: Tx Euler angles (bank, tilt, heading) in [rad], [n_snap, 3] or [1, 3]; nullptr = 0
                                         const arma::mat *rx_orientation = nullptr, // Optional: Rx Euler angles (bank, tilt, heading) in [rad], [n_snap, 3] or [1, 3]; nullptr = 0
                                         const arma::vec *kf_ratio = nullptr,       // Optional: K(snapshot) / K(drop) at rays.center_frequency(i_freq), [n_snap] or [1]; nullptr = 1
                                         bool oxygen = false,                       // TR 7.6.1 oxygen absorption from each path's own length
                                         bool use_absolute_delays = false);         // Include the LOS delay in all delays, as get_channels_*
```

Behaviour:

- `n_snap` = the largest of the row counts of `tx_pos`, `rx_pos`, `tx_orientation`, `rx_orientation` and the element counts of `link_gain_dB`, `kf_ratio` (1 for omitted inputs); every one of them must be 1 or `n_snap`; one-row inputs apply to all snapshots.
- The result for snapshot t is defined as: the paths of `ssf_paths` (same `mobility`, `oxygen`) at the positions of snapshot t, evaluated at frequency `i_freq` with the link gain and K ratio of snapshot t, restricted to the kept rays at `i_freq` (power(r, i_freq) > 0, plus LOS and GR), then `get_channels_planar` ("phase": arrays at `rays.tx_pos` / `rays.rx_pos`, orientations of snapshot t) or `get_channels_spherical` ("drift": arrays at the snapshot positions and orientations), `center_frequency = rays.center_frequency(i_freq)`, `use_absolute_delays` passed on, no fake LOS path. "phase" with orientations equal to those of the previous snapshot (exact comparison) reuses the planar coefficients of that orientation with unit path gains and applies sqrt(path gain) and the displacement phase per snapshot (algorithm 4.14; a test compares it with per-snapshot calls).
- `taps`: the paths summed per tap index (taps in ascending index, count = distinct tap indices of the kept rays, fixed per call; LOS and GR taps present whenever the rays hold them, at zero power in NLOS); `delay` `[n_rx, n_tx, n_tap]` per snapshot = mean of the rays' per-element delays weighted by `rays.power(:, i_freq)` (equal weights when the tap's weight sum is 0) (S2). In TR LOS two taps have delay 0 (P10). `rays`: one path per kept ray in ray order, `delay` `[n_rx, n_tx, n_path]` as `get_channels_*` return it; also `path_gain`, `path_length`, `path_polarization` (M at `i_freq`), `path_angles` `[n_path, 4]` (AOD, EOD, AOA, EOA) per snapshot, and in "drift" `path_fbs_pos`, `path_lbs_pos`. "phase" keeps the delays of the ray-data positions (with `use_absolute_delays` the LOS delay of those positions is included). The delay output is half the size of the coefficients; users who need `[1, 1, ·]` reduce it themselves (docs).
- Output `channel<double>`: `center_frequency` `[1]`; `tx_pos`, `rx_pos` (absolute, the snapshot positions), `tx_orientation`, `rx_orientation` `[3, n_snap]`; `coeff_re`, `coeff_im`, `delay` n_snap cubes; `initial_position` 0; `name` default; `par_names` / `par_data` as in the table below. The object passes `channel::is_valid()`.
- `taps` is exact at the phase centre for the TR methods (rays of a tap share one delay); for `quadriga` it is exact only at the carrier (narrowband per cluster; in-band selectivity 0.18 / 0.52 / 0.79 over 20 / 100 / 400 MHz in UMi NLOS, P17); `rays` is exact at any bandwidth and is the form the QRT route stores. No power correction, no initial-position input.

`par` entries (names exact; S1 and P17):

| Name | Type, size | Content | Present |
|---|---|---|---|
| `tap` | `arma::uvec` `[n_tap]` or `[n_path]` | tap index per output tap (`taps`) or per path (`rays`) | always |
| `n_ray_tap` | `arma::uvec` `[n_tap]` | number of kept rays per tap | `taps` |
| `fbs_pos`, `lbs_pos` | `arma::mat` `[3, n_kept]` | scatterers of the n_kept kept rays in ray order (LOS and GR at the ray-data positions) | `drift` and `taps` |
| `gr_pos` | `arma::mat` `[3, n_snap]` | GR point per snapshot | `drift`, `taps`, GR ray kept |
| `link_gain_dB` | `arma::vec` `[n_snap]` | link gain per snapshot | always |
| `b_med` | `arma::vec` `[2]` | `rays.b_med(i_freq)` (Tx, Rx); NaN without scatterers or NLOS power | always |
| `b_eff` | `arma::vec` `[2]` | `rays.b_eff(i_freq)` (Tx, Rx); without scatterers from a solved copy (default d_min) | always |
| `limit` | `arma::vec` `[2]` | validity limit of the active mode per end: `phase` sqrt(λ·b_eff/8), `drift` 0.2·b_med; Inf where b is NaN | always |
| `ratio` | `arma::vec` `[2]` | max over the snapshots of dist(end, ray-data position) / `limit`; 0 for an end that does not move | always |
| `ratio_total` | `double` | sqrt(ratio_tx² + ratio_rx²); valid ≤ 1, `drift` new segment from 2.5 (0.5·b_med) | always |

Validation (in this order): `tx_array`, `rx_array` (`Input 'tx_array' is invalid: <arrayant message>`); `rays.is_valid()`; `Input 'mobility' must be 'phase', 'drift' or 'auto'.`; `Input 'output' must be 'taps' or 'rays'.`; `Input 'i_freq' must be smaller than the number of frequencies in 'rays'.`; `Input 'link_gain_dB' cannot be empty.` / `... must be finite.`; positions and orientations: `Input 'tx_pos' must have 3 columns.` / `... must be finite.` (each of the four); `kf_ratio`: `... must be finite and >= 0.`; `Number of snapshots in '<name>' must be 1 or <n_snap>.`; then the "drift" checks 6–8 of 3.7. Edge cases: one snapshot; n_ray = 0 (cubes `[n_rx, n_tx, 0]`); NLOS scenario (LOS/GR taps at zero power); data from paths (`link_gain_dB = 0`; `tap` = path index without the optional input); a single kept ray. OpenMP: one parallel loop over the snapshots, `if (n_snap·n_kept·n_rx·n_tx >= L_chan)`, L_chan measured by R9; results independent of the thread count. Cost targets (F8, measured under load, provisional): `phase` with constant orientation 0.035 / 0.075 / 1.0 ms per link and snapshot for 1×1 / 2×2 dual-pol / 8×8 dual-pol BS arrays; `drift` dominated by `get_channels_spherical` (≈ 1.2 µs per element × path pair).

### 3.11 Internal helpers (`src/quadriga_gbsm_helpers.hpp`, R2)

Shared by more than one source file, `static inline`, documented in the header's helper list:

- `qd_gbsm_cds(sp, f)`: c_DS in s of a completed scenpar at frequency f in Hz (algorithm 4.2); used by `qd_gbsm_phases.cpp` (λ_τ) and `ssf_rays.cpp`.
- `qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen)`: from a completed scenpar the counts N, M, G, n_ray = 1 + G + N·M and n_gen = 14·N·M + N of the "ray" sets; used by `qd_gbsm_phases.cpp`, `ssf_rays.cpp` and the MEX/Python wrappers of `gbsm_init_phases` and `ssf_rays` for the zero-copy output sizes (the wrappers complete a copy of `scenpar` with `scenpar_validate` first).
- `qd_gbsm_check_fc` keeps its role with the message name `'center_frequency'` (R1).
- `qd_gbsm_fresnel(eps_re, eps_im, h_t, h_r, d2D, R_par, R_per)`: R∥, R⊥ of TR 7.6-38/39 (algorithm 4.9); used by `ssf_rays.cpp` and `ssf_paths.cpp`. `ssf_clusters.cpp` keeps its own inline code (no refactor of series-1 code).

File-local: the solver kernel (`qd_gbsm_rays.cpp`, behind `gbsm_rays::solve_scatterers`; `gbsm_single_link_channel` calls `solve_scatterers` on a copy), the oxygen table (`ssf_paths.cpp`), the offsets of algorithm 4.3 (`ssf_rays.cpp`).

## 4. Algorithms

Notation: link l, NLOS cluster n (slot s = 1 + G + n), ray m (0-based position in the cluster), frequency k; x_b denotes the value of block b of the "ray" set for (l, n, m) (column b·N·M + n·M + m for b ≤ 12, 13·N·M + n for b = 13, 13·N·M + N + n·M + m for b = 14; section 5.2). Φ_N is the standard normal CDF; angles in the formulas in degrees where marked. `gbsm_rays_from_paths` is fully specified in 3.9.

### 4.1 Seeds of the "ray" sets (P22)

Derived seed = SplitMix64(SplitMix64(seed) XOR tag) >> 1, the series-1 rule of `ssf_init_phases` (`src/qd_gbsm_phases.cpp`, `derive_seed`), with the uint64 tag 2^63 + 2^32 + (set_id + 2) for "ray" (set −1: 2^63 + 2^32 + 1; set k: 2^63 + 2^32 + 2 + k). "cluster" keeps tag 1 (shared) and 2 + k; "lsf" uses the seed directly. Seed −1 passes on. `sos_init_phases(n_coeff, n_gen, P0, derived seed)` gives the Rx side; the Tx side (slice 1) is column m(g) of P0 with the mirror map of 5.3.

### 4.2 c_DS and λ_τ (Table 7.5-6, P21, S5)

- c_DS(f) in s for a completed scenpar and a frequency f in Hz: 3.91 ns if `PerClusterDS` = 0 (TR "N/A"); else max(`PerClusterDS_min`, `PerClusterDS` + `PerClusterDS_gamma`·lg(max(f/10⁹, `PerClusterDS_fc_min`))) ns. (The R0 prototype skips the minimum when `PerClusterDS_gamma` = 0; spec scratch `s5_cds_files.py`: none of the 23 shipped files has `PerClusterDS_gamma` = 0 with `PerClusterDS_min` > `PerClusterDS`, so both rules agree there.)
- c̄_DS = arithmetic mean of c_DS(f) over the entries of `center_frequency`.
- λ_τ = min(max(1.6·c·c̄_DS/c_ASA, 0.1 m), 10⁴ m), c_ASA = `PerClusterAS_A`·π/180; c̄_DS = 0 gives 0.1 m (also 0/0), c_ASA = 0 < c̄_DS gives 10⁴ m. Evaluated only when N > 0. Never binds for the 114 library and NG files (2.93–37.79 m, S5).
- Values for tests (spec scratch, `PerClusterDS_fc_min` as in the files before R1 unless stated): UMi NLOS 3.5 GHz: c_DS 11 ns, c_ASA 22°, λ_τ 13.7415 m; UMi LOS 3.5 GHz: 5 ns, 17°, 8.0832 m; UMa NLOS 3.5 GHz: 4.7078 ns, 15°, 8.6256 m, with `PerClusterDS_fc_min = 6` (after R1): 3.9099 ns, 7.1638 m; UMa NLOS {3.5, 28} GHz: 5.8058 m, with `PerClusterDS_fc_min = 6`: 5.0749 m.

### 4.3 Ray offsets α_m (Table 7.5-3, P11, S3)

Offsets at unit rms spread, in degrees per degree of cluster spread; the ray angle offset is c·α_m with c the cluster spread in degrees.

- M = 20: Table 7.5-3 verbatim in TR ray order: α = (+0.0447, −0.0447, +0.1413, −0.1413, +0.2492, −0.2492, +0.3715, −0.3715, +0.5129, −0.5129, +0.6797, −0.6797, +0.8844, −0.8844, +1.1481, −1.1481, +1.5195, −1.5195, +2.1551, −2.1551) (TR ray 2k − 1 = +a_k, ray 2k = −a_k); rms 1.000038419.
- M < 20 (NG's rule): o = the first M entries of NG's reordered table (−0.8844, 1.1481, 0.6797, −1.1481, −0.6797, 1.5195, −0.0447, 0.3715, −1.5195, −0.2492, −0.5129, 0.5129, 0.8844, −2.1551, −0.1413, 0.1413, 0.0447, 2.1551, −0.3715, 0.2492), α = (o − mean(o)) / rms(o − mean(o)); M = 1 gives α = (0). NG's printed tables for M = 4, 6, 10 are not used (M = 4 is truncated; spec scratch: rule vs printed 6.4·10⁻⁵ rad at M = 4, ≤ 4.9·10⁻⁶ rad at 6 and 10).
- M > 20: K = ⌊M/2⌋, magnitudes q_k = −ln(2·(1 − j_k/(M + 1))) with j_k = M − K + k, k = 1…K (Laplacian quantiles at F_j = j/(M + 1), ascending); list (0 if M is odd), +q_1, −q_1, +q_2, −q_2, …, +q_K, −q_K; scaled so that its rms equals 1.000038419 (the rms of Table 7.5-3).
- Values for tests (spec scratch `s1_offsets.py`, 6 decimals): M = 4: (−0.847110, 1.219260, 0.743054, −1.115204); M = 6: (−0.945308, 0.994948, 0.547806, −1.197040, −0.749898, 1.349492); M = 3: (−1.379551, 0.959273, 0.420279); M = 2: (−1, 1); M = 21: (0, ±0.086626, ±0.182387, ±0.289439, ±0.410804, ±0.550909, ±0.716619, ±0.919431, ±1.180902, ±1.549424, ±2.179417) in the order 0, +, −, +, −, …; M = 26: ±0.033177, ±0.103542, ±0.180033, ±0.263819, ±0.356441, ±0.459983, ±0.577369, ±0.712881, ±0.873159, ±1.069322, ±1.322221, ±1.678661, ±2.288001. Rule outputs below 20: mean 0, rms 1 exactly, not symmetric; above 20: mean 0, rms 1.000038419, symmetric, largest offset monotone in M (S3).
- Sub-clusters only at M = 20 (P11). TR 7.6.2.2 would draw random offsets instead (section 10.3).

### 4.4 Coupling (step 8, P6)

For link l, cluster n and each quantity q with its rank block b_q (AOD 0, AOA 1, ZOD 2, ZOA 3):

- Groups of ray positions: unsplit cluster: one group {0, …, M − 1} with the offset list A = (α_1, …, α_M) of 4.3. Split cluster (M = 20): G_1 = positions 0–9 with A_1 = α of TR rays (1, 2, 3, 4, 5, 6, 7, 8, 19, 20), G_2 = positions 10–15 with A_2 = α of TR rays (9, 10, 11, 12, 17, 18), G_3 = positions 16–19 with A_3 = α of TR rays (13, 14, 15, 16) (Table 7.5-5, ascending TR ray order).
- Within a group, the ray at position m with rank k of x_{b_q}(m) among the group's values (k = 0 for the smallest; equal values: lower position first) gets the offset A[k] for quantity q.
- The four ranks are independent, so AOD–AOA, ZOD–ZOA and AOD–ZOD are coupled randomly (step 8). Evaluated at the drop and kept (the values come in as `rnd_ray`); batch-independent; an unsplit cluster's coupling does not depend on which other clusters are split. Each quantity is a permutation of A within each group (exact test).

### 4.5 Ray delays and taps (step 11, 7.5-26, Table 7.5-5; P4, P10, P21, S6)

- LOS and GR rays: the slot delays (slot 0 is 0; GR (d_GR − d3D)/c from `ssf_clusters`). tap 0 and 3.
- TR methods, NLOS: τ = τ_s + Δ_i·c_DS(f_i_freq) with Δ = (0, 1.28, 2.56) for sub-cluster i of a split cluster, Δ = 0 otherwise; τ_s = slot delay of the geometry slice. tap = 3·(1 + G + n) + i (i = 0 when not split).
- Split selection (TR methods, M = 20): per link, among the NLOS clusters with coefficient power(l, s, i_freq) > 0, the two largest (equal powers: lower n first); one such cluster: only that one; none: no split. LOS and GR rays are never split. The powers of 7.5-6 order (coefficient powers), not 7.5-8 (TR ambiguity A-1).
- `quadriga`, NLOS: τ''_m = 2·c̄_DS·x_14(m); τ'_m = τ''_m − (1/M)·Σ_m τ''_m (zero-mean per cluster instead of the minimum subtraction of 7.6.2.2 item 2); τ = max(τ_s + τ'_m, 0) (clamp, S6); shared across frequencies; tap = 3·(1 + G + n). Measured clamp frequency (S6): 0.2–1.4 % of the active NLOS rays, ≤ 2.9 % of the NLOS power overall, InH NLOS 76–86 % of the links have one, UMa NLOS and InF NLOS none or ≤ 0.01 %, DS unchanged at p50; a clamped ray takes the solver fallback. Zero-mean offsets keep the series-1 exact DS (ray DS / slot DS p50 1.000–1.001, p90 ≤ 1.009; F12).

### 4.6 Ray powers

LOS, GR: the slot powers per frequency. NLOS: P(l, s, k)/M per frequency (sub-cluster powers |R_i|/M·P_n follow from the ray counts 10, 6, 4).

### 4.7 Ray angles (step 7, 7.5-13/18/20; P3)

Cluster spreads in degrees: c_ASD = `PerClusterAS_D`, c_ASA = `PerClusterAS_A`, c_ZSA = `PerClusterES_A`; c_ZSD(l, k) = `PerClusterES_D_scale`·10^(lsp_mu(l, k, 5)) if `PerClusterES_D_scale` > 0, else `PerClusterES_D` (7.5-20 and 7.6-7, LOS and NLOS; TR ambiguity A-2). TR methods use frequency `i_freq`; `quadriga` uses the arithmetic mean over the frequencies of c_ZSD(l, k) (P5).

- `angle_model = "additive"` (default; TR): azimuth φ = wrap(φ_c + c·α·π/180) to [−π, π); zenith θ = 90° − el_c·180/π + c·α (degrees), folded: t = θ mod 360 ∈ [0, 360), t ≥ 180 → 360 − t, elevation = (90 − t)·π/180. AOA (c_ASA, α of q = AOA), AOD (c_ASD), EOA (c_ZSA), EOD (c_ZSD). Same fold for ZOD as for ZOA (series-1 D5).
- `angle_model = "rotation"` (P3, NG): per end, v = (cos e·cos a, cos e·sin a, sin e) with a = c_az·α_az·π/180, e = −c_el·α_el·π/180 (c_az, c_el = c_ASA, c_ZSA at the Rx; c_ASD, c_ZSD at the Tx), rotated by R(φ_c, el_c) = [[cos el_c·cos φ_c, −sin φ_c, −sin el_c·cos φ_c], [cos el_c·sin φ_c, cos φ_c, −sin el_c·sin φ_c], [sin el_c, 0, cos el_c]] (maps (1, 0, 0) to the cluster direction); ray azimuth = atan2(w_y, w_x) wrapped, elevation = atan2(w_z, hypot(w_x, w_y)) with w = R·v. The minus sign of e makes both models equal at el_c = 0 (R0: 8.9·10⁻¹⁶ rad). Measured (F16): the rotation widens the GCS azimuth spread by +9 % at 20–30° and +19 % at 30–40° cluster elevation; link-level ASA/ASD/ZSD change ≤ 1 %, ZSA −1…−4 %.

### 4.8 Polarisation of NLOS rays (steps 9, 10, 7.5-21/22/28, 7.6.16; P16, P19)

Phases: Φ_b = 2π·Φ_N(x_b) − π for b = 5 (θθ), 6 (θφ), 7 (φθ), 8 (φφ) (series-1 W9); all 0 with `random_phase = false`.

- `pol_model = "38.901"` (default): X = μ + σ·x_4 in dB with μ = lsp_mu(l, k, 7), σ = lsp_sigma(l, k, 7) (7.5-21); κ = 10^(X/10); without `pol_variability` η = 1; with it Q_b = `PolVar_sigma`·x_b for b = 9 (θθ), 10 (θφ), 11 (φθ), 12 (φφ), η'_b = 10^(Q_b/10), η_b = η'_b·(2 + 2/κ)/(η'_θθ + η'_φφ + (η'_θφ + η'_φθ)/κ) (7.6-61/62). J = [[√η_θθ·e^{jΦθθ}, √(η_θφ/κ)·e^{jΦθφ}], [√(η_φθ/κ)·e^{jΦφθ}, √η_φφ·e^{jΦφφ}]] (7.5-28, 7.6-60). `quadriga`: x shared, μ, σ, κ and thus η per frequency (P5; decision on η shared confirmed in review round 2). 7.6.16 only for NLOS rays. With `PolVar_sigma = 0` (key default) η = 1 (documented). Measured (A6): XPR of the means 6.95 → 6.41 dB for XPR 8 ± 3 dB.
- `pol_model = "quadriga"` (rotation model, NG): X_lin = 10·lg(lsp(l, k, 7)) + 0.1·σ·x_4, X_c = 10·lg(lsp(l, k, 7)) + 0.1·σ·x_13 (one value per cluster), σ = lsp_sigma(l, k, 7); γ = atan(10^(−X_lin/20)), κ_c = atan(10^(−X_c/20)); J_NG = e^{jΦθθ}·[[cos γ·e^{−jκ_c}, sin γ·e^{jκ_c}], [sin γ·e^{−jκ_c}, −cos γ·e^{jκ_c}]]. Owner end of the link from its drop positions: the end with the larger z; if the z differ by ≤ 1 mm, the larger x; if the x also differ by ≤ 1 mm, the larger y; if all three differ by ≤ 1 mm, the Tx (spec choice). J = J_NG when the Tx is the owner, J_NGᵀ otherwise (P19). Exact swap reciprocity, all NG statistics (F13: co-located X-pol singular-value ratio 0.10–0.22 dB vs 7.6–8.3 dB for the TR model; instantaneous XPR within ±0.4–0.6 dB of the link XPR). Documented price: two separately dropped UTs at equal heights on either side of the tie plane get M and Mᵀ.
- Swap reciprocity of the TR model comes from the mirror pairs 6↔7 and 10↔11 (Φθφ↔Φφθ, η_θφ↔η_φθ) and the self-mirrored blocks 4, 5, 8, 9, 12 (review round 2: max |M_BA − M_ABᵀ| = 0.0 including 7.6-62).
- M of the ray = J in the 8-row layout (section 2.1), per frequency.

### 4.9 LOS and GR rays (7.5-29, 7.6-32…40; series-1 W7, W18, W22)

- LOS: J = diag(1, −1) at all frequencies.
- GR: J = diag(R∥, −R⊥)/R with R² = (|R∥|² + |R⊥|²)/2, from TR 7.6-38/39 with ε = `gr_eps_re` − j·`gr_eps_im`, cos θ = −(h_t + h_r)/d_GR', sin²θ = (d2D/d_GR')², d_GR' = max(d_GR, 10⁻⁵ m); a zero denominator gives that coefficient 0; R = 0 gives J = 0 (the series-1 rules of `ssf_clusters`). The slot power already holds the method's power factor: TR methods K/(K+1)·(d3D/d_GR)²·R², `quadriga` K/(K+1)·R².

### 4.10 Two-bounce solver (P13, F9, F14)

For an NLOS ray with power > 0 at some frequency: T = `tx_pos`, R = `rx_pos`, u_d = (cos eod·cos aod, cos eod·sin aod, sin eod), u_a likewise from (aoa, eoa), L = |R − T| + c·delay; D = T − R, w = u_d − u_a; A = 4 − w·w, B = 4L + 2·D·w, C = L² − D·D; s = 2C/(B + √max(B² − 4AC, 0)) if C > 0, else 0; a non-finite s is 0.

- s ≥ d_min: FBS = T + s·u_d, LBS = R + s·u_a (balanced rule, equal distances from both ends; polyline = L; angles exact); `fallback` 0.
- s < d_min: FBS = T + d_min·u_d, LBS = R + d_min·u_a; `fallback` 1; polyline 2·d_min + |D + d_min·w| > L (shortest polyline with both scatterers at d_min along the ray's own directions; angles exact, delay grows).
- Why one test suffices: f(a, b) = a + b + |D + a·u_d − b·u_a| is non-decreasing in a and b (|w| ≤ 2), so a solution with a, b ≥ d_min exists iff f(d_min, d_min) ≤ L iff s ≥ d_min. The closed form is the smallest root of (4 − |w|²)s² − (4L + 2D·w)s + (L² − |D|²) = 0 in the cancellation-free form; spec scratch: within 1.8·10⁻¹⁵ (relative) of a bisection of 2s + |D + s·w| = L over 2000 random rays.
- LOS: FBS = LBS = (T + R)/2. GR: h_t = T_z, h_r = R_z; h_t + h_r > 0: G = T + h_t/(h_t + h_r)·(R' − T) with R' = (R_x, R_y, −R_z); else G = (T + R)/2; FBS = LBS = G. Zero power at all frequencies: FBS = LBS = (T + R)/2.

### 4.11 Path assembly (`ssf_paths`; P7, P8, P15, P1)

Per kept ray r and frequency k, with Δr_tx = tx_pos − rays.tx_pos, Δr_rx = rx_pos − rays.rx_pos:

1. Geometry. "phase": L_r = |rays.rx_pos − rays.tx_pos| + c·τ_r, angles of the ray, M_r(k) × exp(j·2π·f_k/c·(r̂_rx·Δr_rx + r̂_tx·Δr_tx)) with r̂_tx = unit(aod, eod), r̂_rx = unit(aoa, eoa) (7.5-22, 7.6-31, 7.6-45; sign checked in R0 against the exact spherical phase, second-order error bound held). "drift": NLOS L_r = |T − FBS| + |FBS − LBS| + |LBS − R| at the new ends T, R; departure angles towards the first of (FBS, LBS, R) that differs from T, arrival angles towards the first of (LBS, FBS, T) that differs from R; M unchanged. LOS: midpoint of the new direct line, L = |R − T|, angles of the direct line, M unchanged. GR: G from the new ends (4.10), L = |T − G| + |G − R|, angles towards G; when an end moved: R∥(t), R⊥(t), R(t) from 4.9 at the new geometry and the frequency's permittivity, coefficient power × g(t)/g_0, M = diag(R∥(t), −R⊥(t))·‖M_0‖_F/(√2·R_0) (R_0 at the ray-data positions; R_0 = 0 keeps M_0), with g = (d3D/d_GR')² for the TR methods and 1 for `quadriga` (P2 choices: GR power factor of the method recomputed; spec rule for the Jones, exact at the ray-data positions for any scalar normalisation of M_0, spec scratch 4.4·10⁻¹⁶; equal to the TR form P_0/(g_0·R_0²)·g(t)·|diag(R∥, −R⊥)|², spec scratch 1.7·10⁻¹⁶).
2. K rescale (P8): S_L = coefficient power of the LOS ray at k (0 without LOS ray), S_N = Σ of the NLOS coefficient powers at k, r = `kf_ratio`(k); if S_L + S_N > 0: a_L = r·(S_L + S_N)/(r·S_L + S_N), a_N = (S_L + S_N)/(r·S_L + S_N) (r·S_L + S_N = 0: a_L = 0, a_N = 1); LOS and GR × a_L, NLOS × a_N. This equals the TR coefficient powers with K' = r·K (spec scratch 3.3·10⁻¹⁶); LOS + NLOS keep their sum, GR on top.
3. Link gain: × 10^(`link_gain_dB`(k)/10).
4. Oxygen (7.6.1 with `oxygen`): × 10^(−α(f_k)·L_r/10⁴) (α in dB/km, L in m); α of Table 7.6.1-1 at 52, 53, …, 68 GHz = 0, 1, 2.2, 4, 6.6, 9.7, 12.6, 14.6, 15, 14.6, 14.3, 10.5, 6.8, 3.9, 1.9, 1, 0 dB/km, linear interpolation, 0 below 52 and above 68 GHz. τ_Δ not added (P1, deviation 4.16).

### 4.12 b_med, b_eff and the validity report (P7, P15, S1)

- b_end per ray: Tx: dist(FBS, tx_pos) for NLOS, d3D for LOS, d_GR for GR; Rx: dist(LBS, rx_pos), d3D, d_GR (at the ray-data positions).
- b_med (per end): the NLOS rays with power(r, i_freq) > 0, fallback rays included, sorted by b ascending (equal b: lower ray index first); the first b whose cumulative power reaches half of their total power ("lower weighted median"). NaN without scatterers or without such rays.
- b_eff (per end): sqrt(Σ P_r / Σ (P_r/b_r²)) over the kept rays at i_freq with P_r = power(r, i_freq) > 0 (NLOS, LOS, GR); NaN without scatterers or when no ray has P > 0.
- Limits (S1): `phase` sqrt(λ·b_eff/8) per end, λ = c/f(i_freq), factor 1 for all methods (measured true/predicted p10 ≥ 1.05, p50 1.2–1.5 in 20 cases); `drift` 0.2·b_med per end (F17). b NaN → limit Inf. `ratio` per end = max over snapshots of dist(end, ray-data position)/limit (0 with limit Inf); `ratio_total` = sqrt(ratio_tx² + ratio_rx²) (both ends: measured p50 1.00–1.03 for `phase`, conservative reading for `drift`). Valid ≤ 1; `drift` new segment from 2.5 (0.5·b_med).
- `phase` without scatterers: `gbsm_single_link_channel` solves a copy with d_min = 2 m for the report only.

### 4.13 QRT folding (P12, S4)

Per path p, frequency k: L_p = polyline (LOS d3D, GR |T − G| + |G − R|, NLOS T → FBS → LBS → R) in double; J_stored = M(:, p, k)·sqrt(path_gain(p, k))·L_p·10^((32.45 + 20·lg(f_k/10⁹))/20), which inverts the FSPL of `qrt_file_read` (32.45 + 20·lg f_GHz + 20·lg d, `qd_gain_at_1m` of `src/qd_atmospheric_absorption.hpp`); `qrt_file_read` then returns path_gain·|M|² within float32 (F2: 4.7·10⁻⁸), for `normalize_M` 0 and 1. Interaction codes 129 (GR), 193, 193 (NLOS); `nREF`, `nSCT` as 3.8.

### 4.14 Tap sum, per-element tap delay and the planar speed-up (P17, S2)

- Tap q, element pair (u, s), snapshot t: coefficient = Σ_{r ∈ q} c_r(u, s, t); delay = Σ w_r·τ_r(u, s, t) / Σ w_r with w_r = rays.power(r, i_freq) (antenna-independent; NG's unweighted mean for equal ray powers); equal weights when Σ w_r = 0. Measured (S2): the per-element delay deviates 0.2–2.3 ns from the phase-centre delay on 8×8 to 32×8 arrays at 3.5 GHz.
- "phase" speed-up: c_r(u, s, t) = c_r^unit(u, s, o_t)·sqrt(g_r(t))·exp(j·φ_r(t)), with c_r^unit the planar coefficient for unit path gain and the ray's M at the orientation pair o_t, g_r(t) the path gain of 4.11 at t, φ_r(t) the displacement phase of 4.11; c_r^unit is recomputed when the orientation pair differs from the previous snapshot's. Equal to per-snapshot calls by linearity (test within 1e-12 relative).

### 4.15 TR ambiguities and their resolution

| ID | Clause | Ambiguity | Resolution |
|---|---|---|---|
| A-1 | 7.5 step 11 | "two strongest clusters": by which power | coefficient powers (7.5-6 order); LOS and GR never split; zero-power clusters not candidates; one active cluster only that one; `38.901_MF` per frequency (P4) |
| A-2 | 7.5-20, 7.6-7 | ZOD ray spread in LOS ("same procedure as ZOA using 7.5-17") | 3/8·10^μlgZSD in LOS and NLOS; the LOS sentence concerns the cluster angles only |
| A-3 | 7.6.1 | τ_Δ of 7.6-1 and the absolute TOA (TR 7.6.9): the literal 7.6-1 omits Δτ although the delays contain it | each path's own length, τ_Δ not added (P1); open point 11.3 |
| A-4 | 7.5 step 8 | "couple randomly": no procedure given | one rank field per quantity (AOD, AOA, ZOD, ZOA), same joint distribution as the TR's (rays of a group exchangeable); the AOA field is needed for the swap mirror (P6, P19) |
| A-5 | Table 7.5-3 | offsets defined for M = 20 only | 4.3 (S3) |
| A-6 | 7.6.16 | Q ~ N(0, 3²) fixed; applies to "NLOS channel coefficients" | std from `PolVar_sigma` (v19 files: 3); NLOS rays only; LOS and GR unchanged |
| A-7 | 7.5-18 | zenith fold stated for ZOA only | same fold for ZOD (series 1) |
| A-8 | 7.6.1 | "all rays within one cluster share common oxygen absorption" | each path's own length (P1; +0.13 dB at most for sub-cluster rays) |
| A-9 | 7.5-22, 7.6-31, 7.6-45 | Doppler term written as r̂·v·t | displacement form r̂·Δr (equal for linear motion; the library takes positions) |
| A-10 | 7.6.10 | dual mobility and reciprocity not defined | mirrored random sets (series 1 A7); Tx displacement term of 7.6-45 in `phase` |

### 4.16 Deviations from the TR (with measured size)

- Oxygen (P1, review round 2): own path length, τ_Δ not added: at 60 GHz in the TR methods without absolute TOA 0.017 dB median, 0.71 dB at p99 (UMi NLOS), 1.28 dB at p99 (UMa NLOS); with absolute TOA −0.13 dB at p50 to +1.05 dB at p99; sub-cluster rays up to +0.13 dB more loss.
- `taps` in TR LOS: one tap more than the TR's tap count (LOS tap and the first sub-cluster of the τ = 0 cluster both at delay 0; P10).
- Non-TR options (all documented as such): `angle_model = "rotation"` (F16), `pol_model = "quadriga"` (F13), `drift` mobility of a TR drop (P2), `quadriga` ray delays (zero-mean, clamp; F12, S6), M ≠ 20 offsets (S3), `kf_ratio` (P8), the GR Jones rule in `drift` (4.11).
- `phase` follows the TR rotation literally; its validity limit (S1) is a model property, not a deviation.

## 5. Random sets (P22, S5)

### 5.1 Set table

This table replaces the large-scale generator table of the section description and the column list of `ssf_draw`; it goes into the GBSM section description (section 7.2).

| `set_type`, `set_id` | `n_gen` | Blocks | Decorrelation distance | Mirror pairs | Seed tag | Consumer |
|---|---|---|---|---|---|---|
| "lsf" | 11 | 0–7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 8 absolute TOA, Normal(0,1); 9 ground permittivity, 10 UMa hE, Uniform(0,1) | the LSP keys `DS_lambda` … `XPR_lambda`; `absTOA_lambda`; `SC_lambda` (9); `SF_lambda` (10) | ASD↔ASA, ESD↔ESA | none (seed used directly) | `lsp_generate` `rnd`; `path_loss` `rnd_he` (column 10) |
| "cluster", −1 | 9·N | 0 cluster delay, Uniform(0,1); 1, 2 sign AOD, AOA; 3, 4 sign ZOD, ZOA; 5, 6 offset AOD, AOA; 7, 8 offset ZOD, ZOA, Normal(0,1); column b·N + n | `SC_lambda` | 1↔2, 3↔4, 5↔6, 7↔8 | 1 | `ssf_clusters` `rnd` |
| "cluster", k | N | cluster shadowing, Normal(0,1) | `SC_lambda` | self | 2 + k | `ssf_clusters` `rnd_freq`, slice k |
| "ray", −1 | 14·N·M + N | 0–14 (5.2) | `SC_lambda`; block 14: λ_τ (4.2) | 5.3 | 2^63 + 2^32 + 1 | `ssf_rays` `rnd_ray` (`quadriga`, one slice) |
| "ray", k | 14·N·M + N | 0–14 (5.2); block 14 unused until TR 7.6.2.2 (10.3) | `SC_lambda` | 5.3 | 2^63 + 2^32 + 2 + k | `ssf_rays` `rnd_ray` slice k (TR methods, frequency k) |

A decorrelation distance of 0 is used as 0.1 m in every set, except block 14 of the shared "ray" set, which uses λ_τ also when `SC_lambda = 0` (review round 2, A2-3: 0.1 m would give a per-tap correlation of 0.33 at 0.5 mm). Bias removal: "lsf" columns 0–8 only (as before); no SSF generator is bias-corrected.

### 5.2 Blocks of the "ray" sets

Column b·N·M + n·M + m for blocks 0–12 (ray m of NLOS cluster n), 13·N·M + n for block 13, 13·N·M + N + n·M + m for block 14:

| Block | Variable | Distribution | Mirror | Used by |
|---|---|---|---|---|
| 0, 1 | coupling rank AOD, AOA (step 8) | Normal(0,1) | 0↔1 | all methods |
| 2, 3 | coupling rank ZOD, ZOA | Normal(0,1) | 2↔3 | all methods |
| 4 | XPR per ray: κ (TR model, 7.5-21) or γ (rotation model) | Normal(0,1) | self | both models |
| 5 | phase Φθθ (rotation model: its single phase) | Normal(0,1), mapped with Φ_N | self | both models |
| 6, 7 | phase Φθφ, Φφθ | Normal(0,1), mapped with Φ_N | 6↔7 | TR model |
| 8 | phase Φφφ | Normal(0,1), mapped with Φ_N | self | TR model |
| 9 | η θθ (7.6.16) | Normal(0,1) | self | `pol_variability` |
| 10, 11 | η θφ, η φθ | Normal(0,1) | 10↔11 | `pol_variability` |
| 12 | η φφ | Normal(0,1) | self | `pol_variability` |
| 13 (N columns) | circular XPR κ per cluster | Normal(0,1) | self | rotation model |
| 14 | ray delay u, τ'' = 2·c̄_DS·u | Uniform(0,1) | self | `quadriga` (shared set); per-frequency sets: unused |

### 5.3 Generator indexing and mirror map

- Mirror map m(g) of the "ray" sets: for g = b·N·M + n·M + m with b ≤ 12, m(g) = b'·N·M + n·M + m with b' = (1, 0, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12)[b]; blocks 13 and 14 self-mirrored. Slice 1 of `phase` holds column m(g) of slice 0 in column g, as for "lsf" and "cluster".
- Dual mobility: a self-mirrored generator is (f_g(R) + f_g(T))/sqrt(2 + 2·A_g(D)), a pair (f_g(R) + f_m(T))/sqrt(2), with the existing `qd_gbsm_eval` rules; swapping `tx_pos` and `rx_pos` exchanges the mirror pairs and keeps the self-mirrored blocks, exactly.
- Uniform blocks (cluster 0, ray 14) map the Normal field with 0.5·erfc(−x/√2) (existing rule).
- A column depends neither on the other columns nor on the other links (`sos_init_phases` fills column g from one stream independently of n_gen; the evaluation of a column uses its own phases only).

### 5.4 Rules

- "lsf" and "cluster" keep generators, distributions, decorrelation distances, mirror pairs, seed rules and evaluation bit for bit: the phases are equal to those of the former functions for every seed (platform-independent), the draws equal within the build's `sos_val` kernel (the same code path). Stored seeds and phase arrays reproduce series-1 results.
- A set never changes size: new random quantities come in new sets (a new `set_type` or `set_id`) read through appended optional inputs of their consumers (Procedure A X_n, 7.6.10 α and D, 7.6.2.2 offsets and ray powers). Block 14 of the per-frequency sets is generated now so that 7.6.2.2 can use it later without a size change.
- No generator limit exists: the count follows from the set type (finding 8 corrected; N ≤ 999 only because `NumClusters` ≤ 1000).
- Size and cost at N = 19, M = 20: 5339 generators per "ray" set, phases `[256, 5339, 2]` ≈ 22 MB per set, draws ≈ 33 ns per value and link (≈ 0.18 ms per link, provisional; plan 5.3). M = 1000 (7.6.2.2 later): ≈ 2.7·10⁵ generators, ≈ 1.1 GB of phases, ≈ 9 ms of draws per link and frequency (plan 7.3).

## 6. Test plan

### 6.1 Sources of expected values

- Closed forms and in-test transcriptions of section 4 (scalar double math in the test file or the shared header), written without the code under test (`md/SPEC_CPP_API.md`, "Fixtures and oracles").
- Spec scratch: numbers measured for this spec with small scripts (outputs quoted where used): `s1_offsets.py` (offsets of 4.3 for M = 1…26; NG's rule against its printed tables), `s2_checks.py` (K rescale of 4.11, GR Jones rule of 4.11, λ_τ values of 4.2 from the v19 conf files read with the library's `conf_file_read`, solver closed form against bisection over 2000 random rays), `s3_tolerances.py` (QRT phase errors from F2, the `sos_val` kernel bound, standard errors, phase limits), `s5_cds_files.py` (c_DS keys of the 23 shipped files). The declarations of 3.4 and 3.10 were syntax-checked against the repository headers.
- The R0 prototype (NumPy, TR-literal ray generator on library clusters, self-test 42/42; location in the run briefs) as an independent cross-check during R4, R5, R6, R7. Its results are reported in the hand-overs, not stored in the repository; no Catch2 value depends on it. Use and required changes:

| Use | R0 function | Changes for the cross-check |
|---|---|---|
| TR rays (R4) | `make_drop`, `gen_rays(angle_model='additive', pol_model='38.901')` | feed the library "ray" set columns into its field dict (its coupling block order is AOA, AOD, ZOA, ZOD = library blocks 1, 0, 3, 2; its `xpr` set = blocks 4 and 13, `phase` = 5–8, `eta` = 9–12); permute the rays of split clusters to the library order (positions 0–9 ↔ TR rays 1–8, 19, 20; 10–15 ↔ 9–12, 17, 18; 16–19 ↔ 13–16) for the field input and the outputs; M = 20 only; its `make_drop` calls the pre-R1 function names (use the pre-R1 build it was made for, or rename the calls) |
| `quadriga` ray delays (R5) | `gen_rays(ray_delays='7.6.2.2')` | zero-mean per cluster instead of the minimum subtraction, clamp at 0, u from block 14 (Uniform) instead of Φ_N of a Normal field, tap = cluster |
| rotation polarisation model (R5) | `gen_rays(pol_model='quadriga')` | owner-end transpose added (its matrix is NG's without it) |
| 7.6.16 (R5) | `gen_rays(pol_variability=True)` | none besides the field mapping |
| rotation angle model (R5) | `ray_angles(model='rotation')` | none |
| solver (R6) | `solve_two_bounce` | none (balanced rule, d_min to the own end, fallback as 4.10) |
| path assembly (R7) | `phase_rotation`, `oxygen_loss_db`, `spherical_coeff`, `planar_coeff` | oxygen from each path's own length instead of cluster delay + τ_Δ |
| taps (R9) | `tap_sum`, `tap_delays` | none (power weights) |

### 6.2 Shared test headers

- `tests/catch2_tests/qd_test_ssf.hpp` (series 1, unchanged except the R1 call migration): cluster fixture `make_fix` / `run`, LOS and GR geometry oracles.
- `tests/catch2_tests/qd_test_rays.hpp` (new, R2; ≤ 300 lines at R2, ≤ 500 at the end): includes `qd_test_ssf.hpp`; holds only setup and oracles independent of the functions under test:
  - `make_rnd_ray(n_link, N, M, n_set, seed)`: a "ray" set cube with distinct values (blocks 0–13 uniform in (−2.5, 2.5), block 14 in (0.02, 0.98); `std::mt19937` as `make_fix`), so `ssf_rays` tests do not depend on R3;
  - `make_lsp_stats(fix, seed)`: `lsp_mu`, `lsp_sigma` `[n_link, n_freq, 8]` with distinct finite values (slice 5 in (0.2, 1.2) lg deg, slice 7 μ in (5, 12) dB, σ in (2, 4) dB, others finite) to complement `make_fix`;
  - `tr_offsets(M)`: the offsets of 4.3 written out (Table 7.5-3 constants, NG's table, the Laplacian law), and the Table 7.5-5 groups;
  - `unit_vec(az, el)`, `polyline(T, FBS, LBS, R)`, `specular_point(T, R)`, `fresnel(eps_re, eps_im, h_t, h_r, d2D)` (the latter reusing the series-1 oracle);
  - `make_rays(seed, has_los, has_gr, n_nlos, n_freq, method)`: a hand-made `gbsm_rays` (Tx (3, −4, 25), Rx (120, 45, 1.5); LOS ray with geometric angles, delay 0, M diag(1, −1); optional GR ray with geometric delay and angles and diag(R∥, −R⊥)/R; NLOS rays with distinct delays 10–500 ns, angles, powers and M entries; one zero-power NLOS ray; taps 0, 3, 3·(1 + G + n)), so R6–R9 do not need `ssf_rays`;
  - edit rule: R2 writes it; R4, R5, R7, R9 may append (sequential); R6 and R8 (parallel worktrees) put their helpers into their own test files.
- Method-generic cases loop over `qd_methods` (`qd_test_ssf.hpp`): layout, batch independence, pre-sized outputs, validation of `ssf_rays`; R4 skips "quadriga" in these loops, R5 removes the skip.

### 6.3 Tolerances (justification)

- Exact-by-construction double results (transcriptions, rank permutations, layout, swap of mirrored inputs): 1e-12 relative or absolute in rad, codes and indices exact; swap reciprocity of ray angles 1e-10 rad (series-1 value of the cluster swap test, `test_ssf_clusters_quadriga.cpp`).
- Offsets of 4.3 against the quoted 6-decimal values: 1e-6; rms and mean properties: 1e-12.
- Solver: polyline residual 1e-9 m, angle residual 1e-10 rad (R0: 1.4·10⁻¹² m and 5.8·10⁻¹¹ rad on library drops; the wider bound covers lengths up to km); closed form vs bisection 1e-12 relative (spec scratch 1.8·10⁻¹⁵).
- "lsf"/"cluster" golden values: phases exact (platform-independent); draws 2e-5 absolute (two `sos_val` kernels each within 4e-7·sqrt(2·256) = 9.05·10⁻⁶ of a double evaluation, spec scratch: difference ≤ 1.81·10⁻⁵).
- Marginal distributions of draws: 5 standard errors at 10⁴ links (spec scratch: Normal mean 0.05, Uniform mean 0.0144 → 0.015 as the series-1 test, fraction 0.025), plus the series-1 0.05 on the std.
- QRT (F2, spec scratch): gain·|M|² 1e-6 relative (float32, measured 4.7·10⁻⁸); path length 1e-6 relative (2.7 µm at ≤ 150 m); coefficient phase at 3.5 GHz ≤ 1.98·10⁻⁴ rad for coordinates ≤ 150 m → per coefficient |Δc| ≤ 5e-4·|c| + 1e-6·max|c|; delays 1e-13 s (9.0·10⁻¹⁵ s measured bound ×10).
- `phase` against exact spherical geometry: the second-order bound k·(|Δr_rx|² + |Δr_tx|²)/(2·(b − |Δr|)) per ray (R0 self-test, held at ratio ≤ 1.00).

### 6.4 Test cases per run

Size targets include the series-1 growth of 1.5–2× over plan estimates; no file beyond about the 983 lines of today's `test_ssf_clusters.cpp`.

**R1** (existing tests migrated; MEX and Python tests merged):

| File | Cases (one line each: assertion; oracle) |
|---|---|
| `test_gbsm_phases_lsf.cpp` (git mv of `test_gbsm_phases.cpp`) | all former `lsf_*` cases renamed `"gbsm_init_phases - lsf: …"` / `"gbsm_draw - lsf: …"`; new: "gbsm_init_phases - lsf phases equal the previous version" (golden values, exact); "gbsm_draw - lsf values equal the previous version" (golden, 2e-5); "gbsm_init_phases - input validation" (set type, positions with other set types, existing sections) |
| `test_gbsm_phases_cluster.cpp` (git mv of `test_ssf_phases.cpp`) | former `ssf_*` cases renamed `"… - cluster: …"`; new golden cases for sets −1, 0, 3 (phases exact, draws 2e-5, one dual-mobility and one floor case); "gbsm_draw - input validation" (set type; `center_frequency` ignored for "lsf" and "cluster") |
| `test_conf_file_read.cpp`, `test_conf_file_write.cpp`, `test_scenpar_validate.cpp`, `test_conf_files_v19.cpp` | `SubpathMethod = legacy` accepted; another value throws with key and value in the message; the writer never writes the key (also with `write_defaults`); the 23 shipped files carry no `SubpathMethod` line; the two v19 UMa files read `PerClusterDS_fc_min = 6`; the optional-output cases at lines 913, 926–938 of `test_conf_file_read.cpp` adapted (they must fail before the fix, see 9 R1) |
| tests of `los_probability`, `lsp_generate`, `o2i_loss`, `path_loss` | messages naming `'center_frequency'` where they are asserted |
| MEX `test_gbsm_init_phases.m`, `test_gbsm_draw.m` (replace the four old files); Python `test_gbsm_init_phases.py`, `test_gbsm_draw.py` | the old checks with the new usage lines; `set_id` 0 / k ≥ 1 (MEX), −1 / k (Python); struct/dict rule: field `SubpathMethod` with `legacy` accepted, `mmMAGIC` throws naming key and value, outputs without the field; Python `center_freq=` keyword |

Golden values: before any edit, R1 builds the unchanged library and records with a small program (kept in the run's scratch, not in the repository) 8 phase values per set (positions spread over rows, columns and both slices) for seeds 7 and 123456789, sets "lsf" (with and without bias removal over 5 positions), "cluster" −1, 0, 3 (v19 UMi NLOS, `Comb256`), and 6 draw values per set at 3 links (single, dual mobility, one with `floor_id` = 2), printed with 17 significant digits; the test comment says "values of quadriga-lib before the merge of the random-set functions". Size: migration only plus ≈ 200 new lines.

**R2**: `test_gbsm_rays.cpp` (target 250, cap 400): "gbsm_rays - is_valid accepts a valid object, an empty one and n_ray = 0" ; "gbsm_rays - is_valid reports each failed check" (one SECTION per row of the table of 3.4, exact message); "gbsm_rays - n_ray, n_freq and empty". The stubs of all new functions compile and link; empty test files `test_ssf_rays.cpp`, `test_ssf_rays_quadriga.cpp`, `test_gbsm_rays_solve_scatterers.cpp`, `test_ssf_paths.cpp`, `test_qrt_paths_from_scatterers.cpp`, `test_gbsm_rays_from_paths.cpp`, `test_gbsm_single_link_channel.cpp`, `test_gbsm_phases_ray.cpp` with their header comments; `qd_test_rays.hpp` (6.2).

**R3**: `test_gbsm_phases_ray.cpp` (target 450, cap 700):

| Case | Oracle |
|---|---|
| "gbsm_init_phases - ray: n_gen is 14·N·M + N, empty for N = 0" (N = 19, M = 20 → 5339; M = 1; M = 7) | layout formula |
| "gbsm_init_phases - ray: slice 0 equals sos_init_phases with the derived seed" (sets −1, 0, 3) | SplitMix64 transcription with tag 2^63 + 2^32 + set_id + 2; `sos_init_phases` (supporting) |
| "gbsm_init_phases - ray: slice 1 is the mirror of slice 0" | mirror map of 5.3 |
| "gbsm_init_phases - ray sets are distinct from each other and from the cluster sets" | no equal column between sets −1, 0, 1 and "cluster" −1 for the same seed |
| "gbsm_draw - ray: single mobility values equal the sum of sinusoids" | `sos_preset` + `sos_val` at `SC_lambda` and λ_τ (as the series-1 case) |
| "gbsm_draw - ray: block 14 of the shared set uses lambda_tau" | λ_τ of 4.2 (UMi NLOS 3.5 GHz 13.7415 m; UMa NLOS {3.5, 28} 5.8058 m before / 5.0749 m after R1's conf change); per-frequency block 14 at `SC_lambda`; `SC_lambda = 0`: blocks 0–13 at 0.1 m, shared block 14 at λ_τ; guards: `PerClusterAS_A = 0` → 10⁴ m; c̄_DS = 0 (`PerClusterDS` 1, gamma −1, min 0 at 10 GHz) → 0.1 m |
| "gbsm_draw - ray: swapping Tx and Rx exchanges the mirror blocks" | exact, dual mobility |
| "gbsm_draw - ray: marginal distributions" (custom N = 2, M = 3, 10⁴ links on the series-1 grid) | 5 standard errors (6.3) |
| "gbsm_draw - ray: center_frequency is required by the shared set and ignored otherwise" | — |
| "gbsm_draw - ray: input validation" | messages of 3.2 |

MEX/Python (R3): `test_gbsm_draw.m/.py` gain a "ray" call with `center_freq` (size `[n_link, 14·N·M + N]`), the missing-`center_freq` error, `set_id` k ≥ 1 / k.

**R4**: `test_ssf_rays.cpp` (target 900, cap 1000), fixture `make_fix` + `run` (`ssf_clusters`) on v19 UMi LOS, UMi NLOS, UMa LOS with `GR_enabled = 1`, `make_rnd_ray`, `make_lsp_stats`:

| Case | Oracle |
|---|---|
| "ssf_rays - output layout: entries, sizes, path types, taps and empty scatterers" (methods loop) | layout of 3.5 |
| "ssf_rays - 38.901 matches TR 7.5 steps 7 to 10" (all attributes of all rays, NLOS and LOS scenario, 6 links) | transcription of 4.3–4.9 (additive, TR model) |
| "ssf_rays - 38.901 splits the two strongest clusters into the sub-clusters of Table 7.5-5" (delays 0 / 1.28 / 2.56 c_DS, taps, positions by sub-cluster; equal powers; one active cluster; zero-power clusters; M ≠ 20 no split; `PerClusterDS = 0` → 3.91 ns; UMa c_DS at 3.5 and 28 GHz) | 4.2, 4.5 |
| "ssf_rays - coupling gives each quantity a permutation of the offsets within each group" | exact multiset equality per group (4.4) |
| "ssf_rays - ray offsets for M other than 20" (M = 1, 2, 3, 4, 6, 21, 26; odd M zero ray first) | 6-decimal values of 4.3 (1e-6), mean and rms (1e-12) |
| "ssf_rays - 38.901_MF computes frequency i_freq with its own slices" (2 frequencies; equals a one-frequency call with the slices of frequency 1) | consistency, exact |
| "ssf_rays - LOS and GR rays carry the slot values and the TR Jones matrices" (incl. NLOS scenario, zero LOS power kept) | 4.9 with the series-1 Fresnel oracle |
| "ssf_rays - ray delay spread with sub-clusters equals the analytic value" | DS of the ray powers and delays from the slot powers with the 10/6/4 split (1e-12 relative; R0 self-test exact) |
| "ssf_rays - a link does not depend on the other links" (methods loop) | exact |
| "ssf_rays - pre-sized entries are written in place" (attributes on external memory, strict) | data pointers unchanged, no throw |
| "ssf_rays - random_phase false gives zero initial phases" | — |
| "ssf_rays - N = 0, M = 1 and n_link = 0" | layout |
| "ssf_rays - zenith fold at the poles" (cluster elevations ±89°, offsets across the pole) | 4.7 fold |
| "ssf_rays - input validation" (methods loop; one SECTION per message of 3.5) | messages |

**R5**: `test_ssf_rays_quadriga.cpp` (target 800, cap 1000):

| Case | Oracle |
|---|---|
| "ssf_rays - quadriga matches the algorithm at one and several frequencies" ({3.5}, {3.5, 28} GHz; TR and rotation polarisation model) | transcription of 4.2–4.9 for `quadriga` (c̄_DS, mean c_ZSD, zero-mean delays, shared coupling and phases, per-frequency κ and Jones) |
| "ssf_rays - quadriga ray delays are zero-mean per cluster and clamped at 0" (a cluster near τ = 0) | mean offset 0 (1e-18 s) where no ray is clamped; clamped rays exactly 0; tap = cluster |
| "ssf_rays - quadriga shares the geometry and the phases across frequencies" | equality across frequencies, exact |
| "ssf_rays - rotation angle model equals additive at the horizon and keeps the on-sphere distances" | 4.7: equal at el_c = 0 (1e-12); great-circle distance to the cluster centre independent of el_c (1e-12) |
| "ssf_rays - rotation polarisation model is NG's matrix with the owner-end rule" (owner by z; ties within 1 mm by x, then y; all tied → Tx) | 4.8 transcription; unitarity of J_NG (1e-12) |
| "ssf_rays - polarisation variability follows TR 7.6.16" | Frobenius² = 2 + 2/κ (1e-12); NLOS rays only; `PolVar_sigma = 0` → η = 1 |
| "ssf_rays - both polarisation models are swap-reciprocal for mirrored inputs" (`gbsm_draw` "ray" and "cluster" sets with dual mobility at swapped positions, mirrored LSPs, symmetric cluster spreads; `38.901` and `quadriga`; with and without `pol_variability`; one equal-height pair) | M_B = M_Aᵀ (1e-12), departure ↔ arrival per ray (1e-10 rad), delays and powers (1e-12 relative) |
| "ssf_rays - quadriga input validation" (`i_freq` ≠ 0, block 14 outside [0, 1], `pol_variability` with the rotation model, `lsp` slice 7 ≤ 0 with the rotation model) | messages |
| "Diagnostic - ssf_rays quadriga ray delay spread relative to the slot delay spread" | prints p50/p90; asserts finite |

**R6**: `test_gbsm_rays_solve_scatterers.cpp` (target 500, cap 700) and cases appended to `test_gbsm_rays.cpp` (+150):

| Case | Oracle |
|---|---|
| "gbsm_rays::solve_scatterers - solved rays keep their delay and angles exactly" | polyline = L (1e-9 m), angles from the scatterers (1e-10 rad), equal end distances (1e-9 m), `fallback` 0 |
| "gbsm_rays::solve_scatterers - rays without a solution take the minimum-distance fallback" (zero and small excess; d_min 0.5, 2, 5 m) | FBS = T + d_min·u_d, LBS = R + d_min·u_a; polyline ≥ L; `fallback` 1 |
| "gbsm_rays::solve_scatterers - the closed form agrees with a bisection of the polyline" | bisection in the test (1e-12 relative) |
| "gbsm_rays::solve_scatterers - LOS, GR and zero-power rays" (incl. h_t + h_r = 0) | midpoint; specular point by the image method |
| "gbsm_rays::solve_scatterers - only the scatterer attributes change and given scatterers are overwritten" | equality of all other attributes |
| "gbsm_rays::solve_scatterers - d3D = 0 and n_ray = 0" | 4.10 |
| "gbsm_rays::solve_scatterers - input validation" | messages |
| "gbsm_rays - b_med is the lower weighted median of the NLOS end distances" (hand-made scatterers; fallback rays included; LOS/GR excluded; NaN cases) | sorted cumulative sum in the test |
| "gbsm_rays - b_eff is the power-weighted inverse-square mean of the end distances" (LOS d3D, GR d_GR; `i_freq` selects the powers; NaN cases) | closed form |
| "gbsm_rays - b_med and b_eff validate the object and i_freq" | messages |

**R7**: `test_ssf_paths.cpp` (target 750, cap 1000):

| Case | Oracle |
|---|---|
| "ssf_paths - phase at the ray-data positions returns the ray data" (zero-power rays dropped, LOS/GR kept, `ray_ind`, `path_type`, `fbs_pos` `[3, 0]`) | 4.11 |
| "ssf_paths - phase rotation follows TR 7.5-22 for Rx and Tx displacements" (two frequencies) | exp(j·2π·f/c·r̂·Δr) (1e-12); second-order bound against the exact polyline phase of solved rays (6.3) |
| "ssf_paths - drift at the ray-data positions reproduces the drop geometry" | polyline = L (1e-9 m) except fallback rays (longer); angles (1e-10 rad) |
| "ssf_paths - drift recomputes polylines, angles, LOS and GR at moved ends" (TR and `quadriga` GR power law; with and without permittivity at the reference) | geometry oracles of the header; GR rule of 4.11 |
| "ssf_paths - drift path lengths change by at most the displacement" | abs(ΔL) ≤ abs(Δr) + 1e-9 m (F17) |
| "ssf_paths - kf_ratio rescales LOS and GR against NLOS as K' = r·K" | TR coefficient powers with K' (1e-14) |
| "ssf_paths - oxygen loss from each path's own length" (52.5, 60, 60.5, 63.5, 67.5, 70 GHz) | Table 7.6.1-1 (α = 0.5, 15, 14.8, 8.65, 0.5, 0 dB/km; R0 self-test values) |
| "ssf_paths - auto mobility and optional outputs alone or together give the same values" | consistency |
| "ssf_paths - phase and drift agree for millimetre displacements" (via `get_channels_planar` / `_spherical`, omni elements) | second-order bound (6.3) |
| "ssf_paths - n_ray = 0, NLOS scenario and data-from-paths rays" | layout |
| "ssf_paths - input validation" | messages of 3.7 |
| "Diagnostic - ssf_paths fallback share and b_med on a library drop" (`ssf_rays` + `solve_scatterers`, UMi NLOS `38.901` and `quadriga`) | prints; compare by eye with F9, F17 |

**R8**: `test_qrt_paths_from_scatterers.cpp` (target 350, cap 600) and `test_gbsm_rays_from_paths.cpp` (target 350, cap 600); test files written in the working directory with distinct names, removed at the end:

| Case | Oracle |
|---|---|
| "qrt_paths_from_scatterers - path objects carry the geometry, the codes and the folded Jones matrices" | `n_seg`, coordinates, codes 129 / 193, `iC`, `nREF`, `nSCT`, Jones of 4.13 (float accuracy) |
| "qrt_paths_from_scatterers - qrt_file_read returns path gain times M and the polyline" (3 frequencies; `normalize_M` 0 and 1) | gain·abs(M)² (1e-6 relative), path length (1e-6 relative), `no_int` = `path_type`, `interact_type` codes |
| "qrt_paths_from_scatterers - n_path = 0 and 127 frequencies" | sizes |
| "qrt_paths_from_scatterers - input validation" | messages |
| "gbsm_rays_from_paths - delays, angles and powers from the path data" | geometry oracle (1e-12) |
| "gbsm_rays_from_paths - LOS detection without path_type and the optional tap" | 3.9 rules |
| "gbsm_rays_from_paths - a QRT round trip restores the rays" | delays 1e-13 s, angles 1e-6 rad, power·abs(M)² 1e-6 relative |
| "gbsm_rays_from_paths - a GR point off the ground throws; permittivity is optional" | messages |
| "gbsm_rays_from_paths - a pre-initialised object is written in place" | pointers unchanged |
| "gbsm_rays_from_paths - input validation" | messages |

**R9**: `test_gbsm_single_link_channel.cpp` (target 850, cap 1000):

| Case | Oracle |
|---|---|
| "gbsm_single_link_channel - rays output equals get_channels with the paths of ssf_paths" (`phase` and `drift`, 4 snapshots, dual-pol 2×2 arrays from `arrayant_generate`) | per-snapshot `ssf_paths` + `get_channels_*` (1e-12 relative) |
| "gbsm_single_link_channel - taps sum the rays per tap with power-weighted per-element delays" | sums in the test; zero-weight tap → plain mean |
| "gbsm_single_link_channel - the phase speed-up equals per-snapshot planar calls" (orientation changes at two snapshots) | 1e-12 relative |
| "gbsm_single_link_channel - LOS and GR taps are always present and taps are ascending" (NLOS scenario; TR LOS two zero-delay taps) | — |
| "gbsm_single_link_channel - par holds the validity report" (`phase` with and without scatterers, `drift`, LOS-only link) | 4.12 closed forms; limits for tests: sqrt(λ·b/8) = 1.0347 m at 3.5 GHz and b = 100 m, 0.3658 m at 28 GHz (spec scratch) |
| "gbsm_single_link_channel - the snapshot count follows the per-snapshot inputs" (broadcast of one row; mismatches throw) | — |
| "gbsm_single_link_channel - the QRT route equals the direct route in drift" (static Tx, local coordinates ≤ 150 m, a GR ray, 3 snapshots, `rays` output; read side: `qrt_file_read` → `gbsm_rays_from_paths` with `path_type` from `no_int` and `tap` from the written rays → `gbsm_single_link_channel` with `link_gain_dB = 0`) | tolerances of 6.3 (P12) |
| "gbsm_single_link_channel - results do not depend on the thread count" (`omp_set_num_threads(1)` vs default, guarded by `#ifdef _OPENMP`) | exact |
| "gbsm_single_link_channel - the channel is valid and stores the positions as columns" | `channel::is_valid()` empty |
| "gbsm_single_link_channel - input validation" | messages of 3.10 |

**R10**: MEX tests `test_ssf_rays.m`, `test_ssf_scatterers.m`, `test_ssf_paths.m`, `test_gbsm_rays_from_paths.m`, `test_gbsm_single_link_channel.m` and Python tests `test_ssf_rays.py`, `test_ssf_scatterers.py`, `test_ssf_paths.py`, `test_gbsm_rays_from_paths.py`, `test_gbsm_single_link_channel.py` (target 150–250 lines each), by the default coverage of `md/SPEC_MATLAB_API.md` / `md/SPEC_PYTHON_API.md` plus: all struct fields / dict keys present (empty where empty) on output and the round trip `rays(l) = quadriga_lib.ssf_scatterers(rays(l))` in a struct array; missing optional fields accepted, unknown fields rejected; `i_freq` 1-based and `ray_ind` +1 (MEX); `gr_eps` complex (Python); `M` real `(8, n_ray, n_freq)`; integer classes as section 8.3; the by-value channel copied (`chan`, `par`).

## 7. Documentation plan

### 7.1 Rules and size targets

- Doc blocks describe the API only: purpose, data layout, ranges, units, defaults, the errors a caller can trigger; no formulas or derivations (they go into source comments next to the code); each fact once, in the place where it fits best (description, argument list or section description); TR clauses cited instead of restated; block grammar and pitfalls of `docs_builder/README.md` (no `*` or `|` in running text, tables with blank lines around, ≤ 120 characters per line).
- Reference size: the `ssf_clusters` block is 108 lines, 1042 words, 6889 characters including the delimiters (`src/ssf_clusters.cpp:349–456`, measured). Series 1 produced 1248 doc lines that had to be cut to about 500 (`md/ssf_notes.md`, lessons); the targets below are caps for the C++ blocks (lines incl. delimiters; the MEX and Python blocks of a function stay at or below its C++ block):

| Block | Cap (lines) |
|---|---|
| `gbsm_init_phases` | 45 |
| `gbsm_draw` | 60 |
| `gbsm_rays` (class block: attributes table, members table, rules for several sources) | 75 |
| `ssf_rays` | 100 |
| `.solve_scatterers` | 40 |
| `ssf_paths` | 80 |
| `qrt_paths_from_scatterers` | 50 |
| `gbsm_rays_from_paths` | 50 |
| `gbsm_single_link_channel` (incl. the `par` table) | 100 |
| additions to the GBSM section description (set and block tables, Terms, codes, model properties, workflow) | 110 |

### 7.2 GBSM section description (`src/qd_scenpar.cpp`, `SECTION_DESC`)

- First paragraph: the series covers TR 38.901 clause 7.5 steps 1–11 (antenna-free ray level, channel coefficients by `gbsm_single_link_channel`) with 7.6.1, 7.6.3, 7.6.16; replaces "The ray level (steps 8 to 12) follows in a later version" (R10).
- Workflow paragraph: per BS site and scenario `gbsm_init_phases` once per set ("lsf"; "cluster" −1 and k; "ray" −1 for `quadriga`, k for the TR methods), `gbsm_draw` at the links; `o2i_loss`, `los_probability`, `path_loss`, `lsp_generate`, `ssf_clusters`, `ssf_rays` (one call for `quadriga`, one per frequency for the TR methods); for `drift`, `solve_scatterers` per link (`ssf_scatterers` in MATLAB and Python), whose `b_med` sets segment lengths; per link and segment `gbsm_single_link_channel` with `link_gain_dB` = −PL − O2I + 10·lg SF and `kf_ratio` = K(position)/K(drop) from the large-scale functions (the sum and ratio are the caller's, stated here once); QRT route: `ssf_paths` (`drift`) → `qrt_paths_from_scatterers` → `qrt_file_append`, back with `qrt_file_read` → `gbsm_rays_from_paths`; other path sources fill a `gbsm_rays` directly (R1 the random-set part, R10 the rest).
- "Random sets" paragraph: the set table (5.1) and the block table of the "ray" sets (5.2) with the λ_τ rule in one sentence ("block 14 of the shared set: decorrelation distance 1.6·c·c̄_DS/c_ASA, limited to 0.1 m … 10 km, c̄_DS the mean cluster delay spread over the frequencies passed to `gbsm_draw`") (a formula in words is allowed here as the definition of a generator parameter), the mirror rule, the "set never changes size" rule, and that `SC_lambda = 0` is used as 0.1 m except for that block (R1 "lsf"/"cluster", R3 "ray").
- "Terms" paragraph: the table of 2.3 (R10).
- Codes: path type (0 LOS, 1 GR, 2 NLOS) and tap index (3·slot + sub-cluster; LOS 0, GR 3; ascending in `taps`), once (R10).
- Model properties (R10, one short bullet each): separately dropped UTs keep correlated taps over 43–440 λ while measured fading decorrelates within about 0.4–1 λ: for UTs a few wavelengths apart use one drop plus offsets (F10, P6); `quadriga` per-tap correlation across bands follows the intra-cluster delays (one field; TR methods: independent per band, 7.6.5) (F11, F12); `pol_model = "quadriga"` gives a nearly unitary 2×2 cluster matrix (singular-value ratio 0.10–0.22 dB vs 7.6–8.3 dB for the TR model) (F13); the validity of `phase` and `drift` is stated in `ssf_paths` (not repeated here).

### 7.3 Function blocks (outlines)

- `gbsm_init_phases`: purpose; set types and `set_id` (pointer to the set table); seeds a fixed function of seed, set type and set id, equal on all platforms, sets independent; "lsf" bias removal bullets of the former `lsf_init_phases` condensed; TR 7.4.4 literally with `Exp256` and no bias removal; N = 0 empty; inputs, output; see also `gbsm_draw`, `sos_remove_bias`.
- `gbsm_draw`: purpose; evaluation rules (dual mobility and mirror swap, floors, accuracy within 4e-7·sqrt(2·n_coeff) of a double evaluation, `rnd` may alias the positions); `center_frequency` rule and its frequency-list dependence; "set never changes size"; dual-mobility Tx rate of block 14 (2–8× plane waves); inputs, output; see also `ssf_clusters`, `ssf_rays`, `lsp_generate`.
- `gbsm_rays`: class block (`## Attributes:` table of plan 5.5 with "set by" column; members table incl. `b_med` and `b_eff` in words: the end-to-scatterer distance scale of the `drift` and `phase` validity); rules for rays from several sources (3.4).
- `ssf_rays`: purpose (TR steps 7–10, 7.6.16; QuaDRiGa model); one geometry per call (TR methods frequency `i_freq`, `quadriga` all); ray layout and order (index rule, split clusters by sub-cluster, padding); a method × option table (angle model, polarisation model, ray delays, sub-clusters, shared vs per-frequency draws; defaults TR); M ≠ 20 in one sentence (NG's rule below 20, Laplacian quantiles above, sub-clusters only at 20); `quadriga` clamp frequency (S6 numbers); `pol_variability` with the TR model only, `PolVar_sigma`; swap reciprocity conditions; `i_freq` index spaces; inputs, output; see also.
- `.solve_scatterers` (C++ member block) and the MATLAB/Python blocks `ssf_scatterers`: purpose; rule in words (equal distances, exact angles, fallback along the own directions, d_min); LOS, GR, zero power; fallback frequency (F9 numbers); afterwards `b_med`/`b_eff`; see also.
- `ssf_paths`: lead with the use of each mobility (P7: planar with the arrays at the ray-data positions, else the phase counts twice); what `phase` and `drift` keep and recompute (table); validity limits side by side (`phase`: sqrt(λ·d3D/8) on LOS links = 1.2–1.5 m at 3.5 GHz, 0.44–0.54 m at 28 GHz; NLOS ≈ 0.35 m at 3.5 GHz and 0.11 m at 28 GHz in the TR methods, 0.49–0.67× of the LOS value in `quadriga`; per link sqrt(λ·b_eff/8); `drift` ≈ 0.2·b_med per moving end, 1.5–5.6 m LOS and 7.8–12.4 m NLOS at p50, ≈ 20 % shorter with dual mobility, new segment from 0.5·b_med; |ΔL| ≤ displacement in `drift`); the gain chain in words (K rescale, link gain, oxygen); oxygen deviation numbers (4.16); zero-power rays dropped; inputs, outputs; see also.
- `qrt_paths_from_scatterers`: purpose; folding in words; interaction codes and counters; local coordinates (x, y shifts only) with the F2 numbers; `qrt_file_init` in GHz; writer memory scaling (3.8); what `qrt_file_read` returns.
- `gbsm_rays_from_paths`: purpose; derived attributes; `path_type`/`tap` defaults and why (QRT stores neither); GR on z = 0; permittivity only for `drift` away from the positions; use with `link_gain_dB = 0`.
- `gbsm_single_link_channel`: purpose; per-snapshot rule (planar at the ray-data positions / spherical at the snapshot positions); `taps` vs `rays` with the bandwidth limit of `quadriga` taps (0.18 / 0.52 / 0.79 in-band selectivity over 20 / 100 / 400 MHz, UMi NLOS) and exactness of TR taps at the phase centre; delay shapes and memory (S2); TR LOS two taps at delay 0; positions rows in, columns out; `par` table (3.10) with the validity report (limits by reference to `ssf_paths`); no power correction; MEX/Python copy the output.
- `path` (`src/qd_path.cpp`, R8): `nREF` "Number of reflections (interaction type codes 128–255 except 193)"; `nSCT` "Number of scattering events (interaction type code 193, written by [[qrt_paths_from_scatterers]])".
- `ray_state_update` code table (`src/ray_mesh_interact.cpp`, R8): the row `192+ | 0xC0+ | … | reserved: scattering not implemented` becomes a row for 193 (0xC1; flags refl, scatter, ok; meaning "point scatterer of a GBSM path ([[qrt_paths_from_scatterers]]); ray-traced scattering is not implemented") (S4).
- `qrt_file_read` (`src/qd_channel_qrt.cpp`, R8): the `interact_type` entry adds: GBSM paths of [[qrt_paths_from_scatterers]] carry 129 (GR) and 193 (NLOS); LOS and GR paths are not marked as GBSM.
- `ssf_clusters` (R1): sizes written out, `n_fd` removed, links to `gbsm_draw`.

### 7.4 `api_matching.toml`

The families `gbsm_init_phases`, `gbsm_draw`, `gbsm_rays_from_paths`, `gbsm_single_link_channel` (file stems `api_mex/<stem>.cpp`, `api_python/qpy_<stem>.cpp`) have Python headings `gbsm.init_phases`, `gbsm.draw`, `gbsm.rays_from_paths`, `gbsm.single_link_channel`; rule 2 of `docs_builder/README.md` ("Cross-language buttons", `.` read as `_`) already matches them to the C++ names. P23 asks for entries: R10 adds `gbsm_init_phases = ["gbsm_init_phases"]` etc. only if `make docs` does not match a family by rule 2 (open point 11.6). The family `ssf_scatterers` (MATLAB, Python) calls the C++ member: R10 adds `ssf_scatterers = [".solve_scatterers"]` (member form, as `arrayant_rotate_pattern = [".rotate_pattern", ...]`).

### 7.5 Release-note items (no release-notes file exists in the repository; the manager carries these into the release text)

- Renamed and merged: `lsf_init_phases`, `ssf_init_phases` → `gbsm_init_phases(…, set_type, …)`; `lsf_draw`, `ssf_draw` → `gbsm_draw`; migration lines of 3.1 and 3.2; MATLAB `quadriga_lib.gbsm_init_phases` / `gbsm_draw`, Python `quadriga_lib.gbsm.init_phases` / `gbsm.draw`; "lsf" and "cluster" values unchanged; the silent hand-migration form `gbsm_draw(…, id, nullptr)` (run-time error).
- `SubpathMethod` removed from `conf_file_read`, `conf_file_write`, `scenpar_validate` and the wrappers; files with `SubpathMethod = legacy` still read, other values throw; the struct/dict field `SubpathMethod` = `legacy` is accepted and ignored, outputs no longer have it (Python `sp['SubpathMethod']` → `KeyError`); silent C++ forms: `conf_file_read(fn, sp, &s)` now fills `PL_model`, `nullptr`-padded calls shift by one, `conf_file_write(fn, sp, "legacy")` writes `PL_model = legacy`.
- `fc` → `center_frequency` (C++) in `los_probability`, `lsp_generate`, `o2i_loss`, `path_loss`; wrappers name it `center_freq`; Python keyword calls `fc=` break.
- `PerClusterDS_fc_min = 6` in the v19 UMa LOS and NLOS files (c_DS 3.91 ns below 6 GHz).
- New: "ray" random sets, `gbsm_rays` with the solver `solve_scatterers` (MATLAB/Python `ssf_scatterers`), `ssf_rays`, `ssf_paths`, `qrt_paths_from_scatterers`, `gbsm_rays_from_paths`, `gbsm_single_link_channel`; QRT interaction code 193.
- Zero-copy outputs in the GBSM wrappers (no value change).

## 8. Wrapper plan (P23, P24, P25)

### 8.1 MATLAB / Octave (`api_mex/<name>.cpp`, section "Geometry-based stochastic model")

| Function | Usage line (inputs min–max, outputs) | Index rules |
|---|---|---|
| `gbsm_init_phases` | `phase = quadriga_lib.gbsm_init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, dual_mobility, remove_bias, horizontal );` 2–10, 1 | `set_id` 0 = shared (default), k ≥ 1 = frequency k (C++ k − 1); ignored for "lsf" |
| `gbsm_draw` | `rnd = quadriga_lib.gbsm_draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, center_freq, dual_mobility, horizontal );` 5–11, 1 | `set_id` as above; `floor_id` 0-based floor index (as today) |
| `ssf_rays` | `rays = quadriga_lib.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, power, aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps_re, gr_eps_im, pol_variability, random_phase );` 14–22, 1 (1 × n_link struct array) | `i_freq` 1-based, default 1 |
| `ssf_scatterers` | `rays_out = quadriga_lib.ssf_scatterers( rays_in, d_min );` 1–2, 1 (one link: 1 × 1 struct) | — |
| `ssf_paths` | `[ path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind ] = quadriga_lib.ssf_paths( rays, link_gain_dB, mobility, tx_pos, rx_pos, kf_ratio, oxygen );` 2–7, 0–11 | `ray_ind` 1-based (+1); `path_type` code, not shifted |
| `gbsm_rays_from_paths` | `rays = quadriga_lib.gbsm_rays_from_paths( center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, path_length, M, method, path_type, tap, gr_eps_re, gr_eps_im );` 8–13, 1 | `path_type`, `tap` codes, not shifted |
| `gbsm_single_link_channel` | `[ chan, par ] = quadriga_lib.gbsm_single_link_channel( tx_array, rx_array, rays, link_gain_dB, mobility, output, i_freq, tx_pos, rx_pos, tx_orientation, rx_orientation, kf_ratio, oxygen, use_absolute_delays );` 4–14, 0–2 | `i_freq` 1-based, default 1; `par.tap` code, not shifted |

Defaults equal the C++ defaults; empty `[]` = omitted for every optional input. `chan` as `hdf5_read_channel` returns it (`qd_mex_channel2struct`), `par` the struct of the `par` entries (`qd_mex_any2matlab` per entry). The MEX tests check 1-based `i_freq` (0 throws `Input 'i_freq' cannot be 0 (1-based index).`), `ray_ind` +1, `set_id` 0 / k.

### 8.2 Python (`api_python/qpy_<stem>.cpp`, submodule `gbsm`)

| C++ | Python name and signature | Returns |
|---|---|---|
| `gbsm_init_phases` | `gbsm.init_phases(scenpar, set_type, seed=-1, set_id=-1, preset="Comb256", tx_pos=None, rx_pos=None, dual_mobility=False, remove_bias=True, horizontal=True)` | `phase` |
| `gbsm_draw` | `gbsm.draw(scenpar, set_type, phase, tx_pos, rx_pos, set_id=-1, preset="Comb256", floor_id=None, center_freq=None, dual_mobility=False, horizontal=True)` | `rnd` |
| `ssf_rays` | `gbsm.ssf_rays(scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, power, aod, eod, aoa, eoa, rnd_ray, method="38.901", i_freq=0, angle_model="additive", pol_model="38.901", gr_eps=None, pol_variability=False, random_phase=True)` | list of n_link dicts |
| `ssf_scatterers` | `gbsm.ssf_scatterers(rays, d_min=2.0)` | new dict (unchanged arrays shared, new `fbs_pos`, `lbs_pos`, `fallback`) |
| `ssf_paths` | `gbsm.ssf_paths(rays, link_gain_dB, mobility="auto", tx_pos=None, rx_pos=None, kf_ratio=None, oxygen=False)` | tuple `(path_gain, path_length, M, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind)` |
| `gbsm_rays_from_paths` | `gbsm.rays_from_paths(center_freq, tx_pos, rx_pos, fbs_pos, lbs_pos, path_gain, path_length, M, method="quadriga", path_type=None, tap=None, gr_eps=None)` | dict |
| `gbsm_single_link_channel` | `gbsm.single_link_channel(tx_array, rx_array, rays, link_gain_dB, mobility="auto", output="taps", i_freq=0, tx_pos=None, rx_pos=None, tx_orientation=None, rx_orientation=None, kf_ratio=None, oxygen=False, use_absolute_delays=False, stack=False)` | `(chan, par)`: `chan` as `channel.hdf5_read_channel` returns one channel (`qd_python_channel2dict`, `stack` as there), `par` dict |

Python indices stay 0-based (`set_id` −1 = shared). The permittivity is one complex array `gr_eps` = `gr_eps_re + 1j·gr_eps_im` (shape `(n_link, n_freq)` in `ssf_rays`, `(n_freq,)` in `rays_from_paths` and in the dict), as `lsp_generate` returns and `ssf_clusters` takes it; a real array is cast to complex. Doc headings use the exposed names (`# gbsm.init_phases`, linked `[[gbsm.init_phases]]`, as the `sos` wrappers do; the others keep their bare names `# ssf_rays` etc.).

### 8.3 Outputs, zero-copy and integer types (P25)

| Output | Size known before the call? | Form |
|---|---|---|
| `gbsm_init_phases` `phase` | yes: `n_coeff` from `sos_preset`, `n_gen` from `qd_gbsm_ray_layout` / 11 / 9·N / N after `scenpar_validate` | zero-copy |
| `gbsm_draw` `rnd` | yes: `[rows(rx_pos), cols(phase)]` | zero-copy |
| `ssf_rays` `rays` | yes: n_link entries, n_ray and n_gen from `qd_gbsm_ray_layout`, n_freq_out = 1 (TR) or n_freq | zero-copy init of the struct array / list of dicts (8.5); `gr_eps_*` pre-allocated only with G = 1; `fbs_pos`, `lbs_pos`, `fallback` left empty and not aliased; Python `gr_eps` copied (complex) |
| `ssf_scatterers` `rays_out` | yes | MEX: `mxDuplicateArray` of the input struct, then new `fbs_pos` `[3, n_ray]`, `lbs_pos`, `fallback` `[n_ray]` fields pre-allocated and aliased; Python: new dict sharing the unchanged arrays |
| `ssf_paths` (all) | no (n_path) | copied; MEX integer outputs uint64, Python int64 (`md/SPEC_PYTHON_API.md`, copied integer outputs) |
| `gbsm_rays_from_paths` `rays` | yes (n_path from the inputs) | zero-copy; `fallback` empty; Python `gr_eps` copied |
| `gbsm_single_link_channel` | no (by value) | copied (`chan`, `par`), documented |
| existing GBSM wrappers (`lsp_generate`, `path_loss`, `los_probability`, `o2i_loss`, `ssf_clusters`, `conf_file_*`, `scenpar_validate`) | where the size follows from the inputs and `scenpar` | zero-copy (R1, mechanical; values unchanged) |

Integer attributes of `gbsm_rays` (`path_type`, `tap`, `fallback`) are uint64 in MATLAB and Python (C++ `arma::uword`, zero-copy); inputs accept any real numeric or integer class (cast, values must be non-negative integers).

### 8.4 Scenario struct / dict rule (P24, R1)

`qd_mex_struct2scenpar` and `qd_python_dict2scenpar` lose the `subpath_method` output: a field / key `SubpathMethod` with the value `legacy` is accepted and ignored; any other value throws `Value '<value>' of 'SubpathMethod' is not supported; only 'legacy' is accepted.`; a non-string value throws `Field 'SubpathMethod' must be a string.` (MEX) / `Entry 'SubpathMethod' must be a str.` (Python). `qd_mex_scenpar2struct` and `qd_python_scenpar2dict` drop the field. The field-order sentences of `conf_file_read` and `scenpar_validate` (MEX, Python: "`SubpathMethod` (string) as field 107") change to the numeric keys only.

### 8.5 `gbsm_rays` ↔ struct / dict helpers (P25, R10)

| Helper | File | Role |
|---|---|---|
| `qd_mex_struct2gbsm_rays(const mxArray *s, arma::uword i = 0, bool copy = false)` | `src/mex_quadriga_lib_functions.hpp` | element i of a struct array → `gbsm_rays`, borrowing the MATLAB memory unless `copy` (inputs are read-only) |
| `qd_mex_gbsm_rays2struct(const std::vector<gbsm_rays> &rays)` | same | copy into a 1 × n struct array with all fields |
| `qd_mex_init_gbsm_rays(n_link, n_ray, n_freq, has_gr, std::vector<gbsm_rays> &rays)` | same | creates the 1 × n_link struct array with all fields, pre-allocates and aliases every attribute whose size is known (`tx_pos`, `rx_pos` `[3, 1]`, `center_freq`, `delay`, `power`, angles, `M`, `path_type`, `tap`, `gr_eps_re/_im` when `has_gr`), leaves `fbs_pos`, `lbs_pos`, `fallback` (and `gr_eps_*` without GR) empty; returns the `mxArray` |
| `qd_python_dict2gbsm_rays(const py::dict &d)` | `api_python/python_quadriga_adapter.hpp` | dict → `gbsm_rays` (borrowing where the dtype and layout allow; `gr_eps` complex → re/im copies) |
| `qd_python_gbsm_rays2dict(const gbsm_rays &r)` | same | copy into a dict with all keys |
| `qd_python_init_gbsm_rays(n_link, n_ray, n_freq, has_gr, std::vector<gbsm_rays> &rays)` | same | list of n_link dicts with pre-allocated, aliased numpy arrays as the MEX helper; `gr_eps` filled after the call from the C++ re/im |

- Fields / keys, always all present (empty where empty; MATLAB struct arrays need identical fields, measured in review round 2): `method` (char / str), `tx_pos`, `rx_pos`, `center_freq` (the attribute `center_frequency`, P23), `delay`, `power`, `aod`, `eod`, `aoa`, `eoa`, `M` (real, interleaved, `[8, n_ray, n_freq]` / `(8, n_ray, n_freq)`), `path_type`, `tap`, `gr_eps_re`, `gr_eps_im` (MATLAB) / `gr_eps` (Python, complex), `fbs_pos`, `lbs_pos`, `fallback`. Vectors are columns in MATLAB, 1-D arrays in Python.
- Input: missing `gr_eps*`, `fbs_pos`, `lbs_pos`, `fallback` (and any attribute appended later) mean empty; any other missing field throws `Field '<name>' is missing in 'rays'.`; unknown fields throw `Unknown field '<name>' in 'rays'.`; then `is_valid()` messages.
- No index attribute is shifted (codes only); `ray_ind` and `i_freq` are shifted in the MEX function wrappers (8.1).
- Implementations never reset, resize or re-create an aliased attribute (Armadillo throws for aliased memory of another size, measured in review round 2); C++ functions clear an attribute only when it is not empty (2.1).

## 9. Runs

Common to all runs: test-first per `md/SPEC_CPP_API.md` (stub, tests against the stub with the oracles evaluated, implementation, finish); the rules of 2.2; build in the run's own build folder (`cmake -G Ninja`, `--parallel <n>` by the load); targeted tests with `OMP_WAIT_POLICY=passive <build>/test_bin "<function>*"`; the full `make test` at the end (Catch2; MEX and Python suites where the run touches wrappers; MEX with g++-12); the hand-over report of `md/SPEC_CPP_API.md` ("HAND-OVER REPORT") plus: the list of files changed, every choice not fixed here, open points. No run edits this spec, the plan or `md/ssf_notes.md`; no run commits (the maintainer commits). `make docs` is run by the manager after each merge.

Order and parallelism: R1 → R2 → R3 → {R4 → R5} ∥ R6 ∥ R8 → R7 → R9 → R10. R6 and R8 run in separate worktrees in parallel with R4/R5 and with each other (R8's tests write fixed-name `.qrt` files); R7 needs R4/R5 (taps, method) and R6 (fallback, `b_med`); R9 needs R7 and R8.

### R1 — changes to existing functions (X1–X6)

- Goal: 3.3 and the "lsf"/"cluster" part of 3.1, 3.2 complete with all callers, wrappers, tests and docs; "ray" throws `Set type 'ray' is not available yet.` until R3 (temporary, replaced in R3).
- Files: `include/quadriga_gbsm.hpp` (merged declarations under `// ---- Random sets ----`; `subpath_method` removed; `fc` renamed; `ssf_clusters` comments; the "Large-scale fading" generator comment block replaced by a pointer to the set table); `src/qd_gbsm_phases.cpp` (merge, docs, header comment of the file); `src/qd_scenpar.cpp` (`SubpathMethod` rules, docs, section description: workflow random-set part, set table "lsf"/"cluster"); `src/lsp_generate.cpp`, `src/path_loss.cpp`, `src/los_probability.cpp`, `src/o2i_loss.cpp` (renames, docs, comments); `src/ssf_clusters.cpp` (docs, comments); `src/quadriga_gbsm_helpers.hpp` (`qd_gbsm_check_fc` message, helper list); `src/mex_quadriga_lib_functions.hpp`, `api_python/python_quadriga_adapter.hpp` (scenario struct/dict rule); `api_mex/`: delete `lsf_init_phases.cpp`, `lsf_draw.cpp`, `ssf_init_phases.cpp`, `ssf_draw.cpp`, create `gbsm_init_phases.cpp`, `gbsm_draw.cpp` (full argument lists of 8.1, "ray" passes through), edit the other GBSM wrappers (zero-copy, `center_freq`, struct rule); `api_python/`: the same with `qpy_*` files and `python_gbsm.cpp` (registration of `init_phases`, `draw`; `center_freq` keyword); `data/config/*.conf` (23 files: `SubpathMethod` line removed; 2 UMa files: `PerClusterDS_fc_min = 6`); tests: `git mv tests/catch2_tests/test_gbsm_phases.cpp tests/catch2_tests/test_gbsm_phases_lsf.cpp`, `git mv tests/catch2_tests/test_ssf_phases.cpp tests/catch2_tests/test_gbsm_phases_cluster.cpp`, `qd_test_ssf.hpp`, `test_ssf_clusters_quadriga.cpp`, the conf and scenpar tests, the renamed-argument tests; MEX tests `test_gbsm_init_phases.m`, `test_gbsm_draw.m` replace `test_lsf_*.m`, `test_ssf_init_phases.m`, `test_ssf_draw.m`, plus `test_conf_file_*.m`, `test_scenpar_validate.m`, `test_ssf_clusters.m`; Python tests likewise.
- Rename-to-find (plan 5.4): before changing behaviour, temporarily rename `conf_file_read`, `scenpar_validate`, `conf_file_write`, `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw` in the header (e.g. suffix `_x`), build library, tests, MEX and Python, and fix every call the compiler lists; then rename back. The compiler log goes into the hand-over. The tests at `test_conf_file_read.cpp:913, 926–938` must fail before their fix (proof that the silent shift is caught).
- Golden values first (6.4 R1), from the unchanged library.
- May not touch: anything for the "ray" set beyond the temporary message; `md/`; NG.
- Gates: library without warnings; `make test` (Catch2, MEX MATLAB and Octave, Python); `grep` finds no `lsf_init_phases`, `lsf_draw`, `ssf_init_phases`, `ssf_draw`, `subpath_method` in `include/`, `src/`, `api_*`, `tests/` (except release-note-like history in test comments, none expected), and no `SubpathMethod` in `data/config/`.
- Hands over: the merged functions, the golden values, the compiler log, the list of NG patches needed with the R1 library (plan 8 list; unchanged unless new call sites are found).

### R2 — preparatory change

- Goal: all new declarations (3.4–3.10 verbatim), the `gbsm_rays` members `n_ray`, `n_freq`, `empty`, `is_valid` complete (needed by R6 and R8 in parallel; `b_med`, `b_eff` stubs returning NaN), stub sources with doc stubs, the helpers of 3.11 complete, `qd_test_rays.hpp`, empty test files (6.4 R2).
- Files: `include/quadriga_gbsm.hpp`, `include/quadriga_channel.hpp` (declaration of `qrt_paths_from_scatterers` after `qrt_file_read_raw`), `include/quadriga_lib.hpp` (declaration of `gbsm_single_link_channel` next to `get_channels_ieee_indoor`); new `src/qd_gbsm_rays.cpp`, `src/ssf_rays.cpp`, `src/ssf_paths.cpp`, `src/gbsm_rays_from_paths.cpp`, `src/gbsm_single_link_channel.cpp`; stub of `qrt_paths_from_scatterers` at the end of `src/qd_channel_qrt.cpp`; `src/quadriga_gbsm_helpers.hpp`; tests of 6.4 R2.
- Gates: everything compiles without warnings (also the stubs); `test_gbsm_rays.cpp` passes; full `make test` (Catch2).
- Hands over: the public headers, which later runs do not edit (a run that finds a declaration problem stops and reports it).

### R3 — "ray" set type

- Goal: 3.1, 3.2, 4.1, 4.2, section 5 for "ray"; section description set and block tables (5.1, 5.2).
- Files: `src/qd_gbsm_phases.cpp`, `src/qd_scenpar.cpp` (section description), `test_gbsm_phases_ray.cpp`, MEX/Python `test_gbsm_draw.*` additions.
- Gates: `"gbsm_init_phases*"`, `"gbsm_draw*"` cases; golden "lsf"/"cluster" cases still pass; full `make test` incl. MEX/Python of `gbsm_draw`.

### R4 — `ssf_rays`, TR methods

- Goal: 3.5 for "38.901" and "38.901_MF" with `angle_model = "additive"` and `pol_model = "38.901"`; all validation; algorithms 4.3–4.7, 4.8 (TR model), 4.9; OpenMP limit measured. Until R5, method "quadriga", `angle_model = "rotation"`, `pol_model = "quadriga"` and `pol_variability` throw `Option '<name>' is not available yet.` (temporary, removed in R5).
- Files: `src/ssf_rays.cpp`, `test_ssf_rays.cpp`, appends to `qd_test_rays.hpp`.
- Inputs: R2 (declarations, `qd_test_rays.hpp`); R3 is not needed (the tests supply `rnd_ray` values; the swap tests with `gbsm_draw` sets are in R5).
- Gates: `"ssf_rays*"`; R0 cross-check (6.1) reported; full `make test`.

### R5 — `ssf_rays`, QuaDRiGa models and 7.6.16

- Goal: method "quadriga" (4.2, 4.5, P5 sharing), `angle_model = "rotation"`, `pol_model = "quadriga"` with the owner-end rule, `pol_variability` (4.8); swap tests of both polarisation models; the `ssf_rays` doc block final.
- Files: `src/ssf_rays.cpp`, `test_ssf_rays_quadriga.cpp`, `test_ssf_rays.cpp` (remove the "quadriga" skip), appends to `qd_test_rays.hpp`.
- Gates: `"ssf_rays*"`; R0 cross-check reported; full `make test`.

### R6 — `gbsm_rays::solve_scatterers`, `b_med`, `b_eff` (parallel worktree)

- Goal: 3.6, 4.10, 4.12 (members).
- Files: `src/qd_gbsm_rays.cpp` (`solve_scatterers`, `b_med`, `b_eff`, member doc blocks, class doc block final), `test_gbsm_rays_solve_scatterers.cpp`, cases appended to `test_gbsm_rays.cpp`. Not: `qd_test_rays.hpp`, `src/ssf_rays.cpp`.
- Inputs: `make_rays` of `qd_test_rays.hpp`.
- Gates: `"gbsm_rays*"`; R0 solver cross-check; full `make test`.

### R7 — `ssf_paths`

- Goal: 3.7, 4.11; doc block with the validity limits.
- Files: `src/ssf_paths.cpp`, `test_ssf_paths.cpp`, appends to `qd_test_rays.hpp`.
- Inputs: R4/R5 (`ssf_rays` for the diagnostic case), R6 (scatterers, fallback, `b_med`).
- Gates: `"ssf_paths*"`; full `make test`.

### R8 — `qrt_paths_from_scatterers`, `gbsm_rays_from_paths` (parallel worktree)

- Goal: 3.8, 3.9, 4.13; S4 docs (`path` doc in `src/qd_path.cpp`, `ray_state_update` table in `src/ray_mesh_interact.cpp`, `qrt_file_read` doc in `src/qd_channel_qrt.cpp`).
- Files: `src/qd_channel_qrt.cpp` (function and docs), `src/gbsm_rays_from_paths.cpp`, `src/qd_path.cpp` (doc only), `src/ray_mesh_interact.cpp` (doc table only), `test_qrt_paths_from_scatterers.cpp`, `test_gbsm_rays_from_paths.cpp`. Not: `qd_test_rays.hpp`, any other QRT behaviour.
- Gates: `"qrt_paths_from_scatterers*"`, `"gbsm_rays_from_paths*"`, the existing `"qrt*"` cases; full `make test`.

### R9 — `gbsm_single_link_channel`

- Goal: 3.10, 4.12 (report), 4.14; the QRT round-trip test (P12); OpenMP limit measured.
- Files: `src/gbsm_single_link_channel.cpp`, `test_gbsm_single_link_channel.cpp`, appends to `qd_test_rays.hpp`.
- Inputs: R7, R8.
- Gates: `"gbsm_single_link_channel*"`; full `make test`; one provisional timing of `phase` (1×1, 2×2 dual-pol, 8×8 dual-pol) and `drift` under `flock /tmp/bench.lock` with the load recorded, against the targets of 3.10.

### R10 — wrappers of the new functions and documentation

- Goal: MEX and Python wrappers of `ssf_rays`, `ssf_scatterers`, `ssf_paths`, `gbsm_rays_from_paths`, `gbsm_single_link_channel` (8.1, 8.2), the helpers of 8.5, their tests (6.4 R10); the section description final (7.2: first paragraph, workflow, Terms, codes, model properties); `api_matching.toml` check (7.4); a doc-size check of all new blocks against 7.1 (line counts in the hand-over); the release-note items of 7.5 checked against what was built.
- Files: `api_mex/ssf_rays.cpp`, `ssf_scatterers.cpp`, `ssf_paths.cpp`, `gbsm_rays_from_paths.cpp`, `gbsm_single_link_channel.cpp`; `api_python/qpy_*.cpp` of the same stems, `api_python/python_gbsm.cpp`; `src/mex_quadriga_lib_functions.hpp`, `api_python/python_quadriga_adapter.hpp`; `src/qd_scenpar.cpp` (section description); `docs_builder/content/api_matching.toml` if needed; MEX and Python tests.
- Gates: `make` (MEX MATLAB with g++-12 and Octave), `make python`, the MEX and Python suites, full `make test`; GLIBCXX check of the MEX files (`md/ssf_notes.md` practice: ≤ GLIBCXX_3.4.26).

## 10. Readiness checks (plan section 7)

### 10.1 Multi-frequency paths (`get_channels_multifreq`)

`quadriga` already gives one geometry for all frequencies (P5): `ssf_rays` with all frequencies fills `power [n_ray, n_freq]` and `M [8, n_ray, n_freq]` on shared coupling, ray delays, angles and phases, and `ssf_paths` (`drift`) returns exactly the `fbs_pos`, `lbs_pos`, `path_gain`, `path_length`, `M` that `get_channels_multifreq` and a multi-frequency QRT file (≤ 127 frequencies) take. A later "sample grid" mode (many close frequencies, dense near 60 GHz) is a call with a longer `center_frequency` list; λ_τ then depends on the grid (documented). TR methods do not qualify (7.6.5). No contract change.

### 10.2 Procedure A (TR 7.6.3.2)

Its sign X_n per cluster comes in its own later random set (P22), read through an appended optional input of a Procedure-A function. `ssf_rays` is a pure function of the cluster arrays and the "ray" set values, so updated clusters per step re-derive the rays with the drop's coupling and phases; the fixed ray layout per slot keeps the ray identity. One `gbsm_rays` per segment of ≤ 1 m and one `gbsm_single_link_channel` call each; the per-ray phase continuity across segments is the caller's (rotate the M columns of the new object by the accumulated displacement phase); the two strongest clusters and with them the tap indices may change between steps (documented then). No contract change.

### 10.3 TR 7.6.2.2 and the mobile-scatterer term of 7.6.10

7.6.2.2 for the TR methods enters as appended optional arguments of `ssf_rays` (bandwidth, array size, M_max) after `random_phase`; `NumSubPaths` becomes M_max (the "ray" layout scales); per-frequency ray counts by zero-power padding (dropped by `ssf_paths`); tap = cluster as in `quadriga`; the ray delays use block 14 of the per-frequency sets (generated now at `SC_lambda`); random offsets and unequal ray powers come in new sets through appended optional inputs (P22); `gbsm_rays.power` is already per ray. Cost, not contract: M = 1000, N = 19 gives ≈ 2.7·10⁵ generators per set (plan 7.3). 7.6.10 (7.6-46): appended `time` arguments of `ssf_paths` (one value) and `gbsm_single_link_channel` (`[n_snap]`), an appended per-ray attribute of `gbsm_rays` (the helpers treat a missing appended field as empty, 8.5), α and D in a new set through an appended optional input of `ssf_rays`.

### 10.4 Library-only TR 7.8.2 calibration

Many sites and sectors per batch are supported (per-link `tx_pos`); co-sited sectors share the antenna-free rays (TR: steps 1–9 identical) and differ only in the `gbsm_single_link_channel` call; per-site seeds (series 1); LOS and NLOS links by separate calls per scenario (series-1 W1) joined by the caller; TR antennas from `arrayant_generate`; one snapshot per drop with library defaults = TR. Nothing in this series requires QuaDRiGa-NG.

## 11. Points for the spec review (listed, not decided)

1. LOS detection tolerance of `gbsm_rays_from_paths` (`path_type = nullptr`) after float32 storage in QRT: the spec uses 0.1 mm for dist(FBS, LBS) and path length − d3D (3.9). Paths without interactions come back exactly; other paths near the direct line could be misread at large coordinates (F2: 1.0 mm at 20 km). Alternatives: a relative tolerance (e.g. 4·2⁻²³ of the largest coordinate), or detection by `no_int` only.
2. One name per polarisation model: `pol_model` takes "38.901" and "quadriga", the same strings as `method`, so `method = "38.901", pol_model = "quadriga"` reads oddly; a distinct name (e.g. "rotation") would need a plan change.
3. TR ambiguity A-3: the literal 7.6-1 omits Δτ of the absolute TOA (7.6.9) although the cluster delays contain it; the spec takes each path's own length (P1), measured −0.13 … +1.05 dB against the literal formula.
4. NG change-list extras (plan 8.7; review round 2, B2-8): removal of NG's power sort (`gen_ssf_parameters`), NG's Laplacian CDL weights as per-ray powers of a directly filled `gbsm_rays`, orientations per snapshot and dual mobility in NG's 3GPP baseline.
5. GR Jones in `drift` (4.11): the spec rule M(t) = diag(R∥(t), −R⊥(t))·‖M_0‖_F/(√2·R_0) with the power factor g(t)/g_0 in the gain; it equals the TR form for `ssf_rays` data and stays exact at the ray-data positions for any scalar normalisation of M_0 (e.g. QRT data read with `normalize_M = 1`); the plan says only "GR Jones R(t)/R0".
6. `api_matching.toml` (7.4): rule 2 of the docs builder already matches the four renamed Python families; P23 asks for entries.
7. Owner-end rule when all three coordinates of the two ends differ by ≤ 1 mm (co-located ends): the spec makes the Tx the owner (4.8).
8. `gbsm_rays_from_paths` with `tap = nullptr` gives the LOS path its path index as tap, so `taps` does not list LOS first unless the LOS path is path 0 (3.9); the 5.5 convention (LOS tap 0, GR tap 3) is not enforced by `is_valid`.
9. `kf_ratio` uses the K-factor implied by the coefficient powers (S_L/S_N, 4.11); for objects whose LOS + NLOS powers do not sum to the TR split (data from paths, other sources) the rescale still keeps LOS + NLOS and is the only definition the spec found without a K input.
10. Run split: the `gbsm_rays` members `n_ray`, `n_freq`, `empty`, `is_valid` move from R4 (plan 8.4) to R2 and `b_med`/`b_eff` to R6, because R6 and R8 run in parallel with R4 and need them; `pol_variability` and `angle_model = "rotation"` are placed in R5 with the QuaDRiGa models.
11. The Python `stack` flag of `gbsm.single_link_channel` (8.2), taken from `channel.hdf5_read_channel`, is not in the C++ contract.
12. The GR tolerance of `gbsm_rays_from_paths` (abs(z) ≤ 1 mm) and the `phase` + `use_absolute_delays` rule (LOS delay of the ray-data positions) are spec choices.
13. Done in spec v1 (follows from S1): the header comments of `ssf_paths` and `gbsm_single_link_channel` name the `phase` limit sqrt(λ·b_eff/8) per moving end instead of the plan's fixed NLOS factors (0.22–0.28× TR, 0.49–0.67× `quadriga` of sqrt(λ·d3D/8)), which do not carry to RMa NLOS (0.12–0.18) or InH NLOS (0.42–0.46); the doc blocks give the measured ranges in metres. Reviewers check the wording only.
