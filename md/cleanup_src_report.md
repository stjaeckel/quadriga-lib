# Report — cleanup of `src/ray_mesh_interact.cpp` (task 1 of 3)

Brief: `md/cleanup_src_brief.md`. One section per phase. Scratch is `build_linux/scratch/cl0` (not kept; every
number this report relies on is stated here).

## Phase 0 — inventory and plan (2026-10-20, no source edit)

### 0.1 Baseline

- `git status`: clean apart from `md/acoustic_manager.md` (modified) and `md/cleanup_src_brief.md` (new); nothing
  under `src/` or `tests/` modified. Last commit 6307a64. `md5sum src/ray_mesh_interact.cpp` =
  `000bcdea91c31b555b463fa96ee8dc7a` (5211 lines, 324992 bytes).
- `cmake --build build_linux --parallel 12`: up to date. The three suites with the 12-thread commands of `md/wave.md` 0:

  | suite | result | wall / user time (load 5–7) |
  | --- | --- | --- |
  | `[wave]` | All tests passed (165594 assertions in 181 test cases) | 25.5 s / 29.4 s |
  | `[acoustic]` | All tests passed (42391 assertions in 60 test cases) | 0.9 s / 1.3 s |
  | whole suite | All tests passed (1317534 assertions in 1148 test cases) | 38.7 s / 64.7 s |

  The counts are the brief's. `tests/data/wave_twins.tsv` is asserted inside the suite (1e-12 relative) and passed.
  Printed output: 13969 / 9014 / 23832 lines (the brief's "about 23,800").
- Kept for the later steps (scratch): `test_bin_base` and `libquadriga_base.a` (today's library, for the CPU-time
  comparison of phase 2), `ray_mesh_interact_base.cpp`, and the fresh runs `r1_wave.txt`, `r1_acoustic.txt`,
  `r1_all.txt`. The untracked baselines `test_wave.txt`, `test_acoustic.txt`, `test.txt` in the repository root
  (10:34 today, after 6307a64) were left in place.

### 0.2 The comparison tool

`rowdiff.py` (scratch, 190 lines, Python standard library, run with `python3 -I`):

    python3 -I rowdiff.py BASE NEW [--src tests/catch2_tests] [--max N]

- Aligns the two outputs on whole lines (`difflib`, so a dropped or added row stays local); a changed block that keeps
  its length and the text between the numbers of every row is compared number by number, anything else is reported
  as a structural difference.
- A number passes when it is equal to the baseline or within one unit of the baseline token's last printed digit
  (`0.1234` → 1e-4, `1.234e-05` → 1e-8, `17` → 1); `-0.0000` against `0.0000` passes. NaN against anything fails.
- The summary lines (`Filters:`, `Randomness seeded to:`, `All tests passed (…)`, the `====` rule) are set aside;
  a change in the counts is reported.
- Every differing value is printed with its line in both files, the column, the unit, the row, the table header above
  it (the nearest line starting with `#`), and the case that prints that header, found by searching the test sources
  for the longest literal piece of the header (`TEST_CASE` name and `file:line`). Exit status 0 only when every
  number is within the rule and nothing is structural.

Checks:
- Two runs of today's library (the 10:34 baselines against the fresh runs): `[wave]` 140986 numbers identical,
  `[acoustic]` 112537, whole suite 256712; 0 within, 0 beyond, 0 structural, exit 0 for all three (only the random
  seed line differs, and it is set aside).
- A synthetic copy of the `[wave]` run with one value moved by one digit, one by two digits, one 3-digit exponent
  value by one, a column appended to one row and one row deleted: reported as 2 within, 1 beyond (with case
  *Diagnostic - Interface energy split per polarization*, `test_wave_mesh_interact.cpp:1742`), 2 structural blocks
  (the edited row and the deleted row, each at its own place), exit 1.

Decided: tracked as `tools/acoustic/rowdiff.py` (0.6 item 3; tasks 2 and 3 need it).

### 0.3 Inventory

#### Sections today

| lines | content |
| --- | --- |
| 1–20 | licence, includes, `#include "bits.hpp"`, `#define colocation_dist 0.001` |
| 23–1527 | anonymous namespace: `qd_evanescent`, `MaterialCols`, `Material` (columns, laws, face models, `interact_with`), geometry helpers, solver |
| 1529–1696 | `/*!SECTION`, `medium_gain`, `interface_gain`, `refractive_index` (API block, body, instantiations each) |
| 1698–2884 | `ray_mesh_interact` API block (1698–1859), body (1861–2856), instantiations |
| 2886–5211 | `ray_state_update` API block (2886–3136), body (3138–5168), instantiations |

Line kinds (whole file): 2585 code, 468 code with a trailing comment, 1227 comment-only, 527 in `/*!MD` blocks,
405 blank. Comment-only lines by region: `Material` 224, `MaterialCols` 7, `qd_stack_resolve` 73,
`qd_slab_resolve` 20, `ray_mesh_interact` 173, `ray_state_update` 640, the rest 90.

#### Functions, structs and lambdas

Callers counted in this file (other files call only the five public functions).

| name | lines | callers | what it does |
| --- | --- | --- | --- |
| `qd_evanescent` | 36–39 | 6 sites | an argument of a refraction root that is negative real up to rounding (forces the decaying root) |
| `MaterialCols<dtype>` | 42–179 | 4 public functions | pointers to the material columns; ctor (74–162) checks equal lengths and signs; `is_sheet` (165–178) `thk > 0` |
| `Material` | 182–827 | everywhere | one material's columns in double, defaults = air |
| `Material::Material(cols, idx)` | 214–251 | `MAT` and 12 sites | reads row `idx` (1-based, 0 = air) |
| `eta`, `mu` | 256–266, 269–275 | 16 sites each | relative permittivity (with the Lorentz pole for wave media) and permeability at f |
| `medium_gain` | 278–339 | public `medium_gain`, `qd_mass_amp` ×2, `qd_slab_resolve` ×2, `SCALE_BY_MEDIUM` | in-medium power gain: ITU loss over the refracted path, `alpha`, and the mass-law term with bonded bodies |
| `mass_X0` | 355–368 | `medium_gain`, `qd_mass_amp`, `MASS_RUN` ×3, `SHEET_EVENT` | impedance ratio X0 of a mass-law body (with `res*`, `coi*`) |
| `mass_TL` (static) | 381–395 | `medium_gain` ×2, `qd_stack_resolve`, `SCALE_BY_MEDIUM` | mass term in dB from X0, infinite or finite (Davy) panel |
| `davy_sigma` (static) | 404–411 | `mass_TL` | Davy's forced radiation efficiency of a square panel |
| `excess_gain` | 417–421 | `qd_slab_resolve`, `SCALE_BY_MEDIUM` | gain of the `alpha` column alone |
| `interface_gain` | 424–430 | 11 sites, public `interface_gain` | lumped `att` gain |
| `tf` | 438–447 | `apply_tf_pair`, `tf_face`, `BND` | transmission factor at f (power law + peak), clamped |
| `apply_tf_pair` | 453–462 | `tf_face`, `closed_face` | the tf pair rule on a reflectance |
| `peaked`, `zeta` | 469, 470–475 | `tf_face`, `closed_face`, `TF_FACE` | a peaked mass-law face and its series impedance |
| `tf_face` | 489–505 | `interact_with` ×2, `TF_FACE`, `fresnel_r` ×2 | tf at an open face (transformer), or a peaked face's impedance |
| `closed_face` | 525–537 | `interact_with` ×3, `BND` ×3, `fresnel_r` ×2, `tf_face` | tf at a closed face (the pad) |
| `same_as` | 540–568 | 8 sites | column-wise equality |
| `flux_amp` (static) | 603–616 | `interact_with`, `qd_stack_resolve` | per-channel normal-flux amplitude factor |
| `interact_with` | 636–826 | 15 sites | the two-medium interface: Fresnel TE/TM (or scalar), TIR, tf, held wave in a plate, `att`; 13 parameters, 7 of them out-pointers |
| `qd_reflect` | 830–838 | 4 sites | mirror direction |
| `qd_relaunch_offset` | 842–846 | 3 sites | 8 float ulp at the hit point's magnitude |
| `qd_refract` | 849–860 | 7 sites | Snell direction from a ratio and a complex far cosine |
| `qd_polbasis` | 864–931 | 5 sites | xprmat and gain from TE/TM coefficients and the in/out bases |
| `QdLayer` | 934–938 | solver, `fresnel_r`, both sheet composites | a layer: material and normal thickness |
| `qd_mass_amp` | 945–967 | `qd_stack_resolve` ×3 | amplitude factor of a mass-law layer (capped path, mass term, X0 run) |
| `qd_plate` | 972–975 | 4 sites | plate rule a τ / (a + τ − a τ) |
| `qd_stack_resolve` | 1016–1395 | `fresnel_r`, `SHEET_COMPOSITE`, `SHEET_RESOLVE` | T and R of a sheet or a two-layer stack between half-spaces; 17 parameters |
| ↳ `MED`, `W`, `ADM`, `COS`, `root` | 1041–1108 | inside | medium j, ε μ, admittance, cosine from the invariant, decaying root |
| ↳ held-plate branch | 1054–1097 | inside | a run of mass-law layers met by a held wave |
| ↳ `CLOSED` | 1116–1117 | `BND` | a mass-law layer the wave is evanescent in |
| ↳ leak chain | 1125–1144 | inside | a mass-law sheet on a closed far side |
| ↳ `PLATE_OUT`, `PLATE_FIX`, `TF_FACE` | 1154–1175 | `BND` | plate rule at the last face of a plate; the face's transformer |
| ↳ `BND` | 1176–1256 | the recursion ×2 | boundary j \| j+1 with tf, closed layers, plates (81 lines) |
| ↳ `PHI` | 1261–1268 | the recursion | one-way factor of a coherent layer |
| ↳ recursion, flux, `att`, `UNDISP` | 1270–1394 | inside | far-end recursion per channel, flux conversion, gains, displaced host |
| `qd_slab_resolve` | 1410–1526 | `SLAB_AIRY_FACTOR` | series factor S of a traced solid, with the `eps` gate; 18 parameters |
| ↳ `fresnel_r` | 1449–1490 | inside ×2 | reflection at slab \| neighbour, a sheet neighbour through `qd_stack_resolve` |
| `quadriga_lib::medium_gain` | 1583–1592 | `ray_commit.cpp` ×2 | public: gain over `dist` |
| `quadriga_lib::interface_gain` | 1635–1644 | none in `src/` (API, tests) | public: `att` gain |
| `quadriga_lib::refractive_index` | 1682–1693 | `calc_diffraction_gain.cpp` | public: Re √(ε μ) |
| `quadriga_lib::ray_mesh_interact` | 1862–2856 | `ray_progress.cpp` ×3, `calc_diffraction_gain.cpp`, `point_inside_mesh.cpp` | public: one face event per ray (995 lines) |
| ↳ checks, compaction map, output sizing | 1893–2086 | inside | |
| ↳ `NORMALIZE`, `LENGTH`, `SET1`, `SET3`, `SETL` | 2097–2136 | inside | normalize (2e-7, fallback x), length, output writers |
| ↳ no-hit branch | 2157–2260 | inside | transparent pass with the tube carried |
| ↳ FBS / SBS geometry | 2262–2349 | inside | hit points, normals, angles |
| ↳ classification | 2351–2474 | inside | `out_flags`, `sheet_case` 0–6, `front_eff` |
| ↳ media | 2479–2528 | inside | M1 / M2 / sheet / second sheet |
| ↳ `SHEET_COMPOSITE` | 2551–2613 | 1 site | the sheet composite with air assumed (63 lines, 35 of them comment) |
| ↳ tube incidence, whole-tube TIR, spine direction, tube update, edge length | 2617–2791 | inside | |
| ↳ writes | 2793–2854 | inside | |
| `quadriga_lib::ray_state_update` | 3139–5168 | `ray_progress.cpp` ×3, `calc_diffraction_gain.cpp` | public: per-ray medium state machine (2030 lines) |
| ↳ checks and output sizing | 3176–3343 | inside | `check_state_words` 3237–3244 |
| ↳ word decode | 3356–3434 | main | `mtl_current`, `resolved`, `bent`, `prev_nonpar`, `prev_embedded`, `mass_crossed`, `mass_code`, `followed`, `stack_behind` |
| `MAT` | 3450–3451 | 55 sites in 14 lambdas | `Material(cols, idx)` |
| `SAME` | 3455–3456 | main ×6, `REPLACE_BY_VBS`, `APPLY_TRANSITION` | same medium (index or columns) |
| `TRANSPARENT_TO` | 3463–3470 | `SHEET_EVENT` | equal ε and μ within 1e-9 |
| `IS_SHEET` | 3473–3474 | 7 sites | `thk > 0` for an index |
| sheet case | 3487–3523 | main | `sheet_case` 0–4 from the hit code and both faces |
| `DIST`, `NORMALIZE` | 3526–3541 | 11 / 9 sites | distance; normalize (1e-12, fallback x) |
| geometry setup | 3547–3684 | main | N, O, D, F, S, `dir_prev`, `dir_next`, VBS, `use_vbs`, `refracted`, `bent_enter` |
| `VBS_INTERSECT` | 3602–3625 | 5 sites | intersection of a direction with the face plane |
| `DIST_ADD`, `DIST_SET` | 3692–3706 | 7 / 13 sites | the ledger |
| `fbs_sbs_not_parallel` | 3710–3725 | main | wedge test (dot product of the two normals below 1 − 3.8e-3) |
| `KILL_RAY`, `SCALE_RAY`, `REPLACE_BY_GAIN` | 3728–3783 | 7 / 3 / 3 sites | output writers |
| `EXACT_IN_SLAB_COSINE_SOLVE` | 3810–3907 | memo, `RELAUNCH` | recovers the complex in-medium cosine from the traced one (Newton) |
| `EXACT_IN_SLAB_COSINE` | 3916–3921 | 5 sites | the solve, memoized on `mtl_current` |
| `RELAUNCH` | 3948–3999 | 3 sites | re-aims origN/destN (and tridirN) along a recovered direction |
| `REPLACE_BY_VBS` | 4005–4191 | `APPLY_TRANSITION`, refl. pass | the face at the VBS angle: coefficients, TIR, direction (187 lines) |
| ↳ `EXIT_DIRECTION`, `LEAK_DIRECTION` | 4023–4040, 4091–4102 | inside | directions out of a medium, and of a leak |
| `MASS_CODE`, `MASS_RUN`, `MASS_FOLD` | 4196–4265 | 2 / 3 / 2 sites | thickness code; X0 of the bonded bodies; their thickness in one material |
| `PLATE_ABS`, `PLATE_IN` | 4271–4283 | 3 / 2 sites | face absorption of a plate; the entry face's |
| `SCALE_BY_MEDIUM` | 4287–4399 | 3 sites | the leg's medium gain and phase (exact form, hybrid form, plate rule) |
| `GET_EXIT_MATERIAL` | 4402–4414 | 3 sites | the medium by its exit face, fix flag on mismatch |
| `SLAB_AIRY_FACTOR` | 4426–4480 | 3 sites | series factor of the current leg via `qd_slab_resolve` |
| `SHEET_RESOLVE` | 4501–4677 | `SHEET_EVENT` | sheet composite against the tracked media (177 lines) |
| `SHEET_EVENT` | 4694–4887 | both passes | a sheet event: host leg, composite, words (194 lines) |
| `APPLY_TRANSITION` | 4890–4962 | 11 sites | a bare-face transition or exit |
| dispatch | 4964–5119 | main | kill rule, reflection pass, transmission pass by hit code |
| ledger offset | 5121–5150 | main | the relaunch offset in the ledger (`LEG_FROM_HIT_POINT`) |
| writes | 5152–5166 | main | state words, ledger, codes, verdict |

#### Comments by kind

Every comment-only block (260 blocks, 1227 lines) was read and classed; trailing comments (468 code lines) are
counted apart.

| kind | blocks | lines | left after the rewrite (estimate) |
| --- | --- | --- | --- |
| needed (why the code does what it cannot say; conventions; formulas) | 151 | 751 | ≈ 530 (shortened: many restate the same point twice) |
| mixed: a needed core in stale padding | 25 | 340 | ≈ 150 |
| stale (history, superseded numbers, decision reasoning, wrong) | 8 | 54 | ≈ 5 |
| redundant (restates the next line) | 76 | 82 | 0 |

So about 240 stale lines (54 in pure blocks, ≈ 190 inside the mixed ones), 82 redundant, and ≈ 220 lines of needed
text that say it twice. The stale ones, by kind, with examples:

- History ("used to", "before", "no longer"): the triangle edge that "used to be measured" (2286–2289); the grazing
  clamp that was removed (2553–2577, 25 lines, and 4608–4615); "Total reflection of the outer pair no longer discards
  the sheet. It used to …" (4585–4591); "this was written in before" (medium_gain, 305–309); "no longer collapses"
  (4777–4786); "as it always did" (35); "which is why `qd_stack_resolve` sits at file scope instead of being a lambda"
  (4489–4491).
- Measured numbers of past rounds: "out of eps 1.50 a ray that came in at 40 deg left at 51.9 deg" (4012); "the
  2.5e-5 dB figure … 0.040 dB" (4318–4321); "17.7 deg at 85 deg" (4562); "|S| reads 1.000000 in every such row, worth
  up to 1 dB" (4729); "a porous absorber … turned into a mirror from 37 deg" (706); "5e-4 relative at 25 dB" (318);
  "6e-14 relative" (683, the reason is needed, the number may stay).
- Decision reasoning: "Real-index Snell is NOT that map … the obvious thing to try" (3802–3806); "This stays a special
  case rather than the general fix" (4735–4738); "the one option that had to be rejected" (4611).
- Wrong against the code: "In-medium magnitude/phase stay deferred to close_med" (4003; no `close_med`); "Used by SAME,
  TRN and the slab-factor calls" (3449; no `TRN`); "type-12 ia a o-i-o … bast the SBS" (3543–3546; no type 12, it is
  23); "two Newton steps" (3805, the loop runs to 50 and stops at rounding, its own comment says three);
  "because the two public callers write to different buffers" (4417–4418; one caller); "No-op (false) when normals are
  absent" (3708; `normal_vecN` is required); "spec 3.1" (3628); `Material() = default; // All pointers stay
  nullptr` (211; `Material` has no pointers); "(A21, md/wave.md 8)" (4966, an internal reference); both `ISSUE`
  comments (1578–1580 names `calc_diffraction_gain` as the caller of `medium_gain`, which is now `ray_commit.cpp`;
  1890–1891 is what the API block already states as a rule).
- Duplicates: the same sentence twice in `BND` (1162–1169 and 1247–1251); the sheet "both ports come out of one pass"
  note twice (2606, 4662); the relaunch offset described at 840 and again at 2801.

Trailing comments: most annotate parameters and are kept; about a third restate the line (`// Ray x-index`,
`// Set OK flag`, `// Clear from hit type`) and go.

#### The public API blocks

527 lines. What each says, what can go, and what is stale:

- **`medium_gain`** (1533–1576, 44 lines). Says: the three loss terms; the mass term with its angle shape and Davy's
  panel at length; `att` unused. Internal, can go: the angle shape of the mass term and Davy's details (this entry
  point has cos θ = 1; they belong to `ray_state_update`'s description at most). Missing: the `res*` and `coi*`
  columns act in the mass term, and the permittivity pole in the conductivity loss of a wave medium.
- **`interface_gain`** (1597–1632, 36 lines). Stale: "A `coiA` column of an older table is read and ignored" — it is
  not read; any unknown column is ignored. Internal: the list of columns not used.
- **`refractive_index`** (1649–1679, 31 lines). Stale: "A permittivity resonance (`resF`, `resQ`, `resS`) is part of
  `ε_r`" — not for a material with `m > 0` (`Material::eta` drops the pole there).
- **`ray_mesh_interact`** (1698–1859, 162 lines). Needed: face sides, compaction, sheets (shortened), scalar types,
  inputs, outputs, the `out_typeN` bit and code tables. Internal, can go: why two coincident sheets collapse (the
  exporter story), "Codes 69 and 71 are shared by three cases, deliberately …" (to one sentence), the paragraph on
  provisional sheet media (to one sentence). Stale: `fbsN`, `sbsN` are `[n_rayN, 3]`, not `[n_ray, 3]`; "unit
  reflection" under TIR is the case without `tf` only (with `tf` a closed face passes the pad's leak); "Overlapping mesh
  geometry must be avoided (materials are transparent to radio waves)" reads oddly. Missing: a tube with a vertex past
  the critical angle passes undeviated as a whole (types 2, 5).
- **`ray_state_update`** (2886–3136, 251 lines). Needed: what it corrects, the two passes, inputs, the state-word
  table, outputs, the resolved-type tables. Internal, can go: most of the sheet paragraph (2902–2916), the reasoning
  why `mtl_ind_prev_outN` may hold a sheet, the `69/71 also occur as out_typeN values` note. Stale:
  - `normal_vecN`: "NULL disables the wedge test" — NULL throws; the input is required.
  - `eps`: "A near-pole `S` … re-emits regardless of `eps`" and `verdict_inN`: "A summed series near a pole is clamped
    in magnitude" — neither exists in the code (the only gates are `ρ < eps`, the known non-parallel slab, and an exact
    zero denominator in the stack solver).
  - `eps`: "a round trip at or above one is always resolved" — true for `eps ≤ 1` only; with `eps = 1.5` a series with
    `ρ = 1.2` is re-emitted (the gate is `ρ < eps` alone).
  - "A thin-sheet stack of two different materials … is **not implemented** and returns 0" — it is implemented in
    both functions (`sheet_case` 4, two layers in `qd_stack_resolve`).
  - "With `S` suppressed … reproduces [[calc_diffraction_gain]]" and "See also: [[calc_diffraction_gain]] (the
    reference state machine this function ports)" — `calc_diffraction_gain` now calls `ray_state_update`.
  - `resolved_typeN` bit 4 "(resolved-false-outside, or …)" — it is set on an exit-material mismatch and on a sheet
    event with a deferred buffer only.
  - "Passing all six state args NULL disables tracking — each interaction is corrected on its own (entry loss, TR kill,
    single-hit air-gap `S`)" — the parenthesis does not match the code; without words every ray starts in air.

All five blocks extract with `tools/extract_md.py` today (`-a cpp` into scratch: one entry each, 6374 lines in all);
phase 1 repeats that check.

#### Macros, globals, scoping

- `#define colocation_dist 0.001` (used 9 times) → `constexpr double colocation_dist` in the anonymous namespace.
- Literals repeated without a name: `1.570796326794897` ×7 (note: not the double nearest π/2; it differs by 2 ulp and
  must stay as written), `2.0 * 3.14159265358979323846 * fGHz * 1e9 / 299792458.0` ×4 (k0), `1.1920929e-7` ×3,
  `0.0015` (mass floor) ×4, `17.98`, `8.686 … / 0.0477135`, `20.958450219516818`, the state-word masks `0x3FFF`,
  `0x4000`, `0x8000`, `0x7FFF` (≈ 40 uses), the thickness code `10^(-4 + 6 (k − 1) / 32766)` written twice.
- Everything internal is already in the anonymous namespace; no globals.
- `const`: most locals of the two large functions that do not change are not `const` (`Ox…`, `iFBS`, `fGHz`, …).
- Narrowest scope: `fGHz` is recomputed per ray in `ray_state_update` (a call constant); `material_to_material`,
  `cTE`, `cTM`, `eta` and the tube arrays of `ray_mesh_interact` are declared well before use; `thetaS` holds a cosine
  and then an angle.
- `-Wshadow` (one extra compile of this file): 9 warnings, locals `a`, `b`, `c`, `e` shadowing the `Material`
  columns (472, 528, 529, 729, 791) and `VBS_INTERSECT`'s parameters `O`, `F`, `N` shadowing the ray's. Nothing else
  with `-Wall -Wextra -Wpedantic -Wunused -Wunused-macros`.
- Static members of `Material` that use no material (`mass_TL`, `davy_sigma`, `flux_amp`) and two-medium functions
  written as members of one side (`apply_tf_pair`, `tf_face`, `closed_face`, `interact_with`).
- Out-pointer parameter lists: `interact_with` 13 parameters (7 out), `qd_stack_resolve` 17 (4 out, 8 defaulted),
  `qd_slab_resolve` 18 (2 out, 6 defaulted).
- Repeated blocks: the xprmat/gain write after `qd_polbasis` ×4, the `path_dirN` write ×5, the spherical→Cartesian
  vertex decode ×2, the decaying-root pattern ×6.
- Typos: "NUmber" ×2, "ia", "bast", "an type".

#### Rename table (decided; names revised by Stephan, 2026-10-20)

New file-scope helpers take no `qd_` prefix (`k0()`, `decaying_root()`, `dist3()`, …); the existing `qd_` names
stay (those that tests or tools mention, and `qd_reflect`, `qd_mass_amp`, `qd_evanescent` beside them). Names that tests
or tools mention in comments are kept unless misleading (`qd_refract`, `qd_polbasis`,
`qd_relaunch_offset`, `qd_slab_resolve`, `qd_stack_resolve`, `interact_with`, `tf_face`, `closed_face`, `flux_amp`,
`apply_tf_pair`, `medium_gain`, `mass_X0`, `mass_TL`, `davy_sigma`, `interface_gain`, `tf`, `qd_plate`,
`MaterialCols`, `Material`). The `Material` column members keep the table's column names (`a` … `h`, `thk`, `att`,
…); `tfR` stays (the column is `tf`, the function `tf()`).

| old | new | why |
| --- | --- | --- |
| `#define colocation_dist` | `constexpr double colocation_dist` | scoped constant, same name |
| `1.570796326794897` | `half_pi`, the literal as written, with a one-line comment that the value is deliberate | one named value; it is 2 ulp above the double nearest π/2 and stays |
| k0 expression | `k0(fGHz)` | one expression, same evaluation order; a new helper, so no `qd_` prefix |
| `1.1920929e-7`, `0.0015` | `float_ulp`, `mass_min_path` | named |
| `0x3FFF`, `0x7FFF` | `word_mtl`, `word_code` | the material index (bits 0–13) and the thickness code (bits 0–14) |
| current word `0x8000`, `0x4000` | `cur_summed`, `cur_bent` | one name per bit and word |
| previous word `0x8000`, `0x4000` | `prev_nonparallel`, `prev_embedded` | one name per bit and word (bit 14 also reads as `prev_real_inv` for a solid; the comment says so) |
| buffer word `0x8000`, `0x4000` | `buf_mass_code`, `buf_followed` | one name per bit and word (bit 14 with an index is the stack layer behind; the comment says so) |
| `sheet_case` 0–6 / 0–4 (two numberings) | `enum class SheetCase { none, isolated, glued_entry, glued_exit, stack, duplicate, corner }` | one numbering in both functions |
| rmi `iMF`, `iMS` | `mtl_fbs`, `mtl_sbs` | what they are |
| rmi `M1`, `M2`, `M_slab`, `M_slab2` | `mat_near`, `mat_far`, `mat_sheet`, `mat_sheet2` | near/far is what the code means |
| rmi `eta` (a double) | `snell` | it is the Snell ratio, not a permittivity |
| `eta1_div_eta2`, `eta1_d_eta2` | `n2_ratio` | it is (ε₁μ₁)/(ε₂μ₂) |
| rmi `thetaF`, `thetaS` | `theta_fbs`, `theta_sbs` (+ `cos_sbs`) | one meaning per variable |
| rmi `tir_central`, `out_flags` | `tir_spine`, `hit_code` | |
| rsu `typeH`, `typeR`, `nH` | `hit_code`, `res_code`, `n_hits` | what they are |
| rsu `M1`, `M2` (indices) | `mtl_fbs`, `mtl_sbs` | |
| rsu `resolved` | `leg_summed` | the current leg's series was summed |
| rsu `dist_orig_fbs`, `dist_fbs_dest`, `dist_orig_vbs` | `len_to_face`, `len_face_to_dest`, `len_to_vbs` | |
| rsu `ii` | (dropped, `i_rayN` as `size_t`) | one compact index |
| `MAT`, `SAME`, `TRANSPARENT_TO`, `IS_SHEET` | `material()`, `same_medium()`, `same_eps_mu()`, `is_sheet_mtl()` | functions |
| `DIST`, `NORMALIZE` (×2) | `dist3()`, `normalize_fast()` (multiply by the reciprocal, above 2e-7; `ray_mesh_interact`), `normalize_exact()` (divide, above 1e-12; `ray_state_update`) | named by what they do; both forms kept |
| `VBS_INTERSECT` | `vbs_intersect()` | |
| `DIST_ADD`, `DIST_SET`, `ledger_at_relaunch` | `ledger_add`, `ledger_set` (lambdas), `ledger_at_relaunch` | |
| `fbs_sbs_not_parallel` | `faces_not_parallel()` | |
| `KILL_RAY`, `SCALE_RAY`, `REPLACE_BY_GAIN` | `kill`, `scale`, `pass_through` (lambdas) | the gain argument is always 1 |
| `EXACT_IN_SLAB_COSINE_SOLVE`, `EXACT_IN_SLAB_COSINE` | `recover_cosine()`, `in_medium_cosine()` (memo) | |
| `ct_slab_real_inv`, `face_real_inv` | `real_invariant`, `face_real_invariant` | |
| `RELAUNCH` | `realign_relaunch()` | |
| `REPLACE_BY_VBS`, `EXIT_DIRECTION`, `LEAK_DIRECTION` | `face_at_vbs()`, `exit_direction()`, `leak_direction()` | |
| `MASS_CODE`, `MASS_RUN`, `MASS_FOLD` | `encode_mass_code()` + `decode_mass_code()`, `bonded_x0()`, `bonded_thickness()` | what they compute |
| `PLATE_ABS`, `PLATE_IN`, `plate_out` | `face_absorption()`, `entry_face_absorption()`, `plate_exit_mtl` | |
| `SCALE_BY_MEDIUM` | `close_leg()` | it closes a leg out: gain and phase of the medium |
| `GET_EXIT_MATERIAL` | `exit_material()` | |
| `SLAB_AIRY_FACTOR` | `sum_series()` | |
| `SHEET_RESOLVE`, `SHEET_EVENT`, `SHEET_COMPOSITE` | `sheet_at_tracked_media()`, `sheet_event()`, `sheet_composite()` | |
| `APPLY_TRANSITION` | `transition()` | |
| `LEG_FROM_HIT_POINT` block | `relaunch_ledger_offset()` (file scope) | |
| `fresnel_r` (in `qd_slab_resolve`) | `slab_face_reflection()` | |
| `BND`, `PHI`, `MED`, `ADM`, `W`, `COS` (in `qd_stack_resolve`) | `boundary()`, `layer_phase()`, and members of a `StackMedia` helper | |

Test comments name `SCALE_BY_MEDIUM` (8 hits), `SHEET_EVENT` (7), `SLAB_AIRY_FACTOR` (6), `SHEET_RESOLVE` (4),
`REPLACE_BY_VBS` (2), `EXACT_IN_SLAB…`, `TRANSPARENT_TO`, `APPLY_TRANSITION`, `fresnel_r`; they go stale until
task 2 updates them from this table (decision 3 below). `md/wave.md`, `md/acoustic.md` and
`html_docu/quadriga_lib_material_model.md` (38 hits, rewritten in task 3) are read against this table.

#### Dead code

Established with a coverage build of this file alone (`-O0 --coverage`, linked into a scratch copy of the test binary
in place of its object), the whole suite run once (all passed, 1317534 / 1148; 107 s), `gcov -b`: 90.7 % of 2458
lines executed. 30 lines that are not input checks were never executed; branch counts where noted. Decided (Stephan,
2026-10-20): items 1–6 below are removed (item 5 as corrected), the guards are kept.

Unreachable for every input, not only in the suite (removed; none changes a result):
1. `Material::interface_gain`: `if (loss < 0.0) return 1.0;` (427–428) — `att ≥ 0` is validated when the table is
   read and every `Material` is built from the table, so `att · (f/fRef)^attB` is non-negative or NaN, and NaN fails
   the test; branch taken 0 of 21077594. The API's "clamped to `>= 0`" becomes the rule "`att` must be ≥ 0
   (validated)".
2. `SHEET_RESOLVE`: `thk <= 0.0` and `mtl_slab2 != 0 && thk2 <= 0.0` returns (4508–4511) — a sheet index has
   `thk > 0` by `IS_SHEET`, and a second layer exists only when both faces are sheets; taken 0 times.
3. `qd_stack_resolve`'s argument checks (1032–1036) — every caller passes one or two valid layers with validated
   `thk`; taken 0 times.
4. `ray_mesh_interact`: `if (iFBS)` (2277) and `(p_mtl_ind && iFBS)` (2263) — `iFBS == 0` has `continue`d at 2157;
   `if (i_rayN >= n_rayN) throw` (2794) — the map makes it impossible (and a throw inside the parallel loop would
   terminate the process rather than reach the caller).
5. Parameters that never vary: `qd_polbasis`'s `amplitude` (1.0 at all 5 sites; a product with 1.0 is exact),
   `REPLACE_BY_GAIN`'s `gain` (1.0 at all 3 sites: its `gain == 1.0` test was never false in 3588 calls, so the test
   reduces to `!refl_pass`). The unused out-arguments (`o_te`, `o_tm` at 1075; `e_te`, `e_tm` at 4273) go with the
   `Interface` result struct. Correction: `interact_with`'s `theta` is not removable (the angle path uses it); it is
   documented as ignored where a cosine is given.
6. `(int)typeH & ~32` (4967) — bit 5 was already cleared at 3446.

Guards on input or numerics that no test reaches (kept; listed for completeness): an empty column in `mtl_prop` (82);
`mass_X0` without `m` (358, every caller gates on `m`); `qd_mass_amp`'s opaque-layer cap (956–961); the singular
stack denominator and its three consumers (1296, 2604, 4660, 4773–4774); `trivecN` / `tridirN` reset without a tube
(2043, 2051); the cosine recovery's declines (3861, 3866) and evanescent root (3905, 4349); a tube vertex that cannot
be re-aimed (3970–3985); a zero mass code (4199) and inconsistent mass words (4216, 4230); a sheet event with a
deferred buffer (4723); `VBS_INTERSECT` fallbacks (4881, 4924). The input-validation throws are not counted.

### 0.4 The plan

#### Target layout (one file) — option C (Stephan, 2026-10-20)

Decided after a pro / con of lambdas against file-scope helpers: helpers that are pure, or that both ray
functions need, move to file scope; the per-ray event logic of `ray_state_update` stays as lambdas inside the
function, each with an explicit capture list instead of `[&]` and none nested in another.

```
header overview (≈ 110 comment lines, outline below)
includes
namespace {
  1. Constants and vector helpers      colocation_dist, half_pi, float_ulp, mass_min_path, word masks; k0,
                                       qd_evanescent, decaying_root, dist3, normalize_fast, normalize_exact,
                                       vertex_dir (tridir decode), qd_reflect, qd_refract, qd_relaunch_offset,
                                       qd_polbasis, write_coeff (xprmat column + gain of one ray)
  2. Material model                    MaterialCols<dtype>; Material { columns, eta, mu, medium_gain, mass_X0,
                                       excess_gain, interface_gain, tf, peaked, zeta, same_as };
                                       mass_TL, davy_sigma (free)
  3. Interface physics                 apply_tf_pair, tf_face, closed_face, flux_amp, interact_with (free, A and B
                                       explicit), returning struct Interface { cTE, cTM, cos_t2, n2_ratio, snell,
                                       tir, gain }
  4. Slab and stack solver             QdLayer, qd_mass_amp, qd_plate, StackOpts / Ports, qd_stack_resolve with its
                                       parts (held_plate, leak_chain, boundary, layer_phase, displaced host),
                                       SlabOpts, qd_slab_resolve with slab_face_reflection
  5. Sheets (both ray functions)       enum class SheetCase and its classification from the two faces; mass_cos_in_air
                                       (the cosine a mass term takes behind a refracting near medium, today written
                                       at 2585–2596 and 4629–4651); sheet_ports (the layer pair and the
                                       qd_stack_resolve call of both composites)
  6. Face hit geometry                 FaceHit, fbs_hit, sbs_hit, classify_hit (out_typeN bits), hit_media,
                                       tube_update (vertex incidences, whole-tube TIR, vertex re-aim, edge length)
  7. State machine, pure parts         StateIn and decode_words; encode_mass_code / decode_mass_code; bonded_x0,
                                       bonded_thickness; recover_cosine; face_absorption; same_eps_mu;
                                       vbs_intersect; exit_material; faces_not_parallel; medium_factor (the exact and
                                       the hybrid in-medium factor of a leg, returned as a complex number);
                                       relaunch_ledger_offset
}
medium_gain, interface_gain, refractive_index (API blocks, bodies, instantiations)
ray_mesh_interact (API block, body, instantiations)
ray_state_update (API block, body, instantiations)
```

`ray_mesh_interact` body: validation and output sizing (≈ 190 lines as today, with a small `size_or_reset`), the
output writers `set1` / `set3` / `set_row` as short lambdas, and a per-ray loop of ≈ 70 lines that calls sections 5
and 6 and writes. The no-hit pass uses `vertex_dir` and the writers.

`ray_state_update` body: validation and output sizing (≈ 170 lines), then per ray: decode (`decode_words`), the
geometry setup (≈ 60 lines), the lambdas below, the two passes as plain code, the ledger offset and the writes.
Lambdas kept (each defined once, in order of use, capture list explicit: read-only values captured by copy from
`const` locals, state the lambda changes or that changes after its definition by reference, so the list shows what
it reads and what it writes):
`kill`, `scale`, `pass_through` (the output writers); `in_medium_cosine` (the memo on `mtl_current`);
`realign_relaunch`; `exit_direction`, `leak_direction` (flattened out of `face_at_vbs`); `face_at_vbs`; `sum_series`;
`close_leg` (calls `medium_factor`, the plate rule, `scale`); `sheet_at_tracked_media`; `sheet_event`; `transition`. The
physics inside them is the file-scope helpers of sections 3–7.

#### Header overview, outline

1. What the file holds: the material model and the ray–surface physics of the tracer; the five public functions in
   one line each.
2. How they fit: `ray_progress` (and `calc_diffraction_gain`) call `ray_mesh_interact` and then `ray_state_update`
   once for the reflect pass and once for the transmit pass of every event; `ray_commit` calls `medium_gain` for the
   last leg; `ray_mesh_interact` sees faces only, `ray_state_update` holds the per-ray medium.
3. Units and conventions: frequency in Hz at the API, GHz inside; lengths in m; the incidence angle convention of
   `fbs_angleN` (angle to the face, `|cos| = |cos(θ + π/2)|`); TE/TM and the xprmat layout; transmit coefficients in
   normal-flux amplitude; geometry letters O, D, F, S, V, N; double inside both templates, `float` at the I/O only.
4. The material model: the columns and their units; `MaterialCols` and `Material`; wave media (ITU-R P.2040 ε, σ, μ,
   the Lorentz pole, `alpha`, `att`, `tf` as transformer or pad); sheets (`thk > 0`); the acoustic mapping (scalar
   types 3–5 run the same physics with the TE / admittance form, ε mapped from the impedance, wavelengths kept); the
   mass column `m > 0` as the acoustic switch: a lumped plate with no wave inside, the mass term (X0, Davy's panel),
   bonded bodies, the plate rule, the peaked face.
5. Interface physics: `interact_with` and its face models (open face: Fresnel + transformer; closed face: the pad;
   held wave in a plate).
6. The solver: `qd_stack_resolve` (sheets and two-layer stacks between half-spaces, far-end recursion, displaced host)
   and `qd_slab_resolve` (the series factor of a traced solid and its `eps` gate).
7. The state machine: the three words and their bits (one table), the ledger, the VBS correction, the recovered
   complex cosine, summed versus followed series, the mass code; the order of an event (redo the face, sum the series,
   close the leg, write the words).
8. The section map of the file.

#### Order of steps

Each step: build, the three suites, `rowdiff` against the phase 0 runs, counts unchanged; the report gets the line
and comment counts and the result. Expected result of every step: every number identical (see risks).
1. Comments outside the API blocks: delete the stale and redundant ones, shorten the mixed and the repeated ones.
2. Constants and scoping: `#define` → `constexpr`; named literals (same values); `const`; the 9 shadowing locals
   renamed; call constants hoisted (`fGHz`, `k0`); typos. The dead-code removals of 0.3 (decision 6).
3. Material and interface physics (sections 2, 3): the static members and the two-medium functions become free
   functions; the `Interface` result struct for `interact_with` (15 call sites); `decaying_root` for the six
   repeated roots.
4. Solver (section 4): `StackOpts` / `Ports` and `SlabOpts`; `qd_stack_resolve` split into its parts (`BND` and the
   held-plate branch first).
5. Sections 5 and 6 and the `ray_mesh_interact` body: the shared sheet helpers replace both copies; the face-hit
   geometry and the tube update move out; `vertex_dir`, `write_coeff`.
6. Section 7: the pure parts of the state machine move out of `ray_state_update`; the lambdas that remain call them.
7. The remaining `ray_state_update` lambdas: explicit capture lists, `EXIT_DIRECTION` / `LEAK_DIRECTION` flattened,
   the renames, the passes tidied.
8. The header overview; the API blocks shortened and the stale statements fixed; `tools/extract_md.py` into scratch.
9. Comment-only corrections in `tools/acoustic/txlib.py` and `band_probe.cpp` (from the rename table).

`rowdiff.py` is copied to `tools/acoustic/rowdiff.py` (and the README line added) before step 1, so every step is
compared with the tracked tool.

A provisional CPU-time check after steps 5 and 7 (to catch an inlining loss early); the measured comparison in
phase 2.

#### Expected counts

| | today | expected (option C) |
| --- | --- | --- |
| total lines | 5211 | ≈ 4000 (3800–4200) |
| code lines (with trailing comments) | 3053 | ≈ 2800: no context structs; the duplicated sheet, mass-cosine, vertex, root, xprmat-write and mass-code blocks once each; dead branches gone |
| comment-only lines | 1227 | ≈ 650, the overview included |
| API block lines | 527 | ≈ 320 |
| `ray_mesh_interact` body | 995 | ≈ 450 |
| `ray_state_update` body | 2030 | ≈ 750 (≈ 170 of them validation) |

#### Risks

- Floating-point order. This file is compiled `-O3 -fPIC` for baseline x86-64 (no `-mfma`; the AVX2 flags apply to
  `*_avx2.cpp` only), so moving an expression into a function, inlined or not, does not change a bit; only an
  expression written differently does. The plan writes no expression differently: `k0` keeps the evaluation order
  of the four copies, `decode_mass_code` the order of its two copies, `half_pi` the literal as written (commented as deliberate). Every step is
  therefore expected to leave every printed number identical, and a drift is a defect of the step, not an allowed
  last-digit move.
- `float`: the code reads the dtype values it stored back several times (xprmat and gain scaled in place after a
  dtype store, `path_dirN` read back in the ledger block). The writers keep every store and every read-back where it
  is; no write is fused into a double-precision product.
- State the lambdas share, which the capture lists must carry by reference: `mtl_current` changes inside
  `transition` and the reflection pass, and the cosine memo is keyed on it; `realign_relaunch` calls the recovery per
  vertex and restores `real_invariant`; `face_at_vbs` sets `face_real_invariant`, which the transition then writes into
  the word; `plate_exit_mtl` is set before `close_leg` reads it; `cTE`, `cTM` and `dir_next` written by `face_at_vbs` /
  `sheet_at_tracked_media` are what `sum_series` rebuilds from; the ledger is written at the end of the ray, after the offset.
  A by-copy capture is taken only of a `const` local, so the compiler rejects a copy of anything that changes.
- Shared helpers keep each caller's form: the two normalize helpers differ in arithmetic (multiply by the reciprocal
  above 2e-7 against divide above 1e-12) and both stay; `mass_cos_in_air` takes the recovered cosine where
  `ray_state_update` has one and none in `ray_mesh_interact`; `medium_factor` keeps the product order of today's
  factor.
- OpenMP: the file-scope helpers are pure or write only the ray's own index; no static locals.
- `MAT` builds a `Material` from the columns at each of its 55 sites; replacing a rebuild by a reference to one built
  earlier must take the same index at the same moment (`mtl_current` moves).
- Performance: moving helpers out of the function can change inlining; the ±2 % rule is checked after steps 5 and 7
  and in phase 2.
- Test comments that name the renamed lambdas go stale until task 2.

### 0.5 Side findings (one line each, no items)

- `interface_gain` and `ray_mesh_interact` test `center_frequency <= 0` only; the other three functions also reject a
  non-finite frequency.
- `MaterialCols` ignores unknown columns silently (an old `coiA` column, or a misspelled one).
- The two `ISSUE` comments (dropped, decision 6): the first names `calc_diffraction_gain`; the caller of `medium_gain`
  is now `ray_commit.cpp`, whose last leg to a receiver inside a medium is charged over the straight geometric length
  with the mass term at normal incidence. The second states what the API block already says.
- The near-pole clamp and re-emit that the `ray_state_update` API block describes were removed in aecaf04
  (2026-10-13, "One solver for sheets and stacks, and no clamp at a wave face"); `md/wave.md` 2 says "Nothing is
  clamped". A test case is still named *ray_state_update - resolve clamp near the Airy pole* (task 2).
- Lambda names in the repository are lowercase (69 in 20 source files); uppercase (47) only in this file and
  `calc_diffraction_gain.cpp`.
- `1.570796326794897` is 2 ulp above the double nearest π/2; it is used consistently and stays.

### 0.6 Decisions (Stephan, 2026-10-20)

All taken as recommended; phase 1 starts only after Stephan has reviewed this report again.
Reviewed and accepted (manager, 2026-10-20), with Stephan's name changes in the rename table (state-word bit
names, `normalize_fast` / `normalize_exact`, `same_eps_mu`, `face_at_vbs`, `sheet_at_tracked_media`, no `qd_` prefix
on new helpers, `half_pi` commented as deliberate).

1. **Layout: option C.** Pure and shared helpers at file scope (sections 5–7 of 0.4); the per-ray event logic of
   `ray_state_update` stays as lambdas, flat, each with an explicit capture list (read-only values by copy from
   `const` locals, changed state by reference).
2. **Plan.** The step order of 0.4; the counts are estimates, not targets. Every refactoring step is bit-identical:
   a drifting printed number is a defect of that step. Simplifications that would move last bits (merging the two
   normalize forms, the exact double for π/2, a reordered operand) are not part of any step: they are collected as a
   short list at the phase 1 stop, each explained and measured with `rowdiff`, and done only on Stephan's decision.
3. **`rowdiff`.** Tracked as `tools/acoustic/rowdiff.py`, with one line in `tools/acoustic/README.md`.
4. **Renames.** As in the rename table, in the repository's lowercase style. Names that tests or tools mention and
   that do not mislead are kept (list above the table). Test comments that name a renamed internal are updated in
   task 2 from the table; `md/wave.md`, `md/acoustic.md` are read against it; the material-model page is rewritten in
   task 3. Stephan reviews the new names in the table.
5. **Interface physics.** `apply_tf_pair`, `tf_face`, `closed_face`, `flux_amp`, `interact_with` become free
   functions with both media explicit (A the side the wave is on), names kept, in their own section;
   `interact_with` returns `struct Interface` in place of its seven out-pointers; `mass_TL` and `davy_sigma` become
   free functions as well.
6. **Dead code.** Items 1–6 of the dead-code list removed (item 5 as corrected: `theta` stays); every listed guard
   kept; the rule of item 1 goes into the `interface_gain` / `mtl_prop` text.
7. **`ISSUE` comments.** Both dropped. `medium_gain`'s API block states that it takes one path length, with the mass
   term at normal incidence. The `ray_commit` point is the side finding of 0.5, nothing more.
8. **API statements.** Every stale statement of 0.3 is corrected to what the code does (documentation only), in
   step 8: the near-pole re-emit and clamp removed (the series is summed where ρ ≥ eps, nothing limits it near a
   pole); the `eps ≥ 1` sentence (the gate is `ρ < eps` alone); `normal_vecN` required; the two-sheet stack
   implemented; `calc_diffraction_gain` calls `ray_state_update`; `fbsN` / `sbsN` `[n_rayN, 3]`; bit 4 of
   `resolved_typeN`; `coiA` not read; the resonance not part of ε_r for `m > 0`; unit reflection under TIR only
   without `tf`. The header comment of `thin_slab_threshold` (`include/quadriga_tools.hpp`) is correct and untouched.

## Phase 1 — the cleanup (2026-10-20)

Started on Stephan's go-ahead. Scratch: `build_linux/scratch/cl1` (not kept). Every step: build, the three suites
with the 12-thread commands, `tools/acoustic/rowdiff.py` against the phase 0 runs (`r1_*.txt`).

### 1.1 Setup

- `tools/acoustic/rowdiff.py` tracked (the phase 0 tool, its usage line pointing at the tracked path, the licence
  header of the other tools), one row in `tools/acoustic/README.md`.
- Formatting: the source is in the style of VS Code's C++ fallback format (clang-format, Allman braces, namespace
  indentation, no column limit). After every step the file was formatted with the clang-format bundled with the
  C++ extension in that style (`SortIncludes: false`); whitespace only. On the base file it re-aligns 26 lines of
  trailing comments the base had aligned by an older version and changes nothing else.

### 1.2 Steps

All three suites passed at every step with the phase 0 counts (`[wave]` 165594 / 181, `[acoustic]` 42391 / 60, whole
suite 1317534 / 1148; the twins are asserted inside the suite), and `rowdiff` found every printed number identical at
every step: 140986 / 112537 / 256712 numbers, 0 within one last digit, 0 beyond, 0 structural. No step needed a
decision. No compiler warning in this file at any step (`-Wall -Wextra -Wpedantic`).

| step | what changed | lines | comment-only | API blocks | code (with trailing comments) |
| --- | --- | --- | --- | --- | --- |
| base | | 5211 | 1227 | 527 | 3053 |
| 1 | comments: 63 stale or redundant blocks deleted, 74 rewritten shorter (keyed to the classification of 0.3) | 4674 | 694 | 526 | 3053 |
| 2 | constants and masks (`colocation_dist`, `half_pi`, `float_ulp`, `mass_min_len`, the eight state-word masks, `k0()`); `fGHz` hoisted out of the ray loop; `const`; one shadowing local; typos; the dead code of 0.6 item 6 | 4682 | 699 | 526 | 3051 |
| 3 | `mass_TL`, `davy_sigma` and the two-medium functions out of `Material` (interface physics section); `interact_with` returns `Interface` (15 call sites); `decaying_root` for the six roots | 4652 | 699 | 526 | 3019 |
| 4 | solver: `StackOpts` / `Ports`, `SlabOpts`; `qd_stack_resolve` in parts (`StackMedia`, `stack_held_plate`, `stack_leak_chain`, `PlateRun`, `stack_boundary`, `stack_layer_phase`); `slab_face_reflection` | 4669 | 696 | 526 | 3032 |
| 5 | sheets and face-hit geometry: `SheetCase`, `sheet_ports`, `mass_cos_in_air` (both copies replaced), `RayWriter`, `RmiOut`, `face_plane`, `ray_plane_hit`, `classify_hit`, `hit_media`, the tube helpers, `rmi_no_hit`; the new `ray_mesh_interact` loop | 4705 | 722 | 526 | 3051 |
| 6 | the pure parts of the state machine to file scope (`decode_words`, the mass code, `bonded_x0`, `bonded_thickness`, `recover_cosine`, `face_absorption`, `same_eps_mu`, `vbs_intersect`, `faces_not_parallel`, `exit_material`, `leg_exact_factor`, `sheet_hit`, `relaunch_ledger_offset`); the lambdas call them | 4815 | 737 | 526 | 3136 |
| 7 | the `ray_state_update` loop rewritten: explicit capture lists, the renames, `exit_direction` / `leak_direction` out of the face lambda, `vbs_geometry`, the writer and `write_coeff` for the four copies of the xprmat write | 4711 | 682 | 526 | 3112 |
| 8 | header overview (84 lines), section headers 1–7 (the vector helpers moved up to section 1), the five API blocks shortened and corrected; the remaining renames in `ray_mesh_interact` (`snell`, `n2_ratio`, `theta_fbs`, `theta_sbs`, `mtl_fbs`, `mtl_sbs`, `hit_code`) | 4727 | 769 | 451 | 3112 |
| 9 | `tools/acoustic/txlib.py`: two comments (`mass_TL`, `davy_sigma` are no longer `Material::` members); `band_probe.cpp` needed none | | | | |

`tools/extract_md.py -a cpp` into scratch after step 8: the five entries are there; outside them the extracted page
differs only in the index's line numbers (82 lines), because the entries are shorter.

### 1.3 Against the estimates of 0.4

| | base | estimate | now |
| --- | --- | --- | --- |
| total lines | 5211 | ≈ 4000 | 4727 |
| code (with trailing comments) | 3053 | ≈ 2800 | 3112 |
| comment-only | 1227 | ≈ 650 | 769 (84 of them the header overview) |
| API blocks | 527 | ≈ 320 | 451 (the code tables of `out_typeN` and `resolved_typeN` are most of it) |
| `ray_mesh_interact` body | 995 | ≈ 450 | 440 |
| `ray_state_update` body | 2030 | ≈ 750 | 1267 (170 of them input checks) |

The code is longer than estimated by about 300 lines: the helper signatures, the fourteen small structs, and in
`ray_state_update` the explicit capture lists (two to five lines each for the twelve event lambdas) and the doc
comment of each lambda. The estimate for `ray_state_update` did not count that its lambdas stay in the body
under option C. The comment-only count is above the estimate by the overview and the per-function doc comments
of the new helpers.

### 1.4 What the code is now

- One file, one anonymous namespace before the public functions, in seven sections: 1 constants and vector helpers
  (`k0`, `qd_evanescent`, `decaying_root`, `qd_reflect`, `qd_relaunch_offset`, `qd_refract`, `qd_polbasis`); 2
  material model (`MaterialCols`, `davy_sigma`, `mass_TL`, `Material`); 3 interface physics (`apply_tf_pair`,
  `closed_face`, `tf_face`, `flux_amp`, `Interface`, `interact_with`); 4 slab and stack solver; 5 sheets;
  6 face-hit geometry; 7 the pure parts of the state machine.
- `ray_mesh_interact`: input checks and output sizing as before, then a per-ray loop of about 250 lines that reads
  the hit, calls `face_plane` / `ray_plane_hit`, `classify_hit`, `hit_media`, `interact_with`, the sheet
  composite, `tube_incidence` / `tube_update`, and writes.
- `ray_state_update`: input checks as before; per ray the decode, the setup (geometry, `vbs_geometry`, the ledger),
  the twelve event lambdas, the two passes as plain code, the relaunch offset, the writes. The capture lists
  follow one rule: by copy the values that do not change once the ray is set up (the decoded words, the angles,
  lengths and flags of the leg, the call's pointers and switches); by reference the state an event changes
  (`mtl_current`, the codes, the three output words, the ledger, `cTE` / `cTM`, `dir_next`, the cosine memo and
  `real_invariant`, `face_real_invariant`, `series_summed`, `plate_exit_mtl`), the arrays (`N`, `F`, `S`, `V`, `D`,
  `dir_prev`), `cols`, the writer and the lambdas it calls.

### 1.5 CPU time (provisional; the measured comparison is phase 2)

Single-threaded (`OMP_NUM_THREADS=1`), today's binary (`test_bin_base`, phase 0) and the step's binary alternated,
5 rounds each, user + system time, under `flock /tmp/bench.lock`; ratio = median of the paired new / base ratios.
H1 = *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer*, H2 = *Measurement -
the mirror behind the ray after a back reflection*.

| | after step 5 | | after step 7 | |
| --- | --- | --- | --- | --- |
| | base / new median (s) | ratio (min–max) | base / new median (s) | ratio (min–max) |
| `[wave]` | 24.34 / 23.80 | 0.978 (0.958–0.992) | 24.67 / 24.46 | 0.992 (0.983–0.998) |
| `[acoustic]` | 0.86 / 0.86 | 1.000 (1.000–1.012) | 0.87 / 0.88 | 1.000 (0.977–1.024) |
| H1 | 9.75 / 9.25 | 0.949 (0.946–0.969) | 9.92 / 9.72 | 0.976 (0.960–0.992) |
| H2 | 9.06 / 9.02 | 1.001 (0.982–1.010) | 9.12 / 9.15 | 1.003 (0.986–1.033) |
| 1-min load | 1.8–2.8 | | 2.1–4.0 (4.13 at the start) | |

No slowdown; the step 7 run was at the edge of the load rule (4) at its start.

### 1.6 The rename table as built

Renamed (old → new); everything not listed kept its name (`interact_with`, `tf_face`, `closed_face`, `apply_tf_pair`,
`flux_amp`, `Material` and its members, `MaterialCols`, every existing `qd_` function, `dist_refract`, `dist_geo`,
`use_vbs`, `refracted`, `bent`, `bent_carry`, `bent_enter`, `followed`, `mass_crossed`, `stack_behind`, `cTE`,
`cTM`, `dir_prev`, `dir_next`).

| old | new | kind |
| --- | --- | --- |
| `#define colocation_dist` | `constexpr double colocation_dist` | constant |
| `1.570796326794897`, `1.1920929e-7`, `0.0015` | `half_pi` (commented as deliberate), `float_ulp`, `mass_min_len` | constants (`Material::medium_gain`'s parameter `mass_min_path` keeps its name) |
| `0x3FFF`, `0x7FFF`, `0x8000` / `0x4000` per word | `word_mtl`, `word_code`; `cur_summed`, `cur_bent`; `prev_nonparallel`, `prev_embedded`; `buf_mass_code`, `buf_followed` | masks |
| k0 expression | `k0(fGHz)` | function |
| decaying-root pattern | `decaying_root()` | function |
| `Material::mass_TL`, `Material::davy_sigma`, `Material::flux_amp` | `mass_TL`, `davy_sigma`, `flux_amp` | free functions |
| `A.apply_tf_pair(B, …)`, `A.tf_face(B, …)`, `A.closed_face(B, …)`, `A.interact_with(B, …)` with out-pointers | `apply_tf_pair(A, B, …)`, `tf_face(A, B, …)`, `closed_face(A, B, …)`, `Interface interact_with(A, B, …)` | free functions |
| `eta1_div_eta2`, `eta1_d_eta2` | `n2_ratio` | (also the `Interface` member) |
| `qd_stack_resolve`'s 17 / `qd_slab_resolve`'s 18 parameters | `StackOpts` + `Ports`, `SlabOpts` + `S_te` / `S_tm` by reference | structs |
| `MED`, `W`, `ADM`, `COS`, `CLOSED` | `StackMedia::med`, `w`, `adm`, `cosine`, `closed` | members |
| `BND` (with `PLATE_OUT`, `PLATE_FIX`, `TF_FACE`), `PHI`, `UNDISP` | `stack_boundary` (with `plate_out`, `plate_fix`, `open_face`), `stack_layer_phase`, `undisplace` | functions / lambdas |
| the held-plate branch, the leak chain | `stack_held_plate`, `stack_leak_chain` | functions |
| `fresnel_r` | `slab_face_reflection` | function |
| rmi `NORMALIZE`, rsu `NORMALIZE` | `normalize_fast` (reciprocal, 2e-7), `normalize_exact` (divide, 1e-12) | functions |
| rmi `LENGTH`, rsu `DIST` | (inline), `dist3` | |
| `SET1`, `SET3`, `SETL` | `RayWriter::set1`, `set3`, `set_row`, `set_col` | struct |
| `sheet_case` 0–6 (rmi), 0–4 (rsu) | `enum class SheetCase` | enum |
| rmi `iMF`, `iMS`, `M1`, `M2`, `M_slab`, `M_slab2` | `mtl_fbs`, `mtl_sbs`, `media.near`, `media.far`, `media.sheet`, `media.sheet2` | |
| rmi `eta`, `thetaF`, `thetaS` (cosine, then angle), `tir_central`, `out_flags` | `snell`, `theta_fbs`, `theta_sbs` (+ `cos_sbs`), `tir_spine`, `hit_code` | |
| rmi `p_trivec_tmp`, `p_tridir_tmp`, `edge_length_tmp` | `tube_tv`, `tube_td`, `edge_length` | |
| rmi no-hit branch, FBS / SBS geometry, classification, media, `SHEET_COMPOSITE`, tube blocks | `rmi_no_hit`, `face_plane` + `ray_plane_hit`, `classify_hit`, `hit_media`, (inline with `sheet_ports`), `tube_incidence`, `tube_update`, `tube_edge_length`, `tube_vertex_dir` | functions |
| rsu `ii` | `i_rayN` (the loop counter is `i_ev`) | |
| rsu `typeH`, `typeR`, `nH`, `M1`, `M2`, `resolved`, `total_reflection` | `hit_code`, `res_code`, `n_hits`, `mtl_fbs`, `mtl_sbs`, `leg_summed`, `tir_hit` | |
| rsu bool `prev_embedded` | `prev_is_embedded` (the mask has the name) | |
| rsu `dist_orig_fbs`, `dist_fbs_dest`, `dist_orig_vbs` | `len_to_face`, `len_face_to_dest`, `len_to_vbs` | |
| rsu word decode | `StateIn decode_words()` | function |
| `MAT`, `SAME`, `TRANSPARENT_TO`, `IS_SHEET` | `material()`, `same_medium()`, `same_eps_mu()`, `is_sheet_mtl()` | functions |
| sheet case derivation | `SheetHit sheet_hit()` | function |
| `VBS_INTERSECT`, the VBS block | `vbs_intersect()`, `VbsGeometry vbs_geometry()` | functions |
| `DIST_ADD`, `DIST_SET` | `ledger_add`, `ledger_set` | lambdas |
| `fbs_sbs_not_parallel` | `faces_not_parallel()` | function |
| `KILL_RAY`, `SCALE_RAY`, `REPLACE_BY_GAIN` | `kill`, `scale`, `pass_through` | lambdas |
| the polbasis + write copies | `write_coeff` | lambda |
| `EXACT_IN_SLAB_COSINE_SOLVE`, `EXACT_IN_SLAB_COSINE`, `ct_slab_exact_for` / `_val`, `ct_slab_real_inv` | `recover_cosine()` (the solve) + `recover_in_current` (its gates), `in_medium_cosine`, `ct_memo_for` / `ct_memo`, `real_invariant` | function, lambdas |
| `face_real_inv` | `face_real_invariant` | |
| `RELAUNCH` | `realign_relaunch` | lambda |
| `REPLACE_BY_VBS`, `EXIT_DIRECTION`, `LEAK_DIRECTION` | `face_at_vbs`, `exit_direction`, `leak_direction` (no longer nested) | lambdas |
| `MASS_CODE`, d-code expression ×2 | `encode_mass_code()`, `decode_mass_code()` | functions |
| `MASS_RUN`, `MASS_FOLD` | `bonded_x0()`, `bonded_thickness()` (file scope) via `leg_bonded_x0`, `leg_bonded_thickness` | functions, lambdas |
| `PLATE_ABS`, `PLATE_IN`, `plate_out` | `face_absorption()`, `entry_face_absorption`, `plate_exit_mtl` | function, lambda |
| `SCALE_BY_MEDIUM` (its exact form) | `close_leg` (`leg_exact_factor()`) | lambda (function) |
| `GET_EXIT_MATERIAL` | `exit_mtl` (`exit_material()`) | lambda (function) |
| `SLAB_AIRY_FACTOR` | `sum_series` | lambda |
| `SHEET_RESOLVE`, `SHEET_EVENT` | `sheet_at_tracked_media`, `sheet_event` | lambdas |
| `APPLY_TRANSITION` | `transition` | lambda |
| `LEG_FROM_HIT_POINT` block | `relaunch_ledger_offset()` | function |

Test comments that name the old internals go stale until task 2 (decision 4).

### 1.7 Last-bit simplifications, for decision

None was done. Each was measured as a scratch variant of today's file (compiled into a copy of the library and
linked into a copy of the test binary; the whole suite, `rowdiff` against the phase 0 run):

| | change | assertions | printed numbers (of 256712) | recommendation |
| --- | --- | --- | --- | --- |
| A | one normalize form: `normalize_fast` divides as `normalize_exact` does | 1 fails (a twin at 1e-12: *twin scalar closed face C, bare wall type 3,45*, `test_wave_state_update_branch.cpp:648`) | 987 within one digit, 2408 beyond (691 of them real values, among them integer columns), 2 structural | drop: not a last-bit change |
| B | `half_pi` = 1.5707963267948966 (the double nearest π/2) | all pass | 143 within one digit, 1326 beyond, every one a rounding-noise column (`|x| < 1e-9`, the `d_…` residues) | Stephan's call: no physical value moves past its last digit, but the rule of the brief (every printed number within one digit) is not met; one literal and its comment less |
| C | `davy_sigma`: `k = k0(fGHz)` in place of `20.958450219516818 * fGHz` | all pass | 4 within one digit (a `-0.0000` that becomes `+0.0000` in *Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency*), 0 beyond | do it: one magic number less |

### 1.8 Side findings (one line each, no items)

- The tube's vertex refraction (`tube_update`) takes the principal root of `1 − n2_ratio sin²`, not `decaying_root`;
  the cases where the two differ (an evanescent vertex) are escalated to whole-tube total reflection before, so it
  is never reached.
- The VS Code formatter of this machine aligns trailing comments differently from the base in 26 places.

### 1.9 Stop

Phase 1 is complete (steps 1–9). For Stephan: the list of 1.7 (A drop, B decide, C do), then phase 2 (the final
checks, the measured CPU-time comparison, the read-through).

## Phase 2 — final checks (2026-10-20)

Decided after phase 1 (Stephan, 2026-10-20): 1.7 A dropped; 1.7 B (`half_pi` the double nearest π/2, its
"deliberate value" comment removed; the one-digit rule of the brief is overridden for exactly the values 1.7
measured); 1.7 C (`davy_sigma` takes `k0(fGHz)`). Scratch: `build_linux/scratch/cl1/p2final` (not kept).

### 2.1 B and C applied

- `constexpr double half_pi = 1.5707963267948966;` (was `1.570796326794897` with a one-line comment that the value is
  deliberate; the comment is gone).
- `davy_sigma`: `k = k0(fGHz)` in place of `20.958450219516818 * fGHz`; with `half_pi` now exact, its
  `std::sqrt(1.5707963267948966 / ka)` reads `std::sqrt(half_pi / ka)` (the same double, no bit moves).

### 2.2 The suites against the baselines

Build without a warning in this file; the three suites with the 12-thread commands, `tools/acoustic/rowdiff.py`
against the phase 0 runs:

| suite | result | numbers identical | within one last digit | beyond | structural |
| --- | --- | --- | --- | --- | --- |
| `[wave]` | All tests passed (165594 assertions in 181 test cases) | 139650 | 135 | 1201 | 0 |
| `[acoustic]` | All tests passed (42391 assertions in 60 test cases) | 112400 | 12 | 125 | 0 |
| whole suite | All tests passed (1317534 assertions in 1148 test cases) | 255239 | 147 | 1326 | 0 |

- The counts are the baselines'. `tests/data/wave_twins.tsv` (asserted inside the suite at 1e-12 relative) passes and
  is not rewritten; no file under `tests/` or `include/` changed.
- The whole-suite output equals the two what-if runs of 1.7 line for line: every line that differs from the phase 0
  run is the line of the B run (722 lines) or of the C run (4 lines), no line is changed by both, and no other line
  differs. So the counts are those of 1.7: B 143 within + 1326 beyond, C 4 within (143 + 4 = 147).
- By column kind: the 1326 beyond and 63 of the within are values in e-format with `|x| < 1e-9` in both runs (the
  largest 1.7e-13: the `d_…` residues and the other rounding-noise columns); the other 84 within are a signed zero
  that flips (`-0.000000` ↔ `+0.000000`, `-0.0000` ↔ `+0.0000`, `-0.000` → `0.000`), the 4 of C among them. No
  value that is not rounding noise moved.
- The 1326 by case (whole suite, largest first): *Stack with tf - the scattering form against its reference,
  produce mode and itself* 192; *Lumped columns - the transmission factor on a sheet, both ports* 177; *Neighbour
  tf - each face of a sheet is owed by the pair of media that form it* 125; *Stack with a mass-law layer - resolved
  by segments, not killed* 61; *Lumped columns - an ordered stack: what the core carries, by matrix and by
  scattering* 61; *Glued sheet - the near-to-far flux conversion* 58; 47 more cases with 1–46 each, and 17 values
  under a table header `rowdiff` could not trace to a case. By file: `test_wave_state_update_sheet.cpp` 1143,
  `test_acoustic_state_update.cpp` 131, `test_wave_state_update_chain.cpp` 35, untraced 17.

Then the three baselines regenerated with the commands of `md/wave.md` 0 (`test_wave.txt`, `test_acoustic.txt`,
`test.txt`; counts as above; `rowdiff` of each against the checked run: every number identical), and
`python3 tools/acoustic/check_docs.py`: RESULT all checks pass (tables, keys, the 60 `[acoustic]` cases, the quoted
counts of `md/wave.md` and `md/acoustic.md` against the new baselines, spelling).

### 2.3 CPU time

Single-threaded (`OMP_NUM_THREADS=1`), today's committed library (`build_linux/scratch/cl0/test_bin_base`, built from
6307a64: the base source's md5 is that of `git show HEAD:src/ray_mesh_interact.cpp`) and the final binary
alternated, 7 rounds, user + system time, under `flock /tmp/bench.lock`; ratio = median of the paired new / base
ratios. H1 = *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer*, H2 =
*Measurement - the mirror behind the ray after a back reflection* (the two heaviest ray cases, as in 1.5).

| | run 1 (load 3.1–14.7: contaminated) | | run 2 (load 2.0–3.5) | |
| --- | --- | --- | --- | --- |
| | base / new median (s) | ratio (min–max) | base / new median (s) | ratio (min–max) |
| `[wave]` | 24.18 / 23.93 | 0.987 (0.904–1.003) | 24.29 / 23.89 | 0.983 (0.980–0.997) |
| `[acoustic]` | 0.86 / 0.86 | 1.000 (0.904–1.035) | 0.86 / 0.86 | 1.000 (0.977–1.023) |
| H1 | 9.80 / 9.53 | 0.972 (0.898–1.070) | 9.76 / 9.58 | 0.985 (0.945–0.991) |
| H2 | 9.09 / 9.02 | 1.001 (0.917–1.074) | 9.12 / 9.01 | 0.991 (0.976–1.016) |

Run 1 started at load 14.5 and is reported only for completeness; run 2 stayed below the load rule (4) from start
(2.83) to end (2.09) and is the result: all four within ±2 % (`[wave]` and H1 about 1.5 % faster, `[acoustic]`
and H2 unchanged). No slowdown.

### 2.4 Final counts

| | base (6307a64) | after phase 1 | final |
| --- | --- | --- | --- |
| total lines | 5211 | 4727 | 4726 |
| code (with trailing comments) | 3053 | 3112 | 3112 (2736 + 376 with a trailing comment) |
| comment-only | 1227 | 769 | 768 (223 blocks; 84 of them the header overview) |
| API blocks (`/*!MD`) | 527 | 451 | 451 |
| blank | 405 | 395 | 395 |
| `ray_mesh_interact` body | 995 | 440 | 440 |
| `ray_state_update` body | 2030 | 1267 | 1267 (170 of them input checks) |

`tools/extract_md.py -a cpp` into scratch: exit 0, all five entries present (`medium_gain`, `interface_gain`,
`refractive_index`, `ray_mesh_interact`, `ray_state_update`), 6299 lines, identical to the extract after step 8.

### 2.5 The rename table as built

The table of 1.6 is the final one; phase 2 changed one row: `1.570796326794897` → `half_pi` = 1.5707963267948966,
the double nearest π/2, without a comment (and `davy_sigma`'s `20.958450219516818 * fGHz` → `k0(fGHz)`,
`1.5707963267948966` → `half_pi`). Checked by searching the source for every old name of the table: none is left
except the four that the table keeps on purpose in another place (`total_reflection` in `ray_mesh_interact` and
`tube_update`, renamed only in `ray_state_update`; `plate_out`, the lambda of `stack_boundary`; `M_slab`, a local of
`sheet_at_tracked_media`; `StateIn::resolved`, the decoded bit that `leg_summed` is read from).

### 2.6 The header overview as written

Lines 5–88 of `src/ray_mesh_interact.cpp`, verbatim:

```
// The material model and the ray-surface physics of the ray tracer.
//
// Public functions (each documented in the API block above its definition):
//   medium_gain        power gain of a path through a homogeneous medium: attenuation, and the mass term of a
//                      mass-law material at normal incidence
//   interface_gain     the lumped penetration gain (att) of entering a material
//   refractive_index   Re sqrt(eps mu) of a material
//   ray_mesh_interact  one face event per ray: the hit geometry, the interface coefficients (Fresnel / Jones, or
//                      the composite of a thin sheet), the outgoing direction, the relaunch segment and the ray
//                      tube. It sees faces only and assumes air on every side the mesh does not name.
//   ray_state_update   the per-ray medium state machine: corrects what ray_mesh_interact wrote with the medium the
//                      ray is known to be in (the face at the virtual hit point VBS of the wave's own direction),
//                      closes the leg out (in-medium loss and phase), sums the bounce series of a layer in closed
//                      form where it is strong enough, re-resolves a sheet against the tracked media, and writes
//                      the state words and the ledger for the next event.
// ray_progress and calc_diffraction_gain trace a segment per ray (ray_triangle_intersect) and call
// ray_mesh_interact and then ray_state_update twice per event: once for the reflect pass (types 0, 3), once for
// the transmit pass (1, 2, 4, 5). ray_commit uses medium_gain for the last leg to a receiver inside a medium.
//
// Conventions
// - Frequency in Hz at the API, GHz inside (fGHz). Lengths in m. All arithmetic is double in both templates;
//   float is the storage of inputs and outputs only.
// - Interaction types: 0 reflect, 1 transmit (the tracer keeps the direction; path_dirN reports the Snell one),
//   2 refract (the tracer bends); 3, 4, 5 the same as scalar types, which run the same physics with one pressure
//   coefficient in the TE / admittance form (z = sqrt(eps / mu) is then the acoustic admittance).
// - Face angle: theta = acos(u . n) - pi/2 is the angle to the face, >= 0 for a front-side hit (the vertices
//   counter-clockwise seen from the ray); |cos(theta + pi/2)| is the cosine to the normal.
// - Coefficients: TE / TM; a transmitted coefficient is a normal-flux amplitude (flux_amp), so
//   0.5 (|cTE|^2 + |cTM|^2) is the power that crosses. xprmat per ray: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH]
//   (qd_polbasis) or [Re Im] for the scalar types.
// - Geometry: O the segment's origin, D its destination, F and S the first and second hit (FBS, SBS), N a face
//   normal (oriented to the incidence side where a direction is formed), V the VBS.
// - The relaunch point lies 8 float ulp of the hit point's magnitude past it (qd_relaunch_offset): scenes are
//   meant to sit in local coordinates near the origin.
//
// The material model (MaterialCols reads the columns, Material holds one material in double):
// - eps_r = a (f/fRef)^b - j 17.98 sigma / fGHz with sigma = c (f/fRef)^d (ITU-R P.2040), plus a Lorentz pole
//   (resF, resQ, resS) for a material without the mass column; mu_r = e (f/fRef)^f - j 17.98 g (f/fRef)^h / fGHz.
// - Loss: 8.686 k0 |Im n| and alpha (f/fRef)^alphaB dB/m in the medium; att (f/fRef)^attB dB once at entry.
// - tf (f/fRef)^tfB, plus a resonant peak (tfP, tfF, tfQ), clamped to [-1, 1]: the transmission factor of a face.
//   > 0 leaks reflected energy into transmission, < 0 forces reflection. At an open face (a wave on both sides)
//   it is an ideal transformer that keeps the normal flux (tf_face); at a closed face (past the critical angle) a
//   matched resistive pad in front of the reactance behind it (closed_face).
// - thk > 0 makes a material a sheet: a face that stands for a layer of that thickness, its internal series
//   summed in one event (qd_stack_resolve) instead of traced. Two co-located sheets are an ordered stack.
// - The acoustic mapping (in the material table, not in this code): frequencies are scaled so that wavelengths
//   are kept (1 kHz is 0.874 GHz), and eps, mu are set from the impedance (and the sound speed).
// - m > 0, the mass column, is the acoustic switch. The material is a lumped plate with no wave inside: its eps,
//   mu are an impedance mapping for its faces, no tracer bends into it, a wave in it is held undeviated, and no
//   series is summed in it. Its transmission is the mass term: the impedance ratio X0 = (f/fRef) d of its normal
//   thickness d (mass_X0, with the res* and coi* impedances), 20 log10 X0 dB at normal incidence and the mass
//   law's angle shape (mass_TL; with side > 0 the finite panel of Davy). Bonded mass-law bodies are one plate
//   (their X0 add), and a plate left into a wave medium passes the plate rule of its two outer faces (qd_plate).
//   A peaked mass-law face (tfP > 0) is a locally reacting series impedance (zeta).
//
// Interface physics: interact_with(A, B) is the face between the medium the wave is in and the next one for one
// interaction type, at a real angle or at a recovered complex cosine: Fresnel (ITU-R P.2040 eq. 31, 32), total
// reflection (only towards a lossless medium), tf, the held wave of a plate, att.
//
// The solver: qd_stack_resolve gives the transmit and reflect ports of a sheet or a stack of two between two
// half-spaces (a recursion from the far end; the medium the sheet displaces is taken back out).
// qd_slab_resolve gives the series factor S = 1 / (1 - r_near r_far phi^2) of a traced solid, where its round
// trip rho = |r_near r_far| |phi|^2 reaches eps.
//
// The state machine (ray_state_update). Three 16-bit words per ray carry what an event hands to the next:
//   current  bits 0-13 the medium the ray is in (0 air); bit 14 cur_bent: the leg follows a bend the tracer
//            traced (types 2, 5); bit 15 cur_summed: the leg's series was summed, so a later reflection in the
//            leg (already inside the sum) is killed
//   prev     bits 0-13 the medium the leg was entered from, or the sheet crossed or reflected off (the mirror
//            behind the ray); bit 14 prev_embedded: that sheet lies inside the current medium (for a solid: the
//            shared face was evaluated at a real-invariant cosine); bit 15 prev_nonparallel: the leg's faces are
//            not parallel, so no series is summed (eps > 0) and no cosine recovered
//   buffer   bits 0-13 a deferred medium (overlapping or corner geometry); bit 14 buf_followed: the leg began at
//            a reflection that left its series to the tracer (with an index: the layer of a stack behind the ray);
//            or, under bit 15 buf_mass_code, the thickness code of a bonded mass-law body crossed into the leg
// and the ledger (acc_dist_refracted, acc_dist_geometric) holds the length of the current leg, charged when the
// leg closes. An event: the face is redone at the VBS angle (face_at_vbs), the leg's series is summed
// (sum_series) or left to the tracer, the leg is closed out (close_leg), the words are written. A traced ray
// carries only the real projection of a complex in-medium cosine; recover_cosine inverts the tracer's map to the
// exact cosine, so the entry and the exit of a lossy layer are evaluated at one cosine.
//
// Sections of the anonymous namespace: 1 constants and vector helpers; 2 material model; 3 interface physics;
// 4 slab and stack solver; 5 sheets (both ray functions); 6 face-hit geometry (ray_mesh_interact); 7 the pure
// parts of the state machine (ray_state_update). Then the public functions, each after its API block.
```

### 2.7 Read-through as an agent that sees only `src/ray_mesh_interact.cpp`

Read top to bottom, all 4726 lines. Verdict: the overview plus the code explain the five functions.
`medium_gain`, `interface_gain` and `refractive_index` are one `Material` call each and their API blocks say all of
it. `ray_mesh_interact` reads as a sequence of named steps (`face_plane` / `ray_plane_hit`, `classify_hit`,
`hit_media`, `interact_with`, `sheet_ports`, `tube_incidence` / `tube_update`, the writes), and the out_typeN table
in its API block names every code. `ray_state_update` is the hardest part, but the overview gives the three words,
the ledger and the order of an event, each event lambda says what it does and its capture list shows what it reads
and writes, and the dispatch names the resolved code of every branch. The physics (the recovered cosine, the plate
rule, the pad and the transformer) needs the reader's background; the comments state what is computed and why.

What a reader still has to guess:
- Where the numbers come from: `17.98` (1 / (2π ε0 · 1 GHz)), `8.686` (20 / ln 10), `0.0477135` (c / (2π · 1 GHz):
  `Material::medium_gain` writes k0 as `fGHz / 0.0477135`, not `k0()`), the overview's 1 kHz ↔ 0.874 GHz.
- How a mass-law material's mass enters: X0 = (f / fRef) d makes fRef the frequency at which one metre of it has
  X0 = 1; the code says the formula, not that fRef carries the surface mass and `m` is only the switch.
- Tolerances without a reason: 3.8e-3 in the wedge test, 1e-4 for equal or opposing normals, 1e-6 for a parallel
  vertex ray and the VBS move, the 400 dB cap of `qd_mass_amp`, the 1e5 m vertex cut, `3.0e-20` and `1e19` in
  `qd_polbasis`.
- How `qd_polbasis` builds its V/H bases: uncommented in the function; only the `ray_mesh_interact` API block says
  `eH = ẑ × k̂`, `eV = eH × k̂`.
- The hit codes as bare numbers (1, 5, 7, 13, 15, 21, 23, 29, 31) in `exit_material`, the normal choice and the
  dispatch; their meaning is in the `out_typeN` table of the API block only.
- `& (short)0x7FFF` on the face material indices (the two input checks of `ray_state_update`): an unnamed mask, and
  no word on why bit 15 of a face index is ignored.
- `eta` with two meanings: `Material::eta` is the permittivity, `qd_refract`'s `eta` and `recover_cosine`'s `eta2`
  the Snell ratio (squared).
- `ray_mesh_interact` passes `undeviated = true` to `interact_with` for every hit; that it acts only where the near
  medium is mass-law has to be read out of `interact_with`.
- The `tridirN` write of `ray_mesh_interact` has no tube guard: it relies on an empty Armadillo matrix having a null
  `memptr()` (it does: `init_warm` sets `mem = nullptr` for zero elements, Armadillo 14.2.2).

### 2.8 Stop

Phase 2 is complete. Nothing is committed. Changed against 6307a64: `src/ray_mesh_interact.cpp`,
`tools/acoustic/README.md` (one row), `tools/acoustic/txlib.py` (two comments), the new `tools/acoustic/rowdiff.py`;
`md/acoustic_manager.md` and the two `md/cleanup_src_*.md` are the manager's and this task's. For Stephan: the
commit (the manager writes the message).
