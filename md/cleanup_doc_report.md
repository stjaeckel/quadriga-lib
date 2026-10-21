# Report — the material-model page (task 3 of 3)

Brief: `md/cleanup_doc_brief.md`. One section per phase. Phase 0 makes no edit outside this file.

## 0. Phase 0 — inventory and plan

### 0.1 Baseline

- Start: HEAD f33c8cb (notes only after e34dd2a: `md/acoustic_manager.md`, `md/cleanup_doc_brief.md`, `md/wave.md`);
  `git status`: `M md/acoustic_manager.md` only, nothing under `src/` or `tests/`.
- `cmake --build build_linux --parallel 12`: up to date.
- The three suites, one after another, 12 threads, from the repository root (outputs in `build_linux/scratch/cd0`),
  load 4.92 before, 5.09 after:

  | run | result | rowdiff against the root baseline |
  | --- | --- | --- |
  | `QD_DIAG_RAY=1 … "[wave]"` | All tests passed (163474 assertions in 178 test cases) | 141128 numbers identical, 0 beyond |
  | `QD_DIAG_RAY=1 … "[acoustic]"` | All tests passed (42741 assertions in 59 test cases) | 112451 identical, 0 beyond |
  | `QD_DIAG=1 …` (whole suite) | All tests passed (1344398 assertions in 1264 test cases) | 256768 identical, 0 beyond |

- `python3 -I tools/acoustic/check_docs.py`: all checks pass.
- Source facts confirmed: `src/ray_mesh_interact.cpp` 4725 lines, header overview 5–88, seven sections (1 at 105,
  2 at 254, 3 at 666, 4 at 917, 5 at 1477, 6 at 1506, 7 at 1994), `MaterialCols` at 257 with the 27 columns, API blocks
  `medium_gain` 2445, `interface_gain` 2501, `refractive_index` 2549, `ray_mesh_interact` 2598, `ray_state_update` 3211
  (the brief's 2445 / 2598 / 3211 are the block starts; the definitions follow at 2486 / 2741 / 3415). The page link
  lines are 2733 and 3409, the Data Formats sentence 2621.

### 0.2 The old page against the code

`html_docu/quadriga_lib_material_model.md` (738 lines, sections 1–10). "Right" = still true of the code; "stale" =
the code says otherwise (the code's statement given); "gone" = no longer exists.

| old section | right | stale (the code's statement) | gone |
| --- | --- | --- | --- |
| 1 Introduction and architecture | the two functions and their split; two passes per event; six types; 1-based indices, 0 = air; `colocation_dist` 1 mm; the relaunch offset 8 float ulp of the hit point's magnitude; the face-angle convention | "total internal reflection applies uniformly to all six types": only towards a lossless medium (`interact_with`, 814–816); `ray_mesh_interact` "charges no in-medium attenuation at all": a sheet composite carries its layer's loss and mass term (`stack_layer_phase`, `qd_mass_amp`); the state is three words of 15-bit index + flag: 14-bit index + two flags each (`word_mtl` 0x3FFF, at most 16383 materials) | — |
| 2.1 The parameter table | `fRef`, `a`–`h`, `att`, `attB`, `alpha`, `alphaB` with units and defaults | `m` "slope in dB/decade": a switch, 20 dB per decade whatever its value (`Material::m`, `mass_X0`); `res*` "permittivity resonance": on a mass-law material the plate's resonance in `X₀` (`eta()` adds the pole only where `m = 0`, 511); `coi*` a Lorentzian loss: the bending stiffness of a mass-law plate in `X₀` (582–583); the three roles (interface reflection / transmission / in-medium) do not describe `tf`, the plate rule or the finite panel | `coiA` (removed); missing: `thk`, `tfP`, `tfF`, `tfQ`, `side` (5 of the 27 columns) |
| 2.2 Constitutive laws | `ε`, `μ` power laws with `−j·17.98·σ/f` | the pole is "interface permittivity only, excluded from the propagation phase and from `refractive_index`": it is part of the medium everywhere for `m = 0` (one permittivity at a face and in propagation, 501–514; `refractive_index` includes it, API 2553–2555) | `Material::eta_resonance` |
| 2.3 In-medium loss | the `alpha` term; `mass_min_len` 1.5 mm; bonded bodies add `X₀` and telescope; a body with `m = 0` separates | the dielectric term as `8.686·ℓ/Δ` with the P.2040 attenuation distance: the code writes `8.686·k₀·\|Im √(εμ)\|` (540; the same quantity); `A = m·log10(f_r M)`, `X₀ = (f_r M)^(m/20)`: `X₀ = (f/fRef)·M`, `A = 20·log10 X₀` (`mass_X0`, `mass_TL`); the angle law is now one of two, the finite panel (`side`, Davy's σ, `mass_TL` 426–427) is the default (3.29 m); `res*`, `coi*` act in `X₀`; "the public `medium_gain` evaluates `ℓ_g = ℓ_r`" right, but it now also forms the finite panel at `c = 1` | "for a fitted slope other than 20 … 0.301 (20 − m) dB" |
| 2.4 Lumped interface loss | `att·f_r^attB` once per entry, never on exit; one `att` per glued stack | — | the coincidence Lorentzian; "clamped to 1" (`att ≥ 0` is required, so the loss is never negative) |
| 2.5 The transmission factor | `tf·f_r^tfB` clamped to [−1, 1]; `apply_tf_pair` formula (still the core of both rules) | the redistribution `R₀(1 − tf)` at every face: an ideal transformer at an open face (`tf_face`, 708–732) and a matched pad at a closed face (`closed_face`, 683–706); `tf(f)` has a resonant peak (`tfP`, `tfF`, `tfQ`, 604–619); a peaked mass-law face is a series impedance `ζ` (621–633) | `Material::apply_tf` (single-owner) |
| 2.6 Table validation | equal column lengths; `fRef`, `a`, `e` > 0; non-negative loss columns; index checks before the parallel region | the list: also `thk`, `tfF`, `side` ≥ 0; `tfQ` may be negative (MaterialCols 359–373) | — |
| 3.1–3.2 Angles, admittances, Fresnel | admittance form of P.2040 eqs. 31–32; `n₁₂` Snell ratio; index ratio | resonance "included only at the interface": everywhere; TIR "`n₁₂ sin θ ≥ 1`" for every medium: only for a lossless far medium (`Im(ε₂μ₂) = 0`) | `force_tir` "for ray-tube consistency" (now `force_tir` closes a face for the leak chain) |
| 3.3–3.4 Scalar branch, energy partition of the six types | scalar types use TE | the table (type 1/4 `1 − R_eff`, type 2/5 raw field power): every transmitted coefficient is a per-channel normal-flux amplitude (`flux_amp`, 734–754), so all four transmission types carry the same energy; the "face seen from inside a mass-law medium" rule is now the held wave (827–844: a face reached from inside a plate passes what reaches it and reflects nothing) | the `s` factor, `apply_tf_pair(R0)` on the pressure coefficient |
| 4.1 Hit classification | bits 0–5, codes 0–63 | `compact = true` default: `false` (declaration, 2651) | — (missing: bit 6, the sheet codes 65–79) |
| 4.2–4.5 Materials, direction, polarization, tube | orientation rule; Snell construction; relaunch at FBS / SBS; `min_seg`; `qd_polbasis`; `path_dirN` Snell for types 1/4; whole-tube TIR | the tube TIR test is skipped towards a lossy far medium and for one medium on both sides (3105–3109); a tube made undeviated by a vertex keeps the spine's coefficient, nothing is "re-evaluated with TIR forced" (3097–3099) | the forced re-evaluation |
| 5 EM interpretation | the built-in table (identical to `DEFAULT_MATERIAL_CSV`, `src/obj_file_read.cpp` 23–49); `irr_glass` the only `att` | — | — |
| 6 Acoustic interpretation | the wavelength-preserving frequency map (1 kHz ≈ 0.874 GHz); scalar types 3–5 | the one-parameter rigid family (`ε ≪ 1`, `μ = 1`): the two-parameter mapping is decided for every acoustic material; the porous example from Delany-Bazley (the decided rule is Miki, Komatsu for mineral wool); `res*` as Helmholtz absorbers, `coi*` as a Lorentzian, `m` as a slope; "a geometric trace-through overstates absorption below c/4t": the traced and summed layer reproduces the fluid layer (asserted in *Two-parameter mapping - …*, within 0.0008 in α) | `fRef = 0.874` "for every acoustic material" (a mass-law material's `fRef` carries its surface density) |
| 7 The state machine | VBS idea; two ledgers; walk-off `cos²` for an unbent leg; the Airy factor `S`; the parallelism tolerance 3.8e-3; the survival gate on the stronger polarization; one decision for both passes | the state encoding (RESOLVED, KNOWN-NON-PARALLEL, MASS-CROSSED; 15-bit): `cur_summed`, `cur_bent`, `prev_nonparallel`, `prev_embedded`, `buf_mass_code`, `buf_followed` (120–127); `acc_dist` "two-column array": two arrays `acc_dist_geometric`, `acc_dist_refracted`; the excess phase has three forms (exact at the recovered cosine `leg_exact_factor`, the hybrid, the walk-off; `close_leg` 4074–4123); the stacked-slab persist rule ("the first resolved cavity is the only one"): every traced body sums its own series and a layered wall is a sum of paths; the eps "typically 0.1–0.25", default now 0.15 | the pole clamp `\|1 − r r φ²\| < 10⁻²` (nothing is clamped); the macros `REPLACE_BY_VBS` (4), `REPLACE_BY_GAIN` (4), `SCALE_BY_MEDIUM` (7), `SCALE_RAY` (1), `SLAB_AIRY_FACTOR` (3), `DIST_ADD` (5), `DIST_SET` (3), `KILL` (6), `GET_EXIT_MATERIAL` (1) → `face_at_vbs`, `pass_through`, `close_leg`, `scale`, `sum_series`, `ledger_add`, `ledger_set`, `kill`, `exit_mtl` |
| 8 Dispatch tables | the resolved-type bit layout and the transmission / reflection codes 1–143 | the topology rows name `typeH` (15), `nH` (4), `SAME` (4) and `APPLY_TRANSITION` (10): `hit_code`, `n_hits`, `same_medium`, `transition`; the sheet codes (69–79, 197–207) and the kill of a ray a glued exit left in air are missing | — |
| 9 Validation | — | `test_ray_state_update.cpp`: the suite is `test_wave_*.cpp` and `test_acoustic_*.cpp` (11 files, 237 cases) | the "blind suite" |
| 10 References | P.2040-3, P.833-9, 3GPP TR 38.901 V17.0.0, Born & Wolf, Beranek & Vér, ISO 9613-1 | — | Stokes 1849 stays usable; Delany-Bazley as the porous model (now Miki / Komatsu) |

Stale internal names, against the rename table of `md/cleanup_src_report.md` 1.6: 71 hits, of which 44 are the ten
macros, 23 the dispatch names `typeH`, `nH`, `SAME`, and 4 the `Material::` functions that are now free functions or
gone (`apply_tf`, `apply_tf_pair`, `eta_resonance`, `interact_with`); the brief's 38 was another count. Of the 182 LaTeX lines, the
Fresnel pair (3.2), the slab ports (7.8) and the Airy factor (7.7) carry over after the check against the code; the
rest is rewritten from the source.

### 0.3 The absorption table

**`md/wave.md`**

| section | goes to (new page) | or dropped, why |
| --- | --- | --- |
| 0 scope, first step | the boundary `m > 0` → U4, U10; the first-step commands → D2 | the guard narrative (internal) |
| 1 the two functions | U7 (one solver for sheets and stacks; the series factor of a traced solid), U8 (the redo with state) | — |
| 2 faces, energy, solver, TIR verdict, decaying root | U6 (admittance form, flux-amplitude convention, TIR only towards a lossless medium), U7 (the recursion, nothing clamped) | the decaying-root call sites (internal; one line in D1) |
| 3 `tf` on wave faces | U6 (transformer and pad, one wall as sheet / solid / summed / followed) | — |
| 4 other lumped columns | U4, U5 (`att` once per medium entered; `alpha` over `d/\|Re c\|`; the resonance part of the medium; `μ` everywhere; `coi*` act only with `m`) | the `coiA` removal history (one sentence in U4: an old column is read and ignored) |
| 5 where things are evaluated | U3 (the two tracers, directions written), U7 (where a sheet's layer lies, the displaced host, a sheet under total reflection of its outer pair), U8 (the three propagation forms, the recovered cosine); the recovery gates → D1 | the consumer list of the recovery (internal) |
| 6 the state words | U8 (callers pass them through; what a user may read), D1 (layout table, the sheet in the previous-medium word) | — |
| 7 series, paths, verdicts | U7 (a layered wall is a sum of paths; one verdict per body), U8 (`thin_slab_threshold`, verdicts across a band) | — |
| 8a `alpha` on a tunnelling layer | — | an open convention that moves no row and needs a decision first (no new items) |
| 8 accepted limitations | U13 (marked below) | — |
| 9 the acoustic attachment points | D1 (where `m > 0` acts, as a list of functions) | the old macro names |
| 10 rules for changes | D2 | the line about this page ("out of date"), updated in phase 2 |
| 11 test instruments | D2 (helpers, runners, independent references, what-if build, traps) | the scratch template paths (rigs) |

**`md/acoustic.md`** (sections 1, 2, 5, 7, 9; the decided rules of 10)

| section | goes to | or dropped, why |
| --- | --- | --- |
| 1 the mapping | U10 (two-parameter only; the one-parameter form in one sentence; the frequency scale; a thin wave-medium layer is a limp leaf) | the measurement narrative |
| 2.1 the mass term (with the finite panel, `res*`, `coi*` in `X₀`, the cosine in air) | U5 (formulas), U10 (what each column does) | the M10 history (`m` was a fitted slope) |
| 2.2 bonded bodies are one plate (thickness code, one side per run) | U5 (the plate's `X₀` sum and its telescoping), U12 (one `side` per run), D1 (how the second body learns `X_run`, the thickness code) | — |
| 2.3 the plate rule | U6 (the rule `aτ/(a + τ − aτ)`, the absorption of an outer face, where it is paid), U7 (a mass-law layer in the solver) | the bb336dc comparison (history) |
| 2.4 the held wave | U6 | — |
| 2.5 nothing reflected back into a mass-law body | U6, U7 | — |
| 2.6 the room-side absorption (`tf` law, peak, `ζ`) | U5 (`tf(f)`), U6 (the peaked face), U10 (`tf` is a random-incidence coefficient), U12 (rules) | the M5 / M6 measurement narratives |
| 2.7 the columns on a mass-law material; the 13 fits | U10 (quantity → column, condensed); the three examples carry their rows | the 13-fit table as a whole: the parameter project publishes materials (question Q3) |
| 2.8 leaves as wave-medium sheets | U10 (the leaf route), U12 (its rules), U13 (its limits) | the scratch rig, the M9 numbers that are not limits |
| 5 what stays out of reach | U13 "Out of scope" (scattering and diffraction, flanking, the ground wave, level dependence, placement values, double leaves only as fitted rows) | the expressibility bookkeeping of the rest |
| 7 accepted limitations | U13 (marked below) | — |
| 9 the expressibility test | U10 "Deriving columns from data" (reference types, Paris integral 0°–90°, the 0–90° transmission fit at the lab side, the tolerance as a guideline: about ±3 dB, ±0.05) | the classes, the procedure and the reproducibility table (an internal acceptance test) |
| 10 decisions | the rules that convert data to columns → U10: the tiered rule for a `tf` reference (hard face as measured; an absorber of known build converted to random incidence; mineral wool without build data by the edge term; the rest not a `tf`); porous columns from Miki, Komatsu for mineral wool, over the whole band; `f₀` measured, else the constant 60 empty, 50 filled; the transmission fit on the 0°–90° average at the laboratory side; `side` 3.29 m where absent | decision history, round history, the built-change records |

**Every rule of use, by name** (one line each on the page, U12):

1. Local coordinates: build scenes near the origin; mm-wave through `ray_progress` within about 100 m; never raw
   georeferenced coordinates (the relaunch offset grows with the coordinate). `md/wave.md` 8, `md/acoustic.md` 7.
2. One `att` per bonded run or glued stack (the loss is paid at every entry). `md/acoustic.md` 2.6, A5; old page 2.4.
3. One `side` per bonded run (each body pays its increment at its own side). `md/acoustic.md` 2.2.
4. `tf` together with `m` (a rigid face without `tf` absorbs nothing and its plate passes nothing). 2.3, 2.6, 7.
5. A rigid body carries the mass column (a rigid permittivity without `m` is a wave medium: a sheet of it tunnels).
   2.6; `md/wave.md` 8.
6. A rigid body is lossless (a loss column opens its faces: a 200 mm wall then reflects 1.000000 from 10° on). 2.6.
7. Acoustic materials in the two-parameter mapping (the one-parameter form gets the phase at a rigid half-space wrong
   and over-absorbs on a porous layer). 1; decision of 2026-10-21.
8. Acoustic runs use the scalar types 3–5 (the EM HH rows of a mass-law material are formal, up to 3.9 dB from the
   traced twin). 7.
9. Leaves: a wave-medium sheet (`m = 0`), its loss as a constant ratio on `μ` (`g = e·rr/17.98`, `h = 1`, `fRef = 1`);
   the cavity a meshed box of near-air (`μ` 1.000001: a material identical to air is air); the leaf glued on the box's
   front face, the wall (a mass-law body) on its back face; a double leaf two leaves on one box; a double glazing unit
   one fitted mass-law sheet or two leaves on a box, never two meshed mass-law sheets; a porous fill a traced body at
   the back of the box; behind a fill the wall takes `tf` 0; a box, never a meshed gap (1126 to more than 8000 events).
   2.8.
10. A porous layer between a box and a wall is a traced body, not a sheet (the intersector reports two of three
    co-located faces). 2.8, A21.
11. A source inside a medium: seed the bent flag and use type 5 (an unknown bend leaves a summed series off by 0.13 to
    3.57 and the mass cosine off by up to 3.00 dB). `md/wave.md` 8, `md/acoustic.md` 7.
12. Glue co-located faces exactly: two faces closer than 1 mm along the ray are one event; a pair 100 µm apart is not
    one event at 85°. `md/wave.md` 8; API block.
13. Avoid overlapping geometry (API block); at most two co-located sheets form a stack.
14. Direct callers of `ray_state_update`: pass the state words through unchanged; one `acc_dist_geometric` per run,
    one `acc_dist_refracted` per frequency; hand the base frequency's verdicts to the other frequencies; with types 2
    or 5 hand in `origN` / `destN` on the geometry-defining call. API block; `md/wave.md` 8.
15. `min_gain_dB` near a wall: inside a plate the running gain lies up to `−10·log10 a` below the level the path
    reaches (10 dB at `tf` 0.1), so a floor set within that margin can cut a path. `md/acoustic.md` 7.
16. A leaf box's series is summed with `thin_slab_threshold` up to 0.15 (the default). 2.8.

**Every accepted limitation, kept or dropped.** Kept ones get one line on the page with the size; every size is
re-confirmed in phase 1 from a printed row of the named case or re-measured with a scratch probe, else dropped.

`md/wave.md` 8:

| limitation | verdict |
| --- | --- |
| `tf < 0` forces reflection per channel, also at Brewster (0.731 / 0.500) | dropped: `tf < 0` on an EM face is no typical use |
| a transformed face is `(1 − tf)·R₀` only where its sides are in phase (+1.0 dB at `tf` 0.5, +5.3 dB at 0.9 on `eps 1.50 − 0.40j`) | kept: it changes what `tf` means on a lossy wave face |
| `tf = +1` closes a coherent sheet's reflect port only in phase (1.4e-2) | dropped: fringe |
| a forced face steps where its admittance ratio has modulus one | dropped: a single angle |
| the square-root cusp across a sheet's critical angle | dropped: physics, no defect |
| exactly at a lossless sheet's critical angle the solver declines | dropped: a single point |
| a rigid permittivity without `m` is a wave medium | kept as rule 5 |
| entry and exit of a lossy solid differ by `sec²(arg η)`; next to a lossy medium the two ports are no energy sum (TM 1.178); sums of path powers are no energy statement | kept as one line: the energy convention next to a lossy medium |
| a followed series can diverge (round trip 1.19) | dropped: `ray_progress` sums such a leg (round trip ≥ threshold, always above one) |
| `resS < 0` is a gain medium (2.11), nothing validates it | kept |
| `mtl_prev` one level deep; air beyond every glued exit | kept |
| a sheet between two bodies depends on the intersector's tie | kept as rule 10 plus one line (what remains loses energy only: R 0.154 for 0.994) |
| the followed mark and the stack code share the buffer word with a deferred medium (+0.08 to −0.54 dB, 0.78 dB) | dropped: a nested object inside a layer |
| the current word is corrected in one direction only | dropped: internal |
| an exit out of a wave launched inside a lossy solid takes the real-angle map (0.2°) | dropped: fringe |
| the refracting tracer's segment at a closed face for invariants 1 to 1.045; the direction step at an invariant of one | dropped: reached only from a host entered from a denser medium |
| a type-2/5 caller that does not hand in the geometry | kept as rule 14 |
| the relaunch offset: a body thinner than the nudge is skipped; rays within about 3° of grazing lost in float or AVX2; 3.8° of EM phase at 28 GHz at 1 km | kept (with rule 1) |
| the bent flag, verdict and geometry are the base frequency's | kept: a band run is typical |
| a ray launched inside a host (0.13 to 3.57) | kept as rule 11 |
| the hybrid factor's rounded constants (2.5e-5 dB); the two spellings in `interact_with` (6e-14) | dropped: negligible, internal |
| a co-located pair judged along the ray (100 µm at 85°) | kept as rule 12 |
| what lies between a hit point and the relaunch point is free space (0.013°) | dropped: negligible |
| the tube follows the spine; a tube with an edge past a critical angle passes undeviated | kept (users with tubes and type 2) |
| the refracting tracer at a deferred body (10.27° against 12.62°) | dropped: overlapping geometry is to be avoided |
| a transparent transition between two indices of one material sets the fix flag | dropped: fringe |
| the mid-plane of a one-medium sheet; the reflect port's reference plane | moved to U7 as a statement (where a sheet's layer lies), not a limit |
| the first path through a layered wall is not the wall (−4.88 dB); a run that stops early keeps the difference (+1.37 dB after 20 interactions) | kept |
| a sheet that does not fit its host is capped; a stack thicker than its host passes more (+0.24 to +1.81 dB) | dropped: fringe |
| a sheet inside a host and its traced twin differ by the host's `att` | kept: users choose between sheet and solid |
| an opaque layer transmits to underflow; a "sum" verdict on a non-parallel leg; untested path sums of three solids | dropped: fringe, internal |

`md/acoustic.md` 7 and 2.8 limits:

| limitation | verdict |
| --- | --- |
| the mass term steps at an invariant of one (7.66 / 22.60 dB) | dropped: never reached from air through a parallel face |
| a ray the state does not know to be bent keeps its segment's cosine (0.21 / 0.96 / 3.00 dB) | kept as rule 11 |
| behind a lossy wedge the cosine in air is the real-angle one (0.002–0.096 dB) | dropped: small, fringe |
| the thickness code is quantized (0.0038 dB) and ends at 100 m | dropped: negligible |
| the closed items (mixed slopes; far from the origin; the false −26 dB in float) | dropped: history (rule 1 stays) |
| the state is one level deep: a plate of three or more bodies within ±3.02 dB, a wall with a stack inside up to 3.01 dB | kept (plaster \| brick \| plaster is typical) |
| the plate near its knee exceeds the dB sum (+0.44 dB) | dropped: small, near the knee |
| a plate passes at most what its face absorbs | moved to U6 as a property of the plate rule |
| the first face of a leak is flat; a non-rigid mass-law layer becoming closed steps | dropped: non-rigid mass-law layers are no use case |
| a rigid face without `tf` absorbs nothing | kept as rule 4 |
| a non-rigid mass-law layer beside a lossy coherent layer depends on the order (0.1–0.2 dB) | dropped: no use case |
| the two sides of a lossy lining on a rigid wall differ (0.07–1.38 dB dense, 0.52–2.70 dB porous) | kept: an absorber on a wall is typical |
| an event can pass more than one; `min_gain_dB` inside a plate | kept as rule 15 |
| the HH rows of a mass-law material | kept as rule 8 |
| a held chain has no reference but its solid twin; `ray_mesh_interact`'s own lining in a lossy host | dropped: internal (U1 says the redo with state is the result) |
| a rigid face's absorption is flat in angle, except a peaked face's | moved to U10 as a statement |
| a followed series behind a lining rarer than air diverges on a peaked face; R + T 1.0005 at one angle; tracer 4 under-transmits out of a summed lining (one-parameter mapping) | dropped: one-parameter mapping or a single angle |
| wave media (leaves) keep the infinite plate at grazing | kept |
| a peak on a non-rigid mass-law body | dropped: no use case |
| a leaf has no bending stiffness (coincidence needs the mass column) | kept |
| a meshed gap in place of the box; a fill directly behind a leaf | kept as rule 9 |
| the followed series of a lossy body between strong reflectors diverges (1.145) | dropped: `ray_progress` sums it |
| a leaf past its critical angle tunnels; the thin-layer result needs a leaf thin against the wavelength in it | kept as one line |
| a curtain's relaxation in `μ` is reached only on the random-incidence average | kept |
| the summed box's residual at the resonance (6.5e-5 in R) | dropped: negligible |

### 0.4 The outline

Title (the H1): **The quadriga-lib Material Model and Ray-Surface Physics**; a one-line companion note, then the
contents. Expected lines per section in brackets.

**User part** (about 760 lines)

- **1 What the model is** [30]: homogeneous isotropic media on triangle faces; the five public functions and what each
  applies (`medium_gain`, `interface_gain`, `refractive_index`, `ray_mesh_interact`, `ray_state_update`); the tracers
  that call them (`ray_progress`, `calc_diffraction_gain`); Hz at the API, GHz in the columns; the source is the
  normative reference.
- **2 Scene and mesh conventions** [35]: front side by vertex order; a solid is a closed body, a sheet a face with
  `thk`; co-located faces within 1 mm along the ray (media-to-media, glued sheet, stack); air on every side the mesh
  does not name; the relaunch offset and local coordinates.
- **3 Interaction types and the two passes** [30]: the six types in a table; reflect pass and transmit pass; types 1/4
  keep the direction while `path_dirN` reports the Snell direction, types 2/5 bend; total reflection only towards a
  lossless medium; the ray tube.
- **4 The material columns** [50]: one table of all 27 columns (column, unit, default, meaning, media: wave / mass-law /
  both); `m > 0` as the acoustic switch; the checks `MaterialCols` makes; older columns (`coiA`) are read and ignored.
- **5 The formulas as implemented** [90]: `ε(f)`, `μ(f)`, the Lorentz pole (wave media only); `n`, the admittance; the
  in-medium loss `8.686·k₀·|Im n|` plus `alpha`; `att`; `tf(f)` with its peak; the mass term (`X₀ = (f/fRef)·M`,
  `TL` of the infinite plate, the finite panel with Davy's `σ`, `res*` and `coi*` as impedances in `X₀`, the cosine in
  air); bonded bodies (`X₀` add, the telescoped charge). Each formula with the function it comes from.
- **6 Interface physics and energy** [80]: the Fresnel pair in admittance form (P.2040 eqs. 31, 32); transmitted
  coefficients as per-channel normal-flux amplitudes (`1 − R` per channel from a lossless medium); the total-reflection
  verdict; `tf` as transformer (open face) and pad (closed face); the peaked face `ζ`; the held wave in a plate (no
  bend in, refracted as from air out into a wave medium, nothing reflected back); the plate rule and the absorption of
  an outer face; what an event's two ports mean next to a lossy medium.
- **7 Sheets, stacks and traced solids** [60]: the sheet composite (one recursion, both ports, nothing clamped); where a
  sheet's layer lies and the displaced host; a mass-law layer in a stack; a traced solid and its series `S`; one verdict
  per body; a layered wall as a sum of paths; when to use a sheet and when a solid.
- **8 What `ray_state_update` adds** [50]: the medium a ray is in; the incidence at the virtual hit point; the in-medium
  loss and phase of the leg (exact at the recovered cosine, the hybrid, the walk-off form); the series sum and `eps` /
  `thin_slab_threshold`; band runs (the base frequency's geometry and verdicts); the state words and ledgers from a
  caller's view.
- **9 Electromagnetic use** [50]: the built-in ITU-R P.2040-3 table (`obj_file_read` without a material file; names by
  base name), typical choices of type; worked examples (0.5).
- **10 Acoustic use** [80]: the mapping (two-parameter; the one-parameter form in one sentence; 1 kHz ≈ 0.874 GHz);
  wave media (porous layers, fluids, leaves) and mass-law materials (rigid bodies: the faces from `tf`, the
  transmission from the mass term and the plate rule); the column roles per acoustic quantity; the leaf route on a box
  of air; deriving columns from data (the rules of 0.3).
- **11 Acoustic worked examples** [75]: the wall, the window, the absorber (0.5), each with its columns, the measured
  library values and the deviation from the measured reference.
- **12 Rules of use** [40]: the 16 rules of 0.3, one line each with the reason.
- **13 Limits** [45]: the kept limitations of 0.3, one line each with its size; out of scope (scattering and edge
  diffraction, flanking, the ground wave, level dependence).
- **14 References** [35]: P.2040-3 (with edition), P.833-9, 3GPP TR 38.901 V17.0.0, Born & Wolf, Davy 2009 (both
  papers), Beranek & Vér, Miki 1990, Komatsu 2008, London 1949 or Cremer for the mass law, the measured references of
  the examples (PCA / Litvin & Belliston 1978, Mahn et al. 2023 RR-331, EN 12758 and Quirt 1981, Dupont et al. 2023),
  Paris 1928 for the random-incidence integral — full citations from `references/index.csv`.

**Developer part** (about 100 lines)

- **D1 A map of `src/ray_mesh_interact.cpp`** [55]: the header overview and what it states; the seven sections, one
  line each with the key functions; the public functions; the per-event call sequence of `ray_mesh_interact`
  (`face_plane` / `ray_plane_hit` → `classify_hit` → `hit_media` → `interact_with` → `sheet_ports` →
  direction, tube, relaunch → `qd_polbasis`) and of `ray_state_update` (`decode_words` → `vbs_geometry` → per pass the
  event switch → `transition` / `sheet_event` → `face_at_vbs` → `sum_series` (`qd_slab_resolve`) → `close_leg` →
  `relaunch_ledger_offset` → the words); the state words (layout table) and the ledger; where `m > 0` acts; where the
  hit-code and resolved-code tables are (the two API blocks).
- **D2 How a change is checked** [45]: the first-step commands; the test files and tags (`[wave]`, `[acoustic]`, file
  tags with `-#`); `QD_DIAG_RAY` / `QD_DIAG`; the untracked baselines and `tools/acoustic/rowdiff.py`; the stored twins
  (`tests/data/wave_twins.tsv`, `QD_TWIN_WRITE`); the order of checks (EM and wave rows first, then the acoustic
  rows, then energy: both ports at most one seen from a lossless medium); the what-if build; the rules (measure, do not
  compute by hand; a wider gate over an exclusion; one thread for single-ray fixtures; 12 threads and one suite at a
  time on a shared machine); the traps (run from the repository root; `pkill -f`; a case name with a comma).

### 0.5 The worked examples

Instruments: acoustics through `tools/acoustic/band_probe.cpp` (`ray_progress`, scalar, one ray per angle, a band per
line, T and R summed over every path that ends outside), on Gauss-Legendre angles over 0°–90° (random incidence,
weight `cos θ sin θ`; absorption as the Paris integral), 6 frequencies per third octave, octaves 31.5 Hz–8 kHz; each
construction once as a sheet and once as a traced solid (they must agree within the asserted 0.0042 dB). EM through a
scratch copy of `band_probe` with `scalar_mode = false` (EM types; `band_probe` itself is scalar only). Single-ray
probes on one thread. The report states every number with the probe input.

| example | source data | rule to columns | columns | measured | expected accuracy |
| --- | --- | --- | --- | --- | --- |
| EM 1: concrete wall | `itu_concrete` of the built-in table (P.2040-3 Table 3) | none (table row; `fRef` 1 GHz) | `a` 5.24, `c` 0.0462, `d` 0.7822; a traced solid 200 mm | R and T at 0° and 45° (TE, TM) at 3.5 and 28 GHz, types 0/1 and 0/2 | on the P.2040 single-slab closed form (an independent reference in the probe, as `qd_slab_ref`) to rounding; against the 3GPP TR 38.901 Table 7.4.3-1 concrete loss `5 + 4f` dB at normal incidence the difference is stated (a model fitted to measurements) |
| EM 2: window glass | `itu_glass`; `irr_glass` (3GPP, `att` 23 dB) | none | 6 mm as a sheet (`thk` 0.006) and as a traced solid | the same, plus `irr_glass` | sheet = solid; against 3GPP `2 + 0.2f` (standard glass) and `23 + 0.3f` (IRR glass) stated |
| Acoustic 1: single-leaf wall | cast concrete 152 mm, 347 kg/m², measured octaves 125 Hz–4 kHz (PCA RD066, Litvin & Belliston 1978, test 76-66; `tl_refs.json` "concrete 152"); independent: Mahn et al. 2023 (NRC RR-331) CON150, 150 mm, 374 kg/m², 41 / 42 / 51 / 59 / 65 / 74 dB; room-side absorption from the review's compilation row for concrete | the transmission fit on the 0°–90° average at the laboratory side (`txfit.py`; the decided row of `md/acoustic.md` 2.7, worst 0.57 dB); for contrast the bare mass law from the density (`fRef = (f_GHz/f_Hz)·ρ₀c/(πρ)`, `m` 1, `side` 3.29, no `res*`, `coi*`, `att`) | `thk` 0.152, `fRef` 1.9663e-4, `tf` 2.549e-4 at `fRef` with `tfB` 0.52, `att` 7.01 / −0.09, `coiF` 6925 Hz / `coiQ` 1.19, `resF` 78 Hz / `resQ` 1.44 / `resS` < 0, `side` 3.523; `ε`, `μ` from the two-parameter mapping of concrete (`ρ`, `c` from the review) | TL per octave at random incidence and at 0°, at `side` 3.523 and 3.29; the room-side absorption per octave | the fitted row within its 0.57 dB of RD066 (the library on the formula copy within 0.0042 dB); against CON150 the laboratory spread (1–3 dB per band, other density); the bare mass law several dB off above coincidence, stated |
| Acoustic 2: window | float glass 4 mm, 10 kg/m², EN 12758 generic values via Pilkington (`tl_refs.json` "glass 4 mm"); independent: Quirt 1981 (NRC), 4 mm float, 10.4 kg/m², 23.6 / 25.6 / 29.7 / 33.6 / 32.6 / 28.0 dB; absorption: the class law of the round-1 compilation rows | the same fit at the laboratory side 1.349 m (decided row, worst 0.85 dB) | `thk` 0.004, `fRef` 4.0562e-5, `tf` 5.661 at `fRef` with `tfB` −0.37 (clamped to [−1, 1]), `att` 0.05 / 1.35, `coiF` 2967 Hz / `coiQ` 284, `resF` 138 Hz / `resQ` 2.33 / `resS` > 0, `side` 1.349; `ε`, `μ` of glass | TL per octave at random incidence and at 0°, at 1.349 and 3.29 m; the room-side absorption (flat in angle: the Paris value equals `tf`) | within 0.85 dB of EN 12758; against Quirt the spread between the two references (up to 6 dB at 125 Hz, small specimen) stated |
| Acoustic 3: porous absorber on a rigid wall | glass wool 50 mm, σ 13.1 kPa·s/m² measured, random-incidence absorption by sound-field synthesis 250 Hz–4 kHz (0.407 / 0.720 / 0.887 / 0.937 / 0.940; 125 Hz 0.130 from their JCA model), Dupont et al. 2023, Acta Acustica 7, 26 | porous columns from Komatsu (mineral wool) at the measured σ over 31.5 Hz–8 kHz, `ε = n·z`, `μ = n/z`, a power-law fit of `a`–`h` (`fRef` 1 kHz mapped); the wall a two-parameter concrete mass-law body with `tf` 0 (the convention behind a fill) | fitted in phase 1 (8 columns) | `α₀` and the Paris absorption per octave 125 Hz–4 kHz (and 31.5 / 63 Hz / 8 kHz as model values), the layer as a traced solid and as a sheet glued on the wall | the library on the fluid layer of its columns within 0.0008 (asserted); the deviation from the measured curve is the model's plus the power-law fit's, read against the ±0.05 guideline and stated whatever it is |

None of the three needs a pending decision: the absorber takes the measured σ (no chamber conversion) and the wall
behind it `tf` 0; the two mass-law rows are decided fits. One caveat for the stop (Q1): both mass-law rows carry a
fitted `coiQ`, whose meaning the pending proposal on the bending loss would change.

### 0.6 The API-block correction list

All inside `/*!MD … MD!*/`, comment-only.

| block, line | old text | new text | reason |
| --- | --- | --- | --- |
| `ray_mesh_interact` 2620 | "In-medium attenuation is not applied here ([[ray_state_update]] does), and no spreading loss is applied." | "The loss inside a traced solid is not applied here ([[ray_state_update]] applies it where the leg closes); a sheet's composite includes the loss and the mass term of its layer. No spreading loss is applied." | the composite carries `stack_layer_phase` (complex index, `alpha`) and `qd_mass_amp` (1170–1180, 931–953) |
| `ray_mesh_interact` 2621 | "For a detailed description of the material model see <a …formats.html>Data Formats</a>" | "The material model, its columns, the rules of use and the limits: the companion page under See also; the CSV format of the material table: <a …formats.html>Data Formats</a>" | the page is the description; Data Formats keeps the file format (it contradicts the code in places, 0.7; question Q2 whether to keep the pointer) |
| `ray_mesh_interact` 2729 (`path_dirN`) | "Out of a mass-law medium (`m > 0`) there is no bend with any type: the wave is held undeviated at every face." | "No ray bends into a mass-law medium (`m > 0`). The wave in it is held: into air or another mass-law body it keeps its direction, into a medium without the mass column it is refracted as a wave from air." | `interact_with` 827–844 (`n2_ratio = 1/(ε₂μ₂)`, `bend = !(B.m > 0)`) and 895–896; `md/acoustic.md` 2.4; to be confirmed with a probe before the edit |
| `ray_mesh_interact` 2733, `ray_state_update` 3409 | link text "The quadriga-lib Material Model and Ray-State Machine" | "The quadriga-lib Material Model and Ray-Surface Physics" (the new H1) | the page title changes (Q4) |
| `ray_state_update` 3294 (state table, `mtl_ind_prev` bit 14) | "the sheet in this word lies inside the current medium" | "a sheet in this word lies inside the current medium; with a solid, the face into the current medium was evaluated at a real-invariant cosine" | `transition` writes `prev_embedded` for a solid (4492); `recover_in_current` reads it (3832) |
| `medium_gain` 2454 | "not per metre" | "not per meter" | American spelling |
| `medium_gain`, `interface_gain`, `refractive_index` See also | (no page link) | add "- <a target="_blank" rel="noopener noreferrer" href="quadriga_lib_material_model.md">The quadriga-lib Material Model and Ray-Surface Physics</a> (the formulas and the columns)" | the three helpers apply the formulas the page states; recommended for all three |

Checked and found right: the `medium_gain` formula text (2449–2461, at `c = 1` with the finite panel), the
`interface_gain` and `refractive_index` blocks, the `out_typeN` and `resolved_typeN` code tables (the sheet codes
69–79 and 197–207 against `classify_hit`, `sheet_event`), the `eps` and `verdict_inN` texts, `compact` (default
`false` in the declaration and the text), the `xprmatN` basis text.

### 0.7 `format_materials.md` against the code (report only)

1. The parameter table lacks `thk`, `tf`, `tfB`, `tfP`, `tfF`, `tfQ` (6 of the 27 columns `MaterialCols` reads).
2. `resF`, `resQ`, `resS` "permittivity resonance": on a mass-law material they are the plate's resonance in `X₀`
   (`eta()` adds the pole only where `m = 0`); the ε(f) row writes the pole without that gate.
3. The pole is "active only when `resF > 0` and `resS ≠ 0`": the code also needs `resQ > 0`.
4. The "transmission partition" (light→dense subtracts `1 − |R|²`, dense→light passes through, types 1/4 gated, type 2
   raw Fresnel): every transmitted coefficient is a flux amplitude and the four transmission types carry the same
   energy; there is no such gate.
5. The in-medium loss "`8.686·L/Δ` … plus `mass(f, L)`": the mass term is formed from the normal thickness and the
   cosine in air, not the path, and at normal incidence it is `20·log10 X₀` only for `side = 0` (default 3.29).
6. "There is no scalar refraction mode": type 5 is scalar refraction.
7. "does not model phase": every path carries the complex coefficient (the two-parameter mapping's phase is asserted).
8. "`fRef = 0.875 GHz ≡ 1 kHz`, fixed for every acoustic material": a mass-law material's `fRef` carries its surface
   density (`X₀ = (f/fRef)·M`).
9. The frequency map `874,636` (342.77 m/s) and 1 kHz ≡ 0.875 GHz: the tools and fixtures use `8.7403e5` (343 m/s).
10. Rigid reflectors as a one-parameter medium (`εr ≪ 1`, `μ = 1`), absorption from `a`: a rigid face's absorption is
    `tf` (the pad, `R = 1 − tf`); the rigid-surface table `a = ((1 − √(1−α))/(1 + √(1−α)))²` sets nothing the plate
    rule uses.
11. "on the transmission path the dense→light gate makes the air→wall crossing pass-through, so isolation comes from
    `att` (and `m` for the slope)": the transmission of a mass-law wall is the plate rule of its faces and its mass term.
12. Layered rigid-on-rigid: "isolation is the sum of each entered layer's `att` plus the in-medium losses of the two
    layers": bonded mass-law bodies are one plate (`X₀` add) under the plate rule.
13. "The wave cascade … is traced as separate ray segments": a sheet (`thk`) is summed in one event; a traced body's
    series is summed in closed form where it is strong enough.
14. Mounted porous absorbers: "a phaseless ray tracer cannot resolve it … over-states the low-frequency absorption",
    "clamp the per-interaction reflected power": the traced, summed layer is the coherent fluid layer and nothing is
    clamped; the "surface-impedance boundary" described is not in the code.
15. Resonant absorbers "`resF/resQ/resS` for the peak, on top of a baseline reflector": on a mass-law face the
    absorption peak is `tfP`, `tfF`, `tfQ` (a series impedance); `res*` there shapes the transmission.
16. "the magneto-dielectric TM branch is out of scope; μ is intended for TE / scalar use": `μ` is carried in TM too
    (`flux_amp`, the TM Fresnel pair in admittances).
17. The acoustic default `concrete` row (`a` 2.4e-9, `m` 20, no `tf`, no `fRef`, no `thk`): with `m` and no `tf` the
    face absorbs nothing and the plate passes nothing; its mass level then rests on `fRef` 1 GHz.
18. The section description lists `resF`, `resQ`, `resS` under "interface reflection" for every material and `coi…`
    under the transmission partition: see 2 and 4.

### 0.8 The checks to add to `tools/acoustic/check_docs.py`

A third document entry, `html_docu/quadriga_lib_material_model.md`, with its own checks (the existing ones keep their
two files):

- **columns**: every name `MaterialCols` resolves (parsed from `resolve("…")` in `src/ray_mesh_interact.cpp`) appears
  as a row of the page's column table (a table whose header starts with "column").
- **labels**: no internal label on the page: item ids (`\b[AMNPDW]\d+[a-z′]?\b` outside math and code spans, `D2-b`),
  "round 1", `md/`, "report", "brief", "rig", `build_linux/scratch`.
- **tables** and **spelling**: the existing rules applied to the page.
- **math**: per paragraph (blank-line separated, code fences skipped) an even number of unescaped `$` after removing
  `$$`, and `$$` balanced per file; no `\newcommand`.
- **anchors**: every `](#anchor)` resolves to a heading slug of the page (GitHub rule: lower case, spaces to `-`,
  punctuation dropped).
- README line in `tools/acoustic/README.md` (the table row of `check_docs.py`) names the page.

### 0.9 Side findings (one line each, no items)

- `md/acoustic.md` 2.7: the delimiter row of the 13-fit table is written with `−` (U+2212) instead of `-`, so the
  table does not render on GitHub (`check_docs.py` counts only `|`).
- `interface_gain` and `ray_mesh_interact` test `center_frequency <= 0` without `isfinite` (a NaN frequency passes);
  `medium_gain`, `refractive_index` and `ray_state_update` test it.
- The brief names `map_to_itu_materials` with `obj_file_read`; it is an option of `mitsuba_xml_file_write` (export to
  Sionna presets); the built-in table is what `obj_file_read` uses without a material file.
- `md/wave.md` 10: the line about this page is 496, not 495.

### 0.10 Stop — for Stephan's OK

On the outline (0.4), the absorption table (0.3), the example plan (0.5) and the correction list (0.6). Open choices:

- **Q1** The two mass-law examples use the decided fits, which carry a fitted `coiQ` (284 on the pane, 1.19 on the
  wall); the pending proposal on the bending loss as a resistance would change what `coiQ` means. Keep them (they are
  today's machine; re-measured if the proposal is adopted), or drop `coi*` from the wall and show the bare mass law
  only? Recommended: keep.
- **Q2** The `ray_mesh_interact` block's Data Formats pointer: keep it for the CSV format (recommended) or drop it
  until `format_materials.md` is corrected (0.7).
- **Q3** The 13 fitted constructions of `md/acoustic.md` 2.7 as a table on the page? Recommended: no; the parameter
  project publishes materials, and the examples carry their rows.
- **Q4** The title "The quadriga-lib Material Model and Ray-Surface Physics" (the API link texts follow).
- **Q5** The EM references: the P.2040 slab closed form and the 3GPP TR 38.901 penetration losses. Recommended as in
  0.5.
- **Q6** Limit sizes on the page only where phase 1 re-confirms them from a printed row or a probe; a size that cannot
  be confirmed is re-measured or the line dropped. Recommended.

## 1. Phase 1 — the page

Decisions of 2026-10-21 (Stephan) on 0.10: Q1 keep the decided fit rows with `coiQ`; **if the meaning of `coiQ` changes,
the two mass-law examples (1.2, 1.3) and the 13-row table (1.5) are re-measured.** Q2 the Data Formats pointer kept for
the CSV format, `format_materials.md` untouched (a separate session). Q3 the 13 rows on the page as preliminary
starting rows, re-evaluated with `txlib.py` (no refit); the U+2212 delimiter row of `md/acoustic.md` 2.7 fixed in phase
2. Q4 the title. Q5 both EM references, the 3GPP differences stated, never asserted. Q6 every limit size from today's
library. Q7 a user part of about 600 lines.

Rig: `build_linux/scratch/cd1` (gone after `make clean`; every number the page relies on is below with its input).
- `build.sh SRC OUT` builds a probe against `build_linux/libquadriga.a` (as `tools/acoustic/build_band_probe.sh`).
- `band_probe2.cpp`: `tools/acoustic/band_probe.cpp` with `attB` appended as a 25th material column (the tool has no
  `attB`, and both decided rows carry one; side finding 1.9).
- `em_probe.cpp`: the same probe in EM mode (`scalar_mode = false`, types 1 / 2, frequencies in GHz, VV and HH ports;
  VV is TE, HH is TM for its plane of incidence x–y). `em_shift.cpp`: with `SHIFT` metres added to every x.
- `ac_lib.py` (random-incidence TL over `NA` midpoint angles 0°–90° with weight `cos θ sin θ`, `NS` frequencies per
  third, octave = mean of T over its frequencies; TL at 0°; absorption `1 − |R|²` as the Paris integral), `eval13.py`,
  `table13.py`, `em_run.py`, `run_walls.py`, `dev_walls.py`, `por_run.py`, `por_fit2.py`, `lim_run.py`, `shift_run.py`
  and their `.out` / `.json`.
- All runs one thread (`OMP_NUM_THREADS=1`), `ray_progress` with `thin_slab_threshold` 0.15, 255 interactions, floor
  −300 dB (−250 in the meshed-gap run).

### 1.1 EM examples (`em_run.py`, `em_run.out`)

`itu_concrete` 200 mm as a traced solid; `itu_glass` and `irr_glass` 6 mm as a sheet (`thk` 0.006) and as a traced
solid; built-in columns, `fRef` 1 GHz; in air; 0° and 45°; types 1 and 2 (the reflect pass is type 0). Reference: the
single slab in air written in the script from the admittance-form Fresnel pair with complex angles (`(1 − r²)φ/(1 −
r²φ²)`, `r(1 − φ²)/(1 − r²φ²)`), and for `irr_glass` its `att` paid once on the transmit port and once on the internal
path of the reflect port (`r − √g·(1 − r²)·r·φ²/(1 − r²φ²)`).

| material | f GHz | T 0° (dB) | T 45° TE / TM (dB) | \|R\|² 0° | \|R\|² 45° TE / TM | 3GPP loss (dB) | library − 3GPP (dB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `itu_concrete` 200 mm | 3.5 | −19.03 | −21.08 / −19.07 | 0.15701 | 0.26029 / 0.06820 | 19.0 | +0.03 |
| `itu_concrete` 200 mm | 28 | −90.90 | −96.65 / −94.64 | 0.15446 | 0.26087 / 0.06805 | 117.0 | −26.10 |
| `itu_glass` 6 mm | 3.5 | −2.84 | −4.62 / −1.31 | 0.46380 | 0.63895 / 0.24196 | 2.7 | +0.14 |
| `itu_glass` 6 mm | 28 | −2.54 | −5.05 / −2.25 | 0.19881 | 0.50952 / 0.17190 | 7.6 | −5.06 |
| `irr_glass` 6 mm | 3.5 | −25.82 | −27.58 / −24.30 | 0.19762 | 0.31454 / 0.09480 | 24.1 | +1.72 |
| `irr_glass` 6 mm | 28 | −25.27 | −27.86 / −24.96 | 0.18108 | 0.30942 / 0.09236 | 31.4 | −6.13 |

Every row on the closed form within 0.001 dB in T and 1e-5 in \|R\|² (float), both tracers, sheet and traced solid
alike. 3GPP TR 38.901 V17.0.0 Table 7.4.3-1: concrete `5 + 4f`, standard multi-pane glass `2 + 0.2f`, IRR glass
`23 + 0.3f` dB (f in GHz): a model of measured building penetration with no thickness, stated only.

### 1.2 The single-leaf wall (`run_walls.py`, `dev_walls.py`, `.out`)

Concrete 152 mm, the decided row (fit at the laboratory side 3.523 m), face `a` 3.850e-6, `e` 1901 (the two-parameter
concrete of the test palette, ρ 2281 kg/m³), 192 angles × 6 per third. Library (`band_probe2`, tracer 4) on `txlib.py`
on the same grid within 0.01 dB in every octave; sheet on traced solid within 1.4e-7 dB. Octaves 125 Hz–4 kHz:

| | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | --- | --- | --- | --- | --- | --- |
| library, random incidence, lab side (dB) | 41.4 | 47.0 | 52.1 | 56.9 | 61.5 | 65.0 |
| library, 0° (dB) | 44.0 | 51.4 | 57.5 | 63.3 | 68.9 | 74.6 |
| Litvin & Belliston 1978 (fit reference) | 41.9 | 46.5 | 52.6 | 56.3 | 62.0 | 64.9 |
| − that | −0.54 | +0.54 | −0.54 | +0.58 | −0.54 | +0.09 |
| Mahn et al. 2023, CON150 (150 mm, 374 kg/m²) | 41 | 42 | 51 | 59 | 65 | 74 |
| − that | +0.36 | +5.04 | +1.06 | −2.12 | −3.54 | −9.01 |
| bare mass law of 347 kg/m² (`fRef` 5.0161e-5, `m` 1, `tf` law, `side` 3.29), random | 47.1 | 51.7 | 56.7 | 61.9 | 67.3 | 72.7 |
| − Litvin & Belliston | +5.21 | +5.22 | +4.11 | +5.62 | +5.26 | +7.80 |
| room-side absorption (Paris) | 0.0069 | 0.0098 | 0.0141 | 0.0202 | 0.0290 | 0.0416 |
| − Cox & D'Antonio 2016, concrete floor (0.01 0.02 0.02 0.02 0.02 0.02) | −0.003 | −0.010 | −0.006 | +0.000 | +0.009 | +0.022 |

At `side` 3.29 m the same row reads −0.30 dB at 31.5 Hz and −0.04 to −0.14 dB from 63 Hz (lab side minus 3.29).

### 1.3 The window (same scripts)

Float glass 4 mm, the decided row (lab side 1.349 m), face `a` 1.7665e-6, `e` 2083.3 (the test palette's glass).
Library on `txlib.py` within 0.01 dB (0.34 dB at 8 kHz between the 192 × 6 and the 768 × 12 grid); sheet on traced
solid within 7.4e-6 dB.

| | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | --- | --- | --- | --- | --- | --- |
| library, random incidence, side 1.349 m (dB) | 16.5 | 20.5 | 26.3 | 31.5 | 33.3 | 26.5 |
| library, 0° (dB) | 15.8 | 26.4 | 32.4 | 38.4 | 44.5 | 50.7 |
| EN 12758 generic values (fit reference) | 17 | 20 | 26 | 32 | 33 | 26 |
| − that | −0.49 | +0.51 | +0.30 | −0.51 | +0.26 | +0.51 |
| Quirt 1981, 4 mm float, test 78.2 | 23.6 | 25.6 | 29.7 | 33.6 | 32.6 | 28.0 |
| − that | −7.09 | −5.09 | −3.40 | −2.11 | +0.66 | −1.49 |
| library, random incidence, side 3.29 m | 14.8 | 18.7 | 24.7 | 30.2 | 32.1 | 26.3 |
| room-side absorption (= `tf`, flat in angle) | 0.305 | 0.236 | 0.183 | 0.142 | 0.110 | 0.085 |
| − Everest & Pohlmann, window glass (0.35 0.25 0.18 0.12 0.07 0.04) | −0.045 | −0.014 | +0.003 | +0.021 | +0.039 | +0.045 |

The pane at 1.349 m over 3.29 m: +4.79 dB at 31.5 Hz, +3.42 at 63, +1.72 at 125, +1.3 to +1.9 at 250 Hz–1 kHz.

### 1.4 The porous absorber (`por_run.py`, `por_fit2.py`, `.out`)

Glass wool 50 mm, σ 13112 Pa·s/m² (Dupont et al. 2023, Table 1). Komatsu 2008 eqs. 15–18 (`L = 2 − log10(f/σ)`;
`zc = 1 + 0.00027 L^6.2 − j 0.0047 L^4.1`, `kc = 1 + 0.0004 L^6.2 − j 0.0069 L^4.1`, `exp(+jωt)`), `ε = kc/zc`,
`μ = zc·kc`. Columns fitted (minimax of the relative error of `zc`, `kc` over 60 points 22 Hz–11.3 kHz, `fRef` 0.87403
GHz): `a` 1.13623, `b` −0.0439324, `c` 0.00239113, `d` 1.25334, `e` 2.24946, `f` −0.45515, `g` 0.0880445, `h`
0.149437; worst relative error 0.1554 (40 further starts: 0.1553; the ε / μ objective 0.297, so this is the power
law's limit over the whole band). Wall: 200 mm two-parameter concrete, `m` 1, `tf` 0 (the convention behind a fill),
`side` 3.29. 192 angles × 6 per third.

| octave | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| library, Paris (traced solid = sheet within 3.1e-9) | 0.012 | 0.037 | 0.106 | 0.280 | 0.589 | 0.861 | 0.935 | 0.949 | 0.938 |
| fluid layer of the columns (script) | the same within 2.8e-7 | | | | | | | | |
| fluid layer of the exact Komatsu values | 0.016 | 0.050 | 0.141 | 0.334 | 0.610 | 0.837 | 0.930 | 0.949 | 0.962 |
| Dupont et al., sound-field synthesis, 1.5 × 1.5 m sample (measured) | | | | 0.407 | 0.720 | 0.887 | 0.937 | 0.940 | |
| − that | | | | −0.127 | −0.131 | −0.026 | −0.002 | +0.009 | |
| Dupont et al., JCA model of the infinite layer | | | 0.130 | 0.347 | 0.657 | 0.887 | 0.945 | 0.958 | |
| − that | | | −0.024 | −0.067 | −0.068 | −0.026 | −0.010 | −0.009 | |
| Dupont et al., chamber (Sabine) | | | 0.153 | 0.587 | 0.993 | 1.073 | 1.017 | 0.987 | |

The power-law columns cost up to 0.054 in α against the model itself (250 Hz); the rest is the model and the finite
sample (the reference values are read from the paper's figure).

### 1.5 The 13 starting rows (`eval13.py`, `table13.py`)

`tools/acoustic/txlib.py` on the stored fit vectors (`scratch/grazing/fits/P1_davy_side_lab.json`), columns by
`txfit.columns`, 768 angles × 12 per third, 0°–90°: every worst deviation as stored (0.85, 1.78, 2.54, 0.31, 0.57,
0.01, 0.58, 0.49, 1.56, 2.67, 2.70, 1.95, 0.00 dB). With the columns rounded to 5 significant digits as printed on the
page: the same, except concrete 152 0.56 dB (0.57 at full precision). The page quotes the rounded rows and their worst.

### 1.6 Limit and rule sizes, with their source

| size on the page | source |
| --- | --- |
| first path through a layered wall −4.87 dB at 85°; +1.37 dB after 20 interactions, +0.04 dB after 60 | `test_wave.txt` table "mirror through ray_progress", wall AB 2,6, tracer 4, rows 85° (T1_dB −4.8748; T_dB +1.3709 at 20, +0.0378 at 60) |
| plate of three or more bodies within +3.02 / −3.01 dB | `test_acoustic.txt` "plate depth S" (max +3.0198; plaster \| brick \| plaster −3.0100) |
| a wall with an ordered stack inside up to 3.01 dB from its traced twin | `test_acoustic.txt` "held lining E" (K \| T W \| K, d_twin +3.012846) |
| the two sides of a lossy lining on a rigid wall: 0.07–1.38 dB (dense), 0.52–2.70 dB (porous), 10°–75° | `test_acoustic.txt` "held lining H", rows W \| C and W \| A, type 4, d_sides |
| HH rows of a mass-law material up to 3.91 dB from the twin, sides up to 7.53 dB | same table, W \| A, k 3 |
| a sheet between two bodies: R 0.154 where the fluid layers give 0.994 (125 Hz, 0°) | `test_acoustic.txt` "box 10 mm \| porous sheet 15 mm \| wall" (0.153676 / 0.993591) |
| a lossy rigid body (loss tangent 0.04, `tf` 0.3, 200 mm) reflects 1.000000 from 10°, its entry passes −87.9 to −101.8 dB | `test_acoustic.txt` "lossy verdict J" |
| an unknown bend keeps the segment's cosine in the mass term: 0.21 / 0.96 / 3.00 dB at 10° / 20° / 30° | `test_acoustic.txt` "mass cosine E", rows launched (paid−TL_air) |
| `tf` on a lossy wave face: +1.01 / +5.26 dB at `tf` 0.5 / 0.9 for `eps 1.50 − 0.40j`, +0.03 / +0.22 dB for `eps 5.24 − 0.63j` (normal incidence, against `(1 − tf)·R₀`) | `lim_run.py` 1 (half-space in air, EM type 1, 1 GHz) |
| a negative `resS`: the two ports of a 37.5 mm sheet of `ε` 4 (`resF` 1 GHz, `resQ` 10, `resS` −0.5) add to 14.2 at 0.952 GHz | `lim_run.py` 2 (re-measured; the old 2.11 had other, unstated pole columns) |
| a rigid permittivity without `m` as a sheet: 22 mm of `ε` 2.1609e-8 passes −0.03 to −0.28 dB at 10° to 60° (400–630 Hz) | `lim_run.py` 3 (the md text's −0.23 to −0.88 dB not reproduced: other frequencies; the page quotes today's) |
| the one-parameter mapping: reflection phase 20.00° / 90.00° / 170.00° at 10° / 45° / 85° (two-parameter 0.01° / 0.06° / 0.69°); the 50 mm glass-wool layer on it absorbs up to 0.31 more per angle at 125 Hz (60°), 0.21 at 250 Hz | `lim_run.py` 4 (the decision's 0.39–0.53 was another layer; the page quotes today's) |
| a meshed gap in place of the box: up to 0.29 in \|R\|² off the box at 255 interactions (637 rays against 6) | `lim_run.py` 5 (the board of `band_probe_example.in`; the box reproduces the example's 0.925135 / 0.649358 / 0.444760 / 0.778824) |
| far from the origin (float, 4 mm `itu_glass` traced, 28 GHz): at 1 km within 1.34° of phase and 0.50 dB (85°) of the run at the origin; at 100 m the paths at 88° and 89° are lost | `shift_run.py` (double precision: within 1.2e-5 dB to 1 km, `test_wave.txt` "4 mm glass at 28 GHz moved away from the origin") |
| the relaunch offset, 8 float ulp of the hit point's largest coordinate (at least of 1 m) | the code (`qd_relaunch_offset`) |

Sizes dropped (not printed today, not re-measured): the series of a ray launched in a host (0.13 to 3.57); the TM
reflectance above one next to a lossy medium (1.178); the `min_gain_dB` regime (−0.215 dB); a curtain per angle (0.35);
the meshed-gap event counts (1126 to 8000); the 4 mm pane lost at 10 km (the probe's own float classification fails
there); the float EM phase "3.8° at 1 km" (today 1.34° in this probe). Their lines stay without the number or with
today's.

### 1.7 What moved out of the user part (Q7)

- To the developer map, one line each: the three propagation forms of a closed leg (exact at the recovered cosine, the
  hybrid, the walk-off), the cosine recovery, the relaunch-ledger offset, the placement details of a sheet's layer (the
  host budget per side, the reflect-port reference plane), the bent flag, the thickness code, how the second bonded
  body learns `X_run`, the state-word layout.
- Stays in the source only: the decaying-root branch rule, the tube-vertex construction, the leak direction rules.
- Nothing of the columns, the formulas, the rules of use or the limits moved.

### 1.8 Each formula of the page and the code it is written from

| page | formula or statement | source (`src/ray_mesh_interact.cpp`) |
| --- | --- | --- |
| 2 | face angle, front side, relaunch offset 8 float ulp, co-location 1 mm | `ray_mesh_interact` 3008–3010, `qd_relaunch_offset` 165–169, `colocation_dist` 109, `classify_hit` 1719–1827 |
| 3 | Snell ratio, whole-tube rule, xprmat layouts and gains | `interact_with` 799–800, 3097–3110, `qd_polbasis` 187–252 |
| 4 | the 26 columns, defaults, checks, 16383 materials | `MaterialCols` 257–374, `Material` 430–499, `ray_state_update` 3496 |
| 5.1 | ε, μ, the pole and its gate | `Material::eta` 504–514, `Material::mu` 517–523 |
| 5.2 | `8.686 k0 \|Im n\|` + `alpha`; `att` per entry; the reflect port's `att` | `Material::medium_gain` 526–562, `interface_gain` 598–602, `qd_stack_resolve` 1259–1268 (`Ge`), 1308–1314 |
| 5.3 | `tf(f)` with the peak, clamp | `Material::tf` 610–619 |
| 5.4 | `X0`, res and coi factors, `mass_TL` both forms, `davy_sigma`, 1.5 mm, bonded increment | `mass_X0` 572–585, `mass_TL` 414–428, `davy_sigma` 400–407, `mass_min_len` 117, `medium_gain` 555–560 |
| 6.1 | Fresnel pair in admittances, flux factors, scalar `1 + r` | `interact_with` 863–889, `flux_amp` 741–754 |
| 6.2 | total reflection only towards a lossless medium | `interact_with` 814–816 |
| 6.3 | pair rule, transformer, pad | `apply_tf_pair` 672–681, `tf_face` 715–732, `closed_face` 693–706 |
| 6.4 | `ζ`, `R′` from `tfP`, TE `ζc` / TM `ζ/c`, the gate | `Material::peaked`, `zeta` 626–633, `closed_face` 699–700 |
| 6.5 | held wave, refraction as from air, nothing reflected back, plate rule | `interact_with` 827–844, 895–896, `qd_slab_resolve` 1432–1433, `qd_plate` 958–961, `close_leg` 4104–4110 |
| 7 | the recursion, the displaced host, the series `S` and `ρ`, the 5° wedge | `qd_stack_resolve` 1211–1348, `qd_slab_resolve` 1424–1475, `faces_not_parallel` 2131–2146 |
| 8 | the words, ledgers, VBS, band rules | `decode_words` 2018–2036, `vbs_geometry` 2413–2438, API block 3215–3330 |
| 15 | sections, call sequences, the code `d = 10^(−4 + 6(k−1)/32766)` | 103–2439, 2964–3180, 3623–4681, `encode_mass_code` / `decode_mass_code` 2042–2055 |

## 2. Phase 2 — final checks and the `md/` updates

### 2.1 The `md/` updates

- `md/acoustic.md` 6: A9 closed (the page rewritten from the code). 10: a new block "Taken (Stephan, 2026-10-21)" with
  M8 decided (two-parameter only on the page, the one-parameter form in one sentence, the fixtures that use it stay as
  guards). Header: one sentence that the rules of use, the limits and the formulas for users live in the page. 2.7: the
  delimiter row of the 13-fit table written with `-` instead of U+2212 (the table renders now). Nothing else.
- `md/wave.md` 10 (line 496): the page is the public page of the material model, a change that moves a formula, a
  default, a rule of use or a limit updates it, and `check_docs.py` checks it.

### 2.2 The last read of the page against the code

Read in full after writing; corrected on the way: the sign of the IRR-glass difference from 3GPP (+1.72 dB, the
library's loss above the model), a garbled sentence on the peaked face's normal-incidence absorption, the raw face
where neither side has `tf` (added), the column count (26, not 27: the brief's and phase 0's "27" counted wrong; the
page says 26), uncited references [4] and [7] (now cited at the series and the mass law), the resonance rule's
wording, the wall's deviation quoted on two grids (0.58 dB on the example grid, 0.56 dB in the table).

Ten spot checks, each against the code line:

| # | on the page | in the code |
| --- | --- | --- |
| 1 | `side` default 3.29 m, read only where `m > 0` | `Material::side = 3.29` (457), `same_as` (661), `mass_TL` takes it from the material |
| 2 | 17.98 in `ε″ = 17.98 σ / f` | `eta` 509, `mu` 522 |
| 3 | the pole needs `resQ > 0` and acts only without the mass column | `eta` 511 |
| 4 | `L = d · 8.686 k0 \|Im n\|` | 540: `8.686 · fGHz · \|Im √(εμ)\| / 0.0477135`, `k0 = fGHz / 0.0477135` per m |
| 5 | a traced solid carries the mass term on a leg longer than 1.5 mm | `mass_min_len = 0.0015` (117), `medium_gain` 555 |
| 6 | `tfP = 4R′/(1 + R′)²`, the larger root, `1/R′` for `tfQ < 0`, `tfP > 1` is `R′ = 1` | `zeta` 629–632: `Rp = ((2 − pk) + 2√(1 − pk))/pk`, `pk = min(tfP, 1)` |
| 7 | `T = aτ/(a + τ − aτ)` | `qd_plate` 958–961 |
| 8 | Davy's σ with `κ = max(k·side/2, 0.1)`, `p`, `h`, `q`, `x` | `davy_sigma` 400–407 (0.6366… = 2/π) |
| 9 | the wedge "within about 5°" | `faces_not_parallel` tolerance 3.8e-3 on `\|n_F · n_S\|` (2143–2145), arccos(1 − 0.0038) = 5.0° |
| 10 | `eps` default 0.15, `ray_progress`'s `thin_slab_threshold` 0.15 | `ray_state_update` declaration (3264), `src/ray_progress.cpp` 230 |

### 2.3 Counts

- The page: 795 lines; Part I 713 (of which the starting-row section 29, the built-in ITU table 18, the references
  58; without the references and the two tables about 610), Part II 82.
  After review (Stephan, 2026-10-21) every paragraph and list item is on one line: 538 lines, Part I 490, Part II 48;
  the text is word for word the same (checked), `check_docs.py` passes.
  The 70 inline formulas in the dollar-backtick form were turned into plain `$…$` (that form renders on GitHub only;
  Stephan's viewer showed it raw); the conjugate `^*` is written `^{\ast}` so that no `*` is left inside inline math.
- `src/ray_mesh_interact.cpp`: `git diff --stat` 20 lines (+13 / −7), every changed line inside `/*!MD … MD!*/`
  (checked line by line), 4731 lines: the seven corrections of 0.6 as listed, the `path_dirN` one confirmed first by a
  probe (`held_dir.cpp`: held in a plate at 30°, into air 30.0000°, into `ε` 2.25 19.4712°, which is the refraction
  from air, types 1, 2, 4, 5 alike). `tools/extract_md.py -a cpp` into scratch: the generated C++ API differs from
  before in exactly those lines and the line-number column of the index.
- Suites on the final source, 12 threads, one after another: `[wave]` 163474 assertions in 178 cases, `[acoustic]`
  42741 in 59, whole suite 1344398 in 1264; `rowdiff` against the root baselines identical (141128, 112451, 256768
  numbers, 0 beyond).
- `python3 -I tools/acoustic/check_docs.py`: all checks pass, with the page's new lines (tables, spelling, columns 26
  of 26, labels none, math ok, anchors all resolve). The new checks were tested on a corrupted copy: a removed column,
  three labels, an odd `$` and a broken anchor were each reported.
- Other files: `tools/acoustic/check_docs.py` (+70, the page checks), `tools/acoustic/README.md` (its row).

### 2.4 Side findings (one line each, no items)

- `tools/acoustic/band_probe.cpp` has no `attB` column (24 columns), and every decided fit row carries one; the
  examples used a scratch copy with it appended.
- `tools/acoustic/__pycache__/txlib.cpython-314.pyc` is tracked: any import of `txlib` rewrites it (restored here;
  `PYTHONDONTWRITEBYTECODE=1` avoids it).
- `format_materials.md` (0.7) still describes the old model; a separate session updates it (decision Q2).

### 2.5 Stop

Phase 2 is done; nothing is committed. For Stephan: the page (also on GitHub, rendered), the API-block diff of
`src/ray_mesh_interact.cpp`, the `md/` updates of 2.1, and `html_docu` to regenerate. If the meaning of `coiQ` changes
(the pending bending-loss proposal), 1.2, 1.3 and 1.5 are re-measured.
