# Test Specification: `ray_state_update`

Blind, isolated verification of the public `ray_state_update` API. The implementer of these tests is given only this document, the design spec (`ray_state_update_spec.md`), and the function signature — not the implementation. Tests are Catch2, matching the conventions of `test_ray_mesh_interact.cpp`.

This suite tests `ray_state_update` in isolation: not `ray_mesh_interact`, not `calc_diffraction_gain`. The design spec is the oracle for behavior; all expected values are computed by independent reference math (ITU-R P.2040 Fresnel/medium loss and the closed-form Airy sum), never by calling library internals.

Section references below point into the design spec: §5 gain model, §7 encoding, §8 passes, §9 physics, §10 dispatch tables.

---

## 1. Methodology

`ray_state_update` takes every classifier input as an argument, so each test builds synthetic inputs directly and never calls `ray_mesh_interact` or constructs a mesh. This is what makes blind, deterministic, single-row testing possible: the dispatch keys on `(interaction_type, out_typeN, no_interact, mtl_ind_current_in, mtl_ind_buffer_in, geometry, normal_vecN)`, all set by hand. A one-ray call (`n_ray = n_rayN = 1`) hits exactly one §10 row; multi-ray calls test mapping and batching.

Each test asserts on two independent channels:

1. State outputs (`mtl_ind_prev_out`, `mtl_ind_current_out`, `mtl_ind_buffer_out`) — `short` integers, compared exactly. Crisp pass/fail for "did the right case fire."
2. Gain outputs (`gainN`, `xprmatN`) — compared against an independent oracle with `approx_equal`.

To distinguish the three gain operations (keep `IG`, replace with isotropic, apply `S`), feed a distinctive non-trivial incident `xprmatN` (e.g. power 0.5, not 0 or 1) so that a keep returns the input bit-for-bit, a replace returns the oracle value independent of input, and an `IG · S` returns input × `S`.

Both gain modes (§5) are exercised: EM (`interaction_type ∈ {0,1,2}`) and scalar (`{3,4}`). Use simple real-`eta` materials where a clean closed form is wanted, complex `eta` only in the loss tests. Test both `float` and `double` instantiations; `absdiff` tolerance `1e-9` for double, `1e-5` for float.

---

## 2. Shared helpers and oracles to build

Reuse from `test_ray_mesh_interact.cpp`: `calc_transition_gain`, `calc_transition_gain_mu`, `mtl_matrix_to_map`, `make_cube`, `set_mu`. Add:

- `enc(material, flag)` / `mat_of(w)` / `flag_of(w)` — encode/decode a `short` state word by bit-masking: `mat = w & 0x7FFF`, `flag = w & 0x8000`, `enc(X, true) = (short)(X | 0x8000)`. Must round-trip `enc(0, true) == (short)0x8000 == -32768`, `mat_of(0x8000) == 0`, `flag_of(0x8000) == true`, and `enc(5, true)` must decode to material 5 (the `-X` trap: a naive `-5` would decode to 32763). Mirrors §7.
- `airy_S(r_near, r_far, beta, L)` — `1 / (1 - r_near · r_far · phi²)`, with `phi` built per §9.1: `|phi| = sqrt(medium_gain(slab, L))` (full loss model) and `arg(phi) = -(omega/c) · Re(sqrt(eta_slab)) · L`. The oracle must use the same split, or it will disagree with a correct implementation on lossy materials.
- `slab_RT(eta_in, eta_slab, eta_back, theta_deg, L, freq, tf...)` — independent reference for a parallel slab: Fresnel `r/t` at both interfaces (reuse the construction in `calc_transition_gain`), the one-way `phi` above, and the two-port assembly of §9.2 — transmission `T = t_F·phi·t_B·S`, reflection `R = r12 + t12·t21·r23·phi²·S`, plus absorbed `A`. Returns power `R_slab`, `T_slab`, `A_slab`, with `tf` applied to the effective coefficients per §9.5.
- `make_inputs(...)` — assemble the arma containers for an `n_rayN`-ray call from a per-ray list of `(interaction_type, out_type, no_interact, M1, M2, current_in, buffer_in, prev_in, orig, fbs, sbs, dest, N_fbs, N_sbs, incident_xprmat)`, returning everything ready to pass, including a `mtl_prop` map and `mtl_ind_fbs/sbs` as `short`.

Material palette: `AIR` (eta 1), `DENSE` (eta 4, real, lossless, high mismatch), `LOSSY` (eta 5 + large conductivity, fast attenuation), `ABSORBER` (low loss, high mismatch — the resolve-critical case), `MATCHED` (eps = mu via `set_mu`, zero reflection), `MASS` (nonzero mass-law column, to exercise the §9.1 mass approximation).

---

## 3. Test groups

### Group 0 — Oracle self-consistency

- 0.1 Airy degeneracies. `r_far = 0` → `S = 1` and `T_slab` reduces to the single pass; lossless slab → `R_slab + T_slab = 1` for several `(L, freq)`; matched → `T = 1`, `R = 0`. Guards the oracle before it is used as a reference.
- 0.2 `phi` magnitude source. For a `LOSSY`/`MASS` slab, confirm `|phi(L)|² = medium_gain(slab, L)` in the oracle, so the oracle's `rho` tracks `medium_gain(2L)` (§9.4).

### Group 1 — Encoding and flags (§7)

- 1.1 Round-trip. `enc`/`mat_of`/`flag_of` round-trip including `enc(0,true) == -32768` and `enc(5,true)` decoding to 5 (the `-X` trap).
- 1.2 RESOLVED flag (current). Drive a reflection-pass internal reflection that resolves; assert `mtl_ind_current_out < 0` with `mat_of` equal to the carried material.
- 1.3 NON-PARALLEL flag (prev). Drive an o-i-o entry with a wedge; assert `mtl_ind_prev_out` has bit 15 set and the correct material, including the air case (`(short)0x8000`).
- 1.4 No spurious flags. A plain enter/exit with antiparallel faces leaves both flag bits clear.

### Group 2 — Dispatch coverage (§10, one TEST_CASE per family)

Each §10 row hit with a one-ray synthetic call; assert the three state outputs exactly plus the gain class. Tag each test with the spec row.

- 2.1 Dispatch order and resolved-ray precedence (§10.0). `resolved` set: reflection pass → KILL; transmission i-o (2/8/14) → `IG` (out-couple), `current_out ← 0`, flag cleared; transmission i-i (4/5) → `IG`, `current_out ← (short)(iM | 0x8000)` with `iM = (out_type==5?M2:M1)`, flag kept, `prev_out ← old cur`; transmission other (o-i/edge) → pass-through, unchanged. TR code with `resolved` set → KILL (TR wins, §10.6 intercepts first).
- 2.2 o-i family (§10.1). Branch A (nH==1 t1, nH==2 t7/t13): enter `IG·MED(M1, d(fbs,dest) −off)`, `current_out ← M1`, `+flag`; nested `MED(cur, d(orig,dest))`, `buffer_out ← M1`. Branch B (nH==2 t1): bare `IG`, `current_out ← M1`, `+flag`; nested `MED(cur, d(orig,fbs))`. Branch C (nH>2): `cur==0,buf!=0` → KILL; t1/7 enter; t13 enter with `buffer_out ← M2`; nested `MED(cur, d(orig,fbs)+ray_offset)`.
- 2.3 i-o family (§10.2). Branch A false-inside `IG`; cavity exit `IG·S`, `current_out ← 0`; nH==1 t2 same_mat / else buffer branches; nH==2 t8/14 ii-oo. Branch B i-o-i: `M2==0` → KILL; cavity exit `IG·S`; buffer branches (survives). Branch C/D nH>2 t2/14/8.
- 2.4 M2M (§10.3). Illegal `cur==0` → KILL; `M1==0 or M2==0` → KILL; cavity transition `IG·S·MED(iM, d(fbs,dest) −off)`, `current_out ← iM`, `prev_out ← old cur`; buffer-swap branch; nH>2 spurious / cavity-transition rows.
- 2.5 Edge o-i-o (§10.4). Graze `IG`, `current_out ← 0`; same_mat / else; nH>2 stay-outside and buffer branches. No `S` on any row.
- 2.6 Edge i-o-i (§10.5). `current_out ← (d(fbs,sbs) > 1e-6 ? M2 : 0)`; same_mat / else; nH>2 `cur==0` → KILL, buffer branches. No `S`.
- 2.7 TR (§10.6). `interaction_type == 2` + TR code → KILL, state unchanged.
- 2.8 Reflection pass (§10.7). Front / order-0 (`cur==0`) → bare `IG`, no `S`, no flag; internal/back (`cur!=0`, resolvable) → `IG·S`, RESOLVED flag set.
- 2.9 Global default (§10.0). An unmatched `(out_type, nH, state)` on the transmission path → KILL.

### Group 3 — Mapping, batching, in-place (§6, §12)

- 3.1 `ray_ind` mapping. `n_ray=5`, `n_rayN=2`, `ray_ind = {3,1}`; distinct `current_in` per full-ray row; each output reads its mapped old state and writes compacted `[n_rayN]` in order.
- 3.2 Identity. `ray_ind == nullptr`, `n_ray == n_rayN`.
- 3.3 Old state read-only. `*_in` arrays byte-identical after the call.
- 3.4 In-place + consistency. `gainN == (EM ? 0.5 : 1.0) · Σ|xprmatN|²` per row after the call (mode-dependent, §5). Run with `gainN == nullptr` and with `xprmatN == nullptr` — no crash, the provided buffer patched.
- 3.5 `normal_vecN == nullptr`. Runs; every slab defaults to resolve, gated by survival/clamp (§9.3).
- 3.6 OpenMP determinism. A mixed batch equals the same rows run one-at-a-time (no cross-ray state bleed).

### Group 4 — Gain operation semantics and modes (§5)

- 4.1 `IG` keeps `xprmatN` bit-identical (distinctive input proves no silent replace).
- 4.2 Replace is isotropic and independent of incident `xprmatN` (feed two different inputs, expect the same output).
- 4.3 `IG · S` complex-multiplies each Jones entry: check magnitude and phase against `airy_S`.
- 4.4 Product chains (`MED·TRN·MED`) equal the product of independent oracle factors.
- 4.5 KILL → `xprmatN == 0`, `gainN == 0`.
- 4.6 EM convention. `interaction_type ∈ {0,1,2}`: `gainN = 0.5·Σ|xprmat|²`; an isotropic replace places `sqrt(g)` on VV and HH, zero off-diagonal.
- 4.7 Scalar convention. `interaction_type ∈ {3,4}`: feed a VV-only incident field; `gainN = Σ|xprmat|² = |VV|²` (no 0.5); an isotropic replace places `sqrt(g)` on VV only, with HV, VH, HH zero. Confirms the §5 / §13 scalar identity is not broken by a misplaced `0.5` or an HH write.
- 4.8 `gainN`-only `S` degradation. With `xprmatN == nullptr`, an `IG · S` row applies `gainN *= |S|²` (magnitude only, no phase).

### Group 5 — Fabry-Pérot physics (§9)

- 5.1 Single lossless slab transmission. `DENSE` slab cavity exit; sweep `(L, freq)` across at least one resonance peak and one null; `gainN_out` consistent with `slab_RT`'s `T_slab` and the `airy_S` phase.
- 5.2 Energy conservation, lossless (primary `S` check). Run the reflection and transmission passes on the same `DENSE` slab; `R_slab + T_slab` from the two output powers equals 1 across several `(L, freq, theta)`. Convention-independent; depends on the §9.2 two-port decomposition (front bare `r12`, internal `r23·S`).
- 5.3 Energy conservation, lossy. `ABSORBER` slab: `R + T + A = 1`, `A` matching the round-trip `medium_gain`. Drive it hard (large `A`), per §13, so a dropped return-trip factor breaks the balance.
- 5.4 Mass-law material. `MASS` slab: confirm the implementation sources `|phi|` from `medium_gain` (energy still closes to within the bounded mass-law drift), not from an eta-only `Im(β)` (which would over-resolve and leave `R + T + A > 1` by the alpha/mass loss). Compare resolve behavior on `MASS` vs a dielectric-only slab of equal `medium_gain(2L)`.
- 5.5 Survival gate, `eps = 0.15` (§9.4). Three regimes, checked clearly inside each: `MATCHED` (rho ≈ 0) → single-pass, no `S`; `LOSSY` (rho ≪ 0.15) → single-pass; `ABSORBER` thin, low-loss, high-mismatch (rho ≫ 0.15) → `S` enhancement clearly present. Pass/fail: `|gain_out − single_pass|` below tol for the first two, above a wide margin for the third. Optionally bracket the `rho = 0.15` crossover, but assert behavior only in the clearly-separated regimes.
- 5.6 Parallelism is a magnitude test (§9.3). Stage two calls (`prev_out → prev_in`). At the o-i-o entry, vary `normal_vecN`: (a) antiparallel (`dot ≈ -1`) → flag clear → exit resolves; (b) parallel, same orientation (`dot ≈ +1`, the layered-slab / FP-noise case) → flag clear → exit resolves; (c) genuine wedge (`dot ≈ 0.5`, well inside `1 − tol`) → flag set → exit re-emits. Pass/fail: the `prev_out` flag bit and the presence/absence of `S`. This is the key check that both normal signs count as a slab and only real wedges are rejected.
- 5.7 Two-port reflection decomposition (§9.2). Reflection pass at the front (`cur==0`) returns bare `IG` (no `S`, no flag); at the internal interface (`cur!=0`, gates pass) returns `IG·S` with the RESOLVED flag set. Assert the front has no `S` factor and the internal one does; together with the transparent exit (5.8) this is what makes 5.2 close.
- 5.8 Resolved transparent exit (§10.0). A `resolved` ray on the transmission pass: i-o exit → `IG` (out-couple `t21`), `current_out ← 0`, flag cleared; i-i → `current_out ← (iM | 0x8000)`, flag kept; reflection pass → KILL.
- 5.9 TR forward kill. `interaction_type == 2` + TR code → `gainN == 0`.
- 5.10 `tf` fold-in (§9.5). Apply `tf`; `slab_RT` with the same `tf`. Energy closes (`R + T + A = 1`), and the resolved gain matches the `tf`-effective oracle. Tripwire: building `S` from bare Fresnel while the ports use `tf`-adjusted coefficients fails to close.
- 5.11 Clamp (§9.7). Lossless, near-grazing, thickness tuned toward the pole (`r_near·r_far·phi² → 1`): all eight `xprmatN` reals and `gainN` are finite, `|S|` bounded; re-emit when `|1 − r_near·r_far·phi²| < 1e-2`.

### Group 6 — Cross-pass invariance (§9.8)

- 6.1 For a fixed slab, the reflection-pass and transmission-pass calls make the identical resolve-vs-re-emit decision: assert the gate inputs (`r_near`, `r_far`, `phi`, `rho`, parallelism flag) and the outcome match across the two passes. A disagreement would double-count or lose energy.

### Group 7 — Stacked slabs and flag persistence

- 7.1 Flag persists. Stage a `resolved` ray through two internal interfaces (staged calls feeding `current_out → current_in`): the RESOLVED flag stays set across the internal i-i crossing and clears only on exit to air. Assert the flag bit at each stage.
- 7.2 No energy creation. A two-layer stack resolved by staged calls satisfies `R + T + A ≤ 1` (within tol) — never `> 1`. Regression guard for the persist rule.

### Group 8 — Type parity

- 8.1 A representative subset (one row per family, the energy check, the Airy check, both gain modes) run on `float` and `double`; float within `1e-5`, double within `1e-9`.

### Group 9 — Input validation (the public IO contract)

`ray_state_update` is a public entry point; malformed input must fail fast with a thrown exception (`std::invalid_argument`, the library convention), never undefined behavior — no segfault, no out-of-bounds read, no silent wrong answer, no partial output write. Assert with `CHECK_THROWS_AS(call, std::invalid_argument)` and, where the message is stable, `CHECK_THROWS_WITH(call, Catch::Matchers::ContainsSubstring("..."))` on a keyword (the offending parameter or dimension), not exact text. Pair each malformed case with an otherwise-identical well-formed call that must succeed, so the guard is not over-eager. These tests need no gain oracle — they assert only the throw (and that nothing was written).

The dimension contract (from §4): full-set inputs `orig`/`dest`/`fbs`/`sbs` are `[n_ray, 3]`; `no_interact` and the three `*_in` are `[n_ray]`. Compact-set inputs/outputs `fbs_angleN`, `out_typeN`, `mtl_ind_fbs`, `mtl_ind_sbs`, the three `*_out`, `gainN`, `xprmatN`, `ray_ind` are `[n_rayN]` (or `[n_rayN, k]`); `normal_vecN` is `[n_rayN, 6]`, `xprmatN` is `[n_rayN, 8]`. Optional (nullptr allowed): `normal_vecN`, `ray_ind`, and exactly one of `gainN`/`xprmatN`.

- 9.1 `interaction_type` outside `{0,1,2,3,4}` (e.g. 5, -1) → throw.
- 9.2 `center_frequency <= 0`, and NaN/Inf → throw.
- 9.3 Geometry with a column count other than 3 (`orig`/`dest`/`fbs`/`sbs`) → throw.
- 9.4 `normal_vecN` present but not `[n_rayN, 6]`; `xprmatN` present but not `[n_rayN, 8]` → throw.
- 9.5 Full-set row-count mismatch — any of `orig`/`dest`/`fbs`/`sbs`/`no_interact`/`*_in` disagreeing on `n_ray` → throw.
- 9.6 Compact-set row-count mismatch — any of `out_typeN`/`fbs_angleN`/`mtl_ind_fbs`/`mtl_ind_sbs`/`*_out`/`gainN`/`xprmatN`/`ray_ind`/`normal_vecN` disagreeing on `n_rayN` → throw.
- 9.7 Pointer nullptr handling. Every pointer must be null-checked before its first dereference, so a null parameter yields either the documented default or a thrown exception — never a dereference and crash. Test each pointer individually, with the rest of the call well-formed:
  - Required → throw. Each of `orig`, `dest`, `fbs`, `sbs`, `no_interact`, `fbs_angleN`, `out_typeN`, `mtl_prop`, `mtl_ind_fbs`, `mtl_ind_sbs`, the three `*_in`, the three `*_out`, individually nullptr → `std::invalid_argument` naming the parameter, before any output is written.
  - Optional → documented default, verified by the output (not merely "no throw"): `normal_vecN == nullptr` → flag stays clear, default resolve, and a parallel slab still resolves via the survival gate (assert `S` applied); `ray_ind == nullptr` → identity mapping, outputs align 1:1 (requires `n_ray == n_rayN`, else 9.8 throws); `gainN == nullptr` → `xprmatN` patched correctly, no crash; `xprmatN == nullptr` → `gainN` patched correctly (`S` degrades to `|S|²`, cf. 4.8).
  - `gainN` and `xprmatN` both nullptr → throw (9.12).
  - Because a segfault aborts the Catch2 process rather than registering a failed assertion, isolate each pointer in its own `SECTION`/`TEST_CASE` so a crash is unambiguously attributable to the offending parameter. A crash instead of a throw is itself the failure — it means the null-check is missing or comes after a dereference.
- 9.8 `ray_ind` containing an index `>= n_ray` → throw (prevents an out-of-bounds read of the full-set arrays). With `ray_ind == nullptr` and `n_ray != n_rayN` → throw (identity mapping requires equal sizes).
- 9.9 Material index out of range — a `mtl_ind_fbs`/`mtl_ind_sbs` value, or a `*_in` state word whose `mat = w & 0x7FFF`, exceeding the material count in `mtl_prop` → throw. `mat == 0` (air/empty) is always valid.
- 9.10 Malformed `mtl_prop` — a material entry with the wrong coefficient count or a missing required key → throw (via `mtl_validate`).
- 9.11 Output buffer sizing — the `*_out`/`gainN`/`xprmatN` buffers must end up `[n_rayN]`. If the function requires caller pre-sized buffers, a wrongly sized output throws rather than overruns; if it resizes internally, it must resize to exactly `[n_rayN]`/`[n_rayN, 8]`. Assert whichever the implementation documents.
- 9.12 Both `gainN` and `xprmatN` null → throw. (§5's "either may be nullptr; the other is still patched" implies at least one output is required; with neither, there is nothing to patch.)
- 9.13 Well-formed empty batch (`n_ray == n_rayN == 0`) → clean no-op, no throw, outputs left empty. Boundary case, not an error.

An out-of-range or unexpected `out_typeN` value is data, not malformed IO — it is absorbed by the global-default KILL (§10.0), not an exception. Only structural/contract violations throw.

This group codifies a validation contract that §4 of the design spec states only implicitly (via the dimension and optionality annotations). Consider adding a short "input validation" note to §4 so the implementation and these tests cite the same contract.

---

## 4. Coverage matrix (test → design-spec section)

| Design-spec section | Covered by |
| --- | --- |
| §5 gain model, modes | 3.4, 4.1–4.8 |
| §6 mapping, in-place | 3.1–3.6 |
| §7 encoding, flags | 1.1–1.4 |
| §8 passes, TR scope | 2.7, 5.9 |
| §9.1 cavity, `phi`, `L` | 5.1, 5.4 |
| §9.2 two ports | 5.2, 5.7 |
| §9.3 parallelism (magnitude) | 5.6, 3.5 |
| §9.4 survival gate, `eps=0.15` | 5.5 |
| §9.5 `tf`/Stokes | 5.10 |
| §9.6 TR | 2.7, 5.9 |
| §9.7 clamp | 5.11 |
| §9.8 cross-pass invariance | 6.1 |
| §10.0 order, precedence, default | 2.1, 2.9, 5.8 |
| §10.1 o-i | 2.2 |
| §10.2 i-o | 2.3 |
| §10.3 M2M | 2.4 |
| §10.4 edge o-i-o | 2.5 |
| §10.5 edge i-o-i | 2.6 |
| §10.6 TR | 2.7 |
| §10.7 reflection pass | 2.8, 5.7 |
| §13 energy ledger, scalar | 5.2, 5.3, 7.2, 4.7 |
| §4 signature / IO contract | 9.1–9.13 |

Tag each §10-row test with a `// spec 10.x: <row>` comment so a reviewer can confirm full table coverage at a glance.

---

## 5. Pass/fail philosophy

- State outputs are exact. Integer `short` comparisons; any mismatch is a hard fail. These verify the dispatch fires the intended row.
- Gains use independent oracles. Never compare against a library-internal helper; reimplement the spec's formulas (ITU-R Fresnel/medium loss, the Airy sum with `phi` from `medium_gain`) so a shared bug cannot hide.
- Energy conservation is the strongest physics check and is convention-independent — prefer it over matching an exact `S` value where a sign/averaging convention could differ. Drive lossy cases hard so a missing return-trip factor surfaces.
- Gates are tested in clearly-resolve and clearly-re-emit regimes. `eps` is now the concrete `0.15`, so the crossover may be bracketed, but assertions live in the separated regimes; the clamp and parallelism behaviors are likewise checked away from their thresholds.
- Distinctive inputs (non-trivial incident `xprmatN`, distinct per-ray state, mapped `ray_ind`, VV-only vs full Jones) make replace-vs-multiply, mapping, mode, and cross-ray-bleed failures visible rather than silently passing.
- Input validation fails fast: malformed input throws `std::invalid_argument` before any output is written — never undefined behavior. Each negative case is paired with a passing well-formed call so the guard is not over-eager.