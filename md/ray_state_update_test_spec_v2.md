# Test Specification: `ray_state_update`

Blind, isolated verification of the public `ray_state_update` API. The implementer is given this document, the design spec (`ray_state_update_spec.md`, the behavior oracle for the dispatch tables and physics), and the existing `test_ray_mesh_interact.cpp` (conventions and reusable oracles) — but **not** the implementation of `ray_state_update`. All expected values are computed by independent reference math (ITU-R P.2040 Fresnel/medium loss and the closed-form Airy sum), never by calling library internals.

Tests are Catch2, matching the conventions of `test_ray_mesh_interact.cpp`. American English throughout; plain comments only (no banner/box comments). Both `float` and `double` instantiations; `absdiff` tolerance `1e-9` for double, `1e-5` for float.

---

## 1. API contract (authoritative)

```
# ray_state_update
Batched inside/outside ray-state machine with analytic thin-slab (Fabry-Perot) resolution.
Corrects the per-interaction gainN / xprmatN produced by ray_mesh_interact using a tracked per-ray
medium state, and carries that state forward. Three signed-short words per ray hold the current
medium, the previous medium, and a one-slot next-transition buffer (bit-masked: mat = w & 0x7FFF,
flag = w & 0x8000). Overlays a closed-form thin-slab factor S (the Airy sum). Called twice per
interaction: reflection pass (interaction_type 0 or 3) and transmission/refraction pass (1, 2 or 4).
With S suppressed (the survival gate re-emits) the transmission/refraction path reproduces
calc_diffraction_gain bit-for-bit.

void quadriga_lib::ray_state_update(
    int interaction_type,                                       // 0 EM refl, 1 EM trans, 2 EM refr, 3 scalar refl, 4 scalar trans
    dtype center_frequency,                                     // [Hz]
    const arma::Mat<dtype> *orig,                               // [n_ray, 3], read at g = ray_ind[i]
    const arma::Mat<dtype> *dest,                               // [n_ray, 3]
    const arma::Mat<dtype> *fbs,                                // [n_ray, 3]
    const arma::Mat<dtype> *sbs,                                // [n_ray, 3]
    const arma::u32_vec *no_interact,                           // [n_ray] mesh-hit count nH
    const arma::Col<dtype> *fbs_angleN,                         // [n_rayN] FBS incidence angle (ITU)
    const arma::s32_vec *out_typeN,                             // [n_rayN] typeH from ray_mesh_interact
    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,   // csv_prop of obj_file_read
    const arma::Col<short> *mtl_ind_fbs,                        // [n_rayN] M1 (0 = air)
    const arma::Col<short> *mtl_ind_sbs,                        // [n_rayN] M2 (0 = air)
    const arma::Col<short> *mtl_ind_prev_in    = nullptr,       // [n_ray], read at g; NULL = state 0
    const arma::Col<short> *mtl_ind_current_in = nullptr,       // [n_ray]
    const arma::Col<short> *mtl_ind_buffer_in  = nullptr,       // [n_ray]
    const arma::Mat<dtype> *normal_vecN        = nullptr,       // [n_rayN, 6]; NULL disables wedge test
    arma::Col<short> *mtl_ind_prev_out    = nullptr,            // [n_rayN], written at i; NULL skips write
    arma::Col<short> *mtl_ind_current_out = nullptr,            // [n_rayN]
    arma::Col<short> *mtl_ind_buffer_out  = nullptr,            // [n_rayN]
    arma::Col<dtype> *gainN   = nullptr,                        // [n_rayN] in/out, patched in place
    arma::Mat<dtype> *xprmatN = nullptr,                        // [n_rayN, 8] in/out, cols VV HV VH HH (re,im)
    arma::u32_vec *ray_ind    = nullptr,                        // [n_rayN] -> [n_ray]; NULL = identity
    double eps = 0.15);                                         // resolve threshold (see below)
```

Contract distilled for testing:

- Scalars. `interaction_type` must be in `{0,1,2,3,4}` (else throw). `center_frequency` must be `> 0` and finite (else throw). `is_scalar = interaction_type >= 3`.
- Required pointers (null → throw `std::invalid_argument`): `orig`, `dest`, `fbs`, `sbs`, `no_interact`, `fbs_angleN`, `out_typeN`, `mtl_prop`, `mtl_ind_fbs`, `mtl_ind_sbs`.
- Optional pointers and their null-defaults (null is **not** an error):
  - `mtl_ind_prev_in` / `current_in` / `buffer_in` → read as state `0` (outside, no flags).
  - `normal_vecN` → the parallelism (wedge) test is disabled; slabs default to resolve.
  - `mtl_ind_prev_out` / `current_out` / `buffer_out` → that state word is not written.
  - `gainN`, `xprmatN` → that output is not patched. Both null is a valid state-only update.
  - `ray_ind` → identity (then `n_ray == n_rayN`).
- All six state args null → tracking disabled: each interaction is corrected on its own (entry loss, TR kill, single-hit air-gap `S`), but cross-interaction slab `S` and reflection-bounce `S` are absent (they need the tracked medium).
- `eps` (resolve threshold, default `0.15`): `rho = sqrt(R_near · R_far · medium_gain(slab, 2L))`.
  - `eps == 0` → always resolve (the forward/transmission-only mode; no reflection pass to carry the bounces).
  - `0 < eps < 1` → resolve iff `rho >= eps`, else re-emit.
  - `eps >= 1` → `S` disabled (the gate always re-emits); the transmission/refraction path then reproduces `calc_diffraction_gain` bit-for-bit.
  - `eps < 0` or non-finite → throw.
- `xprmatN` columns are VV, HV, VH, HH (re, im per entry).

The design spec is the oracle for *what* each dispatch row and physics gate produces (§5 gain model, §7 encoding, §8 passes, §9 physics, §10 dispatch); this document is the oracle for *how* to test it.

---

## 2. Methodology

`ray_state_update` takes every classifier input as an argument, so each test builds synthetic inputs directly and never calls `ray_mesh_interact` or constructs a mesh. A one-ray call (`n_ray = n_rayN = 1`) hits exactly one §10 row; multi-ray calls test mapping and batching.

Each test asserts on two independent channels: state outputs (`mtl_ind_*_out`) compared as exact `short` integers, and gain outputs (`gainN`, `xprmatN`) compared against an independent oracle via `CHECK(arma::approx_equal(..., "absdiff", tol))`.

`eps` is the lever for the new physics. Run the **topology** tests (Group 4 dispatch coverage) with `eps >= 1` so `S` is off and the `IG · S` rows reduce to plain `IG`; this is the bit-for-bit port surface. Run the **physics** tests (Group 6) with `eps = 0` (always resolve) to force `S` on, except the survival-gate test, which sweeps `eps` directly.

To distinguish the gain operations (keep `IG`, replace with isotropic, apply `S`), feed a distinctive non-trivial incident `xprmatN` (e.g. power 0.5) so a keep returns the input bit-for-bit, a replace returns the oracle value independent of input, and an `IG · S` returns input × `S`. Exercise both gain modes: EM (`{0,1,2}`) and scalar (`{3,4}`).

---

## 3. Shared helpers and oracles to build

Reuse from `test_ray_mesh_interact.cpp`: `calc_transition_gain`, `calc_transition_gain_mu`, `mtl_matrix_to_map`, `make_cube`, `set_mu`. Add:

- `enc(material, flag)` / `mat_of(w)` / `flag_of(w)` — bit-mask encode/decode: `mat = w & 0x7FFF`, `flag = w & 0x8000`, `enc(X, true) = (short)(X | 0x8000)`. Must round-trip `enc(0,true) == (short)0x8000 == -32768`, `mat_of(0x8000) == 0`, `flag_of(0x8000) == true`, and `enc(5,true)` decode to material 5 (the `-X` trap: a naive `-5` decodes to 32763).
- `airy_S(r_near, r_far, beta, L)` — `1 / (1 - r_near·r_far·phi²)`, `phi` per §9.1: `|phi| = sqrt(medium_gain(slab, L))` (full loss model) and `arg(phi) = -(omega/c)·Re(sqrt(eta_slab))·L`. The oracle must use the same split, or it disagrees with a correct implementation on lossy materials.
- `slab_RT(eta_in, eta_slab, eta_back, theta_deg, L, freq, tf...)` — independent slab reference: Fresnel `r/t` at both interfaces (reuse `calc_transition_gain`'s construction), the one-way `phi` above, and the two-port assembly of §9.2 — `T = t_F·phi·t_B·S`, `R = r12 + t12·t21·r23·phi²·S`, plus absorbed `A`; `tf` applied to the effective coefficients per §9.5. Returns power `R_slab`, `T_slab`, `A_slab`, and the gate quantity `rho`.
- `make_inputs(...)` — assemble the call from a per-ray list of `(interaction_type, out_type, no_interact, M1, M2, current_in, buffer_in, prev_in, orig, fbs, sbs, dest, N_fbs, N_sbs, incident_xprmat)` plus `eps`, returning everything ready to pass, with a `mtl_prop` map and `short` material indices. Allow each pointer to be set null to exercise the optional-arg defaults.

Material palette: `AIR` (eta 1), `DENSE` (eta 4, real, lossless, high mismatch), `LOSSY` (eta 5 + large conductivity), `ABSORBER` (low loss, high mismatch — resolve-critical), `MATCHED` (eps = mu via `set_mu`, zero reflection), `MASS` (nonzero mass-law column).

---

## 4. Dispatch coverage (§10) — run with `eps >= 1` (S off)

Each §10 row hit with a one-ray synthetic call; assert the three state outputs exactly plus the gain class. With `S` off the `IG · S` rows reduce to `IG`. Tag each test `// spec 10.x: <row>`.

- 4.1 Dispatch order and resolved-ray precedence (§10.0). `resolved` set: reflection pass → KILL; transmission i-o (2/8/14) → `IG` (out-couple), `current_out ← 0`, flag cleared; transmission i-i (4/5) → `IG`, `current_out ← (short)(iM | 0x8000)` with `iM = (out_type==5?M2:M1)`, flag kept, `prev_out ← old cur`; transmission other (o-i/edge) → pass-through. TR with `resolved` set → KILL (TR intercepts first, §10.6).
- 4.2 o-i family (§10.1). Branch A (nH==1 t1, nH==2 t7/t13), Branch B (nH==2 t1), Branch C (nH>2): enter / nested / KILL rows, `+flag` wedge test on the two-face entries.
- 4.3 i-o family (§10.2). False-inside `IG`; cavity exit; nH==1 t2 same_mat/else; nH==2 t8/14 ii-oo; i-o-i Branch B; nH>2 Branch C/D.
- 4.4 M2M (§10.3). Illegal → KILL; cavity transition; buffer-swap; nH>2 rows.
- 4.5 Edge o-i-o (§10.4) and 4.6 Edge i-o-i (§10.5). Graze rows, same_mat/else, nH>2 rows, the `d(fbs,sbs) > 1e-6 ? M2 : 0` entry. No `S`.
- 4.7 TR (§10.6). `interaction_type == 2` + TR code → KILL, state unchanged.
- 4.8 Reflection pass (§10.7). Front/order-0 (`cur==0`) → bare `IG`; internal (`cur!=0`, resolvable, needs `eps<1`) → `IG·S` + RESOLVED flag (test this row with `eps=0`).
- 4.9 Global default (§10.0). An unmatched `(out_type, nH, state)` on the transmission path → KILL.

---

## 5. Encoding, mapping, modes, arguments

### 5a. Encoding and flags (§7)
- 5.1 `enc`/`mat_of`/`flag_of` round-trip incl. `enc(0,true) == -32768` and `enc(5,true)` → 5 (the `-X` trap).
- 5.2 RESOLVED flag (current): a reflection-pass internal reflection (`eps=0`) sets `current_out < 0` with the correct material.
- 5.3 NON-PARALLEL flag (prev): a wedge o-i-o entry sets `prev_out` bit 15, incl. the air case `(short)0x8000`.
- 5.4 No spurious flags: a plain enter/exit with antiparallel faces leaves both flag bits clear.

### 5b. Mapping, batching, in-place (§6)
- 5.5 `ray_ind` mapping: `n_ray=5`, `n_rayN=2`, `ray_ind={3,1}`; each output reads its mapped old state, outputs compacted in order.
- 5.6 `*_in` arrays byte-identical after the call (read-only).
- 5.7 In-place + consistency: `gainN == (EM ? 0.5 : 1.0)·Σ|xprmatN|²` per row after the call.
- 5.8 OpenMP determinism: a mixed batch equals the same rows run one-at-a-time.

### 5c. Optional arguments (the API contract)
- 5.9 `mtl_ind_*_in == nullptr` → reads as state 0; an o-i entry from a null state behaves as `cur == 0` (enters), matching an explicit zero state.
- 5.10 `mtl_ind_*_out == nullptr` → that write is skipped, no crash; the other outputs still correct. Test each of the three individually null.
- 5.11 `gainN == nullptr` → only `xprmatN` patched; `xprmatN == nullptr` → only `gainN` patched (`S` → `|S|²`, see 6.x); both null → valid state-only update (state written, no crash, no throw).
- 5.12 `ray_ind == nullptr` → identity (`n_ray == n_rayN`); `normal_vecN == nullptr` → wedge test disabled, a parallel slab still resolves (assert `S` applied with `eps=0`).
- 5.13 No-tracking mode: all six state args null. Assert per-interaction corrections still fire — entry loss (o-i `IG·MED`), TR kill, and single-hit air-gap `S` (i-o-i nH==2, computable from one interaction) — while cross-interaction slab `S` is absent (a separate-event exit sees `cur==0` → false-inside `IG`, no `S`).

### 5d. Gain operation semantics and modes (§5)
- 5.14 `IG` keeps `xprmatN` bit-identical; replace is isotropic and input-independent; `IG·S` complex-multiplies each Jones entry (magnitude and phase vs `airy_S`); product chains equal the oracle product; KILL → zero.
- 5.15 EM convention (`{0,1,2}`): `gainN = 0.5·Σ|xprmat|²`; isotropic replace places `sqrt(g)` on VV and HH.
- 5.16 Scalar convention (`{3,4}`): VV-only field; `gainN = Σ|xprmat|² = |VV|²` (no 0.5); isotropic replace places `sqrt(g)` on VV only, HV/VH/HH zero.
- 5.17 `gainN`-only `S` degradation: with `xprmatN == nullptr`, an `IG·S` row applies `gainN *= |S|²` (magnitude only).

---

## 6. Fabry-Pérot physics (§9) — `eps = 0` (always resolve) unless stated

- 6.1 Single lossless slab transmission. `DENSE` slab; sweep `(L, freq)` across a resonance peak and a null; `gainN_out` consistent with `slab_RT`'s `T_slab` and the `airy_S` phase.
- 6.2 Energy conservation, lossless (primary `S` check). Reflection + transmission passes on the same `DENSE` slab; `R_slab + T_slab == 1` across several `(L, freq, theta)`. Convention-independent; depends on the §9.2 two-port decomposition.
- 6.3 Energy conservation, lossy. `ABSORBER`: `R + T + A == 1`, `A` matching the round-trip `medium_gain`. Drive hard (large `A`) so a missing return-trip factor breaks the balance.
- 6.4 Mass-law material. `MASS` slab: confirm `|phi|` is sourced from `medium_gain` (energy closes to within the bounded mass-law drift), not from an eta-only `Im(β)` (which would over-resolve and give `R + T + A > 1`). Compare against a dielectric-only slab of equal `medium_gain(2L)`.
- 6.5 Survival gate, `eps`-driven. For a fixed slab with known `rho = sqrt(R_near·R_far·medium_gain(2L))`: `eps` clearly below `rho` → resolve (`S` applied); `eps` clearly above `rho` → re-emit (single-pass). Limits: `eps = 0` → resolve even a weak near-lossless cavity; `eps >= 1` → never resolve (output equals the no-`S` topology). Assert inside the separated regimes; the `eps ≈ rho` crossover may be bracketed.
- 6.6 Parallelism is a magnitude test (§9.3). Stage two calls (`prev_out → prev_in`), `eps = 0`. At the o-i-o entry vary `normal_vecN`: antiparallel (`dot ≈ -1`) → flag clear → resolve; parallel same-orientation (`dot ≈ +1`, the layered-slab / float-noise case) → flag clear → resolve; genuine wedge (`dot ≈ 0.5`) → flag set → re-emit. Pass/fail: the `prev_out` flag bit and the presence/absence of `S`. Both signs count as a slab; only real wedges are rejected.
- 6.7 Two-port reflection decomposition (§9.2). Reflection pass at the front (`cur==0`) → bare `IG` (no `S`, no flag); internal interface (`cur!=0`, `eps=0`) → `IG·S` + RESOLVED flag. With the transparent exit (6.8) this is what makes 6.2 close.
- 6.8 Resolved transparent exit (§10.0). A `resolved` ray, transmission pass: i-o exit → `IG` (out-couple `t21`), `current_out ← 0`, flag cleared; i-i → `current_out ← (iM | 0x8000)`, flag kept; reflection pass → KILL.
- 6.9 TR forward kill. `interaction_type == 2` + TR code → `gainN == 0`.
- 6.10 `tf` fold-in (§9.5). Apply `tf`; `slab_RT` with the same `tf`. Energy closes; the resolved gain matches the `tf`-effective oracle. Tripwire: `S` from bare Fresnel while the ports use `tf`-adjusted coefficients fails to close.
- 6.11 Clamp (§9.7). Lossless, near-grazing, thickness toward the pole (`r_near·r_far·phi² → 1`): all eight `xprmatN` reals and `gainN` finite, `|S|` bounded; re-emit when `|1 - r_near·r_far·phi²| < 1e-2`.

---

## 7. Cross-pass invariance and stacked slabs

- 7.1 Cross-pass invariance (§9.8). For a fixed slab, the reflection-pass and transmission-pass calls make the identical resolve-vs-re-emit decision: assert the gate inputs (`r_near`, `r_far`, `phi`, `rho`, parallelism flag) and the outcome match.
- 7.2 Flag persists. Stage a `resolved` ray through two internal interfaces (`current_out → current_in`): the RESOLVED flag stays set across the internal i-i crossing and clears only on exit to air.
- 7.3 No energy creation. A two-layer stack resolved by staged calls satisfies `R + T + A <= 1` — never `> 1`.

---

## 8. Type parity

- 8.1 A representative subset (one row per family, the energy check, the Airy check, both gain modes) on `float` and `double`; float within `1e-5`, double within `1e-9`.

---

## 9. Input validation (the public IO contract)

`ray_state_update` is a public entry point; malformed input must fail fast with `std::invalid_argument`, never undefined behavior — no segfault, no out-of-bounds read, no silent wrong answer, no partial output write. Assert with `CHECK_THROWS_AS(call, std::invalid_argument)` and, where the message is stable, `CHECK_THROWS_WITH(call, Catch::Matchers::ContainsSubstring("..."))` on a keyword. Pair each malformed case with an otherwise-identical well-formed call that must succeed, so the guard is not over-eager.

- 9.1 `interaction_type` outside `{0,1,2,3,4}` (e.g. 5, -1) → throw.
- 9.2 `center_frequency <= 0`, and NaN/Inf → throw.
- 9.3 `eps` invalid: `eps < 0`, and non-finite (NaN/Inf) → throw. Valid and must NOT throw: `eps == 0`, `eps` in `(0,1)`, `eps == 1`, `eps > 1` (the last disables `S`).
- 9.4 Geometry with a column count other than 3 (`orig`/`dest`/`fbs`/`sbs`) → throw.
- 9.5 `normal_vecN` present but not `[n_rayN, 6]`; `xprmatN` present but not `[n_rayN, 8]` → throw.
- 9.6 Full-set row-count mismatch — any of `orig`/`dest`/`fbs`/`sbs`/`no_interact`, or any provided `*_in`, disagreeing on `n_ray` → throw.
- 9.7 Compact-set row-count mismatch — any of `out_typeN`/`fbs_angleN`/`mtl_ind_fbs`/`mtl_ind_sbs`, or any provided `*_out`/`gainN`/`xprmatN`/`ray_ind`/`normal_vecN`, disagreeing on `n_rayN` → throw.
- 9.8 Pointer nullptr handling. Every pointer is null-checked before its first dereference, so a null yields the documented default or a thrown exception — never a crash. Test each pointer individually:
  - Required → throw: `orig`, `dest`, `fbs`, `sbs`, `no_interact`, `fbs_angleN`, `out_typeN`, `mtl_prop`, `mtl_ind_fbs`, `mtl_ind_sbs`, each individually null → `std::invalid_argument` naming the parameter, before any output write.
  - Optional → documented default, verified by the output (not merely "no throw"): `*_in` → state 0 (5.9); `*_out` → skip write (5.10); `gainN`/`xprmatN` → skip patch, both null = state-only (5.11); `normal_vecN` → wedge off (5.12); `ray_ind` → identity (5.12).
  - Isolate each null case in its own `SECTION`/`TEST_CASE` so a crash (rather than a throw) is attributable to the offending parameter. A crash instead of a throw is itself the failure.
- 9.9 `ray_ind` containing an index `>= n_ray` → throw (prevents an out-of-bounds read). `ray_ind == nullptr` with `n_ray != n_rayN` → throw.
- 9.10 Material index out of range — a `mtl_ind_fbs`/`mtl_ind_sbs` value, or a `*_in` state word whose `mat = w & 0x7FFF`, exceeding the material count in `mtl_prop` → throw. `mat == 0` (air) is always valid.
- 9.11 Malformed `mtl_prop` — a material entry with the wrong coefficient count or a missing required key → throw (via `mtl_validate`).
- 9.12 Output buffer sizing — the `*_out`/`gainN`/`xprmatN` buffers must end up `[n_rayN]` / `[n_rayN, 8]`. If the function requires caller pre-sized buffers, a wrongly sized output throws; if it resizes internally, it must resize to exactly the right shape. Assert whichever the implementation documents.
- 9.13 Well-formed empty batch (`n_ray == n_rayN == 0`) → clean no-op, no throw, outputs empty. Boundary, not an error.

An out-of-range or unexpected `out_typeN` value is data, not malformed IO — it is absorbed by the global-default KILL (§10.0), not an exception. Only structural/contract violations throw.

---

## 10. Coverage matrix (test → design-spec section)

| Design-spec section | Covered by |
| --- | --- |
| §5 gain model, modes | 5.7, 5.11, 5.14–5.17 |
| §6 mapping, in-place, optional args | 5.5–5.13, 9.8 |
| §7 encoding, flags | 5.1–5.4 |
| §8 passes, TR scope | 4.7, 6.9 |
| §9.1 cavity, `phi`, `L` | 6.1, 6.4 |
| §9.2 two ports | 6.2, 6.7 |
| §9.3 parallelism (magnitude) | 6.6 |
| §9.4 survival gate (`eps` param) | 6.5 |
| §9.5 `tf`/Stokes | 6.10 |
| §9.6 TR | 4.7, 6.9 |
| §9.7 clamp | 6.11 |
| §9.8 cross-pass invariance | 7.1 |
| §10.0 order, precedence, default | 4.1, 4.9, 6.8 |
| §10.1–§10.5 topology | 4.2–4.6 |
| §10.6 TR | 4.7 |
| §10.7 reflection pass | 4.8, 6.7 |
| eps modes (0 / (0,1) / >=1) | 6.5, 4.x (eps>=1), 5.13 |
| API IO contract | 9.1–9.13 |

---

## 11. Pass/fail philosophy

- State outputs are exact `short` comparisons; any mismatch is a hard fail.
- Gains use independent oracles — reimplement the spec's formulas (ITU-R Fresnel/medium loss, the Airy sum with `phi` from `medium_gain`), never library internals, so a shared bug cannot hide.
- Energy conservation is the strongest physics check and is convention-independent; prefer it over matching an exact `S`. Drive lossy cases hard so a missing return-trip factor surfaces.
- `eps` is the resolve lever: `eps >= 1` isolates the ported topology (`S` off), `eps = 0` forces full resolution, `(0,1)` exercises the gate. Test gate behavior inside clearly-separated regimes; the crossover may be bracketed.
- Input validation fails fast: malformed input throws `std::invalid_argument` before any output is written — never undefined behavior. Each negative case is paired with a passing well-formed call so the guard is not over-eager.
- Distinctive inputs (non-trivial incident `xprmatN`, distinct per-ray state, mapped `ray_ind`, VV-only vs full Jones) make replace-vs-multiply, mapping, mode, and cross-ray-bleed failures visible rather than silently passing.