# Design Specification: `ray_state_update`

A new public, batched function for quadriga-lib. It extracts the inside/outside ray-state machine currently embedded in `calc_diffraction_gain.cpp` (dispatch lines 495–907) into a standalone function, and overlays analytic thin-slab (Fabry-Pérot) resolution. It lives at the bottom of `ray_mesh_interact.cpp`, into which `quadriga_lib_material_helpers.hpp` is merged so all material-interaction code sits in one translation unit.

---

## 1. Purpose

`ray_mesh_interact` reports a per-interaction Fresnel/Jones result for one hit, with no knowledge of whether the ray is currently inside a material. `calc_diffraction_gain` wraps it in a per-ray state machine (a running "current medium" plus a one-slot "next transition" buffer) that decides, hit by hit, what the interaction *means* — entry, exit, embedded face to ignore, illegal state to kill — and corrects the gain accordingly.

`ray_state_update` is that state machine, lifted out, batched, and extended:

- It corrects the per-interaction `gainN`/`xprmatN` from `ray_mesh_interact` using tracked ray state.
- It carries the state forward (three short words per ray).
- It adds closed-form thin-slab resolution: when a ray crosses a parallel slab thin enough to matter, the trace's first-order interaction is multiplied by the Airy sum `S` so a single coefficient captures the full multiple-reflection series, instead of relying on the tracer to follow every internal bounce.

The function must slot into `calc_diffraction_gain` (which becomes a thin dispatcher; the re-wiring is a separate later task) and later into QRT. The acceptance target: a single through-path resolved QRT-style (staged `ray_mesh_interact` + `ray_state_update`) returns identical `gain`/`xprmat`/phase to the diffraction-style path.

---

## 2. Relationship to the ported code

`calc_diffraction_gain`'s dispatch (495–907) is a single `if … else if …` chain keyed on `(nH, typeH, RS, NT)` where `nH` is the hit count for the segment, `typeH` the `ray_mesh_interact` out-type, `RS` the current-medium state word, `NT` the next-transition buffer. `ray_state_update` ports this chain verbatim for the topology/gain decisions (§10), then overlays the new slab physics (§9). The port is bit-identical in its state transitions and ported gains; the only new behavior is the `S` factor, the resolved/non-parallel flags, the TR forward-kill, and the unified global-default KILL — each marked `Changed: yes` in §10.

State carries material indices, not faces. `same_materials(a, b)` reduces to `a == b` on indices, so the port is exact while decoupling the state machine from the `mtl_ind` face table.

---

## 3. Goals and non-goals

Goals: exact port of the state machine on the diffraction reference path; correct single-slab Airy resolution (lossless and lossy); energy safety (never create energy); batched, OpenMP-safe, `float`/`double`.

Non-goals (accepted approximations):

- Stacked-slab higher-order coupling. Multiple adjacent slabs are resolved with first-order inter-slab coupling. Exact in the lossy limit; an accepted approximation otherwise. The persist rule (§10.0, the resolved-ray i-i row that keeps the flag) makes the residual a benign under-count, never an energy-creating over-count.
- i-o-i air-gap cavity. The i-o-i resolution (lines 607–619) treats the gap as air; correct only when the cavity medium is in fact air. Tied to the cavity-medium identification in §9.1.
- Single-event resolution. Resolving an entire slab at its entry from the `FBS`/`SBS` pair in one call is a possible future optimization. It is not implemented and not covered by the §10 tables; the canonical scheme is multi-event (§9.1).

---

## 4. Signature

Templated on `dtype ∈ {float, double}`, explicit instantiation, `void` return, C-style out-params (no public structs). All material indices are signed `short`.

```cpp
template <typename dtype>
void ray_state_update(
    int interaction_type,                              // 0/1/2 EM, 3/4 scalar (Section 8)
    dtype center_frequency,

    // Geometry, full ray set [n_ray, 3], read at g = ray_ind[i]
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *dest,
    const arma::Mat<dtype> *fbs,
    const arma::Mat<dtype> *sbs,
    const arma::u32_vec   *no_interact,                // [n_ray] hit count nH per ray

    // Interaction outputs, compact set [n_rayN], read at i
    const arma::Col<dtype> *fbs_angleN,                // [n_rayN] incidence angle (ITU convention)
    const arma::s32_vec    *out_typeN,                 // [n_rayN] typeH
    const arma::Mat<dtype> *normal_vecN,               // [n_rayN, 6] FBS|SBS normals, optional (nullptr ok)

    const std::unordered_map<std::string, std::vector<dtype>> *mtl_prop,
    const arma::Col<short> *mtl_ind_fbs,               // [n_rayN] M1
    const arma::Col<short> *mtl_ind_sbs,               // [n_rayN] M2

    // Old state, full ray set [n_ray], read at g, READ-ONLY
    const arma::Col<short> *mtl_ind_prev_in,
    const arma::Col<short> *mtl_ind_current_in,
    const arma::Col<short> *mtl_ind_buffer_in,

    // New state, compact set [n_rayN], written at i
    arma::Col<short> *mtl_ind_prev_out,
    arma::Col<short> *mtl_ind_current_out,
    arma::Col<short> *mtl_ind_buffer_out,

    // Gain, compact set [n_rayN], IN/OUT, patched in place
    arma::Col<dtype> *gainN,                           // optional
    arma::Mat<dtype> *xprmatN,                         // [n_rayN, 8] Jones, column order VV, HV, VH, HH (re,im per entry)
    arma::u32_vec *ray_ind);                           // [n_rayN] -> [n_ray], nullptr = identity
```

`xprmatN`'s eight columns are `VV_re, VV_im, HV_re, HV_im, VH_re, VH_im, HH_re, HH_im` — the order `ray_mesh_interact` writes (lines 1079–1086), i.e. VV, HV, VH, HH. The uniform-scale and VV/HH-diagonal operations defined here are order-independent (VV is the first entry, HH the last), but the label is fixed so no future order-dependent operation inherits an error.

`ray_offset` is hardcoded to `0.001`.

---

## 5. Gain / xprmat patch model

`gainN`/`xprmatN` arrive holding `ray_mesh_interact`'s per-interaction result and are corrected in place. The caller later composes the single corrected matrix uniformly. Operations, selected per §10 row:

- IG (keep). Leave `xprmatN` as is.
- Isotropic replace / scale. A medium-loss or transition factor `g` is isotropic and scales the field amplitude by `sqrt(g)`. A row written `MED(...)` replaces the (spurious) interaction with the isotropic value; a row written `IG · MED(...)` scales the kept interaction by `sqrt(g)`.
- Apply `S`. A row written `IG · S` complex-multiplies each Jones entry by `S` (`S = S_re + j·S_im`, applied to each `(re, im)` pair), not a scaling of eight independent reals.
- KILL. Zero `xprmatN` and `gainN`.

The convention is mode-dependent (`is_scalar = interaction_type >= 3`):

| mode | `interaction_type` | `gainN` from `xprmat` | isotropic-replace layout |
|---|---|---|---|
| EM | 0, 1, 2 | `0.5 · Σ|xprmat|²` (all four entries) | `sqrt(g)` on VV and HH, zero off-diagonal |
| scalar (acoustic) | 3, 4 | `Σ|xprmat|²` = `|VV|²` (no 0.5) | `sqrt(g)` on VV only; HV, VH, HH zero |

`ray_mesh_interact` writes scalar fields into VV alone with `gainN = |VV|²` (lines 1014–1026) and EM fields with the `0.5·Σ` convention (lines 1089–1092). Recomputing a patched scalar `gainN` with the `0.5` formula (which halves it), or putting `sqrt(g)` on HH in scalar mode, breaks the `|VV|²` identity — invisible to the `gainN`-only diffraction reference (`xprmatN = nullptr`, so the convention cancels under pure scaling) but fatal once QRT carries `xprmat` for acoustic phase, which the phase-identity goal requires.

`gainN`, when present, is kept consistent with `xprmat` under the mode's convention. Either output may be `nullptr`; the other is still patched.

`S` requires `xprmatN`. `S` is complex; carrying its phase needs the Jones entries. In `gainN`-only mode (`xprmatN == nullptr`) an `IG · S` row degrades to a magnitude-only `gainN *= |S|²` with no phase guarantee, which does not meet the §13 through-path phase target. Full resolution requires `xprmatN`.

---

## 6. Index mapping

`ray_ind[i]` maps a compact-set row `i ∈ [0, n_rayN)` to a full-ray-set row `g ∈ [0, n_ray)`; `nullptr` means identity (and then `n_ray == n_rayN`).

Index-space asymmetry: geometry (`orig`/`dest`/`fbs`/`sbs`) and old state (`*_in`) are full-set `[n_ray]`, read at `g = ray_ind[i]`. Interaction outputs (`fbs_angleN`, `out_typeN`, `normal_vecN`, `mtl_ind_fbs/sbs`) and new state (`*_out`, `gainN`, `xprmatN`) are compact-set `[n_rayN]`, read/written at `i`. This split avoids compacting ~48 B/ray of geometry.

Old state is read-only: the function never writes the `*_in` arrays.

---

## 7. State encoding

Three signed `short` words per ray, bit-masked. Never `abs()` — `abs(-32768)` overflows. Read and write:

```
mat        = w & 0x7FFF            // 0 = outside / air / empty; max 32767
flag       = w & 0x8000            // sign bit
X | flag   = (short)(X | 0x8000)   // "material X with the flag set" — an OR, never -X
```

`X | 0x8000` is the only correct way to set a flag on a material; arithmetic negation `-X` does not encode it (`-5 = 0xFFFB`, whose `mat` field is `0x7FFB`, not 5).

- `mtl_ind_current` — `mat` is the current medium (0 = outside). Bit 15 = RESOLVED flag: a thin-slab reflection has been collapsed into `S` and the ray is awaiting its transparent exit.
- `mtl_ind_prev` — `mat` is the previous medium (the `r_far` medium at a resolve). Bit 15 = KNOWN-NON-PARALLEL flag (§9.3). `(short)0x8000` legitimately means "air, flag set."
- `mtl_ind_buffer` — `mat` only, no flag.

Storage 6 B/ray (≈ 900 MB at 150 M rays), half the u32 alternative.

---

## 8. Passes and geometry types

QRT calls `ray_state_update` twice per interaction, once per physical pass. The `interaction_type → geometry_type` map (`ray_mesh_interact.cpp:271–277`) is:

| `interaction_type` | meaning | `geometry_type` | scalar? | can emit TR? |
|---|---|---|---|---|
| 0 | EM reflection | 0 | no | no |
| 1 | EM transmission | 1 | no | no |
| 2 | EM refraction | 2 | no | yes |
| 3 | scalar reflection | 0 | yes | no |
| 4 | scalar transmission | 1 | yes | no |

- Reflection pass — `interaction_type ∈ {0, 3}` (geometry 0). Produces the reflected ray.
- Transmission/refraction pass — `interaction_type ∈ {1, 2, 4}` (geometry 1 for {1,4}, geometry 2 for {2}). Produces the forward ray.

TR out-codes are emitted only under `geometry_type == 2` (`ray_mesh_interact.cpp:1164`), i.e. only for `interaction_type == 2` — not for all of `{1, 2, 4}`.

---

## 9. New physics: Fabry-Pérot / Airy resolution

### 9.1 Cavity, the Airy factor, and `L`

The full internal multiple-reflection series of a parallel slab is captured by one factor:

```
S = 1 / (1 - r_near · r_far · phi^2)
```

`phi = exp(-j·β·L)` is the one-way in-slab propagation, `L = d(orig, fbs)` the one-way slab path in the canonical multi-event scheme (the internal path from the entry-side ray origin to the exit interface; there is no single-event `|fbs − sbs|` case in the tables). `r_near` is the reflection at the interface being processed (slab side), `r_far` the reflection at the opposite interface.

`phi` magnitude and phase have different sources, which matters for acoustic materials. The full loss model `medium_loss_dB` has three terms: a dielectric term (eta-based), an optional `alpha` power-law term, and the mass-law term. To keep `phi` consistent with the survival gate (§9.4), which uses the full `medium_gain`:

```
|phi(L)| = sqrt(medium_gain(slab, L))            // full loss model: dielectric + alpha + mass
arg(phi) = -(omega/c) · Re(sqrt(eta_slab)) · L   // propagation phase, dielectric only
```

The phase uses the resonance-excluded medium permittivity `eta_slab` (the `alpha`/mass terms are pure loss and add no propagation phase; `eta_resonance` is added by the helpers to the interface/Fresnel permittivity only, never the medium path — a strong pole can drive `Re(eta) < 0` and the medium path takes a real `sqrt(Re eta)`). So the resonance enters only `r_near`/`r_far`, never `β`/`phi`.

Sourcing `|phi|` from `medium_gain` is what makes `|r_near·r_far·phi²| = rho` hold (§9.7): the dielectric and `alpha` terms are linear in distance, so their loss is multiplicative (`medium_gain(2L) = medium_gain(L)²`) and `|phi²| = medium_gain(L) = sqrt(medium_gain(2L))` exactly. The mass-law term is logarithmic in distance (`mass_path = dist·cos²θ` inside a `log10`), so `medium_gain(2L) ≠ medium_gain(L)²` and no multiplicative `phi²` reproduces it. For materials with a nonzero mass term the `|phi²| ↔ rho` identity is therefore dielectric+alpha-exact and mass-approximate — an accepted approximation that slightly under-damps `S` for strongly mass-dominated slabs; the energy ledger (§13) bounds the drift.

`eta_slab` is the `current` medium at the resolve — except the i-o-i air-gap case (lines 607–619), where the cavity is air (accepted approximation, §3).

### 9.2 The two ports

The reflection and transmission ports of a parallel slab are different functions of `S`, not both `baseline · S`. This is what makes the energy ledger close.

Transmission port (forward), assembled across the two events of the multi-event scheme:

```
T_slab = t_F · phi · t_B · S
```

`t_F` is the entry-interface transmission (the IG at the entry event, an earlier call), `t_B` the exit-interface transmission (the IG at the exit event being processed), `phi` the one-way in-slab propagation, `S` the round-trip factor. At the exit the function multiplies the exit IG (`= t_B`) by `S`; the cascade `t_F · … · t_B` is already assembled through the Jones product across the two passes, so the exit-side correction is exactly `IG · S`.

Reflection port. The slab reflectance is a correction added to the front reflection:

```
R_slab = r12 + t12 · t21 · r23 · phi^2 · S
```

realized by decomposition across two reflection-pass events, not a single `IG · S`:

- Entry / order-0 front reflection (`cur == 0`, front face): bare `IG = r12`. No `S`, no flag.
- Internal / back reflection (`cur != 0`, resolvable slab): `IG · S` with `IG = r23`, and the RESOLVED flag is set. This ray then exits the front transparently (§10.0) picking up `t21`: `t12 · phi · r23 · S · phi · t21 = t12·t21·r23·phi²·S`. Summed with the bare front `r12`, this is exactly `R_slab`.

So `IG · S` is correct for the internal reflection only; the front reflection is bare.

In-slab loss/phase ownership — to confirm at tracer integration. Each one-way traversal carries amplitude (from `Im β` / `medium_gain`) and phase (from `Re β`). The invariant: each one-way is applied exactly once.

- T uses one one-way (`orig → fbs`, the down trip): `phi` once.
- R uses two one-ways — the shared down trip (`orig → fbs`) and the internal-reflection up trip (`fbs → front`, a distinct relaunched segment after the back reflection): `phi²`.

In the multi-event scheme the down trip lives on the entry→back segment (traversed once; both T and the R-internal ray branch from it at the back), and R's up trip lives on the back→front segment. The in-slab amplitude+phase on each segment must be owned by exactly one mechanism — the dispatch `MED`/`S` here, or QRT's geometric per-segment propagation — never both. The dispatch's exit rows (§10) and the resolved front-exit (§10.0) apply no `MED` for R's up trip, so if the dispatch owns in-slab loss, the up-trip factor must be added there; if QRT owns it geometrically, the entry `MED` must not also apply it. `S` itself supplies only the round-trip resonant factor `phi²` inside the series, not the explicit first-trip propagation.

### 9.3 Parallelism gate and flag-write policy

A slab needs two parallel faces — the same plane orientation — regardless of normal sign. The governing test is on the magnitude of the normal dot product:

```
faces_parallel(N_FBS, N_SBS) := |dot(N_FBS, N_SBS)| > 1 - tol      // tol = 3.8e-3 (cos ~ 5°)
```

Both `dot ≈ -1` and `dot ≈ +1` count as parallel. The sign cannot be relied on: the opposing face of the slab is frequently an i-i transition, where two coincident faces of adjacent objects carry opposing normals, and floating-point noise decides arbitrarily which of the pair is reported as FBS versus SBS. A signed (antiparallel-only) test fires correctly for a slab in isolation — where FBS is unambiguously the front face and SBS the back, giving `dot ≈ -1` — but fails randomly for layered slabs, re-emitting a valid cavity whenever the noise flips the reported normal to `dot ≈ +1`. The magnitude test fires for both; only genuine wedges and edges, whose faces sit at a real angle (`|dot|` well below 1), are rejected.

Flag-write policy. The KNOWN-NON-PARALLEL flag on `mtl_ind_prev_out` defaults to clear → resolve. The rule is uniform: run the wedge test at every entry that captures both `FBS` and `SBS` in one segment — that is, `nH==2` types 1/7/13 and `nH>2` types 1/7/13. The test: if `d(fbs, sbs) > 1e-6` (the two faces are distinct points, mirroring the type-11 separation check at line 715) and `faces_parallel(N_FBS, N_SBS)` is false, set the flag (re-emit); otherwise leave it clear (resolve).

This is why §10.1's `+flag` annotation sits on all of Branch A, B, and C uniformly: it means "run the wedge test." On the single-face `nH==1` type-1 member of Branch A it is a no-op (no `SBS` face to test, so the flag stays clear and the common single-hit slab resolves, gated downstream by survival/clamp). Type 7 is treated identically at `nH==2` and `nH>2`. The edge type-11 entry (which sets `current` from `cur==0` via `M2`, lines 710–717) is excluded from the wedge test: its edge normals are not a reliable slab pair, so it leaves the flag clear and relies on the survival gate alone.

For types 8/14 (overlapping / edge i-o exits, §10.2): they route through `IG · S` structurally, but `S` is suppressed in practice by whatever flag their entry set plus the survival gate; their S-eligibility rests on those gates, not on the geometry being a guaranteed slab. This is an accepted approximation — re-emit is always energy-safe and is the fallback whenever the gates fail.

If `normal_vecN == nullptr`, the flag stays clear → default resolve, gated only by survival/clamp.

### 9.4 Survival gate (resolve vs re-emit)

Resolve only when the internal round trip carries enough amplitude to matter. The normative round-trip amplitude is

```
rho^2 = R_near · R_far · medium_gain(slab, 2L)
```

i.e. the two interface power reflectances times the full complex `medium_gain` over the round-trip path `2L`. (The form `rho ≈ |r_near|·|r_far|·exp(-alpha·2L)` is not normative: in `medium_loss_dB` the dominant term is the eta-based dielectric loss, with the `alpha` column only an optional additive power-law; coding the literal `exp(-alpha·2L)` would under-count loss and over-resolve lossy slabs.)

Resolve iff `rho >= eps`. `eps` is tied to the engine's drop threshold: `eps ≈ drop_threshold^(1/N_max)` (≈ 0.1–0.25). A fixed thickness cap is wrong (it would re-emit thick low-loss absorbers and explode); a wavelength cap `L > ~50·λ_medium` is permitted only as a numerical backstop.

### 9.5 Effective coefficients: `tf` and Stokes consistency

`S`, `r_near`, `r_far`, and the port coefficients must all be built from one consistent set:

- Cross-interface (Stokes). From one side `r' = -r` and `t · t' = 1 - r²` (lossless). The reflection-port assembly (`t12 · t21 · r23`) and the ledger depend on these; do not recompute `t21` independently.
- Phase convention. A single sign convention for `phi = exp(-j β L)` and the Fresnel `r`, `t` throughout `S` and both ports.
- `tf` fold-in. `tf` modifies magnitude only: `|r_eff|² = tf_apply(|r|², tf)`, preserving Fresnel phase; `t_eff` follows from energy conservation, keeping phase. `S` and both ports must use the same `tf`-effective coefficients; building `S` from bare Fresnel while the ports use `tf`-adjusted values breaks `R + T + A = 1`.

### 9.6 TR (total-reflection) codes

TR out-codes `{3, 6, 9, 12, 15}` appear only for `interaction_type == 2` (§8).

- Type-2 pass + TR code → KILL the forward port (§10.6). No transmitted field under total reflection.
- Reflection pass needs no TR logic. It runs in geometry 0 and never receives a TR code; under TIR the ordinary Fresnel coefficient already has `|R| = 1` (beyond the critical angle `cos_th2` is imaginary), so the reflected ray carries full power with no special case. The internally reflected ray continues and resolves at the opposite interface through the normal machinery.

### 9.7 Clamp and scope

`|r_near · r_far · phi²|` equals `rho` (exactly for the dielectric+alpha loss, approximately for the mass-law term — §9.1), so the pole of `S` sits in the high-`rho` regime the survival gate resolves; the clamp is load-bearing. Concrete rule:

```
if |1 - r_near · r_far · phi^2| < 1e-2   ->   re-emit (do not resolve)   // caps |S| ~ 100
```

Resolve happens in the band `rho >= eps` and `|1 - r_near·r_far·phi²| >= 1e-2`; all outputs must be finite there. The clamp hands the near-pole case back to the tracer as a re-emit — precisely the high-`rho` explosion the survival gate exists to absorb. This is practically negligible (the near-pole needs lossless and grazing and on-resonance simultaneously, a measure-zero set) but is not free safety.

### 9.8 Cross-pass consistency of the resolve decision

`T_slab` (transmission-pass exit) and `R_slab` (reflection-pass internal) carry the same `S` and must make the identical resolve-vs-re-emit decision for a given slab; if the passes disagree, energy is double-counted or lost. The gate inputs — `r_near`, `r_far`, `phi`, `rho`, the parallelism flag — are functions of the slab geometry and materials only, hence pass-invariant, so the decision is necessarily identical. This is asserted in validation (§13).

---

## 10. Dispatch tables

The dispatch implements the following state lifecycle, with the 15-bit material values collapsed into classes (`· S` marks the Airy-application transitions, `× t21` the out-coupling transmission, `flag` the resolved bit):

```
                          o-i entry (1/7/13)
       ┌──────────┐  ───────────────────────────►  ┌──────────┐  ── nested o-i ──►  ┌──────────┐
       │ OUTSIDE  │                                │  INSIDE  │                     │  NESTED  │
       │ cur = 0  │  ◄──────────────────────────── │ cur = M  │  ◄── virtual i-i ── │ buf = M' │
       └──────────┘    cavity exit · S (2/8/14)    │ buf = 0  │                     └──────────┘
          ▲                                        └────┬─────┘
          │                          internal refl · S  │  │ illegal / TR
          │ transparent exit × t21       (refl pass)    │  └──────────────────────┐
          │                                             ▼                         ▼
          │                                      ┌────────────┐   refl pass ┌────────────┐
          └───────────────────────────────────── │  RESOLVED  │ ───────────►│ TERMINATED │
                                                 │  flag set  │             └────────────┘
                                                 └────────────┘

   self-loops (not drawn):  INSIDE → INSIDE   on M2M (4/5), new medium, · S
                            OUTSIDE → OUTSIDE  on false-inside / edge (2/10/11, cur = 0)
```

Notation:

- `cur = current_in & 0x7FFF`, `resolved = current_in & 0x8000`; `buf = buffer_in & 0x7FFF`. No `abs()`.
- `M1 = mtl_ind_fbs`, `M2 = mtl_ind_sbs`. `d(p,q) = qd_calc_length`.
- `MED(m, d)` = `medium_gain(m, d)`; `MED(m, ray_offset)` is a path of length `ray_offset`. `TRN(a, b)` = `transition_gain_linear(a, b)`; `IG` = incoming interaction gain; `S` = Airy factor; `KILL` = zero gain and xprmat.
- `−off` = subtract `ray_offset`, occurring on exactly three rows: clamped at the entry o-i row (line 516, `dist > off ? dist - off : dist`) and unclamped at the M2M back-segment (665) and edge-i-o-i (730) rows. The `+ ray_offset` (addition) cells carry no clamp tag.
- `(survives)` in a state cell means the ray is relaunched and continues (the dispatch's inline relaunch at 613–618 / 641–647), not a flag.
- `+flag` in a state cell means run the §9.3 wedge test on `mtl_ind_prev_out` — set the non-parallel flag iff both faces are captured, `d(fbs,sbs) > 1e-6`, and the faces are not antiparallel. It is a no-op where only one face is present (single-hit `nH==1` entries).
- `Changed` column: blank = verbatim port; `yes` = new behavior.
- State outputs default to copy-through unless a cell overrides them.

### 10.0 Dispatch order, resolved-ray precedence, global default

Exactly one path per call, selected first by pass, then by precedence:

```
REFLECTION pass  (interaction_type ∈ {0,3}):
    if resolved:  -> resolved-reflection row (KILL)
    else:         -> §10.7 reflection table

TRANSMISSION/REFRACTION pass  (interaction_type ∈ {1,2,4}):
    if TR code (out_type ∈ {3,6,9,12,15}):   -> §10.6  (KILL)        // TR wins over resolved
    else if resolved:                        -> resolved-transmission rows
    else:                                    -> topology tables §10.1–§10.5
                                                by (out_type, nH, state); unmatched -> global default KILL
```

Resolved-ray precedence (the flag was set on a slab's internal reflection in §10.7; this ray is now returning to out-couple). `iM = (out_type == 5 ? M2 : M1)`:

| pass | out_type class | gain | state out | changed |
|---|---|---|---|---|
| reflection | any | KILL — the front reflection is already summed in `S` | unchanged | yes |
| transmission, i-o (2, 8, 14) | `IG` (out-coupling `t21`) | `current_out ← 0`, clear resolved flag | yes |
| transmission, i-i (4, 5) | `IG` | `current_out ← (short)(iM \| 0x8000)` (keep resolved flag), `prev_out ← old cur` | yes |
| transmission, other (o-i, edges) | `IG` | unchanged (transparent pass-through) | yes |

The i-i row uses `iM | 0x8000`, not `-M1`: the next medium for type 5 is `M2`, and the flag is set by OR (§7). TR is not in the "other" category — it is intercepted earlier by the §10.6 check, so a resolved ray meeting a TR code is killed, never passed through.

Global default. Any transmission-pass `(out_type, nH, state)` not matched by a topology row → KILL. For `nH > 2` the source enters the 735 block, whose inside-state sub-block (759–895) has no terminal `else` — an unmatched `(out_type, state)` there leaves `power` unchanged and the ray continues. The unified KILL deliberately replaces that no-op pass-through; it is not a literal port at this point. It is safe because the only unmatched inside types are TR and a degenerate `out_type 0`: TR cannot occur on the diffraction reference path (`interaction_type 1/4`, geometry 1); an `out_type 0` inside hit is producible at exact grazing (`theta == 0`, lines 561/581) but is rare, and KILL is the safer treatment. This divergence is not exercised by the port-fidelity test.

The remaining precedence row (evaluated only on the non-resolved transmission path):

| condition | gain | state out | changed |
|---|---|---|---|
| `no_interact == 0` (nH==0) | not processed — caller applies any whole-segment in-medium loss (lines 501–508) | — | — |

### 10.1 o-i family — entry / overlapping-entry

Branch A — `(nH==1, type 1)`, `(nH==2, type 7)`, `(nH==2, type 13)` (lines 509–527). Types 7/13 at nH==2 share the nH==1 type-1 branch.

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `cur == 0` (enter) | `IG · MED(M1, d(fbs,dest) −off (clamped))` | `current_out ← M1`, `prev_out ← 0`, `+flag` | yes (flag) | 513–519 |
| `cur != 0` (nested) | `MED(cur, d(orig,dest))` | `buffer_out ← M1` | | 520–526 |

Branch B — `(nH==2, type 1)` o-i-o (lines 569–599), separate from 7/13.

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `cur == 0` | `IG` (bare) | `current_out ← M1`, `+flag` | yes (flag) | 571–584 |
| `cur != 0` (nested o-i-o) | `MED(cur, d(orig,fbs))` | `buffer_out ← M1` | | 585–599 |

Branch C — `nH > 2`:

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `cur == 0`, `buf != 0` | KILL | | | 739–740 |
| `cur == 0`, type 1/7 | `IG` | `current_out ← M1`, `+flag` | yes (flag) | 741–745 |
| `cur == 0`, type 13 | `IG` | `current_out ← M1`, `buffer_out ← M2`, `+flag` | yes (flag) | 750–755 |
| `cur != 0`, type 1/7/13 (nested) | `MED(cur, d(orig,fbs) + ray_offset)` | `buffer_out ← M1` | | 761–766 |

### 10.2 i-o family — exit / false-inside / virtual transitions

Branch A — `(nH==1, type 2)`, `(nH==2, type 8)`, `(nH==2, type 14)` (lines 528–568). The `same_mat`/`else` split is nH==1 only; the ii-oo row is types 8/14.

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `cur == 0` (false inside), all three | `IG` | unchanged | | 532–535 |
| `cur != 0`, `buf == 0` (cavity exit), all three | `IG · S` | `current_out ← 0` | yes (S) | 536–540 |
| nH==1 type 2 only, `buf != 0`, `same_mat(buf, M1)` | `MED(cur, d(orig,dest))` | `buffer_out ← 0` | | 543–548 |
| nH==1 type 2 only, `buf != 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, buf) · MED(buf, d(fbs,dest))` | `current_out ← buf`, `buffer_out ← 0` | | 549–558 |
| nH==2 types 8/14, `buf != 0` (ii-oo) | `MED(cur, d(orig,fbs)) · TRN(cur, 0)` | `current_out ← 0`, `buffer_out ← 0` | | 560–567 |

Branch B — `(nH==2, type 2)` i-o-i (lines 601–651):

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `buf == 0`, `M2 == 0` | KILL | | | 605–606 |
| `buf == 0`, `M2 != 0` (cavity exit, air gap) | `IG · S` | `current_out ← 0` (survives) | yes (S) | 607–619 |
| `cur != 0`, `buf != 0`, `same_mat(buf, M1)` | `MED(cur, d(orig,fbs) + ray_offset)` | `buffer_out ← 0` (survives) | | 625–630 |
| `cur != 0`, `buf != 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, buf) · MED(buf, ray_offset)` | `current_out ← buf`, `buffer_out ← 0` (survives) | | 631–638 |
| else | KILL | | | 649–650 |

Branch C — `nH > 2`, types 2/14 (lines 767–792) and the multi-hit false-inside:

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| type 2, `cur == 0` (false inside) | `IG` | unchanged | | 746–747 |
| type 2/14, `buf == 0` (cavity exit) | `IG · S` | `current_out ← 0` | yes (S) | 769–773 |
| type 2/14, `buf != 0`, `same_mat(buf, M1)` | `MED(cur, d(orig,fbs) + ray_offset)` | `buffer_out ← 0` | | 777–782 |
| type 2/14, `buf != 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, buf) · MED(buf, ray_offset)` | `current_out ← buf`, `buffer_out ← 0` | | 783–790 |

Branch D — `nH > 2`, type 8 (lines 806–820):

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| `buf == 0` (cavity exit) | `IG · S` | `current_out ← 0` | yes (S) | 808–811 |
| `buf != 0` | `MED(cur, d(orig,fbs)) · TRN(cur, 0)` | `current_out ← 0`, `buffer_out ← 0` | | 813–820 |
| `cur == 0` | KILL (via 756–757) | | | 756–757 |

### 10.3 M2M (i-i) family — types 4/5

`nH == 2` (652–681), `nH > 2` (793–805). `iM = (type 5 ? M2 : M1)`.

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| nH==2, `cur == 0` (illegal) | KILL | | | 654–655 |
| nH==2, `buf == 0`, `M1 == 0 or M2 == 0` (illegal) | KILL | | | 658–659 |
| nH==2, `buf == 0`, else (cavity transition) | `IG · S · MED(iM, d(fbs,dest) −off (unclamped))` | `current_out ← iM`, `prev_out ← old cur` | yes (S) | 660–668 |
| nH==2, `buf != 0` | `MED(cur, d(orig,dest))` | `buffer_out ← (same_mat(buf,M1) ? M2 : M1)` | | 670–678 |
| nH==2, else (dead — `NT==0`/`NT!=0` exhaustive after `RS!=0`) | KILL | | | 679–680 |
| nH>2, `buf != 0` (spurious) | `IG` | `buffer_out ← 0` | | 795–799 |
| nH>2, `buf == 0` (cavity transition) | `IG · S` | `current_out ← iM`, `prev_out ← old cur` | yes (S) | 800–804 |
| nH>2, `cur == 0` | KILL (via 756–757) | | | 756–757 |

### 10.4 Edge o-i-o — type 10

`nH == 2` (682–707), `nH > 2` (822–862). No `S` on any row (graze, not a slab).

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| nH==2, `cur == 0` | `IG` | `current_out ← 0` | | 684–688 |
| nH==2, `cur != 0`, `same_mat(M1, M2)` | `MED(cur, d(orig,dest))` | unchanged | | 691–696 |
| nH==2, `cur != 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, M1) · MED(M1, d(fbs,dest))` | `current_out ← M1` | | 697–705 |
| nH>2, `cur == 0` (stay outside) | `IG` | unchanged | | 748–749 |
| nH>2, `cur != 0`, `buf == 0`, `same_mat(M1, M2)` | `MED(cur, d(orig,fbs) + ray_offset)` | unchanged | | 832–836 |
| nH>2, `cur != 0`, `buf == 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, M1) · MED(M1, ray_offset)` | `current_out ← M1` | | 837–843 |
| nH>2, `cur != 0`, `buf != 0`, `same_mat(buf, M1)` | `MED(cur, d(orig,fbs) + ray_offset)` | `buffer_out ← 0` | | 846–852 |
| nH>2, `cur != 0`, `buf != 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, buf) · MED(buf, ray_offset)` | `current_out ← buf`, `buffer_out ← 0` | | 853–860 |

The `nH > 2, cur == 0` guard inside this branch (824–828) is dead (block reached only when `RS != 0`); the reachable `cur == 0` path is the multi-hit RS==0 stay-outside row above.

### 10.5 Edge i-o-i — type 11

`nH == 2` (708–734), `nH > 2` (863–893). No `S` on any row.

| state | gain | state out | changed | ref |
|---|---|---|---|---|
| nH==2, `cur == 0` | `IG` | `current_out ← (d(fbs,sbs) > 1e-6 ? M2 : 0)` | | 710–717 |
| nH==2, `cur != 0`, `same_mat(M1, M2)` | `MED(cur, d(orig,dest))` | unchanged | | 720–725 |
| nH==2, `cur != 0`, else | `MED(M2, d(fbs,dest) −off (unclamped))` | `current_out ← M2` | | 726–732 |
| nH>2, `cur == 0` | KILL (via 756–757) | | | 756–757 |
| nH>2, `cur != 0`, `buf == 0`, `same_mat(M1, M2)` | `MED(cur, d(orig,fbs) + ray_offset)` | unchanged | | 875–879 |
| nH>2, `cur != 0`, `buf == 0`, else | `MED(cur, d(orig,fbs)) · TRN(cur, M2) · MED(M2, ray_offset)` | `current_out ← M2` | | 881–887 |
| nH>2, `cur != 0`, `buf != 0` (spurious) | `IG` | `buffer_out ← 0` | | 889–893 |

The `nH > 2, cur == 0` guard (865–871) is dead for the same reason as type 10; the reachable path is the KILL above. The `buf == 0, else` row (881–887) makes no buffer write in the source, so its state is copy-through (already 0 since `NT==0`); listed `unchanged`.

### 10.6 TR codes — `interaction_type == 2` only

| condition | gain | state out | changed |
|---|---|---|---|
| TR out-code (any state) | KILL (no forward port) | unchanged | yes |

The `yes` marks the new explicit refraction-pass forward-port handling; the kill itself matches the global default. TR is intercepted before the resolved-ray and topology checks (§10.0). Reflection-pass TIR needs no row (§9.6).

### 10.7 Reflection pass — `interaction_type ∈ {0, 3}` (non-resolved)

| condition | gain | state out | changed |
|---|---|---|---|
| entry / order-0 front reflection (`cur == 0`), or any ordinary reflection / re-emit | `IG` (bare Fresnel; naturally `|R| = 1` under TIR) | copy-through | |
| internal / back reflection of a resolvable parallel slab (`cur != 0`, gates pass) | `IG · S` | copy-through, set resolved flag on `current_out` (`mat \| 0x8000`) | yes |

The two reflection-pass events realize `R_slab = r12 + t12·t21·r23·phi²·S` (§9.2): the front event contributes bare `r12`, the internal event contributes `r23·S` and flags the ray, which exits the front transparently (§10.0, transmission pass) picking up `t21`. A resolved ray reaching the front in the reflection pass is killed by §10.0 (the would-be second reflection is already in `S`). There is no `IG · S` on the front reflection.

---

## 11. New helpers (relocated into `ray_mesh_interact.cpp`)

- `same_materials(a, b)` and `transition_gain_linear(a, b, …)` — relocated as material-index variants (the `mtl_ind` argument is dropped).
- `faces_parallel(N_FBS, N_SBS)` — the signed antiparallel test of §9.3.
- `slab_airy_factor(...)` — computes `S` with the survival gate (§9.4), the clamp (§9.7), and the `tf`-effective coefficients (§9.5).

---

## 12. Instantiation and threading

Explicit `float` and `double` instantiation. OpenMP-safe: each ray reads its own `*_in` rows and writes its own `*_out` rows; no shared mutable state. Compaction is internal.

---

## 13. Validation

- Port fidelity. With `S` disabled (gate forced to re-emit), `ray_state_update` reproduces `calc_diffraction_gain`'s state and gain bit-for-bit across the synthetic geometries. The global-default KILL divergence (§10.0) is not on this path and is asserted separately.
- Through-path identity. A single resolved through-path matches diffraction-style `gain`/`xprmat`/phase (requires `xprmatN`, §5).
- Energy ledger. For a parallel slab, `R_slab + T_slab + A = 1` (lossless: `A = 0`) within tolerance, `R`/`T` from the two passes. Primary `S` check; depends on the reflection-port decomposition (§9.2), the in-slab one-way-each invariant (§9.2), and the Stokes/`tf` consistency (§9.5). Because the in-slab one-way ownership (dispatch vs QRT) is settled at integration, this is the only guard against a missing return-trip factor — the test must drive a strongly lossy slab (large `A`), not just a lossless one, so a dropped up-trip `MED`/`phi` actually breaks the balance.
- Cross-pass invariance. For a fixed slab, the reflection and transmission passes make the same resolve/re-emit decision (§9.8); assert the gate inputs and outcome match.
- Scalar convention. For `interaction_type ∈ {3,4}`, a patched scalar interaction keeps `gainN = |VV|²` with field in VV alone (§5).
- Energy safety. The resolved-flag persist rule yields a benign under-count for stacked slabs, never `R + T + A > 1`.

The companion blind test spec (`test_ray_state_update_spec.md`) exercises every §10 row and the gates with synthetic inputs.
