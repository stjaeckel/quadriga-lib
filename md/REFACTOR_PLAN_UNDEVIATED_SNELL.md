# Refactor Plan: Undeviated-Path Snell Correction (VBS + Segment Architecture)

*Implementation plan for `SPEC_UNDEVIATED_SNELL_CORRECTION.md` (rev 2). Touches
`ray_mesh_interact.cpp`: `ray_mesh_interact`, `ray_state_update`, `Material::interact_with`,
`Material::slab_airy_factor`, and new anonymous-namespace helpers.*

## Principles

- **Five development stages, two independent merges.** Stages 1–2 are non-breaking and merge to
  main on their own. Stages 3–5 are the physics change and **merge as one atomic unit** — the
  intermediate states are incorrect (§Sequencing).
- **Each stage has a single gate.** A stage is done only when its gate passes. Stages 1–2 gate on
  *bit-exact* parity with the existing Catch2 suite plus no benchmark regression. Stages 3–5 gate
  on re-derived coefficient oracles and the new test groups (spec §12); the old hack oracles
  retire with the hack.
- **Rollback tag.** Tag the pre-change commit (`pre-snell-vbs`). The spec's "previous behavior" is
  this tag; there is no compatibility mode in the code.
- **Spec mapping.** The spec's own two-phase note (helpers+outputs, then physics) expands here:
  spec phase 1 = plan stages 1–2; spec phase 2 = plan stages 3–5.

## Stage overview

| Stage | Scope | Breaking | Merge | Gate |
|---|---|---|---|---|
| 1 | Extract helpers; refactor `ray_mesh_interact` onto them | No | Independent | Catch2 bit-exact + benchmark flat |
| 2 | Add `path_dirN`, `ray_indN` outputs to `ray_mesh_interact` | No (append) | Independent | Existing bit-exact + new output tests |
| 3 | `interact_with`: remove d2l, unify tf on EM, TF-TIR fix, combined M2M tf | Yes | Atomic with 4–5 | Per-interface coefficient oracles + gain identity |
| 4 | `ray_state_update` signature extension | Contract | Atomic with 3,5 | Compiles; existing tests pass with inert params |
| 5 | `ray_state_update` VBS + `acc_dist` logic; `slab_airy_factor` walk-off | Yes | Atomic with 3–4 | Full spec §12, anchored on segmentation invariance |

---

## Stage 1 — Extract helpers (non-breaking)

**Goal.** Move the inline geometry/polarization/charge code of `ray_mesh_interact` into
anonymous-namespace helpers it then calls, with zero behavior change, so stage 5 can reuse the
identical conventions. This is the spec's DRY requirement (§5).

**Changes (anon namespace of `ray_mesh_interact.cpp`):**

- `qd_reflect(û, n̂) -> d̂` — mirror direction. Source: the central `geometry_type == 0` branch
  (`FD = OF - 2(OF·N)N`) and the ray-tube vertex reflection.
- `qd_refract(û, n̂, eta, cos_theta2) -> d̂` — Snell direction, including the `Re(cos_theta2)`
  treatment and the final normalization. Source: the central refraction branch and the ray-tube
  vertex refraction.
- `qd_polbasis(û_in, û_out, n̂, cTE, cTM, is_scalar) -> xprmat[8]` — the incoming Q-basis (from
  `û_in`), outgoing U-basis (from `û_out`), the VV/HV/VH/HH sandwich, the `do_base_transform`
  degenerate guard, and the scalar single-coefficient write. Source: the EM xprmat assembly block.
- `qd_hit_charges(...)` — the §4.4 per-hit in-medium charge composition (the `ray_starts_inside`
  `medium_gain` over the origin–FBS length, the transmissive `medium_gain` over `ray_offset`).

**Unchanged.** Signatures, outputs, every numeric result. `qd_vbs` is *not* introduced here (it is
new in stage 5).

**Gate.** Existing Catch2 suite **bit-exact** for float and double. Benchmark suite shows no
regression — the helpers must inline in the hot loop (verify by timing; if needed, force inline).
The polarization sandwich is the highest-risk extraction (many temporaries, sign/order); the
bit-exact gate is precisely what catches a slip.

**Merge.** Independent. Land before stage 2.

---

## Stage 2 — New outputs on `ray_mesh_interact` (non-breaking append)

**Goal.** Expose the two values the state machine consumes, both already computed internally
(spec §4.1).

**Changes (`ray_mesh_interact` signature, appended optional params):**

- `arma::Mat<dtype> *path_dirN = nullptr` — write the normalized continuation direction (the `FD`
  vector the function already builds: mirror for type 0/3, incoming for type 1/4, Snell for type
  2), at the compact index `i_rayN`.
- `arma::u32_vec *ray_indN = nullptr` — the inverse of the internal `output_ray_index` (currently
  built and discarded): 0-based, order-preserving, `ray_indN[output_ray_index[i_ray]-1] = i_ray`.

**Unchanged.** All existing outputs bit-for-bit; d2l, tf, and TIR behavior are still the *old*
behavior (changed in stage 3). So the `path_dirN` TIR-row contract is provisional here and
finalized in stage 3/5.

**Gate.** Existing suite bit-exact (new outputs don't perturb old ones). New tests: `path_dirN`
equals `origN`→`destN` for types 0/2/3 and reflections, the Snell direction for type-2 refraction,
the incoming direction for type-1/4; unit norm. `ray_indN` round-trip and ordering against the
surviving-ray set (spec §12.1 partial, §12.11).

**Merge.** Independent. After this, both non-breaking stages are on main; the physics block (3–5)
develops against a clean baseline.

---

## Stage 3 — `interact_with` physics (breaking)

**Goal.** Remove the d2l hack and unify the tf rule, so the per-interface coefficients are honest
at the real angle (spec §2.1, §4.3, §10). **Develop on the physics branch; do not merge alone** —
see Sequencing.

**Changes (`Material::interact_with`):**

- Delete the EM dense-to-light pass-through block and the `dense2light` out-parameter; remove all
  `dense2light` callers.
- Unify tf on EM type 1: `R_eff = apply_tf(reflection_gain)`, forward gain `= 1 - R_eff`. The
  general rule is "tf shifts reflection energy into transmission"; TIR is its edge case.
- **TF-TIR coefficient fix.** Under TIR, `reflection_gain == 1`, `refraction_gain == 0`, so the
  existing `if (refraction_gain > 0)` rescale is skipped and the forward port would carry gain but
  zero coefficients. When tf opens a forward port and `refraction_gain == 0`, write flat undeviated
  coefficients `cTE = cTM = sqrt(forward_gain) * sqrt(interface_gain)`, so
  `½Σ|x_ij|² == gain` holds and the path is polarization-preserving and undeviated.
- **Combined M2M tf rule** (spec §10) for interfaces where both media define tf, replacing the
  hit-side owner selection: with `tf⁺ = max(tf,0)`, `tf⁻ = max(-tf,0)`,
  `R_leak = R0·(1-tf_A⁺)(1-tf_B⁺)`, `R_eff = R_leak + (1-R_leak)·max(tf_A⁻, tf_B⁻)`. Reduces to
  `apply_tf` at any air boundary; symmetric; in `[0,1]`.

**Effect.** EM type-1 through-slab totals shift from the gate's one-sided `(1-R)` toward the
physical composition (completed by stage 5); EM TIR rows now carry a tf-forward port. Release note,
not a migration (EM tables hold physical constants; §10).

**Gate.** Per-interface coefficient oracles re-derived: Snell-pair reciprocity at `θ_t` (§12.3),
the gain↔xprmat identity including the TF-TIR fix (§12.1), the combined M2M tf rule under
randomized FBS/SBS order and at air boundaries (§12.8). The full slab energy ledger is **not** a
stage-3 gate — it needs the state-machine composition and is gated in stage 5.

**Merge.** Atomic with 4–5.

---

## Stage 4 — `ray_state_update` signature (prerequisite for 5)

**Goal.** Extend the signature to the spec §4.2 shape, with the new parameters accepted but inert,
so stage 5 wires logic into a stable interface.

**Changes (`ray_state_update`):**

- Move `normal_vecN` up next to `fbs_angleN`; it becomes the VBS plane normal (required for the
  corrections — keep it nominally optional in stage 4 so behavior is unchanged, enforce
  required-ness in stage 5).
- Add inputs `path_dir_prev` (`[n_ray,3]`), `acc_dist_in` (`[n_ray]`).
- Add in/out `path_dirN` (`[n_rayN,3]`) and output `acc_dist_outN` (`[n_rayN]`).
- Add `ray_indN`.
- Delete the stale `excess_delayN` line from the doc block — it is not in the real signature and
  carries nothing.
- Update the C++ call sites (`calc_diffraction_gain`, the SBR tracer) to pass the new args, null
  where not yet used. No language bindings exist yet, so binding exposure follows at leisure.

**Unchanged.** All numeric behavior — the new params are inert until stage 5.

**Gate.** Compiles for float and double; existing `ray_state_update` tests pass unchanged with the
new params null/inert.

**Merge.** Atomic with 3,5.

---

## Stage 5 — `ray_state_update` VBS + `acc_dist` logic

**Goal.** Implement the corrected physics: VBS at the real or virtual plane, interface coefficients
at `θ_t`, in-medium loss and Airy length accumulated and charged once per layer, the walk-off
phase, and the `path_dirN` correction (spec §3, §5, §6).

**Changes:**

- New helper `qd_vbs(P0, d̂_p, F, n̂) -> (VBS, d_v, cosθ_t)` (spec §3.1) with the grazing/`s≤0`
  guards, in the anon namespace; reuse `qd_refract`/`qd_reflect` for the continuation.
- VBS plane selection per event: real face (`F = fbs`, `n̂` from `normal_vecN`, FBS/out-4 vs
  SBS/out-5) at a crossing; virtual plane (`F = dest`, `n̂` = carried entry normal from
  `normal_vecN`) at `nH == 0`.
- Split the in-medium charge from the interface charge:
  - `RPL` (via `qd_polbasis` at `θ_t`, complex `rsu_scale` for the folds — *not* the scalar
    `rsu_replace`) writes the interface contribution, in-medium magnitude deferred.
  - `ACC`: `acc_dist += d_v`.
  - `CLOSE` at every event that changes the current medium (i-o exit, i-i transition, terminal
    inside): charge `medium_gain(acc_dist)`, fold `S(θ_t, acc_dist)` and `e^{-jΔ}`, reset
    `acc_dist`. When `acc_dist` is absent, close over the event's own `d_v` (relaunch-aligned mode).
- Apply the dispatch deltas of spec §6, including the new `nH == 0` row.
- `path_dirN` correction: VBS continuation at i-i/M2M; `origN`→`destN` at i-o exit (§6/8.5).
- `slab_airy_factor`: apply the §3.4 walk-off phase
  (`arg φ = -k0·n_re·dist·cos²θ`) and **migrate all six call sites** from `distance(orig,fbs)` to
  `acc_dist`. The walk-off fix must land together with the distance migration or it double-counts
  the angle.
- Enforce `normal_vecN` required; validation per spec §8.

**Gate.** Full spec §12, anchored on:
- **§12.7 segmentation invariance** — one SBR traversal vs. the collinear diffraction spine cut
  into segments whose boundaries land inside bodies; bit-identical for any scene in the
  forced-resolve config (mass-law materials, the 1.5 mm clamp straddled, multi-layer i-i, a
  reflection round-trip). This is the primary correctness gate for the whole change.
- §12.4 transfer-matrix ledger (oblique, lossless/lossy, stacked `R+T≤1`).
- §12.5 Airy walk-off vs. closed-form and bounce-ray sum.
- §12.6 insertion phase vs. transfer-matrix insertion phase; no magnitude double-count.
- §12.10 pruning-order hazard (nonzero oblique type-1 when pruning after this call).
- §12.9 acoustic rigid wall bit-compatibility with the existing calibration.
- Benchmark: the hot loop now does the VBS construction; confirm acceptable cost.

**Merge.** Atomic with 3–4.

---

## Sequencing and landing policy

**Why 3–5 are atomic.** Removing the d2l gate (stage 3) makes an oblique type-1 exit return honest
`T = 0`. The repair — replacement by `t_exit(θ_t)` at the exit row — lands in stage 5. The current
state machine does not perform that replacement; it only folds `S`. So any state between stage 3
and stage 5 silently drops oblique type-1 transmissions. The same applies to the `TRN` calls inside
`ray_state_update`, which would evaluate the de-gated `interact_with` at the wrong (geometric) angle
until stage 5 supplies `θ_t`. Therefore:

- Land **stage 1**, then **stage 2**, each on main behind its own bit-exact gate.
- Develop **stages 3, 4, 5** on one physics branch, in that order, each with its own internal gate,
  and **merge the three together**. Do not merge stage 3 or 4 to main alone.
- Keep the `pre-snell-vbs` tag for rollback.

**Suggested branch structure.** `main` ← stage 1 ← stage 2 ← (branch `snell-vbs`: stage 3 → 4 → 5)
→ squash-or-merge `snell-vbs` to `main` once §12.7 passes.

## Risk register

- **Polarization extraction (stage 1).** Subtle sign/order error in the sandwich. Mitigation:
  bit-exact gate; extract incrementally (reflect/refract first, then polbasis).
- **Hot-loop cost (stages 1, 5).** Helpers must inline (stage 1); the VBS adds work per event
  (stage 5). Mitigation: benchmark gate on both.
- **`acc_dist` ownership at terminal-inside rays (stage 5, spec Q2).** A ray that ends inside a body
  has no closing crossing; its accumulated leg must charge at its terminal event. Mitigation:
  explicit dispatch coverage and a §12.7 sub-case that terminates inside.
- **Combined M2M tf ledger (stage 3, spec Q1).** Confirm port complementarity before relying on the
  rule; covered by §12.8 but the full-ledger confirmation is a stage-5 cross-check.
- **Forward-only re-emit (diffraction).** The module must run `eps = 0` with `parallel_ok` forced
  on; a re-emit would lose energy with no reflection pass. The §12.7 equality only holds with both
  sides in this config. Document in the caller, not the library.
