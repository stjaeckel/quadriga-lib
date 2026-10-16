# Wave physics in `ray_mesh_interact.cpp` — settled conventions and their guards

This document states what the thin-sheet composite and the traced-solid series do for **wave media**,
which test case asserts each statement, what is accepted as a limitation, and how changes are
measured. It is the reference for the wave side and stands on its own. The mass-law add-on is in
`md/acoustic.md`.

## 0. Scope, boundary and first step

**Wave media** are all materials without the mass column (`m = 0`): dielectrics, conductors,
magnetic media, porous absorbers, rigid permittivities of the acoustic mapping that carry no `m`.
EM types (0 reflect, 1 transmit, 2 refract) and scalar types (3, 4, 5) run the same physics; the
scalar types use the TE / admittance form. **Every material with `m > 0` is acoustic** and is out
of scope here; section 9 lists where that add-on attaches. The M9 leaf cases in
`test_acoustic_leaf.cpp` are `[acoustic]` although two of them evaluate wave media only and the others
wave media on a mass-law wall: they guard the acoustic route of `md/acoustic.md` 2.8. So does *Sheet between two
bodies - nothing is created and the wave left in air inside the far body is killed*, which evaluates wave media on an
`m = 0` wall and guards the kill of section 6 for the scalar types and for EM (type 1 with its reflect pass, TE and
TM, the faces as dielectrics).

The wave side has **one small open convention**, the path the `alpha` column is charged over in a
layer that tunnels (section 8a); everything else is settled. It is the regression guard while the
acoustic side is rebuilt: a change made for acoustics sits behind `m > 0` and moves no wave row.

First step of every session, from the repository root:

```
cmake --build build_linux --parallel
./build_linux/test_bin "[wave]" > test_wave.txt          # All tests passed (165110 assertions in 179 test cases)
./build_linux/test_bin "[acoustic]" > test_acoustic.txt  # All tests passed (39371 assertions in 54 test cases)
./build_linux/test_bin > test.txt                        # All tests passed (1314030 assertions in 1140 test cases)
```

The three baselines are untracked (`.gitignore`: `test*.txt`). A change is diffed against them. The
wave tables that print coefficients with every digit also assert them against
`tests/data/wave_twins.tsv` at a relative 1e-12; after a change meant to move them:
`rm tests/data/wave_twins.tsv; QD_TWIN_WRITE=tests/data/wave_twins.tsv ./build_linux/test_bin "[#test_wave_state_update_branch]" -#`.

Test files: `test_wave_mesh_interact.cpp`, `test_wave_progress.cpp`, `test_wave_sheet_vs_solid.cpp`,
`test_wave_state_update_{spec,sheet,chain,branch}.cpp`, with the shared helpers in
`tests/catch2_tests/ray_test_helpers.hpp`. Case names below are quoted as they are in the files;
a name followed by "(wave half)" is a case whose acoustic rows moved to "<name> - the mass-law part".

## 1. The two functions and their jobs

- **`qd_stack_resolve`, the solver,** builds the composite of a sheet or an ordered stack of sheets
  outright, both ports, absolute coefficients. A sheet has no world-space thickness, so its internal
  series is summed in one event. Every sheet event of both public functions goes through it.
- **`qd_slab_resolve`** returns the series factor `S = 1/(1 − rn·rf·φ²)` of a traced solid and nothing
  else; the port the caller holds already carries its own Fresnel coefficient. A sheet or stack named
  as a mirror of that series is resolved by the solver.
- **`ray_state_update` re-resolves every sheet crossing** when the state is kept, so its composite is
  the one that reaches the output. `ray_mesh_interact`'s own composite serves callers without state
  and agrees with the redo wherever it can know what the redo knows.

Asserted in: *One solver - a sheet and a stack are one sum and no mirror is clamped*; *Unentered
branches - ray_mesh_interact's own glued composite against the redo* (wave half).

## 2. Faces, energy and the solver

**Admittance and index.** Per medium `z = sqrt(ε/μ)`, `n = sqrt(ε·μ)`; every cosine follows from the
transverse invariant `ε·μ·(1 − c²)`, conserved across every boundary. Tilted admittances
`η_TE = z·c`, `η_TM = z/c`. The library's TM reflection is minus the admittance-form one and its TM
transmission carries an extra `c₁/c₂`.

**Energy convention.** Every interface coefficient is a per-channel normal-flux amplitude: the raw
Fresnel amplitude times the square root of `Re(z₂c₂)/Re(z₁c₁)` (TE, scalar) or
`Re(z̄₂c₂)/Re(z̄₁c₁)` (TM), formed by `Material::flux_amp`. For a lossless incidence medium the
transmittance is `1 − R` per channel at any angle; all four transmission-class types report the same
energy; consecutive crossings that share a medium telescope when both are formed at the same cosine.
Asserted in: *Diagnostic - Interface energy split per polarization*; *Diagnostic - Transmission
convention baseline, single interface*; *ray_state_update - energy conservation of a lossless slab*.

**The solver is one recursion.** Each boundary is a two-port in raw Fresnel amplitudes, each coherent
layer the factor `φ = exp(−j·k₀·n·c·d)` (times the `alpha` column over `d/|Re c|`), evaluated from the
far end:
`G_j = r_j + t_j·tb_j·G_{j+1}·φ²/(1 + r_j·G_{j+1}·φ²)`, `T_j = t_j·φ·T_{j+1}/(1 + r_j·G_{j+1}·φ²)`.
That is the characteristic matrix in another basis and the Airy pair at one layer. Amplitudes
propagate; the conversion to normal flux is applied once, at the outer pair, transmit port only.
**Nothing is clamped**: a reflection above one seen from a lossy medium is the Fresnel value, a small
denominator is not limited, an opaque layer transmits at its own level. Exactly at the critical
angle of a lossless sheet the pair is 0/0 and the solver declines.
Asserted in: *One solver*; *Layered stack - the characteristic-matrix reference against the
single-layer oracles*; *Stack with tf - the scattering form against its reference, produce mode and
itself*; *ray_state_update - resolve clamp near the Airy pole*.

**The total-reflection verdict** (`Material::interact_with`): total reflection where
`sqrt|ε₁μ₁/ε₂μ₂|·sin θ ≥ 1` **and the second medium is lossless** (`Im(ε₂μ₂) = 0`, resonance included).
A lossy second medium has no critical angle at any frequency: the port is the Fresnel crossing in
normal flux, the direction is `qd_refract`'s real projection, no total-reflection bit, the tube of
`ray_mesh_interact` is not closed by a vertex ray, the outer pair of a sheet in front of it is open.
Decided on the medium, never on a value, so a lossy medium has one verdict for every frequency.
The real-angle and the complex-cosine spelling of the test are kept separately (rounding at a
matched interface depends on it).
Asserted in: *Lossy verdict - a lossy second medium has no critical angle* (wave half);
*ray_progress - a lossy medium has no critical angle and a band takes one path*.

**The decaying root.** Where a refraction root's argument is negative and real to within 1e-9 of its
real part, `qd_evanescent` forces the decaying branch: five call sites, one of them the solver's
`root`, which every layer cosine goes through. Asserted in: *Exits and verdicts -
every exit at the cosine the wave has there*.

## 3. `tf` on wave faces

`tf` moves power between the two ports of a boundary, owed by the pair of media that form it. On
wave faces there are two rules, chosen on the verdict of the face, never on the invariant:

- **The transformer** (`Material::tf_face`) where a wave is on both sides: the face of a coherent
  sheet or layer in the solver, a bare boundary in `interact_with` where the verdict is open, the
  mirror `fresnel_r` of `qd_slab_resolve`. An ideal transformer of real ratio set by the pair rule
  on the modulus of the admittance ratio: normal flux is continuous, the face creates nothing. Where
  the ratio is real it is the pair rule on the reflectance. A pair without `tf` is the raw face; a
  raw reflection of exactly ±1 is left raw; at a match the side that forces less sees the forced
  reflection negative.
- **The pad of a closed face** (`Material::closed_face`) where a wave is on one side only: a matched
  resistive pad in front of the reactance behind the face, formed on the modulus of the incidence
  immittance. It passes what it absorbs; in front of a lossless medium it is `sqrt(R')` on the
  Fresnel reflection and `1 − R'` of the flux, flat and undeviated; at grazing of the far wave it is
  the transformer's limit, so a face has no step at its critical angle beyond the square-root cusp.
  `tf = −1` and a pair without `tf` pass nothing.

A sheet of air on a `tf` wall is the bare wall; a sheet and its traced solid are one wall; a summed
series and a followed one are one wall; the two ports of an event add to at most one seen from a
lossless medium.
Asserted in: *Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the
invariant* (wave half); *Bare tf - a bare face with tf is a transformer where it is open and a flat
leak where it is closed* (wave half); *Neighbour tf - each face of a sheet is owed by the pair of
media that form it*; *Rare sheet - a coherent sheet with tf that is rarer than the invariant passes
no more than it received* (wave half); *Flat branch - a coherent sheet rarer than the invariant
without tf tunnels on the characteristic matrix* (wave half); *Closed face - a coating met from
inside a dense solid*; *Scalar closed face - a lossy box on a rigid wall reflects no more than it
received and is its sheet*; *Lumped columns - the transmission factor on a sheet, both ports*;
*ray_progress - a coherent sheet with tf creates nothing and has no gate on the invariant*;
*ray_progress - a traced solid with tf creates nothing*.

**A coherent sheet is a wave medium at every angle:** denser than the invariant it propagates,
rarer it tunnels and decays with its thickness, lossless or lossy, with or without `tf`. A rigid
permittivity without the mass column tunnels too; that is wave physics and not acoustics (acoustic
materials carry `m`).

## 4. The other lumped columns on wave media

- **`att` and the coincidence term** (`interface_gain`): owed once per medium entered. Sheet
  transmit port: `sqrt(G_sheet·(far is new ? G_far : 1))`. Reflect port: the sheet's gain once, on
  what entered the sheet and came back (the solver's second recursion `Ge`), never the far medium's.
  A sheet in air and its traced solid are one wall on both ports. A negative total loss returns a
  gain of one. Asserted in: *Lumped columns - interface gain on the sheet, glued and embedded
  paths*; *Sheet reflect port - where the layer lies and what the echo pays*; *Acoustic -
  interface_gain penetration and coincidence*; *Ray-Mesh Interact - Penetration loss frequency
  scaling*.
- **`alpha`**: dB per meter of path, a real one-way factor on `φ` over `d/|Re c|`; where the wave
  does not advance in a layer (`Re c = 0`) the layer is opaque to it. Asserted in: *Ray-Mesh Interact
  - Alpha in-medium absorption (medium_gain)*; *Stack with tf*.
- **`resF`, `resQ`, `resS`**: a Lorentz pole added to ε in `Material::eta()`. It is part of the
  medium at every face, cosine, Snell ratio, verdict, layer phase, traversal, displaced medium,
  `medium_gain` and `refractive_index`. A lossy pole is a conductivity of that size, bit for bit; a
  resonant medium is lossy for every verdict at every frequency. Asserted in: *Lumped columns - a
  permittivity resonance is part of the medium on the sheet paths*; *Resonant medium - a permittivity
  resonance is part of the medium a wave crosses and travels in*; *ray_progress - a resonant solid is
  its medium at every frequency of a band*; *Resonance - the public index and in-medium gain carry
  the permittivity pole*. (A pole on a material with `m > 0` is acoustic item A1.)
- **Permeability** is carried everywhere ε is. Asserted in: *Permeability - the sheet, stack and
  glued paths*; *Permeability - the traced solid and a magnetic layer in a stack*; the
  *Ray-Mesh Interact - Permeability …* cases.

## 5. Where things are evaluated: cosines, directions, geometry

**The incidence cosine.** `abs_cos_theta` is a real angle, what a face hit supplies. A wave inside a
traced medium has a complex cosine; the tracer's direction carries only its real projection.
`EXACT_IN_SLAB_COSINE` inverts `qd_refract`'s map by Newton until `|Δu| ≤ 1e-15·u` (3 to 50 steps)
and returns the exact complex cosine. It declines (returns 0, consumers keep the real projection)
where the wave was not bent at the face behind it, the medium is not a real material, the faces are
known not parallel, or entry and slab are matched. Behind a sheet it solves for the real transverse
invariant with no entry medium named; behind a solid it names the solid unless the real-invariant
flag (bit 14 of the previous-medium word) says that face was evaluated at a recovered cosine.
Consumers on that one gate: the Airy boundaries, the slab propagation factor, the coefficient and
verdict in `REPLACE_BY_VBS`, the traversal in `SCALE_BY_MEDIUM`, the cosine `SHEET_RESOLVE` hands
the solver, the verdict of a sheet's outer pair.
Asserted in: *Recovery gate - the hit code a traced run carries*; *Exit boundary - the port at the
recovered angle*; *Sheet in a medium - the composite at the exact in-medium cosine*; *Recovery behind
a sheet - the leg after a sheet crossing at the exact in-medium cosine*; *Recovery behind a lossy
solid - EM, a second transition and a sheet inside the host*; *Recovery behind a lossy solid - a face
that was evaluated at a real angle keeps the solid named*; *Measurement - a host entered from a lossy
solid across a bare face*.

**The glued-sheet flux conversion.** The near-to-far obliquity factor is applied once, on the
transmit port, only where near and far differ (glued entry, glued exit), at the far cosine written
from the near-side sine. Asserted in: *Glued sheet - the near-to-far flux conversion*; *Reciprocity
- glued entry against glued exit*.

**The two tracers.** Types 1 and 4 keep the segment on the incoming direction (geometry
frequency-independent, the physical direction in `path_dirN`). Types 2 and 5 bend the segment at
the base frequency; a leg that follows such a bend carries the **bent flag** (bit 14 of the
current-medium word), written where a medium is entered with type 2 or 5 and the hit code carries
no total-reflection bit. On a bent leg every consumer of the recovered cosine runs as for the
undeviated tracer. A leak is not a bend. Asserted in: *Refracting tracer - a traced bend is known
from the state and the leg is exact* (wave half).

**Directions written.** At a transition and at an exit into air the direction is the one a single
refraction from air writes (the Snell ratio times the length `N` the projection lost), so behind
parallel faces a wave leaves parallel to how it arrived, at any depth. Where the recovery declines
both exits apply the real-angle map. A closed face hands on the direction of the leak: where the
transverse wavenumber has a direction in air (invariant below one) that direction, otherwise the
direction the wave has in the medium it leaves (`EXIT_DIRECTION`, `LEAK_DIRECTION`). Asserted in:
*Direction - a ray leaves a parallel slab parallel to how it arrived* (wave half); *Exits and
verdicts - every exit at the cosine the wave has there*; *Leak direction - the direction a closed
face writes for a wave launched inside a solid*.

**The relaunch geometry.** `ray_mesh_interact` builds `origN`/`destN` a few float ulp past the face
(`qd_relaunch_offset`) and the tube `tridirN`. Where one medium lies on both sides (isolated sheet,
duplicate, ordered stack) the ray continues on the line it arrived on, all types, all frequencies.
Out of a lossy medium with types 2 and 5 `ray_state_update` puts `origN`, `destN` and the tube on
the direction it writes (`RELAUNCH`), on the base-frequency transmit call where the caller hands the
geometry in; all or nothing per ray. Asserted in: *Stack geometry - an ordered stack leaves the ray on
the line it arrived on* (wave half); *ray_progress - an ordered stack leaves the ray on the line it
arrived on*; *ray_progress - refracting tracer: the segment out of a lossy medium is the direction the
wave has*.

**The propagation convention** (`SCALE_BY_MEDIUM`). One question per leg, bent or not, same answer on
every pass. Bent with a recovered cosine: the exact form `exp(−j·k₀·(n·cs·d + q·s))`, conductivity
inside `n`, only `alpha` from `excess_gain`. Bent without one: the ITU-R P.2040 rate along the
refracted path and the phase of the real index (the hybrid form). Not bent: the walk-off form
`−k₀·(n_re·d_t·ct_t − d_i·ct_i)`. The phase is the field at the point where the committed polyline
leaves the medium: where `use_vbs` is set the exact form is carried there by
`exp(−j·k₀·Re(q)·(s_polyline − s))`. The displaced-host term of a sheet carries no transverse part.
Asserted in: *Phase reference - a traversal carries the phase at the point the polyline leaves at*
(wave half); *Diagnostic - Excess-phase walk-off convention*; *Traced solid - air | slab | air chain
against the analytic slab*.

**Where the layer of a sheet lies.** About the mesh plane for one medium on both sides (isolated,
inside a host, ordered stack: `u = thk/2`), in front of the solid for a glued entry, behind it for a
glued exit. Each part takes out no more host than lies on its side (host in front: the leg; behind:
the gap to the second face of the hit or to the destination). Reflect port `S11 = r/φ_d(u_f)²`,
transmit port `S21 = t/φ_d(u_f + u_b)`. Asserted in: *Sheet reflect port - where the layer lies and
what the echo pays*; *ray_progress - the reflect port of a sheet lies where its transmit port puts the
layer*; *Budget - a sheet or a stack inside a host is one wall on every port whether it is capped or
not*; *Unentered branches - the displacement budget caps what a sheet takes back out of its host*.

**A sheet under total reflection of its outer pair** (`far_closed`, glued sheets only). The sheet
stays in the result: the near face is the transformer, the far face the pad of a closed face, the
reflect port the composite, the transmit port what the pad passes of the incident flux; the
displaced host is taken out at the incidence cosine. Without `tf` anywhere the transmit port is
exactly zero. Asserted in: *Total reflection of the outer pair - the sheet stays in the result* (wave
half); *Leak chain - each face at the cosine the wave has and the series on a closed far face* (wave
half).

## 6. The state words

Three signed `short` words per ray, a 14-bit material index (16383 materials) and two flag bits
each. **The state must not grow** (production runs carry about 150 million rays), and no spare bit is
left. All three are decoded in one commented block at the top of `ray_state_update`'s ray loop.

| word | bits 0–13 | bit 14 | bit 15 |
| --- | --- | --- | --- |
| `mtl_ind_current` | medium the ray is in, 0 = air | *bent* (zero in air) | *resolved* |
| `mtl_ind_prev` | medium the leg was entered from; after a sheet event the sheet | with a sheet: *inside* (the sheet lies inside the current medium); with a solid: *real invariant* | *known non-parallel* |
| `mtl_ind_buffer`, bit 15 clear | a deferred medium; with bit 14 the *stack code* (the other layer of an ordered stack behind the ray) | *followed* mark with a zero index | — |
| `mtl_ind_buffer`, bit 15 set | **acoustic:** thickness code | part of the code | **acoustic:** *mass-crossed* |

The flags are decided on media and state, never on a value, except the bent flag, which records the
base frequency's bend of the shared polyline. `ray_subdivide_flag` and `ray_progress` compare the
current word whole, so it has to be zero in air.
Asserted in: *ray_state_update - state and code encoding*; *ray_state_update - flag semantics*;
*ray_state_update - the material index is 14 bits*; *ray_progress - the material count is limited by
the 14-bit state index*; *Unentered branches - the followed mark and the resolved flag across
abutting and nested and overlapping bodies*.

**A sheet in the previous-medium word** is a boundary, never a medium. Readers: the host's series
(the mirror behind the ray: a layer backed by air, or by the host under the inside flag; both layers
under the stack code), the recovery (no medium named), and nothing for a glued exit, which always
exits into air. Only a glued exit leaves the ray in air with a sheet in the word (an isolated sheet in
air writes no word); a ray in that state that meets a body from inside (an exit-class hit code: 1, 5,
7, 13, 21, 29 and their total-reflection variants) was in that body, the sheet lay between two
bodies (section 8), and the ray is killed, every type. Asserted in: *Sheet inside a host - the mirror behind the ray is backed by the host*;
*Lined on both faces - a glued exit behind a sheet exits into air*; *Layered wall - a glued exit
behind a bare-entered second layer exits into air*; *Layered stack - a host layer bounded by a sheet*;
*Sheet between two bodies - nothing is created and the wave left in air inside the far body is killed*
(`test_acoustic_leaf.cpp`).

## 7. Series, paths and verdicts

**A layered wall is a sum of paths.** A traced body is resolved on its own, its series
`1/(1 − r_behind·r_ahead·φ²)` with what bounds it taken as it stands; what a neighboring solid
returns arrives as further paths, one per order, `3 + 4n` events. The first path is not the wall; the
sum is. Asserted in: *Measurement - the mirror behind a second solid and behind a sheet inside a wall*
(wave half); *Measurement - ray_progress follows the coupling between the series of a layered wall*;
*ray_progress - a wall with a stack inside it is on its matrix with the paths of a single sheet*.

**One verdict per body.** Whether a body's series is summed or followed is taken once, by the first
event that closes a leg in it (`ρ = |r_ahead·r_behind|·|φ|²` against `thin_slab_threshold`). Summed:
the reflected ray is *resolved*, its next leg forms no series and its next reflection is killed.
Declined: the reflected ray carries the *followed* mark and the legs it closes form no series. A
transmission starts a new leg and a new verdict. Across a band every frequency takes the base
frequency's verdict (`verdict_inN` / `verdict_outN`). Resolved or followed, never neither; both are
one wall. Asserted in: *Measurement - the mirror behind the ray after a back reflection*; *Measurement
- ray_progress and the mirror behind a back reflection*; *Sheet reflection - a declined series is
followed and not dropped*; *ray_progress - a declined series at a sheet is followed and not dropped*;
*Unentered branches - a verdict handed in on a leg known not parallel and near a pole*.

## 8a. Open: `alpha` on a layer that tunnels

`alpha` is charged over the path `d/|Re c|`. A lossless coherent sheet rarer than the transverse
invariant has `Re c = 0`: it tunnels, and with any `alpha` it is opaque. Measured with a scratch
probe, not in a fixture: 1 mm of the lossless `eps 0.31` with `alpha` 100 dB/m (0.1 dB over its
thickness) passes nothing past its critical angle, in air, inside a host and as a lining. The
candidate is the normal thickness (or the path of the wave's real transverse invariant). It is one
decision together with the path of a lumped plate (`md/acoustic.md`, item A3), and it moves no row
today, since no case carries `alpha` on a tunnelling layer.

## 8. Accepted limitations (wave)

None has a fix in sight or one worth its cost. Sizes are measured.

**Faces and `tf`**
- `tf < 0` forces a reflection per channel, also in TM at the Brewster angle: 0.731 in VV and 0.500 in
  HH at 66.4° on the lossless `eps 5.24` with `tf` −0.5, where the raw face reflects nothing in HH. A
  laminate of two identical `tf < 0` sheets has an internal mirror whose phase no medium defines.
- The reflectance of a transformed face is `(1 − tf)·R0` only where its two sides are in phase. One
  face seen from air: `eps 5.24 − 0.63j` +0.03 dB at `tf` 0.5; `eps 1.50 − 0.40j` +1.0 dB and +5.3 dB
  at normal incidence (where `R0` is 0.016) at `tf` 0.5 and 0.9; `eps 0.31 − 0.20j` at 75° 0.79 where
  `(1 − tf)·R0` is 0.42. For `tf < 0` within 0.3 dB; a lossless wall exactly.
- `tf = +1` closes the reflect port of a coherent sheet only where its faces are in phase: exactly
  zero for the lossless `eps 5.24`, up to 1.4e-2 for `eps 5.24 − 5j`. `tf = −1` closes the transmit
  port exactly.
- A forced face (`tf < 0`) steps where its admittance ratio has modulus one: 6 mm of `eps 0.90 − 0.20j`
  in air between 59.99° and 60.01°, TE and scalar, −0.29 / −1.05 / −2.52 dB at `tf` −0.1 / −0.5 / −0.9,
  as a sheet and as a solid. Exactly at the tie a sheet reads a value of its own (−9.61 dB between
  −0.18 and −1.22 dB): its two faces form the ratio from two spellings of the sine. A single point.
- Across the critical angle of a sheet's outer pair there is the square-root cusp of the far cosine
  and no step: two neighbors 0.02° apart differ by 0.1 to 1.2 dB, as the exact composite does.
- Exactly at its critical angle a lossless sheet without `tf` is 0/0 and the solver declines; the
  event keeps what `ray_mesh_interact` wrote, and an ordered stack kills the ray. A single point.
- A rigid permittivity without the mass column is a wave medium: as a sheet it tunnels (22 mm of
  `eps 2.1609e-8` in air passes −0.23 to −0.88 dB at 10° to 60°); as a traced solid it is closed by
  its first face's verdict and leaves along the wall's normal (0.0005° to 0.0028°); a leak meeting
  a face towards the lower index is judged from inside (4.5 dB between the two directions at 30° for
  an impedance ratio of 3). Acoustic materials carry `m`.

**Energy statements next to a lossy medium**
- Entry and exit of a lossy solid differ by the flux splitting `sec²(arg η)` per channel, exit above
  entry: the energy convention, not a defect.
- The two ports of an event next to a lossy medium are not an energy sum: TM reflectance above one is
  the Fresnel value (1.178 for `eps 0.31` seen from `eps 2.50 − 0.80j` at 45°); the composite of a
  sheet in a lossy host can read above 0 dB (+0.36 dB TM), the event with its host leg is below 0 dB;
  the characteristic matrix itself reads 1.0011 in `eps 2.50 − 2.0j`. Each port is asserted.
- A sum of path powers is not an energy statement next to a lossy medium (a crossing out of it is
  normalised to the forward wave); coherent sums are. A coherent sum over the paths of a leak is not
  one either: a lossless rare solid with `tf` and no mass column reads 1.02 to 1.69 at 45° to 85°.
- A followed bounce series can diverge where the summed one is exact: in a lossy layer rarer than air
  between strong reflectors the round trip `|r1 r2 φ²|` reaches 1.19 (6 mm of `eps 0.31 − 0.20j`
  behind a coating at 85°). Through `ray_progress` such a leg is summed wherever its round trip
  `ρ = |r1 r2|·|φ|²` reaches `thin_slab_threshold`, and a round trip above one always does (the gate in
  `qd_slab_resolve`, at a threshold of 1 as well); two such cavities coupled have no finite path sum (+6.4 to +88.6 dB
  above the matrix at 60° to 85°).
- A negative `resS` is a gain medium (2.11 on the two ports of a 37.5 mm sheet of `ε` 4 with `resS`
  −0.5 at normal incidence). Nothing validates `resS`.

**State, depth and directions**
- `mtl_prev` is one level deep. Air is assumed beyond every glued exit (a lined body nested in
  another medium is not a use case).
- A sheet between two bodies (body | sheet | body, three co-located faces; A21 in `md/acoustic.md` 6)
  builds, but the intersector reports two of the three faces, chosen by mesh order, so the physics
  depends on the order of a tie, in every type: box exit and sheet reported, a glued exit into air;
  box exit and far entry, the sheet ignored; sheet and far entry, the glued entry from the host, exact
  (the fluid layers within 1e-6 in TE and scalar). In the first order the wave the state puts in air
  inside the far body used to create energy where the series was summed: R + T up to 42.19 (scalar
  and EM TE) and 14.46 (EM TM) for an air box, a 15 mm porous sheet and an `m = 0` wall, and
  `ray_progress` ran away with a leaf in front. Since 2026-10-16 the ray is killed where it meets the
  far body from inside (section 6), every type, and R + T ≤ 0.902 over the variants measured
  (`md/m5_a21_report.md` 3.5); what remains loses energy only: the reflect port of box | sheet | air
  (R 0.154 for 0.994 at 125 Hz), or the bare body | body face. A sheet glued on the far body's back
  face is met as air | sheet | air (code 69, nothing created). The same kill takes a legitimate glued
  exit followed by a face met from behind (open or flipped geometry); not seen in the repository
  scenes, which have no glued sheets, and not measured on a real scene. A full resolve needs the
  third face from the intersector or `ray_progress`. Behind a stack inside a host the mirror is the last layer (with
  the stack code: both). A wave two solids deep behind a launched wave or a wedge takes the named
  solve; a reflection off a sheet replaces the name. The followed mark and the stack code share the
  buffer word with the deferred medium: a nested object met on such a leg replaces them (series
  formed there: +0.08 to −0.54 dB around a 20 mm object in 100 mm; 0.78 dB behind a stack).
- The current-medium word is corrected in one direction only (air → medium at the next exit, with
  the fix-flag); nothing writes the other state today.
- An exit out of a wave whose invariant is not real (launched inside a lossy solid, a lossy wedge)
  takes the real-angle map, about 0.2° for `eps 1.50 − 0.40j` at 25°.
- With the undeviated tracer a wrong direction in air is not a wrong place, but a sheet event in air
  reads it. With the refracting tracer the segment at a closed face stays `ray_mesh_interact`'s for an
  invariant between 1 and about 1.045 on `eps 1.50 − 0.40j` (67.9° against a wave at 52.3°). The
  direction a closed face hands on steps at an invariant of one (84.26° at 0.99, 52.27° at 1.01);
  reached only from a host entered from a denser medium, never by a wall in air.
- A caller with types 2 or 5 that does not hand in the geometry keeps the layer's segment behind a
  closed face. `ray_progress` hands it in on the base-frequency transmit call.
- A traced leg is short by the relaunch offset (1e-6 m undeviated, up to 1.06e-5 m refracting at 85°):
  up to 4e-3 dB through the series of a dense slab near grazing. A lossless leg behind a co-located
  pair carries that pair's anchoring (1.6e-6 at 85° through `ray_progress`).
- The bent flag, the series verdict and the geometry are the base frequency's; near a critical angle
  of a lossless dispersive medium another frequency can have the other verdict. Where the cosines
  decide (the 1e-6 test of `use_vbs`) rounding can, near normal incidence and in single precision.
- A ray launched inside a host is a leg the state does not know to be bent: a series summed on it is
  off its followed bounces by 0.13 to 3.57 on a lossless wall. Seed the bent flag and use type 5.
- The hybrid propagation factor (where no cosine is recovered) carries two rounded constants of
  `medium_gain`: 2.5e-5 dB; exact constants would move 517 rows.
- `interact_with` keeps a real-angle and a complex-cosine spelling of the refraction root and of the
  total-reflection test; merging them moves coefficients by 6e-14 and flips a forced reflection's sign
  at a matched face.

**Geometry of a run**
- A co-located pair is judged along the ray (1 mm): a pair 100 µm apart is not one event at 85°.
- What lies between a hit point and the next relaunch point is charged as free space (0.013° between
  the tracers through two solids with a 10 µm gap at 85°).
- The tube is corrected with the spine or not at all; a tube whose edge is past a critical angle while
  the spine is not passes undeviated with the total-reflection bit set.
- With the refracting tracer the segment is bent at the face of a deferred body as if entered from
  air (10.27° against a wave at 12.62° inside an overlap); the crossing into a deferred medium keeps
  the direction formed for medium | air.
- A transparent transition between two indices of one material sets the fix-flag (type 89 for 73).
- The mid-plane of a sheet with one medium on both sides is a convention of the mesh; the reference
  plane of a sheet's reflect port is a phase per frequency, no geometry.

**Paths, hosts and thresholds**
- The first path through a layered wall is not the wall (up to −4.88 dB for a lossless second layer
  at 85°); a run that stops early keeps the difference (+1.37 dB after 20 interactions, +0.04 dB
  after 60). Two regimes of one wall differ by what `min_gain_dB` cuts (−0.215 dB at −100 dB at 85°).
  Two coupled cavities with every series declined do not converge near grazing.
- A sheet that does not fit its host is capped (each half takes no more than the host on its side);
  a stack thicker than its whole host, met from its long side, passes more than it received (+0.24
  to +1.81 dB, both ports 1.05 to 1.62). The intersector reports only the stack's two faces.
- A sheet inside a host and its traced counterpart differ by the host's interface gain (the traced
  solid pays the host again on re-entry).
- An opaque layer transmits at its own level down to underflow (−2045 dB for 2000 dB of loss).
- A verdict of "sum" on a leg known not parallel is not reachable by a consistent caller.
- The path sums of two solids with a far coating, three solids and every EM channel have no fixture
  (the straight path through three solids is recorded, up to −3.52 dB against the wall).

## 9. Where the acoustic add-on attaches

Every one of these is gated on `m > 0` (or on a mass-law medium in the state) and is described in
`md/acoustic.md`; none runs for a wave medium:

- `Material::medium_gain`'s mass term, `mass_X0`, `mass_TL` (the plate rule).
- `qd_mass_amp` and the mass-law segments of `qd_stack_resolve` (a mass-law layer ends the coherent
  segment), its `CLOSED` verdict, the held branch, the leak chain, and the pair rule
  `apply_tf_pair` on the reflectance at an open mass-law face (the one clamp left in the file).
- `Material::interact_with`'s two undeviated branches and the real immittance of a mass-law side.
- `qd_slab_resolve` returns false for a mass-law slab (no series).
- `MASS_RUN`, `MASS_FOLD`, `MASS_CODE` and the buffer word with bit 15 set; `SHEET_RESOLVE`'s
  mass cosine; `APPLY_TRANSITION` writes no bent flag out of a mass-law medium.

A change on the acoustic side must keep these gates: the `[wave]` baseline and the stored twins are
the check that it did. The acoustic route M9 (`md/acoustic.md` 2.8) rests on the wave media of the
two-parameter mapping, so a change on the wave side is checked against `test_acoustic_leaf.cpp` too.

## 10. Rules for changes

- The library is to stay small: a wider gate and one rule over an exclusion kept for digits;
  bit-identity is not required, a row may move within reasonable bounds for a physical reason.
- Measure, don't compute by hand; a fixture that describes a traced run goes through
  `ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update`, or through `ray_progress`.
- Before the source is touched, a candidate is measured on a what-if: a patched copy of
  `src/ray_mesh_interact.cpp` compiled with `build_linux/CMakeFiles/quadriga_static.dir/flags.make`
  and linked in front of `libquadriga.a` (`build_linux/CMakeFiles/test_bin.dir/link.txt`), run from the
  repository root. A plain copy has to reproduce the baselines first.
- Fixtures that call the library one ray at a time use `ub::OneThread`.
- A moved table is read in its unit, column by column; changing a test translation unit alone shifts
  a few residual rows by about 3e-16.
- `html_docu/quadriga_lib_material_model.md` is out of date and out of scope until it is taken up.

## 11. Test instruments

All in `tests/catch2_tests/ray_test_helpers.hpp` unless noted. `QD_TEST_VERBOSE` in a test file
switches its tables on; every table carries at least one numeric bound, so no case is a no-op
without them.

**Independent references** (closed forms with complex angles throughout, no library code):
- `qd_slab_ref`, `qd_slab_ref_R`, `qd_iface_ref` (with the per-channel flux fields),
  `qd_slab_ref_lumped` (air | sheet | air with `att`, `tf` and a resonance).
- `diag_slab_p_pol` / `diag_slab_p_avg`; per stage of a traced leg: `diag_port`, `diag_series`,
  `diag_medium_vbs` / `diag_medium_geo` / `diag_medium_walk` (the three traversal forms),
  `diag_traced_ct` and `diag_ct_exact` (`qd_refract`'s map and the value it projects).
- `diag_sheet_composite` (three-medium Airy pair, both ports, flux conversion, displaced host,
  reflect-port plane and echo gain; a trailing `DiagLump` carries `att`, `tf` and resonances) and
  `diag_sheet_composite_cx` (the same at a complex incidence cosine).
- `diag_cm_stack` (N-layer characteristic matrix, `DiagLayer` with resonance, `alpha`, `tf`, `m`,
  `fRef`, gain) and `diag_sc_stack` (the far-end recursion with `tf` per medium), each reproducing the
  other and the single-layer builders.
- `diag_tf_face` (the transformer, written as a Fresnel face of the scaled ratio), `diag_closed_face`
  (the pad), `diag_tf_pair`, `diag_tf_port`, `diag_leak_series` (a coherent sheet on a closed far side).
- `diag_res` (the Lorentz pole from the column definition), `LumpMtl` / `lump_palette` (a palette with
  `att`, `tf`, resonance, `alpha`, `m`, `fRef`), `diag_ray_at(theta)` (a ray from x = −1 crossing
  x = 0 at y = 0.37 + tan θ).

**Runners:**
- `ub::traced<T>` runs one event through `ray_triangle_intersect`, `ray_mesh_interact` and
  `ray_state_update`, returning what `ray_mesh_interact` wrote (`xpr0`, `dir0`) beside the redo, the
  hit points, the three words, the ledgers, both type codes and the verdict; `ub::Opt` sets `eps`, the
  relaunch switch, a verdict to hand in, a tube. `ub::face`, `ub::tilted`, `ub::build` make the mesh;
  `ub::onward` relaunches on the segment an event wrote; `ub::chain` runs a sequence of types;
  `ub::at_point` is a path's field at the point its polyline ends; `ub::same_bits` compares bitwise;
  `ub::OneThread` is the one-thread guard.
- `bf::walk` sends one ray through, runs both passes at every event, relaunches on the written
  segment, and sums every path back in air at its polyline's point (`eps` 0 sums every series, 1.5
  follows every bounce; `inc` is the sum of path powers). `cf::walk` adds a sink medium (a wall behind
  as a half-space) and a seeded start medium.
- Several cases carry their own chain runner (`traced`, `trace`, `run`, `chain`, `walk`) and the
  `ray_progress` cases a `trace` that calls `ray_progress` until no ray is left; they were left as
  they are.
- `twin()` asserts every-digit rows against `tests/data/wave_twins.tsv`.

**Instruments worth reusing:** the series as a ratio (`eps` 0 over 1.5); the host leg as a ratio of
two path lengths; the thin-sheet ratio (the event over the same event with a 1 mm sheet); the
bare-entered twin (a clean previous-medium word, compared bitwise); the mirror as two references
(half-space and layer, asserted different); the limit of a geometric sequence of paths
(`F0/(1 − F1/F0)`); the plane a reflected sum is referenced to (`exp(−2j·k₀·cos θ·x)`); the loss sweep
to zero (1e-6 against 0) for a branch taken on a value; `tf` ±1e-9 against none.

**Outside the suite:**
- *What-if build* (section 10). Options to compare go behind an environment-variable bitmask read
  once in the copy; the copy with every switch off must reproduce the baselines.
- *Marker build*: a counter per site keyed by a string (site, face, channel, columns present, which
  media are lossy), dumped per case by a Catch2 listener (`CATCH_REGISTER_LISTENER`, `testCaseEnded`
  calling the copy's dump function). Templates: `build_linux/scratch/i47/mk47.py`, `listener.cpp`,
  `build.sh`; the split's classifier in `build_linux/scratch/ts1`.
- *Scratch probe on a test file*: a file outside the repository that includes a test file and adds
  one case, compiled with `build_linux/CMakeFiles/test_bin.dir/flags.make` and linked in place of
  that file's object; a walker probe reading one wall per line from a file named in an environment
  variable scans thousands of walls without a recompile (`build_linux/scratch/i47/probe4.cpp`).

**Traps:**
- Run a relinked binary from the repository root (cases open `tests/data` relatively). Never two
  suites at once; check with `ps`; wrap probes in `timeout`.
- One ray at a time costs twenty times the work without `ub::OneThread`.
- `bf::walk` and `cf::walk` collect a path wherever the state says air; a geometry that leaves the
  wave in air inside a body (a sheet between two bodies) needs `lw::walk`'s rule
  (`test_acoustic_leaf.cpp`): air collected only outside the structure's x range.
- A segment that ends inside a body preloads the ledger and a fixture that relaunches by hand adds
  the leg again: segments must reach past every remaining face (a 2 m segment ends inside a 200 mm
  slab at 85°).
- The angle between two float unit vectors: `atan2` of the cross product, not `acos` (0.014° floor).
- `arma::accu(arma::square(...))` does not link in the test binary; `arma::abs` of an xprmat is per
  real and imaginary row.
- A table field containing `|` must not be pasted with `|` as delimiter; case names with a comma need
  `\,` in a test spec.
- `pkill -f` with a runner's name kills the calling shell; kill by pid.
