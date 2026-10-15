# The acoustic add-on — the mass-column machine, what is entangled, what is open

Acoustics in this library is an add-on to a wave engine. It has to be **accurate on the reflect
side** and make **reasonable predictions on the transmit side**. This document is about the machine,
not about material parameters: the goal is that every quantity a published acoustic material
specifies can be reached by some setting of the columns, in the bands that matter. Parameters come
afterwards. No acoustic materials are defined for the library today (`data/materials_ac.csv` and its
copy under `tools/acoustic_parameter_mapping` are out of date and are no reference for anything).

The boundary is the **mass column**: everything with `m > 0` is acoustic and described here;
everything without it is wave physics, settled and guarded, in `md/wave.md`. Every acoustic change
stays behind `m > 0`; the `[wave]` baseline and `tests/data/wave_twins.tsv` are the check.

The window analysis behind sections 3 to 5, with every source and figure, is in
`build_linux/scratch/i47/NOTES.md`; the classification of the tests in `md/test_split_report.md`.

First step: `md/wave.md`, section 0. The acoustic half:
`./build_linux/test_bin "[acoustic]" > test_acoustic.txt` → `All tests passed (32697 assertions in 46 test cases)`.
Its cases are in `test_acoustic_mesh_interact.cpp`, `test_acoustic_progress.cpp` and
`test_acoustic_state_update.cpp`; 18 of them are "<name> - the mass-law part" of a wave case. The machine
described here is the one after acoustic round 1 (decisions of section 10, taken 2026-10-15; report
`md/acoustic_round1_report.md`, section 5).

## 1. The acoustic mapping

Frequencies are mapped `f_radio = f_acoustic · c_light / c_sound` (1 kHz ≈ 0.874 GHz).

**Two parameters** (N7): a material with impedance `Z` and sound speed `c` is mapped as `ε = n·z`,
`μ = n/z` with `z = Z_air/Z` and `n = c_air/c`, in the columns only (no code reads the mapping). Air |
water is then exact as a traced half-space, in level and critical angle (13.38°); a hard wall reflects
with a phase near zero (concrete: 0.009° at 10°, 0.104° at 60°); a porous layer on it absorbs as the
exact layer (Delany-Bazley, 20 mm, 500 Hz: within 0.0008 in α at 0° to 80°). Rigid bodies then have a
critical angle of a few degrees (glass 3.48°, gypsum 9.44°); below it a traced rigid solid holds every
tracer. Asserted in: *Two-parameter mapping - air | water, a hard wall's phase, a porous layer on it
and every tracer below its critical angle*.

**One parameter** (the older form, still valid): `ε' = (Z_air / Z)²`, `μ = 1`. Rigid bodies are then
optically rarer than air by eight to eleven orders (glass 8.5e-10 for Z = 1.41e7 rayl, concrete
2.4e-9, wood 2.2e-8), air | rigid is past its critical angle from about 0.003°, and a rigid half-space
reflects with a phase of twice the angle (A12). Most fixtures use it.

Either way the mapping fixes the reflection of a half-space and nothing else: the `ε` of a rigid body
is not a propagation index, and a mass-law material's `ε`, `μ` are calibrated for surface impedance
only.

## 2. What the machine does today

Everything here is gated on `m > 0`.

### 2.1 The mass term

`Material::medium_gain` forms it from a surface-mass surrogate `M` with the angle removed (the
normal thickness) and an angle factor `c` (the cosine the wave has in air):

    A  = m·log10((f/fRef)·M)        X₀ = ((f/fRef)·M)^(m/20)
    TL = A·log10(1 + (X₀·c)²) / log10(1 + X₀²)

`A` at normal incidence, zero at tangency, 20 dB per decade of `cos θ` whatever `m` (the cosine enters
through the fluid's admittance, not the panel's impedance). The term is zero below `X₀ = 1` (a kink
in frequency, bounded by 3 dB within a decade of the knee). With `m = 20` and
`fRef = (f_GHz/f_Hz)·ρ₀c/(π·ρ)`, a material constant (4.58e-5 GHz for glass), `X₀ = ωm′/(2ρ₀c)` is the
physical impedance ratio of a limp panel. A traced solid's leg shorter than 1.5 mm carries no term; a
sheet carries it at any thickness. The public `medium_gain` supplies `c = 1` and returns `A`.
Asserted in: *Thin sheet - mass-law angle dependence*; *Acoustic - medium_gain in-medium loss*.

**Two impedances in series with the mass** act in `X₀` (`Material::mass_X0`), both at the cosine `c`
the wave has in air (N3, N4):

- `res*` on a mass-law material is the plate's resonance, not a permittivity pole (`eta()` adds the
  pole only where `m = 0`, so a rigid body with `res*` stays closed). `resS > 0`: the mass-air-mass
  resonance of a double unit or a leaf on its cavity, the dip it makes and no more,
  `X₀ × min(1, |1 − (f·c/resF)² + j·f·c/(resF·resQ)|)` (the cavity is stiffer by `1/c²`, so the dip
  moves up as `1/c`). `resS < 0`: the panel's own first mode, a stiffness below it,
  `X₀ × |1 − (resF/f)² − j·resF/(f·resQ)|`. Sheet and traced solid on the rule within 0.0071 dB.
- `coiF`, `coiQ` on a mass-law material are the bending stiffness of the plate (`coiF` its coincidence
  frequency, `1/coiQ` its loss factor): `X₀ × |1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ)|`, so the diffuse-field
  dip comes from the angle. **Rule of use: `coiA = 0` on a mass-law material** (it would act once more
  in the interface gain).

Bonded bodies add these `X₀` like any other (2.2). Asserted in: *Resonance - on a mass-law material res\*
is the plate's resonance and leaves the permittivity alone*; *Coincidence - on a mass-law material coiF
and coiQ are the bending stiffness of the plate*.

**The cosine is the one the wave has in air**, `sqrt(1 − p)` for the transverse invariant `p`: for a
wall in air the incidence cosine, whatever refracting layer lies in front. `SHEET_RESOLVE` forms it
where the near medium is not air, carries no mass column, the leg is known to be bent, and the sheet
(or a layer of the stack) carries `m > 0`, as the far cosine of near | air at the recovered cosine;
elsewhere the traced cosine already is it. Above an invariant of one the term keeps the cosine the
wave has in the near medium. A plate with a cover on both faces pays as the bare plate (the
thin-cover view). Asserted in: *Mass cosine - a mass-law sheet behind a refracting layer forms its
term at the cosine the wave has in air*.

### 2.2 Bonded bodies are one plate

Mass-law bodies within the co-location distance move with one velocity: their `X₀` add
(`TL = 20·log10(X₀₁ + X₀₂)` at normal incidence, the single-body shape on the sum). A 35 dB panel on a
53 dB wall is 54.03 dB, not 88. Each `X₀` is unclipped, so the zero sits on the sum; `m = 0` is
absent (a resilient underlay between two mass-law bodies separates them, their terms then add in
dB). The charging telescopes: the first body pays its term, a later one `TL(X_run + X₀) − TL(X_run)`
(35 then 19.03 dB; the other way 53 then 1.03 dB). A leg that begins at a reflection starts with
`X_run = 0`. A sheet with its host's permittivity is not merged into a mass-law host; it cuts the leg.

How the second body learns `X_run`:

| pair | learned from |
| --- | --- |
| sheet \| sheet | the solver's run over adjacent mass-law layers, one event |
| sheet \| solid, entry | `mtl_prev` holds the sheet (thickness is a column), the buffer flag says it was crossed |
| sheet \| solid, exit | the solid closes its leg in `SHEET_EVENT` and hands the sheet its `X₀` |
| solid \| solid | `mtl_prev` names the material, the buffer word carries the thickness code |
| a sheet inside a wall | `mtl_prev` holds the sheet with its inside flag, the buffer word the first half's code |
| deeper | the same two words: the code is an equivalent thickness of every body behind |

**The thickness code** (buffer word with bit 15 set): `d = 10^(−4 + 6·(k − 1)/32766)` m, `k = 1 … 32767`,
0.1 mm to 100 m in steps of 0.044 % (0.0038 dB at `m = 20`), quantized again at every bonded face.
`MASS_FOLD` forms the equivalent thickness `d_eq = (Σ (dᵢ·fRef_R/fRefᵢ)^p)^(1/p)`, `p = m/20`, read with
material `R`; no frequency enters, so one code serves every frequency of a run. `MASS_CODE` writes,
`MASS_RUN` reads. A body of another slope is left out of the fold. With `m ≠ 20` two halves of one
material differ from the table's term for the summed thickness by `0.301·(20 − m)` dB (+2.41 dB at
`m = 12`, −3.01 dB at `m = 30`); the +6 dB per doubling is the physical one.
Asserted in: *Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate*;
*Bonded mass - a mass-law sheet inside a mass-law wall*; *Bonded mass - the plate at any depth where
the slopes agree*; *ray_progress - the bonded plate at any depth where the slopes agree*.

### 2.3 The plate rule

A run of bonded mass-law bodies (2.2) is one plate with no wave inside. It reflects at the face met
what that face does not absorb, `R = 1 − a_met`, and passes in all the incoherent series of its faces
and its bodies (N1, R1h):

    T = a·τ / (a + τ − a·τ)

with `a` the smaller absorption of its two outer faces and `τ` the bodies' term: `T ≈ τ` where `τ ≪ a`,
`T → a` for a light panel or at a resonance, never more than either, smooth in `τ`. A heavy wall
passes its mass law, no longer the mass law times the absorption of both faces (at bb336dc a 200 mm
wall with `tf` 0.3 passed 10.45 dB less).

**The absorption of an outer face** is what its pad passes (`Material::closed_face`): `tf` in front of a
lossless medium where the face is closed (every rigid face from 0.003°, one-parameter mapping), the
transformer's `1 − R′` where it is open (a rigid face exactly at normal incidence: `1 − (1 − tf)·R0`, `R0`
its Fresnel reflectance), and next to a lossy medium the flux the pad hands over. A face without `tf`
absorbs nothing and the plate passes nothing: **`tf` is set together with `m`**. `tf` may carry a resonant
peak (2.6). A face between two bodies of the plate passes everything and reflects nothing.

**Where it is paid.** The first face passes `a_in`; each body's leg pays its telescoped term (2.2); the face
out of the plate into a medium without the mass column pays the rest,
`F(min(a_in, a_out), τ) / (a_in·τ)`. For a traced plate `a_in` is re-formed at the exit from the
previous-medium word (`PLATE_IN`, one level deep, section 7); for a sheet or a stack the solver carries it
along the run (`PLATE_FIX`). A mass-law sheet that is a whole plate under a closed outer pair (glued to a
medium the wave cannot enter, the leak chain) passes the rule of its two faces in one event.

Numbers: 4 mm glass and 152 mm concrete with their measured absorption, sheet and traced solid, both
tracers, both mappings: on the rule within 0.0022 dB, the reflection within 0.0008 dB. Wood 20 mm |
concrete 100 mm (`tf` 0.2 | 0.1): `a` = 0.1, the same from both sides within 0.0007 dB as two solids,
sheet on solid and stack. A wood panel glued on concrete without its mass law is reciprocal within
0.00094 dB (the flat first face, section 7). Asserted in: *Plate rule - a rigid mass-law panel passes
the incoherent series of its face and its plate and reflects what its face does not absorb*; *Plate
rule - a wall bonded of two mass-law bodies passes the same in both directions*; *Closed plate - a
mass-law layer the wave is evanescent in passes the plate rule of its two faces*; *Leak reciprocity - a
mass-law panel on a rigid wall as a sheet and as two traced solids*; *Total reflection of the outer
pair - the sheet stays in the result - the mass-law part*; *Bonded mass - …* (all four); *ray_progress -
the plate rule and the absorption peak at every frequency of a band*.

**A closed mass-law layer** (lossless, `Im(ε·μ) = 0`, and the transverse invariant at or above its
`ε·μ`: asked of the layer, so both faces close together) carries the leak as a flux. Seen from a side
with a wave (a coherent layer, a medium without the mass column) its face is the pad of a closed face:
on the way in the flux it passes, on the way out the amplitude that flux launches. A stack of a coherent
layer and a mass-law layer is one wall in both orders where the mass-law layer is closed; where it is
not closed the two orders differ by 0.107 / 0.127 / 0.201 dB at 10° / 30° / 60° (section 7).

**In the solver** a mass-law layer is not made coherent: it ends the coherent segment as a half-space
of its own admittance, contributes one real factor (`qd_mass_amp`: `alpha` over the held path
`thk / c` with `c` the cosine in air, under a 400 dB cap, and the mass term), starts the next segment,
and displaces no host. Nothing returns across it: its reflect port is its near face. A mass-law sheet
named as a mirror in `qd_slab_resolve` is a half-space. Asserted in: *Lumped columns - a bounding sheet:
tf is a layer, a mass-law sheet a half-space*; *Stack with a mass-law layer - resolved by segments, not
killed*; *Held path - alpha on a lumped plate is charged over the path the held wave takes* (22 mm at
100 dB/m: 2.19 / 2.22 / 2.52 / 4.31 / 11.74 dB at 0° / 10° / 30° / 60° / 80°, sheet and traced solid
within 0.001 dB).

### 2.4 The held wave

A mass-law body holds the wave undeviated, whatever its permittivity and the tracer (a body with
`ε` above one included). `interact_with` takes an `undeviated` flag: inside a plate a face passes what
reaches it; towards a medium without the mass column the held wave carries the transverse invariant it
had in air, so it is refracted as from air and closed where air would be past that medium's critical
angle, and `APPLY_TRANSITION` writes the bent flag for the leg that begins there. So sound through a
wall continues in the direction it arrived in, a layered wall has one angle throughout, and the leg
behind a wall into a wave medium is the refraction from air. `ray_mesh_interact` sets the flag always;
`ray_state_update` where no refracted direction exists or the cosines coincide, unless the leg carries
the bent flag.

**A layer behind a held wave.** A mass-law sheet on the far face of a mass-law body, or inside it, is part
of the plate. A layer without the mass column glued on the far face (a lining) is the same wall as its
traced layer, for every tracer, within 0.015 dB (dense `eps 2.50 − 0.80j` and porous `eps 0.31 − 0.20j`,
10° to 75°); the two sides of such a lining differ by the lossy medium's convention (A2, section 7).
A non-rigid mass-law body (`eps 1.50`, `m` 20) passes at every angle, tracer 5 on tracer 4.

**A closed face hands the wall behind a traced layer the direction in air** (the leak's direction),
so the wall's mass term is formed at the incidence cosine.

Asserted in: *Held exit - a lining behind a rigid wall is the same as its traced layer, for every
tracer*; *Held exit - a non-rigid mass-law body holds every tracer and passes at every angle*; *Held
lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid*;
*Ports of a mass-law face from inside - the reflect port is the other direction's*; *Leak direction - …*
(both); *Phase reference - … - the mass-law part* (the held reference, dense and rigid, 0.0000°);
*ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a
band*.

### 2.5 Nothing is reflected back into a mass-law body

`qd_slab_resolve` returns false for a mass-law slab, and since round 1 (N2) a face reached from inside a
held mass-law body reflects nothing: a traced plate passes in one path. A 4 mm pane and a 20 mm wood
panel with every bounce followed pass one path, equal to the summed series; the reflect port of a held
wave at a lining is gone (−999 in *Held lining* D). Asserted in: *Plate rule - a traced mass-law body
passes in one path: nothing is reflected back into it*; *ray_state_update - mass-law material is
excluded from slab resolution* (its exit into air an ordinary one, type 1, at 0° as at 0.1°).

### 2.6 The room-side absorption

`tf` is the absorption of a rigid face (2.3) and, with three columns (A10), it can peak:

    tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f − tfF)/tfF)²)

clamped to `[−1, 1]` as before; `tfF ≥ 0`, `tfQ ≥ 0` are required, and the peak acts where `tfP ≠ 0` and
`tfF > 0`. A 12.5 mm board on battens (floor 0.05, peak 0.25 at 150 Hz, `tfQ` 1.5) absorbs the formula
exactly as sheet and traced solid and passes the plate rule within 0.001 dB, and through `ray_progress` every frequency of a band equals its
single-frequency run. A material CSV carries the three columns like any other; the OBJ writer's
canonical order and the Python / MATLAB readers' key order list them after `tfB`. Asserted in: *Absorption peak - tfP, tfF and tfQ add
a resonant peak to the room-side absorption of a mass-law face*.

**Rule of use:** a rigid body carries the mass column, and `tf` is set together with `m`. A rigid
permittivity without `m` is a wave medium (`md/wave.md`, section 8). A rigid body is lossless: a loss
column opens its faces, and a 200 mm wall with `tf` 0.3 and a loss tangent of 0.04 then reflects
1.000000 from 10° on and passes −88 to −102 dB (*Lossy verdict* J, recorded). `coiA = 0` on a mass-law
material; one `att` per bonded run (A5).

## 3. What was entangled (measured on the window before round 1)

Status after round 1: items 1, 3 and 4 are resolved by the plate rule, `res*` in `X₀` and M2 (section
2); item 2 by the bending stiffness in `X₀` (`coiA = 0` on a mass-law material); item 5 stands (a
double unit is lumped as one sheet with `res*`). The text below is the state at bb336dc.

The window analysis built 4 mm, 6 mm and 4-16-4 glazing from published data (EN 12758 generic octave
values via the Pilkington datasheet, Saint-Gobain ACO-3B, NRC CBD 240, glass absorption tables) and
compared the library with a thin-plate impedance model (mass, bending stiffness with loss factor η,
double unit by transfer matrix).

1. **`tf` has two jobs.** It is the room-side absorption (`R = 1 − tf`) and it scales the transmission
   twice (`T = tf² × mass term × gains`); a formula copy of that sits on the walker to 0.011 dB. With
   `tf` at the measured absorption of window glass (0.132 at 1 kHz, falling as f^−0.62), a 4 mm pane
   with the physical `fRef` reads +3.3 to +42.7 dB above EN 12758 in the octaves 125 Hz to 4 kHz; the
   best fitted `fRef` (×4.3) still leaves −6.8 to +30.1 dB, because `tfB` adds 3.7 dB/oct to the
   transmission slope and a fitted `X₀` moves the knee of the angle shape. The physical plate model
   with the same masses is within 2.5 dB of EN in every octave (η 0.01–0.03, fitted; no sourced loss
   factor for float glass was found). **No setting reaches both the absorption and the transmission
   of a pane.** Concrete (absorption 0.01–0.02 → 34–40 dB of leaks) is expected to be the same; not
   measured.
2. **`coi*` needs `att`.** A negative `coiA` with `att = 0` does nothing (`interface_gain` returns a
   gain of one for a negative total loss; measured, identical rows at 1 and 2 kHz). A coincidence dip
   needs an `att` baseline at least `|coiA|`, which moves the level everywhere else.
3. **A pole on a rigid mass-law body breaks the closed verdict.** A pole has loss at every frequency,
   so the layer is never closed and the open-face quotient (2.3) blows up: the 4 mm pane with a pole
   passes up to +135.6 dB, its two ports read up to 3.6e13, its traced solid −10 to −35 dB, a band
   through `ray_progress` sits 3.3 dB off the single frequencies.
4. **The bounce compensates the leaks** (2.5): below a panel's knee it returns what the two leaks
   took; at a sharp resonance the coherent sum over the bounces exceeds one (A1).
5. **Meshed double leaves cannot resonate.** Two 4 mm sheets 16 mm apart pass the product of the two
   panes with a smooth cavity series and no mass-air-mass resonance (the transfer-matrix model has a
   30 dB peak at 211 Hz): the sheets' coefficients carry no mass reactance. They read −105 dB at 1 kHz
   (four leaks) against EN 35 dB.

## 4. The knobs a published material needs

| quantity (per band) | knob | settable on its own |
| --- | --- | --- |
| room-side absorption of a rigid face | `tf`, `tfB` (power law), `tfP`, `tfF`, `tfQ` (a resonant peak) | yes (the peak: every lining and pane row of round 1 within 0.017 in α, cork board 0.076) |
| transmission level | `fRef` with `thk` | yes: the plate passes about `τ` where `τ ≪ a` |
| transmission slope | `m` | yes |
| coincidence dip | `coiF`, `coiQ` (bending stiffness in `X₀`), `coiA = 0` | yes |
| mass-air-mass dip of a double or triple unit | `res*`, `resS > 0` | yes |
| a panel's first mode, the stiffness region below it | `res*`, `resS < 0` | yes |
| extra loss per entry (frame, seals) | `att`, `attB` | yes; one per bonded run (A5) |
| porous and empirical absorbers | `a`–`d`, `alpha` (wave media), two-parameter mapping | yes (`md/wave.md`; section 1) |
| bonded layers | the plate rule | yes |
| reflection phase of a hard wall | the two-parameter mapping (N7) | yes; the phase of the pad and of a resonant absorber is open (A12) |

Out of reach whatever the knobs: bridging of a meshed double leaf; pane modes below about 50 Hz (a
1.25 m × 1.5 m pane: 10.7 Hz for 4 mm, 16.0 Hz for 6 mm) beyond the one mode of `resS < 0`; flanking.
The fits of round 1 (report section 3: all 13 constructions with the plate rule, `res*` and the bending
stiffness in `X₀`) were measured on patched copies of this code. The one change made after them, the
plate rule in the leak chain (report 5.2), acts only on a mass-law sheet glued to a medium without the
mass column that the wave cannot enter, which none of those constructions has.

## 5. Candidate changes to the machine (history)

**Built in round 1** (section 10, N1–N4, N6, A10): M1 as R1h, M2, M3 as B4s (the capped dip and the
single-panel form, chosen by the sign of `resS`), M4a; the exit fix, the held exit judged by the
invariant of air, `alpha` over the held path, three `tf` columns; the plate rule also in the leak chain.
`src/ray_mesh_interact.cpp` 5082 lines against 5016 at bb336dc (+230 / −164). The text below is the
candidate description as it stood before round 1.

All gated on `m > 0`. Measured on what-ifs in `build_linux/scratch/i47` where stated.

**M1 — `tf` on a closed mass-law plate is the room-side absorption; transmission comes from the
plate.** The reflect port stays `1 − tf`; what crosses is no longer `tf × plate × tf`. Passivity needs
the absorption at least the transmission, which published data always satisfy. Three rules, none
chosen (`τ` = the plate term):

- **R1h:** the incoherent series of face and plate, `T = tf·τ/(tf + τ − tf·τ)`: `T ≈ τ` where `τ ≪ tf`,
  `T → tf` for a light panel or at a resonance, always passive, smooth.
- **R1m:** `T = min(tf, τ)`: exact where the data are physical, a bound with a kink.
- **R2:** one leak, `T = tf·τ`: half the double role remains.

Each is formed on the plate's own columns at the event that closes the leg, so a sheet and its traced
solid can share it. A sheet could also reflect `1 − max(tf, τ)` (an absorption peak where the plate
passes); a traced solid's entry face is met before its thickness is known (not confirmed with a
marker), so it cannot. **Not measured.** It restates 2.3 (the pad's inward leak, the flat leak, the
leak chain), the held chain's per-face leaks (2.4) and the pair rule at mass-law faces; it moves most
acoustic rows, including every rigid wall without `tf`, which passes nothing today and would pass its
mass law.

**M2 — nothing is reflected back into a mass-law body** (no internal bounce; the body is a lumped
plate). Measured as B4k on the window scan (four walls, types 1, 2, 4, 5, 0° to 80°, 30 frequencies
from 150 Hz to 1.5 kHz): sheet on solid within 0.0063 dB, no row with both ports above one. Cost in the
suite: 2 cases and 53 assertions fail (*Ports of a mass-law face from inside - the reflect port is the
other direction's*: the wood | concrete face from inside then sums 0.556 at 70° instead of one;
*Bonded mass - the plate at any depth where the slopes agree*), 774 rows move in 5 tables (the bounce
table of the plate-depth case up to 8.84 dB, the reflect port of a lining met by a held wave gone, two
`ray_progress` tables with fewer paths). Natural together with M1; without M1 it makes light panels
more opaque (2.5).

**M3 — `res*` on a mass-law material is the plate's mass-air-mass resonance.** The pole leaves the
permittivity of a mass-law material (`eta()` adds it only where `m = 0`), so a rigid body stays closed,
and the plate's impedance ratio carries it:
`X₀ × min(1, |1 − (f·c/F)² + j·f·c/(F·Q)|)` with `c` the cosine in air (the cavity is stiffer by `1/c²`,
so the dip moves up as `1/cos θ`). `resS` only switches it on; it cancels out of the factor. Variants
measured as real code (lines against today's 5016):

| | what | lines | suite rows moved | window |
| --- | --- | --- | --- | --- |
| A | pole ignored on `m > 0` | 5018 | 0 | resonance invisible |
| B | factor without the angle | 4999 | 9, rounding | dip at F at every angle; 4 rows of types 1/2 with both ports at 1.32 |
| B4 | factor at `f·c` | 5001 | 9, rounding | on the double-unit model at every angle (45°: 1.62 / 5.64 / 9.24 / −5.24 / −20.18 dB against 1.65 / 5.93 / 10.32 / −5.18 / −20.62); +18 dB/oct above F |
| **B4m** | B4 capped at one | 5002 | 9, rounding | the dip only; with Q 3–5 −4.1 to −7.1 dB at 250 / 500 Hz against EN −7 / −4, the total mass elsewhere |
| C | closed verdict without the pole | 5023 | 0 | oblique sheet as if no pole, solid 13–20 dB apart; `resS` 1 turns glass into a wave medium |

The 9 rounding rows come from writing the mass block of `medium_gain` as one line on `mass_X0` /
`mass_TL` (7 residuals of the mass-law angle table 3.6e-15 → 7.1e-15, two printed zeros change sign).
B4 and B4m: both ports at most one on 5,400 window rows; a band through `ray_progress` on the single
frequencies within 0.0017 dB from either base frequency; hit codes, state words and directions those of
the window without a pole; no step across 0.00167°; walls without the mass column (ε 4 with a pole,
with and without `tf`, types 4, 5, 1, 2): 0 of 2,304 rows moved. Limit: with a sharp resonance (Q 30,
plate term below one at `F/cos θ`) the traced solid's followed bounces sum coherently to up to 2.48
(path powers at most one) and 11.3 dB from its sheet; with Q 2–5 no row. M2 removes it.

**M4 — coincidence.** (a) In the plate term on a mass-law material:
`X₀ × |1 − (f/fc)²·sin⁴θ·(1 + jη)|`, `coiF` = fc, η from `coiQ` — the bending stiffness of the plate
model that put 4 mm and 6 mm on EN 12758 within 2.5 dB; the diffuse-field dip comes from the angle.
(b) Let a negative `coiA` act without an `att` baseline. Needs a marker run first: does any case carry
`coi*` together with `m`? Not measured.

## 6. Open items

**Status after the round-1 source change (2026-10-15).** Closed: A1 (`res*` in `X₀` and M2: on the 84 rows of the resonance
case `T + R` is at most 0.9951), A3 (`alpha` over the held path: sheet on traced solid within 0.001 dB), A4 (K \| W one
plate, both sides and twin 0.0000 dB), A6 (no pair rule and no clamp at mass-law faces), A7 (a mass-law
body holds every tracer; the exit into air is type 1 at 0° as at 0.1°; the dense phase rows on the held
reference), A10 (three columns, in the OBJ / API column lists), A11 (both forms of `res*`;
the low octaves take the model spread, P7). A5: a rule of use, one `att` per bonded run. A2: the lining
sheet is on its traced layer within 0.015 dB for every tracer; the two sides of a lossy lining differ
by the lossy medium's convention (section 7). Open: A8 (low priority), A9 (after this round), A12 (the
two-parameter mapping is in the columns; the phase of the pad `1 − tf` and of a resonant absorber is
not carried). The cells below are the statements as they stood.

| id | statement and size | where it goes |
| --- | --- | --- |
| A1 | **A mass-law panel with a resonance** (a window): today it creates energy (3.3) | M3 (B4m); the solid at a sharp resonance by M2 |
| A2 | **A layer without the mass column behind a held wave.** A coherent lining on the far face of a rigid wall is entered at the Snell cosine of wall \| sheet, formed from the wall's impedance mapping; the same layer as a traced solid behind the wall runs at the incidence angle without the bent flag and meets its own far face past its critical angle. The two sides of a lining differ by 0.22 / 0.59 / 0.47 / 0.96 dB at 10° / 30° / 60° / 75° (dense, `eps 2.50 − 0.80j`) and 0.07 / 0.89 / 0.62 / 1.73 dB (porous, `eps 0.31 − 0.20j`); the dense traced layer passes nothing from 60°; a stack inside a rigid wall is 3 to 5 dB off its solids. The lining met first (a glued entry from air) describes the wave Round 1 (report 3.9): the M1 build as measured had a defect here, a coherent lining glued on the far face of a rigid wall 10 dB below its lined side (the wall's own face kept the transformer while the leg paid the plate rule; one condition fixes it). Judging and bending a held exit into a wave medium by the invariant of air (and writing the bent flag there, +8 lines, copy R1hG) puts the lining sheet on its traced layer within 0.01 dB and tracer 5 on 4; the two sides then differ by 0.06–1.34 dB (dense) and up to 2.78 / 4.57 dB (porous, one / two-parameter wall, 75°), exit side higher, the wave convention of a lossy medium. The solver half (layer cosines from the air invariant) breaks the porous sheet at 60–75° (+10 to +15 dB) and is not proposed. | re-examine after M1 / M2: what face a mass-law half-space and a wave layer form |
| A3 | **The path of `alpha` on a lumped plate.** A rigid mass-law sheet charges `alpha` over `thk/cos` with the in-sheet Snell cosine, which is evanescent: 22 mm with 100 dB/m pays 2.2 dB at 0° and the 400 dB cap from 10° on (in air and on `eps 2.50`), 0.012 dB on a lossy host, while its traced solid pays 2.23 / 2.54 / 4.40 dB at 10° / 30° / 60° (the held path). The same question for a tunnelling coherent layer is the open item of `md/wave.md` (8a) Round 1 (report 3.9): charging `alpha` over thk / cos in air in `qd_mass_amp` (0 lines) puts the sheet on its traced solid at every angle and tracer (2.19 / 2.22 / 2.52 / 4.31 / 11.74 dB at 0–80° for 22 mm at 100 dB/m). | one path: `thk` over the cosine in air (the held one), in `qd_mass_amp`, the held chain and the layer phase; no row carries it today |
| A4 | **Both faces upward.** A mass-law lining behind a held wave whose two faces both lead towards the higher index (wood on concrete) stays on the composite at the Snell cosine of two mappings: +0.002 to +0.008 dB between its two sides (scalar), −0.019 dB in HH | drop the direction condition of the held gate (one condition), or record |
| A5 | **The interface gains of a bonded pair add in dB.** `att` and the coincidence term are owed once per medium entered, so a panel bonded to a wall pays both in full. Zero while `att` is zero; otherwise the lighter body's whole level Round 1 (report 3.9): 3 + 3 dB of `att` on a bonded pair cost 6.00 dB as two solids, sheet on solid and stack, today and under M1. | a convention: the larger `att`, an impedance-like combination, or both; with M1 and M4 |
| A6 | **A lossy mass-law sheet and its traced solid take two `tf` rules,** and the sheet's rule holds the one clamp. The sheet keeps the pair rule on the reflectance (2.3), its traced solid the transformer at every open face. A scratch scan (mass term at nothing, lossy mass-law sheet with `tf` on a rigid wall, 1–50 mm, 5°–85°): the sheet's ports at most 0.99 together, 3.3–10.6 dB from its solid on transmission, 13–35 dB on reflection, mostly the solid's bounces. 120 clamped calls in five cases, all types 1 and 2 | disappears with M1 (no pair rule at mass-law faces) |
| A7 | **The undeviated bit is set at normal incidence** for a mass-law exit without a refracted direction (resolved type 33 at 0°, 1 at 0.1°; a flag, no gain). A mass-law solid with ε above 1 refracted into with type 5 forms its mass term at the in-medium cosine: 0.458 dB below the undeviated tracer at 30° on 100 mm `eps 1.50`; sheet and solid of a non-rigid mass-law panel 9.6 dB apart at 75°, 37 dB under B4 at its resonance Round 1 (report 3.9): under M1 as built in phase 2 the non-rigid `dense` body (`eps 1.50`, m 20) passed nothing from 55° (its held exit judged by its own invariant: NaN phase rows from 60°); judged by the air invariant (R1hG) the rows are finite, +15.2 / +26.2 / +36.4° at 60 / 75 / 85° against the test's refracted reference. Today a traced rigid solid in the two-parameter mapping lets tracer 5 refract in below its critical angle (23.7 dB glass, 28.6 dB gypsum from tracer 4); every M1 copy 0.000 dB. | a mass-law body holds the wave whatever the tracer; cleanup |
| A8 | **A deferred medium displaces the thickness code**: a nested or overlapping object met on a leg that carries a code replaces it, and the leg pays its own term in full: +37.50 dB (a solid on a wall with an object nested in it) and +23.10 dB (a lining on a solid overlapping a second one), from one side only; +25.8 / +11.6 dB at 75° | low priority; a freed bit could flag "an object pending"; may be dropped |
| A9 | **`html_docu/quadriga_lib_material_model.md` is out of date** (the three `tf` rules, the closed layer, the held wave, the mass cosine, the plate rule, the state layout) | after the machine is settled |
| A10 | **A peaked room-side absorption on a rigid face.** The reflect port of a rigid face is `1 − tf` with `tf` a power law in frequency: falling or flat absorption is expressible (window glass within 0.13 in α), a peak is not. Physically the room-side absorption is what the panel passes plus what it dissipates, so it peaks where the panel moves most: a leaf over a cavity (a panel absorber, `f₀ = 60/√(m′d)` Hz, e.g. 85 Hz for 10 kg/m² over 50 mm), a double unit at its mass-air-mass resonance, a pane at its fundamental. Candidates, none measured: (a) with M1 a sheet reflects `1 − max(tf, τ)`, so it absorbs what its plate passes, a peak at a resonance for free; a traced solid can do the same only if its entry face can know its thickness (N5); (b) a resonant shape on `tf` itself (a Lorentzian like the coincidence term, on the absorption, on a mass-law material); (c) as today, a resonant absorber modelled as a wave medium with a pole. To analyse: which constructions of the reference classes show a peak, how large and how wide in octaves, and whether (a) alone reaches them Round 1 (report 3.8): (a) puts window glass at TL 6.7 / 8.9 dB at 125 / 250 Hz (EN 12758: 17 / 20): rejected; (b) a resonant peak on the absorption reaches every lining and pane row within 0.017 in α (cork board 0.076); (c) not measured. | analysis, then a decision; may add a column |
| A11 | **The low octaves.** The bands now reach down to 31.5 Hz, which brings the region below a panel's fundamental into range: a 4 mm pane of 0.6 m × 1.2 m resonates at 34 Hz, of 1.25 m × 1.5 m at 10.7 Hz; a heavy wall's coincidence lies near 100 Hz. Below its fundamental a panel is stiffness-controlled and its transmission loss rises towards low frequencies; at the fundamental it dips. The mass term has no stiffness region: it is zero below `X₀ = 1` and falls with frequency. Candidates: the single-panel form of the M3 factor, `X₀ × |1 − (F/f)² − j·F/(f·Q)|` (stiffness below F, mass above), which `res*` could select instead of the double-unit form; or a documented floor. Published transmission data mostly start at 100 Hz, so the low octaves need model references (and the ray picture itself is weak there: wavelengths of 5 to 11 m). Measured in round 1 (report 3.1): no first-principles model (finite plate, coupling loss into the frame, thick plate, modal sum) reaches the measured heavy walls within the guideline with physical constants (−7.1 dB at 125 Hz and +7.3 dB at 4 kHz on 152 mm concrete); model forms spread 4.5–13.9 dB at 31.5 / 63 Hz for heavy walls and 5.2–13.5 dB for light panels, whose low octaves depend on the panel's size (a lab pane is 7.8–11.2 dB above the infinite plate at 31.5 Hz) | part of the expressibility test; decides whether `res*` needs two forms; the low-octave reference is open (report 3.2) |
| A12 | **The reflection phase of a rigid surface.** A rigid half-space of the one-parameter mapping reflects with a phase of twice the angle (+20° at 10° … +170° at 85° for `eps 2.401e-9`), where a hard wall reflects with a phase near zero at every angle. Coherent sums in a room at low frequency (interference, modes, speaker placement) do not match a measurement without it. The two-parameter mapping (`ε = n·z`, `μ = n/z`) is exact for air \| water (report 1.8: 4.33e-4° at 13.38°, 1.53° at 85°); for rigid materials it moves the critical angle to 3.5° (glass) … 9.4° (gypsum), so a rigid sheet is open below that: not measured. Also to read: the phase of the pad (`1 − tf`) and of a mass-law sheet's reflect port Round 1 (report 3.4, 3.5): with the two-parameter mapping a hard wall's phase is within 0.1° to 60° (3.1° at 89° for glass), a porous layer on a wall absorbs within 0.003 of the exact layer (one-parameter backing: up to +0.39 at 125 Hz, +0.53 at 60°), water is exact traced and through `ray_progress`. A panel absorber's phase swings ±18° through its resonance (model), which the pad does not carry. | priority (Stephan, 2026-10-15); measured with N7 |

## 7. Accepted limitations (acoustic, until the machine changes)

- The mass term steps at an invariant of one (7.66 dB at 0.99, 22.60 dB at 1.01 behind
  `eps 1.50 − 0.40j` for a 26.8 dB sheet); never reached by a ray from air through a face parallel to
  the plate.
- A ray the state does not know to be bent (a source inside a host, a caller without state) keeps its
  segment's cosine in a mass term: 0.21 / 0.96 / 3.00 dB at 10° / 20° / 30° in `eps 2.50`.
- Behind a lossy wedge the cosine in air is the real-angle one (0.002 to 0.096 dB).
- The thickness code is quantized (≤ 3.0e-3 dB three and four bodies deep at `m` 30) and ends at
  100 m (an equivalent thickness of a heavy body in a light material can reach it).
- Mixed slopes fold from one side: over the plate rule up to +7.66 dB (`A20 100 | B30 50 | C20 100`,
  light) and +4.30 dB (`… | C30 50`, heavy, from the other side); a table that gives every bonded
  material one slope never reaches this.
- **The state is one level deep.** A traced plate re-forms its first face's absorption at the exit from
  the previous-medium word, which names only the body before: a plate of three or more bodies pays the
  rule within +3.02 / −3.01 dB on one slope (*plate depth* S: a sheet inside a wall between two bodies;
  plaster \| brick \| plaster), and a wall with an ordered stack inside it is up to 3.01 dB from its
  traced twin (*Held lining* E: K \| T W 3.01, K \| W T 1.76, T \| K W 1.25 dB). The per-ray state
  cannot grow (section 2.2's two words).
- The plate near its knee can exceed the dB sum (+0.44 dB at 70° for `X₀` 6 and 2.4).
- A plate passes at most what its face absorbs: below its knee a light panel passes about `a`, not
  one; what it neither passes nor reflects (`a − T`) is absorbed.
- The first face of a leak is flat: through the plate rule 0.00013–0.00094 dB between the two
  directions of a wood panel on concrete without its mass law (10°–70°). The step of a wave becoming a
  leak is gone (0.0000 dB at 0.0084°). A non-rigid mass-law layer becoming closed is a step (`eps 0.50`,
  22 mm, `tf` 0.2, mass at nothing, inside `eps 2.50 − 0.8j`: −5.85 dB at 44.9°, −7.49 dB at 45.1°;
  −5.39 / −6.99 dB in the lossless host).
- A rigid face without `tf` absorbs nothing, so its plate passes nothing, from either side (rule of use:
  `tf` with `m`).
- **A non-rigid mass-law layer beside a lossy coherent layer** depends on the order: 0.107 / 0.127 /
  0.201 dB between the two orders at 10° / 30° / 60° (*Closed plate* E), 0.016–0.024 dB in a stack.
- **Lossy media split the flux** (the wave convention, `md/wave.md`): a face next to a lossy medium can pass
  more than one in flux. The two sides of a lossy lining on a rigid wall differ, the wall's side
  higher: 0.07–1.38 dB (dense `eps 2.50 − 0.80j`) and 0.52–2.70 dB (porous `eps 0.31 − 0.20j`) at
  10°–75°, scalar. A mass-law sheet without `tf` behind a lossy host, under a closed outer pair, takes
  the smaller face, one: −0.39 to −1.83 dB against the face's flux gain (*Mass cosine* D, invariant
  1.01–1.45).
- **An event can pass more than one.** A plate's exit gives back the absorption of its first face, up to
  `1/a_in` (+8.85 dB on wood 50 mm \| concrete 50 mm); no path passes more than the rule. `ray_progress`
  stops a continuation below `min_gain_dB` (default −140 dB) on the running gain, which inside a plate is
  up to `−10·log10 a_in` below the level the path will reach (10 dB for `tf` 0.1, 20 dB for `tf` 0.01), so
  a path ending within that margin above the threshold can be cut. Not measured on a scene.
- The HH rows (type 1, k 3) of a mass-law material are a formal EM row and no use case: they part from
  the traced twin by up to 3.9 dB (porous lining) and the two sides by up to 7.5 dB.
- The level of a held chain has no reference but its solid twin; the crossings are formed from
  impedance mappings.
- `ray_mesh_interact`'s own mass-law lining is approximate in a lossy host (the redo with state is
  exact).

## 8. Side findings

- `src/quadriga_lib_material_helpers.hpp` carries a second `medium_gain_impl` / `medium_loss_dB` that
  reads the mass column and a second `eta_resonance`; nothing includes it. Remove or reconcile.

## 9. How the machine is accepted: the expressibility test

The machine is accepted when every reference construction can be reached by some setting of the
columns, as sheet and as traced solid, within the guideline tolerance in every band. The stack will be
calibrated against measurements later; this test only shows that the knobs exist.

- **Reference:** published measured data where they exist (EN 12758 generic values, manufacturer
  sheets, measured absorption coefficients); physical models only where no measurement covers a band
  (the low octaves) or to fill in a construction's shape.
- **Bands:** octaves from 31.5 Hz to 8 kHz (the mapping's range is 30 Hz to 8 kHz). Published
  transmission data usually start at 100 or 125 Hz and absorption data at 125 Hz; below that the
  references are models (item A11).
- **Tolerance, a guideline and no hard bound:** transmission loss within about ±3 dB, absorption within
  about ±0.05, coincidence bands reported apart. It may be tightened or loosened as the work goes on.
- **Classes:**
  1. light single panels: glass 4 and 6 mm, gypsum board, a wood panel or a door;
  2. heavy walls: 100 to 200 mm concrete or masonry;
  3. a double or triple glazing unit, lumped as one sheet;
  4. a porous absorber on a rigid wall (absorption);
  5. a lining on a rigid wall;
  6. a layered wall, plaster | brick | plaster;
  7. **air | water**, reflection first, transmission second. Water is not a plate: it is a half-space
     in which sound propagates (`Z ≈ 1.48e6` rayl, `c ≈ 1480` m/s, typical values still to be sourced), a wave medium without the mass
     column. The one-parameter mapping (`ε = (Z_air/Z)²`, `μ = 1`) fixes the reflection but puts the
     critical angle at about 0.016° where water's is about 13° (`sin θc = c_air/c_water`). With the
     permeability column both can be right: `ε = n·z`, `μ = n/z` with `z = Z_air/Z`, `n = c_air/c`.
     To be checked: whether the wave side then gives water's reflection, its critical angle and its
     transmission, and what this two-parameter mapping does to the other rigid materials.
- **Procedure:** for each class, fit the knobs on a formula copy of the library (the window's sits on
  the walker to 0.011 dB); confirm the best fit through the walker as sheet and as traced solid, and
  through `ray_progress` as a band. Report the residual per band, today and per variant (M1: R1h, R1m,
  R2; M2; M3 B4m; M4 a, b).
- **Invariants per variant:** both ports at most one, sheet on solid at the relaunch bound, no step in
  angle, a band on its single frequencies, the `[wave]` baseline and twins untouched, line counts.

## 10. Decisions

**Taken (Stephan, 2026-10-14):**

- P1. **The reference is measured data.** Pragmatic: the whole stack will be calibrated against
  measurements later. Where theory and measurement differ, the knobs have to reach the measurement
  (e.g. B4m over B4 for double glazing above its resonance).
- P2. **Octaves, including the low ones** (31.5 Hz up). The tolerances are guidelines, not hard bounds.
- P3. **The reference classes of section 9, with air | water added.**
- P4. **A peaked room-side absorption on a rigid face needs analysis:** item A10.
- P5. **The settled acoustic conventions are open:** the two leaks of a closed plate (2.3), the bounce
  inside a mass-law body (2.5), the two ports of a face seen from inside a mass-law body (2.4). M1 and
  M2 are candidates for a change, measured like the others.

**Taken (Stephan, 2026-10-15), after round 1 step 0 (report 3.1, 3.2):**

- P6. **The reflect side first.** The goal is in-room wavefields (speaker placement, room treatment): room-side
  absorption per octave, the peaked absorption (A10) and the reflection phase (A12, with N7) come before
  transmission, which has looser requirements (sound insulation across rooms).
- P7. **Bands without a measurement take the model spread as their tolerance** (report 3.1: heavy walls 4.5–13.9 dB,
  light panels 5.2–13.5 dB at 31.5 / 63 Hz). Flanking through the building is out of reach of a sheet (section 4).
- P8. **The rigid-wall reflection phase is an open item** (A12), no longer an accepted limitation.

**Taken (Stephan, 2026-10-15), after round 1 phase 3 (report 3.10); fixtures first, then the source:**

- N1. **The plate rule is R1h**: a closed mass-law plate reflects `1 − a` and passes `a·τ/(a + τ − a·τ)`, `a` the
  smaller absorption of its two outer faces, `τ` its bodies' term (report 2.4, 3.6).
- N2. **M2 with N1**: nothing is reflected back into a held mass-law body; a traced plate passes in one path.
- N3. **`res*` on a mass-law material is the plate's resonance in X0 (B4s)**, not a permittivity pole: `resS > 0` the
  dip of a double unit or lining (capped at one), `resS < 0` the panel's first mode (stiffness below `resF`).
- N4. **M4a**: on a mass-law material `coiF` and `coiQ` are the bending stiffness of the plate in X0
  (`|1 − (f/coiF)² sin⁴θ (1 + j/coiQ)|`); M4b is dropped; rule of use: `coiA = 0` on a mass-law material.
- N5. Closed (measured; no change).
- N6. A2: the exit fix and the held exit into a wave medium judged and bent by the invariant of air (with the bent
  flag); A3: `alpha` over the held path in `qd_mass_amp`; A4, A6: closed by N1; A5: keep the sum, rule of use one `att`
  per bonded run; A7: closed by N1 and the held exit, the non-rigid phase rows restated to the held reference.
- N7 / A12. **The two-parameter mapping** (`ε = n·z`, `μ = n/z`) for every acoustic material.
- A10. **A resonant peak on the absorption, three new columns** `tfP`, `tfF`, `tfQ`:
  `tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f − tfF)/tfF)²)`, clamped as today.
- A11. Both forms of `res*` (N3); the low-octave reference is the model spread (P7).

**Built (2026-10-15)** as decided above, fixtures first; the source change, the restated cases and every
moved table are in the report, section 5. Not committed.

**Measured in round 1 and decided above (the list as it was set on 2026-10-14):**

- N1. The M1 rule (R1h, R1m or R2), on the expressibility residuals and the invariants.
- N2. M2, together with N1; its suite cost on today's conventions is known (2 cases, 53 assertions,
  774 rows).
- N3. M3 (B4m by P1, B4 for comparison), the semantics of `resS` on a mass-law material (a switch, as
  measured), guidance for `resQ`, and whether `res*` needs the single-panel form for the low octaves
  (A11).
- N4. M4 (a) or (b), after a marker run for `coi*` together with `m`.
- N5. Whether a traced solid's entry face can know its thickness (a marker run), for A10.
- N6. A2, A3 (with `md/wave.md` 8a), A4, A5 and A7 re-examined under the new conventions; A6 goes with
  M1; A8 stays low priority; A9 last.
- N7. Air | water: the two-parameter mapping with `μ` on the wave side (section 9, class 7).

## 11. Rig

- `build_linux/scratch/i47`: `panel.py` (thin plate and double unit), `fit.py` (EN octaves, absorption
  rows), `libwin.py` (formula copy of the rigid sheet), `mk47.py` (switch copy with a marker),
  `mkv47.py` (variants A, B, B4, C as real code), `rmi_B4m.cpp`, `rmi_B4k.cpp`, walkers `probe3.cpp` /
  `probe4.cpp` (P4 collects only outside the structure, for meshed cavities), band probe `rp2.cpp`,
  scans `runwin.sh`, `runx.sh`, `runscan.sh`. `NOTES.md` has the sources.
- `build_linux/scratch/ts1`: the marker and split tools of the test split.
- `build_linux/scratch/a1`: round 1. `wi/` the variant generators (`mkc.py`, `mkc_p3.py`) and builds
  (`build.sh <lib.cpp|-> <probe|-> <abs out>`, `bld1.sh`), `p3/` step 0 to 7 rigs, `p4/` fixture
  summaries, `p5/` the restatement scripts and `fails_by_case.py`, `p6/` the per-table diff
  (`tabdiff.py old new`, `rows.py old new <header prefix>`, `colmax.py run <prefix> col|a-b`), the
  what-if without the leak-chain fix (`rmi_noleak.cpp`) and the baselines before this change
  (`old_baselines/`).
- The test instruments and traps of `md/wave.md`, section 11, apply here too. Two more: `build.sh`
  needs absolute output paths; two traced solids with a cavity explode the outside-only walker's tree.
