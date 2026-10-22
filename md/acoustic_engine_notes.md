# The acoustic interface engine of quadriga-lib: development notes

These notes record how the acoustic interface engine of quadriga-lib was built between January and October 2026: what was tried, what was kept or reverted and why, the evidence behind each decision, the failures and what caught them, and the limits at the close of the work. They are the surviving record of that work; the code, the tests and the public material-model page describe the result, these notes the path to it.

## 1. Purpose and scope

The engine computes what happens to a ray at a surface and inside a body: reflection, transmission, refraction and the loss in between. It lives in `src/ray_mesh_interact.cpp` (the public functions `ray_mesh_interact` and `ray_state_update`) and is driven by `ray_progress`, `ray_init` and `ray_commit`. It was written for radio propagation; the work recorded here extended it so that the same engine describes sound in rooms, while its electromagnetic (EM) results stay correct.

The purpose is room acoustics with a radio ray tracer. Good acoustic simulation tools are scarce or expensive, and a ray tracer with beam tubes, coherent paths, polarization and wideband channels already exists in the library. The work began as a personal project of an EM modeler and became more serious over its course, also because the acoustic work improved the EM side (section 11 states how far the record supports that). The intended first subject is a residential listening room of 4.6 × 5.3 × 3.1 m: loudspeaker placement, room treatment and multi-seat room correction.

The larger project has six steps: (1) literature research, (2) the interface engine, (3) the material tables, (4) a reference simulation, (5) measurements, (6) a journal paper. These notes cover step 2 only. The literature research of step 1 is in the separate repository `/sjc/accoustic_research` (`/sjc/accoustic_research/review.md`, `/sjc/accoustic_research/data/material_properties.csv`, `/sjc/accoustic_research/references/index.csv`); these notes cite it where the engine used it.

How to read the notes:
- Commits are given by their short hash; test cases in italics, by the name they carry in `tests/catch2_tests/test_wave_*.cpp` and `tests/catch2_tests/test_acoustic_*.cpp`.
- "Page *n*" is section *n* of `html_docu/quadriga_lib_material_model.md`, the public description of the model with every formula as implemented. "Review *n*" is section *n* of `/sjc/accoustic_research/review.md`; a source key in square brackets resolves in that repository's `references/index.csv`.
- Numbers are given as recorded at the time, with the commit or date; a number that a later change replaced is marked "superseded". Nothing was re-measured for these notes.
- Frequencies of acoustic quantities are acoustic frequencies; the engine works at the EM-equivalent frequency $f_\text{radio} = f\,c_\text{light}/c_\text{sound}$ (1 kHz ≈ 0.874 GHz).
- "EM" means the interaction types 0 (reflection), 1 (transmission) and 2 (refraction) on EM materials; the scalar (acoustic) types are 3 (reflection), 4 (transmission) and 5 (refraction). "Wave medium" is a material without the mass column (`m = 0`); "mass-law material" one with it. The glossary (section 12) defines the engine's own terms.

## 2. The starting point

### 2.1 The EM engine of 2024–2025

`ray_mesh_interact` was added on 2024-01-10 (bb2400e, 736 lines): the interaction of a ray, or a beam tube of three rays, with a triangle mesh at one hit. The coefficients are the Fresnel coefficients of ITU-R P.2040 for a complex permittivity given per material as power laws in frequency, $\varepsilon' = a f^b$ and $\sigma = c f^d$ (f in GHz). The polarization is carried as a Jones matrix. Three interaction types existed: reflection, transmission (the segment keeps its direction, so the geometry is the same at every frequency) and refraction (the segment bends).

Later in 2024 and 2025 came the transmission gain (7a2bb7f), a diffraction gain (8ac052a), the slab method for bounding boxes (47fcdc7, speed only), material names compatible with NVIDIA Sionna (f2505a9, 2025-01-08) and reading BSDF data from OBJ files (00f8acc). Whether a ray was inside a body, and the loss inside it, were handled by a state machine embedded in `calc_diffraction_gain`; it moved into `ray_state_update` in June 2026 (section 3, phase 5).

No publication on the ray tracer exists yet; this section rests on the commits.

### 2.2 The parameter mapping of January 2026

The first acoustic piece was a stand-alone feasibility attempt written by Claude, with quadriga-lib in mind but independent of it, because it was not yet known whether the idea could work. It came into the repository on 2026-01-31 (beed81a, v0.10.2) as a tool folder (a 915-line converter, a README, modeling guidelines and a 28-row material table), together with custom materials in `obj_file_read`, whose CSV format the table used.

Its idea: sound at kHz frequencies and radio waves at GHz frequencies have comparable wavelengths, so a radio ray tracer can trace sound if the frequency is scaled by $c_\text{light}/c_\text{sound} \approx 874{,}636$ and the materials are translated.

Its assumptions:
- Permeability one, so one complex permittivity per material. Matching the normal-incidence reflection of an impedance $Z$ gives $\sqrt{\varepsilon'} = Z_\text{air}/Z$. Rigid materials then become optically rarer than air by eight to eleven orders of magnitude (glass $\varepsilon' = 8.96\times10^{-10}$, concrete $2.401\times10^{-9}$ in its table).
- Three classes: rigid reflectors (reflection exact at normal incidence), porous absorbers (Delany-Bazley, a stated RMS error in $|R|$ of about 0.13) and empirical absorbers converted from Sabine coefficients through $|R| = \sqrt{1-\alpha}$ (about 0.21).
- Requirements on the tracer: watertight meshes, refraction through volumes, and correct handling of $\varepsilon < 1$ including total internal reflection. Geometry rules: rigid walls at their real thickness, absorbers as boxes, people and furniture as volumes of an "effective thickness".
- A random-phase treatment was expected to average out coherent errors.

The mapping was never run in another tracer. What later measurements showed (section 6 has the details):
- A rigid half-space in this mapping reflects with a phase of twice the incidence angle (+20° at 10° to +170° at 85° for $\varepsilon = 2.401\times10^{-9}$), where a hard wall has a phase near zero.
- Air against a rigid body is past its critical angle from about 0.003°: a traced rigid body is closed by its first face, and a wave that does enter leaves along the normal.
- A thin layer in this mapping carries no mass reactance, so the wave physics yields no mass law.
- A porous layer on such a wall absorbs up to 0.39 too much at 125 Hz and 0.53 too much at 60°.

The lesson drawn at the close of step 2: mass-law media cannot be approximated well by the wave-physics model alone. The record adds a nuance. A light limp leaf can be a wave-medium sheet: in the two-parameter mapping of section 4 a 4 mm glass sheet without the mass column follows the limp mass law within 0.002 dB at 0° to 85° (500 Hz). A rigid body traced as a solid cannot: in either mapping it is rarer than air past a critical angle of a few degrees and is closed by its first face. Rigid walls therefore carry the mass column.

### 2.3 The early concept documents

Two design documents of the spring, now only in the git history, show the first plans:
- The concept for the scalar types (75cc82c, 2026-04-14): reflection from the TE coefficient alone, total internal reflection disabled for the acoustic types ("sound waves hitting high-impedance surfaces reflect normally"), transmission $1 - |R|^2$, two of the eight output elements used. Built in ce0d2c2; the disabled total reflection was reverted on 2026-06-16 (section 6).
- A calibration specification (d79c8d6, 2026-05-28): the "two-knob decoupling". Conductivity zero for every acoustic material; the real permittivity sets the reflection and only that, the penetration loss `att` and the in-body absorption `alpha` set absorption and transmission and only that. Eleven octave bands from 16 Hz to 16 kHz, results normalized to air, and a calibration loop of one material per chat. The material model changed the next day (c1c9076), and the decoupling did not survive the first acoustic model (section 6).

## 3. Timeline

Fifteen phases, from the EM origin to the close of step 2. Line counts are those of `src/ray_mesh_interact.cpp` at the commit named. Suite counts are "assertions / test cases".

### Phase 0: the EM engine (2024-01-10 to 2025-10-20)

bb2400e to b413743. The per-hit Fresnel engine of section 2.1, extended over two years for radio scenes. Kept as the core. Its energy convention and total-reflection verdict were rewritten in 2026 (phases 7 and 8).

### Phase 1: the parameter mapping (2026-01-31)

beed81a. The converter of section 2.2 and custom materials in `obj_file_read`. Retired: the two-parameter mapping replaced it (decided 2026-10-15 with 4720a92); the material-model page teaches only the two-parameter form since 23ad1b6; the converter and its table (in the data folder and as a byte-identical copy in the tool folder) were judged out of date on 2026-10-14 and used as a reference for nothing after that.

Within three weeks two more pieces of an acoustic chain arrived outside the interface engine: a parametric loudspeaker directivity model (`generate_speaker`, piston, horn and omni drivers; f1c51d8, 2026-02-17) and multi-frequency channel generation (78de7f7, 2026-02-18). With them the library had sources, materials and wideband channels: loudspeaker, room and listener.

### Phase 2: scalar waves (2026-04-14 to 04-15)

75cc82c (the concept), ce0d2c2 (973 lines). Interaction types 3 (scalar reflection) and 4 (scalar transmission): the pressure coefficient from the TE (admittance) form, a complex scalar in place of the Jones matrix, transmission from energy conservation, and no total reflection for the scalar types. Kept: the scalar types. Reverted: the disabled total reflection, restored for all types by one test in d1a117c (2026-06-16).

### Phase 3: the material columns and the first acoustic model (2026-04-24 to 06-07)

- e544a1f (04-24): nine columns, adding a penetration loss `att` with slope `attB`, an in-body absorption `alpha` in dB/m with slope `alphaB`, and a reference frequency `fRef`. c53da44: the in-medium attenuation as a public function.
- c1c9076 (05-29): a flexible table of up to 16 columns with seven new parameters for the acoustic mapping: `m` (a mass-law slope in dB per decade), `resF`, `resQ`, `resS` (a Lorentz pole on the permittivity, for Helmholtz, membrane and microperforated absorbers) and `coiF`, `coiQ`, `coiA` (a coincidence dip as a Lorentzian in dB).
- f50af9d (06-01): a name-keyed material table (a CSV column is a parameter, no plumbing) and 0-based indices in the C++ core.
- 4122706 (06-02, 1142 lines): the first acoustic model. The mass law as $\max(0,\ m\log_{10}((f/f_\text{ref})\,d))$ on the path length $d$, the coincidence Lorentzian on the interface loss, and the Lorentz pole on the Fresnel permittivity only.
- 1659eae (06-03): a complex permeability (columns `e`, `f`, `g`, `h`). One complex permittivity locks wavenumber and impedance together; with $\mu$ they decouple, $n = \sqrt{\varepsilon\mu}$ and $Z = Z_0\sqrt{\mu/\varepsilon}$. This is what later made the two-parameter mapping possible.
- 227aab6 (06-07): the mass law given an angle dependence (fixed again in fd86176, phase 7).

Kept: the name-keyed table, $\mu$, `res*`, `coiF`/`coiQ` (reinterpreted in 4720a92 as the plate's bending stiffness), `m` (reinterpreted in bc24b0c as a switch). Reverted: `m` as a fitted slope and `coiA` (bc24b0c, 2026-10-17); 0-based indices (c99c5f1, 2026-06-10: 1-based, 0 = air).

### Phase 4: the scalar transmission partition and the transmission factor (2026-06-02 to 06-05)

The rigid proxies of the one-parameter mapping ($\varepsilon \ll 1$) put a critical angle near 0° on every wall, so the Fresnel transmission $1 - |R|^2$ killed oblique transmission. Over two days four commits tried to work around it:
- 006b85e: scalar transmission as a "partition model", pure pass-through, with the isolation carried by `att`, the coincidence dip, the mass law and `alpha`.
- 57bfa17: energy-conserving transmission for crossings into a denser medium, pass-through otherwise.
- 747f46f: gated on the entered permittivity above one, because the relative test had fired at a rigid wall's exit and added a spurious flat −38 dB.
- 52a167c: one shared predicate for the ray tracer and the diffraction gain, which had drifted apart.

cc636a5 dropped the predicate on 06-03. On 06-05 edbbe24 introduced the transmission factor `tf` / `tfB`: the pass-through had set $T = 1$ on the transmitted ray while the reflected ray carried $|R|^2$, so a thin absorber emitted $R + T > 1$, up to +12 dB of spurious reflection at low frequency. `tf` moved power between the two ports with $R + T = 1$ enforced. Two follow-ups fixed `tf` under total reflection (8f229d8, b4666d8: the transmitted ray had stayed at zero past the critical angle, then gone NaN, then lost its phase).

Reverted: the partition (cc636a5) and the remaining dense-to-light pass-through ("honest transmission", d1a117c). Kept: `tf`, reinterpreted twice later (a transformer and a pad on wave faces, 2026-10-12/13; the room-side absorption of a rigid face, 4720a92).

### Phase 5: the state machine and the Snell refactor (2026-06-10 to 06-17)

- 566d6fe: `ray_state_update`, a batched per-ray state machine lifted out of `calc_diffraction_gain`, written from a design specification and with an added closed-form thin-slab resolution. 231187f: `Material` structs.
- c99c5f1: material indices 1-based, 0 = air.
- 235260f: the interface gain folded into `Material::interact_with`.
- bd26014: a "blind" verification suite, 28 cases and about 1800 assertions, written by a separate session from the design and test specifications without seeing the implementation, with independent references (P.2040 Fresnel, the Airy sum checked against a transfer matrix). It found that material $w$ read the properties of row $w+1$ and the top index read out of bounds, throws inside OpenMP regions (a crash), and a stacked-slab energy defect.
- 97163ce: `calc_diffraction_gain` ported onto the state machine (−621 lines).
- The Snell refactor in three stages, from a specification and a staged plan with one gate per stage: a737c08 (geometry and polarization helpers), f691bd5 (a refraction-correct continuation direction `path_dirN`), d1a117c ("honest transmission": the dense-to-light pass-through removed, transmission $1 - R_\text{eff}$; scalar refraction as type 5 with $t = 1 + r$; one total-reflection test for all six types; whole-tube total reflection; `tf` as a two-sided pair rule). Then dcf1220, 06f9d20, 7e3f349: the VBS correction in `ray_state_update` (each interface re-evaluated at the refracted angle the undeviated tracer does not trace).

Kept. The Airy factor and VBS became the slab series of phase 6.

### Phase 6: stateless hit, state machine and the ray-set functions (2026-07-14 to 08-06)

- 529d315 (2648 lines): the interaction model split into a stateless hit (`ray_mesh_interact`) and a state machine (`ray_state_update`) with three signed 16-bit words per ray. The machine owns the inside/outside history, the in-medium loss and the thin-slab factor $S = 1/(1 - r_n r_f \varphi^2)$, with VBS, an excess-phase term with walk-off for the undeviated tracer, and three resolve gates (parallel faces, round-trip survival, a pole clamp).
- 5a963c9: `calc_diffraction_gain` rewritten onto it (about 1010 → 600 lines).
- 220993d `ray_init`, 571c3c3 `ray_progress` (advance a set of up to about $10^8$ rays by one interaction; geometry traced once at the base frequency, the other frequencies recompute only the coefficients), f798cfb and a57d98c (a review round and a 22-case suite; four compaction and indexing bugs, one a segmentation fault), aa3b7c2 (one compaction, an authoritative subdivision flag, 400 MB less at $10^8$ rays), b766159 `ray_commit`.
- 97ae3cc: a mass-law material excluded from the slab resummation. Its permittivity is calibrated for surface impedance only, so resumming the bounces with its phase cancelled the mass loss and drove a thick partition toward transparency.
- 39ee6da: two phase errors exposed by an oblique acoustic slab case. The walk-off weighting was inverted (transmitted paths off by $1/\cos^2\theta$, a factor 2 at 45°), and total reflection hard-set the coefficient to +1, discarding a phase that runs from 0 to −180° between the critical angle and grazing, on the EM and the scalar reflection branches alike.

Kept. The pole clamp was removed in aecaf04 (phase 8).

### Phase 7: thin sheets and the exact slab (2026-08-24 to 09-09)

The thin sheets had an EM motive: customer scene models export fences, glass panes and signs as single sheets of triangles, not as solids.
- 2839bda: a thickness column `thk`. 25bc3de, 0251477: the refracted state tracked per frequency. `ray_progress` had reused the base frequency's Snell angle for every band, so in a dispersive medium the total-reflection test fired spuriously and zeroed whole bands.
- 260e71b, 70cdb44, 72de14f: a sheet (a zero-thickness face with a `thk`) resolved as near | slab | far in one event. Every sheet code had reached the kill branch of `ray_state_update` (gain exactly zero at every angle) before 72de14f.
- 2d80e99, b57db17, 1021f01: the sheet composite put on the exact analytic slab, measured against an Abelès transfer matrix rather than reasoned about. 2.97 dB at grazing → 0.031 dB; four further defects → 0.000000 dB at every angle; the exact complex in-slab cosine recovered by inverting the tracer's own map (to 2.3e-16, where the real projection was off by up to 0.126).
- 8970c0b: the thin-sheet grazing clamp removed. It had frozen the interface at the onset angle (transmission held 28.8 dB above the slab at 89.9°, the reflection at 0.473 where the slab reflects 0.997) and depended on how finely the wall was meshed (27.7 dB between two meshings of one pane at 89.99°).
- 78389b6: one per-channel normal-flux energy convention at every interface. The same crossing had reported different energy depending on whether the tracer bent the ray: 0.88 dB for air to $\varepsilon = 1.50$, 3.60 dB for $5.24 - 0.63j$, up to 17.56 dB for $25 - 8j$ at 85°.
- 82301ab: the exit face of a traced solid evaluated at the recovered complex cosine (up to 4.44 dB at 85°, zero for a real permittivity).
- fd86176: the mass-law angle dependence fixed. The cosine had sat inside the frequency logarithm, so the angle slope was the fitted $m$ (8.5 dB off at 85° for $m = 12$), and the term was cut off between 86.8° and 89.5°.
- c864c28: a characteristic-matrix core for ordered stacks (an ordered pair of sheets had killed the ray; a coated window sat up to 1.92 dB off at 85°). 9d68e22: a sheet-versus-solid suite with an Abelès reference.

2685 → 4531 lines. Kept.

### Phase 8: the sheet/solid integration (2026-10-05 to 10-14)

The sheets of phase 7 had to agree with traced solids in every configuration a scene can build: a sheet glued on a solid, inside a host, between two solids, as a coating, in an ordered stack, with the lumped columns (`att`, `tf`, `alpha`, `m`, the pole), with both tracers, and across a band. A ranked plan was rewritten after every item; the 45 rounds (plan versions 2 to 46) each took one item in one session: analysis, a decision, fixtures written first and failing by a measured amount, then the source change. 32 commits changed the source; their titles state the convention each settled.

The line count rose from 4548 to 5924 (a124ace, 2026-10-12) and fell to 5020 in one commit when the two solvers became one (aecaf04). The themes, in the order they were met:
- The lumped columns in the stack core (56dfb05, 841d0d1): `alpha`, `tf` and the mass law carried in the solver, the pair rule at every boundary; an ordered stack with these columns resolved instead of killed.
- Leak reciprocity and the bonded plate (b484257, b2ce6cd): a wood panel on concrete passed 4.5 dB less into the wall than out of it at 30°, because a face judged from inside a mass-law medium had a critical angle from an index the table does not calibrate. Bonded mass-law bodies became one plate, with a thickness code in the third state word.
- The state words (a196fad): the material index cut to 14 bits (16383 materials) to free two flag bits, since the per-ray state may not grow (production runs carry about 150 million rays). A sheet inside a host is backed by the host.
- The recovery behind sheets and lossy solids (b0f5c05, ba747b2, 8fabe30, 5c56c86): the exact cosine behind a sheet names no medium; a face evaluated at a real angle keeps the solid named; the direction a transition writes.
- The refracting tracer (4d00299, 57c2741): a traced bend is known from the state (the bent flag), and the refracting tracer runs the exact machinery.
- Phase, direction and verdicts (e50aea1, 5eaafc2, ab77976, 5e8f3a4, 6f53915, 62ade69, 7ffa2e8, 0d619a8): the phase of a traversal is the field where the polyline leaves; one verdict per body (summed or followed) across a band; a layered wall is a sum of paths; a lossy second medium has no critical angle.
- Mass-law refinements (dbbf67d, 5c478c4, 1e8590d, 6998b10): the bonded plate at any depth, the mass term at the cosine the wave has in air, a closed mass-law layer, a mass-law sheet behind a held wave.
- Rare sheets and `tf` on wave faces (444931e, dba444d, a124ace, ed65a73, 70dd1b4, 807a1d6): a sheet rarer than the transverse invariant tunnels; `tf` became an ideal transformer where a wave is on both sides of a face and a matched pad where the face is closed. The old quotient $(1 - R')/(1 - R_0)$ had created energy next to lossy media ($T + R$ up to 2.44 on a sheet, up to 50 on a traced porous solid).
- One solver and no clamps (aecaf04, 7441412, 74d6d9b): one recursion for every sheet and stack, no reflection capped at a wave face, no pole clamp; a sheet takes no more host than lies on each side of it; the permittivity pole is part of the medium at every quantity (it had been left out of the propagation index and could amplify, up to $10^{98}$ on a followed series).

The suite grew from 1127718 / 882 (82301ab, 09-08) to 1297934 / 1074 (74d6d9b). Kept as the settled wave side. The mass-law conventions of this phase were reopened on 2026-10-14 (phase 9) and in part replaced in the first acoustic round (phase 10): 4720a92 replaced the mass-law parts of b484257, b2ce6cd, ab77976, 841d0d1, 807a1d6, aecaf04, 6998b10 and 4d00299.

The 32 commits:

| commit | date | lines after | convention settled |
| --- | --- | --- | --- |
| 56dfb05 | 10-05 | 4548 | `alpha` carried in the stack core; fixtures for the lumped columns |
| 841d0d1 | 10-05 | 4764 | `tf` and the mass law in the stack core; the pair rule at every boundary |
| b484257 | 10-06 | 4860 | leak reciprocity at a mass-law face; the leak chain at the wave's own cosine |
| b2ce6cd | 10-06 | 5016 | bonded mass-law bodies as one plate; the reflect port of a mass-law face |
| b0f5c05 | 10-06 | 5031 | a glued exit behind a sheet exits into air; the recovery behind a sheet |
| a196fad | 10-07 | 5101 | a 14-bit material index; a sheet inside a host is backed by the host |
| ba747b2 | 10-07 | 5146 | the recovery behind a lossy solid; the direction a transition writes |
| 8fabe30 | 10-07 | 5147 | a glued exit exits into air, also behind a bare-entered second layer |
| 5c56c86 | 10-07 | 5248 | the recovery in a traced run; the direction a ray leaves a slab in |
| 4d00299 | 10-07 | 5293 | a traced bend in the state; the refracting tracer runs the exact machinery |
| 57c2741 | 10-07 | 5420 | the segment out of a lossy medium follows the direction written |
| e50aea1 | 10-08 | 5457 | the phase of a traversal is the field where the polyline leaves |
| 5eaafc2 | 10-08 | 5442 | a resolved ray leaves at the cosine the wave has; one verdict and one direction for a bare and a coated exit |
| ab77976 | 10-08 | 5457 | a leak keeps its direction through a mass-law body |
| 5e8f3a4 | 10-08 | 5460 | a reflection off a sheet is resolved only where the series was summed; a layered wall is a sum of paths |
| 6f53915 | 10-09 | 5512 | one verdict per body; every frequency takes the base frequency's verdict |
| 62ade69 | 10-09 | 5541 | the reflect port of a sheet lies where its transmit port puts the layer |
| 7ffa2e8 | 10-09 | 5559 | a lossy second medium has no critical angle; the recovery runs until converged |
| 0d619a8 | 10-10 | 5592 | a closed face hands the wall behind it the direction the leak has in air |
| dbbf67d | 10-10 | 5637 | the bonded plate at any depth; a mass-law sheet in a mass-law host is not merged |
| 5c478c4 | 10-10 | 5670 | a mass-law sheet behind a refracting layer forms its term at the cosine in air |
| 1e8590d | 10-10 | 5746 | a mass-law layer the wave is evanescent in passes its two leaks |
| 6998b10 | 10-11 | 5871 | a mass-law sheet behind a held wave is a later face of that wall |
| 444931e | 10-12 | 5874 | an ordered stack leaves the ray on the line it arrived on |
| dba444d | 10-12 | 5917 | a sheet rarer than the invariant with `tf` passes no more than it received |
| a124ace | 10-12 | 5924 | a sheet rarer than the invariant without `tf` tunnels on the characteristic matrix |
| ed65a73 | 10-12 | 5912 | `tf` at the face of a coherent sheet is a transformer and creates nothing |
| 70dd1b4 | 10-13 | 5874 | `tf` at a bare boundary: a transformer where open, a flat leak where closed |
| 807a1d6 | 10-13 | 5894 | `tf` at a closed face is a pad; a scalar closed face reflects at the Fresnel magnitude |
| aecaf04 | 10-13 | 5020 | one solver for sheets and stacks; no clamp at a wave face |
| 7441412 | 10-13 | 5033 | a sheet takes no more host than lies on each side of it |
| 74d6d9b | 10-14 | 5016 | a permittivity resonance is part of the medium it belongs to |

### Phase 9: the fork into wave and acoustic (2026-10-14)

An analysis of a window (a rigid mass-law pane with `tf` and a resonance) showed the acoustic side entangled: a rigid mass-law sheet passed $\text{tf}^2$ times its mass term, so `tf` was both the room-side absorption and a squared transmission factor, and no setting reached both the absorption and the transmission of a pane (3.3 to 42.7 dB above EN 12758 with the measured absorption, against 2.5 dB for a thin-plate model). The coincidence columns needed an `att` baseline, a pole on a rigid mass-law body broke its closed verdict (up to +135.6 dB), and two meshed 4 mm mass-law sheets 16 mm apart transmitted −105 dB at 1 kHz where the measured unit insulates 35 dB, with no mass-air-mass resonance.

Decision (2026-10-14): the engine is built for waves and EM must stay correct; acoustics is an add-on, accurate on reflection and reasonable on transmission. The work was split along the mass column: everything with `m = 0` (EM and scalar types, porous absorbers included) is the wave side, settled and guarded; everything with `m > 0` is acoustic and open to redesign, the conventions of phase 8 included.
- 3110eea: the ray tests split by a marker build that counted, per case, every material built with `m > 0`: `[wave]` 179 cases / 165110 assertions, `[acoustic]` 35 / 31856, the whole suite 1073 / 1298332. The 18 mixed cases were split into a wave half and "… - the mass-law part". The every-digit wave rows are asserted against stored twins (`tests/data/wave_twins.tsv`, relative 1e-12). The largest test file compiled in 21 s instead of 56 s.
- d66e536, daafa6a: the plan replaced by two reference documents, one per side.

### Phase 10: the first acoustic round (2026-10-14 to 10-15)

- bb336dc: reference data per class (glazing from EN 12758 generic values and laboratory data of AGC, brick walls and linings from Riverbank thirds, 152 mm concrete and 190 mm block read from plots, a glass-wool board, absorption compilations, air | water constants); the candidates built as patched copies of the source. No source change.
- First-principles plate models for the bands without a measurement (thin and Mindlin plates, a frame coupling loss, infinite and finite plates, a modal sum): with physical constants the best form missed the measured heavy walls by up to 7 dB in single octaves (concrete −7.1 dB at 125 Hz, +7.3 dB at 4 kHz). The spread between forms at 31.5 and 63 Hz was 4.5 to 13.9 dB for heavy walls and 5.2 to 13.5 dB for light panels. The study stopped there so that no column would be fitted to an unvalidated reference; the decision was to take the model spread as the tolerance of those bands.
- 4720a92 (5016 → 5082 lines, every change behind `m > 0`): the plate rule $T = a\tau/(a + \tau - a\tau)$; nothing reflected back into a mass-law body; `res*` and `coiF`/`coiQ` of a mass-law material moved into the plate's impedance (the pole removed from its permittivity); three columns for a resonant absorption peak (`tfP`, `tfF`, `tfQ`); the two-parameter mapping for every acoustic material; the leak chain fixed (it had multiplied two faces, 12.5 dB non-reciprocal on wood on concrete). `[acoustic]` 32697 / 46, whole suite 1307356 / 1132; 61 acoustic tables moved, each accounted for.
- 04b92bd: the acoustic reference rewritten around the literature review; a light leaf over a cavity built from wave media, as a wave-medium sheet glued on a meshed box of near-air material (no library change, five cases in `tests/catch2_tests/test_acoustic_leaf.cpp`); `tools/acoustic/sabine_to_random.py` and `tools/acoustic/band_probe.cpp`. `[acoustic]` 36705 / 51.

### Phase 11: the resonant face and the knob study (2026-10-16 to 10-17)

- 238444d: the absorption peak on the lumped-resonator law, symmetric in log frequency and zero at both ends (it had been a linear detuning with a floor). A sheet between two bodies (three co-located faces) had left the wave "in air" inside the far body and created energy, $R + T$ up to 42.19 in the scalar types and EM TE and 14.46 in EM TM; such a ray is now killed in every interaction type. 5089 lines.
- 2c5be17: a mass-law face with a peak is a locally reacting series impedance with phase and angle dependence. 5132 lines.
- bc24b0c: `m > 0` switches the mass law on at a fixed 20 dB per decade (it had been a fitted slope, 1.64 to 35.13 in the first round's fits); the coincidence Lorentzian `coiA` removed. Transmission references are fitted on the library's own 0–90° random-incidence average. 5116 lines.

### Phase 12: grazing incidence and the finite panel (2026-10-17 to 10-19)

de60194 (5159 lines): Davy's forced radiation efficiency of a finite panel in the mass term in place of $1/\cos\theta$, with a panel-size column `side` = √(panel area), default 3.29 m. The limiting-angle candidate (78° or 85°) was closed by it. 13 reference constructions within 2.70 dB per octave at their laboratory sizes (2.97 dB with the infinite plate).

### Phase 13: traced-leg precision (2026-10-19 to 10-20)

6307a64 (5211 lines): the relaunch offset of every hit (8·2⁻²³ of the hit point's largest coordinate, about 1 µm per meter) counted in the in-medium distance, so a traced leg is measured from the hit point. Before, a 4 mm pane at 28 GHz carried 4.8° of phase error at 100 m and 48° at 1 km, and its mass term was 0.2 dB too low at 100 m and 2.4 dB at 1 km. In single precision a mass-law leg could be taken for a bent one, which skipped the plate rule (26 dB too little). Rule of use: scenes in local coordinates near the origin.

### Phase 14: cleanup and documentation (2026-10-20 to 10-21)

- 29e1f05: the source restructured with no behavior change: an 84-line header overview, seven sections in one anonymous namespace, comment-only lines 1227 → 768, 5211 → 4725 lines; every printed value bit-identical except two deliberate constants (rounding residues below 1e-9). `tools/acoustic/rowdiff.py` compares two runs number by number.
- a5bdfb7, f0d093d: merges of a bug-fix branch; the intersector reports hits at equal distance in index order on every kernel. 71b894b: the three co-located faces of the sheet between two bodies asserted for every face pair, since the generic and the AVX2 kernel had picked different pairs.
- dc47ff6: the ray tests cleaned. Four redundant cases removed by per-case coverage (1679 lines and 3685 branch keys of the engine still reached); diagnostic tables behind an environment switch; the ray cases 24.9 → 5.9 CPU-s, the whole suite 38 → 19 s. e34dd2a: the switch per area (`QD_DIAG_RAY`).
- 23ad1b6: the material-model page rewritten from the code (538 lines), with worked examples measured through `ray_progress` and 13 preliminary starting rows. a58c18f: the Data Formats material reference (`tools/data_formats_md/format_materials.md`, 307 → 179 lines), the built-in EM table checked against P.2040-3 Table 3.

At the close: `src/ray_mesh_interact.cpp` 4731 lines; `[wave]` 163474 / 178, `[acoustic]` 42741 / 59, whole suite 1344398 / 1264 (generic build).

## 4. The final design in brief

This section names the parts of the engine as they stand at a58c18f and the physics each carries. The formulas as implemented are on the page (sections 4 to 10); the source map is page section 15.

**Two functions and one solver.** `ray_mesh_interact` evaluates one hit without history: the Fresnel coefficients of the face in tilted admittances ($\eta_\text{TE} = z\cos\theta$, $\eta_\text{TM} = z/\cos\theta$, $z = \sqrt{\varepsilon/\mu}$), the reflected or transmitted direction, the beam tube. `ray_state_update` holds the per-ray state and corrects each event with it: the medium the ray is in, the loss in that medium, the series of internal reflections of a traced body, the composite of a sheet. One solver resolves every sheet and every ordered stack of sheets as a recursion over two-ports from the far end, which is the characteristic matrix in another basis; `qd_slab_resolve` returns only the series factor of a traced solid (page 7).

**The energy convention.** Every interface coefficient is a per-channel normal-flux amplitude: the Fresnel amplitude times $\sqrt{\mathrm{Re}(z_2 c_2)/\mathrm{Re}(z_1 c_1)}$ for TE and the scalar types, with the conjugate admittances for TM. For a lossless incidence medium the transmittance is $1 - R$ per channel at any angle, and all four transmission types report the same energy (page 6.1). Total reflection is declared only towards a lossless medium; a lossy medium has no critical angle (page 6.2).

**The mapping.** An acoustic material of impedance $Z$ and sound speed $c$ is entered with $z = Z_\text{air}/Z$ and $n = c_\text{air}/c$ as $\varepsilon = n z$, $\mu = n/z$. Then $\varepsilon$ is the compressibility ratio $\rho_0 c_0^2/(\rho c^2)$ and $\mu$ the density ratio $\rho/\rho_0$, and a fluid interface is exact in level and phase. No code reads the mapping; it is a rule for the columns (page 10.1).

**Wave media and mass-law materials.** The mass column `m` is the boundary. A material without it is a wave medium: EM materials, porous absorbers in the two-parameter mapping, and the leaves of section 4 below; the wave physics is exact for them. A material with `m > 0` is a mass-law material: a rigid body whose permittivity and permeability set only the reflection of its faces and its critical angle, and whose transmission comes from a lumped mass term. Every acoustic mechanism is gated on `m > 0`, so a wave medium never meets one (page 10.2).

**The mass term.** With $X_0 = (f/f_\text{ref})\,M$ the impedance ratio of a limp panel ($f_\text{ref}$ a material constant from the density, $M$ the normal thickness), the term is the limp mass law at 20 dB per decade, with the cosine replaced by Davy's forced radiation efficiency $\sigma$ of a square panel of side `side` [Davy_2009_ForcedRadiationEfficiency]:

$$\text{TL} = A\,\frac{\log_{10}\!\left(1 + (X_0/\sigma)^2\right)}{\log_{10}\!\left(1 + X_0^2\right)} + 10\log_{10}(\sigma\cos\theta), \qquad A = 20\log_{10} X_0,$$

clipped at zero (page 5.4). The plate's own impedance enters $X_0$: the resonance columns as a mass-air-mass dip (`resS > 0`) or a first mode with the stiffness below it (`resS < 0`), and the bending stiffness $|1 - (f/f_c)^2\sin^4\theta\,(1 + j/Q)|$ from `coiF`, `coiQ`.

**The bonded plate.** Mass-law bodies within the co-location distance move with one velocity: their $X_0$ add, and each later body pays the increment $\text{TL}(X_\text{run} + X_0) - \text{TL}(X_\text{run})$. A traced body hands its equivalent thickness to the next one through a logarithmic thickness code in the third state word (0.1 mm to 100 m in steps of 0.044 %, about 0.004 dB).

**The plate rule.** A run of bonded mass-law bodies is one plate with no wave inside. It reflects at the face met what that face does not absorb, $R = 1 - a_\text{met}$, and passes the incoherent series of its faces and its bodies, $T = a\tau/(a + \tau - a\tau)$, with $a$ the smaller absorption of its two outer faces and $\tau$ the bodies' mass term. $T \approx \tau$ where $\tau \ll a$; $T \to a$ for a light panel; $a - T$ is absorbed (page 6.5). Nothing is reflected back into a mass-law body, so a traced plate passes in one path.

**The held wave.** Inside a mass-law body the wave is held undeviated, whatever the permittivity and the tracer. Towards a medium without the mass column it carries the transverse invariant it had in air, so sound behind a wall continues in the direction it arrived in, and a layer behind the wall is met as refraction from air (page 6.5).

**The transmission factor `tf`.** On a face between two wave media it is an ideal transformer: it moves power between the two ports and creates nothing. On a closed face, where a wave is on one side only, it is a matched resistive pad that passes what it absorbs. On a rigid mass-law face it is the room-side absorption, with an optional resonant peak (`tfP`, `tfF`, `tfQ`) on the lumped-resonator law (page 5.3, 6.3).

**The peaked face.** Where a mass-law face carries a peak, the peak is the series impedance of a resonator per $Z_0$, $\zeta = R' + j(1 + R')|Q_\alpha|(f/f_0 - f_0/f)$ with $\text{tfP} = 4R'/(1+R')^2$, over-damped by default and under-damped for `tfQ < 0`. It is met as a locally reacting face against the immittance of the side the wave is on, so it carries phase and angle dependence (page 6.4).

**Leaves on a box.** A light leaf over an unpartitioned cavity (a panel absorber, a double leaf, a membrane, a curtain) is a sheet of a wave medium in the two-parameter mapping, glued on a meshed box of near-air material ($\varepsilon = 1$, $\mu = 1.000001$, since a material identical to air is air), with the wall glued at the back. The resonance, its phase and its angle shift come from the wave physics; no lumped column is involved (page 10.4).

**The state.** Three signed 16-bit words per ray: the current medium, the medium the leg was entered from, and a buffer (a deferred medium or the thickness code). Each has a 14-bit material index and two flag bits; no bit is spare, and the state may not grow (page 8, 15).

## 5. Decisions and their physical reasons

Each entry: what was decided, when, in which commit, and why. Decisions that a later one replaced are in section 6.

**The mass column as the boundary between wave and acoustic physics** (2026-10-14; 3110eea, d66e536). Forty-five rounds of fixes had come from expressing mass physics in a wave engine, and the plan and the tests had grown too large to read. Splitting by a material column, not by interaction type, keeps one rule: EM types on acoustic materials and scalar types on wave media are both possible, and every acoustic change sits behind `m > 0`. The `[wave]` baseline and its stored twins check that no acoustic change moves a wave row.

**EM first** (2026-10-11, restated with the fork). The library's primary use is EM; acoustic handling may be approximate and must never harm EM correctness. In practice: a change in shared code is measured on EM types too, and when a behavior is decided for EM, EM rows are asserted (the sheet between two bodies, 238444d, created energy in EM as well and was fixed for every type).

**The per-ray state must not grow** (2026-10-05). Production runs carry about 150 million rays; every byte per ray counts. When a fix seemed to need history, a stateless gate on a material column or on the direction the tracer holds was looked for first; the 14-bit index (a196fad) freed the last two flag bits; a per-frequency decision became a transient per-event argument (`verdict_inN` / `verdict_outN`) instead of state.

**Less code over bit-identity** (2026-10-12). A wider gate and one rule are preferred over an exclusion kept to protect existing digits; asserted rows may move within reasonable bounds for a physical reason, explained and sized before they move. The one solver (aecaf04, −874 lines) and the transformer and pad of `tf` (70dd1b4, −38 lines) came from this.

**Measured data as the reference** (2026-10-14). The whole chain will be calibrated against measurements later; where theory and measurement differ, the columns have to reach the measurement. Physical models only fill bands without a measurement.

**Octaves from 31.5 Hz, tolerances as guidelines in two tiers** (2026-10-14/15). Transmission loss within about ±3 dB, absorption within about ±0.05, coincidence bands reported apart; a band outside the guideline but within the reproducibility of its reference is reported as such and does not fail. The reproducibility comes from interlaboratory data: ISO 354 $s_R$ of 0.019 to 0.06 at $\alpha_s = 0.1$ [Wittstock_2018_InterlabAbsorptionPart2]; laboratory R ±1 dB from 315 Hz, 2 dB at 200–250 Hz, 3 dB below [Quirt_1982_WindowsI]. Bands without a measurement (31.5 and 63 Hz for most classes) take the spread between physical models as their tolerance: 4.5 to 13.9 dB for heavy walls and 5.2 to 13.5 dB for light panels.

**The reflect side first** (2026-10-15). The goal is the sound field in a room (loudspeaker placement, treatment), so room-side absorption per octave, the resonant peak and the reflection phase come before transmission, whose requirements across rooms are looser. A coherent sum at low frequency cannot match a measurement if the boundary phase is wrong.

**The plate rule** (2026-10-15; 4720a92). A rigid panel had passed $\text{tf}^2$ times its mass term, so its absorption and its transmission could not be set independently (light panels missed by 2.5 to 6.1 dB with `tf` held at the measured absorption). Three rules were measured on 13 constructions with resonance and bending stiffness: $a\tau$ (1.78 dB worst residual), $a\tau/(a + \tau - a\tau)$ (2.51 dB) and a rule on the passivity bound (2.50 dB). $a\tau$ keeps the absorption in the transmission (painting a wall would change its insulation); the bound rule exceeded passivity by 1.6e-6 in 48 rows. The incoherent series is smooth, strictly passive (largest $T + R$ 0.99945), equal to $\tau$ wherever $\tau \ll a$, and is the transmitted / dissipated / reflected split of a panel [Davy_2009_SingleLeafCremer].

**Nothing reflected back into a plate** (2026-10-15; 4720a92). A mass-law body has no physical wave inside (its index is not calibrated), and its internal bounce had compensated the leaks; at a sharp resonance the coherent sum over the bounces exceeded one. With the plate rule the inner faces reflect nothing, and a traced plate passes in one path equal to the summed series.

**Resonance and coincidence in the plate's impedance** (2026-10-15; 4720a92). On a mass-law material the pole on the permittivity made a rigid body lossy, never closed, and able to create energy (sheet and traced solid up to 143 dB apart on 4 mm glass). In $X_0$ the resonance is the plate's: the mass-air-mass dip of a double unit or a lining, or the first mode of a panel. The bending stiffness in $X_0$ recovered the physical coincidence frequencies of the light panels in the fits (4 / 6 mm glass 2825 / 1938 Hz against 2989 / 1993 Hz) and was the decisive column for them (4 mm glass 4.8 → 0.5 dB).

**The two-parameter mapping** (2026-10-15 for every acoustic material; 4720a92; the page teaches only it since 23ad1b6). With $\mu$ available, an acoustic material maps to $\varepsilon = n z$, $\mu = n/z$: air | water is exact in level and critical angle (13.38°); a hard wall reflects with a phase near zero (concrete 0.009° at 10°, 0.104° at 60°); a porous layer on it absorbs as the exact layer (20 mm Delany-Bazley at 500 Hz within 0.0008 in α at 0° to 80°). Rigid bodies then have a critical angle of a few degrees (glass 3.48°, gypsum 9.44°). It is asserted in *Two-parameter mapping - air | water, a hard wall's phase, a porous layer on it and every tracer below its critical angle*.

**A resonant absorption peak with three columns, on the resonator law** (2026-10-15, law 2026-10-16; 4720a92, 238444d). Measured linings and panes absorb with a peak that a power law cannot reach. The alternative (a sheet reflecting $1 - \max(\text{tf}, \tau)$) put window glass at a transmission loss of 6.7 / 8.9 dB at 125 / 250 Hz against 17 / 20 dB in EN 12758. The peak follows the lumped series resonator, $\alpha_0/\alpha_\text{max} = 1/(1 + (Q_\alpha(f/f_0 - f_0/f))^2)$ [Masovic_2021_RoomAcousticsNotes], symmetric in log frequency and zero at both ends; `tfQ` is $Q_\alpha$. The six peaked absorption rows refitted within 0.017 in α, cork board 0.047.

**The peaked face as a locally reacting impedance** (2026-10-16; 2c5be17). The reflection phase of a resonant absorber swings through its resonance (±18° for a panel absorber in a model), and neglecting boundary phase raised level errors by up to about 2.4 dB in phased beam tracing [Jeong_2012_BoundaryConditionsPhasedGA]. The peak became the series impedance $\zeta$; five design questions were measured before the choice:
- $\zeta$ is met against the immittance of the side the wave is on; the cosine in air alone was off by up to 0.68 in $|r|$ behind a lining and created energy behind a porous one.
- $\zeta$ replaces the bare face's reactance; combined with it, the one-parameter mapping brought back its $2\theta$ phase and almost no absorption.
- The power-law part stays a matched pad in front of $\zeta$.
- Gated on `m > 0`: without the gate a peaked wave medium created energy ($R + T$ 1.36).
- Over-damped by default; `tfQ < 0` selects the under-damped root. The two roots have one normal-incidence curve but reflect with opposite sign at $f_0$, and the over-damped face absorbs more toward grazing while the under-damped one absorbs less. The fixture board absorbs 0.288 at normal incidence and 0.420 (over-damped) or 0.214 (under-damped) at random incidence.

**Leaves as wave-medium sheets on a box** (2026-10-15; 04b92bd). For an unpartitioned construction of known build the wave physics gives the leaf's resonance, its phase and its angle shift with no lumped column: a thin layer of a wave medium in the two-parameter mapping has the series impedance $jk_0h\mu = j\omega m''/Z_0$ at any angle, the mass reactance of a limp panel. A partitioned cavity and a face known only from catalog data stay on the peak columns. A meshed air gap is exact where its path series converges but needs 1,126 to more than 8,000 events per ray, so the cavity is a meshed box of near-air material whose series is summed in one event (5 traced events per ray on a half-space, 7 on a finite wall).

**`m` as a switch at 20 dB per decade** (2026-10-17; bc24b0c). The physical mass law rises 20 dB per decade at normal incidence [Davy_2009_SingleLeafCremer], and measured heavy walls rise about 6 dB per octave. A fitted slope (1.64 to 35.13) had been doing the job of what the infinite plate leaves out on heavy walls. With the slope fixed the light panels fit at their own surface mass and coincidence; on heavy walls and linings the job moves to `att` and the level. The fold of bonded bodies lost its exponent and its slope exclusion.

**`coiA` removed** (2026-10-17; bc24b0c). Coincidence on a mass-law plate is the bending stiffness in $X_0$; an angle-free Lorentzian in dB on the interface loss is no physical element, and no material file used it. A table that still carries the column loads and the column is ignored (*Coincidence column - a table that carries coiA loads and the column changes nothing*).

**`att` kept on mass-law materials** (2026-10-17). Removing it alone was measured at 1.51 dB worst residual, but together with the fixed slope the furred lining read 3.33 dB: one non-physical column has to shape heavy walls and linings above coincidence, and `att` is the declared empirical one. Rule of use: one `att` per bonded run.

**Transmission references fitted on the 0–90° average** (2026-10-17). A ray tracer meets a wall at every angle. The waves between 78° and 90° carry 4.3 % of the diffuse weight but pass far more than the average: columns fitted on the 0–78° field-incidence average passed up to 15.3 dB more per octave over 0–90°. Fits run at 192 angles × 6 frequencies per third and are selected and reported at 768 × 12, since a selection on a coarser grid picks columns that alias it (5.71 against 3.33 dB). The fit is `tools/acoustic/txfit.py` on the formula copy `tools/acoustic/txlib.py`, with the references in `tools/acoustic/tl_refs.json`.

**The finite panel and `side`** (2026-10-19; de60194). An infinite plate turns transparent toward grazing (its forced wave radiates $\rho_0 c/\cos\theta$ per unit velocity, which outgrows any mass); below coincidence more than half of what it transmits in a diffuse field comes from the last 12° before grazing (57 % for 4 mm glass at 500 Hz). A finite panel radiates its forced wave with a bounded efficiency. Davy's closed form for a square [Davy_2009_ForcedRadiationEfficiency] follows a windowed finite plate within 0.4 dB diffuse and 0.8 dB per angle; a fixed limiting angle was right in one octave per size only (off by up to 14 dB at 31.5 to 125 Hz), every clamp or soft limit 4 to 34 dB off per angle past 78°. `side` = √(panel area) is within 0.45 dB of the form with true area and perimeter up to an aspect ratio of 3.3 : 1; the default 3.29 m is a 4.0 × 2.7 m wall; 0 is the infinite plate. One `side` per bonded run (a rule of use).

**The relaunch offset in the ledger** (2026-10-19, built 2026-10-20; 6307a64). A decided wave change: each in-medium leg starts with the offset it skipped, so it is measured from the hit point. The relaunch itself stays, since it keeps the next intersection off the same face. A nudge with a normal component was considered and dropped; rays within about 3° of grazing in single precision or with the AVX2 kernel stay a limit.

**Scattering out of scope** (2026-10-15). A per-band scattering coefficient needs new directions or rays per event, a change to the beam tracer beyond acoustics, and a band shares one polyline, so a scattered direction could not differ per frequency within a ray [Savioja_2015_GeometricalModeling]. The material model and the acoustic work stay in `src/ray_mesh_interact.cpp`; a separate project only on strong evidence.

**Rules for absorption references** (2026-10-15). `tf` without a peak is $\alpha(\theta)$ at every angle, so it equals the normal-incidence, random-incidence and field-incidence values at once; it takes a random-incidence reference. Sabine values from reverberation chambers exceed random incidence by 0.1 to 0.35 for porous layers [Dupont_2023_SoundFieldSynthesis] and can exceed one, so they are no `tf`. Measured on three materials with an independent random-incidence reference, Jeong's conversion [Jeong_2013_ReflectionAbsorptionCoefficients] with Thomasson's field impedance of a 3 × 4 m sample [Thomasson_1980_AbsorptionCoefficient] reached an rms of 0.031 (maximum 0.057), the edge-effect correction [Kuhl_1960_EdgeEffect] 0.041, the chamber value capped at one 0.139. Decided: a tiered rule (a hard face as measured; an absorbing product of known construction converted by Jeong's method, `tools/acoustic/sabine_to_random.py`; mineral wool without construction data by the edge correction; anything else marked and kept out of the acceptance). Porous columns are derived from Miki [Miki_1990_ModificationsDelanyBazley], or Komatsu for mineral wool [Komatsu_2008_ImprovementDelanyBazley], over the whole band; the porous reference is a 50 mm glass wool with a measured random-incidence curve [Dupont_2023_SoundFieldSynthesis].

**The membrane and cavity constant** (2026-10-15, revised the same day). Against 10 measured resonances with their construction: the measured $f_0$ first; else the constant 60 for an empty cavity and 50 for a filled one ($f_0 = 60/\sqrt{m''d}$ and $50/\sqrt{m''d}$). A factor 1.34 for framed edges, found first, was mostly the $1/\cos\theta$ angle shift of an unpartitioned cavity under diffuse incidence (section 6).

## 6. Tried and abandoned

Each entry: what it was, why it was tried, why it went, what replaced it, and what it cost where the record says.

**Total reflection disabled for the scalar types** (75cc82c concept, ce0d2c2, 2026-04-15 → d1a117c, 06-16). Rigid bodies in the one-parameter mapping are past their critical angle from about 0.003°, so with total reflection on, sound would not enter them at all. Disabling it made the scalar types differ from EM in a basic verdict. One test for all six types replaced it; the transmission problem it hid moved to the partition and then to `tf`, and was solved only by the mass-law route (`m > 0`, the plate rule, the held wave).

**The scalar transmission partition** (006b85e, 57bfa17, 747f46f, 52a167c, 2026-06-02/03 → cc636a5, 06-03; the last pass-through removed in d1a117c, 06-16). Scalar transmission as pass-through with the isolation in `att`, the mass law and `alpha`. Each gate fixed one case and broke another: pass-through threw away the real surface reflection of a porous absorber; the relative test fired at a rigid wall's exit and added a flat −38 dB; the absolute test still left two code paths that had drifted apart. Dropped after one day.

**The two-knob decoupling** (d79c8d6 specification, 2026-05-28). Conductivity zero for every acoustic material, the real permittivity for reflection only, `att` and `alpha` for absorption and transmission. It held only while transmission was a pass-through; 747f46f already noted that softening the reflection through the permittivity would couple reflection and transmission again, and with `tf` (edbbe24) one column redistributed power between the two ports. The plate rule (4720a92) is what finally set absorption and transmission with different columns.

**`tf` as a redistribution with $R + T = 1$** (edbbe24, 2026-06-05). It fixed the +12 dB of spurious reflection of a thin absorber, but on a rigid sheet it became both the room-side absorption and a squared transmission factor (phase 9). On wave faces it was rebuilt as transformer and pad (2026-10-12/13); on rigid faces it became the absorption alone (4720a92).

**0-based material indices** (f50af9d, 2026-06-01 → c99c5f1, 06-10). The state machine used 1-based indices with 0 = air; the mismatch produced the off-by-one the blind suite found (material $w$ read row $w + 1$, bd26014).

**The fitted mass-law slope** (c1c9076, 2026-05-29 → bc24b0c, 10-17). `m` in dB per decade, fitted per material. In the first acoustic round's fits it took 1.64 to 35.13, a non-physical range: on heavy walls and linings the slope shaped what finite size, coupling loss and modes do. It also made bonded bodies of different slopes unequal (a three-body stack up to 7.66 dB from its rule, 1.32 dB after the fix). The knob study measured that whichever non-physical knob is removed, another takes its job: the slope, a flat `att` with a level far below the mass, or a broad coincidence at 3.7 to 18 kHz with a quality near one.

**The mass-law angle inside the logarithm** (227aab6 → fd86176, 2026-09-08 → de60194). The cosine in the frequency logarithm gave the angle the fitted slope (8.5 dB off at 85° for `m` 12) and cut the term off at 86.8 to 89.5°. The fix put 20 dB per decade of the cosine on the fluid's admittance; the infinite plate's transparency near grazing then led to the finite panel.

**The limiting angle** (proposed 2026-10-17, closed 2026-10-19). Bounding the mass cosine at 78° or 85° [Northwood_1968_PlasterboardWalls] cost six lines and moved 16 assertions; refitted at 0–90° the 13 constructions reached 1.17 dB with a limit at 78° against 1.78 dB without it. The grazing study showed a fixed limit right in one octave per size only; it is a special case of the finite panel ($\cos\theta_l \approx 1.53/\sqrt{kL}$: 78° at 0.9 kHz on a 4.0 × 2.7 m wall, 85° at 5 kHz).

**The coincidence Lorentzian `coiA`** (c1c9076, 4122706 → bc24b0c). An angle-free dip in dB on the interface loss. It needed an `att` baseline to act, it is no physical element, and the bending stiffness in $X_0$ reproduces coincidence from the angle.

**The pole on the permittivity of a rigid body** (4122706 → 4720a92). Intended for resonant absorbers, used for glass and windows. On a mass-law material it made the body lossy at every frequency, opened its closed verdict (+135.6 dB) and let sheet and traced solid differ by up to 143 dB. On wave media the pole stays and is part of the medium at every quantity (74d6d9b).

**The linear detuning of the peak** (4720a92 → 238444d). $\text{tfP}/(1 + (\text{tfQ}(f - f_0)/f_0)^2)$ equals the resonator law near the peak for $\text{tfQ} = 2Q_\alpha$ but is wide below and narrow above, with a floor of $\text{tfP}/(1 + \text{tfQ}^2)$ as $f \to 0$. On the fixture board at 31.5 Hz it read 0.104 against 0.020 on the law.

**The cavity's angle shift on the peak** (proposed 2026-10-15, dropped the same day). The peak of a panel over an unpartitioned cavity moves as $f_0/\cos\theta$ [Toyoda_2011_HoneycombPanelAbsorbers]; a sign on `tfF` would have flagged it. The leaf on a box forms the shift from the wave physics, and a partitioned cavity has none, so the column was not needed.

**The framed-edge factor 1.34** (2026-10-15, revised the same day). Ten measured resonances suggested that framed or clamped leaves resonate 1.34 times above the 60/50 rule. Measured with the leaf on a box of its full depth at random incidence, the framed chamber peaks read 0.80 to 1.11 (geometric mean 0.98): the factor was the cavity's angle shift under diffuse incidence, not edge stiffness. A box made shallower by $1/1.34^2$ would have counted the shift twice (1.05 to 1.49).

**Other candidates measured and dropped** (2026-10-17):
- No `att` on a mass-law material: 1.51 dB alone, 3.33 dB with the fixed slope (section 5).
- The installed loss factor in frequency, $\eta = \eta_\text{int} + C(f/f_c)^{-1/2}$ [Davy_2009_SingleLeafCremer]: no fit moved by more than 0.08 dB, and the fitted $C$ was not Davy's (0 on every light panel, 0.2 to 3.9 on walls).
- The bending loss as a resistance inside the modulus (up to 3 dB at coincidence): neutral on the 13 fits (at most 0.02 dB), since a fitted `coiQ` absorbs it; kept as a pending decision (section 9).

**Two meshed mass-law sheets as a double unit** (tried in the window analysis, 2026-10-14; measured again 2026-10-19). Two 4 mm mass-law sheets 16 mm apart passed −105 dB at 1 kHz where the measured unit insulates about 35 dB, with no mass-air-mass resonance; with a near-air box between them (after the plate rule) 70.1 / 104.3 dB at 1 / 4 kHz, the unbridged double leaf of the transfer matrix (70.2 / 104.7 dB), against 36.4 / 36.5 dB measured for the unit whose spacer bridges the panes. A mass-law sheet carries its term as a power factor, so the box's series cannot resonate on it. A double unit is one lumped sheet with fitted columns, or two wave-medium leaves on a box.

**The meshed air gap behind a leaf** (2026-10-15). Exact where its path series converges, but 1,126 to more than 8,000 events per ray, and `ray_progress` stops at 255 interactions (3.7e-3 in R on the board, 9.1 dB in T on 4|16|4). Replaced by the summed box.

**The thin-sheet grazing clamp** (in the sheet design → 8970c0b, 2026-09-08). Meant to stop a grazing ray from crossing a finite pane, it froze the interface at the onset angle and depended on the meshing (section 3, phase 7). Without it nothing diverges: the transfer function is written in the normal thickness and the complex in-slab cosine, both bounded at every angle.

**Two solvers for one physics** (c864c28 and the sheet's produce mode → aecaf04, 2026-10-13). The characteristic-matrix core and the single-sheet resolve carried one physics in two pieces of code that had to be kept equal; one recursion replaced both and 874 lines.

**Clamps and caps at wave faces** (→ aecaf04, 74d6d9b). A wall lined on both faces had read $T + R$ of 44 with no `tf` under the capped forms; the pole clamp and a loss-cap floor were removed with them. The one clamp left is the pair rule on the reflectance at an open mass-law face.

**The mass-law conventions of the plan rounds** (2026-10-05 to 10-13 → 4720a92). The leaks of a closed plate, the bounce inside a mass-law body and the two ports of a face seen from inside it were settled in phase 8 and reopened by the fork. The plate rule replaced the mass-law parts of eight commits of phase 8 (listed in section 3). Their wave parts stand.

## 7. Failures, bugs and what caught them

Grouped by the instrument that caught them. Sizes are those recorded when each was found.

**Specification-derived tests written without the code.**
- The blind suite of `ray_state_update` (bd26014, 2026-06-11) found material $w$ reading row $w + 1$ with the top index out of bounds, throws inside OpenMP regions that ended the process, unchecked sizes, and a stacked-slab energy defect.
- The `ray_progress` suite (a57d98c, 2026-07-28) found a segmentation fault when every interacting ray was subdivided; the review round before it (f798cfb) found buffers handed on uncompacted, a count that depended on whether a branch ran, and beam buffers returned empty in every beam-mode call.

**An independent analytic reference.** Measuring each candidate mechanism against an Abelès transfer matrix rather than reasoning about it found the four defects of the sheet composite in b57db17 (0.025 → 0.000000 dB). One of them was a correction that applied the per-channel transmittance a second time on a port that already carried it, reproduced to 3e-10. Before that, 2d80e99 had found a total-reflection verdict read from the geometric angle (a glued pair collapsed by up to 26 dB; reciprocity of glued entry and exit 25 dB → 0.07 dB) and a continuation direction never written after a sheet.

**An audit of conventions.** Comparing the four transmission types on one crossing found that they reported different energy (up to 17.56 dB at 85°; 78389b6). Comparing entry and exit cosines found the exit face at the real projection (up to 4.44 dB; 82301ab).

**Oblique and multi-band cases that the older tests lacked.** Tests at normal incidence had passed against broken code: the walk-off inversion and the lost total-reflection phase (39ee6da) appeared only at oblique incidence on an acoustic slab, and the per-frequency Snell angle (0251477) only with an oblique ray into a dispersive medium across a band. Since then the fixtures sweep angles to 85° or 89.9° and run bands through `ray_progress`.

**The energy check $R + T \le 1$ on every new configuration.** It found most of the defects of phase 8: `tf` creating energy next to lossy media ($T + R$ up to 2.44 on a sheet, 50 on a traced porous solid, 3.4 and 1.11 on other lossy solids; ed65a73, 70dd1b4), a porous box on a rigid wall reflecting more than it received with no `tf` anywhere (+5.68 dB for an 8 mm carpet at 60°; 807a1d6), a wall lined on both faces at $T + R = 44$ under the capped forms (aecaf04), a resonant medium amplifying (two ports 1.092 on a 37.5 mm sheet, 110 on 50 mm, $10^{98}$ followed; 74d6d9b), and the sheet between two bodies (42.19; 238444d).

**Reciprocity and twins.** A sheet against its traced solid, a glued entry against the glued exit, a wall from its two sides. Leak reciprocity found the 4.5 dB of the wood panel on concrete (b484257); restating the cases for the plate rule found the leak chain still multiplying two faces, 12.17 to 12.47 dB non-reciprocal (fixed in 4720a92, +8 lines).

**A walker that collects paths only outside the structure.** The sheet between two bodies (2026-10-15) was invisible to the first walkers, which collected a path wherever the state said "air". Inside the far body the state said "air", so energy created there was counted as reflection or ignored. A walker that collects in air only outside the structure's extent exposed $R + T$ of 42.19, and `ray_progress` with a threshold of 1.0 ran away (12.7 GB in 300 s). The test walker of `tests/catch2_tests/test_acoustic_leaf.cpp` follows that rule.

**The second kernel.** The three co-located faces of that geometry are a tie, and the intersector reported a different pair with the AVX2 kernel than with the generic one; 40 of 108 assertions failed in an AVX2 build (71b894b). The case now asserts every pair.

**Printed-row comparison after every change.** Every change was diffed row by row against the stored baseline of the whole suite, and every moved row had to be accounted for (61 acoustic tables in 4720a92, 743 rows in 48 tables in 70dd1b4, 76 cases in 6307a64). Rows that moved by about 3e-16 from recompiling a test file alone were reported apart. A change that moved a wave row stopped the phase.

**Fixture artifacts caught before a decision rested on them.**
- A 152 mm wall missing its plate rule by up to 3.42 dB at 89.9° was the fixture's: the first segment of a helper ended inside the wall and the walker added the leg again (6307a64).
- A reference written for an entry-boundary case was wrong by 4.6 dB; the lossless control, for which the recovery is a provable no-op, exposed it (1021f01).
- Resonance and dip frequencies read from a 14-frequency grid were grid points: "160 Hz at 30°" was 148.6 Hz and "315 Hz at 45°" 298.8 Hz on a 1/96-octave sweep. Peaks are now read from fine sweeps.
- An unbounded transmission fit put a block wall's coincidence at 0.88 Hz, a sub-degree feature a 48-angle fit could not see and the walker could (3 to 15 dB between samplings). Fits are now bounded and selected on a dense grid.
- The first acoustic round's worst residual of 2.51 dB came from an unconverged fit (16-start Nelder-Mead); refitted with differential evolution and selected on a dense grid, the same columns reached 1.23 dB (the knob study of 2026-10-16).

**Claims caught in review.** A helper reported as dead was called through another helper; a cause claimed for a moved row was a test-fixture artifact; the column count of the material table was 26, not 27; the range "valid 1–40 GHz" for the built-in EM table was wrong against P.2040-3 Table 3 (a58c18f). Verifying a factual claim before a decision rests on it became a rule.

**Precision far from the origin.** The 89.9° finding above led to a study of traced legs far from the coordinate origin: every in-medium leg was short by the relaunch offset (6307a64, phase 13), and in single precision a mass-law leg could be taken for a bent one (−26 dB).

**What failure cost.**
- Designs reverted: the partition (one day), the scalar types without total reflection (two months), `tf` as a redistribution (four months before its rebuild), the fitted slope and `coiA` (four and a half months), the mass-law conventions of eight phase-8 commits (one week).
- Rounds: phase 8 took 45 plan rounds, each with fixtures restated (for example 150 rows moved in a124ace, 743 in 70dd1b4).
- Code: `src/ray_mesh_interact.cpp` reached 5924 lines during phase 8 and ended at 4731 after the solver unification and the cleanup.

## 8. Validation evidence

What is asserted in the test suite at the close, with the case that asserts it. Bounds in the fixtures are 2 to 3 times the measured maxima; references are written in the tests from closed forms, never from library code.

### 8.1 EM and wave media

- The per-channel normal-flux transmittance of the four transmission types, both directions, four materials × six angles: exact over 192 rows at a 1e-9 bound (*Diagnostic - Transmission convention baseline, single interface*).
- The thin-sheet composite against the exact analytic slab, five materials × eight angles: 0.000000 dB at a 1e-6 dB bound (*Diagnostic - Thin sheet composite vs analytic slab*); grazing incidence independent of the face to 89.999°, mesh subdivision invariance exact (*Thin sheet - grazing incidence does not depend on the face*).
- A traced air | slab | air chain against the analytic slab, four materials × six angles: 0.000000 dB (*Traced solid - air | slab | air chain against the analytic slab*); the exit port at the recovered angle within 3.8e-15 in amplitude (*Exit boundary - the port at the recovered angle*).
- Ordered stacks against the characteristic matrix (*Layered stack - the characteristic-matrix reference against the single-layer oracles*), and one solver for a sheet and a stack (*One solver - a sheet and a stack are one sum and no mirror is clamped*).
- A sheet reached from inside a medium at the exact complex cosine, both ports, scalar and EM: 9.7e-16 (*Sheet in a medium - the composite at the exact in-medium cosine*).
- Permeability carried wherever the permittivity is: the $\varepsilon \leftrightarrow \mu$ swap exchanges the TE and TM ports; a matched sheet ($\varepsilon = \mu = 4$) reflects nothing at normal incidence.
- A lossy second medium has no critical angle (*Lossy verdict - a lossy second medium has no critical angle*); a resonant medium is its medium at every quantity (*Resonant medium - a permittivity resonance is part of the medium a wave crosses and travels in*).
- A traced slab far from the origin transmits what it does at the origin, to 1 km within 1.2e-5 dB (*Far from the origin - a thin dielectric slab transmits what it does at the origin, summed and followed*).
- The every-digit wave rows asserted against `tests/data/wave_twins.tsv` at a relative 1e-12. From the fork to the close, no acoustic change moved a `[wave]` row; the two wave changes in that time were decided as such (`coiA` removed in bc24b0c, the three cases restated; the relaunch offset in 6307a64, 260 of 736 twin rows rewritten).
- The page's EM worked examples: 200 mm concrete and 6 mm glass of the built-in table at 3.5 and 28 GHz within 0.001 dB of the closed-form P.2040 slab, measured through `ray_progress` (page 9; 23ad1b6).

### 8.2 The acoustic mapping and wave-medium leaves

- Air | water exact in level and critical angle (13.38°; the phase past it 4.33e-4° at 13.38° and 1.53° at 85°); a hard wall's phase 0.009° at 10° and 0.104° at 60° (concrete); a 20 mm porous layer on it within 0.0008 in α at 0° to 80° (*Two-parameter mapping - air | water, a hard wall's phase, a porous layer on it and every tracer below its critical angle*).
- A 4 mm glass sheet as a wave medium on the limp mass law $1/(1 + (\omega m''\cos\theta/2Z_0)^2)$ within 0.002 dB at 0° to 85° at 500 Hz (−31.634 dB at 0°, −10.815 dB at 85°) and within 0.0078 dB from 50 Hz to 1 kHz (*Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf*).
- A 12.5 mm board (8.6 kg/m², loss ratio 0.2) on a 25 mm box of air against the transfer matrix of fluid layers within 6.5e-5 in R summed and 4.7e-6 followed; its peak at 128.4 / 148.6 / 182.4 / 258.8 Hz at 0° / 30° / 45° / 60°, the shift within 0.8 % of $1/\cos\theta$ (*Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle*).
- Double glazing 4|16|4 as two leaves on a box: within 7.0e-3 dB in T; the dip at 211.3 / 244.0 / 298.8 / 422.6 Hz at 0° / 30° / 45° / 60°, against a lumped mass-air-mass frequency of 211.4 Hz (*Leaf - two leaves on a box of air resonate at the mass-air-mass frequency*).
- A porous fill at the back of the box within 2.6e-5 in R of the transfer matrix of its columns (*Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall*); a band through `ray_progress` equal to its single frequencies (*ray_progress - a leaf on a box of air at every frequency of a band*).
- Measured absorbers at random incidence (Paris integral; measured in the analysis, not asserted): a 6 mm MDF membrane over 95 mm peaks at 111 Hz empty and 83 Hz filled against the measured 100 and 80 Hz thirds [Shearer_2016_AbsorptionBelow100Hz]; 3 mm plywood over 30 / 90 / 150 mm at 1.07 / 0.80 / 0.99 times the measured chamber peaks [Sakagami_2007_NonUniformCavity]. The heights depend on the assumed mounting loss.

### 8.3 Mass-law materials

- The mass law: a 4 mm pane with $f_\text{ref}$ from its density on the limp mass law, and `m` 12, 20, 30 bitwise equal (*Mass law - m switches the limp mass law on*); the angle law (*Thin sheet - mass-law angle dependence*).
- The finite panel: Davy's form within 0.4 dB diffuse and 0.8 dB per angle of the windowed finite plate of `tools/acoustic/gzlib.py` for single and double leaves from 1.23 × 1.48 m to 10 × 3 m (*Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency*). Near grazing the panel keeps insulating (4 mm glass, 500 Hz, 89°: a term of 13.2 dB against 1.6 dB for the infinite plate).
- The plate rule: 4 mm glass and 152 mm concrete with their measured absorption, as sheet and traced solid, both tracers, both mappings, on the rule within 0.0022 dB and the reflection within 0.0008 dB (*Plate rule - a rigid mass-law panel passes the incoherent series of its face and its plate and reflects what its face does not absorb*); a bonded wall the same from both sides within 0.0007 dB (*Plate rule - a wall bonded of two mass-law bodies passes the same in both directions*); a traced plate in one path (*Plate rule - a traced mass-law body passes in one path: nothing is reflected back into it*).
- The peaked face against its series impedance in magnitude and phase from 0° to 85°, as sheet and traced solid, on both branches, types 3, 4, 5 and EM type 1 (*Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*); its random-incidence absorption within 4.3e-6 of the closed-form Paris integral (*ray_progress - the random-incidence absorption of a peaked lining is the Paris integral of its series impedance*).
- Every frequency of a band through `ray_progress` equal to its single-frequency run (*ray_progress - the plate rule and the absorption peak at every frequency of a band*).
- A traced mass-law wall far from the origin passes what it passes at the origin, to 5.4e-11 dB up to 1 km (*Far from the origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit point*).

### 8.4 Expressibility against measurements

The acoustic machine is accepted when every reference construction can be reached by some setting of the columns, as sheet and as traced solid, within the guideline in every octave from 31.5 Hz to 8 kHz. This shows that the knobs exist; calibration against measurements is step 3 onward.

- Reference classes: light single panels (glass 4 and 6 mm, gypsum board, plywood), heavy walls (152 mm concrete, brick 95 and 203 mm, 190 mm block), double and triple glazing as one lumped sheet, porous absorbers on a rigid wall, linings on a wall, a layered wall (plaster | brick | plaster), air | water, panel absorbers, curtains, microperforated panels. Transmission references are laboratory values (EN 12758 generic glazing values, AGC laboratory data, Riverbank thirds of brick walls, published concrete and block data); they are collected with their sources in `tools/acoustic/tl_refs.json`.
- Transmission: 13 constructions fitted on the 0–90° random-incidence average at their laboratory panel sizes reach their measured or modeled octaves within 2.70 dB (worst: brick 203 with a furred board; gypsum, a model reference, 2.54 dB); 2.97 dB with the infinite plate (bc24b0c), 2.51 dB in the first round on 0–78° with a fitted slope (superseded). Through the built library: sheet and formula copy within 0.0000 dB, traced solid within 0.0042 dB, $R + T$ below one in every row (de60194). Light panels fit at 0.64 to 1.13 of their surface mass and near their physical coincidence frequency (plywood the furthest, 1563 against 1146 Hz); heavy walls and linings sit at levels of 0.02 to 0.39 of their mass with a flat `att` (section 9).
- Absorption: monotone rows within 0.051 in α with the `tf` power law; the six peaked rows within 0.017 (cork board 0.047) on the resonator law, and on their random-incidence targets as locally reacting faces with the same residuals; the porous reference reached within 0.003 to 0.010 in α at random incidence.
- The page's acoustic worked examples, measured through `ray_progress` (page 11; 23ad1b6): a 152 mm cast concrete wall within 0.58 dB of its laboratory reference (the bare mass law 4.1 to 7.8 dB above it); a 4 mm float glass pane within 0.51 dB of the EN 12758 values; 50 mm glass wool from Komatsu's model at its measured flow resistivity within 0.03 of a sound-field-synthesis measurement from 1 kHz, 0.13 low at 250 and 500 Hz. The 13 fitted rows are published there as preliminary starting rows (page 11.4).

### 8.5 The suite at the close

`[wave]` 163474 assertions in 178 cases; `[acoustic]` 42741 in 59; the whole library suite 1344398 in 1264 (generic build, a58c18f). The ray cases run in 5.9 CPU-seconds single-threaded (dc47ff6). Per-case coverage of `src/ray_mesh_interact.cpp`: 1679 lines and 3685 branch keys reached.

## 9. Limits and open questions at the close of step 2

### 9.1 Limits that matter in typical scenes

- Build scenes in local coordinates near the origin: millimeter-wave scenes through `ray_progress` within about 100 m; never raw georeferenced coordinates, where the relaunch offset exceeds a wall's thickness. Rays within about 3° of grazing can be lost in single precision or with the AVX2 kernel (page 12, 13).
- The state is one level deep. A traced plate re-forms its first face's absorption at its exit from the previous-medium word, which names only the body before; a plate of three or more bodies pays its rule within about ±3 dB, and a wall with an ordered stack inside it is up to 3.01 dB from its traced twin.
- A rigid face's absorption `tf` without a peak is flat in angle and keeps the hard wall's phase; only a peaked face is locally reacting.
- Wave media, the leaves of a meshed double wall included, keep the infinite plate at grazing: the finite panel acts only where `m > 0`.
- A wave-medium leaf has no bending stiffness; coincidence stays with the mass column.
- The followed series of a lossy body between strong reflectors can diverge (round trip above one); only the summed rows, which `ray_progress` takes, mean anything there.
- A plate passes at most what its face absorbs; a rigid face without `tf` absorbs nothing, so its plate passes nothing (rule of use: `tf` together with `m`).
- Heavy walls and linings fit with non-physical columns: a level far below the mass with a flat `att`, or a broad coincidence. Above coincidence these stand for plate physics the infinite and the finite plate do not model; the finite panel does not change it, since the heavy walls are above coincidence from 125 Hz, where grazing carries nothing.
- Out of reach by design: scattering and diffusion, flanking through the building, the ground wave of a spherical reflection, level dependence of microperforated panels, and absorption values that describe a placement in a modal field rather than a material.

### 9.2 Open conventions

- The path the in-body absorption `alpha` is charged over in a lossless layer that tunnels ($\mathrm{Re}\cos\theta = 0$): with any `alpha` such a layer is opaque today. No case carries it, so it moves no row.
- A wall's power-law `tf` behind a porous fill: the pad is formed on the immittance of the medium in front (the board on 25 mm of fill: 0.040 in R off a pad referred to air); physically a wall has a fixed surface impedance. Closed for the peaked face, open for the power-law part; the fixtures use `tf` 0 behind a fill.
- The three co-located faces of a sheet between two bodies: the intersector reports two of three, by index order on every kernel since f0d093d. Energy creation is closed by the kill, but the result depends on which pair is reported (the reflect port of box | sheet | air, R 0.154 for 0.994 at 125 Hz, or the sheet ignored). A full resolve needs the third face from the intersector.
- A nested or overlapping object met on a leg that carries a thickness code replaces the code, and the leg pays its own term in full (+37.50 dB for a solid on a wall with an object nested in it, from one side only). Low priority.

### 9.3 Pending decisions (for the material tables of step 3)

- A check on converted absorption references: a conversion whose fitted flow resistivity is off a measured one by more than about 1.5 times is marked, not used (PU foam 10.3 against 10.6 kPa·s/m² converts; polyester 65.5 against 15.0 does not).
- Two interlaboratory porous rows left out until a polyester model and a framed-sample model exist [Scrosati_2020_InterLaboratoryTest].
- The resonant absorbers as one family (panels and membranes, perforated and slotted panels, microperforated panels) in the acceptance classes.
- The loss of the bending term as a resistance inside the modulus (up to 3 dB at coincidence where $X_0\cos\theta/Q \approx 1$): needed if `coiQ` comes from a measured loss factor, not if it is fitted; the page's mass-law examples would be re-measured.
- A relaxation term on $\mu$ for fabrics, $\mu = \mu_\infty + \Delta\mu/(1 + jf/f_t)$: the power-law columns reach seven measured curtains within 0.008 to 0.039 in random-incidence α but miss single angles by up to 0.35 where $f_t$ lies in the band.

## 10. How the engine was developed

The engine was developed with AI assistance throughout: Claude (Anthropic) wrote the acoustic code, the specifications, the tests and the documents; Stephan Jaeckel, who maintains the library, decided every open point, reviewed every change and made every commit. No acoustic code was written by hand.

### 10.1 Two eras

**January to September 2026: the claude.ai chat.** The work ran mainly in the claude.ai chat. Larger changes started from a specification that Claude wrote from a description of the change, kept in the repository during the work: the design specification of `ray_state_update` and its test specification (566d6fe), the specification of the undeviated-path Snell correction with a staged refactor plan, one gate per stage and a rollback tag (a737c08), the thin-sheet design (260e71b). The "blind" verification suite (bd26014) was written by a separate session that saw only the specifications and the function signature, never the implementation, so its expected values could not copy the code's assumptions; it found the material-index off-by-one in its first run. Commit messages of this era carry the measured sizes of what each change fixed.

**October 2026: agent sessions in the repository.** From 2026-10-05 the work ran in the Claude desktop app on a local checkout, with sessions working directly in the repository.
- The plan era (2026-10-05 to 10-14): one ranked plan of the open sheet/solid issues, rewritten as a self-contained new version after every item; one item per session, in the order analysis from first principles → a decision proposal per question → a decision by Stephan Jaeckel → fixtures written first and failing by a measured amount → the source change → a diff of every printed row against the baseline. 45 plan versions; the transcripts on the machine cover versions 13 to 46.
- The manager and worker era (2026-10-16 to 10-21): a manager session writes a brief per item (why; facts at the start; hard rules; phases), reviews each phase's report and diff against a checklist, re-runs key claims itself, prepares each decision with options and a recommendation, and drafts the commit message. A fresh worker session takes one item in phases with a stop after each and never commits. The manager's state lives in a handoff file, since manager contexts filled up twice (2026-10-16 and 2026-10-21) and chat history is not the record. Documentation tasks used a plan stop and a stop before the commit instead of the fixtures-first phases.

### 10.2 The test discipline

- Fixtures first: a fixture describing a change is written before the source change and fails on the old library by a measured amount; the source change then makes it pass, and nothing else moves.
- Independent references: expected values come from closed forms written in the test (the analytic slab, the characteristic matrix of fluid layers, the limp mass law, the series impedance of a resonator, a column's definition), never from library code. A reference is itself pinned where it can be (a lossless control on which the code is a provable no-op).
- Measure, don't compute by hand (2026-10-07, after an analysis written while commands could not run named a wrong root branch): every statement about what the code does rests on a run. A candidate change is measured before the source is touched, on a what-if build: a patched copy of the source compiled with the library's flags and linked in front of the library, after a plain copy has reproduced the baseline exactly.
- Fixtures drive traced runs: a fixture meant to describe a traced run goes through the intersector, `ray_mesh_interact` and `ray_state_update`, or through `ray_progress`, since synthetic hit codes had differed from what a traced run carries.
- Printed-row comparison: every diagnostic table is printed with its numbers, and after each change every moved row is accounted for in its unit (`tools/acoustic/rowdiff.py` compares two runs number by number; `tools/acoustic/cases.py` reports moved rows per case). Rows that move from recompiling a test file alone (about 3e-16) are reported apart.
- Stored twins: the every-digit wave rows are asserted against `tests/data/wave_twins.tsv`, so the wave side is a regression guard while the acoustic side changes.
- Invariants per change: both ports at most one from a lossless side; a sheet equal to its traced solid; no step in angle; a band equal to its single frequencies; both tracers; summed and followed series.
- Coverage: the four redundant cases removed in dc47ff6 were chosen by per-case line and branch coverage, keeping every line and branch reached before.
- Speed: fixtures that call the library one ray at a time run with one OpenMP thread (20 times faster); the ray cases went from 24.9 to 5.9 CPU-seconds.

### 10.3 Rules that grew from failures

- Check premises and history (2026-10-13): a design had rested on "no code point free" in a state word without checking the decode, and the next session replaced it. Since then a statement that something is unavailable is checked in the code, and a proposal says which commit it replaces and whether it undoes recent work.
- Close issues fully (2026-10-12): prefer the variant that closes a defect in every mode over a gate that leaves it open in one.
- Converge (2026-10-13): a side finding goes into the open item whose code it lives in, not into a new item.
- Usable, not perfect (2026-10-19): no new open items, studies or roadmap entries from side findings, and no work on fringe edge cases; every effect is sized for typical scenes (indoor rooms, site models in local coordinates) before work is proposed. This veto ended the growth of the open list; the open questions of section 9 are what remained.
- Explain in physical terms first (2026-10-14 to 10-21): what a quantity is, what happens, how large it is in a typical scene, before the option is named.
- Commit messages in plain physical terms, with the measured numbers, the line count and the suite counts, and without internal labels or references to transitory documents (2026-10-17).
- Shared machine: at most 12 threads, one suite at a time, timings as single-threaded CPU time with old and new binaries alternated (2026-10-19).

### 10.4 What it caught and what it cost

What the discipline caught is section 7. What the development took, from the record:
- Calendar time: 2026-01-31 (the mapping) to 2026-10-21 (the last documentation commit). Source changes to the engine ran from April to October; the densest stretch was 2026-10-05 to 10-21.
- Commits: 98 commits changed `src/ray_mesh_interact.cpp` from 2026-01-01 to a58c18f; 167 touched the engine, its callers (`calc_diffraction_gain`, `obj_file_read`) or the ray tests.
- Source size: `src/ray_mesh_interact.cpp` grew from 973 lines (ce0d2c2) to 5924 (a124ace) and ended at 4731, after the solver unification (−874 lines) and the restructuring (−486 lines).
- Plan rounds: 45 versions of the sheet/solid plan in ten days.
- Sessions and tokens of the October work, counted from the session transcripts (2026-10-05 to 10-21; one session per row unless noted; "input" is uncached input plus cache writes; the claude.ai chats before October left no counts):

| item | dates | sessions | model calls | output tokens | input tokens | cache reads |
| --- | --- | --- | --- | --- | --- | --- |
| sheet/solid plan rounds (versions 13–46) | 10-05 – 10-14 | 34 | 2600 | 5.85 M | 20.84 M | 0.909 G |
| test split | 10-14 | 1 | 363 | 0.35 M | 1.16 M | 0.171 G |
| wave and acoustic reference documents | 10-14 | 1 | 217 | 0.28 M | 0.73 M | 0.098 G |
| first acoustic round (references, candidates, plate rule) | 10-14 – 10-15 | 1 | 463 | 0.60 M | 1.55 M | 0.194 G |
| acoustic reference rewrite | 10-15 – 10-16 | 1 | 200 | 0.33 M | 2.24 M | 0.095 G |
| leaves on a box | 10-15 | 1 | 161 | 0.23 M | 0.47 M | 0.047 G |
| resonator law; sheet between two bodies | 10-15 – 10-16 | 1 | 191 | 0.23 M | 0.82 M | 0.059 G |
| peaked face as series impedance | 10-16 | 1 | 249 | 0.34 M | 0.68 M | 0.106 G |
| knob study (fixed slope, `coiA`) | 10-16 – 10-17 | 1 | 333 | 0.33 M | 1.94 M | 0.154 G |
| finite panel | 10-17 – 10-19 | 1 | 373 | 0.36 M | 1.83 M | 0.200 G |
| traced-leg precision | 10-19 – 10-20 | 1 | 271 | 0.32 M | 1.36 M | 0.123 G |
| source cleanup | 10-20 | 2 | 244 | 0.43 M | 1.36 M | 0.102 G |
| AVX2 test fix | 10-20 | 1 | 34 | 0.03 M | 0.08 M | 0.003 G |
| test cleanup | 10-20 – 10-21 | 1 | 321 | 0.37 M | 1.53 M | 0.126 G |
| material-model page | 10-21 | 1 | 174 | 0.22 M | 1.11 M | 0.084 G |
| Data Formats reference | 10-21 | 1 | 79 | 0.07 M | 0.20 M | 0.013 G |
| manager sessions | 10-16 – 10-22 | 2 | 481 | 0.47 M | 7.26 M | 0.191 G |
| total | | 52 | 6754 | 10.81 M | 45.16 M | 2.68 G |

The plan rounds took about half of the output tokens of October. These notes themselves (one further session) are not counted.

## 11. Paper material

The paper's angle is the whole pipeline (acoustic simulation with a radio ray tracer, validated by measurement), with the engine as its methods core. This section collects candidate claims with their evidence and limits, negative findings, and figure and table candidates with the test case, tool or probe that regenerates their data. Nothing is plotted here.

### 11.1 Candidate claims

**C1. The two-parameter mapping makes fluid–fluid and porous-on-wall interfaces exact, in level and phase, in an EM engine.** Evidence: air | water exact including the 13.38° critical angle; a hard wall's phase below 0.1° up to 60° (against $2\theta$ in the one-parameter mapping); a 20 mm porous layer on a wall within 0.0008 in α at 0° to 80° (section 8.2). Limits: rigid bodies are rarer than air past a critical angle of a few degrees, so a rigid wall traced as a solid needs the mass column; the mapping is a rule for the columns, and the published material sets come in step 3.

**C2. A light leaf on a meshed air box reproduces panel and double-leaf resonances and their angle shift from wave physics alone.** Evidence: the leaf on the limp mass law within 0.002 dB at 0° to 85°; the board on 25 mm against the transfer matrix within 6.5e-5 in R with its peak shifting as $1/\cos\theta$ within 0.8 %; 4|16|4's dip at 211.3 Hz against 211.4 Hz lumped, shifting with angle; measured absorber peaks at 0.80 to 1.11 times at random incidence (section 8.2). Limits: no bending stiffness in a fluid layer; the infinite plate at grazing; a fill only at the back of the box; heights depend on the assumed mounting loss.

**C3. The plate rule with Davy's finite panel reaches 13 measured constructions within 2.70 dB per octave at random incidence, with one set of columns per construction.** Evidence: section 8.4; the formula copy `tools/acoustic/txlib.py` on the library within 0.0000 dB (sheet) and 0.0042 dB (traced solid). Limits: an expressibility result, not a prediction; heavy walls and linings need non-physical levels and a flat `att`; double units are one lumped sheet; the low octaves take the model spread as tolerance.

**C4. A resonant absorber modeled as a locally reacting series impedance carries the reflection phase and the correct random-incidence absorption.** Evidence: the phase runs from −4.12° to +4.12° through the resonance of the fixture board over-damped and through 180° under-damped; random incidence 0.420 (over-damped) and 0.214 (under-damped) against a normal-incidence 0.288; asserted against the series impedance to 1e-12 (section 8.3). Limits: a flat `tf` keeps the hard wall's phase and is flat in angle; the damping branch is a rule of use; the six refitted rows reach their targets with the residuals of the resonator law (within 0.017, cork 0.047).

**C5. The acoustic add-on leaves the EM results correct, verified by fixtures.** Evidence: analytic slabs to 0.000000 dB, per-channel energy exact at a 1e-9 bound over 192 rows, stored wave twins at a relative 1e-12, no `[wave]` row moved by an acoustic change after the fork (section 8.1). Limits: two wave changes were decided as such after the fork (`coiA`, the relaunch offset).

**C6. The acoustic work improved the EM side.** Evidence in the record: one commit body names an acoustic case as what exposed an EM defect: 39ee6da (2026-08-06), two phase errors "exposed by an oblique acoustic slab case", an inverted walk-off weighting on every transmitted path and a total-reflection phase dropped on the EM and scalar reflection branches. Other EM fixes of the period state their sizes on materials of the acoustic mapping and on EM channels alike, but their bodies do not say what found them. As worded, the claim rests on one commit; a stronger form needs the analysis records of each fix.

**Negative findings** (for the notes; their place in a paper depends on later steps):
- The one-parameter mapping fails at a rigid wall: a phase of $2\theta$, a critical angle near 0.003°, no mass reactance in a thin layer, and up to 0.39 too much absorption for a porous layer on the wall.
- Two meshed mass-law sheets with air between them cannot form a double-leaf resonance; a double unit is one lumped sheet or two wave-medium leaves.
- On heavy walls and linings a non-physical knob always takes over above coincidence, whichever knob is removed; neither the installed loss factor in frequency nor the finite panel changes that.
- Chamber (Sabine) absorption values are not random-incidence values; capped at one they miss an independent random-incidence reference by an rms of 0.139, a model-based conversion by 0.031.
- Physical plate models with physical constants miss measured heavy walls by up to 7 dB in single octaves; the low octaves have no validated reference.

### 11.2 Candidate figures and tables

Diagnostic tables of the ray tests print with the environment variable `QD_DIAG_RAY=1` (`tests/catch2_tests/qd_test_diag.hpp`); the tests assert the same values without it.

| # | content | regenerating source |
| --- | --- | --- |
| F1 | the development curve: line count of `src/ray_mesh_interact.cpp` and suite size over time, with the phases | the git history (`git show <hash>:src/ray_mesh_interact.cpp`), the counts in the commit bodies |
| F2 | reflection phase of a hard wall against angle, one-parameter against two-parameter mapping | *Two-parameter mapping - air \| water, a hard wall's phase, a porous layer on it and every tracer below its critical angle* |
| F3 | air \| water: reflection and transmission against angle across the critical angle | the same case |
| F4 | a wave-medium leaf against the limp mass law, 0° to 85° | *Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf* |
| F5 | a panel absorber on a box: absorption against frequency at 0° / 30° / 45° / 60°, the transfer matrix beside it | *Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle*; a band through `tools/acoustic/band_probe.cpp` with `tools/acoustic/band_probe_example.in` |
| F6 | double glazing as two leaves: the mass-air-mass dip against angle | *Leaf - two leaves on a box of air resonate at the mass-air-mass frequency* |
| F7 | infinite against finite plate: transmission against angle near grazing; the equivalent limiting angle against size and frequency | `tools/acoustic/gzlib.py`; *Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency* |
| F8 | 13 constructions: measured against fitted transmission loss per octave | `tools/acoustic/txfit.py` with `tools/acoustic/tl_refs.json` on `tools/acoustic/txlib.py` |
| T1 | the 13 fitted constructions: columns and residual per octave | the same; page 11.4 |
| F9 | the peaked face: $|r|$ and phase against frequency, both branches; absorption against angle and its random-incidence value | *Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches*; `tools/acoustic/paris_cf.py` |
| F10 | the plate rule: $T$ against $\tau$ for several face absorptions, beside $a\tau$ and the passivity bound | the formula; *Plate rule - a rigid mass-law panel passes the incoherent series of its face and its plate and reflects what its face does not absorb* |
| F11 | chamber values against random incidence: capped, edge-corrected and converted | `tools/acoustic/sabine_to_random.py` (its `check` mode reproduces the validation) |
| F12 | EM: four transmission types on one energy convention; a thin sheet on the analytic slab to 89.999° | *Diagnostic - Transmission convention baseline, single interface*; *Thin sheet - grazing incidence does not depend on the face* |
| F13 | a traced slab far from the origin, before and after the relaunch offset was counted | *Far from the origin - a thin dielectric slab transmits what it does at the origin, summed and followed*; *Far from the origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit point* |
| T2 | validation summary: what is asserted, against what reference, to which bound | section 8 of these notes; the cases named there |
| T3 | the expressibility classes with their reference types and the tolerance against interlaboratory spread | section 5 of these notes; review 3, 5 |
| T4 | decisions and abandoned designs with the reason and the commit | sections 5 and 6 of these notes |

## 12. Glossary

- **Sheet**: a zero-thickness face in the mesh with a thickness column `thk`; its internal series is summed in one event, since the tracer has no world-space thickness to re-emit into. Isolated (air on both sides), glued (on the entry or exit face of a solid), inside a host, or in an ordered stack of two.
- **Traced solid**: a closed body whose thickness the tracer crosses; its internal series is either summed by the series factor or followed as separate paths.
- **Ordered stack**: two sheets in one plane, resolved together by the solver.
- **Solver**: the recursion over two-ports from the far end that resolves every sheet and stack (the characteristic matrix in another basis).
- **Wave medium**: a material without the mass column (`m = 0`): EM materials, porous absorbers, leaves.
- **Mass-law material**: a material with `m > 0`; a rigid body whose transmission is the mass term.
- **Mass term**: the lumped transmission loss of a mass-law body, the limp mass law on $X_0 = (f/f_\text{ref})M$ with Davy's finite panel and the plate's impedance.
- **Plate, plate rule**: a run of bonded mass-law bodies; it reflects $1 - a$ and passes $a\tau/(a + \tau - a\tau)$.
- **Bonded**: within the co-location distance, so moving with one velocity; bonded bodies add their $X_0$.
- **Thickness code**: the equivalent thickness of the bodies of a bonded run, carried in the third state word on a logarithmic scale.
- **Held wave**: the wave inside a mass-law body, held undeviated; it leaves with the transverse invariant it had in air.
- **Leak**: what a closed face passes through its pad.
- **Closed face, open face**: a face with a wave on one side only (past its critical angle, or a rigid face), and a face with a wave on both sides.
- **Transformer**: `tf` on an open face between wave media, an ideal transformer of real ratio; it moves power between the two ports and creates nothing.
- **Pad**: `tf` on a closed face, a matched resistive pad in front of the reactance behind the face; it passes what it absorbs.
- **Pair rule**: the older rule that scaled a face's reflectance by `tf`; it survives only at an open mass-law face.
- **Peaked face**: a mass-law face with `tfP`, `tfF`, `tfQ`, a locally reacting series impedance.
- **Leaf, box**: a light panel built as a wave-medium sheet, and the meshed volume of near-air material behind it that forms the cavity.
- **Leg**: the part of a ray's path between two events inside one medium.
- **Ledger**: the accumulated in-medium distances of a leg (geometric and refracted), from which the loss in the medium is charged.
- **Verdict**: the decision, once per body, whether its internal series is summed (the reflected ray is marked resolved) or followed (each bounce a separate path); every frequency of a band takes the base frequency's verdict.
- **Invariant**: the transverse invariant $\varepsilon\mu(1 - \cos^2\theta)$, conserved across every boundary; a layer whose $\varepsilon\mu$ lies below it carries an evanescent wave and tunnels.
- **Recovery**: the exact complex cosine inside a traced solid, recovered by inverting the map from the incidence angle to the traced direction.
- **VBS**: the correction that re-evaluates an interface at the refracted angle the undeviated tracer does not trace (introduced in 06f9d20; 529d315 names it virtual back-scatter).
- **Tracers**: the undeviated tracer (types 1 and 4, the segment keeps its direction, one geometry for every frequency) and the refracting tracer (types 2 and 5, the segment bends at the base frequency); the bent flag records that a leg follows such a bend.
- **Relaunch offset**: the small distance past the hit point at which a ray is relaunched so that it does not find the same face again.
- **Two-parameter mapping**: $\varepsilon = n z$, $\mu = n/z$ with $z = Z_\text{air}/Z$, $n = c_\text{air}/c$; the one-parameter mapping was $\varepsilon' = (Z_\text{air}/Z)^2$, $\mu = 1$.
- **Expressibility**: the acceptance test of the acoustic machine: every reference construction can be reached by some setting of the columns, within the guideline in every octave.
- **Random incidence, field incidence**: the Paris integral of $\alpha(\theta)$ or $\tau(\theta)$ with weight $\sin 2\theta$ over 0–90°, and the same integral cut near 78°.
- **Stored twins**: the every-digit wave rows stored in `tests/data/wave_twins.tsv` and asserted at a relative 1e-12.
