# Acoustic engine: issues for a later engine stage (2b)

The acoustic interface engine was closed at the end of step 2 (see `md/acoustic_engine_notes.md`). Step 3 builds the
acoustic material table and measures every material and its common combinations through a test rig
(`tests/catch2_tests/test_acoustic_materials.cpp`). Where the engine cannot express a construction, the miss is
recorded here with its size, the independent reference it was measured against, the materials it affects, and an
interim modeling guideline. Nothing here is worked on during step 3; a later engine stage (2b) takes this list as its
input, and the rig's cases become its regression tests.

Each issue: what happens physically; how it was measured; the size; what it affects in typical rooms; the interim
guideline; how to reproduce it.

## Issues

### 1. A porous layer on a rigid wall adds too much insulation at low frequency

- **Construction.** 50 mm glass wool (the columns of page 11.3: Komatsu's model at a flow resistivity of
  13112 Pa·s/m²) as a sheet glued on 152 mm cast concrete (the preliminary row of page 11.4), sound from the wool side.
- **Physics.** Pressed against a heavy wall, the air in a thin porous layer hardly moves at low frequency, so its flow
  resistance has almost nothing to act on: the layer adds next to nothing to the wall's insulation below a few
  hundred hertz and rises toward its own loss at high frequency. In free air the same layer is a resistive screen
  (3.5 dB at 31.5 Hz).
- **Measured** (2026-10-23, normal incidence, `calc_diffraction_gain` with one straight path, scalar mode,
  `thin_slab_threshold` 0): the insulation the wool adds to the wall, in dB.

  | octave (Hz) | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | library | 8.04 | 6.71 | 5.38 | 4.09 | 3.42 | 4.78 | 7.32 | 9.32 | 13.02 |
  | transfer matrix: the same fluid layer on a heavy limp plate (347 kg/m²) | −0.02 | −0.07 | −0.18 | −0.28 | 0.16 | 2.53 | 5.98 | 8.79 | 13.24 |
  | wool alone in air, library and transfer matrix (equal) | 3.52 | 3.85 | 4.20 | 4.56 | 4.92 | 5.44 | 6.85 | 9.33 | 13.00 |

  The reference reproduces the wool alone exactly, so the layer's mapping is the same in both. The wall's fitted knobs
  cancel in the difference (with and without the layer).
- **Size.** Library minus reference: 8.06 / 6.78 / 5.56 / 4.37 / 3.26 / 2.25 / 1.34 / 0.53 / −0.22 dB from 31.5 Hz
  to 8 kHz, so up to 8.1 dB too much insulation at 31.5 Hz and 5.6 dB at 125 Hz, within 0.6 dB from 4 kHz. Room-side reflection of the same construction is right (within 6.3e-3 of the transfer matrix).
  Random incidence and other layers and walls are measured by the rig's combination cases.
- **Affects.** Transmission through any wall, floor or ceiling carrying porous treatment: absorbers on walls, carpets
  on floors, bass traps against walls. It matters for planning treatment across rooms; it does not move the sound
  field inside the treated room.
- **Likely cause (not proven).** The wave-medium layer is crossed as if air were behind it, and the mass-law wall's
  plate rule follows as a separate loss, so the layer's free-air loss is paid on top of the wall's.
- **Interim guideline.** Do not count porous treatment on a wall toward the insulation between rooms below about
  1 kHz; where it matters, fit the treated wall as one row (as page 11.4 does for lined walls), which loses the
  layer's angle-dependent absorption on the room side.
- **Through the rig** (2026-10-23, `Materials rig - a porous layer on a rigid wall: reflection and the added
  insulation`, 342.76271 m/s, one tone per third, 50 angles), the insulation the wool adds, in dB:

  | incidence | quantity | 31.5 | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0° | library | 7.78 | 6.54 | 5.53 | 4.21 | 3.49 | 4.59 | 7.06 | 9.08 | 12.54 |
  | 0° | fluid layer on a limp plate of 346.7 kg/m² | −0.02 | −0.07 | −0.17 | −0.26 | 0.11 | 2.19 | 5.59 | 8.41 | 12.61 |
  | random | library | 3.09 | 3.13 | 2.81 | 2.98 | 4.27 | 6.74 | 9.87 | 14.78 | 21.55 |
  | random | fluid layer on an infinite limp plate | 1.27 | 2.80 | 4.23 | 5.84 | 7.95 | 10.86 | 14.24 | 18.27 | 25.02 |

  At random incidence the library adds more than the model below 125 Hz and less above. That row is not yet a clean
  measure: the library's wall radiates as a finite panel (Davy's σ at `side` 3.52 m), the model's plate is infinite,
  so the wall's radiation does not cancel in the difference at oblique incidence.
- **Reproduce.** `QD_DIAG_RAY=1 ./build_linux/test_bin "Materials rig - a porous layer on a rigid wall*"` prints the
  rows above.

### 2. A band run near grazing can take the wrong bounce decision for a leaf on a box of air

- **Construction.** A 12.5 mm board (8.6 kg/m²) glued on a 25 mm box of air on a mass-law wall (page 10.4), one ray
  carrying a band of frequencies through `ray_progress` at the default `thin_slab_threshold` 0.15.
- **Physics.** Every frequency of a band takes the sum-or-follow decision made at the band's first frequency. Near
  grazing the first frequency follows the box's series, and higher frequencies inherit a decision that is wrong for
  them.
- **Measured** (2026-10-23): at 89° and 1816 Hz the band reflects 1.073 (energy created) where the same frequency
  alone and the transfer matrix give 0.955; with 125 Hz as the first frequency the reflection is off by up to 0.164
  (1.6 kHz, 89.5°); a traced porous layer through `calc_diffraction_gain` at 0.15 is off by 1.0 dB in transmission.
  At threshold 0 every band equals its single frequencies.
- **Size.** 8.3e-4 in octave absorption at random incidence (only angles past 85° are hit); the filled box costs 12
  times more at 0.15.
- **Interim guideline.** Run acoustic scenes with `thin_slab_threshold` 0.

### 3. Two traced wave media back to back on one straight path

- **Construction.** Two wave-medium layers (glass wool | board) both traced as solids, glued face to face, through
  `calc_diffraction_gain` with one straight path.
- **Measured** (2026-10-23): transmission 4.9 dB off the transfer matrix at 16 Hz and above one at 88° (+3.1 dB);
  through `ray_progress` the exchange between the two bodies is followed as paths and depends on the gain floor
  (0.106 in reflection at −120 dB, 0.027 at −250 dB). Two bonded traced mass-law bodies deviate from the bonded plate
  by up to 2.0 dB at 16 kHz (0.05 dB at 1 kHz); the sheet | solid form is exact.
- **Interim guideline.** Build stacks with the room-side layer as a sheet and the layers behind it as traced solids.

## Modeling guidelines collected in step 3

Rules for scene models that follow from measured behavior; they go to the material page (section 12) at the end of
step 3 where they are not there already.

- **Treatment is a sheet on a rigid solid.** The room-side layer of a stack (carpet, porous absorber, the leaf of a
  panel absorber) is a sheet; the layers behind it are traced solids. Not: two traced wave media back to back
  (issue 3), two mass-law sheets on one face (page 12, rule 9), a thin lossless wave layer as a traced solid (it
  reflects totally past its critical angle instead of tunneling).
- **The order of co-located sheets is the order of their faces in the mesh:** the first face is the room side.
  Reversing it turns wool | board into board | wool (reflection up to 0.999 different). Check the face order after
  an export.
- **An air gap behind a panel is a box of cavity air** (`e` 1.000001) glued exactly between the panel and the wall,
  never a free air space between two objects: the box's bounce series is summed in one event, a free gap is followed
  bounce by bounce and runs out of interactions (page 12, rules 9 and 16).
- **`thin_slab_threshold` 0** for acoustic runs (issue 2).
- **Frequencies in the library's unit:** acoustic frequencies map to the library's frequency unit with the sound
  speed 342.76271 m/s (`src/qd_atmospheric_absorption.hpp`; the wrappers convert with it). Material columns in GHz
  are made with the same constant.
