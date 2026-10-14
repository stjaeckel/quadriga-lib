# Acoustic round 1 — report

Round 1 of the acoustic add-on (`md/acoustic.md`, decisions P1–P5, items N1–N7, A10, A11). One section per phase.
Scratch rig: `build_linux/scratch/a1` (`README.txt` lists every file). No code in `src` or `tests` was touched.

## 0. First step

`cmake --build build_linux --parallel` was current except for the test binary, which was relinked. The baselines were
regenerated from the repository root:

| baseline | expected | measured |
| --- | --- | --- |
| `test_wave.txt` `"[wave]"` | 165110 assertions in 179 test cases | **165110 in 179**, all passed |
| `test_acoustic.txt` `"[acoustic]"` | 31856 assertions in 35 test cases | **31856 in 35**, all passed |
| `test.txt` (whole suite) | 1298332 assertions in 1073 test cases | **1306515 in 1121**, all passed |

The whole suite has 48 more cases than the brief says. They are the five test files of the merge `5272033`
(conf-file parser): `test_scenpar_validate.cpp` 7, `test_conf_file_list.cpp` 9, `test_conf_file_read.cpp` 15,
`test_conf_files_v19.cpp` 5, `test_conf_file_write.cpp` 12 (`grep -c TEST_CASE`, 48 in total). `md/wave.md` section 0
still quotes the old count. The new `test.txt` is the baseline every later run of this round is diffed against.

## 1. Phase 1 — research and materials

### 1.1 How the references were gathered

Every number in this section comes from a source named beside it, or from a script in the rig fed with sourced
constants (`mat.py`, `low.py`, `model.py`, outputs `mat.out`, `low.out`). Three kinds of reading:

- **Tables as text**: datasheets extracted with `pdftotext` (AGC, Pilkington, Knauf, Owens Corning, akustik.ua).
- **Tables as images**: BIA Technical Notes 5A, Table 1, read by eye from the scanned table and upscaled crops
  (`ref/bia_tn5a_riverbank.tsv`, the crops are in `src_docs/`).
- **Plots**: the Warnock 1995 Fig. 1 block wall and the PCA RD066 concrete wall 76-66, read by eye from the curves at
  each third-octave grid line, **about ±1 dB** (`ref/warnock1995_fig1.tsv`, `ref/pca_rd066_7666.tsv`).

Octave values from third octaves are energy means of the three thirds (`model.octave_from_thirds`). The air constants
are the mapping's (`acoustic_to_p2040.py`: ρ₀ 1.2 kg/m³, c₀ 343 m/s, Z_air 411.6 rayl, FMAP 8.7403e-4 GHz per Hz).

**Model of the references** (`model.py`; used to extend measurements, never to replace them): an infinite thin plate
with mass, bending stiffness and loss factor η at the trace wavenumber (the i47 `panel.py` form), optionally a first
mode (a stiffness term at f₁₁ of a simply supported a×b plate) and a mounting resistance r_m·ωm′; double units and
linings as exact transfer matrices in (p, v_n); porous layers by Delany–Bazley or Miki; field incidence 0–78°, octave
means over 12 log-spaced frequencies. η is **fitted** to the measured octaves where a measurement exists, and said so.

### 1.2 Class 1 — light single panels

**Transmission loss, octaves (dB)**

| construction | 125 | 250 | 500 | 1k | 2k | 4k | Rw (C;Ctr) | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| float glass 4 mm | 17 | 20 | 26 | 32 | 33 | 26 | 29 (−2;−3) | Pilkington, EN 12758 generic values |
| float glass 6 mm | 18 | 23 | 30 | 35 | 27 | 32 | 31 (−2;−3) | Pilkington, EN 12758 generic values |
| float glass 6 mm | 19.5 | 23.4 | 29.5 | 35.5 | 27.6 | 31.6 | 31 (−2;−3) | AGC, lab, 1.23 × 1.48 m, EN ISO 10140 |
| gypsum board 12.5 mm | — | — | — | — | — | — | — | no measured single-board spectrum found |
| plywood / solid door | — | — | — | — | — | — | — | no measured spectrum found |

AGC has no 4 mm row; the 4 mm reference is the EN 12758 generic one. For gypsum board the only published figures found
are single numbers ("single panels of equivalent mass (e.g. gypsum board) have a rating of 28 dB", Ballagh, NZ
Acoustics 2003/2011) and the coincidence of 13 mm board near 2.5 kHz (ScienceDirect topic overview; 2562 Hz in a JCAA
paper). For wood the only lead was a Chilean study with Rw 17 dB for plywood and an unstated thickness: not usable.
Both are therefore **model references** with sourced constants (below), not measurements.

**Constants**

| panel | m′ (kg/m²) | f_c (Hz) | η | f₁₁ (Hz), size | source |
| --- | --- | --- | --- | --- | --- |
| glass 4 mm | 10.0 | 2989 (12500/t: 3125) | 0.015 fitted | 11.0, 1.23 × 1.48 m | ρ 2500, E 70 GPa, ν 0.22 (i47 NOTES); 12500/t Saint-Gobain ACO-3B; size AGC |
| glass 6 mm | 15.0 | 1993 (2083) | 0.015 fitted | 16.5, 1.23 × 1.48 m | as above |
| gypsum 12.5 mm | 8.6 | 2500 | 0.006 | 10.2, 1.2 × 2.4 m | m′ Knauf Bauplatte GKB 12.5 product page; f_c above; η Irvine, "Damping properties of materials" |
| plywood 18 mm | 10.8 | 1146 | 0.013 | 22.3, 1.2 × 2.4 m | ρ 600, E 5.4 GPa, η 0.013 Irvine; **ν not found**, 0.3 used (enters f_c by 4.6 %) |

No loss factor of float glass was found (as in i47); Irvine's table gives roughly 0.0006–0.002 for glass (scan unclear),
far below what the plate model needs (0.01–0.03): the fitted η carries the mounting and edge losses of a lab pane.

**Model against measurement** (plate model, `mat.out`): 4 mm 14.5 / 20.2 / 26.0 / 31.6 / 35.0 / 25.6 against EN
17 / 20 / 26 / 32 / 33 / 26; 6 mm 17.8 / 23.6 / 29.4 / 34.3 / 26.1 / 32.3 against EN 18 / 23 / 30 / 35 / 27 / 32 — within
2.5 dB in every octave (the 125 Hz band of 4 mm is the largest residual, −2.5 dB).

**Room-side absorption, octaves**

| surface | 125 | 250 | 500 | 1k | 2k | 4k | 8k | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glass, ordinary window | 0.35 | 0.25 | 0.18 | 0.12 | 0.07 | 0.04 | — | Everest & Pohlmann, Master Handbook of Acoustics, App. C (search excerpt; same row on MTSU page, unattributed) |
| 4 mm glass | 0.30 | 0.20 | 0.10 | 0.07 | 0.05 | 0.02 | — | akustik.ua compilation |
| glass, large panes, heavy | 0.18 | 0.06 | 0.04 | 0.03 | 0.02 | 0.02 | — | Master Handbook, App. C (excerpt) |
| 6 mm glass | 0.10 | 0.06 | 0.04 | 0.03 | 0.02 | 0.02 | — | akustik.ua |
| single pane 3 mm | 0.08 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 | pyroomacoustics `materials.json` |
| plasterboard on 50 mm battens | 0.29 | 0.10 | 0.05 | 0.04 | 0.07 | 0.09 | — | akustik.ua |
| plasterboard on frame, 100 mm air | 0.30 | 0.12 | 0.08 | 0.06 | 0.06 | 0.05 | — | akustik.ua |
| 12 mm plywood over 50 mm air gap | 0.25 | 0.05 | 0.04 | 0.03 | 0.03 | 0.02 | — | akustik.ua |
| solid wooden door | 0.14 | 0.10 | 0.06 | 0.08 | 0.10 | 0.10 | 0.10 | pyroomacoustics |

The sources disagree by a factor of two to four at 125 Hz for the same product (window glass 0.08 to 0.35). The
compilations name no measurement per row. The plate model's own absorption (what an infinite pane passes plus what it
dissipates) is 0.037 at 125 Hz for 4 mm and 0.018 for 6 mm: the measured low-frequency absorption of glass is not the
infinite pane, it is its mounting and its modes (section 1.10).

### 1.3 Class 2 — heavy walls

**Transmission loss, octaves from third octaves (dB)**

| wall | m′ kg/m² | 125 | 250 | 500 | 1k | 2k | 4k | rating | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cast concrete 152 mm, plain | 347 | 41.9 | 46.5 | 52.6 | 56.3 | 62.0 | 64.9 | STC 57 | PCA RD066 (Litvin & Belliston, J. ACI 1978), test 76-66, read from plot |
| face brick 95 mm | 189 | 31.6 | 33.5 | 39.3 | 46.7 | 54.7 | 60.7 | STC 45 | BIA TN 5A, Riverbank TL67-70 |
| solid face brick 203 mm | 407 | 37.3 | 40.1 | 47.5 | 52.6 | 59.8 | 65.0 | STC 52 | BIA TN 5A, TL67-68 |
| solid brick 305 mm | 570 | — | — | — | — | — | — | STC 59 | BIA TN 5A, TL67-32 (thirds in `ref/`) |
| concrete block 190 mm | (225) | 35.1 | 41.3 | 44.6 | 50.3 | 56.8 | 63.4 | STC 50 | Warnock 1995, Fig. 1, read from plot; m′ from the paper's Table 1, normal weight, **assumed** for this wall |

The block wall is the only heavy wall measured below 125 Hz: thirds 63 / 80 / 100 Hz = **36.0 / 33.8 / 35.5 dB**. No
octave table of solid concrete was found as text; RD066 gives the plots and STC 57 / 58 for 152 / 203 mm.
Concrete constants: fresh unit weight 2323 kg/m³ and 38.7 MPa at 28 days (RD066), E_cm 33 GPa (EN 1992-1-1 Table 3.1,
C30/37) and ν 0.2 (EN 1992-1-1 3.1.3) → f_c 111 Hz. For brick no E was found; f_c is taken at the measured dip
(160 Hz for 95 mm, 100 Hz for 203 mm).

**Neither physical model describes these walls** (`low.out`): the plate model and the Sharp / INSUL single-panel form
(below f_c/2: 20 log(m′f) − 48; above f_c: 20 log(πm′f/ρ₀c) + 10 log(2ηf/πf_c); INSUL tech note, search excerpt) with a
fitted η are 4 to 10 dB low at 125 Hz and 4 to 11 dB high at 4 kHz. The measured walls rise by about **6 dB per
octave from 125 Hz to 4 kHz** (concrete 41.9 → 64.9 dB over five octaves, 4.6 dB per octave; brick 203 mm 5.5 dB per
octave), which is the mass-law slope, not the 9 dB per octave of an infinite plate above f_c.

**Absorption**: smooth unpainted concrete 0.01 0.01 0.02 0.02 0.02 0.05; rough concrete 0.02 0.03 0.03 0.03 0.04 0.07;
smooth brickwork with flush pointing 0.02 0.03 0.03 0.04 0.05 0.07; porous concrete blocks 0.05 0.05 0.05 0.08 0.14 0.20
(akustik.ua; the pyroomacoustics rows agree within 0.01 where both list a material).

### 1.4 Class 3 — double and triple glazing

| unit | 125 | 250 | 500 | 1k | 2k | 4k | Rw (C;Ctr) | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4-(6–16)-4 | 21 | 17 | 25 | 35 | 37 | 31 | 29 (−1;−4) | Pilkington, EN 12758 generic |
| 4-16-4 | 20.5 | 16.8 | 25.7 | 36.4 | 41.4 | 36.5 | 30 (−1;−4) | AGC, lab |
| 6-15-6 | 21.5 | 21.4 | 31.0 | 38.7 | 30.8 | 39.2 | 32 (−1;−3) | AGC, lab |
| 4-12-4-12-4 (triple) | 18.7 | 19.0 | 28.4 | 41.6 | 46.7 | 39.8 | 33 (−2;−6) | AGC, lab |

Mass-air-mass resonance (Saint-Gobain ACO-3B form, reproduced by i47 to 1 Hz): 4|16|4 **211.4 Hz**, 4|12|4 244.1 Hz.
The infinite transfer-matrix model is 4–12 dB below the measurement from 250 Hz to 1 kHz (as in i47); by P1 the
measurement is the reference. Absorption: double glazing 2–3 mm, > 30 mm gap 0.15 0.05 0.03 0.03 0.02 0.02 0.02;
10 mm gap 0.10 0.07 0.05 0.03 0.02 0.02 0.02 (pyroomacoustics); 0.15 0.05 0.03 0.03 0.02 0.02 (akustik.ua). No
measured absorption of a 4-16-4 unit was found; none of these rows shows a peak at the mass-air-mass resonance.

### 1.5 Class 4 — porous absorber on a rigid wall

| absorber | 125 | 250 | 500 | 1k | 2k | 4k | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glass wool 20 mm, 11 kPa·s/m² | 0.11 | 0.26 | 0.52 | 0.70 | 0.84 | 0.95 | Knauf Insulation TP 120 A, datasheet 04/2018, α_s (mounting not stated) |
| glass wool 30 mm, 11 kPa·s/m² | 0.11 | 0.28 | 0.61 | 0.86 | 0.97 | 1.04 | same |
| OC 706, 25 mm, Type A | 0.01 | 0.22 | 0.67 | 0.97 | 1.05 | 1.06 | Owens Corning 706/707 data sheet 2025, ASTM C423 |
| OC 706, 50 mm, Type A | 0.19 | 0.78 | 1.06 | 1.13 | 1.06 | 1.12 | same |

The Knauf board is the reference: it is the only one with a published airflow resistivity (11 kPa·s/m², EN 29053).
Owens Corning gives none (forum figures for 703 range 16–27 kPa·s/m²; not used). Values above one are reverberation-room
edge effects; the library cannot reach them and the expressibility test will read them as one.

A Delany–Bazley layer on a rigid wall at 11 kPa·s/m² (field 0–78°) gives 0.10 / 0.27 / 0.49 / 0.70 / 0.85 for 20 mm
from 250 Hz up, about one octave below the measured α_s; Miki gives the same within 0.04 and stays positive below
Delany–Bazley's range (92 Hz at this resistivity), where Delany–Bazley goes negative (−0.007 at 31.5 Hz). The
Delany–Bazley real part of the wavenumber (0.0978 X^−0.700) and the Miki coefficients are as commonly quoted, not
confirmed against the papers; the repository tool uses only Delany–Bazley's attenuation term.

### 1.6 Class 5 — a lining on a rigid wall

| construction | m′ kg/m² | 125 | 250 | 500 | 1k | 2k | 4k | STC | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| solid brick 203 mm, bare | 407 | 37.3 | 40.1 | 47.5 | 52.6 | 59.8 | 65.0 | 52 | BIA TN 5A TL67-68 |
| same + 12.7 mm gypsum board on 1×3 wood furring, one face | 423 | 37.0 | 38.7 | 55.9 | 62.5 | 68.4 | 70.4 | 53 | TL69-287 |
| SCR brick 140 mm, bare | 272 | 36.8 | 40.4 | 46.7 | 51.7 | 57.0 | 62.7 | 51 | TL69-286 |
| same + 9.5 mm gypsum board on 25 mm polystyrene dabs, one face | 282 | 37.6 | 35.9 | 46.6 | 61.5 | 66.5 | 71.5 | 49 | TL70-39 |

A lining lowers the transmission loss around its mass-spring resonance and raises it above: −4.4 dB at 250 Hz and
+9.8 dB at 1 kHz for the bonded board (the third at 250 Hz: 35 against 40 dB), −1.4 / +8.4 dB at 250 Hz / 500 Hz for
the furred board. Lining of TL69-287 by weight difference 16.6 kg/m² (board and furring); a 3/4 in furring cavity
gives a leaf-over-cavity resonance of 106.5 Hz with that mass, 147.9 Hz with the 8.6 kg/m² board alone
(f₀ = (1/2π)·√(ρ₀c²/(m′d)), which is the familiar 60/√(m′d)).

**Absorption** of linings: plasterboard on 25 mm battens 0.31 0.33 0.14 0.10 0.10 0.12 (akustik.ua; the compilation lists
the same six values for "12 mm wood panelling on 25 mm battens"); wooden lining, 12 mm on frame 0.27 0.23 0.22 0.15
0.10 0.07 0.06 (pyroomacoustics); thin carpet cemented to concrete 0.02 0.04 0.08 0.20 0.35 0.40 (pyroomacoustics).

### 1.7 Class 6 — plaster | brick | plaster

| wall | m′ kg/m² | 125 | 250 | 500 | 1k | 2k | 4k | STC | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face brick 95 mm, bare | 189 | 31.6 | 33.5 | 39.3 | 46.7 | 54.7 | 60.7 | 45 | TL67-70 |
| brick 105 mm + 12.7 mm sanded plaster **one face** | 207 | 35.6 | 39.7 | 45.7 | 51.4 | 55.9 | 60.2 | 50 | TL69-283 |

**No measured spectrum of a wall plastered on both faces was found.** Single numbers only: 115 mm brickwork plastered
both sides 47, one side 43; 230 mm plastered both sides 55 (Engineering ToolBox, uncited, rating type not stated).
One face of plaster adds 18.1 kg/m² and 4.0 to 6.4 dB between 125 Hz and 1 kHz (1.2 / −0.5 dB at 2 / 4 kHz), more than its mass (0.8 dB): the
plaster seals the brick (the effect Warnock 1995 describes for porous blocks). Absorption: lime cement plaster on
masonry 0.02 0.02 0.03 0.04 0.05 0.05; plaster on solid backing 0.03 0.03 0.02 0.03 0.04 0.05 (akustik.ua).

### 1.8 Class 7 — air | water (N7, the wave side)

**Constants** (`water/consts.py`): water at 20 °C, c = 1482.36 m/s (Bilaniuk & Wong 1993, 148-point equation, as coded
in the NPL technical guide "Speed of sound in pure water"), ρ = 998.21 kg/m³ (Tanaka et al., Metrologia 38 (2001) 301,
coefficients as reproduced by chempy and a LATU paper), Z = 1.4797e6 rayl. With the mapping's air: z = Z_air/Z =
2.7816e−4, n = c_air/c = 0.231388.

| mapping | ε (`a`) | μ (`e`) | critical angle |
| --- | --- | --- | --- |
| one parameter, ε = z², μ = 1 | 7.7376e−8 | 1 | 0.0159° |
| two parameters, ε = n·z, μ = n/z | 6.4364e−5 | 831.84 | 13.379° (asin(c_air/c)) |

**Measured through the library** (`water/wprobe.cpp`: one face, `ray_mesh_interact` types 3, 4, 5, double precision,
13 angles from 0° to 85°), against an independent fluid–fluid reference (`water/wref.py`, `wphase.py`):

- **Two-parameter mapping: exact.** Reflectance, transmittance (normal flux) and refracted direction agree with the
  reference to rounding at every angle (reflectance ≤ 1.7e−13 dB, transmittance ≤ 3.8e−12 dB, direction to 1e−4°;
  22.1272° at 5°, 87.9395° at 13.37°); the critical angle falls between 13.37° (τ = 4.11e−5, open) and 13.38° (closed,
  total-reflection bit set, type 35); the reflection phase is exact past it (4.33e−4° at 13.38°, 1.53° at 85°).
  Transmittance at normal incidence 1.112e−3 (−29.54 dB) in both mappings.
- **One-parameter mapping**: the same at normal incidence, then closed from 0.016°: from 5° on it passes nothing where
  water passes 1.03e−3 at 5° and 7.46e−4 at 10°, and its reflection phase is 2θ (10° at 5°, 170° at 85°) where water's
  is 0° below 13.38° and 1.53° at 85°.

The room-side absorption of a water surface (what it transmits): 1.11e−3 at normal incidence, 4.19e−5 field-averaged
over 0–78° (4.02e−5 over 0–90°).

**What the two-parameter mapping does to a rigid body** (columns in `mat.out`): with n = c₀/c_L, the critical angle
becomes asin(n): glass 3.48°, concrete 4.91°, brick 4.59–6.12°, block 7.16°, gypsum 9.44°, plywood 6.26° (one-parameter:
0.0017° to 0.0164°). A mass-law layer is closed only at or above that invariant (2.3), so under the two-parameter
mapping every rigid sheet is **open below a few degrees** and takes the open-face rules there. Not yet measured;
phase 3.

### 1.9 The low octaves (A11)

Measured transmission data start at 100 Hz (thirds) or 125 Hz (octaves), absorption data at 125 Hz. The single
exception is the 190 mm block wall at 63 and 80 Hz (36.0, 33.8 dB). Model references (`mat.out`, `low.out`):

| octave | glass 4 mm | glass 6 mm | gypsum 12.5 | plywood 18 | concrete 152 | brick 203 |
| --- | --- | --- | --- | --- | --- | --- |
| 31.5 Hz, plate (infinite) | 5.1 | 7.4 | 4.3 | 5.5 | 32.5 | 33.8 |
| 31.5 Hz, plate with first mode | 4.3 | 5.2 | 3.8 | 2.3 | 31.5 | 37.3 |
| 31.5 Hz, Sharp / INSUL | 1.6 | 5.1 | 0.3 | 2.3 | 32.4 | 33.8 |
| 63 Hz, plate (infinite) | 9.3 | 12.3 | 8.3 | 9.8 | 37.1 | 37.7 |
| 63 Hz, plate with first mode | 9.0 | 11.6 | 8.1 | 8.7 | 19.7 | 16.3 |
| 63 Hz, Sharp / INSUL | 7.6 | 11.2 | 6.3 | 8.3 | 35.4 | 32.1 |

- **Light panels**: the first mode of a lab-size pane lies below the 31.5 Hz octave (11.0 / 16.5 Hz for 4 / 6 mm at
  1.23 × 1.48 m; 10–22 Hz for boards of 1.2 × 2.4 m), so the low octaves are mass-controlled. The plate model sits on
  EN 12758 within 2.5 dB above 125 Hz; its low-octave values are the reference, with a spread of 1 to 4 dB between the
  model forms. A smaller pane moves the mode into range (4 mm, 0.6 × 1.2 m: 34 Hz, i47): the mode is a property of the
  pane's size, which no column carries.
- **Heavy walls**: the lab walls' first modes fall into the 31.5 and 63 Hz octaves (32–52 Hz at 4.37 × 2.84 m). A single
  mode puts a dip of 15 to 20 dB into the 63 Hz octave (19.7 dB for concrete), where the only measured heavy wall shows
  a dip of 2 dB at 80 Hz. Both models are 4 to 10 dB off at 125 Hz already. **For heavy walls there is no trustworthy
  reference below 125 Hz** apart from the one block wall. The measured walls continue the 125 Hz value roughly flat
  to 63 Hz (block: 35.1 dB octave at 125 Hz, 36.0 / 33.8 dB thirds at 63 / 80 Hz).
- **Absorption** below 125 Hz: no measurement found for any class. The plate model gives what an infinite pane absorbs
  (passes plus dissipates): 0.37 / 0.13 at 31.5 / 63 Hz for 4 mm glass, 0.31 / 0.07 for 6 mm, 0.42 / 0.16 for gypsum board;
  0.004–0.11 for the heavy walls.

The ray picture itself is weak there: wavelengths of 5.4 m (63 Hz) and 10.9 m (31.5 Hz).

### 1.10 A10 — where room-side absorption peaks

| construction | peak | where | height above its floor | width | source |
| --- | --- | --- | --- | --- | --- |
| plasterboard on 25 mm battens | 0.33 | 250 Hz octave (0.31 at 125) | about 0.2 | about 2 octaves at half height | akustik.ua |
| plasterboard on 50 / 100 mm, plywood over 50 / 150 mm air | 0.25–0.30 | 125 Hz octave or below (falls from 125) | about 0.2 | not resolved (peak at or below the lowest band) | akustik.ua |
| wooden lining 12 mm on frame | 0.27 | 125 Hz or below, falling slowly to 500 Hz | about 0.2 | ≥ 2 octaves | pyroomacoustics |
| cork board 25 mm on battens | 0.65 | 500 Hz | 0.3–0.5 | about 1.5 octaves | akustik.ua |
| ordinary window glass | 0.35 | 125 Hz or below | 0.3 | not resolved | Master Handbook (excerpt) |
| double glazing (2–3 mm) | 0.10–0.15 | 125 Hz or below; none at the mass-air-mass resonance | 0.1 | not resolved | pyroomacoustics, akustik.ua |
| heavy walls, plaster, porous absorbers | none | flat or rising | — | — | 1.3, 1.5, 1.7 |

**Physical reading.** Every peaked rigid face of the reference classes is a **panel that moves**: a leaf over a cavity
(the lining class) or a pane on its mounting. Its peak sits at the leaf-over-cavity resonance f₀ = (1/2π)√(ρ₀c²/(m′d)):
129 / 91 / 65 Hz for 8.6 kg/m² board on 25 / 50 / 100 mm, which matches the measured rows (peak in the 250 Hz octave
on 25 mm battens, at or below 125 Hz on deeper cavities). The model (`low.out`, board | air | rigid wall) reproduces it
only with a **mounting resistance** r_m (a fitted 0.2·ωm′): peak 0.278 at 150 Hz, **1.08 octaves** at half height on
25 mm; with only the board's own η (0.006) it absorbs nothing. So the peak's height is set by the mounting loss and its
frequency by the leaf mass and the cavity, neither of which the rigid wall behind it knows.

What the library would have to express (for phase 2/3): a peak of 0.2–0.35 above a floor of 0.05–0.1, about one to two
octaves wide, between 65 and 250 Hz; a pane's falling absorption from 125 Hz is the upper flank of a peak below the
band range. The double unit's mass-air-mass resonance does not show in its absorption. Candidate (a) of A10 (a sheet
reflects 1 − max(tf, τ)) puts the peak where the plate *passes*, which for a leaf over a cavity is not where the wall
behind it passes anything: the leaf's τ in air is not its absorption on a wall. This is a first reading, measured in
phase 3.

### 1.11 The materials and how every column is obtained

Columns under today's machine (one-parameter mapping), from `mat.out`. Rules of derivation:

- **`a`** = (Z_air / (ρ·c_L))², with c_L the bulk longitudinal speed from E, ν, ρ where those are sourced (glass,
  concrete), else the plate speed from f_c (c_L = c₀²/(1.8·f_c·h), AAS 2015 as used in i47): gypsum, brick, block.
  Two-parameter alternative: ε = n·z, μ = n/z with n = c₀/c_L (columns `a`, `e`).
- **`m` = 20**, **`fRef`** = FMAP·Z_air/(π·ρ) (X₀ = ωm′/(2ρ₀c), the limp-panel impedance ratio), **`thk`** the thickness.
  These are fixed by the mass law; a fitted `fRef` is a phase 3 knob.
- **`tf`, `tfB`**: today R = 1 − tf, so tf is the absorption, a power law fitted log-log to the octaves 125–4000 Hz
  (`tf` holds its value at `fRef`; the library clamps tf at one per frequency).
- **`coiF`** = f_c·FMAP. `coiQ`, `coiA` come from the dip depth and width in phase 3 (today they need an `att` baseline).
- **`resF`**: a double unit's mass-air-mass resonance (M3), a lining's leaf-over-cavity resonance, or a panel's first
  mode (the A11 single-panel form). `resQ` is fitted in phase 3; `resS` is a switch (M3).
- **`att`** = 0 unless phase 3 needs it.

| material | `a` | `fRef` GHz | `thk` m | `tf` at fRef / `tfB` (tf at 1 kHz) | `coiF` GHz | `resF` GHz (what) | two-parameter `a` / `e` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glass 4 mm | 8.479e−10 | 4.5805e−5 | 0.004 | 60.20 / −0.621 (0.132, window glass) | 2.6128 | 9.61e−3 (first mode) | 1.767e−6 / 2083 |
| glass 6 mm | 8.479e−10 | 4.5805e−5 | 0.006 | 15.48 / −0.601 (0.042, heavy plate glass) | 1.7419 | 1.44e−2 (first mode) | 1.767e−6 / 2083 |
| gypsum 12.5 mm | 8.182e−8 | 1.6644e−4 | 0.0125 | 1.053 / −0.295 (0.085, on battens) | 2.1851 | 8.93e−3 (first mode) | 4.691e−5 / 573 |
| plywood 18 mm | 4.758e−8 | 1.9085e−4 | 0.018 | 0.152 / −0.058 (0.094, wooden door) | 1.0015 | 1.95e−2 (first mode) | 2.379e−5 / 500 |
| concrete 152 mm | 2.026e−9 | 5.0212e−5 | 0.152 | 3.14e−4 / +0.417 (0.019) | 0.09696 | 4.12e−2 (first mode, lab) | 3.850e−6 / 1901 |
| brick 95 mm | 2.341e−9 | 5.7726e−5 | 0.0953 | 1.49e−3 / +0.333 (0.037) | 0.1398 | 2.83e−2 | 3.869e−6 / 1653 |
| brick 203 mm | 4.088e−9 | 5.7213e−5 | 0.2032 | 1.49e−3 / +0.333 (0.037) | 0.08740 | 4.53e−2 | 6.818e−6 / 1668 |
| block 190 mm | 1.595e−8 | 9.6699e−5 | 0.19 | 1.57e−3 / +0.432 (0.081) | 0.1093 | 3.63e−2 | 1.574e−5 / 987 |
| 4-16-4, one sheet | 8.479e−10 | 4.5805e−5 | 0.008 (20 kg/m²) | 5.985 / −0.508 (0.040) | 2.6128 (one pane) | 0.18479 (mass-air-mass, M3) | 1.767e−6 / 2083 |
| water (half-space, m = 0) | 7.738e−8 | — | — | — | — | — | 6.436e−5 / 831.8 |
| glass wool 11 kPa·s/m² (m = 0) | wave medium from Delany–Bazley or Miki; two-parameter ε, μ per octave in `mat.out` (complex, μ″ from −62.5 at 31.5 Hz to −0.32 at 8 kHz) | | 0.02 / 0.03 | | | | |

Linings are the board's row with `resF` = f₀·FMAP (129 Hz on 25 mm → 0.1128 GHz) glued to the wall's material.
Plaster | brick | plaster is the brick row with two plaster sheets (12.7 mm sanded plaster, 18.1 kg/m² per face by
weight difference; no further plaster constants found).

**First things the derivation shows** (to be measured in phase 3, not conclusions):
- With today's rule a rigid wall pays tf² of leaks: concrete tf 0.0185 at 1 kHz is 34.7 dB before the mass term, the
  i47 expectation (34–40 dB). With tf from the window-glass absorption the 4 mm pane's tf reaches the clamp (1.000) in
  the 31.5 Hz octave.
- The measured heavy walls rise by about 6 dB per octave (1.3), which a single `m = 20` slope can follow; the infinite
  plate cannot.
- The porous reference has a complex μ under the two-parameter mapping; the library carries magnetic loss in the
  `g`, `h` columns as a power law.

### 1.12 Not found

- A measured third-octave or octave spectrum of a single 12.5 mm gypsum board, of a plywood or particleboard panel,
  and of a solid timber door (searches in English and German; a Dutch door report link served HTML).
- A brick wall plastered on both faces, measured per band.
- A text table of solid concrete per band (RD066 plots were read instead).
- A loss factor of float glass; Poisson's ratio of plywood; Young's modulus of brick.
- Any absorption measurement below 125 Hz; any heavy-wall transmission below 100 Hz except the 190 mm block.
- A source for each row of the absorption compilations (akustik.ua, pyroomacoustics), and the original Delany–Bazley and
  Miki papers (coefficients as commonly quoted).

### 1.13 Sources

- AGC Interpane, "Pocket Stratophone 08/23 DE": https://www.agc-yourglass.com/sites/default/files/2023-09/Pocket_Stratophone_0823_DE.pdf
- Pilkington Optiphon brochure and datasheet (EN 12758 generic values): https://fastglassprocessing.co.uk/wp-content/uploads/2025/04/Pilkington_Optiphon_brochure_and_datasheet.pdf
- Brick Industry Association, Technical Notes 5A (Riverbank Acoustical Laboratories data): https://www.gobrick.com/media/file/5a-sound-insulation---clay-masonry-walls.pdf
- Litvin & Belliston, PCA RD066: https://www.vibrationdata.com/tutorials/RD066.pdf
- Warnock, 7th Canadian Masonry Symposium 1995: https://canadamasonrydesigncentre.com/wp-content/uploads/7th_symposium/F-39.pdf
- Knauf Insulation TP 120 A datasheet: https://www.bobbie.de/media/uattachments/3/knau2405261/knauknauf_insulation_akustik-daemmplatte_tp_120_a.pdf
- Owens Corning 706/707 data sheet: https://www.owenscorning.com/dms/10021863
- akustik.ua absorption coefficients: https://www.acoustic.ua/st/web_absorption_data_eng.pdf
- pyroomacoustics materials database: https://raw.githubusercontent.com/LCAV/pyroomacoustics/master/pyroomacoustics/data/materials.json
- Master Handbook of Acoustics, App. C (excerpt only): https://www.accessengineeringlibrary.com/content/book/9781260473599/back-matter/appendix3 ; MTSU page: https://w1.mtsu.edu/faculty/wroberts/teaching/reverb_2.php
- Knauf Bauplatte GKB 12.5: https://knauf.com/de-DE/p/produkt/bauplatte-gkb-12-5-10003_0010 ; British Gypsum Gyproc WallBoard PDS (7.7 kg/m² minimum): https://www.british-gypsum.com/documents/product-data-sheet-pds/british-gypsum-pds-gyproc-wallboard-12-5mm.pdf
- Ballagh, NZ Acoustics 2003: https://www.acoustics.org.nz/sites/www.acoustics.org.nz/files/journal/pdfs/Ballagh,_K_NZA2003.pdf
- Irvine, "Damping properties of materials" Rev D: http://vibrationdata.com/tutorials_alt/damping.pdf
- EN 1992-1-1 values via eurocodeapplied: https://cdn.eurocodeapplied.com/design/en1992/concrete-design-properties ; BSI slides: https://www.bsigroup.com/LocalFiles/EN-HK/events/7-stephen-hicks-eurocode-requirements-for-concrete-design.pdf
- NPL, speed of sound in pure water: https://resource.npl.co.uk/acoustics/techguides/soundpurewater/
- Tanaka et al. 2001 coefficients (chempy): https://pythonhosted.org/chempy/_modules/chempy/properties/water_density_tanaka_2001.html ; https://www.redalyc.org/journal/6061/606164861006/606164861006.pdf
- INSUL single-panel formulas (search excerpt; page did not resolve here): https://www.insul.s05.system7.co.nz/tech-info/
- Engineering ToolBox, massive walls: https://www.engineeringtoolbox.com/sound-transmission-massive-walls-d_1409.html

## 2. Phase 2 — the candidates as real code

### 2.1 Rig

`build_linux/scratch/a1/wi`. Every candidate is a patched copy of `src/ray_mesh_interact.cpp` written by `mkc.py`
(`rmi_<X>.cpp`), compiled and linked in front of `libquadriga.a` (`build.sh`, from i47). The plain copy `rmi_id.cpp`
reproduces `test.txt` exactly (the seed line only), and every candidate is diffed against the plain copy's own run.
Instruments: `caselist.cpp` (a Catch2 listener that prints each case name, so a moved row is attributed to its case,
`attr.py`), `fails.py` / `expand.py` (failed assertions per case with their values), `walker.cpp` (the i47 P4 walker
ported to the split test files, linked in place of `test_acoustic_state_update.cpp.o`), `m1.spec` / `chk_m1.py`
(walls whose transmission the M1 rules define exactly), `mk_mark.py` (marker copy, N4 and N5). Every suite run from the
repository root, one at a time; every walker under a timeout and one thread.

### 2.2 Marker runs

**N4 — does any case carry `coi*` together with `m`?** No. The coincidence term is active in exactly one case,
*Acoustic - interface_gain penetration and coincidence* (m = 0, att > 0, coiA of both signs, once a negative total),
and in no case with m > 0 (marker `N4` in `interface_gain`, whole suite).

**N5 — can a traced solid's entry face know its thickness?** At the entry event `ray_state_update` holds the second
hit of the segment (`sbsN`, the hit count, the second face's material), and `ray_mesh_interact` the distance to it
(`thicknessN`). Marker `N5` at every traced entry from air, whole suite: **4,204 entries into a mass-law solid, every one
with a second hit on a parallel face** (no entry without one). In 3,038 the second face carries the entered material
(the body's own exit face: the thickness is |F − S| · cos θ for a held wave); in 1,166 another material (a bonded body
behind it, a held lining, a sheet inside it), where the distance is still the body's own thickness but what lies beyond
is not known. Entries into media without the mass column: 931,271, of them 652 without a second hit (rays that end
inside the body). So: **yes, from the second hit, wherever the segment reaches past the body**, which every traced
event of the suite and of `ray_progress` does; a ray that ends inside the body has no second hit.

(A progress message during this phase quoted 5,104 / 3,938 for N5 from a hand sum; the script's totals above are the
measurement.)

### 2.3 The candidates

Line counts against today's 5016 lines. Suite: whole-suite rows that move relative to the plain copy (attributed to
cases), failing cases and assertions of `[acoustic]`. **`[wave]` under every candidate: 165110 assertions in 179 cases
passed, no row moved (the seed line only), the stored twins asserted at 1e-12.**

| candidate | what it is | lines | replaces (commit) | `[acoustic]` failed cases / assertions | rows moved |
| --- | --- | --- | --- | --- | --- |
| **R2** | M1 core + one leak: the plate passes T = a·τ | 5041 (+25) | see 2.4 | 10 / 299 | 5,250 |
| **R1h** | M1 core + incoherent series: T = a·τ / (a + τ − a·τ) | 5041 (+25) | as R2 | 15 / 2,159 | 19,083 |
| **R1m** | M1 core + minimum: T = min(a, τ) | 5041 (+25) | as R2 | 15 / 2,104 | 18,694 |
| **M2** alone | nothing is reflected back into a mass-law body (i47 B4k without B4) | 5033 (+17) | Stokes reflect branch `b2ce6cd`; the followed bounce `97ae3cc` | 2 / 53 | 1,165 |
| **B4** | the pole leaves the permittivity of a mass-law body; X₀·\|1 − (f·c/F)² + j·f·c/(F·Q)\| | 5000 (−16) | pole in `eta()` `74d6d9b`; mass block `b2ce6cd` | 0 / 0 | 9 (rounding, i47) |
| **B4m** | B4, the factor capped at one (the dip only) | 5001 (−15) | as B4 | 0 / 0 | 9 |
| **B4s** | B4m, and `resS < 0` selects the single-panel form X₀·\|1 − (F/f)² − j·F/(f·Q)\| uncapped (A11) | 5006 (−10) | as B4 | 0 / 0 | 9 |
| **M4a** | X₀·\|1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ)\| on a mass-law body (bending stiffness) | 4998 (−18) | mass block `b2ce6cd` | 0 / 0 | 9 |
| **M4b** | a negative coincidence total acts on a mass-law body (no `att` baseline) | 5016 (±0) | the clamp in `interface_gain` `235260f` | 0 / 0 | 0 |

The 9 rows of B4/B4m/B4s/M4a are the rounding rows i47 found (the mass block written as one line on `mass_X0` /
`mass_TL`: 7 residuals of *Thin sheet - mass-law angle dependence* and 2 printed zeros of *Lossy verdict … the mass-law
part*). M4b moves nothing because of N4: no case carries `coi*` with `m`. B4/B4m reproduce i47's suite result; their
window behaviour and the A11 form are measured in phase 3.

### 2.4 M1 as built: one plate rule, reciprocal

Building M1 showed that the rule as stated in `md/acoustic.md` 5 (the leak is the absorption, the transmission comes
from the plate) is not reciprocal when the plate's two faces absorb differently, and needs a definition of "the
absorption of a face" that holds next to a lossy medium. The rule that came out of the measurements:

> A mass-law body (or a run of bonded mass-law bodies) is one plate. Its first outer face passes what it absorbs, a_in,
> exactly as today; its bodies pay their terms τ (the plate rule, unchanged); every face inside the plate and its last
> outer face pass what reaches them, nothing is reflected back into it (this is M2 for these bodies), and the last face
> puts the whole right so that **the plate passes F(min(a_in, a_out), τ)**: a reciprocal number that is never more than
> either face absorbs. R2: F = a·τ; R1h: a·τ/(a + τ − a·τ); R1m: min(a, τ). A face's absorption is what it passes over
> what a face that passed everything would pass, from the same side (the pad's own measure; 1 − |r|² is not that next to
> a lossy medium).

Where it sits in the code:

- `Material::interact_with`: the two held branches (towards the lower index: the other direction's crossing and the
  Stokes reflection, `b484257`, `b2ce6cd`) and the geometry fix-up of the held exit (`ab77976`) become **one branch**: a
  wave held in a mass-law body passes every face it meets (`sqrt(interface_gain)` of the medium entered, the ordinary
  verdict, the geometry of no bend), and reflects nothing; not into a wave medium that is closed for it (that face stays
  the pad). And no tracer bends into a mass-law body (A7): entering one, the geometry handed back is that of no bend,
  and `ray_state_update` writes no bent flag into it (`4d00299`).
- `qd_stack_resolve`: every face between two mass-law layers is transparent; the open faces of every other layer are
  the transformer (**the pair rule at mass-law faces is gone, A6**: `841d0d1`; `apply_tf_pair`'s clamp is no longer
  reachable with a value outside [0, 1]); the closed pad's way out becomes the pad that passes everything
  (`closed_face(…, pass_all)`, `807a1d6`); the leak chain's second crossing is a held face (its scalar phase fix of
  `aecaf04` goes); the held chain drops its direction gate (**A4**, `6998b10`) and puts the plate right at its last face.
  The rule enters at the plate's last face (`PLATE_FIX`), with a_in from the run's first face.
- `ray_state_update`: the leg on which a held plate is left (`plate_out`, set by `APPLY_TRANSITION` and `SHEET_EVENT`)
  pays the rule, with a_in re-formed from the state (`PLATE_IN`: the previous medium's face; for a later bonded body the
  face of the body before it from air; behind a sheet inside the current body that body's own face) and a_out from
  the medium it leaves into (`PLATE_ABS`). `SHEET_RESOLVE` hands the solver a_in for the held chain.

**Checked against its own definition** (`m1.spec`: 4 mm glass and 152 mm concrete as sheet and as traced solid; wood
20 mm (tf 0.2) bonded to concrete 100 mm (tf 0.1) as two solids, as a sheet on a solid and as a stack of two sheets,
**both directions**; types 4 and 5; 0°, 10°, 30°, 60°, 80°; 500 Hz; walker with every series summed):

| rule | max \|T − rule\| | max \|R − (1 − a_in)\| | max T + R | the two directions of wood \| concrete |
| --- | --- | --- | --- | --- |
| R2 | 0.055 dB (concrete at exactly 0°, where the open face reflects 1 − 4z less than one); 0.002 dB elsewhere | 0.0026 dB | 0.986 | equal (−44.26 dB at 10°) |
| R1h | 0.002 dB | 0.0026 dB | 0.986 | equal |
| R1m | 0.002 dB | 0.0026 dB | 0.986 | equal |

Today's machine on the same walls: T = a_in·τ·a_out for a single plate (the two leaks), and 0.88 to 5.07 dB below even
that for the bonded pair (the flat leak of the inner face).

Four things had to be found and fixed on the way, each measured: (1) the one-sided form (exit face transparent, the plate
pays from its entry face only) made a wood | concrete wall 3.0 dB different in its two directions; (2) at exactly 0° a
rigid layer is open and kept today's two transformer faces: 4 mm glass read 6.9 dB below the rule at 0° while 10° sat
on it, and tracer 5 entered the layer bent; (3) a face's absorption written as 1 − |r|² made a lossy coherent layer on a rigid plate read
1.1 to 11.6 dB different from its two sides (*Closed face* F); (4) the held branch at first reported total reflection
on every transmission, which `SHEET_RESOLVE` reads as a closed outer pair: a coherent lining on the far face of a wall
then passed nothing (*Bonded mass*, the m = 0 lining on exit, 60 dB).

### 2.5 What M1 leaves open (measured, in every rule)

- **The state is one level deep.** The plate's entry face is re-formed from the previous-medium word. Four traced
  bodies K | T W | K (concrete, a third body, wood, concrete) read 3.01 dB from the same wall as a stack of sheets
  (−13.01 against −9.99 dB), and K | W T | K 1.76 dB: at the last leg the word names W, whose face absorbs 0.2 where the
  plate was entered through K's 0.1. Exact for one body, two bonded bodies, sheets on and inside a body. No bit is free
  to carry more (`md/wave.md` 6).
- **A non-rigid mass-law layer (one with a wave inside) next to a lossy coherent layer** (laboratory constructions:
  `eps 2.25` or `eps 1.50 − 0.40j` with m 20 beside `eps 2.50 − 0.80j`): the stack's two orders differ by 0.107 / 0.127 /
  0.201 dB at 10° / 30° / 60° (*Closed plate* E, the same for the lossless and the lossy layer), and by 0.016–0.024 dB
  for `eps 1.50 − 0.40j` beside `eps 5.24 − 0.63j` (*Stack with a mass-law layer*); exact today. The plate's open faces
  are read as flux transmittances, which next to a lossy medium are not an energy statement (`md/wave.md` 8). A rigid
  (closed) plate beside the same lossy layer is exact in both orders (*Closed face* F), and every acoustic material of
  section 1 is a rigid plate.
- **The phase of a held leg in a non-rigid mass-law body**: with tracer 5 now held as well (A7), *Phase reference … the
  mass-law part* misses its bound by up to 6.26° (the walk-off phase uses the mapping's index for a wave that keeps the
  air direction). For phase 3 with A7.

### 2.6 The failing assertions, read

**R2** (10 cases, 299 assertions). They restate conventions of `md/acoustic.md` 2, which P5 opened:

- the two leaks (*Closed plate*: `tf² × mass term`, `|z|² = 0.04` for tf 0.2 on each face; *Closed face* G-type sums);
- the Fresnel faces of a non-rigid mass-law sheet as the base of the mass term (*Thin sheet - mass-law angle
  dependence*, up to 18.1 dB at 89.9°: the base the test divides by still has two faces, the sheet now one);
- the solver's reference of the pair-rule convention (*Stack with a mass-law layer*, *Lumped columns … ordered stack*,
  *Total reflection of the outer pair*: 9e−4 to 0.037 absolute against `diag_sc_stack`'s old convention);
- the followed bounce in a mass-law body (*Bonded mass - the plate at any depth*, up to 8.84 dB: M2);
- what wood | concrete pays per direction (*Bonded mass* line 766: 3.01 dB, the old per-face reciprocity);
- and the three residuals of 2.5 (the 1.76 / 3.01 dB rows of *Held lining* E, the order rows, the phase rows).

**R1h, R1m** additionally (15 cases): every assertion that reads the mass term as a separable factor of the transmit
port (`paid == plate`, `paid == TL(c_air)`: *Bonded mass* ×3, *Mass cosine*, *ray_progress - the bonded plate*,
*Leak direction*, *Lossy verdict*): under R1 the plate passes F(a, τ), which is not a·τ, so the term "paid" is no longer
the mass term alone. The cases themselves need restating, not the rule.

**M2 alone**: *Ports of a mass-law face from inside* (18: the reflect port from inside is now zero) and *Bonded mass - the
plate at any depth* (35: the bounce), as i47 measured (53 assertions).

### 2.7 Next (phase 3)

The expressibility tables per class on a formula copy confirmed against the walker, as sheet and as traced solid and
through `ray_progress`; the window rows of B4/B4m and the A11 form; M4a/M4b on glass and gypsum coincidence; A10 with
N5; the invariants per variant; A2, A3, A4, A5, A7 re-examined (A4 and A6 are closed by the M1 core as built; A7 is
half in it).
