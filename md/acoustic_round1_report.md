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

## 3. Phase 3 — references, expressibility, decisions

Scratch: `build_linux/scratch/a1/p3` (`README.txt` lists every file). No code in `src` or `tests` was touched.

### 3.0 First step

`cmake --build build_linux --parallel` was current. Baselines from the repository root: `[wave]` **165110 assertions in
179 test cases**, `[acoustic]` **31856 in 35**, whole suite **1306515 in 1121**, all passed. The candidates were
regenerated (`wi/mkc.py`: R2, R1h, R1m 5041 lines, M2 5033, B4 5000, B4m 5001, B4s 5006, M4a 4998, M4b 5016, against
5016) and rebuilt (`wi/bld1.sh`: `tc_<v>` with the case listener, `w_<v>` the walker). The plain copy `rmi_id`
reproduces `test.txt` (the seed line only). `[wave]` under every candidate: 165110 assertions in 179 cases passed, **0
rows moved** against the plain copy (`wi/c_wave_<v>.txt`), the stored twins asserted.

### 3.1 First-principles references for the bands without a measurement (step 0)

**Method** (`p3/fp.py`; every number below is printed by a script in `p3`). A plate in a rigid baffle between two
half-spaces of air, plane waves, the constants of section 1:

- plate impedance at the trace wavenumber: thin (Kirchhoff, as in phase 1) or thick (Mindlin: shear with correction
  5/6 and rotary inertia, `Z = jωm − j·κGh·k²(D*k² − ρIω²)/(ω(D*k² + κGh − ρIω²))`, complex moduli `(1 + jη)`);
- loss factor: constant (phase 1), or internal plus the coupling loss into the frame, `η = η_int + m′/(C_e·√f)`
  (the 1/√f law of coupling losses: Craik, Applied Acoustics 14 (1981), abstract only; `C_e` fitted);
- **INF**: the infinite plate, field 0–78° (phase 1); **FIN**: the finite a × b plate, the forced wave windowed to the
  plate and radiated by the Rayleigh integral on both faces (spatial windowing, Villot et al., JSV 245 (2001) 433;
  `τ(θ, φ) = 4Z₀²σ/(cos θ·|Z_p + 2Z₀σ|²)`), diffuse 0–90°; **MOD**: the modal sum of the simply supported a × b plate
  (thin-plate modes) under each plane wave, radiated by the same Rayleigh integral with every cross term, light fluid
  (light panels: each mode also loses its own radiation);
- lab sizes: Riverbank 4.37 × 2.84 m (BIA TN 5A, RD066); AGC 1.23 × 1.48 m; boards 1.2 × 2.4 m; **the NRC wall size
  is not in the Warnock paper's text** (a scan), the Riverbank size is used;
- bands: three (FIN, INF) or sixteen (MOD) frequencies per third octave, octaves as energy means of thirds.

**Checks of the numerics** (`t_fp.py`, `heavy_modal.out`): the windowed σ tends to 1/cos θ (1.0019 / 0.9994 at 0° for
2 m / 8 m plates, 2.0425 / 1.9859 at 60°); FIN and MOD, two independent computations, agree on 4 mm glass within
0.1 dB (18.46 / 18.35 dB at 125 Hz, 27.24 / 27.35 dB at 500 Hz); on the heavy walls from about 500 Hz up (concrete 1 kHz
third 59.2 / 60.9 dB). A first FIN run with 24 incidence angles missed the narrow coincidence peak (thirds jumping by
up to 25 dB above 1 kHz); σ is now taken on 48 angles and interpolated onto 4000, where the plate impedance is
evaluated, and INF uses 4000 angles.

**Why the phase 1 models miss the measured heavy walls** (`heavy.out`, thirds 100–5000 Hz, rms of model − measured;
best fit of the loss parameters per form):

| form | concrete 152 | brick 95 | brick 203 | block 190 |
| --- | --- | --- | --- | --- |
| INF, η constant (phase 1) | 8.63 | 4.28 | 6.52 | 6.49 |
| INF, η_int + m′/(C_e√f) | 6.41 | 2.89 | 4.69 | 4.59 |
| FIN, η constant | 8.36 | 4.77 | 6.72 | 6.44 |
| FIN, η_int + m′/(C_e√f) | 6.04 | 3.18 | 4.38 | 4.32 |
| FIN, Mindlin, η_int + m′/(C_e√f) | 5.19 | 2.79 | 3.18 | 3.34 |

Each candidate acts where physics says it should, none closes the gap:

- **Coupling loss into the frame** is the largest single effect: it raises the coincidence region (more loss) and
  lowers the slope above it from 9 to 7.5 dB per octave (η falls as 1/√f). Fitted `C_e` 400 / 600 / 1500 / 300; with
  `η_int` 0 in every best fit. Implied η at 125 Hz 0.078 / 0.028 / 0.024 / 0.067, at 4 kHz 0.014 / 0.005 / 0.004 / 0.012
  (`etas.out`).
- **Finite size** removes the infinite plate's grazing waves (no 78° cap needed) and lifts the coincidence region of
  the windowed form by −0.7 to +1.9 dB at 125 Hz (INF → FIN with the coupling loss); the modal sum lifts it more (concrete third 125 Hz: FIN 32.7, MOD 42.0, measured
  43.2 dB) but is dominated by single modes there: about one mode per third octave at 125 Hz on 152 mm concrete, so
  neighbouring thirds of the modal sum differ by 5–10 dB (concrete 42.0 / 35.6 dB at 125 / 160 Hz).
- **Thick plate** (Mindlin, elastic shear modulus) lowers the highest octaves by 1.5–3 dB (concrete 4 kHz 74.0 →
  72.2 dB); the thin-plate limit (free bending wavelength three thicknesses, Ljunggren 1991, abstract) lies near the
  top of the range for these walls with their elastic constants.
- **Modal behaviour below the first modes** (f₁₁ 47.2 / 32.4 / 51.9 / 41.5 Hz): stiffness control and dips at single
  modes in the 31.5 and 63 Hz octaves (table below), not one dip at one mode.

**The best physical form against the measured octaves** (`spread.out`; FIN Mindlin, elastic shear, fitted `C_e`;
MOD+FIN = the modal sum below 400 Hz in its place):

| wall | form | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concrete 152 | FIN | **−7.1** | **−4.6** | −1.7 | +2.5 | **+3.9** | **+7.3** |
| | MOD+FIN | **−5.3** | −3.0 | −1.4 | +2.5 | **+3.9** | **+7.3** |
| brick 95 | FIN | −1.6 | **−4.0** | +0.2 | +1.0 | +0.5 | +1.4 |
| | MOD+FIN | **−5.6** | **−3.1** | −0.6 | +1.0 | +0.5 | +1.4 |
| brick 203 | FIN | **−6.1** | −1.0 | +0.2 | +2.6 | +2.1 | +2.6 |
| | MOD+FIN | **−5.2** | +0.9 | +0.4 | +2.6 | +2.1 | +2.6 |
| block 190 | FIN | **−3.5** | **−4.9** | +0.9 | +2.9 | **+3.1** | +2.1 |
| | MOD+FIN | −2.4 | −2.8 | +2.2 | +2.9 | **+3.1** | +2.1 |

Bold: outside the guideline of about ±3 dB. **The physical model does not reach the measured heavy walls within the
guideline**: it is 4–7 dB low in the coincidence region (125–250 Hz) on three of four walls and 4–7 dB high on
concrete from 2 kHz.

**What closes it is not physical** (`heavy2.out`): with the shear modulus as a free factor `g` (G = g·E/(2(1 + ν))) the
windowed form fits every wall (rms 1.01 / 2.46 / 1.52 / 2.07 dB, octave residuals within −2.7 … +3.5 dB), but with g
0.05 for cast concrete (a homogeneous solid whose shear modulus follows from E and ν) and a loss factor of **0.62 at
125 Hz** for concrete and 0.32 for the block (`etas.out`), against 0.02–0.08 for the physical fits. And these fits do
not survive the modal sum: with MOD below 400 Hz they read +4.4 / +6.0 dB (concrete 125 / 250 Hz) and +7.0 dB (brick
203 at 250 Hz). For masonry a softer shear (joints, hollow cores) is plausible, g 0.1 is not sourced.

Two readings of the residual pattern, neither tested here: the 1978 concrete measurement may be limited by the
facility at the top (RD066 gives STC 57 for 152 mm and 58 for 203 mm, where the mass ratio alone is worth about 2.5 dB
more; and the thirds 2.5–3.15 kHz are flat, 63.9 / 63.7 dB), and laboratory values below 200 Hz depend on the rooms'
modal fields; both would make the measurement, not the model, the outlier. Neither changes the verdict: the
measurement is the reference (P1).

**The spread of the model forms in the bands without a measurement** (the four forms of each wall: FIN and MOD+FIN,
each with the elastic and with the fitted shear; `spread.out`):

| | 31.5 Hz | 63 Hz | 8 kHz |
| --- | --- | --- | --- |
| concrete 152 | 38.0 … 49.8 (11.8) | 31.7 … 42.8 (11.1) | 70.4 … 77.6 (7.2) |
| brick 95 | 21.0 … 33.8 (12.8) | 28.3 … 36.5 (8.2) | 66.1 … 68.2 (2.1) |
| brick 203 | 40.1 … 44.6 (4.5) | 27.9 … 41.8 (13.9) | 70.2 … 72.4 (2.2) |
| block 190 | 27.8 … 37.7 (9.9) | 30.0 … 37.6 (7.6) | 66.8 … 70.2 (3.4) |

The one independent check, the block wall's measured thirds 63 / 80 / 100 Hz (36.0 / 33.8 / 35.5 dB): the forms read
−0.7 … +1.7 dB at 63 Hz, −5.1 … +4.6 dB at 80 Hz, +1.5 … +3.2 dB at 100 Hz. The 63 Hz third is met by every form; the
80 Hz third separates the modal sum (−5.1 dB, a mode) from the windowed form (+4.2 dB, no modes) by 9 dB.

**Light panels** (`light.out`, `spread.out`): the infinite plate stays on EN 12758 within 2.5 dB (125 Hz–4 kHz, η 0.015
fitted in phase 1); the finite forms within 2.5 dB for 4 mm and within 3.9 dB for 6 mm (125 Hz +3.9 FIN, 2 kHz −3.9).
In the octaves without a measurement:

| | 31.5 Hz | 63 Hz | 8 kHz |
| --- | --- | --- | --- |
| glass 4 mm, 1.23 × 1.48 m | 5.1 … 18.6 (13.5) | 9.3 … 16.0 (6.7) | 37.0 … 37.2 |
| glass 6 mm, 1.23 × 1.48 m | 7.4 … 18.8 (11.4) | 12.3 … 20.5 (8.2) | 43.0 … 43.2 |
| gypsum 12.5, 1.2 × 2.4 m | 4.3 … 12.1 (7.8) | 8.3 … 13.5 (5.2) | 34.0 … 34.1 |
| plywood 18, 1.2 × 2.4 m | 2.3 … 14.0 (11.7) | 9.8 … 15.5 (5.7) | 42.8 … 43.0 |

The windowed finite pane is 7.8–11.2 dB above the infinite plate at 31.5 Hz: a pane small against the wavelength
(10.9 m at 31.5 Hz) radiates weakly; the modal sum lies on either side (plywood 2.3 dB, below the infinite plate:
its first mode at 22.3 Hz). **The low octaves of a panel are a property of its size**, as A11 says of its
modes, and no column carries a size. 8 kHz is well determined (spread ≤ 0.2 dB).

**Absorption below 125 Hz** (`light.out`, `low.out`):

- A pane: the infinite plate with a mounting resistance r_m·ωm′ fitted to the measured 125 / 250 Hz octaves reaches
  at most 0.158 / 0.084 (4 mm, r_m = 1, the best of 0 … 3) where 0.35 / 0.25 is measured (window glass), 0.110 / 0.057
  where 0.18 / 0.06 (6 mm). The measured low-frequency absorption of glass is not an infinite pane on a resistance;
  its extrapolation to 31.5 / 63 Hz (0.46 / 0.28 for 4 mm on that model) has no validated reference.
- A lining (board over a cavity on a wall): the leaf-over-cavity model with a fitted mounting resistance of report
  1.10 (peak 0.278 at 150 Hz on 25 mm, 1.08 octaves); below the peak it falls to 0.005 at 63 Hz and 0.000 at 31.5 Hz
  (r_m 0.2), above the peak it is a model, not a measurement, at every band.
- Heavy walls: what the infinite plate passes and dissipates is 0.001–0.109 at 31.5 Hz and 0.002–0.053 at 63 Hz
  (`mat.out`, with the first mode); the measured surfaces
  absorb 0.01–0.02 at 125 Hz with nothing known below.
- Porous layer (11 kPa·s/m²): Miki stays positive below Delany–Bazley's range: 0.003 / 0.013 (20 mm), 0.006 / 0.021
  (30 mm) at 31.5 / 63 Hz; Delany–Bazley is negative there (−0.007 / −0.004). Spread about 0.02 in α.

### 3.2 Stop

The brief's condition for stopping after step 0 holds: **the heavy-wall model does not reach the measured bands within
the guideline**, with physical or sourced constants (concrete −7.1 / +7.3 dB, brick 203 −6.1 dB, block −4.9 dB,
brick 95 −4.0 dB in single octaves). The form that does reach them needs a shear modulus 10–20 times below the elastic
one and, for concrete, a loss factor of 0.62, and fails again by up to 7 dB once the finite wall's modes replace the
windowed wave below 400 Hz. Steps 1–7 (the expressibility tables, the invariants, A10, A11, N7, the re-examination and
the decision proposals) were not started, so that no knob of the library is fitted to a reference that is not
validated.

What the analysis does give, for the decision on how to continue:

1. In the measured bands (125 Hz–4 kHz) nothing changes: the measurement is the reference (P1), and the expressibility
   test can run there as planned.
2. 8 kHz: well determined for light panels (spread ≤ 0.2 dB); 2.1–7.2 dB for heavy walls (concrete widest).
3. 31.5 and 63 Hz: no validated reference for any class. Spreads 4.5–13.9 dB (heavy walls) and 5.2–13.5 dB (light
   panels) between forms that all describe the same physics; the light panels' low octaves depend on the pane's size;
   the one measured heavy-wall point (block, 63 Hz third) is met by every form within 1.7 dB, its 80 Hz neighbour
   separates them by 9 dB.
4. Absorption below 125 Hz: no validated reference for panes; porous layers within about 0.02; heavy walls and
   linings only from models.

Options (yours to choose): (a) run the expressibility test on 125 Hz–4 kHz (measured) and 8 kHz (light panels), and
carry 31.5 / 63 Hz with the spreads above as their tolerance; (b) take the guideline in the low octaves from the
spread (about ±5 to ±7 dB) and a size class per panel (a lab-size pane, a room-size wall) as the reference geometry;
(c) look for measured low-frequency data first (laboratory TL down to 50 Hz exists for some walls, e.g. NRC reports of
the kind Warnock's Fig. 1 came from); (d) accept the fitted-shear form as an empirical description of masonry only
(two to three fitted quantities per wall), not of concrete.

### 3.3 Decisions after step 0 (Stephan, 2026-10-15)

Recorded in `md/acoustic.md` 10 (P6–P8): the reflect side first (absorption per octave, A10, the reflection phase as
the new open item A12 with N7), then transmission with looser tolerances; bands without a measurement take the model
spread of 3.1 as their tolerance. One check of a practical reference below 125 Hz (`p3/flat.py` → `flat.out`): the
measured 125 Hz octave of each heavy wall, carried down flat, lies inside the model spread at 63 Hz for all four walls
and at 31.5 Hz for three (brick 203: 2.8 dB below the lowest form); it is also what the one measured wall shows (block:
35.1 dB at 125 Hz, 36.0 / 33.8 dB in the thirds 63 / 80 Hz).

### 3.4 The reflect side first: absorption per octave (P6)

Rig `p3/ab`, `p3/por`, `p3/ph`. A rigid mass-law face reflects `R = 1 − tf`, flat in angle, in both mappings and under
every candidate (`p3/ph/ph1.out`: |r| 0.9486 for `tf` 0.1 at every angle; M1 keeps the entry face, report 2.4). Its
room-side absorption is therefore `tf(f)`, a power law clamped at one.

**`tf` power law against the measured rows** (`ab/absfit.out`, minimax; 31.5 / 63 Hz: the band between the flat
continuation of 125 Hz and the model of 3.1; guideline ±0.05):

| class | row | max \|resid\| |
| --- | --- | --- |
| 1 | window glass (MHA) / 6 mm glass / door | 0.045 / 0.008 / 0.033 |
| 1 | plasterboard on frame, 100 mm (falls, then rises) | 0.051 |
| 2 | smooth concrete / brickwork / porous blocks | 0.009 / 0.007 / 0.028 |
| 3 | double glazing | 0.038 |
| 5 | **plasterboard on 25 mm battens** (peak at 250 Hz) | **0.084** |
| 5 | wooden lining on frame | 0.047 |
| 5 | **cork board on battens** (peak at 500 Hz) | **0.228** |
| 6 | plaster on masonry | 0.008 |

Every monotone row is reachable today; the peaked rows are A10 (3.8).

**Class 4, a porous layer on a hard wall** (`por/cmp_id.out`): 20 mm of Delany–Bazley (11 kPa·s/m²) with exact
per-frequency columns in the two-parameter mapping (`ε = n·z`, `μ = n/z`), glued on a 152 mm concrete wall, through the
walker against the exact layer on a hard wall:

| octave | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | --- | --- | --- | --- | --- | --- |
| exact (field 0–78°) | 0.014 | 0.084 | 0.244 | 0.476 | 0.703 | 0.859 |
| wall in the **one-parameter** mapping | **0.403** | **0.390** | **0.426** | 0.537 | 0.713 | 0.879 |
| wall in the **two-parameter** mapping | 0.015 | 0.084 | 0.244 | 0.476 | 0.703 | 0.859 |

Per angle the two-parameter backing is within 0.003 of the exact layer everywhere; the one-parameter backing is off by up
to 0.53 (125 Hz, 60°): its reflection phase of 2θ (3.5) turns a thin absorber into a strong one at oblique incidence.

**Can the library's power-law columns reach the measured absorber?** (`ab/porfit.out`: `a, b, c, d` for ε, `e, f, g,
h` for μ, fitted to Knauf TP 120 A, 31.5 / 63 Hz: the band between Delany–Bazley clipped at zero and Miki): **20 mm within
±0.025, 30 mm within ±0.032 in every octave 31.5 Hz–8 kHz**, on the two-parameter backing; confirmed through the walker at
125 / 500 / 2000 Hz within 0.001 per angle (field 0.1042 / 0.4959 / 0.8652 against the formula's 0.1047 / 0.4962 /
0.8653). Class 4 needs no code, only the two-parameter mapping on the wall behind it (A12).

### 3.5 A12 and N7: the reflection phase and the two-parameter mapping

**The phase of a rigid face** (`ph/ph1.out`, one face through `ray_mesh_interact`, scalar):

| mapping | 1° | 10° | 45° | 60° | 85° | 89° |
| --- | --- | --- | --- | --- | --- | --- |
| one-parameter (today), glass / concrete / gypsum, with or without `tf`, half-space or sheet | 2.0° | 20.0° | 90.0° | 120.0° | 170.0° | 178.0° |
| two-parameter, glass | 0 (open) | 0.009° | 0.055° | 0.095° | 0.628° | 3.1° |
| two-parameter, gypsum | 0 (open) | 0.012° | 0.194° | 0.340° | 2.25° | 11.3° |

The two-parameter mapping gives the fluid-physics phase of a hard wall; `tf` does not change it (|r| 0.9486 for `tf`
0.1, no step at the critical angle). A panel absorber's own phase (12.5 mm board on 25 mm air, the model of 1.10,
`ab/liningphase.out`) swings from −18° to +15° through its resonance, which the pad (phase of the face) does not carry.

**Air | water** (N7, water is m = 0, every candidate identical): through a traced run (`n7/w_*.out`) the two-parameter
mapping is exact for both tracers, T and R within 1.7e-7 dB of the fluid reference at every angle, critical angle between
13.37° and 13.38°; the one-parameter mapping passes nothing from 5° (reference −29.9 dB). Through `ray_progress` with a
band of four frequencies (`rp/cmp_w.py`): T within 7.2e-5 dB (float path storage), R within 2.3e-7 dB, both tracers,
every frequency.

**Rigid materials in the two-parameter mapping** (`n7/n7_*.out`, 500 Hz, 4 mm glass and 12.5 mm gypsum, sheet and
traced solid, across 3.48° / 9.44°): no step at the critical angle in today's copy or any candidate (within 0.05 dB);
the level differs from the one-parameter mapping by at most 0.09 dB. But **today a traced rigid solid below its new
critical angle lets tracer 5 refract into it: 23.7 dB (glass, 3.47°) and 28.6 dB (gypsum, 9.43°) from its type-4 twin**;
under every M1 copy (no tracer bends into a mass-law body) the difference is 0.000 dB and both ports stay below 0.93.

### 3.6 Transmission: what each variant can reach

Rig `p3/tl`. The rule: **`tf` stays at the class's absorption fit** (3.4), only the transmission knobs move: level
(`fRef`), slope (`m`), `att` / `attB`, a resonance (`res*`: + the dip of B4m, − the single-panel form of B4s), the
bending stiffness (M4a: `coiF`, `coiQ` in X0), a negative coincidence dip (M4b). References: measured octaves 125 Hz–4 kHz;
31.5 / 63 Hz and 8 kHz the model spread of 3.1 (P7); gypsum and plywood the model spread in every octave; double units,
the lining on dabs and plaster one face only 125 Hz–4 kHz.

**The formula copy** (`tl/libac.py`) against the walker (`tl/cmp_chk.py`, walls that switch every term on, sheet and
traced solid, types 4 / 5, five angles, four frequencies): R1hX / R1mX within 0.0022 dB, R2X within 0.099 dB (the open
face at exactly 0°), R within 0.0008 dB; today's rules with the new X0 (X) within 0.20 dB as a sheet, its traced solid
1.32 dB off (the followed bounces). Fits at 192 angles × 6 frequencies per third, re-evaluated at 768 × 12; knobs
bounded to physical ranges (coincidence 30 Hz–20 kHz, Q 1–300; resonance 10 Hz–1 kHz, Q 0.3–50): a first unbounded
run put a block wall's coincidence at 0.88 Hz, a sub-degree feature the 48-angle fit could not see and the walker did
(`tl/alias.out`, 3 to 15 dB between samplings).

**Worst octave residual over the 13 constructions** (`tl/rmb_chk.out`, `tl/dense_chk.out`, dB):

| knobs | today (`tf²·τ`) | R2 (`a·τ`) | R1h | R1m |
| --- | --- | --- | --- | --- |
| level, slope, `att` | — | — | 6.16 | — |
| level, slope, `att` + resonance (B4s) | — | — | 4.89 | — |
| level, slope, `att` + bending (M4a) | — | — | 4.67 | — |
| **+ resonance + bending** (bounded) | **6.13** (gypsum) | **1.78** | **2.51** | **2.50** |
| + resonance + bending + M4b | 4.23 (6 mm glass) | 3.73 | 2.66 | 1.98 |

Per construction under R1h with resonance and bending (`rmb_chk.out`): glass 4 mm 0.51, glass 6 mm 1.68, gypsum 1.60,
plywood 1.32, concrete 0.87, brick 95 0.21, brick 203 0.80, block 1.02, 4-16-4 0.71, triple 0.77, furred lining 1.70,
lining on dabs 2.51, plaster one face 0.09. The fitted coincidence frequencies of the light panels and units land on the
physical ones (glass 4 / 6 mm 2825 / 1938 Hz against 2989 / 1993 Hz, gypsum 2602 against 2500, plywood 1105 against
1146, the units 3167 / 2697 Hz against the 4 mm pane's 2989): M4a is the bending stiffness of the plate model, not a
fitted shape. On heavy walls `coiF` is not physical (49 to 5188 Hz): there the knob shapes the slope.

**Through the walker** (`tl/conf_cmp_rm.out`, R1hF, 96 angles, one frequency per third): walker and formula within
0.0014 dB, sheet and traced solid equal, every construction within 2.3 dB except the block wall at 4 / 8 kHz (11.4 / 9.5
dB, its unbounded knobs: the 0.88 Hz coincidence above). **Through `ray_progress`** (`rp/band_R1hF.chk`, fitted glass,
concrete, 4-16-4, sheet and traced solid, both tracers, 0 / 30 / 60°): a band of six frequencies equals its single
frequencies bit for bit, the traced solid passes in one path.

So: **today's rule cannot carry both sides** (with `tf` at the measured absorption it misses the light panels by 2.5 to
6.1 dB whatever else is set); **every M1 rule with resonance and bending stiffness reaches all 13 constructions within the
guideline; M4b is not needed**.

### 3.7 Invariants per variant (`p3/inv`)

Scan: glass 4 mm and gypsum in both mappings, concrete, brick 203, the double unit (every term on: `tf` law, `coi*`, a
single-panel or double-unit resonance), wood | concrete as two solids, sheet on solid and stack, both directions; types
1, 2, 4, 5; 22 angles 0–85° across every critical angle; 63 Hz–4 kHz; summed and followed (24,640 rows per variant).

| | R2X | R1hX | R1mX | R1hG |
| --- | --- | --- | --- | --- |
| max T + R | 0.994 | 0.99945 | **1.0000016** (48 rows, wood \| concrete at 63–250 Hz, 70°: min(a, τ) = a) | 0.99945 |
| sheet on traced solid, scalar | 0.002 dB | 0.002 dB | 0.002 dB | 0.002 dB |
| step across the critical angle, scalar | ≤ 0.121 dB | ≤ 0.004 dB | ≤ 0.001 dB | ≤ 0.004 dB |
| reciprocity of wood \| concrete | 0.001 dB | 0.001 dB | 0.001 dB | 0.001 dB |
| summed / followed; tracer 5 / 4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| `[wave]` and twins | not run | not run | not run | **0 rows** (R2, R1h, R1m alone: 0 rows, 3.0) |
| lines (against 5016) | 5036 | 5036 | 5036 | 5044 |

EM types on the two-parameter gypsum: the open face below 9.44° splits TE and TM (traced 1.23e-5 / 1.76e-5 at 9.43°),
0.86 dB between sheet and solid there, and a 0.86 dB step at 9.43° / 9.45°; no EM type is acoustic. Today's copy did not finish the scan: with no internal
summation in a mass-law body the walker's tree explodes on a light pane (documented limitation, `md/acoustic.md` 2.5).

**Suite cost** (whole suite against the plain copy): R1hF 15 cases / 2,159 assertions (the same as R1h in phase 2);
**R1hG 15 / 2,153**, 19,041 rows moved; `[acoustic]` R1hF against R1h: 151 rows in 9 already failing cases (the X0
rounding rows, the exit fix, 3.9). Passing cases with moved rows under R1hG: *Leak direction* (289 + 88 rows: levels
printed, directions asserted; the wall behind a traced layer −68.90 → −50.87 dB, no second leak), *ray_progress - the
wall behind a traced layer* (198), *Closed face* (39), *Lossy verdict - the mass-law part* (32), and eight more with 1–24
rows (`wi/attr_R1hG.txt`).

### 3.8 A10 and A11

**A10** (`ab/a10fit.out`, `ab/a10tl.out`; formula level, field 0–78°, max |resid| in α):

| row | today | (a) max(tf, τ), B4s | (a) max(tf, τ), B4 | (b) peak on `tf` |
| --- | --- | --- | --- | --- |
| plasterboard on 25 mm battens | 0.084 | 0.015 | 0.082 | **0.005** |
| wooden lining | 0.047 | 0.036 | 0.044 | **0.013** |
| plasterboard on frame | 0.051 | 0.017 | 0.013 | **0.012** |
| window glass | 0.045 | 0.025 | 0.028 | **0.017** |
| double glazing | 0.038 | 0.007 | 0.030 | **0.004** |
| cork board on battens | 0.228 | 0.063 | 0.228 | 0.076 |

(a) ties the absorption to the transmission again: the τ that gives window glass its 0.35 at 125 Hz is its transmission,
**TL 6.7 / 8.9 dB at 125 / 250 Hz against EN 12758's 17 / 20** (and +11.4 dB at 4 kHz). For a lining glued on a wall
it would also need the board's own τ inside a bonded plate, which the plate rule sums away. (b) reaches every lining and
pane row and leaves transmission alone. (c), a wave medium with a pole, was not measured. **What a lining is in the
library**: one sheet on the wall's face (a cavity cannot be meshed into a glued pair, and a meshed double leaf has no
mass-spring resonance, section 3 item 5 of `md/acoustic.md`), its absorption from its own `tf` columns, its transmission
the lined wall as one material (3.6: furred board and board on dabs within 1.7 / 2.5 dB with resonance and bending);
the bonded plate rule would add only its mass.

**A11** (`tl/a11.out`): with the model spread as tolerance the low octaves are mostly reachable without the
single-panel form; gypsum is not (4.6 / 4.8 dB outside its band at 31.5 / 63 Hz with level, slope and `att`; 0.8 / 1.6
with B4s). The fits with a free sign use both forms, and not stably: unbounded (`tl/a11.out`) 8 constructions chose the
single-panel form and 5 the dip, bounded (`tl/rmb_chk.out`) 5 and 8, with different constructions. The data do not pin
the form; both shape the low end. A panel's first mode is a property of its size (3.1); a material row can carry a size class (a row per pane
size), a column cannot.

### 3.9 N6: the open items under M1

- **A defect of the M1 build, found here (A2)**: a coherent lining (m = 0) glued on the far face of a rigid wall read
  10 dB below its lined side under R1hX (−49.61 against −39.66 dB at 10°): `PLATE_OUT` is defined for layers 1…n only,
  so the wall's own face (the near half-space) stayed the transformer and passed `tf` while `ray_state_update` already
  paid the plate rule. **Exit fix** (one condition in `PLATE_OUT`, 0 lines): −39.57 dB.
- **A2** (`n6/a2.out`, lining 10 mm `eps 2.50 − 0.80j` and `0.31 − 0.20j` on a 100 mm rigid wall, tf 0.1, 10–75°):
  behind a held wave the lining is entered with the wall's impedance mapping as invariant (sheet) or unrefracted
  (traced). **R1hG** (the held exit into a wave medium judged and bent by the invariant of air, the bent flag written
  there, +8 lines): sheet and traced layer equal (≤ 0.01 dB), tracer 5 equal to 4 (0.00 dB, after writing the flag;
  4.97 dB without it), the dense traced layer no longer passes nothing from 60°. The two sides of the wall differ by
  0.06–1.34 dB (dense) and up to 2.78 / 4.57 dB (porous, one / two-parameter wall, 75°), the exit side higher: the
  wave convention for a lossy medium (`md/wave.md` 8, flux splitting `sec²(arg η)`); today 4.29 / 10.08 dB. The solver
  half (layer cosines behind a held near medium from the air invariant, R1hXf) breaks the porous sheet at 60–75°
  (−26.95 / −17.87 against −37.10 / −33.32 dB) and is not proposed.
- **A3** (`n6/a3.out`, 22 mm rigid mass-law panel, `alpha` 100 dB/m): the sheet pays the 400 dB cap from 10° on, its
  traced solid 2.22 / 2.52 / 4.31 dB at 10 / 30 / 60°. **A3 fix** (`qd_mass_amp` charges `alpha` over thk / cos in air,
  the held path, 0 lines; replaces `56dfb05`): sheet = traced solid at every angle and tracer (2.19 / 2.22 / 2.52 / 4.31 /
  11.74 dB at 0–80°). The tunnelling coherent layer of `md/wave.md` 8a is not touched (m = 0).
- **A4, A6**: closed by the M1 core (report 2.4); confirmed in every M1 scan (reciprocity 0.001 dB, no pair rule at a
  mass-law face).
- **A5** (`n6/a5.out`): a bonded pair pays both bodies' `att` in dB (3 + 3 = 6.00 dB) as two solids, sheet on solid and
  stack, today and under R1hF. A convention.
- **A7**: the core holds every tracer in a mass-law body. The phase rows (*Phase reference … the mass-law part*, `dense`
  = a non-rigid mass-law body, `eps 1.50`, m 20) read +0.37 / +1.51 / +6.26° at 10 / 20 / 40° against a **refracted**
  reference, and under R1h (phase 2) and R1hF **NaN from 60°**: the held wave was judged at the exit by the body's own
  invariant (√1.5 · sin 60° = 1.06, closed) and passed nothing from 55°. R1hG judges the held exit by the air invariant:
  finite, +15.2 / +26.2 / +36.4° at 60 / 75 / 85° against the refracted reference. The rigid rows stay within 0.0000°.

**The three M1 residuals of report 2.5**: (1) the state one level deep (3.01 dB, four bodies K | T W | K): closing it
needs state the ray does not have; accepted limitation. (2) A non-rigid mass-law layer beside a lossy coherent layer,
0.11–0.20 dB: R1hF reduces the residual column slightly (6.3e-2 → 5.2e-2 at 70°, *Stack with a mass-law layer*); no
acoustic material of section 1 is non-rigid; accepted limitation. (3) The phase rows: the test's refracted reference for
a body M1 holds; a restatement, not a defect, once the NaN rows are fixed (R1hG).

### 3.10 Decision proposals

Each item: what happens now, what would change, the rows that move, the option.

**N1, the M1 rule.** Now a rigid sheet passes `tf² × mass term`: `tf`, the room-side absorption, also scales the
transmission twice, so no setting gives a pane its measured absorption and its measured transmission loss (today's
rule misses them by 2.5 to 6.1 dB with `tf` held at the absorption, 3.6). Under M1 the plate passes `F(a, τ)` and
reflects `1 − a`: absorption and transmission are set by different columns. All three rules reach all 13 constructions
(R2 1.78, R1h 2.51, R1m 2.50 dB). R2 (`a·τ`) keeps `a` in the transmission (painting a wall would change its TL);
R1m sits exactly on the passivity bound and exceeds it by 1.6e-6 in 48 scan rows; R1h (`a·τ/(a + τ − a·τ)`) is smooth,
strictly passive (max T + R 0.99945), equal to τ wherever τ ≪ a. **Proposal: R1h.** Suite: 15 cases / ~2,150
assertions, every one restating a convention of `md/acoustic.md` 2 (report 2.6). Replaces `b484257`, `b2ce6cd`,
`ab77976`, `841d0d1`, `807a1d6`, `aecaf04`, `6998b10`, `4d00299` (report 2.4).

**N2, M2.** Part of the M1 core for every held body (inner and last faces reflect nothing); no separate switch. The
traced solid then passes in one path (3.6) and the A1 bounce at a sharp resonance is gone with the pole (N3).
**Proposal: with N1.**

**N3, M3 and `res*` on a mass-law material.** Now `res*` is a permittivity pole, which makes a rigid body lossy, never
closed, and lets it create energy (A1; on the walls of 3.6 today's sheet and its traced solid differ by up to 143 dB (4 mm glass, single-panel resonance, 4 kHz, 80°), 99 dB (concrete), 76 dB (the double unit)). Proposed: the pole
leaves the permittivity of a mass-law material and acts in X0 (B4s): **`resS > 0` the dip of a double unit or lining
(B4m, capped at one), `resS < 0` the panel's own first mode (stiffness below `resF`)**; the fitted values (bounded R1h, 3.6) are not a guidance yet: the double units chose the single-panel sign at 226 / 231 Hz (Q 1.9 / 3.1), the linings 45 / 240 Hz (Q 1.0 / 2.6, both signs), the light panels 10–121 Hz (Q 2.8–17.6); the sign a fit chooses changes between fit runs (3.8). Lines: part
of the X0 block (R1hX 5036, the plate rule 5041 without it). Replaces the pole in `eta()` for m > 0 (`74d6d9b`) and the
mass block (`b2ce6cd`). **Proposal: B4s.**

**N4, coincidence.** M4a (the bending stiffness of the plate model in X0: `coiF` the coincidence frequency, `1/coiQ`
the loss factor) is the decisive knob for light panels and units (4 mm glass 4.8 → 0.5 dB, 3.6) and recovers their
physical coincidence frequencies. M4b (a negative `coiA` without an `att` baseline) is not needed (worst 2.51 dB without
it) and moves nothing in the suite (N4 marker, phase 2). **Proposal: M4a; drop M4b; rule of use: `coiA` = 0 on a
mass-law material** (the Lorentzian in `interface_gain` keeps its wave meaning).

**N5.** Measured in phase 2 (every traced entry has its second hit); its only use was A10(a), which is not proposed.
**Closed, no change.**

**N6.** A2: R1hG (+8 lines) and the exit fix (0 lines); the two sides of a lossy lining stay on the wave convention
(record under `md/wave.md` 8). A3: the held path in `qd_mass_amp` (0 lines); the wave half (8a) stays open, it moves no
row. A4, A6: closed by N1. A5: keep the sum, rule of use (one `att` per bonded run). A7: closed by N1 + R1hG; the
`dense` phase rows are restated to the held reference. A8 low priority, A9 last.

**N7 and A12, the two-parameter mapping.** Now every rigid face reflects with a phase of 2θ, a porous absorber on a wall
absorbs up to 0.39 too much at 125 Hz, water is closed from 0.016°. With `ε = n·z`, `μ = n/z` (columns only, no code):
water exact (traced and band), a hard wall's phase within 0.1° to 60°, the absorber within 0.003. Consequence: a rigid
sheet is open below 3.5–9.4°; no step there under any copy; today a traced rigid solid lets tracer 5 refract in (23.7 /
28.6 dB), under M1 not. **Proposal: the two-parameter mapping for every acoustic material, with N1.** What it does not
give: the phase of a resonant panel (±18° through its resonance, 3.5), which needs a surface impedance the columns do
not carry: recorded under A12.

**A10.** (a) rejected (window TL 10 dB low). (b) a resonant peak on the absorption reaches every lining and pane row
within 0.017 (cork 0.076); it needs three numbers (height, frequency, width). Options: three new columns, or the mass-law
meaning of `coi*` / `res*` shared with N3 / N4 (conflicts: a pane uses `coi*` for bending and `res*` for its mode).
**Proposal: (b) with three new columns; your decision on the columns.**

**A11.** **Proposal: keep both forms of `res*` (N3)**; the low-octave reference is the spread (P7); a panel's size is a
row, not a column.

**The machine as measured** (R1hG): R1h + B4s + M4a (+ M4b, inert at `coiA` = 0) + exit fix + A3 fix + the held exit by
the air invariant; **5044 lines (+28)**; `[wave]` 0 rows, twins asserted; whole suite 15 cases / 2,153 assertions.

## 4. The fixtures for the decisions (2026-10-15)

Decided (`md/acoustic.md` 10): N1 R1h; N2–N7 as proposed in 3.10; A10 with three new columns `tfP`, `tfF`, `tfQ`
(`tf(f) = tfR·(f/fRef)^tfB + tfP / (1 + (tfQ·(f − tfF)/tfF)²)`, clamped as before; unknown keys are ignored by today's
library, so the fixtures run on it). Eleven new cases, appended to `test_acoustic_state_update.cpp` (10) and
`test_acoustic_progress.cpp` (1), every reference written from the column definitions, every traced row through
`ray_triangle_intersect`, `ray_mesh_interact` and `ray_state_update` (`bf::walk`, `cf::walk`) or through
`ray_progress`, `ub::OneThread` in each. Measured against today's library and against the candidate copy R1hH (R1hG of
3.9 plus the A10 columns, `wi/mkc.py R1hH`, **5064 lines, +48 against 5016**, of which 20 for the columns), `p4/fxsum.out`:

| case | decision | today: failed / assertions, largest deviation | R1hH |
| --- | --- | --- | --- |
| *Plate rule - a rigid mass-law panel passes the incoherent series …* | N1, N7 | 166 / 320, T 37.17 dB off the rule (two leaks), R 0.040 dB | passes; 0.0022 dB, 0.0008 dB |
| *Plate rule - a wall bonded of two mass-law bodies …* | N1 | 18 / 36, T 18.93 dB off the rule | passes; 0.0007 dB, directions 0.001 dB |
| *Plate rule - a traced mass-law body passes in one path …* | N2 | 12 / 24, up to 9 paths behind the body | passes; 1 path, followed = summed |
| *Resonance - on a mass-law material res\* is the plate's resonance …* | N3 | 168 / 168, T 15.68 dB off, R 0.06 where 1 − tf is 0.95 (the pole) | passes; 0.0071 dB, 1.1e-4 |
| *Coincidence - on a mass-law material coiF and coiQ are the bending stiffness …* | N4 | 40 / 40, T 55.74 dB off | passes; 0.0022 dB |
| *Two-parameter mapping - air \| water, a hard wall's phase, a porous layer …* | N7, A12, A7 | 2 / 53: tracer 5 against 4 below the critical angle, 27.81 dB; water, the wall's phase and the porous layer hold today (guards) | passes; 0 dB |
| *Held exit - a lining behind a rigid wall is the same as its traced layer …* | A2 | 14 / 48, sheet against traced layer 951 dB (the dense layer passes nothing from 60°) | passes; 0.015 dB, tracers equal |
| *Held path - alpha on a lumped plate …* | A3 | 8 / 20, sheet against traced solid 397.8 dB (the cap) | passes; 0.001 dB |
| *Held exit - a non-rigid mass-law body holds every tracer …* | A7 | 4 / 8, tracer 5 against 4 6.80 dB (0.04 / 0.45 / 2.91 / 6.80 at 10 / 30 / 60 / 75°) | passes; 0 dB |
| *Absorption peak - tfP, tfF and tfQ …* | A10 | 80 / 80, α 0.25 below the peak, T 25.98 dB off | passes; 0, 0.001 dB |
| *ray_progress - the plate rule and the absorption peak at every frequency of a band* | N1, A10 | 80 / 120, α 0.25, T 26 dB; band = alone today | passes; band = alone |

Two references were corrected after a first run against R1hH, both my statements, not the copy's: the hard wall's
phase is now asserted against the fluid interface of the mapping (152 mm concrete reads 0.104° at 60° and reflects
1 − 4z = 0.99982 at 0°; the 0.1° and 0.9999 bounds had come from glass), and at exactly 0° the rigid face is open (report
2.4), so the absorption peak's reference there carries the pair rule's Fresnel term (+0.0011). A first version of the
non-rigid case passed today because its mass term was zero (X0 0.1); `fRef` 0.01 GHz gives X0 10.

**Nothing else moved** (today's library with the new test files): `[wave]` 165110 / 179, the seed line only; the 35
existing `[acoustic]` cases print exactly as before; the whole suite 1132 cases, 11 failing, 592 failed assertions,
every one in a new case, no existing row moved. `[acoustic]` is now 32773 assertions in 46 cases.

**What the source change will cost** (R1hH, whole suite with the new cases): 15 cases / 2,153 assertions fail, the 15
existing cases of report 2.6 (largest: *Bonded mass - the plate at any depth* 6,166 rows, *Mass cosine* 4,055,
*ray_progress - the bonded plate* 4,364, *Closed plate* 1,049), 24,569 rows moved (`wi/attr_R1hH.txt`); the 11 new cases
pass. Each failing assertion restates a convention the decisions replace (the two leaks, the followed bounce, the pair
rule's reference, the separable mass factor, the refracted phase of a non-rigid body); they are restated with the
source change.

## 5. The source change (2026-10-15)

Decided: R1h, N2–N7 as proposed, A10 with three new columns (`md/acoustic.md` 10). Fixtures first (section 4), then
the source. Not committed.

### 5.0 State

| run | before (bb336dc) | after |
| --- | --- | --- |
| `[wave]` | 165110 / 179 | **165110 / 179**, all passed, every line of `test_wave.txt` identical (the seed line only); the stored twins are asserted inside it |
| `[acoustic]` | 31856 / 35 | **32697 / 46**, all passed |
| whole suite | 1306515 / 1121 | **1307356 / 1132**, all passed |

The lines that differ between the whole-suite run and `test.txt` are exactly the lines that differ in `[acoustic]`
(3352 each, the summary line included); no case outside `[acoustic]` printed anything else. The baselines in the
repository root are replaced; the old ones are in `build_linux/scratch/a1/p6/old_baselines/`.

### 5.1 What is in `src/ray_mesh_interact.cpp`

**5082 lines against 5016** (+230 / −164 by `git diff --stat`). The copy R1hH of section 4 (5064) with the
resonance and bending factors combined in `mass_X0` (B4s with both forms of `res*`, M4a; M4b not built), 10 lines of
comment (the columns `resF` … `coiQ`, `tfP` … `tfQ`, `eta()`, `mass_X0`, `tf()`, the held exit), `require(tfF ≥ 0)`,
`require(tfQ ≥ 0)`, and the leak-chain fix of 5.2 (+8). In one list:

- the plate rule (N1) in the solver (`PLATE_OUT`, `PLATE_FIX`, `qd_plate`), at a traced plate's exit (`PLATE_IN`,
  `PLATE_ABS`) and in the leak chain; the pair rule and its clamp are gone from mass-law faces (A6);
- M2: a face inside a plate passes what reaches it and reflects nothing;
- the exit fix (`PLATE_OUT` includes the near half-space of a held run), A3 (`alpha` over `thk / mass_c`), the held
  exit into a wave medium judged and bent by the invariant of air, with the bent flag (A2, A7);
- `res*` and `coiF` / `coiQ` of a mass-law material in `mass_X0`, the pole removed from `eta()` where `m > 0` (N3, N4);
- `tfP`, `tfF`, `tfQ` (A10).

**The OBJ / API column lists** (`src/obj_file_write.cpp`, `api_python/qpy_obj_file_read.cpp`, `api_mex/obj_file_read.cpp`)
now list `tfP`, `tfF`, `tfQ` after `tfB`. They fix only the order: the readers always passed every CSV column through
and the writer wrote unknown columns after the canonical ones (alphabetically), so a material file could carry the
peak before; now the columns sit in canonical position and `csv_write_defaults` writes them (default 0). The expected
header and default row of the three writer tests (C++, Python, MATLAB) gained the three columns. Measured: C++ OBJ
cases 261 assertions in 17 cases; Python `test_obj_file_write` and `test_obj_file_read` 16 tests; MATLAB the same two,
passed (scratch builds `build_linux/scratch/api` and, with g++-12 for MATLAB's libstdc++ (GLIBCXX ≤ 3.4.30),
`build_linux/scratch/api12`).

### 5.2 Found while restating: the leak chain kept the product

*Leak reciprocity* failed by 12.17–12.47 dB at 10°–70° once its two directions were the same plate (a wood panel glued on
concrete without its mass law, 5.3). Fitting the rows: the entry passed `a·τ·b·G_w·G_c` exactly, both faces
multiplied. The branch is the leak chain (`qd_sheet_resolve`, `leak_chain`): a mass-law sheet whose far side the wave
cannot enter passes "the two crossings a traced solid would produce", and the traced solid now pays the plate rule at
its exit, so the chain no longer equalled it. R1hH did not touch it.

The fix (+8 lines): where neither neighbour carries the mass column (the sheet is the whole plate), each channel is put
on `a·τ / (a + τ − a·τ)` with `a = min(a_in, b)`, `a_in` and `b` the two crossings' flux over the interface gain
`interact_with` folds in (a first version without that division lost `G_w`, `G_c`, measured, corrected). Where a
neighbour carries the column the plate goes on and is paid elsewhere (the outer-pair entry, 5.3).

What it moves, measured against a what-if without it (`p6/rmi_noleak.cpp`, `[acoustic]`, three tables, nothing else):

| table | rows | before → after |
| --- | --- | --- |
| *leak reciprocity as a sheet* | 5 | entry over exit 12.47 / 12.46 / 12.43 / 12.17 dB at 10° / 30° / 45° / 70° → 0.00013 / 0.00017 / 0.00024 / 0.00094 dB (the flat first face, recorded) |
| *mass cosine D* | 8 | the glued exit behind `eps 1.50 − 0.40j` above an invariant of one: −0.39 dB at 1.01 … −1.83 dB at 1.45. The sheet there has no `tf`; its first face passes 1.013–1.525 in flux (the lossy medium's split), the far face 1, so the rule takes one. The traced convention (`a_out` from the far side, 1.318) would give −1.82 dB |
| *held lining G* (twins) | 6 | `W0 100 | K lining`, lining first: amplitude × 1.889 (+5.53 dB = −10 log10 0.28, the second face's leak), phase unchanged. The two orders through W0 now have the same magnitude (0.1416 / 0.1416 at 10°, 0.1416 / 0.1414 at 30°, 0.1415 / 0.1414 at 60°, within 0.012 dB), 5.5 dB apart before |

### 5.3 The restated cases

Every existing case that failed under the change was restated to the decided convention (none weakened; the
references written from the rule, the values measured). 13 cases changed, one renamed (*Closed plate - … passes the
plate rule of its two faces*, was *… passes its two leaks and nothing else*), the helper `diag_sc_stack` (mass-law
branch on the plate rule), and 5 in this step (after the summary of 5.0's first run):

| case | old convention | now |
| --- | --- | --- |
| *Total reflection of the outer pair … - the mass-law part* | entry and exit each a product of crossings | entry: the first face `a` times the panel's term, the rest at the wall's far face; exit (wall without mass law): the plate rule with `a = min(1 − Rn, 1 − Rf)`, the far face open at the wood's cosine. Exact (1e-9). Entry +0.86 dB at 0° … +2.55 dB at 70°, exit +7.80 … +9.18 dB; reflections and the phase unchanged |
| *Phase reference … - the mass-law part* | the dense body (`eps 1.50`, m 20) refracted into | held (A7): the walk-off reference for both bodies, 0.0000° in every row (was +0.37° … +36.4°). Tracer 5 against 4 on the dense body: equal to 4.7e-15 relative (rounding) |
| *Leak reciprocity …* | entry over CONC (m 30) against exit from CONC0 | the same plate both ways (wood on CONC0); the difference is the recorded flat first face through the rule, to 1e-13 |
| *ray_state_update - mass-law material is excluded from slab resolution* | type 33 at 0° (the undeviated bit) | type 1 at 0° and at 0.1° (A7) |
| *Ports of a mass-law face from inside …* | (passed) | stated: inside one plate T = 1, R = 0 from both sides (+12 assertions) |
| *Closed face …*, G | "its two leaks" | the plate's first face, T = 0.5 exactly (+3 assertions) |
| *Leak chain … - the mass-law part* | the step at the air \| wood critical angle recorded (< 5 dB, 4.67) | no step: both angles < 1e-3 dB (0.0000) |

Recorded rather than asserted (no power travels them): the level of the way back in *Bonded mass - sheet on sheet …*
(the bounce returning to the glued sheet, and the bounce inside the second body). The comments of the 11 new cases
that described bb336dc as "today" now say "at bb336dc".

### 5.4 Every moved table

Against `old_baselines/test_acoustic.txt`: 61 tables moved, 17 are new (the 11 new cases and 6 restated ones), 2
were replaced by restated tables (*plate depth S*, *plate depth B*). By cause, the largest change in the table's unit:

**The plate rule (N1), transmit levels.** A plate passes about `τ` where `τ ≪ a`, no longer `τ` times its faces'
absorption; a sheet glued on a mass-law wall pays its first face and its own term and the rest at the wall's far face.
- *closed-face hand-on* (through `ray_progress`, 198 / 396 rows): the power behind the 200 mm wall with `tf` 0.3 +8.1
  to +10.5 dB, 2 to 7 fewer paths behind it (M2); B 48 / 192, C 30 / 60, D 6 / 56: the wall's exit +9.9 to +10.5 dB.
- *lossy verdict* J and K (16 + 16): the lossless 200 mm wall's exit +10.45 dB (the second face's 0.3), K +9.07 to
  +10.06 dB. J with a loss tangent of 0.04 (recorded in the case: a rigid body given a loss is open and reflects
  1.000000 from 10° on): the exit event that created +18.8 to +68.3 dB is now −0.05 to −0.0002 dB and the direction in
  the wall is held (88.81° → the incidence angle), so the path is −87.95 dB at 10° … −101.76 dB at 75° (was −69.1 …
  −33.5 dB); at 0° +10.45 dB as lossless.
- *leak direction, rigid walls* (288): wood 50 \| concrete 50 heavy, total −68.90 → −50.87 dB (the two outer faces' −6.99
  and −10.00 dB and the inner face's −1.04 dB); the exit event can exceed one (+8.85 dB on *refracting tracer, a leak*,
  `air | conc 50 | wood 50 | air`): the plate's exit gives back the first face's absorption (5.5).
- *bonded mass* tables (sheet on sheet, sheet on solid, solid between two sheets, panel on concrete, solid on solid,
  sheet inside a wall, three bodies): levels +7.0 to +14.0 dB, the sheet and stack forms moving together (`d_T` stays
  1e-16 … 1e-13); the restated ones with new columns (`rule_dB`, `paid_dB`).
- *held lining* A, B, C, F: A +7.10 to +10.62 dB, `W | K` −17.83 → −9.99 dB (one plate with the mass at nothing passes its face, `a` = 0.1),
  C −15.68 → −6.98 dB, F the mass paid 43.52 → 33.53 dB; D: the reflect port of the held wave at the lining −7.36 dB → none
  (M2); E (restated): the depth of the state, `d_twin` up to 3.01 dB; H: K \| W −17.86 → −10.00 dB, W \| Kn from −45 to
  −53 dB → nothing from either side, W \| C and W \| A on their twins from the wall's side (scalar 0.0003 / 0.0007 dB; was
  0.63 dB at 10° and nothing at 75° for W \| C, up to 2.17 dB for W \| A).
- *closed plate* A (90 rows, +6.99 dB, the face 0.2 charged once), B (+3.57 to +4.18 dB), C (restated), D (+4.47 to
  +7.98 dB), F1 (the
  sheet event −2.45 → −1.41 dB at 30°, the step at 45° 3.20 → 1.64 dB); *closed face* F (+6.99 dB), G (0.325 → 0.5).
- *mass cosine* A (`rest`: the chain apart from the term, −14.79 → −0.67 dB, the faces no longer charged on top of `τ`),
  D (the leak chain, 5.2); *a sheet without the mass column inside 100 mm of a rigid mass-law wall* (`paid` 60.00 →
  54.77 dB, the rule with `a` = 0.3; `paid − TL` 5e-12).
- *stack coherent \| mass-law against the traced solid with a lining* (24): the level +9.4 to +14.5 dB, stack and
  laminate together (their difference unchanged); the laminate's echo through the lining crosses the plate twice,
  +18.7 to +29.1 dB (−92.27 → −64.09 dB below the reflection at 10°); *ordered stack with a mass-law layer* (32 / 40): `ref |T|` +0.10 (the
  helper on the rule), library on it to 1e-16; *the mass term in a stack* (restated); *mass-law reference at one layer*
  (|T| +0.067 dB at 0° … +1.18 dB at 70°, `d_T` 1e-16); *ordered stack whose second sheet carries m* (`gain_T`
  +0.0058, `d_T` 2e-16); *ray_mesh_interact's own glued entry* (+0.86 to +2.55 dB, own = redo exactly);
  *wood panel on concrete* (5.3).

**M2, nothing reflected back.** *a bounce returning to the glued sheet* (the way back 60.17 → 53.19 dB of mass,
recorded, it carries nothing), *a bounce inside the second body*, *ports of the wood \| concrete face from inside* (T 1,
R 0), *held lining* D, the path counts above.

**The held exit and A7.** *trajectory DENSE* (the segment 24.09° → 30.00°, held), *phase reference* (5.3), *refracting
tracer, a leak* (18 / 24: levels as above, state words unchanged), *closed-face hand-on S* (`bare`: the wall's exit behind
a lossy layer against the bare wall's exit 3e-6 → 1.39 dB porous, 0.54 dB `eps 1.50 − 0.40j`; the exit divides by the
absorption of the face the plate was entered by, which behind a lossy layer is not the bare face's `tf`; sheet against
traced solid unchanged, 3.1e-6), *step from the open to the closed far face* (4.67 → 0.0000 dB).

**Twins with every digit** (the same causes in complex form): *held lining G* (78), *closed plate E* (60), *rare sheet F*
(8 / 12), *sheet tf H* (12 / 24), *closed face H* (9 / 24).

**Rounding only:** *mass cosine B* (11 / 24, the sign of a printed zero), *stack geometry B* (2 / 32, the sign of a
printed zero), the residual columns of the stack tables (1e-17 … 1e-16).

**Restated tables** (header changed): *mass-law angle dependence*, *plate depth through ray_progress*, *plate depth* A
and C, *the mass term in a stack*, the five *bonded mass* tables with new columns, *an m = 0 lining on concrete*,
*closed plate C*.

### 5.5 What the change leaves (recorded in `md/acoustic.md` 7)

- **An event can pass more than one.** A plate's exit gives back the absorption of its first face (`a_in`), up to
  `1 / a_in` (+8.85 dB measured on wood \| concrete). The path never exceeds the rule, but `ray_progress` stops a
  continuation below `min_gain_dB` (default −140 dB) on the running gain, which inside a plate is up to
  `−10 log10 a_in` below the level the path will have (10 dB for `tf` 0.1, 20 dB for `tf` 0.01): a path that ends within
  that margin above the threshold can be cut. Not measured on a scene.
- The state one level deep (+3.02 / −3.01 dB on one slope, 3.01 dB in *Held lining* E), mixed slopes up to +7.66 dB.
- A non-rigid mass-law layer beside a lossy coherent layer depends on the order (0.107 / 0.127 / 0.201 dB).
- The lossy medium's flux split: the two sides of a lossy lining 0.07–1.38 dB (dense) and 0.52–2.70 dB (porous) apart,
  the wall's side higher.
- The closed step of a non-rigid mass-law layer at 45°: 1.64 dB.
