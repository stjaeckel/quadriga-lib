# Acoustic tools

Instruments of the acoustic add-on that outlive the scratch rigs (`build_linux/scratch` is removed by `make clean`).
What they are used for is described in `md/acoustic_engine_notes.md` and on the material-model page `html_docu/quadriga_lib_material_model.md`; source keys in square brackets refer to the literature
review in `/sjc/accoustic_research` (`references/index.csv`).

| file | what it does | used for |
| --- | --- | --- |
| `sabine_to_random.py` | Reverberation-chamber (Sabine) absorption to random incidence by Jeong's method: one Miki flow resistivity per product, fitted so that a finite sample (Thomasson's field impedance of the rectangular patch) reproduces the chamber octaves; the output is the Paris integral (0°–90°) of the infinite layer. Also the edge correction `β·E` for mineral wool. | the rule for turning a chamber value into a `tf` reference (`md/acoustic_engine_notes.md` 5) |
| `check_docs.py` | Mechanical checks of the material-model page `html_docu/quadriga_lib_material_model.md`: tables, spelling, every column `MaterialCols` reads in its column table, no internal labels, `$` balanced per paragraph, in-page anchors resolve; and of the material part of the Data Formats page `tools/data_formats_md/format_materials.md`: tables, spelling, every column `MaterialCols` reads in its column table, no internal labels. Run from the repository root; exit code 1 on a failure. | the check of every change that touches the page or the Data Formats file |
| `band_probe.cpp`, `build_band_probe.sh`, `band_probe_example.in` | One ray through `ray_progress` with a band of frequencies, materials (with the peak columns `tfP`, `tfF`, `tfQ` and the panel size `side`) and plane faces given on stdin; T and R summed over every path that ends outside the structure. | checking that a band equals its single frequencies, and a construction against a transfer matrix (page 10.4) |
| `paris_cf.py` | The random-incidence (Paris, 0°–90°) absorption of a locally reacting face in closed form, and a peaked mass-law face as its series impedance with the pad of `tf` in front: normal-incidence, Paris and field (0°–78°) values per frequency; `check` against the numerical integral and Jeong's values. | the fit target of a peaked row (page 6.4) |
| `txlib.py` | Formula copy of what the library gives a closed mass-law sheet in air: the mass term (`m > 0` the mass law, 20 dB per decade; resonance and bending stiffness in `X₀`), the plate rule, `tf` and the interface gain, at column level; the finite panel of a material's `side` (Davy's forced radiation efficiency in place of `1/cos θ`, 3.29 m where the column is absent, 0 the infinite plate); switches for the candidates the library does not build (a fitted slope, no `att`, the installed loss factor in frequency, a limiting angle (closed), the bending loss as a resistance). Within 0.0042 dB of the library through the walker (traced solid; the sheet 0.0000 dB). | a formula reference for transmission fits and fixtures (page 5.4, 6.5) |
| `txfit.py`, `tl_refs.json` | The transmission fit of the expressibility test (`md/acoustic_engine_notes.md` 5, 8.4) on that formula, vectorized: the knobs of a mass-law material per construction, minimax over octaves, differential evolution and Nelder-Mead, selected at 768 angles × 12 frequencies per third; the 13 reference constructions (references, absorption laws, physical mass and coincidence) in `tl_refs.json`. Default: the library at random incidence (0–90°), each construction at its laboratory side (`side` `"lab"`; a number for one side, 0 for the infinite plate); `columns()` writes `side` and the rigid face's `a`. | the starting rows of page 11.4 (2.70 dB at the laboratory sides; the infinite plate 2.97 dB) and candidate studies |
| `gzlib.py` | Reference models of sound transmission through plates in air, independent of the library: thin and Mindlin plates, a layered transfer matrix (plates and air cavities) with a radiation efficiency on both outer faces, the infinite plate, the windowed finite plate in a baffle (`sigma_win`, Villot et al.), Davy's forced radiation efficiency and its diffuse mean; `python3 tools/acoustic/gzlib.py` runs its checks (the closed form, the 4\|16\|4 resonance at 211.3 Hz, σ against `1/cos θ` and against Leppington and Davy at grazing). No σ tables are stored: `sigma_win` is slow for large plates at high frequency. | the reference of the finite panel (page 5.4) and of the parameter work |
| `cases.py` | Per-case comparison of two `test_bin --durations yes` runs: printed rows that differ and failed assertions per case. | accounting for every moved row of a change |
| `rowdiff.py` | Number-by-number comparison of two plain `test_bin` outputs: the same lines in the same order, every number equal or within one unit of its last printed digit; every differing value printed with its row, the table header above it and the case that prints that header; exit code 1 on anything beyond. Run from the repository root (it reads `tests/catch2_tests` to name the cases). | the tolerant comparison of two runs (page 16) |

## sabine_to_random.py

```
python3 tools/acoustic/sabine_to_random.py convert --thk 0.05 --sample 3 4 --alpha 0.153 0.587 0.993 1.073 1.017 0.987 --sigma 13112
python3 tools/acoustic/sabine_to_random.py check
```

- Octaves 125 Hz to 4 kHz, each the mean of its three third-octave centres. Extended reaction by default (`--local` for
  local reaction). `--sample a b` is the test sample in metres (ISO 354: 10 to 12 m², default 3 × 4 m).
- Use it only where the model fits the product and the sample is a flush patch. Where σ is measured, pass it with
  `--sigma`: a fitted σ off by more than 1.5 times is flagged, and the conversion is not to be used (polyester and a
  framed stone wool failed this, `md/acoustic_engine_notes.md` 9.3).
- `check` reproduces the validation of 2026-10-15: glass wool and PU foam 50 mm against sound-field synthesis
  [Dupont_2023_SoundFieldSynthesis] (rms 0.027 / 0.039), rock wool 50 mm against the tube at 4, 8 and 12 m²
  [Kosten_1960_InternationalComparison] (one curve within 0.010), the two Scrosati products and the Knauf board (σ off).
- The field-impedance table of each sample size takes about 20 s to a few minutes on one thread and is cached in
  `$QD_ACOUSTIC_CACHE` (default `~/.cache/quadriga-lib/acoustic`). Needs numpy.

## band_probe

```
tools/acoustic/build_band_probe.sh /tmp/band_probe          # after cmake --build build_linux
/tmp/band_probe < tools/acoustic/band_probe_example.in
```

The input format is in the head of `band_probe.cpp` (24 columns per material: `coiA` removed on 2026-10-17; the peak `tfP`, `tfF`,
`tfQ` added 2026-10-16 for the peaked face; the panel size `side` last, added 2026-10-19; the faces as `x material flipped`). The
example is the board on a box of air on a mass-law wall; its expected values are in the file. The fixture that guards
the same statement is *ray_progress - a leaf on a box of air at every frequency of a band*
(`tests/catch2_tests/test_acoustic_leaf.cpp`), which also holds the fluid-layer transfer matrix (`lw::stack`) and a
walker without recursion (`lw::walk`).

## paris_cf.py

```
python3 tools/acoustic/paris_cf.py face --tfP 0.25 --tfF 150 --tfQ 0.75 --tf 0.05 --f 63 150 250 500
python3 tools/acoustic/paris_cf.py check
```

- `face`: the series impedance `ζ` of the columns (over-damped; `--tfQ` negative for the under-damped root), and per
  frequency `α₀`, the Paris value `tf + (1 − tf)·P(ζ)` and the field value (numerical). The example is the board of the
  peak cases: Paris 0.183 / 0.420 / 0.290 / 0.121 against `α₀` 0.125 / 0.288 / 0.195 / 0.089. `--tf` is one value, the
  power-law part at the frequencies given.
- `check`: the closed form within 4.9e-10 of the numerical integral on ten impedances; Jeong's maximum 0.951 at a
  normalized impedance of 1.567 and 0.17 / 0.49 / 0.79 at 40 / 10 / 4 [Jeong_2009_AngularDistributionCorrection;
  Jeong_2012_BoundaryConditionsPhasedGA]. No numpy needed.

## txlib.py, txfit.py

```
python3 tools/acoustic/txfit.py --seeds 6                       # the library at random incidence, all 13 constructions
python3 tools/acoustic/txfit.py --variant '{"amax": 78, "mfit": 1}' "glass 4 mm"   # the fitted slope at field incidence (before bc24b0c)
```

- A seed is one differential-evolution run (about 1 min on one core); `--procs` runs them in parallel. The minimax fit is
  multimodal: in the knob study of 2026-10-16 two independent halves of 3 to 9 seeds differed by up to 0.45 dB on single
  constructions (a variant's worst within 0.03 dB); its final fits used 18 seeds per half.
- Reference data: `tl_refs.json` (the first acoustic round, bb336dc: measured octaves 125 Hz–4 kHz, the spread of physical models elsewhere;
  `tf` at the class's absorption law). The knob bounds are those of the first acoustic round plus a level up to 10^4.5 (a 423 kg/m²
  wall needs 3200) and `att` up to 12 dB (`attmax` to change it).
- `txlib.sheet(M, f_hz, theta_rad, V)` takes the library's columns (GHz); `txfit.columns()` turns a fit into them.

## cases.py

```
python3 tools/acoustic/cases.py base_run.txt new_run.txt [rows]
```

Both runs are `./build_linux/test_bin --durations yes` outputs (whole suite or a tag). Rows are the lines a case prints
outside failure messages; a renamed case appears as new.
