# Acoustic tools

Instruments of the acoustic add-on that outlive the scratch rigs (`build_linux/scratch` is removed by `make clean`).
The reference for what they are used for is `md/acoustic.md`; source keys in square brackets refer to the literature
review in `/home/stephan/accoustic_research` (`references/index.csv`).

| file | what it does | used for |
| --- | --- | --- |
| `sabine_to_random.py` | Reverberation-chamber (Sabine) absorption to random incidence by Jeong's method: one Miki flow resistivity per product, fitted so that a finite sample (Thomasson's field impedance of the rectangular patch) reproduces the chamber octaves; the output is the Paris integral (0°–90°) of the infinite layer. Also the edge correction `β·E` for mineral wool. | the A13 rule for a `tf` reference and the class-4 rows (`md/acoustic.md` 6, 9) |
| `band_probe.cpp`, `build_band_probe.sh`, `band_probe_example.in` | One ray through `ray_progress` with a band of frequencies, materials and plane faces given on stdin; T and R summed over every path that ends outside the structure. | checking that a band equals its single frequencies, and a construction against a transfer matrix (`md/acoustic.md` 2.8) |

## sabine_to_random.py

```
python3 tools/acoustic/sabine_to_random.py convert --thk 0.05 --sample 3 4 --alpha 0.153 0.587 0.993 1.073 1.017 0.987 --sigma 13112
python3 tools/acoustic/sabine_to_random.py check
```

- Octaves 125 Hz to 4 kHz, each the mean of its three third-octave centres. Extended reaction by default (`--local` for
  local reaction). `--sample a b` is the test sample in metres (ISO 354: 10 to 12 m², default 3 × 4 m).
- Use it only where the model fits the product and the sample is a flush patch. Where σ is measured, pass it with
  `--sigma`: a fitted σ off by more than 1.5 times is flagged, and the conversion is not to be used (polyester and a
  framed stone wool failed this, `md/acoustic.md` 6, A14).
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

The input format is in the head of `band_probe.cpp` (21 columns per material, the faces as `x material flipped`). The
example is the board on a box of air on a mass-law wall; its expected values are in the file. The fixture that guards
the same statement is *ray_progress - a leaf on a box of air at every frequency of a band*
(`tests/catch2_tests/test_acoustic_leaf.cpp`), which also holds the fluid-layer transfer matrix (`lw::stack`) and a
walker without recursion (`lw::walk`).
