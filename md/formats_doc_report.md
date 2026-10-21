# Report — the material part of the Data Formats page

Brief: `md/formats_doc_brief.md`. Section 0 is the baseline, section 1 the inventory of the old file, section 2 the
result and the stop before the commit. Scratch: `build_linux/scratch/fd0` (not kept; every fact the page relies on is
stated here).

## 0. Baseline

- Start: HEAD 32c55b2; `git status`: `M md/acoustic_manager.md`, `?? md/formats_doc_brief.md`, nothing under `src/`
  or `tests/`.
- `cmake --build build_linux --parallel 12`: up to date.
- Whole suite, `QD_DIAG=1 OMP_NUM_THREADS=12 ./build_linux/test_bin` from the repository root (started after another
  session's `test_bin` had finished): All tests passed (1344398 assertions in 1264 test cases), 29 s, load 4.31 / 4.50.
- `python3 tools/acoustic/check_docs.py`: all checks pass.

## 1. Inventory of the old `format_materials.md` (307 lines)

"Kept" = still in the file (possibly reworded), "changed" = in the file with a corrected statement, "removed" = gone.
"Model prose, now on the page" means the statement belongs to the model, which
`html_docu/quadriga_lib_material_model.md` describes (section given); the code line is that of the page's source.
Code lines (after the edits): `ofr` = `src/obj_file_read.cpp`, `rmi` = `src/ray_mesh_interact.cpp`, `ofw` = `src/obj_file_write.cpp`.

### 1.1 `SECTION_DESC`

| # | old statement | disposition | reason, code line |
| --- | --- | --- | --- |
| D1 | name-keyed table; a triangle references a row by `csv_ind` or `mtl_ind`, names in `csv_names` / `mtl_names` | changed | `mtl_ind` / `mtl_names` are the visual `.mtl` side, not the table (ofr 393–398, two material systems); the table side is `csv_ind`, `csv_names`, `csv_prop` |
| D2 | a value is read as `csv_prop["a"][csv_ind[face]]` | changed to `csv_prop["a"][csv_ind[i] - 1]` | `csv_ind` is 1-based, 0 = no material (ofr 648 `return it->second + 1`, 658 `return 0`) |
| D3 | model for EM (ε, μ, σ after P.2040), reused for an acoustic interpretation (mass law, coincidence, resonant absorption) | removed | model prose, now on the page (§1, §10); the new text links the page |
| D4 | schema-blind; only `name` required; every other column optional numeric; an absent column is defaulted by consumers; empty cells parse as 0 | kept (Reader rules, Columns) | ofr 113–121 (`name`), 124–133 (numeric columns), 162 (empty → 0); defaults rmi 432–457 (`Material`), 473–498 |
| D5 | three roles: interface reflection (`a`–`h`, `resF`, `resQ`, `resS`), interface transmission (`att`, `attB`, "front-side", "not applied on exit"), in-medium attenuation (`c`, `d`, `g`, `h`, `alpha`, `m`, "20 dB per decade from the normal thickness") | removed | model prose (page §5.2, §5.4); the `res*` under reflection for every material is point 18 |
| D6 | "Transmission partition" paragraph (light→dense subtracts `1 − abs(R)²`, dense→light passes through; types 1/4 gated, type 2 raw Fresnel) | removed | point 4: no such gate in the code (page §6.1) |
| D7 | both domains: EM εr > 1, μ = 1; acoustic two-parameter ε, μ — rigid reflectors εr ≪ 1, μ = 1, porous εr ≈ 1, μr > 1 | removed | model prose; the rigid one-parameter form is point 10 (page §10.1, §10.2) |
| D8 | related functions: `obj_file_read`, `ray_mesh_interact`, `calc_diffraction_gain`, `mitsuba_xml_file_write` | kept, `obj_file_write` and `ray_progress` added | `obj_file_write` writes the table (ofw 836–902); `ray_progress` takes it (`mtl_prop`, `src/ray_progress.cpp` 242) |

### 1.2 "1. Material Model"

| # | old statement | disposition | reason, code line |
| --- | --- | --- | --- |
| M1 | each parameter is one column and one key of `csv_prop`; absent → default; only `name` mandatory; column order free | kept (CSV form, Columns) | ofr 113–133; rmi 473–498 |
| M2 | parameter table of 20 columns | changed: 26 columns | point 1: `thk`, `tf`, `tfB`, `tfP`, `tfF`, `tfQ` were missing; names rmi 315–340 (`resolve("…")`), defaults rmi 432–457; units and meanings as page §4 |
| M2a | `g` unit "—" | changed to S/m | page §4; `g` enters μ as `c` enters ε, both through `−17.98·x/f` (rmi 521–522 against 507–508) |
| M2b | `resF`, `resQ`, `resS` "permittivity resonance" | changed: a pole of ε (wave media), the plate's resonance (mass-law); `resS` sign selects the form | point 2: the pole only where `m = 0` (rmi 511 `!(m > 0.0)`), the plate's resonance in `X₀` (page §5.4) |
| M2c | `m` "mass-law switch (> 0: the mass law, 20 dB/decade)" | kept as "any value > 0 makes the material a mass-law material", acts on: switch | rmi 445; page §4 |
| M2d | `side` row with typical panel sizes | changed: the meaning only | the sizes are model use (page §10.3, the 11.4 rows); default 3.29 rmi 457, 0 = infinite plate |
| M2e | `a` "εr at fRef", `e` "μr at fRef" | reworded ε′, μ′ | page §4 |
| M3 | "Frequency laws" table: ε(f) with the pole ungated, σ(f), μ(f), σμ(f), `att(f)` "clamped to ≥ 0", α(f), `mass(f, M) = max(0, 20·log10((f/fRef)·M))` "at normal incidence" | removed | model prose (page §5.1–5.4); the ungated pole is point 2, the mass row point 5 |
| M4 | `n = √(εμ)`, `Z = Z₀√(μ/ε)`, "matches the legacy ε-only model", Fresnel from the admittance, Snell, the loss tangent of the product | removed | model prose (page §5.1, §6.1); "legacy" is history |
| M5 | loss as negative imaginary part (P.2040-1, eq. 9b); total attenuation `8.686·L/Δ + α·L + mass(f, L)` | removed | model prose (page §5.1, §5.2, citing P.2040-3 eq. 9b); the `mass(f, L)` is point 5 |
| M6 | "Permittivity resonance … active only when `resF > 0` and `resS ≠ 0`"; models Helmholtz / membrane absorbers | removed | point 3 (also `resQ > 0`, rmi 511) and point 15 |
| M7 | "Panel size": Davy σ formula, default 3.29, "A bonded run … one `side`" | removed | model prose (page §5.4, rule 3 of §12) |
| M8 | "Coincidence": `X₀·abs(1 − (f/coiF)²·sin⁴θ·(1 + j/coiQ))`; "the former `coiA` Lorentzian was removed (2026-10-17); a `coiA` column … ignored" | removed | model prose (page §5.4); the history sentence is not public text; the page §4 says an old `coiA` is ignored |
| M9 | "Mass-law term": `X₀ = (f/fRef)·M`, angle shape, `fRef` from density | removed | model prose (page §5.4) |
| M10 | "Permeability": μ decouples impedance and index; ε ↔ compressibility, μ ↔ density | removed | model prose (page §5.1, §10.1) |

### 1.3 "2. Electromagnetic Interpretation" and its default table

| # | old statement | disposition | reason, code line |
| --- | --- | --- | --- |
| E1 | `a`…`d` define the complex permittivity, `ε'' = 17.98·σ/f` | removed | model prose (page §5.1) |
| E2 | `e`…`h` likewise; "the magneto-dielectric TM branch is out of scope; μ is intended for TE / scalar use" | removed | point 16 (μ is carried in TM, page §6.1) |
| E3 | Fresnel from `Z = Z₀√(μ/ε)`, normal-incidence `R` | removed | model prose (page §6.1) |
| E4 | three EM modes; type 1 "governed by the dense→light / light→dense partition gate"; type 2 "does not use the gate" | removed | point 4 (page §3, §6.1) |
| E5 | built-in EM materials have εr > 1, μ = 1, use `a`…`att`, `fRef` = 1 GHz | kept (section 2 intro) | ofr 24 header `name,a,b,c,d,att`; `fRef` absent → 1 (rmi 432) |
| E6 | "Values follow Rec. ITU-R P.2040-3, Table 3, valid for 1–40 GHz (ground 1–10 GHz)" | changed | 7 rows are not from Table 3; the ranges are per material and reach 100 GHz (section 2.4 below) |
| E7 | only `a`, `b`, `c`, `d`, `att`; a built-in scene has those keys in `csv_prop` | kept | ofr 24, `fill_csv_table_out` (ofr 474–489) copies the parsed columns |
| E8 | "max fGHz" column | changed to "Range (GHz)" with a "Source" column | section 2.4 |
| E9 | the 25 table rows (a, b, c, d, att) | kept, every value checked | identical to ofr 25–49; the 15 P.2040 rows identical to Table 3 (metal `c` 10⁷) |
| E10 | "`air` is row 0 of the table — the transparent fallback used for unmatched materials when `csv_strict = false`" | changed | `air` is row 1 (index 1); an unmatched name gets index 0, no material (ofr 658), which the ray functions keep at the defaults, i.e. air (rmi 468); `air` and `vacuum` are separate identical rows |
| E11 | `itu_vegetation` follows P.833-9, Figure 2 | kept, with the figure's range | P.833-9 (09/2016) §3: Fig. 2, specific attenuation of woodland, 30 MHz to about 30 GHz |
| E12 | `irr_glass` follows 3GPP TR 38.901 Table 7.4.3-1, the only non-zero `att` | changed: only its `att` is from the TR | Table 7.4.3-1: `L = 23 + 0.3 f` dB; the row has `att` 23, `attB` absent; `a`–`d` are not in the TR |

### 1.4 "3. Acoustic Interpretation" (lines 128–232) — removed as a whole

Model prose, now on the page (§5–§13); the default table is replaced by one sentence linking §11.4. Statement by
statement:

| old statement | point / page |
| --- | --- |
| "the mapping … does not model phase or diffraction" | point 7 (page §1, §10.1: every path carries the complex coefficient) |
| scalar types 3/4, "there is no scalar refraction mode" | point 6 (type 5, page §3) |
| type 4 shares the dense→light / light→dense gate | point 4 |
| `f_radio = f_acoustic × 874,636` (342.77 m/s) | point 9 (page §10.1: 343 m/s, 1 kHz = 0.874 GHz) |
| `fRef = 0.875 GHz ≡ 1 kHz`, fixed for every acoustic material | point 8 (page §5.4: a mass-law `fRef` carries the surface mass) |
| `resF`, `coiF` absolute radio GHz; octave grid 16–16000 Hz, results air-normalized | page §1, §10.1, §10.5 (31.5 Hz–8 kHz) |
| two families: rigid reflectors one-parameter (`εr ≪ 1`, `μ = 1`), isolation from mass law and `att` | point 10 (page §10.1, §10.2, rule 7) |
| porous absorbers `εr ≈ 1`, `μr > 1`, `att = alpha = m = 0` | page §10.2, §10.5 |
| parameter-roles table (incl. "mass-law transmission (rigid partitions) `m > 0`, `fRef`, `thk`") | page §10.3 |
| room-side absorption from ε, μ via `1 − abs(R)²` | point 10 (a rigid face's absorption is `tf`, page §6.3) |
| porous calibration `ε = n/z`, `μ = n·z`, mineral-wool example, band edges | page §10.5 and §11.3 (Miki / Komatsu fits) |
| rigid-surface table `a = ((1 − √(1−α))/(1 + √(1−α)))²` | point 10 |
| "the dense→light gate makes the air→wall crossing pass-through, so isolation comes from `att` (and `m` for the slope)" | point 11 (`m` as a slope; the plate rule, page §6.5) |
| layered materials: reflection from both materials, transmission gated; "wave cascade traced as separate ray segments" | points 4, 13 (page §7) |
| rigid on rigid: isolation the sum of each layer's `att` plus in-medium losses | point 12 (bonded mass-law bodies are one plate, page §5.4) |
| `att` once per entry, also at an internal interface; set `att` on one layer of a glued stack | page §5.2, rule 2 of §12 |
| glued faces coincident within 1 mm with opposing normals | page §2 |
| mounted porous absorbers: "a phaseless ray tracer cannot resolve it", quarter-wave over-prediction | point 14 (the quarter-wave "phaseless tracer" argument) |
| "clamp the per-interaction reflected power" | point 14 (nothing is clamped) |
| surface-impedance boundary `Zs = −j·(z/cosθt)·cot(k·d·cosθt)` | point 14 (not in the code) |
| wave-solver regime below Schroeder | page §13 |
| material classes (rigid, porous, empirical, resonant "`resF/resQ/resS` on top of a baseline reflector") | point 15 (on a mass-law face the peak is `tfP`, `tfF`, `tfQ`) |
| validity: LF floor "P.2040 nominal above 100 MHz ≡ 114 Hz", dispersion exponents, geometry, mass law angle | page §10.5, §13 |
| "Acoustic Default Materials": template table with `concrete` (`a` 2.4e-9, `m` 20), `porous_generic`, `_resonant_ (TBD)` | point 17 (no `tf`: absorbs nothing, passes nothing; `fRef` 1 GHz); replaced by the link to page §11.4 |

### 1.5 "4. Principled BSDF" and "5. Background and References"

| # | old statement | disposition | reason, code line |
| --- | --- | --- | --- |
| B1 | intro, the 17-field table, ranges, defaults, clamp to [0, 1] (ior not) | kept unchanged; a "Parameters:" sub-heading added | defaults ofr 234–247, clamp 345–363; the heading only stops the description line from running into the first sentence in the generated HTML |
| B2 | `Pm` "overrides `Ka` when present"; `Pr` "overrides `Ns` when present" | changed: "the later of `Ka` and `Pm` wins", "the later of `Ns` and `Pr` wins" | ofr 276–285 and 301–311: each keyword assigns the field as it is read |
| B3 | `Ka` → metallic, note "—" | changed: "First component only" | ofr 278–279 read one value |
| R1 | P.2040, P.833, 3GPP TR 38.901 with their roles | kept, titles and years from the documents | P.2040-3 (08/2023), P.833-9 (09/2016), TR 38.901 V17.0.0 (2022-03); 3GPP named as the source of the `att` of `irr_glass` only |
| R2 | 13 Wikipedia links on the physics | removed | the page has the references (§14) |
| R3 | — | added: the page link | the description of the model |

### 1.6 The 18 points of the task-3 report, by name

| point | where it was in the old file | now |
| --- | --- | --- |
| 1 missing 6 columns | M2 | the 26-column table |
| 2 `res*` "permittivity resonance", pole ungated | M2b, M3 | per-kind meaning in the table; formulas removed |
| 3 pole needs `resQ > 0` | M6 | removed |
| 4 transmission gate | D6, E4, 1.4 | removed |
| 5 `mass(f, L)` from the path; `20·log10 X₀` only at `side` 0 | D5, M3, M5 | removed |
| 6 "no scalar refraction" | 1.4 | removed |
| 7 "does not model phase" | 1.4 | removed |
| 8 `fRef` 0.875 GHz fixed | 1.4 | removed |
| 9 874,636 / 342.77 m/s | 1.4 | removed |
| 10 one-parameter rigid mapping, absorption from `a` | D7, 1.4 | removed |
| 11 `att` and `m` for the slope | 1.4 | removed |
| 12 layered rigid on rigid as a sum | 1.4 | removed |
| 13 cascade traced as separate segments | 1.4 | removed |
| 14 phaseless tracer, clamp, surface-impedance boundary | 1.4 | removed |
| 15 resonant absorbers by `res*` | M6, 1.4 | removed |
| 16 TM branch out of scope | E2 | removed |
| 17 acoustic `concrete` row without `tf` | 1.4 | removed (link to page §11.4) |
| 18 `res*` under reflection, `coi…` under the partition | D5, D6 | removed |

### 1.7 New statements, with the code they rest on

| statement | code |
| --- | --- |
| header line first; comma-separated, cells trimmed, no quoting | ofr 100–110, 60–81 |
| UTF-8 BOM skipped; blank lines ignored | ofr 105–107; 142 |
| `name` required at any position, compared case-sensitively; empty header name an error | ofr 113–121; 129–130 |
| name non-empty and unique | ofr 150–154 |
| empty or missing trailing cell → 0, not the default; non-numeric → error | ofr 145–147, 162, 165–173 |
| a 0 in `fRef`, `a`, `e` is rejected; 0 in `side` is the infinite plate | rmi 359–361 (`require(…, true)`), 373, page §4 |
| at least one material; a material file replaces the built-in table | ofr 180–181; 189–199 |
| unknown columns returned and ignored | ofr 474–489 copies every column; `MaterialCols` resolves only its 26 (rmi 315–340) |
| `obj_file_read` checks no values; the functions that take the table check them (lengths, signs, 1-based material in the message) | ofr 95–184 (no range checks); rmi 290–373 |
| `ray_progress` at most 16383 materials | `src/ray_progress.cpp` 364–365 |
| `obj_file_write` writes the table as a `.csv` named after the `.obj` | ofw 836–902 (878) |
| lookup: exact name, then base name before the first dot; `csv_strict` error, else index 0 | ofr 644–659, 87–91 |
| index 0 treated as air | rmi 468 |
| empty `.obj` name: only the table | ofr 515–525 |
| built-in table: only `a`–`d`, `att`; every material a traced solid (no `thk`) | ofr 24; rmi 441 (`thk` 0) |

Where the page and the code differ: nothing in the columns, units, defaults or checks (the page §4 was written from the
same lines). The only difference found is the frequency range of the built-in table (section 2.4).

## 2. Result — stop before the commit

### 2.1 Line counts

| file | before | after |
| --- | ---: | ---: |
| `tools/data_formats_md/format_materials.md` | 307 | 179 |
| generated material part of `formats.html` (scratch build, from the section anchor to the next section) | — | 754 HTML lines |

### 2.2 Diff scope (`git diff --numstat`, without `md/acoustic_manager.md`, which was modified before and not touched)

| file | + | − | what |
| --- | ---: | ---: | --- |
| `tools/data_formats_md/format_materials.md` | 117 | 246 | rewritten (section 1 of this report) |
| `src/obj_file_read.cpp` | 6 | 5 | fixes 1 and 2, inside the `obj_file_read` MD block |
| `api_python/qpy_obj_file_read.cpp` | 2 | 2 | fix 2, inside the MD block |
| `api_mex/obj_file_read.cpp` | 2 | 1 | fix 2, inside the MD block |
| `html_docu/quadriga_lib_material_model.md` | 1 | 1 | fix 3, section 9 |
| `tools/acoustic/check_docs.py` | 32 | 6 | check 7 (the formats file) |
| `tools/acoustic/README.md` | 1 | 1 | the `check_docs.py` row |

- Every changed line under `src/` and `api_*` lies inside a `/*!MD … MD!*/` block (checked hunk by hunk on the old and
  new line numbers of `git diff -U0`). Nothing under `tests/` or `include/` changed.
- The three neighboring fixes:
  1. C++ block: "by name with a trailing Blender `.NNN` suffix removed" → "by the base name (everything before the first
     dot, so Blender sub-materials like `concrete.gray` map to `concrete`)", the wording of the MATLAB and Python blocks;
     code ofr 87–91, 644–659.
  2. The three blocks: "For a detailed description of the material model see Data Formats" → "The material model and
     its columns: [the material-model page]; the CSV format of the material table: [Data Formats]" (the page link as in
     the `ray_mesh_interact` blocks, relative `quadriga_lib_material_model.md`).
  3. Page §9: "the built-in table of Rec. ITU-R P.2040-3, Table 3 [1] (valid 1–40 GHz, ground 1–10 GHz)" → the
     P.2040-3 rows named, their ranges indicative and up to 100 GHz for most materials (brick and plywood 40, marble 60,
     floorboard 50–100 GHz), the ground types limited to 1–10 GHz, and a link to Data Formats for every row's range and
     source.
- `formats.html` built into scratch with the brief's command (`-o build_linux/scratch/fd0/formats.html`) and read: no
  raw Markdown (no backtick, `**`, `#` line, `$`, stray `|`), four tables with constant cell counts (27 × 5 columns,
  26 × 8, 18 × 5, 16 × 3), the code block as `<pre>`, all lists closed; links `cpp_api.html#…` (6 targets),
  `quadriga_lib_material_model.md` and `…#114-starting-rows-for-rigid-constructions` (the slug of the page's heading
  "11.4 Starting rows for rigid constructions"), the two `itu.int` recommendation links. The section's overview table
  lists the four entries in order. `html_docu/formats.html` itself is not regenerated (`make documentation`).
- `tools/extract_md.py -a cpp`, `-a mex`, `-a python` into scratch: all three run; each `obj_file_read` entry carries
  the new sentences once.

### 2.3 Counts

- Build after the edits: `obj_file_read.cpp.o` recompiled, no warning.
- Whole suite, `QD_DIAG=1`, 12 threads: All tests passed (1344398 assertions in 1264 test cases), 30 s, load 1.70 /
  2.00; the output equals the baseline output line by line apart from the random seed. (`[wave]` and `[acoustic]` are
  part of it; their baselines were not rerun separately for a comment-only change.)
- `python3 tools/acoustic/check_docs.py`: all checks pass, including the new lines `tables formats ok`, `spelling
  formats ok`, `columns formats 26 of 26`, `labels formats none`. Negative check: the old file reports `thk`, `tf`,
  `tfB`, `tfP`, `tfF`, `tfQ` missing; a renamed `side` row, an item id with an `md/` path and "colour" are caught.

### 2.4 The P.2040-3 finding

Source: Rec. ITU-R P.2040-3 (08/2023), `references/itu_rec_p2040_3.pdf` in the repository, Table 3 "Material
properties", p. 24–25, and the paragraph below it.

| Table 3 material | a | b | c | d | range (GHz) |
| --- | ---: | ---: | ---: | ---: | --- |
| Vacuum (≈ air) | 1 | 0 | 0 | 0 | 0.001–100 |
| Concrete | 5.24 | 0 | 0.0462 | 0.7822 | 1–100 |
| Brick | 3.91 | 0 | 0.0238 | 0.16 | 1–40 |
| Plasterboard | 2.73 | 0 | 0.0085 | 0.9395 | 1–100 |
| Wood | 1.99 | 0 | 0.0047 | 1.0718 | 0.001–100 |
| Glass | 6.31 | 0 | 0.0036 | 1.3394 | 0.1–100 (a second row 5.79 / 0.0004 / 1.658 for 220–450, not built in) |
| Ceiling board | 1.48 | 0 | 0.0011 | 1.0750 | 1–100 (a second row for 220–450, not built in) |
| Chipboard | 2.58 | 0 | 0.0217 | 0.7800 | 1–100 |
| Plywood | 2.71 | 0 | 0.33 | 0 | 1–40 |
| Marble | 7.074 | 0 | 0.0055 | 0.9262 | 1–60 |
| Floorboard | 3.66 | 0 | 0.0044 | 1.3515 | 50–100 |
| Metal | 1 | 0 | 10⁷ | 0 | 1–100 |
| Very dry ground | 3 | 0 | 0.00015 | 2.52 | 1–10 only |
| Medium dry ground | 15 | −0.1 | 0.035 | 1.63 | 1–10 only |
| Wet ground | 30 | −0.4 | 0.15 | 1.30 | 1–10 only |

The text below the table: the ranges are not hard limits but indicate the measurements behind the models, except
that the 1–10 GHz limits of the three ground types must not be exceeded.

- **The page's "valid 1–40 GHz" is wrong**: most materials reach 100 GHz. Its "ground 1–10 GHz" is right. Corrected
  (fix 3).
- **The formats table's "max fGHz" holds** for every P.2040-3 row (100, except brick 40, plywood 40, marble 60, ground
  10). It lacked the lower limits, of which floorboard (50 GHz) matters; the new "Range" column gives both.
- The built-in coefficients of the 15 P.2040-3 rows (`air`, `vacuum` and 13 `itu_` rows) equal Table 3.
- Not in Table 3: `textiles`, `plastic`, `ceramic`, `sea_water`, `sea_ice`, `water`, `water_ice` (library estimates,
  ranges 1–100 and 1–20 GHz kept from the old table and marked "estimate"), `itu_vegetation` (P.833-9, Fig. 2: 30 MHz
  to about 30 GHz; the old 100 GHz changed to 0.03–30), `irr_glass` (only `att` from TR 38.901 Table 7.4.3-1, whose
  scope is 0.5–100 GHz; the old 100 changed to 0.5–100).

### 2.5 What went where

| content of the old file | now |
| --- | --- |
| table concept, `csv_prop` map, related functions | `SECTION_DESC` (corrected index, page link) |
| column table (20 rows) | "1. Material Table", Columns: 26 rows with unit, default, meaning, acts on (`m > 0` as the switch) |
| reader facts (only `name`, empty → 0) | "1. Material Table", CSV form, Reader rules, Material lookup (extended from ofr) |
| frequency laws, mechanisms, EM and acoustic interpretation, calibration, limits | removed; the page link (§1–§13 of the page) |
| EM default table and notes | "2. Built-in EM Materials" (range and source per row; `air` index corrected) |
| Acoustic Default Materials | removed; one sentence linking page §11.4 |
| Principled BSDF | "3. Principled BSDF" (two notes corrected, one added, a sub-heading) |
| references and Wikipedia links | "4. References": the page, P.2040-3, P.833-9, TR 38.901 |

### 2.6 Side findings (one line each, no items)

- Five other API blocks still say "For a detailed description of the material model see Data Formats":
  `api_mex/ray_mesh_interact.cpp` 29, `api_mex/calc_diffraction_gain.cpp` 23, `api_python/qpy_calc_diffraction_gain.cpp`
  22–23, `api_python/qpy_obj_file_write.cpp` 31, `api_python/qpy_mitsuba_xml_file_write.cpp` 24 (outside the named
  sentences, not changed).
- `include/quadriga_tools.hpp` 111 calls `calc_diffraction_gain`'s `mtl_ind` "0-based"; its API block (68) and code
  (403–404) are 1-based.
- The `obj_file_write` block (ofw 326) gives the written defaults as "`a`, `e`, `fRef` = 1, else 0"; the code writes
  `side` 3.29 (848), and `thk` is not in the canonical order (839–842), so a present `thk` is written among the extra
  columns.
- `std::stod` (ofr 167) reads a leading number: `1.5x` reads as 1.5, `nan` and `inf` are accepted.
- The `.mtl` reader compares `newmtl` names raw and keywords at column 0 (ofr 255, 276–338): a CRLF `.mtl` file or
  indented keywords give default BSDF values.
- Page §9 after the table says `irr_glass` follows 3GPP TR 38.901; only its `att` does (not changed, outside fix 3).
- The seven estimate rows come from commit 75f43b9 (`references/chat_gpt_prompt.txt`), not from a publication.

### 2.7 Stop

Everything is uncommitted; `md/acoustic_manager.md` is untouched by this task. For the commit: the seven files of 2.2
and this report. The generated pages (`formats.html`, `cpp_api.html`, `mex_api.html`, `python_api.html`, the
`quadriga_lib_api_*.md`) are regenerated by `make documentation`.
