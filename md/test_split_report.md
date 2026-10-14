# Test split — report

Findings of the test split described in `md/test_split_brief.md`, one section per phase. The
commands to start a session with are under *First step for future sessions*. Every number
here comes from a run or a search in `build_linux/scratch/ts1`; section 0.2 describes how each was measured.

Two appendices sit next to this file:

- `md/test_split_inventory.csv` — one row per TEST_CASE of the 14 files (367 cases).
- `md/test_split_tables.csv` — one row per printed table of the three ray files (297 tables).

---

## Phase 0 — inventory and analysis (2026-10-14)

No file under `tests/` or `src/` was edited. Scratch work is in `build_linux/scratch/ts1`.

### 0.1 Baseline check, and where the brief's facts differ

`cmake --build build_linux --parallel` had nothing to rebuild. `./build_linux/test_bin > test.txt` ended
with `All tests passed (1297934 assertions in 1074 test cases)` and 19,870 lines.

| brief says | measured |
| --- | --- |
| about 3 min 13 s wall | **37.3 s** wall, 625 MB peak, on an idle machine (`/usr/bin/time -v`). |
| `test.txt` is byte-identical to the committed one; `git status` shows it unchanged | **`test.txt` is in `.gitignore`** (`.gitignore:26`) and not tracked, so `git status` cannot show a change. I compared it with `build_linux/scratch/i47/id.txt`, the plain-copy baseline of the last session (08:54 today): identical except line 1, the random seed. |
| `QD_TEST_VERBOSE` is defined in `test_sheet_vs_solid.cpp` | **It is commented out** there (`//#define QD_TEST_VERBOSE`, line 39), and the file prints only with the environment variable `QD_TEST_VERBOSE=1`. None of its tables are in `test.txt`. Its line 22 still says "currently forced on". |
| case names must not contain a comma | 39 existing case names in these files contain one, for example *Lossless slab, sheet vs solid*. To filter on them, the test spec must escape it as `\,`. |
| compile 55 s / 1.1 GB, 16 s, 9 s | 56.1 s / 1.10 GB, 16.2 s / 0.69 GB, 9.3 s / 0.53 GB (alone, test flags). |

### 0.2 How it was measured (the rig in `build_linux/scratch/ts1`)

- **Marker build.** `mk.py` writes `rmi_mk.cpp`, a copy of `src/ray_mesh_interact.cpp` with a counter in
  `Material(const MaterialCols&, idx)`, the constructor every evaluation in that file goes through, the
  public `medium_gain`, `interface_gain` and `refractive_index` included. It counts `m > 0` (split into
  rigid, `a < 1e-6`), and `m = 0` with a nonzero index. It also counts the entry calls of
  `ray_mesh_interact` and `ray_state_update` per interaction type, and calls of the three public material
  functions.
  - The copy is compiled with the library flags and linked in front of `libquadriga.a`, together with
    `listener.cpp` (`testCaseEnded` → `qdmk_dump`). `build.sh` is the i47 template with the folder changed.
  - With `QD_MARK` unset the marked binary reproduces `test.txt` row for row; this was checked again
    after every rebuild.
  - `QD_MARK=1` dumps per-case counts. `QD_MARK=2` also writes one line `@MASS` in front of the next
    output line after an `m > 0` material was built, which attributes evaluations to table rows.
    `QD_MARK=3` also writes `@CALL` after any counted entry call; this is how tables that only re-print
    earlier values are found. Modes 1 and 2 gave identical results before and after mode 3 was added.
- **Assertions per case.** Each case of the 14 files was run alone (`percase.py`) and its count read from
  the summary line.
- **Assertions per table.** `listener2.cpp`, with `shouldReportAllAssertions`, prints `@A` for every
  assertion in sequence with the output (`alattr.py`). Its total, 196,271, equals the three ray files' sum.
- **Structure.** `struct.py` finds each case's line range by brace matching (comments and strings
  blanked), and each file-scope helper. It also lists the library functions a case reaches directly or
  through file-scope helpers. This is a static approximation; for functions in `ray_mesh_interact.cpp`
  the marker counts are used instead.
- **Run time.** Catch2 `--durations yes` (XML reporter).
- **Output lines.** Lines between a case's duration line and the one before it.
- **Tables.** `tables.py` / `tables3.py` split output into blank-separated blocks. `diag.py` maps each
  block's `# diag` header to the format string that prints it. `tabcat.py` writes the catalogue.
- **Plan citations.** `diag2.py` searches the plan for the case name, for the header's first four
  words, and for the table letter in a "Tables: A …; B …" list. This is a text search and can miss a
  paraphrase.
- **Copies.** `lambdas.py` extracts every lambda of 8 or more lines inside a case and clusters copies
  with a difflib ratio of at least 0.8.
- **Split prototype.** `gen_split3.py` generates the proposed phase-1 layout into `split/`, `cc.sh`
  compiles one translation unit with the test flags, and `cmpcase.py` compares two runs' output keyed by
  case.

### 0.3 Inventory

All 1074 cases pass. The 14 files in the brief's list, plus `test_point_inside_mesh.cpp`, which reaches
`ray_mesh_interact` and was not on the list, hold 367 cases. The full per-case table is
`md/test_split_inventory.csv`, with columns for:

- line range, lines and assertions
- run time and output lines in `test.txt`
- `m > 0` constructions (and rigid), `m = 0` constructions
- whether scalar types or EM types are used, and whether a porous material is named in the case text
- tables, and tables that evaluate `m > 0`
- the library functions reached (`rmi·345` = `ray_mesh_interact` with types 3, 4 and 5)
- class and proposed target file

Appendix A below has the same data for the five ray files.

| file | cases | reach ray_mesh_interact.cpp | lines in cases | assertions | run s | output lines | wave | acoustic | mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_ray_state_update.cpp` | 102 | 100 | 17929 | 120030 | 12.24 | 15562 | 72 | 13 | 17 |
| `test_ray_progress.cpp` | 54 | 44 | 4853 | 72845 | 13.34 | 3457 | 51 | 2 | 1 |
| `test_ray_mesh_interact.cpp` | 33 | 33 | 2673 | 3396 | 0.02 | 396 | 32 | 1 | 0 |
| `test_sheet_vs_solid.cpp` | 15 | 13 | 640 | 215 | 0.00 | 0 | 14 | 1 | 0 |
| `test_acoustic_interactions.cpp` | 11 | 10 | 465 | 82 | 0.11 | 0 | 10 | 1 | 0 |
| `test_gains.cpp` | 5 | 5 | 391 | 121 | 0.15 | 0 | 5 | 0 | 0 |
| `test_calc_diffraction_gain.cpp` | 24 | 23 | 855 | 132 | 0.04 | 0 | 24 | 0 | 0 |
| `test_ray_commit.cpp` | 27 | 2 | 758 | 810644 | 0.12 | 0 | 27 | 0 | 0 |
| `test_ray_subdivide_flag.cpp` | 11 | 2 | 417 | 644 | 0.03 | 0 | 11 | 0 | 0 |
| `test_get_channels_multifreq.cpp` | 36 | 0 | 1057 | 179 | 0.07 | 0 | 36 | 0 | 0 |
| `test_ray_init.cpp` | 15 | 0 | 453 | 936 | 0.01 | 0 | 15 | 0 | 0 |
| `test_obj_file_write.cpp` | 12 | 0 | 886 | 172 | 0.25 | 0 | 12 | 0 | 0 |
| `test_baseband_freq_response_multi.cpp` | 21 | 0 | 846 | 477 | 0.00 | 0 | 21 | 0 | 0 |
| `test_point_inside_mesh.cpp` | 1 | 1 | 43 | 15 | 0.06 | 0 | 1 | 0 | 0 |

Notes on the inventory:

- `test_get_channels_multifreq.cpp`, `test_ray_init.cpp`, `test_obj_file_write.cpp` and
  `test_baseband_freq_response_multi.cpp` reach nothing in `ray_mesh_interact.cpp`. `test_obj_file_write`
  only writes the column names, `m` and `coi*` among them, as text. They are not ray tests and stay where
  they are.
- `test_ray_commit.cpp` (2 cases) and `test_calc_diffraction_gain.cpp` reach the public `medium_gain`
  internally, never with `m > 0`.
- *Mesh builder conventions* (`test_sheet_vs_solid.cpp:453`) has **no assertion** when the variable is
  unset (`assertions: - none -`). Four more cases there end in `CHECK(true)`.
- **Run time is concentrated.** Two wave cases take 17.8 of the suite's 37 s:
  - *Measurement - the mirror behind the ray after a back reflection* (`test_ray_state_update.cpp:12904`), 8.80 s
  - *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer* (`test_ray_progress.cpp:3565`), 8.95 s

  Both already hold the one-thread guard.

### 0.4 Classification by measurement

A case is **wave** when no `Material` with `m > 0` is built while it runs, and **acoustic** when one is.
Within acoustic cases, a case with at least one table whose rows evaluate only `m = 0` materials is
**mixed**. Three tables evaluate nothing at all; each takes the class of the table it summarises:

- *closed-face hand-on B* and *closed-face hand-on S* re-print rows of A, C, D and E: mixed.
- *plate depth S* re-prints rows of A, which has `m > 0` in 768 of its 773 rows: acoustic.

Result over the 14 files: **331 wave, 18 acoustic, 18 mixed.**

| class | cases | lines | assertions | run s | output lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave | 331 | 22,592 | 909,590 | 23.39 | 8,523 |
| acoustic | 18 | 3,313 | 10,049 | 0.92 | 3,616 |
| mixed | 18 | 6,361 | 90,249 | 2.13 | 7,276 |

In the three large files: state update 72 / 13 / 17, progress 51 / 2 / 1, mesh interact 32 / 1 / 0.

**Scalar types on wave media.** 98 wave cases use scalar types 3/4/5, all on materials without the mass
column. Most use plain dielectrics in scalar mode. Five name a porous material or absorber in their text,
for example *Scalar closed face - a lossy box on a rigid wall* (porous rows of `data/materials_ac.csv`)
and *ray_progress - a lossy medium has no critical angle*. All are wave cases, as the brief says.

**Rigid permittivity without the mass column.** Many rows labelled acoustic in their headers sit on the
wave side by the rule, for example *bare tf E, the acoustic main road: a rigid wall 200 mm, eps 2.401e-9,
with tf*. In that table, the 24 rows with `m 30` evaluate the mass column and the 24 without do not.

**Acoustic cases (18)**

| case | lines | assertions | m > 0 built | tables with m > 0 / all |
| --- | ---: | ---: | ---: | --- |
| `acoustic_interactions:161` Acoustic - medium_gain in-medium loss | 52 | 6 | 4 | 0 / 0 |
| `ray_mesh_interact:3173` Thin sheet - mass-law angle dependence | 143 | 164 | 42 | 1 / 1 |
| `ray_progress:4018` ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band | 180 | 1080 | 21862 | 1 / 1 |
| `ray_progress:4199` ray_progress - the bonded plate at any depth where the slopes agree | 232 | 1924 | 4446072 | 1 / 1 |
| `ray_state_update:1095` ray_state_update - mass-law material is excluded from slab resolution | 38 | 3 | 8 | 0 / 0 |
| `ray_state_update:6385` Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space | 122 | 241 | 8 | 1 / 1 |
| `ray_state_update:6966` Stack with a mass-law layer - resolved by segments, not killed | 153 | 284 | 280 | 3 / 3 |
| `ray_state_update:7556` Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids | 142 | 34 | 57 | 2 / 2 |
| `ray_state_update:7958` Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate | 396 | 220 | 1204 | 10 / 10 |
| `ray_state_update:8355` Ports of a mass-law face from inside - the reflect port is the other direction's | 71 | 72 | 72 | 1 / 1 |
| `ray_state_update:9731` Bonded mass - a mass-law sheet inside a mass-law wall | 98 | 72 | 768 | 1 / 1 |
| `ray_state_update:14624` Leak direction - a closed face hands the wall behind a traced layer the direction the wave has in air | 451 | 1639 | 3937 | 6 / 6 |
| `ray_state_update:15076` Bonded mass - the plate at any depth where the slopes agree | 505 | 1736 | 66824 | 4 / 4 |
| `ray_state_update:16099` Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged | 65 | 88 | 518 | 1 / 1 |
| `ray_state_update:16376` Unentered branches - a stack of a coherent and a mass-law layer against the laminate meshed as a solid with a lining | 59 | 120 | 408 | 1 / 1 |
| `ray_state_update:16670` Mass cosine - a mass-law sheet behind a refracting layer forms its term at the cosine the wave has in air | 229 | 1054 | 3796 | 5 / 5 |
| `ray_state_update:17224` Held lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid | 332 | 1292 | 23319 | 8 / 8 |
| `sheet_vs_solid:1082` Acoustic sheet, mass-law route | 45 | 20 | 98 | 0 / 0 |

**Mixed cases (18), and how each would split.** "Whole tables" means the `m > 0` evaluations sit in
tables of their own. "Row families" means some tables hold both kinds of rows, usually one family of a
loop: a palette entry, a stack, or the mass-law family of a twin table.

| case | lines | assertions | wave tables (lines) | acoustic tables (lines) | tables with both: m > 0 rows / rows | split by |
| --- | ---: | ---: | --- | --- | --- | --- |
| `ray_progress:2416` ray_progress - refracting tracer: the segment out of a lossy medium is the  | 386 | 827 | 8 (283) | 1 (5) | - | whole tables |
| `ray_state_update:6132` Lumped columns - an ordered stack: what the core carries, by matrix and by  | 252 | 314 | 2 (41) | 0 (0) | ordered stack whose second she 4/8 | row families |
| `ray_state_update:7183` Total reflection of the outer pair - the sheet stays in the result | 258 | 97 | 3 (34) | 1 (9) | - | whole tables |
| `ray_state_update:7699` Leak chain - each face at the cosine the wave has and the series on a close | 227 | 48 | 5 (42) | 0 (0) | step from the open to the clos 1/3 | row families |
| `ray_state_update:10124` Direction - a ray leaves a parallel slab parallel to how it arrived | 294 | 386 | 3 (112) | 1 (15) | - | whole tables |
| `ray_state_update:10419` Refracting tracer - a traced bend is known from the state and the leg is ex | 510 | 4579 | 5 (287) | 0 (0) | refracting tracer 5/24 | row families |
| `ray_state_update:10930` Phase reference - a traversal carries the phase at the point the polyline l | 455 | 1454 | 6 (388) | 1 (29) | - | whole tables |
| `ray_state_update:11870` Leak direction - a wave held in a mass-law body keeps its direction and a c | 309 | 2916 | 1 (78) | 1 (296) | - | whole tables |
| `ray_state_update:12180` Measurement - the mirror behind a second solid and behind a sheet inside a  | 432 | 690 | 4 (166) | 1 (21) | - | whole tables |
| `ray_state_update:13948` Lossy verdict - a lossy second medium has no critical angle | 675 | 1758 | 12 (346) | 1 (21) | lossy verdict J 16/32 | row families |
| `ray_state_update:16231` Unentered branches - ray_mesh_interact's own glued composite against the re | 144 | 264 | 2 (24) | 0 (0) | ray_mesh_interact's own glued  25/75 | row families |
| `ray_state_update:16900` Closed plate - a mass-law layer the wave is evanescent in passes its two le | 323 | 1742 | 1 (48) | 7 (338) | - | whole tables |
| `ray_state_update:17557` Stack geometry - an ordered stack leaves the ray on the line it arrived on | 222 | 2776 | 2 (127) | 0 (0) | stack geometry A 144/336; stack geometry B 32/96 | row families |
| `ray_state_update:17780` Rare sheet - a coherent sheet with tf that is rarer than the invariant pass | 323 | 1672 | 7 (576) | 0 (0) | rare sheet F 24/152 | row families |
| `ray_state_update:18104` Flat branch - a coherent sheet rarer than the invariant without tf tunnels  | 418 | 1614 | 6 (344) | 1 (22) | flat branch H 12/204 | row families |
| `ray_state_update:18523` Sheet tf - the face of a coherent sheet with tf creates nothing and has no  | 513 | 36846 | 5 (715) | 1 (66) | sheet tf D 6/36; sheet tf H 24/292 | row families |
| `ray_state_update:19086` Bare tf - a bare face with tf is a transformer where it is open and a flat  | 235 | 6991 | 8 (609) | 0 (0) | bare tf E 24/48 | row families |
| `ray_state_update:19495` Closed face - tf at a closed face is a pad and the two ports of a sheet on  | 385 | 25275 | 1 (12) | 4 (102) | closed face A 64/312; closed face E 192/288; closed face H 12/36 | row families |

Two of these become wave cases once a recorded table goes:

- In `ray_progress:2416` the only acoustic table is *trajectory, recorded. DENSE, eps 1.50 with m 20*,
  5 lines.
- In `ray_state_update:16900` the only wave table is the recorded *closed plate F3*; without it the case
  is acoustic.

A split cannot happen in phase 1, whose acceptance keeps the 1074 cases. Every split adds a case, and
each acoustic half needs the case's palette and lambdas, which today are case-local.

### 0.5 Overlap

**`test_acoustic_interactions.cpp` against the newer cases**

| case (line) | assertions | class | the same statement asserted elsewhere | proposal |
| --- | ---: | --- | --- | --- |
| frequency mapping round-trip (99) | 16 | — | none needed: it calls no library function and checks only the file's own `ac2rf` / `rf2ac` constants | delete; it guards no library code |
| refractive_index carries the resonance, not the coincidence (118) | 5 | wave | air and `n = √a`: *Public helpers* (`test_ray_mesh_interact.cpp:1931`). The pole: *Resonance - the public index…* (`:3317`), at 1e-12. Not asserted elsewhere: dispersion via `b`, and a coincidence column leaving the index unchanged | move the two unique checks into *Public helpers*, delete the rest |
| medium_gain in-medium loss (161) | 6 | acoustic | `alpha` and `att`-not-in-medium-gain: *Public helpers*, *Alpha in-medium absorption (medium_gain)* (`:1225`). The mass law through the public `medium_gain`: **only here** | keep in acoustic |
| interface_gain penetration and coincidence (217) | 7 | wave | `att`/`attB`: *Public helpers*. The coincidence Lorentzian, the dip and the clamp: **only here** | keep, wave |
| permittivity resonance absorption peak (286) | 3 | wave | `R_on < R_off` follows from the reflect port with the pole, asserted against the reference to 1e-9 in *Lumped columns - a permittivity resonance is part of the medium* (`test_ray_state_update.cpp:5953`) and *Resonant medium* (`:20183`) | delete |
| convergence with acoustic terms absent (322) | 5 | wave | no other bitwise twin of all-zero acoustic columns through `ray_mesh_interact` and `calc_diffraction_gain` found | keep, wave |
| convergence away from spectral features (361) | 2 | wave | follows from the formula-level checks of 2 and 4 and the resonance cases | delete, or keep (28 lines) |
| scalar total internal reflection (394) | 12 | wave | *Scalar interactions (types 3/4/5)* (`:1638`) covers a denser face and spine TIR. A bare air → rarer face against the closed form at four angles: not found elsewhere | keep, wave |
| pass-through calibration (448) | 10 | wave | partly in *Calc Diffraction Gain - Alpha in-medium absorption* (`test_calc_diffraction_gain.cpp:206`, 1e-7). The chain against `calc_diffraction_gain`: only here | keep, wave |
| embedded material counted once (522) | 2 | wave | not found for a solid inside a solid | keep, wave |
| thin-slab Fabry-Perot resolution (561) | 14 | wave | *Calc Diffraction Gain - Fabry-Perot thin-slab resolution* (`:487`) asserts the Airy closed form to 1e-5 | delete |

**`test_sheet_vs_solid.cpp` against the sheet-against-solid tables.** The file runs every row through
`calc_diffraction_gain` with one forward path, so its "solid" is not the solid's full series. Its own
tables read 1.26 dB solid-against-analytic at 85° for concrete. It asserts sanity and a staged bound of
0.75 relative. The newer cases compare a sheet with its traced solid over every path at 1e-3 dB.

| case (line) | assertions | proposal and the case that asserts the statement now |
| --- | ---: | --- |
| Mesh builder conventions (453) | 0 | delete: no assertion, prints only with the variable |
| Analytic reference reduces to the Airy sum (479) | 12 | delete with the file: it checks the file's own `tmm_power` |
| Isolated sheet vs traced solid, Lossless slab, Material sweep, Thickness sweep (524, 559, 594, 647) | 61, 1, 1, 1 | delete: superseded by *Resonant medium D* (sheet against traced solid, with plain twins) and *bare tf C* |
| **Collapsed duplicate equals a single sheet** (685) | 2 | **keep**: the only assertion that two exactly coincident faces, with reversed and with matched normals, equal one sheet, to 1e-9. *Stack geometry C1* has a `C1\|C1` row only as an every-digit twin with a gap between the faces |
| Corner with sheet degrades to a single hit (731) | 1 | delete: *Unentered branches - a corner with a sheet is the sheet alone…* (`test_ray_state_update.cpp:15760`, 346 assertions) |
| Glued sheet equals a solid of the summed thickness (789) | 70 (finite, positive) | delete: *Glued sheet - the near-to-far flux conversion*, *Reciprocity - glued entry against glued exit*, 1e-9 |
| Sheet embedded in a solid (862) | 30 (finite) | delete: *Sheet in a medium*, *Sheet inside a host*, 1e-9 |
| Ordered stack of two different sheets is killed (940) | 4 (finite) | delete: its name states a rule that is gone, since stacks resolve (*One solver*, *Stack geometry*) |
| Interface gain is applied once on each path (976) | 1 (`CHECK(true)`) | delete: *Lumped columns - interface gain on the sheet, glued and embedded paths*, 1e-9 |
| Grazing incidence, sheet vs solid (1035) | 10 (finite) | delete: *Thin sheet - grazing incidence does not depend on the face*, *Thin sheet - the redo sits on the analytic slab at grazing incidence* |
| Acoustic sheet, mass-law route (1082) | 20 (finite) | delete: acoustic, superseded by *Thin sheet - mass-law angle dependence* and the bonded-mass cases |
| **thk = 0 is identical to no sheet at all** (1132) | 1 (bitwise) | **keep**: no other case compares an absent `thk` column with one present and zero |

The two cases to keep need 260 and 104 lines of the file's helpers through `forward_trace`. They could
be kept with those helpers in a wave file, or rewritten on `ub::traced`. A rewrite measures something
else, so it is your call.

**Repeated copies of one table.** Rows of a table were compared by their numeric fields with 8 or more
digits. Exact row strings shared between tables: 12.

- **The three twin tables of the rare-sheet, flat-branch and sheet-tf cases repeat each other.**
  *sheet tf H* (292 rows) shares 60 rows with *flat branch H* (192) and 56 with *rare sheet F* (152).
  *flat branch H* and *rare sheet F* share 52. These are the same sheet families printed three times.
- *mirror behind, the reflection of a wall coated on its near side* (`test_ray_state_update.cpp:13226`)
  and *reflect port, a wall coated on its near side (SH) … every path* (`:13710`) share 24 rows: the same
  wall from two copies of the walker.
- *ray_progress sheet tf A* and *ray_progress bare tf A* share 11 rows, the walls without `tf`.
- The remaining pairs share 1 to 5 rows of common values; these are coincidences.

**Copies of code.** Lambdas of 8 or more lines inside cases: 158, 3,258 lines. Near-identical clusters
hold **1,484 lines beyond one copy each**:

| lambda | copies | lines | where (file:line) |
| --- | ---: | ---: | --- |
| `traced` (one is named `step`) | 14 | 335 | sta:10071, 10273, 10460, 10965, 11413, 11904, 12216, 12645, 12951, 13463, 13994, 14676, 15027, 15124 |
| `trace` | 4 | 242 | pro:2876, 3115, 3326, 3646 |
| `trace` | 2 | 151 | pro:4972, 5129 |
| `trace` | 2 | 130 | pro:2488, 4490 |
| `cm` | 9 | 129 | sta:12336, 12765, 13046, 13600, 14140; pro:2856, 3095, 3305, 3625 |
| `build` | 8 | 129 | pro:2451, 2833, 3072, 3282, 3602, 3848, 4055, 4461 |
| `face` | 12 | 120 | sta:10305 … 15147; **textually identical to `ub::face`** |
| `trace` | 2 | 119 | pro:4708, 4823 |
| `build` | 10 | 110 | sta:10498 … 14710 |
| `run` | 2 | 102 | sta:12272, 12701 |
| `trace` | 2 | 102 | pro:4077, 4320 |
| `walk` | 2 | 92 | sta:13535, 14079 |
| 13 smaller clusters (`event`, `run`, `entry`, `exit_code`, `face`, `chain`, `launched`/`seed`, `seq*`, …) | 2–4 each | 407 in total | listed in `build_linux/scratch/ts1/lambdas.json` |

By name, `trace` has 13 copies (805 lines) in `test_ray_progress.cpp`. `run` has 16 copies (376 lines):
the chain runner at sta:10520, 11024, 11466, 12272, 12701 and 13004, 29 to 51 lines each, plus smaller
ones. Most of the chain runners are below the 0.8 similarity of the clusters above, because each copy grew
its own switch (`verdict`, `relaunch`, `stateless`, a type and an `eps` per event).

The one-thread guard is written out 6 times in `test_ray_state_update.cpp` and 12 times in
`test_ray_progress.cpp`, next to the 22 uses of `ub::OneThread`.

### 0.6 Diagnostics, table by table

Every printed table in `test.txt` opens with a `# diag` header; none is unnamed. 359 tables:

- 297 in the three ray files, one row each in `md/test_split_tables.csv`
- 62 in the `sos_*` files, which are no ray tests and stay as they are

The CSV gives, per table:

- source line, lines, data rows, rows evaluating `m > 0`, rows preceded by a library call, rows printing
  17 digits
- class
- assertions between its rows and right after it, from the runtime listener, and in its code region,
  from the static count
- up to three of the assertion expressions in its code region (the "which")
- category, and plan citation

| class | tables | lines | asserted between its rows | every digit | recorded | history in header | no assertion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wave | 216 | 12,620 | 213 | 5 (258 lines) | 4 (92) | 5 (91) | 2 (50) |
| mixed | 25 | 3,613 | 20 | 5 (751) | 0 | 0 | 5 (885) |
| acoustic | 56 | 2,885 | 54 | 3 (162) | 5 (184) | 1 (19) | 1 (99) |

A table can carry several categories. Plan citation, by text search:

| class | cited by table letter | by header phrase | by case name only | not found |
| --- | --- | --- | --- | --- |
| wave | 46 | 71 | 36 | 63 tables, 2,225 lines |
| mixed | 11 | 6 | 5 | 3 |
| acoustic | 27 | 6 | 8 | 15 |

**Every digit** (regression twins: rows printed to 17 digits, so a diff shows a moved bit)

| table | class | lines | assertions in rows | case |
| --- | --- | ---: | ---: | --- |
| `sta:17109` closed plate E | acoustic | 62 | 125 | Closed plate |
| `sta:17458` held lining F, the mass paid | acoustic | 19 | 19 | Held lining |
| `sta:17486` held lining G, twins | acoustic | 81 | 154 | Held lining |
| `sta:17716` stack geometry C1, what must not move | wave | 59 | 55 | Stack geometry |
| `sta:17752` stack geometry C2, the stack event out of the horizontal plane | wave | 68 | 126 | Stack geometry |
| `sta:18010` rare sheet E, on a rigid wall | wave | 23 | 41 | Rare sheet |
| `sta:18038` rare sheet F, twins | mixed | 156 | **0** | Rare sheet |
| `sta:18452` flat branch H, twins | mixed | 208 | **0** | Flat branch |
| `sta:18957` sheet tf H, twins | mixed | 296 | **0** | Sheet tf |
| `sta:19227` bare tf E, the acoustic main road | mixed | 51 | 94 | Bare tf |
| `sta:19304` bare tf T, twins | wave | 87 | 166 | Bare tf |
| `sta:19427` scalar closed face C, twins | wave | 21 | 4 | Scalar closed face |
| `sta:19847` closed face H, twins | mixed | 40 | 17 | Closed face |

The assertions in the twin rows check finiteness and bounds, never the printed value. The three largest
twin tables carry none. Three more tables say "twin" in the sense of a comparison twin and are not
regression twins: *resonance-matched sheet*, *held lining D*, and *resonant medium B* (643 lines).

**Recorded** (the header says so)

| table | class | lines | assertions in rows | plan |
| --- | --- | ---: | ---: | --- |
| `pro:2787` trajectory, recorded. DENSE, eps 1.50 with m 20 | acoustic | 5 | 3 | case |
| `sta:11368` outside the item, recorded. Reflection off a rigid half-space | wave | 17 | 13 | — |
| `sta:17170` closed plate F1, recorded | acoustic | 19 | 31 | table |
| `sta:17186` closed plate F2, recorded (the header still says the lining is entered at the Snell cosine — plan §1) | acoustic | 28 | 24 | table |
| `sta:17208` closed plate F3, recorded (the header names the item it was recorded for — plan §1) | wave | 48 | 0 | table |
| `sta:17522` held lining H, recorded, what the rule does not reach | acoustic | 66 | 189 | table |
| `sta:18833` sheet tf D2, recorded: tf < 0 about the tie | wave | 15 | 22 | table |
| `sta:18900` sheet tf G, recorded: on a rigid wall | acoustic | 66 | 63 | table |
| `sta:19284` bare tf H, recorded: tf < 0 about the tie | wave | 12 | 24 | table |

Most "recorded" tables still carry bounds on their rows, for example finite or ≤ 1. "Recorded" means the
value itself is not asserted.

**History in the header** (the header describes an earlier state of the code)

| table | class | lines | what the header records |
| --- | --- | ---: | --- |
| `mes:2889` thin sheet at grazing incidence, large face versus small face | wave | 13 | "the old clamp engaged past …" |
| `mes:3200` mass-law angle dependence | acoustic | 19 | column `TL_superseded_dB` from `qd_mass_TL_clipped`, "the superseded form cut the term off at …" |
| `sta:3089` sheet redo at grazing incidence | wave | 12 | "the old placeholder clamped past 89.94270 deg" |
| `sta:4325` exit port at the recovered complex cosine | wave | 32 | a column "at the real projection, which is what the exit used to be evaluated at" |
| `sta:5286` host layer bounded by a sheet | wave | 22 | "the half-space reflection the solver used before the core landed" |
| `sta:5414` sheet_case 4 | wave | 12 | "This case killed the ray outright before the core landed." |

`qd_mass_TL_clipped` is still read by one assertion. *Mass-law term at normal incidence against the
superseded form* (`test_ray_mesh_interact.cpp:3273`) is in the plan's *What is verified* list: at θ = 0 the
two forms agree. A history column can go; that assertion needs its reference rewritten if the helper goes.

In the headers and comments, "used to" / "no longer" occur 19 times in `test_ray_state_update.cpp`, 6 in
`test_ray_mesh_interact.cpp` and 2 in `test_ray_progress.cpp`. The plan also names the headers of the
rare-sheet cases, which still describe a pair rule continued past the critical angle.

**No assertion between its rows**

| table | class | lines | note |
| --- | --- | ---: | --- |
| `sta:14182` lossy verdict. The magnitude test trips at … | wave | 2 | a header line with two numbers |
| `sta:14879` closed-face hand-on B, levels on the rows of A | mixed | 196 | derived from A; no library call |
| `sta:15067` closed-face hand-on S, the traced solid against the sheet, worst | mixed | 29 | derived; no library call |
| `sta:15466` plate depth S, paid − plate | acoustic | 99 | derived from A; no library call |
| `sta:17208` closed plate F3, recorded | wave | 48 | |
| `sta:18038`, `sta:18452`, `sta:18957` the three twin tables above | mixed | 660 | |

Asserted only after their last row: `sta:6757` *scattering-form reference against diag_cm_stack at tf = 0*
(wave, 2 lines) and `sta:19688` *closed face D* (acoustic, 4 lines).

**What this means for the proposed policy.** 213 of the 216 wave tables carry an assertion between their
rows. "Keep only tables that carry assertions or are twins" therefore keeps **12,435** of 12,620 wave
lines, not a few thousand. Reaching a few thousand needs a different rule; options are in 0.8.

### 0.7 Helpers

**Where they are.** All file-scope helpers of `test_ray_state_update.cpp` and `test_ray_mesh_interact.cpp`
are `static`. Those of `test_ray_progress.cpp` sit in an anonymous namespace. `ub`, `bf` and `cf` are
named namespaces of `static` functions. A header with them can be included by several translation units
without link errors, and the test flags end in `-w`, so unused `static` functions raise no warnings.

**Shared by several cases** (they belong in the header):

- the constants and code tables (`C0`, `QPI`, `TOL`, `FRQ`, `OT_*`, `RT_*`, `rt_bit`, `enc`)
- `cpx`, `sv`, `fresnel_te`, `phi_one_way`, `airy_S`, `half_wave`, `close_at`, `base_palette`
- `Rsu`, `make1` and the `check_*` family
- the reference builders `diag_*` (`diag_cm_stack` 17 cases, `diag_sheet_composite` 15, `diag_sc_stack` 5)
- `DiagLayer`, `DiagSheet`, `DiagLump`, `LumpMtl` / `lump_*` (40 cases)
- `ub` (22 cases), `bf` (5), `cf` (2)

In `test_ray_progress.cpp`, `Scene`, `Cfg`, `make_cfg` and the `check_*` helpers serve 40 of its 54
cases. In `test_ray_mesh_interact.cpp`, the `qd_*` oracles serve 1 to 23 cases.

**Used only by acoustic or mixed cases** (they can move with them in phase 2):

- in `test_ray_state_update.cpp`: `DiagLeak` and `diag_leak_series` (4 cases), `DiagMass` and `diag_bonded_TL` (3 cases)
- in `test_ray_mesh_interact.cpp`: `QdMassRef`, `qd_mass_ref`, `qd_mass_TL`, `qd_mass_TL_clipped` and `QdMassSheet`, 100 lines, one case

**Dead** (no reference besides the definition; checked by search):

- `diag_tf_port_rare` (`test_ray_state_update.cpp:3248`, 9 lines), as the plan says
- the constants `OFF` (`:226`), `OT_COR_IIO`, `OT_COR_OII` (`:242–243`) and `RT_II_SLAB` (`:251`). The last
  three are entries of the code tables, which also document the code space; removing them is optional.

**Defined in more than one file:**

| name | where | difference |
| --- | --- | --- |
| `calc_transition_gain`, `calc_transition_gain_mu`, `set_mu` | state update, mesh interact | code identical (compared with comments stripped) |
| `mtl_matrix_to_map` | all three | state update writes a **0-based** `mtl_ind` (`mtl_ind(f) = m`), mesh interact a **1-based** one (`m + 1`), both under the comment "what obj_file_read would emit". The six state-update cases that call it use only the map; none of them calls `ray_mesh_interact`, so the 0-based index never reaches the library. Progress has a float-only, non-template version |
| `FRQ` | state update, progress | `1.0e9` (double) against `10.0e9f` (float) |

These five are the only name clashes. One shared header therefore needs two renames, and phase 1 can avoid
the clash by giving each origin file its own header.

**Copies that can collapse onto the header** (see the cluster table in 0.5):

- the 12 `face` lambdas are `ub::face` verbatim
- the 10 `build` lambdas match `ub::build`
- the 14 `traced` lambdas predate `ub::traced`; they are close to it but carry per-case switches
- the 13 `trace` lambdas of `test_ray_progress.cpp`
- the 9 copies of the TE matrix `cm`
- the 18 local one-thread guards → `ub::OneThread`

The chain runners `run` (6 copies of 29–51 lines) are the least uniform. Each needs the switches it
grew, so collapsing them is real work, and the rows of their cases are the check.

**Library functions the fixtures could call instead of their own code.** Read in `include/quadriga_tools.hpp`:

- **The references must stay independent.** `diag_medium_gain`, `qd_mass_TL`, `calc_transition_gain`
  and the other oracles exist to be checked against `medium_gain`, `interface_gain` and the rest, so
  they cannot call them.
- `mtl_matrix_to_map` and the hand-built palettes could be replaced by `obj_file_read` with a CSV in
  `tests/data`. That adds files and loses readability; I don't recommend it.
- The quads (`diag_quad_x`, `ub::face`, `make_quad`) could come from `quadriga_lib::plane(scale,
  rotation, location)`. Its triangle split and vertex order differ, so hit points and the face side
  could move rows. Possible, but only with a measured diff; the gain is a few 9-number literals.
- `cube()` is already used where a box is meant.
- Path utilities (`coord2path`, `path_to_tube`) do not apply to the polyline bookkeeping the walkers do.

### 0.8 Proposal

**Phase 1 — layout.** One header per origin file, holding everything outside the test cases verbatim
and in order. The cases move verbatim, each with the comment block directly above it:

- `ray_state_update_helpers.hpp`, 2,423 lines
- `ray_progress_helpers.hpp`, 636 lines
- `ray_mesh_interact_helpers.hpp`, 683 lines

The mixed cases go to files of their own, to be split in phase 2. `test_acoustic_interactions.cpp`,
`test_sheet_vs_solid.cpp` and the files that are not ray tests stay as they are until phase 2.

I generated this layout into `build_linux/scratch/ts1/split` and built a test binary from it (nothing
written into `tests/`):

- **1074 cases, 1,297,934 assertions, all passing.**
- Diagnostic output, compared keyed by case: identical except **27 rows, ≤ 2.8e-16**, in exactly the five
  tables the plan lists under *Rows that move without a library change*.

Compile times, each file alone with the test flags on an idle machine (measured twice; the second round
was within 0.5 s of the first). One round taken while another session ran its suite read up to 40% higher
and was discarded.

| file | cases | lines | compile s | peak MB |
| --- | ---: | ---: | ---: | ---: |
| `test_wave_state_update_spec.cpp` (state machine, `Rsu`/`make1`) | 28 | 1,472 | 6.4 | 492 |
| `test_wave_state_update_sheet.cpp` (references: sheet, stack, lumped columns) | 29 | 4,577 | 11.9 | 530 |
| `test_wave_state_update_chain.cpp` (traced chains, the mirror walkers) | 5 | 2,081 | 9.5 | 542 |
| `test_wave_state_update_branch.cpp` (branch cases, `bf`/`cf`, one solver, budget, resonance) | 10 | 1,243 | 8.9 | 534 |
| `test_acoustic_state_update.cpp` | 13 | 2,676 | 12.1 | 549 |
| `test_mixed_state_update_a.cpp` | 7 | 2,314 | 10.5 | 538 |
| `test_mixed_state_update_b.cpp` | 10 | 3,682 | 18.2 | 648 |
| `test_wave_progress.cpp` | 51 | 4,108 | 13.3 | 573 |
| `test_acoustic_progress.cpp` (2 acoustic + 1 mixed) | 3 | 803 | 6.5 | 503 |
| `test_wave_mesh_interact.cpp` | 32 | 2,564 | 9.3 | 529 |
| `test_acoustic_mesh_interact.cpp` | 1 | 146 | 2.6 | 349 |

- **Parallel build.** All eleven compile in **19.6 s** wall (`xargs -P 12`), against 56.5 s for the three
  files now. Total CPU is 109 s against 81.6 s, because each header costs about 2.2 s per translation unit.
- **Peak memory** drops from 1.10 GB to 0.65 GB.
- The file boundaries are line ranges of the current file (2620, 9860 and 15700), chosen by theme.
- Splitting the mixed cases in phase 2 empties `test_mixed_*` into the wave and acoustic files.

**Phase 2 — cleanup list** (each item is for your OK; every dropped assertion is listed with the case that
keeps its statement):

1. **Delete** from `test_acoustic_interactions.cpp`: the frequency round-trip (16 assertions, no library
   call), the resonance peak (3, kept by `ray_state_update:5953` and `:20183`), the thin-slab Fabry-Perot
   case (14, kept by `test_calc_diffraction_gain.cpp:487`), and optionally *convergence away from spectral
   features* (2).
   - Move the two unique `refractive_index` checks (dispersion via `b`; a coincidence leaves the index
     unchanged) into *Public helpers*.
   - The rest stays: 6 wave cases plus *interface_gain coincidence* and the acoustic *medium_gain mass law*.
     It then fits into the wave and acoustic mesh-interact files.
2. **Delete `test_sheet_vs_solid.cpp`** except *Collapsed duplicate* and *thk = 0*. That removes 13
   cases: 0 + 12 + 61 + 1 + 1 + 1 + 1 + 70 + 30 + 4 + 1 + 10 + 20 assertions, each superseded as listed in
   0.5. Keep the two with their helpers, or rewrite them on `ub` (your call).
3. **Split the 18 mixed cases**: 7 by whole tables, 11 by row families. Each acoustic part becomes its own
   case, with the palette and lambdas it needs.
4. **Dead helpers.** Remove `diag_tf_port_rare` and `OFF`; `OT_COR_IIO`, `OT_COR_OII` and `RT_II_SLAB` are optional.
5. **One shared header.** Merge the three per-origin headers into `ray_test_helpers.hpp`, renaming
   `mtl_matrix_to_map` by index base and `FRQ` by type, and keeping one copy of the three identical ITU
   helpers. Move the acoustic-only helpers (`DiagLeak`, `diag_leak_series`, `DiagMass`,
   `diag_bonded_TL`, `qd_mass_*`) into the acoustic files.
6. **Collapse copies** onto the header: `face` (12, identical), `build` (10), the one-thread guards (18), and
   `cm` (9). Then, case by case with a row diff: `traced` (14), `trace` (13), and the chain runners.
   The first group should move no row beyond the known 1e-16 noise; the second may, and every moved row
   would be reported.
7. **Superseded tables, headers and comments.**
   - Delete the 6 history tables or their history columns (`TL_superseded_dB`, the "used to" column of
     `sta:4325`, …), keeping the normal-incidence assertion against a rewritten reference.
   - Fix the headers of closed plate F2 and F3 and of the rare-sheet cases.
   - Fix the stale "currently forced on" comment and `TOL_STAGED` in `test_sheet_vs_solid.cpp`, if it stays.
   - Remove the 27 "used to / no longer" narratives where they only describe history.
8. **Recorded tables.** Delete the 9 recorded tables, or keep those the plan names as the reference of a §5
   limitation. Seven are cited by table letter; the plan's §5 entries would then point to the case and not
   to a table.
9. **Duplicated twin families.** Keep each shared family in one of *sheet tf H*, *flat branch H* and *rare
   sheet F*, cutting up to about 170 duplicated rows, and merge the two near-side coating tables (24 shared rows).

**Diagnostics policy — the options.** Your proposed policy does not reach a few thousand wave lines (see
0.6).

- **(a) Summary by default** *(my recommendation)*. Every asserted table prints its header, its column
  line and one summary row (its worst residual and the bound). The full rows print only under
  `QD_TEST_VERBOSE=2`. Measured from the current headers: about **1,300 wave lines** including blank lines,
  330 acoustic, 160 mixed. A wave regression still fails an assertion; the full table is one switch away.
- **(b) Twins become stored values.** The 5 wave twin tables (258 lines, 242 rows of 17-digit values)
  become assertions against stored values, kept in a data file under `tests/data`, compared at a relative
  1e-12 or bitwise. These should not be inline literals, or 242 rows become code. The 5 mixed twin tables
  follow once split. With (a) and (b) the wave baseline holds no row that is not either asserted or a summary.
- **(c) Keep full rows only for plan-cited tables** (117 wave tables, 6,751 lines), summary rows for the rest.

Recorded-only and history tables: delete, unless a §5 entry of the plan uses them as its reference (see
item 8). `QD_TEST_VERBOSE` stays a switch in every file. In `test_sheet_vs_solid.cpp` it is off today; if
the file shrinks to two cases there is nothing to print.

**Targets.**

- Wave files: 6–13 s each.
- Acoustic files: 3–18 s each as measured now; the acoustic halves of the mixed cases will add to them.
- Whole test build: under 20 s wall on this machine.
- Baselines, if (a) and (b) are taken: `test_wave.txt` about 1,300 lines and `test_acoustic.txt` about
  500, instead of 19,870 together today.

**Phase 3 (for later).** Tags `[wave]` and `[acoustic]` go on the 367 cases of the 14 files. The cases that
are not ray tests — the 4 files that reach nothing in `ray_mesh_interact.cpp`, and the `sos_*` and other
suites — get neither tag, and stay in `test.txt` if you keep it. The baseline commands would then be
`./build_linux/test_bin "[wave]" > test_wave.txt` and the same with `[acoustic]`.

### 0.9 Decisions for you

1. Phase 1 with one header per origin file (verbatim, no renames), and the merge into one header in phase
   2. Or one header already in phase 1, which needs the two renames and so is not a pure move.
2. The mixed cases: in their own `test_mixed_*` files in phase 1 (my proposal), or straight into the
   acoustic files as the brief says.
3. The file names and the three boundaries of the wave state-update split.
4. The diagnostics policy: (a), (b), (c), or a combination.
5. The two cases kept from `test_sheet_vs_solid.cpp`: with their helpers, or rewritten on `ub`.
6. Whether `test.txt` stays as the whole-suite baseline.

### 0.9a Decisions taken (Stephan, 2026-10-14)

- **How to run it:** phases 1, 2 and 3 in a row, with one report at the end. Stop and ask only if a
  statement would lose its last guard. Every other irregularity (moved rows, compile times) goes into the
  final report.
- **Phase 1:**
  - One header per origin file, moved verbatim: `ray_state_update_helpers.hpp`, `ray_progress_helpers.hpp`,
    `ray_mesh_interact_helpers.hpp`.
  - Mixed cases go into `test_mixed_state_update_a/b.cpp`. The mixed progress case goes into
    `test_acoustic_progress.cpp`.
  - File names and the cuts at lines 2620 / 9860 / 15700 as proposed in 0.8.
- **Phase 2, structure:**
  - Merge the three headers into `ray_test_helpers.hpp`, renaming the `mtl_matrix_to_map` variants and
    `FRQ`, and keeping one copy of the identical ITU helpers. Move the acoustic-only helpers into the
    acoustic files.
  - Split all 18 mixed cases: 7 by whole tables, 11 by row families. The `test_mixed_*` files are removed.
  - Collapse identical copies only: `face`, `build`, the one-thread guards, `cm`, and the duplicated ITU
    helpers. The runners stay as they are.
- **Phase 2, old files:**
  - `test_sheet_vs_solid.cpp`: delete it except *Collapsed duplicate* and *thk = 0*. Those two move
    verbatim with the helpers they need into a wave file.
  - `test_acoustic_interactions.cpp`: delete the frequency round-trip, the resonance peak, Fabry-Perot and
    *away from spectral features*. Move the two unique `refractive_index` checks into *Public helpers*.
    The rest goes into the wave and acoustic mesh-interact files, and the file is deleted.
- **Phase 2, removals:**
  - Remove `diag_tf_port_rare`, `OFF`, `OT_COR_IIO`, `OT_COR_OII` and `RT_II_SLAB`.
  - Fix the stale headers (closed plate F2 and F3, the rare-sheet cases) and remove the history-only
    "used to / no longer" narratives.
  - Merge the duplicated twin families: each family shared by *sheet tf H*, *flat branch H* and *rare
    sheet F* stays in one table, and the two near-side coating tables become one.
  - Recorded and history tables are deleted unless a §5 limitation of the plan uses them as its reference.
    The normal-incidence mass assertion keeps a rewritten reference.
- **Diagnostics:**
  - All rows keep printing, wave and acoustic. No summary rule.
  - The wave every-digit twins additionally get assertions against stored values in a CSV under
    `tests/data`, at 1e-12 relative. Their tables stay printed.
- **Phase 3:**
  - `[wave]` and `[acoustic]` tags on every case of the ray files, giving `test_wave.txt` and
    `test_acoustic.txt`.
  - `test.txt` stays the whole-suite baseline.
  - Cases that are not ray tests get no tag.

### 0.10 Side findings

- **`src/quadriga_lib_material_helpers.hpp`** carries a second `medium_gain_impl` / `medium_loss_dB` that
  reads the mass column, last touched in commit `227aab6` (*Fix: Mass law ignored incidence angle*).
  Nothing in the repository includes it (searched outside `build_linux`). It is outside this task, since
  `src/` is not touched here. It belongs in `acoustic.md` as dead mass-law code to remove or reconcile.
- The two slowest cases, 8.8 s and 9.0 s, are walkers that already run on one thread. If the suite time
  matters, their depth or angle grid is the lever. It is not part of the split.
- Another session's `test_bin`, running in `/sjc/quadriga-lib-conf`, loaded the machine for about three
  minutes during this phase. Compile times were taken only after `pgrep` and the load average showed it idle.

---

## First step for future sessions

From the repository root:

```
cmake --build build_linux --parallel
./build_linux/test_bin "[wave]" > test_wave.txt          # All tests passed (165110 assertions in 179 test cases)
./build_linux/test_bin "[acoustic]" > test_acoustic.txt  # All tests passed (31856 assertions in 35 test cases)
./build_linux/test_bin > test.txt                        # All tests passed (1298332 assertions in 1073 test cases)
```

- `test_wave.txt` has 13,856 lines and `test_acoustic.txt` has 5,355. Both are untracked; `.gitignore` lists
  only `test.txt`, and I did not touch it because it is outside `tests/` and `md/`.
- `test.txt` stays the whole-suite baseline (19,656 lines). The cases that are not ray tests carry no tag
  and appear only there.
- The wave twins are asserted against `tests/data/wave_twins.tsv`. After a change that is meant to move
  them, regenerate it:
  `rm tests/data/wave_twins.tsv; QD_TWIN_WRITE=tests/data/wave_twins.tsv ./build_linux/test_bin "[#test_wave_state_update_branch]" -#`

---

## Phase 1 — pure move (2026-10-14)

Generated by `build_linux/scratch/ts1/gen_phase1.py` from the three original files, saved in `ts1/orig`.
- Each original file got a header holding everything outside its test cases, verbatim and in order:
  `ray_state_update_helpers.hpp`, `ray_progress_helpers.hpp`, `ray_mesh_interact_helpers.hpp`.
- Every case moved verbatim with the comment block directly above it, into
  `test_wave_state_update_{spec,sheet,chain,branch}.cpp`, `test_acoustic_state_update.cpp`,
  `test_mixed_state_update_{a,b}.cpp`, `test_wave_progress.cpp`, `test_acoustic_progress.cpp`,
  `test_wave_mesh_interact.cpp` and `test_acoustic_mesh_interact.cpp`.
- `test_ray_state_update.cpp`, `test_ray_progress.cpp` and `test_ray_mesh_interact.cpp` were deleted.

Acceptance:
- **1074 cases, 1,297,934 assertions, all passing.**
- **Diagnostic output identical row by row, keyed by case**, except 27 rows in their residual columns,
  ≤ 2.8e-16, in exactly the tables the plan lists under *Rows that move without a library change*:
  - *Sheet in a medium* (4 rows, 2.8e-16)
  - *Layered stack - a host layer bounded by a sheet* (3, 3.5e-18)
  - *Lined on both faces* (6, 2.7e-18)
  - *Layered wall* (2, 1.6e-18)
  - *Sheet inside a host - the mirror behind the ray* (12, 6.9e-18)
- Compile time of each file alone, from the phase-1 snapshot in `ts1/phase1`:

  | file | s |
  | --- | ---: |
  | `test_wave_state_update_spec.cpp` | 6.6 |
  | `test_wave_state_update_sheet.cpp` | 12.6 |
  | `test_wave_state_update_chain.cpp` | 10.2 |
  | `test_wave_state_update_branch.cpp` | 9.3 |
  | `test_acoustic_state_update.cpp` | 12.4 |
  | `test_mixed_state_update_a.cpp` | 11.0 |
  | `test_mixed_state_update_b.cpp` | 19.1 |
  | `test_wave_progress.cpp` | 13.7 |
  | `test_acoustic_progress.cpp` | 6.3 |
  | `test_wave_mesh_interact.cpp` | 9.5 |
  | `test_acoustic_mesh_interact.cpp` | 2.7 |

  Before: 56.1 s, 16.2 s and 9.3 s. In parallel the prototype of this layout compiled in 19.6 s.

## Phase 2 — cleanup (2026-10-14)

Each step was built, run, and compared with the run before it, keyed by every case and section
(`ts1/cmp2.py`, `ts1/cmpsplit.py`).

| step | cases | assertions | rows that moved |
| --- | ---: | ---: | --- |
| phase 1 | 1074 | 1,297,934 | — |
| 1. one header | 1074 | 1,297,934 | none |
| 2. dead code, history text, recorded tables | 1074 | 1,297,846 | only edited header lines, the removed column and the deleted tables |
| 3. the 18 mixed cases split | 1091 | 1,297,846 | none: every original case's data rows are exactly its two halves' rows |
| 4. duplicated twin families, copies collapsed | 1091 | 1,297,846 | none besides the removed twin rows |
| 5. the two old files pruned | 1073 | 1,297,596 | none in the cases that remain |
| 6. twins asserted against stored values, headers corrected | 1073 | 1,298,332 | only edited header lines |

**1. One header.** `ray_test_helpers.hpp` replaces the three per-origin headers:
- The state-update copy of `mtl_matrix_to_map`, which writes a 0-based `mtl_ind`, is renamed
  `mtl_matrix_to_map0`. Its 7 callers are in `test_wave_state_update_spec.cpp` and one acoustic case.
- The progress `FRQ` (float, 10 GHz) is renamed `PROG_FRQ`, at 5 uses. The two case-local `FRQ = 1.0e9`
  declarations are untouched.
- Dropped, because the copy that stays has the same code: the mesh-interact copies of
  `calc_transition_gain` (58 lines), `calc_transition_gain_mu` (48) and `set_mu` (9), and the float
  `mtl_matrix_to_map` of progress (38), which the template does at float.
- Zero rows moved.

**2. Removals.**
- Dead: `diag_tf_port_rare` (9 lines), `OFF`, `OT_COR_IIO`, `OT_COR_OII`, `RT_II_SLAB`.
- History text, with every assertion kept:
  - The "old clamp" and "old placeholder" header lines of the two grazing tables.
  - The `TL_superseded_dB` column and `qd_mass_TL_clipped`. The normal-incidence assertion now reads
    `max(0, m log10((f/fRef) thk))`, the column definition.
  - The headers of *exit port at the recovered complex cosine*, *host layer bounded by a sheet* and
    *sheet_case 4*, now describing the control they print instead of the code they once recorded.
- Recorded tables. §5 of the plan uses closed plate F1, held lining H, sheet tf D2, bare tf H and the
  rigid half-space table of *Phase reference* as references, so they stay. The *trajectory, recorded*
  table of `ray_progress` also stays, because open issue 3.7 cites it. Deleted:

  | table | assertions dropped | statement | still asserted by |
  | --- | ---: | --- | --- |
  | closed plate F2 | 24 `REQUIRE(a.size() == 2)` | a chain through a rigid mass-law wall and its rigid lining reaches the second event | *Held lining* A, which asserts the level of the same chains, lining met first and met last, against the solid twin |
  | closed plate F3 | 0 | — | its rows are asserted in *Rare sheet* (plan §1) |
  | sheet tf G | 64 `CHECK(isfinite(T + R))` | both ports of a sheet with tf on a rigid wall with tf 0.3 are finite | *Closed face* A, which asserts T + R < 1 on the same wall for the same sheets, tf, angles and 6 mm |

**3. The 18 mixed cases** (tool `ts1/splitcase.py`, specs `ts1/specs.py`, row filters `ts1/rowfilter.py`):
- **The halves.** Each wave half keeps the original name and gains one comment line pointing to its
  acoustic half. The acoustic half is "<name> - the mass-law part"; it holds the case's setup, the moved
  tables and rows, and a pointer back.
- **Exceptions to the naming.**
  - *Leak direction - a wave held in a mass-law body…* keeps its name on the acoustic half, whose
    statement it is. The wave half is *Leak direction - the direction a closed face writes for a wave
    launched inside a solid*.
  - *Closed face - tf at a closed face is a pad…* keeps its name on the acoustic half; nearly all of its
    rows sit on a rigid wall with `m 20`. The wave half is *Closed face - a coating met from inside a
    dense solid*: the coating rows of A, D2, and the coating twins of H.
  - *Closed plate* moved whole to acoustic: its only wave table, F3, was deleted in step 2.
- **How each split.**
  - Whole tables: *Total reflection*, *Direction*, *Refracting tracer*, *Phase reference*, *Measurement -
    the mirror…*, lossy verdict K, and the `ray_progress` trajectory case.
  - Loops or rows: the declined stack (`m 10` row), the leak-chain step (wood rows), lossy verdict J
    (walls 16 and 17), the own glued composite (the wood panel), stack geometry A and B (pairs with R or
    N), rare sheet F, flat branch H, sheet tf D and H (rows named mass-law or on the rigid wall with
    `m 20`), bare tf E (`m 30`), closed face A (`w < 2`) and H (wall rows).
  - One column split: flat branch G. Its `|T_m|` column, with the mass column, went to the acoustic
    half with its check; T, R, dT and dR stayed.
- **Verification.**
  - The assertion count is unchanged at 1,297,846, so every assertion still runs exactly once.
  - The data rows of every original case equal the union of its two halves' rows, with 0 differences.
  - A marker build of the final suite shows no `[wave]` case building a `Material` with `m > 0`, and
    every `[acoustic]` case building one.

**4. Duplicated twin families.**
- Wave: 13 families, 7 in *rare sheet F* and 6 in *flat branch H*, are printed with every digit by
  *sheet tf H* or by each other, and were removed from these two tables.
  The union of distinct rows is unchanged (452 of 452).
- Acoustic: the rigid mass-law family was in all three acoustic halves; it now stays only in sheet tf H.
  The acoustic half of flat branch H was left empty and was removed. Distinct rows unchanged (36 of 36).
- These tables carry no assertion between their rows, so no assertion was dropped.
- **Not done: merging the two near-side coating tables.** Their 24 shared values are one wall's
  reflection reached by two instruments, a two-path chain (*Measurement - the mirror behind the ray…*)
  and the every-path walker (*Sheet reflect port…*). Each asserts its own instrument against the matrix,
  so neither is a copy of the other.

**4b. Copies collapsed onto the header** (identical code only):
- 18 `face` lambdas became `using ub::face;` and 15 `build` lambdas became `using ub::build;`. Both
  counts include the copies the split duplicated.
- 18 local one-thread guards became `ub::OneThread one_thread;`.
- The 9 `cm` copies are 5 different variants, so they stay. The `traced`/`trace` runners and the chain
  runners stay, as decided.
- Zero rows moved.

**5. The two old files.**
- `test_acoustic_interactions.cpp` is deleted.
  - Its helpers (`ac2rf`, `single_material`, `ref_scalar_reflection_gain`, `rmi_gain`, `Approx`) moved to
    the header; `rf2ac` was dropped, since only the deleted round-trip case used it.
  - Moved: *medium_gain in-medium loss* to `test_acoustic_mesh_interact.cpp`; *interface_gain…*,
    *convergence with acoustic terms absent*, *scalar total internal reflection*, *pass-through
    calibration* and *embedded material counted once* to `test_wave_mesh_interact.cpp`.
  - Two new checks in *Public helpers*: dispersion via `b`, and a coincidence column leaving the index
    unchanged.

  Dropped:

  | case | assertions | statement | still asserted by |
  | --- | ---: | --- | --- |
  | frequency mapping round-trip | 16 | the file's own `ac2rf`/`rf2ac` constants | — (no library code) |
  | refractive_index … not the coincidence | 5 | air = 1, `n = √a`, `b`, coincidence, pole | *Public helpers* (air, `√a`, and now `b` and the coincidence); *Resonance - the public index…* (pole, 1e-12) |
  | permittivity resonance absorption peak | 3 | a pole lowers a face's reflection | *Lumped columns - a permittivity resonance is part of the medium…* (1e-9), *Resonant medium* B |
  | convergence away from spectral features | 2 | far-off features are negligible | the coincidence formula in *interface_gain…* (1e-6), the resonance cases |
  | thin-slab Fabry-Perot resolution | 14 | the resolved slab ripples, the naive one is flat | *Calc Diffraction Gain - Fabry-Perot thin-slab resolution* (Airy to 1e-5, resolution off flat) |

- `test_sheet_vs_solid.cpp` is deleted, and `test_wave_sheet_vs_solid.cpp` keeps *Collapsed duplicate*
  and *thk = 0* verbatim with the helpers they reach. The unused helpers, `TOL_TARGET`/`TOL_STAGED`, and
  the stale "currently forced on" note are gone. Dropped, 13 cases and 212 assertions, each with the
  case that asserts its statement now:
  - *Mesh builder conventions* (0) and *Analytic reference…* (12): neither checks library code.
  - Isolated, lossless, material sweep, thickness sweep (61, 1, 1, 1): *Resonant medium* D, *bare tf* C.
  - *Corner…* (1): *Unentered branches - a corner with a sheet…*.
  - *Glued sheet…* (70): *Glued sheet - the near-to-far flux conversion*, *Reciprocity - glued entry against glued exit*.
  - *Sheet embedded…* (30): *Sheet in a medium*, *Sheet inside a host*.
  - *Ordered stack… killed* (4): *One solver*, *Stack geometry*.
  - *Interface gain applied once* (1): *Lumped columns - interface gain…*.
  - *Grazing…* (10): *Thin sheet - grazing incidence…*, *Thin sheet - the redo…*.
  - *Acoustic sheet, mass-law route* (20): *Thin sheet - mass-law angle dependence*, the bonded-mass cases.

**6. Stale text and stored twins.**
- Rewritten to describe the transformer of the sheet-tf case instead of a continued pair rule: the
  opening comments of *Rare sheet* and *Flat branch*.
- Headers of the split tables now list only the rows their half holds: declined stack, lossy verdict J,
  rare sheet F, flat branch G and H, sheet tf H, bare tf E, closed face A and H.
- The other "used to / no longer" passages that only recorded history are rewritten in the present tense:
  20 of the 24. The four that state present physics stay.
- The case-group banners left in the header are gone: the stale state-update and mesh-interact ones are
  deleted, and the ten `ray_progress` group titles are back in front of their groups.
- `DiagMass` and `diag_bonded_TL` moved to `test_acoustic_state_update.cpp`, and `QdMassRef`,
  `qd_mass_ref`, `qd_mass_TL` and `QdMassSheet` to `test_acoustic_mesh_interact.cpp`.
- **Stored twins.** The ten wave tables that print every digit now also assert each row against
  `tests/data/wave_twins.tsv`, 736 rows, one `CHECK` per row, relative 1e-12, through `twin()` in the
  header. The tables are *stack geometry* C1 and C2, *rare sheet* E and F, *flat branch* H, *sheet tf* H,
  *bare tf* E and T, *scalar closed face* C, and the wave *closed face* H. Not included: the
  float columns of the `ray_progress` trajectory case, which print 9 digits and were not on the twin list.

**Compile times after phase 2** (each file alone, idle machine):

| file | s | peak MB |
| --- | ---: | ---: |
| `test_wave_state_update_spec.cpp` | 7.0 | 517 |
| `test_wave_state_update_sheet.cpp` | 14.1 | 583 |
| `test_wave_state_update_chain.cpp` | 18.8 | 703 |
| `test_wave_state_update_branch.cpp` | 18.0 | 705 |
| `test_wave_progress.cpp` | 16.5 | 682 |
| `test_wave_mesh_interact.cpp` | 10.7 | 525 |
| `test_wave_sheet_vs_solid.cpp` | 3.2 | 371 |
| `test_acoustic_state_update.cpp` | 21.2 | 757 |
| `test_acoustic_progress.cpp` | 5.1 | 485 |
| `test_acoustic_mesh_interact.cpp` | 3.2 | 403 |

- The merged header costs 2.7 s per translation unit, against 2.2 s for each per-origin header. The
  wave chain and branch files grew by the wave halves of the split cases.
- A full parallel rebuild of all 83 test objects takes **39.7 s**. Before, the old state-update file
  alone took 56.1 s, so no parallel build could be shorter than that.

**Run time** (idle machine): the suite 41.4 s, against 37.3 s at the start; the 4 s were not broken
down by case. `[wave]` 24.3 s, `[acoustic]` 1.4 s. An earlier set of runs
read 71 s while a Firefox content process held one core at 100%; every OpenMP team waits on that core.

## Phase 3 — baselines (2026-10-14)

- `[wave]` is on all 179 cases of `test_wave_*.cpp` and `[acoustic]` on all 35 cases of
  `test_acoustic_*.cpp`, appended to any existing tags.
- The other ray-adjacent files (`test_calc_diffraction_gain.cpp`, `test_gains.cpp`,
  `test_ray_commit.cpp`, …) carry no tag, as decided. By measurement they are wave.
- Last lines:
  - `test_wave.txt`: `All tests passed (165110 assertions in 179 test cases)`
  - `test_acoustic.txt`: `All tests passed (31856 assertions in 35 test cases)`
  - `test.txt`: `All tests passed (1298332 assertions in 1073 test cases)`
- See *First step for future sessions* above.

### Files

- New: `tests/catch2_tests/ray_test_helpers.hpp`, the ten `test_wave_*.cpp` / `test_acoustic_*.cpp`
  files, and `tests/data/wave_twins.tsv`.
- Deleted: `test_ray_state_update.cpp`, `test_ray_progress.cpp`, `test_ray_mesh_interact.cpp`,
  `test_sheet_vs_solid.cpp`, `test_acoustic_interactions.cpp`.
- Changed: one comment in `test_ray_commit.cpp`.
- Nothing is staged or committed, and `src/` is untouched.

---

## Appendix A — per-case inventory of the five ray files

Columns: line range, lines, assertions, run time in seconds, output lines in `test.txt`, class, `Material`
constructions with `m > 0`, library functions reached (`rmi·345` = `ray_mesh_interact` with types 3, 4
and 5; `rti` `ray_triangle_intersect`, `rsu` `ray_state_update`, `prog` `ray_progress`, `mg` / `ig` / `ri`
the public `medium_gain` / `interface_gain` / `refractive_index`, `cdg` `calc_diffraction_gain`, `objr`
`obj_file_read`, `subf` `ray_subdivide_flag`, `xprc` `xpr_coeff`). The other nine files are in
`md/test_split_inventory.csv`.

#### `test_ray_state_update.cpp`

| lines | n | asserts | s | out | class | m>0 | calls | case |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 647–669 | 23 | 17 | 0.000 | 0 | wave | 0 |  | ray_state_update - state and code encoding |
| 671–704 | 34 | 25 | 0.024 | 0 | wave | 0 | rsu·4 | ray_state_update - dispatch: o-i entry family |
| 706–752 | 47 | 35 | 0.008 | 0 | wave | 0 | rsu·4 | ray_state_update - dispatch: i-o exit family |
| 754–786 | 33 | 23 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - dispatch: material-to-material family |
| 788–826 | 39 | 29 | 0.003 | 0 | wave | 0 | rsu·4 | ray_state_update - dispatch: corner families |
| 828–851 | 24 | 21 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - dispatch: no crossing and unmatched codes |
| 853–890 | 38 | 31 | 0.002 | 0 | wave | 0 | rsu·03 | ray_state_update - dispatch: reflection pass |
| 892–938 | 47 | 22 | 0.002 | 0 | wave | 0 | rsu·34 | ray_state_update - flag semantics |
| 940–992 | 53 | 33 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - Airy factor against the closed form |
| 994–1047 | 54 | 40 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - survival gate thresholds on eps |
| 1049–1093 | 45 | 3 | 0.000 | 0 | wave | 0 | rsu·4 | ray_state_update - lossy medium feeds \|phi\| from medium_gain |
| 1095–1132 | 38 | 3 | 0.000 | 0 | acoustic | 8 | rsu·4 | ray_state_update - mass-law material is excluded from slab resolution |
| 1134–1160 | 27 | 4 | 0.000 | 0 | wave | 0 | rsu·4 | ray_state_update - transmission factor folds into the resolved gain |
| 1162–1217 | 56 | 19 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - resolve clamp near the Airy pole |
| 1219–1263 | 45 | 28 | 0.001 | 0 | wave | 0 | rsu·34 | ray_state_update - cross-pass invariance of the resolve decision |
| 1265–1280 | 16 | 8 | 0.000 | 0 | wave | 0 | rsu·34 | ray_state_update - a resolved ray passes an exit transparently |
| 1313–1378 | 66 | 44 | 0.002 | 0 | wave | 0 | rsu·14 | ray_state_update - gain consistency and batch determinism |
| 1380–1465 | 86 | 34 | 0.003 | 0 | wave | 0 | rsu·4 | ray_state_update - optional arguments |
| 1467–1515 | 49 | 23 | 0.002 | 0 | wave | 0 | rsu·14 | ray_state_update - gain operation semantics and probe handling |
| 1517–1559 | 43 | 14 | 0.000 | 0 | wave | 0 | rsu·34 | ray_state_update - energy conservation of a lossless slab |
| 1561–1601 | 41 | 4 | 0.000 | 0 | wave | 0 | rsu·34 | ray_state_update - energy safety of a lossy slab |
| 1603–1630 | 28 | 17 | 0.001 | 0 | wave | 0 | rsu·45 | ray_state_update - scalar refraction (interaction type 5) |
| 1632–1672 | 41 | 7 | 0.002 | 0 | wave | 0 | rsu·34 | ray_state_update - path direction correction (VBS) |
| 1674–1745 | 72 | 11 | 0.002 | 0 | wave | 0 | rsu·4 | ray_state_update - accumulated distance ledger |
| 1747–1762 | 16 | 3 | 0.000 | 0 | wave | 0 | rsu·4 | ray_state_update - permeability affects the refractive index |
| 1764–1797 | 34 | 2 | 0.001 | 0 | wave | 0 | rsu·4 | ray_state_update - resonance and coincidence material columns |
| 1869–1873 | 5 | 70 | 0.001 | 0 | wave | 0 | rsu·134 | ray_state_update - type parity between double and float |
| 1875–1949 | 75 | 29 | 0.000 | 0 | wave | 0 | rsu·4 | ray_state_update - ray_ind mapping and read-only inputs |
| 1968–2272 | 305 | 138 | 0.014 | 0 | wave | 0 | rsu·012345 | ray_state_update - input validation |
| 2624–2714 | 91 | 24 | 0.000 | 10 | wave | 0 | rsu·1 | Diagnostic - Slab factor per polarization |
| 2716–2842 | 127 | 15 | 0.000 | 15 | wave | 0 | rsu·4 | Diagnostic - Entry boundary evaluated at two angles |
| 2844–2897 | 54 | 32 | 0.000 | 19 | wave | 0 | rsu·4 | Diagnostic - Series collapse when the previous medium matches the slab |
| 2899–2965 | 67 | 20 | 0.000 | 7 | wave | 0 | rsu·4 | Diagnostic - Host layer carried through a matched sheet |
| 2967–3062 | 96 | 29 | 0.000 | 9 | wave | 0 | rsu·4 | Diagnostic - Excess-phase walk-off convention |
| 3064–3114 | 51 | 27 | 0.000 | 13 | wave | 0 | rsu·1 | Thin sheet - the redo sits on the analytic slab at grazing incidence |
| 3897–4066 | 170 | 473 | 0.001 | 70 | wave | 0 | rsu·01 | Glued sheet - the near-to-far flux conversion |
| 4068–4301 | 234 | 400 | 0.002 | 27 | wave | 0 | rsu·0134 | Sheet in a medium - the composite at the exact in-medium cosine |
| 4303–4413 | 111 | 84 | 0.001 | 33 | wave | 0 | rsu·14 | Exit boundary - the port at the recovered angle |
| 4415–4521 | 107 | 144 | 0.001 | 32 | wave | 0 | rti rmi·4 rsu·4 | Traced solid - air \| slab \| air chain against the analytic slab |
| 4523–4817 | 295 | 188 | 0.001 | 41 | wave | 0 | rti rmi·4 rsu·0134 | Permeability - the traced solid and a magnetic layer in a stack |
| 4827–4942 | 116 | 161 | 0.000 | 74 | wave | 0 |  | Layered stack - the characteristic-matrix reference against the single-layer oracles |
| 4944–5236 | 293 | 336 | 0.001 | 90 | wave | 0 | rsu·01 | Permeability - the sheet, stack and glued paths |
| 5238–5366 | 129 | 98 | 0.001 | 23 | wave | 0 | rsu·4 | Layered stack - a host layer bounded by a sheet |
| 5368–5490 | 123 | 70 | 0.001 | 13 | wave | 0 | rsu·0134 | Layered stack - sheet_case 4 resolves rather than killing the ray |
| 5537–5702 | 166 | 270 | 0.001 | 25 | wave | 0 | rsu·01 | Lumped columns - interface gain on the sheet, glued and embedded paths |
| 5704–5951 | 248 | 860 | 0.005 | 110 | wave | 0 | rsu·0134 | Lumped columns - the transmission factor on a sheet, both ports |
| 5953–6130 | 178 | 170 | 0.001 | 38 | wave | 0 | rsu·0134 | Lumped columns - a permittivity resonance is part of the medium on the sheet paths |
| 6132–6383 | 252 | 314 | 0.005 | 55 | mixed | 32 | rsu·0134 | Lumped columns - an ordered stack: what the core carries, by matrix and by scattering |
| 6385–6506 | 122 | 241 | 0.001 | 55 | acoustic | 8 | rsu·4 | Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space |
| 6517–6697 | 181 | 260 | 0.004 | 97 | wave | 0 | rsu·0134 | Neighbour tf - each face of a sheet is owed by the pair of media that form it |
| 6699–6964 | 266 | 758 | 0.009 | 176 | wave | 0 | rsu·0134 | Stack with tf - the scattering form against its reference, produce mode and itself |
| 6966–7118 | 153 | 284 | 0.002 | 64 | acoustic | 280 | rsu·34 | Stack with a mass-law layer - resolved by segments, not killed |
| 7183–7440 | 258 | 97 | 0.001 | 47 | mixed | 45 | rsu·0134 | Total reflection of the outer pair - the sheet stays in the result |
| 7442–7554 | 113 | 137 | 0.001 | 23 | wave | 0 | rsu·14 | Reciprocity - glued entry against glued exit |
| 7556–7697 | 142 | 34 | 0.001 | 20 | acoustic | 57 | rti rmi·4 rsu·4 cube | Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids |
| 7699–7925 | 227 | 48 | 0.001 | 54 | mixed | 12 | rsu·134 | Leak chain - each face at the cosine the wave has and the series on a closed far face |
| 7958–8353 | 396 | 220 | 0.003 | 121 | acoustic | 1204 | rsu·34 | Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate |
| 8355–8425 | 71 | 72 | 0.001 | 11 | acoustic | 72 | rti rmi·0134 cube | Ports of a mass-law face from inside - the reflect port is the other direction's |
| 8427–8679 | 253 | 356 | 0.005 | 120 | wave | 0 | rsu·0134 | Lined on both faces - a glued exit behind a sheet exits into air |
| 8681–9063 | 383 | 808 | 0.005 | 177 | wave | 0 | rti rmi·4 rsu·0134 | Layered wall - a glued exit behind a bare-entered second layer exits into air |
| 9065–9239 | 175 | 96 | 0.001 | 65 | wave | 0 | rsu·4 | Recovery behind a sheet - the leg after a sheet crossing at the exact in-medium cosine |
| 9241–9354 | 114 | 432 | 0.002 | 60 | wave | 0 | rti rmi·4 rsu·4 | Measurement - a host entered from a lossy solid across a bare face |
| 9356–9557 | 202 | 336 | 0.004 | 80 | wave | 0 | rti rmi·14 rsu·134 | Recovery behind a lossy solid - EM, a second transition and a sheet inside the host |
| 9559–9653 | 95 | 104 | 0.001 | 22 | wave | 0 | rsu·4 | Recovery behind a lossy solid - a face that was evaluated at a real angle keeps the solid named |
| 9655–9729 | 75 | 256 | 0.003 | 70 | wave | 0 | rsu·34 | Sheet inside a host - the mirror behind the ray is backed by the host |
| 9731–9828 | 98 | 72 | 0.001 | 25 | acoustic | 768 | rsu·4 | Bonded mass - a mass-law sheet inside a mass-law wall |
| 9830–9862 | 33 | 6 | 0.001 | 0 | wave | 0 | rsu·4 | ray_state_update - the material index is 14 bits |
| 9864–10122 | 259 | 437 | 0.005 | 98 | wave | 0 | rti rmi·1345 rsu·1345 | Recovery gate - the hit code a traced run carries |
| 10124–10417 | 294 | 386 | 0.004 | 131 | mixed | 66 | rti rmi·345 rsu·345 | Direction - a ray leaves a parallel slab parallel to how it arrived |
| 10419–10928 | 510 | 4579 | 0.043 | 322 | mixed | 145 | rti rmi·2345 rsu·2345 | Refracting tracer - a traced bend is known from the state and the leg is exact |
| 10930–11384 | 455 | 1454 | 0.042 | 424 | mixed | 516 | rti rmi·345 rsu·345 | Phase reference - a traversal carries the phase at the point the polyline leaves at |
| 11386–11868 | 483 | 409 | 0.042 | 162 | wave | 0 | rti rmi·0345 rsu·0345 | Exits and verdicts - every exit at the cosine the wave has there |
| 11870–12178 | 309 | 2916 | 0.061 | 376 | mixed | 4644 | rti rmi·45 rsu·45 | Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on the wave's |
| 12180–12611 | 432 | 690 | 0.052 | 192 | mixed | 165 | rti rmi·345 rsu·345 | Measurement - the mirror behind a second solid and behind a sheet inside a wall |
| 12613–12902 | 290 | 528 | 0.029 | 134 | wave | 0 | rti rmi·345 rsu·345 | Sheet reflection - a declined series is followed and not dropped |
| 12904–13415 | 512 | 3464 | 8.800 | 791 | wave | 0 | rti rmi·345 rsu·345 | Measurement - the mirror behind the ray after a back reflection |
| 13417–13946 | 530 | 1374 | 0.170 | 335 | wave | 0 | rti rmi·345 rsu·345 | Sheet reflect port - where the layer lies and what the echo pays |
| 13948–14622 | 675 | 1758 | 0.010 | 417 | mixed | 397 | rti rmi·012345 rsu·012345 | Lossy verdict - a lossy second medium has no critical angle |
| 14624–15074 | 451 | 1639 | 0.020 | 568 | acoustic | 3937 | rti rmi·45 rsu·45 | Leak direction - a closed face hands the wall behind a traced layer the direction the wave has in air |
| 15076–15580 | 505 | 1736 | 0.042 | 1023 | acoustic | 66824 | rti rmi·345 rsu·345 | Bonded mass - the plate at any depth where the slopes agree |
| 15760–15843 | 84 | 346 | 0.000 | 36 | wave | 0 | rti rmi·34 rsu·34 | Unentered branches - a corner with a sheet is the sheet alone once the state is kept |
| 15845–16009 | 165 | 1692 | 0.008 | 193 | wave | 0 | rti rmi·345 rsu·345 | Unentered branches - the displacement budget caps what a sheet takes back out of its host |
| 16011–16097 | 87 | 224 | 0.001 | 51 | wave | 0 | rti rmi·34 rsu·34 | Unentered branches - the core's loss cap is a floor on the transmit port and leaves the reflect port alone |
| 16099–16163 | 65 | 88 | 0.001 | 29 | acoustic | 518 | rti rmi·45 rsu·45 | Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged |
| 16165–16229 | 65 | 576 | 0.007 | 198 | wave | 0 | rti rmi·012345 rsu·012345 | Unentered branches - sheet and stack and slab series at float against double |
| 16231–16374 | 144 | 264 | 0.001 | 106 | mixed | 115 | rti rmi·012345 rsu·012345 | Unentered branches - ray_mesh_interact's own glued composite against the redo |
| 16376–16434 | 59 | 120 | 0.002 | 32 | acoustic | 408 | rti rmi·345 rsu·345 | Unentered branches - a stack of a coherent and a mass-law layer against the laminate meshed as a solid with a lining |
| 16436–16571 | 136 | 227 | 0.003 | 41 | wave | 0 | rti rmi·345 rsu·345 | Unentered branches - the followed mark and the resolved flag across abutting and nested and overlapping bodies |
| 16573–16668 | 96 | 60 | 0.000 | 23 | wave | 0 | rti rmi·345 rsu·345 | Unentered branches - a verdict handed in on a leg known not parallel and near a pole |
| 16670–16898 | 229 | 1054 | 0.016 | 273 | acoustic | 3796 | rti rmi·45 rsu·45 | Mass cosine - a mass-law sheet behind a refracting layer forms its term at the cosine the wave has in air |
| 16900–17222 | 323 | 1742 | 0.019 | 394 | mixed | 5102 | rti rmi·1245 rsu·1245 | Closed plate - a mass-law layer the wave is evanescent in passes its two leaks and nothing else |
| 17224–17555 | 332 | 1292 | 0.021 | 438 | acoustic | 23319 | rti rmi·012345 rsu·012345 | Held lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid |
| 17557–17778 | 222 | 2776 | 0.022 | 574 | mixed | 1664 | rti rmi·012345 rsu·012345 | Stack geometry - an ordered stack leaves the ray on the line it arrived on |
| 17780–18102 | 323 | 1672 | 0.024 | 740 | mixed | 96 | rti rmi·012345 rsu·012345 | Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more than it received |
| 18104–18521 | 418 | 1614 | 0.026 | 582 | mixed | 150 | rti rmi·012345 rsu·012345 | Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the characteristic matrix |
| 18523–19035 | 513 | 36846 | 0.284 | 1125 | mixed | 424 | rti rmi·0134 rsu·0134 | Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the invariant |
| 19086–19320 | 235 | 6991 | 1.189 | 669 | mixed | 48 | rti rmi·012345 rsu·012345 | Bare tf - a bare face with tf is a transformer where it is open and a flat leak where it is closed |
| 19322–19451 | 130 | 816 | 0.044 | 182 | wave | 0 | rti rmi·01345 rsu·01345 | Scalar closed face - a lossy box on a rigid wall reflects no more than it received and is its sheet |
| 19495–19879 | 385 | 25275 | 0.310 | 771 | mixed | 54414 | rti rmi·0134 rsu·0134 | Closed face - tf at a closed face is a pad and the two ports of a sheet on it add up |
| 19881–20113 | 233 | 1248 | 0.014 | 299 | wave | 0 | rti rmi·0134 rsu·0134 | One solver - a sheet and a stack are one sum and no mirror is clamped |
| 20115–20181 | 67 | 412 | 0.036 | 98 | wave | 0 | rti rmi·345 rsu·345 | Budget - a sheet or a stack inside a host is one wall on every port whether it is capped or not |
| 20183–20350 | 168 | 4256 | 0.745 | 1744 | wave | 0 | rti rmi·012345 rsu·012345 | Resonant medium - a permittivity resonance is part of the medium a wave crosses and travels in |

#### `test_ray_progress.cpp`

| lines | n | asserts | s | out | class | m>0 | calls | case |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 541–581 | 41 | 38 | 0.002 | 0 | wave | 0 | rmi·0 rsu·0 prog cube | ray_progress - single reflection |
| 583–611 | 29 | 33 | 0.001 | 0 | wave | 0 | rmi·2 rsu·2 prog cube | ray_progress - single transmission |
| 613–644 | 32 | 37 | 0.001 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - reflection and transmission in one call |
| 646–668 | 23 | 21 | 0.000 | 0 | wave | 0 | prog cube | ray_progress - no hit terminates the trace |
| 670–691 | 22 | 20 | 0.001 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - every ray hits but none survives the gain gate |
| 693–707 | 15 | 17 | 0.000 | 0 | wave | 0 | prog cube | ray_progress - max_no_interactions = 0 disables tracing |
| 713–749 | 37 | 60 | 0.005 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - reflection and transmission passes can be switched off |
| 751–812 | 62 | 40 | 0.004 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - per-ray counters gate the next generation |
| 818–872 | 55 | 84 | 0.002 | 0 | wave | 0 | prog cube | ray_progress - beam subdivision |
| 874–919 | 46 | 10 | 0.003 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - subdivision requires beam data and a positive limit |
| 921–948 | 28 | 127 | 0.000 | 0 | wave | 0 | prog cube | ray_progress - every ray subdivided |
| 950–984 | 35 | 105 | 0.000 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - beam buffers survive the assembly |
| 990–1058 | 69 | 26414 | 0.005 | 0 | wave | 0 | rmi·0 rsu·0 prog cube | ray_progress - compaction preserves per-ray results |
| 1060–1088 | 29 | 4 | 0.001 | 0 | wave | 0 | rmi·0 rsu·0 prog cube | ray_progress - the reported hit count is independent of subdivision |
| 1094–1161 | 68 | 25 | 0.000 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - multi-frequency |
| 1163–1185 | 23 | 25 | 0.003 | 0 | wave | 0 | rmi·3 rsu·3 prog cube | ray_progress - scalar layout |
| 1187–1220 | 34 | 15 | 0.001 | 0 | wave | 0 | rmi·2 rsu·2 prog cube | ray_progress - medium state across a slab |
| 1222–1260 | 39 | 138 | 0.001 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - sub-mesh partitioning does not change the result |
| 1266–1305 | 40 | 3334 | 0.001 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - a trace terminates and respects its limits |
| 1307–1347 | 41 | 1031 | 0.002 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - repeated calls are deterministic |
| 1349–1389 | 41 | 9 | 0.000 | 0 | wave | 0 | rmi·2 rsu·2 prog cube | ray_progress - accumulated path length |
| 1395–1494 | 100 | 4722 | 0.009 | 0 | wave | 0 | rti rmi·02 rsu·02 prog cube | ray_progress - delegated intersection matches the internal one |
| 1496–1533 | 38 | 3831 | 0.002 | 0 | wave | 0 | rti rmi·02 rsu·02 prog cube | ray_progress - delegated intersection across a multi-generation trace |
| 1535–1561 | 27 | 458 | 0.000 | 0 | wave | 0 | rti rmi·02 rsu·02 prog cube | ray_progress - delegated intersection ignores the acceleration structure |
| 1563–1581 | 19 | 18 | 0.000 | 0 | wave | 0 | rti prog cube | ray_progress - delegated intersection with max_no_interactions = 0 |
| 1583–1636 | 54 | 11 | 0.005 | 0 | wave | 0 | rti rmi·0 prog cube | ray_progress - delegated intersection validation |
| 1642–1773 | 132 | 7996 | 0.008 | 0 | wave | 0 | rti rmi·02 rsu·02 prog subf cube | ray_progress - delegated subdivision flag matches the internal one |
| 1775–1810 | 36 | 176 | 0.000 | 0 | wave | 0 | rti rmi·02 rsu·02 prog subf cube | ray_progress - the delegated flag count is the reported n_subdiv |
| 1812–1854 | 43 | 72 | 0.007 | 0 | wave | 0 | rti rmi·0 rsu·0 prog subf cube | ray_progress - the delegated flag overrides the geometry |
| 1856–1895 | 40 | 12046 | 0.006 | 0 | wave | 0 | rti rmi·02 rsu·02 prog subf cube | ray_progress - delegated subdivision flag across a multi-generation trace |
| 1897–1941 | 45 | 10 | 0.009 | 0 | wave | 0 | rti rmi·02 rsu·02 prog subf cube | ray_progress - delegated subdivision flag validation |
| 1953–1992 | 40 | 53 | 0.000 | 0 | wave | 0 | rti prog subf cube | ray_progress - a flagged ray that misses the mesh is still subdivided |
| 1994–2028 | 35 | 58 | 0.001 | 0 | wave | 0 | prog cube | ray_progress - the end-of-trace signal accounts for subdivision |
| 2030–2069 | 40 | 14 | 0.001 | 0 | wave | 0 | rti prog subf cube | ray_progress - a flagged ray past its subdivision limit is still subdivided |
| 2071–2115 | 45 | 27 | 0.001 | 0 | wave | 0 | rti prog subf cube | ray_progress - a flagged ray inside a medium inherits its propagation state |
| 2117–2152 | 36 | 30 | 0.000 | 0 | wave | 0 | prog cube | ray_progress - the outside-a-medium branch still recomputes and clears |
| 2154–2205 | 52 | 560 | 0.000 | 0 | wave | 0 | rti rmi·0 rsu·0 prog cube | ray_progress - hit and subdivision counts are independent |
| 2211–2398 | 188 | 44 | 0.007 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - input validation |
| 2399–2414 | 16 | 2 | 0.000 | 0 | wave | 0 | rmi·02 rsu·02 prog cube | ray_progress - the material count is limited by the 14-bit state index |
| 2416–2801 | 386 | 827 | 0.032 | 297 | mixed | 17 | rmi·345 rsu·345 prog | ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has |
| 2803–3042 | 240 | 864 | 1.383 | 398 | wave | 0 | rmi·345 rsu·345 prog xprc | Measurement - ray_progress follows the coupling between the series of a layered wall |
| 3044–3245 | 202 | 184 | 0.125 | 69 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - a declined series at a sheet is followed and not dropped |
| 3247–3563 | 317 | 1628 | 1.673 | 529 | wave | 0 | rmi·345 rsu·345 prog xprc | Measurement - ray_progress and the mirror behind a back reflection |
| 3565–3806 | 242 | 979 | 8.951 | 249 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - the reflect port of a sheet lies where its transmit port puts the layer |
| 3808–4016 | 209 | 480 | 0.037 | 127 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - a lossy medium has no critical angle and a band takes one path |
| 4018–4197 | 180 | 1080 | 0.052 | 403 | acoustic | 21862 | rmi·345 rsu·345 prog xprc | ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band |
| 4199–4430 | 232 | 1924 | 0.752 | 534 | acoustic | 4446072 | rmi·345 rsu·345 prog xprc | ray_progress - the bonded plate at any depth where the slopes agree |
| 4432–4683 | 252 | 320 | 0.009 | 129 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - an ordered stack leaves the ray on the line it arrived on |
| 4685–4796 | 112 | 528 | 0.008 | 124 | wave | 0 | rmi·1245 rsu·1245 prog xprc | ray_progress - a coherent sheet with tf that is rarer than the invariant passes no more than it received |
| 4798–4940 | 143 | 344 | 0.008 | 64 | wave | 0 | rmi·1245 rsu·1245 prog xprc | ray_progress - a coherent sheet rarer than the invariant without tf tunnels |
| 4942–5100 | 159 | 752 | 0.101 | 228 | wave | 0 | rmi·012345 rsu·012345 prog xprc | ray_progress - a coherent sheet with tf creates nothing and has no gate on the invariant |
| 5102–5245 | 144 | 908 | 0.086 | 233 | wave | 0 | rmi·012345 rsu·012345 prog xprc | ray_progress - a traced solid with tf creates nothing |
| 5247–5367 | 121 | 72 | 0.028 | 29 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - a wall with a stack inside it is on its matrix with the paths of a single sheet |
| 5369–5487 | 119 | 240 | 0.002 | 44 | wave | 0 | rmi·345 rsu·345 prog xprc | ray_progress - a resonant solid is its medium at every frequency of a band |

#### `test_ray_mesh_interact.cpp`

| lines | n | asserts | s | out | class | m>0 | calls | case |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 464–602 | 139 | 23 | 0.000 | 0 | wave | 0 | rti rmi·01 cube | Ray-Mesh Interact - Air to Air (x-z plane) |
| 604–745 | 142 | 22 | 0.000 | 0 | wave | 0 | rti rmi·012 cube | Ray-Mesh Interact - Air to Dielectric Medium (x-z plane) |
| 747–906 | 160 | 26 | 0.000 | 0 | wave | 0 | rti rmi·012 cube | Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float) |
| 908–1024 | 117 | 14 | 0.000 | 0 | wave | 0 | rti rmi·02 cube | Ray-Mesh Interact - Medium to Medium (x-y plane, double) |
| 1026–1138 | 113 | 9 | 0.000 | 0 | wave | 0 | rti rmi·02 mg cube | Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double) |
| 1140–1187 | 48 | 4 | 0.000 | 0 | wave | 0 | rti rmi·2 cube | Ray-Mesh Interact - fRef parameterization equivalence |
| 1189–1223 | 35 | 1 | 0.000 | 0 | wave | 0 | rti rmi·1 cube | Ray-Mesh Interact - Penetration loss frequency scaling |
| 1225–1270 | 46 | 2 | 0.000 | 0 | wave | 0 | rti rmi·0 mg cube | Ray-Mesh Interact - Alpha in-medium absorption (medium_gain) |
| 1272–1316 | 45 | 13 | 0.000 | 0 | wave | 0 | rti rmi·012 cube | Ray-Mesh Interact - Permeability defaults to 1 (backward compatible) |
| 1318–1366 | 49 | 5 | 0.000 | 0 | wave | 0 | rti rmi·02 cube | Ray-Mesh Interact - Permeability decouples reflection from refraction |
| 1368–1408 | 41 | 3 | 0.000 | 0 | wave | 0 | rti rmi·01 cube | Ray-Mesh Interact - Permeability impedance-matches a dense medium |
| 1410–1446 | 37 | 2 | 0.000 | 0 | wave | 0 | rti rmi·0 cube | Ray-Mesh Interact - eps<->mu swap leaves reflected power invariant |
| 1448–1493 | 46 | 3 | 0.000 | 0 | wave | 0 | rti rmi·0 mg cube | Ray-Mesh Interact - Permeability drives in-medium loss (medium_gain) |
| 1495–1571 | 77 | 14 | 0.000 | 0 | wave | 0 | rti rmi·01234 cube | Ray-Mesh Interact - path_dirN direction contract (45° dielectric, x-z plane) |
| 1573–1636 | 64 | 9 | 0.002 | 0 | wave | 0 | rti rmi·1 cube | Ray-Mesh Interact - ray_indN compaction round-trip |
| 1638–1721 | 84 | 11 | 0.001 | 0 | wave | 0 | rti rmi·345 cube | Ray-Mesh Interact - Scalar interactions (types 3/4/5) gains, invariant, interface gain |
| 1723–1788 | 66 | 12 | 0.000 | 0 | wave | 0 | rti rmi·01 cube | Ray-Mesh Interact - Transmission factor splits reflection/transmission energy |
| 1790–1842 | 53 | 6 | 0.000 | 0 | wave | 0 | rti rmi·2 cube | Ray-Mesh Interact - Ray-tube TIR straddle forces whole-tube pass-through |
| 1844–1892 | 49 | 11 | 0.000 | 0 | wave | 0 | rti rmi·012345 cube | Ray-Mesh Interact - path_dirN collapses to undeviated under spine TIR |
| 1894–1929 | 36 | 2 | 0.000 | 0 | wave | 0 | rti rmi·12 cube | Ray-Mesh Interact - Spherical ray-tube keeps undeviated directions on pass-through |
| 1931–1977 | 47 | 8 | 0.000 | 0 | wave | 0 | mg ig ri | Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain |
| 1979–2097 | 119 | 11 | 0.000 | 0 | wave | 0 | rti rmi·01 cube | Ray-Mesh Interact - out_typeN bit encoding |
| 2099–2152 | 54 | 15 | 0.000 | 0 | wave | 0 | rti rmi·0 cube | Ray-Mesh Interact - compact = false keeps missed rays in place |
| 2161–2243 | 83 | 112 | 0.000 | 17 | wave | 0 | rti rmi·02 | Diagnostic - Interface energy split per polarization |
| 2245–2405 | 161 | 1672 | 0.003 | 197 | wave | 0 | rti rmi·1245 | Diagnostic - Transmission convention baseline, single interface |
| 2407–2580 | 174 | 640 | 0.001 | 51 | wave | 0 | rti rmi·01 | Diagnostic - Thin sheet composite vs analytic slab |
| 2582–2752 | 171 | 364 | 0.006 | 53 | wave | 0 | rti rmi·01 | Thin sheet - permeability and the lumped columns through ray_mesh_interact |
| 2754–2817 | 64 | 28 | 0.000 | 10 | wave | 0 | rti rmi·1 | Diagnostic - In-slab cosine projection, two ways |
| 2870–2924 | 55 | 90 | 0.001 | 14 | wave | 0 | rti rmi·01 | Thin sheet - grazing incidence does not depend on the face |
| 2926–2980 | 55 | 49 | 0.004 | 10 | wave | 0 | rti rmi·01 | Thin sheet - mesh subdivision invariance |
| 2982–3043 | 62 | 42 | 0.000 | 16 | wave | 0 | rti rmi·01 | Diagnostic - Derived lengths and the tangency limit |
| 3173–3315 | 143 | 164 | 0.001 | 20 | acoustic | 42 | rti rmi·01 | Thin sheet - mass-law angle dependence |
| 3317–3354 | 38 | 9 | 0.000 | 8 | wave | 0 | mg ri | Resonance - the public index and in-medium gain carry the permittivity pole |

#### `test_acoustic_interactions.cpp`

| lines | n | asserts | s | out | class | m>0 | calls | case |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 99–113 | 15 | 16 | 0.000 | 0 | wave | 0 |  | Acoustic - frequency mapping round-trip |
| 118–156 | 39 | 5 | 0.000 | 0 | wave | 0 | ri | Acoustic - refractive_index carries the resonance, not the coincidence |
| 161–212 | 52 | 6 | 0.000 | 0 | acoustic | 4 | mg | Acoustic - medium_gain in-medium loss |
| 217–281 | 65 | 7 | 0.000 | 0 | wave | 0 | ig | Acoustic - interface_gain penetration and coincidence |
| 286–317 | 32 | 3 | 0.001 | 0 | wave | 0 | rti rmi·3 cube | Acoustic - permittivity resonance absorption peak |
| 322–356 | 35 | 5 | 0.103 | 0 | wave | 0 | rti rmi·34 rsu·4 mg cdg cube | Acoustic - convergence with acoustic terms absent |
| 361–388 | 28 | 2 | 0.000 | 0 | wave | 0 | rti rmi·34 rsu·4 mg cdg cube | Acoustic - convergence away from spectral features |
| 394–442 | 49 | 12 | 0.001 | 0 | wave | 0 | rti rmi·34 cube | Acoustic - scalar total internal reflection |
| 448–516 | 69 | 10 | 0.001 | 0 | wave | 0 | rti rmi·4 rsu·4 mg cdg cube | Acoustic - pass-through calibration |
| 522–553 | 32 | 2 | 0.000 | 0 | wave | 0 | rmi·4 rsu·4 cdg cube | Acoustic - embedded material counted once |
| 561–609 | 49 | 14 | 0.001 | 0 | wave | 0 | rmi·4 rsu·4 cdg cube | Acoustic - thin-slab Fabry-Perot resolution |

#### `test_sheet_vs_solid.cpp`

| lines | n | asserts | s | out | class | m>0 | calls | case |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| 453–477 | 25 | 0 | 0.000 | 0 | wave | 0 | cube | Mesh builder conventions |
| 479–518 | 40 | 12 | 0.000 | 0 | wave | 0 |  | Analytic reference reduces to the Airy sum |
| 524–553 | 30 | 61 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Isolated sheet vs traced solid, 10 cm concrete |
| 559–588 | 30 | 1 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Lossless slab, sheet vs solid |
| 594–641 | 48 | 1 | 0.001 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Material sweep, sheet vs solid |
| 647–679 | 33 | 1 | 0.001 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Thickness sweep, sheet vs solid |
| 685–725 | 41 | 2 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg | Collapsed duplicate equals a single sheet |
| 731–783 | 53 | 1 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg | Corner with sheet degrades to a single hit |
| 789–856 | 68 | 70 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Glued sheet equals a solid of the summed thickness |
| 862–934 | 73 | 30 | 0.001 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Sheet embedded in a solid equals a solid of the summed thickness |
| 940–970 | 31 | 4 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg | Ordered stack of two different sheets is killed |
| 976–1029 | 54 | 1 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Interface gain is applied once on each path |
| 1035–1076 | 42 | 10 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | Grazing incidence, sheet vs solid |
| 1082–1126 | 45 | 20 | 0.000 | 0 | acoustic | 98 | rmi·4 rsu·4 objr cdg cube | Acoustic sheet, mass-law route |
| 1132–1158 | 27 | 1 | 0.000 | 0 | wave | 0 | rmi·1 rsu·1 objr cdg cube | thk = 0 is identical to no sheet at all |
