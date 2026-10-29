# SSF code review, round 2 (cluster level, series 408c10e … e266efc)

Scope: `git diff 26da9f93..HEAD -- . ':!md'` at HEAD e266efc, reviewed in full (not only the round-1 fixes).
Reference: `md/ssf_spec.md`, `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`, TR 38.901 v19.5.0.

How measured (load average 0.8–1.4 throughout; no timings reported):
- Catch2 `build_ssf/test_bin` (up to date with HEAD; no source newer than the binary), run from the worktree root,
  `OMP_NUM_THREADS=10`, filter `ssf*,lsf*,*v19*`: **All tests passed (34634 assertions in 86 test cases)**.
- Python module `lib/` (up to date): `test_ssf_clusters.py`, `test_ssf_draw.py`, `test_ssf_init_phases.py`: 24 passed.
- `-Wall -Wextra -fsyntax-only` on the five SSF/phase test files: no warning.
- `tools/extract_md.py` regenerated into scratch: identical to the three `html_docu/*.md` files (docs in sync).
- Scratch in `/home/stephan/quadriga-lib-agents/scratch/ssf_cr2/`: `p1.cpp` (edge cases against `libquadriga.a`),
  `fuzz.py` (900 calls: 5 v19 files × 3 methods × GR off/on × absTOA defined/legacy × 15 draws, 7 links, 1–3
  frequencies, K −30…30 dB, DS 0.3 ns–1 µs, spreads 0.1–160°, Normal values ×15 on 5 %, delay values 0 on 4 %,
  random per-link `n_cluster`, `abs_toa` with a 0 entry, O2I, EoD offset), `fuzz2.py`, `fuzz3.py` (library inputs
  from `lsf_draw`/`lsp_generate`/`ssf_draw`, 2000 independent links).

Counts: **0 bugs, 2 major, 6 minor.**

---------------------------------------------------------------------------------------------------------------------

## A. Bugs

None. The fuzz holds every invariant in all 900 calls: outputs finite, azimuths in [−π, π), elevations in [−π/2, π/2],
powers and delays ≥ 0, inactive slots (n ≥ `n_cluster(l)`) at delay 0 / power 0, and for `quadriga` in NLOS the first
active NLOS delay equal to `abs_toa(l)` (legacy scenarios: only where `abs_toa(l) > 0`). B1/M1 of round 1 are closed:
one active cluster gives delay 0 (or `abs_toa`), measured in RMa NLOS/LOS with library inputs (fuzz3, `N_l = 1`:
span 0 for all 2000 links).

## B. Major (documentation, the maintainer's main concern)

### M1. MEX `ssf_clusters`: wrong block numbers for the spatial-consistency rule

`api_mex/ssf_clusters.cpp:21`: "evaluate the blocks **2 to 9** of `rnd` at the drop or segment start". The MEX
`ssf_draw` block (`api_mex/ssf_draw.cpp:20–23`) numbers the blocks `b = 0 ... 8` (block `b` = columns `b·N + 1` to
`(b + 1)·N`), and the MEX `ssf_clusters` slot bullet (line 29–30) uses the same `b·N + n`. A MATLAB user who follows
the text freezes "blocks 2 to 9": the AOD-sign block 1 stays unfrozen and "block 9" does not exist. C++ and Python
say "blocks 1 to 8" (correct).
Fix (line 21): `>= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of \`rnd\` (columns`
`N + 1 to 9·N) at the drop or segment`.

### M2. Which method uses `eod_offset_deg`, `o2i` (and needs `abs_toa`) is lost in MEX/Python and misplaced in C++

The argument lists say "NLOS scenarios" for `eod_offset_deg` and `o2i` but not which method uses them; only the C++
table row "TR ZOD offset, O2I arrival elevation" carries it. MEX and Python have no table (they point to the C++ doc,
which a MATLAB/Python reader or an agent reading `quadriga_lib_api_mex.md` / `_python.md` does not have), so the facts
"`38.901_MF` ignores both, `quadriga` ignores `o2i`" are gone there. The `quadriga` requirement on `abs_toa` sits in
the description, while `rnd_freq`/`lsp_anchor` carry their "required by" in the argument list (round-1 rule:
"required by" / "used by" in the argument lists; task: "method requirements on optional inputs").
Fix, the same in the three blocks (C++ `src/ssf_clusters.cpp:434–440`, MEX `api_mex/ssf_clusters.cpp:71–78`, Python
`api_python/qpy_ssf_clusters.cpp:69–77`; MEX/Python keep their own quoting, "omitted or empty"/`None` and units):
- `eod_offset_deg` — `EoD offset in [deg] from [[lsp_generate]]; used by "38.901" and "quadriga" in NLOS scenarios;`
  `omitted = 0; [n_link, n_freq]`
- `abs_toa` — `Absolute TOA offset in [s] from [[lsp_generate]], >= 0; required by "quadriga" in NLOS scenarios with`
  `absTOA_mu > -30; omitted = no offset; [n_link]`
- `o2i` — `O2I flag per link (mean ZOA 90 deg); used by "38.901" in NLOS scenarios; omitted = none; [n_link]`

and in the description, abs-TOA bullet (C++ 381–383, MEX 38–40, Python 36–38), drop the requirement sentence part:
`- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: "38.901", "38.901_MF" add abs_toa(l) to the NLOS delays;`
`  "quadriga" places the first NLOS cluster at abs_toa(l) (where absTOA_mu <= -30: only where abs_toa(l) > 0)`.
C++ table: delete the row `| TR ZOD offset, O2I arrival elevation | both | neither | ZOD offset |` (now said once,
in the argument list). Net −1 line per block.

## C. Minor

- **m1. `NumClusters = 1` with GR in an NLOS scenario contradicts the NLOS bullet** (doc only; code follows spec 2.3).
  `p1.cpp`: UMa NLOS, `NumClusters = 1`, GR on, ε = 5 − 0.5j: LOS power 1.0000; GR power 0.3260 (`38.901`,
  `38.901_MF`), 0.3305 (`quadriga`). The bullet at `src/ssf_clusters.cpp:376` says "In an NLOS scenario the LOS and
  ground-reflection powers are 0 whatever `lsp` holds"; line 388 says "power 1 in all scenarios, plus the ground
  reflection" without its power. Fix (C++ 388, MEX 45, Python 43):
  `- NumClusters = 1: the LOS path only, power 1 in all scenarios (K/(K+1) = 1, also in the ground-reflection`
  `  power), plus the ground reflection if enabled; rnd, rnd_freq and lsp_anchor may be empty`
  and in the NLOS bullet (C++ 376, MEX 33, Python 31): `... powers are 0 whatever lsp holds (NumClusters > 1);`.
  The "may be empty" clause is a user-needed fact that is now nowhere (tested in "one cluster gives the LOS path only"
  and "each method ignores the inputs it does not use").
- **m2. C++ table, row "Output spreads equal the input spreads"** (`src/ssf_clusters.cpp:368`): for `quadriga` it holds
  without ground reflection only; with GR the DS from the slot powers is below the input (diag of "quadriga delay
  spread equals the input at one frequency": median 0.8826 in UMi LOS with GR, 1.0000 without; the source comment at
  line 206–209 says so). Fix: `| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one`
  `frequency without ground reflection |`. Same table, row "Same delays and angles at all frequencies", `"38.901"`
  cell: `one frequency` → `n/a` (the method takes one frequency).
- **m3. Stale oracle rule** `tests/catch2_tests/test_ssf_clusters_quadriga.cpp:157`: `ref_qd` still uses the
  pre-fix fallback `: 1.0` for `den = 0` (the library uses 0 since the fix, line 284 of `src/ssf_clusters.cpp`). No
  current test reaches it (all `ref_qd` calls have `N_l >= 6`), but the oracle encodes the old defect. Fix: `: 0.0`.
- **m4. Header comment points at a source file** `include/quadriga_gbsm.hpp` (SSF block): "The column layout of the
  random values is documented in src/qd_gbsm_phases.cpp (ssf_draw)". The public header should name the API doc, not
  a source path. Fix: `... The column layout of the random values: see the documentation of ssf_draw`.
- **m5. Repetition in the source comment of `src/qd_gbsm_phases.cpp`** (lines 34–66, 33 lines): the generator table
  repeats the `ssf_draw` MD block of the same file (block, variable, distribution), and the SplitMix64 formula
  (lines 55–58) repeats `derive_seed` 300 lines below, which is the code itself. Only the mirror column m(b) and the
  tags are not visible elsewhere, and both are in `qd_gbsm_gen_ssf` (`mirror_b`) and `derive_seed`. Optional cut to
  ~10 lines: keep the purpose line, "N = NumClusters − 1; column b·N + n = block b, cluster n; blocks and
  distributions: ssf_draw documentation; mirror partner: qd_gbsm_gen_ssf; all generators at SC_lambda (0 → 0.1 m),
  none bias-corrected", the seed rule in one sentence ("SplitMix64 of the LSF seed and the set tag, see
  derive_seed; −1 passes on"), the N = 0 line and the append-only rule.
- **m6. Tests: duplicated constants; size.** `N_UMi_LOS = 12, N_UMi_NLOS = 19` are defined in all three cluster test
  files (`test_ssf_clusters.cpp:69`, `_mf.cpp:54`, `_quadriga.cpp:53`): move to `qd_test_ssf.hpp`. Cluster tests are
  2445 lines (target ≈ 2300). The two `Diagnostic -` cases assert only finiteness (`test_ssf_clusters.cpp:986–1007`,
  22 lines; `test_ssf_clusters_quadriga.cpp:719–755`, 37 lines) and guard no behaviour; if the 2300 target matters,
  delete them (−59 lines; the quadriga drop ratios are covered by the exact DS/AS cases, the 38.901 track rule is the
  caller's). Nothing else is worth cutting: the remaining size is oracles typed independently of the library and
  method-generic loops.

## C'. Checks without finding

- **Coverage against da9c787.** Every test case of the old four files maps to a case now (merged: sizes/slot layout
  + MF sizes → "output sizes, slot layout and the ground-reflection slot" for all methods; the three "one cluster",
  "extreme values", "ignores inputs", "zero LOS power" copies → one method loop each; quadriga 1-/multi-frequency →
  one table-driven case; swap 38.901 + MF → one case). Validation: 32 of 34 old messages are in the new table; the
  two dropped (`'rx_pos' must be finite`, rows of `tx_pos`) are the `qd_gbsm_check_links` pass-through decision of
  round 1, as are the dropped `r_DS`/`NumClusters_min` scenpar messages. Old `ssf_init_phases`/`ssf_draw` and
  `lsf_*` cases: same names, same sections, same assertion count (110 and 124). Assertion sites in the cluster
  files 376 → 213 come from the table and the method loops, not from lost checks; GR slot and link independence now
  cover all methods.
- **Implementation notes moved into the source**: `qd_ssf_tr` (K limit, removal, ρ once), `qd_ssf_mf` (literal
  points of 7.6.5.1, wrap-before-scale), `qd_ssf_quadriga` (algorithm, changes against the paper, properties), GR
  block (geometry, zero-denominator rule), abs-TOA rule at the validation and at the shift, B1 rule at the delay
  scale. Each sits at the code it explains and matches it (read line by line: closed-form delay scale = larger root
  of var = DS², caps after averaging, EoD offset on NLOS departures only, ω = 0 in NLOS, cos θ / sin² θ with d_GR').
  Not bloated (the quadriga header is 16 lines for a 130-line kernel).
- **TR fidelity**: unchanged since round 1 apart from the documented fixes; re-read the diff of e266efc: ρ once per
  cluster, −25 dB threshold constant, `centre` only for the TR methods, GR weight written only with GR — no
  behaviour change (bitwise tests pass).
- **Validation, edge cases**: `NumClusters = 1` (all methods, GR on/off, LOS/NLOS), zero links (C++/Python 0 rows;
  MEX rejects, documented), per-link N, GR on/off, abs-TOA rules (defined/legacy, 0 entries, LOS ignored), K = 0,
  extreme values (fuzz). OK apart from m1 (doc).
- **Side observation (measured, no action)**: `quadriga` matches DS exactly, so with very few active clusters in LOS
  a weak cluster can be placed far out: RMa LOS, library inputs, 2000 links, (max − min NLOS delay)/DS p99/max:
  `N_l = 2` 135/700 (`38.901`: 45/150), `N_l = 4` 79/449 (57/150). At the TR ranges of Table 7.6.15-1 (D1 ≥ 6) the
  tails are like the TR's (UMi LOS N_l = 6: max 171 vs 208; UMa LOS N_l = 10: 113 vs 111). Only RMa (no
  `NumClusters_min`) and user-chosen tiny `n_cluster` reach it; engine-limit list, not this series.
- **OpenMP**: per-link state, read-only inputs, no throw inside the region (all checks before), `R2.data()` of an
  empty vector is never dereferenced without GR. OK.
- **MSVC / rules**: no VLAs, builtins, `M_PI`, `std::to_chars`; `<cstdint>` for `uint64_t`; no BLAS/LAPACK in
  library or tests; helper scoping (kernels file-local, single-use lambdas, `qd_gbsm_gen_ssf` shared by two functions
  in one file, wrapper helpers in the documented private headers). OK.
- **References**: no `md/` path, tracking id or internal remark in any produced file ("D1" in the conf comments is
  TR Table 7.6.15-1's own name). OK.
- **Wrappers vs docs**: MEX `set_id` shift (0 shared, k ≥ 1), `n_cluster` without shift, `floor_id` labels; Python
  `gr_eps` complex with imaginary part ≥ 0 used as `re − j·im` (a sign error in my first fuzz run was rejected with
  the documented message), 3-D `gr_eps` rejected, `o2i` bool/numeric, `None`/empty handling; defaults in the
  bindings match the docs. OK apart from M1.
- **SECTION_DESC** (three files): first paragraph without the output list, new 5/6-line paragraph identical in
  content. Keep.

---------------------------------------------------------------------------------------------------------------------

## D. Documentation, per block

| Block | Lines C++ / MEX / Python | Verdict |
|---|---|---|
| `ssf_init_phases` | 39 / 34 / 34 | keep; optional: drop `n_gen = 9·N (shared set) or N` from the output line (the description already gives `9·N`/`N` per set) or vice versa — the only repetition |
| `ssf_draw` | 51 / 43 / 43 | keep |
| `ssf_clusters` | 106 / 79 / 80 | fix M1 (MEX), M2 (all), m1 (all), m2 (C++); after M2 −1 line each |
| SECTION_DESC | 6 / 7 / 6 new lines | keep |
| Header SSF comments | 3 + per-argument | m4 |

Nothing a user needs was lost in the 1248 → 509 cut, except the two items now in M2 (method use of `eod_offset_deg`,
`o2i` for MEX/Python) and m1 ("`rnd`, `rnd_freq`, `lsp_anchor` may be empty with `NumClusters = 1`"). Checked
present: `rnd` column layout `b·N + n` with the block meanings (ssf_draw, all three APIs, MEX 1-based columns
correct), `set_id` meanings and the MEX shift, slot layout and slot ↔ column identity, inactive slots, LOS-slot rule,
units, `n_fd`, LOS condition, coefficient powers and the GR consumer rule, swap conditions, ranges and sizes.

## E. Verdict

No third full round. The fixes are documentation text (M1, M2, m1, m2, m4: about 15 lines in four files plus
`make documentation`) and one-line test edits (m3, m6); m5 is optional. A short check of that diff by the maintainer
(or a quick targeted pass) suffices; code and tests need no re-review.
