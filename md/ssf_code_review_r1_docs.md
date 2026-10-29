# SSF code review r1 — appendix: de-bloated documentation texts

Companion to `md/ssf_code_review_r1.md` (section D). Every block below is the proposed full replacement of the
`/*!MD ... MD!*/` content (C++), resp. the MEX and Python blocks. Line counts are of the block between `/*!MD` and
`MD!*/`. Facts that leave the blocks are listed in D.2 of the main report with their new place (a source comment next
to the code, or deleted).

Rules applied: the description says what the function does and what it is for; the argument lists hold shapes,
units, ranges, defaults and "required by"; each fact appears once (layout of the random values: `ssf_draw` only;
mobility, floors, precision: by reference to `lsf_draw`; the SC rule of `38.901`: `ssf_clusters` only); no
formulas except where the user must compute something (the GR coefficient); no measured study numbers (they are
not published); no validation-order lines (the messages name the argument; the ranges are in the argument lists).

---------------------------------------------------------------------------------------------------------------------

## 1. C++ `ssf_init_phases` (src/qd_gbsm_phases.cpp:675–739) — 65 → 41 lines

```
/*!MD
# ssf_init_phases
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The cluster random values of the small-scale fading (TR 38.901 clause 7.6.3.1) come in two kinds of sets, with
  `N = NumClusters - 1` NLOS clusters: one shared cluster set (`set_id = -1`, `9·N` generators) and one set per
  carrier frequency (`set_id = k`, `N` generators, `k` = index in the caller's frequency list, the same order in
  every call). [[ssf_draw]] evaluates the phases at the links and lists the columns
- The phases are a fixed function of `seed`, the seed of the large-scale set of the same site and scenario
  ([[lsf_init_phases]]), and `set_id`, equal on all platforms; the sets do not repeat the large-scale phases and are
  independent of each other. Keep an explicit seed (or the phase cubes) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes the link
  types uncorrelated
- `NumClusters = 1`: both sets are empty

## Declaration:
```
void quadriga_lib::ssf_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    arma::cube &phase,
    arma::sword seed = -1,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256");
```

## Inputs:
- **`scenpar`** — Scenario parameters from [[conf_file_read]] or [[scenpar_validate]], missing keys take their
  defaults; `NumClusters` sets `N`
- **`seed`** *(optional)* — Seed of the large-scale set, >= -1; -1 = system random device (new phases per call)
- **`set_id`** *(optional)* — `-1` = shared cluster set; `k >= 0` = set of frequency `k`
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`

## Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side (mirror generators);
  `[n_coeff, n_gen, 2]`, `n_gen = 9·N` (shared set) or `N`

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
MD!*/
```

## 2. C++ `ssf_draw` (src/qd_gbsm_phases.cpp:793–876) — 84 → 53 lines

```
/*!MD
# ssf_draw
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

- Evaluates a set of [[ssf_init_phases]] at the links with the rules of [[lsf_draw]] for single and dual mobility,
  floors, 2D/3D mode and precision; every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table
  7.6.3.1-2; 0 = spatial consistency off, used as 0.1 m)
- Columns, `N = NumClusters - 1`; column `b·N + n` belongs to block `b` and NLOS cluster `n`:
  - shared set (`set_id = -1`, `9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA;
    3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`set_id = k`, `N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly (D2D reciprocity of S. Jaeckel et al., "A Spatially Consistent Geometric
  D2D Small-Scale Fading Model for Multiple Frequencies", 2019)
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `preset`, `horizontal` and the set type must match [[ssf_init_phases]]. `NumClusters = 1` gives `[n_link, 0]`
  (any empty `phase` is accepted)

## Declaration:
```
void quadriga_lib::ssf_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    arma::sword set_id = -1,
    const arma::uvec *floor_id = nullptr,
    bool dual_mobility = false,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters as in [[ssf_init_phases]]; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [[ssf_init_phases]], finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m], one row for all links; used in dual mobility; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** *(optional)* — `-1` = shared cluster set; `k >= 0` = per-frequency set
- **`floor_id`** *(optional)* — Floor index per link as in [[lsf_draw]], 2D mode only; omitted = floor 0; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: the Tx is a second moving device
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]
- **`horizontal`** *(optional)* — `true`: 2D correlation (TR 38.901); `false`: 3D (legacy QuaDRiGa)

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]` (`[0, n_gen]` for no link)

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values; mobility, floor and precision rules)
MD!*/
```

## 3. C++ `ssf_clusters` (src/ssf_clusters.cpp:320–591) — 272 → 98 lines

```
/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [[lsp_generate]] and the random values of [[ssf_draw]] into the clusters of each link, by one of
  three methods:
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Jumps
    along tracks; for spatial consistency (clause 7.6.3.1) evaluate the sign and offset blocks 1 to 8 of `rnd` at
    the drop or segment start and keep them along the track. Clusters 25 dB below the strongest get power 0 (no
    renormalisation); the K-factor in the scaling factors is limited to >= -5 dB
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, anchor spreads from `lsp_anchor`; in LOS the delays and
    angles depend on the frequency through K
  - `"quadriga"`: the method of S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model
    for Multiple Frequencies", 2019, with the LOS direction added to the cluster offsets: continuous along tracks,
    the same delays and angles at all frequencies, exact delay spread at one frequency without ground reflection,
    no per-cluster shadowing;
    for tracks, several frequencies and device-to-device links
- Slots per link: 0 = LOS path, 1 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. Slot `1 + G + n` belongs to column `n` of each block of `rnd` (no
  sorting). Clusters `n >= n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds; legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 0 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa(l)` to the NLOS
  delays; `"quadriga"` shifts them so that the first is at `abs_toa(l)`
- Ground reflection (TR clause 7.6.8) on the plane z = 0. Slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`) with `R² = (abs(R_par)² + abs(R_perp)²)/2` of TR eqs. (7.6-38),
  (7.6-39). The coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled
- Swap reciprocity: with mirrored random values (dual mobility in [[ssf_draw]]) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`"38.901"`: `eod_offset_deg` = 0 and no O2I link;
  `"quadriga"`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (only sizes are checked); `scenpar` is
  validated by [[scenpar_validate]]. Outputs must not share memory with the inputs

## Declaration:
```
void quadriga_lib::ssf_clusters(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::cube &lsp,
    const arma::mat &rnd,
    arma::cube &delay,
    arma::cube &power,
    arma::cube &aod,
    arma::cube &eod,
    arma::cube &aoa,
    arma::cube &eoa,
    const std::string &method = "38.901",
    const arma::cube *rnd_freq = nullptr,
    const arma::cube *lsp_anchor = nullptr,
    const arma::mat *eod_offset_deg = nullptr,
    const arma::vec *abs_toa = nullptr,
    const arma::mat *gr_eps_re = nullptr,
    const arma::mat *gr_eps_im = nullptr,
    const std::vector<bool> *o2i = nullptr,
    const arma::uvec *n_cluster = nullptr,
    arma::uvec *n_active = nullptr);
```

## Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in [m], one row for all links; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [[ssf_draw]] (`set_id = -1`), columns 0 to N-1 in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `"38.901"` (default), `"38.901_MF"` or `"quadriga"`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [[ssf_draw]], slice `k` from `set_id = k`; required by
  `"38.901"` and `"38.901_MF"`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [[lsp_generate]]; required by
  `"38.901_MF"`; `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [[lsp_generate]]; used by `"38.901"` and
  `"quadriga"` in NLOS scenarios; omitted = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [[lsp_generate]], >= 0; omitted = 0; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from
  [[lsp_generate]], real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link; `"38.901"` only (mean ZOA 90 deg); omitted = none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`;
  omitted = N; `[n_link]`

## Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** *(optional)* — Number of NLOS clusters with power > 0; `[n_link]`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [[calc_delay_spread]], [[calc_angular_spreads_sphere]] (spreads of the output)
MD!*/
```

Note on the dropped method × property table (plan decision "a method × property table in the docs"): the per-method
bullets above keep its qualitative rows (continuity, frequency independence, exact spreads, shadowing, reciprocity
conditions, TR terms by argument); its measured ratios come from an unpublished study and cannot be cited. If the
maintainer wants the table kept, the compact form is 7 lines (header + 5 rows: continuity, equal at all frequencies,
exact DS, per-cluster shadowing, `eod_offset_deg`/`o2i` used) and replaces the method sub-bullets' property clauses.

---------------------------------------------------------------------------------------------------------------------

## 4. MEX `SSF_INIT_PHASES` (api_mex/ssf_init_phases.cpp) — 70 → 36 lines

Description as C++ 1, with: `set_id = 0` = shared set, `set_id = k >= 1` = set of the k-th frequency; output
slices 1 (Rx) and 2 (Tx).

```
/*!MD
# SSF_INIT_PHASES
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The cluster random values of the small-scale fading (TR 38.901 clause 7.6.3.1) come in two kinds of sets, with
  `N = NumClusters - 1` NLOS clusters: one shared cluster set (`set_id = 0`, `9·N` generators) and one set per
  carrier frequency (`set_id = k`, `N` generators, `k` = position in the caller's frequency list, the same order
  in every call). [[ssf_draw]] evaluates the phases at the links and lists the columns
- The phases are a fixed function of `seed`, the seed of the large-scale set of the same site and scenario
  ([[lsf_init_phases]]), and `set_id`, equal on all platforms; the sets do not repeat the large-scale phases and are
  independent of each other. Keep an explicit seed (or the phase arrays) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4 makes the link
  types uncorrelated
- `NumClusters = 1`: both sets are empty

## Usage:
```
phase = quadriga_lib.ssf_init_phases( scenpar, seed, set_id, preset );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct from [[conf_file_read]] or [[scenpar_validate]] or any subset of
  its fields (missing fields take their defaults, `SubpathMethod` is ignored); `NumClusters` sets `N`
- **`seed`** *(optional)* — Seed of the large-scale set; -1 = system random device; default: -1
- **`set_id`** *(optional)* — `0` = shared cluster set; `k >= 1` = set of frequency `k`; default: 0
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; default: `'Comb256'`

## Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 1 = Rx side, slice 2 = Tx side (mirror generators);
  `[n_coeff, n_gen, 2]`, `n_gen = 9·N` (shared set) or `N`

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
MD!*/
```

## 5. MEX `SSF_DRAW` (api_mex/ssf_draw.cpp) — 86 → 45 lines

```
/*!MD
# SSF_DRAW
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

- Evaluates a set of [[ssf_init_phases]] at the links with the rules of [[lsf_draw]] for single and dual mobility,
  floors, 2D/3D mode and precision; every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table
  7.6.3.1-2; 0 = spatial consistency off, used as 0.1 m)
- Columns, `N = NumClusters - 1`; block `b = 0 ... 8` holds the columns `b·N + 1` to `(b + 1)·N`, one per NLOS
  cluster:
  - shared set (`set_id = 0`, `9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA;
    3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`set_id = k >= 1`, `N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly (D2D reciprocity of S. Jaeckel et al., "A Spatially Consistent Geometric
  D2D Small-Scale Fading Model for Multiple Frequencies", 2019)
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `preset`, `horizontal` and the set type must match [[ssf_init_phases]]; at least one link is required

## Usage:
```
rnd = quadriga_lib.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, ...
    preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters as in [[ssf_init_phases]]; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [[ssf_init_phases]]; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m], one row for all links; used in dual mobility; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** *(optional)* — `0` = shared cluster set; `k >= 1` = per-frequency set; default: 0
- **`floor_id`** *(optional)* — Floor label per link starting at 0 (no index shift) as in [[lsf_draw]], 2D mode
  only; omitted or empty = floor 0; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: the Tx is a second moving device; default: false
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; default: `'Comb256'`
- **`horizontal`** *(optional)* — `true`: 2D correlation (TR 38.901); `false`: 3D (legacy); default: true

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values; mobility, floor and precision rules)
MD!*/
```

## 6. MEX `SSF_CLUSTERS` (api_mex/ssf_clusters.cpp) — 283 → about 95 lines

Description: the C++ text of 3 with the MATLAB indices (slot 1 = LOS path, slot 2 = ground reflection if
`G = 1`, NLOS cluster `n = 1 ... N` in slot `1 + G + n`, which belongs to column `n` of each block of `rnd`, i.e.
the columns `b·N + n`; "blocks 2 to 9" for the sign and offset blocks under spatial consistency; quotes `'38.901'`),
the last bullet ending "Outputs must not share memory ..." replaced by "At least one link is required". Usage, Inputs
and Outputs:

```
## Usage:
```
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, rx_pos, ...
    lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, o2i, n_cluster );
```

## Inputs:
- **`scenpar`** — Scenario parameters (struct, `SubpathMethod` ignored); used: `NumClusters`, `NumClusters_min`,
  `KF_mu`, `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m], one row for all links; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0, KF (slice 2) >= 0;
  `'38.901'`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [[ssf_draw]] (`set_id = 0`), columns 1 to N in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `'38.901'` (default), `'38.901_MF'` or `'quadriga'`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [[ssf_draw]], slice `k` from `set_id = k`; required by
  `'38.901'` and `'38.901_MF'`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz); required by `'38.901_MF'`;
  `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [[lsp_generate]]; used by `'38.901'` and
  `'quadriga'` in NLOS scenarios; omitted or empty = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [[lsp_generate]], >= 0; omitted or empty = 0;
  `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im`, real part >= 1,
  imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link, logical or numeric; `'38.901'` only; omitted or empty = none;
  `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link (a count) in
  `[min(max(1, NumClusters_min - 1), N), N]`; omitted or empty = N; `[n_link]`

## Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `'38.901_MF'`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `[n_link, 1]`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [[calc_delay_spread]], [[calc_angular_spread]] (spreads of the output)
```

## 7. Python `ssf_init_phases` (api_python/qpy_ssf_init_phases.cpp) — 61 → about 38 lines

Description bullets identical to C++ 1 (0-based `set_id`, `-1` = shared set). Usage, Inputs, Output:

```
## Usage:
```
phase = quadriga_lib.gbsm.ssf_init_phases( scenpar, seed, set_id, preset )
```

## Inputs:
- **`scenpar`** — Scenario parameters; dict from [[conf_file_read]] or [[scenpar_validate]] or any subset of its
  keys (missing keys take their defaults, `SubpathMethod` is ignored); `NumClusters` sets `N`
- **`seed`** — Seed of the large-scale set, int from -1 to 2^63-1; -1 = system random device; default: -1
- **`set_id`** — `-1` = shared cluster set; `k >= 0` = set of frequency `k`; default: -1
- **`preset`** — Sum-of-sinusoids preset, see [[sos.preset]]; default: `"Comb256"`

## Output:
- **`phase`** — Phases in rad, in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side (mirror generators);
  `(n_coeff, n_gen, 2)`, `n_gen = 9·N` (shared set) or `N`

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
```

## 8. Python `ssf_draw` (api_python/qpy_ssf_draw.cpp) — 81 → about 50 lines

Description bullets identical to C++ 2 (last bullet: "... must match [[ssf_init_phases]]. Zero links give
`(0, n_gen)`; `NumClusters = 1` gives `(n_link, 0)`"). Usage, Inputs, Output as the C++ lists with Python types
(`floor_id` "int or float array of non-negative integers", flags `bool`, defaults written out), shapes in
parentheses.

## 9. Python `ssf_clusters` (api_python/qpy_ssf_clusters.cpp) — 246 → about 95 lines

Description bullets identical to C++ 3 (0-based slots; "Outputs must not share memory" dropped; "Zero links give
outputs with 0 rows" added to the slot bullet). Argument list as C++ 3 with these Python differences only:
- **`gr_eps`** *(optional)* — Ground permittivity as [[lsp_generate]] returns it (complex array, real part >= 1,
  imaginary part >= 0; used as `real - j·imag`; a real array = imaginary part 0); required if `GR_enabled > 0`;
  `(n_link, n_freq)`
- **`o2i`** — any numeric or bool array, nonzero = O2I
- **`n_cluster`** — int or float array of non-negative integers
- outputs: float64, `n_active` uint64 `(n_link,)`

---------------------------------------------------------------------------------------------------------------------

## 10. SECTION_DESC (src/qd_scenpar.cpp:699–735, api_mex/conf_file_read.cpp, api_python/qpy_conf_file_read.cpp)

First paragraph: revert the list extension to one clause, 4 → 4 lines but without repeating the cluster outputs
that the new paragraph states:

```
... The functions of this section implement the model of 3GPP TR 38.901, clause 7.5 steps 1 to 7: path loss,
shadow fading, LOS probability, O2I penetration loss, the large-scale parameters (LSPs) of each link and the
clusters.<br><br>
```

New paragraph, 9 → 5 lines (C++; MEX and Python the same with their method quotes):

```
The small-scale part (clause 7.5 steps 5 to 7, cluster level): [[ssf_init_phases]] draws the phases of the cluster
random values per site, scenario and frequency from the LSF seed, [[ssf_draw]] evaluates them at the links, and
[[ssf_clusters]] turns the LSPs and random values into cluster delays, powers and angles with the method `38.901`
(clause 7.5), `38.901_MF` (clause 7.6.5.1) or `quadriga` (spatially consistent, frequency-independent). The ray
level (steps 8 to 12) follows in a later version.<br><br>
```

## 11. Header `include/quadriga_gbsm.hpp:183–236` — 14 + 4 + 2 + 6 comment lines → 3 + 1 + 1 + 3

```
    // ---- Small-scale fading ----
    // Cluster level of TR 38.901 clause 7.5 steps 5 to 7; N = NumClusters - 1 NLOS clusters. The column layout of
    // the random values is documented in src/qd_gbsm_phases.cpp (ssf_draw)

    // Draw the random phases of one small-scale fading generator set (seed derived from the large-scale seed)
    void ssf_init_phases(...);

    // Spatially correlated normalized random values of one small-scale fading generator set at a set of links
    void ssf_draw(...);

    // Cluster delays, powers and angles of a set of links, TR 38.901 clause 7.5 steps 5 to 7
    // - Methods "38.901", "38.901_MF" (clause 7.6.5.1), "quadriga"
    // - Slots: 0 = LOS path, 1 = ground reflection if GR_enabled, then the N NLOS clusters
    void ssf_clusters(...);
```

The per-argument trailing comments stay as they are (they are the header's argument list), except
`rnd_freq` "...; required by 38.901, 38.901_MF" (drop "if N > 0") and `lsp_anchor` likewise.

## 12. File comment `src/qd_gbsm_phases.cpp:34–66` — 33 → 27 lines

Keep the SSF generator index table (it documents `qd_gbsm_gen_ssf` for the developer and is the only place of the
mirror table besides the user doc). Keep the seed paragraph here as the single home of the SplitMix64 rule (it leaves
the MD block), written out:

```
// Seeds of the SSF sets: seed_set = splitmix64( splitmix64(seed) ^ tag ) >> 1 with the seed of the LSF set of the
// same site and scenario, tag = 1 for the shared set and 2 + k for the set of frequency k, and splitmix64(x):
// z = x + 0x9E3779B97F4A7C15; z = (z ^ (z >> 30))·0xBF58476D1CE4E5B9; z = (z ^ (z >> 27))·0x94D049BB133111EB;
// return z ^ (z >> 31) (uint64, wrapping). A set never repeats the LSF phases of the same seed (no shared stream)
// and the sets are independent streams; seed -1 passes -1 on (system random device).
```

Delete the last paragraph ("The machinery in the anonymous namespace takes everything as explicit arguments ...") or
merge it into the existing LSF machinery comment above the helpers: it describes `qd_gbsm_freq` / `qd_gbsm_eval`,
not the SSF index.
