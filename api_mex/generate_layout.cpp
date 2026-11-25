// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_mex_layout.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# GENERATE_LAYOUT
Network layout from a preset: BS sites and a drop of UTs

- Creates a standard evaluation layout with the values of its tables: the hexagonal and indoor layouts of 3GPP TR
  38.901, the test environments of ITU-R M.2412 and the road layouts of 3GPP TR 37.885
- The result has the sites `BS1`, `BS2`, ... with one Tx node per sector and the UTs `UT1`, `UT2`, ... as point drops;
  set the frequencies and the antennas next ([[set_antenna]]), until then the nodes use the generic antenna
- The other inputs change single values of the preset: an entry of a tuple that is NaN or omitted keeps the value of the
  preset; an entry or option that the preset does not use raises an error

## Usage:
```
layout = quadriga_lib.generate_layout( type, n_ut, spacing, grid, area, bs, ut, indoor, options, ...
    per_cell, seed );
```

## Inputs:
- **`type`** — Preset name, not case sensitive:
  | `type` | Layout |
  | ------ | ------ |
  | `38.901_UMi`, `38.901_UMa`, `38.901_RMa`, `38.901_SMa` | Hexagonal, 19 sites of 3 sectors, ISD 200, 500, 1732, 1299 m, BS height 10, 25, 35, 35 m |
  | `38.901_InH_Open`, `38.901_InH_Mixed` | Hall of 120 × 50 m, 6 × 2 BSs at 20 m spacing, BS height 3 m |
  | `38.901_InF_SL`, `38.901_InF_DH` | Hall of 120 × 60 × 10 m, 6 × 3 BSs at 20 m spacing, BS height 1.5 m (SL), 8 m (DH) |
  | `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_HH` | Hall of 300 × 150 × 10 m, 6 × 3 BSs at 50 m spacing, BS height 1.5 m (DL), 8 m (SH, HH) |
  | `M.2412_InH`, `M.2412_DenseUrban`, `M.2412_Rural`, `M.2412_UrbanMacro` | Test environments of Report ITU-R M.2412-0 (see Notes) |
  | `37.885_Urban`, `37.885_Highway` | Roads and vehicles of TR 37.885, no BSs (see Notes) |
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs; default: 0
- **`spacing`** — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used), the spacing of the indoor
  grid or of the urban road grid; default: `[]`
- **`grid`** — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site; `n_y` not
  used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with the boresights 30° (indoor 0°) + k ·
  360° / `n_sector`; default: `[]`
- **`area`** — `[length, width, height]` of the hall or road area along x, y and z in [m], > 0; indoor and road presets,
  the height InF only, the width not for the highway; the BS grid must fit into it; default: `[]`
- **`bs`** — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt (positive down) and
  an offset added to the boresights; default: `[]`
- **`ut`** — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg], [deg], [deg]; a
  value replaces the rule of the preset for all UTs (building UTs get the floor from a given height; the speed of the
  lanes also sets the gaps); min distance (not for roads) and speed >= 0; default: `[]`
- **`indoor`** — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1] (presets with
  O2I states); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa, Dense Urban); floor height in [m], > 0 (UMi, UMa,
  SMa, Dense Urban); default: `[]`
- **`options`** — Struct with one value per field: `p_residential` (SMa), the share of residential buildings in [0, 1];
  `config` (M.2412), `service` (`M.2412_UrbanMacro`), `h_micro` (`M.2412_DenseUrban` config 3) in [m]; `lane_width` in
  [m], `n_lane` (1 to 100), `p_truck` in [0, 1] (TR 37.885); other fields raise an error; default: `[]`
- **`per_cell`** — `true`: `n_ut` UTs in each cell; `false`: `n_ut` UTs uniform over all cells; no effect for the TR
  37.885 presets; default: `true`
- **`seed`** — Seed >= 0; -1 = system random device; default: -1

## Outputs:
- **`layout`** — Layout struct, fields in [[layout_csv_read]]

## Example:
```
layout = quadriga_lib.generate_layout( '38.901_UMa', 10, [], [], [], [], [NaN NaN NaN 0 90], [], [], true, 42 );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2] ) );
```

## Notes:
- **Hexagonal presets** (2 rings, 3 sectors, wrap-around): site (q, r) with `max(abs(q), abs(r), abs(q + r)) <= n` for n
  rings is at `x = sqrt(3) · q · d / 2`, `y = (q / 2 + r) · d` with the ISD d, ordered by ring and then by azimuth in
  (-30°, 330°]. The wrap vectors `a1 = (sqrt(3) · n · d / 2, (3 · n / 2 + 1) · d)` and a2, a1 turned by 60°, make the
  cluster tile the plane
- **`38.901_SMa`**: mechanical tilt 5° at ISD 1299 m and 2° at 1732 m (other ISDs: that of the nearer one); option
  `p_residential` (0.9)
- **Indoor presets**: centred at the origin, one sector, no wrap-around; the BSs form a lattice centred in the hall,
  ordered by y and then by x. The InF presets set the scenario parameters `DS_VS` (V/S of the hall) and `absTOA_Lmax`
  (its largest dimension) on the Tx rows, the TR 38.901 InH presets `absTOA_Lmax` (the largest of length, width and the
  office height 3 m). `38.901_InF_HH` has the layout of SH, since the TR has no calibration layout for it
- **Scenario**: the Tx rows of the TR 38.901 presets get the scenario family (the preset name without `38.901_`); the
  other presets set no scenario
- **ITU-R M.2412 presets** (Report M.2412-0, section 8.3 and Table 5): geometry and drop only, mechanical tilt 0; option
  `config` 1, 2, 3 for the configurations A, B, C (default 1); the minimum distance and O2I fields of the TR 38.901
  model that the Report maps the environment to. `d2d_in` is the TR 38.901 minimum of two uniforms (a library choice;
  the Report's model A at or below 6 GHz uses one uniform). The building-loss shares of Table 5 (20 % high, 80 % low
  loss) are set with `o2i_low = 0.8` of [[init_link_states]], not by the preset
  - `M.2412_InH`: as `38.901_InH_Open` with UT height 1.5 m, without scenario and scenario parameters
  - `M.2412_DenseUrban`: as `38.901_UMa` with ISD 200 m; 80 % in buildings with floors, 20 % in cars at 30 km/h. Config
    3 adds 3 micro sites per macro TRxP, uniform in its cell, one TRxP each, `h_micro` high (option, 10 m), at least 35
    m from the macro sites and 20 m from each other; they form no cells (`cell` = 0): a macro TRxP cell receives `n_ut`
    UTs for itself and for each of its micro TRxPs, at least 10 m from the micro sites, over the whole macro layer
    (Table 5 b: within a cluster)
  - `M.2412_Rural`: as `38.901_RMa` with UT height 1.5 m; configs 1 and 2: 50 % in buildings, 50 % in cars at 120 km/h;
    config 3: ISD 6000 m, 40 % in buildings, 40 % outdoors, 20 % in cars at 30 km/h. With a given indoor ratio, config 3
    keeps one third of the other UTs in cars
  - `M.2412_UrbanMacro`: BS height 25 m, UT height 1.5 m; option `service` 1 (mMTC, default): ISD 500 m (config 1) or
    1732 m (config 2), 80 % in buildings, 3 km/h; 2 (URLLC, configs 1 and 2): ISD 500 m, 20 % in buildings, outdoors 30
    km/h
  - The M.2412 layouts carry no scenario, so a new drop with [[drop_terminals]] reproduces the distances of the preset
    only with `ut` entry 1 (e.g. 35 m) or with the scenario `UMa` on the macro rows and `UMi` on the micro rows
- **TR 37.885 presets** (Table A-1, clause 6.1.2, option A): roads and vehicles, no BSs; `n_ut` must be 0, the vehicles
  follow from the gaps
  - `37.885_Urban`: road grid of `spacing` 433 × 250 m in an `area` of 1299 × 750 m, centred, 2 lanes per direction of
    3.5 m, sidewalks 3 m, 60 km/h
  - `37.885_Highway`: `area` length 2000 m along x, 3 lanes per direction of 4 m, 140 km/h, wrap-around along the road
  - Lanes on the right side of their direction; options `lane_width`, `n_lane` (per direction) and `p_truck` (0);
    passenger vehicles 5 m long with the antenna at 1.6 m, trucks 13 m and 3 m. Along a lane, the gap from the front of
    a vehicle to the rear of the next one is max(2 m, Exp(mean = speed · 2 s)); vehicles head along their lane
- **UT drop**: point drops, `n_ut` per cell or in total, uniform over the cells (the sector part of a site hexagon, the
  hexagon without sectors, the lattice cell around an indoor BS):
  - Minimum 2D distance to the nearest image of the nearest site: UMi 10 m, UMa 35 m (indoor UTs: to d2D-out, TR 36.873
    Table 6-1), RMa and SMa 35 m, InH 0, InF 1 m. A rejected UT is drawn again with its O2I state, so the d2D-out rule
    lowers the building share slightly below its nominal value; a building UT of UMi and UMa needs d2D >= d2D-in also at
    `ut` min distance 0
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings `3 · (n_fl - 1) + 1.5` m with `n_fl ~ U(1, N_fl)`,
    `N_fl ~ U(4, 8)`; SMa residential buildings 1 or 2 floors, commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in
    azimuth, elevation 0. Calibration orientations by `ut`: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C
    tilt 45°, config D tilt 0° and heading 0°
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in
    buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially
    correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or
    10 m (RMa, SMa residential)
- **Reproducibility**: the layout is a function of the seed and does not depend on the thread count; seed -1 draws a
  seed, which is stored in `layout.seed`. `center_freq` is empty

## See also:
- [[drop_terminals]] (the same drop for any set of Tx nodes)
- [[generate_track]] (tracks for the UTs)
- [[set_antenna]] (antennas of the nodes)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const std::string type = qd_mex_get_string(prhs[0], "", "type");
    const arma::uword n_ut = (nrhs < 2) ? 0 : qd_mex_get_scalar<arma::uword>(prhs[1], "n_ut", 0);
    const arma::vec spacing = qd_mex_layout_vec(nrhs, prhs, 2, "spacing"), grid = qd_mex_layout_vec(nrhs, prhs, 3, "grid");
    const arma::vec area = qd_mex_layout_vec(nrhs, prhs, 4, "area"), bs = qd_mex_layout_vec(nrhs, prhs, 5, "bs");
    const arma::vec ut = qd_mex_layout_vec(nrhs, prhs, 6, "ut"), indoor = qd_mex_layout_vec(nrhs, prhs, 7, "indoor");
    const auto options = (nrhs < 9) ? std::unordered_map<std::string, std::vector<double>>() : qd_mex_layout_options(prhs[8], "options");
    const bool per_cell = (nrhs < 10) ? true : qd_mex_get_scalar<bool>(prhs[9], "per_cell", true);
    const arma::sword seed = (nrhs < 11) ? -1 : (arma::sword)qd_mex_get_seed(prhs[10], "seed");

    quadriga_lib::layout layout;
    CALL_QD(layout = quadriga_lib::generate_layout(type, n_ut, spacing, grid, area, bs, ut, indoor, options, per_cell, seed));

    plhs[0] = qd_mex_layout2struct(layout);
}
