// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# generate_layout
Network layout from a preset: BS sites and a drop of UTs

- Presets (not case sensitive) of 3GPP TR 38.901, ITU-R M.2412 and 3GPP TR 37.885 with the values of their tables. BS sites are named `BS1`, `BS2`, ..., with one node per sector; the Tx rows of the TR 38.901 presets get the scenario family (the preset name without `38.901_`), the others no scenario. UTs are named `UT1`, `UT2`, ...
- Hexagonal presets, 2 rings (19 sites), 3 sectors with boresights 30°, 150° and 270°, wrap vectors set:
  - `38.901_UMi`: ISD 200 m, hBS 10 m; `38.901_UMa`: ISD 500 m, hBS 25 m; `38.901_RMa`: ISD 1732 m, hBS 35 m
  - `38.901_SMa`: ISD 1299 m, hBS 35 m, mechanical tilt 5° at ISD 1299 m and 2° at 1732 m (other ISDs: that of the nearer one); option `p_residential` (0.9)
  - Sites ordered by ring and then by azimuth in (−30°, 330°]; the wrap vectors make the cluster tile the plane
- Indoor presets, centred at the origin, one sector, no wrap; a lattice of BSs centred in the hall, ordered by y and then by x:
  - `38.901_InH_Open`, `38.901_InH_Mixed`: 120 × 50 m, 6 × 2 BSs at 20 m, hBS 3 m
  - `38.901_InF_SL`: 120 × 60 × 10 m, 6 × 3 BSs at 20 m, hBS 1.5 m; `38.901_InF_DL`: 300 × 150 × 10 m, 6 × 3 BSs at 50 m, hBS 1.5 m; `38.901_InF_SH`: as DL with hBS 8 m; `38.901_InF_DH`: as SL with hBS 8 m; `38.901_InF_HH`: as SH (the TR has no calibration layout for it)
  - The InF presets set `scenpar_override` `DS_VS` (V/S of the hall) and `absTOA_Lmax` (its largest dimension) on the Tx rows, the TR 38.901 InH presets `absTOA_Lmax` (the largest of length, width and the office height 3 m)
- ITU-R M.2412 presets (Report M.2412-0, section 8.3 and Table 5), geometry and drop only; option `config` 1, 2, 3 for configurations A, B, C (default 1); mechanical tilt 0; minimum distance and O2I fields of the TR 38.901 model that the Report maps the environment to. `d2d_in` is the TR 38.901 minimum of two uniforms (library choice; the Report's model A at or below 6 GHz uses one uniform). The building-loss shares of Table 5 (20 % high, 80 % low loss) are set with `o2i_low = 0.8` of [[init_link_states]], not by the preset:
  - `M.2412_InH`: as `38.901_InH_Open` with hUT 1.5 m, without scenario and `scenpar_override`
  - `M.2412_DenseUrban`: as `38.901_UMa` with ISD 200 m; 80 % in buildings with floors, 20 % in cars at 30 km/h. Config 3 adds 3 micro sites per macro TRxP, uniform in its cell, one TRxP each, `h_micro` high (option, 10 m), at least 35 m from the macro sites and 20 m from each other; they form no cells: a macro TRxP cell receives `n_ut` UTs for itself and for each of its micro TRxPs, at least 10 m from the micro sites, over the whole macro layer (Table 5 b: within a cluster)
  - `M.2412_Rural`: as `38.901_RMa`, hUT 1.5 m; configs 1 and 2: 50 % in buildings, 50 % in cars at 120 km/h; config 3: ISD 6000 m, 40 % in buildings, 40 % outdoors, 20 % in cars at 30 km/h
  - `M.2412_UrbanMacro`: hBS 25 m, hUT 1.5 m; option `service` 1 (mMTC, default): ISD 500 m (config 1) or 1732 m (config 2), 80 % in buildings, 3 km/h; 2 (URLLC, configs 1 and 2): ISD 500 m, 20 % in buildings, outdoors 30 km/h
- TR 37.885 presets (Table A-1, clause 6.1.2, option A), roads and vehicles, no BSs; `n_ut` must be 0, the vehicles follow from the gaps:
  - `37.885_Urban`: road grid of `spacing` 433 × 250 m in an `area` of 1299 × 750 m, centred, 2 lanes per direction of 3.5 m, sidewalks 3 m, 60 km/h
  - `37.885_Highway`: `area` length 2000 m along x, 3 lanes per direction of 4 m, 140 km/h, `wrap` along the road
  - Lanes on the right side of their direction; options `lane_width`, `n_lane` (per direction) and `p_truck` (0); passenger vehicles 5 m long with the antenna at 1.6 m, trucks 13 m and 3 m; along a lane, the gap from the front of a vehicle to the rear of the next one is max(2 m, Exp(mean = speed · 2 s)); vehicles head along their lane
- UT drop, as point drops: `n_ut` per cell or in total, uniform over the cells (the sector part of a site hexagon, the hexagon without sectors, the lattice cell around an indoor BS):
  - Minimum 2D distance to the nearest image of the nearest site: UMi 10 m, UMa 35 m (indoor UTs: to d2D-out, TR 36.873 Table 6-1), RMa and SMa 35 m, InH 0, InF 1 m; a rejected UT is redrawn with its O2I state, so the d2D-out rule lowers the building share slightly below its nominal value
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings `3 (n_fl − 1) + 1.5` m with `n_fl ~ U(1, N_fl)`, `N_fl ~ U(4, 8)`; SMa residential buildings 1 or 2 floors, commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in azimuth, elevation 0
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or 10 m (RMa, SMa residential)
- The layout is a function of the seed, independent of the thread count, with the random streams of [[drop_terminals]]; seed -1 draws a seed, which is stored in `layout["seed"]`. Antennas are not set (the channel functions use the generic antenna until [[set_antenna]] sets one) and `center_freq` is empty

## Usage:
```
layout = quadriga_lib.layout.generate_layout( type, n_ut, *, spacing, grid, area, bs, ut, indoor, options, \
    per_cell, seed )
```

## Inputs:
- **`type`** — Preset name; str
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs; int; default: 0
- **`spacing`** — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used), the spacing of the indoor grid or of the urban road grid; default: `None`
- **`grid`** — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site; `n_y` not used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with boresights 30° (indoor 0°) + k · 360° / `n_sector`; default: `None`
- **`area`** — `[length, width, height]` of the hall or road area along x, y and z in [m], > 0; indoor and road presets, the height InF only, the width not for the highway; the BS grid must fit into it; default: `None`
- **`bs`** — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt (positive down) and an offset added to the boresights; default: `None`
- **`ut`** — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg], [deg], [deg]; a value replaces the rule of the preset for all UTs (building UTs get the floor from a given height; the speed of the lanes also sets the gaps); min distance (not for roads) and speed >= 0 (with the d2D-out rule, a building UT of UMi and UMa needs d2D >= d2D-in also at min distance 0). Calibration orientations: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C tilt 45°, config D tilt 0° and heading 0°; default: `None`
- **`indoor`** — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1] (presets with O2I states); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa, Dense Urban); floor height in [m], > 0 (UMi, UMa, SMa, Dense Urban). With a given ratio, `M.2412_Rural` config 3 keeps one third of the other UTs in cars; default: `None`
- **`options`** — Dict with one value per key: `p_residential` (SMa), the share of residential buildings in [0, 1]; `config` (M.2412), `service` (`M.2412_UrbanMacro`), `h_micro` (`M.2412_DenseUrban` config 3) in [m]; `lane_width` in [m], `n_lane` (1 to 100), `p_truck` in [0, 1] (TR 37.885); other keys raise an error; default: `None`
- **`per_cell`** — `True`: `n_ut` UTs in each cell; `False`: `n_ut` UTs uniform over all cells; no effect for the TR 37.885 presets; default: `True`
- **`seed`** — Seed >= 0; -1 = system random device; default: -1

An entry of a tuple that is NaN or omitted takes the value of the preset; an entry or option that the preset does not use raises an error.

## Outputs:
- **`layout`** — Layout dict, keys in [[layout.csv_read]]

## Example:
```
import quadriga_lib as qd
layout = qd.layout.generate_layout( "38.901_UMa", 10, ut=[float("nan")] * 3 + [0, 90], seed=42 )
layout = qd.layout.set_antenna( layout, "3GPP", {"panel": [4, 4, 1, 2]} )
```

## See also:
- [[drop_terminals]] (the same drop for any Tx side)
- [[generate_track]] (tracks for the UTs)
- [[set_antenna]] (antennas of the nodes)
MD!*/
py::dict generate_layout(const std::string &type,
                         arma::uword n_ut,
                         py::handle spacing,
                         py::handle grid,
                         py::handle area,
                         py::handle bs,
                         py::handle ut,
                         py::handle indoor,
                         py::handle options,
                         py::handle per_cell,
                         long long seed)
{
    const auto spacing_a = qd_python_numpy2arma_Col<double>(spacing, true);
    const auto grid_a = qd_python_numpy2arma_Col<double>(grid, true);
    const auto area_a = qd_python_numpy2arma_Col<double>(area, true);
    const auto bs_a = qd_python_numpy2arma_Col<double>(bs, true);
    const auto ut_a = qd_python_numpy2arma_Col<double>(ut, true);
    const auto indoor_a = qd_python_numpy2arma_Col<double>(indoor, true);
    const auto options_m = qd_pl_options(options, "options");
    const bool per_cell_b = qd_python_flag(per_cell, true, "per_cell");

    const auto lay = quadriga_lib::generate_layout(type, n_ut, spacing_a, grid_a, area_a, bs_a, ut_a, indoor_a, options_m,
                                                   per_cell_b, (arma::sword)seed);
    return qd_python_layout2dict(lay);
}

/*!MD
# drop_terminals
Drop UTs into the cells of the Tx side of a layout

- Cells: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the wrap images when `layout["wrap"]` is set. A site with several sectors is split into wedges by the nearest sector heading (±60° around each boresight for three sectors at equal spacing)
- Drop area: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the images 0, ±a1, ±a2 and ±(a1 − a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- The drop of [[generate_layout]] applies in the cells of each site, with the rules of the scenario family in its `scenario` (minimum distance, heights and floors, speeds, O2I states). Sites without a family (no scenario, a conf name): height 1.5 m, no minimum distance, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- Every site is a cell, also the micro sites of `M.2412_DenseUrban` config 3 (whose own drop counts them with their macro TRxP). A site closer than twice the minimum distance to another one can leave a cell too little room, e.g. micro sites with the scenario `UMa` or `UMi` set; such a cell raises an error
- The Rx side is replaced by the UTs `UT1`, `UT2`, ... (ordered by cell with `per_cell`) with the generic antenna; `pairing` and `links` are emptied. The Tx side, the antenna lists and `layout["seed"]` stay. A failed call leaves the layout unchanged
- Random streams: the positions use the key `drop`, the heading and direction of a UT `ut|<name>`, its floor `floor|<family>|<name>`, the O2I fields `<purpose>|<family>`; the base of all is the seed of the call
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

## Usage:
```
layout_out = quadriga_lib.layout.drop_terminals( layout_in, n_ut, *, bounds, ut, indoor, options, per_cell, \
    seed )
```

## Inputs:
- **`layout_in`** — Valid layout dict, keys in [[layout.csv_read]]; the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs; int
- **`bounds`** — `(x_min, x_max, y_min, y_max)` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless `layout_in["wrap"]` has two vectors; default: `None`
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** — As in [[generate_layout]]; an entry or option that no family of the sites uses raises an error; defaults: None, None, None, True
- **`seed`** — Seed >= 0; -1 = `layout_in["seed"]` if it is set, else the system random device; default: -1

## Outputs:
- **`layout_out`** — The layout with the Rx side replaced by the dropped UTs

## See also:
- [[generate_layout]] (layouts from presets, with this drop)
- [[generate_track]] (tracks for the UTs)
MD!*/
py::dict drop_terminals(const py::dict &layout_in,
                        arma::uword n_ut,
                        py::handle bounds,
                        py::handle ut,
                        py::handle indoor,
                        py::handle options,
                        py::handle per_cell,
                        long long seed)
{
    auto lay = qd_python_dict2layout(layout_in, false, "layout_in");
    const auto bounds_a = qd_python_numpy2arma_Col<double>(bounds, true);
    const auto ut_a = qd_python_numpy2arma_Col<double>(ut, true);
    const auto indoor_a = qd_python_numpy2arma_Col<double>(indoor, true);
    const auto options_m = qd_pl_options(options, "options");
    const bool per_cell_b = qd_python_flag(per_cell, true, "per_cell");

    quadriga_lib::drop_terminals(lay, n_ut, bounds_a, ut_a, indoor_a, options_m, per_cell_b, (arma::sword)seed);
    return qd_python_layout2dict(lay);
}
