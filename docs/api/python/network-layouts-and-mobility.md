# Network layouts and mobility

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-network-layouts-and-mobility

| Function | Summary |
| --- | --- |
| [attach_terminals](#attach_terminals) | Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP |
| [drop_terminals](#drop_terminals) | Drop UTs into the cells of the Tx side of a layout |
| [eval_antennas](#eval_antennas) | Generate the antenna arrays of a layout from the antenna lines |
| [export_rays_qrt](#export_rays_qrt) | Writes the rays of a layout with their link gain to a QRT file |
| [generate_layout](#generate_layout) | Network layout from a preset: BS sites and a drop of UTs |
| [generate_rays](#generate_rays) | Rays of propagation links of a layout (stage 2 of the channel pipeline) |
| [generate_track](#generate_track) | Replace the rows of nodes by a generated track or a point drop |
| [get_channels_layout](#get_channels_layout) | Channel coefficients of a layout in one call |
| [get_gbsm_channels](#get_gbsm_channels) | Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline) |
| [init_link_states](#init_link_states) | Draws the discrete state of the propagation links of a layout (stage 1 of the channel pipeline) |
| [layout.csv_read](#layout.csv_read) | Read the Tx and Rx CSV files of a layout |
| [layout.csv_write](#layout.csv_write) | Write a layout to the Tx and Rx CSV files |
| [sample_track](#sample_track) | Positions and orientations of a node at snapshot times |
| [set_antenna](#set_antenna) | Set the antenna of layout nodes from the arguments of generate_antenna |
| [set_antenna_array](#set_antenna_array) | Set the antenna of layout nodes from an evaluated array |
| [set_antenna_qdant](#set_antenna_qdant) | Set the antenna of layout nodes to an entry of a QDANT file |
| [set_orientation](#set_orientation) | Set the orientation of nodes from their motion, or to a given value |

All functions of this section are in the submodule `quadriga_lib.layout`: `generate_layout` is called as `quadriga_lib.layout.generate_layout`; `layout_csv_read` and `layout_csv_write` are `layout.csv_read` and `layout.csv_write`. `crossfade_segments` is in `quadriga_lib.channel`. A layout is a dict with the member names of the C++ class `layout`: `tx`, `rx` and `links` are nested dicts, `scenpar_override` is a dict with one array per scenario key, `antenna` is a list with one entry per antenna, each a list of arrayant dicts, one per band (the 4D multi-frequency dict of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) cannot hold per-band geometry), and `rays` are lists of ray dicts, see [ssf_rays](geometry-based-stochastic-model.md#ssf_rays).

A **layout** describes the nodes of a radio network and how they move: where each transmitter (Tx) and receiver (Rx) is, which scenario applies there and which antenna it has. It is stored in two flat tables, one per side, and in two CSV files with the same format.

**Terms.** A *node* is one (`name`, `sector`) pair on one side. It has one antenna and one *track*: its *rows*, each a position with orientation and further values that apply from that row on. A node with one row is a *point drop*; it stays at its position, or, with a speed, moves along its direction. A node with several rows is a track and moves along its positions. A *site* is all Tx nodes with one name; its sectors share position and scenario. A *terminal* is all Rx nodes with one name, e.g. the panels of one device. Angles are in degrees in the CSV files and in radians in the dicts. The orientation (bank, tilt, heading) is that of the antenna, a positive tilt points down; the optional direction (azimuth, elevation of travel) is that of the movement, and an unset direction follows the orientation.

**CSV files.** [layout.csv_read](#layout.csv_read) and [layout.csv_write](#layout.csv_write) read and write the files. The first line is the header; the columns are optional, in any order, except `name`, `x`, `y`, `z`. Cells follow the syntax of the other CSV files of the library: quoting with `"`, lines starting with `#` are comments, numbers are decimal (NaN and Inf in any case). Empty cells follow the class of the column:

| Class | Columns | An empty cell is |
| ----- | ------- | ---------------- |
| Identity | `name`, `sector` | `name` is required; an empty `sector` is 0 |
| Per row | `x`, `y`, `z`, `time` | an error; except for the one row of an extra sector of a static site, which takes the position of sector 0; `time` is given on all rows of a node or on none |
| Held | `bank`, `tilt`, `heading`, `direction_az`, `direction_el`, `speed`, `scenario`, `indoor`, `floor`, `d2d_in` | the value of the previous row of the node; on the first row the value of sector 0 for an extra sector, else the default; NaN is a value |
| Fixed | all others | the value of the first row, for an extra sector that of sector 0; later rows are empty or repeat it, and a different value is an error |

| Column | Unit | Meaning |
| ------ | ---- | ------- |
| `bank`, `tilt`, `heading` | degree | Antenna orientation, default 0 |
| `direction_az`, `direction_el` | degree | Direction of travel; empty or NaN azimuth follows the orientation; only for point drops |
| `speed` | m/s | Speed from this row on, default 0 |
| `time` | s | Time of the row, strictly increasing; a node has `time` or a speed |
| `scenario` | | Scenario family or conf file name, never a path |
| `indoor`, `floor`, `d2d_in` | | Indoor state (-1 or empty: draw, 0 outdoor, 1 building, 2 car, 3 residential building), floor from 1 (-1: from z), indoor 2D distance in m (NaN: draw); they change only where `scenario` or `indoor` changes |
| `d_clutter`, `r_clutter`, `h_clutter`, `scenpar.<key>` | | Tx side: clutter inputs and overrides of scenario parameters, equal for all sectors of a site |
| `seed`, `layout_seed`, `wrap` | | Tx side: seed of the site, seed of the layout, lattice vectors in m (`"ax,ay;bx,by"`) |
| `cell` | | Tx side: 0 = the site has no drop cells of its own (see [drop_terminals](#drop_terminals)), default 1; equal for all sectors of a site |
| `pairing` | | Rx side: names of the Tx sites linked to the terminal, empty: all |
| `relative` | | 1: the orientation is relative to the direction of motion |
| `center_freq`, `antenna`, `qdant_file`, `qdant_id` and the inputs of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) | | Antenna columns, see below |

**Antennas.** The antenna columns of a node (type `antenna` as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), or `qdant` with the file `qdant_file` relative to the CSV file and its 1-based entry `qdant_id`) are kept as one text line per antenna in `antenna_args`, with the names of the columns in `antenna_cols`. Nodes with equal lines share one entry of the lists, Tx nodes first, and `i_antenna` is the 1-based index of it, 0 for the generic antenna with two co-located isotropic elements of vertical and horizontal polarization. A column that the type does not use is an error. An extra sector without antenna cells has the antenna of sector 0; the type `none` (no other antenna cells) gives a node no antenna, e.g. an extra sector next to a sector 0 with one. [eval_antennas](#eval_antennas) generates the arrays `antenna` from the lines: a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored. [set_antenna](#set_antenna) sets the antenna of nodes from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), [set_antenna_array](#set_antenna_array) from an evaluated array, which [layout.csv_write](#layout.csv_write) writes to a QDANT file next to the CSV file; [set_antenna_qdant](#set_antenna_qdant) sets an entry of a QDANT file, which the writer refers to.

**Bands.** A row with `center_freq` and no position adds a frequency, and optionally an antenna for it, to its node. The frequencies of the Tx side (else of the Rx side) are the simulation frequencies `center_freq` of the layout. A node without bands has all of them with one antenna description; a node with bands defines all of them. An empty antenna in a band repeats the antenna of the first band.

**Unknown columns** are kept as text in `extra_names` and `extra` and written back unchanged.

**Channels from a layout.** The channel coefficients are computed in three stages. [init_link_states](#init_link_states) draws the discrete state of each *propagation link*, one site and one terminal: its wrap image, scenario and conf file, LOS and indoor state and seeds, stored in `layout["links"]`. [generate_rays](#generate_rays) runs the chain of the section "Geometry-based stochastic model" for chosen links and returns their rays. [get_gbsm_channels](#get_gbsm_channels) applies the antennas, the motion along the tracks and the link gain, for each *channel link*, one Tx node and one Rx node; the sectors of a site share the rays of their propagation link. [get_channels_layout](#get_channels_layout) runs all three stages. [attach_terminals](#attach_terminals) chooses the serving Tx node and site of each terminal by the large-scale gain or by RSRP, for `layout["pairing"]`. [export_rays_qrt](#export_rays_qrt) writes the rays with their link gain to a QRT file, which gives channels for other antennas and positions without the GBSM chain. All random values are addressed: they follow from the seeds, a stream key and the position, so that the evaluation order, the batches and the thread count do not change them, and only the discrete states are stored. `layout["links"]` holds a hash of the layout members that determine the links (positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter, scenario overrides, wrap vectors and whether a site moves); stages 2 and 3 throw when it no longer matches, and `layout["links"] = {}` discards the links. The links hold the master seed they were drawn with, so a copy of `layout["links"]` can be restored after another realisation. Editing `los` or `indoor` of the links between stage 1 and stage 2 forces a state; orientations, antennas and `pairing` may change without new links.

**Indices.** All indices are 0-based, as in C++: the members `offset`, `pairing` and, in `links`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model`, the index arguments of the functions and the functions that return indices. `i_antenna` is the index into the antenna lists with 0 = generic antenna, as in the other APIs; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values.

---
<a id="attach_terminals"></a>
## attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

Also in: [C++](../cpp/network-layouts-and-mobility.md#attach_terminals) · [MATLAB](../mex/network-layouts-and-mobility.md#attach_terminals)

- Evaluates all Tx nodes for each terminal at the first rows of both nodes and `layout["center_freq"][i_freq]`, with the link states of the first segment, the wrap image and the antennas as in [get_gbsm_channels](#get_gbsm_channels); `layout["pairing"]` is ignored
- `"large_scale"` (TR 38.901 Table 7.8-1): the link gain `−PL − O2I + 10 lg(SF)` plus the gain of one unit-gain LOS path from [channel.planar](channel-generation-functions.md#channel.planar) (port weights, element positions, orientations); no rays
- `"rsrp"` (TR 36.873 eq. (8.1-1)): per Rx port the sum of `|h|²` over the rays of [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) (`output = "rays"`, mobility `"phase"`), link gain included, transmit power 1
- Both rules take `|h|²` of Tx port `tx_port` and average it over the Rx ports, so a dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT antenna of the TR
- A terminal takes the highest value of its nodes; the serving node has the highest value, a tie the lowest index. A Tx node without a propagation link (co-located) has −Inf; a terminal without any link gets node 0
- `layout["pairing"] = np.vstack((site, np.arange(n_rx_terminal)))` links the sectors of the serving sites

### Usage:
```python
node, site, gain_dB = quadriga_lib.layout.attach_terminals( layout, *, rule, rays, i_freq, tx_port )
```

### Inputs:
- **`layout`** — Layout dict with links from [init_link_states](#init_link_states), Tx nodes required if there are terminals; keys in [layout.csv_read](#layout.csv_read)
- **`rule`** — `"large_scale"` or `"rsrp"`; default: `"large_scale"`
- **`rays`** — `"rsrp"`: rays of all links from [generate_rays](#generate_rays), possibly edited, required; list of `n_link` lists of `N` ray dicts (a ray dict alone for a link with one), `N = 1` or `n_freq`; default: None
- **`i_freq`** — Frequency index into `layout["center_freq"]`, 0-based; default: 0
- **`tx_port`** — Port of the Tx antennas, 0-based; default: 0

### Outputs:
- **`node`** — Serving Tx node per terminal, 0-based; int64; `(n_rx_terminal,)`
- **`site`** — Serving site per terminal, 0-based; int64; `(n_rx_terminal,)`
- **`gain_dB`** — Coupling gain or RSRP in [dB]; `(n_rx_terminal, n_tx_node)`

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays) (link states and rays)
- [get_gbsm_channels](#get_gbsm_channels) (channels of the linked pairs)

---
<a id="drop_terminals"></a>
## drop_terminals
Drop UTs into the cells of the Tx side of a layout

Also in: [C++](../cpp/network-layouts-and-mobility.md#drop_terminals) · [MATLAB](../mex/network-layouts-and-mobility.md#drop_terminals)

- Cells: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the wrap images when `layout["wrap"]` is set. A site with several sectors is split into wedges by the nearest sector heading (±60° around each boresight for three sectors at equal spacing)
- Drop area: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the images 0, ±a1, ±a2 and ±(a1 − a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- The drop of [generate_layout](#generate_layout) applies in the cells of each site, with the rules of the scenario family in its `scenario` (minimum distance, heights and floors, speeds, O2I states). Sites without a family (no scenario, a conf name): height 1.5 m, no minimum distance, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states. A site without a family has no minimum distance unless `ut` entry 1 gives one, and the distance kept from a site with `cell` = 0 follows that site's own family
- Sites with `cell` = 0, e.g. the micro sites of `M.2412_DenseUrban` config 3, form no cells: with `per_cell`, each of their Tx nodes adds `n_ut` UTs to the cell that holds its position, as the drop of [generate_layout](#generate_layout) does (there 4 `n_ut` per macro TRxP cell), and the UTs keep the minimum distance of the site's family (`ut` entry 1 if given) from them. A site closer than twice the minimum distance to another one can leave a cell too little room; such a cell raises an error
- The Rx side is replaced by the UTs `UT1`, `UT2`, ... (ordered by cell with `per_cell`) with the generic antenna; `pairing` and `links` are emptied. The Tx side, the antenna lists and `layout["seed"]` stay. A failed call leaves the layout unchanged
- Random streams: the positions use the key `drop`, the heading and direction of a UT `ut|<name>`, its floor `floor|<family>|<name>`, the O2I fields `<purpose>|<family>`; the base of all is the seed of the call
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### Usage:
```python
layout_out = quadriga_lib.layout.drop_terminals( layout_in, n_ut, *, bounds, ut, indoor, options, per_cell, \
    seed )
```

### Inputs:
- **`layout_in`** — Valid layout dict, keys in [layout.csv_read](#layout.csv_read); the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs; int
- **`bounds`** — `(x_min, x_max, y_min, y_max)` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless `layout_in["wrap"]` has two vectors; default: `None`
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** — As in [generate_layout](#generate_layout); an entry or option that no family of the sites uses raises an error; defaults: None, None, None, True
- **`seed`** — Seed >= 0; -1 = `layout_in["seed"]` if it is set, else the system random device; default: -1

### Outputs:
- **`layout_out`** — The layout with the Rx side replaced by the dropped UTs

### See also:
- [generate_layout](#generate_layout) (layouts from presets, with this drop)
- [generate_track](#generate_track) (tracks for the UTs)

---
<a id="eval_antennas"></a>
## eval_antennas
Generate the antenna arrays of a layout from the antenna lines

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout) · [MATLAB](../mex/network-layouts-and-mobility.md#eval_antennas)

- Generates the arrays of `antenna` that are not evaluated yet from `antenna_args`; a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored
- A frequency list `center_freq` replaces `layout_in["center_freq"]` and regenerates all lines; arrays without a line are kept and must fit the new list
- Raises an error for a band without description and for an array spaced in wavelengths without a frequency; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### Usage:
```python
layout_out = quadriga_lib.layout.eval_antennas( layout_in, *, center_freq )
```

### Inputs:
- **`layout_in`** — Layout dict, keys in [layout.csv_read](#layout.csv_read)
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = keep `layout_in["center_freq"]`; default: None; `(n_freq,)`

### Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

### See also:
- [layout.csv_read](#layout.csv_read) (reads a layout, optionally with evaluated antennas)
- [set_antenna](#set_antenna) (sets the antennas of nodes)

---
<a id="export_rays_qrt"></a>
## export_rays_qrt
Writes the rays of a layout with their link gain to a QRT file

Also in: [C++](../cpp/network-layouts-and-mobility.md#export_rays_qrt) · [MATLAB](../mex/network-layouts-and-mobility.md#export_rays_qrt)

- Separates the ray generation from the antennas: [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm.rays_from_paths](geometry-based-stochastic-model.md#gbsm.rays_from_paths) (with the method of the links and the `n_interact` of the file as `path_type`) and [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) (link gain 0, mobility `"drift"`) turn a CIR into channels for any arrays, orientations and positions
- One origin per site at its position, with the orientation of sector 0 (else of the first node of the site) and the site name; one destination per terminal with its name and one CIR per row (the nodes of a terminal share their positions; the CIR has the orientation of the first node)
- Per CIR: the rays of the link and of the segment that holds the row (scatterers added where missing), the link gain and the K-factor ratio of [get_gbsm_channels](#get_gbsm_channels) at the row, [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `"drift"`. A CIR read back gives the channel of [get_gbsm_channels](#get_gbsm_channels) (`output = "rays"`, `mobility = "drift"`) at the row within float32 accuracy
- A point drop with speed has one CIR; its motion is applied at the evaluation, at the link gain of the drop
- Only the pairs of `layout["pairing"]` (empty: all) are written; the other CIRs of an origin hold no paths
- Positions in the file are local: the returned shift, the centroid of the site and CIR positions in x and y, is subtracted (z unchanged); add it to the positions read back
- The QRT format stores no tap index and no ground permittivity: a CIR gives rays, not the taps of the TR, and moving a ground-reflection ray from a file raises an error
- The file is written to `fn + ".part"` and renamed at the end; a failed call leaves an existing file unchanged

### Usage:
```python
shift = quadriga_lib.layout.export_rays_qrt( fn, layout, *, rays )
```

### Inputs:
- **`fn`** — QRT file, str or path-like, replaced if it exists
- **`layout`** — Layout dict with links from [init_link_states](#init_link_states): one frequency for the TR methods (`"quadriga"`: 1 to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at a site position; keys in [layout.csv_read](#layout.csv_read)
- **`rays`** — Rays of all links from [generate_rays](#generate_rays), possibly edited, one object with all frequencies per link; `None` = generated for batches of sites with up to 256 links, which bounds the memory; list of `n_link` lists of one ray dict (or the ray dicts alone); default: None

### Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `(2,)`

### See also:
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the rays and the link gain)
- [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm.rays_from_paths](geometry-based-stochastic-model.md#gbsm.rays_from_paths), [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) (evaluation of the file)

---
<a id="generate_layout"></a>
## generate_layout
Network layout from a preset: BS sites and a drop of UTs

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_layout) · [MATLAB](../mex/network-layouts-and-mobility.md#generate_layout)

- Presets (not case sensitive) of 3GPP TR 38.901, ITU-R M.2412 and 3GPP TR 37.885 with the values of their tables. BS sites are named `BS1`, `BS2`, ..., with one node per sector; the Tx rows of the TR 38.901 presets get the scenario family (the preset name without `38.901_`), the others no scenario. UTs are named `UT1`, `UT2`, ...
- Hexagonal presets, 2 rings (19 sites), 3 sectors with boresights 30°, 150° and 270°, wrap vectors set:
  - `38.901_UMi`: ISD 200 m, hBS 10 m; `38.901_UMa`: ISD 500 m, hBS 25 m; `38.901_RMa`: ISD 1732 m, hBS 35 m
  - `38.901_SMa`: ISD 1299 m, hBS 35 m, mechanical tilt 5° at ISD 1299 m and 2° at 1732 m (other ISDs: that of the nearer one); option `p_residential` (0.9)
  - Sites ordered by ring and then by azimuth in (−30°, 330°]; the wrap vectors make the cluster tile the plane
- Indoor presets, centred at the origin, one sector, no wrap; a lattice of BSs centred in the hall, ordered by y and then by x:
  - `38.901_InH_Open`, `38.901_InH_Mixed`: 120 × 50 m, 6 × 2 BSs at 20 m, hBS 3 m
  - `38.901_InF_SL`: 120 × 60 × 10 m, 6 × 3 BSs at 20 m, hBS 1.5 m; `38.901_InF_DL`: 300 × 150 × 10 m, 6 × 3 BSs at 50 m, hBS 1.5 m; `38.901_InF_SH`: as DL with hBS 8 m; `38.901_InF_DH`: as SL with hBS 8 m; `38.901_InF_HH`: as SH (the TR has no calibration layout for it)
  - The InF presets set `scenpar_override` `DS_VS` (V/S of the hall) and `absTOA_Lmax` (its largest dimension) on the Tx rows, the TR 38.901 InH presets `absTOA_Lmax` (the largest of length, width and the office height 3 m)
- ITU-R M.2412 presets (Report M.2412-0, section 8.3 and Table 5), geometry and drop only; option `config` 1, 2, 3 for configurations A, B, C (default 1); mechanical tilt 0; minimum distance and O2I fields of the TR 38.901 model that the Report maps the environment to. `d2d_in` is the TR 38.901 minimum of two uniforms (library choice; the Report's model A at or below 6 GHz uses one uniform). The building-loss shares of Table 5 (20 % high, 80 % low loss) are set with `o2i_low = 0.8` of [init_link_states](#init_link_states), not by the preset:
  - `M.2412_InH`: as `38.901_InH_Open` with hUT 1.5 m, without scenario and `scenpar_override`
  - `M.2412_DenseUrban`: as `38.901_UMa` with ISD 200 m; 80 % in buildings with floors, 20 % in cars at 30 km/h. Config 3 adds 3 micro sites per macro TRxP, uniform in its cell, one TRxP each, `h_micro` high (option, 10 m), at least 35 m from the macro sites and 20 m from each other; they form no cells (`cell` = 0): a macro TRxP cell receives `n_ut` UTs for itself and for each of its micro TRxPs, at least 10 m from the micro sites, over the whole macro layer (Table 5 b: within a cluster)
  - `M.2412_Rural`: as `38.901_RMa`, hUT 1.5 m; configs 1 and 2: 50 % in buildings, 50 % in cars at 120 km/h; config 3: ISD 6000 m, 40 % in buildings, 40 % outdoors, 20 % in cars at 30 km/h
  - `M.2412_UrbanMacro`: hBS 25 m, hUT 1.5 m; option `service` 1 (mMTC, default): ISD 500 m (config 1) or 1732 m (config 2), 80 % in buildings, 3 km/h; 2 (URLLC, configs 1 and 2): ISD 500 m, 20 % in buildings, outdoors 30 km/h
  - The M.2412 layouts carry no scenario, so a re-drop with [drop_terminals](#drop_terminals) reproduces the distances of the preset only with `ut` entry 1 (e.g. 35 m) or with the scenario `UMa` on the macro rows and `UMi` on the micro rows
- TR 37.885 presets (Table A-1, clause 6.1.2, option A), roads and vehicles, no BSs; `n_ut` must be 0, the vehicles follow from the gaps:
  - `37.885_Urban`: road grid of `spacing` 433 × 250 m in an `area` of 1299 × 750 m, centred, 2 lanes per direction of 3.5 m, sidewalks 3 m, 60 km/h
  - `37.885_Highway`: `area` length 2000 m along x, 3 lanes per direction of 4 m, 140 km/h, `wrap` along the road
  - Lanes on the right side of their direction; options `lane_width`, `n_lane` (per direction) and `p_truck` (0); passenger vehicles 5 m long with the antenna at 1.6 m, trucks 13 m and 3 m; along a lane, the gap from the front of a vehicle to the rear of the next one is max(2 m, Exp(mean = speed · 2 s)); vehicles head along their lane
- UT drop, as point drops: `n_ut` per cell or in total, uniform over the cells (the sector part of a site hexagon, the hexagon without sectors, the lattice cell around an indoor BS):
  - Minimum 2D distance to the nearest image of the nearest site: UMi 10 m, UMa 35 m (indoor UTs: to d2D-out, TR 36.873 Table 6-1), RMa and SMa 35 m, InH 0, InF 1 m; a rejected UT is redrawn with its O2I state, so the d2D-out rule lowers the building share slightly below its nominal value
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings `3 (n_fl − 1) + 1.5` m with `n_fl ~ U(1, N_fl)`, `N_fl ~ U(4, 8)`; SMa residential buildings 1 or 2 floors, commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in azimuth, elevation 0
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or 10 m (RMa, SMa residential)
- The layout is a function of the seed, independent of the thread count, with the random streams of [drop_terminals](#drop_terminals); seed -1 draws a seed, which is stored in `layout["seed"]`. Antennas are not set (the channel functions use the generic antenna until [set_antenna](#set_antenna) sets one) and `center_freq` is empty

### Usage:
```python
layout = quadriga_lib.layout.generate_layout( type, n_ut, *, spacing, grid, area, bs, ut, indoor, options, \
    per_cell, seed )
```

### Inputs:
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

### Outputs:
- **`layout`** — Layout dict, keys in [layout.csv_read](#layout.csv_read)

### Example:
```python
import quadriga_lib as qd
layout = qd.layout.generate_layout( "38.901_UMa", 10, ut=[float("nan")] * 3 + [0, 90], seed=42 )
layout = qd.layout.set_antenna( layout, "3GPP", {"panel": [4, 4, 1, 2]} )
```

### See also:
- [drop_terminals](#drop_terminals) (the same drop for any Tx side)
- [generate_track](#generate_track) (tracks for the UTs)
- [set_antenna](#set_antenna) (antennas of the nodes)

---
<a id="generate_rays"></a>
## generate_rays
Rays of propagation links of a layout (stage 2 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_rays) · [MATLAB](../mex/network-layouts-and-mobility.md#generate_rays)

- Runs the GBSM chain of the section "Geometry-based stochastic model" for the links with one site, image, conf and seed: [gbsm.init_phases](geometry-based-stochastic-model.md#gbsm.init_phases) and [gbsm.draw](geometry-based-stochastic-model.md#gbsm.draw) of the random sets, [path_loss](geometry-based-stochastic-model.md#path_loss) (for `rnd_he` and the SF standard deviation), [lsp_generate](geometry-based-stochastic-model.md#lsp_generate), [ssf_clusters](geometry-based-stochastic-model.md#ssf_clusters), [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) and, on request, [ssf_scatterers](geometry-based-stochastic-model.md#ssf_scatterers); at the first row of the segment of the terminal and the site (moved by the wrap image) at the same time (a moving site: sampled at the time of the row, without times on the Rx track the row of the same index)
- TR methods: preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance, `angle_model = "additive"`, `pol_model = "38.901"`, one set of clusters and rays per frequency (`"38.901_MF"`: clusters at the anchor frequency, rays per frequency). `"quadriga"`: bias removal over the positions of all links of the group, the legacy outdoor distance, `"rotation"`, `"quadriga"`, one ray set for all frequencies, always with scatterers
- A site with more than one row or a speed uses dual mobility; other sites are not affected
- The conf and seed of a link follow from its scenario and the edited `los` and `indoor` of `layout["links"]`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged. The floor and the indoor distance follow the indoor state: building UTs on the floor of the Rx side (else from z) at the distance of the stored O2I inputs, others on floor 0 at 0 m; the members `floor_id` and `d2d_in` of the links are those of the drawn state
- The values of a link do not depend on the other requested links

### Usage:
```python
rays = quadriga_lib.layout.generate_rays( layout, *, i_link, scatterers )
```

### Inputs:
- **`layout`** — Layout dict with links from [init_link_states](#init_link_states); keys in [layout.csv_read](#layout.csv_read); raises an error if a member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** — Propagation links, 0-based, without repetition; `None` = all (none for links without entries); default: None; `(n_out,)`
- **`scatterers`** — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`; default: `False`

### Outputs:
- **`rays`** — Rays per link and frequency, a list of `n_out` lists of `N` ray dicts with the keys of [ssf_rays](geometry-based-stochastic-model.md#ssf_rays), `N = n_freq` with one frequency each for the TR methods, `N = 1` with all frequencies for `"quadriga"`

### See also:
- [init_link_states](#init_link_states) (stage 1), [get_gbsm_channels](#get_gbsm_channels) (stage 3)
- [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) (the ray data)

---
<a id="generate_track"></a>
## generate_track
Replace the rows of nodes by a generated track or a point drop

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_track) · [MATLAB](../mex/network-layouts-and-mobility.md#generate_track)

- Each node keeps its first position; the call applies to all nodes with the name of a given node (the sectors of a site, the panels of a terminal), which get the same track. Tracks are horizontal at the height of the first row. The other members of the new rows (`scenario`, `indoor`, `floor`, `d2d_in`, extra columns) take the values of the first row of the node; `time` is cleared
- Types (not case sensitive):
  - `point`: the nodes become point drops with `speed` and an explicit horizontal `direction` (random if not given), the form of short-term mobility in 3GPP TR 38.901; the rows after the first are removed; `length` is not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction; two points, or `ceil(length / spacing) + 1` equidistant points
  - `circular`: a closed circle with the circumference `length` (default 20π m, radius 10 m) and 129 points, or `max(3, ceil(length / spacing)) + 1` points; the first and last point are equal. `direction` is the angle of the start point on the circle seen from its centre: a positive value runs counter-clockwise, a negative one clockwise (e.g. 0 starts in the east of the circle and runs north, -π/2 starts in the south and runs west); a random direction is drawn in [0, 2π), so random circles run counter-clockwise
  - `street`: the route of a car through a city grid as in `qd_track.generate` of the NG library. Straight streets of random length (normal distribution, drawn again below `street_length_min`) are separated by crossings; at a crossing the car turns with the probability `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of length 2 `curve_radius` cos(turn / 2); the direction stays within 105° of the start direction, so the car does not drive in circles. The route has the length `length` (default 1000 m) and is sampled with a point spacing of `spacing` (default 1 m); the chords between the points are slightly shorter than the route at the corners
- `linear`, `circular` and `street` give tracks of equal `speed` without `direction`, and their orientation follows the motion as [set_orientation](#set_orientation) with `force = False` gives it; nodes with `relative = 1` keep the orientation of their first row on all rows. `point` keeps the orientation of the node
- Reproducibility: a seed >= 0 gives the same track on all platforms; each name (all its nodes) has its own stream, keyed by `track|<name>`, so the track does not depend on the other nodes or the order in `i_node`. Seed -1 uses the system random device
- A failed call leaves `nodes` unchanged
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### Usage:
```python
nodes_out = quadriga_lib.layout.generate_track( nodes_in, i_node, type, *, length, speed, direction, spacing, \
    options, seed )
```

### Inputs:
- **`nodes_in`** — Nodes of one side of a layout, e.g. `layout["rx"]`; dict, keys in [layout.csv_read](#layout.csv_read)
- **`i_node`** — Node indices, 0-based, not empty; int or `(n,)`
- **`type`** — str: `point`, `linear`, `circular` or `street`
- **`length`** — Length in [m], finite and >= 0; 0 = default of the type; default: 0
- **`speed`** — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of the node if it is valid (> 0 for a track); without one, `point` gets 0 and the other types raise an error; default: NaN
- **`direction`** — Start direction (azimuth) in [rad], finite; float, one value for all nodes, or `(n,)` with one per entry of `i_node` (nodes of one name need equal values); `None` = uniform random in [0, 2π); default: `None`
- **`spacing`** — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`); default: NaN
- **`options`** — Dict of options of `street`, one value per key: `street_length_min` = 50, `street_length_mu` = 187, `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m], `turn_probability` = 0.5 (in [0, 1]); other keys, and these keys for the other types, raise an error; default: `None`
- **`seed`** — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 raises an error; default: -1

### Outputs:
- **`nodes_out`** — The nodes with the generated rows; dict

### Example:
```python
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "", "rx.csv" )
layout["rx"] = qd.layout.generate_track( layout["rx"], 0, "street", length=500, speed=10, \
    options={"turn_probability": 0.3}, seed=42 )
```

### See also:
- [sample_track](#sample_track) (snapshots of a track)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="get_channels_layout"></a>
## get_channels_layout
Channel coefficients of a layout in one call

Also in: [C++](../cpp/network-layouts-and-mobility.md#get_channels_layout) · [MATLAB](../mex/network-layouts-and-mobility.md#get_channels_layout)

- Runs [init_link_states](#init_link_states) (if `layout["links"]` was not filled), [generate_rays](#generate_rays) and [get_gbsm_channels](#get_gbsm_channels) in batches of at most 256 propagation links (the segments of a track in one batch), so that the rays of a batch bound the memory; the results are those of the manual path with the default settings and do not depend on the batches
- Without links (`layout["links"]["method"]` empty, e.g. after `layout["links"] = {}`), the link states are drawn for this call only and the layout is unchanged; with filled links, these are used, also when edited, and `seed` and `config_dirs` are not used. No linked pair gives an empty result
- The mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers
- The folder of the shipped conf files is added to `config_dirs` by the wrapper

### Usage:
```python
channels, index = quadriga_lib.layout.get_channels_layout( layout, *, times, output, method, mobility, overlap, \
    seed, config_dirs, stack )
```

### Inputs:
- **`layout`** — Layout dict with `center_freq`; keys in [layout.csv_read](#layout.csv_read); antennas as evaluated in `layout["antenna"]`, nodes without one use the generic antenna (see [get_gbsm_channels](#get_gbsm_channels))
- **`times`** — Snapshot times in [s]; `None` = one snapshot per row; default: None; `(n_snap,)`
- **`output`** — `"taps"` or `"rays"`; default: `"taps"`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; `""` = the method of filled links, else `"38.901"`; another method than that of filled links raises an error; default: `""`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"`; default: `"auto"`
- **`overlap`** — Overlap of consecutive segments, in [0, 1]; default: 0.5
- **`seed`** — Master seed for empty links, as in [init_link_states](#init_link_states); with filled links only -1; default: -1
- **`config_dirs`** — Folders of user conf files for empty links, searched before the shipped folder; str or list of str; default: None
- **`stack`** — Snapshot layout of the channels as in [hdf5_read_channel](channel-functions.md#hdf5_read_channel); default: `False`

### Outputs:
- **`channels`** — Channels, Tx node fastest, then Rx node, then frequency; list of channel dicts, keys in [hdf5_read_channel](channel-functions.md#hdf5_read_channel)
- **`index`** — Tx node, Rx node and frequency index of each channel, 0-based; int64; `(3, n_channel)`

### Example:
```python
import numpy as np, quadriga_lib as qd
layout = qd.layout.generate_layout( "38.901_UMa", 1 )
layout["center_freq"] = np.array([3.5e9])
channels, index = qd.layout.get_channels_layout( layout )
```

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the three stages)
- [generate_layout](#generate_layout) (layouts of the TR scenarios)

---
<a id="get_gbsm_channels"></a>
## get_gbsm_channels
Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#get_gbsm_channels) · [MATLAB](../mex/network-layouts-and-mobility.md#get_gbsm_channels)

- One channel per Tx node, Rx node and frequency whose propagation link is in `i_link` and whose pair is in `layout["pairing"]` (empty: all); order: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the sector where all sectors of the side are 0
- Snapshots of both ends from [sample_track](#sample_track) over `times`; a node with one snapshot is repeated to the count of the other; two tracks without `times` need equal row counts
- Link gain per snapshot (see the GBSM terms): [gbsm.draw](geometry-based-stochastic-model.md#gbsm.draw) of the `"lsf"` set at the snapshot, [path_loss](geometry-based-stochastic-model.md#path_loss) with the random environment height `rnd_he` held at the start of the segment, [lsp_generate](geometry-based-stochastic-model.md#lsp_generate) (SF and the K-factor ratio), [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) for building and car UTs with the model of the link state and the deviation at the snapshot; evaluated once for a link whose ends do not move
- [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) with the node antennas at the frequency (entry `i_freq` of the list of bands of the antenna in `layout["antenna"]`, or its one entry for all bands), the generic antenna with two co-located isotropic elements of vertical and horizontal polarization for nodes without one and for antennas that are not evaluated (an antenna spaced in wavelengths that is not evaluated raises an error), and the Tx positions moved by the wrap image of the link; rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency `i_freq` (`"quadriga"`)
- Tracks with several segments: the snapshots of a segment are those whose row (see [sample_track](#sample_track)) is in it; each segment also covers the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row; [crossfade_segments](channel-functions.md#crossfade_segments) joins them. Relative delays: in `"phase"` the paths of each segment relative to its own LOS, in `"drift"` relative to the LOS at the snapshot. The selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing `times`

### Usage:
```python
channels, index = quadriga_lib.layout.get_gbsm_channels( layout, rays, *, i_link, times, output, mobility, oxygen, \
    use_absolute_delays, overlap, stack )
```

### Inputs:
- **`layout`** — Layout dict with links from [init_link_states](#init_link_states); keys in [layout.csv_read](#layout.csv_read); raises an error if a member that determines the links changed since then
- **`rays`** — Rays of the links from [generate_rays](#generate_rays), possibly edited, with the frequencies of the layout; list of `n_out` lists of `N` ray dicts (a ray dict alone for a link with one), `N = 1` or `n_freq`
- **`i_link`** — The propagation links of `rays`, 0-based; `None` = all links; default: None; `(n_out,)`
- **`times`** — Snapshot times in [s], finite, the same for all nodes; `None` = one snapshot per row; default: None; `(n_snap,)`
- **`output`** — `"taps"` or `"rays"`, see [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel); default: `"taps"`
- **`mobility`** — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods, drift for `"quadriga"`); default: `"auto"`
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `False`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `False`
- **`overlap`** — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1], see [crossfade_segments](channel-functions.md#crossfade_segments); default: 0.5
- **`stack`** — Snapshot layout of the channels as in [hdf5_read_channel](channel-functions.md#hdf5_read_channel); default: `False`

### Outputs:
- **`channels`** — Channels; list of channel dicts, keys in [hdf5_read_channel](channel-functions.md#hdf5_read_channel)
- **`index`** — Tx node, Rx node and frequency index of each channel, 0-based; int64; `(3, n_channel)`

### See also:
- [generate_rays](#generate_rays) (stage 2)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="init_link_states"></a>
## init_link_states
Draws the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#init_link_states) · [MATLAB](../mex/network-layouts-and-mobility.md#init_link_states)

- One propagation link per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major (all sites of terminal 0 first); co-located pairs have no link, a later segment that starts at the site raises an error. `layout_in["pairing"]` does not limit the links, it selects at the output of the channel functions
- Scenario of a link: the value of the terminal, else that of the site; a scenario family (e.g. `UMa`) draws the LOS and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes them, any other name is a user conf file used in either state without O2I loss. A conf that several families list (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found with [conf_file_list](geometry-based-stochastic-model.md#conf_file_list), the folders of `config_dirs` first
- Wrap image: the image of the site (`layout["wrap"]`) with the shortest 2D distance to the first position of the terminal, ties to the lowest index in the order 0, +a1, −a1, +a2, −a2, +(a1 − a2), −(a1 − a2); all distances of the link use this virtual site position
- O2I state per terminal from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3): building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the O2I model and `d2d_in` as in [o2i_loss](geometry-based-stochastic-model.md#o2i_loss). The values `indoor`, `floor` and `d2d_in` of the Rx side fix them; an exact conf name that contradicts a fixed `indoor` raises an error. All links of a terminal share one O2I state (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a site fixes it for the links to all sites; InH and InF have none
- `o2i_low` applies to the families with a low-loss and a high-loss building model (UMa, UMi); RMa and SMa have one building model
- LOS state per link: a uniform field per site, image and family at the first position of the segment, at the `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [los_probability](geometry-based-stochastic-model.md#los_probability) at `d2d_out = max(d2D − d2d_in, 0)`; InF-HH is always LOS
- Random streams: [sos.rand](sum-of-sinusoids-random-number-generator.md#sos.rand) with the preset `Exp256` and seeds hashed from a base seed and a key (64-bit FNV-1a, sign bit cleared): base = the site seed or the layout seed; keys `los|<site>|<image>|<family>`, `gbsm|<site>|<image>|<conf>` (the seed of [gbsm.init_phases](geometry-based-stochastic-model.md#gbsm.init_phases)) and, with the layout seed, `<purpose>|<family>` for the O2I fields. Adding a terminal does not change the states of the others
- The TR methods draw the cluster and ray sets of frequency `k` with the index `k` into `layout["center_freq"]`, so a frequency inserted below another changes the rays of the higher bands
- Segments: on the Rx side, a row whose `scenario` or `indoor` value differs from the row before it starts a segment (the values of the Rx side are compared: an empty `scenario` differs from an explicit one), with distances to a moving site at the time of the segment start, with its own propagation links (order: terminal, site, segment), LOS state, indoor state, conf and seed, drawn at its first row; the wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are those of the first row of the track, and a fixed `d2d_in` must be equal at all segment starts. A segment of a long track keeps its LOS state, so long tracks need explicit segments
- A layout without any pair farther apart than 1e-4 m gives links without entries; the later stages then return empty results
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged; the folder of the shipped conf files is added to `config_dirs` by the wrapper

### Usage:
```python
layout_out = quadriga_lib.layout.init_link_states( layout_in, *, method, o2i_low, pol_variability, anchor_freq, \
    sos_preset, seed, config_dirs )
```

### Inputs:
- **`layout_in`** — Layout dict, valid, with `center_freq`; keys in [layout.csv_read](#layout.csv_read)
- **`method`** — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see [generate_rays](#generate_rays); default: `"38.901"`
- **`o2i_low`** — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1], `(1,)`; the other families have one building model; `None` = value of the family; default: None
- **`pol_variability`** — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only; default: `False`
- **`anchor_freq`** — Anchor frequency of `"38.901_MF"` in [Hz]; default: 2e9
- **`sos_preset`** — SOS preset of the GBSM random sets, see [sos.preset](sum-of-sinusoids-random-number-generator.md#sos.preset); `""` = `Exp256` (TR methods), `Comb256` (`"quadriga"`); default: `""`
- **`seed`** — Master seed, >= -1; -1 = `layout_in["seed"]`, if that is -1 one draw of the system random device. The seed used is written to `layout_out["seed"]`; default: -1
- **`config_dirs`** — Folders of user conf files, searched before the shipped folder; str or list of str; default: None

### Outputs:
- **`layout_out`** — The layout with `links` and `seed` set; the rest is unchanged. A copy of `layout_out["links"]` stays usable after another realisation: the links hold their master seed, which the later stages use

### See also:
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (stages 2 and 3)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="layout.csv_read"></a>
## layout.csv_read
Read the Tx and Rx CSV files of a layout

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout_csv_read) · [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_read)

- The nodes of each file are created in order of first appearance; the rows of a node need not be consecutive in the file, and their order is the track order. Angles are converted from degrees to radians
- Columns, the meaning of empty cells, antenna columns and bands are described at the top of this section. Antenna columns are composed into `antenna_cols`, `antenna_args` and `i_antenna`; equal descriptions share one entry, Tx nodes first. A relative `qdant_file` is resolved against the folder of the CSV file and stored as a normalised absolute path
- Simulation frequencies: the argument, else the `center_freq` values of the Tx file, else those of the Rx file; a node with bands must define all of them, and the values in the files must match the argument within a relative 1e-6
- Unknown columns are kept and reported; a column in the wrong file (e.g. `pairing` in the Tx file), a value outside its range, an antenna column that the antenna type does not use and every violation of the rules above raise an error with the file name, line and column; so does a terminal whose empty `pairing` cell leaves it without a pair, a first band without antenna cells next to a later band with antenna cells, and a layout that is not valid, e.g. a name with a comma or a node with one row, `relative` = 1 and no direction
- With `eval_antennas`, the arrays `antenna` are generated as [eval_antennas](#eval_antennas) does; a line that [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without frequencies raise an error
- Large arrays on the default 1° grid need much memory (`3GPP` with `panel` 16,16,1,4 holds about 2 GB per band): give a `res` column or read with `eval_antennas=False`

### Usage:
```python
layout, extra_tx, extra_rx = quadriga_lib.layout.csv_read( fn_tx, fn_rx, *, center_freq, eval_antennas )
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = no nodes on that side
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `None` = from the files; `(n_freq,)`; default: None
- **`eval_antennas`** — `True`: generate the antennas; `False`: `antenna` stays empty, `antenna_args` and `i_antenna` are kept, and the nodes use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `True`

### Outputs:
- **`layout`** — The layout, a dict with the members of the layout class, which all functions of this section take and return; `name` and `links` are empty; angles in rad, all indices 0-based unless noted. Keys of `layout`, of the node dicts `layout["tx"]` and `layout["rx"]` and of `layout["links"]` (a key that is missing or None in an input dict takes the value of an empty member, unknown keys are rejected):
  | Key | Meaning | Type / Size |
  | --- | ------- | ----------- |
  | `name` | Name of the layout, not stored in the CSV files | str |
  | `tx`, `rx` | Nodes of the Tx (BS, AP) and Rx (UT, STA) side, see below | dict |
  | `links` | State of the propagation links of the current realisation, see below; `{}` = none | dict |
  | `seed` | Seed of the layout, -1 = not set | int |
  | `center_freq` | Simulation frequencies in [Hz], strictly ascending | `(n_freq,)` |
  | `antenna` | Evaluated antennas, entry `k` is a list of N arrayant dicts as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), one per band, N = 0 (not evaluated), 1 (all bands) or `n_freq` | list `[n_antenna]` of lists |
  | `antenna_cols` | Names of the antenna columns of the CSV files | list of str |
  | `antenna_args` | CSV lines of the antenna columns of each antenna: none, one for all bands or one per band; `""` = an evaluated array without description | list of lists of str |
  | `wrap` | Lattice vectors of the wrap-around in [m], one per column; empty = none | `(2, 1)` or `(2, 2)` |
  | `pairing` | Tx site index and Rx terminal index of the linked pairs, 0-based; empty = all pairs | int64, `(2, n_pair)` |
  | `name` (`tx`, `rx`) | Site or terminal name of each node | list of str `[n_node]` |
  | `sector` | Sector index | uint32, `(n_node,)` |
  | `i_antenna` | Index into the antenna lists of the layout, 1-based, 0 = generic antenna of two co-located polarized isotropic elements | uint32, `(n_node,)` |
  | `offset` | First row of each node, 0-based | int64, `(n_node,)` |
  | `pos` | Positions in [m] | `(3, n_row)` |
  | `orientation` | Bank, tilt, heading in [rad]; a positive tilt points down | `(3, n_row)` |
  | `direction` | Azimuth, elevation of travel in [rad]; NaN azimuth = follow the orientation; empty = all NaN | `(2, n_row)` |
  | `speed` | Speed in [m/s] from this row on; empty = 0 | `(n_row,)` |
  | `time` | Time in [s]; NaN on all rows of a node = not given; empty = none | `(n_row,)` |
  | `scenario` | Scenario family or conf name, `""` = none; empty = none | list of str `[n_row]` |
  | `indoor` | -1 = draw, 0 = outdoor, 1 = building, 2 = car, 3 = residential building; empty = all -1 | int32, `(n_row,)` |
  | `floor` | Floor number from 1, -1 = from z; empty = all -1 | int32, `(n_row,)` |
  | `d2d_in` | Indoor 2D distance in [m], NaN = draw; empty = all NaN | `(n_row,)` |
  | `clutter` | `d_clutter`, `r_clutter`, `h_clutter`, NaN = model default; empty = all NaN | `(3, n_node)` |
  | `scenpar_override` | Overrides of scenario parameters, one key per parameter, NaN = none | dict of `(n_node,)` |
  | `seed` (`tx`, `rx`) | Seed of the site, -1 = derived from the layout seed; empty = all -1 | int64, `(n_node,)` |
  | `relative` | 1 = orientation relative to the direction of motion; empty = all 0 | uint32, `(n_node,)` |
  | `tx.cell` | 1 = the site has drop cells, 0 = none of its own ([drop_terminals](#drop_terminals)); empty = all 1 | uint32, `(n_node,)` |
  | `extra_names`, `extra` | Names of unknown CSV columns and their cells as text | list of str `[n_extra]`; list `[n_extra]` of lists `[n_row]` of str |
  | `i_tx`, `i_rx`, `i_segment` (`links`) | Site, terminal and segment of each propagation link, 0-based | uint32, `(n_link,)` |
  | `image` | Wrap-around image of the site, 0 = the real site | uint32, `(n_link,)` |
  | `i_conf`, `conf_names` | Index into `conf_names` for each link, 0-based; resolved conf names | uint32, `(n_link,)`; list of str |
  | `los`, `indoor`, `floor_id`, `d2d_in` | 1 = LOS; indoor state at the start of the segment; floor number - 1; indoor 2D distance in [m] | `(n_link,)` |
  | `o2i_rnd`, `o2i_model` | Random inputs of [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) and the chosen building model (0-based) of each terminal | `(n_terminal, 4)`; uint32, `(n_terminal,)` |
  | `gbsm_seed`, `seed` (`links`) | Seed of the random phases of each link; master seed | int64, `(n_link,)`; int |
  | `method`, `sos_preset`, `anchor_freq`, `pol_variability` | Method and SOS preset as used, anchor frequency in [Hz], polarization power variability | str, str, float, bool |
  | `conf_files`, `layout_hash` | Resolved paths of the conf files; hash of the layout members that determine the links | list of str; int |
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; list of str

### See also:
- [layout.csv_write](#layout.csv_write) (stores a layout)
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the antenna types and arguments of the antenna columns)

---
<a id="layout.csv_write"></a>
## layout.csv_write
Write a layout to the Tx and Rx CSV files

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout_csv_write) · [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_write)

- Writes only columns with content other than the default, in the order of the format, unknown columns last; angles in degrees, numbers with up to 15 significant digits (16 or 17 where 15 do not read back); reading the files gives the same layout within these digits, except for `name` and `links`, which are not stored
- With `compact` the cells that the reader repeats from the previous row of the node (held values) or sets to the default (fixed values, which are written on the first row only) are left empty; an extra sector (sector above 0 of a site that has a sector 0) has all its cells in its first row, because an empty cell would be the value of sector 0
- Antennas are written as their CSV lines, with each `qdant_file` relative to the folder of its CSV file. If no antenna line carries the frequencies of the layout, they are declared on the first node: `center_freq` on its first row and one band row per further frequency
- All files are composed first and written to temporary files next to the targets; each target is replaced by a rename only after all files were composed and written, so an error in the layout, the composition or the writing leaves all files unchanged (a failing rename of the second file is not undone for the first)
- Antenna bands that are evaluated arrays without a line (set with [set_antenna_array](#set_antenna_array)) are written to QDANT files next to the first written CSV file that uses the antenna, named `<stem>_<k>.qdant` (`<stem>` is the name of the CSV file without `.csv`, k = 1, 2, ... per CSV file, skipping names that `qdant` lines of the layout use), one file per antenna with one entry per exported band, and become `qdant` lines. Afterwards, the files `<stem>_<k>.qdant` next to a written CSV file that the new files do not reference are deleted; files of other names are never touched. These arrays must be evaluated: a layout whose `antenna` does not hold them raises an error
- A band set with [set_antenna_qdant](#set_antenna_qdant) is a `qdant` line: it is written as a reference to its file, relative to the CSV file, and not exported
- The `seed`, `layout_seed` and `wrap` of the layout are stored in the Tx file; a Tx file for a layout without Tx nodes that has them raises an error. Not expressible in the format and therefore rejected: a scenario that returns to none after a scenario was set, a terminal without a pair in `pairing` (an empty cell means all sites). An extra sector without antenna next to a sector 0 with one is written with the antenna `none`

### Usage:
```python
quadriga_lib.layout.csv_write( fn_tx, fn_rx, layout, *, compact )
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files, str or path-like; `""` = not written; different files (also if the paths are spelled differently)
- **`layout`** — Layout dict, must be valid; fields in [layout.csv_read](#layout.csv_read)
- **`compact`** — `True`: empty cells where the format allows; `False`: every cell is written; default: `True`

### See also:
- [layout.csv_read](#layout.csv_read) (reads a layout)

---
<a id="sample_track"></a>
## sample_track
Positions and orientations of a node at snapshot times

Also in: [C++](../cpp/network-layouts-and-mobility.md#sample_track) · [MATLAB](../mex/network-layouts-and-mobility.md#sample_track)

- A **snapshot** is one sample of the channel in time and space; a **row** is a point stored in the layout. Without `times`, each row of the node is one snapshot. With `times`, a point drop gives as many snapshots as times, and a track is interpolated between its rows
- Point drop (one row): the position is `p0 + v d t` with the speed `v` of the row and the unit vector `d` of its direction (azimuth, elevation; an elevation of NaN is 0). Without direction (NaN azimuth), `d = [cos(tilt) cos(heading), cos(tilt) sin(heading), -sin(tilt)]` from the orientation; bank has no effect. The orientation is constant
- Track (several rows): the time of each row is `time`, else it follows from the arc length and the speed of the row, `t(0) = 0`, `t(i+1) = t(i) + |Δp| / speed(i)`; a step of zero length with speed 0 takes no time. The position is interpolated linearly in time, bank and heading of the orientation along the shorter way and the tilt linearly; before the first and after the last time, the first and the last row are held. If several rows have the same time, the later row holds at that time. The times in `times` need not increase. A track with neither `time` nor a speed raises an error if `times` is given; a moving step with speed 0 raises an error
- Mounted antennas (`relative = 1`): the orientation is relative to the direction of motion, and the output is `R(motion) R(orientation)` in the GCS, with bank 0 in `R(motion)`. For a point drop, the motion is its direction, which must be given. For a track, it is the direction of the step of the row segment that holds the snapshot (the step from a row to the next one, the last step repeated at the end); a step of zero length takes the previous direction, at the start the next one that exists. The stored orientation is interpolated first, then rotated by the motion of the segment
- Members of `nodes` with sizes that do not fit, and a `direction` on a track raise an error; the other rules of a valid layout (see [layout.csv_read](#layout.csv_read)) are not checked

### Usage:
```python
pos, orientation, i_row = quadriga_lib.layout.sample_track( nodes, i_node, times )
```

### Inputs:
- **`nodes`** — Nodes of one side of a layout, e.g. `layout["rx"]`; dict, keys in [layout.csv_read](#layout.csv_read)
- **`i_node`** — Node index, 0-based; int
- **`times`** — Snapshot times in [s], finite; empty = one snapshot per row; `(n_snap,)`

### Outputs:
- **`pos`** — Positions in [m]; `(3, n_snap)`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation; `(3, n_snap)`
- **`i_row`** — Row at or before each snapshot, 0-based, index into the arrays of `nodes` (not into the rows of the node); int64; `(n_snap,)`

### Example:
```python
import numpy as np, quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
pos, ori, i_row = qd.layout.sample_track( layout["rx"], 0, np.arange(0, 1.01, 0.1) )
```

### See also:
- [generate_track](#generate_track) (creates tracks)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="set_antenna"></a>
## set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_antenna) · [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna)

- This form stores the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (see there for their meaning) as an antenna line of the layout, which [layout.csv_write](#layout.csv_write) writes as CSV cells; an array set with [set_antenna_array](#set_antenna_array) is exported to a QDANT file by [layout.csv_write](#layout.csv_write)
- Without `center_freq`, the antenna applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout["center_freq"]` is added, the other bands of the nodes keep their antenna, and nodes that shared the old entry keep it; until all bands of a node are set, the layout is not valid and the functions that check it name the node
- A node set band by band keeps one line per band, also when the lines are equal, so a band added later is reported as not set for it; in a layout with one band, the line of a node set for that band also covers bands added later
- The antenna lists are encoded again after the call as [layout.csv_read](#layout.csv_read) composes them: equal lines share one entry, entries are numbered in order of first use (Tx nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- An array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- A failed call leaves the layout unchanged
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### Usage:
```python
layout_out = quadriga_lib.layout.set_antenna( layout_in, type, args, *, rx_side, i_node, center_freq, evaluate )
```

### Inputs:
- **`layout_in`** — Layout dict, keys in [layout.csv_read](#layout.csv_read); its antenna lists must fit its nodes and frequencies
- **`type`** — str; antenna type of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), e.g. `"3GPP"`, `"dipole"`, or `"none"` (no antenna, for all bands, without `args`)
- **`args`** — Dict with the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) as keys: `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers, vectors or matrices (`steer`, `beams` as there), `separate_beams` and `apply_weights` as bool or 0 or 1; a key that the type does not use raises an error; `pattern` is not available (an array built with it is set with [set_antenna_array](#set_antenna_array)); the key `center_freq` is the same as the argument; e.g. `{"panel": [4, 4, 1, 2], "pol": 3}`; default: None (none)
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none, the nodes of new antennas use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `True`

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```python
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna( layout, "3GPP", {"panel": [4, 4, 1, 2], "pol": 3}, center_freq=3.5e9 )
```

### See also:
- [set_antenna_array](#set_antenna_array) (sets an evaluated array)
- [set_antenna_qdant](#set_antenna_qdant) (sets an entry of a QDANT file)
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the meaning of the arguments)
- [layout.csv_write](#layout.csv_write) (writes the lines and exports the arrays)

---
<a id="set_antenna_array"></a>
## set_antenna_array
Set the antenna of layout nodes from an evaluated array

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna_array) · Not in: C++

- Stores an array, e.g. a measurement read with [qdant_read](antennas-transducers-and-patterns.md#qdant_read), which [layout.csv_write](#layout.csv_write) exports to a QDANT file; for the antenna lists of the layout, the bands and the renumbering of `i_antenna` see [set_antenna](#set_antenna)
- Without `center_freq`, the array applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout_in["center_freq"]` is added, the other bands of the nodes keep their antenna
- The C++ function `set_antenna` with an array argument has this name in Python, because Python does not overload by type; the in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### Usage:
```python
layout_out = quadriga_lib.layout.set_antenna_array( layout_in, array, *, rx_side, i_node, center_freq )
```

### Inputs:
- **`layout_in`** — Layout dict, keys in [layout.csv_read](#layout.csv_read); its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, dict as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna)
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = the same array for all bands; default: NaN

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```python
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
array, _ = qd.arrayant.qdant_read( "handset.qdant" )
layout = qd.layout.set_antenna_array( layout, array, rx_side=True )
```

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads an array from a QDANT file)

---
<a id="set_antenna_qdant"></a>
## set_antenna_qdant
Set the antenna of layout nodes to an entry of a QDANT file

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_antenna_qdant) · [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna_qdant)

- Stores a `qdant` antenna line with the file as normalised absolute path and the entry `qdant_id`, as [layout.csv_read](#layout.csv_read) composes it from the columns `antenna`, `qdant_file` and `qdant_id`; [layout.csv_write](#layout.csv_write) writes this line with the path relative to the CSV file and exports no copy of the array
- The array is read as stored, for all bands; bands, the sharing of entries and the renumbering of `i_antenna` follow [set_antenna](#set_antenna)
- A missing or unreadable file raises an error; without evaluation, an entry that the file does not have raises an error when the antennas are evaluated
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### Usage:
```python
layout_out = quadriga_lib.layout.set_antenna_qdant( layout_in, qdant_file, qdant_id, *, rx_side, i_node, center_freq, evaluate )
```

### Inputs:
- **`layout_in`** — Layout dict, keys in [layout.csv_read](#layout.csv_read); its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — str or path; existing QDANT file, absolute or relative to the current folder
- **`qdant_id`** — int; 1-based entry of the file; default: 1
- **`rx_side`** — `False`: nodes of `layout_in["tx"]`; `True`: nodes of `layout_in["rx"]`; default: `False`
- **`i_node`** — Indices of the nodes of that side, 0-based; int or `(n,)`; `None` = all nodes of the side; default: `None`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `True`: generate all antennas of the layout that are not evaluated yet; `False`: none; default: `True`

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```python
import quadriga_lib as qd
layout, _, _ = qd.layout.csv_read( "tx.csv", "rx.csv" )
layout = qd.layout.set_antenna_qdant( layout, "measured/handset.qdant", 2, rx_side=True )
```

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna)
- [set_antenna_array](#set_antenna_array) (sets an evaluated array)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads the entries of a QDANT file)

---
<a id="set_orientation"></a>
## set_orientation
Set the orientation of nodes from their motion, or to a given value

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_orientation) · [MATLAB](../mex/network-layouts-and-mobility.md#set_orientation)

- The orientation is that of the antenna: bank, tilt, heading = (γ, β, α) of 3GPP TR 38.901; a positive tilt points down (aviation pitch is -tilt), the heading is counted counter-clockwise from the x-axis
- `force = False`, track (several rows): bank 0, tilt and heading from the steps between the rows (forward difference, the last step repeated, or for a closed track the first row, as `qd_track.calc_orientation` of the NG library does); `orientation` is added as an offset and the angles are wrapped to [-π, π]. A step of zero length keeps the previous orientation, at the start the next one that exists; a vertical step has the tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising track has a negative tilt. A track that does not move keeps its orientation (plus the offset)
- `force = False`, point drop (one row): the orientation stays and receives the offset; without an offset it is not touched. Random orientations of point drops are not generated here
- `force = False` leaves nodes with `relative = 1` unchanged, offset included: their orientation is relative to the motion, which [sample_track](#sample_track) applies at each snapshot
- `force = True`: every row of the nodes gets exactly `orientation`, also the nodes with `relative = 1`
- A failed call leaves `nodes` unchanged
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### Usage:
```python
nodes_out = quadriga_lib.layout.set_orientation( nodes_in, *, orientation, i_node, force )
```

### Inputs:
- **`nodes_in`** — Nodes of one side of a layout, e.g. `layout["rx"]`; dict, keys in [layout.csv_read](#layout.csv_read)
- **`orientation`** — Bank, tilt, heading in [rad], 3 finite values, `(3,)`; `None` = 0; default: None
- **`i_node`** — Node indices, 0-based; int or `(n,)`; `None` = all nodes; default: None
- **`force`** — `False`: orientation from the motion of tracks, plus `orientation` as offset; `True`: `orientation` on all rows; default: `False`

### Outputs:
- **`nodes_out`** — The nodes with the new orientations

### Example:
```python
import quadriga_lib as qd
layout["rx"] = qd.layout.set_orientation( layout["rx"] )                          # The antennas look along the motion
layout["rx"] = qd.layout.set_orientation( layout["rx"], orientation=[0, 0, 0.5] )  # The same, turned by 0.5 rad
```

### See also:
- [generate_track](#generate_track) (creates tracks with this orientation)
- [sample_track](#sample_track) (applies the motion to mounted antennas)
