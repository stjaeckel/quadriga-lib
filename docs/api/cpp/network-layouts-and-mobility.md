# Network layouts and mobility

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-network-layouts-and-mobility

| Function | Summary |
| --- | --- |
| [layout](#layout) | Class for a network layout: Tx and Rx nodes, antennas, frequencies and link states |
| [layout_links](#layout_links) | State of the propagation links of one realisation of a layout |
| [layout_nodes](#layout_nodes) | One side of a layout: nodes with their rows, in flat arrays |
| [attach_terminals](#attach_terminals) | Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP |
| [drop_terminals](#drop_terminals) | Drop UTs into the cells of the Tx side of a layout |
| [export_rays_qrt](#export_rays_qrt) | Writes the rays of a layout with their link gain to a QRT file |
| [generate_layout](#generate_layout) | Network layout from a preset: BS sites and a drop of UTs |
| [generate_rays](#generate_rays) | Rays of propagation links of a layout (stage 2 of the channel pipeline) |
| [generate_track](#generate_track) | Replace the rows of nodes by a generated track or a point drop |
| [get_channels_layout](#get_channels_layout) | Channel coefficients of a layout in one call |
| [get_gbsm_channels](#get_gbsm_channels) | Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline) |
| [init_link_states](#init_link_states) | Draws the discrete state of the propagation links of a layout (stage 1 of the channel pipeline) |
| [layout_csv_read](#layout_csv_read) | Read the Tx and Rx CSV files of a layout |
| [layout_csv_write](#layout_csv_write) | Write a layout to the Tx and Rx CSV files |
| [sample_track](#sample_track) | Positions and orientations of a node at snapshot times |
| [set_antenna](#set_antenna) | Set the antenna of layout nodes from the arguments of generate_antenna or from an evaluated array |
| [set_antenna_qdant](#set_antenna_qdant) | Set the antenna of layout nodes to an entry of a QDANT file |
| [set_orientation](#set_orientation) | Set the orientation of nodes from their motion, or to a given value |

A **layout** describes the nodes of a radio network and how they move: where each transmitter (Tx) and receiver (Rx) is, which scenario applies there and which antenna it has. It is stored in two flat tables, one per side, and in two CSV files with the same format.

**Terms.** A *node* is one (`name`, `sector`) pair on one side. It has one antenna and one *track*: its *rows*, each a position with orientation and further values that apply from that row on. A node with one row is a *point drop*; it stays at its position, or, with a speed, moves along its direction. A node with several rows is a track and moves along its positions. A *site* is all Tx nodes with one name; its sectors share position and scenario. A *terminal* is all Rx nodes with one name, e.g. the panels of one device. Angles are in degrees in the CSV files and in radians in the classes. The orientation (bank, tilt, heading) is that of the antenna, a positive tilt points down; the optional direction (azimuth, elevation of travel) is that of the movement, and an unset direction follows the orientation.

**CSV files.** [layout_csv_read](#layout_csv_read) and [layout_csv_write](#layout_csv_write) read and write the files. The first line is the header; the columns are optional, in any order, except `name`, `x`, `y`, `z`. Cells follow the syntax of the other CSV files of the library: quoting with `"`, lines starting with `#` are comments, numbers are decimal (NaN and Inf in any case). Empty cells follow the class of the column:

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

**Antennas.** The antenna columns of a node (type `antenna` as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), or `qdant` with the file `qdant_file` relative to the CSV file and its 1-based entry `qdant_id`) are kept as one text line per antenna in `antenna_args`, with the names of the columns in `antenna_cols`. Nodes with equal lines share one entry of the lists, Tx nodes first, and `i_antenna` is the 1-based index of it, 0 for the generic antenna with two co-located isotropic elements of vertical and horizontal polarization. A column that the type does not use is an error. An extra sector without antenna cells has the antenna of sector 0; the type `none` (no other antenna cells) gives a node no antenna, e.g. an extra sector next to a sector 0 with one. The member function `eval_antennas` of [layout](#layout) generates the arrays `antenna` from the lines: a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored. [set_antenna](#set_antenna) sets the antenna of nodes from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) or from an evaluated array, which [layout_csv_write](#layout_csv_write) writes to a QDANT file next to the CSV file; [set_antenna_qdant](#set_antenna_qdant) sets an entry of a QDANT file, which the writer refers to.

**Bands.** A row with `center_freq` and no position adds a frequency, and optionally an antenna for it, to its node. The frequencies of the Tx side (else of the Rx side) are the simulation frequencies `center_freq` of the layout. A node without bands has all of them with one antenna description; a node with bands defines all of them. An empty antenna in a band repeats the antenna of the first band.

**Unknown columns** are kept as text in `extra_names` and `extra` and written back unchanged.

**Channels from a layout.** The channel coefficients are computed in three stages. [init_link_states](#init_link_states) draws the discrete state of each *propagation link*, one site and one terminal: its wrap image, scenario and conf file, LOS and indoor state and seeds, stored in `layout.links`. [generate_rays](#generate_rays) runs the chain of the section "Geometry-based stochastic model" for chosen links and returns their rays. [get_gbsm_channels](#get_gbsm_channels) applies the antennas, the motion along the tracks and the link gain, for each *channel link*, one Tx node and one Rx node; the sectors of a site share the rays of their propagation link. [get_channels_layout](#get_channels_layout) runs all three stages. [attach_terminals](#attach_terminals) chooses the serving Tx node and site of each terminal by the large-scale gain or by RSRP, for `layout.pairing`. [export_rays_qrt](#export_rays_qrt) writes the rays with their link gain to a QRT file, which gives channels for other antennas and positions without the GBSM chain. All random values are addressed: they follow from the seeds, a stream key and the position, so that the evaluation order, the batches and the thread count do not change them, and only the discrete states are stored. `layout.links` holds a hash of the layout members that determine the links (positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter, scenario overrides, wrap vectors and whether a site moves); stages 2 and 3 throw when it no longer matches, and `layout.links.reset()` discards the links. The links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after another realisation. Editing `los` or `indoor` of the links between stage 1 and stage 2 forces a state; orientations, antennas and `pairing` may change without new links.

**Indices.** All indices are 0-based: the members `offset`, `pairing` and, in `links`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model`, the index arguments of the functions and the functions that return indices. `i_antenna` is the index into the antenna lists with 0 = generic antenna; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values. The MATLAB / Octave API counts the indices from 1.

---
<a id="layout"></a>
## layout
Class for a network layout: Tx and Rx nodes, antennas, frequencies and link states

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#eval_antennas) · [Python](../python/network-layouts-and-mobility.md#eval_antennas)

- `tx` and `rx` are the two sides, each a [layout_nodes](#layout_nodes) object; the nodes of a side may share positions (sectors, panels)
- The antenna lists `antenna_cols`, `antenna_args` and `antenna` are shared by both sides; `i_antenna` of a node indexes them, starting at 1
- Written and read by [layout_csv_read](#layout_csv_read) and [layout_csv_write](#layout_csv_write); the model and the file format are described at the top of this section

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::string name` | string | Name of the layout; not stored in the CSV files |
| `layout_nodes tx`, `layout_nodes rx` | | Tx (BS or AP) and Rx (UT or STA) nodes |
| `layout_links links` | | State of the propagation links of the current realisation, see [layout_links](#layout_links) |
| `arma::sword seed` | scalar | Seed of the layout, -1 = not set |
| `arma::vec center_freq` | `[n_freq]` | Simulation frequencies in [Hz], strictly ascending |
| `std::vector<std::string> antenna_cols` | `[n_col]` | Names of the antenna columns of the CSV files |
| `std::vector<std::vector<std::string>> antenna_args` | `[n_antenna][0, 1 or n_freq]` | CSV lines of the antenna columns: one line for all bands or one per band; no line or an empty line = an evaluated array without description; a line without antenna type = a band that is not set yet |
| `std::vector<std::vector<arrayant<double>>> antenna` | `[n_antenna][0, 1 or n_freq]` | Evaluated arrays, one for all bands or one per band; an entry without arrays or an array without elements is not evaluated, its nodes use the generic antenna; empty = not evaluated |
| `arma::mat wrap` | `[2, 1]` or `[2, 2]` | Lattice vectors in [m], one per column; two vectors form a reduced basis, 0 <= a1·a2 <= min(a1·a1, a2·a2) / 2, so that ±a1, ±a2 and ±(a1 − a2) are the nearest images; empty = none |
| `arma::umat pairing` | `[2, n_pair]` | Tx site index and Rx terminal index of the linked pairs, 0-based; empty = all pairs |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_antenna()` | Number of antennas: length of `antenna_args`, or of `antenna` if there are no arguments |
| `.eval_antennas(center_freq)` | Generates the arrays of `antenna` that are not evaluated yet from `antenna_args`; a frequency list in [Hz] (optional) replaces `center_freq` and regenerates all lines; throws for a band without description and for an array spaced in wavelengths without a frequency |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks all members of both sides and `links`, the sizes and ranges, equal positions, motion (`speed`, `time`, `direction`) and shared values of the nodes of one site or terminal, the antenna lists (a node with a band that is not set is reported), the frequencies, `wrap` and `pairing` |

### See also:
- [layout_nodes](#layout_nodes) (one side of a layout)
- [layout_links](#layout_links) (link states)
- [layout_csv_read](#layout_csv_read), [layout_csv_write](#layout_csv_write) (load and store a layout)
- [set_antenna](#set_antenna), [set_antenna_qdant](#set_antenna_qdant) (set the antennas of nodes)

---
<a id="layout_links"></a>
## layout_links
State of the propagation links of one realisation of a layout

Not in: MATLAB, Python

- Holds the discrete state of the links (line of sight, indoor state, scenario and random seeds), no large-scale parameters, phases or rays; it is not stored in the CSV files
- A propagation link is one site, one terminal and one segment of a track with one resolved scenario; its arrays have one entry per link

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `arma::u32_vec i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of each link, 0-based |
| `arma::u32_vec image` | `[n_link]` | Wrap-around image of the site, 0 = the real site |
| `arma::u32_vec i_conf` | `[n_link]` | Index into `conf_names`, derived from scenario, LOS and indoor state |
| `std::vector<std::string> conf_names` | `[n_conf]` | Resolved conf names |
| `arma::u32_vec los` | `[n_link]` | 1 = line of sight |
| `arma::s32_vec indoor` | `[n_link]` | Indoor state at the start of the segment, as in [layout_nodes](#layout_nodes) |
| `arma::u32_vec floor_id` | `[n_link]` | Floor number - 1, 0 outdoors; of the drawn state, stages 2 and 3 derive it from `indoor` |
| `arma::vec d2d_in` | `[n_link]` | Indoor 2D distance in [m]; of the drawn state, stages 2 and 3 derive it from `indoor` |
| `arma::mat o2i_rnd` | `[n_terminal, 4]` | Random inputs of [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) |
| `arma::u32_vec o2i_model` | `[n_terminal]` | Building O2I model, index into the list of the family; for every terminal of a family with O2I, also outdoors |
| `arma::s64_vec gbsm_seed` | `[n_link]` | Seed of the random phases; equal for one site, image and conf |
| `arma::sword seed` | scalar | Master seed, the base of the streams of all stages; -1 = not set |
| `std::string method`, `sos_preset` | | Method and SOS preset as used; an empty `method` means not filled |
| `double anchor_freq` | scalar | Anchor frequency of the multi-frequency model in [Hz], default 2e9 |
| `bool pol_variability` | scalar | Polarization power variability, default false |
| `std::vector<std::string> conf_files` | `[n_conf]` | Resolved paths of the conf files; empty = not resolved |
| `unsigned long long layout_hash` | scalar | Hash of all layout members that determine the links |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_link()` | Number of links |
| `.empty()` | `true` if there are no links |
| `.reset()` | Empties all members and restores the defaults |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, the indices into `conf_names` and the ranges |

### See also:
- [layout](#layout) (holds the links as `links`)
- [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) (uses `o2i_rnd`)

---
<a id="layout_nodes"></a>
## layout_nodes
One side of a layout: nodes with their rows, in flat arrays

Not in: MATLAB, Python

- The nodes are in order of first appearance; the rows of node `i` are `offset(i)` to `offset(i+1)-1` (the last node to the end), so each node has at least one row
- Optional members may be empty with the meaning given below; `is_valid()` accepts both forms

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::vector<std::string> name` | `[n_node]` | Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading or trailing blank |
| `arma::u32_vec sector` | `[n_node]` | Sector index; (name, sector) is unique |
| `arma::u32_vec i_antenna` | `[n_node]` | 1-based index into the antenna lists of the layout, 0 = generic antenna |
| `arma::uvec offset` | `[n_node]` | First row of each node, 0-based |
| `arma::mat pos` | `[3, n_row]` | Positions in [m] |
| `arma::mat orientation` | `[3, n_row]` | Bank, tilt, heading in [rad] |
| `arma::mat direction` | `[2, n_row]` | Azimuth, elevation of travel in [rad]; NaN azimuth = follow the orientation; empty = all NaN; NaN on all rows of a track |
| `arma::vec speed` | `[n_row]` | Speed in [m/s], applies from this row on; empty = 0 |
| `arma::vec time` | `[n_row]` | Time in [s]; NaN on all rows of a node = not given; empty = none; a node has `time` or a speed |
| `std::vector<std::string> scenario` | `[n_row]` | Scenario family or conf name, "" = none; empty = none |
| `arma::s32_vec indoor` | `[n_row]` | -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1 |
| `arma::s32_vec floor` | `[n_row]` | Floor number from 1, -1 = derive from z; empty = all -1 |
| `arma::vec d2d_in` | `[n_row]` | Indoor 2D distance in [m], NaN = draw; empty = all NaN |
| `arma::mat clutter` | `[3, n_node]` | `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN |
| `std::unordered_map<std::string, arma::vec> scenpar_override` | each `[n_node]` | Overrides of scenario parameters by key, NaN = none |
| `arma::s64_vec seed` | `[n_node]` | Seed of the site, -1 = derived from the seed of the layout; empty = all -1 |
| `arma::u32_vec relative` | `[n_node]` | 1 = orientation relative to the direction of motion; empty = all 0 |
| `arma::u32_vec cell` | `[n_node]` | Tx side: 1 = the site has drop cells, 0 = none of its own ([drop_terminals](#drop_terminals)); equal for all sectors of a site; empty = all 1 |
| `std::vector<std::string> extra_names` | `[n_extra]` | Names of unknown CSV columns |
| `std::vector<std::vector<std::string>> extra` | `[n_extra][n_row]` | Their cells as text |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_node()` | Number of nodes |
| `.n_row()` | Number of rows of all nodes |
| `.n_row(i_node)` | Number of rows of one node (0-based, throws `std::invalid_argument` for an index outside the nodes) |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks sizes, `offset`, finite positions and orientations, names, ranges and the rules for `time`, `speed`, `direction`, `floor` and `d2d_in` |

### See also:
- [layout](#layout) (the class that holds both sides)
- [layout_csv_read](#layout_csv_read) (fills the members from a CSV file)

---
<a id="attach_terminals"></a>
## attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#attach_terminals) · [Python](../python/network-layouts-and-mobility.md#attach_terminals)

- Evaluates all Tx nodes for each terminal at the first rows of both nodes and `layout.center_freq[i_freq]`, with the link states of the first segment, the wrap image and the antennas as in [get_gbsm_channels](#get_gbsm_channels); `layout.pairing` is ignored
- `"large_scale"` (TR 38.901 Table 7.8-1): the link gain `−PL − O2I + 10 lg(SF)` plus the gain of one unit-gain LOS path from [get_channels_planar](channel-generation-functions.md#get_channels_planar) (port weights, element positions, orientations); no rays
- `"rsrp"` (TR 36.873 eq. (8.1-1)): per Rx port the sum of `|h|²` over the rays of [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (`output = "rays"`, mobility `"phase"`), link gain included, transmit power 1
- Both rules take `|h|²` of Tx port `tx_port` and average it over the Rx ports, so a dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT antenna of the TR
- A terminal takes the highest value of its nodes; the serving node has the highest value, a tie the lowest index. A Tx node without a propagation link (co-located) has −Inf; a terminal without any link gets node 0
- `layout.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, n_rx_terminal − 1))` links the sectors of the serving sites

### Declaration:
```cpp
arma::uvec quadriga_lib::attach_terminals(
    const layout &layout,
    const std::string &rule = "large_scale",
    const std::vector<std::vector<gbsm_rays>> &rays = {},
    arma::uvec *site = nullptr,
    arma::mat *gain_dB = nullptr,
    arma::uword i_freq = 0,
    arma::uword tx_port = 0);
```

### Inputs:
- **`layout`** — Layout with links from [init_link_states](#init_link_states); Tx nodes required if there are terminals
- **`rule`** *(optional)* — `"large_scale"` or `"rsrp"`
- **`rays`** *(optional)* — `"rsrp"`: rays of all links from [generate_rays](#generate_rays), possibly edited; `[n_link][1 or n_freq]`
- **`i_freq`** *(optional)* — Frequency index, 0-based
- **`tx_port`** *(optional)* — Port of the Tx antennas, 0-based

### Outputs:
- **`site`** *(optional)* — Serving site per terminal, 0-based; `[n_rx_terminal]`
- **`gain_dB`** *(optional)* — Coupling gain or RSRP in [dB]; `[n_rx_terminal, n_tx_node]`

### Returns:
- Serving Tx node per terminal, 0-based; `[n_rx_terminal]`

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays) (link states and rays)
- [get_gbsm_channels](#get_gbsm_channels) (channels of the linked pairs)

---
<a id="drop_terminals"></a>
## drop_terminals
Drop UTs into the cells of the Tx side of a layout

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#drop_terminals) · [Python](../python/network-layouts-and-mobility.md#drop_terminals)

- Cells: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the wrap images when `layout.wrap` is set. A site with several sectors is split into wedges by the nearest sector heading (±60° around each boresight for three sectors at equal spacing)
- Drop area: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the images 0, ±a1, ±a2 and ±(a1 − a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- The drop of [generate_layout](#generate_layout) applies in the cells of each site, with the rules of the scenario family in its `scenario` (minimum distance, heights and floors, speeds, O2I states). Sites without a family (no scenario, a conf name): height 1.5 m, no minimum distance, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states. A site without a family has no minimum distance unless `ut` entry 1 gives one, and the distance kept from a site with `cell` = 0 follows that site's own family
- Sites with `cell` = 0, e.g. the micro sites of `M.2412_DenseUrban` config 3, form no cells: with `per_cell`, each of their Tx nodes adds `n_ut` UTs to the cell that holds its position, as the drop of [generate_layout](#generate_layout) does (there 4 `n_ut` per macro TRxP cell), and the UTs keep the minimum distance of the site's family (`ut` entry 1 if given) from them. A site closer than twice the minimum distance to another one can leave a cell too little room; such a cell throws
- The Rx side is replaced by the UTs `UT1`, `UT2`, ... (ordered by cell with `per_cell`) with the generic antenna; `pairing` and `links` are emptied. The Tx side, the antenna lists and `layout.seed` stay. A failed call leaves the layout unchanged
- Random streams: the positions use the key `drop`, the heading and direction of a UT `ut|<name>`, its floor `floor|<family>|<name>`, the O2I fields `<purpose>|<family>`; the base of all is the seed of the call

### Declaration:
```cpp
void quadriga_lib::drop_terminals(
    quadriga_lib::layout &layout,
    arma::uword n_ut,
    const arma::vec &bounds = {},
    const arma::vec &ut = {},
    const arma::vec &indoor = {},
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    bool per_cell = true,
    arma::sword seed = -1);
```

### Inputs:
- **`layout`** — Valid layout, updated in place; the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs
- **`bounds`** *(optional)* — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless `layout.wrap` has two vectors
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** *(optional)* — As in [generate_layout](#generate_layout); an entry or option that no family of the sites uses throws
- **`seed`** *(optional)* — Seed >= 0; -1 = `layout.seed` if it is set, else the system random device

### See also:
- [generate_layout](#generate_layout) (layouts from presets, with this drop)
- [generate_track](#generate_track) (tracks for the UTs)

---
<a id="export_rays_qrt"></a>
## export_rays_qrt
Writes the rays of a layout with their link gain to a QRT file

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#export_rays_qrt) · [Python](../python/network-layouts-and-mobility.md#export_rays_qrt)

- Separates the ray generation from the antennas: [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) (with the method of the links and the `n_interact` of the file as `path_type`) and [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (link gain 0, mobility `"drift"`) turn a CIR into channels for any arrays, orientations and positions
- One origin per site at its position, with the orientation of sector 0 (else of the first node of the site) and the site name; one destination per terminal with its name and one CIR per row (the nodes of a terminal share their positions; the CIR has the orientation of the first node)
- Per CIR: the rays of the link and of the segment that holds the row (scatterers added where missing), the link gain and the K-factor ratio of [get_gbsm_channels](#get_gbsm_channels) at the row, [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `"drift"`, [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers). A CIR read back gives the channel of [get_gbsm_channels](#get_gbsm_channels) (`output = "rays"`, `mobility = "drift"`) at the row within float32 accuracy
- A point drop with speed has one CIR; its motion is applied at the evaluation, at the link gain of the drop
- Only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- Positions in the file are local: the returned shift, the centroid of the site and CIR positions in x and y, is subtracted (z unchanged); add it to the positions read back
- The QRT format stores no tap index and no ground permittivity: a CIR gives rays, not the taps of the TR, and moving a ground-reflection ray from a file throws
- The file is written to `fn + ".part"` and renamed at the end; a failed call leaves an existing file unchanged

### Declaration:
```cpp
arma::vec quadriga_lib::export_rays_qrt(
    const std::string &fn,
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays = {});
```

### Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout with links from [init_link_states](#init_link_states): one frequency for the TR methods (`"quadriga"`: 1 to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at a site position
- **`rays`** *(optional)* — Rays of all links from [generate_rays](#generate_rays), possibly edited, one object with all frequencies per link; omitted = generated for batches of sites with up to 256 links, which bounds the memory; `[n_link][1]`

### Returns:
- Shift of the local coordinates in x and y in [m]; `[2]`

### See also:
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the rays and the link gain)
- [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths), [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (evaluation of the file)
- [qrt_paths_from_scatterers](channel-functions.md#qrt_paths_from_scatterers) (the stored paths)

---
<a id="generate_layout"></a>
## generate_layout
Network layout from a preset: BS sites and a drop of UTs

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_layout) · [Python](../python/network-layouts-and-mobility.md#generate_layout)

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
- The layout is a function of the seed, independent of the thread count, with the random streams of [drop_terminals](#drop_terminals); seed -1 draws a seed, which is stored in `layout.seed`. Antennas are not set (the channel functions use the generic antenna until [set_antenna](#set_antenna) sets one) and `center_freq` is empty

### Declaration:
```cpp
quadriga_lib::layout quadriga_lib::generate_layout(
    const std::string &type,
    arma::uword n_ut = 0,
    const arma::vec &spacing = {},
    const arma::vec &grid = {},
    const arma::vec &area = {},
    const arma::vec &bs = {},
    const arma::vec &ut = {},
    const arma::vec &indoor = {},
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    bool per_cell = true,
    arma::sword seed = -1);
```

### Inputs:
- **`type`** — Preset name
- **`n_ut`** *(optional)* — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs
- **`spacing`** *(optional)* — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used), the spacing of the indoor grid or of the urban road grid
- **`grid`** *(optional)* — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site; `n_y` not used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with boresights 30° (indoor 0°) + k · 360° / `n_sector`
- **`area`** *(optional)* — `[length, width, height]` of the hall or road area along x, y and z in [m], > 0; indoor and road presets, the height InF only, the width not for the highway; the BS grid must fit into it
- **`bs`** *(optional)* — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt (positive down) and an offset added to the boresights
- **`ut`** *(optional)* — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg], [deg], [deg]; a value replaces the rule of the preset for all UTs (building UTs get the floor from a given height; the speed of the lanes also sets the gaps); min distance (not for roads) and speed >= 0 (with the d2D-out rule, a building UT of UMi and UMa needs d2D >= d2D-in also at min distance 0). Calibration orientations: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C tilt 45°, config D tilt 0° and heading 0°
- **`indoor`** *(optional)* — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1] (presets with O2I states); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa, Dense Urban); floor height in [m], > 0 (UMi, UMa, SMa, Dense Urban). With a given ratio, `M.2412_Rural` config 3 keeps one third of the other UTs in cars
- **`options`** *(optional)* — One value per key: `p_residential` (SMa), the share of residential buildings in [0, 1]; `config` (M.2412), `service` (`M.2412_UrbanMacro`), `h_micro` (`M.2412_DenseUrban` config 3) in [m]; `lane_width` in [m], `n_lane` (1 to 100), `p_truck` in [0, 1] (TR 37.885); other keys throw
- **`per_cell`** *(optional)* — true: `n_ut` UTs in each cell; false: `n_ut` UTs uniform over all cells; no effect for the TR 37.885 presets
- **`seed`** *(optional)* — Seed >= 0; -1 = system random device

An entry of a tuple that is NaN or omitted takes the value of the preset; an entry or option that the preset does not use throws.

### Returns:
- The layout, see [layout](#layout)

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 10, {}, {}, {}, {}, {NAN, NAN, NAN, 0.0, 90.0}, {}, {}, true, 42);
quadriga_lib::set_antenna(L, "3gpp", {{"panel", {4, 4, 1, 2}}});
```

### See also:
- [drop_terminals](#drop_terminals) (the same drop for any Tx side)
- [generate_track](#generate_track) (tracks for the UTs)
- [set_antenna](#set_antenna) (antennas of the nodes)
- [layout](#layout) (the members)

---
<a id="generate_rays"></a>
## generate_rays
Rays of propagation links of a layout (stage 2 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_rays) · [Python](../python/network-layouts-and-mobility.md#generate_rays)

- Runs the GBSM chain of the section "Geometry-based stochastic model" for the links with one site, image, conf and seed: [gbsm_init_phases](geometry-based-stochastic-model.md#gbsm_init_phases) and [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the random sets, [path_loss](geometry-based-stochastic-model.md#path_loss) (for `rnd_he` and the SF standard deviation), [lsp_generate](geometry-based-stochastic-model.md#lsp_generate), [ssf_clusters](geometry-based-stochastic-model.md#ssf_clusters), [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) and, on request, [ssf_scatterers](geometry-based-stochastic-model.md#ssf_scatterers); at the first row of the segment of the terminal and the site (moved by the wrap image) at the same time (a moving site: sampled at the time of the row, without times on the Rx track the row of the same index)
- TR methods: preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance, `angle_model = "additive"`, `pol_model = "38.901"`, one set of clusters and rays per frequency (`"38.901_MF"`: clusters at the anchor frequency, rays per frequency). `"quadriga"`: bias removal over the positions of all links of the group, the legacy outdoor distance, `"rotation"`, `"quadriga"`, one ray set for all frequencies, always with scatterers
- A site with more than one row or a speed uses dual mobility; other sites are not affected
- The conf and seed of a link follow from its scenario and the edited `los` and `indoor` of `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged. The floor and the indoor distance follow the indoor state: building UTs on the floor of the Rx side (else from z) at the distance of the stored O2I inputs, others on floor 0 at 0 m; the members `floor_id` and `d2d_in` of the links are those of the drawn state
- The values of a link do not depend on the other requested links

### Declaration:
```cpp
std::vector<std::vector<gbsm_rays>> quadriga_lib::generate_rays(
    const layout &layout,
    const uint_list &i_link = {},
    bool scatterers = false);
```

### Inputs:
- **`layout`** — Layout with links from [init_link_states](#init_link_states); throws if a member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** *(optional)* — Propagation links, 0-based, without repetition; omitted = all (none for links without entries); `[n_out]`
- **`scatterers`** *(optional)* — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`

### Returns:
- Rays per link and frequency, `[n_out][n_freq]` with one frequency each for the TR methods, `[n_out][1]` with all frequencies for `"quadriga"`

### See also:
- [init_link_states](#init_link_states) (stage 1), [get_gbsm_channels](#get_gbsm_channels) (stage 3)
- [gbsm_rays](geometry-based-stochastic-model.md#gbsm_rays) (the ray data)

---
<a id="generate_track"></a>
## generate_track
Replace the rows of nodes by a generated track or a point drop

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_track) · [Python](../python/network-layouts-and-mobility.md#generate_track)

- Each node keeps its first position; the call applies to all nodes with the name of a given node (the sectors of a site, the panels of a terminal), which get the same track. Tracks are horizontal at the height of the first row. The other members of the new rows (`scenario`, `indoor`, `floor`, `d2d_in`, extra columns) take the values of the first row of the node; `time` is cleared
- Types (not case sensitive):
  - `point`: the nodes become point drops with `speed` and an explicit horizontal `direction` (random if not given), the form of short-term mobility in 3GPP TR 38.901; the rows after the first are removed; `length` is not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction; two points, or `ceil(length / spacing) + 1` equidistant points
  - `circular`: a closed circle with the circumference `length` (default 20π m, radius 10 m) and 129 points, or `max(3, ceil(length / spacing)) + 1` points; the first and last point are equal. `direction` is the angle of the start point on the circle seen from its centre: a positive value runs counter-clockwise, a negative one clockwise (e.g. 0 starts in the east of the circle and runs north, -π/2 starts in the south and runs west); a random direction is drawn in [0, 2π), so random circles run counter-clockwise
  - `street`: the route of a car through a city grid as in `qd_track.generate` of the NG library. Straight streets of random length (normal distribution, drawn again below `street_length_min`) are separated by crossings; at a crossing the car turns with the probability `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of length 2 `curve_radius` cos(turn / 2); the direction stays within 105° of the start direction, so the car does not drive in circles. The route has the length `length` (default 1000 m) and is sampled with a point spacing of `spacing` (default 1 m); the chords between the points are slightly shorter than the route at the corners
- `linear`, `circular` and `street` give tracks of equal `speed` without `direction`, and their orientation follows the motion as [set_orientation](#set_orientation) with `force = false` gives it; nodes with `relative = 1` keep the orientation of their first row on all rows. `point` keeps the orientation of the node
- Reproducibility: a seed >= 0 gives the same track on all platforms; each name (all its nodes) has its own stream, keyed by `track|<name>`, so the track does not depend on the other nodes or the order in `i_node`. Seed -1 uses the system random device
- A failed call leaves `nodes` unchanged

### Declaration:
```cpp
void quadriga_lib::generate_track(
    quadriga_lib::layout_nodes &nodes,
    const uint_list &i_node,
    const std::string &type,
    double length = 0.0,
    double speed = NAN,
    const arma::vec &direction = {},
    double spacing = NAN,
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    arma::sword seed = -1);
```

### Inputs:
- **`nodes`** — Nodes, updated in place
- **`i_node`** — Node indices, 0-based, not empty
- **`type`** — `point`, `linear`, `circular` or `street`
- **`length`** *(optional)* — Length in [m], finite and >= 0; 0 = default of the type
- **`speed`** *(optional)* — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of the node if it is valid (> 0 for a track); without one, `point` gets 0 and the other types throw
- **`direction`** *(optional)* — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry of `i_node` (nodes of one name need equal values); empty = uniform random in [0, 2π)
- **`spacing`** *(optional)* — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`)
- **`options`** *(optional)* — Options of `street`, one value each: `street_length_min` = 50, `street_length_mu` = 187, `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m], `turn_probability` = 0.5 (in [0, 1]); other keys, and these keys for the other types, throw
- **`seed`** *(optional)* — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 throws

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("", "rx.csv");
quadriga_lib::generate_track(L.rx, {0}, "street", 500.0, 10.0, {}, NAN, {{"turn_probability", {0.3}}}, 42);
```

### See also:
- [sample_track](#sample_track) (snapshots of a track)
- [set_orientation](#set_orientation) (orientation from the motion)
- [layout](#layout) (the members of a node)

---
<a id="get_channels_layout"></a>
## get_channels_layout
Channel coefficients of a layout in one call

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#get_channels_layout) · [Python](../python/network-layouts-and-mobility.md#get_channels_layout)

- Runs [init_link_states](#init_link_states) (if `layout.links` was not filled), [generate_rays](#generate_rays) and [get_gbsm_channels](#get_gbsm_channels) in batches of at most 256 propagation links (the segments of a track in one batch), so that the rays of a batch bound the memory; the results are those of the manual path with the default settings and do not depend on the batches
- Without links (`layout.links.method` empty, e.g. after `reset()`), the link states are drawn for this call only and the layout is unchanged; with filled links, these are used, also when edited, and `seed` and `config_dirs` are not used. No linked pair gives an empty result
- The mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers

### Declaration:
```cpp
std::vector<channel<double>> quadriga_lib::get_channels_layout(
    const layout &layout,
    const arma::vec &times = {},
    const std::string &output = "taps",
    const std::string &method = "",
    arma::umat *index = nullptr,
    const std::string &mobility = "auto",
    double overlap = 0.5,
    arma::sword seed = -1,
    const std::vector<std::string> &config_dirs = {});
```

### Inputs:
- **`layout`** — Layout with `center_freq`; antennas as evaluated in `layout.antenna`, nodes without one use the generic antenna (see [get_gbsm_channels](#get_gbsm_channels))
- **`times`** *(optional)* — Snapshot times in [s]; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` or `"quadriga"`; omitted = the method of filled links, else `"38.901"`; another method than that of filled links throws
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"`
- **`overlap`** *(optional)* — Overlap of consecutive segments, in [0, 1]
- **`seed`** *(optional)* — Master seed for empty links, as in [init_link_states](#init_link_states); with filled links only -1
- **`config_dirs`** *(optional)* — Folders of user conf files for empty links, searched before the shipped folder

### Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

### Returns:
- Channels, Tx node fastest, then Rx node, then frequency; `[n_channel]`

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1);
L.center_freq = {3.5e9};
auto channels = quadriga_lib::get_channels_layout(L);
```

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the three stages)
- [generate_layout](#generate_layout) (layouts of the TR scenarios)

---
<a id="get_gbsm_channels"></a>
## get_gbsm_channels
Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#get_gbsm_channels) · [Python](../python/network-layouts-and-mobility.md#get_gbsm_channels)

- One channel per Tx node, Rx node and frequency whose propagation link is in `i_link` and whose pair is in `layout.pairing` (empty: all); order: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the sector where all sectors of the side are 0
- Snapshots of both ends from [sample_track](#sample_track) over `times`; a node with one snapshot is repeated to the count of the other; two tracks without `times` need equal row counts
- Link gain per snapshot (see the GBSM terms): [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the `"lsf"` set at the snapshot, [path_loss](geometry-based-stochastic-model.md#path_loss) with the random environment height `rnd_he` held at the start of the segment, [lsp_generate](geometry-based-stochastic-model.md#lsp_generate) (SF and the K-factor ratio), [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) for building and car UTs with the model of the link state, the deviation of a building UT at the snapshot, that of a car UT at the first row of the terminal for its whole track (TR 38.901 eq. 7.4-4: one value per UT); evaluated once for a link whose ends do not move
- [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) with the node antennas at the frequency (inner entry `i_freq` of `layout.antenna`, or entry 0 for all bands), the generic antenna with two co-located isotropic elements of vertical and horizontal polarization for nodes without one and for antennas that are not evaluated (an antenna spaced in wavelengths that is not evaluated throws), and the Tx positions moved by the wrap image of the link; rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency `i_freq` (`"quadriga"`)
- Tracks with several segments: the snapshots of a segment are those whose row (see [sample_track](#sample_track)) is in it; each segment also covers the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row; [crossfade_segments](channel-functions.md#crossfade_segments) joins them. Relative delays: in `"phase"` the paths of each segment relative to its own LOS, in `"drift"` relative to the LOS at the snapshot. The selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing `times`

### Declaration:
```cpp
std::vector<channel<double>> quadriga_lib::get_gbsm_channels(
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays,
    const uint_list &i_link = {},
    const arma::vec &times = {},
    const std::string &output = "taps",
    arma::umat *index = nullptr,
    const std::string &mobility = "auto",
    bool oxygen = false,
    bool use_absolute_delays = false,
    double overlap = 0.5);
```

### Inputs:
- **`layout`** — Layout with links from [init_link_states](#init_link_states); throws if a member that determines the links changed since then
- **`rays`** — Rays of the links from [generate_rays](#generate_rays), possibly edited, with the frequencies of the layout; `[n_out][1 or n_freq]`
- **`i_link`** *(optional)* — The propagation links of `rays`, 0-based; omitted = all links; `[n_out]`
- **`times`** *(optional)* — Snapshot times in [s], finite, the same for all nodes; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`, see [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel)
- **`mobility`** *(optional)* — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods, drift for `"quadriga"`)
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays
- **`overlap`** *(optional)* — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1], see [crossfade_segments](channel-functions.md#crossfade_segments)

### Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

### Returns:
- Channels, see [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel); `[n_channel]`

### See also:
- [generate_rays](#generate_rays) (stage 2)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="init_link_states"></a>
## init_link_states
Draws the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#init_link_states) · [Python](../python/network-layouts-and-mobility.md#init_link_states)

- One propagation link per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major (all sites of terminal 0 first); co-located pairs have no link, a later segment that starts at the site throws. `layout.pairing` does not limit the links, it selects at the output of the channel functions
- Scenario of a link: the value of the terminal, else that of the site; a scenario family (e.g. `UMa`) draws the LOS and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes them, any other name is a user conf file used in either state without O2I loss. A conf that several families list (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found with [conf_file_list](geometry-based-stochastic-model.md#conf_file_list), the folders of `config_dirs` first
- Wrap image: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the terminal, ties to the lowest index in the order 0, +a1, −a1, +a2, −a2, +(a1 − a2), −(a1 − a2); all distances of the link use this virtual site position
- O2I state per terminal from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3): building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the O2I model and `d2d_in` as in [o2i_loss](geometry-based-stochastic-model.md#o2i_loss). The values `indoor`, `floor` and `d2d_in` of the Rx side fix them; an exact conf name that contradicts a fixed `indoor` throws. All links of a terminal share one O2I state (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a site fixes it for the links to all sites; InH and InF have none
- `o2i_low` applies to the families with a low-loss and a high-loss building model (UMa, UMi); RMa and SMa have one building model
- LOS state per link: a uniform field per site, image and family at the first position of the segment, at the `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [los_probability](geometry-based-stochastic-model.md#los_probability) at `d2d_out = max(d2D − d2d_in, 0)`; InF-HH is always LOS
- Random streams: [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) with the preset `Exp256` and seeds hashed from a base seed and a key (64-bit FNV-1a, sign bit cleared): base = the site seed or the layout seed; keys `los|<site>|<image>|<family>`, `gbsm|<site>|<image>|<conf>` (the seed of [gbsm_init_phases](geometry-based-stochastic-model.md#gbsm_init_phases)) and, with the layout seed, `<purpose>|<family>` for the O2I fields. Adding a terminal does not change the states of the others
- The TR methods draw the cluster and ray sets of frequency `k` with the index `k` into `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands
- Segments: on the Rx side, a row whose `scenario` or `indoor` value differs from the row before it starts a segment (the values of the Rx side are compared: an empty `scenario` differs from an explicit one), with distances to a moving site at the time of the segment start, with its own propagation links (order: terminal, site, segment), LOS state, indoor state, conf and seed, drawn at its first row; the wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are those of the first row of the track, and a fixed `d2d_in` must be equal at all segment starts. A segment of a long track keeps its LOS state, so long tracks need explicit segments
- A layout without any pair farther apart than 1e-4 m gives links without entries; the later stages then return empty results

### Declaration:
```cpp
void quadriga_lib::init_link_states(
    layout &layout,
    const std::string &method = "38.901",
    const arma::vec &o2i_low = {},
    bool pol_variability = false,
    double anchor_freq = 2.0e9,
    const std::string &sos_preset = "",
    arma::sword seed = -1,
    const std::vector<std::string> &config_dirs = {});
```

### Inputs:
- **`layout`** — Layout, valid, with `center_freq`; `layout.links` and `layout.seed` are written, the rest is unchanged. A copy of `layout.links` stays usable after another realisation: the links hold their master seed, which the later stages use
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see [generate_rays](#generate_rays)
- **`o2i_low`** *(optional)* — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; the other families have one building model; `[1]`
- **`pol_variability`** *(optional)* — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only
- **`anchor_freq`** *(optional)* — Anchor frequency of `"38.901_MF"` in [Hz]
- **`sos_preset`** *(optional)* — SOS preset of the GBSM random sets, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); omitted: `Exp256` (TR methods), `Comb256` (`"quadriga"`)
- **`seed`** *(optional)* — Master seed, >= -1; -1 = `layout.seed`, if that is -1 one draw of the system random device. The seed used is written to `layout.seed`
- **`config_dirs`** *(optional)* — Folders of user conf files, searched before the shipped folder

### See also:
- [layout_links](#layout_links) (the link states)
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (stages 2 and 3)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="layout_csv_read"></a>
## layout_csv_read
Read the Tx and Rx CSV files of a layout

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_read) · [Python](../python/network-layouts-and-mobility.md#layout.csv_read)

- The nodes of each file are created in order of first appearance; the rows of a node need not be consecutive in the file, and their order is the track order. Angles are converted from degrees to radians
- Columns, the meaning of empty cells, antenna columns and bands are described at the top of this section. Antenna columns are composed into `antenna_cols`, `antenna_args` and `i_antenna`; equal descriptions share one entry, Tx nodes first. A relative `qdant_file` is resolved against the folder of the CSV file and stored as a normalised absolute path
- Simulation frequencies: the argument, else the `center_freq` values of the Tx file, else those of the Rx file; a node with bands must define all of them, and the values in the files must match the argument within a relative 1e-6
- Unknown columns are kept and reported; a column in the wrong file (e.g. `pairing` in the Tx file), a value outside its range, an antenna column that the antenna type does not use and every violation of the rules above throw `std::runtime_error` with the file name, line and column; so does a terminal whose empty `pairing` cell leaves it without a pair, a first band without antenna cells next to a later band with antenna cells, and a layout that is not valid ([layout](#layout)), e.g. a name with a comma or a node with one row, `relative` = 1 and no direction
- With `eval_antennas`, the arrays `antenna` are generated as `layout.eval_antennas()` does ([layout](#layout)); a line that [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without frequencies throw `std::runtime_error`
- Large arrays on the default 1° grid need much memory (`3GPP` with `panel` 16,16,1,4 holds about 2 GB per band): give a `res` column or read with `eval_antennas = false`

### Declaration:
```cpp
quadriga_lib::layout quadriga_lib::layout_csv_read(
    const std::string &fn_tx,
    const std::string &fn_rx,
    const arma::vec &center_freq = {},
    bool eval_antennas = true,
    std::vector<std::string> *extra_tx = nullptr,
    std::vector<std::string> *extra_rx = nullptr);
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = no nodes on that side
- **`center_freq`** *(optional)* — Simulation frequencies in [Hz], strictly ascending and positive; empty = from the files; `[n_freq]`
- **`eval_antennas`** *(optional)* — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and `i_antenna` are kept, and the nodes use the generic antenna of two co-located polarized isotropic elements until `layout.eval_antennas()` is called

### Outputs:
- **`extra_tx`**, **`extra_rx`** *(optional)* — Names of the unknown columns of the Tx and Rx file, in file order; written only if the call succeeds

### Returns:
- `quadriga_lib::layout` — The layout; `name` and `links` are empty

### See also:
- [layout_csv_write](#layout_csv_write) (stores a layout)
- [layout](#layout) (the returned class)
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the antenna types and arguments of the antenna columns)

---
<a id="layout_csv_write"></a>
## layout_csv_write
Write a layout to the Tx and Rx CSV files

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_write) · [Python](../python/network-layouts-and-mobility.md#layout.csv_write)

- Writes only columns with content other than the default, in the order of the format, unknown columns last; angles in degrees, numbers with up to 15 significant digits (16 or 17 where 15 do not read back); reading the files gives the same layout within these digits, except for `name` and `links`, which are not stored
- With `compact` the cells that the reader repeats from the previous row of the node (held values) or sets to the default (fixed values, which are written on the first row only) are left empty; an extra sector (sector above 0 of a site that has a sector 0) has all its cells in its first row, because an empty cell would be the value of sector 0
- Antennas are written as their CSV lines, with each `qdant_file` relative to the folder of its CSV file. If no antenna line carries the frequencies of the layout, they are declared on the first node: `center_freq` on its first row and one band row per further frequency
- All files are composed first and written to temporary files next to the targets; each target is replaced by a rename only after all files were composed and written, so an error in the layout, the composition or the writing leaves all files unchanged (a failing rename of the second file is not undone for the first)
- Antenna bands that are evaluated arrays without a line (set with the array form of [set_antenna](#set_antenna)) are written to QDANT files next to the first written CSV file that uses the antenna, named `<stem>_<k>.qdant` (`<stem>` is the name of the CSV file without `.csv`, k = 1, 2, ... per CSV file, skipping names that `qdant` lines of the layout use), one file per antenna with one entry per exported band, and become `qdant` lines. Afterwards, the files `<stem>_<k>.qdant` next to a written CSV file that the new files do not reference are deleted; files of other names are never touched. These arrays must be evaluated: a layout whose `antenna` does not hold them throws
- A band set with [set_antenna_qdant](#set_antenna_qdant) is a `qdant` line: it is written as a reference to its file, relative to the CSV file, and not exported
- The `seed`, `layout_seed` and `wrap` of the layout are stored in the Tx file; a Tx file for a layout without Tx nodes that has them throws. Not expressible in the format and therefore rejected: a scenario that returns to none after a scenario was set, a terminal without a pair in `pairing` (an empty cell means all sites). An extra sector without antenna next to a sector 0 with one is written with the antenna `none`

### Declaration:
```cpp
void quadriga_lib::layout_csv_write(
    const std::string &fn_tx,
    const std::string &fn_rx,
    const quadriga_lib::layout &layout,
    bool compact = true);
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = not written; different files (also if the paths are spelled differently)
- **`layout`** — Layout; must be valid ([layout](#layout))
- **`compact`** *(optional)* — `true`: empty cells where the format allows; `false`: every cell is written

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout)
- [layout](#layout) (the class that is written)

---
<a id="sample_track"></a>
## sample_track
Positions and orientations of a node at snapshot times

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#sample_track) · [Python](../python/network-layouts-and-mobility.md#sample_track)

- A **snapshot** is one sample of the channel in time and space; a **row** is a point stored in the layout. Without `times`, each row of the node is one snapshot. With `times`, a point drop gives as many snapshots as times, and a track is interpolated between its rows
- Point drop (one row): the position is `p0 + v d t` with the speed `v` of the row and the unit vector `d` of its direction (azimuth, elevation; an elevation of NaN is 0). Without direction (NaN azimuth), `d = [cos(tilt) cos(heading), cos(tilt) sin(heading), -sin(tilt)]` from the orientation; bank has no effect. The orientation is constant
- Track (several rows): the time of each row is `time`, else it follows from the arc length and the speed of the row, `t(0) = 0`, `t(i+1) = t(i) + |Δp| / speed(i)`; a step of zero length with speed 0 takes no time. The position is interpolated linearly in time, bank and heading of the orientation along the shorter way and the tilt linearly; before the first and after the last time, the first and the last row are held. If several rows have the same time, the later row holds at that time. The times in `times` need not increase. A track with neither `time` nor a speed throws if `times` is given; a moving step with speed 0 throws
- Mounted antennas (`relative = 1`): the orientation is relative to the direction of motion, and the output is `R(motion) R(orientation)` in the GCS, with bank 0 in `R(motion)`. For a point drop, the motion is its direction, which must be given. For a track, it is the direction of the step of the row segment that holds the snapshot (the step from a row to the next one, the last step repeated at the end); a step of zero length takes the previous direction, at the start the next one that exists. The stored orientation is interpolated first, then rotated by the motion of the segment
- Members of `nodes` with sizes that do not fit, and a `direction` on a track, throw `std::invalid_argument`; the other rules of [layout](#layout) are not checked

### Declaration:
```cpp
void quadriga_lib::sample_track(
    const quadriga_lib::layout_nodes &nodes,
    arma::uword i_node,
    const arma::vec &times,
    arma::mat &pos,
    arma::mat &orientation,
    arma::uvec *i_row = nullptr);
```

### Inputs:
- **`nodes`** — Nodes of one side of a layout ([layout](#layout))
- **`i_node`** — Node index, 0-based
- **`times`** — Snapshot times in [s], finite; `[n_snap]`; empty = one snapshot per row

### Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation; `[3, n_snap]`
- **`i_row`** *(optional)* — Row at or before each snapshot, 0-based, index into the arrays of `nodes` (not into the rows of the node); `[n_snap]`

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
arma::mat pos, ori;
quadriga_lib::sample_track(L.rx, 0, arma::linspace(0.0, 1.0, 11), pos, ori);
```

### See also:
- [generate_track](#generate_track) (creates tracks)
- [set_orientation](#set_orientation) (orientation from the motion)
- [layout](#layout) (the members of a node)

---
<a id="set_antenna"></a>
## set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna or from an evaluated array

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna) · [Python](../python/network-layouts-and-mobility.md#set_antenna)

- The first form stores the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (see there for their meaning) as an antenna line of the layout, which [layout_csv_write](#layout_csv_write) writes as CSV cells; the second form stores an array, e.g. a measurement read with [qdant_read](antennas-transducers-and-patterns.md#qdant_read), which the writer exports to a QDANT file
- Without `center_freq`, the antenna applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout.center_freq` is added, the other bands of the nodes keep their antenna, and nodes that shared the old entry keep it; until all bands of a node are set, `layout.is_valid()` reports it
- A node set band by band keeps one line per band, also when the lines are equal, so a band added later is reported as not set for it; in a layout with one band, the line of a node set for that band also covers bands added later
- The antenna lists are encoded again after the call as [layout_csv_read](#layout_csv_read) composes them: equal lines share one entry, entries are numbered in order of first use (Tx nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- An array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- A failed call leaves the layout unchanged

### Declarations:
```cpp
void quadriga_lib::set_antenna(
    quadriga_lib::layout &layout,
    const std::string &type,
    const std::unordered_map<std::string, arma::mat> &args = {},
    bool rx_side = false,
    const uint_list &i_node = {},
    double center_freq = NAN,
    bool evaluate = true);

void quadriga_lib::set_antenna(
    quadriga_lib::layout &layout,
    const arrayant<double> &array,
    bool rx_side = false,
    const uint_list &i_node = {},
    double center_freq = NAN);
```

### Inputs:
- **`layout`** — Layout, updated in place; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), e.g. `3GPP`, `dipole`, or `none` (no antenna, for all bands, without `args`)
- **`args`** *(optional)* — Arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) by name: `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers, vectors or matrices (`steer`, `beams` row by row as there), `separate_beams` and `apply_weights` as 0 or 1; a name that the type does not use throws; `pattern` is not available (an array built with it is set with the second form); `center_freq` here is the same as the argument
- **`array`** — Valid single-frequency array
- **`rx_side`** *(optional)* — `false`: nodes of `layout.tx`; `true`: nodes of `layout.rx`
- **`i_node`** *(optional)* — Indices of the nodes of that side, 0-based; empty: all nodes of the side
- **`center_freq`** *(optional)* — Band in [Hz], positive; NaN: one antenna for all bands
- **`evaluate`** *(optional)* — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of new antennas use the generic antenna until `layout.eval_antennas()` is called

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_antenna(L, "3GPP", {{"panel", {4, 4, 1, 2}}, {"pol", {3}}}, false, {}, 3.5e9);
quadriga_lib::set_antenna(L, quadriga_lib::qdant_read<double>("handset.qdant")[0], true);
```

### See also:
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the meaning of the arguments)
- [layout](#layout) (the antenna lists and the member function `eval_antennas`)
- [layout_csv_write](#layout_csv_write) (writes the lines and exports the arrays)
- [set_antenna_qdant](#set_antenna_qdant) (sets an entry of a QDANT file)

---
<a id="set_antenna_qdant"></a>
## set_antenna_qdant
Set the antenna of layout nodes to an entry of a QDANT file

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna_qdant) · [Python](../python/network-layouts-and-mobility.md#set_antenna_qdant)

- Stores a `qdant` antenna line with the file as normalised absolute path and the entry `qdant_id`, as [layout_csv_read](#layout_csv_read) composes it from the columns `antenna`, `qdant_file` and `qdant_id`; [layout_csv_write](#layout_csv_write) writes this line with the path relative to the CSV file and exports no copy of the array
- The array is read as stored, for all bands; bands, the sharing of entries and the renumbering of `i_antenna` follow [set_antenna](#set_antenna)
- A missing or unreadable file throws `std::runtime_error`; without evaluation, an entry that the file does not have throws when the antennas are evaluated; a failed call leaves the layout unchanged

### Declaration:
```cpp
void quadriga_lib::set_antenna_qdant(
    quadriga_lib::layout &layout,
    const std::string &qdant_file,
    arma::uword qdant_id = 1,
    bool rx_side = false,
    const uint_list &i_node = {},
    double center_freq = NAN,
    bool evaluate = true);
```

### Inputs:
- **`layout`** — Layout, updated in place; its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — Existing QDANT file, absolute or relative to the current folder
- **`qdant_id`** *(optional)* — 1-based entry of the file
- **`rx_side`** *(optional)* — `false`: nodes of `layout.tx`; `true`: nodes of `layout.rx`
- **`i_node`** *(optional)* — Indices of the nodes of that side, 0-based; empty: all nodes of the side
- **`center_freq`** *(optional)* — Band in [Hz], positive; NaN: one antenna for all bands
- **`evaluate`** *(optional)* — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_antenna_qdant(L, "measured/handset.qdant", 2, true);
```

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna or from an array)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads the entries of a QDANT file)

---
<a id="set_orientation"></a>
## set_orientation
Set the orientation of nodes from their motion, or to a given value

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_orientation) · [Python](../python/network-layouts-and-mobility.md#set_orientation)

- The orientation is that of the antenna: bank, tilt, heading = (γ, β, α) of 3GPP TR 38.901; a positive tilt points down (aviation pitch is -tilt), the heading is counted counter-clockwise from the x-axis
- `force = false`, track (several rows): bank 0, tilt and heading from the steps between the rows (forward difference, the last step repeated, or for a closed track the first row, as `qd_track.calc_orientation` of the NG library does); `orientation` is added as an offset and the angles are wrapped to [-π, π]. A step of zero length keeps the previous orientation, at the start the next one that exists; a vertical step has the tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising track has a negative tilt. A track that does not move keeps its orientation (plus the offset)
- `force = false`, point drop (one row): the orientation stays and receives the offset; without an offset it is not touched. Random orientations of point drops are not generated here
- `force = false` leaves nodes with `relative = 1` unchanged, offset included: their orientation is relative to the motion, which [sample_track](#sample_track) applies at each snapshot
- `force = true`: every row of the nodes gets exactly `orientation`, also the nodes with `relative = 1`
- A failed call leaves `nodes` unchanged

### Declaration:
```cpp
void quadriga_lib::set_orientation(
    quadriga_lib::layout_nodes &nodes,
    const arma::vec &orientation = {},
    const uint_list &i_node = {},
    bool force = false);
```

### Inputs:
- **`nodes`** — Nodes, updated in place
- **`orientation`** *(optional)* — Bank, tilt, heading in [rad], 3 finite values; empty = 0
- **`i_node`** *(optional)* — Node indices, 0-based; empty = all nodes
- **`force`** *(optional)* — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`: `orientation` on all rows

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_orientation(L.rx);                                  // The antennas look along the motion
quadriga_lib::set_orientation(L.rx, arma::vec{0.0, 0.0, 0.5});        // The same, turned by 0.5 rad
```

### See also:
- [generate_track](#generate_track) (creates tracks with this orientation)
- [sample_track](#sample_track) (applies the motion to mounted antennas)
