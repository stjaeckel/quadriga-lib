# Network layouts and mobility

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-network-layouts-and-mobility

| Function | Summary |
| --- | --- |
| [layout](#layout) | Class for a network layout: Tx and Rx nodes, antennas, frequencies and link states |
| [layout_links](#layout_links) | State of the propagation links of one realisation of a layout |
| [layout_nodes](#layout_nodes) | The Tx nodes or the Rx nodes of a layout, as a table of flat arrays |
| [attach_terminals](#attach_terminals) | Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP |
| [drop_terminals](#drop_terminals) | Drop UTs into the cells of the Tx nodes of a layout |
| [export_rays_qrt](#export_rays_qrt) | Write the rays of a layout with their link gain to a QRT file |
| [generate_layout](#generate_layout) | Network layout from a preset: BS sites and a drop of UTs |
| [generate_rays](#generate_rays) | Rays of the propagation links of a layout (stage 2 of the channel pipeline) |
| [generate_track](#generate_track) | Replace the rows of nodes by a generated track or a point drop, or append a node with one |
| [get_channels_from_rays](#get_channels_from_rays) | Channel coefficients of a layout from given rays (stage 3 of the channel pipeline) |
| [get_channels_layout](#get_channels_layout) | Channel coefficients of a layout in one call |
| [init_link_states](#init_link_states) | Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline) |
| [layout_csv_read](#layout_csv_read) | Read the Tx and Rx CSV files of a layout |
| [layout_csv_write](#layout_csv_write) | Write a layout to the Tx and Rx CSV files |
| [sample_track](#sample_track) | Positions and orientations of a node at snapshot times |
| [set_antenna](#set_antenna) | Set the antenna of layout nodes from the arguments of generate_antenna, a QDANT file, an array, or by index |
| [set_orientation](#set_orientation) | Set the orientation of nodes from their motion, or to a given value |

A **layout** describes a radio network for channel simulation: the transmitters (Tx) and receivers (Rx), where they are
and how they move, which antenna each one has and which scenario applies. The functions of this section create layouts,
store them in files, move the nodes along tracks and compute the channels of all Tx-Rx pairs.

**What a layout holds** ([layout](#layout)):

- The **Tx nodes** in `layout.tx` (base stations, access points) and the **Rx nodes** in `layout.rx` (user terminals,
  stations), each a [layout_nodes](#layout_nodes) table. A node has a name, a sector number, an antenna and one or more rows:
  positions with orientation, scenario and indoor state
- The **antennas** of the nodes and the simulation frequencies `layout.center_freq`
- The **link states** `layout.links` of one realisation: LOS and indoor state, scenario and seeds of each Tx-Rx pair
  ([layout_links](#layout_links))

**Typical workflow:**

1. Create a layout: [generate_layout](#generate_layout) from a preset of 3GPP TR 38.901, ITU-R M.2412 or 3GPP TR 37.885, or
   [layout_csv_read](#layout_csv_read) from two CSV files; [drop_terminals](#drop_terminals) drops new UTs into the cells of the Tx nodes
2. Set the frequencies `layout.center_freq` (or give them to [generate_layout](#generate_layout)) and the antennas ([set_antenna](#set_antenna))
3. Optionally, move the nodes: [generate_track](#generate_track) creates tracks, [set_orientation](#set_orientation) turns the antennas along the
   motion, [sample_track](#sample_track) gives the positions at given times
4. Compute the channels with [get_channels_layout](#get_channels_layout)
5. Store the layout with [layout_csv_write](#layout_csv_write)

For more control, step 4 runs in three stages: [init_link_states](#init_link_states) draws the link states, [generate_rays](#generate_rays) the rays of
the geometry-based stochastic model (GBSM), and [get_channels_from_rays](#get_channels_from_rays) applies the antennas, the motion and the link
gain. Between the stages, link states and rays can be inspected or edited, [attach_terminals](#attach_terminals) selects the serving cell
of each terminal, and [export_rays_qrt](#export_rays_qrt) stores the rays in a QRT file.

**Antennas.** Each node has one antenna. [set_antenna](#set_antenna) sets it from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), from an
entry of a QDANT file, from an array or by the index of an antenna that the layout already has; in the CSV files,
antenna columns describe it. A node has one antenna for all bands or one per band. A node without an antenna uses the
generic antenna: two co-located isotropic elements with vertical and horizontal polarization.

**Terms** (full list in the [Glossary](general-usage-notes.md#glossary)):

- **Tx nodes**, **Rx nodes**: the two node tables of a layout, `layout.tx` and `layout.rx`. A function that works on one
  of them takes the table itself or `rx` (`false`: Tx nodes, `true`: Rx nodes)
- **Node**: one transmitter or receiver, identified by its `name` and `sector`, with one antenna and one track
- **Site**: all Tx nodes of one name, e.g. the three sectors of a base station; they share position and scenario
- **Terminal**: all Rx nodes of one name, e.g. the antenna panels of one device; they share their positions
- **Row**: one position and orientation of a node, with the values that apply from there on (speed, scenario, indoor
  state)
- **Point drop**: a node with one row; it stays where it is or, with a speed, moves along its direction
- **Track**: a node with several rows; it moves along its positions
- **Snapshot**: one sample of the channel in time; without snapshot times, each row is one snapshot
- **Segment**: a part of a track with one scenario and indoor state; a new segment starts where one of them changes
- **Propagation link**: one site, one terminal and one segment, with its own LOS and indoor state, large-scale
  parameters and rays
- **Channel link**: one Tx node and one Rx node; it applies their antennas to the rays of its propagation link, which
  the sectors of a site share
- **Band**: one of the simulation frequencies `layout.center_freq`
- **Scenario**: a family (e.g. `UMa`), which draws the LOS and indoor state of each link, or a conf name (e.g.
  `UMa_NLOS`), which fixes them
- **Wrap-around**: copies of the sites, displaced by the lattice vectors `layout.wrap`, so that a cluster of cells tiles
  the plane; each link uses the copy (*image*) nearest to the first position of the terminal, kept for the whole track
- **Orientation**: bank, tilt and heading of the antenna; a positive tilt points down. **Direction**: azimuth and
  elevation of the motion of a point drop

**Conventions.** Angles are in radians in the classes and in degrees in the CSV files. Indices are 0-based, except
`i_antenna`, the antenna of a node, where 0 is the generic antenna and 1 the first antenna of the layout. Random values
follow from the seeds and the positions, not from the order of evaluation, the batches or the number of threads. The
file format is described in [Layout CSV files](../../formats/layout_csv.md).

---
<a id="layout"></a>
## layout
Class for a network layout: Tx and Rx nodes, antennas, frequencies and link states

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#eval_antennas) · [Python](../python/network-layouts-and-mobility.md#eval_antennas)

- Holds the Tx nodes `tx` and the Rx nodes `rx` (each a [layout_nodes](#layout_nodes) table), the antennas they use, the simulation
  frequencies and the link states of one realisation
- Created by [generate_layout](#generate_layout) or [layout_csv_read](#layout_csv_read), stored by [layout_csv_write](#layout_csv_write); the section introduction
  describes the model

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::string name` | string | Name of the layout; not stored in the CSV files |
| `layout_nodes tx`, `layout_nodes rx` | | Tx nodes (BS, AP) and Rx nodes (UT, STA) |
| `layout_links links` | | Link states of the current realisation, see [layout_links](#layout_links) |
| `arma::sword seed` | scalar | Seed of the layout, -1 = not set |
| `arma::vec center_freq` | `[n_freq]` | Simulation frequencies (bands) in [Hz], strictly ascending |
| `std::vector<std::string> antenna_cols` | `[n_col]` | Names of the antenna columns of the lines in `antenna_args` |
| `std::vector<std::vector<std::string>> antenna_args` | `[n_antenna][0, 1 or n_freq]` | Antenna descriptions, one line for all bands or one per band (see Notes) |
| `std::vector<std::vector<arrayant<double>>> antenna` | `[n_antenna][0, 1 or n_freq]` | Evaluated arrays, one for all bands or one per band; empty = not evaluated |
| `arma::mat wrap` | `[2, 1]` or `[2, 2]` | Lattice vectors of the wrap-around in [m], one per column (see Notes); empty = none |
| `arma::umat pairing` | `[2, n_pair]` | Site index and terminal index of the linked pairs, 0-based; empty = all pairs |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_antenna()` | Number of antennas: length of `antenna_args`, or of `antenna` if there are no descriptions |
| `.eval_antennas(center_freq)` | Generates the arrays of `antenna` that are not evaluated yet (see Notes); a frequency list in [Hz] (optional) replaces `center_freq` and regenerates all |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message (see Notes) |

### Notes:
- **Antenna descriptions**: a line of `antenna_args` holds the cells of the antenna columns `antenna_cols` as in the
  [Layout CSV files](../../formats/layout_csv.md#layout-csv-antennas). Nodes with equal lines share one antenna; `i_antenna` of a node
  is its 1-based index, 0 the generic antenna. No line or an empty line is an array without description (array form of
  [set_antenna](#set_antenna)); a line without antenna type is a band that is not set yet
- **Evaluation** (`eval_antennas`): a line for all bands is generated once per band at its frequency, so that an array
  spaced in wavelengths keeps its spacing (an array of co-located elements is stored once for all bands); a line of a
  band at the frequency of that band; a `qdant` line is read as stored. Until an antenna is evaluated (its entry of
  `antenna` holds no arrays, or an array without elements), its nodes use the generic antenna. Throws for a band without
  description and for an array spaced in wavelengths in a layout without frequencies
- **Wrap-around**: a cluster of a few rings of sites is finite, so the cells at its edge lack the interferers that
  surround them in a large network. With `wrap`, the cluster is repeated periodically by the lattice vectors, and each
  link uses the copy (image) of its site nearest to the first position of the terminal, kept for the whole track, which
  the channel generation treats as a site of its own (TR 38.901 Table 7.8-1: geographical distance based wrapping;
  clause 7.5, step 1). Every terminal then has neighbours on all sides, as in the middle of a network. One vector
  repeats the cluster along one direction, e.g. along a road
- **Wrap vectors**: two vectors form a reduced basis,
  $0 \le a_1 \cdot a_2 \le \min(\lvert a_1 \rvert^2, \lvert a_2 \rvert^2) / 2$, so that $\pm a_1$, $\pm a_2$ and
  $\pm(a_1 - a_2)$ are the nearest images
- **Validity** (`is_valid()`): checks both node tables and `links`, the values that the nodes of one site or terminal
  share ([Layout CSV files](../../formats/layout_csv.md#layout-csv-rows)), the antenna lists (a node with a band that is not set is
  reported), the frequencies, `wrap` and `pairing`

### See also:
- [layout_nodes](#layout_nodes) (the Tx nodes and the Rx nodes)
- [layout_links](#layout_links) (link states)
- [layout_csv_read](#layout_csv_read), [layout_csv_write](#layout_csv_write) (load and store a layout)
- [set_antenna](#set_antenna) (sets the antennas of nodes)

---
<a id="layout_links"></a>
## layout_links
State of the propagation links of one realisation of a layout

Not in: MATLAB, Python

- Written by [init_link_states](#init_link_states), read by the later stages; one entry per propagation link (one site, one terminal, one
  segment)
- Holds the discrete state of the links (LOS and indoor state, scenario, seeds), from which the later stages regenerate
  the large-scale parameters, phases and rays; not stored in the CSV files

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `arma::u32_vec i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of each link, 0-based |
| `arma::u32_vec image` | `[n_link]` | Wrap image of the site, 0 = the real site |
| `arma::u32_vec i_conf` | `[n_link]` | Index into `conf_names`, from scenario, LOS and indoor state |
| `std::vector<std::string> conf_names` | `[n_conf]` | Resolved conf names |
| `arma::u32_vec los` | `[n_link]` | 1 = line of sight |
| `arma::s32_vec indoor` | `[n_link]` | Indoor state at the start of the segment, as in [layout_nodes](#layout_nodes) |
| `arma::u32_vec floor_id` | `[n_link]` | Floor number - 1, 0 outdoors, of the drawn state |
| `arma::vec d2d_in` | `[n_link]` | Indoor 2D distance in [m] of the drawn state |
| `arma::mat o2i_rnd` | `[n_terminal, 4]` | Random inputs of [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) |
| `arma::u32_vec o2i_model` | `[n_terminal]` | Building O2I model, index into the list of the family; for every terminal of a family with O2I, also outdoors |
| `arma::s64_vec gbsm_seed` | `[n_link]` | Seed of the random phases; equal for one site, image and conf |
| `arma::sword seed` | scalar | Master seed, the base of the random values of all stages; -1 = not set |
| `std::string method`, `sos_preset` | | Method and SOS preset as used; an empty `method` means not filled |
| `double anchor_freq` | scalar | Anchor frequency of the multi-frequency model in [Hz], default 2e9 |
| `bool pol_variability` | scalar | Polarization power variability, default false |
| `std::vector<std::string> conf_files` | `[n_conf]` | Resolved paths of the conf files; empty = not resolved |
| `unsigned long long layout_hash` | scalar | Hash of the layout members that determine the links (see Notes) |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_link()` | Number of links |
| `.empty()` | `true` if there are no links |
| `.reset()` | Empties all members and restores the defaults |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, the indices into `conf_names` and the ranges |

### Notes:
- **Layout hash**: covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter,
  scenario overrides, wrap vectors and whether a site moves. The later stages throw when it no longer matches the
  layout; `layout.links.reset()` discards the links. Orientations, antennas and `pairing` may change without new links
- **Forced states**: editing `los` or `indoor` before [generate_rays](#generate_rays) forces that state; the conf and its seed follow
  from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values of the forced state
  from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after
  another realisation

### See also:
- [layout](#layout) (holds the links as `links`)
- [init_link_states](#init_link_states) (draws the links)
- [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) (uses `o2i_rnd`)

---
<a id="layout_nodes"></a>
## layout_nodes
The Tx nodes or the Rx nodes of a layout, as a table of flat arrays

Not in: MATLAB, Python

- `layout.tx` holds the Tx nodes, `layout.rx` the Rx nodes. Per-node members have one entry per node, per-row members
  one per row; the members are the columns of the [Layout CSV files](../../formats/layout_csv.md#layout-csv-columns), with angles in
  rad
- The rows of node `i` are `offset(i)` to `offset(i+1) - 1` (the last node to the end), so each node has at least one
  row
- Optional members may be empty, with the meaning given below

### Attributes:
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::vector<std::string> name` | `[n_node]` | Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading or trailing blank |
| `arma::u32_vec sector` | `[n_node]` | Sector number; (`name`, `sector`) is unique |
| `arma::u32_vec i_antenna` | `[n_node]` | Antenna: 1-based index into the antenna lists of the layout, 0 = generic antenna |
| `arma::uvec offset` | `[n_node]` | First row of each node, 0-based |
| `arma::mat pos` | `[3, n_row]` | Positions in [m] |
| `arma::mat orientation` | `[3, n_row]` | Bank, tilt, heading in [rad] |
| `arma::mat direction` | `[2, n_row]` | Azimuth, elevation of travel in [rad]; NaN azimuth = along the orientation; NaN on all rows of a track; empty = all NaN |
| `arma::vec speed` | `[n_row]` | Speed in [m/s] from this row on; empty = 0 |
| `arma::vec time` | `[n_row]` | Time in [s]; NaN on all rows of a node = not given; empty = none |
| `std::vector<std::string> scenario` | `[n_row]` | Scenario family or conf name, "" = none; empty = none |
| `arma::s32_vec indoor` | `[n_row]` | -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1 |
| `arma::s32_vec floor` | `[n_row]` | Floor number from 1, -1 = from z; empty = all -1 |
| `arma::vec d2d_in` | `[n_row]` | Indoor 2D distance in [m], NaN = draw; empty = all NaN |
| `arma::mat clutter` | `[3, n_node]` | `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN |
| `std::unordered_map<std::string, arma::vec> scenpar_override` | each `[n_node]` | Overrides of scenario parameters by key, NaN = none |
| `arma::s64_vec seed` | `[n_node]` | Seed of the site, -1 = derived from the seed of the layout; empty = all -1 |
| `arma::u32_vec relative` | `[n_node]` | 1 = orientation relative to the direction of motion; empty = all 0 |
| `arma::u32_vec cell` | `[n_node]` | Tx nodes only: 0 = the site has no drop cells of its own ([drop_terminals](#drop_terminals)); empty = all 1 |
| `std::vector<std::string> extra_names` | `[n_extra]` | Names of unknown CSV columns |
| `std::vector<std::vector<std::string>> extra` | `[n_extra][n_row]` | Their cells as text |

### Member functions:
| Function | Description |
| -------- | ----------- |
| `.n_node()` | Number of nodes |
| `.n_row()` | Number of rows of all nodes |
| `.n_row(i_node)` | Number of rows of one node (0-based, throws `std::invalid_argument` for an index outside the nodes) |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, `offset`, finite positions and orientations, names, ranges and the rules for `time`, `speed`, `direction`, `floor` and `d2d_in` of the [Layout CSV files](../../formats/layout_csv.md#layout-csv-columns) |

### See also:
- [layout](#layout) (the class that holds the Tx nodes and the Rx nodes)
- [layout_csv_read](#layout_csv_read) (fills the members from a CSV file)

---
<a id="attach_terminals"></a>
## attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#attach_terminals) · [Python](../python/network-layouts-and-mobility.md#attach_terminals)

- Evaluates every Tx node for each terminal and returns the one with the highest coupling gain (`"large_scale"`) or the
  highest RSRP (`"rsrp"`), with its site and the gains of all nodes
- Use it to link each terminal to its serving site in `layout.pairing` (see Example), so that the channel functions
  compute only the serving links
- One call attaches at one frequency, `layout.center_freq[i_freq]`, the first band by default; this suits offloading
  scenarios, where the higher bands serve as data pipes. The result is one serving Tx node per terminal

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

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1, {3.5e9});
quadriga_lib::init_link_states(L);
arma::uvec site;
quadriga_lib::attach_terminals(L, "large_scale", {}, &site);
L.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, site.n_elem - 1));
```

### Notes:
- **Evaluation**: at the first rows of both nodes and `layout.center_freq[i_freq]`, with the link states of the first
  segment, the wrap image and the antennas as in [get_channels_from_rays](#get_channels_from_rays); `layout.pairing` is ignored
- **`"large_scale"`** (TR 38.901 Table 7.8-1): the link gain of [get_channels_from_rays](#get_channels_from_rays) plus the gain of one
  unit-gain LOS path through both antennas (port weights, element positions, orientations), as [get_channels_planar](channel-generators.md#get_channels_planar)
  gives it; no rays
- **`"rsrp"`** (TR 36.873 eq. (8.1-1)): per Rx port the sum of $\lvert h \rvert^2$ over the rays of
  [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (`output = "rays"`, mobility `"phase"`), link gain included, transmit power 1
- **Ports**: both rules take $\lvert h \rvert^2$ of Tx port `tx_port` and average it over the Rx ports, so a
  dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT
  antenna of the TR
- **Choice**: a terminal takes the highest value of its nodes; ties go to the lowest index. A Tx node without a
  propagation link (co-located) has -Inf; a terminal without any link gets node 0
- **Pairing**: `layout.pairing` holds pairs of site and terminal, so the Example links all sectors of the serving site;
  for multi-connectivity, `layout.pairing` may list several sites per terminal, e.g. all sites within a margin of the
  best value in `gain_dB`, and the channel functions then compute the links to all of them

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays) (link states and rays)
- [get_channels_from_rays](#get_channels_from_rays) (channels of the linked pairs)

---
<a id="drop_terminals"></a>
## drop_terminals
Drop UTs into the cells of the Tx nodes of a layout

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#drop_terminals) · [Python](../python/network-layouts-and-mobility.md#drop_terminals)

- Replaces the Rx nodes by new UTs `UT1`, `UT2`, ..., dropped uniformly into the cells of the sites with the rules of
  the scenario of each site (minimum distance, height, speed, indoor state), as [generate_layout](#generate_layout) drops them
- Use it to drop UTs for any set of Tx nodes, e.g. sites read from a CSV file or edited after [generate_layout](#generate_layout)
- The Tx nodes, the antenna lists and `layout.seed` stay; `pairing` and `links` are emptied. The new UTs get the antenna
  that all old Rx nodes shared (`i_antenna`, with its bands), else the generic antenna

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
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs
- **`bounds`** *(optional)* — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required
  unless `layout.wrap` has two vectors
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** *(optional)* — As in [generate_layout](#generate_layout); an entry or option
  that no family of the sites uses throws
- **`seed`** *(optional)* — Seed >= 0; -1 = `layout.seed` if it is set, else the system random device

### Notes:
- **Cells**: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the
  wrap images when `layout.wrap` is set. A site with several sectors is split into wedges by the nearest sector heading
  (±60° around each boresight for three sectors at equal spacing)
- **Drop area**: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the
  images 0, $\pm a_1$, $\pm a_2$ and $\pm(a_1 - a_2)$; else `bounds`, which is then required. Given bounds limit the
  drop in both cases
- **Sites without a scenario family** (no scenario, or a conf name): height 1.5 m, no minimum distance unless `ut` entry
  1 gives one, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- **Sites with `cell` = 0**, e.g. the micro sites of `M.2412_DenseUrban` config 3, form no cells: with `per_cell`, each
  of their Tx nodes adds `n_ut` UTs to the cell that holds its position, and the UTs keep the minimum distance of the
  site's own family (`ut` entry 1 if given) from them. A site closer than twice the minimum distance to another one can
  leave a cell too little room; such a cell throws
- **Order and reproducibility**: with `per_cell` the UTs are ordered by cell. The drop is a function of the seed and
  does not depend on the thread count; each UT draws its heading, direction and floor from its own random stream, keyed
  by its name
- A failed call leaves the layout unchanged

### See also:
- [generate_layout](#generate_layout) (layouts from presets, with this drop)
- [generate_track](#generate_track) (tracks for the UTs)

---
<a id="export_rays_qrt"></a>
## export_rays_qrt
Write the rays of a layout with their link gain to a QRT file

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#export_rays_qrt) · [Python](../python/network-layouts-and-mobility.md#export_rays_qrt)

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [qrt_file_read](channel-data-and-processing.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) and
  [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local; add the returned shift to the positions read back

### Declaration:
```cpp
arma::vec quadriga_lib::export_rays_qrt(
    const std::string &fn,
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays = {});
```

### Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout with links from [init_link_states](#init_link_states): one frequency for the TR methods (`"quadriga"`: 1 to 127),
  Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at a
  site position
- **`rays`** *(optional)* — Rays of all links from [generate_rays](#generate_rays), possibly edited, one object with all frequencies
  per link; omitted = generated for batches of sites with up to 256 links, which bounds the memory; `[n_link][1]`

### Returns:
- Shift of the local coordinates in x and y in [m]; `[2]`

### Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [get_channels_from_rays](#get_channels_from_rays) at the row, [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `"drift"`,
  [qrt_paths_from_scatterers](channel-data-and-processing.md#qrt_paths_from_scatterers). A CIR read back gives the channel of [get_channels_from_rays](#get_channels_from_rays) (`output = "rays"`,
  `mobility = "drift"`) at the row within float32 accuracy
- **Reading back**: [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) with the method of the links and the `n_interact` of the file as
  `path_type`, then [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) with link gain 0 and mobility `"drift"`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the format**: no tap index and no ground permittivity, so a CIR gives rays, not the taps of the TR, and
  moving a ground-reflection ray from a file throws
- The file is written to `fn + ".part"` and renamed at the end; a failed call leaves an existing file unchanged

### See also:
- [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (the rays and the link gain)
- [qrt_file_read](channel-data-and-processing.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths), [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (evaluation of the file)
- [qrt_paths_from_scatterers](channel-data-and-processing.md#qrt_paths_from_scatterers) (the stored paths)

---
<a id="generate_layout"></a>
## generate_layout
Network layout from a preset: BS sites and a drop of UTs

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_layout) · [Python](../python/network-layouts-and-mobility.md#generate_layout)

- Creates a standard evaluation layout with the values of its tables: the hexagonal and indoor layouts of 3GPP TR
  38.901, the test environments of ITU-R M.2412 and the road layouts of 3GPP TR 37.885
- The result has the sites `BS1`, `BS2`, ... with one Tx node per sector and the UTs `UT1`, `UT2`, ... as point drops;
  set the antennas next ([set_antenna](#set_antenna)), until then the nodes use the generic antenna
- The other inputs change single values of the preset: an entry of a tuple that is NaN or omitted keeps the value of the
  preset; an entry or option that the preset does not use throws

### Declaration:
```cpp
quadriga_lib::layout quadriga_lib::generate_layout(
    const std::string &type,
    arma::uword n_ut = 0,
    const arma::vec &center_freq = {},
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
- **`type`** — Preset name, not case sensitive:
  | `type` | Layout |
  | ------ | ------ |
  | `38.901_UMi`, `38.901_UMa`, `38.901_RMa`, `38.901_SMa` | Hexagonal, 19 sites of 3 sectors, ISD 200, 500, 1732, 1299 m, BS height 10, 25, 35, 35 m |
  | `38.901_InH_Open`, `38.901_InH_Mixed` | Hall of 120 × 50 m, 6 × 2 BSs at 20 m spacing, BS height 3 m |
  | `38.901_InF_SL`, `38.901_InF_DH` | Hall of 120 × 60 × 10 m, 6 × 3 BSs at 20 m spacing, BS height 1.5 m (SL), 8 m (DH) |
  | `38.901_InF_DL`, `38.901_InF_SH`, `38.901_InF_HH` | Hall of 300 × 150 × 10 m, 6 × 3 BSs at 50 m spacing, BS height 1.5 m (DL), 8 m (SH, HH) |
  | `M.2412_InH`, `M.2412_DenseUrban`, `M.2412_Rural`, `M.2412_UrbanMacro` | Test environments of Report ITU-R M.2412-0 (see Notes) |
  | `37.885_Urban`, `37.885_Highway` | Roads and vehicles of TR 37.885, no BSs (see Notes) |
- **`n_ut`** *(optional)* — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs
- **`center_freq`** *(optional)* — Simulation frequencies (bands) in [Hz], finite, positive and strictly ascending,
  stored as `layout.center_freq`; empty: not set
- **`spacing`** *(optional)* — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used), the spacing of
  the indoor grid or of the urban road grid
- **`grid`** *(optional)* — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site;
  `n_y` not used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with the boresights 30° (indoor
  0°) + k · 360° / `n_sector`
- **`area`** *(optional)* — `[length, width, height]` of the hall or road area along x, y and z in [m], > 0; indoor and
  road presets, the height InF only, the width not for the highway; the BS grid must fit into it
- **`bs`** *(optional)* — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt
  (positive down) and an offset added to the boresights
- **`ut`** *(optional)* — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg],
  [deg], [deg]; a value replaces the rule of the preset for all UTs (building UTs get the floor from a given height; the
  speed of the lanes also sets the gaps); min distance (not for roads) and speed >= 0
- **`indoor`** *(optional)* — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1]
  (presets with O2I states); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa, Dense Urban); floor height in [m], >
  0 (UMi, UMa, SMa, Dense Urban)
- **`options`** *(optional)* — One value per key: `p_residential` (SMa), the share of residential buildings in [0, 1];
  `config` (M.2412), `service` (`M.2412_UrbanMacro`), `h_micro` (`M.2412_DenseUrban` config 3) in [m]; `lane_width` in
  [m], `n_lane` (1 to 100), `p_truck` in [0, 1] (TR 37.885); other keys throw
- **`per_cell`** *(optional)* — true: `n_ut` UTs in each cell; false: `n_ut` UTs uniform over all cells; no effect for
  the TR 37.885 presets
- **`seed`** *(optional)* — Seed >= 0; -1 = system random device

### Returns:
- The layout, see [layout](#layout)

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 10, {3.5e9}, {}, {}, {}, {}, {NAN, NAN, NAN, 0.0, 90.0}, {}, {}, true, 42);
quadriga_lib::set_antenna(L, "3gpp", {{"panel", {4, 4, 1, 2}}});
```

### Notes:
- **Hexagonal presets** (2 rings, 3 sectors, wrap-around): site $(q, r)$ with
  $\max(\lvert q \rvert, \lvert r \rvert, \lvert q + r \rvert) \le n$ for $n$ rings is at $x = \sqrt{3}\, q d / 2$,
  $y = (q / 2 + r)\, d$ with the ISD $d$, ordered by ring and then by azimuth in (-30°, 330°]. The wrap vectors
  $a_1 = (\sqrt{3}\, n d / 2, \ (3 n / 2 + 1)\, d)$ and $a_2$, $a_1$ turned by 60°, make the cluster tile the plane.
  With the wrap-around, each link uses the copy of its site nearest to the first position of the terminal, kept for the
  whole track, so that the cells at the edge get the interference of a large network (TR 38.901 Table 7.8-1; see the
  Notes of [layout](#layout))
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
  loss) are set with `o2i_low = 0.8` of [init_link_states](#init_link_states), not by the preset
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
  - The M.2412 layouts carry no scenario, so a new drop with [drop_terminals](#drop_terminals) reproduces the distances of the preset
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
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings $3 (n_\mathrm{fl} - 1) + 1.5$ m with
    $n_\mathrm{fl} \sim U(1, N_\mathrm{fl})$, $N_\mathrm{fl} \sim U(4, 8)$; SMa residential buildings 1 or 2 floors,
    commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in
    azimuth, elevation 0. Calibration orientations by `ut`: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C
    tilt 45°, config D tilt 0° and heading 0°
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in
    buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially
    correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or
    10 m (RMa, SMa residential)
- **Reproducibility**: the layout is a function of the seed and does not depend on the thread count; seed -1 draws a
  seed, which is stored in `layout.seed`

### See also:
- [drop_terminals](#drop_terminals) (the same drop for any set of Tx nodes)
- [generate_track](#generate_track) (tracks for the UTs)
- [set_antenna](#set_antenna) (antennas of the nodes)
- [layout](#layout) (the members)

---
<a id="generate_rays"></a>
## generate_rays
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_rays) · [Python](../python/network-layouts-and-mobility.md#generate_rays)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [init_link_states](#init_link_states): large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [get_channels_from_rays](#get_channels_from_rays) applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [attach_terminals](#attach_terminals) and [export_rays_qrt](#export_rays_qrt)
- The rays of a link do not depend on the other requested links

### Declaration:
```cpp
std::vector<std::vector<gbsm_rays>> quadriga_lib::generate_rays(
    const layout &layout,
    const uint_list &i_link = {},
    bool scatterers = false);
```

### Inputs:
- **`layout`** — Layout with links from [init_link_states](#init_link_states); throws if a member that determines the links changed since
  then, or if the sizes of its members do not match
- **`i_link`** *(optional)* — Propagation links, 0-based, without repetition; omitted = all (none for links without
  entries); `[n_out]`
- **`scatterers`** *(optional)* — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`

### Returns:
- Rays per link and frequency, `[n_out][n_freq]` with one frequency each for the TR methods, `[n_out][1]` with all
  frequencies for `"quadriga"`

### Notes:
- **Chain**: [gbsm_init_phases](geometry-based-stochastic-model.md#gbsm_init_phases) and [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the random sets, [path_loss](geometry-based-stochastic-model.md#path_loss) (for `rnd_he` and the SF standard
  deviation), [lsp_generate](geometry-based-stochastic-model.md#lsp_generate), [ssf_clusters](geometry-based-stochastic-model.md#ssf_clusters), [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) and, with `scatterers`, [ssf_scatterers](geometry-based-stochastic-model.md#ssf_scatterers), once for the
  links of one site, image, conf and seed
- **Position**: the first row of the segment of the terminal and the site (moved by its wrap image) at the same time; a
  moving site is sampled at the time of the row, without times on the Rx track at the row of the same index. A site with
  more than one row or a speed uses dual mobility; the other sites are not affected
- **TR methods**: the SOS preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance,
  `angle_model = "additive"`, `pol_model = "38.901"`, one set of clusters and rays per frequency (`"38.901_MF"`:
  clusters at the anchor frequency, rays per frequency)
- **`"quadriga"`**: bias removal over the positions of all links of the group, the legacy outdoor distance,
  `"rotation"`, `"quadriga"`, one ray set for all frequencies, always with scatterers
- **Edited link states**: the conf and the seed of a link follow from its scenario and the `los` and `indoor` of
  `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged.
  Building UTs are on the floor of the Rx nodes (else from z) at the distance of the stored O2I inputs, the others on
  floor 0 at 0 m

### See also:
- [init_link_states](#init_link_states) (stage 1), [get_channels_from_rays](#get_channels_from_rays) (stage 3)
- [gbsm_rays](geometry-based-stochastic-model.md#gbsm_rays) (the ray data)

---
<a id="generate_track"></a>
## generate_track
Replace the rows of nodes by a generated track or a point drop, or append a node with one

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#generate_track) · [Python](../python/network-layouts-and-mobility.md#generate_track)

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [set_orientation](#set_orientation) turns them; [sample_track](#sample_track) gives the positions at
  snapshot times
- An empty `i_node` appends a new node at `start` and gives it the track (see Notes)

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
    arma::sword seed = -1,
    const std::string &name = "",
    const arma::vec &start = {});
```

### Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`), updated in place
- **`i_node`** — Node indices, 0-based; empty: append a new node
- **`type`** — Type of the track, not case sensitive:
  - `point`: a point drop with `speed` and a horizontal `direction`; the rows after the first are removed; `length` is
    not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction
  - `circular`: a closed circle of circumference `length` (default 20π m, radius 10 m); the first and the last point are
    equal
  - `street`: the route of a car through a city grid of straight streets and crossings, of `length` (default 1000 m),
    see Notes
- **`length`** *(optional)* — Length in [m], finite and >= 0; 0 = default of the type
- **`speed`** *(optional)* — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of
  the node if it is valid (> 0 for a track); without one, `point` gets 0 and the other types throw
- **`direction`** *(optional)* — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry
  of `i_node` (nodes of one name need equal values); empty = uniform random in [0, 2π)
- **`spacing`** *(optional)* — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m
  (`street`)
- **`options`** *(optional)* — Options of `street`, one value each: `street_length_min` = 50, `street_length_mu` = 187,
  `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other keys, and these keys for the other types, throw
- **`seed`** *(optional)* — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 throws
- **`name`** *(optional)* — Name of the new node (empty `i_node` only); empty: the next free name
- **`start`** *(optional)* — First position `[x, y, z]` of the new node in [m] (empty `i_node` only); empty: the origin

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("", "rx.csv");
quadriga_lib::generate_track(L.rx, {0}, "street", 500.0, 10.0, {}, NAN, {{"turn_probability", {0.3}}}, 42);
quadriga_lib::generate_track(L.rx, {}, "linear", 100.0, 1.5, {0.0}, NAN, {}, 42, "walker", {10.0, 20.0, 1.5});
```

### Notes:
- **Points**: `linear` has two points, or `ceil(length / spacing) + 1` equidistant points; `circular` 129 points, or
  `max(3, ceil(length / spacing)) + 1`; `street` is sampled with the point spacing `spacing`, and the chords between the
  points are slightly shorter than the route at the corners
- **Direction of a circle**: the angle of the start point seen from the centre of the circle. A positive value runs
  counter-clockwise, a negative one clockwise: 0 starts in the east of the circle and runs north, -π/2 starts in the
  south and runs west. A random direction is drawn in [0, 2π), so random circles run counter-clockwise
- **Street route** (as `qd_track.generate` of the NG library): straight streets of random length (normal distribution,
  drawn again below `street_length_min`) are separated by crossings. At a crossing the car turns with the probability
  `turn_probability` by about 90° (standard deviation 15°) to the left or right, cutting the corner along a path of
  length $2 r \cos(\varphi / 2)$ for the turn angle $\varphi$ and the `curve_radius` $r$. The direction stays within
  105° of the start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [set_orientation](#set_orientation) with `force = false`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- **Append mode**: with an empty `i_node`, a new node is appended with one row at `start` and the default values (sector
  0, the generic antenna, orientation 0, speed 0, no scenario), and gets the track; a track type therefore needs
  `speed`. Its index is `nodes.n_node() - 1` after the call. The default name is the common prefix of the names before
  their number with the next number (`UT3` after `UT1`, `UT2`), else `N<n>`. `name` or `start` with a non-empty `i_node`
  throws
- A failed call leaves `nodes` unchanged

### See also:
- [sample_track](#sample_track) (snapshots of a track)
- [set_orientation](#set_orientation) (orientation from the motion)
- [layout](#layout) (the members of a node)

---
<a id="get_channels_from_rays"></a>
## get_channels_from_rays
Channel coefficients of a layout from given rays (stage 3 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#get_channels_from_rays) · [Python](../python/network-layouts-and-mobility.md#get_channels_from_rays)

- Stage 3 alone, from given rays, e.g. for edited link states or rays; [get_channels_layout](#get_channels_layout) computes the channels
  from the layout in one call. Applies the antennas of the nodes, the motion along their tracks and the link gain to the
  rays of [generate_rays](#generate_rays), and returns one channel per Tx node, Rx node and frequency
- Only the channel links of the given propagation links and of the pairs in `layout.pairing` (empty: all) are computed;
  the sectors of a site share the rays of their propagation link
- Tracks with several segments are joined by [crossfade_segments](channel-data-and-processing.md#crossfade_segments)

### Declaration:
```cpp
std::vector<channel<double>> quadriga_lib::get_channels_from_rays(
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
- **`layout`** — Layout with links from [init_link_states](#init_link_states); throws if a member that determines the links changed since
  then
- **`rays`** — Rays of the links from [generate_rays](#generate_rays), possibly edited, with the frequencies of the layout;
  `[n_out][1 or n_freq]`
- **`i_link`** *(optional)* — The propagation links of `rays`, 0-based; omitted = all links; `[n_out]`
- **`times`** *(optional)* — Snapshot times in [s], finite, the same for all nodes; omitted = one snapshot per row;
  `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`, see [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel)
- **`mobility`** *(optional)* — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods,
  drift for `"quadriga"`)
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays
- **`overlap`** *(optional)* — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1],
  see [crossfade_segments](channel-data-and-processing.md#crossfade_segments)

### Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

### Returns:
- Channels, see [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel); `[n_channel]`

### Notes:
- **Order and names**: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the
  sector where all Tx nodes (or all Rx nodes) have sector 0
- **Snapshots** of both ends from [sample_track](#sample_track) over `times`; a node with one snapshot is repeated to the count of
  the other; two tracks without `times` need equal row counts
- **Link gain** per snapshot, $G = -\mathrm{PL} - L_\mathrm{O2I} + 10 \log_{10} \mathrm{SF}$ in dB: the path loss PL of
  [path_loss](geometry-based-stochastic-model.md#path_loss) with the random environment height `rnd_he` held at the start of the segment, the shadow fading SF and
  the K-factor ratio of [lsp_generate](geometry-based-stochastic-model.md#lsp_generate) from [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the `"lsf"` set at the snapshot, and the O2I loss of
  [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) for building and car UTs with the model of the link state (the deviation of a building UT at the
  snapshot, that of a car UT at the first row of the terminal for its whole track, TR 38.901 eq. 7.4-4). A link whose
  ends do not move is evaluated once
- **Antennas**: [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) with the antenna of each node at the frequency (its band `i_freq` in
  `layout.antenna`, or its only entry for all bands) and the Tx positions moved by the wrap image of the link. Nodes
  without an antenna and antennas that are not evaluated use the generic antenna; an antenna spaced in wavelengths that
  is not evaluated throws
- **Frequencies**: rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency
  `i_freq` (`"quadriga"`)
- **Segments**: the snapshots of a segment are those whose row (see [sample_track](#sample_track)) is in it; each segment also covers
  the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row. The
  selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing
  `times`
- **Relative delays**: in `"phase"` the paths of each segment relative to its own LOS, in `"drift"` relative to the LOS
  at the snapshot

### See also:
- [generate_rays](#generate_rays) (stage 2)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="get_channels_layout"></a>
## get_channels_layout
Channel coefficients of a layout in one call

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#get_channels_layout) · [Python](../python/network-layouts-and-mobility.md#get_channels_layout)

- Computes the channels in one call from the layout: one channel per Tx node, Rx node and frequency, for the pairs in
  `layout.pairing` (empty: all pairs); [get_channels_from_rays](#get_channels_from_rays) is the third stage alone, from given rays
- Supports tracks (with times or speeds), point drops that move with their speed, tracks of several segments, joined by
  the cross-fade of [crossfade_segments](channel-data-and-processing.md#crossfade_segments), and both output forms, `"taps"` and `"rays"`
- Runs the stages [init_link_states](#init_link_states), [generate_rays](#generate_rays) and [get_channels_from_rays](#get_channels_from_rays) with their default settings;
  the results are those of the stages called one by one
- Without link states in `layout.links`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

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
- **`layout`** — Layout with `center_freq`; antennas as evaluated in `layout.antenna`, nodes without one use the generic
  antenna (see [get_channels_from_rays](#get_channels_from_rays))
- **`times`** *(optional)* — Snapshot times in [s]; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` or `"quadriga"`; omitted = the method of filled links, else
  `"38.901"`; another method than that of filled links throws
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
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1, {3.5e9});
auto channels = quadriga_lib::get_channels_layout(L);
```

### Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout.links` holds none when `layout.links.method` is empty, e.g. after `layout.links.reset()`.
  With link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives an empty result
- **Scatterers**: the mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (the three stages)
- [generate_layout](#generate_layout) (layouts of the TR scenarios)

---
<a id="init_link_states"></a>
## init_link_states
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#init_link_states) · [Python](../python/network-layouts-and-mobility.md#init_link_states)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout.links` ([layout_links](#layout_links))
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout.pairing` does not limit the links; it selects at the output of the channel functions

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
- **`layout`** — Layout, valid, with `center_freq`; `layout.links` and `layout.seed` are written, the rest is unchanged
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see
  [generate_rays](#generate_rays)
- **`o2i_low`** *(optional)* — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; omitted:
  the share of TR 38.901 Table 7.8-1 (0.5); the other families have one building model; `[1]`
- **`pol_variability`** *(optional)* — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only
- **`anchor_freq`** *(optional)* — Anchor frequency of `"38.901_MF"` in [Hz]
- **`sos_preset`** *(optional)* — SOS preset of the GBSM random sets, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); omitted: `Exp256` (TR
  methods), `Comb256` (`"quadriga"`)
- **`seed`** *(optional)* — Master seed, >= -1; -1 = `layout.seed`, if that is -1 one draw of the system random device.
  The seed used is written to `layout.seed`
- **`config_dirs`** *(optional)* — Folders of user conf files, searched before the shipped folder

### Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 0 first). Co-located pairs have no link; a later segment that starts at the site throws. A
  layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [conf_file_list](geometry-based-stochastic-model.md#conf_file_list), the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, $+a_1$, $-a_1$, $+a_2$, $-a_2$, $+(a_1 - a_2)$, $-(a_1 - a_2)$; all
  distances of the link use this virtual site position
- **O2I state** per terminal, from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3):
  building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A
  (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the
  O2I model and `d2d_in` as in [o2i_loss](geometry-based-stochastic-model.md#o2i_loss). The values `indoor`, `floor` and `d2d_in` of the Rx nodes fix them; an
  exact conf name that contradicts a fixed `indoor` throws
- **One O2I state per terminal** (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal
  linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a
  site fixes it for the links to all sites; InH and InF have none
- **LOS state** per link: a uniform field per site, image and family at the first position of the segment, at the
  `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [los_probability](geometry-based-stochastic-model.md#los_probability) at
  $d_\mathrm{2D,out} = \max(d_\mathrm{2D} - d_\mathrm{2D,in}, 0)$; InF-HH is always LOS
- **Segments**: on the Rx nodes, a row whose `scenario` or `indoor` differs from the row before it starts a segment (an
  empty `scenario` differs from an explicit one). A segment has its own propagation links (order: terminal, site,
  segment), LOS state, indoor state, conf and seed, drawn at its first row, with the distances to a moving site at the
  time of the segment start. The wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are
  those of the first row of the track; a fixed `d2d_in` must be equal at all segment starts. A segment keeps its LOS
  state along its length, so long tracks need explicit segments
- **Random values**: [sos_rand](sum-of-sinusoids-random-number-generator.md#sos_rand) with the preset `Exp256`; each field has its own seed, hashed from the site seed (else
  the layout seed) and a key of its purpose, site, image and family or conf. `gbsm_seed` is the seed of
  [gbsm_init_phases](geometry-based-stochastic-model.md#gbsm_init_phases)
- **Frequencies**: the TR methods draw the cluster and ray sets of frequency `k` with the index `k` into
  `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands

### See also:
- [layout_links](#layout_links) (the link states)
- [generate_rays](#generate_rays), [get_channels_from_rays](#get_channels_from_rays) (stages 2 and 3)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="layout_csv_read"></a>
## layout_csv_read
Read the Tx and Rx CSV files of a layout

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_read) · [Python](../python/network-layouts-and-mobility.md#layout.csv_read)

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](../../formats/layout_csv.md) and returns the
  layout, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `false`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

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
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = no Tx nodes or no Rx nodes
- **`center_freq`** *(optional)* — Simulation frequencies in [Hz], strictly ascending and positive; empty = from the
  files; `[n_freq]`
- **`eval_antennas`** *(optional)* — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and
  `i_antenna` are kept, and the nodes use the generic antenna until `layout.eval_antennas()` is called

### Outputs:
- **`extra_tx`**, **`extra_rx`** *(optional)* — Names of the unknown columns of the Tx and Rx file, in file order;
  written only if the call succeeds

### Returns:
- `quadriga_lib::layout` — The layout; `name` and `links` are empty

### Notes:
- **Errors**: a violation of the format and a layout that is not valid ([layout](#layout)) throw `std::runtime_error` with the
  file name, the line and the column (see [Reading](../../formats/layout_csv.md#layout-csv-reading)); with `eval_antennas`, so do a
  line that [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a
  layout without frequencies
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](../../formats/layout_csv.md#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas = false`

### See also:
- [layout_csv_write](#layout_csv_write) (stores a layout)
- [layout](#layout) (the returned class)
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the antenna types and arguments of the antenna columns)

---
<a id="layout_csv_write"></a>
## layout_csv_write
Write a layout to the Tx and Rx CSV files

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#layout_csv_write) · [Python](../python/network-layouts-and-mobility.md#layout.csv_write)

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](../../formats/layout_csv.md); reading the files back gives the same layout within the written digits
  (9 significant digits, positions 1 µm, times 1 ns), except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with the array form of [set_antenna](#set_antenna), are written to QDANT
  files next to the CSV file
- A failed call leaves the existing files unchanged

### Declaration:
```cpp
void quadriga_lib::layout_csv_write(
    const std::string &fn_tx,
    const std::string &fn_rx,
    const quadriga_lib::layout &layout,
    bool compact = true);
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = not written; different files (also if the paths are spelled
  differently)
- **`layout`** — Layout; must be valid ([layout](#layout))
- **`compact`** *(optional)* — `true`: cells that the reader fills in are left empty; `false`: every cell is written

### Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](../../formats/layout_csv.md#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them throws. An antenna set from a QDANT file ([set_antenna](#set_antenna)) is written as a
  reference to its file, not exported

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout)
- [layout](#layout) (the class that is written)

---
<a id="sample_track"></a>
## sample_track
Positions and orientations of a node at snapshot times

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#sample_track) · [Python](../python/network-layouts-and-mobility.md#sample_track)

- Returns the position and the orientation of one node at each snapshot time, for point drops and tracks;
  [get_channels_from_rays](#get_channels_from_rays) samples both ends of each link this way
- Without `times`, each row of the node is one snapshot
- The orientation includes the rotation of mounted antennas (`relative = 1`), so it is the orientation in the GCS

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
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`)
- **`i_node`** — Node index, 0-based
- **`times`** — Snapshot times in [s], finite; `[n_snap]`; empty = one snapshot per row

### Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation;
  `[3, n_snap]`
- **`i_row`** *(optional)* — Row at or before each snapshot, 0-based, index into the arrays of `nodes` (not into the
  rows of the node); `[n_snap]`

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
arma::mat pos, ori;
quadriga_lib::sample_track(L.rx, 0, arma::linspace(0.0, 1.0, 11), pos, ori);
```

### Notes:
- **Point drop** (one row): $p(t) = p_0 + v t \hat{d}$ with the speed $v$ of the row and the unit vector $\hat{d}$ of
  its direction (azimuth, elevation; an elevation of NaN is 0). Without a direction (NaN azimuth),
  $\hat{d} = (\cos\beta \cos\alpha, \cos\beta \sin\alpha, -\sin\beta)$ with the tilt $\beta$ and the heading $\alpha$ of
  the orientation; the bank has no effect. The orientation is constant
- **Track** (several rows): the time of row $i$ is `time`, else it follows from the speed $v_i$ of the row, $t_0 = 0$,
  $t_{i+1} = t_i + \lvert p_{i+1} - p_i \rvert / v_i$; a step of zero length with speed 0 takes no time. The position is
  interpolated linearly in time, bank and heading along the shorter way and the tilt linearly; before the first and
  after the last time, the first and the last row hold. If several rows have the same time, the later row holds at that
  time. The times in `times` need not increase. A track with neither `time` nor a speed throws if `times` is given; a
  moving step with speed 0 throws
- **Mounted antennas** (`relative = 1`): the orientation is relative to the direction of motion, and the output is
  $R_\mathrm{motion} R_\mathrm{orientation}$, with bank 0 in $R_\mathrm{motion}$. For a point drop, the motion is its
  direction, which must be given. For a track, it is the direction of the step from the row at or before the snapshot to
  the next row (the last step repeated at the end); a step of zero length takes the previous direction, at the start the
  next one that exists. The stored orientation is interpolated first, then rotated by the motion
- **Checks**: members of `nodes` with sizes that do not fit, and a `direction` on a track, throw
  `std::invalid_argument`; the other rules of [layout](#layout) are not checked

### See also:
- [generate_track](#generate_track) (creates tracks)
- [set_orientation](#set_orientation) (orientation from the motion)
- [layout](#layout) (the members of a node)

---
<a id="set_antenna"></a>
## set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna, a QDANT file, an array, or by index

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_antenna) · [Python](../python/network-layouts-and-mobility.md#set_antenna)

- Sets the antenna of Tx nodes or Rx nodes from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) or from an entry of a QDANT file
  (first form), from an array (second form), e.g. a measured pattern read with [qdant_read](antennas-transducers-and-patterns.md#qdant_read), or to antennas that the
  layout already has, by their index (third form)
- Without `center_freq`, the antenna applies to all bands; with `center_freq`, it is the antenna of that band
- The first form stores an antenna description, which [layout_csv_write](#layout_csv_write) writes as antenna columns (a QDANT file as a
  reference, without a copy); the second form stores the array, which the writer exports to a QDANT file

### Declarations:
```cpp
void quadriga_lib::set_antenna(
    quadriga_lib::layout &layout,
    const std::string &type,
    const std::unordered_map<std::string, arma::mat> &args = {},
    bool rx = false,
    const uint_list &i_node = {},
    double center_freq = NAN,
    bool evaluate = true);

void quadriga_lib::set_antenna(
    quadriga_lib::layout &layout,
    const arrayant<double> &array,
    bool rx = false,
    const uint_list &i_node = {},
    double center_freq = NAN);

void quadriga_lib::set_antenna(
    quadriga_lib::layout &layout,
    const uint_list &i_antenna,
    bool rx = false,
    const uint_list &i_node = {},
    double center_freq = NAN);
```

### Inputs:
- **`layout`** — Layout, updated in place; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), e.g. `3GPP`, `dipole`; a QDANT file, i.e. a name ending in `.qdant`
  (any case), absolute or relative to the current folder; or `none` (no antenna, for all bands, without `args`)
- **`args`** *(optional)* — Arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) by name, with the meaning given there: `res`,
  `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers,
  vectors or matrices in the shapes given there (a `beams` vector is one beam), `separate_beams` and `apply_weights` as
  0 or 1; a name that the type does not use throws; `pattern` is not available (an array built with it is set with the
  second form); `center_freq` here is the same as the argument. For a QDANT file only `id`, the 1-based entry of the
  file (default 1)
- **`array`** — Valid single-frequency array
- **`i_antenna`** — Antenna entries of the layout, 1-based as `i_antenna` of the nodes, 0 = generic antenna; one value
  for all picked nodes or one per node in the order of `i_node`; empty: 0
- **`rx`** *(optional)* — `false`: set the antenna of Tx nodes (`layout.tx`); `true`: of Rx nodes (`layout.rx`)
- **`i_node`** *(optional)* — Indices of the Tx nodes or Rx nodes chosen by `rx`, 0-based; empty: all of them
- **`center_freq`** *(optional)* — Band in [Hz], positive; NaN: one antenna for all bands. Third form: a band of the
  layout, which the nodes take from the given entries
- **`evaluate`** *(optional)* — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none,
  the nodes of new antennas use the generic antenna until `layout.eval_antennas()` is called

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_antenna(L, "3GPP", {{"panel", {4, 4, 1, 2}}, {"pol", {3}}}, false, {}, 3.5e9);
quadriga_lib::set_antenna(L, "measured/handset.qdant", {{"id", {2}}}, true);
quadriga_lib::set_antenna(L, quadriga_lib::qdant_read<double>("handset.qdant")[0], true);
quadriga_lib::set_antenna(L, arma::uvec{2}, true, {0, 1});
```

### Notes:
- **Bands**: with `center_freq`, only that band of the antenna of the nodes is set. A frequency that is not in
  `layout.center_freq` is added; the other bands of the nodes keep their antenna, and nodes that shared the old entry
  keep it. Until all bands of a node are set, `layout.is_valid()` reports it
- **Bands set one by one**: such a node keeps one line per band, also when the lines are equal, so a band added later is
  reported as not set for it. In a layout with one band, the line of a node set for that band also covers bands added
  later
- **Antenna lists**: after the call, equal descriptions share one entry, entries are numbered in order of first use (Tx
  nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- **Deferred evaluation**: an array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in
  a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- **QDANT file**: stored as a normalised absolute path; [layout_csv_write](#layout_csv_write) writes it relative to the CSV file. The
  array is used as stored, not rescaled to the frequency of a band. A missing or unreadable file throws
  `std::runtime_error`; without evaluation, an entry that the file does not have throws when the antennas are evaluated
- **Index form**: with `center_freq`, that band of the antenna of each node becomes the band of its entry, copied as in
  the other forms; the generic antenna (0) applies to all bands only. A node listed twice in `i_node` with different
  entries throws. In C++, a brace list such as `{2}` also builds a `std::string` and makes the call ambiguous, and so
  does a literal `0` or `{}` with only two arguments; pass a number, a variable, an `arma::uvec` or a `uint_list`, e.g.
  `uint_list(0)` or `arma::uvec{0}` to reset nodes to the generic antenna
- A failed call leaves the layout unchanged

### See also:
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the meaning of the arguments)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads the entries of a QDANT file)
- [layout](#layout) (the antenna lists and the member function `eval_antennas`)
- [layout_csv_write](#layout_csv_write) (writes the lines and exports the arrays)

---
<a id="set_orientation"></a>
## set_orientation
Set the orientation of nodes from their motion, or to a given value

Also in: [MATLAB](../mex/network-layouts-and-mobility.md#set_orientation) · [Python](../python/network-layouts-and-mobility.md#set_orientation)

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = false`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = true`: every row of the nodes gets exactly `orientation`

### Declaration:
```cpp
void quadriga_lib::set_orientation(
    quadriga_lib::layout_nodes &nodes,
    const arma::vec &orientation = {},
    const uint_list &i_node = {},
    bool force = false);
```

### Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`), updated in place
- **`orientation`** *(optional)* — Bank, tilt, heading in [rad], 3 finite values; empty = 0
- **`i_node`** *(optional)* — Node indices, 0-based; empty = all nodes
- **`force`** *(optional)* — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`:
  `orientation` on all rows

### Example:
```cpp
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_orientation(L.rx);                                  // The antennas look along the motion
quadriga_lib::set_orientation(L.rx, arma::vec{0.0, 0.0, 0.5});        // The same, turned by 0.5 rad
```

### Notes:
- **Tracks** (`force = false`): forward differences between the rows, the last step repeated, or for a closed track the
  step to the first row, as `qd_track.calc_orientation` of the NG library does; the angles are wrapped to [-π, π]. A
  step of zero length keeps the previous orientation, at the start the next one that exists. A vertical step has the
  tilt -90° (up) or 90° (down) and keeps the previous heading (a deliberate deviation from NG, where it is 0). A rising
  track has a negative tilt; a track that does not move keeps its orientation (plus the offset)
- **Point drops** (`force = false`): without an offset, the orientation is not touched; random orientations are not
  generated here
- **Mounted antennas**: `force = false` leaves nodes with `relative = 1` unchanged, offset included, since their
  orientation is relative to the motion, which [sample_track](#sample_track) applies at each snapshot; `force = true` sets them too
- A failed call leaves `nodes` unchanged

### See also:
- [generate_track](#generate_track) (creates tracks with this orientation)
- [sample_track](#sample_track) (applies the motion to mounted antennas)
