# Network layouts and mobility

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-network-layouts-and-mobility

| Function | Summary |
| --- | --- |
| [attach_terminals](#attach_terminals) | Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP |
| [drop_terminals](#drop_terminals) | Drop UTs into the cells of the Tx nodes of a layout |
| [eval_antennas](#eval_antennas) | Generate the antenna arrays of a layout from the antenna lines |
| [export_rays_qrt](#export_rays_qrt) | Write the rays of a layout with their link gain to a QRT file |
| [generate_layout](#generate_layout) | Network layout from a preset: BS sites and a drop of UTs |
| [generate_rays](#generate_rays) | Rays of the propagation links of a layout (stage 2 of the channel pipeline) |
| [generate_track](#generate_track) | Replace the rows of nodes by a generated track or a point drop |
| [get_channels_layout](#get_channels_layout) | Channel coefficients of a layout in one call |
| [get_gbsm_channels](#get_gbsm_channels) | Channel coefficients of a layout from given rays (stage 3 of the channel pipeline) |
| [init_link_states](#init_link_states) | Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline) |
| [layout_csv_read](#layout_csv_read) | Read the Tx and Rx CSV files of a layout |
| [layout_csv_write](#layout_csv_write) | Write a layout to the Tx and Rx CSV files |
| [sample_track](#sample_track) | Positions and orientations of a node at snapshot times |
| [set_antenna](#set_antenna) | Set the antenna of layout nodes from the arguments of generate_antenna |
| [set_antenna_array](#set_antenna_array) | Set the antenna of layout nodes from an evaluated array |
| [set_antenna_qdant](#set_antenna_qdant) | Set the antenna of layout nodes to an entry of a QDANT file |
| [set_orientation](#set_orientation) | Set the orientation of nodes from their motion, or to a given value |

A **layout** describes a radio network for channel simulation: the transmitters (Tx) and receivers (Rx), where they are
and how they move, which antenna each one has and which scenario applies. The functions of this section create layouts,
store them in files, move the nodes along tracks and compute the channels of all Tx-Rx pairs.

**What a layout holds** (a struct, fields in [layout_csv_read](#layout_csv_read)):

- The **Tx nodes** in `layout.tx` (base stations, access points) and the **Rx nodes** in `layout.rx` (user terminals,
  stations), each a struct of flat arrays. A node has a name, a sector number, an antenna and one or more rows:
  positions with orientation, scenario and indoor state
- The **antennas** of the nodes and the simulation frequencies `layout.center_freq`
- The **link states** `layout.links` of one realisation: LOS and indoor state, scenario and seeds of each Tx-Rx pair
  ([init_link_states](#init_link_states))

**Typical workflow:**

1. Create a layout: [generate_layout](#generate_layout) from a preset of 3GPP TR 38.901, ITU-R M.2412 or 3GPP TR 37.885, or
   [layout_csv_read](#layout_csv_read) from two CSV files; [drop_terminals](#drop_terminals) drops new UTs into the cells of the Tx nodes
2. Set the frequencies `layout.center_freq` and the antennas ([set_antenna](#set_antenna), [set_antenna_array](#set_antenna_array),
   [set_antenna_qdant](#set_antenna_qdant)); [eval_antennas](#eval_antennas) generates the antennas for new frequencies
3. Optionally, move the nodes: [generate_track](#generate_track) creates tracks, [set_orientation](#set_orientation) turns the antennas along the
   motion, [sample_track](#sample_track) gives the positions at given times
4. Compute the channels with [get_channels_layout](#get_channels_layout)
5. Store the layout with [layout_csv_write](#layout_csv_write)

For more control, step 4 runs in three stages: [init_link_states](#init_link_states) draws the link states, [generate_rays](#generate_rays) the rays of
the geometry-based stochastic model (GBSM), and [get_gbsm_channels](#get_gbsm_channels) applies the antennas, the motion and the link
gain. Between the stages, link states and rays can be inspected or edited, [attach_terminals](#attach_terminals) selects the serving cell
of each terminal, and [export_rays_qrt](#export_rays_qrt) stores the rays in a QRT file.

**Antennas.** Each node has one antenna. [set_antenna](#set_antenna) sets it from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna),
[set_antenna_array](#set_antenna_array) from an array, [set_antenna_qdant](#set_antenna_qdant) from an entry of a QDANT file; in the CSV files, antenna
columns describe it. [eval_antennas](#eval_antennas) generates the arrays from these descriptions. A node has one antenna for all
bands or one per band. A node without an antenna uses the generic antenna: two co-located isotropic elements with
vertical and horizontal polarization.

**Terms** (full list in the [Glossary](../cpp/general-usage-notes.md#glossary)):

- **Tx nodes**, **Rx nodes**: the two node structs of a layout, `layout.tx` and `layout.rx`. A function that works on
  one of them takes the struct itself or `rx_side` (`false`: Tx nodes, `true`: Rx nodes)
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
  the plane; each link uses the copy (*image*) nearest to the terminal
- **Orientation**: bank, tilt and heading of the antenna; a positive tilt points down. **Direction**: azimuth and
  elevation of the motion of a point drop

**Conventions.** Angles are in radians in the structs and in degrees in the CSV files. Indices are 1-based: the index
arguments, the returned indices and the index fields `offset`, `pairing` and, in `layout.links`, `i_tx`, `i_rx`,
`i_segment`, `i_conf` and `o2i_model`. `i_antenna`, the antenna of a node, is 0 for the generic antenna and 1 for the
first antenna of the layout; `sector`, `image`, `floor_id`, `indoor`, `floor` and the seeds are values, not indices.
Random values follow from the seeds and the positions, not from the order of evaluation, the batches or the number of
threads. The file format is described in [Layout CSV files](../../formats/layout_csv.md).

---
<a id="attach_terminals"></a>
## attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

Also in: [C++](../cpp/network-layouts-and-mobility.md#attach_terminals) · [Python](../python/network-layouts-and-mobility.md#attach_terminals)

- Evaluates every Tx node for each terminal and returns the one with the highest coupling gain (`'large_scale'`) or the
  highest RSRP (`'rsrp'`), with its site and the gains of all nodes
- Use it to link each terminal to its serving site in `layout.pairing` (see Example), so that the channel functions
  compute only the serving links

### Usage:
```matlab
[ node, site, gain_dB ] = quadriga_lib.attach_terminals( layout, rule, rays, i_freq, tx_port );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); Tx nodes required if
  there are terminals
- **`rule`** — `'large_scale'` or `'rsrp'`; default: `'large_scale'`
- **`rays`** — `'rsrp'`: rays of all links from [generate_rays](#generate_rays), possibly edited, required; cell array `[n_link, 1]`
  of `1xN` struct arrays, `N = 1` or `n_freq`; default: `{}`
- **`i_freq`** — Frequency index into `layout.center_freq`, 1-based; default: 1
- **`tx_port`** — Port of the Tx antennas, 1-based; default: 1

### Outputs:
- **`node`** — Serving Tx node per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`site`** — Serving site per terminal, 1-based; uint64; `[n_rx_terminal, 1]`
- **`gain_dB`** — Coupling gain or RSRP in [dB]; `[n_rx_terminal, n_tx_node]`

### Example:
```matlab
[ ~, site ] = quadriga_lib.attach_terminals( layout, 'large_scale' );
layout.pairing = [ site.'; 1:numel( site ) ];
```

### Notes:
- **Evaluation**: at the first rows of both nodes and `layout.center_freq(i_freq)`, with the link states of the first
  segment, the wrap image and the antennas as in [get_gbsm_channels](#get_gbsm_channels); `layout.pairing` is ignored
- **`'large_scale'`** (TR 38.901 Table 7.8-1): the link gain of [get_gbsm_channels](#get_gbsm_channels) plus the gain of one unit-gain LOS
  path through both antennas (port weights, element positions, orientations), as [get_channels_planar](channel-generation-functions.md#get_channels_planar) gives it; no
  rays
- **`'rsrp'`** (TR 36.873 eq. (8.1-1)): per Rx port the sum of `abs(h)^2` over the rays of [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel)
  (`output = 'rays'`, mobility `'phase'`), link gain included, transmit power 1
- **Ports**: both rules take `abs(h)^2` of Tx port `tx_port` and average it over the Rx ports, so a dual-polarized UT
  (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT antenna of the TR
- **Choice**: a terminal takes the highest value of its nodes; ties go to the lowest index. A Tx node without a
  propagation link (co-located) has -Inf; a terminal without any link gets node 1
- **Pairing**: `layout.pairing` holds pairs of site and terminal, so the Example links all sectors of the serving site

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays) (link states and rays)
- [get_gbsm_channels](#get_gbsm_channels) (channels of the linked pairs)

---
<a id="drop_terminals"></a>
## drop_terminals
Drop UTs into the cells of the Tx nodes of a layout

Also in: [C++](../cpp/network-layouts-and-mobility.md#drop_terminals) · [Python](../python/network-layouts-and-mobility.md#drop_terminals)

- Replaces the Rx nodes by new UTs `UT1`, `UT2`, ..., dropped uniformly into the cells of the sites with the rules of
  the scenario of each site (minimum distance, height, speed, indoor state), as [generate_layout](#generate_layout) drops them
- Use it to drop UTs for any set of Tx nodes, e.g. sites read from a CSV file or edited after [generate_layout](#generate_layout)
- The Tx nodes, the antenna lists and `layout.seed` stay; `pairing` and `links` are emptied, and the new UTs have the
  generic antenna

### Usage:
```matlab
layout_out = quadriga_lib.drop_terminals( layout_in, n_ut, bounds, ut, indoor, options, per_cell, seed );
```

### Inputs:
- **`layout_in`** — Valid layout struct, fields in [layout_csv_read](#layout_csv_read); the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 1e7 in total; 0 = no UTs
- **`bounds`** — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless
  `layout_in.wrap` has two vectors; default: `[]`
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** — As in [generate_layout](#generate_layout); an entry or option that no family
  of the sites uses raises an error; defaults: `[]`, `[]`, `[]`, `true`
- **`seed`** — Seed >= 0; -1 = `layout_in.seed` if it is set, else the system random device; default: -1

### Outputs:
- **`layout_out`** — The layout with the Rx nodes replaced by the dropped UTs

### Notes:
- **Cells**: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the
  wrap images when `layout_in.wrap` is set. A site with several sectors is split into wedges by the nearest sector
  heading (±60° around each boresight for three sectors at equal spacing)
- **Drop area**: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the
  images 0, ±a1, ±a2 and ±(a1 - a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- **Sites without a scenario family** (no scenario, or a conf name): height 1.5 m, no minimum distance unless `ut` entry
  1 gives one, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- **Sites with `cell` = 0**, e.g. the micro sites of `M.2412_DenseUrban` config 3, form no cells: with `per_cell`, each
  of their Tx nodes adds `n_ut` UTs to the cell that holds its position, and the UTs keep the minimum distance of the
  site's own family (`ut` entry 1 if given) from them. A site closer than twice the minimum distance to another one can
  leave a cell too little room; such a cell raises an error
- **Order and reproducibility**: with `per_cell` the UTs are ordered by cell. The drop is a function of the seed and
  does not depend on the thread count; each UT draws its heading, direction and floor from its own random stream, keyed
  by its name
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [generate_layout](#generate_layout) (layouts from presets, with this drop)
- [generate_track](#generate_track) (tracks for the UTs)

---
<a id="eval_antennas"></a>
## eval_antennas
Generate the antenna arrays of a layout from the antenna lines

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout) · [Python](../python/network-layouts-and-mobility.md#eval_antennas)

- Generates the arrays `antenna` of a layout from its antenna descriptions `antenna_args`, for the entries that are not
  evaluated yet, e.g. after [layout_csv_read](#layout_csv_read) or [set_antenna](#set_antenna) with the evaluation turned off
- With `center_freq`, sets new simulation frequencies and generates all antennas for them

### Usage:
```matlab
layout_out = quadriga_lib.eval_antennas( layout_in, center_freq );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read)
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; replaces `layout_in.center_freq`
  and regenerates all descriptions; `[]` = keep `layout_in.center_freq`; default: `[]`; `[n_freq]`

### Outputs:
- **`layout_out`** — The layout with evaluated `antenna`

### Notes:
- **Antenna descriptions**: a line of `antenna_args` holds the cells of the antenna columns `antenna_cols` as in the
  [Layout CSV files](../../formats/layout_csv.md#layout-csv-antennas). Nodes with equal lines share one antenna; `i_antenna` of a node
  is its index, 0 the generic antenna. No line or an empty line is an array without description (set with
  [set_antenna_array](#set_antenna_array)); a line without antenna type is a band that is not set yet
- **Evaluation**: a line for all bands is generated once per band at its frequency, so that an array spaced in
  wavelengths keeps its spacing (an array of co-located elements is stored once for all bands); a line of a band at the
  frequency of that band; a `qdant` line is read as stored. Until an antenna is evaluated (its entry of `antenna` holds
  no arrays, or an array without elements), its nodes use the generic antenna
- **New frequencies**: arrays without a description are kept and must fit the new list
- **Errors**: a band without description and an array spaced in wavelengths in a layout without frequencies raise an
  error; the message names the node
- This is the member function `eval_antennas` of the C++ layout class; its in/out argument `layout` is split into
  `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout, optionally with evaluated antennas)
- [set_antenna](#set_antenna) (sets the antennas of nodes)

---
<a id="export_rays_qrt"></a>
## export_rays_qrt
Write the rays of a layout with their link gain to a QRT file

Also in: [C++](../cpp/network-layouts-and-mobility.md#export_rays_qrt) · [Python](../python/network-layouts-and-mobility.md#export_rays_qrt)

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) and
  [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local; add the returned shift to the positions read back

### Usage:
```matlab
shift = quadriga_lib.export_rays_qrt( fn, layout, rays );
```

### Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states): one frequency for the TR methods (`'quadriga'`: 1
  to 127), Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal
  at a site position; fields in [layout_csv_read](#layout_csv_read)
- **`rays`** — Rays of all links from [generate_rays](#generate_rays), possibly edited, one object with all frequencies per link; `{}`
  = generated for batches of sites with up to 256 links, which bounds the memory; cell array `[n_link, 1]` of 1x1
  structs; default: `{}`

### Outputs:
- **`shift`** — Shift of the local coordinates in x and y in [m]; `[2, 1]`

### Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [get_gbsm_channels](#get_gbsm_channels) at the row, [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `'drift'`. A CIR read back
  gives the channel of [get_gbsm_channels](#get_gbsm_channels) (`output = 'rays'`, `mobility = 'drift'`) at the row within float32
  accuracy
- **Reading back**: [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) with the method of the links and the `n_interact` of the file as
  `path_type`, then [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) with link gain 0 and mobility `'drift'`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the format**: no tap index and no ground permittivity, so a CIR gives rays, not the taps of the TR, and
  moving a ground-reflection ray from a file raises an error
- The file is written to `fn + '.part'` and renamed at the end; a failed call leaves an existing file unchanged

### See also:
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the rays and the link gain)
- [qrt_file_read](channel-functions.md#qrt_file_read), [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths), [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (evaluation of the file)

---
<a id="generate_layout"></a>
## generate_layout
Network layout from a preset: BS sites and a drop of UTs

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_layout) · [Python](../python/network-layouts-and-mobility.md#generate_layout)

- Creates a standard evaluation layout with the values of its tables: the hexagonal and indoor layouts of 3GPP TR
  38.901, the test environments of ITU-R M.2412 and the road layouts of 3GPP TR 37.885
- The result has the sites `BS1`, `BS2`, ... with one Tx node per sector and the UTs `UT1`, `UT2`, ... as point drops;
  set the frequencies and the antennas next ([set_antenna](#set_antenna)), until then the nodes use the generic antenna
- The other inputs change single values of the preset: an entry of a tuple that is NaN or omitted keeps the value of the
  preset; an entry or option that the preset does not use raises an error

### Usage:
```matlab
layout = quadriga_lib.generate_layout( type, n_ut, spacing, grid, area, bs, ut, indoor, options, ...
    per_cell, seed );
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

### Outputs:
- **`layout`** — Layout struct, fields in [layout_csv_read](#layout_csv_read)

### Example:
```matlab
layout = quadriga_lib.generate_layout( '38.901_UMa', 10, [], [], [], [], [NaN NaN NaN 0 90], [], [], true, 42 );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2] ) );
```

### Notes:
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

### See also:
- [drop_terminals](#drop_terminals) (the same drop for any set of Tx nodes)
- [generate_track](#generate_track) (tracks for the UTs)
- [set_antenna](#set_antenna) (antennas of the nodes)

---
<a id="generate_rays"></a>
## generate_rays
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_rays) · [Python](../python/network-layouts-and-mobility.md#generate_rays)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [init_link_states](#init_link_states): large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [get_gbsm_channels](#get_gbsm_channels) applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [attach_terminals](#attach_terminals) and [export_rays_qrt](#export_rays_qrt)
- The rays of a link do not depend on the other requested links

### Usage:
```matlab
rays = quadriga_lib.generate_rays( layout, i_link, scatterers );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); raises an error if a
  member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** — Propagation links, 1-based, without repetition; `[]` = all (none for links without entries); default:
  `[]`; `[n_out]`
- **`scatterers`** — Solve the first- and last-bounce scatterers, needed for the mobility `'drift'`; default: `false`

### Outputs:
- **`rays`** — Rays per link and frequency, a cell array `[n_out, 1]` of `1xN` struct arrays with the fields of
  [ssf_rays](geometry-based-stochastic-model.md#ssf_rays), `N = n_freq` with one frequency each for the TR methods, `N = 1` with all frequencies for `'quadriga'`

### Notes:
- **Chain**: [gbsm_init_phases](geometry-based-stochastic-model.md#gbsm_init_phases) and [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the random sets, [path_loss](geometry-based-stochastic-model.md#path_loss) (for `rnd_he` and the SF standard
  deviation), [lsp_generate](geometry-based-stochastic-model.md#lsp_generate), [ssf_clusters](geometry-based-stochastic-model.md#ssf_clusters), [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) and, with `scatterers`, [ssf_scatterers](geometry-based-stochastic-model.md#ssf_scatterers), once for the
  links of one site, image, conf and seed
- **Position**: the first row of the segment of the terminal and the site (moved by its wrap image) at the same time; a
  moving site is sampled at the time of the row, without times on the Rx track at the row of the same index. A site with
  more than one row or a speed uses dual mobility; the other sites are not affected
- **TR methods**: the SOS preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance,
  `angle_model = 'additive'`, `pol_model = '38.901'`, one set of clusters and rays per frequency (`'38.901_MF'`:
  clusters at the anchor frequency, rays per frequency)
- **`'quadriga'`**: bias removal over the positions of all links of the group, the legacy outdoor distance,
  `'rotation'`, `'quadriga'`, one ray set for all frequencies, always with scatterers
- **Edited link states**: the conf and the seed of a link follow from its scenario and the `los` and `indoor` of
  `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged.
  Building UTs are on the floor of the Rx nodes (else from z) at the distance of the stored O2I inputs, the others on
  floor 0 at 0 m

### See also:
- [init_link_states](#init_link_states) (stage 1), [get_gbsm_channels](#get_gbsm_channels) (stage 3)
- [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) (the ray data)

---
<a id="generate_track"></a>
## generate_track
Replace the rows of nodes by a generated track or a point drop

Also in: [C++](../cpp/network-layouts-and-mobility.md#generate_track) · [Python](../python/network-layouts-and-mobility.md#generate_track)

- Gives nodes a straight, circular or street-like track, or makes them point drops with a speed and a direction, the
  short-term mobility of 3GPP TR 38.901
- All nodes with the name of a given node (the sectors of a site, the panels of a terminal) get the same track. It
  starts at the first position of the node and is horizontal at the height of the first row
- The antennas look along the motion, as [set_orientation](#set_orientation) turns them; [sample_track](#sample_track) gives the positions at
  snapshot times

### Usage:
```matlab
nodes_out = quadriga_lib.generate_track( nodes_in, i_node, type, length, speed, direction, spacing, ...
    options, seed );
```

### Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`i_node`** — Node indices, 1-based, not empty
- **`type`** — Type of the track, not case sensitive:
  - `point`: a point drop with `speed` and a horizontal `direction`; the rows after the first are removed; `length` is
    not used
  - `linear`: a straight track of `length` (default 1 m) in the start direction
  - `circular`: a closed circle of circumference `length` (default 20π m, radius 10 m); the first and the last point are
    equal
  - `street`: the route of a car through a city grid of straight streets and crossings, of `length` (default 1000 m),
    see Notes
- **`length`** — Length in [m], finite and >= 0; 0 = default of the type; default: 0
- **`speed`** — Speed in [m/s], finite and >= 0, and > 0 for tracks; NaN = the speed of the first row of the node if it
  is valid (> 0 for a track); without one, `point` gets 0 and the other types raise an error; default: NaN
- **`direction`** — Start direction (azimuth) in [rad], finite; one value for all nodes, or one per entry of `i_node`
  (nodes of one name need equal values); `[]` = uniform random in [0, 2π); default: `[]`
- **`spacing`** — Point spacing in [m], > 0; NaN = two points (`linear`), 129 points (`circular`), 1 m (`street`);
  default: NaN
- **`options`** — Struct of options of `street`, one value per field: `street_length_min` = 50, `street_length_mu` =
  187, `street_length_std` = 83 (all in [m], >= 0; with a std of 0, mu must be >= min and > 0), `curve_radius` = 10 [m],
  `turn_probability` = 0.5 (in [0, 1]); other fields, and these fields for the other types, raise an error; default:
  `[]`
- **`seed`** — Seed, >= 0 for reproducible draws; -1 = system random device; below -1 raises an error; default: -1

### Outputs:
- **`nodes_out`** — The nodes with the generated rows

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( '', 'rx.csv' );
layout.rx = quadriga_lib.generate_track( layout.rx, 1, 'street', 500, 10, [], NaN, struct( 'turn_probability', 0.3 ), 42 );
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
  length `2 · r · cos(φ / 2)` for the turn angle φ and the `curve_radius` r. The direction stays within 105° of the
  start direction, so that the car does not drive in circles
- **Values of the new rows**: `linear`, `circular` and `street` give tracks of equal `speed` without `direction`,
  oriented along the motion as [set_orientation](#set_orientation) with `force = false`; nodes with `relative = 1` keep the orientation
  of their first row on all rows. `point` keeps the orientation of the node. `scenario`, `indoor`, `floor`, `d2d_in` and
  the extra columns take the values of the first row; `time` is cleared
- **Reproducibility**: a seed >= 0 gives the same track on all platforms. Each name (all its nodes) has its own random
  stream, so the track does not depend on the other nodes or the order in `i_node`
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### See also:
- [sample_track](#sample_track) (snapshots of a track)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="get_channels_layout"></a>
## get_channels_layout
Channel coefficients of a layout in one call

Also in: [C++](../cpp/network-layouts-and-mobility.md#get_channels_layout) · [Python](../python/network-layouts-and-mobility.md#get_channels_layout)

- Computes one channel per Tx node, Rx node and frequency, for the pairs in `layout.pairing` (empty: all pairs)
- Runs the stages [init_link_states](#init_link_states), [generate_rays](#generate_rays) and [get_gbsm_channels](#get_gbsm_channels) with their default settings; the
  results are those of the stages called one by one
- Without link states in `layout.links`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

### Usage:
```matlab
[ channels, index ] = quadriga_lib.get_channels_layout( layout, times, output, method, mobility, ...
    overlap, seed, config_dirs );
```

### Inputs:
- **`layout`** — Layout struct with `center_freq`; fields in [layout_csv_read](#layout_csv_read); antennas as evaluated in
  `layout.antenna`, nodes without one use the generic antenna (see [get_gbsm_channels](#get_gbsm_channels))
- **`times`** — Snapshot times in [s]; `[]` = one snapshot per row; default: `[]`; `[n_snap]`
- **`output`** — `'taps'` or `'rays'`; default: `'taps'`
- **`method`** — `'38.901'`, `'38.901_MF'` or `'quadriga'`; `''` = the method of filled links, else `'38.901'`; another
  method than that of filled links raises an error; default: `''`
- **`mobility`** — `'phase'`, `'drift'` or `'auto'`; default: `'auto'`
- **`overlap`** — Overlap of consecutive segments, in [0, 1]; default: 0.5
- **`seed`** — Master seed for empty links, as in [init_link_states](#init_link_states); with filled links only -1; default: -1
- **`config_dirs`** — Folders of user conf files for empty links, searched before the shipped folder; string or cell
  array of strings; default: `{}`

### Outputs:
- **`channels`** — Channels, Tx node fastest, then Rx node, then frequency; struct array, fields in
  [hdf5_read_channel](channel-functions.md#hdf5_read_channel); `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

### Example:
```matlab
layout = quadriga_lib.generate_layout( '38.901_UMa', 1 );
layout.center_freq = 3.5e9;
channels = quadriga_lib.get_channels_layout( layout );
```

### Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout.links` holds none when `layout.links.method` is empty, e.g. after `layout.links = []`. With
  link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives an empty result
- **Scatterers**: the mobility `'drift'` (also `'auto'` with `'quadriga'`) solves the scatterers
- The C++ output `index` is the second output; the wrapper adds the folder of the shipped conf files to `config_dirs`

### See also:
- [init_link_states](#init_link_states), [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (the three stages)
- [generate_layout](#generate_layout) (layouts of the TR scenarios)

---
<a id="get_gbsm_channels"></a>
## get_gbsm_channels
Channel coefficients of a layout from given rays (stage 3 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#get_gbsm_channels) · [Python](../python/network-layouts-and-mobility.md#get_gbsm_channels)

- Applies the antennas of the nodes, the motion along their tracks and the link gain to the rays of [generate_rays](#generate_rays),
  and returns one channel per Tx node, Rx node and frequency
- Only the channel links of the given propagation links and of the pairs in `layout.pairing` (empty: all) are computed;
  the sectors of a site share the rays of their propagation link
- Tracks with several segments are joined by [crossfade_segments](channel-functions.md#crossfade_segments)

### Usage:
```matlab
[ channels, index ] = quadriga_lib.get_gbsm_channels( layout, rays, i_link, times, output, mobility, ...
    oxygen, use_absolute_delays, overlap );
```

### Inputs:
- **`layout`** — Layout struct with links from [init_link_states](#init_link_states); fields in [layout_csv_read](#layout_csv_read); raises an error if a
  member that determines the links changed since then
- **`rays`** — Rays of the links from [generate_rays](#generate_rays), possibly edited, with the frequencies of the layout; cell array
  `[n_out, 1]` of `1xN` struct arrays, `N = 1` or `n_freq`
- **`i_link`** — The propagation links of `rays`, 1-based; `[]` = all links; default: `[]`; `[n_out]`
- **`times`** — Snapshot times in [s], finite, the same for all nodes; `[]` = one snapshot per row; default: `[]`;
  `[n_snap]`
- **`output`** — `'taps'` or `'rays'`, see [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel); default: `'taps'`
- **`mobility`** — `'phase'`, `'drift'` (needs rays with scatterers) or `'auto'` (phase for the TR methods, drift for
  `'quadriga'`); default: `'auto'`
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `false`
- **`use_absolute_delays`** — Include the LOS delay in all delays; default: `false`
- **`overlap`** — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1], see
  [crossfade_segments](channel-functions.md#crossfade_segments); default: 0.5

### Outputs:
- **`channels`** — Channels, struct array, fields in [hdf5_read_channel](channel-functions.md#hdf5_read_channel); `[1, n_channel]`
- **`index`** — Tx node, Rx node and frequency index of each channel, 1-based; uint64; `[3, n_channel]`

### Notes:
- **Order and names**: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the
  sector where all Tx nodes (or all Rx nodes) have sector 0
- **Snapshots** of both ends from [sample_track](#sample_track) over `times`; a node with one snapshot is repeated to the count of
  the other; two tracks without `times` need equal row counts
- **Link gain** per snapshot, `G = -PL - L_O2I + 10 · log10(SF)` in dB: the path loss PL of [path_loss](geometry-based-stochastic-model.md#path_loss) with the
  random environment height `rnd_he` held at the start of the segment, the shadow fading SF and the K-factor ratio of
  [lsp_generate](geometry-based-stochastic-model.md#lsp_generate) from [gbsm_draw](geometry-based-stochastic-model.md#gbsm_draw) of the `'lsf'` set at the snapshot, and the O2I loss of [o2i_loss](geometry-based-stochastic-model.md#o2i_loss) for building
  and car UTs with the model of the link state (the deviation of a building UT at the snapshot, that of a car UT at the
  first row of the terminal for its whole track, TR 38.901 eq. 7.4-4). A link whose ends do not move is evaluated once
- **Antennas**: [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) with the node antennas at the frequency (entry `i_freq` of the `1xN` struct
  array of the antenna in `layout.antenna`, or its one entry for all bands) and the Tx positions moved by the wrap image
  of the link. Nodes without an antenna and antennas that are not evaluated use the generic antenna; an antenna spaced
  in wavelengths that is not evaluated raises an error
- **Frequencies**: rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency
  `i_freq` (`'quadriga'`)
- **Segments**: the snapshots of a segment are those whose row (see [sample_track](#sample_track)) is in it; each segment also covers
  the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row. The
  selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing
  `times`
- **Relative delays**: in `'phase'` the paths of each segment relative to its own LOS, in `'drift'` relative to the LOS
  at the snapshot
- The C++ output `index` is the second output

### See also:
- [generate_rays](#generate_rays) (stage 2)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="init_link_states"></a>
## init_link_states
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

Also in: [C++](../cpp/network-layouts-and-mobility.md#init_link_states) · [Python](../python/network-layouts-and-mobility.md#init_link_states)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout_out.links` (fields in
  [layout_csv_read](#layout_csv_read))
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout.pairing` does not limit the links; it selects at the output of the channel functions

### Usage:
```matlab
layout_out = quadriga_lib.init_link_states( layout_in, method, o2i_low, pol_variability, anchor_freq, ...
    sos_preset, seed, config_dirs );
```

### Inputs:
- **`layout_in`** — Layout struct, valid, with `center_freq`; fields in [layout_csv_read](#layout_csv_read)
- **`method`** — `'38.901'`, `'38.901_MF'` (TR 38.901 clause 7.6.5.1) or `'quadriga'`, see [generate_rays](#generate_rays); default:
  `'38.901'`
- **`o2i_low`** — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; the other families have
  one building model; `[]` = the share of TR 38.901 Table 7.8-1 (0.5); default: `[]`
- **`pol_variability`** — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only; default: `false`
- **`anchor_freq`** — Anchor frequency of `'38.901_MF'` in [Hz]; default: 2e9
- **`sos_preset`** — SOS preset of the GBSM random sets, see [sos_preset](sum-of-sinusoids-random-number-generator.md#sos_preset); `''` = `Exp256` (TR methods), `Comb256`
  (`'quadriga'`); default: `''`
- **`seed`** — Master seed, >= -1; -1 = `layout_in.seed`, if that is -1 one draw of the system random device; default:
  -1
- **`config_dirs`** — Folders of user conf files, searched before the shipped folder; string or cell array of strings;
  default: `{}`

### Outputs:
- **`layout_out`** — The layout with `links` and `seed` (the seed used) set; the rest is unchanged

### Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 1 first). Co-located pairs have no link; a later segment that starts at the site raises an
  error. A layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [conf_file_list](geometry-based-stochastic-model.md#conf_file_list), the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); all distances of the
  link use this virtual site position
- **O2I state** per terminal, from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3):
  building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A
  (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the
  O2I model and `d2d_in` as in [o2i_loss](geometry-based-stochastic-model.md#o2i_loss). The values `indoor`, `floor` and `d2d_in` of the Rx nodes fix them; an
  exact conf name that contradicts a fixed `indoor` raises an error
- **One O2I state per terminal** (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal
  linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a
  site fixes it for the links to all sites; InH and InF have none
- **LOS state** per link: a uniform field per site, image and family at the first position of the segment, at the
  `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [los_probability](geometry-based-stochastic-model.md#los_probability) at
  `d2d_out = max(d2d - d2d_in, 0)`; InF-HH is always LOS
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
- **Layout hash**: `links.layout_hash` covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I
  columns, clutter, scenario overrides, wrap vectors and whether a site moves. The later stages raise an error when it
  no longer matches the layout; `layout.links = []` discards the links. Orientations, antennas and `pairing` may change
  without new links
- **Forced states**: editing `los` or `indoor` of `layout.links` before [generate_rays](#generate_rays) forces that state; the conf
  and its seed follow from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values
  of the forced state from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after
  another realisation
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged; the
  wrapper adds the folder of the shipped conf files to `config_dirs`

### See also:
- [generate_rays](#generate_rays), [get_gbsm_channels](#get_gbsm_channels) (stages 2 and 3)
- [get_channels_layout](#get_channels_layout) (all stages in one call)

---
<a id="layout_csv_read"></a>
## layout_csv_read
Read the Tx and Rx CSV files of a layout

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout_csv_read) · [Python](../python/network-layouts-and-mobility.md#layout.csv_read)

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](../../formats/layout_csv.md) and returns the
  layout struct, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `false`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

### Usage:
```matlab
[ layout, extra_tx, extra_rx ] = quadriga_lib.layout_csv_read( fn_tx, fn_rx, center_freq, eval_antennas );
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = no Tx nodes or no Rx nodes
- **`center_freq`** — Simulation frequencies in [Hz], strictly ascending and positive; `[]` = from the files; default:
  `[]`; `[n_freq]`
- **`eval_antennas`** — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and `i_antenna`
  are kept, and the nodes use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `true`

### Outputs:
- **`layout`** — The layout, a 1x1 struct, which all functions of this section take and return; `name` and `links` are
  empty. Angles in rad; `offset`, `pairing`, `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model` are 1-based. A field
  that is missing or empty in an input struct takes the value of an empty member; unknown fields are rejected.

  Fields of `layout`:
  - `name` — Name of the layout; not stored in the CSV files; String
  - `tx`, `rx` — Tx nodes (BS, AP) and Rx nodes (UT, STA), fields below; Struct
  - `links` — Link states of the current realisation, fields below; `[]` = none; Struct
  - `seed` — Seed of the layout, -1 = not set; int64, scalar
  - `center_freq` — Simulation frequencies (bands) in [Hz], strictly ascending; `[n_freq, 1]`
  - `antenna_cols` — Names of the antenna columns of the lines in `antenna_args`; Cell of strings
  - `antenna_args` — Antenna descriptions, one line for all bands or one per band (see [eval_antennas](#eval_antennas)); Cell of cells
    of strings
  - `antenna` — Evaluated arrays, entry `k` a 1xN struct array as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), N = 0 (not evaluated), 1 (all
    bands) or `n_freq`; Cell `[n_antenna, 1]`
  - `wrap` — Lattice vectors of the wrap-around in [m], one per column; two vectors form a reduced basis,
    `0 <= a1·a2 <= min(abs(a1)^2, abs(a2)^2) / 2`, so that ±a1, ±a2 and ±(a1 - a2) are the nearest images; empty = none;
    `[2, 1]` or `[2, 2]`
  - `pairing` — Site index and terminal index of the linked pairs; empty = all pairs; uint64, `[2, n_pair]`

  Fields of the node structs `layout.tx` and `layout.rx`:
  - `name` — Site or terminal name, 1 to 255 bytes, without comma, semicolon, quote and line break, and without leading
    or trailing blank; Cell `[n_node, 1]`
  - `sector` — Sector number; (`name`, `sector`) is unique; uint32, `[n_node, 1]`
  - `i_antenna` — Antenna: index into the antenna lists of the layout, 0 = generic antenna; uint32, `[n_node, 1]`
  - `offset` — First row of each node; the rows of node `i` are `offset(i)` to `offset(i+1) - 1` (the last node to the
    end); uint64, `[n_node, 1]`
  - `pos` — Positions in [m]; `[3, n_row]`
  - `orientation` — Bank, tilt, heading in [rad]; `[3, n_row]`
  - `direction` — Azimuth, elevation of travel in [rad]; NaN azimuth = along the orientation; NaN on all rows of a
    track; empty = all NaN; `[2, n_row]`
  - `speed` — Speed in [m/s] from this row on; empty = 0; `[n_row, 1]`
  - `time` — Time in [s]; NaN on all rows of a node = not given; empty = none; `[n_row, 1]`
  - `scenario` — Scenario family or conf name, `''` = none; empty = none; Cell `[n_row, 1]`
  - `indoor` — -1 = draw, 0 outdoor, 1 building, 2 car, 3 residential building; empty = all -1; int32, `[n_row, 1]`
  - `floor` — Floor number from 1, -1 = from z; empty = all -1; int32, `[n_row, 1]`
  - `d2d_in` — Indoor 2D distance in [m], NaN = draw; empty = all NaN; `[n_row, 1]`
  - `clutter` — `d_clutter`, `r_clutter` and `h_clutter`, NaN = model default; empty = all NaN; `[3, n_node]`
  - `scenpar_override` — Overrides of scenario parameters, one field per key, NaN = none; Struct of `[n_node, 1]`
  - `seed` — Seed of the site, -1 = derived from the seed of the layout; empty = all -1; int64, `[n_node, 1]`
  - `relative` — 1 = orientation relative to the direction of motion; empty = all 0; uint32, `[n_node, 1]`
  - `cell` — Tx nodes only: 0 = the site has no drop cells of its own ([drop_terminals](#drop_terminals)); empty = all 1; uint32,
    `[n_node, 1]`
  - `extra_names`, `extra` — Names of unknown CSV columns and their cells as text; Cell `[n_extra, 1]` of cells
    `[n_row, 1]`

  Fields of `layout.links`:
  - `i_tx`, `i_rx`, `i_segment` — Site, terminal and segment of each link; uint32, `[n_link, 1]`
  - `image` — Wrap image of the site, 0 = the real site; uint32, `[n_link, 1]`
  - `i_conf`, `conf_names` — Index into `conf_names`, from scenario, LOS and indoor state; resolved conf names; uint32,
    `[n_link, 1]`; cell
  - `los`, `indoor`, `floor_id`, `d2d_in` — 1 = line of sight; indoor state at the start of the segment, as in the
    nodes; floor number - 1 (0 outdoors) and indoor 2D distance in [m] of the drawn state; `[n_link, 1]`
  - `o2i_rnd`, `o2i_model` — Random inputs of [o2i_loss](geometry-based-stochastic-model.md#o2i_loss); building O2I model, index into the list of the family, for
    every terminal of a family with O2I, also outdoors; `[n_terminal, 4]`; uint32, `[n_terminal, 1]`
  - `gbsm_seed`, `seed` — Seed of the random phases, equal for one site, image and conf; master seed, the base of the
    random values of all stages, -1 = not set; int64, `[n_link, 1]`; scalar
  - `method`, `sos_preset`, `anchor_freq`, `pol_variability` — Method and SOS preset as used (an empty `method` means
    not filled); anchor frequency of the multi-frequency model in [Hz]; polarization power variability; Strings, scalar,
    logical
  - `conf_files`, `layout_hash` — Resolved paths of the conf files, empty = not resolved; hash of the layout fields that
    determine the links (see [init_link_states](#init_link_states)); Cell; uint64
- **`extra_tx`**, **`extra_rx`** — Names of the unknown columns of the Tx and Rx file, in file order; cell array of
  strings; `[n_extra, 1]`

### Notes:
- **Errors**: a violation of the format and a layout that is not valid raise an error with the file name, the line and
  the column (see [Reading](../../formats/layout_csv.md#layout-csv-reading)); with `eval_antennas`, so do a line that
  [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a layout without
  frequencies
- **Validity**: a valid layout, which the functions of this section check, has node structs with fitting sizes, a
  fitting `offset`, finite positions and orientations, valid names and values in range, and follows the rules of the
  [Layout CSV files](../../formats/layout_csv.md#layout-csv-columns) for `time`, `speed`, `direction`, `floor` and `d2d_in` and for the
  values that the nodes of one site or terminal share ([Nodes and rows](../../formats/layout_csv.md#layout-csv-rows)); the antenna
  lists (a node with a band that is not set is reported), the frequencies, `wrap`, `pairing` and `links` are checked too
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](../../formats/layout_csv.md#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas = false`

### See also:
- [layout_csv_write](#layout_csv_write) (stores a layout)
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the antenna types and arguments of the antenna columns)

---
<a id="layout_csv_write"></a>
## layout_csv_write
Write a layout to the Tx and Rx CSV files

Also in: [C++](../cpp/network-layouts-and-mobility.md#layout_csv_write) · [Python](../python/network-layouts-and-mobility.md#layout.csv_write)

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](../../formats/layout_csv.md); reading the files back gives the same layout within 15 significant
  digits, except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with [set_antenna_array](#set_antenna_array), are written to QDANT files next
  to the CSV file
- A failed call leaves the existing files unchanged

### Usage:
```matlab
quadriga_lib.layout_csv_write( fn_tx, fn_rx, layout, compact );
```

### Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; `''` = not written; different files (also if the paths are spelled
  differently)
- **`layout`** — Layout struct, must be valid; fields in [layout_csv_read](#layout_csv_read)
- **`compact`** — `true`: cells that the reader fills in are left empty; `false`: every cell is written; default: `true`

### Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](../../formats/layout_csv.md#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them raises an error. An antenna set with [set_antenna_qdant](#set_antenna_qdant) is written as a
  reference to its file, not exported

### See also:
- [layout_csv_read](#layout_csv_read) (reads a layout)

---
<a id="sample_track"></a>
## sample_track
Positions and orientations of a node at snapshot times

Also in: [C++](../cpp/network-layouts-and-mobility.md#sample_track) · [Python](../python/network-layouts-and-mobility.md#sample_track)

- Returns the position and the orientation of one node at each snapshot time, for point drops and tracks;
  [get_gbsm_channels](#get_gbsm_channels) samples both ends of each link this way
- Without `times`, each row of the node is one snapshot
- The orientation includes the rotation of mounted antennas (`relative = 1`), so it is the orientation in the GCS

### Usage:
```matlab
[ pos, orientation, i_row ] = quadriga_lib.sample_track( nodes, i_node, times );
```

### Inputs:
- **`nodes`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`i_node`** — Node index, 1-based
- **`times`** — Snapshot times in [s], finite; `[]` = one snapshot per row; `[n_snap]`

### Outputs:
- **`pos`** — Positions in [m]; `[3, n_snap]`
- **`orientation`** — Orientations (bank, tilt, heading) in [rad] in the GCS, including the mounting rotation;
  `[3, n_snap]`
- **`i_row`** — Row at or before each snapshot, 1-based, index into the arrays of `nodes` (not into the rows of the
  node); uint64; `[n_snap, 1]`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
[ pos, ori ] = quadriga_lib.sample_track( layout.rx, 1, 0:0.1:1 );
```

### Notes:
- **Point drop** (one row): `p(t) = p0 + v · t · d` with the speed v of the row and the unit vector d of its direction
  (azimuth, elevation; an elevation of NaN is 0). Without a direction (NaN azimuth),
  `d = (cos(β) · cos(α), cos(β) · sin(α), -sin(β))` with the tilt β and the heading α of the orientation; the bank has
  no effect. The orientation is constant
- **Track** (several rows): the time of row i is `time`, else it follows from the speed v(i) of the row, `t(1) = 0`,
  `t(i+1) = t(i) + abs(p(i+1) - p(i)) / v(i)`; a step of zero length with speed 0 takes no time. The position is
  interpolated linearly in time, bank and heading along the shorter way and the tilt linearly; before the first and
  after the last time, the first and the last row hold. If several rows have the same time, the later row holds at that
  time. The times in `times` need not increase. A track with neither `time` nor a speed raises an error if `times` is
  given; a moving step with speed 0 raises an error
- **Mounted antennas** (`relative = 1`): the orientation is relative to the direction of motion, and the output is
  `R_motion · R_orientation`, with bank 0 in `R_motion`. For a point drop, the motion is its direction, which must be
  given. For a track, it is the direction of the step from the row at or before the snapshot to the next row (the last
  step repeated at the end); a step of zero length takes the previous direction, at the start the next one that exists.
  The stored orientation is interpolated first, then rotated by the motion
- **Checks**: fields of `nodes` with sizes that do not fit, and a `direction` on a track, raise an error; the other
  rules of a valid layout ([layout_csv_read](#layout_csv_read)) are not checked

### See also:
- [generate_track](#generate_track) (creates tracks)
- [set_orientation](#set_orientation) (orientation from the motion)

---
<a id="set_antenna"></a>
## set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_antenna) · [Python](../python/network-layouts-and-mobility.md#set_antenna)

- Sets the antenna of Tx nodes or Rx nodes from the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna); [set_antenna_array](#set_antenna_array) sets an
  array, e.g. a measured pattern read with [qdant_read](antennas-transducers-and-patterns.md#qdant_read)
- Without `center_freq`, the antenna applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the arguments as an antenna description, which [layout_csv_write](#layout_csv_write) writes as antenna columns

### Usage:
```matlab
layout_out = quadriga_lib.set_antenna( layout_in, type, args, rx_side, i_node, center_freq, evaluate );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read); its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna), e.g. `'3GPP'`, `'dipole'`, or `'none'` (no antenna, for all bands,
  without `args`)
- **`args`** — Struct with the arguments of [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) as fields, with the meaning given there: `res`,
  `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers,
  vectors or matrices (`steer`, `beams` row by row as there), `separate_beams` and `apply_weights` as 0 or 1; a field
  that the type does not use raises an error; `pattern` is not available (an array built with it is set with
  [set_antenna_array](#set_antenna_array)); the field `center_freq` is the same as the argument; e.g.
  `struct( 'panel', [4 4 1 2], 'pol', 3 )`; default: `[]` (none)
- **`rx_side`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of
  new antennas use the generic antenna until [eval_antennas](#eval_antennas) is called; default: `true`

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna( layout, '3GPP', struct( 'panel', [4 4 1 2], 'pol', 3 ), false, [], 3.5e9 );
```

### Notes:
- **Bands**: with `center_freq`, only that band of the antenna of the nodes is set. A frequency that is not in
  `layout_in.center_freq` is added; the other bands of the nodes keep their antenna, and nodes that shared the old entry
  keep it. Until all bands of a node are set, the layout is not valid, and the functions that check it name the node
- **Bands set one by one**: such a node keeps one line per band, also when the lines are equal, so a band added later is
  reported as not set for it. In a layout with one band, the line of a node set for that band also covers bands added
  later
- **Antenna lists**: after the call, equal descriptions share one entry, entries are numbered in order of first use (Tx
  nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- **Deferred evaluation**: an array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in
  a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (the meaning of the arguments)
- [eval_antennas](#eval_antennas) (generates the antennas)
- [layout_csv_write](#layout_csv_write) (writes the descriptions and exports the arrays)
- [set_antenna_array](#set_antenna_array) (sets an array)
- [set_antenna_qdant](#set_antenna_qdant) (sets an entry of a QDANT file)

---
<a id="set_antenna_array"></a>
## set_antenna_array
Set the antenna of layout nodes from an evaluated array

Also in: [Python](../python/network-layouts-and-mobility.md#set_antenna_array) · Not in: C++

- Sets the antenna of Tx nodes or Rx nodes from an array, e.g. a measured pattern read with [qdant_read](antennas-transducers-and-patterns.md#qdant_read)
- Without `center_freq`, the array applies to all bands; with `center_freq`, it is the antenna of that band
- Stores the array, which [layout_csv_write](#layout_csv_write) exports to a QDANT file

### Usage:
```matlab
layout_out = quadriga_lib.set_antenna_array( layout_in, array, rx_side, i_node, center_freq );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read); its antenna lists must fit its nodes and frequencies
- **`array`** — Valid single-frequency array, 1x1 struct, fields as in [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna)
- **`rx_side`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_array( layout, quadriga_lib.qdant_read( 'handset.qdant' ), true );
```

### Notes:
- **Bands and antenna lists**: as in [set_antenna](#set_antenna)
- This is the second form of the C++ function `set_antenna`, which takes an array; it has its own name in MATLAB,
  because MATLAB does not overload by type. The C++ in/out argument `layout` is split into `layout_in` / `layout_out`,
  so that the input stays unchanged

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads an array from a QDANT file)
- [layout_csv_write](#layout_csv_write) (exports the array)

---
<a id="set_antenna_qdant"></a>
## set_antenna_qdant
Set the antenna of layout nodes to an entry of a QDANT file

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_antenna_qdant) · [Python](../python/network-layouts-and-mobility.md#set_antenna_qdant)

- Sets the antenna of Tx nodes or Rx nodes to an array stored in a QDANT file, e.g. a measured or simulated pattern
- The layout refers to the file: [layout_csv_write](#layout_csv_write) writes the reference (the columns `antenna` = `qdant`,
  `qdant_file`, `qdant_id`), relative to the CSV file, and exports no copy of the array
- The array is used as stored, for all bands; the rules for bands and the antenna lists are those of [set_antenna](#set_antenna)

### Usage:
```matlab
layout_out = quadriga_lib.set_antenna_qdant( layout_in, qdant_file, qdant_id, rx_side, i_node, center_freq, evaluate );
```

### Inputs:
- **`layout_in`** — Layout struct, fields in [layout_csv_read](#layout_csv_read); its antenna lists must fit its nodes and frequencies
- **`qdant_file`** — Existing QDANT file, absolute or relative to the current folder
- **`qdant_id`** — 1-based entry of the file; default: 1
- **`rx_side`** — `false`: set the antenna of Tx nodes (`layout_in.tx`); `true`: of Rx nodes (`layout_in.rx`); default:
  `false`
- **`i_node`** — Indices of the Tx nodes or Rx nodes chosen by `rx_side`, 1-based; `[]` = all of them; default: `[]`
- **`center_freq`** — Band in [Hz], positive; NaN = one antenna for all bands; default: NaN
- **`evaluate`** — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none; default:
  `true`

### Outputs:
- **`layout_out`** — The layout with the new antenna lists and `i_antenna`

### Example:
```matlab
layout = quadriga_lib.layout_csv_read( 'tx.csv', 'rx.csv' );
layout = quadriga_lib.set_antenna_qdant( layout, 'measured/handset.qdant', 2, true );
```

### Notes:
- **Path**: the file is stored as a normalised absolute path
- **Errors**: a missing or unreadable file raises an error; without evaluation, an entry that the file does not have
  raises an error when the antennas are evaluated
- The C++ in/out argument `layout` is split into `layout_in` / `layout_out`, so that the input stays unchanged

### See also:
- [set_antenna](#set_antenna) (sets the antenna from the arguments of generate_antenna)
- [set_antenna_array](#set_antenna_array) (sets an array)
- [qdant_read](antennas-transducers-and-patterns.md#qdant_read) (reads the entries of a QDANT file)

---
<a id="set_orientation"></a>
## set_orientation
Set the orientation of nodes from their motion, or to a given value

Also in: [C++](../cpp/network-layouts-and-mobility.md#set_orientation) · [Python](../python/network-layouts-and-mobility.md#set_orientation)

- Turns the antennas of tracks along their motion, e.g. for a UT held in the direction of travel, or sets all rows of
  the nodes to a fixed orientation
- `force = false`: tracks get bank 0 and the tilt and heading of the steps between their rows, plus `orientation` as an
  offset; point drops keep their orientation plus the offset
- `force = true`: every row of the nodes gets exactly `orientation`

### Usage:
```matlab
nodes_out = quadriga_lib.set_orientation( nodes_in, orientation, i_node, force );
```

### Inputs:
- **`nodes_in`** — The Tx nodes or the Rx nodes of a layout (`layout.tx` or `layout.rx`); fields in [layout_csv_read](#layout_csv_read)
- **`orientation`** — Bank, tilt, heading in [rad], 3 finite values; `[]` = 0; default: `[]`
- **`i_node`** — Node indices, 1-based; `[]` = all nodes; default: `[]`
- **`force`** — `false`: orientation from the motion of tracks, plus `orientation` as offset; `true`: `orientation` on
  all rows; default: `false`

### Outputs:
- **`nodes_out`** — The nodes with the new orientations

### Example:
```matlab
layout.rx = quadriga_lib.set_orientation( layout.rx );                    % The antennas look along the motion
layout.rx = quadriga_lib.set_orientation( layout.rx, [0 0 0.5] );         % The same, turned by 0.5 rad
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
- The C++ in/out argument `nodes` is split into `nodes_in` / `nodes_out`, so that the input stays unchanged

### See also:
- [generate_track](#generate_track) (creates tracks with this orientation)
- [sample_track](#sample_track) (applies the motion to mounted antennas)
