# Layout system: implementation plan

This document specifies a layout system for quadriga-lib: data structures for BS/AP and UT/STA positions, orientations, antennas, scenarios and mobility; a CSV format; layout generators; and an orchestration of the GBSM functions from a layout to channel coefficients. It records every decision an implementation session needs. Sections 2 to 9 are the contract. Section 10 maps the QuaDRiGa-NG functions, section 11 lists the work packages and section 12 the tests.

The conventions of `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` apply. Sources, headers and tests do not cite this document (SPEC rule); they may cite the TR and other published material.

## 1. Scope

### 1.1 Goals

- One layout class and a set of stand-alone functions that combine the functionality of QuaDRiGa-NG's `qd_layout` and `qd_track`. A later NG integration replaces much of NG's implementation by library calls without changing NG's public API.
- All TR 38.901 clause 7.2 layouts and the clause 7.8 calibration layouts, including the speed and direction that drive short-term mobility, with wrap-around.
- Further generators with published definitions: ITU-R M.2412, TR 37.885 V2X road layouts, and track generators.
- A one-call path from layout to channel coefficients, and a staged manual path where GBSM data can be edited.
- Export of GBSM rays to QRT files, so that ray generation and antenna evaluation are separate.
- Holding positions and configurations for ray-tracing runs. The layout system does not call a ray tracer.
- One CSV file for the BS/AP side and one for the UT/STA side, with optional columns in any order and a central column table.

### 1.2 Out of scope

- Aerial, HST, NTN and indoor-room layouts: their defining TRs (36.777, 38.802, 38.811/38.821, 38.808) are not part of this work. The class and generator interface must be able to take them later (section 7.6).
- ISAC targets (TR clause 7.9): they need a channel model the library does not have.
- Frequency-dependent antenna models and true wideband arrays (section 5.3).
- Acoustics: `generate_speaker` and loudspeaker layouts (ITU-R BS.2051) are not connected in this version, because the speaker model will be reworked. Nothing in the format blocks them: speakers are Tx nodes, and a new preset is a table row.
- A one-function TR shortcut such as `get_channels_38901`: considered after the layout system has landed.
- Full 3GPP calibration: a separate later session, which may file issues against this implementation. Its metrics (coupling loss and geometry or SIR of the serving link, serving-link DS, ASD, ASA, ZSD, ZSA with the circular spread of Annex A.1, singular-value ratios, RSRP) belong to that session; this project provides attachment, coupling gain and channels.
- Channel interpolation: coefficients are computed at the requested positions. The `times` input (section 8.7) can feed an interpolation later.
- Traffic simulation, turning at V2X intersections, geographic coordinates, KML, plotting.
- Compatibility with the current QRT CSV format. Its columns are a source of ideas only, and no legacy detection is added.

### 1.3 Decision log

| ID | Decision |
|---|---|
| D1 | `name` is the site or terminal. An optional integer `sector` column separates sectors of one site. Rows with equal `name` and `sector` are a track. |
| D2 | Orientation is the antenna orientation. An optional direction is the movement direction. Direction omitted or NaN: direction = orientation (full vector, heading and tilt). |
| D3 | Segments and channel merging are in scope. A scenario change along a track starts a segment. |
| D4 | First implementation: all generators with available sources; the design holds later scenarios. |
| D5 | One CSV column per generator argument, with the C++ argument names. |
| D6 | One array per band is in scope (section 4.5). Deferred: one physical wideband array over several bands, until frequency-dependent models exist. |
| D7 | One CIR per layout row in QRT export. TR methods export a single frequency; multi-frequency export only for `quadriga`. |
| D8 | One `layout` class with two flat tables (`tx`, `rx`). Unknown CSV columns are kept and reported. |
| D9 | `scenario` holds a family or an exact conf name, on both sides; the UT value wins. LOS and indoor states are drawn from the seed; fixed by columns, exact names or by editing the link table `layout.links`. |
| D10 | Three public pipeline steps and one call that runs them. No builder object: discrete link states in `layout.links`, rays as outputs, SOS phases not stored, random values addressed by seed, key and position. |
| D11 | Wrap-around by lattice vectors and a virtual Tx position per propagation link. |
| R1 | The image is the one with the shortest 2D distance, per propagation link. |
| R2 | The image is chosen once at the drop and kept along the track. |
| R3 | The image index enters the seed: each copy is a separate site (TR clause 7.5). |
| U | Angles are degrees in the CSV and radians in the class. |
| D12 | Empty cells follow three column classes (section 4.3). |
| D13 | Optional `times` vector on the channel stage, the same for all terminals. |
| D14 | `attach_terminals` with the large-scale rule and the RSRP rule. |
| D15 | Merge gate: unit tests against TR oracles. No calibration in this project. |
| D16 | Names: class `layout`, header `quadriga_layout.hpp`, Python submodule `layout`. Functions have no `layout_` prefix, except `layout_csv_read` and `layout_csv_write`. |
| D17 | Shared CSV code is the header-only unit `src/qd_csv.hpp`, next to `src/qd_hdf5.hpp`. The material reader moves onto it. |
| D18 | `generate_layout` is one dispatcher with grouped tuple arguments and an `options` map. |
| D19 | `n_ut` is per cell by default; `per_cell = false` makes it the total. |
| D20 | Orientation names stay bank, tilt, heading = TR (γ, β, α) = (slant, downtilt, bearing). The Glossary line is corrected; aviation pitch = −tilt. |
| D21 | Function argument of type `layout` is named `layout`. |
| D22 | Bands: rows without position add per-band antennas; the Tx side defines the frequency list; a node without `center_freq` inherits all bands (section 4.5). |
| D23 | A change of `indoor` along a track starts a segment, like a change of `scenario` (section 8.5). |
| D24 | The frequency list is always stored in the CSV (section 4.5). |
| D25 | In Python, `layout.antenna` is a list of lists of 3D arrayant dicts, a deliberate deviation from the 4D multi-frequency dict, which cannot hold per-band geometry (section 9). |
| D26 | TR drops are point drops: one row with orientation, speed and direction; `generate_track` has a `point` type, and random TR orientations come from the `ut` tuple. |
| D27 | Antennas without generator text are exported to a QDANT file on write; the CSV type `qdant` with `qdant_file` and `qdant_id` reads them. |
| D28 | Seeds (`seed` per site, `layout_seed`), `pairing` and the O2I columns are stored in the CSV. |
| D29 | `relative = 1` gives an orientation relative to the direction of motion (mounted antennas). |
| D30 | `ut` tuple holds bank, tilt and heading; `drop_terminals` takes `bounds` as an argument. |
| D31 | Stale links are detected by a hash (6 ms per 100,000 rows); `layout_links::reset` clears them. |
| D32 | The conf lookup of issue #15 is used; wrappers pass their shipped folder. |
| D33 | Names: `export_rays_qrt`, `crossfade_segments`. Doc sections: everything declared in `quadriga_layout.hpp` in "Network layouts and mobility"; `crossfade_segments` in "Channel functions". |
| D34 | `set_antenna` takes `generate_antenna` arguments as a name-value map; `generate_antenna` gets no text outputs. |

### 1.4 Terms

- **Site:** all Tx rows with one `name`. Sectors of a site share position and rays.
- **Node:** one (`name`, `sector`) pair on one side. It has one antenna and one track.
- **Terminal:** all Rx rows with one `name`. A terminal normally has one node; a multi-panel terminal has several, which share their positions.
- **Row:** one position of one node.
- **Propagation link:** one site, one terminal, one segment. It has one LOS state, one set of LSPs and one set of clusters and rays. This is `n_link` of the GBSM functions.
- **Channel link:** one Tx node and one Rx node. It applies their antennas to the rays of its propagation link(s) and gives one `channel` per frequency.
- **Segment:** a part of a track with one resolved scenario.
- **Family:** a scenario name without a state, e.g. `UMa`. **Conf name:** the name of one conf file without extension and without the prefix `3GPP_38.901_v19_`, or a full file stem.
- **Image:** one of the wrap-around copies of a site, index 0 = the real site.

## 2. Functions and files

### 2.1 Public functions

A user works with one class and these functions. Function names describe what the function does and carry no `layout_` prefix, as in the antenna section; only the CSV functions name both layout and CSV. `generate_*` functions are object factories.

Doc sections, for now: everything declared in `include/quadriga_layout.hpp`, i.e. every function that takes a `layout`, goes to the new section **"Network layouts and mobility"**. `crossfade_segments`, which works on channels only and is declared in `include/quadriga_channel.hpp`, goes to the existing **"Channel functions"**. The maintainer may reorganise the sections later.

**Layouts and mobility:**

| Function | What it does | Why it exists |
|---|---|---|
| `layout_csv_read` | Reads the Tx and Rx CSV files and their QDANT files into a `layout`, and evaluates the antennas (optional) | Load a stored layout |
| `layout_csv_write` | Writes a `layout` to the two CSV files, and antennas without a generator description to a QDANT file | Store or exchange a layout in any state, also for ray tracing |
| `generate_layout` | Creates a complete layout from a preset such as `38.901_UMa` | The standard scenarios in one call |
| `drop_terminals` | Adds randomly placed UTs to a layout | Hand-placed BS with a standard UT drop |
| `generate_track` | Gives nodes a point drop with speed and direction, or a linear, circular or street track | Mobility without writing positions by hand |
| `sample_track` | Returns positions and orientations of one node at the snapshot times | Turns rows, speed and direction into snapshots; used by the channel functions and NG |
| `set_orientation` | Sets orientations from the motion, or to a fixed value | Antennas that follow a track or point one way |
| `set_antenna` | Assigns an antenna to nodes, by `generate_antenna` arguments or as an evaluated array | Antennas for generated or hand-built layouts, also from measurements |
| `set_antenna_qdant` | Assigns an entry of a QDANT file to nodes, stored as a `qdant` line (added in the implementation) | Measured patterns that the CSV files reference instead of copying |

**Channel generation from a layout** (same section; `crossfade_segments` in "Channel functions"):

| Function | What it does | Why it exists |
|---|---|---|
| `get_channels_layout` | **One call from layout to channel coefficients** | The normal path |
| `init_link_states` | Stage 1: draws LOS, indoor and scenario per link into `layout.links` | Inspect or force link states |
| `generate_rays` | Stage 2: computes `gbsm_rays` for chosen links | Inspect or edit rays; bounded memory |
| `get_gbsm_channels` | Stage 3: applies antennas, motion and link gain to rays | Channels from edited links or rays |
| `attach_terminals` | Finds the serving cell of each UT | Serving-link metrics and limiting the computed links |
| `export_rays_qrt` | Writes the GBSM rays of a layout to a QRT file | Separate ray generation from antenna evaluation |
| `crossfade_segments` | Joins the channels of consecutive track segments by a power cross-fade | Scenario changes along a track; used by stage 3 |

Public members: `layout::eval_antennas`, `layout_links::reset`, the `is_valid` members and the size getters.

### 2.2 Internal helpers

These are `static` in their source file and not part of the API: the CSV column table; antenna evaluation of one parameter line; hexagonal site and lattice construction; the preset tables; cell sampling; scenario family resolution; the seed hash; nearest wrap image; O2I and LOS state draws; the link-gain evaluation; grouping of links by site, image, conf and seed. Helpers that several source files need (the column table, the seed hash, the family table and the O2I field draws) are declared in the internal header `src/qd_layout_internal.hpp`, which is not named `qd_helpers_*` and is included only by the layout sources. WP2 adds the column table; WP5 adds the seed hash, the family table and the O2I draws, which `generate_layout` needs for the drop; WP7 reuses them.

### 2.3 Files

| File | Content |
|---|---|
| `include/quadriga_layout.hpp` | Classes and declarations; includes `quadriga_gbsm.hpp` only. Added to `quadriga_lib.hpp`. |
| `src/qd_csv.hpp` | Header-only CSV reader and writer. The name must not match `qd_helpers_*`: CMake makes those headers dependencies of every MEX file and the Python module. Included only by the files that parse CSV. |
| `src/qd_layout.cpp` | Class members, validation, column table, antenna evaluation, `set_antenna`. |
| `src/layout_csv.cpp` | `layout_csv_read`, `layout_csv_write`. |
| `src/generate_layout.cpp` | `generate_layout`, `drop_terminals`, preset tables. |
| `src/generate_track.cpp` | `generate_track`, `sample_track`, `set_orientation`. |
| `src/get_channels_layout.cpp` | `layout_links` members, `init_link_states`, `generate_rays`, `get_gbsm_channels`, `get_channels_layout`, `attach_terminals`, `export_rays_qrt`. May be split if it exceeds about 2000 lines. |
| `src/crossfade_segments.cpp` (section "Channel functions") | `crossfade_segments`, declared in `include/quadriga_channel.hpp`. The pipeline functions are declared in `include/quadriga_layout.hpp`, under their own topic group. |
| `api_mex/<function>.cpp` | One MEX file per function. Shared struct conversions in `api_mex/qd_mex_layout.hpp`, included only by these files. |
| `api_python/python_layout.cpp`, `api_python/qpy_layout*.cpp` | New submodule `layout`; add the dispatcher to the explicit list in `CMakeLists.txt` and to `python_main.cpp`. |
| `tests/catch2_tests/test_<function>.cpp`, `test_qd_csv.cpp` | Catch2 suites. |
| `tests/fuzzing/targets/fz_layout_csv.cpp` | Fuzz target of the reader. |
| `docs_builder/content/api_intro_cpp.md` | Glossary entries for the new shared quantities. |

The doc block of each source file sets its section. Documentation text lives only in the `SECTION`, `SECTION_DESC` and `MD` blocks of the sources (and the Glossary in `docs_builder/content/`); `docs/` is generated by `make docs` and is never edited by hand. Wrapper helper headers `qd_helpers_*` must not include `quadriga_layout.hpp`, so that wrappers of other sections do not pay for it. Selectors (`i_node`, `i_link`) are `const uint_list &` (SPEC "Declarations", since #98); data arrays of the classes (`offset`, `sector`, `i_antenna`, `pairing`) stay Armadillo types. Function arguments of type `layout` are named `layout`; this compiles without warnings, and inside such a function the class is reached as `quadriga_lib::layout`. Angle units of function arguments: generator tuples and options typed by hand (`generate_layout`, `drop_terminals`, `generate_antenna` arguments in `set_antenna`) are in degrees, as in the antenna generators; values that go into class members unchanged (`orientation` of `set_orientation`, `direction` of `generate_track`) are in radians, as the members.

**Conf files.** Since issue #15 (commit 53396ee8), `conf_file_list` searches the shipped folder after the given folders and `conf_file_read` accepts a shipped scenario name. The C++ core knows the folder set at build time; the MEX files and the Python module find it at run time next to their own location, which also works for pip and moved installs. The layout functions therefore resolve scenario names with `conf_file_list(config_dirs)`, and their wrappers append the folder they found to `config_dirs` before the call. The shipped folder is found in the wrappers as follows. MEX: the ten-line `qd_mex_conf_dir` of `api_mex/conf_file_list.cpp` (it calls `qd_conf_shipped_dir` of the library) is copied into `api_mex/qd_mex_layout.hpp`, because each MEX file is its own shared object. Python: `qd_python_conf_dir` in `api_python/qpy_conf_file.cpp` loses its `static` and is forward-declared in `python_layout.cpp`. Only stage 1 resolves conf files: `init_link_states` stores the resolved paths in `layout.links.conf_files`, and all later steps read those, so they take no `config_dirs`. Only the wrappers of `init_link_states` and `get_channels_layout` append the shipped folder.

## 3. Data structures

Illustrative comments are shortened; the real header follows the SPEC comment format. The classes are non-templated and use `double`: positions at kilometre scale need it, and `gbsm_single_link_channel` is `double` only.

```cpp
namespace quadriga_lib
{
    // One side of a layout; flat arrays, nodes in order of first appearance
    class layout_nodes
    {
    public:
        std::vector<std::string> name;      // Site or terminal name, [n_node]
        arma::u32_vec sector;               // Sector index, [n_node]
        arma::u32_vec i_antenna;            // 1-based index into the antenna lists, 0 = generic xpol, [n_node]
        arma::uvec offset;                  // First row of each node, 0-based, [n_node]
        arma::mat pos;                      // Positions in [m], [3, n_row]
        arma::mat orientation;              // Bank, tilt, heading in [rad], [3, n_row]
        arma::mat direction;                // Azimuth, elevation of travel in [rad], [2, n_row]; NaN = orientation; empty = all NaN
        arma::vec speed;                    // Speed in [m/s], applies from this row on, [n_row]; empty = 0
        arma::vec time;                     // Time in [s], [n_row]; NaN on all rows of a node = not given; empty = none
        std::vector<std::string> scenario;  // Family or conf name, "" = none, [n_row]; empty = none
        arma::s32_vec indoor;               // -1 = draw, 0 = outdoor, 1 = building (commercial), 2 = car, 3 = residential building, [n_row]; empty = all -1
        arma::s32_vec floor;                // Floor number from 1, -1 = derive from z, [n_row]; empty = all -1
        arma::vec d2d_in;                   // Indoor 2D distance in [m], NaN = draw, [n_row]; empty = all NaN
        arma::mat clutter;                  // d_clutter, r_clutter, h_clutter, NaN = model default, [3, n_node]; empty = all NaN
        std::unordered_map<std::string, arma::vec> scenpar_override; // Scenario parameter overrides by key, NaN = none, each [n_node]
        arma::s64_vec seed;                 // Site seed, -1 = derived from layout.seed, [n_node]; empty = all -1
        arma::u32_vec relative;             // 1 = orientation relative to the motion, [n_node]; empty = all 0
        std::vector<std::string> extra_names;        // Names of kept columns, [n_extra]
        std::vector<std::vector<std::string>> extra; // Their raw cells, [n_extra][n_row]

        arma::uword n_node() const;
        arma::uword n_row() const;
        arma::uword n_row(arma::uword i_node) const; // Rows of one node
        std::string is_valid() const;                // Empty if valid
    };

    // Discrete state of the propagation links of one realisation (section 8.4)
    class layout_links
    {
    public:
        arma::u32_vec i_tx, i_rx, i_segment;   // Site, terminal, segment, 0-based, [n_link]
        arma::u32_vec image, i_conf;           // Wrap image; index into conf_names, [n_link]
        std::vector<std::string> conf_names;   // Resolved conf names, [n_conf]
        arma::u32_vec los;                     // 1 = LOS, [n_link]
        arma::s32_vec indoor;                  // O2I state at the segment start, [n_link]
        arma::u32_vec floor_id;                // floor - 1, outdoor 0, [n_link]
        arma::vec d2d_in;                      // Indoor 2D distance in [m], [n_link]
        arma::mat o2i_rnd;                     // Inputs of o2i_loss, [n_terminal, 4]
        arma::u32_vec o2i_model;               // Chosen O2I model, index into the family's list, [n_terminal]
        arma::s64_vec gbsm_seed;               // Seed of gbsm_init_phases, [n_link]
        arma::sword seed = -1;                 // Master seed
        std::string method, sos_preset;        // As used
        double anchor_freq = 2.0e9;            // Anchor frequency of 38.901_MF
        bool pol_variability = false;
        std::vector<std::string> conf_files;   // Resolved paths of conf_names, [n_conf]
        unsigned long long layout_hash = 0;    // Hash of all layout members that determine the links
        arma::uword n_link() const;
        bool empty() const;
        void reset();                          // Clears all members
        std::string is_valid() const;
    };

    class layout
    {
    public:
        std::string name;                   // Name of the layout
        layout_nodes tx, rx;                // BS/AP side and UT/STA side
        layout_links links;                 // Link states of the current realisation; empty until init_link_states
        arma::sword seed = -1;              // Layout seed; -1 = not set
        arma::vec center_freq;              // Simulation frequencies in [Hz], ascending, [n_freq]
        std::vector<std::vector<arrayant<double>>> antenna;  // Evaluated antennas, [n_antenna][1 or n_freq]; empty = not evaluated
        std::vector<std::string> antenna_cols;               // Names of the antenna columns, [n_col]
        std::vector<std::vector<std::string>> antenna_args;  // CSV lines of values aligned with antenna_cols, [n_antenna][1 or n_freq]
        arma::mat wrap;                     // Lattice vectors in [m], one per column, [2, 1] or [2, 2]; empty = none
        arma::umat pairing;                 // Optional [2, n_pair]: Tx site index, Rx terminal index, 0-based; empty = all pairs farther apart than 1e-4 m

        arma::uword n_antenna() const;      // Length of antenna_args, or of antenna if it has no args
        void eval_antennas(const arma::vec &center_freq = {}); // Generates 'antenna' from 'antenna_args'; a given frequency list replaces the member
        std::string is_valid() const;       // Empty if valid
    };
}
```

**Antennas.** There is no separate antenna parameter class. The layout holds the evaluated arrays and, next to them, the text they were generated from:

```cpp
layout.antenna_cols = {"center_freq", "antenna", "panel"};
layout.antenna_args = {{"3.5e9,3gpp,\"8,8,1,2\"", "28e9,3gpp,\"16,16,1,4\""}, // One array per band
                       {",dipole,"}};                                         // One line, evaluated for all bands
```

Equal text means the same antenna, so identical terminals share one entry. `layout_csv_write` writes these lines back. An antenna set as an evaluated array (section 5.4) has an empty inner list in `antenna_args`; `layout_csv_write` exports it to a QDANT file, so every layout can be written.

An inner entry of `antenna` is an independent array for its band: element count, ports and positions may differ between bands. `is_valid` of a `std::vector<arrayant>` is therefore not applied to it, and `get_channels_multifreq` is not used in this version.

**Node without antenna.** If `layout.antenna` is empty or `i_antenna` is 0, the channel functions use the generic `xpol` antenna: two co-located isotropic elements with V and H polarization, so that the 2 × 2 coefficient matrix is the polarization transfer matrix of each path.

Validation rules of `layout::is_valid` (each gives one message, in SPEC style):

- Sizes of all members; `offset(0) == 0`, strictly increasing, last entry below `n_row`; finite positions; finite orientations.
- (`name`, `sector`) pairs are unique; names are non-empty and at most 255 bytes (the QRT limit).
- All nodes of one site, and all nodes of one terminal, have equal positions row by row, with the same row count or one row (static). The nodes of one site have equal `scenario`, `clutter` and `scenpar_override`; the nodes of one terminal have equal `scenario`, `indoor`, `floor` and `d2d_in`.
- `i_antenna` is at most `n_antenna`. `antenna_args` is empty or has one entry per antenna, each with 0, 1 or `n_freq` lines; `antenna` is empty or has `n_antenna` entries with 1 or `n_freq` inner entries. `center_freq` is ascending, finite and positive. `direction` row 0 (azimuth) is NaN on every row of a node with more than one row (row 1 is then ignored). `floor` and `d2d_in` change only at a segment start.
- A node has either `time` or a non-zero `speed`, not both. `time` is all finite or all NaN per node and strictly increasing where finite; `speed` is finite and `>= 0`.
- A moving part of a track (consecutive rows at different positions) with `speed = 0` and no `time` is allowed only if the node has neither speed nor time at all: rows are then snapshots without a time axis.
- `scenario` contains no path separator (`/`, `\`). The scenario of a Tx site is constant along its rows.
- `wrap` is `[2, 1]` (one axis, e.g. a highway) or `[2, 2]` with linearly independent columns; `pairing` indices are in range and pairs are unique.
- `indoor` in {-1, 0, 1, 2, 3}; `floor` is -1 or >= 1; `d2d_in` is NaN or >= 0.

## 4. CSV format

### 4.1 Files

A layout is two files, one for the Tx side and one for the Rx side, with the same format. Either may be omitted (`""`) on read and write. Rows define nodes and track positions, columns define parameters. Columns are optional and in any order, except that `name`, `x`, `y`, `z` are required. Example:

```csv
# Three-sector site and a dipole site
name,sector,x,y,z,heading,tilt,center_freq,antenna,element,panel,spacing,pol,steer
BS1,0,0,0,25,30,10,3.5e9,3gpp,"65,65,30,30,8","4,4,1,2","0.5,0.5,2,2.5","3,2","6,0"
BS1,1,,,,150,,,,,,,,
BS1,2,,,,270,,,,,,,,
BS2,0,500,0,25,90,0,,dipole,,,,,
```

```csv
name,x,y,z,heading,speed,direction_az,scenario,antenna
UT1,10,20,1.5,45,0.833,120,UMa,dipole
UT2,0,0,1.5,0,,,,dipole
UT2,1,0,1.5,,,,,
UT2,2,0,1.5,90,,,,
UT2,2,1,1.5,,,,UMa_NLOS,
```

### 4.2 Syntax (`qd_csv.hpp`)

- Comma-separated, first non-comment line is the header. Header names are case-sensitive, trimmed and unique, otherwise the reader throws.
- Cells may be quoted with `"`, with `""` as an escaped quote; a quoted cell may contain commas and semicolons, not line breaks. Whitespace around unquoted cells is trimmed.
- Lines starting with `#` (after optional whitespace) and blank lines are skipped. LF and CRLF, and a UTF-8 BOM, are accepted.
- A row with fewer cells than the header is padded with empty cells. A row with more cells throws.
- Numbers are parsed locale-independently, the whole cell must be consumed. `NaN`, `Inf`, `-Inf` are accepted in any case. `std::from_chars` for `double` is not available (MATLAB R2024a libstdc++); move the strict number core of `qd_conf_parse_number` (`src/qd_scenpar.cpp`) into `qd_csv.hpp` with an `allow_nan` option. The conf reader keeps rejecting NaN and is an allowed includer of this header.
- List cells: entries separated by `,`. Matrix cells: rows separated by `;`, entries by `,`, so `"6,0;10,0"` is `[2, 2]`. For `steer`, cell rows are matrix rows; for `beams`, cell rows are matrix columns, as `generate_antenna` expects (`steer` `[n, 2]`, `beams` `[2 or 3, n_beams]`). A one-row value is a row for `steer` and a column for `beams`; the same holds for vectors in the argument map of `set_antenna`.
- The writer quotes a cell only when it contains a comma, semicolon, quote or leading or trailing whitespace. Numbers use up to 15 significant digits, or 16 or 17 where 15 do not read back to the same `double`, as `conf_file_write`. Angle columns are written with 15 significant digits: integer and half degrees then read back bit-identical, other values within 1e-14 rad (measured).
- Errors are `std::runtime_error` with file name, line and column name.
- Minimal interface: a table object with `header`, `cell(row, col)` as `std::string_view` into one owned buffer, the source line per row, `find(name)`, `get_double`, `get_list`, `get_matrix`, and a `write` function. Cells are offsets into the buffer, so that a file with several 100,000 rows does not allocate one string per cell.

The material reader in `src/obj_file.cpp` moves onto this unit with its behaviour kept: `name` column, empty cell = 0, non-finite values rejected, duplicate names rejected, long rows ignored and duplicate headers tolerated by options of the reader. Deliberate changes: quoting becomes available, a `#` line is no longer a row (no shipped material starts with `#`; check in the work package), and hexadecimal numbers, which `std::stod` accepted, are rejected.

### 4.3 Column classes and empty cells

| Class | Meaning of an empty cell | Rule |
|---|---|---|
| Identity (`name`, `sector`) | `name` required; `sector` empty = 0 | Define the node. |
| Fixed | Takes the node's first-row value; for an extra sector, the value of sector 0 of the site | Per band: a later non-empty value that differs from the first throws. Identical repeats are accepted. Band rows: section 4.5. |
| Per row (`x`, `y`, `z`, `time`) | `x`, `y`, `z`: throws, except for an extra sector of a static site, which has one row and takes the position of sector 0. `time`: allowed only if all rows of the node are empty (NaN) | A file may mix timed and untimed nodes. |
| Held | Repeats the previous row of the node; on the first row, the value of sector 0 of the site for an extra sector, otherwise the column default | `NaN` is a value: in `direction_az` it means "follow the orientation". |

A moving multi-sector site writes all rows for each sector with equal positions.

### 4.4 Column table

All columns are defined in one central table in `src/qd_layout.cpp`, with name, class, value kind, unit conversion, and the member or antenna argument it maps to. Adding a parameter means adding one table entry. Readers, writers and error messages are driven by this table.

| Column | Class | Kind | Unit in file | Member, meaning |
|---|---|---|---|---|
| `name` | Identity | text | | Site or terminal |
| `sector` | Identity | integer | | Sector index, default 0 |
| `x`, `y`, `z` | Per row | number | m | `pos` |
| `time` | Per row | number | s | `time` |
| `bank`, `tilt`, `heading` | Held | number | deg | `orientation`, default 0; positive tilt points down |
| `direction_az`, `direction_el` | Held | number | deg | `direction`; `direction_az` NaN or empty: follow the orientation; `direction_el` empty = 0, positive = up |
| `speed` | Held | number | m/s | `speed`, default 0 |
| `scenario` | Held | text | | Family or conf name |
| `indoor` | Held | integer | | -1 or empty = draw, 0 outdoor, 1 building (commercial), 2 car, 3 residential building |
| `floor` | Held | integer | | -1 or empty = derive from `z` |
| `d2d_in` | Held | number | m | NaN or empty = draw |
| `d_clutter`, `r_clutter`, `h_clutter` | Fixed, Tx side | number | m, 1, m | `clutter`, inputs of `los_probability`; empty = default of the model; equal for all sectors of a site |
| `scenpar.<key>` | Fixed, Tx side | number | as the key | `scenpar_override`: one scenario parameter for the links of that site, e.g. `scenpar.DS_VS`; equal for all sectors of a site; an unknown key throws at `init_link_states` |
| `wrap` | Fixed, Tx side | matrix | m | `"ax,ay;bx,by"` or `"ax,ay"`, one vector per cell row; all non-empty cells equal |
| `seed` | Fixed, Tx side | integer | | Seed of the site; empty = derived from the layout seed and the site name |
| `layout_seed` | Fixed, Tx side | integer | | `layout.seed`; all non-empty cells equal; the writer puts it on the first Tx row |
| `pairing` | Fixed, Rx side | list | | Names of the Tx sites linked to this terminal; empty = all (`layout.pairing`) |
| `relative` | Fixed | integer | | 1 = orientation relative to the direction of motion (section 6.1); default 0 |
| `antenna` | Fixed | text | | `generate_antenna` type, or `qdant` |
| `qdant_file`, `qdant_id` | Fixed | text, integer | | For `antenna = qdant`: QDANT file relative to the CSV file, and its 1-based entry (default 1) |
| `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `separate_beams`, `apply_weights`, `device_size`, `locations` | Fixed | list or matrix | as `generate_antenna` | Arguments of `generate_antenna` |

Further rules:

- Any other column is kept as text in `extra_names` and `extra`, written back unchanged, and reported through the optional output of `layout_csv_read`, so that a typo is visible.
- Pairing, seeds and the O2I columns are stored, so that a layout can be saved after attachment and reproduced from its files. Link states that the user forced by editing `layout.links` are not stored (D9).
- `layout::name` is not stored in the CSV; `layout_csv_read` leaves it empty.
- An antenna column that the type does not accept throws at read, naming the row and the type. The table holds the accepted arguments per generator tag.
- `center_freq` in a row defines a band (section 4.5). The simulation frequencies are the member `layout::center_freq`.

### 4.5 Bands

A **band row** adds a frequency, and optionally an antenna for it, to its node. It has `name`, optionally `sector`, a `center_freq`, and otherwise only antenna columns. Such rows are not track rows, so tracks stay small and the rule "one antenna per node" holds per band.

```csv
# Tx file: one array per band
name,x,y,z,center_freq,antenna,panel
BS1,0,0,25,3.5e9,3gpp,"8,8,1,2"
BS1,,,,28e9,3gpp,"16,16,1,4"

# Tx file: the same design in both bands, λ/2 in each
name,x,y,z,center_freq,antenna,panel
BS2,500,0,25,3.5e9,3gpp,"8,8,1,2"
BS2,,,,28e9,,

# Rx file: inherits all bands
name,x,y,z,antenna
UT1,10,20,1.5,dipole
```

**Row classification**, in this order:

1. The first row of a node. For an extra sector of a static site, `x`, `y`, `z` may be empty; its held values and its first band may be set.
2. A later row with empty `x`, `y`, `z` and a non-empty `center_freq` is a band row. Any column other than the identity columns, `center_freq` and antenna columns that is set on it throws, including unknown extra columns.
3. Any other row is a position row. Its `center_freq` is empty or equal to that of the node's first row.

**Band rules:**

1. The frequency list is the ascending set of `center_freq` values on the Tx side. If the Tx side has none, the Rx side defines it. An explicit `center_freq` argument of `layout_csv_read` replaces it, and values in the files must then match within a relative tolerance of 1e-6.
2. A node without `center_freq`, on either side, inherits all bands: one line, evaluated per band (section 5.3).
3. A node with explicit bands must define **all** of them, on the Rx side too. A missing or unknown band throws.
4. A band row with empty antenna columns repeats the antenna of the node's first band at this frequency. Such a node stores one line without frequency, so it is equal to a node that inherits.
5. The Fixed rule of section 4.3 applies per band. An extra sector without antenna cells inherits all bands of sector 0; with its own antenna cells it defines its own bands by the same rules.
6. Band rows may appear anywhere in the file. Lines of an antenna are sorted by band index.

**Writer.** If `layout.center_freq` is not carried by any antenna line, `layout_csv_write` declares it on the first Tx node: `center_freq` on its first row and one band row with empty antenna columns per further frequency. The frequency list is therefore always stored in the file.

Shared across bands (TR clause 7.6.5): positions, orientations, LOS state and LSP random values. Random draws always use the full frequency list, also for later band masks, because a shorter list changes the `quadriga` rays.

## 5. File and antenna functions

### 5.1 `layout_csv_read`

```cpp
layout layout_csv_read(
    const std::string &fn_tx,                       // Tx CSV file, "" = no Tx nodes
    const std::string &fn_rx,                       // Rx CSV file, "" = no Rx nodes
    const arma::vec &center_freq = {},              // Simulation frequencies in [Hz]; empty = from the files (section 4.5)
    bool eval_antennas = true,                      // false: antennas are not generated, the channel functions use the generic xpol
    std::vector<std::string> *extra_tx = nullptr,   // Optional output: names of unknown Tx columns
    std::vector<std::string> *extra_rx = nullptr);  // Optional output: names of unknown Rx columns
```

Nodes are created in order of first appearance. Position rows of a node need not be consecutive in the file, and their file order is the track order. The antenna columns of each node are composed into `antenna_args`; equal entries are shared, Tx side first, then Rx side, so the antenna order is deterministic. A relative `qdant_file` path is resolved against the folder of the CSV file that names it and stored as a normalised absolute path, so that equal text in two folders stays two antennas.

With `eval_antennas = false`, `layout.antenna` stays empty, while `antenna_args` and `i_antenna` are kept. The channel functions then use `xpol`, and a later `layout.eval_antennas()` or `layout_csv_write` still has the full description. `i_antenna` is not reset to 0 here, because the row would otherwise lose its antenna on a read-write round trip.

### 5.2 `layout_csv_write`

```cpp
void layout_csv_write(
    const std::string &fn_tx,          // Tx CSV file, "" = not written
    const std::string &fn_rx,          // Rx CSV file, "" = not written
    const layout &layout,              // Layout
    bool compact = true);              // true: empty cells where the class allows them; false: every cell written
```

- Only columns with non-default content are written. Columns follow the order of the table, extra columns last.
- An antenna band that is an evaluated array without text (set by the array form, section 5.4) is written to a QDANT file of its antenna. The file is named after the CSV file being written that first references the antenna: that CSV's name without `.csv`, plus `_<k>.qdant`, with a running index `k` from 1 per CSV file (so Rx-only antennas take the stem of `fn_rx`). A multi-frequency antenna needs a whole file, so each antenna gets its own file, with one entry per exported band (the vector form of `qdant_write`). In the CSV, each exported band becomes a `qdant` line with `qdant_file` and `qdant_id` = the band's entry, written as band rows (section 4.5) when the antenna has several bands; a band given by generator text keeps its text line.
- After the new files are in place, the writer deletes the files matching `<stem>_<k>.qdant` of each written CSV that no `qdant_file` cell of the new CSV files references. Files of any other name are never touched. The documentation states this. A read-write-read cycle to the same file names therefore keeps the QDANT files that are still referenced. A layout with such antennas but not evaluated (`layout.antenna` empty) throws, because there is nothing to write.
- All files are composed first and written afterwards. Existing files are replaced; a failed call leaves them unchanged (write to temporary files in the same folder and rename). QDANT files are written only if needed.
- `qdant_file` paths are written relative to the folder of the target CSV.
- Read-write-read gives equal objects within the number format of section 4.2, with antennas from the QDANT file then described by `qdant` lines, and except for `name` and `links`, which are not stored. `pairing`, seeds and the O2I columns are stored (section 4.4).

### 5.3 `layout::eval_antennas`

Generates `layout.antenna` from `antenna_args`, with `generate_antenna` or `qdant_read`, in parallel over the antennas where it pays (SPEC OpenMP rules, `qd_batch_loop`). An entry with an empty inner list (an evaluated array without text) is kept as it is. Entries that are already evaluated stay unchanged unless a new frequency list is given. An antenna entry with 0 inner arrays is valid and means not evaluated; its nodes use `xpol`.

- An antenna with one line per band is generated once per band at that band's frequency.
- An antenna with one line and no `center_freq` is generated once per entry of `layout.center_freq`, so a λ/2 array stays λ/2 in each band. Its element positions in metres differ per band.
- If the result has co-located elements (`omni`, `xpol`, `dipole`, ...), one inner entry is stored for all bands.
- An array whose `spacing` is given in wavelengths with no frequency anywhere is deferred by `set_antenna` (its nodes use `xpol` until a frequency is set); `eval_antennas` and the channel functions throw for it, because its size in metres is undefined. This does not apply to `3GPP-UT` and `3GPP-CPE`, whose `device_size` is in metres.
- A `qdant` line reads entry `qdant_id` (default 1) of `qdant_file`, used as stored and not rescaled. One line applies to all bands; per-band entries use band rows.
- Within an antenna, bands may mix text lines and evaluated arrays: an empty line string at a band means "this band is an evaluated array". `eval_antennas` keeps such bands, and the writer exports them.
- A true wideband array, one physical array over several bands, stays open until frequency-dependent antenna models land (D6). It will get its own function (`set_antenna_mf`), as speakers will (`set_speaker`); the inner vector per antenna already holds one entry per frequency for them.
- The channel functions take inner entry `i_freq` if there are `n_freq` entries and entry 0 if there is one; any other size throws.

### 5.4 `set_antenna`

```cpp
// From generate_antenna arguments; the layout stores them as CSV text
void set_antenna(
    layout &layout,
    const std::string &type,                                       // Antenna type, as generate_antenna
    const std::unordered_map<std::string, arma::mat> &args = {},   // generate_antenna arguments by name, e.g. {"panel", {4, 4, 1, 2}}; flags as 0 or 1
    bool rx_side = false,                                          // false: Tx side, true: Rx side
    const uint_list &i_node = {},                                  // Node indices, 0-based; empty = all nodes of that side
    double center_freq = NAN,                                      // Band in [Hz]; NaN = one antenna for all bands (section 4.5)
    bool evaluate = true);                                         // true: generate all antennas not yet evaluated

// From an evaluated array, e.g. a measurement or a CST export read with qdant_read
void set_antenna(
    layout &layout,
    const arrayant<double> &array,          // Single-frequency array
    bool rx_side = false,
    const uint_list &i_node = {},
    double center_freq = NAN);              // Band in [Hz]; NaN = the same array for all bands
```

- The first form takes the arguments of `generate_antenna` by name, and its documentation refers to `generate_antenna` for their meaning. Names are checked against the central column table, which is also the CSV column definition, so a new `generate_antenna` argument is one table entry and no signature change. An unknown name, or one the type does not use, throws. Values are numbers or matrices (rows of `steer` and `beams` as in `generate_antenna`), so no CSV quoting appears in user code. `res` and `center_freq` of `generate_antenna` are also accepted in the map; `center_freq` there and the argument must not both be given. `pattern` is not available here; an array built from a pattern uses the second form.
- `generate_antenna` itself is unchanged; it gets no text outputs.
- The second form stores the array with an empty text entry; `layout_csv_write` exports it to the QDANT file (section 5.2).
- Added in the implementation: `set_antenna_qdant(layout, qdant_file, qdant_id = 1, rx_side, i_node, center_freq, evaluate)` stores a `qdant` line with the normalised absolute path, with the band and copy-on-write rules of the first form; `layout_csv_write` writes it as a reference relative to the CSV file and exports no copy. A missing or unreadable file throws `std::runtime_error`.
- With a `center_freq`, the call sets one band of the nodes' antenna. A node gets a new antenna entry for this (copy on write), so other nodes that shared the old entry are not affected. Until all bands are set, `is_valid` reports the node as incomplete. Frequencies not in `layout.center_freq` are added to it. A per-band call on a node whose single line covers all bands first turns that line into explicit lines for all bands, then replaces band `center_freq`.
- Arguments that the type does not use throw, as in `generate_antenna`. With `evaluate = false`, or after `layout_csv_read` with `eval_antennas = false`, the channel functions use `xpol` until `eval_antennas` is called.

## 6. Mobility

### 6.1 Direction and orientation

- Both are given in the GCS. The orientation is the antenna orientation (bank, tilt, heading) = TR (γ, β, α) = (slant, downtilt, bearing), with a positive tilt pointing down. Aviation pitch is −tilt, and aviation yaw runs clockwise from north.
- A one-row node is a **point drop**: it stays at its position, and with a non-zero `speed` it moves along `direction` during the snapshot times (TR clause 7.5 step 1f). This is how all TR presets are generated: drop position, orientation and velocity, no track. If `direction` is NaN, the direction is the orientation's forward axis: `d = [cos(tilt) cos(heading), cos(tilt) sin(heading), -sin(tilt)]`. Bank has no effect.
- A multi-row node is a **track**. It moves along its positions, and `direction` on such a node throws.
- TR generators write an explicit horizontal direction and a random orientation per the TR (heading uniform, tilt and bank from the preset), so that a tilted handheld does not move into the ground.
- An absent orientation stays 0. It is not derived from the motion implicitly. `set_orientation` derives it on request for tracks; `generate_track` calls it.
- **Mounted antennas** (column `relative`, section 4.4): with `relative = 1`, the stored orientation is relative to the direction of motion, like an antenna mounted on a car body. The orientation used at a snapshot is `R(motion) · R(orientation)`, where `R(motion)` has heading and tilt of the direction of travel and no bank. A point drop uses its `direction`; with `relative = 1` and `direction` NaN the node throws, because its direction would otherwise come from the orientation it defines. On a track, `R(motion)` at a snapshot uses the direction of the row segment that holds it (forward difference, the last segment repeated at the end), as `set_orientation` does. `set_orientation` with `force = false` and `generate_track` leave the orientation of `relative = 1` nodes unchanged, so the motion is not applied twice.

### 6.2 Rows and snapshots, `sample_track`

A **row** is a point stored in the layout: a drop position, or a point of a track with its time or speed. A **snapshot** is one sample of the channel in time and space, produced by `sample_track` for the channel functions. Without `times`, each row gives one snapshot, and the two coincide. With `times`, a point drop gives as many snapshots as times, and a track is interpolated, so rows and snapshots differ.

```cpp
void sample_track(
    const layout_nodes &nodes,
    arma::uword i_node,
    const arma::vec &times,           // Snapshot times in [s], [n_snap]; empty = one snapshot per row
    arma::mat &pos,                   // Output: positions in [m], [3, n_snap]
    arma::mat &orientation,           // Output: orientations in [rad] in the GCS, incl. the mounting rotation, [3, n_snap]
    arma::uvec *i_row = nullptr);     // Optional output: row at or before each snapshot, [n_snap]
```

| Node | `times` empty | `times` given |
|---|---|---|
| Point drop, no speed | One snapshot | The position at all times |
| Point drop with speed | One snapshot at `p0` | `p0 + v d t`, constant orientation |
| Track rows with `time` | One snapshot per row | Linear position, SLERP orientation; before the first and after the last time the end value is held |
| Track rows with `speed` | One snapshot per row | Time from arc length, `t(0) = 0`, `t(i+1) = t(i) + |Δp| / speed(i)`; a moving step with `speed = 0` throws; otherwise as above |
| Track rows without either | One snapshot per row | Throws, unless the node has one row |

`times` is the same for all nodes. A track that ends earlier holds its last position. Orientation interpolation uses unit quaternions or `fast_slerp` along each angle as NG does; the implementation chooses the shorter code and tests a 90° turn.

### 6.3 `generate_track`

```cpp
void generate_track(
    layout_nodes &nodes,
    const uint_list &i_node,              // Nodes, 0-based; each keeps its first position
    const std::string &type,              // "point", "linear", "circular", "street"
    double length = 0.0,                  // Length in [m]; "point": unused
    double speed = NAN,                   // Speed in [m/s]; NaN = keep an existing valid speed
    const arma::vec &direction = {},      // Start direction (azimuth) in [rad], [1] or [n]; empty = uniform random
    double spacing = NAN,                 // Point spacing in [m]; NaN = type default
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    arma::sword seed = -1);
```

- `point`: the nodes become point drops with `speed` and an explicit horizontal `direction` (random if not given). The rows after the first are removed, and `time` is cleared. This is the TR form of short-term mobility.
- `linear` (two points, or `length / spacing` points), `circular` (default 129 points, closed; the sign of the direction sets the rotation), `street` (options `street_length_min = 50`, `street_length_mu = 187`, `street_length_std = 83`, `curve_radius = 10`, `turn_probability = 0.5`, default spacing 1 m) follow NG's `qd_track.generate`. They clear `direction` and call `set_orientation` with `force = false`.
- The call applies to all nodes of a terminal or site.

### 6.4 `set_orientation`

```cpp
void set_orientation(
    layout_nodes &nodes,
    const arma::vec &orientation = {},  // [bank, tilt, heading] in [rad]; empty = 0
    const uint_list &i_node = {},       // Nodes, 0-based; empty = all
    bool force = false);                // false: from the motion of tracks, plus 'orientation' as offset; true: 'orientation' at all rows
```

- `force = false`, tracks: bank 0, tilt and heading from forward differences, the last value repeated (or the first for a closed track), as `qd_track.calc_orientation`, and `orientation` is added as an offset. A step of zero length keeps the previous orientation. A positive tilt points down, so a rising track has negative tilt.
- `force = false`, point drops: their orientation stays and receives the offset. Random TR orientations of point drops come from `generate_layout` and `drop_terminals` (the `ut` tuple), not from this function.
- `force = true`: every row of the nodes gets exactly `orientation`.
- For mounted antennas, `relative = 1` is usually the simpler way: the orientation is then given once relative to the vehicle and follows the motion at every snapshot.

## 7. Generators

### 7.1 `generate_layout`

```cpp
layout generate_layout(
    const std::string &type,        // Preset, section 7.5
    arma::uword n_ut = 0,           // UTs per cell (per_cell) or in total; 0 = no UTs
    const arma::vec &spacing = {},  // [d_x, d_y] in m; hexagonal: d_x = ISD; NaN = preset value
    const arma::vec &grid = {},     // [n_x, n_y, n_sector]; hexagonal: n_x = rings (0 = one site), n_y unused; NaN = preset value
    const arma::vec &area = {},     // [length, width, height] of the hall, floor or road section in m; NaN = preset value
    const arma::vec &bs = {},       // [height, tilt, sector azimuth offset] in m, deg, deg; NaN = preset value
    const arma::vec &ut = {},       // [height, min 2D distance, speed, bank, tilt, heading] in m, m, m/s, deg, deg, deg; NaN = preset rule (heading: uniform random)
    const arma::vec &indoor = {},   // [indoor ratio, min floors, max floors, floor height in m]; NaN = preset value
    const std::unordered_map<std::string, std::vector<double>> &options = {}, // Scenario-specific values
    bool per_cell = true,           // true: n_ut per cell; false: n_ut in total
    arma::sword seed = -1);         // Seed of the drop; -1 = system random device; stored as layout.seed
```

Rules:

- Tuples may be shorter than documented. An entry `NaN` takes the preset value; integer entries such as those of `grid` must be whole numbers. A value that the preset does not use throws, as `generate_antenna` does.
- `options` keys are listed per preset; an unknown key throws.
- Extension rules: a value used by several families is appended to its tuple; a value used by one family goes into `options`; a new argument only for a group shared by several families, appended at the end; a new preset is one row in the central table plus a builder if its geometry is new.
- Tx site names are `BS1`, `BS2`, ...; UT names `UT1`, `UT2`, ... The output has no antennas, so channel functions use `xpol` until `set_antenna` assigns one. `layout.center_freq` stays empty; the user sets it.
- The drop is deterministic for a given seed and independent of the thread count.

### 7.2 `drop_terminals`

```cpp
void drop_terminals(
    layout &layout,                  // Tx side defines the cells; Rx side is replaced
    arma::uword n_ut,
    const arma::vec &bounds = {},    // [x_min, x_max, y_min, y_max] in m; required without wrap vectors
    const arma::vec &ut = {},        // As generate_layout
    const arma::vec &indoor = {},
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    bool per_cell = true,
    arma::sword seed = -1);
```

It applies the drop of section 7.4 to any Tx side. Cells are the Voronoi regions of the sites, split into 120° wedges for three-sector sites. With wrap vectors, the wrapped cluster bounds the drop; without them, `bounds` does. Hand-placed BS layouts have no wrap vectors, so `bounds` is needed in most custom uses and is a direct argument. It is not called `area`, which in `generate_layout` means the size of a hall.

### 7.3 Hexagonal geometry

- **Orientation:** flat-top site hexagons with circumradius `D / sqrt(3)`, boresights at 30, 150 and 270 degrees pointing to edge midpoints (TR Table 7.8-1, figure). The six nearest neighbours are at 30° + k·60°. This is the grid of NG, Sionna, ns-3 and 5G-LENA.
- **Sites:** with `u = D (sqrt(3)/2, 1/2)` and `v = D (0, 1)`, the sites are `q u + r v` with `max(|q|, |r|, |q + r|) <= n_rings`, so `3n² + 3n + 1` sites.
- **Site order:** by ring, then by ascending azimuth in (−30°, 330°]. This equals NG's order for 3 rings (checked).
- **Lattice:** `a1 = n u + (n + 1) v`, `a2` = `a1` rotated by +60°. For `n = 2`: `a1 = (sqrt(3), 4) D`, `a2 = (−3 sqrt(3)/2, 7/2) D`. The seven images are 0, ±a1, ±a2, ±(a1 − a2). A numerical check confirmed that this set tiles the plane exactly once for 1 to 6 rings.
- **Sector cell:** the part of the site hexagon within ±60° of the boresight. Each of the three has one third of the hexagon area.

### 7.4 UT drop

- `per_cell = true`: each cell receives exactly `n_ut` UTs, uniform over its area by rejection sampling. A cell is a sector cell for hexagonal layouts with sectors, the site hexagon without sectors, and the lattice cell around a BS for indoor grids (the hall is partitioned).
- `per_cell = false`: `n_ut` UTs, uniform over the union of all cells.
- V2X presets derive the vehicle count from the gap process, and `n_ut > 0` throws for them.
- **Minimum distance**, to the nearest image of the nearest site. UMi and UMa: to `d2D` for outdoor UTs and to `d2D-out` for indoor UTs (TR 36.873 Table 6-1, note 1). RMa and SMa: to `d2D` (TR Tables 7.2-3 and 7.2-5 have no such note). InH: 0. InF: 1 m. A rejected UT is redrawn with its position and indoor distance together.
- UT heading is uniform in [0, 360) by default, tilt and bank are 0. Direction of travel is drawn independently, uniform in azimuth with elevation 0, and written explicitly. Speed follows the preset. Calibration orientations are set through the `ut` tuple: Table 7.8-2 is tilt 90°; Table 7.8-2A configs B and C are tilt 45°; config D is tilt 0° and heading 0°.
- **States written by the generator.** For O2I families, `generate_layout` evaluates the O2I fields of section 8.5 with its `seed` as master seed and the same keys, because the minimum-distance rule needs `d2D-in`. It writes `indoor`, `floor` and `d2d_in`. The drop thus keeps the spatial correlation of TR clause 7.6.3.3. The floor sets `z` by `hUT = 3(n_fl − 1) + 1.5`.

### 7.5 Presets

Values are from the cited tables. A preset defines its scenario family, geometry and drop. The generator sets `scenario` on the Tx rows where a family exists.

**TR 38.901, hexagonal (Tables 7.2-1, 7.2-3, 7.2-5).** All: 2 rings (19 sites), 3 sectors, wrap vectors set. `bs` tilt is the mechanical tilt in the library's sign (positive down).

| Preset | ISD (m) | hBS (m) | Mech. tilt | Min distance (m) | Non-outdoor UTs | hUT | Speed |
|---|---|---|---|---|---|---|---|
| `38.901_UMi` | 200 | 10 | 0 | 10 (`d2D-out` indoors) | 80 % building | `3(n_fl − 1) + 1.5`; outdoor `n_fl = 1`; indoor `n_fl ~ U(1, N_fl)`, `N_fl ~ U(4, 8)` | 3 km/h |
| `38.901_UMa` | 500 | 25 | 0 | 35 (`d2D-out` indoors) | 80 % building | as UMi | 3 km/h |
| `38.901_RMa` | 1732 | 35 | 0 | 35 (`d2D`) | 50 % building, 50 % car | 1.5 m | Building 3, car 40 km/h (TR gives none; library choice, as SMa) |
| `38.901_SMa` | 1299 | 35 | 5° at ISD 1299, 2° at ISD 1732, otherwise the nearer of the two | 35 (`d2D`) | 80 % building, 20 % car | Car 1.5 m; residential floors 1 to 2; commercial 1 to 5; `1.5 + 3(n_fl − 1)` | Building 3, car 40 km/h |

The electrical tilts of Table 7.8-1 (102° for UMa and UMi, 110° for indoor) belong to the antenna (`steer`), never to `tilt`. The SMa values 95° and 92° of Tables 7.8-1A and 7.8-2A are zenith angles of a mechanical tilt, i.e. 5° and 2° down (our reading).

SMa options: `p_residential` (0.9). Residential buildings use `d2D-in` up to 10 m, commercial up to 25 m. RMa uses up to 10 m, UMa and UMi up to 25 m. `d2D-in` is the minimum of two uniform variables (TR 7.4.3.1). Alternative ISDs (RMa 5000 m, SMa 1732 m) are given by `spacing`.

**TR 38.901, indoor (Tables 7.2-2, 7.2-4, 7.8-7).** The hall or floor is centred at the origin. No sectors, no wrap, no O2I.

| Preset | Area (m) | BS grid | Spacing (m) | hBS (m) | hUT (m) | Min d2D | Speed |
|---|---|---|---|---|---|---|---|
| `38.901_InH_Open`, `38.901_InH_Mixed` | 120 × 50 × 3 | 6 × 2 | 20 | 3 | 1 | 0 | 3 km/h |
| `38.901_InF_SL` | 120 × 60 × 10 | 6 × 3 | 20 | 1.5 | 1.5 | 1 m | 3 km/h (TR gives none) |
| `38.901_InF_DL` | 300 × 150 × 10 | 6 × 3 | 50 | 1.5 | 1.5 | 1 m | as SL |
| `38.901_InF_SH` | 300 × 150 × 10 | 6 × 3 | 50 | 8 | 1.5 | 1 m | as SL |
| `38.901_InF_DH` | 120 × 60 × 10 | 6 × 3 | 20 | 8 | 1.5 | 1 m | as SL |
| `38.901_InF_HH` | 300 × 150 × 10 | 6 × 3 | 50 | 8 | 3 (library value above `h_clutter = 2 m`, not a TR value) | 1 m | as SL |

InH positions are `x = −50:20:50`, `y = ±10` after centring. InF positions keep `D / 2` to the walls. The InF presets write `scenpar_override` for `DS_VS` (V/S of the hall) and `absTOA_Lmax` (largest dimension) on the Tx rows. InF-HH has no calibration layout in the TR; the preset uses the big hall, and its documentation says so.

**ITU-R M.2412 (Report M.2412-0, sections 8.5 and 8.6, Table 5).** Geometry and drop only, `scenario` stays empty and the user sets it. The presets take `options["config"]` (A, B, C as 1, 2, 3) and, for Urban Macro, `options["service"]` (1 = mMTC, 2 = URLLC).

| Preset | Layout | Notes |
|---|---|---|
| `M.2412_InH` | 120 × 50 m, 12 sites, 20 m | hUT 1.5 m |
| `M.2412_DenseUrban` | Macro ISD 200 m, 19 sites × 3 | Micro layer of `n_micro = 3` random sites per macro cell only in config C; option `h_micro` |
| `M.2412_Rural` | ISD 1732 m; 6000 m in config C | Config C: 40 % indoor, 40 % pedestrian, 20 % car, in-car speed 30 km/h instead of 120 km/h |
| `M.2412_UrbanMacro` | ISD 500 m or 1732 m | mMTC: 80 % indoor, 3 km/h. URLLC: 80 % outdoor, 30 km/h outdoors |

Remaining heights and shares are read from Table 5 of the Report in WP6, with the table reference in the code comment.

**TR 37.885 (V2X).** Roads and vehicles only: `37.885_Urban` (433 × 250 m road grid, 2 lanes per direction, lane width 3.5 m, sidewalk 3 m, minimum area 1299 × 750 m) and `37.885_Highway` (3 lanes per direction, lane width 4 m, length at least 2000 m, wrap along the road with a one-vector `wrap`). The position step along a lane is the bumper-to-bumper gap `max(2 m, Exp(mean = v · 2 s))` plus the vehicle length. Speeds and the type mix follow option A. Vehicles move straight along their lane; turning is out of scope. Options: `lane_width`, `n_lane`, `p_truck`. BS placement is defined in TR 36.885, which is not part of the available sources, so the presets create no BS, and `scenario` stays empty.

### 7.6 Later scenarios

An IRS or a sensing target is a further set of nodes with the same columns plus its own (a third file of the same format, and a third `layout_nodes` member). Dual mobility of the existing nodes is already covered. Aerial heights, HST, NTN and ISAC targets are added as presets, `options` keys and table columns. A target list would become a third `layout_nodes` member, appended to the class.

## 8. Channel generation from a layout

### 8.1 Overview

The channel generation from a layout runs in three stages. `get_channels_layout` runs all three; the manual path calls them one by one.

| Stage | Function | Consumes | Produces | Scope |
|---|---|---|---|---|
| 1 | `init_link_states` | Positions, scenarios, O2I columns, wrap vectors, seeds, frequencies | `layout.links`: per propagation link the segment, wrap image, scenario file, LOS state, O2I state and seed | The discrete random decisions of TR clause 7.5 step 1 and clause 7.6.3. No LSPs, no rays. |
| 2 | `generate_rays` | The layout with its links, a list of links | `gbsm_rays` per link and band: delays, powers, angles, polarization, optionally scatterers, at the start of each segment | TR steps 2 to 11 without antennas: path loss, LSPs, clusters and rays. |
| 3 | `get_gbsm_channels` | The layout with antennas and tracks, the links, the rays, snapshot times | One `channel` per Tx node, Rx node and band | Antennas, motion (phase rotation or drift), link gain along the track, and the cross-fade between segments. |

`attach_terminals` uses stage 1 (large-scale rule) or stage 2 (RSRP rule). `export_rays_qrt` writes the rays of stage 2 together with the link gain of stage 3 to a file. `crossfade_segments` is the last step inside stage 3.

### 8.2 Design

The pipeline has no builder object. QuaDRiGa's `qd_builder` grew because one object is input bundle, random-number owner, store of all intermediate results and cache state at once. Sionna's scenario object and ns-3's per-link caches avoid part of this, but both still hold mutable state with invalidation logic. Here:

- `layout` holds the deployment and, in `layout.links`, the discrete link states of the current realisation. Editing them is the way to force a state.
- `gbsm_rays` is the existing plain class and is an output, returned to the caller.
- All random values are **addressed**: they are a function of the seed, a key and the position (section 8.5). SOS phases, LSPs and path loss are therefore recomputed where they are needed instead of being stored. Evaluation order, batch size and thread count cannot change the result.
- **Stale links.** `layout.links` stores a hash over everything that determines the links (positions, names, sectors, frequencies, seeds, the per-site dual-mobility flag, `scenario`, `indoor`, `floor`, `d2d_in`, `clutter`, `scenpar_override`, `wrap`). Stages 2 and 3 recompute it and throw on a mismatch. Measured cost: 6 ms per 100,000 rows and 55 ms per million (byte-wise FNV-1a, one thread, idle machine), small against the ray generation. A hash is used because members are public and the MATLAB and Python wrappers pass the layout as a struct or dict: an edit there cannot be intercepted by setters, so automatic invalidation is not possible. `layout.links.reset()` clears the links explicitly. Antennas, orientations and `pairing` are not part of the hash: `layout.links` holds all non-co-located site and terminal pairs, and `pairing` selects from them at the output.

Measured cost of recomputing the `lsf` phases of one group (one thread, load 0.9): 0.25 ms for the TR methods and about 18 ms for `quadriga` with bias removal over 1000 positions; 45 kB per group. `get_channels_layout` keeps them in a local variable for the duration of one call. No persistent cache is added; a stored phase member can be appended later without changing a function.

### 8.3 Functions

```cpp
// One call
std::vector<channel<double>> get_channels_layout(
    const layout &layout,
    const arma::vec &times = {},                         // Snapshot times in [s]; empty = one snapshot per row
    const std::string &output = "taps",                  // "taps" or "rays"
    const std::string &method = "",                      // "38.901", "38.901_MF", "quadriga"; "" = method of layout.links, else "38.901"; a different method than the links throws
    arma::umat *index = nullptr,                         // Optional output: i_tx_node, i_rx_node, i_freq per channel, [3, n_channel]
    const std::string &mobility = "auto",                // As gbsm_single_link_channel
    double overlap = 0.5,                                // Segment overlap, 0 to 1, as crossfade_segments
    arma::sword seed = -1,                               // Master seed, used if layout.links is empty; -1 = layout.seed, else the system random device; a seed with filled links throws
    const std::vector<std::string> &config_dirs = {});   // User conf folders, searched before the shipped ones; ignored with filled links

// Manual path, step 1: link states, written to layout.links
void init_link_states(
    layout &layout,
    const std::string &method = "38.901",
    const arma::vec &o2i_low = {},                       // Low-loss share of building O2I, [1]; empty = family value
    bool pol_variability = false,                        // TR 7.6.16; TR methods only
    double anchor_freq = 2.0e9,                          // Anchor frequency of 38.901_MF in [Hz]
    const std::string &sos_preset = "",                  // "": Exp256 for TR methods, Comb256 for quadriga
    arma::sword seed = -1,                               // -1 = layout.seed, else the system random device
    const std::vector<std::string> &config_dirs = {});   // User conf folders, searched before the shipped ones

// Manual path, step 2: rays
std::vector<std::vector<gbsm_rays>> generate_rays(       // [n_out][1 or n_freq]
    const layout &layout,
    const uint_list &i_link = {},                        // Propagation links, 0-based, [n_out]; empty = all
    bool scatterers = false);                            // Compute FBS and LBS; always done for "quadriga"

// Manual path, step 3: channels
std::vector<channel<double>> get_gbsm_channels(
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays,     // From generate_rays, possibly edited
    const uint_list &i_link = {},                        // The links of 'rays'; empty = all
    const arma::vec &times = {},
    const std::string &output = "taps",                  // "taps" or "rays"
    arma::umat *index = nullptr,
    const std::string &mobility = "auto",                // "drift" without scatterers throws
    bool oxygen = false,                                 // TR 7.6.1
    bool use_absolute_delays = false,                    // Include the LOS delay in all delays
    double overlap = 0.5);
```

`get_gbsm_channels` returns channels only for channel links whose propagation links are all present in `i_link`. `get_channels_layout` uses `layout.links` if it is filled. Otherwise it draws the link states on a local copy and leaves the layout unchanged. It runs the steps in batches whose size bounds the memory, and concatenates the results in canonical order. Its advanced settings stay at their defaults; other settings need the manual path. `n_cluster` (TR 7.6.15) is not exposed. The steps are called stage 1, 2 and 3 below.

Call rules: `init_link_states` throws on an empty `layout.center_freq`; `generate_layout` leaves it empty, and the user sets it. `generate_rays`, `get_gbsm_channels`, `attach_terminals` and `export_rays_qrt` throw `"Call 'init_link_states' first."` on empty links. In `get_channels_layout`, `config_dirs` is ignored when the links are filled, because `layout.links.conf_files` holds the resolved paths, and scatterers are requested whenever the resolved mobility is `drift`.

**Working below the manual path.** The GBSM functions stay public and independent of the layout. A user obtains the SOS phases of a link with `gbsm_init_phases(scenpar, "lsf", layout.links.gbsm_seed(i_link), -1, preset, ...)` and can feed them into `gbsm_draw` and the following functions. With `quadriga`, the bias removal additionally needs the positions of all links of the group. Rays produced this way can go into `get_gbsm_channels`. Its link gain is computed from `gbsm_seed`, so a user who draws with another seed sets `layout.links.gbsm_seed` to it.

### 8.4 `layout_links`

A class with public members, set by name, held as `layout.links`. It holds no LSPs, phases or rays, and it is not written to the CSV, because the seed reproduces it.

| Member | Size | Meaning |
|---|---|---|
| `i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of the propagation link, 0-based |
| `image` | `[n_link]` | Wrap image index, 0 = real site (not an array index, not shifted in MATLAB) |
| `i_conf` | `[n_link]` | Index into `conf_names`; output, derived from the scenario, `los` and `indoor` |
| `conf_names` | `[n_conf]` | Resolved conf names |
| `los` | `[n_link]` | 1 = LOS |
| `indoor`, `floor_id`, `d2d_in` | `[n_link]` | O2I state at the segment start; `floor_id = floor − 1`, outdoor 0 |
| `o2i_rnd`, `o2i_model` | `[n_terminal, 4]`, `[n_terminal]` | Inputs of `o2i_loss` and the chosen model per terminal |
| `gbsm_seed` | `[n_link]` | Seed of `gbsm_init_phases`; equal for one site, image and conf |
| `seed`, `method`, `sos_preset`, `anchor_freq`, `pol_variability`, `conf_files` | | As used |
| `layout_hash` | | Hash of all layout members that determine the links, checked by the later steps |

The documented manual path is to edit `los` or `indoor` between stage 1 and stage 2. `i_conf` is an output: stages 2 and 3 derive the conf of each link from its scenario, `los` and `indoor`. If it differs from the stored value, they use the derived conf and its hashed seed for that link. A `gbsm_seed` set by the user is used as long as the conf is unchanged. The layout is `const` in these steps, so the stored table is not modified. With `quadriga`, an edit moves a link between groups, which can change the bias removal and thus the values of other links in those groups. A new realisation overwrites `layout.links`. To keep two, copy `layout.links`, not the layout with its antennas.

### 8.5 Stage 1

1. Validate the layout. Resolve frequencies, method and seeds. Master seed `-1` is replaced by one draw of the system random device. Site and terminal indices follow the order of first appearance.
2. Resolve scenarios per segment and link: the Rx row value, else the Tx site value, else throw. On the Rx side a change of `scenario` or `indoor` starts a new segment. The Tx value is constant along a track.
3. Family to conf: a central table maps each family to its conf names (without the prefix `3GPP_38.901_v19_`), LOS-probability model, O2I models and weights, and maximum indoor distance.

| Family | LOS conf | NLOS conf | O2I LOS | O2I NLOS | LOS model | Building O2I models (weights) | `d2D-in` max (m) | Non-building UTs |
|---|---|---|---|---|---|---|---|---|
| `UMi` | `UMi_LOS` | `UMi_NLOS` | `UMi_LOS_O2I` | `UMi_NLOS_O2I` | `38.901_UMi` | `low`, `high` (0.5, 0.5) | 25 | Outdoor |
| `UMa` | `UMa_LOS` | `UMa_NLOS` | `UMa_LOS_O2I` | `UMa_NLOS_O2I` | `38.901_UMa` | `low`, `high` (0.5, 0.5) | 25 | Outdoor |
| `RMa` | `RMa_LOS` | `RMa_NLOS` | `RMa_LOS_O2I` | `RMa_NLOS_O2I` | `38.901_RMa` | `low` (1) | 10 | `car` |
| `SMa` | `SMa_LOS` | `SMa_NLOS` | `SMa_LOS_O2I` | `SMa_NLOS_O2I` | `38.901_SMa` | `low_A` (1) | 10 residential, 25 commercial | `car` |
| `InH_Open`, `InH_Mixed` | `Indoor_LOS` | `Indoor_NLOS` | | | `38.901_InH_Open`, `38.901_InH_Mixed` | none | | none |
| `InF_SL`, `InF_DL`, `InF_SH`, `InF_DH` | `InF_LOS` | `InF_NLOS_<type>` | | | `38.901_InF_<type>` | none | | none |
| `InF_HH` | `InF_LOS` | | | | `38.901_InF_HH` | none | | none |

An exact conf name is looked up in this table. It fixes LOS or NLOS and building or outdoor for all links of that terminal in that segment. A name that is not in the table is a user conf file: it is used for LOS and NLOS alike, with no O2I. Conf files are found by name with `conf_file_list(config_dirs)`: the given folders first, the shipped folder last (section 2.3).

4. **Wrap image**, per propagation link: the image with the shortest 2D distance from the site to the first position of the terminal. Ties take the lowest index in the order 0, +a1, −a1, +a2, −a2, +(a1 − a2), −(a1 − a2); a one-vector `wrap` has images 0, +a1, −a1. The image is kept for all segments of a link. A terminal outside the wrapped cluster still gets the nearest image. All later distances use the virtual Tx position.
5. **O2I state per terminal**, for families with O2I; InH and InF store `indoor = 0`, `d2d_in = 0` and have no O2I loss. Unless fixed in the layout, the values come from spatially correlated `sos_rand` fields at the first position (x, y), with the correlation distances of TR Table 7.6.3.1-2 as all-correlated values (Table 7.6.3.4-1): indoor state, building or car class and O2I model 50 m (`IO_state_lambda`), the two `d2D-in` uniforms 25 m, loss deviation 10 m. `U < share` selects building or car. Shares for custom layouts are 0.8 (UMi, UMa, SMa) and 0.5 (RMa). In RMa and SMa all non-building UTs are in cars. SMa draws residential or commercial with `p_residential = 0.9`. The random values are drawn once per terminal at its first position. A later segment with another fixed `indoor` uses the same values, with the building or car model chosen from its own state; a drawn state (`indoor = -1`) is constant along the track.
6. **O2I loss:** `o2i_rnd` is `[n_terminal, 4]`, and the chosen model is stored in `o2i_model`, so that later steps do not depend on `o2i_low`. A fixed `d2d_in` sets columns 1 and 2 to `d2d_in / d2d_in_max`, and a value above the maximum throws. `o2i_loss` runs once per (model set, `d2d_in_max`) group and supplies both the loss and `d2d_in`, so that the LOS distance and the penetration loss agree. Cars use `car` (non-metallic) and `d2d_in = 0`.
7. **Floor:** `floor = -1` means `n_fl = max(round((z − 1.5) / 3) + 1, 1)`, so that it matches the height. `generate_layout` draws the floors per UT, uncorrelated (the TR defines no correlation for them), with the key `floor|<family>|<terminal>`. Outdoor and car: floor 1. `floor_id = floor − 1`.
8. **LOS state** per propagation link, unless fixed by an exact name: `U < p_LOS` with `los_probability` at `d2d_out = max(d2D − d2d_in, 0)` and a uniform value from `sos_rand` (x, y) seeded per site and image. The correlation distance is `LOS_state_lambda` of the family's non-O2I NLOS file (InH: `Indoor_NLOS`; InF: the NLOS file of the clutter type). InF with a `d_clutter` column uses `d_clutter / 2` (Table 7.6.3.1-2). InF-HH has `p_LOS = 1` and needs no draw. A track without a scenario change keeps its LOS state, so long tracks need explicit segments.
9. **Conf:** from the table of step 3, O2I columns for building UTs. `init_link_states` resolves and stores every conf of each family in use (LOS, NLOS and both O2I variants), whether drawn or not, so that edited states find their files; stages 2 and 3 throw if a derived conf is not in `conf_names`. Car UTs use the outdoor LOS or NLOS conf with `o2i = 0`, plus the car loss (TR eq. 7.4-4, our reading of "outdoor UTs (in car)").
10. The state draws depend only on position and seed, without bias removal over the UT set, so adding a UT does not change the states of the others. All `sos_rand` calls use preset `Exp256`.

**Seeds.** Each random stream has its own seed: a 64-bit FNV-1a hash over a base seed and a key string, masked to a non-negative `arma::sword`. Names, not indices, are hashed, so that reordering rows does not change a site's values. Site-specific streams use the site's `seed` as base if it is set, otherwise the layout seed; all-correlated streams use the layout seed. `init_link_states` uses `layout.seed` when its `seed` argument is -1, draws one from the system random device if both are -1, and stores the seed used in `layout.links.seed`. A layout written with its seeds is therefore reproducible from its files alone. The base seed of all streams is the resolved layout seed. `init_link_states` writes the seed it used into `layout.seed`, before computing `layout_hash`, so that the files reproduce the run also when a `seed` argument was given. `drop_terminals` with seed -1 uses `layout.seed` if it is set, else the system random device, and does not change `layout.seed`. `generate_track` takes no layout; with seed -1 it uses the system random device, and its draws (direction, street lengths and turns) use the keys `track|<node name>`.

| Stream | Key | Correlation type |
|---|---|---|
| Indoor state, O2I class (building or car), O2I model, `d2D-in` uniforms, loss deviation | `<purpose>|<family>` with purposes `io`, `o2i_class`, `o2i_model`, `d2din1`, `d2din2`, `o2i_dev` | All-correlated (spatial field) |
| Floor of a generated UT | `floor|<family>|<terminal>` | Per UT |
| LOS state | `los|<site>|<image>|<family>` | Site-specific |
| GBSM sets | `gbsm|<site>|<image>|<conf>` | Site-specific |

### 8.6 Stage 2

For each group of links with one site, image, conf and `gbsm_seed`, following the workflow in the section description of `src/qd_scenpar.cpp`:

1. `gbsm_init_phases` for `lsf`, `cluster` and `ray` with the group's seed from section 8.5, and `dual_mobility` of the site. Settings by method:

| Setting | `38.901`, `38.901_MF` | `quadriga` |
|---|---|---|
| SOS preset | `Exp256` | `Comb256` |
| Bias removal | off | on, over the 2D positions of all links of the group |
| `angle_model`, `pol_model` | `additive`, `38.901` | `rotation`, `quadriga` |
| `horizontal`, `cholesky`, `clip_spreads` | `true` | `true` |
| `path_loss` | **without** `d2d_in`: TR geometric distance, eq. 7.4-2 | with `d2d_in` (legacy distance), 0 for outdoor links |

2. `gbsm_draw` per set with `floor_id`, `path_loss` with column 10 (`rnd_he`), `lsp_generate` with `sf_sigma_dB`. `38.901_MF` also computes LSPs at `anchor_freq` with the same random values.
3. `ssf_clusters` (with `o2i` = building UT) and `ssf_rays` per method. `38.901`: per frequency. `38.901_MF`: clusters once, rays per frequency. `quadriga`: once for all frequencies.
4. `ssf_scatterers` if requested, always for `quadriga`.

With `Exp256` and no bias removal, results do not depend on the batch. For `quadriga`, bias removal runs over all positions of the group, regardless of the requested subset, so that subsets give the same values. Groups run in parallel with `qd_batch_loop` when they are large enough (measured limit).

### 8.7 Stage 3

1. For each channel link and frequency: sample both nodes with `sample_track` over the requested `times`. Rays of a link are taken as entry `i_freq` with `gbsm_single_link_channel` frequency index 0 when there are `n_freq` entries (TR methods), and as entry 0 with frequency index `i_freq` when there is one (`quadriga`); `attach_terminals` uses the same rule. A one-row node is expanded to the snapshot count of the other; two multi-row nodes without `times` need equal row counts. The array is the node's antenna, or `xpol` (section 3).
2. Link gain per snapshot: `−PL − O2I + 10 lg(SF)`. The `lsf` phases are recomputed from the group seed. `gbsm_draw` (`lsf`) gives the values at the segment start and at moved positions, `path_loss` uses `rnd_he` **held** at its segment-start value (otherwise the loss steps along the track), then `lsp_generate`, `o2i_loss` for building and car UTs with the stored model, the fixed distance columns and the deviation column at the moved position, and `kf_ratio`, as in the section description of `src/qd_scenpar.cpp`. A node that does not move needs one evaluation only.
3. `gbsm_single_link_channel` with the Tx position shifted by the link's image offset, per segment, with absolute delays internally when the link has several segments, so that merged segments share one delay reference. The result is converted to the requested delay form.
4. Tracks with several segments: the channels of consecutive segments are joined by `crossfade_segments`. A segment other than the first also covers the snapshots of the preceding segment, as in NG's `get_subtrack`, and its rays refer to the first position of the segment.
5. Output in canonical order: `i_tx_node` varies fastest, then `i_rx_node`, then `i_freq` slowest, with only the paired links. `channel::name` is `<tx name>:<sector>_<rx name>:<sector>`, without the sector when the node's sector column is all 0. The `index` output gives the position in a 3D grid for HDF5 storage.

Dual mobility (both ends move, as in V2V, NG's dual-mobility mode or a moving relay or IRS) is part of this version: the Tx file has the same columns as the Rx file, so Tx point drops with speed and Tx tracks are written the same way, and stage 3 samples both ends over the same `times`. A moving Tx is allowed. `dual_mobility` is set **per site**, when that site has more than one row or a non-zero speed, so that one moving BS does not change the realisations of static sites.

### 8.8 `attach_terminals`

```cpp
arma::uvec attach_terminals(                                 // Serving Tx node per Rx terminal, 0-based, [n_rx_terminal]
    const layout &layout,
    const std::string &rule = "large_scale",                 // "large_scale" or "rsrp"
    const std::vector<std::vector<gbsm_rays>> &rays = {},    // "rsrp": rays of all links from generate_rays, required; "large_scale": unused
    arma::uvec *site = nullptr,                              // Optional output: serving site per Rx terminal, 0-based, [n_rx_terminal]
    arma::mat *gain_dB = nullptr,                            // Optional output: coupling gain or RSRP in [dB], [n_rx_terminal, n_tx_node]
    arma::uword i_freq = 0,
    arma::uword tx_port = 0);                                // BS port
```

- **`large_scale`** (TR Table 7.8-1, "pathloss considering LOS angle"): `−PL − O2I + 10 lg(SF)` plus the antenna gain of a single unit-gain LOS path towards the virtual Tx through `get_channels_planar`, which includes the port's element coupling and position phases and so the array gain and electrical tilt. It needs no rays. The inclusion of SF and O2I is our reading; the full calibration session checks it.
- **`rsrp`** (TR Tables 7.8-2, 7.8-2A, formula in TR 36.873 clause 8.1): per Rx port the sum of the ray powers of the given Tx port at the drop position, then the mean over Rx ports. It evaluates `gbsm_single_link_channel` with `output = "rays"` and does not use tap output, so attachment does not follow fast fading.
- `rsrp` requires the rays of all links, from `generate_rays`, which the caller usually reuses for the channels; nothing is computed and discarded. RSRP consumes the rays, the link gain at the drop position and the Tx and Rx antennas from `layout.antenna`: the per-path coefficients of `gbsm_single_link_channel` with `output = "rays"` give, per Rx port, `Σ_paths |h|²` of the chosen Tx port, averaged over the Rx ports. This computation is an internal helper on a `channel` object, so that a later analysis session can make it public (a channel statistic) without changing `attach_terminals`.
- A terminal with several panels takes the highest value over its nodes. A tie takes the lowest node index. Attachment evaluates all sites and ignores an existing `pairing`. The caller stores the result as `layout.pairing = [site.t(); regspace<arma::urowvec>(0, n_rx_terminal − 1)]`, which links all sectors of the serving site to the terminal and is written to the CSV.

### 8.9 `export_rays_qrt`

```cpp
arma::vec export_rays_qrt(                                 // Returns the x-y shift of the local coordinates in [m], [2]
    const std::string &fn,
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays = {}); // Rays of all links; empty = computed per site, which bounds the memory
```

- One origin per site at the site position, with sector 0's orientation. One destination per terminal, one CIR per row of the terminal (all panels of a terminal share their positions; the CIR uses the orientation of the terminal's first node).
- All positions in the file are local. The shift is the centroid of all site and CIR positions in x and y, with z unchanged. It is returned, and the caller adds it back. The file format has no field for it.
- For each origin: for each row, take the rays of the row's segment, add scatterers if missing, compute the stage 3 link gain and `kf_ratio` for that row, call `ssf_paths(..., "drift")`, then `qrt_paths_from_scatterers`. Collect the paths of all CIRs and call `qrt_file_append` once for the origin. The file is created first with `qrt_file_init`, with all CIR positions, one destination per terminal and its `cir_offset`.
- A point drop with speed writes one CIR. Its motion is applied at evaluation: `qrt_file_read`, `gbsm_rays_from_paths` with the method, `gbsm_single_link_channel` with the chosen arrays and positions.

Throws: more than one frequency with a TR method; a Tx node with more than one row or a non-zero speed; a terminal without a paired site (incomplete pairing); wrap-around (one origin holds one position, the images differ per CIR); a terminal at a site position (zero-length LOS).

Known limits, stated in the documentation: the QRT format stores no tap index and no ground permittivity. Evaluation from a file gives rays, not TR taps, and moving a ground-reflection ray from a file throws.

### 8.10 `crossfade_segments`

```cpp
channel<double> crossfade_segments(
    const std::vector<channel<double>> &segments,  // Channels of one link in track order, each overlapping its predecessor
    const arma::uvec &n_overlap,                   // Snapshots shared with the preceding segment, [n_segment]; entry 0 is 0
    double overlap = 0.5);                         // Part of the shared snapshots used for the ramp, 0 to 1; 0 = hard switch
```

It is a port of `qd_channel.merge` and `merge_coeff` of QuaDRiGa-NG, renamed after what it does. Over the shared snapshots, the paths of the old segment fade out and those of the new segment fade in, in power, with weights `1 − w` and `w`; `w` follows NG's ramp `0.5 (1 + sin((x − 0.5) π))` over the ramp part `x` of the overlap. One rule applies to both output forms of stage 3:

- **NLOS:** taps are matched as in NG and cross-faded, so the tap count stays fixed. Rays are kept side by side (the union of both segments), with coefficients scaled by `sqrt(1 − w)` and `sqrt(w)`. Rays of the two segments need no labels: the function knows them by input segment.
- **LOS:** one LOS path is kept. Its phase, delay and path members come from one source segment, the new one unless its LOS power is 0, then the old one. Its coefficient is scaled so that `|h|² = (1 − w) |h_old|² + w |h_new|²`, using the path gain for rays and the tap power for taps. This deliberately deviates from NG, which mixes the two LOS coefficients linearly: in phase mobility the two segments evaluate the LOS with planar expansions about different reference points, so their phases differ and a coherent mix would fade in the ramp.
- **GR:** the same rule as LOS if both segments have a GR path; otherwise a `sqrt` ramp of the one that exists.

LOS and GR are identified by the `tap` entry in `par_data` (LOS 0, GR 3 if enabled), which `gbsm_single_link_channel` writes for both output forms. Delays are consistent because stage 3 uses absolute delays internally. The output holds `name`, `center_freq`, positions, orientations, `coeff_re`, `coeff_im` and `delay`, equal `n_rx` and `n_tx`, and valid `is_valid()`. For rays, the path-level members (`path_gain` scaled by the weight, `path_length`, `polmat` scaled by the square root of the weight, `path_angles`, `fbs_pos`, `lbs_pos`) are carried for every kept ray, because each ray keeps its geometry. For taps they are not carried, because a matched tap has no single geometry. `par_data` is not carried. The NG faults of section 10.3 are fixed in the port. It is in the section "Channel functions", and in Python `qd.channel.crossfade_segments`.

## 9. Wrappers

All public functions of section 2.1 get MEX and Python wrappers following the SPEC documents.

- A `layout` is a MATLAB struct and a Python dict with the C++ member names; `tx`, `rx` and `links` are nested. Matrices keep their C++ layout. `scenpar_override` is a nested struct or dict with the scenario keys as fields.
- `antenna` is `[n_antenna][1 or n_freq]`: in MATLAB a cell array of 1xN struct arrays, converted with `validate = false`; in Python a list of lists of ordinary 3D arrayant dicts, one dict per band. The 4D multi-frequency dict is not used here, because it holds one set of grids and element positions and per-band arrays differ in these (D25). Each entry is validated on its own. `antenna_args` is a nested cell array or list.
- `rays` is a cell or list per link, using the existing `gbsm_rays` conversions.
- **Indices** are 0-based in Python and 1-based in MATLAB for: `offset`, `pairing`, `i_node`, `i_link`, `index`, `i_freq`, `tx_port`, the output `i_row` of `sample_track`, the output of `attach_terminals`, and the members `i_tx`, `i_rx`, `i_segment`, `i_conf` and `o2i_model` of `layout.links`. Not shifted: `i_antenna` (1-based, 0 = xpol), `sector`, `image`, `floor`, `floor_id`, `indoor`, `relative` and the seeds (values).
- **In/out arguments** follow the SPEC split into `_in` and `_out`: `layout_out = set_antenna(layout_in, ...)`, `layout_out = drop_terminals(layout_in, ...)`, `layout_out = init_link_states(layout_in, ...)`, `layout_out = eval_antennas(layout_in, center_freq)`, `nodes_out = generate_track(nodes_in, ...)`, `nodes_out = set_orientation(nodes_in, ...)`.
- `set_antenna` takes the argument map as a dict in Python (`qd.layout.set_antenna(layout, "3GPP", {"panel": [4, 4, 1, 2]})`) and as a struct in MATLAB (`struct('panel', [4 4 1 2])`); a small conversion of dict or struct to the map is written once. In Python, `args` of `set_antenna` and `n_ut` of `generate_layout` are positional optionals and are added to `POSITIONAL_OPTIONAL` in `tests/python_tests/test_keyword_only.py`. The array form is a separate name in the wrappers, `set_antenna_array`, because Python and MATLAB do not overload by type.
- Conf folders: the wrappers of `get_channels_layout` and `init_link_states` append the shipped folder they find at run time to `config_dirs` (section 2.3).
- Python submodules follow the doc sections. `layout`: `qd.layout.csv_read`, `csv_write`, `generate_layout`, `drop_terminals`, `generate_track`, `sample_track`, `set_orientation`, `set_antenna`, `set_antenna_array`, `set_antenna_qdant`, `eval_antennas`, `get_channels_layout`, `init_link_states`, `generate_rays`, `get_gbsm_channels`, `attach_terminals`, `export_rays_qrt`. `channel`: `qd.channel.crossfade_segments`. MATLAB uses the C++ names in `quadriga_lib`. Doc headings follow the SPEC: bare names (`# generate_layout`, `# crossfade_segments`), and `# layout.csv_read`, `# layout.csv_write` for the two names that lose the prefix `layout_`. WP12 adds `[families]` entries in `docs_builder/content/api_matching.toml` for `set_antenna_array` and `eval_antennas`; bare headings need no `[python_names]` entries.
- SPEC updates in the same work: the tuple exception in `md/SPEC_CPP_API.md` ("Declarations") includes the `generate_layout` tuples, `grid` included (it stays `arma::vec`, because 0 rings is a valid value and `uint_list` has no "not given" value); in `md/SPEC_PYTHON_API.md` the submodule list includes `layout` (section "Network layouts and mobility"), and the prefix list includes `layout_` in `layout`.

## 10. QuaDRiGa-NG mapping

This section is informative. The NG integration is a separate later session; this project changes nothing in NG. The mapping shows that the library can carry NG's functions, and which NG faults the port fixes.

### 10.1 `qd_layout`

| NG | Layout system | Action |
|---|---|---|
| Properties `tx_track`, `rx_track`, `tx_position`, `rx_position`, `tx_name`, `rx_name`, `no_tx`, `no_rx` | `layout::tx`, `rx` | Facade in NG |
| `tx_array`, `rx_array` | `layout::antenna`, `i_antenna` | Facade in NG |
| `pairing`, `no_links` | `layout::pairing` | Facade; NG's Tx/Rx indices |
| `generate` | `generate_layout` | Port; NG's merged sector arrays remain as one node |
| `randomize_rx_positions` | `drop_terminals` plus `generate_track` | Rewrite; RNG parity is not a goal |
| `set_scenario` | `init_link_states` | Rewrite |
| `gen_o2i_loss` | `init_link_states` | Rewrite |
| `set_pairing('all')` | Default | Port; excludes co-located nodes (distance ≤ 1e-4 m) |
| `set_pairing('power')` | `attach_terminals` | Later, when NG's builder is replaced |
| `init_builder`, `get_channels`, private `parse_positions`, `sort_channels` | `get_channels_layout` and the manual path | Rewrite; NG keeps its signature |
| `get_channels_seg` | `generate_rays` with `i_link`, then `get_gbsm_channels` | Rewrite in NG on top of the manual path |
| `set_satellite_pos` | Not in this project | Stays in NG until NTN is planned |
| `power_map`, `visualize`, `kml2layout`, `layout2kml`, `copy`, `call_private_fcn` | | Stay in NG |
| Private `get_channels_seg_unmerged`, `ctrans`, `get_wgs84_trans` | | Drop after a caller check in NG |

### 10.2 `qd_track`

| NG | Layout system | Action |
|---|---|---|
| `generate` | `generate_track` | Port |
| `interpolate` | `sample_track` | Rewrite |
| `interpolate_positions`, `interpolate_movement` | | Stay as NG wrappers |
| `calc_orientation` | `set_orientation` | Port |
| `set_speed`, `movement_profile` | `speed`, `time` columns | Facade in NG |
| `get_length`, `positions_abs` | Direct on `pos` | Facade in NG |
| `set_scenario`, `add_segment`, `split_segment` | Row-wise `scenario` | Stay in NG on top of the class, or port when the integration needs them |
| `get_subtrack`, `correct_overlap` | `get_gbsm_channels` | Rewrite |
| `checksum`, `copy` | | NG only; a cache uses its own revision |

### 10.3 NG faults fixed in the port

The library implementation does not copy these faults; it implements the intended behaviour, and the tests of the affected functions cover it. NG itself is not changed in this project.

From reading the NG source, not run:

- `get_channels.m:184` divides by `mp(2, ...)` instead of time differences.
- `merge.m:358-359` reads `tx_position` for `tx_orientation`.
- `set_scenario.m:247` and `:316-320` use wrong indices when `rx` is a subset.
- The pairing de-duplication key `tx + n*rx` collides.
- `get_channels.m:427` matches names with a regexp, so `Rx1` matches `Rx10`. Use indices.
- `get_subtrack.m` and `parse_positions.m` treat the end of a closed track differently.
- `qd_track.interpolate` reads `par.kf` and `par.pg` that are not always present.

### 10.4 New functions

The public functions are listed in section 2.1, the internal helpers in section 2.2.

## 11. Work plan

The implementation runs unattended, overnight, with a manager session and sub-agents. This section is written so that the manager can run it without asking the maintainer.

### 11.1 Setup and rules

- The plan is committed on main before the run. The work happens on a branch `layout` in its own scratch worktree in the agents folder outside the repository, never in the main checkout. Builds stay in the worktree; the main checkout and its `build_linux` are not touched. Uncommitted changes in the main checkout are not part of the run; the maintainer commits or stashes them before.
- One commit per work package, following `CONTRIBUTING.md`, section "Commit messages": 1 to 3 short paragraphs, a specific subject, no internal item ids, no references to `md/` planning documents. Nothing is pushed or merged. The maintainer reviews all commits in one go before the merge.
- Test-first where it applies: write the test, see it fail in the expected way, then implement. New tests must not add significantly to the suite's run time (section 12).
- Expected values come from the TR, closed forms or brute force, never from the function under test. Supporting tasks use library functions.
- Be conservative with tokens: pass agents only the sections and decisions they need, not the whole conversation or plan.
- Watch for unexpected breakage: after each package, the full C++ suite runs once (about 3 minutes on an idle machine), not only the new tests.
- The machine is expected to be idle; the load is still checked before builds and benchmarks, as the global instructions require: builds use all cores; test runs use `OMP_WAIT_POLICY=passive`; OpenMP size limits are measured with the default policy, wrapped in the benchmark lock, with the load recorded.

### 11.2 Agents per work package

Each package runs these steps in order; agents run one after another, so each builds on the previous output.

1. **Implement** (one agent): the package's sections of this plan, the SPEC documents, and the code it builds on. It writes tests first, implements, runs the affected tests, and reports what it did and every deviation.
2. **Review** (one independent agent, or two focused ones for WP7 and WP10, one on the code and one on the physics): it receives the package's plan sections and the diff, not the implementer's reasoning, and works in its own scratch space. It reports bugs and also answers: Can the implementation be made less complex? Which requirement leads to the most complex code, and how would relaxing it lead to a leaner design? Which tests are too strict or too loose (for example bit-identical results versus reasonable numerical bounds)?
3. **Fix**: the implementer, or the manager for small items, fixes what needs no decision. This includes small faults in existing library functions that the work exposes (a wrong result, a crash, a missing check), each fixed test-first in its own commit before the package's commit. Structural changes need the maintainer's OK and are recorded, not made: a change of an existing interface, a decision of section 1.3, model behaviour or a TR value, or a redesign of an existing function.
4. **Commit** after the full suite passes.

Model and effort per role (maximum Opus at high effort):

| Work package | Implement | Review |
|---|---|---|
| WP1, WP2, WP4, WP11, WP12 | Sonnet, high | Sonnet, high |
| WP3, WP5, WP6, WP8, WP9 | Opus, high | Sonnet, high |
| WP7, WP10 | Opus, high | Opus, high (two focused reviewers) |

The manager may change a model or merge two small packages when it saves time, and records why.

### 11.3 Packages, order and time

| WP | Content | Depends on | Time box |
|---|---|---|---|
| WP1 | `qd_csv.hpp`; material reader migration; `test_qd_csv`; extend `fz_material_csv` | | 35 min |
| WP2 | Classes, column table, validation, `layout_csv_read` and `layout_csv_write` without antenna evaluation, `fz_layout_csv` | WP1 | 60 min |
| WP3 | `layout::eval_antennas`, both forms of `set_antenna`, band rows, QDANT export on write | WP2 | 50 min |
| WP4 | `sample_track`, `generate_track` incl. `point`, `set_orientation`, mounted antennas | WP2 | 40 min |
| WP5 | Hexagonal and indoor builders, `drop_terminals`, TR 38.901 presets; seed hash, family table and O2I field draws in `qd_layout_internal.hpp` | WP4 | 60 min |
| WP6 | M.2412 and TR 37.885 presets (geometry and drop) | WP5 | 40 min |
| WP7 | `layout_links`, the three stages without segments, `get_channels_layout`, wrap-around | WP3, WP5 | 150 min |
| WP8 | `attach_terminals` | WP7 | 30 min |
| WP9 | `export_rays_qrt` | WP7 | 40 min |
| WP10 | Segments in stages 1 to 3 (split on a change of `scenario` or `indoor`, per-segment conf, O2I state and rays) and `crossfade_segments` | WP7 | 75 min |
| WP11 | MEX wrappers and tests for the committed packages; `api_mex/*.hpp` added to the MEX dependencies in `CMakeLists.txt` | All earlier packages committed or parked | 90 min |
| WP12 | Python wrappers and tests, SPEC and glossary updates (incl. the order of the Euler angle names), `make docs`, rows of `md/tr38901_inventory.md`; doc headings and `api_matching.toml` entries as in section 9 | WP11 | 90 min |
| WP13 | Final checks: full C++ suite, ASan, short fuzz runs of `fz_layout_csv` and `fz_material_csv`, MATLAB and Octave tests, Python tests, Windows build; a fix commit only if something fails | WP12 | 40 min |

A time box covers implement, review, fix and commit; the full suite runs once per package, before the commit. The time boxes add up to about 14 hours, more than the 10 available; the core (WP1 to WP5, WP7, WP10) takes about 8 hours. The time boxes are estimates for planning, not limits. The run continues until all packages are done, also beyond the night; the manager does not stop or park a package because of time. Guidance:

- A package is parked only when it is stuck: the same failure after two fix attempts, or a decision that only the maintainer can make. Its tests and partial code go to a separate branch, the reason goes into the questions file, and the run continues with packages that do not depend on it.
- The order of 11.3 is also the priority order. WP6, WP8 and WP9 can follow later if something blocks them; WP11 and WP12 wrap only the functions whose packages were committed.
- If the machine becomes busy later, the manager reduces thread counts as the global instructions describe, and does not treat slower runs as failures.

### 11.4 Questions and hand-over

- A point that needs the maintainer (section 11.2, step 3; an unavailable source; a contradiction in this plan) goes into a questions file in the manager's scratch folder, in full: what is decided, the background, options with trade-offs, what would be correct, a recommendation. The manager continues with independent work.
- The hand-over lists per package: commit, test results as actually run, review findings and how each was handled, deviations with reasons, measured OpenMP limits with load, side findings, and the open questions.

## 12. Tests

Each public function gets a `"<function> - input validation"` case (SPEC "Tests"); `test_layout_csv` also covers the `is_valid` rules of section 3 (names longer than 255 bytes, dependent wrap vectors, sites with unequal positions, a node with both `time` and `speed`) and `layout_links::reset`. The whole new suite should add well under 5 s to `test_bin`, and each file should stay under about 0.5 s. Sizes are small and distinctive, and the fixtures use library helpers except for the function under test. No test depends on a random realisation; statistical checks use fixed seeds with bounds that follow from the sample size. The compile-time rule: test files include `quadriga_layout.hpp`, not additional heavy headers, and share one fixture header `qd_test_layout.hpp`.

| Suite | Oracles and failure modes |
|---|---|
| `test_qd_csv` | Quotes, escapes, comments, BOM, CRLF, short and long rows, numbers, `NaN`, `Inf`, list and matrix cells, error line numbers, number round trip |
| `test_layout_csv` | `pairing`, `seed`, `layout_seed` and `relative` columns round-trip; Column order independence; each column class; extra sector inheritance; antenna change throws; held values; `NaN` vs empty direction; degrees to radians; unknown columns kept and reported; round trip; file unchanged on failure; path in `scenario` throws |
| `test_set_antenna` | Both forms; write, read and write to the same file names keeps the referenced QDANT files and deletes only unreferenced `<stem>_<k>.qdant`; a multi-band array round-trips with one entry per band; the argument form gives the same array as `generate_antenna` with the same arguments; an array set by the second form is exported to QDANT on write and read back equal; copy on write for a per-band call on a shared antenna; De-duplication by text; order Tx then Rx; arguments reach `generate_antenna` (compare with direct calls); QDANT path relative to the CSV; column merge; a spaced array without a frequency throws; band rows of section 4.5, including the classification, a band row with empty antenna columns, and a missing band; `eval_antennas = false` keeps `antenna_args` and `i_antenna` and round-trips |
| `test_sample_track` | Closed forms for each row of the table in section 6.2; hold at ends; 90° SLERP; direction fallback with tilt; direction on a track throws |
| `test_generate_track` | `point` keeps one row and sets speed and a horizontal direction; length, closure, spacing; `set_orientation` on a straight and a rising track, with offset and with `force`; mounted antenna (`relative = 1`) on a track turning by 90° and on a point drop; `generate_track` on a `relative = 1` node keeps its orientation; `street` tracks keep the length and turn statistics of the options within bounds |
| `test_generate_layout` | Site count `3n²+3n+1`; distance to the nearest neighbour equals ISD; neighbours at 30°+k·60°; site order; lattice tiles exactly once (brute force); sector azimuths; every preset against the table values of the TR; `InH` and `InF` positions in closed form; exactly `n_ut` per cell; minimum distance; floor rule range; indoor and car shares within bounds; `d2D-out` minimum for indoor UMa UTs; SMa tilt per ISD; V2X gap includes the vehicle length and `n_ut > 0` throws; unused or unknown override throws |
| `test_drop_terminals` | Voronoi cells and 120° wedges of three-sector sites; exactly `n_ut` per cell; `bounds` required without wrap vectors; the Rx side is replaced |
| `test_init_link_states` | Generic `xpol` gives the polarization matrix as coefficients; a changed layout throws; repeated calls of step 2 for a subset give the values of the full call; Family to conf resolution; exact name forces state; UT value wins over BS; wrap image equals brute-force nearest; image changes the seed; states unchanged when a UT is added; seed reproducibility; batch independence of rays; sectors share rays and differ by antennas; link gain equals direct `path_loss` + `o2i_loss` + SF, for building and car UTs; output order and `index`; one link equals a hand-wired call of the GBSM chain for all three methods; O2I path loss of TR methods uses the geometric distance (eq. 7.4-2); LOS distance and penetration loss use the same `d2d_in`; fixed `d2d_in` reaches `o2i_loss`; car UTs use outdoor confs; InH and InF have no O2I; `rnd_he` held along a track; a moving site leaves other sites' values unchanged; default `quadriga` run works; `drift` without scatterers throws |
| `test_init_link_states` (segments, WP10) | A change of `scenario` or of `indoor` starts a segment; the image is kept across segments; the rays of segment k refer to its first row; the cross-fade of stage 3 is continuous in power |
| `test_get_channels_layout` | Dual mobility: a moving Tx and a moving Rx give the relative Doppler of both motions; a static site keeps its values when another site moves; A layout written with seeds gives the same link states after reading; per-band arrays with different ports reach the channel; Empty links leave the layout unchanged; filled and edited links are used; stale links throw after each hashed member changes; `pairing` selects at the output; `drift` requests scatterers; empty `center_freq` throws |
| `test_attach_terminals` | Hand-computed gains on a two-site layout; large-scale gain includes the port coupling and electrical tilt; RSRP equals the sum of ray powers and differs from tap power on a case with coherent rays |
| `test_export_rays_qrt` | Round trip through `qrt_file_read` and `gbsm_rays_from_paths` against direct channels at the drop position and for a moved UT, with the returned shift applied; each documented throw |
| `test_crossfade_segments` | Equal segments reproduce the input; power continuity across the ramp; hard switch at `overlap = 0`; ray output keeps one LOS ray whose power is continuous across the ramp, and the NLOS ray power sums to the interpolated value; tap and ray output give the same total power at each snapshot; path union; regression for the orientation fault of NG |

TR oracles are the table values and formulas of TR 38.901 clauses 7.2, 7.4 and 7.8. Calibration metrics are not part of this suite (D15).

Fuzz targets: `fz_layout_csv` on `layout_csv_read` with arbitrary bytes, as `fz_material_csv`.

## 13. Status of the decisions

All decisions of the interview are in sections 1 to 12. These defaults were confirmed by the maintainer:

- A track without orientation columns keeps orientation 0 (section 6.1). RMa speed is 3 km/h in buildings and 40 km/h in cars, InF 3 km/h (section 7.5).
- QRT export with wrap-around throws (section 8.9). `scenpar.<key>` columns override scenario parameters per site (section 4.4). `indoor` codes are 0 to 3 (section 3).
- M.2412 and V2X presets are geometry and drop only (section 7.5). `quadriga` keeps the legacy O2I distance (section 8.6). Large-scale attachment includes SF and O2I (section 8.8). InF-HH uses library values (section 7.5).
- Loudspeaker presets are removed for now. The layout format already holds Tx-only nodes, so a later BS.2051 preset is a new table row.

Decided in the fourth round: the QRT export is `export_rays_qrt`; the segment join is `crossfade_segments`; everything declared in `quadriga_layout.hpp` is in "Network layouts and mobility", and `crossfade_segments` in "Channel functions"; `set_antenna` takes `generate_antenna` arguments as a name-value map, plus the array form; the CSV stores seeds and pairing, not forced link states; mounted antennas use the `relative` column.

A third, optional links file for forced link states can be added later without changing the two main files: one row per forced propagation link with the columns `tx`, `rx`, `segment` and the forced values (`los`, `indoor`), keyed by names, applied by `init_link_states` after the draw, and checked against the layout's names and segments.

Open for later work, not blocking: wideband antenna arrays (`set_antenna_mf`, D6), speakers (`set_speaker`), band masks per node, the QRT format's missing fields, and a TR shortcut such as `get_channels_38901`.
