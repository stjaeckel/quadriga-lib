+++
title = "Layout CSV files"
summary = "Two CSV files that hold a network layout: the Tx and Rx nodes with their tracks, scenarios and antennas"
order = 3
+++

A network layout is stored in two CSV files of the same format: the Tx file holds the Tx nodes (base stations, access points), the Rx file the Rx nodes (user terminals). Each line is one row of a node: a position with an orientation and the values that apply from there on. Each column is one parameter; only `name`, `x`, `y` and `z` are required, the others are optional and may come in any order. The terms (node, site, terminal, row, band) are those of the section "Network layouts and mobility" of the API pages.

Related functions:

- C++ [layout_csv_read](cpp_api.html#layout_csv_read) and [layout_csv_write](cpp_api.html#layout_csv_write)
- MATLAB / Octave [layout_csv_read](mex_api.html#layout_csv_read) and [layout_csv_write](mex_api.html#layout_csv_write)
- Python [layout.csv_read](python_api.html#layout.csv_read) and [layout.csv_write](python_api.html#layout.csv_write)

## Example {#layout-csv-example}

Tx file with a three-sector macro site and a small cell:

```text
# name and sector identify a node
name,sector,x,y,z,tilt,heading,scenario,antenna,panel,pol
BS1,0,0,0,25,10,30,UMa,3GPP,"4,4,1,2","3,2"
BS1,1,,,,,150,,,,
BS1,2,,,,,270,,,,
BS2,0,200,0,10,0,180,UMi,dipole,,
```

Rx file with a moving UT and a UT on a track:

```text
name,x,y,z,heading,speed,direction_az
UT1,50,20,1.5,,0.833,120
UT2,100,-50,1.5,0,1,
UT2,110,-50,1.5,,,
UT2,110,-40,1.5,90,,
```

- Site `BS1` has three nodes, sectors 0, 1 and 2. Sectors 1 and 2 take the position, tilt, scenario and antenna of sector 0 and set their own heading.
- `BS2` is a single node with a dipole antenna.
- `UT1` is a point drop that moves at 0.833 m/s (3 km/h) toward the azimuth 120°. `UT2` is a track of three rows at 1 m/s; it turns to the heading 90° at the last row. Both UTs have no antenna columns and use the generic antenna.

## Syntax {#layout-csv-syntax}

- Comma-separated UTF-8 text; LF or CRLF line ends; a byte-order mark is skipped.
- The first line that is neither blank nor a comment is the header with the column names. Names are case-sensitive and unique.
- Lines whose first non-blank character is `#` and blank lines are skipped.
- A cell can be quoted with `"`, with `""` for a quote inside; a quoted cell can hold commas and semicolons, but no line break. Blanks around an unquoted cell are removed.
- A line with fewer cells than the header is filled up with empty cells; a line with more cells is an error.
- Numbers are decimal and do not depend on the locale (`25`, `-1.5`, `3.5e9`); `NaN`, `Inf` and `-Inf` are accepted in any case.
- List cells hold numbers separated by commas (`"4,4,1,2"`); matrix cells hold rows separated by semicolons (`"6,0;10,0"` has two rows).
- Units: positions in m, angles in degrees, times in s, speeds in m/s, frequencies in Hz.

## Nodes and rows {#layout-csv-rows}

- The pair (`name`, `sector`) identifies a node. All lines with the same pair are its rows; they need not be consecutive, and their order in the file is the order of the track. Nodes are numbered in the order of their first line.
- Names have 1 to 255 bytes, without comma, semicolon, quote and line break, and without a leading or trailing blank.
- The Tx nodes of one name form a site, the Rx nodes of one name a terminal. They share their positions, motion and scenario row by row; the sectors of a site also share `seed`, `cell`, the clutter and the `scenpar.` columns, the nodes of a terminal `indoor`, `floor` and `d2d_in`.
- An *extra sector* is a node with a sector above 0 of a site that has a sector 0. Its empty cells take the values of sector 0, so it needs only the cells in which it differs, e.g. the heading. An extra sector of a static site has one row and may leave `x`, `y`, `z` empty.
- A node with one row is a point drop, a node with several rows a track.
- A *band row* has a `center_freq` and no position; it adds a band to its node and is not a row of the track (see [Bands](#layout-csv-bands)).

## Empty cells {#layout-csv-empty-cells}

What an empty cell means depends on the class of the column:

| Class | Columns | An empty cell |
| ----- | ------- | ------------- |
| Identity | `name`, `sector` | `name` is required; an empty `sector` is 0 |
| Per row | `x`, `y`, `z`, `time` | is an error for `x`, `y`, `z`, except in the one row of an extra sector of a static site; `time` is given on all rows of a node or on none |
| Held | `bank`, `tilt`, `heading`, `direction_az`, `direction_el`, `speed`, `scenario`, `indoor`, `floor`, `d2d_in` | repeats the value of the previous row of the node; on the first row it is the value of sector 0 for an extra sector, else the default. `NaN` is a value, not an empty cell |
| Fixed | all other columns | takes the value of the first row of the node, for an extra sector that of sector 0. Later rows are empty or repeat the value; a different value is an error |

## Columns {#layout-csv-columns}

| Column | Class | Unit | Meaning |
| ------ | ----- | ---- | ------- |
| `name` | Identity | | Site or terminal name, required |
| `sector` | Identity | | Sector number, a whole number >= 0; default 0 |
| `x`, `y`, `z` | Per row | m | Position, required |
| `time` | Per row | s | Time of the row, strictly increasing along the track; a node has `time` or a `speed`, not both |
| `bank`, `tilt`, `heading` | Held | deg | Orientation of the antenna; a positive tilt points down; default 0 |
| `direction_az`, `direction_el` | Held | deg | Direction of travel of a point drop; empty or NaN azimuth: along the orientation; elevation default 0, positive up; a track takes its direction from its positions |
| `speed` | Held | m/s | Speed from this row on, >= 0; default 0 |
| `scenario` | Held | | Scenario family (e.g. `UMa`) or conf name (e.g. `UMa_NLOS`, or the name of a user conf file, never a path); empty: none. On the Tx file, constant along the rows of a site; on the Rx file, a change starts a new segment |
| `indoor` | Held | | Indoor state: -1 draw (default), 0 outdoor, 1 building, 2 car, 3 residential building |
| `floor` | Held | | Floor number from 1; -1 (default): derived from `z` |
| `d2d_in` | Held | m | Indoor 2D distance, >= 0; NaN (default): drawn |
| `d_clutter`, `r_clutter`, `h_clutter` | Fixed, Tx file | m, 1, m | Clutter size, density and height of the InF scenarios; default: that of the model |
| `scenpar.<key>` | Fixed, Tx file | as the key | Overrides one scenario parameter for the links of the site, e.g. `scenpar.DS_VS` |
| `seed` | Fixed, Tx file | | Seed of the site, >= -1; -1 (default): derived from the layout seed and the site name |
| `layout_seed` | Fixed, Tx file | | Seed of the layout, >= -1; all cells equal |
| `wrap` | Fixed, Tx file | m | Lattice vectors of the wrap-around, one per matrix row: `"ax,ay"` or `"ax,ay;bx,by"`; all cells equal |
| `cell` | Fixed, Tx file | | 0: the site has no drop cells of its own (see `drop_terminals`); default 1 |
| `pairing` | Fixed, Rx file | | Names of the sites linked to the terminal, as a list; empty: all sites |
| `relative` | Fixed | | 1: the orientation is relative to the direction of motion (an antenna mounted on a vehicle); default 0 |
| `center_freq` | Fixed | Hz | Band of the antenna cells of the row, see [Bands](#layout-csv-bands) |
| `antenna`, `qdant_file`, `qdant_id`, the inputs of `generate_antenna` | Fixed | | Antenna of the node, see [Antennas](#layout-csv-antennas) |

`indoor`, `floor` and `d2d_in` change only at rows where `scenario` or `indoor` changes. A column that belongs to the other file, e.g. `pairing` in the Tx file, is an error.

**Unknown columns** are kept as text, reported by the reader and written back unchanged, so that a typo in a column name is visible.

## Antennas {#layout-csv-antennas}

- `antenna` sets the type: a type of `generate_antenna` (`omni`, `xpol`, `dipole`, `3GPP`, `multibeam`, `3GPP-UT`, ...), `qdant` for an array read from a QDANT file, or `none` for no antenna. A node without antenna cells uses the generic antenna, two co-located isotropic elements with vertical and horizontal polarization.
- The other inputs of `generate_antenna` are columns of the same name and meaning: `res`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `separate_beams` and `apply_weights` (0 or 1), `device_size`, `locations`. An empty cell takes the default of `generate_antenna`. A column that the type does not take is an error; `pattern` has no column.
- Tuples are list cells (`panel` `"4,4,1,2"`). `steer` and `beams` are matrix cells with the rows of the matrix that `generate_antenna` takes: `steer` `"12,0"` is one row (tilt, pan), `beams` `"0,30;0,10"` two beams with azimuths 0° and 30° and elevations 0° and 10°.
- `qdant`: `qdant_file` is the file, absolute or relative to the folder of the CSV file, `qdant_id` the 1-based entry (default 1). The array is used as stored, for all bands.
- An extra sector without antenna cells has the antenna of sector 0; `none` gives it no antenna.
- Nodes with equal antenna cells share one antenna of the layout.
- Without a `center_freq`, the antenna is generated once per band at its frequency, so that an array spaced in wavelengths keeps its spacing in each band.
- Large arrays need much memory on the default 1° grid of `generate_antenna`: `3GPP` with `panel` `"16,16,1,4"` holds about 2 GB per band. A `res` column with a coarser grid, or reading without generating the antennas, avoids this.

## Bands {#layout-csv-bands}

A layout can simulate several frequencies, the bands. A node can have one antenna for all bands or one antenna per band.

- The bands of the layout are the `center_freq` values of the Tx file; if it has none, those of the Rx file. Frequencies given to the reader replace them; the values in the files must then match within a relative 1e-6.
- A node without `center_freq` has one antenna description for all bands.
- A node with `center_freq` sets the antenna per band: the first row holds the first band, a band row each further band. A band row has `name`, `sector`, `center_freq` and antenna columns only. The node must set all bands of the layout, also in the Rx file.
- A band row with empty antenna cells repeats the antenna of the first band at its frequency. A band row with antenna cells needs antenna cells in the first band.

```text
name,x,y,z,center_freq,antenna,panel
BS1,0,0,25,3.5e9,3gpp,"8,8,1,2"
BS1,,,,28e9,3gpp,"16,16,1,4"
BS2,500,0,25,3.5e9,3gpp,"8,8,1,2"
BS2,,,,28e9,,
```

`BS1` has its own array per band; `BS2` uses the same design in both bands, with half-wavelength spacing in each. Rx nodes without `center_freq` take both bands.

## Reading {#layout-csv-reading}

- Angles are converted from degrees to radians.
- A relative `qdant_file` is resolved against the folder of the CSV file.
- The reader checks every rule of this page and the validity of the layout. It stops with an error that names the file, the line and the column, e.g. for a value outside its range, an antenna column that the type does not take, a `pairing` that names an unknown site or a site twice, or a terminal whose empty `pairing` cell leaves it without a site farther than 1e-4 m.

## Writing {#layout-csv-writing}

- Only columns with values other than the default are written, in the order of the [Columns](#layout-csv-columns) table, unknown columns last. Angles are in degrees; numbers have up to 15 significant digits, or 16 or 17 where 15 do not read back to the same value. Reading the files back gives the same layout within these digits, except for its name and link states, which the files do not hold.
- In the compact form, cells that the reader would fill in (held values repeated from the previous row, fixed values after the first row) are left empty. An extra sector is written with all its cells in its first row.
- Antennas are written as antenna cells; a `qdant_file` is written relative to the folder of the CSV file. If no antenna cells carry the bands, the first node declares them: `center_freq` on its first row and a band row for each further band.
- An antenna that is an array without antenna cells (e.g. a measured pattern) is written to a QDANT file next to the CSV file, `<stem>_<k>.qdant`, with `<stem>` the name of the CSV file without `.csv` and k = 1, 2, ... (names that `qdant` cells of the layout use are skipped), one file per antenna with one entry per band, and referred to by `qdant` cells. Files `<stem>_<k>.qdant` next to a written CSV file that the new files do not refer to are deleted; files of other names are never touched.
- `seed`, `layout_seed` and `wrap` of the layout go into the Tx file.
- The writer composes all files first and replaces the targets only when all are written, so a failed call leaves the existing files unchanged (only a failed replacement of the second file does not undo that of the first).
- Not expressible, and therefore an error: a scenario that returns to none after a scenario was set, a terminal that `pairing` links to no site (an empty cell means all sites), and a seed or wrap vectors of a layout without Tx nodes. An extra sector without antenna next to a sector 0 with one is written with the antenna `none`.
