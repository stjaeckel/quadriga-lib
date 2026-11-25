// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "qd_csv.hpp"
#include "qd_helpers_shared.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!SECTION_DESC
A **layout** describes the nodes of a radio network and how they move: where each transmitter (Tx) and receiver (Rx) is, which scenario applies there and which antenna it has. It is stored in two flat tables, one per side, and in two CSV files with the same format.

**Terms.** A *node* is one (`name`, `sector`) pair on one side. It has one antenna and one *track*: its *rows*, each a position with orientation and further values that apply from that row on. A node with one row is a *point drop*; it stays at its position, or, with a speed, moves along its direction. A node with several rows is a track and moves along its positions. A *site* is all Tx nodes with one name; its sectors share position and scenario. A *terminal* is all Rx nodes with one name, e.g. the panels of one device. Angles are in degrees in the CSV files and in radians in the classes. The orientation (bank, tilt, heading) is that of the antenna, a positive tilt points down; the optional direction (azimuth, elevation of travel) is that of the movement, and an unset direction follows the orientation.

**CSV files.** [[layout_csv_read]] and [[layout_csv_write]] read and write the files. The first line is the header; the columns are optional, in any order, except `name`, `x`, `y`, `z`. Cells follow the syntax of the other CSV files of the library: quoting with `"`, lines starting with `#` are comments, numbers are decimal (NaN and Inf in any case). Empty cells follow the class of the column:

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
| `pairing` | | Rx side: names of the Tx sites linked to the terminal, empty: all |
| `relative` | | 1: the orientation is relative to the direction of motion |
| `center_freq`, `antenna`, `qdant_file`, `qdant_id` and the inputs of [[generate_antenna]] | | Antenna columns, see below |

**Antennas.** The antenna columns of a node (type `antenna` as in [[generate_antenna]], or `qdant` with the file `qdant_file` relative to the CSV file and its 1-based entry `qdant_id`) are kept as one text line per antenna in `antenna_args`, with the names of the columns in `antenna_cols`. Nodes with equal lines share one entry of the lists, Tx nodes first, and `i_antenna` is the 1-based index of it, 0 for the generic antenna with two co-located isotropic elements of vertical and horizontal polarization. A column that the type does not use is an error. An extra sector without antenna cells has the antenna of sector 0; the type `none` (no other antenna cells) gives a node no antenna, e.g. an extra sector next to a sector 0 with one. The member function `eval_antennas` of [[layout]] generates the arrays `antenna` from the lines: a line without `center_freq` once per simulation frequency, so that an array spaced in wavelengths keeps its spacing in each band (co-located elements give one array for all bands), a line of a band at its frequency, a `qdant` line by reading its entry as stored. [[set_antenna]] sets the antenna of nodes from the arguments of [[generate_antenna]] or from an evaluated array, which [[layout_csv_write]] writes to a QDANT file next to the CSV file.

**Bands.** A row with `center_freq` and no position adds a frequency, and optionally an antenna for it, to its node. The frequencies of the Tx side (else of the Rx side) are the simulation frequencies `center_freq` of the layout. A node without bands has all of them with one antenna description; a node with bands defines all of them. An empty antenna in a band repeats the antenna of the first band.

**Unknown columns** are kept as text in `extra_names` and `extra` and written back unchanged.

**Channels from a layout.** The channel coefficients are computed in three stages. [[init_link_states]] draws the discrete state of each *propagation link*, one site and one terminal: its wrap image, scenario and conf file, LOS and indoor state and seeds, stored in `layout.links`. [[generate_rays]] runs the chain of the section "Geometry-based stochastic model" for chosen links and returns their rays. [[get_gbsm_channels]] applies the antennas, the motion along the tracks and the link gain, for each *channel link*, one Tx node and one Rx node; the sectors of a site share the rays of their propagation link. [[get_channels_layout]] runs all three stages. [[attach_terminals]] chooses the serving Tx node and site of each terminal by the large-scale gain or by RSRP, for `layout.pairing`. All random values are addressed: they follow from the seeds, a stream key and the position, so that the evaluation order, the batches and the thread count do not change them, and only the discrete states are stored. `layout.links` holds a hash of the layout members that determine the links (positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter, scenario overrides, wrap vectors and whether a site moves); stages 2 and 3 throw when it no longer matches, and `layout.links.reset()` discards the links. The links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after another realisation. Editing `los` or `indoor` of the links between stage 1 and stage 2 forces a state; orientations, antennas and `pairing` may change without new links.
SECTION_DESC!*/

/*!MD
# layout
Class for a network layout: Tx and Rx nodes, antennas, frequencies and link states

- `tx` and `rx` are the two sides, each a [[layout_nodes]] object; the nodes of a side may share positions (sectors, panels)
- The antenna lists `antenna_cols`, `antenna_args` and `antenna` are shared by both sides; `i_antenna` of a node indexes them, starting at 1
- Written and read by [[layout_csv_read]] and [[layout_csv_write]]; the model and the file format are described at the top of this section

## Attributes:<br>
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `std::string name` | string | Name of the layout; not stored in the CSV files |
| `layout_nodes tx`, `layout_nodes rx` | | Tx (BS or AP) and Rx (UT or STA) nodes |
| `layout_links links` | | State of the propagation links of the current realisation, see [[layout_links]] |
| `arma::sword seed` | scalar | Seed of the layout, -1 = not set |
| `arma::vec center_freq` | `[n_freq]` | Simulation frequencies in [Hz], strictly ascending |
| `std::vector<std::string> antenna_cols` | `[n_col]` | Names of the antenna columns of the CSV files |
| `std::vector<std::vector<std::string>> antenna_args` | `[n_antenna][0, 1 or n_freq]` | CSV lines of the antenna columns: one line for all bands or one per band; no line or an empty line = an evaluated array without description; a line without antenna type = a band that is not set yet |
| `std::vector<std::vector<arrayant<double>>> antenna` | `[n_antenna][0, 1 or n_freq]` | Evaluated arrays, one for all bands or one per band; an entry without arrays or an array without elements is not evaluated, its nodes use the generic antenna; empty = not evaluated |
| `arma::mat wrap` | `[2, 1]` or `[2, 2]` | Lattice vectors in [m], one per column; two vectors form a reduced basis, 0 <= a1·a2 <= min(a1·a1, a2·a2) / 2, so that ±a1, ±a2 and ±(a1 − a2) are the nearest images; empty = none |
| `arma::umat pairing` | `[2, n_pair]` | Tx site index and Rx terminal index of the linked pairs, 0-based; empty = all pairs |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `.n_antenna()` | Number of antennas: length of `antenna_args`, or of `antenna` if there are no arguments |
| `.eval_antennas(center_freq)` | Generates the arrays of `antenna` that are not evaluated yet from `antenna_args`; a frequency list in [Hz] (optional) replaces `center_freq` and regenerates all lines; throws for a band without description and for an array spaced in wavelengths without a frequency |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks all members of both sides and `links`, the sizes and ranges, equal positions, motion (`speed`, `time`, `direction`) and shared values of the nodes of one site or terminal, the antenna lists (a node with a band that is not set is reported), the frequencies, `wrap` and `pairing` |

## See also:
- [[layout_nodes]] (one side of a layout)
- [[layout_links]] (link states)
- [[layout_csv_read]], [[layout_csv_write]] (load and store a layout)
- [[set_antenna]] (sets the antennas of nodes)
MD!*/

/*!MD
# layout_nodes
One side of a layout: nodes with their rows, in flat arrays

- The nodes are in order of first appearance; the rows of node `i` are `offset(i)` to `offset(i+1)-1` (the last node to the end), so each node has at least one row
- Optional members may be empty with the meaning given below; `is_valid()` accepts both forms

## Attributes:<br>
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
| `std::vector<std::string> extra_names` | `[n_extra]` | Names of unknown CSV columns |
| `std::vector<std::vector<std::string>> extra` | `[n_extra][n_row]` | Their cells as text |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `.n_node()` | Number of nodes |
| `.n_row()` | Number of rows of all nodes |
| `.n_row(i_node)` | Number of rows of one node (0-based, throws `std::invalid_argument` for an index outside the nodes) |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks sizes, `offset`, finite positions and orientations, names, ranges and the rules for `time`, `speed`, `direction`, `floor` and `d2d_in` |

## See also:
- [[layout]] (the class that holds both sides)
- [[layout_csv_read]] (fills the members from a CSV file)
MD!*/

// ---- The column table ----

namespace qd_layout
{
static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
static constexpr double inf = std::numeric_limits<double>::infinity();
static constexpr double deg = 0.017453292519943295769236907684886; // pi / 180

// Order of the entries = order of 'col_id' = order of the columns in a written file
static const column table[n_col] = {
    // name, class, kind, side, scale, default, lo, hi, antenna_types
    {"name", col_class::identity, col_kind::text, col_side::both, 1.0, 0.0, -inf, inf, 0},
    {"sector", col_class::identity, col_kind::integer, col_side::both, 1.0, 0.0, 0.0, 4294967295.0, 0},
    {"x", col_class::per_row, col_kind::number, col_side::both, 1.0, 0.0, -inf, inf, 0},
    {"y", col_class::per_row, col_kind::number, col_side::both, 1.0, 0.0, -inf, inf, 0},
    {"z", col_class::per_row, col_kind::number, col_side::both, 1.0, 0.0, -inf, inf, 0},
    {"time", col_class::per_row, col_kind::number, col_side::both, 1.0, nan, -inf, inf, 0},
    {"bank", col_class::held, col_kind::number, col_side::both, deg, 0.0, -inf, inf, 0},
    {"tilt", col_class::held, col_kind::number, col_side::both, deg, 0.0, -inf, inf, 0},
    {"heading", col_class::held, col_kind::number, col_side::both, deg, 0.0, -inf, inf, 0},
    {"direction_az", col_class::held, col_kind::number, col_side::both, deg, nan, -inf, inf, 0},
    {"direction_el", col_class::held, col_kind::number, col_side::both, deg, 0.0, -inf, inf, 0},
    {"speed", col_class::held, col_kind::number, col_side::both, 1.0, 0.0, 0.0, inf, 0},
    {"scenario", col_class::held, col_kind::text, col_side::both, 1.0, 0.0, -inf, inf, 0},
    {"indoor", col_class::held, col_kind::integer, col_side::both, 1.0, -1.0, -1.0, 3.0, 0},
    {"floor", col_class::held, col_kind::integer, col_side::both, 1.0, -1.0, -1.0, 2147483647.0, 0},
    {"d2d_in", col_class::held, col_kind::number, col_side::both, 1.0, nan, 0.0, inf, 0},
    {"d_clutter", col_class::fixed, col_kind::number, col_side::tx, 1.0, nan, -inf, inf, 0},
    {"r_clutter", col_class::fixed, col_kind::number, col_side::tx, 1.0, nan, -inf, inf, 0},
    {"h_clutter", col_class::fixed, col_kind::number, col_side::tx, 1.0, nan, -inf, inf, 0},
    {"scenpar.", col_class::fixed, col_kind::number, col_side::tx, 1.0, nan, -inf, inf, 0},
    {"wrap", col_class::fixed, col_kind::matrix, col_side::tx, 1.0, nan, -inf, inf, 0},
    {"seed", col_class::fixed, col_kind::integer, col_side::tx, 1.0, -1.0, -1.0, inf, 0},
    {"layout_seed", col_class::fixed, col_kind::integer, col_side::tx, 1.0, -1.0, -1.0, inf, 0},
    {"pairing", col_class::fixed, col_kind::names, col_side::rx, 1.0, 0.0, -inf, inf, 0},
    {"relative", col_class::fixed, col_kind::integer, col_side::both, 1.0, 0.0, 0.0, 1.0, 0},
    // Antenna columns; the types are the groups of generate_antenna that use the input
    {"center_freq", col_class::fixed, col_kind::number, col_side::both, 1.0, nan, 0.0, inf, t_all},
    {"antenna", col_class::fixed, col_kind::text, col_side::both, 1.0, 0.0, -inf, inf, t_all},
    {"qdant_file", col_class::fixed, col_kind::text, col_side::both, 1.0, 0.0, -inf, inf, t_qdant},
    {"qdant_id", col_class::fixed, col_kind::integer, col_side::both, 1.0, 1.0, 1.0, 2147483647.0, t_qdant},
// The arguments of generate_antenna (a flag is 0 or 1)
#define QD_LAYOUT_X(name, kind, types) {#name, col_class::fixed, col_kind::kind, col_side::both, 1.0, nan, -inf, inf, types},
    QD_LAYOUT_ANTENNA_ARGS(QD_LAYOUT_X)
#undef QD_LAYOUT_X
};

const column &column_of(int id) { return table[id]; }

int find_column(std::string_view name)
{
    if (name.size() > 8 && name.substr(0, 8) == "scenpar.")
        return c_scenpar;
    for (int i = 0; i < n_col; ++i)
        if (i != c_scenpar && name == table[i].name)
            return i;
    return -1;
}

unsigned antenna_type_mask(const std::string &type)
{
    if (type == "omni" || type == "xpol" || type == "dipole" || type == "short-dipole" || type == "half-wave-dipole")
        return t_simple;
    if (type == "custom")
        return t_custom;
    if (type == "ula")
        return t_ula;
    if (type == "3GPP" || type == "3gpp")
        return t_3gpp;
    if (type == "multibeam")
        return t_multibeam;
    if (type == "3GPP-UT" || type == "3gpp-ut" || type == "3GPP-CPE" || type == "3gpp-cpe")
        return t_device;
    if (type == "qdant")
        return t_qdant;
    if (type == "none")
        return t_none;
    return 0;
}

std::vector<std::string> split_antenna_line(const std::string &line, size_t n_cols)
{
    std::vector<std::string> cells(n_cols);
    if (n_cols == 0)
        return cells;
    std::string text;
    for (size_t c = 0; c < n_cols; ++c)
        text += (c == 0 ? "c" : ",c") + std::to_string(c);
    text += '\n' + line + '\n';
    const qd_csv::table T = qd_csv::parse(std::move(text), "antenna line");
    if (T.n_rows() != 0)
        for (size_t c = 0; c < n_cols; ++c)
            cells[c] = std::string(T.cell(0, c));
    return cells;
}

antenna_cells parse_antenna_line(const std::string &line, const std::vector<std::string> &cols)
{
    antenna_cells a;
    const std::vector<std::string> cells = split_antenna_line(line, cols.size());
    for (size_t k = 0; k < cols.size(); ++k)
    {
        const int id = find_column(cols[k]);
        if (id >= c_center_freq)
            a[(size_t)(id - c_center_freq)] = cells[k];
    }
    return a;
}

std::string join_antenna_line(const antenna_cells &cells, const std::vector<std::string> &cols)
{
    std::string line;
    for (size_t k = 0; k < cols.size(); ++k)
    {
        const int id = find_column(cols[k]);
        line += (k == 0 ? "" : ",") + qd_csv::format_cell(id >= c_center_freq ? cells[(size_t)(id - c_center_freq)] : std::string());
    }
    return line;
}

std::vector<std::string> used_antenna_cols(const std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                                           const std::vector<antenna_cells> &extra)
{
    bool used[n_antenna_col] = {};
    auto mark = [&](const antenna_cells &a)
    {
        for (int k = 0; k < n_antenna_col; ++k)
            used[k] = used[k] || !a[(size_t)k].empty();
    };
    for (const auto &lines : args)
        for (const std::string &line : lines)
            if (!line.empty())
                mark(parse_antenna_line(line, cols));
    for (const antenna_cells &a : extra)
        mark(a);
    std::vector<std::string> out;
    for (int k = 0; k < n_antenna_col; ++k)
        if (used[k])
            out.push_back(table[c_center_freq + k].name);
    return out;
}

void rewrite_antenna_lines(std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                           const std::vector<std::string> &new_cols)
{
    if (cols == new_cols)
        return;
    for (auto &lines : args)
        for (std::string &line : lines)
            if (!line.empty())
                line = join_antenna_line(parse_antenna_line(line, cols), new_cols);
}

double cell_number(const qd_csv::table &T, size_t row, size_t col, int id)
{
    const column &C = table[id];
    const bool nan_ok = std::isnan(C.dflt) || id == c_scenpar;
    const double v = T.get_double(row, col, 0.0, nan_ok, id == c_scenpar);
    const double lo = C.kind == col_kind::flag ? 0.0 : C.lo, hi = C.kind == col_kind::flag ? 1.0 : C.hi;
    if ((C.kind == col_kind::integer || C.kind == col_kind::flag) && v != std::floor(v))
        throw std::runtime_error(T.where(row, col) + " must be an integer ('" + std::string(T.cell(row, col)) + "').");
    if (!std::isnan(v) && (v < lo || v > hi))
        throw std::runtime_error(T.where(row, col) + " is outside the allowed range ('" + std::string(T.cell(row, col)) + "').");
    return v * C.scale;
}

long long cell_int64(const qd_csv::table &T, size_t row, size_t col, int id)
{
    const std::string text(T.cell(row, col)); // The cell is trimmed; strtoll needs the end of the text
    char *end = nullptr;
    errno = 0;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || end != text.c_str() + text.size() || text.find_first_of(" \t") != std::string::npos)
        throw std::runtime_error(T.where(row, col) + " must be an integer ('" + text + "').");
    if (errno == ERANGE || v < (long long)table[id].lo)
        throw std::runtime_error(T.where(row, col) + " is outside the allowed range ('" + text + "').");
    return v;
}

// Throws unless the text of an antenna cell has the syntax of its kind
static void qd_check_antenna_cell(const qd_csv::table &T, size_t r, size_t c, int id)
{
    switch (table[id].kind)
    {
    case col_kind::number:
    case col_kind::integer:
    case col_kind::flag:
        cell_number(T, r, c, id);
        break;
    case col_kind::list:
        T.get_list(r, c);
        break;
    case col_kind::int_list:
        for (double v : T.get_list(r, c, false, false))
            if (v != std::floor(v) || v < 0.0 || v > 4294967295.0)
                throw std::runtime_error(T.where(r, c) + " must have non-negative integer entries.");
        break;
    case col_kind::matrix:
        T.get_matrix(r, c);
        break;
    default:
        break;
    }
}

void check_antenna_cells(const qd_csv::table &T, size_t r, const int *ci)
{
    auto cell = [&](int id) { return ci[id] < 0 ? std::string_view() : T.cell(r, (size_t)ci[id]); };
    if (!cell(c_center_freq).empty())
        cell_number(T, r, (size_t)ci[c_center_freq], c_center_freq);
    bool any = false;
    for (int id = c_antenna; id < n_col; ++id)
        any = any || !cell(id).empty();
    if (!any)
        return;
    const std::string type(cell(c_antenna));
    const std::string where_row = T.source + ", line " + std::to_string(T.line(r));
    if (type.empty())
        throw std::runtime_error(where_row + ": antenna cells are set but the antenna type ('antenna') is empty.");
    const unsigned mask = antenna_type_mask(type);
    if (mask == 0)
        throw std::runtime_error(T.where(r, (size_t)ci[c_antenna]) + " has an unknown antenna type ('" + type + "').");
    for (int id = c_qdant_file; id < n_col; ++id)
    {
        if (cell(id).empty())
            continue;
        if ((table[id].antenna_types & mask) == 0)
            throw std::runtime_error(T.where(r, (size_t)ci[id]) + " is not used by the antenna type '" + type + "'.");
        qd_check_antenna_cell(T, r, (size_t)ci[id], id);
    }
    if (mask == t_qdant && cell(c_qdant_file).empty())
        throw std::runtime_error(T.where(r, (size_t)ci[c_antenna]) + " is the antenna type 'qdant', which needs a 'qdant_file'.");
}

// ---- Member mapping ----

std::vector<size_t> site_of_node(const quadriga_lib::layout_nodes &nodes, size_t *n_site)
{
    std::unordered_map<std::string, size_t> index;
    std::vector<size_t> site(nodes.name.size());
    for (size_t i = 0; i < nodes.name.size(); ++i)
        site[i] = index.emplace(nodes.name[i], index.size()).first->second;
    if (n_site != nullptr)
        *n_site = index.size();
    return site;
}

flat flatten(const quadriga_lib::layout_nodes &N)
{
    flat F;
    F.n_node = N.name.size(), F.n_row = N.pos.n_cols;
    F.name = N.name;
    F.offset.assign(N.offset.begin(), N.offset.end());

    auto per_node = [&](int id, size_t n) { F.v[id].assign(n, table[id].dflt); };
    per_node(c_sector, F.n_node);
    for (size_t i = 0; i < F.n_node && i < N.sector.n_elem; ++i)
        F.v[c_sector][i] = (double)N.sector[i];

    const size_t R = F.n_row;
    for (int id : {c_x, c_y, c_z})
        F.v[id].resize(R);
    for (int id : {c_bank, c_tilt, c_heading})
        F.v[id].resize(R);
    for (size_t r = 0; r < R; ++r)
    {
        for (int k = 0; k < 3; ++k)
        {
            F.v[c_x + k][r] = N.pos(k, r);
            F.v[c_bank + k][r] = N.orientation.n_cols == R && N.orientation.n_rows == 3 ? N.orientation(k, r) : 0.0;
        }
    }
    if (N.direction.n_cols == R && N.direction.n_rows == 2 && R != 0)
    {
        F.v[c_direction_az].resize(R), F.v[c_direction_el].resize(R);
        for (size_t r = 0; r < R; ++r)
            F.v[c_direction_az][r] = N.direction(0, r), F.v[c_direction_el][r] = N.direction(1, r);
    }
    auto copy_vec = [&](int id, const arma::vec &x) { if (x.n_elem == R && R != 0) F.v[id].assign(x.begin(), x.end()); };
    copy_vec(c_speed, N.speed), copy_vec(c_time, N.time), copy_vec(c_d2d_in, N.d2d_in);
    if (N.indoor.n_elem == R && R != 0)
        F.v[c_indoor].assign(N.indoor.begin(), N.indoor.end());
    if (N.floor.n_elem == R && R != 0)
        F.v[c_floor].assign(N.floor.begin(), N.floor.end());
    if (N.scenario.size() == R && R != 0)
        F.txt[c_scenario] = N.scenario;

    if (N.clutter.n_cols == F.n_node && N.clutter.n_rows == 3 && F.n_node != 0)
        for (int k = 0; k < 3; ++k)
            F.v[c_d_clutter + k].assign(N.clutter.row(k).begin(), N.clutter.row(k).end());
    for (const auto &kv : N.scenpar_override)
        F.scenpar[kv.first].assign(kv.second.begin(), kv.second.end());
    if (N.seed.n_elem == F.n_node && F.n_node != 0)
        F.seed.assign(N.seed.begin(), N.seed.end());
    if (N.relative.n_elem == F.n_node && F.n_node != 0)
        F.v[c_relative].assign(N.relative.begin(), N.relative.end());
    F.extra_names = N.extra_names;
    F.extra = N.extra;
    return F;
}

// True if the array has a value other than the default
static bool has_value(const std::vector<double> &v, double dflt)
{
    for (double x : v)
        if (!same_value(x, dflt))
            return true;
    return false;
}

quadriga_lib::layout_nodes unflatten(const flat &F)
{
    quadriga_lib::layout_nodes N;
    const size_t n = F.n_node, R = F.n_row;
    N.name = F.name;
    N.sector.set_size(n), N.i_antenna.zeros(n), N.offset.set_size(n);
    for (size_t i = 0; i < n; ++i)
    {
        N.sector[i] = F.v[c_sector].size() == n ? (unsigned)F.v[c_sector][i] : 0u;
        N.offset[i] = F.offset[i];
    }
    N.pos.set_size(3, R), N.orientation.set_size(3, R);
    for (size_t r = 0; r < R; ++r)
        for (int k = 0; k < 3; ++k)
        {
            N.pos(k, r) = F.v[c_x + k].size() == R ? F.v[c_x + k][r] : 0.0;
            N.orientation(k, r) = F.v[c_bank + k].size() == R ? F.v[c_bank + k][r] : 0.0;
        }

    auto given = [&](int id, size_t len) { return F.v[id].size() == len && len != 0 && has_value(F.v[id], table[id].dflt); };
    auto to_vec = [&](int id) { return arma::vec(std::vector<double>(F.v[id])); };

    // Direction: stored only if some azimuth is given
    if (F.v[c_direction_az].size() == R && R != 0 && has_value(F.v[c_direction_az], nan))
    {
        N.direction.set_size(2, R);
        for (size_t r = 0; r < R; ++r)
        {
            N.direction(0, r) = F.v[c_direction_az][r];
            N.direction(1, r) = F.v[c_direction_el].size() == R ? F.v[c_direction_el][r] : 0.0;
        }
    }
    if (given(c_speed, R))
        N.speed = to_vec(c_speed);
    if (given(c_time, R))
        N.time = to_vec(c_time);
    if (given(c_d2d_in, R))
        N.d2d_in = to_vec(c_d2d_in);
    if (given(c_indoor, R))
        N.indoor = arma::conv_to<arma::s32_vec>::from(to_vec(c_indoor));
    if (given(c_floor, R))
        N.floor = arma::conv_to<arma::s32_vec>::from(to_vec(c_floor));
    if (F.txt[c_scenario].size() == R && R != 0 &&
        std::any_of(F.txt[c_scenario].begin(), F.txt[c_scenario].end(), [](const std::string &s) { return !s.empty(); }))
        N.scenario = F.txt[c_scenario];

    bool clutter = false;
    for (int k = 0; k < 3; ++k)
        clutter = clutter || given(c_d_clutter + k, n);
    if (clutter)
    {
        N.clutter.set_size(3, n);
        for (int k = 0; k < 3; ++k)
            for (size_t i = 0; i < n; ++i)
                N.clutter(k, i) = F.v[c_d_clutter + k].size() == n ? F.v[c_d_clutter + k][i] : nan;
    }
    for (const auto &kv : F.scenpar)
        if (kv.second.size() == n && has_value(kv.second, nan))
            N.scenpar_override[kv.first] = arma::vec(std::vector<double>(kv.second));
    if (std::any_of(F.seed.begin(), F.seed.end(), [](long long s) { return s != -1; }) && F.seed.size() == n)
        N.seed = arma::s64_vec(std::vector<long long>(F.seed));
    if (given(c_relative, n))
        N.relative = arma::conv_to<arma::u32_vec>::from(to_vec(c_relative));
    N.extra_names = F.extra_names;
    N.extra = F.extra;
    return N;
}

} // namespace qd_layout

// ---- Validation ----

arma::uword quadriga_lib::layout_nodes::n_node() const { return (arma::uword)name.size(); }
arma::uword quadriga_lib::layout_nodes::n_row() const { return pos.n_cols; }
arma::uword quadriga_lib::layout_nodes::n_row(arma::uword i_node) const
{
    if (i_node >= offset.n_elem)
        throw std::invalid_argument("Input 'i_node' is out of range.");
    return (i_node + 1 < offset.n_elem ? offset[i_node + 1] : pos.n_cols) - offset[i_node];
}

std::string quadriga_lib::layout_nodes::is_valid() const
{
    const arma::uword nn = (arma::uword)name.size(), nr = pos.n_cols;
    if (sector.n_elem != nn || i_antenna.n_elem != nn || offset.n_elem != nn)
        return "Members 'name', 'sector', 'i_antenna' and 'offset' must have one entry per node.";
    if (pos.n_rows != 3 && !pos.is_empty())
        return "Member 'pos' must have 3 rows.";
    if ((orientation.n_rows != 3 && !orientation.is_empty()) || orientation.n_cols != nr)
        return "Member 'orientation' must have 3 rows and one column per row of 'pos'.";
    if (!direction.is_empty() && (direction.n_rows != 2 || direction.n_cols != nr))
        return "Member 'direction' must be empty or have 2 rows and one column per row of 'pos'.";
    if ((!speed.is_empty() && speed.n_elem != nr) || (!time.is_empty() && time.n_elem != nr) ||
        (!d2d_in.is_empty() && d2d_in.n_elem != nr) || (!indoor.is_empty() && indoor.n_elem != nr) ||
        (!floor.is_empty() && floor.n_elem != nr) || (!scenario.empty() && scenario.size() != nr))
        return "Members 'speed', 'time', 'scenario', 'indoor', 'floor' and 'd2d_in' must be empty or have one entry per row.";
    if (!clutter.is_empty() && (clutter.n_rows != 3 || clutter.n_cols != nn))
        return "Member 'clutter' must be empty or have 3 rows and one column per node.";
    for (const auto &kv : scenpar_override)
        if (kv.second.n_elem != nn)
            return "Member 'scenpar_override' must have one entry per node for the key '" + kv.first + "'.";
    if ((!seed.is_empty() && seed.n_elem != nn) || (!relative.is_empty() && relative.n_elem != nn))
        return "Members 'seed' and 'relative' must be empty or have one entry per node.";
    if (extra.size() != extra_names.size())
        return "Members 'extra_names' and 'extra' must have the same length.";
    for (const auto &col : extra)
    {
        if (col.size() != nr)
            return "Each entry of 'extra' must have one cell per row.";
        for (const std::string &s : col)
            if (s.find_first_of("\r\n") != std::string::npos)
                return "The cells of 'extra' must not contain a line break.";
    }
    for (size_t e = 0; e < extra_names.size(); ++e)
        if (qd_layout::find_column(extra_names[e]) >= 0 || extra_names[e].find_first_of("\r\n") != std::string::npos ||
            std::count(extra_names.begin(), extra_names.end(), extra_names[e]) > 1)
            return "Member 'extra_names' must not repeat a name or a column of the format, and must not contain a line break ('" + extra_names[e] + "').";
    for (const auto &kv : scenpar_override)
        if (kv.first.empty() || kv.first.find_first_of("\r\n") != std::string::npos)
            return "Member 'scenpar_override' must not have an empty key or a key with a line break.";

    // Rows of the nodes
    if (nn == 0 && nr != 0)
        return "Member 'pos' has rows but there are no nodes.";
    for (arma::uword i = 0; i < nn; ++i)
        if ((i == 0 && offset[0] != 0) || (i > 0 && offset[i] <= offset[i - 1]) || offset[i] >= nr)
            return "Member 'offset' must start at 0, increase strictly and stay below the number of rows.";
    if (!pos.is_finite())
        return "Member 'pos' must be finite.";
    if (!orientation.is_finite())
        return "Member 'orientation' must be finite.";

    // Names
    std::unordered_set<std::string> seen;
    for (arma::uword i = 0; i < nn; ++i)
    {
        if (name[i].empty() || name[i].size() > 255)
            return "Node names must have 1 to 255 bytes.";
        if (name[i].find_first_of(",;\"\r\n") != std::string::npos || qd_csv::is_blank(name[i].front()) || qd_csv::is_blank(name[i].back()))
            return "Node names must not contain a comma, semicolon, quote or line break, and must not start or end with a blank ('" + name[i] + "').";
        if (!seen.insert(std::to_string(sector[i]) + ":" + name[i]).second)
            return "The pair of name '" + name[i] + "' and sector " + std::to_string(sector[i]) + " is used by two nodes.";
    }

    // Values per row
    if (!direction.is_empty())
        for (arma::uword i = 0; i < direction.n_elem; ++i)
            if (std::isinf(direction[i]))
                return "Member 'direction' must not be infinite.";
    for (arma::uword r = 0; r < speed.n_elem; ++r)
        if (!std::isfinite(speed[r]) || speed[r] < 0.0)
            return "Member 'speed' must be finite and >= 0.";
    for (arma::uword r = 0; r < time.n_elem; ++r)
        if (std::isinf(time[r]))
            return "Member 'time' must be finite or NaN.";
    for (arma::uword r = 0; r < d2d_in.n_elem; ++r)
        if (std::isinf(d2d_in[r]) || d2d_in[r] < 0.0)
            return "Member 'd2d_in' must be NaN or >= 0.";
    for (arma::uword r = 0; r < indoor.n_elem; ++r)
        if (indoor[r] < -1 || indoor[r] > 3)
            return "Member 'indoor' must be -1, 0, 1, 2 or 3.";
    for (arma::uword r = 0; r < floor.n_elem; ++r)
        if (floor[r] != -1 && floor[r] < 1)
            return "Member 'floor' must be -1 or >= 1.";
    for (arma::uword r = 0; r < scenario.size(); ++r)
        if (scenario[r].find_first_of("/\\\r\n") != std::string::npos)
            return "Member 'scenario' must not contain a path separator or a line break ('" + scenario[r] + "').";
    for (arma::uword i = 0; i < seed.n_elem; ++i)
        if (seed[i] < -1)
            return "Member 'seed' must be >= -1.";
    for (arma::uword i = 0; i < relative.n_elem; ++i)
        if (relative[i] > 1)
            return "Member 'relative' must be 0 or 1.";

    // Per node
    for (arma::uword i = 0; i < nn; ++i)
    {
        const arma::uword first = offset[i], n = n_row(i), last = first + n;
        const std::string who = "Node '" + name[i] + "' (sector " + std::to_string(sector[i]) + "): ";

        bool has_time = false, has_nan_time = false, has_speed = false;
        for (arma::uword r = first; r < last && !time.is_empty(); ++r)
        {
            (std::isnan(time[r]) ? has_nan_time : has_time) = true;
            if (r > first && !std::isnan(time[r]) && !std::isnan(time[r - 1]) && !(time[r] > time[r - 1]))
                return who + "'time' must increase strictly.";
        }
        if (has_time && has_nan_time)
            return who + "'time' must be given on all rows or on none.";
        for (arma::uword r = first; r < last && !speed.is_empty(); ++r)
            has_speed = has_speed || speed[r] != 0.0;
        if (has_time && has_speed)
            return who + "must have either 'time' or 'speed', not both.";

        if (n == 1 && !relative.is_empty() && relative[i] == 1 && (direction.is_empty() || std::isnan(direction(0, first))))
            return who + "'relative' = 1 needs a 'direction' on a node with one row.";
        if (n > 1)
        {
            for (arma::uword r = first; r < last && !direction.is_empty(); ++r)
                if (!std::isnan(direction(0, r)))
                    return who + "a track must not have a 'direction'.";
            if (!has_time && has_speed)
                for (arma::uword r = first; r + 1 < last; ++r)
                    if (speed[r] == 0.0 && (pos(0, r) != pos(0, r + 1) || pos(1, r) != pos(1, r + 1) || pos(2, r) != pos(2, r + 1)))
                        return who + "rows " + std::to_string(r - first) + " and " + std::to_string(r - first + 1) +
                               " are at different positions, but 'speed' is 0 and there is no 'time'.";

            // The floor and the indoor distance change only where a segment starts
            for (arma::uword r = first + 1; r < last; ++r)
            {
                const bool same_segment = (scenario.empty() || scenario[r] == scenario[r - 1]) && (indoor.is_empty() || indoor[r] == indoor[r - 1]);
                if (same_segment && ((!floor.is_empty() && floor[r] != floor[r - 1]) ||
                                     (!d2d_in.is_empty() && !qd_layout::same_value(d2d_in[r], d2d_in[r - 1]))))
                    return who + "'floor' and 'd2d_in' can change only where 'scenario' or 'indoor' changes.";
            }
        }
    }
    return "";
}

// ---- Layout ----

arma::uword quadriga_lib::layout::n_antenna() const
{
    return antenna_args.empty() ? (arma::uword)antenna.size() : (arma::uword)antenna_args.size();
}

namespace
{
// Equal positions, and for a site also equal values of the shared members, row by row, of all nodes with one name
std::string qd_check_groups(const quadriga_lib::layout_nodes &N, bool is_site)
{
    size_t n_group = 0;
    const std::vector<size_t> group = qd_layout::site_of_node(N, &n_group);
    std::vector<size_t> first(n_group, (size_t)-1);
    for (size_t i = 0; i < group.size(); ++i)
    {
        if (first[group[i]] == (size_t)-1)
        {
            first[group[i]] = i;
            continue;
        }
        const size_t a = first[group[i]];
        const std::string who = "Nodes '" + N.name[i] + "' (sectors " + std::to_string(N.sector[a]) + " and " + std::to_string(N.sector[i]) + "): ";
        const arma::uword na = N.n_row(a), nb = N.n_row(i), oa = N.offset[a], ob = N.offset[i];
        if (na != nb)
            return who + "must have the same number of rows.";
        for (arma::uword r = 0; r < na; ++r)
        {
            if (N.pos(0, oa + r) != N.pos(0, ob + r) || N.pos(1, oa + r) != N.pos(1, ob + r) || N.pos(2, oa + r) != N.pos(2, ob + r))
                return who + "must have equal positions in each row.";
            // The first node stands for the motion of all nodes with its name
            const bool diff_speed = !N.speed.is_empty() && N.speed[oa + r] != N.speed[ob + r];
            const bool diff_time = !N.time.is_empty() && !qd_layout::same_value(N.time[oa + r], N.time[ob + r]);
            const bool diff_dir = !N.direction.is_empty() && (!qd_layout::same_value(N.direction(0, oa + r), N.direction(0, ob + r)) ||
                                                              !qd_layout::same_value(N.direction(1, oa + r), N.direction(1, ob + r)));
            if (diff_speed || diff_time || diff_dir)
                return who + "must have equal 'speed', 'time' and 'direction' in each row.";
            const bool diff_scenario = !N.scenario.empty() && N.scenario[oa + r] != N.scenario[ob + r];
            const bool diff_indoor = !N.indoor.is_empty() && N.indoor[oa + r] != N.indoor[ob + r];
            const bool diff_floor = !N.floor.is_empty() && N.floor[oa + r] != N.floor[ob + r];
            const bool diff_d2d = !N.d2d_in.is_empty() && !qd_layout::same_value(N.d2d_in[oa + r], N.d2d_in[ob + r]);
            if (diff_scenario || (!is_site && (diff_indoor || diff_floor || diff_d2d)))
                return who + (is_site ? "must have equal 'scenario'." : "must have equal 'scenario', 'indoor', 'floor' and 'd2d_in'.");
        }
        if (is_site)
        {
            for (int k = 0; k < 3 && !N.clutter.is_empty(); ++k)
                if (!qd_layout::same_value(N.clutter(k, a), N.clutter(k, i)))
                    return who + "must have equal 'clutter'.";
            for (const auto &kv : N.scenpar_override)
                if (!qd_layout::same_value(kv.second[a], kv.second[i]))
                    return who + "must have equal 'scenpar_override' for the key '" + kv.first + "'.";
            if (!N.seed.is_empty() && N.seed[a] != N.seed[i])
                return who + "must have equal 'seed'.";
        }
    }
    return "";
}
} // namespace

std::string quadriga_lib::layout::is_valid() const
{
    std::string msg = tx.is_valid();
    if (!msg.empty())
        return "tx: " + msg;
    msg = rx.is_valid();
    if (!msg.empty())
        return "rx: " + msg;
    msg = links.is_valid();
    if (!msg.empty())
        return "links: " + msg;
    if (!(msg = qd_check_groups(tx, true)).empty())
        return "tx: " + msg;
    if (!(msg = qd_check_groups(rx, false)).empty())
        return "rx: " + msg;

    // The scenario of a Tx site is constant along its rows
    for (arma::uword i = 0; i < tx.n_node(); ++i)
        for (arma::uword r = tx.offset[i] + 1; r < tx.offset[i] + tx.n_row(i) && !tx.scenario.empty(); ++r)
            if (tx.scenario[r] != tx.scenario[r - 1])
                return "tx: Node '" + tx.name[i] + "': the scenario of a Tx site must be constant along its rows.";

    // Antennas
    const arma::uword n_ant = n_antenna(), n_freq = center_freq.n_elem;
    for (arma::uword i = 0; i < tx.i_antenna.n_elem; ++i)
        if (tx.i_antenna[i] > n_ant)
            return "tx: Member 'i_antenna' is larger than the number of antennas.";
    for (arma::uword i = 0; i < rx.i_antenna.n_elem; ++i)
        if (rx.i_antenna[i] > n_ant)
            return "rx: Member 'i_antenna' is larger than the number of antennas.";
    if (!antenna_args.empty() && !antenna.empty() && antenna_args.size() != antenna.size())
        return "Members 'antenna' and 'antenna_args' must have the same length.";
    for (arma::uword i = 0; i < center_freq.n_elem; ++i)
        if (!std::isfinite(center_freq[i]) || center_freq[i] <= 0.0 || (i > 0 && !(center_freq[i] > center_freq[i - 1])))
            return "Member 'center_freq' must be finite, positive and strictly ascending.";
    for (const auto &lines : antenna)
        if (!lines.empty() && lines.size() != 1 && lines.size() != n_freq)
            return "Each entry of 'antenna' must have 0, 1 or n_freq arrays.";
    {
        std::unordered_set<std::string> cols;
        for (const std::string &c : antenna_cols)
        {
            const int id = qd_layout::find_column(c);
            if (id < qd_layout::c_center_freq || !cols.insert(c).second)
                return "Member 'antenna_cols' must hold names of antenna columns, each once ('" + c + "').";
        }
    }
    // The lines of an antenna are valid for their types, as the reader checks the antenna columns; one line has no
    // frequency, n_freq lines carry the frequencies of center_freq; an empty line is an evaluated array, a line
    // without type a band without description
    int ci[qd_layout::n_col];
    std::fill(ci, ci + qd_layout::n_col, -1);
    std::string header;
    for (size_t j = 0; j < antenna_cols.size(); ++j)
        ci[qd_layout::find_column(antenna_cols[j])] = (int)j, header += (j == 0 ? "" : ",") + antenna_cols[j];
    auto user = [&](size_t a) // The first node that uses antenna a (0-based), for the messages
    {
        for (const layout_nodes *N : {&tx, &rx})
            for (arma::uword i = 0; i < N->i_antenna.n_elem; ++i)
                if (N->i_antenna[i] == a + 1)
                    return std::string(N == &tx ? "tx: " : "rx: ") + "Node '" + N->name[i] + "' (sector " + std::to_string(N->sector[i]) + "): its antenna ";
        return std::string("Member 'antenna_args': an entry that no node uses ");
    };
    auto has_arrays = [&](size_t a) // The evaluated arrays of an antenna without description are there
    {
        return antenna.empty() || (!antenna[a].empty() && std::all_of(antenna[a].begin(), antenna[a].end(), [](const arrayant<double> &x) { return x.n_elements() != 0; }));
    };
    for (size_t a = 0; a < antenna_args.size(); ++a)
    {
        const auto &lines = antenna_args[a];
        if (lines.size() > 1 && lines.size() != n_freq)
            return "Each entry of 'antenna_args' must have 0, 1 or n_freq lines.";
        if ((lines.empty() || (lines.size() == 1 && lines[0].empty())) && !has_arrays(a))
            return user(a) + "has neither an antenna line nor an evaluated array.";
        for (size_t i = 0; i < lines.size(); ++i)
        {
            if (lines[i].empty())
            {
                if (lines.size() > 1 && !antenna.empty() && (antenna[a].size() != n_freq || antenna[a][i].n_elements() == 0))
                    return user(a) + "has an evaluated array at the band " + qd_csv::format_number(center_freq[i]) +
                           " Hz, but 'antenna' does not hold it.";
                continue;
            }
            std::string freq_cell, type_cell;
            try
            {
                if (lines[i].find_first_of("\r\n") != std::string::npos)
                    throw std::runtime_error("the line contains a line break");
                if (antenna_cols.empty())
                    throw std::runtime_error("the line has cells, but there are no antenna columns");
                const qd_csv::table T = qd_csv::parse(header + "\n" + lines[i] + "\n", "antenna line");
                if (T.n_rows() == 1)
                {
                    qd_layout::check_antenna_cells(T, 0, ci);
                    if (ci[qd_layout::c_center_freq] >= 0)
                        freq_cell = std::string(T.cell(0, (size_t)ci[qd_layout::c_center_freq]));
                    if (ci[qd_layout::c_antenna] >= 0)
                        type_cell = std::string(T.cell(0, (size_t)ci[qd_layout::c_antenna]));
                }
                else if (!qd_csv::trim(lines[i]).empty())
                    throw std::runtime_error("the line is a comment");
            }
            catch (const std::exception &e)
            {
                return "Member 'antenna_args' has a line that is not valid ('" + lines[i] + "'): " + e.what();
            }
            double f = 0.0;
            const bool has_f = !freq_cell.empty() && qd_csv::parse_number(freq_cell, f) == qd_csv::number_status::ok;
            if (lines.size() > 1 && !(has_f && qd_layout::same_freq(f, center_freq[i])))
                return "Member 'antenna_args': the 'center_freq' of the antenna line " + std::to_string(i) + " must be the frequency " +
                       std::to_string(i) + " of 'center_freq' (one line per frequency).";
            if (lines.size() == 1 && !freq_cell.empty())
                return "Member 'antenna_args': the 'center_freq' of a single antenna line must be empty (the line applies to all frequencies).";
            if (type_cell.empty() && lines.size() > 1)
                return user(a) + "has no description for the band " + qd_csv::format_number(center_freq[i]) + " Hz; set it with set_antenna.";
            if (type_cell.empty())
                return user(a) + "has a line without antenna type.";
            if (type_cell == "none")
                return user(a) + "has a line of the type 'none', which is not stored as a line ('i_antenna' = 0 is no antenna).";
        }
    }
    if (antenna_args.empty())
        for (size_t a = 0; a < antenna.size(); ++a)
            if (!has_arrays(a))
                return user(a) + "has neither an antenna line nor an evaluated array.";
    if (seed < -1)
        return "Member 'seed' must be >= -1.";

    // Lattice vectors and pairs
    if (!wrap.is_empty())
    {
        if (wrap.n_rows != 2 || (wrap.n_cols != 1 && wrap.n_cols != 2) || !wrap.is_finite())
            return "Member 'wrap' must be finite with 2 rows and 1 or 2 columns.";
        const double a = std::hypot(wrap(0, 0), wrap(1, 0));
        if (a == 0.0)
            return "Member 'wrap' must not have a zero vector.";
        if (wrap.n_cols == 2)
        {
            const double b = std::hypot(wrap(0, 1), wrap(1, 1)), det = wrap(0, 0) * wrap(1, 1) - wrap(1, 0) * wrap(0, 1);
            if (!(std::abs(det) > 1.0e-9 * a * b))
                return "Member 'wrap' must have linearly independent columns.";
            // A reduced basis: then +-a1, +-a2 and +-(a1 - a2) are the nearest lattice points, the wrap images of the links
            const double dot = wrap(0, 0) * wrap(0, 1) + wrap(1, 0) * wrap(1, 1), m = std::min(a * a, b * b);
            if (dot < -1.0e-3 * m || dot > 0.5 * m * (1.0 + 1.0e-3))
                return "Member 'wrap' must have columns a1, a2 with 0 <= a1.a2 <= min(|a1|^2, |a2|^2) / 2, so that +-a1, +-a2 and +-(a1 - a2) are the "
                       "nearest images: replace a2 by -a2 if a1.a2 < 0, or by a2 - a1 if a1.a2 is too large.";
        }
    }
    if (!pairing.is_empty())
    {
        size_t n_site = 0, n_terminal = 0;
        qd_layout::site_of_node(tx, &n_site), qd_layout::site_of_node(rx, &n_terminal);
        if (pairing.n_rows != 2)
            return "Member 'pairing' must have 2 rows.";
        std::unordered_set<unsigned long long> pairs;
        for (arma::uword k = 0; k < pairing.n_cols; ++k)
        {
            if (pairing(0, k) >= n_site || pairing(1, k) >= n_terminal)
                return "Member 'pairing' has an index outside the sites or terminals.";
            if (!pairs.insert((unsigned long long)pairing(0, k) * n_terminal + pairing(1, k)).second)
                return "Member 'pairing' must not list a pair twice.";
        }
    }
    return "";
}

// ---- Antennas ----

namespace
{
using namespace qd_layout;
using qd_ant = quadriga_lib::arrayant<double>;
constexpr size_t cf = (size_t)c_center_freq; // Offset of the antenna columns in 'antenna_cells'

// Source of the array of one band: entry >= 0 is the array 'band' of layout.antenna[entry] before the call; -1 = none
// (not evaluated); -2 = the array given to set_antenna; -3 - k = the generated array k
struct qd_slot
{
    long entry = -1, band = 0;
};
inline bool qd_is_set(const qd_slot &s) { return s.entry != -1; }

// The antenna lists of a layout while they change: the text, and where the array of each band comes from
struct qd_ant_state
{
    std::vector<double> freq;
    std::vector<std::string> cols;
    std::vector<std::vector<std::string>> args; // [n_antenna][0, 1 or n_freq]
    std::vector<std::vector<qd_slot>> slots;    // [n_antenna][0, 1 or n_freq]
};

qd_ant_state qd_state_of(const quadriga_lib::layout &L)
{
    qd_ant_state S;
    S.freq.assign(L.center_freq.begin(), L.center_freq.end());
    S.cols = L.antenna_cols;
    const size_t n = (size_t)L.n_antenna();
    S.args = L.antenna_args;
    S.args.resize(n);
    S.slots.resize(n);
    for (size_t i = 0; i < n && i < L.antenna.size(); ++i)
    {
        S.slots[i].resize(L.antenna[i].size());
        for (size_t b = 0; b < L.antenna[i].size(); ++b)
            if (L.antenna[i][b].n_elements() != 0)
                S.slots[i][b] = {(long)i, (long)b};
    }
    return S;
}

// Kind of an antenna line: an evaluated array (empty line), a description, or a band without a description (no type)
enum
{
    k_array,
    k_text,
    k_missing
};
int qd_line_kind(const std::string &line, const std::vector<std::string> &cols)
{
    if (line.empty())
        return k_array;
    return parse_antenna_line(line, cols)[c_antenna - cf].empty() ? k_missing : k_text;
}

// True if all elements are at one position
bool qd_colocated(const qd_ant &a)
{
    const arma::mat &p = a.element_pos;
    for (arma::uword c = 1; c < p.n_cols; ++c)
        if (p(0, c) != p(0, 0) || p(1, c) != p(1, 0) || p(2, c) != p(2, 0))
            return false;
    return true;
}

// Sizes and ranges of the antenna lists that the functions of this section need; "" if they fit
std::string qd_check_lists(const quadriga_lib::layout &L)
{
    const size_t n = (size_t)L.n_antenna(), nf = L.center_freq.n_elem;
    if (!L.antenna_args.empty() && !L.antenna.empty() && L.antenna_args.size() != L.antenna.size())
        return "Members 'antenna' and 'antenna_args' must have the same length.";
    for (const quadriga_lib::layout_nodes *N : {&L.tx, &L.rx})
    {
        if (N->i_antenna.n_elem != N->name.size())
            return "Member 'i_antenna' must have one entry per node.";
        for (arma::uword i = 0; i < N->i_antenna.n_elem; ++i)
            if (N->i_antenna[i] > n)
                return "Member 'i_antenna' is larger than the number of antennas.";
    }
    for (size_t i = 0; i < nf; ++i)
        if (!std::isfinite(L.center_freq[i]) || L.center_freq[i] <= 0.0 || (i > 0 && !(L.center_freq[i] > L.center_freq[i - 1])))
            return "Member 'center_freq' must be finite, positive and strictly ascending.";
    for (size_t j = 0; j < L.antenna_cols.size(); ++j)
        if (find_column(L.antenna_cols[j]) < c_center_freq || std::count(L.antenna_cols.begin(), L.antenna_cols.end(), L.antenna_cols[j]) > 1)
            return "Member 'antenna_cols' must hold names of antenna columns, each once.";
    for (const auto &lines : L.antenna_args)
        if (lines.size() > 1 && lines.size() != nf)
            return "Each entry of 'antenna_args' must have 0, 1 or n_freq lines.";
    for (const auto &arrays : L.antenna)
        if (arrays.size() > 1 && arrays.size() != nf)
            return "Each entry of 'antenna' must have 0, 1 or n_freq arrays.";
    return "";
}

// The array of an antenna line at a frequency (NaN = the default of generate_antenna)
qd_ant qd_generate(const std::string &line, const std::vector<std::string> &cols, double freq)
{
    int ci[n_col];
    std::fill(ci, ci + n_col, -1);
    std::string header;
    for (size_t j = 0; j < cols.size(); ++j)
        ci[find_column(cols[j])] = (int)j, header += (j == 0 ? "" : ",") + cols[j];
    const qd_csv::table T = qd_csv::parse(header + "\n" + line + "\n", "antenna line");
    if (T.n_rows() != 1)
        throw std::invalid_argument("The antenna line '" + line + "' is not valid.");
    check_antenna_cells(T, 0, ci);
    auto has = [&](int id) { return ci[id] >= 0 && !T.cell(0, (size_t)ci[id]).empty(); };
    if (!has(c_antenna))
        throw std::invalid_argument("The antenna line '" + line + "' has no antenna type.");
    const std::string type(T.cell(0, (size_t)ci[c_antenna]));
    if (antenna_type_mask(type) == t_qdant)
    {
        const arma::uword id = has(c_qdant_id) ? (arma::uword)T.get_double(0, (size_t)ci[c_qdant_id]) : 1;
        const std::string fn(T.cell(0, (size_t)ci[c_qdant_file]));
        try
        {
            return quadriga_lib::qdant_read<double>(fn, {id})[0];
        }
        catch (const std::invalid_argument &e) // An entry that the file does not have
        {
            throw std::invalid_argument("QDANT file '" + fn + "': " + e.what());
        }
    }
    auto num = [&](int id) { return has(id) ? T.get_double(0, (size_t)ci[id]) : NAN; };
    auto vec = [&](int id) { return has(id) ? arma::vec(T.get_list(0, (size_t)ci[id])) : arma::vec(); };
    auto mat = [&](int id) { return has(id) ? T.get_matrix(0, (size_t)ci[id]) : arma::mat(); };
    const arma::vec panel = vec(c_panel), pol = vec(c_pol), ports = vec(c_ports), locations = vec(c_locations);
    return quadriga_lib::generate_antenna<double>(type, num(c_res), freq, vec(c_element), panel, vec(c_spacing), pol, ports,
                                                  mat(c_steer), mat(c_beams), vec(c_device_size), locations, qd_ant(),
                                                  has(c_separate_beams) && num(c_separate_beams) != 0.0,
                                                  has(c_apply_weights) && num(c_apply_weights) != 0.0);
}

// Number of pattern samples of the grid of a generated antenna; a QDANT file counts as large
double qd_grid_points(const std::string &line, const std::vector<std::string> &cols)
{
    const antenna_cells a = parse_antenna_line(line, cols);
    const std::string &type = a[c_antenna - cf];
    if (antenna_type_mask(type) == t_qdant)
        return 1.0e6;
    double res = NAN;
    if (qd_csv::parse_number(a[c_res - cf], res, true) != qd_csv::number_status::ok || !(res > 0.0))
        res = type == "omni" || type == "xpol" ? 10.0 : 1.0; // Defaults of generate_antenna
    res = std::min(std::max(res, 0.1), 90.0);
    return (std::floor(360.0 / res) + 1.0) * (std::floor(180.0 / res) + 1.0);
}

// Generates the arrays of the described bands that have none. 'strict': a band without a description, an array band
// without its array and an array spaced in wavelengths without a frequency throw; else they stay without an array
// (the last one is deferred until a frequency is set). 'only': the marked entries, nullptr = all. 'who': the node that
// the messages name for each entry
void qd_eval(qd_ant_state &S, std::vector<qd_ant> &gen, bool strict, const std::vector<char> *only, const std::vector<std::string> &who)
{
    const size_t nf = S.freq.size(), n = S.args.size();
    auto band_text = [&](size_t b) { return "the band " + qd_csv::format_number(S.freq[b]) + " Hz"; };
    auto fail = [&](size_t i, const std::string &m) { throw std::invalid_argument(who[i] + ": " + m); };
    struct job
    {
        size_t entry, line, band;
        bool all; // A single line for all bands, generated at the first band
        qd_ant out;
    };
    std::vector<job> jobs;
    for (size_t i = 0; i < n; ++i)
    {
        if (only != nullptr && !(*only)[i])
            continue;
        const std::vector<std::string> &lines = S.args[i];
        std::vector<qd_slot> &sl = S.slots[i];
        const bool all_set = !sl.empty() && std::all_of(sl.begin(), sl.end(), qd_is_set);
        if (lines.empty() || (lines.size() == 1 && lines[0].empty()))
        {
            if (strict && !all_set)
                fail(i, "the antenna has neither an antenna line nor an evaluated array.");
            continue;
        }
        if (lines.size() == 1)
        {
            if (qd_line_kind(lines[0], S.cols) == k_missing)
            {
                if (strict)
                    fail(i, "the antenna line has no antenna type.");
                continue;
            }
            if (!(all_set && (sl.size() == 1 || sl.size() == nf)))
                sl.clear(), jobs.push_back({i, 0, 0, true, qd_ant()});
            continue;
        }
        if (lines.size() != nf)
        {
            if (strict)
                fail(i, "the antenna must have 0, 1 or n_freq lines.");
            continue;
        }
        if (sl.size() != nf)
            sl.assign(nf, qd_slot());
        for (size_t b = 0; b < nf; ++b)
        {
            const int kind = qd_line_kind(lines[b], S.cols);
            if (kind == k_array && strict && !qd_is_set(sl[b]))
                fail(i, band_text(b) + " is an evaluated array, but there is no array for it.");
            else if (kind == k_missing && strict)
                fail(i, "there is no antenna for " + band_text(b) + "; set it with set_antenna.");
            else if (kind == k_text && !qd_is_set(sl[b]))
                jobs.push_back({i, b, b, false, qd_ant()});
        }
    }

    // The errors of the jobs are kept and the one of the first failing job is thrown after the loop, so that the
    // message does not depend on the threads
    auto run = [&](std::vector<job> &J)
    {
        // Serial cost of a job, measured (g++-12 -O3, Ryzen 9 5950X): about 7 us plus 20 ns per pattern sample of
        // one element; parallel runs are 1.1 to 1.8 times faster from about 20 us of serial work (2x was reached in
        // single runs only: the generation is bound by memory allocation). Elements are not counted here.
        double work_us = 0.0;
        for (const job &x : J)
            work_us += 7.0 + 0.02 * qd_grid_points(S.args[x.entry][x.line], S.cols);
        std::vector<std::string> err(J.size());
        std::vector<char> bad(J.size(), 0); // 1 = std::invalid_argument, 2 = other
        qd_batch_loop(J.size(), J.size() >= 2 && work_us >= 20.0, "Antenna", [&](size_t j)
                      {
            try
            {
                J[j].out = qd_generate(S.args[J[j].entry][J[j].line], S.cols, nf == 0 ? NAN : S.freq[J[j].band]);
            }
            catch (const std::invalid_argument &e)
            {
                bad[j] = 1, err[j] = e.what();
            }
            catch (const std::exception &e)
            {
                bad[j] = 2, err[j] = e.what();
            } });
        for (size_t j = 0; j < J.size(); ++j)
            if (bad[j] != 0)
            {
                const std::string m = who[J[j].entry] + ": " + (J[j].all ? "" : band_text(J[j].band) + ": ") + err[j];
                if (bad[j] == 1)
                    throw std::invalid_argument(m);
                throw std::runtime_error(m);
            }
    };
    auto store = [&](job &x)
    {
        S.slots[x.entry][x.band] = {-3 - (long)gen.size(), 0};
        gen.push_back(std::move(x.out));
    };

    // A single line is generated at the first band; the other bands follow if the elements are not co-located
    run(jobs);
    std::vector<job> more;
    for (job &x : jobs)
    {
        if (!x.all)
        {
            store(x);
            continue;
        }
        const unsigned mask = antenna_type_mask(parse_antenna_line(S.args[x.entry][0], S.cols)[c_antenna - cf]);
        const bool colocated = qd_colocated(x.out);
        if (nf == 0 && !colocated && (mask & (t_ula | t_3gpp | t_multibeam)) != 0)
        {
            if (strict)
                fail(x.entry, "the element spacing is given in wavelengths, but the layout has no frequency; set 'center_freq'.");
            continue;
        }
        S.slots[x.entry].assign(nf <= 1 || colocated || mask == t_qdant ? 1 : nf, qd_slot());
        store(x);
        for (size_t b = 1; b < S.slots[x.entry].size(); ++b)
            more.push_back({x.entry, 0, b, false, qd_ant()});
    }
    run(more);
    for (job &x : more)
        store(x);
}

// Writes the state into the layout: the text, the frequencies and the arrays (moved where they are used once)
void qd_commit(quadriga_lib::layout &L, qd_ant_state &S, std::vector<qd_ant> &gen, const qd_ant *given)
{
    std::vector<std::vector<unsigned>> uses(L.antenna.size());
    for (size_t i = 0; i < L.antenna.size(); ++i)
        uses[i].assign(L.antenna[i].size(), 0);
    for (const auto &sl : S.slots)
        for (const qd_slot &s : sl)
            if (s.entry >= 0)
                ++uses[(size_t)s.entry][(size_t)s.band];
    std::vector<std::vector<qd_ant>> A(S.args.size());
    bool any = false;
    for (size_t i = 0; i < S.args.size(); ++i)
    {
        if (std::none_of(S.slots[i].begin(), S.slots[i].end(), qd_is_set))
            continue; // Not evaluated
        A[i].resize(S.slots[i].size());
        for (size_t b = 0; b < S.slots[i].size(); ++b)
        {
            const qd_slot &s = S.slots[i][b];
            if (s.entry >= 0)
            {
                qd_ant &src = L.antenna[(size_t)s.entry][(size_t)s.band];
                if (--uses[(size_t)s.entry][(size_t)s.band] == 0)
                    A[i][b] = std::move(src);
                else
                    A[i][b] = src;
            }
            else if (s.entry == -2)
                A[i][b] = *given;
            else if (s.entry <= -3)
                A[i][b] = std::move(gen[(size_t)(-3 - s.entry)]);
        }
        any = true;
    }
    L.antenna = any ? std::move(A) : std::vector<std::vector<qd_ant>>();
    L.center_freq = arma::vec(S.freq);
    L.antenna_cols = S.cols;
    L.antenna_args = S.args;
}

// The antenna columns 'cols' and those that the cells 'need' use, in the order of the table
std::vector<std::string> qd_cols_with(const std::vector<std::string> &cols, const antenna_cells &need)
{
    bool in[n_antenna_col] = {};
    for (const std::string &c : cols)
        in[(size_t)find_column(c) - cf] = true;
    std::vector<std::string> out;
    for (size_t k = 0; k < (size_t)n_antenna_col; ++k)
        if (in[k] || !need[k].empty())
            out.push_back(column_of((int)(cf + k)).name);
    return out;
}

// Description of a node in messages, e.g. "Tx node 'BS1', sector 0"
std::string qd_node_text(const quadriga_lib::layout_nodes &N, size_t n, bool rx)
{
    return std::string(rx ? "Rx" : "Tx") + " node '" + N.name[n] + "', sector " + std::to_string(N.sector[n]);
}

// The node that the messages about each antenna entry name: the first that uses it, Tx nodes first; 'where' replaces
// the description of the nodes of a side (nullptr = qd_node_text)
std::vector<std::string> qd_who(const quadriga_lib::layout &L, const std::vector<arma::u32> ia[2], size_t n_antenna,
                                const std::vector<std::string> *const where[2])
{
    std::vector<std::string> who(n_antenna);
    for (int s = 0; s < 2; ++s)
        for (size_t n = 0; n < ia[s].size(); ++n)
            if (ia[s][n] != 0 && who[ia[s][n] - 1].empty())
                who[ia[s][n] - 1] = where[s] != nullptr && n < where[s]->size() ? (*where[s])[n] : qd_node_text(s == 0 ? L.tx : L.rx, n, s == 1);
    for (std::string &w : who)
        if (w.empty())
            w = "An antenna that no node uses";
    return who;
}

// Sets the antenna of the nodes 'i_node' of one side to the description 'cells', to the array 'given' or, with both
// nullptr, to none, for all bands (freq NaN) or for one band. The antenna of each node is decoded into its lines and
// arrays, the picked nodes get their new antenna, and the table is encoded again as the reader does: equal descriptions
// (lines without arrays) share one entry, an antenna with arrays stays shared by the nodes that shared it, entries are
// numbered in order of first use (Tx nodes first), and unused entries disappear. eval_mode: 0 = no evaluation, 1 = the
// entries that had arrays before or hold the given array, 2 = all entries not evaluated yet
void qd_set_antenna(quadriga_lib::layout &L, const antenna_cells *cells, const qd_ant *given, bool rx_side,
                    const quadriga_lib::uint_list &i_node, double freq, int eval_mode)
{
    std::string msg = qd_check_lists(L);
    if (!msg.empty())
        throw std::invalid_argument("Input 'layout' is not valid: " + msg);
    if (!std::isnan(freq) && !(std::isfinite(freq) && freq > 0.0))
        throw std::invalid_argument("Input 'center_freq' must be positive and finite, or NaN.");
    const size_t n_node = (size_t)(rx_side ? L.rx : L.tx).n_node();
    std::vector<char> picked(n_node, i_node.vec().is_empty() ? 1 : 0);
    for (arma::uword v : i_node.vec())
    {
        if (v >= n_node)
            throw std::invalid_argument("Input 'i_node' has an index outside the nodes of the " + std::string(rx_side ? "Rx" : "Tx") + " side.");
        picked[(size_t)v] = 1;
    }

    qd_ant_state S = qd_state_of(L);
    std::vector<char> had_arrays(S.args.size());
    for (size_t i = 0; i < S.args.size(); ++i)
        had_arrays[i] = std::any_of(S.slots[i].begin(), S.slots[i].end(), qd_is_set);

    // Columns of the new line, and center_freq for one line per band
    {
        antenna_cells need = cells != nullptr ? *cells : antenna_cells();
        if (!std::isnan(freq))
            need[0] = "f";
        const std::vector<std::string> cols = qd_cols_with(S.cols, need);
        rewrite_antenna_lines(S.args, S.cols, cols);
        S.cols = cols;
    }

    // The band: a frequency of the list, or a new one, which the antennas with one line per band do not describe yet
    long band = -1;
    if (!std::isnan(freq))
    {
        for (size_t b = 0; b < S.freq.size(); ++b)
            if (same_freq(S.freq[b], freq))
                band = (long)b;
        if (band < 0)
        {
            const size_t n_old = S.freq.size(), p = (size_t)(std::upper_bound(S.freq.begin(), S.freq.end(), freq) - S.freq.begin());
            S.freq.insert(S.freq.begin() + (long)p, freq);
            antenna_cells missing;
            missing[0] = qd_csv::format_number(freq);
            for (size_t i = 0; i < S.args.size(); ++i)
            {
                std::vector<std::string> &lines = S.args[i];
                std::vector<qd_slot> &sl = S.slots[i];
                if (lines.size() > 1)
                {
                    lines.insert(lines.begin() + (long)p, join_antenna_line(missing, S.cols));
                    if (sl.size() == n_old)
                        sl.insert(sl.begin() + (long)p, qd_slot());
                }
                else if (lines.size() == 1 && !lines[0].empty() && !sl.empty())
                {
                    // One line for all bands: kept if its array holds for every band (co-located, or read from a file)
                    const bool qdant = antenna_type_mask(parse_antenna_line(lines[0], S.cols)[c_antenna - cf]) == t_qdant;
                    if (!(sl.size() == 1 && sl[0].entry >= 0 && (qdant || qd_colocated(L.antenna[(size_t)sl[0].entry][(size_t)sl[0].band]))))
                        sl.clear();
                }
            }
            band = (long)p;
        }
    }
    const size_t nf = S.freq.size();

    // The new antenna of the picked nodes whose antenna was 'a' (1-based, 0 = none). One band: the antenna of the node
    // turned into one line (or array) per band, then the band replaced
    struct value
    {
        std::vector<std::string> lines;
        std::vector<qd_slot> slots;
        bool want = false; // Evaluated in eval_mode 1
    };
    auto compose = [&](size_t a)
    {
        value v;
        v.want = given != nullptr;
        if (band < 0)
        {
            if (cells != nullptr)
                v.lines.push_back(join_antenna_line(*cells, S.cols));
            else if (given != nullptr)
                v.slots.push_back({-2, 0});
            return v;
        }
        const std::vector<std::string> old = a == 0 ? std::vector<std::string>() : S.args[a - 1];
        const std::vector<qd_slot> old_sl = a == 0 ? std::vector<qd_slot>() : S.slots[a - 1];
        v.want = v.want || (a != 0 && had_arrays[a - 1]);
        v.lines.assign(nf, std::string()), v.slots.assign(nf, qd_slot());
        for (size_t b = 0; b < nf; ++b)
        {
            if (old.size() == nf && nf > 1) // One line per band
            {
                v.lines[b] = old[b];
                v.slots[b] = old_sl.size() == nf ? old_sl[b] : qd_slot();
                continue;
            }
            if (a != 0)
                v.slots[b] = old_sl.size() == 1 ? old_sl[0] : (old_sl.size() == nf ? old_sl[b] : qd_slot());
            if (a != 0 && (old.empty() || old[0].empty())) // One array for all bands
                continue;
            antenna_cells m; // One line for all bands, or no antenna: no description for the band
            if (a != 0)
                m = parse_antenna_line(old[0], S.cols);
            m[0] = qd_csv::format_number(S.freq[b]);
            v.lines[b] = join_antenna_line(m, S.cols);
        }
        if (cells != nullptr)
        {
            antenna_cells m = *cells;
            m[0] = qd_csv::format_number(S.freq[(size_t)band]);
            v.lines[(size_t)band] = join_antenna_line(m, S.cols), v.slots[(size_t)band] = qd_slot();
        }
        else
            v.lines[(size_t)band].clear(), v.slots[(size_t)band] = {-2, 0};
        if (nf == 1) // The only band: one line without frequency, or the array
        {
            if (v.lines[0].empty())
                v.lines.clear();
            else
            {
                antenna_cells m = parse_antenna_line(v.lines[0], S.cols);
                m[0].clear();
                v.lines[0] = join_antenna_line(m, S.cols);
            }
        }
        return v;
    };

    // Encoding in the order of the nodes, Tx first; 'origin' identifies an antenna with arrays
    qd_ant_state T;
    T.freq = S.freq, T.cols = S.cols;
    std::vector<char> want;
    std::map<std::string, arma::u32> index, done;
    auto n_set = [](const std::vector<qd_slot> &sl) { return std::count_if(sl.begin(), sl.end(), qd_is_set); };
    auto encode = [&](value v, const std::string &origin) -> arma::u32
    {
        if (v.lines.empty() && v.slots.empty())
            return 0;
        const bool text = !v.lines.empty() && std::none_of(v.lines.begin(), v.lines.end(), [](const std::string &l) { return l.empty(); });
        std::string key = text ? "t" : origin;
        for (size_t b = 0; text && b < v.lines.size(); ++b)
            key += "\n" + v.lines[b];
        const auto it = index.find(key);
        if (it == index.end())
        {
            T.args.push_back(std::move(v.lines)), T.slots.push_back(std::move(v.slots)), want.push_back(v.want);
            return index[key] = (arma::u32)T.args.size();
        }
        const size_t t = it->second - 1; // The same description: the arrays of the more evaluated one
        if (n_set(v.slots) > n_set(T.slots[t]))
            T.slots[t] = std::move(v.slots);
        want[t] = want[t] || v.want;
        return it->second;
    };
    std::vector<arma::u32> ia[2];
    for (int s = 0; s < 2; ++s)
    {
        const arma::u32_vec &old = (s == 0 ? L.tx : L.rx).i_antenna;
        for (arma::uword n = 0; n < old.n_elem; ++n)
        {
            const arma::u32 a = old[n];
            const bool pick = (s == 1) == rx_side && picked[n];
            const std::string origin = pick ? (band < 0 ? "n" : "c" + std::to_string(a)) : "o" + std::to_string(a);
            const auto it = done.find(origin);
            if (it != done.end())
                ia[s].push_back(it->second);
            else if (!pick && a == 0)
                ia[s].push_back(done[origin] = 0);
            else
                ia[s].push_back(done[origin] = encode(pick ? compose(a) : value{S.args[a - 1], S.slots[a - 1], had_arrays[a - 1] != 0}, origin));
        }
    }
    const std::vector<std::string> cols = used_antenna_cols(T.args, T.cols);
    rewrite_antenna_lines(T.args, T.cols, cols);
    T.cols = cols;

    std::vector<qd_ant> gen;
    const std::vector<std::string> *const where[2] = {nullptr, nullptr};
    if (eval_mode != 0)
        qd_eval(T, gen, false, eval_mode == 1 ? &want : nullptr, qd_who(L, ia, T.args.size(), where));
    qd_commit(L, T, gen, given);
    L.tx.i_antenna = arma::u32_vec(ia[0]);
    L.rx.i_antenna = arma::u32_vec(ia[1]);
}
} // namespace

void qd_layout::eval_antennas(quadriga_lib::layout &L, const arma::vec &center_freq_new, const std::vector<std::string> *where_tx,
                              const std::vector<std::string> *where_rx)
{
    const std::string msg = qd_check_lists(L);
    if (!msg.empty())
        throw std::invalid_argument("The layout is not valid: " + msg);
    qd_ant_state S = qd_state_of(L);
    const std::vector<arma::u32> ia[2] = {std::vector<arma::u32>(L.tx.i_antenna.begin(), L.tx.i_antenna.end()),
                                          std::vector<arma::u32>(L.rx.i_antenna.begin(), L.rx.i_antenna.end())};
    const std::vector<std::string> *const where[2] = {where_tx, where_rx};
    const std::vector<std::string> who = qd_who(L, ia, S.args.size(), where);
    if (!center_freq_new.is_empty())
    {
        for (arma::uword i = 0; i < center_freq_new.n_elem; ++i)
            if (!std::isfinite(center_freq_new[i]) || center_freq_new[i] <= 0.0 || (i > 0 && !(center_freq_new[i] > center_freq_new[i - 1])))
                throw std::invalid_argument("Input 'center_freq' must be finite, positive and strictly ascending.");
        const size_t nf = center_freq_new.n_elem;
        for (size_t i = 0; i < S.args.size(); ++i) // All checks first: a failing call changes nothing
        {
            const std::vector<std::string> &lines = S.args[i];
            const std::string nfs = "Input 'center_freq' has " + std::to_string(nf) + " frequencies; " + who[i] + ": the antenna has ";
            if ((lines.empty() || (lines.size() == 1 && lines[0].empty())) && S.slots[i].size() > 1 && S.slots[i].size() != nf)
                throw std::invalid_argument(nfs + "evaluated arrays for " + std::to_string(S.slots[i].size()) + " bands.");
            if (lines.size() < 2)
                continue;
            if (lines.size() != nf)
                throw std::invalid_argument(nfs + "one line for each of " + std::to_string(lines.size()) + " bands.");
            for (size_t b = 0; b < nf; ++b)
            {
                double f = 0.0;
                const std::string cell = lines[b].empty() ? std::string() : parse_antenna_line(lines[b], S.cols)[0];
                if (!lines[b].empty() && (qd_csv::parse_number(cell, f) != qd_csv::number_status::ok || !same_freq(f, center_freq_new[b])))
                    throw std::invalid_argument("Input 'center_freq': entry " + std::to_string(b) + " does not match; " + who[i] +
                                                ": the antenna has the band '" + cell + "'.");
            }
        }
        for (size_t i = 0; i < S.args.size(); ++i) // The lines are generated again; evaluated arrays stay
        {
            const std::vector<std::string> &lines = S.args[i];
            std::vector<qd_slot> &sl = S.slots[i];
            if (lines.size() == 1 && !lines[0].empty())
                sl.clear();
            for (size_t b = 0; lines.size() > 1 && b < lines.size() && b < sl.size(); ++b)
                if (!lines[b].empty())
                    sl[b] = qd_slot();
        }
        S.freq.assign(center_freq_new.begin(), center_freq_new.end());
    }
    std::vector<qd_ant> gen;
    qd_eval(S, gen, true, nullptr, who);
    qd_commit(L, S, gen, nullptr);
}

void quadriga_lib::layout::eval_antennas(const arma::vec &center_freq_new)
{
    qd_layout::eval_antennas(*this, center_freq_new, nullptr, nullptr);
}

/*!MD
# set_antenna
Set the antenna of layout nodes from the arguments of generate_antenna or from an evaluated array

- The first form stores the arguments of [[generate_antenna]] (see there for their meaning) as an antenna line of the layout, which [[layout_csv_write]] writes as CSV cells; the second form stores an array, e.g. a measurement read with [[qdant_read]], which the writer exports to a QDANT file
- Without `center_freq`, the antenna applies to all bands. With `center_freq`, only that band of the nodes' antenna is set: a frequency that is not in `layout.center_freq` is added, the other bands of the nodes keep their antenna, and nodes that shared the old entry keep it; until all bands of a node are set, `layout.is_valid()` reports it
- A node set band by band keeps one line per band, also when the lines are equal, so a band added later is reported as not set for it; in a layout with one band, the line of a node set for that band also covers bands added later
- The antenna lists are encoded again after the call as [[layout_csv_read]] composes them: equal lines share one entry, entries are numbered in order of first use (Tx nodes first), and entries that no node uses are removed, so `i_antenna` of other nodes can change
- An array spaced in wavelengths (`ula`, `3GPP`, `multibeam` with several element positions) in a layout without frequencies is generated only when a frequency is set; its nodes use the generic antenna meanwhile
- A failed call leaves the layout unchanged

## Declarations:
```
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

## Inputs:
- **`layout`** — Layout, updated in place; its antenna lists must fit its nodes and frequencies
- **`type`** — Antenna type of [[generate_antenna]], e.g. `3GPP`, `dipole`, or `none` (no antenna, for all bands, without `args`)
- **`args`** *(optional)* — Arguments of [[generate_antenna]] by name: `res`, `center_freq`, `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `beams`, `device_size`, `locations` as numbers, vectors or matrices (`steer`, `beams` row by row as there), `separate_beams` and `apply_weights` as 0 or 1; a name that the type does not use throws; `pattern` is not available (an array built with it is set with the second form); `center_freq` here is the same as the argument
- **`array`** — Valid single-frequency array
- **`rx_side`** *(optional)* — `false`: nodes of `layout.tx`; `true`: nodes of `layout.rx`
- **`i_node`** *(optional)* — Indices of the nodes of that side, 0-based; empty: all nodes of the side
- **`center_freq`** *(optional)* — Band in [Hz], positive; NaN: one antenna for all bands
- **`evaluate`** *(optional)* — `true`: generate all antennas of the layout that are not evaluated yet; `false`: none, the nodes of new antennas use the generic antenna until `layout.eval_antennas()` is called

## Example:
```
quadriga_lib::layout L = quadriga_lib::layout_csv_read("tx.csv", "rx.csv");
quadriga_lib::set_antenna(L, "3GPP", {{"panel", {4, 4, 1, 2}}, {"pol", {3}}}, false, {}, 3.5e9);
quadriga_lib::set_antenna(L, quadriga_lib::qdant_read<double>("handset.qdant")[0], true);
```

## See also:
- [[generate_antenna]] (the meaning of the arguments)
- [[layout]] (the antenna lists and the member function `eval_antennas`)
- [[layout_csv_write]] (writes the lines and exports the arrays)
MD!*/

void quadriga_lib::set_antenna(quadriga_lib::layout &layout, const std::string &type, const std::unordered_map<std::string, arma::mat> &args,
                               bool rx_side, const uint_list &i_node, double center_freq, bool evaluate)
{
    const unsigned mask = antenna_type_mask(type);
    if (mask == 0 || mask == t_qdant)
        throw std::invalid_argument("Input 'type' must be an antenna type of generate_antenna or 'none' ('" + type + "').");
    antenna_cells cells;
    cells[c_antenna - cf] = type;
    double freq = center_freq;
    std::vector<std::string> keys;
    for (const auto &kv : args)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (const std::string &key : keys)
    {
        const arma::mat &v = args.at(key);
        auto fail = [&](const std::string &what) { throw std::invalid_argument("Input 'args': '" + key + "' must be " + what + "."); };
        if (key == "center_freq")
        {
            if (v.n_elem != 1)
                fail("a scalar");
            if (!std::isnan(v[0]) && !std::isnan(center_freq))
                throw std::invalid_argument("Inputs 'center_freq' and 'args' must not both give 'center_freq'.");
            if (!std::isnan(v[0]))
                freq = v[0];
            continue;
        }
        if (key == "pattern")
            throw std::invalid_argument("Input 'args': 'pattern' is not available; an array built from a pattern is set with the array form of set_antenna.");
        const int id = find_column(key);
        if (id < c_res)
            throw std::invalid_argument("Input 'args' has the unknown argument '" + key + "'.");
        if ((column_of(id).antenna_types & mask) == 0)
            throw std::invalid_argument("Input 'args': the argument '" + key + "' is not used by the antenna type '" + type + "'.");
        if (v.is_empty())
            continue;
        std::string &cell = cells[(size_t)id - cf];
        switch (column_of(id).kind)
        {
        case col_kind::number:
            if (v.n_elem != 1)
                fail("a scalar");
            cell = qd_csv::format_number(v[0]);
            break;
        case col_kind::flag:
            if (v.n_elem != 1 || (v[0] != 0.0 && v[0] != 1.0))
                fail("0 or 1");
            cell = v[0] == 1.0 ? "1" : "0";
            break;
        case col_kind::matrix:
            cell = qd_csv::format_matrix(v);
            break;
        default: // Lists
            if (!v.is_vec())
                fail("a vector");
            if (column_of(id).kind == col_kind::int_list)
                for (double x : v)
                    if (!(x >= 0.0 && x <= 4294967295.0 && std::floor(x) == x))
                        fail("a vector of non-negative integers");
            cell = qd_csv::format_list(std::vector<double>(v.begin(), v.end()));
        }
    }
    if (mask == t_none && !std::isnan(freq))
        throw std::invalid_argument("Input 'center_freq': the antenna type 'none' applies to all bands.");
    qd_set_antenna(layout, mask == t_none ? nullptr : &cells, nullptr, rx_side, i_node, freq, evaluate ? 2 : 0);
}

void quadriga_lib::set_antenna(quadriga_lib::layout &layout, const arrayant<double> &array, bool rx_side, const uint_list &i_node, double center_freq)
{
    const std::string msg = array.is_valid();
    if (!msg.empty())
        throw std::invalid_argument("Input 'array' is not valid: " + msg);
    qd_set_antenna(layout, nullptr, &array, rx_side, i_node, center_freq, 1);
}
