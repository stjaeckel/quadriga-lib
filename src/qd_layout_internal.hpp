// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Internal header of the layout sources (private; not named qd_helpers_* so that the wrappers do not depend on it)
//
// The column table: every column of the layout CSV files is one entry of the table in qd_layout.cpp, with its name,
// class (what an empty cell means), value kind, side, unit conversion and, for antenna columns, the generator types
// that accept it. The CSV reader and writer are driven by the table; 'flat' is the form of one side that follows the
// table (one array per column), with 'flatten' and 'unflatten' as the only code that maps columns to class members.
// A new column is a new entry of the table (and of 'col_id'), and a line in 'flatten' and 'unflatten'.

#ifndef qd_layout_internal_hpp
#define qd_layout_internal_hpp

#include "quadriga_layout.hpp"
#include "qd_csv.hpp"

#include <array>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace qd_layout
{

// Antenna generator types as bits, for the 'antenna_types' of an antenna column
enum : unsigned
{
    t_simple = 1,    // omni, xpol, dipole, short-dipole, half-wave-dipole
    t_custom = 2,
    t_ula = 4,
    t_3gpp = 8,
    t_multibeam = 16,
    t_device = 32,   // 3GPP-UT, 3GPP-CPE
    t_qdant = 64,    // the antenna is read from a QDANT file
    t_all = 127,     // the types above
    t_none = 128     // 'none': no antenna (i_antenna = 0); takes no other antenna column
};

// The arguments of generate_antenna that have a column: name, kind of value, generator types that use it. A new
// argument is one line here; it is a column after 'qdant_id', for the antenna lines and for the reader and writer
#define QD_LAYOUT_ANTENNA_ARGS(X)                                                  \
    X(res, number, t_all & ~t_qdant)                                              \
    X(element, list, t_custom | t_3gpp | t_multibeam | t_device)                  \
    X(panel, int_list, t_ula | t_3gpp | t_multibeam)                              \
    X(spacing, list, t_ula | t_3gpp | t_multibeam)                                \
    X(pol, int_list, t_ula | t_3gpp | t_multibeam | t_device)                     \
    X(ports, int_list, t_ula | t_3gpp)                                            \
    X(steer, matrix, t_ula | t_3gpp)                                              \
    X(beams, matrix, t_multibeam)                                                 \
    X(separate_beams, flag, t_multibeam)                                          \
    X(apply_weights, flag, t_ula | t_3gpp | t_multibeam)                          \
    X(device_size, list, t_device)                                                \
    X(locations, int_list, t_device)

// Columns in the order of the table; the antenna columns are the last ones, from c_center_freq
enum col_id : int
{
    c_name, c_sector, c_x, c_y, c_z, c_time, c_bank, c_tilt, c_heading, c_direction_az, c_direction_el, c_speed,
    c_scenario, c_indoor, c_floor, c_d2d_in, c_d_clutter, c_r_clutter, c_h_clutter, c_scenpar, c_wrap, c_seed,
    c_layout_seed, c_pairing, c_relative,
    c_center_freq, c_antenna, c_qdant_file, c_qdant_id,
#define QD_LAYOUT_X(name, kind, types) c_##name,
    QD_LAYOUT_ANTENNA_ARGS(QD_LAYOUT_X)
#undef QD_LAYOUT_X
    n_col
};
constexpr int n_antenna_col = n_col - c_center_freq;

// What an empty cell means: identity (name, sector), fixed (value of the first row, or of sector 0 for an extra
// sector), per row (position and time), held (repeats the previous row)
enum class col_class : unsigned char { identity, fixed, per_row, held };

// Kind of value: text; integer; number; list of numbers; list of non-negative integers; matrix (rows separated by ';');
// list of names; flag (0 or 1)
enum class col_kind : unsigned char { text, integer, number, list, int_list, matrix, names, flag };

enum class col_side : unsigned char { both, tx, rx };

struct column
{
    const char *name;      // Name in the header of the file; "scenpar." is the prefix of the scenario parameter columns
    col_class cls;         // Class of the column
    col_kind kind;         // Kind of value
    col_side side;         // File that may hold it
    double scale;          // Value in the file times 'scale' is the value in the class (degree to radian)
    double dflt;           // Value of an omitted column and of an empty first cell (the number 0 for text)
    double lo, hi;         // Range of a number, checked when the cell is read
    unsigned antenna_types; // Antenna columns: generator types that accept the column, else 0
};

// Entry of the table
const column &column_of(int id);

// Id of the column with this header name, or -1; "scenpar.<key>" with a non-empty key is c_scenpar
int find_column(std::string_view name);

// Bit set of the antenna type of the 'antenna' column (as accepted by generate_antenna, or "qdant"); 0 = unknown
unsigned antenna_type_mask(const std::string &type);

// The cells of an antenna line (CSV text of the antenna columns of a layout) as text
std::vector<std::string> split_antenna_line(const std::string &line, size_t n_cols);

// Cells of an antenna line by column, index = column id - c_center_freq
using antenna_cells = std::array<std::string, n_antenna_col>;

// The cells of a line of the antenna columns 'cols' (names), and the line of cells for 'cols'
antenna_cells parse_antenna_line(const std::string &line, const std::vector<std::string> &cols);
std::string join_antenna_line(const antenna_cells &cells, const std::vector<std::string> &cols);

// Names of the antenna columns that some cell of the lines uses, in the order of the table; 'extra' are cells of lines
// still to come
std::vector<std::string> used_antenna_cols(const std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                                           const std::vector<antenna_cells> &extra = {});

// The lines of 'args' rewritten from the columns 'cols' to 'new_cols'
void rewrite_antenna_lines(std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                           const std::vector<std::string> &new_cols);

// Number of a cell of column 'id' in class units; throws std::runtime_error with the file, line and column if the cell
// is not a number of the column or outside its range. Integer columns need integer values
double cell_number(const qd_csv::table &T, size_t row, size_t col, int id);

// Integer of a cell as a 64-bit number (seeds), at least the minimum of the column 'id'
long long cell_int64(const qd_csv::table &T, size_t row, size_t col, int id);

// Throws std::runtime_error unless the antenna cells of a row of T are a valid antenna description: the type is known,
// the columns are used by the type, the cells have the syntax of their kind. 'ci' gives the column of T for each
// column id, -1 = not in T
void check_antenna_cells(const qd_csv::table &T, size_t row, const int *ci);

// layout::eval_antennas, with the description of each node in the messages ('where_tx', 'where_rx': one text per
// node, e.g. with file and line; nullptr = "Tx node '<name>', sector <s>")
void eval_antennas(quadriga_lib::layout &L, const arma::vec &center_freq, const std::vector<std::string> *where_tx,
                   const std::vector<std::string> *where_rx);

// True if two frequencies are equal within a relative 1e-6
inline bool same_freq(double a, double b) { return std::abs(a - b) <= 1.0e-6 * std::max(std::abs(a), std::abs(b)); }

// One side of a layout in the form of the table; an empty array is an omitted column
struct flat
{
    size_t n_node = 0, n_row = 0;
    std::vector<std::string> name;                  // [n_node]
    std::vector<size_t> offset;                     // [n_node]
    std::vector<double> v[n_col];                   // Numbers in class units: per row (per-row and held columns), per node (others)
    std::vector<long long> seed;                    // Seeds of the sites, per node (exact, not a double)
    std::vector<std::string> txt[n_col];            // Text columns: scenario per row, pairing per node
    std::map<std::string, std::vector<double>> scenpar; // Scenario parameter columns by key, per node
    std::vector<std::string> extra_names;           // Unknown columns, [n_extra]
    std::vector<std::vector<std::string>> extra;    // Their cells, [n_extra][n_row]
};

// Member mapping of the table; 'unflatten' leaves a member empty when its column holds only default values (the
// meaning of the empty member) and does not set 'i_antenna'
flat flatten(const quadriga_lib::layout_nodes &nodes);
quadriga_lib::layout_nodes unflatten(const flat &F);

// Index of the site (or terminal) of each node: nodes with equal names, numbered in order of first appearance
std::vector<size_t> site_of_node(const quadriga_lib::layout_nodes &nodes, size_t *n_site = nullptr);

// True if two numbers are equal, with NaN equal to NaN
inline bool same_value(double a, double b) { return a == b || (a != a && b != b); }

} // namespace qd_layout

#endif
