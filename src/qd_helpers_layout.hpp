// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Helpers shared by the layout sources (qd_layout.cpp, layout_csv.cpp, generate_layout.cpp, generate_track.cpp and
// layout_pipeline.cpp)
//
// The column table: every column of the layout CSV files is one entry of the table below, with its name, class (what an
// empty cell means), value kind, side, unit conversion and, for antenna columns, the generator types that accept it.
// The CSV reader and writer are driven by the table; 'flat' in layout_csv.cpp is the form of one side that follows the
// table (one array per column), with 'flatten' and 'unflatten' as the only code that maps columns to class members.
// A new column is a new entry of the table (and of 'col_id'), and a line in 'flatten' and 'unflatten'.

#ifndef qd_helpers_layout_hpp
#define qd_helpers_layout_hpp

#include "quadriga_layout.hpp"
#include "quadriga_sos.hpp"
#include "qd_csv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Seed of one random stream of a layout: the 64-bit FNV-1a hash of the 8 bytes of the base seed (least significant byte
// first) followed by the bytes of the key, with the sign bit cleared. Equal base seeds and keys give equal seeds on all
// platforms; the keys name the stream, e.g. "track|<node name>"
static inline arma::sword qd_layout_seed_hash(arma::sword base_seed, const std::string &key)
{
    unsigned long long h = 14695981039346656037ULL; // FNV offset basis
    const unsigned long long s = (unsigned long long)base_seed;
    for (int i = 0; i < 8; ++i)
        h = (h ^ ((s >> (8 * i)) & 0xFFULL)) * 1099511628211ULL; // FNV prime
    for (unsigned char c : key)
        h = (h ^ c) * 1099511628211ULL;
    return (arma::sword)(h & 0x7FFFFFFFFFFFFFFFULL);
}

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
    c_layout_seed, c_cell, c_pairing, c_relative,
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

static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
static constexpr double inf = std::numeric_limits<double>::infinity();
static constexpr double deg = 0.017453292519943295769236907684886; // pi / 180

// Order of the entries = order of 'col_id' = order of the columns in a written file
// Each translation unit has its own copy of the table, so pointers into it must not cross translation units
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
    {"cell", col_class::fixed, col_kind::integer, col_side::tx, 1.0, 1.0, 0.0, 1.0, 0},
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

// Entry of the table
static inline const column &column_of(int id) { return table[id]; }

// Id of the column with this header name, or -1; "scenpar.<key>" with a non-empty key is c_scenpar
static inline int find_column(std::string_view name)
{
    if (name.size() > 8 && name.substr(0, 8) == "scenpar.")
        return c_scenpar;
    for (int i = 0; i < n_col; ++i)
        if (i != c_scenpar && name == table[i].name)
            return i;
    return -1;
}

// Bit set of the antenna type of the 'antenna' column (as accepted by generate_antenna, or "qdant"); 0 = unknown
static inline unsigned antenna_type_mask(const std::string &type)
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

// The cells of an antenna line (CSV text of the antenna columns of a layout) as text
static inline std::vector<std::string> split_antenna_line(const std::string &line, size_t n_cols)
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

// Cells of an antenna line by column, index = column id - c_center_freq
using antenna_cells = std::array<std::string, n_antenna_col>;

// The cells of a line of the antenna columns 'cols' (names), and the line of cells for 'cols'
static inline antenna_cells parse_antenna_line(const std::string &line, const std::vector<std::string> &cols)
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

static inline std::string join_antenna_line(const antenna_cells &cells, const std::vector<std::string> &cols)
{
    std::string line;
    for (size_t k = 0; k < cols.size(); ++k)
    {
        const int id = find_column(cols[k]);
        line += (k == 0 ? "" : ",") + qd_csv::format_cell(id >= c_center_freq ? cells[(size_t)(id - c_center_freq)] : std::string());
    }
    return line;
}

// Names of the antenna columns that some cell of the lines uses, in the order of the table; 'extra' are cells of lines
// still to come
static inline std::vector<std::string> used_antenna_cols(const std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                                                         const std::vector<antenna_cells> &extra = {})
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

// The lines of 'args' rewritten from the columns 'cols' to 'new_cols'
static inline void rewrite_antenna_lines(std::vector<std::vector<std::string>> &args, const std::vector<std::string> &cols,
                                         const std::vector<std::string> &new_cols)
{
    if (cols == new_cols)
        return;
    for (auto &lines : args)
        for (std::string &line : lines)
            if (!line.empty())
                line = join_antenna_line(parse_antenna_line(line, cols), new_cols);
}

// Number of a cell of column 'id' in class units; throws std::runtime_error with the file, line and column if the cell
// is not a number of the column or outside its range. Integer columns need integer values
static inline double cell_number(const qd_csv::table &T, size_t row, size_t col, int id)
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

// Throws unless the text of an antenna cell has the syntax of its kind
static inline void qd_check_antenna_cell(const qd_csv::table &T, size_t r, size_t c, int id)
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
        if (const arma::uword n = T.get_matrix(r, c).n_cols; id == c_beams && n != 2 && n != 3)
            throw std::runtime_error(T.where(r, c) + " must have 2 or 3 entries per beam ('" + std::string(T.cell(r, c)) + "').");
        break;
    default:
        break;
    }
}

// Throws std::runtime_error unless the antenna cells of a row of T are a valid antenna description: the type is known,
// the columns are used by the type, the cells have the syntax of their kind. 'ci' gives the column of T for each
// column id, -1 = not in T
static inline void check_antenna_cells(const qd_csv::table &T, size_t r, const int *ci)
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

// layout::eval_antennas, with the description of each node in the messages ('where_tx', 'where_rx': one text per
// node, e.g. with file and line; nullptr = "Tx node '<name>', sector <s>"). Declared here, with its code in qd_layout.cpp:
// it is the entry to the antenna evaluation there (about 600 lines), as qd_arrayant_check in qd_helpers_arrayant.hpp
void eval_antennas(quadriga_lib::layout &L, const arma::vec &center_freq, const std::vector<std::string> *where_tx,
                   const std::vector<std::string> *where_rx);

// True if two frequencies are equal within a relative 1e-6
static inline bool same_freq(double a, double b) { return std::abs(a - b) <= 1.0e-6 * std::max(std::abs(a), std::abs(b)); }

// Index of the site (or terminal) of each node: nodes with equal names, numbered in order of first appearance
static inline std::vector<size_t> site_of_node(const quadriga_lib::layout_nodes &nodes, size_t *n_site = nullptr)
{
    std::unordered_map<std::string, size_t> index;
    std::vector<size_t> site(nodes.name.size());
    for (size_t i = 0; i < nodes.name.size(); ++i)
        site[i] = index.emplace(nodes.name[i], index.size()).first->second;
    if (n_site != nullptr)
        *n_site = index.size();
    return site;
}

// True if two numbers are equal, with NaN equal to NaN
static inline bool same_value(double a, double b) { return a == b || (a != a && b != b); }

// Number text of the layout files and antenna lines: plain digits from 1e-3 to below 1e6, else engineering notation
// with an exponent that is a multiple of 3 ("3.5e9", "28e9", "250e-6"), zero as "0", NaN and Inf as in qd_csv;
// digits = 0: the shortest text that reads back to the same double, else rounded to that many significant digits
static inline std::string format_value(double v, int digits = 0)
{
    if (!std::isfinite(v))
        return qd_csv::format_number(v);
    if (v == 0.0)
        return "0";
    // The significant digits and the decimal exponent, read from the "%.*e" text (the locale's point does not matter)
    std::string d;
    int x = 0;
    char buf[48];
    for (int p = digits > 0 ? std::min(digits, 17) : 1;; ++p)
    {
        std::snprintf(buf, sizeof(buf), "%.*e", p - 1, std::abs(v));
        const char *e = std::strchr(buf, 'e');
        d.clear();
        for (const char *c = buf; c < e; ++c)
            if (*c >= '0' && *c <= '9')
                d += *c;
        x = std::atoi(e + 1);
        if (digits > 0 || p >= 17)
            break;
        double w = 0.0;
        const std::string t = d.substr(0, 1) + (d.size() > 1 ? "." + d.substr(1) : std::string()) + "e" + std::to_string(x);
        if (qd_csv::parse_number(t, w) == qd_csv::number_status::ok && w == std::abs(v))
            break;
    }
    while (d.size() > 1 && d.back() == '0')
        d.pop_back();
    // The digits with the decimal point after n of them
    auto place = [&](int n) -> std::string
    {
        if (n <= 0)
            return "0." + std::string((size_t)-n, '0') + d;
        if ((size_t)n >= d.size())
            return d + std::string((size_t)n - d.size(), '0');
        return d.substr(0, (size_t)n) + "." + d.substr((size_t)n);
    };
    const std::string sign = v < 0.0 ? "-" : "";
    if (x >= -3 && x <= 5)
        return sign + place(x + 1);
    const int x3 = (int)std::floor((double)x / 3.0) * 3;
    return sign + place(x - x3 + 1) + "e" + std::to_string(x3);
}

// ---- Scenario families and the O2I draws ----

// One scenario family of TR 38.901: its conf names (file stems without the prefix "3GPP_38.901_v19_"; "" = none), the
// model of los_probability, the building O2I models of o2i_loss with their weights, the maximum indoor 2D distance and
// the share of building UTs when the indoor state is drawn. A family without O2I has no O2I confs, models or share
struct family
{
    std::string name;                     // Family, e.g. "UMa"
    std::string los_conf, nlos_conf;      // Confs of LOS and NLOS links of outdoor and car UTs; InF_HH has no NLOS
    std::string los_o2i_conf, nlos_o2i_conf; // Confs of building UTs; "" = no O2I
    std::string los_model;                // Model of los_probability
    std::vector<std::string> o2i_models;  // Building O2I models of o2i_loss; empty = no O2I
    std::vector<double> o2i_weights;      // Their probabilities, [n_model]
    double d2d_in_max;                    // Maximum indoor 2D distance in [m]; SMa: residential buildings; 0 = no O2I
    double d2d_in_max_commercial;         // SMa: commercial buildings; else equal to d2d_in_max
    double building_share;                // Share of building UTs of a drawn indoor state; 0 = no O2I
    double p_residential;                 // SMa: share of residential among the building UTs; 0 = no building classes
    bool car;                             // true: the other UTs are in cars (o2i_loss model "car"); false: outdoors
};

// The table of the families, and the entry of a family name (nullptr if the name is no family). Each translation unit
// has its own copy of the table, so family pointers must not cross translation units
static inline const std::vector<family> &families()
{
    // Conf names of the shipped files. LOS models of TR 38.901 Table 7.4.2-1. Building O2I models of clause 7.4.3.1: UMa
    // and UMi low and high loss with the weights of Table 7.8-1 (50 %, 50 %), RMa only the low-loss model (clause
    // 7.4.3.1), SMa the low-loss A model (Table 7.8-1A). Maximum d2D-in of clause 7.4.3.1: 25 m for UMa, UMi and SMa
    // commercial buildings, 10 m for RMa and SMa residential buildings. Building shares of a drawn state: Table 7.2-1 (80 %
    // indoor, the others outdoors), Table 7.2-3 (50 % indoor, 50 % in cars), Table 7.2-5 (80 % indoor, outdoor UTs in cars;
    // 90 % of the buildings residential)
    static const std::vector<family> T = {
        {"UMi", "UMi_LOS", "UMi_NLOS", "UMi_LOS_O2I", "UMi_NLOS_O2I", "38.901_UMi", {"low", "high"}, {0.5, 0.5}, 25.0, 25.0, 0.8, 0.0, false},
        {"UMa", "UMa_LOS", "UMa_NLOS", "UMa_LOS_O2I", "UMa_NLOS_O2I", "38.901_UMa", {"low", "high"}, {0.5, 0.5}, 25.0, 25.0, 0.8, 0.0, false},
        {"RMa", "RMa_LOS", "RMa_NLOS", "RMa_LOS_O2I", "RMa_NLOS_O2I", "38.901_RMa", {"low"}, {1.0}, 10.0, 10.0, 0.5, 0.0, true},
        {"SMa", "SMa_LOS", "SMa_NLOS", "SMa_LOS_O2I", "SMa_NLOS_O2I", "38.901_SMa", {"low_A"}, {1.0}, 10.0, 25.0, 0.8, 0.9, true},
        {"InH_Open", "Indoor_LOS", "Indoor_NLOS", "", "", "38.901_InH_Open", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InH_Mixed", "Indoor_LOS", "Indoor_NLOS", "", "", "38.901_InH_Mixed", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_SL", "InF_LOS", "InF_NLOS_SL", "", "", "38.901_InF_SL", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_DL", "InF_LOS", "InF_NLOS_DL", "", "", "38.901_InF_DL", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_SH", "InF_LOS", "InF_NLOS_SH", "", "", "38.901_InF_SH", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_DH", "InF_LOS", "InF_NLOS_DH", "", "", "38.901_InF_DH", {}, {}, 0.0, 0.0, 0.0, 0.0, false},
        {"InF_HH", "InF_LOS", "", "", "", "38.901_InF_HH", {}, {}, 0.0, 0.0, 0.0, 0.0, false}}; // InF-HH is 100 % LOS (Table 7.2-4)
    return T;
}

static inline const family *find_family(const std::string &name)
{
    for (const family &f : families())
        if (f.name == name)
            return &f;
    return nullptr;
}

// The random streams of the O2I state of a terminal; the purpose names in the stream keys are "io", "o2i_class",
// "o2i_model", "d2din1", "d2din2" and "o2i_dev". Reading of the three 50 m fields (TR 38.901 clause 7.6.3.3 has an indoor
// state field and a building-type field): "io" decides building or not, the other UTs are outdoors or in cars as the
// family says (no separate building-or-car field, which would repeat "io" where all non-building UTs are in cars);
// "o2i_class" decides residential or commercial in SMa; "o2i_model" is the building type, i.e. the O2I model
enum o2i_stream : int
{
    s_io,        // Indoor state: building UT if the value is below the share of building UTs
    s_o2i_class, // SMa: residential building if the value is below p_residential, else commercial
    s_o2i_model, // O2I model: column 0 of the input 'rnd' of o2i_loss
    s_d2din1,    // The two uniform variables of the indoor distance: columns 1 and 2 of 'rnd'
    s_d2din2,
    s_o2i_dev    // Deviation of the penetration loss, Normal(0, 1): column 3 of 'rnd'
};

// One spatially correlated O2I field of a family at the first positions 'xy' of terminals, [n, 2] in [m] (TR 38.901
// clauses 7.6.3.1 and 7.6.3.3, all-correlated): sos_rand (sos_randn for s_o2i_dev) with the preset Exp256, seeded with
// qd_layout_seed_hash(base_seed, "<purpose>|<family>"), with the decorrelation distance 50 m (s_io, s_o2i_class,
// s_o2i_model), 25 m (s_d2din1, s_d2din2) or 10 m (s_o2i_dev). Returns the values, [n]
static inline arma::vec o2i_field(arma::sword base_seed, const std::string &family, int stream, const arma::mat &xy)
{
    // "io": building or not (the others outdoors or in cars, by family); "o2i_class": SMa residential or commercial;
    // "o2i_model": building type (O2I model); see the comment at o2i_stream
    static const char *purpose[] = {"io", "o2i_class", "o2i_model", "d2din1", "d2din2", "o2i_dev"};
    // Decorrelation distances (TR 38.901 clause 7.6.3.3): indoor state 50 m (Table 7.6.3.1-2), building type 50 m (also
    // for the class residential or commercial), the uniform variables of the indoor distance 25 m, the deviation of the
    // penetration loss 10 m
    static const double dist[] = {50.0, 50.0, 50.0, 25.0, 25.0, 10.0};
    if (stream < s_io || stream > s_o2i_dev)
        throw std::invalid_argument("Input 'stream' is not an O2I stream.");
    if (xy.n_rows == 0)
        return arma::vec();
    const arma::sword seed = qd_layout_seed_hash(base_seed, std::string(purpose[stream]) + "|" + family);
    arma::mat val;
    if (stream == s_o2i_dev)
        quadriga_lib::sos_randn<double>(arma::vec{dist[stream]}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    else
        quadriga_lib::sos_rand<double>(arma::vec{dist[stream]}, xy, val, arma::mat(), arma::vec(), "Exp256", seed);
    return val.col(0);
}

// The input 'rnd' of o2i_loss of terminals at 'xy' [n, 2]: the fields s_o2i_model, s_d2din1, s_d2din2, s_o2i_dev as
// columns, [n, 4]
static inline arma::mat o2i_rnd(arma::sword base_seed, const std::string &family, const arma::mat &xy)
{
    arma::mat rnd(xy.n_rows, 4);
    for (int s = s_o2i_model; s <= s_o2i_dev; ++s)
        rnd.col((arma::uword)(s - s_o2i_model)) = o2i_field(base_seed, family, s, xy);
    return rnd;
}

// Indoor 2D distance of a building UT from the columns 1 and 2 of 'rnd', as o2i_loss gives it for its building models
// (TR 38.901 clause 7.4.3.1: the minimum of two uniform variables in [0, d2d_in_max])
static inline double d2d_in_of(double u1, double u2, double d2d_in_max) { return std::min(u1, u2) * d2d_in_max; }

// Small portable generator for the streams of one UT (SplitMix64): the state starts at the stream seed; each draw adds
// 0x9E3779B97F4A7C15 to the state, mixes it (xor-shift 30, times 0xBF58476D1CE4E5B9, xor-shift 27, times
// 0x94D049BB133111EB, xor-shift 31) and gives the upper 53 bits divided by 2^53, a uniform value in [0, 1). Seeding is
// free, unlike std::mt19937_64, which matters with one stream per UT
struct hash_rng
{
    unsigned long long s;
    explicit hash_rng(arma::sword seed) : s((unsigned long long)seed) {}
    double uniform()
    {
        unsigned long long z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        return (double)(z >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Floor of a generated UT, drawn without spatial correlation from the stream "floor|<family>|<terminal>" (hash_rng with
// that seed, two uniform draws u1, u2): the building has N_fl = n_min + floor(u1 (n_max - n_min + 1)) floors and the UT
// is on floor 1 + floor(u2 N_fl), i.e. N_fl ~ U{n_min ... n_max} and n_fl ~ U{1 ... N_fl} as in TR 36.873 Table 6-1;
// n_min = n_max = N gives U{1 ... N}
static inline int draw_floor(arma::sword base_seed, const std::string &family, const std::string &terminal, int n_min, int n_max)
{
    hash_rng rng(qd_layout_seed_hash(base_seed, "floor|" + family + "|" + terminal));
    const double u1 = rng.uniform(), u2 = rng.uniform();
    const int n_fl = std::min(n_min + (int)std::floor(u1 * (double)(n_max - n_min + 1)), n_max);
    return std::min(1 + (int)std::floor(u2 * (double)n_fl), n_fl);
}

// Floor of a UT from its height when 'floor' is -1: max(round((z - 1.5) / 3) + 1, 1), the inverse of hUT = 3 (n_fl - 1) + 1.5
static inline int floor_from_z(double z) { return std::max((int)std::floor((z - 1.5) / 3.0 + 0.5) + 1, 1); }

} // namespace qd_layout

#endif
