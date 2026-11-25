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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
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

// Number text of the layout files and antenna lines: plain digits from 1e-3 to below 1e6, else engineering notation
// with an exponent that is a multiple of 3 ("3.5e9", "28e9", "250e-6"), zero as "0", NaN and Inf as in qd_csv;
// digits = 0: the shortest text that reads back to the same double, else rounded to that many significant digits
std::string format_value(double v, int digits = 0);

// ---- Scenario families and the O2I draws (generate_layout.cpp) ----

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

// The table of the families, and the entry of a family name (nullptr if the name is no family)
const std::vector<family> &families();
const family *find_family(const std::string &name);

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
arma::vec o2i_field(arma::sword base_seed, const std::string &family, int stream, const arma::mat &xy);

// The input 'rnd' of o2i_loss of terminals at 'xy' [n, 2]: the fields s_o2i_model, s_d2din1, s_d2din2, s_o2i_dev as
// columns, [n, 4]
arma::mat o2i_rnd(arma::sword base_seed, const std::string &family, const arma::mat &xy);

// Indoor 2D distance of a building UT from the columns 1 and 2 of 'rnd', as o2i_loss gives it for its building models
// (TR 38.901 clause 7.4.3.1: the minimum of two uniform variables in [0, d2d_in_max])
inline double d2d_in_of(double u1, double u2, double d2d_in_max) { return std::min(u1, u2) * d2d_in_max; }

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
int draw_floor(arma::sword base_seed, const std::string &family, const std::string &terminal, int n_min, int n_max);

// Floor of a UT from its height when 'floor' is -1: max(round((z - 1.5) / 3) + 1, 1), the inverse of hUT = 3 (n_fl - 1) + 1.5
inline int floor_from_z(double z) { return std::max((int)std::floor((z - 1.5) / 3.0 + 0.5) + 1, 1); }

} // namespace qd_layout

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

// ---- Channel pipeline: helpers shared by init_link_states.cpp (stage 1) and get_channels_layout.cpp (stages 2 and 3) ----

// A scenario family of the table in generate_layout.cpp
using qd_family = qd_layout::family;

static const std::string qd_v19 = "3GPP_38.901_v19_"; // Prefix of the shipped TR 38.901 conf files

// The sites of the Tx side and the terminals of the Rx side: index of each node, the nodes of each
struct qd_sides
{
    std::vector<size_t> site_of_tx, term_of_rx;
    size_t n_site = 0, n_term = 0;
    std::vector<std::vector<arma::uword>> site_nodes, term_nodes;
};

// The sites and terminals of a layout
static inline qd_sides qd_make_sides(const quadriga_lib::layout &L)
{
    qd_sides S;
    S.site_of_tx = qd_layout::site_of_node(L.tx, &S.n_site);
    S.term_of_rx = qd_layout::site_of_node(L.rx, &S.n_term);
    S.site_nodes.resize(S.n_site), S.term_nodes.resize(S.n_term);
    for (size_t i = 0; i < S.site_of_tx.size(); ++i)
        S.site_nodes[S.site_of_tx[i]].push_back((arma::uword)i);
    for (size_t i = 0; i < S.term_of_rx.size(); ++i)
        S.term_nodes[S.term_of_rx[i]].push_back((arma::uword)i);
    return S;
}

// Rx rows at which the segments of terminal t start (indices into the rows of the Rx side): the first row and each row
// whose scenario or indoor state differs from the row before it, on the first node of the terminal
static inline std::vector<arma::uword> qd_segment_starts(const quadriga_lib::layout &L, const qd_sides &S, size_t t)
{
    const arma::uword i = S.term_nodes[t][0], r0 = L.rx.offset[i], n = L.rx.n_row(i);
    std::vector<arma::uword> v = {r0};
    for (arma::uword r = r0 + 1; r < r0 + n; ++r)
        if ((!L.rx.scenario.empty() && L.rx.scenario[r] != L.rx.scenario[r - 1]) || (!L.rx.indoor.is_empty() && L.rx.indoor[r] != L.rx.indoor[r - 1]))
            v.push_back(r);
    return v;
}

// Moving site: more than one row or a speed (dual mobility of the GBSM functions); all nodes of a site have equal motion
static inline bool qd_site_moves(const quadriga_lib::layout_nodes &N, arma::uword i_node)
{
    const arma::uword first = N.offset[i_node], n = N.n_row(i_node);
    if (n > 1)
        return true;
    return !N.speed.is_empty() && N.speed[first] != 0.0;
}

// Time of row r of node i as sample_track gives it: its 'time', else the arc length over the speed from the first row
// (a point drop: 0); false for a track with neither
static inline bool qd_row_time(const quadriga_lib::layout_nodes &N, arma::uword i, arma::uword r, double &t)
{
    const arma::uword first = N.offset[i], n = N.n_row(i);
    if (!N.time.is_empty() && !std::isnan(N.time[first]))
        return t = N.time[r], true;
    bool timed = n == 1;
    for (arma::uword k = first; k < first + n && !N.speed.is_empty(); ++k)
        timed = timed || N.speed[k] != 0.0;
    t = 0.0;
    for (arma::uword k = first; k < r; ++k)
    {
        const double len = arma::norm(N.pos.col(k + 1) - N.pos.col(k)), v = N.speed.is_empty() ? 0.0 : N.speed[k];
        t += (len > 0.0 && v > 0.0) ? len / v : 0.0;
    }
    return timed;
}

// Position of Tx node i_tx paired with Rx row r of node i_rx (the start of a segment), without the wrap image: a static
// node its row; a moving one sampled at the time of the row if both nodes have times, else (the pairing of snapshots
// without times) the Tx row with the index of r in its node, the last one for a shorter Tx track; [3]
static inline arma::vec qd_tx_at(const quadriga_lib::layout &L, arma::uword i_tx, arma::uword i_rx, arma::uword r)
{
    const arma::uword ft = L.tx.offset[i_tx], nt = L.tx.n_row(i_tx);
    if (!qd_site_moves(L.tx, i_tx))
        return L.tx.pos.col(ft);
    double t = 0.0, t_tx = 0.0;
    if (qd_row_time(L.rx, i_rx, r, t) && qd_row_time(L.tx, i_tx, ft, t_tx))
    {
        arma::mat p, o;
        quadriga_lib::sample_track(L.tx, i_tx, arma::vec{t}, p, o);
        return p.col(0);
    }
    return L.tx.pos.col(ft + std::min(r - L.rx.offset[i_rx], nt - 1));
}

// The wrap images: 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); one vector: 0, +a1, -a1; [2, n_img]
static inline arma::mat qd_images(const arma::mat &wrap)
{
    arma::mat I(2, 1, arma::fill::zeros);
    if (wrap.n_cols >= 1)
        I = arma::join_rows(I, wrap.col(0), -wrap.col(0));
    if (wrap.n_cols == 2)
        I = arma::join_rows(I, arma::join_rows(wrap.col(1), -wrap.col(1), wrap.col(0) - wrap.col(1), wrap.col(1) - wrap.col(0)));
    return I;
}

// A scenario name resolved: a family (states drawn), a conf of the family table (states fixed), or a user conf
struct qd_scen
{
    const qd_family *fam = nullptr; // Family; nullptr for a user conf
    std::string name;               // The name in the layout
    int los = -1, building = -1;    // Fixed by an exact name, -1 = drawn
    std::string exact() const       // The conf of an exact name
    {
        if (los < 0)
            return "";
        return building ? (los ? fam->los_o2i_conf : fam->nlos_o2i_conf) : (los ? fam->los_conf : fam->nlos_conf);
    }
    bool has_o2i() const { return fam != nullptr && fam->d2d_in_max > 0.0; }
};

// A family, or a conf name of the family table (with or without the prefix of the shipped files), else a user conf. A
// conf that several families list (Indoor_LOS, InF_LOS) belongs to 'prefer' if that lists it, else to the first
static inline qd_scen qd_resolve(const std::string &name, const qd_family *prefer)
{
    qd_scen s;
    s.name = name;
    if ((s.fam = qd_layout::find_family(name)) != nullptr)
        return s;
    const std::string n = name.compare(0, qd_v19.size(), qd_v19) == 0 ? name.substr(qd_v19.size()) : name;
    auto lists = [&](const qd_family &f)
    {
        const std::string *c[4] = {&f.los_conf, &f.nlos_conf, &f.los_o2i_conf, &f.nlos_o2i_conf};
        for (int k = 0; k < 4; ++k)
            if (!c[k]->empty() && *c[k] == n)
            {
                s.fam = &f, s.los = (k % 2 == 0) ? 1 : 0, s.building = k / 2;
                return true;
            }
        return false;
    };
    if (prefer != nullptr && lists(*prefer))
        return s;
    for (const qd_family &f : qd_layout::families())
        if (lists(f))
            return s;
    s.fam = nullptr;
    return s;
}

// The conf of a link: a user conf, the conf of an exact name, or from the family with the LOS and indoor state
static inline std::string qd_conf_of(const qd_scen &s, unsigned los, int indoor)
{
    if (s.fam == nullptr)
        return s.name;
    if (s.los >= 0)
        return s.exact();
    const bool building = (indoor == 1 || indoor == 3) && s.has_o2i();
    return building ? (los ? s.fam->los_o2i_conf : s.fam->nlos_o2i_conf) : (los ? s.fam->los_conf : s.fam->nlos_conf);
}

// The resolved scenario of the link of site s and terminal t at the Rx row r_seg: the value of the terminal, else that
// of the site ("" = none); an exact name prefers the family of the site's scenario. Cached by name and family
struct qd_scenarios
{
    std::map<std::pair<std::string, const qd_family *>, qd_scen> cache;
    const qd_scen &get(const std::string &name, const qd_family *prefer)
    {
        auto it = cache.find({name, prefer});
        if (it == cache.end())
            it = cache.emplace(std::make_pair(name, prefer), qd_resolve(name, prefer)).first;
        return it->second;
    }
    const qd_scen &of(const quadriga_lib::layout &L, const qd_sides &S, size_t s, arma::uword r_seg, bool *from_rx = nullptr)
    {
        static const std::string none;
        const std::string &tx = L.tx.scenario.empty() ? none : L.tx.scenario[L.tx.offset[S.site_nodes[s][0]]];
        const std::string &rx = L.rx.scenario.empty() ? none : L.rx.scenario[r_seg];
        if (from_rx != nullptr)
            *from_rx = !rx.empty();
        const qd_family *site_family = tx.empty() ? nullptr : get(tx, nullptr).fam;
        return get(rx.empty() ? tx : rx, site_family);
    }
};


// The base seed of the streams of a site: its seed, else the master seed
static inline arma::sword qd_site_base(const quadriga_lib::layout_nodes &tx, arma::uword i_node, arma::sword master)
{
    return (!tx.seed.is_empty() && tx.seed[i_node] != -1) ? (arma::sword)tx.seed[i_node] : master;
}

// The O2I family of each terminal: the family of the first of its links (in site order) whose scenario is a family
// with O2I; nullptr = none. All links of a terminal share its O2I state (TR 38.901 clause 7.6.3.3)
static inline std::vector<const qd_family *> qd_term_families(const std::vector<qd_scen> &scen, const arma::u32_vec &i_rx, size_t n_term)
{
    std::vector<const qd_family *> F(n_term, nullptr);
    for (arma::uword l = 0; l < i_rx.n_elem; ++l)
        if (F[i_rx[l]] == nullptr && scen[l].has_o2i())
            F[i_rx[l]] = scen[l].fam;
    return F;
}

// Maximum indoor distance of a building state of a family (SMa: residential 3, commercial 1)
static inline double qd_d2d_in_max(const qd_family &f, int indoor) { return indoor == 1 ? f.d2d_in_max_commercial : f.d2d_in_max; }

// Floor of a building UT at Rx row r: the floor column, else from the height
static inline int qd_floor(const quadriga_lib::layout_nodes &rx, arma::uword r)
{
    const int fl = rx.floor.is_empty() ? -1 : rx.floor[r];
    return fl >= 1 ? fl : qd_layout::floor_from_z(rx.pos(2, r));
}

// Floor index and indoor 2D distance of a link in the indoor state 'io' of a terminal of family f (nullptr = no O2I):
// building UTs on their floor at the distance of the columns c1, c2 of the O2I inputs, as o2i_loss gives it; others on
// floor 0 at 0 m. Used for the drawn states in stage 1 and for edited states in stages 2 and 3
static inline void qd_o2i_geometry(const qd_family *f, int io, double c1, double c2, int floor, arma::u32 &floor_id, double &d2d_in)
{
    const bool building = f != nullptr && (io == 1 || io == 3);
    floor_id = building ? (arma::u32)(floor - 1) : 0;
    d2d_in = building ? qd_layout::d2d_in_of(c1, c2, qd_d2d_in_max(*f, io)) : 0.0;
}

// Byte-wise 64-bit FNV-1a
struct qd_fnv
{
    unsigned long long h = 14695981039346656037ULL;
    void bytes(const void *p, size_t n)
    {
        const unsigned char *c = (const unsigned char *)p;
        for (size_t i = 0; i < n; ++i)
            h = (h ^ c[i]) * 1099511628211ULL;
    }
    void u64(unsigned long long v) { bytes(&v, sizeof(v)); }
    void dbl(double v)
    {
        if (v == 0.0)
            v = 0.0; // -0 and 0 are equal
        if (std::isnan(v))
            v = std::numeric_limits<double>::quiet_NaN();
        bytes(&v, sizeof(v));
    }
    void str(const std::string &s) { u64(s.size()), bytes(s.data(), s.size()); }
};

// Hash of the members of a layout that determine its links; empty optional members hash as their default values. The
// layout seed is not part of it: the links hold the master seed they were drawn with, so a stored copy of the links
// stays usable after another realisation
static inline unsigned long long qd_links_hash(const quadriga_lib::layout &L)
{
    qd_fnv H;
    H.u64(L.center_freq.n_elem);
    for (double f : L.center_freq)
        H.dbl(f);
    H.u64(L.wrap.n_rows), H.u64(L.wrap.n_cols);
    for (double w : L.wrap)
        H.dbl(w);
    for (const quadriga_lib::layout_nodes *N : {&L.tx, &L.rx})
    {
        const arma::uword nn = N->n_node(), nr = N->pos.n_cols;
        H.u64(nn), H.u64(nr);
        for (arma::uword i = 0; i < nn; ++i)
        {
            H.str(N->name[i]), H.u64(N->sector[i]), H.u64(N->offset[i]);
            H.u64(N->seed.is_empty() ? (unsigned long long)-1 : (unsigned long long)N->seed[i]);
            for (arma::uword k = 0; k < 3; ++k)
                H.dbl(N->clutter.is_empty() ? arma::datum::nan : N->clutter(k, i));
            if (N == &L.tx)
                H.u64(qd_site_moves(*N, i) ? 1 : 0);
        }
        for (arma::uword r = 0; r < nr; ++r)
        {
            H.dbl(N->pos(0, r)), H.dbl(N->pos(1, r)), H.dbl(N->pos(2, r));
            H.str(N->scenario.empty() ? std::string() : N->scenario[r]);
            H.u64(N->indoor.is_empty() ? (unsigned long long)-1 : (unsigned long long)N->indoor[r]);
            H.u64(N->floor.is_empty() ? (unsigned long long)-1 : (unsigned long long)N->floor[r]);
            H.dbl(N->d2d_in.is_empty() ? arma::datum::nan : N->d2d_in[r]);
        }
        std::vector<std::string> keys;
        for (const auto &kv : N->scenpar_override)
            keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (const std::string &k : keys)
        {
            H.str(k);
            for (double v : N->scenpar_override.at(k))
                H.dbl(v);
        }
    }
    return H.h;
}

// Throws unless 'method' is one of the methods of the channel pipeline
static inline void qd_check_method(const std::string &method)
{
    if (method != "38.901" && method != "38.901_MF" && method != "quadriga")
        throw std::invalid_argument("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
}

// Stage 1 into 'K' without changing the layout (init_link_states.cpp); get_channels_layout draws local links with it
void qd_init_links(const quadriga_lib::layout &L, const std::string &method, const arma::vec &o2i_low, bool pol_variability,
                   double anchor_freq, const std::string &sos_preset, arma::sword seed, const std::vector<std::string> &config_dirs,
                   quadriga_lib::layout_links &K);

#endif
