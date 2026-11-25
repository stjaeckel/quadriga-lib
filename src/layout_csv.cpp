// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "qd_csv.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# layout_csv_read
Read the Tx and Rx CSV files of a layout

- Reads the Tx file and the Rx file of a layout in the [Layout CSV format](formats.html#layout_csv) and returns the
  layout, with angles in rad
- Generates the antennas of the nodes from their antenna columns, unless `eval_antennas` is `false`
- Keeps unknown columns and returns their names, so that a typo in a column name is visible

## Declaration:
```
quadriga_lib::layout quadriga_lib::layout_csv_read(
    const std::string &fn_tx,
    const std::string &fn_rx,
    const arma::vec &center_freq = {},
    bool eval_antennas = true,
    std::vector<std::string> *extra_tx = nullptr,
    std::vector<std::string> *extra_rx = nullptr);
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = no Tx nodes or no Rx nodes
- **`center_freq`** *(optional)* — Simulation frequencies in [Hz], strictly ascending and positive; empty = from the
  files; `[n_freq]`
- **`eval_antennas`** *(optional)* — `true`: generate the antennas; `false`: `antenna` stays empty, `antenna_args` and
  `i_antenna` are kept, and the nodes use the generic antenna until `layout.eval_antennas()` is called

## Outputs:
- **`extra_tx`**, **`extra_rx`** *(optional)* — Names of the unknown columns of the Tx and Rx file, in file order;
  written only if the call succeeds

## Returns:
- `quadriga_lib::layout` — The layout; `name` and `links` are empty

## Notes:
- **Errors**: a violation of the format and a layout that is not valid ([[layout]]) throw `std::runtime_error` with the
  file name, the line and the column (see [Reading](formats.html#layout-csv-reading)); with `eval_antennas`, so do a
  line that [[generate_antenna]] rejects, a QDANT file that cannot be read and an array spaced in wavelengths in a
  layout without frequencies
- **Antennas**: equal descriptions share one entry of the antenna lists, Tx nodes first; a relative `qdant_file` is
  stored as a normalised absolute path
- **Memory**: large arrays on the default 1° grid need much memory ([Antennas](formats.html#layout-csv-antennas)); give
  a `res` column or read with `eval_antennas = false`

## See also:
- [[layout_csv_write]] (stores a layout)
- [[layout]] (the returned class)
- [[generate_antenna]] (the antenna types and arguments of the antenna columns)
MD!*/

/*!MD
# layout_csv_write
Write a layout to the Tx and Rx CSV files

- Writes the Tx nodes to the Tx file and the Rx nodes to the Rx file in the
  [Layout CSV format](formats.html#layout_csv); reading the files back gives the same layout within 15 significant
  digits, except for `name` and `links`, which are not stored
- Antennas without a description, e.g. a measured array set with the array form of [[set_antenna]], are written to QDANT
  files next to the CSV file
- A failed call leaves the existing files unchanged

## Declaration:
```
void quadriga_lib::layout_csv_write(
    const std::string &fn_tx,
    const std::string &fn_rx,
    const quadriga_lib::layout &layout,
    bool compact = true);
```

## Inputs:
- **`fn_tx`**, **`fn_rx`** — Tx and Rx CSV files; "" = not written; different files (also if the paths are spelled
  differently)
- **`layout`** — Layout; must be valid ([[layout]])
- **`compact`** *(optional)* — `true`: cells that the reader fills in are left empty; `false`: every cell is written

## Notes:
- **Rules of the writer**: the columns written, the number format, the compact form, the names of the QDANT files and
  what cannot be written are described in [Writing](formats.html#layout-csv-writing)
- **QDANT files**: one file `<stem>_<k>.qdant` per antenna without description, with one entry per band. Files of this
  name pattern next to a written CSV file that the new files do not refer to are deleted. The arrays must be evaluated:
  a layout whose `antenna` does not hold them throws. An antenna set from a QDANT file ([[set_antenna]]) is written as a
  reference to its file, not exported

## See also:
- [[layout_csv_read]] (reads a layout)
- [[layout]] (the class that is written)
MD!*/

using namespace qd_layout;
namespace fs = std::filesystem;

namespace
{
constexpr size_t npos = qd_csv::table::npos;
const double qd_nan = std::numeric_limits<double>::quiet_NaN();

using acells = std::array<std::string, n_antenna_col>; // Cells of the antenna columns, index = column id - c_center_freq

// One band of a node: the antenna cells of the row that defines it
struct band_cells
{
    bool has_freq = false; // The row has a center_freq
    double freq = 0.0;
    bool content = false;  // Some antenna cell other than center_freq is set
    size_t row = 0;        // Table row, for messages
    acells cells;
};

// One side of a layout as read from its file
struct side_data
{
    std::string fn;
    flat F;
    std::vector<std::vector<band_cells>> bands; // [n_node]
    std::vector<long> ref;                      // Sector 0 of the site for an extra sector, else -1
    std::vector<size_t> first_line;             // Line of the first row of each node
    arma::mat wrap;
    long long layout_seed = -1;
    bool has_pairing = false;
};

// Path of a QDANT file relative to the folder of the CSV file, as a normalised absolute path with '/' separators
std::string qd_resolve_qdant(const std::string &csv_fn, const std::string &cell)
{
    fs::path p(cell);
    if (p.is_relative())
        p = fs::path(csv_fn).parent_path() / p;
    return fs::absolute(p).lexically_normal().generic_string();
}

// The antenna cells of one row; with 'check', they are validated against the antenna type
acells qd_antenna_row(const qd_csv::table &T, size_t r, const int *ci, const std::string &fn, bool check, bool *content)
{
    if (check)
        check_antenna_cells(T, r, ci);
    acells a;
    bool any = false;
    for (int k = 0; k < n_antenna_col; ++k)
    {
        const int id = c_center_freq + k;
        if (ci[id] == (int)npos)
            continue;
        a[(size_t)k] = std::string(T.cell(r, (size_t)ci[id]));
        any = any || (id != c_center_freq && !a[(size_t)k].empty());
    }
    if (content != nullptr)
        *content = any;
    if (any && !a[c_qdant_file - c_center_freq].empty())
        a[c_qdant_file - c_center_freq] = qd_resolve_qdant(fn, a[c_qdant_file - c_center_freq]);
    return a;
}

// Equality of fixed values; NaN equals NaN
inline bool equal_value(double a, double b) { return same_value(a, b); }
template <class T>
bool equal_value(const T &a, const T &b) { return a == b; }

side_data qd_read_side(const std::string &fn, bool is_tx, std::vector<std::string> *extra_names)
{
    side_data S;
    S.fn = fn;
    const qd_csv::table T = qd_csv::read(fn);
    const size_t n_c = T.n_cols();

    // Header: the column of each table entry, the scenario parameters and the unknown columns
    int ci[n_col];
    for (int &x : ci)
        x = (int)npos;
    std::vector<std::pair<std::string, size_t>> scenpar_col;
    std::vector<size_t> extra_col;
    for (size_t c = 0; c < n_c; ++c)
    {
        const int id = find_column(T.header[c]);
        if (id < 0)
        {
            extra_col.push_back(c);
            continue;
        }
        const col_side side = column_of(id).side;
        if ((side == col_side::tx && !is_tx) || (side == col_side::rx && is_tx))
            throw std::runtime_error(fn + ": the column '" + T.header[c] + "' is only used in the " + (is_tx ? "Rx" : "Tx") + " file.");
        if (id == c_scenpar)
            scenpar_col.emplace_back(T.header[c].substr(8), c);
        else
            ci[id] = (int)c;
    }
    for (int id : {c_name, c_x, c_y, c_z})
        if (ci[id] == (int)npos)
            throw std::runtime_error(fn + ": the required column '" + column_of(id).name + "' is missing.");
    if (extra_names != nullptr)
        for (size_t c : extra_col)
            extra_names->push_back(T.header[c]);

    // Pass 1: nodes and the role of each row
    flat &F = S.F;
    std::unordered_map<std::string, size_t> node_index;
    std::vector<std::vector<size_t>> rows, band_rows; // [n_node]
    std::vector<unsigned> sector;
    const size_t n_rows = T.n_rows();
    for (size_t r = 0; r < n_rows; ++r)
    {
        const std::string name(T.cell(r, (size_t)ci[c_name]));
        if (name.empty() || name.size() > 255)
            throw std::runtime_error(T.where(r, (size_t)ci[c_name]) + " must have 1 to 255 bytes.");
        unsigned sec = 0;
        if (ci[c_sector] != (int)npos && !T.cell(r, (size_t)ci[c_sector]).empty())
            sec = (unsigned)cell_number(T, r, (size_t)ci[c_sector], c_sector);
        const auto ins = node_index.emplace(std::to_string(sec) + ":" + name, rows.size());
        if (ins.second)
        {
            F.name.push_back(name);
            sector.push_back(sec);
            rows.emplace_back(), band_rows.emplace_back();
            rows.back().push_back(r);
            continue;
        }
        const size_t n = ins.first->second;
        const bool no_pos = T.cell(r, (size_t)ci[c_x]).empty() && T.cell(r, (size_t)ci[c_y]).empty() && T.cell(r, (size_t)ci[c_z]).empty();
        if (no_pos && ci[c_center_freq] != (int)npos && !T.cell(r, (size_t)ci[c_center_freq]).empty())
            band_rows[n].push_back(r);
        else
            rows[n].push_back(r);
    }
    const size_t N = F.n_node = rows.size();
    F.offset.resize(N);
    size_t R = 0;
    for (size_t n = 0; n < N; ++n)
        F.offset[n] = R, R += rows[n].size();
    F.n_row = R;

    // Storage of the present columns, filled with the defaults
    auto alloc = [&](int id)
    {
        const bool per_node = column_of(id).cls == col_class::fixed || column_of(id).cls == col_class::identity;
        F.v[id].assign(per_node ? N : R, column_of(id).dflt);
    };
    for (int id = c_sector; id <= c_relative; ++id)
        if (id != c_scenpar && id != c_wrap && id != c_seed && id != c_layout_seed && id != c_pairing && id != c_scenario && (ci[id] != (int)npos || id == c_x || id == c_y || id == c_z || id == c_sector))
            alloc(id);
    if (ci[c_scenario] != (int)npos)
        F.txt[c_scenario].assign(R, std::string());
    if (ci[c_seed] != (int)npos)
        F.seed.assign(N, -1);
    if (ci[c_pairing] != (int)npos)
        F.txt[c_pairing].assign(N, std::string());
    for (const auto &kv : scenpar_col)
        F.scenpar[kv.first].assign(N, qd_nan);
    for (size_t k = 0; k < N; ++k)
        F.v[c_sector][k] = (double)sector[k];
    for (int id : {c_bank, c_tilt, c_heading})
        if (F.v[id].empty())
            F.v[id].assign(R, 0.0);

    // Sector 0 of the site is the reference of an extra sector
    std::unordered_map<std::string, size_t> sector0;
    for (size_t n = 0; n < N; ++n)
        if (sector[n] == 0)
            sector0[F.name[n]] = n;
    S.ref.assign(N, -1);
    for (size_t n = 0; n < N; ++n)
        if (sector[n] != 0 && sector0.count(F.name[n]) != 0)
            S.ref[n] = (long)sector0[F.name[n]];
    S.bands.resize(N);
    S.first_line.resize(N);

    for (size_t c : extra_col)
        F.extra_names.push_back(T.header[c]), F.extra.emplace_back(R);

    auto cell = [&](size_t r, int id) { return T.cell(r, (size_t)ci[id]); };
    bool wrap_set = false, seed_set = false;

    // Pass 2: the values of one node
    auto resolve = [&](size_t n)
    {
        const std::vector<size_t> &R_n = rows[n];
        const size_t K = R_n.size(), o = F.offset[n];
        const long ref = S.ref[n];
        S.first_line[n] = T.line(R_n[0]);
        const std::string who = " (node '" + F.name[n] + "', sector " + std::to_string(sector[n]) + ")";

        // Positions: empty only on the single row of an extra sector of a static site
        for (size_t k = 0; k < K; ++k)
        {
            const size_t r = R_n[k];
            const bool e[3] = {cell(r, c_x).empty(), cell(r, c_y).empty(), cell(r, c_z).empty()};
            if (e[0] && e[1] && e[2] && k == 0 && K == 1 && ref >= 0 && rows[(size_t)ref].size() == 1)
            {
                for (int i = 0; i < 3; ++i)
                    F.v[c_x + i][o] = F.v[c_x + i][F.offset[(size_t)ref]];
                continue;
            }
            for (int i = 0; i < 3; ++i)
                if (e[i])
                    throw std::runtime_error(T.where(r, (size_t)ci[c_x + i]) + " is empty" + who + ".");
            for (int i = 0; i < 3; ++i)
                F.v[c_x + i][o + k] = T.get_double(r, (size_t)ci[c_x + i], 0.0, false, false);
        }

        if (ci[c_time] != (int)npos)
        {
            const bool first_empty = cell(R_n[0], c_time).empty();
            for (size_t k = 0; k < K; ++k)
            {
                if (cell(R_n[k], c_time).empty() != first_empty)
                    throw std::runtime_error(T.where(R_n[k], (size_t)ci[c_time]) + " must be given on all rows of a node or on none" + who + ".");
                if (!first_empty)
                    F.v[c_time][o + k] = cell_number(T, R_n[k], (size_t)ci[c_time], c_time);
            }
        }

        // Held values
        for (int id = c_bank; id <= c_d2d_in; ++id)
        {
            if (ci[id] == (int)npos)
                continue;
            if (id == c_scenario)
            {
                std::string prev = ref >= 0 ? F.txt[c_scenario][F.offset[(size_t)ref]] : std::string();
                for (size_t k = 0; k < K; ++k)
                {
                    const std::string_view t = cell(R_n[k], c_scenario);
                    if (!t.empty())
                        prev = std::string(t);
                    F.txt[c_scenario][o + k] = prev;
                }
                continue;
            }
            double prev = ref >= 0 ? F.v[id][F.offset[(size_t)ref]] : column_of(id).dflt;
            for (size_t k = 0; k < K; ++k)
            {
                if (!cell(R_n[k], id).empty())
                    prev = cell_number(T, R_n[k], (size_t)ci[id], id);
                F.v[id][o + k] = prev;
            }
        }

        // Fixed values: the first row, for an extra sector the value of sector 0; later rows are empty or equal
        auto fixed = [&](size_t col, auto &out, auto dflt, auto parse)
        {
            auto value = ref >= 0 ? out[(size_t)ref] : dflt;
            if (!T.cell(R_n[0], col).empty())
                value = parse(R_n[0]);
            for (size_t k = 1; k < K; ++k)
                if (!T.cell(R_n[k], col).empty() && !equal_value(parse(R_n[k]), value))
                    throw std::runtime_error(T.where(R_n[k], col) + " differs from the first row of the node" + who + ".");
            out[n] = value;
        };
        for (int id : {c_d_clutter, c_r_clutter, c_h_clutter, c_cell, c_relative})
            if (ci[id] != (int)npos)
                fixed((size_t)ci[id], F.v[id], column_of(id).dflt, [&](size_t r) { return cell_number(T, r, (size_t)ci[id], id); });
        if (ci[c_seed] != (int)npos)
            fixed((size_t)ci[c_seed], F.seed, -1LL, [&](size_t r) { return cell_int64(T, r, (size_t)ci[c_seed], c_seed); });
        for (const auto &kv : scenpar_col)
            fixed(kv.second, F.scenpar[kv.first], qd_nan, [&](size_t r) { return cell_number(T, r, kv.second, c_scenpar); });
        if (ci[c_pairing] != (int)npos)
            fixed((size_t)ci[c_pairing], F.txt[c_pairing], std::string(), [&](size_t r) { return std::string(cell(r, c_pairing)); });

        // Layout-wide values: wrap and layout_seed must be equal wherever they are given
        for (size_t k = 0; k < K; ++k)
        {
            if (ci[c_wrap] != (int)npos && !cell(R_n[k], c_wrap).empty())
            {
                arma::mat M = T.get_matrix(R_n[k], (size_t)ci[c_wrap], false, false);
                if (M.n_cols != 2 || M.n_rows > 2)
                    throw std::runtime_error(T.where(R_n[k], (size_t)ci[c_wrap]) + " must have 1 or 2 rows (one lattice vector each) of 2 entries.");
                M = M.t();
                if (wrap_set && (M.n_cols != S.wrap.n_cols || !arma::all(arma::vectorise(M == S.wrap))))
                    throw std::runtime_error(T.where(R_n[k], (size_t)ci[c_wrap]) + " differs from the 'wrap' of an earlier row.");
                S.wrap = M, wrap_set = true;
            }
            if (ci[c_layout_seed] != (int)npos && !cell(R_n[k], c_layout_seed).empty())
            {
                const long long v = cell_int64(T, R_n[k], (size_t)ci[c_layout_seed], c_layout_seed);
                if (seed_set && v != S.layout_seed)
                    throw std::runtime_error(T.where(R_n[k], (size_t)ci[c_layout_seed]) + " differs from the 'layout_seed' of an earlier row.");
                S.layout_seed = v, seed_set = true;
            }
        }

        // Unknown columns are kept as text; they are not allowed in band rows
        for (size_t e = 0; e < extra_col.size(); ++e)
            for (size_t k = 0; k < K; ++k)
                F.extra[e][o + k] = std::string(T.cell(R_n[k], extra_col[e]));

        // Antennas: the first row defines the first band; later rows of the node repeat or leave it empty
        std::vector<band_cells> &bands = S.bands[n];
        {
            band_cells b;
            b.row = R_n[0];
            b.cells = qd_antenna_row(T, R_n[0], ci, fn, true, &b.content);
            b.has_freq = !b.cells[0].empty();
            b.freq = b.has_freq ? cell_number(T, R_n[0], (size_t)ci[c_center_freq], c_center_freq) : 0.0;
            bands.push_back(b);
        }
        for (size_t k = 1; k < K; ++k)
        {
            const acells a = qd_antenna_row(T, R_n[k], ci, fn, false, nullptr);
            for (int j = 0; j < n_antenna_col; ++j)
                if (!a[(size_t)j].empty() && a[(size_t)j] != bands[0].cells[(size_t)j] &&
                    !(j == 0 && bands[0].has_freq && cell_number(T, R_n[k], (size_t)ci[c_center_freq], c_center_freq) == bands[0].freq))
                    throw std::runtime_error(T.where(R_n[k], (size_t)ci[c_center_freq + j]) + " differs from the first row of the node" + who + ".");
        }
        for (size_t r : band_rows[n])
        {
            // A band row has the identity columns, center_freq and antenna columns only
            for (int id = c_sector + 1; id < c_center_freq; ++id)
                if (ci[id] != (int)npos && !cell(r, id).empty())
                    throw std::runtime_error(T.where(r, (size_t)ci[id]) + " must be empty in a band row" + who + ".");
            for (size_t c : extra_col)
                if (!T.cell(r, c).empty())
                    throw std::runtime_error(T.where(r, c) + " must be empty in a band row" + who + ".");
            for (const auto &kv : scenpar_col)
                if (!T.cell(r, kv.second).empty())
                    throw std::runtime_error(T.where(r, kv.second) + " must be empty in a band row" + who + ".");
            band_cells b;
            b.row = r;
            b.cells = qd_antenna_row(T, r, ci, fn, true, &b.content);
            b.has_freq = true;
            b.freq = cell_number(T, r, (size_t)ci[c_center_freq], c_center_freq);
            bands.push_back(b);
        }
    };
    for (size_t n = 0; n < N; ++n)
        if (S.ref[n] < 0)
            resolve(n);
    for (size_t n = 0; n < N; ++n)
        if (S.ref[n] >= 0)
            resolve(n);

    S.has_pairing = std::any_of(F.txt[c_pairing].begin(), F.txt[c_pairing].end(), [](const std::string &s) { return !s.empty(); });
    return S;
}

} // namespace

// ---- Frequencies and antennas ----

namespace
{
long qd_find_band(const std::vector<double> &F, double f)
{
    for (size_t i = 0; i < F.size(); ++i)
        if (same_freq(F[i], f))
            return (long)i;
    return -1;
}

// Final antenna lines of one node: empty = no antenna of its own. The lines have the antenna cells of the node, one
// per band (or one without a frequency), in the order of the frequency list.
std::vector<acells> qd_node_lines(const side_data &S, size_t n, const std::vector<double> &F, const char *side, bool *none)
{
    const std::vector<band_cells> &B = S.bands[n];
    const std::string where = S.fn + ", node '" + S.F.name[n] + "': ";
    *none = B[0].content && B[0].cells[c_antenna - c_center_freq] == "none";
    if (*none) // No antenna in all bands; band rows may declare frequencies, but not antennas
    {
        for (size_t b = 1; b < B.size(); ++b)
            if (B[b].content)
                throw std::runtime_error(S.fn + ", line " + std::to_string(S.first_line[n]) + ": the node '" + S.F.name[n] +
                                         "' has the antenna 'none', which applies to all bands, but a band row has antenna cells.");
        return {};
    }
    if (!B[0].has_freq)
    {
        if (B.size() > 1)
            throw std::runtime_error(S.fn + ", line " + std::to_string(S.first_line[n]) + ": the first row of the node '" + S.F.name[n] +
                                     "' needs a 'center_freq' because the node has band rows.");
        if (!B[0].content)
            return {};
        return {B[0].cells};
    }

    // Explicit bands: the node must define all bands of the list, each once
    std::vector<long> idx(B.size());
    std::vector<bool> seen(F.size(), false);
    bool others = false;
    for (size_t b = 0; b < B.size(); ++b)
    {
        idx[b] = qd_find_band(F, B[b].freq);
        if (idx[b] < 0)
            throw std::runtime_error(where + "the band " + qd_csv::format_number(B[b].freq) + " Hz is not in the frequency list of the layout (" + side + " file).");
        if (seen[(size_t)idx[b]])
            throw std::runtime_error(where + "the band " + qd_csv::format_number(B[b].freq) + " Hz is defined twice.");
        seen[(size_t)idx[b]] = true;
        others = others || (b > 0 && B[b].content);
    }
    if (B.size() != F.size())
        throw std::runtime_error(where + "defines " + std::to_string(B.size()) + " of " + std::to_string(F.size()) +
                                 " bands; a node with bands must define all of them.");
    if (!B[0].content)
    {
        if (others)
            throw std::runtime_error(where + "a band row has antenna cells, but the first band has none.");
        return {};
    }
    if (!others) // The same antenna in all bands: one line without frequency
    {
        acells a = B[0].cells;
        a[0].clear();
        return {a};
    }
    std::vector<acells> lines(B.size());
    for (size_t b = 0; b < B.size(); ++b)
    {
        acells a = B[b].content ? B[b].cells : B[0].cells;
        a[0] = B[b].cells[0];
        lines[(size_t)idx[b]] = a;
    }
    return lines;
}

std::string qd_join_line(const acells &a, const std::vector<int> &cols)
{
    std::string line;
    for (size_t k = 0; k < cols.size(); ++k)
        line += (k == 0 ? "" : ",") + qd_csv::format_cell(a[(size_t)(cols[k] - c_center_freq)]);
    return line;
}
} // namespace

quadriga_lib::layout quadriga_lib::layout_csv_read(const std::string &fn_tx, const std::string &fn_rx, const arma::vec &center_freq,
                                                   bool eval_antennas, std::vector<std::string> *extra_tx, std::vector<std::string> *extra_rx)
{
    if (!center_freq.is_empty())
    {
        if (!center_freq.is_finite() || arma::any(center_freq <= 0.0))
            throw std::invalid_argument("Input 'center_freq' must be finite and positive.");
        for (arma::uword i = 1; i < center_freq.n_elem; ++i)
            if (!(center_freq[i] > center_freq[i - 1]))
                throw std::invalid_argument("Input 'center_freq' must be strictly ascending.");
    }

    std::vector<std::string> names_tx, names_rx;
    side_data TX, RX;
    if (!fn_tx.empty())
        TX = qd_read_side(fn_tx, true, &names_tx);
    if (!fn_rx.empty())
        RX = qd_read_side(fn_rx, false, &names_rx);

    // Frequency list: given, else the bands of the Tx side, else those of the Rx side
    std::vector<double> F;
    if (!center_freq.is_empty())
        F.assign(center_freq.begin(), center_freq.end());
    else
        for (const side_data *S : {&TX, &RX})
        {
            for (const auto &bands : S->bands)
                for (const band_cells &b : bands)
                    if (b.has_freq && qd_find_band(F, b.freq) < 0)
                        F.push_back(b.freq);
            if (!F.empty())
                break;
        }
    std::sort(F.begin(), F.end());

    // Antennas of the nodes: Tx nodes first, then Rx nodes
    std::vector<std::vector<acells>> lines[2];
    std::vector<char> none[2];
    bool used[n_col] = {};
    for (int s = 0; s < 2; ++s)
    {
        const side_data &S = s == 0 ? TX : RX;
        for (size_t n = 0; n < S.F.n_node; ++n)
        {
            bool is_none = false;
            lines[s].push_back(qd_node_lines(S, n, F, s == 0 ? "Tx" : "Rx", &is_none));
            none[s].push_back(is_none);
            for (const acells &a : lines[s].back())
                for (int k = 0; k < n_antenna_col; ++k)
                    used[c_center_freq + k] = used[c_center_freq + k] || !a[(size_t)k].empty();
        }
    }
    std::vector<int> cols;
    for (int id = c_center_freq; id < n_col; ++id)
        if (used[id])
            cols.push_back(id);

    quadriga_lib::layout L;
    for (int id : cols)
        L.antenna_cols.push_back(column_of(id).name);
    std::map<std::vector<std::string>, arma::u32> registry;
    for (int s = 0; s < 2; ++s)
    {
        const side_data &S = s == 0 ? TX : RX;
        quadriga_lib::layout_nodes &N = s == 0 ? L.tx : L.rx;
        N = unflatten(S.F);
        for (size_t n = 0; n < S.F.n_node; ++n)
        {
            if (lines[s][n].empty())
                continue;
            std::vector<std::string> text;
            for (const acells &a : lines[s][n])
                text.push_back(qd_join_line(a, cols));
            const auto ins = registry.emplace(text, (arma::u32)(registry.size() + 1));
            if (ins.second)
                L.antenna_args.push_back(text);
            N.i_antenna[n] = ins.first->second;
        }
        for (size_t n = 0; n < S.F.n_node; ++n) // An extra sector without antenna cells has the antenna of sector 0
            if (lines[s][n].empty() && !none[s][n] && S.ref[n] >= 0)
                N.i_antenna[n] = N.i_antenna[(size_t)S.ref[n]];
    }

    L.center_freq = arma::vec(F);
    L.wrap = TX.wrap;
    L.seed = (arma::sword)TX.layout_seed;

    // Pairing: the terminals that list sites are linked to these sites, the others to all sites at a distance
    if (RX.has_pairing)
    {
        std::unordered_map<std::string, size_t> site;
        std::vector<size_t> first_node_of_site;
        for (size_t n = 0; n < L.tx.n_node(); ++n)
            if (site.emplace(L.tx.name[n], site.size()).second)
                first_node_of_site.push_back(n);
        size_t n_terminal = 0;
        const std::vector<size_t> terminal = site_of_node(L.rx, &n_terminal);
        std::vector<size_t> first_node_of_terminal(n_terminal, (size_t)-1);
        for (size_t n = 0; n < terminal.size(); ++n)
            if (first_node_of_terminal[terminal[n]] == (size_t)-1)
                first_node_of_terminal[terminal[n]] = n;
        std::vector<arma::uword> pairs;
        for (size_t n = 0; n < terminal.size(); ++n)
            if (RX.F.txt[c_pairing][n] != RX.F.txt[c_pairing][first_node_of_terminal[terminal[n]]])
                throw std::runtime_error(fn_rx + ", line " + std::to_string(RX.first_line[n]) + ": the 'pairing' of the node '" + L.rx.name[n] +
                                         "' differs from that of the other nodes of the terminal.");
        for (size_t t = 0; t < n_terminal; ++t)
        {
            const size_t node = first_node_of_terminal[t], n_before = pairs.size();
            const std::string &text = RX.F.txt[c_pairing][node];
            if (text.empty())
            {
                for (size_t s = 0; s < first_node_of_site.size(); ++s)
                {
                    const arma::uword a = L.tx.offset[first_node_of_site[s]], b = L.rx.offset[node];
                    if (arma::norm(L.tx.pos.col(a) - L.rx.pos.col(b)) > 1.0e-4)
                        pairs.push_back(s), pairs.push_back(t);
                }
                if (pairs.size() == n_before)
                    throw std::runtime_error(fn_rx + ", line " + std::to_string(RX.first_line[node]) + ": the terminal '" + L.rx.name[node] +
                                             "' has no pair: its empty 'pairing' cell means all sites, but no site is farther than 1e-4 m.");
                continue;
            }
            std::set<size_t> listed;
            size_t pos = 0;
            for (;;)
            {
                const size_t end = std::min(text.find(',', pos), text.size());
                const std::string_view entry = qd_csv::trim(std::string_view(text).substr(pos, end - pos));
                const auto it = site.find(std::string(entry));
                if (it == site.end())
                    throw std::runtime_error(fn_rx + ": the 'pairing' of terminal '" + L.rx.name[node] + "' names the site '" + std::string(entry) +
                                             "', which is not in the Tx file.");
                if (!listed.insert(it->second).second)
                    throw std::runtime_error(fn_rx + ": the 'pairing' of terminal '" + L.rx.name[node] + "' names the site '" + std::string(entry) + "' twice.");
                pairs.push_back(it->second), pairs.push_back(t);
                if (end >= text.size())
                    break;
                pos = end + 1;
            }
        }
        L.pairing = arma::umat(pairs.data(), 2, pairs.size() / 2);
    }

    const std::string files = "'" + (fn_tx.empty() ? fn_rx : fn_tx) + (fn_tx.empty() || fn_rx.empty() ? "" : "' and '" + fn_rx) + "'";
    const std::string msg = L.is_valid();
    if (!msg.empty())
        throw std::runtime_error("The layout in " + files + " is not valid: " + msg);
    if (eval_antennas) // The messages name the file and the first line of the node
    {
        std::vector<std::string> where[2];
        for (int s = 0; s < 2; ++s)
        {
            const side_data &S = s == 0 ? TX : RX;
            for (size_t n = 0; n < S.F.n_node; ++n)
                where[s].push_back(S.fn + ", line " + std::to_string(S.first_line[n]) + " (" + (s == 0 ? "Tx" : "Rx") + " node '" + S.F.name[n] +
                                   "', sector " + std::to_string((unsigned)S.F.v[c_sector][n]) + ")");
        }
        try
        {
            qd_layout::eval_antennas(L, {}, &where[0], &where[1]);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error(e.what());
        }
    }

    if (extra_tx != nullptr)
        *extra_tx = names_tx;
    if (extra_rx != nullptr)
        *extra_rx = names_rx;
    return L;
}

// ---- Writer ----

namespace
{
// Cell text of a number in the units of the file
std::string qd_fmt(double v, int id)
{
    const double scale = column_of(id).scale;
    return scale == 1.0 ? qd_csv::format_number(v) : qd_csv::format_number(v / scale, 15, 15);
}

// An absolute path relative to the folder of the CSV file 'fn' (by the text of the paths); other paths are kept
std::string qd_relative_path(const std::string &path, const std::string &fn)
{
    const fs::path p(path);
    if (path.empty() || !p.is_absolute())
        return path;
    const fs::path rel = p.lexically_relative(fs::absolute(fs::path(fn)).parent_path().lexically_normal());
    return rel.empty() ? path : rel.generic_string();
}

template <class T>
bool qd_has_value(const std::vector<T> &v, T dflt)
{
    for (const T &x : v)
        if (!equal_value(x, dflt))
            return true;
    return false;
}

// Text of one side of a layout. An extra sector (sector > 0 of a site with a sector 0) gets all its cells in the first
// row, because an empty cell would be the value of sector 0; 'compact' leaves out what the reader repeats from the
// previous row of the node and the defaults of the other nodes
std::string qd_compose_side(const quadriga_lib::layout &L, bool is_tx, const std::string &fn, bool compact, const std::vector<std::string> &acols,
                            const std::vector<std::vector<std::string>> &aargs)
{
    const quadriga_lib::layout_nodes &N = is_tx ? L.tx : L.rx;
    const flat F = flatten(N);
    const size_t n_node = F.n_node;
    size_t n_site = 0;
    const std::vector<size_t> site = site_of_node(N, &n_site);
    std::vector<size_t> first_of_site(n_site, (size_t)-1);
    std::vector<long> ref(n_node, -1), sector0(n_site, -1);
    for (size_t n = 0; n < n_node; ++n)
    {
        if (first_of_site[site[n]] == (size_t)-1)
            first_of_site[site[n]] = n;
        if (N.sector[n] == 0)
            sector0[site[n]] = (long)n;
    }
    for (size_t n = 0; n < n_node; ++n)
        if (N.sector[n] != 0)
            ref[n] = sector0[site[n]];
    auto n_row = [&](size_t n) { return (size_t)N.n_row((arma::uword)n); };

    // Antenna lines of each node: cells per line
    std::vector<std::vector<std::vector<std::string>>> lines(n_node);
    std::vector<int> ant_cols_all;
    for (const std::string &c : acols)
        ant_cols_all.push_back(find_column(c));
    for (size_t n = 0; n < n_node; ++n)
    {
        if (N.i_antenna[n] == 0)
            continue;
        if (aargs.empty() || aargs[N.i_antenna[n] - 1].empty())
            throw std::invalid_argument("Antenna " + std::to_string(N.i_antenna[n]) + " of node '" + N.name[n] + "' has no CSV description.");
        for (const std::string &line : aargs[N.i_antenna[n] - 1])
            lines[n].push_back(split_antenna_line(line, acols.size()));
    }
    // An extra sector without antenna next to a sector 0 with one writes 'none': an empty cell would be the antenna of sector 0
    std::vector<char> write_none(n_node, 0);
    for (size_t n = 0; n < n_node; ++n)
        write_none[n] = ref[n] >= 0 && N.i_antenna[n] == 0 && N.i_antenna[(size_t)ref[n]] != 0;

    // The frequency list is declared on the first node of the first non-empty side if no antenna line of a node carries it
    bool carried = false;
    for (size_t k = 0; k < ant_cols_all.size(); ++k)
        if (ant_cols_all[k] == c_center_freq)
            for (const quadriga_lib::layout_nodes *S : {&L.tx, &L.rx})
                for (arma::uword i = 0; i < S->i_antenna.n_elem; ++i)
                    if (S->i_antenna[i] != 0)
                        for (const std::string &line : aargs[S->i_antenna[i] - 1])
                            carried = carried || !split_antenna_line(line, acols.size())[k].empty();
    const bool declare = !carried && !L.center_freq.is_empty() && n_node != 0 && (is_tx || L.tx.n_node() == 0);

    // Columns
    std::vector<int> cols = {c_name};
    std::vector<std::string> scenpar_keys;
    auto add_if = [&](int id, bool cond) { if (cond) cols.push_back(id); };
    add_if(c_sector, qd_has_value(F.v[c_sector], 0.0));
    for (int id : {c_x, c_y, c_z})
        cols.push_back(id);
    for (int id = c_time; id <= c_d2d_in; ++id)
        if (id == c_scenario)
            add_if(id, qd_has_value(F.txt[id], std::string()));
        else
            add_if(id, qd_has_value(F.v[id], column_of(id).dflt));
    for (int id : {c_d_clutter, c_r_clutter, c_h_clutter})
        add_if(id, qd_has_value(F.v[id], column_of(id).dflt));
    for (const auto &kv : F.scenpar)
        if (qd_has_value(kv.second, std::numeric_limits<double>::quiet_NaN()))
            scenpar_keys.push_back(kv.first);
    add_if(c_scenpar, !scenpar_keys.empty());
    add_if(c_wrap, is_tx && !L.wrap.is_empty());
    add_if(c_seed, qd_has_value(F.seed, -1LL));
    add_if(c_layout_seed, is_tx && L.seed != -1);
    add_if(c_cell, is_tx && qd_has_value(F.v[c_cell], 1.0));
    add_if(c_pairing, !is_tx && !L.pairing.is_empty());
    add_if(c_relative, qd_has_value(F.v[c_relative], 0.0));
    for (int id = c_center_freq; id < n_col; ++id)
    {
        bool any = (id == c_center_freq && declare) || (id == c_antenna && std::count(write_none.begin(), write_none.end(), 1) != 0);
        for (size_t k = 0; k < ant_cols_all.size() && !any; ++k)
            if (ant_cols_all[k] == id)
                for (size_t n = 0; n < n_node && !any; ++n)
                    for (const auto &line : lines[n])
                        any = any || !line[k].empty();
        add_if(id, any);
    }

    qd_csv::table T;
    T.source = fn;
    std::vector<int> col_of_header; // id of each header entry, -1 = scenpar key, -2 = extra
    std::vector<size_t> header_arg;
    for (int id : cols)
    {
        if (id == c_scenpar)
            for (size_t k = 0; k < scenpar_keys.size(); ++k)
                T.header.push_back("scenpar." + scenpar_keys[k]), col_of_header.push_back(-1), header_arg.push_back(k);
        else
            T.header.push_back(column_of(id).name), col_of_header.push_back(id), header_arg.push_back(0);
    }
    for (size_t e = 0; e < F.extra_names.size(); ++e)
        T.header.push_back(F.extra_names[e]), col_of_header.push_back(-2), header_arg.push_back(e);
    const size_t n_h = T.header.size();
    std::vector<size_t> h_of_id(n_col, (size_t)-1);
    for (size_t h = 0; h < n_h; ++h)
        if (col_of_header[h] >= 0)
            h_of_id[(size_t)col_of_header[h]] = h;
    // Position of the antenna columns of the lines in the header
    std::vector<size_t> line_h(ant_cols_all.size(), (size_t)-1);
    for (size_t k = 0; k < ant_cols_all.size(); ++k)
        if (ant_cols_all[k] >= 0)
            line_h[k] = h_of_id[(size_t)ant_cols_all[k]];

    // Pairing text per terminal
    std::vector<std::string> pairing_text;
    if (!is_tx && !L.pairing.is_empty())
    {
        std::vector<std::string> site_name;
        std::unordered_set<std::string> seen;
        for (size_t n = 0; n < L.tx.n_node(); ++n)
            if (seen.insert(L.tx.name[n]).second)
                site_name.push_back(L.tx.name[n]);
        pairing_text.assign(n_site, std::string());
        for (arma::uword k = 0; k < L.pairing.n_cols; ++k)
        {
            std::string &t = pairing_text[L.pairing(1, k)];
            t += (t.empty() ? "" : ",") + site_name[L.pairing(0, k)];
        }
        for (size_t t = 0; t < n_site; ++t)
            if (pairing_text[t].empty())
                throw std::invalid_argument("Terminal '" + N.name[first_of_site[t]] + "' has no entry in 'pairing', which cannot be written because an empty cell means all sites.");
    }

    std::vector<std::string> row(n_h);
    auto emit = [&] { T.add_row(row); std::fill(row.begin(), row.end(), std::string()); };
    auto set_antenna_cells = [&](const std::vector<std::string> &line)
    {
        for (size_t j = 0; j < line_h.size(); ++j)
            if (line_h[j] != (size_t)-1)
                row[line_h[j]] = line[j];
        if (h_of_id[c_qdant_file] != (size_t)-1)
            row[h_of_id[c_qdant_file]] = qd_relative_path(row[h_of_id[c_qdant_file]], fn);
    };

    for (size_t n = 0; n < n_node; ++n)
    {
        const size_t K = n_row(n), o = F.offset[n];
        const long rf = ref[n];
        const std::string sector_text = std::to_string(N.sector[n]);

        for (size_t k = 0; k < K; ++k)
        {
            const bool full = !compact || (rf >= 0 && k == 0); // Write the cell even if the reader would repeat it
            for (size_t h = 0; h < n_h; ++h)
            {
                const int id = col_of_header[h];
                std::string &cellt = row[h];
                if (id == c_name)
                    cellt = N.name[n];
                else if (id == c_sector)
                    cellt = sector_text;
                else if (id == -2)
                    cellt = F.extra[header_arg[h]][o + k];
                else if (id == c_x || id == c_y || id == c_z || id == c_time)
                    cellt = qd_fmt(F.v[id][o + k], id);
                else if (id == c_scenario)
                {
                    const std::string &cur = F.txt[id][o + k];
                    const std::string &prev = k > 0 ? F.txt[id][o + k - 1] : (rf >= 0 ? F.txt[id][F.offset[(size_t)rf]] : std::string());
                    if (cur.empty() && !prev.empty())
                        throw std::invalid_argument("Node '" + N.name[n] + "': the scenario cannot return to none after a scenario was set, an empty cell repeats the previous row.");
                    if (full || cur != (k > 0 ? prev : std::string()))
                        cellt = cur;
                }
                else if (id >= c_bank && id <= c_d2d_in)
                {
                    if (F.v[id].empty())
                        continue;
                    const double cur = F.v[id][o + k], prev = k > 0 ? F.v[id][o + k - 1] : column_of(id).dflt;
                    if (full || !same_value(cur, prev))
                        cellt = qd_fmt(cur, id);
                }
                else if (id == c_wrap || id == c_layout_seed)
                {
                    if (compact && !(n == 0 && k == 0))
                        continue;
                    cellt = id == c_wrap ? qd_csv::format_matrix(L.wrap.t()) : std::to_string((long long)L.seed);
                }
                else if (id == -1 || id == c_d_clutter || id == c_r_clutter || id == c_h_clutter || id == c_seed || id == c_cell || id == c_pairing ||
                         id == c_relative)
                {
                    // Fixed values: on the first row (on all rows if not compact); the default only for an extra sector
                    if (compact && k > 0)
                        continue;
                    if (id == c_pairing)
                        cellt = pairing_text[site[n]];
                    else if (id == c_seed)
                        cellt = (full || F.seed[n] != -1) ? std::to_string(F.seed[n]) : std::string();
                    else
                    {
                        const std::vector<double> &v = id == -1 ? F.scenpar.at(scenpar_keys[header_arg[h]]) : F.v[id];
                        const double dflt = id == -1 ? std::numeric_limits<double>::quiet_NaN() : column_of(id).dflt;
                        if (full || !same_value(v[n], dflt))
                            cellt = id == -1 ? qd_csv::format_number(v[n]) : qd_fmt(v[n], id);
                    }
                }
            }
            // Antenna cells of the first band
            if (!lines[n].empty() && (k == 0 || !compact))
                set_antenna_cells(lines[n][0]);
            if (write_none[n] && (k == 0 || !compact))
                row[h_of_id[c_antenna]] = "none";
            if (declare && n == 0 && (k == 0 || !compact) && h_of_id[c_center_freq] != (size_t)-1)
                row[h_of_id[c_center_freq]] = qd_csv::format_number(L.center_freq[0]);
            emit();
        }

        // Band rows: the further bands of the node
        auto band_row = [&]()
        {
            for (size_t h = 0; h < n_h; ++h)
                if (col_of_header[h] == c_name)
                    row[h] = N.name[n];
                else if (col_of_header[h] == c_sector)
                    row[h] = sector_text;
        };
        for (size_t b = 1; b < lines[n].size(); ++b)
        {
            band_row();
            set_antenna_cells(lines[n][b]);
            emit();
        }
        if (declare && n == 0)
            for (arma::uword b = 1; b < L.center_freq.n_elem; ++b)
            {
                band_row();
                row[h_of_id[c_center_freq]] = qd_csv::format_number(L.center_freq[b]);
                emit();
            }
    }
    std::ostringstream os;
    qd_csv::write(os, T);
    return os.str();
}
} // namespace

// ---- QDANT export ----

namespace
{
// Name of a CSV file without the folder and without ".csv"
std::string qd_csv_stem(const std::string &fn)
{
    std::string s = fs::path(fn).filename().string();
    if (s.size() > 4)
    {
        std::string ext = s.substr(s.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (ext == ".csv")
            s.resize(s.size() - 4);
    }
    return s;
}

// A path as a normalised absolute path with '/' separators
std::string qd_abs(const fs::path &p) { return fs::absolute(p).lexically_normal().generic_string(); }

// The antenna lines of the written files and the QDANT files that they need. Each band that is an evaluated array
// without text becomes a 'qdant' line; the bands of one antenna go to one file, named after the first written CSV file
// that uses the antenna: <stem>_<k>.qdant with k from 1 per CSV file, skipping names that a 'qdant' line of the layout uses
struct qd_export
{
    std::vector<std::string> cols;
    std::vector<std::vector<std::string>> args;
    std::vector<std::string> files;                                  // Absolute paths
    std::vector<std::vector<quadriga_lib::arrayant<double>>> arrays; // Entries of each file
    std::vector<std::string> referenced;                             // QDANT files of the lines of the written nodes
};

qd_export qd_plan_export(const quadriga_lib::layout &L, const std::string fn[2])
{
    qd_export E;
    const size_t n = (size_t)L.n_antenna(), nf = L.center_freq.n_elem, c_type = c_antenna - c_center_freq,
                 c_file = c_qdant_file - c_center_freq, c_id = c_qdant_id - c_center_freq;
    E.cols = L.antenna_cols;
    E.args = L.antenna_args;
    E.args.resize(n);

    std::set<std::string> used; // QDANT files of the lines of the layout
    for (const auto &lines : E.args)
        for (const std::string &line : lines)
            if (!line.empty())
            {
                const antenna_cells a = parse_antenna_line(line, E.cols);
                if (antenna_type_mask(a[c_type]) == t_qdant && !a[c_file].empty())
                    used.insert(qd_abs(a[c_file]));
            }

    // Antennas in order of first use by a written file, and the side of that file
    std::vector<int> owner(n, -1);
    std::vector<size_t> order;
    std::vector<std::string> who(n); // The first node that uses the antenna, for the messages
    for (int s = 0; s < 2; ++s)
        if (!fn[s].empty())
        {
            const quadriga_lib::layout_nodes &N = s == 0 ? L.tx : L.rx;
            for (arma::uword i = 0; i < N.i_antenna.n_elem; ++i)
                if (N.i_antenna[i] != 0 && owner[N.i_antenna[i] - 1] < 0)
                {
                    const size_t a = N.i_antenna[i] - 1;
                    owner[a] = s, order.push_back(a);
                    who[a] = std::string(s == 0 ? "Tx" : "Rx") + " node '" + N.name[i] + "', sector " + std::to_string(N.sector[i]) + ": ";
                }
        }

    struct new_line
    {
        size_t a, b;
        bool all; // One line for all bands
        antenna_cells cells;
    };
    std::vector<new_line> pending;
    std::map<std::string, unsigned> next_k; // By folder and stem
    for (size_t a : order)
    {
        const std::vector<std::string> &lines = E.args[a];
        const size_t n_arr = L.antenna.empty() ? 0 : L.antenna[a].size();
        const bool all = lines.empty() || (lines.size() == 1 && lines[0].empty());
        std::vector<size_t> bands;
        if (all && (n_arr != nf || nf < 2))
            bands = {0};
        else if (all || lines.size() > 1)
            for (size_t b = 0; b < nf; ++b)
                if (all || lines[b].empty())
                    bands.push_back(b);
        if (bands.empty())
            continue;
        const bool one = all && bands.size() == 1, one_band = one || nf == 0;
        std::vector<quadriga_lib::arrayant<double>> vec;
        for (size_t b : bands)
        {
            if (b >= n_arr || L.antenna[a][b].n_elements() == 0)
                throw std::invalid_argument(who[a] + "the antenna is an evaluated array, but 'antenna' does not hold it" +
                                            (L.antenna.empty() ? " (the layout is not evaluated)" : "") + ", so there is nothing to write.");
            const std::string msg = L.antenna[a][b].is_valid();
            if (!msg.empty())
                throw std::invalid_argument(who[a] + (one_band ? "" : "the band " + qd_csv::format_number(L.center_freq[b]) + " Hz: ") +
                                            "the evaluated array is not valid: " + msg);
            vec.push_back(L.antenna[a][b]);
        }
        const std::string &csv = fn[owner[a]];
        const fs::path folder = fs::absolute(fs::path(csv)).parent_path();
        const std::string stem = qd_csv_stem(csv), key = qd_abs(folder / stem);
        unsigned &k = next_k.emplace(key, 1).first->second;
        std::string path;
        for (;; ++k)
            if (used.count(path = qd_abs(folder / (stem + "_" + std::to_string(k) + ".qdant"))) == 0)
                break;
        ++k;
        E.files.push_back(path);
        E.arrays.push_back(std::move(vec));
        for (size_t j = 0; j < bands.size(); ++j)
        {
            new_line p{a, bands[j], one, antenna_cells()};
            p.cells[c_type] = "qdant", p.cells[c_file] = path, p.cells[c_id] = std::to_string(j + 1);
            if (!one)
                p.cells[0] = qd_csv::format_number(L.center_freq[bands[j]]);
            pending.push_back(p);
        }
    }

    // The columns of the new lines are added; then the lines are replaced
    std::vector<antenna_cells> extra;
    for (const new_line &p : pending)
        extra.push_back(p.cells);
    const std::vector<std::string> cols = used_antenna_cols(E.args, E.cols, extra);
    rewrite_antenna_lines(E.args, E.cols, cols);
    E.cols = cols;
    for (const new_line &p : pending)
    {
        std::vector<std::string> &lines = E.args[p.a];
        if (p.all)
            lines = {join_antenna_line(p.cells, E.cols)};
        else
        {
            if (lines.size() != nf)
                lines.assign(nf, std::string());
            lines[p.b] = join_antenna_line(p.cells, E.cols);
        }
    }

    // The QDANT files that the written CSV files reference
    for (size_t a : order)
        for (const std::string &line : E.args[a])
            if (!line.empty())
            {
                const antenna_cells c = parse_antenna_line(line, E.cols);
                if (!c[c_file].empty())
                    E.referenced.push_back(qd_abs(c[c_file]));
            }
    return E;
}

// Deletes the files <stem>_<k>.qdant next to the CSV file that none of 'referenced' is
void qd_delete_unreferenced(const std::string &csv, const std::vector<std::string> &referenced)
{
    std::error_code ec;
    const fs::path folder = fs::absolute(fs::path(csv)).parent_path();
    const std::string prefix = qd_csv_stem(csv) + "_", suffix = ".qdant";
    std::vector<fs::path> remove;
    for (fs::directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec))
    {
        const std::string name = it->path().filename().string();
        if (name.size() <= prefix.size() + suffix.size() || name.compare(0, prefix.size(), prefix) != 0 ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string k = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        if (k[0] == '0' || k.find_first_not_of("0123456789") != std::string::npos || !it->is_regular_file(ec))
            continue;
        bool keep = false;
        for (const std::string &r : referenced)
        {
            std::error_code e;
            keep = keep || qd_abs(it->path()) == r || fs::equivalent(it->path(), r, e);
        }
        if (!keep)
            remove.push_back(it->path());
    }
    for (const fs::path &p : remove)
        fs::remove(p, ec);
}
} // namespace

void quadriga_lib::layout_csv_write(const std::string &fn_tx, const std::string &fn_rx, const quadriga_lib::layout &layout, bool compact)
{
    std::error_code ec;
    if (!fn_tx.empty() && !fn_rx.empty() && fs::weakly_canonical(fn_tx, ec) == fs::weakly_canonical(fn_rx, ec))
        throw std::invalid_argument("Inputs 'fn_tx' and 'fn_rx' must be different files.");
    const std::string msg = layout.is_valid();
    if (!msg.empty())
        throw std::invalid_argument("Input 'layout' is not valid: " + msg);
    if (!fn_tx.empty() && layout.tx.n_node() == 0 && (layout.seed != -1 || !layout.wrap.is_empty()))
        throw std::invalid_argument("The 'seed' and 'wrap' of the layout can only be written to the Tx file, which has no nodes.");
    if (!fn_tx.empty() && layout.tx.n_node() == 0 && layout.rx.n_node() == 0 && !layout.center_freq.is_empty())
        throw std::invalid_argument("The frequencies of the layout need a node in the Tx or Rx file to be written.");

    // Compose all files first, write them next to their targets, then replace the targets: QDANT files first
    const std::string fn[2] = {fn_tx, fn_rx};
    const qd_export E = qd_plan_export(layout, fn);
    std::string text[2];
    for (int s = 0; s < 2; ++s)
        if (!fn[s].empty())
            text[s] = qd_compose_side(layout, s == 0, fn[s], compact, E.cols, E.args);

    std::string tmp[2];
    std::vector<std::string> tmp_q(E.files.size());
    auto cleanup = [&]
    {
        for (const std::string &t : tmp)
            if (!t.empty())
            {
                std::error_code e;
                fs::remove(t, e);
            }
        for (const std::string &t : tmp_q)
            if (!t.empty())
            {
                std::error_code e;
                fs::remove(t, e);
            }
    };
    try
    {
        for (size_t f = 0; f < E.files.size(); ++f)
        {
            tmp_q[f] = E.files[f] + ".part";
            std::error_code ec_stale;
            fs::remove(tmp_q[f], ec_stale); // A temporary file of an earlier failed call
            quadriga_lib::qdant_write(tmp_q[f], E.arrays[f]); // The arrays are checked by the plan
        }
        for (int s = 0; s < 2; ++s)
        {
            if (fn[s].empty())
                continue;
            tmp[s] = fn[s] + ".part";
            std::ofstream f(tmp[s], std::ios::binary | std::ios::trunc);
            if (!f.is_open())
                throw std::runtime_error("Cannot open file '" + tmp[s] + "' for writing.");
            f.write(text[s].data(), (std::streamsize)text[s].size());
            f.flush();
            if (!f)
                throw std::runtime_error("Cannot write file '" + tmp[s] + "'.");
        }
        for (size_t f = 0; f < E.files.size(); ++f)
        {
            fs::rename(tmp_q[f], E.files[f]);
            tmp_q[f].clear();
        }
        for (int s = 0; s < 2; ++s)
            if (!fn[s].empty())
            {
                fs::rename(tmp[s], fn[s]);
                tmp[s].clear();
            }
    }
    catch (const fs::filesystem_error &e)
    {
        cleanup();
        throw std::runtime_error(std::string("Cannot replace the file: ") + e.what());
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    // QDANT files of the stems of the written CSV files that the new files do not reference
    for (int s = 0; s < 2; ++s)
        if (!fn[s].empty())
            qd_delete_unreferenced(fn[s], E.referenced);
}
