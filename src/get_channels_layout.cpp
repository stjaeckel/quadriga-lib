// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "qd_helpers_shared.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# layout_links
State of the propagation links of one realisation of a layout

- Holds the discrete state of the links (line of sight, indoor state, scenario and random seeds), no large-scale parameters, phases or rays; it is not stored in the CSV files
- A propagation link is one site, one terminal and one segment of a track with one resolved scenario; its arrays have one entry per link

## Attributes:<br>
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `arma::u32_vec i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of each link, 0-based |
| `arma::u32_vec image` | `[n_link]` | Wrap-around image of the site, 0 = the real site |
| `arma::u32_vec i_conf` | `[n_link]` | Index into `conf_names`, derived from scenario, LOS and indoor state |
| `std::vector<std::string> conf_names` | `[n_conf]` | Resolved conf names |
| `arma::u32_vec los` | `[n_link]` | 1 = line of sight |
| `arma::s32_vec indoor` | `[n_link]` | Indoor state at the start of the segment, as in [[layout_nodes]] |
| `arma::u32_vec floor_id` | `[n_link]` | Floor number - 1, 0 outdoors; of the drawn state, stages 2 and 3 derive it from `indoor` |
| `arma::vec d2d_in` | `[n_link]` | Indoor 2D distance in [m]; of the drawn state, stages 2 and 3 derive it from `indoor` |
| `arma::mat o2i_rnd` | `[n_terminal, 4]` | Random inputs of [[o2i_loss]] |
| `arma::u32_vec o2i_model` | `[n_terminal]` | Building O2I model, index into the list of the family; for every terminal of a family with O2I, also outdoors |
| `arma::s64_vec gbsm_seed` | `[n_link]` | Seed of the random phases; equal for one site, image and conf |
| `arma::sword seed` | scalar | Master seed, the base of the streams of all stages; -1 = not set |
| `std::string method`, `sos_preset` | | Method and SOS preset as used; an empty `method` means not filled |
| `double anchor_freq` | scalar | Anchor frequency of the multi-frequency model in [Hz], default 2e9 |
| `bool pol_variability` | scalar | Polarization power variability, default false |
| `std::vector<std::string> conf_files` | `[n_conf]` | Resolved paths of the conf files; empty = not resolved |
| `unsigned long long layout_hash` | scalar | Hash of all layout members that determine the links |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `.n_link()` | Number of links |
| `.empty()` | `true` if there are no links |
| `.reset()` | Empties all members and restores the defaults |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, the indices into `conf_names` and the ranges |

## See also:
- [[layout]] (holds the links as `links`)
- [[o2i_loss]] (uses `o2i_rnd`)
MD!*/

arma::uword quadriga_lib::layout_links::n_link() const { return i_tx.n_elem; }

bool quadriga_lib::layout_links::empty() const { return i_tx.n_elem == 0; }

void quadriga_lib::layout_links::reset() { *this = layout_links(); }

std::string quadriga_lib::layout_links::is_valid() const
{
    const arma::uword n = i_tx.n_elem;
    if (i_rx.n_elem != n || i_segment.n_elem != n || image.n_elem != n || i_conf.n_elem != n || los.n_elem != n ||
        indoor.n_elem != n || floor_id.n_elem != n || d2d_in.n_elem != n || gbsm_seed.n_elem != n)
        return "The members 'i_tx', 'i_rx', 'i_segment', 'image', 'i_conf', 'los', 'indoor', 'floor_id', 'd2d_in' and 'gbsm_seed' must have one entry per link.";
    for (arma::uword i = 0; i < n; ++i)
    {
        if (i_conf[i] >= conf_names.size())
            return "Member 'i_conf' has an index outside 'conf_names'.";
        if (los[i] > 1)
            return "Member 'los' must be 0 or 1.";
        if (indoor[i] < -1 || indoor[i] > 3)
            return "Member 'indoor' must be -1, 0, 1, 2 or 3.";
        if (std::isinf(d2d_in[i]) || d2d_in[i] < 0.0)
            return "Member 'd2d_in' must be NaN or >= 0.";
    }
    if (!conf_files.empty() && conf_files.size() != conf_names.size())
        return "Member 'conf_files' must be empty or have one entry per conf name.";
    if ((!o2i_rnd.is_empty() && o2i_rnd.n_cols != 4) || o2i_rnd.n_rows != o2i_model.n_elem)
        return "Member 'o2i_rnd' must have 4 columns and one row per entry of 'o2i_model'.";
    if (seed < -1)
        return "Member 'seed' must be >= -1.";
    if (!(anchor_freq > 0.0) || !std::isfinite(anchor_freq))
        return "Member 'anchor_freq' must be positive and finite.";
    return "";
}

// ---- Pipeline helpers ----

namespace
{
using qd_spmap = std::unordered_map<std::string, double>;
using qd_plmap = std::unordered_map<std::string, std::vector<double>>;
using qd_family = qd_layout::family;

const std::string qd_v19 = "3GPP_38.901_v19_"; // Prefix of the shipped TR 38.901 conf files

// Number as short text for messages
std::string qd_num(double x)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", x);
    return buf;
}

// Runs f and puts 'who' in front of the message of an error, keeping std::invalid_argument
template <class F>
void qd_named(const std::string &who, F &&f)
{
    try
    {
        f();
    }
    catch (const std::invalid_argument &e)
    {
        throw std::invalid_argument(who + e.what());
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(who + e.what());
    }
}

// The sites of the Tx side and the terminals of the Rx side: index of each node, the nodes of each
struct qd_sides
{
    std::vector<size_t> site_of_tx, term_of_rx;
    size_t n_site = 0, n_term = 0;
    std::vector<std::vector<arma::uword>> site_nodes, term_nodes;
};

qd_sides qd_make_sides(const quadriga_lib::layout &L)
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

// Rx row at the start of a segment of terminal t; a track has one segment so far
arma::uword qd_segment_row(const quadriga_lib::layout &L, const qd_sides &S, size_t t, arma::uword i_segment)
{
    (void)i_segment;
    return L.rx.offset[S.term_nodes[t][0]];
}

// The wrap images: 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2); one vector: 0, +a1, -a1; [2, n_img]
arma::mat qd_images(const arma::mat &wrap)
{
    arma::mat I(2, 1, arma::fill::zeros);
    if (wrap.n_cols >= 1)
        I = arma::join_rows(I, wrap.col(0), -wrap.col(0));
    if (wrap.n_cols == 2)
        I = arma::join_rows(I, arma::join_rows(wrap.col(1), -wrap.col(1), wrap.col(0) - wrap.col(1), wrap.col(1) - wrap.col(0)));
    return I;
}

// Cheap checks of the sizes of all members of one side that the pipeline indexes, O(n_node + number of members)
bool qd_sizes_ok(const quadriga_lib::layout_nodes &N)
{
    const arma::uword nn = (arma::uword)N.name.size(), nr = N.pos.n_cols;
    if (N.sector.n_elem != nn || N.i_antenna.n_elem != nn || N.offset.n_elem != nn || N.pos.n_rows != 3 || N.orientation.n_rows != 3 ||
        N.orientation.n_cols != nr || (nn == 0 && nr != 0))
        return false;
    for (arma::uword i = 0; i < nn; ++i)
        if ((i == 0 && N.offset[0] != 0) || (i > 0 && N.offset[i] <= N.offset[i - 1]) || N.offset[i] >= nr)
            return false;
    auto per_row = [nr](arma::uword n) { return n == 0 || n == nr; };
    auto per_node = [nn](arma::uword n) { return n == 0 || n == nn; };
    if (!per_row(N.speed.n_elem) || !per_row(N.time.n_elem) || !per_row(N.indoor.n_elem) || !per_row(N.floor.n_elem) ||
        !per_row(N.d2d_in.n_elem) || !per_row((arma::uword)N.scenario.size()) || !per_node(N.seed.n_elem) || !per_node(N.relative.n_elem))
        return false;
    if ((!N.direction.is_empty() && (N.direction.n_rows != 2 || N.direction.n_cols != nr)) ||
        (!N.clutter.is_empty() && (N.clutter.n_rows != 3 || N.clutter.n_cols != nn)))
        return false;
    for (const auto &kv : N.scenpar_override)
        if (kv.second.n_elem != nn)
            return false;
    return true;
}

// Stages 2 and 3 do not repeat layout.is_valid (about 0.4 us per row, too much for calls on a few links of a large layout);
// the hash covers the validated members that determine the links, and these checks the sizes of the members that the
// pipeline indexes, including the antennas. A failed check reports the message of is_valid
void qd_check_sizes(const quadriga_lib::layout &L)
{
    const arma::uword n_ant = L.n_antenna(), n_freq = L.center_freq.n_elem;
    bool ok = qd_sizes_ok(L.tx) && qd_sizes_ok(L.rx) && (L.antenna.empty() || L.antenna.size() == n_ant);
    for (const quadriga_lib::layout_nodes *N : {&L.tx, &L.rx})
        ok = ok && (N->i_antenna.is_empty() || N->i_antenna.max() <= n_ant);
    for (const auto &v : L.antenna)
        ok = ok && (v.empty() || v.size() == 1 || v.size() == n_freq);
    for (const auto &v : L.antenna_args)
        ok = ok && (v.empty() || v.size() == 1 || v.size() == n_freq);
    if (!ok)
    {
        const std::string msg = L.is_valid();
        throw std::invalid_argument("Input 'layout' is not valid: " + (msg.empty() ? std::string("the sizes of its members do not match.") : msg));
    }
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
qd_scen qd_resolve(const std::string &name, const qd_family *prefer)
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
std::string qd_conf_of(const qd_scen &s, unsigned los, int indoor)
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

// Moving site: more than one row or a speed (dual mobility of the GBSM functions); all nodes of a site have equal motion
bool qd_site_moves(const quadriga_lib::layout_nodes &N, arma::uword i_node)
{
    const arma::uword first = N.offset[i_node], n = N.n_row(i_node);
    if (n > 1)
        return true;
    return !N.speed.is_empty() && N.speed[first] != 0.0;
}

// The base seed of the streams of a site: its seed, else the master seed
arma::sword qd_site_base(const quadriga_lib::layout_nodes &tx, arma::uword i_node, arma::sword master)
{
    return (!tx.seed.is_empty() && tx.seed[i_node] != -1) ? (arma::sword)tx.seed[i_node] : master;
}

// The O2I family of each terminal: the family of the first of its links (in site order) whose scenario is a family
// with O2I; nullptr = none. All links of a terminal share its O2I state (TR 38.901 clause 7.6.3.3)
std::vector<const qd_family *> qd_term_families(const std::vector<qd_scen> &scen, const arma::u32_vec &i_rx, size_t n_term)
{
    std::vector<const qd_family *> F(n_term, nullptr);
    for (arma::uword l = 0; l < i_rx.n_elem; ++l)
        if (F[i_rx[l]] == nullptr && scen[l].has_o2i())
            F[i_rx[l]] = scen[l].fam;
    return F;
}

// Weights of the building models of a family, with the share of the low-loss model if given (families with "low"
// and "high"; the others have one building model)
arma::vec qd_o2i_weights(const qd_family &f, const arma::vec &o2i_low)
{
    arma::vec w(f.o2i_weights);
    if (!o2i_low.is_empty() && f.o2i_models.size() == 2 && f.o2i_models[0] == "low" && f.o2i_models[1] == "high")
        w = {o2i_low[0], 1.0 - o2i_low[0]};
    return w;
}

// Maximum indoor distance of a building state of a family (SMa: residential 3, commercial 1)
double qd_d2d_in_max(const qd_family &f, int indoor) { return indoor == 1 ? f.d2d_in_max_commercial : f.d2d_in_max; }

// Floor of a building UT at Rx row r: the floor column, else from the height
int qd_floor(const quadriga_lib::layout_nodes &rx, arma::uword r)
{
    const int fl = rx.floor.is_empty() ? -1 : rx.floor[r];
    return fl >= 1 ? fl : qd_layout::floor_from_z(rx.pos(2, r));
}

// Floor index and indoor 2D distance of a link in the indoor state 'io' of a terminal of family f (nullptr = no O2I):
// building UTs on their floor at the distance of the columns c1, c2 of the O2I inputs, as o2i_loss gives it; others on
// floor 0 at 0 m. Used for the drawn states in stage 1 and for edited states in stages 2 and 3
void qd_o2i_geometry(const qd_family *f, int io, double c1, double c2, int floor, arma::u32 &floor_id, double &d2d_in)
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
unsigned long long qd_links_hash(const quadriga_lib::layout &L)
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

// Checks of the arguments shared by the channel functions
void qd_check_channel_args(const std::string &output, const std::string &mobility, double overlap)
{
    if (output != "taps" && output != "rays")
        throw std::invalid_argument("Input 'output' must be 'taps' or 'rays'.");
    if (mobility != "phase" && mobility != "drift" && mobility != "auto")
        throw std::invalid_argument("Input 'mobility' must be 'phase', 'drift' or 'auto'.");
    if (!(overlap >= 0.0 && overlap <= 1.0))
        throw std::invalid_argument("Input 'overlap' must be in [0, 1].");
}

void qd_check_method(const std::string &method)
{
    if (method != "38.901" && method != "38.901_MF" && method != "quadriga")
        throw std::invalid_argument("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
}

// Stage 1 into 'K' without changing the layout
void qd_init_links(const quadriga_lib::layout &L, const std::string &method, const arma::vec &o2i_low, bool pol_variability,
                   double anchor_freq, const std::string &sos_preset, arma::sword seed, const std::vector<std::string> &config_dirs,
                   quadriga_lib::layout_links &K)
{
    // Arguments
    qd_check_method(method);
    if (o2i_low.n_elem > 1 || (o2i_low.n_elem == 1 && !(o2i_low[0] >= 0.0 && o2i_low[0] <= 1.0)))
        throw std::invalid_argument("Input 'o2i_low' must be empty or one value in [0, 1].");
    if (pol_variability && method == "quadriga")
        throw std::invalid_argument("Input 'pol_variability' applies to the TR methods only.");
    if (!(anchor_freq > 0.0) || !std::isfinite(anchor_freq))
        throw std::invalid_argument("Input 'anchor_freq' must be positive and finite.");
    const std::string preset = !sos_preset.empty() ? sos_preset : (method == "quadriga" ? "Comb256" : "Exp256");
    try
    {
        quadriga_lib::sos_preset<double>(preset);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("Input 'sos_preset' is not a known preset ('" + preset + "').");
    }
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");

    // The layout; its old links are replaced, so their state does not matter
    std::string msg = L.is_valid();
    if (msg.compare(0, 7, "links: ") == 0)
    {
        quadriga_lib::layout C = L;
        C.links.reset();
        msg = C.is_valid();
    }
    if (!msg.empty())
        throw std::invalid_argument("Input 'layout' is not valid: " + msg);
    if (L.center_freq.is_empty())
        throw std::invalid_argument("Member 'center_freq' of 'layout' must not be empty; set the simulation frequencies.");

    // Master seed
    arma::sword master = seed != -1 ? seed : L.seed;
    if (master == -1)
    {
        std::random_device rd;
        master = (arma::sword)((((unsigned long long)rd() << 32) ^ (unsigned long long)rd()) & 0x7FFFFFFFFFFFFFFFULL);
    }

    const qd_sides S = qd_make_sides(L);
    const arma::mat img = qd_images(L.wrap);
    const arma::uword n_img = img.n_cols;

    // Tracks of the terminals: one segment, i.e. scenario and indoor state constant along the rows
    for (size_t t = 0; t < S.n_term; ++t)
    {
        const arma::uword i = S.term_nodes[t][0], r0 = L.rx.offset[i], n = L.rx.n_row(i);
        for (arma::uword r = r0 + 1; r < r0 + n; ++r)
            if ((!L.rx.scenario.empty() && L.rx.scenario[r] != L.rx.scenario[r0]) || (!L.rx.indoor.is_empty() && L.rx.indoor[r] != L.rx.indoor[r0]))
                throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': a change of 'scenario' or 'indoor' along a track is not supported yet.");
    }

    // Pairs farther apart than 1e-4 m, terminal-major; the image with the shortest 2D distance
    std::vector<arma::uword> l_tx, l_rx, l_img, l_row;
    std::vector<double> l_d2d;
    for (size_t t = 0; t < S.n_term; ++t)
    {
        const arma::uword rr = qd_segment_row(L, S, t, 0);
        for (size_t s = 0; s < S.n_site; ++s)
        {
            const arma::uword rt = L.tx.offset[S.site_nodes[s][0]];
            const double dx = L.rx.pos(0, rr) - L.tx.pos(0, rt), dy = L.rx.pos(1, rr) - L.tx.pos(1, rt), dz = L.rx.pos(2, rr) - L.tx.pos(2, rt);
            arma::uword best = 0;
            double d = arma::datum::inf;
            for (arma::uword k = 0; k < n_img; ++k)
            {
                const double e = std::hypot(dx - img(0, k), dy - img(1, k));
                if (e < d)
                    d = e, best = k;
            }
            if (std::sqrt(d * d + dz * dz) <= 1.0e-4)
                continue;
            l_tx.push_back(s), l_rx.push_back(t), l_img.push_back(best), l_d2d.push_back(d), l_row.push_back(rr);
        }
    }
    const arma::uword n_link = (arma::uword)l_tx.size();

    // Scenarios of the links
    qd_scenarios SC;
    std::vector<qd_scen> scen(n_link);
    std::vector<char> scen_from_rx(n_link, 0);
    for (arma::uword l = 0; l < n_link; ++l)
    {
        bool from_rx = false;
        scen[l] = SC.of(L, S, l_tx[l], l_row[l], &from_rx);
        scen_from_rx[l] = from_rx;
        if (scen[l].name.empty())
            throw std::invalid_argument("No scenario for site '" + L.tx.name[S.site_nodes[l_tx[l]][0]] + "' and terminal '" +
                                        L.rx.name[S.term_nodes[l_rx[l]][0]] + "': set 'scenario' on the Tx or the Rx side.");
    }

    // Conf names: all confs of each family in use, user confs; resolved to files
    std::vector<std::string> conf_names;
    std::vector<char> is_user;
    auto add_conf = [&](const std::string &c, bool user)
    {
        if (!c.empty() && std::find(conf_names.begin(), conf_names.end(), c) == conf_names.end())
            conf_names.push_back(c), is_user.push_back(user);
    };
    for (const qd_scen &s : scen)
        if (s.fam == nullptr)
            add_conf(s.name, true);
        else
            for (const std::string *c : {&s.fam->los_conf, &s.fam->nlos_conf, &s.fam->los_o2i_conf, &s.fam->nlos_o2i_conf})
                add_conf(*c, false);
    std::vector<std::string> file_names, file_dirs;
    const std::vector<std::string> stems = quadriga_lib::conf_file_list(config_dirs, &file_names, &file_dirs, false, true);
    std::vector<std::string> conf_files(conf_names.size());
    for (size_t c = 0; c < conf_names.size(); ++c)
    {
        const std::string stem = is_user[c] ? conf_names[c] : qd_v19 + conf_names[c];
        const auto it = std::find(stems.begin(), stems.end(), stem);
        if (it == stems.end())
        {
            // A name of the layout that is no conf file: name the node it came from
            for (arma::uword l = 0; l < n_link; ++l)
                if (scen[l].fam == nullptr && scen[l].name == conf_names[c])
                {
                    const std::string who = scen_from_rx[l] ? "Terminal '" + L.rx.name[S.term_nodes[l_rx[l]][0]] + "'" : "Site '" + L.tx.name[S.site_nodes[l_tx[l]][0]] + "'";
                    throw std::invalid_argument(who + ": scenario '" + conf_names[c] +
                                                "' is neither a scenario family nor a conf file in 'config_dirs' or the shipped folder.");
                }
            throw std::runtime_error("The conf file '" + stem + ".conf' of the library is missing.");
        }
        const size_t j = (size_t)(it - stems.begin());
        const std::string &dir = file_dirs[j];
        conf_files[c] = dir.empty() ? file_names[j] : (dir.back() == '/' || dir.back() == '\\' ? dir : dir + "/") + file_names[j];
    }
    auto conf_index = [&](const std::string &c) { return (arma::uword)(std::find(conf_names.begin(), conf_names.end(), c) - conf_names.begin()); };

    // O2I state per terminal, from the fields of its O2I family at its first position. The O2I inputs, the model and the
    // floor are stored for every terminal of a family with O2I, whatever its state, so that an edited state finds them
    const std::vector<const qd_family *> term_fam = qd_term_families(scen, arma::conv_to<arma::u32_vec>::from(l_rx), S.n_term);
    std::vector<int> t_indoor(S.n_term, 0), t_floor(S.n_term, 1);
    arma::mat o2i_rnd(S.n_term, 4, arma::fill::zeros);
    arma::u32_vec o2i_model(S.n_term, arma::fill::zeros);
    std::vector<int> t_fixed(S.n_term, -1); // Building state fixed by an exact name, and the name
    std::vector<std::string> t_fixed_name(S.n_term);
    for (arma::uword l = 0; l < n_link; ++l)
        if (scen[l].fam == term_fam[l_rx[l]] && scen[l].los >= 0 && t_fixed[l_rx[l]] < 0)
            t_fixed[l_rx[l]] = scen[l].building, t_fixed_name[l_rx[l]] = scen[l].name;
    for (const qd_family &f : qd_layout::families())
    {
        std::vector<size_t> T;
        for (size_t t = 0; t < S.n_term; ++t)
            if (term_fam[t] == &f)
                T.push_back(t);
        if (T.empty())
            continue;
        arma::mat xy(T.size(), 2);
        for (size_t k = 0; k < T.size(); ++k)
        {
            const arma::uword r = qd_segment_row(L, S, T[k], 0);
            xy(k, 0) = L.rx.pos(0, r), xy(k, 1) = L.rx.pos(1, r);
        }
        const arma::vec u_io = qd_layout::o2i_field(master, f.name, qd_layout::s_io, xy);
        const arma::vec u_class = f.p_residential > 0.0 ? qd_layout::o2i_field(master, f.name, qd_layout::s_o2i_class, xy) : arma::vec();
        arma::mat rnd = qd_layout::o2i_rnd(master, f.name, xy);
        for (size_t k = 0; k < T.size(); ++k)
        {
            const size_t t = T[k];
            const arma::uword i = S.term_nodes[t][0], r = qd_segment_row(L, S, t, 0);
            const int fixed = L.rx.indoor.is_empty() ? -1 : L.rx.indoor[r];
            const int b = t_fixed[t];
            if ((b == 1 && (fixed == 0 || fixed == 2)) || (b == 0 && (fixed == 1 || fixed == 3)))
                throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': the conf '" + t_fixed_name[t] + "' fixes " +
                                            (b == 1 ? "a building" : "an outdoor") + " state, but 'indoor' is " + std::to_string(fixed) + ".");
            const int cls = (fixed == 1 || fixed == 3) ? fixed : (u_class.is_empty() ? 1 : (u_class[k] < f.p_residential ? 3 : 1)); // Building class
            int state = fixed;
            if (b == 1 && fixed < 0)
                state = cls;
            else if (b == 0 && fixed < 0)
                state = f.car ? 2 : 0;
            else if (fixed < 0)
                state = u_io[k] < f.building_share ? cls : (f.car ? 2 : 0);
            t_indoor[t] = state;
            t_floor[t] = qd_floor(L.rx, r);
            const double dmax = qd_d2d_in_max(f, cls), d = L.rx.d2d_in.is_empty() ? arma::datum::nan : L.rx.d2d_in[r];
            if (!std::isnan(d) && (state == 1 || state == 3) && d > dmax)
                throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': 'd2d_in' of " + qd_num(d) + " m is above the maximum indoor distance " +
                                            qd_num(dmax) + " m of the family '" + f.name + "'.");
            if (!std::isnan(d) && d <= dmax)
                rnd(k, 1) = rnd(k, 2) = dmax > 0.0 ? d / dmax : 0.0;
            o2i_rnd.row(t) = rnd.row(k);
        }
        arma::uvec mi; // The building model of each terminal, from column 0
        quadriga_lib::o2i_loss(arma::vec{L.center_freq[0]}, rnd, f.o2i_models, qd_o2i_weights(f, o2i_low), f.d2d_in_max, nullptr, &mi);
        for (size_t k = 0; k < T.size(); ++k)
            o2i_model[T[k]] = (arma::u32)mi[k];
    }

    // States of the links
    K.reset();
    K.i_tx.set_size(n_link), K.i_rx.set_size(n_link), K.i_segment.zeros(n_link), K.image.set_size(n_link), K.i_conf.set_size(n_link);
    K.los.set_size(n_link), K.indoor.set_size(n_link), K.floor_id.set_size(n_link), K.d2d_in.set_size(n_link), K.gbsm_seed.set_size(n_link);
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const size_t t = l_rx[l];
        K.i_tx[l] = (arma::u32)l_tx[l], K.i_rx[l] = (arma::u32)t, K.image[l] = (arma::u32)l_img[l];
        const qd_family *f = scen[l].has_o2i() ? term_fam[t] : nullptr;
        K.indoor[l] = f != nullptr ? t_indoor[t] : 0;
        qd_o2i_geometry(f, K.indoor[l], o2i_rnd(t, 1), o2i_rnd(t, 2), t_floor[t], K.floor_id[l], K.d2d_in[l]);
    }

    // LOS states: fixed by name, always LOS (InF-HH), or drawn per site, image and family
    std::map<std::tuple<size_t, arma::uword, const qd_family *>, std::vector<arma::uword>> draw;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const qd_scen &s = scen[l];
        if (s.fam == nullptr)
            K.los[l] = 0; // A user conf is used in either state
        else if (s.los >= 0)
            K.los[l] = (arma::u32)s.los;
        else if (s.fam->nlos_conf.empty())
            K.los[l] = 1;
        else
            draw[{l_tx[l], l_img[l], s.fam}].push_back(l);
    }
    std::map<const qd_family *, double> lambda;
    for (const auto &kv : draw)
    {
        const size_t s = std::get<0>(kv.first);
        const arma::uword k = std::get<1>(kv.first), i_node = S.site_nodes[s][0], rt = L.tx.offset[i_node];
        const qd_family *f = std::get<2>(kv.first);
        if (lambda.find(f) == lambda.end())
            lambda[f] = quadriga_lib::conf_file_read(conf_files[conf_index(f->nlos_conf)]).at("LOS_state_lambda");
        const double dcl = L.tx.clutter.is_empty() ? arma::datum::nan : L.tx.clutter(0, i_node);
        const double rcl = L.tx.clutter.is_empty() ? arma::datum::nan : L.tx.clutter(1, i_node);
        const double hcl = L.tx.clutter.is_empty() ? arma::datum::nan : L.tx.clutter(2, i_node);
        const bool inf = f->name.compare(0, 3, "InF") == 0;
        double lam = (inf && dcl > 0.0) ? 0.5 * dcl : lambda[f]; // TR 38.901 Table 7.6.3.1-2: d_clutter / 2
        const auto ov = L.tx.scenpar_override.find("LOS_state_lambda");
        if (ov != L.tx.scenpar_override.end() && !std::isnan(ov->second[i_node]))
            lam = ov->second[i_node];
        const std::vector<arma::uword> &ls = kv.second;
        const arma::uword n = (arma::uword)ls.size();
        arma::mat xy(n, 2);
        arma::vec d_out(n), h_ut(n);
        for (arma::uword j = 0; j < n; ++j)
        {
            const arma::uword r = l_row[ls[j]];
            xy(j, 0) = L.rx.pos(0, r), xy(j, 1) = L.rx.pos(1, r), h_ut(j) = L.rx.pos(2, r);
            d_out(j) = std::max(l_d2d[ls[j]] - K.d2d_in[ls[j]], 0.0);
        }
        const arma::vec p = quadriga_lib::los_probability(f->los_model, d_out, arma::vec{L.tx.pos(2, rt)}, h_ut, dcl, rcl, hcl);
        const std::string key = "los|" + L.tx.name[i_node] + "|" + std::to_string(k) + "|" + f->name;
        arma::mat u;
        quadriga_lib::sos_rand<double>(arma::vec{lam}, xy, u, arma::mat(), arma::vec(), "Exp256",
                                       qd_layout_seed_hash(qd_site_base(L.tx, i_node, master), key));
        for (arma::uword j = 0; j < n; ++j)
            K.los[ls[j]] = u(j, 0) < p(j) ? 1 : 0;
    }

    // Confs and seeds
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const std::string c = qd_conf_of(scen[l], K.los[l], K.indoor[l]);
        K.i_conf[l] = conf_index(c);
        const arma::uword i_node = S.site_nodes[l_tx[l]][0];
        K.gbsm_seed[l] = qd_layout_seed_hash(qd_site_base(L.tx, i_node, master), "gbsm|" + L.tx.name[i_node] + "|" + std::to_string(l_img[l]) + "|" + c);
    }
    K.conf_names = conf_names, K.conf_files = conf_files;
    K.o2i_rnd = o2i_rnd, K.o2i_model = o2i_model;
    K.seed = master, K.method = method, K.sos_preset = preset, K.anchor_freq = anchor_freq, K.pol_variability = pol_variability;
    K.layout_hash = qd_links_hash(L);
}

// One conf file as read
struct qd_conf
{
    qd_spmap sp;
    std::string pl;
    qd_plmap pp;
};

// The stages 2 and 3 of one layout with its links: derived confs, seeds and O2I geometry, groups of links with one site,
// image, conf and seed, cached SOS phases of the "lsf" sets
class qd_pipe
{
  public:
    const quadriga_lib::layout &L;
    const quadriga_lib::layout_links &K;
    qd_sides S;
    arma::mat img;
    bool qd, mf;
    arma::uword n_freq;
    std::vector<qd_scen> scen;                    // Per link
    std::vector<const qd_family *> term_fam;      // Per terminal
    std::vector<arma::uword> conf, group, row;    // Per link: derived conf index, group, Rx row at the segment start
    std::vector<arma::sword> seed;                // Per link
    arma::u32_vec floor_id;                       // Per link, from the (edited) indoor state
    arma::vec d2d_in;                             // Per link, from the (edited) indoor state
    std::vector<char> dual;                       // Per site
    std::vector<qd_conf> confs;                   // Per conf name
    std::vector<char> loaded;                     // Conf files read
    struct grp
    {
        arma::uword site, image, conf;
        arma::sword seed;
        std::vector<arma::uword> links;
        qd_spmap sp;    // Conf with the overrides of the site
        arma::cube lsf; // SOS phases of the "lsf" set; empty until needed
        arma::mat tx;   // Virtual position of the site at its first row, [3, 1]
    };
    std::vector<grp> groups;

    static constexpr size_t qd_par_rays = 4; // Size limit of the parallel ray generation (groups), see qd_par_channels

    qd_pipe(const quadriga_lib::layout &layout, const quadriga_lib::layout_links &links, bool check_hash) : L(layout), K(links)
    {
        if (K.method.empty())
            throw std::invalid_argument("Call 'init_link_states' first.");
        std::string msg = K.is_valid();
        if (!msg.empty())
            throw std::invalid_argument("Member 'links' of 'layout' is not valid: " + msg);
        qd_check_sizes(L);
        if (check_hash && qd_links_hash(L) != K.layout_hash)
            throw std::invalid_argument("The layout has changed since 'init_link_states': call it again, or 'links.reset()' to discard the links.");
        qd_check_method(K.method);
        qd = K.method == "quadriga", mf = K.method == "38.901_MF";
        n_freq = L.center_freq.n_elem;
        S = qd_make_sides(L);
        img = qd_images(L.wrap);
        if (!L.pairing.is_empty() && (L.pairing.n_rows != 2 || L.pairing.row(0).max() >= S.n_site || L.pairing.row(1).max() >= S.n_term))
            throw std::invalid_argument("Input 'layout' is not valid: member 'pairing' must have 2 rows with indices of sites and terminals.");
        const arma::uword n_link = K.n_link();
        if (K.o2i_rnd.n_rows != S.n_term || K.conf_files.size() != K.conf_names.size())
            throw std::invalid_argument("Member 'links' of 'layout' does not match the layout: call 'init_link_states' again.");
        for (arma::uword l = 0; l < n_link; ++l)
        {
            if (K.i_tx[l] >= S.n_site || K.i_rx[l] >= S.n_term || K.image[l] >= img.n_cols)
                throw std::invalid_argument("Member 'links' of 'layout' has a site, terminal or image index outside the layout.");
            if (K.i_segment[l] != 0)
                throw std::invalid_argument("Member 'links' of 'layout': segments are not supported yet.");
        }

        // Derived confs, seeds and O2I geometry
        qd_scenarios SC;
        scen.resize(n_link), conf.resize(n_link), seed.resize(n_link), row.resize(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            row[l] = qd_segment_row(L, S, K.i_rx[l], K.i_segment[l]);
            scen[l] = SC.of(L, S, K.i_tx[l], row[l]);
            const std::string c = qd_conf_of(scen[l], K.los[l], K.indoor[l]);
            const auto jt = std::find(K.conf_names.begin(), K.conf_names.end(), c);
            if (c.empty())
                throw std::invalid_argument("Entry " + std::to_string(l) + ": the family '" + scen[l].fam->name + "' has no conf for the state of the link (NLOS).");
            if (jt == K.conf_names.end())
                throw std::invalid_argument("Entry " + std::to_string(l) + ": the conf '" + c + "' of the state of the link is not in 'links.conf_names'.");
            conf[l] = (arma::uword)(jt - K.conf_names.begin());
            const arma::uword i_node = S.site_nodes[K.i_tx[l]][0];
            seed[l] = conf[l] == K.i_conf[l] ? (arma::sword)K.gbsm_seed[l]
                                              : qd_layout_seed_hash(qd_site_base(L.tx, i_node, K.seed), "gbsm|" + L.tx.name[i_node] + "|" +
                                                                                                            std::to_string(K.image[l]) + "|" + c);
        }
        term_fam = qd_term_families(scen, K.i_rx, S.n_term);
        floor_id.set_size(n_link), d2d_in.set_size(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::uword t = K.i_rx[l];
            qd_o2i_geometry(scen[l].has_o2i() ? term_fam[t] : nullptr, K.indoor[l], K.o2i_rnd(t, 1), K.o2i_rnd(t, 2), qd_floor(L.rx, row[l]),
                            floor_id[l], d2d_in[l]);
        }
        dual.resize(S.n_site);
        for (size_t s = 0; s < S.n_site; ++s)
            dual[s] = qd_site_moves(L.tx, S.site_nodes[s][0]) ? 1 : 0;

        // Groups
        std::map<std::tuple<arma::uword, arma::uword, arma::uword, arma::sword>, arma::uword> gi;
        group.resize(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const auto key = std::make_tuple((arma::uword)K.i_tx[l], (arma::uword)K.image[l], conf[l], seed[l]);
            auto it = gi.find(key);
            if (it == gi.end())
            {
                it = gi.emplace(key, (arma::uword)groups.size()).first;
                grp g;
                g.site = K.i_tx[l], g.image = K.image[l], g.conf = conf[l], g.seed = seed[l];
                const arma::uword r = L.tx.offset[S.site_nodes[g.site][0]];
                g.tx = L.tx.pos.col(r);
                g.tx(0, 0) += img(0, g.image), g.tx(1, 0) += img(1, g.image);
                groups.push_back(g);
            }
            group[l] = it->second;
            groups[it->second].links.push_back(l);
        }
        confs.resize(K.conf_names.size());
        loaded.assign(K.conf_names.size(), 0);
    }

    // Position of the terminal of a link at the start of its segment, [3, 1]
    arma::mat rx0(arma::uword l) const { return L.rx.pos.col(row[l]); }

    bool building(arma::uword l) const { return K.indoor[l] == 1 || K.indoor[l] == 3; }

    // Reads the confs and prepares scenpar and the "lsf" phases of the groups of 'links' (serial: file access)
    void prepare(const std::vector<arma::uword> &links)
    {
        std::vector<char> need(groups.size(), 0);
        for (arma::uword l : links)
            need[group[l]] = 1;
        for (size_t g = 0; g < groups.size(); ++g)
            if (need[g] && groups[g].sp.empty())
            {
                grp &G = groups[g];
                if (!loaded[G.conf])
                {
                    qd_conf &c = confs[G.conf];
                    c.sp = quadriga_lib::conf_file_read(K.conf_files[G.conf], &c.pl, &c.pp);
                    loaded[G.conf] = 1;
                }
                G.sp = confs[G.conf].sp;
                const arma::uword i_node = S.site_nodes[G.site][0];
                for (const auto &kv : L.tx.scenpar_override)
                    if (!std::isnan(kv.second[i_node]))
                        G.sp[kv.first] = kv.second[i_node];
                if (!L.tx.scenpar_override.empty())
                    quadriga_lib::scenpar_validate(G.sp);
            }
    }

    // The "lsf" phases of a group; bias removal over the first positions of all its links for quadriga
    const arma::cube &lsf(arma::uword g)
    {
        grp &G = groups[g];
        if (G.lsf.is_empty())
        {
            arma::mat rx;
            if (qd)
            {
                rx.set_size(3, G.links.size());
                for (size_t j = 0; j < G.links.size(); ++j)
                    rx.col(j) = rx0(G.links[j]);
            }
            G.lsf = quadriga_lib::gbsm_init_phases(G.sp, "lsf", G.seed, -1, K.sos_preset, qd ? G.tx : arma::mat(), rx, dual[G.site] != 0, qd, true);
        }
        return G.lsf;
    }
    // Rays of the links 'ls' of group g into out[pos[j]]
    void group_rays(arma::uword g, const std::vector<arma::uword> &ls, const std::vector<arma::uword> &pos, bool scatterers,
                    std::vector<std::vector<quadriga_lib::gbsm_rays>> &out)
    {
        const grp &G = groups[g];
        const qd_spmap &sp = G.sp;
        const qd_conf &c = confs[G.conf];
        const std::string &preset = K.sos_preset;
        const bool dm = dual[G.site] != 0;
        const arma::uword n = (arma::uword)ls.size();
        const arma::vec &fc = L.center_freq;
        arma::mat rx(3, n);
        arma::u32_vec fl(n);
        arma::vec d2d(n);
        std::vector<bool> o2i(n);
        for (arma::uword j = 0; j < n; ++j)
            rx.col(j) = rx0(ls[j]), fl[j] = floor_id[ls[j]], d2d[j] = d2d_in[ls[j]], o2i[j] = building(ls[j]);
        const arma::mat &tx = G.tx;

        arma::mat rnd, sf, eod, gre, gim, rc;
        quadriga_lib::gbsm_draw(sp, "lsf", lsf(g), tx, rx, rnd, -1, preset, fl, {}, dm, true);
        quadriga_lib::path_loss(c.pl, c.pp, sp, fc, tx, rx, &sf, qd ? d2d : arma::vec(), arma::vec(rnd.col(10)));
        const bool gr = sp.at("GR_enabled") != 0.0;
        arma::cube lsp, mu, sg;
        arma::vec toa;
        quadriga_lib::lsp_generate(sp, fc, tx, rx, rnd, lsp, sf, &toa, gr ? &gre : nullptr, gr ? &gim : nullptr, &eod, &mu, &sg);
        arma::cube ph = quadriga_lib::gbsm_init_phases(sp, "cluster", G.seed, -1, preset);
        quadriga_lib::gbsm_draw(sp, "cluster", ph, tx, rx, rc, -1, preset, fl, {}, dm, true);
        arma::cube D, P, AOD, EOD, AOA, EOA;
        std::vector<quadriga_lib::gbsm_rays> rays;
        if (qd)
        {
            quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rc, D, P, AOD, EOD, AOA, EOA, "quadriga", {}, {}, eod, toa, gre, gim, o2i);
            arma::mat rr;
            ph = quadriga_lib::gbsm_init_phases(sp, "ray", G.seed, -1, preset);
            quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, rr, -1, preset, fl, fc, dm, true);
            const arma::cube rnd_ray(rr.memptr(), rr.n_rows, rr.n_cols, 1, false, true);
            quadriga_lib::ssf_rays(sp, tx, rx, fc, lsp, mu, sg, D, P, AOD, EOD, AOA, EOA, rnd_ray, rays, "quadriga", 0, "rotation", "quadriga", gre, gim);
            for (arma::uword j = 0; j < n; ++j)
                out[pos[j]].assign(1, std::move(rays[j]));
        }
        else
        {
            const arma::uword N = (arma::uword)sp.at("NumClusters") - 1;
            arma::cube rf(n, N, n_freq), rr;
            for (arma::uword k = 0; k < n_freq; ++k)
            {
                arma::mat x;
                ph = quadriga_lib::gbsm_init_phases(sp, "cluster", G.seed, (arma::sword)k, preset);
                quadriga_lib::gbsm_draw(sp, "cluster", ph, tx, rx, x, (arma::sword)k, preset, fl, {}, dm, true);
                if (N > 0)
                    rf.slice(k) = x;
                ph = quadriga_lib::gbsm_init_phases(sp, "ray", G.seed, (arma::sword)k, preset);
                quadriga_lib::gbsm_draw(sp, "ray", ph, tx, rx, x, (arma::sword)k, preset, fl, {}, dm, true);
                if (k == 0)
                    rr.set_size(n, x.n_cols, n_freq);
                rr.slice(k) = x;
            }
            for (arma::uword j = 0; j < n; ++j)
                out[pos[j]].resize(n_freq);
            if (mf)
            {
                arma::cube la;
                quadriga_lib::lsp_generate(sp, arma::vec{K.anchor_freq}, tx, rx, rnd, la);
                quadriga_lib::ssf_clusters(sp, tx, rx, lsp, rc, D, P, AOD, EOD, AOA, EOA, "38.901_MF", rf, la, eod, toa, gre, gim, o2i);
                for (arma::uword k = 0; k < n_freq; ++k)
                {
                    quadriga_lib::ssf_rays(sp, tx, rx, fc, lsp, mu, sg, D, P, AOD, EOD, AOA, EOA, rr, rays, "38.901_MF", k, "additive", "38.901",
                                           gre, gim, K.pol_variability);
                    for (arma::uword j = 0; j < n; ++j)
                        out[pos[j]][k] = std::move(rays[j]);
                }
            }
            else
                for (arma::uword k = 0; k < n_freq; ++k)
                {
                    const arma::cube lk = lsp.cols(k, k), mk = mu.cols(k, k), sk = sg.cols(k, k), rfk = rf.slices(k, k), rrk = rr.slices(k, k);
                    const arma::mat ek = eod.col(k), grk = gr ? arma::mat(gre.col(k)) : arma::mat(), gik = gr ? arma::mat(gim.col(k)) : arma::mat();
                    quadriga_lib::ssf_clusters(sp, tx, rx, lk, rc, D, P, AOD, EOD, AOA, EOA, "38.901", rfk, {}, ek, toa, grk, gik, o2i);
                    quadriga_lib::ssf_rays(sp, tx, rx, arma::vec{fc(k)}, lk, mk, sk, D, P, AOD, EOD, AOA, EOA, rrk, rays, "38.901", 0, "additive",
                                           "38.901", grk, gik, K.pol_variability);
                    for (arma::uword j = 0; j < n; ++j)
                        out[pos[j]][k] = std::move(rays[j]);
                }
        }
        if (scatterers || qd)
            for (arma::uword j = 0; j < n; ++j)
                for (auto &r : out[pos[j]])
                    quadriga_lib::ssf_scatterers(r);
    }

    // Rays of the links 'sel' in this order, [n_out][1 or n_freq]
    std::vector<std::vector<quadriga_lib::gbsm_rays>> rays(const std::vector<arma::uword> &sel, bool scatterers)
    {
        prepare(sel);
        std::vector<std::vector<arma::uword>> ls(groups.size()), pos(groups.size());
        for (size_t j = 0; j < sel.size(); ++j)
            ls[group[sel[j]]].push_back(sel[j]), pos[group[sel[j]]].push_back((arma::uword)j);
        std::vector<arma::uword> todo;
        for (size_t g = 0; g < groups.size(); ++g)
            if (!ls[g].empty())
            {
                todo.push_back((arma::uword)g);
                if (qd)
                    lsf((arma::uword)g); // bias removal before the parallel region; cached for stage 3
            }
        std::vector<std::vector<quadriga_lib::gbsm_rays>> out(sel.size());
        qd_batch_loop(todo.size(), todo.size() >= qd_par_rays, nullptr, [&](size_t i)
                      {
                          const arma::uword g = todo[i];
                          qd_named("Site '" + L.tx.name[S.site_nodes[groups[g].site][0]] + "', conf '" + K.conf_names[groups[g].conf] + "': ", [&]
                                   {
                                       if (!qd)
                                           lsf(g); // each group is handled by one thread
                                       group_rays(g, ls[g], pos[g], scatterers, out);
                                   });
                      });
        return out;
    }

    // Link gain [n_snap or 1, n_freq] and K-factor ratio of link l at the Tx and Rx positions T, R [3, n] (virtual Tx)
    void link_gain(arma::uword l, const arma::mat &T, const arma::mat &R, arma::mat &gain, arma::mat &kfr)
    {
        const arma::uword g = group[l];
        const grp &G = groups[g];
        const qd_conf &c = confs[G.conf];
        const arma::uword n = R.n_cols;
        const arma::vec &fc = L.center_freq;
        const arma::mat A = arma::join_rows(G.tx, T), B = arma::join_rows(rx0(l), R); // Segment start, then the snapshots
        const arma::u32_vec fl(n + 1, arma::fill::value(floor_id[l]));
        arma::mat rnd, sf;
        quadriga_lib::gbsm_draw(G.sp, "lsf", groups[g].lsf, A, B, rnd, -1, K.sos_preset, fl, {}, dual[G.site] != 0, true);
        const arma::vec he(n + 1, arma::fill::value(rnd(0, 10))); // hE held at the segment start
        const arma::mat pl = quadriga_lib::path_loss(c.pl, c.pp, G.sp, fc, A, B, &sf, qd ? arma::vec(n + 1, arma::fill::value(d2d_in[l])) : arma::vec(), he);
        arma::cube lsp;
        quadriga_lib::lsp_generate(G.sp, fc, A, B, rnd, lsp, sf);
        arma::mat loss;
        const arma::uword t = K.i_rx[l];
        const qd_family *f = term_fam[t];
        const int io = K.indoor[l];
        if (f != nullptr && scen[l].has_o2i() && io >= 1)
        {
            arma::mat x(n + 1, 4);
            for (arma::uword i = 0; i <= n; ++i)
                x.row(i) = K.o2i_rnd.row(t);
            arma::mat xy;
            std::vector<arma::uword> moved;
            for (arma::uword i = 1; i <= n; ++i)
                if (B(0, i) != B(0, 0) || B(1, i) != B(1, 0))
                    moved.push_back(i);
            if (!moved.empty())
            {
                xy.set_size(moved.size(), 2);
                for (size_t j = 0; j < moved.size(); ++j)
                    xy(j, 0) = B(0, moved[j]), xy(j, 1) = B(1, moved[j]);
                const arma::vec dev = qd_layout::o2i_field(K.seed, f->name, qd_layout::s_o2i_dev, xy);
                for (size_t j = 0; j < moved.size(); ++j)
                    x(moved[j], 3) = dev[j];
            }
            if (io == 2)
                loss = quadriga_lib::o2i_loss(fc, x, {"car"}, arma::vec{1.0}, 0.0);
            else
            {
                const arma::uword m = std::min((arma::uword)K.o2i_model[t], (arma::uword)f->o2i_models.size() - 1);
                loss = quadriga_lib::o2i_loss(fc, x, {f->o2i_models[m]}, arma::vec{1.0}, qd_d2d_in_max(*f, io));
            }
        }
        gain.set_size(n, n_freq), kfr.set_size(n, n_freq);
        for (arma::uword k = 0; k < n_freq; ++k)
            for (arma::uword i = 0; i < n; ++i)
            {
                gain(i, k) = -pl(i + 1, k) - (loss.is_empty() ? 0.0 : loss(i + 1, k)) + 10.0 * std::log10(lsp(i + 1, k, 2));
                kfr(i, k) = lsp(i + 1, k, 1) / lsp(0, k, 1);
            }
    }

};

// The array of node i of a side at frequency k: inner entry k of its antenna, entry 0 of one for all bands, or the
// generic antenna for a node without antenna and for an antenna that is not evaluated. An antenna spaced in wavelengths
// that is not evaluated throws: its size in metres is not defined
const quadriga_lib::arrayant<double> *qd_array(const quadriga_lib::layout &L, const quadriga_lib::layout_nodes &N, const char *side, arma::uword i,
                                               arma::uword k, const std::vector<quadriga_lib::arrayant<double>> &xpol)
{
    const arma::uword a = N.i_antenna[i];
    if (a == 0)
        return &xpol[k];
    if (!L.antenna.empty() && !L.antenna[a - 1].empty())
    {
        const auto &v = L.antenna[a - 1];
        const quadriga_lib::arrayant<double> &A = v.size() == 1 ? v[0] : v[k];
        if (A.n_elements() != 0)
            return &A;
    }
    if (!L.antenna_args.empty() && !L.antenna_args[a - 1].empty())
    {
        const auto &lines = L.antenna_args[a - 1];
        const std::string &line = lines.size() == 1 ? lines[0] : lines[k];
        const unsigned spaced = qd_layout::t_ula | qd_layout::t_3gpp | qd_layout::t_multibeam;
        if (!line.empty() &&
            (qd_layout::antenna_type_mask(qd_layout::parse_antenna_line(line, L.antenna_cols)[qd_layout::c_antenna - qd_layout::c_center_freq]) & spaced))
            throw std::invalid_argument(std::string(side) + " node '" + N.name[i] + "' (sector " + std::to_string(N.sector[i]) +
                                        "): its antenna is spaced in wavelengths and not evaluated; call 'eval_antennas' of the layout.");
    }
    return &xpol[k];
}

// Name of a node in a channel name: "<name>:<sector>", without the sector if all sectors of the side are 0
std::string qd_node_name(const quadriga_lib::layout_nodes &N, arma::uword i)
{
    return arma::any(N.sector != 0) ? N.name[i] + ":" + std::to_string(N.sector[i]) : N.name[i];
}

// Size limits of the parallel loops, measured (one thread vs. 32, idle machine): a channel costs about 1.2 ms (2x at 4
// channels), the link gain of a link 0.1 to 0.7 ms (far above 20 us per item); the ray generation of a group 55 ms with
// the parallel loops inside the GBSM functions, which a parallel loop over the groups switches off (2x at 4 groups)
constexpr size_t qd_par_channels = 4; // Channels
constexpr size_t qd_par_links = 2;    // Links of the link gain

// One channel link: Tx node, Rx node and the positions (in the list of propagation links of a call) of its propagation
// links, one per segment of the track
struct qd_item
{
    arma::uword tx, rx;
    std::vector<arma::uword> j;
};

// The channel links of the paired propagation links 'sel', in canonical order: Rx node, then Tx node
std::vector<qd_item> qd_items(const qd_pipe &P, const std::vector<arma::uword> &sel)
{
    const quadriga_lib::layout &L = P.L;
    std::vector<char> paired;
    if (!L.pairing.is_empty())
    {
        paired.assign(P.S.n_site * P.S.n_term, 0);
        for (arma::uword p = 0; p < L.pairing.n_cols; ++p)
            paired[L.pairing(0, p) * P.S.n_term + L.pairing(1, p)] = 1;
    }
    std::map<std::pair<arma::uword, arma::uword>, std::vector<arma::uword>> pairs; // (site, terminal) -> propagation links
    for (size_t j = 0; j < sel.size(); ++j)
    {
        const arma::uword s = P.K.i_tx[sel[j]], u = P.K.i_rx[sel[j]];
        if (paired.empty() || paired[s * P.S.n_term + u])
            pairs[{s, u}].push_back((arma::uword)j);
    }
    std::vector<qd_item> items;
    for (const auto &kv : pairs)
        for (arma::uword r : P.S.term_nodes[kv.first.second])
            for (arma::uword t : P.S.site_nodes[kv.first.first])
                items.push_back({t, r, kv.second});
    std::sort(items.begin(), items.end(), [](const qd_item &a, const qd_item &b) { return a.rx != b.rx ? a.rx < b.rx : a.tx < b.tx; });
    return items;
}

// Stage 3 for the channel links 'items' of the propagation links 'sel' with their rays; channels [n_freq * n_item],
// frequency slowest
std::vector<quadriga_lib::channel<double>> qd_channels(qd_pipe &P, const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays,
                                                       const std::vector<arma::uword> &sel, const std::vector<qd_item> &items, const arma::vec &times,
                                                       const std::string &output, const std::string &mobility, bool oxygen, bool use_absolute_delays)
{
    const quadriga_lib::layout &L = P.L;
    const quadriga_lib::layout_links &K = P.K;
    const arma::uword n_freq = P.n_freq;
    if (rays.size() != sel.size())
        throw std::invalid_argument("Input 'rays' must have one entry per link of 'i_link'.");
    for (size_t j = 0; j < sel.size(); ++j)
    {
        const auto &r = rays[j];
        if (r.size() != 1 && r.size() != n_freq)
            throw std::invalid_argument("Entry " + std::to_string(j) + ": Input 'rays' must have 1 or n_freq objects per link.");
        for (size_t k = 0; k < r.size(); ++k)
        {
            const bool one = r.size() == 1;
            const arma::uword nf = one ? n_freq : 1;
            bool ok = r[k].center_freq.n_elem == nf;
            for (arma::uword i = 0; ok && i < nf; ++i)
                ok = qd_layout::same_freq(r[k].center_freq[i], L.center_freq[one ? i : k]);
            if (!ok)
                throw std::invalid_argument("Entry " + std::to_string(j) + ": The frequencies of 'rays' do not match 'center_freq' of the layout.");
        }
    }
    for (arma::uword k = 0; k < times.n_elem; ++k)
        if (!std::isfinite(times[k]))
            throw std::invalid_argument("Input 'times' must be finite.");

    // The arrays of the nodes in use (serial: an antenna that is not evaluated may throw)
    std::vector<quadriga_lib::arrayant<double>> xpol(n_freq);
    for (arma::uword k = 0; k < n_freq; ++k)
        xpol[k] = quadriga_lib::generate_antenna<double>("xpol", NAN, L.center_freq[k]);
    std::vector<const quadriga_lib::arrayant<double> *> atx(L.tx.n_node() * n_freq, nullptr), arx(L.rx.n_node() * n_freq, nullptr);
    for (const qd_item &it : items)
        for (arma::uword k = 0; k < n_freq; ++k)
        {
            if (atx[it.tx * n_freq + k] == nullptr)
                atx[it.tx * n_freq + k] = qd_array(L, L.tx, "Tx", it.tx, k, xpol);
            if (arx[it.rx * n_freq + k] == nullptr)
                arx[it.rx * n_freq + k] = qd_array(L, L.rx, "Rx", it.rx, k, xpol);
        }

    // Positions and link gain per propagation link
    const size_t n_sel = sel.size();
    std::vector<arma::mat> T(n_sel), R(n_sel), G(n_sel), KF(n_sel);
    std::vector<char> used(n_sel, 0);
    for (const qd_item &it : items)
        for (arma::uword j : it.j)
            used[j] = 1;
    std::vector<arma::uword> need;
    for (size_t j = 0; j < n_sel; ++j)
        if (used[j])
            need.push_back(sel[j]);
    P.prepare(need);
    for (arma::uword l : need)
        P.lsf(P.group[l]);
    qd_batch_loop(n_sel, n_sel >= qd_par_links, nullptr, [&](size_t j)
                  {
                      if (!used[j])
                          return;
                      const arma::uword l = sel[j], nt = P.S.site_nodes[K.i_tx[l]][0], nr = P.S.term_nodes[K.i_rx[l]][0];
                      arma::mat o;
                      quadriga_lib::sample_track(L.tx, nt, times, T[j], o);
                      quadriga_lib::sample_track(L.rx, nr, times, R[j], o);
                      if (T[j].n_cols != R[j].n_cols)
                      {
                          if (T[j].n_cols == 1)
                              T[j] = arma::repmat(T[j], 1, R[j].n_cols);
                          else if (R[j].n_cols == 1)
                              R[j] = arma::repmat(R[j], 1, T[j].n_cols);
                          else
                              throw std::invalid_argument("Channel '" + qd_node_name(L.tx, nt) + "_" + qd_node_name(L.rx, nr) + "': the Tx and the Rx track have " +
                                                          std::to_string(T[j].n_cols) + " and " + std::to_string(R[j].n_cols) +
                                                          " rows; give 'times' or tracks with equal row counts.");
                      }
                      T[j].row(0) += P.img(0, K.image[l]), T[j].row(1) += P.img(1, K.image[l]);
                      auto fixed = [](const arma::mat &X)
                      {
                          for (arma::uword i = 1; i < X.n_cols; ++i)
                              if (X(0, i) != X(0, 0) || X(1, i) != X(1, 0) || X(2, i) != X(2, 0))
                                  return false;
                          return true;
                      };
                      if (fixed(T[j]) && fixed(R[j])) // A link whose ends do not move needs one evaluation
                          P.link_gain(l, T[j].col(0), R[j].col(0), G[j], KF[j]);
                      else
                          P.link_gain(l, T[j], R[j], G[j], KF[j]);
                  });

    // Channels; one segment per track so far, so a channel link has one propagation link
    const size_t n_item = items.size(), n_out = n_item * n_freq;
    std::vector<quadriga_lib::channel<double>> out(n_out);
    qd_batch_loop(n_out, n_out >= qd_par_channels, nullptr, [&](size_t c)
                  {
                      const arma::uword k = (arma::uword)(c / n_item);
                      const qd_item &it = items[c % n_item];
                      const arma::uword j = it.j[0];
                      const std::string name = qd_node_name(L.tx, it.tx) + "_" + qd_node_name(L.rx, it.rx);
                      qd_named("Channel '" + name + "': ", [&]
                               {
                                   arma::mat pos, otx, orx;
                                   quadriga_lib::sample_track(L.tx, it.tx, times, pos, otx);
                                   quadriga_lib::sample_track(L.rx, it.rx, times, pos, orx);
                                   const arma::uword n = T[j].n_cols;
                                   if (otx.n_cols != n)
                                       otx = arma::repmat(otx, 1, n);
                                   if (orx.n_cols != n)
                                       orx = arma::repmat(orx, 1, n);
                                   const auto &r = rays[j];
                                   const quadriga_lib::gbsm_rays &ray = r.size() == n_freq ? r[k] : r[0];
                                   const arma::uword i_freq = r.size() == n_freq ? 0 : k;
                                   out[c] = quadriga_lib::gbsm_single_link_channel(*atx[it.tx * n_freq + k], *arx[it.rx * n_freq + k], ray, G[j].col(k), mobility,
                                                                                   output, i_freq, T[j], R[j], otx, orx, KF[j].col(k), oxygen, use_absolute_delays);
                                   out[c].name = name;
                               });
                  });
    return out;
}

// Tx node, Rx node and frequency index of channels [n_freq * n_item]
void qd_index(const std::vector<qd_item> &items, arma::uword n_freq, arma::umat *index)
{
    if (index == nullptr)
        return;
    const size_t n_item = items.size();
    index->set_size(3, n_item * n_freq);
    for (size_t c = 0; c < n_item * n_freq; ++c)
        (*index)(0, c) = items[c % n_item].tx, (*index)(1, c) = items[c % n_item].rx, (*index)(2, c) = c / n_item;
}

// The links of an i_link argument: all, or the given ones without repetition
std::vector<arma::uword> qd_selection(const quadriga_lib::uint_list &i_link, arma::uword n_link)
{
    std::vector<arma::uword> sel;
    const arma::uvec &v = i_link.vec();
    if (v.is_empty())
    {
        sel.resize(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
            sel[l] = l;
        return sel;
    }
    std::vector<char> seen(n_link, 0);
    for (arma::uword l : v)
    {
        if (l >= n_link)
            throw std::invalid_argument("Input 'i_link' has an index outside the links.");
        if (seen[l])
            throw std::invalid_argument("Input 'i_link' must not repeat a link.");
        seen[l] = 1;
        sel.push_back(l);
    }
    return sel;
}
} // namespace

/*!MD
# init_link_states
Draws the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

- One propagation link per site and terminal farther apart than 1e-4 m, terminal-major (all sites of terminal 0 first); co-located pairs have no link. `layout.pairing` does not limit the links, it selects at the output of the channel functions
- Scenario of a link: the value of the terminal, else that of the site; a scenario family (e.g. `UMa`) draws the LOS and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes them, any other name is a user conf file used in either state without O2I loss. A conf that several families list (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found with [[conf_file_list]], the folders of `config_dirs` first
- Wrap image: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the terminal, ties to the lowest index in the order 0, +a1, −a1, +a2, −a2, +(a1 − a2), −(a1 − a2); all distances of the link use this virtual site position
- O2I state per terminal from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3): building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the O2I model and `d2d_in` as in [[o2i_loss]]. The values `indoor`, `floor` and `d2d_in` of the Rx side fix them; an exact conf name that contradicts a fixed `indoor` throws. All links of a terminal share one O2I state (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a site fixes it for the links to all sites; InH and InF have none
- `o2i_low` applies to the families with a low-loss and a high-loss building model (UMa, UMi); RMa and SMa have one building model
- LOS state per link: a uniform field per site, image and family at the first position of the terminal, at the `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [[los_probability]] at `d2d_out = max(d2D − d2d_in, 0)`; InF-HH is always LOS
- Random streams: [[sos_rand]] with the preset `Exp256` and seeds hashed from a base seed and a key (64-bit FNV-1a, sign bit cleared): base = the site seed or the layout seed; keys `los|<site>|<image>|<family>`, `gbsm|<site>|<image>|<conf>` (the seed of [[gbsm_init_phases]]) and, with the layout seed, `<purpose>|<family>` for the O2I fields. Adding a terminal does not change the states of the others
- The TR methods draw the cluster and ray sets of frequency `k` with the index `k` into `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands
- A track keeps the state of its first row; a change of `scenario` or `indoor` along a track throws (not supported yet)
- A layout without any pair farther apart than 1e-4 m gives links without entries; the later stages then return empty results

## Declaration:
```
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

## Inputs:
- **`layout`** — Layout, valid, with `center_freq`; `layout.links` and `layout.seed` are written, the rest is unchanged. A copy of `layout.links` stays usable after another realisation: the links hold their master seed, which the later stages use
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see [[generate_rays]]
- **`o2i_low`** *(optional)* — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; the other families have one building model; `[1]`
- **`pol_variability`** *(optional)* — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only
- **`anchor_freq`** *(optional)* — Anchor frequency of `"38.901_MF"` in [Hz]
- **`sos_preset`** *(optional)* — SOS preset of the GBSM random sets, see [[sos_preset]]; omitted: `Exp256` (TR methods), `Comb256` (`"quadriga"`)
- **`seed`** *(optional)* — Master seed, >= -1; -1 = `layout.seed`, if that is -1 one draw of the system random device. The seed used is written to `layout.seed`
- **`config_dirs`** *(optional)* — Folders of user conf files, searched before the shipped folder

## See also:
- [[layout_links]] (the link states)
- [[generate_rays]], [[get_gbsm_channels]] (stages 2 and 3)
- [[get_channels_layout]] (all stages in one call)
MD!*/

void quadriga_lib::init_link_states(layout &layout, const std::string &method, const arma::vec &o2i_low, bool pol_variability, double anchor_freq,
                                    const std::string &sos_preset, arma::sword seed, const std::vector<std::string> &config_dirs)
{
    layout_links K;
    qd_init_links(layout, method, o2i_low, pol_variability, anchor_freq, sos_preset, seed, config_dirs, K);
    layout.seed = K.seed;
    layout.links = std::move(K);
}

/*!MD
# generate_rays
Rays of propagation links of a layout (stage 2 of the channel pipeline)

- Runs the GBSM chain of the section "Geometry-based stochastic model" for the links with one site, image, conf and seed: [[gbsm_init_phases]] and [[gbsm_draw]] of the random sets, [[path_loss]] (for `rnd_he` and the SF standard deviation), [[lsp_generate]], [[ssf_clusters]], [[ssf_rays]] and, on request, [[ssf_scatterers]]; at the first position of the site (moved by the wrap image) and of the terminal
- TR methods: preset of the links (default `Exp256`), no bias removal, path loss at the geometric distance, `angle_model = "additive"`, `pol_model = "38.901"`, one set of clusters and rays per frequency (`"38.901_MF"`: clusters at the anchor frequency, rays per frequency). `"quadriga"`: bias removal over the positions of all links of the group, the legacy outdoor distance, `"rotation"`, `"quadriga"`, one ray set for all frequencies, always with scatterers
- A site with more than one row or a speed uses dual mobility; other sites are not affected
- The conf and seed of a link follow from its scenario and the edited `los` and `indoor` of `layout.links`; a changed conf uses its hashed seed, an edited `gbsm_seed` is used while the conf is unchanged. The floor and the indoor distance follow the indoor state: building UTs on the floor of the Rx side (else from z) at the distance of the stored O2I inputs, others on floor 0 at 0 m; the members `floor_id` and `d2d_in` of the links are those of the drawn state
- The values of a link do not depend on the other requested links

## Declaration:
```
std::vector<std::vector<gbsm_rays>> quadriga_lib::generate_rays(
    const layout &layout,
    const uint_list &i_link = {},
    bool scatterers = false);
```

## Inputs:
- **`layout`** — Layout with links from [[init_link_states]]; throws if a member that determines the links changed since then, or if the sizes of its members do not match
- **`i_link`** *(optional)* — Propagation links, 0-based, without repetition; omitted = all (none for links without entries); `[n_out]`
- **`scatterers`** *(optional)* — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`

## Returns:
- Rays per link and frequency, `[n_out][n_freq]` with one frequency each for the TR methods, `[n_out][1]` with all frequencies for `"quadriga"`

## See also:
- [[init_link_states]] (stage 1), [[get_gbsm_channels]] (stage 3)
- [[gbsm_rays]] (the ray data)
MD!*/

std::vector<std::vector<quadriga_lib::gbsm_rays>> quadriga_lib::generate_rays(const layout &layout, const uint_list &i_link, bool scatterers)
{
    qd_pipe P(layout, layout.links, true);
    return P.rays(qd_selection(i_link, layout.links.n_link()), scatterers);
}

/*!MD
# get_gbsm_channels
Channel coefficients of the channel links of propagation links of a layout (stage 3 of the channel pipeline)

- One channel per Tx node, Rx node and frequency whose propagation link is in `i_link` and whose pair is in `layout.pairing` (empty: all); order: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the sector where all sectors of the side are 0
- Snapshots of both ends from [[sample_track]] over `times`; a node with one snapshot is repeated to the count of the other; two tracks without `times` need equal row counts
- Link gain per snapshot (see the GBSM terms): [[gbsm_draw]] of the `"lsf"` set at the snapshot, [[path_loss]] with the random environment height `rnd_he` held at the start of the track, [[lsp_generate]] (SF and the K-factor ratio), [[o2i_loss]] for building and car UTs with the model of the link state and the deviation at the snapshot; evaluated once for a link whose ends do not move
- [[gbsm_single_link_channel]] with the node antennas at the frequency (inner entry `i_freq` of `layout.antenna`, or entry 0 for all bands), the generic antenna with two co-located isotropic elements of vertical and horizontal polarization for nodes without one and for antennas that are not evaluated (an antenna spaced in wavelengths that is not evaluated throws), and the Tx positions moved by the wrap image of the link; rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency `i_freq` (`"quadriga"`)

## Declaration:
```
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

## Inputs:
- **`layout`** — Layout with links from [[init_link_states]]; throws if a member that determines the links changed since then
- **`rays`** — Rays of the links from [[generate_rays]], possibly edited, with the frequencies of the layout; `[n_out][1 or n_freq]`
- **`i_link`** *(optional)* — The propagation links of `rays`, 0-based; omitted = all links; `[n_out]`
- **`times`** *(optional)* — Snapshot times in [s], finite, the same for all nodes; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`, see [[gbsm_single_link_channel]]
- **`mobility`** *(optional)* — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods, drift for `"quadriga"`)
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays
- **`overlap`** *(optional)* — Overlap of consecutive segments of a track, in [0, 1]; not used yet (one segment per track)

## Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

## Returns:
- Channels, see [[gbsm_single_link_channel]]; `[n_channel]`

## See also:
- [[generate_rays]] (stage 2)
- [[get_channels_layout]] (all stages in one call)
MD!*/

std::vector<quadriga_lib::channel<double>> quadriga_lib::get_gbsm_channels(const layout &layout, const std::vector<std::vector<gbsm_rays>> &rays,
                                                                           const uint_list &i_link, const arma::vec &times, const std::string &output,
                                                                           arma::umat *index, const std::string &mobility, bool oxygen,
                                                                           bool use_absolute_delays, double overlap)
{
    qd_check_channel_args(output, mobility, overlap);
    qd_pipe P(layout, layout.links, true);
    const std::vector<arma::uword> sel = qd_selection(i_link, layout.links.n_link());
    const std::vector<qd_item> items = qd_items(P, sel);
    std::vector<channel<double>> out = qd_channels(P, rays, sel, items, times, output, mobility, oxygen, use_absolute_delays);
    qd_index(items, P.n_freq, index);
    return out;
}

/*!MD
# get_channels_layout
Channel coefficients of a layout in one call

- Runs [[init_link_states]] (if `layout.links` was not filled), [[generate_rays]] and [[get_gbsm_channels]] in batches of at most 256 propagation links, so that the rays of a batch bound the memory; the results are those of the manual path with the default settings and do not depend on the batches
- Without links (`layout.links.method` empty, e.g. after `reset()`), the link states are drawn for this call only and the layout is unchanged; with filled links, these are used, also when edited, and `seed` and `config_dirs` are not used. No linked pair gives an empty result
- The mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers

## Declaration:
```
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

## Inputs:
- **`layout`** — Layout with `center_freq`; antennas as evaluated in `layout.antenna`, nodes without one use the generic antenna (see [[get_gbsm_channels]])
- **`times`** *(optional)* — Snapshot times in [s]; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` or `"quadriga"`; omitted = the method of filled links, else `"38.901"`; another method than that of filled links throws
- **`mobility`** *(optional)* — `"phase"`, `"drift"` or `"auto"`
- **`overlap`** *(optional)* — Overlap of consecutive segments, in [0, 1]
- **`seed`** *(optional)* — Master seed for empty links, as in [[init_link_states]]; with filled links only -1
- **`config_dirs`** *(optional)* — Folders of user conf files for empty links, searched before the shipped folder

## Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

## Returns:
- Channels, Tx node fastest, then Rx node, then frequency; `[n_channel]`

## Example:
```
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1);
L.center_freq = {3.5e9};
auto channels = quadriga_lib::get_channels_layout(L);
```

## See also:
- [[init_link_states]], [[generate_rays]], [[get_gbsm_channels]] (the three stages)
- [[generate_layout]] (layouts of the TR scenarios)
MD!*/

std::vector<quadriga_lib::channel<double>> quadriga_lib::get_channels_layout(const layout &layout, const arma::vec &times, const std::string &output,
                                                                             const std::string &method, arma::umat *index, const std::string &mobility,
                                                                             double overlap, arma::sword seed, const std::vector<std::string> &config_dirs)
{
    qd_check_channel_args(output, mobility, overlap);
    if (!method.empty())
        qd_check_method(method);
    layout_links local;
    const bool filled = !layout.links.method.empty();
    if (filled)
    {
        if (!method.empty() && method != layout.links.method)
            throw std::invalid_argument("Input 'method' is '" + method + "', but the links of the layout were drawn for '" + layout.links.method + "'.");
        if (seed != -1)
            throw std::invalid_argument("Input 'seed' must be -1 when the links of the layout are filled.");
    }
    else
        qd_init_links(layout, method.empty() ? "38.901" : method, {}, false, 2.0e9, "", seed, config_dirs, local);
    qd_pipe P(layout, filled ? layout.links : local, filled);
    const bool scatterers = mobility == "drift" || (mobility == "auto" && P.qd);

    // The paired links ordered by group, in batches of up to 256 links; the random sets are drawn once per group and
    // batch, the SOS phases of the "lsf" sets stay cached in P, and the values of a link do not depend on its batch
    std::vector<arma::uword> links;
    {
        std::vector<char> pair(P.S.n_site * P.S.n_term, layout.pairing.is_empty() ? 1 : 0);
        for (arma::uword p = 0; p < layout.pairing.n_cols; ++p)
            pair[layout.pairing(0, p) * P.S.n_term + layout.pairing(1, p)] = 1;
        for (arma::uword l = 0; l < P.K.n_link(); ++l)
            if (pair[P.K.i_tx[l] * P.S.n_term + P.K.i_rx[l]])
                links.push_back(l);
        std::stable_sort(links.begin(), links.end(), [&](arma::uword x, arma::uword y) { return P.group[x] < P.group[y]; });
    }
    const std::vector<qd_item> items = qd_items(P, links); // All channel links, canonical; j indexes 'links'
    const size_t batch = 256, n_item = items.size(), n_batch = (links.size() + batch - 1) / batch;
    std::vector<std::vector<size_t>> of_batch(n_batch); // The channel links of each batch
    for (size_t i = 0; i < n_item; ++i)
        of_batch[items[i].j[0] / batch].push_back(i);
    std::vector<channel<double>> out(n_item * P.n_freq);
    for (size_t b = 0; b < n_batch; ++b)
    {
        const size_t a = b * batch;
        const std::vector<arma::uword> sel(links.begin() + (std::ptrdiff_t)a, links.begin() + (std::ptrdiff_t)std::min(a + batch, links.size()));
        std::vector<qd_item> bi;
        for (size_t i : of_batch[b])
        {
            bi.push_back(items[i]);
            for (arma::uword &j : bi.back().j)
                j -= (arma::uword)a;
        }
        auto ch = qd_channels(P, P.rays(sel, scatterers), sel, bi, times, output, mobility, false, false);
        for (arma::uword k = 0; k < P.n_freq; ++k)
            for (size_t q = 0; q < bi.size(); ++q)
                out[k * n_item + of_batch[b][q]] = std::move(ch[k * bi.size() + q]);
    }
    qd_index(items, P.n_freq, index);
    return out;
}
