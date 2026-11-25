// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "qd_helpers_shared.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <map>
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

- Written by [[init_link_states]], read by the later stages; one entry per propagation link (one site, one terminal, one
  segment)
- Holds the discrete state of the links (LOS and indoor state, scenario, seeds), from which the later stages regenerate
  the large-scale parameters, phases and rays; not stored in the CSV files

## Attributes:<br>
| Attribute | Size | Description |
| --------- | ---- | ----------- |
| `arma::u32_vec i_tx`, `i_rx`, `i_segment` | `[n_link]` | Site, terminal and segment of each link, 0-based |
| `arma::u32_vec image` | `[n_link]` | Wrap image of the site, 0 = the real site |
| `arma::u32_vec i_conf` | `[n_link]` | Index into `conf_names`, from scenario, LOS and indoor state |
| `std::vector<std::string> conf_names` | `[n_conf]` | Resolved conf names |
| `arma::u32_vec los` | `[n_link]` | 1 = line of sight |
| `arma::s32_vec indoor` | `[n_link]` | Indoor state at the start of the segment, as in [[layout_nodes]] |
| `arma::u32_vec floor_id` | `[n_link]` | Floor number - 1, 0 outdoors, of the drawn state |
| `arma::vec d2d_in` | `[n_link]` | Indoor 2D distance in [m] of the drawn state |
| `arma::mat o2i_rnd` | `[n_terminal, 4]` | Random inputs of [[o2i_loss]] |
| `arma::u32_vec o2i_model` | `[n_terminal]` | Building O2I model, index into the list of the family; for every terminal of a family with O2I, also outdoors |
| `arma::s64_vec gbsm_seed` | `[n_link]` | Seed of the random phases; equal for one site, image and conf |
| `arma::sword seed` | scalar | Master seed, the base of the random values of all stages; -1 = not set |
| `std::string method`, `sos_preset` | | Method and SOS preset as used; an empty `method` means not filled |
| `double anchor_freq` | scalar | Anchor frequency of the multi-frequency model in [Hz], default 2e9 |
| `bool pol_variability` | scalar | Polarization power variability, default false |
| `std::vector<std::string> conf_files` | `[n_conf]` | Resolved paths of the conf files; empty = not resolved |
| `unsigned long long layout_hash` | scalar | Hash of the layout members that determine the links (see Notes) |

## Member functions:<br>
| Function | Description |
| -------- | ----------- |
| `.n_link()` | Number of links |
| `.empty()` | `true` if there are no links |
| `.reset()` | Empties all members and restores the defaults |
| `.is_valid()` | `std::string`: empty if the object is valid, else an error message; checks the sizes, the indices into `conf_names` and the ranges |

## Notes:
- **Layout hash**: covers the positions, names, sectors, frequencies, site seeds, scenarios, the O2I columns, clutter,
  scenario overrides, wrap vectors and whether a site moves. The later stages throw when it no longer matches the
  layout; `layout.links.reset()` discards the links. Orientations, antennas and `pairing` may change without new links
- **Forced states**: editing `los` or `indoor` before [[generate_rays]] forces that state; the conf and its seed follow
  from it. `floor_id` and `d2d_in` stay those of the drawn state; the later stages derive the values of the forced state
  from `indoor`
- **Restoring**: the links hold the master seed they were drawn with, so a copy of `layout.links` can be restored after
  another realisation

## See also:
- [[layout]] (holds the links as `links`)
- [[init_link_states]] (draws the links)
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

// Cheap checks of the sizes of all members of one side that the pipeline indexes, O(n_node + number of members)
bool qd_sizes_ok(const quadriga_lib::layout_nodes &N)
{
    const arma::uword nn = (arma::uword)N.name.size(), nr = N.pos.n_cols;
    const bool none = nn == 0 && N.pos.is_empty() && N.orientation.is_empty(); // a side without nodes, e.g. default-constructed
    if (N.sector.n_elem != nn || N.i_antenna.n_elem != nn || N.offset.n_elem != nn ||
        (!none && (N.pos.n_rows != 3 || N.orientation.n_rows != 3 || N.orientation.n_cols != nr || nn == 0)))
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
    std::vector<std::vector<arma::uword>> seg;    // Per terminal: the Rx rows at which its segments start
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
        arma::mat tx;   // Virtual position of the site at the start of the segments of the group, [3, 1]
    };
    std::vector<grp> groups;

    static constexpr size_t qd_par_rays = 4; // Size limit of the parallel ray generation (groups), see qd_par_channels

    qd_pipe(const quadriga_lib::layout &layout, const quadriga_lib::layout_links &links, bool check_hash, bool check_pairing = true) : L(layout), K(links)
    {
        if (K.method.empty())
            throw std::invalid_argument("Call 'init_link_states' first.");
        std::string msg = K.is_valid();
        if (!msg.empty())
            throw std::invalid_argument("Member 'links' of 'layout' is not valid: " + msg);
        qd_check_sizes(L);
        if (check_hash && qd_links_hash(L) != K.layout_hash)
            throw std::invalid_argument("The layout has changed since 'init_link_states': call 'init_link_states' again.");
        qd_check_method(K.method);
        qd = K.method == "quadriga", mf = K.method == "38.901_MF";
        n_freq = L.center_freq.n_elem;
        S = qd_make_sides(L);
        img = qd_images(L.wrap);
        if (check_pairing && !L.pairing.is_empty() && (L.pairing.n_rows != 2 || L.pairing.row(0).max() >= S.n_site || L.pairing.row(1).max() >= S.n_term))
            throw std::invalid_argument("Input 'layout' is not valid: member 'pairing' must have 2 rows with indices of sites and terminals.");
        const arma::uword n_link = K.n_link();
        if (K.o2i_rnd.n_rows != S.n_term || K.conf_files.size() != K.conf_names.size())
            throw std::invalid_argument("Member 'links' of 'layout' does not match the layout: call 'init_link_states' again.");
        seg.resize(S.n_term);
        for (size_t t = 0; t < S.n_term; ++t)
            seg[t] = qd_segment_starts(L, S, t);
        for (arma::uword l = 0; l < n_link; ++l)
            if (K.i_tx[l] >= S.n_site || K.i_rx[l] >= S.n_term || K.image[l] >= img.n_cols || K.i_segment[l] >= seg[K.i_rx[l]].size())
                throw std::invalid_argument("Member 'links' of 'layout' has a site, terminal, segment or image index outside the layout.");

        // Derived confs, seeds and O2I geometry
        qd_scenarios SC;
        scen.resize(n_link), conf.resize(n_link), seed.resize(n_link), row.resize(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            row[l] = seg[K.i_rx[l]][K.i_segment[l]];
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

        // Groups; the site position paired with the first row of the segment (a moving site: at the same time)
        std::map<std::tuple<arma::uword, arma::uword, arma::uword, arma::sword, double, double, double>, arma::uword> gi;
        group.resize(n_link);
        for (arma::uword l = 0; l < n_link; ++l)
        {
            const arma::vec p = qd_tx_at(L, S.site_nodes[K.i_tx[l]][0], S.term_nodes[K.i_rx[l]][0], row[l]);
            const auto key = std::make_tuple((arma::uword)K.i_tx[l], (arma::uword)K.image[l], conf[l], seed[l], p[0], p[1], p[2]);
            auto it = gi.find(key);
            if (it == gi.end())
            {
                it = gi.emplace(key, (arma::uword)groups.size()).first;
                grp g;
                g.site = K.i_tx[l], g.image = K.image[l], g.conf = conf[l], g.seed = seed[l];
                g.tx = arma::mat(p);
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
            // The deviation of a building UT follows its field to the moved positions; that of a car UT is drawn once per
            // UT (TR 38.901 eq. 7.4-4) at the first row of the terminal, where o2i_rnd was drawn
            arma::mat xy;
            std::vector<arma::uword> moved; // positions other than the first row of the terminal
            const arma::uword r0 = seg[t][0];
            for (arma::uword i = 1; i <= n && io != 2; ++i)
                if (B(0, i) != L.rx.pos(0, r0) || B(1, i) != L.rx.pos(1, r0))
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
// links, one per segment of the track, in segment order
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
    for (auto &kv : pairs)
    {
        std::sort(kv.second.begin(), kv.second.end(), [&](arma::uword a, arma::uword b) { return P.K.i_segment[sel[a]] < P.K.i_segment[sel[b]]; });
        for (arma::uword r : P.S.term_nodes[kv.first.second])
            for (arma::uword t : P.S.site_nodes[kv.first.first])
                items.push_back({t, r, kv.second});
    }
    std::sort(items.begin(), items.end(), [](const qd_item &a, const qd_item &b) { return a.rx != b.rx ? a.rx < b.rx : a.tx < b.tx; });
    return items;
}

// Checks the entries of 'rays': 1 or n_freq objects per link with the frequencies of the layout
void qd_check_rays(const quadriga_lib::layout &L, const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays)
{
    const arma::uword n_freq = L.center_freq.n_elem;
    for (size_t j = 0; j < rays.size(); ++j)
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
}

// Stage 3 for the channel links 'items' of the propagation links 'sel' with their rays; channels [n_freq * n_item],
// frequency slowest
std::vector<quadriga_lib::channel<double>> qd_channels(qd_pipe &P, const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays,
                                                       const std::vector<arma::uword> &sel, const std::vector<qd_item> &items, const arma::vec &times,
                                                       const std::string &output, const std::string &mobility, bool oxygen, bool use_absolute_delays,
                                                       double overlap)
{
    const quadriga_lib::layout &L = P.L;
    const quadriga_lib::layout_links &K = P.K;
    const arma::uword n_freq = P.n_freq;
    if (rays.size() != sel.size())
        throw std::invalid_argument("Input 'rays' must have one entry per link of 'i_link'.");
    qd_check_rays(L, rays);
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

    // Positions per propagation link over all snapshots, and the snapshots [own0, own1) of its segment
    const size_t n_sel = sel.size();
    std::vector<arma::mat> T(n_sel), R(n_sel), G(n_sel), KF(n_sel);
    std::vector<arma::uword> own0(n_sel, 0), own1(n_sel, 0), first(n_sel, 0);
    std::vector<char> used(n_sel, 0);
    for (const qd_item &it : items)
        for (arma::uword j : it.j)
            used[j] = 1;
    qd_batch_loop(n_sel, n_sel >= qd_par_links, nullptr, [&](size_t j)
                  {
                      if (!used[j])
                          return;
                      const arma::uword l = sel[j], t = K.i_rx[l], nt = P.S.site_nodes[K.i_tx[l]][0], nr = P.S.term_nodes[t][0];
                      arma::mat o;
                      arma::uvec i_row;
                      quadriga_lib::sample_track(L.tx, nt, times, T[j], o);
                      quadriga_lib::sample_track(L.rx, nr, times, R[j], o, &i_row);
                      if (T[j].n_cols != R[j].n_cols)
                      {
                          if (T[j].n_cols == 1)
                              T[j] = arma::repmat(T[j], 1, R[j].n_cols);
                          else if (R[j].n_cols == 1)
                              R[j] = arma::repmat(R[j], 1, T[j].n_cols), i_row = arma::repmat(i_row, R[j].n_cols, 1);
                          else
                              throw std::invalid_argument("Channel '" + qd_node_name(L.tx, nt) + "_" + qd_node_name(L.rx, nr) + "': the Tx and the Rx track have " +
                                                          std::to_string(T[j].n_cols) + " and " + std::to_string(R[j].n_cols) +
                                                          " rows; give 'times' or tracks with equal row counts.");
                      }
                      T[j].row(0) += P.img(0, K.image[l]), T[j].row(1) += P.img(1, K.image[l]);
                      const std::vector<arma::uword> &st = P.seg[t];
                      own1[j] = R[j].n_cols;
                      if (st.size() > 1) // the snapshots whose row is in the segment, in one run
                      {
                          const arma::uword k = K.i_segment[l], a = st[k], b = k + 1 < st.size() ? st[k + 1] : (arma::uword)-1;
                          own0[j] = own1[j] = 0;
                          bool found = false;
                          for (arma::uword s = 0; s < i_row.n_elem; ++s)
                          {
                              const bool in = i_row[s] >= a && i_row[s] < b;
                              if (in && found && own1[j] != s)
                                  throw std::invalid_argument("Terminal '" + L.rx.name[nr] + "': the snapshots of a segment are not consecutive; a track with several segments needs increasing 'times'.");
                              if (in)
                                  own0[j] = found ? own0[j] : s, own1[j] = s + 1, found = true;
                          }
                      }
                  });

    // The segments of each channel link with snapshots: a segment also covers the snapshots of the preceding one
    for (const qd_item &it : items)
    {
        long long prev = -1;
        for (arma::uword j : it.j)
        {
            if (own1[j] == own0[j])
                continue;
            first[j] = own0[j];
            if (prev >= 0)
            {
                const arma::uword p = (arma::uword)prev;
                if (own1[p] != own0[j]) // segments between them have no snapshots
                    throw std::invalid_argument("Channel '" + qd_node_name(L.tx, it.tx) + "_" + qd_node_name(L.rx, it.rx) +
                                                "': the selected segments of a track and their snapshots must be consecutive.");
                first[j] = own0[p];
            }
            prev = (long long)j;
        }
        if (prev < 0)
            throw std::invalid_argument("Channel '" + qd_node_name(L.tx, it.tx) + "_" + qd_node_name(L.rx, it.rx) + "': no snapshot is in its selected segments.");
    }

    // Link gain per propagation link over its snapshots
    std::vector<arma::uword> need;
    for (size_t j = 0; j < n_sel; ++j)
        if (used[j])
            need.push_back(sel[j]);
    P.prepare(need);
    for (arma::uword l : need)
        P.lsf(P.group[l]);
    qd_batch_loop(n_sel, n_sel >= qd_par_links, nullptr, [&](size_t j)
                  {
                      if (!used[j] || own1[j] == own0[j])
                          return;
                      const arma::uword l = sel[j];
                      if (first[j] != 0 || own1[j] != R[j].n_cols)
                          T[j] = T[j].cols(first[j], own1[j] - 1), R[j] = R[j].cols(first[j], own1[j] - 1);
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

    // Channels; a channel link with several segments: one channel per segment, joined by crossfade_segments. "phase":
    // each segment with the requested delays (its paths relative to its own LOS); "drift": absolute delays, then minus
    // the LOS delay at the snapshot positions (equal to the per-segment form, and one reference for empty tap slots)
    const size_t n_item = items.size(), n_out = n_item * n_freq;
    std::vector<quadriga_lib::channel<double>> out(n_out);
    qd_batch_loop(n_out, n_out >= qd_par_channels, nullptr, [&](size_t c)
                  {
                      const arma::uword k = (arma::uword)(c / n_item);
                      const qd_item &it = items[c % n_item];
                      const std::string name = qd_node_name(L.tx, it.tx) + "_" + qd_node_name(L.rx, it.rx);
                      qd_named("Channel '" + name + "': ", [&]
                               {
                                   arma::mat pos, otx, orx;
                                   quadriga_lib::sample_track(L.tx, it.tx, times, pos, otx);
                                   quadriga_lib::sample_track(L.rx, it.rx, times, pos, orx);
                                   std::vector<arma::uword> chain;
                                   for (arma::uword j : it.j)
                                       if (own1[j] > own0[j])
                                           chain.push_back(j);
                                   const bool merge = chain.size() > 1;
                                   std::vector<quadriga_lib::channel<double>> parts(chain.size());
                                   const quadriga_lib::gbsm_rays &ray0 = rays[chain[0]].size() == n_freq ? rays[chain[0]][k] : rays[chain[0]][0];
                                   const bool drift = mobility == "drift" || (mobility == "auto" && ray0.method == "quadriga");
                                   for (size_t q = 0; q < chain.size(); ++q)
                                   {
                                       const arma::uword j = chain[q], a = first[j], b = own1[j] - 1;
                                       const arma::mat ot = otx.n_cols == 1 ? otx : otx.cols(a, b), orr = orx.n_cols == 1 ? orx : orx.cols(a, b);
                                       const auto &r = rays[j];
                                       const quadriga_lib::gbsm_rays &ray = r.size() == n_freq ? r[k] : r[0];
                                       const arma::uword i_freq = r.size() == n_freq ? 0 : k;
                                       parts[q] = quadriga_lib::gbsm_single_link_channel(*atx[it.tx * n_freq + k], *arx[it.rx * n_freq + k], ray, G[j].col(k), mobility,
                                                                                         output, i_freq, T[j], R[j], ot, orr, KF[j].col(k), oxygen,
                                                                                         use_absolute_delays || (merge && drift));
                                   }
                                   if (!merge)
                                   {
                                       out[c] = std::move(parts[0]);
                                       out[c].name = name;
                                       return;
                                   }
                                   arma::uvec n_ov(chain.size(), arma::fill::zeros);
                                   for (size_t q = 1; q < chain.size(); ++q)
                                       n_ov[q] = own1[chain[q - 1]] - own0[chain[q - 1]];
                                   out[c] = quadriga_lib::crossfade_segments(parts, n_ov, overlap);
                                   out[c].name = name;
                                   if (drift && !use_absolute_delays)
                                       for (arma::uword i = 0; i < out[c].n_snap(); ++i)
                                           out[c].delay[i] -= arma::norm(out[c].rx_pos.col(i) - out[c].tx_pos.col(i)) / qd_speed_of_light;
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

// RSRP of Tx port p at snapshot s of a channel with one path per ray (output "rays"), linear: per Rx port the sum of
// |h|^2 over the paths, the mean over the Rx ports (TR 36.873 eq. (8.1-1)). A channel statistic; 0 without Rx ports
double qd_rsrp(const quadriga_lib::channel<double> &c, arma::uword p, arma::uword s = 0)
{
    const arma::cube &re = c.coeff_re[s], &im = c.coeff_im[s];
    double sum = 0.0;
    for (arma::uword i = 0; i < re.n_slices; ++i)
        for (arma::uword r = 0; r < re.n_rows; ++r)
        {
            const double a = re(r, p, i), b = im(r, p, i);
            sum += a * a + b * b;
        }
    return re.n_rows == 0 ? 0.0 : sum / (double)re.n_rows;
}

// Antenna gain of one unit-gain LOS path from the Tx position T to the Rx position R (get_channels_planar, Jones matrix
// diag(1, -1) of TR 38.901 eq. (7.5-29)): the mean over the Rx ports of |h|^2 of Tx port p, linear
double qd_los_gain(const quadriga_lib::arrayant<double> &ta, const quadriga_lib::arrayant<double> &ra, const arma::mat &T, const arma::mat &To,
                   const arma::mat &R, const arma::mat &Ro, double fc, arma::uword p)
{
    const double dx = R(0, 0) - T(0, 0), dy = R(1, 0) - T(1, 0), dz = R(2, 0) - T(2, 0), dh = std::hypot(dx, dy);
    const arma::vec aod = {std::atan2(dy, dx)}, eod = {std::atan2(dz, dh)}, aoa = {std::atan2(-dy, -dx)}, eoa = {-eod[0]};
    arma::mat M(8, 1, arma::fill::zeros);
    M(0, 0) = 1.0, M(6, 0) = -1.0;
    arma::cube re, im, dl;
    quadriga_lib::get_channels_planar<double>(ta, ra, T, To, R, Ro, aod, eod, aoa, eoa, arma::vec{1.0}, arma::vec{std::sqrt(dh * dh + dz * dz)}, M,
                                              re, im, dl, fc);
    double sum = 0.0;
    for (arma::uword r = 0; r < re.n_rows; ++r)
        sum += re(r, p, 0) * re(r, p, 0) + im(r, p, 0) * im(r, p, 0);
    return re.n_rows == 0 ? 0.0 : sum / (double)re.n_rows;
}

// Size limits of the parallel evaluation of node pairs in attach_terminals (one thread vs. 32, the loop follows serial work,
// so waking the threads costs about 0.3 ms): the LOS gain of a pair costs 4 us (generic antennas) to 23 us (32-element
// panel), 2x at 32 pairs for the panel; the RSRP of a pair about 0.4 ms (full TR ray sets), 2x at 4 pairs
constexpr size_t qd_par_attach_ls = 32;
constexpr size_t qd_par_attach_rsrp = 4;
// Positions [3, n] in the local coordinates of a QRT export: x and y minus the shift
arma::mat qd_local(arma::mat X, const arma::vec &shift)
{
    if (X.n_cols > 0)
        X.row(0) -= shift[0], X.row(1) -= shift[1];
    return X;
}

// The content of a QRT export besides the paths: origins, destinations with their CIRs, the paired links and the shift
struct qd_export
{
    arma::mat site_pos, site_ori;                       // Position and orientation (sector 0, else the first node) of each site, [3, n_site]
    std::vector<arma::mat> cir_pos, cir_ori;            // Rows of the first node of each terminal, [n_term] of [3, n_row]
    arma::u32_vec cir_offset;                           // First CIR of each terminal, [n_term]
    std::vector<std::string> dest_names;                // Terminal names, [n_term]
    std::vector<std::vector<arma::uword>> site_links;   // Paired links of each site, terminal-major, [n_site]
    arma::vec shift;                                    // Centroid of the site and CIR positions in x and y, [2]
    arma::uword n_cir = 0;
};

// Checks of export_rays_qrt that need the pipeline, and the content of the file; throws before anything is written
qd_export qd_export_setup(const qd_pipe &P, const std::vector<std::vector<quadriga_lib::gbsm_rays>> &rays)
{
    const quadriga_lib::layout &L = P.L;
    const quadriga_lib::layout_links &K = P.K;
    const quadriga_lib::layout_nodes &TX = L.tx, &RX = L.rx;
    const qd_sides &S = P.S;
    const arma::uword n_link = K.n_link();
    const size_t n_site = S.n_site, n_term = S.n_term;
    if (!P.qd && P.n_freq != 1)
        throw std::invalid_argument("The method '" + K.method + "' exports one frequency, but 'center_freq' of the layout has " + std::to_string(P.n_freq) +
                                    "; use one layout per frequency, or 'quadriga'.");
    if (P.n_freq > 127)
        throw std::invalid_argument("A QRT file holds at most 127 frequencies, but 'center_freq' of the layout has " + std::to_string(P.n_freq) + ".");
    if (!L.wrap.is_empty())
        throw std::invalid_argument("Input 'layout' has wrap-around, which a QRT file cannot hold (an origin has one position, the wrap images of a site "
                                    "differ per terminal); clear 'wrap' and call 'init_link_states' again.");
    if (n_site == 0)
        throw std::invalid_argument("Input 'layout' has no Tx nodes.");
    if (n_term == 0)
        throw std::invalid_argument("Input 'layout' has no Rx nodes.");
    for (arma::uword i = 0; i < TX.n_node(); ++i)
        if (qd_site_moves(TX, i))
            throw std::invalid_argument("Tx node '" + TX.name[i] + "' (sector " + std::to_string(TX.sector[i]) +
                                        "): a QRT origin has one position; it must have one row and no speed.");
    std::vector<char> paired(n_site * n_term, L.pairing.is_empty() ? 1 : 0);
    for (arma::uword p = 0; p < L.pairing.n_cols; ++p)
        paired[L.pairing(0, p) * n_term + L.pairing(1, p)] = 1;
    std::vector<std::vector<arma::uword>> link(n_site * n_term); // The propagation links of each pair, one per segment
    for (arma::uword l = 0; l < n_link; ++l)
        link[K.i_tx[l] * n_term + K.i_rx[l]].push_back(l);

    qd_export E;
    E.site_pos.set_size(3, n_site), E.site_ori.set_size(3, n_site);
    for (size_t s = 0; s < n_site; ++s)
    {
        arma::uword i = S.site_nodes[s][0];
        for (arma::uword j : S.site_nodes[s])
            if (TX.sector[j] == 0)
            {
                i = j;
                break;
            }
        arma::mat pos, o;
        quadriga_lib::sample_track(TX, i, {}, pos, o);
        E.site_pos.col(s) = pos.col(0), E.site_ori.col(s) = o.col(0);
    }
    E.cir_pos.resize(n_term), E.cir_ori.resize(n_term), E.cir_offset.set_size(n_term), E.dest_names.resize(n_term), E.site_links.resize(n_site);
    for (size_t t = 0; t < n_term; ++t)
    {
        const arma::uword i = S.term_nodes[t][0];
        quadriga_lib::sample_track(RX, i, {}, E.cir_pos[t], E.cir_ori[t]);
        E.cir_offset[t] = (arma::u32)E.n_cir, E.dest_names[t] = RX.name[i];
        E.n_cir += E.cir_pos[t].n_cols;
        bool any = false;
        for (size_t s = 0; s < n_site; ++s)
        {
            if (!paired[s * n_term + t])
                continue;
            any = true;
            const std::vector<arma::uword> &ls = link[s * n_term + t];
            bool at_site = ls.empty(); // co-located at the first row: no propagation link
            for (arma::uword r = 0; r < E.cir_pos[t].n_cols && !at_site; ++r)
                at_site = arma::norm(E.cir_pos[t].col(r) - E.site_pos.col(s)) <= 1.0e-4;
            if (at_site)
                throw std::invalid_argument("Terminal '" + RX.name[i] + "' has a row at the position of site '" + TX.name[S.site_nodes[s][0]] +
                                            "'; a CIR needs a LOS path of non-zero length.");
            E.site_links[s].insert(E.site_links[s].end(), ls.begin(), ls.end());
        }
        if (!any)
            throw std::invalid_argument("Terminal '" + RX.name[i] + "' has no site in 'pairing': each QRT destination needs the CIRs of at least one origin.");
    }
    if (!rays.empty())
    {
        if (rays.size() != n_link)
            throw std::invalid_argument("Input 'rays' must be empty or hold the rays of all links of the layout (from 'generate_rays').");
        for (size_t j = 0; j < rays.size(); ++j)
            if (rays[j].size() != 1)
                throw std::invalid_argument("Entry " + std::to_string(j) + ": Input 'rays' must hold one object with all frequencies per link.");
        qd_check_rays(L, rays);
    }
    E.shift = arma::sum(E.site_pos.rows(0, 1), 1);
    for (const arma::mat &X : E.cir_pos)
        E.shift += arma::sum(X.rows(0, 1), 1);
    E.shift /= (double)(n_site + E.n_cir);
    return E;
}

// The paths of the CIRs of one link for qrt_file_append: per row of its segment the link gain and K-factor ratio of
// stage 3, ssf_paths ("drift") in the coordinates of the layout, qrt_paths_from_scatterers in local coordinates
std::vector<quadriga_lib::path> qd_export_link(qd_pipe &P, const qd_export &E, arma::uword l, const quadriga_lib::gbsm_rays &rays)
{
    const arma::uword t = P.K.i_rx[l], k = P.K.i_segment[l], off = P.L.rx.offset[P.S.term_nodes[t][0]];
    const std::vector<arma::uword> &st = P.seg[t];
    const arma::uword r0 = st[k] - off, r1 = k + 1 < st.size() ? st[k + 1] - off : E.cir_pos[t].n_cols; // rows of the segment
    const quadriga_lib::gbsm_rays *r = &rays;
    quadriga_lib::gbsm_rays with_scatterers;
    if (r->fbs_pos.is_empty())
    {
        with_scatterers = *r;
        quadriga_lib::ssf_scatterers(with_scatterers);
        r = &with_scatterers;
    }
    const arma::mat R = E.cir_pos[t].cols(r0, r1 - 1), &T = P.groups[P.group[l]].tx;
    arma::mat g, kfr, pg, fbs, lbs;
    P.link_gain(l, arma::repmat(T, 1, R.n_cols), R, g, kfr);
    const arma::mat tx = qd_local(T, E.shift);
    std::vector<quadriga_lib::path> out;
    for (arma::uword i = 0; i < R.n_cols; ++i)
    {
        arma::vec len;
        arma::cube pm;
        arma::u32_vec type;
        quadriga_lib::ssf_paths(*r, g.row(i).t(), pg, len, pm, "drift", T, R.col(i), kfr.row(i).t(), nullptr, nullptr, nullptr, nullptr, &fbs, &lbs, &type);
        std::vector<quadriga_lib::path> q = quadriga_lib::qrt_paths_from_scatterers(P.L.center_freq, tx, qd_local(R.col(i), E.shift), qd_local(fbs, E.shift),
                                                                                     qd_local(lbs, E.shift), pg, pm, type, E.cir_offset[t] + (unsigned)(r0 + i));
        out.insert(out.end(), std::make_move_iterator(q.begin()), std::make_move_iterator(q.end()));
    }
    return out;
}

// Size limit of the parallel loop over the links of a site in export_rays_qrt (one thread vs. 32): a link costs about
// 0.5 ms plus 0.25 ms per CIR. Passive wait policy: 1.6x at 3 one-row links, 2x at 4, 3.3x at 8; default policy: the
// wake-up of the threads after serial work varied between runs from below 0.1 ms to 9 ms, unstable below 8 links and
// 1.5x to 2x at 8; 8 is safe with both policies
constexpr size_t qd_par_export = 8;
} // namespace


/*!MD
# generate_rays
Rays of the propagation links of a layout (stage 2 of the channel pipeline)

- Runs the chain of the section "Geometry-based stochastic model" for the chosen links, from the link states of
  [[init_link_states]]: large-scale parameters, clusters, rays and, on request, the scatterers
- The rays have no antennas; [[get_gbsm_channels]] applies them. Edit the rays in between to study a change of the
  propagation, or pass them to [[attach_terminals]] and [[export_rays_qrt]]
- The rays of a link do not depend on the other requested links

## Declaration:
```
std::vector<std::vector<gbsm_rays>> quadriga_lib::generate_rays(
    const layout &layout,
    const uint_list &i_link = {},
    bool scatterers = false);
```

## Inputs:
- **`layout`** — Layout with links from [[init_link_states]]; throws if a member that determines the links changed since
  then, or if the sizes of its members do not match
- **`i_link`** *(optional)* — Propagation links, 0-based, without repetition; omitted = all (none for links without
  entries); `[n_out]`
- **`scatterers`** *(optional)* — Solve the first- and last-bounce scatterers, needed for the mobility `"drift"`

## Returns:
- Rays per link and frequency, `[n_out][n_freq]` with one frequency each for the TR methods, `[n_out][1]` with all
  frequencies for `"quadriga"`

## Notes:
- **Chain**: [[gbsm_init_phases]] and [[gbsm_draw]] of the random sets, [[path_loss]] (for `rnd_he` and the SF standard
  deviation), [[lsp_generate]], [[ssf_clusters]], [[ssf_rays]] and, with `scatterers`, [[ssf_scatterers]], once for the
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
Channel coefficients of a layout from given rays (stage 3 of the channel pipeline)

- Applies the antennas of the nodes, the motion along their tracks and the link gain to the rays of [[generate_rays]],
  and returns one channel per Tx node, Rx node and frequency
- Only the channel links of the given propagation links and of the pairs in `layout.pairing` (empty: all) are computed;
  the sectors of a site share the rays of their propagation link
- Tracks with several segments are joined by [[crossfade_segments]]

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
- **`layout`** — Layout with links from [[init_link_states]]; throws if a member that determines the links changed since
  then
- **`rays`** — Rays of the links from [[generate_rays]], possibly edited, with the frequencies of the layout;
  `[n_out][1 or n_freq]`
- **`i_link`** *(optional)* — The propagation links of `rays`, 0-based; omitted = all links; `[n_out]`
- **`times`** *(optional)* — Snapshot times in [s], finite, the same for all nodes; omitted = one snapshot per row;
  `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`, see [[gbsm_single_link_channel]]
- **`mobility`** *(optional)* — `"phase"`, `"drift"` (needs rays with scatterers) or `"auto"` (phase for the TR methods,
  drift for `"quadriga"`)
- **`oxygen`** *(optional)* — Oxygen absorption of TR 38.901 clause 7.6.1
- **`use_absolute_delays`** *(optional)* — Include the LOS delay in all delays
- **`overlap`** *(optional)* — Part of the snapshots shared by consecutive segments that the cross-fade uses, in [0, 1],
  see [[crossfade_segments]]

## Output:
- **`index`** *(optional)* — Tx node, Rx node and frequency index of each channel, 0-based; `[3, n_channel]`

## Returns:
- Channels, see [[gbsm_single_link_channel]]; `[n_channel]`

## Notes:
- **Order and names**: Tx node fastest, then Rx node, then frequency; name `<tx>:<sector>_<rx>:<sector>`, without the
  sector where all Tx nodes (or all Rx nodes) have sector 0
- **Snapshots** of both ends from [[sample_track]] over `times`; a node with one snapshot is repeated to the count of
  the other; two tracks without `times` need equal row counts
- **Link gain** per snapshot, $G = -\mathrm{PL} - L_\mathrm{O2I} + 10 \log_{10} \mathrm{SF}$ in dB: the path loss PL of
  [[path_loss]] with the random environment height `rnd_he` held at the start of the segment, the shadow fading SF and
  the K-factor ratio of [[lsp_generate]] from [[gbsm_draw]] of the `"lsf"` set at the snapshot, and the O2I loss of
  [[o2i_loss]] for building and car UTs with the model of the link state (the deviation of a building UT at the
  snapshot, that of a car UT at the first row of the terminal for its whole track, TR 38.901 eq. 7.4-4). A link whose
  ends do not move is evaluated once
- **Antennas**: [[gbsm_single_link_channel]] with the antenna of each node at the frequency (its band `i_freq` in
  `layout.antenna`, or its only entry for all bands) and the Tx positions moved by the wrap image of the link. Nodes
  without an antenna and antennas that are not evaluated use the generic antenna; an antenna spaced in wavelengths that
  is not evaluated throws
- **Frequencies**: rays with `n_freq` entries are used as entry `i_freq` (TR methods), rays with one entry at frequency
  `i_freq` (`"quadriga"`)
- **Segments**: the snapshots of a segment are those whose row (see [[sample_track]]) is in it; each segment also covers
  the snapshots of the preceding one, with its link gain, rays and K-factor ratio referred to its first row. The
  selected segments of a channel link must be consecutive (segments without snapshots are skipped), with increasing
  `times`
- **Relative delays**: in `"phase"` the paths of each segment relative to its own LOS, in `"drift"` relative to the LOS
  at the snapshot

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
    std::vector<channel<double>> out = qd_channels(P, rays, sel, items, times, output, mobility, oxygen, use_absolute_delays, overlap);
    qd_index(items, P.n_freq, index);
    return out;
}

/*!MD
# get_channels_layout
Channel coefficients of a layout in one call

- Computes one channel per Tx node, Rx node and frequency, for the pairs in `layout.pairing` (empty: all pairs)
- Runs the stages [[init_link_states]], [[generate_rays]] and [[get_gbsm_channels]] with their default settings; the
  results are those of the stages called one by one
- Without link states in `layout.links`, draws them for this call only and leaves the layout unchanged; with link
  states, uses them, also when they were edited

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
- **`layout`** — Layout with `center_freq`; antennas as evaluated in `layout.antenna`, nodes without one use the generic
  antenna (see [[get_gbsm_channels]])
- **`times`** *(optional)* — Snapshot times in [s]; omitted = one snapshot per row; `[n_snap]`
- **`output`** *(optional)* — `"taps"` or `"rays"`
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` or `"quadriga"`; omitted = the method of filled links, else
  `"38.901"`; another method than that of filled links throws
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
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1, {3.5e9});
auto channels = quadriga_lib::get_channels_layout(L);
```

## Notes:
- **Memory**: the links are processed in batches of at most 256 propagation links (the segments of a track in one
  batch), so that the rays of one batch bound the memory; the results do not depend on the batches
- **Link states**: `layout.links` holds none when `layout.links.method` is empty, e.g. after `layout.links.reset()`.
  With link states, `seed` and `config_dirs` are not used. A layout without linked pairs gives an empty result
- **Scatterers**: the mobility `"drift"` (also `"auto"` with `"quadriga"`) solves the scatterers

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

    // The paired links ordered by the group of the first segment of their pair, the segments of a pair together, in
    // batches of up to 256 links that do not split a pair; the random sets are drawn once per group and batch, the SOS
    // phases of the "lsf" sets stay cached in P, and the values of a link do not depend on its batch
    std::vector<arma::uword> links;
    {
        const size_t n_pair = P.S.n_site * P.S.n_term;
        std::vector<char> pair(n_pair, layout.pairing.is_empty() ? 1 : 0);
        for (arma::uword p = 0; p < layout.pairing.n_cols; ++p)
            pair[layout.pairing(0, p) * P.S.n_term + layout.pairing(1, p)] = 1;
        std::vector<arma::uword> head(n_pair, P.K.n_link()); // the first link of each pair
        for (arma::uword l = 0; l < P.K.n_link(); ++l)
        {
            const size_t q = P.K.i_tx[l] * P.S.n_term + P.K.i_rx[l];
            if (pair[q])
                links.push_back(l), head[q] = std::min(head[q], l);
        }
        auto key = [&](arma::uword l)
        {
            const arma::uword h = head[P.K.i_tx[l] * P.S.n_term + P.K.i_rx[l]];
            return std::make_tuple(P.group[h], h, P.K.i_segment[l]);
        };
        std::sort(links.begin(), links.end(), [&](arma::uword x, arma::uword y) { return key(x) < key(y); });
    }
    const std::vector<qd_item> items = qd_items(P, links); // All channel links, canonical; j indexes 'links'
    const size_t batch = 256, n_item = items.size();
    std::vector<size_t> start = {0}; // First position in 'links' of each batch
    for (size_t a = 0; a < links.size();)
    {
        size_t e = a + 1;
        while (e < links.size() && P.K.i_segment[links[e]] != 0) // the rest of the pair
            ++e;
        if (e - start.back() > batch && a > start.back())
            start.push_back(a);
        a = e;
    }
    start.push_back(links.size());
    const size_t n_batch = start.size() - 1;
    std::vector<std::vector<size_t>> of_batch(n_batch); // The channel links of each batch
    for (size_t i = 0; i < n_item; ++i)
        of_batch[(size_t)(std::upper_bound(start.begin(), start.end(), (size_t)items[i].j[0]) - start.begin()) - 1].push_back(i);
    std::vector<channel<double>> out(n_item * P.n_freq);
    for (size_t b = 0; b < n_batch; ++b)
    {
        const size_t a = start[b];
        const std::vector<arma::uword> sel(links.begin() + (std::ptrdiff_t)a, links.begin() + (std::ptrdiff_t)start[b + 1]);
        std::vector<qd_item> bi;
        for (size_t i : of_batch[b])
        {
            bi.push_back(items[i]);
            for (arma::uword &j : bi.back().j)
                j -= (arma::uword)a;
        }
        auto ch = qd_channels(P, P.rays(sel, scatterers), sel, bi, times, output, mobility, false, false, overlap);
        for (arma::uword k = 0; k < P.n_freq; ++k)
            for (size_t q = 0; q < bi.size(); ++q)
                out[k * n_item + of_batch[b][q]] = std::move(ch[k * bi.size() + q]);
    }
    qd_index(items, P.n_freq, index);
    return out;
}

/*!MD
# attach_terminals
Serving Tx node of each terminal of a layout by the large-scale gain or by RSRP

- Evaluates every Tx node for each terminal and returns the one with the highest coupling gain (`"large_scale"`) or the
  highest RSRP (`"rsrp"`), with its site and the gains of all nodes
- Use it to link each terminal to its serving site in `layout.pairing` (see Example), so that the channel functions
  compute only the serving links
- One call attaches at one frequency, `layout.center_freq[i_freq]`, the first band by default; this suits offloading
  scenarios, where the higher bands serve as data pipes. The result is one serving Tx node per terminal

## Declaration:
```
arma::uvec quadriga_lib::attach_terminals(
    const layout &layout,
    const std::string &rule = "large_scale",
    const std::vector<std::vector<gbsm_rays>> &rays = {},
    arma::uvec *site = nullptr,
    arma::mat *gain_dB = nullptr,
    arma::uword i_freq = 0,
    arma::uword tx_port = 0);
```

## Inputs:
- **`layout`** — Layout with links from [[init_link_states]]; Tx nodes required if there are terminals
- **`rule`** *(optional)* — `"large_scale"` or `"rsrp"`
- **`rays`** *(optional)* — `"rsrp"`: rays of all links from [[generate_rays]], possibly edited; `[n_link][1 or n_freq]`
- **`i_freq`** *(optional)* — Frequency index, 0-based
- **`tx_port`** *(optional)* — Port of the Tx antennas, 0-based

## Outputs:
- **`site`** *(optional)* — Serving site per terminal, 0-based; `[n_rx_terminal]`
- **`gain_dB`** *(optional)* — Coupling gain or RSRP in [dB]; `[n_rx_terminal, n_tx_node]`

## Returns:
- Serving Tx node per terminal, 0-based; `[n_rx_terminal]`

## Example:
```
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 1, {3.5e9});
quadriga_lib::init_link_states(L);
arma::uvec site;
quadriga_lib::attach_terminals(L, "large_scale", {}, &site);
L.pairing = arma::join_cols(site.t(), arma::regspace<arma::urowvec>(0, site.n_elem - 1));
```

## Notes:
- **Evaluation**: at the first rows of both nodes and `layout.center_freq[i_freq]`, with the link states of the first
  segment, the wrap image and the antennas as in [[get_gbsm_channels]]; `layout.pairing` is ignored
- **`"large_scale"`** (TR 38.901 Table 7.8-1): the link gain of [[get_gbsm_channels]] plus the gain of one unit-gain LOS
  path through both antennas (port weights, element positions, orientations), as [[get_channels_planar]] gives it; no
  rays
- **`"rsrp"`** (TR 36.873 eq. (8.1-1)): per Rx port the sum of $\lvert h \rvert^2$ over the rays of
  [[gbsm_single_link_channel]] (`output = "rays"`, mobility `"phase"`), link gain included, transmit power 1
- **Ports**: both rules take $\lvert h \rvert^2$ of Tx port `tx_port` and average it over the Rx ports, so a
  dual-polarized UT (such as the generic antenna) reads 3 dB below one co-polarized element; a calibration sets the UT
  antenna of the TR
- **Choice**: a terminal takes the highest value of its nodes; ties go to the lowest index. A Tx node without a
  propagation link (co-located) has -Inf; a terminal without any link gets node 0
- **Pairing**: `layout.pairing` holds pairs of site and terminal, so the Example links all sectors of the serving site;
  for multi-connectivity, `layout.pairing` may list several sites per terminal, e.g. all sites within a margin of the
  best value in `gain_dB`, and the channel functions then compute the links to all of them

## See also:
- [[init_link_states]], [[generate_rays]] (link states and rays)
- [[get_gbsm_channels]] (channels of the linked pairs)
MD!*/

arma::uvec quadriga_lib::attach_terminals(const layout &layout, const std::string &rule, const std::vector<std::vector<gbsm_rays>> &rays,
                                          arma::uvec *site, arma::mat *gain_dB, arma::uword i_freq, arma::uword tx_port)
{
    const bool rsrp = rule == "rsrp";
    if (!rsrp && rule != "large_scale")
        throw std::invalid_argument("Input 'rule' must be 'large_scale' or 'rsrp'.");
    qd_pipe P(layout, layout.links, true, false); // pairing is not used
    const layout_links &K = layout.links;
    const arma::uword n_freq = P.n_freq, n_link = K.n_link(), n_tx = layout.tx.n_node(), n_rx = layout.rx.n_node();
    const size_t n_term = P.S.n_term, n_site = P.S.n_site;
    if (i_freq >= n_freq)
        throw std::invalid_argument("Input 'i_freq' must select one of the " + std::to_string(n_freq) + " frequencies of the layout.");
    if (rsrp)
    {
        if (rays.size() != n_link)
            throw std::invalid_argument("Input 'rays' must hold the rays of all links of the layout (from 'generate_rays') for the rule 'rsrp'.");
        qd_check_rays(layout, rays);
    }
    if (n_term == 0) // no terminals: empty outputs
    {
        if (site != nullptr)
            site->set_size(0);
        if (gain_dB != nullptr)
            gain_dB->set_size(0, n_tx);
        return arma::uvec();
    }
    if (n_tx == 0)
        throw std::invalid_argument("Input 'layout' has no Tx nodes.");

    // The arrays at frequency i_freq (serial: an antenna that is not evaluated may throw), the port of the Tx arrays
    std::vector<arrayant<double>> xpol(n_freq);
    xpol[i_freq] = generate_antenna<double>("xpol", NAN, layout.center_freq[i_freq]);
    std::vector<const arrayant<double> *> atx(n_tx), arx(n_rx);
    for (arma::uword i = 0; i < n_tx; ++i)
    {
        atx[i] = qd_array(layout, layout.tx, "Tx", i, i_freq, xpol);
        if (tx_port >= atx[i]->n_ports())
            throw std::invalid_argument("Input 'tx_port' must select one of the " + std::to_string(atx[i]->n_ports()) + " ports of the antenna of Tx node '" +
                                        layout.tx.name[i] + "' (sector " + std::to_string(layout.tx.sector[i]) + ").");
    }
    for (arma::uword i = 0; i < n_rx; ++i)
        arx[i] = qd_array(layout, layout.rx, "Rx", i, i_freq, xpol);

    // Orientations of the nodes at their first row (in the GCS, as sample_track gives them)
    arma::mat otx(3, n_tx), orx(3, n_rx);
    for (int side = 0; side < 2; ++side)
    {
        const layout_nodes &N = side == 0 ? layout.tx : layout.rx;
        arma::mat &O = side == 0 ? otx : orx;
        for (arma::uword i = 0; i < N.n_node(); ++i)
        {
            arma::mat pos, o;
            sample_track(N, i, {}, pos, o);
            O.col(i) = o.col(0);
        }
    }

    // Link gain of the first segment of each pair at the drop
    std::vector<arma::uword> all, link(n_site * n_term, n_link);
    for (arma::uword l = 0; l < n_link; ++l)
        if (K.i_segment[l] == 0)
            all.push_back(l), link[K.i_tx[l] * n_term + K.i_rx[l]] = l;
    P.prepare(all);
    for (arma::uword l : all)
        P.lsf(P.group[l]);
    arma::vec lg(n_link);
    qd_batch_loop(all.size(), all.size() >= qd_par_links, nullptr, [&](size_t i)
                  {
                      const arma::uword l = all[i];
                      arma::mat g, kfr;
                      P.link_gain(l, P.groups[P.group[l]].tx, P.rx0(l), g, kfr);
                      lg[l] = g(0, i_freq);
                  });

    // Value of each pair of a Tx node and an Rx node with a link, [n_rx, n_tx]
    arma::mat V(n_rx, n_tx);
    const size_t n_pair = (size_t)(n_rx * n_tx);
    qd_batch_loop(n_pair, n_pair >= (rsrp ? qd_par_attach_rsrp : qd_par_attach_ls), nullptr, [&](size_t c)
                  {
                      const arma::uword r = (arma::uword)(c % n_rx), i = (arma::uword)(c / n_rx);
                      const arma::uword l = link[P.S.site_of_tx[i] * n_term + P.S.term_of_rx[r]];
                      if (l == n_link)
                      {
                          V(r, i) = -arma::datum::inf;
                          return;
                      }
                      const arma::mat T = P.groups[P.group[l]].tx, R = P.rx0(l);
                      qd_named("Channel '" + qd_node_name(layout.tx, i) + "_" + qd_node_name(layout.rx, r) + "': ", [&]
                               {
                                   if (!rsrp)
                                       V(r, i) = lg[l] + 10.0 * std::log10(qd_los_gain(*atx[i], *arx[r], T, otx.col(i), R, orx.col(r), layout.center_freq[i_freq], tx_port));
                                   else
                                   {
                                       const auto &y = rays[l];
                                       const bool per_freq = y.size() == n_freq; // as in stage 3
                                       const gbsm_rays &ray = per_freq ? y[i_freq] : y[0];
                                       const channel<double> ch = gbsm_single_link_channel(*atx[i], *arx[r], ray, arma::vec{lg[l]}, "phase", "rays",
                                                                                           per_freq ? 0 : i_freq, T, R, otx.col(i), orx.col(r));
                                       V(r, i) = 10.0 * std::log10(qd_rsrp(ch, tx_port));
                                   }
                               });
                  });

    // The highest value of the nodes of each terminal; the serving node, ties to the lowest index
    arma::mat G(n_term, n_tx);
    G.fill(-arma::datum::inf);
    for (arma::uword r = 0; r < n_rx; ++r)
        for (arma::uword i = 0; i < n_tx; ++i)
            if (V(r, i) > G(P.S.term_of_rx[r], i))
                G(P.S.term_of_rx[r], i) = V(r, i);
    arma::uvec serving(n_term, arma::fill::zeros);
    for (size_t t = 0; t < n_term; ++t)
        for (arma::uword i = 1; i < n_tx; ++i)
            if (G(t, i) > G(t, serving[t]))
                serving[t] = i;
    if (site != nullptr)
    {
        site->set_size(n_term);
        for (size_t t = 0; t < n_term; ++t)
            (*site)[t] = (arma::uword)P.S.site_of_tx[serving[t]];
    }
    if (gain_dB != nullptr)
    {
        gain_dB->set_size(n_term, n_tx);
        std::copy(G.begin(), G.end(), gain_dB->begin());
    }
    return serving;
}

/*!MD
# export_rays_qrt
Write the rays of a layout with their link gain to a QRT file

- Stores the rays of the linked pairs with their link gain as CIRs: one origin per site, one destination per terminal
  and one CIR per row of the terminal
- Separates the ray generation from the antennas: [[qrt_file_read]], [[gbsm_rays_from_paths]] and
  [[gbsm_single_link_channel]] turn a CIR into channels for any arrays, orientations and positions, without the GBSM
  chain
- Positions in the file are local; add the returned shift to the positions read back

## Declaration:
```
arma::vec quadriga_lib::export_rays_qrt(
    const std::string &fn,
    const layout &layout,
    const std::vector<std::vector<gbsm_rays>> &rays = {});
```

## Inputs:
- **`fn`** — QRT file, replaced if it exists
- **`layout`** — Layout with links from [[init_link_states]]: one frequency for the TR methods (`"quadriga"`: 1 to 127),
  Tx nodes with one row and no speed, no wrap-around, each terminal paired with a site and no row of a terminal at a
  site position
- **`rays`** *(optional)* — Rays of all links from [[generate_rays]], possibly edited, one object with all frequencies
  per link; omitted = generated for batches of sites with up to 256 links, which bounds the memory; `[n_link][1]`

## Returns:
- Shift of the local coordinates in x and y in [m]; `[2]`

## Notes:
- **Origins and destinations**: an origin at each site, with the orientation of sector 0 (else of the first node of the
  site) and the site name; a destination per terminal with its name and one CIR per row, with the orientation of the
  first node of the terminal (its nodes share their positions)
- **Content of a CIR**: the rays of the link and of the segment that holds the row (scatterers added where missing), the
  link gain and the K-factor ratio of [[get_gbsm_channels]] at the row, [[ssf_paths]] with `"drift"`,
  [[qrt_paths_from_scatterers]]. A CIR read back gives the channel of [[get_gbsm_channels]] (`output = "rays"`,
  `mobility = "drift"`) at the row within float32 accuracy
- **Reading back**: [[gbsm_rays_from_paths]] with the method of the links and the `n_interact` of the file as
  `path_type`, then [[gbsm_single_link_channel]] with link gain 0 and mobility `"drift"`
- **Point drops with speed**: one CIR; the motion is applied at the evaluation, at the link gain of the drop
- **Pairs**: only the pairs of `layout.pairing` (empty: all) are written; the other CIRs of an origin hold no paths
- **Shift**: the centroid of the site and CIR positions in x and y is subtracted; z is unchanged
- **Limits of the format**: no tap index and no ground permittivity, so a CIR gives rays, not the taps of the TR, and
  moving a ground-reflection ray from a file throws
- The file is written to `fn + ".part"` and renamed at the end; a failed call leaves an existing file unchanged

## See also:
- [[generate_rays]], [[get_gbsm_channels]] (the rays and the link gain)
- [[qrt_file_read]], [[gbsm_rays_from_paths]], [[gbsm_single_link_channel]] (evaluation of the file)
- [[qrt_paths_from_scatterers]] (the stored paths)
MD!*/

arma::vec quadriga_lib::export_rays_qrt(const std::string &fn, const layout &layout, const std::vector<std::vector<gbsm_rays>> &rays)
{
    if (fn.empty())
        throw std::invalid_argument("Input 'fn' must not be empty.");
    std::error_code ec;
    if (std::filesystem::is_directory(fn, ec))
        throw std::invalid_argument("Input 'fn' is a folder; give the name of a file.");
    qd_pipe P(layout, layout.links, true);
    const qd_export E = qd_export_setup(P, rays);
    arma::mat cp(E.n_cir, 3), co(E.n_cir, 3);
    for (size_t t = 0; t < E.cir_pos.size(); ++t)
        if (E.cir_pos[t].n_cols > 0)
        {
            cp.rows(E.cir_offset[t], E.cir_offset[t] + E.cir_pos[t].n_cols - 1) = qd_local(E.cir_pos[t], E.shift).t();
            co.rows(E.cir_offset[t], E.cir_offset[t] + E.cir_pos[t].n_cols - 1) = E.cir_ori[t].t();
        }

    // One origin per site, written into a temporary file that replaces fn at the end. Without given rays, the rays are
    // generated for batches of consecutive sites with up to 256 links (at least one site), which bounds the memory
    const std::string tmp = fn + ".part";
    const size_t n_site = P.S.n_site;
    try
    {
        qrt_file_init<double>(tmp, layout.center_freq, cp, co, E.dest_names, E.cir_offset, (unsigned)n_site);
        for (size_t s0 = 0, s1 = 0; s0 < n_site; s0 = s1)
        {
            std::vector<arma::uword> batch;
            for (; s1 < n_site && (s1 == s0 || batch.size() + E.site_links[s1].size() <= 256); ++s1)
                batch.insert(batch.end(), E.site_links[s1].begin(), E.site_links[s1].end());
            const std::vector<std::vector<gbsm_rays>> own = rays.empty() ? P.rays(batch, true) : std::vector<std::vector<gbsm_rays>>();
            P.prepare(batch);
            for (arma::uword l : batch)
                P.lsf(P.group[l]);
            for (size_t s = s0, first = 0; s < s1; first += E.site_links[s].size(), ++s)
            {
                const std::vector<arma::uword> &ls = E.site_links[s];
                std::vector<std::vector<path>> paths(ls.size());
                qd_batch_loop(ls.size(), ls.size() >= qd_par_export, nullptr, [&](size_t j)
                              {
                                  const arma::uword l = ls[j];
                                  qd_named("Terminal '" + E.dest_names[layout.links.i_rx[l]] + "': ", [&]
                                           { paths[j] = qd_export_link(P, E, l, rays.empty() ? own[first + j][0] : rays[l][0]); });
                              });
                std::vector<path> all;
                for (auto &v : paths)
                    all.insert(all.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
                qrt_file_append<double>(tmp, all, qd_local(E.site_pos.col(s), E.shift), E.site_ori.col(s), layout.tx.name[P.S.site_nodes[s][0]]);
            }
        }
        std::filesystem::rename(tmp, fn);
    }
    catch (...)
    {
        std::filesystem::remove(tmp, ec);
        throw;
    }
    return E.shift;
}
