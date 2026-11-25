// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
// Number as short text for messages
std::string qd_num(double x)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", x);
    return buf;
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
} // namespace

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

    // Segments of the tracks: a change of scenario or indoor state starts one
    std::vector<std::vector<arma::uword>> seg(S.n_term);
    for (size_t t = 0; t < S.n_term; ++t)
        seg[t] = qd_segment_starts(L, S, t);

    // Pairs farther apart than 1e-4 m at the first row, terminal-major, then site, then segment; the image with the
    // shortest 2D distance at the first row, kept for all segments; distances to the site at the time of each segment start
    std::vector<arma::uword> l_tx, l_rx, l_img, l_row, l_seg;
    std::vector<double> l_d2d;
    for (size_t t = 0; t < S.n_term; ++t)
        for (size_t s = 0; s < S.n_site; ++s)
        {
            arma::uword best = 0;
            for (arma::uword k = 0; k < seg[t].size(); ++k)
            {
                const arma::uword rr = seg[t][k];
                const arma::vec p = qd_tx_at(L, S.site_nodes[s][0], S.term_nodes[t][0], rr); // a moving site at the time of the row
                const double dx = L.rx.pos(0, rr) - p[0], dy = L.rx.pos(1, rr) - p[1], dz = L.rx.pos(2, rr) - p[2];
                double d = arma::datum::inf;
                if (k == 0)
                    for (arma::uword m = 0; m < n_img; ++m)
                    {
                        const double e = std::hypot(dx - img(0, m), dy - img(1, m));
                        if (e < d)
                            d = e, best = m;
                    }
                else
                    d = std::hypot(dx - img(0, best), dy - img(1, best));
                if (std::sqrt(d * d + dz * dz) <= 1.0e-4)
                {
                    if (k == 0)
                        break;
                    throw std::invalid_argument("Terminal '" + L.rx.name[S.term_nodes[t][0]] + "': a later segment starts at the position of site '" +
                                                L.tx.name[S.site_nodes[s][0]] + "'; a segment start needs a LOS path of non-zero length.");
                }
                l_tx.push_back(s), l_rx.push_back(t), l_img.push_back(best), l_d2d.push_back(d), l_row.push_back(rr), l_seg.push_back(k);
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
    std::vector<std::vector<int>> t_indoor(S.n_term), t_floor(S.n_term); // Per terminal and segment
    for (size_t t = 0; t < S.n_term; ++t)
        t_indoor[t].assign(seg[t].size(), 0), t_floor[t].assign(seg[t].size(), 1);
    arma::mat o2i_rnd(S.n_term, 4, arma::fill::zeros);
    arma::u32_vec o2i_model(S.n_term, arma::fill::zeros);
    std::vector<std::vector<int>> t_fixed(S.n_term); // Building state of a segment fixed by an exact name, and the name
    std::vector<std::vector<std::string>> t_fixed_name(S.n_term);
    for (size_t t = 0; t < S.n_term; ++t)
        t_fixed[t].assign(seg[t].size(), -1), t_fixed_name[t].resize(seg[t].size());
    for (arma::uword l = 0; l < n_link; ++l)
        if (scen[l].fam == term_fam[l_rx[l]] && scen[l].los >= 0 && t_fixed[l_rx[l]][l_seg[l]] < 0)
            t_fixed[l_rx[l]][l_seg[l]] = scen[l].building, t_fixed_name[l_rx[l]][l_seg[l]] = scen[l].name;
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
            const arma::uword r = seg[T[k]][0];
            xy(k, 0) = L.rx.pos(0, r), xy(k, 1) = L.rx.pos(1, r);
        }
        const arma::vec u_io = qd_layout::o2i_field(master, f.name, qd_layout::s_io, xy);
        const arma::vec u_class = f.p_residential > 0.0 ? qd_layout::o2i_field(master, f.name, qd_layout::s_o2i_class, xy) : arma::vec();
        arma::mat rnd = qd_layout::o2i_rnd(master, f.name, xy);
        for (size_t k = 0; k < T.size(); ++k)
        {
            const size_t t = T[k];
            const arma::uword i = S.term_nodes[t][0];
            const int drawn_cls = u_class.is_empty() ? 1 : (u_class[k] < f.p_residential ? 3 : 1); // Building class of a drawn state
            double d_fixed = arma::datum::nan; // A fixed d2d_in: one value for all segment starts of the terminal
            int d_cls = -1;
            for (size_t m = 0; m < seg[t].size(); ++m)
            {
                const arma::uword r = seg[t][m];
                const int fixed = L.rx.indoor.is_empty() ? -1 : L.rx.indoor[r];
                const int b = t_fixed[t][m];
                if ((b == 1 && (fixed == 0 || fixed == 2)) || (b == 0 && (fixed == 1 || fixed == 3)))
                    throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': the conf '" + t_fixed_name[t][m] + "' fixes " +
                                                (b == 1 ? "a building" : "an outdoor") + " state, but 'indoor' is " + std::to_string(fixed) + ".");
                const int cls = (fixed == 1 || fixed == 3) ? fixed : drawn_cls;
                int state = fixed;
                if (b == 1 && fixed < 0)
                    state = cls;
                else if (b == 0 && fixed < 0)
                    state = f.car ? 2 : 0;
                else if (fixed < 0)
                    state = u_io[k] < f.building_share ? cls : (f.car ? 2 : 0);
                t_indoor[t][m] = state;
                t_floor[t][m] = qd_floor(L.rx, r);
                const double dmax = qd_d2d_in_max(f, cls), d = L.rx.d2d_in.is_empty() ? arma::datum::nan : L.rx.d2d_in[r];
                if (!std::isnan(d) && (state == 1 || state == 3) && d > dmax)
                    throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': 'd2d_in' of " + qd_num(d) + " m is above the maximum indoor distance " +
                                                qd_num(dmax) + " m of the family '" + f.name + "'.");
                if (!std::isnan(d) && !std::isnan(d_fixed) && d != d_fixed)
                    throw std::invalid_argument("Terminal '" + L.rx.name[i] + "': 'd2d_in' is " + qd_num(d_fixed) + " m and " + qd_num(d) +
                                                " m at the starts of its segments; a terminal has one indoor distance.");
                if (!std::isnan(d) && std::isnan(d_fixed))
                    d_fixed = d, d_cls = cls;
            }
            const double dmax = qd_d2d_in_max(f, d_cls);
            if (!std::isnan(d_fixed) && d_fixed <= dmax)
                rnd(k, 1) = rnd(k, 2) = dmax > 0.0 ? d_fixed / dmax : 0.0;
            o2i_rnd.row(t) = rnd.row(k);
        }
        arma::uvec mi; // The building model of each terminal, from column 0
        quadriga_lib::o2i_loss(arma::vec{L.center_freq[0]}, rnd, f.o2i_models, qd_o2i_weights(f, o2i_low), f.d2d_in_max, nullptr, &mi);
        for (size_t k = 0; k < T.size(); ++k)
            o2i_model[T[k]] = (arma::u32)mi[k];
    }

    // States of the links
    K.reset();
    K.i_tx.set_size(n_link), K.i_rx.set_size(n_link), K.i_segment.set_size(n_link), K.image.set_size(n_link), K.i_conf.set_size(n_link);
    K.los.set_size(n_link), K.indoor.set_size(n_link), K.floor_id.set_size(n_link), K.d2d_in.set_size(n_link), K.gbsm_seed.set_size(n_link);
    for (arma::uword l = 0; l < n_link; ++l)
    {
        const size_t t = l_rx[l];
        const arma::uword m = l_seg[l];
        K.i_tx[l] = (arma::u32)l_tx[l], K.i_rx[l] = (arma::u32)t, K.i_segment[l] = (arma::u32)m, K.image[l] = (arma::u32)l_img[l];
        const qd_family *f = scen[l].has_o2i() ? term_fam[t] : nullptr;
        K.indoor[l] = f != nullptr ? t_indoor[t][m] : 0;
        qd_o2i_geometry(f, K.indoor[l], o2i_rnd(t, 1), o2i_rnd(t, 2), t_floor[t][m], K.floor_id[l], K.d2d_in[l]);
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

/*!SECTION
Network layouts and mobility
SECTION!*/

/*!MD
# init_link_states
Draw the discrete state of the propagation links of a layout (stage 1 of the channel pipeline)

- Draws for each propagation link (one site, one terminal, one segment) the state that the later stages need: the wrap
  image, the scenario and conf file, the LOS and indoor state and the seeds, stored in `layout.links` ([[layout_links]])
- The states follow from the seeds and the positions, so adding a terminal does not change the states of the others
- `layout.pairing` does not limit the links; it selects at the output of the channel functions

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
- **`layout`** — Layout, valid, with `center_freq`; `layout.links` and `layout.seed` are written, the rest is unchanged
- **`method`** *(optional)* — `"38.901"`, `"38.901_MF"` (TR 38.901 clause 7.6.5.1) or `"quadriga"`, see
  [[generate_rays]]
- **`o2i_low`** *(optional)* — Share of the low-loss model among the building UTs of UMa and UMi, in [0, 1]; omitted:
  the share of TR 38.901 Table 7.8-1 (0.5); the other families have one building model; `[1]`
- **`pol_variability`** *(optional)* — Polarization power variability of TR 38.901 clause 7.6.16; TR methods only
- **`anchor_freq`** *(optional)* — Anchor frequency of `"38.901_MF"` in [Hz]
- **`sos_preset`** *(optional)* — SOS preset of the GBSM random sets, see [[sos_preset]]; omitted: `Exp256` (TR
  methods), `Comb256` (`"quadriga"`)
- **`seed`** *(optional)* — Master seed, >= -1; -1 = `layout.seed`, if that is -1 one draw of the system random device.
  The seed used is written to `layout.seed`
- **`config_dirs`** *(optional)* — Folders of user conf files, searched before the shipped folder

## Notes:
- **Links**: one per site, terminal and segment, for pairs farther apart than 1e-4 m at the first row, terminal-major
  (all sites of terminal 0 first). Co-located pairs have no link; a later segment that starts at the site throws. A
  layout without any pair farther apart gives links without entries, and the later stages return empty results
- **Scenario** of a link: the value of the terminal, else that of the site. A scenario family (e.g. `UMa`) draws the LOS
  and indoor state, a conf name of the family table (e.g. `UMa_NLOS_O2I`, also with the prefix `3GPP_38.901_v19_`) fixes
  them, and any other name is a user conf file, used in either state without O2I loss. A conf that several families list
  (`Indoor_LOS`, `InF_LOS`) belongs to the family of the site if that lists it, else to the first. Conf files are found
  with [[conf_file_list]], the folders of `config_dirs` first
- **Wrap image**: the image of the site (`layout.wrap`) with the shortest 2D distance to the first position of the
  terminal, ties to the lowest index in the order 0, $+a_1$, $-a_1$, $+a_2$, $-a_2$, $+(a_1 - a_2)$, $-(a_1 - a_2)$; all
  distances of the link use this virtual site position
- **O2I state** per terminal, from spatially correlated fields at its first position (TR 38.901 clause 7.6.3.3):
  building UTs with the share of TR 38.901 Table 7.2-1 (0.8 in UMa and UMi), Table 7.2-3 (0.5 in RMa) and Table 7.8-1A
  (0.8 in SMa), the other UTs outdoors or, in RMa and SMa, in cars; residential or commercial building in SMa (0.9); the
  O2I model and `d2d_in` as in [[o2i_loss]]. The values `indoor`, `floor` and `d2d_in` of the Rx nodes fix them; an
  exact conf name that contradicts a fixed `indoor` throws
- **One O2I state per terminal** (TR 38.901: all the links from a UT have the same indoor/outdoor state): a terminal
  linked to sites of several families has the state of the first family with O2I in site order, and an exact name on a
  site fixes it for the links to all sites; InH and InF have none
- **LOS state** per link: a uniform field per site, image and family at the first position of the segment, at the
  `LOS_state_lambda` of the NLOS conf (InF with `d_clutter`: `d_clutter / 2`), below [[los_probability]] at
  $d_\mathrm{2D,out} = \max(d_\mathrm{2D} - d_\mathrm{2D,in}, 0)$; InF-HH is always LOS
- **Segments**: on the Rx nodes, a row whose `scenario` or `indoor` differs from the row before it starts a segment (an
  empty `scenario` differs from an explicit one). A segment has its own propagation links (order: terminal, site,
  segment), LOS state, indoor state, conf and seed, drawn at its first row, with the distances to a moving site at the
  time of the segment start. The wrap image and the O2I random values (`o2i_rnd`, `o2i_model`, a drawn indoor state) are
  those of the first row of the track; a fixed `d2d_in` must be equal at all segment starts. A segment keeps its LOS
  state along its length, so long tracks need explicit segments
- **Random values**: [[sos_rand]] with the preset `Exp256`; each field has its own seed, hashed from the site seed (else
  the layout seed) and a key of its purpose, site, image and family or conf. `gbsm_seed` is the seed of
  [[gbsm_init_phases]]
- **Frequencies**: the TR methods draw the cluster and ray sets of frequency `k` with the index `k` into
  `layout.center_freq`, so a frequency inserted below another changes the rays of the higher bands

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
