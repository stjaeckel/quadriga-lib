// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_layout.hpp"
#include "qd_layout_internal.hpp"
#include "qd_helpers_shared.hpp"
#include "quadriga_sos.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*!SECTION
Network layouts and mobility
SECTION!*/

// ---- Family table and O2I draws (declared in qd_layout_internal.hpp) ----

const std::vector<qd_layout::family> &qd_layout::families()
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

const qd_layout::family *qd_layout::find_family(const std::string &name)
{
    for (const family &f : families())
        if (f.name == name)
            return &f;
    return nullptr;
}

arma::vec qd_layout::o2i_field(arma::sword base_seed, const std::string &family, int stream, const arma::mat &xy)
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

arma::mat qd_layout::o2i_rnd(arma::sword base_seed, const std::string &family, const arma::mat &xy)
{
    arma::mat rnd(xy.n_rows, 4);
    for (int s = s_o2i_model; s <= s_o2i_dev; ++s)
        rnd.col((arma::uword)(s - s_o2i_model)) = o2i_field(base_seed, family, s, xy);
    return rnd;
}

int qd_layout::draw_floor(arma::sword base_seed, const std::string &family, const std::string &terminal, int n_min, int n_max)
{
    hash_rng rng(qd_layout_seed_hash(base_seed, "floor|" + family + "|" + terminal));
    const double u1 = rng.uniform(), u2 = rng.uniform();
    const int n_fl = std::min(n_min + (int)std::floor(u1 * (double)(n_max - n_min + 1)), n_max);
    return std::min(1 + (int)std::floor(u2 * (double)n_fl), n_fl);
}

namespace
{
constexpr double qd_pi = 3.141592653589793;
constexpr double qd_two_pi = 6.283185307179586;
constexpr double qd_rad = qd_pi / 180.0; // One degree in radian
constexpr double qd_kmh = 1.0 / 3.6;     // One km/h in m/s
constexpr double qd_inf = std::numeric_limits<double>::infinity();
constexpr double qd_nan = std::numeric_limits<double>::quiet_NaN();
using qd_options = std::unordered_map<std::string, std::vector<double>>;

// Largest number of UTs of one drop: far above system-level use, small enough that the arrays fit into memory
constexpr arma::uword qd_max_ut = 10000000;

inline double qd_wrap_pi(double a) { return a - qd_two_pi * std::round(a / qd_two_pi); }

// A tuple argument with named entries; NaN or an omitted entry is "not given"
struct qd_tuple
{
    const arma::vec &v;
    const char *arg;
    std::vector<const char *> names;

    void check_size() const
    {
        if (v.n_elem > (arma::uword)names.size())
            throw std::invalid_argument("Input '" + std::string(arg) + "' must have at most " + std::to_string(names.size()) + " entries.");
    }
    bool given(arma::uword i) const { return i < v.n_elem && !std::isnan(v[i]); }
    std::string what(arma::uword i) const { return "Input '" + std::string(arg) + "': '" + names[i] + "' "; }
    void unused(arma::uword i, const std::string &by) const
    {
        if (given(i))
            throw std::invalid_argument(what(i) + "is not used by " + by + ".");
    }
    // Entry i, or 'dflt' if not given; a given value must be finite, in [lo, hi] (lo excluded if 'open'), and whole if 'whole'
    double get(arma::uword i, double dflt, double lo, double hi, bool open, bool whole, const char *rule) const
    {
        if (!given(i))
            return dflt;
        const double x = v[i];
        if (!std::isfinite(x) || x < lo || (open && x == lo) || x > hi || (whole && x != std::floor(x)))
            throw std::invalid_argument(what(i) + "must be " + rule + ".");
        return x;
    }
};

// The drop values of a preset row; the O2I state (share, cars, building classes) comes from the family table
struct qd_drop_values
{
    double h_ut;            // UT height in [m] outdoors, in cars and in buildings without floors
    double d_min;           // Minimum 2D distance to the nearest image of the nearest site in [m]
    bool d_min_out;         // Building UTs keep d_min as d2D-out = d2D - d2D-in
    double speed, speed_car; // Speed of pedestrian (outdoor and building) UTs and of car UTs in [km/h]
    int n_fl_min, n_fl_max; // Floors of a building N_fl ~ U{n_fl_min ... n_fl_max}; 0 = no floors
    int n_fl_res, n_fl_com; // Floors of residential and commercial buildings; 0 = no building classes
};

// Drop values of a site without a preset family (no scenario, a conf name)
constexpr qd_drop_values qd_no_family = {1.5, 0.0, false, 0.0, 0.0, 0, 0, 0, 0};

// Rules of the drop in the cells of one site
struct qd_rules
{
    qd_drop_values v = qd_no_family;
    const qd_layout::family *fam = nullptr; // Family with O2I states; nullptr = no O2I
    double share = 0.0;                     // Share of building UTs
    double p_residential = 0.0;             // Share of residential buildings
    double floor_height = 3.0;              // Floor height in [m], TR 36.873 Table 6-1: hUT = 3 (n_fl - 1) + 1.5
};

qd_rules qd_rules_of(const qd_drop_values &v, const std::string &family)
{
    qd_rules r;
    r.v = v;
    const qd_layout::family *f = qd_layout::find_family(family);
    if (f != nullptr && f->building_share > 0.0)
        r.fam = f, r.share = f->building_share, r.p_residential = f->p_residential;
    return r;
}

// UT values that the 'ut' tuple fixes for all UTs, NaN = rule of the site; angles in [rad]
struct qd_ut_values
{
    double h_ut = qd_nan, d_min = qd_nan, speed = qd_nan, bank = 0.0, tilt = 0.0, heading = qd_nan;
};

// Applies 'ut', 'indoor' and 'options' to the rules of the sites; 'by' names the preset or the layout in messages
qd_ut_values qd_apply_args(std::vector<qd_rules> &rules, const arma::vec &ut, const arma::vec &indoor, const qd_options &options,
                           const std::string &by)
{
    const qd_tuple U{ut, "ut", {"height", "min distance", "speed", "bank", "tilt", "heading"}};
    const qd_tuple I{indoor, "indoor", {"indoor ratio", "min floors", "max floors", "floor height"}};
    U.check_size();
    I.check_size();
    qd_ut_values o;
    o.h_ut = U.get(0, qd_nan, -qd_inf, qd_inf, false, false, "finite");
    o.d_min = U.get(1, qd_nan, 0.0, qd_inf, false, false, "finite and >= 0");
    o.speed = U.get(2, qd_nan, 0.0, qd_inf, false, false, "finite and >= 0");
    o.bank = U.get(3, 0.0, -qd_inf, qd_inf, false, false, "finite") * qd_rad;
    o.tilt = U.get(4, 0.0, -qd_inf, qd_inf, false, false, "finite") * qd_rad;
    o.heading = U.get(5, qd_nan, -qd_inf, qd_inf, false, false, "finite") * qd_rad;

    bool any_o2i = false, any_nfl = false, any_class = false;
    for (const qd_rules &r : rules)
    {
        any_o2i = any_o2i || r.fam != nullptr;
        any_nfl = any_nfl || r.v.n_fl_max > 0;
        any_class = any_class || r.v.n_fl_res > 0;
    }
    if (!any_o2i)
        I.unused(0, by);
    if (!any_nfl)
        I.unused(1, by), I.unused(2, by);
    if (!any_nfl && !any_class)
        I.unused(3, by);
    const double share = I.get(0, qd_nan, 0.0, 1.0, false, false, "in [0, 1]");
    const double n_min = I.get(1, qd_nan, 1.0, 1000.0, false, true, "a whole number in [1, 1000]");
    const double n_max = I.get(2, qd_nan, 1.0, 1000.0, false, true, "a whole number in [1, 1000]");
    const double fh = I.get(3, qd_nan, 0.0, qd_inf, true, false, "finite and > 0");

    double p_res = qd_nan;
    for (const auto &kv : options)
    {
        if (kv.first != "p_residential")
            throw std::invalid_argument("Input 'options' has an unknown key ('" + kv.first + "').");
        if (!any_class)
            throw std::invalid_argument("Option 'p_residential' is not used by " + by + ".");
        if (kv.second.size() != 1 || !(kv.second[0] >= 0.0 && kv.second[0] <= 1.0))
            throw std::invalid_argument("Option 'p_residential' must be one number in [0, 1].");
        p_res = kv.second[0];
    }

    for (qd_rules &r : rules)
    {
        if (r.fam != nullptr && !std::isnan(share))
            r.share = share;
        if (r.v.n_fl_max > 0)
        {
            r.v.n_fl_min = std::isnan(n_min) ? r.v.n_fl_min : (int)n_min;
            r.v.n_fl_max = std::isnan(n_max) ? r.v.n_fl_max : (int)n_max;
            if (r.v.n_fl_min > r.v.n_fl_max)
                throw std::invalid_argument("Input 'indoor': 'min floors' must not exceed 'max floors' (" + std::to_string(r.v.n_fl_min) +
                                            " > " + std::to_string(r.v.n_fl_max) + ").");
        }
        if (!std::isnan(fh))
            r.floor_height = fh;
        if (r.v.n_fl_res > 0 && !std::isnan(p_res))
            r.p_residential = p_res;
        if (!std::isnan(o.d_min))
            r.v.d_min = o.d_min;
    }
    return o;
}

// Throws unless the drop has at most qd_max_ut UTs
void qd_check_n_ut(arma::uword n_ut, arma::uword n_cell, bool per_cell)
{
    if (n_ut > qd_max_ut || (per_cell && n_cell > 0 && n_ut > qd_max_ut / n_cell))
        throw std::invalid_argument("Input 'n_ut' is too large: a drop has at most " + std::to_string(qd_max_ut) + " UTs.");
}

// The cells of a Tx side: sites with their position and sector headings, the wrap images and the drop area
struct qd_cells
{
    size_t n_site = 0;
    std::vector<double> sx, sy;                 // Site positions, [n_site]
    std::vector<std::vector<arma::uword>> node; // Tx nodes (sectors) of each site
    std::vector<std::vector<double>> head;      // Their headings in [rad]
    std::vector<double> ix, iy;                 // Image offsets, image 0 first
    bool bounded = false;
    double b[4] = {0.0, 0.0, 0.0, 0.0};   // Bounds of the drop
    double box[4] = {0.0, 0.0, 0.0, 0.0}; // Sampling box

    // Tx node of the cell of a point and its distance 'd' to the site, or -1 outside the drop area. The cell is that of the
    // nearest (site, image) with ties to the lower site and image index; it must be image 0. A site with several sectors is
    // split by the nearest heading
    long long cell(double x, double y, double &d) const
    {
        if (bounded && (x < b[0] || x > b[1] || y < b[2] || y > b[3]))
            return -1;
        double best = qd_inf;
        size_t bs = 0, bi = 0;
        for (size_t s = 0; s < n_site; ++s)
            for (size_t k = 0; k < ix.size(); ++k)
            {
                const double dx = x - sx[s] - ix[k], dy = y - sy[s] - iy[k], d2 = dx * dx + dy * dy;
                if (d2 < best)
                    best = d2, bs = s, bi = k;
            }
        if (bi != 0)
            return -1;
        d = std::sqrt(best);
        const std::vector<double> &h = head[bs];
        if (h.size() == 1)
            return (long long)node[bs][0];
        const double az = std::atan2(y - sy[bs], x - sx[bs]);
        size_t best_k = 0;
        double best_a = qd_inf;
        for (size_t k = 0; k < h.size(); ++k)
        {
            const double a = std::abs(qd_wrap_pi(az - h[k]));
            if (a < best_a)
                best_a = a, best_k = k;
        }
        return (long long)node[bs][best_k];
    }
};

// Largest distance from 0 of the region of points that are nearer to 0 than to any image offset: the corners of the
// polygon bounded by the bisectors x . w = |w|^2 / 2
double qd_image_cell_radius(const std::vector<double> &ix, const std::vector<double> &iy)
{
    const size_t n = ix.size();
    double r = 0.0;
    for (size_t i = 1; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
        {
            const double ci = 0.5 * (ix[i] * ix[i] + iy[i] * iy[i]), cj = 0.5 * (ix[j] * ix[j] + iy[j] * iy[j]);
            const double det = ix[i] * iy[j] - iy[i] * ix[j];
            if (std::abs(det) <= 1.0e-12 * (ci + cj))
                continue;
            const double x = (ci * iy[j] - cj * iy[i]) / det, y = (ix[i] * cj - ix[j] * ci) / det;
            bool inside = true;
            for (size_t k = 1; k < n && inside; ++k)
            {
                const double ck = 0.5 * (ix[k] * ix[k] + iy[k] * iy[k]);
                inside = x * ix[k] + y * iy[k] <= ck * (1.0 + 1.0e-9);
            }
            if (inside)
                r = std::max(r, std::hypot(x, y));
        }
    return r;
}

// The cells of the Tx side 'tx' (valid) with the wrap vectors and optional bounds [4]; throws if two sectors of a site have
// the same heading (their split leaves one of them without area)
qd_cells qd_make_cells(const quadriga_lib::layout_nodes &tx, const std::vector<size_t> &site, size_t n_site, const arma::mat &wrap,
                       const arma::vec &bounds)
{
    qd_cells C;
    C.n_site = n_site;
    C.sx.assign(n_site, 0.0), C.sy.assign(n_site, 0.0);
    C.node.resize(n_site), C.head.resize(n_site);
    for (arma::uword i = 0; i < tx.n_node(); ++i)
    {
        const size_t s = site[i];
        const arma::uword r = tx.offset[i];
        if (C.node[s].empty())
            C.sx[s] = tx.pos(0, r), C.sy[s] = tx.pos(1, r);
        for (size_t k = 0; k < C.node[s].size(); ++k)
            if (std::abs(qd_wrap_pi(tx.orientation(2, r) - C.head[s][k])) < 1.0e-6)
                throw std::invalid_argument("Site '" + tx.name[i] + "': sectors " + std::to_string(tx.sector[C.node[s][k]]) + " and " +
                                            std::to_string(tx.sector[i]) + " have the same heading; the cells of a site are split by the sector headings.");
        C.node[s].push_back(i);
        C.head[s].push_back(tx.orientation(2, r));
    }

    // Images in the order 0, +a1, -a1, +a2, -a2, +(a1 - a2), -(a1 - a2)
    C.ix = {0.0}, C.iy = {0.0};
    if (wrap.n_cols >= 1)
        C.ix.insert(C.ix.end(), {wrap(0, 0), -wrap(0, 0)}), C.iy.insert(C.iy.end(), {wrap(1, 0), -wrap(1, 0)});
    if (wrap.n_cols == 2)
    {
        const double dx = wrap(0, 0) - wrap(0, 1), dy = wrap(1, 0) - wrap(1, 1);
        C.ix.insert(C.ix.end(), {wrap(0, 1), -wrap(0, 1), dx, -dx}), C.iy.insert(C.iy.end(), {wrap(1, 1), -wrap(1, 1), dy, -dy});
    }

    C.bounded = !bounds.is_empty();
    if (C.bounded)
        for (int k = 0; k < 4; ++k)
            C.b[k] = C.box[k] = bounds[(arma::uword)k];
    if (wrap.n_cols == 2)
    {
        // A point of the cluster is nearer to its site than to the images of that site: within the radius of the image cell
        const double R = qd_image_cell_radius(C.ix, C.iy);
        const double box[4] = {*std::min_element(C.sx.begin(), C.sx.end()) - R, *std::max_element(C.sx.begin(), C.sx.end()) + R,
                               *std::min_element(C.sy.begin(), C.sy.end()) - R, *std::max_element(C.sy.begin(), C.sy.end()) + R};
        for (int k = 0; k < 4; ++k)
            C.box[k] = !C.bounded ? box[k] : (k % 2 == 0 ? std::max(C.box[k], box[k]) : std::min(C.box[k], box[k]));
        if (!(C.box[0] < C.box[1] && C.box[2] < C.box[3]))
            throw std::invalid_argument("Input 'bounds' does not overlap the wrapped cluster of the sites.");
    }
    return C;
}

// The UT drop into the cells of a Tx side; returns the new Rx side. 'site' is the site of each Tx node, 'rules' per site.
// The result is a function of the seed: the candidates come from one stream, the per-UT values from streams keyed by name
quadriga_lib::layout_nodes qd_drop(const quadriga_lib::layout_nodes &tx, const std::vector<size_t> &site, const qd_cells &C,
                                   const std::vector<qd_rules> &rules, const qd_ut_values &o, arma::uword n_ut, bool per_cell, arma::sword seed)
{
    const arma::uword n_cell = tx.n_node(), n_target = per_cell ? n_ut * n_cell : n_ut;
    std::vector<arma::uword> count(n_cell, 0);

    // An accepted UT: position, cell and the O2I fields at its position (evaluated once, kept for the states)
    struct qd_ut
    {
        double x, y;
        arma::uword cell;
        double io, cl, u1, u2;
    };
    std::vector<qd_ut> A;
    A.reserve(n_target);

    // Candidates from one stream. A candidate passes the geometric test if its cell is not full and d2D >= d_min; its O2I
    // fields are then evaluated (per batch and family), and a building UT of a family with the d2D-out rule is also
    // rejected if d2D - d2D-in < d_min, so a rejected UT is redrawn with its position and indoor distance together. The
    // fields are a function of the position, so the result does not depend on the batch size
    std::mt19937_64 rng = qd_rng(qd_layout_seed_hash(seed, "drop"));
    const double w = C.box[1] - C.box[0], h = C.box[3] - C.box[2];
    const unsigned long long max_cand = std::max(200000ULL, 1000ULL * (unsigned long long)n_target);
    unsigned long long n_cand = 0;
    std::vector<double> cx, cy, cd, io, cl, u1, u2;
    std::vector<long long> cc;
    while ((arma::uword)A.size() < n_target)
    {
        if (n_cand >= max_cand)
        {
            std::string where;
            for (arma::uword i = 0; i < n_cell && per_cell; ++i)
                if (count[i] < n_ut)
                {
                    where = " in the cell of Tx node '" + tx.name[i] + "', sector " + std::to_string(tx.sector[i]);
                    break;
                }
            throw std::invalid_argument("No room for the terminals" + where + ": after " + std::to_string(n_cand) +
                                        " draws, the bounds and the minimum distance leave too little area.");
        }
        const size_t B = (size_t)std::min<arma::uword>(4096, std::max<arma::uword>(64, 8 * (n_target - (arma::uword)A.size())));
        cx.resize(B), cy.resize(B), cd.assign(B, 0.0), cc.resize(B), io.assign(B, 1.0), cl.assign(B, 1.0), u1.assign(B, 0.0), u2.assign(B, 0.0);
        std::map<const qd_layout::family *, std::vector<size_t>> need;
        for (size_t j = 0; j < B; ++j)
        {
            cx[j] = C.box[0] + w * qd_rng_uniform(rng);
            cy[j] = C.box[2] + h * qd_rng_uniform(rng);
            cc[j] = C.cell(cx[j], cy[j], cd[j]);
            if (cc[j] < 0 || (per_cell && count[(size_t)cc[j]] == n_ut) || cd[j] < rules[site[(size_t)cc[j]]].v.d_min)
                cc[j] = -1;
            else if (rules[site[(size_t)cc[j]]].fam != nullptr)
                need[rules[site[(size_t)cc[j]]].fam].push_back(j);
        }
        n_cand += B;

        for (const auto &kv : need)
        {
            arma::mat xy(kv.second.size(), 2);
            for (size_t i = 0; i < kv.second.size(); ++i)
                xy(i, 0) = cx[kv.second[i]], xy(i, 1) = cy[kv.second[i]];
            const std::string &fn = kv.first->name;
            const arma::vec f_io = qd_layout::o2i_field(seed, fn, qd_layout::s_io, xy), f_u1 = qd_layout::o2i_field(seed, fn, qd_layout::s_d2din1, xy),
                            f_u2 = qd_layout::o2i_field(seed, fn, qd_layout::s_d2din2, xy);
            const arma::vec f_cl = kv.first->p_residential > 0.0 ? qd_layout::o2i_field(seed, fn, qd_layout::s_o2i_class, xy) : arma::vec();
            for (size_t i = 0; i < kv.second.size(); ++i)
            {
                const size_t j = kv.second[i];
                io[j] = f_io[i], u1[j] = f_u1[i], u2[j] = f_u2[i];
                if (!f_cl.is_empty())
                    cl[j] = f_cl[i];
            }
        }

        for (size_t j = 0; j < B && (arma::uword)A.size() < n_target; ++j)
        {
            if (cc[j] < 0)
                continue;
            const size_t c = (size_t)cc[j];
            if (per_cell && count[c] == n_ut)
                continue;
            const qd_rules &r = rules[site[c]];
            if (r.fam != nullptr && r.v.d_min_out && io[j] < r.share)
            {
                const bool residential = r.v.n_fl_res > 0 && cl[j] < r.p_residential;
                const double d_max = residential ? r.fam->d2d_in_max : r.fam->d2d_in_max_commercial;
                if (cd[j] - qd_layout::d2d_in_of(u1[j], u2[j], d_max) < r.v.d_min)
                    continue;
            }
            A.push_back({cx[j], cy[j], c, io[j], cl[j], u1[j], u2[j]});
            ++count[c];
        }
    }
    if (per_cell) // Order: by cell, else as drawn
        std::stable_sort(A.begin(), A.end(), [](const qd_ut &a, const qd_ut &b) { return a.cell < b.cell; });

    const arma::uword n = (arma::uword)A.size();
    quadriga_lib::layout_nodes R;
    R.name.resize(n);
    R.sector.zeros(n);
    R.i_antenna.zeros(n);
    R.offset.set_size(n);
    for (arma::uword k = 0; k < n; ++k)
        R.offset[k] = k;
    R.pos.set_size(3, n);
    R.orientation.set_size(3, n);
    R.direction.set_size(2, n);
    R.speed.set_size(n);
    bool any_o2i = false;
    for (const qd_ut &u : A)
        any_o2i = any_o2i || rules[site[u.cell]].fam != nullptr;
    if (any_o2i)
    {
        R.indoor.set_size(n), R.indoor.fill(-1);
        R.floor.set_size(n), R.floor.fill(-1);
        R.d2d_in.set_size(n), R.d2d_in.fill(qd_nan);
    }
    for (arma::uword k = 0; k < n; ++k)
    {
        const qd_ut &u = A[k];
        const qd_rules &r = rules[site[u.cell]];
        const std::string name = "UT" + std::to_string(k + 1);
        R.name[k] = name;
        double z = std::isnan(o.h_ut) ? r.v.h_ut : o.h_ut;
        int state = -1;
        if (r.fam != nullptr)
        {
            const bool building = u.io < r.share;
            const bool residential = building && r.v.n_fl_res > 0 && u.cl < r.p_residential;
            const bool floors = r.v.n_fl_max > 0 || r.v.n_fl_res > 0;
            state = building ? (residential ? 3 : 1) : (r.fam->car ? 2 : 0);
            int fl = 1;
            if (building && floors && !std::isnan(o.h_ut)) // The floor of the given height
                fl = std::max((int)std::floor((o.h_ut - 1.5) / r.floor_height + 0.5) + 1, 1);
            else if (building && floors) // TR 36.873 Table 6-1: hUT = 3 (n_fl - 1) + 1.5
            {
                const int n_lo = r.v.n_fl_max > 0 ? r.v.n_fl_min : (residential ? r.v.n_fl_res : r.v.n_fl_com);
                const int n_hi = r.v.n_fl_max > 0 ? r.v.n_fl_max : n_lo;
                fl = qd_layout::draw_floor(seed, r.fam->name, name, n_lo, n_hi);
                z = r.floor_height * (double)(fl - 1) + 1.5;
            }
            R.indoor[k] = state;
            R.floor[k] = fl;
            const double d_max = residential ? r.fam->d2d_in_max : r.fam->d2d_in_max_commercial;
            R.d2d_in[k] = building ? qd_layout::d2d_in_of(u.u1, u.u2, d_max) : 0.0;
        }
        R.pos(0, k) = u.x, R.pos(1, k) = u.y, R.pos(2, k) = z;

        // Heading and direction from the stream of the UT; both are drawn, so that one does not depend on the other
        qd_layout::hash_rng g(qd_layout_seed_hash(seed, "ut|" + name));
        const double u_head = g.uniform(), u_dir = g.uniform();
        R.orientation(0, k) = o.bank, R.orientation(1, k) = o.tilt;
        R.orientation(2, k) = std::isnan(o.heading) ? qd_wrap_pi(qd_two_pi * u_head) : o.heading;
        R.direction(0, k) = qd_wrap_pi(qd_two_pi * u_dir), R.direction(1, k) = 0.0;
        R.speed[k] = (!std::isnan(o.speed) ? o.speed : (state == 2 ? r.v.speed_car : r.v.speed) * qd_kmh);
    }
    return R;
}

// ---- Presets ----

struct qd_preset;

// The arguments of the geometry of a preset
struct qd_geometry_args
{
    qd_tuple spacing, grid, area, bs;
    const qd_options &options;
};

// A builder writes the Tx side and the wrap vectors of a layout, and the bounds of the drop (empty = the wrapped cluster)
using qd_builder = void (*)(const qd_preset &, const qd_geometry_args &, quadriga_lib::layout &, arma::vec &);

// A row of the preset table: a new preset is a row, plus a builder if its geometry is new
struct qd_preset
{
    const char *name;    // Preset name
    const char *family;  // Scenario family of the Tx rows and of the O2I state of the drop; "" = none
    qd_builder build;    // Geometry
    double spacing[2];   // ISD (hexagonal) or BS spacing in x and y in [m]
    double grid[3];      // Rings (hexagonal) or BSs in x and y; sectors per site
    double area[3];      // Length, width, height of the hall in [m]; NaN = not used
    double h_bs, tilt;   // BS height in [m] and mechanical tilt in [deg] (positive down); NaN tilt = SMa rule
    bool hall_scenpar;   // Writes the hall parameters DS_VS and absTOA_Lmax (InF)
    qd_drop_values drop; // UT drop
};

void qd_build_hexagonal(const qd_preset &p, const qd_geometry_args &a, quadriga_lib::layout &L, arma::vec &bounds);
void qd_build_indoor(const qd_preset &p, const qd_geometry_args &a, quadriga_lib::layout &L, arma::vec &bounds);

// The presets: TR 38.901 Tables 7.2-1 (UMi, UMa: 19 sites, 3 sectors, ISD 200 m and 500 m, hBS 10 m and 25 m), 7.2-3 (RMa:
// ISD 1732 m, hBS 35 m), 7.2-5 (SMa: ISD 1299 m, hBS 35 m), 7.2-2 (InH: 120 m x 50 m x 3 m, ISD 20 m, hBS 3 m, the 12 BSs
// of Figure 7.2-1), 7.8-7 (InF halls 120 m x 60 m and 300 m x 150 m, 10 m high, 18 BSs with D = 20 m or 50 m, hBS 1.5 m
// for SL and DL, 8 m for SH and DH). InF-HH has no calibration layout in the TR: the big hall with hBS 8 m. Mechanical
// tilt 0 except SMa (rule from the ISD); the electrical tilts of Table 7.8-1 belong to the antennas.
// Drop: UT height 1.5 m (Tables 7.2-3, 7.2-5, 7.8-7; UMi and UMa outdoors, TR 36.873 Table 6-1), 1 m InH (Table 7.2-2),
// 3 m InF-HH (library value above the clutter height 2 m); minimum distance 10 m UMi, 35 m UMa (Table 7.2-1), RMa (Table
// 7.2-3), SMa (Table 7.2-5), 0 InH (Table 7.2-2), 1 m InF (Table 7.8-7), to d2D-out for indoor UMi and UMa UTs (TR 36.873
// Table 6-1, note 1); speed 3 km/h (Tables 7.2-1, 7.2-2, 7.2-5; InF and RMa: the TR gives none, the same), in cars 40 km/h
// (Table 7.2-5; RMa: as SMa); floors N_fl ~ U(4, 8) in UMi and UMa (TR 36.873 Table 6-1), up to 2 floors in residential
// and up to 5 in commercial SMa buildings (Table 7.2-5)
const qd_preset qd_presets[] = {
    {"38.901_UMi", "UMi", qd_build_hexagonal, {200.0, qd_nan}, {2.0, qd_nan, 3.0}, {qd_nan, qd_nan, qd_nan}, 10.0, 0.0, false, {1.5, 10.0, true, 3.0, 3.0, 4, 8, 0, 0}},
    {"38.901_UMa", "UMa", qd_build_hexagonal, {500.0, qd_nan}, {2.0, qd_nan, 3.0}, {qd_nan, qd_nan, qd_nan}, 25.0, 0.0, false, {1.5, 35.0, true, 3.0, 3.0, 4, 8, 0, 0}},
    {"38.901_RMa", "RMa", qd_build_hexagonal, {1732.0, qd_nan}, {2.0, qd_nan, 3.0}, {qd_nan, qd_nan, qd_nan}, 35.0, 0.0, false, {1.5, 35.0, false, 3.0, 40.0, 0, 0, 0, 0}},
    {"38.901_SMa", "SMa", qd_build_hexagonal, {1299.0, qd_nan}, {2.0, qd_nan, 3.0}, {qd_nan, qd_nan, qd_nan}, 35.0, qd_nan, false, {1.5, 35.0, false, 3.0, 40.0, 0, 0, 2, 5}},
    {"38.901_InH_Open", "InH_Open", qd_build_indoor, {20.0, 20.0}, {6.0, 2.0, 1.0}, {120.0, 50.0, qd_nan}, 3.0, 0.0, false, {1.0, 0.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InH_Mixed", "InH_Mixed", qd_build_indoor, {20.0, 20.0}, {6.0, 2.0, 1.0}, {120.0, 50.0, qd_nan}, 3.0, 0.0, false, {1.0, 0.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InF_SL", "InF_SL", qd_build_indoor, {20.0, 20.0}, {6.0, 3.0, 1.0}, {120.0, 60.0, 10.0}, 1.5, 0.0, true, {1.5, 1.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InF_DL", "InF_DL", qd_build_indoor, {50.0, 50.0}, {6.0, 3.0, 1.0}, {300.0, 150.0, 10.0}, 1.5, 0.0, true, {1.5, 1.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InF_SH", "InF_SH", qd_build_indoor, {50.0, 50.0}, {6.0, 3.0, 1.0}, {300.0, 150.0, 10.0}, 8.0, 0.0, true, {1.5, 1.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InF_DH", "InF_DH", qd_build_indoor, {20.0, 20.0}, {6.0, 3.0, 1.0}, {120.0, 60.0, 10.0}, 8.0, 0.0, true, {1.5, 1.0, false, 3.0, 3.0, 0, 0, 0, 0}},
    {"38.901_InF_HH", "InF_HH", qd_build_indoor, {50.0, 50.0}, {6.0, 3.0, 1.0}, {300.0, 150.0, 10.0}, 8.0, 0.0, true, {3.0, 1.0, false, 3.0, 3.0, 0, 0, 0, 0}}};

// The drop rules of a scenario: those of the preset row of its family, else the defaults of a site without family
qd_rules qd_rules_of_scenario(const std::string &scenario)
{
    for (const qd_preset &p : qd_presets)
        if (!scenario.empty() && scenario == p.family)
            return qd_rules_of(p.drop, p.family);
    return qd_rules_of(qd_no_family, "");
}

std::string qd_by_preset(const qd_preset &p) { return "the preset '" + std::string(p.name) + "'"; }

// Hexagonal grid with wrap-around, flat-top site hexagons with circumradius ISD / sqrt(3). Sites q u + r v with
// u = ISD (sqrt(3)/2, 1/2), v = ISD (0, 1) and max(|q|, |r|, |q + r|) <= rings, i.e. 3 n^2 + 3 n + 1 sites, ordered by ring
// and by azimuth in (-30, 330] deg; boresights 30 deg + k 360 deg / n_sector (TR 38.901 Table 7.8-1: 30, 150 and 270 deg).
// Lattice a1 = n u + (n + 1) v, a2 = a1 rotated by +60 deg: the cluster and its 6 neighbour copies 0, +-a1, +-a2,
// +-(a1 - a2) tile the plane exactly once (for n = 2: a1 = (sqrt(3), 4) ISD, a2 = (-3 sqrt(3) / 2, 7 / 2) ISD)
void qd_build_hexagonal(const qd_preset &p, const qd_geometry_args &a, quadriga_lib::layout &L, arma::vec &bounds)
{
    const std::string by = qd_by_preset(p);
    a.spacing.unused(1, by), a.grid.unused(1, by);
    for (arma::uword k = 0; k < 3; ++k)
        a.area.unused(k, by);
    const double isd = a.spacing.get(0, p.spacing[0], 0.0, qd_inf, true, false, "finite and > 0");
    const int n = (int)a.grid.get(0, p.grid[0], 0.0, 50.0, false, true, "a whole number in [0, 50]");
    const int ns = (int)a.grid.get(2, p.grid[2], 1.0, 12.0, false, true, "a whole number in [1, 12]");
    const double h_bs = a.bs.get(0, p.h_bs, -qd_inf, qd_inf, false, false, "finite");
    // SMa: mechanical downtilt 95 deg at ISD 1299 m and 92 deg at 1732 m (Tables 7.8-1A, 7.8-2A), zenith angles, i.e. 5 and
    // 2 deg down; other ISDs take the value of the nearer one
    const double tilt_rule = std::isnan(p.tilt) ? (std::abs(isd - 1299.0) <= std::abs(isd - 1732.0) ? 5.0 : 2.0) : p.tilt;
    const double tilt = a.bs.get(1, tilt_rule, -qd_inf, qd_inf, false, false, "finite") * qd_rad;
    const double offset = a.bs.get(2, 0.0, -qd_inf, qd_inf, false, false, "finite");
    struct site
    {
        int ring;
        double az, x, y;
    };
    std::vector<site> S;
    const double s3 = std::sqrt(3.0);
    for (int q = -n; q <= n; ++q)
        for (int r = -n; r <= n; ++r)
        {
            const int ring = std::max(std::max(std::abs(q), std::abs(r)), std::abs(q + r));
            if (ring > n)
                continue;
            const double x = q * isd * s3 / 2.0, y = q * isd / 2.0 + r * isd;
            double az = std::atan2(y, x) / qd_rad;
            if (az <= -30.0 + 1.0e-9)
                az += 360.0;
            S.push_back({ring, ring == 0 ? 0.0 : az, x, y});
        }
    std::sort(S.begin(), S.end(), [](const site &u, const site &v) { return u.ring != v.ring ? u.ring < v.ring : u.az < v.az; });

    quadriga_lib::layout_nodes &T = L.tx;
    const arma::uword nn = (arma::uword)(S.size() * (size_t)ns);
    T.name.resize(nn);
    T.sector.set_size(nn);
    T.i_antenna.zeros(nn);
    T.offset.set_size(nn);
    T.pos.set_size(3, nn);
    T.orientation.set_size(3, nn);
    T.scenario.assign(nn, p.family);
    for (size_t i = 0; i < S.size(); ++i)
        for (int k = 0; k < ns; ++k)
        {
            const arma::uword j = (arma::uword)(i * (size_t)ns + (size_t)k);
            T.name[j] = "BS" + std::to_string(i + 1);
            T.sector[j] = (arma::u32)k;
            T.offset[j] = j;
            T.pos(0, j) = S[i].x, T.pos(1, j) = S[i].y, T.pos(2, j) = h_bs;
            T.orientation(0, j) = 0.0, T.orientation(1, j) = tilt;
            T.orientation(2, j) = qd_wrap_pi((30.0 + offset + 360.0 * (double)k / (double)ns) * qd_rad);
        }
    const double a1x = n * isd * s3 / 2.0, a1y = n * isd / 2.0 + (n + 1) * isd;
    L.wrap = {{a1x, 0.5 * a1x - 0.5 * s3 * a1y}, {a1y, 0.5 * s3 * a1x + 0.5 * a1y}};
    bounds.reset();
}

// Indoor grid centred at the origin: n_x by n_y BSs with the spacing d_x, d_y, ordered by y and then by x; the hall is
// the drop area. TR 38.901 Figure 7.2-1 (InH), Table 7.8-7 (InF: D/2 from the walls)
void qd_build_indoor(const qd_preset &p, const qd_geometry_args &a, quadriga_lib::layout &L, arma::vec &bounds)
{
    const std::string by = qd_by_preset(p);
    const double dx = a.spacing.get(0, p.spacing[0], 0.0, qd_inf, true, false, "finite and > 0");
    const double dy = a.spacing.get(1, p.spacing[1], 0.0, qd_inf, true, false, "finite and > 0");
    const int nx = (int)a.grid.get(0, p.grid[0], 1.0, 100.0, false, true, "a whole number in [1, 100]");
    const int ny = (int)a.grid.get(1, p.grid[1], 1.0, 100.0, false, true, "a whole number in [1, 100]");
    const int ns = (int)a.grid.get(2, p.grid[2], 1.0, 12.0, false, true, "a whole number in [1, 12]");
    const double len = a.area.get(0, p.area[0], 0.0, qd_inf, true, false, "finite and > 0");
    const double wid = a.area.get(1, p.area[1], 0.0, qd_inf, true, false, "finite and > 0");
    if (!p.hall_scenpar)
        a.area.unused(2, by);
    const double hgt = p.hall_scenpar ? a.area.get(2, p.area[2], 0.0, qd_inf, true, false, "finite and > 0") : qd_nan;
    const double h_bs = a.bs.get(0, p.h_bs, -qd_inf, qd_inf, false, false, "finite");
    const double tilt = a.bs.get(1, p.tilt, -qd_inf, qd_inf, false, false, "finite") * qd_rad;
    const double offset = a.bs.get(2, 0.0, -qd_inf, qd_inf, false, false, "finite");
    const double tol = 1.0e-9 * std::max(len, wid);
    if ((nx - 1) * dx > len + tol || (ny - 1) * dy > wid + tol)
        throw std::invalid_argument("Inputs 'grid' and 'spacing': the grid of " + std::to_string(nx) + " x " + std::to_string(ny) +
                                    " BSs does not fit into the area.");

    quadriga_lib::layout_nodes &T = L.tx;
    const arma::uword nn = (arma::uword)(nx * ny * ns);
    T.name.resize(nn);
    T.sector.set_size(nn);
    T.i_antenna.zeros(nn);
    T.offset.set_size(nn);
    T.pos.set_size(3, nn);
    T.orientation.set_size(3, nn);
    T.scenario.assign(nn, p.family);
    arma::uword j = 0;
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            for (int k = 0; k < ns; ++k, ++j)
            {
                T.name[j] = "BS" + std::to_string(iy * nx + ix + 1);
                T.sector[j] = (arma::u32)k;
                T.offset[j] = j;
                T.pos(0, j) = (ix - 0.5 * (nx - 1)) * dx, T.pos(1, j) = (iy - 0.5 * (ny - 1)) * dy, T.pos(2, j) = h_bs;
                T.orientation(0, j) = 0.0, T.orientation(1, j) = tilt;
                T.orientation(2, j) = qd_wrap_pi((offset + 360.0 * (double)k / (double)ns) * qd_rad);
            }
    if (p.hall_scenpar)
    {
        // V/S of the hall for the DS of InF (TR 38.901 Table 7.5-6 Part-3) and the largest dimension L = max(length, width,
        // height) that bounds the excess delay (clause 7.6.9)
        const double vs = len * wid * hgt / (2.0 * (len * wid + len * hgt + wid * hgt));
        T.scenpar_override["DS_VS"] = arma::vec(nn, arma::fill::value(vs));
        T.scenpar_override["absTOA_Lmax"] = arma::vec(nn, arma::fill::value(std::max(std::max(len, wid), hgt)));
    }
    L.wrap.reset();
    bounds = {-0.5 * len, 0.5 * len, -0.5 * wid, 0.5 * wid};
}


std::string qd_lower(const std::string &s)
{
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return t;
}

// A seed >= 0, or one draw of the system random device
arma::sword qd_resolve_seed(arma::sword seed)
{
    if (seed >= 0)
        return seed;
    std::mt19937_64 rng = qd_rng(-1);
    return (arma::sword)(rng() & 0x7FFFFFFFFFFFFFFFULL);
}
} // namespace

/*!MD
# generate_layout
Network layout from a preset: BS sites and a drop of UTs

- Presets (not case sensitive) of 3GPP TR 38.901 with the values of its tables. BS sites are named `BS1`, `BS2`, ..., with one node per sector; the Tx rows get the scenario family (the preset name without `38.901_`). UTs are named `UT1`, `UT2`, ...
- Hexagonal presets, 2 rings (19 sites), 3 sectors with boresights 30°, 150° and 270°, wrap vectors set:
  - `38.901_UMi`: ISD 200 m, hBS 10 m; `38.901_UMa`: ISD 500 m, hBS 25 m; `38.901_RMa`: ISD 1732 m, hBS 35 m
  - `38.901_SMa`: ISD 1299 m, hBS 35 m, mechanical tilt 5° at ISD 1299 m and 2° at 1732 m (other ISDs: that of the nearer one); option `p_residential` (0.9)
  - Sites ordered by ring and then by azimuth in (−30°, 330°]; the wrap vectors make the cluster tile the plane
- Indoor presets, centred at the origin, one sector, no wrap; a lattice of BSs centred in the hall, ordered by y and then by x:
  - `38.901_InH_Open`, `38.901_InH_Mixed`: 120 × 50 m, 6 × 2 BSs at 20 m, hBS 3 m
  - `38.901_InF_SL`: 120 × 60 × 10 m, 6 × 3 BSs at 20 m, hBS 1.5 m; `38.901_InF_DL`: 300 × 150 × 10 m, 6 × 3 BSs at 50 m, hBS 1.5 m; `38.901_InF_SH`: as DL with hBS 8 m; `38.901_InF_DH`: as SL with hBS 8 m; `38.901_InF_HH`: as SH (the TR has no calibration layout for it)
  - The InF presets set `scenpar_override` `DS_VS` (V/S of the hall) and `absTOA_Lmax` (its largest dimension) on the Tx rows
- UT drop, as point drops: `n_ut` per cell or in total, uniform over the cells (the sector part of a site hexagon, the hexagon without sectors, the lattice cell around an indoor BS):
  - Minimum 2D distance to the nearest image of the nearest site: UMi 10 m, UMa 35 m (indoor UTs: to d2D-out, TR 36.873 Table 6-1), RMa and SMa 35 m, InH 0, InF 1 m; a rejected UT is redrawn
  - Height 1.5 m (InH 1 m, InF-HH 3 m); in UMi and UMa buildings `3 (n_fl − 1) + 1.5` m with `n_fl ~ U(1, N_fl)`, `N_fl ~ U(4, 8)`; SMa residential buildings 1 or 2 floors, commercial 1 to 5
  - Speed 3 km/h, in cars (RMa, SMa) 40 km/h; heading uniform in [0°, 360°), bank and tilt 0; direction uniform in azimuth, elevation 0
  - O2I states (`indoor`, `floor`, `d2d_in`) for UMi, UMa (80 % in buildings, the others outdoors), RMa (50 % in buildings, 50 % in cars) and SMa (80 % in buildings, 90 % of them residential, 20 % in cars), from spatially correlated fields at the UT positions (TR 38.901 clause 7.6.3.3); `d2d_in` up to 25 m (UMi, UMa, SMa commercial) or 10 m (RMa, SMa residential)
- The layout is a function of the seed, independent of the thread count, with the random streams of [[drop_terminals]]; seed -1 draws a seed, which is stored in `layout.seed`. Antennas are not set (the channel functions use the generic antenna until [[set_antenna]] sets one) and `center_freq` is empty

## Declaration:
```
quadriga_lib::layout quadriga_lib::generate_layout(
    const std::string &type,
    arma::uword n_ut = 0,
    const arma::vec &spacing = {},
    const arma::vec &grid = {},
    const arma::vec &area = {},
    const arma::vec &bs = {},
    const arma::vec &ut = {},
    const arma::vec &indoor = {},
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    bool per_cell = true,
    arma::sword seed = -1);
```

## Inputs:
- **`type`** — Preset name
- **`n_ut`** *(optional)* — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs
- **`spacing`** *(optional)* — `[d_x, d_y]` in [m], > 0: the ISD of a hexagonal preset (`d_y` not used) or the spacing of the indoor grid
- **`grid`** *(optional)* — `[n_x, n_y, n_sector]`, whole numbers: rings of a hexagonal preset (0 to 50, 0 = one site; `n_y` not used) or the size of the indoor grid (1 to 100); sectors per site (1 to 12), with boresights 30° (indoor 0°) + k · 360° / `n_sector`
- **`area`** *(optional)* — `[length, width, height]` of the hall along x, y and z in [m], > 0; indoor presets only, the height InF only; the BS grid must fit into it
- **`bs`** *(optional)* — `[height, tilt, azimuth offset]` of the BSs in [m], [deg], [deg]: the mechanical tilt (positive down) and an offset added to the boresights
- **`ut`** *(optional)* — `[height, min distance, speed, bank, tilt, heading]` of the UTs in [m], [m], [m/s], [deg], [deg], [deg]; a value replaces the rule of the preset for all UTs (building UTs get the floor from a given height); min distance and speed >= 0 (with the d2D-out rule, a building UT of UMi and UMa needs d2D >= d2D-in also at min distance 0). Calibration orientations: TR Table 7.8-2 is tilt 90°, Table 7.8-2A configs B and C tilt 45°, config D tilt 0° and heading 0°
- **`indoor`** *(optional)* — `[indoor ratio, min floors, max floors, floor height]`: share of building UTs in [0, 1] (UMi, UMa, RMa, SMa); range of `N_fl`, whole numbers in [1, 1000] (UMi, UMa); floor height in [m], > 0 (UMi, UMa, SMa)
- **`options`** *(optional)* — One value per key: `p_residential` (SMa), the share of residential buildings in [0, 1]; other keys throw
- **`per_cell`** *(optional)* — true: `n_ut` UTs in each cell; false: `n_ut` UTs uniform over all cells
- **`seed`** *(optional)* — Seed >= 0; -1 = system random device

An entry of a tuple that is NaN or omitted takes the value of the preset; an entry or option that the preset does not use throws.

## Returns:
- The layout, see [[layout]]

## Example:
```
quadriga_lib::layout L = quadriga_lib::generate_layout("38.901_UMa", 10, {}, {}, {}, {}, {NAN, NAN, NAN, 0.0, 90.0}, {}, {}, true, 42);
quadriga_lib::set_antenna(L, "3gpp", {{"panel", {4, 4, 1, 2}}});
```

## See also:
- [[drop_terminals]] (the same drop for any Tx side)
- [[generate_track]] (tracks for the UTs)
- [[set_antenna]] (antennas of the nodes)
- [[layout]] (the members)
MD!*/

quadriga_lib::layout quadriga_lib::generate_layout(const std::string &type, arma::uword n_ut, const arma::vec &spacing,
                                                   const arma::vec &grid, const arma::vec &area, const arma::vec &bs, const arma::vec &ut,
                                                   const arma::vec &indoor, const std::unordered_map<std::string, std::vector<double>> &options,
                                                   bool per_cell, arma::sword seed)
{
    const qd_preset *p = nullptr;
    for (const qd_preset &q : qd_presets)
        if (qd_lower(type) == qd_lower(q.name))
            p = &q;
    if (p == nullptr)
    {
        std::string list;
        for (const qd_preset &q : qd_presets)
            list += (list.empty() ? "'" : ", '") + std::string(q.name) + "'";
        throw std::invalid_argument("Input 'type' must be a preset: " + list + " ('" + type + "').");
    }
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");
    qd_check_n_ut(n_ut, 0, false);

    const qd_geometry_args a{{spacing, "spacing", {"d_x", "d_y"}},
                             {grid, "grid", {"n_x", "n_y", "n_sector"}},
                             {area, "area", {"length", "width", "height"}},
                             {bs, "bs", {"height", "tilt", "azimuth offset"}},
                             options};
    a.spacing.check_size(), a.grid.check_size(), a.area.check_size(), a.bs.check_size();

    layout L;
    arma::vec bounds;
    p->build(*p, a, L, bounds);
    qd_check_n_ut(n_ut, L.tx.n_node(), per_cell);

    size_t n_site = 0;
    const std::vector<size_t> site = qd_layout::site_of_node(L.tx, &n_site);
    std::vector<qd_rules> rules(n_site, qd_rules_of(p->drop, p->family));
    const qd_ut_values o = qd_apply_args(rules, ut, indoor, options, qd_by_preset(*p));

    L.seed = qd_resolve_seed(seed);
    if (n_ut > 0)
        L.rx = qd_drop(L.tx, site, qd_make_cells(L.tx, site, n_site, L.wrap, bounds), rules, o, n_ut, per_cell, L.seed);
    return L;
}

/*!MD
# drop_terminals
Drop UTs into the cells of the Tx side of a layout

- Cells: the Voronoi regions of the sites (the Tx nodes of one name, at the position of their first row), with the wrap images when `layout.wrap` is set. A site with several sectors is split into wedges by the nearest sector heading (±60° around each boresight for three sectors at equal spacing)
- Drop area: with two wrap vectors, the wrapped cluster, i.e. the points nearest to image 0 of a site among the images 0, ±a1, ±a2 and ±(a1 − a2); else `bounds`, which is then required. Given bounds limit the drop in both cases
- The drop of [[generate_layout]] applies in the cells of each site, with the rules of the scenario family in its `scenario` (minimum distance, heights and floors, speeds, O2I states). Sites without a family (no scenario, a conf name): height 1.5 m, no minimum distance, speed 0, no O2I state; their UTs get `indoor` -1, `floor` -1 and `d2d_in` NaN when other UTs have states
- The Rx side is replaced by the UTs `UT1`, `UT2`, ... (ordered by cell with `per_cell`) with the generic antenna; `pairing` and `links` are emptied. The Tx side, the antenna lists and `layout.seed` stay. A failed call leaves the layout unchanged
- Random streams: the positions use the key `drop`, the heading and direction of a UT `ut|<name>`, its floor `floor|<family>|<name>`, the O2I fields `<purpose>|<family>`; the base of all is the seed of the call

## Declaration:
```
void quadriga_lib::drop_terminals(
    quadriga_lib::layout &layout,
    arma::uword n_ut,
    const arma::vec &bounds = {},
    const arma::vec &ut = {},
    const arma::vec &indoor = {},
    const std::unordered_map<std::string, std::vector<double>> &options = {},
    bool per_cell = true,
    arma::sword seed = -1);
```

## Inputs:
- **`layout`** — Valid layout, updated in place; the sectors of one site need different headings
- **`n_ut`** — UTs per cell (`per_cell`) or in total, at most 10<sup>7</sup> in total; 0 = no UTs
- **`bounds`** *(optional)* — `[x_min, x_max, y_min, y_max]` in [m], finite, `x_min < x_max`, `y_min < y_max`; required unless `layout.wrap` has two vectors
- **`ut`**, **`indoor`**, **`options`**, **`per_cell`** *(optional)* — As in [[generate_layout]]; an entry or option that no family of the sites uses throws
- **`seed`** *(optional)* — Seed >= 0; -1 = `layout.seed` if it is set, else the system random device

## See also:
- [[generate_layout]] (layouts from presets, with this drop)
- [[generate_track]] (tracks for the UTs)
MD!*/

void quadriga_lib::drop_terminals(layout &layout, arma::uword n_ut, const arma::vec &bounds, const arma::vec &ut, const arma::vec &indoor,
                                  const std::unordered_map<std::string, std::vector<double>> &options, bool per_cell, arma::sword seed)
{
    if (layout.tx.n_node() == 0)
        throw std::invalid_argument("Input 'layout' has no Tx nodes.");
    const std::string err = layout.is_valid();
    if (!err.empty())
        throw std::invalid_argument("Input 'layout' is not valid: " + err);
    if (!bounds.is_empty() && (bounds.n_elem != 4 || !bounds.is_finite() || !(bounds[0] < bounds[1]) || !(bounds[2] < bounds[3])))
        throw std::invalid_argument("Input 'bounds' must be empty or [x_min, x_max, y_min, y_max] with finite values, x_min < x_max and y_min < y_max.");
    if (bounds.is_empty() && layout.wrap.n_cols != 2)
        throw std::invalid_argument("Input 'bounds' is required: the layout has no two wrap vectors that bound the drop.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");
    qd_check_n_ut(n_ut, layout.tx.n_node(), per_cell);

    size_t n_site = 0;
    const std::vector<size_t> site = qd_layout::site_of_node(layout.tx, &n_site);
    std::vector<qd_rules> rules(n_site, qd_rules_of_scenario(""));
    if (!layout.tx.scenario.empty())
        for (arma::uword i = 0; i < layout.tx.n_node(); ++i)
            rules[site[i]] = qd_rules_of_scenario(layout.tx.scenario[layout.tx.offset[i]]);
    const qd_ut_values o = qd_apply_args(rules, ut, indoor, options, "the scenarios of the layout");

    const arma::sword s = seed >= 0 ? seed : qd_resolve_seed(layout.seed);
    layout_nodes R;
    if (n_ut > 0)
        R = qd_drop(layout.tx, site, qd_make_cells(layout.tx, site, n_site, layout.wrap, bounds), rules, o, n_ut, per_cell, s);
    layout.rx = std::move(R);
    layout.pairing.reset();
    layout.links.reset();
}
