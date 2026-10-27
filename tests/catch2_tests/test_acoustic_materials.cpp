// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_RAY=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

// The acoustic material table data/materials_acoustic.csv and the rig that measures it. A material case reads its row
// by name and makes one call (mt::check_material): the row as a sheet and as a traced solid of its thickness must be
// one wall; tier 1 against independent physics written here (the plate rule of the material-model page for mass-law
// rows, the transfer matrix of fluid layers for wave media); tier 2 against the measured rows of the literature, given
// with their sources. The rig's self-checks define their constructions inline. Plane waves throughout:
//   - stacks, room side first: sheets (one face, thickness in thk) in front, traced solids (two faces) behind, one sheet
//     glued at the last solid's exit; planes x = const of 400 m (the panel size enters only through side);
//   - transmission: calc_diffraction_gain, one straight path per angle (lod 0, scalar, threshold 0, no spreading); for
//     a box of air in front of a fill the paths of the reflection ray set with at most 10 interactions (the exchange
//     between box and fill is a series of paths);
//   - reflection: one aimed ray per angle, all angles in one ray set through ray_progress (scalar, refraction off,
//     threshold 0) until no ray is left; the paths back in air summed coherently per angle at their polyline's point;
//   - bands: absorption (Paris) and transmission averaged over 0-90 deg with the weight cos sin; a third the mean of its
//     tones, an octave the mean of its thirds (absorption arithmetic, transmission in energy), as the research data;
//   - frequencies: acoustic Hz times qd_acoustic_freq_scale (342.76271 m/s). The rows of page 11.4 were fitted at
//     343 m/s and are used unchanged (their octaves move by at most 0.017 dB).

#include "ray_test_helpers.hpp"
#include "qd_atmospheric_absorption.hpp"

#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

#include <cctype>
#include <cstring>
#include <numeric>
#include <random>
#include <regex>
#include <tuple>
#include <set>

namespace mt
{
    typedef std::complex<double> cd;
    typedef std::map<std::string, double> Row;
    typedef std::vector<std::vector<cd>> Field; // [angle][tone]
    static const double FS = qd_acoustic_freq_scale, CS = qd_speed_of_sound, RHO0 = 1.2; // Hz per Hz; m/s; kg/m3 of the mapping
    static const char *const COLS[26] = {"fRef", "a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB", "alpha", "alphaB", "m",
                                         "resF", "resQ", "resS", "coiF", "coiQ", "tf", "tfB", "tfP", "tfF", "tfQ", "side"};
    static double col(const Row &r, const char *k) // a column, or its default
    {
        const auto it = r.find(k);
        return it != r.end() ? it->second : (!std::strcmp(k, "fRef") || !std::strcmp(k, "a") || !std::strcmp(k, "e")) ? 1.0 : !std::strcmp(k, "side") ? 3.29 : 0.0;
    }
    static double pow_law(const Row &r, const char *v, const char *x, double fa) { return col(r, v) * std::pow(fa * FS * 1e-9 / col(r, "fRef"), col(r, x)); }
    static double dB(double p) { return p > 0.0 ? 10.0 * std::log10(p) : -999.0; }
    static double rad(double d) { return d * QPI / 180.0; }
    static Row with(Row r, const char *k, double v) { return r[k] = v, r; }

    // ---- the table: data/materials_acoustic.csv through obj_file_read, one row by name ----
    struct Table { std::vector<std::string> names; std::unordered_map<std::string, std::vector<double>> prop; };
    static const Table &table()
    {
        static Table T;
        if (T.names.empty())
            quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "data/materials_acoustic.csv",
                                                nullptr, &T.names, &T.prop);
        return T;
    }
    static Row row(const std::string &name)
    {
        const Table &T = table();
        const size_t i = (size_t)(std::find(T.names.begin(), T.names.end(), name) - T.names.begin());
        if (i == T.names.size())
            throw std::invalid_argument("mt::row: no material '" + name + "' in data/materials_acoustic.csv");
        Row r;
        for (const auto &kv : T.prop)
            r[kv.first] = kv.second[i];
        return r;
    }

    // ---- stacks, room side first ----
    enum Kind { SHEET, SOLID };
    struct Layer { Row row; Kind kind; double thk; };
    typedef std::vector<Layer> Stack;
    static bool mass(const Row &r) { return col(r, "m") > 0.0; }
    static bool lossless(const Row &r) { return !mass(r) && col(r, "c") == 0.0 && col(r, "g") == 0.0 && col(r, "alpha") == 0.0; }
    static bool cavity(const Row &r) { return lossless(r) && std::abs(col(r, "a") - 1.0) < 1e-3 && std::abs(col(r, "e") - 1.0) < 1e-3; } // e 1.000001
    static bool fill(const Layer &a, const Layer &b) // a box of air in front of a traced fill (not a dense leaf)
    {
        return a.kind == SOLID && b.kind == SOLID && cavity(a.row) && !mass(b.row) && col(b.row, "e") <= 10.0;
    }
    // The stack rule (measured in the plan): sheets on the room side, at most two and then with air behind (three
    // co-located faces: R up to 0.98 off), and one sheet glued at the last solid's exit; no two mass-law sheets on one
    // face (rule 9); no two traced wave media back to back (one straight path misses their exchange), except a box of
    // air in front of a non-dense fill; no lossless wave medium with a critical angle as a solid.
    static void stack_rule(const Stack &S)
    {
        auto no = [](const char *why) { throw std::invalid_argument(std::string("mt::stack: ") + why); };
        if (S.empty())
            no("empty");
        for (size_t i = 0, ns = 0; i < S.size(); ++i)
        {
            const Row &r = S[i].row, *p = i ? &S[i - 1].row : nullptr;
            if (!(S[i].thk > 0.0))
                no("a layer needs a thickness");
            const bool exit = S[i].kind == SHEET && p && S[i - 1].kind == SOLID && i + 1 == S.size();
            if (S[i].kind == SHEET && !exit && (i != ns++ || ns > 2 || (p && mass(r) && mass(*p))))
                no("a sheet between traced solids, more than two sheets, or two mass-law sheets on one face");
            if (S[i].kind == SOLID && ns == 2)
                no("three co-located faces (two sheets on a solid)");
            if (S[i].kind == SOLID && lossless(r) && !cavity(r) && col(r, "a") * col(r, "e") < 1.0)
                no("a lossless wave medium with a critical angle as a traced solid (use a sheet)");
            if (S[i].kind == SOLID && p && S[i - 1].kind == SOLID && !mass(r) && !mass(*p) && !fill(S[i - 1], S[i]))
                no("two traced wave media back to back");
        }
    }
    // The library's own geometry, 400 m square faces (the panel size enters only through side): a sheet is
    // quadriga_lib::plane, its +z normal turned to -x (the room), in the plane x = 0 in stack order (the first face is
    // the room side); a solid is a closed quadriga_lib::cube of its thickness behind the one before; an exit sheet is a
    // plane at the last solid's back face (the engine takes the order from the solid's face: either normal gives the
    // same numbers); layer k is material k, solids with thk 0.
    struct Scene { arma::fmat mesh; arma::uvec mi; rp::Pal pal; float x_end = 0.0f; };
    static Scene scene(const Stack &st)
    {
        stack_rule(st);
        Scene S;
        auto add = [&](const arma::fmat &m, size_t k)
        { S.mesh = arma::join_cols(S.mesh, m), S.mi = arma::join_cols(S.mi, arma::uvec(m.n_rows, arma::fill::value(k + 1))); };
        const size_t ex = st.back().kind == SHEET && st.size() > 1 && st[st.size() - 2].kind == SOLID; // an exit sheet
        for (size_t i = 0; i + ex < st.size(); ++i)
            if (st[i].kind == SHEET)
                add(quadriga_lib::plane<float>({200.0, 200.0}, {0.0, -0.5 * QPI, 0.0}, {0.0, 0.0, 0.0}), i);
        for (size_t i = 0; i < st.size(); ++i)
            if (st[i].kind == SOLID)
                add(quadriga_lib::cube<float>({0.5 * st[i].thk, 200.0, 200.0}, {0.0, 0.0, 0.0}, {S.x_end + 0.5 * st[i].thk, 0.0, 0.0}), i),
                    S.x_end += (float)st[i].thk;
        if (ex)
            add(quadriga_lib::plane<float>({200.0, 200.0}, {0.0, -0.5 * QPI, 0.0}, {(double)S.x_end, 0.0, 0.0}), st.size() - 1);
        for (const char *k : COLS)
            for (const Layer &l : st)
                S.pal[k].push_back((float)(std::strcmp(k, "thk") ? col(l.row, k) : (l.kind == SHEET ? l.thk : 0.0)));
        return S;
    }

    // ---- the grid: averaging angles (weights cos sin), then the report angles (weight 0); tones, 125 Hz first ----
    static const double THIRDS[27] = {25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                                      5000, 6300, 8000, 10000};
    static const double OCT[9] = {31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000};
    struct Grid { std::vector<double> th, w, f; std::vector<int> third; int na = 0; }; // deg, weights, Hz, index into THIRDS
    static Grid grid(int n_angles = 50, int per_third = 1, bool report_only = false)
    {
        Grid G;
        G.na = report_only ? 0 : n_angles;
        for (int i = 0; i < G.na; ++i)
            G.th.push_back((i + 0.5) * 90.0 / n_angles), G.w.push_back(std::sin(2.0 * rad(G.th.back())));
        const double s = std::accumulate(G.w.begin(), G.w.end(), 0.0);
        for (double &v : G.w)
            v /= s;
        for (double t : {0.0, 5.0, 10.0, 30.0, 45.0, 60.0, 70.0, 80.0})
            G.th.push_back(t), G.w.push_back(0.0);
        for (int t = 0; t < 27; ++t)
            for (int j = 0; j < per_third; ++j)
                G.f.push_back(THIRDS[t] * std::pow(2.0, ((j + 0.5) / per_third - 0.5) / 3.0)), G.third.push_back(t);
        size_t b = 0;
        for (size_t q = 0; q < G.f.size(); ++q)
            b = std::abs(std::log(G.f[q] / 125.0)) < std::abs(std::log(G.f[b] / 125.0)) ? q : b;
        std::swap(G.f[0], G.f[b]), std::swap(G.third[0], G.third[b]);
        return G;
    }

    // ---- the library ----
    // The ray of angle i lies in the plane z = -0.005 - 0.01 i: off the faces' diagonals (z = -y at the front faces,
    // z = y at the back faces of a cube), its hit point (0, 0.37, z) names it
    static float zi(size_t i) { return -0.005f - 0.01f * (float)i; }
    // Transmission: calc_diffraction_gain, one straight path per angle, one call per tone
    static Field transmit(const Scene &S, const Grid &G)
    {
        const size_t n = G.th.size();
        arma::fmat o(n, 3), d(n, 3);
        for (size_t i = 0; i < n; ++i)
        {
            const double c = std::cos(rad(G.th[i])), s = std::sin(rad(G.th[i])), L = (S.x_end + 5.0) / std::max(c, 1e-6);
            o.row(i) = arma::frowvec{(float)(-5.0 * c), (float)(0.37 - 5.0 * s), zi(i)}, d.row(i) = arma::frowvec{(float)(L * c), (float)(0.37 + L * s), zi(i)};
        }
        Field T(n, std::vector<cd>(G.f.size()));
        for (size_t q = 0; q < G.f.size(); ++q)
        {
            arma::fvec g;
            arma::fmat x; // [2, n]: the normalized coefficient; the power is in g
            quadriga_lib::calc_diffraction_gain<float>(o, d, S.mesh, S.mi, S.pal, (float)(G.f[q] * FS), 0, &g, &x, nullptr, 0, nullptr, 0, 0, true, 0.0);
            for (size_t i = 0; i < n; ++i)
                T[i][q] = std::polar(std::sqrt((double)g(i)), std::arg(cd(x(0, i), x(1, i))));
        }
        return T;
    }
    // Reflection: one aimed ray per angle from 0.9 m in front of its hit point, every angle in one ray set, ray_progress
    // until no ray is left (at most 127 tones per band, the first its base); the paths back in air (an odd number of
    // reflections) summed coherently per angle at their polyline's point; on request the paths that leave behind the
    // stack (an even number) with at most 10 interactions into T, referred to the plane wave going on. R is uncapped
    // (the engine's 60 interactions) unless cap is given; only T behind a filled box is taken at the cap of 10.
    static Field reflect(const Scene &S, const Grid &G, int &calls, long &rays, Field *T = nullptr, int cap = 200)
    {
        const size_t n = G.th.size(), nf = G.f.size();
        Field R(n, std::vector<cd>(nf, 0.0));
        if (T)
            T->assign(n, std::vector<cd>(nf, 0.0));
        calls = 0, rays = 0;
        for (size_t c0 = 0, m; c0 < nf; c0 += m)
        {
            m = std::min((size_t)127, nf - c0);
            arma::fmat orig(n, 3), dest(n, 3), acc(n, 1 + m, arma::fill::zeros);
            arma::fcube pd(n, 3, m, arma::fill::zeros);
            arma::fvec cf(m);
            std::vector<quadriga_lib::path> paths(n);
            for (size_t q = 0; q < m; ++q)
                cf(q) = (float)(G.f[c0 + q] * FS);
            for (size_t i = 0; i < n; ++i)
            {
                const float c = (float)std::cos(rad(G.th[i])), s = (float)std::sin(rad(G.th[i]));
                orig.row(i) = arma::frowvec{-0.9f * c, 0.37f - 0.9f * s, zi(i)}, dest.row(i) = arma::frowvec{299.0f * c, 0.37f + 299.0f * s, zi(i)};
                pd.tube(i, 0, i, 0).fill(c), pd.tube(i, 1, i, 1).fill(s), paths[i].init(0, m, true);
            }
            arma::Col<short> prev(n, arma::fill::zeros), cur(n, arma::fill::zeros), buf(n, arma::fill::zeros);
            for (int g = 0; g < cap && orig.n_rows; ++g)
            {
                quadriga_lib::ray_progress(S.mesh, S.mi, S.pal, cf, -1.0f, 0.37f, 0.0f, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr,
                                           nullptr, nullptr, 60, 255, 255, 0, -120.0f, 3.0f, 0.0f, false, true);
                ++calls, rays += (long)orig.n_rows;
                for (arma::uword i = 0; i < orig.n_rows; ++i)
                {
                    const quadriga_lib::path &p = paths[i];
                    const size_t ns = p.n_seg(), k = (size_t)std::lround((-p.coord(0)[2] - 0.005f) / 0.01f);
                    const float *a = p.coord(0), *b = p.coord(ns - 1);
                    const bool back = p.nREF % 2 == 0, on = !back || (T && g < 10 && b[0] > S.x_end - 1e-3f && dest(i, 0) > b[0]);
                    if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1) || !on || k >= n)
                        continue; // in air, extended by this call, turned back (or gone on behind)
                    // the polyline from the first interaction on: the path's length less its launch leg from the origin
                    const double L = (double)p.length() - std::sqrt(std::pow(a[0] + 1.0, 2) + std::pow(a[1] - 0.37, 2) + std::pow((double)a[2], 2));
                    const double x = back ? std::cos(rad(G.th[k])) * ((double)b[0] - a[0]) : 0.0;
                    for (size_t q = 0; q < m; ++q)
                        (back ? (*T)[k][c0 + q] : R[k][c0 + q]) += cd(p.xpr_coeff(q)[0], p.xpr_coeff(q)[1]) *
                                        std::exp(cd(0.0, 2.0 * QPI * cf(q) / C0 * (x + std::sin(rad(G.th[k])) * ((double)b[1] - a[1]) - L)));
                }
            }
        }
        return R;
    }
    // A stack on a grid: R and T per node, random-incidence absorption and transmission per tone and per octave
    struct Out { Grid g; Field R, T; std::vector<double> af, tauf; double a8[9], t8[9]; int calls = 0; long rays = 0; };
    template <class F>
    static std::array<double, 9> octaves(const Grid &G, F v) // the mean of a per-tone quantity over each octave's tones
    {
        std::array<double, 9> s{}, n{};
        for (size_t q = 0; q < G.f.size(); ++q)
            s[G.third[q] / 3] += v(q), n[G.third[q] / 3] += 1.0;
        for (int o = 0; o < 9; ++o)
            s[o] /= std::max(n[o], 1.0);
        return s;
    }
    static Out bands(Out O) // random-incidence absorption and transmission per tone and per octave
    {
        const Grid &G = O.g;
        O.af.assign(G.f.size(), 0.0), O.tauf.assign(G.f.size(), 0.0);
        for (size_t q = 0; q < G.f.size(); ++q)
            for (size_t i = 0; i < G.th.size(); ++i)
                O.af[q] += G.w[i] * (1.0 - std::norm(O.R[i][q])), O.tauf[q] += G.w[i] * std::norm(O.T[i][q]);
        const auto a = octaves(G, [&](size_t q) { return O.af[q]; }), t = octaves(G, [&](size_t q) { return O.tauf[q]; });
        for (int o = 0; o < 9; ++o)
            O.a8[o] = a[o], O.t8[o] = -10.0 * std::log10(t[o]);
        return O;
    }
    static Out run(const Stack &st, const Grid &G)
    {
        const Scene S = scene(st);
        Out O;
        const bool bf = std::adjacent_find(st.begin(), st.end(), fill) != st.end();
        O.g = G, O.T = bf ? Field() : transmit(S, G), O.R = reflect(S, G, O.calls, O.rays, bf ? &O.T : nullptr);
        return bands(O);
    }

    // ---- tier 1: independent physics ----
    // The transfer matrix of fluid layers (lw::stack of test_acoustic_leaf.cpp): (p, v_n), exp(+j w t), impedances
    // relative to rho0 c0 at the sine of air; behind the layers air, a fluid half-space behind a matched pad that absorbs
    // tf of what arrives from air, or a limp plate of surface mass mpp (kg/m2) with air behind it, radiating as an
    // infinite plate or (side > 0) as a finite panel, Davy's sigma in place of 1 / cos.
    enum Backing { AIR, HALF, PLATE };
    // A layer's columns, or a porous model at the flow resistivity sigma (Pa s/m2) named in the row as "miki"
    // [Miki_1990_ModificationsDelanyBazley], "komatsu" [Komatsu_2008_ImprovementDelanyBazley] or "db" (Delany and Bazley
    // 1970, printed only): z = Zc / rho0 c0, n = k / k0, eps = n / z, mu = n z
    static void eps_mu(const Row &M, double fa, cd &e, cd &m)
    {
        const cd J(0.0, 1.0);
        const double sm = col(M, "miki"), sk = col(M, "komatsu"), sd = col(M, "db"), L = 2.0 - std::log10(fa / sk), X = RHO0 * fa / sd;
        cd n = 0.0, z = 1.0;
        if (sm > 0.0)
            z = 1.0 + (0.0699 - 0.107 * J) * std::pow(fa / sm, -0.632), n = 1.0 + (0.109 - 0.160 * J) * std::pow(fa / sm, -0.618);
        else if (sk > 0.0)
            z = 1.0 + 0.00027 * std::pow(L, 6.2) - J * 0.0047 * std::pow(L, 4.1), n = 1.0 + 0.0004 * std::pow(L, 6.2) - J * 0.0069 * std::pow(L, 4.1);
        else if (sd > 0.0)
            z = 1.0 + 0.0571 * std::pow(X, -0.754) - J * 0.087 * std::pow(X, -0.732), n = 1.0 + 0.0978 * std::pow(X, -0.700) - J * 0.189 * std::pow(X, -0.595);
        e = n != 0.0 ? n / z : cd(pow_law(M, "a", "b", fa), -17.98 * pow_law(M, "c", "d", fa) / (fa * FS * 1e-9));
        m = n != 0.0 ? n * z : cd(pow_law(M, "e", "f", fa), -17.98 * pow_law(M, "g", "h", fa) / (fa * FS * 1e-9));
    }
    // Davy's forced radiation efficiency of a square panel of side `side` at the cosine c [Davy_2009_ForcedRadiationEfficiency]
    static double davy(double fa, double c, double side)
    {
        const double k = 2.0 * QPI * fa / CS, ka = std::max(0.5 * k * side, 0.1), cc = std::min(c, 1.0);
        const double p = std::min(1.0, 1.3 * std::sqrt(QPI / (2.0 * ka))), h = 1.0 / (2.0 / 3.0 * std::sqrt(2.0 * ka / QPI) - 0.124);
        const double q = 2.0 * QPI / (k * k * side * side), x = cc >= p ? cc : h - (h / p - 1.0) * cc;
        return 1.0 / std::sqrt(x * x + q * q);
    }
    static cd q_of(cd eps, cd mu, double s) // the normal wavenumber over k0: the root that decays or carries the wave away
    {
        const cd q = std::sqrt(eps * mu - s * s);
        return (q.imag() > 0.0) ? -q : q;
    }
    static void ref_tmm(const Stack &L, double fa, double thd, double &T, double &R, Backing bk = AIR, const Row *half = nullptr, double mpp = 0.0, double side = 0.0)
    {
        const cd J(0.0, 1.0);
        const double s = std::sin(rad(thd)), Za = 1.0 / std::cos(rad(thd));
        cd A = 1.0, B = 0.0, C = 0.0, D = 1.0, e, m;
        auto mul = [&](cd a, cd b, cd c, cd d) { const cd A2 = A * a + B * c, B2 = A * b + B * d, C2 = C * a + D * c; D = C * b + D * d, A = A2, B = B2, C = C2; };
        for (const Layer &l : L)
        {
            eps_mu(l.row, fa, e, m);
            const cd q = q_of(e, m, s), z = m / q, ph = 2.0 * QPI * fa / CS * q * l.thk;
            mul(std::cos(ph), J * z * std::sin(ph), J * std::sin(ph) / z, std::cos(ph));
        }
        if (bk == PLATE) // a pressure jump j w m'' v_n at every angle
            mul(1.0, J * 2.0 * QPI * fa * mpp / (RHO0 * CS), 0.0, 1.0);
        if (bk != HALF)
        {
            const double Zb = bk == PLATE && side > 0.0 ? davy(fa, std::cos(rad(thd)), side) : Za; // the radiating side
            const cd vt = 2.0 / (A * Zb + B + Za * (C * Zb + D));
            R = std::norm((A * Zb + B) * vt - 1.0), T = Za * Zb * std::norm(vt);
            return;
        }
        eps_mu(*half, fa, e, m);
        const double tf = std::max(0.0, std::min(1.0, pow_law(*half, "tf", "tfB", fa)));
        const cd Zh = m / q_of(e, m, s), g = std::sqrt(1.0 - tf) * (Zh - Za) / (Zh + Za), Zt = Za * (1.0 + g) / (1.0 - g), Zs = (A * Zt + B) / (C * Zt + D);
        R = std::norm((Zs - Za) / (Zs + Za)), T = -1.0; // no transmission reference into the half-space
    }
    // The plate rule of the material-model page for a bonded run of mass-law layers in air (5.3, 5.4, 6.3, 6.5): the
    // bodies' X0 (resonance and bending factors included) added, the mass term with Davy's forced radiation efficiency
    // of a square of side `side` [Davy_2009_ForcedRadiationEfficiency], T = a tau / (a + tau - a tau) times each
    // body's entry loss, a the smaller absorption of the outer faces (the pad of tf in front of the Fresnel face of the
    // mapping, which reflects totally past the face's critical angle), R = 1 - a of the near face.
    static double mass_TL(double X, double c, double fa, double side)
    {
        if (!(X > 1.0) || !(c > 0.0))
            return 0.0;
        const double A = 20.0 * std::log10(X);
        if (!(side > 0.0))
            return c >= 1.0 ? A : A * std::log10(1.0 + X * c * X * c) / std::log10(1.0 + X * X);
        const double sg = davy(fa, c, side);
        return std::max(0.0, A * std::log10(1.0 + (X / sg) * (X / sg)) / std::log10(1.0 + X * X) + 10.0 * std::log10(sg * std::min(c, 1.0)));
    }
    static void ref_plate(const Stack &L, double fa, double thd, double &T, double &R)
    {
        const double c = std::cos(rad(thd)), fG = fa * FS * 1e-9, s2 = 1.0 - c * c;
        double X = 0.0, g = 1.0, a[2];
        for (const Layer &l : L)
        {
            const Row &M = l.row;
            const double F = col(M, "resF"), Q = col(M, "resQ"), cF = col(M, "coiF"), cQ = col(M, "coiQ"), y = F / fG, xr = fG * c / F, w = std::pow(fG / cF * s2, 2);
            double x = fG / col(M, "fRef") * l.thk, att = pow_law(M, "att", "attB", fa);
            if (F > 0 && Q > 0 && col(M, "resS") != 0.0)
                x *= col(M, "resS") > 0 ? std::min(1.0, std::abs(cd(1.0 - xr * xr, xr / Q))) : std::abs(cd(1.0 - y * y, -y / Q));
            if (cF > 0 && cQ > 0)
                x *= std::abs(cd(1.0 - w, -w / cQ));
            X += x, g *= att > 0.0 ? std::pow(10.0, -0.1 * att) : 1.0;
        }
        for (int k = 0; k < 2; ++k)
        {
            const Row &M = k ? L.back().row : L.front().row;
            const double nn = col(M, "a") * col(M, "e"), c2 = std::sqrt(std::max(0.0, 1.0 - s2 / nn)), z2 = std::sqrt(col(M, "a") / col(M, "e"));
            a[k] = 1.0 - (1.0 - std::max(0.0, std::min(1.0, pow_law(M, "tf", "tfB", fa)))) * (s2 < nn ? std::pow((c - z2 * c2) / (c + z2 * c2), 2) : 1.0);
        }
        const double am = std::min(a[0], a[1]), tau = std::pow(10.0, -0.1 * mass_TL(X, c, fa, col(L.front().row, "side")));
        T = (am > 0.0 && tau > 0.0) ? am * tau / (am + tau - am * tau) * g : 0.0, R = 1.0 - a[0];
    }
    // The reference of a stack: the plate rule (all mass-law), the fluid layers in air (all wave media), or the wave
    // layers on the half-space of the first mass-law layer behind its pad (R only); none for a wave layer behind a
    // mass-law layer
    static void ref_auto(const Stack &L, double fa, double thd, double &T, double &R)
    {
        const auto k = std::find_if(L.begin(), L.end(), [](const Layer &l) { return mass(l.row); });
        if (std::any_of(k, L.end(), [](const Layer &l) { return !mass(l.row); }))
            throw std::invalid_argument("mt::ref_auto: no reference for a wave layer behind a mass-law layer");
        if (k == L.end())
            ref_tmm(L, fa, thd, T, R);
        else if (k == L.begin())
            ref_plate(L, fa, thd, T, R);
        else
            ref_tmm(Stack(L.begin(), k), fa, thd, T, R, HALF, &k->row);
    }

    // ---- per node against the reference (largest |dT| in dB where both are above 1e-25, n/a without a T reference;
    // |dR|; R + T), and the twin ----
    struct Dev { double dT = 0.0, dR = 0.0, RT = 0.0, fT = 0.0, aT = 0.0, fR = 0.0, aR = 0.0; bool hasT = false; };
    static Dev against(const char *name, const Out &O, const Stack &st)
    {
        Dev D;
        for (size_t q = 0; q < O.g.f.size(); ++q)
            for (size_t i = 0; i < O.g.th.size(); ++i)
            {
                double Tr, Rr, Tl = std::norm(O.T[i][q]), Rl = std::norm(O.R[i][q]);
                ref_auto(st, O.g.f[q], O.g.th[i], Tr, Rr);
                D.RT = std::max(D.RT, Tl + Rl), D.hasT |= Tr >= 0.0;
                if (!std::isfinite(Tl + Rl)) // a NaN or inf fails every bound
                    D.dT = D.dR = D.RT = INFINITY;
                if (Tr > 1e-25 && Tl > 1e-25 && std::abs(dB(Tl) - dB(Tr)) > D.dT)
                    D.dT = std::abs(dB(Tl) - dB(Tr)), D.fT = O.g.f[q], D.aT = O.g.th[i];
                if (std::abs(Rl - Rr) > D.dR)
                    D.dR = std::abs(Rl - Rr), D.fR = O.g.f[q], D.aR = O.g.th[i];
            }
        if (D.hasT)
            QD_DIAG("%s,%zu,%.2e,%g,%.1f,%.2e,%g,%.1f,%.8f,%d,%ld\n", name, O.g.th.size(), D.dT, D.fT, D.aT, D.dR, D.fR, D.aR, D.RT, O.calls, O.rays);
        else
            QD_DIAG("%s,%zu,n/a,n/a,n/a,%.2e,%g,%.1f,%.8f,%d,%ld\n", name, O.g.th.size(), D.dR, D.fR, D.aR, D.RT, O.calls, O.rays);
        return D;
    }
    static const char *DEV_HEAD = "construction,angles,max_dT_dB,at_Hz,at_deg,max_dR,at_Hz,at_deg,max_R+T,calls,rays\n";

    // ---- diagnostics: per report angle and octave center R and T in dB and phase; random-incidence octaves ----
    static void rows(const char *name, const Out &O)
    {
        if (!qd_diag())
            return;
        std::printf("\n# diag: %s, per angle at the octave centers: R and T in dB, phase in deg (plane wave)\ntheta,quantity", name);
        for (double f : OCT)
            std::printf(",%g", f);
        for (size_t i = O.g.na; i < O.g.th.size(); ++i)
            for (int k = 0; k < 4; ++k)
            {
                static const char *Q[4] = {"R_dB", "R_deg", "T_dB", "T_deg"};
                std::printf("\n%.0f,%s", O.g.th[i], Q[k]);
                for (double fo : OCT) // the tone at the octave center, where the grid has one
                {
                    const size_t q = (size_t)(std::find(O.g.f.begin(), O.g.f.end(), fo) - O.g.f.begin());
                    const cd v = q < O.g.f.size() ? (k < 2 ? O.R : O.T)[i][q] : cd(std::nan(""));
                    std::printf(",%.3f", k % 2 ? std::arg(v) * 180.0 / QPI : dB(std::norm(v)));
                }
            }
        std::printf("\n# diag: %s, random incidence (0-90 deg, %d angles, %zu tones): alpha and tl (dB) per octave\nquantity", name, O.g.na, O.g.f.size());
        for (double f : OCT)
            std::printf(",%g", f);
        std::printf("\nalpha");
        for (double v : O.a8)
            std::printf(",%.4f", v);
        std::printf("\ntl_dB");
        for (double v : O.t8)
            std::printf(",%.2f", v);
        std::printf("\n");
    }

    // ---- tier 2: a measured row (source key, quantity, coefficient type) at the octaves it states; a gap band takes an
    // interval (the model spread); the row evaluated at the specimen's side, a transmission reference shifted by the
    // mass law to the row's surface mass (20 log10 of the ratio), random incidence or 0 deg; asserted within the
    // guideline tol, deviations printed. A regression guard is a Measured row of the library's own octaves. ----
    struct Measured
    {
        const char *key, *quantity, *coefficient_type;
        bool tl; // transmission loss in dB, else absorption
        double tol;
        std::vector<double> octaves_hz;
        std::vector<std::pair<double, double>> lo_hi;
        double side = 0.0, mpp = 0.0, angle = -1.0; // specimen side (0: the row's), surface mass (0: no shift), 0 deg or random (-1)
    };
    // The surface mass of a row at a thickness: e rho0 thk, or the row's own "mpp" (a lumped row's stated surface mass)
    static double mpp(const Row &r, double thk) { return r.count("mpp") ? r.at("mpp") : col(r, "e") * RHO0 * thk; }
    // A class envelope: walls (surface mass, octave values in dB) each shifted to the row's surface mass by the mass law,
    // asserted within 1 dB of the band they span (wish-list 1.10)
    static Measured envelope(const char *key, const std::vector<double> &oct, const std::vector<std::pair<double, std::vector<double>>> &walls, double m_row, double side)
    {
        Measured M{key, "R_dB", "class envelope, mass-shifted", true, 1.0, oct, {}, side};
        for (size_t k = 0; k < oct.size(); ++k)
        {
            double lo = 1e9, hi = -1e9;
            for (const auto &w : walls)
                lo = std::min(lo, w.second.at(k) + 20.0 * std::log10(m_row / w.first)), hi = std::max(hi, w.second.at(k) + 20.0 * std::log10(m_row / w.first));
            M.lo_hi.push_back({lo, hi});
        }
        return M;
    }
    static void tier2(const char *name, const Out &O, const Measured &M, double m_row)
    {
        QD_DIAG("\n# diag: %s, tier 2 against %s (%s, %s; side %.4g m, %s, mass shift %+.3f dB): octave, library, reference, deviation outside the reference\noctave,library,ref_lo,ref_hi,deviation\n",
                name, M.key, M.quantity, M.coefficient_type, M.side, M.angle < 0.0 ? "random" : "0 deg", M.tl && M.mpp > 0.0 ? 20.0 * std::log10(m_row / M.mpp) : 0.0);
        REQUIRE(M.angle <= 0.0); // the grid's first report angle is 0 deg
        const auto a0 = octaves(O.g, [&](size_t q) { return M.tl ? std::norm(O.T[O.g.na][q]) : 1.0 - std::norm(O.R[O.g.na][q]); });
        const double sh = M.tl && M.mpp > 0.0 ? 20.0 * std::log10(m_row / M.mpp) : 0.0;
        for (size_t k = 0; k < M.octaves_hz.size(); ++k)
        {
            const int o = (int)(std::find(OCT, OCT + 9, M.octaves_hz[k]) - OCT);
            REQUIRE(o < 9);
            const double v = M.angle < 0.0 ? (M.tl ? O.t8[o] : O.a8[o]) : (M.tl ? -10.0 * std::log10(a0[o]) : a0[o]);
            const double lo = M.lo_hi[k].first + sh, hi = M.lo_hi[k].second + sh, dev = v > hi ? v - hi : (v < lo ? v - lo : 0.0);
            QD_DIAG("%g,%.4f,%.4f,%.4f,%+.4f\n", M.octaves_hz[k], v, lo, hi, dev);
            INFO(name << ", " << M.key << ", octave " << M.octaves_hz[k] << " Hz");
            CHECK((std::isfinite(v) && std::abs(dev) <= M.tol));
        }
    }

    // ---- one call per isolated material (a row with thk 0 at the thickness t). The row as a traced solid on the full
    // grid and as a sheet on the report angles (the twin, per node): tier 1 for both, the twin, R + T <= 1; the
    // diagnostic tables; for a mass-law row the bare limp mass law of e rho0 thk with Davy's sigma at its side
    // [London_1949_SingleWalls]; tier 2 for every measured row given. Bounds at or above 2-3 x the measured maxima, by
    // type: mass-law rows; dense leaves (mu' > 10: 2.6e-2 dB, 2.0e-3 at 89.1 deg); porous layers and half-spaces (rock
    // wool at 0.5 m: tier 1 3.0e-3 in R, the twin 2.7e-3 dB and 1.1e-3); wave sheets (6.7e-6 dB, 7.8e-8).
    static Out check_material(const std::string &name, const Row &row, const std::vector<Measured> &refs = {}, const Grid &G = grid(), double t = 0.0)
    {
        Row r = row;
        if (!(col(r, "thk") > 0.0))
            r["thk"] = t;
        const double thk = col(r, "thk");
        REQUIRE(thk > 0.0);
        const bool ml = mass(r), dense = !ml && col(r, "e") > 10.0;
        const Stack sheet = {{r, SHEET, thk}}, solid = {{r, SOLID, thk}};
        const Out X = run(solid, G), S = run(sheet, grid(G.na, (int)(G.f.size() / 27), true));
        QD_DIAG("\n# diag: %s per node against the %s\n%s", name.c_str(), ml ? "plate rule" : "fluid layer", DEV_HEAD);
        const Dev DX = against((name + " solid").c_str(), X, solid), DS = against((name + " sheet").c_str(), S, sheet);
        double dT = 0.0, dR = 0.0;
        for (size_t q = 0; q < G.f.size(); ++q)
            for (size_t i = 0, off = X.g.th.size() - S.g.th.size(); i < S.g.th.size(); ++i)
            {
                const double ta = std::norm(X.T[i + off][q]), tb = std::norm(S.T[i][q]);
                dT = (ta > 1e-25 && tb > 1e-25) ? std::max(dT, std::abs(dB(ta) - dB(tb))) : dT;
                const double ra = std::norm(X.R[i + off][q]), rb = std::norm(S.R[i][q]);
                dR = std::isfinite(ta + tb + ra + rb) ? std::max(dR, std::abs(ra - rb)) : INFINITY; // a NaN fails
            }
        QD_DIAG("# %s, sheet against solid on the report angles: dT %.2e dB, dR %.2e\n", name.c_str(), dT, dR);
        INFO(name);
        CHECK(DX.dT < (ml ? 2e-4 : dense ? 0.06 : 6e-3));
        CHECK(DX.dR < (ml ? 3e-7 : dense ? 5e-3 : 9e-3));
        CHECK(DS.dT < (ml ? 2e-4 : 3e-5));
        CHECK(DS.dR < 3e-7);
        CHECK(std::max(DX.RT, DS.RT) <= 1.0 + 1e-6);
        CHECK(dT < (ml ? 2e-4 : dense ? 5e-4 : 8e-3));
        CHECK(dR < (ml ? 1e-6 : dense ? 2e-5 : 3.5e-3));
        rows((name + ", traced solid").c_str(), X);
        if (ml && qd_diag())
        {
            const double mpp = mt::mpp(r, thk);
            const auto bare = octaves(G, [&](size_t q)
                                      { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * std::pow(10.0, -0.1 * mass_TL(QPI * G.f[q] * mpp / (RHO0 * CS), std::cos(rad(G.th[i])), G.f[q], col(r, "side"))); return t; });
            std::printf("# diag: %s, physical models: bare limp mass law of %.1f kg/m2 at side %.2f m, random incidence (dB)\noctave,library_tl,bare_mass_law_tl\n",
                        name.c_str(), mpp, col(r, "side"));
            for (int o = 0; o < 9; ++o)
                std::printf("%g,%.2f,%.2f\n", OCT[o], X.t8[o], -10.0 * std::log10(bare[o]));
        }
        for (const Measured &M : refs) // at the specimen's side (transmission only)
            tier2(name.c_str(), M.side > 0.0 && M.side != col(r, "side") ? bands([&] { Out Y = X; Y.T = transmit(scene({{with(r, "side", M.side), SOLID, thk}}), G); return Y; }()) : X, M, mpp(r, thk));
        return X;
    }
    // A combination: the stack on a grid per node against ref_auto, bounds on |dT| (dB) and |dR|, R + T <= 1
    static Out check_stack(const char *name, const Stack &st, double bT, double bR, const Grid &G = grid())
    {
        const Out O = run(st, G);
        const Dev D = against(name, O, st);
        INFO(name);
        if (D.hasT) // bT NaN where the reference has no T
            CHECK(D.dT < bT);
        CHECK(D.dR < bR);
        CHECK(D.RT <= 1.0 + 1e-6);
        return O;
    }
    // The insulation a layer adds to a wall (with minus without, so the wall's fitted knobs cancel), library against
    // the fluid layer on a limp plate of the wall's surface mass (e rho0 thk), at 0 deg and at random incidence, the
    // plate infinite and finite (Davy's sigma at the wall's side); and the layer alone in air (asserted against the
    // fluid layer; it is also the series estimate, the layer's loss paid on top of the wall's). Printed, not asserted
    // (engine stage-2b issue 1).
    static void added_insulation(const char *name, const Stack &layer, const Stack &wall, const Grid &G = grid(), double bT = 3e-5, double bR = 3e-7)
    {
        Stack both = layer;
        both.insert(both.end(), wall.begin(), wall.end());
        const Out A = run(both, G), B = run(wall, G), W = run(layer, G);
        const double mpp = mt::mpp(wall[0].row, wall[0].thk);
        auto plate = [&](const Stack &L, size_t q, double th, double side = 0.0) { double T, R; ref_tmm(L, G.f[q], th, T, R, PLATE, nullptr, mpp, side); return T; };
        auto at0 = [&](const Out &O) { return octaves(G, [&](size_t q) { return std::norm(O.T[G.na][q]); }); }; // the report angle 0 deg
        auto ri = [&](const Out &O) { return octaves(G, [&](size_t q) { return O.tauf[q]; }); };
        auto pl0 = [&](const Stack &L) { return octaves(G, [&](size_t q) { return plate(L, q, 0.0); }); };
        auto plr = [&](const Stack &L, double side = 0.0) { return octaves(G, [&](size_t q) { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * plate(L, q, G.th[i], side); return t; }); };
        QD_DIAG("# diag: the insulation %s adds (dB): library and the fluid layer on a limp plate of %.1f kg/m2\nincidence,quantity", name, mpp);
        for (double f : OCT)
            QD_DIAG(",%g", f);
        const std::array<double, 9> one = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const double sd = col(wall[0].row, "side");
        const std::vector<std::tuple<const char *, const char *, std::array<double, 9>, std::array<double, 9>>> L = {
            {"0 deg", "library", at0(A), at0(B)}, {"0 deg", "plate_tmm", pl0(layer), pl0({})}, {"random", "library", ri(A), ri(B)},
            {"random", "plate_tmm", plr(layer), plr({})}, {"0 deg", "layer_alone_library", at0(W), one}, {"random", "layer_alone_library", ri(W), one},
            {"random", "plate_tmm_finite", plr(layer, sd), plr({}, sd)}};
        for (const auto &l : L)
        {
            QD_DIAG("\n%s,%s", std::get<0>(l), std::get<1>(l));
            for (int o = 0; o < 9; ++o)
                QD_DIAG(",%.2f", 10.0 * std::log10(std::get<3>(l)[o] / std::get<2>(l)[o]));
        }
        QD_DIAG("\n%s", DEV_HEAD);
        const Dev D = against((std::string(name) + ": the layer alone in air").c_str(), W, layer);
        CHECK(D.dT < bT);
        CHECK(D.dR < bR);
    }
    // A lumped row at k times its thickness: thk and fRef times k, every power law's value at fRef times k^exponent
    // (wish-list 1.7.2), so every column is unchanged at every frequency; resF, coiF, tfF are absolute. Mass-law rows only.
    static Row rescale(Row r, double k)
    {
        r["mpp"] = mpp(r, col(r, "thk")); // the surface mass stays
        r["thk"] = col(r, "thk") * k, r["fRef"] = col(r, "fRef") * k;
        for (const auto &vx : std::vector<std::pair<const char *, const char *>>{{"tf", "tfB"}, {"att", "attB"}, {"alpha", "alphaB"}, {"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}})
            r[vx.first] = col(r, vx.first) * std::pow(k, col(r, vx.second));
        return r;
    }

    // ---- the object test: a closed box of a row, half sizes h, standing on a rigid floor (z = 0) in a diffuse field.
    // Plane waves from n_dir directions of the upper hemisphere (Fibonacci, equal solid angle), each a jittered grid of
    // about n_ray parallel rays over the body and its floor image (a ray that misses both misses the body);
    // ray_progress (scalar, refraction off, threshold 0) with at most 10 interactions, 3 reflections, -60 dB (rays
    // dropped below it count as absorbed); the intersection passed in, so the power of the rays that leave is known.
    // A = (1/pi) * the absorption cross-section integrated over the hemisphere = (2 / n_dir) * sum (the body and its
    // image in free space: 4 x the mean cross-section, halved). chord: the same rays through body and image,
    // 1 - exp(-m l) (an absorbing volume of power attenuation m per meter); left: the area still in flight at the cap.
    struct Obj { std::vector<double> A; double chord = 0.0, left = 0.0; };
    static Obj object_area(const Row &body, const arma::vec3 &h, const std::vector<double> &fa, double m = 0.0, int n_dir = 16, int n_ray = 65)
    {
        const size_t nf = fa.size();
        Obj R{std::vector<double>(nf, 0.0)};
        const arma::fmat mesh = arma::join_cols(quadriga_lib::cube<float>({200.0, 200.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}), quadriga_lib::cube<float>(h, {0.0, 0.0, 0.0}, {0.0, 0.0, h(2)}));
        const arma::uvec mi = arma::join_cols(arma::uvec(12, arma::fill::value(1)), arma::uvec(12, arma::fill::value(2)));
        const Row floor = {{"a", 3.850e-6}, {"e", 1901}, {"m", 20}, {"fRef", 5.0212e-5}}; // rigid, tf 0
        rp::Pal pal;
        for (const char *k : COLS)
            pal[k] = {(float)col(floor, k), (float)col(body, k)};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fa[q] * FS);
        const arma::vec3 hi = {h(0), h(1), 2.0 * h(2)}; // body and image: |x| < hi
        const double Rb = std::sqrt(arma::dot(hi, hi)), dx = std::sqrt(2.0 * (hi(0) * hi(1) + hi(0) * hi(2) + hi(1) * hi(2)) / n_ray); // mean projection S / 4
        const int nx = (int)(Rb / dx) + 1;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int k = 0; k < n_dir; ++k)
        {
            const double z = (k + 0.5) / n_dir, r = std::sqrt(1.0 - z * z), ph = QPI * (3.0 - std::sqrt(5.0)) * k, D = Rb / z + 1.0;
            const arma::vec3 d = {-r * std::cos(ph), -r * std::sin(ph), -z}, u = arma::vec3{d(1), -d(0), 0.0} / r, v = {-d(2) * u(1), d(2) * u(0), d(0) * u(1) - d(1) * u(0)};
            std::vector<arma::vec3> O;
            for (int i = -nx; i < nx; ++i)
                for (int j = -nx; j < nx; ++j)
                {
                    const arma::vec3 o = ((i + U(rng)) * dx) * u + ((j + U(rng)) * dx) * v - D * d, t0 = (-hi - o) / d, t1 = (hi - o) / d; // slab test
                    const double a = arma::max(arma::min(t0, t1)), b = arma::min(arma::max(t0, t1));
                    if (b > a)
                        O.push_back(o), R.chord += dx * dx * (1.0 - std::exp(-m * (b - a))) * 2.0 / n_dir;
                }
            const size_t n = O.size();
            arma::fmat orig(n, 3), dest(n, 3), acc(n, 1 + nf, arma::fill::zeros);
            arma::fcube pd(n, 3, nf);
            std::vector<quadriga_lib::path> paths(n);
            for (size_t i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a)
                    orig(i, a) = (float)O[i](a), dest(i, a) = (float)(O[i](a) + (2.0 * D + 50.0) * d(a)), pd.tube(i, a, i, a).fill((float)d(a)), paths[i].init(0, nf, true);
            arma::Col<short> prev(n, arma::fill::zeros), cur(n, arma::fill::zeros), buf(n, arma::fill::zeros);
            arma::u32_vec noi, fbs, sbs;
            auto power = [&](size_t q, bool missed) // of every ray, or of the rays that miss the mesh
            {
                double p = 0.0;
                for (arma::uword i = 0; i < orig.n_rows; ++i)
                    p += (!missed || fbs(i) == 0) ? std::norm(cd(paths[i].xpr_coeff(q)[0], paths[i].xpr_coeff(q)[1])) : 0.0;
                return p;
            };
            std::vector<double> pin(nf, (double)n), miss(nf);
            for (int g = 0; g < 10 && orig.n_rows; ++g)
            {
                quadriga_lib::ray_triangle_intersect<float>(&orig, &dest, &mesh, nullptr, nullptr, &noi, &fbs, &sbs);
                for (size_t q = 0; q < nf; ++q)
                    miss[q] = power(q, true);
                quadriga_lib::ray_progress(mesh, mi, pal, cf, 0.0f, 0.0f, 0.0f, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr, 10, 3, 255, 0,
                                           -60.0f, 3.0f, 0.0f, false, true, &noi, &fbs, &sbs);
                for (size_t q = 0; q < nf; ++q)
                    R.A[q] += dx * dx * (pin[q] - miss[q] - power(q, false)) * 2.0 / n_dir, pin[q] = power(q, false);
            }
            R.left += dx * dx * pin[0] * 2.0 / n_dir;
        }
        return R;
    }
}

// ============================================================================================================
// The rig's self-checks: inline constructions against independent references
// ============================================================================================================

namespace mt_rows // inline rows: page 11.4 (fitted at 343 m/s), page 11.3, the leaf fixtures of test_acoustic_leaf.cpp
{
    using mt::Row;
    static const Row GLASS4 = {{"thk", 0.004}, {"fRef", 4.0562e-05}, {"tf", 5.6609}, {"tfB", -0.37}, {"att", 6.7899e-08}, {"attB", 1.3503}, {"coiF", 2.5931}, {"coiQ", 284.23},
                               {"resF", 0.12066}, {"resQ", 2.3289}, {"resS", 1}, {"side", 1.3492}, {"m", 1}, {"a", 1.7665e-6}, {"e", 2083.3}};
    static const Row CONCRETE152 = {{"thk", 0.152}, {"fRef", 0.00019663}, {"tf", 0.00025486}, {"tfB", 0.52}, {"att", 14.459}, {"attB", -0.086239}, {"coiF", 6.0529},
                                    {"coiQ", 1.19}, {"resF", 0.068191}, {"resQ", 1.4354}, {"resS", -1}, {"side", 3.5229}, {"m", 1}, {"a", 3.850e-6}, {"e", 1901}};
    static const Row WOOL = {{"thk", 0.05}, {"a", 1.1362}, {"b", -0.043932}, {"c", 0.0023911}, {"d", 1.2533}, {"e", 2.2495}, {"f", -0.45515}, {"g", 0.088044},
                             {"h", 0.14944}, {"fRef", 0.87403}}; // Komatsu at 13112 Pa s/m2
    static const Row BOARD = {{"thk", 0.0125}, {"a", 4.691e-5}, {"e", 573.0}, {"g", 573.0 * 0.2 / 17.98}, {"h", 1.0}}; // limp leaf, 8.6 kg/m2, loss ratio 0.2
    static const Row MDF = {{"a", 4.5e-5}, {"e", 775.0 / 1.2}, {"g", 775.0 / 1.2 * 0.2 / 17.98}, {"h", 1.0}};
    static const Row ROCKWOOL = {{"a", 1.31823}, {"b", -0.0125593}, {"c", 0.00481048}, {"d", 1.80573}, {"e", 0.823363}, {"f", 0.0954384}, {"g", 0.221513},
                                 {"h", -0.0150261}, {"fRef", 0.87403}}; // Miki at 32.8 kPa s/m2
    static const Row AIRBOX = {{"e", 1.000001}};
    static const Row WATER = {{"a", 6.445e-5}, {"e", 833.4}}; // lossless, critical angle 13.4 deg from air
    static Row wall(double tf) { return {{"a", 3.850e-6}, {"e", 1901}, {"m", 20}, {"fRef", 5.0212e-5}, {"tf", tf}}; }
    // a limp mass-law row of density rho (kg/m3) and wave speed cl (m/s): e = rho / rho0, fRef from rho (page 5.4)
    static Row limp(double rho, double cl, double tf, double side, double fc = 0.0, double eta = 1.0)
    {
        return {{"m", 1}, {"e", rho / 1.2}, {"a", std::pow(mt::CS / cl, 2) * 1.2 / rho}, {"fRef", mt::FS * 1e-9 * 1.2 * mt::CS / (QPI * rho)}, {"tf", tf},
                {"side", side}, {"coiF", fc * mt::FS * 1e-9}, {"coiQ", 1.0 / eta}};
    }
    // Hopkins 448: dense aggregate blocks 215 mm and 13 mm plaster on both faces, 430 kg/m2, specimen 3.53 x 2.63 m
    static const Row PLASTER = limp(1300.0, 2000.0, 0.02, std::sqrt(3.53 * 2.63));                        // 16.9 kg/m2 at 13 mm
    static const Row BLOCK = limp(396.2 / 0.215, 3500.0, 0.01, std::sqrt(3.53 * 2.63), 159.0, 0.02);       // 396.2 kg/m2 at 215 mm
    static const Row GLAZING = {{"thk", 0.008}, {"fRef", 0.00011189}, {"tf", 1.3406}, {"tfB", -0.36}, {"att", 0.044026}, {"attB", 0.53547}, {"coiF", 2.686},
                                {"coiQ", 300}, {"resF", 0.18446}, {"resQ", 49.56}, {"resS", -1}, {"side", 1.3492}, {"m", 1}, {"a", 1.7665e-6}, {"e", 2083.3}}; // 4-16-4
}

TEST_CASE("Materials rig - isolated layers: sheet equals traced solid, against the plate rule and the fluid layer", "[acoustic][materials]")
{
    // Why: every material case is this call. A mass-law pane against the plate rule, a porous layer and a limp leaf
    // against the transfer matrix of their columns in air; each as a traced solid on the full grid and as a sheet on
    // the report angles, which must agree node by node.
    // A row with thk 0 at a thickness (rock wool as a layer and as a porous half-space). The 0 deg comparison against
    // the fluid layer at normal incidence (absorption and transmission loss; measured 7.6e-7 and 1.5e-4 dB).
    using namespace mt_rows;
    const mt::Grid G = mt::grid();
    auto at0 = [&](bool tl)
    {
        mt::Measured M{"transfer matrix at 0 deg", tl ? "R_dB" : "alpha_normal", "normal incidence", tl, tl ? 4e-4 : 2e-6, std::vector<double>(mt::OCT, mt::OCT + 9), {}, 0.0, 0.0, 0.0};
        for (double x : mt::octaves(G, [&](size_t q) { double T, R; mt::ref_tmm({{WOOL, mt::SHEET, 0.05}}, G.f[q], 0.0, T, R); return tl ? T : 1.0 - R; }))
            M.lo_hi.push_back({tl ? -10.0 * std::log10(x) : x, tl ? -10.0 * std::log10(x) : x});
        return M;
    };
    mt::check_material("glass 4 mm", GLASS4);
    mt::check_material("glass wool 50 mm", WOOL, {at0(false), at0(true)});
    mt::check_material("board 12.5 mm", BOARD);
    mt::check_material("rock wool at 75 mm", ROCKWOOL, {}, G, 0.075);
    mt::check_material("rock wool at 0.5 m", ROCKWOOL, {}, G, 0.5);
}

TEST_CASE("Materials rig - combinations against the fluid layers and the bonded plate; forbidden stacks", "[acoustic][materials]")
{
    // Why: combinations are room-side sheets on traced layers, two sheets on one face (the first face the room side),
    // leaves on a box of air (page 10.4) and one sheet at the last solid's exit. Per node against the transfer matrix
    // of the layers in air, on the half-space of the first mass-law layer behind the pad of its tf (R only, T NaN; wool
    // on concrete: the pad is formed on the medium in front, notes 9.2, up to 6.3e-3), or the bonded plate (one side).
    // The filled box follows the exchange between box and fill as paths: 12 angles, its alpha within 0.0041 of 192
    // angles x 6 tones per third (grid_probe.py, 2026-10-23). Forbidden stacks throw: two traced wave media back to
    // back (4.9 dB off), two mass-law sheets (17 dB), a sheet between two solids, a lossless wave medium with a critical
    // angle as a solid, three sheets, three co-located faces (R up to 0.98 off), a dense leaf traced behind a box (R up
    // to 1e13); a wave layer behind a mass-law layer has no reference.
    using namespace mt_rows;
    using mt::SHEET;
    using mt::SOLID;
    QD_DIAG("\n# diag: combinations per node against the fluid layers or the bonded plate\n%s", mt::DEV_HEAD);
    struct Case { const char *name; mt::Stack st; double bT, bR; int na; };
    const std::vector<Case> C = {
        {"glass wool | board, sheet | sheet", {{WOOL, SHEET, 0.05}, {BOARD, SHEET, 0.0125}}, 3e-5, 3e-7, 50},
        {"board | glass wool, sheet | sheet", {{BOARD, SHEET, 0.0125}, {WOOL, SHEET, 0.05}}, 3e-5, 3e-7, 50},
        {"glass wool | board, sheet | solid", {{WOOL, SHEET, 0.05}, {BOARD, SOLID, 0.0125}}, 0.15, 1e-2, 50},
        {"board | glass wool, sheet | solid", {{BOARD, SHEET, 0.0125}, {WOOL, SOLID, 0.05}}, 8e-3, 1e-2, 50},
        {"glass 4 | glass 4 bonded, sheet | solid", {{GLASS4, SHEET, 0.004}, {GLASS4, SOLID, 0.004}}, 2e-4, 3e-7, 50},
        {"board on 25 mm of air on the wall (tf 0.02)", {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.025}, {wall(0.02), SOLID, 0.2}}, NAN, 4e-4, 50},
        {"MDF on 20 mm of air, 75 mm rock wool, wall (tf 0)", {{MDF, SHEET, 0.006}, {AIRBOX, SOLID, 0.02}, {ROCKWOOL, SOLID, 0.075}, {wall(0.0), SOLID, 0.2}}, NAN, 5e-3, 12},
        {"glass wool on concrete 152 mm (its tf)", {{WOOL, SHEET, 0.05}, {CONCRETE152, SOLID, 0.152}}, NAN, 0.015, 50},
        {"glass wool on a tf-0 wall", {{WOOL, SHEET, 0.05}, {wall(0.0), SOLID, 0.2}}, NAN, 3e-7, 50},
        {"double leaf: board | 75 mm of air | board at the exit", {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.075}, {BOARD, SHEET, 0.0125}}, 9e-3, 8e-5, 50},
        {"board | 75 mm of air | glass 4 at the exit", {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.075}, {GLASS4, SHEET, 0.004}}, NAN, 2e-3, 50},
        {"glass wool | concrete 152 mm | plaster at the exit", {{WOOL, SHEET, 0.05}, {CONCRETE152, SOLID, 0.152}, {mt::with(PLASTER, "side", 3.5229), SHEET, 0.013}}, NAN, 0.015, 50},
        {"glass wool solid | board at the exit", {{WOOL, SOLID, 0.05}, {BOARD, SHEET, 0.0125}}, 2e-3, 1.2e-4, 50},
        {"curtain 3 mm | 100 mm of air | glass 4 at the exit", {{ROCKWOOL, SHEET, 0.003}, {AIRBOX, SOLID, 0.1}, {GLASS4, SHEET, 0.004}}, NAN, 2e-3, 50},
        {"concrete 152 mm | plaster 13 mm at the exit, bonded", {{CONCRETE152, SOLID, 0.152}, {mt::with(PLASTER, "side", 3.5229), SHEET, 0.013}}, 1e-5, 3e-7, 50},
        {"plaster | block 215 mm | plaster at the exit, bonded", {{PLASTER, SHEET, 0.013}, {BLOCK, SOLID, 0.215}, {PLASTER, SHEET, 0.013}}, 2e-4, 3e-7, 50},
    };
    for (const Case &c : C)
    {
        const mt::Out O = mt::check_stack(c.name, c.st, c.bT, c.bR, mt::grid(c.na));
        if (c.st.size() > 2)
            mt::rows(c.name, O);
    }
    for (const mt::Stack &bad : {mt::Stack{{WOOL, SOLID, 0.05}, {BOARD, SOLID, 0.0125}}, mt::Stack{{GLASS4, SHEET, 0.004}, {GLASS4, SHEET, 0.004}},
                                 mt::Stack{{CONCRETE152, SOLID, 0.152}, {WOOL, SHEET, 0.05}, {CONCRETE152, SOLID, 0.152}}, mt::Stack{{WATER, SOLID, 0.1}},
                                 mt::Stack{{WOOL, SHEET, 0.05}, {BOARD, SHEET, 0.0125}, {BOARD, SHEET, 0.0125}},
                                 mt::Stack{{WOOL, SHEET, 0.01}, {BOARD, SHEET, 0.015}, {CONCRETE152, SOLID, 0.152}},
                                 mt::Stack{{WOOL, SHEET, 0.01}, {BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.05}, {wall(0.02), SOLID, 0.2}},
                                 mt::Stack{{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.075}, {BOARD, SOLID, 0.0125}}})
        CHECK_THROWS_AS(mt::scene(bad), std::invalid_argument);
    double T, R; // no reference for a wave layer behind a mass-law layer
    for (const mt::Stack &bad : {mt::Stack{{CONCRETE152, SOLID, 0.152}, {WOOL, SHEET, 0.05}}, mt::Stack{{GLASS4, SHEET, 0.004}, {AIRBOX, SOLID, 0.075}, {BOARD, SHEET, 0.0125}}})
        CHECK_THROWS_AS(mt::ref_auto(bad, 125.0, 0.0, T, R), std::invalid_argument);
    CHECK_NOTHROW(mt::scene({{WATER, SHEET, 0.1}}));
    mt::Out O = mt::run(C[9].st, mt::grid()); // the transmission from the paths equals transmit() (the double leaf)
    const mt::Field Tt = O.T, Rt = mt::reflect(mt::scene(C[9].st), O.g, O.calls, O.rays, &O.T);
    double d = 0.0;
    for (size_t i = 0; i < O.g.th.size() * O.g.f.size(); ++i)
        d = std::max(d, std::abs(mt::dB(std::norm(O.T[i % O.g.th.size()][i / O.g.th.size()])) - mt::dB(std::norm(Tt[i % O.g.th.size()][i / O.g.th.size()]))));
    QD_DIAG("# diag: double leaf, T from the paths against transmit(): %.2e dB\n", d);
    CHECK(d < 1.5e-4);
}

TEST_CASE("Materials rig - a band equals its single frequencies at threshold 0", "[acoustic][materials]")
{
    // Why: all tones of a grid run in one band (125 Hz first). At thin_slab_threshold 0 every series is summed, so no
    // tone inherits a follow verdict of the band's first one (at 0.15 the board on its box reflects up to 0.164 off near
    // grazing). The board on its box, every angle, three tones alone against the same tones in the band.
    using namespace mt_rows;
    const mt::Scene S = mt::scene({{BOARD, mt::SHEET, 0.0125}, {AIRBOX, mt::SOLID, 0.025}, {wall(0.02), mt::SOLID, 0.2}});
    const mt::Grid G = mt::grid();
    int calls;
    long rays;
    const mt::Field B = mt::reflect(S, G, calls, rays);
    double worst = 0.0;
    for (double fa : {160.0, 1600.0, 6300.0})
    {
        const size_t q = (size_t)(std::find(G.f.begin(), G.f.end(), fa) - G.f.begin());
        mt::Grid g1 = G;
        g1.f = {fa}, g1.third = {G.third[q]};
        const mt::Field A = mt::reflect(S, g1, calls, rays);
        for (size_t i = 0; i < G.th.size(); ++i)
            worst = std::max(worst, std::abs(A[i][0] - B[i][q]));
    }
    QD_DIAG("\n# diag: board on its box, band against alone, largest |R_band - R_alone| (complex) %.2e\n", worst);
    CHECK(worst < 1e-6);
}

TEST_CASE("Materials rig - a porous layer on a rigid wall: the added insulation", "[acoustic][materials]")
{
    // Why: treatment on rigid bodies is the common combination. 50 mm glass wool as a sheet on cast concrete 152 mm:
    // the insulation the wool adds, library against the fluid layer on a limp plate (346.7 kg/m2), printed side by side
    // and not asserted (the library adds up to 8 dB too much at 31.5 Hz); the wool alone in air asserted against the
    // same matrix, so both sides map the layer alike.
    // ISSUE 1: a porous layer on a rigid wall adds too much insulation at low frequency.
    using namespace mt_rows;
    mt::added_insulation("50 mm glass wool to 152 mm concrete", {{WOOL, mt::SHEET, 0.05}}, {{CONCRETE152, mt::SOLID, 0.152}});
}

TEST_CASE("Materials rig - engine limits printed: a filled box on one path, wave layers before a pane, three bonded bodies", "[acoustic][materials]")
{
    // Why: the measured sizes of known engine limits, printed for the stage-2b list (no assertion; the same
    // constructions in their working forms are asserted in the combinations case).
    using namespace mt_rows;
    using mt::SHEET;
    using mt::SOLID;
    const mt::Grid G = mt::grid(), G12 = mt::grid(12);
    QD_DIAG("\n# diag: engine limits per node against the fluid layers or the bonded plate\n%s", mt::DEV_HEAD);
    // ISSUE 3: a leaf behind a filled box (the precise drywall recipe). Its T is the exchange between box and fill, a
    // series of paths: through the paths of the reflection ray set at most 10 interactions (5.0 dB off the fluid
    // layers), on one straight path (calc_diffraction_gain, as the scene tracer's direct path through a wall: 9.5 dB);
    // R uncapped (1.45e-2) and at 10 interactions (0.22).
    const mt::Stack dw = {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.025}, {ROCKWOOL, SOLID, 0.05}, {BOARD, SHEET, 0.0125}};
    mt::Out O = mt::run(dw, G12);
    mt::against("ISSUE 3: board | air | rock wool | board, T from the paths (10 interactions), R uncapped", O, dw);
    mt::rows("ISSUE 3: board | air | rock wool | board", O);
    O.T = mt::transmit(mt::scene(dw), G12), O.R = mt::reflect(mt::scene(dw), G12, O.calls, O.rays, nullptr, 10);
    mt::against("ISSUE 3: board | air | rock wool | board, T on one straight path, R at 10 interactions", O, dw);
    // ISSUE 4: plaster | block | plaster with the far plaster traced (the exit-sheet form is exact): the state is one
    // level deep. Random-incidence tl, library against the bonded plate of the three, and Hopkins 448 measured
    // [Hopkins_2007_SoundInsulation, research line 448 at 2dad906; the probe rows are physical, not fitted to it].
    const mt::Stack pb = {{PLASTER, SHEET, 0.013}, {BLOCK, SOLID, 0.215}, {PLASTER, SOLID, 0.013}};
    const mt::Out P = mt::run(pb, G);
    mt::against("ISSUE 4: plaster | block 215 mm | plaster traced", P, pb);
    const auto pl = mt::octaves(G, [&](size_t q) { double t = 0.0; for (int i = 0; i < G.na; ++i) { double T, R; mt::ref_plate(pb, G.f[q], G.th[i], T, R); t += G.w[i] * T; } return t; });
    const double H448[9] = {NAN, 35.4, 39.6, 42.7, 52.2, 58.3, 65.8, 70.6, NAN};
    QD_DIAG("# diag: ISSUE 4, random incidence tl (dB)\noctave,library,bonded_plate,hopkins_448\n");
    for (int o = 0; o < 9; ++o)
        QD_DIAG("%g,%.2f,%.2f,%.1f\n", mt::OCT[o], P.t8[o], -10.0 * std::log10(pl[o]), H448[o]);
    // ISSUE 1: a curtain (rock-wool columns, 3 mm) on 100 mm of air before a 4 mm pane: the insulation the curtain and
    // the box add, library against the fluid layers on a limp plate of the pane's mass (infinite and finite at the
    // pane's side) and against the layers alone in air (the series estimate)
    mt::added_insulation("a curtain on 100 mm of air to glass 4 mm", {{ROCKWOOL, SHEET, 0.003}, {AIRBOX, SOLID, 0.1}}, {{GLASS4, SHEET, 0.004}}, G, 2e-4, 2e-6);
}

TEST_CASE("Materials rig - tier 2 at the specimen's side and mass, the class envelope", "[acoustic][materials]")
{
    // Why: a reference is evaluated at its specimen's side and shifted by the mass law to the row's surface mass
    // (e rho0 thk: 346.7 kg/m2 for these concrete columns at 152 mm, as Litvin's wall); a class row against the envelope
    // of its walls, each shifted, within 1 dB. The concrete columns of page 11.4 as a solid row at 150 mm against Litvin
    // 76-66 at its laboratory side 3.523 m (shift -0.114 dB; the row at 3.5229 m, so transmission is rerun) and the envelope of 76-66 and 76-77 (152 and 203 mm,
    // [Litvin_1980_ConcreteWallsTL], research lines 228, 229 at 2dad906).
    using namespace mt_rows;
    const mt::Row c = mt::with(CONCRETE152, "thk", 0.0);
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, L1 = {42.1, 46.5, 52.6, 56.5, 61.7, 64.6}, L2 = {43.8, 47.3, 54.7, 58.1, 63.3, 67.3};
    mt::Measured M{"Litvin_1980_ConcreteWallsTL 76-66", "R_dB", "laboratory sound reduction index (ASTM E90)", true, 3.0, oct, {}, 3.523, 346.7};
    for (double v : L1)
        M.lo_hi.push_back({v, v});
    mt::check_material("concrete columns at 150 mm", c, {M, mt::envelope("Litvin 76-66 and 76-77", oct, {{346.7, L1}, {471.6, L2}}, mt::mpp(c, 0.150), 3.5229)}, mt::grid(), 0.150);
}

TEST_CASE("Materials rig - porous rows against their models", "[acoustic][materials]")
{
    // Why: porous columns are power laws fitted to a model at a flow resistivity; the model is written here. Random-
    // incidence absorption of the layer on a rigid backing, row against model (bounds about 2 x the measured fit miss); the
    // transmission loss at 0 deg in air and Delany-Bazley at the same resistivity printed.
    using namespace mt_rows;
    const mt::Grid G = mt::grid();
    static const mt::Row W0 = wall(0.0);
    struct Case { const char *name; mt::Row row, model; double thk, b; };
    for (const Case &c : std::vector<Case>{{"glass wool, Komatsu 13112 Pa s/m2", WOOL, {{"komatsu", 13112.0}}, 0.05, 0.11},
                                           {"rock wool, Miki 32.8 kPa s/m2", ROCKWOOL, {{"miki", 32800.0}}, 0.075, 0.10}})
    {
        const double sg = c.model.begin()->second;
        auto alpha = [&](const mt::Row &r) { return mt::octaves(G, [&](size_t q) { double a = 0.0; for (int i = 0; i < G.na; ++i) { double T, R; mt::ref_tmm({{r, mt::SHEET, c.thk}}, G.f[q], G.th[i], T, R, mt::HALF, &W0); a += G.w[i] * (1.0 - R); } return a; }); };
        auto tl0 = [&](const mt::Row &r) { return mt::octaves(G, [&](size_t q) { double T, R; mt::ref_tmm({{r, mt::SHEET, c.thk}}, G.f[q], 0.0, T, R); return T; }); };
        const auto ar = alpha(c.row), am = alpha(c.model), ad = alpha({{"db", sg}}), tr = tl0(c.row), tm = tl0(c.model);
        QD_DIAG("\n# diag: %s, %.0f mm on a rigid backing\noctave,alpha_row,alpha_model,alpha_delany_bazley,tl0_row_minus_model_dB\n", c.name, 1e3 * c.thk);
        double worst = 0.0;
        for (int o = 0; o < 9; ++o)
            QD_DIAG("%g,%.4f,%.4f,%.4f,%+.3f\n", mt::OCT[o], ar[o], am[o], ad[o], 10.0 * std::log10(tm[o] / tr[o])), worst = std::max(worst, std::isnan(ar[o] - am[o]) ? INFINITY : std::abs(ar[o] - am[o]));
        INFO(c.name);
        CHECK(worst < c.b);
    }
}

TEST_CASE("Materials rig - a lumped row rescaled with fRef is the same wall", "[acoustic][materials]")
{
    // Why: lumped rows carry the visible thickness (wish-list 1.7.2): thk and fRef times k with every power law
    // rescaled leaves every column unchanged at every frequency. Double glazing 4-16-4 of page 11.4 at 8 and 24 mm
    // (k = 3; fRef alone would move tf at 125 Hz from 0.1124 to 0.1670), sheet and solid, per node.
    using namespace mt_rows;
    const mt::Row R3 = mt::rescale(GLAZING, 3.0);
    const mt::Grid G = mt::grid();
    for (mt::Kind k : {mt::SHEET, mt::SOLID})
    {
        const mt::Out A = mt::run({{GLAZING, k, 0.008}}, G), B = mt::run({{R3, k, 0.024}}, G);
        double dT = 0.0, dR = 0.0;
        for (size_t q = 0; q < G.f.size(); ++q)
            for (size_t i = 0; i < G.th.size(); ++i)
                dT = std::max(dT, std::abs(mt::dB(std::norm(A.T[i][q])) - mt::dB(std::norm(B.T[i][q])))), dR = std::max(dR, std::abs(std::norm(A.R[i][q]) - std::norm(B.R[i][q])));
        QD_DIAG("# diag: double glazing at 8 mm against 24 mm rescaled, %s: dT %.2e dB, dR %.2e\n", k == mt::SHEET ? "sheet" : "solid", dT, dR);
        CHECK(dT < (k == mt::SHEET ? 3e-6 : 1.3e-4));
        CHECK(dR < 3e-7);
    }
}

TEST_CASE("Materials rig - objects: absorption area of a hard box, an absorbing volume and a porous body on the floor", "[acoustic][materials]")
{
    // Why: objects take a per-object absorption area measured in a reverberation chamber (wish-list 2.4); in a diffuse
    // field it is 4 x the mean absorption cross-section. 16 directions x about 65 rays, the octave centers, at most 10
    // interactions, 3 reflections, -60 dB: within 1.3 % of 64 directions x 2000 rays x 18 thirds (phase-3a probe, three
    // seeds, 0.19 s). A hard box (1.0 x 0.5 x 2.0 m, tf 0.1) against alpha S of its five exposed faces (-3.0 %: the
    // 16 directions; -1.9 % at 32, -0.4 % at 64); an absorbing
    // volume of the same size (m 0.05 per meter) against the chords of its rays (4mV printed); a porous cushion
    // (0.6 x 0.6 x 0.4 m of the glass-wool columns) against the dense grid, 0.8365 / 0.9007 / 0.9994 / 1.1195 / 1.2205 /
    // 1.2551 m2, as a regression guard.
    using namespace mt_rows;
    const std::vector<double> F = {125, 250, 500, 1000, 2000, 4000}, REF = {0.8365, 0.9007, 0.9994, 1.1195, 1.2205, 1.2551};
    const mt::Obj B = mt::object_area(wall(0.1), {0.5, 0.25, 1.0}, F), V = mt::object_area({{"e", 1.0}, {"alpha", 4.3429448 * 0.05}}, {0.5, 0.25, 1.0}, F, 0.05),
                  P = mt::object_area(WOOL, {0.3, 0.3, 0.2}, F);
    QD_DIAG("\n# diag: objects on the floor, absorption area (m2) per octave; volume: chords %.4f, 4mV %.4f; left at the cap %.1e / %.1e / %.1e\noctave,hard_box,alpha_S,volume,cushion,cushion_dense\n",
            V.chord, 4.0 * 0.05 * 1.0, B.left, V.left, P.left);
    for (size_t q = 0; q < F.size(); ++q)
    {
        QD_DIAG("%g,%.4f,%.4f,%.4f,%.4f,%.4f\n", F[q], B.A[q], 0.1 * 6.5, V.A[q], P.A[q], REF[q]);
        CHECK(std::abs(B.A[q] / 0.65 - 1.0) < 0.06);
        CHECK(std::abs(V.A[q] / V.chord - 1.0) < 1e-3);
        CHECK(std::abs(P.A[q] / REF[q] - 1.0) < 0.02);
    }
    // ISSUE 6: a ray within 0.2 deg of grazing a box face re-hits the face at its launch point at every call, splits and
    // gains power. The ray of phase 1 (the hard box, direction 36 of 64), alone; rays and their power per call printed.
    const arma::fmat mesh = arma::join_cols(quadriga_lib::cube<float>({200.0, 200.0, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}), quadriga_lib::cube<float>({0.5, 0.25, 1.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    rp::Pal pal;
    for (const char *k : mt::COLS)
        pal[k] = {(float)mt::col(wall(0.0), k), (float)mt::col(wall(0.1), k)};
    const arma::fvec cf = {(float)(1000.0 * mt::FS)};
    const double nd = std::sqrt(0.2223 * 0.2223 + 45.5788 * 45.5788 + 31.6455 * 31.6455);
    const arma::frowvec dir = {(float)(0.2223 / nd), (float)(45.5788 / nd), (float)(-31.6455 / nd)}, hit = {-0.5f, 0.0313f, 1.5332f};
    arma::fmat orig = hit - dir, dest = hit + 50.0f * dir, acc(1, 2, arma::fill::zeros);
    arma::fcube pd(1, 3, 1);
    for (int a = 0; a < 3; ++a)
        pd(0, a, 0) = dir(a);
    std::vector<quadriga_lib::path> paths(1);
    paths[0].init(0, 1, true);
    arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
    QD_DIAG("# diag: ISSUE 6, one grazing ray on the hard box\ncall,rays,power\n");
    for (int g = 0; g < 10 && orig.n_rows; ++g)
    {
        quadriga_lib::ray_progress(mesh, arma::join_cols(arma::uvec(12, arma::fill::value(1)), arma::uvec(12, arma::fill::value(2))), pal, cf, 0.0f, 0.0f, 0.0f, orig, dest, prev, cur, buf,
                                   pd, acc, paths, nullptr, nullptr, nullptr, nullptr, 10, 255, 255, 0, -60.0f, 3.0f, 0.0f, false, true);
        double p = 0.0;
        for (arma::uword i = 0; i < orig.n_rows; ++i)
            p += std::norm(mt::cd(paths[i].xpr_coeff(0)[0], paths[i].xpr_coeff(0)[1]));
        QD_DIAG("%d,%llu,%.6f\n", g, (unsigned long long)orig.n_rows, p);
    }
}

// ============================================================================================================
// The table and its materials
// ============================================================================================================

TEST_CASE("Materials table - every row has a material case and every column a valid value", "[acoustic][materials]")
{
    // Why: the table and its tests must not drift. A row's family is its name without the sheet suffix _<NN>mm; every
    // family has one case "Material - <family>[ + <family> ...]" and every family a case names has a row; all 26
    // columns; fRef, a, e positive, the columns the library requires non-negative; lowercase snake_case names.
    const mt::Table &T = mt::table();
    REQUIRE(!T.names.empty());
    for (const char *k : mt::COLS)
        CHECK(T.prop.count(k) == 1);
    std::multiset<std::string> cases; // the families named by the cases
    for (const Catch::TestCaseInfo *tc : Catch::getRegistryHub().getTestCaseRegistry().getAllInfos())
        if (tc->name.rfind("Material - ", 0) == 0)
            for (size_t a = 11, b; a != std::string::npos; a = b == std::string::npos ? b : b + 3)
                b = tc->name.find(" + ", a), cases.insert(tc->name.substr(a, b == std::string::npos ? b : b - a));
    std::set<std::string> fams;
    for (size_t i = 0; i < T.names.size(); ++i)
    {
        const std::string &n = T.names[i], f = std::regex_replace(n, std::regex("_[0-9]+(_[0-9]+)?mm$"), "");
        std::smatch mm; // the suffix is thk in mm (an underscore for the decimal point); no suffix, thk 0
        fams.insert(f);
        INFO("row " << n << ", family " << f);
        CHECK(cases.count(f) == 1);
        CHECK(std::abs(1000.0 * T.prop.at("thk")[i] - (std::regex_search(n, mm, std::regex("_([0-9]+)(_([0-9]+))?mm$")) ? std::stod(mm[1].str() + "." + (mm[3].matched ? mm[3].str() : "0")) : 0.0)) < 1e-9);
        CHECK(std::all_of(n.begin(), n.end(), [](char ch) { return std::islower((unsigned char)ch) || std::isdigit((unsigned char)ch) || ch == '_'; }));
        for (const char *k : {"fRef", "a", "e"})
            CHECK(T.prop.at(k)[i] > 0.0);
        for (const char *k : {"c", "g", "thk", "att", "alpha", "m", "resF", "resQ", "coiF", "coiQ", "tfF", "side"})
            CHECK(T.prop.at(k)[i] >= 0.0);
    }
    for (const std::string &c : cases)
        CHECK(fams.count(c) == 1);
}

TEST_CASE("Material - concrete_cast", "[acoustic][materials]")
{
    // Cast concrete 152 mm, the preliminary row of page 11.4 (fitted at 343 m/s, laboratory side 3.52 m); class 2,
    // heavy walls. Transmission: Litvin and Belliston, PCA RD066, test 76-66 (cast concrete 152 mm, 346.7 kg/m2,
    // opening 4.37 x 2.84 m), the laboratory octaves 125 Hz - 4 kHz the row was fitted to (page 11.1, tools/acoustic/
    // tl_refs.json; the research data reads 42.1 / 46.5 / 52.6 / 56.5 / 61.7 / 64.6 dB from the same plot); the gap
    // bands 31.5, 63 Hz and 8 kHz take the spread of physical models (tl_refs.json, notes 5). Absorption: "Smooth
    // unpainted concrete", Cox and D'Antonio, Appendix A (a hard face, used as stated, page 10.5).
    mt::check_material("concrete_cast_152mm", mt::row("concrete_cast_152mm"),
                       {{"Litvin_1980_ConcreteWallsTL", "R_dB", "laboratory sound reduction index (ASTM E90), octave = energy mean of thirds", true, 3.0,
                         {31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000},
                         {{38.0, 49.8}, {31.7, 42.8}, {41.926, 41.926}, {46.474, 46.474}, {52.627, 52.627}, {56.315, 56.315}, {62.028, 62.028}, {64.897, 64.897}, {70.4, 77.6}}},
                        {"Cox_2016_AbsorbersDiffusers", "alpha_random", "compilation, type not stated", false, 0.05,
                         {125, 250, 500, 1000, 2000, 4000},
                         {{0.01, 0.01}, {0.01, 0.01}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.05, 0.05}}}});
}
