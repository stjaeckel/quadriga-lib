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
//   - stacks, room side first: sheets (one face, thickness in thk) in front, traced solids (two faces) behind; planes
//     x = const of 400 m (the panel size enters only through side);
//   - transmission: calc_diffraction_gain, one straight path per angle (lod 0, scalar, threshold 0, no spreading);
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
    // The stack rule (measured in the plan): sheets only on the room side, at most two (page 12 rule 13); no two
    // mass-law sheets on one face (rule 9); no two traced wave media back to back (one straight path misses their
    // exchange), except a box of air in front of a fill; no lossless wave medium with a critical angle as a solid.
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
            if (S[i].kind == SHEET && (i != ns++ || ns > 2 || (p && mass(r) && mass(*p))))
                no("a sheet behind a traced solid, more than two sheets, or two mass-law sheets on one face");
            if (S[i].kind == SOLID && lossless(r) && !cavity(r) && col(r, "a") * col(r, "e") < 1.0)
                no("a lossless wave medium with a critical angle as a traced solid (use a sheet)");
            if (S[i].kind == SOLID && p && S[i - 1].kind == SOLID && !mass(r) && !mass(*p) && !cavity(*p))
                no("two traced wave media back to back");
        }
    }
    // The library's own geometry, 400 m square faces (the panel size enters only through side): a sheet is
    // quadriga_lib::plane, its +z normal turned to -x (the room), in the plane x = 0 in stack order (the first face is
    // the room side); a solid is a closed quadriga_lib::cube of its thickness behind the one before; layer k is material
    // k, solids with thk 0.
    struct Scene { arma::fmat mesh; arma::uvec mi; rp::Pal pal; float x_end = 0.0f; };
    static Scene scene(const Stack &st)
    {
        stack_rule(st);
        Scene S;
        auto add = [&](const arma::fmat &m, size_t k)
        { S.mesh = arma::join_cols(S.mesh, m), S.mi = arma::join_cols(S.mi, arma::uvec(m.n_rows, arma::fill::value(k + 1))); };
        for (size_t i = 0; i < st.size(); ++i)
            if (st[i].kind == SHEET)
                add(quadriga_lib::plane<float>({200.0, 200.0}, {0.0, -0.5 * QPI, 0.0}, {0.0, 0.0, 0.0}), i);
        for (size_t i = 0; i < st.size(); ++i)
            if (st[i].kind == SOLID)
                add(quadriga_lib::cube<float>({0.5 * st[i].thk, 200.0, 200.0}, {0.0, 0.0, 0.0}, {S.x_end + 0.5 * st[i].thk, 0.0, 0.0}), i),
                    S.x_end += (float)st[i].thk;
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
    // Reflection: one aimed ray per angle from 1 m in front of its hit point, every angle in one ray set, ray_progress
    // until no ray is left (at most 127 tones per band, the first its base); the paths back in air (an odd number of
    // reflections) summed coherently per angle at their polyline's point
    static Field reflect(const Scene &S, const Grid &G, int &calls, long &rays)
    {
        const size_t n = G.th.size(), nf = G.f.size();
        Field R(n, std::vector<cd>(nf, 0.0));
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
                pd.tube(i, 0, i, 0).fill(c), pd.tube(i, 1, i, 1).fill(s), paths[i].init(0, m, true), paths[i].set_length(0.1f);
            }
            arma::Col<short> prev(n, arma::fill::zeros), cur(n, arma::fill::zeros), buf(n, arma::fill::zeros);
            for (int g = 0; g < 200 && orig.n_rows; ++g)
            {
                quadriga_lib::ray_progress(S.mesh, S.mi, S.pal, cf, -1.0f, 0.37f, 0.0f, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr,
                                           nullptr, nullptr, 60, 255, 255, 0, -120.0f, 3.0f, 0.0f, false, true);
                ++calls, rays += (long)orig.n_rows;
                for (arma::uword i = 0; i < orig.n_rows; ++i)
                {
                    const quadriga_lib::path &p = paths[i];
                    const size_t ns = p.n_seg(), k = (size_t)std::lround((-p.coord(0)[2] - 0.005f) / 0.01f);
                    if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1) || p.nREF % 2 == 0 || k >= n)
                        continue; // in air, extended by this call, turned back
                    const float *a = p.coord(0), *b = p.coord(ns - 1);
                    // the polyline from the first interaction on: the path's length less its launch leg from the origin
                    const double L = (double)p.length() - std::sqrt(std::pow(a[0] + 1.0, 2) + std::pow(a[1] - 0.37, 2) + std::pow((double)a[2], 2));
                    for (size_t q = 0; q < m; ++q)
                        R[k][c0 + q] += cd(p.xpr_coeff(q)[0], p.xpr_coeff(q)[1]) *
                                        std::exp(cd(0.0, 2.0 * QPI * cf(q) / C0 * (std::sin(rad(G.th[k])) * ((double)b[1] - a[1]) - L)));
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
    static Out run(const Stack &st, const Grid &G)
    {
        const Scene S = scene(st);
        Out O;
        O.g = G, O.T = transmit(S, G), O.R = reflect(S, G, O.calls, O.rays), O.af.assign(G.f.size(), 0.0), O.tauf.assign(G.f.size(), 0.0);
        for (size_t q = 0; q < G.f.size(); ++q)
            for (size_t i = 0; i < G.th.size(); ++i)
                O.af[q] += G.w[i] * (1.0 - std::norm(O.R[i][q])), O.tauf[q] += G.w[i] * std::norm(O.T[i][q]);
        const auto a = octaves(G, [&](size_t q) { return O.af[q]; }), t = octaves(G, [&](size_t q) { return O.tauf[q]; });
        for (int o = 0; o < 9; ++o)
            O.a8[o] = a[o], O.t8[o] = -10.0 * std::log10(t[o]);
        return O;
    }

    // ---- tier 1: independent physics ----
    // The transfer matrix of fluid layers (lw::stack of test_acoustic_leaf.cpp): (p, v_n), exp(+j w t), impedances
    // relative to rho0 c0 at the sine of air; behind the layers air, a fluid half-space behind a matched pad that absorbs
    // tf of what arrives from air, or a limp plate of surface mass mpp (kg/m2) with air behind it.
    enum Backing { AIR, HALF, PLATE };
    static void eps_mu(const Row &M, double fa, cd &e, cd &m)
    {
        e = cd(pow_law(M, "a", "b", fa), -17.98 * pow_law(M, "c", "d", fa) / (fa * FS * 1e-9));
        m = cd(pow_law(M, "e", "f", fa), -17.98 * pow_law(M, "g", "h", fa) / (fa * FS * 1e-9));
    }
    static cd q_of(cd eps, cd mu, double s) // the normal wavenumber over k0: the root that decays or carries the wave away
    {
        const cd q = std::sqrt(eps * mu - s * s);
        return (q.imag() > 0.0) ? -q : q;
    }
    static void ref_tmm(const Stack &L, double fa, double thd, double &T, double &R, Backing bk = AIR, const Row *half = nullptr, double mpp = 0.0)
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
            const cd vt = 2.0 / (A * Za + B + Za * (C * Za + D));
            R = std::norm((A * Za + B) * vt - 1.0), T = std::norm(Za * vt);
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
        const double A = 20.0 * std::log10(X), k = 2.0 * QPI * fa / CS, ka = std::max(0.5 * k * side, 0.1), cc = std::min(c, 1.0);
        if (!(side > 0.0))
            return c >= 1.0 ? A : A * std::log10(1.0 + X * c * X * c) / std::log10(1.0 + X * X);
        const double p = std::min(1.0, 1.3 * std::sqrt(QPI / (2.0 * ka))), h = 1.0 / (2.0 / 3.0 * std::sqrt(2.0 * ka / QPI) - 0.124);
        const double q = 2.0 * QPI / (k * k * side * side), x = cc >= p ? cc : h - (h / p - 1.0) * cc, sg = 1.0 / std::sqrt(x * x + q * q);
        return std::max(0.0, A * std::log10(1.0 + (X / sg) * (X / sg)) / std::log10(1.0 + X * X) + 10.0 * std::log10(sg * cc));
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
    // layers on the half-space of a mass-law back layer behind its pad
    static void ref_auto(const Stack &L, double fa, double thd, double &T, double &R)
    {
        const size_t nm = (size_t)std::count_if(L.begin(), L.end(), [](const Layer &l) { return mass(l.row); });
        if (nm == L.size())
            ref_plate(L, fa, thd, T, R);
        else if (nm == 0)
            ref_tmm(L, fa, thd, T, R);
        else
            ref_tmm(Stack(L.begin(), L.end() - 1), fa, thd, T, R, HALF, &L.back().row);
    }

    // ---- per node against the reference (largest |dT| in dB where both are above 1e-25, |dR|, R + T), and the twin ----
    struct Dev { double dT = 0.0, dR = 0.0, RT = 0.0, fT = 0.0, aT = 0.0, fR = 0.0, aR = 0.0; };
    static Dev against(const char *name, const Out &O, const Stack &st)
    {
        Dev D;
        for (size_t q = 0; q < O.g.f.size(); ++q)
            for (size_t i = 0; i < O.g.th.size(); ++i)
            {
                double Tr, Rr, Tl = std::norm(O.T[i][q]), Rl = std::norm(O.R[i][q]);
                ref_auto(st, O.g.f[q], O.g.th[i], Tr, Rr);
                D.RT = std::max(D.RT, Tl + Rl);
                if (Tr > 1e-25 && Tl > 1e-25 && std::abs(dB(Tl) - dB(Tr)) > D.dT)
                    D.dT = std::abs(dB(Tl) - dB(Tr)), D.fT = O.g.f[q], D.aT = O.g.th[i];
                if (std::abs(Rl - Rr) > D.dR)
                    D.dR = std::abs(Rl - Rr), D.fR = O.g.f[q], D.aR = O.g.th[i];
            }
        QD_DIAG("%s,%zu,%.2e,%g,%.1f,%.2e,%g,%.1f,%.8f,%d,%ld\n", name, O.g.th.size(), D.dT, D.fT, D.aT, D.dR, D.fR, D.aR, D.RT, O.calls, O.rays);
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
    // interval (the model spread); asserted within the guideline tol, deviations printed ----
    struct Measured
    {
        const char *key, *quantity, *coefficient_type;
        bool tl; // transmission loss in dB, else absorption
        double tol;
        std::vector<double> octaves_hz;
        std::vector<std::pair<double, double>> lo_hi;
    };
    static void tier2(const char *name, const Out &O, const Measured &M)
    {
        QD_DIAG("\n# diag: %s, tier 2 against %s (%s, %s): octave, library, reference, deviation outside the reference\noctave,library,ref_lo,ref_hi,deviation\n",
                name, M.key, M.quantity, M.coefficient_type);
        for (size_t k = 0; k < M.octaves_hz.size(); ++k)
        {
            const int o = (int)(std::find(OCT, OCT + 9, M.octaves_hz[k]) - OCT);
            REQUIRE(o < 9);
            const double v = M.tl ? O.t8[o] : O.a8[o], lo = M.lo_hi[k].first, hi = M.lo_hi[k].second, dev = v > hi ? v - hi : (v < lo ? v - lo : 0.0);
            QD_DIAG("%g,%.4f,%.4f,%.4f,%+.4f\n", M.octaves_hz[k], v, lo, hi, dev);
            INFO(name << ", " << M.key << ", octave " << M.octaves_hz[k] << " Hz");
            CHECK(std::abs(dev) <= M.tol);
        }
    }

    // ---- one call per isolated material. The row as a traced solid of its thickness on the full grid and as a sheet on
    // the report angles (the twin is a per-node identity; the solid is the cheaper form, 7.3 against 15.3 ms for
    // concrete on one thread): tier 1 for both, the twin, R + T <= 1; the diagnostic tables; for a mass-law row its
    // physical model (the bare limp mass law of its surface mass e rho0 thk with Davy's sigma at its side, no fitted
    // knob [London_1949_SingleWalls]); tier 2 for every measured row given. Bounds 2 to 3 times the measured maxima, by
    // type: mass-law rows; a traced porous layer (7.8e-4 dB, 3.7e-5 at 89.1 deg); a traced dense leaf (mu' > 10:
    // 2.6e-2 dB, 2.0e-3 at 89.1 deg); wave sheets (3.4e-6 dB, 7.8e-8); the twin (3.3e-5 dB mass law, 1.7e-4 dB and
    // 6.7e-6 wave media).
    static Out check_material(const std::string &name, const Row &r, const std::vector<Measured> &refs = {}, const Grid &G = grid())
    {
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
                dR = std::max(dR, std::abs(std::norm(X.R[i + off][q]) - std::norm(S.R[i][q])));
            }
        QD_DIAG("# %s, sheet against solid on the report angles: dT %.2e dB, dR %.2e\n", name.c_str(), dT, dR);
        INFO(name);
        CHECK(DX.dT < (ml ? 2e-4 : dense ? 0.06 : 6e-3));
        CHECK(DX.dR < (ml ? 3e-7 : dense ? 5e-3 : 6e-4));
        CHECK(DS.dT < (ml ? 2e-4 : 3e-5));
        CHECK(DS.dR < 3e-7);
        CHECK(std::max(DX.RT, DS.RT) <= 1.0 + 1e-6);
        CHECK(dT < (ml ? 2e-4 : 5e-4));
        CHECK(dR < (ml ? 1e-6 : 2e-5));
        rows((name + ", traced solid").c_str(), X);
        if (ml && qd_diag())
        {
            const double mpp = col(r, "e") * RHO0 * thk;
            const auto bare = octaves(G, [&](size_t q)
                                      { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * std::pow(10.0, -0.1 * mass_TL(QPI * G.f[q] * mpp / (RHO0 * CS), std::cos(rad(G.th[i])), G.f[q], col(r, "side"))); return t; });
            std::printf("# diag: %s, physical models: bare limp mass law of %.1f kg/m2 at side %.2f m, random incidence (dB)\noctave,library_tl,bare_mass_law_tl\n",
                        name.c_str(), mpp, col(r, "side"));
            for (int o = 0; o < 9; ++o)
                std::printf("%g,%.2f,%.2f\n", OCT[o], X.t8[o], -10.0 * std::log10(bare[o]));
        }
        for (const Measured &M : refs)
            tier2(name.c_str(), X, M);
        return X;
    }
    // A combination: the stack on a grid per node against ref_auto, bounds on |dT| (dB) and |dR|, R + T <= 1
    static Out check_stack(const char *name, const Stack &st, double bT, double bR, const Grid &G = grid())
    {
        const Out O = run(st, G);
        const Dev D = against(name, O, st);
        INFO(name);
        CHECK(D.dT < bT);
        CHECK(D.dR < bR);
        CHECK(D.RT <= 1.0 + 1e-6);
        return O;
    }
    // The insulation a layer adds to a wall (with minus without, so the wall's fitted knobs cancel), library against
    // the fluid layer on a limp plate of the wall's surface mass (e rho0 thk), at 0 deg and at random incidence; and
    // the layer alone in air (asserted against the fluid layer). Printed, not asserted (engine stage-2b issue 1).
    static void added_insulation(const char *name, const Stack &layer, const Stack &wall, const Grid &G = grid())
    {
        Stack both = layer;
        both.insert(both.end(), wall.begin(), wall.end());
        const Out A = run(both, G), B = run(wall, G), W = run(layer, G);
        const double mpp = col(wall[0].row, "e") * RHO0 * wall[0].thk;
        auto plate = [&](const Stack &L, size_t q, double th) { double T, R; ref_tmm(L, G.f[q], th, T, R, PLATE, nullptr, mpp); return T; };
        auto at0 = [&](const Out &O) { return octaves(G, [&](size_t q) { return std::norm(O.T[G.na][q]); }); }; // the report angle 0 deg
        auto ri = [&](const Out &O) { return octaves(G, [&](size_t q) { return O.tauf[q]; }); };
        auto pl0 = [&](const Stack &L) { return octaves(G, [&](size_t q) { return plate(L, q, 0.0); }); };
        auto plr = [&](const Stack &L) { return octaves(G, [&](size_t q) { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * plate(L, q, G.th[i]); return t; }); };
        QD_DIAG("# diag: the insulation %s adds (dB): library and the fluid layer on a limp plate of %.1f kg/m2\nincidence,quantity", name, mpp);
        for (double f : OCT)
            QD_DIAG(",%g", f);
        const std::array<double, 9> one = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const std::vector<std::tuple<const char *, const char *, std::array<double, 9>, std::array<double, 9>>> L = {
            {"0 deg", "library", at0(A), at0(B)}, {"0 deg", "plate_tmm", pl0(layer), pl0({})}, {"random", "library", ri(A), ri(B)},
            {"random", "plate_tmm", plr(layer), plr({})}, {"0 deg", "wool_alone_library", at0(W), one}, {"random", "wool_alone_library", ri(W), one}};
        for (const auto &l : L)
        {
            QD_DIAG("\n%s,%s", std::get<0>(l), std::get<1>(l));
            for (int o = 0; o < 9; ++o)
            {
                QD_DIAG(",%.2f", 10.0 * std::log10(std::get<3>(l)[o] / std::get<2>(l)[o]));
                CHECK(std::isfinite(10.0 * std::log10(std::get<3>(l)[o] / std::get<2>(l)[o])));
            }
        }
        QD_DIAG("\n%s", DEV_HEAD);
        const Dev D = against((std::string(name) + ": the layer alone in air").c_str(), W, layer);
        CHECK(D.dT < 3e-5);
        CHECK(D.dR < 3e-7);
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
}

TEST_CASE("Materials rig - isolated layers: sheet equals traced solid, against the plate rule and the fluid layer", "[acoustic][materials]")
{
    // Why: every material case is this call. A mass-law pane against the plate rule, a porous layer and a limp leaf
    // against the transfer matrix of their columns in air; each as a traced solid on the full grid and as a sheet on
    // the report angles, which must agree node by node.
    using namespace mt_rows;
    mt::check_material("glass 4 mm", GLASS4);
    mt::check_material("glass wool 50 mm", WOOL);
    mt::check_material("board 12.5 mm", BOARD);
}

TEST_CASE("Materials rig - combinations against the fluid layers and the bonded plate; forbidden stacks", "[acoustic][materials]")
{
    // Why: combinations are a room-side sheet on traced layers or two sheets on one face (the first face the room
    // side), and leaves on a box of air (page 10.4). Per node against the transfer matrix of the layers in air, on the
    // wall's half-space behind the pad of its tf (wool on concrete: the pad is formed on the medium in front, notes 9.2,
    // up to 6.3e-3), or the bonded plate. The filled box follows the exchange between box and fill as paths: 12 angles,
    // its alpha within 0.0041 of 192 angles x 6 tones per third (grid_probe.py, 2026-10-23). Forbidden stacks throw:
    // two traced wave media back to back (4.9 dB off), two mass-law sheets (17 dB), a sheet behind a solid, a lossless
    // wave medium with a critical angle as a solid, three sheets.
    using namespace mt_rows;
    using mt::SHEET;
    using mt::SOLID;
    QD_DIAG("\n# diag: combinations per node against the fluid layers or the bonded plate\n%s", mt::DEV_HEAD);
    struct Case
    {
        const char *name;
        mt::Stack st;
        double bT, bR;
        int na;
    };
    const std::vector<Case> C = {
        {"glass wool | board, sheet | sheet", {{WOOL, SHEET, 0.05}, {BOARD, SHEET, 0.0125}}, 3e-5, 3e-7, 50},
        {"board | glass wool, sheet | sheet", {{BOARD, SHEET, 0.0125}, {WOOL, SHEET, 0.05}}, 3e-5, 3e-7, 50},
        {"glass wool | board, sheet | solid", {{WOOL, SHEET, 0.05}, {BOARD, SOLID, 0.0125}}, 0.15, 1e-2, 50},
        {"board | glass wool, sheet | solid", {{BOARD, SHEET, 0.0125}, {WOOL, SOLID, 0.05}}, 8e-3, 1e-2, 50},
        {"glass 4 | glass 4 bonded, sheet | solid", {{GLASS4, SHEET, 0.004}, {GLASS4, SOLID, 0.004}}, 2e-4, 3e-7, 50},
        {"board on 25 mm of air on the wall (tf 0.02)", {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.025}, {wall(0.02), SOLID, 0.2}}, 1.0, 4e-4, 50},
        {"MDF on 20 mm of air, 75 mm rock wool, wall (tf 0)", {{MDF, SHEET, 0.006}, {AIRBOX, SOLID, 0.02}, {ROCKWOOL, SOLID, 0.075}, {wall(0.0), SOLID, 0.2}}, 1.0, 5e-3, 12},
        {"glass wool on concrete 152 mm (its tf)", {{WOOL, SHEET, 0.05}, {CONCRETE152, SOLID, 0.152}}, 1.0, 0.015, 50},
        {"glass wool on a tf-0 wall", {{WOOL, SHEET, 0.05}, {wall(0.0), SOLID, 0.2}}, 1.0, 3e-7, 50},
    };
    for (const Case &c : C)
    {
        const mt::Out O = mt::check_stack(c.name, c.st, c.bT, c.bR, mt::grid(c.na));
        if (c.st.size() > 2)
            mt::rows(c.name, O);
    }
    for (const mt::Stack &bad : {mt::Stack{{WOOL, SOLID, 0.05}, {BOARD, SOLID, 0.0125}}, mt::Stack{{GLASS4, SHEET, 0.004}, {GLASS4, SHEET, 0.004}},
                                 mt::Stack{{CONCRETE152, SOLID, 0.152}, {WOOL, SHEET, 0.05}}, mt::Stack{{WATER, SOLID, 0.1}},
                                 mt::Stack{{WOOL, SHEET, 0.05}, {BOARD, SHEET, 0.0125}, {BOARD, SHEET, 0.0125}}})
        CHECK_THROWS_AS(mt::scene(bad), std::invalid_argument);
    CHECK_NOTHROW(mt::scene({{WATER, SHEET, 0.1}}));
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
    // and not asserted (the library adds up to 8 dB too much at 31.5 Hz: engine stage-2b issue 1); the wool alone in
    // air asserted against the same matrix, so both sides map the layer alike.
    using namespace mt_rows;
    mt::added_insulation("50 mm glass wool to 152 mm concrete", {{WOOL, mt::SHEET, 0.05}}, {{CONCRETE152, mt::SOLID, 0.152}});
}

// ============================================================================================================
// The table and its materials
// ============================================================================================================

TEST_CASE("Materials table - every row has a material case and every column a valid value", "[acoustic][materials]")
{
    // Why: the table and its tests must not drift. Every row has a case "Material - <name>" and every such case a row;
    // all 26 columns; fRef, a, e positive, the columns the library requires non-negative; lowercase snake_case names.
    const mt::Table &T = mt::table();
    REQUIRE(!T.names.empty());
    for (const char *k : mt::COLS)
        CHECK(T.prop.count(k) == 1);
    std::set<std::string> cases;
    for (const Catch::TestCaseInfo *tc : Catch::getRegistryHub().getTestCaseRegistry().getAllInfos())
        if (tc->name.rfind("Material - ", 0) == 0)
            cases.insert(tc->name.substr(11));
    for (size_t i = 0; i < T.names.size(); ++i)
    {
        const std::string &n = T.names[i];
        INFO("row " << n);
        CHECK(cases.count(n) == 1);
        CHECK(std::all_of(n.begin(), n.end(), [](char ch) { return std::islower((unsigned char)ch) || std::isdigit((unsigned char)ch) || ch == '_'; }));
        for (const char *k : {"fRef", "a", "e"})
            CHECK(T.prop.at(k)[i] > 0.0);
        for (const char *k : {"c", "g", "thk", "att", "alpha", "m", "resF", "resQ", "coiF", "coiQ", "tfF", "side"})
            CHECK(T.prop.at(k)[i] >= 0.0);
    }
    for (const std::string &c : cases)
        CHECK(std::find(T.names.begin(), T.names.end(), c) != T.names.end());
}

TEST_CASE("Material - concrete_cast_152mm", "[acoustic][materials]")
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
