// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_progress, acoustic: cases that evaluate a material with the mass column (m > 0), and one mixed case.

#include "ray_test_helpers.hpp"

TEST_CASE("ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // A porous layer, eps 0.31 - 0.20j / fGHz, 50 mm on a rigid wall 200 mm with tf 0.3 and its
    // mass law (m 30). The direction of the wave in the layer depends on the frequency; what a
    // wave from air carries across the closed face, its transverse wavenumber, does not. Through
    // the tracer, scalar, float, one ray until none is left. Recorded per frequency: the strongest
    // path behind the wall, the direction it is handed on with, the segment, and where it leaves.
    //   PWm  porous solid on the wall     SWm  the same layer as a sheet on the wall
    //   Wm   the wall bare                PWt, SWt, Wt  the same with a wall that has tf and no m
    // Asserted on the layer as a traced solid: every frequency of a band against the same
    // frequency alone, and behind the wall with its mass law the direction and the segment on
    // the incidence angle. The level against the sheet is recorded: the layer's series is
    // followed at this threshold and the strongest path is not the sum.
    const double PI = 3.14159265358979323846, C0 = 299792458.0;
    const float gap = 1.0e-5f;
    const std::complex<double> J(0.0, 1.0);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: porous solid, 2: porous sheet 50 mm, 3: rigid wall tf 0.3 m 30, 4: rigid wall tf 0.3
    pal["a"] = {0.31f, 0.31f, 2.401e-9f, 2.401e-9f};
    pal["b"] = std::vector<float>(4, 0.0f);
    pal["c"] = {(float)(0.20 / 17.98), (float)(0.20 / 17.98), 0.0f, 0.0f};
    pal["d"] = std::vector<float>(4, 0.0f);
    pal["fRef"] = {1.0f, 1.0f, 0.001f, 1.0f};
    pal["thk"] = {0.0f, 0.05f, 0.0f, 0.0f};
    pal["tf"] = {0.0f, 0.0f, 0.3f, 0.3f};
    pal["m"] = {0.0f, 0.0f, 30.0f, 0.0f};

    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::fmat &mesh, arma::uvec &mi)
    {
        const float h = 200.0f;
        mesh.set_size(2 * x.size(), 9);
        mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k])
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            mi(2 * k) = (arma::uword)mat[k], mi(2 * k + 1) = (arma::uword)mat[k];
        }
    };

    struct Sum
    {
        double Tp = 0.0, T1 = 0.0, dir = -1.0, seg = -1.0, dy = 0.0; // T1: the strongest path, with its direction, segment and lateral walk
        int nT = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float x_back, const std::vector<double> &fGHz) -> std::vector<Sum>
    {
        const size_t nf = fGHz.size();
        std::vector<Sum> S(nf);
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int g = 0; g < 40; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)12, 100, 100, 0, -250.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue;
                const float *first = paths[i].coord(0), *last = paths[i].coord(ns - 1);
                if (!(dest(i, 0) > orig(i, 0) && last[0] > x_back - 1.0e-3f))
                    continue;
                const double sx = (double)dest(i, 0) - orig(i, 0), sy = (double)dest(i, 1) - orig(i, 1), sz = (double)dest(i, 2) - orig(i, 2);
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    const double P = (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
                    S[q].Tp += P, ++S[q].nT;
                    if (P > S[q].T1)
                    {
                        const double dx = pd(i, 0, q), dyv = pd(i, 1, q), dz = pd(i, 2, q);
                        S[q].T1 = P;
                        S[q].dir = std::atan2(std::sqrt(dyv * dyv + dz * dz), std::abs(dx)) * 180.0 / PI;
                        S[q].seg = std::atan2(std::sqrt(sy * sy + sz * sz), std::abs(sx)) * 180.0 / PI;
                        S[q].dy = (double)last[1] - (double)first[1];
                    }
                }
            }
        }
        return S;
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Geo
    {
        const char *name;
        arma::fmat mesh;
        arma::uvec mi;
        float x_back;
    };
    std::vector<Geo> geo(6);
    for (int w = 0; w < 2; ++w)
    {
        const int W = 3 + w;
        Geo &A = geo[3 * w], &B = geo[3 * w + 1], &Cc = geo[3 * w + 2];
        A.name = w ? "PWt" : "PWm", A.x_back = 0.25f;
        build({0.0f, 0.05f, 0.05f + gap, 0.25f + gap}, {1, 1, W, W}, {false, true, false, true}, A.mesh, A.mi);
        B.name = w ? "SWt" : "SWm", B.x_back = 0.2f;
        build({0.0f, gap, 0.2f + gap}, {2, W, W}, {false, false, true}, B.mesh, B.mi);
        Cc.name = w ? "Wt" : "Wm", Cc.x_back = 0.2f;
        build({0.0f, 0.2f}, {W, W}, {false, true}, Cc.mesh, Cc.mi);
    }

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: closed-face hand-on through ray_progress, scalar, 12 interactions, thin_slab_threshold 0.15. PW: porous\n");
    std::printf("# solid 50 mm on a rigid wall 200 mm with tf 0.3; SW: the same layer as a sheet; W: the wall bare. m: the wall\n");
    std::printf("# with its mass law (m 30); t: without. band: the frequencies of the run, the base first; f: the frequency of\n");
    std::printf("# the row. Tp: the power sum behind the wall, dB; T1: the strongest path; dir: the direction that path is\n");
    std::printf("# handed on with at this frequency; seg: its segment, deg against the normal; dy: its lateral walk, m\n");
    std::printf("geo,tracer,band,f,theta,nT,Tp_dB,T1_dB,dir,seg,dy_m\n");
#endif
    const std::vector<std::vector<double>> bands = {{0.5}, {1.0}, {2.0}, {1.0, 0.5}, {1.0, 2.0}, {0.5, 1.0}, {2.0, 1.0}};
    std::map<std::string, Sum> alone; // geometry, tracer, frequency, angle
    auto key = [](const char *g, int refr, double f, double thd)
    { return std::string(g) + "," + std::to_string(refr) + "," + std::to_string((int)(10.0 * f)) + "," + std::to_string((int)thd); };
    for (const Geo &G : geo)
        for (int refr = 0; refr < 2; ++refr)
            for (const auto &band : bands)
                for (double thd : {30.0, 60.0, 80.0})
                {
                    std::vector<Sum> Z = trace(G.mesh, G.mi, thd, refr != 0, G.x_back, band);
                    for (size_t q = 0; q < band.size(); ++q)
                    {
                        INFO(G.name << ", refr " << refr << ", band of " << band.size() << " based on " << band[0] << ", f " << band[q] << ", theta " << thd);
                        CHECK(Z[q].nT >= 1);
                        if (band.size() == 1)
                            alone[key(G.name, refr, band[q], thd)] = Z[q];
                        else // a frequency of a band is what it is alone
                            CHECK(std::abs(pdB(Z[q].Tp) - pdB(alone.at(key(G.name, refr, band[q], thd)).Tp)) < 2.0e-3);
                        if (std::string(G.name) == "PWm" || std::string(G.name) == "SWm" || std::string(G.name) == "Wm")
                        {
                            CHECK(std::abs(Z[q].dir - thd) < 1.0e-3); // held through the wall at the incidence angle
                            CHECK(std::abs(Z[q].seg - thd) < 1.0e-3);
                        }
#ifdef QD_TEST_VERBOSE
                        char bn[32];
                        if (band.size() == 2)
                            std::snprintf(bn, sizeof(bn), "%.1f+%.1f", band[0], band[1]);
                        else
                            std::snprintf(bn, sizeof(bn), "%.1f", band[0]);
                        std::printf("%s,%d,%s,%.1f,%.0f,%d,%.4f,%.4f,%.4f,%.4f,%+.5f\n", G.name, refr ? 5 : 4, bn, band[q], thd, Z[q].nT, pdB(Z[q].Tp), pdB(Z[q].T1),
                                    Z[q].dir, Z[q].seg, Z[q].dy);
#endif
                    }
                }
}

TEST_CASE("ray_progress - the bonded plate at any depth where the slopes agree", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // Plate depth through the tracer. Bonded mass-law stacks, rigid and lossless with a tf leak,
    // scalar, float, one ray until none is left, both tracers, alone and in a band. The mass a
    // stack paid is the strongest path behind it against the same path on a palette whose mass
    // arguments are taken to nothing (fRef 1e9), per frequency. The stack is one plate: on the twin
    // it passes a, the smaller absorption of its two outer faces (their tf), and with its mass
    // a tau / (a + tau - a tau) (the plate rule, md/acoustic.md 10 N1), tau the plate of every body
    // from the column definitions. Asserted: what the rule pays above a, at any depth where the
    // slopes agree, and from the side on which they agree where they do not; a frequency of a band
    // pays what it pays alone, since the state words of a run are the base frequency's.
    const double PI = 3.14159265358979323846;
    const float gap = 1.0e-5f;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    // 1 A20, 2 B20, 3 C20, 4 C30, 5 plaster, 6 brick, 7 lining L20 15 mm, 8 sheet S20 12 mm, 9 sheet HS 8 mm with B20's permittivity
    const std::vector<float> eps = {2.401e-9f, 2.1609e-8f, 6.0e-9f, 6.0e-9f, 1.2e-8f, 4.0e-9f, 5.0e-9f, 1.0e-8f, 2.1609e-8f};
    const std::vector<float> tfv = {0.10f, 0.20f, 0.15f, 0.15f, 0.20f, 0.10f, 0.15f, 0.20f, 0.20f};
    const std::vector<float> thk = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.015f, 0.012f, 0.008f};
    const std::vector<float> mv = {20.0f, 20.0f, 20.0f, 30.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f};
    const double fR = 0.001;
    auto pal_of = [&](bool twin) -> Pal
    {
        Pal p;
        p["a"] = eps, p["b"] = std::vector<float>(eps.size(), 0.0f), p["c"] = std::vector<float>(eps.size(), 0.0f);
        p["d"] = std::vector<float>(eps.size(), 0.0f), p["thk"] = thk, p["tf"] = tfv, p["m"] = mv;
        p["fRef"] = std::vector<float>(eps.size(), twin ? 1.0e9f : (float)fR);
        return p;
    };
    const Pal pal = pal_of(false), pal_twin = pal_of(true);

    // A stack along x as tokens: O / C open and close a solid, S a sheet, g advances by d
    struct Tok
    {
        char k;
        int mat;
        float d;
    };
    auto O = [](int m) { return Tok{'O', m, 0.0f}; };
    auto C = [](int m) { return Tok{'C', m, 0.0f}; };
    auto S = [](int m) { return Tok{'S', m, 0.0f}; };
    auto G = [](float d) { return Tok{'g', 0, d}; };
    struct Geo
    {
        arma::fmat mesh;
        arma::uvec mi;
        float x_back = 0.0f;
        double a = 1.0;                          // the smaller tf of the two outer faces
        std::vector<std::array<double, 3>> body; // m, fRef, normal thickness
    };
    auto build = [&](std::vector<Tok> t, bool rev) -> Geo
    {
        if (rev)
        {
            std::reverse(t.begin(), t.end());
            for (Tok &q : t)
                q.k = (q.k == 'O') ? 'C' : ((q.k == 'C') ? 'O' : q.k);
        }
        Geo g;
        std::vector<float> xs;
        std::vector<int> mats;
        std::vector<bool> flip;
        float x = 0.0f, leg = 0.0f;
        int cur = 0;
        bool last_face = false;
        for (const Tok &q : t)
        {
            if (q.k == 'g')
            {
                x += q.d, last_face = false;
                if (cur)
                    leg += q.d;
                continue;
            }
            if (last_face)
                x += gap;
            xs.push_back(x), mats.push_back(q.mat), flip.push_back(q.k == 'C'), last_face = true;
            if (q.k != 'O' && cur && leg > 0.0f)
                g.body.push_back({(double)mv[(size_t)cur - 1], fR, (double)leg}), leg = 0.0f;
            if (q.k == 'O')
                cur = q.mat, leg = 0.0f;
            else if (q.k == 'C')
                cur = 0;
            else
                g.body.push_back({(double)mv[(size_t)q.mat - 1], fR, (double)thk[(size_t)q.mat - 1]});
        }
        g.x_back = x;
        g.a = std::min((double)tfv[(size_t)t.front().mat - 1], (double)tfv[(size_t)t.back().mat - 1]);
        const float h = 200.0f;
        g.mesh.set_size(2 * xs.size(), 9);
        g.mi.set_size(2 * xs.size());
        for (size_t k = 0; k < xs.size(); ++k)
        {
            if (!flip[k])
                g.mesh.row(2 * k) = arma::frowvec{xs[k], -h, -h, xs[k], -h, h, xs[k], h, -h},
                g.mesh.row(2 * k + 1) = arma::frowvec{xs[k], h, -h, xs[k], -h, h, xs[k], h, h};
            else
                g.mesh.row(2 * k) = arma::frowvec{xs[k], -h, -h, xs[k], h, -h, xs[k], -h, h},
                g.mesh.row(2 * k + 1) = arma::frowvec{xs[k], h, -h, xs[k], h, h, xs[k], -h, h};
            g.mi(2 * k) = (arma::uword)mats[k], g.mi(2 * k + 1) = (arma::uword)mats[k];
        }
        return g;
    };
    // The plate of every body, from the column definitions
    auto plate_of = [](const Geo &g, double fGHz, double ct)
    {
        double X = 0.0;
        for (const auto &b : g.body)
            X += std::pow((fGHz / b[1]) * b[2], 0.05 * b[0]);
        if (!(X > 1.0))
            return 0.0;
        const double A = 20.0 * std::log10(X), x = X * ct;
        return (ct >= 1.0) ? A : A * std::log10(1.0 + x * x) / std::log10(1.0 + X * X);
    };

    // The strongest path behind the wall, power per frequency, and the number of paths
    auto trace = [&](const Pal &pl, const Geo &g, double thd, bool refr, const std::vector<double> &fGHz, int &nT) -> std::vector<double>
    {
        const size_t nf = fGHz.size();
        std::vector<double> T1(nf, 0.0);
        nT = 0;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        for (int it = 0; it < 40; ++it)
        {
            quadriga_lib::ray_progress(g.mesh, g.mi, pl, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)12, 100, 100, 0, -250.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(it + 1))
                    continue;
                const float *last = paths[i].coord(ns - 1);
                if (!(dest(i, 0) > orig(i, 0) && last[0] > g.x_back - 1.0e-3f))
                    continue;
                ++nT;
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    T1[q] = std::max(T1[q], (double)xc[0] * xc[0] + (double)xc[1] * xc[1]);
                }
            }
        }
        return T1;
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Stack
    {
        const char *name;
        std::vector<Tok> t;
        int exact; // bit 0: the plate as named, bit 1: from the other side; clear: recorded
    };
    // The plate's entry face is re-formed from the state, which is one level deep: on the last leg the
    // previous-medium word names the body before it, whose face stands in for the stack's first one
    // (md/acoustic.md 7). Where that face absorbs more than the first (A20 100 | B20 | C20 as named,
    // the lining as named) or less (plaster | brick | plaster) the level is off by the ratio, at most
    // 3.01 dB in this palette, and the row is recorded with that bound; mixed slopes as before.
    const std::vector<Stack> stacks = {
        {"A20 100 | B20 100 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(3), G(0.1f), C(3)}, 2},
        {"A20 100 | B20 13 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.013f), C(2), O(3), G(0.1f), C(3)}, 2},
        {"A20 100 | B20 100 | C30 50", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(4), G(0.05f), C(4)}, 0},
        {"plaster 15 | brick 115 | plaster 15", {O(5), G(0.015f), C(5), O(6), G(0.115f), C(6), O(5), G(0.015f), C(5)}, 0},
        {"lining L20 > B20 60 | S20 | B20 40", {S(7), O(2), G(0.06f), S(8), G(0.04f), C(2)}, 2},
        {"merged: B20 60 | HS 8 | B20 40", {O(2), G(0.06f), S(9), G(0.04f), C(2)}, 3},
    };

#ifdef QD_TEST_VERBOSE
    std::printf("\n# diag: plate depth through ray_progress, scalar, float, 12 interactions. dir: f as named, r from the other side.\n");
    std::printf("# band: the frequencies of the run in GHz, the base first; f: the frequency of the row. plate: every body as one\n");
    std::printf("# plate; paid: the strongest path behind the stack against the same path without the mass arguments, dB;\n");
    std::printf("# rule: what the plate rule pays above a, dB(a) - dB(a tau / (a + tau - a tau)); over: paid - rule; nT: paths behind\n");
    std::printf("stack,dir,tracer,band,f,theta,nT,a,plate,rule,paid,over\n");
#endif
    const std::vector<std::vector<double>> bands = {{1.0}, {0.25}, {4.0}, {1.0, 0.25}, {1.0, 4.0}, {0.25, 1.0}, {4.0, 1.0}};
    std::map<std::string, double> alone;
    for (const Stack &St : stacks)
        for (int rev = 0; rev < 2; ++rev)
        {
            const Geo g = build(St.t, rev != 0);
            for (int refr = 0; refr < 2; ++refr)
                for (const auto &band : bands)
                    for (double thd : {0.0, 60.0})
                    {
                        int nT = 0, nZ = 0;
                        const std::vector<double> T = trace(pal, g, thd, refr != 0, band, nT), Z = trace(pal_twin, g, thd, refr != 0, band, nZ);
                        for (size_t q = 0; q < band.size(); ++q)
                        {
                            const double paid = pdB(Z[q]) - pdB(T[q]), plate = plate_of(g, band[q], std::cos(thd * PI / 180.0));
                            const double tau = std::pow(10.0, -0.1 * plate), rule = pdB(g.a) - pdB(g.a * tau / (g.a + tau - g.a * tau));
                            INFO(St.name << ", dir " << rev << ", refr " << refr << ", band of " << band.size() << " based on " << band[0] << ", f " << band[q] << ", theta " << thd);
                            CHECK(nT >= 1);
                            CHECK(nZ >= 1);
                            const std::string key = std::string(St.name) + "," + std::to_string(rev) + "," + std::to_string(refr) + "," +
                                                    std::to_string((int)(100.0 * band[q])) + "," + std::to_string((int)thd);
                            if (band.size() == 1)
                                alone[key] = paid;
                            else // a frequency of a band pays what it pays alone
                                CHECK(std::abs(paid - alone.at(key)) < 2.0e-3);
                            // exactly at 0 deg the rigid face is open and passes its Fresnel transmittance besides tf
                            // (0.0106 dB for B20)
                            if ((St.exact >> rev) & 1)
                                CHECK(std::abs(paid - rule) < 0.02);
                            else if (St.t[2].mat != 4 && St.t.back().mat != 4) // the depth of the state, the slopes agreeing
                                CHECK(std::abs(paid - rule) < 3.02);
#ifdef QD_TEST_VERBOSE
                            char bn[32];
                            if (band.size() == 2)
                                std::snprintf(bn, sizeof(bn), "%.2f+%.2f", band[0], band[1]);
                            else
                                std::snprintf(bn, sizeof(bn), "%.2f", band[0]);
                            std::printf("%s,%s,%d,%s,%.2f,%.0f,%d,%.2f,%.4f,%.4f,%.4f,%+.4f\n", St.name, rev ? "r" : "f", refr ? 5 : 4, bn, band[q], thd, nT, g.a,
                                        plate, rule, paid, paid - rule);
#endif
                        }
                    }
        }
}

TEST_CASE("ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has - the mass-law part", "[acoustic]")
{
    // The part of "ray_progress - refracting tracer: the segment out of a lossy medium is the
    // direction the wave has" that evaluates a material with the mass column (m > 0); the description
    // is there.
    const double PI = 3.14159265358979323846;
    const float L = 0.10f, gap = 1.0e-5f;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal pal; // 1: eps 5.24-0.63j, 2: 5.24, 3: sheet 5.24-0.63j 6 mm, 4: 9.0-1.20j, 5: 1.50-0.40j, 6: 1.50
    pal["a"] = {5.24f, 5.24f, 5.24f, 9.0f, 1.50f, 1.50f};
    pal["b"] = std::vector<float>(6, 0.0f);
    pal["c"] = {(float)(0.63 / 17.98), 0.0f, (float)(0.63 / 17.98), (float)(1.20 / 17.98), (float)(0.40 / 17.98), 0.0f};
    pal["d"] = std::vector<float>(6, 0.0f);
    pal["fRef"] = std::vector<float>(6, 1.0f);
    pal["thk"] = {0.0f, 0.0f, 0.006f, 0.0f, 0.0f, 0.0f};

    Pal rig; // DENSE: eps 1.50 with the mass column, a mass-law solid the wave is refracted into
    rig["a"] = {1.5f};
    rig["fRef"] = {0.001f};
    rig["tf"] = {0.0f};
    rig["m"] = {20.0f};
    rig["thk"] = {0.0f};

    struct Wall
    {
        arma::fmat mesh;
        arma::uvec mi;
    };
    auto build = [](const std::vector<float> &x, const std::vector<int> &mat, const std::vector<bool> &flip) -> Wall
    {
        const float h = 200.0f;
        Wall W;
        W.mesh.set_size(2 * x.size(), 9);
        W.mi.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            if (!flip[k]) // normal along -x, front side towards the source
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h},
                W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                W.mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h},
                W.mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
            W.mi(2 * k) = (arma::uword)mat[k], W.mi(2 * k + 1) = (arma::uword)mat[k];
        }
        return W;
    };

    struct Gen // the launch configuration after one call
    {
        arma::fmat orig, dest, dir, trivec, tridir, hit;
        arma::Col<short> cur, prev;
        std::vector<std::string> types;
        arma::uword n() const { return orig.n_rows; }
        arma::frowvec seg(arma::uword i) const
        {
            arma::frowvec d = dest.row(i) - orig.row(i);
            return d / nrm3(d);
        }
    };
    struct Run
    {
        std::vector<Gen> g;
        arma::frowvec k;     // launched spine direction
        arma::fmat v0;       // launched vertex directions, one per row
    };
    auto trace = [&](const Pal &P, const Wall &W, double thd, bool refr, int n_gen, int max_ref, float slab,
                     bool beam, int force_gen) -> Run
    {
        Run R;
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        R.k = arma::frowvec{(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {1.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros), trivec, tridir;
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * R.k;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * R.k;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = R.k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        if (beam)
        {
            trivec.zeros(1, 9), tridir.zeros(1, 9), R.v0.zeros(3, 3);
            arma::frowvec u = arma::cross(arma::frowvec{0.0f, 0.0f, 1.0f}, R.k);
            u /= nrm3(u);
            arma::frowvec v = arma::cross(R.k, u);
            const float ha = 0.5f * (float)(PI / 180.0);
            for (int j = 0; j < 3; ++j)
            {
                const float phi = (float)(2.0 * PI * (double)j / 3.0);
                arma::frowvec vj = std::cos(ha) * R.k + std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                vj /= nrm3(vj);
                for (int c = 0; c < 3; ++c)
                    tridir(0, 3 * j + c) = vj(c), trivec(0, 3 * j + c) = r0 * (vj(c) - R.k(c)), R.v0(j, c) = vj(c);
            }
        }
        for (int g = 0; g < n_gen; ++g)
        {
            std::vector<bool> ff(orig.n_rows, g == force_gen);
            quadriga_lib::ray_progress(W.mesh, W.mi, P, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       beam ? &trivec : nullptr, beam ? &tridir : nullptr, nullptr, nullptr,
                                       20, (uint8_t)max_ref, 10, (uint8_t)(force_gen >= 0 ? 2 : 0), -300.0f, 3.0f, slab, refr, true,
                                       nullptr, nullptr, nullptr, force_gen >= 0 ? &ff : nullptr);
            if (orig.n_rows == 0)
                break;
            Gen G;
            G.orig = orig, G.dest = dest, G.dir = pd.slice(0), G.cur = cur, G.prev = prev;
            if (beam)
                G.trivec = trivec, G.tridir = tridir;
            G.hit.zeros(orig.n_rows, 3);
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                std::string ty;
                for (auto c : paths[i].interaction_type_codes())
                    ty += std::to_string((unsigned)c) + "/";
                G.types.push_back(ty);
                if (paths[i].n_seg())
                    for (int c = 0; c < 3; ++c)
                        G.hit(i, c) = paths[i].coord(paths[i].n_seg() - 1)[c];
            }
            R.g.push_back(G);
        }
        return R;
    };
    auto dmax = [](const arma::frowvec &a, const arma::frowvec &b) -> double
    {
        double d = 0.0;
        for (int c = 0; c < 3; ++c)
            d = std::max(d, (double)std::abs(a(c) - b(c)));
        return d;
    };
    auto apart = [&](const arma::frowvec &a, const arma::frowvec &b) -> double // degrees, well conditioned near zero
    {
        double cx = (double)a(1) * b(2) - (double)a(2) * b(1), cy = (double)a(2) * b(0) - (double)a(0) * b(2),
               cz = (double)a(0) * b(1) - (double)a(1) * b(0);
        return std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), (double)a(0) * b(0) + (double)a(1) * b(1) + (double)a(2) * b(2)) * 180.0 / PI;
    };
    auto find = [](const Gen &G, const std::string &ty) -> long long
    {
        for (size_t i = 0; i < G.types.size(); ++i)
            if (G.types[i] == ty)
                return (long long)i;
        return -1;
    };
    auto raw = [](const Gen &G, arma::uword i)
    {
        QD_DIAG("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", G.orig(i, 0), G.orig(i, 1), G.orig(i, 2), G.dest(i, 0), G.dest(i, 1), G.dest(i, 2));
    };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    const double TOL = 1.0e-6;

    // ------------------------------------------------------------------------------------------
    // Recorded: a mass-law solid the wave is refracted into (issue 15), and a second frequency
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: trajectory, recorded. DENSE, eps 1.50 with m 20, 30 deg: ray_mesh_interact holds the wave\n");
    QD_DIAG("# undeviated in a mass-law medium, and the segment behind it is not this item's\n");
    QD_DIAG("tracer,seg_deg,dir_deg,raw\n");
    for (bool refr : {true, false})
    {
        Wall W = build({0.0f, L}, {1, 1}, {false, true});
        Run R = trace(rig, W, 30.0, refr, 2, 0, 0.15f, false, -1);
        REQUIRE(R.g.size() == 2);
        arma::frowvec x = {1.0f, 0.0f, 0.0f};
        QD_DIAG("%d,%.4f,%.4f,", refr ? 5 : 4, apart(R.g[1].seg(0), x), apart(R.g[1].dir.row(0), x));
        raw(R.g[1], 0);
        QD_DIAG("\n");
        CHECK(dmax(R.g[1].dir.row(0), R.k) < TOL);
    }
}

TEST_CASE("ray_progress - the plate rule and the absorption peak at every frequency of a band", "[acoustic]")
{
    // N1 and A10 through the tracer: a 12.5 mm board (rigid, m 20, tf 0.05 with a peak tfP 0.25 at 150 Hz on the
    // lumped-resonator law, M5, tfQ = Q_alpha 0.75) as a sheet and as a traced solid, scalar, float, one ray until
    // none is left, a band of five frequencies on the base frequency's geometry. Each frequency: the room side
    // absorbs tf(f), the plate passes a tau / (a + tau - a tau); and the band equals each frequency run alone. At
    // bb336dc the peak columns were not read and the plate passed tf^2 tau.
    ub::OneThread one_thread;
    const double PI = 3.14159265358979323846, FMAP = 299792458.0 / 343.0 * 1.0e-9;
    const double a = 8.1817e-08, thk = 0.0125, fRef = 1.6644e-04, tf0 = 0.05, tfP = 0.25, tfF_hz = 150.0, tfQ = 0.75;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    auto mass_TL = [](double X0, double c)
    {
        if (!(X0 > 1.0) || !(c > 0.0))
            return 0.0;
        double A = 20.0 * std::log10(X0);
        return (c >= 1.0) ? A : A * std::log10(1.0 + X0 * c * X0 * c) / std::log10(1.0 + X0 * X0);
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    struct TR
    {
        double T = 0.0, R = 0.0;
    };
    // one ray through ray_progress until none is left; path powers collected in air, behind (T) and in front (R)
    auto trace = [&](bool sheet, double thd, bool refr, const std::vector<double> &f_hz) -> std::vector<TR>
    {
        const size_t nf = f_hz.size();
        Pal pal;
        pal["a"] = {(float)a}, pal["fRef"] = {(float)fRef}, pal["thk"] = {sheet ? (float)thk : 0.0f}, pal["tf"] = {(float)tf0}, pal["m"] = {20.0f};
        pal["tfP"] = {(float)tfP}, pal["tfF"] = {(float)(tfF_hz * FMAP)}, pal["tfQ"] = {(float)tfQ};
        const float h = 200.0f, xb = sheet ? 0.0f : (float)thk;
        std::vector<float> x = sheet ? std::vector<float>{0.0f} : std::vector<float>{0.0f, xb};
        arma::fmat mesh(2 * x.size(), 9);
        arma::uvec mi(2 * x.size(), arma::fill::ones);
        for (size_t k = 0; k < x.size(); ++k)
            if (k == 0)
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], -h, h, x[k], h, -h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], -h, h, x[k], h, h};
            else
                mesh.row(2 * k) = arma::frowvec{x[k], -h, -h, x[k], h, -h, x[k], -h, h}, mesh.row(2 * k + 1) = arma::frowvec{x[k], h, -h, x[k], h, h, x[k], -h, h};
        const double th = thd * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        arma::frowvec kv = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(f_hz[q] * FMAP * 1.0e9);
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 1 + nf, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * kv, dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * kv;
        arma::fcube pd(1, 3, nf);
        for (size_t q = 0; q < nf; ++q)
            pd.slice(q) = kv;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, nf, true);
        paths[0].set_length(r0);
        std::vector<TR> S(nf);
        for (int g = 0; g < 40; ++g)
        {
            quadriga_lib::ray_progress(mesh, mi, pal, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths, nullptr, nullptr, nullptr, nullptr,
                                       (uint8_t)12, 100, 100, 0, -250.0f, 3.0f, 0.15f, refr, true, nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
            for (arma::uword i = 0; i < orig.n_rows; ++i)
            {
                const size_t ns = paths[i].n_seg();
                if ((cur(i) & (short)0x3FFF) != 0 || ns != (size_t)(g + 1))
                    continue;
                const float *last = paths[i].coord(ns - 1);
                const bool fwd = dest(i, 0) > orig(i, 0);
                for (size_t q = 0; q < nf; ++q)
                {
                    const float *xc = paths[i].xpr_coeff(q);
                    const double P = (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
                    if (fwd && last[0] > xb - 1.0e-3f)
                        S[q].T += P;
                    else if (!fwd && last[0] < 1.0e-3f)
                        S[q].R += P;
                }
            }
        }
        return S;
    };
    const std::vector<double> band = {150.0, 63.0, 250.0, 500.0, 1000.0};
    QD_DIAG("\n# diag: ray_progress, a 12.5 mm board with an absorption peak, sheet (S) and traced solid (X), scalar, float\n");
    QD_DIAG("kind,tracer,theta,f_Hz,alpha,alpha_ref,T_dB,T_rule_dB,band_minus_alone_dB\n");
    for (int kind = 0; kind < 2; ++kind)
        for (bool refr : {false, true})
            for (double thd : {20.0, 50.0})
            {
                std::vector<TR> B = trace(kind == 0, thd, refr, band);
                for (size_t q = 0; q < band.size(); ++q)
                {
                    TR A = trace(kind == 0, thd, refr, {band[q]})[0];
                    const double fG = band[q] * FMAP, c = std::cos(thd * PI / 180.0), x = tfQ * (band[q] / tfF_hz - tfF_hz / band[q]);
                    const double ar = std::min(1.0, tf0 + tfP / (1.0 + x * x)), tau = std::pow(10.0, -0.1 * mass_TL((fG / fRef) * thk, c));
                    const double Tr = ar * tau / (ar + tau - ar * tau);
                    QD_DIAG("%s,%d,%.0f,%.0f,%.5f,%.5f,%.3f,%.3f,%+.2e\n", kind ? "X" : "S", refr ? 5 : 4, thd, band[q], 1.0 - B[q].R, ar, pdB(B[q].T), pdB(Tr),
                            pdB(B[q].T) - pdB(A.T));
                    INFO((kind ? "traced" : "sheet") << ", tracer " << (refr ? 5 : 4) << ", " << thd << " deg, " << band[q] << " Hz");
                    CHECK(std::abs((1.0 - B[q].R) - ar) < 1.0e-3);
                    CHECK(std::abs(pdB(B[q].T) - pdB(Tr)) < 0.02);
                    CHECK(std::abs(pdB(B[q].T) - pdB(A.T)) < 1.0e-4);
                }
            }
}
