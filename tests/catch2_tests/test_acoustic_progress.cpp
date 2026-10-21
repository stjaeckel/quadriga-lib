// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_progress, acoustic: cases that evaluate a material with the mass column (m > 0), and one mixed case.

#include "ray_test_helpers.hpp"

TEST_CASE("ray_progress - the wall behind a traced layer is met at the incidence angle at every frequency of a band", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // A porous layer (eps 0.31 - 0.20j / fGHz, 50 mm) on a rigid wall (200 mm, tf 0.3, with or without the mass column,
    // m 30). The direction of the wave in the layer depends on the frequency; its transverse wavenumber, which a wave
    // from air carries across the closed face, does not. One ray through ray_progress until none is left, scalar,
    // float, both tracers. Asserted, the layer as a traced solid and as a sheet: every frequency of a band passes what
    // it passes alone, and behind the mass-law wall the direction handed on and the segment are at the incidence angle.
    // Printed: the power behind the wall and its strongest path.
    //   PWm  porous solid on the wall     SWm  the same layer as a sheet on the wall
    //   Wm   the wall bare                PWt, SWt, Wt  the same with a wall that has tf and no m
    const double PI = 3.14159265358979323846;
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

    struct Sum
    {
        double Tp = 0.0, T1 = 0.0, dir = -1.0, seg = -1.0, dy = 0.0; // T1: the strongest path, with its direction, segment and lateral walk
        int nT = 0;
    };
    auto trace = [&](const arma::fmat &mesh, const arma::uvec &mi, double thd, bool refr, float x_back, const std::vector<double> &fGHz) -> std::vector<Sum>
    {
        const size_t nf = fGHz.size();
        std::vector<Sum> S(nf);
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        rp::Opt o;
        o.calls = 40, o.max_int = 12, o.min_gain = -250.0f, o.refr = refr;
        rp::walk(mesh, mi, pal, cf, thd, o, [&](const rp::Ray &r)
                 {
            const arma::uword i = r.i;
            const float *first = r.first(), *last = r.last();
            if (!(r.fwd() && last[0] > x_back - 1.0e-3f))
                return false;
            const double sx = (double)r.dest(i, 0) - r.orig(i, 0), sy = (double)r.dest(i, 1) - r.orig(i, 1), sz = (double)r.dest(i, 2) - r.orig(i, 2);
            for (size_t q = 0; q < nf; ++q)
            {
                const float *xc = r.xc(q);
                const double P = (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
                S[q].Tp += P, ++S[q].nT;
                if (P > S[q].T1)
                {
                    const double dx = r.pd(i, 0, q), dyv = r.pd(i, 1, q), dz = r.pd(i, 2, q);
                    S[q].T1 = P;
                    S[q].dir = std::atan2(std::sqrt(dyv * dyv + dz * dz), std::abs(dx)) * 180.0 / PI;
                    S[q].seg = std::atan2(std::sqrt(sy * sy + sz * sz), std::abs(sx)) * 180.0 / PI;
                    S[q].dy = (double)last[1] - (double)first[1];
                }
            }
            return false; });
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
        rp::planes({0.0f, 0.05f, 0.05f + gap, 0.25f + gap}, {1, 1, W, W}, {false, true, false, true}, A.mesh, A.mi);
        B.name = w ? "SWt" : "SWm", B.x_back = 0.2f;
        rp::planes({0.0f, gap, 0.2f + gap}, {2, W, W}, {false, false, true}, B.mesh, B.mi);
        Cc.name = w ? "Wt" : "Wm", Cc.x_back = 0.2f;
        rp::planes({0.0f, 0.2f}, {W, W}, {false, true}, Cc.mesh, Cc.mi);
    }

    QD_DIAG("\n# diag: closed-face hand-on through ray_progress, scalar, 12 interactions, thin_slab_threshold 0.15. PW: porous\n");
    QD_DIAG("# solid 50 mm on a rigid wall 200 mm with tf 0.3; SW: the same layer as a sheet; W: the wall bare. m: the wall\n");
    QD_DIAG("# with its mass law (m 30); t: without. band: the frequencies of the run, the base first; f: the frequency of\n");
    QD_DIAG("# the row. Tp: the power sum behind the wall, dB; T1: the strongest path; dir: the direction that path is\n");
    QD_DIAG("# handed on with at this frequency; seg: its segment, deg against the normal; dy: its lateral walk, m\n");
    QD_DIAG("geo,tracer,band,f,theta,nT,Tp_dB,T1_dB,dir,seg,dy_m\n");
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
                        if (qd_diag())
                        {
                            char bn[32];
                            if (band.size() == 2)
                                std::snprintf(bn, sizeof(bn), "%.1f+%.1f", band[0], band[1]);
                            else
                                std::snprintf(bn, sizeof(bn), "%.1f", band[0]);
                            std::printf("%s,%d,%s,%.1f,%.0f,%d,%.4f,%.4f,%.4f,%.4f,%+.5f\n", G.name, refr ? 5 : 4, bn, band[q], thd, Z[q].nT, pdB(Z[q].Tp), pdB(Z[q].T1),
                                        Z[q].dir, Z[q].seg, Z[q].dy);
                        }
                    }
                }
}

TEST_CASE("ray_progress - the bonded plate at any depth", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // Bonded mass-law stacks (rigid, lossless, a tf leak at their faces) through the tracer: scalar, float, one ray
    // until none is left, both tracers, alone and in a band. What a stack pays is its strongest path behind it against
    // the same path with the mass arguments taken to nothing (fRef 1e9). The stack is one plate: it passes a, the
    // smaller tf of its two outer faces, and with its mass a tau / (a + tau - a tau), tau the term of all its bodies
    // from the column definitions. Asserted at any depth: what that rule pays above a (m > 0 is a switch, so C30 pays
    // what m 20 pays and every stack has one slope), and a frequency of a band pays what it pays alone, since the state
    // words of a run are the base frequency's.
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
        p["side"] = std::vector<float>(eps.size(), 0.0f); // the infinite plate: the case states the bonded run, not the radiation
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
    auto O = [](int m)
    { return Tok{'O', m, 0.0f}; };
    auto C = [](int m)
    { return Tok{'C', m, 0.0f}; };
    auto S = [](int m)
    { return Tok{'S', m, 0.0f}; };
    auto G = [](float d)
    { return Tok{'g', 0, d}; };
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
            X += (b[0] > 0.0) ? (fGHz / b[1]) * b[2] : 0.0; // X0 = (f / fRef) M, the mass law at any m > 0
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
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(fGHz[q] * 1.0e9);
        rp::Opt o;
        o.calls = 40, o.max_int = 12, o.min_gain = -250.0f, o.refr = refr;
        rp::walk(g.mesh, g.mi, pl, cf, thd, o, [&](const rp::Ray &r)
                 {
            if (!(r.fwd() && r.last()[0] > g.x_back - 1.0e-3f))
                return false;
            ++nT;
            for (size_t q = 0; q < nf; ++q)
            {
                const float *xc = r.xc(q);
                T1[q] = std::max(T1[q], (double)xc[0] * xc[0] + (double)xc[1] * xc[1]);
            }
            return false; });
        return T1;
    };
    auto pdB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    struct Stack
    {
        const char *name;
        std::vector<Tok> t;
        int exact; // bit 0: the plate as named, bit 1: from the other side; clear: held to the depth bound
    };
    // The plate's entry face is re-formed from the state, which holds one previous medium: on the last leg that is the
    // body before, whose face stands in for the stack's first one. Where it absorbs more than the first (A20 100 | B20
    // | C20 as named, the lining as named) or less (plaster | brick | plaster), the level is off by that ratio, at most
    // 3.01 dB in this palette, and the row is held to that bound.
    const std::vector<Stack> stacks = {
        {"A20 100 | B20 100 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(3), G(0.1f), C(3)}, 2},
        {"A20 100 | B20 13 | C20 100", {O(1), G(0.1f), C(1), O(2), G(0.013f), C(2), O(3), G(0.1f), C(3)}, 2},
        {"A20 100 | B20 100 | C30 50", {O(1), G(0.1f), C(1), O(2), G(0.1f), C(2), O(4), G(0.05f), C(4)}, 2},
        {"plaster 15 | brick 115 | plaster 15", {O(5), G(0.015f), C(5), O(6), G(0.115f), C(6), O(5), G(0.015f), C(5)}, 0},
        {"lining L20 > B20 60 | S20 | B20 40", {S(7), O(2), G(0.06f), S(8), G(0.04f), C(2)}, 2},
        {"merged: B20 60 | HS 8 | B20 40", {O(2), G(0.06f), S(9), G(0.04f), C(2)}, 3},
    };

    QD_DIAG("\n# diag: plate depth through ray_progress, scalar, float, 12 interactions. dir: f as named, r from the other side.\n");
    QD_DIAG("# band: the frequencies of the run in GHz, the base first; f: the frequency of the row. plate: every body as one\n");
    QD_DIAG("# plate; paid: the strongest path behind the stack against the same path without the mass arguments, dB;\n");
    QD_DIAG("# rule: what the plate rule pays above a, dB(a) - dB(a tau / (a + tau - a tau)); over: paid - rule; nT: paths behind\n");
    QD_DIAG("stack,dir,tracer,band,f,theta,nT,a,plate,rule,paid,over\n");
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
                            else // the depth of the state
                                CHECK(std::abs(paid - rule) < 3.02);
                            if (qd_diag())
                            {
                                char bn[32];
                                if (band.size() == 2)
                                    std::snprintf(bn, sizeof(bn), "%.2f+%.2f", band[0], band[1]);
                                else
                                    std::snprintf(bn, sizeof(bn), "%.2f", band[0]);
                                std::printf("%s,%s,%d,%s,%.2f,%.0f,%d,%.2f,%.4f,%.4f,%.4f,%+.4f\n", St.name, rev ? "r" : "f", refr ? 5 : 4, bn, band[q], thd, nT, g.a,
                                            plate, rule, paid, paid - rule);
                            }
                        }
                    }
        }
}

TEST_CASE("ray_progress - refracting tracer: the segment out of a lossy medium is the direction the wave has - the mass-law part", "[acoustic]")
{
    // A mass-law solid is a lumped plate: the refracting tracer does not bend the wave into it, so the direction
    // handed on inside it is the launched one. A 100 mm solid of eps 1.50 with m 20, one ray at 30 deg through
    // ray_progress, float, both tracers; the segment behind it is printed.
    const double PI = 3.14159265358979323846;
    const float L = 0.10f;
    typedef std::unordered_map<std::string, std::vector<float>> Pal;

    Pal rig;
    rig["a"] = {1.5f};
    rig["fRef"] = {0.001f};
    rig["tf"] = {0.0f};
    rig["m"] = {20.0f};
    rig["thk"] = {0.0f};

    arma::fmat mesh(4, 9);
    arma::uvec mi(4, arma::fill::ones);
    {
        const float h = 200.0f;
        mesh.row(0) = arma::frowvec{0.0f, -h, -h, 0.0f, -h, h, 0.0f, h, -h}, mesh.row(1) = arma::frowvec{0.0f, h, -h, 0.0f, -h, h, 0.0f, h, h};
        mesh.row(2) = arma::frowvec{L, -h, -h, L, h, -h, L, -h, h}, mesh.row(3) = arma::frowvec{L, h, -h, L, h, h, L, -h, h};
    }
    auto apart = [&](const arma::frowvec &a, const arma::frowvec &b) -> double // degrees, well conditioned near zero
    {
        double cx = (double)a(1) * b(2) - (double)a(2) * b(1), cy = (double)a(2) * b(0) - (double)a(0) * b(2),
               cz = (double)a(0) * b(1) - (double)a(1) * b(0);
        return std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), (double)a(0) * b(0) + (double)a(1) * b(1) + (double)a(2) * b(2)) * 180.0 / PI;
    };

    QD_DIAG("\n# diag: trajectory, recorded. DENSE, eps 1.50 with m 20, 30 deg: ray_mesh_interact holds the wave\n");
    QD_DIAG("# undeviated in a mass-law medium, and the segment behind it is not this item's\n");
    QD_DIAG("tracer,seg_deg,dir_deg,raw\n");
    for (bool refr : {true, false})
    {
        const double th = 30.0 * PI / 180.0;
        const float Ox = -1.0f, Oy = 0.37f - (float)std::tan(th), Oz = 0.0f, r0 = 0.1f, len = 300.0f;
        const arma::frowvec k = {(float)std::cos(th), (float)std::sin(th), 0.0f};
        arma::fvec cf = {1.0e9f};
        arma::fmat orig(1, 3), dest(1, 3), acc(1, 2, arma::fill::zeros);
        orig.row(0) = arma::frowvec{Ox, Oy, Oz} + r0 * k;
        dest.row(0) = arma::frowvec{Ox, Oy, Oz} + (r0 + len) * k;
        arma::fcube pd(1, 3, 1);
        pd.slice(0) = k;
        arma::Col<short> prev(1, arma::fill::zeros), cur(1, arma::fill::zeros), buf(1, arma::fill::zeros);
        std::vector<quadriga_lib::path> paths(1);
        paths[0].init(0, 1, true);
        paths[0].set_length(r0);
        int n_gen = 0;
        for (int g = 0; g < 2; ++g, ++n_gen)
        {
            quadriga_lib::ray_progress(mesh, mi, rig, cf, Ox, Oy, Oz, orig, dest, prev, cur, buf, pd, acc, paths,
                                       nullptr, nullptr, nullptr, nullptr,
                                       20, (uint8_t)0, 10, (uint8_t)0, -300.0f, 3.0f, 0.15f, refr, true,
                                       nullptr, nullptr, nullptr, nullptr);
            if (orig.n_rows == 0)
                break;
        }
        REQUIRE(n_gen == 2);
        const arma::frowvec x = {1.0f, 0.0f, 0.0f}, d = dest.row(0) - orig.row(0), dir = pd.slice(0).row(0);
        QD_DIAG("%d,%.4f,%.4f,", refr ? 5 : 4, apart(d / nrm3(d), x), apart(dir, x));
        QD_DIAG("%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", orig(0, 0), orig(0, 1), orig(0, 2), dest(0, 0), dest(0, 1), dest(0, 2));
        QD_DIAG("\n");
        double dm = 0.0;
        for (int c = 0; c < 3; ++c)
            dm = std::max(dm, (double)std::abs(dir(c) - k(c)));
        CHECK(dm < 1.0e-6);
    }
}

TEST_CASE("ray_progress - the plate rule and the absorption peak at every frequency of a band", "[acoustic]")
{
    // The plate rule and the absorption peak through the tracer: a 12.5 mm board (rigid, m 20, tf 0.05 with a peak tfP
    // 0.25 at 150 Hz, tfQ = Q_alpha 0.75) as a sheet and as a traced solid, scalar, float, one ray until none is left,
    // a band of five frequencies on the base frequency's geometry. At each frequency the room side is a locally
    // reacting face, the series impedance zeta = R' + j (1 + R') tfQ (f/tfF - tfF/f) with tfP = 4 R'/(1 + R')^2 (R' >=
    // 1), reflecting (zeta c - 1)/(zeta c + 1) behind the matched pad of tf, so it absorbs a = 1 - (1 - tf)|r_zeta|^2
    // at the cosine c; the plate passes a tau / (a + tau - a tau); and the band equals each frequency run alone.
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
        pal["side"] = {0.0f}; // the infinite plate: the case states the plate rule and the peak, not the radiation
        const float xb = sheet ? 0.0f : (float)thk;
        std::vector<float> x = sheet ? std::vector<float>{0.0f} : std::vector<float>{0.0f, xb};
        arma::fmat mesh;
        arma::uvec mi;
        rp::planes(x, std::vector<int>(x.size(), 1), sheet ? std::vector<bool>{false} : std::vector<bool>{false, true}, mesh, mi);
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(f_hz[q] * FMAP * 1.0e9);
        std::vector<TR> S(nf);
        rp::Opt o;
        o.calls = 40, o.max_int = 12, o.min_gain = -250.0f, o.refr = refr;
        rp::walk(mesh, mi, pal, cf, thd, o, [&](const rp::Ray &r)
                 {
            const float *last = r.last();
            const bool fwd = r.fwd();
            for (size_t q = 0; q < nf; ++q)
            {
                const float *xc = r.xc(q);
                const double P = (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
                if (fwd && last[0] > xb - 1.0e-3f)
                    S[q].T += P;
                else if (!fwd && last[0] < 1.0e-3f)
                    S[q].R += P;
            }
            return false; });
        return S;
    };
    const std::vector<double> band = {150.0, 63.0, 250.0, 500.0, 1000.0};
    QD_DIAG("\n# diag: ray_progress, a 12.5 mm board with an absorption peak, sheet (S) and traced solid (X), scalar, float\n");
    QD_DIAG("kind,tracer,theta,f_Hz,alpha,alpha_ref,T_dB,T_rule_dB,band_minus_alone_dB\n");
    double mA = 0.0, mT = 0.0, mB = 0.0;
    for (int kind = 0; kind < 2; ++kind)
        for (bool refr : {false, true})
            for (double thd : {20.0, 50.0})
            {
                std::vector<TR> B = trace(kind == 0, thd, refr, band);
                for (size_t q = 0; q < band.size(); ++q)
                {
                    TR A = trace(kind == 0, thd, refr, {band[q]})[0];
                    const double fG = band[q] * FMAP, c = std::cos(thd * PI / 180.0), Rp = ((2.0 - tfP) + 2.0 * std::sqrt(1.0 - tfP)) / tfP;
                    const std::complex<double> z(Rp, (1.0 + Rp) * tfQ * (band[q] / tfF_hz - tfF_hz / band[q]));
                    const double ar = 1.0 - (1.0 - tf0) * std::norm((z * c - 1.0) / (z * c + 1.0)), tau = std::pow(10.0, -0.1 * mass_TL((fG / fRef) * thk, c));
                    const double Tr = ar * tau / (ar + tau - ar * tau);
                    QD_DIAG("%s,%d,%.0f,%.0f,%.5f,%.5f,%.3f,%.3f,%+.2e\n", kind ? "X" : "S", refr ? 5 : 4, thd, band[q], 1.0 - B[q].R, ar, pdB(B[q].T), pdB(Tr),
                            pdB(B[q].T) - pdB(A.T));
                    INFO((kind ? "traced" : "sheet") << ", tracer " << (refr ? 5 : 4) << ", " << thd << " deg, " << band[q] << " Hz");
                    mA = std::max(mA, std::abs((1.0 - B[q].R) - ar)), mT = std::max(mT, std::abs(pdB(B[q].T) - pdB(Tr))), mB = std::max(mB, std::abs(pdB(B[q].T) - pdB(A.T)));
                    CHECK(std::abs((1.0 - B[q].R) - ar) < 1.5e-7);
                    CHECK(std::abs(pdB(B[q].T) - pdB(Tr)) < 1.5e-3);
                    CHECK(std::abs(pdB(B[q].T) - pdB(A.T)) < 1.0e-4);
                }
            }
    QD_DIAG("# max |alpha - ref| %.2e, max |T - rule| %.2e dB, max band - alone %.2e dB\n", mA, mT, mB);
}

TEST_CASE("ray_progress - the random-incidence absorption of a peaked lining is the Paris integral of its series impedance", "[acoustic]")
{
    // A peaked face is locally reacting, so its random-incidence absorption is the Paris integral of 1 - |r(theta)|^2
    // with the weight sin 2 theta over 0 to 90 deg, not its normal-incidence value. Plasterboard on 25 mm battens: tf
    // 0.0933 at 1 kHz with tfB 0.1423, tfP 0.2869 at 181.6 Hz, tfQ 1.319, on a rigid mass-law face (two-parameter
    // concrete, m 20), as a sheet (200 mm) and as a traced solid, tracer 4, float, one ray per angle until none is
    // left, 27 thirds from 25 Hz to 10 kHz in one band, 48 Gauss-Legendre angles; an octave is the mean of its three
    // thirds. Reference: the closed form of the Paris integral of a locally reacting face of impedance z = r + j x,
    //   P = (8 r/|z|^2) (1 - (r/|z|^2) ln(1 + 2 r + |z|^2) + ((r^2 - x^2)/(x |z|^2)) atan(x/(1 + r))),
    // behind the pad: alpha = tf + (1 - tf) P. The lining's measured octave values are printed beside it.
    ub::OneThread one_thread;
    const double PI = 3.14159265358979323846, FMAP = 299792458.0 / 343.0 * 1.0e-9;
    const double a = 3.850e-6, mu = 1901.0, fRef = 5.0212e-5, t1k = 0.09332, tB = 0.1423, tfP = 0.2869, tfF_hz = 181.6, tfQ = 1.319, H = 0.2;
    const double tfR = t1k * std::pow(fRef / (1000.0 * FMAP), tB);
    typedef std::unordered_map<std::string, std::vector<float>> Pal;
    const std::vector<double> OCT = {31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000};
    std::vector<double> f3;
    for (double fc : OCT)
        for (int k = -1; k <= 1; ++k)
            f3.push_back(fc * std::pow(2.0, k / 3.0));
    // 48-point Gauss-Legendre on 0 to 90 deg, weights times sin 2 theta, normalized
    const int NG = 48;
    std::vector<double> th(NG), w(NG);
    double ws = 0.0;
    for (int i = 0; i < NG; ++i)
    {
        double x = std::cos(PI * (i + 0.75) / (NG + 0.5)), dp = 0.0;
        for (int it = 0; it < 100; ++it)
        {
            double p0 = 1.0, p1 = x;
            for (int n = 2; n <= NG; ++n)
            {
                const double p2 = ((2.0 * n - 1.0) * x * p1 - (n - 1.0) * p0) / n;
                p0 = p1, p1 = p2;
            }
            dp = NG * (x * p1 - p0) / (x * x - 1.0);
            const double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1.0e-15)
                break;
        }
        th[i] = 45.0 * (x + 1.0), w[i] = 2.0 / ((1.0 - x * x) * dp * dp) * std::sin(2.0 * th[i] * PI / 180.0), ws += w[i];
    }
    for (double &v : w)
        v /= ws;
    // the reflected power per frequency of one ray through ray_progress, every path back in air in front of the face
    auto trace = [&](bool sheet, double thd) -> std::vector<double>
    {
        const size_t nf = f3.size();
        Pal pal;
        pal["a"] = {(float)a}, pal["e"] = {(float)mu}, pal["fRef"] = {(float)fRef}, pal["thk"] = {sheet ? (float)H : 0.0f}, pal["m"] = {20.0f};
        pal["tf"] = {(float)tfR}, pal["tfB"] = {(float)tB}, pal["tfP"] = {(float)tfP}, pal["tfF"] = {(float)(tfF_hz * FMAP)}, pal["tfQ"] = {(float)tfQ};
        const float xb = sheet ? 0.0f : (float)H;
        std::vector<float> x = sheet ? std::vector<float>{0.0f} : std::vector<float>{0.0f, xb};
        arma::fmat mesh;
        arma::uvec mi;
        rp::planes(x, std::vector<int>(x.size(), 1), sheet ? std::vector<bool>{false} : std::vector<bool>{false, true}, mesh, mi);
        arma::fvec cf(nf);
        for (size_t q = 0; q < nf; ++q)
            cf(q) = (float)(f3[q] * FMAP * 1.0e9);
        std::vector<double> R(nf, 0.0);
        rp::Opt o;
        o.calls = 40, o.max_int = 12, o.min_gain = -250.0f;
        rp::walk(mesh, mi, pal, cf, thd, o, [&](const rp::Ray &r)
                 {
            if (r.fwd() || r.last()[0] >= 1.0e-3f)
                return false;
            for (size_t q = 0; q < nf; ++q)
            {
                const float *xc = r.xc(q);
                R[q] += (double)xc[0] * xc[0] + (double)xc[1] * xc[1];
            }
            return false; });
        return R;
    };
    // the reference: the pad of the power law in front of the Paris integral of zeta, per third
    auto paris = [&](double f)
    {
        const double Rp = ((2.0 - tfP) + 2.0 * std::sqrt(1.0 - tfP)) / tfP, r = Rp, x = (1.0 + Rp) * tfQ * (f / tfF_hz - tfF_hz / f), a2 = r * r + x * x;
        const double P = (8.0 * r / a2) * (1.0 - (r / a2) * std::log(1.0 + 2.0 * r + a2) + ((r * r - x * x) / (x * a2)) * std::atan(x / (1.0 + r)));
        const double t = std::min(1.0, std::max(0.0, tfR * std::pow(f * FMAP / fRef, tB)));
        return t + (1.0 - t) * P;
    };
    const std::vector<double> TGT = {std::nan(""), std::nan(""), 0.31, 0.33, 0.14, 0.10, 0.10, 0.12, 0.12}; // the measured octave values, 125 Hz to 4 kHz; 8 kHz as 4 kHz
    QD_DIAG("\n# diag: plasterboard on 25 mm battens, Paris 0-90 deg through ray_progress, octaves, against the closed form\n");
    QD_DIAG("kind,f_Hz,paris,paris_ref,d,target\n");
    double mP = 0.0;
    for (int kind = 0; kind < 2; ++kind)
    {
        std::vector<double> al(f3.size(), 0.0);
        for (int i = 0; i < NG; ++i)
        {
            std::vector<double> R = trace(kind == 0, th[i]);
            for (size_t q = 0; q < f3.size(); ++q)
                al[q] += w[i] * (1.0 - R[q]);
        }
        for (size_t o = 0; o < OCT.size(); ++o)
        {
            double v = 0.0, ref = 0.0;
            for (int k = 0; k < 3; ++k)
                v += al[3 * o + k] / 3.0, ref += paris(f3[3 * o + k]) / 3.0;
            QD_DIAG("%s,%.1f,%.6f,%.6f,%.1e,%.3f\n", kind ? "X" : "S", OCT[o], v, ref, std::abs(v - ref), TGT[o]);
            mP = std::max(mP, std::abs(v - ref));
            INFO((kind ? "traced" : "sheet") << ", octave " << OCT[o] << " Hz");
            CHECK(std::abs(v - ref) < 1.2e-5);
        }
    }
    QD_DIAG("# max |Paris - closed form| %.2e\n", mP);
}
