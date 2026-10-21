// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Leaves as wave-medium sheets. Under the two-parameter mapping (eps = n z, mu = n / z, so mu = rho / rho0) a thin
// layer of a wave medium has the series impedance j w m'' / Z0: a sheet with m = 0 is a limp leaf. Glued on a meshed
// box of near-air material on a mass-law wall, its cavity series is summed in one event. The leaf cases evaluate wave
// media only, the box cases a mass-law wall. Every reference is the transfer matrix of fluid layers (or the limp mass
// law) written here, not library code.

#include "ray_test_helpers.hpp"

namespace lw
{
    typedef std::complex<double> cd;
    static const double GHZ = C0 / 343.0 * 1.0e-9; // GHz per acoustic Hz, the acoustic mapping
    static double dB(double g) { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; }
    static double deg(cd x) { return std::arg(x) * 180.0 / QPI; }

    // A material by its columns: eps = a r^b - j 17.98 c r^d / fGHz, mu = e r^f - j 17.98 g r^h / fGHz, r = fGHz / fRef
    struct Mtl
    {
        double a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0, g = 0.0, h = 0.0, thk = 0.0, tf = 0.0, m = 0.0, fRef = 1.0;
    };
    // A wave-medium leaf: eps a, mu = mu' (1 - j rr) at every frequency (h = 1, fRef = 1), thickness thk
    static Mtl leaf(double a, double mu, double thk, double rr)
    {
        Mtl M;
        M.a = a, M.e = mu, M.g = mu * rr / 17.98, M.h = 1.0, M.thk = thk;
        return M;
    }
    static ub::Pal pal(const std::vector<Mtl> &v)
    {
        ub::Pal P;
        for (const char *k : {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB", "alpha", "alphaB", "fRef", "tf", "tfB", "resF", "resQ", "resS", "m"})
            P[k] = std::vector<double>(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i)
        {
            P["a"][i] = v[i].a, P["b"][i] = v[i].b, P["c"][i] = v[i].c, P["d"][i] = v[i].d, P["e"][i] = v[i].e, P["f"][i] = v[i].f;
            P["g"][i] = v[i].g, P["h"][i] = v[i].h, P["thk"][i] = v[i].thk, P["tf"][i] = v[i].tf, P["m"][i] = v[i].m, P["fRef"][i] = v[i].fRef;
        }
        return P;
    }
    static void eps_mu(const Mtl &M, double fG, cd &eps, cd &mu)
    {
        const double r = fG / M.fRef;
        eps = cd(M.a * std::pow(r, M.b), -17.98 * M.c * std::pow(r, M.d) / fG);
        mu = cd(M.e * std::pow(r, M.f), -17.98 * M.g * std::pow(r, M.h) / fG);
    }

    // ---- the reference: fluid layers in (p, v_n), exp(+j w t), impedances relative to rho0 c0 ----
    // A layer of the mapping has rho = mu rho0 and K = rho0 c0^2 / eps; its normal wavenumber over k0 is
    // q = sqrt(eps mu - s^2) at the sine s of air, its normal impedance mu / q.
    struct Layer
    {
        cd eps, mu;
        double h;
    };
    static cd q_of(cd eps, cd mu, double s)
    {
        cd q = std::sqrt(eps * mu - s * s);
        return (q.imag() > 0.0) ? -q : q; // the root that decays or carries the wave away
    }
    struct Ref
    {
        cd r, t; // pressure reflection at the front face; t into air behind the last face (air backing only)
    };
    // The stack on a backing: air (be = 0), or a fluid half-space (be, bm) behind a matched pad that absorbs tf of what
    // arrives from air at every angle (backing reflection sqrt(1 - tf) r_half); tf = 0 is the half-space itself.
    static Ref stack(const std::vector<Layer> &L, cd be, cd bm, double tf, double k0, double thd)
    {
        const cd J(0.0, 1.0);
        const double s = std::sin(thd * QPI / 180.0), Za = 1.0 / std::cos(thd * QPI / 180.0);
        cd A = 1.0, B = 0.0, C = 0.0, D = 1.0;
        for (const Layer &l : L)
        {
            const cd q = q_of(l.eps, l.mu, s), z = l.mu / q, ph = k0 * q * l.h, c = std::cos(ph), sn = std::sin(ph);
            const cd A2 = A * c + B * J * sn / z, B2 = A * J * z * sn + B * c, C2 = C * c + D * J * sn / z, D2 = C * J * z * sn + D * c;
            A = A2, B = B2, C = C2, D = D2;
        }
        Ref out;
        if (be == 0.0)
        {
            const cd vt = 2.0 / (A * Za + B + Za * (C * Za + D));
            out.r = (A * Za + B) * vt - 1.0, out.t = Za * vt;
            return out;
        }
        const cd Zh = bm / q_of(be, bm, s), g = std::sqrt(1.0 - tf) * (Zh - Za) / (Zh + Za), Zt = Za * (1.0 + g) / (1.0 - g);
        const cd Zs = (A * Zt + B) / (C * Zt + D);
        out.r = (Zs - Za) / (Zs + Za);
        return out;
    }
    static Layer layer(const Mtl &M, double f_hz, double h)
    {
        cd e, m;
        eps_mu(M, f_hz * GHZ, e, m);
        return {e, m, h};
    }

    // ---- the traced run ----
    // cf::walk without the recursion (a followed series of a lossless box needs thousands of events, and one stack
    // frame per event overflows the default stack): every path of one ray through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update, both passes at every event, collected where the ray is in air outside the
    // structure's x range or has entered the sink medium, into T where it runs towards +x. A ray the state puts in air
    // inside that range is followed on: a sheet between two bodies leaves the wave in air inside the second body, and
    // collecting it there would hide what the library does next. Scalar types (reflect pass type 3), or EM types 1 / 2
    // (reflect pass type 0) on one channel of xprmat, ch 0 (TE) or 3 (TM). ev: traced events with a hit; a path weaker
    // than floor in power is dropped and counted in drop.
    struct Acc
    {
        cd T = 0.0, R = 0.0;
        long ev = 0;
        double drop = 0.0;
    };
    static Acc walk(const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double thd, double eps, int sink, int maxdepth,
                    double floor = 1.0e-24, int ch = 0)
    {
        struct Node
        {
            ub::Ev in;
            bool has_in = false;
            arma::mat o, d;
            cd a = 1.0;
            double L = 0.0, y0 = 0.0;
            int depth = 0;
        };
        const double k = 2.0 * QPI * f / C0, kt = k * std::sin(thd * QPI / 180.0);
        const arma::mat X = arma::join_rows(mesh.col(0), mesh.col(3), mesh.col(6));
        const double xmin = X.min(), xmax = X.max(); // the structure's x range
        Acc A;
        std::vector<Node> todo(1);
        diag_ray_at(thd, todo[0].o, todo[0].d);
        while (!todo.empty())
        {
            Node n = std::move(todo.back());
            todo.pop_back();
            for (int it : {tr, tr < 3 ? 0 : 3})
            {
                ub::Opt op;
                op.eps = eps, op.relaunch = (it == tr);
                ub::Ev E = ub::traced<double>(P, mesh, mi, it, f, n.o, n.d, n.has_in ? &n.in : nullptr, op);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                ++A.ev;
                const cd b = n.a * cpx(E.xpr, 0, ch);
                const double L2 = n.L + (n.has_in ? std::sqrt(ub::sumsq(E.fbs - n.in.fbs)) : 0.0), yy0 = n.has_in ? n.y0 : E.fbs(0, 1);
                const int cur = E.cur(0) & 0x3FFF;
                const bool fwd = (E.destN(0, 0) - E.origN(0, 0)) > 0.0;
                if ((cur == 0 && (fwd ? E.fbs(0, 0) >= xmax - 1.0e-6 : E.fbs(0, 0) <= xmin + 1.0e-6)) || (sink != 0 && cur == sink))
                {
                    const cd ph = std::exp(cd(0.0, -k * L2)) * std::exp(cd(0.0, kt * (E.fbs(0, 1) - yy0)));
                    (fwd ? A.T : A.R) += b * ph;
                    continue;
                }
                if (n.depth + 1 >= maxdepth || std::norm(b) < floor)
                {
                    A.drop += std::norm(b);
                    continue;
                }
                Node m;
                m.in = E, m.has_in = true, m.a = b, m.L = L2, m.y0 = yy0, m.depth = n.depth + 1;
                ub::onward(E, m.o, m.d);
                todo.push_back(std::move(m));
            }
        }
        return A;
    }

    // The materials of the cases
    static Mtl wall(double tf) // two-parameter concrete with the mass column (rho 2281 kg/m3)
    {
        Mtl M;
        M.a = 3.850e-6, M.e = 1901.0, M.m = 20.0, M.fRef = 5.0212e-5, M.tf = tf;
        return M;
    }
    static Mtl air_body() // a meshed box of near-air material: identical to air it would be air
    {
        Mtl M;
        M.e = 1.000001;
        return M;
    }
    static const Mtl BOARD = leaf(4.691e-5, 573.0, 0.0125, 0.2); // 12.5 mm gypsum, 8.6 kg/m2, mounting resistance 0.2
    static const Mtl GLASS = leaf(1.767e-6, 2083.0, 0.004, 0.0); // 4 mm glass, 10 kg/m2

    // The peak of y over log f, parabolic through the largest sample and its neighbours
    static double peak_f(const std::vector<double> &f, const std::vector<double> &y)
    {
        size_t i = 0;
        for (size_t k = 1; k < y.size(); ++k)
            if (y[k] > y[i])
                i = k;
        if (i == 0 || i + 1 == y.size())
            return f[i];
        const double d = y[i - 1] - 2.0 * y[i] + y[i + 1], t = (d != 0.0) ? 0.5 * (y[i - 1] - y[i + 1]) / d : 0.0;
        return std::exp(std::log(f[i]) + t * std::log(f[i + 1] / f[i]));
    }
}

TEST_CASE("Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf", "[acoustic]")
{
    // 4 mm glass with m = 0 (eps 1.767e-6, mu 2083: m'' 10 kg/m2) as a sheet in air, 50 Hz to 1 kHz, 0 to 85 deg,
    // tracers 4 and 5. Its level is the limp mass law t = 1 / (1 + j w m'' cos / (2 Z0)), r = 1 - t at any angle (the
    // cosine enters through the fluid's admittance). Its phase is the fluid layer's with the reference planes on the
    // mesh plane the sheet lies about (r and t times exp(j k0 h cos)); the zero-thickness law lacks the air path over h
    // (4.2 deg at 1 kHz), so the phase is asserted against the layer.
    ub::OneThread one_thread;
    const ub::Pal P = lw::pal({lw::GLASS});
    arma::uvec mi;
    const arma::mat mesh = ub::build({0.0}, {1}, {false}, mi);
    const double h = lw::GLASS.thk;
    QD_DIAG("\n# diag: a 4 mm glass leaf (m = 0) in air against the limp mass law (level) and the fluid layer (phase)\n");
    QD_DIAG("type,f_Hz,theta,T_dB,T_limp_dB,R_dB,R_limp_dB,phT,phT_layer,phR,phR_layer\n");
    for (int tr : {4, 5})
        for (double fa : {50.0, 100.0, 250.0, 500.0, 1000.0})
            for (double thd : {0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0})
            {
                const double k0 = 2.0 * QPI * fa / 343.0, c = std::cos(thd * QPI / 180.0);
                const lw::cd tl = 1.0 / lw::cd(1.0, k0 * lw::GLASS.e * h * c / 2.0), rl = 1.0 - tl;
                lw::Ref F = lw::stack({lw::layer(lw::GLASS, fa, h)}, 0.0, 0.0, 0.0, k0, thd);
                const lw::cd sh = std::exp(lw::cd(0.0, k0 * h * c));
                lw::Acc W = lw::walk(P, mesh, mi, tr, fa * lw::GHZ * 1.0e9, thd, 0.0, 0, 10);
                QD_DIAG("%d,%.0f,%.0f,%.5f,%.5f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f\n", tr, fa, thd, lw::dB(std::norm(W.T)), lw::dB(std::norm(tl)),
                        lw::dB(std::norm(W.R)), lw::dB(std::norm(rl)), lw::deg(W.T), lw::deg(F.t * sh), lw::deg(W.R), lw::deg(F.r * sh));
                INFO("type " << tr << ", " << fa << " Hz, " << thd << " deg");
                CHECK(std::abs(lw::dB(std::norm(W.T)) - lw::dB(std::norm(tl))) < 0.01);
                CHECK(std::abs(lw::dB(std::norm(W.R)) - lw::dB(std::norm(rl))) < 0.01);
                CHECK(std::abs(lw::dB(std::norm(W.T)) - lw::dB(std::norm(F.t))) < 1.0e-4);
                CHECK(std::abs(lw::deg(W.T / (F.t * sh))) < 1.0e-3);
                CHECK(std::abs(lw::deg(W.R / (F.r * sh))) < 1.0e-3);
            }
}

TEST_CASE("Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle", "[acoustic]")
{
    // The board (12.5 mm gypsum, 8.6 kg/m2, its mounting resistance a loss ratio 0.2 on mu) glued on a meshed box of
    // near-air material (25 mm) glued on a mass-law wall: two-parameter concrete with m 20, tf 0 (closed: it reflects
    // its half-space's Fresnel coefficient) or tf 0.02 (a pad on it), as a half-space or as a traced wall of 200 mm. 50
    // Hz to 1 kHz, 0 to 60 deg, tracers 4 and 5, the box's series summed (one event). Followed (every bounce, down to a
    // path power of 1e-12) on six rows of tf 0.02 on the half-space. Reference: leaf | air 25 mm | the half-space
    // behind the pad, phase on the mesh plane (the glued leaf lies in front of it: r times exp(2 j k0 h cos)).
    // Asserted: summed and followed on the reference, the walker here equal to cf::walk on the summed rows, and the
    // angle shift of the resonance, 1 / cos theta.
    ub::OneThread one_thread;
    const double depth = 0.025, h = lw::BOARD.thk;
    const std::vector<double> FA = {50, 63, 80, 100, 112, 125, 140, 160, 200, 250, 315, 400, 500, 1000};
    QD_DIAG("\n# diag: the board on a box of air on a mass-law wall, against leaf | air | wall. mode: sum (eps 0) or fol (eps 1.5)\n");
    QD_DIAG("tf,wall,type,mode,f_Hz,theta,R,R_ref,alpha,phase_dev_deg,events,drop\n");
    double worst[2] = {0.0, 0.0};
    // one row: the walk against the reference; mode 0 summed, 1 followed
    auto row = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, double tf, int fin, int tr, int mode, double fa, double thd)
    {
        const double k0 = 2.0 * QPI * fa / 343.0, c = std::cos(thd * QPI / 180.0), f = fa * lw::GHZ * 1.0e9;
        const lw::Mtl Wm = lw::wall(tf);
        const lw::cd rr = lw::stack({lw::layer(lw::BOARD, fa, h), {1.0, 1.0, depth}}, Wm.a, Wm.e, tf, k0, thd).r * std::exp(lw::cd(0.0, 2.0 * k0 * h * c));
        lw::Acc W = lw::walk(P, mesh, mi, tr, f, thd, mode ? 1.5 : 0.0, fin ? 0 : 2, 4000, mode ? 1.0e-12 : 1.0e-24);
        const double dR = std::abs(std::norm(W.R) - std::norm(rr)), dph = std::abs(lw::deg(W.R / rr));
        worst[mode] = std::max(worst[mode], dR);
        QD_DIAG("%.2f,%s,%d,%s,%.0f,%.0f,%.8f,%.8f,%.5f,%.5f,%ld,%.1e\n", tf, fin ? "200mm" : "half", tr, mode ? "fol" : "sum", fa, thd,
                std::norm(W.R), std::norm(rr), 1.0 - std::norm(W.R) - std::norm(W.T), dph, W.ev, W.drop);
        INFO("tf " << tf << (fin ? ", wall 200 mm" : ", half-space") << ", type " << tr << (mode ? ", followed, " : ", summed, ") << fa << " Hz, " << thd << " deg");
        CHECK(dR < (mode ? 1.0e-5 : 2.0e-4));
        CHECK(dph < (mode ? 2.0e-3 : 0.02));
        CHECK(W.drop < 1.0e-8);
        if (mode == 0)
        {
            CHECK(W.ev <= 7);
            if (!fin)
            {
                // the walker here against cf::walk, which recurses, on rows a few events deep
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                cf::Acc C;
                cf::walk(P, mesh, mi, tr, f, 2.0 * QPI * f / C0 * std::sin(thd * QPI / 180.0), 0.0, 2, nullptr, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 200, C);
                CHECK(std::abs(C.R[0] - W.R) < 1.0e-12);
            }
        }
    };
    for (double tf : {0.0, 0.02})
    {
        const ub::Pal P = lw::pal({lw::BOARD, lw::wall(tf), lw::air_body()});
        for (int fin : {0, 1})
        {
            arma::uvec mi;
            const arma::mat mesh = ub::build({0.0, 0.0, depth, depth, fin ? depth + 0.2 : 0.6}, {1, 3, 3, 2, 2}, {false, false, true, false, true}, mi);
            for (int tr : {4, 5})
                for (double fa : FA)
                    for (double thd : {0.0, 30.0, 45.0, 60.0})
                        row(P, mesh, mi, tf, fin, tr, 0, fa, thd);
            if (tf == 0.02 && fin == 0)
                for (double fa : {50.0, 125.0, 250.0})
                    for (double thd : {0.0, 60.0})
                        row(P, mesh, mi, tf, fin, 4, 1, fa, thd);
        }
    }
    QD_DIAG("# largest |R - R_ref|: summed %.2e, followed %.2e\n", worst[0], worst[1]);

    // The angle shift: the peak of alpha = 1 - R (summed, tf 0, half-space) on a 1/96-octave sweep, against the reference's
    // peak on the same sweep and against 1 / cos theta (the cavity is stiffer by 1 / cos^2 at oblique incidence)
    const ub::Pal P = lw::pal({lw::BOARD, lw::wall(0.0), lw::air_body()});
    arma::uvec mi;
    const arma::mat mesh = ub::build({0.0, 0.0, depth, depth, 0.6}, {1, 3, 3, 2, 2}, {false, false, true, false, true}, mi);
    std::vector<double> fs;
    for (int q = 0; q <= 180; ++q)
        fs.push_back(90.0 * std::pow(2.0, q / 96.0));
    QD_DIAG("\n# diag: the angle shift of the board's resonance on the box (1/96-octave sweep)\n");
    QD_DIAG("theta,f_peak,f_peak_ref,ratio_to_0deg,inv_cos\n");
    double f0 = 0.0;
    for (double thd : {0.0, 30.0, 45.0, 60.0})
    {
        std::vector<double> al, ar;
        for (double fa : fs)
        {
            const double k0 = 2.0 * QPI * fa / 343.0;
            lw::Acc W = lw::walk(P, mesh, mi, 4, fa * lw::GHZ * 1.0e9, thd, 0.0, 2, 40);
            al.push_back(1.0 - std::norm(W.R));
            ar.push_back(1.0 - std::norm(lw::stack({lw::layer(lw::BOARD, fa, h), {1.0, 1.0, depth}}, 3.850e-6, 1901.0, 0.0, k0, thd).r));
        }
        const double fp = lw::peak_f(fs, al), fr = lw::peak_f(fs, ar), ic = 1.0 / std::cos(thd * QPI / 180.0);
        if (thd == 0.0)
            f0 = fp;
        QD_DIAG("%.0f,%.2f,%.2f,%.4f,%.4f\n", thd, fp, fr, fp / f0, ic);
        INFO("angle shift, " << thd << " deg");
        CHECK(std::abs(fp / fr - 1.0) < 1.0e-3);
        CHECK(std::abs(fp / f0 / ic - 1.0) < 0.02); // the lumped law; the matrix reads 2.016 at 60 deg
    }
}

TEST_CASE("Leaf - two leaves on a box of air resonate at the mass-air-mass frequency", "[acoustic]")
{
    // 4|16|4: two 4 mm glass leaves (m = 0) glued on both faces of a meshed box of near-air material, air on both
    // sides, the box summed in one event, tracers 4 and 5. Reference: glass | air 16 mm | glass in air, phases on the
    // mesh planes (r times exp(2 j k0 h cos), t times exp(j k0 cos (24 - 16 mm))). The leaves resonate on the cavity:
    // the transmission loss dips near the mass-air-mass frequency (211 Hz lumped) and the dip moves up as 1 / cos
    // theta. Mass-law sheets in the one-parameter mapping carry no mass reactance and do not resonate.
    ub::OneThread one_thread;
    const double gap = 0.016, h = lw::GLASS.thk;
    const ub::Pal P = lw::pal({lw::GLASS, lw::air_body()});
    arma::uvec mi;
    const arma::mat mesh = ub::build({0.0, 0.0, gap, gap}, {1, 2, 2, 1}, {false, false, true, false}, mi);
    const std::vector<double> FA = {50, 63, 80, 100, 112, 125, 140, 160, 200, 250, 315, 400, 500, 1000};
    QD_DIAG("\n# diag: 4|16|4 on a box of air against glass | air | glass\n");
    QD_DIAG("type,f_Hz,theta,TL_dB,TL_ref_dB,R,R_ref,phT_dev,phR_dev,events\n");
    for (int tr : {4, 5})
        for (double thd : {0.0, 30.0, 45.0, 60.0})
            for (double fa : FA)
            {
                const double k0 = 2.0 * QPI * fa / 343.0, c = std::cos(thd * QPI / 180.0);
                lw::Ref F = lw::stack({lw::layer(lw::GLASS, fa, h), {1.0, 1.0, gap}, lw::layer(lw::GLASS, fa, h)}, 0.0, 0.0, 0.0, k0, thd);
                const lw::cd rr = F.r * std::exp(lw::cd(0.0, 2.0 * k0 * h * c)), tt = F.t * std::exp(lw::cd(0.0, k0 * c * 2.0 * h));
                lw::Acc W = lw::walk(P, mesh, mi, tr, fa * lw::GHZ * 1.0e9, thd, 0.0, 0, 40);
                QD_DIAG("%d,%.0f,%.0f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%ld\n", tr, fa, thd, -lw::dB(std::norm(W.T)), -lw::dB(std::norm(tt)), std::norm(W.R), std::norm(rr),
                        std::abs(lw::deg(W.T / tt)), std::abs(lw::deg(W.R / rr)), W.ev);
                INFO("type " << tr << ", " << fa << " Hz, " << thd << " deg");
                CHECK(std::abs(lw::dB(std::norm(W.T)) - lw::dB(std::norm(tt))) < 0.02);
                CHECK(std::abs(std::norm(W.R) - std::norm(rr)) < 1.0e-3);
                CHECK(std::abs(lw::deg(W.T / tt)) < 0.1);
                CHECK(std::abs(lw::deg(W.R / rr)) < 0.1);
                CHECK(W.ev <= 5);
            }

    // The dip: the peak of the transmitted power on a 1/96-octave sweep from 150 / cos to 300 / cos Hz, against the
    // reference's on the same sweep and against 1 / cos theta of the dip at normal incidence
    QD_DIAG("\n# diag: the dip of 4|16|4 on the box (1/96-octave sweep)\n");
    QD_DIAG("theta,f_dip,f_dip_ref,TL_dip_dB,ratio_to_0deg,inv_cos\n");
    double f0 = 0.0;
    for (double thd : {0.0, 30.0, 45.0, 60.0})
    {
        const double ic = 1.0 / std::cos(thd * QPI / 180.0);
        std::vector<double> fs, tl, tr_;
        for (int q = 0; q <= 96; ++q)
        {
            const double fa = 150.0 * ic * std::pow(2.0, q / 96.0), k0 = 2.0 * QPI * fa / 343.0;
            fs.push_back(fa);
            tl.push_back(std::norm(lw::walk(P, mesh, mi, 4, fa * lw::GHZ * 1.0e9, thd, 0.0, 0, 40).T));
            tr_.push_back(std::norm(lw::stack({lw::layer(lw::GLASS, fa, h), {1.0, 1.0, gap}, lw::layer(lw::GLASS, fa, h)}, 0.0, 0.0, 0.0, k0, thd).t));
        }
        const double fp = lw::peak_f(fs, tl), fr = lw::peak_f(fs, tr_);
        if (thd == 0.0)
            f0 = fp;
        QD_DIAG("%.0f,%.2f,%.2f,%.3f,%.4f,%.4f\n", thd, fp, fr, -lw::dB(*std::max_element(tl.begin(), tl.end())), fp / f0, ic);
        INFO("the dip, " << thd << " deg");
        CHECK(std::abs(fp / fr - 1.0) < 1.0e-3);
        CHECK(std::abs(fp / f0 / ic - 1.0) < 0.03);
    }
}

TEST_CASE("ray_progress - a leaf on a box of air at every frequency of a band", "[acoustic]")
{
    // The board on its box on a mass-law wall (tf 0.02, 200 mm) and 4|16|4 on its box, through ray_progress (scalar,
    // float, one ray until none is left, thin_slab_threshold 0.15, which sums the box), a band of frequencies across
    // the resonance on the base frequency's geometry (the first of the band) against each frequency alone and against
    // the fluid layers, tracers 4 and 5. Paths are summed at their polyline's point where they end in air in front of
    // the first face (R) or behind the last (T). A meshed air gap between the faces would be out of reach of the 255
    // interactions ray_progress allows (3.7e-3 in R on the board, 9.1 dB on 4|16|4); the box of near-air material is
    // summed in one event.
    ub::OneThread one_thread;
    typedef std::unordered_map<std::string, std::vector<float>> PalF;
    const std::complex<double> J(0.0, 1.0);
    auto run = [&](const ub::Pal &Pd, const std::vector<float> &x, const std::vector<int> &mat, const std::vector<int> &flip, int tr, double thd,
                   const std::vector<double> &fa, std::vector<lw::cd> &T, std::vector<lw::cd> &R)
    {
        PalF pal;
        for (const auto &kv : Pd)
            pal[kv.first] = std::vector<float>(kv.second.begin(), kv.second.end());
        const size_t nf = fa.size();
        arma::fmat mesh;
        arma::uvec mi;
        rp::planes(x, mat, std::vector<bool>(flip.begin(), flip.end()), mesh, mi);
        const float x_front = *std::min_element(x.begin(), x.end()), x_back = *std::max_element(x.begin(), x.end());
        const double th = thd * QPI / 180.0, st = std::sin(th);
        arma::fvec fq(nf);
        for (size_t q = 0; q < nf; ++q)
            fq(q) = (float)(fa[q] * lw::GHZ * 1.0e9);
        T.assign(nf, 0.0), R.assign(nf, 0.0);
        rp::Opt o;
        o.calls = 60, o.max_int = 60, o.max_refl = 255, o.max_trans = 255, o.min_gain = -250.0f, o.refr = tr == 5;
        rp::walk(mesh, mi, pal, fq, thd, o, [&](const rp::Ray &r)
                 {
            const float *first = r.first(), *last = r.last();
            const bool fwd = r.fwd();
            const bool isT = fwd && last[0] > x_back - 1.0e-3f, isR = !fwd && last[0] < x_front + 1.0e-3f;
            if (!isT && !isR)
                return false;
            const double Lin = r.poly();
            for (size_t q = 0; q < nf; ++q)
            {
                const double k = 2.0 * QPI * fq(q) / C0;
                const float *xc = r.xc(q);
                const lw::cd F = lw::cd((double)xc[0], (double)xc[1]) * std::exp(-J * k * Lin) * std::exp(J * k * st * ((double)last[1] - (double)first[1]));
                (isT ? T[q] : R[q]) += F;
            }
            return false; });
    };

    QD_DIAG("\n# diag: ray_progress, a band against each frequency alone and against the fluid layers\n");
    QD_DIAG("config,type,theta,f_Hz,R,R_single,R_ref,TL_dB,TL_ref_dB,phR_dev,phT_dev\n");
    // the board on its box on the wall (tf 0.02, 200 mm), 80 to 315 Hz
    {
        const double depth = 0.025, h = lw::BOARD.thk, tf = 0.02;
        const ub::Pal P = lw::pal({lw::BOARD, lw::wall(tf), lw::air_body()});
        const std::vector<double> band = {80, 90, 100, 112, 125, 140, 160, 180, 200, 225, 250, 280, 315};
        for (int tr : {4, 5})
            for (double thd : {0.0, 30.0, 45.0, 60.0})
            {
                std::vector<lw::cd> T, R, T1, R1;
                run(P, {0.0f, 0.0f, 0.025f, 0.025f, 0.225f}, {1, 3, 3, 2, 2}, {0, 0, 1, 0, 1}, tr, thd, band, T, R);
                for (size_t q = 0; q < band.size(); ++q)
                {
                    run(P, {0.0f, 0.0f, 0.025f, 0.025f, 0.225f}, {1, 3, 3, 2, 2}, {0, 0, 1, 0, 1}, tr, thd, {band[q]}, T1, R1);
                    const double k0 = 2.0 * QPI * band[q] / 343.0, c = std::cos(thd * QPI / 180.0);
                    const lw::cd rr = lw::stack({lw::layer(lw::BOARD, band[q], h), {1.0, 1.0, depth}}, 3.850e-6, 1901.0, tf, k0, thd).r * std::exp(lw::cd(0.0, 2.0 * k0 * h * c));
                    QD_DIAG("board,%d,%.0f,%.0f,%.8f,%.8f,%.8f,,,%.5f,\n", tr, thd, band[q], std::norm(R[q]), std::norm(R1[0]), std::norm(rr), std::abs(lw::deg(R[q] / rr)));
                    INFO("board, type " << tr << ", " << thd << " deg, " << band[q] << " Hz");
                    CHECK(std::abs(R[q] - R1[0]) < 1.0e-6);
                    CHECK(std::abs(std::norm(R[q]) - std::norm(rr)) < 2.0e-4);
                    CHECK(std::abs(lw::deg(R[q] / rr)) < 0.02);
                }
            }
    }
    // 4|16|4 on its box, 125 to 400 Hz
    {
        const double gap = 0.016, h = lw::GLASS.thk;
        const ub::Pal P = lw::pal({lw::GLASS, lw::air_body()});
        const std::vector<double> band = {125, 140, 160, 180, 200, 225, 250, 280, 315, 355, 400};
        for (int tr : {4, 5})
            for (double thd : {0.0, 30.0, 45.0, 60.0})
            {
                std::vector<lw::cd> T, R, T1, R1;
                run(P, {0.0f, 0.0f, 0.016f, 0.016f}, {1, 2, 2, 1}, {0, 0, 1, 0}, tr, thd, band, T, R);
                for (size_t q = 0; q < band.size(); ++q)
                {
                    run(P, {0.0f, 0.0f, 0.016f, 0.016f}, {1, 2, 2, 1}, {0, 0, 1, 0}, tr, thd, {band[q]}, T1, R1);
                    const double k0 = 2.0 * QPI * band[q] / 343.0, c = std::cos(thd * QPI / 180.0);
                    lw::Ref F = lw::stack({lw::layer(lw::GLASS, band[q], h), {1.0, 1.0, gap}, lw::layer(lw::GLASS, band[q], h)}, 0.0, 0.0, 0.0, k0, thd);
                    const lw::cd rr = F.r * std::exp(lw::cd(0.0, 2.0 * k0 * h * c)), tt = F.t * std::exp(lw::cd(0.0, k0 * c * 2.0 * h));
                    QD_DIAG("4|16|4,%d,%.0f,%.0f,%.8f,%.8f,%.8f,%.4f,%.4f,%.5f,%.5f\n", tr, thd, band[q], std::norm(R[q]), std::norm(R1[0]), std::norm(rr),
                            -lw::dB(std::norm(T[q])), -lw::dB(std::norm(tt)), std::abs(lw::deg(R[q] / rr)), std::abs(lw::deg(T[q] / tt)));
                    INFO("4|16|4, type " << tr << ", " << thd << " deg, " << band[q] << " Hz");
                    CHECK(std::abs(R[q] - R1[0]) < 1.0e-6);
                    CHECK(std::abs(T[q] - T1[0]) < 1.0e-6);
                    CHECK(std::abs(lw::dB(std::norm(T[q])) - lw::dB(std::norm(tt))) < 0.02);
                    CHECK(std::abs(std::norm(R[q]) - std::norm(rr)) < 1.0e-3);
                    CHECK(std::abs(lw::deg(R[q] / rr)) < 0.1);
                    CHECK(std::abs(lw::deg(T[q] / tt)) < 0.1);
                }
            }
    }
}

TEST_CASE("Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill and wall", "[acoustic]")
{
    // Shearer's filled membrane absorber [Shearer_2016_AbsorptionBelow100Hz, Sec. 3.6]: 6 mm MDF (4.65 kg/m2, a leaf
    // with m = 0, loss ratio 0.2 on mu) glued on a box of near-air material (20 mm) glued on a porous body (75 mm rock
    // wool: Miki at 32.8 kPa s/m2 in the two-parameter mapping, power-law columns fitted on its zc and kc over 40 Hz -
    // 1 kHz) glued on the mass-law wall with tf 0. Each body's series summed, the exchange between the two bodies
    // followed as paths (up to a few hundred events). Reference: MDF | air | the same columns | the concrete half-
    // space. Not asserted: the followed mode (the series of a lossy layer between strong reflectors diverges, round
    // trip 1.145), a fill directly behind the leaf with air behind it (no finite path sum), tf behind a fill (a pad
    // matched to air, not to the fill), and the comparison with Shearer's measured peak (80 Hz third).
    ub::OneThread one_thread;
    const lw::Mtl MDF = lw::leaf(4.5e-5, 775.0 / 1.2, 0.006, 0.2);
    lw::Mtl W = lw::wall(0.0), A = lw::air_body(), F;
    F.a = 1.31823, F.b = -0.0125593, F.c = 0.00481048, F.d = 1.80573, F.e = 0.823363, F.f = 0.0954384, F.g = 0.221513, F.h = -0.0150261;
    F.fRef = 1000.0 * lw::GHZ;
    const ub::Pal P = lw::pal({MDF, W, A, F});
    arma::uvec mi;
    const arma::mat mesh = ub::build({0.0, 0.0, 0.020, 0.020, 0.095, 0.095, 0.6}, {1, 3, 3, 4, 4, 2, 2}, {false, false, true, false, true, false, true}, mi);
    QD_DIAG("\n# diag: MDF | air 20 mm | rock wool 75 mm | wall (tf 0), summed, against the fluid layers with the same columns\n");
    QD_DIAG("type,f_Hz,theta,alpha,alpha_ref,dR,events,drop\n");
    for (int tr : {4, 5})
        for (double fa : {40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 1000})
            for (double thd : {0.0, 30.0, 45.0, 60.0})
            {
                const double k0 = 2.0 * QPI * fa / 343.0;
                const lw::cd rr = lw::stack({lw::layer(MDF, fa, MDF.thk), {1.0, 1.0, 0.020}, lw::layer(F, fa, 0.075)}, W.a, W.e, 0.0, k0, thd).r;
                lw::Acc S = lw::walk(P, mesh, mi, tr, fa * lw::GHZ * 1.0e9, thd, 0.0, 2, 4000);
                QD_DIAG("%d,%.0f,%.0f,%.5f,%.5f,%.2e,%ld,%.1e\n", tr, fa, thd, 1.0 - std::norm(S.R), 1.0 - std::norm(rr), std::abs(std::norm(S.R) - std::norm(rr)), S.ev, S.drop);
                INFO("type " << tr << ", " << fa << " Hz, " << thd << " deg");
                CHECK(std::abs(std::norm(S.R) - std::norm(rr)) < 5.0e-4);
                CHECK(S.drop < 1.0e-12);
            }
}

TEST_CASE("Sheet between two bodies - nothing is created and the wave left in air inside the far body is killed", "[acoustic]")
{
    // A porous sheet co-located with the exit face of a box of near-air material and the entry face of a wall: box 0 to
    // 10 mm | sheet 15 mm | wall from 10 mm, faces in the plane x = 10 mm in the order box exit, sheet, wall entry. The
    // intersector reports two of the three tied faces, and its kernel's arithmetic decides which two. Box exit + sheet
    // is a glued exit with air assumed beyond it, so the wave is left in air inside the wall; the wall's exit face, met
    // from there, kills it instead of taking it as an exit into air.
    // Asserted, scalar types 4 and 5 on both walls and EM type 1 per channel (TE, TM) on the m = 0 wall, the faces as
    // dielectrics: R + T <= 1, summed and followed, on the pair the kernel picks; at the triple face, every pair handed
    // in, the hit code, the medium after it, and the kill at the wall's exit face (both passes) where the wave was left
    // in air. Printed, not asserted: R against air 10 mm | porous 15 mm | the concrete half-space; the porous layer as
    // a traced body glued on the wall (the fill case above) is the form that matches it.
    ub::OneThread one_thread;
    const double x1 = 0.010;
    lw::Mtl F; // the rock wool of the fill case (Miki 32.8 kPa s/m2), as a 15 mm sheet
    F.a = 1.31823, F.b = -0.0125593, F.c = 0.00481048, F.d = 1.80573, F.e = 0.823363, F.f = 0.0954384, F.g = 0.221513, F.h = -0.0150261;
    F.fRef = 1000.0 * lw::GHZ, F.thk = 0.015;
    QD_DIAG("\n# diag: box 10 mm | porous sheet 15 mm | wall, the sheet co-located with box exit and wall entry, against the fluid layers\n");
    QD_DIAG("wall,type,mode,f_Hz,theta,R,T,R+T,R_ref,events,drop\n");
    for (int mass : {0, 1})
    {
        lw::Mtl W = lw::wall(0.0); // two-parameter concrete, tf 0; m 20 (mass-law) or m 0 (a wave medium)
        if (!mass)
            W.m = 0.0, W.fRef = 1.0;
        const ub::Pal P = lw::pal({W, lw::air_body(), F});
        arma::uvec mi;
        const arma::mat mesh = ub::build({0.0, x1, x1, x1, 0.6}, {2, 2, 3, 1, 1}, {false, true, false, false, true}, mi);
        struct Run
        {
            int tr, ch;
            const char *name;
        };
        std::vector<Run> runs = {{4, 0, "4"}, {5, 0, "5"}};
        if (!mass) // EM on the dielectric configuration: type 1, its reflect pass type 0, TE and TM
            runs.push_back({1, 0, "1 TE"}), runs.push_back({1, 3, "1 TM"});
        for (const Run &r : runs)
            for (int mode : {0, 1})
                for (double fa : {125.0, 1000.0})
                    for (double thd : {0.0, 60.0})
                    {
                        const double k0 = 2.0 * QPI * fa / 343.0;
                        const lw::cd rr = lw::stack({{1.0, 1.0, x1}, lw::layer(F, fa, F.thk)}, W.a, W.e, 0.0, k0, thd).r;
                        lw::Acc S = lw::walk(P, mesh, mi, r.tr, fa * lw::GHZ * 1.0e9, thd, mode ? 1.5 : 0.0, 1, 4000, mode ? 1.0e-12 : 1.0e-24, r.ch);
                        const double rt = std::norm(S.R) + std::norm(S.T);
                        QD_DIAG("%s,%s,%s,%.0f,%.0f,%.6f,%.6f,%.6f,%.6f,%ld,%.1e\n", mass ? "m20" : "m0", r.name, mode ? "fol" : "sum", fa, thd, std::norm(S.R),
                                std::norm(S.T), rt, std::norm(rr), S.ev, S.drop);
                        INFO((mass ? "mass-law wall" : "m = 0 wall") << ", type " << r.name << (mode ? ", followed, " : ", summed, ") << fa << " Hz, " << thd << " deg");
                        CHECK(rt <= 1.0 + 1.0e-6);
                    }
        // The triple face: event 1 enters the box, event 2 is the triple face (transmit), event 3 the wall's exit face.
        // Which two of the three tied faces the intersector reports is decided by its kernel's arithmetic (GENERIC: box
        // exit + sheet, AVX2: box exit + wall entry), so every ordered pair is handed in at event 2. Box exit + sheet is
        // the glued exit (69), the wave left in air and killed at the wall's exit face (both passes); box exit + wall
        // entry is body | body (5, or 7 with the wall entry first), sheet + wall entry the glued entry (71), both in the
        // wall and not killed, with the total-reflection bit beyond the wall's critical angle (60 deg). The pair the
        // kernel picks is one of them, with that pair's outcome.
        for (int tr : {4, 5, 1})
        {
            if (tr == 1 && mass)
                continue;
            for (double thd : {0.0, 60.0})
            {
                const double f = 125.0 * lw::GHZ * 1.0e9;
                arma::mat o, d, o2, d2;
                diag_ray_at(thd, o, d);
                ub::Opt op;
                op.eps = 0.0, op.relaunch = true;
                const ub::Ev E1 = ub::traced<double>(P, mesh, mi, tr, f, o, d, nullptr, op);
                ub::onward(E1, o2, d2);
                unsigned t[3]; // the triangle event 2 crosses on the box exit, the sheet and the wall entry (faces 1-3)
                for (arma::uword k = 1; k <= 3; ++k)
                {
                    const arma::mat fm = mesh.rows(2 * k, 2 * k + 1);
                    arma::u32_vec h;
                    quadriga_lib::ray_triangle_intersect<double>(&o2, &d2, &fm, nullptr, nullptr, nullptr, &h);
                    t[k - 1] = (unsigned)(2 * k) + h(0);
                }
                struct Out
                {
                    ub::Ev E2;
                    uint8_t rt[2] = {0, 0};
                    unsigned nh[2] = {0, 0};
                };
                const auto event = [&](unsigned fi, unsigned si) // fi = 0: the kernel's pair
                {
                    Out R;
                    ub::Opt q = op;
                    q.fi_in = fi, q.si_in = si;
                    R.E2 = ub::traced<double>(P, mesh, mi, tr, f, o2, d2, &E1, q);
                    arma::mat o3, d3;
                    ub::onward(R.E2, o3, d3);
                    int n = 0;
                    for (int it : {tr, tr < 3 ? 0 : 3})
                    {
                        const ub::Ev E3 = ub::traced<double>(P, mesh, mi, it, f, o3, d3, &R.E2, op);
                        R.rt[n] = E3.rt, R.nh[n++] = E3.nh;
                    }
                    return R;
                };
                INFO((mass ? "mass-law wall" : "m = 0 wall") << ", type " << tr << ", " << thd << " deg");
                const int code[3][3] = {{0, 69, 5}, {69, 0, 71}, {7, 71, 0}}, tir = thd > 0.0 ? 32 : 0;
                Out D[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                    {
                        if (a == b)
                            continue;
                        D[a][b] = event(t[a], t[b]);
                        const Out &R = D[a][b];
                        const bool wall = a == 2 || b == 2;
                        INFO("faces " << a + 1 << ", " << b + 1 << ": hit code " << (int)R.E2.ot << ", resolved " << (int)R.rt[0] << ", " << (int)R.rt[1]);
                        CHECK((int)R.E2.ot == code[a][b] + (wall ? tir : 0));
                        CHECK((R.E2.cur(0) & 0x3FFF) == (wall ? 1 : 0));
                        for (int n : {0, 1})
                        {
                            CHECK(R.nh[n] > 0);
                            CHECK((R.rt[n] != 0) == wall);
                        }
                    }
                const Out K = event(0, 0);
                int ka = -1, kb = -1;
                for (int a = 0; a < 3; ++a)
                    ka = K.E2.fi == t[a] ? a : ka, kb = K.E2.si == t[a] ? a : kb;
                INFO("the kernel's pair: triangles " << K.E2.fi << ", " << K.E2.si);
                REQUIRE((ka >= 0 && kb >= 0 && ka != kb));
                CHECK(K.E2.ot == D[ka][kb].E2.ot);
                CHECK(K.E2.cur(0) == D[ka][kb].E2.cur(0));
                CHECK(K.rt[0] == D[ka][kb].rt[0]);
                CHECK(K.rt[1] == D[ka][kb].rt[1]);
            }
        }
    }
}
