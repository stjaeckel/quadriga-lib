// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_RAY=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[wave]"

// ray_state_update, wave: branch cases, walkers, the one solver, the budget and the resonant medium.

#include "ray_test_helpers.hpp"

TEST_CASE("Unentered branches - a corner with a sheet is the sheet alone once the state is kept", "[wave]")
{
    ub::OneThread one_thread;
    // A sheet that meets another face at an angle, within the co-location distance along the ray. ray_mesh_interact
    // drops the second face and clears the co-location and corner bits; the redo re-derives an isolated sheet from the
    // material indices. So with the state kept the event is the sheet without its neighbour, bit for bit, whatever the
    // neighbour is. What ray_mesh_interact wrote itself is printed: the bare face of a half-space of the sheet's
    // material, not the composite. A sheet as the second face is dropped altogether.
    const std::vector<LumpMtl> lp = {
        {5.24, 0.63, 0.006}, // 1 sheet 6 mm
        {1.50, 0.40, 0.0},   // 2 solid
        {2.50, 0.80, 0.010}, // 3 a second sheet 10 mm
    };
    ub::Pal pal = lump_palette(lp);
    QD_DIAG("\n# diag: corner with a sheet, scalar, 1 GHz, second face 0.3 mm further along the ray and tilted by 40 deg.\n");
    QD_DIAG("# own: what ray_mesh_interact wrote, redo: after ray_state_update; d_own, d_redo: complex distance to the same first face alone\n");
    QD_DIAG("config,theta,type,nh,ot,rt,own_dB,redo_dB,d_own,d_redo,redo_bits,words,origN_x\n");
    for (double thd : {20.0, 50.0})
    {
        double th = thd * QPI / 180.0, hy = 0.37 + std::tan(th);
        double px = 3.0e-4 * std::cos(th), py = hy + 3.0e-4 * std::sin(th);
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        struct Cfg
        {
            const char *name;
            arma::mat mesh;
            std::vector<int> mats;
            int twin;
        };
        std::vector<Cfg> cfg;
        cfg.push_back({"sheet alone", ub::face(0.0, false), {1}, -1});
        cfg.push_back({"sheet | sheet 3", arma::join_cols(ub::face(0.0, false), ub::tilted(px, py, 40.0, false)), {1, 3}, 0});
        cfg.push_back({"sheet | solid front-hit", arma::join_cols(ub::face(0.0, false), ub::tilted(px, py, 40.0, false)), {1, 2}, 0});
        cfg.push_back({"sheet | solid back-hit", arma::join_cols(ub::face(0.0, false), ub::tilted(px, py, 40.0, true)), {1, 2}, 0});
        cfg.push_back({"sheet back-hit alone", ub::face(0.0, true), {1}, -1});
        cfg.push_back({"sheet back-hit | solid", arma::join_cols(ub::face(0.0, true), ub::tilted(px, py, 40.0, false)), {1, 2}, 4});
        cfg.push_back({"solid alone", ub::face(0.0, false), {2}, -1});
        cfg.push_back({"solid | sheet", arma::join_cols(ub::face(0.0, false), ub::tilted(px, py, 40.0, false)), {2, 1}, 6});
        std::vector<ub::Ev> keep[2];
        for (size_t q = 0; q < cfg.size(); ++q)
            for (int it : {4, 3})
            {
                arma::uvec mi(cfg[q].mesh.n_rows);
                for (arma::uword r = 0; r < mi.n_elem; ++r)
                    mi(r) = (arma::uword)cfg[q].mats[r / 2];
                ub::Opt op;
                op.eps = 0.0;
                ub::Ev E = ub::traced<double>(pal, cfg[q].mesh, mi, it, FRQ, o, d, nullptr, op);
                keep[it == 3].push_back(E);
                const bool has_twin = cfg[q].twin >= 0;
                const ub::Ev &W = keep[it == 3][has_twin ? cfg[q].twin : q];
                std::complex<double> a = cpx(E.xpr0, 0, 0), b = cpx(E.xpr, 0, 0);
                const bool bits = ub::same_bits(E.xpr, W.xpr), sheet_first = cfg[q].mats[0] != 2;
                QD_DIAG("%s,%.0f,%d,%u,%d,%d,%.4f,%.4f,%.3e,%.3e,%d,%04x.%04x.%04x,%.3e\n", cfg[q].name, thd, it, E.nh, (int)E.ot, (int)E.rt,
                        ub::dB(a), ub::dB(b), std::abs(a - cpx(W.xpr0, 0, 0)), std::abs(b - cpx(W.xpr, 0, 0)), (int)bits,
                        ub::w(E.prev(0)), ub::w(E.cur(0)), ub::w(E.buf(0)), E.origN(0, 0));
                INFO("corner " << cfg[q].name << " theta " << thd << " type " << it);
                REQUIRE(E.nh >= 1);
                CHECK(bits);                                      // the redo is the first face alone
                CHECK(std::abs(a - cpx(W.xpr0, 0, 0)) < 1.0e-12); // and so is what ray_mesh_interact wrote
                CHECK(E.rt == W.rt);
                CHECK(E.prev(0) == W.prev(0));
                CHECK(E.cur(0) == W.cur(0));
                CHECK(E.buf(0) == W.buf(0));
                if (has_twin)
                {
                    CHECK(E.nh >= 2);
                    CHECK((E.ot & 0x1C) == 0);                  // co-location, same-direction and corner bits are cleared
                    CHECK(((E.ot & 0x40) != 0) == sheet_first); // the sheet bit says whether the face that is kept is the sheet
                    if (it != 3)
                        CHECK(E.origN(0, 0) > 1.0e-4); // relaunched behind the face that was dropped
                    if (!sheet_first)
                        CHECK(ub::same_bits(E.xpr0, W.xpr0)); // a sheet as second face leaves no trace at all
                }
                if (sheet_first)
                {
                    CHECK(E.rt == ((it == 3) ? 197 : 69));
                    CHECK((E.prev(0) == 0 && E.cur(0) == 0 && E.buf(0) == 0));
                }
            }
    }
}

TEST_CASE("Unentered branches - the displacement budget caps what a sheet takes back out of its host", "[wave]")
{
    ub::OneThread one_thread;
    // A sheet inside a host takes its thickness of host back out of the transit: half of it on
    // either side of the mesh plane, and on each side no more than the host that lies there,
    // because taking a medium out is a division. The transmit port, the reflect port and the
    // mirror of the host's series place the layer alike, so a capped sheet is still one wall. A
    // sheet thicker than its host is not a valid geometry, but nothing stops a mesh from holding
    // one, so the rows run past it: thk / L from 0.1 to 3 in a 10 mm host.
    //
    // The composite is isolated as the ratio of the event to the same event with a 1 mm sheet:
    // the host leg and the state are the same in both. The reference is diag_cm_stack with what
    // the two sides allow, formed here from the points of the event as normal thicknesses: the
    // transmit port over both parts, the reflect port over twice the part in front, the mirror on
    // the leg behind the sheet over twice what that leg allows.
    const double L = 0.010, f = FRQ, k0 = 2.0 * QPI * f / C0;
    const std::complex<double> J(0.0, 1.0);
    const std::vector<double> thk = {0.001, 0.005, 0.0079, 0.0081, 0.009, 0.010, 0.015, 0.030};
    std::vector<LumpMtl> lp = {{1.50, 0.40, 0.0}};
    for (double t : thk)
        lp.push_back({5.24, 0.63, t});
    for (double t : thk)
        lp.push_back({1.50, 0.00, t});
    ub::Pal pal = lump_palette(lp);
    const std::complex<double> eH = lump_eps(lp[0], f), nH = std::sqrt(eH);

    QD_DIAG("\n# diag: displacement budget, a sheet inside 10 mm of eps 1.50-0.40j, scalar, both tracers. T, R: the composite over the\n");
    QD_DIAG("# 1 mm sheet's against diag_cm_stack capped per side, complex. budget_over_L: what the two sides allow over the host's\n");
    QD_DIAG("# thickness. uncap_dB, uncap_deg: the capped transmit port over the uncapped one. chain_dB: entry, sheet and exit, series\n");
    QD_DIAG("# declined; summed_dB: with the exit's series summed. S: the host's series on the leg behind the sheet against the layer\n");
    QD_DIAG("# mirror capped by its leg; mirror: that mirror's magnitude\n");
    QD_DIAG("sheet,tracer,theta,thk_over_L,budget_over_L,cap,T,R,uncap_dB,uncap_deg,chain_dB,summed_dB,S,mirror\n");
    for (int fam = 0; fam < 2; ++fam)
        for (int tr : {4, 5})
            for (double thd : {20.0, 45.0, 70.0, 85.0})
            {
                const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                const std::complex<double> ct = std::sqrt(1.0 - s2 / eH);
                std::complex<double> T1, R1, Tr1, Rr1;
                for (size_t q = 0; q < thk.size(); ++q)
                {
                    const int iS = 2 + fam * (int)thk.size() + (int)q;
                    const std::complex<double> eS = lump_eps(lp[iS - 1], f);
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, 0.5 * L, L}, {1, iS, 1}, {false, false, true}, mi);
                    arma::mat o, d;
                    diag_ray_at(thd, o, d);
                    ub::Opt dec, sum;
                    dec.relaunch = sum.relaunch = true, sum.eps = 0.0;
                    ub::Ev E0 = ub::traced<double>(pal, mesh, mi, tr, f, o, d, nullptr, dec);
                    ub::onward(E0, o, d);
                    ub::Ev Et = ub::traced<double>(pal, mesh, mi, tr, f, o, d, &E0, dec), Er = ub::traced<double>(pal, mesh, mi, 3, f, o, d, &E0);
                    arma::mat o2, d2;
                    ub::onward(Et, o2, d2);
                    ub::Ev X0 = ub::traced<double>(pal, mesh, mi, tr, f, o2, d2, &Et, dec), X1 = ub::traced<double>(pal, mesh, mi, tr, f, o2, d2, &Et, sum);
                    REQUIRE((E0.nh && Et.nh == 2 && Er.nh == 2 && X0.nh));

                    // What the event sees, as normal thicknesses: the leg in front, and the gap ahead
                    // to the second face of the hit. Each half of the sheet takes no more than its side
                    const double hf = 0.5 * L - E0.fbs(0, 0), hb = std::abs(Et.sbs(0, 0) - Et.fbs(0, 0));
                    const double un = std::min(0.5 * thk[q], hf), ubk = std::min(0.5 * thk[q], hb), budget = un + ubk;
                    DiagLayer lay;
                    lay.e = eS, lay.d = thk[q];
                    DiagSheet W = diag_cm_stack(eH, 1.0, {lay}, eH, 1.0, ct, f, false, true, budget);
                    DiagSheet WR = diag_cm_stack(eH, 1.0, {lay}, eH, 1.0, ct, f, false, true, 2.0 * un);
                    DiagSheet U = diag_cm_stack(eH, 1.0, {lay}, eH, 1.0, ct, f, false, true);
                    std::complex<double> T = cpx(Et.xpr, 0, 0), R = cpx(Er.xpr, 0, 0);
                    if (q == 0)
                        T1 = T, R1 = R, Tr1 = W.T_te, Rr1 = WR.R_te;
                    const double dT = std::abs(T / T1 - W.T_te / Tr1), dR = std::abs(R / R1 - WR.R_te / Rr1);
                    const bool cap = budget < thk[q];
                    const std::complex<double> ch = cpx(E0.xpr, 0, 0) * T * cpx(X0.xpr, 0, 0), chs = cpx(E0.xpr, 0, 0) * T * cpx(X1.xpr, 0, 0);

                    // The series on the leg behind the sheet: the sheet a layer backed by the host, centred,
                    // the half on this side capped by the leg
                    const double dleg = L - Et.fbs(0, 0);
                    DiagSheet Mr = diag_cm_stack(eH, 1.0, {lay}, eH, 1.0, ct, f, true, true, 2.0 * std::min(0.5 * thk[q], dleg));
                    const std::complex<double> rb = Mr.R_te;
                    const double mir = std::abs(rb);
                    const std::complex<double> ra = (nH * ct - std::cos(th)) / (nH * ct + std::cos(th));
                    const std::complex<double> Sref = 1.0 / (1.0 - ra * rb * std::exp(-2.0 * J * k0 * nH * ct * dleg));
                    const double dS = std::abs(cpx(X1.xpr, 0, 0) / cpx(X0.xpr, 0, 0) - Sref);

                    QD_DIAG("%s,%d,%.0f,%.2f,%.4f,%d,%.3e,%.3e,%.6f,%.4f,%.4f,%.4f,%.3e,%.6f\n", fam ? "1.50" : "5.24-0.63j", tr, thd, thk[q] / L, budget / L,
                            (int)cap, dT, dR, ub::dB(W.T_te / U.T_te), ub::deg(W.T_te / U.T_te), ub::dB(ch), ub::dB(chs), dS, mir);
                    INFO("budget sheet " << fam << " tracer " << tr << " theta " << thd << " thk/L " << thk[q] / L);
                    CHECK(dT < 1.0e-9);
                    CHECK(dR < 1.0e-9);
                    CHECK(dS < 1.0e-9);
                    CHECK(std::isfinite(std::norm(T)));
                    CHECK(std::isfinite(std::norm(R)));
                    CHECK((Et.rt == 77 && Er.rt == 205));
                    // Graceful where the sheet does not fit: the wall passes less than it received
                    CHECK(std::norm(ch) < 1.0);
                    CHECK(std::norm(chs) < 1.0);
                    CHECK(mir < 1.0);
                }
            }

    // An ordered stack inside a host. The second face of the hit is the other sheet, so nothing
    // says what lies behind, and the leg in front is taken for both sides: a stack whose half
    // thickness fits that leg is not capped.
    QD_DIAG("\n# diag: displacement budget, an ordered stack eps 5.24-0.63j | eps 2.50-0.80j inside eps 1.50-0.40j, scalar. front_mm:\n");
    QD_DIAG("# host in front of the stack's face; fits: the stack's half thickness is less than that. T, R as above, against\n");
    QD_DIAG("# diag_cm_stack capped per side over the 0.5 + 0.5 mm stack's. budget_mm: twice what the leg in front allows of half the\n");
    QD_DIAG("# stack. uncap_dB, uncap_deg: the capped transmit port over the uncapped one, zero for a stack that fits\n");
    QD_DIAG("host_mm,front_mm,stack_mm,tracer,theta,budget_mm,fits,cap,T,R,uncap_dB,uncap_deg,chain_dB\n");
    struct Sk
    {
        double Lh, xs, ta, tb;
    };
    const std::vector<Sk> sk = {{0.010, 0.005, 0.0005, 0.0005}, {0.010, 0.005, 0.002, 0.001}, {0.010, 0.005, 0.003, 0.002}, {0.010, 0.005, 0.006, 0.003}, {0.100, 0.010, 0.0005, 0.0005}, {0.100, 0.010, 0.004, 0.003}, {0.100, 0.010, 0.006, 0.006}, {0.100, 0.050, 0.0005, 0.0005}, {0.100, 0.050, 0.006, 0.006}, {0.100, 0.090, 0.0005, 0.0005}, {0.100, 0.090, 0.006, 0.006}, {0.010, 0.002, 0.0005, 0.0005}, {0.010, 0.002, 0.006, 0.006}};
    std::vector<LumpMtl> lq = {{1.50, 0.40, 0.0}};
    for (const Sk &K : sk)
        lq.push_back({5.24, 0.63, K.ta}), lq.push_back({2.50, 0.80, K.tb});
    ub::Pal pal_q = lump_palette(lq);
    for (int tr : {4, 5})
        for (double thd : {20.0, 60.0})
        {
            const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            const std::complex<double> ct = std::sqrt(1.0 - s2 / eH);
            std::complex<double> T1, R1, Tr1, Rr1;
            for (size_t q = 0; q < sk.size(); ++q)
            {
                const Sk &K = sk[q];
                const int iA = 2 + 2 * (int)q, iB = iA + 1;
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0, K.xs, K.xs + 1.0e-5, K.Lh}, {1, iA, iB, 1}, {false, false, true, true}, mi);
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                ub::Opt dec;
                dec.relaunch = true;
                ub::Ev E0 = ub::traced<double>(pal_q, mesh, mi, tr, f, o, d, nullptr, dec);
                ub::onward(E0, o, d);
                ub::Ev Et = ub::traced<double>(pal_q, mesh, mi, tr, f, o, d, &E0, dec), Er = ub::traced<double>(pal_q, mesh, mi, 3, f, o, d, &E0);
                arma::mat o2, d2;
                ub::onward(Et, o2, d2);
                ub::Ev X0 = ub::traced<double>(pal_q, mesh, mi, tr, f, o2, d2, &Et, dec);
                REQUIRE((E0.nh && Et.nh >= 2 && Er.nh >= 2 && X0.nh));
                const double tt = K.ta + K.tb, budget = 2.0 * std::min(0.5 * tt, K.xs - E0.fbs(0, 0));
                std::vector<DiagLayer> lay = {lump_layer(lq[iA - 1], f), lump_layer(lq[iB - 1], f)};
                DiagSheet W = diag_cm_stack(eH, 1.0, lay, eH, 1.0, ct, f, false, true, budget);
                DiagSheet U = diag_cm_stack(eH, 1.0, lay, eH, 1.0, ct, f, false, true);
                std::complex<double> T = cpx(Et.xpr, 0, 0), R = cpx(Er.xpr, 0, 0);
                if (K.ta == 0.0005) // the thin stack of this host and position
                    T1 = T, R1 = R, Tr1 = W.T_te, Rr1 = W.R_te;
                const double dT = std::abs(T / T1 - W.T_te / Tr1), dR = std::abs(R / R1 - W.R_te / Rr1);
                const bool cap = budget < tt, fits = 0.5 * tt < K.xs && 0.5 * tt < K.Lh - K.xs;
                const std::complex<double> ch = cpx(E0.xpr, 0, 0) * T * cpx(X0.xpr, 0, 0);
                QD_DIAG("%.0f,%.0f,%.1f,%d,%.0f,%.4f,%d,%d,%.3e,%.3e,%.6f,%.4f,%.4f\n", 1e3 * K.Lh, 1e3 * K.xs, 1e3 * tt, tr, thd, 1e3 * budget, (int)fits,
                        (int)cap, dT, dR, ub::dB(W.T_te / U.T_te), ub::deg(W.T_te / U.T_te), ub::dB(ch));
                INFO("stack budget, host " << K.Lh << " front " << K.xs << " stack " << tt << " tracer " << tr << " theta " << thd);
                CHECK((Et.rt == 77 && Er.rt == 205));
                CHECK(dT < 1.0e-9);
                CHECK(dR < 1.0e-9);
                CHECK(std::norm(ch) < 1.0);
                CHECK(std::isfinite(std::norm(T)));
                CHECK(std::isfinite(std::norm(R)));
                if (fits) // on the uncapped reference
                    CHECK(!cap);
            }
        }
}

TEST_CASE("Unentered branches - the core's loss cap is a floor on the transmit port and leaves the reflect port alone", "[wave]")
{
    ub::OneThread one_thread;
    // qd_stack_resolve caps each layer's phase at 46.05 Np of imaginary part, 400 dB one way,
    // because the matrix is built from cos and sin of that phase. Below the cap the core is the
    // uncapped reference on both ports. Above it the reflect port is still the reference, and the
    // transmit port stands at the floor: the reference raised by what the cap took off.
    const double f = FRQ, k0 = 2.0 * QPI * f / C0, cap_Np = 46.05, tM = 0.001, gap = 1.0e-5, thd = 30.0;
    const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
    const std::complex<double> J(0.0, 1.0);
    const std::vector<double> loss_dB = {100.0, 300.0, 390.0, 399.0, 401.0, 450.0, 600.0, 2000.0};
    std::vector<LumpMtl> lp = {
        {2.50, 0.00, 0.002},           // 1 a coherent partner layer
        {2.50, 0.00, 0.002, 0.0, 0.2}, // 2 the same with tf, which takes the scattering form
        {1.50, 0.40, 0.0},             // 3 a host
    };
    for (double a : loss_dB)
    {
        double Np = a / 8.685889638, E = 2.0 * (Np / (k0 * tM)) * (Np / (k0 * tM));
        lp.push_back({1.0, E, tM});
    }
    ub::Pal pal = lump_palette(lp);

    QD_DIAG("\n# diag: the core's loss cap, 1 mm layer of eps 1 - jE with a 2 mm eps 2.50 layer, stack in air, 30 deg, scalar.\n");
    QD_DIAG("# loss_dB: one-way loss of the layer at this angle. R: reflect port against diag_cm_stack (form m) or diag_sc_stack (form s),\n");
    QD_DIAG("# complex. T_dB: transmit port. dT: up to 400 dB relative to the reference, above it (capped = 1, where a cap once stood) the\n");
    QD_DIAG("# level in dB against the reference\n");
    QD_DIAG("form,order,loss_dB,capped,R,T_dB,dT\n");
    for (int form = 0; form < 2; ++form)
        for (int order = 0; order < 2; ++order)
            for (size_t q = 0; q < loss_dB.size(); ++q)
            {
                const int iP = form ? 2 : 1, iM = 4 + (int)q;
                const std::complex<double> eM = lump_eps(lp[iM - 1], f);
                const double Np = -std::imag(k0 * std::sqrt(eM) * std::sqrt(1.0 - s2 / eM) * tM);
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0, gap}, {order ? iP : iM, order ? iM : iP}, {false, true}, mi);
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                ub::Ev Et = ub::traced<double>(pal, mesh, mi, 4, f, o, d, nullptr), Er = ub::traced<double>(pal, mesh, mi, 3, f, o, d, nullptr);
                REQUIRE((Et.nh == 2 && Er.nh == 2));
                std::vector<DiagLayer> lay = {lump_layer(lp[(order ? iP : iM) - 1], f), lump_layer(lp[(order ? iM : iP) - 1], f)};
                DiagSheet W = form ? diag_sc_stack(1.0, 1.0, 0.0, lay, 1.0, 1.0, 0.0, std::cos(th), f, false, true)
                                   : diag_cm_stack(1.0, 1.0, lay, 1.0, 1.0, std::cos(th), f, false, true);
                std::complex<double> T = cpx(Et.xpr, 0, 0), R = cpx(Er.xpr, 0, 0);
                const bool capped = Np > cap_Np;
                const double dR = std::abs(R - W.R_te);
                const double dT = capped ? std::abs(ub::dB(T / W.T_te)) : std::abs(T - W.T_te) / std::abs(W.T_te); // no floor: the reference at every loss
                QD_DIAG("%s,%s,%.2f,%d,%.3e,%.2f,%.3e\n", form ? "s" : "m", order ? "P|M" : "M|P", 8.685889638 * Np, (int)capped, dR, ub::dB(T), dT);
                INFO("loss cap form " << form << " order " << order << " loss " << loss_dB[q]);
                CHECK((Et.rt == 69 && Er.rt == 197));
                CHECK(dR < 1.0e-12);
                CHECK(dT < (capped ? 1.0e-6 : 1.0e-9));
                CHECK(std::isfinite(std::norm(T)));
                CHECK(std::isfinite(std::norm(R)));
            }

    // As the mirror of a host's series: an opaque lining on the far face of a traced solid. The
    // mirror is the half-space of the lining's material, whatever is behind it.
    QD_DIAG("\n# diag: the same layer as a lining on the far face of 100 mm of eps 1.50-0.40j, reflect pass, 30 deg: the host's series,\n");
    QD_DIAG("# eps 0 over eps 1.5, against the series whose mirror is the layer backed by air (S_layer) and the half-space of the\n");
    QD_DIAG("# layer's material (S_half)\n");
    QD_DIAG("loss_dB,capped,abs_S,S_layer,S_half\n");
    const std::complex<double> eH = lump_eps(lp[2], f), nH = std::sqrt(eH), ct = std::sqrt(1.0 - s2 / eH);
    for (size_t q = 0; q < loss_dB.size(); ++q)
    {
        const int iM = 4 + (int)q;
        const std::complex<double> eM = lump_eps(lp[iM - 1], f), nM = std::sqrt(eM), cM = std::sqrt(1.0 - s2 / eM);
        const double Np = -std::imag(k0 * nM * cM * tM);
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, 0.100, 0.100 + gap}, {3, 3, iM}, {false, true, false}, mi);
        auto c0 = ub::chain(pal, mesh, mi, {4, 3}, f, thd, {1.5, 1.5}), c1 = ub::chain(pal, mesh, mi, {4, 3}, f, thd, {1.5, 0.0});
        REQUIRE((c0.size() == 2 && c1.size() == 2));
        const std::complex<double> S = cpx(c1[1].xpr, 0, 0) / cpx(c0[1].xpr, 0, 0);
        DiagLayer lay = lump_layer(lp[iM - 1], f);
        std::complex<double> rl = diag_cm_stack(eH, 1.0, {lay}, 1.0, 1.0, ct, f, true, true).R_te;
        std::complex<double> rh = (nH * ct - nM * cM) / (nH * ct + nM * cM);
        const std::complex<double> rb = (nH * ct - std::cos(th)) / (nH * ct + std::cos(th));
        const std::complex<double> ph2 = std::exp(-2.0 * J * k0 * nH * ct * (0.100 - c0[0].fbs(0, 0)));
        const double dl = std::abs(S - 1.0 / (1.0 - rl * rb * ph2)), dh = std::abs(S - 1.0 / (1.0 - rh * rb * ph2));
        QD_DIAG("%.2f,%d,%.6f,%.3e,%.3e\n", 8.685889638 * Np, (int)(Np > cap_Np), std::abs(S), dl, dh);
        INFO("loss cap mirror, loss " << loss_dB[q]);
        CHECK((c1[1].rt & 0x04) != 0);
        CHECK(dl < 1.0e-9);
        CHECK(dh < 1.0e-9);
    }
}

TEST_CASE("Unentered branches - sheet and stack and slab series at float against double", "[wave]")
{
    ub::OneThread one_thread;
    // The composite, the core and the slab series are computed in double and written to dtype, and
    // the positions a chain runs on are the dtype's. The same chains at float and at double,
    // event by event: every hit code, resolved type and state word equal, and every coefficient
    // within what single-precision positions and a single-precision palette leave.
    const double f = FRQ, gap = 1.0e-5;
    const std::vector<LumpMtl> lp = {
        {1.50, 0.40, 0.0},   // 1 solid
        {5.24, 0.63, 0.006}, // 2 sheet 6 mm
        {2.50, 0.80, 0.010}, // 3 second sheet 10 mm
    };
    ub::Pal pal = lump_palette(lp);
    struct St
    {
        const char *name;
        std::vector<double> x;
        std::vector<int> mat;
        std::vector<bool> flip;
        std::vector<std::vector<int>> seq; // 1 = the transmit type of the row, 3 = reflect
    };
    const std::vector<St> stacks = {
        {"sheet in air", {0.0}, {2}, {false}, {{1}, {3}}},
        {"glued entry", {0.0, gap, 0.100}, {2, 1, 1}, {false, false, true}, {{1, 1}, {3}}},
        {"glued exit", {0.0, 0.100, 0.100 + gap}, {1, 1, 2}, {false, true, false}, {{1, 1}, {1, 3, 1}}},
        {"sheet inside", {0.0, 0.050, 0.100}, {1, 2, 1}, {false, false, true}, {{1, 1, 1}, {1, 3, 1}}},
        {"ordered stack", {0.0, gap}, {2, 3}, {false, true}, {{1}, {3}}},
        {"bare slab", {0.0, 0.100}, {1, 1}, {false, true}, {{1, 1}, {1, 3, 1}}},
    };
    QD_DIAG("\n# diag: float against double, the same chain event by event, every series summed. rel: largest distance of an xprmat\n");
    QD_DIAG("# entry over the largest entry of that event, worst event of the path; codes: hit code, resolved type and the three\n");
    QD_DIAG("# words equal on every event\n");
    QD_DIAG("stack,type,path,theta,events,rel,chain_dB,d_chain_dB,codes\n");
    double worst = 0.0;
    for (const St &S : stacks)
        for (int tr : {4, 5, 1, 2})
            for (size_t ip = 0; ip < S.seq.size(); ++ip)
                for (double thd : {0.0, 30.0, 60.0, 80.0})
                {
                    arma::uvec mi;
                    arma::mat mesh = ub::build(S.x, S.mat, S.flip, mi);
                    std::vector<int> seq;
                    for (int t : S.seq[ip])
                        seq.push_back((t == 3) ? ((tr == 4 || tr == 5) ? 3 : 0) : tr);
                    auto cd = ub::chain_t<double>(pal, mesh, mi, seq, f, thd, {0.0}), cf = ub::chain_t<float>(pal, mesh, mi, seq, f, thd, {0.0});
                    REQUIRE((cd.size() == seq.size() && cf.size() == seq.size()));
                    double rel = 0.0, pd = 1.0, pf = 1.0;
                    bool codes = true;
                    for (size_t q = 0; q < cd.size(); ++q)
                    {
                        double scale = arma::abs(cd[q].xpr).max(), dist = arma::abs(cf[q].xpr - cd[q].xpr).max();
                        rel = std::max(rel, dist / scale);
                        codes = codes && cd[q].ot == cf[q].ot && cd[q].rt == cf[q].rt && cd[q].prev(0) == cf[q].prev(0) &&
                                cd[q].cur(0) == cf[q].cur(0) && cd[q].buf(0) == cf[q].buf(0);
                        pd *= ub::sumsq(cd[q].xpr), pf *= ub::sumsq(cf[q].xpr);
                    }
                    worst = std::max(worst, rel);
                    QD_DIAG("%s,%d,%zu,%.0f,%zu,%.3e,%.4f,%.3e,%d\n", S.name, tr, ip, thd, cd.size(), rel, 10.0 * std::log10(pd), 10.0 * std::log10(pf / pd), (int)codes);
                    INFO("float " << S.name << " type " << tr << " path " << ip << " theta " << thd);
                    CHECK(codes);
                    CHECK(rel < 1.0e-4);
                }
    QD_DIAG("# worst %.3e\n", worst);
}

TEST_CASE("Unentered branches - the followed mark and the resolved flag across abutting and nested and overlapping bodies", "[wave]")
{
    ub::OneThread one_thread;
    // A leg that began at a reflection carries what that reflection decided: the followed mark in the buffer word where
    // the series was left to the tracer, the resolved flag where it was summed. Three ways such a leg meets a face that
    // is not the end of its body.
    //   A. A wall meshed as two abutting blocks of one material. The shared face is a transparent transition, the mark
    //   is carried, and the wall is the wall meshed as one block.
    //   B. An object nested in the wall. Its medium is stored over the mark, and the event that closes the leg judges
    //   its series for itself (an accepted limitation: the state holds one word for both).
    //   C. A body that overlaps the wall. A resolved ray that leaves the wall into it has the coefficient of the two
    //   faces of the hit rewritten to wall | body like any other ray, and forms no series.
    const double f = FRQ, gap = 1.0e-5;
    const std::vector<LumpMtl> lp = {
        {1.50, 0.40, 0.0},               // 1 lossy host
        {1.50, 0.40, 0.0},               // 2 the same material under a second index
        {1.50, 0.00, 0.0},               // 3 lossless host
        {1.50, 0.00, 0.0},               // 4 the same under a second index
        {5.24, 0.00, 0.0},               // 5 a lossless dense object
        {2.401e-9, 0.0, 0.0, 0.0, 0.3},  // 6 rigid, tf 0.3, no mass column
        {2.1609e-8, 0.0, 0.0, 0.0, 0.2}, // 7 rigid of another impedance, tf 0.2
    };
    ub::Pal pal = lump_palette(lp);
    auto words = [](const ub::Ev &E)
    {
        char b[32];
        std::snprintf(b, sizeof(b), "%04x.%04x.%04x", ub::w(E.prev(0)), ub::w(E.cur(0)), ub::w(E.buf(0)));
        return std::string(b);
    };
    auto prod = [](const std::vector<ub::Ev> &c)
    {
        std::complex<double> v = 1.0;
        for (const auto &E : c)
            v *= cpx(E.xpr, 0, 0);
        return v;
    };

    QD_DIAG("\n# diag: followed mark A, a wall of two abutting 50 mm blocks against one 100 mm block, enter, reflect at the far face\n");
    QD_DIAG("# with the series declined, leave by the front face. d_dB, d_deg: the path against the one-block wall. bits: the last event\n");
    QD_DIAG("# at eps 0 against eps 1.5. w_refl, w_cross: prev.cur.buf after the reflection and after the crossing back\n");
    QD_DIAG("blocks,tracer,theta,rt_cross,d_dB,d_deg,bits,w_refl,w_cross\n");
    for (const std::pair<int, int> &B : {std::pair<int, int>{1, 1}, {1, 2}, {3, 4}})
        for (int tr : {4, 5})
            for (double thd : {20.0, 60.0})
            {
                arma::uvec ma, m1;
                arma::mat mesh_a = ub::build({0.0, 0.050, 0.050 + gap, 0.100}, {B.first, B.first, B.second, B.second}, {false, true, false, true}, ma);
                arma::mat mesh_1 = ub::build({0.0, 0.100}, {B.first, B.first}, {false, true}, m1);
                auto ca = ub::chain(pal, mesh_a, ma, {tr, tr, 3, tr, tr}, f, thd, {1.5}), cb = ub::chain(pal, mesh_a, ma, {tr, tr, 3, tr, tr}, f, thd, {1.5, 1.5, 1.5, 1.5, 0.0});
                auto c1 = ub::chain(pal, mesh_1, m1, {tr, 3, tr}, f, thd, {1.5});
                REQUIRE((ca.size() == 5 && cb.size() == 5 && c1.size() == 3));
                const std::complex<double> q = prod(ca) / prod(c1);
                const bool bits = ub::same_bits(ca[4].xpr, cb[4].xpr);
                QD_DIAG("%d|%d,%d,%.0f,%d,%.3e,%.4f,%d,%s,%s\n", B.first, B.second, tr, thd, (int)ca[3].rt, ub::dB(q), ub::deg(q), (int)bits, words(ca[2]).c_str(), words(ca[3]).c_str());
                INFO("mark A blocks " << B.first << "|" << B.second << " tracer " << tr << " theta " << thd);
                CHECK(ub::w(ca[2].buf(0)) == 0x4000u);                    // the reflection left the series to the tracer
                CHECK(ub::w(ca[3].buf(0)) == 0x4000u);                    // and the shared face carries the mark
                CHECK((ca[3].rt & ~0x10) == 73);                          // a same-medium pass
                CHECK(((ca[3].rt & 0x10) != 0) == (B.first != B.second)); // with the fix-flag where the two blocks have two indices
                CHECK(ca[3].prev(0) == ca[2].prev(0));
                CHECK(bits); // so the leg forms no series
                CHECK(std::abs(ub::dB(q)) < 2.0e-3);
                CHECK(std::abs(ub::deg(q)) < 0.05);
            }

    QD_DIAG("\n# diag: followed mark B, a 20 mm eps 5.24 object nested in the middle of 100 mm of host, the same path. S_dB: the series\n");
    QD_DIAG("# the last event forms, eps 0 over eps 1.5; twin: the same without the object, which forms none\n");
    QD_DIAG("host,tracer,theta,rt_in,rt_out,w_refl,w_in,w_out,S_dB,S_deg,twin_bits\n");
    for (int iH : {1, 3})
        for (int tr : {4, 5})
            for (double thd : {20.0, 60.0})
            {
                arma::uvec mn, m1;
                arma::mat mesh_n = ub::build({0.0, 0.040, 0.060, 0.100}, {iH, 5, 5, iH}, {false, false, true, true}, mn);
                arma::mat mesh_1 = ub::build({0.0, 0.100}, {iH, iH}, {false, true}, m1);
                const std::vector<int> seq = {tr, tr, tr, 3, tr, tr, tr};
                auto ca = ub::chain(pal, mesh_n, mn, seq, f, thd, {1.5}), cb = ub::chain(pal, mesh_n, mn, seq, f, thd, {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0});
                auto t0 = ub::chain(pal, mesh_1, m1, {tr, 3, tr}, f, thd, {1.5}), t1 = ub::chain(pal, mesh_1, m1, {tr, 3, tr}, f, thd, {1.5, 1.5, 0.0});
                REQUIRE((ca.size() == 7 && cb.size() == 7 && t0.size() == 3 && t1.size() == 3));
                const std::complex<double> S = cpx(cb[6].xpr, 0, 0) / cpx(ca[6].xpr, 0, 0);
                const bool twin = ub::same_bits(t0[2].xpr, t1[2].xpr);
                QD_DIAG("%d,%d,%.0f,%d,%d,%s,%s,%s,%.4f,%.4f,%d\n", iH, tr, thd, (int)ca[4].rt, (int)ca[5].rt, words(ca[3]).c_str(), words(ca[4]).c_str(),
                        words(ca[5]).c_str(), ub::dB(S), ub::deg(S), (int)twin);
                INFO("mark B host " << iH << " tracer " << tr << " theta " << thd);
                CHECK(ub::w(ca[3].buf(0)) == 0x4000u); // marked at the reflection
                CHECK(ca[4].rt == 72);                 // the nested object is deferred
                CHECK(ca[4].buf(0) == 5);              // and its medium replaces the mark
                CHECK(ca[5].buf(0) == 0);              // which does not come back when the object is left
                CHECK(twin);                           // without the object the leg forms no series
                CHECK(std::abs(S - 1.0) > 1.0e-3);     // with it the leg judges for itself: the accepted limitation
            }

    QD_DIAG("\n# diag: resolved flag C, a body from -50 mm to 30 mm overlapping a wall from 0 to 100 mm, every series summed: enter the\n");
    QD_DIAG("# body, cross into the wall, reflect at its far face, come back and leave the wall into the body. The crossing back\n");
    QD_DIAG("# against the same crossing where the body ends at the wall's face: d_dB, d_deg. bits: that event at eps 0 against 1.5\n");
    QD_DIAG("wall|body,tracer,theta,rt,rt_abut,d_dB,d_deg,bits,w_before,w_after\n");
    // level: the row is asserted in dB; point: also in phase. With the refracting tracer ray_mesh_interact bends the
    // segment at the face of a deferred body as if it were entered from air, so the two polylines differ inside the
    // overlap and each coefficient is the field at its own. The rigid rows carry no mass column, where an exit is
    // written along the normal (an accepted limitation), and are printed only.
    struct Ov
    {
        int a, b, tr;
        double thd;
        bool level, point;
    };
    for (const Ov &V : {Ov{3, 5, 5, 30.0, true, false}, Ov{3, 5, 5, 60.0, true, false}, Ov{1, 5, 4, 0.0, true, true}, Ov{3, 5, 4, 0.0, true, true},
                        Ov{1, 5, 5, 30.0, true, false}, Ov{6, 7, 4, 30.0, false, false}, Ov{6, 7, 5, 60.0, false, false}})
    {
        arma::uvec mo, mb;
        arma::mat mesh_o = ub::build({-0.050, 0.0, 0.030, 0.100}, {V.b, V.a, V.b, V.a}, {false, false, true, true}, mo);
        arma::mat mesh_b = ub::build({-0.050, 0.0, gap, 0.100}, {V.b, V.b, V.a, V.a}, {false, true, false, true}, mb);
        const std::vector<int> so = {V.tr, V.tr, V.tr, 3, V.tr, V.tr, V.tr}, sb = {V.tr, V.tr, 3, V.tr, V.tr};
        auto co = ub::chain(pal, mesh_o, mo, so, f, V.thd, {0.0}), ce = ub::chain(pal, mesh_o, mo, so, f, V.thd, {0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0});
        auto cb = ub::chain(pal, mesh_b, mb, sb, f, V.thd, {0.0});
        REQUIRE((co.size() == 7 && ce.size() == 7 && cb.size() == 5));
        const std::complex<double> q = cpx(co[5].xpr, 0, 0) / cpx(cb[3].xpr, 0, 0);
        const bool bits = ub::same_bits(co[5].xpr, ce[5].xpr);
        QD_DIAG("%d|%d,%d,%.0f,%d,%d,%.3e,%.4f,%d,%s,%s\n", V.a, V.b, V.tr, V.thd, (int)co[5].rt, (int)cb[3].rt, ub::dB(q), ub::deg(q), (int)bits,
                words(co[4]).c_str(), words(co[5]).c_str());
        INFO("resolved C wall " << V.a << " body " << V.b << " tracer " << V.tr << " theta " << V.thd);
        CHECK((ub::w(co[3].cur(0)) & 0x8000u) != 0); // the reflection summed the wall's series
        CHECK((ub::w(co[4].cur(0)) & 0x8000u) != 0); // the body's face met from inside the wall keeps the flag
        CHECK(co[4].buf(0) == V.b);                  // and defers the body
        CHECK(mat_of(co[5].cur(0)) == V.b);          // the wall's own face then leads into it
        CHECK((ub::w(co[5].cur(0)) & 0x8000u) == 0);
        CHECK(co[5].buf(0) == 0);
        CHECK(bits); // a resolved ray forms no series
        if (V.level)
            CHECK(std::abs(ub::dB(q)) < 2.0e-3);
        if (V.point)
            CHECK(std::abs(ub::deg(q)) < 0.05);
    }
}

TEST_CASE("Unentered branches - a verdict handed in on a leg known not parallel and near a pole", "[wave]")
{
    ub::OneThread one_thread;
    // Where a verdict is handed in the series is summed exactly where it says. Two gates a call
    // deciding for itself reads are then not read: the known-non-parallel flag of the leg, and
    // the pole, where a denominator below 1e-2 is clamped in magnitude instead of the bounces
    // being handed back. The first cannot be reached by a caller that uses one threshold for a
    // run: the base frequency sums on such a leg only at a threshold of zero, where the gate is
    // not read without a verdict either. Its rows set the flag by hand.
    const double gap = 1.0e-5;
    const std::complex<double> J(0.0, 1.0);
    const std::vector<LumpMtl> lp = {
        {1.50, 0.40, 0.0}, // 1 lossy solid
        {1.0e4, 0.0, 0.0}, // 2 lossless and very dense: its faces reflect 0.993 from inside at 80 deg
    };
    ub::Pal pal = lump_palette(lp);
    (void)gap;
    const std::vector<uint8_t> v1 = {1}, v0 = {0};

    QD_DIAG("\n# diag: a verdict on a leg flagged not parallel, 100 mm of eps 1.50-0.40j, 30 deg, the event that closes the leg.\n");
    QD_DIAG("# sum: verdict 1 at eps 0.15 against no verdict at eps 0; decline: verdict 0 at eps 0 against no verdict at eps 0.15;\n");
    QD_DIAG("# bitwise. S_dB: what the series is worth on that event\n");
    QD_DIAG("tracer,pass,sum_bits,decline_bits,vout_sum,vout_decline,S_dB\n");
    for (int tr : {4, 5})
        for (int it : {tr, 3})
        {
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, 0.100}, {1, 1}, {false, true}, mi);
            arma::mat o, d;
            diag_ray_at(30.0, o, d);
            ub::Opt en;
            en.relaunch = true;
            ub::Ev E0 = ub::traced<double>(pal, mesh, mi, tr, FRQ, o, d, nullptr, en);
            ub::onward(E0, o, d);
            E0.prev(0) = (short)(ub::w(E0.prev(0)) | 0x8000u); // the leg is known not parallel
            auto run = [&](double eps, const std::vector<uint8_t> *v)
            {
                ub::Opt op;
                op.eps = eps, op.vin = v, op.relaunch = it != 3;
                return ub::traced<double>(pal, mesh, mi, it, FRQ, o, d, &E0, op);
            };
            ub::Ev a = run(0.15, nullptr), b = run(0.15, &v1), c = run(0.0, nullptr), e = run(0.0, &v0);
            auto same = [](const ub::Ev &x, const ub::Ev &y)
            { return ub::same_bits(x.xpr, y.xpr) && x.prev(0) == y.prev(0) && x.cur(0) == y.cur(0) && x.buf(0) == y.buf(0) && x.rt == y.rt; };
            const std::complex<double> S = cpx(b.xpr, 0, 0) / cpx(a.xpr, 0, 0);
            QD_DIAG("%d,%s,%d,%d,%d,%d,%.4f\n", tr, (it == 3) ? "reflect" : "transmit", (int)same(b, c), (int)same(e, a), (int)b.vout, (int)e.vout, ub::dB(S));
            INFO("verdict on a non-parallel leg, tracer " << tr << " type " << it);
            CHECK(same(b, c));
            CHECK(same(e, a));
            CHECK((b.vout == 1 && c.vout == 1 && a.vout == 0 && e.vout == 0));
            CHECK(std::abs(S - 1.0) > 1.0e-3);
        }

    QD_DIAG("\n# diag: a verdict near a pole, 20 mm of lossless eps 1e4 at 80 deg, the frequency tuned to a resonance of the leg and\n");
    QD_DIAG("# detuned by psi rad of round-trip phase. den: the Airy denominator from the media. given: verdict 1; own: no verdict at\n");
    QD_DIAG("# eps 0, which sums near the pole as anywhere else. abs_S: the series the verdict run applies; S: against the Airy series,\n");
    QD_DIAG("# which is finite and is not clamped; off_dB: the series against the exact one\n");
    QD_DIAG("pass,psi,f_GHz,den,vout_given,vout_own,own_declines,abs_S,S,off_dB\n");
    for (int it : {4, 3})
        for (double psi : {0.0, 0.003, 0.008, 0.5})
        {
            const double thd = 80.0, th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), Ls = 0.020;
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, Ls}, {2, 2}, {false, true}, mi);
            arma::mat o, d;
            diag_ray_at(thd, o, d);
            ub::Ev G = ub::traced<double>(pal, mesh, mi, 4, FRQ, o, d, nullptr); // the geometry does not depend on the frequency
            const double dl = Ls - G.fbs(0, 0), b = std::sqrt(1.0e4 - s2), c = std::cos(th);
            const double N = std::round(2.0 * b * dl * FRQ / C0), fN = N * C0 / (2.0 * b * dl), f = fN * (1.0 + psi / (2.0 * QPI * N));
            const double rs = (b - c) / (b + c), k = 2.0 * QPI * f / C0;
            const std::complex<double> den = 1.0 - rs * rs * std::exp(-2.0 * J * k * b * dl);
            ub::Ev E0 = ub::traced<double>(pal, mesh, mi, 4, f, o, d, nullptr);
            arma::mat o1, d1;
            ub::onward(E0, o1, d1);
            auto run = [&](double eps, const std::vector<uint8_t> *v)
            {
                ub::Opt op;
                op.eps = eps, op.vin = v;
                return ub::traced<double>(pal, mesh, mi, it, f, o1, d1, &E0, op);
            };
            ub::Ev g = run(0.15, &v1), own = run(0.0, nullptr), dec = run(1.5, nullptr);
            const std::complex<double> S = cpx(g.xpr, 0, 0) / cpx(dec.xpr, 0, 0);
            const bool pole = std::abs(den) < 1.0e-2, own_dec = ub::same_bits(own.xpr, dec.xpr);
            const std::complex<double> Sref = 1.0 / den;
            QD_DIAG("%s,%.3f,%.9f,%.6f,%d,%d,%d,%.6f,%.3e,%.4f\n", (it == 3) ? "reflect" : "transmit", psi, f * 1e-9, std::abs(den), (int)g.vout, (int)own.vout,
                    (int)own_dec, std::abs(S), std::abs(S - Sref) / std::abs(Sref), ub::dB(S * den));
            INFO("verdict near a pole, type " << it << " psi " << psi);
            CHECK(g.vout == 1);
            CHECK(own.vout == 1); // no pole hands the bounces back
            CHECK(!own_dec);
            CHECK(std::abs(S - Sref) / std::abs(Sref) < 1.0e-6);
            if (pole)
                CHECK(std::abs(S) > 100.0); // the rows are inside the band a clamp once stood on
            CHECK(ub::same_bits(g.xpr, own.xpr));
        }
}

TEST_CASE("Scalar closed face - a lossy box on a rigid wall reflects no more than it received and is its sheet", "[wave]")
{
    ub::OneThread one_thread;
    // A closed face has nothing propagating behind it: a rigid wall of the acoustic mapping from 0.003 deg on. Seen
    // from a lossless medium its Fresnel reflection has unit magnitude; seen from a lossy one it has not: part of what
    // arrives is absorbed in the medium at the wall, and for porous absorbers, which are rarer than air, the magnitude
    // is far below one. A closed face reflects at the Fresnel magnitude in the scalar types as in the EM types, capped
    // at one as the far face of a sheet caps it; a magnitude of one would hand the absorbed power back, and a porous
    // box on a rigid floor would reflect more than it received.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update. The materials are porous
    // rows of data/materials_ac.csv at 1 kHz, without tf and without the mass column; types 1 and 0 on them are the
    // control and not EM materials.
    //
    // A. The reflect port of the box, every path, against the sheet of its thickness
    // B. One reflection off the wall inside the box, scalar against the TE channel
    // C. Twins with every digit: what the rule must not move
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5, k0 = 2.0 * QPI * f / C0;
    const cd J(0.0, 1.0);
    struct Row
    {
        const char *name;
        double a, c, thk;
    };
    // name, a, c of the table (eps'' = 17.98 c at 1 GHz, which stands for 1 kHz), thickness of the guidelines
    const std::vector<Row> rows = {{"carpet_thin", 0.312032, 0.0191446, 0.008}, {"carpet_thick", 0.195963, 0.0229164, 0.020}, {"curtain_light_cotton", 0.548166, 0.0123962, 0.008}, {"foam_light_melamine", 0.312032, 0.0191446, 0.050}, {"mineral_wool", 0.141626, 0.0247974, 0.050}};
    const double eW = 2.401e-9; // concrete
    auto kz_of = [](cd e, double p)
    { cd v = std::sqrt(e - p); return (std::imag(v) > 0.0 || (std::imag(v) == 0.0 && std::real(v) < 0.0)) ? -v : v; };
    auto sum = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int tr, double thd, double eps)
    {
        bf::Acc A;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        bf::walk(P, mesh, mi, tr, f, thd, eps, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 400, A);
        return A;
    };

    // ---- A. The reflect port of the box against the sheet ----
    QD_DIAG("\n# diag: scalar closed face A, a porous box on concrete, no tf anywhere, through the intersector. R_sheet: the layer as a sheet\n");
    QD_DIAG("# on the wall, dB. R_sum, R_fol: the box as a traced solid, every path, series summed and every bounce followed, dB. d_sum,\n");
    QD_DIAG("# d_fol: the box over the sheet, dB; deg: its phase, both referenced to the wall face. k: 0 scalar or VV, 3 HH. nR: paths\n");
    QD_DIAG("material,mm,type,k,theta,R_sheet,R_sum,R_fol,d_sum,d_fol,deg,nR\n");
    for (const Row &M : rows)
        for (int tr : {4, 5, 1})
            for (double thd : {10.0, 30.0, 45.0, 60.0, 75.0, 85.0})
            {
                ub::Pal P = lump_palette({{M.a, 17.98 * M.c}, {eW}, {M.a, 17.98 * M.c, M.thk}});
                arma::uvec ms, mb;
                arma::mat mesh_s = ub::build({0.0, gap, gap + 0.2}, {3, 2, 2}, {false, false, true}, ms);
                arma::mat mesh_b = ub::build({0.0, M.thk, M.thk + gap, M.thk + gap + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, mb);
                const double c = std::cos(thd * QPI / 180.0);
                const cd owed = std::exp(-2.0 * J * k0 * c * M.thk); // the box stands behind the plane the sheet's layer lies in front of
                bf::Acc S = sum(P, mesh_s, ms, tr, thd, 0.0), B0 = sum(P, mesh_b, mb, tr, thd, 0.0), B1 = sum(P, mesh_b, mb, tr, thd, 1.5);
                for (int ch = 0; ch < ((tr > 2) ? 1 : 2); ++ch)
                {
                    const cd Rs = S.R[ch] * owed, R0 = B0.R[ch], R1 = B1.R[ch];
                    QD_DIAG("%s,%.0f,%d,%d,%.0f,%.4f,%.4f,%.4f,%+.4f,%+.4f,%+.3f,%d\n", M.name, M.thk * 1e3, tr, ch ? 3 : 0, thd, ub::dB(Rs), ub::dB(R0), ub::dB(R1),
                            ub::dB(R0 / Rs), ub::dB(R1 / Rs), ub::deg(R0 / Rs), B1.nR);
                    INFO("A, " << M.name << ", type " << tr << ", k " << (ch ? 3 : 0) << ", theta " << thd);
                    // Nothing passes a wall without tf, and nothing is created in front of it
                    CHECK(std::norm(B0.T[ch]) == 0.0);
                    CHECK(std::norm(R0) < 1.0 + 1.0e-6);
                    CHECK(std::norm(R1) < 1.0 + 1.0e-6);
                    // One layer as a box and as a sheet, whether its series is summed or followed
                    CHECK(std::abs(ub::dB(R0 / Rs)) < 2.0e-2);
                    CHECK(std::abs(ub::dB(R1 / Rs)) < 2.0e-2);
                    CHECK(std::abs(ub::deg(R0 / Rs)) < 0.2);
                }
            }

    // ---- B. One reflection off the wall, scalar against TE ----
    QD_DIAG("\n# diag: scalar closed face B, one path: enter the box, reflect off the wall, both events with the series declined. sc, vv: the\n");
    QD_DIAG("# path in the scalar type and in the VV channel, which is TE in this frame, dB. rF: the Fresnel reflection box | wall written\n");
    QD_DIAG("# from the two media, dB. d: scalar over VV, dB\n");
    QD_DIAG("material,theta,sc,vv,rF,d\n");
    for (const Row &M : rows)
        for (double thd : {10.0, 30.0, 45.0, 60.0, 75.0, 85.0})
        {
            ub::Pal P = lump_palette({{M.a, 17.98 * M.c}, {eW}});
            arma::uvec mb;
            arma::mat mesh_b = ub::build({0.0, M.thk, M.thk + gap, M.thk + gap + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, mb);
            auto cs = ub::chain(P, mesh_b, mb, {4, 3}, f, thd, {1.5}), ce = ub::chain(P, mesh_b, mb, {1, 0}, f, thd, {1.5});
            REQUIRE((cs.size() == 2 && ce.size() == 2));
            const cd vs = cpx(cs[0].xpr, 0, 0) * cpx(cs[1].xpr, 0, 0), ve = cpx(ce[0].xpr, 0, 0) * cpx(ce[1].xpr, 0, 0);
            const double p = std::pow(std::sin(thd * QPI / 180.0), 2);
            const cd y1 = kz_of(cd(M.a, -17.98 * M.c), p), y2 = kz_of(cd(eW, 0.0), p), rF = (y1 - y2) / (y1 + y2);
            QD_DIAG("%s,%.0f,%.4f,%.4f,%.4f,%+.6f\n", M.name, thd, ub::dB(vs), ub::dB(ve), ub::dB(rF), ub::dB(vs / ve));
            INFO("B, " << M.name << ", theta " << thd);
            CHECK(std::abs(rF) < 1.0); // the rows of this table are below one, where the cap does not act
            CHECK(std::abs(ub::dB(vs / ve)) < 1.0e-9);
        }

    // ---- C. Twins with every digit ----
    QD_DIAG("\n# diag: scalar closed face C, twins, the complex coefficient with every digit: the path of B for a lossless box, the same in the\n");
    QD_DIAG("# VV and HH channels for a lossy one, the bare wall from air, and a lossy box entered and left through its front face\n");
    QD_DIAG("twin,theta,re,im\n");
    for (double thd : {10.0, 45.0, 75.0})
    {
        ub::Pal P = lump_palette({{2.50, 0.0}, {eW}, {0.312032, 17.98 * 0.0191446}});
        arma::uvec m1, m3, mw;
        arma::mat mesh1 = ub::build({0.0, 0.02, 0.02 + gap, 0.22 + gap}, {1, 1, 2, 2}, {false, true, false, true}, m1);
        arma::mat mesh3 = ub::build({0.0, 0.02, 0.02 + gap, 0.22 + gap}, {3, 3, 2, 2}, {false, true, false, true}, m3);
        arma::mat meshw = ub::build({0.0, 0.2}, {2, 2}, {false, true}, mw);
        auto prod = [](const std::vector<ub::Ev> &v, int k)
        { cd x = 1.0; for (const ub::Ev &e : v) x *= cpx(e.xpr, 0, k); return x; };
        const cd a = prod(ub::chain(P, mesh1, m1, {4, 3}, f, thd, {1.5}), 0), b = prod(ub::chain(P, mesh1, m1, {5, 3}, f, thd, {1.5}), 0);
        const cd c0 = prod(ub::chain(P, mesh3, m3, {1, 0}, f, thd, {1.5}), 0), c3 = prod(ub::chain(P, mesh3, m3, {1, 0}, f, thd, {1.5}), 3);
        const cd w = prod(ub::chain(P, meshw, mw, {3}, f, thd, {1.5}), 0), e = prod(ub::chain(P, mesh3, m3, {3}, f, thd, {1.5}), 0);
        QD_DIAG("lossless box type 4,%.0f,%.17g,%.17g\n", thd, a.real(), a.imag());
        twin("scalar closed face C", twin_key("lossless box type 4,%.0f", thd), {a.real(), a.imag()});
        QD_DIAG("lossless box type 5,%.0f,%.17g,%.17g\n", thd, b.real(), b.imag());
        twin("scalar closed face C", twin_key("lossless box type 5,%.0f", thd), {b.real(), b.imag()});
        QD_DIAG("lossy box VV,%.0f,%.17g,%.17g\n", thd, c0.real(), c0.imag());
        twin("scalar closed face C", twin_key("lossy box VV,%.0f", thd), {c0.real(), c0.imag()});
        QD_DIAG("lossy box HH,%.0f,%.17g,%.17g\n", thd, c3.real(), c3.imag());
        twin("scalar closed face C", twin_key("lossy box HH,%.0f", thd), {c3.real(), c3.imag()});
        QD_DIAG("bare wall type 3,%.0f,%.17g,%.17g\n", thd, w.real(), w.imag());
        twin("scalar closed face C", twin_key("bare wall type 3,%.0f", thd), {w.real(), w.imag()});
        QD_DIAG("lossy box front face type 3,%.0f,%.17g,%.17g\n", thd, e.real(), e.imag());
        twin("scalar closed face C", twin_key("lossy box front face type 3,%.0f", thd), {e.real(), e.imag()});
        INFO("C, theta " << thd);
        CHECK(std::abs(std::abs(a) - std::abs(prod(ub::chain(P, mesh1, m1, {4}, f, thd, {1.5}), 0))) < 1.0e-12); // a lossless box: the wall returns all of it
        CHECK(std::abs(std::abs(w) - 1.0) < 1.0e-12);
    }
}

TEST_CASE("One solver - a sheet and a stack are one sum and no mirror is clamped", "[wave]")
{
    ub::OneThread one_thread;
    // One solver resolves every sheet and every stack, and the series of a traced solid takes its mirrors as the faces
    // are: a reflection above one, seen from a lossy medium, is the Fresnel value and is not capped before the series
    // is summed (a cap there makes a wall lined on both faces pass 44 times what it received). Through
    // ray_triangle_intersect, ray_mesh_interact and ray_state_update.
    //   1 host eps 5.24, 2 host eps 5.24 with tf -0.5, 3 .. sheets and solids as listed per table.
    //   No material here carries the mass column, and only table H carries tf.
    typedef std::complex<double> cd;
    const cd J(0.0, 1.0), U(1.0, 0.0);
    const double f = FRQ, G = 1.0e-5, k0 = 2.0 * QPI * f / C0;
    auto dBp = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };

    // A: a traced solid with a coating, every series summed, every path, against the matrix of the wall
    QD_DIAG("\n# diag: one solver A, a solid 6 mm with a 6 mm coating on its near face, its far face or both, every series summed, every\n");
    QD_DIAG("# path. dT, dR: the two ports over the characteristic matrix of the wall, dB; sum: the two ports together; sum_cm: the matrix's\n");
    QD_DIAG("wall,coat,solid,type,ch,theta,nT,dT,dR,sum,sum_cm\n");
    {
        const std::vector<std::pair<double, double>> coats = {{2.5, 0.0}, {5.24, 0.0}}, solids = {{0.31, 0.2}, {0.9, 0.2}, {1.5, 0.4}};
        const double dS = 0.006, dC = 0.006;
        for (int kind = 0; kind < 3; ++kind)
            for (const auto &c : coats)
                for (const auto &s : solids)
                {
                    ub::Pal P = lump_palette({{s.first, s.second, 0.0}, {c.first, c.second, dC}});
                    const cd eS(s.first, -s.second), eC(c.first, -c.second);
                    arma::uvec mi;
                    arma::mat mesh = (kind == 0)   ? ub::build({0.0, G, G + dS}, {2, 1, 1}, {false, false, true}, mi)
                                     : (kind == 1) ? ub::build({0.0, dS, dS + G}, {1, 1, 2}, {false, true, false}, mi)
                                                   : ub::build({0.0, G, G + dS, dS + 2.0 * G}, {2, 1, 1, 2}, {false, false, true, false}, mi);
                    std::vector<DiagLayer> lay;
                    if (kind != 1)
                        lay.push_back({eC, U, dC});
                    lay.push_back({eS, U, dS});
                    if (kind != 0)
                        lay.push_back({eC, U, dC});
                    for (int tr : {4, 1})
                        for (double thd : {45.0, 75.0, 85.0})
                        {
                            arma::mat o, d;
                            diag_ray_at(thd, o, d);
                            bf::Acc A;
                            bf::walk(P, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 60, A);
                            DiagSheet W = diag_cm_stack(U, U, lay, U, U, std::cos(thd * QPI / 180.0), f, false, tr > 2);
                            for (int ch = 0; ch < ((tr > 2) ? 1 : 2); ++ch)
                            {
                                const double T = std::norm(A.T[ch]), R = std::norm(A.R[ch]), Tm = std::norm(ch ? W.T_tm : W.T_te), Rm = std::norm(ch ? W.R_tm : W.R_te);
                                QD_DIAG("%s,%g-%gj,%g-%gj,%d,%s,%.0f,%d,%+.4f,%+.4f,%.6f,%.6f\n", kind == 0 ? "near" : (kind == 1 ? "far" : "both"), c.first, c.second,
                                        s.first, s.second, tr, (tr > 2) ? "sc" : (ch ? "TM" : "TE"), thd, A.nT, dBp(T) - dBp(Tm), dBp(R) - dBp(Rm), T + R, Tm + Rm);
                                INFO("A, kind " << kind << ", coat " << c.first << ", solid " << s.first << "-" << s.second << "j, type " << tr << ", ch " << ch << ", theta " << thd);
                                CHECK(std::abs(dBp(T) - dBp(Tm)) < 5.0e-3); // the relaunch offsets of a 6 mm leg
                                CHECK(std::abs(dBp(R) - dBp(Rm)) < 2.0e-2); // 0.0072 dB where the wall reflects 15 dB less than it passes
                                CHECK(T + R < 1.0005);
                            }
                        }
                }
    }

    // A seeded leg: a ray launched inside the lossless host eps 5.24, past the critical angle of host | air
    auto seed = [&](double thd)
    {
        ub::Ev S;
        const double th = thd * QPI / 180.0;
        S.prev = arma::Col<short>{0}, S.cur = arma::Col<short>{1}, S.buf = arma::Col<short>{0};
        S.path_dirN = arma::mat{{std::cos(th), std::sin(th), 0.0}};
        S.accg = arma::vec{0.1}, S.accr = arma::vec{0.1};
        return S;
    };

    // B: the far face of a sheet on a closed far side. C: the coating against its traced solid
    QD_DIAG("\n# diag: one solver B, a coating on the lossless eps 5.24 met from inside, past the critical angle of eps 5.24 | air, type 0,\n");
    QD_DIAG("# no tf. R_te, R_tm: the reflect port of the sheet event; d_te, d_tm: against the characteristic matrix, in amplitude.\n");
    QD_DIAG("# solid: the same layer as a traced solid, every series summed, every path, the power it returns; s/s: sheet over solid, dB\n");
    QD_DIAG("sheet,mm,theta,R_te,R_tm,d_te,d_tm,solid_te,solid_tm,s/s_te,s/s_tm\n");
    for (const auto &s : std::vector<std::pair<double, double>>{{5.24, 5.0}, {5.24, 0.63}, {9.0, 1.2}, {1.5, 0.4}})
        for (double thk : {0.0002, 0.001, 0.006})
            for (double thd : {27.0, 37.0, 60.0})
            {
                ub::Pal P = lump_palette({{5.24, 0.0, 0.0}, {s.first, s.second, 0.0}, {s.first, s.second, thk}});
                const cd eH(5.24, 0.0), eS(s.first, -s.second);
                const double th = thd * QPI / 180.0, kt = k0 * std::sin(th) * std::sqrt(5.24);
                arma::uvec mS, mX;
                arma::mat meshS = ub::build({0.0, G}, {1, 3}, {true, false}, mS), meshX = ub::build({0.0, G, G + thk}, {1, 2, 2}, {true, false, true}, mX);
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                ub::Ev S0 = seed(thd), E = ub::traced<double>(P, meshS, mS, 0, f, o, d, &S0);
                REQUIRE((E.nh == 2 && E.rt != 0));
                DiagSheet W = diag_cm_stack(eH, U, {{eS, U, thk}}, U, U, std::cos(th), f, true, false);
                const double Rte = std::abs(cpx(E.xpr, 0, 0)), Rtm = std::abs(cpx(E.xpr, 0, 3));
                cf::Acc A;
                cf::walk(P, meshX, mX, 1, f, kt, 0.0, 1, &S0, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 400, A);
                const double Xte = std::norm(A.R[0]), Xtm = std::norm(A.R[1]);
                QD_DIAG("%g-%gj,%.1f,%.0f,%.6f,%.6f,%.3e,%.3e,%.6f,%.6f,%+.4f,%+.4f\n", s.first, s.second, thk * 1e3, thd, Rte, Rtm, std::abs(Rte - std::abs(W.R_te)),
                        std::abs(Rtm - std::abs(W.R_tm)), Xte, Xtm, dBp(Rte * Rte) - dBp(Xte), dBp(Rtm * Rtm) - dBp(Xtm));
                INFO("B, sheet " << s.first << "-" << s.second << "j, " << thk * 1e3 << " mm, theta " << thd);
                CHECK(std::abs(Rte - std::abs(W.R_te)) < 1.0e-9);
                CHECK(std::abs(Rtm - std::abs(W.R_tm)) < 1.0e-9); // TM: above one at the far face, and not capped
                CHECK(Rte * Rte < 1.0 + 1.0e-9);
                CHECK(Rtm * Rtm < 1.0 + 1.0e-9);
                CHECK(Xte < 1.0005); // the traced solid returns no more than it received, in either channel
                CHECK(Xtm < 1.0005);
                CHECK(std::abs(dBp(Rte * Rte) - dBp(Xte)) < 0.03); // a coating is its traced solid
                CHECK(std::abs(dBp(Rtm * Rtm) - dBp(Xtm)) < 0.03);
            }

    // E: no pole clamp. A lossless sheet at a transmission resonance near grazing, and with a forced reflection
    QD_DIAG("\n# diag: one solver E, a lossless sheet of eps 5.24 in air at a transmission resonance, scalar. N: half waves across the sheet;\n");
    QD_DIAG("# T_dB: the transmit port; d_cm: both ports against the characteristic matrix, complex (rows without tf); sum: the two ports together\n");
    QD_DIAG("tf,theta,N,mm,T_dB,d_cm,sum\n");
    for (double tf : {0.0, -0.9, -0.99})
        for (double thd : {30.0, 85.0, 89.0})
            for (int N : {1, 3})
            {
                const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), thk = N * QPI / (k0 * std::sqrt(5.24 - s2));
                LumpMtl M;
                M.a = 5.24, M.thk = thk, M.tf = tf;
                ub::Pal P = lump_palette({M});
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0}, {1}, {false}, mi), o, d;
                diag_ray_at(thd, o, d);
                ub::Ev Et = ub::traced<double>(P, mesh, mi, 4, f, o, d, nullptr), Er = ub::traced<double>(P, mesh, mi, 3, f, o, d, nullptr);
                REQUIRE((Et.nh >= 1 && Er.nh >= 1));
                const cd T = cpx(Et.xpr, 0, 0), R = cpx(Er.xpr, 0, 0);
                DiagSheet W = diag_cm_stack(U, U, {{cd(5.24, 0.0), U, thk}}, U, U, std::cos(th), f, false, true);
                const double d_cm = (tf == 0.0) ? std::max(std::abs(T - W.T_te), std::abs(R - W.R_te)) : 0.0, sum = std::norm(T) + std::norm(R);
                QD_DIAG("%g,%.1f,%d,%.3f,%+.4f,%.3e,%.9f\n", tf, thd, N, thk * 1e3, ub::dB(T), d_cm, sum);
                INFO("E, tf " << tf << ", theta " << thd << ", N " << N);
                CHECK(d_cm < 1.0e-9);
                CHECK(std::abs(sum - 1.0) < 1.0e-9); // a lossless sheet between lossless media, at a resonance as anywhere else
                CHECK(ub::dB(T) > -1.0e-6);          // and at the resonance it passes everything
            }

    // F: ray_mesh_interact's own result for an ordered stack is the redo's, however the two normals are turned
    QD_DIAG("\n# diag: one solver F, an ordered stack in air, 6 mm eps 5.24-0.63j and 6 mm eps 1.50-0.40j, the four orientations of the two\n");
    QD_DIAG("# normals. ot: the hit code; own_dB, redo_dB: ray_mesh_interact's coefficient and the redo's; d_own: largest entry of their difference\n");
    QD_DIAG("normals,type,theta,ot,own_dB,redo_dB,d_own\n");
    {
        ub::Pal P = lump_palette({{5.24, 0.63, 0.006}, {1.50, 0.40, 0.006}});
        const char *nm[4] = {"front|back", "front|front", "back|back", "back|front"};
        const bool fl[4][2] = {{false, true}, {false, false}, {true, true}, {true, false}};
        for (int q = 0; q < 4; ++q)
            for (int it : {4, 3, 1, 0})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, G}, {1, 2}, {fl[q][0], fl[q][1]}, mi), o, d;
                    diag_ray_at(thd, o, d);
                    ub::Ev E = ub::traced<double>(P, mesh, mi, it, f, o, d, nullptr);
                    REQUIRE(E.nh == 2);
                    const double d_own = arma::abs(E.xpr - E.xpr0).max();
                    QD_DIAG("%s,%d,%.0f,%d,%.4f,%.4f,%.3e\n", nm[q], it, thd, (int)E.ot, ub::dB(cpx(E.xpr0, 0, 0)), ub::dB(cpx(E.xpr, 0, 0)), d_own);
                    INFO("F, normals " << nm[q] << ", type " << it << ", theta " << thd);
                    CHECK(d_own < 1.0e-9);
                    CHECK((E.ot & 0x20) == 0); // a stack has one medium on both sides and is never closed
                }
    }

    // G: an ordered stack inside a host. The mirror of the host's series is both layers, ahead of the leg in front and behind the leg behind
    QD_DIAG("\n# diag: one solver G, air | host 20 mm | 6 mm eps 5.24-0.63j, 6 mm eps 2.50 | host 30 mm | air, scalar. S: the host's series\n");
    QD_DIAG("# on the leg in front of the stack, eps 0 over eps 1.5; d_2: against the series whose mirror ahead is both layers backed by the\n");
    QD_DIAG("# host; d_1: against the first layer alone. buf: the buffer word behind the stack, the mark with the layer the previous-medium\n");
    QD_DIAG("# word does not name. b_2, b_1: the same series on the leg behind, against both layers and against the last one alone. dT, dR:\n");
    QD_DIAG("# every path, every series summed, the two ports over the matrix of the wall, dB\n");
    QD_DIAG("host,theta,abs_S,d_2,d_1,buf,b_2,b_1,nT,dT,dR\n");
    for (const auto &h : std::vector<std::pair<double, double>>{{1.5, 0.4}, {2.5, 0.8}})
        for (double thd : {10.0, 30.0, 60.0})
        {
            const double d1 = 0.020, d2 = 0.030, t1 = 0.006, t2 = 0.006, th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            ub::Pal P = lump_palette({{h.first, h.second, 0.0}, {5.24, 0.63, t1}, {2.50, 0.0, t2}});
            const cd eH(h.first, -h.second), e1(5.24, -0.63), e2(2.50, 0.0), cH = std::sqrt(1.0 - s2 / eH), bH = std::sqrt(eH) * cH;
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, d1, d1 + G, d1 + G + d2}, {1, 2, 3, 1}, {false, false, true, true}, mi);
            auto a0 = ub::chain(P, mesh, mi, {4, 4}, f, thd, {1.5, 0.0}), a1 = ub::chain(P, mesh, mi, {4, 4}, f, thd, {1.5, 1.5});
            REQUIRE((a0.size() == 2 && a1.size() == 2));
            const cd S = cpx(a0[1].xpr, 0, 0) / cpx(a1[1].xpr, 0, 0);
            const double L = d1 - a0[0].fbs(0, 0);
            const cd ph = std::exp(-J * k0 * bH * L), r_b = (bH - std::cos(th)) / (bH + std::cos(th));
            const cd r_2 = diag_cm_stack(eH, U, {{e1, U, t1}, {e2, U, t2}}, eH, U, cH, f, false, true).R_te;
            const cd r_1 = diag_cm_stack(eH, U, {{e1, U, t1}}, eH, U, cH, f, false, true).R_te;
            const cd S2 = 1.0 / (1.0 - r_b * r_2 * ph * ph), S1 = 1.0 / (1.0 - r_b * r_1 * ph * ph);
            // the leg behind the stack
            auto b0 = ub::chain(P, mesh, mi, {4, 4, 4}, f, thd, {1.5, 1.5, 0.0}), b1 = ub::chain(P, mesh, mi, {4, 4, 4}, f, thd, {1.5, 1.5, 1.5});
            REQUIRE((b0.size() == 3 && b1.size() == 3));
            const cd Sb = cpx(b0[2].xpr, 0, 0) / cpx(b1[2].xpr, 0, 0), phb = std::exp(-J * k0 * bH * (d1 + G + d2 - b0[1].sbs(0, 0))); // from the SBS the pair relaunches at
            const cd q_2 = diag_cm_stack(eH, U, {{e2, U, t2}, {e1, U, t1}}, eH, U, cH, f, false, true).R_te;
            const cd q_1 = diag_cm_stack(eH, U, {{e2, U, t2}}, eH, U, cH, f, false, true).R_te;
            const cd Sb2 = 1.0 / (1.0 - r_b * q_2 * phb * phb), Sb1 = 1.0 / (1.0 - r_b * q_1 * phb * phb);
            // every path
            arma::mat o, d;
            diag_ray_at(thd, o, d);
            bf::Acc A;
            bf::walk(P, mesh, mi, 4, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 26, A);
            const double tS = t1 + t2;
            DiagSheet W = diag_cm_stack(U, U, {{eH, U, d1 - 0.5 * tS}, {e1, U, t1}, {e2, U, t2}, {eH, U, d2 - 0.5 * tS}}, U, U, std::cos(th), f, false, true);
            const double dT = dBp(std::norm(A.T[0])) - dBp(std::norm(W.T_te)), dR = dBp(std::norm(A.R[0])) - dBp(std::norm(W.R_te));
            QD_DIAG("%g-%gj,%.0f,%.6f,%.3e,%.3e,%04x,%.3e,%.3e,%d,%+.4f,%+.4f\n", h.first, h.second, thd, std::abs(S), std::abs(S - S2), std::abs(S - S1),
                    ub::w(a0[1].buf(0)), std::abs(Sb - Sb2), std::abs(Sb - Sb1), A.nT, dT, dR);
            INFO("G, host " << h.first << "-" << h.second << "j, theta " << thd);
            CHECK(std::abs(S - S2) < 1.0e-6);           // L is the leg from the hit point
            CHECK(std::abs(S - S1) > 1.0e-4);           // and the two mirrors are told apart
            CHECK(a0[1].buf(0) == (short)(0x4000 | 2)); // the previous-medium word names the last layer, the buffer word the first
            CHECK(std::abs(Sb - Sb2) < 1.0e-6);         // so the leg behind sums against both
            CHECK(std::abs(Sb - Sb1) > 1.0e-4);
            CHECK(A.nT <= 8);             // and its bounces are no rays
            CHECK(std::abs(dT) < 5.0e-3); // four layers on the characteristic matrix, as a sum of paths
            CHECK(std::abs(dR) < 1.0e-2);
        }

    // H: a forced reflection at an exact match is one face from its two sides, also at a real angle on both
    QD_DIAG("\n# diag: one solver H, the face eps 5.24 | eps 5.24 with tf -0.5 from its two sides, ray_mesh_interact alone, type 3, 5 to 85 deg\n");
    QD_DIAG("# in steps of 0.5 deg. opposite: angles at which the two reflections are each other's negative; same: at which they are equal\n");
    QD_DIAG("angles,opposite,same,largest |r1 + r2|\n");
    {
        ub::Pal P = lump_palette({{5.24, 0.0, 0.0}, {5.24, 0.0, 0.0, 0.0, -0.5}});
        int n = 0, opp = 0, same = 0;
        double worst = 0.0;
        for (double thd = 5.0; thd < 85.25; thd += 0.5)
        {
            arma::uvec m12, m21;
            arma::mat mesh12 = ub::build({0.0, G}, {1, 2}, {true, false}, m12), mesh21 = ub::build({0.0, G}, {2, 1}, {true, false}, m21), o, d;
            diag_ray_at(thd, o, d);
            ub::Ev E1 = ub::traced<double>(P, mesh12, m12, 3, f, o, d, nullptr), E2 = ub::traced<double>(P, mesh21, m21, 3, f, o, d, nullptr);
            const cd r1 = cpx(E1.xpr0, 0, 0), r2 = cpx(E2.xpr0, 0, 0);
            ++n, opp += std::abs(r1 + r2) < 1.0e-9, same += std::abs(r1 - r2) < 1.0e-9;
            worst = std::max(worst, std::abs(r1 + r2));
            INFO("H, theta " << thd);
            CHECK(std::abs(std::norm(r1) - 0.5) < 1.0e-9); // a forced reflection at a matched face is |tf|
        }
        QD_DIAG("%d,%d,%d,%.3e\n", n, opp, same, worst);
        CHECK(opp == n);
    }
}

TEST_CASE("Budget - a sheet or a stack inside a host is one wall on every port whether it is capped or not", "[wave]")
{
    ub::OneThread one_thread;
    // The cap per side is a placement: the layer between what is left of the host on either side.
    // A wall of lossless media then returns what it received, whatever the sheet's thickness is,
    // and a sheet or a stack that fits is the wall air | host Lf - h | layers | host Lb - h | air.
    // Every path, every series summed, scalar, both tracers, 20 and 60 deg.
    const double f = FRQ;
    struct Geo
    {
        double Lf, Lb, tot;
    };
    auto walls = [&](const char *tag, const std::vector<LumpMtl> &lp, const std::vector<Geo> &geo, bool lossless, double bound)
    {
        const int nlay = (int)lp.size() - 1;
        const std::complex<double> eH = lump_eps(lp[0], f);
        for (const Geo &g : geo)
            for (int tr : {4, 5})
                for (double thd : {20.0, 60.0})
                {
                    std::vector<LumpMtl> q = lp;
                    std::vector<DiagLayer> wall(1);
                    const double h = 0.5 * g.tot, th = thd * QPI / 180.0;
                    wall[0].e = eH, wall[0].d = std::max(0.0, g.Lf - h);
                    for (int k = 1; k <= nlay; ++k)
                        q[k].thk = g.tot * ((nlay == 1) ? 1.0 : ((k == 1) ? 0.6 : 0.4)), wall.push_back(lump_layer(q[k], f));
                    wall.push_back(wall[0]), wall.back().d = std::max(0.0, g.Lb - h);
                    ub::Pal P = lump_palette(q);
                    arma::uvec mi;
                    arma::mat mesh = (nlay == 1) ? ub::build({0.0, g.Lf, g.Lf + g.Lb}, {1, 2, 1}, {false, false, true}, mi)
                                                 : ub::build({0.0, g.Lf, g.Lf + 1.0e-5, g.Lf + g.Lb + 1.0e-5}, {1, 2, 3, 1}, {false, false, true, true}, mi);
                    arma::mat o, d;
                    diag_ray_at(thd, o, d);
                    bf::Acc A;
                    bf::walk(P, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 26, A);
                    DiagSheet W = diag_cm_stack(1.0, 1.0, wall, 1.0, 1.0, std::cos(th), f, false, true);
                    const double pT = std::norm(A.T[0]), pR = std::norm(A.R[0]), sum = pT + pR;
                    const bool fits = h < g.Lf && h < g.Lb;
                    const double dT = 10.0 * std::log10(pT / std::norm(W.T_te)), dR = 10.0 * std::log10(pR / std::norm(W.R_te));
                    QD_DIAG("%s,%d,%.0f+%.0f,%.0f,%d,%.0f,%d,%d,%.6f,%+.4f,%+.4f,%.1e\n", tag, nlay, 1e3 * g.Lf, 1e3 * g.Lb, 1e3 * g.tot, tr, thd, (int)fits, A.nT, sum,
                            fits ? dT : 0.0, fits ? dR : 0.0, A.drop);
                    INFO("walls " << tag << " layers " << nlay << " Lf " << g.Lf << " Lb " << g.Lb << " tot " << g.tot << " tracer " << tr << " theta " << thd);
                    CHECK(std::isfinite(sum));
                    if (bound > 0.0)
                        CHECK(sum < bound);
                    if (lossless)
                        CHECK(sum > 0.999);
                    if (fits && bound > 0.0)
                    {
                        CHECK(std::abs(dT) < 5.0e-3);
                        CHECK(std::abs(dR) < 1.0e-2);
                    }
                    if (bound > 0.0)
                        CHECK(A.nT < 40); // the leg behind a stack is summed, and no path is cut
                }
    };
    QD_DIAG("\n# diag: one wall, a sheet (1 layer) or an ordered stack (2 layers, 60 %% and 40 %%) of thickness tot inside a host, front + back mm.\n");
    QD_DIAG("# lossless: eps 5.24 | eps 2.50 in eps 1.50. offcentre: a sheet of eps 1.50 in eps 1.50-0.40j that does not fit. long: recorded,\n");
    QD_DIAG("# eps 1.50 | eps 1.60 in eps 1.50-0.40j, a stack thicker than its whole host met from the long side, which nothing in the event\n");
    QD_DIAG("# can know. sum: both ports, every path. dT, dR: against the wall where the layers fit, dB. drop: power the walk cut\n");
    QD_DIAG("family,layers,host_mm,tot_mm,tracer,theta,fits,nT,sum,dT,dR,drop\n");
    const std::vector<Geo> g1 = {{0.05, 0.05, 0.012}, {0.005, 0.005, 0.005}, {0.01, 0.09, 0.012}, {0.09, 0.01, 0.012}, {0.005, 0.005, 0.009}, {0.01, 0.09, 0.030}, {0.09, 0.01, 0.030}, {0.005, 0.005, 0.015}, {0.01, 0.09, 0.120}};
    walls("lossless", {{1.50, 0.0, 0.0}, {5.24, 0.0, 0.001}, {2.50, 0.0, 0.001}}, g1, true, 1.0 + 1.0e-4);
    walls("lossless", {{1.50, 0.0, 0.0}, {5.24, 0.0, 0.001}}, g1, true, 1.0 + 1.0e-4);
    walls("offcentre", {{1.50, 0.40, 0.0}, {1.50, 0.0, 0.001}}, {{0.09, 0.01, 0.090}, {0.09, 0.01, 0.110}, {0.01, 0.09, 0.140}}, false, 1.0);
    walls("long", {{1.50, 0.40, 0.0}, {1.50, 0.0, 0.001}, {1.60, 0.0, 0.001}}, {{0.09, 0.01, 0.110}, {0.09, 0.01, 0.140}}, false, 0.0);
}

TEST_CASE("Resonant medium - a permittivity resonance is part of the medium a wave crosses and travels in", "[wave]")
{
    ub::OneThread one_thread;
    // A Lorentz pole in the permittivity is a dispersive dielectric: one medium of eps + res at its
    // faces and inside it. A sheet of eps 4 whose pole is -3 is a sheet of air, a lossy pole is the
    // same medium as a conductivity of that size, a wall of lossless media with a pole in it returns
    // what it received, and a resonant sheet is its traced solid. Every path through
    // ray_triangle_intersect, ray_mesh_interact and ray_state_update (bf::walk), types 4, 5, 1, 2,
    // 1 GHz. The pole is written through resF, resQ, resS: at 1 GHz resF 2 and resQ 1e300 give the
    // real value 4 resS / 3, resF 1 gives -j resS resQ.
    const double f = FRQ, k = 2.0 * QPI * f / C0;
    auto mtl = [](double a, double ei, double rF, double rQ, double rS)
    {
        LumpMtl M;
        M.a = a, M.ei = ei, M.resF = rF, M.resQ = rQ, M.resS = rS;
        return M;
    };
    const LumpMtl MATCH = mtl(4.0, 0.0, 2.0, 1.0e300, -2.25), DOWN = mtl(4.0, 0.0, 2.0, 1.0e300, -1.5), UP = mtl(4.0, 0.0, 2.0, 1.0e300, 0.93);
    const LumpMtl LOSSY = mtl(4.0, 0.0, 1.0, 5.0, 0.5), TWIN = mtl(4.0, 2.5, 0.0, 0.0, 0.0), PLAIN = mtl(4.0, 0.0, 0.0, 0.0, 0.0);
    const LumpMtl STRONG = mtl(2.5, 0.0, 0.8, 30.0, 2.0), HOST = mtl(2.5, 0.0, 0.0, 0.0, 0.0), SHEET = mtl(5.24, 0.0, 0.0, 0.0, 0.0);
    const LumpMtl AIR = mtl(1.0, 0.0, 0.0, 0.0, 0.0), E2 = mtl(2.0, 0.0, 0.0, 0.0, 0.0); // the media MATCH, DOWN and UP are; UP is SHEET

    // kind 0: a sheet of M, d thick, in air; 1: a traced solid of M from 0 to d; 2: a sheet of M, d thick, in the
    // middle of 100 mm of H. Series summed (eps 0) or every bounce followed (eps 1.5).
    auto walk = [&](const LumpMtl &M, double d, int kind, const LumpMtl &H, int tr, double thd, double eps, int depth)
    {
        std::vector<LumpMtl> lp = {M, H};
        lp[0].thk = (kind == 1) ? 0.0 : d, lp[1].thk = 0.0;
        ub::Pal P = lump_palette(lp);
        arma::uvec mi;
        arma::mat mesh = (kind == 0) ? ub::build({0.0}, {1}, {false}, mi)
                                     : ((kind == 1) ? ub::build({0.0, d}, {1, 1}, {false, true}, mi)
                                                    : ub::build({0.0, 0.05, 0.10}, {2, 1, 2}, {false, false, true}, mi));
        arma::mat o, dd;
        diag_ray_at(thd, o, dd);
        bf::Acc A;
        bf::walk(P, mesh, mi, tr, f, thd, eps, nullptr, o, dd, 1.0, 1.0, 0.0, 0.0, 0, depth, A);
        return A;
    };
    auto sum = [](const bf::Acc &A, int ch)
    { return std::norm(A.T[ch]) + std::norm(A.R[ch]); };
    const int types[4] = {4, 5, 1, 2};

    // A. The sheet whose pole takes eps 4 to 1 is a sheet of air: no reflection, and a transmission of one with no phase, at
    // every angle. Its traced solid is 6 or 37.5 mm of air: exp(-j k d cos) on the thickness axis.
    QD_DIAG("\n# diag: resonant medium A, eps 4 with a pole of -3, a sheet and its traced solid in air. dT: |T - 1| of the sheet, |T / exp(-j k d cos) - 1|\n");
    QD_DIAG("# of the solid summed and followed; arg: the sheet's phase, deg; R: |R|. Worst over both channels\n");
    QD_DIAG("mm,type,theta,dT_sheet,arg_sheet,R_sheet,dT_solid,dT_followed,R_solid\n");
    for (double d : {0.006, 0.0375})
        for (int tr : types)
            for (double thd : {0.0, 30.0, 60.0, 85.0})
            {
                const std::complex<double> ph = std::exp(std::complex<double>(0.0, -k * d * std::cos(thd * QPI / 180.0)));
                bf::Acc S = walk(MATCH, d, 0, HOST, tr, thd, 0.0, 60), X = walk(MATCH, d, 1, HOST, tr, thd, 0.0, 60), F = walk(MATCH, d, 1, HOST, tr, thd, 1.5, 60);
                double dS = 0.0, rS = 0.0, dX = 0.0, dF = 0.0, rX = 0.0, aS = 0.0;
                for (int ch = 0; ch < 2; ++ch)
                {
                    dS = std::max(dS, std::abs(S.T[ch] - 1.0)), rS = std::max(rS, std::abs(S.R[ch])), aS = std::max(aS, std::abs(ub::deg(S.T[ch])));
                    dX = std::max(dX, std::abs(X.T[ch] / ph - 1.0)), dF = std::max(dF, std::abs(F.T[ch] / ph - 1.0)), rX = std::max(rX, std::abs(X.R[ch]));
                }
                QD_DIAG("%.1f,%d,%.0f,%.3e,%.6f,%.3e,%.3e,%.3e,%.3e\n", 1e3 * d, tr, thd, dS, aS, rS, dX, dF, rX);
                INFO("A matched sheet " << d << " type " << tr << " theta " << thd);
                CHECK(dS < 1.0e-12);
                CHECK(rS < 1.0e-12);
                CHECK(dX < 1.0e-12);
                CHECK(dF < 1.0e-12);
                CHECK(rX < 1.0e-12);
            }

    // B. A medium with a pole is the medium eps + pole: eps 4 with a pole of -2.5j against eps 4 - 2.5j from the
    // conductivity column, and the lossless poles against eps 1, 2 and 5.24. Complex, both ports, both channels. As a sheet,
    // a traced solid summed and followed, a resonant sheet inside a plain host and a plain sheet inside a resonant host.
    QD_DIAG("\n# diag: resonant medium B, a medium with a pole against its twin without one, every path. d: largest complex distance of T\n");
    QD_DIAG("# and R over both channels; T, R: the twin's ports in dB, channel 0\n");
    QD_DIAG("medium,config,mm,type,theta,mode,d,T_dB,R_dB\n");
    struct Cfg
    {
        const char *name;
        int kind;
        bool res_in_host;
    };
    struct Pair
    {
        const char *name;
        LumpMtl M, T;
    };
    for (const Pair &pr : {Pair{"lossy", LOSSY, TWIN}, Pair{"match", MATCH, AIR}, Pair{"down", DOWN, E2}, Pair{"up", UP, SHEET}})
        for (const Cfg &c : {Cfg{"sheet", 0, false}, Cfg{"solid", 1, false}, Cfg{"sheet in host", 2, false}, Cfg{"in resonant host", 2, true}})
            for (double d : {0.006, 0.0375})
                for (int tr : types)
                    for (double thd : {0.0, 30.0, 60.0, 85.0})
                        for (double eps : {0.0, 1.5})
                        {
                            if (eps > 0.0 && c.kind != 1)
                                continue;
                            bf::Acc A = c.res_in_host ? walk(SHEET, d, 2, pr.M, tr, thd, eps, 26) : walk(pr.M, d, c.kind, HOST, tr, thd, eps, 60);
                            bf::Acc B = c.res_in_host ? walk(SHEET, d, 2, pr.T, tr, thd, eps, 26) : walk(pr.T, d, c.kind, HOST, tr, thd, eps, 60);
                            double dd = 0.0;
                            for (int ch = 0; ch < 2; ++ch)
                                dd = std::max(dd, std::max(std::abs(A.T[ch] - B.T[ch]), std::abs(A.R[ch] - B.R[ch])));
                            QD_DIAG("%s,%s,%.1f,%d,%.0f,%s,%.3e,%.4f,%.4f\n", pr.name, c.name, 1e3 * d, tr, thd, eps > 0.0 ? "fol" : "sum", dd, ub::dB(B.T[0]),
                                    ub::dB(B.R[0]));
                            INFO("B " << pr.name << " " << c.name << " " << d << " type " << tr << " theta " << thd << " eps " << eps);
                            CHECK(dd < 1.0e-9);
                        }

    // C. Both ports together. A wall of lossless media with a pole in it returns what it received; one with a lossy pole
    // no more than that, the strong pole that takes eps 2.5 to -1.04 - 0.26j included. The plain row is the walker's own
    // residual on a lossless wall without a pole.
    QD_DIAG("\n# diag: resonant medium C, both ports of a wall together, every path, worst channel. kind: sheet, solid, a resonant sheet in\n");
    QD_DIAG("# 100 mm of eps 2.50, and a sheet of eps 5.24 in 100 mm of the resonant medium\n");
    QD_DIAG("medium,config,mm,type,theta,mode,sum\n");
    struct Med
    {
        const char *name;
        LumpMtl M;
        bool lossless;
    };
    for (const Med &m : {Med{"plain", PLAIN, true}, Med{"down", DOWN, true}, Med{"up", UP, true}, Med{"match", MATCH, true}, Med{"lossy", LOSSY, false},
                         Med{"strong", STRONG, false}})
        for (const Cfg &c : {Cfg{"sheet", 0, false}, Cfg{"solid", 1, false}, Cfg{"sheet in host", 2, false}, Cfg{"in resonant host", 2, true}})
            for (double d : {0.006, 0.05})
                for (int tr : types)
                    for (double thd : {0.0, 30.0, 60.0, 85.0})
                        for (double eps : {0.0, 1.5})
                        {
                            if ((eps > 0.0 && c.kind != 1) || (c.kind == 2 && thd > 60.0))
                                continue;
                            bf::Acc A = c.res_in_host ? walk(SHEET, d, 2, m.M, tr, thd, eps, 120) : walk(m.M, d, c.kind, HOST, tr, thd, eps, 120);
                            const double s = std::max(sum(A, 0), sum(A, 1));
                            QD_DIAG("%s,%s,%.0f,%d,%.0f,%s,%.6f\n", m.name, c.name, 1e3 * d, tr, thd, eps > 0.0 ? "fol" : "sum", s);
                            INFO("C " << m.name << " " << c.name << " " << d << " type " << tr << " theta " << thd << " eps " << eps);
                            CHECK(std::isfinite(s));
                            CHECK(s < (m.lossless ? 1.0 + 2.5e-4 : 1.0));
                            if (m.lossless)
                                CHECK(s > 1.0 - 2.5e-4);
                        }

    // D. A resonant sheet is its traced solid, level and phase, both ports, and a followed series is the summed one. The
    // plain rows are the tracer's own residual on media without a pole: eps 4, and eps 2 and 5.24, which DOWN and UP are.
    QD_DIAG("\n# diag: resonant medium D, a sheet against its traced solid in air, worst channel. T, R: the solid against the sheet\n");
    QD_DIAG("# moved by exp(-j k d cos), dB and deg; fol: the followed solid against the summed one, dB and deg\n");
    QD_DIAG("medium,mm,type,theta,T_dB,T_deg,R_dB,R_deg,fol_dB,fol_deg\n");
    for (const Med &m : {Med{"plain", PLAIN, true}, Med{"eps 2", E2, true}, Med{"eps 5.24", SHEET, true}, Med{"down", DOWN, true}, Med{"up", UP, true},
                         Med{"lossy", LOSSY, false}})
        for (double d : {0.006, 0.0375})
            for (int tr : types)
                for (double thd : {0.0, 30.0, 60.0, 85.0})
                {
                    const std::complex<double> ph = std::exp(std::complex<double>(0.0, -k * d * std::cos(thd * QPI / 180.0)));
                    bf::Acc S = walk(m.M, d, 0, HOST, tr, thd, 0.0, 60), X = walk(m.M, d, 1, HOST, tr, thd, 0.0, 60), F = walk(m.M, d, 1, HOST, tr, thd, 1.5, 60);
                    double w[6] = {0, 0, 0, 0, 0, 0};
                    for (int ch = 0; ch < 2; ++ch)
                    {
                        std::complex<double> qT = X.T[ch] / (S.T[ch] * ph), qR = X.R[ch] / (S.R[ch] * ph), qF = F.T[ch] / X.T[ch];
                        const double v[6] = {ub::dB(qT), ub::deg(qT), ub::dB(qR), ub::deg(qR), ub::dB(qF), ub::deg(qF)};
                        for (int i = 0; i < 6; ++i)
                            w[i] = (std::abs(v[i]) > std::abs(w[i])) ? v[i] : w[i];
                    }
                    QD_DIAG("%s,%.1f,%d,%.0f,%+.5f,%+.4f,%+.5f,%+.4f,%+.5f,%+.4f\n", m.name, 1e3 * d, tr, thd, w[0], w[1], w[2], w[3], w[4], w[5]);
                    INFO("D " << m.name << " " << d << " type " << tr << " theta " << thd);
                    CHECK(std::abs(w[0]) < 2.0e-2);
                    CHECK(std::abs(w[1]) < 5.0e-2);
                    CHECK(std::abs(w[2]) < 1.0e-2);
                    CHECK(std::abs(w[3]) < 5.0e-2);
                    CHECK(std::abs(w[4]) < 6.0e-2);
                    CHECK(std::abs(w[5]) < 1.0e-1);
                }
}

TEST_CASE("Unentered branches - ray_mesh_interact's own glued composite against the redo", "[wave]")
{
    ub::OneThread one_thread;
    // What ray_mesh_interact writes for a glued pair serves a caller without state; with the state
    // kept the redo replaces it. From air the two are handed the same media and the same cosine,
    // so they have to be one number: under a closed outer pair on the transmit port, for a sheet
    // rarer than the invariant and for one denser, and with att, tf and a
    // resonance on the sheet and on the wall behind it. A glued exit is reached with a seeded
    // state, a ray launched inside a lossless solid; there the redo also closes the host leg, a
    // phase common to every entry. A tube with one edge past the critical angle is the last table.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    const double f = FRQ, gap = 1.0e-5;
    const std::vector<LumpMtl> lp = {
        {2.1609e-8, 0.0, 0.022, 3.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 wood panel 22 mm, att 3 dB, tf 0.2, mass law
        {2.401e-9, 0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},    // 2 concrete, att 2 dB, tf 0.1, mass law
        {0.31, 0.0, 0.010, 0.0, 0.2},                                       // 3 lossless sheet rarer than air, tf 0.2
        {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                     // 4 rigid wall, tf 0.3
        {5.24, 0.63, 0.006, 3.0, 0.3, 2.0, 5.0, 0.5},                       // 5 sheet with att, tf and a resonance
        {1.50, 0.40, 0.0, 2.0, -0.2},                                       // 6 solid with att and a forcing tf
        {5.24, 0.00, 0.0},                                                  // 7 lossless dense solid
        {1.50, 0.40, 0.006, 0.0, 0.5},                                      // 8 coating 6 mm, tf 0.5
        {2.50, 0.80, 0.010},                                                // 9 lossy sheet 10 mm
        {0.50, 0.00, 0.0},                                                  // 10 lossless solid rarer than air: critical angle 45 deg
    };
    ub::Pal pal = lump_palette(lp);

    QD_DIAG("\n# diag: ray_mesh_interact's own glued entry against the redo, from air. own: largest distance of an xprmat entry, complex.\n");
    QD_DIAG("# ref: the scalar port against diag_leak_series (closed pair, ordinary crossing) or diag_sheet_composite with its lumped\n");
    QD_DIAG("# columns (open pair); -1 where neither applies\n");
    QD_DIAG("stack,type,theta,ot,rt,lib_dB,own,ref\n");
    struct St
    {
        const char *name;
        int sheet, wall;
    };
    for (const St &S : {St{"eps 0.31 tf 0.2 | rigid tf 0.3", 3, 4}, St{"lumped sheet | lumped solid", 5, 6}})
        for (int it : {4, 5, 3, 1, 0})
            for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
            {
                const double th = thd * QPI / 180.0;
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0, gap}, {S.sheet, S.wall}, {false, false}, mi);
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                ub::Opt op;
                op.relaunch = it != 3 && it != 0;
                ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, nullptr, op);
                REQUIRE(E.nh == 2);
                const double own = arma::abs(E.xpr - E.xpr0).max();
                const LumpMtl &Ms = lp[S.sheet - 1], &Mw = lp[S.wall - 1];
                const std::complex<double> eS = lump_eps(Ms, f), eW = lump_eps(Mw, f), lib = cpx(E.xpr, 0, 0);
                double ref = -1.0;
                if (it >= 4 && S.sheet == 3 && (E.ot & 0x20) != 0 && std::sin(th) * std::sin(th) < 0.31) // closed pair, an ordinary crossing into the sheet
                    ref = std::abs(lib - diag_leak_series(1.0, eS, eW, 0.0, Ms.tf, Mw.tf, Ms.thk, std::cos(th), f, true).T_te);
                if (it >= 3 && S.sheet == 5)
                {
                    DiagLump Lp;
                    Lp.g_slab = std::pow(10.0, -0.1 * Ms.att), Lp.g_far = std::pow(10.0, -0.1 * Mw.att);
                    Lp.tf = Ms.tf, Lp.tf_far = Mw.tf, Lp.res_slab = lump_res(Ms, f);
                    DiagSheet W = diag_sheet_composite(1.0, eS, eW, Ms.thk, std::cos(th), f, false, true, 1.0, 1.0, 1.0, Lp);
                    ref = std::abs(lib - ((it == 3) ? W.R_te : W.T_te));
                }
                QD_DIAG("%s,%d,%.0f,%d,%d,%.4f,%.3e,%.3e\n", S.name, it, thd, (int)E.ot, (int)E.rt, ub::dB(lib), own, ref);
                INFO("own glued entry " << S.name << " type " << it << " theta " << thd);
                CHECK(own < 1.0e-12);
                if (ref >= 0.0)
                    CHECK(ref < 1.0e-9);
                if (S.sheet != 5 && thd > 0.0)
                    CHECK((E.ot & 0x20) != 0); // the outer pair is closed on these rows
            }

    QD_DIAG("\n# diag: ray_mesh_interact's own glued exit, eps 5.24 | 6 mm eps 1.50-0.40j tf 0.5 | air, EM, ray launched in the solid;\n");
    QD_DIAG("# critical angle of the outer pair 25.9 deg. mag: largest distance of an entry's magnitude; leg: the redo over the own\n");
    QD_DIAG("# coefficient, HH against VV, which is the host leg the redo closes and has to be one number\n");
    QD_DIAG("type,theta,ot,rt,VV_dB,HH_dB,mag,leg\n");
    for (int it : {1, 2, 0})
        for (double thd : {15.0, 22.0, 30.0, 45.0})
        {
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, gap}, {7, 8}, {true, false}, mi);
            arma::mat o, d;
            diag_ray_at(thd, o, d);
            ub::Ev seed;
            seed.prev = sv({0}), seed.cur = sv({7}), seed.buf = sv({0});
            seed.path_dirN = (d - o) / std::sqrt(ub::sumsq(d - o));
            seed.accg = arma::vec{0.0}, seed.accr = arma::vec{0.0};
            ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, &seed);
            REQUIRE(E.nh == 2);
            const std::complex<double> v0 = cpx(E.xpr0, 0, 0), v1 = cpx(E.xpr, 0, 0), h0 = cpx(E.xpr0, 0, 3), h1 = cpx(E.xpr, 0, 3);
            double mag = 0.0;
            for (int q = 0; q < 4; ++q)
                mag = std::max(mag, std::abs(std::abs(cpx(E.xpr, 0, q)) - std::abs(cpx(E.xpr0, 0, q))));
            const double leg = (std::abs(v0) > 1.0e-12 && std::abs(h0) > 1.0e-12) ? std::abs(v1 / v0 - h1 / h0) : 0.0;
            QD_DIAG("%d,%.0f,%d,%d,%.4f,%.4f,%.3e,%.3e\n", it, thd, (int)E.ot, (int)E.rt, ub::dB(v1), ub::dB(h1), mag, leg);
            INFO("own glued exit type " << it << " theta " << thd);
            CHECK(mag < 1.0e-12);
            CHECK(leg < 1.0e-12);
        }

    QD_DIAG("\n# diag: a ray tube with one edge past the critical angle of the outer pair, air | 10 mm eps 2.50-0.80j | eps 0.50, 45 deg,\n");
    QD_DIAG("# type 5, tube 0.6 deg wide. own_dB: ray_mesh_interact's transmit port; redo_dB: after ray_state_update; t4: the redo\n");
    QD_DIAG("# against the undeviated tracer's event without a tube, largest distance of an entry; seg, dir: angle of the relaunched\n");
    QD_DIAG("# segment and of the written direction, deg\n");
    QD_DIAG("theta,ot,ot_no_tube,rt,rt_4,own_dB,redo_dB,t4,seg,dir,dir_4,cur,cur_4\n");
    auto ang = [](const arma::mat &v)
    { return std::atan2(std::sqrt(v(0, 1) * v(0, 1) + v(0, 2) * v(0, 2)), std::abs(v(0, 0))) * 180.0 / QPI; };
    for (double thd : {43.0, 44.6, 46.0})
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, gap}, {9, 10}, {false, false}, mi);
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        auto dirv = [](double a, double z)
        {
            arma::rowvec v = {std::cos(a * QPI / 180.0), std::sin(a * QPI / 180.0), z};
            return arma::rowvec(v / std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)));
        };
        arma::mat trivec(1, 9), tridir(1, 9);
        trivec.row(0) = arma::rowvec{0.0, 0.001, 0.0, 0.0, -0.001, 0.001, 0.0, -0.001, -0.001};
        tridir.row(0) = arma::join_rows(arma::join_rows(dirv(thd + 0.6, 0.0), dirv(thd - 0.6, 0.0)), dirv(thd, 0.01));
        ub::Opt tube, plain;
        tube.trivec = &trivec, tube.tridir = &tridir, tube.relaunch = true, plain.relaunch = true;
        ub::Ev E = ub::traced<double>(pal, mesh, mi, 5, f, o, d, nullptr, tube), N5 = ub::traced<double>(pal, mesh, mi, 5, f, o, d, nullptr, plain);
        ub::Ev E4 = ub::traced<double>(pal, mesh, mi, 4, f, o, d, nullptr);
        REQUIRE(E.nh == 2);
        const double t4 = arma::abs(E.xpr - E4.xpr).max();
        QD_DIAG("%.1f,%d,%d,%d,%d,%.4f,%.4f,%.3e,%.4f,%.4f,%.4f,%04x,%04x\n", thd, (int)E.ot, (int)N5.ot, (int)E.rt, (int)E4.rt, ub::dB(cpx(E.xpr0, 0, 0)),
                ub::dB(cpx(E.xpr, 0, 0)), t4, ang(E.destN - E.origN), ang(E.path_dirN), ang(E4.path_dirN), ub::w(E.cur(0)), ub::w(E4.cur(0)));
        INFO("tube edge, theta " << thd);
        // Whatever the tube did to the hit code, the redo is the undeviated tracer's event
        CHECK(ub::same_bits(E.xpr, E4.xpr));
        CHECK(E.rt == E4.rt);
        CHECK(std::abs(ang(E.path_dirN) - ang(E4.path_dirN)) < 1.0e-9);
        if (thd > 44.0 && thd < 45.0) // the spine is open and one edge is not
        {
            CHECK(((E.ot & 0x20) != 0 && (N5.ot & 0x20) == 0));     // the edge closes the whole tube
            CHECK(arma::abs(E.xpr - E.xpr0).max() < 1.0e-12);       // and a caller without state gets the spine's coefficient, as the redo does
            CHECK(std::norm(cpx(E.xpr, 0, 0)) > 0.1);               // where the redo has the composite of the open pair
            CHECK(E.cur(0) == E4.cur(0));                           // on a leg that is not flagged bent
            CHECK(std::abs(ang(E.destN - E.origN) - thd) < 1.0e-6); // because the segment was not bent
        }
        else
            CHECK(arma::abs(E.xpr - E.xpr0).max() < 1.0e-12);
    }
}

TEST_CASE("Stack geometry - an ordered stack leaves the ray on the line it arrived on", "[wave]")
{
    ub::OneThread one_thread;
    // Two co-located sheets of different material are one event with one medium on both sides.
    // The transverse wavenumber and the medium are what they were in front, so the ray goes on
    // along the line it arrived on, whatever the layers are, whichever comes first and however
    // the two normals are turned; each vertex ray of a tube keeps its direction. With types 2
    // and 5 ray_mesh_interact builds the segment and the tube, and with every type its own
    // path_dirN. Through ray_triangle_intersect, ray_mesh_interact and ray_state_update, double.
    //   C1, C2, L   coherent sheets, no tf and no mass column: the EM rows
    //   Q           a rigid sheet with tf and no mass column: types 1 and 2 on it are not EM
    //               materials (R and N, with the mass column, are in the mass-law part)
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    const double f = FRQ, gap = 1.0e-5;
    const std::vector<LumpMtl> lp = {
        {5.24, 0.63, 0.006},                                                // 1 C1
        {1.50, 0.40, 0.006},                                                // 2 C2
        {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 3 R
        {1.50, 0.0, 0.022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},      // 4 N
        {2.50, 0.0, 0.0},                                                   // 5 H, lossless host
        {2.50, 0.80, 0.0},                                                  // 6 Hl, lossy host
        {2.50, 0.0, 0.003},                                                 // 7 L
        {2.1609e-8, 0.0, 0.022, 0.0, 0.2},                                  // 8 Q
    };
    const ub::Pal pal = lump_palette(lp);
    const char *mn[] = {"air", "C1", "C2", "R", "N", "H", "Hl", "L", "Q"};
    const std::vector<std::pair<bool, bool>> flips = {{false, true}, {false, false}, {true, true}, {true, false}};
    const char *fn[] = {"front|back", "front|front", "back|back", "back|front"};
    auto unit = [](arma::mat d)
    { d /= std::sqrt(ub::sumsq(d)); return d; };
    auto dmax = [](const arma::mat &a, const arma::mat &b)
    { double v = 0.0; for (arma::uword i = 0; i < 3; ++i) v = std::max(v, std::abs(a(0, i) - b(0, i))); return v; };
    auto ang = [](const arma::mat &d)
    { return std::atan2(std::sqrt(d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2)), std::abs(d(0, 0))) * 180.0 / QPI; };
    auto sig = [](const arma::mat &x)
    { double v = 0.0; for (arma::uword i = 0; i < x.n_elem; ++i) v += 0.731 * (double)(i + 1) * x(i); return v; };
    auto fan = [](const arma::mat &u, arma::mat &tv, arma::mat &td) // three vertex rays about the spine
    {
        const double e[3][3] = {{0.0, 0.004, 0.0}, {0.0, -0.002, 0.0035}, {0.0, -0.002, -0.0035}};
        tv.zeros(1, 9), td.zeros(1, 9);
        for (int k = 0; k < 3; ++k)
        {
            double v[3] = {u(0, 0) + e[k][0], u(0, 1) + e[k][1], u(0, 2) + e[k][2]};
            const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            for (int c = 0; c < 3; ++c)
                td(0, 3 * k + c) = v[c] / n;
        }
    };
    const double TOL = 1.0e-9;

    // ---- A. In air, one event: every pair, every orientation of the two normals, every type ----
    QD_DIAG("\n# diag: stack geometry A, two co-located sheets in air, one event with a tube. seg, own: the segment and ray_mesh_interact's own\n");
    QD_DIAG("# path_dirN, deg from the normal. Largest component of a unit-vector difference: e_seg the segment against the direction\n");
    QD_DIAG("# written, e_in against the incoming direction, e_own the own path_dirN against the incoming direction, e_tube the vertex\n");
    QD_DIAG("# rays against the directions they arrived in. bits: origN, destN and tridirN are those of type 1 or 4, bitwise (recorded)\n");
    QD_DIAG("pair,normals,type,theta,ot,rt,seg,own,e_seg,e_in,e_own,e_tube,bits\n");
    for (auto pr : std::vector<std::pair<int, int>>{{1, 2}, {2, 1}, {7, 1}, {8, 1}})
        for (size_t q = 0; q < flips.size(); ++q)
            for (int it : {2, 5, 1, 4})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, gap}, {pr.first, pr.second}, {flips[q].first, flips[q].second}, mi), o, d, tv, td;
                    diag_ray_at(thd, o, d);
                    const arma::mat u = unit(d - o);
                    fan(u, tv, td);
                    ub::Opt op;
                    op.relaunch = true, op.trivec = &tv, op.tridir = &td;
                    const ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, nullptr, op);
                    const ub::Ev E1 = ub::traced<double>(pal, mesh, mi, (it == 2 || it == 1) ? 1 : 4, f, o, d, nullptr, op);
                    const arma::mat seg = unit(E.destN - E.origN);
                    double e_tube = 0.0;
                    for (int k = 0; k < 3; ++k)
                        e_tube = std::max(e_tube, dmax(E.tridirN.cols(3 * k, 3 * k + 2), td.cols(3 * k, 3 * k + 2)));
                    const double e_seg = dmax(seg, unit(E.path_dirN)), e_in = dmax(seg, u), e_own = dmax(unit(E.dir0), u);
                    const bool bits = ub::same_bits(E.origN, E1.origN) && ub::same_bits(E.destN, E1.destN) && ub::same_bits(E.tridirN, E1.tridirN);
                    QD_DIAG("%s|%s,%s,%d,%.0f,%u,%u,%.4f,%.4f,%.3e,%.3e,%.3e,%.3e,%d\n", mn[pr.first], mn[pr.second], fn[q], it, thd, (unsigned)E.ot, (unsigned)E.rt,
                            ang(seg), ang(E.dir0), e_seg, e_in, e_own, e_tube, (int)bits);
                    INFO("A, " << mn[pr.first] << "|" << mn[pr.second] << ", " << fn[q] << ", type " << it << ", theta " << thd);
                    CHECK(E.rt == 69);
                    CHECK(e_seg < TOL);  // the segment is the direction written
                    CHECK(e_in < TOL);   // which is the one the ray arrived in
                    CHECK(e_own < TOL);  // and what a caller without state is handed
                    CHECK(e_tube < TOL); // each vertex ray in the direction it arrived in
                }

    // ---- B. Inside a host, and where the next wall is met ----
    // Host 22 mm with the stack in its middle, entered across a bare face or through a glued
    // sheet L, and a 50 mm eps 2.50 wall 0.5 m behind. The twin has the first layer alone in
    // place of the stack: a single sheet, which leaves the ray where it was. A leg behind a
    // glued entry carries that entry's anchoring, so there the stack is held to the line of the
    // segment it was reached on, and to the direction written at the size of the anchoring.
    QD_DIAG("\n# diag: stack geometry B, the stack in the middle of 22 mm of host, a 50 mm eps 2.50 wall 0.5 m behind, five events (six behind\n");
    QD_DIAG("# a glued entry is not: the glued entry is one event). e_stk, e_stk_in: the segment behind the stack against the direction\n");
    QD_DIAG("# written and against the segment it was reached on; e_exit: the segment behind the host against the direction written;\n");
    QD_DIAG("# rt_exit, rt_twin: resolved type of that exit and of the twin's. dy_wall: where the wall is met against the twin with the first layer alone, m; e_wall:\n");
    QD_DIAG("# the segment inside the wall against the twin's; dL: the chain level against the undeviated tracer's, dB\n");
    QD_DIAG("host,entry,stack,normals,type,theta,seg_stk,seg_exit,e_stk,e_stk_in,e_exit,rt_exit,rt_twin,dy_wall,e_wall,dL,codes\n");
    for (int H : {5, 6})
        for (int glued : {0, 1})
            for (auto pr : std::vector<std::pair<int, int>>{{1, 2}, {2, 1}})
                for (size_t q = 0; q < 2; ++q)
                    for (int it : {2, 5})
                        for (double thd : {30.0, 60.0})
                        {
                            const double d1 = 0.011;
                            auto wall = [&](bool stack, arma::uvec &mi)
                            {
                                std::vector<double> x;
                                std::vector<int> m;
                                std::vector<bool> fl;
                                if (glued)
                                    x.push_back(-d1 - gap), m.push_back(7), fl.push_back(false);
                                x.push_back(-d1), m.push_back(H), fl.push_back(false);
                                x.push_back(0.0), m.push_back(pr.first), fl.push_back(flips[q].first);
                                if (stack)
                                    x.push_back(gap), m.push_back(pr.second), fl.push_back(flips[q].second);
                                x.push_back(d1), m.push_back(H), fl.push_back(true);
                                x.push_back(0.5), m.push_back(5), fl.push_back(false), x.push_back(0.55), m.push_back(5), fl.push_back(true);
                                return ub::build(x, m, fl, mi);
                            };
                            arma::uvec mi, m1;
                            const arma::mat mesh = wall(true, mi), mesh1 = wall(false, m1);
                            const auto c = ub::chain(pal, mesh, mi, std::vector<int>(5, it), f, thd, {1.5});
                            const auto t = ub::chain(pal, mesh1, m1, std::vector<int>(5, it), f, thd, {1.5});
                            const auto s = ub::chain(pal, mesh, mi, std::vector<int>(5, (it == 2) ? 1 : 4), f, thd, {1.5});
                            INFO("B, host " << mn[H] << ", glued " << glued << ", " << mn[pr.first] << "|" << mn[pr.second] << ", " << fn[q] << ", type " << it << ", theta " << thd);
                            std::string codes;
                            for (const auto &e : c)
                                codes += std::to_string((unsigned)e.ot) + "/" + std::to_string((unsigned)e.rt) + " ";
                            REQUIRE((t.size() == 5 && s.size() == 5));
                            CHECK(c.size() == 5); // the bent segment can miss an event: a chain that ends early is the defect
                            if (c.size() != 5)
                            {
                                QD_DIAG("%s,%s,%s|%s,%s,%d,%.0f,chain of %zu events,%s\n", mn[H], glued ? "glued" : "bare", mn[pr.first], mn[pr.second], fn[q], it, thd, c.size(), codes.c_str());
                                continue;
                            }
                            std::complex<double> zc = 1.0, zs = 1.0;
                            for (size_t k = 0; k < 5; ++k)
                                zc *= cpx(c[k].xpr, 0, 0), zs *= cpx(s[k].xpr, 0, 0);
                            const arma::mat sg1 = unit(c[1].destN - c[1].origN), sg2 = unit(c[2].destN - c[2].origN);
                            const double e_stk = dmax(sg1, unit(c[1].path_dirN)), e_stk_in = dmax(sg1, unit(c[0].destN - c[0].origN));
                            const double e_exit = dmax(sg2, unit(c[2].path_dirN)), dyw = c[3].fbs(0, 1) - t[3].fbs(0, 1);
                            const double e_wall = dmax(unit(c[3].destN - c[3].origN), unit(t[3].destN - t[3].origN)), dL = ub::dB(zc) - ub::dB(zs);
                            QD_DIAG("%s,%s,%s|%s,%s,%d,%.0f,%.4f,%.4f,%.3e,%.3e,%.3e,%u,%u,%.3e,%.3e,%+.6f,%s\n", mn[H], glued ? "glued" : "bare", mn[pr.first], mn[pr.second], fn[q], it, thd,
                                    ang(sg1), ang(sg2), e_stk, e_stk_in, e_exit, (unsigned)c[2].rt, (unsigned)t[2].rt, dyw, e_wall, dL, codes.c_str());
                            CHECK(e_stk_in < TOL);                 // on the line it was reached on
                            CHECK(e_stk < (glued ? 2.0e-4 : TOL)); // which is the direction written
                            CHECK(e_exit < (glued ? 2.0e-4 : TOL));
                            CHECK(c[2].rt == t[2].rt);                     // the exit behind the stack is the exit behind a single sheet
                            if (!glued)                                    // behind a glued entry that entry's anchoring can set the bit, for the twin as well
                                CHECK((c[2].rt & 2) == 0);                 // behind the stack the wave's direction is the segment again
                            CHECK(std::abs(dyw) < (glued ? 1.0e-6 : TOL)); // the next wall is met where the single sheet meets it
                            CHECK(e_wall < (glued ? 2.0e-4 : TOL));
                            CHECK(std::abs(dL) < 1.0e-3); // and with the level of the undeviated tracer
                        }

    // ---- C. Twins that must not move, with every digit ----
    QD_DIAG("\n# diag: stack geometry C1, what must not move: origN, destN and a weighted sum of tridirN with every digit. A single sheet, a\n");
    QD_DIAG("# collapsed duplicate, a glued entry, a bare entry with every type, and the stack with types 1, 4, 0 and 3\n");
    QD_DIAG("what,type,theta,ot,rt,origN_x,origN_y,origN_z,destN_x,destN_y,destN_z,tube\n");
    for (int w = 0; w < 5; ++w)
        for (int it : {1, 2, 4, 5, 0, 3})
            for (double thd : {30.0, 60.0})
            {
                if (w == 4 && (it == 2 || it == 5))
                    continue;
                arma::uvec mi;
                arma::mat mesh, o, d, tv, td;
                const char *nm = "";
                if (w == 0)
                    mesh = ub::build({0.0}, {1}, {false}, mi), nm = "C1 alone";
                else if (w == 1)
                    mesh = ub::build({0.0, gap}, {1, 1}, {false, true}, mi), nm = "C1|C1";
                else if (w == 2)
                    mesh = ub::build({0.0, gap, 0.1}, {1, 6, 6}, {false, false, true}, mi), nm = "C1 on Hl";
                else if (w == 3)
                    mesh = ub::build({0.0, 0.1}, {6, 6}, {false, true}, mi), nm = "Hl";
                else
                    mesh = ub::build({0.0, gap}, {1, 2}, {false, true}, mi), nm = "C1|C2";
                diag_ray_at(thd, o, d);
                fan(unit(d - o), tv, td);
                ub::Opt op;
                op.relaunch = (it != 0 && it != 3), op.trivec = &tv, op.tridir = &td;
                const ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, nullptr, op);
                QD_DIAG("%s,%d,%.0f,%u,%u,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", nm, it, thd, (unsigned)E.ot, (unsigned)E.rt, E.origN(0, 0), E.origN(0, 1), E.origN(0, 2),
                        E.destN(0, 0), E.destN(0, 1), E.destN(0, 2), sig(E.tridirN));
                twin("stack geometry C1", twin_key("%s,%d,%.0f", nm, it, thd), {(double)E.ot, (double)E.rt, E.origN(0, 0), E.origN(0, 1), E.origN(0, 2), E.destN(0, 0), E.destN(0, 1), E.destN(0, 2), sig(E.tridirN)});
                CHECK(E.rt != 0);
            }

    // The stack event itself, on a ray out of the horizontal plane so that the basis of an EM
    // coefficient shows. The redo's coefficient, both codes and the words are printed with every
    // digit. What ray_mesh_interact wrote is printed for types 1, 4 and 5; with type 2 it is
    // written in the basis of the outgoing direction and is type 1's once that is the incoming one.
    QD_DIAG("\n# diag: stack geometry C2, the stack event on a ray out of the horizontal plane, with every digit. xpr: the redo's coefficient\n");
    QD_DIAG("# as a weighted sum; own: ray_mesh_interact's (not printed for type 2); d_own: the own coefficient against that of type 1 or 4,\n");
    QD_DIAG("# largest entry\n");
    QD_DIAG("stack,normals,type,theta,ot,rt,prev.cur.buf,xpr,own,d_own\n");
    for (auto pr : std::vector<std::pair<int, int>>{{1, 2}, {2, 1}})
        for (size_t q = 0; q < flips.size(); ++q)
            for (int it : {1, 2, 4, 5})
                for (double thd : {30.0, 60.0})
                {
                    arma::uvec mi;
                    const arma::mat mesh = ub::build({0.0, gap}, {pr.first, pr.second}, {flips[q].first, flips[q].second}, mi);
                    const arma::mat o = {{-1.0, 0.37, 0.2}}, d = {{1.0, 0.37 + 2.0 * std::tan(thd * QPI / 180.0), 1.5}};
                    const ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, nullptr), E1 = ub::traced<double>(pal, mesh, mi, (it < 3) ? 1 : 4, f, o, d, nullptr);
                    double d_own = 0.0;
                    for (arma::uword i = 0; i < E.xpr0.n_elem; ++i)
                        d_own = std::max(d_own, std::abs(E.xpr0(i) - E1.xpr0(i)));
                    QD_DIAG("%s|%s,%s,%d,%.0f,%u,%u,%u.%u.%u,%.17g,", mn[pr.first], mn[pr.second], fn[q], it, thd, (unsigned)E.ot, (unsigned)E.rt, ub::w(E.prev(0)), ub::w(E.cur(0)),
                            ub::w(E.buf(0)), sig(E.xpr));
                    if (it == 2)
                        QD_DIAG("-,%.3e\n", d_own);
                    else
                        QD_DIAG("%.17g,%.3e\n", sig(E.xpr0), d_own);
                    {
                        std::vector<double> v = {(double)E.ot, (double)E.rt, (double)ub::w(E.prev(0)), (double)ub::w(E.cur(0)), (double)ub::w(E.buf(0)), sig(E.xpr)};
                        if (it != 2)
                            v.push_back(sig(E.xpr0));
                        twin("stack geometry C2", twin_key("%s|%s,%s,%d,%.0f", mn[pr.first], mn[pr.second], fn[q], it, thd), v);
                    }
                    INFO("C2, " << mn[pr.first] << "|" << mn[pr.second] << ", " << fn[q] << ", type " << it << ", theta " << thd);
                    CHECK(E.rt == 69);
                    CHECK(d_own < 1.0e-12); // one own coefficient for the bent and the undeviated type
                }
}

TEST_CASE("Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more than it received", "[wave]")
{
    ub::OneThread one_thread;
    // A sheet without the mass column whose permittivity is below the transverse invariant carries an evanescent wave,
    // or for a lossy sheet a strongly damped one, and the Fresnel reflectance of its faces is close to one for a reason
    // that is not a missing forward port: the wave tunnels. A rule that rescales the raw amplitude by (1 - R') / (1 -
    // R0) would be a gain there: 20 dB per decade of whatever loss keeps 1 - R0 off zero, the host's or the sheet's
    // own, and a step where it is zero. tf acts on such a face as on every face of a coherent sheet, as the ideal
    // transformer of the sheet-tf case, formed on the modulus of the admittance ratio: the tunnelled wave stays and the
    // face creates nothing. Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update. S
    // is 6 mm of eps 0.31 - j es, past its critical angle from 33.83 deg in every medium entered from air. These are EM
    // materials: tf, no mass column.
    //
    // A. The sheet inside a host: every loss of the host, both signs of tf, tf on the host, scalar and EM
    // B. The sheet with a loss of its own, inside a host and in air
    // C. The sheet as a lining, on the near and on the far face of a host
    // D. The core: the sheet in an ordered stack, both orders, in air and inside the host
    // E. Under a closed outer pair: the reflect port; the transmit port with every digit
    // F. Twins that must not move, with every digit
    const double f = FRQ, gap = 1.0e-5, d1 = 0.010, d2 = 0.012, ZERO = 1.0e-9;
    const double losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    auto lin = [](double x)
    { return std::pow(10.0, 0.1 * x); };
    auto lvl = [](const std::vector<ub::Ev> &v, size_t n, int k)
    { return (v.size() == n) ? ub::dB(cpx(v[n - 1].xpr, 0, k)) : -999.0; };
    auto refl = [](int tr)
    { return (tr == 4 || tr == 5) ? 3 : 0; };
    // The sheet event inside a host, 10 mm in front and 12 mm behind: both ports, with the host leg in front
    auto inside = [&](const ub::Pal &P, int tr, double thd, int k, double &T, double &R)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, d1, d1 + d2}, {2, 1, 2}, {false, false, true}, mi);
        T = lvl(ub::chain(P, mesh, mi, {tr, tr}, f, thd, {1.5}), 2, k), R = lvl(ub::chain(P, mesh, mi, {tr, refl(tr)}, f, thd, {1.5}), 2, k);
    };

    // ---- A. Inside a host ----
    QD_DIAG("\n# diag: rare sheet A, S 6 mm eps 0.31 inside 10 mm + 12 mm of eps 2.50 - j ei, the sheet event with its host leg, dB.\n");
    QD_DIAG("# tf: on the sheet, or on the host where the sheet has none. k: 0 scalar / VV (TE), 3 HH (TM). sum: T + R, linear\n");
    QD_DIAG("type,k,tf_on,tf,theta,ei,T,R,sum\n");
    for (int tr : {4, 5, 1, 2})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (int cfg = 0; cfg < 4; ++cfg) // sheet 0.5, sheet 0.1, sheet -0.5, host 0.5
                for (double thd : {45.0, 60.0, 75.0})
                {
                    const double tS = (cfg == 0) ? 0.5 : ((cfg == 1) ? 0.1 : ((cfg == 2) ? -0.5 : 0.0)), tH = (cfg == 3) ? 0.5 : 0.0;
                    double Tl[5], Rl[5];
                    for (int q = 0; q < 5; ++q)
                    {
                        ub::Pal P = lump_palette({{0.31, 0.0, 0.006, 0.0, tS}, {2.50, losses[q], 0.0, 0.0, tH}});
                        inside(P, tr, thd, k, Tl[q], Rl[q]);
                        QD_DIAG("%d,%d,%s,%.1f,%.0f,%.0e,%.4f,%.4f,%.6f\n", tr, k, cfg == 3 ? "host" : "sheet", cfg == 3 ? tH : tS, thd, losses[q], Tl[q], Rl[q], lin(Tl[q]) + lin(Rl[q]));
                        INFO("A, type " << tr << ", k " << k << ", cfg " << cfg << ", theta " << thd << ", ei " << losses[q]);
                        CHECK(Tl[q] < ZERO); // no more than it received, on either port
                        CHECK(Rl[q] < ZERO);
                        if (losses[q] == 0.0)                              // and on their sum where the host is lossless; in a lossy host the incident and the
                            CHECK(lin(Tl[q]) + lin(Rl[q]) < 1.0 + 1.0e-6); // reflected wave interfere and the sum is printed only
                    }
                    INFO("A, continuity at zero loss, type " << tr << ", k " << k << ", cfg " << cfg << ", theta " << thd);
                    CHECK(std::abs(Tl[3] - Tl[4]) < 1.0e-3); // ei 1e-6 against ei 0
                    CHECK(std::abs(Rl[3] - Rl[4]) < 1.0e-3);
                }
        }

    QD_DIAG("\n# diag: rare sheet A2, tf to zero, the same event at 45 deg, EM type 1. The sheet without tf is the tunnelled wave, in both\n");
    QD_DIAG("# channels and at every loss of the host, and a tf of 1e-9 has to sit on it\n");
    QD_DIAG("ei,tf,T_te,R_te,T_tm,R_tm\n");
    for (double ei : {0.8, 1.0e-3, 0.0})
    {
        double T0 = 0.0, R0 = 0.0;
        for (double tS : {0.0, 1.0e-9, -1.0e-9, 1.0e-3, -1.0e-3, 0.1, -0.1})
        {
            ub::Pal P = lump_palette({{0.31, 0.0, 0.006, 0.0, tS}, {2.50, ei, 0.0}});
            double T, R, Tm, Rm;
            inside(P, 1, 45.0, 0, T, R), inside(P, 1, 45.0, 3, Tm, Rm);
            QD_DIAG("%.0e,%.0e,%.4f,%.4f,%.4f,%.4f\n", ei, tS, T, R, Tm, Rm);
            if (tS == 0.0)
                T0 = T, R0 = R;
            INFO("A2, ei " << ei << ", tf " << tS);
            CHECK(T < ZERO);
            CHECK(Tm < ZERO);
            if (std::abs(tS) == 1.0e-9)
            {
                CHECK(std::abs(T - T0) < 1.0e-6); // continuous in tf at zero
                CHECK(std::abs(R - R0) < 1.0e-6);
            }
            CHECK(T > -3.0); // and the tunnelled wave is not dropped, with or without tf
        }
    }

    QD_DIAG("\n# diag: rare sheet A3, across the critical angle of the sheet, 33.83 deg, tf 0.5 on the sheet, scalar\n");
    QD_DIAG("ei,theta,T,R,sum\n");
    for (double ei : {0.8, 1.0e-3, 0.0})
    {
        double Tb = 0.0;
        for (double thd : {33.0, 33.80, 33.87, 34.5, 36.0})
        {
            ub::Pal P = lump_palette({{0.31, 0.0, 0.006, 0.0, 0.5}, {2.50, ei, 0.0}});
            double T, R;
            inside(P, 4, thd, 0, T, R);
            QD_DIAG("%.0e,%.2f,%.4f,%.4f,%.6f\n", ei, thd, T, R, lin(T) + lin(R));
            INFO("A3, ei " << ei << ", theta " << thd);
            CHECK(T < ZERO);
            CHECK(lin(T) + lin(R) < 1.0 + 1.0e-9); // the two ports of one sheet
            if (thd == 33.80)
                Tb = T;
            if (thd == 33.87)
                CHECK(std::abs(T - Tb) < 0.5); // no step where the wave in the sheet becomes evanescent
        }
    }

    // ---- B. A loss of its own ----
    QD_DIAG("\n# diag: rare sheet B, S 6 mm eps 0.31 - j es with tf, inside eps 2.50 - 0.80j, inside eps 2.50, and alone in air, scalar, dB.\n");
    QD_DIAG("# sum is recorded: for a lossy sheet with tf the two ports are not bounded by one together, which is an issue of its own\n");
    QD_DIAG("where,tf,theta,es,T,R,sum\n");
    for (int where = 0; where < 3; ++where)
        for (double tS : {0.5, -0.5})
            for (double thd : {45.0, 60.0})
            {
                double Tl[5], Rl[5];
                const double es[5] = {0.2, 1.0e-2, 1.0e-3, 1.0e-6, 0.0};
                for (int q = 0; q < 5; ++q)
                {
                    ub::Pal P = lump_palette({{0.31, es[q], 0.006, 0.0, tS}, {2.50, where == 0 ? 0.8 : 0.0, 0.0}});
                    if (where < 2)
                        inside(P, 4, thd, 0, Tl[q], Rl[q]);
                    else
                    {
                        arma::uvec mi;
                        arma::mat mesh = ub::build({0.0}, {1}, {false}, mi);
                        Tl[q] = lvl(ub::chain(P, mesh, mi, {4}, f, thd, {1.5}), 1, 0), Rl[q] = lvl(ub::chain(P, mesh, mi, {3}, f, thd, {1.5}), 1, 0);
                    }
                    QD_DIAG("%s,%.1f,%.0f,%.0e,%.4f,%.4f,%.6f\n", where == 0 ? "host 2.50-0.80j" : (where == 1 ? "host 2.50" : "air"), tS, thd, es[q], Tl[q], Rl[q], lin(Tl[q]) + lin(Rl[q]));
                    INFO("B, where " << where << ", tf " << tS << ", theta " << thd << ", es " << es[q]);
                    CHECK(Tl[q] < ZERO);
                    CHECK(Rl[q] < ZERO);
                }
                INFO("B, continuity at zero loss of the sheet, where " << where << ", tf " << tS << ", theta " << thd);
                CHECK(std::abs(Tl[3] - Tl[4]) < 1.0e-3);
                CHECK(std::abs(Rl[3] - Rl[4]) < 1.0e-3);
            }

    // ---- C. The sheet as a lining ----
    QD_DIAG("\n# diag: rare sheet C, S 6 mm eps 0.31 with tf as a lining of 10 mm eps 2.50 - j ei: entry, on the near face (one event), and\n");
    QD_DIAG("# exit, on the far face (the second event, with the host leg), series declined, scalar, dB. own: ray_mesh_interact's entry\n");
    QD_DIAG("type,tf,theta,ei,entry,exit,entry_R,exit_R,own_entry,exit-entry\n");
    for (int tr : {4, 5})
        for (double tS : {0.5, -0.5})
            for (double thd : {45.0, 60.0})
            {
                double En[5], Ex[5];
                for (int q = 0; q < 5; ++q)
                {
                    ub::Pal P = lump_palette({{0.31, 0.0, 0.006, 0.0, tS}, {2.50, losses[q], 0.0}});
                    arma::uvec mx, me;
                    arma::mat m_ex = ub::build({0.0, d1, d1 + gap}, {2, 2, 1}, {false, true, false}, mx);
                    arma::mat m_en = ub::build({0.0, gap, gap + d1}, {1, 2, 2}, {false, false, true}, me);
                    auto a = ub::chain(P, m_en, me, {tr}, f, thd, {1.5});
                    En[q] = lvl(a, 1, 0), Ex[q] = lvl(ub::chain(P, m_ex, mx, {tr, tr}, f, thd, {1.5}), 2, 0);
                    const double EnR = lvl(ub::chain(P, m_en, me, {3}, f, thd, {1.5}), 1, 0), ExR = lvl(ub::chain(P, m_ex, mx, {tr, 3}, f, thd, {1.5}), 2, 0);
                    const double own = a.empty() ? -999.0 : ub::dB(cpx(a[0].xpr0, 0, 0));
                    QD_DIAG("%d,%.1f,%.0f,%.0e,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f\n", tr, tS, thd, losses[q], En[q], Ex[q], EnR, ExR, own, Ex[q] - En[q]);
                    INFO("C, type " << tr << ", tf " << tS << ", theta " << thd << ", ei " << losses[q]);
                    CHECK(En[q] < ZERO);
                    CHECK(Ex[q] < ZERO);
                    CHECK(EnR < ZERO);
                    CHECK(ExR < ZERO);
                    CHECK(En[q] > -30.0);                  // the lining tunnels, with either sign of tf
                    CHECK(std::abs(own - En[q]) < 1.0e-9); // the two public functions agree on a glued entry
                    if (losses[q] == 0.0)
                        CHECK(std::abs(Ex[q] - En[q]) < 1.0e-6); // a lossless lined wall is one wall from its two sides
                }
                INFO("C, continuity at zero loss, type " << tr << ", tf " << tS << ", theta " << thd);
                CHECK(std::abs(En[3] - En[4]) < 1.0e-3);
                CHECK(std::abs(Ex[3] - Ex[4]) < 1.0e-3);
            }

    // ---- D. The core ----
    QD_DIAG("\n# diag: rare sheet D, the ordered stack of S 6 mm eps 0.31 with tf and C 10 mm eps 2.50 - j ei without, both orders, in air (one\n");
    QD_DIAG("# event) and inside 10 mm + 12 mm of eps 2.50 - j ei (the stack event with its host leg), scalar, dB\n");
    QD_DIAG("where,tf,theta,ei,S|C,C|S,d\n");
    for (int where = 0; where < 2; ++where)
        for (double tS : {0.5, -0.5})
            for (double thd : {45.0, 60.0})
            {
                double L[5][2];
                for (int q = 0; q < 5; ++q)
                {
                    ub::Pal P = lump_palette({{0.31, 0.0, 0.006, 0.0, tS}, {2.50, losses[q], 0.0}, {2.50, losses[q], 0.010}});
                    for (int ord = 0; ord < 2; ++ord)
                    {
                        arma::uvec mi;
                        const int m1 = ord ? 3 : 1, m2 = ord ? 1 : 3;
                        if (where == 0)
                        {
                            arma::mat mesh = ub::build({0.0, gap}, {m1, m2}, {false, true}, mi);
                            L[q][ord] = lvl(ub::chain(P, mesh, mi, {4}, f, thd, {1.5}), 1, 0);
                        }
                        else
                        {
                            arma::mat mesh = ub::build({0.0, d1, d1 + gap, d1 + gap + d2}, {2, m1, m2, 2}, {false, false, true, true}, mi);
                            L[q][ord] = lvl(ub::chain(P, mesh, mi, {4, 4}, f, thd, {1.5}), 2, 0);
                        }
                    }
                    QD_DIAG("%s,%.1f,%.0f,%.0e,%.4f,%.4f,%.2e\n", where ? "in host" : "in air", tS, thd, losses[q], L[q][0], L[q][1], L[q][1] - L[q][0]);
                    INFO("D, where " << where << ", tf " << tS << ", theta " << thd << ", ei " << losses[q]);
                    CHECK(L[q][0] < ZERO);
                    CHECK(L[q][1] < ZERO);
                    CHECK(L[q][0] > -30.0);
                    CHECK(std::abs(L[q][1] - L[q][0]) < 1.0e-9); // the transmit port of a stack does not read the order of its layers
                }
                INFO("D, continuity at zero loss, where " << where << ", tf " << tS << ", theta " << thd);
                CHECK(std::abs(L[3][0] - L[4][0]) < 1.0e-3);
                CHECK(std::abs(L[3][1] - L[4][1]) < 1.0e-3);
            }

    // ---- E. Under a closed outer pair ----
    // Both ports are the composite on the pad of the closed wall face. The transmit port is
    // printed with every digit; the echo of the reflect port crosses the near face twice.
    QD_DIAG("\n# diag: rare sheet E, S 10 mm eps 0.31 - j es with tf 0.5 on a rigid wall eps 2.401e-9 with tf tW, the glued entry, scalar.\n");
    QD_DIAG("# R: the reflect port, dB. T_re, T_im: the transmit port, with every digit\n");
    QD_DIAG("tW,theta,es,R,T_re,T_im\n");
    for (double tW : {0.0, 0.3})
        for (double thd : {45.0, 60.0})
        {
            double Rl[5];
            const double es[5] = {0.2, 1.0e-2, 1.0e-3, 1.0e-6, 0.0};
            for (int q = 0; q < 5; ++q)
            {
                ub::Pal P = lump_palette({{0.31, es[q], 0.010, 0.0, 0.5}, {2.401e-9, 0.0, 0.0, 0.0, tW}});
                arma::uvec me;
                arma::mat m_en = ub::build({0.0, gap, gap + 0.1}, {1, 2, 2}, {false, false, true}, me);
                auto a = ub::chain(P, m_en, me, {4}, f, thd, {1.5});
                Rl[q] = lvl(ub::chain(P, m_en, me, {3}, f, thd, {1.5}), 1, 0);
                const std::complex<double> t = a.empty() ? std::complex<double>(0.0, 0.0) : cpx(a[0].xpr, 0, 0);
                QD_DIAG("%.1f,%.0f,%.0e,%.4f,%.17g,%.17g\n", tW, thd, es[q], Rl[q], std::real(t), std::imag(t));
                twin("rare sheet E", twin_key("%.1f,%.0f,%.0e", tW, thd, es[q]), {std::real(t), std::imag(t)});
                INFO("E, tW " << tW << ", theta " << thd << ", es " << es[q]);
                CHECK(Rl[q] < ZERO);
                CHECK(std::norm(t) < 1.0);
            }
            INFO("E, continuity at zero loss of the sheet, tW " << tW << ", theta " << thd);
            CHECK(std::abs(Rl[3] - Rl[4]) < 1.0e-3);
        }

    // ---- F. Twins that must not move ----
    QD_DIAG("\n# diag: rare sheet F, twins, the complex coefficient of the sheet event with every digit, scalar and EM (VV, HH).\n");
    QD_DIAG("# A sheet denser than the invariant, a rare sheet without tf anywhere, a rare sheet below its critical angle and a sheet under a\n");
    QD_DIAG("# closed outer pair; families that sheet tf H or flat branch H print are not repeated here\n");
    QD_DIAG("row,type,port,theta,re,im,re_hh,im_hh\n");
    {
        const double BIG = 1.0e12;
        ub::Pal P = lump_palette({{5.24, 0.63, 0.006, 0.0, 0.5},                                            // 1 dense lossy sheet with tf
                                  {5.24, 0.0, 0.006, 0.0, -0.5},                                            // 2 dense lossless sheet, tf below zero
                                  {0.31, 0.0, 0.006},                                                       // 3 rare sheet without tf
                                  {0.31, 0.0, 0.006, 0.0, 0.5},                                             // 4 rare sheet with tf
                                  {0.50, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG},      // 5 rare mass-law sheet
                                  {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG}, // 6 rigid mass-law sheet
                                  {2.50, 0.80, 0.0},                                                        // 7 lossy host
                                  {2.50, 0.0, 0.0},                                                         // 8 lossless host
                                  {1.50, 0.40, 0.0, 0.0, 0.5},                                              // 9 host with tf
                                  {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                           // 10 rigid wall
                                  {0.31, 0.20, 0.006, 0.0, 0.5}});                                          // 11 porous sheet with tf
        struct Row
        {
            const char *name;
            int sheet, host; // host 0: the sheet alone in air; host < 0: the sheet glued on the near face of -host
            std::vector<double> th;
        };
        const std::vector<Row> rows = {
            {"dense lossless tf -0.5 inside lossless host", 2, 8, {20.0, 45.0, 60.0}},
            {"dense lossy tf 0.5 inside host with tf", 1, 9, {20.0, 45.0, 60.0}},
            {"rare no tf inside lossless host", 3, 8, {20.0, 45.0, 60.0}},
            {"rare tf 0.5 below its critical angle - lossy host", 4, 7, {10.0, 20.0, 30.0}},
            {"porous tf 0.5 below its critical angle - in air", 11, 0, {10.0, 20.0, 30.0}},
        };
        for (const auto &R : rows)
            for (int tr : {4, 1})
                for (int port = 0; port < 2; ++port)
                {
                    if (R.host < 0 && port)
                        continue; // the reflect port under a closed outer pair is table E
                    for (double thd : R.th)
                    {
                        arma::uvec mi;
                        arma::mat mesh;
                        std::vector<int> seq;
                        const int last = port ? refl(tr) : tr;
                        if (R.host == 0)
                            mesh = ub::build({0.0}, {R.sheet}, {false}, mi), seq = {last};
                        else if (R.host < 0)
                            mesh = ub::build({0.0, gap, gap + 0.1}, {R.sheet, -R.host, -R.host}, {false, false, true}, mi), seq = {last};
                        else
                            mesh = ub::build({0.0, d1, d1 + d2}, {R.host, R.sheet, R.host}, {false, false, true}, mi), seq = {tr, last};
                        auto a = ub::chain(P, mesh, mi, seq, f, thd, {1.5});
                        const bool ok = a.size() == seq.size();
                        const std::complex<double> z = ok ? cpx(a.back().xpr, 0, 0) : 0.0, zh = (ok && tr == 1) ? cpx(a.back().xpr, 0, 3) : 0.0;
                        QD_DIAG("%s,%d,%s,%.0f,%.17g,%.17g,%.17g,%.17g\n", R.name, tr, port ? "R" : "T", thd, std::real(z), std::imag(z), std::real(zh), std::imag(zh));
                        twin("rare sheet F", twin_key("%s,%d,%s,%.0f", R.name, tr, port ? "R" : "T", thd), {std::real(z), std::imag(z), std::real(zh), std::imag(zh)});
                    }
                }
    }
}

TEST_CASE("Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the characteristic matrix", "[wave]")
{
    ub::OneThread one_thread;
    // A lossless sheet without the mass column whose eps mu is below the transverse invariant
    // carries an evanescent wave, and a layer of finite thickness between two media that carry a
    // wave passes part of it: frustrated total reflection. The characteristic matrix carries that
    // and is the reference here, on both ports and in both channels, at every loss of the host.
    // A face of such a sheet reads a Fresnel reflectance of one next to a lossless medium, and
    // above one in TM next to a lossy one, and neither says that the face has no forward port.
    // The face of a rare sheet is the transformer of the sheet-tf case, with or without tf; at
    // tf = 0 it is the raw amplitude, which is the matrix.
    //
    // S is 6 mm of eps 0.31, past its critical angle from 33.83 deg in every medium entered from
    // air; the hosts are eps 2.50 - j ei. EM materials: no tf, no mass column. Everything runs
    // through ray_triangle_intersect, ray_mesh_interact and ray_state_update.
    //
    // A. Inside a host, both ports, TE and TM, every loss, both tracers
    // B. Alone in air, across the critical angle; ray_mesh_interact's own composite
    // C. As a lining, on the near and on the far face of a host
    // D. tf to zero: 1e-9 and -1e-9 on the rows of A, B and C
    // E. One sheet against the ordered stack of its two halves
    // F. Under a closed outer pair without tf: nothing passes, the reflect port is the matrix
    // G. A rigid sheet of the acoustic mapping without the mass column: a wave medium, on the matrix
    // H. Twins with every digit
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    const double f = FRQ, gap = 1.0e-5, d1 = 0.010, d2 = 0.012, k0 = 2.0 * QPI * f / C0, TOL = 1.0e-9;
    const double eS = 0.31, tS = 0.006, losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    const std::complex<double> U(1.0, 0.0), cS(eS, 0.0), cRef(3.0, 0.0);
    auto val = [](const std::vector<ub::Ev> &v, size_t n, int k)
    { return (v.size() == n) ? cpx(v[n - 1].xpr, 0, k) : std::complex<double>(0.0, 0.0); };
    auto dBc = [](std::complex<double> z)
    { return (std::abs(z) > 0.0) ? 20.0 * std::log10(std::abs(z)) : -999.0; };
    auto refl = [](int tr)
    { return (tr == 4 || tr == 5) ? 3 : 0; };
    auto rel = [](std::complex<double> a, std::complex<double> b)
    { return std::abs(a - b) / std::abs(b); };
    // Sheet material 1 and a reference sheet 3, 1 mm of eps 3.0 that is never rare; host 2
    auto pal_of = [&](double ei, double tf)
    { return lump_palette({{eS, 0.0, tS, 0.0, tf}, {2.50, ei, 0.0}, {3.0, 0.0, 0.001}}); };
    auto host_of = [&](double ei)
    { return std::complex<double>(2.50, -ei); };
    auto ct_of = [&](double ei, double thd)
    { double s = std::sin(thd * QPI / 180.0); return std::sqrt(1.0 - s * s / host_of(ei)); };
    auto legdB = [&](double ei, double thd)
    { return 20.0 * std::log10(std::abs(std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(host_of(ei)) * ct_of(ei, thd) * d1))); };
    // The sheet event inside the host, 10 mm in front and 12 mm behind, both ports
    auto inside = [&](const ub::Pal &P, int sheet, int tr, double thd, int k, std::complex<double> &T, std::complex<double> &R, double da = 0.010, double db = 0.012)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, da, da + db}, {2, sheet, 2}, {false, false, true}, mi);
        T = val(ub::chain(P, mesh, mi, {tr, tr}, f, thd, {1.5}), 2, k), R = val(ub::chain(P, mesh, mi, {tr, refl(tr)}, f, thd, {1.5}), 2, k);
    };
    auto in_air = [&](const ub::Pal &P, int sheet, int tr, double thd, int k, std::complex<double> &T, std::complex<double> &R, std::complex<double> *own = nullptr)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0}, {sheet}, {false}, mi);
        auto a = ub::chain(P, mesh, mi, {tr}, f, thd, {1.5});
        T = val(a, 1, k), R = val(ub::chain(P, mesh, mi, {refl(tr)}, f, thd, {1.5}), 1, k);
        if (own)
            *own = a.empty() ? std::complex<double>(0.0, 0.0) : cpx(a[0].xpr0, 0, k);
    };
    auto entry = [&](const ub::Pal &P, int sheet, int tr, double thd, int k, std::complex<double> &T, std::complex<double> &R, std::complex<double> *own = nullptr)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, gap, gap + d1}, {sheet, 2, 2}, {false, false, true}, mi);
        auto a = ub::chain(P, mesh, mi, {tr}, f, thd, {1.5});
        T = val(a, 1, k), R = val(ub::chain(P, mesh, mi, {refl(tr)}, f, thd, {1.5}), 1, k);
        if (own)
            *own = a.empty() ? std::complex<double>(0.0, 0.0) : cpx(a[0].xpr0, 0, k);
    };
    auto exit_of = [&](const ub::Pal &P, int sheet, int tr, double thd, int k, std::complex<double> &T, std::complex<double> &R)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, d1, d1 + gap}, {2, 2, sheet}, {false, true, false}, mi);
        T = val(ub::chain(P, mesh, mi, {tr, tr}, f, thd, {1.5}), 2, k), R = val(ub::chain(P, mesh, mi, {tr, refl(tr)}, f, thd, {1.5}), 2, k);
    };
    auto pick = [](const DiagSheet &W, int k, bool port_r)
    { return port_r ? (k ? W.R_tm : W.R_te) : (k ? W.T_tm : W.T_te); };

    // ---- A. Inside a host ----
    QD_DIAG("\n# diag: flat branch A, S 6 mm eps 0.31 without tf inside 10 mm + 12 mm of eps 2.50 - j ei, the sheet event with its host leg.\n");
    QD_DIAG("# T, R: dB; T_cm, R_cm: the characteristic matrix times the host leg. dT, dR: relative distance in the complex plane of the event over\n");
    QD_DIAG("# the same event with a 1 mm sheet of eps 3.0 from the same ratio of two matrices. k: 0 scalar / VV (TE), 3 HH (TM)\n");
    QD_DIAG("type,k,theta,ei,T,T_cm,R,R_cm,dT,dR,sum\n");
    for (int tr : {1, 2, 4, 5})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (double thd : {45.0, 60.0, 75.0})
            {
                double Tl[5], Rl[5];
                for (int q = 0; q < 5; ++q)
                {
                    const double ei = losses[q];
                    ub::Pal P = pal_of(ei, 0.0);
                    std::complex<double> T, R, T1, R1;
                    inside(P, 1, tr, thd, k, T, R), inside(P, 3, tr, thd, k, T1, R1);
                    DiagSheet W = diag_cm_stack(host_of(ei), U, {{cS, U, tS}}, host_of(ei), U, ct_of(ei, thd), f, false, tr > 2);
                    DiagSheet W1 = diag_cm_stack(host_of(ei), U, {{cRef, U, 0.001}}, host_of(ei), U, ct_of(ei, thd), f, false, tr > 2);
                    const double dT = rel(T / T1, pick(W, k, false) / pick(W1, k, false)), dR = rel(R / R1, pick(W, k, true) / pick(W1, k, true));
                    Tl[q] = dBc(T), Rl[q] = dBc(R);
                    const double sum = std::norm(T) + std::norm(R);
                    QD_DIAG("%d,%d,%.0f,%.0e,%.4f,%.4f,%.4f,%.4f,%.2e,%.2e,%.6f\n", tr, k, thd, ei, Tl[q], dBc(pick(W, k, false)) + legdB(ei, thd), Rl[q],
                            dBc(pick(W, k, true)) + legdB(ei, thd), dT, dR, sum);
                    INFO("A, type " << tr << ", k " << k << ", theta " << thd << ", ei " << ei);
                    CHECK(dT < TOL); // the tunnelled wave, on the matrix
                    CHECK(dR < TOL);
                    CHECK(Tl[q] < TOL); // no more than it received, on either port
                    CHECK(Rl[q] < TOL);
                    if (ei == 0.0)
                        CHECK(std::abs(sum - 1.0) < TOL); // lossless throughout: the two ports are the whole
                }
                INFO("A, continuity at zero loss, type " << tr << ", k " << k << ", theta " << thd);
                CHECK(std::abs(Tl[3] - Tl[4]) < 1.0e-3); // ei 1e-6 against ei 0
                CHECK(std::abs(Rl[3] - Rl[4]) < 1.0e-3);
            }
        }

    // ---- B. Alone in air ----
    QD_DIAG("\n# diag: flat branch B, S alone in air, across its critical angle of 33.83 deg, one event. dT, dR: distance from the matrix in the\n");
    QD_DIAG("# complex plane. d_own: ray_mesh_interact's own transmit coefficient against the redo's. sum: |T|^2 + |R|^2\n");
    QD_DIAG("type,k,theta,T,T_cm,R,R_cm,dT,dR,d_own,sum\n");
    for (int tr : {1, 2, 4, 5})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (double thd : {20.0, 33.0, 33.80, 33.87, 34.5, 45.0, 60.0, 80.0})
            {
                ub::Pal P = pal_of(0.0, 0.0);
                std::complex<double> T, R, own;
                in_air(P, 1, tr, thd, k, T, R, &own);
                DiagSheet W = diag_cm_stack(U, U, {{cS, U, tS}}, U, U, std::cos(thd * QPI / 180.0), f, false, tr > 2);
                const double dT = std::abs(T - pick(W, k, false)), dR = std::abs(R - pick(W, k, true)), sum = std::norm(T) + std::norm(R);
                QD_DIAG("%d,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.2e,%.2e,%.2e,%.9f\n", tr, k, thd, dBc(T), dBc(pick(W, k, false)), dBc(R), dBc(pick(W, k, true)), dT, dR, std::abs(own - T), sum);
                INFO("B, type " << tr << ", k " << k << ", theta " << thd);
                CHECK(dT < TOL);
                CHECK(dR < TOL);
                CHECK(std::abs(sum - 1.0) < TOL);
                CHECK(std::abs(own - T) < 1.0e-12); // the two public functions agree
            }
        }

    // ---- C. As a lining ----
    QD_DIAG("\n# diag: flat branch C, S without tf as a lining of 10 mm eps 2.50 - j ei. Entry, air | S | host, one event: distance from the matrix\n");
    QD_DIAG("# in the complex plane, and ray_mesh_interact's own. Exit, host | S | air, the second event with its host leg, series declined: relative\n");
    QD_DIAG("# distance of the event over the same event with a 1 mm lining of eps 3.0 from the ratio of two matrices. Levels in dB\n");
    QD_DIAG("type,k,theta,ei,en_T,en_R,d_enT,d_enR,d_own,ex_T,ex_R,d_exT,d_exR,exT-enT\n");
    for (int tr : {1, 2, 4, 5})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (double thd : {45.0, 60.0})
            {
                double En[5], Ex[5];
                for (int q = 0; q < 5; ++q)
                {
                    const double ei = losses[q];
                    ub::Pal P = pal_of(ei, 0.0);
                    std::complex<double> T, R, own, X, XR, X1, XR1;
                    entry(P, 1, tr, thd, k, T, R, &own), exit_of(P, 1, tr, thd, k, X, XR), exit_of(P, 3, tr, thd, k, X1, XR1);
                    DiagSheet We = diag_cm_stack(U, U, {{cS, U, tS}}, host_of(ei), U, std::cos(thd * QPI / 180.0), f, false, tr > 2);
                    DiagSheet Wx = diag_cm_stack(host_of(ei), U, {{cS, U, tS}}, U, U, ct_of(ei, thd), f, true, tr > 2);
                    DiagSheet Wx1 = diag_cm_stack(host_of(ei), U, {{cRef, U, 0.001}}, U, U, ct_of(ei, thd), f, true, tr > 2);
                    const double d_enT = std::abs(T - pick(We, k, false)), d_enR = std::abs(R - pick(We, k, true));
                    const double d_exT = rel(X / X1, pick(Wx, k, false) / pick(Wx1, k, false)), d_exR = rel(XR / XR1, pick(Wx, k, true) / pick(Wx1, k, true));
                    En[q] = dBc(T), Ex[q] = dBc(X);
                    QD_DIAG("%d,%d,%.0f,%.0e,%.4f,%.4f,%.2e,%.2e,%.2e,%.4f,%.4f,%.2e,%.2e,%.6f\n", tr, k, thd, ei, En[q], dBc(R), d_enT, d_enR, std::abs(own - T), Ex[q], dBc(XR),
                            d_exT, d_exR, Ex[q] - En[q]);
                    INFO("C, type " << tr << ", k " << k << ", theta " << thd << ", ei " << ei);
                    CHECK(d_enT < TOL);
                    CHECK(d_enR < TOL);
                    CHECK(d_exT < TOL);
                    CHECK(d_exR < TOL);
                    CHECK(std::abs(own - T) < 1.0e-12); // the two public functions agree on a glued entry
                    CHECK(Ex[q] < TOL);
                    CHECK(dBc(R) < TOL);
                    CHECK(dBc(XR) < TOL);
                    if (ei == 0.0)
                        CHECK(std::abs(Ex[q] - En[q]) < 1.0e-6); // a lossless lined wall is one wall from its two sides
                }
                INFO("C, continuity at zero loss, type " << tr << ", k " << k << ", theta " << thd);
                CHECK(std::abs(En[3] - En[4]) < 1.0e-3);
                CHECK(std::abs(Ex[3] - Ex[4]) < 1.0e-3);
            }
        }

    // ---- D. tf to zero ----
    QD_DIAG("\n# diag: flat branch D, tf to zero: the events of A, B and C with tf 1e-9 and -1e-9 on the sheet against the sheet without tf, dB.\n");
    QD_DIAG("# T0, R0: without tf. dT+, dR+, dT-, dR-: the level with tf 1e-9 and -1e-9 less the level without\n");
    QD_DIAG("where,type,k,theta,ei,T0,R0,dT+,dR+,dT-,dR-\n");
    for (int where = 0; where < 4; ++where) // inside, in air, entry, exit
        for (int tr : {1, 4})
            for (int k : {0, 3})
            {
                if (tr > 2 && k)
                    continue;
                for (double thd : {45.0, 60.0})
                    for (int q = 0; q < 5; ++q)
                    {
                        const double ei = losses[q];
                        if (where == 1 && q)
                            continue;
                        double L[3][2];
                        const double tfs[3] = {0.0, 1.0e-9, -1.0e-9};
                        for (int v = 0; v < 3; ++v)
                        {
                            ub::Pal P = pal_of(ei, tfs[v]);
                            std::complex<double> T, R;
                            if (where == 0)
                                inside(P, 1, tr, thd, k, T, R);
                            else if (where == 1)
                                in_air(P, 1, tr, thd, k, T, R);
                            else if (where == 2)
                                entry(P, 1, tr, thd, k, T, R);
                            else
                                exit_of(P, 1, tr, thd, k, T, R);
                            L[v][0] = dBc(T), L[v][1] = dBc(R);
                        }
                        QD_DIAG("%s,%d,%d,%.0f,%.0e,%.4f,%.4f,%.2e,%.2e,%.2e,%.2e\n", where == 0 ? "inside" : (where == 1 ? "air" : (where == 2 ? "entry" : "exit")), tr, k, thd, ei,
                                L[0][0], L[0][1], L[1][0] - L[0][0], L[1][1] - L[0][1], L[2][0] - L[0][0], L[2][1] - L[0][1]);
                        INFO("D, where " << where << ", type " << tr << ", k " << k << ", theta " << thd << ", ei " << ei);
                        for (int v = 1; v < 3; ++v)
                        {
                            CHECK(std::abs(L[v][0] - L[0][0]) < 1.0e-6); // a negligible tf is a negligible change, on both ports
                            CHECK(std::abs(L[v][1] - L[0][1]) < 1.0e-6);
                        }
                    }
            }

    // ---- E. One sheet against the stack of its two halves ----
    // The ordered stack takes the core's matrix. Its two faces are 10 um apart in the mesh, so in a
    // lossy host the leg in front differs by that much, which is 5e-4 dB; lossless, the levels
    // are compared at 1e-9.
    QD_DIAG("\n# diag: flat branch E, S 6 mm against the ordered stack of 2 mm + 4 mm of the same eps 0.31, no tf, the transmit port, dB:\n");
    QD_DIAG("# in air (one event) and inside 10 mm + 12 mm of eps 2.50 - j ei (with the host leg)\n");
    QD_DIAG("where,type,k,theta,ei,one,stack,d\n");
    for (int where = 0; where < 2; ++where)
        for (int tr : {1, 4})
            for (int k : {0, 3})
            {
                if (tr > 2 && k)
                    continue;
                for (double thd : {45.0, 60.0})
                    for (double ei : {0.8, 0.0})
                    {
                        if (!where && ei > 0.0)
                            continue;
                        ub::Pal P = lump_palette({{eS, 0.0, tS}, {2.50, ei, 0.0}, {eS, 0.0, 0.002}, {eS, 0.0, 0.004}});
                        arma::uvec m1, m2;
                        std::vector<ub::Ev> a, b;
                        if (!where)
                        {
                            arma::mat A = ub::build({0.0}, {1}, {false}, m1), B = ub::build({0.0, gap}, {3, 4}, {false, true}, m2);
                            a = ub::chain(P, A, m1, {tr}, f, thd, {1.5}), b = ub::chain(P, B, m2, {tr}, f, thd, {1.5});
                        }
                        else
                        {
                            arma::mat A = ub::build({0.0, d1, d1 + d2}, {2, 1, 2}, {false, false, true}, m1);
                            arma::mat B = ub::build({0.0, d1, d1 + gap, d1 + gap + d2}, {2, 3, 4, 2}, {false, false, true, true}, m2);
                            a = ub::chain(P, A, m1, {tr, tr}, f, thd, {1.5}), b = ub::chain(P, B, m2, {tr, tr}, f, thd, {1.5});
                        }
                        const size_t n = where ? 2 : 1;
                        const double one = dBc(val(a, n, k)), stk = dBc(val(b, n, k));
                        QD_DIAG("%s,%d,%d,%.0f,%.0e,%.6f,%.6f,%.2e\n", where ? "host" : "air", tr, k, thd, ei, one, stk, one - stk);
                        INFO("E, where " << where << ", type " << tr << ", k " << k << ", theta " << thd << ", ei " << ei);
                        CHECK(std::abs(one - stk) < ((ei > 0.0) ? 2.0e-3 : TOL)); // one sheet and its two halves are one layer
                    }
            }

    // ---- F. Under a closed outer pair without tf ----
    // The far side carries no wave, and without tf the transmit port is exactly zero. The
    // reflect port is the composite: the layer on its closed backing.
    QD_DIAG("\n# diag: flat branch F, S without tf glued on a rigid wall eps 2.401e-9 without tf, the glued entry. |T|: exactly zero.\n");
    QD_DIAG("# dR: the reflect port against the matrix air | S | wall, distance in the complex plane\n");
    QD_DIAG("type,k,theta,|T|,|R|,R_re,R_im,dR\n");
    for (int tr : {1, 4})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (double thd : {20.0, 45.0, 60.0, 80.0})
            {
                ub::Pal P = lump_palette({{eS, 0.0, tS}, {2.401e-9, 0.0, 0.0}});
                std::complex<double> T, R;
                entry(P, 1, tr, thd, k, T, R);
                DiagSheet W = diag_cm_stack(U, U, {{cS, U, tS}}, std::complex<double>(2.401e-9, 0.0), U, std::cos(thd * QPI / 180.0), f, false, tr > 2);
                const double dR = std::abs(R - pick(W, k, true));
                QD_DIAG("%d,%d,%.0f,%.3g,%.12f,%.12f,%.12f,%.2e\n", tr, k, thd, std::abs(T), std::abs(R), std::real(R), std::imag(R), dR);
                INFO("F, type " << tr << ", k " << k << ", theta " << thd);
                CHECK(std::abs(T) == 0.0);
                CHECK(std::abs(std::abs(R) - 1.0) < TOL);
                CHECK(dR < TOL);
            }
        }

    // ---- G. A rigid sheet of the acoustic mapping ----
    // An acoustic material, scalar types. Its permittivity is an impedance mapping, and the mass
    // column is what says so: with it the sheet is a closed mass-law layer and without tf passes
    // nothing. Without the column it is a wave medium like any other and tunnels, on the matrix.
    QD_DIAG("\n# diag: flat branch G, a rigid sheet 22 mm eps 2.1609e-8 without tf, scalar, in air, inside 100 mm + 100 mm of eps 2.50 and as a\n");
    QD_DIAG("# lining of it. T, R: without the mass column, dB; dT, dR: its distance from the matrix (inside: relative, by the\n");
    QD_DIAG("# ratio against a 1 mm sheet of eps 3.0)\n");
    QD_DIAG("where,type,theta,T,R,dT,dR\n");
    for (int where = 0; where < 3; ++where)
        for (int tr : {4, 5})
            for (double thd : {10.0, 30.0, 60.0})
            {
                const std::complex<double> eR(2.1609e-8, 0.0), eH(2.50, 0.0);
                ub::Pal P0 = lump_palette({{2.1609e-8, 0.0, 0.022}, {2.50, 0.0, 0.0}, {3.0, 0.0, 0.001}});
                std::complex<double> T, R, T1, R1;
                double dT, dR;
                if (where == 0)
                {
                    in_air(P0, 1, tr, thd, 0, T, R);
                    DiagSheet W = diag_cm_stack(U, U, {{eR, U, 0.022}}, U, U, std::cos(thd * QPI / 180.0), f, false, true);
                    dT = std::abs(T - W.T_te), dR = std::abs(R - W.R_te);
                }
                else if (where == 1)
                {
                    inside(P0, 1, tr, thd, 0, T, R, 0.1, 0.1), inside(P0, 3, tr, thd, 0, T1, R1, 0.1, 0.1);
                    DiagSheet W = diag_cm_stack(eH, U, {{eR, U, 0.022}}, eH, U, ct_of(0.0, thd), f, false, true);
                    DiagSheet W1 = diag_cm_stack(eH, U, {{cRef, U, 0.001}}, eH, U, ct_of(0.0, thd), f, false, true);
                    dT = rel(T / T1, W.T_te / W1.T_te), dR = rel(R / R1, W.R_te / W1.R_te);
                }
                else
                {
                    entry(P0, 1, tr, thd, 0, T, R);
                    DiagSheet W = diag_cm_stack(U, U, {{eR, U, 0.022}}, eH, U, std::cos(thd * QPI / 180.0), f, false, true);
                    dT = std::abs(T - W.T_te), dR = std::abs(R - W.R_te);
                }
                QD_DIAG("%s,%d,%.0f,%.4f,%.4f,%.2e,%.2e\n", where == 0 ? "air" : (where == 1 ? "inside" : "lining"), tr, thd, dBc(T), dBc(R), dT, dR);
                INFO("G, where " << where << ", type " << tr << ", theta " << thd);
                CHECK(dT < TOL); // without it: a wave medium, on the matrix
                CHECK(dR < TOL);
            }

    // ---- H. Twins ----
    QD_DIAG("\n# diag: flat branch H, twins, the complex coefficient of the sheet event with every digit, scalar and EM (VV, HH). Sheets that are\n");
    QD_DIAG("# denser than the invariant, lossy, below their critical angle, with tf, or under a closed outer pair, beside the rows of this\n");
    QD_DIAG("# case: a diff of the output shows which rows a change reaches; families that sheet tf H prints are not repeated here\n");
    QD_DIAG("row,type,port,theta,re,im,re_hh,im_hh\n");
    {
        const double BIG = 1.0e12;
        ub::Pal P = lump_palette({{5.24, 0.0, 0.006},                                                       // 1 dense lossless sheet
                                  {5.24, 0.63, 0.006},                                                      // 2 dense lossy sheet
                                  {0.31, 0.0, 0.006},                                                       // 3 rare sheet without tf
                                  {0.31, 0.0, 0.006, 0.0, 0.5},                                             // 4 rare sheet with tf
                                  {0.31, 0.20, 0.006},                                                      // 5 lossy rare sheet without tf
                                  {0.31, 0.20, 0.006, 0.0, 0.5},                                            // 6 lossy rare sheet with tf
                                  {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG}, // 7 rigid mass-law sheet with tf
                                  {2.50, 0.80, 0.0},                                                        // 8 lossy host
                                  {2.50, 0.0, 0.0},                                                         // 9 lossless host
                                  {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                           // 10 rigid wall with tf
                                  {2.401e-9, 0.0, 0.0},                                                     // 11 rigid wall without tf
                                  {2.1609e-8, 0.0, 0.022}});                                                // 12 rigid sheet without tf and without the mass column
        struct Row
        {
            const char *name;
            int sheet, host; // host 0: the sheet alone in air; host < 0: the sheet glued on the near face of -host
            std::vector<double> th;
        };
        const std::vector<Row> rows = {
            {"dense lossless no tf in air", 1, 0, {20.0, 45.0, 60.0}},
            {"dense lossless no tf inside lossy host", 1, 8, {20.0, 45.0, 60.0}},
            {"rare no tf below its critical angle - in air", 3, 0, {10.0, 20.0, 30.0}},
            {"rare no tf below its critical angle - lossy host", 3, 8, {10.0, 20.0, 30.0}},
            {"lossy rare no tf inside lossy host", 5, 8, {45.0, 60.0}},
            {"rare tf 0.5 inside lossless host", 4, 9, {45.0, 60.0}},
            {"rare tf 0.5 inside lossy host", 4, 8, {45.0, 60.0}},
            {"leak chain - rare tf 0.5 on rigid wall with tf", 4, -10, {45.0, 60.0}},
            {"closed pair - lossy rare no tf on rigid wall without tf", 5, -11, {45.0, 60.0}},
            {"closed pair - dense lossy no tf on rigid wall without tf", 2, -11, {45.0, 60.0}},
            {"this case - rare no tf in air", 3, 0, {45.0, 60.0}},
            {"this case - rare no tf inside lossy host", 3, 8, {45.0, 60.0}},
            {"this case - rare no tf on lossy host", 3, -8, {45.0, 60.0}},
            {"this case - rare no tf on rigid wall without tf", 3, -11, {45.0, 60.0}},
            {"this case - rigid no column in air", 12, 0, {30.0, 60.0}},
        };
        for (const auto &R : rows)
            for (int tr : {4, 1})
                for (int port = 0; port < 2; ++port)
                    for (double thd : R.th)
                    {
                        arma::uvec mi;
                        arma::mat mesh;
                        std::vector<int> seq;
                        const int last = port ? refl(tr) : tr;
                        if (R.host == 0)
                            mesh = ub::build({0.0}, {R.sheet}, {false}, mi), seq = {last};
                        else if (R.host < 0)
                            mesh = ub::build({0.0, gap, gap + 0.1}, {R.sheet, -R.host, -R.host}, {false, false, true}, mi), seq = {last};
                        else
                            mesh = ub::build({0.0, d1, d1 + d2}, {R.host, R.sheet, R.host}, {false, false, true}, mi), seq = {tr, last};
                        auto a = ub::chain(P, mesh, mi, seq, f, thd, {1.5});
                        const bool ok = a.size() == seq.size();
                        const std::complex<double> z = ok ? cpx(a.back().xpr, 0, 0) : 0.0, zh = (ok && tr == 1) ? cpx(a.back().xpr, 0, 3) : 0.0;
                        QD_DIAG("%s,%d,%s,%.0f,%.17g,%.17g,%.17g,%.17g\n", R.name, tr, port ? "R" : "T", thd, std::real(z), std::imag(z), std::real(zh), std::imag(zh));
                        twin("flat branch H", twin_key("%s,%d,%s,%.0f", R.name, tr, port ? "R" : "T", thd), {std::real(z), std::imag(z), std::real(zh), std::imag(zh)});
                    }
    }
}

TEST_CASE("Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the invariant", "[wave]")
{
    ub::OneThread one_thread;
    // tf moves power between the two ports of a boundary. Between two lossless media that carry a wave that is one
    // statement about amplitudes as well: the impedance step of the face is moved towards its neighbour (tf > 0) or
    // away from it (tf < 0). Where the two sides are not in phase - a lossy sheet, a sheet the wave is evanescent in, a
    // lossy host - the power statement does not fix the amplitudes, and a rule that scales the reflection to (1 - tf)
    // R0 and rescales the transmission to match would create energy: by the quotient (1 - R') / (1 - R0), whose 1 - R0
    // is not the flux a face passes next to a lossy medium; by the forcing factor, which raises a mirror at the phase
    // of a complex reflection; and on a lining, where a tunnelling face would be given a lower mirror and keep its
    // coupling.
    //
    // The rule asserted here: at the face of a coherent sheet tf is an ideal transformer of real ratio. The tangential
    // field is scaled by k and its partner by 1 / k, so the normal flux is continuous whatever the two media are, and
    // the face creates nothing. The ratio comes from the pair rule applied to the modulus of the admittance ratio of
    // the two sides; where that ratio is real it is the pair rule on the reflectance, digit for digit. One expression
    // for a propagating and a tunnelling wave, so nothing is decided on the transverse invariant. Mass-law sheets and a
    // far face under a closed outer pair keep the pair rule.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update. EM materials throughout:
    // tf, no mass column; type 4 is the scalar type on the same materials. The rigid wall of D, G and H is an acoustic
    // material.
    //
    // A. Both ports of one sheet event, in air, inside a host and as a lining: at most one together, the lossless sheet
    // one, and the two powers on the reference of the rule
    // B. A lossless sheet below its critical angle: where the pair rule has it
    // C. A sheet without tf: on the characteristic matrix
    // D. No step where a sheet becomes rarer than the invariant, neighbours 0.02 deg apart
    // E. The end points tf = -1 and tf = +1
    // G. On a rigid wall with tf: both ports, printed
    // S. An ordered stack through the core
    // H. Twins with every digit
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5, k0 = 2.0 * QPI * f / C0, ONE = 1.0 + 1.0e-6, TOL = 1.0e-9;
    const cd U(1.0, 0.0);
    auto val = [](const std::vector<ub::Ev> &v, size_t n, int k)
    { return (v.size() == n) ? cpx(v[n - 1].xpr, 0, k) : cd(0.0, 0.0); };
    auto refl = [](int tr)
    { return (tr == 4 || tr == 5) ? 3 : 0; };
    auto dBp = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    enum
    {
        AIR,
        INSIDE,
        ENTRY,
        EXIT
    };
    // Both ports of the sheet event. Material 1 is the sheet, 2 the host; inside a host and at an
    // exit the event carries its host leg
    auto ports = [&](const ub::Pal &P, int cfg, int tr, double thd, int k, double leg, cd &T, cd &R)
    {
        arma::uvec mi;
        arma::mat mesh;
        std::vector<int> sT, sR;
        if (cfg == AIR)
            mesh = ub::build({0.0}, {1}, {false}, mi), sT = {tr}, sR = {refl(tr)};
        else if (cfg == INSIDE)
            mesh = ub::build({0.0, leg, 2.0 * leg}, {2, 1, 2}, {false, false, true}, mi), sT = {tr, tr}, sR = {tr, refl(tr)};
        else if (cfg == ENTRY)
            mesh = ub::build({0.0, gap, gap + leg}, {1, 2, 2}, {false, false, true}, mi), sT = {tr}, sR = {refl(tr)};
        else
            mesh = ub::build({0.0, leg, leg + gap}, {2, 2, 1}, {false, true, false}, mi), sT = {tr, tr}, sR = {tr, refl(tr)};
        T = val(ub::chain(P, mesh, mi, sT, f, thd, {1.5}), sT.size(), k), R = val(ub::chain(P, mesh, mi, sR, f, thd, {1.5}), sR.size(), k);
    };

    // The reference of the rule, in powers, written in tilted admittances and sharing nothing with
    // the library: eta = kz for TE and the scalar type, eps / kz for TM, with kz = sqrt(eps - p)
    // on the decaying root and p the transverse invariant, sin^2 of the angle in air.
    auto kz_of = [](cd e, double p)
    { cd v = std::sqrt(e - p); return (std::imag(v) > 0.0 || (std::imag(v) == 0.0 && std::real(v) < 0.0)) ? -v : v; };
    struct Fc
    {
        cd r, t, tb;
    };
    // Boundary a | b with a transformer of ratio k = sqrt(nu) in front of b
    auto face_ref = [](cd a, cd b, double tf_a, double tf_b) -> Fc
    {
        const double x0 = std::abs(b / a), rm = (1.0 - x0) / (1.0 + x0), Rm = diag_tf_pair(rm * rm, tf_a, tf_b);
        const double s = (rm < 0.0) ? -std::sqrt(Rm) : std::sqrt(Rm), nu = ((1.0 - s) / (1.0 + s)) / x0, k = std::sqrt(nu);
        const cd r = (a - nu * b) / (a + nu * b);
        return {r, k * (1.0 + r), (1.0 - r) / k};
    };
    // Returns false where a face is within 1e-6 of an admittance ratio of modulus one and its pair forces: the direction a
    // forced face is moved in flips there, as the sign of a forced reflection flips at a matched boundary (table D2)
    auto sheet_ref = [&](cd en, cd es, cd ef, double tf_n, double tf_s, double tf_f, double d, double thd, bool tm, double &Tp, double &Rp) -> bool
    {
        const double sn = std::sin(thd * QPI / 180.0), p = sn * sn;
        auto adm = [&](cd e)
        { return tm ? e / kz_of(e, p) : kz_of(e, p); };
        const cd a = adm(en), s = adm(es), b = adm(ef), phi = std::exp(cd(0.0, -1.0) * k0 * kz_of(es, p) * d);
        const Fc n = face_ref(a, s, tf_n, tf_s), q = face_ref(s, b, tf_s, tf_f);
        const cd den = 1.0 + n.r * q.r * phi * phi;
        Rp = std::norm(n.r + n.t * n.tb * q.r * phi * phi / den), Tp = std::norm(n.t * q.t * phi / den) * std::real(b) / std::real(a);
        const bool tie_n = std::abs(std::abs(s / a) - 1.0) < 1.0e-6 && (tf_n < 0.0 || tf_s < 0.0), tie_f = std::abs(std::abs(b / s) - 1.0) < 1.0e-6 && (tf_s < 0.0 || tf_f < 0.0);
        return !(tie_n || tie_f);
    };

    struct Sh
    {
        double a, ei;
    };
    const std::vector<Sh> sheets = {{0.31, 0.20}, {0.90, 0.20}, {1.50, 0.40}, {5.24, 0.63}, {5.24, 5.0}, {0.31, 0.0}, {0.90, 0.0}, {5.24, 0.0}};

    // ---- A. Both ports of one sheet event ----
    QD_DIAG("\n# diag: sheet tf A, both ports of one sheet event, the worst over 0, 20, 37, 45, 60, 73 and 85 deg and 1, 6 and 25 mm. tf on the\n");
    QD_DIAG("# sheet, or 0.5 on the host. sum: T + R in power, the largest, at theta / mm; above: rows above one / rows. dT, dR: the largest\n");
    QD_DIAG("# distance of a power from the reference of the rule. Inside a host and at an exit the event carries its host leg. Inside the\n");
    QD_DIAG("# lossy host the sum is recorded and each port asserted: next to a lossy medium the incident and the reflected wave interfere, and\n");
    QD_DIAG("# T + R is not what a face conserves there. ch: sc scalar (6 mm only), TE (VV), TM (HH)\n");
    QD_DIAG("where,sheet,tf_on,tf,ch,sum,theta,mm,above,rows,dT,dR\n");
    {
        struct Cf
        {
            const char *name;
            int cfg;
            double ha, hei, leg;
            bool ref;
        };
        const std::vector<Cf> cfgs = {{"air", AIR, 1.0, 0.0, 0.010, true}, {"inside 2.50", INSIDE, 2.50, 0.0, 0.060, true}, {"entry on 2.50", ENTRY, 2.50, 0.0, 0.010, true}, {"exit from 2.50", EXIT, 2.50, 0.0, 0.010, true}, {"entry on 2.50-0.80j", ENTRY, 2.50, 0.80, 0.010, true}, {"inside 2.50-0.80j", INSIDE, 2.50, 0.80, 0.010, false}};
        const double tfs[4][2] = {{0.5, 0.0}, {0.9, 0.0}, {-0.5, 0.0}, {0.0, 0.5}}; // sheet, host
        const char *chn[3] = {"sc", "TE", "TM"};
        for (const Cf &C : cfgs)
            for (const Sh &S : sheets)
                for (int it = 0; it < 4; ++it)
                {
                    if (C.cfg == AIR && it == 3)
                        continue;
                    const double tS = tfs[it][0], tH = tfs[it][1];
                    double w_sum[3] = {0.0, 0.0, 0.0}, w_th[3] = {0.0, 0.0, 0.0}, w_mm[3] = {0.0, 0.0, 0.0}, w_dT[3] = {0.0, 0.0, 0.0}, w_dR[3] = {0.0, 0.0, 0.0};
                    int n_above[3] = {0, 0, 0}, n_rows[3] = {0, 0, 0};
                    for (double thk : {0.001, 0.006, 0.025})
                    {
                        if (thk > 0.8 * C.leg && C.cfg == INSIDE)
                            continue; // the sheet has to fit its host
                        ub::Pal P = lump_palette({{S.a, S.ei, thk, 0.0, tS}, {C.ha, C.hei, 0.0, 0.0, tH}});
                        const cd eS(S.a, -S.ei), eH(C.ha, -C.hei);
                        const cd en = (C.cfg == INSIDE || C.cfg == EXIT) ? eH : U, ef = (C.cfg == INSIDE || C.cfg == ENTRY) ? eH : U;
                        const double tn = (C.cfg == INSIDE || C.cfg == EXIT) ? tH : 0.0, tff = (C.cfg == INSIDE || C.cfg == ENTRY) ? tH : 0.0;
                        for (int tr : {1, 4})
                        {
                            if (tr == 4 && thk != 0.006)
                                continue;
                            for (double thd : {0.0, 20.0, 37.0, 45.0, 60.0, 73.0, 85.0})
                                for (int k : {0, 3})
                                {
                                    if (tr == 4 && k)
                                        continue;
                                    const int c = (tr == 4) ? 0 : (k ? 2 : 1);
                                    cd T, R;
                                    ports(P, C.cfg, tr, thd, k, C.leg, T, R);
                                    const double pT = std::norm(T), pR = std::norm(R), sum = pT + pR;
                                    ++n_rows[c];
                                    if (sum > ONE)
                                        ++n_above[c];
                                    if (sum > w_sum[c])
                                        w_sum[c] = sum, w_th[c] = thd, w_mm[c] = 1.0e3 * thk;
                                    INFO("A, " << C.name << ", sheet " << S.a << " - " << S.ei << "j, tf " << tS << " / " << tH << ", " << 1.0e3 * thk << " mm, type " << tr << ", k " << k << ", theta " << thd);
                                    CHECK(std::isfinite(sum));
                                    if (C.ref)
                                        CHECK(sum < ONE); // the sheet creates nothing
                                    else
                                        CHECK((pT < ONE && pR < ONE)); // in a lossy host the sum is printed only, see the header
                                    if (S.ei == 0.0 && C.hei == 0.0)
                                        CHECK(std::abs(sum - 1.0) < 1.0e-6); // and a lossless sheet between lossless media loses nothing, tunnelling or not
                                    if (C.ref)
                                    {
                                        double Tp, Rp;
                                        if (sheet_ref(en, eS, ef, tn, tS, tff, thk, thd, k != 0, Tp, Rp))
                                        {
                                            w_dT[c] = std::max(w_dT[c], std::abs(pT - Tp)), w_dR[c] = std::max(w_dR[c], std::abs(pR - Rp));
                                            CHECK(std::abs(pT - Tp) < TOL); // the rule, on both ports
                                            CHECK(std::abs(pR - Rp) < TOL);
                                        }
                                    }
                                }
                        }
                    }
                    for (int c = 0; c < 3; ++c)
                        QD_DIAG("%s,%.2f-%.2fj,%s,%.1f,%s,%.6f,%.0f,%.0f,%d,%d,%.2e,%.2e\n", C.name, S.a, S.ei, it == 3 ? "host" : "sheet", it == 3 ? tH : tS, chn[c], w_sum[c], w_th[c],
                                w_mm[c], n_above[c], n_rows[c], w_dT[c], w_dR[c]);
                }
    }

    // ---- B. A lossless sheet below its critical angle, and C. a sheet without tf ----
    QD_DIAG("\n# diag: sheet tf B, a lossless 6 mm sheet in air below its critical angle against the pair rule on the reflectance\n");
    QD_DIAG("# (diag_sheet_composite), and C, a sheet without tf in air and on eps 2.50 - 0.80j against the characteristic matrix; the largest\n");
    QD_DIAG("# distance in the complex plane over the angles, both ports, scalar, TE and TM\n");
    QD_DIAG("table,sheet,tf,where,angles,dT,dR\n");
    for (const Sh &S : sheets)
    {
        const cd eS(S.a, -S.ei), eH(2.50, -0.80);
        if (S.ei == 0.0)
            for (double tS : {0.5, 0.9, -0.5, -0.9})
            {
                ub::Pal P = lump_palette({{S.a, 0.0, 0.006, 0.0, tS}, {2.50, 0.0, 0.0}});
                double dT = 0.0, dR = 0.0;
                int n = 0;
                for (double thd : {5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 85.0})
                {
                    if (std::sin(thd * QPI / 180.0) * std::sin(thd * QPI / 180.0) >= S.a - 0.02)
                        continue; // at or past the critical angle of the sheet
                    ++n;
                    for (int tr : {4, 1})
                        for (int k : {0, 3})
                        {
                            if (tr == 4 && k)
                                continue;
                            cd T, R;
                            ports(P, AIR, tr, thd, k, 0.010, T, R);
                            DiagLump lump;
                            lump.tf = tS;
                            DiagSheet W = diag_sheet_composite(U, eS, U, 0.006, std::cos(thd * QPI / 180.0), f, false, tr == 4, U, U, U, lump);
                            const double a = std::abs(T - (k ? W.T_tm : W.T_te)), b = std::abs(R - (k ? W.R_tm : W.R_te));
                            dT = std::max(dT, a), dR = std::max(dR, b);
                            INFO("B, sheet " << S.a << ", tf " << tS << ", type " << tr << ", k " << k << ", theta " << thd);
                            CHECK(a < TOL); // where the two sides are in phase the rule is the pair rule
                            CHECK(b < TOL);
                        }
                }
                QD_DIAG("B,%.2f,%.1f,air,%d,%.2e,%.2e\n", S.a, tS, n, dT, dR);
            }
        for (int where = 0; where < 2; ++where)
        {
            ub::Pal P = lump_palette({{S.a, S.ei, 0.006}, {2.50, 0.80, 0.0}});
            double dT = 0.0, dR = 0.0;
            int n = 0;
            for (double thd : {5.0, 20.0, 37.0, 45.0, 60.0, 73.0, 85.0})
            {
                ++n;
                for (int tr : {4, 1})
                    for (int k : {0, 3})
                    {
                        if (tr == 4 && k)
                            continue;
                        cd T, R;
                        ports(P, where ? ENTRY : AIR, tr, thd, k, 0.010, T, R);
                        DiagSheet W = diag_cm_stack(U, U, {{eS, U, 0.006}}, where ? eH : U, U, cd(std::cos(thd * QPI / 180.0), 0.0), f, false, tr == 4);
                        const double a = std::abs(T - (k ? W.T_tm : W.T_te)), b = std::abs(R - (k ? W.R_tm : W.R_te));
                        dT = std::max(dT, a), dR = std::max(dR, b);
                        INFO("C, sheet " << S.a << " - " << S.ei << "j, where " << where << ", type " << tr << ", k " << k << ", theta " << thd);
                        CHECK(a < TOL); // a sheet without tf is on the matrix and stays on it
                        CHECK(b < TOL);
                    }
            }
            QD_DIAG("C,%.2f-%.2fj,0.0,%s,%d,%.2e,%.2e\n", S.a, S.ei, where ? "entry on 2.50-0.80j" : "air", n, dT, dR);
        }
    }

    // ---- D. No step where a sheet becomes rarer than the invariant ----
    // The transmit port in dB. The reflect port sits at -18 dB to -52 dB on these rows, with a
    // slope of its own of up to 0.016 dB per 0.02 deg and a kink at the critical angle of a
    // lossless sheet, so it is asserted in amplitude against the same pair of angles with no tf
    // anywhere, which is the matrix and has that slope and that kink.
    QD_DIAG("\n# diag: sheet tf D, neighbours about the angle at which a sheet becomes rarer than the invariant, |eps| = sin^2: 37.40 deg for\n");
    QD_DIAG("# eps 0.31 - 0.20j, 73.78 deg for 0.90 - 0.20j, and the critical angle 33.83 deg of the lossless eps 0.31. T, R: dB at the first\n");
    QD_DIAG("# angle; dT: dB between the two; dR: the same in amplitude; dR_0: dR with no tf anywhere\n");
    QD_DIAG("row,ch,theta_a,theta_b,T,R,dT,dR,dR_0\n");
    {
        struct St
        {
            const char *name;
            int cfg;
            LumpMtl S, H;
            double ta, tb, leg;
            bool leak; // under a closed outer pair
        };
        const std::vector<St> rows = {
            {"porous tf 0.5 in air", AIR, {0.31, 0.20, 0.006, 0.0, 0.5}, {2.50, 0.0, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous tf -0.5 in air", AIR, {0.31, 0.20, 0.006, 0.0, -0.5}, {2.50, 0.0, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous tf 0.5 inside 2.50", INSIDE, {0.31, 0.20, 0.006, 0.0, 0.5}, {2.50, 0.0, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous tf 0.5 inside 2.50-0.80j", INSIDE, {0.31, 0.20, 0.006, 0.0, 0.5}, {2.50, 0.80, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous tf 0.5 entry on 2.50", ENTRY, {0.31, 0.20, 0.006, 0.0, 0.5}, {2.50, 0.0, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous tf 0.5 exit from 2.50", EXIT, {0.31, 0.20, 0.006, 0.0, 0.5}, {2.50, 0.0, 0.0}, 37.39, 37.41, 0.010, false},
            {"porous without tf inside 2.50 with tf 0.5", INSIDE, {0.31, 0.20, 0.006}, {2.50, 0.0, 0.0, 0.0, 0.5}, 37.39, 37.41, 0.010, false},
            {"0.90-0.20j tf 0.9 in air", AIR, {0.90, 0.20, 0.006, 0.0, 0.9}, {2.50, 0.0, 0.0}, 73.77, 73.79, 0.010, false},
            {"0.90-0.20j tf 0.9 inside 2.50", INSIDE, {0.90, 0.20, 0.006, 0.0, 0.9}, {2.50, 0.0, 0.0}, 73.77, 73.79, 0.010, false},
            {"lossless 0.31 tf 0.5 inside 2.50-0.80j", INSIDE, {0.31, 0.0, 0.006, 0.0, 0.5}, {2.50, 0.80, 0.0}, 33.83, 33.834, 0.010, false},
        };
        for (const St &W : rows)
            for (int tr : {4, 1})
                for (int k : {0, 3})
                {
                    if (tr == 4 && k)
                        continue;
                    ub::Pal P = lump_palette({W.S, W.H});
                    cd Ta, Ra, Tb, Rb;
                    ports(P, W.cfg, tr, W.ta, k, W.leg, Ta, Ra), ports(P, W.cfg, tr, W.tb, k, W.leg, Tb, Rb);
                    LumpMtl S0 = W.S, H0 = W.H;
                    S0.tf = 0.0, H0.tf = 0.0;
                    ub::Pal P0 = lump_palette({S0, H0});
                    cd Tx, Ra0, Rb0;
                    ports(P0, W.cfg, tr, W.ta, k, W.leg, Tx, Ra0), ports(P0, W.cfg, tr, W.tb, k, W.leg, Tx, Rb0);
                    const double dT = dBp(std::norm(Tb)) - dBp(std::norm(Ta)), dR = std::abs(Rb) - std::abs(Ra), dR0 = std::abs(Rb0) - std::abs(Ra0);
                    QD_DIAG("%s,%s,%.3f,%.3f,%.4f,%.4f,%+.5f,%+.2e,%+.2e\n", W.name, tr == 4 ? "sc" : (k ? "TM" : "TE"), W.ta, W.tb, dBp(std::norm(Ta)), dBp(std::norm(Ra)), dT, dR, dR0);
                    INFO("D, " << W.name << ", type " << tr << ", k " << k);
                    CHECK(std::abs(dT) < 0.01);             // no step on the transmit port
                    if (W.S.ei > 0.0)                       // the critical angle of a lossless sheet is a kink of its own, with tf and without: printed
                        CHECK(std::abs(dR - dR0) < 1.0e-3); // nor on the reflect port
                    if (!W.leak)
                        CHECK(std::norm(Ta) + std::norm(Ra) < ONE);
                }
    }

    // Where the admittance ratio of a face has modulus one and tf forces, the direction the face is moved in flips:
    // away from its neighbour is up on one side and down on the other. Between media in phase that is a matched
    // boundary, where a forced reflection changes sign and nothing else; where they are not in phase the two ports
    // step. Printed; each side creates nothing. eps 0.90 - 0.20j in air: modulus one at 60 deg in TE and the scalar
    // type.
    QD_DIAG("\n# diag: sheet tf D2, recorded: tf < 0 on a 6 mm sheet of eps 0.90 - 0.20j in air about 60 deg, where the admittance ratio of its\n");
    QD_DIAG("# faces has modulus one in TE. T, R: dB at the first angle; dT, dR: dB between the two; sum: T + R at the first angle\n");
    QD_DIAG("tf,ch,theta_a,theta_b,T,R,dT,dR,sum\n");
    for (double tS : {-0.1, -0.5, -0.9, 0.5})
        for (int tr : {4, 1})
            for (int k : {0, 3})
            {
                if (tr == 4 && k)
                    continue;
                ub::Pal P = lump_palette({{0.90, 0.20, 0.006, 0.0, tS}, {2.50, 0.0, 0.0}});
                cd Ta, Ra, Tb, Rb;
                ports(P, AIR, tr, 59.99, k, 0.010, Ta, Ra), ports(P, AIR, tr, 60.01, k, 0.010, Tb, Rb);
                QD_DIAG("%.1f,%s,59.99,60.01,%.4f,%.4f,%+.4f,%+.4f,%.6f\n", tS, tr == 4 ? "sc" : (k ? "TM" : "TE"), dBp(std::norm(Ta)), dBp(std::norm(Ra)), dBp(std::norm(Tb)) - dBp(std::norm(Ta)),
                        dBp(std::norm(Rb)) - dBp(std::norm(Ra)), std::norm(Ta) + std::norm(Ra));
                INFO("D2, tf " << tS << ", type " << tr << ", k " << k);
                CHECK(std::norm(Ta) + std::norm(Ra) < ONE);
                CHECK(std::norm(Tb) + std::norm(Rb) < ONE);
            }

    // ---- E. The end points ----
    // tf = -1 closes the transmit port of every sheet. tf = +1 closes the reflect port where the two sides of a face
    // are in phase; where they are not, a real ratio cannot match the face and what is left is printed.
    QD_DIAG("\n# diag: sheet tf E, the end points, 6 mm sheet in air and inside eps 2.50, power, the largest over 0, 45 and 75 deg, scalar, TE and TM\n");
    QD_DIAG("sheet,where,T_at_tf-1,R_at_tf+1\n");
    for (const Sh &S : sheets)
        for (int cfg : {(int)AIR, (int)INSIDE})
        {
            double wT = 0.0, wR = 0.0;
            for (double tS : {-1.0, 1.0})
            {
                ub::Pal P = lump_palette({{S.a, S.ei, 0.006, 0.0, tS}, {2.50, 0.0, 0.0}});
                for (int tr : {4, 1})
                    for (int k : {0, 3})
                        for (double thd : {0.0, 45.0, 75.0})
                        {
                            if (tr == 4 && k)
                                continue;
                            cd T, R;
                            ports(P, cfg, tr, thd, k, 0.010, T, R);
                            INFO("E, sheet " << S.a << " - " << S.ei << "j, cfg " << cfg << ", tf " << tS << ", type " << tr << ", k " << k << ", theta " << thd);
                            CHECK(std::norm(T) + std::norm(R) < ONE);
                            if (tS < 0.0)
                            {
                                wT = std::max(wT, std::norm(T));
                                CHECK(std::norm(T) < 1.0e-24); // forced: nothing passes
                            }
                            else
                            {
                                wR = std::max(wR, std::norm(R));
                                if (S.ei == 0.0 && S.a > 1.0)
                                    CHECK(std::norm(R) < 1.0e-24); // matched: nothing comes back
                                CHECK(std::norm(R) < 0.02);        // and little where the face cannot be matched
                            }
                        }
            }
            QD_DIAG("%.2f-%.2fj,%s,%.3e,%.3e\n", S.a, S.ei, cfg == AIR ? "air" : "inside 2.50", wT, wR);
        }

    // ---- S. An ordered stack through the core ----
    QD_DIAG("\n# diag: sheet tf S, the ordered stack A | B in air, 6 mm each, tf on A, both orders, the worst T + R over 0, 30, 45, 60, 75 and 85 deg\n");
    QD_DIAG("A,B,tf,order,ch,sum,theta,above\n");
    {
        const std::vector<std::pair<Sh, Sh>> pairs = {{{5.24, 0.63}, {1.50, 0.40}}, {{0.31, 0.20}, {5.24, 0.63}}, {{0.90, 0.20}, {2.50, 0.0}}, {{0.31, 0.0}, {2.50, 0.0}}, {{5.24, 0.0}, {2.50, 0.0}}};
        for (const auto &pr : pairs)
            for (double tA : {0.5, 0.9, -0.5})
                for (int order = 0; order < 2; ++order)
                {
                    ub::Pal P = lump_palette({{pr.first.a, pr.first.ei, 0.006, 0.0, tA}, {pr.second.a, pr.second.ei, 0.006}});
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, gap}, {order ? 2 : 1, order ? 1 : 2}, {false, false}, mi);
                    double w[3] = {0.0, 0.0, 0.0}, wt[3] = {0.0, 0.0, 0.0};
                    int na[3] = {0, 0, 0};
                    for (int tr : {4, 1})
                        for (int k : {0, 3})
                            for (double thd : {0.0, 30.0, 45.0, 60.0, 75.0, 85.0})
                            {
                                if (tr == 4 && k)
                                    continue;
                                const int c = (tr == 4) ? 0 : (k ? 2 : 1);
                                const cd T = val(ub::chain(P, mesh, mi, {tr}, f, thd, {1.5}), 1, k), R = val(ub::chain(P, mesh, mi, {refl(tr)}, f, thd, {1.5}), 1, k);
                                const double sum = std::norm(T) + std::norm(R);
                                if (sum > w[c])
                                    w[c] = sum, wt[c] = thd;
                                if (sum > ONE)
                                    ++na[c];
                                INFO("S, A " << pr.first.a << " - " << pr.first.ei << "j, B " << pr.second.a << " - " << pr.second.ei << "j, tf " << tA << ", order " << order << ", type " << tr << ", k " << k << ", theta " << thd);
                                CHECK(sum < ONE);
                                if (pr.first.ei == 0.0 && pr.second.ei == 0.0)
                                    CHECK(std::abs(sum - 1.0) < 1.0e-6);
                            }
                    const char *chn[3] = {"sc", "TE", "TM"};
                    for (int c = 0; c < 3; ++c)
                        QD_DIAG("%.2f-%.2fj,%.2f-%.2fj,%.1f,%s,%s,%.6f,%.0f,%d\n", pr.first.a, pr.first.ei, pr.second.a, pr.second.ei, tA, order ? "B|A" : "A|B", chn[c], w[c], wt[c], na[c]);
                }
    }

    // ---- H. Twins ----
    QD_DIAG("\n# diag: sheet tf H, twins, the complex coefficient of the sheet event with every digit, scalar and EM (VV, HH): sheets with tf\n");
    QD_DIAG("# whose faces are in phase, sheets with tf whose faces are not, sheets without tf, sheets under a closed outer pair and\n");
    QD_DIAG("# ordered stacks. A diff of the output shows which rows a change to a face reaches\n");
    QD_DIAG("row,type,port,theta,re,im,re_hh,im_hh\n");
    {
        const double BIG = 1.0e12;
        ub::Pal P = lump_palette({{5.24, 0.0, 0.006, 0.0, 0.5},                                             // 1 dense lossless sheet, tf 0.5
                                  {5.24, 0.0, 0.006, 0.0, -0.5},                                            // 2 dense lossless sheet, tf -0.5
                                  {5.24, 0.63, 0.006, 0.0, 0.5},                                            // 3 dense lossy sheet, tf 0.5
                                  {5.24, 0.63, 0.006, 0.0, -0.5},                                           // 4 dense lossy sheet, tf -0.5
                                  {0.31, 0.20, 0.006, 0.0, 0.5},                                            // 5 porous sheet, tf 0.5
                                  {0.31, 0.0, 0.006, 0.0, 0.5},                                             // 6 rare lossless sheet, tf 0.5
                                  {5.24, 0.63, 0.006},                                                      // 7 dense lossy sheet without tf
                                  {0.31, 0.20, 0.006},                                                      // 8 porous sheet without tf
                                  {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG}, // 9 rigid mass-law sheet with tf
                                  {2.50, 0.80, 0.0},                                                        // 10 lossy host
                                  {2.50, 0.0, 0.0},                                                         // 11 lossless host
                                  {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                           // 12 rigid wall with tf
                                  {2.50, 0.0, 0.0, 0.0, 0.5},                                               // 13 lossless host with tf
                                  {1.50, 0.40, 0.006},                                                      // 14 a second sheet without tf
                                  {2.50, 0.0, 0.006}});                                                     // 15 a lossless second sheet without tf
        struct Row
        {
            const char *name;
            int sheet, host; // host 0: alone in air; host < 0: glued on the near face of -host; host > 100: the ordered stack sheet | host - 100 in air
            std::vector<double> th;
        };
        const std::vector<Row> rows = {
            {"in phase - dense lossless tf 0.5 in air", 1, 0, {20.0, 45.0, 60.0}},
            {"in phase - dense lossless tf -0.5 in air", 2, 0, {20.0, 45.0, 60.0}},
            {"in phase - dense lossless tf 0.5 inside lossless host", 1, 11, {20.0, 45.0, 60.0}},
            {"in phase - dense lossless tf 0.5 on lossless host", 1, -11, {20.0, 45.0, 60.0}},
            {"in phase - rare tf 0.5 below its critical angle in air", 6, 0, {10.0, 20.0, 30.0}},
            {"in phase - dense lossless tf 0.5 | lossless second sheet as a stack", 1, 115, {20.0, 45.0}},
            {"not in phase - dense lossy tf 0.5 in air", 3, 0, {20.0, 45.0, 60.0}},
            {"not in phase - dense lossy tf -0.5 in air", 4, 0, {20.0, 45.0, 60.0}},
            {"not in phase - dense lossy tf 0.5 inside lossy host", 3, 10, {20.0, 45.0, 60.0}},
            {"not in phase - dense lossless tf 0.5 inside lossy host", 1, 10, {20.0, 45.0, 60.0}},
            {"not in phase - dense lossy tf 0.5 on lossless host", 3, -11, {20.0, 45.0, 60.0}},
            {"not in phase - dense lossy no tf inside host with tf", 7, 13, {20.0, 45.0, 60.0}},
            {"not in phase - porous tf 0.5 in air", 5, 0, {20.0, 45.0, 60.0}},
            {"not in phase - rare tf 0.5 past its critical angle in air", 6, 0, {45.0, 60.0}},
            {"not in phase - rare tf 0.5 on lossless host", 6, -11, {45.0, 60.0}},
            {"not in phase - dense lossy tf 0.5 | second sheet as a stack", 3, 114, {20.0, 45.0}},
            {"no tf - dense lossy in air", 7, 0, {20.0, 45.0, 60.0}},
            {"no tf - dense lossy inside lossy host", 7, 10, {20.0, 45.0, 60.0}},
            {"no tf - porous in air", 8, 0, {20.0, 45.0, 60.0}},
            {"no tf - porous on lossless host", 8, -11, {20.0, 45.0, 60.0}},
            {"no tf - dense lossy | second sheet as a stack", 7, 114, {20.0, 45.0}},
            {"closed pair - dense lossy tf 0.5 on rigid wall with tf", 3, -12, {20.0, 45.0, 60.0}},
            {"closed pair - porous tf 0.5 on rigid wall with tf", 5, -12, {20.0, 45.0, 60.0}},
            {"closed pair - dense lossy no tf on rigid wall with tf", 7, -12, {20.0, 45.0, 60.0}},
        };
        for (const auto &R : rows)
            for (int tr : {4, 1})
                for (int port = 0; port < 2; ++port)
                    for (double thd : R.th)
                    {
                        arma::uvec mi;
                        arma::mat mesh;
                        std::vector<int> seq;
                        const int last = port ? refl(tr) : tr;
                        if (R.host == 0)
                            mesh = ub::build({0.0}, {R.sheet}, {false}, mi), seq = {last};
                        else if (R.host > 100)
                            mesh = ub::build({0.0, gap}, {R.sheet, R.host - 100}, {false, false}, mi), seq = {last};
                        else if (R.host < 0)
                            mesh = ub::build({0.0, gap, gap + 0.1}, {R.sheet, -R.host, -R.host}, {false, false, true}, mi), seq = {last};
                        else
                            mesh = ub::build({0.0, 0.010, 0.022}, {R.host, R.sheet, R.host}, {false, false, true}, mi), seq = {tr, last};
                        auto a = ub::chain(P, mesh, mi, seq, f, thd, {1.5});
                        const bool ok = a.size() == seq.size();
                        const cd z = ok ? cpx(a.back().xpr, 0, 0) : 0.0, zh = (ok && tr == 1) ? cpx(a.back().xpr, 0, 3) : 0.0;
                        QD_DIAG("%s,%d,%s,%.0f,%.17g,%.17g,%.17g,%.17g\n", R.name, tr, port ? "R" : "T", thd, std::real(z), std::imag(z), std::real(zh), std::imag(zh));
                        twin("sheet tf H", twin_key("%s,%d,%s,%.0f", R.name, tr, port ? "R" : "T", thd), {std::real(z), std::imag(z), std::real(zh), std::imag(zh)});
                    }
    }
}

TEST_CASE("Bare tf - a bare face with tf is a transformer where it is open and a flat leak where it is closed", "[wave]")
{
    ub::OneThread one_thread;
    // tf at a bare boundary, the face of a traced solid. A face with a wave behind it is the ideal
    // transformer the face of a coherent sheet is, per channel, and the mirrors of the solid's
    // series follow it; a closed face keeps the flat leak. Measured as the two ports of a wall,
    // each a sum over every path, with the series summed and with every bounce followed.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5;
    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    auto mk = [](double a, double ei, double thk, double tf, double m = 0.0, double fRef = 1.0e9)
    {
        LumpMtl L;
        L.a = a, L.ei = ei, L.thk = thk, L.tf = tf, L.m = m, L.fRef = fRef;
        return L;
    };
    auto ports = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int tr, double thd, bool follow, int maxdepth = 400)
    {
        bf::Acc A;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        bf::walk(P, mesh, mi, tr, f, thd, follow ? 1.5 : 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, maxdepth, A);
        return A;
    };
    auto event = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int it, double thd, int k) -> cd
    {
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        ub::Ev E = ub::traced<double>(P, mesh, mi, it, f, o, d, nullptr);
        return (E.nh == 0 || E.rt == 0) ? cd(0.0, 0.0) : cpx(E.xpr, 0, k);
    };
    struct Mt
    {
        const char *name;
        double a, ei;
    };
    const std::vector<Mt> mats = {{"5.24-0.63j", 5.24, 0.63}, {"2.50-0.80j", 2.50, 0.80}, {"1.50-0.40j", 1.50, 0.40}, {"0.90-0.20j", 0.90, 0.20}, {"0.31-0.20j", 0.31, 0.20}, {"5.24", 5.24, 0.0}, {"2.50", 2.50, 0.0}};

    QD_DIAG("\n# diag: bare tf A, one traced solid with tf in air, both ports as sums over every path, series summed (sum) and every bounce\n");
    QD_DIAG("# followed (fol). Worst T + R over 0 to 85 deg and 1, 6, 50 mm, with its row; above: rows above 1.0005 of 18 (followed: 9). Scalar: one channel\n");
    QD_DIAG("solid,tf,type,mode,worst,at_mm,at_deg,above,worst_hh,at_mm,at_deg,above_hh,least\n");
    for (const Mt &M : mats)
        for (double tf : {0.5, 0.9, -0.5})
            for (int tr : {4, 5, 1, 2})
                for (int follow = 0; follow < 2; ++follow)
                {
                    double w[2] = {0.0, 0.0}, wl[2] = {0.0, 0.0}, wt[2] = {0.0, 0.0}, least = 1.0e300;
                    int above[2] = {0, 0};
                    for (double L : {0.001, 0.006, 0.050})
                        for (double thd : follow ? std::vector<double>{0.0, 45.0, 85.0} : std::vector<double>{0.0, 20.0, 45.0, 60.0, 75.0, 85.0})
                        {
                            ub::Pal P = lump_palette({mk(M.a, M.ei, 0.0, tf)});
                            arma::uvec mi;
                            arma::mat mesh = ub::build({0.0, L}, {1, 1}, {false, true}, mi);
                            bf::Acc A = ports(P, mesh, mi, tr, thd, follow != 0);
                            for (int c = 0; c < ((tr > 2) ? 1 : 2); ++c)
                            {
                                const double s = std::norm(A.T[c]) + std::norm(A.R[c]);
                                if (s > w[c])
                                    w[c] = s, wl[c] = L, wt[c] = thd;
                                above[c] += (s > 1.0005);
                                least = std::min(least, s);
                                INFO("solid " << M.name << ", tf " << tf << ", type " << tr << ", follow " << follow << ", " << 1.0e3 * L << " mm, theta " << thd << ", channel " << c);
                                CHECK(s < 1.0005); // the solid passes and returns no more than it received, in each channel
                                if (M.ei == 0.0)
                                    CHECK(s > 0.999); // and a lossless one loses nothing
                            }
                        }
                    QD_DIAG("%s,%.1f,%d,%s,%.6g,%.0f,%.0f,%d,%.6g,%.0f,%.0f,%d,%.6f\n", M.name, tf, tr, follow ? "fol" : "sum", w[0], 1.0e3 * wl[0], wt[0], above[0], w[1],
                            1.0e3 * wl[1], wt[1], above[1], least);
                }

    QD_DIAG("\n# diag: bare tf B, one face, air | solid with tf, the entry event: transmitted and reflected power per channel, EM (VV, HH) and\n");
    QD_DIAG("# scalar. 66.4 deg is the Brewster angle of eps 5.24: the last column is recorded there, a forced reflection in HH\n");
    QD_DIAG("solid,tf,theta,T_vv,R_vv,sum_vv,T_hh,R_hh,sum_hh,T_sc,R_sc,sum_sc\n");
    for (const Mt &M : mats)
        for (double tf : {0.5, 0.9, -0.5})
            for (double thd : {20.0, 60.0, 66.4, 75.0})
            {
                ub::Pal P = lump_palette({mk(M.a, M.ei, 0.0, tf)});
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0}, {1}, {false}, mi);
                const double Tv = std::norm(event(P, mesh, mi, 1, thd, 0)), Rv = std::norm(event(P, mesh, mi, 0, thd, 0));
                const double Th = std::norm(event(P, mesh, mi, 1, thd, 3)), Rh = std::norm(event(P, mesh, mi, 0, thd, 3));
                const double Ts = std::norm(event(P, mesh, mi, 4, thd, 0)), Rs = std::norm(event(P, mesh, mi, 3, thd, 0));
                QD_DIAG("%s,%.1f,%.1f,%.6f,%.6f,%.9f,%.6f,%.6f,%.9f,%.6f,%.6f,%.9f\n", M.name, tf, thd, Tv, Rv, Tv + Rv, Th, Rh, Th + Rh, Ts, Rs, Ts + Rs);
                INFO("face, solid " << M.name << ", tf " << tf << ", theta " << thd);
                // Seen from a lossless medium a face passes what it does not reflect, in each channel
                CHECK(std::abs(Tv + Rv - 1.0) < 1.0e-9);
                CHECK(std::abs(Th + Rh - 1.0) < 1.0e-9);
                CHECK(std::abs(Ts + Rs - 1.0) < 1.0e-9);
            }

    QD_DIAG("\n# diag: bare tf C, a layer with tf in air as a traced solid (every path, series summed) against the same layer as a sheet, dB,\n");
    QD_DIAG("# 6 mm. d: solid less sheet. Scalar and EM (VV, HH). 65 deg and not 60: eps 0.90 - 0.20j has an admittance ratio of modulus\n");
    QD_DIAG("# one at 60 deg, where rounding decides the side of a forced face (table H)\n");
    QD_DIAG("layer,tf,type,theta,T_solid,T_sheet,d_T,R_solid,R_sheet,d_R,d_T_hh,d_R_hh\n");
    for (const Mt &M : mats)
        for (double tf : {0.5, -0.5})
            for (int tr : {4, 1})
                for (double thd : {10.0, 30.0, 45.0, 65.0, 75.0})
                {
                    const double L = 0.006;
                    ub::Pal P = lump_palette({mk(M.a, M.ei, 0.0, tf), mk(M.a, M.ei, L, tf)});
                    arma::uvec mi, ms;
                    arma::mat mesh = ub::build({0.0, L}, {1, 1}, {false, true}, mi), sheet = ub::build({0.0}, {2}, {false}, ms);
                    bf::Acc A = ports(P, mesh, mi, tr, thd, false), B = ports(P, sheet, ms, tr, thd, false);
                    double d[4];
                    for (int c = 0; c < 2; ++c)
                        d[2 * c] = dB(std::norm(A.T[c])) - dB(std::norm(B.T[c])), d[2 * c + 1] = dB(std::norm(A.R[c])) - dB(std::norm(B.R[c]));
                    QD_DIAG("%s,%.1f,%d,%.0f,%.4f,%.4f,%+.5f,%.4f,%.4f,%+.5f,%+.5f,%+.5f\n", M.name, tf, tr, thd, dB(std::norm(A.T[0])), dB(std::norm(B.T[0])), d[0],
                            dB(std::norm(A.R[0])), dB(std::norm(B.R[0])), d[1], d[2], d[3]);
                    INFO("solid against sheet, " << M.name << ", tf " << tf << ", type " << tr << ", theta " << thd);
                    for (int q = 0; q < ((tr > 2) ? 2 : 4); ++q)
                        if (q % 2 == 0 || std::norm(q < 2 ? B.R[0] : B.R[1]) > 1.0e-4) // a reflect port near a null has no level to compare
                            CHECK(std::abs(d[q]) < 2.0e-3);                            // one wall, at the relaunch bound
                }

    QD_DIAG("\n# diag: bare tf D, a sheet of air (eps 1, 6 mm) on a wall with tf against the bare wall, the entry event, both ports: the\n");
    QD_DIAG("# largest distance in the complex plane over scalar, VV and HH\n");
    QD_DIAG("wall,tf,theta,d_T,d_R\n");
    for (const Mt &M : mats)
        for (double tf : {0.5, -0.5})
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                ub::Pal P = lump_palette({mk(M.a, M.ei, 0.0, tf), mk(1.0, 0.0, 0.006, 0.0)});
                arma::uvec mb, mg;
                arma::mat bare = ub::build({0.0}, {1}, {false}, mb), glued = ub::build({0.0, gap}, {2, 1}, {false, false}, mg);
                double dT = 0.0, dR = 0.0;
                for (int tr : {4, 1})
                    for (int k : {0, 3})
                        if (tr == 1 || k == 0)
                        {
                            dT = std::max(dT, std::abs(event(P, glued, mg, tr, thd, k) - event(P, bare, mb, tr, thd, k)));
                            if (thd > 0.0 || tr == 4) // the EM reflect port at exactly normal incidence has its own basis
                                dR = std::max(dR, std::abs(event(P, glued, mg, tr - 1, thd, k) - event(P, bare, mb, tr - 1, thd, k)));
                        }
                QD_DIAG("%s,%.1f,%.0f,%.2e,%.2e\n", M.name, tf, thd, dT, dR);
                INFO("sheet of air on " << M.name << ", tf " << tf << ", theta " << thd);
                CHECK(dT < 1.0e-9); // a sheet of air on a wall is the wall
                CHECK(dR < 1.0e-9);
            }

    QD_DIAG("\n# diag: bare tf E, the acoustic main road: a rigid wall 200 mm, eps 2.401e-9, with tf, without the mass column,\n");
    QD_DIAG("# scalar, tracer 4 and 5. R: reflectance of the first face; T: what enters; both with every digit\n");
    QD_DIAG("m,tf,type,theta,R,T,R+tf-1,T-tf\n");
    for (double m : {0.0})
        for (double tf : {0.1, 0.3, 0.9})
            for (int tr : {4, 5})
                for (double thd : {10.0, 45.0, 75.0, 85.0})
                {
                    ub::Pal P = lump_palette({mk(2.401e-9, 0.0, 0.0, tf, m, 1.0e-3)});
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, 0.2}, {1, 1}, {false, true}, mi);
                    const double R = std::norm(event(P, mesh, mi, 3, thd, 0)), T = std::norm(event(P, mesh, mi, tr, thd, 0));
                    QD_DIAG("%.0f,%.1f,%d,%.0f,%.17g,%.17g,%.2e,%.2e\n", m, tf, tr, thd, R, T, R + tf - 1.0, T - tf);
                    twin("bare tf E", twin_key("%.0f,%.1f,%d,%.0f", m, tf, tr, thd), {R, T});
                    INFO("rigid wall, m " << m << ", tf " << tf << ", type " << tr << ", theta " << thd);
                    CHECK(std::abs(R + tf - 1.0) < 1.0e-12); // closed: one minus its leak
                    CHECK(std::abs(T - tf) < 1.0e-12);       // and the leak, flat
                }

    QD_DIAG("\n# diag: bare tf F, a rigid wall with a loss column, which is not a valid material: eps 2.401e-9 (1 - j tan), tf 0.3, 200 mm, no\n");
    QD_DIAG("# mass column, scalar. R: reflectance of the first face; chain: entry and exit together, dB. It has to fail gracefully\n");
    QD_DIAG("tan,type,theta,R,chain_dB\n");
    for (double td : {0.0, 1.0e-6, 1.0e-3, 0.04})
        for (int tr : {4, 5})
            for (double thd : {10.0, 45.0, 75.0})
            {
                ub::Pal P = lump_palette({mk(2.401e-9, 2.401e-9 * td, 0.0, 0.3)});
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0, 0.2}, {1, 1}, {false, true}, mi);
                auto a = ub::chain(P, mesh, mi, {tr, tr}, f, thd, {1.5});
                const double R = std::norm(event(P, mesh, mi, 3, thd, 0));
                const double c = (a.size() == 2) ? std::norm(cpx(a[0].xpr, 0, 0)) * std::norm(cpx(a[1].xpr, 0, 0)) : 0.0;
                QD_DIAG("%.0e,%d,%.0f,%.6f,%.4f\n", td, tr, thd, R, dB(c));
                INFO("rigid wall with loss " << td << ", type " << tr << ", theta " << thd);
                CHECK(std::isfinite(c));
                CHECK(c < 1.0 + 1.0e-9); // no amplification
                CHECK(R < 1.0 + 1.0e-9);
            }

    QD_DIAG("\n# diag: bare tf G, a forced reflection at a face between two media of one permittivity, lossless eps 2.50, tf -0.5 on the\n");
    QD_DIAG("# host: behind a 6 mm coating without tf (every path, series summed), and behind a 20 mm solid without tf (16 events).\n");
    QD_DIAG("# other: the permittivity of the coating or of the first solid. scalar\n");
    QD_DIAG("wall,other,theta,T,R,sum,dropped\n");
    for (int wall = 0; wall < 2; ++wall)
        for (double e2 : {2.5, 2.5001, 2.6})
            for (double thd : {0.0, 45.0, 60.0})
            {
                ub::Pal P = lump_palette({mk(2.5, 0.0, 0.0, -0.5), mk(e2, 0.0, wall ? 0.0 : 0.006, 0.0)});
                arma::uvec mi;
                arma::mat mesh = wall ? ub::build({0.0, 0.02, 0.02 + gap, 0.04 + gap}, {2, 2, 1, 1}, {false, true, false, true}, mi)
                                      : ub::build({0.0, gap, gap + 0.006}, {2, 1, 1}, {false, false, true}, mi);
                bf::Acc A = ports(P, mesh, mi, 4, thd, false, wall ? 16 : 400);
                const double T = std::norm(A.T[0]), R = std::norm(A.R[0]);
                QD_DIAG("%s,%.4f,%.0f,%.6f,%.6f,%.6f,%.1e\n", wall ? "two solids" : "coated", e2, thd, T, R, T + R, A.drop);
                INFO("matched face, wall " << wall << ", other " << e2 << ", theta " << thd);
                CHECK(std::abs(T + R - 1.0) < 2.0e-3); // a lossless wall, matched exactly or not
            }

    QD_DIAG("\n# diag: bare tf H, recorded: 6 mm of eps 0.90 - 0.20j with tf < 0 in air about 60 deg, where the admittance ratio of a face\n");
    QD_DIAG("# has modulus one in the scalar and TE channel and a forced face changes side: as a traced solid (every path, series summed)\n");
    QD_DIAG("# and as a sheet, T and R in dB, scalar. At 60 deg exactly rounding decides the side\n");
    QD_DIAG("tf,as,T_59.99,T_60.00,T_60.01,step_T,R_59.99,R_60.00,R_60.01,step_R\n");
    for (double tf : {-0.1, -0.5, -0.9, 0.5})
        for (int sh = 0; sh < 2; ++sh)
        {
            ub::Pal P = lump_palette({mk(0.90, 0.20, 0.0, tf), mk(0.90, 0.20, 0.006, tf)});
            arma::uvec mi;
            arma::mat mesh = sh ? ub::build({0.0}, {2}, {false}, mi) : ub::build({0.0, 0.006}, {1, 1}, {false, true}, mi);
            double T[3], R[3];
            for (int q = 0; q < 3; ++q)
            {
                bf::Acc A = ports(P, mesh, mi, 4, 59.99 + 0.01 * q, false);
                T[q] = dB(std::norm(A.T[0])), R[q] = dB(std::norm(A.R[0]));
                CHECK((T[q] < 2.0e-3 && R[q] < 2.0e-3)); // each port at most 0 dB on both sides of the step
            }
            QD_DIAG("%.1f,%s,%.4f,%.4f,%.4f,%+.4f,%.4f,%.4f,%.4f,%+.4f\n", tf, sh ? "sheet" : "solid", T[0], T[1], T[2], T[2] - T[0], R[0], R[1], R[2], R[2] - R[0]);
        }

    QD_DIAG("\n# diag: bare tf T, twins, what must not move: a solid without tf in air, every path, series summed, the two ports with every\n");
    QD_DIAG("# digit, and the entry event of a rigid wall\n");
    QD_DIAG("row,type,theta,T_re,T_im,R_re,R_im,T_hh_re,T_hh_im,R_hh_re,R_hh_im\n");
    for (const Mt &M : mats)
        for (int tr : {4, 5, 1, 2})
            for (double thd : {20.0, 60.0, 80.0})
            {
                ub::Pal P = lump_palette({mk(M.a, M.ei, 0.0, 0.0)});
                arma::uvec mi;
                arma::mat mesh = ub::build({0.0, 0.006}, {1, 1}, {false, true}, mi);
                bf::Acc A = ports(P, mesh, mi, tr, thd, false);
                QD_DIAG("no tf %s,%d,%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", M.name, tr, thd, A.T[0].real(), A.T[0].imag(), A.R[0].real(), A.R[0].imag(),
                        A.T[1].real(), A.T[1].imag(), A.R[1].real(), A.R[1].imag());
                twin("bare tf T", twin_key("no tf %s,%d,%.0f", M.name, tr, thd), {A.T[0].real(), A.T[0].imag(), A.R[0].real(), A.R[0].imag(), A.T[1].real(), A.T[1].imag(), A.R[1].real(), A.R[1].imag()});
                CHECK(std::norm(A.T[0]) + std::norm(A.R[0]) < 1.0005);
                CHECK(std::norm(A.T[1]) + std::norm(A.R[1]) < 1.0005);
            }
}

TEST_CASE("Closed face - a coating met from inside a dense solid", "[wave]")
{
    ub::OneThread one_thread;
    // A closed face has a wave on one side only: behind it nothing propagates, a rigid wall of the acoustic mapping
    // from 0.003 deg on, air behind a coated solid met from inside past the critical angle. tf says that a share leaks
    // through. Next to a lossless medium that carries a wave that is one statement, a share of the forward wave's flux.
    // Next to a lossy medium, or a sheet the wave is evanescent in, the flux towards the face is carried by the forward
    // and the reflected wave together, and a share of the forward wave alone is not what the reflection leaves: taken
    // as the leak, it makes the two ports of a sheet on a rigid wall add to more than they received.
    //
    // The rule asserted here: the leak of a closed face is a matched resistive pad in front of it, formed on the
    // modulus of the admittance of the side the wave is on, as the transformer of an open face forms its ratio on a
    // modulus. The face passes what it absorbs. In front of a lossless medium that carries a wave it is sqrt(R') on the
    // Fresnel reflection and 1 - R' of the flux, R' the pair rule on a reflectance of one; at a critical angle it is
    // the limit of the transformer. One rule at the far face of a coherent sheet, at the bare face of a traced solid,
    // at the mirror of a summed series and at the faces of a closed mass-law layer.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update. The coated solid is an EM
    // material with tf and no mass column. The rows on a rigid wall, which carries the mass column, are in "Closed face
    // - tf at a closed face is a pad and the two ports of a sheet on it add up".
    //
    // A. Both ports of a coating met from inside: at most one together, and on the reference of the rule
    // D2. No step in the rule at the critical angle of the outer pair
    // H. Twins with every digit
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5, k0 = 2.0 * QPI * f / C0, ONE = 1.0 + 1.0e-6, eW = 2.401e-9;
    const cd J(0.0, 1.0);
    auto refl = [](int tr)
    { return (tr > 2) ? 3 : 0; };
    auto dBp = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    auto launched = [](int cur, double thd)
    {
        ub::Ev S;
        const double th = thd * QPI / 180.0;
        S.prev = arma::Col<short>{0}, S.cur = arma::Col<short>{(short)cur}, S.buf = arma::Col<short>{0};
        S.path_dirN = arma::mat{{std::cos(th), std::sin(th), 0.0}};
        S.accg = arma::vec{0.1}, S.accr = arma::vec{0.1};
        return S;
    };
    enum
    {
        WALL,
        COAT
    };
    // Both ports of the sheet event. Material 1 is the sheet, 2 the wall behind it (WALL) or the host the ray is launched in (COAT)
    auto ports = [&](const ub::Pal &P, int cfg, int tr, double thd, int k, cd &T, cd &R)
    {
        arma::uvec mi;
        arma::mat mesh = (cfg == WALL) ? ub::build({0.0, gap, gap + 0.1}, {1, 2, 2}, {false, false, true}, mi) : ub::build({0.0, gap}, {2, 1}, {true, false}, mi);
        ub::Ev S = launched(2, thd);
        auto t = ub::chain(P, mesh, mi, {tr}, f, thd, {1.5}, (cfg == COAT) ? &S : nullptr), r = ub::chain(P, mesh, mi, {refl(tr)}, f, thd, {1.5}, (cfg == COAT) ? &S : nullptr);
        T = t.empty() ? cd(0.0) : cpx(t[0].xpr, 0, k), R = r.empty() ? cd(0.0) : cpx(r[0].xpr, 0, k);
    };

    // The reference of the rule, in powers, in tilted admittances and sharing nothing with the
    // library: kz for TE and the scalar type, eps / kz for TM, kz = sqrt(eps - p) on the decaying
    // root, p the transverse invariant. Near face: a transformer of ratio sqrt(nu). Far face,
    // closed: the pad, a load |s| (nu |s| + b) / (|s| + nu b) in front of the reactance b
    auto kz_of = [](cd e, double p)
    { cd v = std::sqrt(e - p); return (std::imag(v) > 0.0 || (std::imag(v) == 0.0 && std::real(v) < 0.0)) ? -v : v; };
    auto pad_ref = [&](cd en, cd es, cd ef, double tf_n, double tf_s, double tf_f, double d, double p, bool tm, double &Tp, double &Rp) -> bool
    {
        auto adm = [&](cd e)
        { return tm ? e / kz_of(e, p) : kz_of(e, p); };
        const cd a = adm(en), s = adm(es), b = adm(ef), phi = std::exp(-J * k0 * kz_of(es, p) * d);
        cd r1 = (a - s) / (a + s), t1 = 1.0 + r1, tb1 = 1.0 - r1;
        const double x0 = std::abs(s / a);
        if (tf_n != 0.0 || tf_s != 0.0)
        {
            const double rm = (1.0 - x0) / (1.0 + x0), Rm = diag_tf_pair(rm * rm, tf_n, tf_s), q = (rm < 0.0) ? -std::sqrt(Rm) : std::sqrt(Rm);
            const double nu = ((1.0 - q) / (1.0 + q)) / x0, kk = std::sqrt(nu);
            r1 = (a - nu * s) / (a + nu * s), t1 = kk * (1.0 + r1), tb1 = (1.0 - r1) / kk;
        }
        if (std::imag(ef) == 0.0 && p < std::real(ef)) // the far side carries a wave: the transformer, and the flux it hands on
        {
            cd r2 = (s - b) / (s + b), t2 = 1.0 + r2;
            const double y0 = std::abs(b / s);
            if (tf_s != 0.0 || tf_f != 0.0)
            {
                const double rm = (1.0 - y0) / (1.0 + y0), Rm = diag_tf_pair(rm * rm, tf_s, tf_f), q = (rm < 0.0) ? -std::sqrt(Rm) : std::sqrt(Rm);
                const double nu = ((1.0 - q) / (1.0 + q)) / y0;
                r2 = (s - nu * b) / (s + nu * b), t2 = std::sqrt(nu) * (1.0 + r2);
            }
            const cd den = 1.0 + r1 * r2 * phi * phi;
            Rp = std::norm(r1 + t1 * tb1 * r2 * phi * phi / den), Tp = std::norm(t1 * t2 * phi / den) * std::real(b) / std::real(a);
            return true;
        }
        const double sc = std::sqrt(diag_tf_pair(1.0, tf_s, tf_f)), nu = (1.0 - sc) / (1.0 + sc), sm = std::abs(s);
        const cd bj(0.0, std::imag(b)), YL = (nu > 0.0) ? sm * (nu * sm + bj) / (sm + nu * bj) : b;
        cd r2 = (nu > 0.0) ? (s - YL) / (s + YL) : (s - b) / (s + b); // at its own magnitude, above one where it is
        const cd den = 1.0 + r1 * r2 * phi * phi;
        Rp = std::norm(r1 + t1 * tb1 * r2 * phi * phi / den);
        Tp = (nu > 0.0) ? std::norm(t1 * phi * (1.0 + r2) / den) * std::real(YL) / std::real(a) : 0.0;
        return !(std::abs(x0 - 1.0) < 1.0e-6 && (tf_n < 0.0 || tf_s < 0.0)); // the tie of a forced near face
    };

    struct Sh
    {
        double a, ei;
    };
    const std::vector<Sh> sheets = {{0.31, 0.20}, {0.90, 0.20}, {1.50, 0.40}, {5.24, 0.63}, {5.24, 5.0}, {0.31, 0.0}, {0.90, 0.0}, {5.24, 0.0}};
    const std::vector<Sh> coats = {{0.31, 0.20}, {0.90, 0.20}, {1.50, 0.40}, {2.50, 0.80}, {5.24, 0.63}, {9.0, 1.2}, {0.31, 0.0}, {1.50, 0.0}, {2.50, 0.0}, {9.0, 0.0}};
    const char *chn[3] = {"sc", "TE", "TM"};

    // ---- A. Both ports of one sheet event under a closed outer pair ----
    QD_DIAG("\n# diag: closed face A, both ports of a sheet event under a closed outer pair, the worst over 1, 6 and 25 mm and seven angles:\n");
    QD_DIAG("# 27 to 85 deg for a coating on eps 5.24 met from inside (critical angle 25.9 deg). tf: of the sheet. sum: T + R in\n");
    QD_DIAG("# power, the largest, at theta / mm; above: rows above one / rows. dT, dR: the largest distance of a power from the reference of the\n");
    QD_DIAG("# rule. ch: sc scalar, TE (VV), TM (HH)\n");
    QD_DIAG("where,sheet,tf,ch,sum,theta,mm,above,rows,dT,dR\n");
    for (int w = 2; w < 3; ++w)
    {
        const int cfg = (w < 2) ? WALL : COAT;
        const double tW = (w == 0) ? 0.3 : 0.0;
        const char *where = (w == 0) ? "rigid wall tf 0.3" : ((w == 1) ? "rigid wall" : "coating on 5.24");
        const cd en = (cfg == WALL) ? cd(1.0) : cd(5.24), ef = (cfg == WALL) ? cd(eW) : cd(1.0);
        for (const Sh &S : (cfg == WALL) ? sheets : coats)
            for (double tS : {0.5, 0.9, -0.5, 0.0})
            {
                double w_sum[3] = {0.0, 0.0, 0.0}, w_th[3] = {0.0, 0.0, 0.0}, w_mm[3] = {0.0, 0.0, 0.0}, w_dT[3] = {0.0, 0.0, 0.0}, w_dR[3] = {0.0, 0.0, 0.0};
                int n_above[3] = {0, 0, 0}, n_rows[3] = {0, 0, 0};
                for (double thk : {0.001, 0.006, 0.025})
                    for (double thd : (cfg == WALL) ? std::vector<double>{5.0, 20.0, 37.0, 45.0, 60.0, 73.0, 85.0} : std::vector<double>{27.0, 30.0, 37.0, 45.0, 60.0, 73.0, 85.0})
                        for (int tr : {4, 1})
                            for (int k : {0, 3})
                            {
                                if (tr == 4 && k == 3)
                                    continue;
                                const int ch = (tr == 4) ? 0 : ((k == 0) ? 1 : 2);
                                ub::Pal P = (cfg == WALL) ? lump_palette({{S.a, S.ei, thk, 0.0, tS}, {eW, 0.0, 0.0, 0.0, tW, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}})
                                                          : lump_palette({{S.a, S.ei, thk, 0.0, tS}, {5.24, 0.0}});
                                cd T, R;
                                ports(P, cfg, tr, thd, k, T, R);
                                const double sT = std::norm(T), sR = std::norm(R), sum = sT + sR, p = std::real(en) * std::pow(std::sin(thd * QPI / 180.0), 2);
                                double Tp = 0.0, Rp = 0.0;
                                const bool ok = pad_ref(en, cd(S.a, -S.ei), ef, 0.0, tS, tW, thk, p, ch == 2, Tp, Rp);
                                ++n_rows[ch];
                                if (sum > ONE)
                                    ++n_above[ch];
                                if (sum > w_sum[ch])
                                    w_sum[ch] = sum, w_th[ch] = thd, w_mm[ch] = thk * 1e3;
                                if (ok)
                                    w_dT[ch] = std::max(w_dT[ch], std::abs(sT - Tp)), w_dR[ch] = std::max(w_dR[ch], std::abs(sR - Rp));
                                INFO("A, " << where << ", sheet " << S.a << " - " << S.ei << "j, tf " << tS << ", " << chn[ch] << ", " << thk * 1e3 << " mm, theta " << thd);
                                CHECK(sum < ONE);
                                if (S.ei == 0.0) // a lossless sheet on a lossless closed side returns or passes all of it
                                    CHECK(std::abs(sum - 1.0) < 1.0e-6);
                                if (tS == 0.0 && tW == 0.0)
                                    CHECK(sT == 0.0); // without tf anywhere exactly nothing passes
                                if (ok)
                                {
                                    CHECK(std::abs(sT - Tp) < 1.0e-9);
                                    CHECK(std::abs(sR - Rp) < 1.0e-9);
                                }
                            }
                for (int ch = 0; ch < 3; ++ch)
                    QD_DIAG("%s,%.2f-%.2fj,%.1f,%s,%.6f,%.0f,%.0f,%d,%d,%.2e,%.2e\n", where, S.a, S.ei, tS, chn[ch], w_sum[ch], w_th[ch], w_mm[ch], n_above[ch], n_rows[ch], w_dT[ch],
                            w_dR[ch]);
            }
    }

    // Across the critical angle of the outer pair the far cosine has a square-root cusp, so two
    // neighbours differ by tenths of a dB on the exact rule as well. The statement is made in
    // two halves: the event sits on the reference on both sides, the open face below and the
    // pad above; and the reference has no step, its two limits at the angle agree
    QD_DIAG("\n# diag: closed face D2, eps 5.24 | 6 mm coating with tf 0.5 | air about the critical angle of the outer pair, 25.91 deg, powers in\n");
    QD_DIAG("# dB. lo, hi: 0.01 deg below and above, the event; dT, dR: its distance from the reference, the open far face below and the pad\n");
    QD_DIAG("# above. lim_T, lim_R: the reference at an invariant of 1 + 1e-10 over 1 - 1e-10, dB, the step of the rule itself\n");
    QD_DIAG("coating,ch,T_lo,T_hi,R_lo,R_hi,dT_lo,dT_hi,dR_lo,dR_hi,lim_T,lim_R\n");
    {
        const double tcO = std::asin(std::sqrt(1.0 / 5.24)) * 180.0 / QPI;
        for (const Sh &S : std::vector<Sh>{{1.50, 0.40}, {5.24, 0.63}, {0.31, 0.0}, {9.0, 0.0}})
            for (int k : {0, 3})
            {
                ub::Pal P = lump_palette({{S.a, S.ei, 0.006, 0.0, 0.5}, {5.24, 0.0}});
                cd T[2], R[2];
                double Tp[2], Rp[2], Tl[2], Rl[2];
                for (int h = 0; h < 2; ++h)
                {
                    const double thd = tcO + (h ? 0.01 : -0.01);
                    ports(P, COAT, 1, thd, k, T[h], R[h]);
                    pad_ref(cd(5.24), cd(S.a, -S.ei), cd(1.0), 0.0, 0.5, 0.0, 0.006, 5.24 * std::pow(std::sin(thd * QPI / 180.0), 2), k == 3, Tp[h], Rp[h]);
                    pad_ref(cd(5.24), cd(S.a, -S.ei), cd(1.0), 0.0, 0.5, 0.0, 0.006, 1.0 + (h ? 1.0e-10 : -1.0e-10), k == 3, Tl[h], Rl[h]);
                }
                QD_DIAG("%.2f-%.2fj,%s,%.4f,%.4f,%.4f,%.4f,%.2e,%.2e,%.2e,%.2e,%+.6f,%+.6f\n", S.a, S.ei, k ? "TM" : "TE", dBp(std::norm(T[0])), dBp(std::norm(T[1])), dBp(std::norm(R[0])),
                        dBp(std::norm(R[1])), std::abs(std::norm(T[0]) - Tp[0]), std::abs(std::norm(T[1]) - Tp[1]), std::abs(std::norm(R[0]) - Rp[0]), std::abs(std::norm(R[1]) - Rp[1]),
                        dBp(Tl[1]) - dBp(Tl[0]), dBp(Rl[1]) - dBp(Rl[0]));
                INFO("D2, coating " << S.a << " - " << S.ei << "j, k " << k);
                for (int h = 0; h < 2; ++h)
                {
                    CHECK(std::abs(std::norm(T[h]) - Tp[h]) < 1.0e-9);
                    CHECK(std::abs(std::norm(R[h]) - Rp[h]) < 1.0e-9);
                }
                CHECK(std::abs(dBp(Tl[1]) - dBp(Tl[0])) < 1.0e-3);
                CHECK(std::abs(dBp(Rl[1]) - dBp(Rl[0])) < 1.0e-3);
            }
    }

    // ---- H. Twins with every digit ----
    QD_DIAG("\n# diag: closed face H, twins, the complex coefficient of the sheet event with every digit, 6 mm, 45 deg: a lossless sheet denser\n");
    QD_DIAG("# than the invariant with tf as a coating on eps 5.24 and a lossy sheet without tf as a coating. port: T or R. k: 0 scalar\n");
    QD_DIAG("# or VV, 3 HH\n");
    QD_DIAG("twin,type,k,port,re,im\n");
    {
        struct Tw
        {
            const char *name;
            int cfg;
            LumpMtl s, w;
        };
        const LumpMtl host = {5.24, 0.0};
        const std::vector<Tw> tw = {{"dense 9.0 tf 0.5 coating on 5.24", COAT, {9.0, 0.0, 0.006, 0.0, 0.5}, host},
                                    {"lossy 1.50-0.40j no tf coating on 5.24", COAT, {1.50, 0.40, 0.006}, host}};
        for (const Tw &W : tw)
            for (int tr : {4, 1})
                for (int k : {0, 3})
                {
                    if (tr == 4 && k == 3)
                        continue;
                    cd T, R;
                    ports(lump_palette({W.s, W.w}), W.cfg, tr, 45.0, k, T, R);
                    QD_DIAG("%s,%d,%d,T,%.17g,%.17g\n", W.name, tr, k, T.real(), T.imag());
                    QD_DIAG("%s,%d,%d,R,%.17g,%.17g\n", W.name, tr, k, R.real(), R.imag());
                    twin("closed face H", twin_key("%s,%d,%d", W.name, tr, k), {T.real(), T.imag(), R.real(), R.imag()});
                    INFO("H, " << W.name << ", type " << tr << ", k " << k);
                    CHECK(std::norm(T) + std::norm(R) < ONE);
                }
    }
}
