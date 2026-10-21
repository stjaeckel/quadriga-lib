// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_state_update, wave: traced chains through the intersector and the mirror walkers.

#include "ray_test_helpers.hpp"

TEST_CASE("Recovery gate - the hit code a traced run carries", "[wave]")
{
    // A hand-built event carries a synthetic hit code, 1 for a bare exit and 69 for a glued one. A traced run does not:
    // with the undeviated tracer a ray inside a solid travels at its external angle, ray_mesh_interact judges the exit
    // face at that angle, and from sqrt|eps| sin(theta_ext) >= 1 its code carries the total-reflection bit, 33 and 101.
    // That bit is a statement about the face ahead at an angle the wave does not have. The recovery inverts the bend at
    // the face behind, and declines on it.
    //
    // Four instruments. The code ray_mesh_interact writes for the exit, asserted. The twin: the same event with that
    // code and with the bit cleared, bitwise, which needs no reference. The absolute level with the real code, against
    // the analytic slab and the composite at the exact cosine. And the two-event run through the intersector with
    // nothing synthetic in it, for the undeviated and for the refracting tracer; the lossy rows of the refracting
    // tracer are printed and not asserted.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, LOSSY = 5, DENSE = 2;
    const std::complex<double> e_air(1.0, 0.0), e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    // The code ray_mesh_interact writes for an exit met along the external direction: the exit
    // face is the back of a quad at x = 0, with a sheet face 10 um behind it for a glued exit
    auto exit_code = [&](int itype, int host, bool glued, double thd) -> uint8_t
    {
        arma::mat mesh = diag_quad_x(200.0, 0.0);
        arma::uvec mtl_ind(2, arma::fill::value((arma::uword)host));
        if (glued)
        {
            mesh = arma::join_cols(mesh, diag_quad_x(200.0, -1.0e-5));
            mtl_ind = arma::uvec{(arma::uword)host, (arma::uword)host, (arma::uword)SHEET, (arma::uword)SHEET};
        }
        double t = std::tan(thd * QPI / 180.0);
        arma::mat orig{{1.0, 0.37, 0.0}}, dest{{-1.0, 0.37 + 2.0 * t, 0.0}}, fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        std::vector<uint8_t> ot;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, &ot, (arma::mat *)nullptr, true,
                                        (arma::u32_vec *)nullptr);
        REQUIRE(ot.size() == 1);
        return ot[0];
    };
    // One event closing a leg of normal thickness L along (dx, dy)
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, double th, double eps,
                     double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.run();
        return C;
    };
    auto same = [](const Rsu<double> &A, const Rsu<double> &B)
    {
        for (arma::uword k = 0; k < A.xprmat.n_rows; ++k)
            if ((double)A.xprmat(k, 0) != (double)B.xprmat(k, 0))
                return false;
        return (double)A.gain(0) == (double)B.gain(0);
    };

    // ---- Bare exit ----
    QD_DIAG("\n# diag: recovery gate, bare exit. air | slab 100 mm | air, scalar, the exit event along the\n");
    QD_DIAG("# direction a tracer holds, series resolved. code is what ray_mesh_interact writes for that exit\n");
    QD_DIAG("# met at the external angle. d_real and d_syn are the chain against the analytic slab with\n");
    QD_DIAG("# that code and with code 1; twin_T, twin_R, twin_EM say whether the two runs are bit-identical\n");
    QD_DIAG("mtl,theta_ext,code,d_real_dB,d_syn_dB,twin_T,twin_R,twin_EM,rtype\n");
    for (int m : {5, 1, 4, 6})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            std::complex<double> es = diag_eps(m, f_hz);
            double ct = diag_traced_ct(es, th), st = std::sqrt(1.0 - ct * ct);
            uint8_t code = exit_code(4, m, false, thd);
            bool past = std::sqrt(std::abs(es)) * std::sin(th) >= 1.0;
            std::complex<double> t_in, u;
            diag_port(e_air, es, std::complex<double>(std::cos(th), 0.0), true, t_in, u);
            auto Xr = event(4, code, 1u, m, 0, 0, m, th, 0.0, ct, st);
            auto Xs = event(4, OT_EXIT, 1u, m, 0, 0, m, th, 0.0, ct, st);
            auto Rr = event(3, exit_code(3, m, false, thd), 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Rs = event(3, OT_EXIT, 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Er = event(1, exit_code(1, m, false, thd), 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Es = event(1, OT_EXIT, 1u, m, 0, 0, m, th, 1.5, ct, st);
            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);
            double d_real = dB(std::norm(t_in * cpx(Xr.xprmat, 0, 0))) - dB(p_te);
            double d_syn = dB(std::norm(t_in * cpx(Xs.xprmat, 0, 0))) - dB(p_te);
            QD_DIAG("%d,%.1f,%u,%+.6f,%+.6f,%d,%d,%d,%u\n", m, thd, (unsigned)code, d_real, d_syn, (int)same(Xr, Xs),
                    (int)same(Rr, Rs), (int)same(Er, Es), (unsigned)Xr.resolved_type[0]);
            INFO("bare exit, material " << m << ", theta_ext " << thd);
            CHECK((int)code == (past ? 33 : 1)); // the bit is the geometric-angle verdict
            CHECK(std::abs(d_syn) < 1.0e-6);     // the control: the synthetic code on the slab
            CHECK(std::abs(d_real) < 1.0e-6);
            CHECK(same(Xr, Xs));
            CHECK(same(Rr, Rs));
            CHECK(same(Er, Es));
        }

    // ---- Glued exit ----
    QD_DIAG("\n# diag: recovery gate, glued exit. host 100 mm | coating 6 mm eps 5.24 - 0.63j | air, scalar,\n");
    QD_DIAG("# host series declined. d_T and d_R: the event with the code ray_mesh_interact writes against\n");
    QD_DIAG("# the composite at the exact cosine times the host leg; s_T the same with code 69\n");
    QD_DIAG("host,theta_ext,code,d_T_dB,d_R_dB,s_T_dB,twin_T,twin_R,twin_EM,rtype\n");
    for (int host : {5, 6})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            std::complex<double> e2 = diag_eps(host, f_hz), c2 = std::sqrt(1.0 - s2 / e2);
            double ct = diag_traced_ct(e2, th), st = std::sqrt(1.0 - ct * ct);
            uint8_t code = exit_code(4, host, true, thd);
            bool past = std::sqrt(std::abs(e2)) * std::sin(th) >= 1.0;
            auto Tr = event(4, code, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Ts = event(4, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Rr = event(3, exit_code(3, host, true, thd), 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Rs = event(3, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Er = event(1, exit_code(1, host, true, thd), 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Es = event(1, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e_air, thk, c2, f_hz, true, true);
            // A sheet event replaces the field, so the probe feed is not in it
            double ref_T = std::exp(2.0 * k0 * std::imag(std::sqrt(e2) * c2) * L) * std::norm(Ref.T_te);
            // A reflect pass at a sheet closes the host leg as the transmit pass does
            double ref_R = std::exp(2.0 * k0 * std::imag(std::sqrt(e2) * c2) * L) * std::norm(Ref.R_te);
            double d_T = dB(std::norm(cpx(Tr.xprmat, 0, 0))) - dB(ref_T), d_R = dB(std::norm(cpx(Rr.xprmat, 0, 0))) - dB(ref_R);
            double s_T = dB(std::norm(cpx(Ts.xprmat, 0, 0))) - dB(ref_T);
            QD_DIAG("%d,%.1f,%u,%+.6f,%+.6f,%+.6f,%d,%d,%d,%u\n", host, thd, (unsigned)code, d_T, d_R, s_T, (int)same(Tr, Ts),
                    (int)same(Rr, Rs), (int)same(Er, Es), (unsigned)Tr.resolved_type[0]);
            INFO("glued exit, host " << host << ", theta_ext " << thd);
            CHECK((int)code == (past ? 101 : 69));
            CHECK(std::abs(s_T) < 1.0e-6); // the control
            CHECK(std::abs(d_T) < 1.0e-6);
            CHECK(std::abs(d_R) < 1.0e-6);
            CHECK(same(Tr, Ts));
            CHECK(same(Rr, Rs));
            CHECK(same(Er, Es));
        }

    // ---- A face that is closed at the recovered cosine ----
    // The lossy host entered from a lossless dense solid with a transverse invariant above one:
    // the face to air is closed, the bit is right, and the reflection is evaluated at the
    // recovered cosine or, where the recovery declines, at the projection. The far medium's root
    // has an argument that is real and negative up to rounding.
    QD_DIAG("\n# diag: recovery gate, a closed face. Reflection at host | air inside eps 1.50 - 0.40j entered\n");
    QD_DIAG("# from a lossless eps 5.24 solid, invariant p above one, scalar. arg with code 1 and with code\n");
    QD_DIAG("# 33, with the dense solid named and with a sheet in the previous-medium word; step is the\n");
    QD_DIAG("# change of arg_1_named from the row above\n");
    QD_DIAG("p,ct_traced,dB_1,arg_1_named,arg_1_sheet,arg_33_named,arg_33_sheet,step,rtype\n");
    {
        const std::complex<double> e_d = diag_eps(DENSE, f_hz), e_l = diag_eps(LOSSY, f_hz);
        double prev_arg = 0.0;
        bool first = true;
        for (double pinv = 1.10; pinv < 1.46; pinv += 0.025)
        {
            double s1 = std::sqrt(pinv / std::real(e_d)), th1 = std::asin(s1);
            std::complex<double> eps_r = e_d / e_l, c2 = std::sqrt(1.0 - eps_r * s1 * s1);
            double eta = std::sqrt(std::abs(eps_r)), r = std::real(c2);
            double ct = r / std::sqrt((eta * s1) * (eta * s1) + r * r), st = std::sqrt(1.0 - ct * ct);
            auto A1 = event(3, OT_EXIT, 1u, LOSSY, 0, DENSE, LOSSY, th1, 1.5, ct, st);
            auto B1 = event(3, OT_EXIT, 1u, LOSSY, 0, SHEET, LOSSY, th1, 1.5, ct, st);
            auto A3 = event(3, 33, 1u, LOSSY, 0, DENSE, LOSSY, th1, 1.5, ct, st);
            auto B3 = event(3, 33, 1u, LOSSY, 0, SHEET, LOSSY, th1, 1.5, ct, st);
            auto deg = [&](const Rsu<double> &X)
            { return std::arg(cpx(X.xprmat, 0, 0)) * 180.0 / QPI; };
            double step = deg(A1) - prev_arg;
            step -= 360.0 * std::round(step / 360.0);
            QD_DIAG("%.3f,%.9f,%.6f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%u\n", pinv, ct, dB(std::norm(cpx(A1.xprmat, 0, 0))), deg(A1),
                    deg(B1), deg(A3), deg(B3), first ? 0.0 : step, (unsigned)A1.resolved_type[0]);
            INFO("closed face, p " << pinv);
            CHECK(std::abs(cpx(A1.xprmat, 0, 0) - cpx(B1.xprmat, 0, 0)) < 1.0e-9 * std::abs(cpx(A1.xprmat, 0, 0)));
            CHECK(same(A1, A3)); // the bit does not move the event
            CHECK(same(B1, B3));
            if (!first)
                CHECK(std::abs(step) < 45.0); // no branch flip between neighbours
            prev_arg = deg(A1), first = false;
        }
    }

    // ---- Nothing synthetic: two events through the intersector ----
    QD_DIAG("\n# diag: recovery gate, the traced run. air | slab 100 mm | air through ray_triangle_intersect,\n");
    QD_DIAG("# ray_mesh_interact and ray_state_update, two events, scalar, series resolved, against the\n");
    QD_DIAG("# analytic slab. itype 4 is the undeviated tracer, 5 the refracting one. out_deg is the\n");
    QD_DIAG("# direction written at the exit\n");
    QD_DIAG("itype,mtl,theta_ext,code,rtype,out_deg,chain_dB,slab_dB,d_dB\n");
    {
        arma::mat mesh = arma::join_cols(diag_quad_x(200.0, 0.0), diag_quad_x(200.0, L));
        mesh.row(2) = arma::rowvec{L, -200.0, -200.0, L, 200.0, -200.0, L, -200.0, 200.0}; // outward normal
        mesh.row(3) = arma::rowvec{L, 200.0, -200.0, L, 200.0, 200.0, L, -200.0, 200.0};
        struct Ev
        {
            arma::mat origN, destN, path_dirN, xpr;
            arma::Col<short> prev, cur, buf;
            arma::vec accg, accr;
            uint8_t ot = 0, rt = 0;
        };
        auto traced = [&](int itype, int m, const arma::mat &orig, const arma::mat &dest, const Ev *in) -> Ev
        {
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));
            arma::mat fbs, sbs, fbsN, sbsN, nv;
            arma::u32_vec nh, fi, si;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
            Ev E;
            arma::vec gain, fang;
            std::vector<uint8_t> ot, rt;
            quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fi, &si, (arma::mat *)nullptr,
                                            (arma::mat *)nullptr, &E.origN, &E.destN, &fbsN, &sbsN, &gain, &E.xpr,
                                            (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                            (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
            arma::Col<short> mf(1), ms(1);
            mf(0) = fi(0) ? (short)m : (short)0, ms(0) = si(0) ? (short)m : (short)0;
            quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &fbsN, &sbsN, &nh, &fang, &nv, &ot, &mtl, &mf, &ms,
                                           in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                           in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                           &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                           (arma::u32_vec *)nullptr, 0.0);
            E.ot = ot[0], E.rt = rt[0];
            return E;
        };
        for (int itype : {4, 5})
            for (int m : {5, 6})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0;
                    std::complex<double> es = diag_eps(m, f_hz);
                    arma::mat orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev A = traced(itype, m, orig, dest, nullptr);
                    arma::mat d = A.destN - A.origN;
                    d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                    arma::mat dest2 = A.origN + 2.0 * d;
                    Ev B = traced(itype, m, A.origN, dest2, &A);
                    double ch = std::norm(cpx(A.xpr, 0, 0) * cpx(B.xpr, 0, 0));
                    double p_te = 0.0, p_tm = 0.0;
                    diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);
                    double out_deg = std::atan2(std::abs(B.path_dirN(0, 1)), std::abs(B.path_dirN(0, 0))) * 180.0 / QPI;
                    QD_DIAG("%d,%d,%.1f,%u,%u,%.4f,%.6f,%.6f,%+.6f\n", itype, m, thd, (unsigned)B.ot, (unsigned)B.rt, out_deg,
                            dB(ch), dB(p_te), dB(ch) - dB(p_te));
                    INFO("traced run, itype " << itype << ", material " << m << ", theta_ext " << thd);
                    REQUIRE((B.ot & 0x1F) == 1);
                    // The relaunch offset moves the leg by a few float ulp, so the bound is not the 1e-6 dB of a
                    // hand-built event. The refracting tracer's lossy rows are printed: the recovery does not run where
                    // traced and geometric cosines agree
                    if (itype == 4 || m == 6)
                        CHECK(std::abs(dB(ch) - dB(p_te)) < 1.0e-3);
                }
    }
}

TEST_CASE("Exits and verdicts - every exit at the cosine the wave has there", "[wave]")
{
    ub::OneThread one_thread;
    // Every exit and every verdict is formed at the cosine the wave has at that face:
    //   A. a resolved ray of the undeviated tracer by two more routes: reflected off a coating on
    //      the far side (always resolved, whatever eps is), and returning across a bare face into
    //      a first solid;
    //   B. a bare exit behind a wedge against the glued one, through the intersector;
    //   C. the verdict and the direction at a bare and at a glued exit, invariant p around one;
    //   D. the reflect port of a coated lossy solid past its critical angle.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, thk = 0.006, gap = 1.0e-5;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, LOSSY = 5, DENSE = 2;
    const std::complex<double> J(0.0, 1.0);
    typedef std::complex<double> cd;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](cd z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr, gain;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &E.gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &mtl, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &E.gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        cd vv = 1.0;
        double Lin = 0.0, dy = 0.0;
        std::vector<Ev> ev;
        std::vector<double> offy;
        bool ok = true;
        cd abs_vv(double k) const { return vv * std::exp(cd(0.0, -k * Lin)); }
    };
    auto run = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, std::vector<int> seq, double thd, double eps) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            Ev E = traced(mesh, mtl_ind, itype, orig, dest, have ? &P : nullptr, eps, itype != 0 && itype != 3);
            if (E.nh == 0)
                break;
            C.ev.push_back(E);
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            C.offy.push_back(std::abs(E.origN(0, 1) - fy));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = E.origN, dest = E.origN + 2.0 * d;
            P = E, have = true;
        }
        return C;
    };
    auto rts = [](const Chain &c)
    {
        std::string s;
        for (const auto &E : c.ev)
            s += std::to_string((unsigned)E.rt) + "/";
        return s;
    };

    // Everything from the transverse invariant p. Admittances z c (TE) and z / c (TM); the far
    // side is air, with the decaying root above p = 1. R is the reflect port of near | sheet | air
    // or, without a sheet, of near | air; pT the per-channel normal-flux transmittance (TE only).
    struct Ref
    {
        cd c_near, R;
        double pT = 0.0, leg = 1.0; // leg: power decay of the host over a normal thickness
    };
    auto ref_of = [&](cd e_near, bool sheet, double p, bool tm, double d_host) -> Ref
    {
        const cd e_s = diag_eps(SHEET, f_hz);
        Ref r;
        r.c_near = std::sqrt(1.0 - p / e_near);
        cd cs = std::sqrt(1.0 - p / e_s), cf = (p < 1.0) ? cd(std::sqrt(1.0 - p), 0.0) : cd(0.0, -std::sqrt(p - 1.0));
        cd zn = std::sqrt(e_near), zs = std::sqrt(e_s);
        cd an = tm ? zn / r.c_near : zn * r.c_near, as = tm ? zs / cs : zs * cs, af = tm ? 1.0 / cf : cf;
        cd t;
        if (sheet)
        {
            cd r1 = (an - as) / (an + as), r2 = (as - af) / (as + af), ph = std::exp(-J * k0 * zs * cs * thk);
            r.R = (r1 + r2 * ph * ph) / (1.0 + r1 * r2 * ph * ph);
            t = (1.0 + r1) * (1.0 + r2) * ph / (1.0 + r1 * r2 * ph * ph);
        }
        else
            r.R = (an - af) / (an + af), t = 1.0 + r.R;
        r.pT = (p < 1.0) ? std::real(af) / std::real(an) * std::norm(t) : 0.0;
        r.leg = std::exp(2.0 * k0 * std::imag(zn * r.c_near) * d_host);
        return r;
    };

    // ---- A1. A resolved ray that was reflected off a coating on the far side ----
    QD_DIAG("\n# diag: exits and verdicts, a ray reflected off a far-side coating. air | host 100 mm | coating 6 mm eps 5.24 - 0.63j |\n");
    QD_DIAG("# air through the intersector: enter, reflect off the coating (type 3), leave by the front face. Host\n");
    QD_DIAG("# series declined (eps 1.5); the reflection off a sheet flags the ray resolved all the same. Scalar,\n");
    QD_DIAG("# against t01 R t10 phi^2 with R the reflect port of host | coating | air. ph_poly at the polyline's point\n");
    QD_DIAG("tracer,host,theta,rt,d_dB,ph_poly\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                double th = thd * QPI / 180.0, st = std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                Chain c = run(mesh, mi, {tr, 3, tr}, thd, 1.5);
                INFO("far-side coating, tracer " << tr << ", host " << h << ", theta_ext " << thd);
                if (!c.ok || c.ev.size() != 3 || std::norm(c.vv) == 0.0)
                {
                    QD_DIAG("%d,%d,%.1f,%s,dead\n", tr, h, thd, rts(c).c_str());
                    CHECK(false); // the wave crosses the front face
                    continue;
                }
                cd eh = diag_eps(h, f_hz), zh = std::sqrt(eh);
                Ref r = ref_of(eh, true, st * st, false, 0.0);
                cd a0 = std::cos(th), b0 = zh * r.c_near, t01 = 2.0 * a0 / (a0 + b0), t10 = 2.0 * b0 / (a0 + b0);
                double Lx = (L - c.ev[0].fbs(0, 0)) + c.ev[1].fbs(0, 0);
                cd a = c.abs_vv(k0) / (t01 * r.R * t10 * std::exp(-J * k0 * zh * r.c_near * Lx));
                QD_DIAG("%d,%d,%.1f,%s,%+.6f,%+.4f\n", tr, h, thd, rts(c).c_str(), dB(std::norm(a)), deg(a * std::exp(J * k0 * st * c.dy)));
                CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                CHECK(std::abs(deg(a * std::exp(J * k0 * st * c.dy))) < 0.05);
            }

    // ---- A2. A resolved ray returning across a bare face into the first solid ----
    QD_DIAG("\n# diag: exits and verdicts, a resolved ray across a bare face into a solid. air | A 50 mm | B 100 mm | air through the\n");
    QD_DIAG("# intersector, every series resolved (eps 0): enter A, cross into B, reflect at B | air, cross back into\n");
    QD_DIAG("# A, leave by the front face. No reference: the undeviated tracer over the refracting one, each at its\n");
    QD_DIAG("# polyline's exit point. rt4, rt5: the resolved types of the five events\n");
    QD_DIAG("pair,theta,rt4,rt5,r45_dB,r45_deg\n");
    for (auto P : std::vector<std::pair<int, int>>{{2, 5}, {1, 5}, {2, 6}, {5, 2}})
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0})
        {
            double st = std::sin(thd * QPI / 180.0);
            arma::uvec mi;
            arma::mat mesh = build({0.0, 0.05, 0.05 + gap, 0.15 + gap}, {P.first, P.first, P.second, P.second}, {false, true, false, true}, mi);
            Chain c4 = run(mesh, mi, {4, 4, 3, 4, 4}, thd, 0.0), c5 = run(mesh, mi, {5, 5, 3, 5, 5}, thd, 0.0);
            INFO("two solids, resolved, " << P.first << "|" << P.second << ", theta_ext " << thd);
            bool dead4 = !c4.ok || c4.ev.size() != 5 || std::norm(c4.vv) == 0.0, dead5 = !c5.ok || c5.ev.size() != 5 || std::norm(c5.vv) == 0.0;
            if (dead4 || dead5)
            {
                QD_DIAG("%d|%d,%.1f,%s,%s,%s\n", P.first, P.second, thd, rts(c4).c_str(), rts(c5).c_str(), dead5 ? "dead5" : "dead4");
                CHECK(false); // no ray dies on this path with either tracer
                continue;
            }
            cd r45 = (c4.abs_vv(k0) * std::exp(J * k0 * st * c4.dy)) / (c5.abs_vv(k0) * std::exp(J * k0 * st * c5.dy));
            QD_DIAG("%d|%d,%.1f,%s,%s,%+.6f,%+.4f\n", P.first, P.second, thd, rts(c4).c_str(), rts(c5).c_str(), dB(std::norm(r45)), deg(r45));
            CHECK(std::abs(dB(std::norm(r45))) < 2.0e-3); // two polylines, each short by its own offsets
            CHECK(std::abs(deg(r45)) < 0.05);
        }

    // ---- B. A wedge: the bare exit against the glued one ----
    // Entry face at x = 0, exit face through (L, 0.37) with its outward normal turned by 10 deg
    // towards +y, so the entry event sets the non-parallel flag itself and the recovery declines
    // on the leg. Angles are against the exit normal. in: the direction the wave has; seg_in: the
    // segment; out: the direction written; seg_out: the relaunched segment; map: the real-angle
    // map applied to the direction the wave has; rmi: the same map applied to the segment.
    QD_DIAG("\n# diag: exits and verdicts, a wedge. air | host | air, exit face tilted by 10 deg, through the intersector, scalar,\n");
    QD_DIAG("# series declined. bare: a bare exit; glued: the same exit behind a 6 mm coating. Angles in deg against\n");
    QD_DIAG("# the exit normal: in, the direction the wave has; seg_in, the segment; out, the direction written;\n");
    QD_DIAG("# seg_out, the relaunched segment; map, the real-angle map of in; rmi, the real-angle map of seg_in.\n");
    QD_DIAG("# prev: the previous-medium word the entry wrote. gain: the exit event, dB\n");
    QD_DIAG("tracer,host,theta,exit,prev,rt,in,seg_in,out,seg_out,map,rmi,gain_dB\n");
    {
        const double a = 10.0 * QPI / 180.0, ca = std::cos(a), sa = std::sin(a), y0 = 0.37;
        auto tilt = [&](arma::mat m, double shift)
        {
            for (arma::uword r = 0; r < m.n_rows; ++r)
                for (arma::uword v = 0; v < 3; ++v)
                {
                    double x = m(r, 3 * v) - L, y = m(r, 3 * v + 1) - y0;
                    m(r, 3 * v) = L + x * ca - y * sa + shift * ca, m(r, 3 * v + 1) = y0 + x * sa + y * ca + shift * sa;
                }
            return m;
        };
        auto ang = [&](double dx, double dy) // against the exit normal
        { return std::atan2(std::abs(dy * ca - dx * sa), std::abs(dx * ca + dy * sa)) * 180.0 / QPI; };
        auto map_of = [&](cd e, double in_deg)
        {
            double s = std::sin(in_deg * QPI / 180.0);
            return std::atan2(std::sqrt(std::abs(e)) * s, std::real(std::sqrt(1.0 - e * s * s))) * 180.0 / QPI;
        };
        double bare_x = 0.0, bare_y = 0.0;
        for (int tr : {4, 5})
            for (int h : {5, 6})
                for (double thd : {10.0, 20.0, 40.0, 60.0})
                    for (int glued = 0; glued < 2; ++glued)
                    {
                        double t = std::tan(thd * QPI / 180.0);
                        arma::mat mesh = arma::join_cols(face(0.0, false), tilt(face(L, true), 0.0));
                        arma::uvec mi(4, arma::fill::value((arma::uword)h));
                        if (glued)
                        {
                            mesh = arma::join_cols(mesh, tilt(face(L, true), gap));
                            mi = arma::uvec{(arma::uword)h, (arma::uword)h, (arma::uword)h, (arma::uword)h, (arma::uword)SHEET, (arma::uword)SHEET};
                        }
                        arma::mat orig{{-1.0, y0 - t, 0.0}}, dest{{1.0, y0 + t, 0.0}};
                        Ev A = traced(mesh, mi, tr, orig, dest, nullptr, 1.5, true);
                        arma::mat d = A.destN - A.origN;
                        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                        arma::mat dest2 = A.origN + 2.0 * d;
                        Ev B = traced(mesh, mi, tr, A.origN, dest2, &A, 1.5, true);
                        arma::mat d2 = B.destN - B.origN;
                        cd eh = diag_eps(h, f_hz);
                        double in_deg = ang(A.path_dirN(0, 0), A.path_dirN(0, 1)), seg_deg = ang(d(0, 0), d(0, 1));
                        QD_DIAG("%d,%d,%.1f,%s,%04x,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%+.4f\n", tr, h, thd, glued ? "glued" : "bare ",
                                (unsigned)(unsigned short)A.prev(0), (unsigned)B.rt, in_deg, seg_deg, ang(B.path_dirN(0, 0), B.path_dirN(0, 1)),
                                ang(d2(0, 0), d2(0, 1)), map_of(eh, in_deg), map_of(eh, seg_deg), dB(std::norm(cpx(B.xpr, 0, 0))));
                        INFO("wedge, tracer " << tr << ", host " << h << ", theta_ext " << thd << ", glued " << glued);
                        CHECK(B.nh > 0);
                        CHECK(((unsigned short)A.prev(0) & 0x8000u) != 0u); // the entry saw the wedge
                        if (!glued)
                            bare_x = B.path_dirN(0, 0), bare_y = B.path_dirN(0, 1);
                        else // a bare and a glued exit write one direction
                            CHECK(std::max(std::abs(B.path_dirN(0, 0) - bare_x), std::abs(B.path_dirN(0, 1) - bare_y)) < 1.0e-9);
                    }
    }

    // ---- C and D. The lossy host entered from a lossless dense solid, invariant p around and above one ----
    // The code ray_mesh_interact writes for the exit met along the segment, which with the
    // undeviated tracer runs at the angle the wave had in the dense solid
    auto exit_code = [&](int itype, int host, bool glued, double th) -> uint8_t
    {
        arma::mat mesh = diag_quad_x(200.0, 0.0);
        arma::uvec mtl_ind(2, arma::fill::value((arma::uword)host));
        if (glued)
        {
            mesh = arma::join_cols(mesh, diag_quad_x(200.0, -1.0e-5));
            mtl_ind = arma::uvec{(arma::uword)host, (arma::uword)host, (arma::uword)SHEET, (arma::uword)SHEET};
        }
        double t = std::tan(th);
        arma::mat orig{{1.0, 0.37, 0.0}}, dest{{-1.0, 0.37 + 2.0 * t, 0.0}}, fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        std::vector<uint8_t> ot;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, &ot, (arma::mat *)nullptr, true,
                                        (arma::u32_vec *)nullptr);
        REQUIRE(ot.size() == 1);
        return ot[0];
    };
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, double th, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th); // the undeviated tracer's
        C.run();
        return C;
    };
    const cd e_d = diag_eps(DENSE, f_hz), e_l = diag_eps(LOSSY, f_hz);
    auto dir_of = [&](double p, double &th1, double &ct, double &st, double &N2)
    {
        double s1 = std::sqrt(p / std::real(e_d));
        th1 = std::asin(s1);
        cd c2 = std::sqrt(1.0 - p / e_l);
        double eta = std::sqrt(std::abs(e_d / e_l)), r = std::real(c2);
        N2 = (eta * s1) * (eta * s1) + r * r; // the squared length the projection lost
        ct = r / std::sqrt(N2), st = std::sqrt(1.0 - ct * ct);
    };

    QD_DIAG("\n# diag: exits and verdicts, the verdict at an exit, invariant p around one. Host eps 1.50 - 0.40j 100 mm entered from a\n");
    QD_DIAG("# lossless eps 5.24 solid, named in the previous-medium word, undeviated tracer, scalar, series declined.\n");
    QD_DIAG("# The wave is closed towards air from p = 1. N2: the squared length the projection lost; p/N2: what the\n");
    QD_DIAG("# real-angle test reads. Per exit: hit code, resolved type, tr = its total-reflection bit, the event\n");
    QD_DIAG("# against the reference in dB (ref 0 above p = 1: gain printed as is), and the direction written, deg\n");
    QD_DIAG("p,N2,p/N2,b_code,b_rt,b_tr,b_gain_dB,b_ref_dB,b_out,g_code,g_rt,g_tr,g_gain_dB,g_ref_dB,g_out\n");
    for (double p : {0.50, 0.90, 0.95, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10, 1.20, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        uint8_t cb = exit_code(4, LOSSY, false, th1), cg = exit_code(4, LOSSY, true, th1);
        auto Xb = event(4, cb, 1u, LOSSY, 0, DENSE, LOSSY, th1, ct, st);
        auto Xg = event(4, cg, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        Ref rb = ref_of(e_l, false, p, false, L), rg = ref_of(e_l, true, p, false, L);
        auto out = [](const Rsu<double> &X)
        { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
        double gb = std::norm(cpx(Xb.xprmat, 0, 0)), gg = std::norm(cpx(Xg.xprmat, 0, 0));
        QD_DIAG("%.2f,%.6f,%.6f,%u,%u,%d,%+.6f,%+.6f,%.4f,%u,%u,%d,%+.6f,%+.6f,%.4f\n", p, N2, p / N2,
                (unsigned)cb, (unsigned)Xb.resolved_type[0], (int)rt_bit(Xb.resolved_type[0], 5), dB(gb), dB(rb.pT * rb.leg), out(Xb),
                (unsigned)cg, (unsigned)Xg.resolved_type[0], (int)rt_bit(Xg.resolved_type[0], 5), dB(gg), dB(rg.pT * rg.leg), out(Xg));
        INFO("verdict, p " << p);
        // Closed exactly where the invariant exceeds the far medium's, bare and glued alike
        CHECK(rt_bit(Xb.resolved_type[0], 5) == (p > 1.0));
        CHECK(rt_bit(Xg.resolved_type[0], 5) == (p > 1.0));
        if (p > 1.0) // a closed pair hands on the direction the wave has, and passes nothing without tf
        {
            CHECK(std::max(std::abs((double)Xg.path_dirN(0, 0) - ct), std::abs((double)Xg.path_dirN(0, 1) - st)) < 1.0e-12);
            CHECK(gb == 0.0);
            CHECK(gg == 0.0);
        }
        else
        {
            CHECK(std::abs(dB(gb) - dB(rb.pT * rb.leg)) < 1.0e-6);
            CHECK(std::abs(dB(gg) - dB(rg.pT * rg.leg)) < 1.0e-6);
        }
    }

    QD_DIAG("\n# diag: exits and verdicts, the reflect port of a coated lossy solid. The same host and entry, host | coating 6 mm\n");
    QD_DIAG("# eps 5.24 - 0.63j | air, reflect pass, series declined, against the composite at the exact cosine times\n");
    QD_DIAG("# the host leg. sc: scalar (type 3). vv, hh: the two EM channels (type 0) against the TE and the TM\n");
    QD_DIAG("# reference; amplitudes, difference of magnitudes. bare: the same without the coating, scalar\n");
    QD_DIAG("p,code,rt,|R_sc|,ref_te,d_sc,|vv|,d_vv_te,|hh|,ref_tm,d_hh_tm,|R_bare|,ref_bare,d_bare\n");
    for (double p : {0.50, 0.90, 1.02, 1.04, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        uint8_t c3 = exit_code(3, LOSSY, true, th1), c0 = exit_code(0, LOSSY, true, th1);
        auto S = event(3, c3, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        auto E = event(0, c0, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        auto B = event(3, exit_code(3, LOSSY, false, th1), 1u, LOSSY, 0, DENSE, LOSSY, th1, ct, st);
        Ref rte = ref_of(e_l, true, p, false, L), rtm = ref_of(e_l, true, p, true, L), rb = ref_of(e_l, false, p, false, L);
        double a = std::sqrt(rte.leg);
        double sc = std::abs(cpx(S.xprmat, 0, 0)), vv = std::abs(cpx(E.xprmat, 0, 0)), hh = std::abs(cpx(E.xprmat, 0, 3)), bb = std::abs(cpx(B.xprmat, 0, 0));
        QD_DIAG("%.2f,%u,%u,%.9f,%.9f,%+.3e,%.9f,%+.3e,%.9f,%.9f,%+.3e,%.9f,%.9f,%+.3e\n", p, (unsigned)c3, (unsigned)S.resolved_type[0],
                sc, a * std::abs(rte.R), sc - a * std::abs(rte.R), vv, vv - a * std::abs(rte.R), hh, a * std::abs(rtm.R),
                hh - a * std::abs(rtm.R), bb, a * std::abs(rb.R), bb - a * std::abs(rb.R));
        INFO("reflect port, p " << p);
        // Scalar and TE at the exact cosine, at every invariant: the recovery runs until it has
        // converged. TM as well: no clamp stands on the closed far face
        CHECK(std::abs(sc - a * std::abs(rte.R)) < 1.0e-9);
        CHECK(std::abs(vv - a * std::abs(rte.R)) < 1.0e-9);
        CHECK(std::abs(hh - a * std::abs(rtm.R)) < 1.0e-9);
    }

    // The recovery above p = 1. It inverts qd_refract's map by Newton from the real-index start,
    // until the step is below rounding. Three steps leave 2e-14
    // at p = 1.2 and 9e-9 at 1.45. Written a second time here, with three steps and with twelve:
    // u is the recovered sin^2 in the dense solid, and the event is on the reference at p.
    QD_DIAG("\n# diag: exits and verdicts, the recovery above p = 1. The map inverted a second time in the fixture, three Newton\n");
    QD_DIAG("# steps and twelve. du3, du12: the recovered invariant over the true one, minus one. d_ref: the scalar\n");
    QD_DIAG("# reflect port of the coated host against the reference at p; d_3: against the reference at the\n");
    QD_DIAG("# invariant three steps give\n");
    QD_DIAG("p,du3,du12,d_ref,d_3\n");
    for (double p : {0.50, 0.90, 0.99, 1.10, 1.20, 1.30, 1.40, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        cd eps_r = e_d / e_l;
        double eta2 = std::abs(eps_r), t = st / ct;
        auto solve = [&](int n)
        {
            double u = t * t / (eta2 * (1.0 + t * t));
            u = (u > 1.0) ? 1.0 : u;
            for (int k = 0; k < n; ++k)
            {
                cd w = std::sqrt(1.0 - eps_r * u);
                double fu = eta2 * u - t * t * std::real(w) * std::real(w), dfu = eta2 + t * t * std::real(w) * std::real(eps_r / w);
                u -= fu / dfu;
                u = (u < 0.0) ? 0.0 : ((u > 1.0) ? 1.0 : u);
            }
            return u;
        };
        double u3 = solve(3), u12 = solve(12), u0 = p / std::real(e_d);
        auto S = event(3, exit_code(3, LOSSY, true, th1), 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        Ref r0 = ref_of(e_l, true, p, false, L), r3 = ref_of(e_l, true, u3 * std::real(e_d), false, L);
        double sc = std::abs(cpx(S.xprmat, 0, 0));
        QD_DIAG("%.2f,%+.3e,%+.3e,%+.3e,%+.3e\n", p, u3 / u0 - 1.0, u12 / u0 - 1.0, sc - std::sqrt(r0.leg) * std::abs(r0.R),
                sc - std::sqrt(r3.leg) * std::abs(r3.R));
        INFO("recovery above one, p " << p);
        CHECK(std::abs(sc - std::sqrt(r0.leg) * std::abs(r0.R)) < 1.0e-12); // converged: on the reference at p
        CHECK(std::abs(u12 / u0 - 1.0) < 1.0e-12);
    }

    // The window with a transmission factor on the pair: what the transmit port of the glued exit
    // is where the verdict reads open on a closed wave, against where it reads closed.
    QD_DIAG("\n# diag: exits and verdicts, the verdict window with tf. The glued exit of the table above, transmit port, with tf 0.5\n");
    QD_DIAG("# on the coating, on the host, and on neither. tr: the total-reflection bit of the resolved type\n");
    QD_DIAG("p,p/N2,tr,none_dB,tf_sheet_dB,tf_host_dB\n");
    {
        auto pal_s = mtl, pal_h = mtl;
        pal_s["tf"] = std::vector<double>(9, 0.0), pal_h["tf"] = std::vector<double>(9, 0.0);
        pal_s["tf"][SHEET - 1] = 0.5, pal_h["tf"][LOSSY - 1] = 0.5;
        auto ev_pal = [&](const std::unordered_map<std::string, std::vector<double>> &pal, uint8_t ot, double th, double dx, double dy)
        {
            auto C = make1<double>(pal, 4, ot, 2u, SHEET, LOSSY, DENSE, LOSSY, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
            C.run();
            return C;
        };
        double leak_prev = 0.0;
        for (double p : {0.95, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.10, 1.20})
        {
            double th1, ct, st, N2;
            dir_of(p, th1, ct, st, N2);
            uint8_t cg = exit_code(4, LOSSY, true, th1);
            auto X0 = ev_pal(mtl, cg, th1, ct, st), Xs = ev_pal(pal_s, cg, th1, ct, st), Xh = ev_pal(pal_h, cg, th1, ct, st);
            QD_DIAG("%.2f,%.6f,%d,%+.4f,%+.4f,%+.4f\n", p, p / N2, (int)rt_bit(Xs.resolved_type[0], 5), dB(std::norm(cpx(X0.xprmat, 0, 0))),
                    dB(std::norm(cpx(Xs.xprmat, 0, 0))), dB(std::norm(cpx(Xh.xprmat, 0, 0))));
            INFO("verdict window with tf, p " << p);
            double leak = dB(std::norm(cpx(Xs.xprmat, 0, 0)));
            CHECK(rt_bit(Xs.resolved_type[0], 5) == (p > 1.0));
            if (p > 1.0)
            {
                CHECK(leak > -10.0); // the leak of the coating crosses from p = 1 on
                if (p > 1.015)
                    CHECK(std::abs(leak - leak_prev) < 0.5); // and has no hole
                CHECK(std::norm(cpx(X0.xprmat, 0, 0)) == 0.0);
                CHECK(std::norm(cpx(Xh.xprmat, 0, 0)) == 0.0);
            }
            leak_prev = leak;
        }
    }
}

TEST_CASE("Sheet reflection - a declined series is followed and not dropped", "[wave]")
{
    ub::OneThread one_thread;
    // A host leg that closes at a sheet either has its bounce series summed or leaves the bounces to the tracer. Only a
    // ray whose series was summed carries the resolved flag: with the series declined, the next reflection of that ray
    // is followed, not killed.
    //
    // Through ray_triangle_intersect, ray_mesh_interact and ray_state_update, scalar, both tracers. Walls: HS, air |
    // host 100 | coating 6 mm | air; HSH, air | host 50 | sheet | host 50 | air. eps 1.5 declines every series, eps 0
    // sums every one.
    // A. The current-medium word after the reflection off the sheet.
    // B. HS, declined: the transmitted paths, one more round trip each, and the reflected ones, as geometric sequences
    // whose limits are the wall. Control: eps 0 on one path.
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs, sbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        E.sbs = sbsN;
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev;
        std::vector<double> fx, bx; // per event: x of the hit point, x where the leg that follows starts (qd_leg_start)
        bool ok = true;
        std::string rts, words;
        std::complex<double> at_point(double k, double st) const // the field at the polyline's exit point
        {
            return vv * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * st * dy));
        }
    };
    // seq: one type per event, or one type repeated until the ray is out. epsv likewise
    auto run = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq,
                   const std::vector<double> &epsv, double thd, double f) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        const int n_max = seq.size() == 1 ? 16 : (int)seq.size();
        for (int k = 0; k < n_max; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : seq[(size_t)k];
            double eps = epsv.size() == 1 ? epsv[0] : epsv[(size_t)k];
            bool tx = itype != 0 && itype != 3;
            Ev E = traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, eps, tx);
            if (E.nh == 0)
            {
                C.ok = seq.size() == 1;
                break;
            }
            C.rts += std::to_string((unsigned)E.rt) + "/";
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            // long enough to cross every face that is left
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            C.fx.push_back(fx), C.bx.push_back(qd_leg_start(E, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        if (seq.size() > 1 && C.ev.size() != seq.size())
            C.ok = false;
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };

    const double L = 0.10, k1 = 2.0 * QPI * FRQ / C0;

    // ---- A. The state word ----
    QD_DIAG("\n# diag: sheet reflection, the word. Enter, then reflect off the sheet (type 3). wall HS: a coating on the far\n");
    QD_DIAG("# face; HSH: a sheet inside the host. S: the reflect event at this eps over the one at eps 1.5. res: the\n");
    QD_DIAG("# resolved flag of the current-medium word after it. words as prev.cur.buf\n");
    QD_DIAG("wall,host,tracer,eps,theta,|S|,res,rt,words\n");
    for (int wall = 0; wall < 2; ++wall)
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (double eps : {1.5, 0.0})
                    for (double thd : angles)
                    {
                        arma::uvec mi;
                        arma::mat mesh = wall ? build({0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, mi)
                                              : build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                        Chain c = run(mtl, mesh, mi, {tr, 3}, {eps}, thd, FRQ), d = run(mtl, mesh, mi, {tr, 3}, {1.5}, thd, FRQ);
                        INFO("word, wall " << wall << ", host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                        REQUIRE((c.ok && d.ok));
                        std::complex<double> S = cpx(c.ev[1].xpr, 0, 0) / cpx(d.ev[1].xpr, 0, 0);
                        bool res = (c.ev[1].cur(0) & (short)0x8000) != 0, summed = std::abs(S - 1.0) > 1.0e-12;
                        QD_DIAG("%s,%d,%d,%.1f,%.0f,%.6f,%d,%u,%s\n", wall ? "HSH" : "HS ", h, tr, eps, thd, std::abs(S), (int)res, (unsigned)c.ev[1].rt,
                                c.words.c_str());
                        CHECK(mat_of(c.ev[1].cur(0)) == h);
                        CHECK(res == summed); // resolved exactly where the series was summed
                        if (eps == 0.0)
                            CHECK(summed);
                    }

    // ---- B. The paths of a wall coated on its far side ----
    auto seqT = [](int tr, int n)
    {
        std::vector<int> s = {tr};
        for (int i = 0; i < n; ++i)
            s.push_back(3), s.push_back(3);
        s.push_back(tr);
        return s;
    };
    auto seqR = [](int tr, int n) // n >= 1: n reflections off the coating, n - 1 off the front face
    {
        std::vector<int> s = {tr, 3};
        for (int i = 1; i < n; ++i)
            s.push_back(3), s.push_back(3);
        s.push_back(tr);
        return s;
    };
    QD_DIAG("\n# diag: sheet reflection, the paths of air | host 100 | coating 6 mm | air with every series declined (eps 1.5).\n");
    QD_DIAG("# T path n: enter, n round trips (off the coating, off the front face), leave through the coating. R path 0: off\n");
    QD_DIAG("# the front face; R path n: enter, off the coating, n - 1 round trips, leave by the front face. Each at its\n");
    QD_DIAG("# polyline's point. T0, T01: the first one and two T paths against the matrix of the wall, dB; Tinf: the limit\n");
    QD_DIAG("# T0 / (1 - T1/T0); geo: |T2/T1 - T1/T0|. R01, Rinf and its phase likewise, Rinf = R0 + R1 / (1 - R2/R1).\n");
    QD_DIAG("# c0: the control, eps 0 on one path; clamp: what the clamp of a mirror above one costs that control\n");
    QD_DIAG("# dR: |Rinf - R| over the front-face term; canc: the wall's reflection over the front-face term, dB\n");
    QD_DIAG("host,tracer,theta,dR,canc_dB,T0_dB,T01_dB,Tinf_dB,geo,R01_dB,Rinf_dB,Rinf_deg,c0_dB,clamp_dB,rtsT1,rtsR2\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                Chain t[3], r[3], r0 = run(mtl, mesh, mi, {3}, {1.5}, thd, FRQ), c0 = run(mtl, mesh, mi, {tr, tr}, {0.0}, thd, FRQ);
                std::complex<double> FT[3] = {0.0, 0.0, 0.0}, FR[4] = {0.0, 0.0, 0.0, 0.0};
                bool all = r0.ok && c0.ok;
                for (int n = 0; n < 3; ++n)
                {
                    t[n] = run(mtl, mesh, mi, seqT(tr, n), {1.5}, thd, FRQ), r[n] = run(mtl, mesh, mi, seqR(tr, n + 1), {1.5}, thd, FRQ);
                    if (t[n].ok)
                        FT[n] = t[n].at_point(k1, st);
                    if (r[n].ok)
                        FR[n + 1] = r[n].at_point(k1, st);
                    all = all && t[n].ok && r[n].ok;
                }
                INFO("paths, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((t[0].ok && r0.ok && c0.ok));
                FR[0] = r0.at_point(k1, st);
                double Lx = t[0].fx[1] - t[0].bx[0];
                std::complex<double> eH = diag_eps(h, FRQ), eS = diag_eps(SHEET, FRQ);
                RT W = cm(e_air, {{eH, Lx}, {eS, thk}}, e_air, s2, k1);
                // the clamp of a mirror above one: the coating seen from the host, above one in magnitude
                std::complex<double> Rs = cm(eH, {{eS, thk}}, e_air, s2, k1).r, r_fr = rint(eH, e_air, s2), pp = phi(eH, s2, k1, Lx);
                std::complex<double> Rc = (std::abs(Rs) > 1.0) ? Rs / std::abs(Rs) : Rs;
                double clamp_dB = 2.0 * dB(std::abs((1.0 - r_fr * Rs * pp * pp) / (1.0 - r_fr * Rc * pp * pp)));
                std::complex<double> qT = FT[1] / FT[0], Tinf = FT[0] / (1.0 - qT), qR = FR[2] / FR[1], Rinf = FR[0] + FR[1] / (1.0 - qR);
                double c0_dB = dB(std::norm(c0.at_point(k1, st)) / std::norm(W.t));
                if (all)
                    QD_DIAG("%d,%d,%.0f,%.1e,%+.2f,", h, tr, thd, std::abs(Rinf - W.r) / std::abs(FR[0]), dB(std::norm(W.r / FR[0])));
                if (all)
                    QD_DIAG("%+.4f,%+.4f,%+.6f,%.1e,%+.4f,%+.6f,%+.4f,%+.6f,%+.6f,%s,%s\n", dB(std::norm(FT[0] / W.t)),
                            dB(std::norm((FT[0] + FT[1]) / W.t)), dB(std::norm(Tinf / W.t)), std::abs(FT[2] / FT[1] - qT),
                            dB(std::norm((FR[0] + FR[1]) / W.r)), dB(std::norm(Rinf / W.r)), deg(Rinf / W.r), c0_dB, clamp_dB, t[1].rts.c_str(),
                            r[1].rts.c_str());
                else
                    QD_DIAG("%d,%d,%.0f,dead,dead,%+.4f,dead,dead,dead,%+.4f,dead,dead,%+.6f,%+.6f,%s,%s\n", h, tr, thd, dB(std::norm(FT[0] / W.t)),
                            dB(std::norm((FR[0] + FR[1]) / W.r)), c0_dB, clamp_dB, t[1].rts.c_str(), r[1].rts.c_str());
                CHECK(std::abs(c0_dB) < 1.0e-3); // the control: summed, one path is the wall; clamp_dB is what the clamp was worth
                CHECK(all);                      // no bounce of a declined series dies
                if (all)
                {
                    const double bound = (tr == 5 && thd == 85.0) ? 1.0e-2 : 1.0e-3; // the relaunch offsets, as for two solids
                    CHECK(std::abs(FT[2] / FT[1] - qT) < 1.0e-4);
                    CHECK(std::abs(dB(std::norm(Tinf / W.t))) < bound);
                    // The reflected limit against the front-face term, which is what the relaunch
                    // offsets act on: where the wall reflects 10 dB less than its front face, the
                    // same distance is more in dB (1.3e-3 dB at 16 dB of cancellation)
                    CHECK(std::abs(Rinf - W.r) < 2.0e-4 * std::abs(FR[0]));
                    CHECK(std::abs(dB(std::norm(Rinf / W.r))) < ((std::norm(W.r / FR[0]) < 0.1) ? 5.0e-3 : bound));
                    CHECK(std::abs(deg(Rinf / W.r)) < 0.05);
                }
            }
}

TEST_CASE("Measurement - the mirror behind the ray after a back reflection", "[wave]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // A series is formed from the face ahead of the ray and the one behind it, and the one behind is read from the
    // previous-medium word. A back reflection at a bare face does not write that word. Through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update, scalar, double, 1 GHz, both tracers.
    //
    // Walls, all standing in air:
    //   H    host 100                          HS   host 100 | coating 6 mm
    //   SH   coating 6 mm | host 100           AB   A 50 | B 100, two solids across a bare face
    //   HSH  host 50 | sheet 6 mm | host 50
    //
    // A. One leg at a time. The event that closes a leg which began at a back reflection at a bare face, at eps 0 over
    // the same at eps 1.5. Asserted: such a leg forms no series at any threshold (the verdict is inherited), the
    // reflection leaves the followed mark and the previous-medium word as it was. Printed: the series against the form
    // with the mirror that is behind the ray and with what the word holds, and the threshold at which the event would
    // stop summing, by bisection.
    // B. Every path, a printed table: one ray, every reflected and transmitted ray followed until it is in air, dead or
    // 20 events old, summed at the polyline's point. Four runs per row: the library at eps 0.15; the same with eps 1.5
    // handed to every event on a leg that began at a declined reflection (the verdict inherited); the same with the
    // mirror written into the word by the fixture after a declined reflection at a bare face (the word made true); and
    // eps 0.
    // C. The reflection of a wall coated on its near side, a printed table.
    // D. A verdict handed in, asserted.
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs, sbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch,
                      const std::vector<uint8_t> *vin = nullptr, std::vector<uint8_t> *vout = nullptr) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr,
                                       (arma::mat *)nullptr, (arma::mat *)nullptr, vin, vout);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        E.sbs = sbsN;
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        std::vector<Ev> ev;
        std::vector<double> fx, bx; // per event: x of the hit point, x where the leg that follows starts
        bool ok = true;
        std::string words;
    };
    auto run = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq, const std::vector<double> &epsv, double thd) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false;
        for (size_t k = 0; k < seq.size(); ++k)
        {
            bool tx = seq[k] != 0 && seq[k] != 3;
            Ev E = traced(mtl, mesh, mtl_ind, seq[k], FRQ, orig, dest, have ? &P : nullptr, epsv.size() == 1 ? epsv[0] : epsv[k], tx);
            if (E.nh == 0 || E.rt == 0)
            {
                C.ok = false;
                break;
            }
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d; // long enough to cross every face that is left
            C.fx.push_back(E.fbs(0, 0)), C.bx.push_back(qd_leg_start(E, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };
    auto unit = [](std::complex<double> r) // a mirror above one, clamped to the unit circle
    { return (std::abs(r) > 1.0) ? r / std::abs(r) : r; };
    const double k1 = 2.0 * QPI * FRQ / C0;
    std::vector<double> angles;
    for (double a = 10.0; a < 86.0; a += 5.0)
        angles.push_back(a);

    // The walls. leg: 0 = the only body, or the first one; 1 = the second
    struct Wall
    {
        const char *name;
        int a, b; // host, or first and second solid
    };
    auto mesh_of = [&](const std::string &w, int a, int b, arma::uvec &mi) -> arma::mat
    {
        if (w == "H")
            return build({0.0, 0.10}, {a, a}, {false, true}, mi);
        if (w == "HS")
            return build({0.0, 0.10, 0.10 + gap}, {a, a, SHEET}, {false, true, true}, mi);
        if (w == "SH")
            return build({0.0, gap, 0.10 + gap}, {SHEET, a, a}, {false, false, true}, mi);
        if (w == "AB")
            return build({0.0, 0.05, 0.05 + gap, 0.15 + gap}, {a, a, b, b}, {false, true, false, true}, mi);
        return build({0.0, 0.05, 0.10}, {a, SHEET, a}, {false, false, true}, mi); // HSH
    };

    // ---- A. One leg at a time ----
    QD_DIAG("\n# diag: mirror behind, one leg. The event that closes a leg which began at a back reflection at a bare face, at\n");
    QD_DIAG("# eps 0 over the same at eps 1.5: the series S it forms. leg: the body and the path, t a transmission, r a\n");
    QD_DIAG("# reflection. d_true: |S - S| with the mirror that is behind the ray; d_word: with what the previous-medium word\n");
    QD_DIAG("# holds. rho_true, rho_word: |r_ahead r_behind| |phi|^2 of the two, from the reference. rho_lib: the eps at which\n");
    QD_DIAG("# that event stops summing, by bisection; rho_1st: the same for the first leg in that body, whose word is true.\n");
    QD_DIAG("# worth_dB: 20 log |S_word / S_true|. win: 1 where rho_true < 0.15 <= rho_word, -1 for the reverse\n");
    QD_DIAG("wall,a,b,leg,tracer,theta,|S|,d_true,d_word,rho_true,rho_word,rho_lib,rho_1st,worth_dB,win,words\n");
    struct LegCase
    {
        const char *wall, *leg;
        int a, b;
        std::string seq, seq1;   // t/r per event; the last event of each is the one measured
        int body;                // which solid the leg is in: 0 = a, 1 = b
        int ahead, behind, word; // 0 air, 1 the other solid, 2 the sheet backed by air, 3 the sheet backed by the host
    };
    std::vector<LegCase> legs;
    for (int h : {5, 6})
    {
        legs.push_back({"HS", "host:trrt", h, 0, "trrt", "tt", 0, 2, 0, 2});
        legs.push_back({"SH", "host:trt", h, 0, "trt", "tt", 0, 2, 0, 2});
        legs.push_back({"HSH", "half1:trrt", h, 0, "trrt", "tt", 0, 3, 0, 3});
        legs.push_back({"HSH", "half2:ttrt", h, 0, "ttrt", "ttt", 0, 3, 0, 3});
    }
    for (int b : {5, 6})
    {
        legs.push_back({"AB", "A:trt", 2, b, "trt", "tt", 0, 0, 1, 0});
        legs.push_back({"AB", "B:ttrt", 2, b, "ttrt", "ttt", 1, 1, 0, 1});
    }
    for (const LegCase &lc : legs)
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                arma::uvec mi;
                arma::mat mesh = mesh_of(lc.wall, lc.a, lc.b, mi);
                auto seq_of = [&](const std::string &s)
                {
                    std::vector<int> q;
                    for (char c : s)
                        q.push_back(c == 't' ? tr : 3);
                    return q;
                };
                auto last_at = [&](const std::string &s, double e, std::complex<double> &x, Chain *keep = nullptr) -> bool
                {
                    std::vector<double> ev(s.size(), 1.5);
                    ev.back() = e;
                    Chain c = run(mesh, mi, seq_of(s), ev, thd);
                    if (!c.ok)
                        return false;
                    x = cpx(c.ev.back().xpr, 0, 0);
                    if (keep)
                        *keep = c;
                    return true;
                };
                auto rho_lib = [&](const std::string &s) -> double // the eps at which the last event stops summing
                {
                    std::complex<double> x0, x;
                    if (!last_at(s, 1.5, x0) || !last_at(s, 1.0e-9, x) || std::abs(x - x0) <= 1.0e-12 * std::abs(x0))
                        return -1.0;
                    double lo = 1.0e-9, hi = 1.5;
                    for (int it = 0; it < 30; ++it)
                    {
                        double mid = 0.5 * (lo + hi);
                        last_at(s, mid, x);
                        if (std::abs(x - x0) > 1.0e-12 * std::abs(x0))
                            lo = mid;
                        else
                            hi = mid;
                    }
                    return 0.5 * (lo + hi);
                };
                Chain c;
                std::complex<double> x0, x1;
                INFO("leg, wall " << lc.wall << ", " << lc.leg << ", a " << lc.a << ", b " << lc.b << ", tracer " << tr << ", theta " << thd);
                bool ok = last_at(lc.seq, 0.0, x1, &c) && last_at(lc.seq, 1.5, x0);
                if (!ok)
                {
                    QD_DIAG("%s,%d,%d,%s,%d,%.0f,dead\n", lc.wall, lc.a, lc.b, lc.leg, tr, thd);
                    continue;
                }
                std::complex<double> S = x1 / x0;
                size_t n = c.ev.size();
                std::complex<double> eM = diag_eps(lc.body ? lc.b : lc.a, FRQ), eO = diag_eps(lc.body ? lc.a : lc.b, FRQ), eS = diag_eps(SHEET, FRQ);
                auto mirror = [&](int kind) -> std::complex<double>
                {
                    if (kind == 0)
                        return rint(eM, e_air, s2);
                    if (kind == 1)
                        return rint(eM, eO, s2);
                    return unit(cm(eM, {{eS, thk}}, kind == 2 ? e_air : eM, s2, k1).r);
                };
                std::complex<double> pp = phi(eM, s2, k1, std::abs(c.fx[n - 1] - c.bx[n - 2]));
                std::complex<double> ra = mirror(lc.ahead), rb = mirror(lc.behind), rw = mirror(lc.word);
                std::complex<double> S_true = 1.0 / (1.0 - ra * rb * pp * pp), S_word = 1.0 / (1.0 - ra * rw * pp * pp);
                double rho_t = std::abs(ra * rb) * std::norm(pp), rho_w = std::abs(ra * rw) * std::norm(pp);
                QD_DIAG("%s,%d,%d,%s,%d,%.0f,%.6f,%.3e,%.3e,%.4f,%.4f,%.4f,%.4f,%+.4f,%d,%s\n", lc.wall, lc.a, lc.b, lc.leg, tr, thd, std::abs(S),
                        std::abs(S - S_true), std::abs(S - S_word), rho_t, rho_w, rho_lib(lc.seq), rho_lib(lc.seq1), 2.0 * dB(std::abs(S_word / S_true)),
                        (rho_t < 0.15 && rho_w >= 0.15) ? 1 : ((rho_w < 0.15 && rho_t >= 0.15) ? -1 : 0), c.words.c_str());
                CHECK(std::isfinite(std::abs(S)));
                // The verdict is inherited: a leg that begins at a declined reflection forms no
                // series, whatever the threshold of the event that closes it
                CHECK(x1 == x0);
                for (size_t k = 0; k + 1 < n; ++k)
                    if (lc.seq[k] == 'r')
                    {
                        // followed: the mark in the buffer word, and the ray not resolved
                        CHECK((unsigned short)c.ev[k].buf(0) == 0x4000u);
                        CHECK((c.ev[k].cur(0) & (short)0x8000) == 0);
                        // a bare face leaves the previous-medium word as it was, so the recovery
                        // reads what it read before the reflection
                        if ((c.ev[k].rt & 0x40) == 0)
                            CHECK(c.ev[k].prev(0) == c.ev[k - 1].prev(0));
                    }
            }

    // ---- C. The reflection of a wall coated on its near side, every series summed (a printed table) ----
    // At eps 0 the wall reflects on two paths: off the coating, and enter, off the bare far face with the host's
    // series, leave through the coating. Both at the polyline's point.
    auto field = [&](const Chain &c, double st) -> std::complex<double>
    {
        std::complex<double> v = 1.0;
        double Lin = 0.0;
        for (size_t k = 0; k < c.ev.size(); ++k)
        {
            v *= cpx(c.ev[k].xpr, 0, 0);
            if (k && mat_of(c.ev[k - 1].cur(0)) != 0)
                Lin += std::sqrt(std::pow(c.ev[k].fbs(0, 0) - c.ev[k - 1].fbs(0, 0), 2) + std::pow(c.ev[k].fbs(0, 1) - c.ev[k - 1].fbs(0, 1), 2));
        }
        return v * std::exp(-J * k1 * Lin) * std::exp(J * k1 * st * (c.ev.back().fbs(0, 1) - c.ev[0].fbs(0, 1)));
    };
    QD_DIAG("\n# diag: mirror behind, the reflection of a wall coated on its near side. air | coating 6 mm | host 100 | air, eps 0.\n");
    QD_DIAG("# R0: off the coating; R1: enter, off the far face, leave through the coating. R_dB, R_deg: R0 + R1 against the\n");
    QD_DIAG("# matrix of coating | host with the coating in front of the mesh plane, where the reflect port of a glued entry\n");
    QD_DIAG("# puts it. T_dB: enter, leave by the far face, against the matrix. R1/R0: dB\n");
    QD_DIAG("host,tracer,theta,R1/R0_dB,R_dB,R_deg,T_dB,words_R1\n");
    if (qd_diag())
    {
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                    arma::uvec mi;
                    arma::mat mesh = mesh_of("SH", h, 0, mi);
                    Chain c0 = run(mesh, mi, {3}, {0.0}, thd), c1 = run(mesh, mi, {tr, 3, tr}, {0.0}, thd), ct = run(mesh, mi, {tr, tr}, {0.0}, thd);
                    if (!(c0.ok && c1.ok && ct.ok))
                        continue;
                    std::complex<double> R0 = field(c0, st), R1 = field(c1, st), T = field(ct, st);
                    RT W = cm(e_air, {{diag_eps(SHEET, FRQ), thk}, {diag_eps(h, FRQ), std::abs(ct.fx[1] - ct.bx[0])}}, e_air, s2, k1);
                    std::complex<double> Ra = R0 + R1, Rw = W.r * std::exp(2.0 * J * k1 * std::cos(th) * thk);
                    QD_DIAG("%d,%d,%.0f,%+.2f,%+.6f,%+.4f,%+.6f,%s\n", h, tr, thd, dB(std::norm(R1 / R0)), dB(std::norm(Ra / Rw)),
                            std::arg(Ra / Rw) * 180.0 / QPI, dB(std::norm(T / W.t)), c1.words.c_str());
                }
    }

    // ---- B. Every path (a printed table) ----
    struct Out
    {
        std::complex<double> T = 0.0, R = 0.0;
        int nT = 0, nR = 0, n_sum = 0, n_cut = 0; // n_sum: events on a leg that began at a declined reflection and summed a series
        double worst = 0.0;                       // the largest |20 log |S|| among them
    };
    struct Node
    {
        arma::mat orig, dest;
        Ev P;
        bool have = false, inside = false, followed = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, Lin = 0.0;
        std::complex<double> vv = 1.0;
        int depth = 0;
    };
    // mode 0: the library; 1: eps 1.5 on a leg that began at a declined reflection; 2: the mirror
    // written into the previous-medium word after a declined reflection at a bare face
    auto walk = [&](const arma::mat &mesh, const arma::uvec &mi, int tr, double eps, int mode, double thd, int max_ev, double floor_dB,
                    double x_back, int A, int B, bool two_solids) -> Out
    {
        Out O;
        const double st = std::sin(thd * QPI / 180.0), floor_lin = std::pow(10.0, 0.1 * floor_dB);
        std::vector<Node> stack(1);
        diag_ray_at(thd, stack[0].orig, stack[0].dest);
        while (!stack.empty())
        {
            Node Nn = stack.back();
            stack.pop_back();
            for (int itype : {3, tr})
            {
                const bool tx = itype != 3;
                const double e_ev = (mode == 1 && Nn.followed) ? 1.5 : eps;
                Ev E = traced(mtl, mesh, mi, itype, FRQ, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, e_ev, tx);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                if (Nn.followed && mode != 1) // did this event sum a series?
                {
                    Ev D = traced(mtl, mesh, mi, itype, FRQ, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, 1.5, tx);
                    std::complex<double> S = cpx(E.xpr, 0, 0) / cpx(D.xpr, 0, 0);
                    if (D.rt != 0 && std::abs(S - 1.0) > 1.0e-12)
                        ++O.n_sum, O.worst = std::max(O.worst, std::abs(2.0 * dB(std::abs(S))));
                }
                Node M;
                M.vv = Nn.vv * cpx(E.xpr, 0, 0);
                double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                M.y0 = Nn.depth ? Nn.y0 : fy;
                M.Lin = Nn.Lin + (Nn.inside ? std::sqrt((fx - Nn.xp) * (fx - Nn.xp) + (fy - Nn.yp) * (fy - Nn.yp)) : 0.0);
                M.xp = fx, M.yp = fy, M.depth = Nn.depth + 1;
                const int cur = mat_of(E.cur(0));
                M.inside = cur != 0;
                arma::mat d = E.destN - E.origN;
                d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                if (!M.inside) // in air: collected
                {
                    std::complex<double> F = M.vv * std::exp(-J * k1 * M.Lin) * std::exp(J * k1 * st * (fy - M.y0));
                    if (d(0, 0) > 0.0 && fx > x_back - 1.0e-3)
                        O.T += F, ++O.nT;
                    else if (d(0, 0) < 0.0 && fx < 1.0e-3)
                        O.R += F, ++O.nR;
                    continue;
                }
                if (M.depth >= max_ev || std::norm(M.vv) < floor_lin)
                {
                    ++O.n_cut;
                    continue;
                }
                const bool declined_refl = !tx && (E.cur(0) & (short)0x8000) == 0;
                M.followed = declined_refl;
                M.P = E, M.have = true;
                if (mode == 2 && declined_refl && (E.rt & 0x40) == 0) // a bare face: the fixture writes what the ray was reflected off
                {
                    int m = (two_solids && std::abs(fx - 0.05) < 1.0e-3) ? (cur == A ? B : A) : 0;
                    M.P.prev(0) = (short)((E.prev(0) & (short)0x8000) | m | (m ? 0x4000 : 0));
                }
                M.orig = E.origN, M.dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
                stack.push_back(M);
            }
        }
        return O;
    };

    QD_DIAG("\n# diag: mirror behind, every path. One ray, every reflected and transmitted ray followed until it is in air, dead\n");
    QD_DIAG("# or 20 events old, summed at the polyline's point, against the matrix of the wall (HSH: against the run at\n");
    QD_DIAG("# eps 0 with 60 events), dB. lib: the library at eps 0.15; inh: eps 1.5 on every leg that began at a declined\n");
    QD_DIAG("# reflection; true: the mirror written into the word after a declined reflection at a bare face; res: eps 0.\n");
    QD_DIAG("# n: the paths behind and in front of the wall, lib. sum: the events on a leg that began at a declined reflection\n");
    QD_DIAG("# and summed a series, lib; worth: the largest of them, dB; sum_true: the same count with the word made true.\n");
    QD_DIAG("# cut: the rays stopped at 20 events, lib\n");
    QD_DIAG("wall,a,b,tracer,theta,nT,nR,cut,sum,worth,sum_true,T_lib,T_inh,T_true,T_res,R_lib,R_inh,R_true,R_res,T_inh-true,R_inh-true\n");
    std::vector<Wall> walls = {{"H", 5, 0}, {"H", 6, 0}, {"HS", 5, 0}, {"HS", 6, 0}, {"SH", 5, 0}, {"SH", 6, 0}, {"AB", 2, 5}, {"AB", 2, 6}, {"HSH", 5, 0}, {"HSH", 6, 0}};
    if (qd_diag())
    {
        for (const Wall &w : walls)
            for (int tr : {4, 5})
                for (double thd : angles)
                {
                    const std::string nm = w.name;
                    double s2 = std::sin(thd * QPI / 180.0) * std::sin(thd * QPI / 180.0);
                    arma::uvec mi;
                    arma::mat mesh = mesh_of(nm, w.a, w.b, mi);
                    const double x_back = (nm == "AB") ? 0.15 : 0.10;
                    const bool two = nm == "AB";
                    Out L = walk(mesh, mi, tr, 0.15, 0, thd, 20, -400.0, x_back, w.a, w.b, two);
                    Out I = walk(mesh, mi, tr, 0.15, 1, thd, 20, -400.0, x_back, w.a, w.b, two);
                    Out U = walk(mesh, mi, tr, 0.15, 2, thd, 20, -400.0, x_back, w.a, w.b, two);
                    Out Z = walk(mesh, mi, tr, 0.0, 0, thd, 20, -400.0, x_back, w.a, w.b, two);
                    std::complex<double> eA = diag_eps(w.a, FRQ), eB = diag_eps(w.b, FRQ), eS = diag_eps(SHEET, FRQ);
                    RT W;
                    if (nm == "H")
                        W = cm(e_air, {{eA, 0.10}}, e_air, s2, k1);
                    else if (nm == "HS")
                        W = cm(e_air, {{eA, 0.10}, {eS, thk}}, e_air, s2, k1);
                    else if (nm == "SH")
                        W = cm(e_air, {{eS, thk}, {eA, 0.10}}, e_air, s2, k1);
                    else if (nm == "AB")
                        W = cm(e_air, {{eA, 0.05}, {eB, 0.10}}, e_air, s2, k1);
                    else
                    {
                        Out Z60 = walk(mesh, mi, tr, 0.0, 0, thd, 60, -400.0, x_back, w.a, w.b, two);
                        W.t = Z60.T, W.r = Z60.R;
                    }
                    QD_DIAG("%s,%d,%d,%d,%.0f,%d,%d,%d,%d,%.3f,%d,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.1e,%+.1e\n", w.name, w.a, w.b, tr, thd,
                            L.nT, L.nR, L.n_cut, L.n_sum, L.worst, U.n_sum, dB(std::norm(L.T / W.t)), dB(std::norm(I.T / W.t)), dB(std::norm(U.T / W.t)),
                            dB(std::norm(Z.T / W.t)), dB(std::norm(L.R / W.r)), dB(std::norm(I.R / W.r)), dB(std::norm(U.R / W.r)), dB(std::norm(Z.R / W.r)),
                            dB(std::norm(I.T / U.T)), dB(std::norm(I.R / U.R)));
                }
    }

    // ---- D. A verdict that is handed in ----
    // The event that closes the first leg in a body, both passes: with the verdict "sum" handed
    // in it is the event at eps 0 whatever eps says, with "do not sum" the event at eps 1.5, and
    // the verdict handed out says which of the two a call was.
    for (const char *wn : {"H", "HS", "AB"})
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (int itype : {3, tr})
                    for (double thd : {10.0, 40.0, 60.0, 75.0, 85.0})
                    {
                        arma::uvec mi;
                        arma::mat mesh = mesh_of(wn, std::string(wn) == "AB" ? 2 : h, h, mi), orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev E1 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 1.5, true);
                        arma::mat d = E1.destN - E1.origN;
                        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                        orig = E1.origN, dest = E1.origN + (1.2 / std::abs(d(0, 0))) * d;
                        const bool tx = itype != 3;
                        const std::vector<uint8_t> yes = {1}, no = {0};
                        std::vector<uint8_t> v0, v15, vy, vn;
                        Ev S0 = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 0.0, tx, nullptr, &v0);
                        Ev S15 = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 1.5, tx, nullptr, &v15);
                        Ev Sy = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 1.5, tx, &yes, &vy);
                        Ev Sn = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 0.0, tx, &no, &vn);
                        INFO("verdict, wall " << wn << ", host " << h << ", tracer " << tr << ", type " << itype << ", theta " << thd);
                        REQUIRE((S0.rt != 0 && S15.rt != 0));
                        CHECK(cpx(S0.xpr, 0, 0) != cpx(S15.xpr, 0, 0)); // the two are told apart
                        CHECK(cpx(Sy.xpr, 0, 0) == cpx(S0.xpr, 0, 0));
                        CHECK(cpx(Sn.xpr, 0, 0) == cpx(S15.xpr, 0, 0));
                        CHECK((Sy.cur(0) == S0.cur(0) && Sy.buf(0) == S0.buf(0) && Sy.prev(0) == S0.prev(0)));
                        CHECK((Sn.cur(0) == S15.cur(0) && Sn.buf(0) == S15.buf(0) && Sn.prev(0) == S15.prev(0)));
                        CHECK((v0[0] == 1 && v15[0] == 0 && vy[0] == 1 && vn[0] == 0));
                    }
}

TEST_CASE("Sheet reflect port - where the layer lies and what the echo pays", "[wave]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // A sheet has no thickness in the mesh. Its composite stands for a layer of its thickness
    // that lies around the mesh plane, and the tracer turns a reflected ray around at that plane,
    // so the reflect port owes the round trip through the part of the layer in front of it:
    //   glued entry, air | sheet | solid     the layer lies in the air in front, all of it
    //   one medium on both sides             the mesh plane is the mid-plane, half of it
    //   glued exit, solid | sheet | air      the layer lies in the air beyond, none
    // And the part of the reflection that entered the sheet pays the sheet's interface gain
    // once, as the echo of a traced solid does. Through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update, scalar, double.
    //
    // Every reflected sum is the plane-wave reflection referenced to the plane x = 0 and every
    // transmitted one the transfer from the plane the polyline enters at to the one it leaves at.
    //
    // A. A wall coated on its near side through ray_mesh_interact alone, no state
    // B. A wall coated on its near side and a wall lined on both faces, every path
    // C. An absorber on a rigid wall, as a sheet and as a traced solid
    // D. A sheet in front of a wall across an air gap
    // E. A sheet inside a wall
    // F. A sheet and a traced solid of one material and thickness, with att
    const double gap = 1.0e-5;
    const std::complex<double> J(0.0, 1.0), one(1.0, 0.0);
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, xpr0, fbs; // xpr0: what ray_mesh_interact wrote
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch, bool stateless = false) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        E.xpr0 = E.xpr, E.ot = ot[0], E.nh = nh(0);
        if (stateless)
            return E;
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.rt = rt[0];
        return E;
    };
    using ub::build;
    using ub::face;
    auto onward = [](const Ev &E, arma::mat &orig, arma::mat &dest) // long enough to cross every face that is left
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
        return d(0, 0);
    };

    // Every path. One ray, both passes at every event, every ray followed until it has left the
    // structure, is dead, max_ev events old or below the floor. A ray in air between two bodies
    // is followed like any other.
    struct Out
    {
        std::complex<double> T = 0.0, R = 0.0, R0 = 0.0; // R0: the path that is reflected at the first event
        int nT = 0, nR = 0, n_cut = 0;
    };
    struct Node
    {
        arma::mat orig, dest;
        Ev P;
        bool have = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, Lin = 0.0, dirx = 1.0;
        std::complex<double> vv = 1.0;
        int depth = 0;
    };
    auto walk = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double eps, double thd, int max_ev,
                    double floor_dB, double x_front, double x_back) -> Out
    {
        Out O;
        const double st = std::sin(thd * QPI / 180.0), floor_lin = std::pow(10.0, 0.1 * floor_dB), k = 2.0 * QPI * f / C0;
        std::vector<Node> stack(1);
        diag_ray_at(thd, stack[0].orig, stack[0].dest);
        while (!stack.empty())
        {
            Node Nn = stack.back();
            stack.pop_back();
            for (int itype : {3, tr})
            {
                const bool tx = itype != 3;
                Ev E = traced(pal, mesh, mi, itype, f, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, eps, tx);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                Node M;
                M.vv = Nn.vv * cpx(E.xpr, 0, 0);
                double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                M.y0 = Nn.depth ? Nn.y0 : fy;
                M.Lin = Nn.Lin + (Nn.depth ? std::sqrt((fx - Nn.xp) * (fx - Nn.xp) + (fy - Nn.yp) * (fy - Nn.yp)) : 0.0);
                M.xp = fx, M.yp = fy, M.depth = Nn.depth + 1;
                M.dirx = onward(E, M.orig, M.dest);
                if (mat_of(E.cur(0)) == 0 && ((M.dirx < 0.0 && fx < x_front + 1.0e-3) || (M.dirx > 0.0 && fx > x_back - 1.0e-3)))
                {
                    std::complex<double> F = M.vv * std::exp(-J * k * M.Lin) * std::exp(J * k * st * (fy - M.y0));
                    if (M.dirx > 0.0)
                        O.T += F, ++O.nT;
                    else
                    {
                        O.R += F, ++O.nR;
                        if (M.depth == 1)
                            O.R0 = F;
                    }
                    continue;
                }
                if (M.depth >= max_ev || std::norm(M.vv) < floor_lin)
                {
                    ++O.n_cut;
                    continue;
                }
                M.P = E, M.have = true;
                stack.push_back(M);
            }
        }
        return O;
    };

    // The reference: the scalar channel, admittance b / mu with b = sqrt(eps mu - sin^2), the
    // decaying root, layer phase k b d. It shares nothing with the library.
    struct Lay
    {
        std::complex<double> e, mu;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, std::complex<double> mu, double s2)
    {
        std::complex<double> b = std::sqrt(e * mu - s2);
        return (std::imag(b) > 0.0) ? -b : b;
    };
    auto cm = [&](const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT // from air
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, l.mu, s2), y = b / l.mu, dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / y, a21 = J * y * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(one, one, s2), hf = bof(ef, one, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    const std::vector<double> angles = {10.0, 25.0, 40.0, 55.0, 70.0, 85.0};

    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const double thk = 0.006, k1 = 2.0 * QPI * FRQ / C0;
    const std::complex<double> eS = diag_eps(SHEET, FRQ);

    // A reflected sum against its wall. Where the wall reflects 10 dB less than the path off its first face, the two
    // nearly cancel and the same distance reads larger in dB; such a row, and a row whose mirror is clamped above one,
    // is held against the first-face term.
    auto on_wall = [&](std::complex<double> R, std::complex<double> ref, std::complex<double> R0, bool clamp) -> bool
    {
        if (clamp || std::abs(ref) < 0.316 * std::abs(R0))
            return std::abs(R - ref) < 5.0e-3 * std::abs(R0);
        return std::abs(dB(R / ref)) < 5.0e-3 && std::abs(deg(R / ref)) < 0.05;
    };

    // ---- A. A wall coated on its near side through ray_mesh_interact alone ----
    // What a caller without state gets. First the glued entry itself, both ports, against the
    // redo of ray_state_update. Then, on the lossless host with the refracting tracer, whose
    // segments are the wave, the paths of the wall from ray_mesh_interact's coefficients alone,
    // the host legs as exp(-j k0 n L) over the polyline: reflected off the coating; in, off the
    // far face, out; and the same with one more bounce. The three give the limit of the sequence.
    QD_DIAG("\n# diag: reflect port, a wall coated on its near side, ray_mesh_interact alone. air | coating 6 mm | host 100 | air.\n");
    QD_DIAG("# dR_own, dT_own: the glued entry of ray_mesh_interact against the redo of ray_state_update, complex difference.\n");
    QD_DIAG("# Host 6 and tracer 5 only, no state: R_dB, R_deg: the limit of the reflected paths against the matrix of\n");
    QD_DIAG("# coating | host with the coating in front of the mesh plane; T_dB: the limit of the transmitted paths against the\n");
    QD_DIAG("# matrix. R1/R0: dB\n");
    QD_DIAG("host,tracer,theta,dR_own,dT_own,R1/R0_dB,R_dB,R_deg,T_dB\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, gap, 0.10 + gap}, {SHEET, h, h}, {false, false, true}, mi), orig, dest;
                diag_ray_at(thd, orig, dest);
                Ev er = traced(mtl, mesh, mi, 3, FRQ, orig, dest, nullptr, 0.0, false), et = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 0.0, true);
                INFO("own composite, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((er.nh != 0 && et.nh != 0));
                double dR = std::abs(cpx(er.xpr0, 0, 0) - cpx(er.xpr, 0, 0)), dT = std::abs(cpx(et.xpr0, 0, 0) - cpx(et.xpr, 0, 0));
                CHECK(dR < 1.0e-12); // the two functions form one composite for an entry from air
                CHECK(dT < 1.0e-12);
                if (h != 6 || tr != 5)
                {
                    QD_DIAG("%d,%d,%.0f,%.3e,%.3e\n", h, tr, thd, dR, dT);
                    continue;
                }
                const double n_host = std::real(std::sqrt(diag_eps(h, FRQ)));
                auto stateless = [&](const std::vector<int> &seq, bool &ok) -> std::complex<double>
                {
                    arma::mat o, d;
                    diag_ray_at(thd, o, d);
                    std::complex<double> v = 1.0;
                    double Lin = 0.0, xp = 0.0, yp = 0.0, y0 = 0.0;
                    bool inside = false;
                    ok = true;
                    for (size_t q = 0; q < seq.size(); ++q)
                    {
                        Ev E = traced(mtl, mesh, mi, seq[q], FRQ, o, d, nullptr, 0.0, false, true);
                        if (E.nh == 0)
                        {
                            ok = false;
                            break;
                        }
                        v *= cpx(E.xpr0, 0, 0);
                        double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                        if (q == 0)
                            y0 = fy;
                        else if (inside)
                            Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
                        if (seq[q] != 3)
                            inside = !inside;
                        xp = fx, yp = fy;
                        onward(E, o, d);
                    }
                    return v * std::exp(-J * k1 * n_host * Lin) * std::exp(J * k1 * st * (yp - y0));
                };
                bool ok0, ok1, ok2, ok3, ok4;
                std::complex<double> R0 = stateless({3}, ok0), R1 = stateless({tr, 3, tr}, ok1), R2 = stateless({tr, 3, 3, 3, tr}, ok2);
                std::complex<double> T1 = stateless({tr, tr}, ok3), T2 = stateless({tr, 3, 3, tr}, ok4);
                REQUIRE((ok0 && ok1 && ok2 && ok3 && ok4));
                RT W = cm({{eS, one, thk}, {diag_eps(h, FRQ), one, 0.10 + gap}}, one, s2, k1);
                std::complex<double> ref = W.r * std::exp(2.0 * J * k1 * std::cos(th) * thk); // the coating in front of the plane
                std::complex<double> Ra = R0 + R1 / (1.0 - R2 / R1), T = T1 / (1.0 - T2 / T1);
                QD_DIAG("%d,%d,%.0f,%.3e,%.3e,%+.2f,%+.6f,%+.4f,%+.6f\n", h, tr, thd, dR, dT, dB(R1 / R0), dB(Ra / ref), deg(Ra / ref), dB(T / W.t));
                CHECK(on_wall(Ra, ref, R0, false));
                CHECK(std::abs(dB(T / W.t)) < 5.0e-3);
            }

    // ---- B. A wall coated on its near side and a wall lined on both faces, every path ----
    QD_DIAG("\n# diag: reflect port, a wall coated on its near side (SH) and lined on both faces (SHS), every path. coating 6 mm,\n");
    QD_DIAG("# host 100. eps 0 and 0.15, 30 events. R_dB, R_deg: the reflected sum against the matrix of the wall with the near\n");
    QD_DIAG("# coating in front of the mesh plane; R_ff: its distance from that wall over the path off the first coating.\n");
    QD_DIAG("# T_dB: the transmitted sum against the matrix; Td_deg: its phase with exp(-j k0 cos(theta) thk) per coating taken\n");
    QD_DIAG("# out. The rows at 85 deg on host 5 carry the clamp of section 5\n");
    QD_DIAG("wall,host,tracer,eps,theta,nT,nR,R_dB,R_deg,R_ff,T_dB,Td_deg\n");
    for (int lined = 0; lined < 2; ++lined)
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (double eps : {0.0, 0.15})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = lined ? build({0.0, gap, 0.10 + gap, 0.10 + 2.0 * gap}, {SHEET, h, h, SHEET}, {false, false, true, true}, mi)
                                               : build({0.0, gap, 0.10 + gap}, {SHEET, h, h}, {false, false, true}, mi);
                        arma::mat orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev e0 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 1.5, true);
                        onward(e0, orig, dest);
                        Ev e1 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, &e0, 1.5, true);
                        const double d_host = std::abs(e1.fbs(0, 0) - e0.fbs(0, 0)); // the thickness the leg crossed
                        Out O = walk(mtl, mesh, mi, tr, FRQ, eps, thd, 30, -300.0, 0.0, 0.10);
                        std::vector<Lay> ls = {{eS, one, thk}, {diag_eps(h, FRQ), one, d_host}};
                        if (lined)
                            ls.push_back({eS, one, thk});
                        RT W = cm(ls, one, s2, k1);
                        std::complex<double> D1 = std::exp(-J * k1 * std::cos(th) * thk), ref = W.r / (D1 * D1);
                        const bool clamp = h == 5 && thd == 85.0;
                        INFO("lined " << lined << ", host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                        CHECK((O.nT >= 1 && O.nR >= 2));
                        QD_DIAG("%s,%d,%d,%.2f,%.0f,%d,%d,%+.6f,%+.4f,%.2e,%+.6f,%+.4f\n", lined ? "SHS" : "SH", h, tr, eps, thd, O.nT, O.nR, dB(O.R / ref),
                                deg(O.R / ref), std::abs(O.R - ref) / std::abs(O.R0), dB(O.T / W.t), deg(O.T * (lined ? D1 * D1 : D1) / W.t));
                        CHECK(on_wall(O.R, ref, O.R0, clamp));
                        if (!clamp)
                            CHECK(std::abs(dB(O.T / W.t)) < 5.0e-3);
                    }

    // ---- C. An absorber on a rigid wall, as a sheet and as a traced solid ----
    // Mineral wool 50 mm, the two-column-pair fit of the phase-reference case, on a rigid wall
    // eps 2.401e-9 whose face stands at x = 0 in both meshes. As a sheet the wool is a face at
    // x = 0, co-located with the wall; as a traced solid it stands in front of the wall, from
    // x = -50 mm. The two are one wall. The reference is the layer on that wall with its front
    // face at x = -50 mm, so referenced to x = 0 it carries exp(+2j k0 cos(theta) W). Its mirror
    // wool | rigid is the Fresnel value, above one where that is above one,
    // which is what a sheet does and what a traced solid does at a closed face, so the two are
    // asserted against each other on every row.
    {
        const double W = 0.05, fscale = C0 / 343.0;
        Pal pal; // 1: wool, a solid; 2: the rigid wall; 3: wool, a sheet of 50 mm
        pal["a"] = {1.16, 2.401e-9, 1.16}, pal["b"] = {-0.04, 0.0, -0.04}, pal["c"] = {0.0038, 0.0, 0.0038}, pal["d"] = {1.46, 0.0, 1.46};
        pal["e"] = {1.66, 1.0, 1.66}, pal["f"] = {-0.19, 0.0, -0.19}, pal["g"] = {0.091, 0.0, 0.091}, pal["h"] = {0.13, 0.0, 0.13};
        pal["fRef"] = {0.874, 0.874, 0.874}, pal["thk"] = {0.0, 0.0, W};
        QD_DIAG("\n# diag: reflect port, an absorber on a rigid wall. air | mineral wool 50 mm | rigid, scalar, the wall face at x = 0.\n");
        QD_DIAG("# |R|_dB: the reflect port of the sheet. own: ray_mesh_interact's against the redo, complex difference. sheet:\n");
        QD_DIAG("# the sheet against the reference, complex distance. s/s_dB, s/s_deg: the sheet over the summed paths of the\n");
        QD_DIAG("# traced solid. owed_deg: 2 k0 cos(theta) W, not wrapped. r12_dB: the Fresnel reflection wool | rigid\n");
        QD_DIAG("tracer,f_Hz,theta,nR,|R|_dB,own,sheet,s/s_dB,s/s_deg,owed_deg,r12_dB\n");
        for (int tr : {4, 5})
            for (double fa : {250.0, 500.0, 1000.0, 2000.0, 4000.0})
                for (double thd : {20.0, 40.0, 60.0, 75.0})
                {
                    const double f1 = fa * fscale, fG = f1 * 1e-9, fr = fG / 0.874, th = thd * QPI / 180.0, k = 2.0 * QPI * f1 / C0;
                    const double st = std::sin(th), s2 = st * st;
                    std::complex<double> es(1.16 * std::pow(fr, -0.04), -17.98 * 0.0038 * std::pow(fr, 1.46) / fG);
                    std::complex<double> mus(1.66 * std::pow(fr, -0.19), -17.98 * 0.091 * std::pow(fr, 0.13) / fG);
                    arma::uvec miS, miB;
                    arma::mat meshS = build({0.0, gap, gap + 0.2}, {3, 2, 2}, {false, false, true}, miS);
                    arma::mat meshB = build({-W, 0.0, gap, gap + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, miB), orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev er = traced(pal, meshS, miS, 3, f1, orig, dest, nullptr, 0.0, false);
                    Out O = walk(pal, meshB, miB, tr, f1, 0.0, thd, 30, -200.0, -W, gap + 0.2);
                    INFO("absorber, tracer " << tr << ", f " << fa << ", theta " << thd);
                    REQUIRE((er.nh != 0 && er.rt != 0 && O.nR >= 1));
                    std::complex<double> owed = std::exp(2.0 * J * k * std::cos(th) * W);
                    std::complex<double> A_sheet = cpx(er.xpr, 0, 0), A_solid = O.R * owed; // both referenced to x = 0
                    std::complex<double> y0 = bof(one, one, s2), b1 = bof(es, mus, s2), y1 = b1 / mus, y2 = bof(std::complex<double>(2.401e-9, 0.0), one, s2);
                    std::complex<double> r01 = (y0 - y1) / (y0 + y1), r12 = (y1 - y2) / (y1 + y2), p2 = std::exp(-2.0 * J * k * b1 * W);
                    const double r12_dB = dB(r12), own = std::abs(cpx(er.xpr0, 0, 0) - cpx(er.xpr, 0, 0));
                    std::complex<double> ref = (r01 + r12 * p2) / (1.0 + r01 * r12 * p2) * owed;
                    QD_DIAG("%d,%.0f,%.0f,%d,%+.4f,%.3e,%.3e,%+.6f,%+.4f,%.3f,%+.3f\n", tr, fa, thd, O.nR, dB(A_sheet), own, std::abs(A_sheet / ref - 1.0),
                            dB(A_sheet / A_solid), deg(A_sheet / A_solid), 2.0 * k * std::cos(th) * W * 180.0 / QPI, r12_dB);
                    CHECK(own < 1.0e-12); // a glued entry under a closed outer pair, both functions
                    CHECK(std::abs(A_sheet / ref - 1.0) < 1.0e-9);
                    // The sheet and the traced solid are one wall
                    CHECK(std::abs(dB(A_sheet / A_solid)) < 5.0e-3);
                    CHECK(std::abs(deg(A_sheet / A_solid)) < 0.05);
                }
    }

    // ---- D. A sheet in front of a wall across an air gap ----
    // air | sheet 6 mm | air 30 mm | host 100 | air, the sheet's face at x = 0 and the wall's at x = 30 mm. The layer
    // of the sheet occupies 6 mm of that air, [-u, 6 mm - u]. Three walls, each referenced to x = 0: u = 0 (behind the
    // face, in the gap), 3 mm (centred) and 6 mm (in front). The mesh plane is the mid-plane, so the paths sum to the
    // centred wall on both ports, and the other two are printed to show they are told apart.
    QD_DIAG("\n# diag: reflect port, a sheet in front of a wall across an air gap. air | sheet 6 mm | air 30 mm | host 100 | air,\n");
    QD_DIAG("# eps 0, every path, floor -120 dB. The summed paths against the wall whose layer lies at [-u, thk - u]: uh centred,\n");
    QD_DIAG("# asserted; u0 behind the face and ut in front, recorded. dB and deg. cut: rays stopped at the floor\n");
    QD_DIAG("host,tracer,theta,nT,nR,cut,R_uh,Rdeg_uh,T_uh,Tdeg_uh,R_u0,Rdeg_u0,T_u0,R_ut,Rdeg_ut,T_ut\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : {10.0, 40.0, 70.0})
            {
                const double g = 0.03, th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), ck = k1 * std::cos(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, g, g + 0.10}, {SHEET, h, h}, {false, false, true}, mi);
                Out L = walk(mtl, mesh, mi, tr, FRQ, 0.0, thd, 200, -120.0, 0.0, g + 0.10);
                std::complex<double> rr[3], tt[3];
                const double uu[3] = {0.0, 0.5 * thk, thk};
                for (int q = 0; q < 3; ++q)
                {
                    RT Wu = cm({{eS, one, thk}, {one, one, g - thk + uu[q]}, {diag_eps(h, FRQ), one, 0.10}}, one, s2, k1);
                    rr[q] = Wu.r * std::exp(2.0 * J * ck * uu[q]), tt[q] = Wu.t * std::exp(J * ck * uu[q]);
                }
                INFO("air gap, host " << h << ", tracer " << tr << ", theta " << thd);
                CHECK((L.nT >= 1 && L.nR >= 2));
                QD_DIAG("%d,%d,%.0f,%d,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.4f,%+.3f,%+.4f\n", h, tr, thd, L.nT, L.nR, L.n_cut, dB(L.R / rr[1]),
                        deg(L.R / rr[1]), dB(L.T / tt[1]), deg(L.T / tt[1]), dB(L.R / rr[0]), deg(L.R / rr[0]), dB(L.T / tt[0]), dB(L.R / rr[2]),
                        deg(L.R / rr[2]), dB(L.T / tt[2]));
                CHECK(std::abs(dB(L.R / rr[1])) < 5.0e-3);
                CHECK(std::abs(deg(L.R / rr[1])) < 0.05);
                CHECK(std::abs(dB(L.T / tt[1])) < 5.0e-3);
                CHECK(std::abs(deg(L.T / tt[1])) < 0.05);
                CHECK(std::abs(L.R / rr[0] - 1.0) > 1.0e-2); // the three walls are told apart
                CHECK(std::abs(L.R / rr[2] - 1.0) > 1.0e-2);
            }

    // ---- E. A sheet inside a wall ----
    // air | host 50 | sheet 6 mm | host 50 | air. The sheet displaces 3 mm of host on either side of its face, so the
    // wall is host 47 | sheet | host 47, on both ports, whether every series is summed, every bounce followed, or the
    // threshold falls between. w44: the same sums against host 50 | sheet | host 44, printed.
    QD_DIAG("\n# diag: reflect port, a sheet inside a wall. air | host 50 | sheet 6 mm | host 50 | air, every path, floor -100 dB;\n");
    QD_DIAG("# eps 1.5 with 20 events, eps 0 and 0.15 with 60. The summed paths against host 47 | sheet | host 47, dB and deg;\n");
    QD_DIAG("# w44: against host 50 | sheet | host 44, dB\n");
    QD_DIAG("host,tracer,eps,theta,nT,nR,cut,R_dB,R_deg,T_dB,T_deg,R_w44,T_w44\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double eps : {1.5, 0.0, 0.15})
                for (double thd : {10.0, 40.0, 60.0})
                {
                    const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    const std::complex<double> eH = diag_eps(h, FRQ);
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, 0.05, 0.10}, {h, SHEET, h}, {false, false, true}, mi);
                    Out L = walk(mtl, mesh, mi, tr, FRQ, eps, thd, eps > 1.0 ? 20 : 60, -100.0, 0.0, 0.10);
                    RT W = cm({{eH, one, 0.047}, {eS, one, thk}, {eH, one, 0.047}}, one, s2, k1), W44 = cm({{eH, one, 0.05}, {eS, one, thk}, {eH, one, 0.044}}, one, s2, k1);
                    INFO("sheet inside, host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                    CHECK((L.nT >= 1 && L.nR >= 2));
                    QD_DIAG("%d,%d,%.2f,%.0f,%d,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.4f\n", h, tr, eps, thd, L.nT, L.nR, L.n_cut, dB(L.R / W.r), deg(L.R / W.r),
                            dB(L.T / W.t), deg(L.T / W.t), dB(L.R / W44.r), dB(L.T / W44.t));
                    CHECK(std::abs(dB(L.R / W.r)) < 1.0e-2);
                    CHECK(std::abs(deg(L.R / W.r)) < 0.05);
                    CHECK(std::abs(dB(L.T / W.t)) < 1.0e-2);
                    CHECK(std::abs(deg(L.T / W.t)) < 0.05);
                }

    // The same port without a reference: a wall with the sheet off its middle, host 30 | sheet |
    // host 70, and its mirror image, host 70 | sheet | host 30. The transmission through a layered
    // wall between two half-spaces of one medium is the same from either side.
    QD_DIAG("\n# diag: reflect port, a sheet off the middle of a wall, from either side. air | host 30 | sheet 6 mm | host 70 | air\n");
    QD_DIAG("# over its mirror image, the transmitted sums, eps 1.5, 20 events, floor -100 dB, dB and deg\n");
    QD_DIAG("host,tracer,theta,T_dB,T_deg\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : {10.0, 40.0, 60.0})
            {
                arma::uvec mi, mj;
                arma::mat m37 = build({0.0, 0.03, 0.10}, {h, SHEET, h}, {false, false, true}, mi), m73 = build({0.0, 0.07, 0.10}, {h, SHEET, h}, {false, false, true}, mj);
                Out A = walk(mtl, m37, mi, tr, FRQ, 1.5, thd, 20, -100.0, 0.0, 0.10), B = walk(mtl, m73, mj, tr, FRQ, 1.5, thd, 20, -100.0, 0.0, 0.10);
                INFO("either side, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((A.nT >= 1 && B.nT >= 1));
                QD_DIAG("%d,%d,%.0f,%+.4f,%+.3f\n", h, tr, thd, dB(A.T / B.T), deg(A.T / B.T));
                CHECK(std::abs(dB(A.T / B.T)) < 1.0e-3);
                CHECK(std::abs(deg(A.T / B.T)) < 0.01);
            }

    // ---- F. A sheet and a traced solid of one material and thickness, with att ----
    // 20 mm of eps 5.24 - 0.63j and of eps 5.24, in air, with att 0, 3 and 10 dB. The sheet is
    // one face and one event, its layer centred on the face; the solid is two faces 20 mm apart
    // with its series summed, its front face on the mesh plane. So the sheet's reflection is the
    // solid's times exp(+j k0 cos(theta) d). once: the Airy reflection with the echo, the part
    // that entered the layer, multiplied by sqrt(G). A traced leg is short by the relaunch
    // offset, so the two representations meet at 1e-3 dB and the sheet meets the form at 1e-9.
    QD_DIAG("\n# diag: reflect port, a sheet against a traced solid with att. 20 mm in air, scalar, 1 GHz. E/r: the echo over the\n");
    QD_DIAG("# front-face term of the Airy reflection, dB. sheet: the sheet's reflect port against the form whose echo carries\n");
    QD_DIAG("# sqrt(G), at the mid-plane, complex distance. solid_dB, solid_deg: the summed reflection of the solid at eps 0\n");
    QD_DIAG("# against that form at its front face. s/s: the sheet over the solid, both at the solid's front face. own:\n");
    QD_DIAG("# ray_mesh_interact's reflect port against the redo. T_s/s: the transmit ports, sheet over solid, dB. fol: the\n");
    QD_DIAG("# solid's reflected sum with every bounce followed over the summed one. none_dB: the sheet against the Airy\n");
    QD_DIAG("# reflection without the gain, which is what att is worth on the port\n");
    QD_DIAG("eps_im,att,tracer,theta,E/r_dB,sheet,solid_dB,solid_deg,s/s_dB,s/s_deg,own,T_s/s_dB,fol_dB,none_dB\n");
    for (double ei : {0.63, 0.0})
        for (double att : {0.0, 3.0, 10.0})
        {
            const double d = 0.02;
            LumpMtl solid, sheet;
            solid.a = 5.24, solid.ei = ei, solid.att = att;
            sheet = solid, sheet.thk = d;
            Pal pal = lump_palette({solid, sheet});
            const std::complex<double> e = lump_eps(solid, FRQ);
            const double sg = std::pow(10.0, -0.05 * att);
            for (int tr : {4, 5})
                for (double thd : {0.0, 20.0, 40.0, 60.0, 75.0, 85.0})
                {
                    const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    arma::uvec miS, miB;
                    arma::mat meshS = build({0.0}, {2}, {false}, miS), meshB = build({0.0, d}, {1, 1}, {false, true}, miB), orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev er = traced(pal, meshS, miS, 3, FRQ, orig, dest, nullptr, 0.0, false), et = traced(pal, meshS, miS, tr, FRQ, orig, dest, nullptr, 0.0, true);
                    Out O = walk(pal, meshB, miB, tr, FRQ, 0.0, thd, 30, -300.0, 0.0, d), Fo = walk(pal, meshB, miB, tr, FRQ, 1.5, thd, 60, -300.0, 0.0, d);
                    INFO("att, eps_im " << ei << ", att " << att << ", tracer " << tr << ", theta " << thd);
                    REQUIRE((er.nh != 0 && er.rt != 0 && et.rt != 0 && O.nR >= 2 && O.nT >= 1));
                    std::complex<double> y0 = bof(one, one, s2), y1 = bof(e, one, s2), r01 = (y0 - y1) / (y0 + y1), p2 = std::exp(-2.0 * J * k1 * y1 * d);
                    std::complex<double> echo = -(1.0 - r01 * r01) * r01 * p2 / (1.0 - r01 * r01 * p2), none = r01 + echo, once = r01 + sg * echo;
                    std::complex<double> half = std::exp(J * k1 * std::cos(th) * d), Rs = cpx(er.xpr, 0, 0), own = cpx(er.xpr0, 0, 0);
                    QD_DIAG("%.2f,%.0f,%d,%.0f,%+.2f,%.3e,%+.6f,%+.4f,%+.6f,%+.4f,%.3e,%+.6f,%+.6f,%+.4f\n", ei, att, tr, thd, dB(echo / r01),
                            std::abs(Rs / (once * half) - 1.0), dB(O.R / once), deg(O.R / once), dB(Rs / (O.R * half)), deg(Rs / (O.R * half)),
                            std::abs(own - Rs), dB(cpx(et.xpr, 0, 0) / O.T), dB(Fo.R / O.R), dB(Rs / none));
                    CHECK(std::abs(Rs / (once * half) - 1.0) < 1.0e-9);
                    CHECK(std::abs(own - Rs) < 1.0e-12);
                    CHECK(std::abs(dB(O.R / once)) < 1.0e-3);        // the traced solid pays the gain once
                    CHECK(std::abs(dB(Rs / (O.R * half))) < 1.0e-3); // and the two representations are one wall
                    CHECK(std::abs(deg(Rs / (O.R * half))) < 0.05);
                    CHECK(std::abs(dB(cpx(et.xpr, 0, 0) / O.T)) < 5.0e-3); // a traced leg is short by the relaunch offset
                }
        }
}

TEST_CASE("Direction - a ray leaves a parallel slab parallel to how it arrived", "[wave]")
{
    // Across parallel faces the transverse wavenumber of a wave that started in air is real and conserved, so behind
    // any stack of layers the direction in air has tangential part sqrt(p) and normal part sqrt(1 - p) with p =
    // sin^2(theta_ext): the direction the ray came in at.
    //
    // Measured at every writer of a direction into air, with the undeviated tracer, which is the mode in which
    // ray_state_update writes one: a glued exit, a glued exit behind a sheet and behind an ordered stack inside the
    // host, a bare exit and a bare exit of a resolved ray, the last two through the intersector because a bare exit
    // keeps what ray_mesh_interact wrote unless ray_state_update replaces it. Lossless host as the control.
    //
    // d_t and d_n are the tangential and the normal component of the written unit vector minus those of the incoming
    // one. map_t and map_n are what the real-angle map makes of the traced direction before normalization, minus
    // sqrt(p) and sqrt(1 - p): the tangential shortfall 1 / N of the projection, and the far cosine formed at a real
    // angle in a lossy host.
    //
    // Two kinds of row keep the direction they had and are pinned to it: a host whose recovery declines (no direction
    // supplied, or faces known not parallel), which takes the real-angle map; and a closed outer pair, whose leak is
    // undeviated. The refracting tracer's bare-exit rows are printed.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    const double f_hz = FRQ, L = 0.10;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, SHEET2 = 8, DENSE = 2;
    const uint8_t OT_SHEET_ISO = 67, OT_STACK = 71, OT_GLUED_EXIT = 69;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double th, double dx,
                     double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        C.run();
        return C;
    };
    // The real-angle map out of a medium into air at the real in-medium angle (c, s), unnormalized
    auto real_map = [](std::complex<double> e, double s, double &a, double &b)
    {
        a = std::sqrt(std::abs(e)) * s;
        b = std::real(std::sqrt(1.0 - e * s * s));
    };

    // ---- Glued exit: entered through a bare face, behind a sheet inside the host, behind a stack ----
    QD_DIAG("\n# diag: direction, glued exit. air | host 100 mm | coating 6 mm | air, undeviated tracer, the\n");
    QD_DIAG("# exit along the direction a tracer holds. route: bare-entered, behind a sheet inside the\n");
    QD_DIAG("# host, behind an ordered stack inside the host. d_pass is the direction the sheet or stack\n");
    QD_DIAG("# event hands on against the one it received\n");
    QD_DIAG("route,host,theta_ext,out_deg,d_t,d_n,map_t,map_n,d_pass\n");
    for (int host : {5, 6})
        for (int route = 0; route < 3; ++route)
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                std::complex<double> eh = diag_eps(host, f_hz);
                double ct = diag_traced_ct(eh, th), st = std::sqrt(1.0 - ct * ct);
                double dx = ct, dy = st, d_pass = 0.0;
                int prev = 0, cur = host, buf = 0;
                if (route > 0)
                {
                    auto S = route == 1 ? event(4, OT_SHEET_ISO, 1u, SHEET, 0, 0, host, 0, th, ct, st)
                                        : event(4, OT_STACK, 2u, SHEET, SHEET2, 0, host, 0, th, ct, st);
                    REQUIRE(S.resolved_type[0] != 0);
                    dx = (double)S.path_dirN(0, 0), dy = (double)S.path_dirN(0, 1);
                    d_pass = std::max(std::abs(dx - ct), std::abs(dy - st));
                    prev = (int)S.prev_out(0), cur = (int)S.cur_out(0), buf = (int)S.buf_out(0);
                }
                auto X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, prev, cur, buf, th, dx, dy);
                double ox = std::abs((double)X.path_dirN(0, 0)), oy = std::abs((double)X.path_dirN(0, 1));
                double a, b;
                real_map(eh, st, a, b);
                QD_DIAG("%s,%d,%.1f,%.4f,%+.3e,%+.3e,%+.3e,%+.3e,%.3e\n", route == 0 ? "bare " : (route == 1 ? "sheet" : "stack"),
                        host, thd, std::atan2(oy, ox) * 180.0 / QPI, oy - std::sin(th), ox - std::cos(th), a - std::sin(th),
                        b - std::cos(th), d_pass);
                INFO("glued exit, route " << route << ", host " << host << ", theta_ext " << thd);
                CHECK(d_pass < 1.0e-12); // near == far: the event hands the direction on
                CHECK(std::abs(oy - std::sin(th)) < 1.0e-9);
                CHECK(std::abs(ox - std::cos(th)) < 1.0e-9);
            }

    // ---- Rows that keep the direction they have ----
    QD_DIAG("\n# diag: direction, glued exits that keep it. nodir: no direction supplied, the reported angle\n");
    QD_DIAG("# is the in-medium one. wedge: faces known not parallel. Both take the real-angle map, d_map\n");
    QD_DIAG("# against it. closed: the outer pair is closed, the leak is undeviated, d_map against dir_prev\n");
    QD_DIAG("kind,host,theta,out_deg,d_map\n");
    for (int host : {5, 6})
        for (int kind = 0; kind < 2; ++kind)
            for (double thd : {10.0, 20.0, 40.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> eh = diag_eps(host, f_hz);
                double c_in = std::cos(th), s_in = std::sin(th); // the real in-medium angle the map reads
                Rsu<double> X;
                if (kind == 0)
                {
                    X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, SHEET, host, 0, host, 0, L / c_in, 0.5, 0.0, QPI / 2.0 - th, 1.5);
                    X.orig(0, 0) = X.fbs(0, 0) - L, X.orig(0, 1) = -L * std::tan(th);
                    X.dest(0, 0) = X.fbs(0, 0) + 0.5 * c_in, X.dest(0, 1) = 0.5 * s_in;
                    X.has_path_dirN = true;
                    X.path_dirN.zeros(1, 3);
                    X.path_dirN(0, 0) = c_in, X.path_dirN(0, 1) = s_in;
                    X.run();
                }
                else
                {
                    c_in = diag_traced_ct(eh, th), s_in = std::sqrt(1.0 - c_in * c_in);
                    X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, (int)enc(0, true), host, 0, th, c_in, s_in);
                }
                double a, b;
                real_map(eh, s_in, a, b);
                double n = std::sqrt(a * a + b * b);
                double ox = std::abs((double)X.path_dirN(0, 0)), oy = std::abs((double)X.path_dirN(0, 1));
                double d_map = std::max(std::abs(ox - b / n), std::abs(oy - a / n));
                QD_DIAG("%s,%d,%.1f,%.4f,%.3e\n", kind ? "wedge " : "nodir ", host, thd, std::atan2(oy, ox) * 180.0 / QPI, d_map);
                INFO("kept, kind " << kind << ", host " << host << ", theta " << thd);
                CHECK(d_map < 1.0e-9);
            }
    for (double thd : {60.0, 70.0})
    {
        // Launched in the lossless host past its critical angle, the dense solid named
        double th = thd * QPI / 180.0;
        auto X = event(4, OT_GLUED_EXIT, 2u, SHEET, 6, DENSE, 6, 0, th, std::cos(th), std::sin(th));
        double d_map = std::max(std::abs((double)X.path_dirN(0, 0) - std::cos(th)), std::abs((double)X.path_dirN(0, 1) - std::sin(th)));
        QD_DIAG("closed,6,%.1f,%.4f,%.3e\n", thd,
                std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI, d_map);
        INFO("closed outer pair, theta " << thd);
        CHECK(rt_bit(X.resolved_type[0], 5));
        CHECK(d_map < 1.0e-12);
    }

    // ---- Bare exit, through the intersector ----
    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, const arma::mat &orig, const arma::mat &dest, const Ev *in,
                      double eps) -> Ev
    {
        arma::mat fbs, sbs, fbsN, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &fbsN, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &fbsN, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    // The next segment: from where the event relaunched, 2 m along the direction it relaunched in
    auto onward = [](const Ev &E, arma::mat &orig, arma::mat &dest)
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + 2.0 * d;
    };
    // A face at x with its normal along -x (flip false) or +x (flip true)
    using ub::face;

    QD_DIAG("\n# diag: direction, bare exit. air | slab 100 mm | air through the intersector, two events, and\n");
    QD_DIAG("# three for the resolved ray: enter, reflect at the far face with the series collapsed, leave\n");
    QD_DIAG("# through the front face, where the ray has to leave along the mirror of how it arrived.\n");
    QD_DIAG("# itype 4 is the undeviated tracer, 5 the refracting one, whose rows are recorded\n");
    QD_DIAG("itype,route,mtl,theta_ext,code,rtype,out_deg,d_t,d_n\n");
    {
        arma::mat mesh = arma::join_cols(face(0.0, false), face(L, true));
        for (int itype : {4, 5})
            for (int route = 0; route < 2; ++route)
                for (int m : {5, 6})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0;
                        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));
                        arma::mat orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev A = traced(mtl, mesh, mtl_ind, itype, orig, dest, nullptr, 0.0);
                        onward(A, orig, dest);
                        Ev B = traced(mtl, mesh, mtl_ind, route ? 3 : itype, orig, dest, &A, 0.0);
                        if (route)
                        {
                            REQUIRE(rt_bit(B.rt, 2)); // the reflection collapsed the series
                            onward(B, orig, dest);
                            B = traced(mtl, mesh, mtl_ind, itype, orig, dest, &B, 0.0);
                        }
                        double ox = std::abs(B.path_dirN(0, 0)), oy = std::abs(B.path_dirN(0, 1));
                        QD_DIAG("%d,%s,%d,%.1f,%u,%u,%.4f,%+.3e,%+.3e\n", itype, route ? "resolved" : "through ", m, thd,
                                (unsigned)B.ot, (unsigned)B.rt, std::atan2(oy, ox) * 180.0 / QPI, oy - std::sin(th), ox - std::cos(th));
                        INFO("bare exit, itype " << itype << ", route " << route << ", material " << m << ", theta_ext " << thd);
                        REQUIRE((B.ot & 0x1F) == 1);
                        REQUIRE(mat_of(B.cur(0)) == 0);
                        CHECK((B.path_dirN(0, 0) > 0.0) == (route == 0)); // onward, or back out of the front face
                        if (itype == 4)
                        {
                            CHECK(std::abs(oy - std::sin(th)) < 1.0e-9);
                            CHECK(std::abs(ox - std::cos(th)) < 1.0e-9);
                        }
                    }
    }
}

TEST_CASE("Refracting tracer - a traced bend is known from the state and the leg is exact", "[wave]")
{
    ub::OneThread one_thread;
    // With types 2 and 5 ray_mesh_interact bends the segment itself, so inside a solid the traced and the geometric
    // cosine coincide and nothing can be read off their difference. That the leg follows a bend is a fact of the
    // polyline, and the event that entered the medium knows it: bit 14 of the current-medium word, set where a type 2
    // or 5 event refracted into the medium, clear where it entered under total reflection, carried while the ray stays
    // in the medium and zero in air. Every consumer of the recovered cosine then runs on that leg as it does for the
    // undeviated tracer, and the traversal takes no walk-off, since the lateral shift is in the polyline.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update. The quantity compared is
    // the absolute transfer between the first and the last hit point, the chain coefficient with the free-space
    // reference over the in-wall polyline put back:
    //   - against the analytic slab at the polyline's exit point, power and phase;
    //   - tracer 5 against tracer 4. Each traversal carries the phase at the point its own polyline leaves at, so each
    //     absolute transfer is taken there, times exp(+j k0 sin(theta) dy) with dy that polyline's lateral walk, and
    //     the two are one number.
    // Printed and not asserted: the trajectory (destN - origN) out of a lossy wall.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, gap = 1.0e-5;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const short BENT = (short)0x4000;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    auto dB = [](double g)
    { return 10.0 * std::log10(g); };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    // relaunch: hand ray_state_update the geometry ray_mesh_interact built, so that the segment out of
    // a lossy medium follows the direction it writes. Off for every table but the last one.
    bool relaunch = false;
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, double f, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, const Ev *in_state, double eps) -> Ev
    {
        // in: direction and refracted ledger; in_state: the state words and the geometric ledger,
        // which in a multi-frequency run are the base frequency's
        if (!in_state)
            in_state = in;
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in_state->prev : nullptr, in ? &in_state->cur : nullptr, in ? &in_state->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in_state->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        std::complex<double> vv = 1.0, hh = 1.0; // chain coefficient, VV (TE, scalar) and HH (TM)
        double Lin = 0.0, dy = 0.0;              // in-wall polyline length, lateral walk first to last hit
        double out_x = 0.0, out_y = 0.0, traj = 0.0;
        std::vector<Ev> ev;
        bool ok = true;
        std::complex<double> abs_vv(double k) const { return vv * std::exp(std::complex<double>(0.0, -k * Lin)); }
    };
    // seq: one type repeated until the ray is out, or one type per event. f_base != 0: geometry
    // and state words at f_base, coefficient and direction at f, as ray_progress runs a band.
    auto run = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh, const arma::uvec &mtl_ind,
                   std::vector<int> seq, double thd, double eps, double f, double f_base) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P, Pb;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            Ev B = traced(pal, mesh, mtl_ind, itype, f_base != 0.0 ? f_base : f, orig, dest, have ? &Pb : nullptr, nullptr, eps);
            if (B.nh == 0)
                break;
            Ev E = f_base != 0.0 ? traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, have ? &Pb : nullptr, eps) : B;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0), C.hh *= cpx(E.xpr, 0, (E.xpr.n_rows == 8) ? 3 : 0);
            double fx = B.fbs(0, 0), fy = B.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(B.cur(0)) != 0;
            C.out_x = std::abs(E.path_dirN(0, 0)), C.out_y = std::abs(E.path_dirN(0, 1));
            arma::mat d = B.destN - B.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            C.traj = std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI;
            orig = B.origN, dest = B.origN + 2.0 * d;
            P = E, Pb = B, have = true;
            C.ev.push_back(E);
        }
        return C;
    };
    // The analytic slab, TE, complex: transmit, and the part of the reflect port that entered
    // The thickness is the one the polyline crossed: ray_mesh_interact relaunches a few float ulp
    // of the coordinate magnitude past the face, along the outgoing direction, so the leg is short
    // by the normal part of that, 1e-6 m for the undeviated tracer and up to 1e-5 m for the
    // refracting one at 85 deg, which the series of a dense slab turns into 4e-3 dB.
    auto slab = [&](std::complex<double> es, double th, double f, bool series, double Lx, std::complex<double> &t, std::complex<double> &r_in)
    {
        std::complex<double> c1 = std::sqrt(1.0 - std::sin(th) * std::sin(th) / es), n = std::sqrt(es);
        std::complex<double> a = std::cos(th), b = n * c1, r10 = (b - a) / (b + a);
        std::complex<double> t01 = 2.0 * a / (a + b), t10 = 2.0 * b / (a + b);
        std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * (2.0 * QPI * f / C0) * n * c1 * Lx);
        std::complex<double> S = series ? 1.0 / (1.0 - r10 * r10 * phi * phi) : std::complex<double>(1.0, 0.0);
        t = t01 * t10 * phi * S, r_in = t01 * r10 * t10 * phi * phi * S;
    };

    // ---- Bare slab: the state, the chain against the analytic slab, the two tracers, the direction ----
    QD_DIAG("\n# diag: refracting tracer, bare slab. air | 100 mm | air through the intersector, series resolved.\n");
    QD_DIAG("# d_dB, ph: tracer 5 against the analytic slab at the polyline's exit point. r54: tracer 5 over 4,\n");
    QD_DIAG("# absolute transfer. tm_dB: tracer 2, HH against the TM slab. cur: current word after the entry,\n");
    QD_DIAG("# hex. out: path_dirN at the exit, traj: destN - origN there, deg, recorded\n");
    QD_DIAG("mtl,theta,cur4,cur5,rt5,d_dB,ph_deg,r54_dB,r54_deg,te2_dB,tm_dB,out_deg,traj_deg\n");
    for (int m : {5, 6, 1, 4})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            arma::uvec mi;
            arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
            Chain c4 = run(mtl, mesh, mi, {4}, thd, 0.0, f_hz, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, 0.0, f_hz, 0.0);
            Chain c2 = run(mtl, mesh, mi, {2}, thd, 0.0, f_hz, 0.0);
            REQUIRE((c4.ok && c5.ok && c2.ok));
            REQUIRE((c4.ev.size() == 2 && c5.ev.size() == 2 && c2.ev.size() == 2));
            std::complex<double> t, t4, r_in;
            slab(diag_eps(m, f_hz), th, f_hz, true, L - c5.ev[0].fbs(0, 0), t, r_in);
            slab(diag_eps(m, f_hz), th, f_hz, true, L - c4.ev[0].fbs(0, 0), t4, r_in);
            std::complex<double> q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t;
            // each over the slab it crossed, each at the point its own polyline leaves at
            std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t) /
                                       (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy)) / t4);
            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(diag_eps(m, f_hz), L - c2.ev[0].fbs(0, 0), th, f_hz, p_te, p_tm);
            QD_DIAG("%d,%.1f,%04x,%04x,%u,%+.6f,%+.4f,%+.6f,%+.4f,%+.6f,%+.6f,%.4f,%.4f\n", m, thd, (unsigned)(unsigned short)c4.ev[0].cur(0),
                    (unsigned)(unsigned short)c5.ev[0].cur(0), (unsigned)c5.ev[1].rt, dB(std::norm(q)), deg(q), dB(std::norm(r54)), deg(r54),
                    dB(std::norm(c2.vv) / p_te), dB(std::norm(c2.hh) / p_tm), std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI, c5.traj);
            INFO("bare slab, material " << m << ", theta_ext " << thd);
            // The state: the bent leg says so, with either refracting type, and the undeviated one never
            CHECK(c5.ev[0].cur(0) == (short)(m | BENT));
            CHECK(c2.ev[0].cur(0) == (short)(m | BENT));
            CHECK(c4.ev[0].cur(0) == (short)m);
            CHECK(c5.ev[1].cur(0) == (short)0); // zero in air
            CHECK(c5.ev[1].prev(0) == c4.ev[1].prev(0));
            CHECK(c5.ev[1].rt == 5); // the VBS bit keeps its meaning: the VBS is the FBS here
            // The chain
            CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
            CHECK(std::abs(deg(q)) < 0.05);
            CHECK(std::abs(dB(std::norm(r54))) < 1.0e-3);
            CHECK(std::abs(deg(r54)) < 0.05);
            CHECK(std::abs(dB(std::norm(c2.vv) / p_te)) < 1.0e-3);
            CHECK(std::abs(dB(std::norm(c2.hh) / p_tm)) < 1.0e-3);
            // The direction written at the exit is the one the ray arrived in
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }

    // ---- Sheets on and in the host, and two solids: tracer 5 against tracer 4 ----
    struct Stack
    {
        const char *name;
        std::vector<double> x;
        std::vector<int> mat;
        std::vector<bool> flip;
        double eps;
        bool phase;   // the phase is asserted
        int sheet_ev; // event after which the ray is behind a sheet inside the host, or -1
    };
    std::vector<Stack> stacks;
    for (int h : {5, 6})
    {
        stacks.push_back({"glued_entry ", {0.0, gap, L}, {SHEET, h, h}, {false, false, true}, 0.0, true, -1});
        stacks.push_back({"glued_exit  ", {0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, 0.0, true, -1});
        stacks.push_back({"lined       ", {0.0, gap, L, L + gap}, {SHEET, h, h, SHEET}, {false, false, true, true}, 0.0, true, -1});
        stacks.push_back({"sheet_inside", {0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, 0.0, true, 1});
    }
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {5, 1}, {2, 5}, {2, 6}, {1, 6}})
        stacks.push_back({"two_solids  ", {0.0, 0.05, 0.05 + gap, 0.15 + gap}, {P.first, P.first, P.second, P.second}, {false, true, false, true}, 1.5, true, -1});
    QD_DIAG("\n# diag: refracting tracer, layered walls. Tracer 5 over tracer 4, absolute transfer, scalar. Sheets:\n");
    QD_DIAG("# series resolved. two_solids: series declined. Each tracer at the point its polyline leaves at.\n");
    QD_DIAG("# words: the last in-wall state words of tracer 5, hex\n");
    QD_DIAG("stack,mats,theta,rt4,rt5,r54_dB,r54_deg,prev5,cur5,out_deg,traj_deg\n");
    for (const auto &S : stacks)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            arma::uvec mi;
            arma::mat mesh = build(S.x, S.mat, S.flip, mi);
            Chain c4 = run(mtl, mesh, mi, {4}, thd, S.eps, f_hz, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, S.eps, f_hz, 0.0);
            REQUIRE((c4.ok && c5.ok));
            REQUIRE(c4.ev.size() == c5.ev.size());
            size_t n = c5.ev.size();
            std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy))) /
                                       (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy)));
            std::string mats, r4, r5;
            for (int v : S.mat)
                mats += std::to_string(v);
            for (size_t k = 0; k < n; ++k)
                r4 += std::to_string((unsigned)c4.ev[k].rt) + "/", r5 += std::to_string((unsigned)c5.ev[k].rt) + "/";
            QD_DIAG("%s,%s,%.1f,%s,%s,%+.6f,%+.4f,%04x,%04x,%.4f,%.4f\n", S.name, mats.c_str(), thd, r4.c_str(), r5.c_str(), dB(std::norm(r54)),
                    deg(r54), (unsigned)(unsigned short)c5.ev[n - 2].prev(0), (unsigned)(unsigned short)c5.ev[n - 2].cur(0),
                    std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI, c5.traj);
            INFO(S.name << " " << mats << ", theta_ext " << thd);
            // No reference here, so the two polylines are compared as they are, each short by its
            // own relaunch offsets: 1.1e-3 dB on the lined lossless wall at 85 deg, below 6e-4 elsewhere
            CHECK(std::abs(dB(std::norm(r54))) < 2.0e-3);
            if (S.phase)
                CHECK(std::abs(deg(r54)) < 0.05);
            for (size_t k = 0; k < n; ++k)
            {
                // The words of the two tracers differ in the one bit and nowhere else, and the bit
                // is set on every in-wall leg: carried through a sheet inside the host, written
                // again by a transition that refracted
                bool in_wall = mat_of(c5.ev[k].cur(0)) != 0;
                CHECK(c5.ev[k].prev(0) == c4.ev[k].prev(0));
                CHECK(c5.ev[k].buf(0) == c4.ev[k].buf(0));
                CHECK(c5.ev[k].cur(0) == (short)(c4.ev[k].cur(0) | (in_wall ? BENT : (short)0)));
                CHECK((c4.ev[k].cur(0) & BENT) == 0);
            }
            if (S.sheet_ev >= 0)
                CHECK((c5.ev[(size_t)S.sheet_ev].cur(0) & BENT) != 0);
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }

    // ---- Back reflection: the reflect pass does not name the tracer, the word does ----
    QD_DIAG("\n# diag: refracting tracer, back reflection. Enter (5), reflect at the far face (3), leave by the\n");
    QD_DIAG("# front face (5), against t01 r10 t10 phi^2 [S] at the polyline's exit point. eps 0: the reflection\n");
    QD_DIAG("# collapses the series and the ray leaves resolved. cur: the word after the reflection, hex\n");
    QD_DIAG("mtl,eps,theta,rt5,cur5,d_dB,ph_deg\n");
    for (int m : {5, 6})
        for (double eps : {1.5, 0.0})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                arma::uvec mi;
                arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                Chain c5 = run(mtl, mesh, mi, {5, 3, 5}, thd, eps, f_hz, 0.0);
                REQUIRE((c5.ok && c5.ev.size() == 3));
                std::complex<double> t, r_in;
                slab(diag_eps(m, f_hz), th, f_hz, eps == 0.0, L - c5.ev[0].fbs(0, 0), t, r_in);
                std::complex<double> q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / r_in;
                std::string r5;
                for (const auto &E : c5.ev)
                    r5 += std::to_string((unsigned)E.rt) + "/";
                QD_DIAG("%d,%.1f,%.1f,%s,%04x,%+.6f,%+.4f\n", m, eps, thd, r5.c_str(), (unsigned)(unsigned short)c5.ev[1].cur(0), dB(std::norm(q)), deg(q));
                INFO("back reflection, material " << m << ", eps " << eps << ", theta_ext " << thd);
                CHECK((c5.ev[1].cur(0) & BENT) != 0); // a reflection keeps the leg's bit
                CHECK(mat_of(c5.ev[1].cur(0)) == m);
                CHECK(flag_of(c5.ev[1].cur(0)) == (eps == 0.0));
                CHECK(c5.ev[2].cur(0) == (short)0);
                CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
                CHECK(std::abs(deg(q)) < 0.05);
                CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9); // leaves along the mirror of how it arrived
                CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
            }

    // ---- A second frequency of a run (printed) ----
    QD_DIAG("\n# diag: refracting tracer, a second frequency. Geometry and state words at 1 GHz, coefficient and\n");
    QD_DIAG("# direction at f1, as ray_progress runs a band. eps 1.50 - 0.40j at 1 GHz, 100 mm, against the slab at\n");
    QD_DIAG("# f1 at the polyline's exit point, power and phase\n");
    QD_DIAG("f1_GHz,theta,d_dB,ph_deg,out_deg\n");
    for (double f1 : {0.5e9, 2.0e9})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0;
            arma::uvec mi;
            arma::mat mesh = build({0.0, L}, {5, 5}, {false, true}, mi);
            Chain c5 = run(mtl, mesh, mi, {5}, thd, 0.0, f1, f_hz);
            REQUIRE((c5.ok && c5.ev.size() == 2));
            std::complex<double> t, r_in;
            slab(diag_eps(5, f1), th, f1, true, L - c5.ev[0].fbs(0, 0), t, r_in);
            std::complex<double> q = c5.abs_vv(k1) * std::exp(std::complex<double>(0.0, k1 * std::sin(th) * c5.dy)) / t;
            QD_DIAG("%.1f,%.1f,%+.6f,%+.4f,%.4f\n", f1 * 1e-9, thd, dB(std::norm(q)), deg(q), std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI);
            INFO("second frequency " << f1 << ", theta_ext " << thd);
            CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
            CHECK(std::abs(deg(q)) < 0.05);
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }
    // ---- The trajectory: the segment out of a lossy medium is the direction written ----
    // ray_state_update is handed origN and destN and puts them on one line through the point the ray leaves at, along
    // the direction it writes, wherever that direction is the one of a wave out of a lossy medium whose cosine was
    // recovered. Everything else of the event is what it was, so the chain is the one of the tables above, and the leg
    // in a second solid runs where the transition says the wave runs.
    relaunch = true;
    auto lossy = [&](int m)
    { return std::imag(diag_eps(m, f_hz)) != 0.0; };
    auto seg_err = [](const Ev &E) -> double // largest component of the unit segment against path_dirN
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        double e = 0.0;
        for (int c = 0; c < 3; ++c)
            e = std::max(e, std::abs(d(0, c) - E.path_dirN(0, c)));
        return e;
    };
    QD_DIAG("\n# diag: refracting tracer, the trajectory. As the tables above, with origN and destN handed to\n");
    QD_DIAG("# ray_state_update. traj: destN - origN behind the wall, deg. d_out: that segment against path_dirN,\n");
    QD_DIAG("# largest component. d_in: the same for the worst event that leaves a lossy medium, the transition\n");
    QD_DIAG("# of two solids included. d_dB, ph: the bare slab against the analytic one. r54: tracer 5 over 4.\n");
    QD_DIAG("# rt: the resolved types, with the geometry and without\n");
    QD_DIAG("stack,mats,theta,tracer,rt,rt_without,traj_deg,d_out,d_in,d_dB,ph_deg,r54_dB,r54_deg\n");
    {
        std::vector<Stack> all;
        for (int m : {5, 6, 1, 4})
            all.push_back({"bare        ", {0.0, L}, {m, m}, {false, true}, 0.0, true, -1});
        for (const auto &S : stacks)
            all.push_back(S);
        for (const auto &S : all)
            for (double thd : angles)
                for (int itype : {5, 2})
                {
                    bool bare = S.x.size() == 2;
                    if (itype == 2 && !bare)
                        continue;
                    double th = thd * QPI / 180.0;
                    arma::uvec mi;
                    arma::mat mesh = build(S.x, S.mat, S.flip, mi);
                    Chain c5 = run(mtl, mesh, mi, {itype}, thd, S.eps, f_hz, 0.0);
                    relaunch = false;
                    Chain c4 = run(mtl, mesh, mi, {4}, thd, S.eps, f_hz, 0.0), c0 = run(mtl, mesh, mi, {itype}, thd, S.eps, f_hz, 0.0);
                    relaunch = true;
                    REQUIRE((c4.ok && c5.ok && c0.ok));
                    REQUIRE((c4.ev.size() == c5.ev.size() && c0.ev.size() == c5.ev.size()));
                    size_t n = c5.ev.size();
                    double d_in = 0.0;
                    for (size_t k = 1; k < n; ++k)
                    {
                        int left = mat_of(c5.ev[k - 1].cur(0));
                        if (lossy(left) && mat_of(c5.ev[k].cur(0)) != left)
                            d_in = std::max(d_in, seg_err(c5.ev[k]));
                    }
                    std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy))) /
                                               (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy))),
                                         q(1.0, 0.0);
                    if (bare)
                    {
                        std::complex<double> t, t4, r_in;
                        slab(diag_eps(S.mat[0], f_hz), th, f_hz, true, L - c5.ev[0].fbs(0, 0), t, r_in);
                        slab(diag_eps(S.mat[0], f_hz), th, f_hz, true, L - c4.ev[0].fbs(0, 0), t4, r_in);
                        q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t;
                        r54 *= t4 / t; // each over the slab it crossed
                    }
                    std::string mats, r5, r0;
                    for (int v : S.mat)
                        mats += std::to_string(v);
                    for (size_t k = 0; k < n; ++k)
                        r5 += std::to_string((unsigned)c5.ev[k].rt) + "/", r0 += std::to_string((unsigned)c0.ev[k].rt) + "/";
                    QD_DIAG("%s,%s,%.1f,%d,%s,%s,%.4f,%.3e,%.3e,%+.6f,%+.4f,%+.6f,%+.4f\n", S.name, mats.c_str(), thd, itype, r5.c_str(), r0.c_str(), c5.traj,
                            seg_err(c5.ev[n - 1]), d_in, dB(std::norm(q)), deg(q), dB(std::norm(r54)), deg(r54));
                    INFO("trajectory, " << S.name << " " << mats << ", theta_ext " << thd << ", type " << itype);
                    // The segment behind the wall is the direction written, and that is the one the ray arrived in
                    CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
                    CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
                    CHECK(d_in < 1.0e-9);
                    if (lossy(mat_of(c5.ev[n - 2].cur(0))))
                        CHECK(seg_err(c5.ev[n - 1]) < 1.0e-9);
                    // Nothing else of an event moves: the words and the type codes. One bit follows the
                    // geometry by its own definition: the VBS bit says that the direction the wave has
                    // differs from the segment, which behind a lossy first solid it does not
                    for (size_t k = 0; k < n; ++k)
                    {
                        CHECK(c5.ev[k].prev(0) == c0.ev[k].prev(0));
                        CHECK(c5.ev[k].cur(0) == c0.ev[k].cur(0));
                        CHECK(c5.ev[k].buf(0) == c0.ev[k].buf(0));
                        CHECK((c5.ev[k].rt & 0xFD) == (c0.ev[k].rt & 0xFD));
                        CHECK((c5.ev[k].rt & 0x02) <= (c0.ev[k].rt & 0x02));
                        if (k > 0 && lossy(mat_of(c5.ev[k - 1].prev(0))) && mat_of(c5.ev[k - 1].prev(0)) != SHEET && mat_of(c5.ev[k - 1].cur(0)) != 0)
                            CHECK((c5.ev[k].rt & 0x02) == 0); // the leg behind a lossy solid: the VBS is the FBS
                    }
                    // The chain
                    if (itype == 5)
                    {
                        CHECK(std::abs(dB(std::norm(r54))) < 2.0e-3);
                        if (S.phase)
                            CHECK(std::abs(deg(r54)) < 0.05);
                    }
                    if (bare)
                    {
                        CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
                        CHECK(std::abs(deg(q)) < 0.05);
                        // One wall: the events are the ones of the run without the geometry, bit for bit
                        for (size_t k = 0; k < n; ++k)
                            CHECK(cpx(c5.ev[k].xpr, 0, 0) == cpx(c0.ev[k].xpr, 0, 0));
                    }
                }
    }
    relaunch = false;
}

TEST_CASE("Phase reference - a traversal carries the phase at the point the polyline leaves at", "[wave]")
{
    ub::OneThread one_thread;
    // The tracer commits a polyline, and the path length, the delay and the free-space phase of a
    // path are taken from it. A coefficient is therefore owed the field at the point where the
    // polyline leaves a body. For a plane wave that field is the stack's transfer on the thickness
    // axis times exp(-j k0 sin(theta) dy), with dy the lateral walk of the polyline, whatever the
    // polyline is: one path, and per frequency the value the field has where that path leaves. The
    // geometry is never moved for it. Everything runs through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update.
    //
    // Two points are compared on every row:
    //   ray:  the point where the refracted ray of that frequency leaves, lateral walk s
    //   poly: the point where the polyline leaves, lateral walk dy
    // ph_x = arg( coefficient * exp(-j k0 L_in) * exp(+j k0 sin(theta) lateral_x) / reference ).
    // A band is run as ray_progress runs it: geometry, state words and the geometric ledger from
    // the base-frequency event, with origN and destN handed in on the base-frequency transmit call,
    // and the coefficient, the direction and the refracted ledger from the event at that frequency.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    auto dB = [](double g)
    { return 10.0 * std::log10(g); };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };
    const std::complex<double> J(0.0, 1.0);

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs, sbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, double f, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, const Ev *in_state, double eps, bool relaunch) -> Ev
    {
        if (!in_state)
            in_state = in;
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in_state->prev : nullptr, in ? &in_state->cur : nullptr, in ? &in_state->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in_state->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        E.sbs = sbsN;
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev, evb;    // the events at f, and at the base frequency (geometry, words)
        // per event, for the leg it closes: polyline length and lateral walk between the two hit
        // points, the length the ledger was given (from where the leg started, qd_leg_start, to the hit point), its normal part,
        // and the lateral offset of the relaunch point from the hit point it left
        std::vector<double> len, lat, led, ledx, ledy, offy;
        bool ok = true;
        std::complex<double> abs_vv(double k) const { return vv * std::exp(std::complex<double>(0.0, -k * Lin)); }
    };
    auto run = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh, const arma::uvec &mtl_ind,
                   std::vector<int> seq, double thd, double eps, double f, double f_base) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P, Pb;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, xb = 0.0, yb = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            bool tx = itype != 0 && itype != 3;
            Ev B = traced(pal, mesh, mtl_ind, itype, f_base != 0.0 ? f_base : f, orig, dest, have ? &Pb : nullptr, nullptr, eps, tx);
            if (B.nh == 0)
                break;
            Ev E = f_base != 0.0 ? traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, have ? &Pb : nullptr, eps, false) : B;
            if (E.rt == 0 || B.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = B.fbs(0, 0), fy = B.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            double seg = std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            double ox = fx - xb, oy = fy - yb;
            C.len.push_back(inside ? seg : 0.0), C.lat.push_back(inside ? std::abs(fy - yp) : 0.0);
            C.led.push_back(inside ? std::sqrt(ox * ox + oy * oy) : 0.0), C.ledx.push_back(inside ? std::abs(ox) : 0.0), C.ledy.push_back(inside ? std::abs(oy) : 0.0);
            C.offy.push_back(std::abs(B.origN(0, 1) - fy));
            if (inside)
                C.Lin += seg;
            xp = fx, yp = fy, C.dy = fy - y0;
            xb = qd_leg_start(B, 0), yb = qd_leg_start(B, 1); // where the next leg's ledger starts
            inside = mat_of(B.cur(0)) != 0;
            arma::mat d = B.destN - B.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = B.origN, dest = B.origN + 2.0 * d;
            P = E, Pb = B, have = true;
            C.ev.push_back(E), C.evb.push_back(B);
        }
        return C;
    };
    auto tan_of = [](const Ev &E) -> double // tangent of the written direction against the x axis
    { return std::abs(E.path_dirN(0, 1)) / std::abs(E.path_dirN(0, 0)); };

    // One face pair air | medium, scalar (TE), in admittances: z c, with z = sqrt(eps / mu)
    struct Sl
    {
        std::complex<double> t01, t10, r10, n, c1, nc; // nc = n c1, the normal wavenumber over k0
    };
    auto slabp = [&](std::complex<double> es, std::complex<double> mus, double th) -> Sl
    {
        Sl s;
        s.c1 = std::sqrt(1.0 - std::sin(th) * std::sin(th) / (es * mus)), s.n = std::sqrt(es * mus), s.nc = s.n * s.c1;
        std::complex<double> a = std::cos(th), b = std::sqrt(es / mus) * s.c1;
        s.r10 = (b - a) / (b + a), s.t01 = 2.0 * a / (a + b), s.t10 = 2.0 * b / (a + b);
        return s;
    };
    auto phi_of = [&](const Sl &s, double f, double Lx)
    { return std::exp(-J * (2.0 * QPI * f / C0) * s.nc * Lx); };

    // ---- A. A transmitted chain, both tracers, the base frequency and two more of a band ----
    QD_DIAG("\n# diag: phase reference, transmitted. air | 100 mm | air through the intersector, series resolved, scalar,\n");
    QD_DIAG("# against the analytic slab. Base frequency 1 GHz; 0.5 and 2 GHz are run as frequencies of a band.\n");
    QD_DIAG("# ph_ray: at the point where the refracted ray of that frequency leaves. ph_poly: at the polyline's exit\n");
    QD_DIAG("# point. s, dy: the lateral walk of the ray and of the polyline, mm\n");
    QD_DIAG("tracer,mtl,f_GHz,theta,d_dB,ph_ray,ph_poly,s_mm,dy_mm\n");
    for (int tr : {4, 5})
        for (int m : {5, 6, 1})
            for (double f1 : {1.0e9, 0.5e9, 2.0e9})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                    Chain c = run(mtl, mesh, mi, {tr}, thd, 0.0, f1, f1 == FRQ ? 0.0 : FRQ);
                    REQUIRE((c.ok && c.ev.size() == 2));
                    double Lx = L - c.evb[0].fbs(0, 0);
                    Sl s = slabp(diag_eps(m, f1), 1.0, th);
                    std::complex<double> phi = phi_of(s, f1, Lx), t = s.t01 * s.t10 * phi / (1.0 - s.r10 * s.r10 * phi * phi);
                    double s_ray = Lx * tan_of(c.ev[0]) + c.offy[0];
                    std::complex<double> a = c.abs_vv(k1) / t;
                    std::complex<double> q_ray = a * std::exp(J * k1 * std::sin(th) * s_ray), q_poly = a * std::exp(J * k1 * std::sin(th) * c.dy);
                    QD_DIAG("%d,%d,%.1f,%.1f,%+.6f,%+.4f,%+.4f,%.4f,%.4f\n", tr, m, f1 * 1e-9, thd, dB(std::norm(a)), deg(q_ray), deg(q_poly),
                            s_ray * 1e3, c.dy * 1e3);
                    INFO("transmitted, tracer " << tr << ", material " << m << ", f " << f1 << ", theta_ext " << thd);
                    CHECK(std::abs(dB(std::norm(a))) < 1.0e-3); // every power is where it is
                    CHECK(std::abs(deg(q_poly)) < 0.05);
                }

    // ---- B. A back reflection, the two legs apart ----
    QD_DIAG("\n# diag: phase reference, back reflection. Enter, reflect at the far face (type 3), leave by the front\n");
    QD_DIAG("# face, against t01 r10 t10 phi^2 [S]. eps 1.5: series declined; eps 0: the reflection collapses it\n");
    QD_DIAG("# and the ray leaves resolved. Per leg: the event that closes it against r10 phi [S] (leg 1, closed\n");
    QD_DIAG("# on the reflect pass) and t10 phi (leg 2, closed at the exit), phase; leg1_dB: leg 1 in power\n");
    QD_DIAG("tracer,mtl,f_GHz,eps,theta,rt,d_dB,ph_ray,ph_poly,leg1_ray,leg1_poly,leg2_ray,leg2_poly,leg1_dB\n");
    for (int tr : {4, 5})
        for (int m : {5, 6})
            for (double f1 : {1.0e9, 2.0e9})
                for (double eps : {1.5, 0.0})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0, st = std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                        Chain c = run(mtl, mesh, mi, {tr, 3, tr}, thd, eps, f1, f1 == FRQ ? 0.0 : FRQ);
                        std::string rt;
                        for (const auto &E : c.ev)
                            rt += std::to_string((unsigned)E.rt) + "/";
                        if (!c.ok || c.ev.size() != 3 || std::norm(c.vv) == 0.0)
                        {
                            QD_DIAG("%d,%d,%.1f,%.1f,%.1f,%s,dead\n", tr, m, f1 * 1e-9, eps, thd, rt.c_str());
                            INFO("back reflection, tracer " << tr << ", material " << m << ", f " << f1 << ", eps " << eps << ", theta_ext " << thd);
                            CHECK(false); // no ray dies on this path: the wave crosses the front face
                            continue;
                        }
                        double Lx1 = L - c.evb[0].fbs(0, 0), Lx2 = c.evb[1].fbs(0, 0), tt = tan_of(c.ev[0]);
                        Sl s = slabp(diag_eps(m, f1), 1.0, th);
                        std::complex<double> p1 = phi_of(s, f1, Lx1), p2 = phi_of(s, f1, Lx2);
                        std::complex<double> S = (eps == 0.0) ? 1.0 / (1.0 - s.r10 * s.r10 * p1 * p1) : std::complex<double>(1.0, 0.0);
                        std::complex<double> a = c.abs_vv(k1) / (s.t01 * s.r10 * s.t10 * p1 * p2 * S);
                        double s_ray = (Lx1 + Lx2) * tt + c.offy[0] + c.offy[1];
                        std::complex<double> a1 = cpx(c.ev[1].xpr, 0, 0) * std::exp(-J * k1 * c.len[1]) / (s.r10 * p1 * S);
                        std::complex<double> a2 = cpx(c.ev[2].xpr, 0, 0) * std::exp(-J * k1 * c.len[2]) / (s.t10 * p2);
                        double s1 = Lx1 * tt + c.offy[0], s2 = Lx2 * tt + c.offy[1];
                        QD_DIAG("%d,%d,%.1f,%.1f,%.1f,%s,%+.6f,%+.4f,%+.4f", tr, m, f1 * 1e-9, eps, thd, rt.c_str(), dB(std::norm(a)),
                                deg(a * std::exp(J * k1 * st * s_ray)), deg(a * std::exp(J * k1 * st * c.dy)));
                        QD_DIAG(",%+.4f,%+.4f,%+.4f,%+.4f,%+.6f\n", deg(a1 * std::exp(J * k1 * st * s1)), deg(a1 * std::exp(J * k1 * st * c.lat[1])),
                                deg(a2 * std::exp(J * k1 * st * s2)), deg(a2 * std::exp(J * k1 * st * c.lat[2])), dB(std::norm(a1)));
                        INFO("back reflection, tracer " << tr << ", material " << m << ", f " << f1 << ", eps " << eps << ", theta_ext " << thd);
                        CHECK(std::isfinite(std::norm(a)));
                        // Leg 1, closed on the reflect pass: the exact form, with the series where it collapsed
                        CHECK(std::abs(dB(std::norm(a1))) < 1.0e-3);
                        CHECK(std::abs(deg(a1 * std::exp(J * k1 * st * c.lat[1]))) < 0.05);
                        // The whole path, a resolved ray of the undeviated tracer included
                        CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                        CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05);
                        CHECK(std::abs(deg(a2 * std::exp(J * k1 * st * c.lat[2]))) < 0.05);
                    }

    // ---- C. A sheet inside a host: which point the composite and its displaced-host term sit on ----
    QD_DIAG("\n# diag: phase reference, a sheet inside a host. air | host 50 | sheet 6 mm eps 5.24 - 0.63j | host 50 | air,\n");
    QD_DIAG("# host series declined, scalar, 1 GHz, against the single pass through host (100 - 6 mm) and the sheet's\n");
    QD_DIAG("# Airy pair. Four points: poly, the polyline's; host, where a ray refracted in the host alone leaves,\n");
    QD_DIAG("# the sheet taken as host; true, where the ray leaves that is also refracted in the sheet; less, the\n");
    QD_DIAG("# host ray over 100 - 6 mm, which is where a transverse term in the displaced-host division would put it\n");
    QD_DIAG("tracer,host,theta,d_dB,ph_poly,ph_host,ph_true,ph_less\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th), thk = 0.006;
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, mi);
                Chain c = run(mtl, mesh, mi, {tr}, thd, 1.5, FRQ, 0.0);
                REQUIRE((c.ok && c.ev.size() == 3));
                double Dw = (0.05 - c.evb[0].fbs(0, 0)) + (L - c.evb[1].fbs(0, 0)), tth = tan_of(c.ev[0]);
                Sl sh = slabp(diag_eps(h, FRQ), 1.0, th), ss = slabp(diag_eps(SHEET, FRQ), 1.0, th);
                std::complex<double> bh = sh.nc, bs = ss.nc, r_sh = (bs - bh) / (bs + bh);
                std::complex<double> ps = phi_of(ss, FRQ, thk), Ts = (2.0 * bh / (bh + bs)) * (2.0 * bs / (bh + bs)) * ps / (1.0 - r_sh * r_sh * ps * ps);
                std::complex<double> a = c.abs_vv(k1) / (sh.t01 * phi_of(sh, FRQ, Dw - thk) * Ts * sh.t10);
                double tts = st / (std::sqrt(std::abs(diag_eps(SHEET, FRQ))) * std::real(ss.c1)); // qd_refract's map from air
                double off = c.offy[0] + c.offy[1];
                QD_DIAG("%d,%d,%.1f,%+.6f,%+.4f,%+.4f,%+.4f,%+.4f\n", tr, h, thd, dB(std::norm(a)), deg(a * std::exp(J * k1 * st * c.dy)),
                        deg(a * std::exp(J * k1 * st * (Dw * tth + off))), deg(a * std::exp(J * k1 * st * ((Dw - thk) * tth + thk * tts + off))),
                        deg(a * std::exp(J * k1 * st * ((Dw - thk) * tth + off))));
                INFO("sheet inside a host, tracer " << tr << ", host " << h << ", theta_ext " << thd);
                CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05); // the composite and the host legs on one point
            }

    // ---- D. An acoustic absorber as a traced solid, over a band ----
    // Mineral wool at about 12 kPa s/m^2, the two-column-pair fit of the material model, 50 mm; and a
    // rigid backing, eps 2.401e-9. Frequencies are acoustic, scaled by c0 / 343 m/s. The index is
    // about 1.5 - 0.75j at 1 kHz, so the wave is refracted into the layer at every angle.
    {
        const double W = 0.05, fscale = C0 / 343.0, gapw = 1.0e-5;
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {1.16, 2.401e-9}, pal["b"] = {-0.04, 0.0}, pal["c"] = {0.0038, 0.0}, pal["d"] = {1.46, 0.0};
        pal["e"] = {1.66, 1.0}, pal["f"] = {-0.19, 0.0}, pal["g"] = {0.091, 0.0}, pal["h"] = {0.13, 0.0};
        pal["fRef"] = {0.874, 0.874};
        auto wool = [&](double f_hz, std::complex<double> &es, std::complex<double> &mus)
        {
            double fG = f_hz * 1e-9, fr = fG / 0.874;
            es = std::complex<double>(1.16 * std::pow(fr, -0.04), -17.98 * 0.0038 * std::pow(fr, 1.46) / fG);
            mus = std::complex<double>(1.66 * std::pow(fr, -0.19), -17.98 * 0.091 * std::pow(fr, 0.13) / fG);
        };
        const std::vector<double> band = {250.0, 500.0, 1000.0, 2000.0, 4000.0}, ang = {20.0, 40.0, 60.0, 75.0};
        const double f_base = 1000.0 * fscale;
        QD_DIAG("\n# diag: phase reference, an acoustic absorber as a traced solid. Mineral wool 50 mm, scalar, base\n");
        QD_DIAG("# frequency 1 kHz, the others run as frequencies of a band. through: air | wool | air, series\n");
        QD_DIAG("# resolved. backed: air | wool | rigid wall, enter, reflect at the wall, leave, series declined.\n");
        QD_DIAG("# n: the index of the wool at that frequency. ph_ray, ph_poly, s, dy as above. The wall is taken at\n");
        QD_DIAG("# the Fresnel coefficient of wool | eps 2.401e-9, at its own magnitude, decaying root; r12: that Fresnel\n");
        QD_DIAG("# coefficient itself, power in dB and phase, recorded\n");
        QD_DIAG("chain,tracer,f_Hz,n_re,n_im,theta,rt,d_dB,ph_ray,ph_poly,s_mm,dy_mm,r12_dB,r12_deg\n");
        for (int back = 0; back < 2; ++back)
            for (int tr : {4, 5})
                for (double fa : band)
                    for (double thd : ang)
                    {
                        double f1 = fa * fscale, th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0, st = std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = back ? build({0.0, W, W + gapw, W + gapw + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, mi)
                                              : build({0.0, W}, {1, 1}, {false, true}, mi);
                        Chain c = run(pal, mesh, mi, back ? std::vector<int>{tr, 3, tr} : std::vector<int>{tr}, thd, back ? 1.5 : 0.0, f1,
                                      fa == 1000.0 ? 0.0 : f_base);
                        std::string rt;
                        for (const auto &E : c.ev)
                            rt += std::to_string((unsigned)E.rt) + "/";
                        REQUIRE((c.ok && c.ev.size() == (back ? 3u : 2u)));
                        std::complex<double> es, mus;
                        wool(f1, es, mus);
                        Sl s = slabp(es, mus, th);
                        double Lx1 = W - c.evb[0].fbs(0, 0), Lx2 = back ? c.evb[1].fbs(0, 0) : 0.0, tt = tan_of(c.ev[0]);
                        std::complex<double> p1 = phi_of(s, f1, Lx1), ref;
                        double r12_dB = 0.0, r12_deg = 0.0;
                        if (back)
                        {
                            // The wall from inside the wool: the decaying root in the rigid body
                            std::complex<double> e2(2.401e-9, 0.0), c2 = -J * std::sqrt(st * st / e2 - 1.0);
                            std::complex<double> b1 = std::sqrt(es / mus) * s.c1, b2 = std::sqrt(e2) * c2, r12 = (b1 - b2) / (b1 + b2);
                            // A closed face reflects at the Fresnel coefficient, above one where that is above one
                            r12_dB = dB(std::norm(r12)), r12_deg = deg(r12);
                            ref = s.t01 * r12 * s.t10 * p1 * phi_of(s, f1, Lx2);
                        }
                        else
                            ref = s.t01 * s.t10 * p1 / (1.0 - s.r10 * s.r10 * p1 * p1);
                        std::complex<double> a = c.abs_vv(k1) / ref;
                        double s_ray = (Lx1 + Lx2) * tt + c.offy[0] + (back ? c.offy[1] : 0.0);
                        QD_DIAG("%s,%d,%.0f,%.4f,%.4f,%.1f,%s,%+.6f,%+.4f,%+.4f,%.4f,%.4f,%+.4f,%+.3f\n", back ? "backed " : "through", tr, fa, std::real(s.n),
                                std::imag(s.n), thd, rt.c_str(), dB(std::norm(a)), deg(a * std::exp(J * k1 * st * s_ray)),
                                deg(a * std::exp(J * k1 * st * c.dy)), s_ray * 1e3, c.dy * 1e3, r12_dB, r12_deg);
                        INFO("absorber, " << (back ? "backed" : "through") << ", tracer " << tr << ", f " << fa << ", theta_ext " << thd);
                        CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                        CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05);
                    }
    }

    // ---- E. Two solids across a bare face: where the phase between the two tracers comes from ----
    // The absolute transfer puts the free-space reference back over the polyline between hit points.
    // A coefficient carries that reference over the ledger only, which starts where the leg started -
    // the hit point, or the far face of a co-located pair - and ends at the next hit point. What lies
    // across a co-located pair, the gap between its two faces, is in the polyline and in no ledger,
    // and is charged as free space along the polyline. At the polyline's
    // point what is left of such a piece of length u and lateral walk v is u - sin(theta) v, summed
    // to U. And a leg is as thick as its ledger is long along the normal, d. Both differ between the
    // two tracers, and the phase that follows from the geometry alone is
    // -k0 (U5 - U4) - k0 sum Re(n c) (d5 - d4).
    QD_DIAG("\n# diag: phase reference, two solids across a bare face. air | A 50 | B 100 | air, series declined, scalar,\n");
    QD_DIAG("# 1 GHz. r54: tracer 5 over tracer 4, each at its polyline's exit point. U: what no ledger holds, um. pU,\n");
    QD_DIAG("# pd: the phase that follows from U and from the thickness the ledgers hold, deg. res: r54_deg - pU - pd\n");
    QD_DIAG("pair,gap_um,theta,r54_dB,r54_deg,U4_um,U5_um,pU_deg,pd_deg,res_deg\n");
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {2, 6}})
        for (double g : {1.0e-5, 1.0e-6, 5.0e-5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, 0.05 + g, 0.15 + g}, {P.first, P.first, P.second, P.second}, {false, true, false, true}, mi);
                Chain c4 = run(mtl, mesh, mi, {4}, thd, 1.5, FRQ, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, 1.5, FRQ, 0.0);
                REQUIRE((c4.ok && c5.ok && c4.ev.size() == 3 && c5.ev.size() == 3));
                std::complex<double> r54 = (c5.abs_vv(k1) * std::exp(J * k1 * st * c5.dy)) / (c4.abs_vv(k1) * std::exp(J * k1 * st * c4.dy));
                double U4 = 0.0, U5 = 0.0, pd = 0.0;
                for (size_t k = 1; k < 3; ++k)
                {
                    U4 += c4.len[k] - c4.led[k] - st * (c4.lat[k] - c4.ledy[k]), U5 += c5.len[k] - c5.led[k] - st * (c5.lat[k] - c5.ledy[k]);
                    Sl s = slabp(diag_eps(k == 1 ? P.first : P.second, FRQ), 1.0, th);
                    pd -= k1 * std::real(s.nc) * (c5.ledx[k] - c4.ledx[k]);
                }
                double pU = -k1 * (U5 - U4);
                QD_DIAG("%d|%d,%.0f,%.1f,%+.6f,%+.4f,%.3f,%.3f,%+.4f,%+.4f,%+.4f\n", P.first, P.second, g * 1e6, thd, dB(std::norm(r54)), deg(r54),
                        U4 * 1e6, U5 * 1e6, pU * 180.0 / QPI, pd * 180.0 / QPI, deg(r54) - (pU + pd) * 180.0 / QPI);
                INFO("two solids " << P.first << "|" << P.second << ", gap " << g << ", theta_ext " << thd);
                CHECK(std::abs(dB(std::norm(r54))) < 2.0e-2);                // 2e-3 at the 10 um of the refracting-tracer case
                CHECK(std::abs(deg(r54) - (pU + pd) * 180.0 / QPI) < 0.005); // nothing but the geometry
                CHECK(std::abs(deg(r54)) < 0.05);
            }

    // ---- G. Printed: the phase a rigid body reflects with, seen from air ----
    // The backed rows above take the wall at the Fresnel phase of the permittivity it is mapped to. Seen from air that
    // phase is measured here, for the one-parameter mapping (eps = (Z_air / Z)^2, mu = 1) and for a body with the same
    // impedance and an index of 0.1 (eps = z n, mu = n / z).
    {
        const double zr = std::sqrt(2.401e-9), nr = 0.1;
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.401e-9, zr * nr};
        pal["e"] = {1.0, nr / zr};
        QD_DIAG("\n# diag: outside the item, recorded. Reflection off a rigid half-space from air, type 3, 1 GHz,\n");
        QD_DIAG("# through the intersector. one: eps 2.401e-9, mu 1. two: the same impedance with an index of 0.1\n");
        QD_DIAG("mapping,theta,refl_dB,refl_deg\n");
        for (int m : {1, 2})
            for (double thd : {0.0, 10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.5}, {m, m}, {false, true}, mi), orig, dest;
                diag_ray_at(thd, orig, dest);
                Ev E = traced(pal, mesh, mi, 3, FRQ, orig, dest, nullptr, nullptr, 1.5, false);
                std::complex<double> r = cpx(E.xpr, 0, 0);
                QD_DIAG("%s,%.1f,%+.6f,%+.4f\n", m == 1 ? "one" : "two", thd, dB(std::norm(r)), deg(r));
                INFO("rigid half-space, mapping " << m << ", theta " << thd);
                CHECK(std::abs(std::norm(r) - 1.0) < 1.0e-3);
            }
    }
}

TEST_CASE("Leak direction - the direction a closed face writes for a wave launched inside a solid", "[wave]")
{
    // The part of "Leak direction - a wave held in a mass-law body keeps its direction and a closed
    // face hands on the wave's" that evaluates no material with the mass column (m > 0); the
    // description is there.
    const double f_hz = FRQ, gap = 1.0e-5, L = 0.10;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    typedef std::unordered_map<std::string, std::vector<double>> Pal;

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, const arma::mat &orig,
                      const arma::mat &dest, const Ev *in, double eps) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, &E.origN, &E.destN);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    using ub::build;
    using ub::face;
    auto unit = [](arma::mat d) -> arma::mat
    {
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        return d;
    };
    auto ang = [](const arma::mat &d) -> double // against the x axis, which every face normal is on
    { return std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI; };

    struct Row
    {
        std::vector<double> seg, dir, g, out;
        std::vector<unsigned> code, rt, cur;
        double tot = 0.0, dy = 0.0;
        bool ok = true;
    };

    // ---- B. The direction a closed face writes ----
    {
        // diag_palette with a tenth material, RARE: eps 1, lossless, a solid. The host eps
        // 1.50 - 0.40j is entered from the lossless eps 5.24 solid the ray is launched in, with a
        // transverse invariant p; from p = 1 the wave is closed towards air and towards RARE.
        const int DENSE = 2, SHEET = 3, HOST = 5, RARE = 10;
        Pal base = diag_palette<double>();
        for (auto &kv : base)
            kv.second.push_back(kv.first == "a" ? 1.0 : (kv.first == "fRef" ? 1.0 : 0.0));
        Pal with_tf = base;
        with_tf["tf"] = std::vector<double>(10, 0.0);
        with_tf["tf"][(size_t)SHEET - 1] = 0.5, with_tf["tf"][(size_t)RARE - 1] = 0.5;
        const double e_d = 5.24;

        QD_DIAG("\n# diag: leak direction, the direction a closed face writes. Launched in a lossless eps 5.24 solid, into host eps\n");
        QD_DIAG("# 1.50 - 0.40j 100 mm, invariant p; then bare: host | air. glued: host | coating 6 mm | air. trans: host | RARE\n");
        QD_DIAG("# eps 1, a solid. Through the intersector, scalar, series declined. tf: 0.5 on the coating and on RARE.\n");
        QD_DIAG("# in: the direction the wave has in the host; seg: the segment; out: the direction written; traj: the\n");
        QD_DIAG("# relaunched segment, all deg against the normal. tr: the total-reflection bit of the resolved type\n");
        QD_DIAG("p,tracer,face,tf,code,rt,tr,in_deg,seg_deg,out_deg,traj_deg,gain_dB,cur\n");
        for (double p : {0.90, 1.01, 1.04, 1.10, 1.20, 1.45})
            for (int itype : {4, 5})
                for (int kind = 0; kind < 3; ++kind)
                    for (int tf = 0; tf < 2; ++tf)
                    {
                        arma::uvec mi;
                        arma::mat mesh = kind == 0   ? build({0.0, gap, L}, {DENSE, HOST, HOST}, {true, false, true}, mi)
                                         : kind == 1 ? build({0.0, gap, L, L + gap}, {DENSE, HOST, HOST, SHEET}, {true, false, true, true}, mi)
                                                     : build({0.0, gap, L, L + gap, L + 0.05}, {DENSE, HOST, HOST, RARE, RARE}, {true, false, true, false, true}, mi);
                        double th1 = std::asin(std::sqrt(p / e_d));
                        arma::mat orig{{-0.05, 0.37, 0.0}}, dest{{-0.05 + 2.0 * std::cos(th1), 0.37 + 2.0 * std::sin(th1), 0.0}};
                        Ev seed;
                        seed.prev = arma::Col<short>{(short)0}, seed.cur = arma::Col<short>{(short)DENSE}, seed.buf = arma::Col<short>{(short)0};
                        seed.path_dirN = unit(dest - orig), seed.accg = arma::vec{0.0}, seed.accr = arma::vec{0.0};
                        const Pal &pal = tf ? with_tf : base;

                        // Two events by hand, so that the second can be read in full
                        Ev A = traced(pal, mesh, mi, itype, orig, dest, &seed, 1.5);
                        REQUIRE(A.nh != 0);
                        REQUIRE(mat_of(A.cur(0)) == HOST);
                        arma::mat d = unit(A.destN - A.origN), o2 = A.origN, d2 = A.origN + 2.0 * d;
                        Ev B = traced(pal, mesh, mi, itype, o2, d2, &A, 1.5);
                        REQUIRE(B.nh != 0);
                        QD_DIAG("%.2f,%d,%s,%d,%u,%u,%d,%.4f,%.4f,%.4f,%.4f,%+.4f,%04x\n", p, itype, kind == 0 ? "bare " : (kind == 1 ? "glued" : "trans"), tf,
                                (unsigned)B.ot, (unsigned)B.rt, (int)rt_bit(B.rt, 5), ang(A.path_dirN), ang(d), ang(B.path_dirN),
                                ang(B.destN - B.origN), dB(std::norm(cpx(B.xpr, 0, 0))), (unsigned)(unsigned short)B.cur(0));
                        INFO("closed face, p " << p << ", tracer " << itype << ", face " << kind << ", tf " << tf);
                        CHECK(rt_bit(B.rt, 5) == (p > 1.0));
                        if (p > 1.0) // closed: the leak travels undeviated, whatever the face is
                        {
                            arma::mat w = unit(A.path_dirN), o = unit(B.path_dirN);
                            CHECK(std::max(std::abs(o(0, 0) - w(0, 0)), std::abs(o(0, 1) - w(0, 1))) < 1.0e-12);
                        }
                    }
    }
}

TEST_CASE("Measurement - the mirror behind a second solid and behind a sheet inside a wall", "[wave]")
{
    ub::OneThread one_thread;
    // Two traced solids that share a face are two series, and the second one's takes the first as a half-space where
    // the wall has it as a layer backed by air; a sheet inside a wall is the same limit. Everything runs through
    // ray_triangle_intersect, ray_mesh_interact and ray_state_update, scalar, both tracers. The references are written
    // here as a TE characteristic matrix in the admittance b = sqrt(eps - sin^2), sharing no code with the library.
    // Asserted: each series is its own body's between half-spaces, and the paths a tracer follows sum to the wall; the
    // rest is printed.
    //
    // A. The series of each solid as a ratio, the resolving run over the declined one, against the factorized form with
    // the mirror behind as a half-space and as a layer backed by air.
    // B. The transmitted field as a sum over the paths a tracer follows, every series resolved: the straight path, then
    // the paths that reflect at the far face, return into the first solid, reflect at the front face and come forward
    // again, each at its polyline's point.
    // C. The same two for a wall with a sheet inside it.
    // D. Three solids: what each candidate mirror would leave on the third solid's series.
    // E. The thickness code: the code the library writes, and what its step costs the mirror.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs, sbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        E.sbs = sbsN;
        return E;
    };
    using ub::build;
    using ub::face;
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev;
        std::vector<double> fx, bx; // per event: x of the hit point, x where the leg that follows starts (qd_leg_start)
        bool ok = true;
        std::string rts, words;
        std::complex<double> at_point(double k, double st) const // the field at the polyline's exit point
        {
            return vv * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * st * dy));
        }
    };
    // seq: one type per event, or one type repeated until the ray is out. epsv likewise
    auto run = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq,
                   const std::vector<double> &epsv, double thd, double f) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        const int n_max = seq.size() == 1 ? 16 : (int)seq.size();
        for (int k = 0; k < n_max; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : seq[(size_t)k];
            double eps = epsv.size() == 1 ? epsv[0] : epsv[(size_t)k];
            bool tx = itype != 0 && itype != 3;
            Ev E = traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, eps, tx);
            if (E.nh == 0)
            {
                C.ok = seq.size() == 1;
                break;
            }
            C.rts += std::to_string((unsigned)E.rt) + "/";
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            // long enough to cross every face that is left
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            C.fx.push_back(fx), C.bx.push_back(qd_leg_start(E, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        if (seq.size() > 1 && C.ev.size() != seq.size())
            C.ok = false;
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto tint = [&](std::complex<double> ea, std::complex<double> eb, double s2)
    { return 2.0 * bof(ea, s2) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };

    const double L1 = 0.05, L2 = 0.10;
    const double k1 = 2.0 * QPI * FRQ / C0;

    // ---- A. The series of each solid ----
    QD_DIAG("\n# diag: mirror, the series of each solid. air | A 50 mm | B 100 mm | air through the intersector, scalar, 1 GHz.\n");
    QD_DIAG("# S_B: B's exit, resolving over declined. d_half: |S_B - S| with A behind as a half-space; d_layer: with A as a\n");
    QD_DIAG("# layer of its thickness backed by air; sep: the two references apart; worth_dB: 20 log |S_layer / S_half|.\n");
    QD_DIAG("# |R_h|, |R_l|: the mirror behind, half-space and layer. d_A: A's own series against air | B half-space.\n");
    QD_DIAG("# d_wall: the straight path, every series resolved, against the matrix of the whole wall, dB and deg at the\n");
    QD_DIAG("# polyline's point. resid: the same with S_B replaced by S_layer. fact: the factorized form against the matrix\n");
    QD_DIAG("tracer,A,B,theta,|S_B|,arg_S_B,d_half,d_layer,sep,worth_dB,|R_h|,|R_l|,d_A,d_wall_dB,d_wall_deg,resid_dB,resid_deg,fact\n");
    for (int tr : {4, 5})
        for (int B : {5, 6})
            for (int A : {2, 1})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {A, A, B, B}, {false, true, false, true}, mi);
                    Chain cD = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 1.5}, thd, FRQ);
                    Chain cB = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 0.0}, thd, FRQ);
                    Chain cA = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 0.0, 1.5}, thd, FRQ);
                    Chain c0 = run(mtl, mesh, mi, {tr, tr, tr}, {0.0, 0.0, 0.0}, thd, FRQ);
                    INFO("series, tracer " << tr << ", A " << A << ", B " << B << ", theta " << thd);
                    REQUIRE((cD.ok && cB.ok && cA.ok && c0.ok));
                    double LA = cD.fx[1] - cD.bx[0], LB = cD.fx[2] - cD.bx[1]; // the thickness each leg crossed
                    std::complex<double> eA = diag_eps(A, FRQ), eB = diag_eps(B, FRQ);
                    std::complex<double> S_B = cpx(cB.ev[2].xpr, 0, 0) / cpx(cD.ev[2].xpr, 0, 0);
                    std::complex<double> S_A = cpx(cA.ev[1].xpr, 0, 0) / cpx(cD.ev[1].xpr, 0, 0);
                    std::complex<double> pA = phi(eA, s2, k1, LA), pB = phi(eB, s2, k1, LB);
                    std::complex<double> R_h = rint(eB, eA, s2), R_l = cm(eB, {{eA, LA}}, e_air, s2, k1).r, r_out = rint(eB, e_air, s2);
                    std::complex<double> S_half = 1.0 / (1.0 - R_h * r_out * pB * pB), S_layer = 1.0 / (1.0 - R_l * r_out * pB * pB);
                    std::complex<double> S_A_ref = 1.0 / (1.0 - rint(eA, e_air, s2) * rint(eA, eB, s2) * pA * pA);
                    std::complex<double> W = cm(e_air, {{eA, LA}, {eB, LB}}, e_air, s2, k1).t;
                    std::complex<double> fact = cm(e_air, {{eA, LA}}, eB, s2, k1).t * pB * tint(eB, e_air, s2) * S_layer;
                    std::complex<double> F0 = c0.at_point(k1, st) / W, F0c = F0 * S_layer / S_B;
                    QD_DIAG("%d,%d,%d,%.0f,%.6f,%+.3f,%.3e,%.3e,%.3e,%+.4f,%.4f,%.4f,%.3e,%+.4f,%+.3f,%+.6f,%+.4f,%.1e\n", tr, A, B, thd,
                            std::abs(S_B), deg(S_B), std::abs(S_B - S_half), std::abs(S_B - S_layer), std::abs(S_half - S_layer),
                            2.0 * dB(std::abs(S_layer / S_half)), std::abs(R_h), std::abs(R_l), std::abs(S_A - S_A_ref),
                            dB(std::norm(F0)), deg(F0), dB(std::norm(F0c)), deg(F0c), std::abs(fact - W) / std::abs(W));
                    CHECK(std::abs(fact - W) < 1.0e-12 * std::abs(W)); // the factorization is the wall, reference against reference
                    // Each series is the one of its own body between half-spaces: that is the
                    // decomposition, and the paths below carry what the half-spaces do not return
                    CHECK(std::abs(S_B - S_half) < 1.0e-9);
                    CHECK(std::abs(S_A - S_A_ref) < 1.0e-9);
                    CHECK(std::abs(S_half - S_layer) > 1.0e-3);   // the two mirrors can be told apart
                    CHECK(std::abs(dB(std::norm(F0c))) < 1.0e-3); // the first path is off the wall by the mirror and nothing else
                }

    // ---- B. The paths a tracer follows, summed ----
    auto seq_of = [](int tr, int n) // the path that couples the two series n times
    {
        std::vector<int> s = {tr, tr};
        for (int i = 0; i < n; ++i)
            s.push_back(3), s.push_back(tr), s.push_back(3), s.push_back(tr);
        s.push_back(tr);
        return s;
    };
    QD_DIAG("\n# diag: mirror, the paths a tracer follows. The same wall, every series resolved (eps 0). Path n reflects n times\n");
    QD_DIAG("# at B's far face, returns into A, reflects at A's front face and comes forward again: 3 + 4 n events. Each path\n");
    QD_DIAG("# at its polyline's exit point; d0, d01, d012: the sum of the first one, two and three paths against the matrix\n");
    QD_DIAG("# of the whole wall, dB; ph: its phase, deg. p1_dB: path 1 over path 0. geo: |F2/F1 - F1/F0|, the paths a\n");
    QD_DIAG("# geometric sequence. dinf: its limit F0 / (1 - F1/F0) against the matrix, dB and deg. rts1: the resolved types of path 1\n");
    QD_DIAG("tracer,A,B,theta,d0_dB,d01_dB,d012_dB,ph0,ph01,ph012,p1_dB,p2_dB,geo,dinf_dB,dinf_deg,rts1\n");
    for (int tr : {4, 5})
        for (int B : {5, 6})
            for (int A : {2, 1})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {A, A, B, B}, {false, true, false, true}, mi);
                    Chain c[3];
                    std::complex<double> F[3] = {0.0, 0.0, 0.0};
                    for (int n = 0; n < 3; ++n)
                    {
                        c[n] = run(mtl, mesh, mi, seq_of(tr, n), {0.0}, thd, FRQ);
                        if (c[n].ok)
                            F[n] = c[n].at_point(k1, st);
                    }
                    INFO("paths, tracer " << tr << ", A " << A << ", B " << B << ", theta " << thd);
                    REQUIRE(c[0].ok);
                    double LA = c[0].fx[1] - c[0].bx[0], LB = c[0].fx[2] - c[0].bx[1];
                    std::complex<double> W = cm(e_air, {{diag_eps(A, FRQ), LA}, {diag_eps(B, FRQ), LB}}, e_air, s2, k1).t;
                    std::complex<double> s0 = F[0] / W, s1 = (F[0] + F[1]) / W, s2c = (F[0] + F[1] + F[2]) / W;
                    std::complex<double> q = F[1] / F[0], sinf = s0 / (1.0 - q);
                    QD_DIAG("%d,%d,%d,%.0f,%+.4f,%+.4f,%+.4f,%+.3f,%+.3f,%+.3f,%+.2f,%+.2f,%.1e,%+.6f,%+.4f,%s%s\n", tr, A, B, thd, dB(std::norm(s0)),
                            dB(std::norm(s1)), dB(std::norm(s2c)), deg(s0), deg(s1), deg(s2c), dB(std::norm(F[1] / F[0])),
                            dB(std::norm(F[2] / F[0])), std::abs(F[2] / F[1] - q), dB(std::norm(sinf)), deg(sinf), c[1].rts.c_str(),
                            c[1].ok ? "" : " dead");
                    // The paths the tracer follows are a geometric sequence and their sum is the wall
                    REQUIRE((c[1].ok && c[2].ok));
                    CHECK(std::abs(F[2] / F[1] - q) < 1.0e-4);
                    // Every leg is measured from the point it started at, so the paths hold the thicknesses of the
                    // reference. With the refracting tracer at 85 deg a lossless second layer still leaves 1.3e-3 dB on
                    // the limit
                    CHECK(std::abs(dB(std::norm(sinf))) < ((tr == 5 && B == 6 && thd == 85.0) ? 2.0e-3 : 1.0e-3));
                    CHECK(std::abs(deg(sinf)) < 0.05);
                }
    {
        arma::uvec mi;
        arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {2, 2, 5, 5}, {false, true, false, true}, mi);
        Chain c = run(mtl, mesh, mi, seq_of(4, 1), {0.0}, 40.0, FRQ);
        QD_DIAG("# the state words after each event of path 1, A 2, B 5, 40 deg, tracer 4, as prev.cur.buf: %s\n", c.words.c_str());
    }

    // ---- C. A wall with a sheet inside it ----
    QD_DIAG("\n# diag: mirror, a sheet inside a wall. air | host 50 | sheet 6 mm eps 5.24 - 0.63j | host 50 | air, scalar, 1 GHz.\n");
    QD_DIAG("# S_2: the second half's series at the exit; d_host: against the sheet behind backed by a half-space of the host;\n");
    QD_DIAG("# d_wall: against the sheet and the first half backed by air; worth_dB: 20 log of the two. d_1: the first half's\n");
    QD_DIAG("# series at the sheet event against air | sheet backed by a half-space of the host. Paths as above, the sheet\n");
    QD_DIAG("# crossed in place of the shared face, against the matrix of host | sheet | host with 6 mm of host divided out\n");
    QD_DIAG("# (W). dinf: the limit of the paths against W; dinf2: against host 47 | sheet 6 | host 47, nothing divided out;\n");
    QD_DIAG("# dq: the path ratio F1/F0 against the one of W, r^2 Ts^2 pa^2 pb^2 S_1 S_host, and dq_d: with Ts over the 6 mm of host\n");
    QD_DIAG("tracer,host,theta,|S_2|,arg_S_2,d_host,d_wall,sep,worth_dB,|R_h|,|R_w|,d_1,d0_dB,d01_dB,d012_dB,ph0,ph01,ph012,p1_dB,fact,geo,dinf_dB,dinf_deg,dinf2_dB,dinf2_deg,dq,dq_d,rts1\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, 0.10}, {h, SHEET, h}, {false, false, true}, mi);
                Chain cD = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 1.5}, thd, FRQ);
                Chain c2 = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 0.0}, thd, FRQ);
                Chain c1 = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 0.0, 1.5}, thd, FRQ);
                INFO("sheet wall, tracer " << tr << ", host " << h << ", theta " << thd);
                REQUIRE((cD.ok && c2.ok && c1.ok));
                double La = cD.fx[1] - cD.bx[0], Lb = cD.fx[2] - cD.bx[1];
                std::complex<double> eH = diag_eps(h, FRQ), eS = diag_eps(SHEET, FRQ);
                std::complex<double> S_2 = cpx(c2.ev[2].xpr, 0, 0) / cpx(cD.ev[2].xpr, 0, 0);
                std::complex<double> S_1 = cpx(c1.ev[1].xpr, 0, 0) / cpx(cD.ev[1].xpr, 0, 0);
                std::complex<double> pa = phi(eH, s2, k1, La), pb = phi(eH, s2, k1, Lb), r_out = rint(eH, e_air, s2);
                RT Sh = cm(eH, {{eS, thk}}, eH, s2, k1);
                std::complex<double> R_h = Sh.r, R_w = cm(eH, {{eS, thk}, {eH, La}}, e_air, s2, k1).r;
                std::complex<double> S_host = 1.0 / (1.0 - R_h * r_out * pb * pb), S_wall = 1.0 / (1.0 - R_w * r_out * pb * pb);
                std::complex<double> S_1_ref = 1.0 / (1.0 - r_out * R_h * pa * pa);
                std::complex<double> W = cm(e_air, {{eH, La}, {eS, thk}, {eH, Lb}}, e_air, s2, k1).t / phi(eH, s2, k1, thk);
                std::complex<double> fact = tint(e_air, eH, s2) * pa * S_1_ref * (Sh.t / phi(eH, s2, k1, thk)) * pb * S_wall * tint(eH, e_air, s2);
                Chain c[3];
                std::complex<double> F[3] = {0.0, 0.0, 0.0};
                for (int n = 0; n < 3; ++n)
                {
                    c[n] = run(mtl, mesh, mi, seq_of(tr, n), {0.0}, thd, FRQ);
                    if (c[n].ok)
                        F[n] = c[n].at_point(k1, st);
                }
                REQUIRE(c[0].ok);
                std::complex<double> s0 = F[0] / W, s1 = (F[0] + F[1]) / W, s2c = (F[0] + F[1] + F[2]) / W;
                std::complex<double> q = F[1] / F[0], sinf = s0 / (1.0 - q);
                std::complex<double> W2 = cm(e_air, {{eH, La - 0.5 * thk}, {eS, thk}, {eH, Lb - 0.5 * thk}}, e_air, s2, k1).t;
                std::complex<double> sinf2 = sinf * W / W2;
                std::complex<double> q_W = r_out * r_out * Sh.t * Sh.t * pa * pa * pb * pb * S_1_ref * S_host;
                std::complex<double> q_d = q_W / (phi(eH, s2, k1, thk) * phi(eH, s2, k1, thk));
                QD_DIAG("%d,%d,%.0f,%.6f,%+.3f,%.3e,%.3e,%.3e,%+.4f,%.4f,%.4f,%.3e,%+.4f,%+.4f,%+.4f,%+.3f,%+.3f,%+.3f,%+.2f,%.1e,%.1e,%+.6f,%+.4f,%+.6f,%+.4f,%.2e,%.2e,%s%s\n", tr, h,
                        thd, std::abs(S_2), deg(S_2), std::abs(S_2 - S_host), std::abs(S_2 - S_wall), std::abs(S_host - S_wall),
                        2.0 * dB(std::abs(S_wall / S_host)), std::abs(R_h), std::abs(R_w), std::abs(S_1 - S_1_ref), dB(std::norm(s0)),
                        dB(std::norm(s1)), dB(std::norm(s2c)), deg(s0), deg(s1), deg(s2c), dB(std::norm(F[1] / F[0])),
                        std::abs(fact - W) / std::abs(W), std::abs(F[2] / F[1] - q), dB(std::norm(sinf)), deg(sinf), dB(std::norm(sinf2)),
                        deg(sinf2), std::abs(q - q_W) / std::abs(q_W), std::abs(q - q_d) / std::abs(q_d), c[1].rts.c_str(), c[1].ok ? "" : " dead");
                CHECK(std::abs(fact - W) < 1.0e-12 * std::abs(W));
                CHECK(std::isfinite(std::abs(S_2)));
            }
    {
        arma::uvec mi;
        arma::mat mesh = build({0.0, 0.05, 0.10}, {5, SHEET, 5}, {false, false, true}, mi);
        Chain c = run(mtl, mesh, mi, seq_of(4, 1), {0.0}, 40.0, FRQ);
        QD_DIAG("# the state words after each event of path 1, host 5, 40 deg, tracer 4, as prev.cur.buf: %s\n", c.words.c_str());
    }

    // ---- D. Three solids ----
    QD_DIAG("\n# diag: mirror, three solids. air | A 50 | B 100 | C 50 | air, every series resolved, tracer 4. S_C: C's series\n");
    QD_DIAG("# at the exit. d_half: against B behind as a half-space. In dB on the path, 20 log |S_x / S_exact| with S_exact\n");
    QD_DIAG("# the mirror {B, A} backed by air: half, B a half-space; B_air, B a layer backed by air; B_A, B a layer backed\n");
    QD_DIAG("# by a half-space of A. d_wall: the straight path against the matrix of the whole wall\n");
    QD_DIAG("A,B,C,theta,|S_C|,d_half,half_dB,B_air_dB,B_A_dB,d_wall_dB\n");
    for (int trip = 0; trip < 3; ++trip)
        for (double thd : angles)
        {
            const int A = trip == 0 ? 2 : (trip == 1 ? 1 : 2), B = trip == 0 ? 6 : 5, Cm = trip == 0 ? 2 : (trip == 1 ? 4 : 6);
            const double x1 = L1, x2 = x1 + gap, x3 = x2 + L2, x4 = x3 + gap, x5 = x4 + L1;
            double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
            arma::uvec mi;
            arma::mat mesh = build({0.0, x1, x2, x3, x4, x5}, {A, A, B, B, Cm, Cm}, {false, true, false, true, false, true}, mi);
            Chain cD = run(mtl, mesh, mi, {4, 4, 4, 4}, {0.0, 0.0, 0.0, 1.5}, thd, FRQ);
            Chain c0 = run(mtl, mesh, mi, {4, 4, 4, 4}, {0.0, 0.0, 0.0, 0.0}, thd, FRQ);
            INFO("three solids, " << A << " " << B << " " << Cm << ", theta " << thd);
            REQUIRE((cD.ok && c0.ok));
            double LA = c0.fx[1] - c0.bx[0], LB = c0.fx[2] - c0.bx[1], LC = c0.fx[3] - c0.bx[2];
            std::complex<double> eA = diag_eps(A, FRQ), eB = diag_eps(B, FRQ), eC = diag_eps(Cm, FRQ);
            std::complex<double> S_C = cpx(c0.ev[3].xpr, 0, 0) / cpx(cD.ev[3].xpr, 0, 0), pC = phi(eC, s2, k1, LC), r_out = rint(eC, e_air, s2);
            auto S_of = [&](std::complex<double> R)
            { return 1.0 / (1.0 - R * r_out * pC * pC); };
            std::complex<double> S_half = S_of(rint(eC, eB, s2)), S_Bair = S_of(cm(eC, {{eB, LB}}, e_air, s2, k1).r);
            std::complex<double> S_BA = S_of(cm(eC, {{eB, LB}}, eA, s2, k1).r), S_ex = S_of(cm(eC, {{eB, LB}, {eA, LA}}, e_air, s2, k1).r);
            std::complex<double> W = cm(e_air, {{eA, LA}, {eB, LB}, {eC, LC}}, e_air, s2, k1).t;
            QD_DIAG("%d,%d,%d,%.0f,%.6f,%.3e,%+.4f,%+.4f,%+.4f,%+.4f\n", A, B, Cm, thd, std::abs(S_C), std::abs(S_C - S_half),
                    2.0 * dB(std::abs(S_half / S_ex)), 2.0 * dB(std::abs(S_Bair / S_ex)), 2.0 * dB(std::abs(S_BA / S_ex)),
                    dB(std::norm(c0.at_point(k1, st) / W)));
            CHECK(std::isfinite(std::abs(S_C)));
        }
}

TEST_CASE("Lossy verdict - a lossy second medium has no critical angle", "[wave]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // sqrt|e1 mu1 / e2 mu2| sin(theta) >= 1 is the critical angle of two lossless media. A lossy second medium has
    // none: it always admits a transmitted wave, with a complex cosine, and the flux form closes no port on it. So the
    // magnitude test holds where the second medium is lossless and nowhere else, and every reader of the verdict - the
    // coefficient, the hit code, the direction, the bent flag, the ray tube, the crossings of a mass-law sheet - is
    // continuous through the angle at which the test would trip. A rigid body of the acoustic mapping is lossless and
    // stays closed. Through ray_triangle_intersect, ray_mesh_interact and ray_state_update; the rows of J with a loss
    // column are printed.
    //
    // A. A porous sheet in air, both ports, against the analytic slab
    // B. A porous sheet on a rigid wall and on a leaking one, both ports
    // C. A sheet in front of a lossy half-space rarer than air: the outer pair
    // D. The same porous medium as a traced solid in air: per event, and every path
    // E. The traced solid on a leaking wall against the sheet of its thickness
    // F. ray_mesh_interact alone: the ray tube
    // G. EM: air into the porous medium, per channel
    // H. EM: a lossless dense solid in front of a lossy rarer one, bare and as a coating
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    const double gap = 1.0e-5;
    const std::complex<double> J(0.0, 1.0), one(1.0, 0.0);
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };
    auto ang = [](const arma::mat &v) // angle from the x axis, deg
    { return std::acos(std::abs(v(0, 0)) / std::sqrt(v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1) + v(0, 2) * v(0, 2))) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, xpr0, fbs, dir0; // xpr0, dir0: what ray_mesh_interact wrote
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        E.xpr0 = E.xpr, E.dir0 = E.path_dirN, E.ot = ot[0], E.nh = nh(0);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.rt = rt[0];
        return E;
    };
    using ub::build;
    using ub::face;
    auto onward = [](const Ev &E, arma::mat &orig, arma::mat &dest)
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
        return d(0, 0);
    };
    // One path: the events of seq in a row, each relaunched on the segment the one before wrote
    auto chain = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, const std::vector<int> &seq, double f, double thd,
                     const Ev *start, double eps) -> std::vector<Ev>
    {
        std::vector<Ev> out;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        for (size_t q = 0; q < seq.size(); ++q)
        {
            Ev E = traced(pal, mesh, mi, seq[q], f, o, d, q ? &out[q - 1] : start, eps, seq[q] != 3 && seq[q] != 0);
            out.push_back(E);
            if (E.nh == 0 || E.rt == 0)
                break;
            onward(E, o, d);
        }
        return out;
    };

    // Every path, as in the reflect-port case
    struct Out
    {
        std::complex<double> T = 0.0, R = 0.0;
        int nT = 0, nR = 0, n_cut = 0;
    };
    struct Node
    {
        arma::mat orig, dest;
        Ev P;
        bool have = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, Lin = 0.0, dirx = 1.0;
        std::complex<double> vv = 1.0;
        int depth = 0;
    };
    auto walk = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double eps, double thd, int max_ev,
                    double floor_dB, double x_front, double x_back) -> Out
    {
        Out O;
        const double st = std::sin(thd * QPI / 180.0), floor_lin = std::pow(10.0, 0.1 * floor_dB), k = 2.0 * QPI * f / C0;
        std::vector<Node> stack(1);
        diag_ray_at(thd, stack[0].orig, stack[0].dest);
        while (!stack.empty())
        {
            Node Nn = stack.back();
            stack.pop_back();
            for (int itype : {3, tr})
            {
                const bool tx = itype != 3;
                Ev E = traced(pal, mesh, mi, itype, f, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, eps, tx);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                Node M;
                M.vv = Nn.vv * cpx(E.xpr, 0, 0);
                double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                M.y0 = Nn.depth ? Nn.y0 : fy;
                M.Lin = Nn.Lin + (Nn.depth ? std::sqrt((fx - Nn.xp) * (fx - Nn.xp) + (fy - Nn.yp) * (fy - Nn.yp)) : 0.0);
                M.xp = fx, M.yp = fy, M.depth = Nn.depth + 1;
                M.dirx = onward(E, M.orig, M.dest);
                if (mat_of(E.cur(0)) == 0 && ((M.dirx < 0.0 && fx < x_front + 1.0e-3) || (M.dirx > 0.0 && fx > x_back - 1.0e-3)))
                {
                    std::complex<double> F = M.vv * std::exp(-J * k * M.Lin) * std::exp(J * k * st * (fy - M.y0));
                    if (M.dirx > 0.0)
                        O.T += F, ++O.nT;
                    else
                        O.R += F, ++O.nR;
                    continue;
                }
                if (M.depth >= max_ev || std::norm(M.vv) < floor_lin)
                {
                    ++O.n_cut;
                    continue;
                }
                M.P = E, M.have = true;
                stack.push_back(M);
            }
        }
        return O;
    };

    // The reference: the scalar channel, admittance b = sqrt(eps - sin^2), the decaying root, layer
    // phase k b d. It shares nothing with the library.
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    {
        std::complex<double> b = std::sqrt(e - s2);
        return (std::imag(b) > 0.0) ? -b : b;
    };
    auto cm = [&](const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT // from air
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(one, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };

    const double f_hz = FRQ, k1 = 2.0 * QPI * f_hz / C0, dS = 0.050;
    const std::vector<LumpMtl> lp = {
        {0.31, 0.20, 0.010},                                                 // 1 porous sheet 10 mm, rarer than air
        {0.31, 0.20, 0.0},                                                   // 2 the same medium as a solid
        {2.401e-9, 0.0, 0.0},                                                // 3 rigid wall
        {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                      // 4 rigid wall leaking, tf 0.3
        {0.31, 0.20, dS},                                                    // 5 porous sheet 50 mm
        {2.50, 0.80, 0.010},                                                 // 6 dense lossy sheet 10 mm
        {0.31, 0.20, 0.010, 0.0, 0.2},                                       // 7 porous sheet 10 mm with tf 0.2
        {0.31, 0.20, 0.0, 0.0, 0.2},                                         // 8 porous solid with tf 0.2
        {5.24, 0.00, 0.0},                                                   // 9 EM: lossless dense solid
        {1.50, 0.40, 0.0},                                                   // 10 EM: lossy solid, rarer than 9
        {1.50, 0.40, 0.006, 0.0, 0.5},                                       // 11 EM: the same as a 6 mm sheet with tf 0.5
        {1.50, 0.40, 0.006},                                                 // 12 EM: the same as a 6 mm sheet
        {0.50, 0.60, 0.0},                                                   // 13 a rarer solid with more loss, the test trips at 62.1 deg
        {0.31, 0.60, 0.0},                                                   // 14 the same at 55.3 deg
        {2.401e-9, 1.0e-10, 0.0, 0.0, 0.3},                                  // 15 the leaking rigid wall with a loss tangent of 0.04
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},     // 16 the leaking rigid wall with its mass law
        {2.401e-9, 1.0e-10, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001}, // 17 with both
    };
    Pal mtl = lump_palette(lp);
    const std::complex<double> eP = lump_eps(lp[1], f_hz), eW = lump_eps(lp[2], f_hz), eD = lump_eps(lp[5], f_hz);
    const std::complex<double> eH = lump_eps(lp[8], f_hz), eL = lump_eps(lp[9], f_hz);
    const double tc = std::asin(std::sqrt(std::abs(eP))) * 180.0 / QPI;        // where the test trips, air | porous
    const double tcH = std::asin(std::sqrt(std::abs(eL / eH))) * 180.0 / QPI;  // dense | lossy rarer
    const double tcA = std::asin(std::sqrt(std::abs(1.0 / eH))) * 180.0 / QPI; // dense | air, a critical angle
    QD_DIAG("\n# diag: lossy verdict. The magnitude test trips at %.4f deg for air | eps %.2f%+.2fj, at %.4f deg for\n", tc, std::real(eP), std::imag(eP), tcH);
    QD_DIAG("# eps 5.24 | eps %.2f%+.2fj; eps 5.24 | air has its critical angle at %.4f deg\n", std::real(eL), std::imag(eL), tcA);
    const std::vector<double> angles = {10.0, 20.0, 30.0, tc - 0.01, tc + 0.01, 45.0, 60.0, 75.0, 85.0};
    auto dir_in = [&](std::complex<double> e1, std::complex<double> e2, double thd) // qd_refract's map, deg
    {
        double th = thd * QPI / 180.0, sn = std::sqrt(std::abs(e1 / e2));
        std::complex<double> c2 = std::sqrt(1.0 - (e1 / e2) * std::sin(th) * std::sin(th));
        return std::atan2(sn * std::sin(th), std::abs(std::real(c2))) * 180.0 / QPI;
    };

    // ---- A. A porous sheet in air ----
    QD_DIAG("\n# diag: lossy verdict A, porous sheet 10 mm in air, scalar, first event. T, R: the port against the analytic slab,\n");
    QD_DIAG("# dB and deg; own: ray_mesh_interact's composite against the redo, complex difference; dir: direction written, deg\n");
    QD_DIAG("tracer,theta,ot,rt,T_lib_dB,T_dB,T_deg,R_lib_dB,R_dB,R_deg,own_T,own_R,dir\n");
    for (int tr : {4, 5})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), c = std::cos(th);
            arma::uvec mi;
            arma::mat mesh = build({0.0}, {1}, {false}, mi);
            Ev et = chain(mtl, mesh, mi, {tr}, f_hz, thd, nullptr, 0.0)[0], er = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, 0.0)[0];
            REQUIRE((et.nh != 0 && er.nh != 0));
            RT W = cm({{eP, 0.010}}, one, s2, k1);
            std::complex<double> und = std::exp(J * k1 * c * 0.010), T = cpx(et.xpr, 0, 0), R = cpx(er.xpr, 0, 0);
            QD_DIAG("%d,%.2f,%d,%d,%.4f,%+.6f,%+.4f,%.4f,%+.6f,%+.4f,%.2e,%.2e,%.4f\n", tr, thd, (int)et.ot, (int)et.rt, dB(T), dB(T / (W.t * und)), deg(T / (W.t * und)),
                    dB(R), dB(R / (W.r * und)), deg(R / (W.r * und)), std::abs(cpx(et.xpr0, 0, 0) - T), std::abs(cpx(er.xpr0, 0, 0) - R), ang(et.path_dirN));
            INFO("A, tracer " << tr << ", theta " << thd);
            CHECK((int)et.ot == 67);
            CHECK(std::abs(T / (W.t * und) - 1.0) < 1.0e-9);
            CHECK(std::abs(R / (W.r * und) - 1.0) < 1.0e-9);
        }

    // ---- B. A porous sheet on a rigid wall ----
    QD_DIAG("\n# diag: lossy verdict B, porous sheet 10 mm on a rigid wall, scalar, the glued entry. T_ref: the chain of an\n");
    QD_DIAG("# ordinary crossing, its series summed (diag_leak_series), which has no test on the near face; dT: complex distance\n");
    QD_DIAG("# from it over its magnitude. R_ref: the composite on the rigid backing. T_own: ray_mesh_interact's composite.\n");
    QD_DIAG("# cur: the current word written, hex\n");
    QD_DIAG("sheet,wall,tracer,theta,ot,rt,T_dB,T_ref_dB,dT,T_own_dB,|R|^2,ref_|R|^2,dR,dir,cur\n");
    struct RowB
    {
        const char *sheet, *wall;
        int sh, wl;
    };
    for (const RowB &B : std::vector<RowB>{{"porous", "rigid", 1, 3}, {"porous", "rigid tf .3", 1, 4}, {"porous tf .2", "rigid", 7, 3}, {"porous tf .2", "rigid tf .3", 7, 4}})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                if (tr == 5 && B.wl != 4)
                    continue;
                double th = thd * QPI / 180.0, c = std::cos(th);
                const LumpMtl &Sm = lp[(size_t)B.sh - 1], &Wm = lp[(size_t)B.wl - 1];
                arma::uvec mi;
                arma::mat mesh = build({0.0, gap, 0.2 + gap}, {B.sh, B.wl, B.wl}, {false, false, true}, mi);
                Ev et = chain(mtl, mesh, mi, {tr}, f_hz, thd, nullptr, 0.0)[0], er = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, 0.0)[0];
                REQUIRE((et.nh != 0 && er.nh != 0));
                DiagLeak D = diag_leak_series(1.0, eP, eW, 0.0, Sm.tf, Wm.tf, Sm.thk, c, f_hz, true);
                DiagLump Lq;
                Lq.tf = Sm.tf, Lq.tf_far = Wm.tf;
                DiagSheet Dr = diag_sheet_composite(1.0, eP, eW, Sm.thk, c, f_hz, false, true, 1.0, 1.0, 1.0, Lq);
                std::complex<double> T = cpx(et.xpr, 0, 0), R = cpx(er.xpr, 0, 0);
                QD_DIAG("%s,%s,%d,%.2f,%d,%d,%.4f,%.4f,%.2e,%.4f,%.6f,%.6f,%.2e,%.4f,%04x\n", B.sheet, B.wall, tr, thd, (int)et.ot, (int)et.rt, dB(T), dB(D.T_te),
                        (std::abs(D.T_te) > 0.0) ? std::abs(T - D.T_te) / std::abs(D.T_te) : std::abs(T), dB(cpx(et.xpr0, 0, 0)), std::norm(R), std::norm(Dr.R_te),
                        std::abs(R - Dr.R_te), ang(et.path_dirN), (unsigned)(unsigned short)et.cur(0));
                INFO("B, " << B.sheet << " | " << B.wall << ", tracer " << tr << ", theta " << thd);
                CHECK((int)et.ot == 103); // the outer pair air | rigid is closed, and stays so
                CHECK(std::abs(ang(et.path_dirN) - thd) < 1.0e-9);
                CHECK(std::abs(R - Dr.R_te) < 1.0e-9);
                CHECK(std::abs(T - D.T_te) < 1.0e-9 * std::abs(D.T_te) + 1.0e-12); // an ordinary crossing at every angle
                CHECK(std::abs(cpx(et.xpr0, 0, 0) - T) < 1.0e-12);                 // ray_mesh_interact's own composite agrees
            }

    // ---- C. A sheet in front of a lossy half-space rarer than air: the outer pair ----
    QD_DIAG("\n# diag: lossy verdict C, dense lossy sheet 10 mm glued on the porous medium as a half-space, scalar. The outer pair\n");
    QD_DIAG("# air | porous trips, the near face never does. T, R against the analytic layer on the half-space, the transmit\n");
    QD_DIAG("# port in normal flux; dB and deg\n");
    QD_DIAG("tracer,theta,ot,rt,T_lib_dB,T_ref_dB,T_dB,T_deg,R_lib_dB,R_dB,R_deg,T_own_dB,dir,cur\n");
    for (int tr : {4, 5})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), c = std::cos(th);
            arma::uvec mi;
            arma::mat mesh = build({0.0, gap, 0.5 + gap}, {6, 2, 2}, {false, false, true}, mi);
            Ev et = chain(mtl, mesh, mi, {tr}, f_hz, thd, nullptr, 0.0)[0], er = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, 0.0)[0];
            REQUIRE((et.nh != 0 && er.nh != 0));
            RT W = cm({{eD, 0.010}}, eP, s2, k1);
            std::complex<double> und = std::exp(J * k1 * c * 0.010), T = cpx(et.xpr, 0, 0), R = cpx(er.xpr, 0, 0);
            std::complex<double> Tr = W.t * std::sqrt(std::real(bof(eP, s2)) / c) * und, Rr = W.r * und * und;
            QD_DIAG("%d,%.2f,%d,%d,%.4f,%.4f,%+.6f,%+.4f,%.4f,%+.6f,%+.4f,%.4f,%.4f,%04x\n", tr, thd, (int)et.ot, (int)et.rt, dB(T), dB(Tr), dB(T / Tr), deg(T / Tr),
                    dB(R), dB(R / Rr), deg(R / Rr), dB(cpx(et.xpr0, 0, 0)), ang(et.path_dirN), (unsigned)(unsigned short)et.cur(0));
            INFO("C, tracer " << tr << ", theta " << thd);
            CHECK((int)et.ot == 71);
            CHECK(std::abs(T / Tr - 1.0) < 1.0e-9);
            CHECK(std::abs(R / Rr - 1.0) < 1.0e-9);
            CHECK(std::abs(cpx(et.xpr0, 0, 0) - T) < 1.0e-12);
            CHECK(std::abs(ang(et.path_dirN) - dir_in(one, eP, thd)) < 1.0e-6);
            CHECK((unsigned)(unsigned short)et.cur(0) == ((tr == 5) ? 0x4002u : 0x0002u)); // the refracting tracer bent
        }

    // ---- D. The porous medium as a traced solid in air ----
    QD_DIAG("\n# diag: lossy verdict D1, porous solid 50 mm in air, scalar, per event. entry: air | porous, against the half-space\n");
    QD_DIAG("# 1 - |r|^2 and |r|^2; exit: the event behind it. dir: direction written; seg: the segment relaunched on; d0: what\n");
    QD_DIAG("# ray_mesh_interact wrote; deg. cur: the current word after the entry, hex\n");
    QD_DIAG("tracer,theta,ot,rt,T_dB,T_ref_dB,|R|^2,ref_|R|^2,d0,dir,seg,cur,ot2,rt2,T2_dB,dir2,seg2\n");
    for (int tr : {4, 5})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), c = std::cos(th);
            arma::uvec mi;
            arma::mat mesh = build({0.0, dS}, {2, 2}, {false, true}, mi);
            std::vector<Ev> ev = chain(mtl, mesh, mi, {tr, tr}, f_hz, thd, nullptr, 0.0);
            Ev er = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, 0.0)[0];
            std::complex<double> b = bof(eP, s2), r = (c - b) / (c + b);
            arma::mat sg = ev[0].destN - ev[0].origN;
            QD_DIAG("%d,%.2f,%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.4f,%04x", tr, thd, (int)ev[0].ot, (int)ev[0].rt, dB(cpx(ev[0].xpr, 0, 0)), 10.0 * std::log10(1.0 - std::norm(r)),
                    std::norm(cpx(er.xpr, 0, 0)), std::norm(r), ang(ev[0].dir0), ang(ev[0].path_dirN), ang(sg), (unsigned)(unsigned short)ev[0].cur(0));
            if (ev.size() > 1 && ev[1].nh != 0)
            {
                arma::mat s2g = ev[1].destN - ev[1].origN;
                QD_DIAG(",%d,%d,%.4f,%.4f,%.4f\n", (int)ev[1].ot, (int)ev[1].rt, dB(cpx(ev[1].xpr, 0, 0)), ang(ev[1].path_dirN), ang(s2g));
            }
            else
                QD_DIAG(",,,,,\n");
            INFO("D1, tracer " << tr << ", theta " << thd);
            CHECK((int)ev[0].ot == 3);
            CHECK(std::abs(std::norm(cpx(ev[0].xpr, 0, 0)) - (1.0 - std::norm(r))) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(er.xpr, 0, 0)) - std::norm(r)) < 1.0e-9);
            CHECK(std::abs(ang(ev[0].path_dirN) - dir_in(one, eP, thd)) < 1.0e-6);
            CHECK(std::abs(ang(sg) - ((tr == 5) ? dir_in(one, eP, thd) : thd)) < 1.0e-6);
            CHECK((unsigned)(unsigned short)ev[0].cur(0) == ((tr == 5) ? 0x4002u : 0x0002u));
            REQUIRE((ev.size() > 1 && ev[1].nh != 0));
            CHECK(std::abs(ang(ev[1].path_dirN) - thd) < 1.0e-6); // out of a parallel slab as it arrived
        }

    QD_DIAG("\n# diag: lossy verdict D2, porous solid 50 mm in air, scalar, every path, 24 events, floor -150 dB. T, R: the sums\n");
    QD_DIAG("# against the analytic slab, dB and deg; lib: the sum itself\n");
    QD_DIAG("tracer,eps,theta,nT,nR,T_lib_dB,T_ref_dB,T_dB,T_deg,R_lib_dB,R_ref_dB,R_dB,R_deg\n");
    for (int tr : {4, 5})
        for (double eps : {0.0, 0.15})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, dS}, {2, 2}, {false, true}, mi);
                Out O = walk(mtl, mesh, mi, tr, f_hz, eps, thd, 24, -150.0, 0.0, dS);
                RT W = cm({{eP, dS}}, one, s2, k1);
                QD_DIAG("%d,%.2f,%.2f,%d,%d,%.4f,%.4f,%+.6f,%+.4f,%.4f,%.4f,%+.6f,%+.4f\n", tr, eps, thd, O.nT, O.nR, dB(O.T), dB(W.t), (O.nT ? dB(O.T / W.t) : -999.0),
                        (O.nT ? deg(O.T / W.t) : 0.0), dB(O.R), dB(W.r), dB(O.R / W.r), deg(O.R / W.r));
                INFO("D2, tracer " << tr << ", eps " << eps << ", theta " << thd);
                CHECK(O.nT >= 1);
                CHECK(O.nR >= 2);
                if (O.nT >= 1)
                {
                    CHECK(std::abs(dB(O.T / W.t)) < 5.0e-3); // a traced leg is short by the relaunch offset
                    CHECK(std::abs(deg(O.T / W.t)) < 0.05);
                }
                CHECK(std::abs(dB(O.R / W.r)) < 5.0e-3);
                CHECK(std::abs(deg(O.R / W.r)) < 0.05);
            }

    // ---- E. The traced solid on a leaking wall against the sheet of its thickness ----
    QD_DIAG("\n# diag: lossy verdict E, porous 50 mm on a rigid wall with tf 0.3, scalar: as a traced solid, three events (entry,\n");
    QD_DIAG("# porous | wall, wall | air), and as a sheet, two (glued entry, wall | air). Power of each event and of the path,\n");
    QD_DIAG("# dB; ref: the sheet's chain of an ordinary crossing, series summed. dirW: direction in the wall, deg\n");
    QD_DIAG("tracer,eps,theta,solid_e1,solid_e2,solid_e3,solid_path,ot1,rt1,ot2,rt2,dirW_solid,sheet_e1,sheet_e2,sheet_path,ref_e1,dirW_sheet,R_solid,R_sheet\n");
    for (int tr : {4, 5})
        for (double eps : {0.0, 0.15})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, c = std::cos(th);
                arma::uvec mi, mj;
                arma::mat mesh = build({0.0, dS, dS + gap, dS + 0.2 + gap}, {2, 2, 4, 4}, {false, true, false, true}, mi);
                arma::mat mesh2 = build({0.0, gap, 0.2 + gap}, {5, 4, 4}, {false, false, true}, mj);
                std::vector<Ev> so = chain(mtl, mesh, mi, {tr, tr, tr}, f_hz, thd, nullptr, eps), sh = chain(mtl, mesh2, mj, {tr, tr}, f_hz, thd, nullptr, eps);
                Ev rs = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, eps)[0], rh = chain(mtl, mesh2, mj, {3}, f_hz, thd, nullptr, eps)[0];
                DiagLeak D = diag_leak_series(1.0, eP, eW, 0.0, 0.0, 0.3, dS, c, f_hz, true);
                auto g = [&](const std::vector<Ev> &v, size_t q)
                { return (q < v.size() && v[q].nh != 0) ? dB(cpx(v[q].xpr, 0, 0)) : -999.0; };
                auto tot = [&](const std::vector<Ev> &v, size_t n)
                {
                    double s = 0.0;
                    for (size_t q = 0; q < n; ++q)
                        s += g(v, q);
                    return (s < -900.0) ? -999.0 : s;
                };
                QD_DIAG("%d,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n", tr, eps, thd, g(so, 0), g(so, 1), g(so, 2), tot(so, 3),
                        (int)so[0].ot, (int)so[0].rt, (so.size() > 1) ? (int)so[1].ot : -1, (so.size() > 1) ? (int)so[1].rt : -1,
                        (so.size() > 1 && so[1].nh) ? ang(so[1].path_dirN) : -1.0, g(sh, 0), g(sh, 1), tot(sh, 2), dB(D.T_te), ang(sh[0].path_dirN),
                        std::norm(cpx(rs.xpr, 0, 0)), std::norm(cpx(rh.xpr, 0, 0)));
                INFO("E, tracer " << tr << ", eps " << eps << ", theta " << thd);
                const bool live = g(so, 0) > -900.0 && g(so, 1) > -900.0 && g(sh, 0) > -900.0;
                CHECK(live); // the leak into the wall exists at every angle
                CHECK(std::abs(g(sh, 0) - dB(D.T_te)) < 1.0e-6);
                if (live && eps == 0.0) // series summed: the solid's two events are the sheet's one
                    CHECK(std::abs(g(so, 0) + g(so, 1) - g(sh, 0)) < 5.0e-3);
            }

    // ---- F. ray_mesh_interact alone: the ray tube ----
    QD_DIAG("\n# diag: lossy verdict F, ray_mesh_interact alone with a ray tube, type 5, air | porous solid. The spine at theta,\n");
    QD_DIAG("# two vertex rays at theta -+ 0.6 deg in the plane and one tilted out of it. gain0: without the tube\n");
    QD_DIAG("theta,ot_tube,gain_tube_dB,ot_spine,gain_spine_dB,dir_tube,dir_spine,v1,v2,v3\n");
    for (double thd : {tc - 3.0, tc - 0.3, tc + 0.3, tc + 3.0})
    {
        arma::uvec mi;
        arma::mat mesh = build({0.0, dS}, {2, 2}, {false, true}, mi), orig, dest, fbs, sbs;
        diag_ray_at(thd, orig, dest);
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        auto dirv = [](double a, double z)
        {
            arma::rowvec v = {std::cos(a * QPI / 180.0), std::sin(a * QPI / 180.0), z};
            return arma::rowvec(v / std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)));
        };
        arma::mat trivec(1, 9), tridir(1, 9);
        trivec.row(0) = arma::rowvec{0.0, 0.001, 0.0, 0.0, -0.001, 0.001, 0.0, -0.001, -0.001};
        arma::rowvec v1 = dirv(thd + 0.6, 0.0), v2 = dirv(thd - 0.6, 0.0), v3 = dirv(thd, 0.01);
        tridir.row(0) = arma::join_rows(arma::join_rows(v1, v2), v3);
        arma::mat oN, dN, fN, sN, xp, tvN, tdN, pd, oN2, dN2, fN2, sN2, xp2, pd2;
        arma::vec gn, gn2;
        std::vector<uint8_t> ot, ot2;
        quadriga_lib::ray_mesh_interact(5, f_hz, &orig, &dest, &mesh, &mi, &mtl, &fi, &si, &trivec, &tridir, &oN, &dN, &fN, &sN, &gn, &xp, &tvN, &tdN,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr, &ot, &pd, false,
                                        (arma::u32_vec *)nullptr);
        quadriga_lib::ray_mesh_interact(5, f_hz, &orig, &dest, &mesh, &mi, &mtl, &fi, &si, (arma::mat *)nullptr, (arma::mat *)nullptr, &oN2, &dN2, &fN2,
                                        &sN2, &gn2, &xp2, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, &ot2, &pd2, false, (arma::u32_vec *)nullptr);
        arma::mat w1 = tdN.cols(0, 2), w2 = tdN.cols(3, 5), w3 = tdN.cols(6, 8);
        QD_DIAG("%.2f,%d,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", thd, (int)ot[0], (gn(0) > 0.0) ? 10.0 * std::log10(gn(0)) : -999.0, (int)ot2[0],
                (gn2(0) > 0.0) ? 10.0 * std::log10(gn2(0)) : -999.0, ang(pd), ang(pd2), ang(w1), ang(w2), ang(w3));
        INFO("F, theta " << thd);
        CHECK((int)ot[0] == 3); // no vertex closes a tube on a lossy medium
        CHECK(std::abs(gn(0) - gn2(0)) < 1.0e-12);
        CHECK(std::abs(ang(pd) - ang(pd2)) < 1.0e-9);
    }

    // ---- I. The recovery in a lossy solid rarer than air ----
    // A wave from air leaves a parallel slab at the angle it arrived at. Where the exit direction
    // is off, the cosine recovered in the solid is. Past the angle the real-index start of the
    // solve is far from the root, and the solve has to run until it has converged.
    QD_DIAG("\n# diag: lossy verdict I, a lossy solid rarer than air, 50 mm in air, scalar, entry and exit. trip: where the\n");
    QD_DIAG("# magnitude test trips. d_dir: the direction the exit writes minus theta, deg; T, R: every path against the\n");
    QD_DIAG("# analytic slab, dB; inside: the direction in the solid, deg\n");
    QD_DIAG("eps,trip,tracer,theta,ot,inside,d_dir,T_dB,R_dB\n");
    for (int m : {2, 13, 14})
        for (int tr : {4, 5})
        {
            const std::complex<double> eM = lump_eps(lp[(size_t)m - 1], f_hz);
            const double tm = std::asin(std::sqrt(std::abs(eM))) * 180.0 / QPI;
            for (double thd : {10.0, 20.0, 30.0, 37.0, 45.0, 50.0, 55.0, 60.0, 62.0, 75.0, 85.0})
            {
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, dS}, {m, m}, {false, true}, mi);
                std::vector<Ev> ev = chain(mtl, mesh, mi, {tr, tr}, f_hz, thd, nullptr, 0.0);
                REQUIRE((ev.size() == 2 && ev[1].nh != 0));
                Out O = walk(mtl, mesh, mi, tr, f_hz, 0.0, thd, 24, -150.0, 0.0, dS);
                RT W = cm({{eM, dS}}, one, s2, k1);
                (void)tm;
                QD_DIAG("%.2f%+.2fj,%.2f,%d,%.0f,%d,%.4f,%+.6f,%+.6f,%+.6f\n", std::real(eM), std::imag(eM), tm, tr, thd, (int)ev[0].ot, ang(ev[0].path_dirN),
                        ang(ev[1].path_dirN) - thd, (O.nT ? dB(O.T / W.t) : -999.0), dB(O.R / W.r));
                INFO("I, material " << m << ", tracer " << tr << ", theta " << thd);
                CHECK(O.nT >= 1);
                CHECK(std::abs(ang(ev[1].path_dirN) - thd) < 1.0e-4);
                if (O.nT >= 1)
                    CHECK(std::abs(dB(O.T / W.t)) < 5.0e-3);
                CHECK(std::abs(dB(O.R / W.r)) < 5.0e-3);
            }
        }

    // ---- J. A rigid wall that is given a loss ----
    // Rigid bodies of the acoustic mapping are lossless and closed from 0.003 deg. What a loss
    // column on such a body does to its tf leak, under whatever the verdict is.
    QD_DIAG("\n# diag: lossy verdict J, a rigid wall 200 mm with tf 0.3 in air, scalar, tracer 4 and 5: lossless and with a loss\n");
    QD_DIAG("# tangent of 0.04. Entry and exit in dB, the direction in the wall and\n");
    QD_DIAG("# behind it in deg, the reflectance of the first face\n");
    QD_DIAG("wall,tracer,theta,ot1,rt1,e1_dB,dir1,ot2,rt2,e2_dB,dir2,|R|^2\n");
    for (int wl : {4, 15})
        for (int tr : {4, 5})
            for (double thd : {0.0, 10.0, 45.0, 75.0})
            {
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.2}, {wl, wl}, {false, true}, mi);
                std::vector<Ev> ev = chain(mtl, mesh, mi, {tr, tr}, f_hz, thd, nullptr, 1.0e3);
                Ev er = chain(mtl, mesh, mi, {3}, f_hz, thd, nullptr, 1.0e3)[0];
                const bool two = ev.size() > 1 && ev[1].nh != 0;
                QD_DIAG("%s,%d,%.0f,%d,%d,%.4f,%.4f,%d,%d,%.4f,%.4f,%.6f\n", (wl == 4) ? "lossless" : ((wl == 15) ? "lossy" : ((wl == 16) ? "lossless m" : "lossy m")), tr, thd,
                        (int)ev[0].ot, (int)ev[0].rt, dB(cpx(ev[0].xpr, 0, 0)), ang(ev[0].path_dirN), two ? (int)ev[1].ot : -1, two ? (int)ev[1].rt : -1,
                        two ? dB(cpx(ev[1].xpr, 0, 0)) : -999.0, two ? ang(ev[1].path_dirN) : -1.0, std::norm(cpx(er.xpr, 0, 0)));
                INFO("J, wall " << wl << ", tracer " << tr << ", theta " << thd);
                if ((wl == 4 || wl == 16) && thd > 0.0) // lossless: closed from 0.003 deg, the flat leak, undeviated
                {
                    CHECK((int)ev[0].ot == 35);
                    CHECK(std::abs(std::norm(cpx(ev[0].xpr, 0, 0)) - 0.3) < 1.0e-9);
                    CHECK(std::abs(std::norm(cpx(er.xpr, 0, 0)) - 0.7) < 1.0e-9);
                    CHECK(std::abs(ang(ev[0].path_dirN) - thd) < 1.0e-9);
                }
            }

    // ---- G. EM: air into the porous medium, per channel ----
    auto fres = [&](std::complex<double> e1, std::complex<double> e2, double c1, std::complex<double> &rte, std::complex<double> &rtm)
    {
        std::complex<double> z1 = std::sqrt(e1), z2 = std::sqrt(e2), a = 1.0 - (e1 / e2) * (1.0 - c1 * c1);
        std::complex<double> c2 = std::sqrt(a);
        if (std::imag(c2) > 0.0)
            c2 = -c2;
        rte = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2), rtm = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2);
    };
    QD_DIAG("\n# diag: lossy verdict G, EM, air | porous solid, the entry per channel. Transmit power against 1 - |r|^2 and\n");
    QD_DIAG("# reflect power against |r|^2 of the half-space; k0, k3: the xprmat entries VV and HH\n");
    QD_DIAG("tracer,theta,ot,rt,T_k0_dB,T_k3_dB,ref_T_te_dB,ref_T_tm_dB,R_k0,R_k3,ref_R_te,ref_R_tm,dir,cur\n");
    for (int tr : {1, 2})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, c = std::cos(th);
            arma::uvec mi;
            arma::mat mesh = build({0.0, dS}, {2, 2}, {false, true}, mi);
            Ev et = chain(mtl, mesh, mi, {tr}, f_hz, thd, nullptr, 0.0)[0], er = chain(mtl, mesh, mi, {0}, f_hz, thd, nullptr, 0.0)[0];
            std::complex<double> rte, rtm;
            fres(one, eP, c, rte, rtm);
            QD_DIAG("%d,%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%04x\n", tr, thd, (int)et.ot, (int)et.rt, dB(cpx(et.xpr, 0, 0)), dB(cpx(et.xpr, 0, 3)),
                    10.0 * std::log10(1.0 - std::norm(rte)), 10.0 * std::log10(1.0 - std::norm(rtm)), std::norm(cpx(er.xpr, 0, 0)), std::norm(cpx(er.xpr, 0, 3)),
                    std::norm(rte), std::norm(rtm), ang(et.path_dirN), (unsigned)(unsigned short)et.cur(0));
            INFO("G, tracer " << tr << ", theta " << thd);
            CHECK((int)et.ot == 3);
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 0)) - (1.0 - std::norm(rte))) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 3)) - (1.0 - std::norm(rtm))) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(er.xpr, 0, 0)) - std::norm(rte)) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(er.xpr, 0, 3)) - std::norm(rtm)) < 1.0e-9);
            CHECK((unsigned)(unsigned short)et.cur(0) == ((tr == 2) ? 0x4002u : 0x0002u));
        }

    // ---- H. EM: a lossless dense solid in front of a lossy rarer one ----
    // The ray is launched inside the dense solid: across parallel faces from air the invariant
    // stays below one and the test never trips on this pair. No series is summed (eps 1e3), so
    // an event is its interface and the lossless leg it closes.
    auto launched = [&](double thd)
    {
        Ev S;
        double th = thd * QPI / 180.0;
        S.prev = sv({0}), S.cur = sv({9}), S.buf = sv({0});
        S.path_dirN = arma::mat{{std::cos(th), std::sin(th), 0.0}};
        S.accg = arma::vec{0.1}, S.accr = arma::vec{0.1};
        return S;
    };
    const std::vector<double> anglesH = {10.0, 20.0, tcA + 1.0, 30.0, tcH - 0.01, tcH + 0.01, 40.0, 50.0, 65.0, 80.0};
    QD_DIAG("\n# diag: lossy verdict H1, EM, eps 5.24 | eps 1.50 - 0.40j as two solids, the wave launched in the first at theta.\n");
    QD_DIAG("# The transition per channel against 1 - |r|^2 and |r|^2 of the half-space\n");
    QD_DIAG("tracer,theta,ot,rt,T_k0_dB,T_k3_dB,ref_T_te_dB,ref_T_tm_dB,R_k0,R_k3,ref_R_te,ref_R_tm,d0,dir,cur,prev\n");
    for (int tr : {1, 2})
        for (double thd : anglesH)
        {
            double th = thd * QPI / 180.0, c = std::cos(th);
            arma::uvec mi;
            arma::mat mesh = build({0.0, gap, 0.1 + gap}, {9, 10, 10}, {true, false, true}, mi);
            Ev S = launched(thd);
            Ev et = chain(mtl, mesh, mi, {tr}, f_hz, thd, &S, 1.0e3)[0], er = chain(mtl, mesh, mi, {0}, f_hz, thd, &S, 1.0e3)[0];
            REQUIRE((et.nh != 0 && er.nh != 0));
            std::complex<double> rte, rtm;
            fres(eH, eL, c, rte, rtm);
            QD_DIAG("%d,%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%04x,%04x\n", tr, thd, (int)et.ot, (int)et.rt, dB(cpx(et.xpr, 0, 0)), dB(cpx(et.xpr, 0, 3)),
                    10.0 * std::log10(1.0 - std::norm(rte)), 10.0 * std::log10(1.0 - std::norm(rtm)), std::norm(cpx(er.xpr, 0, 0)), std::norm(cpx(er.xpr, 0, 3)),
                    std::norm(rte), std::norm(rtm), ang(et.dir0), ang(et.path_dirN), (unsigned)(unsigned short)et.cur(0), (unsigned)(unsigned short)et.prev(0));
            INFO("H1, tracer " << tr << ", theta " << thd);
            CHECK((int)et.ot == 5);
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 0)) - (1.0 - std::norm(rte))) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 3)) - (1.0 - std::norm(rtm))) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(er.xpr, 0, 0)) - std::norm(rte)) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(er.xpr, 0, 3)) - std::norm(rtm)) < 1.0e-9);
            CHECK(std::abs(ang(et.path_dirN) - dir_in(eH, eL, thd)) < 1.0e-6);
            CHECK((unsigned)(unsigned short)et.cur(0) == ((tr == 2) ? 0x400au : 0x000au));
        }

    QD_DIAG("\n# diag: lossy verdict H2, EM, eps 5.24 | 6 mm eps 1.50 - 0.40j | air, a coated solid past the critical angle of its\n");
    QD_DIAG("# outer pair, the wave launched in the solid. The near face trips the magnitude test at %.4f deg. Transmit port per\n", tcH);
    QD_DIAG("# channel against the chain of an ordinary crossing with its series; reflect port per channel. tf: of the coating\n");
    QD_DIAG("tf,theta,ot,rt,T_k0_dB,T_k3_dB,ref_T_te_dB,ref_T_tm_dB,R_k0,R_k3\n");
    for (int sh : {11, 12})
        for (double thd : anglesH)
        {
            if (thd < tcA)
                continue;
            double th = thd * QPI / 180.0, c = std::cos(th);
            const LumpMtl &Sm = lp[(size_t)sh - 1];
            arma::uvec mi;
            arma::mat mesh = build({0.0, gap}, {9, sh}, {true, false}, mi);
            Ev S = launched(thd);
            Ev et = chain(mtl, mesh, mi, {1}, f_hz, thd, &S, 1.0e3)[0], er = chain(mtl, mesh, mi, {0}, f_hz, thd, &S, 1.0e3)[0];
            REQUIRE((et.nh != 0 && er.nh != 0));
            DiagLeak D = diag_leak_series(eH, eL, 1.0, 0.0, Sm.tf, 0.0, Sm.thk, c, f_hz, false);
            QD_DIAG("%.1f,%.2f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n", Sm.tf, thd, (int)et.ot, (int)et.rt, dB(cpx(et.xpr, 0, 0)), dB(cpx(et.xpr, 0, 3)), dB(D.T_te), dB(D.T_tm),
                    std::norm(cpx(er.xpr, 0, 0)), std::norm(cpx(er.xpr, 0, 3)));
            INFO("H2, tf " << Sm.tf << ", theta " << thd);
            CHECK((int)et.ot == 101); // eps 5.24 | air is a critical angle and stays one
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 0)) - std::norm(D.T_te)) < 1.0e-9);
            CHECK(std::abs(std::norm(cpx(et.xpr, 0, 3)) - std::norm(D.T_tm)) < 1.0e-9);
        }
}

TEST_CASE("Far from the origin - a thin dielectric slab transmits what it does at the origin, summed and followed", "[wave]")
{
    // ray_mesh_interact relaunches a ray 8 float ulp of the hit point's largest coordinate past it, about 1 um per
    // metre, and the leg inside a solid is measured from the hit point all the same: the ledger carries the part the
    // relaunch skips. A leg short by that offset would cost k0 (n - 1) times it in phase, through 4 mm of glass at 28
    // GHz 4.8 deg at 100 m and 48 deg at 1 km, and move the slab's series by up to 0.7 dB at 100 m. The slab here is
    // moved away from the origin with its ray, in its plane and along its normal, to 10 m, 100 m and 1 km: what it
    // transmits is the slab of its true thickness, TE and TM, both EM tracers, the series summed in one event (eps 0)
    // and every bounce followed (eps 1.5). itu_glass of data/materials_rf.csv at 28 GHz.
    ub::OneThread one_thread;
    const double f = 28.0e9, thk = 0.004, sig = 0.0036 * std::pow(28.0, 1.3394); // ITU-R P.2040: eps' 6.31, sigma = c f^d
    const std::complex<double> eS(6.31, -17.98 * sig / 28.0);
    ub::Pal P = lump_palette({{6.31, 17.98 * sig, 0.0}});
    auto dB = [](double g)
    { return 10.0 * std::log10(g); };
    QD_DIAG("\n# diag: 4 mm glass at 28 GHz moved away from the origin with its ray; T against the slab of its true thickness, dB\n");
    QD_DIAG("type,eps,theta,shift,along,dTE,dTM\n");
    double worst = 0.0;
    for (int tr : {1, 2})
        for (double eps : {0.0, 1.5})
            for (double thd : {0.0, 30.0, 60.0, 80.0})
                for (double R : {0.0, 10.0, 100.0, 1000.0})
                    for (int along : {0, 1})
                    {
                        if (R == 0.0 && along)
                            continue;
                        double p_te = 0.0, p_tm = 0.0;
                        diag_slab_p_pol(eS, thk, thd * QPI / 180.0, f, p_te, p_tm);
                        bf::Acc A = bf::walk_far(P, {0.0, thk}, {1, 1}, {false, true}, tr, f, thd, eps, along ? R : 0.0, along ? 0.0 : R);
                        const double dTE = dB(std::norm(A.T[0]) / p_te), dTM = dB(std::norm(A.T[1]) / p_tm);
                        worst = std::max({worst, std::abs(dTE), std::abs(dTM)});
                        QD_DIAG("%d,%.1f,%.0f,%.0f,%s,%+.6f,%+.6f\n", tr, eps, thd, R, along ? "normal" : "plane", dTE, dTM);
                        INFO("type " << tr << ", eps " << eps << ", " << thd << " deg, " << R << " m " << (along ? "along the normal" : "in the plane"));
                        REQUIRE(A.nT > 0);
                        CHECK(std::abs(dTE) < 1.0e-4);
                        CHECK(std::abs(dTM) < 1.0e-4);
                    }
    QD_DIAG("# largest |dT| %.2e dB\n", worst);
}
