// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_RAY=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[wave]"

// ray_state_update, wave: sheets, stacks and the lumped columns against their references.

#include "ray_test_helpers.hpp"

TEST_CASE("Diagnostic - Slab factor per polarization", "[wave]")
{
    // The traced solid's TE/TM ratio through a slab exit: the port's per-channel transmission times the series factor,
    // |T_X|^2 |S_X|^2, and nothing else. The ratio is exactly 1 at normal incidence, so the sweep makes the statement.
    // Printed beside it: the per-channel energy target (1 - R_X) |S_X|^2, whose gap to the asserted ratio is the TE/TM
    // obliquity asymmetry of a lossy interface in the sqrt(1 - R) amplitude convention; and ratio_pre, the port with a
    // second single-pass transmittance, which the assertion excludes.
    //
    // The probe field is symmetric, because the function rebuilds the coefficient from the two channel coefficients on
    // this path rather than scaling what it was handed. The slab is eps 1.50, whose critical angle is 54.7 deg, so the
    // sweep stays below total reflection.
    //
    // Channel identification: make1 puts orig, fbs and dest on the x axis with the normal along x, so the incidence
    // plane is degenerate, the polarization basis takes its identity branch, and VV is TM while HH is TE. That is the
    // opposite of the oblique fixture of the ray_mesh_interact cases.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SLAB = 5; // eps 1.50 - 0.40j

    QD_DIAG("\n# diag: slab factor per polarization, eps 1.50 - 0.40j, L = 100 mm, i-o exit\n");
    QD_DIAG("# critical angle asin(1/n) = %.2f deg; the sweep stays below it\n",
            std::asin(1.0 / std::real(std::sqrt(diag_eps(SLAB, f_hz)))) * 180.0 / QPI);
    QD_DIAG("theta_n,gain_out,VV2_tm,HH2_te,ratio_out,ratio_post,ratio_pre,ratio_ref,"
            "d_vs_post,d_vs_ref,rtype\n");

    for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_EXIT, 1u,
                               /*M1*/ SLAB, /*M2*/ 0, /*prev*/ 0, /*cur*/ SLAB, /*buf*/ 0,
                               /*d_orig_fbs*/ L, /*d_fbs_dest*/ 0.5, /*d_fbs_sbs*/ 0.5,
                               /*grazing angle*/ QPI / 2.0 - theta_n, /*eps: always resolve*/ 0.0);
        C.run();

        double vv2 = C.xprmat(0, 0) * C.xprmat(0, 0) + C.xprmat(1, 0) * C.xprmat(1, 0);
        double hh2 = C.xprmat(6, 0) * C.xprmat(6, 0) + C.xprmat(7, 0) * C.xprmat(7, 0);
        double lo = std::min(vv2, hh2), hi = std::max(vv2, hh2);
        double ratio_out = (hi > 0.0) ? lo / hi : 0.0;

        // Slab-side exit interface at the same angle the function used, per channel.
        std::complex<double> es = diag_eps(SLAB, f_hz), ea(1.0, 0.0);
        double ct = std::cos(theta_n);
        std::complex<double> z1 = std::sqrt(es), z2 = std::sqrt(ea);
        std::complex<double> c2 = std::sqrt(1.0 - (es / ea) * (1.0 - ct * ct));
        std::complex<double> r_te = (z1 * ct - z2 * c2) / (z1 * ct + z2 * c2);
        std::complex<double> r_tm = (z2 * ct - z1 * c2) / (z2 * ct + z1 * c2);
        std::complex<double> t_te = (2.0 * z1 * ct) / (z1 * ct + z2 * c2);
        std::complex<double> t_tm = (2.0 * z1 * ct) / (z2 * ct + z1 * c2);
        std::complex<double> phi = diag_phi(es, L, theta_n, f_hz);
        std::complex<double> S_te = 1.0 / (1.0 - r_te * r_te * phi * phi);
        std::complex<double> S_tm = 1.0 / (1.0 - r_tm * r_tm * phi * phi);
        double R_te = std::norm(r_te), R_tm = std::norm(r_tm);

        auto ratio_of = [](double a, double b)
        { double lo = std::min(a, b), hi = std::max(a, b); return (hi > 0.0) ? lo / hi : 0.0; };

        // Asserted: the port times the series, nothing else. Common factors drop out of a ratio.
        double ratio_post = ratio_of(std::norm(t_te) * std::norm(S_te),
                                     std::norm(t_tm) * std::norm(S_tm));
        // Printed: the same with the port's per-channel single-pass transmittance a second time; both boundaries of
        // this fixture are slab | air at the same angle, so that is (1 - R_X)^2.
        double ratio_pre = ratio_of(std::norm(t_te) * std::norm(S_te) * (1.0 - R_te) * (1.0 - R_te),
                                    std::norm(t_tm) * std::norm(S_tm) * (1.0 - R_tm) * (1.0 - R_tm));
        // Energy target, for the amplitude-convention residual.
        double ratio_ref = ratio_of((1.0 - R_te) * std::norm(S_te), (1.0 - R_tm) * std::norm(S_tm));

        QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u\n",
                thd, C.gain(0), vv2, hh2, ratio_out, ratio_post, ratio_pre, ratio_ref,
                ratio_out - ratio_post, ratio_out - ratio_ref,
                (unsigned)C.resolved_type[0]);

        CHECK(std::isfinite(C.gain(0)));
        CHECK(C.gain(0) >= 0.0);
        CHECK(std::abs(C.gain(0) - 0.5 * (vv2 + hh2)) < 1e-9);

        // The slab factor is S alone.
        INFO("theta_n " << thd << ", ratio_out " << ratio_out << ", ratio_post " << ratio_post);
        CHECK(std::abs(ratio_out - ratio_post) < 1e-9);
    }
}

TEST_CASE("Diagnostic - Entry boundary evaluated at two angles", "[wave]")
{
    // One slab crossing at the exact complex in-slab cosine. ray_mesh_interact applies the entry boundary at the true
    // incidence angle with the complex refraction cosine kept; ray_state_update recovers that cosine from the traced
    // direction, so the Airy denominator sums its series against the coefficient the path carries, not against one
    // re-derived from the real projection the tracer holds.
    //
    // The fixture hands the function what a tracer would produce: fbs_angle carries the geometric external angle,
    // path_dir_prev the direction qd_refract wrote, so the in-slab cosine the function sees is that direction's real
    // projection. Supplying path_dir_prev arms the recovery; without it nothing was projected and the gate declines.
    //
    // The series is isolated against a closed form. A run with air behind the slab over one with an electrically
    // identical twin behind it (far Fresnel coefficient zero, series factor one) would not isolate it: the twin makes
    // the entry and slab permittivities equal, one of the recovery's decline conditions, so the two runs would evaluate
    // the exit port at different cosines, up to 4.1 dB apart at 85 deg. With the port on the recovered cosine the whole
    // leg is analytic. The lossless control pins the reference: for a real permittivity the normalization inside
    // qd_refract is inert and both cosines coincide.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();

    QD_DIAG("\n# diag: entry boundary, recovered complex angle versus the real projection\n");
    QD_DIAG("theta_ext,ct_traced,|S_applied|,arg_S_applied,|S_exact|,arg_S_exact,|S_traced|,"
            "arg_S_traced,d_vs_exact_dB,d_traced_vs_exact_dB,rtype\n");

    for (int m : {5, 6}) // 5 = lossy eps 1.50 - 0.40j, 6 = lossless control
    {
        QD_DIAG("# material %d (%s)\n", m, (m == 5) ? "lossy eps 1.50 - 0.40j" : "lossless eps 1.50");
        for (double thd : {0.0, 20.0, 40.0, 60.0, 75.0})
        {
            std::complex<double> es = diag_eps(m, f_hz), ea(1.0, 0.0);
            double th_ext = thd * QPI / 180.0;
            double ct_tr = diag_traced_ct(es, th_ext);
            std::complex<double> ct_ex = diag_ct_exact(es, th_ext);

            auto C = make1<double>(mtl, /*scalar transmission*/ 4, OT_EXIT, 1u,
                                   /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                   L, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th_ext,
                                   /*always resolve*/ 0.0);
            // The refracted direction the tracer left behind: |dir . N| is the traced in-slab cosine
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = ct_tr;
            C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct_tr * ct_tr);
            C.run();

            std::complex<double> out = cpx(C.xprmat, 0, 0);
            std::complex<double> probe = cpx(C.x_in, 0, 0);
            unsigned rtype_seen = (unsigned)C.resolved_type[0];

            // Whether the recovery runs is decided by use_vbs, not by the angle: at normal incidence the traced and the
            // geometric direction coincide, nothing was projected, and the function declines. It then keeps both the
            // medium_gain propagation factor and the interface coefficient ray_mesh_interact wrote, so the probe field
            // survives into the output instead of being rebuilt; the reference follows it on both counts.
            bool recovery_fires = std::abs(ct_tr - std::cos(th_ext)) > 1e-6;

            double dr = L / ct_tr;
            double thk_n = dr * ct_tr; // normal thickness, pure geometry
            const double k0 = 2.0 * QPI * f_hz / C0;
            double n_re = std::real(std::sqrt(es));

            // The carrier: everything the exit leg applies apart from the series.
            std::complex<double> carrier, S_ex;
            if (recovery_fires)
            {
                std::complex<double> T_te, T_tm, S_te, S_tm;
                diag_port(es, ea, ct_ex, true, T_te, T_tm);
                diag_series(es, ea, ea, ct_ex, thk_n, f_hz, true, S_te, S_tm);
                carrier = T_te * diag_medium_vbs(es, dr, ct_tr, L, ct_ex, f_hz, std::cos(th_ext));
                S_ex = S_te;
            }
            else
            {
                std::complex<double> phi_tr =
                    std::sqrt(diag_medium_gain(es, dr, f_hz)) *
                    std::exp(std::complex<double>(0.0, -k0 * n_re * dr * ct_tr * ct_tr));
                std::complex<double> r_tr = diag_r_slabside(es, ea, std::complex<double>(ct_tr, 0.0));
                S_ex = 1.0 / (1.0 - r_tr * r_tr * phi_tr * phi_tr);
                carrier = probe * diag_medium_geo(es, dr, L, f_hz);
            }

            std::complex<double> S_applied = (std::abs(carrier) > 0.0)
                                                 ? out / carrier
                                                 : std::complex<double>(0.0, 0.0);

            // The reference the code would produce if both boundaries and the propagation factor
            // were built from the real projection instead of the recovered cosine. Printed only.
            std::complex<double> phi_tr =
                std::sqrt(diag_medium_gain(es, dr, f_hz)) *
                std::exp(std::complex<double>(0.0, -k0 * n_re * dr * ct_tr * ct_tr));
            std::complex<double> r_tr = diag_r_slabside(es, ea, std::complex<double>(ct_tr, 0.0));
            std::complex<double> S_tr = 1.0 / (1.0 - r_tr * r_tr * phi_tr * phi_tr);

            auto dB = [](std::complex<double> x, std::complex<double> y)
            { return 20.0 * std::log10(std::max(1e-300, std::abs(x)) / std::max(1e-300, std::abs(y))); };

            QD_DIAG("%.1f,%.9f,%.9f,%.3f,%.9f,%.3f,%.9f,%.3f,%.6f,%.6f,%u\n",
                    thd, ct_tr, std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                    std::abs(S_ex), std::arg(S_ex) * 180.0 / QPI,
                    std::abs(S_tr), std::arg(S_tr) * 180.0 / QPI,
                    dB(S_applied, S_ex), dB(S_tr, S_ex), rtype_seen);

            // The series is built from the recovered angle, not its projection.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(S_applied - S_ex) < 1e-9);

            // The lossless control pins the reference itself: there the normalization inside qd_refract is inert, so
            // both cosines coincide and the two references agree exactly. A gap here means diag_traced_ct no longer
            // matches the map qd_refract applies.
            if (m == 6)
                CHECK(std::abs(S_tr - S_ex) < 1e-12);
        }
    }
}

TEST_CASE("Diagnostic - Series collapse when the previous medium matches the slab", "[wave]")
{
    // The host layer's Airy denominator with a matched sheet as one of its bounding media: r = 0 there, the denominator
    // is exactly 1 and the internal series is gone. ray_state_update does not write a matched sheet into the
    // previous-medium word (it does not cut the layer there), so the state is set by hand; a sheet whose impedance is
    // merely close to the host's is resolved as a layer in the layered-stack cases.
    //
    // Material 7 is electrically identical to material 5 and differs only by carrying thk. The first row of each pair
    // is the ordinary case (previous medium = air, series present); the second replaces it with the matched sheet. |S|
    // and arg(S) stay put.
    const double f_hz = FRQ;
    auto mtl = diag_palette<double>();
    const int SLAB = 5, SHEET = 7;

    QD_DIAG("\n# diag: series collapse with a matched sheet as the previous medium\n");
    QD_DIAG("theta_n,L_mm,prev,|S_applied|,arg_S_applied,|S_ref_air|,arg_S_ref_air,d_dB,resolve_bit,rtype\n");

    for (double L : {0.05, 0.10})
        for (double thd : {0.0, 20.0, 40.0, 50.0})
            for (int prev : {0, SHEET})
            {
                double theta_n = thd * QPI / 180.0;
                auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                       /*M1*/ SLAB, /*M2*/ 0, /*prev*/ prev, /*cur*/ SLAB, /*buf*/ 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 0.0);
                C.run();

                std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
                std::complex<double> applied = (std::abs(in) > 0.0) ? out / in : std::complex<double>(0.0, 0.0);
                std::complex<double> es = diag_eps(SLAB, f_hz), ea(1.0, 0.0);
                std::complex<double> close = diag_close(es, L, theta_n, f_hz);
                std::complex<double> S_applied = (std::abs(close) > 0.0) ? applied / close
                                                                         : std::complex<double>(0.0, 0.0);

                std::complex<double> phi = diag_phi(es, L, theta_n, f_hz);
                std::complex<double> r = diag_r_slabside(es, ea, std::complex<double>(std::cos(theta_n), 0.0));
                std::complex<double> S_ref = 1.0 / (1.0 - r * r * phi * phi);
                double d_dB = 20.0 * std::log10(std::max(1e-300, std::abs(S_applied)) /
                                                std::max(1e-300, std::abs(S_ref)));

                QD_DIAG("%.1f,%.1f,%d,%.9f,%.3f,%.9f,%.3f,%.6f,%d,%u\n",
                        thd, L * 1000.0, prev, std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                        std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI, d_dB,
                        (int)rt_bit(C.resolved_type[0], 2), (unsigned)C.resolved_type[0]);

                CHECK(std::isfinite(C.gain(0)));
                CHECK(C.gain(0) >= 0.0);
            }
}

TEST_CASE("Diagnostic - Host layer carried through a matched sheet", "[wave]")
{
    // A sheet crossing where the sheet is electrically identical to the host it sits in. The sheet is not a boundary,
    // so the layer is not cut there: the accumulated distance keeps running, the medium words stay put, and the series
    // resolves once at the far side over the whole path.
    //
    // Materials 5 and 7 differ only in thk, the column that does not affect a boundary: two media with the same eps and
    // mu.
    //
    // The control is material 3, whose permittivity differs from the host's: it splits the layer. It has to be a sheet
    // (thk > 0), or hit code 67 does not re-derive as a sheet and the event kills the ray.
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const uint8_t OT_SHEET_FRONT = 67; // isolated sheet, front-side hit
    const int HOST = 5, MATCHED = 7, MISMATCHED = 3;

    QD_DIAG("\n# diag: host layer carried through a matched sheet\n");
    QD_DIAG("sheet_mtl,theta_n,prev_out,cur_out,acc_geo_out,acc_ref_out,gain_out,merged,rtype\n");

    for (int sheet_mtl : {MATCHED, MISMATCHED})
        for (double thd : {0.0, 30.0})
        {
            double theta_n = thd * QPI / 180.0;
            auto C = make1<double>(mtl, 4, OT_SHEET_FRONT, 1u,
                                   /*M1 = the sheet face*/ sheet_mtl, /*M2*/ 0,
                                   /*prev*/ 0, /*cur = host*/ HOST, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 0.0);
            C.has_acc_geo_out = true;
            C.has_acc_ref_out = true;
            C.run();

            bool merged = (mat_of(C.prev_out(0)) == 0) && ((double)C.acc_geo_out(0) > 0.5 * L);
            QD_DIAG("%d,%.1f,%d,%d,%.6f,%.6f,%.9f,%d,%u\n",
                    sheet_mtl, thd, mat_of(C.prev_out(0)), mat_of(C.cur_out(0)),
                    (double)C.acc_geo_out(0), (double)C.acc_ref_out(0), (double)C.gain(0),
                    (int)merged, (unsigned)C.resolved_type[0]);

            CHECK(std::isfinite(C.gain(0)));
            CHECK(C.gain(0) >= 0.0);

            INFO("sheet material " << sheet_mtl << ", theta_n " << thd);
            if (sheet_mtl == MATCHED)
            {
                // The layer continues: nothing about the medium state may change, and the distance
                // accumulated up to the sheet has to survive rather than being reset. The ray goes on a
                // relaunch offset past the sheet, which the ledger carries from here on.
                CHECK(mat_of(C.prev_out(0)) == 0);
                CHECK(mat_of(C.cur_out(0)) == HOST);
                CHECK(std::abs((double)C.acc_geo_out(0) - (L + qd_relaunch_nudge(L))) < 1e-12);

                // And the crossing itself is a no-op. The sheet occupies thk of the medium it sits
                // in, so a sheet made of that same medium replaces it with itself: both interfaces
                // are transparent, the series is empty, and the propagation factor cancels against
                // the displacement of an identical medium over the same thickness. Gain 1, exactly.
                CHECK(std::abs((double)C.gain(0) - 1.0) < 1e-9);
            }
            else
            {
                // A real boundary still ends the layer and records the sheet as the medium behind.
                CHECK(mat_of(C.prev_out(0)) == MISMATCHED);
                CHECK(mat_of(C.cur_out(0)) == HOST);
            }
        }
}

TEST_CASE("Diagnostic - Excess-phase walk-off convention", "[wave]")
{
    // The in-medium factor on the walk-off branch is k0 (n dist_refract cos2 - dist_geo cos2_geo), where the cos^2
    // factors are present only when the tracer did not trace the bend and are 1 when it did. A thin sheet is never
    // displaced and carries the cos^2 form, so the two describe different geometries and their phases differ.
    //
    // Both terms sit on the thickness axis. With d_t = dist_refract ct_t and d_i = dist_geo ct_i the applied argument
    // is -k0 (n d_t ct_t - d_i ct_i), the tilted-slab excess phase k0 d (n cos_t - cos_i) referenced to the emergent
    // wavefront rather than to the ray path. An undeviated tracer holds no cos_t, so cos_i stands in and the expression
    // is k0 d cos_i (n - 1) = k0 L cos_i^2 (n - 1): the cos^2 stands for the untraced lateral shift.
    //
    // The case isolates that factor. With no path_dir the exact-cosine path is not taken and no coefficient is written;
    // eps = 1.5 exceeds any reachable rho, so no series is summed: out / in is the medium factor alone. The lossless
    // material keeps the in-slab angle real and medium_gain at 1. Without path_dir, dist_refract equals dist_geo and
    // ct_t equals ct_i, so the reference is called in its general form with those equalities passed in as arguments:
    // the row states what it rests on.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SLAB = 6; // lossless eps 1.50

    QD_DIAG("\n# diag: excess-phase walk-off convention, lossless eps 1.50, L = 100 mm\n");
    QD_DIAG("theta_n,arg_applied,arg_cos2_fbs,arg_raw_path,d_vs_cos2_fbs,d_vs_raw\n");

    for (double thd : {0.0, 20.0, 40.0, 60.0, 70.0, 80.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                               /*M1*/ SLAB, /*M2*/ 0, /*prev*/ 0, /*cur*/ SLAB, /*buf*/ 0,
                               L, 0.5, 0.5, QPI / 2.0 - theta_n, /*eps: never resolve*/ 1.5);
        C.run();

        std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
        std::complex<double> applied = (std::abs(in) > 0.0) ? out / in : std::complex<double>(0.0, 0.0);

        double n_re = std::real(std::sqrt(diag_eps(SLAB, f_hz)));
        double k0 = 2.0 * QPI * f_hz / C0;
        double ct = std::cos(theta_n);
        double phase_cos2 = -k0 * (n_re - 1.0) * L * ct * ct; // walk-off form, fbs cosine
        double phase_raw = -k0 * (n_re - 1.0) * L;            // raw path length, no walk-off

        auto wrap = [](double d)
        {
            while (d > 180.0)
                d -= 360.0;
            while (d <= -180.0)
                d += 360.0;
            return d;
        };
        double a = std::arg(applied) * 180.0 / QPI;
        double c2 = phase_cos2 * 180.0 / QPI, rw = phase_raw * 180.0 / QPI;

        QD_DIAG("%.1f,%.3f,%.3f,%.3f,%.3f,%.3f\n", thd, wrap(a), wrap(c2), wrap(rw),
                wrap(a - c2), wrap(a - rw));

        CHECK(std::isfinite(C.gain(0)));
        CHECK(C.gain(0) >= 0.0);

        // The applied factor against the general walk-off traversal form, with the fixture's equalities (ct_t = ct_i =
        // ct, dist_refract = dist_geo = L) passed as arguments. The magnitude is asserted with the phase: the palette
        // carries no alpha and the material is lossless, so medium_gain is 1, and an interface coefficient or a summed
        // series reaching this path would show.
        std::complex<double> ref = diag_medium_walk(diag_eps(SLAB, f_hz), L, ct, L, ct, f_hz);

        INFO("angle " << thd);
        CHECK(std::abs(wrap(a - std::arg(ref) * 180.0 / QPI)) < 1.0e-2);
        CHECK(std::abs(std::abs(applied) - std::abs(ref)) < 1.0e-12);

        // The raw-path convention, the other column the table prints, is excluded by a margin that grows with the
        // angle: the row tells the two apart. They coincide at normal incidence, so the margin is claimed only past it.
        if (thd > 0.0)
            CHECK(std::abs(wrap(a - rw)) > 1.0);
    }
}

TEST_CASE("Thin sheet - the redo sits on the analytic slab at grazing incidence", "[wave]")
{
    // ray_state_update re-resolves every sheet crossing when medium tracking is on, so in a run with state its
    // composite is the one that reaches the output. It has no mesh: nothing about the face reaches it. The composite is
    // the analytic slab all the way to tangency.
    //
    // The sheet is isolated in air, so mtl_current is 0: no host leg is closed and no medium factor applied, and the
    // gain written is the composite itself. It is written, not scaled (the coefficient is rebuilt with unit amplitude),
    // so the incoming probe field sets the polarization basis and not the level.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;               // eps 5.24 - 0.63j, thk 6 mm
    const uint8_t OT_SHEET_FRONT = 67; // isolated sheet, front-side hit
    const std::complex<double> eps_s = diag_eps(SHEET, f_hz);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: sheet redo at grazing incidence, thk 6 mm, eps 5.24 - 0.63j\n");
    QD_DIAG("angle,gain_out,gain_dB,ana_dB,d_dB,rtype\n");

    for (double thd : {80.0, 85.0, 88.0, 89.5, 89.9, 89.94, 89.96, 89.99, 89.999})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_FRONT, 1u,
                               /*M1 = the sheet face*/ SHEET, /*M2*/ 0,
                               /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                               /*d_orig_fbs*/ 1.0, /*d_fbs_dest*/ 0.5, /*d_fbs_sbs*/ 0.5,
                               QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
        C.run();

        double g = (double)C.gain(0);
        double ana = diag_slab_p_avg(eps_s, thk, theta_n, f_hz);

        QD_DIAG("%.4f,%.9e,%.6f,%.6f,%.6f,%u\n",
                thd, g, dB(g), dB(ana), dB(g) - dB(ana), (unsigned)C.resolved_type[0]);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(g));
        CHECK(g >= 0.0);
        CHECK(std::abs(dB(g) - dB(ana)) < 1.0e-6);
    }
}

TEST_CASE("Glued sheet - the near-to-far flux conversion", "[wave]")
{
    // The composite applies the near-to-far obliquity factor once, on the transmit port, so that a glued sheet reports
    // the same energy as the equivalent traced entry. The factor is gated on near != far, so only a glued entry (air |
    // sheet | solid) and a glued exit (solid | sheet | air) take it.
    //
    // Six stacks: two on the trivial branch and four on the other. The embedded row pins the gate as near == far rather
    // than as "far is air": a sheet inside a medium, where the factor is exactly 1.
    //
    // Isolation. The composite is what is measured, so nothing else may multiply it.
    //   - Entry rows run with mtl_current = 0, so no host factor is applied and the written field is the composite;
    //     they are asserted complex.
    //   - Rows inside a medium close out a host leg. eps = 1.5 puts the survival gate above any reachable rho, so no
    //     series is summed, and the leg contributes sqrt(medium_gain) with a phase. These rows are asserted per channel
    //     in power, where the phase drops out: fx is a real positive scale, and the composite phase is pinned by the
    //     air | sheet | air cases.
    //
    // Channel identification: make1 puts orig, fbs and dest on the x axis with the normal along x, so the incidence
    // plane is degenerate, the polarization basis takes its identity branch, and VV is TM while HH is TE, for the
    // reflected direction as well as the transmitted one.
    //
    // The exit sweeps stop at 50 deg: the internal critical angle of eps 1.50 against air is 54.7 deg, and past it the
    // slab-side TM reflection of a lossy sheet exceeds one, a question of its own.
    const double f_hz = FRQ;
    const double thk = 0.006; // the sheet material's own thickness column
    const double L = 0.10;    // host path up to the sheet, only read on the rows inside a medium
    auto mtl = diag_palette<double>();

    const uint8_t OT_SHEET_ISO = 67;   // ok | sheet, no co-located face
    const uint8_t OT_GLUED_ENTRY = 71; // ok | front | co-located | sheet -> a glued entry
    const uint8_t OT_GLUED_EXIT = 69;  // ok | co-located | sheet      -> a glued exit
    const int SHEET = 3, LOSSY = 1, LO = 5, LO_CLEAN = 6;

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2;     // the co-located non-sheet face, 0 when there is none
        int cur;    // medium the ray is travelling in
        int e_near; // palette index of the near medium (0 = air)
        int e_far;  // palette index of the far medium
        bool disp_far;
        double lim;   // highest angle swept
        bool trivial; // near == far, so the factor must be exactly 1
    };
    const std::vector<Row> rows = {
        {"isolated   air | sheet | air", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 85.0, true},
        {"embedded   LO_CLEAN | sheet | LO_CLEAN", OT_SHEET_ISO, 1u, 0, LO_CLEAN, LO_CLEAN, LO_CLEAN, false, 50.0, true},
        {"entry      air | sheet | LOSSY", OT_GLUED_ENTRY, 2u, LOSSY, 0, 0, LOSSY, false, 85.0, false},
        {"entry      air | sheet | LO", OT_GLUED_ENTRY, 2u, LO, 0, 0, LO, false, 85.0, false},
        {"exit       LO_CLEAN | sheet | air", OT_GLUED_EXIT, 2u, LO_CLEAN, LO_CLEAN, LO_CLEAN, 0, true, 50.0, false},
        {"exit       LO | sheet | air", OT_GLUED_EXIT, 2u, LO, LO, LO, 0, true, 50.0, false},
    };

    QD_DIAG("\n# diag: glued-sheet near-to-far flux conversion, sheet eps 5.24 - 0.63j, thk 6 mm\n");
    QD_DIAG("# fx_dB is what the composite carries against one built with the factor omitted\n");
    QD_DIAG("stack,theta_n,fx_te,fx_tm,fx_te_dB,fx_tm_dB,P_te_meas,P_te_ref,P_tm_meas,P_tm_ref,"
            "d_te,d_tm,rtype\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto C = make1<double>(mtl, /*EM transmission*/ 1, R.otype, R.nH,
                                   /*M1 = the sheet face*/ SHEET, /*M2*/ R.m2,
                                   /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n,
                                   /*eps: the host series must not fire*/ 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(diag_eps(R.e_near, f_hz), diag_eps(SHEET, f_hz),
                                               diag_eps(R.e_far, f_hz), thk, ct, f_hz,
                                               R.disp_far, false);

            // The host leg the crossing closes out. Lossless hosts contribute exactly 1 in magnitude; the lossy one
            // contributes the ITU-R P.2040 path form over L, with which the leg is closed when no path_dir is supplied.
            double host_g = (R.cur == 0) ? 1.0 : diag_medium_gain(diag_eps(R.cur, f_hz), L, f_hz);

            double vv2 = C.xprmat(0, 0) * C.xprmat(0, 0) + C.xprmat(1, 0) * C.xprmat(1, 0); // TM
            double hh2 = C.xprmat(6, 0) * C.xprmat(6, 0) + C.xprmat(7, 0) * C.xprmat(7, 0); // TE
            double P_te = std::norm(D.T_te) * host_g, P_tm = std::norm(D.T_tm) * host_g;

            QD_DIAG("%s,%.1f,%.6f,%.6f,%+.4f,%+.4f,%.9e,%.9e,%.9e,%.9e,%.2e,%.2e,%u\n",
                    R.name, thd, D.fx_te, D.fx_tm,
                    20.0 * std::log10(D.fx_te > 0.0 ? D.fx_te : 1.0),
                    20.0 * std::log10(D.fx_tm > 0.0 ? D.fx_tm : 1.0),
                    hh2, P_te, vv2, P_tm, hh2 - P_te, vv2 - P_tm,
                    (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0); // the ray survives: a hit code that fails the sheet
                                              // re-derivation would kill it

            // The composite, per channel, against the hand-built stack.
            CHECK(std::abs(hh2 - P_te) < 1.0e-9);
            CHECK(std::abs(vv2 - P_tm) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (P_te + P_tm)) < 1.0e-9);

            // Where nothing else multiplies the composite, pin the phase too.
            if (R.cur == 0)
            {
                CHECK(std::abs(cpx(C.xprmat, 0, 0) - D.T_tm) < 1.0e-9);
                CHECK(std::abs(cpx(C.xprmat, 0, 3) - D.T_te) < 1.0e-9);
            }

            if (R.trivial)
            {
                // near == far: the factor is identically 1, not merely close to it. This is what
                // separates the two branches, and it holds for a sheet embedded in a medium as well
                // as for one in air.
                CHECK(D.fx_te == 1.0);
                CHECK(D.fx_tm == 1.0);
            }
            else
            {
                // And the non-trivial branch is worth enough that the assertion above cannot pass
                // with the factor removed. It is 3.6 dB at normal incidence for the lossy entry, so
                // this does not rely on grazing angles.
                CHECK(std::abs(20.0 * std::log10(D.fx_te)) > 0.5);
                CHECK(std::abs(hh2 - P_te / (D.fx_te * D.fx_te)) > 1.0e-6);
                CHECK(std::abs(vv2 - P_tm / (D.fx_tm * D.fx_tm)) > 1.0e-6);
            }
        }

    // The reflect port carries no flux conversion, because a reflection turns around in the near
    // medium where the obliquity ratio is 1. Measured on the entry rows only, where mtl_current is 0
    // and the reflection pass applies nothing beyond the composite itself.
    QD_DIAG("\n# diag: glued-sheet reflect port, which carries no flux conversion\n");
    QD_DIAG("stack,theta_n,R_te_meas,R_te_ref,R_tm_meas,R_tm_ref,d_te,d_tm,rtype\n");

    for (const auto &R : rows)
    {
        if (R.cur != 0)
            continue;
        for (double thd : {0.0, 20.0, 40.0, 60.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto C = make1<double>(mtl, /*EM reflection*/ 0, R.otype, R.nH,
                                   SHEET, R.m2, /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(diag_eps(R.e_near, f_hz), diag_eps(SHEET, f_hz),
                                               diag_eps(R.e_far, f_hz), thk, ct, f_hz,
                                               R.disp_far, false);

            std::complex<double> vv = cpx(C.xprmat, 0, 0), hh = cpx(C.xprmat, 0, 3);
            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%u\n",
                    R.name, thd, std::abs(hh), std::abs(D.R_te), std::abs(vv), std::abs(D.R_tm),
                    std::abs(hh - D.R_te), std::abs(vv - D.R_tm), (unsigned)C.resolved_type[0]);

            INFO(R.name << " reflect, theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            CHECK(std::abs(hh - D.R_te) < 1.0e-9);
            CHECK(std::abs(vv - D.R_tm) < 1.0e-9);
        }
    }
}

TEST_CASE("Sheet in a medium - the composite at the exact in-medium cosine", "[wave]")
{
    // A sheet met inside a traced solid: the incidence cosine there is complex, and the tracer holds only its real
    // projection. The composite is built at the exact in-medium cosine, and the case asserts it on both ports against a
    // reference at that cosine.
    //
    // What separates it from the glued-sheet case: there no path_dir is supplied, so the cosine handed to the composite
    // is a real incidence cosine. Here path_dir_prev carries the refracted direction a tracer would have written, so
    // the projection and the exact cosine separate.
    //
    // Geometry, as in the exit-boundary case: a ray crosses air | LO at the external angle, the tracer writes the
    // refracted direction, and the sheet is met after 100 mm of normal thickness. fbs_angle stays the external angle
    // because the geometric ray was never bent.
    //
    // Two stacks:
    //   embedded    LO | sheet | LO,  an isolated sheet, near == far so the flux conversion is 1
    //   glued exit  LO | sheet | air, near != far so the conversion is live
    // and both ports of each.
    //
    // Isolation. eps = 1.5 puts the survival gate above any reachable rho, so no series is summed and the only thing
    // multiplying the composite is the host leg: the exact form on the transmission pass, where the recovery is armed,
    // and the walk-off form on the reflection pass, where it is not.
    //
    // Modes. The scalar run pins the complex coefficient outright. The EM run pins the gain, the Frobenius norm of the
    // Jones matrix, which is invariant under the polarization basis; the refracted direction has a transverse
    // component, so the basis is not the degenerate one of the glued-sheet case.
    const double f_hz = FRQ;
    const double thk = 0.006; // the sheet material's own thickness column
    const double L = 0.10;    // normal thickness of the host layer up to the sheet
    auto mtl = diag_palette<double>();

    const uint8_t OT_SHEET_ISO = 67;  // ok | sheet, no co-located face   -> an isolated sheet
    const uint8_t OT_GLUED_EXIT = 69; // ok | co-located | sheet          -> a glued exit
    const int SHEET = 3, LO = 5;

    const std::complex<double> e_h = diag_eps(LO, f_hz);    // host, eps 1.50 - 0.40j
    const std::complex<double> e_s = diag_eps(SHEET, f_hz); // sheet, eps 5.24 - 0.63j
    const std::complex<double> e_air(1.0, 0.0);

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2; // the co-located non-sheet face, 0 when there is none
        std::complex<double> e_far;
        bool disp_far; // the sheet sits on the side that is not the solid
        bool trivial;  // near == far, so the flux conversion is identically 1
    };
    const std::vector<Row> rows = {
        {"embedded   LO | sheet | LO", OT_SHEET_ISO, 1u, 0, e_h, false, true},
        {"glued exit LO | sheet | air", OT_GLUED_EXIT, 2u, LO, e_air, true, false},
    };

    QD_DIAG("\n# diag: sheet reached from inside a traced solid, host eps 1.50 - 0.40j over 100 mm,\n");
    QD_DIAG("# sheet eps 5.24 - 0.63j, thk 6 mm, 1 GHz. The composite is built at ct_traced; the\n");
    QD_DIAG("# reference is the same composite at the exact complex in-medium cosine.\n");
    QD_DIAG("# A is the composite amplitude moving with the cosine, B is the flux conversion failing\n");
    QD_DIAG("# to cancel against the entry that produced it, and A + B is the total on the TE port.\n");
    QD_DIAG("stack,theta_ext,ct_traced,ct_exact_re,ct_exact_im,A_te_dB,B_te_dB,tot_te_dB,tot_tm_dB,"
            "R_te_dB,R_tm_dB,d_T_scalar,d_T_gain,d_R_scalar,d_R_gain,d_host_T,d_host_R,rtype_T,rtype_R\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(e_h, th);              // the real projection the tracer holds
            std::complex<double> cx = diag_ct_exact(e_h, th); // the exact complex in-medium cosine
            double dr = L / ct;                               // refracted path up to the sheet

            // The gate use_vbs applies. At normal incidence the projection and the exact value coincide, the recovery
            // declines and the traversal keeps its walk-off form: that row is a control.
            bool armed = std::abs(ct - std::cos(th)) > 1.0e-6;

            std::complex<double> Mt = armed ? diag_medium_vbs(e_h, dr, ct, L, cx, f_hz, std::cos(th))
                                            : diag_medium_geo(e_h, L, L, f_hz);
            // The reflect pass closes a bent leg with the same form; the walk-off form is left
            // to a wave that was not bent, which here is the normal-incidence control
            std::complex<double> Mr = armed ? Mt : diag_medium_walk(e_h, dr, ct, L, std::cos(th), f_hz);

            // The reference, at the exact cosine, and the same composite at the real projection,
            // printed beside it as the size of what the exact cosine moves; it is never the target.
            DiagSheet Xs = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cx, f_hz, R.disp_far, true);
            DiagSheet Xe = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cx, f_hz, R.disp_far, false);
            std::complex<double> cp(ct, 0.0);
            DiagSheet Pe = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cp, f_hz, R.disp_far, false);

            auto run = [&](int itype, double host_len)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH,
                                       /*M1 = the sheet face*/ SHEET, /*M2*/ R.m2,
                                       /*prev = what the host was entered through*/ 0,
                                       /*cur*/ LO, /*buf*/ 0,
                                       host_len, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th,
                                       /*eps: the host series must not fire*/ 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };

            auto Ct = run(4, L); // scalar transmission
            auto Ce = run(1, L); // EM transmission
            auto Cr = run(3, L); // scalar reflection
            auto Cx = run(0, L); // EM reflection

            // The same two crossings over twice the host path. Only the traversal changes: the
            // composite is a function of the cosine and the materials and is identical on both, so
            // it cancels out of the ratio below.
            auto Ct2 = run(4, 2.0 * L);
            auto Cr2 = run(3, 2.0 * L);

            std::complex<double> refTs = Xs.T_te * Mt, refRs = Xs.R_te * Mr;
            double refTg = 0.5 * (std::norm(Xe.T_te * Mt) + std::norm(Xe.T_tm * Mt));
            double refRg = 0.5 * (std::norm(Xe.R_te * Mr) + std::norm(Xe.R_tm * Mr));

            double d_Ts = std::abs(cpx(Ct.xprmat, 0, 0) - refTs);
            double d_Tg = std::abs((double)Ce.gain(0) - refTg);
            double d_Rs = std::abs(cpx(Cr.xprmat, 0, 0) - refRs);
            double d_Rg = std::abs((double)Cx.gain(0) - refRg);

            // The host leg on its own, isolated from the composite: two runs that differ only in host path length, so
            // the composite (a function of the cosine and the materials) cancels in the ratio, and both traversal forms
            // are exponentials in the path length, so the ratio over L and 2L is the factor over L. diag_medium_walk is
            // asserted nowhere else, so the reflect pass gets its own row.
            double d_hT = std::abs(cpx(Ct2.xprmat, 0, 0) / cpx(Ct.xprmat, 0, 0) - Mt);
            double d_hR = std::abs(cpx(Cr2.xprmat, 0, 0) / cpx(Cr.xprmat, 0, 0) - Mr);

            auto dB = [](std::complex<double> a, std::complex<double> b)
            { return 20.0 * std::log10(std::max(1e-300, std::abs(a)) / std::max(1e-300, std::abs(b))); };

            double B_te = (Xe.fx_te > 0.0 && Pe.fx_te > 0.0) ? 20.0 * std::log10(Pe.fx_te / Xe.fx_te) : 0.0;
            double tot_te = dB(Pe.T_te, Xe.T_te), tot_tm = dB(Pe.T_tm, Xe.T_tm);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,"
                    "%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%u,%u\n",
                    R.name, thd, ct, std::real(cx), std::imag(cx),
                    tot_te - B_te, B_te, tot_te, tot_tm,
                    dB(Pe.R_te, Xe.R_te), dB(Pe.R_tm, Xe.R_tm),
                    d_Ts, d_Tg, d_Rs, d_Rg, d_hT, d_hR,
                    (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO(R.name << ", theta_ext " << thd);

            // The ray survives: a hit code that fails the sheet re-derivation would kill it.
            REQUIRE(Ct.resolved_type[0] != 0);
            REQUIRE(Cr.resolved_type[0] != 0);
            REQUIRE(std::isfinite(Ct.gain(0)));
            REQUIRE(std::isfinite(Cr.gain(0)));

            // The complex builder reproduces the real-cosine one at a real cosine, term for term.
            DiagSheet Pd = diag_sheet_composite(e_h, e_s, R.e_far, thk, ct, f_hz, R.disp_far, false);
            CHECK(std::abs(Pd.T_te - Pe.T_te) < 1.0e-15);
            CHECK(std::abs(Pd.T_tm - Pe.T_tm) < 1.0e-15);
            CHECK(std::abs(Pd.R_te - Pe.R_te) < 1.0e-15);
            CHECK(std::abs(Pd.R_tm - Pe.R_tm) < 1.0e-15);

            // Two conditions the measurement depends on, asserted so that a failure of either shows as itself. The
            // slab-side TM reflection stays below one: the sweep stays below the internal critical angle of the near |
            // far pair, the traced angle reaching only 51.7 deg at 85 deg of external incidence.
            std::complex<double> zs = std::sqrt(e_s), zf = std::sqrt(R.e_far);
            std::complex<double> r_slab_p = (zf * Pe.c_slab - zs * Pe.c_far) / (zf * Pe.c_slab + zs * Pe.c_far);
            std::complex<double> r_slab_x = (zf * Xe.c_slab - zs * Xe.c_far) / (zf * Xe.c_slab + zs * Xe.c_far);
            CHECK(std::abs(r_slab_p) < 1.0);
            CHECK(std::abs(r_slab_x) < 1.0);

            // The displacement budget must not cap, so the reference may divide out the full
            // thickness. Air costs nothing to displace and is uncapped, so only the embedded stack
            // needs it: the budget is 80 percent of the host path the crossing can see.
            if (!R.disp_far)
                CHECK(0.8 * (dr + 0.5) * ct > thk);

            // The two cosines differ, so the case can fail.
            if (thd > 0.0)
                CHECK(std::abs(cx - cp) > 1.0e-6);

            // The flux conversion is identically 1 wherever near == far, at a complex cosine as well
            // as at a real one. The gate is near == far, not "far is air".
            if (R.trivial)
            {
                CHECK(Xe.fx_te == 1.0);
                CHECK(Xe.fx_tm == 1.0);
            }

            // The host leg the crossing closes out, on its own and first: independent of the composite by construction,
            // so a failure here is a statement about the traversal reference, not about the incidence cosine.
            CHECK(d_hT < 1.0e-12);
            CHECK(d_hR < 1.0e-12);

            // The two references are far enough apart that the composite assertions below cannot pass at both cosines.
            if (thd >= 20.0)
            {
                CHECK(std::abs(Pe.T_te - Xe.T_te) > 1.0e-6);
                CHECK(std::abs(Pe.R_tm - Xe.R_tm) > 1.0e-6);
            }

            // The composite at the exact in-medium cosine, both ports, both modes.
            CHECK(d_Ts < 1.0e-9);
            CHECK(d_Tg < 1.0e-9);
            CHECK(d_Rs < 1.0e-9);
            CHECK(d_Rg < 1.0e-9);
        }
}

TEST_CASE("Exit boundary - the port at the recovered angle", "[wave]")
{
    // The far face of a traced solid against a reference at the exact recovered cosine, not at the real projection the
    // tracer holds. The whole exit leg is asserted: ray_state_update writes T_exit S M, the port, the series factor and
    // the in-medium traversal, all three reproduced from closed forms that share no code with the library. The port
    // alone is printed as out / (S M).
    //
    // Nothing divides one run by another: the exact-cosine path rebuilds the field from the interface coefficient, so
    // the written value is absolute and the probe cancels out.
    //
    // Two modes. The scalar run pins the complex coefficient outright. The EM run pins the gain, the Frobenius norm of
    // the Jones matrix, which is invariant under the polarization basis, so it measures the TE and TM ports without
    // depending on which element each lands in.
    const double f_hz = FRQ;
    const double L = 0.10; // normal thickness of the traversed layer
    auto mtl = diag_palette<double>();

    QD_DIAG("\n# diag: exit port at the recovered complex cosine, i-o exit, normal thickness 100 mm\n");
    QD_DIAG("# port_ref is the exit interface at the exact angle; port_traced is the same interface\n");
    QD_DIAG("# at the real projection of the traced cosine, the control the exact port is told apart from\n");
    QD_DIAG("mtl,theta_ext,ct_traced,|port_meas|,|port_ref|,|port_traced|,d_port_dB,"
            "d_amp,d_gain_em,rtype\n");

    for (int m : {5, 1, 4, 6})
    {
        QD_DIAG("# material %d, eps %.2f %+.2fj\n", m,
                std::real(diag_eps(m, f_hz)), std::imag(diag_eps(m, f_hz)));
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
        {
            std::complex<double> es = diag_eps(m, f_hz), ea(1.0, 0.0);
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(es, th);
            std::complex<double> cs = diag_ct_exact(es, th);
            double dr = L / ct; // refracted path the tracer walked

            // Every angle in this sweep bends the direction, so the recovery is armed on all of
            // them. Normal incidence is excluded deliberately: there is nothing projected there and
            // the exit keeps the geometric path, which the entry-boundary case covers instead.
            REQUIRE(std::abs(ct - std::cos(th)) > 1e-6);

            std::complex<double> Ts_te, Ts_tm, Te_te, Te_tm;
            diag_port(es, ea, cs, true, Ts_te, Ts_tm);  // scalar port at the exact cosine
            diag_port(es, ea, cs, false, Te_te, Te_tm); // EM port, per channel

            std::complex<double> Ss_te, Ss_tm, Se_te, Se_tm;
            diag_series(es, ea, ea, cs, L, f_hz, true, Ss_te, Ss_tm);
            diag_series(es, ea, ea, cs, L, f_hz, false, Se_te, Se_tm);

            std::complex<double> M = diag_medium_vbs(es, dr, ct, L, cs, f_hz, std::cos(th));

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_EXIT, 1u,
                                       /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                       L, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th,
                                       /*always resolve*/ 0.0);
                // The refracted direction the tracer left behind
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };

            auto Cs = run(4); // scalar transmission
            auto Ce = run(1); // EM transmission

            std::complex<double> out = cpx(Cs.xprmat, 0, 0);
            std::complex<double> ref = Ts_te * Ss_te * M;
            std::complex<double> carrier = Ss_te * M;
            std::complex<double> port_meas = (std::abs(carrier) > 0.0)
                                                 ? out / carrier
                                                 : std::complex<double>(0.0, 0.0);

            // The same port at the real projection, the control the exact port is told apart from.
            std::complex<double> Tt_te, Tt_tm;
            diag_port(es, ea, std::complex<double>(ct, 0.0), true, Tt_te, Tt_tm);

            double g_ref = 0.5 * (std::norm(Te_te * Se_te * M) + std::norm(Te_tm * Se_tm * M));

            auto dB = [](double x, double y)
            { return 20.0 * std::log10(std::max(1e-300, x) / std::max(1e-300, y)); };

            QD_DIAG("%d,%.1f,%.9f,%.9f,%.9f,%.9f,%.6f,%.3e,%.3e,%u\n",
                    m, thd, ct, std::abs(port_meas), std::abs(Ts_te), std::abs(Tt_te),
                    dB(std::abs(Tt_te), std::abs(Ts_te)),
                    std::abs(out - ref), std::abs(Ce.gain(0) - g_ref),
                    (unsigned)Cs.resolved_type[0]);

            // The exit leg is the port at the recovered angle, the series and the traversal, and
            // nothing else.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(out - ref) < 1e-9);

            // Same statement per channel, read through a basis-invariant quantity so the EM run
            // pins the TM port as well.
            CHECK(std::abs(Ce.gain(0) - g_ref) < 1e-9);

            // The port moves, so the case can fail. The recovery changes nothing for a real permittivity, which is why
            // the lossless control is exempt.
            if (m != 6 && thd >= 40.0)
                CHECK(std::abs(Tt_te - Ts_te) > 1e-6);
        }
    }
}

TEST_CASE("Traced solid - air | slab | air chain against the analytic slab", "[wave]")
{
    // The end-to-end statement: the chain is assembled from the ports the two public functions write, nothing
    // reconstructed in between. ray_mesh_interact supplies the entry coefficient and the refracted direction,
    // ray_state_update the exit leg (exit port, series, traversal) from that direction; their product is the analytic
    // slab, which shares no code with either.
    //
    // The tracer is undeviated (type 4): the geometric ray continues straight, so the far face is met at the external
    // angle while path_dirN carries the Snell direction. A refracting tracer (type 5) bends the geometry itself and the
    // two angles coincide.
    //
    // The entry-boundary case divides the exit factor out, so an exit at the wrong cosine would cancel there; here it
    // cancels against nothing.
    const double f_hz = FRQ;
    const double L = 0.10; // normal thickness of the slab
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);

    QD_DIAG("\n# diag: traced chain, entry port from ray_mesh_interact, exit leg from ray_state_update\n");
    QD_DIAG("# air | slab | air, normal thickness 100 mm, 1 GHz; oracle is the analytic slab, TE channel\n");
    QD_DIAG("mtl,theta_ext,ct_mesh,ct_map,d_ct,chain_dB,ana_dB,d_dB,rtype\n");

    for (int m : {5, 1, 4, 6})
    {
        QD_DIAG("# material %d, eps %.2f %+.2fj\n", m,
                std::real(diag_eps(m, f_hz)), std::imag(diag_eps(m, f_hz)));
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
        {
            std::complex<double> es = diag_eps(m, f_hz);
            double th = thd * QPI / 180.0;

            // Entry: one face, air -> slab, undeviated transmission.
            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));

            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL,
                                                 &fbs_ind, &sbs_ind);
            REQUIRE(fbs_ind.n_elem == 1);
            REQUIRE(fbs_ind(0) != 0);

            arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                            &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                            &path_dirN, true, (arma::u32_vec *)nullptr);

            REQUIRE(path_dirN.n_rows == 1);
            std::complex<double> a_entry(xprmatN(0, 0), xprmatN(1, 0));

            double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
            double ct_mesh = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
            double ct_map = diag_traced_ct(es, th); // the same map, evaluated by hand

            // Exit: the far face of the layer, reached along the direction the mesh call wrote.
            // fbs_angle stays the external angle - the geometric ray was never bent - and the
            // normal thickness enters as the origin-to-face distance.
            auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                   /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - th, /*always resolve*/ 0.0);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = path_dirN(0, 0);
            C.path_dir_prev(0, 1) = path_dirN(0, 1);
            C.path_dir_prev(0, 2) = path_dirN(0, 2);
            C.run();

            // Absolute, because the exit rebuilds the field rather than scaling it.
            std::complex<double> a_exit = cpx(C.xprmat, 0, 0);
            std::complex<double> chain = a_entry * a_exit;

            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);

            double chain_dB = 10.0 * std::log10(std::max(1e-300, std::norm(chain)));
            double ana_dB = 10.0 * std::log10(std::max(1e-300, p_te));

            QD_DIAG("%d,%.1f,%.9f,%.9f,%.3e,%.6f,%.6f,%.6f,%u\n",
                    m, thd, ct_mesh, ct_map, std::abs(ct_mesh - ct_map),
                    chain_dB, ana_dB, chain_dB - ana_dB, (unsigned)C.resolved_type[0]);

            // The direction the tracer wrote is the map the recovery inverts. A gap here means the
            // two have drifted apart and every reference below is measuring the wrong thing.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(ct_mesh - ct_map) < 1e-9);

            // Entry port, traversal, exit port and series, telescoping onto the analytic slab.
            CHECK(std::abs(chain_dB - ana_dB) < 1e-6);

            // The oracle is the average of the two channels; the scalar chain measures TE, so the average is printed
            // only where the two coincide. The per-channel split of the reference against its average.
            CHECK(std::abs(diag_slab_p_avg(es, L, th, f_hz) - 0.5 * (p_te + p_tm)) < 1e-15);
        }
    }
}

TEST_CASE("Permeability - the traced solid and a magnetic layer in a stack", "[wave]")
{
    // A traced solid with mu != 1, and a magnetic layer inside an ordered stack. The analytic slab takes the three
    // substitutions of the sheet builders, z = sqrt(eps/mu), the Snell ratio as a ratio of eps mu products and n =
    // sqrt(eps mu); diag_traced_ct and diag_ct_exact take the third, since the map qd_refract applies is built from the
    // product.
    //
    // What a magnetic traced solid adds over a magnetic sheet:
    //
    //   The telescoping property does not read mu. The entry port carries sqrt(Re(z_s c_s)/Re(z_a c_a)) and the exit
    //   port its reciprocal, and the pair cancels whatever z_s is. What a magnetic run checks is that both crossings
    //   read the same z_s and the same c_s, which the recovered cosine provides.
    //
    //   The swap leaves the geometry bit-identical. The Snell ratio, the refraction root, the total-reflection test,
    //   the recovered cosine, medium_gain's loss rate and the traversal's index are all eps mu products, so an eps <->
    //   mu swap moves nothing about the traced path, the recovered cosine or the traversal - only the two interface
    //   channels, which it exchanges.
    const double f_hz = FRQ;
    const double L = 0.10;    // normal thickness of the traced solid
    const double thk = 0.006; // sheet thickness in the stack block
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    struct MuRow
    {
        double a, c, e, g, thk;
    };
    const int MSOL = 1, MSWP = 2, MMAT = 3, MLAY = 4, DLAY = 5;
    const std::vector<MuRow> pal = {
        {1.50, 0.40 / 17.98, 1.20, 0.0, 0.0},            // 1 MSOL, magnetic lossy solid
        {1.20, 0.0, 1.50, 0.40 / 17.98, 0.0},            // 2 MSWP, its eps<->mu swap
        {2.00, 0.0, 2.00, 0.0, 0.0},                     // 3 MMAT, matched: z = 1, n = 2
        {5.24, 0.63 / 17.98, 2.10, 0.35 / 17.98, 0.006}, // 4 MLAY, magnetic sheet layer
        {1.50, 0.40 / 17.98, 1.00, 0.0, 0.006},          // 5 DLAY, dielectric sheet layer
    };
    const std::complex<double> e_sol(1.50, -0.40), mu_sol(1.20, 0.0);
    const std::complex<double> e_swp(1.20, 0.0), mu_swp(1.50, -0.40);
    const std::complex<double> e_mat(2.0, 0.0), mu_mat(2.0, 0.0);
    const std::complex<double> e_lay(5.24, -0.63), mu_lay(2.10, -0.35);
    const std::complex<double> e_dly(1.50, -0.40);

    std::unordered_map<std::string, std::vector<double>> mtl;
    for (const char *k : {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB",
                          "alpha", "alphaB", "fRef"})
        mtl[k] = std::vector<double>(pal.size(), 0.0);
    for (size_t i = 0; i < pal.size(); ++i)
    {
        mtl["a"][i] = pal[i].a, mtl["c"][i] = pal[i].c;
        mtl["e"][i] = pal[i].e, mtl["g"][i] = pal[i].g;
        mtl["thk"][i] = pal[i].thk, mtl["fRef"][i] = 1.0;
    }

    // ---- A traced air | slab | air chain through a magnetic solid ----
    // Assembled from the ports the two public functions write, as the dielectric chain case does: the entry coefficient
    // and the refracted direction from ray_mesh_interact, the exit leg from ray_state_update, nothing reconstructed in
    // between. The sweep stops at 75 deg: the internal critical angle of eps mu = 1.80 - 0.48j is 47.6 deg and the
    // traced internal angle reaches 45.5 deg at 75 deg of external incidence.
    QD_DIAG("\n# diag: permeability, traced air | slab | air chain, normal thickness 100 mm\n");
    QD_DIAG("# MSOL is eps 1.50 - 0.40j with mu 1.20; MSWP is the eps<->mu swap of it, so the two\n");
    QD_DIAG("# share every cosine and every phase and differ only in the admittance.\n");
    QD_DIAG("mtl,theta_ext,ct_mesh,ct_map,d_ct,chain_dB,ana_dB,d_dB,rtype\n");

    struct Solid
    {
        const char *name;
        int idx;
        std::complex<double> e, mu;
    };
    const std::vector<Solid> solids = {
        {"MSOL", MSOL, e_sol, mu_sol},
        {"MSWP", MSWP, e_swp, mu_swp},
        {"MMAT", MMAT, e_mat, mu_mat},
    };

    // The sweep starts at 10 deg, as the dielectric chain case does. At normal incidence the traced and geometric
    // cosines coincide, the exact-cosine path is not taken, and the traversal falls back to the path form, 2.5e-5 dB
    // from the exact one, which would not clear the bound below; that gap has nothing to do with permeability.
    for (const auto &S : solids)
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0;

            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)S.idx));

            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL,
                                                 &fbs_ind, &sbs_ind);
            REQUIRE(fbs_ind.n_elem == 1);
            REQUIRE(fbs_ind(0) != 0);

            arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                            &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                            &path_dirN, true, (arma::u32_vec *)nullptr);

            REQUIRE(path_dirN.n_rows == 1);
            std::complex<double> a_entry(xprmatN(0, 0), xprmatN(1, 0));

            double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
            double ct_mesh = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
            double ct_map = diag_traced_ct(S.e, th, S.mu);

            auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                   /*M1*/ S.idx, /*M2*/ 0, /*prev*/ 0, /*cur*/ S.idx, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - th, /*always resolve*/ 0.0);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = path_dirN(0, 0);
            C.path_dir_prev(0, 1) = path_dirN(0, 1);
            C.path_dir_prev(0, 2) = path_dirN(0, 2);
            C.run();

            std::complex<double> chain = a_entry * cpx(C.xprmat, 0, 0);

            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(S.e, L, th, f_hz, p_te, p_tm, S.mu);

            double chain_dB = 10.0 * std::log10(std::max(1e-300, std::norm(chain)));
            double ana_dB = 10.0 * std::log10(std::max(1e-300, p_te));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.3e,%.6f,%.6f,%.6f,%u\n",
                    S.name, thd, ct_mesh, ct_map, std::abs(ct_mesh - ct_map),
                    chain_dB, ana_dB, chain_dB - ana_dB, (unsigned)C.resolved_type[0]);

            INFO(S.name << ", theta_ext " << thd);

            // The direction the tracer wrote is the map the recovery inverts, with the product
            // eps*mu in both. A gap here means the map and the reference have drifted apart.
            CHECK(std::abs(ct_mesh - ct_map) < 1.0e-9);

            // Entry port, traversal, exit port and series, telescoping onto the analytic slab.
            CHECK(std::abs(chain_dB - ana_dB) < 1.0e-6);
            CHECK(std::abs(diag_slab_p_avg(S.e, L, th, f_hz, S.mu) - 0.5 * (p_te + p_tm)) < 1.0e-15);

            // The matched solid, eps = mu = 2, has unit admittance, so neither face reflects at
            // normal incidence and the lossless slab is fully transparent there. At any angle its
            // two channels stay equal, because equal admittances collapse the TE and TM Fresnel
            // expressions onto the same function of the two cosines. No permittivity reaches
            // either statement.
            if (S.idx == MMAT)
                CHECK(std::abs(p_te - p_tm) < 1.0e-14);
        }

    // The matched solid at normal incidence, on the reference alone, where the chain cannot be used
    // for the reason above. With z = 1 on both sides neither face reflects and a lossless slab is
    // fully transparent whatever its index; the transmitted field is the propagation across it and
    // nothing else. No permittivity reaches this, because one that moves n also moves z.
    {
        double p_te = 0.0, p_tm = 0.0;
        diag_slab_p_pol(e_mat, L, 0.0, f_hz, p_te, p_tm, mu_mat);
        INFO("matched solid at normal incidence");
        CHECK(std::abs(p_te - 1.0) < 1.0e-12);
        CHECK(std::abs(p_tm - 1.0) < 1.0e-12);
    }

    // The swap, read off the two sweeps above. Scalar mode carries the TE channel only, so the
    // exchange shows up as the swapped run's TE equalling the original run's TM. The traced cosine
    // is asserted identical between the two, which is the geometry half of the statement.
    QD_DIAG("\n# diag: permeability, eps<->mu swap on a traced solid\n");
    QD_DIAG("theta_ext,d_ct,p_te_swp,p_tm_sol,d_exchange,separation_dB\n");

    for (double thd : {0.0, 10.0, 20.0, 40.0, 60.0, 75.0})
    {
        double th = thd * QPI / 180.0;
        double a_te = 0.0, a_tm = 0.0, b_te = 0.0, b_tm = 0.0;
        diag_slab_p_pol(e_sol, L, th, f_hz, a_te, a_tm, mu_sol);
        diag_slab_p_pol(e_swp, L, th, f_hz, b_te, b_tm, mu_swp);

        double ct_a = diag_traced_ct(e_sol, th, mu_sol), ct_b = diag_traced_ct(e_swp, th, mu_swp);
        double sep = 10.0 * std::log10(a_te / a_tm);

        QD_DIAG("%.1f,%.3e,%.9e,%.9e,%.3e,%+.4f\n", thd, std::abs(ct_a - ct_b),
                b_te, a_tm, std::abs(b_te - a_tm), sep);

        INFO("swap, theta_ext " << thd);
        CHECK(std::abs(ct_a - ct_b) < 1.0e-12); // the traced path does not move
        CHECK(std::abs(b_te - a_tm) < 1.0e-14); // and the two channels exchange
        CHECK(std::abs(b_tm - a_te) < 1.0e-14);

        // Past normal incidence the two channels are far enough apart that the exchange is a
        // statement rather than an identity between two equal numbers.
        if (thd >= 40.0)
            CHECK(std::abs(sep) > 1.0e-2);
    }

    // ---- A magnetic layer inside an ordered two-sheet stack, air | 6 mm | 6 mm | air ----
    // diag_cm_stack carries (eps, mu) per layer. What this adds over a magnetic single sheet is the layer product: two
    // layers whose admittances and indices are independent of each other, and the order statement has to survive that.
    QD_DIAG("\n# diag: permeability, ordered stack with one magnetic layer\n");
    QD_DIAG("# air | MLAY (eps 5.24 - 0.63j, mu 2.10 - 0.35j) | DLAY (eps 1.50 - 0.40j) | air\n");
    QD_DIAG("order,theta_n,ref_|T|,ref_|R|,out_|T|,out_|R|,dT,dR,rtype_T,rtype_R\n");

    const uint8_t OT_STACK_FBS = 71; // FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // SBS layer traversed first

    struct MuOrd
    {
        const char *name;
        uint8_t otype;
        std::complex<double> e1, mu1, e2, mu2;
    };
    const std::vector<MuOrd> orders = {
        {"M|D", OT_STACK_FBS, e_lay, mu_lay, e_dly, U},
        {"D|M", OT_STACK_SBS, e_dly, U, e_lay, mu_lay},
    };

    for (const auto &O : orders)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);

            DiagSheet Xs = diag_cm_stack(e_air, U, {{O.e1, O.mu1, thk}, {O.e2, O.mu2, thk}},
                                         e_air, U, ci, f_hz, false, /*scalar*/ true);
            DiagSheet Xe = diag_cm_stack(e_air, U, {{O.e1, O.mu1, thk}, {O.e2, O.mu2, thk}},
                                         e_air, U, ci, f_hz, false, /*EM*/ false);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, O.otype, 2u,
                                       /*M1*/ MLAY, /*M2*/ DLAY,
                                       /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };

            auto Ct = run(4), Ce = run(1), Cr = run(3), Cx = run(0);

            double gT = 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
            double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));
            double dT = std::abs(cpx(Ct.xprmat, 0, 0) - Xs.T_te);
            double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u,%u\n",
                    O.name, thd, std::abs(Xs.T_te), std::abs(Xs.R_te),
                    std::abs(cpx(Ct.xprmat, 0, 0)), std::abs(cpx(Cr.xprmat, 0, 0)),
                    dT, dR, (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO("magnetic stack " << O.name << ", theta_n " << thd);
            CHECK(gT > 1.0e-3);
            CHECK(gR > 1.0e-3);
            CHECK(Ct.resolved_type[0] != 0);
            CHECK(Cr.resolved_type[0] != 0);
            CHECK(dT < 1.0e-9);
            CHECK(dR < 1.0e-9);
            CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
            CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);
        }

    // Reciprocity survives a magnetic layer: the transmit port of the stack does not depend on
    // which layer comes first and the reflect port does, with the layers now differing in their
    // admittance as well as their index.
    for (double thd : {0.0, 45.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet MD = diag_cm_stack(e_air, U, {{e_lay, mu_lay, thk}, {e_dly, U, thk}},
                                     e_air, U, ci, f_hz, false, false);
        DiagSheet DM = diag_cm_stack(e_air, U, {{e_dly, U, thk}, {e_lay, mu_lay, thk}},
                                     e_air, U, ci, f_hz, false, false);
        INFO("magnetic stack order, theta_n " << thd);
        CHECK(std::abs(MD.T_te - DM.T_te) < 1.0e-14);
        CHECK(std::abs(MD.T_tm - DM.T_tm) < 1.0e-14);
        CHECK(std::abs(MD.R_te - DM.R_te) > 1.0e-3);
    }
}

TEST_CASE("Layered stack - the characteristic-matrix reference against the single-layer oracles", "[wave]")
{
    // diag_cm_stack reproduces diag_sheet_composite and diag_sheet_composite_cx exactly at one layer, the references
    // the composite is asserted against. Nothing here touches the library: two independently written closed forms
    // agree, so a failure in the cases below is attributable to the solver rather than to the reference.
    //
    // Four further properties of the matrix product, each of which would catch a different transcription error:
    //   a layer of zero thickness is inert, which pins the identity element,
    //   splitting one layer into two halves is inert, which pins the product and the phase,
    //   the transmit port is invariant under layer order, which is reciprocity (det M = 1),
    //   the reflect port is not, which is why an ordered stack carries its order.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0);

    const std::complex<double> e_air(1.0, 0.0);
    const std::complex<double> e_s = diag_eps(3, f_hz);  // sheet, eps 5.24 - 0.63j
    const std::complex<double> e_lo = diag_eps(5, f_hz); // LO, eps 1.50 - 0.40j
    const std::complex<double> e_d = diag_eps(4, f_hz);  // DENSER, eps 9.0 - 1.20j

    struct Stack
    {
        const char *name;
        std::complex<double> e_near, e_far;
        bool disp_far;
    };
    const std::vector<Stack> stacks = {
        {"air | sheet | air", e_air, e_air, false},
        {"air | sheet | LO ", e_air, e_lo, false},
        {"LO  | sheet | air", e_lo, e_air, true},
        {"LO  | sheet | LO ", e_lo, e_lo, false},
        {"DEN | sheet | LO ", e_d, e_lo, false},
    };

    QD_DIAG("\n# diag: characteristic-matrix reference against the single-layer builders\n");
    QD_DIAG("stack,mode,theta_n,d_real_builder,d_complex_builder\n");

    double worst_real = 0.0, worst_cx = 0.0;
    for (const auto &S : stacks)
        for (int mode = 0; mode < 2; ++mode) // 0 = EM, 1 = scalar
            for (double thd : {0.0, 10.0, 30.0, 50.0, 70.0, 85.0, 89.9})
            {
                bool scalar = (mode == 1);
                double th = thd * QPI / 180.0, ct = std::cos(th);

                DiagSheet A = diag_sheet_composite(S.e_near, e_s, S.e_far, thk, ct, f_hz,
                                                   S.disp_far, scalar);
                DiagSheet B = diag_cm_stack(S.e_near, U, {{e_s, U, thk}}, S.e_far, U,
                                            std::complex<double>(ct, 0.0), f_hz, S.disp_far, scalar);

                double d_re = std::max(std::max(std::abs(A.T_te - B.T_te), std::abs(A.T_tm - B.T_tm)),
                                       std::max(std::abs(A.R_te - B.R_te), std::abs(A.R_tm - B.R_tm)));

                // The same statement at a complex incidence cosine, the argument the core is handed where the near
                // medium is a traced solid.
                std::complex<double> cx = diag_ct_exact(e_lo, th);
                DiagSheet Ac = diag_sheet_composite_cx(e_lo, e_s, S.e_far, thk, cx, f_hz,
                                                       S.disp_far, scalar);
                DiagSheet Bc = diag_cm_stack(e_lo, U, {{e_s, U, thk}}, S.e_far, U, cx, f_hz,
                                             S.disp_far, scalar);
                double d_cx = std::max(std::max(std::abs(Ac.T_te - Bc.T_te), std::abs(Ac.T_tm - Bc.T_tm)),
                                       std::max(std::abs(Ac.R_te - Bc.R_te), std::abs(Ac.R_tm - Bc.R_tm)));

                worst_real = std::max(worst_real, d_re);
                worst_cx = std::max(worst_cx, d_cx);

                QD_DIAG("%s,%s,%.1f,%.3e,%.3e\n", S.name, scalar ? "scalar" : "EM    ", thd, d_re, d_cx);

                INFO(S.name << ", " << (scalar ? "scalar" : "EM") << ", theta_n " << thd);
                CHECK(d_re < 1.0e-14);
                CHECK(d_cx < 1.0e-14);
            }

    QD_DIAG("# worst against the real-cosine builder %.3e, against the complex one %.3e\n",
            worst_real, worst_cx);

    // Structural properties of the product, on a stack the two cases below both use.
    for (double thd : {0.0, 40.0, 80.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);

        DiagSheet one = diag_cm_stack(e_air, U, {{e_s, U, thk}}, e_air, U, ci, f_hz, false, false);
        DiagSheet zero = diag_cm_stack(e_air, U, {{e_s, U, thk}, {e_d, U, 0.0}}, e_air, U, ci, f_hz,
                                       false, false);
        DiagSheet half = diag_cm_stack(e_air, U, {{e_s, U, 0.5 * thk}, {e_s, U, 0.5 * thk}},
                                       e_air, U, ci, f_hz, false, false);

        DiagSheet AB = diag_cm_stack(e_air, U, {{e_s, U, thk}, {e_lo, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        DiagSheet BA = diag_cm_stack(e_air, U, {{e_lo, U, thk}, {e_s, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);

        INFO("theta_n " << thd);

        // A layer of zero thickness contributes the identity.
        CHECK(std::abs(one.T_te - zero.T_te) < 1.0e-14);
        CHECK(std::abs(one.R_tm - zero.R_tm) < 1.0e-14);

        // One layer and two halves of it are the same layer.
        CHECK(std::abs(one.T_te - half.T_te) < 1.0e-14);
        CHECK(std::abs(one.R_tm - half.R_tm) < 1.0e-14);

        // Reciprocity: the transmit port of a stack does not depend on which layer comes first.
        CHECK(std::abs(AB.T_te - BA.T_te) < 1.0e-14);
        CHECK(std::abs(AB.T_tm - BA.T_tm) < 1.0e-14);

        // The reflect port does: an ordered stack carries its order for this alone, and a reference that lost the order
        // would show here.
        CHECK(std::abs(AB.R_te - BA.R_te) > 1.0e-3);
    }
}

TEST_CASE("Permeability - the sheet, stack and glued paths", "[wave]")
{
    // The e, f, g, h columns in every expression the sheet and stack paths form: sheets, stacks, glued pairs and traced
    // chains with mu != 1.
    //
    // Where mu enters. Every site is one of two forms. Through the product eps mu: the index n = sqrt(eps mu) in every
    // propagation factor, every Snell ratio and refraction root, the total-reflection test, medium_gain's loss rate,
    // the recovered cosine, the displaced host's cosine and index, and the traversal, which never sees mu on its own.
    // Through the quotient eps/mu: the admittance z in every Fresnel amplitude, the core's tilted admittances z c and z
    // / c, and the flux factors Re(z c) and Re(conj(z) c).
    //
    // So there is one magnetic degree of freedom: n and z stop being the same number. A single permittivity fixes both,
    // which is why no dielectric palette reaches this, and why an eps <-> mu swap is the sharpest probe: it holds n,
    // and with it every cosine and phase, while inverting z. The single-interface permeability cases involve no layer,
    // so they cannot see a propagation phase built at the wrong index or a series built at the wrong admittance.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    // A local palette, so diag_palette and diag_eps stay dielectric and every existing residual
    // stays where it is. Same column convention: eps = a - j*17.98*c/fGHz, mu = e - j*17.98*g/fGHz.
    struct MuRow
    {
        double a, c, e, g, thk;
    };
    const std::complex<double> e_air(1.0, 0.0), U(1.0, 0.0);
    const std::complex<double> e_ms(5.24, -0.63), mu_ms(2.10, -0.35); // magnetic sheet
    const std::complex<double> e_mh(1.50, -0.40), mu_mh(1.60, 0.0);   // magnetic host
    const std::complex<double> e_k(4.0, 0.0), mu_k(4.0, 0.0);         // matched dense sheet

    const int MSHEET = 1, MSWAP = 2, MHOST = 3, MATCHED = 4;
    const std::vector<MuRow> pal = {
        {5.24, 0.63 / 17.98, 2.10, 0.35 / 17.98, thk}, // 1 MSHEET
        {2.10, 0.35 / 17.98, 5.24, 0.63 / 17.98, thk}, // 2 MSWAP, the eps<->mu swap of MSHEET
        {1.50, 0.40 / 17.98, 1.60, 0.0, 0.0},          // 3 MHOST, a magnetic solid
        {4.00, 0.0, 4.00, 0.0, thk},                   // 4 MATCHED, eps = mu so z = 1, n = 4
    };

    std::unordered_map<std::string, std::vector<double>> mtl;
    for (const char *k : {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB",
                          "alpha", "alphaB", "fRef"})
        mtl[k] = std::vector<double>(pal.size(), 0.0);
    for (size_t i = 0; i < pal.size(); ++i)
    {
        mtl["a"][i] = pal[i].a, mtl["c"][i] = pal[i].c;
        mtl["e"][i] = pal[i].e, mtl["g"][i] = pal[i].g;
        mtl["thk"][i] = pal[i].thk, mtl["fRef"][i] = 1.0;
    }

    const uint8_t OT_SHEET_ISO = 67;   // ok | sheet, no co-located face
    const uint8_t OT_GLUED_ENTRY = 71; // ok | front | co-located | sheet -> a glued entry

    // Channel identification, as in the flux-conversion case: make1 puts orig, fbs and dest on the x axis with the
    // normal along x, so the incidence plane is degenerate, the polarization basis takes its identity branch, and VV is
    // TM while HH is TE.
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ---- The generalized builders against diag_cm_stack, at mu != 1 ----
    // Nothing here touches the library. diag_cm_stack takes (eps, mu) pairs and shares no code with the two
    // single-layer builders, so this establishes the three substitutions before they serve as oracles below.
    QD_DIAG("\n# diag: permeability, generalized single-layer builders against diag_cm_stack\n");
    QD_DIAG("stack,mode,theta_n,d_real_builder,d_complex_builder\n");

    struct MuStack
    {
        const char *name;
        std::complex<double> e_near, mu_near, e_far, mu_far;
        bool disp_far;
    };
    const std::vector<MuStack> stacks = {
        {"air  | Msheet | air ", e_air, U, e_air, U, false},
        {"air  | Msheet | Mhst", e_air, U, e_mh, mu_mh, false},
        {"Mhst | Msheet | air ", e_mh, mu_mh, e_air, U, true},
        {"Mhst | Msheet | Mhst", e_mh, mu_mh, e_mh, mu_mh, false},
    };

    double worst_re = 0.0, worst_cx = 0.0;
    for (const auto &S : stacks)
        for (int mode = 0; mode < 2; ++mode)
            for (double thd : {0.0, 10.0, 30.0, 50.0, 70.0, 85.0, 89.9})
            {
                bool scalar = (mode == 1);
                double th = thd * QPI / 180.0, ct = std::cos(th);

                DiagSheet A = diag_sheet_composite(S.e_near, e_ms, S.e_far, thk, ct, f_hz,
                                                   S.disp_far, scalar, S.mu_near, mu_ms, S.mu_far);
                DiagSheet B = diag_cm_stack(S.e_near, S.mu_near, {{e_ms, mu_ms, thk}},
                                            S.e_far, S.mu_far, std::complex<double>(ct, 0.0),
                                            f_hz, S.disp_far, scalar);
                double d_re = std::max(std::max(std::abs(A.T_te - B.T_te), std::abs(A.T_tm - B.T_tm)),
                                       std::max(std::abs(A.R_te - B.R_te), std::abs(A.R_tm - B.R_tm)));

                // The same at a complex incidence cosine, which is what a sheet reached from inside
                // a magnetic solid is handed. The cosine is built from the eps*mu product, so this
                // also pins that the transverse invariant carries mu.
                std::complex<double> cx =
                    std::sqrt(1.0 - ((e_air * U) / (e_mh * mu_mh)) * (1.0 - ct * ct));
                DiagSheet Ac = diag_sheet_composite_cx(e_mh, e_ms, S.e_far, thk, cx, f_hz,
                                                       S.disp_far, scalar, mu_mh, mu_ms, S.mu_far);
                DiagSheet Bc = diag_cm_stack(e_mh, mu_mh, {{e_ms, mu_ms, thk}}, S.e_far, S.mu_far,
                                             cx, f_hz, S.disp_far, scalar);
                double d_cx = std::max(std::max(std::abs(Ac.T_te - Bc.T_te), std::abs(Ac.T_tm - Bc.T_tm)),
                                       std::max(std::abs(Ac.R_te - Bc.R_te), std::abs(Ac.R_tm - Bc.R_tm)));

                worst_re = std::max(worst_re, d_re), worst_cx = std::max(worst_cx, d_cx);
                QD_DIAG("%s,%s,%.1f,%.3e,%.3e\n", S.name, scalar ? "scalar" : "EM    ", thd, d_re, d_cx);

                INFO(S.name << ", " << (scalar ? "scalar" : "EM") << ", theta_n " << thd);
                CHECK(d_re < 1.0e-14);
                CHECK(d_cx < 1.0e-14);

                // And the complex-cosine builder reproduces the real-cosine one at a real cosine with mu live, the
                // invariant the sheet-in-a-medium case rests on.
                DiagSheet Ar = diag_sheet_composite_cx(S.e_near, e_ms, S.e_far, thk,
                                                       std::complex<double>(ct, 0.0), f_hz,
                                                       S.disp_far, scalar, S.mu_near, mu_ms, S.mu_far);
                CHECK(std::abs(A.T_te - Ar.T_te) < 1.0e-14);
                CHECK(std::abs(A.R_tm - Ar.R_tm) < 1.0e-14);
            }

    QD_DIAG("# worst against the real-cosine builder %.3e, against the complex one %.3e\n",
            worst_re, worst_cx);

    // ---- An isolated magnetic sheet in air, and a glued entry into a magnetic solid, against the generalized
    // reference ----
    // The isolated rows run with mtl_current = 0, so no host factor is applied and the written field is the composite
    // outright; those are asserted complex.
    QD_DIAG("\n# diag: permeability through the library, sheet eps 5.24 - 0.63j, mu 2.10 - 0.35j\n");
    QD_DIAG("stack,theta_n,T_te_meas,T_te_ref,T_tm_meas,T_tm_ref,d_te,d_tm,rtype\n");

    struct LibRow
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2;
        int cur;
        std::complex<double> e_far, mu_far;
        bool disp_far;
    };
    const std::vector<LibRow> lrows = {
        {"isolated air | Msheet | air ", OT_SHEET_ISO, 1u, 0, 0, e_air, U, false},
        {"entry    air | Msheet | Mhst", OT_GLUED_ENTRY, 2u, MHOST, 0, e_mh, mu_mh, false},
    };

    for (const auto &R : lrows)
        for (double thd : {0.0, 20.0, 40.0, 60.0, 80.0, 85.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto C = make1<double>(mtl, /*EM transmission*/ 1, R.otype, R.nH,
                                   /*M1 = the sheet face*/ MSHEET, /*M2*/ R.m2,
                                   /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n,
                                   /*eps: the host series must not fire*/ 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(e_air, e_ms, R.e_far, thk, ct, f_hz,
                                               R.disp_far, false, U, mu_ms, R.mu_far);

            std::complex<double> t_te = TE(C.xprmat), t_tm = TM(C.xprmat);
            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%u\n", R.name, thd,
                    std::abs(t_te), std::abs(D.T_te), std::abs(t_tm), std::abs(D.T_tm),
                    std::abs(t_te - D.T_te), std::abs(t_tm - D.T_tm), (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0);
            CHECK(std::abs(t_te - D.T_te) < 1.0e-9);
            CHECK(std::abs(t_tm - D.T_tm) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm))) < 1.0e-9);
        }

    // ---- The eps <-> mu swap, on the sheet only, in air ----
    // Swapping holds n = sqrt(eps mu), and so every cosine and the propagation factor, and replaces z by 1/z. With both
    // hosts at unit admittance every reflection and transmission product of the Airy pair maps onto the other
    // polarization, so the swap exchanges the TE and TM ports exactly, on both ports. The power-averaged gain is
    // invariant under that exchange, so the statement can only be made per channel: an admittance read as sqrt(eps)
    // would still report the same total.
    QD_DIAG("\n# diag: permeability, eps<->mu swap on an isolated sheet in air\n");
    QD_DIAG("theta_n,d_te_vs_tm,d_tm_vs_te,d_gain,separation\n");

    for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto mk = [&](int idx)
        {
            auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_ISO, 1u,
                                   idx, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();
            return C;
        };
        auto A = mk(MSHEET), B = mk(MSWAP);

        std::complex<double> a_te = TE(A.xprmat), a_tm = TM(A.xprmat);
        std::complex<double> b_te = TE(B.xprmat), b_tm = TM(B.xprmat);
        double d1 = std::abs(b_te - a_tm), d2 = std::abs(b_tm - a_te);
        double dg = std::abs((double)A.gain(0) - (double)B.gain(0));
        double sep = std::abs(a_te - a_tm);

        QD_DIAG("%.1f,%.3e,%.3e,%.3e,%.6f\n", thd, d1, d2, dg, sep);

        INFO("swap, theta_n " << thd);
        REQUIRE(A.resolved_type[0] != 0);
        REQUIRE(B.resolved_type[0] != 0);
        CHECK(d1 < 1.0e-9);
        CHECK(d2 < 1.0e-9);
        CHECK(dg < 1.0e-12);

        // Past normal incidence the two channels are far enough apart that the exchange above is a
        // statement rather than an identity between two equal numbers.
        if (thd > 0.0)
            CHECK(sep > 1.0e-3);
    }

    // ---- An impedance-matched dense sheet, eps = mu = 4, so z = 1 and n = 4 ----
    // At normal incidence both interfaces vanish and the sheet is invisible except for its phase: the composite is
    // exactly the propagation it carries over the host it displaces, exp(-j k0 thk (n - 1)), with unit magnitude and no
    // reflection. No dielectric can do that, because a single permittivity that moves n also moves z.
    // At oblique incidence it reflects, and the reflection is polarization-independent: with z equal on both sides the
    // TE and TM Fresnel expressions collapse onto one function of the two cosines.
    QD_DIAG("\n# diag: permeability, impedance-matched dense sheet, eps = mu = 4, thk 6 mm\n");
    QD_DIAG("theta_n,|T|,arg_T_deg,pred_deg,|R_te|,|R_tm|,d_pol\n");

    const double k0 = 2.0 * QPI * f_hz / C0;
    for (double thd : {0.0, 20.0, 45.0, 70.0})
    {
        double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

        auto Ct = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_ISO, 1u,
                                MATCHED, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
        Ct.run();
        auto Cr = make1<double>(mtl, /*EM reflection*/ 0, OT_SHEET_ISO, 1u,
                                MATCHED, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
        Cr.run();

        DiagSheet D = diag_sheet_composite(e_air, e_k, e_air, thk, ct, f_hz, false, false,
                                           U, mu_k, U);

        std::complex<double> t_te = TE(Ct.xprmat);
        double pred = -k0 * thk * (4.0 - 1.0) * 180.0 / QPI; // normal incidence only
        double arg_t = std::arg(t_te) * 180.0 / QPI;
        double r_te = std::abs(TE(Cr.xprmat)), r_tm = std::abs(TM(Cr.xprmat));

        QD_DIAG("%.1f,%.9f,%.6f,%.6f,%.9f,%.9f,%.2e\n", thd, std::abs(t_te), arg_t,
                (thd == 0.0) ? pred : 0.0, r_te, r_tm, std::abs(r_te - r_tm));

        INFO("matched sheet, theta_n " << thd);
        CHECK(std::abs(t_te - D.T_te) < 1.0e-9);
        CHECK(std::abs(TM(Ct.xprmat) - D.T_tm) < 1.0e-9);

        // The reflection carries no polarization dependence at any angle.
        CHECK(std::abs(std::abs(D.R_te) - std::abs(D.R_tm)) < 1.0e-14);
        CHECK(std::abs(r_te - r_tm) < 1.0e-9);

        if (thd == 0.0)
        {
            // Invisible except for the phase, exactly.
            CHECK(std::abs(D.R_te) < 1.0e-15);
            CHECK(std::abs(std::abs(t_te) - 1.0) < 1.0e-9);
            CHECK(std::abs(arg_t - pred) < 1.0e-3);
            CHECK((double)Cr.gain(0) < 1.0e-12);
        }
        else
        {
            // And it is a real reflection, so the normal-incidence statement is not the whole story.
            CHECK(r_te > 1.0e-3);
        }
    }
}

TEST_CASE("Layered stack - a host layer bounded by a sheet", "[wave]")
{
    // An i-o exit out of a host layer whose previous-medium word holds a sheet. The host's series runs between the face
    // ahead and the sheet behind, and the sheet is a layer with air behind it, not a half-space of its material: the
    // reflection behind the ray is the characteristic matrix of the sheet on its backing.
    //
    // Two sheets are swept. SHEET (eps 5.24 - 0.63j) is a real boundary and measures the coupling across the layer.
    // LO_NEAR (eps 1.5015 - 0.40j) is near-matched to the host: as a half-space it would reflect 2.5e-4 and the host's
    // series would collapse, while the wave passes through the sheet almost unimpeded and still sees the boundary
    // behind it.
    //
    // The reference is the two-layer stack the crossing is. The harness isolates the series factor, so the reference is
    // written as
    //     S = 1 / (1 - r_ahead * r_behind * phi^2)
    // with r_behind taken from the characteristic matrix. That factorization is exact: substituting the matrix value
    // for r_behind reproduces the full two-layer stack.
    //
    // No path_dir is supplied, so abs_cos_theta_t is the fbs cosine, the propagation factor is the path form diag_phi
    // reproduces, and both boundaries are taken from the slab side at that real cosine; the reference is built at the
    // same cosine.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9;

    const std::complex<double> e_h = diag_eps(HOST, f_hz), e_air(1.0, 0.0);

    struct Row
    {
        const char *name;
        int prev;
        bool near_matched;
    };
    const std::vector<Row> rows = {
        {"SHEET   eps 5.24 - 0.63j", SHEET, false},
        {"LO_NEAR eps 1.5015- 0.40j", NEAR, true},
    };

    QD_DIAG("\n# diag: host layer bounded by a sheet, host eps 1.50 - 0.40j, sheet thk 6 mm, 1 GHz.\n");
    QD_DIAG("# S_applied is the series SLAB_AIRY_FACTOR produces; S_ref is the same series with the\n");
    QD_DIAG("# sheet resolved as a layer backed by air, from the characteristic matrix. r_hs is the\n");
    QD_DIAG("# half-space reflection of the sheet and r_cm the layer value; they are asserted apart, so the\n");
    QD_DIAG("# series tells the two backings apart.\n");
    QD_DIAG("sheet,L_mm,theta_n,|S_applied|,arg_S_applied,|S_ref|,arg_S_ref,|dS|,d_dB,|r_hs|,|r_cm|,rtype\n");

    for (const auto &R : rows)
        for (double L : {0.05, 0.10})
            for (double thd : {0.0, 20.0, 40.0, 50.0})
            {
                double theta_n = thd * QPI / 180.0;
                const std::complex<double> e_s = diag_eps(R.prev, f_hz);

                auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                       /*M1*/ HOST, /*M2*/ 0, /*prev*/ R.prev, /*cur*/ HOST, /*buf*/ 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, /*eps: always resolve*/ 0.0);
                C.run();

                std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
                std::complex<double> applied = (std::abs(in) > 0.0) ? out / in
                                                                    : std::complex<double>(0.0, 0.0);
                std::complex<double> close = diag_close(e_h, L, theta_n, f_hz);
                std::complex<double> S_applied = (std::abs(close) > 0.0)
                                                     ? applied / close
                                                     : std::complex<double>(0.0, 0.0);

                // The reference. cs is the cosine correction mode uses on both boundaries, and the
                // matrix is anchored in the host so that its transverse invariant is the same one.
                std::complex<double> cs(std::cos(theta_n), 0.0);
                std::complex<double> phi = diag_phi(e_h, L, theta_n, f_hz);
                std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> r_hs = diag_r_slabside(e_h, e_s, cs);

                DiagSheet B = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz,
                                            false, true, /*undisplaced*/ 0.0);
                std::complex<double> S_ref = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);

                double dS = std::abs(S_applied - S_ref);
                double d_dB = 20.0 * std::log10(std::max(1e-300, std::abs(S_applied)) /
                                                std::max(1e-300, std::abs(S_ref)));

                QD_DIAG("%s,%.1f,%.1f,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%+.4f,%.3e,%.3e,%u\n",
                        R.name, L * 1000.0, thd,
                        std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                        std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI,
                        dS, d_dB, std::abs(r_hs), std::abs(B.R_te),
                        (unsigned)C.resolved_type[0]);

                INFO(R.name << ", L " << L * 1000.0 << " mm, theta_n " << thd);

                // The row has to be a resolve, or there is no series to measure.
                REQUIRE(C.resolved_type[0] != 0);
                REQUIRE(rt_bit(C.resolved_type[0], 2));
                REQUIRE(std::isfinite(C.gain(0)));

                // The two far-side reflections are far enough apart that the assertion below discriminates; a statement
                // about the references.
                CHECK(std::abs(r_hs - B.R_te) > 1.0e-3);

                // A near-matched sheet is nearly not a boundary, so the layer it stands for is nearly six more
                // millimetres of host. At normal incidence the two descriptions of that stack are one object and the
                // reference converges onto the extended-path form linearly in the mismatch. Only at normal incidence:
                // at an angle the layer uses k0 n thk cos and the extended path the walk-off cos^2, and their gap is
                // not this statement.
                if (R.near_matched && thd == 0.0)
                {
                    std::complex<double> phi2 = diag_phi(e_h, L + thk, theta_n, f_hz);
                    std::complex<double> S_ext = 1.0 / (1.0 - r_ahead * r_ahead * phi2 * phi2);
                    CHECK(std::abs(S_ref - S_ext) < 1.0e-4);
                }

                // The host layer's series resolves against the sheet as a layer, magnitude and
                // phase together.
                CHECK(std::abs(std::abs(S_applied) - std::abs(S_ref)) < 1.0e-9);
                CHECK(std::abs(S_applied - S_ref) < 1.0e-9);
            }
}

TEST_CASE("Layered stack - an ordered stack of two sheets resolves rather than killing the ray", "[wave]")
{
    // Two co-located sheets of different materials are an ordered stack: one event, resolved on the characteristic
    // matrix of the two layers.
    //
    // The stack here is air | A | B | air with both sheets 6 mm, so mtl_current bounds both sides, near == far, the
    // flux conversion is identically 1, and the displaced host is 12 mm of air: the composite is a pure two-layer
    // matrix with nothing else multiplying it. With mtl_current at 0 no host leg is closed and the state words are
    // untouched, so the written field is the composite.
    //
    // Both orders are run. Bit 1 of the hit code carries the traversal order ray_mesh_interact derived from the face
    // normals, the layer hit on its front side being traversed first, with bit 3 recording that the order fell back to
    // the geometric FBS -> SBS sequence.
    //
    // Transmission is order-independent by reciprocity and only the reflect port separates the two, which is asserted
    // on the reference as well as on the solver, so that the order can be seen to matter at all.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0);
    const std::complex<double> e_air(1.0, 0.0);

    const uint8_t OT_STACK_FBS = 71; // ok | front | co-located | sheet: FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // ok | back  | co-located | sheet: SBS layer traversed first

    const int A = 3, B = 7; // SHEET eps 5.24 - 0.63j and LO_SHEET eps 1.50 - 0.40j, both 6 mm
    const std::complex<double> e_A = diag_eps(A, f_hz), e_B = diag_eps(B, f_hz);

    struct Ord
    {
        const char *name;
        uint8_t otype;
        int m1, m2;
        std::complex<double> first, second;
    };
    const std::vector<Ord> orders = {
        {"A|B", OT_STACK_FBS, A, B, e_A, e_B},
        {"B|A", OT_STACK_SBS, A, B, e_B, e_A},
    };

    QD_DIAG("\n# diag: a stack of two sheets, air | 6 mm | 6 mm | air, sheets eps 5.24 - 0.63j and\n");
    QD_DIAG("# eps 1.50 - 0.40j, 1 GHz. ref_* is the two-layer characteristic matrix, out_* is what\n");
    QD_DIAG("# the stack resolve wrote.\n");
    QD_DIAG("order,theta_n,ref_|T|,ref_|R|,out_|T|,out_gain_T,out_gain_R,dT,dR,rtype_T,rtype_R\n");

    for (const auto &O : orders)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);

            // near == far == air, so the conversion is 1 and the displaced host is air over the whole 12 mm, which the
            // displacement budget leaves uncapped for an air host.
            DiagSheet Xs = diag_cm_stack(e_air, U, {{O.first, U, thk}, {O.second, U, thk}},
                                         e_air, U, ci, f_hz, false, /*scalar*/ true);
            DiagSheet Xe = diag_cm_stack(e_air, U, {{O.first, U, thk}, {O.second, U, thk}},
                                         e_air, U, ci, f_hz, false, /*EM*/ false);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, O.otype, 2u,
                                       /*M1*/ O.m1, /*M2*/ O.m2,
                                       /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };

            auto Ct = run(4); // scalar transmission
            auto Ce = run(1); // EM transmission
            auto Cr = run(3); // scalar reflection
            auto Cx = run(0); // EM reflection

            double gT = 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
            double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));

            double dT = std::abs(cpx(Ct.xprmat, 0, 0) - Xs.T_te);
            double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u,%u\n",
                    O.name, thd, std::abs(Xs.T_te), std::abs(Xs.R_te),
                    std::abs(cpx(Ct.xprmat, 0, 0)), (double)Ce.gain(0), (double)Cx.gain(0),
                    dT, dR, (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO("order " << O.name << ", theta_n " << thd);

            // The reference carries real energy on both ports, so a zero output is a dropped ray
            // and not a genuinely dark stack.
            CHECK(gT > 1.0e-3);
            CHECK(gR > 1.0e-3);

            // The stack resolves, both ports, scalar and EM.
            CHECK(Ct.resolved_type[0] != 0);
            CHECK(Cr.resolved_type[0] != 0);
            CHECK(dT < 1.0e-9);
            CHECK(dR < 1.0e-9);
            CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
            CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);
        }

    // The order matters, and only in reflection. Asserted on the reference alone, so a core that
    // lost the traversal order fails here rather than passing a transmission-only check.
    for (double thd : {0.0, 45.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet AB = diag_cm_stack(e_air, U, {{e_A, U, thk}, {e_B, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        DiagSheet BA = diag_cm_stack(e_air, U, {{e_B, U, thk}, {e_A, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        INFO("theta_n " << thd);
        CHECK(std::abs(AB.T_te - BA.T_te) < 1.0e-14);
        CHECK(std::abs(AB.T_tm - BA.T_tm) < 1.0e-14);
        CHECK(std::abs(AB.R_te - BA.R_te) > 1.0e-3);
    }
}

TEST_CASE("Lumped columns - interface gain on the sheet, glued and embedded paths", "[wave]")
{
    // The interface gains on a sheet: sqrt(G_sheet * (far is new ? G_far : 1)) on the transmit port. An interface gain
    // is a penetration loss owed once per medium entered, which is what interact_with charges for a single
    // transmission-class crossing: the entered medium's, never the one being left. A sheet crossing enters the sheet,
    // so it owes the sheet's; it enters the far medium as well only if the wave was not already in it:
    //
    //   isolated   air  | sheet | air     G_sheet             air owes nothing
    //   entry      air  | sheet | wall    G_sheet * G_wall    the wall is entered here and nowhere else
    //   embedded   host | sheet | host    G_sheet             the host was entered earlier and paid then
    //   exit       host | sheet | air     G_sheet             leaving, and air owes nothing
    //
    // The embedded row is the one the gate exists for: testing for air instead of for near == far would charge the host
    // a second time at every sheet inside it, which the row makes cost the full 5 dB of the host's column. The
    // reference does not copy the gate: DiagLump::g_far is set by the row to what the far medium is owed.
    //
    // The reflect port owes nothing for the part that turns around at the near face; what entered the sheet and came
    // back out pays the sheet's gain once, as the echo of a traced solid does.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    const int SH = 1, SH0 = 2, WALL = 3, HOST = 4, HOST0 = 5;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk, /*att*/ 3.0}, // 1 SH     the sheet, 3 dB
        {5.24, 0.63, thk, 0.0},         // 2 SH0    its twin without the column
        {1.50, 0.40, 0.0, 5.0},         // 3 WALL   a lossy solid, 5 dB
        {1.50, 0.00, 0.0, 5.0},         // 4 HOST   a lossless solid, 5 dB
        {1.50, 0.00, 0.0, 0.0},         // 5 HOST0  its twin without the column
    };
    auto mtl = lump_palette(pal);
    auto E = [&](int idx)
    { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };

    const double G_sh = std::pow(10.0, -0.1 * 3.0), G_wall = std::pow(10.0, -0.1 * 5.0);

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2, cur, near, far;
        bool disp_far;
        double g_far; // what the far medium is owed
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air  | SH | air ", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 1.0, 85.0},
        {"entry     air  | SH | WALL", OT_GLUED_ENTRY, 2u, WALL, 0, 0, WALL, false, G_wall, 85.0},
        {"embedded  HOST | SH | HOST", OT_SHEET_ISO, 1u, 0, HOST, HOST, HOST, false, 1.0, 50.0},
        {"exit      HOST | SH | air ", OT_GLUED_EXIT, 2u, HOST, HOST, HOST, 0, true, 1.0, 50.0},
    };

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };
    auto P = [](std::complex<double> v)
    { return std::norm(v); };

    QD_DIAG("\n# diag: lumped interface gain, sheet eps 5.24 - 0.63j att 3 dB, wall and host att 5 dB\n");
    QD_DIAG("# paid_dB is the gain the composite carries against the same stack with the sheet's att\n");
    QD_DIAG("# removed; owed_dB is what the table in the case header says the crossing owes in total\n");
    QD_DIAG("stack,theta_n,P_te_meas,P_te_ref,P_tm_meas,P_tm_ref,d_te,d_tm,paid_dB,owed_dB,rtype\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 20.0, 40.0, 50.0, 70.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto run = [&](int sheet, int m2, int cur, int itype)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH, sheet, m2, 0, cur, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto C = run(SH, R.m2, R.cur, 1);
            auto C0 = run(SH0, R.m2, R.cur, 1); // the same crossing, sheet without the column

            DiagLump Lp;
            Lp.g_slab = G_sh, Lp.g_far = R.g_far;
            DiagSheet D = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz, R.disp_far,
                                               false, 1.0, 1.0, 1.0, Lp);

            // Hosts are lossless here, so the leg a row inside a medium closes out contributes a
            // pure phase and the channel powers are the composite's own.
            double p_te = P(TE(C.xprmat)), p_tm = P(TM(C.xprmat));
            double paid = 10.0 * std::log10(p_te / P(TE(C0.xprmat)));
            double owed = 10.0 * std::log10(G_sh * R.g_far);

            QD_DIAG("%s,%.1f,%.9e,%.9e,%.9e,%.9e,%.2e,%.2e,%+.6f,%+.6f,%u\n", R.name, thd,
                    p_te, P(D.T_te), p_tm, P(D.T_tm), p_te - P(D.T_te), p_tm - P(D.T_tm),
                    paid, owed, (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0);

            // Each gain owed is paid, once.
            CHECK(std::abs(p_te - P(D.T_te)) < 1.0e-9);
            CHECK(std::abs(p_tm - P(D.T_tm)) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (P(D.T_te) + P(D.T_tm))) < 1.0e-9);
            if (R.cur == 0) // nothing else multiplies the composite: pin it complex
            {
                CHECK(std::abs(TE(C.xprmat) - D.T_te) < 1.0e-9);
                CHECK(std::abs(TM(C.xprmat) - D.T_tm) < 1.0e-9);
            }

            // Exact by construction: the sheet's gain is one real factor on the transmit port, so
            // the field with and without it differ by sqrt(G_sheet) and by nothing else.
            CHECK(std::abs(p_te / P(TE(C0.xprmat)) - G_sh) < 1.0e-14);
            CHECK(std::abs(p_tm / P(TM(C0.xprmat)) - G_sh) < 1.0e-14);

            // And the assertion above cannot pass with a gain dropped or paid twice: each is worth
            // dB, not rounding.
            CHECK(std::abs(p_te - P(D.T_te) / G_sh) > 1.0e-3 * p_te);
            CHECK(std::abs(p_te - P(D.T_te) * G_sh) > 1.0e-3 * p_te);

            if (R.near == HOST && R.far == HOST)
            {
                // The embedded sheet does not pay the host's gain. Stated twice: against the reference, where charging
                // it would cost the full 5 dB, and against the host's twin without the column, exact by construction:
                // with near == far the host's att column is never read.
                auto Ch = run(SH, 0, HOST0, 1);
                CHECK(std::abs(p_te - P(D.T_te) * G_wall) > 0.5 * p_te);
                CHECK(p_te == P(TE(Ch.xprmat)));
                CHECK(p_tm == P(TM(Ch.xprmat)));
            }
            if (R.far == WALL)
            {
                // The glued entry pays the wall's, which no other event on the path will: without it the wall's
                // penetration loss would disappear for every wall that carries a coating.
                CHECK(std::abs(p_te - P(D.T_te) / G_wall) > 0.5 * p_te);
            }

            // The reflect port carries the sheet's gain on the part that entered the sheet, once,
            // and never the wall's. Complex on the rows where the composite is the whole of what
            // is written; and told apart from the sheet without the column.
            if (R.cur == 0)
            {
                auto Cr = run(SH, R.m2, R.cur, 0);
                auto Cr0 = run(SH0, R.m2, R.cur, 0);
                CHECK(std::abs(TE(Cr.xprmat) - D.R_te) < 1.0e-9);
                CHECK(std::abs(TM(Cr.xprmat) - D.R_tm) < 1.0e-9);
                CHECK(std::abs(TE(Cr.xprmat) - TE(Cr0.xprmat)) > 1.0e-3 * std::abs(TE(Cr0.xprmat)));
                CHECK(std::abs(TM(Cr.xprmat) - TM(Cr0.xprmat)) > 1.0e-3 * std::abs(TM(Cr0.xprmat)));
            }
        }
}

TEST_CASE("Lumped columns - the transmission factor on a sheet, both ports", "[wave]")
{
    // tf at the faces of a sheet. Each face is the ideal transformer of the sheet-tf case; between real admittances
    // that is, per boundary and per channel, with R0 the raw Fresnel reflectance and R' the pair rule on it:
    //
    //     r' = sqrt(R') exp(j arg r)        t' = t sqrt((1 - R') / (1 - R0))
    //
    // so the reflectance is replaced, the transmission keeps its Fresnel phase, and its power moves by exactly what the
    // reflection gave up or took. tf > 0 leaks, R' = R0 (1 - tf), which scales |r| by sqrt(1 - tf) at every angle but
    // scales t by an amount that grows with R0; tf < 0 forces, R' = R0 + (1 - R0)|tf|, which scales t by sqrt(1 - |tf|)
    // at every angle. Both faces take the sheet's tf.
    //
    // tf is not a boundary admittance, so the characteristic matrix cannot hold it. The Airy pair can, because it is
    // written in the per-boundary r and t that tf rescales, and diag_sheet_composite is in that form; the reference is
    // the same convention written a second time. Two statements below are independent of it.
    //
    //   Unitarity. In flux-normalized amplitudes a lossless boundary below its critical angle is [[r, tau], [tau, -r]]
    //   with r and tau real and r^2 + tau^2 = 1. The rescale keeps both phases and the sum, so the boundary stays
    //   unitary, and a lossless sheet between two unitary boundaries conserves energy per channel at any tf.
    //
    //   The end points. tf = +1 removes the reflection, so the reflect port is exactly zero and a lossless sheet is a
    //   pure phase; tf = -1 makes both faces mirrors, so the transmit port is exactly zero.
    //
    // A tf-bearing pair, sheet and solid both, is the pair rule: each face is owed by the two media that form it. The
    // last block asserts a wall's tf under a coating against the reference carrying the wall's column; the full sweep
    // is in the neighbour-tf case below.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SH = 1, WALL = 2, HOST = 3, SHL = 4;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2, cur, near, far;
        bool disp_far;
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air  | SH | air ", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 85.0},
        {"entry     air  | SH | WALL", OT_GLUED_ENTRY, 2u, WALL, 0, 0, WALL, false, 85.0},
        {"exit      HOST | SH | air ", OT_GLUED_EXIT, 2u, HOST, HOST, HOST, 0, true, 50.0},
    };

    QD_DIAG("\n# diag: transmission factor on a 6 mm sheet, eps 5.24 - 0.63j, both ports\n");
    QD_DIAG("# dT_dB and dR_dB are against the same crossing at tf = 0, from the reference\n");
    QD_DIAG("stack,tf,theta_n,|T_te|,|T_tm|,|R_te|,|R_tm|,d_T,d_R,d_scalar,dT_dB,dR_dB\n");

    double worst_T = 0.0, worst_R = 0.0, worst_E = 0.0;
    for (double tf : {-1.0, -0.5, -0.25, 0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, tf}, // 1 SH    the sheet
            {1.50, 0.40, 0.0},          // 2 WALL  a lossy solid
            {1.50, 0.00, 0.0},          // 3 HOST  a lossless solid
            {5.24, 0.00, thk, 0.0, tf}, // 4 SHL   a lossless sheet with the same tf
        };
        auto mtl = lump_palette(pal);
        auto E = [&](int idx)
        { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };

        DiagLump Lp;
        Lp.tf = tf;

        for (const auto &R : rows)
            for (double thd : {0.0, 20.0, 45.0, 50.0, 70.0, 85.0})
            {
                if (thd > R.lim)
                    continue;
                double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

                auto run = [&](int itype)
                {
                    auto C = make1<double>(mtl, itype, R.otype, R.nH, SH, R.m2, 0, R.cur, 0,
                                           L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                    C.run();
                    return C;
                };
                auto Ct = run(1), Cr = run(0);

                DiagSheet D = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                   R.disp_far, false, 1.0, 1.0, 1.0, Lp);
                DiagSheet D0 = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                    R.disp_far, false);

                std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);
                std::complex<double> r_te = TE(Cr.xprmat), r_tm = TM(Cr.xprmat);

                // In air nothing else multiplies either port and both are compared complex. Inside
                // a host the transmit port picks up the phase of the leg it closes and the reflect
                // port the host's own bookkeeping, so that row is compared in transmitted power.
                double d_T, d_R = 0.0, d_S = 0.0;
                if (R.cur == 0)
                {
                    d_T = std::max(std::abs(t_te - D.T_te), std::abs(t_tm - D.T_tm));
                    d_R = std::max(std::abs(r_te - D.R_te), std::abs(r_tm - D.R_tm));

                    // Scalar mode carries one channel, the TE / admittance form.
                    DiagSheet Ds = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                        R.disp_far, true, 1.0, 1.0, 1.0, Lp);
                    auto St = run(4), Sr = run(3);
                    d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te),
                                   std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
                }
                else
                    d_T = std::max(std::abs(std::norm(t_te) - std::norm(D.T_te)),
                                   std::abs(std::norm(t_tm) - std::norm(D.T_tm)));

                worst_T = std::max(worst_T, d_T), worst_R = std::max(worst_R, std::max(d_R, d_S));

                double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
                double gT0 = 0.5 * (std::norm(D0.T_te) + std::norm(D0.T_tm));
                double gR = 0.5 * (std::norm(D.R_te) + std::norm(D.R_tm));
                double gR0 = 0.5 * (std::norm(D0.R_te) + std::norm(D0.R_tm));

                QD_DIAG("%s,%+.2f,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%.2e,%+.4f,%+.4f\n",
                        R.name, tf, thd, std::abs(t_te), std::abs(t_tm), std::abs(r_te), std::abs(r_tm),
                        d_T, d_R, d_S,
                        (gT > 0.0) ? 10.0 * std::log10(gT / gT0) : -999.0,
                        (gR > 0.0) ? 10.0 * std::log10(gR / gR0) : -999.0);

                INFO(R.name << ", tf " << tf << ", theta_n " << thd);
                REQUIRE(std::isfinite(Ct.gain(0)));
                REQUIRE(std::isfinite(Cr.gain(0)));

                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(d_S < 1.0e-9);
                CHECK(std::abs((double)Ct.gain(0) - gT) < 1.0e-9);

                // tf moves both ports by far more than they are asserted to, so the assertions
                // above are not passing on a rescale that never ran. The transmit port moves least
                // for a weak leak into a glued solid, 0.076 dB at tf = 0.25, and the reflect port
                // least for a weak forced reflection at grazing incidence, 0.87 dB at tf = -0.25.
                CHECK(std::abs(gT - gT0) > 1.0e-3 * gT0);
                CHECK(std::abs(gR - gR0) > 1.0e-3 * gR0);

                if (R.cur == 0)
                {
                    // The end points. tf = -1 closes the transmit port exactly. tf = +1 closes the
                    // reflect port exactly where the two sides of a face are in phase, which the
                    // sheet-tf case asserts; this sheet is lossy and leaves -35 dB in air and, on
                    // the lossy wall, -24 dB up to 50 deg and -18.7 dB at 85 deg.
                    if (tf == 1.0)
                        CHECK((double)Cr.gain(0) < 2.0e-2);
                    if (tf == -1.0)
                        CHECK((double)Ct.gain(0) == 0.0);
                }
            }

        // Unitarity, on a lossless sheet in air. No reference enters: this is a statement about
        // the rule itself, and it is the reason the rescale has the form it has.
        for (double thd : {0.0, 30.0, 60.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0;
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_SHEET_ISO, 1u, SHL, 0, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), Cr = run(0);
            double e_te = std::norm(TE(Ct.xprmat)) + std::norm(TE(Cr.xprmat));
            double e_tm = std::norm(TM(Ct.xprmat)) + std::norm(TM(Cr.xprmat));
            worst_E = std::max(worst_E, std::max(std::abs(e_te - 1.0), std::abs(e_tm - 1.0)));

            INFO("lossless sheet, tf " << tf << ", theta_n " << thd);
            CHECK(std::abs(e_te - 1.0) < 1.0e-12);
            CHECK(std::abs(e_tm - 1.0) < 1.0e-12);
            if (tf == 1.0) // no reflection at either face: the sheet is a pure phase
                CHECK(std::abs(std::abs(TE(Ct.xprmat)) - 1.0) < 1.0e-12);
        }
    }
    QD_DIAG("# worst: transmit %.3e, reflect and scalar %.3e, energy of a lossless sheet %.3e\n",
            worst_T, worst_R, worst_E);

    // The far medium's tf. A glued entry into a wall that carries one, with a sheet that does not, against the same
    // entry with the wall's column removed. The wall's boundary is sheet | wall rather than air | wall, so the column
    // is not worth what it is worth on the bare wall, and a forced far face adds a series against the sheet's front
    // face. Asserted against the reference. bare_dB is what the same column is worth on the uncoated wall, from the
    // rule interact_with applies to a single crossing: the TE/TM-averaged reflectance R becomes R (1 - tf) for a leak
    // and R + (1 - R)|tf| for a forced reflection, and the transmitted power moves by (1 - R') / (1 - R).
    QD_DIAG("\n# diag: the far medium's tf on a glued entry, air | sheet (tf 0) | wall (tf)\n");
    QD_DIAG("# composite_dB is what the wall's column moves the coated entry by, bare_dB what it\n");
    QD_DIAG("# moves the uncoated wall by\n");
    QD_DIAG("wall_tf,theta_n,composite_dB,bare_dB\n");
    for (double wtf : {0.5, -0.5})
        for (double thd : {0.0, 45.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            double g[2];
            for (int k = 0; k < 2; ++k)
            {
                const std::vector<LumpMtl> pal = {
                    {5.24, 0.63, thk},                       // 1 the sheet, no tf
                    {1.50, 0.40, 0.0, 0.0, (k ? wtf : 0.0)}, // 2 the wall
                };
                auto mtl = lump_palette(pal);
                auto C = make1<double>(mtl, 1, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                g[k] = (double)C.gain(0);
            }

            std::complex<double> ew(1.50, -0.40), zw = std::sqrt(ew);
            std::complex<double> cw = std::sqrt(1.0 - (1.0 / ew) * (1.0 - ct * ct));
            double Rb = 0.5 * (std::norm((ct - zw * cw) / (ct + zw * cw)) +
                               std::norm((zw * ct - cw) / (zw * ct + cw)));
            double Rb_tf = (wtf >= 0.0) ? Rb * (1.0 - wtf) : Rb + (1.0 - Rb) * (-wtf);

            QD_DIAG("%+.1f,%.1f,%+.6f,%+.6f\n", wtf, thd, 10.0 * std::log10(g[1] / g[0]),
                    10.0 * std::log10((1.0 - Rb_tf) / (1.0 - Rb)));
            CHECK(std::isfinite(g[1]));
            CHECK(g[1] > 0.0);

            double gr[2];
            for (int k = 0; k < 2; ++k)
            {
                DiagLump Lw;
                Lw.tf_far = k ? wtf : 0.0;
                DiagSheet Dw = diag_sheet_composite(1.0, std::complex<double>(5.24, -0.63), ew, thk, ct, f_hz,
                                                    false, false, 1.0, 1.0, 1.0, Lw);
                gr[k] = 0.5 * (std::norm(Dw.T_te) + std::norm(Dw.T_tm));
            }
            INFO("coated wall, wall tf " << wtf << ", theta_n " << thd);
            CHECK(std::abs(10.0 * std::log10(g[1] / g[0]) - 10.0 * std::log10(gr[1] / gr[0])) < 1.0e-6);
            CHECK(std::abs(g[1] - g[0]) > 1.0e-3 * g[0]); // and the wall's column is read at all
        }
}

TEST_CASE("Lumped columns - a permittivity resonance is part of the medium on the sheet paths", "[wave]")
{
    // A resonance added at every interface quantity - the admittances and the cosines - and left out of every
    // propagation index is not a wave any medium supports at oblique incidence, and with a lossy pole it is not
    // passive. A Lorentz pole is a dispersive dielectric, one medium of eps + res at its faces and inside it, which is
    // what the library and the references carry.
    //
    // Three kinds of statement.
    //
    //   Against the reference with the pole in the permittivity, 1e-9, complex wherever nothing else multiplies the
    //   composite: a resonant sheet in air, a resonant sheet glued to a plain solid, a plain sheet glued to a resonant
    //   solid, and a resonant sheet inside a resonant host. The host of the last row is lossy at 1 GHz through its
    //   pole, so the leg the event closes is a decay, compared in power with the decay written from the medium.
    //
    //   Against the resonance dropped, as a margin.
    //
    //   Without a reference at all. A lossless sheet of eps 4 whose pole takes it to exactly 1 is a sheet of air: no
    //   reflection and a transmission of one with no phase, at every angle and in both channels. A pole at the faces
    //   only would leave it exp(-j k0 thk cos(theta) (2 - 1)), which is printed beside it.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;
    const double k0 = 2.0 * QPI * f_hz / C0;

    const int SHR = 1, WALL = 2, WALLR = 3, HOSTR = 4, SH = 5, MATCH = 6;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk, 0.0, 0.0, 1.2, 5.0, 0.8},       // 1 SHR    resonant sheet
        {1.50, 0.40, 0.0},                                // 2 WALL   plain lossy solid
        {1.50, 0.40, 0.0, 0.0, 0.0, 0.8, 4.0, 0.5},       // 3 WALLR  resonant lossy solid
        {1.50, 0.00, 0.0, 0.0, 0.0, 0.8, 4.0, 0.5},       // 4 HOSTR  resonant solid, lossless index
        {5.24, 0.63, thk},                                // 5 SH     plain sheet
        {4.00, 0.00, thk, 0.0, 0.0, 2.0, 1.0e300, -2.25}, // 6 MATCH  eps 4, interface eps exactly 1
    };
    auto mtl = lump_palette(pal);
    const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
    auto E = [&](int idx)
    { return (idx == 0) ? one : lump_eps(pal[(size_t)idx - 1], f_hz); };
    auto Rz = [&](int idx)
    { return (idx == 0) ? zero : lump_res(pal[(size_t)idx - 1], f_hz); };

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int sheet, m2, cur, near, far;
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air   | SHR | air  ", OT_SHEET_ISO, 1u, SHR, 0, 0, 0, 0, 85.0},
        {"entry     air   | SHR | WALL ", OT_GLUED_ENTRY, 2u, SHR, WALL, 0, 0, WALL, 85.0},
        {"entry     air   | SH  | WALLR", OT_GLUED_ENTRY, 2u, SH, WALLR, 0, 0, WALLR, 85.0},
        {"embedded  HOSTR | SHR | HOSTR", OT_SHEET_ISO, 1u, SHR, 0, HOSTR, HOSTR, HOSTR, 50.0},
    };

    QD_DIAG("\n# diag: permittivity resonance on the sheet paths. sheet res (%.4f, %.4f), solid res\n",
            std::real(Rz(SHR)), std::imag(Rz(SHR)));
    QD_DIAG("# (%.4f, %.4f) at 1 GHz. d_ref is against the reference with the pole in the permittivity,\n",
            std::real(Rz(WALLR)), std::imag(Rz(WALLR)));
    QD_DIAG("# the embedded row in power with its host leg; sep_dropped is how far the same field sits from\n");
    QD_DIAG("# a reference with the resonance removed. leg_dB: the host leg's decay, dB\n");
    QD_DIAG("stack,theta_n,|T_te|,|T_tm|,d_ref,d_refl,d_scalar,sep_dropped,leg_dB,rtype\n");

    double worst = 0.0;
    for (const auto &R : rows)
        for (double thd : {0.0, 20.0, 40.0, 50.0, 70.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH, R.sheet, R.m2, 0, R.cur, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1);

            DiagLump Lp;
            Lp.res_near = Rz(R.near), Lp.res_slab = Rz(R.sheet), Lp.res_far = Rz(R.far);

            // The pole in the permittivity, and the resonance dropped.
            DiagSheet D = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz, false,
                                               false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Dd = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz, false,
                                                false);

            std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);

            // The embedded row's event closes the host leg of L, which the composite does not
            // carry. Its phase is the host's; its decay is the ITU-R P.2040 rate 8.686 k0 |Im n|
            // dB per meter along the leg, k0 through the constant c / (2 pi 1e9) = 0.0477135 m GHz
            // as the library rounds it. Compared in power there.
            const double leg = (R.cur == 0) ? 1.0
                                            : std::pow(10.0, -0.1 * 8.686 * (f_hz * 1e-9 / 0.0477135) *
                                                                 std::abs(std::imag(std::sqrt(E(R.cur) + Rz(R.cur)))) * L);
            auto dist = [&](const DiagSheet &X)
            {
                if (R.cur == 0)
                    return std::max(std::abs(t_te - X.T_te), std::abs(t_tm - X.T_tm));
                return std::max(std::abs(std::norm(t_te) - leg * std::norm(X.T_te)),
                                std::abs(std::norm(t_tm) - leg * std::norm(X.T_tm)));
            };
            double d_ref = dist(D), sep_d = dist(Dd);

            double d_refl = 0.0, d_scalar = 0.0;
            if (R.cur == 0)
            {
                auto Cr = run(0);
                d_refl = std::max(std::abs(TE(Cr.xprmat) - D.R_te), std::abs(TM(Cr.xprmat) - D.R_tm));

                DiagSheet Ds = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz,
                                                    false, true, 1.0, 1.0, 1.0, Lp);
                auto St = run(4), Sr = run(3);
                d_scalar = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te),
                                    std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
            }
            worst = std::max(worst, std::max(d_ref, std::max(d_refl, d_scalar)));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.2e,%.2e,%.2e,%.3e,%.4f,%u\n", R.name, thd,
                    std::abs(t_te), std::abs(t_tm), d_ref, d_refl, d_scalar, sep_d, 10.0 * std::log10(leg),
                    (unsigned)Ct.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(Ct.gain(0)));
            REQUIRE(Ct.resolved_type[0] != 0);
            CHECK(d_ref < 1.0e-9);
            CHECK(d_refl < 1.0e-9);
            CHECK(d_scalar < 1.0e-9);
            CHECK(std::abs((double)Ct.gain(0) - leg * 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm))) < 1.0e-9);

            // The resonance dropped is excluded by a margin six orders above the bound, on every row.
            CHECK(sep_d > 1.0e-3);
        }
    QD_DIAG("# worst against the reference %.3e\n", worst);

    // The sheet whose pole takes it to air.
    QD_DIAG("\n# diag: resonance-matched sheet, eps 4 with a pole, eps + res (%.17g, %.1e), thk 6 mm.\n",
            std::real(E(MATCH) + Rz(MATCH)), std::imag(E(MATCH) + Rz(MATCH)));
    QD_DIAG("# A sheet of air: no reflection, T = 1. split_deg is the phase the interface / index split gave it,\n");
    QD_DIAG("# -k0 thk cos(theta) (2 - 1), recorded\n");
    QD_DIAG("theta_n,|R_te|,|R_tm|,|T-1|,arg_T_deg,split_deg\n");

    for (double thd : {0.0, 30.0, 60.0, 85.0})
    {
        double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
        auto run = [&](int itype)
        {
            auto C = make1<double>(mtl, itype, OT_SHEET_ISO, 1u, MATCH, 0, 0, 0, 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();
            return C;
        };
        auto Ct = run(1), Cr = run(0);

        std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);
        QD_DIAG("%.1f,%.3e,%.3e,%.3e,%.9f,%.9f\n", thd, std::abs(TE(Cr.xprmat)), std::abs(TM(Cr.xprmat)),
                std::abs(t_te - 1.0), std::arg(t_te) * 180.0 / QPI, -k0 * thk * ct * 180.0 / QPI);

        INFO("resonance-matched sheet, theta_n " << thd);
        CHECK(std::abs(TE(Cr.xprmat)) < 1.0e-12);
        CHECK(std::abs(TM(Cr.xprmat)) < 1.0e-12);
        CHECK(std::abs(t_te - 1.0) < 1.0e-12);
        CHECK(std::abs(t_tm - 1.0) < 1.0e-12);
    }
}

TEST_CASE("Neighbour tf - each face of a sheet is owed by the pair of media that form it", "[wave]")
{
    // Each face of a sheet is owed by the pair of media that form it, as a bare boundary is (interact_with combines its
    // two media through apply_tf_pair), per channel:
    //
    //     near face   pair(tf_near, tf_sheet)        far face   pair(tf_sheet, tf_far)
    //
    // It reduces to the sheet's own tf, bit for bit, wherever the neighbour carries none; those rows are the control.
    //
    // The limit. A sheet whose permittivity is its host's is not a boundary on the host side, so an air-like sheet on a
    // tf-bearing wall is the bare tf-bearing wall: asserted against the bare boundary written from interact_with's
    // rule, the transformer of a face with a wave behind it, complex in scalar mode and per channel in EM, for a
    // lossless and for a lossy wall.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SH = 1, WALL = 2, HOST = 3;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Combo
    {
        double tf_s, tf_n;
    };
    const std::vector<Combo> combos = {{0.0, 0.0}, {0.25, 0.0}, {-0.5, 0.0}, {0.0, 0.5}, {0.0, -0.5}, {0.25, 0.5}, {-0.25, 0.5}, {0.25, -0.5}, {-0.5, -0.25}};

    QD_DIAG("\n# diag: neighbour tf by the pair rule, 6 mm sheet eps 5.24 - 0.63j. entry is\n");
    QD_DIAG("# air | SH | WALL (eps 1.50 - 0.40j, tf_n), exit is HOST (eps 1.50, tf_n) | SH | air.\n");
    QD_DIAG("# nb_dB is what the neighbour's column is worth on the transmit gain, from the reference\n");
    QD_DIAG("stack,tf_s,tf_n,theta_n,d_T,d_R,d_scalar,nb_dB\n");

    double worst_ctl = 0.0, worst_nb = 0.0;
    for (const auto &K : combos)
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, K.tf_s}, // 1 SH
            {1.50, 0.40, 0.0, 0.0, K.tf_n}, // 2 WALL  lossy solid carrying the neighbour's tf
            {1.50, 0.00, 0.0, 0.0, K.tf_n}, // 3 HOST  lossless solid carrying the neighbour's tf
        };
        auto mtl = lump_palette(pal);
        auto E = [&](int idx)
        { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };
        bool control = (K.tf_n == 0.0);

        // Glued entry, complex on both ports: in air nothing else multiplies the composite.
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, SH, WALL, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            DiagLump Lp, Lo;
            Lp.tf = K.tf_s, Lp.tf_far = K.tf_n;
            Lo.tf = K.tf_s; // the sheet as sole owner
            DiagSheet D = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Ds = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, true, 1.0, 1.0, 1.0, Lp);
            DiagSheet Do = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, false, 1.0, 1.0, 1.0, Lo);

            auto Ct = run(1), Cr = run(0), St = run(4), Sr = run(3);
            double d_T = std::max(std::abs(TE(Ct.xprmat) - D.T_te), std::abs(TM(Ct.xprmat) - D.T_tm));
            double d_R = std::max(std::abs(TE(Cr.xprmat) - D.R_te), std::abs(TM(Cr.xprmat) - D.R_tm));
            double d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
            double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
            double gO = 0.5 * (std::norm(Do.T_te) + std::norm(Do.T_tm));
            double worst = std::max(d_T, std::max(d_R, d_S));
            (control ? worst_ctl : worst_nb) = std::max(control ? worst_ctl : worst_nb, worst);

            QD_DIAG("entry,%+.2f,%+.2f,%.1f,%.2e,%.2e,%.2e,%+.4f\n", K.tf_s, K.tf_n, thd, d_T, d_R, d_S,
                    (gT > 0.0 && gO > 0.0) ? 10.0 * std::log10(gT / gO) : -999.0);
            INFO("entry, tf_s " << K.tf_s << ", tf_n " << K.tf_n << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
            CHECK(d_S < 1.0e-9);
        }

        // Glued exit, reached from inside a lossless host. The host leg multiplies both ports by a
        // pure phase, so they are compared in power per channel.
        for (double thd : {0.0, 20.0, 45.0, 50.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SH, HOST, 0, HOST, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            DiagLump Lp, Lo;
            Lp.tf = K.tf_s, Lp.tf_near = K.tf_n;
            Lo.tf = K.tf_s;
            DiagSheet D = diag_sheet_composite(E(HOST), E(SH), 1.0, thk, ct, f_hz, true, false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Do = diag_sheet_composite(E(HOST), E(SH), 1.0, thk, ct, f_hz, true, false, 1.0, 1.0, 1.0, Lo);

            auto Ct = run(1), Cr = run(0);
            double d_T = std::max(std::abs(std::norm(TE(Ct.xprmat)) - std::norm(D.T_te)),
                                  std::abs(std::norm(TM(Ct.xprmat)) - std::norm(D.T_tm)));
            double d_R = std::max(std::abs(std::norm(TE(Cr.xprmat)) - std::norm(D.R_te)),
                                  std::abs(std::norm(TM(Cr.xprmat)) - std::norm(D.R_tm)));
            double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
            double gO = 0.5 * (std::norm(Do.T_te) + std::norm(Do.T_tm));
            (control ? worst_ctl : worst_nb) = std::max(control ? worst_ctl : worst_nb, std::max(d_T, d_R));

            QD_DIAG("exit ,%+.2f,%+.2f,%.1f,%.2e,%.2e,%.2e,%+.4f\n", K.tf_s, K.tf_n, thd, d_T, d_R, 0.0,
                    (gT > 0.0 && gO > 0.0) ? 10.0 * std::log10(gT / gO) : -999.0);
            INFO("exit, tf_s " << K.tf_s << ", tf_n " << K.tf_n << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
        }
    }
    QD_DIAG("# worst: control rows %.3e, rows with a tf-bearing neighbour %.3e\n", worst_ctl, worst_nb);

    // The limit: an air-like sheet on a tf-bearing wall against the bare tf-bearing wall. The
    // face of the composite and the bare boundary are one rule, the transformer of a face with a
    // wave behind it, per channel, for a lossless and for a lossy wall.
    QD_DIAG("\n# diag: air | sheet eps 1 (6 mm) | wall eps 1.50 - j ei (tf) against the bare wall, the transformer per channel\n");
    QD_DIAG("wall_ei,wall_tf,theta_n,d_scalar_T,d_scalar_R,d_gain_T,d_gain_R,d_chan,moved_dB\n");
    for (double wei : {0.0, 0.40})
        for (double wtf : {0.5, -0.5})
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                const std::vector<LumpMtl> pal = {
                    {1.00, 0.00, thk},          // 1 a sheet of the host's own permittivity
                    {1.50, wei, 0.0, 0.0, wtf}, // 2 the wall
                    {1.50, wei, 0.0, 0.0, 0.0}, // 3 the wall without its column
                };
                auto mtl = lump_palette(pal);
                double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
                auto run = [&](int itype, int wall)
                {
                    auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, 1, wall, 0, 0, 0,
                                           L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                    C.run();
                    return C;
                };

                // The bare boundary air | wall, from the rule interact_with applies: the transformer of an open face, per
                // channel
                std::complex<double> ew = lump_eps(pal[1], f_hz), zw = std::sqrt(ew);
                std::complex<double> cw = std::sqrt(1.0 - (1.0 / ew) * (1.0 - ct * ct));
                std::complex<double> r_te = (ct - zw * cw) / (ct + zw * cw), r_tm = (zw * ct - cw) / (zw * ct + cw);
                std::complex<double> t_te = 2.0 * ct / (ct + zw * cw), t_tm = 2.0 * ct / (zw * ct + cw);
                double s_te = std::sqrt(std::real(zw * cw) / ct), s_tm = std::sqrt(std::real(std::conj(zw) * cw) / ct);
                std::complex<double> q_te = r_te, q_tm = r_tm;
                std::complex<double> c_te = diag_tf_face(q_te, 0.0, wtf), c_tm = diag_tf_face(q_tm, 0.0, wtf);

                // The layer lies in the air in front of the wall, so the wall is on the mesh plane and
                // the reflect port is the bare wall's, phase included
                std::complex<double> back(1.0, 0.0);
                std::complex<double> Ts_ref = (1.0 + r_te) * s_te * c_te;
                std::complex<double> Rs_ref = q_te * back;

                auto St = run(4, 2), Sr = run(3, 2), Ct = run(1, 2), Cr = run(0, 2), C0 = run(1, 3);
                double d_sT = std::abs(cpx(St.xprmat, 0, 0) - Ts_ref), d_sR = std::abs(cpx(Sr.xprmat, 0, 0) - Rs_ref);
                double d_gT = std::abs((double)Ct.gain(0) - 0.5 * (std::norm(t_te * s_te * c_te) + std::norm(t_tm * s_tm * c_tm)));
                double d_gR = std::abs((double)Cr.gain(0) - 0.5 * (std::norm(q_te) + std::norm(q_tm)));
                double d_ch = std::max(std::abs(TE(Ct.xprmat) - t_te * s_te * c_te), std::abs(TM(Ct.xprmat) - t_tm * s_tm * c_tm));
                if (thd > 0.0) // the EM reflect port at exactly normal incidence has its own basis
                    d_ch = std::max(d_ch, std::max(std::abs(TE(Cr.xprmat) - q_te * back), std::abs(TM(Cr.xprmat) - q_tm * back)));
                QD_DIAG("%.2f,%+.1f,%.1f,%.2e,%.2e,%.2e,%.2e,%.2e,%+.6f\n", wei, wtf, thd, d_sT, d_sR, d_gT, d_gR, d_ch,
                        10.0 * std::log10((double)Ct.gain(0) / (double)C0.gain(0)));
                INFO("limit, wall ei " << wei << ", wall tf " << wtf << ", theta_n " << thd);
                const double lim = 1.0e-9;
                CHECK(d_sT < lim);
                CHECK(d_sR < lim);
                CHECK(d_gT < lim);
                CHECK(d_gR < lim);
                CHECK(d_ch < lim);
            }
}

TEST_CASE("Stack with tf - the scattering form against its reference, produce mode and itself", "[wave]")
{
    ub::OneThread one_thread;
    // A characteristic matrix has no per-boundary r and t to rescale, so a stack with tf is resolved in the scattering
    // form. Asserted:
    //
    //   the reference against the two it has to reduce to - diag_cm_stack at tf = 0, and diag_sheet_composite with tf
    //   at one layer;
    //   the library's ordered stack against that reference, both orders, both ports, scalar and EM;
    //   three statements that need no reference - a lossless stack conserves energy per channel at any tf, the transmit
    //   port is invariant under layer order, and tf = +1 / -1 on every layer closes the reflect / transmit port
    //   exactly;
    //   the core against produce mode - two 6 mm sheets of one permittivity and one tf > 0 are a single 12 mm sheet,
    //   because a matched boundary has no reflection to leak;
    //   a bounding sheet with tf resolved as a layer, with a negligible tf leaving the series where the tf-free layer
    //   puts it.
    const double f_hz = FRQ, thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const uint8_t OT_STACK_FBS = 71, OT_STACK_SBS = 69;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ---- The reference against the forms it reduces to. No library call. ----
    {
        const std::complex<double> eA(5.24, -0.63), eB(1.50, -0.40);
        double w_cm = 0.0, w_one = 0.0;
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
            for (int sc = 0; sc < 2; ++sc)
            {
                std::complex<double> ci(std::cos(thd * QPI / 180.0), 0.0);
                std::vector<DiagLayer> lay = {{eA, U, thk, std::complex<double>(0.3, -0.1), 0.0},
                                              {eB, U, thk, std::complex<double>(0.0, 0.0), 50.0}};
                for (int far = 0; far < 2; ++far)
                {
                    std::complex<double> ef = far ? eB : e_air;
                    DiagSheet X = diag_cm_stack(e_air, U, lay, ef, U, ci, f_hz, false, sc != 0);
                    DiagSheet Y = diag_sc_stack(e_air, U, 0.0, lay, ef, U, 0.0, ci, f_hz, false, sc != 0);
                    w_cm = std::max(w_cm, std::max(std::max(std::abs(X.T_te - Y.T_te), std::abs(X.T_tm - Y.T_tm)),
                                                   std::max(std::abs(X.R_te - Y.R_te), std::abs(X.R_tm - Y.R_tm))));
                }
                for (double tfs : {-0.5, 0.25, 1.0})
                    for (double tfn : {0.0, 0.5, -0.25})
                    {
                        DiagLump Lp;
                        Lp.tf = tfs, Lp.tf_far = tfn;
                        DiagSheet X = diag_sheet_composite(e_air, eA, eB, thk, std::real(ci), f_hz, false, sc != 0,
                                                           1.0, 1.0, 1.0, Lp);
                        std::vector<DiagLayer> one = {{eA, U, thk, std::complex<double>(0.0, 0.0), 0.0, tfs}};
                        DiagSheet Y = diag_sc_stack(e_air, U, 0.0, one, eB, U, tfn, ci, f_hz, false, sc != 0);
                        w_one = std::max(w_one, std::max(std::max(std::abs(X.T_te - Y.T_te), std::abs(X.T_tm - Y.T_tm)),
                                                         std::max(std::abs(X.R_te - Y.R_te), std::abs(X.R_tm - Y.R_tm))));
                    }
            }
        QD_DIAG("\n# diag: scattering-form reference. against diag_cm_stack at tf = 0, two layers with a\n");
        QD_DIAG("# resonance and alpha: %.3e. against diag_sheet_composite with tf, one layer: %.3e\n", w_cm, w_one);
        CHECK(w_cm < 1.0e-13);
        CHECK(w_one < 1.0e-14);
    }

    // ---- The library's ordered stack against the reference ----
    QD_DIAG("\n# diag: ordered stack with tf, air | 6 mm | 6 mm | air. A eps 5.24 - 0.63j,\n");
    QD_DIAG("# B eps 1.50 - 0.40j; AL and BL are their lossless twins\n");
    QD_DIAG("stack,tf,order,theta_n,ref_|T_te|,ref_|R_te|,d_scalar,d_EM_T,d_EM_R,rtype_T,rtype_R\n");

    double worst_ctl = 0.0, worst_tf = 0.0, worst_E = 0.0, worst_ord = 0.0;
    for (double tf : {0.0, -1.0, -0.5, 0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk},          // 1 A
            {1.50, 0.40, thk, 0.0, tf}, // 2 B_TF
            {5.24, 0.63, thk, 0.0, tf}, // 3 A_TF
            {5.24, 0.00, thk, 0.0, tf}, // 4 AL_TF
            {1.50, 0.00, thk, 0.0, tf}, // 5 BL_TF
        };
        auto mtl = lump_palette(pal);
        auto run = [&](int itype, uint8_t otype, int m1, int m2, double th)
        {
            auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
            C.run();
            return C;
        };
        struct St
        {
            const char *name;
            int m1, m2;
            bool all; // every layer carries the tf
        };
        const std::vector<St> stacks = {{"A | B_TF   ", 1, 2, false}, {"A_TF | B_TF", 3, 2, true}};

        for (const auto &S : stacks)
            for (int ord = 0; ord < 2; ++ord)
                for (double thd : {0.0, 20.0, 45.0, 70.0})
                {
                    double th = thd * QPI / 180.0;
                    std::complex<double> ci(std::cos(th), 0.0);
                    uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                    int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;
                    std::vector<DiagLayer> lay = {lump_layer(pal[(size_t)first - 1], f_hz),
                                                  lump_layer(pal[(size_t)second - 1], f_hz)};
                    DiagSheet Xs = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, true);
                    DiagSheet Xe = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, false);

                    auto St_ = run(4, ot, S.m1, S.m2, th), Sr = run(3, ot, S.m1, S.m2, th);
                    auto Et = run(1, ot, S.m1, S.m2, th), Er = run(0, ot, S.m1, S.m2, th);
                    double d_S = std::max(std::abs(cpx(St_.xprmat, 0, 0) - Xs.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Xs.R_te));
                    double d_T = std::max(std::abs(TE(Et.xprmat) - Xe.T_te), std::abs(TM(Et.xprmat) - Xe.T_tm));
                    double d_R = std::max(std::abs(TE(Er.xprmat) - Xe.R_te), std::abs(TM(Er.xprmat) - Xe.R_tm));
                    double w = std::max(d_S, std::max(d_T, d_R));
                    ((tf == 0.0) ? worst_ctl : worst_tf) = std::max((tf == 0.0) ? worst_ctl : worst_tf, w);

                    QD_DIAG("%s,%+.2f,%s,%.1f,%.9f,%.9f,%.2e,%.2e,%.2e,%u,%u\n", S.name, tf, ord ? "2|1" : "1|2", thd,
                            std::abs(Xe.T_te), std::abs(Xe.R_te), d_S, d_T, d_R,
                            (unsigned)Et.resolved_type[0], (unsigned)Er.resolved_type[0]);
                    INFO(S.name << ", tf " << tf << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                    CHECK(Et.resolved_type[0] != 0); // resolved, not killed
                    CHECK(Er.resolved_type[0] != 0);
                    CHECK(d_S < 1.0e-9);
                    CHECK(d_T < 1.0e-9);
                    CHECK(d_R < 1.0e-9);

                    // The end points: tf = -1 exact; tf = +1 exact for faces in phase only, these are lossy
                    if (tf == -1.0)
                        CHECK((double)Et.gain(0) == 0.0);
                    if (tf == 1.0 && S.all)
                        CHECK((double)Er.gain(0) < 2.0e-3);
                }

        // Lossless stack: energy per channel, and the transmit port under layer order
        for (double thd : {0.0, 30.0, 60.0, 80.0})
        {
            double th = thd * QPI / 180.0;
            auto T1 = run(1, OT_STACK_FBS, 4, 5, th), R1 = run(0, OT_STACK_FBS, 4, 5, th);
            auto T2 = run(1, OT_STACK_SBS, 4, 5, th), R2 = run(0, OT_STACK_SBS, 4, 5, th);
            double e_te = std::norm(TE(T1.xprmat)) + std::norm(TE(R1.xprmat));
            double e_tm = std::norm(TM(T1.xprmat)) + std::norm(TM(R1.xprmat));
            double d_o = std::max(std::abs(TE(T1.xprmat) - TE(T2.xprmat)), std::abs(TM(T1.xprmat) - TM(T2.xprmat)));
            double d_r = std::abs(TE(R1.xprmat) - TE(R2.xprmat));
            if (tf != 0.0)
                worst_E = std::max(worst_E, std::max(std::abs(e_te - 1.0), std::abs(e_tm - 1.0))),
                worst_ord = std::max(worst_ord, d_o);
            INFO("lossless stack, tf " << tf << ", theta_n " << thd);
            CHECK(std::abs(e_te - 1.0) < 1.0e-12);
            CHECK(std::abs(e_tm - 1.0) < 1.0e-12);
            CHECK(d_o < 1.0e-14);
            if (tf > -1.0 && tf < 1.0) // at tf = +1 both orders reflect nothing
                CHECK(d_r > 1.0e-3);
        }
    }
    QD_DIAG("# worst: tf = 0 rows %.3e, tf rows %.3e, energy of a lossless stack %.3e, order %.3e\n",
            worst_ctl, worst_tf, worst_E, worst_ord);

    // ---- The core against produce mode ----
    // Two sheets of one permittivity and one tf > 0 share a matched boundary, R0 = 0, which has nothing to leak; the
    // stack is the 12 mm sheet.
    QD_DIAG("\n# diag: tf, the core against produce mode. 6 mm | 6 mm of eps 5.24 - 0.63j, one tf,\n");
    QD_DIAG("# against one 12 mm sheet with that tf\ntf,theta_n,|T_te|,|R_te|,d_T,d_R,d_scalar\n");
    for (double tf : {0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, tf},                // 1
            {5.24, 0.63, thk, 0.0, tf, 0.0, 1.0, 0.0}, // 2 the same sheet; resQ alone is inert and keeps it distinct
            {5.24, 0.63, 2.0 * thk, 0.0, tf},          // 3 12 mm
        };
        auto mtl = lump_palette(pal);
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            auto stk = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_STACK_FBS, 2u, 1, 2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto one = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, 67, 1u, 3, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto St = stk(1), Sr = stk(0), Pt = one(1), Pr = one(0);
            auto Ss = stk(4), Sq = stk(3), Ps = one(4), Pq = one(3);
            double d_T = std::max(std::abs(TE(St.xprmat) - TE(Pt.xprmat)), std::abs(TM(St.xprmat) - TM(Pt.xprmat)));
            double d_R = std::max(std::abs(TE(Sr.xprmat) - TE(Pr.xprmat)), std::abs(TM(Sr.xprmat) - TM(Pr.xprmat)));
            double d_S = std::max(std::abs(cpx(Ss.xprmat, 0, 0) - cpx(Ps.xprmat, 0, 0)),
                                  std::abs(cpx(Sq.xprmat, 0, 0) - cpx(Pq.xprmat, 0, 0)));
            QD_DIAG("%+.2f,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e\n", tf, thd, std::abs(TE(Pt.xprmat)), std::abs(TE(Pr.xprmat)),
                    d_T, d_R, d_S);
            INFO("tf against produce mode, tf " << tf << ", theta_n " << thd);
            CHECK(St.resolved_type[0] != 0);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
            CHECK(d_S < 1.0e-9);
        }
    }

    // ---- A bounding sheet with tf, resolved as a layer ----
    // The host's own tf belongs to the host | sheet face under the same rule, not to the layer's composite reflectance.
    {
        const std::vector<LumpMtl> pal = {
            {1.50, 0.40, 0.0},              // 1 HOST
            {1.50, 0.40, 0.0, 0.0, 0.3},    // 2 HOST with tf 0.3
            {5.24, 0.63, thk},              // 3 SHEET
            {5.24, 0.63, thk, 0.0, 1.0e-9}, // 4 SHEET tf 1e-9
            {5.24, 0.63, thk, 0.0, 0.25},   // 5 SHEET tf 0.25
            {5.24, 0.63, thk, 0.0, -0.25},  // 6 SHEET tf -0.25
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int host, prev;
        };
        const std::vector<Row> rows = {
            {"plain host | SHEET        ", 1, 3},
            {"plain host | SHEET tf 1e-9", 1, 4},
            {"plain host | SHEET tf 0.25", 1, 5},
            {"plain host | SHEET tf -.25", 1, 6},
            {"host tf .3 | SHEET        ", 2, 3},
            {"host tf .3 | SHEET tf 0.25", 2, 5},
        };
        QD_DIAG("\n# diag: host layer bounded by a sheet carrying tf, host eps 1.50 - 0.40j. S_ref has the\n");
        QD_DIAG("# sheet as a layer backed by air with tf by the pair rule; d_plain is against the same\n");
        QD_DIAG("# layer with no tf anywhere\nrow,L_mm,theta_n,|S_applied|,|S_ref|,d_ref,d_plain\n");
        for (const auto &R : rows)
            for (double Lh : {0.05, 0.10})
                for (double thd : {0.0, 20.0, 40.0, 50.0})
                {
                    double theta_n = thd * QPI / 180.0;
                    const LumpMtl &H = pal[(size_t)R.host - 1], &Sm = pal[(size_t)R.prev - 1];
                    const std::complex<double> e_h = lump_eps(H, f_hz), e_s = lump_eps(Sm, f_hz);

                    auto C = make1<double>(mtl, 4, OT_EXIT, 1u, R.host, 0, R.prev, R.host, 0,
                                           Lh, 0.5, 0.5, QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
                    C.run();
                    std::complex<double> S_applied = (cpx(C.xprmat, 0, 0) / cpx(C.x_in, 0, 0)) /
                                                     diag_close(e_h, Lh, theta_n, f_hz);

                    std::complex<double> cs(std::cos(theta_n), 0.0);
                    std::complex<double> phi = diag_phi(e_h, Lh, theta_n, f_hz);
                    std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs), r_ahead0 = r_ahead;
                    if (H.tf != 0.0)
                        diag_tf_face(r_ahead, H.tf, 0.0); // an open bare face: the transformer, as its mirror
                    DiagSheet B = diag_sc_stack(e_h, U, H.tf, {lump_layer(Sm, f_hz)}, e_air, U, 0.0, cs, f_hz,
                                                false, true, /*undisplaced*/ 0.0);
                    DiagSheet B0 = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0);
                    std::complex<double> S_ref = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);
                    std::complex<double> S_plain = 1.0 / (1.0 - r_ahead0 * B0.R_te * phi * phi);
                    double d_ref = std::abs(S_applied - S_ref), d_plain = std::abs(S_applied - S_plain);

                    QD_DIAG("%s,%.1f,%.1f,%.9f,%.9f,%.3e,%.3e\n", R.name, Lh * 1000.0, thd,
                            std::abs(S_applied), std::abs(S_ref), d_ref, d_plain);
                    INFO(R.name << ", L " << Lh * 1000.0 << " mm, theta_n " << thd);
                    CHECK(rt_bit(C.resolved_type[0], 2));
                    CHECK(d_ref < 1.0e-9);
                    if (Sm.tf == 1.0e-9 && H.tf == 0.0) // no step: a negligible tf is a negligible change
                        CHECK(d_plain < 1.0e-8);
                }
    }
}

TEST_CASE("Reciprocity - glued entry against glued exit", "[wave]")
{
    // One three-medium stack resolved from both sides: air | sheet | solid entered from the air at
    // theta_ext, and solid | sheet | air left from inside the solid along the direction a tracer
    // holds after refracting in at that same theta_ext. The two are the same pair of waves
    // reversed, so in per-channel normal-flux powers they owe the same number.
    //
    // Lossless solid. Exact: the raw amplitudes of a reciprocal stack differ by the ratio of the two
    // outer tilted admittances, and the two flux conversions are that ratio and its inverse. This is
    // the row that guards the conversion - with it removed the two would sit 1.76 dB apart at normal
    // incidence.
    //
    // Lossy solid. Not exact, and the residual has a closed form. With eta = z c for TE and z / c
    // for TM in the solid, exit over entry is |eta|^2 / Re(eta)^2 = sec^2(arg eta) per channel: the
    // incident and reflected waves in a lossy medium interfere, so splitting its flux into two
    // independent parts is not exact, and the flux form is the one that composes across a chain.
    // That is the settled energy convention, so the residual is asserted against its closed form
    // and held to a bound, never asserted to zero. The normal-incidence row is called out:
    // no cosine enters there and the residual is sec^2(arg z) alone, 0.073946 dB for eps 1.50 - 0.40j.
    //
    // The host leg the exit closes is isolated by running it over L and 2 L and taking the ratio,
    // which holds whatever cosine the composite is built at.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SHEET = 3, LO = 5, LO_CLEAN = 6;

    struct Row
    {
        const char *name;
        int host;
        bool lossless;
    };
    const std::vector<Row> rows = {{"lossless eps 1.50       ", LO_CLEAN, true}, {"lossy eps 1.50 - 0.40j  ", LO, false}};

    QD_DIAG("\n# diag: glued entry / exit reciprocity, air | 6 mm eps 5.24 - 0.63j | solid, 1 GHz.\n");
    QD_DIAG("# x_over_e_* is exit over entry in dB; ref_* is the closed form sec^2(arg eta) of the solid,\n");
    QD_DIAG("# zero for a lossless one. d_* is the relative distance of the measured ratio from it\n");
    QD_DIAG("solid,theta_ext,P_entry_scalar,P_exit_scalar,x_over_e_scalar_dB,ref_te_dB,ref_tm_dB,"
            "x_over_e_gain_dB,ref_gain_dB,d_scalar,d_gain\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> e_h = diag_eps(R.host, f_hz);
            double ct = diag_traced_ct(e_h, th);              // the real projection the tracer holds
            std::complex<double> cx = diag_ct_exact(e_h, th); // the exact complex in-solid cosine

            auto enter = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, SHEET, R.host, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto leave = [&](int itype, double host_len)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SHEET, R.host, 0, R.host, 0,
                                       host_len, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };
            auto Es = enter(4), Ee = enter(1);
            auto Xs = leave(4, L), Xs2 = leave(4, 2.0 * L), Xe = leave(1, L);
            REQUIRE(Es.resolved_type[0] != 0);
            REQUIRE(Xs.resolved_type[0] != 0);

            double host2 = std::norm(cpx(Xs2.xprmat, 0, 0) / cpx(Xs.xprmat, 0, 0)); // host leg over L, power
            double Pe_s = std::norm(cpx(Es.xprmat, 0, 0)), Px_s = std::norm(cpx(Xs.xprmat, 0, 0)) / host2;
            double Pe_te = std::norm(cpx(Ee.xprmat, 0, 3)), Pe_tm = std::norm(cpx(Ee.xprmat, 0, 0));
            double Ge = (double)Ee.gain(0), Gx = (double)Xe.gain(0) / host2;

            std::complex<double> z = std::sqrt(e_h);
            double c_te = std::cos(std::arg(z * cx)), c_tm = std::cos(std::arg(z / cx));
            double k_te = 1.0 / (c_te * c_te), k_tm = 1.0 / (c_tm * c_tm);
            double ref_G = 0.5 * (Pe_te * k_te + Pe_tm * k_tm);
            double d_s = std::abs(Px_s - Pe_s * k_te) / Pe_s, d_g = std::abs(Gx - ref_G) / Ge;
            auto dB = [](double g)
            { return 10.0 * std::log10(g); };

            QD_DIAG("%s,%.1f,%.9f,%.9f,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f,%.3e,%.3e\n", R.name, thd, Pe_s, Px_s,
                    dB(Px_s / Pe_s), dB(k_te), dB(k_tm), dB(Gx / Ge), dB(ref_G / Ge), d_s, d_g);

            INFO(R.name << ", theta_ext " << thd);
            CHECK(std::abs(Ge - 0.5 * (Pe_te + Pe_tm)) < 1.0e-12); // the entry gain is the channel mean
            CHECK(d_s < 1.0e-9);
            CHECK(d_g < 1.0e-9);
            if (R.lossless)
            {
                CHECK(k_te == 1.0); // the closed form is identically 1, so the two rows above read zero
                CHECK(k_tm == 1.0);
            }
            else
            {
                // The flux-splitting residual, held to a bound. TE grows towards grazing
                // and TM falls, so the bound is on the larger channel and on the mean.
                CHECK(dB(Px_s / Pe_s) > 0.0);
                CHECK(dB(Px_s / Pe_s) < 0.50);
                CHECK(std::abs(dB(Gx / Ge)) < 0.25);
                if (thd == 0.0) // no cosine enters: sec^2(arg z) and nothing else
                {
                    CHECK(std::abs(dB(Px_s / Pe_s) - 0.073946) < 1.0e-6);
                    CHECK(std::abs(dB(Gx / Ge) - 0.073946) < 1.0e-6);
                }
            }
        }
}

TEST_CASE("Lined on both faces - a glued exit behind a sheet exits into air", "[wave]")
{
    // A glued exit, solid | sheet | air, behind a solid that was entered through a sheet: after the sheet crossing the
    // previous-medium word holds that sheet, and a sheet in that word is not a medium. Beyond the sheet ahead is air,
    // which ray_mesh_interact's own composite assumes for every glued exit, and the exit leaves the ray in air with the
    // sheet it crossed as the previous medium.
    //
    // No path_dir is supplied, so every boundary is at the fbs cosine and the host leg closes in the path form
    // diag_close reproduces. The composite is measured with the host's series declined (eps 1.5), and the series as the
    // ratio of the run that resolves (eps 0) over it.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9, CLEAN = 6;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const std::complex<double> e_h = diag_eps(HOST, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Row
    {
        const char *name;
        int s1, s2;
    };
    const std::vector<Row> rows = {
        {"S | host | S", SHEET, SHEET},
        {"N | host | S", NEAR, SHEET},
        {"S | host | N", SHEET, NEAR},
    };
    const std::vector<double> angles = {0.0, 20.0, 40.0};

    auto entry = [&](int itype, int s1, double th)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, s1, HOST, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.run();
        return C;
    };
    auto exitev = [&](int itype, int s2, int host, int prev, int cur, double th, double eps)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, s2, host, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.run();
        return C;
    };

    // ---- The composite and the state words, both ports, scalar ----
    QD_DIAG("\n# diag: lined on both faces, air | S1 | host | S2 | air as two events, host eps 1.50 - 0.40j\n");
    QD_DIAG("# over 100 mm, S eps 5.24 - 0.63j and N eps 1.5015 - 0.40j, 6 mm, scalar, host series declined.\n");
    QD_DIAG("# The second event against the reference with air beyond (d_ref) and against the same exit\n");
    QD_DIAG("# with a clean previous-medium word (d_bare); c_ref is that control run against the reference\n");
    QD_DIAG("stack,theta_n,port,meas_dB,ref_dB,d_ref,d_bare,c_ref,prev,cur,resolved,buf,rtype\n");
    for (const auto &R : rows)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto E1 = entry(4, R.s1, th);
            INFO(R.name << ", theta_n " << thd);
            check_state(E1, R.s1, HOST, 0);

            DiagSheet D = diag_sheet_composite(e_h, diag_eps(R.s2, f_hz), e_air, thk, ct, f_hz, true, true);
            std::complex<double> close = diag_close(e_h, L, th, f_hz);
            for (int port = 0; port < 2; ++port)
            {
                int itype = port ? 3 : 4;
                auto A = exitev(itype, R.s2, HOST, (int)E1.prev_out(0), (int)E1.cur_out(0), th, 1.5);
                auto B = exitev(itype, R.s2, HOST, 0, HOST, th, 1.5);
                std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
                std::complex<double> ref = (port ? D.R_te : D.T_te) * close;
                double d_ref = std::abs(a - ref) / std::abs(ref), d_bare = std::abs(a - b) / std::abs(b);
                double c_ref = std::abs(b - ref) / std::abs(ref);
                QD_DIAG("%s,%.1f,%s,%.4f,%.4f,%.3e,%.3e,%.3e,%d,%d,%d,%d,%u\n", R.name, thd, port ? "R" : "T",
                        dB(std::norm(a)), dB(std::norm(ref)), d_ref, d_bare, c_ref, mat_of(A.prev_out(0)),
                        mat_of(A.cur_out(0)), (int)flag_of(A.cur_out(0)), (int)A.buf_out(0),
                        (unsigned)A.resolved_type[0]);
                INFO((port ? "reflect" : "transmit"));
                CHECK(c_ref < 1.0e-9); // the reference is the one the bare-entered exit sits on
                CHECK(d_ref < 1.0e-9); // air beyond the second sheet
                CHECK(a == b);         // and the exit does not know how the host was entered
                if (port)
                {
                    check_state(A, R.s2, enc(HOST, false), 0x4000); // a reflection stays in the host; its series was declined, so it is not resolved but followed
                    check_rtype(A, 207);
                }
                else
                {
                    check_state(A, R.s2, 0, 0); // the ray is in air, behind the sheet it crossed
                    check_state(B, R.s2, 0, 0); // as it is after any glued exit
                    check_rtype(A, 71);
                }
            }
        }

    // ---- The same in EM, per entry of the matrix, against the bare-entered exit ----
    QD_DIAG("\n# diag: lined on both faces, EM. Gain of the second event against the same exit with a clean\n");
    QD_DIAG("# previous-medium word; d_x is the largest difference over the eight matrix entries\n");
    QD_DIAG("stack,theta_n,port,gain_dB,bare_dB,d_gain_dB,d_x,cur\n");
    for (const auto &R : rows)
        for (double thd : angles)
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                auto E1 = entry(1, R.s1, th);
                auto A = exitev(port ? 0 : 1, R.s2, HOST, (int)E1.prev_out(0), (int)E1.cur_out(0), th, 1.5);
                auto B = exitev(port ? 0 : 1, R.s2, HOST, 0, HOST, th, 1.5);
                double d_x = 0.0;
                for (arma::uword k = 0; k < 8; ++k)
                    d_x = std::max(d_x, std::abs((double)A.xprmat(k, 0) - (double)B.xprmat(k, 0)));
                QD_DIAG("%s,%.1f,%s,%.4f,%.4f,%+.4f,%.3e,%d\n", R.name, thd, port ? "R" : "T", dB((double)A.gain(0)),
                        dB((double)B.gain(0)), dB((double)A.gain(0)) - dB((double)B.gain(0)), d_x,
                        mat_of(A.cur_out(0)));
                INFO(R.name << ", EM, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                CHECK(d_x == 0.0);
                CHECK((double)A.gain(0) == (double)B.gain(0));
                CHECK(mat_of(A.cur_out(0)) == (port ? HOST : 0));
            }

    // ---- The host's series between two sheets ----
    // Its mirrors are the sheet ahead and the sheet behind, each a layer of its own thickness
    // with air beyond. The bare-entered row has a half-space of air behind and is the control.
    QD_DIAG("\n# diag: lined on both faces, the host's series. S_applied is the resolving run over the\n");
    QD_DIAG("# declined one; S_ref has both sheets as layers backed by air. behind = 0 is the bare-entered\n");
    QD_DIAG("# control, with a half-space of air behind\n");
    QD_DIAG("ahead,behind,theta_n,port,|S_applied|,arg_S_applied,|S_ref|,arg_S_ref,|dS|,d_dB\n");
    for (const auto &R : rows)
        for (int ctrl = 0; ctrl < 2; ++ctrl)
            for (double thd : angles)
                for (int port = 0; port < 2; ++port)
                {
                    double th = thd * QPI / 180.0;
                    int itype = port ? 3 : 4, prev = ctrl ? 0 : R.s1;
                    auto A1 = exitev(itype, R.s2, HOST, prev, HOST, th, 0.0);
                    auto A0 = exitev(itype, R.s2, HOST, prev, HOST, th, 1.5);
                    std::complex<double> S = cpx(A1.xprmat, 0, 0) / cpx(A0.xprmat, 0, 0);

                    std::complex<double> cs(std::cos(th), 0.0);
                    std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                    DiagSheet Ba = diag_cm_stack(e_h, U, {{diag_eps(R.s2, f_hz), U, thk}}, e_air, U, cs, f_hz,
                                                 false, true, 0.0);
                    std::complex<double> r_behind = diag_r_slabside(e_h, e_air, cs);
                    if (!ctrl)
                        r_behind = diag_cm_stack(e_h, U, {{diag_eps(R.s1, f_hz), U, thk}}, e_air, U, cs, f_hz,
                                                 false, true, 0.0)
                                       .R_te;
                    std::complex<double> S_ref = 1.0 / (1.0 - Ba.R_te * r_behind * phi * phi);
                    QD_DIAG("%d,%d,%.1f,%s,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%+.4f\n", R.s2, prev, thd, port ? "R" : "T",
                            std::abs(S), std::arg(S) * 180.0 / QPI, std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI,
                            std::abs(S - S_ref), 20.0 * std::log10(std::abs(S) / std::abs(S_ref)));
                    INFO(R.name << (ctrl ? ", bare-entered control" : "") << ", series, theta_n " << thd
                                << (port ? ", reflect" : ", transmit"));
                    REQUIRE(rt_bit(A1.resolved_type[0], 2));
                    CHECK(std::abs(S - S_ref) < 1.0e-9);
                }

    // ---- The continuation direction ----
    // The outer pair decides it, and the outer pair is host | air. The ray is tilted in the xy
    // plane so the direction the segment has and the angle the hit reports agree.
    QD_DIAG("\n# diag: lined on both faces, the direction the transmitted ray leaves in, degrees from the\n");
    QD_DIAG("# normal, against the same exit with a clean previous-medium word\n");
    QD_DIAG("stack,theta_n,out_deg,bare_deg,d_dir\n");
    for (const auto &R : rows)
        for (double thd : {10.0, 20.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            Rsu<double> C[2];
            for (int k = 0; k < 2; ++k)
            {
                C[k] = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, R.s2, HOST, k ? 0 : R.s1, HOST, 0, L / std::cos(th), 0.5,
                                     0.0, QPI / 2.0 - th, 1.5);
                C[k].orig(0, 0) = C[k].fbs(0, 0) - L;
                C[k].orig(0, 1) = -L * std::tan(th);
                C[k].dest(0, 0) = C[k].fbs(0, 0) + 0.5 * std::cos(th);
                C[k].dest(0, 1) = 0.5 * std::sin(th);
                C[k].has_path_dirN = true;
                C[k].path_dirN.zeros(1, 3);
                C[k].path_dirN(0, 0) = std::cos(th), C[k].path_dirN(0, 1) = std::sin(th);
                C[k].run();
            }
            auto deg = [&](const Rsu<double> &X)
            { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
            double d_dir = 0.0;
            for (arma::uword k = 0; k < 3; ++k)
                d_dir = std::max(d_dir, std::abs((double)C[0].path_dirN(0, k) - (double)C[1].path_dirN(0, k)));
            QD_DIAG("%s,%.1f,%.4f,%.4f,%.3e\n", R.name, thd, deg(C[0]), deg(C[1]), d_dir);
            INFO(R.name << ", direction, theta_n " << thd);
            CHECK(d_dir == 0.0);
            CHECK(deg(C[1]) > thd + 1.0); // host to air bends away from the normal
        }

    // ---- A resolved ray at a sheet ----
    // A reflection that collapsed the host's bounce series flags the ray as resolved, and the close-out of that leg
    // does not apply the series a second time, at a bare exit and at a sheet event alike. Measured as the resolving run
    // over the declined one, with the flag set and without it, for a glued exit, an embedded sheet and an ordered
    // stack.
    QD_DIAG("\n# diag: a resolved ray at a sheet. S is what the host's series applies at the event; it has\n");
    QD_DIAG("# to be one for a resolved ray, whatever it is for an unresolved one\n");
    QD_DIAG("event,host,prev,theta_n,|S_unresolved|,|S_resolved|,|S_resolved - 1|,cur,rtype\n");
    struct Ev
    {
        const char *name;
        uint8_t ot;
        unsigned nH;
        int m1, m2, prev;
    };
    for (int host : {HOST, CLEAN})
    {
        const std::vector<Ev> evs = {
            {"glued exit", OT_GLUED_EXIT, 2u, SHEET, host, 0},
            {"glued exit", OT_GLUED_EXIT, 2u, SHEET, host, SHEET},
            {"embedded  ", OT_SHEET_ISO, 1u, SHEET, 0, 0},
            {"stack     ", OT_GLUED_ENTRY, 2u, SHEET, NEAR, 0},
        };
        for (const auto &E : evs)
            for (double thd : {0.0, 40.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> S[2];
                bool same[2] = {false, false}; // the two runs wrote the same field, bit for bit
                Rsu<double> last;
                for (int res = 0; res < 2; ++res)
                {
                    std::complex<double> v[2];
                    for (int k = 0; k < 2; ++k)
                    {
                        auto C = make1<double>(mtl, 4, E.ot, E.nH, E.m1, E.m2, E.prev, (int)enc(host, res != 0), 0, L,
                                               0.5, 0.5, QPI / 2.0 - th, k ? 0.0 : 1.5);
                        C.run();
                        v[k] = cpx(C.xprmat, 0, 0);
                        last = C;
                    }
                    S[res] = v[1] / v[0];
                    same[res] = (v[1] == v[0]);
                }
                QD_DIAG("%s,%d,%d,%.1f,%.9f,%.9f,%.3e,%d,%u\n", E.name, host, E.prev, thd, std::abs(S[0]), std::abs(S[1]),
                        std::abs(S[1] - 1.0), mat_of(last.cur_out(0)), (unsigned)last.resolved_type[0]);
                INFO(E.name << ", host " << host << ", prev " << E.prev << ", theta_n " << thd);
                CHECK(std::abs(S[0] - 1.0) > 1.0e-3); // the row has a series to apply
                CHECK(same[1]);                       // and a resolved ray does not get it
            }
    }
}

TEST_CASE("Layered wall - a glued exit behind a bare-entered second layer exits into air", "[wave]")
{
    // A glued exit, solid | sheet | air, on a wall of two solid layers with a coating on its far side, air | A | B |
    // coating | air: the leg in B was entered from A, so the previous-medium word names A. Beyond the sheet of a glued
    // exit is air; the medium the host was entered from is behind the ray and is the mirror behind it in the host's
    // series, nothing else.
    //
    // Four instruments. Per event, with no path_dir: the exit with A in the previous-medium word, with and without bit
    // 14, against the composite with air beyond and against the same exit with a clean word, bitwise, both ports,
    // scalar and EM, the state words and the direction. The host's series as the ratio of the resolving run over the
    // declined one, with the coating backed by air and by A as two references. The chain, entry from ray_mesh_interact,
    // the transition A -> B and the glued exit fed the words the transition wrote, with path_dir: against the single
    // pass through A and B times the composite, and, with B's series resolved, against the characteristic matrix of {B,
    // coating} between the half-spaces A and air. And the next wall after the exit, which has to be entered from air.
    //
    // The matrix of the whole wall, air | A | B | coating | air, is printed and not asserted: two traced solids are
    // resolved as two series, and the mirror behind B is A as a half-space where the whole wall has A as a layer backed
    // by air.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, CLEAN = 6, SHEET = 3, WALL = 4;
    const uint8_t OT_GLUED_EXIT = 69;
    const std::complex<double> e_h = diag_eps(HOST, f_hz), e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<int> firsts = {2, 1, 4}; // eps 5.24, eps 5.24 - 0.63j, eps 9.0 - 1.20j

    auto exitev = [&](int itype, int prev, double th, double eps)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SHEET, HOST, prev, HOST, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.run();
        return C;
    };

    // ---- The composite and the state words, both ports, scalar ----
    QD_DIAG("\n# diag: layered wall, the glued exit of B | coating | air with A in the previous-medium word.\n");
    QD_DIAG("# B eps 1.50 - 0.40j over 100 mm, coating eps 5.24 - 0.63j 6 mm, scalar, no path_dir, host series\n");
    QD_DIAG("# declined. A is 2 (eps 5.24), 1 (eps 5.24 - 0.63j) or 4 (eps 9.0 - 1.20j); flag is bit 14 of\n");
    QD_DIAG("# the word. d_ref against the composite with air beyond, d_bare against the same exit with a\n");
    QD_DIAG("# clean previous-medium word, c_ref that control against the reference\n");
    QD_DIAG("A,flag,theta_n,port,meas_dB,ref_dB,d_ref,d_bare,c_ref,prev,cur,resolved,buf,rtype\n");
    for (int first : firsts)
        for (int flag = 0; flag < 2; ++flag)
            for (double thd : {0.0, 20.0, 40.0})
            {
                double th = thd * QPI / 180.0, ct = std::cos(th);
                int pw = first | (flag ? 0x4000 : 0);
                DiagSheet D = diag_sheet_composite(e_h, e_s, e_air, thk, ct, f_hz, true, true);
                std::complex<double> close = diag_close(e_h, L, th, f_hz);
                for (int port = 0; port < 2; ++port)
                {
                    int itype = port ? 3 : 4;
                    auto A = exitev(itype, pw, th, 1.5);
                    auto B = exitev(itype, 0, th, 1.5);
                    std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
                    std::complex<double> ref = (port ? D.R_te : D.T_te) * close;
                    double d_ref = std::abs(a - ref) / std::abs(ref), d_bare = std::abs(a - b) / std::abs(b);
                    double c_ref = std::abs(b - ref) / std::abs(ref);
                    QD_DIAG("%d,%d,%.1f,%s,%.4f,%.4f,%.3e,%.3e,%.3e,%d,%d,%d,%d,%u\n", first, flag, thd, port ? "R" : "T",
                            dB(std::norm(a)), dB(std::norm(ref)), d_ref, d_bare, c_ref, mat_of(A.prev_out(0)),
                            mat_of(A.cur_out(0)), (int)flag_of(A.cur_out(0)), (int)A.buf_out(0),
                            (unsigned)A.resolved_type[0]);
                    INFO("A " << first << ", flag " << flag << ", theta_n " << thd << (port ? ", reflect" : ", transmit"));
                    CHECK(c_ref < 1.0e-9); // the reference is the one the bare-entered exit sits on
                    CHECK(d_ref < 1.0e-9); // air beyond the coating
                    CHECK(a == b);         // and the exit does not know what the host was entered from
                    if (port)
                    {
                        check_state(A, SHEET, enc(HOST, false), 0x4000); // a reflection stays in the host; its series was declined, so it is not resolved but followed
                        check_rtype(A, 207);
                    }
                    else
                    {
                        check_state(A, SHEET, 0, 0); // the ray is in air, behind the sheet it crossed
                        check_rtype(A, 71);
                    }
                }
            }

    // ---- The same in EM, per entry of the matrix ----
    QD_DIAG("\n# diag: layered wall, EM. Gain of the glued exit with A in the previous-medium word against\n");
    QD_DIAG("# the same exit with a clean word; d_x is the largest difference over the eight matrix entries\n");
    QD_DIAG("A,theta_n,port,gain_dB,bare_dB,d_gain_dB,d_x,cur\n");
    for (int first : firsts)
        for (double thd : {0.0, 20.0, 40.0})
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                auto A = exitev(port ? 0 : 1, first, th, 1.5);
                auto B = exitev(port ? 0 : 1, 0, th, 1.5);
                double d_x = 0.0;
                for (arma::uword k = 0; k < 8; ++k)
                    d_x = std::max(d_x, std::abs((double)A.xprmat(k, 0) - (double)B.xprmat(k, 0)));
                QD_DIAG("%d,%.1f,%s,%.4f,%.4f,%+.4f,%.3e,%d\n", first, thd, port ? "R" : "T", dB((double)A.gain(0)),
                        dB((double)B.gain(0)), dB((double)A.gain(0)) - dB((double)B.gain(0)), d_x, mat_of(A.cur_out(0)));
                INFO("A " << first << ", EM, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                CHECK(d_x == 0.0);
                CHECK((double)A.gain(0) == (double)B.gain(0));
                CHECK(mat_of(A.cur_out(0)) == (port ? HOST : 0));
            }

    // ---- The host's series: the coating ahead backed by air, A behind as a half-space ----
    QD_DIAG("\n# diag: layered wall, the host's series at the glued exit. S_applied is the resolving run over\n");
    QD_DIAG("# the declined one. S_air has the coating as a layer backed by air and A behind as a half-space;\n");
    QD_DIAG("# S_A backs the coating with A\n");
    QD_DIAG("A,theta_n,port,|S_applied|,arg_S_applied,|S_air|,arg_S_air,|dS_air|,|dS_A|,sep\n");
    for (int first : firsts)
        for (double thd : {0.0, 20.0, 40.0})
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                int itype = port ? 3 : 4;
                auto A1 = exitev(itype, first, th, 0.0);
                auto A0 = exitev(itype, first, th, 1.5);
                std::complex<double> S = cpx(A1.xprmat, 0, 0) / cpx(A0.xprmat, 0, 0);
                std::complex<double> cs(std::cos(th), 0.0), e_a = diag_eps(first, f_hz);
                std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                std::complex<double> r_behind = diag_r_slabside(e_h, e_a, cs);
                std::complex<double> r_air = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_A = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_a, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> S_air = 1.0 / (1.0 - r_air * r_behind * phi * phi);
                std::complex<double> S_A = 1.0 / (1.0 - r_A * r_behind * phi * phi);
                QD_DIAG("%d,%.1f,%s,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%.3e,%.3e\n", first, thd, port ? "R" : "T", std::abs(S),
                        std::arg(S) * 180.0 / QPI, std::abs(S_air), std::arg(S_air) * 180.0 / QPI, std::abs(S - S_air),
                        std::abs(S - S_A), std::abs(S_air - S_A));
                INFO("A " << first << ", series, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                REQUIRE(rt_bit(A1.resolved_type[0], 2));
                CHECK(std::abs(S_air - S_A) > 1.0e-3); // the two backings can be told apart
                CHECK(std::abs(S - S_air) < 1.0e-9);
            }

    // ---- The continuation direction, no recovery involved ----
    QD_DIAG("\n# diag: layered wall, the direction the transmitted ray leaves in, degrees from the normal,\n");
    QD_DIAG("# against the same exit with a clean previous-medium word\n");
    QD_DIAG("A,theta_n,out_deg,bare_deg,d_dir\n");
    for (int first : firsts)
        for (double thd : {10.0, 20.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            Rsu<double> C[2];
            for (int k = 0; k < 2; ++k)
            {
                C[k] = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, SHEET, HOST, k ? 0 : first, HOST, 0, L / std::cos(th), 0.5,
                                     0.0, QPI / 2.0 - th, 1.5);
                C[k].orig(0, 0) = C[k].fbs(0, 0) - L;
                C[k].orig(0, 1) = -L * std::tan(th);
                C[k].dest(0, 0) = C[k].fbs(0, 0) + 0.5 * std::cos(th);
                C[k].dest(0, 1) = 0.5 * std::sin(th);
                C[k].has_path_dirN = true;
                C[k].path_dirN.zeros(1, 3);
                C[k].path_dirN(0, 0) = std::cos(th), C[k].path_dirN(0, 1) = std::sin(th);
                C[k].run();
            }
            auto deg = [&](const Rsu<double> &X)
            { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
            double d_dir = 0.0;
            for (arma::uword k = 0; k < 3; ++k)
                d_dir = std::max(d_dir, std::abs((double)C[0].path_dirN(0, k) - (double)C[1].path_dirN(0, k)));
            QD_DIAG("%d,%.1f,%.4f,%.4f,%.3e\n", first, thd, deg(C[0]), deg(C[1]), d_dir);
            INFO("A " << first << ", direction, theta_n " << thd);
            CHECK(d_dir == 0.0);
            CHECK(deg(C[1]) > thd + 1.0); // host to air bends away from the normal
        }

    // ---- The chain, air | A | B | coating | air, with path_dir ----
    struct Ent
    {
        std::complex<double> te;
        double c, s; // traced cosine and sine in the first solid
    };
    auto entry = [&](int first, double thd) -> Ent
    {
        arma::mat orig, dest, fbs, sbs;
        diag_ray_at(thd, orig, dest);
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                        &path_dirN, true, (arma::u32_vec *)nullptr);
        REQUIRE(path_dirN.n_rows == 1);
        Ent E;
        E.te = std::complex<double>(xprmatN(0, 0), xprmatN(1, 0));
        double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
        E.c = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
        E.s = std::sqrt(std::max(0.0, 1.0 - E.c * E.c));
        return E;
    };
    // One event closing a leg of normal thickness Ln along (dx, dy). A media-to-media event takes
    // its plane normal from the second face, reversed to oppose the ray.
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double Ln,
                     double th, double eps, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, Ln, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        if (ot == OT_M2M_M1)
            C.normals(0, 3) = -1.0;
        C.run();
        return C;
    };
    auto port_p = [](std::complex<double> ea, std::complex<double> eb, std::complex<double> ca)
    {
        std::complex<double> te, tm;
        diag_port(ea, eb, ca, true, te, tm);
        return std::norm(te);
    };
    auto decay = [&](std::complex<double> e, std::complex<double> c, double Ln)
    { return std::exp(2.0 * k0 * std::imag(std::sqrt(e) * c) * Ln); };
    auto clampr = [](std::complex<double> r)
    { return (std::abs(r) > 1.0) ? r / std::abs(r) : r; };

    QD_DIAG("\n# diag: layered wall, the chain air | A 50 mm | B 100 mm | coating 6 mm | air, scalar, path_dir\n");
    QD_DIAG("# supplied, the glued exit fed the words the transition A -> B wrote. Series declined: d_T and\n");
    QD_DIAG("# d_R against the single pass through A and B times the composite B | coating | air at the exact\n");
    QD_DIAG("# cosine, c_T the same for the exit with a clean previous-medium word, d_twin the two exits\n");
    QD_DIAG("# against each other. out_deg is the direction the ray leaves in and d_dir its distance from\n");
    QD_DIAG("# the clean-word exit's. prev, cur, buf, rtype after the transmit pass\n");
    QD_DIAG("A,B,theta_ext,ref_T_dB,d_T,d_R,c_T,d_twin,out_deg,d_dir,prev,cur,buf,rtype\n");
    const std::vector<double> ext = {10.0, 40.0, 60.0, 75.0, 85.0};
    for (int host : {HOST, CLEAN})
        for (int first : {2, 1})
            for (double thd : ext)
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(first, thd);
                auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, 1.5, E.c, E.s);
                double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                int pv = (int)T.prev_out(0), cu = (int)T.cur_out(0), bu = (int)T.buf_out(0);
                auto Xt = event(4, OT_GLUED_EXIT, 2u, SHEET, host, pv, cu, bu, L2, th, 1.5, dx, dy);
                auto Xr = event(3, OT_GLUED_EXIT, 2u, SHEET, host, pv, cu, bu, L2, th, 1.5, dx, dy);
                auto Ct = event(4, OT_GLUED_EXIT, 2u, SHEET, host, 0, host, 0, L2, th, 1.5, dx, dy);

                DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e_air, thk, c2, f_hz, true, true);
                double front = port_p(e_air, e1, c0) * port_p(e1, e2, c1) * decay(e1, c1, L1);
                double pre = std::norm(E.te * cpx(T.xprmat, 0, 0));
                double ref_T = front * decay(e2, c2, L2) * std::norm(Ref.T_te);
                // A reflect pass at a sheet closes the host leg as the transmit pass does
                double ref_R = front * decay(e2, c2, L2) * std::norm(Ref.R_te);
                double ch_T = pre * std::norm(cpx(Xt.xprmat, 0, 0)), ch_R = pre * std::norm(cpx(Xr.xprmat, 0, 0));
                double cl_T = pre * std::norm(cpx(Ct.xprmat, 0, 0));
                double d_dir = 0.0;
                for (arma::uword k = 0; k < 3; ++k)
                    d_dir = std::max(d_dir, std::abs((double)Xt.path_dirN(0, k) - (double)Ct.path_dirN(0, k)));
                double out_deg = std::atan2(std::abs((double)Xt.path_dirN(0, 1)), std::abs((double)Xt.path_dirN(0, 0))) * 180.0 / QPI;
                QD_DIAG("%d,%d,%.1f,%.6f,%+.6f,%+.6f,%+.6f,%+.6f,%.4f,%.3e,%d,%d,%d,%u\n", first, host, thd, dB(ref_T),
                        dB(ch_T) - dB(ref_T), dB(ch_R) - dB(ref_R), dB(cl_T) - dB(ref_T), dB(ch_T) - dB(cl_T), out_deg,
                        d_dir, mat_of(Xt.prev_out(0)), mat_of(Xt.cur_out(0)), (int)Xt.buf_out(0),
                        (unsigned)Xt.resolved_type[0]);
                INFO("chain, A " << first << ", B " << host << ", theta_ext " << thd);
                REQUIRE(mat_of(T.prev_out(0)) == first);
                CHECK(std::abs(dB(cl_T) - dB(ref_T)) < 1.0e-6); // the control: the clean-word exit on the reference
                CHECK(std::abs(dB(ch_T) - dB(ref_T)) < 1.0e-6);
                CHECK(std::abs(dB(ch_R) - dB(ref_R)) < 1.0e-6);
                CHECK(std::abs(dB(ch_T) - dB(cl_T)) < 1.0e-9);
                CHECK(d_dir < 1.0e-9); // the direction is that of B -> air
                check_state(Xt, SHEET, 0, 0);
                check_rtype(Xt, 71);
                check_state(Xr, SHEET, enc(host, false), 0x4000); // declined: the reflected ray is the tracer's, followed
                check_rtype(Xr, 207);
            }

    QD_DIAG("\n# diag: layered wall, the same chain with B's series resolved, against the characteristic\n");
    QD_DIAG("# matrix of {B, coating} between the half-spaces A and air times the single pass through A.\n");
    QD_DIAG("# clamp_dB is what the clamp on a mirror's reflectance costs. d_wall is recorded, not asserted:\n");
    QD_DIAG("# both series resolved against the matrix of the whole wall, air | A | B | coating | air\n");
    QD_DIAG("A,B,theta_ext,chain_dB,ref_dB,d_dB,|r_sheet|,clamp_dB,wall_dB,d_wall,rtype\n");
    for (int host : {HOST, CLEAN})
        for (int first : {2, 1})
            for (double thd : ext)
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(first, thd);
                double chain[2];
                Rsu<double> X;
                for (int k = 0; k < 2; ++k) // k = 1 resolves A's series as well
                {
                    auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, k ? 0.0 : 1.5, E.c, E.s);
                    X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, (int)T.prev_out(0), (int)T.cur_out(0), (int)T.buf_out(0),
                              L2, th, 0.0, (double)T.path_dirN(0, 0), (double)T.path_dirN(0, 1));
                    chain[k] = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                }
                DiagSheet M = diag_cm_stack(e1, U, {{e2, U, L2}, {e_s, U, thk}}, e_air, U, c1, f_hz, false, true, 0.0);
                double ref = port_p(e_air, e1, c0) * decay(e1, c1, L1) * std::norm(M.T_te);
                DiagSheet W = diag_cm_stack(e_air, U, {{e1, U, L1}, {e2, U, L2}, {e_s, U, thk}}, e_air, U, c0, f_hz, false,
                                            true, 0.0);

                std::complex<double> r_s = diag_cm_stack(e2, U, {{e_s, U, thk}}, e_air, U, c2, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_b = diag_r_slabside(e2, e1, c2);
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e2) * c2 * L2);
                double clamp_dB = 20.0 * std::log10(std::abs(1.0 - r_b * r_s * ph * ph) /
                                                    std::abs(1.0 - clampr(r_b) * clampr(r_s) * ph * ph));
                QD_DIAG("%d,%d,%.1f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%.6f,%+.6f,%u\n", first, host, thd, dB(chain[0]), dB(ref),
                        dB(chain[0]) - dB(ref), std::abs(r_s), clamp_dB, dB(std::norm(W.T_te)),
                        dB(chain[1]) - dB(std::norm(W.T_te)), (unsigned)X.resolved_type[0]);
                INFO("chain resolved, A " << first << ", B " << host << ", theta_ext " << thd);
                CHECK(std::abs(dB(chain[0]) - dB(ref)) < 1.0e-6); // on the matrix: no mirror is clamped; clamp_dB is what a clamp would cost
                if (!(host == HOST && thd == 85.0))
                    CHECK(clamp_dB == 0.0); // the clamp is engaged on the lossy host at 85 deg only
            }

    // ---- The next wall after the exit ----
    // The ray is in air. The next wall reflects off its front face, is entered through it and is
    // left through its back face exactly as by a ray with clean state words.
    QD_DIAG("\n# diag: layered wall, the next wall (eps 9.0 - 1.20j, 100 mm) after the glued exit, fed the words\n");
    QD_DIAG("# the exit wrote, against the same three events with clean words. Words as prev/cur/buf\n");
    QD_DIAG("A,theta_n,event,gain_dB,clean_dB,words_in,words_out,clean_out,rtype,clean_rtype\n");
    for (int first : firsts)
        for (double thd : {0.0, 10.0})
        {
            double th = thd * QPI / 180.0;
            auto X = exitev(4, first, th, 1.5);
            int w[3] = {(int)X.prev_out(0), (int)X.cur_out(0), (int)X.buf_out(0)};
            int z[3] = {0, 0, 0};
            const char *names[3] = {"reflect", "enter  ", "leave  "};
            for (int ev = 0; ev < 3; ++ev)
            {
                Rsu<double> R[2];
                int in[2][3];
                for (int k = 0; k < 2; ++k)
                {
                    int *s = k ? z : w;
                    in[k][0] = s[0], in[k][1] = s[1], in[k][2] = s[2];
                    if (ev == 2)
                        R[k] = make1<double>(mtl, 4, OT_EXIT, 1u, WALL, 0, s[0], s[1], s[2], L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                    else
                        R[k] = make1<double>(mtl, ev ? 4 : 3, OT_ENTRY, 1u, WALL, 0, s[0], s[1], s[2], 0.5, L, 0.5,
                                             QPI / 2.0 - th, 1.5);
                    R[k].run();
                }
                QD_DIAG("%d,%.1f,%s,%.4f,%.4f,%d/%d/%d,%d/%d/%d,%d/%d/%d,%u,%u\n", first, thd, names[ev],
                        dB((double)R[0].gain(0)), dB((double)R[1].gain(0)), in[0][0], in[0][1], in[0][2],
                        (int)R[0].prev_out(0), (int)R[0].cur_out(0), (int)R[0].buf_out(0), (int)R[1].prev_out(0),
                        (int)R[1].cur_out(0), (int)R[1].buf_out(0), (unsigned)R[0].resolved_type[0],
                        (unsigned)R[1].resolved_type[0]);
                INFO("next wall, A " << first << ", theta_n " << thd << ", " << names[ev]);
                CHECK(cpx(R[0].xprmat, 0, 0) == cpx(R[1].xprmat, 0, 0));
                CHECK((int)R[0].resolved_type[0] == (int)R[1].resolved_type[0]);
                CHECK((int)R[0].cur_out(0) == (int)R[1].cur_out(0));
                CHECK((int)R[0].buf_out(0) == (int)R[1].buf_out(0));
                if (ev == 1) // the words the entry wrote feed the exit
                {
                    w[0] = (int)R[0].prev_out(0), w[1] = (int)R[0].cur_out(0), w[2] = (int)R[0].buf_out(0);
                    z[0] = (int)R[1].prev_out(0), z[1] = (int)R[1].cur_out(0), z[2] = (int)R[1].buf_out(0);
                }
            }
        }
}

TEST_CASE("Recovery behind a sheet - the leg after a sheet crossing at the exact in-medium cosine", "[wave]")
{
    // A traced ray holds a real direction inside a lossy host while the propagation cosine there is complex; the
    // recovery returns the complex value by inverting the refraction map of the pair previous medium -> current medium.
    // After a sheet crossing the previous-medium word holds the sheet, and the leg behind it is still at the exact
    // cosine: the exit interface, the host's series, the traversal and a sheet composite on that leg are built at the
    // cosine the entry was evaluated at.
    //
    // Two instruments. End to end, the chains air | S | host | air and air | S | host | S | air, entered through a
    // glued sheet and assembled from what ray_state_update writes, against the characteristic matrix of the whole
    // stack, in power: the displaced air and the free-space reference are pure phases. And per event, the same leg with
    // a sheet in the previous-medium word against the same leg with a clean word, host series declined, which needs no
    // reference.
    //
    // A lossless host has a real cosine and nothing to recover, so its rows sit on the reference whatever the recovery
    // does and are the control on the chain and on the reference.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int LOSSY = 5, CLEAN = 6, SHEET = 3;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const std::complex<double> e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    // Glued entry from air at the external angle, tilted in the xy plane so the segment and the
    // reported angle agree. Returns the event; path_dirN is the direction the tracer then holds.
    auto glued_entry = [&](int host, double th)
    {
        auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, SHEET, host, 0, 0, 0, 0.5, 0.5, 0.0, QPI / 2.0 - th, 1.5);
        C.orig(0, 0) = C.fbs(0, 0) - 0.5 * std::cos(th);
        C.orig(0, 1) = -0.5 * std::sin(th);
        C.dest(0, 0) = C.fbs(0, 0) + 0.5 * std::cos(th);
        C.dest(0, 1) = 0.5 * std::sin(th);
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        C.run();
        return C;
    };
    // A leg of normal thickness L inside the host, closed by the given event, along a direction
    auto leg = [&](uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double th, double eps,
                   double dx, double dy)
    {
        auto C = make1<double>(mtl, 4, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.run();
        return C;
    };

    QD_DIAG("\n# diag: recovery behind a sheet, end to end. air | S | host 100 mm | air (coated) and\n");
    QD_DIAG("# air | S | host 100 mm | S | air (lined), S eps 5.24 - 0.63j 6 mm, scalar, path_dir supplied,\n");
    QD_DIAG("# against the characteristic matrix of the whole stack. Host 6 is lossless and the control.\n");
    QD_DIAG("# r_sheet is the reflection of a sheet backed by air seen from the host; where it exceeds\n");
    QD_DIAG("# one the series clamps it, and clamp_dB is what that costs\n");
    QD_DIAG("chain,host,theta_ext,ct_event,ct_map,chain_dB,ref_dB,d_dB,|r_sheet|,clamp_dB,rtype\n");
    for (int host : {CLEAN, LOSSY})
        for (int lined = 0; lined < 2; ++lined)
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                const std::complex<double> e_h = diag_eps(host, f_hz);
                auto E1 = glued_entry(host, th);
                double dx = (double)E1.path_dirN(0, 0), dy = (double)E1.path_dirN(0, 1);
                auto E2 = lined ? leg(OT_GLUED_EXIT, 2u, SHEET, host, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                      (int)E1.buf_out(0), th, 0.0, dx, dy)
                                : leg(OT_EXIT, 1u, host, 0, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                      (int)E1.buf_out(0), th, 0.0, dx, dy);
                std::complex<double> chain = cpx(E1.xprmat, 0, 0) * cpx(E2.xprmat, 0, 0);

                std::vector<DiagLayer> lay = {{e_s, U, thk}, {e_h, U, L}};
                if (lined)
                    lay.push_back({e_s, U, thk});
                DiagSheet Ref = diag_cm_stack(e_air, U, lay, e_air, U, std::complex<double>(std::cos(th), 0.0), f_hz,
                                              false, true, 0.0);
                double chain_dB = dB(std::norm(chain)), ref_dB = dB(std::norm(Ref.T_te));
                double ct_map = diag_traced_ct(e_h, th);

                // A clamp of a mirror's reflectance into [0, 1] would matter only where a sheet backed by air, seen
                // from a lossy host close to its critical angle, reflects more than one: clamp_dB is what such a clamp
                // would cost, formed from the factorized series.
                std::complex<double> cs = diag_ct_exact(e_h, th);
                std::complex<double> r_s = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_c = (std::abs(r_s) > 1.0) ? r_s / std::abs(r_s) : r_s;
                std::complex<double> r_a = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -1.0) * (2.0 * QPI * f_hz / C0) *
                                                   std::sqrt(e_h) * cs * L);
                double clamp_dB = lined ? 20.0 * std::log10(std::abs(1.0 - r_s * r_s * ph * ph) / std::abs(1.0 - r_c * r_c * ph * ph))
                                        : 20.0 * std::log10(std::abs(1.0 - r_a * r_s * ph * ph) / std::abs(1.0 - r_a * r_c * ph * ph));

                QD_DIAG("%s,%d,%.1f,%.9f,%.9f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%u\n", lined ? "lined " : "coated", host, thd,
                        std::abs(dx), ct_map, chain_dB, ref_dB, chain_dB - ref_dB, std::abs(r_s), clamp_dB,
                        (unsigned)E2.resolved_type[0]);
                INFO((lined ? "lined" : "coated") << ", host " << host << ", theta_ext " << thd);
                CHECK(std::abs(std::abs(dx) - ct_map) < 1.0e-9); // the entry wrote the map the recovery inverts
                CHECK(std::abs(chain_dB - ref_dB) < 1.0e-6);     // on the matrix: no mirror is clamped; clamp_dB is what a clamp would cost
                if (!(host == LOSSY && thd == 85.0))
                    CHECK(clamp_dB == 0.0); // such a clamp would act on the one row and nowhere else
            }

    QD_DIAG("\n# diag: recovery behind a sheet, per event. One leg of the lossy host (eps 1.50 - 0.40j,\n");
    QD_DIAG("# 100 mm) along the direction a tracer holds after entering from air, series declined, with\n");
    QD_DIAG("# a sheet in the previous-medium word against the same leg with a clean word. The bare\n");
    QD_DIAG("# exit is the far face of a coated wall and of a wall with a sheet inside it\n");
    QD_DIAG("event,theta_ext,sheet_dB,clean_dB,d_dB,d_rel\n");
    struct Ev
    {
        const char *name;
        uint8_t ot;
        unsigned nH;
        int m1, m2;
    };
    const std::vector<Ev> evs = {
        {"bare exit ", OT_EXIT, 1u, LOSSY, 0},
        {"glued exit", OT_GLUED_EXIT, 2u, SHEET, LOSSY},
        {"embedded  ", OT_SHEET_ISO, 1u, SHEET, 0},
    };
    const std::complex<double> e_l = diag_eps(LOSSY, f_hz);
    for (const auto &E : evs)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(e_l, th), st = std::sqrt(1.0 - ct * ct);
            auto A = leg(E.ot, E.nH, E.m1, E.m2, SHEET, LOSSY, 0, th, 1.5, ct, st);
            auto B = leg(E.ot, E.nH, E.m1, E.m2, 0, LOSSY, 0, th, 1.5, ct, st);
            std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
            double d_rel = std::abs(a - b) / std::abs(b);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%+.6f,%.3e\n", E.name, thd, dB(std::norm(a)), dB(std::norm(b)),
                    dB(std::norm(a)) - dB(std::norm(b)), d_rel);
            INFO(E.name << ", theta_ext " << thd);
            CHECK(d_rel < 1.0e-12);
        }
    // The recovered cosine does not depend on which lossless medium the wave came from: the
    // equation reads the entry medium only through the real transverse invariant. A leg entered
    // from a dense lossless solid, named in the previous-medium word, against the same leg with
    // a sheet there, where the recovery cannot name the entry medium. The last two rows are past
    // what air could launch, invariant above one. Such a wave cannot leave into air either, so
    // those two are measured on a face that admits it, the transition into the dense solid. The
    // total reflection at the air face is not used: at a recovered cosine its evanescent root
    // has an argument that is real and negative up to rounding, and the branch it takes is not
    // defined by the media.
    QD_DIAG("\n# diag: recovery behind a sheet, the entry medium. A bare exit out of the lossy host along\n");
    QD_DIAG("# the direction a tracer holds after entering from a lossless solid, eps 5.24, with that solid\n");
    QD_DIAG("# in the previous-medium word against a sheet there. p is the transverse invariant; past one\n");
    QD_DIAG("# the face is host | dense solid\n");
    QD_DIAG("p,theta_entry,face,ct_traced,named_dB,named_arg,sheet_dB,sheet_arg,d_rel\n");
    const int DENSE = 2;
    const std::complex<double> e_d = diag_eps(DENSE, f_hz);
    for (double pinv : {0.2, 0.8, 1.2, 1.45})
    {
        double s1 = std::sqrt(pinv / std::real(e_d)), th1 = std::asin(s1);
        std::complex<double> eps_r = e_d / e_l;
        std::complex<double> c2 = std::sqrt(1.0 - eps_r * s1 * s1);
        double eta = std::sqrt(std::abs(eps_r)), r = std::real(c2);
        double ct = r / std::sqrt((eta * s1) * (eta * s1) + r * r), st = std::sqrt(1.0 - ct * ct);
        bool m2m = pinv > 1.0;
        auto A = m2m ? leg(OT_M2M_M1, 2u, LOSSY, DENSE, SHEET, LOSSY, 0, th1, 1.5, ct, st)
                     : leg(OT_EXIT, 1u, LOSSY, 0, SHEET, LOSSY, 0, th1, 1.5, ct, st);
        auto B = m2m ? leg(OT_M2M_M1, 2u, LOSSY, DENSE, DENSE, LOSSY, 0, th1, 1.5, ct, st)
                     : leg(OT_EXIT, 1u, LOSSY, 0, DENSE, LOSSY, 0, th1, 1.5, ct, st);
        std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
        double d_rel = std::abs(a - b) / std::abs(b);
        QD_DIAG("%.2f,%.3f,%s,%.9f,%.6f,%+.4f,%.6f,%+.4f,%.3e\n", pinv, th1 * 180.0 / QPI, m2m ? "dense" : "air", ct,
                dB(std::norm(b)), std::arg(b) * 180.0 / QPI, dB(std::norm(a)), std::arg(a) * 180.0 / QPI, d_rel);
        INFO("entry medium, p " << pinv);
        REQUIRE(std::abs(b) > 0.0);
        CHECK(d_rel < 1.0e-12);
    }
}

TEST_CASE("Measurement - a host entered from a lossy solid across a bare face", "[wave]")
{
    // The recovery inverts the refraction map of the pair previous medium -> current medium, and the other cases enter
    // from air or from a lossless solid. Here the host is entered from a lossy solid across a bare shared face, air |
    // first | host | air, so the previous-medium word names a lossy medium.
    //
    // A ray that starts in air keeps a real transverse wavenumber across parallel faces however lossy the layers are,
    // so the exact cosine in the host is sqrt(1 - sin^2(theta_ext) / eps_h) and the first solid drops out of it. The
    // reference is the single pass through the stack at those cosines, in power, with every series declined (eps 1.5)
    // so that nothing but the three ports and the two traversals is in the chain.
    //
    // The last event is run twice along the same direction: with the state words the transition wrote, which name the
    // first solid, and with a clean previous-medium word, where the recovery solves for a real invariant with no medium
    // named.
    const double f_hz = FRQ, L1 = 0.05, L2 = 0.10, k0 = 2.0 * QPI * f_hz / C0;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    QD_DIAG("\n# diag: measurement, air | first 50 mm | host 100 mm | air across bare faces, scalar, every\n");
    QD_DIAG("# series declined, against the single pass at the exact cosines. named: the last event with the\n");
    QD_DIAG("# words the transition wrote; clean: the same event with a clean previous-medium word.\n");
    QD_DIAG("# ct_host is the traced cosine in the host, ct_one what one refraction from air would give\n");
    QD_DIAG("first,host,theta_ext,ct_host,ct_one,ref_dB,named_dB,d_named,clean_dB,d_clean,prev,rtype\n");
    // first, host. The last three put the solid with the large loss tangent in front, where the
    // projection of the first leg is furthest from a unit vector.
    const int pairs[9][2] = {{2, 5}, {1, 5}, {4, 5}, {2, 6}, {1, 6}, {4, 6}, {5, 1}, {5, 4}, {5, 6}};
    for (const auto &pr : pairs)
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
        {
            const int first = pr[0], host = pr[1];
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);

            // Entry air -> first, from ray_mesh_interact, as the traced-chain case does
            arma::mat orig, dest, fbs, sbs;
            diag_ray_at(thd, orig, dest);
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
            arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                            &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                            &path_dirN, true, (arma::u32_vec *)nullptr);
            REQUIRE(path_dirN.n_rows == 1);
            std::complex<double> a0(xprmatN(0, 0), xprmatN(1, 0));
            double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
            double c1t = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
            double s1t = std::sqrt(std::max(0.0, 1.0 - c1t * c1t));

            // Transition first -> host, in the frame make1 uses (normal along x)
            auto T = make1<double>(mtl, 4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            T.has_path_dir = true;
            T.path_dir_prev.zeros(1, 3);
            T.path_dir_prev(0, 0) = c1t, T.path_dir_prev(0, 1) = s1t;
            T.has_path_dirN = true;
            T.path_dirN.zeros(1, 3);
            T.path_dirN(0, 0) = std::cos(th), T.path_dirN(0, 1) = std::sin(th);
            T.normals(0, 3) = -1.0; // the host's front face: its normal opposes the ray
            T.run();
            double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);

            // Exit host -> air along the direction the transition wrote
            std::complex<double> a2[2];
            Rsu<double> X;
            for (int k = 0; k < 2; ++k)
            {
                X = make1<double>(mtl, 4, OT_EXIT, 1u, host, 0, k ? 0 : (int)T.prev_out(0), (int)T.cur_out(0),
                                  k ? 0 : (int)T.buf_out(0), L2, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                X.has_path_dir = true;
                X.path_dir_prev.zeros(1, 3);
                X.path_dir_prev(0, 0) = dx, X.path_dir_prev(0, 1) = dy;
                X.run();
                a2[k] = cpx(X.xprmat, 0, 0);
            }
            std::complex<double> a1 = cpx(T.xprmat, 0, 0);

            // Single pass at the exact cosines, written from the media alone
            std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
            std::complex<double> z0(1.0, 0.0), z1 = std::sqrt(e1), z2 = std::sqrt(e2);
            std::complex<double> t01 = 2.0 * z0 * c0 / (z0 * c0 + z1 * c1), t12 = 2.0 * z1 * c1 / (z1 * c1 + z2 * c2),
                                 t20 = 2.0 * z2 * c2 / (z2 * c2 + z0 * c0);
            double ref = std::norm(t01 * t12 * t20) *
                         std::exp(2.0 * k0 * (std::imag(z1 * c1) * L1 + std::imag(z2 * c2) * L2));
            double named = std::norm(a0 * a1 * a2[0]), clean = std::norm(a0 * a1 * a2[1]);
            double ct_one = diag_traced_ct(e2, th);

            QD_DIAG("%d,%d,%.1f,%.9f,%.9f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%d,%u\n", first, host, thd, std::abs(dx), ct_one,
                    dB(ref), dB(named), dB(named) - dB(ref), dB(clean), dB(clean) - dB(ref),
                    mat_of(T.prev_out(0)), (unsigned)X.resolved_type[0]);
            INFO("first " << first << ", host " << host << ", theta_ext " << thd);
            REQUIRE(mat_of(T.prev_out(0)) == first);
            REQUIRE(std::isfinite(named));
            // Every row, the lossless first solid being the control on the chain and on the
            // reference: the ray is not killed, the word names nothing the recovery reads, the
            // host direction is the one a single refraction from air writes, and the chain is
            // on the single pass.
            CHECK(named > 0.0);
            CHECK(std::abs(dB(named) - dB(clean)) < 1.0e-9);
            CHECK(std::abs(std::abs(dx) - ct_one) < 1.0e-9);
            CHECK(std::abs(dB(named) - dB(ref)) < 1.0e-6);
            // Bit 14 of the word the transition wrote: set out of a lossy solid whose face
            // was evaluated at the recovered cosine, clear out of a lossless one
            CHECK((((int)T.prev_out(0) & 0x4000) != 0) == (std::imag(e1) != 0.0));
        }
}

TEST_CASE("Recovery behind a lossy solid - EM, a second transition and a sheet inside the host", "[wave]")
{
    // The consumers of the recovered cosine the bare-face measurement does not reach. All three
    // chains start air | first | host across bare faces with every series declined (eps 1.5), the
    // entry from ray_mesh_interact, and are compared with the single pass at the exact cosines
    // sqrt(1 - sin^2(theta_ext) / eps), which no layer in front changes.
    //
    //   EM       the exit port per channel, VV being TE and HH being TM in this frame
    //   3 solids air | A | B | C | air: the port and the written direction of a transition out of
    //            a leg that was itself entered from a solid, and a second leg behind a named solid
    //   sheet    air | first | host | sheet | host | air: the composite of a sheet reached from
    //            inside the host, both ports, and the leg behind it
    //
    // A lossless first solid is the control on each chain and on its reference.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, thk = 0.006;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const uint8_t OT_SHEET_ISO = 67;
    const int SHEET = 3;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Ent
    {
        std::complex<double> te, tm; // VV and HH of the entry; scalar: both the one coefficient
        double c, s;                 // traced cosine and sine in the first solid
    };
    auto entry = [&](int itype, int first, double thd) -> Ent
    {
        arma::mat orig, dest, fbs, sbs;
        diag_ray_at(thd, orig, dest);
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                        &path_dirN, true, (arma::u32_vec *)nullptr);
        REQUIRE(path_dirN.n_rows == 1);
        Ent E;
        E.te = std::complex<double>(xprmatN(0, 0), xprmatN(1, 0));
        E.tm = (itype < 3) ? std::complex<double>(xprmatN(6, 0), xprmatN(7, 0)) : E.te;
        double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
        E.c = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
        E.s = std::sqrt(std::max(0.0, 1.0 - E.c * E.c));
        return E;
    };
    // One event of ray_state_update closing a leg of normal thickness L along (dx, dy). A
    // media-to-media event takes its plane normal from the second face, reversed to oppose the ray.
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double L,
                     double th, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        if (ot == OT_M2M_M1)
            C.normals(0, 3) = -1.0;
        C.run();
        return C;
    };
    // Power of one port at an exact cosine, per channel, and of one normal-thickness decay
    auto port = [](std::complex<double> ea, std::complex<double> eb, std::complex<double> ca, bool scalar, double &pte, double &ptm)
    {
        std::complex<double> te, tm;
        diag_port(ea, eb, ca, scalar, te, tm);
        pte = std::norm(te), ptm = std::norm(tm);
    };
    auto decay = [&](std::complex<double> e, std::complex<double> c, double L)
    { return std::exp(2.0 * k0 * std::imag(std::sqrt(e) * c) * L); };
    const std::complex<double> e0(1.0, 0.0);

    // ---- EM ---------------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, EM. air | first 50 mm | host 100 mm | air, per channel,\n");
    QD_DIAG("# chain minus the single pass at the exact cosines, dB\n");
    QD_DIAG("first,host,theta_ext,ref_te_dB,ref_tm_dB,d_te,d_tm,rtype\n");
    for (int host : {5, 6})
        for (int first : {2, 1})
            for (double thd : {40.0, 60.0, 75.0, 85.0})
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(1, first, thd);
                auto T = event(1, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, E.c, E.s);
                auto X = event(1, OT_EXIT, 1u, host, 0, (int)T.prev_out(0), (int)T.cur_out(0), (int)T.buf_out(0), L2, th,
                               (double)T.path_dirN(0, 0), (double)T.path_dirN(0, 1));
                double ch_te = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                double ch_tm = std::norm(E.tm * cpx(T.xprmat, 0, 3) * cpx(X.xprmat, 0, 3));
                double a[2], b[2], c[2];
                port(e0, e1, c0, false, a[0], a[1]), port(e1, e2, c1, false, b[0], b[1]), port(e2, e0, c2, false, c[0], c[1]);
                double dec = decay(e1, c1, L1) * decay(e2, c2, L2);
                double r_te = a[0] * b[0] * c[0] * dec, r_tm = a[1] * b[1] * c[1] * dec;
                QD_DIAG("%d,%d,%.1f,%.6f,%.6f,%+.6f,%+.6f,%u\n", first, host, thd, dB(r_te), dB(r_tm), dB(ch_te) - dB(r_te),
                        dB(ch_tm) - dB(r_tm), (unsigned)X.resolved_type[0]);
                INFO("EM, first " << first << ", host " << host << ", theta_ext " << thd);
                CHECK(ch_te > 0.0);
                CHECK(std::abs(dB(ch_te) - dB(r_te)) < 1.0e-6);
                CHECK(std::abs(dB(ch_tm) - dB(r_tm)) < 1.0e-6);
            }

    // ---- Three solids -------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, a second transition. air | A 50 mm | B 100 mm | C 100 mm | air,\n");
    QD_DIAG("# scalar. d_T2 is the transition B -> C alone against its port and its leg; ct_C the traced\n");
    QD_DIAG("# cosine in C and ct_one what one refraction from air gives; d_named and d_clean the chain\n");
    QD_DIAG("# with the last event fed the words the transition wrote and a clean previous-medium word\n");
    QD_DIAG("A,B,C,theta_ext,ct_C,ct_one,ref_dB,d_T2,d_named,d_clean,rtype\n");
    const int triples[4][3] = {{2, 6, 2}, {1, 5, 4}, {5, 1, 6}, {4, 5, 1}};
    for (const auto &tr : triples)
        for (double thd : {40.0, 60.0, 75.0, 85.0})
        {
            const int A = tr[0], B = tr[1], C = tr[2];
            const double LA = 0.05, LB = 0.10, LC = 0.10;
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            std::complex<double> eA = diag_eps(A, f_hz), eB = diag_eps(B, f_hz), eC = diag_eps(C, f_hz);
            std::complex<double> c0(std::cos(th), 0.0), cA = std::sqrt(1.0 - s2 / eA), cB = std::sqrt(1.0 - s2 / eB),
                                                        cC = std::sqrt(1.0 - s2 / eC);
            Ent E = entry(4, A, thd);
            auto T1 = event(4, OT_M2M_M1, 2u, A, B, 0, A, 0, LA, th, E.c, E.s);
            auto T2 = event(4, OT_M2M_M1, 2u, B, C, (int)T1.prev_out(0), (int)T1.cur_out(0), (int)T1.buf_out(0), LB, th,
                            (double)T1.path_dirN(0, 0), (double)T1.path_dirN(0, 1));
            double dx = (double)T2.path_dirN(0, 0), dy = (double)T2.path_dirN(0, 1);
            double pw[2];
            Rsu<double> X;
            for (int k = 0; k < 2; ++k)
            {
                X = event(4, OT_EXIT, 1u, C, 0, k ? 0 : (int)T2.prev_out(0), (int)T2.cur_out(0), k ? 0 : (int)T2.buf_out(0),
                          LC, th, dx, dy);
                pw[k] = std::norm(E.te * cpx(T1.xprmat, 0, 0) * cpx(T2.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
            }
            double p01, pAB, pBC, pC0, u;
            port(e0, eA, c0, true, p01, u), port(eA, eB, cA, true, pAB, u), port(eB, eC, cB, true, pBC, u), port(eC, e0, cC, true, pC0, u);
            double ref = p01 * pAB * pBC * pC0 * decay(eA, cA, LA) * decay(eB, cB, LB) * decay(eC, cC, LC);
            double d_T2 = dB(std::norm(cpx(T2.xprmat, 0, 0))) - dB(pBC * decay(eB, cB, LB));
            double ct_one = diag_traced_ct(eC, th);
            QD_DIAG("%d,%d,%d,%.1f,%.9f,%.9f,%.6f,%+.6f,%+.6f,%+.6f,%u\n", A, B, C, thd, std::abs(dx), ct_one, dB(ref), d_T2,
                    dB(pw[0]) - dB(ref), dB(pw[1]) - dB(ref), (unsigned)X.resolved_type[0]);
            INFO("three solids " << A << " | " << B << " | " << C << ", theta_ext " << thd);
            REQUIRE(mat_of(T2.prev_out(0)) == B);
            CHECK(pw[0] > 0.0);
            CHECK(std::abs(d_T2) < 1.0e-6);
            CHECK(std::abs(std::abs(dx) - ct_one) < 1.0e-9);
            CHECK(std::abs(dB(pw[0]) - dB(pw[1])) < 1.0e-9);
            CHECK(std::abs(dB(pw[0]) - dB(ref)) < 1.0e-6);
        }

    // ---- A sheet inside the host --------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, a sheet inside the host. air | first 50 mm | host 50 mm |\n");
    QD_DIAG("# sheet 6 mm eps 5.24 - 0.63j | host 50 mm | air, scalar. d_ev is the sheet event against the\n");
    QD_DIAG("# composite at the exact cosine and the host leg in front of it; d_twin the same event against\n");
    QD_DIAG("# its twin with a clean previous-medium word; d_chain the transmitted chain end to end\n");
    QD_DIAG("first,host,theta_ext,port,ev_dB,d_ev,d_twin,d_chain\n");
    const std::complex<double> e_s = diag_eps(SHEET, f_hz);
    for (int host : {5, 6})
        for (int first : {2, 1})
            for (double thd : {40.0, 60.0, 75.0, 85.0})
                for (int rp = 0; rp < 2; ++rp)
                {
                    const double L1 = 0.05, La = 0.05, Lb = 0.05;
                    double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                    std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                    Ent E = entry(4, first, thd);
                    auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, E.c, E.s);
                    double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                    auto S = event(rp ? 3 : 4, OT_SHEET_ISO, 1u, SHEET, 0, (int)T.prev_out(0), (int)T.cur_out(0),
                                   (int)T.buf_out(0), La, th, dx, dy);
                    auto S0 = event(rp ? 3 : 4, OT_SHEET_ISO, 1u, SHEET, 0, 0, (int)T.cur_out(0), 0, La, th, dx, dy);
                    DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e2, thk, c2, f_hz, false, true);
                    double ev = std::norm(cpx(S.xprmat, 0, 0)), tw = std::norm(cpx(S0.xprmat, 0, 0));
                    // Both passes close the host leg with the normal-thickness decay
                    double leg = decay(e2, c2, La);
                    double ev_ref = std::norm(rp ? Ref.R_te : Ref.T_te) * leg;
                    double d_chain = 0.0;
                    if (!rp)
                    {
                        auto X = event(4, OT_EXIT, 1u, host, 0, (int)S.prev_out(0), (int)S.cur_out(0), (int)S.buf_out(0), Lb,
                                       th, (double)S.path_dirN(0, 0), (double)S.path_dirN(0, 1));
                        double p01, p12, p20, u;
                        port(e0, e1, c0, true, p01, u), port(e1, e2, c1, true, p12, u), port(e2, e0, c2, true, p20, u);
                        double ref = p01 * p12 * std::norm(Ref.T_te) * p20 * decay(e1, c1, L1) * decay(e2, c2, La + Lb);
                        double ch = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(S.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                        d_chain = dB(ch) - dB(ref);
                    }
                    QD_DIAG("%d,%d,%.1f,%s,%.6f,%+.6f,%+.6f,%+.6f\n", first, host, thd, rp ? "R" : "T", dB(ev), dB(ev) - dB(ev_ref),
                            dB(ev) - dB(tw), d_chain);
                    INFO("sheet inside the host, first " << first << ", host " << host << ", theta_ext " << thd << ", port " << rp);
                    REQUIRE(mat_of(S.prev_out(0)) == SHEET);
                    CHECK(std::abs(dB(ev) - dB(tw)) < 1.0e-9);
                    CHECK(std::abs(dB(ev) - dB(ev_ref)) < 1.0e-6);
                    CHECK(std::abs(d_chain) < 1.0e-6);
                }
}

TEST_CASE("Recovery behind a lossy solid - a face that was evaluated at a real angle keeps the solid named", "[wave]")
{
    // Where the recovery declined on the leg in the first
    // solid, the shared face was evaluated at a real angle in that solid, and a real angle in a
    // lossy medium is a complex transverse invariant, e1 sin^2(theta_1). The cosine in the host is
    // then sqrt(1 - (e1 / e2) sin^2(theta_1)), which is what the solve with the first solid named
    // returns and what a solve for a real invariant does not.
    //
    //   wedge     the first solid entered from air with its faces known not parallel, so its own
    //             recovery declines; the transition is ray_state_update's own and writes the
    //             direction. theta is the external angle.
    //   launched  a wave that starts inside the first solid at the real angle theta there. No
    //             direction reaches the transition, so the direction in the host is written here
    //             as ray_mesh_interact writes it, from qd_refract at the real angle.
    //
    // The last event, host | air, is compared with its port at that cosine and its traversal, with
    // the first solid in the previous-medium word (named) and with a clean word (clean).
    const double f_hz = FRQ, L1 = 0.05, L2 = 0.10;
    auto mtl = diag_palette<double>();
    const int FIRST = 1;
    const std::complex<double> e1 = diag_eps(FIRST, f_hz), e0(1.0, 0.0);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    QD_DIAG("\n# diag: measurement, a host entered across a face evaluated at a real angle in the lossy first\n");
    QD_DIAG("# solid, eps 5.24 - 0.63j, scalar. theta is external for the wedge and inside the first solid\n");
    QD_DIAG("# for the launched wave; |p| is the magnitude of the transverse invariant. The last event,\n");
    QD_DIAG("# host 100 mm | air, against its port and traversal at the cosine the face was evaluated at\n");
    QD_DIAG("route,host,theta,theta_1,|p|,ct_host,ref_dB,d_named,d_clean,prev,rtype\n");
    for (int route = 0; route < 2; ++route)
        for (int host : {5, 6})
            for (double thd : (route ? std::vector<double>{10.0, 20.0, 23.0, 25.0} : std::vector<double>{40.0, 60.0, 75.0, 85.0}))
            {
                double th = thd * QPI / 180.0;
                std::complex<double> e2 = diag_eps(host, f_hz);
                double c1r, s1r, dx, dy;
                int prev_named = FIRST;
                if (route == 0)
                {
                    // traced direction in the first solid, one refraction from air
                    c1r = diag_traced_ct(e1, th), s1r = std::sqrt(1.0 - c1r * c1r);
                    auto T = make1<double>(mtl, 4, OT_M2M_M1, 2u, FIRST, host, (int)enc(0, true), FIRST, 0, L1, 0.5, 0.5,
                                           QPI / 2.0 - th, 1.5);
                    T.has_path_dir = true;
                    T.path_dir_prev.zeros(1, 3);
                    T.path_dir_prev(0, 0) = c1r, T.path_dir_prev(0, 1) = s1r;
                    T.has_path_dirN = true;
                    T.path_dirN.zeros(1, 3);
                    T.path_dirN(0, 0) = std::cos(th), T.path_dirN(0, 1) = std::sin(th);
                    T.normals(0, 3) = -1.0;
                    T.run();
                    dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                    prev_named = (int)T.prev_out(0);
                    REQUIRE(mat_of(T.prev_out(0)) == FIRST);
                }
                else
                    c1r = std::cos(th), s1r = std::sin(th);

                // The cosine the face was evaluated at, and the direction qd_refract writes from it
                std::complex<double> eps_r = e1 / e2, c2w = std::sqrt(1.0 - eps_r * s1r * s1r);
                double eta = std::sqrt(std::abs(eps_r)), rr = std::real(c2w);
                double ct_map = rr / std::sqrt((eta * s1r) * (eta * s1r) + rr * rr);
                if (route == 1)
                    dx = ct_map, dy = std::sqrt(1.0 - ct_map * ct_map);

                double pw[2];
                Rsu<double> X;
                for (int k = 0; k < 2; ++k)
                {
                    X = make1<double>(mtl, 4, OT_EXIT, 1u, host, 0, k ? 0 : prev_named, host, 0, L2, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                    X.has_path_dir = true;
                    X.path_dir_prev.zeros(1, 3);
                    X.path_dir_prev(0, 0) = dx, X.path_dir_prev(0, 1) = dy;
                    X.run();
                    pw[k] = std::norm(cpx(X.xprmat, 0, 0));
                }
                std::complex<double> te, tm;
                diag_port(e2, e0, c2w, true, te, tm);
                double ct_h = std::abs(dx);
                double ref = std::norm(te) * std::norm(diag_medium_vbs(e2, L2 / ct_h, ct_h, L2, c2w, f_hz));
                QD_DIAG("%s,%d,%.1f,%.3f,%.4f,%.9f,%.6f,%+.6f,%+.6f,%d,%u\n", route ? "launched" : "wedge   ", host, thd,
                        std::acos(c1r) * 180.0 / QPI, std::abs(e1) * s1r * s1r, ct_h, dB(ref), dB(pw[0]) - dB(ref),
                        dB(pw[1]) - dB(ref), mat_of((short)prev_named), (unsigned)X.resolved_type[0]);
                INFO((route ? "launched" : "wedge") << ", host " << host << ", theta " << thd);
                CHECK(std::abs(ct_h - ct_map) < 1.0e-9); // the direction is the real-angle map
                REQUIRE(std::isfinite(pw[0]));
                REQUIRE(std::isfinite(pw[1]));
                // The word carries no flag, so the recovery keeps the solid named and the event sits on the cosine the
                // face was evaluated at. The clean column is printed: it is what a solve for a real invariant makes of
                // this wave.
                CHECK((prev_named & 0x4000) == 0);
                CHECK(pw[0] > 0.0);
                CHECK(std::abs(dB(pw[0]) - dB(ref)) < 1.0e-6);
            }
}

TEST_CASE("Sheet inside a host - the mirror behind the ray is backed by the host", "[wave]")
{
    // The state half of the backing question. After a sheet is crossed inside a host the
    // previous-medium word holds the sheet, and the next close-out of the host resolves it as a
    // layer backed by air. Behind a sheet inside a host is the host. Two events: the crossing,
    // host | sheet | host, and the bare far face of the host fed the three state words the
    // crossing wrote. The series is the resolving run (eps 0) over the declined one (eps 1.5) and
    // is compared with the factorized form, the sheet a layer from the characteristic matrix,
    // once backed by the host and once by air.
    //
    // No path_dir is supplied, so every boundary is at the fbs cosine and the propagation factor
    // is the hybrid form diag_phi reproduces. A coating, air | sheet | host, is the control: its
    // sheet is backed by air and has to stay there.
    const double f_hz = FRQ, thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71;
    const std::complex<double> e_h = diag_eps(HOST, f_hz);

    QD_DIAG("\n# diag: sheet inside a host, the host's series on the leg behind it. Host eps 1.50 - 0.40j,\n");
    QD_DIAG("# sheet 6 mm, scalar. S_applied is the resolving run over the declined one; S_host has the\n");
    QD_DIAG("# sheet behind the ray backed by the host, S_air by air. kind 0 is the sheet inside the host,\n");
    QD_DIAG("# kind 1 the coating, whose sheet is backed by air\n");
    QD_DIAG("kind,sheet,L_mm,theta_n,port,|S_applied|,|S_host|,|S_air|,d_host_dB,d_air_dB,|dS_host|,|dS_air|,prev\n");
    for (int kind = 0; kind < 2; ++kind)
        for (int sheet : {SHEET, NEAR})
            for (double L : {0.05, 0.10})
                for (double thd : {0.0, 20.0, 40.0, 50.0})
                    for (int port = 0; port < 2; ++port)
                    {
                        double th = thd * QPI / 180.0;
                        const std::complex<double> e_s = diag_eps(sheet, f_hz);
                        auto A = kind ? make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, HOST, 0, 0, 0, 0.5, 0.5, 0.5,
                                                      QPI / 2.0 - th, 1.5)
                                      : make1<double>(mtl, 4, OT_SHEET_ISO, 1u, sheet, 0, 0, HOST, 0, L, 0.5, 0.5,
                                                      QPI / 2.0 - th, 1.5);
                        A.run();
                        INFO("kind " << kind << ", sheet " << sheet << ", L " << L << ", theta_n " << thd << ", port " << port);
                        REQUIRE(mat_of(A.cur_out(0)) == HOST);

                        std::complex<double> v[2];
                        uint8_t rt = 0;
                        for (int k = 0; k < 2; ++k)
                        {
                            auto B = make1<double>(mtl, port ? 3 : 4, OT_EXIT, 1u, HOST, 0, (int)A.prev_out(0),
                                                   (int)A.cur_out(0), (int)A.buf_out(0), L, 0.5, 0.5, QPI / 2.0 - th,
                                                   k ? 0.0 : 1.5);
                            B.run();
                            v[k] = cpx(B.xprmat, 0, 0);
                            rt = B.resolved_type[0];
                        }
                        REQUIRE(rt_bit(rt, 2));
                        std::complex<double> S = v[1] / v[0];

                        std::complex<double> cs(std::cos(th), 0.0);
                        std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                        std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                        // inside the host the sheet is centred on its face; a lining lies in the air beyond
                        std::complex<double> r_host = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_h, U, cs, f_hz, false, true).R_te;
                        std::complex<double> r_air = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, true, true).R_te;
                        std::complex<double> S_host = 1.0 / (1.0 - r_ahead * r_host * phi * phi);
                        std::complex<double> S_air = 1.0 / (1.0 - r_ahead * r_air * phi * phi);
                        QD_DIAG("%d,%d,%.0f,%.1f,%s,%.9f,%.9f,%.9f,%+.4f,%+.4f,%.3e,%.3e,%d\n", kind, sheet, L * 1000.0, thd,
                                port ? "R" : "T", std::abs(S), std::abs(S_host), std::abs(S_air),
                                20.0 * std::log10(std::abs(S) / std::abs(S_host)),
                                20.0 * std::log10(std::abs(S) / std::abs(S_air)), std::abs(S - S_host),
                                std::abs(S - S_air), mat_of(A.prev_out(0)));
                        CHECK(std::abs(S_host - S_air) > 1.0e-3); // the row can tell the two backings apart
                        if (kind)
                            CHECK(std::abs(S - S_air) < 1.0e-9); // a coating is backed by air
                        else
                            CHECK(std::abs(S - S_host) < 1.0e-9); // a sheet inside a host by the host
                    }
}

TEST_CASE("ray_state_update - the material index is 14 bits", "[wave]")
{
    // Bit 14 of each state word is a flag, so a material index runs to 16383.
    auto pal = diag_palette<double>();
    auto big = [&](size_t n)
    {
        auto m = pal;
        for (auto &kv : m)
            kv.second.resize(n, kv.second[5]); // pad with the lossless solid, material 6
        return m;
    };
    {
        // The largest index, as the current medium, with the resolved flag beside it
        auto m = big(16383);
        auto C = make1<double>(m, 4, OT_EXIT, 1u, 16383, 0, 0, 16383, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(C.run());
        CHECK((int)C.prev_out(0) == 16383);
        auto D = make1<double>(m, 4, OT_EXIT, 1u, 16383, 0, 0, (int)enc(16383, true), 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(D.run());
        CHECK((int)D.prev_out(0) == 16383);
    }
    {
        // One material more does not fit the index
        auto m = big(16384);
        auto C = make1<double>(m, 4, OT_EXIT, 1u, 6, 0, 0, 6, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_THROWS_AS(C.run(), std::invalid_argument);
    }
    {
        // Bit 14 of the previous-medium word is a flag and not part of the index
        auto C = make1<double>(pal, 4, OT_EXIT, 1u, 5, 0, 0x4000 | 3, 5, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(C.run());
    }
}

TEST_CASE("Lumped columns - an ordered stack: what the core carries, by matrix and by scattering", "[wave]")
{
    // The characteristic-matrix core carries the two columns it can express in a stack, and the scattering form the
    // three it cannot.
    //
    // Carried. The interface gains become a product, one per layer entered plus the far medium's where it is newly
    // entered, on the transmit port only. A resonance is part of each layer's permittivity, at its faces and in its
    // phase.
    //
    // alpha is carried as well, folded into the layer phase as nepers over the path thk / |Re c|. It has two statements
    // here. Against the reference, which carries it the same way. And against produce mode, with no reference at all: a
    // sheet of eps 5.24 - 0.63j with 100 dB/m stacked on its twin without the column is 12 mm of one permittivity with
    // no boundary in the middle, so every wave crosses both halves and the stack is the single 12 mm sheet at 50 dB/m,
    // on both ports. One is resolved by the core and the other by produce mode.
    //
    // tf and m have no matrix expression. They are resolved in the scattering form - tf per boundary by the pair rule,
    // a mass-law layer as the end of a coherent segment - and the last block asserts the row with tf against that
    // reference. The row with m is in "Lumped columns - an ordered stack: ... - the mass-law part".
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    const int A = 1, B = 2, A_ATT = 3, B_ATT = 4, A_RES = 5, B_ALPHA = 6, B_TF = 7;
    const int A_ALPHA = 9, A_THICK = 10;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk},                                      // 1 A
        {1.50, 0.40, thk},                                      // 2 B
        {5.24, 0.63, thk, 3.0},                                 // 3 A_ATT    3 dB
        {1.50, 0.40, thk, 2.0},                                 // 4 B_ATT    2 dB
        {5.24, 0.63, thk, 0.0, 0.0, 1.2, 5.0, 0.8},             // 5 A_RES    resonant
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0},       // 6 B_ALPHA  50 dB/m
        {1.50, 0.40, thk, 0.0, 0.25},                           // 7 B_TF     tf 0.25
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0},  // 8 B_M  mass law
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0},      // 9 A_ALPHA  100 dB/m
        {5.24, 0.63, 2.0 * thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0}, // 10 A_THICK 12 mm at 50 dB/m
    };
    auto mtl = lump_palette(pal);
    auto E = [&](int idx)
    { return lump_eps(pal[(size_t)idx - 1], f_hz); };
    auto Rz = [&](int idx)
    { return lump_res(pal[(size_t)idx - 1], f_hz); };
    auto Al = [&](int idx)
    { return pal[(size_t)idx - 1].alpha; };

    const uint8_t OT_STACK_FBS = 71; // FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // SBS layer traversed first

    auto run = [&](int itype, uint8_t otype, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0,
                               0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // ---- Carried: att on both layers, a resonance in one, and alpha in one. ----
    struct Carry
    {
        const char *name;
        int m1, m2; // the two faces, FBS and SBS
        double g;   // product of the layer gains
    };
    const double G_A = std::pow(10.0, -0.1 * 3.0), G_B = std::pow(10.0, -0.1 * 2.0);
    const std::vector<Carry> carried = {
        {"att 3 dB | att 2 dB", A_ATT, B_ATT, G_A * G_B},
        {"resonant | plain   ", A_RES, B, 1.0},
        {"plain | alpha 50   ", A, B_ALPHA, 1.0},
    };

    QD_DIAG("\n# diag: ordered stack carrying att, a resonance and alpha, air | 6 mm | 6 mm | air.\n");
    QD_DIAG("# paid_dB is the EM transmit gain against the same stack with neither column\n");
    QD_DIAG("stack,order,theta_n,ref_|T|,ref_|R|,dT,dR,d_gain_T,d_gain_R,paid_dB\n");

    for (const auto &S : carried)
        for (int ord = 0; ord < 2; ++ord)
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> ci(std::cos(th), 0.0);
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;

                std::vector<DiagLayer> lay = {{E(first), U, thk, Rz(first), Al(first)},
                                              {E(second), U, thk, Rz(second), Al(second)}};
                DiagSheet Xs = diag_cm_stack(e_air, U, lay, e_air, U, ci, f_hz, false, true);
                DiagSheet Xe = diag_cm_stack(e_air, U, lay, e_air, U, ci, f_hz, false, false);
                double ig = std::sqrt(S.g);

                auto Ct = run(4, ot, S.m1, S.m2, th), Ce = run(1, ot, S.m1, S.m2, th);
                auto Cr = run(3, ot, S.m1, S.m2, th), Cx = run(0, ot, S.m1, S.m2, th);
                auto C0 = run(1, ot, A, B, th); // neither column

                // The reflect port with att: what entered a layer and came back paid that layer's
                // gain once. A matrix has no per-layer echo to charge, so that row takes its
                // reflect port from the scattering reference, which carries a gain per layer.
                if (S.g != 1.0)
                {
                    std::vector<DiagLayer> lg = lay;
                    lg[0].g = (first == A_ATT) ? G_A : G_B, lg[1].g = (second == A_ATT) ? G_A : G_B;
                    DiagSheet Ys = diag_sc_stack(e_air, U, 0.0, lg, e_air, U, 0.0, ci, f_hz, false, true);
                    DiagSheet Ye = diag_sc_stack(e_air, U, 0.0, lg, e_air, U, 0.0, ci, f_hz, false, false);
                    CHECK(std::abs(Ys.R_te - Xs.R_te) > 1.0e-3); // and the gain is worth something on it
                    Xs.R_te = Ys.R_te, Xe.R_te = Ye.R_te, Xe.R_tm = Ye.R_tm;
                }

                double gT = S.g * 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
                double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));
                double dT = std::abs(cpx(Ct.xprmat, 0, 0) - ig * Xs.T_te);
                double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e,%.3e,%+.6f\n", S.name, ord ? "2|1" : "1|2",
                        thd, std::abs(ig * Xs.T_te), std::abs(Xs.R_te), dT, dR,
                        std::abs((double)Ce.gain(0) - gT), std::abs((double)Cx.gain(0) - gR),
                        10.0 * std::log10((double)Ce.gain(0) / (double)C0.gain(0)));

                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                REQUIRE(Ct.resolved_type[0] != 0);
                REQUIRE(Cr.resolved_type[0] != 0);

                // Each layer's gain once on the transmit port, once on what came back out of it
                // on the reflect port, and the resonance in each layer's permittivity, scalar and EM.
                CHECK(dT < 1.0e-9);
                CHECK(dR < 1.0e-9);
                CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
                CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);

                if (S.g != 1.0)
                {
                    // The gains are one real factor on the transmit port. A stack with att takes
                    // the scattering form and its twin without the column the matrix, two forms
                    // of one product, so the ratio is at the rounding floor and not exact.
                    CHECK(std::abs((double)Ce.gain(0) / (double)C0.gain(0) - S.g) < 1.0e-12);
                    // A gain dropped, or one layer's paid twice, is worth dB.
                    CHECK(std::abs((double)Ce.gain(0) - gT / G_A) > 1.0e-3 * gT);
                    CHECK(std::abs((double)Ce.gain(0) - gT * G_B) > 1.0e-3 * gT);
                }
                else
                {
                    // The column moved the stack: the plain reference is excluded by a margin.
                    DiagSheet Xp = diag_cm_stack(e_air, U, {{E(first), U, thk}, {E(second), U, thk}},
                                                 e_air, U, ci, f_hz, false, true);
                    CHECK(std::abs(cpx(Ct.xprmat, 0, 0) - Xp.T_te) > 1.0e-3);
                }
            }

    // ------------------------------------------------------------------------------------------
    // alpha against produce mode. The core's two-layer stack and produce mode's single sheet are
    // the same object, so the two solvers are measured against each other through the library.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: alpha, the core against produce mode. stack 6 mm at 100 dB/m | 6 mm at 0,\n");
    QD_DIAG("# eps 5.24 - 0.63j, against one 12 mm sheet at 50 dB/m. moved_dB is what the column\n");
    QD_DIAG("# is worth on the stack's transmit gain\n");
    QD_DIAG("order,theta_n,|T_te|,|R_te|,d_T,d_R,moved_dB\n");
    for (int ord = 0; ord < 2; ++ord)
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto one = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, 67, 1u, A_THICK, 0, 0, 0, 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto St = run(1, ot, A_ALPHA, A, th), Sr = run(0, ot, A_ALPHA, A, th);
            auto Pt = one(1), Pr = one(0);
            auto S0 = run(1, ot, A, B, th), Sa = run(1, ot, A, B_ALPHA, th);

            double d_T = std::max(std::abs(cpx(St.xprmat, 0, 0) - cpx(Pt.xprmat, 0, 0)),
                                  std::abs(cpx(St.xprmat, 0, 3) - cpx(Pt.xprmat, 0, 3)));
            double d_R = std::max(std::abs(cpx(Sr.xprmat, 0, 0) - cpx(Pr.xprmat, 0, 0)),
                                  std::abs(cpx(Sr.xprmat, 0, 3) - cpx(Pr.xprmat, 0, 3)));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.3e,%.3e,%+.6f\n", ord ? "2|1" : "1|2", thd,
                    std::abs(cpx(St.xprmat, 0, 3)), std::abs(cpx(Sr.xprmat, 0, 3)), d_T, d_R,
                    10.0 * std::log10((double)Sa.gain(0) / (double)S0.gain(0)));

            INFO("alpha against produce mode, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            REQUIRE(St.resolved_type[0] != 0);
            REQUIRE(Pt.resolved_type[0] != 0);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);

            // And the column is worth tenths of a dB on a stack, so neither side is passing on an
            // alpha that was never read.
            CHECK((double)Sa.gain(0) < 0.99 * (double)S0.gain(0));
        }

    // ---- The two columns the matrix cannot hold, on the second layer, both orders, both passes. ----
    struct Decl
    {
        const char *name;
        int m2;
    };
    const std::vector<Decl> declined = {
        {"tf 0.25     ", B_TF},
    };

    QD_DIAG("\n# diag: ordered stack whose second sheet carries tf, resolved in the scattering\n");
    QD_DIAG("# form. gain_plain is the same stack without the column\n");
    QD_DIAG("column,order,theta_n,gain_plain_T,gain_T,gain_R,d_T,d_R,d_scalar,rtype_T,rtype_R\n");

    for (const auto &S : declined)
        for (int ord = 0; ord < 2; ++ord)
            for (double thd : {0.0, 45.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> ci(std::cos(th), 0.0);
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : A, second = ord ? A : S.m2;

                auto C0 = run(1, ot, A, B, th);
                auto Ct = run(1, ot, A, S.m2, th), Cr = run(0, ot, A, S.m2, th);
                auto St = run(4, ot, A, S.m2, th), Sr = run(3, ot, A, S.m2, th);

                std::vector<DiagLayer> lay = {lump_layer(pal[(size_t)first - 1], f_hz),
                                              lump_layer(pal[(size_t)second - 1], f_hz)};
                DiagSheet Xe = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, false);
                DiagSheet Xs = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, true);
                double d_T = std::max(std::abs(cpx(Ct.xprmat, 0, 3) - Xe.T_te), std::abs(cpx(Ct.xprmat, 0, 0) - Xe.T_tm));
                double d_R = std::max(std::abs(cpx(Cr.xprmat, 0, 3) - Xe.R_te), std::abs(cpx(Cr.xprmat, 0, 0) - Xe.R_tm));
                double d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Xs.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Xs.R_te));

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.9f,%.3e,%.3e,%.3e,%u,%u\n", S.name, ord ? "2|1" : "1|2", thd,
                        (double)C0.gain(0), (double)Ct.gain(0), (double)Cr.gain(0), d_T, d_R, d_S,
                        (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                REQUIRE(C0.resolved_type[0] != 0);
                CHECK((double)C0.gain(0) > 1.0e-3);

                // Resolved on both passes and in both modes, and on the reference.
                CHECK(Ct.resolved_type[0] != 0);
                CHECK(Cr.resolved_type[0] != 0);
                CHECK(St.resolved_type[0] != 0);
                CHECK((double)Ct.gain(0) > 1.0e-3);
                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(d_S < 1.0e-9);
            }
}

TEST_CASE("Total reflection of the outer pair - the sheet stays in the result", "[wave]")
{
    // Where the near | far pair is past its critical angle the sheet stays in the result. Total reflection of the outer
    // pair says one thing only, that no Fresnel wave propagates in the far medium; it says nothing about the sheet.
    //
    // In EM this is a coated solid at a steep in-medium angle. Under the acoustic mapping, eps = (Z_air / Z)^2, it is
    // every sheet on a rigid wall at every angle: air | concrete has a critical angle of 0.003 deg.
    //
    // Reflect port. The wave enters the sheet, turns around at the far face and comes back, which is the ordinary
    // composite with a totally reflecting far face - an absorber on a rigid backing. Asserted against
    // diag_sheet_composite, whose reflect port never needed a forward port in the far medium.
    //
    // Transmit port. The Fresnel part is zero; what crosses is the tf leak, and a leak that crosses the sheet pays what
    // the sheet charges. A coherent sheet is the composite with the pad of a closed face on its far side, which passes
    // what it absorbs: one rule for both ports, with the series. A mass-law sheet has no wave inside and is the two
    // crossings a traced solid of its material would make, with the mass term between them:
    //
    //     near | sheet      interact_with's transmission, the sheet's interface gain
    //     sheet | far       interact_with's transmission, the far medium's interface gain
    //
    // The displaced host is taken out at the incidence cosine because the far side carries no wave. Asserted in power
    // and complex, on lossless permittivities.
    //
    // Each event is evaluated at the cosine the wave has there: the incidence cosine after a first crossing that was
    // itself a leak, the Snell cosine after an ordinary one. A leak inside a mass-law sheet is not judged from inside
    // it at the far face and is charged what the other direction charges, and a wave that entered by an ordinary
    // crossing carries the sheet's internal series. Both have their own case; the rows here follow the same rules.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name with the suffix "
    // - the mass-law part".
    const double f_hz = FRQ;
    const double L = 0.10;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ---- Reflect port, acoustic: a lossy sheet on a rigid wall, air | sheet | rigid ----
    {
        const std::vector<LumpMtl> pal = {
            {0.31, 0.20, 0.010},            // 1 a porous layer, rarer than air
            {2.50, 0.80, 0.010},            // 2 a dense lossy layer
            {2.401e-9, 0.0, 0.0},           // 3 rigid wall
            {2.401e-9, 0.0, 0.0, 0.0, 0.3}, // 4 rigid wall leaking, tf 0.3
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int sh, wall;
        };
        const std::vector<Row> rows = {{"porous | rigid      ", 1, 3}, {"dense  | rigid      ", 2, 3}, {"porous | rigid tf .3", 1, 4}, {"dense  | rigid tf .3", 2, 4}};
        QD_DIAG("\n# diag: reflect port of a 10 mm lossy sheet on a rigid wall, scalar, air | sheet | rigid.\n");
        QD_DIAG("# alpha_ref is the absorbed fraction 1 - |R|^2 of the reference; 0 deg is below the\n");
        QD_DIAG("# critical angle and is the control row\nstack,theta_n,|R|^2,ref_|R|^2,alpha_ref,d_R,R0_far\n");
        for (const auto &R : rows)
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
                const LumpMtl &Sm = pal[(size_t)R.sh - 1], &Wm = pal[(size_t)R.wall - 1];
                auto C = make1<double>(mtl, 3, OT_GLUED_ENTRY, 2u, R.sh, R.wall, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                DiagLump Lp;
                Lp.tf_far = Wm.tf;
                DiagSheet D = diag_sheet_composite(1.0, lump_eps(Sm, f_hz), lump_eps(Wm, f_hz), Sm.thk, ct, f_hz,
                                                   false, true, 1.0, 1.0, 1.0, Lp);
                // raw slab-side reflectance of the far face, above one past the critical angle
                std::complex<double> zs = std::sqrt(lump_eps(Sm, f_hz)), zf = std::sqrt(lump_eps(Wm, f_hz));
                double R0f = std::norm((zs * D.c_slab - zf * D.c_far) / (zs * D.c_slab + zf * D.c_far));
                double d_R = std::abs(cpx(C.xprmat, 0, 0) - D.R_te);

                QD_DIAG("%s,%.1f,%.9f,%.9f,%.6f,%.3e,%.6f\n", R.name, thd, (double)C.gain(0), std::norm(D.R_te),
                        1.0 - std::norm(D.R_te), d_R, R0f);
                INFO(R.name << ", theta_n " << thd);
                CHECK(R0f <= 1.0);               // the row is outside the clamp
                CHECK(std::norm(D.R_te) < 0.99); // and the sheet absorbs, so a bare mirror cannot pass
                CHECK(d_R < 1.0e-9);
            }
    }

    // ---- Reflect port, EM and scalar: a coated solid past its critical angle, HOST | SH | air ----
    {
        const double thk = 0.006;
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk}, // 1 SH
            {1.50, 0.00, 0.0}, // 2 HOST, lossless: the host leg is a pure phase
        };
        auto mtl = lump_palette(pal);
        const std::complex<double> e_s = lump_eps(pal[0], f_hz), e_h(1.50, 0.0);
        QD_DIAG("\n# diag: reflect port of a coated solid, eps 1.50 | 6 mm eps 5.24 - 0.63j | air, critical\n");
        QD_DIAG("# angle 54.7 deg. loss_dB is the reference TE reflectance, 0 for a bare mirror\n");
        QD_DIAG("theta_n,|R_te|^2,ref_|R_te|^2,loss_dB,d_te,d_scalar,|R_tm|^2,ref_|R_tm|^2\n");
        for (double thd : {40.0, 50.0, 60.0, 70.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Cr = run(0), Sr = run(3);
            DiagSheet D = diag_sheet_composite(e_h, e_s, 1.0, thk, ct, f_hz, true, false);
            double d_te = std::abs(std::norm(TE(Cr.xprmat)) - std::norm(D.R_te));
            double d_sc = std::abs(std::norm(cpx(Sr.xprmat, 0, 0)) - std::norm(D.R_te));
            QD_DIAG("%.1f,%.9f,%.9f,%+.4f,%.3e,%.3e,%.9f,%.9f\n", thd, std::norm(TE(Cr.xprmat)), std::norm(D.R_te),
                    10.0 * std::log10(std::norm(D.R_te)), d_te, d_sc, std::norm(TM(Cr.xprmat)), std::norm(D.R_tm));
            INFO("coated solid, theta_n " << thd);
            CHECK(d_te < 1.0e-9);
            CHECK(d_sc < 1.0e-9);
            // TM past the critical angle: the slab-side reflectance of the closed far face is above one, and nothing
            // caps it.
            CHECK(std::abs(std::abs(TM(Cr.xprmat)) - std::abs(D.R_tm)) < 1.0e-9);
        }
    }

    // ---- Transmit leak through a coherent sheet, EM and scalar: HOST | SH (tf) | air past critical ----
    {
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {
            {5.24, 0.00, thk, 0.0, tf_s}, // 1 lossless sheet with a leak
            {1.50, 0.00, 0.0},            // 2 HOST
        };
        auto mtl = lump_palette(pal);
        QD_DIAG("\n# diag: tf leak through a coated solid past its critical angle, eps 1.50 | 6 mm eps 5.24\n");
        QD_DIAG("# (tf 0.5) | air, lossless throughout\ntheta_n,gain_EM,ref_EM,gain_scalar,ref_scalar,d_complex\n");
        for (double thd : {60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), St = run(4);
            // The wave enters the sheet by an ordinary crossing, so it meets the closed far face at
            // the Snell cosine and carries the sheet's internal series; diag_leak_series is that
            // statement. The host leg the crossing closes is a pure phase, removed with diag_close.
            DiagLeak De = diag_leak_series(1.5, 5.24, 1.0, 0.0, tf_s, 0.0, thk, ct, f_hz, false);
            DiagLeak Ds = diag_leak_series(1.5, 5.24, 1.0, 0.0, tf_s, 0.0, thk, ct, f_hz, true);
            double ref_em = 0.5 * (std::norm(De.T_te) + std::norm(De.T_tm)), ref_sc = std::norm(Ds.T_te);
            std::complex<double> close = diag_close(std::complex<double>(1.5, 0.0), L, th, f_hz);
            double d_cx = std::max(std::max(std::abs(TE(Ct.xprmat) / close - De.T_te), std::abs(TM(Ct.xprmat) / close - De.T_tm)),
                                   std::abs(cpx(St.xprmat, 0, 0) / close - Ds.T_te));

            QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.3e\n", thd, (double)Ct.gain(0), ref_em, (double)St.gain(0), ref_sc, d_cx);
            INFO("leak through a coated solid, theta_n " << thd);
            CHECK(std::abs((double)Ct.gain(0) - ref_em) < 1.0e-9);
            CHECK(std::abs((double)St.gain(0) - ref_sc) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
        }
    }
}

TEST_CASE("Leak chain - each face at the cosine the wave has and the series on a closed far face", "[wave]")
{
    // The transmit port of a sheet whose outer pair is past its critical angle. Two things about
    // it.
    //
    // The cosine. The far face is met at the Snell cosine of the sheet; by transitivity that face
    // is closed whenever the outer pair is, and passes its tf leak or nothing. A coherent sheet
    // rarer than the invariant is a wave medium like any other: the wave in it is evanescent, it
    // tunnels to the far face and decays with the thickness. Only a mass-law sheet, which has no
    // wave inside, is entered as a leak and keeps the incidence cosine.
    //
    // The series. A sheet has no interior the tracer could follow a bounce into, so what the far
    // face reflects has to be summed here or it is lost: the transmit port of a coherent sheet
    // carries the series the reflect port carries. The mass-law route carries none.
    //
    // The rows that evaluate a material with the mass column (m > 0) are in the case of the same name
    // with the suffix " - the mass-law part".
    const double f_hz = FRQ;
    const double L = 0.10;
    const double k0 = 2.0 * QPI * f_hz / C0;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    // ---- No tf anywhere: a closed outer pair transmits nothing, whatever the sheet is ----
    {
        auto mtl = diag_palette<double>(); // 2 = eps 5.24 solid, 8 = eps 1.50 sheet, both lossless
        QD_DIAG("\n# diag: closed outer pair with no tf, eps 5.24 | 6 mm eps 1.50 | air. The pair closes at\n");
        QD_DIAG("# 25.9 deg, the host | sheet face at 32.3 deg; the exact transmit port is zero throughout\n");
        QD_DIAG("theta_n,gain_EM,gain_scalar\n");
        for (double thd : {26.0, 28.0, 30.0, 32.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            auto Ce = make1<double>(mtl, 1, OT_GLUED_EXIT, 2u, 8, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Cs = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 8, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Ce.run(), Cs.run();
            QD_DIAG("%.1f,%.9f,%.9f\n", thd, (double)Ce.gain(0), (double)Cs.gain(0));
            INFO("closed pair, no tf, EM palette, theta_n " << thd);
            CHECK((double)Ce.gain(0) == 0.0);
            CHECK((double)Cs.gain(0) == 0.0);
        }

        const std::vector<LumpMtl> pal = {{0.31, 0.0, 0.010}, {0.05, 0.0, 0.0}};
        auto mta = lump_palette(pal);
        QD_DIAG("\n# diag: the same in the acoustic direction, air | 10 mm eps 0.31 | eps 0.05, scalar. The pair\n");
        QD_DIAG("# closes at 12.9 deg; the sheet | far face judged at the incidence cosine only at 23.7 deg\n");
        QD_DIAG("theta_n,gain\n");
        for (double thd : {13.0, 15.0, 20.0, 23.0, 30.0})
        {
            double th = thd * QPI / 180.0;
            auto C = make1<double>(mta, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.run();
            QD_DIAG("%.1f,%.9f\n", thd, (double)C.gain(0));
            INFO("closed pair, no tf, acoustic palette, theta_n " << thd);
            CHECK((double)C.gain(0) == 0.0);
        }
    }

    // ---- A coherent lossless sheet rarer than the invariant on a closed wall: it tunnels to the pad ----
    {
        const double thk = 0.010, e_s = 0.31, e_c = 2.401e-9, tf_s = 0.3, tf_c = 0.1;
        const std::vector<LumpMtl> pal = {{e_s, 0.0, thk, 0.0, tf_s}, {e_c, 0.0, 0.0, 0.0, tf_c}};
        auto mtl = lump_palette(pal);
        QD_DIAG("\n# diag: coherent lossless sheet entered by a leak, air | 10 mm eps 0.31 (tf 0.3) | rigid\n");
        QD_DIAG("# (tf 0.1), scalar. ref is the two flat leaks, recorded; evan_dB is the Snell-cosine decay;\n");
        QD_DIAG("# d_complex: distance from the tunnelling sheet on the pad of the wall face\ntheta_n,gain,ref,gain_minus_ref_dB,evan_dB,d_complex\n");
        for (double thd : {40.0, 45.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th), s2 = 1.0 - ct * ct;
            auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.run();
            double ref = (1.0 - diag_tf_pair(1.0, 0.0, tf_s)) * (1.0 - diag_tf_pair(1.0, tf_s, tf_c));
            double zs = std::sqrt(e_s), zc = std::sqrt(e_c);
            std::complex<double> c2n(0.0, -std::sqrt(s2 / e_s - 1.0)), c2f(0.0, -std::sqrt((e_s / e_c) * s2 - 1.0));
            double arg_n = std::arg(1.0 + (ct - zs * c2n) / (ct + zs * c2n));
            double arg_f = std::arg(1.0 + (zs * ct - zc * c2f) / (zs * ct + zc * c2f));
            std::complex<double> ref_cx = std::polar(std::sqrt(ref), arg_n + arg_f - k0 * (zs - 1.0) * thk * ct);
            double evan = dB(std::exp(-2.0 * k0 * thk * std::sqrt(s2 - e_s)));
            (void)ref_cx;
            // The sheet is a wave medium and tunnels to the pad of the closed wall face; the two flat leaks of a sheet
            // entered as a leak are printed beside it
            DiagLeak Dp = diag_leak_series(1.0, e_s, e_c, 0.0, tf_s, tf_c, thk, ct, f_hz, true);
            double d_cx = std::abs(cpx(C.xprmat, 0, 0) - Dp.T_te);
            QD_DIAG("%.1f,%.9f,%.9f,%+.4f,%+.4f,%.3e\n", thd, (double)C.gain(0), ref, dB((double)C.gain(0) / ref), evan, d_cx);
            INFO("coherent sheet rarer than the invariant on a closed wall, theta_n " << thd);
            CHECK(std::abs((double)C.gain(0) - std::norm(Dp.T_te)) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
        }
    }

    // ---- The series on a closed far face, EM and scalar: eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air ----
    {
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {{5.24, 0.00, thk, 0.0, tf_s}, {1.50, 0.00, 0.0}};
        auto mtl = lump_palette(pal);
        const std::complex<double> e_h(1.50, 0.0), e_s(5.24, 0.0), e_a(1.0, 0.0);
        QD_DIAG("\n# diag: leak through a coated solid with its series, eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air,\n");
        QD_DIAG("# lossless. S_dB is the series factor of the scalar channel; energy is |R|^2 + |T|^2 in scalar\n");
        QD_DIAG("# mode, which a lossless stack owes exactly\n");
        QD_DIAG("theta_n,gain_EM,ref_EM,gain_scalar,ref_scalar,S_dB,d_complex,energy_scalar\n");
        for (double thd : {60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), St = run(4), Sr = run(3);
            DiagLeak De = diag_leak_series(e_h, e_s, e_a, 0.0, tf_s, 0.0, thk, ct, f_hz, false);
            DiagLeak Ds = diag_leak_series(e_h, e_s, e_a, 0.0, tf_s, 0.0, thk, ct, f_hz, true);
            std::complex<double> close = diag_close(e_h, L, th, f_hz); // the host leg, a pure phase
            double ref_em = 0.5 * (std::norm(De.T_te) + std::norm(De.T_tm)), ref_sc = std::norm(Ds.T_te);
            double d_cx = std::max(std::max(std::abs(TE(Ct.xprmat) / close - De.T_te), std::abs(TM(Ct.xprmat) / close - De.T_tm)),
                                   std::abs(cpx(St.xprmat, 0, 0) / close - Ds.T_te));
            double energy = (double)St.gain(0) + (double)Sr.gain(0);
            QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%+.4f,%.3e,%.12f\n", thd, (double)Ct.gain(0), ref_em,
                    (double)St.gain(0), ref_sc, dB(Ds.S_te), d_cx, energy);
            INFO("leak through a coated solid with its series, theta_n " << thd);
            CHECK(std::abs((double)Ct.gain(0) - ref_em) < 1.0e-9);
            CHECK(std::abs((double)St.gain(0) - ref_sc) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
            CHECK(std::abs(energy - 1.0) < 1.0e-12); // needs no reference
        }
    }

    // ---- The series on a closed far face, acoustic: an absorber without tf on a leaking rigid wall ----
    {
        const double tf_c = 0.3;
        const std::vector<LumpMtl> pal = {
            {2.50, 0.80, 0.010},             // 1 a dense lossy layer
            {0.31, 0.20, 0.010},             // 2 a porous layer, rarer than air
            {2.401e-9, 0.0, 0.0, 0.0, tf_c}, // 3 rigid wall leaking, tf 0.3
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int sh;
            double lim; // the air | sheet face must be an ordinary crossing on the row
        };
        const std::vector<Row> rows = {{"dense  | rigid tf .3", 1, 70.0}, {"porous | rigid tf .3", 2, 70.0}};
        QD_DIAG("\n# diag: leak into a rigid wall (tf 0.3) through a 10 mm absorber without tf, scalar,\n");
        QD_DIAG("# air | sheet | rigid. S_dB is the series factor, the part a three-event chain leaves out\n");
        QD_DIAG("stack,theta_n,T_dB,ref_T_dB,S_dB,d_complex\n");
        for (const auto &R : rows)
            for (double thd : {10.0, 20.0, 45.0, 70.0})
            {
                if (thd > R.lim)
                    continue;
                double th = thd * QPI / 180.0, ct = std::cos(th);
                const LumpMtl &Sm = pal[(size_t)R.sh - 1];
                auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, R.sh, 3, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                DiagLeak D = diag_leak_series(1.0, lump_eps(Sm, f_hz), lump_eps(pal[2], f_hz), 0.0, 0.0, tf_c,
                                              Sm.thk, ct, f_hz, true);
                double d_cx = std::abs(cpx(C.xprmat, 0, 0) - D.T_te);
                QD_DIAG("%s,%.1f,%.4f,%.4f,%+.4f,%.3e\n", R.name, thd, dB((double)C.gain(0)), dB(std::norm(D.T_te)), dB(D.S_te), d_cx);
                INFO(R.name << ", theta_n " << thd);
                CHECK(d_cx < 1.0e-9 * std::abs(D.T_te) + 1.0e-12);
            }
    }

    // ---- The step at the critical angle of the outer pair (printed) ----
    {
        // Just below, the composite; just above, the chain. With the series on both sides and the far face at one
        // cosine on both sides, the scalar step closes. In EM the near crossing is per channel in the chain as in the
        // composite, and the table prints what is left.
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {{5.24, 0.00, thk, 0.0, tf_s}, {1.50, 0.00, 0.0}};
        auto mtl = lump_palette(pal);
        const double thc = std::asin(std::sqrt(1.0 / 1.5)) * 180.0 / QPI, dth = 1.0e-5;
        auto run = [&](int itype, double thd)
        {
            auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - thd * QPI / 180.0, 1.5);
            C.run();
            return C;
        };
        auto Sb = run(4, thc - dth), Sa = run(4, thc + dth), Eb = run(1, thc - dth), Ea = run(1, thc + dth);
        double st_s = dB((double)Sa.gain(0)) - dB((double)Sb.gain(0));
        double st_te = dB(std::norm(TE(Ea.xprmat))) - dB(std::norm(TE(Eb.xprmat)));
        double st_tm = dB(std::norm(TM(Ea.xprmat))) - dB(std::norm(TM(Eb.xprmat)));
        QD_DIAG("\n# diag: step from the open to the closed far face at the critical angle of the outer pair,\n");
        QD_DIAG("# above minus below in dB, +- 1e-5 deg\nstack,theta_c,step_scalar_dB,step_te_dB,step_tm_dB\n");
        QD_DIAG("eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air,%.4f,%+.4f,%+.4f,%+.4f\n", thc, st_s, st_te, st_tm);
        CHECK(std::abs(st_s) < 0.01);
        CHECK(std::abs(st_te) < 0.5);
        CHECK(std::abs(st_tm) < 0.5);
    }
}
