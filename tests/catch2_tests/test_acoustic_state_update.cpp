// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_state_update, acoustic: cases that evaluate a material with the mass column (m > 0).

#include "ray_test_helpers.hpp"

// A palette with the side column at zero: the infinite plate (md/acoustic.md 2.1), for the cases whose statement is
// written for it - a bonded run, the mass cosine, a resonance, the bending stiffness, a closed plate, a peaked face -
// rather than for the radiation of a finite panel ("Finite plate - ...")
static inline std::unordered_map<std::string, std::vector<double>> side0(std::unordered_map<std::string, std::vector<double>> P)
{
    size_t n = 0;
    for (const auto &kv : P)
        n = std::max(n, kv.second.size());
    P["side"] = std::vector<double>(n, 0.0);
    return P;
}

// One body of a bonded mass-law plate, for diag_bonded_TL: the mass column, the reference frequency and
// the normal thickness, exactly the three quantities Material::medium_gain forms its term from. The
// column is a switch (M10): m > 0 is the limp mass law at 20 dB per decade, whatever its value.
struct DiagMass
{
    double m, fRef, thk;
};

// Mass term of a bonded plate, in dB, written from the column definitions. Bonded bodies move with
// one velocity, so their transfer impedances are in series and what adds is the impedance ratio
//     X0 = (f / fRef) M
// of each body, not its term in dB. The plate is then one body: A = 20 log10(sum X0), with the
// single-body angle shape. A body contributes its X0 unclipped, so the hard zero sits on the sum;
// a body without the column (m = 0) is absent.
static inline double diag_bonded_TL(const std::vector<DiagMass> &bodies, double f_hz, double cosv)
{
    double X = 0.0;
    for (const auto &B : bodies)
    {
        double arg = (f_hz * 1e-9 / B.fRef) * B.thk;
        if (B.m > 0.0 && arg > 0.0)
            X += arg;
    }
    if (!(X > 1.0) || !(cosv > 0.0))
        return 0.0;
    double A = 20.0 * std::log10(X);
    if (cosv >= 1.0)
        return A;
    double x = X * cosv;
    return A * std::log10(1.0 + x * x) / std::log10(1.0 + X * X);
}

TEST_CASE("ray_state_update - mass-law material is excluded from slab resolution", "[acoustic]")
{
    // A mass-law slab carries a lumped transmission surrogate: eta/mu are calibrated for the
    // surface impedance only, so the in-slab phase of phi is not physical and the internal series
    // must not be resummed. This is a material property, not an accuracy threshold, so the gate
    // holds even at eps = 0.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map0<double>(M, ind, mtl);
    mtl["m"] = {10.0};

    const double F10 = 10.0e9;
    const double L = 40.0 * half_wave(2.0, F10);

    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                           std::complex<double>(0.5, 0.3), F10);
    C.run();

    // Measure medium_gain over L the same way, with eps high so no S can appear
    auto D = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 1.5,
                           std::complex<double>(0.5, 0.3), F10);
    D.run();
    double g_L = std::norm(vvf(D));

    double k0 = 2.0 * QPI * F10 / C0;
    std::complex<double> close = std::sqrt(g_L) * std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L));

    CHECK(std::abs(vvf(C) / close - 1.0) < 1e-6); // bare close-out, S = 1
    // Not RT_EXIT_SLAB. The wave held in the mass-law medium leaves it into air as refracted from
    // air (md/acoustic.md 10, A7): an ordinary exit, the same type at normal incidence as off it.
    check_rtype(C, RT_EXIT);
    auto C1 = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0 - 0.1 * QPI / 180.0, 0.0,
                            std::complex<double>(0.5, 0.3), F10);
    C1.run();
    check_rtype(C1, RT_EXIT);

    // Teeth: without the gate this slab would have resolved to a clearly non-unit S
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    CHECK(std::abs(airy_S(r, r, phi_one_way(2.0, g_L, F10, L)) - 1.0) > 1e-2);
}

TEST_CASE("Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space", "[acoustic]")
{
    // The other consequence of a declining core, and the quieter one. A host layer whose
    // previous-medium word holds a sheet resolves its internal series against that sheet as a
    // LAYER, through qd_stack_resolve. Where the sheet carries m or tf the core returns false and
    // fresnel_r keeps the half-space reflection it computed first: no thickness, no structure,
    // nothing behind it. That is the treatment the core was added to replace, and the ray is not
    // killed, so nothing marks the row.
    //
    // What it costs is the gap between two columns of the backing table in the plan, and it is not
    // uniformly a loss. The half-space is exact when whatever is behind the sheet has the sheet's
    // own permittivity and is 1.560 dB off in |S| when it is air; the air-backed layer is the
    // other way round. So a declining sheet moves a coating with air behind it from 0 to 1.560 dB
    // of error, and a sheet with a matching solid behind it from 1.560 dB to 0. The first is the
    // motivating geometry.
    //
    // The fallback is a step: the decline tests the column for being non-zero and not for
    // mattering. alpha is carried by the core, and the alpha rows here are the guard on that: a
    // negligible alpha, 1e-6 dB/m - 6e-9 dB across this sheet - has to leave the sheet a layer, on
    // the layer series to the same bound as the sheet without it.
    //
    // tf is carried as well: the sheet is a layer, with tf by the pair rule on both of its faces,
    // and the row is asserted against the layer series built from the scattering reference. The
    // half-space column records what a decline would cost.
    //
    // A mass-law sheet keeps the half-space, and that is not a fallback. Its reflect port is its
    // near interface and nothing else, by the mass-law convention, so the half-space series IS
    // its series. Asserted against the half-space, to 1e-9, with the half-space and layer series
    // far enough apart that neither assertion could pass by accident.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    const int HOST = 1;
    const std::vector<LumpMtl> pal = {
        {1.50, 0.40, 0.0},                                         // 1 HOST
        {5.24, 0.63, thk},                                         // 2 SHEET        resolves as a layer
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6},        // 3 SHEET alpha  1e-6 dB/m, a layer too
        {5.24, 0.63, thk, 0.0, 0.25},                              // 4 SHEET tf     0.25
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0},     // 5 SHEET m      10
        {1.5015, 0.40, thk},                                       // 6 NEAR         resolves as a layer
        {1.5015, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6},      // 7 NEAR alpha   1e-6 dB/m, a layer too
    };
    auto mtl = lump_palette(pal);
    const std::complex<double> e_h = lump_eps(pal[0], f_hz);

    struct Row
    {
        const char *name;
        int prev;
        bool declines;
    };
    const std::vector<Row> rows = {
        {"SHEET   plain        ", 2, false},
        {"SHEET   alpha 1e-6   ", 3, false},
        {"SHEET   tf 0.25      ", 4, false},
        {"SHEET   m 10         ", 5, true},
        {"LO_NEAR plain        ", 6, false},
        {"LO_NEAR alpha 1e-6   ", 7, false},
    };

    QD_DIAG("\n# diag: host layer bounded by a sheet the core declines on, host eps 1.50 - 0.40j.\n");
    QD_DIAG("# S_layer is the series with the sheet resolved as a layer backed by air, S_hs the\n");
    QD_DIAG("# series against the sheet as a half-space. cost_dB is |S_hs| against |S_layer|, which\n");
    QD_DIAG("# is what the decline moves the host's series by when air is what is behind the sheet\n");
    QD_DIAG("sheet,L_mm,theta_n,|S_applied|,|S_layer|,|S_hs|,d_layer,d_hs,cost_dB,rtype\n");

    double worst_cost = 0.0;
    for (const auto &R : rows)
        for (double L : {0.05, 0.10})
            for (double thd : {0.0, 20.0, 40.0, 50.0})
            {
                double theta_n = thd * QPI / 180.0;
                const std::complex<double> e_s = lump_eps(pal[(size_t)R.prev - 1], f_hz);

                auto C = make1<double>(mtl, 4, OT_EXIT, 1u, HOST, 0, R.prev, HOST, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
                C.run();

                std::complex<double> S_applied = (cpx(C.xprmat, 0, 0) / cpx(C.x_in, 0, 0)) /
                                                 diag_close(e_h, L, theta_n, f_hz);

                std::complex<double> cs(std::cos(theta_n), 0.0);
                std::complex<double> phi = diag_phi(e_h, L, theta_n, f_hz);
                std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> r_hs = diag_r_slabside(e_h, e_s, cs);
                // The layer, with the sheet's tf where it has one; at tf = 0 this is diag_cm_stack
                DiagLayer Ls{e_s, U, thk};
                Ls.tf = pal[(size_t)R.prev - 1].tf;
                DiagSheet B = diag_sc_stack(e_h, U, 0.0, {Ls}, e_air, U, 0.0, cs, f_hz,
                                            false, true, /*undisplaced*/ 0.0);

                std::complex<double> S_layer = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);
                std::complex<double> S_hs = 1.0 / (1.0 - r_ahead * r_hs * phi * phi);
                double d_layer = std::abs(S_applied - S_layer), d_hs = std::abs(S_applied - S_hs);
                double cost = 20.0 * std::log10(std::abs(S_hs) / std::abs(S_layer));
                if (R.declines)
                    worst_cost = std::max(worst_cost, std::abs(cost));

                QD_DIAG("%s,%.1f,%.1f,%.9f,%.9f,%.9f,%.3e,%.3e,%+.4f,%u\n", R.name, L * 1000.0, thd,
                        std::abs(S_applied), std::abs(S_layer), std::abs(S_hs), d_layer, d_hs, cost,
                        (unsigned)C.resolved_type[0]);

                INFO(R.name << ", L " << L * 1000.0 << " mm, theta_n " << thd);
                REQUIRE(C.resolved_type[0] != 0);
                REQUIRE(rt_bit(C.resolved_type[0], 2)); // the host series did resolve
                REQUIRE(std::isfinite(C.gain(0)));

                // The two series are far enough apart that landing on one excludes the other.
                CHECK(std::abs(S_hs - S_layer) > 1.0e-3);

                if (R.declines)
                    CHECK(d_hs < 1.0e-9); // a mass-law sheet: its reflect port is its near interface
                else
                    CHECK(d_layer < 1.0e-9); // a layer: no column, or one the core carries
            }
    QD_DIAG("# worst |cost| over the half-space rows %.4f dB\n", worst_cost);

    // The step is at least the plan's figure for this geometry, 1.560 dB over 0 to 50 deg at 50 mm.
    CHECK(worst_cost > 1.5);
}

TEST_CASE("Stack with a mass-law layer - resolved by segments, not killed", "[acoustic]")
{
    // Item 2, the mass-law half. A mass-law layer is not a coherent layer and is not made one. It
    // ends the coherent segment in front of it as a half-space of its own admittance - which is
    // what its eta and mu are calibrated for - contributes one real factor, and starts the next
    // segment. Nothing returns across it. With no coherent layer at all that is produce mode's
    // mass-law route for a single sheet, which is the first thing asserted, on the reference and
    // through the library, and it holds before the source change.
    //
    // There is no exact reference for a mass-law layer inside a coherent stack, so beyond the
    // convention written a second time the assertions are the ones that need none:
    //
    //   the transmit port is invariant under layer order, exactly where no lossy coherent layer
    //   touches the plate; beside one the plate's open faces are flux transmittances, which next to
    //   a lossy medium are not an energy statement, and the two orders differ by up to 0.03 dB
    //   (md/acoustic.md 7), recorded with that bound;
    //   the reflect port does not read m, and not what lies behind the mass-law layer;
    //   the mass column is a switch (M10): a layer with m 10 pays exactly what it pays with m 20;
    //   a lossless-permittivity stack reflects and transmits no more than it received;
    //   an air-like coherent sheet in front of a mass-law sheet is that mass-law sheet.
    //
    // Scalar only: the mass law is an acoustic column. Two mass-law layers with no coherent layer
    // between them are bonded and pay as one plate, X0 summed; that rule has its own case.
    const double f_hz = FRQ, thk = 0.006;
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const uint8_t OT_STACK_FBS = 71, OT_STACK_SBS = 69;

    const int A = 1, M = 2, M2 = 3, AIRS = 4, AL = 5, ML = 6, M_TF = 7, MB = 8, B = 9;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk},                                                // 1 A      coherent
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 2 M      m 20, argument 6
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.001},     // 3 M2     m 10: the switch, M again
        {1.00, 0.00, thk},                                                // 4 AIRS   an air-like coherent sheet
        {4.00, 0.00, thk},                                                // 5 AL     lossless coherent
        {2.25, 0.00, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 6 ML     lossless, m 20
        {1.50, 0.40, thk, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 7 M_TF   m 20 and tf 0.3
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.001},     // 8 MB     a second mass-law sheet
        {1.50, 0.40, thk},                                                // 9 B      M without its column
    };
    auto mtl = side0(lump_palette(pal));
    auto Lay = [&](int idx)
    { return lump_layer(pal[(size_t)idx - 1], f_hz); };
    auto stk = [&](int itype, uint8_t otype, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };
    auto one = [&](int itype, int m1, double th)
    {
        auto C = make1<double>(mtl, itype, 67, 1u, m1, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // The reference at one mass-law layer against produce mode's own route. Holds today.
    QD_DIAG("\n# diag: mass-law reference at one layer against a single mass-law sheet, eps 1.50 - 0.40j,\n");
    QD_DIAG("# 6 mm, m 20, argument 6\ntheta_n,TL_dB,|T|,|R|,d_T,d_R\n");
    for (double thd : {0.0, 20.0, 45.0, 70.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet X = diag_sc_stack(e_air, U, 0.0, {Lay(M)}, e_air, U, 0.0, ci, f_hz, false, true);
        auto Ct = one(4, M, th), Cr = one(3, M, th);
        double d_T = std::abs(cpx(Ct.xprmat, 0, 0) - X.T_te), d_R = std::abs(cpx(Cr.xprmat, 0, 0) - X.R_te);
        QD_DIAG("%.1f,%.4f,%.9f,%.9f,%.3e,%.3e\n", thd, diag_mass_TL(20.0, 0.001, f_hz, thk, std::cos(th)),
                std::abs(X.T_te), std::abs(X.R_te), d_T, d_R);
        INFO("single mass-law sheet, theta_n " << thd);
        CHECK(d_T < 1.0e-9);
        CHECK(d_R < 1.0e-9);
    }

    struct St
    {
        const char *name;
        int m1, m2;
    };
    const std::vector<St> stacks = {
        {"A | B (control) ", A, B}, {"A | M           ", A, M}, {"A | M_TF        ", A, M_TF},
        {"AL | ML lossless", AL, ML}, {"M | MB both mass", M, MB},
    };

    QD_DIAG("\n# diag: ordered stack with a mass-law layer, air | 6 mm | 6 mm | air, scalar.\n");
    QD_DIAG("# energy is |R|^2 + |T|^2 of the library's two ports\n");
    QD_DIAG("stack,order,theta_n,ref_|T|,ref_|R|,d_T,d_R,d_order,energy,rtype_T,rtype_R\n");
    for (const auto &S : stacks)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);
            std::complex<double> T_ord[2];
            for (int ord = 0; ord < 2; ++ord)
            {
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;
                DiagSheet X = diag_sc_stack(e_air, U, 0.0, {Lay(first), Lay(second)}, e_air, U, 0.0, ci, f_hz,
                                            false, true, -1.0, 0.0, 0.0, std::cos(th));
                auto Ct = stk(4, ot, S.m1, S.m2, th), Cr = stk(3, ot, S.m1, S.m2, th);
                std::complex<double> t = cpx(Ct.xprmat, 0, 0), r = cpx(Cr.xprmat, 0, 0);
                T_ord[ord] = t;
                double d_T = std::abs(t - X.T_te), d_R = std::abs(r - X.R_te);
                double d_o = ord ? std::abs(T_ord[0] - T_ord[1]) : 0.0;
                double en = std::norm(t) + std::norm(r);

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e,%.9f,%u,%u\n", S.name, ord ? "2|1" : "1|2", thd,
                        std::abs(X.T_te), std::abs(X.R_te), d_T, d_R, d_o, en,
                        (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);
                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                CHECK(Ct.resolved_type[0] != 0); // survives
                CHECK(Cr.resolved_type[0] != 0);
                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(std::abs(t) > 1.0e-4); // and carries energy
                if (ord && (S.m1 == A))
                    CHECK(std::abs(20.0 * std::log10(std::abs(T_ord[0]) / std::abs(T_ord[1]))) < 0.03); // beside a lossy layer, the bound
                else if (ord)
                    CHECK(d_o < 1.0e-14); // reciprocity of the transmit port
                if (S.m1 == AL)
                    CHECK(en <= 1.0 + 1.0e-12); // lossless permittivities: nothing is created
            }
        }

    // Statements about the mass term alone
    QD_DIAG("\n# diag: the mass term in a stack. m 10 against m 20 on the same layer (a switch: the same), and an\n");
    QD_DIAG("# air-like sheet in front of the mass-law sheet against that sheet on its own\n");
    QD_DIAG("theta_n,order,TL_dB,paid_ref_dB,paid_dB,d_R_m,d_null_T,d_null_R\n");
    for (double thd : {0.0, 20.0, 45.0, 70.0})
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto T20 = stk(4, ot, A, M, th), T10 = stk(4, ot, A, M2, th);
            auto R20 = stk(3, ot, A, M, th), R10 = stk(3, ot, A, M2, th);
            double TL = diag_mass_TL(20.0, 0.001, f_hz, thk, ct);
            double paid = -10.0 * std::log10((double)T20.gain(0) / (double)T10.gain(0));
            const std::complex<double> ci(ct, 0.0);
            const int f1 = ord ? M : A, s1 = ord ? A : M, f2 = ord ? M2 : A, s2 = ord ? A : M2;
            DiagSheet X20 = diag_sc_stack(e_air, U, 0.0, {Lay(f1), Lay(s1)}, e_air, U, 0.0, ci, f_hz, false, true, -1.0, 0.0, 0.0, ct);
            DiagSheet X10 = diag_sc_stack(e_air, U, 0.0, {Lay(f2), Lay(s2)}, e_air, U, 0.0, ci, f_hz, false, true, -1.0, 0.0, 0.0, ct);
            const double paid_ref = -10.0 * std::log10(std::norm(X20.T_te) / std::norm(X10.T_te));
            double d_Rm = std::abs(cpx(R20.xprmat, 0, 0) - cpx(R10.xprmat, 0, 0));

            // air-like coherent sheet | M against M alone. The stack is centred on the mesh plane
            // over its coherent thickness, so M stands half of thk behind the plane when the
            // air-like sheet is in front and half of thk in front of it otherwise.
            auto Nt = stk(4, ot, AIRS, M, th), Nr = stk(3, ot, AIRS, M, th);
            auto Pt = one(4, M, th), Pr = one(3, M, th);
            std::complex<double> back = std::exp(std::complex<double>(0.0, (ord ? 1.0 : -1.0) * k0 * thk * ct));
            double d_nT = std::abs(cpx(Nt.xprmat, 0, 0) - cpx(Pt.xprmat, 0, 0));
            double d_nR = std::abs(cpx(Nr.xprmat, 0, 0) - cpx(Pr.xprmat, 0, 0) * back);

            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e\n", thd, ord ? "2|1" : "1|2", TL, paid_ref, paid, d_Rm, d_nT, d_nR);
            INFO("mass term, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(paid_ref == 0.0);               // the reference: m is a switch
            CHECK(T20.gain(0) == T10.gain(0));    // and so is the library's, bitwise
            CHECK(TL > 1.0);                      // the term is worth dB, so the row is not passing on a zero term
            CHECK(d_Rm < 1.0e-15);                // the reflect port never reads m
            CHECK(std::abs(cpx(R20.xprmat, 0, 0)) > 1.0e-3);
            CHECK(d_nT < 1.0e-12);
            CHECK(d_nR < 1.0e-12);
        }
}

TEST_CASE("Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids", "[acoustic]")
{
    // A tf leak keeps its direction, so the reverse of a leak that crossed a face at theta is the
    // crossing of that face from the other side at the same theta. Wood (eps 2.1609e-8) on concrete
    // (eps 2.401e-9) has a Snell ratio of 3: from the wood the face is past its critical angle
    // beyond 19.47 deg and passes only the flat leak, from the concrete it never is and is an
    // ordinary crossing. Same face, same two media, same angle, a factor of three.
    //
    // The angle 19.47 deg is formed from the index of a mass-law material, which is calibrated for
    // surface impedance only and is not a propagation index. So a face is not judged from inside a
    // mass-law medium by a wave that is held undeviated: it is charged what the other direction
    // charges at that angle, with the interface gain of the medium entered.
    //
    // Owed between the two directions, and taken out here: the interface gain of the medium newly
    // entered. Recorded, not owed and not removed: on the sheet rows the air | wood face is the
    // settled flat leak on entry (R0 = 1) and a Fresnel crossing on exit (R0 = 1 - 4 z cos), which
    // the plate rule turns into at most 0.001 dB. Everything else has to vanish.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9;
    const double tf_w = 0.2, tf_c = 0.1, att_w = 1.0, att_c = 2.0;
    const double G_w = std::pow(10.0, -0.1 * att_w), G_c = std::pow(10.0, -0.1 * att_c);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    // ------------------------------------------------------------------------------------------
    // As a sheet: air | wood | concrete against concrete | wood | air
    // ------------------------------------------------------------------------------------------
    {
        // The concrete carries no mass column, so the panel alone is the plate in both directions
        // and each event passes the plate rule of its two faces (md/acoustic.md 10 N1). A wall with
        // the column would make panel and wall one plate whose series is paid at the wall's far
        // face, which neither event reaches (the outer-pair case).
        const double thk = 0.020, L = 0.10;
        const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
        const std::vector<LumpMtl> pal = {
            {e_w, 0.0, thk, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 WOOD   panel, m 20
            {e_c, 0.0, 0.0, att_c, tf_c},                                  // 2 CONC0  without its mass law
        };
        auto mtl = side0(lump_palette(pal));
        QD_DIAG("\n# diag: leak reciprocity as a sheet, wood panel (20 mm, tf 0.2, att 1 dB, m 20) on concrete\n");
        QD_DIAG("# (tf 0.1, att 2 dB), scalar. e_minus_x is entry minus exit with the wall's interface gain\n");
        QD_DIAG("# taken out; near_face is the flat first face against the Fresnel exit, the part recorded\n");
        QD_DIAG("theta_n,entry_T_dB,exit_T_dB,e_minus_x_dB,near_face_dB,rest_dB\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th), s2 = 1.0 - ct * ct;
            auto Et = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Xt = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Et.run(), Xt.run();
            double d = dB((double)Et.gain(0) / G_c) - dB((double)Xt.gain(0));

            // The air | wood face: flat on entry past its critical angle, Fresnel on exit at the
            // cosine the exit holds in the wood. At normal incidence both directions are Fresnel.
            // The wood | concrete face is the same number both ways: past the critical angle of
            // air | concrete the leak of the pair, and the plate takes the smaller of its faces.
            double zw = std::sqrt(e_w), cw = std::sqrt(1.0 - (e_c / e_w) * s2), ca = std::sqrt(1.0 - e_c * s2);
            double r = (zw * cw - ca) / (zw * cw + ca);
            double tau = std::pow(10.0, -0.1 * diag_mass_TL(20.0, 0.001, f_hz, thk, ct));
            auto plate = [&](double a)
            { return a * tau / (a + tau - a * tau); };
            double a_wc = 1.0 - diag_tf_pair(1.0, tf_w, tf_c);
            double near_face = (thd > 0.0) ? dB(plate(std::min(1.0 - diag_tf_pair(1.0, 0.0, tf_w), a_wc)) /
                                                plate(std::min(1.0 - diag_tf_pair(r * r, tf_w, 0.0), a_wc)))
                                           : 0.0;

            QD_DIAG("%.1f,%.4f,%.4f,%+.6f,%+.6f,%+.3e\n", thd, dB((double)Et.gain(0)), dB((double)Xt.gain(0)), d, near_face, d - near_face);
            INFO("wood on concrete as a sheet, theta_n " << thd);
            CHECK(std::abs(d) < 0.02);           // the stated bound, the recorded near-face term inside it
            CHECK(std::abs(d - near_face) < 1.0e-9);  // and nothing else: the wood | concrete face is one number
        }
    }

    // ------------------------------------------------------------------------------------------
    // As two traced solids: the wood | concrete face through ray_mesh_interact, both directions
    // ------------------------------------------------------------------------------------------
    {
        // Two adjacent cubes sharing the wall at x = 1. The coefficient of an undeviated crossing
        // is the one ray_mesh_interact writes; ray_state_update replaces it only where the tracer
        // supplied a refracted direction, which a leak does not have.
        const std::vector<LumpMtl> pal = {
            {e_w, 0.0, 0.0, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0}, // 1 WOOD as a solid
            {e_c, 0.0, 0.0, att_c, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0}, // 2 CONC as a solid
        };
        auto mtl = side0(lump_palette(pal));
        arma::mat mesh = quadriga_lib::cube<double>();
        arma::mat cube2 = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
        mesh = arma::join_cols(mesh, cube2);
        arma::uvec mtl_ind(mesh.n_rows);
        for (arma::uword i = 0; i < mesh.n_rows; ++i)
            mtl_ind(i) = (i < mesh.n_rows / 2) ? 1 : 2;

        auto face = [&](bool wood_to_conc, double th)
        {
            double dy = 0.2 * std::tan(th), xa = wood_to_conc ? 0.9 : 1.1, xb = wood_to_conc ? 1.1 : 0.9;
            arma::mat orig = {{xa, -0.4, 0.1}}, dest = {{xb, -0.4 + dy, 0.1}};
            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
            arma::mat origN, destN, xprmatN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN);
            REQUIRE(gainN.n_elem == 1);
            return (double)gainN(0);
        };

        QD_DIAG("\n# diag: leak reciprocity as two traced solids, the wood | concrete face through\n");
        QD_DIAG("# ray_mesh_interact, scalar. Each direction with the interface gain of the medium it enters\n");
        QD_DIAG("# taken out; w2c_minus_c2w has to vanish\n");
        QD_DIAG("theta_n,wood_to_conc,conc_to_wood,w2c_minus_c2w_dB\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            double wc = face(true, th) / G_c, cw = face(false, th) / G_w;
            QD_DIAG("%.1f,%.9f,%.9f,%+.6f\n", thd, wc, cw, dB(wc) - dB(cw));
            INFO("wood | concrete as two traced solids, theta_n " << thd);
            CHECK(cw > 0.5);                               // the face is open from the concrete at every angle
            CHECK(std::abs(dB(wc) - dB(cw)) < 1.0e-9);     // exact: one crossing, read from either side
        }
    }

    // ------------------------------------------------------------------------------------------
    // Guard: a wave the tracer refracted into a mass-law solid is a Fresnel wave and is left alone
    // ------------------------------------------------------------------------------------------
    {
        const double L = 0.10;
        const std::vector<LumpMtl> pal = {
            {4.0, 0.0, 0.0},                                             // 1 plain solid
            {4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0},    // 2 the same with a mass law, term zero at this path
        };
        auto mtl = side0(lump_palette(pal));
        for (double thd : {30.0, 60.0})
        {
            double th = thd * QPI / 180.0, ct = diag_traced_ct(std::complex<double>(4.0, 0.0), th);
            auto run = [&](int M)
            {
                auto C = make1<double>(mtl, 4, OT_EXIT, 1, M, 0, 0, M, 0, L, 2.0, 0.5, QPI / 2.0 - th, 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return (double)C.gain(0);
            };
            INFO("refracted wave in a mass-law solid, theta_ext " << thd);
            double g_plain = run(1), g_mass = run(2);
            CHECK(g_plain > 0.1);
            CHECK(std::abs(g_plain - g_mass) < 1.0e-12);
        }
    }
}

TEST_CASE("Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate", "[acoustic]")
{
    // Item 1. Two mass-law bodies within the co-location distance have no air gap, so they move as
    // one plate and their transfer impedances add: TL = 20 log10(X01 + X02), the pair being one
    // body in every other respect. Each body paying its own term would add the terms in dB,
    // which is the decoupled double leaf.
    //
    // Charging is telescoped. The first body crossed pays its own term, every later one pays
    // TL(X_run + X_own) - TL(X_run), so no event carries a factor above one and the product over
    // the plate is the pair. Where the second body is a traced solid it needs the first one's
    // surface mass at its own close-out: a sheet's is its thk column, a solid's is the normal
    // thickness of the leg just closed, which travels as a logarithmic code in the buffer word
    // under its flag bit. That code is quantized, so the solid | solid rows are bounded at
    // 0.01 dB and everything else at 1e-9.
    //
    // Two instruments. Where both bodies are one material with m = 20, X0 is linear in thickness
    // and "one body of the summed surface mass" is a statement that does not use the rule: the
    // pair is compared with that body through the library. Everywhere else the mass paid is
    // isolated as a ratio against the same bodies with their mass argument taken to nothing
    // (fRef 1e9), which leaves every face, tf and interface gain in place. The pair is one plate:
    // on the twin it passes a, the smaller absorption of its two outer faces, with its mass
    // a tau / (a + tau - a tau) (md/acoustic.md 10 N1), tau = diag_bonded_TL; what is paid above a
    // is the reference.
    //
    // The mass column is a switch (M10): the rigid board and the concrete keep m 30 and pay the limp
    // mass law at 20 dB per decade, which every row with them guards.
    //
    // The ray runs on the x axis and the angle comes from fbs_angle, so a solid of normal
    // thickness d is given the path d / cos.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9, tf_w = 0.2, tf_c = 0.1;
    const double tw = 0.020, Lw = 0.10, Lc = 0.10, fR = 0.001, fZ = 1.0e9;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69, OT_STACK_FBS = 71, OT_STACK_SBS = 69;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    const int P8 = 1, P12 = 2, P20 = 3, P20Z = 4, WS = 5, WSF = 6, WS2 = 7, SB = 8, CONC = 9, CONCZ = 10,
              P8Z = 11, SBZ = 12, CARPET = 13, WSZ = 14;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                  // 1  P8     wood sheet, X0 8
        {e_w, 0.0, 0.012, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                  // 2  P12    X0 12
        {e_w, 0.0, tw, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                     // 3  P20    X0 20
        {e_w, 0.0, tw, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                     // 4  P20Z   P20 without its mass
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                    // 5  WS     wood solid
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR * Lw / (Lw + tw)},   // 6  WSF    WS with P20's mass folded in
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                    // 7  WS2    WS under a second index
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},                  // 8  SB     a rigid board, m 30 (a switch)
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},                    // 9  CONC   concrete solid, m 30
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},                    // 10 CONCZ  without its mass
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                  // 11 P8Z
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},                  // 12 SBZ
        {2.50, 0.80, 0.010},                                                         // 13 CARPET a lossy lining, m 0
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                    // 14 WSZ
        {e_w, 0.0, 0.012, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                  // 15 P12Z
    };
    const int P12Z = 15;
    auto mtl = side0(lump_palette(pal));
    mtl["attB"][(size_t)WS2 - 1] = 1.0; // att is zero, so this separates WS2 from WS and does nothing else

    const std::vector<double> angles = {0.0, 30.0, 60.0, 75.0};
    // what the plate rule pays above a_in, dB: the plate passes a tau / (a + tau - a tau), a the smaller
    // absorption of its outer faces, and a_in is what its first face passed
    // the absorption of a rigid face to air: tf where it is closed, and exactly at normal incidence, where
    // it is open, the pair rule on its Fresnel reflectance, 1 - (1 - tf) R0
    auto face = [](double eps, double tf, double ct)
    {
        const double z = std::sqrt(eps), R0 = ((1.0 - z) / (1.0 + z)) * ((1.0 - z) / (1.0 + z));
        return (ct >= 1.0) ? 1.0 - (1.0 - tf) * R0 : tf;
    };
    auto rule = [&](double a_in, double a_out, double plate_dB)
    {
        const double a = std::min(a_in, a_out), tau = std::pow(10.0, -0.1 * plate_dB);
        return dB(a_in) - dB(a * tau / (a + tau - a * tau));
    };
    const DiagMass b_p8{20.0, fR, 0.008}, b_p20{20.0, fR, tw}, b_sb{30.0, fR, 0.012};
    const DiagMass b_ws{20.0, fR, Lw}, b_conc{30.0, fR, Lc};

    // The reference at one body is the single-body term
    for (double thd : angles)
    {
        double ct = std::cos(thd * QPI / 180.0);
        CHECK(std::abs(diag_bonded_TL({b_p20}, f_hz, ct) - diag_mass_TL(20.0, fR, f_hz, tw, ct)) < 1.0e-12);
        CHECK(std::abs(diag_bonded_TL({b_conc}, f_hz, ct) - diag_mass_TL(30.0, fR, f_hz, Lc, ct)) < 1.0e-12);
    }

    auto stk = [&](int itype, uint8_t ot, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, ot, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };
    auto one = [&](int itype, int m1, double th)
    {
        auto C = make1<double>(mtl, itype, 67, 1u, m1, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // Sheet on sheet
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: bonded mass, sheet on sheet. Wood 8 mm | wood 12 mm against one 20 mm sheet of the\n");
    QD_DIAG("# same wood (m 20, tf 0.2), scalar, both orders. over_dB is what the stack pays above the\n");
    QD_DIAG("# single sheet; sum_minus_pair is what adding in dB would cost\n");
    QD_DIAG("theta_n,order,T_one_dB,T_stack_dB,over_dB,sum_minus_pair,d_T,d_R\n");
    for (double thd : angles)
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto St = stk(4, ot, P8, P12, th), Sr = stk(3, ot, P8, P12, th);
            auto Ot = one(4, P20, th), Or = one(3, P20, th);
            double over = dB((double)Ot.gain(0)) - dB((double)St.gain(0));
            double add = diag_mass_TL(20.0, fR, f_hz, 0.008, ct) + diag_mass_TL(20.0, fR, f_hz, 0.012, ct) -
                         diag_mass_TL(20.0, fR, f_hz, tw, ct);
            double d_T = std::abs(cpx(St.xprmat, 0, 0) - cpx(Ot.xprmat, 0, 0)) / std::abs(cpx(Ot.xprmat, 0, 0));
            double d_R = std::abs(cpx(Sr.xprmat, 0, 0) - cpx(Or.xprmat, 0, 0));
            QD_DIAG("%.1f,%s,%.4f,%.4f,%+.6f,%+.6f,%.3e,%.3e\n", thd, ord ? "2|1" : "1|2", dB((double)Ot.gain(0)),
                    dB((double)St.gain(0)), over, add, d_T, d_R);
            INFO("sheet on sheet, one material, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9); // one sheet of the summed surface mass, complex
            CHECK(d_R < 1.0e-9); // the reflect port is the first face and never read the mass
        }

    QD_DIAG("\n# diag: bonded mass, sheet on sheet, two materials. Wood 8 mm (m 20, X0 8) | board 12 mm\n");
    QD_DIAG("# (m 30, X0 41.6), tf on both. paid is the mass the stack pays, against the rule with a = 0.1 and\n");
    QD_DIAG("# against the terms added in dB\n");
    QD_DIAG("theta_n,order,paid_dB,plate_dB,rule_dB,sum_dB,paid_minus_rule\n");
    for (double thd : angles)
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto St = stk(4, ot, P8, SB, th), Sz = stk(4, ot, P8Z, SBZ, th);
            double paid = dB((double)Sz.gain(0)) - dB((double)St.gain(0));
            double plate = diag_bonded_TL({b_p8, b_sb}, f_hz, ct), ref = rule(face(e_c, tf_c, ct), face(e_c, tf_c, ct), plate);
            double sum = diag_mass_TL(20.0, fR, f_hz, 0.008, ct) + diag_mass_TL(30.0, fR, f_hz, 0.012, ct);
            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%.6f,%+.3e\n", thd, ord ? "2|1" : "1|2", paid, plate, ref, sum, paid - ref);
            INFO("sheet on sheet, two materials, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(std::abs(paid - ref) < 1.0e-9);
            CHECK(std::abs(sum - plate) > 0.1); // the row can tell the two rules apart
        }

    // ------------------------------------------------------------------------------------------
    // Sheet on a traced solid, both directions
    // ------------------------------------------------------------------------------------------
    // Entry is two events: the sheet crossing, then the close-out of the solid's leg at its far
    // face, fed the three state words the crossing wrote. Exit is one event: the host leg closes
    // and the sheet is crossed together.
    struct Chain
    {
        std::complex<double> T; // sheet composite times the solid's close-out factor
        double g;
        int buf;
    };
    auto entry = [&](int sheet, int solid, double L, double th)
    {
        auto E1 = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, solid, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        E1.run();
        auto E2 = make1<double>(mtl, 4, OT_EXIT, 1, solid, 0, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                (int)E1.buf_out(0), L / std::cos(th), 2.0, 0.5, QPI / 2.0 - th, 1.5);
        E2.run();
        Chain R;
        R.T = cpx(E1.xprmat, 0, 0) * vvf(E2);
        R.g = std::norm(R.T), R.buf = (int)E1.buf_out(0);
        return R;
    };
    auto exitc = [&](int sheet, int solid, double L, double th)
    {
        auto X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, sheet, solid, 0, solid, 0, L / std::cos(th), 0.5, 0.5,
                               QPI / 2.0 - th, 1.5);
        X.run();
        Chain R;
        R.T = cpx(X.xprmat, 0, 0), R.g = (double)X.gain(0), R.buf = (int)X.buf_out(0);
        return R;
    };

    QD_DIAG("\n# diag: bonded mass, sheet on solid, one material. Wood 20 mm sheet on 100 mm of the same wood\n");
    QD_DIAG("# against that solid with the sheet's mass folded into its fRef (m 20), scalar. Entry is\n");
    QD_DIAG("# air | sheet | solid and the solid's close-out; exit is solid | sheet | air\n");
    QD_DIAG("theta_n,dir,T_folded_dB,T_pair_dB,over_dB,sum_minus_pair,d_T\n");
    for (double thd : angles)
        for (int dir = 0; dir < 2; ++dir)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Chain P = dir ? exitc(P20, WS, Lw, th) : entry(P20, WS, Lw, th);
            Chain F = dir ? exitc(P20Z, WSF, Lw, th) : entry(P20Z, WSF, Lw, th);
            double add = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(20.0, fR, f_hz, Lw, ct) -
                         diag_mass_TL(20.0, fR, f_hz, tw + Lw, ct);
            double d_T = std::abs(P.T - F.T) / std::abs(F.T);
            QD_DIAG("%.1f,%s,%.4f,%.4f,%+.6f,%+.6f,%.3e\n", thd, dir ? "exit " : "entry", dB(F.g), dB(P.g),
                    dB(F.g) - dB(P.g), add, d_T);
            INFO("sheet on solid, one material, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
        }

    QD_DIAG("\n# diag: bonded mass, wood panel (20 mm, m 20, tf 0.2) on concrete (100 mm, m 30, tf 0.1,\n");
    QD_DIAG("# att 2 dB), scalar. paid is the mass the pair pays in that direction, against the rule with a = 0.1\n");
    QD_DIAG("theta_n,dir,paid_dB,plate_dB,rule_dB,sum_dB,paid_minus_rule\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_p20, b_conc}, f_hz, ct);
        double sum = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(30.0, fR, f_hz, Lc, ct);
        double paid_dir[2];
        for (int dir = 0; dir < 2; ++dir)
        {
            Chain P = dir ? exitc(P20, CONC, Lc, th) : entry(P20, CONC, Lc, th);
            Chain Z = dir ? exitc(P20Z, CONCZ, Lc, th) : entry(P20Z, CONCZ, Lc, th);
            double paid = dB(Z.g) - dB(P.g);
            paid_dir[dir] = paid;
            const double ref = rule(face(e_c, tf_c, ct), face(e_c, tf_c, ct), plate);
            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%.6f,%+.3e\n", thd, dir ? "exit " : "entry", paid, plate, ref, sum, paid - ref);
            INFO("wood panel on concrete, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(std::abs(paid - ref) < 1.0e-9);
        }
        CHECK(std::abs(paid_dir[0] - paid_dir[1]) < 1.0e-9); // the two directions through the pair
        CHECK(sum - plate > 10.0);
    }

    // A lining without the column is absent from the plate: the concrete is a plate of its own term,
    // its outer faces the one to air (tf 0.1) and the one to the lining, which absorbs at least as much
    QD_DIAG("\n# diag: an m = 0 lining on concrete (lossy 10 mm, eps 2.50 - 0.80j). The concrete's mass,\n");
    QD_DIAG("# paid against the rule with its own term and a = 0.1\ntheta_n,dir,paid_dB,own_dB,rule_dB,d\n");
    for (double thd : angles)
        for (int dir = 0; dir < 2; ++dir)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Chain P = dir ? exitc(CARPET, CONC, Lc, th) : entry(CARPET, CONC, Lc, th);
            Chain Z = dir ? exitc(CARPET, CONCZ, Lc, th) : entry(CARPET, CONCZ, Lc, th);
            double paid = dB(Z.g) - dB(P.g), own = diag_mass_TL(30.0, fR, f_hz, Lc, ct), ref = rule(face(e_c, tf_c, ct), face(e_c, tf_c, ct), own);
            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%+.3e\n", thd, dir ? "exit " : "entry", paid, own, ref, paid - ref);
            INFO("m = 0 lining on concrete, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(std::abs(paid - ref) < 1.0e-9);
        }

    // ------------------------------------------------------------------------------------------
    // Solid on solid, both directions, and the leg that begins at a reflection
    // ------------------------------------------------------------------------------------------
    // No interface coefficient is written on these events - no refracted direction is supplied,
    // so the one ray_mesh_interact wrote stands - and the permittivities are lossless, so the
    // applied factor is the legs' share of the plate and a phase: the first face (from air, not in
    // the chain) passed a_in, the tf of the first body, and the legs pay the rest of the rule.
    struct Leg
    {
        double paid; // dB, this event
        int prev, cur, buf;
    };
    auto m2m = [&](int a, int b, double d, double th, int prev, int buf)
    {
        auto C = make1<double>(mtl, 4, OT_M2M_M1, 2u, a, b, prev, a, buf, d / std::cos(th), 2.0, 0.0, QPI / 2.0 - th, 1.5);
        C.run();
        return Leg{-dB(std::norm(vvf(C))), (int)C.prev_out(0), (int)C.cur_out(0), (int)C.buf_out(0)};
    };
    auto far_face = [&](int itype, int b, double d, double th, int prev, int buf)
    {
        auto C = make1<double>(mtl, itype, OT_EXIT, 1, b, 0, prev, b, buf, d / std::cos(th), 2.0, 0.5, QPI / 2.0 - th, 1.5);
        C.run();
        return Leg{-dB(std::norm(vvf(C))), (int)C.prev_out(0), (int)C.cur_out(0), (int)C.buf_out(0)};
    };

    QD_DIAG("\n# diag: bonded mass, solid on solid. Wood 100 mm (m 20, X0 100) and concrete 100 mm (m 30,\n");
    QD_DIAG("# X0 1000), and wood 60 mm | wood 40 mm against one 100 mm solid. first and second are what\n");
    QD_DIAG("# the two close-outs pay; flag is bit 15 of the buffer word after the shared face\n");
    QD_DIAG("pair,theta_n,first_dB,second_dB,paid_dB,plate_dB,rule_dB,sum_dB,paid_minus_rule,flag\n");
    struct SS
    {
        const char *name;
        int a, b;
        double da, db;
        DiagMass ma, mb;
    };
    const std::vector<SS> pairs = {
        {"wood | concrete", WS, CONC, Lw, Lc, b_ws, b_conc},
        {"concrete | wood", CONC, WS, Lc, Lw, b_conc, b_ws},
        {"wood 60 | wood 40", WS, WS2, 0.06, 0.04, {20.0, fR, 0.06}, {20.0, fR, 0.04}},
        {"wood 40 | wood 60", WS2, WS, 0.04, 0.06, {20.0, fR, 0.04}, {20.0, fR, 0.06}},
    };
    for (const auto &S : pairs)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Leg A = m2m(S.a, S.b, S.da, th, 0, 0);
            Leg B = far_face(4, S.b, S.db, th, A.prev, A.buf);
            double paid = A.paid + B.paid, plate = diag_bonded_TL({S.ma, S.mb}, f_hz, ct);
            double own_a = diag_bonded_TL({S.ma}, f_hz, ct), own_b = diag_bonded_TL({S.mb}, f_hz, ct);
            const LumpMtl &Ma = pal[(size_t)S.a - 1], &Mb = pal[(size_t)S.b - 1];
            const double ref = rule(face(Ma.a, Ma.tf, ct), face(Mb.a, Mb.tf, ct), plate);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.3e,%d\n", S.name, thd, A.paid, B.paid, paid, plate, ref,
                    own_a + own_b, paid - ref, (A.buf & 0x8000) ? 1 : 0);
            INFO(S.name << ", theta_n " << thd);
            CHECK(std::abs(A.paid - own_a) < 1.0e-9); // the first body crossed pays its own term
            CHECK(std::abs(paid - ref) < 0.01);       // the thickness code is quantized
            CHECK((A.buf & 0x8000) != 0);             // and travels under the flag
            if (S.a == WS && S.b == WS2)              // one material: one solid of the summed thickness
                CHECK(std::abs(paid - rule(face(e_w, tf_w, ct), face(e_w, tf_w, ct), diag_mass_TL(20.0, fR, f_hz, 0.10, ct))) < 0.01);
        }

    // The reflection at the concrete's far face from inside carries nothing in a traced run (nothing is
    // reflected back into a held plate, md/acoustic.md 10 N2; asserted through the intersector in "Plate
    // rule - a traced mass-law body passes in one path"). What ray_state_update does with such a leg is
    // still bookkeeping that holds: the reflection pass closes the leg and clears the code. The level of
    // the way back is recorded, not asserted: no power travels it.
    QD_DIAG("\n# diag: a bounce inside the second body, wood | concrete. The reflection at the concrete's far\n");
    QD_DIAG("# face closes the first traversal and clears the code; the way back is recorded (it carries nothing\n");
    QD_DIAG("# in a traced run)\n");
    QD_DIAG("theta_n,fwd_dB,refl_buf,back_conc_dB,conc_own_dB,back_wood_dB,back_dB,plate_dB\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_ws, b_conc}, f_hz, ct), own_c = diag_bonded_TL({b_conc}, f_hz, ct);
        Leg A = m2m(WS, CONC, Lw, th, 0, 0);
        Leg R = far_face(3, CONC, Lc, th, A.prev, A.buf);  // reflect at the far face
        Leg C = m2m(CONC, WS, Lc, th, R.prev, R.buf);      // back through the concrete
        Leg W = far_face(4, WS, Lw, th, C.prev, C.buf);    // and out through the wood
        QD_DIAG("%.1f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", thd, A.paid + R.paid, R.buf, C.paid, own_c, W.paid,
                C.paid + W.paid, plate);
        INFO("bounce in the second body, theta_n " << thd);
        CHECK(std::abs(A.paid + R.paid - plate) < 0.01); // the reflection pass closes the leg as the exit does
        CHECK(R.buf == 0x4000);                          // followed, and no mass flag: nothing bonded was crossed on the way back yet
        CHECK(std::abs(C.paid - own_c) < 1.0e-9);        // so the stale previous medium is not read
    }

    // ------------------------------------------------------------------------------------------
    // A solid between two sheets, and a bounce returning to the glued sheet it entered by
    // ------------------------------------------------------------------------------------------
    // Both leave the solid through a glued sheet with a sheet in the previous-medium word. That
    // word is not a medium there: what lies beyond the sheet ahead is air, and the event leaves
    // the ray in air. The mass rule needs nothing new: the host leg hands the sheet ahead its own
    // X0 plus what it was entered through.
    //
    // The mass paid is a ratio of two runs through the same faces, so it cannot see the far
    // medium. The level can: the one-material sandwich is compared with the solid carrying all
    // three masses, entered through a massless sheet and left through a massless sheet with a
    // clean previous-medium word, and the returning bounce with the same exit entered through
    // nothing.
    auto exit_from = [&](int sheet, int solid, double L, double th, int prev, int cur, int buf)
    {
        auto X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, sheet, solid, prev, cur, buf, L / std::cos(th), 0.5, 0.5,
                               QPI / 2.0 - th, 1.5);
        X.run();
        return X;
    };
    auto entry_ev = [&](int sheet, int solid, double th)
    {
        auto E = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, solid, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        E.run();
        return E;
    };

    QD_DIAG("\n# diag: bonded mass, a solid between two sheets, one material. Wood 8 mm | wood 100 mm | wood\n");
    QD_DIAG("# 12 mm against the solid with both sheets' mass folded into its fRef (m 20), scalar, two\n");
    QD_DIAG("# events. The reference leaves through its second sheet with a clean previous-medium word.\n");
    QD_DIAG("# prev, cur and buf are the state words after the exit\n");
    QD_DIAG("theta_n,T_folded_dB,T_pair_dB,over_dB,d_T,prev,cur,buf\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0;
        auto E1 = entry_ev(P8, WS, th), F1 = entry_ev(P8Z, WSF, th);
        auto E2 = exit_from(P12, WS, Lw, th, (int)E1.prev_out(0), (int)E1.cur_out(0), (int)E1.buf_out(0));
        auto F2 = exit_from(P12Z, WSF, Lw, th, 0, WSF, 0);
        std::complex<double> Tp = cpx(E1.xprmat, 0, 0) * cpx(E2.xprmat, 0, 0);
        std::complex<double> Tf = cpx(F1.xprmat, 0, 0) * cpx(F2.xprmat, 0, 0);
        double d_T = std::abs(Tp - Tf) / std::abs(Tf);
        QD_DIAG("%.1f,%.4f,%.4f,%+.6f,%.3e,%d,%d,%d\n", thd, dB(std::norm(Tf)), dB(std::norm(Tp)),
                dB(std::norm(Tf)) - dB(std::norm(Tp)), d_T, mat_of(E2.prev_out(0)), mat_of(E2.cur_out(0)),
                (int)E2.buf_out(0));
        INFO("solid between two sheets, one material, theta_n " << thd);
        check_state(E1, P8, WS, (int)(short)0x8000); // the sheet was crossed, and says so
        CHECK(d_T < 1.0e-9);                         // one body of the summed surface mass, with air beyond
        check_state(E2, P12, 0, 0);                  // and the ray is in air
    }

    QD_DIAG("\n# diag: bonded mass, wood panel 20 mm | concrete 100 mm | rigid board 12 mm (m 20, 30, 30).\n");
    QD_DIAG("# paid is the mass the two events pay, against the rule with the plate of all three and a = 0.1\n");
    QD_DIAG("theta_n,first_dB,second_dB,paid_dB,plate_dB,rule_dB,sum_dB,paid_minus_rule,cur\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        auto E1 = entry_ev(P20, CONC, th), Z1 = entry_ev(P20Z, CONCZ, th);
        auto E2 = exit_from(SB, CONC, Lc, th, (int)E1.prev_out(0), (int)E1.cur_out(0), (int)E1.buf_out(0));
        auto Z2 = exit_from(SBZ, CONCZ, Lc, th, (int)Z1.prev_out(0), (int)Z1.cur_out(0), (int)Z1.buf_out(0));
        double first = dB((double)Z1.gain(0)) - dB((double)E1.gain(0));
        double second = dB((double)Z2.gain(0)) - dB((double)E2.gain(0));
        double plate = diag_bonded_TL({b_p20, b_conc, b_sb}, f_hz, ct);
        const double a = std::min(face(e_w, tf_w, ct), face(e_c, tf_c, ct)), ref = rule(a, a, plate);
        double sum = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(30.0, fR, f_hz, Lc, ct) +
                     diag_mass_TL(30.0, fR, f_hz, 0.012, ct);
        QD_DIAG("%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.3e,%d\n", thd, first, second, first + second, plate, ref, sum,
                first + second - ref, mat_of(E2.cur_out(0)));
        INFO("solid between two sheets, three bodies, theta_n " << thd);
        CHECK(std::abs(first + second - ref) < 1.0e-9);
        CHECK(sum - plate > 10.0);
        check_state(E2, SB, 0, 0);
    }

    // As above: the reflection at the bare far face from inside carries nothing in a traced run, so the
    // way back is bookkeeping only: the state words are asserted, the level is recorded.
    QD_DIAG("\n# diag: a bounce returning to the glued sheet it entered by. Wood panel 20 mm on concrete\n");
    QD_DIAG("# 100 mm: in through the panel, reflected at the bare far face, out through the panel. fwd\n");
    QD_DIAG("# and back are the mass each traversal pays (back recorded: the way back carries nothing); d_bare\n");
    QD_DIAG("# compares the return event with the same exit entered with a clean previous-medium word, recorded\n");
    QD_DIAG("theta_n,fwd_dB,back_dB,plate_dB,refl_prev,refl_buf,back_minus_plate,over_bare_dB,d_bare,cur\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_p20, b_conc}, f_hz, ct);
        auto E1 = entry_ev(P20, CONC, th), Z1 = entry_ev(P20Z, CONCZ, th);
        Leg R = far_face(3, CONC, Lc, th, (int)E1.prev_out(0), (int)E1.buf_out(0));
        Leg Rz = far_face(3, CONCZ, Lc, th, (int)Z1.prev_out(0), (int)Z1.buf_out(0));
        auto X = exit_from(P20, CONC, Lc, th, R.prev, R.cur, R.buf);
        auto Xz = exit_from(P20Z, CONCZ, Lc, th, Rz.prev, Rz.cur, Rz.buf);
        auto B = exit_from(P20, CONC, Lc, th, 0, CONC, 0);
        double fwd = dB((double)Z1.gain(0)) - dB((double)E1.gain(0)) + R.paid - Rz.paid;
        double back = dB((double)Xz.gain(0)) - dB((double)X.gain(0));
        double d_bare = std::abs(cpx(X.xprmat, 0, 0) - cpx(B.xprmat, 0, 0)) / std::abs(cpx(B.xprmat, 0, 0));
        QD_DIAG("%.1f,%.6f,%.6f,%.6f,%d,%d,%+.3e,%+.6f,%.3e,%d\n", thd, fwd, back, plate, R.prev, R.buf, back - plate,
                dB((double)X.gain(0)) - dB((double)B.gain(0)), d_bare, mat_of(X.cur_out(0)));
        INFO("bounce returning to the glued sheet, theta_n " << thd);
        CHECK(std::abs(fwd - plate) < 1.0e-9);  // the way in pays the plate
        CHECK(R.prev == P20);                   // the stale sheet is still in the word
        CHECK(R.buf == 0x4000);                 // but it was not crossed on this traversal: followed, no mass flag
        check_state(X, P20, 0, 0);
    }
}

TEST_CASE("Ports of a mass-law face from inside - the reflect port is the other direction's", "[acoustic]")
{
    // Item 3. A wave held undeviated inside a mass-law medium, at a face towards the lower index,
    // is transmitted as the other direction transmits. Its reflection was still judged from
    // inside, with the critical angle the transmit rule exists to avoid, so the two ports of one
    // face added to more than one: 1.50 at 30 deg for wood | concrete.
    //
    // The reflect port is the reflection the other direction sees, sign reversed, which is
    // Stokes' relation for one face read from its two sides at the cosine the wave holds. For a
    // lossless pair it conserves, in scalar mode and in EM; the EM row is asserted on the
    // TE/TM mean.
    //
    // Through ray_mesh_interact, whose incidence angle is the geometric one, so the wave is
    // always held undeviated there. No interface gain, so the ports are the face alone.
    //
    // Wood and concrete both carry the mass column, so their face is inside one plate (md/acoustic.md
    // 10, N1 and N2): it passes what reaches it and reflects nothing, from either side. The relations
    // above then hold with T 1 and R 0.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0}, // 1 WOOD as a solid
        {e_c, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0}, // 2 CONC as a solid
    };
    auto mtl = lump_palette(pal);
    arma::mat mesh = quadriga_lib::cube<double>();
    arma::mat cube2 = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
    mesh = arma::join_cols(mesh, cube2);
    arma::uvec mtl_ind(mesh.n_rows);
    for (arma::uword i = 0; i < mesh.n_rows; ++i)
        mtl_ind(i) = (i < mesh.n_rows / 2) ? 1 : 2;

    struct Port
    {
        double g;
        std::complex<double> c;
    };
    auto face = [&](int itype, bool wood_to_conc, double th)
    {
        double dy = 0.2 * std::tan(th), xa = wood_to_conc ? 0.9 : 1.1, xb = wood_to_conc ? 1.1 : 0.9;
        arma::mat orig = {{xa, -0.4, 0.1}}, dest = {{xb, -0.4 + dy, 0.1}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN);
        REQUIRE(gainN.n_elem == 1);
        return Port{(double)gainN(0), std::complex<double>(xprmatN(0, 0), xprmatN(1, 0))};
    };

    QD_DIAG("\n# diag: ports of the wood | concrete face from inside the wood (tf 0.2 | 0.1, lossless, no att),\n");
    QD_DIAG("# through ray_mesh_interact. T_in and R_in are the scalar ports from the wood, R_other the\n");
    QD_DIAG("# reflection from the concrete; d_stokes is |r_in + r_other|; sum_EM the EM gains from the wood\n");
    QD_DIAG("theta_n,T_in,R_in,sum,R_other,d_R,d_stokes,sum_EM\n");
    for (double thd : {0.0, 10.0, 19.0, 30.0, 45.0, 70.0})
    {
        double th = thd * QPI / 180.0;
        Port T = face(4, true, th), R = face(3, true, th), Ro = face(3, false, th), To = face(4, false, th);
        Port Te = face(1, true, th), Re = face(0, true, th);
        double d_st = std::abs(R.c + Ro.c);
        QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%.9f\n", thd, T.g, R.g, T.g + R.g, Ro.g, std::abs(R.g - Ro.g),
                d_st, Te.g + Re.g);
        INFO("wood | concrete from inside the wood, theta_n " << thd);
        CHECK(std::abs(T.g - To.g) < 1.0e-12);        // settled: the transmit port is the other direction's
        CHECK(std::abs(Ro.g + To.g - 1.0) < 1.0e-12); // the concrete side conserves, today and after
        CHECK(std::abs(T.g + R.g - 1.0) < 1.0e-12);   // and so does the wood side
        CHECK(std::abs(R.g - Ro.g) < 1.0e-12);        // because it is the same reflectance
        CHECK(d_st < 1.0e-12);                        // with the sign a face has from its other side
        CHECK(std::abs(Te.g + Re.g - 1.0) < 1.0e-12); // EM, on the TE/TM mean
        CHECK(std::abs(T.g - 1.0) < 1.0e-12);         // inside one plate: everything passes
        CHECK(R.g < 1.0e-12);                         // and nothing is reflected
    }
}

TEST_CASE("Bonded mass - a mass-law sheet inside a mass-law wall", "[acoustic]")
{
    // Plate depth, the one case that reaches a result. A wall in two halves with a sheet between
    // them is one plate of three bodies. The first half pays its own term and the sheet the
    // increment, both at the crossing; the second half then knows only the sheet, so it pays
    // almost its whole term again. What it needs is the first half's thickness, which is its own
    // material's and travels as the thickness code.
    //
    // Two events: the crossing inside the wall, and the wall's bare far face fed the three state
    // words the crossing wrote. The mass paid is the ratio against the same bodies with their
    // mass argument taken to nothing. The three bodies are one plate whose outer faces are the
    // wall's: its first face (from air, not in the chain) passed a, the wall's tf, and the legs pay
    // the rest of the plate rule, dB(a) - dB(a tau / (a + tau - a tau)) (md/acoustic.md 10 N1). A
    // sheet without the column separates the halves, by the settled rule, and is the control: each
    // half is a plate of its own. Every sheet differs from its wall in permittivity; one that does
    // not is merged into the host leg and is another case. The mass column is a switch (M10): the
    // board and the concrete keep m 30 and pay the slope-20 law, so a stack of m 20 and m 30 bodies
    // is one plate of one slope, which their rows guard.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9, tf_w = 0.2, tf_c = 0.1, fR = 0.001, fZ = 1.0e9;
    const uint8_t OT_SHEET_ISO = 67;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const int WS = 3, WSZ = 4, SB = 5, SBZ = 6, CONC = 7, CONCZ = 8, P20 = 9, P20Z = 10, CARPET = 11;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 1  P8     wood sheet
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 2  P8Z
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},   // 3  WS     wood solid
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},   // 4  WSZ
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR}, // 5  SB     rigid board
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ}, // 6  SBZ
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},   // 7  CONC   concrete solid
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},   // 8  CONCZ
        {e_w, 0.0, 0.020, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 9  P20    wood sheet
        {e_w, 0.0, 0.020, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 10 P20Z
        {2.50, 0.80, 0.010},                                        // 11 CARPET a lossy layer, m 0
        {1.0e-8, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 12 D8   a denser wood sheet, m 20
        {1.0e-8, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 13 D8Z
    };
    const int D8 = 12, D8Z = 13;
    auto mtl = side0(lump_palette(pal));
    // the wall's face to air (tf where it is closed, the pair rule on its Fresnel reflectance at normal
    // incidence, where it is open), and what the legs of a plate pay above that face
    auto face = [](double eps, double tf, double ct)
    {
        const double z = std::sqrt(eps), R0 = ((1.0 - z) / (1.0 + z)) * ((1.0 - z) / (1.0 + z));
        return (ct >= 1.0) ? 1.0 - (1.0 - tf) * R0 : tf;
    };
    auto rule = [&](double a, double plate_dB)
    {
        const double tau = std::pow(10.0, -0.1 * plate_dB);
        return dB(a) - dB(a * tau / (a + tau - a * tau));
    };

    struct Row
    {
        const char *name;
        int sheet, sheetz, wall, wallz;
        double d1, d2;
        DiagMass ms, mw; // sheet, and the wall per meter of thickness set below
    };
    const std::vector<Row> rows = {
        {"wood 60 | dense 8 | wood 40   ", D8, D8Z, WS, WSZ, 0.06, 0.04, {20.0, fR, 0.008}, {20.0, fR, 0.0}},
        {"wood 50 | board 12 | wood 50  ", SB, SBZ, WS, WSZ, 0.05, 0.05, {30.0, fR, 0.012}, {20.0, fR, 0.0}},
        {"conc 50 | wood 20 | conc 50   ", P20, P20Z, CONC, CONCZ, 0.05, 0.05, {20.0, fR, 0.020}, {30.0, fR, 0.0}},
        {"conc 30 | wood 20 | conc 70   ", P20, P20Z, CONC, CONCZ, 0.03, 0.07, {20.0, fR, 0.020}, {30.0, fR, 0.0}},
        {"conc 50 | m=0 layer | conc 50 ", CARPET, CARPET, CONC, CONCZ, 0.05, 0.05, {0.0, fR, 0.010}, {30.0, fR, 0.0}},
    };

    QD_DIAG("\n# diag: bonded mass, a sheet inside a wall. first is what the crossing pays (the first half\n");
    QD_DIAG("# and the sheet), second what the far face pays (the second half). plate is the three bodies\n");
    QD_DIAG("# as one; the last row has a sheet without the column, which separates the halves\n");
    QD_DIAG("stack,theta_n,first_dB,second_dB,paid_dB,plate_dB,rule_dB,sum_dB,paid_minus_rule,buf_flag\n");
    for (const auto &R : rows)
        for (double thd : {0.0, 30.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            double paid_ev[2] = {0.0, 0.0};
            int buf = 0;
            Rsu<double> A[2];
            for (int z = 0; z < 2; ++z)
            {
                int sh = z ? R.sheetz : R.sheet, wl = z ? R.wallz : R.wall;
                A[z] = make1<double>(mtl, 4, OT_SHEET_ISO, 1u, sh, 0, 0, wl, 0, R.d1 / ct, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                A[z].run();
                auto B = make1<double>(mtl, 4, OT_EXIT, 1u, wl, 0, (int)A[z].prev_out(0), (int)A[z].cur_out(0),
                                       (int)A[z].buf_out(0), R.d2 / ct, 2.0, 0.5, QPI / 2.0 - th, 1.5);
                B.run();
                double sgn = z ? 1.0 : -1.0;
                paid_ev[0] += sgn * dB(std::norm(cpx(A[z].xprmat, 0, 0)));
                paid_ev[1] += sgn * dB(std::norm(vvf(B)));
                if (!z)
                    buf = (int)A[z].buf_out(0);
            }
            DiagMass h1 = R.mw, h2 = R.mw;
            h1.thk = R.d1, h2.thk = R.d2;
            double paid = paid_ev[0] + paid_ev[1];
            double own1 = diag_bonded_TL({h1}, f_hz, ct), own2 = diag_bonded_TL({h2}, f_hz, ct);
            bool massy = R.ms.m > 0.0;
            double plate = massy ? diag_bonded_TL({h1, R.ms, h2}, f_hz, ct) : own1 + own2;
            double sum = own1 + own2 + (massy ? diag_bonded_TL({R.ms}, f_hz, ct) : 0.0);
            const double a = face(pal[(size_t)R.wall - 1].a, pal[(size_t)R.wall - 1].tf, ct);
            const double ref = massy ? rule(a, plate) : rule(a, own1) + rule(a, own2);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.6f,%d\n", R.name, thd, paid_ev[0], paid_ev[1], paid, plate, ref, sum,
                    paid - ref, (buf & 0x8000) ? 1 : 0);
            INFO(R.name << ", theta_n " << thd);
            REQUIRE(mat_of(A[0].prev_out(0)) == R.sheet);
            if (massy)
            {
                CHECK(std::abs(paid_ev[0] - diag_bonded_TL({h1, R.ms}, f_hz, ct)) < 1.0e-9); // the crossing pays two bodies
                CHECK(std::abs(paid - ref) < 0.01);                                           // and the far face the rest of the plate
                CHECK((buf & 0x8000) != 0);
            }
            else
                CHECK(std::abs(paid - ref) < 1.0e-9); // separated: each half a plate of its own
        }
}

TEST_CASE("Leak direction - a closed face hands the wall behind a traced layer the direction the wave has in air", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // A leak carries the transverse wavenumber across a closed face. For a wave that came from
    // air through parallel faces that is the incidence direction, whatever the layer in front
    // made of the wave, and it is the same at every frequency. So a layer on a leaking wall,
    // meshed as a traced solid, hands the wall what the same layer as a sheet hands it: the
    // direction in the wall and behind it, the segment, the mass term and the exit event are
    // the sheet's. Above an invariant of one no direction in air exists and the face hands on
    // the direction the wave has. Through ray_triangle_intersect, ray_mesh_interact and
    // ray_state_update, scalar, both tracers. The wall's series behind a solid and behind a
    // sheet differ by the mirror behind the ray and are recorded; so is the direction behind
    // a wall with tf and no mass column, which is the Snell bend of its mapping in every
    // representation.
    //
    // A. Directions, 3 layers x 4 walls, to 85 deg
    // B. Levels on the same rows: the events up to the wall, the wall's exit, its mass term as a
    //    ratio against the same wall with its mass argument taken to nothing, its series as the
    //    resolving run over the declined one
    // C. The wall | air event behind the traced porous layer run again with what it is handed
    //    changed one thing at a time
    // D. A host entered from a denser solid, invariant p, in front of the leaking wall
    // E. The ray tube at the closed face, refracting tracer
    // S. The traced solid against the sheet, worst over the angles, with every digit
    const double gap = 1.0e-5, f_hz = FRQ, dS = 0.050, dW = 0.200;
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto unit = [](arma::mat d) -> arma::mat
    {
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        return d;
    };
    auto ang = [](const arma::mat &d) -> double // against the x axis, which every face normal is on
    { return std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat o, d, origN, destN, path_dirN, xpr, fbs; // o, d: the segment the event was met on
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
        E.o = orig, E.d = dest;
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
    using ub::face;
    using ub::build;
    // n transmit events in a row, each relaunched on the segment the one before wrote
    auto chain = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int itype, size_t n, arma::mat o, arma::mat d,
                     const Ev *seed, double eps) -> std::vector<Ev>
    {
        std::vector<Ev> out;
        for (size_t q = 0; q < n; ++q)
        {
            Ev E = traced(pal, mesh, mi, itype, o, d, q ? &out[q - 1] : seed, eps);
            out.push_back(E);
            if (E.nh == 0 || E.rt == 0)
                break;
            arma::mat s = unit(E.destN - E.origN);
            o = E.origN, d = E.origN + (1.2 / std::abs(s(0, 0))) * s;
        }
        return out;
    };
    auto g_of = [&](const std::vector<Ev> &v, size_t q)
    { return (q < v.size() && v[q].nh != 0 && v[q].rt != 0) ? dB(cpx(v[q].xpr, 0, 0)) : -999.0; };
    auto a_of = [&](const std::vector<Ev> &v, size_t q) // direction written, deg
    { return (q < v.size() && v[q].nh != 0 && v[q].rt != 0) ? ang(v[q].path_dirN) : -1.0; };
    auto s_of = [&](const std::vector<Ev> &v, size_t q) // relaunched segment, deg
    { return (q < v.size() && v[q].nh != 0 && v[q].rt != 0) ? ang(v[q].destN - v[q].origN) : -1.0; };
    auto df = [](double a, double b)
    { return (a < -900.0 || b < -900.0) ? -999.0 : a - b; };

    const std::vector<LumpMtl> lp = {
        {0.31, 0.20, 0.0},                                             // 1 porous, lossy, rarer than air: solid
        {0.31, 0.20, dS},                                              // 2 the same as a sheet
        {1.50, 0.00, 0.0},                                             // 3 lossless dense layer: solid
        {1.50, 0.00, dS},                                              // 4 the same as a sheet
        {1.50, 0.40, 0.0},                                             // 5 lossy dense layer: solid
        {1.50, 0.40, dS},                                              // 6 the same as a sheet
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001}, // 7 rigid wall, tf 0.3, m 30
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0e9}, // 8 the same, its mass argument taken to nothing
        {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                // 9 rigid wall, tf 0.3, no mass column
        {5.24, 0.63, 0.0},                                             // 10 a second solid that is not rigid: dense, lossy
        {0.20, 0.00, 0.0, 0.0, 0.5},                                   // 11 not rigid, lossless, rarer than air, tf 0.5: closed from 26.6 deg
        {5.24, 0.00, 0.0},                                             // 12 lossless dense solid, for D
    };
    const Pal mtl = lump_palette(lp);
    struct LayerT
    {
        const char *name;
        int solid, sheet;
    };
    struct WallT
    {
        const char *name;
        int w, twin; // twin: the same wall without a mass term, 0 if it has none
    };
    const std::vector<LayerT> layers = {{"porous 0.31-0.20j", 1, 2}, {"lossless 1.50", 3, 4}, {"lossy 1.50-0.40j", 5, 6}};
    const std::vector<WallT> walls = {{"rigid tf m", 7, 8}, {"rigid tf", 9, 0}, {"dense 5.24-0.63j", 10, 0}, {"rare 0.20 tf", 11, 0}};
    const std::vector<double> angles = {10.0, 20.0, 30.0, 37.0, 45.0, 60.0, 75.0, 85.0};

    struct Cfg
    {
        std::vector<Ev> so, sh, ba; // traced solid (3 events), sheet (2), bare wall (2)
    };
    auto run = [&](int tr, const LayerT &Ly, int w, double thd, double eps) -> Cfg
    {
        Cfg C;
        arma::uvec mi, mj, mk;
        arma::mat m1 = build({0.0, dS, dS + gap, dS + dW + gap}, {Ly.solid, Ly.solid, w, w}, {false, true, false, true}, mi);
        arma::mat m2 = build({0.0, gap, dW + gap}, {Ly.sheet, w, w}, {false, false, true}, mj);
        arma::mat m3 = build({0.0, dW}, {w, w}, {false, true}, mk);
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        C.so = chain(mtl, m1, mi, tr, 3, o, d, nullptr, eps);
        C.sh = chain(mtl, m2, mj, tr, 2, o, d, nullptr, eps);
        C.ba = chain(mtl, m3, mk, tr, 2, o, d, nullptr, eps);
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // A. Directions
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed-face hand-on A, directions. A layer 50 mm on a wall 200 mm, scalar, series declined, through the\n");
    QD_DIAG("# intersector. solid: entry, layer | wall, wall | air. sheet: glued entry, wall | air. bare: the wall alone.\n");
    QD_DIAG("# inL: the direction in the layer; inW: the direction written into the wall; segW: the segment in the wall;\n");
    QD_DIAG("# out: the direction written behind the wall; seg: the segment behind it, all deg against the normal.\n");
    QD_DIAG("# ot, rt: hit code and resolved type of layer | wall and of wall | air; w2: prev.cur.buf after layer | wall\n");
    QD_DIAG("tracer,layer,wall,theta,inL,so_inW,so_segW,so_out,so_seg,sh_inW,sh_segW,sh_out,sh_seg,ba_inW,ba_out,ba_seg,ot2,rt2,ot3,rt3,w2\n");
    std::vector<std::string> rowsB, rowsS;
    for (int tr : {4, 5})
        for (const LayerT &Ly : layers)
            for (const WallT &W : walls)
            {
                double w_in = 0.0, w_out = 0.0, w_seg = 0.0, w_exit = 0.0, w_mass = 0.0, w_into = 0.0, w_ser = 0.0, w_bare = 0.0;
                for (double thd : angles)
                {
                    Cfg D = run(tr, Ly, W.w, thd, 1.5), R = run(tr, Ly, W.w, thd, 0.0);
                    INFO("A, tracer " << tr << ", " << Ly.name << " | " << W.name << ", theta " << thd);
                    CHECK(D.so.size() >= 2);
                    CHECK(D.sh.size() >= 1);
                    const bool so3 = D.so.size() == 3;
                    QD_DIAG("%d,%s,%s,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,", tr, Ly.name, W.name, thd, a_of(D.so, 0),
                            a_of(D.so, 1), s_of(D.so, 1), a_of(D.so, 2), s_of(D.so, 2), a_of(D.sh, 0), s_of(D.sh, 0), a_of(D.sh, 1), s_of(D.sh, 1),
                            a_of(D.ba, 0), a_of(D.ba, 1), s_of(D.ba, 1), (D.so.size() > 1) ? (int)D.so[1].ot : -1, (D.so.size() > 1) ? (int)D.so[1].rt : -1,
                            so3 ? (int)D.so[2].ot : -1, so3 ? (int)D.so[2].rt : -1);
                    if (D.so.size() > 1 && D.so[1].nh != 0)
                        QD_DIAG("%04x.%04x.%04x\n", (unsigned)(unsigned short)D.so[1].prev(0), (unsigned)(unsigned short)D.so[1].cur(0),
                                (unsigned)(unsigned short)D.so[1].buf(0));
                    else
                        QD_DIAG("-\n");

                    // The mass term: the wall against its twin without one, each configuration
                    double m_so = -999.0, m_sh = -999.0, m_ba = -999.0;
                    if (W.twin)
                    {
                        Cfg T = run(tr, Ly, W.twin, thd, 1.5);
                        m_so = df(g_of(D.so, 2), g_of(T.so, 2)), m_sh = df(g_of(D.sh, 1), g_of(T.sh, 1)), m_ba = df(g_of(D.ba, 1), g_of(T.ba, 1));
                    }
                    char b[512];
                    std::snprintf(b, sizeof(b), "%d,%s,%s,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", tr, Ly.name, W.name, thd,
                                  df(g_of(R.so, 0) + g_of(R.so, 1), 0.0), g_of(R.sh, 0), g_of(D.ba, 0), g_of(D.so, 2), g_of(D.sh, 1), g_of(D.ba, 1), m_so, m_sh,
                                  m_ba, df(g_of(R.so, 2), g_of(D.so, 2)), df(g_of(R.sh, 1), g_of(D.sh, 1)), df(g_of(R.ba, 1), g_of(D.ba, 1)));
                    rowsB.push_back(b);

                    // The wall behind the traced layer is the wall behind the sheet
                    if (W.w != 10)
                    {
                        REQUIRE((D.so.size() == 3 && D.sh.size() == 2 && D.so[2].rt != 0 && D.sh[1].rt != 0));
                        CHECK(std::abs(a_of(D.so, 1) - a_of(D.sh, 0)) < 1.0e-6); // the direction in the wall
                        CHECK(std::abs(a_of(D.so, 2) - a_of(D.sh, 1)) < 1.0e-6); // the direction behind it
                        CHECK(std::abs(g_of(D.so, 2) - g_of(D.sh, 1)) < 1.0e-4); // the wall | air event
                        if (W.w != 11) // rigid: a leak at every angle, the segment is the incidence line
                        {
                            CHECK(std::abs(a_of(D.so, 1) - thd) < 1.0e-6);
                            CHECK(std::abs(s_of(D.so, 1) - thd) < 1.0e-6);
                            CHECK(std::abs(s_of(D.so, 2) - s_of(D.sh, 1)) < 1.0e-6);
                        }
                        if (W.twin) // held through the mass-law wall, and the mass term at the incidence cosine
                        {
                            CHECK(std::abs(a_of(D.so, 2) - thd) < 1.0e-6);
                            CHECK(std::abs(m_so - m_sh) < 5.0e-3);
                            CHECK(std::abs(m_so - m_ba) < 5.0e-3);
                        }
                    }

                    if (D.so.size() == 3 && D.sh.size() == 2 && D.so[2].rt != 0 && D.sh[1].rt != 0 && D.ba.size() == 2)
                    {
                        w_in = std::max(w_in, std::abs(a_of(D.so, 1) - a_of(D.sh, 0))), w_out = std::max(w_out, std::abs(a_of(D.so, 2) - a_of(D.sh, 1)));
                        w_seg = std::max(w_seg, std::abs(s_of(D.so, 2) - s_of(D.sh, 1))), w_exit = std::max(w_exit, std::abs(g_of(D.so, 2) - g_of(D.sh, 1)));
                        w_into = std::max(w_into, std::abs(g_of(R.so, 0) + g_of(R.so, 1) - g_of(R.sh, 0)));
                        w_ser = std::max(w_ser, std::abs((g_of(R.so, 2) - g_of(D.so, 2)) - (g_of(R.ba, 1) - g_of(D.ba, 1))));
                        w_bare = std::max(w_bare, std::abs(g_of(D.so, 2) - g_of(D.ba, 1)));
                        if (W.twin)
                            w_mass = std::max(w_mass, std::abs(m_so - m_sh));
                    }
                }
                char b[512];
                std::snprintf(b, sizeof(b), "%d,%s,%s,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e\n", tr, Ly.name, W.name, w_in, w_out, w_seg, w_into, w_exit, w_bare, w_mass, w_ser);
                rowsS.push_back(b);
            }

    // ------------------------------------------------------------------------------------------
    // B. Levels
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed-face hand-on B, levels on the rows of A, dB. into: the events up to the wall, series summed;\n");
    QD_DIAG("# exit: the wall | air event, series declined; mass: that event over the same wall with its mass argument taken\n");
    QD_DIAG("# to nothing; ser: that event with the series summed over declined. -999: no such event or no such column\n");
    QD_DIAG("tracer,layer,wall,theta,so_into,sh_into,ba_into,so_exit,sh_exit,ba_exit,so_mass,sh_mass,ba_mass,so_ser,sh_ser,ba_ser\n");
    for (const std::string &s : rowsB)
        QD_DIAG("%s", s.c_str());

    // ------------------------------------------------------------------------------------------
    // C. The wall | air event, with what it is handed changed one thing at a time
    // ------------------------------------------------------------------------------------------
    // now    the event as the chain runs it
    // inc    handed the incidence direction; with tracer 5 also met on a segment along it
    // dir    handed the incidence direction on the segment as it stands
    // none   handed no direction
    // clean  handed a clean previous-medium word
    QD_DIAG("\n# diag: closed-face hand-on C, the wall | air event behind the traced porous layer, run again with what it is\n");
    QD_DIAG("# handed changed. in: the direction handed in; seg: the segment it is met on; out: the direction written, deg;\n");
    QD_DIAG("# gain in dB; mass: over the twin wall run the same way; vbs, tr: bits 1 and 5 of the resolved type; w: the\n");
    QD_DIAG("# words handed in, prev.cur.buf\n");
    QD_DIAG("tracer,wall,theta,var,in,seg,out,gain_dB,mass_dB,rt,vbs,tr,w\n");
    for (int tr : {4, 5})
        for (const WallT &W : {walls[0], walls[1]})
            for (double thd : {10.0, 45.0, 85.0})
            {
                const double th = thd * QPI / 180.0;
                const arma::mat inc{{std::cos(th), std::sin(th), 0.0}};
                arma::uvec mi, mt;
                arma::mat mesh = build({0.0, dS, dS + gap, dS + dW + gap}, {1, 1, W.w, W.w}, {false, true, false, true}, mi);
                arma::mat mesh_t = build({0.0, dS, dS + gap, dS + dW + gap}, {1, 1, W.twin ? W.twin : W.w, W.twin ? W.twin : W.w}, {false, true, false, true}, mt);
                arma::mat o, d;
                diag_ray_at(thd, o, d);
                std::vector<Ev> so = chain(mtl, mesh, mi, tr, 2, o, d, nullptr, 1.5), st = chain(mtl, mesh_t, mt, tr, 2, o, d, nullptr, 1.5);
                REQUIRE((so.size() == 2 && st.size() == 2));
                for (int var = 0; var < 5; ++var)
                {
                    auto third = [&](const std::vector<Ev> &v, const arma::mat &msh, const arma::uvec &mx) -> Ev
                    {
                        Ev P = v[1];
                        arma::mat s = unit(P.destN - P.origN);
                        if (var == 1 || var == 2)
                            P.path_dirN = inc;
                        if (var == 1 && tr == 5)
                            s = inc;
                        if (var == 3)
                            P.path_dirN.zeros();
                        if (var == 4)
                            P.prev(0) = (short)0;
                        arma::mat oo = P.origN, dd = P.origN + (1.2 / std::abs(s(0, 0))) * s;
                        Ev E = traced(mtl, msh, mx, tr, oo, dd, &P, 1.5);
                        E.prev = P.prev, E.cur = P.cur, E.buf = P.buf, E.accg = arma::vec{ang(P.path_dirN)}; // what was handed in, for the row
                        return E;
                    };
                    Ev E = third(so, mesh, mi), T = third(st, mesh_t, mt);
                    const bool ok = E.nh != 0 && E.rt != 0;
                    QD_DIAG("%d,%s,%.0f,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%04x.%04x.%04x\n", tr, W.name, thd,
                            var == 0 ? "now  " : (var == 1 ? "inc  " : (var == 2 ? "dir  " : (var == 3 ? "none " : "clean"))), E.accg(0), ang(E.d - E.o),
                            ok ? ang(E.path_dirN) : -1.0, ok ? dB(cpx(E.xpr, 0, 0)) : -999.0,
                            (ok && W.twin && T.rt != 0) ? dB(cpx(E.xpr, 0, 0)) - dB(cpx(T.xpr, 0, 0)) : -999.0, (int)E.rt, (int)rt_bit(E.rt, 1), (int)rt_bit(E.rt, 5),
                            (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0), (unsigned)(unsigned short)E.buf(0));
                    INFO("C, tracer " << tr << ", " << W.name << ", theta " << thd << ", variant " << var);
                    CHECK(E.nh != 0);
                }
            }

    // ------------------------------------------------------------------------------------------
    // D. A host entered from a denser solid, in front of the leaking wall
    // ------------------------------------------------------------------------------------------
    // Launched in the lossless eps 5.24 solid with seeded state words, into the host with a
    // transverse invariant p, then host | wall and wall | air. Below one the wave has a
    // direction in air, asin(sqrt(p)); above one it has none.
    QD_DIAG("\n# diag: closed-face hand-on D, a host 100 mm entered from the lossless eps 5.24 solid the ray is launched in,\n");
    QD_DIAG("# invariant p, then the rigid wall 200 mm with tf 0.3, with and without m. inH: the direction in the host;\n");
    QD_DIAG("# air: asin(sqrt(p)), the direction that invariant has in air; inW: written into the wall; segW: the segment\n");
    QD_DIAG("# in the wall; out: written behind the wall; g2, g3: host | wall and wall | air, dB\n");
    QD_DIAG("host,wall,p,tracer,inH,air,inW,segW,out,g2_dB,g3_dB,ot2,rt2,ot3,rt3,w2\n");
    for (int host : {5, 3})
        for (const WallT &W : {walls[0], walls[1]})
            for (double p : {0.50, 0.90, 0.99, 1.01, 1.04, 1.10, 1.45})
                for (int tr : {4, 5})
                {
                    const double L = 0.10, th1 = std::asin(std::sqrt(p / 5.24));
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, gap, L, L + gap, L + dW + gap}, {12, host, host, W.w, W.w}, {true, false, true, false, true}, mi);
                    arma::mat orig{{-0.05, 0.37, 0.0}}, dest{{-0.05 + 2.0 * std::cos(th1), 0.37 + 2.0 * std::sin(th1), 0.0}};
                    Ev seed;
                    seed.prev = arma::Col<short>{(short)0}, seed.cur = arma::Col<short>{(short)12}, seed.buf = arma::Col<short>{(short)0};
                    seed.path_dirN = unit(dest - orig), seed.accg = arma::vec{0.0}, seed.accr = arma::vec{0.0};
                    std::vector<Ev> v = chain(mtl, mesh, mi, tr, 3, orig, dest, &seed, 1.5);
                    INFO("D, host " << host << ", " << W.name << ", p " << p << ", tracer " << tr);
                    REQUIRE(v.size() >= 2);
                    const bool three = v.size() == 3;
                    if (p > 1.0) // no direction in air: the direction the wave has
                        CHECK(std::abs(a_of(v, 1) - a_of(v, 0)) < 1.0e-9);
                    else
                    {
                        const double air = std::asin(std::sqrt(p)) * 180.0 / QPI;
                        CHECK(std::abs(a_of(v, 1) - air) < 1.0e-6);
                        if (tr == 5) // the segment follows, and the mass-law wall holds it
                        {
                            CHECK(std::abs(s_of(v, 1) - air) < 1.0e-6);
                            if (W.twin)
                                CHECK(std::abs(a_of(v, 2) - air) < 1.0e-6);
                        }
                    }
                    QD_DIAG("%s,%s,%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,", host == 5 ? "1.50-0.40j" : "1.50", W.name, p, tr, a_of(v, 0),
                            (p < 1.0) ? std::asin(std::sqrt(p)) * 180.0 / QPI : -1.0, a_of(v, 1), s_of(v, 1), a_of(v, 2), g_of(v, 1), g_of(v, 2),
                            (v.size() > 1) ? (int)v[1].ot : -1, (v.size() > 1) ? (int)v[1].rt : -1, three ? (int)v[2].ot : -1, three ? (int)v[2].rt : -1);
                    if (v.size() > 1 && v[1].nh != 0)
                        QD_DIAG("%04x.%04x.%04x\n", (unsigned)(unsigned short)v[1].prev(0), (unsigned)(unsigned short)v[1].cur(0), (unsigned)(unsigned short)v[1].buf(0));
                    else
                        QD_DIAG("-\n");
                }

    // ------------------------------------------------------------------------------------------
    // E. The ray tube at the closed face
    // ------------------------------------------------------------------------------------------
    // Entry and layer | wall with a tube, type 5, the geometry handed to ray_state_update as
    // ray_progress hands it. Each vertex ray against the direction it was launched in.
    QD_DIAG("\n# diag: closed-face hand-on E, the ray tube, refracting tracer, layer 50 mm on the rigid wall with tf 0.3 and m 30.\n");
    QD_DIAG("# Two vertex rays at theta -+ 0.6 deg in the plane, one tilted out of it. Per ray: the angle it was launched at, in\n");
    QD_DIAG("# the layer, and in the wall, deg against the normal; sp: the spine\n");
    QD_DIAG("layer,theta,sp_air,sp_layer,sp_wall,v1_air,v1_layer,v1_wall,v2_air,v2_layer,v2_wall,v3_air,v3_layer,v3_wall,ot2,rt2\n");
    auto ang3 = [](double x, double y, double z)
    { return std::atan2(std::sqrt(y * y + z * z), std::abs(x)) * 180.0 / QPI; };
    for (const LayerT &Ly : layers)
        for (double thd : {10.0, 45.0, 75.0})
        {
            arma::uvec mi;
            arma::mat mesh = build({0.0, dS, dS + gap, dS + dW + gap}, {Ly.solid, Ly.solid, 7, 7}, {false, true, false, true}, mi);
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
            const arma::mat launched = tridir;

            struct Tb
            {
                arma::mat origN, destN, dirN, tvN, tdN, xpr;
                arma::Col<short> prev, cur, buf;
                arma::vec accg, accr;
                uint8_t ot = 0, rt = 0;
            };
            auto step = [&](const arma::mat &oo, const arma::mat &dd, const arma::mat &tv, const arma::mat &td, const Tb *in) -> Tb
            {
                arma::mat fbs, sbs, fbsN, sbsN, nv;
                arma::u32_vec nh, fi, si;
                quadriga_lib::ray_triangle_intersect(&oo, &dd, &mesh, &fbs, &sbs, &nh, &fi, &si);
                Tb E;
                arma::vec gain, fang;
                std::vector<uint8_t> ot, rt;
                quadriga_lib::ray_mesh_interact(5, f_hz, &oo, &dd, &mesh, &mi, &mtl, &fi, &si, &tv, &td, &E.origN, &E.destN, &fbsN, &sbsN, &gain, &E.xpr,
                                                &E.tvN, &E.tdN, &fang, (arma::vec *)nullptr, (arma::vec *)nullptr, &nv, &ot, &E.dirN, false,
                                                (arma::u32_vec *)nullptr);
                arma::Col<short> mf(1), ms(1);
                mf(0) = fi(0) ? (short)mi(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mi(si(0) - 1) : (short)0;
                quadriga_lib::ray_state_update(5, f_hz, &oo, &dd, &fbsN, &sbsN, &nh, &fang, &nv, &ot, &mtl, &mf, &ms, in ? &in->prev : nullptr,
                                               in ? &in->cur : nullptr, in ? &in->buf : nullptr, in ? &in->dirN : nullptr, in ? &in->accg : nullptr,
                                               in ? &in->accr : nullptr, &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.dirN, &E.accg, &E.accr, &rt,
                                               (arma::u32_vec *)nullptr, 1.5, &E.origN, &E.destN, &td, &E.tdN);
                E.ot = ot[0], E.rt = rt[0];
                return E;
            };
            Tb A = step(o, d, trivec, tridir, nullptr);
            arma::mat s1 = unit(A.destN - A.origN), o2 = A.origN, d2 = A.origN + (1.2 / std::abs(s1(0, 0))) * s1;
            Tb B = step(o2, d2, A.tvN, A.tdN, &A);
            arma::mat s2 = unit(B.destN - B.origN);
            QD_DIAG("%s,%.0f,%.4f,%.4f,%.4f", Ly.name, thd, thd, ang3(s1(0, 0), s1(0, 1), s1(0, 2)), ang3(s2(0, 0), s2(0, 1), s2(0, 2)));
            for (arma::uword v = 0; v < 3; ++v)
                QD_DIAG(",%.4f,%.4f,%.4f", ang3(launched(0, 3 * v), launched(0, 3 * v + 1), launched(0, 3 * v + 2)),
                        ang3(A.tdN(0, 3 * v), A.tdN(0, 3 * v + 1), A.tdN(0, 3 * v + 2)), ang3(B.tdN(0, 3 * v), B.tdN(0, 3 * v + 1), B.tdN(0, 3 * v + 2)));
            QD_DIAG(",%d,%d\n", (int)B.ot, (int)B.rt);
            INFO("E, " << Ly.name << ", theta " << thd);
            CHECK(B.rt != 0);
            CHECK(std::abs(ang3(s2(0, 0), s2(0, 1), s2(0, 2)) - thd) < 1.0e-6);
            for (arma::uword v = 0; v < 3; ++v) // each vertex ray continues in the direction it was launched in
                CHECK(std::abs(ang3(B.tdN(0, 3 * v), B.tdN(0, 3 * v + 1), B.tdN(0, 3 * v + 2)) -
                               ang3(launched(0, 3 * v), launched(0, 3 * v + 1), launched(0, 3 * v + 2))) < 1.0e-6);
        }

    // ------------------------------------------------------------------------------------------
    // S. The traced solid against the sheet, worst over the angles
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed-face hand-on S, the traced solid against the sheet, worst over 10 to 85 deg. in: direction in the\n");
    QD_DIAG("# wall; out: direction behind it; seg: the segment behind it, deg. into: the events up to the wall, series\n");
    QD_DIAG("# summed; exit: the wall | air event; bare: that event against the bare wall's; mass: the wall's mass term, dB;\n");
    QD_DIAG("# ser: the wall's series behind the solid against the bare wall's, dB\n");
    QD_DIAG("tracer,layer,wall,in,out,seg,into,exit,bare,mass,ser\n");
    for (const std::string &q : rowsS)
        QD_DIAG("%s", q.c_str());
}

TEST_CASE("Bonded mass - the plate at any depth", "[acoustic]")
{
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
    // Plate depth. A leg learns what it is bonded to from the previous-medium word and the buffer
    // word, which name one body behind the ray, or two where that body is a sheet inside the
    // current medium. Every mass-law body has the slope of the mass law (M10: m > 0 is a switch;
    // the bodies named 30 keep m 30 and pay as m 20, which their rows guard), so bonded bodies are,
    // at every frequency, one thicker body of one of them: the thickness code carries that
    // equivalent thickness and the plate is exact at any depth, up to the entry face the state
    // re-forms one level deep. Recorded: a deferred medium, which displaces the code, and an
    // equivalent thickness above the 100 m the code reaches. A mass-law sheet with its mass-law host's permittivity cuts the host leg like
    // any other sheet. Event by event through ray_triangle_intersect, ray_mesh_interact and
    // ray_state_update, scalar, both tracers. The mass an event paid is its gain against the same event on a palette whose mass
    // arguments are taken to nothing (every fRef times 1e12), which leaves every face, tf and state word in
    // place. The plate passes the incoherent series of its outer faces and its term (md/acoustic.md 10 N1):
    // on the twin a, the smaller tf of the two outer faces, with its mass a tau / (a + tau - a tau), tau the
    // plate diag_bonded_TL over the bodies in the order they are crossed. The reference is what the rule
    // pays above a; per event the increment the plate takes there is recorded.
    //
    // A. Every stack, both directions, far above and near the knee, to 75 deg
    // S. The same, worst over tracers and angles
    // B. A bounce: nothing is reflected back into a held mass-law body (N2), so a reflection inside the stack carries nothing
    // C. A band: each frequency alone, and whether its state words are the base frequency's
    const double gap = 1.0e-5;
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dBz = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto unit = [](arma::mat d) -> arma::mat
    {
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        return d;
    };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f_hz, const arma::mat &orig,
                      const arma::mat &dest, const Ev *in) -> Ev
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
                                       (arma::u32_vec *)nullptr, 1.5, &E.origN, &E.destN);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    using ub::face;

    // A stack along x as tokens: O / C open and close a solid, S is a sheet, N / n open and close
    // an object nested in the solid the ray is in, g advances by d. Faces with no advance between
    // them are co-located, a gap apart, and are one event.
    struct Tok
    {
        char k;
        int mat;
        double d;
    };
    auto O = [](int m) { return Tok{'O', m, 0.0}; };
    auto C = [](int m) { return Tok{'C', m, 0.0}; };
    auto S = [](int m) { return Tok{'S', m, 0.0}; };
    auto N = [](int m) { return Tok{'N', m, 0.0}; };
    auto n = [](int m) { return Tok{'n', m, 0.0}; };
    auto G = [](double d) { return Tok{'g', 0, d}; };
    auto reversed = [](std::vector<Tok> t)
    {
        std::reverse(t.begin(), t.end());
        for (Tok &q : t)
            q.k = (q.k == 'O') ? 'C' : ((q.k == 'C') ? 'O' : ((q.k == 'N') ? 'n' : ((q.k == 'n') ? 'N' : q.k)));
        return t;
    };
    auto build = [&](const std::vector<Tok> &t, arma::uvec &mtl_ind, size_t &n_ev) -> arma::mat
    {
        arma::mat mesh;
        std::vector<int> mats;
        double x = 0.0;
        bool last_face = false;
        n_ev = 0;
        for (const Tok &q : t)
        {
            if (q.k == 'g')
            {
                x += q.d, last_face = false;
                continue;
            }
            if (last_face)
                x += gap;
            else
                ++n_ev;
            arma::mat f = face(x, q.k == 'C' || q.k == 'n');
            mesh = mats.empty() ? f : arma::join_cols(mesh, f);
            mats.push_back(q.mat), last_face = true;
        }
        mtl_ind.set_size(2 * mats.size());
        for (size_t k = 0; k < mats.size(); ++k)
            mtl_ind(2 * k) = (arma::uword)mats[k], mtl_ind(2 * k + 1) = (arma::uword)mats[k];
        return mesh;
    };

    // The palette. Solids and sheets of the acoustic mapping: rigid, lossless, a tf leak and the
    // mass column. X0 = ((f / fRef) d)^(m / 20), so with fRef 0.001 at 1 GHz a 100 mm body of
    // m 20 has X0 100. heavy: that; light: fRef 0.05, the same body at X0 2; twin: every fRef
    // times 1e12, so no body has a mass term and the bodies keep their ratios.
    const double fR = 0.001;
    const int A20 = 1, B20 = 2, C20 = 3, A30 = 4, B30 = 5, C30 = 6, PL = 7, BR20 = 8, BR30 = 9, S20 = 10, S30 = 11, L20 = 12, L30 = 13,
              HS = 14, D20 = 15, T20 = 16, T30 = 17, E20 = 18, F20 = 19, G20 = 20, HS30 = 21;
    const std::vector<LumpMtl> base = {
        {2.401e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 20.0, fR},    // 1  A20
        {2.1609e-8, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 20.0, fR},   // 2  B20
        {6.0e-9, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 20.0, fR},      // 3  C20
        {2.401e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 30.0, fR},    // 4  A30
        {2.1609e-8, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 30.0, fR},   // 5  B30
        {6.0e-9, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 30.0, fR},      // 6  C30
        {1.2e-8, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 20.0, fR},      // 7  PL    plaster, solid
        {4.0e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 20.0, fR},      // 8  BR20  brick, solid
        {4.0e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 30.0, fR},      // 9  BR30
        {1.0e-8, 0.0, 0.012, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 20.0, fR},    // 10 S20   sheet 12 mm
        {1.5e-9, 0.0, 0.012, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 30.0, fR},    // 11 S30   sheet 12 mm
        {5.0e-9, 0.0, 0.015, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 20.0, fR},    // 12 L20   lining 15 mm
        {5.0e-9, 0.0, 0.015, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 30.0, fR},    // 13 L30   lining 15 mm
        {2.1609e-8, 0.0, 0.008, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 14 HS    sheet 8 mm with B20's permittivity: merged
        {3.0e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 20.0, fR},      // 15 D20   an object nested in a wall
        {7.0e-9, 0.0, 0.008, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 20.0, fR},    // 16 T20   second sheet of a stack, 8 mm
        {7.0e-9, 0.0, 0.008, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 30.0, fR},    // 17 T30
        {8.0e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 20.0, 0.4 * fR}, // 18 E20   2.5 times the surface mass per thickness
        {9.0e-9, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 20.0, 0.01 * fR}, // 19 F20   a hundred times
        {1.1e-8, 0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 20.0, 100.0 * fR}, // 20 G20   a hundredth
        {2.1609e-8, 0.0, 0.008, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0, 30.0, fR}, // 21 HS30  sheet 8 mm with B30's permittivity: merged
    };
    auto pal_of = [&](double fRef_scale, bool twin) -> Pal
    {
        std::vector<LumpMtl> v = base;
        for (LumpMtl &M : v)
            M.fRef = twin ? M.fRef * 1.0e12 : M.fRef * fRef_scale; // the twin keeps the ratios between the bodies
        return side0(lump_palette(v));
    };
    const Pal p_twin = pal_of(1.0, true);
    struct Regime
    {
        const char *name;
        double scale;
        Pal pal;
    };
    const std::vector<Regime> regimes = {{"heavy", 1.0, pal_of(1.0, false)}, {"light", 50.0, pal_of(50.0, false)}};

    // The bodies of a stack in the order they are crossed, each with the event that pays it
    struct Body
    {
        DiagMass M;
        size_t ev;
        bool sheet;
    };
    auto bodies = [&](const std::vector<Tok> &t, double scale) -> std::vector<Body>
    {
        std::vector<Body> out;
        auto mass = [&](int mat, double d)
        { return DiagMass{base[(size_t)mat - 1].m, base[(size_t)mat - 1].fRef * scale, d}; };
        int cur = 0, pend = 0;
        double leg = 0.0;
        size_t ev = 0;
        bool last_face = false, any = false;
        for (const Tok &q : t)
        {
            if (q.k == 'g')
            {
                if (cur)
                    leg += q.d;
                last_face = false;
                continue;
            }
            if (!last_face && any)
                ++ev;
            last_face = true, any = true;
            if (q.k == 'O')
            {
                if (cur)
                    pend = q.mat; // overlapping: deferred until the solid the ray is in is left
                else
                    cur = q.mat, leg = 0.0;
            }
            else if (q.k == 'C')
            {
                if (q.mat != cur)
                    continue;
                if (leg > 0.0)
                    out.push_back({mass(cur, leg), ev, false});
                cur = pend, pend = 0, leg = 0.0;
            }
            else if (q.k == 'S')
            {
                if (cur && leg > 0.0)
                    out.push_back({mass(cur, leg), ev, false}), leg = 0.0;
                out.push_back({mass(q.mat, base[(size_t)q.mat - 1].thk), ev, true});
            }
        }
        return out;
    };

    // The events of one path: types[k] is the interaction type of event k
    auto chain = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, const std::vector<int> &types, double thd, double f_hz) -> std::vector<Ev>
    {
        std::vector<Ev> out;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        for (size_t q = 0; q < types.size(); ++q)
        {
            Ev E = traced(pal, mesh, mi, types[q], f_hz, o, d, q ? &out[q - 1] : nullptr);
            out.push_back(E);
            if (E.nh == 0 || E.rt == 0)
                break;
            arma::mat s = unit(E.destN - E.origN);
            o = E.origN, d = E.origN + (1.2 / std::abs(s(0, 0))) * s;
        }
        return out;
    };
    auto words = [](const Ev &E)
    {
        char b[32];
        std::snprintf(b, sizeof(b), "%04x.%04x.%04x", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                      (unsigned)(unsigned short)E.buf(0));
        return std::string(b);
    };
    // Mass paid per event, dB: the twin over the run
    auto paid_of = [&](const std::vector<Ev> &r, const std::vector<Ev> &t, std::vector<double> &p) -> bool
    {
        p.assign(r.size(), 0.0);
        if (r.size() != t.size())
            return false;
        for (size_t q = 0; q < r.size(); ++q)
        {
            if (r[q].nh == 0 || r[q].rt == 0 || t[q].nh == 0 || t[q].rt == 0)
                return false;
            if (r[q].prev(0) != t[q].prev(0) || r[q].cur(0) != t[q].cur(0) || r[q].buf(0) != t[q].buf(0))
                return false;
            p[q] = dBz(cpx(t[q].xpr, 0, 0)) - dBz(cpx(r[q].xpr, 0, 0));
        }
        return true;
    };
    auto X_of = [](const DiagMass &B, double f_hz)
    {
        double arg = (f_hz * 1e-9 / B.fRef) * B.thk;
        return (B.m > 0.0 && arg > 0.0) ? arg : 0.0;
    };
    (void)X_of;

    struct Stack
    {
        const char *name;
        std::vector<Tok> t;
        int exact; // bit 0: the plate as named, bit 1: from the other side; bits 2, 3: recorded within the depth of the state; clear: recorded
    };
    // The plate's entry face is re-formed from the state, which is one level deep (md/acoustic.md 7): on
    // the last leg the previous-medium word names the body before it, whose face stands in for the stack's
    // first. Where it absorbs differently (A20 | B20 | C20 as named: 0.2 for 0.1) the level is off by the
    // ratio, at most 3.01 dB in this palette (bits 2 and 3, bounded); a deferred medium as before.
    const std::vector<Stack> stacks = {
        {"A20 100 | B20 100 | C20 100", {O(A20), G(0.1), C(A20), O(B20), G(0.1), C(B20), O(C20), G(0.1), C(C20)}, 2 | 4},
        {"A20 30 | B20 100 | C20 200", {O(A20), G(0.03), C(A20), O(B20), G(0.1), C(B20), O(C20), G(0.2), C(C20)}, 2 | 4},
        {"A20 100 | B20 13 | C20 100", {O(A20), G(0.1), C(A20), O(B20), G(0.013), C(B20), O(C20), G(0.1), C(C20)}, 2 | 4},
        {"A30 60 | B30 100 | C30 30", {O(A30), G(0.06), C(A30), O(B30), G(0.1), C(B30), O(C30), G(0.03), C(C30)}, 2 | 4},
        {"A20 100 | B20 100 | C30 50", {O(A20), G(0.1), C(A20), O(B20), G(0.1), C(B20), O(C30), G(0.05), C(C30)}, 2 | 4},
        {"A20 100 | B30 50 | C20 100", {O(A20), G(0.1), C(A20), O(B30), G(0.05), C(B30), O(C20), G(0.1), C(C20)}, 2 | 4},
        {"A20 50 | B20 50 | C20 50 | A20 50", {O(A20), G(0.05), C(A20), O(B20), G(0.05), C(B20), O(C20), G(0.05), C(C20), O(A20), G(0.05), C(A20)}, 3},
        {"plaster 15 | brick20 115 | plaster 15", {O(PL), G(0.015), C(PL), O(BR20), G(0.115), C(BR20), O(PL), G(0.015), C(PL)}, 4 | 8},
        {"plaster 15 | brick30 115 | plaster 15", {O(PL), G(0.015), C(PL), O(BR30), G(0.115), C(BR30), O(PL), G(0.015), C(PL)}, 4 | 8},
        {"lining L20 > A20 100 | B20 100", {S(L20), O(A20), G(0.1), C(A20), O(B20), G(0.1), C(B20)}, 2 | 4},
        {"lining L20 > B20 60 | S20 | B20 40", {S(L20), O(B20), G(0.06), S(S20), G(0.04), C(B20)}, 2 | 4},
        {"lining L20 > B20 60 | S30 | B20 40", {S(L20), O(B20), G(0.06), S(S30), G(0.04), C(B20)}, 2 | 4},
        {"lining L30 > B20 60 | S20 | B20 40", {S(L30), O(B20), G(0.06), S(S20), G(0.04), C(B20)}, 2 | 4},
        {"A20 100 | B20 60 | S20 | B20 40", {O(A20), G(0.1), C(A20), O(B20), G(0.06), S(S20), G(0.04), C(B20)}, 2 | 4},
        {"B20 60 | S20 T20 | B20 40", {O(B20), G(0.06), S(S20), S(T20), G(0.04), C(B20)}, 3},
        {"B20 60 | S20 T30 | B20 40", {O(B20), G(0.06), S(S20), S(T30), G(0.04), C(B20)}, 3},
        {"B20 60 | S30 T20 | B20 40", {O(B20), G(0.06), S(S30), S(T20), G(0.04), C(B20)}, 3},
        {"deferred: A20 100 | C20 100 [D20 100] C20 100", {O(A20), G(0.1), C(A20), O(C20), G(0.1), N(D20), G(0.1), n(D20), G(0.1), C(C20)}, 2},
        {"deferred: lining L20 > A20 100 [B20 100 A20 100] B20 100", {S(L20), O(A20), G(0.1), O(B20), G(0.1), C(A20), G(0.1), C(B20)}, 2},
        {"merged: B20 60 | HS 8 | B20 40", {O(B20), G(0.06), S(HS), G(0.04), C(B20)}, 3},
        {"merged: A20 100 | B20 60 | HS 8 | B20 40", {O(A20), G(0.1), C(A20), O(B20), G(0.06), S(HS), G(0.04), C(B20)}, 2 | 4},
        {"merged: B30 60 | HS30 8 | B30 40", {O(B30), G(0.06), S(HS30), G(0.04), C(B30)}, 3},
        {"fRef: A20 100 | E20 60 | C20 100", {O(A20), G(0.1), C(A20), O(E20), G(0.06), C(E20), O(C20), G(0.1), C(C20)}, 3},
        {"fRef: F20 100 | G20 10 | C20 100", {O(F20), G(0.1), C(F20), O(G20), G(0.01), C(G20), O(C20), G(0.1), C(C20)}, 2},
    };
    const std::vector<double> angles = {0.0, 30.0, 60.0, 75.0};
    // what the plate rule pays above a, dB, a the smaller tf of the stack's two outer faces
    auto a_of = [&](const std::vector<Tok> &t) { return std::min(base[(size_t)t.front().mat - 1].tf, base[(size_t)t.back().mat - 1].tf); };
    auto rule_of = [](double a, double plate)
    {
        const double tau = std::pow(10.0, -0.1 * plate);
        return 10.0 * std::log10(a) - 10.0 * std::log10(a * tau / (a + tau - a * tau));
    };
    const size_t NE = 6;

    // ------------------------------------------------------------------------------------------
    // A. Every stack
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: plate depth A. Bonded mass-law stacks through the intersector, scalar, transmit events only. dir: f as\n");
    QD_DIAG("# named, r from the other side. plate: diag_bonded_TL of every body; sum: the bodies' own terms added in dB;\n");
    QD_DIAG("# rule: what the plate rule pays above a; paid: what the events paid; over: paid - rule; rest: the chain without its mass. p0..: paid per event;\n# s0..: what the plate takes at that\n");
    QD_DIAG("# event, its increment over the bodies crossed before; w0..: prev.cur.buf written by that event. -: no event\n");
    QD_DIAG("stack,dir,regime,tracer,theta,plate,sum,rule,paid,over,rest");
    for (size_t q = 0; q < NE; ++q)
        QD_DIAG(",p%d", (int)q);
    for (size_t q = 0; q < NE; ++q)
        QD_DIAG(",s%d", (int)q);
    for (size_t q = 0; q < NE; ++q)
        QD_DIAG(",w%d", (int)q);
    QD_DIAG("\n");
    std::vector<std::string> rowsS;
    for (const Stack &St : stacks)
        for (int rev = 0; rev < 2; ++rev)
        {
            const std::vector<Tok> tk = rev ? reversed(St.t) : St.t;
            arma::uvec mi;
            size_t n_ev = 0;
            arma::mat mesh = build(tk, mi, n_ev);
            for (const Regime &Rg : regimes)
            {
                const std::vector<Body> bd = bodies(tk, Rg.scale);
                double w_max = -1.0e9, w_min = 1.0e9, at0 = 0.0, w_sum = 0.0;
                for (int tr : {4, 5})
                    for (double thd : angles)
                    {
                        const double ct = std::cos(thd * QPI / 180.0);
                        std::vector<Ev> r = chain(Rg.pal, mesh, mi, std::vector<int>(n_ev, tr), thd, FRQ);
                        std::vector<Ev> t = chain(p_twin, mesh, mi, std::vector<int>(n_ev, tr), thd, FRQ);
                        std::vector<double> p;
                        const bool ok = paid_of(r, t, p) && r.size() == n_ev && (r.back().cur(0) & (short)0x3FFF) == 0;
                        INFO("A, " << St.name << ", dir " << rev << ", " << Rg.name << ", tracer " << tr << ", theta " << thd);
                        CHECK(ok);
                        std::vector<DiagMass> all;
                        std::vector<double> should(NE, 0.0);
                        double plate = 0.0, sum = 0.0, paid = 0.0;
                        for (const Body &B : bd)
                        {
                            all.push_back(B.M);
                            const double pl = diag_bonded_TL(all, FRQ, ct);
                            if (B.ev < NE)
                                should[B.ev] += pl - plate;
                            plate = pl, sum += diag_bonded_TL({B.M}, FRQ, ct);
                        }
                        for (double v : p)
                            paid += v;
                        double rest = 0.0; // the chain without its mass, dB
                        for (size_t q = 0; ok && q < t.size(); ++q)
                            rest += dBz(cpx(t[q].xpr, 0, 0));
                        const double rule = rule_of(a_of(tk), plate);
                        QD_DIAG("%s,%s,%s,%d,%.0f,%.4f,%.4f,%.4f,%.4f,%+.4f,%.6f", St.name, rev ? "r" : "f", Rg.name, tr, thd, plate, sum, rule, ok ? paid : -999.0,
                                ok ? paid - rule : -999.0, ok ? rest : -999.0);
                        for (size_t q = 0; q < NE; ++q)
                            (ok && q < p.size()) ? QD_DIAG(",%.4f", p[q]) : QD_DIAG(",-");
                        for (size_t q = 0; q < NE; ++q)
                            (q < n_ev) ? QD_DIAG(",%.4f", should[q]) : QD_DIAG(",-");
                        for (size_t q = 0; q < NE; ++q)
                            (q < r.size()) ? QD_DIAG(",%s", words(r[q]).c_str()) : QD_DIAG(",-");
                        QD_DIAG("\n");
                        if (ok && ((St.exact >> rev) & 1)) // the rule, to the code's quantization and the open face at 0 deg
                            CHECK(std::abs(paid - rule) < 0.02);
                        else if (ok && ((St.exact >> (2 + rev)) & 1)) // within the depth of the state
                            CHECK(std::abs(paid - rule) < 3.03);
                        if (ok)
                        {
                            w_max = std::max(w_max, paid - rule), w_min = std::min(w_min, paid - rule);
                            if (thd == 0.0 && tr == 4)
                                at0 = paid - rule, w_sum = sum - plate;
                        }
                    }
                char b[256];
                std::snprintf(b, sizeof(b), "%s,%s,%s,%+.4f,%+.4f,%+.4f,%+.4f\n", St.name, rev ? "r" : "f", Rg.name, at0, w_max, w_min, w_sum);
                rowsS.push_back(b);
            }
        }

    QD_DIAG("\n# diag: plate depth S. over0: paid - rule at normal incidence; max, min: its range over both tracers and 0 to\n");
    QD_DIAG("# 75 deg; sum0: what adding the bodies in dB would be over the plate at normal incidence\n");
    QD_DIAG("stack,dir,regime,over0,max,min,sum0\n");
    for (const std::string &s : rowsS)
        QD_DIAG("%s", s.c_str());

    // ------------------------------------------------------------------------------------------
    // B. A bounce
    // ------------------------------------------------------------------------------------------
    // Forward to event refl, reflected there from inside the stack. A held mass-law body is a lumped
    // plate with no wave inside: nothing is reflected back into it, so the reflected path carries nothing.
    QD_DIAG("\n# diag: plate depth B, a bounce, heavy. refl: the event that reflects, counted from 0, from inside the stack;\n");
    QD_DIAG("# R_refl: the power the reflection carries; p0..: paid per event up to it; w0..: prev.cur.buf written by that event\n");
    QD_DIAG("stack,dir,refl,tracer,theta,plate,R_refl");
    const size_t NB = 8;
    for (size_t q = 0; q < NB; ++q)
        QD_DIAG(",p%d", (int)q);
    for (size_t q = 0; q < NB; ++q)
        QD_DIAG(",w%d", (int)q);
    QD_DIAG("\n");
    struct Bounce
    {
        size_t stack;
        int rev;
        size_t refl;
    };
    const std::vector<Bounce> bounces = {{0, 0, 3}, {0, 0, 2}, {2, 0, 3}, {4, 0, 3}, {4, 1, 3}, {7, 0, 3}, {9, 0, 2}, {9, 1, 2}, {10, 0, 2}, {10, 1, 2}, {13, 0, 3}};
    for (const Bounce &Bn : bounces)
    {
        const Stack &St = stacks[Bn.stack];
        const std::vector<Tok> tk = Bn.rev ? reversed(St.t) : St.t;
        arma::uvec mi;
        size_t n_ev = 0;
        arma::mat mesh = build(tk, mi, n_ev);
        const std::vector<Body> bd = bodies(tk, 1.0);
        for (int tr : {4, 5})
            for (double thd : {0.0, 60.0})
            {
                const double ct = std::cos(thd * QPI / 180.0);
                std::vector<int> types(2 * Bn.refl + 1, tr);
                types[Bn.refl] = 3;
                types.resize(Bn.refl + 1);
                std::vector<Ev> r = chain(regimes[0].pal, mesh, mi, types, thd, FRQ), t = chain(p_twin, mesh, mi, types, thd, FRQ);
                std::vector<double> p;
                const bool ok = r.size() == types.size() && t.size() == types.size();
                INFO("B, " << St.name << ", dir " << Bn.rev << ", refl " << Bn.refl << ", tracer " << tr << ", theta " << thd);
                CHECK(ok);
                std::vector<DiagMass> all;
                for (const Body &B : bd) // a sheet the ray is reflected off is not crossed
                    if (B.ev < Bn.refl || (B.ev == Bn.refl && !B.sheet))
                        all.push_back(B.M);
                const double plate = diag_bonded_TL(all, FRQ, ct);
                const double R_refl = ok ? std::norm(cpx(r.back().xpr, 0, 0)) : -1.0;
                if (ok)
                    CHECK(R_refl == 0.0); // nothing comes back into a held plate
                for (size_t q = 0; ok && q < types.size(); ++q)
                    p.push_back(dBz(cpx(t[q].xpr, 0, 0)) - dBz(cpx(r[q].xpr, 0, 0)));
                QD_DIAG("%s,%s,%d,%d,%.0f,%.4f,%.3e", St.name, Bn.rev ? "r" : "f", (int)Bn.refl, tr, thd, plate, R_refl);
                for (size_t q = 0; q < NB; ++q)
                    (ok && q < p.size() && q < Bn.refl) ? QD_DIAG(",%.4f", p[q]) : QD_DIAG(",-");
                for (size_t q = 0; q < NB; ++q)
                    (q < r.size()) ? QD_DIAG(",%s", words(r[q]).c_str()) : QD_DIAG(",-");
                QD_DIAG("\n");
            }
    }

    // ------------------------------------------------------------------------------------------
    // C. A band
    // ------------------------------------------------------------------------------------------
    // The state words of a run are the base frequency's. Each frequency is run alone here, and
    // its words are compared with those of 1 GHz: whatever differs could not be shared.
    QD_DIAG("\n# diag: plate depth C, a band, heavy at 1 GHz, tracer 4. Each frequency run alone. same: 1 where every state word\n");
    QD_DIAG("# of the chain is the one 1 GHz writes\n");
    QD_DIAG("stack,dir,f_GHz,theta,plate,rule,paid,over,same\n");
    for (size_t is : {(size_t)0, (size_t)1, (size_t)4, (size_t)5, (size_t)8, (size_t)10, (size_t)12, (size_t)14})
        for (int rev = 0; rev < 2; ++rev)
        {
            const Stack &St = stacks[is];
            const std::vector<Tok> tk = rev ? reversed(St.t) : St.t;
            arma::uvec mi;
            size_t n_ev = 0;
            arma::mat mesh = build(tk, mi, n_ev);
            const std::vector<Body> bd = bodies(tk, 1.0);
            for (double thd : {0.0, 60.0})
            {
                const double ct = std::cos(thd * QPI / 180.0);
                std::vector<Ev> r0 = chain(regimes[0].pal, mesh, mi, std::vector<int>(n_ev, 4), thd, FRQ);
                for (double fG : {0.25, 1.0, 4.0})
                {
                    std::vector<Ev> r = chain(regimes[0].pal, mesh, mi, std::vector<int>(n_ev, 4), thd, fG * 1.0e9);
                    std::vector<Ev> t = chain(p_twin, mesh, mi, std::vector<int>(n_ev, 4), thd, fG * 1.0e9);
                    std::vector<double> p;
                    const bool ok = paid_of(r, t, p) && r.size() == n_ev && r0.size() == n_ev;
                    INFO("C, " << St.name << ", dir " << rev << ", f " << fG << ", theta " << thd);
                    CHECK(ok);
                    std::vector<DiagMass> all;
                    for (const Body &B : bd)
                        all.push_back(B.M);
                    const double plate = diag_bonded_TL(all, fG * 1.0e9, ct);
                    double paid = 0.0;
                    bool same = ok;
                    for (size_t q = 0; ok && q < n_ev; ++q)
                        paid += p[q], same = same && r[q].prev(0) == r0[q].prev(0) && r[q].cur(0) == r0[q].cur(0) && r[q].buf(0) == r0[q].buf(0);
                    const double rule = rule_of(a_of(tk), plate);
                    QD_DIAG("%s,%s,%.2f,%.0f,%.4f,%.4f,%.4f,%+.4f,%d\n", St.name, rev ? "r" : "f", fG, thd, plate, rule, ok ? paid : -999.0, ok ? paid - rule : -999.0,
                            (int)same);
                    CHECK(same); // the words are shared by every frequency of a run
                    if (ok && ((St.exact >> rev) & 1))
                        CHECK(std::abs(paid - rule) < 0.02);
                    else if (ok && ((St.exact >> (2 + rev)) & 1))
                        CHECK(std::abs(paid - rule) < 3.03);
                }
            }
        }
}

TEST_CASE("Unentered branches - a sheet without the mass column of its mass-law host's permittivity is merged", "[acoustic]")
{
    ub::OneThread one_thread;
    // The merge of a matched sheet into its host leg is switched off only where host and sheet
    // both carry the mass column. A matched sheet without it is no boundary and no body: the
    // host stays one leg and pays the mass term of its whole thickness, where a sheet of another
    // permittivity cuts it into two bodies that pay in dB. The mass paid is the run over its twin
    // whose fRef takes the mass argument to nothing; the wall is a plate whose faces absorb its tf
    // (0.3), so what it pays is the plate rule above that, dB(a) - dB(a tau / (a + tau - a tau))
    // (md/acoustic.md 10 N1), tau the term of one body of the thickness crossed.
    const double f = FRQ, Lw = 0.100;
    const std::vector<LumpMtl> lp = {
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001}, // 1 rigid wall, tf 0.3, m 30 (a switch: slope 20)
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0e9}, // 2 its twin without a mass argument
        {2.401e-9, 0.0, 0.008},                                          // 3 a sheet of the wall's permittivity, no mass column
        {2.1609e-8, 0.0, 0.008},                                         // 4 a sheet of another permittivity, no mass column
    };
    ub::Pal pal = lump_palette(lp);
    QD_DIAG("\n# diag: a sheet without the mass column inside 100 mm of a rigid mass-law wall, scalar. paid: mass term the chain pays, the\n");
    QD_DIAG("# run over its twin without a mass argument; TL: the plate rule with one body of the thickness crossed. d_alone: the chain against\n");
    QD_DIAG("# the wall without the sheet, dB. words: prev.cur.buf after the entry and after the sheet\n");
    QD_DIAG("tracer,theta,sheet,rt,paid_dB,TL_dB,paid_minus_TL,d_alone_dB,words_in,words_out\n");
    for (int tr : {4, 5})
        for (double thd : {0.0, 10.0, 30.0, 60.0})
        {
            const double cth = std::cos(thd * QPI / 180.0);
            double lvl[3][2], dleg = 0.0;
            ub::Ev Ein, Esh[3];
            for (int s = 0; s < 3; ++s)   // 0 the wall alone, 1 with the matched sheet, 2 with the other sheet
                for (int tw = 0; tw < 2; ++tw)
                {
                    arma::uvec mi;
                    arma::mat mesh = s ? ub::build({0.0, 0.5 * Lw, Lw}, {1 + tw, 2 + s, 1 + tw}, {false, false, true}, mi)
                                       : ub::build({0.0, Lw}, {1 + tw, 1 + tw}, {false, true}, mi);
                    auto c = ub::chain(pal, mesh, mi, s ? std::vector<int>{tr, tr, tr} : std::vector<int>{tr, tr}, f, thd, {1.5});
                    REQUIRE(c.size() == (s ? 3u : 2u));
                    std::complex<double> v = 1.0;
                    for (const auto &E : c)
                        v *= cpx(E.xpr, 0, 0);
                    lvl[s][tw] = ub::dB(v);
                    if (tw == 0)
                    {
                        dleg = Lw - c[0].fbs(0, 0), Ein = c[0];
                        if (s)
                            Esh[s] = c[1];
                    }
                }
            const double z = std::sqrt(2.401e-9), R0 = ((1.0 - z) / (1.0 + z)) * ((1.0 - z) / (1.0 + z));
            const double a = (cth >= 1.0) ? 1.0 - 0.7 * R0 : 0.3; // the wall's face: open exactly at normal incidence
            const double tau = std::pow(10.0, -0.1 * diag_mass_TL(30.0, 0.001, f, dleg, cth));
            const double TL = 10.0 * std::log10(a) - 10.0 * std::log10(a * tau / (a + tau - a * tau));
            for (int s = 0; s < 3; ++s)
            {
                const double paid = lvl[s][1] - lvl[s][0];
                QD_DIAG("%d,%.0f,%s,%d,%.6f,%.6f,%.3e,%.3e,%04x.%04x.%04x,%04x.%04x.%04x\n", tr, thd, s ? (s == 1 ? "matched" : "other") : "none",
                        s ? (int)Esh[s].rt : 0, paid, TL, paid - TL, lvl[s][0] - lvl[0][0], ub::w(Ein.prev(0)), ub::w(Ein.cur(0)), ub::w(Ein.buf(0)),
                        s ? ub::w(Esh[s].prev(0)) : 0u, s ? ub::w(Esh[s].cur(0)) : 0u, s ? ub::w(Esh[s].buf(0)) : 0u);
                INFO("merged sheet, tracer " << tr << " theta " << thd << " stack " << s);
                if (s < 2)
                    CHECK(std::abs(paid - TL) < 1.0e-3); // one body of the whole thickness
                if (s == 1)
                {
                    CHECK(std::abs(lvl[1][0] - lvl[0][0]) < 1.0e-3); // the wall with the sheet is the wall
                    CHECK((Esh[1].prev(0) == Ein.prev(0) && Esh[1].cur(0) == Ein.cur(0) && Esh[1].buf(0) == Ein.buf(0))); // the leg carries on
                }
                if (s == 2)
                    CHECK(paid - TL > 10.0); // another permittivity separates the halves, which then pay in dB
            }
        }
}

TEST_CASE("Unentered branches - a stack of a coherent and a mass-law layer against the laminate meshed as a solid with a lining", "[acoustic]")
{
    ub::OneThread one_thread;
    // A stack is centred over its coherent thickness, and a mass-law layer has none: the coherent
    // layer lies about the mesh plane and the mass-law layer on its far face. That is the laminate
    // whose coherent layer is a traced solid of that thickness with the mass-law layer as a lining
    // on one face, with the stack's plane in the middle of the solid. The traced laminate runs
    // through other code: a glued sheet event and the solid's own series. Paths are taken to the
    // point their polyline ends at; the reflected ones are referenced to the stack's plane.
    const double f = FRQ, gap = 1.0e-5, tC = 0.010, k = 2.0 * QPI * f / C0;
    const std::complex<double> J(0.0, 1.0);
    const std::vector<LumpMtl> lp = {
        {2.50, 0.80, tC},                                                    // 1 the coherent layer as a sheet
        {2.50, 0.80, 0.0},                                                   // 2 the same as a solid
        {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},  // 3 mass-law layer, 26.8 dB
        {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.010},  // 4 the same near its knee, 6.8 dB
    };
    ub::Pal pal = lump_palette(lp);
    QD_DIAG("\n# diag: stack coherent | mass-law against the traced solid with a lining, scalar, undeviated and refracting tracer, every\n");
    QD_DIAG("# series summed. T, R: the stack's port over the traced laminate's, dB and deg. side C: the coherent layer met first;\n");
    QD_DIAG("# side M: the mass-law layer met first, where the traced laminate also has an echo through the lining that a stack does\n");
    QD_DIAG("# not carry: echo_dB is that path over the reflection off the lining, and R is taken against the first path alone\n");
    QD_DIAG("# mass_cos_dB: the mass term at the cosine the wave has in the coherent layer less the term at the incidence cosine, which\n");
    QD_DIAG("# is what a lining on the far face of a traced solid would pay more than the stack if its term read the cosine in the solid\n");
    QD_DIAG("side,mass,tracer,theta,T_stack_dB,T_dB,T_deg,R_stack_dB,R_dB,R_deg,echo_dB,mass_cos_dB\n");
    for (int side = 0; side < 2; ++side)
        for (int iM : {3, 4})
            for (int tr : {4, 5})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    const double c = std::cos(thd * QPI / 180.0);
                    arma::uvec ms, mt;
                    arma::mat mesh_s = side ? ub::build({0.0, gap}, {iM, 1}, {false, true}, ms) : ub::build({0.0, gap}, {1, iM}, {false, true}, ms);
                    arma::mat mesh_t = side ? ub::build({0.0, gap, gap + tC}, {iM, 2, 2}, {false, false, true}, mt)
                                            : ub::build({0.0, tC, tC + gap}, {2, 2, iM}, {false, true, false}, mt);
                    auto sT = ub::chain(pal, mesh_s, ms, {tr}, f, thd, {0.0}), sR = ub::chain(pal, mesh_s, ms, {3}, f, thd, {0.0});
                    auto tT = ub::chain(pal, mesh_t, mt, {tr, tr}, f, thd, {0.0}), tR1 = ub::chain(pal, mesh_t, mt, {3}, f, thd, {0.0});
                    auto tR2 = ub::chain(pal, mesh_t, mt, {tr, 3, tr}, f, thd, {0.0});
                    REQUIRE((sT.size() == 1 && sR.size() == 1 && tT.size() == 2 && tR1.size() == 1 && tR2.size() == 3));
                    const std::complex<double> Ts = cpx(sT[0].xpr, 0, 0) * std::exp(-J * k * c * tC), Rs = cpx(sR[0].xpr, 0, 0);
                    const std::complex<double> Tt = ub::at_point(tT, f, thd), R1 = ub::at_point(tR1, f, thd), R2 = ub::at_point(tR2, f, thd);
                    const std::complex<double> Rt = (side ? R1 : R1 + R2) * std::exp(J * k * c * tC);
                    // The cosine the wave has in the coherent layer, off the direction the entry wrote
                    const arma::mat &dC = tT[0].path_dirN;
                    const double cC = std::abs(dC(0, 0)) / std::sqrt(ub::sumsq(dC));
                    const LumpMtl &Mm = lp[iM - 1];
                    const double mass_cos = side ? 0.0 : diag_mass_TL(Mm.m, Mm.fRef, f, Mm.thk, cC) - diag_mass_TL(Mm.m, Mm.fRef, f, Mm.thk, c);
                    QD_DIAG("%s,%s,%d,%.0f,%.4f,%.6f,%.4f,%.4f,%.6f,%.4f,%.4f,%.6f\n", side ? "M" : "C", (iM == 3) ? "26.8 dB" : "6.8 dB", tr, thd, ub::dB(Ts),
                            ub::dB(Ts / Tt), ub::deg(Ts / Tt), ub::dB(Rs), ub::dB(Rs / Rt), ub::deg(Rs / Rt), ub::dB(R2 / R1), mass_cos);
                    INFO("mixed stack side " << side << " mass " << iM << " tracer " << tr << " theta " << thd);
                    // One laminate on the reflect port, from either side, and one partition
                    CHECK(std::abs(ub::dB(Rs / Rt)) < 5.0e-3);
                    CHECK(std::abs(ub::deg(Rs / Rt)) < 0.05);
                    CHECK(std::abs(ub::deg(Ts / Tt)) < 0.05);
                    // And on the transmit port, from either side: the lining behind the traced solid
                    // forms its mass term at the cosine the wave has in air, as the stack does
                    CHECK(std::abs(ub::dB(Ts / Tt)) < 5.0e-3);
                }
}

TEST_CASE("Mass cosine - a mass-law sheet behind a refracting layer forms its term at the cosine the wave has in air", "[acoustic]")
{
    ub::OneThread one_thread;
    // The mass term of a plate has an angle factor, and the angle is the one the transverse
    // invariant has in air: the plate is driven at the trace wavenumber, and the term is
    // calibrated on a plate in air. Behind a layer the wave was refracted into, that is the
    // incidence cosine and not the cosine in the layer. The mass paid is taken as a ratio: the
    // chain on a twin palette whose mass argument is taken to nothing over the chain itself,
    // which leaves every face, tf and the m > 0 gates in place. The sheet's faces absorb all
    // that meets them (tf 1): the plate rule then passes the term itself, a tau / (a + tau - a tau)
    // with a = 1 (md/acoustic.md 10 N1), whatever host the face is met from.
    const double f = FRQ, gap = 1.0e-5;
    const std::vector<LumpMtl> lp = {
        {2.1609e-8, 0.0, 0.022, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 mass-law sheet, 26.8 dB
        {2.1609e-8, 0.0, 0.022, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.010}, // 2 the same near its knee, 6.8 dB
        {2.50, 0.80, 0.0},                                                  // 3 lossy host
        {2.50, 0.0, 0.0},                                                   // 4 lossless host
        {0.31, 0.20, 0.0},                                                  // 5 porous host, rarer than air
        {5.24, 0.63, 0.006},                                                // 6 coherent sheet
        {5.24, 0.0, 0.0},                                                   // 7 dense lossless solid
        {1.50, 0.40, 0.0},                                                  // 8 lossy host, entered from 7
    };
    std::vector<LumpMtl> lt = lp;
    lt[0].fRef *= 1.0e12, lt[1].fRef *= 1.0e12;
    const ub::Pal pal = side0(lump_palette(lp)), twin = side0(lump_palette(lt));
    auto TL = [&](int iM, double c) { return diag_mass_TL(lp[iM - 1].m, lp[iM - 1].fRef, f, lp[iM - 1].thk, c); };
    auto prod = [](const std::vector<ub::Ev> &v) { std::complex<double> z = 1.0; for (const auto &e : v) z *= cpx(e.xpr, 0, 0); return z; };
    auto cosx = [](const arma::mat &d) { return std::abs(d(0, 0)) / std::sqrt(ub::sumsq(d)); };
    auto ang = [](const arma::mat &d) { return std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI; };
    auto hname = [](int h) { return h == 3 ? "2.50-0.80j" : (h == 4 ? "2.50" : (h == 5 ? "0.31-0.20j" : "1.50-0.40j")); };

    // ------------------------------------------------------------------------------------------
    // A. On the far face of a host, inside it, and in a stack inside it, against the sheets in air
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mass cosine A, a mass-law sheet 22 mm reached from inside a host without the mass column, scalar, through the\n");
    QD_DIAG("# intersector. paid: the mass paid, twin over run, dB. TL_air, TL_host: the term at the incidence cosine and at the cosine the\n");
    QD_DIAG("# wave has in the host. air: paid by the same sheet or stack standing in air, through the library. rest: the chain on the twin,\n");
    QD_DIAG("# dB, what a change to the mass term must not move. dir, seg: direction written and segment behind the wall less the incidence\n");
    QD_DIAG("# angle, deg; dy: last hit point against the same host without the sheet, m (a stack with the refracting tracer is recorded)\n");
    QD_DIAG("geom,host,mass,tracer,theta,paid,TL_air,TL_host,paid-TL_air,paid-air,rest,dir,seg,dy\n");
    for (int geom = 0; geom < 4; ++geom)
        for (int H : {3, 4, 5})
            for (int iM : {1, 2})
                for (int tr : {4, 5})
                    for (double thd : {10.0, 30.0, 60.0})
                    {
                        const double d1 = 0.010, d2 = 0.012, c_air = std::cos(thd * QPI / 180.0);
                        arma::uvec mi, m0, ma;
                        arma::mat mesh, mesh0, mesha;
                        const char *nm = "";
                        if (geom == 0)
                            mesh = ub::build({0.0, d1, d1 + gap}, {H, H, iM}, {false, true, false}, mi), mesh0 = ub::build({0.0, d1}, {H, H}, {false, true}, m0), nm = "glued exit";
                        else if (geom == 1)
                            mesh = ub::build({0.0, d1, d1 + d2}, {H, iM, H}, {false, false, true}, mi), mesh0 = ub::build({0.0, d1 + d2}, {H, H}, {false, true}, m0), nm = "sheet inside";
                        else
                            mesh = ub::build({0.0, d1, d1 + gap, d1 + gap + d2}, {H, (geom == 2) ? 6 : iM, (geom == 2) ? iM : 6, H}, {false, false, false, true}, mi),
                            mesh0 = ub::build({0.0, d1 + gap + d2}, {H, H}, {false, true}, m0), nm = (geom == 2) ? "stack C|M inside" : "stack M|C inside";
                        if (geom < 2)
                            mesha = ub::face(0.0, false), ma = arma::uvec{(arma::uword)iM, (arma::uword)iM};
                        else
                            mesha = ub::build({0.0, gap}, {(geom == 2) ? 6 : iM, (geom == 2) ? iM : 6}, {false, false}, ma);
                        const std::vector<int> seq = (geom == 0) ? std::vector<int>{tr, tr} : std::vector<int>{tr, tr, tr};
                        auto a = ub::chain(pal, mesh, mi, seq, f, thd, {1.5}), b = ub::chain(twin, mesh, mi, seq, f, thd, {1.5});
                        auto c = ub::chain(pal, mesh0, m0, {tr, tr}, f, thd, {1.5});
                        auto sa = ub::chain(pal, mesha, ma, {tr}, f, thd, {1.5}), sb = ub::chain(twin, mesha, ma, {tr}, f, thd, {1.5});
                        INFO("A, " << nm << ", host " << hname(H) << ", mass " << iM << ", tracer " << tr << ", theta " << thd);
                        REQUIRE((a.size() == seq.size() && b.size() == seq.size() && c.size() == 2 && sa.size() == 1 && sb.size() == 1));
                        const double paid = ub::dB(prod(b) / prod(a)), air = ub::dB(prod(sb) / prod(sa)), c_h = cosx(a[0].path_dirN);
                        const ub::Ev &E = a.back();
                        const double d_dir = ang(E.path_dirN) - thd, d_seg = ang(E.destN - E.origN) - thd, dy = E.fbs(0, 1) - c.back().fbs(0, 1);
                        QD_DIAG("%s,%s,%s,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e\n", nm, hname(H), (iM == 1) ? "26.8 dB" : "6.8 dB", tr, thd, paid, TL(iM, c_air),
                                TL(iM, c_h), paid - TL(iM, c_air), paid - air, ub::dB(prod(b)), d_dir, d_seg, dy);
                        CHECK(std::abs(paid - TL(iM, c_air)) < 1.0e-6); // the term at the cosine in air
                        CHECK(std::abs(paid - air) < 1.0e-6);           // which is what the same sheets pay in air
                        CHECK(std::abs(air - TL(iM, c_air)) < 1.0e-9);
                        CHECK(std::abs(d_dir) < 1.0e-4); // and the wall is left in the direction it was met in
                        if (geom < 2)
                            CHECK(std::abs(dy) < 1.0e-9); // where the same host without the sheet is left
                    }

    // ------------------------------------------------------------------------------------------
    // B. One wall from its two sides
    // ------------------------------------------------------------------------------------------
    // The laminate of the branch case, C a traced solid 10 mm with the mass-law sheet as a lining,
    // and the stack of the two as sheets, from the side of C and from the side of M. The mass paid
    // is one number. The whole transmission is not: recorded as d_sides, with what it reads.
    QD_DIAG("\n# diag: mass cosine B, the laminate from its two sides, C 10 mm eps 2.50 - j ei as a traced solid with the mass-law lining, and\n");
    QD_DIAG("# the stack of the two. paid_*: mass paid, dB, traced laminate from the C side and from the M side, stack C|M and M|C.\n");
    QD_DIAG("# d_sides: the traced laminate's transmission from the M side over that from the C side, dB, recorded; tan2: 10 log10 tan^2 theta\n");
    QD_DIAG("ei,mass,tracer,theta,TL_air,paid_trC,paid_trM,paid_stCM,paid_stMC,d_sides,d_sides_stack,tan2\n");
    for (double ei : {0.80, 0.0})
        for (int iM : {1, 2})
            for (int tr : {4, 5})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    const double tC = 0.010, th = thd * QPI / 180.0;
                    std::vector<LumpMtl> q = {lp[iM - 1], {2.50, ei, tC}, {2.50, ei, 0.0}};
                    std::vector<LumpMtl> qt = q;
                    qt[0].fRef *= 1.0e12;
                    double v[2][4];
                    for (int k = 0; k < 2; ++k)
                    {
                        ub::Pal P = side0(lump_palette(k ? qt : q));
                        arma::uvec m1, m2, m3, m4;
                        arma::mat sCM = ub::build({0.0, gap}, {2, 1}, {false, true}, m1), sMC = ub::build({0.0, gap}, {1, 2}, {false, true}, m2);
                        arma::mat tCs = ub::build({0.0, tC, tC + gap}, {3, 3, 1}, {false, true, false}, m3), tMs = ub::build({0.0, gap, gap + tC}, {1, 3, 3}, {false, false, true}, m4);
                        v[k][0] = ub::dB(prod(ub::chain(P, tCs, m3, {tr, tr}, f, thd, {0.0})));
                        v[k][1] = ub::dB(prod(ub::chain(P, tMs, m4, {tr, tr}, f, thd, {0.0})));
                        v[k][2] = ub::dB(prod(ub::chain(P, sCM, m1, {tr}, f, thd, {0.0})));
                        v[k][3] = ub::dB(prod(ub::chain(P, sMC, m2, {tr}, f, thd, {0.0})));
                    }
                    const double ref = TL(iM, std::cos(th));
                    QD_DIAG("%.2f,%s,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n", ei, (iM == 1) ? "26.8 dB" : "6.8 dB", tr, thd, ref, v[1][0] - v[0][0], v[1][1] - v[0][1],
                            v[1][2] - v[0][2], v[1][3] - v[0][3], v[0][1] - v[0][0], v[0][3] - v[0][2], 10.0 * std::log10(std::tan(th) * std::tan(th)));
                    INFO("B, ei " << ei << ", mass " << iM << ", tracer " << tr << ", theta " << thd);
                    for (int j = 0; j < 4; ++j)
                        CHECK(std::abs(v[1][j] - v[0][j] - ref) < 1.0e-6); // one mass term, from either side, traced or as a stack
                }

    // ------------------------------------------------------------------------------------------
    // C. A wedge: the recovery declines and the cosine is the real-angle one
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mass cosine C, a host whose exit face is tilted by 10 deg, the mass-law lining on it. Cosines against the exit normal.\n");
    QD_DIAG("# TL_air: the term at Re sqrt(1 - eps (1 - c^2)), c the cosine the wave has in the host; TL_host: at c; word: the\n");
    QD_DIAG("# previous-medium word the entry wrote, which carries the non-parallel flag\n");
    QD_DIAG("host,tracer,theta,word,paid,TL_air,TL_host,paid-TL_air\n");
    for (int H : {3, 4})
        for (int tr : {4, 5})
            for (double thd : {10.0, 30.0, 60.0})
            {
                const double al = 10.0 * QPI / 180.0, L = 0.050, nx = std::cos(al), ny = std::sin(al), y0 = 0.37 + std::tan(thd * QPI / 180.0);
                arma::mat mesh = arma::join_cols(arma::join_cols(ub::face(0.0, false), ub::tilted(L, y0, 10.0, true)), ub::tilted(L + gap * nx, y0 + gap * ny, 10.0, false));
                arma::uvec mi = {(arma::uword)H, (arma::uword)H, (arma::uword)H, (arma::uword)H, 1, 1};
                auto a = ub::chain(pal, mesh, mi, {tr, tr}, f, thd, {1.5}), b = ub::chain(twin, mesh, mi, {tr, tr}, f, thd, {1.5});
                INFO("C, host " << hname(H) << ", tracer " << tr << ", theta " << thd);
                REQUIRE((a.size() == 2 && b.size() == 2));
                const arma::mat &dh = a[0].path_dirN;
                const double c_h = std::abs(dh(0, 0) * nx + dh(0, 1) * ny) / std::sqrt(ub::sumsq(dh));
                const std::complex<double> eH(lp[H - 1].a, -lp[H - 1].ei);
                const double c_a = std::real(std::sqrt(1.0 - eH * (1.0 - c_h * c_h))), paid = ub::dB(prod(b) / prod(a));
                QD_DIAG("%s,%d,%.0f,%04x,%.6f,%.6f,%.6f,%.6f\n", hname(H), tr, thd, ub::w(a[0].prev(0)), paid, TL(1, c_a), TL(1, c_h), paid - TL(1, c_a));
                CHECK((ub::w(a[0].prev(0)) & 0x8000u) != 0u);
                CHECK(std::abs(paid - TL(1, c_a)) < 1.0e-6);
            }

    // ------------------------------------------------------------------------------------------
    // D. An invariant across one
    // ------------------------------------------------------------------------------------------
    // The host is entered from the lossless eps 5.24 solid the ray is launched in. Below an
    // invariant of one the cosine in air is sqrt(1 - p); from one on there is none and the term
    // keeps the cosine the wave has in the host, as the direction a closed face hands on does.
    QD_DIAG("\n# diag: mass cosine D, a host 50 mm entered from the lossless eps 5.24 solid the ray is launched in, invariant p, then the\n");
    QD_DIAG("# mass-law sheet on its far face or inside it; the sheet event alone. TL_air at sqrt(1 - p), -1 from one on; T: that event, dB\n");
    QD_DIAG("geom,host,tracer,p,rt,cos_host,paid,TL_host,TL_air,T_dB\n");
    for (int geom = 0; geom < 2; ++geom)
        for (int H : {8, 4})
            for (int tr : {4, 5})
                for (double p : {0.50, 0.90, 0.99, 1.01, 1.04, 1.10, 1.45})
                {
                    const double L = 0.050, th1 = std::asin(std::sqrt(p / 5.24));
                    arma::uvec mi;
                    arma::mat mesh = geom ? ub::build({0.0, gap, L, 2.0 * L}, {7, H, 1, H}, {true, false, false, true}, mi)
                                          : ub::build({0.0, gap, L, L + gap}, {7, H, H, 1}, {true, false, true, false}, mi);
                    arma::mat orig{{-0.05, 0.37, 0.0}}, dest{{-0.05 + 2.0 * std::cos(th1), 0.37 + 2.0 * std::sin(th1), 0.0}};
                    ub::Ev seed;
                    seed.prev = arma::Col<short>{(short)0}, seed.cur = arma::Col<short>{(short)7}, seed.buf = arma::Col<short>{(short)0};
                    seed.path_dirN = (dest - orig) / std::sqrt(ub::sumsq(dest - orig)), seed.accg = arma::vec{0.0}, seed.accr = arma::vec{0.0};
                    ub::Opt op;
                    op.eps = 1.5, op.relaunch = true;
                    std::complex<double> e2[2];
                    double ch = -1.0;
                    int rt2 = -1;
                    for (int k = 0; k < 2; ++k)
                    {
                        const ub::Pal &P = k ? twin : pal;
                        ub::Ev E1 = ub::traced<double>(P, mesh, mi, tr, f, orig, dest, &seed, op);
                        arma::mat o, d;
                        ub::onward(E1, o, d);
                        ub::Ev E2 = ub::traced<double>(P, mesh, mi, tr, f, o, d, &E1, op);
                        e2[k] = cpx(E2.xpr, 0, 0);
                        if (!k)
                            rt2 = E2.rt, ch = cosx(E1.path_dirN);
                    }
                    INFO("D, geom " << geom << ", host " << hname(H) << ", tracer " << tr << ", p " << p);
                    REQUIRE(std::abs(e2[0]) > 0.0);
                    const double paid = ub::dB(e2[1] / e2[0]), ca = (p < 1.0) ? std::sqrt(1.0 - p) : -1.0;
                    QD_DIAG("%s,%s,%d,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.4f\n", geom ? "sheet inside" : "glued exit", hname(H), tr, p, rt2, ch, paid, TL(1, ch),
                            (ca > 0.0) ? TL(1, ca) : -1.0, ub::dB(e2[0]));
                    CHECK(std::abs(paid - TL(1, (p < 1.0) ? ca : ch)) < 1.0e-6);
                }

    // ------------------------------------------------------------------------------------------
    // E. What keeps its cosine, and ray_mesh_interact's own result
    // ------------------------------------------------------------------------------------------
    // A ray inside the lossless host with no state and no direction handed in is a leg the state
    // does not know to be bent: it keeps the cosine of its segment. ray_mesh_interact's own glued
    // exit has no state either: with the undeviated tracer its segment runs at the external angle
    // and the term is the one in air, with the refracting tracer it is the cosine of the bent
    // segment, which is recorded.
    QD_DIAG("\n# diag: mass cosine E. launched: a ray inside the lossless host at a real angle, no state, glued exit; own: what ray_mesh_interact\n");
    QD_DIAG("# wrote for the glued exit of table A, host 2.50, before the redo. paid against the term at the cosine of the segment\n");
    QD_DIAG("row,tracer,theta,paid,TL_seg,TL_air,paid-TL_seg,paid-TL_air\n");
    for (int tr : {4, 5})
        for (double thh : {10.0, 20.0, 30.0})
        {
            const double L = 0.050, th1 = thh * QPI / 180.0;
            arma::uvec mi;
            arma::mat mesh = ub::build({L, L + gap}, {4, 1}, {true, false}, mi);
            arma::mat orig{{-0.05, 0.37, 0.0}}, dest{{-0.05 + 2.0 * std::cos(th1), 0.37 + 2.0 * std::sin(th1), 0.0}};
            ub::Opt op;
            op.eps = 1.5;
            ub::Ev A = ub::traced<double>(pal, mesh, mi, tr, f, orig, dest, nullptr, op), B = ub::traced<double>(twin, mesh, mi, tr, f, orig, dest, nullptr, op);
            const double c_h = std::cos(th1), c_s = std::sqrt(1.0 - 2.5 * (1.0 - c_h * c_h)), paid = ub::dB(cpx(B.xpr, 0, 0) / cpx(A.xpr, 0, 0));
            QD_DIAG("launched,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f\n", tr, thh, paid, TL(1, c_h), TL(1, c_s), paid - TL(1, c_h), paid - TL(1, c_s));
            INFO("E launched, tracer " << tr << ", theta " << thh);
            CHECK(std::abs(paid - TL(1, c_h)) < 1.0e-9);
        }
    for (int tr : {4, 5})
        for (double thd : {10.0, 30.0, 60.0})
        {
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, 0.010, 0.010 + gap}, {4, 4, 1}, {false, true, false}, mi);
            auto a = ub::chain(pal, mesh, mi, {tr, tr}, f, thd, {1.5}), b = ub::chain(twin, mesh, mi, {tr, tr}, f, thd, {1.5});
            INFO("E own, tracer " << tr << ", theta " << thd);
            REQUIRE((a.size() == 2 && b.size() == 2));
            const double c_air = std::cos(thd * QPI / 180.0), c_seg = (tr == 4) ? c_air : cosx(a[0].path_dirN);
            const double own = ub::dB(cpx(b[1].xpr0, 0, 0) / cpx(a[1].xpr0, 0, 0));
            QD_DIAG("own,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f\n", tr, thd, own, TL(1, c_seg), TL(1, c_air), own - TL(1, c_seg), own - TL(1, c_air));
            CHECK(std::abs(own - TL(1, c_air)) < 1.0e-9); // the cosine the wave has in air, with either tracer
        }
}

TEST_CASE("Held lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall as its traced solid", "[acoustic]")
{
    ub::OneThread one_thread;
    // A wave inside a rigid mass-law wall is held undeviated: it is a leak, it keeps the incidence
    // angle, and a face towards the lower index is charged what the other direction charges. A
    // mass-law sheet on the far face of such a wall, inside it, or a stack of two inside it, is a
    // later face of the same wall and is met the same way: near | sheet and sheet | far through
    // Material::interact_with with the wave held, at the incidence cosine, and the mass term. The
    // reference needs no formula: the same bodies meshed as traced solids, which are held today.
    //
    // The rule is taken towards the lower index, the gate of the undeviated rule. Towards the
    // higher index the sheet keeps the Snell cosine of the two mappings and its rows must not
    // move; they are printed with every digit in table G.
    //
    // Every level is taken with the mass argument at nothing (fRef times 1e12), which leaves the
    // faces, tf and the m > 0 gates in place; through ray_triangle_intersect, ray_mesh_interact
    // and ray_state_update.
    //
    // A. A rigid lining on the far face, towards the lower index: both sides and the solid twin
    // B. The same with types 1 and 2, per channel
    // C. A rigid sheet inside a rigid wall against three solids
    // D. The reflect port the held wave sees, against the face of the solid twin
    // E. Two rigid sheets inside a rigid wall, every order, against four solids
    // F. The mass paid
    // G. Twins that must not move, with every digit
    // H. Recorded: what the rule does not reach
    const double f = FRQ, g = 1.0e-5, BIG = 1.0e12;
    auto pal = [&](double fr)
    {
        std::vector<LumpMtl> q = {
            {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr}, // 1 Ws wood, sheet
            {2.1609e-8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},   // 2 W
            {2.401e-9, 0.0, 0.050, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},  // 3 Ks concrete, sheet
            {2.401e-9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},    // 4 K
            {9.604e-9, 0.0, 0.030, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr}, // 5 Ts a third rigid body, sheet
            {9.604e-9, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},   // 6 T
            {2.1609e-8, 0.0, 0.0, 0.0, 0.2},                                         // 7 W0 wood without the mass column
            {2.25, 0.0, 0.006, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},      // 8 MLs mass-law, not rigid, sheet
            {2.25, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},        // 9 ML
            {2.401e-9, 0.0, 0.050, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},  // 10 Kns concrete without tf, sheet
            {2.401e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * fr},    // 11 Kn
            {2.50, 0.80, 0.010},                                                     // 12 Cs coherent, sheet
            {2.50, 0.80, 0.0},                                                       // 13 C
            {0.31, 0.20, 0.010},                                                     // 14 As porous, sheet
            {0.31, 0.20, 0.0}};                                                      // 15 A
        return lump_palette(q);
    };
    const ub::Pal P0 = pal(BIG), P1 = pal(1.0);
    struct Body
    {
        const char *name;
        int sheet, solid;
        double thk;
    };
    const Body Wb = {"W", 1, 2, 0.022}, Kb = {"K", 3, 4, 0.050}, Tb = {"T", 5, 6, 0.030}, W0b = {"W0", 0, 7, 0.022}, MLb = {"ML", 8, 9, 0.006},
               Knb = {"Kn", 10, 11, 0.050}, Cb = {"C", 12, 13, 0.010}, Ab = {"A", 14, 15, 0.010};
    const bool F = false, T = true;
    auto prod = [](const std::vector<ub::Ev> &v, int k)
    { std::complex<double> z = 1.0; for (const auto &e : v) z *= cpx(e.xpr, 0, k); return z; };
    // A chain that ends early passed nothing
    auto level = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int n, int tr, double thd, int k)
    {
        auto a = ub::chain(P, mesh, mi, std::vector<int>((size_t)n, tr), f, thd, {1.5});
        return ((int)a.size() == n) ? prod(a, k) : std::complex<double>(0.0, 0.0);
    };
    // Wall X 100 mm with Y on one face: 0 the lining met last, 1 met first, 2 and 3 the same two as traced solids
    auto lined = [&](const Body &X, const Body &Y, int form, arma::uvec &mi, int &n)
    {
        n = (form < 2) ? 2 : 3;
        if (form == 0)
            return ub::build({0.0, 0.1, 0.1 + g}, {X.solid, X.solid, Y.sheet}, {F, T, F}, mi);
        if (form == 1)
            return ub::build({0.0, g, g + 0.1}, {Y.sheet, X.solid, X.solid}, {F, F, T}, mi);
        if (form == 2)
            return ub::build({0.0, 0.1, 0.1 + g, 0.1 + g + Y.thk}, {X.solid, X.solid, Y.solid, Y.solid}, {F, T, F, T}, mi);
        return ub::build({0.0, Y.thk, Y.thk + g, Y.thk + g + 0.1}, {Y.solid, Y.solid, X.solid, X.solid}, {F, T, F, T}, mi);
    };
    // X 50 mm, Y, X 50 mm: 0 Y a sheet inside, 1 three solids
    auto inside = [&](const Body &X, const Body &Y, int form, arma::uvec &mi, int &n)
    {
        n = form ? 4 : 3;
        if (form == 0)
            return ub::build({0.0, 0.05, 0.1}, {X.solid, Y.sheet, X.solid}, {F, F, T}, mi);
        const double a = 0.05, b = a + g, c = b + Y.thk, d = c + g;
        return ub::build({0.0, a, b, c, d, d + 0.05}, {X.solid, X.solid, Y.solid, Y.solid, X.solid, X.solid}, {F, T, F, T, F, T}, mi);
    };
    // X 50 mm, Y and Z, X 50 mm: 0 an ordered stack of two sheets, 1 four solids
    auto stacked = [&](const Body &X, const Body &Y, const Body &Z, int form, arma::uvec &mi, int &n)
    {
        n = form ? 5 : 3;
        if (form == 0)
            return ub::build({0.0, 0.05, 0.05 + g, 0.1}, {X.solid, Y.sheet, Z.sheet, X.solid}, {F, F, T, T}, mi);
        const double a = 0.05, b = a + g, c = b + Y.thk, d = c + g, e = d + Z.thk, h = e + g;
        return ub::build({0.0, a, b, c, d, e, h, h + 0.05}, {X.solid, X.solid, Y.solid, Y.solid, Z.solid, Z.solid, X.solid, X.solid}, {F, T, F, T, F, T, F, T}, mi);
    };

    // ------------------------------------------------------------------------------------------
    // A. The lining, towards the lower index
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining A, a rigid mass-law wall 100 mm with a rigid mass-law lining rarer by its mapping, mass at nothing, scalar, dB.\n");
    QD_DIAG("# W eps 2.1609e-8 tf 0.2, T eps 9.604e-9 tf 0.15, K eps 2.401e-9 tf 0.1. wall, lining: the side met first, the lining a sheet;\n");
    QD_DIAG("# sol_w, sol_l: the same two as traced solids. d_twin: wall - sol_w; d_sides: wall - lining, the flat first faces of the two bodies\n");
    QD_DIAG("pair,type,theta,wall,lining,sol_w,sol_l,d_twin,d_sides\n");
    for (const auto &pr : std::vector<std::pair<Body, Body>>{{Wb, Kb}, {Tb, Kb}, {Wb, Tb}})
        for (int tr : {4, 5})
        {
            double prev = 0.0, prev_th = -1.0;
            for (double thd : {0.0, 10.0, 19.0, 19.4, 19.6, 20.0, 29.9, 30.1, 41.7, 41.9, 45.0, 60.0, 75.0})
            {
                double L[4];
                for (int form = 0; form < 4; ++form)
                {
                    arma::uvec mi;
                    int n;
                    arma::mat mesh = lined(pr.first, pr.second, form, mi, n);
                    L[form] = ub::dB(level(P0, mesh, mi, n, tr, thd, 0));
                }
                QD_DIAG("%s | %s,%d,%.1f,%.4f,%.4f,%.4f,%.4f,%+.6f,%+.6f\n", pr.first.name, pr.second.name, tr, thd, L[0], L[1], L[2], L[3], L[0] - L[2], L[0] - L[1]);
                INFO("A, " << pr.first.name << " | " << pr.second.name << ", type " << tr << ", theta " << thd);
                for (int j = 0; j < 4; ++j)
                    CHECK(L[j] < 0.0);                    // a passive wall passes no more than it received
                CHECK(std::abs(L[0] - L[2]) < 1.0e-3);    // the lining is its traced solid
                CHECK(std::abs(L[0] - L[1]) < 0.02);      // one transmission from the two sides, to the flat first face
                if (prev_th >= 0.0 && thd - prev_th < 0.5) // across the critical angle of the pair: no step
                    CHECK(std::abs(L[0] - prev) < 0.01);
                prev = L[0], prev_th = thd;
            }
        }

    // ------------------------------------------------------------------------------------------
    // B. Types 1 and 2, per channel
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining B, W 100 mm | K lining with types 1 and 2, VV (k 0) and HH (k 3), dB. The mass column has no meaning in EM\n");
    QD_DIAG("# and is not forbidden there; the row says that the rule is one rule per channel\n");
    QD_DIAG("type,k,theta,wall,lining,sol_w,sol_l,d_twin\n");
    for (int tr : {1, 2})
        for (int k : {0, 3})
            for (double thd : {10.0, 30.0, 60.0})
            {
                double L[4];
                for (int form = 0; form < 4; ++form)
                {
                    arma::uvec mi;
                    int n;
                    arma::mat mesh = lined(Wb, Kb, form, mi, n);
                    L[form] = ub::dB(level(P0, mesh, mi, n, tr, thd, k));
                }
                QD_DIAG("%d,%d,%.0f,%.4f,%.4f,%.4f,%.4f,%+.6f\n", tr, k, thd, L[0], L[1], L[2], L[3], L[0] - L[2]);
                INFO("B, type " << tr << ", k " << k << ", theta " << thd);
                CHECK(L[0] < 0.0);
                CHECK(std::abs(L[0] - L[2]) < 1.0e-3);
            }

    // ------------------------------------------------------------------------------------------
    // C. A rigid sheet inside a rigid wall
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining C, X 50 mm | Y | X 50 mm, Y a sheet inside the wall and Y a traced solid, mass at nothing, scalar, dB\n");
    QD_DIAG("wall,sheet,type,theta,sheet_dB,solids_dB,d_twin\n");
    for (const auto &pr : std::vector<std::pair<Body, Body>>{{Wb, Kb}, {Tb, Kb}, {Wb, Tb}})
        for (int tr : {4, 5})
            for (double thd : {0.0, 10.0, 19.0, 20.0, 30.0, 45.0, 60.0, 75.0})
            {
                double L[2];
                for (int form = 0; form < 2; ++form)
                {
                    arma::uvec mi;
                    int n;
                    arma::mat mesh = inside(pr.first, pr.second, form, mi, n);
                    L[form] = ub::dB(level(P0, mesh, mi, n, tr, thd, 0));
                }
                QD_DIAG("%s,%s,%d,%.0f,%.4f,%.4f,%+.6f\n", pr.first.name, pr.second.name, tr, thd, L[0], L[1], L[0] - L[1]);
                INFO("C, " << pr.first.name << " | " << pr.second.name << ", type " << tr << ", theta " << thd);
                CHECK(L[0] < 0.0);
                CHECK(std::abs(L[0] - L[1]) < 1.0e-3);
            }

    // ------------------------------------------------------------------------------------------
    // D. The reflect port
    // ------------------------------------------------------------------------------------------
    // Enter the wall, reflect at the face the lining, the sheet or the stack sits on. The solid
    // twin reflects off the same plane, so the two are compared in the complex plane.
    QD_DIAG("\n# diag: held lining D, the reflect port of the event the held wave meets, against the same face of the solid twin: |R|^2 in dB\n");
    QD_DIAG("# and |R - R_twin|. lining: W 100 | K; inside: W 50 | K | W 50; stack: W 50 | K T | W 50, whose twin is the W | K face\n");
    QD_DIAG("geom,type,k,theta,R_dB,twin_dB,abs_d\n");
    for (int geo = 0; geo < 3; ++geo)
        for (int tr : {4, 1})
            for (int k : {0, 3})
            {
                if (tr > 2 && k)
                    continue;
                for (double thd : {10.0, 30.0, 60.0})
                {
                    arma::uvec m1, m2;
                    int n;
                    arma::mat sh = (geo == 0) ? lined(Wb, Kb, 0, m1, n) : ((geo == 1) ? inside(Wb, Kb, 0, m1, n) : stacked(Wb, Kb, Tb, 0, m1, n));
                    arma::mat so = (geo == 0) ? lined(Wb, Kb, 2, m2, n) : ((geo == 1) ? inside(Wb, Kb, 1, m2, n) : stacked(Wb, Kb, Tb, 1, m2, n));
                    auto a = ub::chain(P0, sh, m1, {tr, tr - 1}, f, thd, {1.5}), b = ub::chain(P0, so, m2, {tr, tr - 1}, f, thd, {1.5});
                    REQUIRE((a.size() == 2 && b.size() == 2));
                    const std::complex<double> R = cpx(a[1].xpr, 0, k), Rt = cpx(b[1].xpr, 0, k);
                    QD_DIAG("%s,%d,%d,%.0f,%.4f,%.4f,%.3e\n", geo == 0 ? "lining" : (geo == 1 ? "inside" : "stack"), tr, k, thd, ub::dB(R), ub::dB(Rt), std::abs(R - Rt));
                    INFO("D, geom " << geo << ", type " << tr << ", k " << k << ", theta " << thd);
                    CHECK(std::abs(R - Rt) < 1.0e-4);
                }
            }

    // ------------------------------------------------------------------------------------------
    // E. Two rigid sheets inside a rigid wall
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining E, X 50 mm | Y Z | X 50 mm, Y and Z an ordered stack of two sheets and the two as traced solids, mass at\n");
    QD_DIAG("# nothing, scalar, dB. rev: the stack in the other order, which is the wall from its other side. d_twin: the depth of the state\n");
    QD_DIAG("stack,type,theta,stack_dB,solids_dB,rev_dB,d_twin,d_rev\n");
    {
        const Body B3[3] = {Wb, Kb, Tb};
        for (int x = 0; x < 3; ++x)
            for (int sw = 0; sw < 2; ++sw)
                for (int tr : {4, 5})
                    for (double thd : {0.0, 10.0, 19.0, 20.0, 30.0, 60.0, 75.0})
                    {
                        const Body &X = B3[x], &Y = B3[(x + 1 + sw) % 3], &Z = B3[(x + 2 - sw) % 3];
                        arma::uvec m1, m2, m3;
                        int n1, n2, n3;
                        arma::mat s1 = stacked(X, Y, Z, 0, m1, n1), s2 = stacked(X, Y, Z, 1, m2, n2), s3 = stacked(X, Z, Y, 0, m3, n3);
                        const double L0 = ub::dB(level(P0, s1, m1, n1, tr, thd, 0)), L1 = ub::dB(level(P0, s2, m2, n2, tr, thd, 0)), L2 = ub::dB(level(P0, s3, m3, n3, tr, thd, 0));
                        QD_DIAG("%s | %s %s | %s,%d,%.0f,%.4f,%.4f,%.4f,%+.6f,%+.6f\n", X.name, Y.name, Z.name, X.name, tr, thd, L0, L1, L2, L0 - L1, L0 - L2);
                        INFO("E, " << X.name << " | " << Y.name << " " << Z.name << ", type " << tr << ", theta " << thd);
                        CHECK(L0 < 0.0);
                        // The stack is its traced solids to the depth of the state: four traced bodies re-form
                        // the plate's entry face from the body before the last (md/acoustic.md 7), and where its
                        // tf differs from the wall's the traced level is off by their ratio (K | T W 3.01 dB,
                        // K | W T 1.76 dB, T | K W 1.25 dB; the other orders exact), recorded with that bound
                        CHECK(std::abs(L0 - L1) < 3.02);
                        CHECK(std::abs(L0 - L2) < 1.0e-3); // and the stack is one wall in both orders
                    }
    }

    // ------------------------------------------------------------------------------------------
    // F. The mass paid
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining F, the mass paid, the chain with the mass argument at nothing over the chain with fRef 1 MHz, dB, scalar.\n");
    QD_DIAG("# W 100 | K lining from both sides, W 50 | K | W 50 and W 50 | K T | W 50. Printed with every digit: a change to the faces must not move it\n");
    QD_DIAG("geom,theta,paid\n");
    for (int geo = 0; geo < 4; ++geo)
    {
        double side[2][4];
        int q = 0;
        for (double thd : {0.0, 30.0, 60.0, 75.0})
        {
            arma::uvec mi;
            int n;
            arma::mat mesh = (geo < 2) ? lined(Wb, Kb, geo, mi, n) : ((geo == 2) ? inside(Wb, Kb, 0, mi, n) : stacked(Wb, Kb, Tb, 0, mi, n));
            const std::complex<double> z0 = level(P0, mesh, mi, n, 4, thd, 0), z1 = level(P1, mesh, mi, n, 4, thd, 0);
            const double paid = ub::dB(z0) - ub::dB(z1);
            QD_DIAG("%s,%.0f,%.17g\n", geo == 0 ? "lining wall first" : (geo == 1 ? "lining first" : (geo == 2 ? "inside" : "stack")), thd, paid);
            INFO("F, geom " << geo << ", theta " << thd);
            CHECK(paid > 20.0); // 100 mm and 50 mm at 1 GHz against 1 MHz: a term worth something
            if (geo < 2)
                side[geo][q] = paid;
            if (geo == 1)
                CHECK(std::abs(side[0][q] - side[1][q]) < 0.01); // the plate from its two sides, to the thickness code
            ++q;
        }
    }

    // ------------------------------------------------------------------------------------------
    // G. Twins that must not move
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining G, twins, scalar, the complex chain coefficient with every digit. Towards the higher index, a wall without\n");
    QD_DIAG("# the mass column, a lining met first, a mass-law lining that is not rigid, every solid twin\n");
    QD_DIAG("row,type,theta,re,im\n");
    {
        struct Row
        {
            const char *name;
            int kind; // 0 lined, 1 inside
            Body X, Y;
            int form;
        };
        const std::vector<Row> rows = {
            {"K 100 | W lining - wall first", 0, Kb, Wb, 0}, {"K 100 | W lining - lining first", 0, Kb, Wb, 1}, {"K 100 | W - solids", 0, Kb, Wb, 2},
            {"K 50 | W | K 50 - inside", 1, Kb, Wb, 0},      {"W 100 | K lining - lining first", 0, Wb, Kb, 1}, {"W 100 | K - solids", 0, Wb, Kb, 2},
            {"K | W 100 - solids", 0, Wb, Kb, 3},            {"W0 100 | K lining - wall first", 0, W0b, Kb, 0}, {"W0 100 | K lining - lining first", 0, W0b, Kb, 1},
            {"W 100 | ML lining - wall first", 0, Wb, MLb, 0}, {"W 100 | ML lining - lining first", 0, Wb, MLb, 1}, {"W 50 | ML | W 50 - inside", 1, Wb, MLb, 0},
            {"W 50 | K | W 50 - solids", 1, Wb, Kb, 1},
        };
        for (const auto &R : rows)
            for (int tr : {4, 5})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    arma::uvec mi;
                    int n;
                    arma::mat mesh = R.kind ? inside(R.X, R.Y, R.form, mi, n) : lined(R.X, R.Y, R.form, mi, n);
                    const std::complex<double> z = level(P0, mesh, mi, n, tr, thd, 0);
                    QD_DIAG("%s,%d,%.0f,%.17g,%.17g\n", R.name, tr, thd, std::real(z), std::imag(z));
                    INFO("G, " << R.name << ", type " << tr << ", theta " << thd);
                    CHECK(std::norm(z) < 1.0);
                    CHECK(std::norm(z) > 0.0);
                }
    }

    // ------------------------------------------------------------------------------------------
    // H. Recorded
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: held lining H, recorded, what the rule does not reach. wall, lining: the side met first; sol_w: the two as traced solids,\n");
    QD_DIAG("# wall first; -999: nothing passed. K | W and W | ML are one plate, on the twin from both sides in scalar; a lining without\n");
    QD_DIAG("# tf absorbs nothing, so the plate passes nothing from either side (W | Kn); a layer without the mass column behind a held\n");
    QD_DIAG("# wave (W | C, W | A) is on its twin from the wall's side in scalar, its own side is the lossy medium's (md/acoustic.md A2).\n");
    QD_DIAG("# The HH rows (type 1, k 3) of a mass-law material are no use case\n");
    QD_DIAG("pair,type,k,theta,wall,lining,sol_w,d_sides,d_twin\n");
    for (const auto &pr : std::vector<std::pair<Body, Body>>{{Kb, Wb}, {Wb, MLb}, {Wb, Knb}, {Wb, Cb}, {Wb, Ab}})
        for (int tr : {4, 1})
            for (int k : {0, 3})
            {
                if (tr > 2 && k)
                    continue;
                for (double thd : {10.0, 30.0, 60.0, 75.0})
                {
                    double L[3];
                    for (int form = 0; form < 3; ++form)
                    {
                        arma::uvec mi;
                        int n;
                        arma::mat mesh = lined(pr.first, pr.second, form, mi, n);
                        auto a = ub::chain(P0, mesh, mi, std::vector<int>((size_t)n, tr), f, thd, {0.0});
                        L[form] = ((int)a.size() == n) ? ub::dB(prod(a, k)) : -999.0;
                    }
                    QD_DIAG("%s | %s,%d,%d,%.0f,%.4f,%.4f,%.4f,%+.4f,%+.4f\n", pr.first.name, pr.second.name, tr, k, thd, L[0], L[1], L[2], L[0] - L[1], L[0] - L[2]);
                    INFO("H, " << pr.first.name << " | " << pr.second.name << ", type " << tr << ", k " << k << ", theta " << thd);
                    for (int j = 0; j < 3; ++j)
                        CHECK(L[j] < 0.0); // recorded, and bounded: nothing here passes more than it received
                    // A mass-law lining that is not rigid: one of its faces leads towards the lower
                    // index, so it is a later face of the wall, and the wall is its solid twin
                    if (std::string(pr.first.name) == Wb.name && std::string(pr.second.name) == MLb.name)
                        CHECK(std::abs(L[0] - L[2]) < 1.0e-3);
                }
            }
}

TEST_CASE("Lumped columns - an ordered stack: what the core carries, by matrix and by scattering - the mass-law part", "[acoustic]")
{
    // The part of "Lumped columns - an ordered stack: what the core carries, by matrix and by
    // scattering" that evaluates a material with the mass column (m > 0); the description is there.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    const int A = 1, B = 2, A_ATT = 3, B_ATT = 4, A_RES = 5, B_ALPHA = 6, B_TF = 7, B_M = 8;
    const int A_ALPHA = 9, A_THICK = 10;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk},                              // 1 A
        {1.50, 0.40, thk},                              // 2 B
        {5.24, 0.63, thk, 3.0},                         // 3 A_ATT    3 dB
        {1.50, 0.40, thk, 2.0},                         // 4 B_ATT    2 dB
        {5.24, 0.63, thk, 0.0, 0.0, 1.2, 5.0, 0.8},     // 5 A_RES    resonant
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0}, // 6 B_ALPHA  50 dB/m
        {1.50, 0.40, thk, 0.0, 0.25},                   // 7 B_TF     tf 0.25
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0}, // 8 B_M  mass law
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0},       // 9 A_ALPHA  100 dB/m
        {5.24, 0.63, 2.0 * thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0},  // 10 A_THICK 12 mm at 50 dB/m
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

    // ------------------------------------------------------------------------------------------
    // The two columns the matrix cannot hold, on the second layer, both orders, both passes.
    // ------------------------------------------------------------------------------------------
    struct Decl
    {
        const char *name;
        int m2;
    };
    const std::vector<Decl> declined = {
        {"m 10        ", B_M},
    };

    QD_DIAG("\n# diag: ordered stack whose second sheet carries m, resolved in the scattering\n");
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

                // Resolved on both passes and in both modes, and on the reference. The mass law is
                // an acoustic column, so its EM row is the scattering form's two channels and no
                // statement about a use case.
                CHECK(Ct.resolved_type[0] != 0);
                CHECK(Cr.resolved_type[0] != 0);
                CHECK(St.resolved_type[0] != 0);
                CHECK((double)Ct.gain(0) > 1.0e-3);
                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(d_S < 1.0e-9);
            }
}

TEST_CASE("Total reflection of the outer pair - the sheet stays in the result - the mass-law part", "[acoustic]")
{
    // The part of "Total reflection of the outer pair - the sheet stays in the result" that evaluates
    // a material with the mass column (m > 0); the description is there.
    const double f_hz = FRQ;
    const double L = 0.10;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ------------------------------------------------------------------------------------------
    // Both ports, acoustic: a wood panel on a concrete wall, both mass-law, both with tf
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.020;
        const double tf_w = 0.2, tf_c = 0.1, att_w = 1.0, att_c = 2.0;
        const std::vector<LumpMtl> pal = {
            {2.1609e-8, 0.0, thk, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 WOOD   m 20, argument 20
            {2.401e-9, 0.0, 0.0, att_c, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},  // 2 CONC   as a wall
            {2.401e-9, 0.0, 0.0, att_c, tf_c},                                   // 3 CONC0  without its mass law
        };
        auto mtl = side0(lump_palette(pal));
        const double e_w = 2.1609e-8, e_c = 2.401e-9;
        const double G_w = std::pow(10.0, -0.1 * att_w), G_c = std::pow(10.0, -0.1 * att_c);

        // Transmitted power fraction of one lossless crossing e1 -> e2 at incidence angle th, by
        // the rule of Material::interact_with: the flat leak 1 - R' past the critical angle, and
        // 1 - R' below it, with R' the pair rule on the Fresnel reflectance. Seen from air it is
        // what the panel's face absorbs.
        auto cross = [](double e1, double e2, double tf1, double tf2, double th)
        {
            double s = std::sin(th), c = std::cos(th), q = (e1 / e2) * s * s;
            if (std::sqrt(e1 / e2) * s >= 1.0)
                return 1.0 - diag_tf_pair(1.0, tf1, tf2);
            double z1 = std::sqrt(e1), z2 = std::sqrt(e2), c2 = std::sqrt(1.0 - q);
            double r = (z1 * c - z2 * c2) / (z1 * c + z2 * c2);
            return 1.0 - diag_tf_pair(r * r, tf1, tf2);
        };

        QD_DIAG("\n# diag: wood panel (20 mm, tf 0.2, att 1 dB, m 20) on concrete (tf 0.1, att 2 dB), scalar.\n");
        QD_DIAG("# entry is air | wood | concrete, exit is concrete | wood | air. *_dB are power gains;\n");
        QD_DIAG("# TL_w is the panel's mass term\n");
        QD_DIAG("theta_n,TL_w,entry_T_dB,ref_entry_T_dB,entry_R,ref_entry_R,exit_T_dB,ref_exit_T_dB,exit_R,ref_exit_R\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            double TL = diag_mass_TL(20.0, 0.001, f_hz, thk, ct), gM = std::pow(10.0, -0.1 * TL);
            auto dB = [](double g)
            { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

            // Entry, from the room
            auto Et = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Er = make1<double>(mtl, 3, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Et.run(), Er.run();
            // Panel and wall are one plate (md/acoustic.md 10 N1). Its first face passes what it
            // absorbs, a = x1, and the panel's leg pays its own term; the inner face of the two
            // mass-law bodies passes all, and the series of the plate and the wall's term are paid
            // at the wall's far face, which this event does not reach.
            bool leak = std::sqrt(1.0 / e_w) * std::sin(th) >= 1.0;
            double x1 = cross(1.0, e_w, 0.0, tf_w, th);
            double ref_eT = x1 * G_w * gM * G_c, ref_eR = 1.0 - x1;

            // Phase of the entry leak. Each crossing keeps the phase interact_with gives it,
            // arg(1 + R) of its own Fresnel reflection with the decaying root past the critical
            // angle, and a mass-law sheet adds none of its own. The wood | concrete term is zero in
            // every row (below its critical angle from either side).
            auto arg_T = [](double e1, double e2, double th_)
            {
                double s_ = std::sin(th_), c_ = std::cos(th_), a_ = 1.0 - (e1 / e2) * s_ * s_;
                std::complex<double> c2 = (a_ < 0.0) ? std::complex<double>(0.0, -std::sqrt(-a_))
                                                     : std::complex<double>(std::sqrt(a_), 0.0);
                std::complex<double> z1 = std::sqrt(e1), z2 = std::sqrt(e2);
                return std::arg(1.0 + (z1 * c_ - z2 * c2) / (z1 * c_ + z2 * c2));
            };
            std::complex<double> ref_eT_cx = std::polar(std::sqrt(ref_eT), arg_T(1.0, e_w, th) +
                                                                              (leak ? arg_T(e_c, e_w, th) : arg_T(e_w, e_c, th)));
            CHECK(std::abs(cpx(Et.xprmat, 0, 0) - ref_eT_cx) < 1.0e-9 * std::sqrt(ref_eT));

            // Exit, from inside the wall. Concrete without its mass law, so the leg it closes is a
            // pure phase and the panel alone is the plate. It is entered from a wave medium, so the
            // wave in it is refracted (cosine cw) and its far face to air is open. The plate
            // reflects what its near face reflects (the pair rule on the concrete | wood face) and
            // passes the plate rule of its two faces, a = min(1 - Rn, 1 - Rf).
            auto Xt = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 1, 3, 0, 3, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Xr = make1<double>(mtl, 3, OT_GLUED_EXIT, 2u, 1, 3, 0, 3, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Xt.run(), Xr.run();
            double s2 = 1.0 - ct * ct;
            double zc = std::sqrt(e_c), zw = std::sqrt(e_w);
            double cw = std::sqrt(1.0 - (e_c / e_w) * s2), ca = std::sqrt(1.0 - e_c * s2);
            double rn = (zc * ct - zw * cw) / (zc * ct + zw * cw), rf = (zw * cw - ca) / (zw * cw + ca);
            double Rn = diag_tf_pair(rn * rn, tf_c, tf_w), Rf = diag_tf_pair(rf * rf, tf_w, 0.0);
            double a_x = std::min(1.0 - Rn, 1.0 - Rf);
            double ref_xT = a_x * gM / (a_x + gM - a_x * gM) * G_w, ref_xR = Rn;

            QD_DIAG("%.1f,%.4f,%.4f,%.4f,%.9f,%.9f,%.4f,%.4f,%.9f,%.9f\n", thd, TL, dB((double)Et.gain(0)), dB(ref_eT),
                    (double)Er.gain(0), ref_eR, dB((double)Xt.gain(0)), dB(ref_xT), (double)Xr.gain(0), ref_xR);
            INFO("wood on concrete, theta_n " << thd);
            CHECK(std::abs((double)Et.gain(0) - ref_eT) < 1.0e-9 * ref_eT + 1.0e-15);
            CHECK(std::abs((double)Er.gain(0) - ref_eR) < 1.0e-9);
            CHECK(std::abs((double)Xt.gain(0) - ref_xT) < 1.0e-9 * ref_xT + 1.0e-15);
            CHECK(std::abs((double)Xr.gain(0) - ref_xR) < 1.0e-9);
        }
    }
}

TEST_CASE("Leak chain - each face at the cosine the wave has and the series on a closed far face - the mass-law part", "[acoustic]")
{
    // The part of "Leak chain - each face at the cosine the wave has and the series on a closed far
    // face" that evaluates a material with the mass column (m > 0); the description is there.
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

    // ------------------------------------------------------------------------------------------
    // The step at the critical angle of the outer pair, recorded
    // ------------------------------------------------------------------------------------------
    {
        QD_DIAG("\n# diag: step from the open to the closed far face at the critical angle of the outer pair,\n");
        QD_DIAG("# above minus below in dB, +- 1e-5 deg\nstack,theta_c,step_scalar_dB,step_te_dB,step_tm_dB\n");
        // Wood on concrete. The outer pair closes at asin(z_conc) = 0.0028 deg and the air | wood
        // face at asin(z_wood) = 0.0084 deg. Between them the wave in the wood is a refracted one
        // near grazing and the far face is closed for it, so the first angle is continuous. At the
        // second the wave becomes a leak and its angle in the wood drops from grazing to the
        // incidence angle. Panel and wall are one plate (md/acoustic.md 10 N1): the entry passes the
        // first face and the panel's term and the inner face passes all, so neither angle is a step.
        // At bb336dc the second was 4.67 dB, 10 log10(0.82 / 0.28), the inner face's leak.
        const double e_w = 2.1609e-8, e_c = 2.401e-9;
        const std::vector<LumpMtl> pw = {
            {e_w, 0.0, 0.020, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},
            {e_c, 0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},
        };
        auto mw = lump_palette(pw);
        auto wood = [&](double thd)
        {
            auto C = make1<double>(mw, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - thd * QPI / 180.0, 1.5);
            C.run();
            return dB((double)C.gain(0));
        };
        const double t_outer = std::asin(std::sqrt(e_c)) * 180.0 / QPI, t_near = std::asin(std::sqrt(e_w)) * 180.0 / QPI;
        double st_o = wood(t_outer * (1.0 + 1.0e-6)) - wood(t_outer * (1.0 - 1.0e-6));
        double st_n = wood(t_near * (1.0 + 1.0e-6)) - wood(t_near * (1.0 - 1.0e-6));
        QD_DIAG("wood panel on concrete at the outer pair,%.6f,%+.4f,,\n", t_outer, st_o);
        QD_DIAG("wood panel on concrete at the air | wood face,%.6f,%+.4f,,\n", t_near, st_n);
        CHECK(std::abs(st_o) < 1.0e-3);
        CHECK(std::abs(st_n) < 1.0e-3);
    }
}

TEST_CASE("Direction - a ray leaves a parallel slab parallel to how it arrived - the mass-law part", "[acoustic]")
{
    // The part of "Direction - a ray leaves a parallel slab parallel to how it arrived" that evaluates
    // a material with the mass column (m > 0); the description is there.
    const double f_hz = FRQ, L = 0.10;
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

    // ------------------------------------------------------------------------------------------
    // A leak through a mass-law face
    // ------------------------------------------------------------------------------------------
    // Rigid materials, eps = (Z_air / Z)^2: a wave enters as a tf leak and is held undeviated
    // at every face of the body, towards either index: as two traced solids, through a sheet
    // whose outer pair is closed, and at every exit into air.
    QD_DIAG("\n# diag: direction, a leak. Wood eps 2.1609e-8, concrete eps 2.401e-9, both with tf and the mass\n");
    QD_DIAG("# column, 30 deg in, undeviated tracer. One row per event; kept says whether the row is\n");
    QD_DIAG("# asserted undeviated (1)\n");
    QD_DIAG("stack,event,code,rtype,out_deg,d_deg,kept\n");
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8};
        pal["fRef"] = {0.001, 0.001, 0.001};
        pal["tf"] = {0.2, 0.1, 0.2};
        pal["m"] = {20.0, 30.0, 20.0};
        pal["thk"] = {0.0, 0.0, 0.02};
        const int WOOD = 1, CONC = 2, PANEL = 3;
        const double g = 1.0e-5, thd = 30.0;
        struct Stack
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
            std::vector<int> kept; // per event
        };
        const std::vector<Stack> stacks = {
            {"air | wood 50 | conc 50 | air  ", {0.0, 0.05, 0.05 + g, 0.10}, {WOOD, WOOD, CONC, CONC}, {false, true, false, true}, {1, 1, 1}},
            {"air | conc 50 | wood 50 | air  ", {0.0, 0.05, 0.05 + g, 0.10}, {CONC, CONC, WOOD, WOOD}, {false, true, false, true}, {1, 1, 1}},
            {"air | panel | conc 100 | air   ", {0.0, g, 0.10}, {PANEL, CONC, CONC}, {false, false, true}, {1, 1}},
            {"air | conc 100 | panel | air   ", {0.0, 0.10, 0.10 + g}, {CONC, CONC, PANEL}, {false, true, true}, {1, 1}},
            {"air | panel | air              ", {0.0}, {PANEL}, {false}, {1}},
        };
        for (const auto &S : stacks)
        {
            arma::mat mesh;
            arma::uvec mtl_ind(2 * S.x.size());
            for (size_t k = 0; k < S.x.size(); ++k)
            {
                mesh = k ? arma::join_cols(mesh, face(S.x[k], S.flip[k])) : face(S.x[k], S.flip[k]);
                mtl_ind(2 * k) = (arma::uword)S.mat[k], mtl_ind(2 * k + 1) = (arma::uword)S.mat[k];
            }
            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            Ev P;
            const Ev *in = nullptr;
            for (size_t k = 0; k < S.kept.size(); ++k)
            {
                Ev E = traced(pal, mesh, mtl_ind, 4, orig, dest, in, 1.5);
                REQUIRE(E.nh != 0);
                double out = std::atan2(std::abs(E.path_dirN(0, 1)), std::abs(E.path_dirN(0, 0))) * 180.0 / QPI;
                QD_DIAG("%s,%d,%u,%u,%.4f,%+.4f,%d\n", S.name, (int)k, (unsigned)E.ot, (unsigned)E.rt, out, out - thd, S.kept[k]);
                INFO(S.name << ", event " << k);
                if (S.kept[k])
                    CHECK(std::abs(out - thd) < 1.0e-9);
                onward(E, orig, dest);
                P = E, in = &P;
            }
        }
    }
}

TEST_CASE("Refracting tracer - a traced bend is known from the state and the leg is exact - the mass-law part", "[acoustic]")
{
    // The part of "Refracting tracer - a traced bend is known from the state and the leg is exact"
    // that evaluates a material with the mass column (m > 0); the description is there.
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
    using ub::face;
    using ub::build;
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

    // ------------------------------------------------------------------------------------------
    // A leak is not a bend: rigid bodies, entered under total reflection
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: refracting tracer, a leak. Wood eps 2.1609e-8, concrete eps 2.401e-9, tf and the mass column,\n");
    QD_DIAG("# 30 deg. The entry is a total reflection, the wave is held at every face, the bit stays clear and\n");
    QD_DIAG("# every event is the undeviated tracer's. Recorded: DENSE, eps 1.50 with m 20, a mass-law solid\n");
    QD_DIAG("# the wave is refracted into\n");
    QD_DIAG("stack,tracer,event,code,rtype,gain_dB,out_deg,prev,cur,buf\n");
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8, 1.5};
        pal["fRef"] = {0.001, 0.001, 0.001, 0.001};
        pal["tf"] = {0.2, 0.1, 0.2, 0.0};
        pal["m"] = {20.0, 30.0, 20.0, 20.0};
        pal["thk"] = {0.0, 0.0, 0.02, 0.0};
        struct St
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
            bool rigid;
        };
        const std::vector<St> sts = {
            {"air | wood 50 | conc 50 | air", {0.0, 0.05, 0.05 + gap, L}, {1, 1, 2, 2}, {false, true, false, true}, true},
            {"air | conc 50 | wood 50 | air", {0.0, 0.05, 0.05 + gap, L}, {2, 2, 1, 1}, {false, true, false, true}, true},
            {"air | panel | conc 100 | air ", {0.0, gap, L}, {3, 2, 2}, {false, false, true}, true},
            {"air | conc 100 | panel | air ", {0.0, L, L + gap}, {2, 2, 3}, {false, true, true}, true},
            {"air | DENSE m20 100 | air    ", {0.0, L}, {4, 4}, {false, true}, false},
        };
        for (const auto &S : sts)
        {
            arma::uvec mi;
            arma::mat mesh = build(S.x, S.mat, S.flip, mi);
            Chain c4 = run(pal, mesh, mi, {4}, 30.0, 1.5, f_hz, 0.0), c5 = run(pal, mesh, mi, {5}, 30.0, 1.5, f_hz, 0.0);
            REQUIRE(c4.ev.size() == c5.ev.size());
            for (int t = 0; t < 2; ++t)
            {
                const Chain &C = t ? c5 : c4;
                for (size_t k = 0; k < C.ev.size(); ++k)
                    QD_DIAG("%s,%d,%d,%u,%u,%+.6f,%.4f,%04x,%04x,%04x\n", S.name, t ? 5 : 4, (int)k, (unsigned)C.ev[k].ot, (unsigned)C.ev[k].rt,
                            dB(std::norm(cpx(C.ev[k].xpr, 0, 0))),
                            std::atan2(std::abs(C.ev[k].path_dirN(0, 1)), std::abs(C.ev[k].path_dirN(0, 0))) * 180.0 / QPI,
                            (unsigned)(unsigned short)C.ev[k].prev(0), (unsigned)(unsigned short)C.ev[k].cur(0),
                            (unsigned)(unsigned short)C.ev[k].buf(0));
            }
            if (!S.rigid)
                continue;
            INFO(S.name);
            for (size_t k = 0; k < c5.ev.size(); ++k)
            {
                CHECK(c5.ev[k].cur(0) == c4.ev[k].cur(0)); // no bit: the wave was not bent into the body
                CHECK(c5.ev[k].prev(0) == c4.ev[k].prev(0));
                CHECK(c5.ev[k].buf(0) == c4.ev[k].buf(0));
                CHECK(c5.ev[k].rt == c4.ev[k].rt);
                CHECK(cpx(c5.ev[k].xpr, 0, 0) == cpx(c4.ev[k].xpr, 0, 0));
            }
        }
    }
}

TEST_CASE("Phase reference - a traversal carries the phase at the point the polyline leaves at - the mass-law part", "[acoustic]")
{
    // The part of "Phase reference - a traversal carries the phase at the point the polyline leaves
    // at" that evaluates a material with the mass column (m > 0); the description is there.
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
        arma::mat origN, destN, path_dirN, xpr, fbs;
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
        return E;
    };
    using ub::face;
    using ub::build;
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev, evb;    // the events at f, and at the base frequency (geometry, words)
        // per event, for the leg it closes: polyline length and lateral walk between the two hit
        // points, the length the ledger was given (relaunch point to hit point), its normal part,
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
        double xp = 0.0, yp = 0.0, y0 = 0.0;
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
            double ox = fx - orig(0, 0), oy = fy - orig(0, 1);
            C.len.push_back(inside ? seg : 0.0), C.lat.push_back(inside ? std::abs(fy - yp) : 0.0);
            C.led.push_back(inside ? std::sqrt(ox * ox + oy * oy) : 0.0), C.ledx.push_back(inside ? std::abs(ox) : 0.0), C.ledy.push_back(inside ? std::abs(oy) : 0.0);
            C.offy.push_back(std::abs(B.origN(0, 1) - fy));
            if (inside)
                C.Lin += seg;
            xp = fx, yp = fy, C.dy = fy - y0;
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

    // ------------------------------------------------------------------------------------------
    // F. The legs the exact form does not take: a mass-law solid, dense and rigid. A mass-law body is a
    // lumped plate with no wave inside; it holds the wave undeviated whatever its permittivity and
    // the tracer (md/acoustic.md 10, A7), so both are measured against the walk-off form
    // ------------------------------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 1.5};
        pal["fRef"] = {0.001, 0.001};
        pal["tf"] = {0.2, 0.0};
        pal["m"] = {20.0, 20.0};
        QD_DIAG("\n# diag: phase reference, a mass-law solid, scalar, 1 GHz. The traversal alone, as the ratio of the\n");
        QD_DIAG("# absolute transfer through 100 mm over that through 50 mm. dense: eps 1.50, m 20; rigid: wood\n");
        QD_DIAG("# eps 2.1609e-8, tf 0.2, m 20. Both held, against exp(-j k0 n cos(theta) d), the walk-off form.\n");
        QD_DIAG("# bits: tracer 5 against tracer 4, every event\n");
        QD_DIAG("body,tracer,theta,cur,ph_ray,ph_poly,bits\n");
        for (int m : {2, 1})
            for (int tr : {4, 5})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th);
                    arma::uvec mi;
                    arma::mat mesh1 = build({0.0, 0.5 * L}, {m, m}, {false, true}, mi), mesh2 = build({0.0, L}, {m, m}, {false, true}, mi);
                    Chain c1 = run(pal, mesh1, mi, {tr}, thd, 1.5, FRQ, 0.0), c2 = run(pal, mesh2, mi, {tr}, thd, 1.5, FRQ, 0.0);
                    Chain d1 = run(pal, mesh1, mi, {4}, thd, 1.5, FRQ, 0.0);
                    REQUIRE((c1.ok && c2.ok && c1.ev.size() == 2 && c2.ev.size() == 2 && d1.ev.size() == 2));
                    double n = std::sqrt(m == 2 ? 1.5 : 2.1609e-8), tt = tan_of(c1.ev[0]);
                    double ct = std::cos(th); // held undeviated
                    std::complex<double> a = (c2.abs_vv(k1) / c1.abs_vv(k1)) / std::exp(-J * k1 * n * ct * 0.5 * L);
                    bool same = true;
                    for (size_t k = 0; k < 2; ++k)
                        same = same && cpx(c1.ev[k].xpr, 0, 0) == cpx(d1.ev[k].xpr, 0, 0) && c1.ev[k].rt == d1.ev[k].rt &&
                               c1.ev[k].cur(0) == d1.ev[k].cur(0);
                    QD_DIAG("%s,%d,%.1f,%04x,%+.4f,%+.4f,%s\n", m == 2 ? "dense" : "rigid", tr, thd, (unsigned)(unsigned short)c1.ev[0].cur(0),
                            deg(a * std::exp(J * k1 * st * 0.5 * L * tt)), deg(a * std::exp(J * k1 * st * (c2.dy - c1.dy))), same ? "same" : "differ");
                    INFO("mass-law solid " << m << ", tracer " << tr << ", theta_ext " << thd);
                    CHECK(std::abs(std::abs(c2.abs_vv(k1) / c1.abs_vv(k1)) - 1.0) < 1.0); // a traversal attenuates or holds
                    CHECK(std::abs(deg(a * std::exp(J * k1 * st * (c2.dy - c1.dy)))) < 0.05);
                    if (m == 1)
                        CHECK(same); // a leak is the undeviated tracer's with either type, bit for bit
                }
    }
}

TEST_CASE("Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on the wave's", "[acoustic]")
{
    // A mass-law body is a lumped plate with no wave inside. A wave that entered it as a tf leak
    // is held undeviated, and a face that is charged the same from both sides at one angle is a
    // face without a bend: the wave keeps its direction at every face, towards either index,
    // in the segment and in path_dirN, with either tracer. A wall then transmits parallel to the
    // direction the wave arrived in, a layered wall has one angle throughout, and the leg behind
    // a shared face is not a bent one.
    //
    // A. Rigid walls through the intersector, both tracers. var now is the code. hold is the same
    //    run with the fixture doing the holding: after an event that leaves a mass-law medium it
    //    hands on the direction the wave came in with, relaunches along it where the tracer bends
    //    (type 5), and clears the bent flag of the leg that begins. hold_air does so only where
    //    the event leaves into air, and is recorded: it is the option that was not taken.
    //    Asserted: every direction and every segment on the incidence angle, to 1e-9; no bent
    //    flag; every event of now on hold; the refracting tracer on the undeviated one, in
    //    every word and type code, and in every gain exactly on one wall. Scalar, so the
    //    coefficient of an event does not read the direction it writes.
    // B. A face under total reflection at the corrected angle hands on the direction the wave
    //    has: a bare exit and a transition into another solid as a sheet's closed outer pair
    //    does. The relaunched segment of the refracting tracer is recorded.
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
    using ub::face;
    using ub::build;
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
    auto chain = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int itype, arma::mat orig, arma::mat dest,
                     const Ev *seed, int hold) -> Row
    {
        Row R;
        const std::vector<double> &mcol = pal.at("m");
        Ev P;
        bool have = seed != nullptr;
        if (seed)
            P = *seed;
        double y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            arma::mat sd = unit(dest - orig), wd = have ? unit(P.path_dirN) : sd;
            Ev E = traced(pal, mesh, mi, itype, orig, dest, have ? &P : nullptr, 1.5);
            if (E.nh == 0)
                break;
            int mb = have ? mat_of(P.cur(0)) : 0, ma = mat_of(E.cur(0));
            double g = dB(std::norm(cpx(E.xpr, 0, 0)));
            R.seg.push_back(ang(sd)), R.dir.push_back(ang(wd)), R.g.push_back(g), R.code.push_back(E.ot), R.rt.push_back(E.rt);
            if (E.rt == 0)
            {
                R.ok = false;
                break;
            }
            R.tot += g;
            if (k == 0)
                y0 = E.fbs(0, 1);
            R.dy = E.fbs(0, 1) - y0;
            arma::mat d = unit(E.destN - E.origN);
            bool held = mb != 0 && mcol[(size_t)mb - 1] > 0.0 && (hold == 1 || (hold == 2 && ma == 0));
            if (held)
            {
                E.path_dirN = wd;
                if (itype == 5)
                    d = wd;
                if (ma != 0)
                    E.cur(0) = (short)(E.cur(0) & ~(short)0x4000);
            }
            R.out.push_back(ang(E.path_dirN)), R.cur.push_back((unsigned)(unsigned short)E.cur(0));
            // Long enough to cross every face that is left: a segment that ends inside a body
            // preloads the ledger for a ray that ends there, and the next event would add to it
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            P = E, have = true;
        }
        return R;
    };

    // ------------------------------------------------------------------------------------------
    // A. Gains per event, the later faces at the bent and at the incidence angle
    // ------------------------------------------------------------------------------------------
    {
        const int WOOD = 1, CONC = 2, PANEL = 3;
        Pal heavy;
        heavy["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8};
        heavy["fRef"] = {0.001, 0.001, 0.001};
        heavy["tf"] = {0.2, 0.1, 0.2};
        heavy["m"] = {20.0, 30.0, 20.0};
        heavy["thk"] = {0.0, 0.0, 0.02};
        Pal light = heavy;
        light["fRef"] = {0.02, 0.02, 0.02};
        struct St
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
        };
        const std::vector<St> sts = {
            {"wood 50 | conc 50              ", {0.0, 0.05, 0.05 + gap, 0.10}, {WOOD, WOOD, CONC, CONC}, {false, true, false, true}},
            {"conc 50 | wood 50              ", {0.0, 0.05, 0.05 + gap, 0.10}, {CONC, CONC, WOOD, WOOD}, {false, true, false, true}},
            {"panel | conc 100               ", {0.0, gap, 0.10}, {PANEL, CONC, CONC}, {false, false, true}},
            {"conc 100 | panel               ", {0.0, 0.10, 0.10 + gap}, {CONC, CONC, PANEL}, {false, true, true}},
            {"conc 100 / air 500 / conc 100 ", {0.0, 0.10, 0.60, 0.70}, {CONC, CONC, CONC, CONC}, {false, true, false, true}},
            {"conc 100 / air 500 / panel|conc", {0.0, 0.10, 0.60, 0.60 + gap, 0.70}, {CONC, CONC, PANEL, CONC, CONC}, {false, true, false, false, true}},
            {"conc 100 / air 500 / panel     ", {0.0, 0.10, 0.60}, {CONC, CONC, PANEL}, {false, true, false}},
            {"conc|wood / air 500 / wood 50  ", {0.0, 0.05, 0.05 + gap, 0.10, 0.60, 0.65}, {CONC, CONC, WOOD, WOOD, WOOD, WOOD}, {false, true, false, true, false, true}},
        };
        QD_DIAG("\n# diag: leak direction, rigid walls. Wood eps 2.1609e-8 tf 0.2 m 20, concrete eps 2.401e-9 tf 0.1 m 30,\n");
        QD_DIAG("# panel: the wood as a 20 mm sheet. pal heavy: fRef 0.001 (X0 50 and 354 at 50 mm); light: fRef 0.02 (2.5 and 3.95).\n");
        QD_DIAG("# Through the intersector, scalar, tracer 4 undeviated and 5 refracting. var now: the code. hold: every face out of\n");
        QD_DIAG("# a mass-law medium hands on the direction the wave came in with. hold_air: only an exit into air does.\n");
        QD_DIAG("# Per event k: dk the angle of the direction the wave is handed in with, deg; sk the angle of the segment; gk the\n");
        QD_DIAG("# gain in dB. tot: the sum. out: the direction after the last event. dy: lateral walk first to last hit, m\n");
        QD_DIAG("pal,stack,tracer,var,theta,d0,s0,g0,d1,s1,g1,d2,s2,g2,d3,s3,g3,d4,s4,g4,tot_dB,out_deg,dy_m,codes,rts,curs\n");
        double d54 = 0.0; // largest difference of an event's gain between the two tracers, dB
        for (int ip = 0; ip < 2; ++ip)
            for (const auto &S : sts)
                for (double thd : {30.0, 60.0, 75.0})
                {
                    Row RR[2][3];
                    for (int it = 0; it < 2; ++it)
                        for (int hold = 0; hold < 3; ++hold)
                        {
                            int itype = it ? 5 : 4;
                            arma::uvec mi;
                            arma::mat mesh = build(S.x, S.mat, S.flip, mi), orig, dest;
                            diag_ray_at(thd, orig, dest);
                            RR[it][hold] = chain(ip ? light : heavy, mesh, mi, itype, orig, dest, nullptr, hold);
                            const Row &R = RR[it][hold];
                            INFO(S.name << ", tracer " << itype << ", theta " << thd << ", variant " << hold);
                            REQUIRE(R.ok);
                            QD_DIAG("%s,%s,%d,%s,%.0f", ip ? "light" : "heavy", S.name, itype, hold == 0 ? "now     " : (hold == 1 ? "hold    " : "hold_air"), thd);
                            for (size_t k = 0; k < 5; ++k)
                                if (k < R.g.size())
                                    QD_DIAG(",%.4f,%.4f,%+.4f", R.dir[k], R.seg[k], R.g[k]);
                                else
                                    QD_DIAG(",,,");
                            QD_DIAG(",%+.4f,%.4f,%+.5f,", R.tot, R.out.empty() ? 0.0 : R.out.back(), R.dy);
                            for (unsigned c : R.code)
                                QD_DIAG("%u/", c);
                            QD_DIAG(",");
                            for (unsigned c : R.rt)
                                QD_DIAG("%u/", c);
                            QD_DIAG(",");
                            for (unsigned c : R.cur)
                                QD_DIAG("%04x/", c);
                            QD_DIAG("\n");
                        }
                    const double rad = QPI / 180.0;
                    for (int it = 0; it < 2; ++it)
                    {
                        const Row &N = RR[it][0], &H = RR[it][1];
                        INFO((ip ? "light, " : "heavy, ") << S.name << ", tracer " << (it ? 5 : 4) << ", theta " << thd);
                        REQUIRE(N.g.size() == H.g.size());
                        for (size_t k = 0; k < N.g.size(); ++k)
                        {
                            INFO("event " << k);
                            CHECK(std::abs(N.out[k] - thd) * rad < 1.0e-9); // the direction written
                            CHECK(std::abs(N.seg[k] - thd) * rad < 1.0e-9); // the segment the event was met on
                            CHECK(std::abs(N.dir[k] - thd) * rad < 1.0e-9); // the direction handed in
                            CHECK((N.cur[k] & 0x4000u) == 0u);              // no leg is a bent one
                            CHECK(std::abs(N.g[k] - H.g[k]) < 1.0e-9);      // every face at the incidence angle
                        }
                    }
                    {
                        // The refracting tracer is the undeviated one on a rigid stack
                        const Row &N4 = RR[0][0], &N5 = RR[1][0];
                        const bool gapped = std::string(S.name).find('/') != std::string::npos;
                        INFO((ip ? "light, " : "heavy, ") << S.name << ", tracer 5 on tracer 4, theta " << thd);
                        REQUIRE(N4.g.size() == N5.g.size());
                        for (size_t k = 0; k < N4.g.size(); ++k)
                        {
                            INFO("event " << k);
                            CHECK(N5.code[k] == N4.code[k]);
                            CHECK(N5.rt[k] == N4.rt[k]);
                            CHECK(N5.cur[k] == N4.cur[k]);
                            // Exact on one wall. Behind an air gap the two tracers relaunch from
                            // points built differently and the last digit can differ: recorded
                            if (!gapped)
                                CHECK(N5.g[k] == N4.g[k]);
                            d54 = std::max(d54, std::abs(N5.g[k] - N4.g[k]));
                        }
                        CHECK(std::abs(N5.dy - N4.dy) < 1.0e-4); // the same place, up to the relaunch offsets
                    }
                }
        QD_DIAG("# largest difference of one event between tracer 5 and tracer 4, dB: %.1e\n", d54);
    }

}

TEST_CASE("Measurement - the mirror behind a second solid and behind a sheet inside a wall - the mass-law part", "[acoustic]")
{
    // The part of "Measurement - the mirror behind a second solid and behind a sheet inside a wall"
    // that evaluates a material with the mass column (m > 0); the description is there.
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
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0); E.sbs = sbsN;
        return E;
    };
    using ub::face;
    using ub::build;
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev;
        std::vector<double> fx, bx; // per event: x of the hit point, x where the leg that follows starts (qd_leg_start)
        bool ok = true;
        std::string rts, words;
        std::complex<double> at_point(double k, double st) const // the field at the polyline's exit point
        { return vv * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * st * dy)); }
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

    // ------------------------------------------------------------------------------------------
    // E. The thickness code
    // ------------------------------------------------------------------------------------------
    // The library writes the code only out of a mass-law solid, so A carries the mass column
    // here and nothing else of the row is read: the buffer word after the transition A -> B.
    QD_DIAG("\n# diag: mirror, the thickness code. air | A d | B 100 mm | air with the mass column on A (eps 5.24), tracer 4:\n");
    QD_DIAG("# word: the buffer word after A -> B; k: its code; k_fix: the code formula applied to the thickness the leg\n");
    QD_DIAG("# crossed; d_q: the decoded thickness; rel: d_q over the crossed thickness less one; off: the crossed thickness\n");
    QD_DIAG("# over the meshed one less one, the relaunch offset. ph: what d_q - d is on a round trip through A, deg, at 1\n");
    QD_DIAG("# and 60 GHz. dS: 20 log |S_layer(d_q) / S_layer(d)| for B eps 1.50 lossless behind A, dB, at 1 and 60 GHz\n");
    QD_DIAG("d_mm,theta,word,k,k_fix,d_q_mm,rel,off,ph_1GHz,ph_60GHz,dS_1GHz,dS_60GHz\n");
    {
        Pal pm = mtl;
        pm["m"] = std::vector<double>(9, 0.0);
        pm["m"][1] = 20.0;
        for (double d : {0.05, 0.10, 0.0731, 0.0123, 0.20})
            for (double thd : {10.0, 60.0, 85.0})
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, d, d + gap, d + gap + L2}, {2, 2, 6, 6}, {false, true, false, true}, mi);
                Chain c = run(pm, mesh, mi, {4, 4}, {1.5, 1.5}, thd, FRQ);
                INFO("thickness code, d " << d << ", theta " << thd);
                REQUIRE(c.ok);
                unsigned word = (unsigned)(unsigned short)c.ev[1].buf(0);
                int k = (int)(word & 0x7FFFu);
                double LA = c.fx[1] - c.bx[0];
                double kf = 1.0 + std::round((std::log10(LA) + 4.0) * (32766.0 / 6.0));
                double d_q = std::pow(10.0, -4.0 + 6.0 * (double)(k - 1) / 32766.0);
                double ph[2], dS[2];
                for (int i = 0; i < 2; ++i)
                {
                    double f = i ? 60.0e9 : 1.0e9, kk = 2.0 * QPI * f / C0;
                    std::complex<double> eA = diag_eps(2, f), eB = diag_eps(6, f), pB = phi(eB, s2, kk, L2), r_out = rint(eB, e_air, s2);
                    ph[i] = 2.0 * kk * std::real(bof(eA, s2)) * (d_q - LA) * 180.0 / QPI;
                    std::complex<double> Sq = 1.0 / (1.0 - cm(eB, {{eA, d_q}}, e_air, s2, kk).r * r_out * pB * pB);
                    std::complex<double> Sx = 1.0 / (1.0 - cm(eB, {{eA, LA}}, e_air, s2, kk).r * r_out * pB * pB);
                    dS[i] = 2.0 * dB(std::abs(Sq / Sx));
                }
                QD_DIAG("%.2f,%.0f,%04x,%d,%.0f,%.5f,%+.2e,%+.2e,%+.4f,%+.3f,%+.5f,%+.4f\n", d * 1e3, thd, word, k, kf, d_q * 1e3, d_q / LA - 1.0,
                        LA / d - 1.0, ph[0], ph[1], dS[0], dS[1]);
                CHECK((word & 0x8000u) != 0u); // the code is there to be read
            }
    }
}

TEST_CASE("Lossy verdict - a lossy second medium has no critical angle - the mass-law part", "[acoustic]")
{
    // The part of "Lossy verdict - a lossy second medium has no critical angle" that evaluates a
    // material with the mass column (m > 0); the description is there.
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    ub::OneThread one_thread;
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
    using ub::face;
    using ub::build;
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
        {0.31, 0.20, 0.010},                    // 1 porous sheet 10 mm, rarer than air
        {0.31, 0.20, 0.0},                      // 2 the same medium as a solid
        {2.401e-9, 0.0, 0.0},                   // 3 rigid wall
        {2.401e-9, 0.0, 0.0, 0.0, 0.3},         // 4 rigid wall leaking, tf 0.3
        {0.31, 0.20, dS},                       // 5 porous sheet 50 mm
        {2.50, 0.80, 0.010},                    // 6 dense lossy sheet 10 mm
        {0.31, 0.20, 0.010, 0.0, 0.2},          // 7 porous sheet 10 mm with tf 0.2
        {0.31, 0.20, 0.0, 0.0, 0.2},            // 8 porous solid with tf 0.2
        {5.24, 0.00, 0.0},                      // 9 EM: lossless dense solid
        {1.50, 0.40, 0.0},                      // 10 EM: lossy solid, rarer than 9
        {1.50, 0.40, 0.006, 0.0, 0.5},          // 11 EM: the same as a 6 mm sheet with tf 0.5
        {1.50, 0.40, 0.006},                    // 12 EM: the same as a 6 mm sheet
        {0.50, 0.60, 0.0},                      // 13 a rarer solid with more loss, the test trips at 62.1 deg
        {0.31, 0.60, 0.0},                      // 14 the same at 55.3 deg
        {2.401e-9, 1.0e-10, 0.0, 0.0, 0.3},     // 15 the leaking rigid wall with a loss tangent of 0.04
        {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},     // 16 the leaking rigid wall with its mass law
        {2.401e-9, 1.0e-10, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001}, // 17 with both
    };
    Pal mtl = lump_palette(lp);
    const std::complex<double> eP = lump_eps(lp[1], f_hz), eW = lump_eps(lp[2], f_hz), eD = lump_eps(lp[5], f_hz);
    const std::complex<double> eH = lump_eps(lp[8], f_hz), eL = lump_eps(lp[9], f_hz);
    const double tc = std::asin(std::sqrt(std::abs(eP))) * 180.0 / QPI;      // where the test trips, air | porous
    const double tcH = std::asin(std::sqrt(std::abs(eL / eH))) * 180.0 / QPI; // dense | lossy rarer
    const double tcA = std::asin(std::sqrt(std::abs(1.0 / eH))) * 180.0 / QPI; // dense | air, a critical angle
    const std::vector<double> angles = {10.0, 20.0, 30.0, tc - 0.01, tc + 0.01, 45.0, 60.0, 75.0, 85.0};
    auto dir_in = [&](std::complex<double> e1, std::complex<double> e2, double thd) // qd_refract's map, deg
    {
        double th = thd * QPI / 180.0, sn = std::sqrt(std::abs(e1 / e2));
        std::complex<double> c2 = std::sqrt(1.0 - (e1 / e2) * std::sin(th) * std::sin(th));
        return std::atan2(sn * std::sin(th), std::abs(std::real(c2))) * 180.0 / QPI;
    };

    // ------------------------------------------------------------------------------------------
    // J. A rigid wall that is given a loss
    // ------------------------------------------------------------------------------------------
    // Rigid bodies of the acoustic mapping are lossless and closed from 0.003 deg. What a loss
    // column on such a body does to its tf leak, under whatever the verdict is.
    QD_DIAG("\n# diag: lossy verdict J, a rigid wall 200 mm with tf 0.3 in air, scalar, tracer 4 and 5: with its mass law (m 30),\n");
    QD_DIAG("# lossless and with a loss tangent of 0.04. Entry and exit in dB, the direction in the wall and\n");
    QD_DIAG("# behind it in deg, the reflectance of the first face\n");
    QD_DIAG("wall,tracer,theta,ot1,rt1,e1_dB,dir1,ot2,rt2,e2_dB,dir2,|R|^2\n");
    for (int wl : {16, 17})
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

    // ------------------------------------------------------------------------------------------
    // K. The direction a traced absorber and a sheet hand a mass-law wall
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: lossy verdict K, porous 50 mm on a rigid wall 200 mm with tf 0.3 and its mass law (m 30), scalar, series\n");
    QD_DIAG("# summed. As a traced solid (entry, porous | wall, wall | air) and as a sheet (glued entry,\n");
    QD_DIAG("# wall | air). into: the events up to the wall, dB; exit: the wall | air event with the wall's mass term, dB;\n");
    QD_DIAG("# dirW: the direction in the wall, deg; out: the direction behind the wall, deg\n");
    QD_DIAG("tracer,theta,solid_into,solid_exit,solid_dirW,solid_out,sheet_into,sheet_exit,sheet_dirW,sheet_out,exit_diff\n");
    for (int tr : {4, 5})
        for (double thd : {10.0, 20.0, 30.0, 37.0, 45.0, 60.0, 75.0, 85.0})
        {
            arma::uvec mi, mj;
            arma::mat mesh = build({0.0, dS, dS + gap, dS + 0.2 + gap}, {2, 2, 16, 16}, {false, true, false, true}, mi);
            arma::mat mesh2 = build({0.0, gap, 0.2 + gap}, {5, 16, 16}, {false, false, true}, mj);
            std::vector<Ev> so = chain(mtl, mesh, mi, {tr, tr, tr}, f_hz, thd, nullptr, 0.0), sh = chain(mtl, mesh2, mj, {tr, tr}, f_hz, thd, nullptr, 0.0);
            REQUIRE((so.size() == 3 && sh.size() == 2 && so[2].nh != 0 && sh[1].nh != 0));
            double a = dB(cpx(so[0].xpr, 0, 0)) + dB(cpx(so[1].xpr, 0, 0)), b = dB(cpx(so[2].xpr, 0, 0)), c = dB(cpx(sh[0].xpr, 0, 0)), d = dB(cpx(sh[1].xpr, 0, 0));
            QD_DIAG("%d,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%+.4f\n", tr, thd, a, b, ang(so[1].path_dirN), ang(so[2].path_dirN), c, d, ang(sh[0].path_dirN),
                    ang(sh[1].path_dirN), b - d);
            INFO("K, tracer " << tr << ", theta " << thd);
            CHECK(std::abs(b - d) < 5.0e-3);                         // the wall's exit with its mass term
            CHECK(std::abs(ang(so[1].path_dirN) - thd) < 1.0e-6);    // the direction in the wall
            CHECK(std::abs(ang(so[2].path_dirN) - thd) < 1.0e-6);    // and behind it
        }
}

TEST_CASE("Unentered branches - ray_mesh_interact's own glued composite against the redo - the mass-law part", "[acoustic]")
{
    // The part of "Unentered branches - ray_mesh_interact's own glued composite against the redo" that
    // evaluates a material with the mass column (m > 0); the description is there.
    ub::OneThread one_thread;
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
    for (const St &S : {St{"wood panel | concrete", 1, 2}})
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
}

TEST_CASE("Closed plate - a mass-law layer the wave is evanescent in passes the plate rule of its two faces", "[acoustic]")
{
    ub::OneThread one_thread;
    // A mass-law layer that is lossless and in which the wave is evanescent - a rigid layer of the
    // acoustic mapping at every practical angle - is closed on both faces. It is one plate: its faces
    // absorb what the pad of a closed face passes, tf in front of a lossless medium and what the pad
    // passes over what a pad that passes everything would pass next to a lossy one, and the plate
    // passes the incoherent series of the smaller of the two and its term, a tau / (a + tau - a tau)
    // (md/acoustic.md 10 N1). The leak is a statement about normal flux, of the side that carries
    // one, so a chain of raw amplitudes that such a layer cuts is converted to flux piece by piece
    // and not once across the layer. Every level here is taken with the mass argument at nothing
    // (fRef times 1e12), which leaves the faces, tf and the m > 0 gates in place; everything runs
    // through ray_triangle_intersect, ray_mesh_interact and ray_state_update.
    //
    // A. The stack C | M in both orders and the laminate with C a traced solid, from both sides
    // B. tf on the neighbour, and no tf on the layer
    // C. The sheet event: a lining on the far face, on the near face, and the sheet inside a host
    // D. A rigid layer with a mass-law layer that is not rigid, both orders
    // E. Twins that must not move, with every digit
    // F. Recorded: a mass-law layer rarer than the invariant across its critical angle, a rigid
    //    lining on a rigid traced wall, a coherent sheet with tf rarer than its lossy host
    const double f = FRQ, gap = 1.0e-5, tC = 0.010, BIG = 1.0e12;
    const double tfM = 0.2;
    auto prod = [](const std::vector<ub::Ev> &v, int k)
    { std::complex<double> z = 1.0; for (const auto &e : v) z *= cpx(e.xpr, 0, k); return z; };
    auto rigid = [&](double tf, double thk, double fRef)
    { return LumpMtl{2.1609e-8, 0.0, thk, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}; };
    // The absorption of the rigid face seen from a medium eps_h (scalar, TE form): what its pad passes
    // over what a pad that passes everything passes, diag_closed_face with tf and with tf 1
    auto a_face = [](std::complex<double> eps_h, double tf, double tf_h, double thd)
    {
        const double s = std::sin(thd * QPI / 180.0);
        const std::complex<double> ch = std::sqrt(1.0 - s * s / eps_h), y = std::sqrt(eps_h) * ch;
        const std::complex<double> cm = std::complex<double>(0.0, -std::sqrt(s * s / 2.1609e-8 - 1.0)), zm = std::sqrt(2.1609e-8) * cm;
        std::complex<double> r1 = (y - zm) / (y + zm), r2 = r1;
        const double k = diag_closed_face(r1, y, tf, tf_h), kp = diag_closed_face(r2, y, 1.0, tf_h);
        return (kp > 0.0) ? (k / kp) * (k / kp) : 0.0;
    };
    auto plate = [](double a, double tau) { return (a > 0.0 && tau > 0.0) ? a * tau / (a + tau - a * tau) : 0.0; };
    // Entry over exit of a lossy medium, the flux splitting sec^2(arg eta) per channel, dB
    auto split = [](double ei, double thd, int k)
    {
        const std::complex<double> e(2.50, -ei);
        const double s = std::sin(thd * QPI / 180.0);
        const std::complex<double> c = std::sqrt(1.0 - s * s / e), z = std::sqrt(e), x = k ? std::conj(z) * c : z * c;
        return 10.0 * std::log10(std::norm(x) / (std::real(x) * std::real(x)));
    };

    // ------------------------------------------------------------------------------------------
    // A. Stack and traced laminate, both sides, every loss of the coherent layer
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed plate A, C 10 mm eps 2.50 - j ei with M rigid 22 mm tf 0.2 m 20, mass argument at nothing, through the intersector.\n");
    QD_DIAG("# st: the stack of the two as sheets, tr: C a traced solid with M as a lining, series summed; CM: C met first. dB. k: 0 scalar / VV, 3 HH.\n");
    QD_DIAG("# d_st, d_tr: M side over C side; split: 10 log10 sec^2(arg eta) of C in that channel, what entry and exit of a lossy medium differ by\n");
    QD_DIAG("# paid: the mass paid by stCM with the 26.8 dB sheet, less what the plate rule pays with a = min(tf, the face from C) (scalar rows, -1 otherwise)\n");
    QD_DIAG("type,k,ei,theta,stCM,stMC,trC,trM,d_st,d_tr,split,d_st-split,d_tr-split,stCM-trC,stMC-trM,paid-TL\n");
    for (int tr : {4, 5, 1, 2})
        for (int k : {0, 3})
        {
            if (tr > 2 && k)
                continue;
            for (double thd : {10.0, 30.0, 60.0})
            {
                double zero_loss[4] = {0.0, 0.0, 0.0, 0.0}, tiny_loss[4] = {0.0, 0.0, 0.0, 0.0};
                for (double ei : {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0})
                {
                    double v[2][4];
                    for (int tw = 0; tw < 2; ++tw) // 0: mass argument at nothing, 1: the 26.8 dB sheet
                    {
                        std::vector<LumpMtl> q = {rigid(tfM, 0.022, tw ? 0.001 : 0.001 * BIG), {2.50, ei, tC}, {2.50, ei, 0.0}};
                        ub::Pal P = side0(lump_palette(q));
                        arma::uvec m1, m2, m3, m4;
                        arma::mat sCM = ub::build({0.0, gap}, {2, 1}, {false, true}, m1), sMC = ub::build({0.0, gap}, {1, 2}, {false, true}, m2);
                        arma::mat tCs = ub::build({0.0, tC, tC + gap}, {3, 3, 1}, {false, true, false}, m3), tMs = ub::build({0.0, gap, gap + tC}, {1, 3, 3}, {false, false, true}, m4);
                        auto a = ub::chain(P, sCM, m1, {tr}, f, thd, {0.0}), b = ub::chain(P, sMC, m2, {tr}, f, thd, {0.0});
                        auto c = ub::chain(P, tCs, m3, {tr, tr}, f, thd, {0.0}), d = ub::chain(P, tMs, m4, {tr, tr}, f, thd, {0.0});
                        REQUIRE((a.size() == 1 && b.size() == 1 && c.size() == 2 && d.size() == 2));
                        v[tw][0] = ub::dB(prod(a, k)), v[tw][1] = ub::dB(prod(b, k)), v[tw][2] = ub::dB(prod(c, k)), v[tw][3] = ub::dB(prod(d, k));
                    }
                    const double *L = v[0], sp = split(ei, thd, k), d_st = L[1] - L[0], d_tr = L[3] - L[2];
                    const double tau = std::pow(10.0, -0.1 * diag_mass_TL(20.0, 0.001, f, 0.022, std::cos(thd * QPI / 180.0)));
                    const double a = std::min(tfM, a_face(std::complex<double>(2.50, -ei), tfM, 0.0, thd));
                    const double paid = (tr > 2) ? (v[0][0] - v[1][0]) + 10.0 * std::log10(plate(a, tau) / a) : -1.0;
                    QD_DIAG("%d,%d,%.0e,%.0f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.2e,%.2e,%.6f,%.6f,%.2e\n", tr, k, ei, thd, L[0], L[1], L[2], L[3], d_st, d_tr, sp,
                            d_st - sp, d_tr - sp, L[0] - L[2], L[1] - L[3], paid);
                    INFO("A, type " << tr << ", k " << k << ", ei " << ei << ", theta " << thd);
                    for (int j = 0; j < 4; ++j)
                        CHECK(L[j] < 0.0); // a passive wall passes no more than it received
                    CHECK(std::abs(d_st) < 1.0e-6); // one wall from its two sides; the flux splitting of C is recorded beside it
                    CHECK(std::abs(d_tr) < 1.0e-5); // and so is the traced laminate, to its relaunch offsets
                    CHECK(std::abs(L[0] - L[2]) < 5.0e-3); // and the stack is the traced laminate, from either side
                    CHECK(std::abs(L[1] - L[3]) < 5.0e-3);
                    if (tr > 2)
                        CHECK(std::abs(paid) < 1.0e-6); // the mass paid is the plate rule's
                    for (int j = 0; j < 4; ++j)
                    {
                        if (ei == 1.0e-6)
                            tiny_loss[j] = L[j];
                        if (ei == 0.0)
                            zero_loss[j] = L[j];
                    }
                }
                INFO("A, continuity at zero loss, type " << tr << ", k " << k << ", theta " << thd);
                for (int j = 0; j < 4; ++j)
                    CHECK(std::abs(tiny_loss[j] - zero_loss[j]) < 1.0e-3); // ei 1e-6 against ei 0
            }
        }

    // ------------------------------------------------------------------------------------------
    // B. tf on the neighbour, and a layer without tf
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed plate B, the same four with tf on C and with no tf on M, scalar. A closed face with no tf on either side\n");
    QD_DIAG("# passes nothing: -999 is an exact zero, and a level below -120 dB is the rounding of 1 - R0 at a lossless face\n");
    QD_DIAG("type,tfM,tfC,ei,theta,stCM,stMC,trC,trM,d_st,split\n");
    for (int tr : {4, 5})
        for (double tM : {0.2, 0.0})
            for (double tfC : {0.0, 0.3})
                for (double ei : {0.8, 1.0e-3, 0.0})
                    for (double thd : {10.0, 30.0, 60.0})
                    {
                        if (tM == 0.2 && tfC == 0.0)
                            continue; // table A
                        std::vector<LumpMtl> q = {rigid(tM, 0.022, 0.001 * BIG), {2.50, ei, tC, 0.0, tfC}, {2.50, ei, 0.0, 0.0, tfC}};
                        ub::Pal P = side0(lump_palette(q));
                        arma::uvec m1, m2, m3, m4;
                        arma::mat sCM = ub::build({0.0, gap}, {2, 1}, {false, true}, m1), sMC = ub::build({0.0, gap}, {1, 2}, {false, true}, m2);
                        arma::mat tCs = ub::build({0.0, tC, tC + gap}, {3, 3, 1}, {false, true, false}, m3), tMs = ub::build({0.0, gap, gap + tC}, {1, 3, 3}, {false, false, true}, m4);
                        auto a = ub::chain(P, sCM, m1, {tr}, f, thd, {0.0}), b = ub::chain(P, sMC, m2, {tr}, f, thd, {0.0});
                        auto c = ub::chain(P, tCs, m3, {tr, tr}, f, thd, {0.0}), d = ub::chain(P, tMs, m4, {tr, tr}, f, thd, {0.0});
                        // A ray that passes nothing ends at the event that closed it
                        const double L[4] = {(a.size() == 1) ? ub::dB(prod(a, 0)) : -999.0, (b.size() == 1) ? ub::dB(prod(b, 0)) : -999.0,
                                             (c.size() == 2) ? ub::dB(prod(c, 0)) : -999.0, (d.size() == 2) ? ub::dB(prod(d, 0)) : -999.0};
                        const double sp = split(ei, thd, 0);
                        QD_DIAG("%d,%.1f,%.1f,%.0e,%.0f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n", tr, tM, tfC, ei, thd, L[0], L[1], L[2], L[3], L[1] - L[0], sp);
                        INFO("B, type " << tr << ", tfM " << tM << ", tfC " << tfC << ", ei " << ei << ", theta " << thd);
                        for (int j = 0; j < 4; ++j)
                            CHECK(L[j] < ((tM == 0.0) ? -120.0 : 0.0)); // no tf on the layer: its face to air passes nothing
                        if (tM > 0.0)
                            CHECK(std::abs(L[1] - L[0]) < 1.0e-6); // one wall from its two sides
                    }

    // ------------------------------------------------------------------------------------------
    // C. The sheet event alone
    // ------------------------------------------------------------------------------------------
    // On the near face of a host the event is the plate, the smaller absorption of its two faces
    // (its mass at nothing), whatever the host is. On the far face and inside the host it also closes
    // the host leg in front of it, and the two are then one number: one sheet, one leg, one plate.
    QD_DIAG("\n# diag: closed plate C, the sheet event of M rigid tf 0.2 on a host 10 mm: entry (near face), exit (far face), inside (12 mm of\n");
    QD_DIAG("# host behind), series declined, scalar. leak: 10 log10 min(tf, the face from the host). own: what ray_mesh_interact wrote for that event\n");
    QD_DIAG("type,tfM,host,theta,entry,exit,inside,leak,entry-leak,exit-inside,own_entry,own_exit\n");
    {
        const double he[5][2] = {{2.50, 0.80}, {2.50, 1.0e-3}, {2.50, 0.0}, {1.50, 0.40}, {0.31, 0.20}};
        for (int tr : {4, 5})
            for (double tM : {0.2, 0.0})
                for (int H = 0; H < 5; ++H)
                    for (double thd : {10.0, 30.0, 60.0})
                    {
                        std::vector<LumpMtl> q = {rigid(tM, 0.022, 0.001 * BIG), {he[H][0], he[H][1], 0.0}};
                        ub::Pal P = side0(lump_palette(q));
                        const double d1 = 0.010, d2 = 0.012;
                        arma::uvec mi, mx, me;
                        arma::mat m_in = ub::build({0.0, d1, d1 + d2}, {2, 1, 2}, {false, false, true}, mi);
                        arma::mat m_ex = ub::build({0.0, d1, d1 + gap}, {2, 2, 1}, {false, true, false}, mx);
                        arma::mat m_en = ub::build({0.0, gap, gap + d1}, {1, 2, 2}, {false, false, true}, me);
                        auto a = ub::chain(P, m_en, me, {tr}, f, thd, {1.5}), b = ub::chain(P, m_ex, mx, {tr, tr}, f, thd, {1.5}), c = ub::chain(P, m_in, mi, {tr, tr}, f, thd, {1.5});
                        REQUIRE((a.size() == 1 && b.size() == 2 && c.size() == 2));
                        const double en = ub::dB(cpx(a[0].xpr, 0, 0)), ex = ub::dB(cpx(b[1].xpr, 0, 0)), in = ub::dB(cpx(c[1].xpr, 0, 0));
                        const double o_en = ub::dB(cpx(a[0].xpr0, 0, 0)), o_ex = ub::dB(cpx(b[1].xpr0, 0, 0));
                        const double leaks = (tM > 0.0) ? 10.0 * std::log10(std::min(tM, a_face(std::complex<double>(he[H][0], -he[H][1]), tM, 0.0, thd))) : -999.0;
                        QD_DIAG("%d,%.1f,%.2f-%.0ej,%.0f,%.4f,%.4f,%.4f,%.4f,%.2e,%.2e,%.4f,%.4f\n", tr, tM, he[H][0], he[H][1], thd, en, ex, in, leaks, en - leaks, ex - in, o_en, o_ex);
                        INFO("C, type " << tr << ", tfM " << tM << ", host " << he[H][0] << " - " << he[H][1] << "j, theta " << thd);
                        if (tM > 0.0)
                        {
                            // Next to a lossless host the event is the plate of its two faces and a lining and the
                            // sheet inside are one sheet. Next to a lossy one the plate is seen through the pad of a
                            // closed face: what it passes into the host and what it passes out of it are one face read
                            // from its two sides
                            if (he[H][1] == 0.0)
                            {
                                CHECK(std::abs(en - leaks) < 1.0e-9);
                                CHECK(std::abs(ex - in) < 1.0e-6);
                            }
                            CHECK(std::abs((en - leaks) + (ex - in)) < 1.0e-6);
                            CHECK(std::abs(o_en - en) < 1.0e-9);                       // and from ray_mesh_interact alone
                            // the path pays the host leg on top of the plate, up to the flux splitting the exit out of a
                            // lossy host carries (en - leaks, the wave convention: exit above entry)
                            CHECK(ub::dB(cpx(b[0].xpr, 0, 0)) + ex < leaks + std::abs(en - leaks) + 1.0e-9);
                            CHECK(o_ex < 0.0);
                        }
                        else
                        {
                            CHECK(en < -120.0);
                            CHECK(ex < -120.0);
                            CHECK(in < -120.0);
                        }
                    }
    }

    // ------------------------------------------------------------------------------------------
    // D. A rigid layer with a mass-law layer that is not rigid
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed plate D, the stack in air of M rigid 22 mm tf 0.2 and MB eps e 12 mm m 20 tf t, mass arguments at nothing, scalar\n");
    QD_DIAG("epsMB,tfMB,theta,M|MB,MB|M,d\n");
    for (double eB : {1.5, 2.5})
        for (double tB : {0.0, 0.2})
            for (double thd : {10.0, 30.0, 60.0})
            {
                std::vector<LumpMtl> q = {rigid(tfM, 0.022, 0.001 * BIG), {eB, 0.0, 0.012, 0.0, tB, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG}};
                ub::Pal P = side0(lump_palette(q));
                arma::uvec m1, m2;
                arma::mat s1 = ub::build({0.0, gap}, {1, 2}, {false, true}, m1), s2 = ub::build({0.0, gap}, {2, 1}, {false, true}, m2);
                auto a = ub::chain(P, s1, m1, {4}, f, thd, {0.0}), b = ub::chain(P, s2, m2, {4}, f, thd, {0.0});
                REQUIRE((a.size() == 1 && b.size() == 1));
                const double x = ub::dB(prod(a, 0)), y = ub::dB(prod(b, 0));
                QD_DIAG("%.1f,%.1f,%.0f,%.4f,%.4f,%.6f\n", eB, tB, thd, x, y, y - x);
                INFO("D, eps " << eB << ", tf " << tB << ", theta " << thd);
                CHECK(std::abs(y - x) < 1.0e-9); // lossless: one wall, exactly
                CHECK(x < 0.0);
            }

    // ------------------------------------------------------------------------------------------
    // E. Twins that must not move
    // ------------------------------------------------------------------------------------------
    // Printed with every digit, so a diff of this table shows a moved bit where no assertion
    // would: a rigid sheet in air with and without its mass term, two rigid sheets as a stack,
    // a stack whose mass-law layer has a permittivity above one in both orders, and a rigid
    // sheet on a rigid traced wall. The values that have a closed form are asserted on it.
    QD_DIAG("\n# diag: closed plate E, twins, scalar, the complex coefficient with every digit\n");
    QD_DIAG("row,type,theta,re,im\n");
    {
        std::vector<LumpMtl> q = {rigid(0.2, 0.022, 0.001 * BIG),                                  // 1 rigid sheet, mass at nothing
                                  rigid(0.2, 0.022, 0.001),                                        // 2 the same, 26.8 dB
                                  rigid(0.1, 0.012, 0.001),                                        // 3 a second rigid sheet
                                  {2.50, 0.80, tC},                                                // 4 coherent sheet
                                  {2.25, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},   // 5 mass-law, permittivity above one
                                  {1.50, 0.40, 0.006, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},  // 6 the same lossy, with tf
                                  {2.401e-9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}}; // 7 rigid wall, traced
        ub::Pal P = side0(lump_palette(q));
        struct Row
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
            int n;
        };
        const std::vector<Row> rows = {
            {"rigid sheet in air - no mass", {0.0}, {1}, {false}, 1},
            {"rigid sheet in air - 26.8 dB", {0.0}, {2}, {false}, 1},
            {"stack rigid | rigid", {0.0, gap}, {2, 3}, {false, true}, 1},
            {"stack rigid | rigid reversed", {0.0, gap}, {3, 2}, {false, true}, 1},
            {"stack C | ML 2.25", {0.0, gap}, {4, 5}, {false, true}, 1},
            {"stack ML 2.25 | C", {0.0, gap}, {5, 4}, {false, true}, 1},
            {"stack C | ML lossy tf", {0.0, gap}, {4, 6}, {false, true}, 1},
            {"stack ML lossy tf | C", {0.0, gap}, {6, 4}, {false, true}, 1},
            {"rigid sheet on rigid wall - entry", {0.0, gap, gap + 0.1}, {2, 7, 7}, {false, false, true}, 2},
            {"rigid sheet on rigid wall - exit", {0.0, 0.1, 0.1 + gap}, {7, 7, 2}, {false, true, false}, 2},
        };
        for (const auto &R : rows)
            for (int tr : {4, 5})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    arma::uvec mi;
                    arma::mat mesh = ub::build(R.x, R.mat, R.flip, mi);
                    auto a = ub::chain(P, mesh, mi, std::vector<int>((size_t)R.n, tr), f, thd, {1.5});
                    REQUIRE((int)a.size() == R.n);
                    const std::complex<double> z = prod(a, 0);
                    QD_DIAG("%s,%d,%.0f,%.17g,%.17g\n", R.name, tr, thd, std::real(z), std::imag(z));
                    INFO("E, " << R.name << ", type " << tr << ", theta " << thd);
                    CHECK(std::norm(z) < 1.0);
                    if (R.mat.size() == 1 && R.mat[0] == 1)
                        CHECK(std::abs(std::norm(z) - 0.2) < 1.0e-12); // the plate of two faces of tf 0.2, its mass at nothing
                }
        // The order of a stack whose mass-law layer is not closed: the plate's open faces are read as flux
        // transmittances, which next to a lossy medium are not an energy statement; the two orders differ
        // by up to 0.21 dB (md/acoustic.md 7), recorded with that bound
        QD_DIAG("\n# diag: closed plate E, the two orders of a stack whose mass-law layer is not closed, dB\n");
        QD_DIAG("layer,theta,C|ML,ML|C,d\n");
        for (int ml : {5, 6})
            for (double thd : {10.0, 30.0, 60.0})
            {
                arma::uvec m1, m2;
                arma::mat s1 = ub::build({0.0, gap}, {4, ml}, {false, true}, m1), s2 = ub::build({0.0, gap}, {ml, 4}, {false, true}, m2);
                auto a = ub::chain(P, s1, m1, {4}, f, thd, {1.5}), b = ub::chain(P, s2, m2, {4}, f, thd, {1.5});
                INFO("E, order, layer " << ml << ", theta " << thd);
                QD_DIAG("%d,%.0f,%.4f,%.4f,%+.4f\n", ml, thd, ub::dB(prod(a, 0)), ub::dB(prod(b, 0)), ub::dB(prod(b, 0)) - ub::dB(prod(a, 0)));
                CHECK(std::abs(ub::dB(prod(a, 0)) - ub::dB(prod(b, 0))) < 0.21);
            }
    }

    // ------------------------------------------------------------------------------------------
    // F. Recorded
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed plate F1, recorded: a mass-law sheet eps 0.50, 22 mm, tf 0.2, m 20 (mass at nothing) inside 10 mm + 12 mm of host,\n");
    QD_DIAG("# closed from 45 deg; the sheet event, scalar, dB\n");
    QD_DIAG("host,theta,event\n");
    for (double ei : {0.8, 0.0})
        for (double thd : {30.0, 40.0, 44.0, 44.9, 45.1, 46.0, 50.0, 60.0})
        {
            std::vector<LumpMtl> q = {{0.50, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG}, {2.50, ei, 0.0}};
            ub::Pal P = side0(lump_palette(q));
            arma::uvec mi;
            arma::mat mesh = ub::build({0.0, 0.010, 0.022}, {2, 1, 2}, {false, false, true}, mi);
            auto a = ub::chain(P, mesh, mi, {4, 4}, f, thd, {1.5});
            REQUIRE(a.size() == 2);
            QD_DIAG("2.50-%.1fj,%.1f,%.4f\n", ei, thd, ub::dB(cpx(a[1].xpr, 0, 0)));
            CHECK(ub::dB(cpx(a[1].xpr, 0, 0)) < 0.0);
        }
}

TEST_CASE("Stack geometry - an ordered stack leaves the ray on the line it arrived on - the mass-law part", "[acoustic]")
{
    // The part of "Stack geometry - an ordered stack leaves the ray on the line it arrived on" that
    // evaluates a material with the mass column (m > 0); the description is there.
    ub::OneThread one_thread;
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
    auto unit = [](arma::mat d) { d /= std::sqrt(ub::sumsq(d)); return d; };
    auto dmax = [](const arma::mat &a, const arma::mat &b) { double v = 0.0; for (arma::uword i = 0; i < 3; ++i) v = std::max(v, std::abs(a(0, i) - b(0, i))); return v; };
    auto ang = [](const arma::mat &d) { return std::atan2(std::sqrt(d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2)), std::abs(d(0, 0))) * 180.0 / QPI; };
    auto sig = [](const arma::mat &x) { double v = 0.0; for (arma::uword i = 0; i < x.n_elem; ++i) v += 0.731 * (double)(i + 1) * x(i); return v; };
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

    // ------------------------------------------------------------------------------------------
    // A. In air, one event: every pair, every orientation of the two normals, every type
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: stack geometry A, two co-located sheets in air, one event with a tube. seg, own: the segment and ray_mesh_interact's own\n");
    QD_DIAG("# path_dirN, deg from the normal. Largest component of a unit-vector difference: e_seg the segment against the direction\n");
    QD_DIAG("# written, e_in against the incoming direction, e_own the own path_dirN against the incoming direction, e_tube the vertex\n");
    QD_DIAG("# rays against the directions they arrived in. bits: origN, destN and tridirN are those of type 1 or 4, bitwise (recorded)\n");
    QD_DIAG("pair,normals,type,theta,ot,rt,seg,own,e_seg,e_in,e_own,e_tube,bits\n");
    for (auto pr : std::vector<std::pair<int, int>>{{1, 3}, {3, 1}, {4, 1}})
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

    // ------------------------------------------------------------------------------------------
    // B. Inside a host, and where the next wall is met
    // ------------------------------------------------------------------------------------------
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
            for (auto pr : std::vector<std::pair<int, int>>{{1, 3}})
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
                            CHECK(c[2].rt == t[2].rt); // the exit behind the stack is the exit behind a single sheet
                            if (!glued)                // behind a glued entry that entry's anchoring can set the bit, for the twin as well
                                CHECK((c[2].rt & 2) == 0); // behind the stack the wave's direction is the segment again
                            CHECK(std::abs(dyw) < (glued ? 1.0e-6 : TOL)); // the next wall is met where the single sheet meets it
                            CHECK(e_wall < (glued ? 2.0e-4 : TOL));
                            CHECK(std::abs(dL) < 1.0e-3); // and with the level of the undeviated tracer
                        }
}

TEST_CASE("Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more than it received - the mass-law part", "[acoustic]")
{
    // The part of "Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no
    // more than it received" that evaluates a material with the mass column (m > 0); the description
    // is there.
    ub::OneThread one_thread;
    const double f = FRQ, gap = 1.0e-5, d1 = 0.010, d2 = 0.012, ZERO = 1.0e-9;
    const double losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    auto lin = [](double x) { return std::pow(10.0, 0.1 * x); };
    auto lvl = [](const std::vector<ub::Ev> &v, size_t n, int k) { return (v.size() == n) ? ub::dB(cpx(v[n - 1].xpr, 0, k)) : -999.0; };
    auto refl = [](int tr) { return (tr == 4 || tr == 5) ? 3 : 0; };
    // The sheet event inside a host, 10 mm in front and 12 mm behind: both ports, with the host leg in front
    auto inside = [&](const ub::Pal &P, int tr, double thd, int k, double &T, double &R)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, d1, d1 + d2}, {2, 1, 2}, {false, false, true}, mi);
        T = lvl(ub::chain(P, mesh, mi, {tr, tr}, f, thd, {1.5}), 2, k), R = lvl(ub::chain(P, mesh, mi, {tr, refl(tr)}, f, thd, {1.5}), 2, k);
    };


    // ------------------------------------------------------------------------------------------
    // F. Twins that must not move
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: rare sheet F, twins, the complex coefficient of the sheet event with every digit, scalar and EM (VV, HH).\n");
    QD_DIAG("# A rare mass-law sheet inside a lossy host; the rigid mass-law sheet is in sheet tf H\n");
    QD_DIAG("row,type,port,theta,re,im,re_hh,im_hh\n");
    {
        const double BIG = 1.0e12;
        ub::Pal P = lump_palette({{5.24, 0.63, 0.006, 0.0, 0.5},                                             // 1 dense lossy sheet with tf
                                  {5.24, 0.0, 0.006, 0.0, -0.5},                                             // 2 dense lossless sheet, tf below zero
                                  {0.31, 0.0, 0.006},                                                        // 3 rare sheet without tf
                                  {0.31, 0.0, 0.006, 0.0, 0.5},                                              // 4 rare sheet with tf
                                  {0.50, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG},       // 5 rare mass-law sheet
                                  {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001 * BIG},  // 6 rigid mass-law sheet
                                  {2.50, 0.80, 0.0},                                                         // 7 lossy host
                                  {2.50, 0.0, 0.0},                                                          // 8 lossless host
                                  {1.50, 0.40, 0.0, 0.0, 0.5},                                               // 9 host with tf
                                  {2.401e-9, 0.0, 0.0, 0.0, 0.3},                                            // 10 rigid wall
                                  {0.31, 0.20, 0.006, 0.0, 0.5}});                                           // 11 porous sheet with tf
        struct Row
        {
            const char *name;
            int sheet, host; // host 0: the sheet alone in air; host < 0: the sheet glued on the near face of -host
            std::vector<double> th;
        };
        const std::vector<Row> rows = {
            {"rare mass-law tf 0.2 inside lossy host", 5, 7, {20.0, 45.0, 60.0}},
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
                    }
                }
    }
}

TEST_CASE("Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the characteristic matrix - the mass-law part", "[acoustic]")
{
    // The part of "Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the
    // characteristic matrix" that evaluates a material with the mass column (m > 0); the description
    // is there.
    ub::OneThread one_thread;
    const double f = FRQ, gap = 1.0e-5, d1 = 0.010, d2 = 0.012, k0 = 2.0 * QPI * f / C0, TOL = 1.0e-9;
    const double eS = 0.31, tS = 0.006, losses[5] = {0.8, 0.1, 1.0e-3, 1.0e-6, 0.0};
    const std::complex<double> U(1.0, 0.0), cS(eS, 0.0), cRef(3.0, 0.0);
    auto val = [](const std::vector<ub::Ev> &v, size_t n, int k) { return (v.size() == n) ? cpx(v[n - 1].xpr, 0, k) : std::complex<double>(0.0, 0.0); };
    auto dBc = [](std::complex<double> z) { return (std::abs(z) > 0.0) ? 20.0 * std::log10(std::abs(z)) : -999.0; };
    auto refl = [](int tr) { return (tr == 4 || tr == 5) ? 3 : 0; };
    auto rel = [](std::complex<double> a, std::complex<double> b) { return std::abs(a - b) / std::abs(b); };
    // Sheet material 1 and a reference sheet 3, 1 mm of eps 3.0 that is never rare; host 2
    auto pal_of = [&](double ei, double tf) { return lump_palette({{eS, 0.0, tS, 0.0, tf}, {2.50, ei, 0.0}, {3.0, 0.0, 0.001}}); };
    auto host_of = [&](double ei) { return std::complex<double>(2.50, -ei); };
    auto ct_of = [&](double ei, double thd) { double s = std::sin(thd * QPI / 180.0); return std::sqrt(1.0 - s * s / host_of(ei)); };
    auto legdB = [&](double ei, double thd) { return 20.0 * std::log10(std::abs(std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(host_of(ei)) * ct_of(ei, thd) * d1))); };
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
    auto pick = [](const DiagSheet &W, int k, bool port_r) { return port_r ? (k ? W.R_tm : W.R_te) : (k ? W.T_tm : W.T_te); };


    // ------------------------------------------------------------------------------------------
    // G. A rigid sheet of the acoustic mapping
    // ------------------------------------------------------------------------------------------
    // An acoustic material, scalar types. Its permittivity is an impedance mapping, and the mass
    // column is what says so: with it the sheet is a closed mass-law layer and without tf passes
    // nothing. Without the column it is a wave medium like any other and tunnels, on the matrix.
    QD_DIAG("\n# diag: flat branch G, a rigid sheet 22 mm eps 2.1609e-8 without tf, scalar, in air, inside 100 mm + 100 mm of eps 2.50 and as a\n");
    QD_DIAG("# lining of it. |T_m|: with the mass column (m 20)\n");
    QD_DIAG("where,type,theta,|T_m|\n");
    for (int where = 0; where < 3; ++where)
        for (int tr : {4, 5})
            for (double thd : {10.0, 30.0, 60.0})
            {
                ub::Pal Pm = lump_palette({{2.1609e-8, 0.0, 0.022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, {2.50, 0.0, 0.0}, {3.0, 0.0, 0.001}});
                std::complex<double> Tm, Rm;
                if (where == 0)
                    in_air(Pm, 1, tr, thd, 0, Tm, Rm);
                else if (where == 1)
                    inside(Pm, 1, tr, thd, 0, Tm, Rm, 0.1, 0.1);
                else
                    entry(Pm, 1, tr, thd, 0, Tm, Rm);
                QD_DIAG("%s,%d,%.0f,%.3g\n", where == 0 ? "air" : (where == 1 ? "inside" : "lining"), tr, thd, std::abs(Tm));
                INFO("G, where " << where << ", type " << tr << ", theta " << thd);
                CHECK(std::abs(Tm) < 1.0e-12); // with the mass column: closed
            }
}

TEST_CASE("Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the invariant - the mass-law part", "[acoustic]")
{
    // The part of "Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on
    // the invariant" that evaluates a material with the mass column (m > 0); the description is there.
    ub::OneThread one_thread;
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5, k0 = 2.0 * QPI * f / C0, ONE = 1.0 + 1.0e-6, TOL = 1.0e-9;
    const cd U(1.0, 0.0);
    auto val = [](const std::vector<ub::Ev> &v, size_t n, int k) { return (v.size() == n) ? cpx(v[n - 1].xpr, 0, k) : cd(0.0, 0.0); };
    auto refl = [](int tr) { return (tr == 4 || tr == 5) ? 3 : 0; };
    auto dBp = [](double p) { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    enum { AIR, INSIDE, ENTRY, EXIT };
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
    auto kz_of = [](cd e, double p) { cd v = std::sqrt(e - p); return (std::imag(v) > 0.0 || (std::imag(v) == 0.0 && std::real(v) < 0.0)) ? -v : v; };
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
        auto adm = [&](cd e) { return tm ? e / kz_of(e, p) : kz_of(e, p); };
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


    // ------------------------------------------------------------------------------------------
    // D. No step where a sheet becomes rarer than the invariant
    // ------------------------------------------------------------------------------------------
    // The transmit port in dB. The reflect port sits at -18 dB to -52 dB on these rows, with a
    // slope of its own of up to 0.016 dB per 0.02 deg and a kink at the critical angle of a
    // lossless sheet, so it is asserted in amplitude against the same pair of angles with no tf
    // anywhere, which is the matrix and has that slope and that kink.
    QD_DIAG("\n# diag: sheet tf D, neighbours about the angle at which a sheet becomes rarer than the invariant, |eps| = sin^2: 37.40 deg for\n");
    QD_DIAG("# eps 0.31 - 0.20j, 73.78 deg for 0.90 - 0.20j, and the critical angle 33.83 deg of the lossless eps 0.31. T, R: dB at the first\n");
    QD_DIAG("# angle; dT: dB between the two; dR: the same in amplitude; dR_0: dR with no tf anywhere\n");
    QD_DIAG("row,ch,theta_a,theta_b,T,R,dT,dR,dR_0\n");
    {
        const LumpMtl rigid = {2.401e-9, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9};
        struct St
        {
            const char *name;
            int cfg;
            LumpMtl S, H;
            double ta, tb, leg;
            bool leak; // under a closed outer pair
        };
        const std::vector<St> rows = {
            {"porous tf 0.2 on a rigid wall with tf 0.3", ENTRY, {0.31, 0.20, 0.006, 0.0, 0.2}, rigid, 37.39, 37.41, 0.100, true},
            {"0.90-0.20j tf 0.9 on a rigid wall with tf 0.3", ENTRY, {0.90, 0.20, 0.006, 0.0, 0.9}, rigid, 73.77, 73.79, 0.100, true},
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
                    CHECK(std::abs(dT) < 0.01);         // no step on the transmit port
                    if (W.S.ei > 0.0)                       // the critical angle of a lossless sheet is a kink of its own, with tf and without: recorded
                        CHECK(std::abs(dR - dR0) < 1.0e-3); // nor on the reflect port
                    if (!W.leak)
                        CHECK(std::norm(Ta) + std::norm(Ra) < ONE);
                }
    }

    // ------------------------------------------------------------------------------------------
    // H. Twins
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: sheet tf H, twins, the complex coefficient of the sheet event with every digit, scalar and EM (VV, HH): sheets with tf\n");
    QD_DIAG("# a rigid mass-law sheet with tf, inside a lossy host and in air\n");
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
            {"mass-law - rigid tf 0.2 inside lossy host", 9, 10, {20.0, 45.0, 60.0}},
            {"mass-law - rigid tf 0.2 in air", 9, 0, {20.0, 45.0, 60.0}},
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
                    }
    }
}

TEST_CASE("Bare tf - a bare face with tf is a transformer where it is open and a flat leak where it is closed - the mass-law part", "[acoustic]")
{
    // The part of "Bare tf - a bare face with tf is a transformer where it is open and a flat leak
    // where it is closed" that evaluates a material with the mass column (m > 0); the description is
    // there.
    ub::OneThread one_thread;
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5;
    auto dB = [](double p) { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    auto mk = [](double a, double ei, double thk, double tf, double m = 0.0, double fRef = 1.0e9) {
        LumpMtl L;
        L.a = a, L.ei = ei, L.thk = thk, L.tf = tf, L.m = m, L.fRef = fRef;
        return L;
    };
    auto ports = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int tr, double thd, bool follow, int maxdepth = 400) {
        bf::Acc A;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        bf::walk(P, mesh, mi, tr, f, thd, follow ? 1.5 : 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, maxdepth, A);
        return A;
    };
    auto event = [&](const ub::Pal &P, const arma::mat &mesh, const arma::uvec &mi, int it, double thd, int k) -> cd {
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
    const std::vector<Mt> mats = {{"5.24-0.63j", 5.24, 0.63}, {"2.50-0.80j", 2.50, 0.80}, {"1.50-0.40j", 1.50, 0.40}, {"0.90-0.20j", 0.90, 0.20},
                                  {"0.31-0.20j", 0.31, 0.20}, {"5.24", 5.24, 0.0},        {"2.50", 2.50, 0.0}};


    QD_DIAG("\n# diag: bare tf E, the acoustic main road: a rigid wall 200 mm, eps 2.401e-9, with tf, with the mass column (m 30),\n");
    QD_DIAG("# scalar, tracer 4 and 5. R: reflectance of the first face; T: what enters; both with every digit\n");
    QD_DIAG("m,tf,type,theta,R,T,R+tf-1,T-tf\n");
    for (double m : {30.0})
        for (double tf : {0.1, 0.3, 0.9})
            for (int tr : {4, 5})
                for (double thd : {10.0, 45.0, 75.0, 85.0})
                {
                    ub::Pal P = lump_palette({mk(2.401e-9, 0.0, 0.0, tf, m, 1.0e-3)});
                    arma::uvec mi;
                    arma::mat mesh = ub::build({0.0, 0.2}, {1, 1}, {false, true}, mi);
                    const double R = std::norm(event(P, mesh, mi, 3, thd, 0)), T = std::norm(event(P, mesh, mi, tr, thd, 0));
                    QD_DIAG("%.0f,%.1f,%d,%.0f,%.17g,%.17g,%.2e,%.2e\n", m, tf, tr, thd, R, T, R + tf - 1.0, T - tf);
                    INFO("rigid wall, m " << m << ", tf " << tf << ", type " << tr << ", theta " << thd);
                    CHECK(std::abs(R + tf - 1.0) < 1.0e-12); // closed: one minus its leak
                    CHECK(std::abs(T - tf) < 1.0e-12);       // and the leak, flat
                }
}

TEST_CASE("Closed face - tf at a closed face is a pad and the two ports of a sheet on it add up", "[acoustic]")
{
    ub::OneThread one_thread;
    // A closed face has a wave on one side only: behind it nothing propagates, a rigid wall of the
    // acoustic mapping from 0.003 deg on, air behind a coated solid met from inside past the
    // critical angle. tf says that a share leaks through. Next to a lossless medium that carries
    // a wave that is one statement, a share of the forward wave's flux. Next to a lossy medium,
    // or a sheet the wave is evanescent in, the flux towards the face is carried by the forward
    // and the reflected wave together, and a share of the forward wave alone is not what the
    // reflection leaves: the two ports of a sheet on a rigid wall added to more than they
    // received, a lossless sheet rarer than the invariant was entered as a leak and charged no
    // decay, and the same sheet with a loss of 1e-6 sat 55 dB below it.
    //
    // The rule asserted here: the leak of a closed face is a matched resistive pad in front of
    // it, formed on the modulus of the admittance of the side the wave is on, as the transformer
    // of an open face forms its ratio on a modulus. The face passes what it absorbs. In front of
    // a lossless medium that carries a wave it is sqrt(R') on the Fresnel reflection and 1 - R'
    // of the flux, R' the pair rule on a reflectance of one; at a critical angle it is the limit
    // of the transformer. One rule at the far face of a coherent sheet, at the bare face of a
    // traced solid, at the mirror of a summed series and at the faces of a closed mass-law layer.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update.
    // The rigid wall is an acoustic material with the mass column, and types 1 and 0 on it are
    // not EM. The rows of a coating met from inside a dense solid, an EM material, are in "Closed
    // face - a coating met from inside a dense solid".
    //
    // A. Both ports of a sheet on a rigid wall: at most one together, and on the reference of the
    //    rule
    // C. No step where a rare sheet loses its loss, 1 to 200 mm
    // D. No step at the critical angle of the sheet
    // E. The sheet against its traced solid, series summed and followed
    // F. A closed plate with a lossy layer: one wall from its two sides, as a stack and as a laminate
    // G. The end points: no tf, tf = -1, and the plate with the mass column
    // H. Twins with every digit
    typedef std::complex<double> cd;
    const double f = FRQ, gap = 1.0e-5, k0 = 2.0 * QPI * f / C0, ONE = 1.0 + 1.0e-6, eW = 2.401e-9;
    const cd J(0.0, 1.0);
    auto refl = [](int tr) { return (tr > 2) ? 3 : 0; };
    auto dBp = [](double p) { return (p > 0.0) ? 10.0 * std::log10(p) : -999.0; };
    auto launched = [](int cur, double thd) {
        ub::Ev S;
        const double th = thd * QPI / 180.0;
        S.prev = arma::Col<short>{0}, S.cur = arma::Col<short>{(short)cur}, S.buf = arma::Col<short>{0};
        S.path_dirN = arma::mat{{std::cos(th), std::sin(th), 0.0}};
        S.accg = arma::vec{0.1}, S.accr = arma::vec{0.1};
        return S;
    };
    enum { WALL, COAT };
    // Both ports of the sheet event. Material 1 is the sheet, 2 the wall behind it (WALL) or the host the ray is launched in (COAT)
    auto ports = [&](const ub::Pal &P, int cfg, int tr, double thd, int k, cd &T, cd &R) {
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
    auto kz_of = [](cd e, double p) { cd v = std::sqrt(e - p); return (std::imag(v) > 0.0 || (std::imag(v) == 0.0 && std::real(v) < 0.0)) ? -v : v; };
    auto pad_ref = [&](cd en, cd es, cd ef, double tf_n, double tf_s, double tf_f, double d, double p, bool tm, double &Tp, double &Rp) -> bool {
        auto adm = [&](cd e) { return tm ? e / kz_of(e, p) : kz_of(e, p); };
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

    // ------------------------------------------------------------------------------------------
    // A. Both ports of one sheet event under a closed outer pair
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face A, both ports of a sheet event under a closed outer pair, the worst over 1, 6 and 25 mm and seven angles:\n");
    QD_DIAG("# 5 to 85 deg on a rigid wall. tf: of the sheet. sum: T + R in\n");
    QD_DIAG("# power, the largest, at theta / mm; above: rows above one / rows. dT, dR: the largest distance of a power from the reference of the\n");
    QD_DIAG("# rule. ch: sc scalar, TE (VV), TM (HH)\n");
    QD_DIAG("where,sheet,tf,ch,sum,theta,mm,above,rows,dT,dR\n");
    for (int w = 0; w < 2; ++w)
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

    // ------------------------------------------------------------------------------------------
    // C. A rare sheet losing its loss
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face C, S eps 0.31 - j es with tf 0.5 on a rigid wall with tf tW, the glued entry, scalar, 60 deg, both ports in dB.\n");
    QD_DIAG("# step: the row over the row with es 1e-6, dB, transmit and reflect\n");
    QD_DIAG("tW,mm,es,T,R,sum,step_T,step_R\n");
    for (double tW : {0.3, 0.0})
        for (double thk : {0.001, 0.010, 0.100, 0.200})
        {
            double T6 = 0.0, R6 = 0.0;
            for (double es : {1.0e-6, 0.2, 1.0e-2, 1.0e-3, 0.0})
            {
                ub::Pal P = lump_palette({{0.31, es, thk, 0.0, 0.5}, {eW, 0.0, 0.0, 0.0, tW, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}});
                cd T, R;
                ports(P, WALL, 4, 60.0, 0, T, R);
                if (es == 1.0e-6)
                    T6 = std::norm(T), R6 = std::norm(R);
                QD_DIAG("%.1f,%.0f,%.0e,%.4f,%.4f,%.6f,%+.4f,%+.4f\n", tW, thk * 1e3, es, dBp(std::norm(T)), dBp(std::norm(R)), std::norm(T) + std::norm(R), dBp(std::norm(T)) - dBp(T6),
                        dBp(std::norm(R)) - dBp(R6));
                INFO("C, tW " << tW << ", " << thk * 1e3 << " mm, es " << es);
                if (es == 0.0)
                {
                    CHECK(std::abs(dBp(std::norm(T)) - dBp(T6)) < 1.0e-3);
                    CHECK(std::abs(dBp(std::norm(R)) - dBp(R6)) < 1.0e-3);
                }
            }
        }

    // ------------------------------------------------------------------------------------------
    // D. No step at a critical angle
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face D, neighbours 0.02 deg apart about a critical angle, both ports in dB and the step between them. sheet: 10 mm\n");
    QD_DIAG("# eps 0.31 with tf 0.5 on a rigid wall with tf 0.3 about the critical angle of the sheet, 33.83 deg, scalar\n");
    QD_DIAG("case,ch,theta_lo,theta_hi,T_lo,T_hi,R_lo,R_hi,step_T,step_R\n");
    {
        const double tcS = std::asin(std::sqrt(0.31)) * 180.0 / QPI;
        auto row = [&](const char *name, const ub::Pal &P, int cfg, int tr, int k, double tc, double bT, double bR) {
            cd T0, R0, T1, R1;
            ports(P, cfg, tr, tc - 0.01, k, T0, R0), ports(P, cfg, tr, tc + 0.01, k, T1, R1);
            const double sT = dBp(std::norm(T1)) - dBp(std::norm(T0)), sR = dBp(std::norm(R1)) - dBp(std::norm(R0));
            QD_DIAG("%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%+.4f,%+.4f\n", name, (tr == 4) ? "sc" : ((k == 0) ? "TE" : "TM"), tc - 0.01, tc + 0.01, dBp(std::norm(T0)), dBp(std::norm(T1)),
                    dBp(std::norm(R0)), dBp(std::norm(R1)), sT, sR);
            INFO("D, " << name << ", type " << tr << ", k " << k);
            CHECK(std::abs(sT) < bT);
            CHECK(std::abs(sR) < bR);
        };
        row("sheet 0.31", lump_palette({{0.31, 0.0, 0.010, 0.0, 0.5}, {eW, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}}), WALL, 4, 0, tcS, 0.05, 0.5);
    }

    // ------------------------------------------------------------------------------------------
    // E. The sheet against its traced solid
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face E, a lossy layer with tf on a rigid wall with tf 0.3 as a sheet and as a traced solid, every path up to the wall.\n");
    QD_DIAG("# T, R: the sheet's ports, dB. dT, dR: the solid over the sheet with its series summed, dB; fT, fR: with every bounce followed.\n");
    QD_DIAG("# S_sum, S_fol: T + R of the solid. k: 0 scalar or VV, 3 HH\n");
    QD_DIAG("layer,tf,mm,type,k,theta,T,R,dT,dR,fT,fR,S_sum,S_fol\n");
    for (const Sh &S : std::vector<Sh>{{0.31, 0.20}, {0.90, 0.20}, {1.50, 0.40}, {5.24, 0.63}})
        for (double tS : {0.5, -0.5, 0.0})
            for (double thk : {0.006, 0.025})
                for (int tr : {4, 1})
                    for (double thd : {10.0, 45.0, 60.0, 85.0})
                    {
                        ub::Pal P = lump_palette({{S.a, S.ei, 0.0, 0.0, tS}, {eW, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, {S.a, S.ei, thk, 0.0, tS}});
                        arma::uvec ms, mb;
                        arma::mat mesh_s = ub::build({0.0, gap, gap + 0.1}, {3, 2, 2}, {false, false, true}, ms);
                        arma::mat mesh_b = ub::build({0.0, thk, thk + gap, thk + gap + 0.1}, {1, 1, 2, 2}, {false, true, false, true}, mb), o, d;
                        diag_ray_at(thd, o, d);
                        const double kt = k0 * std::sin(thd * QPI / 180.0);
                        cf::Acc As, A0, A1;
                        cf::walk(P, mesh_s, ms, tr, f, kt, 0.0, 2, nullptr, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 400, As);
                        cf::walk(P, mesh_b, mb, tr, f, kt, 0.0, 2, nullptr, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 400, A0);
                        cf::walk(P, mesh_b, mb, tr, f, kt, 1.5, 2, nullptr, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 400, A1);
                        for (int ch = 0; ch < ((tr > 2) ? 1 : 2); ++ch)
                        {
                            const double Ts = std::norm(As.T[ch]), Rs = std::norm(As.R[ch]), T0 = std::norm(A0.T[ch]), R0 = std::norm(A0.R[ch]), T1 = std::norm(A1.T[ch]), R1 = std::norm(A1.R[ch]);
                            QD_DIAG("%.2f-%.2fj,%.1f,%.0f,%d,%d,%.0f,%.4f,%.4f,%+.4f,%+.4f,%+.4f,%+.4f,%.6f,%.6f\n", S.a, S.ei, tS, thk * 1e3, tr, ch ? 3 : 0, thd, dBp(Ts), dBp(Rs),
                                    dBp(T0) - dBp(Ts), dBp(R0) - dBp(Rs), dBp(T1) - dBp(Ts), dBp(R1) - dBp(Rs), T0 + R0, T1 + R1);
                            INFO("E, layer " << S.a << " - " << S.ei << "j, tf " << tS << ", " << thk * 1e3 << " mm, type " << tr << ", k " << (ch ? 3 : 0) << ", theta " << thd);
                            CHECK(T0 + R0 < ONE);
                            CHECK(T1 + R1 < ONE);
                            CHECK(std::abs(dBp(T0) - dBp(Ts)) < 2.0e-2);
                            CHECK(std::abs(dBp(R0) - dBp(Rs)) < 2.0e-2);
                            CHECK(std::abs(dBp(T1) - dBp(Ts)) < 2.0e-2);
                            CHECK(std::abs(dBp(R1) - dBp(Rs)) < 2.0e-2);
                        }
                    }

    // ------------------------------------------------------------------------------------------
    // F. A closed plate with a lossy layer
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face F, C 10 mm with M rigid 22 mm tf 0.2 m 20, mass argument at nothing. st: the stack of the two as sheets;\n");
    QD_DIAG("# tr: C a traced solid with M as a lining, series summed; CM: C met first. Transmit port, dB. d_st, d_tr: M side over C side;\n");
    QD_DIAG("# d_CM, d_MC: stack over laminate. k: 0 scalar or VV, 3 HH\n");
    QD_DIAG("C,type,k,theta,stCM,stMC,trC,trM,d_st,d_tr,d_CM,d_MC\n");
    for (const Sh &S : std::vector<Sh>{{2.50, 0.80}, {2.50, 0.0}, {0.31, 0.20}})
        for (int tr : {4, 1})
            for (int k : {0, 3})
                for (double thd : {10.0, 30.0, 60.0})
                {
                    if (tr == 4 && k == 3)
                        continue;
                    const double tC = 0.010, c = std::cos(thd * QPI / 180.0);
                    ub::Pal P = lump_palette({{S.a, S.ei, tC}, {S.a, S.ei}, {2.1609e-8, 0.0, 0.022, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}});
                    arma::uvec m1, m2, m3, m4;
                    arma::mat sCM = ub::build({0.0, gap}, {1, 3}, {false, true}, m1), sMC = ub::build({0.0, gap}, {3, 1}, {false, true}, m2);
                    arma::mat tCm = ub::build({0.0, tC, tC + gap}, {2, 2, 3}, {false, true, false}, m3), tMc = ub::build({0.0, gap, gap + tC}, {3, 2, 2}, {false, false, true}, m4);
                    auto a = ub::chain(P, sCM, m1, {tr}, f, thd, {0.0}), b = ub::chain(P, sMC, m2, {tr}, f, thd, {0.0});
                    auto cc = ub::chain(P, tCm, m3, {tr, tr}, f, thd, {0.0}), dd = ub::chain(P, tMc, m4, {tr, tr}, f, thd, {0.0});
                    REQUIRE((a.size() == 1 && b.size() == 1 && cc.size() == 2 && dd.size() == 2));
                    const double stCM = dBp(std::norm(cpx(a[0].xpr, 0, k))), stMC = dBp(std::norm(cpx(b[0].xpr, 0, k)));
                    const double trC = dBp(std::norm(cpx(cc[0].xpr, 0, k) * cpx(cc[1].xpr, 0, k))), trM = dBp(std::norm(cpx(dd[0].xpr, 0, k) * cpx(dd[1].xpr, 0, k)));
                    (void)c;
                    QD_DIAG("%.2f-%.2fj,%d,%d,%.0f,%.4f,%.4f,%.4f,%.4f,%+.6f,%+.6f,%+.6f,%+.6f\n", S.a, S.ei, tr, k, thd, stCM, stMC, trC, trM, stMC - stCM, trM - trC, stCM - trC, stMC - trM);
                    INFO("F, C " << S.a << " - " << S.ei << "j, type " << tr << ", k " << k << ", theta " << thd);
                    CHECK(std::abs(stMC - stCM) < 1.0e-6); // one wall from its two sides
                    CHECK(std::abs(trM - trC) < 1.0e-3);
                    CHECK(std::abs(stCM - trC) < 5.0e-3); // one wall as a stack and as a laminate
                    CHECK(std::abs(stMC - trM) < 5.0e-3);
                }

    // ------------------------------------------------------------------------------------------
    // G. The end points
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face G, the end points on a rigid wall, scalar, 60 deg, powers. none: no tf on sheet or wall. closed: tf -1 on the\n");
    QD_DIAG("# sheet, wall tf 0.3. plate: the lossless eps 0.31 sheet with tf 0.5 and the mass column on a wall with tf 0.3, its first face\n");
    QD_DIAG("case,sheet,mm,T,R\n");
    double plate_T = -1.0;
    for (double thk : {0.001, 0.010, 0.100})
    {
        for (const Sh &S : std::vector<Sh>{{0.31, 0.20}, {0.31, 0.0}, {5.24, 0.63}})
            for (int cs = 0; cs < 2; ++cs)
            {
                ub::Pal P = lump_palette({{S.a, S.ei, thk, 0.0, cs ? -1.0 : 0.0}, {eW, 0.0, 0.0, 0.0, cs ? 0.3 : 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}});
                cd T, R;
                ports(P, WALL, 4, 60.0, 0, T, R);
                QD_DIAG("%s,%.2f-%.2fj,%.0f,%.9f,%.9f\n", cs ? "closed" : "none", S.a, S.ei, thk * 1e3, std::norm(T), std::norm(R));
                INFO("G, case " << cs << ", sheet " << S.a << " - " << S.ei << "j, " << thk * 1e3 << " mm");
                CHECK(std::norm(T) == 0.0);
                CHECK(std::norm(R) < ONE);
            }
        ub::Pal P = lump_palette({{0.31, 0.0, thk, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, {eW, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}});
        cd T, R;
        ports(P, WALL, 4, 60.0, 0, T, R);
        QD_DIAG("plate,0.31-0.00j,%.0f,%.9f,%.9f\n", thk * 1e3, std::norm(T), std::norm(R));
        INFO("G, plate, " << thk * 1e3 << " mm");
        if (plate_T < 0.0)
            plate_T = std::norm(T);
        CHECK(std::abs(std::norm(T) - plate_T) < 1.0e-12); // at every thickness
        CHECK(std::abs(std::norm(T) - 0.5) < 1.0e-12);     // the plate's first face, what it absorbs; the rest at the wall's far face
        CHECK(std::abs(std::norm(R) - 0.5) < 1.0e-12);     // and its near face, 1 - tf
    }

    // ------------------------------------------------------------------------------------------
    // H. Twins with every digit
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: closed face H, twins, the complex coefficient of the sheet event with every digit, 6 mm, 45 deg: a lossless sheet denser\n");
    QD_DIAG("# than the invariant with tf on a rigid wall with tf, a sheet without tf on a wall without tf, a mass-law sheet and a\n");
    QD_DIAG("# rigid mass-law sheet on the wall with tf. port: T or R. k: 0 scalar or VV, 3 HH\n");
    QD_DIAG("twin,type,k,port,re,im\n");
    {
        struct Tw
        {
            const char *name;
            int cfg;
            LumpMtl s, w;
        };
        const LumpMtl wall = {eW, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, wall0 = {eW, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, host = {5.24, 0.0};
        const std::vector<Tw> tw = {{"dense 5.24 tf 0.5 on wall tf 0.3", WALL, {5.24, 0.0, 0.006, 0.0, 0.5}, wall},
                                    {"lossy 0.31-0.20j no tf on wall no tf", WALL, {0.31, 0.20, 0.006}, wall0},
                                    {"mass-law 2.50 tf 0.2 on wall tf 0.3", WALL, {2.50, 0.0, 0.006, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, wall},
                                    {"rigid mass-law tf 0.2 on wall tf 0.3", WALL, {2.1609e-8, 0.0, 0.006, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e9}, wall}};
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
                    INFO("H, " << W.name << ", type " << tr << ", k " << k);
                    CHECK(std::norm(T) + std::norm(R) < ONE);
                }
    }
}

// ================================================================================================
// Round 1 decisions (md/acoustic.md 10, 2026-10-15): the plate rule R1h (N1), M2 (N2), res* as the
// plate's resonance (N3), the bending stiffness in X0 (N4), the two-parameter mapping (N7 / A12),
// the held exit (A2, A7), alpha over the held path (A3), the absorption peak tfP / tfF / tfQ (A10).
// Every reference is written from the column definitions of md/acoustic.md, not from library code.
// ================================================================================================
namespace r1
{
    static const double FMAP = 299792458.0 / 343.0 * 1.0e-9; // GHz per Hz, the acoustic mapping
    static double dB(double g) { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; }
    // The single-body mass term of md/acoustic.md 2.1 written in X0: zero below one, 20 log10 X0 at
    // normal incidence, log10(1 + (X0 c)^2) normalized in angle
    static double mass_TL(double X0, double c)
    {
        if (!(X0 > 1.0) || !(c > 0.0))
            return 0.0;
        double A = 20.0 * std::log10(X0);
        return (c >= 1.0) ? A : A * std::log10(1.0 + X0 * c * X0 * c) / std::log10(1.0 + X0 * X0);
    }
    // R1h: what a plate whose outer faces absorb at least a and whose bodies have the term tau passes
    static double plate(double a, double tau) { return (a > 0.0 && tau > 0.0) ? a * tau / (a + tau - a * tau) : 0.0; }
    // The plate's resonance (N3) and its bending stiffness (N4) as factors on X0; f, F in GHz, c the cosine in air
    static double res_fac(double f, double c, double F, double Q, double S)
    {
        if (!(F > 0.0 && Q > 0.0) || S == 0.0)
            return 1.0;
        if (S > 0.0)
        {
            double x = f * c / F;
            return std::min(1.0, std::abs(std::complex<double>(1.0 - x * x, x / Q)));
        }
        double y = F / f;
        return std::abs(std::complex<double>(1.0 - y * y, -y / Q));
    }
    static double coi_fac(double f, double c, double cF, double cQ)
    {
        if (!(cF > 0.0 && cQ > 0.0))
            return 1.0;
        double s2 = 1.0 - c * c, w = (f / cF) * (f / cF) * s2 * s2;
        return std::abs(std::complex<double>(1.0 - w, -w / cQ));
    }
    struct Out
    {
        double T = 0.0, R = 0.0;
        int nT = 0;
    };
    // One ray through a wall of faces (x, material, flip), every path through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update (bf::walk), eps 0 sums every series, 1.5 follows every bounce
    static Out wall(const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, int tr, double f, double thd,
                    double eps = 0.0)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build(x, mat, flip, mi), o, d;
        diag_ray_at(thd, o, d);
        bf::Acc A;
        bf::walk(P, mesh, mi, tr, f, thd, eps, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 200, A);
        return {std::norm(A.T[0]), std::norm(A.R[0]), A.nT};
    }
    // The same with a half-space behind: a ray that has entered the sink medium is collected into T
    static Out sink(const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, int tr, double f, double thd,
                    int snk)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build(x, mat, flip, mi), o, d;
        diag_ray_at(thd, o, d);
        cf::Acc A;
        const double kt = 2.0 * QPI * f / C0 * std::sin(thd * QPI / 180.0);
        cf::walk(P, mesh, mi, tr, f, kt, 0.0, snk, nullptr, true, o, d, 1.0, 1.0, 0.0, 0.0, 0, 200, A);
        return {std::norm(A.T[0]), std::norm(A.R[0]), A.nT};
    }
    // Set a column of every material at once
    static void col(ub::Pal &P, const char *k, const std::vector<double> &v) { P[k] = v; }
    // Davy's forced radiation efficiency of a finite panel, azimuth mean, written from the paper
    // [Davy_2009_ForcedRadiationEfficiency, Eqs. 19, 28, 34, 36, 37; n = 2, w = 1.3, beta = 0.124], for a square of
    // side s (2a = s, S = s^2) at f_hz in air (k = 2 pi f / 343) and the cosine g of the incidence
    static double davy_sigma(double f_hz, double g, double s)
    {
        const double k = 2.0 * QPI * f_hz / 343.0, ka = 0.5 * k * s, S = s * s;
        const double p = std::min(1.0, 1.3 * std::sqrt(QPI / (2.0 * ka)));             // Eq. 19
        const double h = 1.0 / (2.0 / 3.0 * std::sqrt(2.0 * ka / QPI) - 0.124);         // Eq. 28
        const double al = h / p - 1.0, q = 2.0 * QPI / (k * k * S);                      // Eqs. 34, 36
        const double x = (g >= p) ? g : h - al * g;                                      // Eq. 37
        return 1.0 / std::sqrt(x * x + q * q);
    }
    // The mass term of a panel of side s (md/grazing_report.md 0.7, decision 2026-10-19): the radiation cosine of
    // mass_TL replaced by 1 / sigma and the finite plate's gain 1 / (sigma c), the transmission clipped at one;
    // s = 0 is the infinite plate
    static double mass_TL_side(double X0, double c, double f_hz, double s)
    {
        if (!(s > 0.0))
            return mass_TL(X0, c);
        if (!(X0 > 1.0) || !(c > 0.0))
            return 0.0;
        const double sg = davy_sigma(f_hz, std::min(c, 1.0), s), x = X0 / sg;
        const double TL = 20.0 * std::log10(X0) * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0) + 10.0 * std::log10(sg * std::min(c, 1.0));
        return std::max(0.0, TL);
    }
    // r1::wall with a ray that starts close to the wall and whose path through it is centred on y = 0.37, so that
    // a ray within a tenth of a degree of grazing still crosses a traced solid on the 400 m faces. The first segment
    // reaches past the last face: one that ends inside a body preloads the rest of the leg, which the walker's
    // relaunch would add a second time (87 m through 152 mm at 89.9 deg read as 137 m, md/precision_report.md 0.2)
    static Out wall_g(const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, int tr, double f, double thd)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build(x, mat, flip, mi);
        const double t = std::tan(thd * QPI / 180.0), dl = std::max(std::min(1.0, 50.0 / std::max(t, 1.0e-12)), x.back() - x.front() + 0.05);
        const double y0 = 0.37 - 0.5 * (x.back() - x.front()) * t;
        arma::mat o = {{-dl, y0 - dl * t, 0.0}}, d = {{dl, y0 + dl * t, 0.0}};
        bf::Acc A;
        bf::walk(P, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 200, A);
        return {std::norm(A.T[0]), std::norm(A.R[0]), A.nT};
    }
}

TEST_CASE("Plate rule - a rigid mass-law panel passes the incoherent series of its face and its plate and reflects what its face does not absorb", "[acoustic]")
{
    // N1 (R1h): a closed mass-law plate in air reflects 1 - a and passes a tau / (a + tau - a tau), a the
    // absorption of its outer faces (tf of the material, the pad), tau its mass term. As a sheet and as a
    // traced solid, both tracers, in the one- and the two-parameter mapping (N7). At bb336dc the plate
    // passed a^2 tau, the two leaks. The tables carry no side column: tau is the term of a panel of the default
    // side 3.29 m (Davy's forced radiation efficiency, "Finite plate - ...").
    ub::OneThread one_thread;
    // 4 mm glass (tf of window glass, falling), 152 mm concrete (tf of concrete, rising): report 1.11
    struct M
    {
        const char *name;
        double a, mu, thk, tfR, tfB, fRef;
    };
    const std::vector<M> mats = {{"glass 4 one", 8.4794e-10, 1.0, 0.004, 60.198, -0.6212, 4.5805e-05},
                                 {"glass 4 two", 1.7665e-06, 2083.3, 0.004, 60.198, -0.6212, 4.5805e-05},
                                 {"concrete 152 one", 2.0260e-09, 1.0, 0.152, 3.1395e-04, 0.4174, 5.0212e-05},
                                 {"concrete 152 two", 3.8503e-06, 1900.5, 0.152, 3.1395e-04, 0.4174, 5.0212e-05}};
    QD_DIAG("\n# diag: plate rule R1h, a rigid mass-law panel in air: walker T, R against a tau / (a + tau - a tau)\n");
    QD_DIAG("# and 1 - a, a = tf(f), tau the mass term; S sheet, X traced solid\n");
    QD_DIAG("material,kind,type,f_Hz,theta,a,T_dB,T_rule_dB,dT_dB,R_dB,R_rule_dB,dR_dB\n");
    for (const auto &m : mats)
        for (double f_hz : {500.0, 2000.0})
        {
            const double f = f_hz * r1::FMAP * 1.0e9, fG = f_hz * r1::FMAP;
            const double a = std::min(1.0, m.tfR * std::pow(fG / m.fRef, m.tfB));
            for (int kind = 0; kind < 2; ++kind)
            {
                std::vector<LumpMtl> lp = {{m.a, 0.0, kind ? 0.0 : m.thk, 0.0, m.tfR, 0.0, 0.0, 0.0, 0.0, 20.0, m.fRef}};
                ub::Pal P = lump_palette(lp);
                r1::col(P, "tfB", {m.tfB}), r1::col(P, "e", {m.mu});
                for (int tr : {4, 5})
                    for (double thd : {0.0, 10.0, 30.0, 60.0, 80.0})
                    {
                        const double c = std::cos(thd * QPI / 180.0);
                        const double tau = std::pow(10.0, -0.1 * r1::mass_TL_side((fG / m.fRef) * m.thk, c, f_hz, 3.29)); // no side column: 3.29 m
                        r1::Out W = kind ? r1::wall(P, {0.0, m.thk}, {1, 1}, {false, true}, tr, f, thd) : r1::wall(P, {0.0}, {1}, {false}, tr, f, thd);
                        const double Tr = r1::plate(a, tau), Rr = 1.0 - a;
                        QD_DIAG("%s,%s,%d,%.0f,%.0f,%.4f,%.3f,%.3f,%+.4f,%.4f,%.4f,%+.5f\n", m.name, kind ? "X" : "S", tr, f_hz, thd, a, r1::dB(W.T), r1::dB(Tr),
                                r1::dB(W.T) - r1::dB(Tr), r1::dB(W.R), r1::dB(Rr), r1::dB(W.R) - r1::dB(Rr));
                        INFO(m.name << (kind ? " traced" : " sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                        CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.01);
                        CHECK(std::abs(r1::dB(W.R) - r1::dB(Rr)) < 0.003);
                    }
            }
        }
}

TEST_CASE("Plate rule - a wall bonded of two mass-law bodies passes the same in both directions", "[acoustic]")
{
    // N1: wood 20 mm (tf 0.2) bonded to concrete 100 mm (tf 0.1), the test palette's media, m 20: one plate
    // of the summed X0, its outer faces absorbing 0.2 and 0.1. It passes a tau / (a + tau - a tau) with a
    // the smaller, 0.1, from either side; as two traced solids (X), a sheet on a solid (H) and a stack of
    // two sheets (S). At bb336dc each face leaked and the inner face was the flat leak: the two directions differed.
    ub::OneThread one_thread;
    const double e_w = 2.1609e-8, e_c = 2.401e-9, f = FRQ, fG = f * 1.0e-9, fRef = 0.001, dw = 0.02, dc = 0.10;
    auto P_of = [&](double tw, double tc)
    {
        std::vector<LumpMtl> lp = {{e_w, 0.0, tw, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}, {e_c, 0.0, tc, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
        return lump_palette(lp);
    };
    ub::Pal PX = P_of(0.0, 0.0), PH = P_of(dw, 0.0), PS = P_of(dw, dc);
    QD_DIAG("\n# diag: a bonded wood | concrete wall, both directions: T against the rule with a = 0.1 and the summed X0\n");
    QD_DIAG("form,type,theta,T_wc_dB,T_cw_dB,T_rule_dB,d_wc,d_cw\n");
    for (int tr : {4, 5})
        for (double thd : {10.0, 30.0, 60.0})
        {
            const double c = std::cos(thd * QPI / 180.0);
            const double tau = std::pow(10.0, -0.1 * r1::mass_TL((fG / fRef) * (dw + dc), c)), Tr = r1::plate(0.1, tau);
            struct F
            {
                const char *form;
                r1::Out wc, cw;
            };
            const std::vector<F> forms = {
                {"X", r1::wall(PX, {0.0, dw, dw, dw + dc}, {1, 1, 2, 2}, {false, true, false, true}, tr, f, thd),
                 r1::wall(PX, {0.0, dc, dc, dc + dw}, {2, 2, 1, 1}, {false, true, false, true}, tr, f, thd)},
                {"H", r1::wall(PH, {0.0, 0.0, dc}, {1, 2, 2}, {false, false, true}, tr, f, thd),
                 r1::wall(PH, {0.0, dc, dc}, {2, 2, 1}, {false, true, false}, tr, f, thd)},
                {"S", r1::wall(PS, {0.0, 0.0}, {1, 2}, {false, false}, tr, f, thd), r1::wall(PS, {0.0, 0.0}, {2, 1}, {false, false}, tr, f, thd)}};
            for (const auto &x : forms)
            {
                QD_DIAG("%s,%d,%.0f,%.3f,%.3f,%.3f,%+.4f,%+.4f\n", x.form, tr, thd, r1::dB(x.wc.T), r1::dB(x.cw.T), r1::dB(Tr), r1::dB(x.wc.T) - r1::dB(Tr),
                        r1::dB(x.cw.T) - r1::dB(Tr));
                INFO("form " << x.form << ", type " << tr << ", " << thd << " deg");
                CHECK(std::abs(r1::dB(x.wc.T) - r1::dB(x.cw.T)) < 0.01);
                CHECK(std::abs(r1::dB(x.wc.T) - r1::dB(Tr)) < 0.01);
            }
        }
}

TEST_CASE("Plate rule - a traced mass-law body passes in one path: nothing is reflected back into it", "[acoustic]")
{
    // N2 (M2): a held mass-law body is a lumped plate with no wave inside; a face reached from inside
    // passes what reaches it and reflects nothing. Following every bounce (eps 1.5) a traced 4 mm pane
    // and a 20 mm wood panel pass in one path, equal to the summed series. At bb336dc the bounces were
    // followed and each traversal paid the mass term again.
    ub::OneThread one_thread;
    QD_DIAG("\n# diag: paths behind a traced mass-law body, every bounce followed\n");
    QD_DIAG("body,type,theta,nT,T_fol_dB,T_sum_dB\n");
    struct B
    {
        const char *name;
        double a, thk, tf, fRef;
    };
    for (const B &b : std::vector<B>{{"glass 4 mm", 8.4794e-10, 0.004, 0.13, 4.5805e-05}, {"wood 20 mm", 2.1609e-8, 0.02, 0.2, 0.001}})
    {
        std::vector<LumpMtl> lp = {{b.a, 0.0, 0.0, 0.0, b.tf, 0.0, 0.0, 0.0, 0.0, 20.0, b.fRef}};
        ub::Pal P = lump_palette(lp);
        const double f = 500.0 * r1::FMAP * 1.0e9;
        for (int tr : {4, 5})
            for (double thd : {0.0, 30.0, 60.0})
            {
                r1::Out F = r1::wall(P, {0.0, b.thk}, {1, 1}, {false, true}, tr, f, thd, 1.5), S = r1::wall(P, {0.0, b.thk}, {1, 1}, {false, true}, tr, f, thd, 0.0);
                QD_DIAG("%s,%d,%.0f,%d,%.4f,%.4f\n", b.name, tr, thd, F.nT, r1::dB(F.T), r1::dB(S.T));
                INFO(b.name << ", type " << tr << ", " << thd << " deg");
                CHECK(F.nT == 1);
                CHECK(std::abs(r1::dB(F.T) - r1::dB(S.T)) < 1.0e-6);
            }
    }
}

TEST_CASE("Resonance - on a mass-law material res* is the plate's resonance and leaves the permittivity alone", "[acoustic]")
{
    // N3 (B4s): resS > 0, the dip of a double unit or a lining, X0 min(1, |1 - (f c/resF)^2 + j f c/(resF resQ)|);
    // resS < 0, the panel's first mode, X0 |1 - (resF/f)^2 - j resF/(f resQ)|. A rigid body with res* stays
    // closed: it reflects 1 - tf as without, its sheet and its traced solid are one wall. At bb336dc res* was a
    // permittivity pole: the body was lossy at every frequency, never closed, and its sheet and traced solid
    // parted by up to 143 dB (report 3.10).
    ub::OneThread one_thread;
    const double a = 8.4794e-10, thk = 0.008, tf = 0.05, fRef = 4.5805e-05; // 20 kg/m2 of glass, the lumped double unit
    struct C
    {
        const char *name;
        double F_hz, Q, S;
        std::vector<double> f_hz;
    };
    const std::vector<C> cases = {{"dip (resS +1, 211 Hz, Q 4)", 211.4, 4.0, 1.0, {125.0, 211.4, 250.0, 500.0}},
                                  {"first mode (resS -1, 60 Hz, Q 3)", 60.0, 3.0, -1.0, {31.5, 60.0, 125.0}}};
    QD_DIAG("\n# diag: res* on a mass-law sheet / traced solid: T against the rule with X0 times the resonance factor, R against 1 - tf\n");
    QD_DIAG("case,kind,type,f_Hz,theta,T_dB,T_rule_dB,dT,R,R_rule\n");
    for (const auto &k : cases)
        for (int kind = 0; kind < 2; ++kind)
        {
            std::vector<LumpMtl> lp = {{a, 0.0, kind ? 0.0 : thk, 0.0, tf, k.F_hz * r1::FMAP, k.Q, k.S, 0.0, 20.0, fRef}};
            ub::Pal P = side0(lump_palette(lp));
            for (int tr : {4, 5})
                for (double f_hz : k.f_hz)
                    for (double thd : {0.0, 30.0, 60.0})
                    {
                        const double fG = f_hz * r1::FMAP, f = fG * 1.0e9, c = std::cos(thd * QPI / 180.0);
                        const double X0 = (fG / fRef) * thk * r1::res_fac(fG, c, k.F_hz * r1::FMAP, k.Q, k.S);
                        const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL(X0, c)));
                        r1::Out W = kind ? r1::wall(P, {0.0, thk}, {1, 1}, {false, true}, tr, f, thd) : r1::wall(P, {0.0}, {1}, {false}, tr, f, thd);
                        QD_DIAG("%s,%s,%d,%.1f,%.0f,%.3f,%.3f,%+.4f,%.6f,%.6f\n", k.name, kind ? "X" : "S", tr, f_hz, thd, r1::dB(W.T), r1::dB(Tr),
                                r1::dB(W.T) - r1::dB(Tr), W.R, 1.0 - tf);
                        INFO(k.name << (kind ? " traced" : " sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                        CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.01);
                        CHECK(std::abs(W.R - (1.0 - tf)) < 1.0e-3);
                    }
        }
}

TEST_CASE("Coincidence - on a mass-law material coiF and coiQ are the bending stiffness of the plate", "[acoustic]")
{
    // N4 (M4a): X0 |1 - (f/coiF)^2 sin^4(theta) (1 + j/coiQ)|: the diffuse-field dip of a pane comes from the
    // angle. 4 mm glass, coincidence 2989 Hz, loss factor 0.015 (coiQ 66.7). At bb336dc coiF / coiQ acted only
    // through coiA in the interface gain: the plate had no coincidence. coiA is no column any more (D2-b).
    ub::OneThread one_thread;
    const double a = 8.4794e-10, thk = 0.004, tf = 0.05, fRef = 4.5805e-05, cF = 2989.0 * r1::FMAP, cQ = 66.7;
    QD_DIAG("\n# diag: the bending stiffness of a 4 mm pane, sheet and traced solid, scalar\n");
    QD_DIAG("kind,type,f_Hz,theta,T_dB,T_rule_dB,dT\n");
    for (int kind = 0; kind < 2; ++kind)
    {
        std::vector<LumpMtl> lp = {{a, 0.0, kind ? 0.0 : thk, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
        ub::Pal P = side0(lump_palette(lp));
        r1::col(P, "coiF", {cF}), r1::col(P, "coiQ", {cQ});
        for (int tr : {4, 5})
            for (double f_hz : {1000.0, 4000.0})
                for (double thd : {0.0, 30.0, 59.8, 70.0, 80.0})
                {
                    const double fG = f_hz * r1::FMAP, f = fG * 1.0e9, c = std::cos(thd * QPI / 180.0);
                    const double X0 = (fG / fRef) * thk * r1::coi_fac(fG, c, cF, cQ);
                    const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL(X0, c)));
                    r1::Out W = kind ? r1::wall(P, {0.0, thk}, {1, 1}, {false, true}, tr, f, thd) : r1::wall(P, {0.0}, {1}, {false}, tr, f, thd);
                    QD_DIAG("%s,%d,%.0f,%.1f,%.3f,%.3f,%+.4f\n", kind ? "X" : "S", tr, f_hz, thd, r1::dB(W.T), r1::dB(Tr), r1::dB(W.T) - r1::dB(Tr));
                    INFO((kind ? "traced" : "sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.01);
                }
    }
}

TEST_CASE("Coincidence column - a table that carries coiA loads and the column changes nothing - the mass-law part", "[acoustic]")
{
    // D2-b on a mass-law material: a 4 mm pane with its bending stiffness (coiF 2989 Hz, coiQ 66.7, which act in
    // X0) and a coiA column, read from a table, passes and reflects exactly what it does without the column,
    // bitwise, at the coincidence frequency: as a traced solid and as a sheet, EM type 1 (both channels) and
    // scalar types 4 and 5.
    ub::OneThread one_thread;
    const char *FN = "coiA_table_mass.csv";
    const char *HEADER = "name,a,tf,m,fRef,thk,coiF,coiQ,coiA";
    char rows[512];
    const double cF = 2989.0 * r1::FMAP;
    std::snprintf(rows, sizeof(rows), "air,1,0,0,1,0,0,0,0\npane,8.4794e-10,0.05,20,4.5805e-05,0,%.9g,66.7,6\nsheet,8.4794e-10,0.05,20,4.5805e-05,0.004,%.9g,66.7,6\n", cF, cF);
    const char *ROWS = rows;

    // a table with a coiA column, read through obj_file_read (table only, name-keyed, schema-blind)
    std::unordered_map<std::string, std::vector<double>> pal;
    {
        std::ofstream csv(FN);
        REQUIRE(csv.is_open());
        csv << HEADER << "\n" << ROWS;
        csv.close();
        arma::uvec csv_ind;
        std::vector<std::string> csv_names;
        REQUIRE_NOTHROW(quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            FN, &csv_ind, &csv_names, &pal));
        std::remove(FN);
        REQUIRE(csv_names.size() == 3);
        REQUIRE(pal.count("coiA") == 1); // the column is read and handed on
    }
    std::unordered_map<std::string, std::vector<double>> bare = pal;
    bare.erase("coiA");

    const double f = cF * 1.0e9;
    CHECK(quadriga_lib::interface_gain(pal, 2, f) == quadriga_lib::interface_gain(bare, 2, f));
    for (int kind = 0; kind < 2; ++kind) // 0 traced solid 4 mm, 1 sheet 4 mm
        for (int tr : {1, 4, 5})
            for (double thd : {0.0, 30.0, 59.8})
            {
                arma::uvec mi;
                arma::mat mesh = kind ? ub::build({0.0}, {3}, {false}, mi) : ub::build({0.0, 0.004}, {2, 2}, {false, true}, mi), o, d;
                diag_ray_at(thd, o, d);
                bf::Acc A, B;
                bf::walk(pal, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 60, A);
                bf::walk(bare, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 60, B);
                INFO((kind ? "sheet" : "traced solid") << ", type " << tr << ", " << thd << " deg");
                CHECK(A.nT == B.nT);
                CHECK(A.nR == B.nR);
                CHECK((A.T[0] == B.T[0] && A.T[1] == B.T[1]));
                CHECK((A.R[0] == B.R[0] && A.R[1] == B.R[1]));
                CHECK(std::norm(A.T[0]) > 1.0e-8); // the row is not passing on nothing
            }
}

TEST_CASE("Mass law - m switches the limp mass law on", "[acoustic]")
{
    // M10: m > 0 switches the mass term on, and the term is the limp mass law at 20 dB per decade whatever
    // the value of m. A 4 mm pane of float glass (2500 kg/m3) with fRef written from its density, so that
    // (f / fRef) thk is the impedance ratio X = omega m'' / (2 rho0 c) of the limp panel (md/acoustic.md 2.1):
    // fRef = (f_GHz / f_Hz) rho0 c / (pi rho). The reference is the thin-plate law of 3.4 without bending,
    // 10 log10(1 + (X cos)^2), normalized as in 2.1 to 20 log10 X at normal incidence; the pane passes the
    // plate rule of that term and its face (tf 0.05). The same pane with m 12 and m 30 passes the same,
    // bitwise. As a sheet and as a traced solid, both acoustic tracers. The table carries no side column, so the
    // pane radiates as a panel of side 3.29 m: the reference takes Davy's forced radiation efficiency in place of
    // 1 / cos (decision 2026-10-19, "Finite plate - ..."); its term lies above the infinite plate's by 6.5 / 4.3 / 1.0 dB at
    // 80 deg and 125 / 500 / 2000 Hz, by 2.6 dB at 60 deg and 125 Hz, and by less than 0.04 dB elsewhere in the table.
    ub::OneThread one_thread;
    const double a = 8.4794e-10, rho = 2500.0, rho0c = 1.204 * 343.0, thk = 0.004, tf = 0.05;
    const double fRef = r1::FMAP * rho0c / (QPI * rho);
    QD_DIAG("\n# diag: m switches the limp mass law on, 4 mm glass, sheet (S) and traced solid (X), scalar\n");
    QD_DIAG("kind,m,type,f_Hz,theta,T_dB,T_ref_dB,dT\n");
    for (int kind = 0; kind < 2; ++kind)
        for (int tr : {4, 5})
            for (double f_hz : {125.0, 500.0, 2000.0, 4000.0})
                for (double thd : {0.0, 30.0, 60.0, 80.0})
                {
                    const double c = std::cos(thd * QPI / 180.0), X = 2.0 * QPI * f_hz * rho * thk / (2.0 * rho0c);
                    const double TL = r1::mass_TL_side(X, c, f_hz, 3.29); // no side column: a panel of side 3.29 m
                    const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * TL));
                    double T20 = 0.0;
                    for (double m : {20.0, 12.0, 30.0})
                    {
                        std::vector<LumpMtl> lp = {{a, 0.0, kind ? 0.0 : thk, 0.0, tf, 0.0, 0.0, 0.0, 0.0, m, fRef}};
                        ub::Pal P = lump_palette(lp);
                        r1::Out W = kind ? r1::wall(P, {0.0, thk}, {1, 1}, {false, true}, tr, f_hz * r1::FMAP * 1.0e9, thd)
                                         : r1::wall(P, {0.0}, {1}, {false}, tr, f_hz * r1::FMAP * 1.0e9, thd);
                        QD_DIAG("%s,%.0f,%d,%.0f,%.0f,%.4f,%.4f,%+.4f\n", kind ? "X" : "S", m, tr, f_hz, thd, r1::dB(W.T), r1::dB(Tr), r1::dB(W.T) - r1::dB(Tr));
                        INFO((kind ? "traced" : "sheet") << ", m " << m << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                        CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.01);
                        if (m == 20.0)
                            T20 = W.T;
                        else
                            CHECK(W.T == T20); // m is a switch
                    }
                }
}

TEST_CASE("Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency", "[acoustic]")
{
    // Decision 2026-10-19 (md/grazing_report.md 0.7): the mass term of a mass-law material takes the radiation
    // efficiency of the forced wave on a finite panel in place of the infinite plate's 1 / cos, sigma_D of Davy's
    // closed form for a square of side s = sqrt(area) (2a = s, S = s^2), with the gain of a finite plate 1 / (sigma c),
    // the transmission clipped at one. The plate's own impedance is untouched. The reference is written here from
    // Davy's equations (r1::davy_sigma), not from library code. A 4 mm pane with side 1.35 m and a 152 mm concrete
    // wall with side 3.29 m, limp (fRef from the density, as in "Mass law - m switches the limp mass law on"), the
    // face of each tf 0.05: the panel passes the plate rule of that face and the term. Sheet and traced solid, both
    // acoustic tracers, 0 to 89.9 deg, fine near grazing.
    //
    // What the finite size changes, read off the rows: near grazing the infinite plate turns transparent and the
    // finite one does not (glass 500 Hz, 89 deg: a term of 1.6 dB against 13.2 dB); at 31.5 Hz the small pane radiates
    // less than an infinite plate even at normal incidence (sigma 1 / sqrt(1 + q^2)), so its term rises there.
    ub::OneThread one_thread;
    const double rho0c = 1.204 * 343.0, tf = 0.05;
    struct M
    {
        const char *name;
        double a, rho, thk, side;
    };
    const std::vector<M> mats = {{"glass 4 mm", 8.4794e-10, 2500.0, 0.004, 1.35}, {"concrete 152 mm", 2.0260e-09, 347.0 / 0.152, 0.152, 3.29}};
    const std::vector<double> angles = {0.0, 30.0, 45.0, 60.0, 70.0, 75.0, 78.0, 80.0, 82.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 89.5, 89.9};
    QD_DIAG("\n# diag: finite plate, Davy's forced radiation efficiency in the mass term; S sheet, X traced solid, scalar\n");
    QD_DIAG("# TL_side the reference term at the panel's side, TL_inf the infinite plate's\n");
    QD_DIAG("material,kind,type,f_Hz,theta,sigma,TL_side_dB,TL_inf_dB,T_dB,T_ref_dB,dT\n");
    for (const auto &m : mats)
    {
        const double fRef = r1::FMAP * rho0c / (QPI * m.rho);
        for (int kind = 0; kind < 2; ++kind)
        {
            std::vector<LumpMtl> lp = {{m.a, 0.0, kind ? 0.0 : m.thk, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
            ub::Pal P = lump_palette(lp);
            r1::col(P, "side", {m.side});
            for (int tr : {4, 5})
                for (double f_hz : {31.5, 63.0, 125.0, 500.0, 2000.0})
                    for (double thd : angles)
                    {
                        const double c = std::cos(thd * QPI / 180.0), X = 2.0 * QPI * f_hz * m.rho * m.thk / (2.0 * rho0c);
                        const double TLs = r1::mass_TL_side(X, c, f_hz, m.side), TLi = r1::mass_TL(X, c);
                        const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * TLs));
                        r1::Out W = kind ? r1::wall_g(P, {0.0, m.thk}, {1, 1}, {false, true}, tr, f_hz * r1::FMAP * 1.0e9, thd)
                                         : r1::wall_g(P, {0.0}, {1}, {false}, tr, f_hz * r1::FMAP * 1.0e9, thd);
                        QD_DIAG("%s,%s,%d,%.1f,%.1f,%.5f,%.4f,%.4f,%.4f,%.4f,%+.4f\n", m.name, kind ? "X" : "S", tr, f_hz, thd, r1::davy_sigma(f_hz, c, m.side),
                                TLs, TLi, r1::dB(W.T), r1::dB(Tr), r1::dB(W.T) - r1::dB(Tr));
                        INFO(m.name << (kind ? " traced" : " sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                        REQUIRE(W.nT > 0);
                        CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.01);
                    }
        }
        // the size reaches normal incidence at 31.5 Hz: the small pane's term rises by 7.8 dB, the wall's by 3.0 dB
        // (the reference, printed), far beyond the 0.01 dB tolerance above
        const double X31 = 2.0 * QPI * 31.5 * m.rho * m.thk / (2.0 * rho0c);
        QD_DIAG("# %s, 31.5 Hz, normal incidence: TL_side - TL_inf = %+.3f dB\n", m.name, r1::mass_TL_side(X31, 1.0, 31.5, m.side) - r1::mass_TL(X31, 1.0));
        CHECK(r1::mass_TL_side(X31, 1.0, 31.5, m.side) - r1::mass_TL(X31, 1.0) > 1.0);
    }
}

TEST_CASE("Finite plate - side 0 is the infinite plate, a table without the column is side 3.29, a wave medium ignores it, a negative side is refused, a bonded run takes the side of the body that forms its term", "[acoustic]")
{
    // The side column of a mass-law material (decision 2026-10-19): side = 0 gives the infinite plate, the mass term
    // of md/acoustic.md 2.1; a table without the column takes side 3.29 m (sqrt of 10.8 m^2, a 4.0 x 2.7 m wall),
    // bitwise; the column is read only where m > 0, so a wave medium passes and reflects bitwise the same with it;
    // side < 0 is refused like every other non-negative column. A bonded run of mass-law bodies is one panel and
    // should carry one side (rule of use, like one att per run): where the sides differ, each body pays its
    // increment at its own side, TL(X_run + X0; s) - TL(X_run; s), so the run passes
    // TL(X1; s1) + TL(X1 + X2; s2) - TL(X1; s2).
    ub::OneThread one_thread;
    const double rho0c = 1.204 * 343.0, tf = 0.05, rho = 2500.0, thk = 0.004, fRef = r1::FMAP * rho0c / (QPI * rho);
    QD_DIAG("\n# diag: side 0 against the infinite plate's rule, 4 mm glass, S sheet, X traced solid, scalar\n");
    QD_DIAG("kind,type,f_Hz,theta,T_dB,T_ref_dB,dT\n");
    for (int kind = 0; kind < 2; ++kind)
    {
        std::vector<LumpMtl> lp = {{8.4794e-10, 0.0, kind ? 0.0 : thk, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
        ub::Pal P0 = lump_palette(lp), Pn = lump_palette(lp), Pd = lump_palette(lp);
        r1::col(P0, "side", {0.0}), r1::col(Pd, "side", {3.29});
        for (int tr : {4, 5})
            for (double f_hz : {31.5, 500.0})
                for (double thd : {0.0, 60.0, 85.0, 89.5})
                {
                    const double f = f_hz * r1::FMAP * 1.0e9, c = std::cos(thd * QPI / 180.0), X = 2.0 * QPI * f_hz * rho * thk / (2.0 * rho0c);
                    auto W = [&](const ub::Pal &P)
                    { return kind ? r1::wall_g(P, {0.0, thk}, {1, 1}, {false, true}, tr, f, thd) : r1::wall_g(P, {0.0}, {1}, {false}, tr, f, thd); };
                    r1::Out W0 = W(P0), Wn = W(Pn), Wd = W(Pd);
                    const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL(X, c)));
                    QD_DIAG("%s,%d,%.1f,%.1f,%.6f,%.6f,%+.2e\n", kind ? "X" : "S", tr, f_hz, thd, r1::dB(W0.T), r1::dB(Tr), r1::dB(W0.T) - r1::dB(Tr));
                    INFO((kind ? "traced" : "sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(r1::dB(W0.T) - r1::dB(Tr)) < 0.01); // side 0: the infinite plate (the face's open normal incidence: 0.008 dB)
                    CHECK(Wn.T == Wd.T); // no column: side 3.29, bitwise
                    CHECK(Wn.R == Wd.R);
                }
    }
    {
        // a wave medium (m = 0) with a side column passes and reflects what it does without one, bitwise
        for (int kind = 0; kind < 2; ++kind)
            for (int tr : {1, 4, 5})
                for (double thd : {0.0, 60.0, 85.0})
                {
                    std::vector<LumpMtl> lp = {{5.24, 0.3, kind ? 0.0 : 0.0125, 0.0, 0.2}};
                    ub::Pal Pw = lump_palette(lp), Ps = lump_palette(lp);
                    r1::col(Ps, "side", {1.0});
                    auto W = [&](const ub::Pal &P)
                    { return kind ? r1::wall_g(P, {0.0, 0.0125}, {1, 1}, {false, true}, tr, FRQ, thd) : r1::wall_g(P, {0.0}, {1}, {false}, tr, FRQ, thd); };
                    r1::Out A = W(Pw), B = W(Ps);
                    INFO("wave medium, " << (kind ? "traced" : "sheet") << ", type " << tr << ", " << thd << " deg");
                    CHECK(A.T == B.T);
                    CHECK(A.R == B.R);
                }
    }
    {
        // side < 0 is refused
        std::unordered_map<std::string, std::vector<double>> bad = {{"a", {8.4794e-10}}, {"m", {20.0}}, {"fRef", {fRef}}, {"side", {-1.0}}};
        CHECK_THROWS_AS(quadriga_lib::medium_gain<double>(bad, 1, 0.1, 1.0e9), std::invalid_argument);
    }
    {
        // a bonded run: two 152 mm concrete bodies, one solid on the other, the first of side 1.35 m, the second of
        // side 3.29 m; reference the rule above, against the two single-side readings of the run
        const double rc = 347.0 / 0.152, Lc = 0.152, fRc = r1::FMAP * rho0c / (QPI * rc);
        std::vector<LumpMtl> lp = {{2.0260e-09, 0.0, 0.0, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRc}, {2.0260e-09, 0.0, 0.0, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRc}};
        ub::Pal P = lump_palette(lp);
        r1::col(P, "side", {1.35, 3.29});
        r1::col(P, "attB", {0.0, 1.0}); // att is zero: this only keeps the two rows apart as materials
        QD_DIAG("\n# diag: a bonded run of two concrete bodies, sides 1.35 | 3.29 m, traced solids, scalar\n");
        QD_DIAG("type,f_Hz,theta,T_dB,T_rule_dB,T_all_s1_dB,T_all_s2_dB,dT\n");
        for (int tr : {4, 5})
            for (double f_hz : {31.5, 125.0, 500.0})
                for (double thd : {0.0, 60.0, 85.0})
                {
                    const double c = std::cos(thd * QPI / 180.0), X1 = 2.0 * QPI * f_hz * rc * Lc / (2.0 * rho0c), X2 = X1;
                    const double TLr = r1::mass_TL_side(X1, c, f_hz, 1.35) + r1::mass_TL_side(X1 + X2, c, f_hz, 3.29) - r1::mass_TL_side(X1, c, f_hz, 3.29);
                    const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * TLr));
                    const double T1 = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL_side(X1 + X2, c, f_hz, 1.35)));
                    const double T2 = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL_side(X1 + X2, c, f_hz, 3.29)));
                    r1::Out W = r1::wall_g(P, {0.0, Lc, Lc, 2.0 * Lc}, {1, 1, 2, 2}, {false, true, false, true}, tr, f_hz * r1::FMAP * 1.0e9, thd);
                    QD_DIAG("%d,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f,%+.4f\n", tr, f_hz, thd, r1::dB(W.T), r1::dB(Tr), r1::dB(T1), r1::dB(T2), r1::dB(W.T) - r1::dB(Tr));
                    INFO("bonded run, type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 0.02); // the run's code is quantized (Bonded mass, 0.01 dB)
                }
    }
}

TEST_CASE("Two-parameter mapping - air | water, a hard wall's phase, a porous layer on it and every tracer below its critical angle", "[acoustic]")
{
    // N7 / A12: eps = n z, mu = n / z, z = Z_air / Z, n = c_air / c. Water (20 C: c 1482.36 m/s, Bilaniuk & Wong;
    // rho 998.21 kg/m3, Tanaka et al.; air 1.2 kg/m3, 343 m/s) as a traced half-space: T and R of the fluid
    // interface. A hard wall (concrete, n = 343 / c_L): reflection phase within 0.1 deg to 60 deg. A porous
    // layer (Delany-Bazley, 11 kPa s/m2, 20 mm, 500 Hz) on that wall: the exact layer on a hard backing.
    // These held at bb336dc too (guards). A traced rigid solid below its critical angle (3.48 deg glass,
    // 9.44 deg gypsum) holds every tracer: at bb336dc tracer 5 refracted into it.
    ub::OneThread one_thread;
    const double Za = 1.2 * 343.0, Zw = 998.21 * 1482.36, zw = Za / Zw, nw = 343.0 / 1482.36;
    {
        std::vector<LumpMtl> lp = {{nw * zw}};
        ub::Pal P = lump_palette(lp);
        r1::col(P, "e", {nw / zw});
        QD_DIAG("\n# diag: air | water, two-parameter mapping, traced half-space, against the fluid interface\n");
        QD_DIAG("type,theta,T_dB,T_ref_dB,R_dB,R_ref_dB\n");
        for (int tr : {4, 5})
            for (double thd : {0.0, 5.0, 10.0, 13.0, 13.37, 13.38, 30.0, 60.0})
            {
                const double t = thd * QPI / 180.0, s2 = std::sin(t) / nw, c1 = std::cos(t);
                const std::complex<double> c2 = (s2 < 1.0) ? std::complex<double>(std::sqrt(1.0 - s2 * s2), 0.0) : std::complex<double>(0.0, -std::sqrt(s2 * s2 - 1.0));
                const std::complex<double> r = (Zw * c1 - Za * c2) / (Zw * c1 + Za * c2);
                const double Rr = std::norm(r), Tr = (s2 < 1.0) ? 1.0 - Rr : 0.0;
                r1::Out W = r1::sink(P, {0.0, 1.0}, {1, 1}, {false, true}, tr, FRQ, thd, 1);
                QD_DIAG("%d,%.2f,%.6f,%.6f,%.8f,%.8f\n", tr, thd, r1::dB(W.T), r1::dB(Tr), r1::dB(W.R), r1::dB(Rr));
                INFO("air | water, type " << tr << ", " << thd << " deg");
                CHECK(std::abs(r1::dB(W.R) - r1::dB(Rr)) < 1.0e-6);
                if (Tr > 0.0)
                    CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 1.0e-6);
                else
                    CHECK(W.T < 1.0e-20);
            }
    }
    {
        // the reflection phase of a hard wall: one face through ray_mesh_interact, scalar
        std::unordered_map<std::string, std::vector<double>> pal = {{"a", {3.8503e-06}}, {"e", {1900.5}}, {"m", {20.0}}, {"fRef", {5.0212e-05}}};
        const double h = 100.0;
        arma::mat mesh = {{0.0, -h, -h, 0.0, -h, h, 0.0, h, -h}};
        arma::uvec mi = {1};
        // reference: the fluid interface of the mapping, z = sqrt(eps / mu), n = sqrt(eps mu), decaying past n
        const double zr = std::sqrt(3.8503e-06 / 1900.5), nr = std::sqrt(3.8503e-06 * 1900.5);
        QD_DIAG("\n# diag: the reflection phase of a hard wall (concrete, two-parameter), one face, scalar\n");
        QD_DIAG("theta,R,R_ref,phase_deg,phase_ref_deg\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 60.0})
        {
            double t = thd * QPI / 180.0;
            arma::mat orig = {{-1.0, -std::tan(t), 0.0}}, dest = {{1.0, std::tan(t), 0.0}}, origN, destN, fbsN, sbsN, xpr;
            arma::u32_vec fbs = {1}, sbs = {0};
            arma::vec gain;
            quadriga_lib::ray_mesh_interact<double>(3, FRQ, &orig, &dest, &mesh, &mi, &pal, &fbs, &sbs, nullptr, nullptr, &origN, &destN, &fbsN, &sbsN, &gain, &xpr);
            const double ph = std::atan2(xpr(1, 0), xpr(0, 0)) * 180.0 / QPI, st = std::sin(t) / nr;
            const std::complex<double> c2 = (st < 1.0) ? std::complex<double>(std::sqrt(1.0 - st * st), 0.0) : std::complex<double>(0.0, -std::sqrt(st * st - 1.0));
            const std::complex<double> rr = (std::cos(t) - zr * c2) / (std::cos(t) + zr * c2);
            QD_DIAG("%.0f,%.9f,%.9f,%+.5f,%+.5f\n", thd, gain(0), std::norm(rr), ph, std::arg(rr) * 180.0 / QPI);
            INFO("hard wall, " << thd << " deg");
            CHECK(std::abs(ph - std::arg(rr) * 180.0 / QPI) < 1.0e-6);
            CHECK(std::abs(gain(0) - std::norm(rr)) < 1.0e-9);
            CHECK(std::abs(ph) < 0.2); // a hard wall: near zero, where the one-parameter mapping gives 2 theta
        }
    }
    {
        // a porous layer on a hard wall: Delany-Bazley at 500 Hz in the two-parameter mapping
        const double f_hz = 500.0, fG = f_hz * r1::FMAP, d = 0.02, X = 1.2 * f_hz / 11000.0, k0 = 2.0 * QPI * f_hz / 343.0;
        const std::complex<double> Zc = Za * std::complex<double>(1.0 + 0.0571 * std::pow(X, -0.754), -0.087 * std::pow(X, -0.732));
        const std::complex<double> kc = k0 * std::complex<double>(1.0 + 0.0978 * std::pow(X, -0.700), -0.189 * std::pow(X, -0.595));
        const std::complex<double> z = Za / Zc, n = kc / k0, eps = n * z, mu = n / z;
        std::vector<LumpMtl> lp = {{eps.real(), -eps.imag() * fG, d}, {3.8503e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 5.0212e-05}};
        ub::Pal P = lump_palette(lp);
        r1::col(P, "e", {mu.real(), 1900.5}), r1::col(P, "g", {-mu.imag() * fG / 17.98, 0.0});
        QD_DIAG("\n# diag: a porous layer 20 mm (11 kPa s/m2) glued on a hard wall, 500 Hz: alpha = 1 - R against the exact layer\n");
        QD_DIAG("theta,alpha,alpha_ref\n");
        for (double thd : {0.0, 30.0, 60.0, 80.0})
        {
            const double t = thd * QPI / 180.0;
            std::complex<double> cz = std::sqrt(1.0 - (std::sin(t) / n) * (std::sin(t) / n));
            if ((n * cz).imag() > 0.0)
                cz = -cz;
            const std::complex<double> zin = std::complex<double>(0.0, -1.0) / (z * cz * std::tan(k0 * n * cz * d));
            const double ar = 1.0 - std::norm((zin - 1.0 / std::cos(t)) / (zin + 1.0 / std::cos(t)));
            r1::Out W = r1::sink(P, {0.0, 0.0, 1.0}, {1, 2, 2}, {false, false, true}, 4, fG * 1.0e9, thd, 2);
            QD_DIAG("%.0f,%.5f,%.5f\n", thd, 1.0 - W.R, ar);
            INFO("porous layer on a hard wall, " << thd << " deg");
            CHECK(std::abs((1.0 - W.R) - ar) < 0.005);
        }
    }
    {
        // every tracer below the critical angle of a rigid traced solid
        struct B
        {
            const char *name;
            double a, mu, thk, th;
        };
        QD_DIAG("\n# diag: a traced rigid solid below its two-parameter critical angle, tracer 5 against 4\n");
        QD_DIAG("body,theta,T4_dB,T5_dB\n");
        for (const B &b : std::vector<B>{{"glass 4 mm", 1.7665e-06, 2083.3, 0.004, 3.47}, {"gypsum 12.5 mm", 4.6908e-05, 573.33, 0.0125, 9.43}})
        {
            std::vector<LumpMtl> lp = {{b.a, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0e-4}};
            ub::Pal P = lump_palette(lp);
            r1::col(P, "e", {b.mu});
            const double f = 500.0 * r1::FMAP * 1.0e9;
            r1::Out T4 = r1::wall(P, {0.0, b.thk}, {1, 1}, {false, true}, 4, f, b.th), T5 = r1::wall(P, {0.0, b.thk}, {1, 1}, {false, true}, 5, f, b.th);
            QD_DIAG("%s,%.2f,%.3f,%.3f\n", b.name, b.th, r1::dB(T4.T), r1::dB(T5.T));
            INFO(b.name << " at " << b.th << " deg");
            CHECK(std::abs(r1::dB(T5.T) - r1::dB(T4.T)) < 0.01);
        }
    }
}

TEST_CASE("Held exit - a lining behind a rigid wall is the same as its traced layer, for every tracer", "[acoustic]")
{
    // A2 and the exit of the plate: a coherent lining (m 0, 10 mm, dense eps 2.50 - 0.80j or porous 0.31 - 0.20j)
    // on the far face of a rigid wall (100 mm, tf 0.1, m 20) in the one- and two-parameter mapping. The wave
    // held in the wall carries the invariant it had in air; the face out of the plate passes what reached it.
    // As a glued sheet and as a traced layer the lining is one wall, and the tracers agree. At bb336dc the
    // sheet and the layer differed and the dense traced layer passed nothing from 60 deg.
    ub::OneThread one_thread;
    struct W
    {
        const char *name;
        double a, mu;
    };
    struct L
    {
        const char *name;
        double a, ei;
    };
    QD_DIAG("\n# diag: a lining behind a rigid wall, glued sheet (H) against traced layer (T), types 4 and 5, 1 GHz\n");
    QD_DIAG("wall,lining,theta,H4_dB,T4_dB,H5_dB,T5_dB\n");
    for (const W &w : std::vector<W>{{"one", 2.401e-09, 1.0}, {"two", 3.8503e-06, 1900.5}})
        for (const L &l : std::vector<L>{{"dense", 2.5, 0.8}, {"porous", 0.31, 0.2}})
        {
            std::vector<LumpMtl> lpH = {{l.a, l.ei, 0.01}, {w.a, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}};
            std::vector<LumpMtl> lpT = {{l.a, l.ei, 0.0}, {w.a, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}};
            ub::Pal PH = lump_palette(lpH), PT = lump_palette(lpT);
            r1::col(PH, "e", {1.0, w.mu}), r1::col(PT, "e", {1.0, w.mu});
            for (double thd : {10.0, 30.0, 60.0, 75.0})
            {
                double H[2], T[2];
                for (int k = 0; k < 2; ++k)
                {
                    H[k] = r1::wall(PH, {0.0, 0.1, 0.1}, {2, 2, 1}, {false, true, false}, 4 + k, FRQ, thd).T;
                    T[k] = r1::wall(PT, {0.0, 0.1, 0.1, 0.11}, {2, 2, 1, 1}, {false, true, false, true}, 4 + k, FRQ, thd).T;
                }
                QD_DIAG("%s,%s,%.0f,%.3f,%.3f,%.3f,%.3f\n", w.name, l.name, thd, r1::dB(H[0]), r1::dB(T[0]), r1::dB(H[1]), r1::dB(T[1]));
                INFO(w.name << "-parameter wall, " << l.name << " lining, " << thd << " deg");
                CHECK(std::abs(r1::dB(H[0]) - r1::dB(T[0])) < 0.05);
                CHECK(std::abs(r1::dB(H[1]) - r1::dB(H[0])) < 0.01);
                CHECK(std::abs(r1::dB(T[1]) - r1::dB(T[0])) < 0.01);
            }
        }
}

TEST_CASE("Held path - alpha on a lumped plate is charged over the path the held wave takes", "[acoustic]")
{
    // A3: a rigid mass-law panel (22 mm, tf 0.2, m 20) with alpha 100 dB/m pays it over thk / cos in air,
    // as a sheet and as a traced solid. At bb336dc the sheet charged it over the in-sheet cosine of the impedance
    // mapping, which is evanescent: the 400 dB cap from 10 deg on.
    ub::OneThread one_thread;
    QD_DIAG("\n# diag: what alpha 100 dB/m costs a 22 mm rigid mass-law panel, sheet (S) and traced solid (X), 1 GHz\n");
    QD_DIAG("type,theta,S_dB,X_dB\n");
    auto P_of = [](double thk, double alpha)
    {
        std::vector<LumpMtl> lp = {{2.1609e-08, 0.0, thk, 0.0, 0.2, 0.0, 0.0, 0.0, alpha, 20.0, 0.001}};
        return lump_palette(lp);
    };
    for (int tr : {4, 5})
        for (double thd : {0.0, 10.0, 30.0, 60.0, 80.0})
        {
            double cost[2];
            for (int kind = 0; kind < 2; ++kind)
            {
                ub::Pal P0 = P_of(kind ? 0.0 : 0.022, 0.0), P1 = P_of(kind ? 0.0 : 0.022, 100.0);
                auto run = [&](const ub::Pal &P)
                { return kind ? r1::wall(P, {0.0, 0.022}, {1, 1}, {false, true}, tr, FRQ, thd).T : r1::wall(P, {0.0}, {1}, {false}, tr, FRQ, thd).T; };
                cost[kind] = r1::dB(run(P0)) - r1::dB(run(P1));
            }
            QD_DIAG("%d,%.0f,%.3f,%.3f\n", tr, thd, cost[0], cost[1]);
            INFO("type " << tr << ", " << thd << " deg");
            CHECK(std::abs(cost[0] - cost[1]) < 0.01);
            CHECK(cost[1] > 2.0);
        }
}

TEST_CASE("Held exit - a non-rigid mass-law body holds every tracer and passes at every angle", "[acoustic]")
{
    // A7: a mass-law body is a lumped plate whatever its permittivity. A 100 mm body of eps 1.50 with m 20
    // (a laboratory construction: it has a wave inside) holds the wave at the angle it has in air, so
    // tracer 5 reads what tracer 4 reads; its exit into air is judged by the invariant of air, so it passes
    // at every angle. At bb336dc tracer 5 refracted into it.
    ub::OneThread one_thread;
    std::vector<LumpMtl> lp = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.01}}; // X0 10 at 1 GHz: a 20 dB term
    ub::Pal P = lump_palette(lp);
    QD_DIAG("\n# diag: a 100 mm non-rigid mass-law body (eps 1.50, m 20, X0 10), traced, 1 GHz\n");
    QD_DIAG("theta,T4_dB,T5_dB\n");
    for (double thd : {10.0, 30.0, 60.0, 75.0})
    {
        r1::Out T4 = r1::wall(P, {0.0, 0.1}, {1, 1}, {false, true}, 4, FRQ, thd), T5 = r1::wall(P, {0.0, 0.1}, {1, 1}, {false, true}, 5, FRQ, thd);
        QD_DIAG("%.0f,%.4f,%.4f\n", thd, r1::dB(T4.T), r1::dB(T5.T));
        INFO(thd << " deg");
        CHECK(T4.T > 1.0e-6);
        CHECK(std::abs(r1::dB(T5.T) - r1::dB(T4.T)) < 0.01);
    }
}

// M6: the peak of a mass-law face as a locally reacting face, written from the column definitions. The series
// impedance of a resonator per Z0, zeta = R' + j (1 + R') |tfQ| (f/tfF - tfF/f), with tfP = 4 R' / (1 + R')^2: the
// over-damped root R' >= 1, or the under-damped one 1/R' where tfQ < 0; tfP above one is R' = 1. It reflects
// (zeta c - 1)/(zeta c + 1) at the cosine c in air (TE and the scalar type), (zeta/c - 1)/(zeta/c + 1) for TM; the
// power-law part tf is a matched pad in front of it, |r|^2 = (1 - tf) |r_zeta|^2.
namespace m6t
{
    typedef std::complex<double> cd;
    static cd zeta(double tfP, double tfF, double tfQ, double f)
    {
        const double a = std::min(tfP, 1.0), hi = ((2.0 - a) + 2.0 * std::sqrt(1.0 - a)) / a, Rp = (tfQ < 0.0) ? 1.0 / hi : hi;
        return cd(Rp, (1.0 + Rp) * std::abs(tfQ) * (f / tfF - tfF / f));
    }
    static cd r_face(cd z, double c, bool tm = false)
    {
        const cd w = tm ? z / c : z * c;
        return (w - 1.0) / (w + 1.0);
    }
    // The board of the peak cases: 12.5 mm, m 20, tf 0.05 with a peak 0.25 at 150 Hz, Q_alpha 0.75, in the one-parameter
    // mapping or as gypsum in the two-parameter one; thk 0 for a traced solid
    static ub::Pal board(double a, double mu, double thk, double tfQ, double tf0 = 0.05, double tfP = 0.25)
    {
        std::vector<LumpMtl> lp = {{a, 0.0, thk, 0.0, tf0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.6644e-04}};
        ub::Pal P = lump_palette(lp);
        r1::col(P, "e", {mu}), r1::col(P, "tfP", {tfP}), r1::col(P, "tfF", {150.0 * r1::FMAP}), r1::col(P, "tfQ", {tfQ});
        return P;
    }
    // Every path of one ray through a wall (bf::walk, eps 0 sums every series): R and T at their polyline's point, R
    // referenced to the first face, channel 0 the scalar type or VV, 1 HH
    static bf::Acc walk(const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, int tr, double f,
                        double thd)
    {
        arma::uvec mi;
        arma::mat mesh = ub::build(x, mat, flip, mi), o, d;
        diag_ray_at(thd, o, d);
        bf::Acc A;
        bf::walk(P, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 200, A);
        return A;
    }
}

TEST_CASE("Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a mass-law face", "[acoustic]")
{
    // A10 on the lumped-resonator law (M5) and the face as its series impedance (M6, namespace m6t above). A rigid
    // mass-law panel absorbs 1 - (1 - tf)|r_zeta(theta)|^2 on its room side, which at normal incidence is the law
    // tfR (f/fRef)^tfB + tfP / (1 + (tfQ (f/tfF - tfF/f))^2) less tf times the peak, and passes the plate rule with
    // that absorption at the angle (N1), as a sheet and as a traced solid, in the one- and the two-parameter mapping. A
    // lining of plasterboard on battens: floor 0.05, peak 0.25 at 150 Hz, Q_alpha 0.75. Before M6 the face absorbed the
    // law at every angle, and exactly at normal incidence the rigid face was open (the pair rule on its Fresnel R0); at
    // bb336dc the three columns were not read: the face absorbed the floor alone.
    ub::OneThread one_thread;
    const double thk = 0.0125, fRef = 1.6644e-04, tf0 = 0.05, tfP = 0.25, tfF_hz = 150.0, tfQ = 0.75;
    struct Map
    {
        const char *name;
        double a, mu;
    };
    QD_DIAG("\n# diag: the absorption peak of a 12.5 mm board, sheet (S) and traced solid (X), scalar, one- and two-parameter mapping\n");
    QD_DIAG("map,kind,type,f_Hz,theta,alpha,alpha_ref,T_dB,T_rule_dB\n");
    double dA = 0.0, dT = 0.0;
    for (const Map &mp : std::vector<Map>{{"one", 8.1817e-08, 1.0}, {"two", 4.6908e-05, 573.33}})
        for (int kind = 0; kind < 2; ++kind)
        {
            const ub::Pal P = side0(m6t::board(mp.a, mp.mu, kind ? 0.0 : thk, tfQ));
            for (int tr : {4, 5})
                for (double f_hz : {63.0, 150.0, 250.0, 500.0, 1000.0})
                    for (double thd : {0.0, 45.0})
                    {
                        const double fG = f_hz * r1::FMAP, f = fG * 1.0e9, c = std::cos(thd * QPI / 180.0);
                        const double ar = 1.0 - (1.0 - tf0) * std::norm(m6t::r_face(m6t::zeta(tfP, tfF_hz, tfQ, f_hz), c));
                        const double Tr = r1::plate(ar, std::pow(10.0, -0.1 * r1::mass_TL((fG / fRef) * thk, c)));
                        r1::Out W = kind ? r1::wall(P, {0.0, thk}, {1, 1}, {false, true}, tr, f, thd) : r1::wall(P, {0.0}, {1}, {false}, tr, f, thd);
                        QD_DIAG("%s,%s,%d,%.0f,%.0f,%.5f,%.5f,%.3f,%.3f\n", mp.name, kind ? "X" : "S", tr, f_hz, thd, 1.0 - W.R, ar, r1::dB(W.T), r1::dB(Tr));
                        dA = std::max(dA, std::abs((1.0 - W.R) - ar)), dT = std::max(dT, std::abs(r1::dB(W.T) - r1::dB(Tr)));
                        INFO(mp.name << "-parameter " << (kind ? "traced" : "sheet") << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                        CHECK(std::abs((1.0 - W.R) - ar) < 1.0e-12);
                        CHECK(std::abs(r1::dB(W.T) - r1::dB(Tr)) < 2.0e-3);
                    }
        }
    QD_DIAG("# max |alpha - ref| %.2e, max |T - rule| %.2e dB\n", dA, dT);
}

TEST_CASE("Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 to 85 deg, as a sheet and as a traced solid, on both branches", "[acoustic]")
{
    // M6 (md/acoustic.md 4, A12): a mass-law face with a peak (m > 0, tfP > 0, tfF > 0) reflects as the locally reacting
    // face of namespace m6t, with its phase and its angle dependence, behind the pad of its power-law part. The board of
    // the peak cases, one- and two-parameter mapping (gypsum), sheet (S) and traced solid (X), every row through
    // ray_triangle_intersect, ray_mesh_interact and ray_state_update (bf::walk; type 3 one event):
    //   A  the scalar types 4, 5, 3 on both branches (tfQ 0.75 and -0.75: tfQ < 0 is accepted), 63 Hz to 1 kHz, 0 to 85
    //      deg, against sqrt(1 - tf) r_zeta, and R + T <= 1;
    //   B  a 1/96-octave sweep at 0 deg, 75 to 300 Hz: the phase extremes at the frequencies the reference puts them
    //      (-4.117 deg at 83.58 Hz, +4.117 deg at 269.21 Hz);
    //   C  tfP 1.0 and 1.2: R' = 1 on both branches, the two equal;
    //   D  EM type 1: VV (zeta c - 1)/(zeta c + 1), HH -(zeta/c - 1)/(zeta/c + 1) (a rigid face reflects -1 in HH),
    //      from 10 deg (at normal incidence the plane of incidence and the channels' basis are not defined);
    //   E  the gate: a wave medium (m 0, eps 2.50, 100 mm traced) with the same columns keeps the flat pad, the
    //      transformer of a coherent layer at each face on the pair rule, R and |T|^2 against that layer written here
    //      (within 2.9e-6; R + T reads up to 1 + 1.15e-6 on this traced dielectric, the same before M6);
    //   F  behind a 10 mm lining (eps 2.50 - 0.80j) glued and traced on the board wall (tf 0): the face is met against
    //      the lining's immittance, the transfer matrix of a locally reacting face behind a layer;
    //   G  a sweep of +-0.05 deg in 0.001 deg at the critical angle of the two-parameter board: on zeta, and no step.
    // Before M6 the face reflected sqrt(1 - tf) with the bare face's phase (2 theta in the one-parameter mapping),
    // tfQ < 0 was refused (a failed check here, not an abort), and behind a lining the pad was matched to it.
    typedef std::complex<double> cd;
    ub::OneThread one_thread;
    const double thk = 0.0125, tf0 = 0.05, tfP = 0.25, F = 150.0;
    struct Map
    {
        const char *name;
        double a, mu;
    };
    const std::vector<Map> maps = {{"one", 8.1817e-08, 1.0}, {"two", 4.6908e-05, 573.33}};
    auto deg = [](cd z)
    { return std::arg(z) * 180.0 / QPI; };
    // the walk, with a palette that today's library may refuse (tfQ < 0): a failed check, and the row reads NaN
    auto WALK = [&](const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, int tr, double f, double thd)
    {
        bf::Acc A;
        A.R[0] = A.R[1] = A.T[0] = A.T[1] = cd(std::nan(""), 0.0);
        CHECK_NOTHROW(A = m6t::walk(P, x, mat, flip, tr, f, thd));
        return A;
    };
    auto R3 = [&](const ub::Pal &P, const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, double f, double thd)
    {
        cd R(std::nan(""), 0.0);
        arma::uvec mi;
        arma::mat mesh = ub::build(x, mat, flip, mi), o, d;
        diag_ray_at(thd, o, d);
        ub::Opt op;
        op.eps = 0.0;
        CHECK_NOTHROW(R = cpx(ub::traced<double>(P, mesh, mi, 3, f, o, d, nullptr, op).xpr, 0, 0));
        return R;
    };
    double mA = 0.0, mRT = 0.0, mB = 0.0, mC = 0.0, mD = 0.0, mDRT = 0.0, mE = 0.0, mET = 0.0, mERT = 0.0, mFH = 0.0, mFX = 0.0, mG = 0.0, mGs = 0.0;

    QD_DIAG("\n# diag: A, the peaked board as its series impedance, |R| and phase against the reference\n");
    QD_DIAG("map,kind,tfQ,type,f_Hz,theta,R_abs,R_deg,ref_abs,ref_deg,dR,R+T\n");
    for (const Map &mp : maps)
        for (double q : {0.75, -0.75})
            for (int kind = 0; kind < 2; ++kind)
            {
                ub::Pal P;
                CHECK_NOTHROW(P = m6t::board(mp.a, mp.mu, kind ? 0.0 : thk, q));
                const std::vector<double> x = kind ? std::vector<double>{0.0, thk} : std::vector<double>{0.0};
                const std::vector<int> mt = kind ? std::vector<int>{1, 1} : std::vector<int>{1};
                const std::vector<bool> fl = kind ? std::vector<bool>{false, true} : std::vector<bool>{false};
                for (int tr : {4, 5, 3})
                    for (double f_hz : {63.0, 150.0, 250.0, 1000.0})
                        for (double thd : {0.0, 10.0, 30.0, 45.0, 60.0, 75.0, 85.0})
                        {
                            const double f = f_hz * r1::FMAP * 1.0e9, c = std::cos(thd * QPI / 180.0);
                            const cd ref = std::sqrt(1.0 - tf0) * m6t::r_face(m6t::zeta(tfP, F, q, f_hz), c);
                            cd R;
                            double rt = 0.0;
                            if (tr == 3)
                                R = R3(P, x, mt, fl, f, thd);
                            else
                            {
                                bf::Acc W = WALK(P, x, mt, fl, tr, f, thd);
                                R = W.R[0], rt = std::norm(W.R[0]) + std::norm(W.T[0]);
                            }
                            QD_DIAG("%s,%s,%+.2f,%d,%.0f,%.0f,%.6f,%+.4f,%.6f,%+.4f,%.1e,%.6f\n", mp.name, kind ? "X" : "S", q, tr, f_hz, thd, std::abs(R), deg(R),
                                    std::abs(ref), deg(ref), std::abs(R - ref), rt);
                            mA = std::max(mA, std::abs(R - ref)), mRT = std::max(mRT, rt);
                            INFO("A " << mp.name << "-parameter " << (kind ? "traced" : "sheet") << ", tfQ " << q << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                            CHECK(std::abs(R - ref) < 1.0e-12);
                            CHECK(rt <= 1.0);
                        }
            }

    QD_DIAG("\n# diag: B, the phase extremes on a 1/96-octave sweep at 0 deg, one-parameter sheet, over-damped\n");
    {
        const ub::Pal P = m6t::board(maps[0].a, 1.0, thk, 0.75);
        int iL[2] = {0, 0}, iR[2] = {0, 0};
        double pL[2] = {1e9, -1e9}, pR[2] = {1e9, -1e9};
        std::vector<double> fs;
        for (int k = 0; k <= 192; ++k)
            fs.push_back(75.0 * std::pow(2.0, k / 96.0));
        for (size_t k = 0; k < fs.size(); ++k)
        {
            const cd R = m6t::walk(P, {0.0}, {1}, {false}, 4, fs[k] * r1::FMAP * 1.0e9, 0.0).R[0];
            const cd ref = std::sqrt(1.0 - tf0) * m6t::r_face(m6t::zeta(tfP, F, 0.75, fs[k]), 1.0);
            mB = std::max(mB, std::abs(R - ref));
            if (deg(R) < pL[0])
                pL[0] = deg(R), iL[0] = (int)k;
            if (deg(R) > pL[1])
                pL[1] = deg(R), iL[1] = (int)k;
            if (deg(ref) < pR[0])
                pR[0] = deg(ref), iR[0] = (int)k;
            if (deg(ref) > pR[1])
                pR[1] = deg(ref), iR[1] = (int)k;
        }
        QD_DIAG("extreme,f_Hz,phase_deg,ref_f_Hz,ref_phase_deg,d_deg\nmin,%.2f,%+.4f,%.2f,%+.4f,%.1e\nmax,%.2f,%+.4f,%.2f,%+.4f,%.1e\n", fs[iL[0]], pL[0], fs[iR[0]],
                pR[0], std::abs(pL[0] - pR[0]), fs[iL[1]], pL[1], fs[iR[1]], pR[1], std::abs(pL[1] - pR[1]));
        for (int e = 0; e < 2; ++e)
        {
            INFO("B, the phase " << (e ? "maximum" : "minimum"));
            CHECK(iL[e] == iR[e]);
            CHECK(std::abs(pL[e] - pR[e]) < 1.0e-12);
        }
        CHECK(mB < 1.0e-12);
    }

    QD_DIAG("\n# diag: C, tfP 1.0 and 1.2 on both branches (R' = 1), one-parameter sheet, type 4\n");
    QD_DIAG("f_Hz,theta,R_1.0_over,R_1.0_under,R_1.2_over,ref,max_d\n");
    for (double f_hz : {100.0, 150.0, 225.0})
        for (double thd : {0.0, 45.0, 85.0})
        {
            const double f = f_hz * r1::FMAP * 1.0e9;
            const cd ref = std::sqrt(1.0 - tf0) * m6t::r_face(m6t::zeta(1.0, F, 0.75, f_hz), std::cos(thd * QPI / 180.0));
            cd R[4];
            int k = 0;
            for (double p : {1.0, 1.2})
                for (double q : {0.75, -0.75})
                {
                    ub::Pal P;
                    CHECK_NOTHROW(P = m6t::board(maps[0].a, 1.0, thk, q, tf0, p));
                    R[k++] = WALK(P, {0.0}, {1}, {false}, 4, f, thd).R[0];
                }
            double d = 0.0;
            for (const cd &r : R)
                d = std::max(d, std::abs(r - ref));
            mC = std::max(mC, d);
            QD_DIAG("%.0f,%.0f,%.6f,%.6f,%.6f,%.6f,%.1e\n", f_hz, thd, std::abs(R[0]), std::abs(R[1]), std::abs(R[2]), std::abs(ref), d);
            INFO("C, " << f_hz << " Hz, " << thd << " deg");
            CHECK(d < 1.0e-12);
        }

    QD_DIAG("\n# diag: D, EM type 1 on the peaked board, VV against (zeta c - 1)/(zeta c + 1), HH against -(zeta/c - 1)/(zeta/c + 1)\n");
    QD_DIAG("map,kind,f_Hz,theta,VV_abs,VV_ref_abs,HH_abs,HH_ref_abs,dVV,dHH,RT_VV,RT_HH\n");
    for (const Map &mp : maps)
        for (int kind = 0; kind < 2; ++kind)
        {
            const ub::Pal P = m6t::board(mp.a, mp.mu, kind ? 0.0 : thk, 0.75);
            for (double f_hz : {63.0, 150.0, 250.0})
                for (double thd : {10.0, 30.0, 45.0, 60.0, 75.0, 85.0})
                {
                    const double c = std::cos(thd * QPI / 180.0);
                    const cd z = m6t::zeta(tfP, F, 0.75, f_hz), rv = std::sqrt(1.0 - tf0) * m6t::r_face(z, c), rh = -std::sqrt(1.0 - tf0) * m6t::r_face(z, c, true);
                    bf::Acc W = kind ? m6t::walk(P, {0.0, thk}, {1, 1}, {false, true}, 1, f_hz * r1::FMAP * 1.0e9, thd)
                                     : m6t::walk(P, {0.0}, {1}, {false}, 1, f_hz * r1::FMAP * 1.0e9, thd);
                    const double rt0 = std::norm(W.R[0]) + std::norm(W.T[0]), rt1 = std::norm(W.R[1]) + std::norm(W.T[1]);
                    QD_DIAG("%s,%s,%.0f,%.0f,%.6f,%.6f,%.6f,%.6f,%.1e,%.1e,%.6f,%.6f\n", mp.name, kind ? "X" : "S", f_hz, thd, std::abs(W.R[0]), std::abs(rv),
                            std::abs(W.R[1]), std::abs(rh), std::abs(W.R[0] - rv), std::abs(W.R[1] - rh), rt0, rt1);
                    mD = std::max(mD, std::max(std::abs(W.R[0] - rv), std::abs(W.R[1] - rh))), mDRT = std::max(mDRT, std::max(rt0, rt1));
                    INFO("D " << mp.name << "-parameter " << (kind ? "traced" : "sheet") << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(W.R[0] - rv) < 1.0e-12);
                    CHECK(std::abs(W.R[1] - rh) < 1.0e-12);
                    CHECK(rt0 <= 1.0);
                    CHECK(rt1 <= 1.0);
                }
        }

    QD_DIAG("\n# diag: E, the gate: a wave medium (m 0, eps 2.50, 100 mm traced) with tf 0.05 and the peak keeps the flat pad\n");
    QD_DIAG("type,f_Hz,theta,R_abs,R_deg,ref_abs,ref_deg,T2,T2_ref,R+T\n");
    {
        const double h = 0.1;
        for (double f_hz : {63.0, 150.0})
        {
            std::vector<LumpMtl> lp = {{2.5, 0.0, 0.0, 0.0, tf0}};
            ub::Pal P = lump_palette(lp);
            r1::col(P, "tfP", {tfP}), r1::col(P, "tfF", {F * r1::FMAP}), r1::col(P, "tfQ", {0.75});
            const double x = 0.75 * (f_hz / F - F / f_hz), tf = tf0 + tfP / (1.0 + x * x), f = f_hz * r1::FMAP * 1.0e9, k0 = 2.0 * QPI * f / C0;
            for (int tr : {4, 5})
                for (double thd : {0.0, 30.0, 60.0})
                {
                    // the layer: each face the transformer of the pair rule, R' = (1 - tf) r^2 on the Fresnel r
                    const double c1 = std::cos(thd * QPI / 180.0), s1 = std::sin(thd * QPI / 180.0);
                    const cd n = std::sqrt(cd(2.5, 0.0)), c2 = std::sqrt(1.0 - s1 * s1 / 2.5), r = (c1 - n * c2) / (c1 + n * c2), b = (1.0 - r) / (1.0 + r);
                    const double x0 = std::abs(b), rm = (1.0 - x0) / (1.0 + x0), s = (rm < 0.0 ? -1.0 : 1.0) * std::sqrt(rm * rm * (1.0 - tf)), nu = ((1.0 - s) / (1.0 + s)) / x0;
                    const cd rr = (1.0 - nu * b) / (1.0 + nu * b), p2 = std::exp(cd(0.0, -2.0) * k0 * n * h * c2);
                    const double tt = 1.0 - std::norm(rr);
                    const cd ref = rr - tt * rr * p2 / (1.0 - rr * rr * p2);
                    const double T2ref = std::norm(tt / (1.0 - rr * rr * p2));
                    bf::Acc W = m6t::walk(P, {0.0, h}, {1, 1}, {false, true}, tr, f, thd);
                    const double rt = std::norm(W.R[0]) + std::norm(W.T[0]);
                    QD_DIAG("%d,%.0f,%.0f,%.6f,%+.4f,%.6f,%+.4f,%.6f,%.6f,%.6f\n", tr, f_hz, thd, std::abs(W.R[0]), deg(W.R[0]), std::abs(ref), deg(ref), std::norm(W.T[0]),
                            T2ref, rt);
                    mE = std::max(mE, std::abs(W.R[0] - ref)), mET = std::max(mET, std::abs(std::norm(W.T[0]) - T2ref)), mERT = std::max(mERT, rt);
                    INFO("E, type " << tr << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(W.R[0] - ref) < 8.0e-6);
                    CHECK(std::abs(std::norm(W.T[0]) - T2ref) < 8.0e-6);
                    CHECK(rt <= 1.0 + 3.0e-6); // the traced dielectric reads 1 + 1.15e-6 at most, before M6 as after
                }
        }
    }

    QD_DIAG("\n# diag: F, behind a 10 mm lining (eps 2.50 - 0.80j), glued (H, the plane of the wall face) and traced (X), type 4\n");
    QD_DIAG("kind,f_Hz,theta,R_abs,R_deg,ref_abs,ref_deg,dR\n");
    {
        const double h = 0.01;
        const cd eL(2.5, -0.8);
        for (double f_hz : {100.0, 150.0, 250.0})
        {
            const double f = f_hz * r1::FMAP * 1.0e9, k0 = 2.0 * QPI * f / C0;
            for (int kind = 0; kind < 2; ++kind)
            {
                std::vector<LumpMtl> lp = {{2.5, 0.0, kind ? 0.0 : h}, {maps[0].a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.6644e-04}};
                ub::Pal P = lump_palette(lp);
                r1::col(P, "c", {0.8 * f_hz * r1::FMAP / 17.98, 0.0}), r1::col(P, "tfP", {0.0, tfP}), r1::col(P, "tfF", {0.0, F * r1::FMAP}), r1::col(P, "tfQ", {0.0, 0.75});
                for (double thd : {0.0, 30.0, 60.0, 75.0})
                {
                    const double c = std::cos(thd * QPI / 180.0), s2 = 1.0 - c * c;
                    cd cL = std::sqrt(1.0 - s2 / eL);
                    cL = (std::imag(cL) > 0.0) ? -cL : cL;
                    const cd yL = std::sqrt(eL) * cL, z = m6t::zeta(tfP, F, 0.75, f_hz), rf = (yL - 1.0 / z) / (yL + 1.0 / z), r01 = (c - yL) / (c + yL);
                    const cd ph = std::exp(cd(0.0, -2.0) * k0 * std::sqrt(eL) * cL * h), ref = (r01 + rf * ph) / (1.0 + r01 * rf * ph);
                    cd R = kind ? m6t::walk(P, {0.0, h, h, h + 0.1}, {1, 1, 2, 2}, {false, true, false, true}, 4, f, thd).R[0]
                                : m6t::walk(P, {0.0, 0.0, 0.1}, {1, 2, 2}, {false, false, true}, 4, f, thd).R[0] * std::exp(cd(0.0, -2.0) * k0 * h * c);
                    QD_DIAG("%s,%.0f,%.0f,%.6f,%+.4f,%.6f,%+.4f,%.1e\n", kind ? "X" : "H", f_hz, thd, std::abs(R), deg(R), std::abs(ref), deg(ref), std::abs(R - ref));
                    (kind ? mFX : mFH) = std::max(kind ? mFX : mFH, std::abs(R - ref));
                    INFO("F " << (kind ? "traced" : "glued") << ", " << f_hz << " Hz, " << thd << " deg");
                    CHECK(std::abs(R - ref) < (kind ? 1.0e-4 : 1.0e-12));
                }
            }
        }
    }

    QD_DIAG("\n# diag: G, +-0.05 deg at the critical angle of the two-parameter board (open below it), 150 Hz, type 4\n");
    {
        const double tc = std::asin(std::sqrt(maps[1].a * maps[1].mu)) * 180.0 / QPI, f = 150.0 * r1::FMAP * 1.0e9;
        for (int kind = 0; kind < 2; ++kind)
        {
            const ub::Pal P = m6t::board(maps[1].a, maps[1].mu, kind ? 0.0 : thk, 0.75);
            cd prev(0.0, 0.0);
            for (int k = -50; k <= 50; ++k)
            {
                const double thd = tc + 0.001 * k;
                const cd R = kind ? m6t::walk(P, {0.0, thk}, {1, 1}, {false, true}, 4, f, thd).R[0] : m6t::walk(P, {0.0}, {1}, {false}, 4, f, thd).R[0];
                const cd ref = std::sqrt(1.0 - tf0) * m6t::r_face(m6t::zeta(tfP, F, 0.75, 150.0), std::cos(thd * QPI / 180.0));
                mG = std::max(mG, std::abs(R - ref));
                if (k > -50)
                    mGs = std::max(mGs, std::abs(R - prev));
                prev = R;
            }
        }
        QD_DIAG("critical_deg,max_dR,max_step\n%.4f,%.1e,%.1e\n", tc, mG, mGs);
        CHECK(mG < 1.0e-12);
        CHECK(mGs < 1.0e-6);
    }
    QD_DIAG("# maxima: A %.1e (R+T %.6f), B %.1e, C %.1e, D %.1e (R+T %.6f), E %.1e / %.1e (R+T %.6f), F glued %.1e traced %.1e, G %.1e step %.1e\n", mA,
            mRT, mB, mC, mD, mDRT, mE, mET, mERT, mFH, mFX, mG, mGs);
}

TEST_CASE("Far from the origin - a traced mass-law wall passes what it passes at the origin, its leg measured from the hit point", "[acoustic]")
{
    // Decision 2026-10-19 (md/precision_report.md 0): ray_mesh_interact relaunches a ray 8 float ulp of the hit point's
    // largest coordinate past it, about 1 um per metre, and the leg inside a solid is measured from the hit point all the
    // same: the ledger carries the part the relaunch skips. Before, the leg was short by it, and a 4 mm pane crossed 100 m
    // from the origin was traced as 3.905 mm. The wall here is moved away from the origin with its ray, in its plane and
    // along its normal, to 10 m, 100 m and 1 km: it passes what it passes at the origin, to 1e-5 dB, and that is the plate
    // rule of its true thickness (to 5e-3 dB: the face at exact normal incidence costs 4 mm glass 1.8e-3 dB at 125 Hz).
    // 4 mm glass and 152 mm concrete, limp, side 0 (the infinite plate), the face of each tf 0.05; both acoustic tracers.
    ub::OneThread one_thread;
    const double rho0c = 1.204 * 343.0, tf = 0.05;
    struct M
    {
        const char *name;
        double a, rho, thk;
    };
    const std::vector<M> mats = {{"glass 4 mm", 8.4794e-10, 2500.0, 0.004}, {"concrete 152 mm", 2.0260e-09, 347.0 / 0.152, 0.152}};
    QD_DIAG("\n# diag: a traced mass-law wall moved away from the origin with its ray; T against the plate rule of its true thickness\n");
    QD_DIAG("material,type,f_Hz,theta,shift,along,T_dB,T_ref_dB,dT,dT_origin\n");
    double worst = 0.0, worst0 = 0.0;
    for (const auto &m : mats)
    {
        const double fRef = r1::FMAP * rho0c / (QPI * m.rho);
        std::vector<LumpMtl> lp = {{m.a, 0.0, 0.0, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
        ub::Pal P = side0(lump_palette(lp));
        for (int tr : {4, 5})
            for (double f_hz : {125.0, 500.0, 2000.0})
                for (double thd : {0.0, 45.0, 80.0})
                {
                    double T0 = 0.0; // the wall at the origin
                    for (double R : {0.0, 10.0, 100.0, 1000.0})
                        for (int along : {0, 1})
                        {
                            if (R == 0.0 && along)
                                continue;
                            const double c = std::cos(thd * QPI / 180.0), X = 2.0 * QPI * f_hz * m.rho * m.thk / (2.0 * rho0c);
                            const double Tr = r1::plate(tf, std::pow(10.0, -0.1 * r1::mass_TL(X, c)));
                            bf::Acc A = bf::walk_far(P, {0.0, m.thk}, {1, 1}, {false, true}, tr, f_hz * r1::FMAP * 1.0e9, thd, 0.0, along ? R : 0.0,
                                                     along ? 0.0 : R);
                            const double T = r1::dB(std::norm(A.T[0])), dT = T - r1::dB(Tr);
                            if (R == 0.0)
                                T0 = T;
                            worst = std::max(worst, std::abs(dT)), worst0 = std::max(worst0, std::abs(T - T0));
                            QD_DIAG("%s,%d,%.0f,%.0f,%.0f,%s,%.5f,%.5f,%+.6f,%+.2e\n", m.name, tr, f_hz, thd, R, along ? "normal" : "plane", T, r1::dB(Tr), dT,
                                    T - T0);
                            INFO(m.name << ", type " << tr << ", " << f_hz << " Hz, " << thd << " deg, " << R << " m " << (along ? "along the normal" : "in the plane"));
                            REQUIRE(A.nT == 1);
                            CHECK(std::abs(T - T0) < 1.0e-5);
                            CHECK(std::abs(dT) < 5.0e-3);
                        }
                }
    }
    QD_DIAG("# largest |dT| against the rule %.2e dB, against the origin %.2e dB\n", worst, worst0);
}

TEST_CASE("Far from the origin - in single precision a mass-law leg is not taken for a bent one", "[acoustic]")
{
    // Decision 2026-10-19 (md/precision_report.md 0): ray_state_update tells a leg the tracer bent from one it held by
    // the direction the previous event wrote against the segment's own; the two cosines are compared to 1e-6. In single
    // precision 100 m from the origin, on the short segments a walker relaunches on, rounding alone moves them apart by
    // more: a mass-law leg was taken for a bent one, its plate rule was skipped and the wall passed its two leaks, 26 dB
    // too little. In a mass-law medium the test allows the rounding of the segment's ends (float only). The direct path
    // through 4 mm glass and 152 mm concrete, entry and exit event, in single precision against double, 0 to 85 deg.
    ub::OneThread one_thread;
    const double rho0c = 1.204 * 343.0, tf = 0.05, R = 100.0, f_hz = 500.0, f = f_hz * r1::FMAP * 1.0e9;
    struct M
    {
        const char *name;
        double a, rho, thk;
    };
    const std::vector<M> mats = {{"glass 4 mm", 8.4794e-10, 2500.0, 0.004}, {"concrete 152 mm", 2.0260e-09, 347.0 / 0.152, 0.152}};
    QD_DIAG("\n# diag: the direct path through a mass-law wall 100 m from the origin, float against double, dB\n");
    QD_DIAG("material,type,theta,T_double_dB,T_float_dB,d_dB\n");
    for (const auto &m : mats)
    {
        const double fRef = r1::FMAP * rho0c / (QPI * m.rho);
        std::vector<LumpMtl> lp = {{m.a, 0.0, 0.0, 0.0, tf, 0.0, 0.0, 0.0, 0.0, 20.0, fRef}};
        ub::Pal P = side0(lump_palette(lp));
        arma::uvec mi;
        arma::mat mesh = ub::build({0.0, m.thk}, {1, 1}, {false, true}, mi);
        for (arma::uword r = 0; r < mesh.n_rows; ++r)
            for (int v = 0; v < 3; ++v)
                mesh(r, 3 * v + 1) += R;
        for (int tr : {4, 5})
            for (int i = 0; i <= 17; ++i)
            {
                const double thd = 5.0 * i, t = std::tan(thd * QPI / 180.0), dl = std::max(std::min(1.0, 50.0 / std::max(t, 1.0e-12)), m.thk + 0.05);
                const double y0 = R + 0.37 - 0.5 * m.thk * t;
                ub::Opt op;
                op.relaunch = true;
                auto path = [&](bool single)
                {
                    arma::mat o = {{-dl, y0 - dl * t, 0.0}}, d = {{dl, y0 + dl * t, 0.0}}, o2, d2;
                    ub::Ev E0 = single ? ub::traced<float>(P, mesh, mi, tr, f, o, d, nullptr, op) : ub::traced<double>(P, mesh, mi, tr, f, o, d, nullptr, op);
                    ub::onward(E0, o2, d2);
                    ub::Ev E1 = single ? ub::traced<float>(P, mesh, mi, tr, f, o2, d2, &E0, op) : ub::traced<double>(P, mesh, mi, tr, f, o2, d2, &E0, op);
                    return r1::dB(std::norm(cpx(E0.xpr, 0, 0) * cpx(E1.xpr, 0, 0)));
                };
                const double Td = path(false), Tf = path(true);
                QD_DIAG("%s,%d,%.0f,%.4f,%.4f,%+.4f\n", m.name, tr, thd, Td, Tf, Tf - Td);
                INFO(m.name << ", type " << tr << ", " << thd << " deg");
                CHECK(std::abs(Tf - Td) < 0.05);
            }
    }
}
