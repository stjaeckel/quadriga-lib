// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material cases of the floors group (finishes, carpets, underlays); the rig is acoustic_material_rig.hpp.
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

#include "acoustic_material_rig.hpp"

TEST_CASE("Material - parquet + vinyl", "[acoustic][materials]")
{
    // Hard floor finishes glued to the slab, class 1: the finish solids `parquet` and `vinyl` (thk 0) at 15 and 3 mm,
    // the sheets parquet_15mm and vinyl_3mm; att 0 and the host's side 3.22 m. Columns: e the density, 710 kg/m3 for
    // parquet (birch plywood [Hopkins_2007_SoundInsulation#153] standing in for hardwood; a from its quasi-longitudinal
    // speed 3850 m/s, [Hopkins_2007_SoundInsulation#156]) and 1400 kg/m3 for vinyl (stated, no measured row; a from
    // 1300 m/s); fRef the physical mass level; tf and tfB the power laws of PTB #394 (0.0583 at 1 kHz, exponent 0.159)
    // and PTB #601 (0.0466 at 1 kHz, exponent 0.454). No coiF or res*: a glued finish bends with the slab, and the
    // plate rule adds each bonded body's own coincidence factor, so a parquet with its own coincidence (1122 Hz at
    // 15 mm) would add 3.8 dB at 4 kHz to the slab where its mass adds 0.27 dB.
    // Absorption, hard faces used as measured, within 0.05: PTB #394 "parquet fixed in asphalt, on concrete"
    // [PTB_2012_AbsorptionTable#48] and PTB #601 "rubber, cork, linoleum+underlay or vinyl+underlay stuck to
    // concrete" [PTB_2012_AbsorptionTable#73] (Sabine; their 63 Hz and 8 kHz are copies, dropped). Printed: the
    // compilations [Cox_2016_AbsorbersDiffusers#100] (the values of PTB #394) and [Cox_2016_AbsorbersDiffusers#72]
    // (linoleum or vinyl stuck to concrete), the room estimates [Aspoeck_2020_BRAS#58] and [Bork_2000_RR2#8], and
    // parquet on a counterfloor [PTB_2012_AbsorptionTable#49] (another build, 0.20 at 125 Hz). On the slab the room
    // side reads the finish's face alone.
    // On concrete 150 and 200 mm, as a finish solid (two bonded traced bodies) and as a sheet on the slab solid: per
    // node against the bonded plate (measured 2.7e-3 dB for the solid, 3.1e-6 dB for the sheet), the two forms equal,
    // and the insulation the finish adds within 1 dB of the limp mass law of the physical masses (side 3.22 m). The
    // concrete row's mass level stands for 0.53 times its physical mass (heavy-wall coincidence engine limit), so the
    // finish adds up to 0.43 dB more than its mass law (parquet, 4 kHz, 150 mm), printed. Valid: alpha 125 Hz - 4 kHz.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000};
    auto ptb = [&](const char *key, const std::vector<double> &v)
    {
        mt::Measured M{key, "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct, {}};
        for (double x : v)
            M.lo_hi.push_back({x, x});
        return M;
    };
    struct Finish { const char *name, *sheet; double t; mt::Measured ref; std::vector<std::pair<const char *, std::vector<double>>> printed; };
    const std::vector<Finish> F = {
        {"parquet", "parquet_15mm", 0.015, ptb("PTB_2012_AbsorptionTable #394", {0.04, 0.04, 0.07, 0.06, 0.06, 0.07}),
         {{"cox100", {0.04, 0.04, 0.07, 0.06, 0.06, 0.07}}, {"bras58", {0.058, 0.036, 0.033, 0.048, 0.087, 0.101}}, {"ptb395", {0.2, 0.15, 0.1, 0.1, 0.05, 0.1}}}},
        {"vinyl", "vinyl_3mm", 0.003, ptb("PTB_2012_AbsorptionTable #601", {0.02, 0.02, 0.04, 0.05, 0.05, 0.10}),
         {{"cox72", {0.02, 0.02, 0.03, 0.04, 0.04, 0.05}}, {"bork8", {0.02, 0.03, 0.03, 0.04, 0.06, 0.05}}}}};
    const mt::Row c = mt::row("concrete");
    const mt::Grid G = mt::grid();
    for (const Finish &f : F)
    {
        const mt::Row r = mt::row(f.name), rs = mt::row(f.sheet);
        const mt::Out X = mt::check_material(std::string(f.name) + " at " + std::to_string((int)std::lround(1e3 * f.t)) + " mm", r, {f.ref}, G, f.t);
        mt::check_material(f.sheet, rs);
        QD_DIAG("# diag: %s, absorption against the printed references\noctave,library", f.name);
        for (const auto &p : f.printed)
            QD_DIAG(",%s", p.first);
        for (int o = 0; o < 6; ++o)
        {
            QD_DIAG("\n%g,%.4f", oct[o], X.a8[o + 2]);
            for (const auto &p : f.printed)
                QD_DIAG(",%.3f", p.second[o]);
        }
        QD_DIAG("\n");
        for (double tc : {0.150, 0.200})
        {
            const std::string on = " | concrete " + std::to_string((int)std::lround(1e3 * tc)) + " mm";
            const mt::Out B = mt::run({{c, mt::SOLID, tc}}, G);
            const mt::Out A = mt::check_stack((f.name + std::string(" solid") + on).c_str(), {{r, mt::SOLID, f.t}, {c, mt::SOLID, tc}}, 8e-3, 2e-7, G);
            const mt::Out S = mt::check_stack((f.sheet + on).c_str(), {{rs, mt::SHEET, f.t}, {c, mt::SOLID, tc}}, 9e-6, 2e-7, G);
            double dT = 0.0, dR = 0.0; // the finish solid against the sheet, per node
            for (size_t q = 0; q < G.f.size(); ++q)
                for (size_t i = 0; i < G.th.size(); ++i)
                    dT = std::max(dT, std::abs(mt::dB(std::norm(A.T[i][q])) - mt::dB(std::norm(S.T[i][q])))),
                    dR = std::max(dR, std::abs(std::norm(A.R[i][q]) - std::norm(S.R[i][q])));
            // the limp mass law of the physical masses at random incidence (Davy's sigma at the slab's side)
            auto limp = [&](double m)
            {
                return mt::octaves(G, [&](size_t q)
                                   { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * std::pow(10.0, -0.1 * mt::mass_TL(QPI * G.f[q] * m / (mt::RHO0 * mt::CS), std::cos(mt::rad(G.th[i])), G.f[q], mt::col(c, "side"))); return t; });
            };
            const double mc = mt::mpp(c, tc), mf = mt::mpp(r, f.t);
            const auto l0 = limp(mc), l1 = limp(mc + mf);
            QD_DIAG("# diag: %s%s: solid against sheet per node dT %.2e dB, dR %.2e; the insulation added (dB), %.2f kg/m2 on %.1f kg/m2\n"
                    "octave,library_solid,library_sheet,mass_law,alpha_room_side\n", f.name, on.c_str(), dT, dR, mf, mc);
            INFO(f.name << on);
            CHECK(dT < 8e-3);
            CHECK(dR < 1.8e-8);
            for (int o = 0; o < 9; ++o)
            {
                const double ml = 10.0 * std::log10(l0[o] / l1[o]), la = A.t8[o] - B.t8[o];
                QD_DIAG("%g,%.3f,%.3f,%.3f,%.4f\n", mt::OCT[o], la, S.t8[o] - B.t8[o], ml, A.a8[o]);
                CHECK(std::abs(la - ml) <= 1.0);
            }
        }
    }
}

TEST_CASE("Material - carpet_tile", "[acoustic][materials]")
{
    // Carpet tiles and low-pile rugs, class 3: the sheet carpet_tile_11mm, the porous pile of a hard-backed tile; the
    // dense backing acts as the hard floor. A porous sheet keeps its true thickness, so thk is the pile depth (6.5 mm),
    // the name the tile a modeler sees. Columns: power laws of Miki's model [Miki_1990_ModificationsDelanyBazley] at
    // 16469 Pa s/m2, fitted at 342.76271 m/s on the worst random-incidence absorption miss over 63 Hz - 8 kHz at 4, 6.5
    // and 11 mm on a rigid backing (0.0065), with the 0 deg transmission loss of 6.5 mm in air within 0.2 dB over 63 Hz
    // - 4 kHz. Resistivity and depth are fitted to the chamber values of an 11 mm tile
    // [Salford_2020_CarpetTileSpecimen#1] (12.04 m2; no backing, pile or mass stated) by Jeong's conversion
    // (tools/acoustic/sabine_to_random.py with the depth free, 3 x 4 m, extended reaction; finite-sample fit within
    // 0.036; at the full 11 mm no resistivity fits better than 0.053). No resistivity of this tile is measured; the
    // fitted one lies inside measured pile resistivities: rug piles alone 11797-47604 Pa s/m2
    // [Ardekani_2022_PersianRug#4] to #6, piles fitted in complete rugs 10003-40162 (#7 to #12), car carpet piles 20000
    // and 30000 [Aretz_2014_CarCompartmentPartI#14], #15. Absorption of the sheet on the concrete slab: 63 Hz within
    // 0.05 of 0.00-0.01, the four neighbour builds [Milliken_ND_CarpetTileCertificates#1],
    // [TFI_2018_NylonCarpetTileCushion#1], [Kiwa_2021_EForceComfortBack#1], [Bietz_2025_CombiningRoomVolumes#2] (a hard
    // face there); 125-500 Hz within 0.05 of Salford as measured (a hard face, 0.03-0.09); 1-4 kHz within 0.107 (0.05
    // plus the conversion's maximum error) of Salford converted to random incidence (0.198 / 0.337 / 0.489). The
    // conversion lowers 1 and 2 kHz by 0.092 and 0.063, so the raw values are printed (library 0.081 below at 1 kHz),
    // and the conversion with local reaction at 11 mm (library within 0.037). Printed: a hard-backed tile within 0.05
    // (Milliken #1); cushion backs act as a mass on a spring, up to 0.40 at 500 Hz against the row's 0.10 (Kiwa #1, TFI
    // #1, [BCTC_2014_EcosoftCarpetTile#1]); denser 13-15 mm rugs absorb more above 1 kHz (Ardekani #20 and #13, normal
    // incidence, against the library at 0 deg); the rug on the finish solids parquet and vinyl absorbs 0.003-0.030 more
    // than on the slab (63 Hz - 4 kHz; the finish's own face behind the pile) and changes the floor's transmission by
    // -0.69 to +0.57 dB.
    // Reasonable meant: within the converted chamber values of one tile and the hard-face values of its neighbours.
    // Tier A 1 to 500 Hz, 2 above. Valid: hard-backed tiles and low-pile rugs, absorption 63 Hz - 4 kHz.
    const mt::Grid G = mt::grid();
    const mt::Row r = mt::row("carpet_tile_11mm"), c = mt::row("concrete"), w0 = mt::with(c, "tf", 0.0);
    mt::check_material("carpet_tile_11mm", r);

    // The row against its model on a rigid backing (random incidence, 63 Hz - 8 kHz; measured 0.0065, bound 0.02)
    auto alpha = [&](const mt::Row &m, double d) { return mt::octaves(G, [&](size_t q) { double a = 0.0; for (int i = 0; i < G.na; ++i) { double T, R; mt::ref_tmm({{m, mt::SHEET, d}}, G.f[q], G.th[i], T, R, mt::HALF, &w0); a += G.w[i] * (1.0 - R); } return a; }); };
    for (double d : {0.004, 0.0065, 0.011})
    {
        const auto ar = alpha(r, d), am = alpha({{"miki", 16469.0}}, d);
        QD_DIAG("\n# diag: carpet_tile, %.1f mm pile on a rigid backing, random incidence\noctave,alpha_row,alpha_miki\n", 1e3 * d);
        double worst = 0.0;
        for (int o = 1; o < 9; ++o)
            QD_DIAG("%g,%.4f,%.4f\n", mt::OCT[o], ar[o], am[o]), worst = std::max(worst, std::isnan(ar[o] - am[o]) ? INFINITY : std::abs(ar[o] - am[o]));
        INFO("carpet_tile against Miki at " << 1e3 * d << " mm");
        CHECK(worst < 0.02);
    }

    // Tier 2: the sheet on the concrete slab (150 mm solid) through the library
    const mt::Measured n63{"Milliken #1, TFI #1, Kiwa #1, Bietz #2", "alpha_random", "Sabine, hard face, as measured", false, 0.05, {63}, {{0.00, 0.01}}};
    const mt::Measured sal{"Salford_2020_CarpetTileSpecimen #1", "alpha_random", "Sabine, hard face, as measured", false, 0.05, {125, 250, 500},
                           {{0.03, 0.03}, {0.04, 0.04}, {0.09, 0.09}}};
    const mt::Measured conv{"Salford_2020_CarpetTileSpecimen #1", "alpha_random", "Sabine converted to random incidence (Jeong, 3 x 4 m, Miki 6.5 mm)", false, 0.107,
                            {1000, 2000, 4000}, {{0.198, 0.198}, {0.337, 0.337}, {0.489, 0.489}}};
    const mt::Stack on = {{r, mt::SHEET, 0.0065}, {c, mt::SOLID, 0.15}};
    const mt::Out O = mt::run(on, G);
    mt::rows("carpet_tile_11mm on concrete 150 mm", O);
    for (const mt::Measured *M : {&n63, &sal, &conv})
        mt::tier2("carpet_tile_11mm on concrete 150 mm", O, *M, 0.0);
    const auto a0 = mt::octaves(G, [&](size_t q) { return 1.0 - std::norm(O.R[G.na][q]); });
    const double ref[8][6] = {{0.03, 0.04, 0.09, 0.29, 0.40, 0.50},    {0.012, 0.033, 0.082, 0.173, 0.324, 0.538}, {0.02, 0.04, 0.08, 0.17, 0.36, 0.49},
                              {0.01, 0.06, 0.30, 0.25, 0.38, 0.53},    {-0.01, 0.06, 0.32, 0.20, 0.26, 0.37},      {0.00, 0.05, 0.40, 0.30, 0.30, 0.40},
                              {0.01, 0.02, 0.05, 0.13, 0.35, 0.69},    {0.02, 0.04, 0.13, 0.31, 0.68, 0.88}};
    QD_DIAG("\n# diag: carpet_tile printed: library random incidence and 0 deg against Salford raw (Sabine) and converted with local reaction "
            "(11 mm), Sabine values of other builds, normal-incidence values of rugs\noctave,library,library_0deg,salford_raw,salford_local,milliken1,kiwa1,tfi1,"
            "bctc1,ardekani20,ardekani13\n");
    for (int o = 0; o < 6; ++o)
    {
        QD_DIAG("%g,%.4f,%.4f", mt::OCT[o + 2], O.a8[o + 2], a0[o + 2]);
        for (int k = 0; k < 8; ++k)
            QD_DIAG(",%.3f", ref[k][o]);
        QD_DIAG("\n");
    }

    // Printed: the rug on the finish solids parquet (15 mm) and vinyl (3 mm) on the slab, against the rug on the slab
    for (const auto &[fn, ft] : std::vector<std::pair<const char *, double>>{{"parquet", 0.015}, {"vinyl", 0.003}})
    {
        const mt::Row fin = mt::row(fn);
        const mt::Out F = mt::run({{r, mt::SHEET, 0.0065}, {fin, mt::SOLID, ft}, {c, mt::SOLID, 0.15}}, G), B = mt::run({{fin, mt::SOLID, ft}, {c, mt::SOLID, 0.15}}, G);
        QD_DIAG("\n# diag: carpet_tile on %s on the slab, minus on the slab (alpha, tl in dB); tl the rug adds to the finished floor\noctave,d_alpha,d_tl,added_tl\n", fn);
        for (int o = 0; o < 9; ++o)
            QD_DIAG("%g,%+.4f,%+.2f,%+.2f\n", mt::OCT[o], F.a8[o] - O.a8[o], F.t8[o] - O.t8[o], F.t8[o] - B.t8[o]);
    }

    // ENGINE LIMIT (porous layer on a rigid wall): the pile adds too much insulation to the slab at 0 deg; printed
    // against the fluid layer on a limp plate, the layer alone in air asserted.
    mt::added_insulation("carpet_tile_11mm to concrete 150 mm", {{r, mt::SHEET, 0.0065}}, {{c, mt::SOLID, 0.15}}, G);
}
