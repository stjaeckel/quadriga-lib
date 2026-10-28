// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_RAY=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

// The rig (namespace mt) is in acoustic_material_rig.hpp; this file holds its self-checks and the table case.

#include "acoustic_material_rig.hpp"

#include <catch2/catch_test_case_info.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

#include <cctype>
#include <regex>
#include <set>

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
    // Hopkins [Hopkins_2007_SoundInsulation#68]: dense aggregate blocks 215 mm and 13 mm plaster on both faces,
    // 430 kg/m2, specimen 3.53 x 2.63 m
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
    // on concrete: the pad is formed on the medium in front, up to 6.3e-3), or the bonded plate (one side).
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
    // ENGINE LIMIT (porous layer on a rigid wall): it adds too much insulation at low frequency.
    using namespace mt_rows;
    mt::added_insulation("50 mm glass wool to 152 mm concrete", {{WOOL, mt::SHEET, 0.05}}, {{CONCRETE152, mt::SOLID, 0.152}});
}

TEST_CASE("Materials rig - engine limits printed: a filled box on one path, wave layers before a pane, three bonded bodies", "[acoustic][materials]")
{
    // Why: known engine limits, printed as measured (no assertion; the same
    // constructions in their working forms are asserted in the combinations case).
    using namespace mt_rows;
    using mt::SHEET;
    using mt::SOLID;
    const mt::Grid G = mt::grid(), G12 = mt::grid(12);
    QD_DIAG("\n# diag: engine limits per node against the fluid layers or the bonded plate\n%s", mt::DEV_HEAD);
    // ENGINE LIMIT (wave media back to back): a leaf behind a filled box (the precise drywall recipe). Its T is the
    // exchange between box and fill, a series of paths: through the paths of the reflection ray set at most 10
    // interactions (5.0 dB off the fluid layers), on one straight path (calc_diffraction_gain, as the scene tracer's
    // direct path through a wall: 9.5 dB); R uncapped (1.45e-2) and at 10 interactions (0.22).
    const mt::Stack dw = {{BOARD, SHEET, 0.0125}, {AIRBOX, SOLID, 0.025}, {ROCKWOOL, SOLID, 0.05}, {BOARD, SHEET, 0.0125}};
    mt::Out O = mt::run(dw, G12);
    mt::against("ENGINE LIMIT (wave media back to back): board | air | rock wool | board, T from the paths (10 interactions), R uncapped", O, dw);
    mt::rows("ENGINE LIMIT (wave media back to back): board | air | rock wool | board", O);
    O.T = mt::transmit(mt::scene(dw), G12), O.R = mt::reflect(mt::scene(dw), G12, O.calls, O.rays, nullptr, 10);
    mt::against("ENGINE LIMIT (wave media back to back): board | air | rock wool | board, T on one straight path, R at 10 interactions", O, dw);
    // ENGINE LIMIT (three bonded bodies): plaster | block | plaster with the far plaster traced (the exit-sheet form is
    // exact): the state is one level deep. Random-incidence tl, library against the bonded plate of the three, and
    // Hopkins measured [Hopkins_2007_SoundInsulation#68] (the probe rows are physical, not fitted to it).
    const mt::Stack pb = {{PLASTER, SHEET, 0.013}, {BLOCK, SOLID, 0.215}, {PLASTER, SOLID, 0.013}};
    const mt::Out P = mt::run(pb, G);
    mt::against("ENGINE LIMIT (three bonded bodies): plaster | block 215 mm | plaster traced", P, pb);
    const auto pl = mt::octaves(G, [&](size_t q) { double t = 0.0; for (int i = 0; i < G.na; ++i) { double T, R; mt::ref_plate(pb, G.f[q], G.th[i], T, R); t += G.w[i] * T; } return t; });
    const double H448[9] = {NAN, 35.4, 39.6, 42.7, 52.2, 58.3, 65.8, 70.6, NAN};
    QD_DIAG("# diag: ENGINE LIMIT (three bonded bodies), random incidence tl (dB)\noctave,library,bonded_plate,hopkins_448\n");
    for (int o = 0; o < 9; ++o)
        QD_DIAG("%g,%.2f,%.2f,%.1f\n", mt::OCT[o], P.t8[o], -10.0 * std::log10(pl[o]), H448[o]);
    // ENGINE LIMIT (porous layer on a rigid wall): a curtain (rock-wool columns, 3 mm) on 100 mm of air before a 4 mm
    // pane: the insulation the curtain and the box add, library against the fluid layers on a limp plate of the pane's
    // mass (infinite and finite at the pane's side) and against the layers alone in air (the series estimate)
    mt::added_insulation("a curtain on 100 mm of air to glass 4 mm", {{ROCKWOOL, SHEET, 0.003}, {AIRBOX, SOLID, 0.1}}, {{GLASS4, SHEET, 0.004}}, G, 2e-4, 2e-6);
}

TEST_CASE("Materials rig - tier 2 at the specimen's side and mass, the class envelope", "[acoustic][materials]")
{
    // Why: a reference is evaluated at its specimen's side and shifted by the mass law to the row's surface mass
    // (e rho0 thk: 346.7 kg/m2 for these concrete columns at 152 mm, as Litvin's wall); a class row against the envelope
    // of its walls, each shifted, within 1 dB. The concrete columns of page 11.4 as a solid row at 150 mm against Litvin
    // 76-66 at its laboratory side 3.523 m (shift -0.114 dB; the row at 3.5229 m, so transmission is rerun) and the envelope of 76-66 and 76-77 (152 and 203 mm,
    // [Litvin_1980_ConcreteWallsTL#1], [Litvin_1980_ConcreteWallsTL#2]).
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
    // Why: lumped rows carry the visible thickness: thk and fRef times k with every power law rescaled leaves every
    // column unchanged at every frequency. Double glazing 4-16-4 of page 11.4 at 8 and 24 mm (k = 3; fRef alone would
    // move tf at 125 Hz from 0.1124 to 0.1670), sheet and solid, per node.
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
    // Why: objects take a per-object absorption area measured in a reverberation chamber; in a diffuse
    // field it is 4 x the mean absorption cross-section. 16 directions x about 65 rays, the octave centers, at most 10
    // interactions, 3 reflections, -60 dB: within 1.3 % of 64 directions x 2000 rays x 18 thirds (a probe, three
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
    // ENGINE LIMIT (grazing relaunch): a ray within 0.2 deg of grazing a box face re-hits the face at its launch point
    // at every call, splits and gains power. The ray found on 64 directions (the hard box, direction 36), alone; rays and
    // their power per call printed.
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
    QD_DIAG("# diag: ENGINE LIMIT (grazing relaunch), one grazing ray on the hard box\ncall,rays,power\n");
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
    // family has one case "Material - <family>[ + <family> ...]" and every family a case names has a row; the suffix
    // is thk in mm, except that a porous row (a lossy wave medium, m 0) may carry less: a porous sheet keeps its true
    // thickness, which can be less than the product's (a carpet's pile on its backing); all 26 columns; fRef, a, e
    // positive, the columns the library requires non-negative; lowercase snake_case names.
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
        std::smatch mm; // the suffix in mm (an underscore for the decimal point); no suffix, thk 0
        fams.insert(f);
        INFO("row " << n << ", family " << f);
        CHECK(cases.count(f) == 1);
        const double sfx = std::regex_search(n, mm, std::regex("_([0-9]+)(_([0-9]+))?mm$")) ? std::stod(mm[1].str() + "." + (mm[3].matched ? mm[3].str() : "0")) : 0.0,
                     thk = 1000.0 * T.prop.at("thk")[i];
        const bool porous = T.prop.at("m")[i] == 0.0 && (T.prop.at("c")[i] > 0.0 || T.prop.at("g")[i] > 0.0);
        CHECK((porous && sfx > 0.0 ? thk > 0.0 && thk <= sfx + 1e-9 : std::abs(thk - sfx) < 1e-9));
        CHECK(std::all_of(n.begin(), n.end(), [](char ch) { return std::islower((unsigned char)ch) || std::isdigit((unsigned char)ch) || ch == '_'; }));
        for (const char *k : {"fRef", "a", "e"})
            CHECK(T.prop.at(k)[i] > 0.0);
        for (const char *k : {"c", "g", "thk", "att", "alpha", "m", "resF", "resQ", "coiF", "coiQ", "tfF", "side"})
            CHECK(T.prop.at(k)[i] >= 0.0);
    }
    for (const std::string &c : cases)
        CHECK(fams.count(c) == 1);
}
