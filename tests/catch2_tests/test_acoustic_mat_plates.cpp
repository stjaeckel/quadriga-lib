// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material cases of the thin-plate group (glass, glazing, boards, metal, doors); the rig is acoustic_material_rig.hpp.
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

#include "acoustic_material_rig.hpp"

TEST_CASE("Material - glass", "[acoustic][materials]")
{
    // Single-pane float glass, class 1: the sheets glass_4mm and glass_6mm (side 1.30 m, a pane of 1.2 x 1.4 m).
    // Columns: density 2500 kg/m3 in e [Hopkins_2007_SoundInsulation#85]; the coincidence is physical, coiF = fc h / h
    // with fc h = 12.5 Hz m at 343 m/s (12.48 at 342.76 m/s; 3121 Hz at 4 mm, 2080 Hz at 6 mm)
    // [Hopkins_2007_SoundInsulation#84] and coiQ = 1 / 0.005 from the internal loss factor 0.003-0.006
    // [Hopkins_2007_SoundInsulation#86]; fRef, att and res* fitted at 342.76271 m/s to the laboratory values below (4096
    // angles x 12 tones per third); tf and tfB the power law of PTB #207 (0.0285 at 1 kHz, exponent -0.371), a 3 mm pane
    // [PTB_2012_AbsorptionTable#11] (Sabine, a neighbour build; its 63 Hz and 8 kHz are copies), printed. The face
    // absorption caps the transmission at every angle, so it also sets the depth of the coincidence dip (an engine
    // limit, "pane coincidence capped by the face absorption"; the laboratory values are met with the measured absorption).
    // Transmission, each within 3 dB at its specimen's side and mass-shifted: Quirt [Quirt_1981_WindowsTLMeasurements#24]
    // (4 mm, 10.4 kg/m2) and [Quirt_1981_WindowsTLMeasurements#25] (6 mm, 14.4 kg/m2), ASTM E90, octaves the energy mean
    // of thirds, panes 0.56 x 1.68 m (side 0.970 m); AGC [AGC_2023_PocketStratophone#11] (6 mm, 15 kg/m2, EN ISO 10140,
    // 1.23 x 1.48 m, side 1.349 m); at 6 mm their envelope within 1 dB; 31.5, 63 Hz and 8 kHz take the spread of physical
    // models at 1.349 m (tools/acoustic/tl_refs.json). The grid has 200 angles: above the coincidence the 8 kHz octave
    // reads up to 3 dB off on 50.
    // Reasonable meant: within the laboratory values and the physical plate (within 0.9 dB of Quirt at 6 mm). The
    // EN 12758 generic values [Pilkington_2020_OptiphonBrochure#3], [Pilkington_2020_OptiphonBrochure#4] (conservative
    // design values, size not stated) are printed: the row reads up to 4.3 dB above them at 4 mm; Quirt's panes read
    // 4-5 dB above them at 125 and 250 Hz after mass and side, of which about 1 dB is the pane size. Valid: R 125 Hz - 4 kHz.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, Q4 = {23.6, 25.6, 29.7, 33.6, 32.6, 28.0}, Q6 = {22.9, 26.5, 30.7, 34.6, 29.4, 32.9},
                              A6 = {19.5, 23.4, 29.5, 35.5, 27.6, 31.6}, P[2] = {{17, 20, 26, 32, 33, 26}, {18, 23, 30, 35, 27, 32}};
    const std::vector<std::pair<double, double>> gap[2] = {{{5.1, 18.6}, {9.3, 16.0}, {37.0, 37.2}}, {{7.4, 18.8}, {12.3, 20.5}, {43.0, 43.2}}};
    auto lab = [&](const char *key, const char *type, const std::vector<double> &v, double m, double side, const std::vector<std::pair<double, double>> &gaps)
    {
        mt::Measured M{key, "R_dB", type, true, 3.0, {}, {}, side, m};
        for (size_t k = 0; k < v.size(); ++k)
            M.octaves_hz.push_back(oct[k]), M.lo_hi.push_back({v[k], v[k]});
        for (size_t k = 0; k < gaps.size(); ++k) // 31.5 Hz, 63 Hz, 8 kHz
            M.octaves_hz.push_back(k < 2 ? 31.5 * (1 + k) : 8000), M.lo_hi.push_back(gaps[k]);
        return M;
    };
    const char *astm = "laboratory sound reduction index (ASTM E90), octave = energy mean of thirds";
    const mt::Row g4 = mt::row("glass_4mm"), g6 = mt::row("glass_6mm");
    const mt::Out X4 = mt::check_material("glass_4mm", g4, {lab("Quirt 4 mm", astm, Q4, 10.4, 0.970, {}), lab("model spread", "physical models", {}, 0.0, 1.349, gap[0])},
                                          mt::grid(200, 1));
    const mt::Out X6 = mt::check_material("glass_6mm", g6,
                                          {lab("Quirt 6 mm", astm, Q6, 14.4, 0.970, {}), lab("AGC 6 mm", "manufacturer (EN ISO 10140)", A6, 15.0, 1.349, gap[1]),
                                           mt::envelope("Quirt 6 mm and AGC 6 mm", oct, {{14.4, Q6}, {15.0, A6}}, mt::mpp(g6, 0.006), 1.349)},
                                          mt::grid(200, 1));
    const mt::Out *X[2] = {&X4, &X6};
    const double ptb[6] = {0.08, 0.04, 0.03, 0.03, 0.02, 0.02};
    for (int k = 0; k < 2; ++k)
    {
        QD_DIAG("# diag: glass_%dmm (row at its side) against the EN 12758 generic values and its absorption against PTB #207, printed\noctave,library_tl,generic,library_alpha,ptb207\n", 4 + 2 * k);
        for (int o = 0; o < 6; ++o)
            QD_DIAG("%g,%.2f,%.0f,%.4f,%.2f\n", oct[o], X[k]->t8[o + 2], P[k][o], X[k]->a8[o + 2], ptb[o]);
    }
}

TEST_CASE("Material - door_solid + door_hollow_core", "[acoustic][materials]")
{
    // Interior doors, class 1: the lumped sheets door_solid_40mm and door_hollow_core_40mm (side 1.34 m, a door of
    // 0.9 x 2.0 m). Lumped: thk is the visible 40 mm, e = m'' / (rho0 thk) carries the stated surface mass at 40 mm
    // (23.9 and 9.3 kg/m2), and a 44 mm leaf is the same row rescaled with fRef (asserted below); a from the
    // quasi-longitudinal speeds of chipboard and MDF [Hopkins_2007_SoundInsulation#29], [Hopkins_2007_SoundInsulation#118].
    // fRef (the mass level), att and the res* and coi* knobs fitted at 342.76271 m/s to the references below (2048 angles
    // x 6 tones per third and this case's grid). The mass levels are 0.74 and 0.69 times the stated masses.
    // Solid door: coiF (2.68 kHz) and coiQ are fit knobs with a resS +1 dip at 817 Hz; the physical plate (chipboard
    // fc h 29.5 Hz m, 655 Hz at 45 mm, eta 0.01, [Hopkins_2007_SoundInsulation#25], [Hopkins_2007_SoundInsulation#27])
    // gives a dip 6-9 dB deeper than the measured plateau of 27-32 dB over 630 Hz - 2.5 kHz (heavy-wall coincidence
    // engine limit, extended to timber leaves). Hollow-core door: resS +1 at the physical mass-air-mass frequency of two
    // 4.5 mm MDF faces (760 kg/m3 [Hopkins_2007_SoundInsulation#115]) over 26 mm of air, 283 Hz at normal incidence
    // (random-incidence minimum 306 Hz; Kim's 315 Hz third within 1.4 dB); coiF the faces' coincidence (fc h 25.3 Hz m
    // [Hopkins_2007_SoundInsulation#114], 5.61 kHz, eta 0.01 [Hopkins_2007_SoundInsulation#116]); att rises with
    // frequency in place of the double leaf's steeper slope above the resonance. tf and tfB the power law of PTB #396
    // [PTB_2012_AbsorptionTable#50] (solid wooden door, Sabine, a hard face used as measured; 0.0917 at 1 kHz, exponent
    // -0.0575; its 63 Hz and 8 kHz are copies), both rows.
    // Transmission of sealed leaves, each within 3 dB at its specimen's side and mass-shifted: Quirt
    // [Quirt_1986_DoubleDoors#1] (45 mm solid core, hardboard faces, fully sealed, 23.9 kg/m2, leaf 0.914 x 2.134 m, side
    // 1.397 m; ASTM E90, octaves the energy mean of thirds read from a figure) and Morin [Morin_1993_AccessDoorsCMHC#2]
    // (44 mm, 25.2 kg/m2, caulked, sound intensity, opening 1.010 x 2.017 m, side 1.427 m), their envelope within 1 dB;
    // Kim [Lee_2010_InteriorDoorAperture#1] (35 mm lattice core, 4.5 mm MDF faces, all joints sealed, KS F 2808 = ISO
    // 140-3, leaf 0.875 x 2.215 m, side 1.392 m, mass not stated: no shift). 31.5, 63 Hz and 8 kHz take the spread of
    // three physical models at the specimen's side (limp plate with Davy's sigma, field incidence 0-78 deg, the plate
    // with the physical coincidence).
    // Reasonable meant: within the sealed laboratory leaves; printed only, the row at its side: NBS 632
    // [Berendt_1967_NBSMonograph77#14] (warble tones; 6.8 dB above Quirt at 125 Hz after the mass shift), the gasketed
    // state Morin [Morin_1993_AccessDoorsCMHC#1] (2.3-4 dB below the caulked one), NBS 634 / 633
    // [Berendt_1967_NBSMonograph77#16], [Berendt_1967_NBSMonograph77#15] (caulked / with a bottom gap, 11-24 dB below
    // the row from 1 kHz). Valid: R 125 Hz - 4 kHz, sealed leaf.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, oct4 = {250, 500, 1000, 2000}, Q = {23.7, 29.3, 31.6, 28.0, 30.1, 35.3},
                              M2 = {27.5, 31.6, 30.3, 29.9}, K = {20.7, 18.9, 24.6, 32.3, 37.5, 43.8};
    auto lab = [](const char *key, const char *type, const std::vector<double> &o, const std::vector<double> &v, double side, double m, const std::vector<double> &gaps)
    {
        mt::Measured M{key, "R_dB", type, true, 3.0, {}, {}, side, m};
        for (size_t k = 0; k < gaps.size(); k += 2) // 31.5 Hz, 63 Hz, 8 kHz
            M.octaves_hz.push_back(k < 4 ? 31.5 * (1 + k / 2) : 8000), M.lo_hi.push_back({gaps[k], gaps[k + 1]});
        for (size_t k = 0; k < v.size(); ++k)
            M.octaves_hz.push_back(o[k]), M.lo_hi.push_back({v[k], v[k]});
        return M;
    };
    const mt::Row s = mt::row("door_solid_40mm"), h = mt::row("door_hollow_core_40mm");
    const mt::Measured ptb{"PTB_2012_AbsorptionTable #396", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                           {{0.14, 0.14}, {0.1, 0.1}, {0.06, 0.06}, {0.08, 0.08}, {0.1, 0.1}, {0.1, 0.1}}};
    const mt::Out XS = mt::check_material("door_solid_40mm", s,
                                          {lab("Quirt 1986 #1", "laboratory sound reduction index (ASTM E90), octave = energy mean of thirds", oct, Q, 1.397, 23.9,
                                               {10.5, 21.9, 16.0, 23.1, 50.9, 57.7}),
                                           lab("Morin 1993 #2", "sound intensity, caulked, octave = energy mean of thirds", oct4, M2, 1.427, 25.2, {}),
                                           mt::envelope("Quirt 1986 #1 and Morin 1993 #2", oct4, {{23.9, {Q[1], Q[2], Q[3], Q[4]}}, {25.2, M2}}, mt::mpp(s, 0.04), 1.397), ptb});
    const mt::Out XH = mt::check_material("door_hollow_core_40mm", h,
                                          {lab("Kim 2010 #1", "laboratory sound reduction index (ISO 140-3), octave = energy mean of thirds", oct, K, 1.392, 0.0,
                                               {4.2, 10.4, 8.7, 14.6, 25.1, 49.5})});
    const mt::Out *X[2] = {&XS, &XH};
    const mt::Row *r[2] = {&s, &h};
    const double nbs[3][6] = {{30, 30.4, 29.2, 29.4, 36.5, 44}, {19, 21, 20.8, 19.2, 20.2, 29}, {14, 18.2, 19, 17.7, 16.9, 21}}, m1[4] = {25.2, 27.6, 26.9, 26.2};
    for (int k = 0; k < 2; ++k)
    {
        // a 44 mm leaf: the row rescaled with fRef is the same wall
        const mt::Out Y = mt::run({{mt::rescale(*r[k], 1.1), mt::SHEET, 0.044}}, mt::grid());
        for (int o = 0; o < 9; ++o)
        {
            CHECK(std::abs(Y.t8[o] - X[k]->t8[o]) < 1e-4);
            CHECK(std::abs(Y.a8[o] - X[k]->a8[o]) < 1e-6);
        }
        const double m = mt::mpp(*r[k], 0.04);
        QD_DIAG("# diag: %s (row at its side) against NBS %s (mass-shifted)%s and its absorption against PTB #396, printed\noctave,library_tl,%s,library_alpha,ptb396\n",
                k ? "door_hollow_core_40mm" : "door_solid_40mm", k ? "634 and 633" : "632", k ? "" : " and Morin 1993 #1", k ? "nbs634,nbs633" : "nbs632,morin1");
        for (int o = 0; o < 6; ++o)
            QD_DIAG("%g,%.2f,%.2f,%.2f,%.4f,%.2f\n", oct[o], X[k]->t8[o + 2], nbs[k ? 1 : 0][o] + 20.0 * std::log10(m / (k ? 9.3 : 22.5)),
                    k ? nbs[2][o] : (o >= 1 && o <= 4 ? m1[o - 1] + 20.0 * std::log10(m / 25.2) : NAN), X[k]->a8[o + 2], ptb.lo_hi[o].first);
    }
}

TEST_CASE("Material - glazing_double + glazing_triple", "[acoustic][materials]")
{
    // Sealed glazing units, class 1: the lumped sheets glazing_double_24mm (4-16-4) and glazing_triple_36mm (4-12-4-12-4),
    // side 1.30 m (a pane of 1.2 x 1.4 m). Lumped: thk is the visible thickness, e = m'' / (rho0 thk) carries the stated
    // 20 and 30 kg/m2; a = (c0 / cL)^2 / e of glass. The panes are physical: float glass 2500 kg/m3, fc h 12.5 Hz m (3123
    // Hz at 4 mm, coiF) and coiQ = 1 / 0.005 [Hopkins_2007_SoundInsulation#84], [Hopkins_2007_SoundInsulation#85],
    // [Hopkins_2007_SoundInsulation#86]; resS +1 at the physical mass-air-mass frequency at normal incidence, 211.3 Hz for
    // 16 mm of air and the lower of the triple's two, 172.5 Hz (the upper, 298.8 Hz, has no column; library minima at
    // 229 and 186 Hz by a fine sweep). fRef (the mass level), resQ and att fitted at 342.76271 m/s to AGC (2048 angles x 6
    // tones per third and 50 x 1); att rises with frequency in place of the double leaf's steeper slope above
    // the resonance. The mass levels are 0.57 and 0.42 times the stated masses. tf and tfB the power law of PTB #208
    // [PTB_2012_AbsorptionTable#12] (2-3 mm double glazing, 10 mm gap, Sabine, a neighbour build; 0.0337 at 1 kHz,
    // exponent -0.508; its 63 Hz and 8 kHz are copies), printed. The face absorption caps the transmission at every
    // angle, so it sets the depth of the coincidence dip and of the mass-air-mass dip, which f0 / cos(theta) carries into
    // every octave above f0 at oblique angles: 6.6 dB at 250 Hz in the double unit against a face that absorbs all.
    // Transmission within 3 dB at the specimen's side: AGC [AGC_2023_PocketStratophone#2] (4-16-4, 20 kg/m2, EN ISO
    // 10140, 1.23 x 1.48 m, side 1.349 m). 31.5, 63 Hz and 8 kHz take the spread of three physical models at 1.349 m (the
    // layered unit of thin plates and air with Davy's sigma, the same unit infinite at field incidence 0-78 deg, the limp
    // total mass with Davy's sigma). The grid has 200 angles: on 50 the triple's 8 kHz octave reads 0.5 dB off.
    // Reasonable meant: within the unit's own laboratory values and the model spread. Printed, the row at the specimen's
    // side and mass-shifted: Quirt [Quirt_1981_WindowsTLMeasurements#11], [Quirt_1981_WindowsTLMeasurements#13] (4-13-4
    // and 4-19-4, 20.8 kg/m2, ASTM E90, octaves the energy mean of thirds, panes 0.56 x 1.68 m, side 0.970 m), 3-6 dB
    // above AGC at 500 Hz - 1 kHz, of which the cavity explains at most 1.3 dB; the EN 12758 generic values
    // [Pilkington_2020_OptiphonBrochure#8] (conservative design values, size not stated); for the triple Quirt
    // [Quirt_1981_WindowsTLMeasurements#26] (3-10-3-10-3, 24.3 kg/m2, a neighbour build). Valid: R 125 Hz - 4 kHz.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, A2 = {20.5, 16.8, 25.7, 36.4, 41.4, 36.5}, A14 = {18.7, 19.0, 28.4, 41.6, 46.7, 39.8},
                              Q[3] = {{21.9, 20.9, 30.6, 39.5, 40.4, 35.1}, {19.1, 24.2, 32.1, 40.5, 41.4, 35.0}, {21.0, 20.6, 27.3, 39.7, 45.2, 38.6}},
                              P8 = {21, 17, 25, 35, 37, 31}, ptb = {0.1, 0.07, 0.05, 0.03, 0.02, 0.02};
    auto lab = [&](const char *key, const char *type, const std::vector<double> &v, double side, double m, const std::vector<double> &gaps)
    {
        mt::Measured M{key, "R_dB", type, true, 3.0, {}, {}, side, m};
        for (size_t k = 0; k < gaps.size(); k += 2) // 31.5 Hz, 63 Hz, 8 kHz
            M.octaves_hz.push_back(k < 4 ? 31.5 * (1 + k / 2) : 8000), M.lo_hi.push_back({gaps[k], gaps[k + 1]});
        for (size_t k = 0; k < v.size(); ++k)
            M.octaves_hz.push_back(oct[k]), M.lo_hi.push_back({v[k], v[k]});
        return M;
    };
    const mt::Grid G = mt::grid(200, 1);
    auto tl = [&](const mt::Row &r, double side) // the row's random-incidence octaves (dB) at a specimen's side
    {
        const mt::Field T = mt::transmit(mt::scene({{mt::with(r, "side", side), mt::SOLID, mt::col(r, "thk")}}), G);
        return mt::octaves(G, [&](size_t q) { double t = 0.0; for (int i = 0; i < G.na; ++i) t += G.w[i] * std::norm(T[i][q]); return t; });
    };
    const mt::Row d = mt::row("glazing_double_24mm"), t = mt::row("glazing_triple_36mm");
    const mt::Out XD = mt::check_material("glazing_double_24mm", d,
                                          {lab("AGC #2", "manufacturer (EN ISO 10140)", A2, 1.349, 20.0, {9.26, 20.93, 14.31, 21.75, 48.41, 54.05})}, G);
    const mt::Out XT = mt::check_material("glazing_triple_36mm", t, {lab("model spread", "physical models", {}, 1.349, 0.0, {12.17, 24.38, 17.39, 25.32, 57.57, 59.15})}, G);
    const auto d1 = tl(d, 1.349), d0 = tl(d, 0.970), t1 = tl(t, 1.349), t0 = tl(t, 0.970);
    const double sq = 20.0 * std::log10(20.0 / 20.8), s26 = 20.0 * std::log10(30.0 / 24.3);
    QD_DIAG("# diag: glazing_double_24mm against Quirt #11 / #13 (row at 0.970 m, mass-shifted) and the EN 12758 generic values (row at 1.349 m), "
            "its absorption against PTB #208, printed\noctave,library_tl_0.970,quirt11,quirt13,library_tl_1.349,generic,library_alpha,ptb208\n");
    for (int o = 0; o < 6; ++o)
        QD_DIAG("%g,%.2f,%.2f,%.2f,%.2f,%.0f,%.4f,%.2f\n", oct[o], -10.0 * std::log10(d0[o + 2]), Q[0][o] + sq, Q[1][o] + sq, -10.0 * std::log10(d1[o + 2]), P8[o], XD.a8[o + 2], ptb[o]);
    // ENGINE LIMIT (pane coincidence capped by the face absorption): the face absorption caps the triple unit's
    // mass-air-mass dip at 125 and 250 Hz (3.4 and 3.3 dB above AGC #14), its coincidence dip at 4 kHz (3.3 dB above) and
    // the octaves above f0 (1 kHz 3.3 dB below); AGC [AGC_2023_PocketStratophone#14] (4-12-4-12-4, 30 kg/m2, as AGC #2) is
    // printed, not asserted
    QD_DIAG("# diag: glazing_triple_36mm against AGC #14 (row at 1.349 m; engine limit, printed) and Quirt #26 (row at 0.970 m, mass-shifted), "
            "its absorption against PTB #208, printed\noctave,library_tl_1.349,agc14,deviation,library_tl_0.970,quirt26,library_alpha,ptb208\n");
    for (int o = 0; o < 6; ++o)
        QD_DIAG("%g,%.2f,%.1f,%+.2f,%.2f,%.2f,%.4f,%.2f\n", oct[o], -10.0 * std::log10(t1[o + 2]), A14[o], -10.0 * std::log10(t1[o + 2]) - A14[o],
                -10.0 * std::log10(t0[o + 2]), Q[2][o] + s26, XT.a8[o + 2], ptb[o]);
}

TEST_CASE("Material - gypsum_board + plywood", "[acoustic][materials]")
{
    // Free single boards, class 1: the sheets gypsum_board_12_5mm (side 3.22 m, a wall of boards) and plywood_19mm
    // (side 1.77 m, a sheet of 2.6 x 1.2 m). Gypsum: 680 kg/m3 in e [Hopkins_2007_SoundInsulation#135]; a from cL 1810
    // m/s [Hopkins_2007_SoundInsulation#138]; the coincidence is physical, coiF = fc h / h with fc h = 35.8 Hz m at 343
    // m/s (2860 Hz at 342.76 m/s) [Hopkins_2007_SoundInsulation#134] and coiQ = 1 / 0.0125
    // [Hopkins_2007_SoundInsulation#136] (0.0136 in [Quirt_1995_GypsumWallsSummaryIR693#20]); att 0, no resonance; only
    // fRef fitted (mass level 7.66 kg/m2, 0.90 x 8.5). Plywood: 710 kg/m3 in e [Hopkins_2007_SoundInsulation#153] (the
    // specimens' density is not stated); a from cL 3850 m/s [Hopkins_2007_SoundInsulation#156]; coiF and coiQ are fit
    // knobs: an orthotropic sheet has two coincidences (Wareing's minima at the 800 and 1250 Hz thirds), one column
    // takes the upper, 1245 Hz, where the isotropic fc h 16.8 Hz m [Hopkins_2007_SoundInsulation#152] (883 Hz, eta
    // 0.016 [Hopkins_2007_SoundInsulation#154]) reads 3.3 dB below #5 at 1 kHz; fRef, att fitted (mass level 8.42
    // kg/m2, 0.62 x 13.49). Fits at 342.76271 m/s (4096 angles x 12 tones per third and this case's grid). The face
    // absorption caps the transmission at every angle (an engine limit, "pane coincidence capped by the face
    // absorption"): below the coincidence a light board passes more than a hard face absorbs. Gypsum tf and tfB: the
    // power law of VanHout [VanHout_2016_SuspendedCeilings#9] (Sabine, the 10 mm plasterboard back of a composite tile
    // laid on the floor, a neighbour build; 0.0692 at 1 kHz, exponent -0.411), printed; a law of the board's own
    // transmission plus a painted plaster face [PTB_2012_AbsorptionTable#40] misses the references by 3.5 dB. Plywood
    // tf and tfB: the minimax law of Quirt [Quirt_1983_WindowsII#1] (19 mm unfinished plywood lying on the chamber
    // floor, Sabine; 0.081 at 1 kHz, exponent 0.305), asserted; the free sheet also passes what it transmits, which the
    // floor mounting does not. Against a face that absorbs all, the face law adds 1.8 / 6.5 dB to gypsum at 63 Hz / 4
    // kHz and 4.5 / 5.0 dB to plywood at 63 Hz / 1 kHz.
    // Transmission, each within 3 dB at its specimen's side and mass-shifted: Hopkins
    // [Hopkins_2007_SoundInsulation#132] (12.5 mm, no frame, 10.8 kg/m2, intensity, octaves the energy mean of thirds
    // read from a figure, 3.53 x 2.63 m, side 3.047 m), Northwood [Northwood_1968_PlasterboardWalls#2] (12.7 mm, taped
    // joints, 10.3 kg/m2, ASTM E90, mean of three, 3.05 x 2.44 m, side 2.728 m), their envelope within 1 dB; Wareing
    // [Wareing_2016_OrthotropicPlywood#5] (19 mm) and [Wareing_2016_OrthotropicPlywood#7] (21 mm), small specimens
    // clamped, 0.95 x 1.55 m (side 1.213 m), intensity. 31.5 Hz, 8 kHz (gypsum) and 31.5, 63 Hz, 8 kHz (plywood) take
    // the spread of three physical models at the asserted side (the thin plate of the constants above with Davy's
    // sigma, the same infinite at field incidence 0-78 deg, the limp mass with Davy's sigma). The grid has 200 angles:
    // on 50 the 8 kHz octave reads up to 2.1 dB off.
    // Reasonable meant: within the laboratory boards. Printed: Wareing [Wareing_2015_SampleSizeVariations#1] /
    // [Wareing_2015_SampleSizeVariations#2] (10 mm gypsum, large on studs / small); [Wareing_2016_OrthotropicPlywood#4]
    // (17 mm small: its coincidence moves with 1 / h, which the mass shift does not carry; 3.2 dB at 4 kHz) and
    // [Wareing_2016_OrthotropicPlywood#6] (21 mm large on a glued frame: 2.3-4.5 dB below the small one above fc, where
    // the side adds at most 0.8 dB; the row reads up to 4.5 dB above it). Valid: gypsum R 63 Hz - 4 kHz, plywood R 125
    // Hz - 4 kHz.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, oct7 = {63, 125, 250, 500, 1000, 2000, 4000}, H = {12.6, 16.2, 21.6, 26.9, 32.0, 34.5, 28.8},
                              N = {16.5, 19.8, 24.6, 29.7, 30.6, 25.6}, W[2] = {{12.5, 16.0, 20.6, 26.9, 30.2, 24.0}, {15.8, 19.9, 24.9, 29.3, 30.9, 24.9}},
                              P5 = {18.9, 22.9, 25.8, 25.1, 26.0, 32.7}, P7 = {18.6, 23.5, 25.8, 24.8, 27.0, 33.3}, P4 = {17.1, 20.3, 24.2, 26.3, 25.1, 29.1},
                              P6 = {16.2, 18.6, 21.4, 20.3, 23.5, 31.0}, vh = {0.28, 0.12, 0.06, 0.04, 0.04, 0.08};
    const double sH = std::sqrt(3.53 * 2.63), sN = std::sqrt(3.05 * 2.44), sS = std::sqrt(0.95 * 1.55), sL = std::sqrt(4.8 * 2.4);
    auto lab = [](const char *key, const char *type, const std::vector<double> &o, const std::vector<double> &v, double side, double m)
    {
        mt::Measured M{key, "R_dB", type, true, 3.0, o, {}, side, m};
        for (double x : v)
            M.lo_hi.push_back({x, x});
        return M;
    };
    const char *iso = "laboratory sound reduction index (sound intensity), octave = energy mean of thirds";
    const mt::Row g = mt::row("gypsum_board_12_5mm"), p = mt::row("plywood_19mm");
    const mt::Measured gapG{"model spread", "R_dB", "physical models", true, 3.0, {31.5, 8000}, {{4.29, 7.76}, {35.35, 46.00}}, sH};
    const mt::Measured gapP{"model spread", "R_dB", "physical models", true, 3.0, {31.5, 63, 8000}, {{6.76, 18.44}, {11.46, 19.10}, {46.75, 50.72}}, sS};
    const mt::Measured quirt{"Quirt_1983_WindowsII #1", "alpha_random", "Sabine (ISO 354), on the chamber floor", false, 0.05, oct,
                             {{0.018, 0.018}, {0.018, 0.018}, {0.085, 0.085}, {0.116, 0.116}, {0.092, 0.092}, {0.089, 0.089}}};
    const mt::Out XG = mt::check_material("gypsum_board_12_5mm", g,
                                          {lab("Hopkins #132", iso, oct7, H, sH, 10.8), lab("Northwood #2", "laboratory sound reduction index (ASTM E90), octave = energy mean of thirds", oct, N, sN, 10.3),
                                           mt::envelope("Hopkins #132 and Northwood #2", oct, {{10.8, {H.begin() + 1, H.end()}}, {10.3, N}}, mt::mpp(g, 0.0125), sH), gapG},
                                          mt::grid(200, 1));
    const mt::Out XP = mt::check_material("plywood_19mm", p, {lab("Wareing 2016 #5", iso, oct, P5, sS, mt::mpp(p, 0.019)), lab("Wareing 2016 #7", iso, oct, P7, sS, mt::mpp(p, 0.021)), gapP, quirt},
                                          mt::grid(200, 1));
    const mt::Grid G = mt::grid(200, 1);
    auto tl = [&](const mt::Row &r, double side) // the row's random-incidence octaves (dB) at a specimen's side
    {
        const mt::Field T = mt::transmit(mt::scene({{mt::with(r, "side", side), mt::SOLID, mt::col(r, "thk")}}), G);
        const auto t = mt::octaves(G, [&](size_t q) { double s = 0.0; for (int i = 0; i < G.na; ++i) s += G.w[i] * std::norm(T[i][q]); return s; });
        std::array<double, 9> d;
        for (int o = 0; o < 9; ++o)
            d[o] = -10.0 * std::log10(t[o]);
        return d;
    };
    const auto gL = tl(g, sL), gS = tl(g, sS), pL = tl(p, sL), pS = tl(p, sS);
    const double sw = 20.0 * std::log10(0.0125 / 0.010), s4 = 20.0 * std::log10(0.019 / 0.017), s6 = 20.0 * std::log10(0.019 / 0.021);
    QD_DIAG("# diag: gypsum_board_12_5mm against Wareing 2015 #1 / #2 (10 mm, row at 3.394 / 1.213 m, mass-shifted) and its absorption against VanHout #9, "
            "printed\noctave,library_tl_3.394,wareing1,library_tl_1.213,wareing2,library_alpha,vanhout9\n");
    for (int o = 0; o < 6; ++o)
        QD_DIAG("%g,%.2f,%.2f,%.2f,%.2f,%.4f,%.2f\n", oct[o], gL[o + 2], W[0][o] + sw, gS[o + 2], W[1][o] + sw, XG.a8[o + 2], vh[o]);
    QD_DIAG("# diag: plywood_19mm against Wareing 2016 #4 (17 mm, row at 1.213 m) and #6 (21 mm large, row at 3.394 m), mass-shifted, printed\n"
            "octave,library_tl_1.213,wareing4,library_tl_3.394,wareing6\n");
    for (int o = 0; o < 6; ++o)
        QD_DIAG("%g,%.2f,%.2f,%.2f,%.2f\n", oct[o], pS[o + 2], P4[o] + s4, pL[o + 2], P6[o] + s6);
}
