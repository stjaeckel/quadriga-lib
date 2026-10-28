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
