// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material cases of the heavy group (concrete, masonry, clt, plaster, screed); the rig is acoustic_material_rig.hpp.
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

#include "acoustic_material_rig.hpp"

TEST_CASE("Material - concrete", "[acoustic][materials]")
{
    // Cast concrete, class 1: the solid `concrete` (thk 0) at 150 and 203 mm, the sheets concrete_150mm and
    // concrete_200mm. Columns: density 2281 kg/m3 in e (Litvin 76-66); tf and tfB the power law of PTB #30 and Nell
    // (0.0204 at 1 kHz, exponent 0.3965); the mass-law knobs fitted at 342.76271 m/s to the envelope below at 150 and
    // 203 mm (4096 angles x 12 tones per third, side 3.523 m). coiF (4.41 kHz) and att are fit knobs, not the physical
    // coincidence (fc h = 17.1 Hz m, 114 Hz at 150 mm; [Hopkins_2007_SoundInsulation#58]): the library's single loss
    // factor cannot follow the measured one, which falls from 0.04 to 0.007 over 125 Hz - 4 kHz
    // ([Mahn_2023_GuideAirborneRR331#9], [Mahn_2023_GuideAirborneRR331#11]), and the physical plate with a constant
    // loss factor misses the envelope by 7 to 11 dB.
    // Transmission [Litvin_1980_ConcreteWallsTL#1] and [Litvin_1980_ConcreteWallsTL#2]: tests 76-66 (152 mm,
    // 346.7 kg/m2) and 76-77 (203 mm, 471.6 kg/m2), opening 4.37 x 2.84 m (side 3.523 m), octaves the energy mean of
    // thirds read from a plot (+/-0.5 dB); each within 3 dB, mass-shifted to the row's surface mass; their envelope
    // within 1 dB; 31.5, 63 Hz and 8 kHz take the spread of physical models (tools/acoustic/tl_refs.json). Absorption,
    // hard faces used as measured, within 0.05: PTB #30 "smooth unpainted concrete" [PTB_2012_AbsorptionTable#26]
    // (Sabine; its 63 Hz and 8 kHz are copies, dropped) and the painted boundary of a reverberation room
    // [Nell_TIP_TieftonAbsorber#4] (Sabine from T30).
    // Reasonable meant: within the Litvin pair; Mahn's NRC walls [Mahn_2023_GuideAirborneRR331#8] and
    // [Mahn_2023_GuideAirborneRR331#10] (thirds at the octave centres, CON150 of unknown size) are printed only, 3 dB
    // below the pair at 250 Hz and 9 dB above at 4 kHz. Valid: 150-203 mm, R 125 Hz - 4 kHz.
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000}, L1 = {42.1, 46.5, 52.6, 56.5, 61.7, 64.6}, L2 = {43.8, 47.3, 54.7, 58.1, 63.3, 67.3};
    auto litvin = [&](const char *test, const std::vector<double> &v, double m, bool gaps)
    {
        mt::Measured M{test, "R_dB", "laboratory sound reduction index (ASTM E90), octave = energy mean of thirds", true, 3.0, {}, {}, 3.523, m};
        if (gaps)
            M.octaves_hz = {31.5, 63}, M.lo_hi = {{38.0, 49.8}, {31.7, 42.8}};
        for (size_t k = 0; k < oct.size(); ++k)
            M.octaves_hz.push_back(oct[k]), M.lo_hi.push_back({v[k], v[k]});
        if (gaps)
            M.octaves_hz.push_back(8000), M.lo_hi.push_back({70.4, 77.6});
        return M;
    };
    const mt::Measured ptb{"PTB_2012_AbsorptionTable #30", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                           {{0.01, 0.01}, {0.01, 0.01}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.05, 0.05}}};
    const mt::Measured nell{"Nell_TIP_TieftonAbsorber", "alpha_mean_boundary", "Sabine from T30; a hard face", false, 0.05, {31.5, 63, 125, 250, 500, 1000},
                            {{0.006, 0.006}, {0.0076, 0.0076}, {0.0068, 0.0068}, {0.011, 0.011}, {0.0144, 0.0144}, {0.0189, 0.0189}}};
    const mt::Row c = mt::row("concrete");
    const double mahn[2][6] = {{41, 42, 51, 59, 65, 74}, {42, 48, 55, 60, 67, 78}}, m_mahn[2] = {374.0, 463.5}; // CON150, CON200
    for (int k = 0; k < 2; ++k)
    {
        const double t = k ? 0.203 : 0.150;
        const mt::Out X = mt::check_material(k ? "concrete at 203 mm" : "concrete at 150 mm", c,
                                             {litvin("Litvin 76-66", L1, 346.7, true), litvin("Litvin 76-77", L2, 471.6, false),
                                              mt::envelope("Litvin 76-66 and 76-77", oct, {{346.7, L1}, {471.6, L2}}, mt::mpp(c, t), 3.523), ptb, nell},
                                             mt::grid(), t);
        QD_DIAG("# diag: concrete at %.0f mm against Mahn %s, printed (row at its side; Mahn mass-shifted)\noctave,library,mahn\n", 1e3 * t, k ? "CON200" : "CON150");
        for (int o = 0; o < 6; ++o)
            QD_DIAG("%g,%.2f,%.2f\n", oct[o], X.t8[o + 2], mahn[k][o] + 20.0 * std::log10(mt::mpp(c, t) / m_mahn[k]));
    }
    mt::check_material("concrete_150mm", mt::row("concrete_150mm"));
    mt::check_material("concrete_200mm", mt::row("concrete_200mm"));
}

TEST_CASE("Material - masonry_heavy", "[acoustic][materials]")
{
    // Heavy masonry plastered on both faces (solid clay brick, dense block, sand-lime), class 1: the solid
    // `masonry_heavy` (thk 0) at 150, 200 and 250 mm, the sheets masonry_heavy_150mm, _200mm, _250mm. Columns: density
    // 1772 kg/m3 in e (the mean of the two walls measured plastered on both faces, 1760 and 1784 kg/m3); tf and tfB the
    // power law of PTB #344 [PTB_2012_AbsorptionTable#38] and PTB #346 [PTB_2012_AbsorptionTable#40] (Sabine, hard faces
    // used as measured; 0.0270 at 1 kHz, exponent 0.157); the mass-law knobs fitted at 342.76271 m/s to the references
    // below at 150, 200 and 250 mm, on 4096 angles x 12 tones per third and on this case's grid (within 0.25 dB of each
    // other). coiF (7.4 kHz), coiQ, resF and the mass level are fit knobs: the level stands for 0.70 times the physical
    // mass e rho0 t (pinned at the physical mass, the fit misses by 4.0 dB), so a bonded addition counts about 1.4 times
    // its mass law. Above coincidence a wall gains 9 dB per doubling of thickness (20 lg m + 10 lg h); the class's walls
    // gain 7.9-9.4 dB at 125 Hz - 1 kHz, the row 6.02 dB (heavy-wall coincidence engine limit): at most 1.3 dB at the
    // ends of 150-250 mm, below the 2-3 dB scatter of walls of one mass; at 300 mm -4.1 dB against BIA #11 (printed).
    // Transmission, walls plastered on both faces first, otherwise a wall plus one measured increment per missing face
    // (the band of the three increments), each within 3 dB, mass-shifted to the row's surface mass: at 250 mm Weise
    // [Weise_2003_PruefstandAbweichungen#1] (sand-lime 240 mm, thin gypsum plaster, 250 mm, 440 kg/m2, mean of 12
    // laboratories; size not stated, taken at the row's 3.22 m, 0.04 dB from the 10 m2 of ISO 140-1; 31.5, 63 Hz and
    // 8 kHz the spread of physical models, tools/acoustic/tl_refs.json, brick 203 mm) and Hopkins
    // [Hopkins_2007_SoundInsulation#68] (dense block 215 mm, plastered, 430 kg/m2, 3.53 x 2.63 m); at 150 mm BIA
    // [BIA_2000_TN5AClayMasonry#8] (brick 140 mm, plaster on one face, 4.37 x 2.84 m) plus one increment (166 mm
    // finished). The envelope of these three and [BIA_2000_TN5AClayMasonry#12] plus two increments, within 1 dB, is 3-6 dB
    // wide, wider than its tolerance: the single references carry the test. Increments of one face:
    // [BIA_2000_TN5AClayMasonry#8] minus [BIA_2000_TN5AClayMasonry#7], [BIA_2000_TN5AClayMasonry#14] minus
    // [BIA_2000_TN5AClayMasonry#15], [Warnock_1984_ConcreteBlockWalls#5] minus [Warnock_1984_ConcreteBlockWalls#4];
    // 18.03 kg/m2 of plaster per face (their mean). Reasonable meant: within the walls of one mass (Weise and Hopkins
    // differ by up to 3.9 dB at 440 and 430 kg/m2). Printed: the ends at 100 mm ([Hopkins_2007_SoundInsulation#67],
    // [BIA_2000_TN5AClayMasonry#3], [Warnock_1984_ConcreteBlockWalls#1], [BIA_2000_TN5AClayMasonry#1]) and 300 mm
    // ([BIA_2000_TN5AClayMasonry#11]); at 250 mm [Weise_2003_PruefstandAbweichungen#2], Secchi wall A
    // [Secchi_2003_LaterizioLabInSitu#1], the NBS brick walls [NBS_1955_BMS144SoundInsulation#35] to
    // [NBS_1955_BMS144SoundInsulation#37] (warble tones, weighed with the frame), the grouted wall
    // [BIA_2000_TN5AClayMasonry#6]; the rule "bare wall plus one increment per face" against Weise. Valid: 150-250 mm,
    // R 125 Hz - 4 kHz.
    using V = std::vector<double>;
    using Walls = std::vector<std::tuple<const char *, double, V, int>>; // name, surface mass, octaves (NAN: none), faces added
    const V oct = {125, 250, 500, 1000, 2000, 4000}, inc[3] = {{2.5, 1.1, 1.5, 2.0, 1.5, 0.0}, {2.7, 1.7, 2.5, 3.0, 1.6, 1.0}, {1.7, 1.7, 2.3, 2.6, 1.8, 2.6}};
    const double PM = 18.03;
    auto plus = [&](const V &v, int n, int lmh) // n faces added with the lowest (-1), mean (0) or highest (+1) increment
    {
        V p = v;
        for (size_t k = 0; k < 6; ++k)
            p[k] += n * (lmh < 0 ? std::min({inc[0][k], inc[1][k], inc[2][k]}) : lmh > 0 ? std::max({inc[0][k], inc[1][k], inc[2][k]}) : (inc[0][k] + inc[1][k] + inc[2][k]) / 3.0);
        return p;
    };
    auto ref = [&](const char *key, const V &v, int n, double side, double m)
    {
        mt::Measured M{key, "R_dB", n ? "laboratory sound reduction index plus measured plaster increments" : "laboratory sound reduction index", true, 3.0, oct, {}, side, m + n * PM};
        for (size_t k = 0; k < 6; ++k)
            M.lo_hi.push_back({plus(v, n, -1)[k], plus(v, n, 1)[k]});
        return M;
    };
    const V weise = {42.6, 44.1, 52.4, 59.6, 64.9, 66.7}, hop68 = {39.6, 42.7, 52.2, 58.3, 65.8, 70.6}, bia8 = {39.3, 41.5, 48.2, 53.7, 58.5, 62.7};
    const std::vector<std::pair<double, V>> bare = {{189.0, {31.6, 33.5, 39.3, 46.7, 54.7, 60.7}}, {272.4, {36.8, 40.4, 46.7, 51.7, 57.0, 62.7}},
                                                    {406.7, {37.3, 40.1, 47.5, 52.6, 59.8, 65.0}}, {459.9, {42.6, 47.5, 55.9, 61.8, 66.0, 70.5}},
                                                    {569.8, {44.0, 47.7, 54.2, 60.7, 65.6, 68.7}}}; // BIA #3, #7, #12, #6, #11
    mt::Measured W = ref("Weise #1", weise, 0, 0.0, 440.0);
    for (const auto &[f, lo, hi] : std::vector<std::tuple<double, double, double>>{{31.5, 40.1, 44.6}, {63, 27.9, 41.8}, {8000, 70.2, 72.4}}) // at 407 kg/m2
        W.octaves_hz.push_back(f), W.lo_hi.push_back({lo + 20.0 * std::log10(440.0 / 407.0), hi + 20.0 * std::log10(440.0 / 407.0)});
    const mt::Measured ptb344{"PTB_2012_AbsorptionTable #344", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.03, 0.03}, {0.04, 0.04}, {0.05, 0.05}, {0.05, 0.05}}};
    const mt::Measured ptb346{"PTB_2012_AbsorptionTable #346", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}}};
    const mt::Row r = mt::row("masonry_heavy");
    auto show = [&](const char *title, const mt::Out &X, double t, const Walls &walls) // printed: the row minus each wall's band
    {
        QD_DIAG("\n# diag: masonry_heavy at %.0f mm against %s, printed (row at its side, walls mass-shifted): row minus the band (dB)\nwall,125,250,500,1000,2000,4000", 1e3 * t, title);
        for (const auto &[name, m, v, n] : walls)
        {
            const V lo = plus(v, n, -1), hi = plus(v, n, 1);
            const double s = 20.0 * std::log10(mt::mpp(r, t) / (m + n * PM));
            QD_DIAG("\n%s", name);
            for (int k = 0; k < 6; ++k)
            {
                const double x = X.t8[k + 2] - s;
                QD_DIAG(",%+.2f", std::isnan(v[k]) ? NAN : x > hi[k] ? x - hi[k] : x < lo[k] ? x - lo[k] : 0.0);
            }
        }
        QD_DIAG("\n");
    };
    for (double t : {0.150, 0.200, 0.250})
    {
        std::vector<mt::Measured> refs = {mt::envelope("Weise #1, Hopkins #68, BIA #8 + 1 and #12 + 2 increments", oct,
                                                       {{440.0, weise}, {430.0, hop68}, {296.9 + PM, plus(bia8, 1, -1)}, {296.9 + PM, plus(bia8, 1, 1)},
                                                        {406.7 + 2 * PM, plus(bare[2].second, 2, -1)}, {406.7 + 2 * PM, plus(bare[2].second, 2, 1)}},
                                                       mt::mpp(r, t), 0.0),
                                          ptb344, ptb346};
        if (t < 0.175)
            refs.push_back(ref("BIA #8 + 1 increment", bia8, 1, 3.53, 296.9));
        if (t > 0.225)
            refs.push_back(W), refs.push_back(ref("Hopkins #68", hop68, 0, std::sqrt(3.53 * 2.63), 430.0));
        const mt::Out X = mt::check_material(t < 0.175 ? "masonry_heavy at 150 mm" : t < 0.225 ? "masonry_heavy at 200 mm" : "masonry_heavy at 250 mm", r, refs, mt::grid(), t);
        if (t > 0.225)
        {
            QD_DIAG("\n# diag: masonry_heavy at 250 mm, 63 Hz: library %.2f dB, Weise #1 mass-shifted %.2f dB (printed)\n", X.t8[1], 40.7 + 20.0 * std::log10(mt::mpp(r, t) / 440.0));
            show("walls of about 250 mm", X, t,
                 {{"Weise #2", 440.0, {43.6, 46.8, 54.3, 60.1, 65.8, 68.0}, 0}, {"Secchi A", 400.0, {43.2, 44.4, 48.7, 55.1, 61.1, NAN}, 0},
                  {"NBS #35", 449.0, {NAN, 48, 48, 56, 56, NAN}, 0}, {"NBS #36", 474.0, {NAN, 48, 49, 57, 59, NAN}, 0}, {"NBS #37", 425.0, {NAN, 50, 48, 56, 64, NAN}, 0},
                  {"BIA #6 + 2", bare[3].first, bare[3].second, 2}});
        }
    }
    // The rule "bare wall plus one increment per face" against Weise: the least-squares trend of the bare BIA walls in
    // log2 of the surface mass at 440 kg/m2, plus two mean increments
    V trend(6);
    for (size_t k = 0; k < 6; ++k)
    {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (const auto &[m, v] : bare)
            sx += std::log2(m), sy += v[k], sxx += std::log2(m) * std::log2(m), sxy += std::log2(m) * v[k];
        const double b = (5.0 * sxy - sx * sy) / (5.0 * sxx - sx * sx);
        trend[k] = (sy - b * sx) / 5.0 + b * std::log2(440.0);
    }
    QD_DIAG("\n# diag: Weise #1 minus (bare BIA trend at 440 kg/m2 plus two mean increments), printed\noctave,dB\n");
    for (int k = 0; k < 6; ++k)
        QD_DIAG("%g,%+.2f\n", oct[k], weise[k] - plus(trend, 2, 0)[k]);
    show("light walls", mt::check_material("masonry_heavy at 100 mm", r, {}, mt::grid(), 0.100), 0.100,
         {{"Hopkins #67 + 1", 200.0, {27.4, 33, 42.8, 51.1, 59.8, 66.3}, 1}, {"BIA #3 + 2", bare[0].first, bare[0].second, 2},
          {"Warnock #1 + 2", 195.2, {28.0, 36.2, 41.3, 50.4, 56.9, 60.0}, 2}, {"BIA #1 + 1", 207.0, {35.6, 39.7, 45.7, 51.4, 55.9, 60.2}, 1}});
    show("the 305 mm wall", mt::check_material("masonry_heavy at 300 mm", r, {}, mt::grid(), 0.300), 0.300, {{"BIA #11 + 2", bare[4].first, bare[4].second, 2}});
    mt::check_material("masonry_heavy_150mm", mt::row("masonry_heavy_150mm"));
    mt::check_material("masonry_heavy_200mm", mt::row("masonry_heavy_200mm"));
    mt::check_material("masonry_heavy_250mm", mt::row("masonry_heavy_250mm"));
}

TEST_CASE("Material - masonry_hollow", "[acoustic][materials]")
{
    // Hollow-unit masonry plastered on both faces (hollow clay brick, hollow block of any aggregate), class 1: the solid
    // `masonry_hollow` (thk 0) at 100, 166, 200 and 250 mm, the sheets masonry_hollow_100mm, _200mm, _250mm. Columns:
    // density 1385 kg/m3 in e (the mean of the four walls plastered on both faces below, 1271-1558 kg/m3; the fit does
    // not depend on it, the mass level absorbs it); tf and tfB the plaster law of masonry_heavy (PTB #344
    // [PTB_2012_AbsorptionTable#38], PTB #346 [PTB_2012_AbsorptionTable#40]); the mass-law knobs fitted at 342.76271 m/s
    // to the references below on 4096 angles x 12 tones per third and on this case's grid. coiF (4.8 kHz), coiQ, resF and
    // the mass level are fit knobs: the level stands for 0.63 times the physical mass e rho0 t (pinned at the physical
    // mass, the fit misses by 3.35 dB).
    // Transmission, each mass-shifted to the row's surface mass, within 3 dB: at 166 mm Warnock
    // [Warnock_1984_ConcreteBlockWalls#3] (140 mm hollow normal-weight block, 13 mm sand plaster both faces, 229 kg/m2:
    // 199 kg/m2 of blocks plus 15 kg/m2 per face, [Warnock_1984_ConcreteBlockWalls#5] minus #4), 125 Hz - 2 kHz, its 4 kHz
    // octave (two of three thirds) printed; at 200 mm Granzotto wall B [Granzotto_2019_ClayHollowBrickWalls#2] (206 mm,
    // 321 kg/m2; 31.5, 63 Hz and 8 kHz the spread of physical models, tools/acoustic/tl_refs.json, block 190); at 250 mm
    // walls C [Granzotto_2019_ClayHollowBrickWalls#3] (240 mm, 305 kg/m2) and E [Granzotto_2019_ClayHollowBrickWalls#5]
    // (270 mm, 359 kg/m2) as one band: walls of one build and thickness that differ by more than the tolerance (here up to
    // 6 dB at 2-4 kHz at one mass) are asserted as their band; at 100 mm BIA [BIA_2000_TN5AClayMasonry#14] (clay tile
    // 90 mm, 16 mm plaster on one face, 4.37 x 2.84 m) plus one measured increment for the other face (the band of
    // [BIA_2000_TN5AClayMasonry#14] minus #15, the same tile, [BIA_2000_TN5AClayMasonry#8] minus #7 and
    // [Warnock_1984_ConcreteBlockWalls#5] minus #4; 14.6 kg/m2): 138.1 kg/m2 against the row's 138.5. The sides of
    // Warnock and Granzotto are not stated: taken at the row's 3.22 m (the NRC opening of 3.66 x 2.44 m, 2.99 m, is only
    // inferred for Warnock; it reads 0.17 dB higher at 125 Hz). The envelope of the four walls, within 1 dB, is 1.3-6.8 dB
    // wide. Reasonable meant: within the walls of one mass (the plastered block reads 6 dB above clay wall E at 2 kHz; the
    // seven Granzotto walls spread 2.7-5.9 dB). Printed: 300 mm against wall G [Granzotto_2019_ClayHollowBrickWalls#7];
    // walls A, D, F [Granzotto_2019_ClayHollowBrickWalls#1], #4, #6; Secchi [Secchi_2003_LaterizioLabInSitu#1] to #3
    // (filled cores); the NBS tile, cinder and pumice walls plastered on both faces [NBS_1955_BMS144SoundInsulation#39] to
    // [NBS_1955_BMS144SoundInsulation#53] (warble tones, weighed with the frame); bare walls plus increments: BIA
    // [BIA_2000_TN5AClayMasonry#13] plus two clay increments, Zeitler [Zeitler_2015_ConcreteBlockRR334#3] and Warnock
    // [Warnock_1995_BlockConstructions#1] plus the two-face increment of the porous block
    // ([Warnock_1984_ConcreteBlockWalls#3] minus #2, mostly by sealing). Valid: 100-250 mm, R 125 Hz - 4 kHz.
    using V = std::vector<double>;
    const V oct = {125, 250, 500, 1000, 2000, 4000}, inc[3] = {{2.7, 1.7, 2.5, 3.0, 1.6, 1.0}, {2.5, 1.1, 1.5, 2.0, 1.5, 0.0}, {1.7, 1.7, 2.3, 2.6, 1.8, 2.6}};
    const V w3 = {35.6, 40.4, 46.6, 53.2, 59.4, 60.3}, w2 = {30.2, 33.9, 40.7, 47.4, 54.1, 54.8}, gB = {42.1, 44.1, 48.3, 54.0, 59.5, 61.6},
            gC = {40.4, 42.4, 45.2, 52.6, 58.3, 61.9}, gE = {44.0, 43.8, 48.9, 54.4, 57.3, 57.4}, b14 = {30.3, 32.3, 35.7, 41.9, 47.7, 52.5};
    auto one = [&](const char *key, const V &v, size_t n, double side, double m) // a single wall, its first n octaves
    {
        mt::Measured M{key, "R_dB", "laboratory sound reduction index", true, 3.0, V(oct.begin(), oct.begin() + n), {}, side, m};
        for (size_t k = 0; k < n; ++k)
            M.lo_hi.push_back({v[k], v[k]});
        return M;
    };
    mt::Measured B = one("Granzotto B", gB, 6, 0.0, 321.0), bia = one("BIA #14 + 1 increment", b14, 6, 3.523, 123.5 + 14.6);
    for (const auto &[f, lo, hi] : std::vector<std::tuple<double, double, double>>{{31.5, 27.8, 37.7}, {63, 30.0, 37.6}, {8000, 66.8, 70.2}}) // at 225 kg/m2
        B.octaves_hz.push_back(f), B.lo_hi.push_back({lo + 20.0 * std::log10(321.0 / 225.0), hi + 20.0 * std::log10(321.0 / 225.0)});
    bia.coefficient_type = "laboratory sound reduction index plus a measured plaster increment";
    for (size_t k = 0; k < 6; ++k)
        bia.lo_hi[k] = {b14[k] + std::min({inc[0][k], inc[1][k], inc[2][k]}), b14[k] + std::max({inc[0][k], inc[1][k], inc[2][k]})};
    const mt::Measured ptb344{"PTB_2012_AbsorptionTable #344", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.03, 0.03}, {0.04, 0.04}, {0.05, 0.05}, {0.05, 0.05}}};
    const mt::Measured ptb346{"PTB_2012_AbsorptionTable #346", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}}};
    const mt::Row r = mt::row("masonry_hollow");
    auto show = [&](const char *title, const mt::Out &X, double t, const std::vector<std::tuple<const char *, double, V>> &walls) // printed: row minus wall (dB)
    {
        QD_DIAG("\n# diag: masonry_hollow at %.0f mm against %s, printed (row at its side, walls mass-shifted): row minus wall (dB)\nwall,125,250,500,1000,2000,4000", 1e3 * t, title);
        for (const auto &[name, m, v] : walls)
        {
            QD_DIAG("\n%s", name);
            for (int k = 0; k < 6; ++k)
                QD_DIAG(",%+.2f", X.t8[k + 2] - v[k] - 20.0 * std::log10(mt::mpp(r, t) / m));
        }
        QD_DIAG("\n");
    };
    for (double t : {0.100, 0.166, 0.200, 0.250})
    {
        std::vector<mt::Measured> refs = {mt::envelope("Warnock #3, Granzotto B, C and E", oct, {{229.0, w3}, {321.0, gB}, {305.0, gC}, {359.0, gE}}, mt::mpp(r, t), 0.0), ptb344, ptb346};
        if (t < 0.15)
            refs.push_back(bia);
        else if (t < 0.18)
            refs.push_back(one("Warnock #3", w3, 5, 0.0, 229.0));
        else if (t < 0.225)
            refs.push_back(B);
        else
        {
            refs.push_back(mt::envelope("Granzotto C and E as one band", oct, {{305.0, gC}, {359.0, gE}}, mt::mpp(r, t), 0.0));
            refs.back().coefficient_type = "laboratory sound reduction index, two walls of one build, mass-shifted";
            refs.back().tol = 3.0;
        }
        char name[40];
        std::snprintf(name, sizeof(name), "masonry_hollow at %.0f mm", 1e3 * t);
        const mt::Out X = mt::check_material(name, r, refs, mt::grid(), t);
        if (t > 0.15 && t < 0.18)
            QD_DIAG("\n# diag: masonry_hollow at 166 mm, 4 kHz (two of three thirds): library %.2f dB, Warnock #3 mass-shifted %.2f dB (printed)\n", X.t8[7], w3[5] + 20.0 * std::log10(mt::mpp(r, t) / 229.0));
        if (t < 0.15)
            show("the NBS walls plastered on both faces", X, t,
                 {{"NBS #39", 234.0, {NAN, 44, 44, 49, 58, NAN}}, {"NBS #40", 190.0, {NAN, 39, 42, 47, 54, NAN}}, {"NBS #41", 181.0, {NAN, 41, 37, 45, 52, NAN}},
                  {"NBS #42", 181.0, {NAN, 41, 42, 44, 50, NAN}}, {"NBS #43", 142.0, {NAN, 41, 40, 42, 50, NAN}}, {"NBS #44", 134.0, {31, 31, 36, 47, 50, 58}},
                  {"NBS #45", 183.0, {30, 35, 44, 52, 56, 65}}, {"NBS #46", 163.0, {33, 33, 42, 46, 49, 62}}, {"NBS #47", 137.0, {NAN, 41, 36, 43, 51, NAN}},
                  {"NBS #48", 137.0, {NAN, 42, 41, 44, 50, NAN}}, {"NBS #49", 123.0, {29, 35.9, 35.7, 39, 48, 51}}, {"NBS #50", 145.0, {30, 30, 38, 48, 53, 59}},
                  {"NBS #51", 175.0, {36, 37.7, 43.5, 51, 55, 62}}, {"NBS #52", 157.0, {34, 36.7, 41.9, 51, 57, 64}}, {"NBS #53", 124.0, {32, 33.4, 35.4, 39, 42, 52}}});
        if (t > 0.18 && t < 0.225)
        {
            const V b13 = {32.6, 35.0, 39.5, 45.2, 51.3, 52.2}, z3 = {35.6, 37.9, 44.4, 49.9, 56.8, 61.7}, w95 = {35.0, 41.1, 44.4, 50.1, 56.3, 63.0};
            V a(6), b(6), c(6);
            for (size_t k = 0; k < 6; ++k)
                a[k] = b13[k] + 2.0 * inc[0][k], b[k] = z3[k] + w3[k] - w2[k], c[k] = w95[k] + w3[k] - w2[k];
            show("bare walls plus increments", X, t, {{"BIA #13 + 2 clay increments", 198.2 + 29.2, a}, {"Zeitler #3 + block increment", 238.0 + 30.0, b}, {"Warnock 1995 #1 + block increment", 236.2 + 30.0, c}});
        }
        if (t > 0.225)
            show("walls of 210-330 mm", X, t,
                 {{"Granzotto A", 330.0, {41.0, 45.7, 47.0, 52.0, 56.6, 60.3}}, {"Granzotto D", 343.0, {44.9, 40.9, 48.5, 53.0, 58.8, 60.6}},
                  {"Granzotto F", 327.0, {39.5, 42.6, 46.7, 51.4, 57.4, 59.4}}, {"Secchi A", 400.0, {43.2, 44.4, 48.7, 55.1, 61.1, NAN}},
                  {"Secchi B", 371.0, {40.2, 45.9, 48.2, 54.3, 60.0, NAN}}, {"Secchi C", 390.0, {45.4, 49.9, 50.8, 57.5, 62.3, NAN}}});
    }
    show("Granzotto G", mt::check_material("masonry_hollow at 300 mm", r, {}, mt::grid(), 0.300), 0.300, {{"Granzotto G", 377.0, {44.5, 46.1, 49.3, 53.1, 58.2, 59.9}}});
    mt::check_material("masonry_hollow_100mm", mt::row("masonry_hollow_100mm"));
    mt::check_material("masonry_hollow_200mm", mt::row("masonry_hollow_200mm"));
    mt::check_material("masonry_hollow_250mm", mt::row("masonry_hollow_250mm"));
}

TEST_CASE("Material - masonry_light", "[acoustic][materials]")
{
    // Light masonry plastered on both faces (aircrete, expanded-clay and lightweight-aggregate solid units), class 1: the
    // solid `masonry_light` (thk 0) at 100 and 128 mm, the sheets masonry_light_100mm and _150mm. Columns: density
    // 620 kg/m3 in e (the geometric centre of the stated 480-800 kg/m3); tf and tfB the plaster law of masonry_heavy (PTB
    // #344 [PTB_2012_AbsorptionTable#38], PTB #346 [PTB_2012_AbsorptionTable#40]); the mass-law knobs fitted at
    // 342.76271 m/s to the two references below on 4096 angles x 12 tones per third and on this case's grid. coiF
    // (463 Hz), coiQ (1.80), resF and the mass level are fit knobs: the level stands for 0.40 times the physical mass
    // e rho0 t (pinned at the physical mass, the fit misses by 4.1 dB), so a bonded addition counts about 2.5 times its
    // mass law. The bare walls show a coincidence dip at fc h 44-60 Hz m (500 Hz at 100 mm, Granzotto's fits); plaster
    // acts as added mass and damps that dip (the plastered walls show none), so the row carries a broad trough.
    // Transmission, within 3 dB: at 100 mm the four bare 10 cm walls [Granzotto_2026_AACAirborneInsulation#2] to #5
    // (thin-bed mortar, ISO 10140, thirds from 100 Hz) as one band, not mass-shifted: their density is stated only as
    // 480-800 kg/m3 (48-80 kg/m2, 4.44 dB of mass law, printed). A finished 100 mm wall with thin lightweight plaster
    // (13 mm at 650 kg/m3, [Hopkins_2007_SoundInsulation#96]) weighs what the bare 100 mm wall weighs (-0.4 to +0.8 dB
    // over 480-800 kg/m3), so the band is used as measured. Its 31.5, 63 Hz and 8 kHz take the spread of 144 plate models
    // (tools/acoustic/gzlib.py: thin and Mindlin plates, infinite and finite at 3.22 m; 48-80 kg/m2, coincidence
    // 440-600 Hz, loss factor 0.001-0.03). At 128 mm Hopkins [Hopkins_2007_SoundInsulation#4] (aircrete 115 mm, 13 mm
    // lightweight plaster on one face, 71 kg/m2, 3.53 x 2.63 m), mass-shifted. Granzotto's specimen size is not stated:
    // taken at the row's 3.22 m (0.04 dB from the 10 m2 of ISO 10140). Reasonable meant: between the bare walls (a dip
    // at 400-630 Hz) and the plastered wall (none), which differ in shape by 6-8 dB at one mass. The coincidence column
    // is fixed while a wall's dip moves as 1/h (heavy-wall coincidence engine limit): the row against the same row with
    // coiF scaled as 1/h misses by 1.9 / 2.9 / 4.3 dB at 128 / 150 / 190 mm, so 150-250 mm are printed. Printed: the bare
    // walls Davy [Davy_2017_AACPanels#2] (75 mm), Granzotto #1 (7 cm), #6, #7 (12 cm), #8 (12.5 cm, build unknown, not
    // shifted), the plastered #9 (16.5 cm, build inferred); Warnock [Warnock_1995_BlockConstructions#4] (wood-fibre
    // blocks, 2 mm plaster on one face) and [Warnock_1995_BlockConstructions#2] (hollow lightweight blocks, plaster on one
    // face: the hollow class); the row's Rw (ISO 717-1) against Granzotto's law over 50 walls, Rw = 29.0 lg m - 15.5
    // [Granzotto_2026_AACAirborneInsulation#10]. Valid: 90-130 mm, R 125 Hz - 4 kHz.
    using V = std::vector<double>;
    const V oct = {125, 250, 500, 1000, 2000, 4000};
    const mt::Row r = mt::row("masonry_light");
    mt::Measured band = mt::envelope("Granzotto #2 to #5 as one band", oct,
                                     {{1.0, {27.5, 29.7, 27.1, 34.4, 42.6, 49.3}}, {1.0, {27.9, 31.1, 26.3, 35.0, 43.3, 48.8}},
                                      {1.0, {31.0, 29.4, 28.8, 37.1, 43.6, 49.9}}, {1.0, {33.3, 34.0, 28.6, 37.8, 46.4, 53.7}}},
                                     1.0, 0.0); // density not stated: not mass-shifted
    band.coefficient_type = "laboratory sound reduction index, four walls of one build", band.tol = 3.0;
    for (const auto &[f, lo, hi] : std::vector<std::tuple<double, double, double>>{{31.5, 14.13, 26.77}, {63, 18.96, 29.96}, {8000, 39.38, 68.46}})
        band.octaves_hz.push_back(f), band.lo_hi.push_back({lo, hi});
    mt::Measured hop{"Hopkins #4", "R_dB", "laboratory sound reduction index (intensity), plaster on one face", true, 3.0, oct, {}, std::sqrt(3.53 * 2.63), 71.0};
    for (double v : {23.6, 26.9, 33.8, 42.2, 50.8, 56.0})
        hop.lo_hi.push_back({v, v});
    const mt::Measured ptb344{"PTB_2012_AbsorptionTable #344", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.03, 0.03}, {0.04, 0.04}, {0.05, 0.05}, {0.05, 0.05}}};
    const mt::Measured ptb346{"PTB_2012_AbsorptionTable #346", "alpha_random", "Sabine (ISO 354), compiled; a hard face", false, 0.05, oct,
                              {{0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}, {0.02, 0.02}}};
    auto rw = [](const mt::Out &X) // ISO 717-1 on the thirds 100 Hz - 3.15 kHz (one tone per third)
    {
        static const double ref[16] = {33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56};
        double R[16] = {};
        for (size_t q = 0; q < X.g.f.size(); ++q)
            if (X.g.third[q] >= 6 && X.g.third[q] < 22)
                R[X.g.third[q] - 6] = -10.0 * std::log10(X.tauf[q]);
        auto below = [&](int s) { double u = 0.0; for (int k = 0; k < 16; ++k) u += std::max(0.0, ref[k] + s - R[k]); return u; };
        int s = -40;
        while (below(s + 1) <= 32.0) // the highest 1 dB shift with at most 32 dB of unfavourable deviations
            ++s;
        return 52 + s;
    };
    const std::vector<std::tuple<double, const char *, double, V>> walls = { // thickness, wall, surface mass (0: not shifted), octaves
        {0.070, "Granzotto #1 (600 kg/m3 blocks)", 42.0, {31.1, 32.7, 27.5, 30.3, 41.9, 49.8}},
        {0.075, "Davy #2", 38.2, {27.8, 31.5, 28.2, 32.9, 42.4, 49.1}},
        {0.090, "Warnock 1995 #4 (wood-fibre blocks)", 98.0, {32.5, 36.0, 37.5, 42.8, 48.4, 55.0}},
        {0.120, "Granzotto #6 (500-600 kg/m3, at 66 kg/m2)", 66.0, {26.9, 30.1, 28.2, 34.9, 44.5, 51.8}},
        {0.120, "Granzotto #7 (500-600 kg/m3, at 66 kg/m2)", 66.0, {29.8, 31.6, 26.3, 36.1, 43.2, 46.9}},
        {0.125, "Granzotto #8 (build unknown)", 0.0, {28.6, 31.1, 31.6, 37.6, 42.9, 46.3}},
        {0.165, "Granzotto #9 (plastered, 120.1 kg/m2)", 120.1, {41.9, 39.0, 39.7, 46.4, 51.2, 51.7}},
        {0.190, "Warnock 1995 #2 (hollow lightweight blocks)", 183.0, {34.1, 39.8, 44.1, 49.5, 54.2, 58.4}}};
    std::vector<std::pair<double, mt::Out>> runs;
    for (double t : {0.070, 0.075, 0.090, 0.100, 0.120, 0.125, 0.128, 0.150, 0.165, 0.190, 0.200, 0.250})
    {
        std::vector<mt::Measured> refs;
        if (t == 0.100)
            refs = {band, ptb344, ptb346};
        else if (t == 0.128)
            refs = {hop, ptb344, ptb346};
        char name[40];
        std::snprintf(name, sizeof(name), "masonry_light at %.0f mm", 1e3 * t);
        runs.push_back({t, mt::check_material(name, r, refs, mt::grid(), t)});
    }
    QD_DIAG("\n# diag: masonry_light, the stated block density 480-800 kg/m3 at 100 mm: 48-80 kg/m2, %.2f dB of mass law (printed)\n", 20.0 * std::log10(800.0 / 480.0));
    QD_DIAG("# diag: masonry_light, printed: Rw of the row and Granzotto's law at e rho0 t; row (at its side) minus wall, mass-shifted (dB)\nthickness_mm,wall,125,250,500,1000,2000,4000");
    for (const auto &[t, X] : runs)
    {
        const double m = mt::mpp(r, t);
        QD_DIAG("\n%.0f,Rw %d (law %.1f)", 1e3 * t, rw(X), 29.0 * std::log10(m) - 15.5);
        for (const auto &[tw, wall, mw, v] : walls)
            if (tw == t)
            {
                QD_DIAG("\n%.0f,%s", 1e3 * t, wall);
                for (int k = 0; k < 6; ++k)
                    QD_DIAG(",%+.2f", X.t8[k + 2] - v[k] - (mw > 0.0 ? 20.0 * std::log10(m / mw) : 0.0));
            }
    }
    QD_DIAG("\n");
    mt::check_material("masonry_light_100mm", mt::row("masonry_light_100mm"));
    mt::check_material("masonry_light_150mm", mt::row("masonry_light_150mm"));
}
