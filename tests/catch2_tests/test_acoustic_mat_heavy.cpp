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
