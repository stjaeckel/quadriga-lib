// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material cases of the porous group (glass wool, rock wool, foams, fibre panels); the rig is acoustic_material_rig.hpp.
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

#include "acoustic_material_rig.hpp"

TEST_CASE("Material - glass_wool", "[acoustic][materials]")
{
    // Glass wool, class 5: the solid `glass_wool` (thk 0) at 50 and 100 mm, the sheets glass_wool_50mm and
    // glass_wool_100mm. Columns: power laws of the JCA model [Jaouen_2026_APMRJCA] at the specimen's measured set
    // [Dupont_2023_SoundFieldSynthesis#2] (sigma 13112 Pa s/m2, porosity 0.954, tortuosity 1.0, lengths 55.6 and 116 um),
    // fitted at 342.76271 m/s on the worst random-incidence absorption miss over 63 Hz - 8 kHz at 25, 50, 100 and
    // 200 mm on a rigid backing (0.0088), with the 0 deg transmission loss of 50 mm in air within 0.3 dB over
    // 63 Hz - 4 kHz. Komatsu and Miki at the same resistivity read 0.049 and 0.033 below the measurement at 1 kHz.
    // Absorption of the 50 mm sheet on a rigid wall (concrete with tf 0), within 0.05 of the band between
    // [Dupont_2023_SoundFieldSynthesis#1] (JCA prediction of the infinite layer, 125 Hz - 4 kHz) and
    // [Dupont_2023_SoundFieldSynthesis#3] (sound-field synthesis on a 1.5 x 1.5 m sample, 250 Hz - 4 kHz), and within
    // 0.107 (0.05 plus the conversion's maximum error) of the chamber values [Dupont_2023_SoundFieldSynthesis#4]
    // converted to random incidence (tools/acoustic/sabine_to_random.py, 3 x 4 m, extended reaction; fitted sigma
    // 19048 Pa s/m2, 1.45 times the measured one). The synthesis values alone are printed: read as an infinite layer
    // they need 1.83 times the measured resistivity, and they lie 0.08-0.09 above the infinite JCA layer at 250 and
    // 500 Hz. That excess is attributed to the finite sample; the cause is bounded (diffraction of a 1.5 x 1.5 m patch in
    // a diffuse field adds up to 0.56), not measured. 25-200 mm against the JCA model itself; there is no reference at
    // other thicknesses. Printed: Komatsu, Miki and Delany-Bazley at 13112 Pa s/m2, the raw chamber values (above one)
    // and the 100 mm glass-fibre board [VanHout_2016_SuspendedCeilings#17] (Sabine, 100 kg/m3, no resistivity).
    // Reasonable meant: within the JCA prediction and the synthesis measurement of one specimen, which agree within
    // 0.06. Valid: 25-200 mm by the model, absorption 250 Hz - 4 kHz.
    const mt::Grid G = mt::grid();
    const mt::Row r = mt::row("glass_wool"), w0 = mt::with(mt::row("concrete"), "tf", 0.0);
    const mt::Row jca = {{"jca", 13112.0}, {"phi", 0.954}, {"tort", 1.0}, {"lv", 55.6e-6}, {"lt", 116e-6}};
    mt::check_material("glass_wool at 50 mm", r, {}, G, 0.05);
    mt::check_material("glass_wool at 100 mm", r, {}, G, 0.1);
    mt::check_material("glass_wool_50mm", mt::row("glass_wool_50mm"));
    mt::check_material("glass_wool_100mm", mt::row("glass_wool_100mm"));

    // The row against its model on a rigid backing (random incidence, 63 Hz - 8 kHz; measured 0.0088, bound 0.025)
    auto alpha = [&](const mt::Row &m, double d) { return mt::octaves(G, [&](size_t q) { double a = 0.0; for (int i = 0; i < G.na; ++i) { double T, R; mt::ref_tmm({{m, mt::SHEET, d}}, G.f[q], G.th[i], T, R, mt::HALF, &w0); a += G.w[i] * (1.0 - R); } return a; }); };
    for (double d : {0.025, 0.05, 0.1, 0.2})
    {
        const auto ar = alpha(r, d), am = alpha(jca, d), ak = alpha({{"komatsu", 13112.0}}, d), ai = alpha({{"miki", 13112.0}}, d), ad = alpha({{"db", 13112.0}}, d);
        QD_DIAG("\n# diag: glass_wool, %.0f mm on a rigid backing, random incidence\noctave,alpha_row,alpha_jca,alpha_komatsu,alpha_miki,alpha_delany_bazley\n", 1e3 * d);
        double worst = 0.0;
        for (int o = 1; o < 9; ++o)
            QD_DIAG("%g,%.4f,%.4f,%.4f,%.4f,%.4f\n", mt::OCT[o], ar[o], am[o], ak[o], ai[o], ad[o]), worst = std::max(worst, std::isnan(ar[o] - am[o]) ? INFINITY : std::abs(ar[o] - am[o]));
        INFO("glass_wool against JCA at " << 1e3 * d << " mm");
        CHECK(worst < 0.025);
    }

    // Tier 2: the 50 mm sheet on a rigid wall through the library
    const std::vector<double> oct = {125, 250, 500, 1000, 2000, 4000};
    const mt::Measured band{"Dupont_2023_SoundFieldSynthesis #1 and #3", "alpha_random", "band: JCA prediction (infinite layer) to sound-field synthesis (1.5 x 1.5 m)", false, 0.05, oct,
                            {{0.130, 0.130}, {0.347, 0.407}, {0.657, 0.720}, {0.887, 0.887}, {0.937, 0.945}, {0.940, 0.958}}};
    const mt::Measured conv{"Dupont_2023_SoundFieldSynthesis #4", "alpha_random", "Sabine (ISO 354) converted to random incidence (Jeong, 3 x 4 m)", false, 0.107, oct,
                            {{0.152, 0.152}, {0.370, 0.370}, {0.670, 0.670}, {0.887, 0.887}, {0.944, 0.944}, {0.950, 0.950}}};
    const mt::Out O = mt::run({{mt::row("glass_wool_50mm"), mt::SHEET, 0.05}, {w0, mt::SOLID, 0.15}}, G);
    mt::rows("glass_wool_50mm on a rigid wall", O);
    mt::tier2("glass_wool_50mm on a rigid wall", O, band, 0.0);
    mt::tier2("glass_wool_50mm on a rigid wall", O, conv, 0.0);
    const double sfs[6] = {NAN, 0.407, 0.720, 0.887, 0.937, 0.940}, sab[6] = {0.153, 0.587, 0.993, 1.073, 1.017, 0.987}, vh100[6] = {0.74, 1.10, 1.17, 1.13, 1.09, 1.07};
    const mt::Out H = mt::run({{mt::row("glass_wool_100mm"), mt::SHEET, 0.1}, {w0, mt::SOLID, 0.15}}, G);
    QD_DIAG("\n# diag: glass_wool printed: 50 mm against the synthesis values (finite sample) and the raw chamber values; 100 mm against the board (Sabine)\noctave,library_50mm,sfs,library_minus_sfs,sabine_raw,library_100mm,vanhout_100mm_sabine\n");
    for (int o = 0; o < 6; ++o)
        QD_DIAG("%g,%.4f,%.3f,%+.4f,%.3f,%.4f,%.2f\n", oct[o], O.a8[o + 2], sfs[o], O.a8[o + 2] - sfs[o], sab[o], H.a8[o + 2], vh100[o]);

    // ENGINE LIMIT (porous layer on a rigid wall): the 50 mm sheet adds too much insulation to a concrete wall at low
    // frequency; printed against the fluid layer on a limp plate, the layer alone in air asserted.
    mt::added_insulation("glass_wool_50mm to concrete 150 mm", {{mt::row("glass_wool_50mm"), mt::SHEET, 0.05}}, {{mt::row("concrete"), mt::SOLID, 0.15}}, G);
}
