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
