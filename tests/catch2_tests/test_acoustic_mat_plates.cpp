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
