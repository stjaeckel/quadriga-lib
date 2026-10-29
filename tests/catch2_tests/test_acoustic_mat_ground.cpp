// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Material cases of the ground group (ground classes, paving, vegetation); the rig is acoustic_material_rig.hpp.
//   QD_DIAG_RAY=1 ./build_linux/test_bin "[materials]"

#include "acoustic_material_rig.hpp"

TEST_CASE("Material - ground_snow + ground_forest_floor + ground_grass + ground_grass_dry + ground_soil + ground_gravel_loose + ground_gravel_compacted",
          "[acoustic][materials]")
{
    // Ground, class 6: porous half-spaces, each a traced solid (thk 0) drawn at least 1 m thick (from 25 Hz within
    // 0.0014 of its half-space in R). The classes are Delany-Bazley at a class flow resistivity (Harmonoise /
    // Nord2000): snow A 12.5, forest floor B 31.5, grass D 200 and dry grass C 80, soil E 500, compacted gravel
    // F 2000 kPa s/m2 [Fuchs_2025_GroundImpedanceNTACOU104#2] to #7 (CNOSSOS G 1, 1, 1, 1, 0.7, 0.3
    // [Kephalopoulos_2012_CNOSSOS#1] to #6); loose gravel 10 kPa s/m2 [VanRenterghem_2015_NaturalMeans#4]. Columns:
    // power laws fitted at 342.76271 m/s to the complex plane-wave reflection of the Delany-Bazley half-space (extended
    // reaction) at 0, 30, 60 and 80-89.9 deg, 25 Hz - 10 kHz, with Re mu >= 1 and a weak pull of n towards
    // Delany-Bazley. Below a frequency that rises with sigma (to 8 kHz at F) Delany-Bazley's impedance phase lies
    // beyond -45 deg (to -50.8), which no passive medium reaches: the rows miss it by up to 0.065 in R, at most 0.99 dB
    // in the octave ground effect of a source and a receiver 20-200 m apart (plane waves). Their effective
    // Delany-Bazley resistivity is within 1.05 times the class's. The quantity asserted is the complex reflection at
    // the angles of a ground path and at 0 deg, per third (bound 0.15); a locally reacting Delany-Bazley face (how the
    // classes were fitted) and Miki at the class resistivity are printed. Only flow resistivities are measured, no
    // reflection spectra (DB-type fits unless noted, kPa s/m2): snow 15-30 and 25-50 [Piercy_1981_PropagationReview#3],
    // #11; forest floor 40, 63 [Nordtest_1999_NTACOU104#7], #1, 20-80 Piercy #5; grass 150-300 Piercy #6, 228 and 277
    // [Bashir_2014_PhDRoughMixedImpedance#9], #12, lawn 80 in summer and 200 in winter (layer fits)
    // [Attenborough_2016_ExploitingGroundEffects#1], #2; soil 630 and 2000 Nordtest #3, #2, 106-450
    // [FHWA_2010_GroundPavementEffects#7]; compacted gravel 2000-4000 Piercy #7; loose gravel 1.5-59 (airflow)
    // [Bashir_2014_PhDRoughMixedImpedance#155]. Loose gravel is a half-space: a bed of 0.1 m on soil reflects up to
    // 4.3 dB differently in the ground effect, 0.5 m 0.1 dB. Reasonable meant: within the class definition, against
    // which the rows' own fit floor is the limit. Valid: 31.5 Hz - 10 kHz, solids of 1 m or more.
    const mt::Grid G = mt::grid();
    mt::Grid Gg = mt::grid(50, 1, true); // the angles of a ground path, report angles only
    Gg.th = {0.0, 80.0, 85.0, 88.0, 89.0, 89.5, 89.9}, Gg.w.assign(Gg.th.size(), 0.0);
    auto r_half = [](const mt::Row &M, double fa, double th, bool local) // the complex reflection of a half-space
    {
        mt::cd e, m;
        mt::eps_mu(M, fa, e, m);
        const double Za = 1.0 / std::cos(mt::rad(th));
        const mt::cd Zh = local ? std::sqrt(m / e) : m / mt::q_of(e, m, std::sin(mt::rad(th)));
        return (Zh - Za) / (Zh + Za);
    };
    const std::vector<std::pair<const char *, double>> rows = {{"ground_snow", 12.5e3}, {"ground_forest_floor", 31.5e3}, {"ground_grass", 200e3},
                                                               {"ground_grass_dry", 80e3}, {"ground_soil", 500e3}, {"ground_gravel_loose", 10e3},
                                                               {"ground_gravel_compacted", 2e6}};
    for (const auto &rs : rows)
    {
        const std::string n = rs.first;
        const mt::Row r = mt::row(n), db = {{"db", rs.second}}, mk = {{"miki", rs.second}};
        const mt::Stack st = {{r, mt::SOLID, 1.0}};
        mt::check_material(n + " at 1 m", r, {}, G, 1.0);
        const mt::Out O = mt::run(st, Gg);
        QD_DIAG("\n# diag: %s at 1 m on the angles of a ground path against the fluid layer\n%s", n.c_str(), mt::DEV_HEAD);
        const mt::Dev D = mt::against((n + " at 1 m, ground path").c_str(), O, st);
        // the library against Delany-Bazley (asserted), its locally reacting face and Miki (printed): per angle the
        // largest |R - R_model| (complex) over each octave's thirds
        QD_DIAG("\n# diag: %s, |R_library - R_model| (complex), the largest per octave\nangle,model", n.c_str());
        for (double f : mt::OCT)
            QD_DIAG(",%g", f);
        double worst = 0.0;
        for (size_t i = 0; i < Gg.th.size(); ++i)
            for (int k = 0; k < 3; ++k)
            {
                std::array<double, 9> mx{};
                for (size_t q = 0; q < Gg.f.size(); ++q)
                    if (Gg.f[q] > 28.0) // 31.5 Hz - 10 kHz; a NaN fails
                    {
                        const double d = std::abs(O.R[i][q] - r_half(k == 2 ? mk : db, Gg.f[q], Gg.th[i], k == 1));
                        mx[Gg.third[q] / 3] = std::max(mx[Gg.third[q] / 3], std::isnan(d) ? INFINITY : d);
                    }
                QD_DIAG("\n%.1f,%s", Gg.th[i], k == 0 ? "delany_bazley" : k == 1 ? "delany_bazley_local" : "miki");
                for (double v : mx)
                    QD_DIAG(",%.4f", v);
                worst = k == 0 ? std::max(worst, *std::max_element(mx.begin(), mx.end())) : worst;
            }
        QD_DIAG("\n# %s: worst |R_library - R_delany_bazley| %.4f\n", n.c_str(), worst);
        INFO(n << " at 1 m on the angles of a ground path");
        CHECK(D.dR < 9e-3);
        CHECK(worst < 0.15);
    }
}
