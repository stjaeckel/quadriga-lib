// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_preset*"

// Verification suite for quadriga_lib::sos_preset.
//
// Pins down: the 12 names and table sizes, the ACF tables, the scaling with the decorrelation distance,
// the quality of the shipped frequency tables, that the tables are what the documented generation
// produces, float and double, input validation.
// Oracles:
//  - ACF: the three formulas, written here
//  - scaling: dist ~ d, freq ~ 1/d
//  - quality: the measures of sos_calc_mse (which has its own suite) against the frequency tables of the
//    MATLAB class qd_sos of QuaDRiGa, which serve as reference. The numbers in the table below were
//    measured on those tables with sos_calc_mse and 100 test directions (md/LOG_QD_SOS.md, WI-5). A preset
//    has to reach the reference value minus 10·log10( N_ref / N ), because the mean squared error of a
//    sum of N sinusoids scales with 1/N and the presets with 128 and 256 sinusoids are compared with
//    reference tables of 150 and 300. No allowance where the preset is the larger table.
//  - no sinusoid below the documented lower frequency limit, and a flat ACF beyond the fitted range
//  - statistics: the correlation of the values of sos_val between two positions over many phase sets
//    equals the target ACF within the fit error
//  - generation: the smallest tables are regenerated with sos_generate and the documented settings and
//    must have the same directions and an equivalent fit

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

namespace
{
    const double pi = 3.141592653589793;

    // Oracle: target ACF at distance D for the decorrelation distance d
    double ref_acf(int type, double D, double d)
    {
        double g = std::exp(-(D / d) * (D / d)), e = std::exp(-D / d);
        return type == 0 ? e : (type == 1 ? g : (D <= d ? g : e));
    }

    struct Preset
    {
        const char *name;
        int type;              // 0 = Exp, 1 = Gauss, 2 = Comb
        arma::uword n_coeff;   // Sinusoids in the preset
        arma::uword n_ref;     // Sinusoids in the reference table of qd_sos
        double ref_dir, ref_core, ref_all; // Reference table: sos_calc_mse with 100 directions, in dB
        arma::sword seed;      // Seed of the documented generation
    };

    const Preset presets[12] = {
        {"Exp128", 0, 128, 150, 27.20, 28.71, 25.35, 39},
        {"Exp256", 0, 256, 300, 30.59, 31.74, 27.34, 3},
        {"Exp512", 0, 512, 500, 32.80, 34.02, 27.91, 2},
        {"Exp1024", 0, 1024, 1000, 35.53, 36.86, 31.04, 5},
        {"Gauss128", 1, 128, 150, 29.39, 31.04, 24.37, 13},
        {"Gauss256", 1, 256, 300, 32.02, 34.19, 27.93, 32},
        {"Gauss512", 1, 512, 500, 34.14, 35.97, 29.70, 6},
        {"Gauss1024", 1, 1024, 1000, 37.77, 39.52, 31.08, 21},
        {"Comb128", 2, 128, 150, 29.33, 30.49, 24.23, 13},
        {"Comb256", 2, 256, 300, 32.36, 33.78, 26.41, 25},
        {"Comb512", 2, 512, 500, 34.48, 35.85, 27.69, 37},
        {"Comb1024", 2, 1024, 1000, 37.27, 38.73, 28.54, 47}};
}

TEST_CASE("sos_preset - all presets return the documented sizes and the ACF formulas")
{
    for (const Preset &p : presets)
    {
        arma::mat freq;
        arma::vec acf, dist;
        quadriga_lib::sos_preset(p.name, freq, &acf, &dist);
        INFO(p.name);
        REQUIRE(freq.n_rows == p.n_coeff);
        REQUIRE(freq.n_cols == 3);
        REQUIRE(acf.n_elem == 200);
        REQUIRE(dist.n_elem == 200);

        double max_d = 0.0, max_a = 0.0;
        for (arma::uword i = 0; i < 200; ++i)
        {
            max_d = std::max(max_d, std::abs(dist(i) - 0.25 * (double)i));
            max_a = std::max(max_a, std::abs(acf(i) - ref_acf(p.type, 0.25 * (double)i, 10.0)));
        }
        CHECK(max_d < 1e-13);
        CHECK(max_a < 1e-14);
        CHECK(acf(0) == 1.0);
        CHECK(std::abs(acf(40) - std::exp(-1.0)) < 1e-14); // 10 m
        CHECK(arma::abs(freq).max() > 0.01);
        CHECK(freq.is_finite());
    }
}

TEST_CASE("sos_preset - the decorrelation distance scales distances and frequencies")
{
    arma::mat f10, f25, f04;
    arma::vec a10, d10, a25, d25, a04, d04;
    quadriga_lib::sos_preset("Comb256", f10, &a10, &d10, 10.0);
    quadriga_lib::sos_preset("Comb256", f25, &a25, &d25, 25.0);
    quadriga_lib::sos_preset("Comb256", f04, &a04, &d04, 0.4);
    REQUIRE(f25.n_rows == 256);
    REQUIRE(d25.n_elem == 200);
    CHECK(arma::abs(f10).max() > 0.01);

    CHECK(arma::approx_equal(d25, d10 * 2.5, "absdiff", 1e-12));
    CHECK(arma::approx_equal(d04, d10 * 0.04, "absdiff", 1e-13));
    CHECK(arma::approx_equal(a25, a10, "absdiff", 1e-14));
    CHECK(arma::approx_equal(a04, a10, "absdiff", 1e-14));
    CHECK(arma::approx_equal(f25 * 2.5, f10, "reldiff", 1e-14));
    CHECK(arma::approx_equal(f04 * 0.04, f10, "reldiff", 1e-14));
    CHECK(std::abs(d25(199) - 124.375) < 1e-12);
}

TEST_CASE("sos_preset - the tables reach the fit of the reference tables and have no slow sinusoids")
{
    QD_DIAG("\n# diag: sos_preset quality, sos_calc_mse with 100 test directions, values in dB\n");
    QD_DIAG("preset,mse_dir,ref_dir,mse_core,ref_core,mse_all,ref_all,allowance,tail2,tail4,tail10,tail20,min_f_d,max_f_d\n");
    for (const Preset &p : presets)
    {
        arma::mat freq;
        arma::vec acf, dist, tail;
        quadriga_lib::sos_preset(p.name, freq, &acf, &dist);
        REQUIRE(freq.n_rows == p.n_coeff);

        double core = 0.0, all = 0.0;
        double mse = quadriga_lib::sos_calc_mse(freq, acf, dist, &core, &all, &tail, (arma::mat *)nullptr, 100);
        REQUIRE(tail.n_elem == 4);

        double f_lo = 1e9, f_hi = 0.0;
        for (arma::uword n = 0; n < p.n_coeff; ++n)
        {
            double f = 10.0 * std::sqrt(freq(n, 0) * freq(n, 0) + freq(n, 1) * freq(n, 1) + freq(n, 2) * freq(n, 2));
            f_lo = std::min(f_lo, f), f_hi = std::max(f_hi, f);
        }

        // Allowance for the smaller number of sinusoids, 0.69 dB for 128 against 150 and 256 against 300
        double allowance = p.n_coeff < p.n_ref ? 10.0 * std::log10((double)p.n_ref / (double)p.n_coeff) : 0.0;
        QD_DIAG("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n", p.name, mse, p.ref_dir, core, p.ref_core,
                all, p.ref_all, allowance, tail(0), tail(1), tail(2), tail(3), f_lo, f_hi);

        INFO(p.name);
        CHECK(mse >= p.ref_dir - allowance);
        CHECK(core >= p.ref_core - allowance);
        CHECK(all >= p.ref_all - allowance);

        // Lower limit 0.05 / d and upper limit 15 / 49.75 m of the generation; the reference tables have
        // between 1 and 31 sinusoids below 0.01 / d
        CHECK(f_lo >= 0.05 - 1e-9);
        CHECK(f_hi <= 150.0 / 49.75 + 1e-9);

        // Beyond the fitted range the ACF stays at the level of the fit error; the reference tables show
        // plateaus of up to 0.055
        CHECK(arma::abs(tail).max() < 0.015);
    }
}

TEST_CASE("sos_preset - the correlation of the generated values follows the target ACF")
{
    // 20000 phase sets, pairs of positions at 2.5, 5, 10, 20 and 40 m along three different directions
    const arma::uword n_gen = 20000;
    const double dd[5] = {2.5, 5.0, 10.0, 20.0, 40.0};
    const double dir[3][3] = {{1.0, 0.0, 0.0}, {0.6, 0.8, 0.0}, {0.36, -0.48, 0.8}};

    std::mt19937_64 rng(1);
    arma::mat phase(256, n_gen);
    for (arma::uword i = 0; i < phase.n_elem; ++i)
        phase(i) = 2.0 * pi * ((double)(rng() >> 11) / 9007199254740992.0 - 0.5);

    QD_DIAG("\n# diag: sos_preset correlation over %llu phase sets, decorrelation distance 10 m\n", (unsigned long long)n_gen);
    QD_DIAG("preset,direction,distance,correlation,target,diff\n");
    for (const char *name : {"Exp256", "Gauss256", "Comb256"})
    {
        int type = name[0] == 'E' ? 0 : (name[0] == 'G' ? 1 : 2);
        arma::mat freq, pos(16, 3), val;
        quadriga_lib::sos_preset(name, freq);
        REQUIRE(freq.n_rows == 256);
        pos.row(0) = arma::rowvec({123.0, -45.0, 6.0});
        for (arma::uword k = 0; k < 3; ++k)
            for (arma::uword j = 0; j < 5; ++j)
                pos.row(1 + 5 * k + j) = pos.row(0) + dd[j] * arma::rowvec({dir[k][0], dir[k][1], dir[k][2]});
        quadriga_lib::sos_val(freq, phase, pos, val);
        REQUIRE(val.n_rows == 16);
        REQUIRE(val.n_cols == n_gen);

        for (arma::uword k = 0; k < 3; ++k)
            for (arma::uword j = 0; j < 5; ++j)
            {
                double cov = 0.0;
                for (arma::uword g = 0; g < n_gen; ++g)
                    cov += val(0, g) * val(1 + 5 * k + j, g) / (double)n_gen;
                double target = ref_acf(type, dd[j], 10.0);
                QD_DIAG("%s,%llu,%.1f,%.4f,%.4f,%+.4f\n", name, (unsigned long long)k, dd[j], cov, target, cov - target);

                // Fit error of the 256 tables: 31 to 36 dB, i.e. an rms deviation of 0.016 to 0.028, with
                // single directions up to about twice that; Monte-Carlo error 0.01. Limit 0.08.
                INFO(name << ", direction " << k << ", distance " << dd[j]);
                CHECK(std::abs(cov - target) < 0.08);
            }
    }
}

TEST_CASE("sos_preset - the documented generation gives tables of the same quality")
{
    // The optimiser takes accept/reject decisions, so a difference in the last bit of its input or of the
    // math library can lead to another table. A bit-wise reproduction can therefore not be asserted across
    // platforms; the difference is printed. Asserted is what must hold everywhere: the regenerated table
    // has the same directions and limits and a fit within 1.5 dB of the shipped table (the spread of the
    // fit over 48 seeds was 1.5 to 3 dB for the 128 tables).
    // The tables with 128 and 256 sinusoids are regenerated here (6 s on one core). Those with 512 and 1024
    // take another 60 s; they were regenerated once, on 2026-10-12, with a largest difference of 1.1e-16
    // (md/LOG_QD_SOS.md) and are pinned by the checksums below.
    for (arma::uword i : {0, 1, 4, 5, 8, 9})
    {
        const Preset &p = presets[i];
        arma::mat freq, regen;
        arma::vec acf, dist;
        quadriga_lib::sos_preset(p.name, freq, &acf, &dist);
        REQUIRE(freq.n_rows == p.n_coeff);
        quadriga_lib::sos_generate(acf, dist, p.n_coeff, regen, (arma::vec *)nullptr, (arma::mat *)nullptr, 3, true, 300, true, 0.005, 15.0 / 49.75, p.seed);
        REQUIRE(regen.n_rows == p.n_coeff);

        double mse_ship = quadriga_lib::sos_calc_mse(freq, acf, dist, (double *)nullptr, (double *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr, 100);
        double mse_regen = quadriga_lib::sos_calc_mse(regen, acf, dist, (double *)nullptr, (double *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr, 100);
        double mx = arma::abs(regen - freq).max(), max_cross = 0.0;
        for (arma::uword n = 0; n < p.n_coeff; ++n)
        {
            // Same direction per sinusoid: the cross product of the two rows vanishes
            double cx = freq(n, 1) * regen(n, 2) - freq(n, 2) * regen(n, 1), cy = freq(n, 2) * regen(n, 0) - freq(n, 0) * regen(n, 2);
            double cz = freq(n, 0) * regen(n, 1) - freq(n, 1) * regen(n, 0);
            max_cross = std::max(max_cross, std::sqrt(cx * cx + cy * cy + cz * cz));
        }
        QD_DIAG("\n# diag: sos_preset %s regenerated with seed %lld: max abs difference %.3e, fit %.2f dB shipped, %.2f dB regenerated\n",
                p.name, (long long)p.seed, mx, mse_ship, mse_regen);
        INFO(p.name);
        CHECK(max_cross < 1e-15);
        CHECK(std::abs(mse_regen - mse_ship) < 1.5);
        CHECK(arma::abs(freq).max() > 0.01);
    }
}

TEST_CASE("sos_preset - optional outputs, pre-sized outputs and float")
{
    arma::mat ref;
    arma::vec acf, dist;
    quadriga_lib::sos_preset("Gauss128", ref, &acf, &dist, 7.0);
    REQUIRE(ref.n_rows == 128);
    CHECK(arma::abs(ref).max() > 0.01);

    arma::mat only;
    quadriga_lib::sos_preset("Gauss128", only, (arma::vec *)nullptr, (arma::vec *)nullptr, 7.0);
    CHECK(arma::approx_equal(only, ref, "absdiff", 0.0));

    arma::mat sized(128, 3);
    sized.fill(9.0);
    const double *ptr = sized.memptr();
    arma::vec a2(3), d2(200);
    quadriga_lib::sos_preset("Gauss128", sized, &a2, &d2, 7.0);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, ref, "absdiff", 0.0));
    CHECK(arma::approx_equal(a2, acf, "absdiff", 0.0));
    CHECK(arma::approx_equal(d2, dist, "absdiff", 0.0));

    arma::fmat ff;
    arma::fvec af, df;
    quadriga_lib::sos_preset("Gauss128", ff, &af, &df, 7.0f);
    REQUIRE(ff.n_rows == 128);
    REQUIRE(af.n_elem == 200);
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(ff) - ref).max() < 1e-7);
    CHECK(arma::abs(arma::conv_to<arma::vec>::from(af) - acf).max() < 1e-7);
    CHECK(arma::abs(arma::conv_to<arma::vec>::from(df) - dist).max() < 1e-5);
}

TEST_CASE("sos_preset - input validation")
{
    arma::mat freq;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_preset("Comb256", freq));
        CHECK_NOTHROW(quadriga_lib::sos_preset("Exp1024", freq, (arma::vec *)nullptr, (arma::vec *)nullptr, 0.1));
    }
    SECTION("name")
    {
        for (const char *bad : {"", "Comb", "Comb300", "comb256", "Comb256 ", "Gaus256", "Exp12", "256"})
        {
            INFO("name '" << bad << "'");
            CHECK_THROWS_AS(quadriga_lib::sos_preset(bad, freq), std::invalid_argument);
        }
    }
    SECTION("dist_decorr")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_preset("Comb256", freq, (arma::vec *)nullptr, (arma::vec *)nullptr, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_preset("Comb256", freq, (arma::vec *)nullptr, (arma::vec *)nullptr, -10.0), std::invalid_argument);
    }
}

TEST_CASE("sos_preset - the stored tables have the recorded checksums")
{
    // The quality checks above accept any table that fits well enough, so a changed digit in a stored value
    // would pass them. Here the stored values are recovered from the output for 10 m (signed frequency of
    // sinusoid n = its row of 'freq' projected on direction n of the lattice, written out below) and their
    // sum and sum of squares are compared with the values recorded on 2026-10-12 from the shipped tables.
    // Tolerance: 1e-15 relative, for the sum relative to the sum of the magnitudes (the signs cancel). The
    // recovery costs about 3 roundings per value of 1e-16 relative each, so a change of one value in its
    // 13th significant digit or above is found.
    struct Checksum
    {
        const char *name;
        double sum, sum_sq;
    };
    const Checksum checksums[12] = {
        {"Exp128", -1.1755402660477672, 0.52583332308168751},
        {"Exp256", 0.88464156886416656, 1.3262291202441001},
        {"Exp512", -2.3764133012248845, 3.4268328544980253},
        {"Exp1024", -0.39260313311863704, 9.0433012007154794},
        {"Gauss128", -0.031945837392546278, 0.18176846417403708},
        {"Gauss256", 0.63440497652643635, 0.37392721771931497},
        {"Gauss512", -0.28576905590346352, 0.76676763658347047},
        {"Gauss1024", 0.63041768449333779, 1.5367689503845079},
        {"Comb128", 0.71261646207751705, 0.19244971340848105},
        {"Comb256", -0.62463800916330869, 0.40855176059174952},
        {"Comb512", 0.49710400165227242, 0.82900975893720374},
        {"Comb1024", 1.6375798044747341, 1.6694938051669446}};

    QD_DIAG("\n# diag: sos_preset checksums of the stored values\npreset,sum,recorded,rel_diff,sum_sq,recorded,rel_diff\n");
    for (const Checksum &c : checksums)
    {
        arma::mat freq;
        quadriga_lib::sos_preset(c.name, freq);
        const arma::uword n_coeff = freq.n_rows;
        double sum = 0.0, sum_sq = 0.0, sum_abs = 0.0;
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            // Direction n of the spiral lattice: sin(elevation) = (n + 0.5) / n_coeff, azimuth = n · pi · (3 - sqrt(5))
            double el = std::asin(((double)n + 0.5) / (double)n_coeff), az = (double)n * pi * (3.0 - std::sqrt(5.0));
            double v = freq(n, 0) * std::cos(az) * std::cos(el) + freq(n, 1) * std::sin(az) * std::cos(el) + freq(n, 2) * std::sin(el);
            sum += v, sum_sq += v * v, sum_abs += std::abs(v);
        }
        double d1 = std::abs(sum - c.sum) / sum_abs, d2 = std::abs(sum_sq - c.sum_sq) / c.sum_sq;
        QD_DIAG("%s,%.17g,%.17g,%.1e,%.17g,%.17g,%.1e\n", c.name, sum, c.sum, d1, sum_sq, c.sum_sq, d2);
        INFO(c.name);
        CHECK(d1 < 1e-15);
        CHECK(d2 < 1e-15);
    }
}

TEST_CASE("sos_preset - outputs that share memory with each other throw")
{
    // The three outputs are written one after the other; an output on the memory of another one (or the same
    // object) overwrote it. Such calls now throw before anything is written.
    arma::mat freq;
    arma::vec acf, dist;
    quadriga_lib::sos_preset<double>("Exp128", freq, &acf, &dist);

    auto message = [](arma::mat &f, arma::vec *a, arma::vec *d) -> std::string
    {
        try
        {
            quadriga_lib::sos_preset<double>("Exp128", f, a, d);
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        return "no error";
    };

    arma::mat f1;
    arma::vec same(200, arma::fill::value(7.0));
    CHECK(message(f1, &same, &same) == "Output 'dist' cannot share memory with 'acf'.");
    CHECK(arma::all(same == 7.0));
    CHECK(f1.n_elem == 0); // nothing written

    // 'acf' on part of the memory of 'freq'
    arma::mat f2(128, 3, arma::fill::value(5.0));
    arma::vec a2(f2.memptr() + 100, 200, false, true), d2;
    CHECK(message(f2, &a2, &d2) == "Output 'acf' cannot share memory with 'freq'.");
    CHECK(arma::all(arma::vectorise(f2) == 5.0));

    // 'dist' overlaps the end of 'acf'
    arma::vec block(300, arma::fill::value(3.0));
    arma::vec a3(block.memptr(), 200, false, true), d3(block.memptr() + 150, 150, false, true);
    arma::mat f3;
    CHECK(message(f3, &a3, &d3) == "Output 'dist' cannot share memory with 'acf'.");
    CHECK(arma::all(block == 3.0));

    // Separate outputs next to each other in one block are accepted
    arma::vec block4(400);
    arma::vec a4(block4.memptr(), 200, false, true), d4(block4.memptr() + 200, 200, false, true);
    arma::mat f4;
    quadriga_lib::sos_preset<double>("Exp128", f4, &a4, &d4);
    CHECK(arma::approx_equal(a4, acf, "absdiff", 0.0));
    CHECK(arma::approx_equal(d4, dist, "absdiff", 0.0));
}
