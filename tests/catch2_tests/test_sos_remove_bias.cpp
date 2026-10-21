// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_remove_bias*"

// Verification suite for quadriga_lib::sos_remove_bias.
//
// Pins down: that the sample mean is zero after the call in all three device modes, the closed form of
// the offset, that one common offset is applied per generator, the phase range, the two diagnostic
// outputs, the documented effect on the variance, float and double, input validation.
// Oracles, independent of the function:
//  - the mean of the values of sos_val (GENERIC kernel, own suite) over the positions, before and after
//  - the closed form offset = pi/2 - arg(Z), with Z accumulated here in scalar complex arithmetic
//  - a brute-force scan: the mean as a function of a common phase offset, sampled on 3600 points, has
//    its zero where the function put it
//  - the variance after the correction, 1 - mean(c^2)/v, with c and v computed here from the ACF of
//    the sinusoids
// Conventions: frequencies random up to 0.05 / m (decorrelation distance about 10 m), positions inside
// a square of 60 m at 1.5 m height unless stated.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_SOS")

namespace
{
    const double pi = 3.141592653589793;

    // Uniform random matrix in [lo, hi), own generator so that the fixtures do not depend on the library
    arma::mat rnd(arma::uword n_rows, arma::uword n_cols, double lo, double hi, unsigned seed)
    {
        std::mt19937_64 rng(seed);
        arma::mat m(n_rows, n_cols);
        for (arma::uword i = 0; i < m.n_elem; ++i)
            m(i) = lo + (hi - lo) * ((double)(rng() >> 11) / 9007199254740992.0);
        return m;
    }

    arma::mat positions(arma::uword n, double edge, unsigned seed)
    {
        arma::mat p = rnd(n, 3, -0.5 * edge, 0.5 * edge, seed);
        p.col(2).fill(1.5);
        return p;
    }

    // Mean over the positions of the values of sos_val for generator g (GENERIC kernel)
    double mean_val(const arma::mat &freq, const arma::mat &phase, arma::uword g, const arma::mat &pos, const arma::mat *pos_b = nullptr,
                    const arma::mat *phase_b = nullptr, const arma::vec *acf = nullptr, const arma::vec *dist = nullptr, double scale = 1.0)
    {
        arma::mat ph = phase.col(g), phb, val;
        if (phase_b != nullptr)
            phb = phase_b->col(g);
        quadriga_lib::sos_val(freq, ph, pos, val, pos_b, phase_b != nullptr ? &phb : (arma::mat *)nullptr, acf, dist, scale, false, 1);
        return arma::mean(val.col(0));
    }

    // Oracle: Z = sum_n exp( j·phi_n ) · mean_p exp( j·2·pi · f_n·x_p ) for one device
    std::complex<double> ref_Z(const arma::mat &freq, const arma::mat &phase, arma::uword g, const arma::mat &pos, double scale = 1.0)
    {
        std::complex<double> Z(0.0, 0.0);
        for (arma::uword n = 0; n < freq.n_rows; ++n)
        {
            std::complex<double> z(0.0, 0.0);
            for (arma::uword p = 0; p < pos.n_rows; ++p)
            {
                double arg = 0.0;
                for (arma::uword c = 0; c < pos.n_cols; ++c)
                    arg += scale * freq(n, c) * pos(p, c);
                z += std::polar(1.0, 2.0 * pi * arg);
            }
            Z += z / (double)pos.n_rows * std::polar(1.0, phase(n, g));
        }
        return Z;
    }

    // Distance between two angles modulo pi
    double diff_mod_pi(double a, double b)
    {
        double d = std::remainder(a - b, pi);
        return std::abs(d);
    }
}

TEST_CASE("sos_remove_bias - one device: the mean vanishes and the offset is the closed form")
{
    arma::mat freq = rnd(48, 3, -0.05, 0.05, 1), phase0 = rnd(48, 5, -pi, pi, 2), pos = positions(40, 60.0, 3);
    arma::mat phase = phase0, offset, before;
    quadriga_lib::sos_remove_bias(freq, phase, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset, &before);
    REQUIRE(phase.n_rows == 48);
    REQUIRE(phase.n_cols == 5);
    REQUIRE(offset.n_rows == 5);
    REQUIRE(offset.n_cols == 1);
    REQUIRE(before.n_rows == 5);
    REQUIRE(before.n_cols == 1);

    QD_DIAG("\n# diag: sos_remove_bias one device\ngen,mean_before,reported,mean_after,offset,closed_form\n");
    for (arma::uword g = 0; g < 5; ++g)
    {
        double m0 = mean_val(freq, phase0, g, pos), m1 = mean_val(freq, phase, g, pos);
        std::complex<double> Z = ref_Z(freq, phase0, g, pos);
        double closed = pi / 2.0 - std::arg(Z);
        QD_DIAG("%llu,%.6f,%.6f,%.2e,%.6f,%.6f\n", (unsigned long long)g, m0, before(g, 0), m1, offset(g, 0), std::remainder(closed, pi));
        INFO("generator " << g);
        CHECK(std::abs(m0) > 1e-3);                      // There was something to remove
        CHECK(std::abs(m1) < 1e-13);                     // 48 terms of magnitude 0.2, double precision
        CHECK(std::abs(before(g, 0) - m0) < 1e-13);
        CHECK(diff_mod_pi(offset(g, 0), closed) < 1e-12);
        CHECK(std::abs(offset(g, 0)) <= pi / 2.0 + 1e-12); // The smaller of the two roots

        // One common offset on all phases of the generator, result wrapped to [-pi, pi)
        double max_dev = 0.0;
        for (arma::uword n = 0; n < 48; ++n)
            max_dev = std::max(max_dev, std::abs(std::remainder(phase(n, g) - phase0(n, g) - offset(g, 0), 2.0 * pi)));
        CHECK(max_dev < 1e-13);
    }
    CHECK(phase.min() >= -pi);
    CHECK(phase.max() < pi);
}

TEST_CASE("sos_remove_bias - a brute-force scan over the offset finds the same zero")
{
    arma::mat freq = rnd(32, 3, -0.05, 0.05, 4), phase0 = rnd(32, 1, -pi, pi, 5), pos = positions(25, 40.0, 6);
    arma::mat phase = phase0, offset;
    quadriga_lib::sos_remove_bias(freq, phase, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset);
    REQUIRE(offset.n_elem == 1);

    // The mean as a function of the offset is a sinusoid with period 2·pi and two zeros half a period apart
    const arma::uword n_scan = 3600;
    double best = 1e9, best_offset = 0.0, largest = 0.0;
    for (arma::uword k = 0; k < n_scan; ++k)
    {
        double delta = -pi + 2.0 * pi * (double)k / (double)n_scan;
        arma::mat ph = phase0 + delta;
        double m = std::abs(mean_val(freq, ph, 0, pos));
        if (m < best)
            best = m, best_offset = delta;
        largest = std::max(largest, m);
    }
    QD_DIAG("\n# diag: sos_remove_bias scan: zero at %.5f rad (abs mean %.2e, largest %.3f), function returned %.5f rad\n",
            best_offset, best, largest, offset(0, 0));
    CHECK(largest > 0.01);
    CHECK(diff_mod_pi(best_offset, offset(0, 0)) <= 2.0 * pi / (double)n_scan); // Within one scan step
    CHECK(std::abs(mean_val(freq, phase, 0, pos)) < best + 1e-13);              // At least as good as the scan
}

TEST_CASE("sos_remove_bias - two devices on the same map: the mean of the combined value vanishes")
{
    arma::mat freq = rnd(48, 3, -0.05, 0.05, 7), phase0 = rnd(48, 4, -pi, pi, 8);
    arma::mat pa = positions(40, 60.0, 9), pb = positions(40, 60.0, 10);
    pb.row(0) = pa.row(0); // One pair at the same place
    arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);

    arma::mat phase = phase0, offset, before;
    quadriga_lib::sos_remove_bias(freq, phase, pa, &pb, (arma::mat *)nullptr, &acf, &dist, &offset, &before);
    REQUIRE(offset.n_rows == 4);
    REQUIRE(offset.n_cols == 1);
    REQUIRE(before.n_cols == 1);

    QD_DIAG("\n# diag: sos_remove_bias same map\ngen,mean_before,reported,mean_after,offset\n");
    for (arma::uword g = 0; g < 4; ++g)
    {
        double m0 = mean_val(freq, phase0, g, pa, &pb, nullptr, &acf, &dist), m1 = mean_val(freq, phase, g, pa, &pb, nullptr, &acf, &dist);
        QD_DIAG("%llu,%.6f,%.6f,%.2e,%.6f\n", (unsigned long long)g, m0, before(g, 0), m1, offset(g, 0));
        INFO("generator " << g);
        CHECK(std::abs(m0) > 1e-3);
        CHECK(std::abs(m1) < 1e-13);
        CHECK(std::abs(before(g, 0) - m0) < 1e-13);
        CHECK(std::abs(offset(g, 0)) <= pi / 2.0 + 1e-12);
        double max_dev = 0.0;
        for (arma::uword n = 0; n < 48; ++n)
            max_dev = std::max(max_dev, std::abs(std::remainder(phase(n, g) - phase0(n, g) - offset(g, 0), 2.0 * pi)));
        CHECK(max_dev < 1e-13);
    }

    // The single-device mean is not zero in this mode: the correction belongs to the combined value
    CHECK(std::abs(mean_val(freq, phase, 0, pa)) > 1e-6);
}

TEST_CASE("sos_remove_bias - two devices on different maps: each device has zero mean on its own positions")
{
    arma::mat freq = rnd(48, 3, -0.05, 0.05, 11), pa0 = rnd(48, 3, -pi, pi, 12), pb0 = rnd(48, 3, -pi, pi, 13);
    arma::mat pos_a = positions(40, 60.0, 14), pos_b = positions(40, 60.0, 15), pos_b25 = positions(25, 60.0, 16);

    for (double scale : {1.0, 2.5})
    {
        arma::mat ph_a = pa0, ph_b = pb0, offset, before;
        quadriga_lib::sos_remove_bias(freq, ph_a, pos_a, &pos_b, &ph_b, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset, &before, scale);
        REQUIRE(offset.n_rows == 3);
        REQUIRE(offset.n_cols == 2);
        REQUIRE(before.n_rows == 3);
        REQUIRE(before.n_cols == 2);

        arma::mat freq_b = freq * scale;
        QD_DIAG("\n# diag: sos_remove_bias two maps, frequency scale %.1f\ngen,before_a,before_b,after_a,after_b,after_combined,offset_a,offset_b\n", scale);
        for (arma::uword g = 0; g < 3; ++g)
        {
            double a0 = mean_val(freq, pa0, g, pos_a), b0 = mean_val(freq_b, pb0, g, pos_b);
            double a1 = mean_val(freq, ph_a, g, pos_a), b1 = mean_val(freq_b, ph_b, g, pos_b);
            double comb = mean_val(freq, ph_a, g, pos_a, &pos_b, &ph_b, nullptr, nullptr, scale);
            QD_DIAG("%llu,%.6f,%.6f,%.2e,%.2e,%.2e,%.5f,%.5f\n", (unsigned long long)g, a0, b0, a1, b1, comb, offset(g, 0), offset(g, 1));
            INFO("scale " << scale << ", generator " << g);
            CHECK(std::abs(a0) > 1e-3);
            CHECK(std::abs(b0) > 1e-3);
            CHECK(std::abs(a1) < 1e-13);
            CHECK(std::abs(b1) < 1e-13);
            CHECK(std::abs(comb) < 1e-13);
            CHECK(std::abs(before(g, 0) - a0) < 1e-13);
            CHECK(std::abs(before(g, 1) - b0) < 1e-13);
            CHECK(diff_mod_pi(offset(g, 0), pi / 2.0 - std::arg(ref_Z(freq, pa0, g, pos_a))) < 1e-12);
            CHECK(diff_mod_pi(offset(g, 1), pi / 2.0 - std::arg(ref_Z(freq, pb0, g, pos_b, scale))) < 1e-12);
        }
        CHECK(ph_b.min() >= -pi);
        CHECK(ph_b.max() < pi);
        CHECK(arma::abs(ph_b - pb0).max() > 1e-3);
    }

    // The two devices may have different numbers of positions in this mode
    arma::mat ph_a = pa0, ph_b = pb0;
    quadriga_lib::sos_remove_bias(freq, ph_a, pos_a, &pos_b25, &ph_b);
    CHECK(std::abs(mean_val(freq, ph_a, 1, pos_a)) < 1e-13);
    CHECK(std::abs(mean_val(freq, ph_b, 1, pos_b25)) < 1e-13);
}

TEST_CASE("sos_remove_bias - the variance after the correction is that of a field with zero sample mean")
{
    // 4000 phase sets, 30 positions inside 20 m with a decorrelation distance of about 10 m. Documented:
    // variance at position x after the correction = 1 - c(x)^2 / v, with R the ACF of the sinusoids,
    // c(x) = mean_p R( x - x_p ) and v = mean_pq R( x_p - x_q ). Here that removes almost half the variance.
    const arma::uword n_coeff = 256, n_gen = 4000, n_pos = 30;
    arma::mat freq;
    quadriga_lib::sos_preset("Comb256", freq);
    REQUIRE(freq.n_rows == n_coeff);
    arma::mat phase = rnd(n_coeff, n_gen, -pi, pi, 17), pos = positions(n_pos, 20.0, 18), val0, val1;

    quadriga_lib::sos_val(freq, phase, pos, val0, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    quadriga_lib::sos_remove_bias(freq, phase, pos);
    quadriga_lib::sos_val(freq, phase, pos, val1, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    REQUIRE(val1.n_rows == n_pos);
    REQUIRE(val1.n_cols == n_gen);

    // Oracle from the ACF of the sinusoids
    arma::mat R(n_pos, n_pos);
    for (arma::uword p = 0; p < n_pos; ++p)
        for (arma::uword q = 0; q < n_pos; ++q)
        {
            double r = 0.0;
            for (arma::uword n = 0; n < n_coeff; ++n)
                r += std::cos(2.0 * pi * (freq(n, 0) * (pos(p, 0) - pos(q, 0)) + freq(n, 1) * (pos(p, 1) - pos(q, 1)) + freq(n, 2) * (pos(p, 2) - pos(q, 2))));
            R(p, q) = r / (double)n_coeff;
        }
    double v = arma::accu(R) / (double)(n_pos * n_pos), pred = 0.0, var0 = 0.0, var1 = 0.0, worst_mean = 0.0;
    for (arma::uword p = 0; p < n_pos; ++p)
    {
        double c = arma::accu(R.row(p)) / (double)n_pos;
        pred += (1.0 - c * c / v) / (double)n_pos;
    }
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        double m = 0.0;
        for (arma::uword p = 0; p < n_pos; ++p)
        {
            var0 += val0(p, g) * val0(p, g) / (double)(n_pos * n_gen), var1 += val1(p, g) * val1(p, g) / (double)(n_pos * n_gen);
            m += val1(p, g) / (double)n_pos;
        }
        worst_mean = std::max(worst_mean, std::abs(m));
    }
    QD_DIAG("\n# diag: sos_remove_bias variance, %llu phase sets\nvariance_of_mean,variance_before,variance_after,predicted\n%.4f,%.4f,%.4f,%.4f\n",
            (unsigned long long)n_gen, v, var0, var1, pred);

    // The 30 positions are strongly correlated, so each phase set contributes only a few independent
    // samples: the Monte-Carlo error of the average variance is about sqrt(2/(3·4000)) = 0.013; limit 5 sigma
    CHECK(worst_mean < 1e-13);
    CHECK(std::abs(var0 - 1.0) < 0.065);
    CHECK(std::abs(var1 - pred) < 0.065);
    CHECK(pred < 0.8); // The case is one where the correction matters
}

TEST_CASE("sos_remove_bias - a second call changes nothing, outputs are optional, float works")
{
    arma::mat freq = rnd(48, 3, -0.05, 0.05, 19), phase0 = rnd(48, 3, -pi, pi, 20), pos = positions(40, 60.0, 21);

    arma::mat once = phase0, twice, offset(7, 7), before;
    quadriga_lib::sos_remove_bias(freq, once, pos);
    twice = once;
    quadriga_lib::sos_remove_bias(freq, twice, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset, &before);
    REQUIRE(offset.n_rows == 3);
    REQUIRE(offset.n_cols == 1);
    CHECK(arma::abs(once - phase0).max() > 1e-3);
    CHECK(arma::abs(offset).max() < 1e-12);
    CHECK(arma::abs(before).max() < 1e-13);
    CHECK(arma::abs(twice - once).max() < 1e-12);

    // Positions with two columns: z = 0
    arma::mat p2 = pos.cols(0, 1), p3 = pos, a = phase0, b = phase0;
    p3.col(2).zeros();
    quadriga_lib::sos_remove_bias(freq, a, p2);
    quadriga_lib::sos_remove_bias(freq, b, p3);
    CHECK(arma::approx_equal(a, b, "absdiff", 1e-14));

    // Float: the phases are rounded to single precision, which limits the remaining mean. A phase error of
    // 2.4e-7 rad on 48 terms of amplitude 0.2 gives at most 2.3e-6.
    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq), pf = arma::conv_to<arma::fmat>::from(phase0), posf = arma::conv_to<arma::fmat>::from(pos), of, bf;
    quadriga_lib::sos_remove_bias(ff, pf, posf, (arma::fmat *)nullptr, (arma::fmat *)nullptr, (arma::fvec *)nullptr, (arma::fvec *)nullptr, &of, &bf);
    REQUIRE(of.n_rows == 3);
    REQUIRE(bf.n_rows == 3);
    arma::mat ffd = arma::conv_to<arma::mat>::from(ff), pfd = arma::conv_to<arma::mat>::from(pf), posd = arma::conv_to<arma::mat>::from(posf);
    for (arma::uword g = 0; g < 3; ++g)
        CHECK(std::abs(mean_val(ffd, pfd, g, posd)) < 5e-6);
    CHECK(pf.min() >= -(float)pi);
    CHECK(pf.max() < (float)pi);
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(pf) - phase0).max() > 1e-3);
}

TEST_CASE("sos_remove_bias - input validation")
{
    arma::mat freq = rnd(10, 3, -0.05, 0.05, 22), phase = rnd(10, 2, -pi, pi, 23), phase_b = rnd(10, 2, -pi, pi, 24);
    arma::mat pos = positions(8, 60.0, 25), pos_b = positions(8, 60.0, 26);
    arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);
    arma::mat *no_mat = nullptr;
    arma::vec *no_vec = nullptr;

    SECTION("valid baseline")
    {
        arma::mat p = phase, pb = phase_b;
        CHECK_NOTHROW(quadriga_lib::sos_remove_bias(freq, p, pos));
        CHECK_NOTHROW(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &acf, &dist));
        CHECK_NOTHROW(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, no_mat, no_mat, 2.0));
    }
    SECTION("freq")
    {
        arma::mat empty, four(10, 4, arma::fill::ones), p = phase;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(empty, p, pos), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(four, p, pos), std::invalid_argument);
    }
    SECTION("phase")
    {
        arma::mat empty, rows(9, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, empty, pos), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, rows, pos), std::invalid_argument);
    }
    SECTION("pos and pos_b")
    {
        arma::mat empty, f2 = freq.cols(0, 1), rows = positions(7, 60.0, 27), p = phase;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, empty), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(f2, p, pos), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &rows, no_mat, &acf, &dist), std::invalid_argument); // Same map needs pairs
        // pos_b with more columns than freq, on the same map and on two maps
        arma::mat p2 = pos.cols(0, 1), pb = phase_b;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(f2, p, p2, &pos_b, no_mat, &acf, &dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(f2, p, p2, &pos_b, &pb), std::invalid_argument);
        CHECK_NOTHROW(quadriga_lib::sos_remove_bias(f2, p, p2, &p2, &pb));
    }
    SECTION("phase_b")
    {
        arma::mat cols(10, 3, arma::fill::ones), p = phase, pb = phase_b;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &cols), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, no_mat, &pb), std::invalid_argument);
    }
    SECTION("acf and dist")
    {
        arma::vec bad = dist + 1.0;
        arma::mat p = phase;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &acf, &bad), std::invalid_argument);
        // Non-finite table
        arma::vec inf_dist = dist;
        inf_dist(50) = arma::datum::inf;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &acf, &inf_dist), std::invalid_argument);
        // An ACF value of -1 or below makes the normalisation of the pair singular
        arma::vec minus_one = acf, below = acf;
        minus_one(7) = -1.0, below(7) = -1.2;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &minus_one, &dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &below, &dist), std::invalid_argument);
        CHECK(arma::approx_equal(p, phase, "absdiff", 0.0));
    }
    SECTION("phase_b is the same object as phase")
    {
        // One matrix for both devices would be corrected twice, and the second correction undoes the first
        arma::mat p = phase, pb = phase;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &p), std::invalid_argument);
        CHECK(arma::approx_equal(p, phase, "absdiff", 0.0));
        // Two matrices with equal values are two maps with the same phases and are accepted
        CHECK_NOTHROW(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb));
    }
    SECTION("freq_scale_b")
    {
        arma::mat p = phase, pb = phase_b;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, no_mat, no_mat, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, no_mat, &acf, &dist, no_mat, no_mat, 2.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, no_mat, no_mat, no_vec, no_vec, no_mat, no_mat, 2.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, no_mat, no_mat, std::nan("")), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, no_mat, no_mat, arma::datum::inf), std::invalid_argument);
    }
    SECTION("a failed call leaves the phases untouched")
    {
        arma::mat p = phase, empty;
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, empty), std::invalid_argument);
        CHECK(arma::approx_equal(p, phase, "absdiff", 0.0));
    }
}

TEST_CASE("sos_remove_bias - an argument that is written cannot share memory with another argument")
{
    // 'phase' and 'phase_b' are updated in place, 'offset' and 'mean_before' are written. A second matrix
    // object on the same memory, or on part of it, is rejected like the same object: the call throws before
    // anything is written. Memory that only touches another argument is accepted.
    arma::mat freq = rnd(10, 3, -0.05, 0.05, 22), phase = rnd(10, 2, -pi, pi, 23), phase_b = rnd(10, 2, -pi, pi, 24);
    arma::mat pos = positions(8, 60.0, 25), pos_b = positions(8, 60.0, 26);
    arma::mat *no_mat = nullptr;
    arma::vec *no_vec = nullptr;

    SECTION("phase_b on the memory of phase")
    {
        arma::mat p = phase;
        arma::mat pb(p.memptr(), p.n_rows, p.n_cols, false, true);
        std::string msg;
        try
        {
            quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb);
        }
        catch (const std::invalid_argument &e)
        {
            msg = e.what();
        }
        CHECK(msg.find("'phase_b'") != std::string::npos);
        CHECK(msg.find("'phase'") != std::string::npos);
        CHECK(arma::approx_equal(p, phase, "absdiff", 0.0));
    }
    SECTION("phase_b on part of the memory of phase")
    {
        // One block of three columns: phase on columns 0 and 1, phase_b on columns 1 and 2
        arma::mat block = arma::join_rows(phase, phase_b.col(1)), keep = block;
        arma::mat p(block.colptr(0), 10, 2, false, true), pb(block.colptr(1), 10, 2, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb), std::invalid_argument);
        CHECK(arma::approx_equal(block, keep, "absdiff", 0.0));
    }
    SECTION("phase on part of the memory of an input")
    {
        // One block: pos on elements 0 to 23, phase on 20 to 39
        arma::vec block(40);
        std::copy(pos.begin(), pos.end(), block.begin());
        std::copy(phase.begin() + 4, phase.end(), block.begin() + 24);
        arma::vec keep = block;
        arma::mat ps(block.memptr(), 8, 3, false, true), p(block.memptr() + 20, 10, 2, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, ps), std::invalid_argument);
        // freq [3, 3] on elements 22 to 30, phase [3, 2] on 20 to 25
        arma::mat f(block.memptr() + 22, 3, 3, false, true), p3(block.memptr() + 20, 3, 2, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(f, p3, pos), std::invalid_argument);
        CHECK(arma::approx_equal(block, keep, "absdiff", 0.0));
    }
    SECTION("offset and mean_before on the memory of another argument")
    {
        arma::mat p = phase, ps = pos, pb = phase_b;
        arma::mat off_p(p.memptr() + 5, 2, 1, false, true), mean_ps(ps.memptr() + 3, 2, 1, false, true);
        arma::mat off_pb(pb.memptr() + 16, 2, 2, false, true), off(2, 1), mean(2, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, no_mat, no_mat, no_vec, no_vec, &off_p), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, ps, no_mat, no_mat, no_vec, no_vec, &off, &mean_ps), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, &off_pb), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(freq, p, pos, no_mat, no_mat, no_vec, no_vec, &off, &off), std::invalid_argument);
        CHECK(arma::approx_equal(p, phase, "absdiff", 0.0));
        CHECK(arma::approx_equal(ps, pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, phase_b, "absdiff", 0.0));
    }
    SECTION("memory next to another argument is accepted")
    {
        arma::mat p_ref = phase, pb_ref = phase_b, off_ref, mean_ref;
        quadriga_lib::sos_remove_bias(freq, p_ref, pos, &pos_b, &pb_ref, no_vec, no_vec, &off_ref, &mean_ref);

        // One block: phase on elements 0 to 19, phase_b on 20 to 39, offset on 40 to 43, mean_before on 44 to 47
        arma::vec block(48);
        std::copy(phase.begin(), phase.end(), block.begin());
        std::copy(phase_b.begin(), phase_b.end(), block.begin() + 20);
        arma::mat p(block.memptr(), 10, 2, false, true), pb(block.memptr() + 20, 10, 2, false, true);
        arma::mat off(block.memptr() + 40, 2, 2, false, true), mean(block.memptr() + 44, 2, 2, false, true);
        REQUIRE_NOTHROW(quadriga_lib::sos_remove_bias(freq, p, pos, &pos_b, &pb, no_vec, no_vec, &off, &mean));
        CHECK(arma::approx_equal(p, p_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, pb_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(off, off_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(mean, mean_ref, "absdiff", 0.0));
    }
}

TEST_CASE("sos_remove_bias - a generator whose mean is zero for every offset is left unchanged")
{
    // One sinusoid with a wavelength of 4 m, positions at 0 and 2 m: the two values are s and -s for every
    // phase, z = ( 1 + exp( j·pi ) ) / 2 is zero up to the rounding of pi, and abs(Z) = 6e-17 is below the
    // threshold of 1e-14 · n_coeff. The phases, also one outside [-pi, pi), must come back bit-identical.
    arma::mat freq = {{0.25, 0.0, 0.0}}, pos = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    arma::mat phase = {{5.0, -0.3, 0.0}}, given = phase, offset(1, 1), before;
    offset.fill(9.0);

    quadriga_lib::sos_remove_bias(freq, phase, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset, &before);
    REQUIRE(offset.n_rows == 3);
    REQUIRE(offset.n_cols == 1);
    REQUIRE(before.n_rows == 3);
    QD_DIAG("\n# diag: sos_remove_bias below the threshold: offsets %g %g %g, means %.3e %.3e %.3e\n", offset(0), offset(1), offset(2), before(0), before(1), before(2));
    for (arma::uword g = 0; g < 3; ++g)
    {
        INFO("generator " << g);
        CHECK(offset(g, 0) == 0.0);
        CHECK(phase(0, g) == given(0, g));
        CHECK(std::abs(before(g, 0)) < 1e-15);
    }

    // One position more and the mean is no longer zero for every offset: the generator is corrected
    arma::mat pos3 = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.7, 0.0, 0.0}};
    phase = given;
    quadriga_lib::sos_remove_bias(freq, phase, pos3, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &offset);
    CHECK(std::abs(offset(0, 0)) > 1e-3);
    CHECK(phase(0, 0) != given(0, 0));
    CHECK(std::abs(mean_val(freq, phase, 0, pos3)) < 1e-14);
}

TEST_CASE("sos_remove_bias - phases outside the interval come back inside it")
{
    // The same generators with 20 rad added to every phase. A common phase offset is what the function
    // solves for, and its zeros are pi apart: the total offset 20 + offset_shifted equals the offset of
    // the unshifted input modulo pi, and the corrected phases of the two calls agree modulo pi. The
    // corrected phases are wrapped into [-pi, pi).
    arma::mat freq = rnd(48, 3, -0.05, 0.05, 30), phase0 = rnd(48, 4, -pi, pi, 31), pos = positions(40, 60.0, 32);
    arma::mat inside = phase0, shifted = phase0 + 20.0, off_in, off_sh;
    quadriga_lib::sos_remove_bias(freq, inside, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &off_in);
    quadriga_lib::sos_remove_bias(freq, shifted, pos, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, &off_sh);
    REQUIRE(off_sh.n_rows == 4);
    QD_DIAG("\n# diag: sos_remove_bias input shifted by 20 rad: output %.4f to %.4f\n", shifted.min(), shifted.max());
    CHECK(shifted.min() >= -pi);
    CHECK(shifted.max() < pi);
    CHECK(shifted.max() - shifted.min() > 5.0);
    for (arma::uword g = 0; g < 4; ++g)
    {
        INFO("generator " << g);
        CHECK(std::abs(mean_val(freq, shifted, g, pos)) < 1e-13);
        CHECK(std::abs(off_sh(g, 0)) <= 0.5 * pi + 1e-12);
        CHECK(diff_mod_pi(20.0 + off_sh(g, 0), off_in(g, 0)) < 1e-12);
        for (arma::uword n = 0; n < 48; ++n)
            CHECK(diff_mod_pi(shifted(n, g), inside(n, g)) < 1e-12);
    }
}

TEST_CASE("sos_remove_bias - devices with different numbers of position columns equal zero-padded positions")
{
    arma::mat freq = rnd(30, 3, -0.05, 0.05, 33), phase0 = rnd(30, 3, -pi, pi, 34), phase_b0 = rnd(30, 3, -pi, pi, 35);
    arma::mat pa3 = rnd(25, 3, -30.0, 30.0, 36), pb3 = rnd(25, 3, -30.0, 30.0, 37);
    arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);

    for (int mode = 1; mode <= 2; ++mode)
        for (arma::uword ca = 1; ca <= 3; ++ca)
            for (arma::uword cb = 1; cb <= 3; ++cb)
            {
                arma::mat pa_pad = pa3, pb_pad = pb3;
                for (arma::uword c = ca; c < 3; ++c)
                    pa_pad.col(c).zeros();
                for (arma::uword c = cb; c < 3; ++c)
                    pb_pad.col(c).zeros();
                arma::mat pa = pa3.cols(0, ca - 1), pb = pb3.cols(0, cb - 1);
                arma::mat ph = phase0, phb = phase_b0, ph_ref = phase0, phb_ref = phase_b0, off, off_ref, mean, mean_ref;
                quadriga_lib::sos_remove_bias(freq, ph, pa, &pb, mode == 2 ? &phb : (arma::mat *)nullptr, &acf, &dist, &off, &mean, mode == 2 ? 0.7 : 1.0);
                quadriga_lib::sos_remove_bias(freq, ph_ref, pa_pad, &pb_pad, mode == 2 ? &phb_ref : (arma::mat *)nullptr, &acf, &dist, &off_ref, &mean_ref, mode == 2 ? 0.7 : 1.0);
                INFO("mode " << mode << ", columns " << ca << " and " << cb);
                REQUIRE(off.n_cols == (arma::uword)mode);
                CHECK(arma::abs(ph - phase0).max() > 1e-3);
                CHECK(arma::approx_equal(ph, ph_ref, "absdiff", 0.0));
                CHECK(arma::approx_equal(phb, phb_ref, "absdiff", 0.0));
                CHECK(arma::approx_equal(off, off_ref, "absdiff", 0.0));
                CHECK(arma::approx_equal(mean, mean_ref, "absdiff", 0.0));
            }
}

TEST_CASE("sos_remove_bias - results do not depend on the number of threads")
{
    // 1000 positions and 100 sinusoids: the sums over the positions run in a parallel loop over the sinusoids
    arma::mat freq = rnd(100, 3, -0.05, 0.05, 38), phase0 = rnd(100, 7, -pi, pi, 39), phase_b0 = rnd(100, 7, -pi, pi, 40);
    arma::mat pos = positions(1000, 300.0, 41), pos_b = positions(1000, 300.0, 42);
    arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);

    for (int mode = 0; mode <= 2; ++mode)
    {
        arma::mat ph[2], phb[2], off[2], mean[2];
        for (int run = 0; run < 2; ++run)
        {
            ph[run] = phase0, phb[run] = phase_b0;
#ifdef _OPENMP
            int n_threads = omp_get_max_threads();
            if (run == 1)
                omp_set_num_threads(1);
#endif
            quadriga_lib::sos_remove_bias(freq, ph[run], pos, mode == 0 ? (arma::mat *)nullptr : &pos_b, mode == 2 ? &phb[run] : (arma::mat *)nullptr,
                                          &acf, &dist, &off[run], &mean[run], mode == 2 ? 0.7 : 1.0);
#ifdef _OPENMP
            omp_set_num_threads(n_threads);
#endif
        }
        INFO("mode " << mode);
        CHECK(arma::abs(ph[0] - phase0).max() > 1e-3);
        CHECK(arma::approx_equal(ph[0], ph[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(phb[0], phb[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(off[0], off[1], "absdiff", 0.0));
        CHECK(arma::approx_equal(mean[0], mean[1], "absdiff", 0.0));
    }
}
