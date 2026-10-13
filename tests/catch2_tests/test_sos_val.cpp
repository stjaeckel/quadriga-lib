// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::sos_val.
//
// Pins down: the value of the sum of sinusoids, the batching over phase columns, the two kernels and
// their agreement far from the origin, the independence of a value from the rest of the call, the
// three device modes, the Uniform mapping, sizes, float and double, input validation.
// It leaves to other suites: the frequency tables and how well they reproduce a target ACF
// (test_sos_preset), the phase stream (test_sos_init_phases).
//
// Oracles, all written in scalar double math in this file:
//  - ref_s: direct transcription of s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n )
//  - hand cases with one sinusoid, where the value is a known multiple of sqrt(2)
//  - expectation over uniformly random phases: mean 0, variance 1, covariance between two positions
//    (1/N) · sum_n cos( 2·pi · f_n·dx ), all exact for any frequency table
//
// Conventions: frequencies in 1/m as rows {fx, fy, fz}; positions in m as rows {x, y, z}; test
// frequencies are random with components up to about 0.3 / m, i.e. wavelengths of a few metres.
// Tolerances: GENERIC is double throughout, 1e-12. AVX2 evaluates sine and cosine in single precision:
// each of the N terms has an absolute error below 4e-7 · sqrt(2/N) (about 3 float spacings of a value
// near 1, times the amplitude), so the sum is within 4e-7 · sqrt(2·N) in the worst case.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

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

    // Oracle: s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n ), frequencies scaled by 'scale'
    double ref_s(const arma::mat &freq, const arma::mat &phase, arma::uword g, const arma::mat &pos, arma::uword p, double scale = 1.0)
    {
        double acc = 0.0;
        for (arma::uword n = 0; n < freq.n_rows; ++n)
        {
            double arg = 0.0;
            for (arma::uword c = 0; c < pos.n_cols; ++c)
                arg += scale * freq(n, c) * pos(p, c);
            acc += std::cos(2.0 * pi * arg + phase(n, g));
        }
        return std::sqrt(2.0 / (double)freq.n_rows) * acc;
    }

    // Oracle: linear interpolation of the ACF table, last value beyond the last sample point
    double ref_acf(const arma::vec &acf, const arma::vec &dist, double d)
    {
        if (d >= dist(dist.n_elem - 1))
            return acf(acf.n_elem - 1);
        arma::uword i = 0;
        while (dist(i + 1) <= d)
            ++i;
        return acf(i) + (d - dist(i)) / (dist(i + 1) - dist(i)) * (acf(i + 1) - acf(i));
    }

    // Worst-case deviation of the AVX2 kernel from double precision, see the header comment
    // Mean of the element-wise product of two rows (plain loop, the tests are linked without BLAS)
    double mean_prod(const arma::mat &m, arma::uword r1, arma::uword r2)
    {
        double acc = 0.0;
        for (arma::uword c = 0; c < m.n_cols; ++c)
            acc += m(r1, c) * m(r2, c);
        return acc / (double)m.n_cols;
    }

    double tol_avx2(arma::uword n_coeff) { return 4e-7 * std::sqrt(2.0 * (double)n_coeff); }

    bool has_avx2() { return quadriga_lib::quadriga_lib_has_AVX2(); }
}

TEST_CASE("sos_val - a single sinusoid on the axis gives the hand-computed values")
{
    // f = 0.25 / m along x, N = 1: s(x) = sqrt(2) · cos( pi/2 · x + phi )
    arma::mat freq = {{0.25, 0.0, 0.0}};
    arma::mat phase = {{0.0, pi / 2.0}};
    arma::mat pos = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {4.0, 5.0, -7.0}};
    const double r2 = std::sqrt(2.0);
    arma::mat ref = {{r2, 0.0}, {0.0, -r2}, {-r2, 0.0}, {0.0, r2}, {r2, 0.0}};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val;
        quadriga_lib::sos_val(freq, phase, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(val.n_rows == 5);
        REQUIRE(val.n_cols == 2);

        QD_DIAG("\n# diag: sos_val single sinusoid, kernel %d\n", kernel);
        QD_DIAG("x,val_phi0,ref_phi0,val_phi90,ref_phi90\n");
        for (arma::uword p = 0; p < 5; ++p)
            QD_DIAG("%.0f,%.9f,%.9f,%.9f,%.9f\n", pos(p, 0), val(p, 0), ref(p, 0), val(p, 1), ref(p, 1));

        INFO("kernel " << kernel);
        CHECK(arma::approx_equal(val, ref, "absdiff", kernel == 1 ? 1e-14 : tol_avx2(1)));
    }
}

TEST_CASE("sos_val - values match the brute-force sum for several generators")
{
    // 11 positions cross one AVX2 block of 8, 6 generators cross one group of 4
    arma::mat freq = rnd(7, 3, -0.3, 0.3, 1), phase = rnd(7, 6, -pi, pi, 2), pos = rnd(11, 3, -50.0, 50.0, 3);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val;
        quadriga_lib::sos_val(freq, phase, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(val.n_rows == 11);
        REQUIRE(val.n_cols == 6);

        double max_err = 0.0;
        QD_DIAG("\n# diag: sos_val brute force, kernel %d\n", kernel);
        QD_DIAG("pos,gen,val,ref,diff\n");
        for (arma::uword g = 0; g < 6; ++g)
            for (arma::uword p = 0; p < 11; ++p)
            {
                double ref = ref_s(freq, phase, g, pos, p);
                if (p < 2)
                    QD_DIAG("%llu,%llu,%.12f,%.12f,%.3e\n", (unsigned long long)p, (unsigned long long)g, val(p, g), ref, val(p, g) - ref);
                max_err = std::max(max_err, std::abs(val(p, g) - ref));
            }
        QD_DIAG("max_abs_err,%.3e\n", max_err);
        INFO("kernel " << kernel);
        CHECK(max_err < (kernel == 1 ? 1e-12 : tol_avx2(7)));
        CHECK(arma::abs(val).max() > 0.5); // Guards against an all-zero result passing a loose check
    }
}

TEST_CASE("sos_val - a shift by one wavelength leaves a single sinusoid unchanged")
{
    arma::mat freq = {{0.11, -0.07, 0.05}}, phase(1, 1);
    phase(0, 0) = 0.7;
    double f2 = freq(0, 0) * freq(0, 0) + freq(0, 1) * freq(0, 1) + freq(0, 2) * freq(0, 2);
    arma::mat pos(4, 3);
    pos.row(0) = arma::rowvec({3.0, -2.0, 1.5});
    pos.row(1) = pos.row(0) + freq / f2;        // One wavelength along f
    pos.row(2) = pos.row(0) - 3.0 * freq / f2;  // Three wavelengths back
    pos.row(3) = pos.row(0) + 0.5 * freq / f2;  // Half a wavelength: sign flip

    arma::mat val;
    quadriga_lib::sos_val(freq, phase, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr,
                          (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    REQUIRE(val.n_rows == 4);
    CHECK(std::abs(val(0, 0)) > 0.1);
    CHECK(std::abs(val(1, 0) - val(0, 0)) < 1e-12);
    CHECK(std::abs(val(2, 0) - val(0, 0)) < 1e-12);
    CHECK(std::abs(val(3, 0) + val(0, 0)) < 1e-12);
}

TEST_CASE("sos_val - a column of a batched call equals the call with that phase column alone")
{
    arma::mat freq = rnd(40, 3, -0.3, 0.3, 4), phase = rnd(40, 9, -pi, pi, 5), pos = rnd(19, 3, -80.0, 80.0, 6);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat all;
        quadriga_lib::sos_val(freq, phase, pos, all, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(all.n_cols == 9);
        CHECK(arma::abs(all).max() > 0.5);
        for (arma::uword g = 0; g < 9; ++g)
        {
            arma::mat one, ph = phase.col(g);
            quadriga_lib::sos_val(freq, ph, pos, one, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                  (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
            INFO("kernel " << kernel << ", generator " << g);
            CHECK(arma::approx_equal(one, all.col(g), "absdiff", 0.0)); // Bit-identical
        }
    }
}

TEST_CASE("sos_val - the value at a position does not depend on the other positions of the call")
{
    // One probe position is moved through every slot of position sets with 1 to 20 entries. This covers
    // all 8 lanes of the AVX2 kernel, full and incomplete blocks. All results must be bit-identical.
    arma::mat freq = rnd(33, 3, -0.3, 0.3, 7), phase = rnd(33, 5, -pi, pi, 8);
    arma::mat probe = {{12.5, -33.25, 1.5}};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat ref;
        quadriga_lib::sos_val(freq, phase, probe, ref, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(ref.n_cols == 5);
        CHECK(arma::abs(ref).max() > 0.1);

        arma::uword n_checked = 0, n_equal = 0;
        for (arma::uword n_pos = 1; n_pos <= 20; ++n_pos)
            for (arma::uword slot = 0; slot < n_pos; ++slot)
            {
                arma::mat pos = rnd(n_pos, 3, -200.0, 200.0, 100 + (unsigned)(n_pos * 31 + slot)), val;
                pos.row(slot) = probe;
                quadriga_lib::sos_val(freq, phase, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                      (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
                ++n_checked;
                if (val.n_rows == n_pos && arma::approx_equal(val.row(slot), ref, "absdiff", 0.0))
                    ++n_equal;
            }
        QD_DIAG("\n# diag: sos_val probe position, kernel %d: %llu of %llu placements bit-identical\n", kernel,
                (unsigned long long)n_equal, (unsigned long long)n_checked);
        INFO("kernel " << kernel);
        CHECK(n_equal == n_checked);
    }
}

TEST_CASE("sos_val - the AVX2 kernel agrees with GENERIC at any distance from the origin")
{
    if (!has_avx2())
        return;

    // Frequencies up to 3 / m correspond to a decorrelation distance of about 1 m. At an offset of 1e6 m
    // the argument reaches 2e7 rad. The error of the AVX2 kernel must stay at the level it has at the origin.
    const arma::uword n_coeff = 300;
    arma::mat freq = rnd(n_coeff, 3, -3.0, 3.0, 9), phase = rnd(n_coeff, 4, -pi, pi, 10);
    arma::vec offset = {0.0, 1e2, 1e3, 1e4, 1e5, 1e6};

    QD_DIAG("\n# diag: sos_val AVX2 against GENERIC, N = %llu, tolerance %.2e\n", (unsigned long long)n_coeff, tol_avx2(n_coeff));
    QD_DIAG("offset_m,max_abs_diff,rms_diff\n");
    for (arma::uword i = 0; i < offset.n_elem; ++i)
    {
        arma::mat pos = rnd(64, 3, -5.0, 5.0, 11 + (unsigned)i) + offset(i), a, b;
        quadriga_lib::sos_val(freq, phase, pos, a, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
        quadriga_lib::sos_val(freq, phase, pos, b, (arma::mat *)nullptr, (arma::mat *)nullptr,
                              (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 2);
        REQUIRE(a.n_elem == 256);
        REQUIRE(b.n_elem == 256);
        double mx = arma::abs(a - b).max(), rms = 0.0;
        for (arma::uword k = 0; k < 256; ++k)
            rms += (a(k) - b(k)) * (a(k) - b(k)) / 256.0;
        rms = std::sqrt(rms);
        QD_DIAG("%.0e,%.3e,%.3e\n", offset(i), mx, rms);
        INFO("offset " << offset(i));
        CHECK(mx < tol_avx2(n_coeff));
        CHECK(arma::abs(a).max() > 0.5);

        // GENERIC itself against the brute-force oracle at this offset; the oracle loses accuracy with the
        // size of the argument (2e7 rad carry an absolute error of about 4e-9), hence the scaled tolerance
        double ref = ref_s(freq, phase, 2, pos, 5);
        CHECK(std::abs(a(5, 2) - ref) < 1e-12 + 1e-13 * offset(i));
    }
}

TEST_CASE("sos_val - missing position columns count as zero")
{
    arma::mat freq = rnd(12, 3, -0.3, 0.3, 12), phase = rnd(12, 2, -pi, pi, 13);
    arma::mat pos3 = rnd(9, 3, -40.0, 40.0, 14);
    pos3.col(2).zeros();
    arma::mat pos2 = pos3.cols(0, 1);
    arma::mat pos1 = pos3.col(0), pos1z = pos3;
    pos1z.col(1).zeros();

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat v3, v2, v1, v1z;
        quadriga_lib::sos_val(freq, phase, pos3, v3, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pos2, v2, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pos1, v1, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pos1z, v1z, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        INFO("kernel " << kernel);
        REQUIRE(v2.n_rows == 9);
        CHECK(arma::abs(v3).max() > 0.5);
        CHECK(arma::approx_equal(v2, v3, "absdiff", 0.0));
        CHECK(arma::approx_equal(v1, v1z, "absdiff", 0.0));
        CHECK(arma::abs(v1 - v3).max() > 0.01); // The y coordinate matters when it is given
    }

    // Frequencies with one and two columns
    arma::mat f1 = freq.col(0), f2 = freq.cols(0, 1), val;
    quadriga_lib::sos_val(f1, phase, pos1, val, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    CHECK(std::abs(val(3, 1) - ref_s(f1, phase, 1, pos1, 3)) < 1e-12);
    quadriga_lib::sos_val(f2, phase, pos2, val, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    CHECK(std::abs(val(3, 1) - ref_s(f2, phase, 1, pos2, 3)) < 1e-12);
}

TEST_CASE("sos_val - over random phases the values have zero mean, unit variance and the ACF of the sinusoids")
{
    // 20000 generators at 3 positions. Expectations are exact for any frequency table:
    // mean 0, variance 1, covariance (1/N) · sum_n cos( 2·pi · f_n·dx ).
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat freq = rnd(n_coeff, 3, -0.1, 0.1, 15), phase = rnd(n_coeff, n_gen, -pi, pi, 16);
    arma::mat pos = {{10.0, 20.0, 1.5}, {12.0, 21.0, 1.5}, {40.0, -15.0, 3.0}};

    arma::mat val;
    quadriga_lib::sos_val(freq, phase, pos, val);
    REQUIRE(val.n_rows == 3);
    REQUIRE(val.n_cols == n_gen);

    // Monte-Carlo errors for n_gen samples of a near-Normal variable: mean 1/sqrt(n) = 0.0071,
    // variance sqrt(2/n) = 0.01, covariance at most sqrt(2/n) = 0.01. Limits are 5 sigma.
    QD_DIAG("\n# diag: sos_val statistics over %llu phase sets\n", (unsigned long long)n_gen);
    QD_DIAG("pos,mean,variance\n");
    for (arma::uword p = 0; p < 3; ++p)
    {
        double mean = arma::mean(val.row(p)), var = mean_prod(val, p, p);
        QD_DIAG("%llu,%.5f,%.5f\n", (unsigned long long)p, mean, var);
        INFO("position " << p);
        CHECK(std::abs(mean) < 5.0 * 0.0071);
        CHECK(std::abs(var - 1.0) < 5.0 * 0.01);
    }
    QD_DIAG("pair,covariance,expected,diff\n");
    for (arma::uword q = 1; q < 3; ++q)
    {
        double cov = mean_prod(val, 0, q), ref = 0.0;
        for (arma::uword n = 0; n < n_coeff; ++n)
        {
            double fd = 0.0;
            for (arma::uword c = 0; c < 3; ++c)
                fd += freq(n, c) * (pos(q, c) - pos(0, c));
            ref += std::cos(2.0 * pi * fd) / (double)n_coeff;
        }
        QD_DIAG("0-%llu,%.5f,%.5f,%.5f\n", (unsigned long long)q, cov, ref, cov - ref);
        INFO("pair 0-" << q);
        CHECK(std::abs(cov - ref) < 5.0 * 0.01);
    }
}

TEST_CASE("sos_val - two devices on the same map follow the normalised sum and are symmetric")
{
    arma::mat freq = rnd(20, 3, -0.2, 0.2, 17), phase = rnd(20, 3, -pi, pi, 18);
    arma::mat pa = rnd(10, 3, -30.0, 30.0, 19), pb = rnd(10, 3, -30.0, 30.0, 20);
    pb.row(0) = pa.row(0);                               // Devices at the same place
    pb.row(1) = pa.row(1) + arma::rowvec({0.3, 0.0, 0.0}); // Close together
    pb.row(2) = pa.row(2) + arma::rowvec({500.0, 0.0, 0.0}); // Beyond the ACF table
    arma::vec acf = {1.0, 0.6, 0.3, 0.1}, dist = {0.0, 2.0, 5.0, 20.0};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val, swp, single;
        quadriga_lib::sos_val(freq, phase, pa, val, &pb, (arma::mat *)nullptr, &acf, &dist, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pb, swp, &pa, (arma::mat *)nullptr, &acf, &dist, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pa, single, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(val.n_rows == 10);
        REQUIRE(val.n_cols == 3);

        double tol = kernel == 1 ? 1e-12 : tol_avx2(20);
        double max_err = 0.0;
        QD_DIAG("\n# diag: sos_val two devices, same map, kernel %d\n", kernel);
        QD_DIAG("pos,D,A,val,ref,diff\n");
        for (arma::uword p = 0; p < 10; ++p)
            for (arma::uword g = 0; g < 3; ++g)
            {
                double D = arma::norm(pa.row(p) - pb.row(p)), A = ref_acf(acf, dist, D);
                double ref = (ref_s(freq, phase, g, pa, p) + ref_s(freq, phase, g, pb, p)) / std::sqrt(2.0 + 2.0 * A);
                if (g == 0 && p < 4)
                    QD_DIAG("%llu,%.3f,%.4f,%.9f,%.9f,%.3e\n", (unsigned long long)p, D, A, val(p, g), ref, val(p, g) - ref);
                max_err = std::max(max_err, std::abs(val(p, g) - ref));
            }
        INFO("kernel " << kernel);
        CHECK(max_err < tol);
        CHECK(arma::abs(val).max() > 0.3);

        // Swapping the devices gives the same value: the sum is commutative and D is unchanged
        CHECK(arma::approx_equal(val, swp, "absdiff", 1e-14));

        // Devices at the same place: ( s + s ) / sqrt( 2 + 2·1 ) = s
        CHECK(arma::approx_equal(val.row(0), single.row(0), "absdiff", 1e-14));
    }
}

TEST_CASE("sos_val - two devices on the same map keep unit variance for small and large distances")
{
    // The variance of s(a) + s(b) over the phases is 2 + 2·R(a - b) with R the ACF of the sinusoids.
    // The table handed to the function is that ACF sampled along the x-axis, and the devices are
    // displaced along x, so the normalised value must have unit variance for every displacement.
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat freq = rnd(n_coeff, 3, -0.05, 0.05, 21), phase = rnd(n_coeff, n_gen, -pi, pi, 22);
    arma::vec dist = arma::regspace(0.0, 0.5, 40.0), acf(dist.n_elem);
    for (arma::uword i = 0; i < dist.n_elem; ++i)
    {
        double r = 0.0;
        for (arma::uword n = 0; n < n_coeff; ++n)
            r += std::cos(2.0 * pi * freq(n, 0) * dist(i)) / (double)n_coeff;
        acf(i) = r;
    }

    arma::vec dx = {0.0, 0.5, 3.0, 10.0, 25.0, 40.0};
    arma::mat pa(dx.n_elem, 3), pb;
    for (arma::uword p = 0; p < dx.n_elem; ++p)
        pa.row(p) = arma::rowvec({5.0 + 7.0 * (double)p, -3.0, 2.0});
    pb = pa;
    pb.col(0) += dx;

    arma::mat val;
    quadriga_lib::sos_val(freq, phase, pa, val, &pb, (arma::mat *)nullptr, &acf, &dist);
    REQUIRE(val.n_rows == dx.n_elem);
    REQUIRE(val.n_cols == n_gen);

    // Standard deviation of the sample variance: sqrt(2/n) = 0.01, limit 5 sigma
    QD_DIAG("\n# diag: sos_val same map, variance over %llu phase sets\n", (unsigned long long)n_gen);
    QD_DIAG("dx,acf,variance\n");
    for (arma::uword p = 0; p < dx.n_elem; ++p)
    {
        double var = mean_prod(val, p, p);
        QD_DIAG("%.1f,%.4f,%.5f\n", dx(p), acf(arma::uword(dx(p) / 0.5)), var);
        INFO("displacement " << dx(p));
        CHECK(std::abs(var - 1.0) < 5.0 * 0.01);
    }
}

TEST_CASE("sos_val - two devices on different maps use the second phases, the frequency scale and 1/sqrt(2)")
{
    arma::mat freq = rnd(20, 3, -0.2, 0.2, 23), phase = rnd(20, 3, -pi, pi, 24), phase_b = rnd(20, 3, -pi, pi, 25);
    arma::mat pa = rnd(10, 3, -30.0, 30.0, 26), pb = rnd(10, 3, -30.0, 30.0, 27);
    arma::vec acf = {1.0, 0.6, 0.3, 0.1}, dist = {0.0, 2.0, 5.0, 20.0};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
        for (double scale : {1.0, 2.5})
        {
            arma::mat val, with_acf;
            quadriga_lib::sos_val(freq, phase, pa, val, &pb, &phase_b, (arma::vec *)nullptr, (arma::vec *)nullptr, scale, false, kernel);
            quadriga_lib::sos_val(freq, phase, pa, with_acf, &pb, &phase_b, &acf, &dist, scale, false, kernel);
            REQUIRE(val.n_rows == 10);
            REQUIRE(val.n_cols == 3);

            double max_err = 0.0;
            for (arma::uword p = 0; p < 10; ++p)
                for (arma::uword g = 0; g < 3; ++g)
                {
                    double ref = (ref_s(freq, phase, g, pa, p) + ref_s(freq, phase_b, g, pb, p, scale)) / std::sqrt(2.0);
                    max_err = std::max(max_err, std::abs(val(p, g) - ref));
                }
            QD_DIAG("\n# diag: sos_val two maps, kernel %d, scale %.1f: max_abs_err %.3e\n", kernel, scale, max_err);
            INFO("kernel " << kernel << ", scale " << scale);
            CHECK(max_err < (kernel == 1 ? 1e-12 : tol_avx2(20)));
            CHECK(arma::abs(val).max() > 0.3);
            CHECK(arma::approx_equal(val, with_acf, "absdiff", 0.0)); // The ACF table is not used
        }

    // The scale changes the result, and the second device does not use the first phases
    arma::mat v1, v2, v3;
    quadriga_lib::sos_val(freq, phase, pa, v1, &pb, &phase_b, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    quadriga_lib::sos_val(freq, phase, pa, v2, &pb, &phase_b, (arma::vec *)nullptr, (arma::vec *)nullptr, 2.5, false, 1);
    quadriga_lib::sos_val(freq, phase, pa, v3, &pb, &phase, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    CHECK(arma::abs(v1 - v2).max() > 0.05);
    CHECK(arma::abs(v1 - v3).max() > 0.05);
}

TEST_CASE("sos_val - the Uniform output is the Normal output mapped through the Normal CDF")
{
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat freq = rnd(n_coeff, 3, -0.1, 0.1, 28), phase = rnd(n_coeff, n_gen, -pi, pi, 29);
    arma::mat pos = {{7.0, -4.0, 1.5}};

    arma::mat nrm, uni;
    quadriga_lib::sos_val(freq, phase, pos, nrm, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    quadriga_lib::sos_val(freq, phase, pos, uni, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, true, 1);
    REQUIRE(uni.n_cols == n_gen);

    double max_err = 0.0;
    for (arma::uword g = 0; g < n_gen; ++g)
        max_err = std::max(max_err, std::abs(uni(0, g) - 0.5 * std::erfc(-nrm(0, g) / std::sqrt(2.0))));
    CHECK(max_err < 1e-14);
    CHECK(uni.min() > 0.0);
    CHECK(uni.max() < 1.0);
    CHECK(uni.max() - uni.min() > 0.9);

    // 10 bins over the phase sets. A sum of 64 sinusoids is close to Normal; the expected share is 0.1 with
    // a Monte-Carlo standard deviation of sqrt(0.1·0.9/n) = 0.0021. The limit of 0.015 is 5 sigma plus an
    // allowance of 0.004 for the difference between the sum of 64 sinusoids and a Normal variable.
    arma::vec hist(10, arma::fill::zeros);
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        arma::uword b = (arma::uword)std::floor(uni(0, g) * 10.0);
        hist(b < 10 ? b : 9) += 1.0 / (double)n_gen;
    }
    QD_DIAG("\n# diag: sos_val Uniform histogram over %llu phase sets\nbin,share\n", (unsigned long long)n_gen);
    for (arma::uword b = 0; b < 10; ++b)
    {
        QD_DIAG("%llu,%.4f\n", (unsigned long long)b, hist(b));
        INFO("bin " << b);
        CHECK(std::abs(hist(b) - 0.1) < 0.015);
    }

    // The mapping is applied after the combination of two devices
    arma::mat pb = {{9.0, -4.0, 1.5}}, ph2 = phase.cols(0, 4), ph2b = phase.cols(5, 9), n2, u2;
    quadriga_lib::sos_val(freq, ph2, pos, n2, &pb, &ph2b, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
    quadriga_lib::sos_val(freq, ph2, pos, u2, &pb, &ph2b, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, true, 1);
    for (arma::uword g = 0; g < 5; ++g)
        CHECK(std::abs(u2(0, g) - 0.5 * std::erfc(-n2(0, g) / std::sqrt(2.0))) < 1e-14);
}

TEST_CASE("sos_val - empty optional matrices mean omitted, outputs are sized and overwritten")
{
    arma::mat freq = rnd(10, 3, -0.3, 0.3, 30), phase = rnd(10, 2, -pi, pi, 31), pos = rnd(5, 3, -20.0, 20.0, 32);
    arma::mat empty;
    arma::vec vempty;

    arma::mat ref, val;
    quadriga_lib::sos_val(freq, phase, pos, ref);
    REQUIRE(ref.n_rows == 5);
    REQUIRE(ref.n_cols == 2);
    CHECK(arma::abs(ref).max() > 0.3);

    quadriga_lib::sos_val(freq, phase, pos, val, &empty, &empty, &vempty, &vempty);
    CHECK(arma::approx_equal(val, ref, "absdiff", 0.0));

    arma::mat sized(5, 2);
    sized.fill(99.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_val(freq, phase, pos, sized);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, ref, "absdiff", 0.0));

    arma::mat wrong(2, 9);
    wrong.fill(99.0);
    quadriga_lib::sos_val(freq, phase, pos, wrong);
    REQUIRE(wrong.n_rows == 5);
    REQUIRE(wrong.n_cols == 2);
    CHECK(arma::approx_equal(wrong, ref, "absdiff", 0.0));
}

TEST_CASE("sos_val - float agrees with double within single precision")
{
    arma::mat freq = rnd(50, 3, -0.3, 0.3, 33), phase = rnd(50, 4, -pi, pi, 34);
    arma::mat pa = rnd(21, 3, -100.0, 100.0, 35), pb = rnd(21, 3, -100.0, 100.0, 36);
    arma::vec acf = {1.0, 0.6, 0.3, 0.1}, dist = {0.0, 2.0, 5.0, 20.0};
    arma::fmat ff = arma::conv_to<arma::fmat>::from(freq), pf = arma::conv_to<arma::fmat>::from(phase);
    arma::fmat paf = arma::conv_to<arma::fmat>::from(pa), pbf = arma::conv_to<arma::fmat>::from(pb);
    arma::fvec af = arma::conv_to<arma::fvec>::from(acf), df = arma::conv_to<arma::fvec>::from(dist);

    // Inputs rounded to float: the argument 2·pi·f·x of magnitude up to about 300 rad carries a relative
    // error of 1.2e-7 from each of f and x, i.e. up to 7e-5 rad per term. The sum of 50 terms with
    // amplitude 0.2 is then within about 1e-3 in the worst case; random errors give about 1e-4.
    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat vd;
        arma::fmat vf;
        quadriga_lib::sos_val(freq, phase, pa, vd, &pb, (arma::mat *)nullptr, &acf, &dist, 1.0, false, kernel);
        quadriga_lib::sos_val(ff, pf, paf, vf, &pbf, (arma::fmat *)nullptr, &af, &df, 1.0f, false, kernel);
        REQUIRE(vf.n_rows == 21);
        REQUIRE(vf.n_cols == 4);
        double mx = arma::abs(arma::conv_to<arma::mat>::from(vf) - vd).max();
        QD_DIAG("\n# diag: sos_val float against double, kernel %d: max_abs_diff %.3e\n", kernel, mx);
        INFO("kernel " << kernel);
        CHECK(mx < 1e-3);
        CHECK(arma::abs(vd).max() > 0.3);
    }
}

TEST_CASE("sos_val - results are repeatable and do not depend on the number of threads")
{
    // 1000 positions are 125 blocks of the AVX2 kernel, spread over all threads
    arma::mat freq = rnd(100, 3, -0.3, 0.3, 37), phase = rnd(100, 7, -pi, pi, 38), pos = rnd(1000, 3, -500.0, 500.0, 39);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat a, b, c;
        quadriga_lib::sos_val(freq, phase, pos, a, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, pos, b, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        quadriga_lib::sos_val(freq, phase, pos, c, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
#ifdef _OPENMP
        omp_set_num_threads(n_threads);
#endif
        INFO("kernel " << kernel);
        REQUIRE(a.n_rows == 1000);
        CHECK(arma::abs(a).max() > 0.5);
        CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
        CHECK(arma::approx_equal(a, c, "absdiff", 0.0));

        // Two devices, on the same map and on two maps, Normal and Uniform: the combination, the
        // normalisation and the mapping run in parallel loops as well
        arma::mat pos_b = rnd(1000, 3, -500.0, 500.0, 139), phase_b = rnd(100, 7, -pi, pi, 140);
        arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);
        for (int mode = 1; mode <= 2; ++mode)
            for (bool uniform : {false, true})
            {
                arma::mat all, one;
                const arma::mat *pb = (mode == 2) ? &phase_b : nullptr;
                quadriga_lib::sos_val(freq, phase, pos, all, &pos_b, pb, &acf, &dist, mode == 2 ? 1.5 : 1.0, uniform, kernel);
#ifdef _OPENMP
                omp_set_num_threads(1);
#endif
                quadriga_lib::sos_val(freq, phase, pos, one, &pos_b, pb, &acf, &dist, mode == 2 ? 1.5 : 1.0, uniform, kernel);
#ifdef _OPENMP
                omp_set_num_threads(n_threads);
#endif
                INFO("mode " << mode << ", uniform " << uniform);
                REQUIRE(all.n_rows == 1000);
                CHECK(all.max() - all.min() > 0.5);
                CHECK(arma::approx_equal(all, one, "absdiff", 0.0));
            }
    }
}

TEST_CASE("sos_val - the automatic kernel is AVX2 when available")
{
    arma::mat freq = rnd(100, 3, -0.3, 0.3, 40), phase = rnd(100, 2, -pi, pi, 41), pos = rnd(30, 3, -50.0, 50.0, 42);
    arma::mat v0, vk;
    quadriga_lib::sos_val(freq, phase, pos, v0);
    quadriga_lib::sos_val(freq, phase, pos, vk, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, has_avx2() ? 2 : 1);
    REQUIRE(v0.n_rows == 30);
    CHECK(arma::abs(v0).max() > 0.5);
    CHECK(arma::approx_equal(v0, vk, "absdiff", 0.0));
}

TEST_CASE("sos_val - input validation")
{
    arma::mat freq = rnd(10, 3, -0.3, 0.3, 43), phase = rnd(10, 2, -pi, pi, 44), phase_b = rnd(10, 2, -pi, pi, 45);
    arma::mat pos = rnd(5, 3, -20.0, 20.0, 46), pos_b = rnd(5, 3, -20.0, 20.0, 47), val;
    arma::vec acf = {1.0, 0.5, 0.2}, dist = {0.0, 1.0, 3.0};
    arma::mat *no_mat = nullptr;
    arma::vec *no_vec = nullptr;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val));
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf, &dist));
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b));
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, no_vec, no_vec, 2.0));
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 1.0, true, 1));
    }
    SECTION("freq")
    {
        arma::mat empty, four(10, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_val(empty, phase, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(four, phase, pos, val), std::invalid_argument);
    }
    SECTION("phase")
    {
        arma::mat empty, rows(9, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, empty, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, rows, pos, val), std::invalid_argument);
    }
    SECTION("pos")
    {
        arma::mat empty, f2 = freq.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, empty, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(f2, phase, pos, val), std::invalid_argument); // More columns than freq
    }
    SECTION("pos_b")
    {
        arma::mat rows(4, 3, arma::fill::ones), f2 = freq.cols(0, 1), p2 = pos.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &rows, no_mat, &acf, &dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(f2, phase, p2, val, &pos_b, no_mat, &acf, &dist), std::invalid_argument);
    }
    SECTION("phase_b")
    {
        arma::mat cols(10, 3, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &cols), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, &phase_b), std::invalid_argument); // Needs pos_b
    }
    SECTION("acf and dist")
    {
        arma::vec bad_dist = {0.0, 2.0, 1.0}, short_dist = {0.0, 1.0};
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b), std::invalid_argument); // Same map needs the table
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf, &bad_dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf, &short_dist), std::invalid_argument);
        // Non-finite table
        arma::vec inf_dist = {0.0, 1.0, arma::datum::inf}, nan_acf = {1.0, std::nan(""), 0.2};
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf, &inf_dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &nan_acf, &dist), std::invalid_argument);
    }
    SECTION("acf values of -1 or below on the same map")
    {
        // The normalisation 1 / sqrt( 2 + 2·A ) is singular at A = -1 and undefined below
        arma::vec minus_one = {1.0, -1.0, -1.0}, below = {1.0, 0.0, -1.5}, near = {1.0, -0.999, -0.999};
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &minus_one, &dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &below, &dist), std::invalid_argument);
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &near, &dist));
        CHECK(val.is_finite());
        // On two maps the table is not used and not checked
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, &minus_one, &dist));
    }
    SECTION("freq_scale_b")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, no_vec, no_vec, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, no_vec, no_vec, -1.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, no_vec, no_vec, std::nan("")), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, &phase_b, no_vec, no_vec, arma::datum::inf), std::invalid_argument);
        // A scale other than 1 on one map is rejected
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, &pos_b, no_mat, &acf, &dist, 2.0), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 2.0), std::invalid_argument);
    }
    SECTION("use_kernel")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 1.0, false, 3), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 1.0, false, -1), std::invalid_argument);
    }
    SECTION("a kernel that is not available")
    {
        // Runs in a build without AVX2 or on a CPU without it; with AVX2 the same call must work
        if (!has_avx2())
            CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 1.0, false, 2), std::invalid_argument);
        else
            CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, val, no_mat, no_mat, no_vec, no_vec, 1.0, false, 2));
    }
    SECTION("a failed call leaves the output untouched")
    {
        arma::mat keep(2, 2);
        keep.fill(7.0);
        arma::mat empty;
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, empty, keep), std::invalid_argument);
        CHECK(keep.n_rows == 2);
        CHECK(keep(1, 1) == 7.0);
    }
    SECTION("the output cannot be an input")
    {
        // 'val' is resized and written while the inputs are read. Each input in turn is passed as output;
        // the call must throw and leave that matrix as it was. A copy with the same values is accepted.
        arma::mat f = freq, ph = phase, p = pos, pb = pos_b, phb = phase_b;
        CHECK_THROWS_AS(quadriga_lib::sos_val(f, phase, pos, f), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, ph, pos, ph), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, p, p), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, pb, &pb, &phase_b), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, pos, phb, &pos_b, &phb), std::invalid_argument);
        CHECK(arma::approx_equal(f, freq, "absdiff", 0.0));
        CHECK(arma::approx_equal(ph, phase, "absdiff", 0.0));
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, pos_b, "absdiff", 0.0));
        CHECK(arma::approx_equal(phb, phase_b, "absdiff", 0.0));

        // The ACF table: a column vector is a matrix and can be passed as output
        arma::vec a = acf, d = dist;
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(freq, phase, pos, a, &pos_b, no_mat, &a, &dist), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(freq, phase, pos, d, &pos_b, no_mat, &acf, &d), std::invalid_argument);
        CHECK(arma::approx_equal(a, acf, "absdiff", 0.0));
        CHECK(arma::approx_equal(d, dist, "absdiff", 0.0));

        // Another matrix object on the memory of the positions
        arma::mat view(p.memptr(), p.n_rows, p.n_cols, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val(freq, phase, p, view), std::invalid_argument);
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));

        arma::mat out = pos;
        CHECK_NOTHROW(quadriga_lib::sos_val(freq, phase, pos, out));
        CHECK(out.n_rows == pos.n_rows);
        CHECK(out.n_cols == phase.n_cols);
    }
}

TEST_CASE("sos_val - a NaN or Inf coordinate gives NaN at that position only")
{
    // 19 positions: three AVX2 blocks, the bad positions share their blocks with good ones
    arma::mat freq = rnd(40, 3, -0.3, 0.3, 50), phase = rnd(40, 5, -pi, pi, 51), pos = rnd(19, 3, -80.0, 80.0, 52), bad = pos;
    bad(3, 1) = std::nan(""), bad(7, 0) = arma::datum::inf, bad(12, 2) = -arma::datum::inf;

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat ref, val;
        quadriga_lib::sos_val(freq, phase, pos, ref, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        quadriga_lib::sos_val(freq, phase, bad, val, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
        REQUIRE(val.n_rows == 19);
        REQUIRE(val.n_cols == 5);
        arma::uword n_nan = 0, n_same = 0;
        for (arma::uword p = 0; p < 19; ++p)
            for (arma::uword g = 0; g < 5; ++g)
            {
                if (p == 3 || p == 7 || p == 12)
                    n_nan += std::isnan(val(p, g)) ? 1 : 0;
                else
                    n_same += (val(p, g) == ref(p, g)) ? 1 : 0;
            }
        INFO("kernel " << kernel);
        CHECK(n_nan == 15);
        CHECK(n_same == 80); // Bit-identical to the call without the bad positions
        CHECK(arma::abs(ref).max() > 0.5);
    }
}

TEST_CASE("sos_val - the AVX2 kernel stays within its bound up to 1e9 periods from the origin")
{
    if (!has_avx2())
        return;

    // Components of 0.1 / m at an offset of 1e10 m: the argument f·x is about 1e9 periods per axis. A double
    // of that size resolves 1.2e-7 periods, and the two kernels round the sum of the three products
    // differently (fused multiply-add in AVX2), so each of the N terms differs by up to about 1e-6 rad.
    // With random signs the sum differs by about sqrt(2) · 5e-7; the bound of the kernel still holds.
    // Further out it does not: the documentation limits the statement to about 1e9 periods.
    const arma::uword n_coeff = 256;
    arma::mat freq = rnd(n_coeff, 3, -0.1, 0.1, 53), phase = rnd(n_coeff, 4, -pi, pi, 54);

    QD_DIAG("\n# diag: sos_val AVX2 against GENERIC far from the origin, N = %llu, bound %.2e\n", (unsigned long long)n_coeff, tol_avx2(n_coeff));
    QD_DIAG("offset_m,periods,max_abs_diff\n");
    for (double offset : {1e8, 1e9, 1e10})
    {
        arma::mat pos = rnd(64, 3, -50.0, 50.0, 55) + offset, a, b;
        quadriga_lib::sos_val(freq, phase, pos, a, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
        quadriga_lib::sos_val(freq, phase, pos, b, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 2);
        REQUIRE(a.n_elem == 256);
        double mx = arma::abs(a - b).max();
        QD_DIAG("%.0e,%.0e,%.3e\n", offset, 0.1 * offset, mx);
        INFO("offset " << offset);
        CHECK(mx < tol_avx2(n_coeff));
        CHECK(arma::abs(a).max() > 0.5);
        CHECK(a.is_finite());
    }
}

TEST_CASE("sos_val - the AVX2 kernel stays within its bound for the largest preset")
{
    if (!has_avx2())
        return;

    // 1024 sinusoids are the case with the loosest bound, 4e-7 · sqrt(2048) = 1.8e-5
    QD_DIAG("\n# diag: sos_val AVX2 against GENERIC, Comb1024, bound %.2e\ndist_decorr,max_abs_diff\n", tol_avx2(1024));
    for (double d : {10.0, 1.0})
    {
        arma::mat freq, phase = rnd(1024, 6, -pi, pi, 56), pos = rnd(200, 3, -500.0, 500.0, 57), a, b;
        quadriga_lib::sos_preset("Comb1024", freq, (arma::vec *)nullptr, (arma::vec *)nullptr, d);
        quadriga_lib::sos_val(freq, phase, pos, a, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 1);
        quadriga_lib::sos_val(freq, phase, pos, b, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, 2);
        REQUIRE(a.n_elem == 1200);
        double mx = arma::abs(a - b).max();
        QD_DIAG("%.0f,%.3e\n", d, mx);
        INFO("dist_decorr " << d);
        CHECK(mx < tol_avx2(1024));
        CHECK(arma::abs(a).max() > 0.5);
    }
}

TEST_CASE("sos_val - two devices with different numbers of position columns equal zero-padded positions")
{
    arma::mat freq = rnd(30, 3, -0.2, 0.2, 58), phase = rnd(30, 5, -pi, pi, 59), phase_b = rnd(30, 5, -pi, pi, 60);
    arma::mat pa3 = rnd(13, 3, -30.0, 30.0, 61), pb3 = rnd(13, 3, -30.0, 30.0, 62);
    arma::vec dist = arma::regspace(0.0, 1.0, 50.0), acf = arma::exp(-dist / 10.0);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
        for (int mode = 1; mode <= 2; ++mode)
            for (arma::uword ca = 1; ca <= 3; ++ca)
                for (arma::uword cb = 1; cb <= 3; ++cb)
                {
                    // Padded: all three columns, the ones not given set to 0
                    arma::mat pa_pad = pa3, pb_pad = pb3;
                    for (arma::uword c = ca; c < 3; ++c)
                        pa_pad.col(c).zeros();
                    for (arma::uword c = cb; c < 3; ++c)
                        pb_pad.col(c).zeros();
                    arma::mat pa = pa3.cols(0, ca - 1), pb = pb3.cols(0, cb - 1), val, ref;
                    const arma::mat *ph_b = (mode == 2) ? &phase_b : nullptr;
                    quadriga_lib::sos_val(freq, phase, pa, val, &pb, ph_b, &acf, &dist, mode == 2 ? 0.7 : 1.0, false, kernel);
                    quadriga_lib::sos_val(freq, phase, pa_pad, ref, &pb_pad, ph_b, &acf, &dist, mode == 2 ? 0.7 : 1.0, false, kernel);
                    INFO("kernel " << kernel << ", mode " << mode << ", columns " << ca << " and " << cb);
                    REQUIRE(val.n_rows == 13);
                    CHECK(arma::abs(ref).max() > 0.3);
                    CHECK(arma::approx_equal(val, ref, "absdiff", 0.0));
                }
}

TEST_CASE("sos_val - the float instantiation is the double computation on the float inputs")
{
    // The float call converts its inputs to double, computes as the double call does and rounds the result.
    // Compared with the double call on exactly the same (float) input values, only the rounding of the
    // output remains: half a float spacing of a value up to 4, i.e. 2.4e-7. This separates the computation
    // from the rounding of the inputs, which the case "float agrees with double" has to allow for.
    arma::fmat ff = arma::conv_to<arma::fmat>::from(rnd(50, 3, -0.3, 0.3, 63)), pf = arma::conv_to<arma::fmat>::from(rnd(50, 4, -pi, pi, 64));
    arma::fmat pbf = arma::conv_to<arma::fmat>::from(rnd(50, 4, -pi, pi, 65));
    arma::fmat paf = arma::conv_to<arma::fmat>::from(rnd(21, 3, -100.0, 100.0, 66)), pbpf = arma::conv_to<arma::fmat>::from(rnd(21, 3, -100.0, 100.0, 67));
    arma::fvec af = {1.0f, 0.6f, 0.3f, 0.1f}, df = {0.0f, 2.0f, 5.0f, 20.0f};
    arma::mat fd = arma::conv_to<arma::mat>::from(ff), pd = arma::conv_to<arma::mat>::from(pf), pbd = arma::conv_to<arma::mat>::from(pbf);
    arma::mat pad = arma::conv_to<arma::mat>::from(paf), pbpd = arma::conv_to<arma::mat>::from(pbpf);
    arma::vec ad = arma::conv_to<arma::vec>::from(af), dd = arma::conv_to<arma::vec>::from(df);
    const float scale_f = 0.7f;

    QD_DIAG("\n# diag: sos_val float against the double call on the float inputs\nkernel,mode,uniform,max_abs_diff\n");
    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
        for (int mode = 0; mode < 3; ++mode)
            for (bool uniform : {false, true})
            {
                arma::fmat vf;
                arma::mat vd;
                if (mode == 0)
                {
                    quadriga_lib::sos_val(ff, pf, paf, vf, (arma::fmat *)nullptr, (arma::fmat *)nullptr, (arma::fvec *)nullptr, (arma::fvec *)nullptr, 1.0f, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, uniform, kernel);
                }
                else if (mode == 1)
                {
                    quadriga_lib::sos_val(ff, pf, paf, vf, &pbpf, (arma::fmat *)nullptr, &af, &df, 1.0f, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, &pbpd, (arma::mat *)nullptr, &ad, &dd, 1.0, uniform, kernel);
                }
                else
                {
                    quadriga_lib::sos_val(ff, pf, paf, vf, &pbpf, &pbf, (arma::fvec *)nullptr, (arma::fvec *)nullptr, scale_f, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, &pbpd, &pbd, (arma::vec *)nullptr, (arma::vec *)nullptr, (double)scale_f, uniform, kernel);
                }
                REQUIRE(vf.n_rows == 21);
                REQUIRE(vf.n_cols == 4);
                double mx = arma::abs(arma::conv_to<arma::mat>::from(vf) - vd).max();
                QD_DIAG("%d,%d,%d,%.3e\n", kernel, mode, (int)uniform, mx);
                INFO("kernel " << kernel << ", mode " << mode << ", uniform " << uniform);
                CHECK(mx < 2e-6);
                CHECK(vd.max() - vd.min() > 0.3);
            }
}

TEST_CASE("sos_val - the value of a generator does not depend on the number of generators of the call")
{
    // The AVX2 kernel accumulates the generators in groups of 8 with one smaller group for the remaining
    // 1 to 7, the GENERIC kernel in groups of 4. 1 to 18 generators cover every group size and several
    // groups; 300 positions cross the tiles of 16 blocks in which the AVX2 kernel walks the positions and
    // end in an incomplete block. Every column must be bit-identical to its own single-generator call.
    arma::mat freq = rnd(33, 3, -0.3, 0.3, 70), phase = rnd(33, 18, -pi, pi, 71), pos = rnd(300, 3, -80.0, 80.0, 72);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat single(300, 18);
        for (arma::uword g = 0; g < 18; ++g)
        {
            arma::mat one, ph = phase.col(g);
            quadriga_lib::sos_val(freq, ph, pos, one, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
            REQUIRE(one.n_rows == 300);
            single.col(g) = one;
        }
        CHECK(arma::abs(single).max() > 0.5);

        arma::uword n_equal = 0;
        for (arma::uword n_gen = 1; n_gen <= 18; ++n_gen)
        {
            arma::mat val, ph = phase.cols(0, n_gen - 1);
            quadriga_lib::sos_val(freq, ph, pos, val, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr, 1.0, false, kernel);
            REQUIRE(val.n_cols == n_gen);
            n_equal += arma::approx_equal(val, single.cols(0, n_gen - 1), "absdiff", 0.0) ? 1 : 0;
        }
        INFO("kernel " << kernel);
        CHECK(n_equal == 18);
    }
}
