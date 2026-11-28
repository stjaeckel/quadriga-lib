// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_SOS=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_SOS=1 ./build_linux/test_bin "sos_val*"

// Verification suite for quadriga_lib::sos_val.
//
// Pins down: the value of the sum of sinusoids, the batching over phase columns, the two kernels and
// their agreement far from the origin, the independence of a value from the rest of the call, the
// two-device generator (reciprocal and not; swap symmetry, unit variance, decorrelation when one device
// moves away), the Uniform mapping, sizes, float and double, input validation.
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

#include "qd_test_diag.hpp"
#include "qd_test_sos_oracle.hpp"
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

    // Oracle: s(x) = sqrt(2/N) · sum_n cos( 2·pi · f_n·x + phi_n ), frequencies scaled by 'scale'
    double ref_s(const arma::mat &spatial_freq, const arma::mat &phase, arma::uword g, const arma::mat &pos, arma::uword p, double scale = 1.0)
    {
        double acc = 0.0;
        for (arma::uword n = 0; n < spatial_freq.n_rows; ++n)
        {
            double arg = 0.0;
            for (arma::uword c = 0; c < pos.n_cols; ++c)
                arg += scale * spatial_freq(n, c) * pos(p, c);
            acc += std::cos(2.0 * pi * arg + phase(n, g));
        }
        return std::sqrt(2.0 / (double)spatial_freq.n_rows) * acc;
    }

    // Oracle: s(a, b) of qd_test_sos_oracle.hpp at row q of pa and pb, generator g (3-column inputs)
    double ref_s2(const arma::mat &spatial_freq, const arma::mat &phase, arma::uword g, const arma::mat &pa, const arma::mat &pb,
                  arma::uword q, const std::vector<arma::uword> &p, double scale)
    {
        const double a[3] = {pa(q, 0), pa(q, 1), pa(q, 2)}, b[3] = {pb(q, 0), pb(q, 1), pb(q, 2)};
        return qd_ref_sos2(spatial_freq, phase.colptr(g), a, b, p, scale);
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
    arma::mat spatial_freq = {{0.25, 0.0, 0.0}};
    arma::mat phase = {{0.0, pi / 2.0}};
    arma::mat pos = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {4.0, 5.0, -7.0}};
    const double r2 = std::sqrt(2.0);
    arma::mat ref = {{r2, 0.0}, {0.0, -r2}, {-r2, 0.0}, {0.0, r2}, {r2, 0.0}};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val;
        quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, kernel);
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
    arma::mat spatial_freq = rnd(7, 3, -0.3, 0.3, 1), phase = rnd(7, 6, -pi, pi, 2), pos = rnd(11, 3, -50.0, 50.0, 3);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val;
        quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, kernel);
        REQUIRE(val.n_rows == 11);
        REQUIRE(val.n_cols == 6);

        double max_err = 0.0;
        QD_DIAG("\n# diag: sos_val brute force, kernel %d\n", kernel);
        QD_DIAG("pos,gen,val,ref,diff\n");
        for (arma::uword g = 0; g < 6; ++g)
            for (arma::uword p = 0; p < 11; ++p)
            {
                double ref = ref_s(spatial_freq, phase, g, pos, p);
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
    arma::mat spatial_freq = {{0.11, -0.07, 0.05}}, phase(1, 1);
    phase(0, 0) = 0.7;
    double f2 = spatial_freq(0, 0) * spatial_freq(0, 0) + spatial_freq(0, 1) * spatial_freq(0, 1) + spatial_freq(0, 2) * spatial_freq(0, 2);
    arma::mat pos(4, 3);
    pos.row(0) = arma::rowvec({3.0, -2.0, 1.5});
    pos.row(1) = pos.row(0) + spatial_freq / f2;        // One wavelength along f
    pos.row(2) = pos.row(0) - 3.0 * spatial_freq / f2;  // Three wavelengths back
    pos.row(3) = pos.row(0) + 0.5 * spatial_freq / f2;  // Half a wavelength: sign flip

    arma::mat val;
    quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, 1);
    REQUIRE(val.n_rows == 4);
    CHECK(std::abs(val(0, 0)) > 0.1);
    CHECK(std::abs(val(1, 0) - val(0, 0)) < 1e-12);
    CHECK(std::abs(val(2, 0) - val(0, 0)) < 1e-12);
    CHECK(std::abs(val(3, 0) + val(0, 0)) < 1e-12);
}

TEST_CASE("sos_val - a column of a batched call equals the call with that phase column alone")
{
    arma::mat spatial_freq = rnd(40, 3, -0.3, 0.3, 4), phase = rnd(40, 9, -pi, pi, 5), pos = rnd(19, 3, -80.0, 80.0, 6);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat all;
        quadriga_lib::sos_val(spatial_freq, phase, pos, all, {}, 1.0, true, false, kernel);
        REQUIRE(all.n_cols == 9);
        CHECK(arma::abs(all).max() > 0.5);
        for (arma::uword g = 0; g < 9; ++g)
        {
            arma::mat one, ph = phase.col(g);
            quadriga_lib::sos_val(spatial_freq, ph, pos, one, {}, 1.0, true, false, kernel);
            INFO("kernel " << kernel << ", generator " << g);
            const double tol = kernel == 1 ? 0.0 : 5e-7 * arma::abs(all).max(); // GENERIC: bit-identical
            CHECK(arma::approx_equal(one, all.col(g), "absdiff", tol));
        }
    }
}

TEST_CASE("sos_val - the value at a position does not depend on the other positions of the call")
{
    // One probe position is moved through every slot of position sets with 1 to 20 entries. This covers
    // all 8 lanes of the AVX2 kernel, full and incomplete blocks. All results must be bit-identical.
    arma::mat spatial_freq = rnd(33, 3, -0.3, 0.3, 7), phase = rnd(33, 5, -pi, pi, 8);
    arma::mat probe = {{12.5, -33.25, 1.5}};

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat ref;
        quadriga_lib::sos_val(spatial_freq, phase, probe, ref, {}, 1.0, true, false, kernel);
        REQUIRE(ref.n_cols == 5);
        CHECK(arma::abs(ref).max() > 0.1);

        arma::uword n_checked = 0, n_equal = 0;
        for (arma::uword n_pos = 1; n_pos <= 20; ++n_pos)
            for (arma::uword slot = 0; slot < n_pos; ++slot)
            {
                arma::mat pos = rnd(n_pos, 3, -200.0, 200.0, 100 + (unsigned)(n_pos * 31 + slot)), val;
                pos.row(slot) = probe;
                quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, kernel);
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
    arma::mat spatial_freq = rnd(n_coeff, 3, -3.0, 3.0, 9), phase = rnd(n_coeff, 4, -pi, pi, 10);
    arma::vec offset = {0.0, 1e2, 1e3, 1e4, 1e5, 1e6};

    QD_DIAG("\n# diag: sos_val AVX2 against GENERIC, N = %llu, tolerance %.2e\n", (unsigned long long)n_coeff, tol_avx2(n_coeff));
    QD_DIAG("offset_m,max_abs_diff,rms_diff\n");
    for (arma::uword i = 0; i < offset.n_elem; ++i)
    {
        arma::mat pos = rnd(64, 3, -5.0, 5.0, 11 + (unsigned)i) + offset(i), a, b;
        quadriga_lib::sos_val(spatial_freq, phase, pos, a, {}, 1.0, true, false, 1);
        quadriga_lib::sos_val(spatial_freq, phase, pos, b, {}, 1.0, true, false, 2);
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
        double ref = ref_s(spatial_freq, phase, 2, pos, 5);
        CHECK(std::abs(a(5, 2) - ref) < 1e-12 + 1e-13 * offset(i));
    }
}

TEST_CASE("sos_val - missing position columns count as zero")
{
    arma::mat spatial_freq = rnd(12, 3, -0.3, 0.3, 12), phase = rnd(12, 2, -pi, pi, 13);
    arma::mat pos3 = rnd(9, 3, -40.0, 40.0, 14);
    pos3.col(2).zeros();
    arma::mat pos2 = pos3.cols(0, 1);
    arma::mat pos1 = pos3.col(0), pos1z = pos3;
    pos1z.col(1).zeros();

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat v3, v2, v1, v1z;
        quadriga_lib::sos_val(spatial_freq, phase, pos3, v3, {}, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, pos2, v2, {}, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, pos1, v1, {}, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, pos1z, v1z, {}, 1.0, true, false, kernel);
        INFO("kernel " << kernel);
        REQUIRE(v2.n_rows == 9);
        CHECK(arma::abs(v3).max() > 0.5);
        CHECK(arma::approx_equal(v2, v3, "absdiff", 0.0));
        CHECK(arma::approx_equal(v1, v1z, "absdiff", 0.0));
        CHECK(arma::abs(v1 - v3).max() > 0.01); // The y coordinate matters when it is given
    }

    // Frequencies with one and two columns
    arma::mat f1 = spatial_freq.col(0), f2 = spatial_freq.cols(0, 1), val;
    quadriga_lib::sos_val(f1, phase, pos1, val, {}, 1.0, true, false, 1);
    CHECK(std::abs(val(3, 1) - ref_s(f1, phase, 1, pos1, 3)) < 1e-12);
    quadriga_lib::sos_val(f2, phase, pos2, val, {}, 1.0, true, false, 1);
    CHECK(std::abs(val(3, 1) - ref_s(f2, phase, 1, pos2, 3)) < 1e-12);
}

TEST_CASE("sos_val - over random phases the values have zero mean, unit variance and the ACF of the sinusoids")
{
    // 20000 generators at 3 positions. Expectations are exact for any frequency table:
    // mean 0, variance 1, covariance (1/N) · sum_n cos( 2·pi · f_n·dx ).
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat spatial_freq = rnd(n_coeff, 3, -0.1, 0.1, 15), phase = rnd(n_coeff, n_gen, -pi, pi, 16);
    arma::mat pos = {{10.0, 20.0, 1.5}, {12.0, 21.0, 1.5}, {40.0, -15.0, 3.0}};

    arma::mat val;
    quadriga_lib::sos_val(spatial_freq, phase, pos, val);
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
                fd += spatial_freq(n, c) * (pos(q, c) - pos(0, c));
            ref += std::cos(2.0 * pi * fd) / (double)n_coeff;
        }
        QD_DIAG("0-%llu,%.5f,%.5f,%.5f\n", (unsigned long long)q, cov, ref, cov - ref);
        INFO("pair 0-" << q);
        CHECK(std::abs(cov - ref) < 5.0 * 0.01);
    }
}

TEST_CASE("sos_val - two devices follow the sum over both positions with the permuted, scaled frequencies")
{
    arma::mat spatial_freq = rnd(20, 3, -0.2, 0.2, 23), phase = rnd(20, 3, -pi, pi, 24);
    arma::mat pa = rnd(10, 3, -30.0, 30.0, 26), pb = rnd(10, 3, -30.0, 30.0, 27);
    const std::vector<arma::uword> p = qd_perm(20);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
        for (double scale : {1.0, 2.5})
        {
            arma::mat val;
            quadriga_lib::sos_val(spatial_freq, phase, pa, val, pb, scale, false, false, kernel);
            REQUIRE(val.n_rows == 10);
            REQUIRE(val.n_cols == 3);

            double max_err = 0.0;
            for (arma::uword q = 0; q < 10; ++q)
                for (arma::uword g = 0; g < 3; ++g)
                    max_err = std::max(max_err, std::abs(val(q, g) - ref_s2(spatial_freq, phase, g, pa, pb, q, p, scale)));
            QD_DIAG("\n# diag: sos_val two devices, kernel %d, scale %.1f: max_abs_err %.3e\n", kernel, scale, max_err);
            INFO("kernel " << kernel << ", scale " << scale);
            CHECK(max_err < (kernel == 1 ? 1e-12 : tol_avx2(20)));
            CHECK(arma::abs(val).max() > 0.3);
        }

    // A fixed second device is the one-device generator with the phases phi_n + 2·pi·b·f_p(n)·y (exact limit of one
    // device staying; GENERIC, rounding of the phase sum only)
    {
        const arma::rowvec y = {7.0, -3.0, 2.5};
        const arma::mat pby = arma::repmat(y, 10, 1);
        arma::mat ph_y = phase, v_two, v_one;
        for (arma::uword n = 0; n < 20; ++n)
            for (arma::uword g = 0; g < 3; ++g)
                ph_y(n, g) += 2.0 * pi * 2.5 * arma::dot(spatial_freq.row(p[n]), y);
        quadriga_lib::sos_val(spatial_freq, phase, pa, v_two, pby, 2.5, false, false, 1);
        quadriga_lib::sos_val(spatial_freq, ph_y, pa, v_one, {}, 1.0, true, false, 1);
        CHECK(arma::abs(v_two - v_one).max() < 1e-12);
    }

    // The second device at the origin gives the one-device value; the scale changes the result
    arma::mat v0, single, v1, v2;
    quadriga_lib::sos_val(spatial_freq, phase, pa, v0, arma::mat(10, 3, arma::fill::zeros), 1.0, false, false, 1);
    quadriga_lib::sos_val(spatial_freq, phase, pa, single, {}, 1.0, true, false, 1);
    quadriga_lib::sos_val(spatial_freq, phase, pa, v1, pb, 1.0, false, false, 1);
    quadriga_lib::sos_val(spatial_freq, phase, pa, v2, pb, 2.5, false, false, 1);
    CHECK(arma::approx_equal(v0, single, "absdiff", 1e-14));
    CHECK(arma::abs(v1 - v2).max() > 0.05);
}

TEST_CASE("sos_val - reciprocal two devices follow the normalised sum and are symmetric")
{
    arma::mat spatial_freq = rnd(20, 3, -0.2, 0.2, 17), phase = rnd(20, 3, -pi, pi, 18);
    arma::mat pa = rnd(10, 3, -30.0, 30.0, 19), pb = rnd(10, 3, -30.0, 30.0, 20);
    pb.row(0) = pa.row(0);                                   // Devices at the same place
    pb.row(1) = pa.row(1) + arma::rowvec({0.3, 0.0, 0.0});   // Close together
    pb.row(2) = pa.row(2) + arma::rowvec({500.0, 0.0, 0.0}); // Far apart
    const std::vector<arma::uword> p = qd_perm(20);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val, swp, dflt;
        quadriga_lib::sos_val(spatial_freq, phase, pa, val, pb, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, pb, swp, pa, 1.0, true, false, kernel);
        REQUIRE(val.n_rows == 10);
        REQUIRE(val.n_cols == 3);

        double tol = kernel == 1 ? 1e-12 : 2.0 * tol_avx2(20);
        double max_err = 0.0;
        QD_DIAG("\n# diag: sos_val reciprocal two devices, kernel %d\npos,c,val,ref,diff\n", kernel);
        for (arma::uword q = 0; q < 10; ++q)
        {
            double c = 0.0;
            for (arma::uword n = 0; n < 20; ++n)
            {
                double arg = 0.0;
                for (arma::uword k = 0; k < 3; ++k)
                    arg += (spatial_freq(n, k) - spatial_freq(p[n], k)) * (pa(q, k) - pb(q, k));
                c += std::cos(2.0 * pi * arg) / 20.0;
            }
            for (arma::uword g = 0; g < 3; ++g)
            {
                double ref = (ref_s2(spatial_freq, phase, g, pa, pb, q, p, 1.0) + ref_s2(spatial_freq, phase, g, pb, pa, q, p, 1.0)) / std::sqrt(2.0 + 2.0 * c);
                if (g == 0 && q < 4)
                    QD_DIAG("%llu,%.4f,%.9f,%.9f,%.3e\n", (unsigned long long)q, c, val(q, g), ref, val(q, g) - ref);
                max_err = std::max(max_err, std::abs(val(q, g) - ref));
            }
        }
        INFO("kernel " << kernel);
        CHECK(max_err < tol);
        CHECK(arma::abs(val).max() > 0.3);

        // Swapping the devices gives the same value bit for bit; reciprocal is the default
        CHECK(arma::all(arma::vectorise(val == swp)));
        quadriga_lib::sos_val(spatial_freq, phase, pa, dflt, pb, 1.0, true, false, kernel);
        CHECK(arma::all(arma::vectorise(val == dflt)));
    }
}

TEST_CASE("sos_val - reciprocal two devices keep unit variance for small and large distances")
{
    // The variance of s(a, b) + s(b, a) over the phases is 2 + 2·c with c the correlation of the two terms, which the
    // function computes from the frequencies, so the normalised value must have unit variance for every displacement
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat spatial_freq = rnd(n_coeff, 3, -0.05, 0.05, 21), phase = rnd(n_coeff, n_gen, -pi, pi, 22);

    arma::vec dx = {0.0, 0.5, 3.0, 10.0, 25.0, 40.0};
    arma::mat pa(dx.n_elem, 3), pb;
    for (arma::uword q = 0; q < dx.n_elem; ++q)
        pa.row(q) = arma::rowvec({5.0 + 7.0 * (double)q, -3.0, 2.0});
    pb = pa;
    pb.col(0) += dx;

    arma::mat val;
    quadriga_lib::sos_val(spatial_freq, phase, pa, val, pb);
    REQUIRE(val.n_rows == dx.n_elem);
    REQUIRE(val.n_cols == n_gen);

    // Standard deviation of the sample variance: sqrt(2/n) = 0.01, limit 5 sigma
    QD_DIAG("\n# diag: sos_val reciprocal two devices, variance over %llu phase sets\ndx,variance\n", (unsigned long long)n_gen);
    for (arma::uword q = 0; q < dx.n_elem; ++q)
    {
        double var = mean_prod(val, q, q);
        QD_DIAG("%.1f,%.5f\n", dx(q), var);
        INFO("displacement " << dx(q));
        CHECK(std::abs(var - 1.0) < 5.0 * 0.01);
    }
}

TEST_CASE("sos_val - the Uniform output is the Normal output mapped through the Normal CDF")
{
    const arma::uword n_coeff = 64, n_gen = 20000;
    arma::mat spatial_freq = rnd(n_coeff, 3, -0.1, 0.1, 28), phase = rnd(n_coeff, n_gen, -pi, pi, 29);
    arma::mat pos = {{7.0, -4.0, 1.5}};

    arma::mat nrm, uni;
    quadriga_lib::sos_val(spatial_freq, phase, pos, nrm, {}, 1.0, true, false, 1);
    quadriga_lib::sos_val(spatial_freq, phase, pos, uni, {}, 1.0, true, true, 1);
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
    arma::mat pb = {{9.0, -4.0, 1.5}}, ph2 = phase.cols(0, 4), n2, u2;
    quadriga_lib::sos_val(spatial_freq, ph2, pos, n2, pb, 1.0, true, false, 1);
    quadriga_lib::sos_val(spatial_freq, ph2, pos, u2, pb, 1.0, true, true, 1);
    for (arma::uword g = 0; g < 5; ++g)
        CHECK(std::abs(u2(0, g) - 0.5 * std::erfc(-n2(0, g) / std::sqrt(2.0))) < 1e-14);
}

TEST_CASE("sos_val - empty optional matrices mean omitted, outputs are sized and overwritten")
{
    arma::mat spatial_freq = rnd(10, 3, -0.3, 0.3, 30), phase = rnd(10, 2, -pi, pi, 31), pos = rnd(5, 3, -20.0, 20.0, 32);
    arma::mat empty;

    arma::mat ref, val;
    quadriga_lib::sos_val(spatial_freq, phase, pos, ref);
    REQUIRE(ref.n_rows == 5);
    REQUIRE(ref.n_cols == 2);
    CHECK(arma::abs(ref).max() > 0.3);

    quadriga_lib::sos_val(spatial_freq, phase, pos, val, empty);
    CHECK(arma::approx_equal(val, ref, "absdiff", 0.0));

    arma::mat sized(5, 2);
    sized.fill(99.0);
    const double *ptr = sized.memptr();
    quadriga_lib::sos_val(spatial_freq, phase, pos, sized);
    CHECK(sized.memptr() == ptr);
    CHECK(arma::approx_equal(sized, ref, "absdiff", 0.0));

    arma::mat wrong(2, 9);
    wrong.fill(99.0);
    quadriga_lib::sos_val(spatial_freq, phase, pos, wrong);
    REQUIRE(wrong.n_rows == 5);
    REQUIRE(wrong.n_cols == 2);
    CHECK(arma::approx_equal(wrong, ref, "absdiff", 0.0));
}

TEST_CASE("sos_val - float agrees with double within single precision")
{
    arma::mat spatial_freq = rnd(50, 3, -0.3, 0.3, 33), phase = rnd(50, 4, -pi, pi, 34);
    arma::mat pa = rnd(21, 3, -100.0, 100.0, 35), pb = rnd(21, 3, -100.0, 100.0, 36);
    arma::fmat ff = arma::conv_to<arma::fmat>::from(spatial_freq), pf = arma::conv_to<arma::fmat>::from(phase);
    arma::fmat paf = arma::conv_to<arma::fmat>::from(pa), pbf = arma::conv_to<arma::fmat>::from(pb);

    // Inputs rounded to float: the argument 2·pi·f·x of magnitude up to about 300 rad carries a relative
    // error of 1.2e-7 from each of f and x, i.e. up to 7e-5 rad per term. The sum of 50 terms with
    // amplitude 0.2 is then within about 1e-3 in the worst case; random errors give about 1e-4.
    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat vd;
        arma::fmat vf;
        quadriga_lib::sos_val(spatial_freq, phase, pa, vd, pb, 1.0, true, false, kernel);
        quadriga_lib::sos_val(ff, pf, paf, vf, pbf, 1.0, true, false, kernel);
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
    arma::mat spatial_freq = rnd(100, 3, -0.3, 0.3, 37), phase = rnd(100, 7, -pi, pi, 38), pos = rnd(1000, 3, -500.0, 500.0, 39);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat a, b, c;
        quadriga_lib::sos_val(spatial_freq, phase, pos, a, {}, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, pos, b, {}, 1.0, true, false, kernel);
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        quadriga_lib::sos_val(spatial_freq, phase, pos, c, {}, 1.0, true, false, kernel);
#ifdef _OPENMP
        omp_set_num_threads(n_threads);
#endif
        INFO("kernel " << kernel);
        REQUIRE(a.n_rows == 1000);
        CHECK(arma::abs(a).max() > 0.5);
        CHECK(arma::approx_equal(a, b, "absdiff", 0.0));
        CHECK(arma::approx_equal(a, c, "absdiff", 0.0));

        // Two devices, reciprocal and not, Normal and Uniform: the combination, the normalisation and the mapping run in
        // parallel loops as well
        arma::mat pos_b = rnd(1000, 3, -500.0, 500.0, 139);
        for (int mode = 1; mode <= 2; ++mode)
            for (bool uniform : {false, true})
            {
                arma::mat all, one;
                quadriga_lib::sos_val(spatial_freq, phase, pos, all, pos_b, mode == 2 ? 1.5 : 1.0, mode == 1, uniform, kernel);
#ifdef _OPENMP
                omp_set_num_threads(1);
#endif
                quadriga_lib::sos_val(spatial_freq, phase, pos, one, pos_b, mode == 2 ? 1.5 : 1.0, mode == 1, uniform, kernel);
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
    arma::mat spatial_freq = rnd(100, 3, -0.3, 0.3, 40), phase = rnd(100, 2, -pi, pi, 41), pos = rnd(30, 3, -50.0, 50.0, 42);
    arma::mat v0, vk;
    quadriga_lib::sos_val(spatial_freq, phase, pos, v0);
    quadriga_lib::sos_val(spatial_freq, phase, pos, vk, {}, 1.0, true, false, has_avx2() ? 2 : 1);
    REQUIRE(v0.n_rows == 30);
    CHECK(arma::abs(v0).max() > 0.5);
    CHECK(arma::approx_equal(v0, vk, "absdiff", 0.0));
}

TEST_CASE("sos_val - input validation")
{
    arma::mat spatial_freq = rnd(10, 3, -0.3, 0.3, 43), phase = rnd(10, 2, -pi, pi, 44);
    arma::mat pos = rnd(5, 3, -20.0, 20.0, 46), pos_b = rnd(5, 3, -20.0, 20.0, 47), val;

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val));
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b));
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, 1.0, false));
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, 2.0, false));
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, true, 1));
    }
    SECTION("spatial_freq")
    {
        arma::mat empty, four(10, 4, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_val(empty, phase, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(four, phase, pos, val), std::invalid_argument);
    }
    SECTION("phase")
    {
        arma::mat empty, rows(9, 2, arma::fill::ones);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, empty, pos, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, rows, pos, val), std::invalid_argument);
    }
    SECTION("pos")
    {
        arma::mat empty, f2 = spatial_freq.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, empty, val), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(f2, phase, pos, val), std::invalid_argument); // More columns than spatial_freq
    }
    SECTION("pos_b")
    {
        arma::mat rows(4, 3, arma::fill::ones), f2 = spatial_freq.cols(0, 1), p2 = pos.cols(0, 1);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, rows), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(f2, phase, p2, val, pos_b), std::invalid_argument);
    }
    SECTION("spatial_freq_scale_b")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, 0.0, false), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, -1.0, false), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, std::nan(""), false), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, arma::datum::inf, false), std::invalid_argument);
        // A scale other than 1 needs a second device and the non-reciprocal mode
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 2.0, false), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, 2.0, true), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, pos_b, 2.0), std::invalid_argument);
    }
    SECTION("use_kernel")
    {
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, 3), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, -1), std::invalid_argument);
    }
    SECTION("a kernel that is not available")
    {
        // Runs in a build without AVX2 or on a CPU without it; with AVX2 the same call must work
        if (!has_avx2())
            CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, 2), std::invalid_argument);
        else
            CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, val, {}, 1.0, true, false, 2));
    }
    SECTION("a failed call leaves the output untouched")
    {
        arma::mat keep(2, 2);
        keep.fill(7.0);
        arma::mat empty;
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, empty, keep), std::invalid_argument);
        CHECK(keep.n_rows == 2);
        CHECK(keep(1, 1) == 7.0);
    }
    SECTION("the output cannot be an input")
    {
        // 'val' is resized and written while the inputs are read. Each input in turn is passed as output;
        // the call must throw and leave that matrix as it was. A copy with the same values is accepted.
        arma::mat f = spatial_freq, ph = phase, p = pos, pb = pos_b;
        CHECK_THROWS_AS(quadriga_lib::sos_val(f, phase, pos, f), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, ph, pos, ph), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, p, p), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, pos, pb, pb), std::invalid_argument);
        CHECK(arma::approx_equal(f, spatial_freq, "absdiff", 0.0));
        CHECK(arma::approx_equal(ph, phase, "absdiff", 0.0));
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, pos_b, "absdiff", 0.0));

        // Another matrix object on the memory of the positions
        arma::mat view(p.memptr(), p.n_rows, p.n_cols, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val(spatial_freq, phase, p, view), std::invalid_argument);
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));

        arma::mat out = pos;
        CHECK_NOTHROW(quadriga_lib::sos_val(spatial_freq, phase, pos, out));
        CHECK(out.n_rows == pos.n_rows);
        CHECK(out.n_cols == phase.n_cols);
    }
}

TEST_CASE("sos_val - an output that shares only part of the memory of an input throws")
{
    // 'val' as a matrix on foreign memory that starts inside an input, or ends inside one. With two devices
    // the first kernel writes 'val' before the second kernel and the normalisation read the inputs again, so an
    // overlap gives wrong values. Every such call must throw before anything is written; memory that only touches an
    // input is accepted.
    arma::mat spatial_freq = rnd(10, 3, -0.3, 0.3, 60), phase = rnd(10, 1, -pi, pi, 61);
    arma::mat pos = rnd(40, 3, -20.0, 20.0, 63), pos_b = pos + 1.0;

    arma::mat ref_recip, ref_plain;
    quadriga_lib::sos_val<double>(spatial_freq, phase, pos, ref_recip, pos_b);
    quadriga_lib::sos_val<double>(spatial_freq, phase, pos, ref_plain, pos_b, 1.5, false);

    SECTION("val on column 1 of pos, every device mode")
    {
        arma::mat p = pos;
        arma::mat v(p.colptr(1), 40, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, phase, p, v, pos_b), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, phase, p, v, pos_b, 1.5, false), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, phase, p, v), std::invalid_argument);
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));
    }
    SECTION("val starts before an input and ends inside it")
    {
        // One block: val on elements 10 to 49, pos_b on 45 to 164
        arma::vec block(165);
        std::copy(pos_b.begin(), pos_b.end(), block.begin() + 45);
        arma::mat pb(block.memptr() + 45, 40, 3, false, true), v(block.memptr() + 10, 40, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, phase, pos, v, pb), std::invalid_argument);
        CHECK(arma::approx_equal(pb, pos_b, "absdiff", 0.0));
    }
    SECTION("each input in turn")
    {
        // A one-element output on the last element of each input (one position: 'val' is [1, 1])
        arma::mat f = spatial_freq, ph = phase, pb = pos_b, p1 = pos.row(0), pb1 = pos_b.row(0);
        arma::mat vf(f.memptr() + f.n_elem - 1, 1, 1, false, true), vph(ph.memptr() + ph.n_elem - 1, 1, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(f, phase, p1, vf, pb1), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, ph, p1, vph, pb1), std::invalid_argument);
        arma::mat vpb(pb.memptr() + 70, 40, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val<double>(spatial_freq, phase, pos, vpb, pb), std::invalid_argument);
        CHECK(arma::approx_equal(f, spatial_freq, "absdiff", 0.0));
        CHECK(arma::approx_equal(ph, phase, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, pos_b, "absdiff", 0.0));
    }
    SECTION("the error names the output")
    {
        arma::mat p = pos;
        arma::mat v(p.colptr(2), 40, 1, false, true);
        std::string msg;
        try
        {
            quadriga_lib::sos_val<double>(spatial_freq, phase, p, v, pos_b);
        }
        catch (const std::invalid_argument &e)
        {
            msg = e.what();
        }
        CHECK(msg.find("'val'") != std::string::npos);
    }
    SECTION("float")
    {
        arma::fmat ff = arma::conv_to<arma::fmat>::from(spatial_freq), phf = arma::conv_to<arma::fmat>::from(phase);
        arma::fmat pf = arma::conv_to<arma::fmat>::from(pos), pbf = arma::conv_to<arma::fmat>::from(pos_b), keep = pf;
        arma::fmat v(pf.colptr(1), 40, 1, false, true);
        CHECK_THROWS_AS(quadriga_lib::sos_val<float>(ff, phf, pf, v, pbf), std::invalid_argument);
        CHECK(arma::approx_equal(pf, keep, "absdiff", 0.0f));
    }
    SECTION("memory next to an input is accepted")
    {
        // One block: pos on elements 0 to 119, val on 120 to 159, pos_b on 160 to 279
        arma::vec block(280);
        std::copy(pos.begin(), pos.end(), block.begin());
        std::copy(pos_b.begin(), pos_b.end(), block.begin() + 160);
        arma::mat p(block.memptr(), 40, 3, false, true), pb(block.memptr() + 160, 40, 3, false, true);
        arma::mat v(block.memptr() + 120, 40, 1, false, true);
        REQUIRE_NOTHROW(quadriga_lib::sos_val<double>(spatial_freq, phase, p, v, pb));
        CHECK(arma::approx_equal(v, ref_recip, "absdiff", 0.0));
        REQUIRE_NOTHROW(quadriga_lib::sos_val<double>(spatial_freq, phase, p, v, pb, 1.5, false));
        CHECK(arma::approx_equal(v, ref_plain, "absdiff", 0.0));
        CHECK(arma::approx_equal(p, pos, "absdiff", 0.0));
        CHECK(arma::approx_equal(pb, pos_b, "absdiff", 0.0));
    }
}

TEST_CASE("sos_val - a NaN or Inf coordinate gives NaN at that position only")
{
    // 19 positions: three AVX2 blocks, the bad positions share their blocks with good ones
    arma::mat spatial_freq = rnd(40, 3, -0.3, 0.3, 50), phase = rnd(40, 5, -pi, pi, 51), pos = rnd(19, 3, -80.0, 80.0, 52), bad = pos;
    bad(3, 1) = std::nan(""), bad(7, 0) = arma::datum::inf, bad(12, 2) = -arma::datum::inf;

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat ref, val;
        quadriga_lib::sos_val(spatial_freq, phase, pos, ref, {}, 1.0, true, false, kernel);
        quadriga_lib::sos_val(spatial_freq, phase, bad, val, {}, 1.0, true, false, kernel);
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
    arma::mat spatial_freq = rnd(n_coeff, 3, -0.1, 0.1, 53), phase = rnd(n_coeff, 4, -pi, pi, 54);

    QD_DIAG("\n# diag: sos_val AVX2 against GENERIC far from the origin, N = %llu, bound %.2e\n", (unsigned long long)n_coeff, tol_avx2(n_coeff));
    QD_DIAG("offset_m,periods,max_abs_diff\n");
    for (double offset : {1e8, 1e9, 1e10})
    {
        arma::mat pos = rnd(64, 3, -50.0, 50.0, 55) + offset, a, b;
        quadriga_lib::sos_val(spatial_freq, phase, pos, a, {}, 1.0, true, false, 1);
        quadriga_lib::sos_val(spatial_freq, phase, pos, b, {}, 1.0, true, false, 2);
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
        arma::mat spatial_freq, phase = rnd(1024, 6, -pi, pi, 56), pos = rnd(200, 3, -500.0, 500.0, 57), a, b;
        spatial_freq = quadriga_lib::sos_preset("Comb1024", (arma::vec *)nullptr, (arma::vec *)nullptr, d);
        quadriga_lib::sos_val(spatial_freq, phase, pos, a, {}, 1.0, true, false, 1);
        quadriga_lib::sos_val(spatial_freq, phase, pos, b, {}, 1.0, true, false, 2);
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
    arma::mat spatial_freq = rnd(30, 3, -0.2, 0.2, 58), phase = rnd(30, 5, -pi, pi, 59);
    arma::mat pa3 = rnd(13, 3, -30.0, 30.0, 61), pb3 = rnd(13, 3, -30.0, 30.0, 62);

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
                    quadriga_lib::sos_val(spatial_freq, phase, pa, val, pb, mode == 2 ? 0.7 : 1.0, mode == 1, false, kernel);
                    quadriga_lib::sos_val(spatial_freq, phase, pa_pad, ref, pb_pad, mode == 2 ? 0.7 : 1.0, mode == 1, false, kernel);
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
    arma::fmat paf = arma::conv_to<arma::fmat>::from(rnd(21, 3, -100.0, 100.0, 66)), pbpf = arma::conv_to<arma::fmat>::from(rnd(21, 3, -100.0, 100.0, 67));
    arma::mat fd = arma::conv_to<arma::mat>::from(ff), pd = arma::conv_to<arma::mat>::from(pf);
    arma::mat pad = arma::conv_to<arma::mat>::from(paf), pbpd = arma::conv_to<arma::mat>::from(pbpf);
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
                    quadriga_lib::sos_val(ff, pf, paf, vf, {}, 1.0f, true, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, {}, 1.0, true, uniform, kernel);
                }
                else if (mode == 1)
                {
                    quadriga_lib::sos_val(ff, pf, paf, vf, pbpf, 1.0, true, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, pbpd, 1.0, true, uniform, kernel);
                }
                else
                {
                    quadriga_lib::sos_val(ff, pf, paf, vf, pbpf, scale_f, false, uniform, kernel);
                    quadriga_lib::sos_val(fd, pd, pad, vd, pbpd, (double)scale_f, false, uniform, kernel);
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
    arma::mat spatial_freq = rnd(33, 3, -0.3, 0.3, 70), phase = rnd(33, 18, -pi, pi, 71), pos = rnd(300, 3, -80.0, 80.0, 72);

    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat single(300, 18);
        for (arma::uword g = 0; g < 18; ++g)
        {
            arma::mat one, ph = phase.col(g);
            quadriga_lib::sos_val(spatial_freq, ph, pos, one, {}, 1.0, true, false, kernel);
            REQUIRE(one.n_rows == 300);
            single.col(g) = one;
        }
        CHECK(arma::abs(single).max() > 0.5);

        arma::uword n_equal = 0;
        for (arma::uword n_gen = 1; n_gen <= 18; ++n_gen)
        {
            arma::mat val, ph = phase.cols(0, n_gen - 1);
            quadriga_lib::sos_val(spatial_freq, ph, pos, val, {}, 1.0, true, false, kernel);
            REQUIRE(val.n_cols == n_gen);
            n_equal += arma::approx_equal(val, single.cols(0, n_gen - 1), "absdiff", 0.0) ? 1 : 0;
        }
        INFO("kernel " << kernel);
        CHECK(n_equal == 18);
    }
}

TEST_CASE("sos_val - two devices: a link decorrelates when one device moves away")
{
    // 2000 generators, Exp256 at 20 m. Links (Rx = pos, Tx = pos_b along x): L0 Rx 500 m, Tx 0; L1 Rx moved 20 m;
    // L2 Rx moved 1000 m; L3 Tx moved 1000 m; L4 both moved 1000 m. Correlation over the generators against L0:
    // L1 the one-device ACF at 20 m (exp(-1); the preset reaches it within 0.05 along an axis), L2 to L4 about 0: at
    // one displacement the residual of the sinusoids is a fixed value of rms 1/sqrt(2N) = 0.044 (up to about 4 times
    // that), plus a sampling error of 0.022, so |corr| < 0.2. The sum of two one-device fields gives (1 + 0.37)/2 =
    // 0.68 for L1 and 0.5 for L2 and L3. L5: both devices moved 20 m, the product of the two ACFs exp(-2) = 0.135
    // (within 0.1); the sum gives 0.37.
    const arma::mat F = quadriga_lib::sos_preset<double>("Exp256", nullptr, nullptr, 20.0);
    const arma::mat P = quadriga_lib::sos_init_phases<double>(F.n_rows, 2000, 3);
    const arma::mat rx = {{500.0, 0.0, 0.0}, {520.0, 0.0, 0.0}, {1500.0, 0.0, 0.0}, {500.0, 0.0, 0.0}, {1500.0, 0.0, 0.0}, {520.0, 0.0, 0.0}};
    const arma::mat tx = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1000.0, 0.0, 0.0}, {1000.0, 0.0, 0.0}, {0.0, 20.0, 0.0}};
    arma::mat val;
    quadriga_lib::sos_val(F, P, rx, val, tx);
    auto corr = [&](arma::uword k)
    { return arma::as_scalar(arma::cor(arma::vec(val.row(0).t()), arma::vec(val.row(k).t()))); };
    QD_DIAG("\n# diag: sos_val two devices, correlation with L0: L1 %.3f, L2 %.3f, L3 %.3f, L4 %.3f, L5 %.3f\n", corr(1), corr(2), corr(3), corr(4), corr(5));
    CHECK(std::abs(corr(1) - std::exp(-1.0)) < 0.1);
    CHECK(std::abs(corr(2)) < 0.2);
    CHECK(std::abs(corr(3)) < 0.2);
    CHECK(std::abs(corr(4)) < 0.2);
    CHECK(std::abs(corr(5) - std::exp(-2.0)) < 0.1);
}

TEST_CASE("sos_val - reciprocal two devices give NaN where the normalisation is singular")
{
    // Two sinusoids: the permutation swaps them, c = cos( 2·pi · (f_0 - f_1)·d ) = -1 at d = 2.5 m for f = 0.1, 0.3 / m
    const arma::mat spatial_freq = {{0.1, 0.0, 0.0}, {0.3, 0.0, 0.0}}, phase = {{0.4, -1.1}, {2.0, 0.3}};
    const arma::mat pa = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, pb = {{2.5, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    for (int kernel = 1; kernel <= (has_avx2() ? 2 : 1); ++kernel)
    {
        arma::mat val;
        quadriga_lib::sos_val(spatial_freq, phase, pa, val, pb, 1.0, true, false, kernel);
        INFO("kernel " << kernel);
        CHECK(val.row(0).has_nan());
        CHECK(val.row(1).is_finite());
    }
    arma::mat ph = phase;
    CHECK_THROWS_AS(quadriga_lib::sos_remove_bias(spatial_freq, ph, pa, pb), std::invalid_argument);
}
