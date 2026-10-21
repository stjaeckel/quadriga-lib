// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"
#include <algorithm>
#include <cmath>
#include <random>

TEST_CASE("calc_delay_spread - Basic single CIR")
{
    // Single CIR with 3 paths: delays = [0, 1e-6, 2e-6], powers = [1.0, 0.5, 0.25]
    arma::vec tau = {0.0, 1e-6, 2e-6};
    arma::vec pow = {1.0, 0.5, 0.25};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    REQUIRE(mean_delay.n_elem == 1);

    // Expected mean delay: (1.0*0 + 0.5*1e-6 + 0.25*2e-6) / (1.0+0.5+0.25) = 1e-6 / 1.75
    double expected_mean = 1.0e-6 / 1.75;
    CHECK(std::abs(mean_delay(0) - expected_mean) < 1e-14);

    // Verify delay spread is positive and reasonable
    CHECK(ds(0) > 0.0);
    CHECK(ds(0) < 2e-6); // Must be less than max delay spread
}

TEST_CASE("calc_delay_spread - Single path yields zero spread")
{
    arma::vec tau = {5e-6};
    arma::vec pow = {1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(ds(0)) < 1e-20);
    CHECK(std::abs(mean_delay(0) - 5e-6) < 1e-20);
}

TEST_CASE("calc_delay_spread - Equal power two paths")
{
    // Two equal-power paths: DS should be half the delay difference
    arma::vec tau = {0.0, 2e-6};
    arma::vec pow = {1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 1);

    // Mean delay = 1e-6
    CHECK(std::abs(mean_delay(0) - 1e-6) < 1e-14);

    // DS = sqrt( 0.5*(0-1e-6)^2 + 0.5*(2e-6-1e-6)^2 ) = 1e-6
    CHECK(std::abs(ds(0) - 1e-6) < 1e-14);
}

TEST_CASE("calc_delay_spread - Multiple CIRs")
{
    arma::vec tau1 = {0.0, 1e-6};
    arma::vec pow1 = {1.0, 1.0};

    arma::vec tau2 = {0.0, 1e-6, 2e-6};
    arma::vec pow2 = {1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau1, tau2};
    std::vector<arma::vec> powers = {pow1, pow2};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0, &mean_delay);

    REQUIRE(ds.n_elem == 2);
    REQUIRE(mean_delay.n_elem == 2);

    // CIR 1: mean = 0.5e-6, DS = 0.5e-6
    CHECK(std::abs(mean_delay(0) - 0.5e-6) < 1e-14);
    CHECK(std::abs(ds(0) - 0.5e-6) < 1e-14);

    // CIR 2: mean = 1e-6
    CHECK(std::abs(mean_delay(1) - 1e-6) < 1e-14);
    CHECK(ds(1) > 0.0);
}

TEST_CASE("calc_delay_spread - Threshold filters weak paths")
{
    // Path 1: 1.0 W (0 dB), Path 2: 0.001 W (-30 dB), Path 3: 0.5 W (-3 dB)
    arma::vec tau = {0.0, 10e-6, 1e-6};
    arma::vec pow = {1.0, 0.001, 0.5};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // With 20 dB threshold, path 2 (-30 dB) should be excluded
    arma::vec ds_20 = quadriga_lib::calc_delay_spread(delays, powers, 20.0);

    // With 100 dB threshold, all paths included
    arma::vec ds_all = quadriga_lib::calc_delay_spread(delays, powers, 100.0);

    // DS with all paths should be larger (path at 10e-6 increases spread)
    CHECK(ds_all(0) > ds_20(0));
}

TEST_CASE("calc_delay_spread - Granularity bins paths")
{
    // Two paths very close together at ~100 ns and ~110 ns, one path at 1000 ns
    arma::vec tau = {100e-9, 110e-9, 1000e-9};
    arma::vec pow = {1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // Without granularity
    arma::vec ds_no_gran = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 0.0);

    // With 50 ns granularity - first two paths should be binned together
    arma::vec ds_gran = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 50e-9);

    // Both should produce positive delay spreads
    CHECK(ds_no_gran(0) > 0.0);
    CHECK(ds_gran(0) > 0.0);
}

TEST_CASE("calc_delay_spread - Without mean_delay output")
{
    arma::vec tau = {0.0, 1e-6};
    arma::vec pow = {1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    // Call without mean_delay pointer
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(ds(0) - 0.5e-6) < 1e-14);
}

TEST_CASE("calc_delay_spread - Float precision")
{
    arma::fvec tau = {0.0f, 1e-6f, 2e-6f};
    arma::fvec pow = {1.0f, 1.0f, 1.0f};

    std::vector<arma::fvec> delays = {tau};
    std::vector<arma::fvec> powers = {pow};

    arma::fvec mean_delay;
    arma::fvec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0f, 0.0f, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    CHECK(std::abs(mean_delay(0) - 1e-6f) < 1e-10f);
    CHECK(ds(0) > 0.0f);
}

TEST_CASE("calc_delay_spread - Input validation")
{
    std::vector<arma::vec> empty_delays;
    std::vector<arma::vec> powers = {{1.0}};

    // Empty delays
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(empty_delays, powers), std::invalid_argument);

    // Mismatched sizes
    std::vector<arma::vec> delays = {{0.0}, {1e-6}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays, powers), std::invalid_argument);

    // Mismatched path counts
    std::vector<arma::vec> delays2 = {{0.0, 1e-6}};
    std::vector<arma::vec> powers2 = {{1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays2, powers2), std::invalid_argument);

    // Negative threshold
    std::vector<arma::vec> delays3 = {{0.0}};
    std::vector<arma::vec> powers3 = {{1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays3, powers3, -10.0), std::invalid_argument);

    // Negative granularity
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread(delays3, powers3, 100.0, -1.0), std::invalid_argument);
}

TEST_CASE("calc_delay_spread - Granularity with mean_delay output")
{
    arma::vec tau = {0.0, 50e-9, 500e-9, 550e-9};
    arma::vec pow = {1.0, 1.0, 1.0, 1.0};

    std::vector<arma::vec> delays = {tau};
    std::vector<arma::vec> powers = {pow};

    arma::vec mean_delay;
    arma::vec ds = quadriga_lib::calc_delay_spread(delays, powers, 100.0, 100e-9, &mean_delay);

    REQUIRE(ds.n_elem == 1);
    REQUIRE(mean_delay.n_elem == 1);

    // With 100 ns granularity, paths at 0/50ns -> bin 0, paths at 500/550ns -> bin 500ns
    // Two bins with equal total power -> mean delay ≈ 250 ns
    CHECK(mean_delay(0) > 0.0);
    CHECK(ds(0) > 0.0);
}

// Reference: the dense binning over the global delay range (the algorithm before the sparse binning), followed by
// the direct computation on the binned profile
template <typename dtype>
static arma::Col<dtype> ds_dense_reference(const std::vector<arma::Col<dtype>> &d, const std::vector<arma::Col<dtype>> &p,
                                           dtype threshold, dtype g, arma::Col<dtype> *mean_delay)
{
    dtype gmin = d[0].min(), gmax = d[0].max();
    for (auto &x : d)
        gmin = std::min(gmin, x.min()), gmax = std::max(gmax, x.max());
    dtype min_bin = std::floor(gmin / g) * g, max_bin = std::ceil(gmax / g) * g;
    arma::uword n_bins = (arma::uword)std::round((max_bin - min_bin) / g) + 1;
    arma::Col<dtype> axis(n_bins);
    for (arma::uword b = 0; b < n_bins; ++b)
        axis(b) = min_bin + (dtype)b * g;
    std::vector<arma::Col<dtype>> bd(d.size()), bp(d.size());
    for (size_t i = 0; i < d.size(); ++i)
    {
        arma::Col<dtype> pdp(n_bins, arma::fill::zeros);
        for (arma::uword k = 0; k < d[i].n_elem; ++k)
        {
            arma::uword b = (arma::uword)std::round((d[i](k) - min_bin) / g);
            if (b < n_bins)
                pdp(b) += p[i](k);
        }
        bd[i] = axis, bp[i] = pdp;
    }
    return quadriga_lib::calc_delay_spread<dtype>(bd, bp, threshold, (dtype)0.0, mean_delay);
}

TEST_CASE("calc_delay_spread - Granularity: sparse binning equals the dense reference")
{
    std::mt19937 gen(31);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::uniform_int_distribution<int> N(1, 40);
    std::vector<arma::vec> d(60), p(60);
    for (size_t i = 0; i < d.size(); ++i)
    {
        int n = N(gen);
        d[i].set_size(n), p[i].set_size(n);
        for (int k = 0; k < n; ++k)
            d[i](k) = 2.0e-6 * U(gen) + 0.3e-6 * (double)(i % 3), p[i](k) = std::pow(10.0, -3.0 * U(gen));
        if (n > 3)
            d[i](1) = d[i](0), d[i](2) = d[i](0) + 1.0e-12; // shared bins (summed in path order)
    }
    for (double g : {1.0e-9, 7.3e-9, 50.0e-9, 1.0e-6})
        for (double th : {100.0, 20.0})
        {
            arma::vec md, md_ref;
            arma::vec ds = quadriga_lib::calc_delay_spread<double>(d, p, th, g, &md);
            arma::vec ds_ref = ds_dense_reference<double>(d, p, th, g, &md_ref);
            INFO("granularity " << g << ", threshold " << th);
            CHECK(arma::approx_equal(ds, ds_ref, "absdiff", 0.0));
            CHECK(arma::approx_equal(md, md_ref, "absdiff", 0.0));
        }

    // Float, and delays with a negative offset
    std::vector<arma::fvec> df(d.size()), pf(d.size());
    for (size_t i = 0; i < d.size(); ++i)
        df[i] = arma::conv_to<arma::fvec>::from(d[i] - 0.5e-6), pf[i] = arma::conv_to<arma::fvec>::from(p[i]);
    arma::fvec mdf, mdf_ref;
    arma::fvec dsf = quadriga_lib::calc_delay_spread<float>(df, pf, 100.0f, 5.0e-9f, &mdf);
    arma::fvec dsf_ref = ds_dense_reference<float>(df, pf, 100.0f, 5.0e-9f, &mdf_ref);
    CHECK(arma::approx_equal(dsf, dsf_ref, "absdiff", 0.0f));
    CHECK(arma::approx_equal(mdf, mdf_ref, "absdiff", 0.0f));
}

TEST_CASE("calc_delay_spread - Granularity: a 10 s delay span at 1 ns needs no dense bin axis")
{
    // 1e10 bins over the global span: the dense binning would need 80 GB per CIR
    std::vector<arma::vec> d = {{0.0, 10.0}, {5.0, 5.0 + 2.0e-9}}, p = {{1.0, 1.0}, {1.0, 3.0}};
    arma::vec md;
    arma::vec ds = quadriga_lib::calc_delay_spread<double>(d, p, 100.0, 1.0e-9, &md);
    CHECK(std::abs(ds(0) - 5.0) < 1e-9);
    CHECK(std::abs(md(0) - 5.0) < 1e-9);
    CHECK(std::abs(ds(1) - std::sqrt(0.1875) * 2.0e-9) < 1e-14); // powers 0.25 / 0.75, 2 ns apart
}

TEST_CASE("calc_delay_spread - Granularity and delays must be finite, bin count bounded")
{
    std::vector<arma::vec> d = {{0.0, 1.0e-6}}, p = {{1.0, 0.5}};
    const double nan = std::nan(""), inf = INFINITY;
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(d, p, 100.0, nan), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(d, p, 100.0, inf), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(d, p, 100.0, -1.0e-9), std::invalid_argument);

    // Non-finite delays with granularity > 0
    for (double bad : {nan, inf, -inf})
    {
        std::vector<arma::vec> db = {{0.0, 1.0e-6}, {2.0e-7, bad}};
        std::vector<arma::vec> pb = {{1.0, 0.5}, {1.0, 0.5}};
        INFO("delay " << bad);
        CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(db, pb, 100.0, 1.0e-9), std::invalid_argument);
    }

    // (global span) / granularity >= 2^53 throws; just below is accepted
    std::vector<arma::vec> dl = {{0.0, 1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(dl, p, 100.0, 1.0e-16), std::invalid_argument);
    CHECK_NOTHROW(quadriga_lib::calc_delay_spread<double>(dl, p, 100.0, 1.0e-15));

    // Delay / granularity beyond the floating-point range (the bin grid origin would be infinite)
    std::vector<arma::vec> d1 = {{1.0, 1.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<double>(d1, p, 100.0, 1.0e-320), std::invalid_argument);
    std::vector<arma::fvec> df = {{0.1f, 0.1f}}, pf = {{1.0f, 0.5f}};
    CHECK_THROWS_AS(quadriga_lib::calc_delay_spread<float>(df, pf, 100.0f, 1.0e-40f), std::invalid_argument);

    // Without granularity the delays are not checked (as before)
    std::vector<arma::vec> dn = {{0.0, nan}};
    CHECK_NOTHROW(quadriga_lib::calc_delay_spread<double>(dn, p, 100.0, 0.0));
}
