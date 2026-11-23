// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Quick validity check of the library functions (qd_arrayant_is_valid_quick): passes at once when the cache of the
// arrayant (check_ptr) matches the data pointers and the member sizes fit together, else runs the full check

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_helpers_arrayant.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// True if the validity cache of an arrayant (private member check_ptr) is empty
template <typename dtype>
static bool unmarked(const quadriga_lib::arrayant<dtype> &ant)
{
    for (const auto &p : quadriga_lib::qd_arrayant_access::check_ptr(ant))
        if (p != nullptr)
            return false;
    return true;
}

#if defined(_MSC_VER)
#define ARRAYANT_TEST_NOINLINE __declspec(noinline)
#else
#define ARRAYANT_TEST_NOINLINE __attribute__((noinline))
#endif

// The default constructor must initialise check_ptr. The arrayant is created twice in the same stack slot;
// small matrices live inside the Armadillo objects, so stale check_ptr values of the first object would match the
// data pointers of the second one and let the quick check pass an invalid, never validated object.
static ARRAYANT_TEST_NOINLINE std::string arrayant_check_ptr_helper(bool first)
{
    quadriga_lib::arrayant<double> a;
    a.e_theta_re.ones(2, 2, 1), a.e_theta_im.zeros(2, 2, 1), a.e_phi_re.zeros(2, 2, 1), a.e_phi_im.zeros(2, 2, 1);
    a.elevation_grid = {-0.5, 0.5};
    a.element_pos.zeros(3, 1), a.port_weights_re.ones(1, 1), a.port_weights_im.zeros(1, 1);
    if (first)
    {
        a.azimuth_grid = {-1.0, 1.0};
        return a.is_valid();
    }
    a.azimuth_grid = {1.0, -1.0}; // invalid: unsorted; never validated
    return qd_arrayant_is_valid_quick(a);
}

TEST_CASE("Arrayant quick check - Default constructor initialises check_ptr")
{
    quadriga_lib::arrayant<double> a;
    CHECK(unmarked(a));

    CHECK(arrayant_check_ptr_helper(true).empty());
    CHECK(!arrayant_check_ptr_helper(false).empty());
}

// Armadillo can keep the data pointer when a member is resized, e.g. within its built-in storage for small objects
// or for a heap buffer with the same number of values. The quick check must still notice that the member sizes
// no longer fit together, so that interpolate throws instead of reading element_pos and the grids out of bounds.
// Whether a resize keeps the pointer differs between Armadillo versions, so the test sets check_ptr to the current
// pointers after the resize. This gives the same state on every version.
TEST_CASE("Arrayant quick check - Detects resized members with unchanged data pointers")
{
    quadriga_lib::arrayant<double> a;
    arma::mat az = {0.1}, el = {0.0}, vr, vi, hr, hi;

    SECTION("Built-in storage")
    {
        a.e_theta_re.zeros(1, 1, 1), a.e_theta_im.zeros(1, 1, 1), a.e_phi_re.zeros(1, 1, 1), a.e_phi_im.zeros(1, 1, 1);
        a.azimuth_grid = {0.0}, a.elevation_grid = {0.0};
        a.element_pos.zeros(3, 1), a.port_weights_re.eye(1, 1), a.port_weights_im.zeros(1, 1);
        REQUIRE(a.is_valid().empty());
        a.e_theta_re.ones(1, 1, 64), a.e_theta_im.zeros(1, 1, 64), a.e_phi_re.zeros(1, 1, 64), a.e_phi_im.zeros(1, 1, 64);
    }

    SECTION("Heap storage, same number of values")
    {
        a.e_theta_re.ones(2, 2, 20), a.e_theta_im.zeros(2, 2, 20), a.e_phi_re.zeros(2, 2, 20), a.e_phi_im.zeros(2, 2, 20);
        a.azimuth_grid = {-0.5, 0.5}, a.elevation_grid = {-0.5, 0.5};
        a.element_pos.zeros(3, 20), a.port_weights_re.eye(20, 20), a.port_weights_im.zeros(20, 20);
        REQUIRE(a.is_valid().empty());
        a.e_theta_re.ones(2, 1, 40), a.e_theta_im.zeros(2, 1, 40), a.e_phi_re.zeros(2, 1, 40), a.e_phi_im.zeros(2, 1, 40);
    }

    // Pointers as if the resize had kept them all, so a pointer comparison alone cannot see the change
    qd_arrayant_mark_valid(a);
    REQUIRE(!a.is_valid().empty());

    REQUIRE(!qd_arrayant_is_valid_quick(a).empty()); // stops here if the quick check misses the change
    CHECK_THROWS_AS(quadriga_lib::interpolate_pattern(a, az, el, vr, vi, hr, hi), std::invalid_argument);
}

// A value changed in place keeps the data pointers: the quick check passes, the public check finds the change.
// The quick check does not write to the object, a failed public check leaves the cache unchanged.
TEST_CASE("Arrayant quick check - Passes a marked object changed in place, the public check does not")
{
    auto a = quadriga_lib::generate_antenna<double>("omni", 30.0);
    a.azimuth_grid.memptr()[0] = a.azimuth_grid.memptr()[1]; // Not strictly ascending

    const std::string msg = "Values of 'azimuth_grid' must be sorted in strictly ascending order.";
    CHECK(qd_arrayant_is_valid_quick(a).empty());
    CHECK(a.is_valid() == msg);
    CHECK(qd_arrayant_is_valid_quick(a).empty());

    // A copy gets the full check of its source and is not marked
    quadriga_lib::arrayant<double> c = a;
    CHECK(unmarked(c));

    std::vector<quadriga_lib::arrayant<double>> vec(2, quadriga_lib::generate_antenna<double>("omni", 30.0));
    for (auto &v : vec) // The same change in all entries, so that only the per-entry check can see it
        v.azimuth_grid.memptr()[0] = v.azimuth_grid.memptr()[1];
    CHECK(qd_arrayant_vec_is_valid_quick(vec).empty());
    CHECK(quadriga_lib::is_valid(vec) == "Entry 0: " + msg);

    // An unmarked object gets the full check
    auto b = quadriga_lib::generate_antenna<double>("omni", 30.0);
    qd_arrayant_clear_mark(b);
    b.azimuth_grid.memptr()[0] = b.azimuth_grid.memptr()[1];
    CHECK(qd_arrayant_is_valid_quick(b) == msg);
    CHECK(unmarked(b));
}

// A passing is_valid marks the object (cache check_ptr holds its data pointers), a failing one does not
TEST_CASE("Arrayant quick check - is_valid marks a valid object")
{
    quadriga_lib::arrayant<double> b;
    b.e_theta_re.ones(2, 3, 1), b.e_theta_im.zeros(2, 3, 1), b.e_phi_re.zeros(2, 3, 1), b.e_phi_im.zeros(2, 3, 1);
    b.azimuth_grid = {-1.0, 0.0, 1.0}, b.elevation_grid = {-0.5, 0.5};
    b.element_pos.zeros(3, 1), b.port_weights_re.ones(1, 1), b.port_weights_im.zeros(1, 1);
    REQUIRE(unmarked(b));
    CHECK(b.is_valid().empty());
    CHECK(qd_arrayant_cache_matches(b));

    qd_arrayant_clear_mark(b);
    b.azimuth_grid = {1.0, 0.0, -1.0};
    CHECK(!b.is_valid().empty());
    CHECK(unmarked(b));
}

// Copy and move construction and assignment mark the target with its own data pointers when the source is valid.
// Small members (up to 16 values) are stored inside the Armadillo objects, so a move copies them to new addresses.
TEST_CASE("Arrayant copy and move - The target of a valid source is marked with its own data")
{
    auto big = quadriga_lib::generate_antenna<double>("xpol", 10.0); // Pattern on the heap
    quadriga_lib::arrayant<double> small;                         // All members inside the objects
    small.e_theta_re.ones(1, 2, 1), small.e_theta_im.zeros(1, 2, 1), small.e_phi_re.zeros(1, 2, 1), small.e_phi_im.zeros(1, 2, 1);
    small.azimuth_grid = {-1.0, 1.0}, small.elevation_grid = {0.0};
    REQUIRE(small.is_valid().empty());

    for (auto *src : {&big, &small})
    {
        REQUIRE(qd_arrayant_cache_matches(*src));

        quadriga_lib::arrayant<double> a(*src);
        CHECK(qd_arrayant_cache_matches(a));

        quadriga_lib::arrayant<double> b;
        b = *src;
        CHECK(qd_arrayant_cache_matches(b));

        quadriga_lib::arrayant<double> c(std::move(a));
        CHECK(qd_arrayant_cache_matches(c));
        CHECK(unmarked(a));

        quadriga_lib::arrayant<double> d;
        d = std::move(b);
        CHECK(qd_arrayant_cache_matches(d));
        CHECK(unmarked(b));
        CHECK(d.is_valid().empty());
    }

    // An invalid source gives an unmarked target; assignment drops an old mark of the target
    auto bad = small;
    bad.e_theta_im.set_size(1, 3, 1);
    quadriga_lib::arrayant<double> g(bad);
    CHECK(unmarked(g));
    auto e = big;
    e = bad;
    CHECK(unmarked(e));
    quadriga_lib::arrayant<double> f(std::move(bad));
    CHECK(unmarked(f));
    CHECK(!f.is_valid().empty());
}

// Library functions check their inputs read only (threads may share an input): an unmarked valid input stays unmarked
TEST_CASE("Arrayant quick check - Library functions do not mark their inputs")
{
    quadriga_lib::arrayant<double> a;
    a.e_theta_re.ones(3, 4, 1), a.e_theta_im.zeros(3, 4, 1), a.e_phi_re.zeros(3, 4, 1), a.e_phi_im.zeros(3, 4, 1);
    a.azimuth_grid = {-3.0, -1.0, 1.0, 3.0}, a.elevation_grid = {-1.0, 0.0, 1.0};
    a.element_pos.zeros(3, 1), a.port_weights_re.ones(1, 1), a.port_weights_im.zeros(1, 1);
    REQUIRE(unmarked(a));

    quadriga_lib::apply_port_weights(a);
    arma::mat az = {0.1}, el = {0.2}, vr, vi, hr, hi;
    quadriga_lib::interpolate_pattern(a, az, el, vr, vi, hr, hi);
    CHECK(!qd_arrayant_cache_matches(a));

    std::vector<quadriga_lib::arrayant<double>> vec(1);
    vec[0].e_theta_re = a.e_theta_re, vec[0].e_theta_im = a.e_theta_im, vec[0].e_phi_re = a.e_phi_re, vec[0].e_phi_im = a.e_phi_im;
    vec[0].azimuth_grid = a.azimuth_grid, vec[0].elevation_grid = a.elevation_grid;
    REQUIRE(unmarked(vec[0]));
    quadriga_lib::join_arrays(vec, vec);
    CHECK(!qd_arrayant_cache_matches(vec[0]));

    qd_arrayant_clear_mark(vec[0]); // Each call starts from an unmarked input
    const std::string fn = "test_arrayant_is_valid_internal.qdant";
    quadriga_lib::qdant_write(fn, vec);
    std::remove(fn.c_str());
    CHECK(!qd_arrayant_cache_matches(vec[0]));

    arma::mat fbs(3, 1), pg(1, 1, arma::fill::ones);
    fbs.col(0) = arma::vec({5.0, 1.0, 1.0});
    const arma::mat lbs = fbs;
    arma::vec plen = {12.0}, freq = {299792458.0};
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0, M(6, 0, 0) = -1.0;
    std::vector<arma::cube> cr, ci, dl;
    qd_arrayant_clear_mark(vec[0]); // Each call starts from an unmarked input
    quadriga_lib::get_channels_multifreq<double>(vec, vec, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {10.0, 0.0, 1.0}, {0.0, 0.0, 0.0},
                                                 fbs, lbs, pg, plen, M, freq, freq, cr, ci, dl);
    REQUIRE(cr.size() == 1);
    CHECK(!qd_arrayant_cache_matches(vec[0]));
}

// The generators return marked antennas, so the computing functions take the quick path
TEST_CASE("Arrayant quick check - Generated antennas are marked")
{
    const arma::mat beams = {{0.0, 20.0, 40.0}, {0.0, 10.0, 0.0}, {1.0, 1.0, 1.0}}; // az; el; weight
    for (arma::uword pol : {1ULL, 2ULL, 3ULL})
        for (arma::uword M : {1ULL, 2ULL, 4ULL})
            for (bool sep : {false, true})
            {
                auto a = quadriga_lib::generate_antenna<double>("multibeam", 10.0, 3.0e9, {}, {M, 2}, {}, {pol}, {}, {}, beams, {}, {}, {}, sep);
                CHECK(qd_arrayant_cache_matches(a));
            }
    CHECK(qd_arrayant_cache_matches(quadriga_lib::generate_antenna<double>("omni", 10.0)));
    CHECK(qd_arrayant_cache_matches(quadriga_lib::generate_antenna<double>("3GPP", 10.0, 3.0e9, {}, {2, 2}, {}, {1})));
}
