// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Quick validity check of the library functions (qd_arrayant_is_valid_quick): passes at once when the cache of the
// arrayant (check_ptr) matches the data pointers and the member sizes fit together, else runs the full check

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"
#include "qd_arrayant_functions.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    a.element_pos.zeros(3, 1), a.coupling_re.ones(1, 1), a.coupling_im.zeros(1, 1);
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
    for (const auto &p : a.check_ptr)
        CHECK(p == nullptr);

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
        REQUIRE(a.validate().empty());
        a.e_theta_re.ones(1, 1, 64), a.e_theta_im.zeros(1, 1, 64), a.e_phi_re.zeros(1, 1, 64), a.e_phi_im.zeros(1, 1, 64);
    }

    SECTION("Heap storage, same number of values")
    {
        a.e_theta_re.ones(2, 2, 20), a.e_theta_im.zeros(2, 2, 20), a.e_phi_re.zeros(2, 2, 20), a.e_phi_im.zeros(2, 2, 20);
        a.azimuth_grid = {-0.5, 0.5}, a.elevation_grid = {-0.5, 0.5};
        REQUIRE(a.validate().empty());
        a.e_theta_re.ones(2, 1, 40), a.e_theta_im.zeros(2, 1, 40), a.e_phi_re.zeros(2, 1, 40), a.e_phi_im.zeros(2, 1, 40);
    }

    // Pointers as if the resize had kept them all, so a pointer comparison alone cannot see the change
    a.check_ptr[0] = a.e_theta_re.memptr(), a.check_ptr[1] = a.e_theta_im.memptr();
    a.check_ptr[2] = a.e_phi_re.memptr(), a.check_ptr[3] = a.e_phi_im.memptr();
    REQUIRE(!a.is_valid().empty());

    REQUIRE(!qd_arrayant_is_valid_quick(a).empty()); // stops here if the quick check misses the change
    CHECK_THROWS_AS(quadriga_lib::arrayant_interpolate(a, az, el, vr, vi, hr, hi), std::invalid_argument);
}

// A value changed in place keeps the data pointers: the quick check passes, the public check finds the change.
// The quick check does not write to the object, a failed public check leaves the cache unchanged.
TEST_CASE("Arrayant quick check - Passes a marked object changed in place, the public check does not")
{
    auto a = quadriga_lib::generate_arrayant_omni<double>(30.0);
    a.azimuth_grid.memptr()[0] = a.azimuth_grid.memptr()[1]; // Not strictly ascending

    const std::string msg = "Values of 'azimuth_grid' must be sorted in strictly ascending order.";
    CHECK(qd_arrayant_is_valid_quick(a).empty());
    CHECK(a.is_valid() == msg);
    CHECK(qd_arrayant_is_valid_quick(a).empty());

    // A copy gets the full check of its source and is not marked
    quadriga_lib::arrayant<double> c = a;
    CHECK(c.check_ptr[0] == nullptr);

    std::vector<quadriga_lib::arrayant<double>> vec(2, quadriga_lib::generate_arrayant_omni<double>(30.0));
    for (auto &v : vec) // The same change in all entries, so that only the per-entry check can see it
        v.azimuth_grid.memptr()[0] = v.azimuth_grid.memptr()[1];
    CHECK(qd_arrayant_is_valid_multi_quick(vec).empty());
    CHECK(quadriga_lib::arrayant_is_valid_multi(vec) == "Entry 0: " + msg);

    // An unmarked object gets the full check
    auto b = quadriga_lib::generate_arrayant_omni<double>(30.0);
    b.check_ptr[0] = nullptr;
    b.azimuth_grid.memptr()[0] = b.azimuth_grid.memptr()[1];
    CHECK(qd_arrayant_is_valid_quick(b) == msg);
    CHECK(b.check_ptr[0] == nullptr);
}

// A passing is_valid marks the object (cache check_ptr holds its data pointers), a failing one does not
TEST_CASE("Arrayant quick check - is_valid marks a valid object")
{
    quadriga_lib::arrayant<double> b;
    b.e_theta_re.ones(2, 3, 1), b.e_theta_im.zeros(2, 3, 1), b.e_phi_re.zeros(2, 3, 1), b.e_phi_im.zeros(2, 3, 1);
    b.azimuth_grid = {-1.0, 0.0, 1.0}, b.elevation_grid = {-0.5, 0.5};
    b.element_pos.zeros(3, 1), b.coupling_re.ones(1, 1), b.coupling_im.zeros(1, 1);
    REQUIRE(b.check_ptr[0] == nullptr);
    CHECK(b.is_valid().empty());
    CHECK(qd_arrayant_cache_matches(b));

    b.check_ptr[0] = nullptr;
    b.azimuth_grid = {1.0, 0.0, -1.0};
    CHECK(!b.is_valid().empty());
    CHECK(b.check_ptr[0] == nullptr);
}

// Copy and move construction and assignment mark the target with its own data pointers when the source is valid.
// Small members (up to 16 values) are stored inside the Armadillo objects, so a move copies them to new addresses.
TEST_CASE("Arrayant copy and move - The target of a valid source is marked with its own data")
{
    auto marked_own = [](const quadriga_lib::arrayant<double> &x)
    {
        const double *p[9] = {x.e_theta_re.memptr(), x.e_theta_im.memptr(), x.e_phi_re.memptr(), x.e_phi_im.memptr(),
                              x.azimuth_grid.memptr(), x.elevation_grid.memptr(), x.element_pos.memptr(),
                              x.coupling_re.memptr(), x.coupling_im.memptr()};
        for (int i = 0; i < 9; ++i)
            if (x.check_ptr[i] != p[i])
                return false;
        return true;
    };
    auto unmarked = [](const quadriga_lib::arrayant<double> &x)
    {
        for (const auto &p : x.check_ptr)
            if (p != nullptr)
                return false;
        return true;
    };

    auto big = quadriga_lib::generate_arrayant_xpol<double>(10.0); // Pattern on the heap
    quadriga_lib::arrayant<double> small;                         // All members inside the objects
    small.e_theta_re.ones(1, 2, 1), small.e_theta_im.zeros(1, 2, 1), small.e_phi_re.zeros(1, 2, 1), small.e_phi_im.zeros(1, 2, 1);
    small.azimuth_grid = {-1.0, 1.0}, small.elevation_grid = {0.0};
    REQUIRE(small.validate().empty());

    for (auto *src : {&big, &small})
    {
        REQUIRE(marked_own(*src));

        quadriga_lib::arrayant<double> a(*src);
        CHECK(marked_own(a));
        CHECK(a.check_ptr[0] != src->check_ptr[0]);

        quadriga_lib::arrayant<double> b;
        b = *src;
        CHECK(marked_own(b));

        quadriga_lib::arrayant<double> c(std::move(a));
        CHECK(marked_own(c));
        CHECK(unmarked(a));

        quadriga_lib::arrayant<double> d;
        d = std::move(b);
        CHECK(marked_own(d));
        CHECK(unmarked(b));
        CHECK(d.is_valid().empty());
    }

    // An invalid source gives an unmarked target; assignment drops an old mark of the target
    auto bad = small.copy();
    bad.e_theta_im.set_size(1, 3, 1);
    auto e = big.copy();
    e = bad;
    CHECK(unmarked(e));
    quadriga_lib::arrayant<double> f(std::move(bad));
    CHECK(unmarked(f));
    CHECK(!f.is_valid().empty());
}

// A copy of a valid arrayant is marked as validated with its own data: its cache holds the data
// pointers of the copy. The members of the pattern used here have more than 16 elements, so they live on the heap and
// keep their data pointers when the copy is returned. A copy of an invalid arrayant gets no cache.
TEST_CASE("Arrayant copy - A copy of a valid arrayant is marked as validated")
{
    // 6 elements on a 10 degree grid: grids of 19 and 37 values, positions [3, 6], coupling [6, 6]
    auto a = quadriga_lib::generate_arrayant_3GPP<double>(1, 3, 3.0e9, 3, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 10.0);
    REQUIRE(a.validate().empty());
    REQUIRE(a.element_pos.n_elem > 16);
    REQUIRE(a.coupling_im.n_elem > 16);

    const auto b = a.copy();
    const arma::Cube<double> *pattern[4] = {&b.e_theta_re, &b.e_theta_im, &b.e_phi_re, &b.e_phi_im};
    for (int i = 0; i < 4; ++i)
        CHECK(b.check_ptr[i] == pattern[i]->memptr());
    CHECK(b.check_ptr[4] == b.azimuth_grid.memptr());
    CHECK(b.check_ptr[5] == b.elevation_grid.memptr());
    CHECK(b.check_ptr[6] == b.element_pos.memptr());
    CHECK(b.check_ptr[7] == b.coupling_re.memptr());
    CHECK(b.check_ptr[8] == b.coupling_im.memptr());
    CHECK(b.check_ptr[0] != a.check_ptr[0]); // Own data, not the cache of the source
    CHECK(b.is_valid().empty());

    // The copy of an invalid arrayant has an empty cache and fails the check
    auto c = a.copy();
    c.e_theta_im.set_size(19, 36, 6);
    const auto d = c.copy();
    for (int i = 0; i < 9; ++i)
        CHECK(d.check_ptr[i] == nullptr);
    CHECK(!d.is_valid().empty());
}

// Library functions check their inputs read only (threads may share an input): an unmarked valid input stays unmarked
TEST_CASE("Arrayant quick check - Library functions do not mark their inputs")
{
    quadriga_lib::arrayant<double> a;
    a.e_theta_re.ones(3, 4, 1), a.e_theta_im.zeros(3, 4, 1), a.e_phi_re.zeros(3, 4, 1), a.e_phi_im.zeros(3, 4, 1);
    a.azimuth_grid = {-3.0, -1.0, 1.0, 3.0}, a.elevation_grid = {-1.0, 0.0, 1.0};
    a.element_pos.zeros(3, 1), a.coupling_re.ones(1, 1), a.coupling_im.zeros(1, 1);
    REQUIRE(a.check_ptr[0] == nullptr);

    a.combine_pattern();
    arma::mat az = {0.1}, el = {0.2}, vr, vi, hr, hi;
    quadriga_lib::arrayant_interpolate(a, az, el, vr, vi, hr, hi);
    CHECK(a.check_ptr[0] == nullptr);

    std::vector<quadriga_lib::arrayant<double>> vec(1);
    vec[0].e_theta_re = a.e_theta_re, vec[0].e_theta_im = a.e_theta_im, vec[0].e_phi_re = a.e_phi_re, vec[0].e_phi_im = a.e_phi_im;
    vec[0].azimuth_grid = a.azimuth_grid, vec[0].elevation_grid = a.elevation_grid;
    REQUIRE(vec[0].check_ptr[0] == nullptr);
    quadriga_lib::arrayant_concat_multi(vec, vec);
    CHECK(vec[0].check_ptr[0] == nullptr);

    for (auto &p : vec[0].check_ptr) // Each call starts from an unmarked input
        p = nullptr;
    const std::string fn = "test_arrayant_is_valid_internal.qdant";
    quadriga_lib::qdant_write_multi(fn, vec);
    std::remove(fn.c_str());
    CHECK(vec[0].check_ptr[0] == nullptr);

    arma::mat fbs(3, 1), pg(1, 1, arma::fill::ones);
    fbs.col(0) = arma::vec({5.0, 1.0, 1.0});
    const arma::mat lbs = fbs;
    arma::vec plen = {12.0}, freq = {299792458.0};
    arma::cube M(8, 1, 1, arma::fill::zeros);
    M(0, 0, 0) = 1.0, M(6, 0, 0) = -1.0;
    std::vector<arma::cube> cr, ci, dl;
    for (auto &p : vec[0].check_ptr) // Each call starts from an unmarked input
        p = nullptr;
    quadriga_lib::get_channels_multifreq<double>(vec, vec, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                 fbs, lbs, pg, plen, M, freq, freq, cr, ci, dl);
    REQUIRE(cr.size() == 1);
    CHECK(vec[0].check_ptr[0] == nullptr);
}

// The generators return marked antennas, so the computing functions take the quick path
TEST_CASE("Arrayant quick check - Generated antennas are marked")
{
    arma::vec az = {0.0, 20.0, 40.0}, el = {0.0, 10.0, 0.0}, w = {1.0, 1.0, 1.0};
    for (unsigned pol : {1u, 2u, 3u})
        for (arma::uword M : {1ULL, 2ULL, 4ULL})
            for (bool sep : {false, true})
            {
                auto a = quadriga_lib::generate_arrayant_multibeam<double>(M, 2, az, el, w, 3.0e9, pol, 0.5, 120.0, 120.0, 0.0, 10.0, sep, false);
                CHECK(qd_arrayant_cache_matches(a));
            }
    CHECK(qd_arrayant_cache_matches(quadriga_lib::generate_arrayant_omni<double>(10.0)));
    CHECK(qd_arrayant_cache_matches(quadriga_lib::generate_arrayant_3GPP<double>(2, 2, 3.0e9, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, nullptr, 10.0)));
}
