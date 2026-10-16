// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>

TEST_CASE("Arrayant append")
{
    auto ant1 = quadriga_lib::generate_arrayant_dipole<float>();
    auto ant2 = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);

    ant1.coupling_re.at(0) = 2.0;
    ant2.coupling_re.at(0) = 3.0;

    auto ant = ant1.append(&ant2);

    arma::fmat T;
    T = {{2.0, 0.0}, {0.0, 3.0}};
    CHECK(arma::approx_equal(ant.coupling_re, T, "absdiff", 1.0e-6f));

    float g0 = ant.calc_directivity_dBi(0);
    float g1 = ant.calc_directivity_dBi(1);

    CHECK(std::abs(g0 - 1.760964f) < 0.0001f);
    CHECK(std::abs(g1 - 25.627f) < 0.001f);
}

// B-043: the default constructor must initialise check_ptr. The arrayant is created twice in the same stack slot;
// small matrices live inside the Armadillo objects, so stale check_ptr values of the first object would match the
// data pointers of the second one and let the quick is_valid() pass an invalid, never validated object.
static __attribute__((noinline)) std::string arrayant_check_ptr_helper(bool first)
{
    quadriga_lib::arrayant<double> a;
    a.e_theta_re.ones(2, 2, 1), a.e_theta_im.zeros(2, 2, 1), a.e_phi_re.zeros(2, 2, 1), a.e_phi_im.zeros(2, 2, 1);
    a.elevation_grid = {-0.5, 0.5};
    a.element_pos.zeros(3, 1), a.coupling_re.ones(1, 1), a.coupling_im.zeros(1, 1);
    if (first)
    {
        a.azimuth_grid = {-1.0, 1.0};
        return a.validate();
    }
    a.azimuth_grid = {1.0, -1.0}; // invalid: unsorted; never validated
    return a.is_valid();
}

TEST_CASE("Arrayant - Default constructor initialises check_ptr")
{
    quadriga_lib::arrayant<double> a;
    for (auto p : a.check_ptr)
        CHECK(p == nullptr);

    CHECK(arrayant_check_ptr_helper(true).empty());
    CHECK(!arrayant_check_ptr_helper(false).empty());
}
