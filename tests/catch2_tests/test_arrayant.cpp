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