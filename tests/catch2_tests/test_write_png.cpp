// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

TEST_CASE("Write PNG")
{
    const arma::uword N = 500;  // size along one edge
    const double maxVal = 10.0; // value at the upper‑right corner
    arma::mat gradient(N, N, arma::fill::none);
    const double scale = maxVal / double(N); // (= 10 / 98 for N = 50)

    for (arma::uword r = 0; r < N; ++r)
        for (arma::uword c = 0; c < N; ++c)
            gradient(r, c) = scale * c;

    quadriga_lib::write_png(gradient, "test.png");

    REQUIRE(std::filesystem::exists("test.png"));

    std::remove("test.png");
}