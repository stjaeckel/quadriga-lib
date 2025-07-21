// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

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