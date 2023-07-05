// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "quadriga_lib.hpp"

#include <iostream>
#include <string>

TEST_CASE("Generate Arrayant - Minimal test omni")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    CHECK(ant.name == "omni");
}

TEST_CASE("Generate Arrayant - Minimal test dipole")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<float>();
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 1.760964f) < 0.0001);
    REQUIRE_THROWS_AS(ant.calc_directivity_dBi(1), std::invalid_argument);
}

TEST_CASE("Generate Arrayant - Minimal test Half-wave dipole")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 2.15f) < 0.001);
}

TEST_CASE("Generate Arrayant - Custom")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);
    float directivity = ant.calc_directivity_dBi(0);
    CHECK(std::abs(directivity - 25.627f) < 0.001);
}
