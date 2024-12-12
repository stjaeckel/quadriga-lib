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

TEST_CASE("Arrayant OBJ Export")
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<float>(1, 2);
    CHECK(ant.name == "3gpp");
    ant.name = "g_pp";

    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obx"), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", 0.0), std::invalid_argument);

    ant.export_obj_file("testantenna.obj", 30.0, "jet", 0.4, 4);

    std::remove("testantenna.obj");
    std::remove("testantenna.mtl");
}