// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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