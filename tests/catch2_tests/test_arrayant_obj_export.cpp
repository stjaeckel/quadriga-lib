// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_lib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

TEST_CASE("Arrayant OBJ Export")
{
    auto ant = quadriga_lib::generate_arrayant_3GPP<float>(1, 2);
    CHECK(ant.name == "3gpp");
    ant.name = "g_pp";

    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obx"), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", 0.0), std::invalid_argument);

    // Non-finite range or radius would give NaN or Inf vertex coordinates
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", inf), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", nan), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", 30.0, "jet", nan), std::invalid_argument);
    REQUIRE_THROWS_AS(ant.export_obj_file("testantenna.obj", 30.0, "jet", inf), std::invalid_argument);

    // Non-finite field values
    for (float v : {nan, inf})
    {
        auto b = ant.copy();
        b.e_phi_re(b.n_elevation() / 2, b.n_azimuth() / 2, 0) = v; // A grid point the export samples
        REQUIRE_THROWS_AS(b.export_obj_file("testantenna.obj"), std::invalid_argument);
    }

    ant.export_obj_file("testantenna.obj", 30.0, "jet", 0.4, 4);

    // An all-zero pattern has no shape to draw
    auto z = quadriga_lib::generate_arrayant_omni<float>(10.0f);
    z.e_theta_re.zeros();
    REQUIRE_THROWS_AS(z.export_obj_file("testantenna.obj"), std::invalid_argument);

    std::remove("testantenna.obj");
    std::remove("testantenna.mtl");
}

// A subnormal directivity range and a pattern whose peak power is subnormal used to overflow the scale factors to Inf
TEST_CASE("Arrayant OBJ Export - tiny range and tiny pattern give finite vertices")
{
    auto vertices_finite = [](const std::string &fn)
    {
        std::ifstream f(fn);
        REQUIRE(f.is_open());
        size_t n = 0;
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("v ", 0) == 0)
            {
                double x = 0.0, y = 0.0, z = 0.0;
                REQUIRE(std::sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z) == 3);
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                    return false;
                ++n;
            }
        return n > 0;
    };

    auto ant = quadriga_lib::generate_arrayant_dipole<double>(10.0);
    ant.export_obj_file("testantenna_tiny.obj", std::numeric_limits<double>::denorm_min(), "jet", 1.0, 2);
    CHECK(vertices_finite("testantenna_tiny.obj"));

    ant.e_theta_re *= 1.0e-160, ant.e_theta_im *= 1.0e-160, ant.e_phi_re *= 1.0e-160, ant.e_phi_im *= 1.0e-160;
    ant.export_obj_file("testantenna_tiny.obj", 30.0, "jet", 1.0, 2);
    CHECK(vertices_finite("testantenna_tiny.obj"));
    std::remove("testantenna_tiny.obj"), std::remove("testantenna_tiny.mtl");
}

TEST_CASE("Arrayant OBJ Export - every vertex is written once per element")
{
    // Vertex 0 of an element used to be written again for every face that uses it
    auto ant = quadriga_lib::generate_arrayant_omni<double>();
    ant.copy_element(0, 1); // two elements
    ant.element_pos(0, 1) = 2.0;
    ant.export_obj_file("testantenna_unique.obj", 30.0, "jet", 0.4, 3);

    std::ifstream f("testantenna_unique.obj");
    REQUIRE(f.is_open());
    std::vector<std::vector<std::string>> verts; // per object
    std::vector<size_t> max_index;               // largest face index per object
    size_t n_vert_total = 0, n_faces = 0;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("o ", 0) == 0)
            verts.emplace_back(), max_index.push_back(0);
        else if (line.rfind("v ", 0) == 0)
        {
            REQUIRE(!verts.empty());
            verts.back().push_back(line);
            ++n_vert_total;
        }
        else if (line.rfind("f ", 0) == 0)
        {
            REQUIRE(!verts.empty());
            ++n_faces;
            size_t a = 0, b = 0, c = 0;
            REQUIRE(std::sscanf(line.c_str(), "f %zu %zu %zu", &a, &b, &c) == 3);
            size_t m = std::max(a, std::max(b, c));
            max_index.back() = std::max(max_index.back(), m);
        }
    }
    f.close();
    REQUIRE(verts.size() == 2);
    CHECK(n_faces > 0);
    size_t n_before = 0;
    for (size_t o = 0; o < verts.size(); ++o)
    {
        std::set<std::string> uniq(verts[o].begin(), verts[o].end());
        CHECK(uniq.size() == verts[o].size()); // no vertex written twice
        CHECK(max_index[o] == n_before + verts[o].size()); // every written vertex is used by a face
        n_before += verts[o].size();
    }
    CHECK(n_before == n_vert_total);

    std::remove("testantenna_unique.obj");
    std::remove("testantenna_unique.mtl");
}
