// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <iostream>
#include <fstream>
#include <string>

static inline bool my_fancy_cube(std::string fn, std::string mtl_name = "", std::string mtl_name2 = "")
{
    std::ofstream file(fn);

    if (file.is_open())
    {
        file << "# A very nice, but useless comment ;-)\n";
        file << "o Cube\n";
        file << "v 1.0 1.0 1.0\n";
        file << "v 1.0 1.0 -1.0\n";
        file << "v 1.0 -1.0 1.0\n";
        file << "v 1.0 -1.0 -1.0\n";
        file << "v -1.0 1.0 1.0\n";
        file << "v -1.0 1.0 -1.0\n";
        file << "v -1.0 -1.0 1.0\n";
        file << "v -1.0 -1.0 -1.0\n";
        file << "s 0\n";
        if (!mtl_name.empty())
            file << "usemtl " << mtl_name << "\n";
        file << "f 5 3 1\n";
        file << "f 3 8 4\n";
        file << "f 7 6 8\n";
        file << "f 2 8 6\n";
        if (!mtl_name2.empty())
            file << "usemtl " << mtl_name2 << "\n";
        file << "f 1 4 2\n";
        file << "f 5 2 6\n";
        file << "f 5 7 3\n";
        file << "f 3 7 8\n";
        file << "f 7 5 6\n";
        file << "f 2 4 8\n";
        file << "f 1 3 4\n";
        file << "f 5 1 2\n";
        file.close();
        return true;
    }
    return false;
}

TEST_CASE("Test OBJ File Read - Simple test")
{
    REQUIRE(my_fancy_cube("cube.obj"));

    // Verify number of faces
    auto n_faces = quadriga_lib::obj_file_read<double>("cube.obj");
    CHECK(n_faces == 12ULL);

    // Containers for data
    arma::mat mesh, mtl_prop, vert_list;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind;
    std::vector<std::string> obj_names, mtl_names;

    // Read file
    quadriga_lib::obj_file_read<double>("cube.obj", &mesh, &mtl_prop, &vert_list, &face_ind, &obj_ind, &mtl_ind, &obj_names, &mtl_names);

    // Expected values
    arma::mat vert_list_correct = {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, -1.0},
        {1.0, -1.0, 1.0},
        {1.0, -1.0, -1.0},
        {-1.0, 1.0, 1.0},
        {-1.0, 1.0, -1.0},
        {-1.0, -1.0, 1.0},
        {-1.0, -1.0, -1.0}};

    arma::umat face_ind_correct = {
        {5, 3, 1},
        {3, 8, 4},
        {7, 6, 8},
        {2, 8, 6},
        {1, 4, 2},
        {5, 2, 6},
        {5, 7, 3},
        {3, 7, 8},
        {7, 5, 6},
        {2, 4, 8},
        {1, 3, 4},
        {5, 1, 2}};

    face_ind_correct -= 1;
    auto face_ind_correct_u32 = arma::conv_to<arma::umat>::from(face_ind_correct);

    arma::mat mesh_correct = arma::join_rows(
        vert_list_correct.rows(face_ind_correct.col(0)),
        vert_list_correct.rows(face_ind_correct.col(1)),
        vert_list_correct.rows(face_ind_correct.col(2)));

    // Checks
    CHECK(mesh.n_rows == 12);
    CHECK(mesh.n_cols == 9);

    CHECK(mtl_prop.n_rows == 12);
    CHECK(mtl_prop.n_cols == 5);

    CHECK(vert_list.n_rows == 8);
    CHECK(vert_list.n_cols == 3);

    CHECK(face_ind.n_rows == 12);
    CHECK(face_ind.n_cols == 3);

    CHECK(obj_ind.n_rows == 12);
    CHECK(mtl_ind.n_rows == 12);

    CHECK(obj_names.size() == 1);
    CHECK(mtl_names.empty());

    CHECK(arma::all(mtl_prop.col(0) == 1.0));
    CHECK(arma::all(arma::all(mtl_prop.cols(1, 4) == 0.0)));

    CHECK(arma::approx_equal(vert_list, vert_list_correct, "absdiff", 1e-14));
    CHECK(arma::approx_equal(face_ind, face_ind_correct_u32, "absdiff", 1e-14));
    CHECK(arma::approx_equal(mesh, mesh_correct, "absdiff", 1e-14));

    CHECK(arma::all(obj_ind == 1U));
    CHECK(arma::all(mtl_ind == 0U));
    CHECK(obj_names[0] == "Cube");

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Read - Materials")
{
    arma::mat mtl_prop, mtl_correct;
    arma::uvec obj_ind, mtl_ind;
    std::vector<std::string> mtl_names;

    // Check "air"
    mtl_correct = {{1.0, 0.0, 0.0, 0.0, 0.0}};
    REQUIRE(my_fancy_cube("cube.obj", "air"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);
    CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[0] == "air");
    CHECK(arma::all(mtl_ind == 1U));

    // Check dual materials
    REQUIRE(my_fancy_cube("cube.obj", "itu_concrete", "itu_wood"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{5.24, 0.0, 0.0462, 0.7822, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[0] == "itu_concrete");
    CHECK(mtl_ind(0) == 1U);

    mtl_correct = {{1.99, 0.0, 0.0047, 1.0718, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[1] == "itu_wood");
    CHECK(mtl_ind(4) == 2U);

    // Materials can have a suffix separated by a dot
    REQUIRE(my_fancy_cube("cube.obj", "itu_brick.001", "itu_metal.shiny.001"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{3.91, 0.0, 0.0238, 0.16, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[0] == "itu_brick.001");

    mtl_correct = {{1.0, 0.0, 1.0e7, 0.0, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[1] == "itu_metal.shiny.001");

    // Custom materials
    REQUIRE(my_fancy_cube("cube.obj", "itu_brick.001::1.1:0.1:0.2:-3:20", "something_new::5"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{1.1, 0.1, 0.2, -3.0, 20.0}};
    CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[0] == "itu_brick.001::1.1:0.1:0.2:-3:20");
    CHECK(mtl_ind(0) == 1U);

    mtl_correct = {{5.0, 0.0, 0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[1] == "something_new::5");
    CHECK(mtl_ind(4) == 2U);

    // Custom materials
    REQUIRE(my_fancy_cube("cube.obj", "BLA::1.1:0.1:0.2:-3:20:.001", "something_new::5"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{1.1, 0.1, 0.2, -3.0, 20.0}};
    CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[0] == "BLA::1.1:0.1:0.2:-3:20:.001");

    mtl_correct = {{5.0, 0.0, 0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct, "absdiff", 1e-14));
    CHECK(mtl_names[1] == "something_new::5");

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Read - Custom Materials csv")
{
    arma::mat mtl_prop;
    arma::uvec mtl_ind;
    std::vector<std::string> mtl_names;

    // Test 1: Basic custom materials
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att\n";
        csv_file << "custom_material_1,2.5,0.0,0.001,0.5,5.0\n";
        csv_file << "custom_material_2,4.0,-0.1,0.05,1.2,10.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1", "custom_material_2"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct_1 = {{2.5, 0.0, 0.001, 0.5, 5.0}};
        CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct_1, "absdiff", 1e-14));
        CHECK(mtl_names[0] == "custom_material_1");
        CHECK(mtl_ind(0) == 1U);

        arma::mat mtl_correct_2 = {{4.0, -0.1, 0.05, 1.2, 10.0}};
        CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct_2, "absdiff", 1e-14));
        CHECK(mtl_names[1] == "custom_material_2");
        CHECK(mtl_ind(4) == 2U);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 2: Jumbled column order
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "att,d,c,b,a,name\n";
        csv_file << "5.0,0.5,0.001,0.0,2.5,custom_material_1\n";
        csv_file << "10.0,1.2,0.05,-0.1,4.0,custom_material_2\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1", "custom_material_2"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct_1 = {{2.5, 0.0, 0.001, 0.5, 5.0}};
        CHECK(arma::approx_equal(mtl_prop.row(0), mtl_correct_1, "absdiff", 1e-14));
        CHECK(mtl_names[0] == "custom_material_1");

        arma::mat mtl_correct_2 = {{4.0, -0.1, 0.05, 1.2, 10.0}};
        CHECK(arma::approx_equal(mtl_prop.row(4), mtl_correct_2, "absdiff", 1e-14));
        CHECK(mtl_names[1] == "custom_material_2");

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 3: Missing columns - missing 'att'
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d\n";
        csv_file << "custom_material_1,2.5,0.0,0.001,0.5\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv"),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 4: Missing columns - missing multiple columns
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b\n";
        csv_file << "custom_material_1,2.5,0.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv"),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 5: Duplicate material names
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att\n";
        csv_file << "custom_material_1,2.5,0.0,0.001,0.5,5.0\n";
        csv_file << "custom_material_1,4.0,-0.1,0.05,1.2,10.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv"),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 6: Non-existent CSV file
    {
        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                                            "nonexistent.csv"),
                        std::invalid_argument);

        std::remove("cube.obj");
    }
}
