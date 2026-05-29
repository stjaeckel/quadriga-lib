// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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

// Compare a cropped mtl_prop row against an expected 9-element row by trimming
// the expectation to the actual width of mtl_prop.
static inline bool row_matches(const arma::mat &mtl_prop, arma::uword row,
                               const arma::mat &expected_9, double tol = 1e-14)
{
    arma::uword w = mtl_prop.n_cols;
    return arma::approx_equal(mtl_prop.row(row), expected_9.cols(0, w - 1), "absdiff", tol);
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
    CHECK(mtl_prop.n_cols == 1); // vacuum scene → only column 0 (a) survives the crop

    CHECK(vert_list.n_rows == 8);
    CHECK(vert_list.n_cols == 3);

    CHECK(face_ind.n_rows == 12);
    CHECK(face_ind.n_cols == 3);

    CHECK(obj_ind.n_rows == 12);
    CHECK(mtl_ind.n_rows == 12);

    CHECK(obj_names.size() == 1);
    CHECK(mtl_names.empty());

    CHECK(arma::all(mtl_prop.col(0) == 1.0));
    // cols 1..15 cropped away (all at defaults) — consumers default them

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
    mtl_correct = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    REQUIRE(my_fancy_cube("cube.obj", "air"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);
    CHECK(row_matches(mtl_prop, 0, mtl_correct));
    CHECK(mtl_names[0] == "air");
    CHECK(arma::all(mtl_ind == 1U));

    // Check dual materials
    REQUIRE(my_fancy_cube("cube.obj", "itu_concrete", "itu_wood"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 0, mtl_correct));
    CHECK(mtl_names[0] == "itu_concrete");
    CHECK(mtl_ind(0) == 1U);

    mtl_correct = {{1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 4, mtl_correct));
    CHECK(mtl_names[1] == "itu_wood");
    CHECK(mtl_ind(4) == 2U);

    // Materials can have a suffix separated by a dot
    REQUIRE(my_fancy_cube("cube.obj", "itu_brick.001", "itu_metal.shiny.001"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{3.91, 0.0, 0.0238, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 0, mtl_correct));
    CHECK(mtl_names[0] == "itu_brick.001");

    mtl_correct = {{1.0, 0.0, 1.0e7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 4, mtl_correct));
    CHECK(mtl_names[1] == "itu_metal.shiny.001");

    // Custom materials
    REQUIRE(my_fancy_cube("cube.obj", "itu_brick.001::1.1:0.1:0.2:-3:20", "something_new::5"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{1.1, 0.1, 0.2, -3.0, 20.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 0, mtl_correct));
    CHECK(mtl_names[0] == "itu_brick.001::1.1:0.1:0.2:-3:20");
    CHECK(mtl_ind(0) == 1U);

    mtl_correct = {{5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 4, mtl_correct));
    CHECK(mtl_names[1] == "something_new::5");
    CHECK(mtl_ind(4) == 2U);

    // Custom materials
    REQUIRE(my_fancy_cube("cube.obj", "BLA::1.1:0.1:0.2:-3:20:.001", "something_new::5"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr, &obj_ind, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    CHECK(row_matches(mtl_prop, 4, mtl_correct));
    CHECK(mtl_names[1] == "something_new::5");

    REQUIRE(my_fancy_cube("cube.obj",
                          "full_inline::2.2:0.05:0.01:0.7:4.0:0.3:0.8:0.2:3.5",
                          "partial_inline::6.0:0:0:0:0:0:0.1")); // only alpha set; rest default
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                        nullptr, &mtl_ind, nullptr, &mtl_names);

    mtl_correct = {{2.2, 0.05, 0.01, 0.7, 4.0, 0.3, 0.8, 0.2, 3.5}};
    CHECK(row_matches(mtl_prop, 0, mtl_correct));

    mtl_correct = {{6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0}}; // fRef defaults to 1
    CHECK(row_matches(mtl_prop, 4, mtl_correct));

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

        arma::mat mtl_correct_1 = {{2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct_1));
        CHECK(mtl_names[0] == "custom_material_1");
        CHECK(mtl_ind(0) == 1U);

        arma::mat mtl_correct_2 = {{4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 4, mtl_correct_2));
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

        arma::mat mtl_correct_1 = {{2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct_1));
        CHECK(mtl_names[0] == "custom_material_1");

        arma::mat mtl_correct_2 = {{4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 4, mtl_correct_2));
        CHECK(mtl_names[1] == "custom_material_2");

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 3: Missing columns - missing 'att'
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "a,b,c,d,att\n";
        csv_file << "2.5,0.0,0.001,0.5,5.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv"),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 4: CSV with only name+a; all other fields default
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a\n";
        csv_file << "custom_material_1,2.5\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct = {{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct));

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

    // Test 7: CSV with new frequency-dependent columns populated
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att,attB,alpha,fRef,alphaB\n";
        csv_file << "lossy_wall,4.5,0.1,0.02,0.8,3.0,0.2,0.5,2.4,0.15\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "lossy_wall"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct = {{4.5, 0.1, 0.02, 0.8, 3.0, 0.2, 0.5, 0.15, 2.4}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 8: Subset of optional columns; unspecified ones take defaults
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        // fRef given but attB/alpha/alphaB absent -> defaults 0,0,0
        csv_file << "name,a,c,fRef\n";
        csv_file << "partial,3.0,0.01,5.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "partial"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct = {{3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 9: Empty cells in optional columns fall back to defaults
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att,attB,alpha,alphaB,fRef\n";
        csv_file << "sparse,2.0,,0.005,,1.5,,,,\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "sparse"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind, nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv");

        arma::mat mtl_correct = {{2.0, 0.0, 0.005, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 0, mtl_correct));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }
}
