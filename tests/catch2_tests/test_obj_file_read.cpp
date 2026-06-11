// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

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

// Map helpers: csv_prop maps column name -> per-material value vector (length n_csv).

// Read the property value for material index iM from column 'key'.
// Returns the per-column default when the column is absent (mirrors consumer behavior).
static inline double prop_at(const std::unordered_map<std::string, std::vector<double>> &p,
                             const std::string &key, arma::uword iM, double def)
{
    auto it = p.find(key);
    if (it == p.end() || it->second.empty())
        return def;
    return it->second[iM];
}

// Check the standard EM columns for material index iM against an expected
// {a,b,c,d,att,attB,alpha,alphaB,fRef} row, applying the documented defaults
// (a=1, fRef=1, everything else 0) for any column not present in the map.
static inline bool em_row_matches(const std::unordered_map<std::string, std::vector<double>> &p,
                                  arma::uword iM, const std::array<double, 9> &e, double tol = 1e-14)
{
    const std::array<std::pair<const char *, double>, 9> cols = {{{"a", 1.0}, {"b", 0.0}, {"c", 0.0}, {"d", 0.0}, {"att", 0.0}, {"attB", 0.0}, {"alpha", 0.0}, {"alphaB", 0.0}, {"fRef", 1.0}}};
    for (size_t k = 0; k < 9; ++k)
        if (std::abs(prop_at(p, cols[k].first, iM, cols[k].second) - e[k]) > tol)
            return false;
    return true;
}

TEST_CASE("Test OBJ File Read - Simple test")
{
    REQUIRE(my_fancy_cube("cube.obj"));

    // Verify number of faces
    auto n_faces = quadriga_lib::obj_file_read<double>("cube.obj");
    CHECK(n_faces == 12ULL);

    // Containers for data
    arma::mat mesh, vert_list;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind, csv_ind;
    std::vector<std::string> obj_names, mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Read file (geometry + .mtl side + csv side from the default table)
    quadriga_lib::obj_file_read<double>("cube.obj", &mesh, &vert_list, &face_ind, &obj_ind, &obj_names,
                                        &mtl_ind, &mtl_names, nullptr,
                                        "", &csv_ind, &csv_names, &csv_prop);

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

    // Geometry checks
    CHECK(mesh.n_rows == 12);
    CHECK(mesh.n_cols == 9);

    CHECK(vert_list.n_rows == 8);
    CHECK(vert_list.n_cols == 3);

    CHECK(face_ind.n_rows == 12);
    CHECK(face_ind.n_cols == 3);

    CHECK(obj_ind.n_rows == 12);
    CHECK(mtl_ind.n_rows == 12);
    CHECK(csv_ind.n_rows == 12);

    CHECK(obj_names.size() == 1);

    // No usemtl in the file -> .mtl side empty (no material assigned), csv side is the full default table
    CHECK(mtl_names.size() == 0);

    CHECK(csv_names.size() > 1);     // full default table returned
    CHECK(csv_names[0] == "air");    // air is the first table entry
    CHECK(csv_prop.count("a") == 1); // table has at least the 'a' column

    // Geometry equality
    CHECK(arma::approx_equal(vert_list, vert_list_correct, "absdiff", 1e-14));
    CHECK(arma::approx_equal(face_ind, face_ind_correct_u32, "absdiff", 1e-14));
    CHECK(arma::approx_equal(mesh, mesh_correct, "absdiff", 1e-14));

    // No materials referenced -> all faces are index 0 (no material)
    CHECK(arma::all(obj_ind == 0U)); // single object, 0-based
    CHECK(arma::all(mtl_ind == 0U));
    CHECK(arma::all(csv_ind == 0U));
    CHECK(obj_names[0] == "Cube");

    // Air at csv row 0 is transparent
    CHECK(em_row_matches(csv_prop, 0, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Read - Default-table materials")
{
    arma::uvec csv_ind;
    std::vector<std::string> mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Single material from the built-in table
    REQUIRE(my_fancy_cube("cube.obj", "air"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &mtl_names, nullptr,
                                        "", &csv_ind, &csv_names, &csv_prop);

    // .mtl side records the raw usemtl name
    CHECK(mtl_names.size() == 1);
    CHECK(mtl_names[0] == "air");
    // csv side resolves every face to the 'air' row of the table
    CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));
    CHECK(arma::all(csv_ind == csv_ind(0)));

    // Two distinct materials, first four faces vs. the rest
    REQUIRE(my_fancy_cube("cube.obj", "itu_concrete", "itu_wood"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &mtl_names, nullptr,
                                        "", &csv_ind, &csv_names, &csv_prop);

    CHECK(mtl_names.size() == 2);
    CHECK(mtl_names[0] == "itu_concrete");
    CHECK(mtl_names[1] == "itu_wood");

    CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0}));
    CHECK(em_row_matches(csv_prop, csv_ind(4) - 1, {1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0}));
    CHECK(csv_ind(0) != csv_ind(4)); // different table rows

    // Blender duplicate suffix ".NNN" is stripped for the csv lookup,
    // but the raw name is preserved in mtl_names (for the .mtl/bsdf match)
    REQUIRE(my_fancy_cube("cube.obj", "itu_brick.001", "itu_metal.shiny.001"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &mtl_names, nullptr,
                                        "", &csv_ind, &csv_names, &csv_prop);

    CHECK(mtl_names[0] == "itu_brick.001");
    CHECK(mtl_names[1] == "itu_metal.shiny.001"); // everything after the first dot is stripped

    CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {3.91, 0.0, 0.0238, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0}));
    CHECK(em_row_matches(csv_prop, csv_ind(4) - 1, {1.0, 0.0, 1.0e7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Read - Unknown materials and strict flag")
{
    arma::uvec csv_ind;
    std::vector<std::string> mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Material not in the table, non-strict -> index 0 (no material)
    REQUIRE(my_fancy_cube("cube.obj", "not_a_real_material"));
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &mtl_names, nullptr,
                                        "", &csv_ind, &csv_names, &csv_prop, false);

    CHECK(mtl_names[0] == "not_a_real_material"); // raw name still recorded on the .mtl side
    CHECK(csv_ind(0) == 0U);                      // unmatched, non-strict -> 0 (no material)

    // Same scene, strict -> throws because the material is absent from the table
    REQUIRE(my_fancy_cube("cube.obj", "not_a_real_material"));
    CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                        nullptr, &mtl_names, nullptr,
                                                        "", &csv_ind, &csv_names, &csv_prop, true),
                    std::invalid_argument);

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Read - Table-only (empty fn_obj)")
{
    arma::uvec csv_ind;
    std::vector<std::string> csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Empty fn_obj + empty fn_csv -> full built-in default table, no geometry
    arma::uword n = quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                        nullptr, nullptr, nullptr,
                                                        "", &csv_ind, &csv_names, &csv_prop);
    CHECK(n == 0ULL);
    CHECK(csv_ind.is_empty());   // no faces
    CHECK(csv_names.size() > 1); // full table
    CHECK(csv_names[0] == "air");
    CHECK(em_row_matches(csv_prop, 0, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));

    // Every present column has the same length as csv_names
    for (const auto &kv : csv_prop)
        CHECK(kv.second.size() == csv_names.size());
}

TEST_CASE("Test OBJ File Read - Custom material CSV")
{
    arma::uvec csv_ind;
    std::vector<std::string> mtl_names, csv_names;
    std::unordered_map<std::string, std::vector<double>> csv_prop;

    // Test 1: Basic custom materials
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att\n";
        csv_file << "air,1.0,0.0,0.0,0.0,0.0\n"; // air at table row 0
        csv_file << "custom_material_1,2.5,0.0,0.001,0.5,5.0\n";
        csv_file << "custom_material_2,4.0,-0.1,0.05,1.2,10.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1", "custom_material_2"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0}));
        CHECK(mtl_names[0] == "custom_material_1");

        CHECK(em_row_matches(csv_prop, csv_ind(4) - 1, {4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0}));
        CHECK(mtl_names[1] == "custom_material_2");

        CHECK(csv_ind(0) != csv_ind(4));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 2: Jumbled column order (key-by-name, so order is irrelevant)
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "att,d,c,b,a,name\n";
        csv_file << "0.0,0.0,0.0,0.0,1.0,air\n";
        csv_file << "5.0,0.5,0.001,0.0,2.5,custom_material_1\n";
        csv_file << "10.0,1.2,0.05,-0.1,4.0,custom_material_2\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1", "custom_material_2"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0}));
        CHECK(em_row_matches(csv_prop, csv_ind(4) - 1, {4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0}));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 3: Missing 'name' column -> error (name is the mandatory join key)
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "a,b,c,d,att\n";
        csv_file << "2.5,0.0,0.001,0.5,5.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 4: CSV with only name+a; all other columns absent -> consumer defaults
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a\n";
        csv_file << "air,1.0\n";
        csv_file << "custom_material_1,2.5\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        // Only 'a' is present in the map; b..alphaB default to 0, fRef defaults to 1
        CHECK(csv_prop.count("a") == 1);
        CHECK(csv_prop.count("c") == 0);
        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 5: Duplicate material names -> error
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att\n";
        csv_file << "custom_material_1,2.5,0.0,0.001,0.5,5.0\n";
        csv_file << "custom_material_1,4.0,-0.1,0.05,1.2,10.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, &mtl_names, nullptr,
                                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop),
                        std::invalid_argument);

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 6: Non-existent CSV file -> error
    {
        REQUIRE(my_fancy_cube("cube.obj", "custom_material_1"));

        CHECK_THROWS_AS(quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            nullptr, &mtl_names, nullptr,
                                                            "nonexistent.csv", &csv_ind, &csv_names, &csv_prop),
                        std::invalid_argument);

        std::remove("cube.obj");
    }

    // Test 7: Frequency-dependent columns populated
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att,attB,alpha,fRef,alphaB\n";
        csv_file << "air,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0\n";
        csv_file << "lossy_wall,4.5,0.1,0.02,0.8,3.0,0.2,0.5,2.4,0.15\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "lossy_wall"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {4.5, 0.1, 0.02, 0.8, 3.0, 0.2, 0.5, 0.15, 2.4}));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 8: Subset of optional columns; unspecified ones take defaults
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        // fRef given but attB/alpha/alphaB absent -> consumer defaults
        csv_file << "name,a,c,fRef\n";
        csv_file << "air,1.0,0.0,1.0\n";
        csv_file << "partial,3.0,0.01,5.0\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "partial"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0}));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }

    // Test 9: Empty cells in optional columns parse as 0
    {
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());

        csv_file << "name,a,b,c,d,att,attB,alpha,alphaB,fRef\n";
        csv_file << "air,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0\n";
        csv_file << "sparse,2.0,,0.005,,1.5,,,,\n";
        csv_file.close();

        REQUIRE(my_fancy_cube("cube.obj", "sparse"));

        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, &mtl_names, nullptr,
                                            "custom_materials.csv", &csv_ind, &csv_names, &csv_prop);

        // Empty cells are 0; fRef cell is empty here so it parses as 0 (cell present but blank)
        CHECK(em_row_matches(csv_prop, csv_ind(0) - 1, {2.0, 0.0, 0.005, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0}));

        std::remove("cube.obj");
        std::remove("custom_materials.csv");
    }
}