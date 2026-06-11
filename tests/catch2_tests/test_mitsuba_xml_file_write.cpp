// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include "quadriga_tools.hpp"

TEST_CASE("Test Mitsuba XML File Write")
{
    // ---------------------------------------------------------------------
    // 1. Read reference OBJ
    // ---------------------------------------------------------------------
    arma::mat vert_list, bsdf;
    arma::umat face_ind;
    arma::uvec obj_ind, mtl_ind;
    std::vector<std::string> obj_names, mtl_names;

    quadriga_lib::obj_file_read<double>(
        "tests/data/test_scene_pbr.obj",
        nullptr, // mesh (not needed)
        &vert_list, &face_ind, &obj_ind, &obj_names,
        &mtl_ind, &mtl_names, &bsdf);

    // ---------------------------------------------------------------------
    // 2. Write the Mitsuba XML scene
    // ---------------------------------------------------------------------
    const std::filesystem::path xml_file = "test_scene_x.xml";
    quadriga_lib::mitsuba_xml_file_write<double>(
        xml_file.string(),
        vert_list, face_ind, obj_ind, mtl_ind,
        obj_names, mtl_names,
        /*bsdf names*/ {}, /*map_to_itu_materials=*/true);

    const std::filesystem::path mesh_folder = xml_file.stem().string() + "_meshes";

    // ---------------------------------------------------------------------
    // 3. Assertions
    // ---------------------------------------------------------------------
    REQUIRE(std::filesystem::exists(xml_file));          // XML file created
    REQUIRE(std::filesystem::is_directory(mesh_folder)); // mesh folder exists
    REQUIRE_FALSE(std::filesystem::is_empty(mesh_folder));

    // ---------------------------------------------------------------------
    // 4. Clean-up (delete XML + folder)
    // ---------------------------------------------------------------------
    std::error_code ec; // swallow any I/O errors
    std::filesystem::remove_all(mesh_folder, ec);
    std::filesystem::remove(xml_file, ec);
}

TEST_CASE("Test Mitsuba XML File Write - 1-based material indices")
{
    // Minimal two-triangle mesh, single object, two materials.
    arma::mat vert_list = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
    arma::umat face_ind = {{0, 1, 2}, {1, 3, 2}};
    arma::uvec obj_ind = {0, 0};                  // single object, 0-based
    arma::uvec mtl_ind = {1, 2};                  // 1-based: materials 1 and 2 (no 0 = all faces assigned)
    std::vector<std::string> obj_names = {"obj"};
    std::vector<std::string> mtl_names = {"concrete", "wood"};

    const std::filesystem::path xml_file = "test_scene_1based.xml";
    const std::filesystem::path mesh_folder = xml_file.stem().string() + "_meshes";

    // Valid 1-based input writes successfully (object is split per material).
    quadriga_lib::mitsuba_xml_file_write<double>(
        xml_file.string(), vert_list, face_ind, obj_ind, mtl_ind,
        obj_names, mtl_names, /*bsdf*/ {}, /*map_to_itu_materials=*/false);

    CHECK(std::filesystem::exists(xml_file));
    CHECK(std::filesystem::is_directory(mesh_folder));

    std::error_code ec;
    std::filesystem::remove_all(mesh_folder, ec);
    std::filesystem::remove(xml_file, ec);

    // A face with no material (mtl_ind == 0) must be rejected.
    arma::uvec mtl_ind_no_material = {0, 1};
    CHECK_THROWS_AS(
        quadriga_lib::mitsuba_xml_file_write<double>(
            xml_file.string(), vert_list, face_ind, obj_ind, mtl_ind_no_material,
            obj_names, mtl_names, /*bsdf*/ {}, /*map_to_itu_materials=*/false),
        std::invalid_argument);
}