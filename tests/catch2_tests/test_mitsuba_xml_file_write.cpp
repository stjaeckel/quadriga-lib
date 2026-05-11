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
        nullptr, nullptr,
        &vert_list, &face_ind, &obj_ind, &mtl_ind,
        &obj_names, &mtl_names, &bsdf);

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