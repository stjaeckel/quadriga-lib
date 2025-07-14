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