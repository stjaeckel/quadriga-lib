// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include "quadriga_lib.hpp"

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

// Byte-exact output of the writer before the index map rewrite (B-056), for a scene with a split object, a vertex
// shared by two objects and faces whose vertex order differs from the ascending global order
TEST_CASE("Test Mitsuba XML File Write - XML and PLY output are byte-identical to the reference")
{
    arma::mat vert_list = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.5},
                           {2.0, 0.0, 1.0}, {2.5, 1.5, 1.0}, {3.0, 0.25, 2.0}};
    arma::umat face_ind = {{6, 2, 4}, {0, 1, 2}, {4, 5, 6}, {2, 1, 3}, {5, 3, 6}};
    arma::uvec obj_ind = {1, 0, 1, 0, 1};
    arma::uvec mtl_ind = {2, 1, 1, 1, 2};
    std::vector<std::string> obj_names = {"wall", "floor"}, mtl_names = {"concrete", "glass"};
    arma::mat bsdf(2, 17);
    for (arma::uword m = 0; m < 2; ++m)
        for (arma::uword c = 0; c < 17; ++c)
            bsdf(m, c) = 0.0625 * (double)((m * 17 + c) % 16);

    quadriga_lib::mitsuba_xml_file_write<double>("mref.xml", vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names, bsdf);

    auto slurp = [](const std::string &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };

    const std::string xml_expected =
        "<?xml version=\"1.0\"?>\n"
        "<scene version=\"3.0.0\">\n"
        "<bsdf type=\"twosided\" id=\"concrete\" name=\"concrete\">\n"
        "<bsdf type=\"principled\">\n"
        "<rgb name=\"base_color\" value=\"0.000000,0.062500,0.125000\" />\n"
        "<float name=\"metallic\" value=\"0.3125\" />\n"
        "<float name=\"roughness\" value=\"0.25\" />\n"
        "<float name=\"anisotropic\" value=\"0.875\" />\n"
        "<float name=\"sheen\" value=\"0.6875\" />\n"
        "<float name=\"clearcoat\" value=\"0.75\" />\n"
        "<float name=\"clearcoat_gloss\" value=\"0.1875\" />\n"
        "<float name=\"eta\" value=\"0.375\" />\n"
        "</bsdf>\n"
        "</bsdf>\n"
        "<bsdf type=\"twosided\" id=\"glass\" name=\"glass\">\n"
        "<bsdf type=\"principled\">\n"
        "<rgb name=\"base_color\" value=\"0.062500,0.125000,0.187500\" />\n"
        "<float name=\"metallic\" value=\"0.375\" />\n"
        "<float name=\"roughness\" value=\"0.3125\" />\n"
        "<float name=\"anisotropic\" value=\"0.9375\" />\n"
        "<float name=\"sheen\" value=\"0.75\" />\n"
        "<float name=\"clearcoat\" value=\"0.8125\" />\n"
        "<float name=\"clearcoat_gloss\" value=\"0.125\" />\n"
        "<float name=\"eta\" value=\"0.4375\" />\n"
        "</bsdf>\n"
        "</bsdf>\n"
        "<shape type=\"ply\" id=\"wall\" name=\"wall\">\n"
        "<string name=\"filename\" value=\"mref_meshes/wall.ply\" />\n"
        "<boolean name=\"face_normals\" value=\"true\" />\n"
        "<ref id=\"concrete\" name=\"bsdf\" />\n"
        "</shape>\n"
        "<shape type=\"ply\" id=\"floor_concrete\" name=\"floor_concrete\">\n"
        "<string name=\"filename\" value=\"mref_meshes/floor_concrete.ply\" />\n"
        "<boolean name=\"face_normals\" value=\"true\" />\n"
        "<ref id=\"concrete\" name=\"bsdf\" />\n"
        "</shape>\n"
        "<shape type=\"ply\" id=\"floor_glass\" name=\"floor_glass\">\n"
        "<string name=\"filename\" value=\"mref_meshes/floor_glass.ply\" />\n"
        "<boolean name=\"face_normals\" value=\"true\" />\n"
        "<ref id=\"glass\" name=\"bsdf\" />\n"
        "</shape>\n"
        "</scene>\n";
    CHECK(slurp("mref.xml") == xml_expected);

    auto header = [](unsigned n_vert, unsigned n_face)
    {
        return "ply\nformat binary_little_endian 1.0\ncomment Created by Quadriga-Lib " + quadriga_lib::quadriga_lib_version() +
               "\nelement vertex " + std::to_string(n_vert) + "\nproperty float x\nproperty float y\nproperty float z\n" +
               "element face " + std::to_string(n_face) + "\nproperty list uchar uint vertex_indices\nend_header\n";
    };

    {
        const unsigned char payload[] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f,
            0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x02, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        };
        CHECK(slurp("mref_meshes/wall.ply") == header(4, 2) + std::string((const char *)payload, sizeof(payload)));
    }

    {
        const unsigned char payload[] = {
            0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0x40,
            0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x3e,
            0x00, 0x00, 0x00, 0x40, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
            0x00,
        };
        CHECK(slurp("mref_meshes/floor_concrete.ply") == header(3, 1) + std::string((const char *)payload, sizeof(payload)));
    }

    {
        const unsigned char payload[] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x40, 0x03, 0x04, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        };
        CHECK(slurp("mref_meshes/floor_glass.ply") == header(5, 2) + std::string((const char *)payload, sizeof(payload)));
    }

    std::error_code ec;
    std::filesystem::remove_all("mref_meshes", ec);
    std::filesystem::remove("mref.xml", ec);
}

TEST_CASE("Test Mitsuba XML File Write - PLY paths are relative to the XML file")
{
    // Mitsuba resolves a relative PLY path against the directory of the XML file; an XML in another directory, given
    // by a relative or an absolute path, must refer to "<stem>_meshes/<object>.ply" and not to the path of the writer
    arma::mat vert_list = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    arma::umat face_ind = {{0, 1, 2}};
    arma::uvec obj_ind = {0}, mtl_ind = {1};
    std::vector<std::string> obj_names = {"tri"}, mtl_names = {"concrete"};

    const std::filesystem::path dir = "mitsuba_rel_path_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directory(dir);

    auto slurp = [](const std::filesystem::path &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };

    for (const std::filesystem::path &xml : {dir / "scene.xml", std::filesystem::absolute(dir / "scene_abs.xml")})
    {
        quadriga_lib::mitsuba_xml_file_write<double>(xml.string(), vert_list, face_ind, obj_ind, mtl_ind, obj_names, mtl_names);
        const std::string stem = xml.stem().string();
        const std::string txt = slurp(xml);
        CHECK(txt.find("<string name=\"filename\" value=\"" + stem + "_meshes/tri.ply\" />") != std::string::npos);
        CHECK(txt.find(dir.string()) == std::string::npos); // no directory of the writer in the XML
        CHECK(std::filesystem::exists(xml.parent_path() / (stem + "_meshes") / "tri.ply"));
    }
    std::filesystem::remove_all(dir, ec);
}
