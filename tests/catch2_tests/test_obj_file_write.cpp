// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <string>
#include <vector>
#include <cstdio>
#include <filesystem>

// A unit cube: 8 vertices, 12 triangular faces (same geometry as the reader test)
static inline arma::mat cube_vertices()
{
    return arma::mat{
        {1.0, 1.0, 1.0},
        {1.0, 1.0, -1.0},
        {1.0, -1.0, 1.0},
        {1.0, -1.0, -1.0},
        {-1.0, 1.0, 1.0},
        {-1.0, 1.0, -1.0},
        {-1.0, -1.0, 1.0},
        {-1.0, -1.0, -1.0}};
}

// 0-based face indices into cube_vertices()
static inline arma::umat cube_faces()
{
    arma::umat f = {
        {5, 3, 1}, {3, 8, 4}, {7, 6, 8}, {2, 8, 6}, {1, 4, 2}, {5, 2, 6}, {5, 7, 3}, {3, 7, 8}, {7, 5, 6}, {2, 4, 8}, {1, 3, 4}, {5, 1, 2}};
    return f - 1;
}

// Assemble a [n_face, 9] mesh from a vertex list and 0-based face indices
static inline arma::mat make_mesh(const arma::mat &V, const arma::umat &F)
{
    return arma::join_rows(V.rows(F.col(0)), V.rows(F.col(1)), V.rows(F.col(2)));
}

// Compare a cropped mtl_prop row against an expected 9-element row by trimming
// the expectation to the actual width of mtl_prop.
static inline bool row_matches(const arma::mat &mtl_prop, arma::uword row,
                               const arma::mat &expected_9, double tol = 1e-14)
{
    arma::uword w = mtl_prop.n_cols;
    return arma::approx_equal(mtl_prop.row(row), expected_9.cols(0, w - 1), "absdiff", tol);
}

TEST_CASE("Test OBJ File Write - Mesh round-trip (geometry only)")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    // Write from mesh; no objects, no materials
    arma::mat vlo;
    arma::umat fio;
    quadriga_lib::obj_file_write<double>("cube.obj", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio);

    // The derived outputs should weld the cube down to 8 unique vertices
    CHECK(vlo.n_rows == 8);
    CHECK(vlo.n_cols == 3);
    CHECK(fio.n_rows == 12);
    CHECK(fio.n_cols == 3);
    CHECK(arma::approx_equal(make_mesh(vlo, fio), mesh, "absdiff", 1e-12));

    // No materials -> no .mtl file
    CHECK_FALSE(std::filesystem::exists("cube.mtl"));

    // Read back
    arma::mat mesh_rd, mtl_prop, vert_list_rd;
    arma::umat face_ind_rd;
    arma::uvec obj_ind_rd, mtl_ind_rd;
    std::vector<std::string> obj_names_rd, mtl_names_rd;

    auto n_faces = quadriga_lib::obj_file_read<double>("cube.obj", &mesh_rd, &mtl_prop, &vert_list_rd,
                                                       &face_ind_rd, &obj_ind_rd, &mtl_ind_rd,
                                                       &obj_names_rd, &mtl_names_rd);

    CHECK(n_faces == 12ULL);
    CHECK(vert_list_rd.n_rows == 8);
    CHECK(arma::approx_equal(mesh_rd, mesh, "absdiff", 1e-12));
    CHECK(arma::approx_equal(make_mesh(vert_list_rd, face_ind_rd), mesh, "absdiff", 1e-12));

    CHECK(obj_names_rd.size() == 1);
    CHECK(obj_names_rd[0] == "object");
    CHECK(mtl_names_rd.empty());
    CHECK(arma::all(obj_ind_rd == 1U));
    CHECK(arma::all(mtl_ind_rd == 0U));

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Write - vert_list / face_ind round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();

    // Write directly from vertex list + face indices (no mesh)
    arma::mat vlo;
    arma::umat fio;
    quadriga_lib::obj_file_write<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                         &vlo, &fio, &V, &F);

    // In this mode the outputs are exact copies of the inputs
    CHECK(arma::approx_equal(vlo, V, "absdiff", 1e-14));
    CHECK(fio.n_rows == F.n_rows);
    CHECK(fio.n_cols == F.n_cols);
    CHECK(arma::all(arma::vectorise(fio) == arma::vectorise(F)));

    // Read back and compare the reconstructed geometry
    arma::mat vert_list_rd;
    arma::umat face_ind_rd;
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, &vert_list_rd, &face_ind_rd);

    arma::mat mesh_expected = make_mesh(V, F);
    CHECK(arma::approx_equal(make_mesh(vert_list_rd, face_ind_rd), mesh_expected, "absdiff", 1e-12));

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Write - Materials round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::ones<arma::uvec>(12);
    std::vector<std::string> obj_names = {"Cube"};
    arma::mat vlo;
    arma::umat fio;

    SECTION("Named ITU materials")
    {
        arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
        mtl_ind.subvec(4, 11).fill(2); // faces 0-3 = concrete, 4-11 = wood
        std::vector<std::string> mtl_names = {"itu_concrete", "itu_wood"};

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names, &vlo, &fio);
        CHECK(std::filesystem::exists("cube.mtl"));

        arma::mat mtl_prop;
        arma::uvec mtl_ind_rd;
        std::vector<std::string> mtl_names_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind_rd, nullptr, &mtl_names_rd);

        REQUIRE(mtl_names_rd.size() == 2);
        CHECK(mtl_names_rd[0] == "itu_concrete");
        CHECK(mtl_names_rd[1] == "itu_wood");

        arma::mat concrete = {{5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::mat wood = {{1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0}};

        CHECK(row_matches(mtl_prop, 0, concrete));
        CHECK(row_matches(mtl_prop, 4, wood));

        CHECK(arma::all(mtl_ind_rd.subvec(0, 3) == 1U));
        CHECK(arma::all(mtl_ind_rd.subvec(4, 11) == 2U));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }

    SECTION("Custom inline material (:: syntax)")
    {
        arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
        std::vector<std::string> mtl_names = {"glass::6.0:0:0.1:1.2"};

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names, &vlo, &fio);

        arma::mat mtl_prop;
        arma::uvec mtl_ind_rd;
        std::vector<std::string> mtl_names_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                            nullptr, &mtl_ind_rd, nullptr, &mtl_names_rd);

        REQUIRE(mtl_names_rd.size() == 1);
        CHECK(mtl_names_rd[0] == "glass::6.0:0:0.1:1.2");

        arma::mat glass = {{6.0, 0.0, 0.1, 1.2, 0.0, 0.0, 0.0, 0.0, 1.0}};
        CHECK(row_matches(mtl_prop, 0, glass));
        CHECK(arma::all(mtl_ind_rd == 1U));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }
}

TEST_CASE("Test OBJ File Write - BSDF round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::ones<arma::uvec>(12);
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
    std::vector<std::string> obj_names = {"Cube"};
    std::vector<std::string> mtl_names = {"painted"};

    // Distinct, non-default values; clamped fields kept inside [0, 1], ior in a sane range
    arma::mat bsdf = {{0.1, 0.2, 0.3,    // base color RGB
                       0.7,              // transparency (d)
                       0.4,              // roughness (Pr)
                       0.6,              // metallic (Pm)
                       1.7,              // ior (Ni)
                       0.8,              // specular (Ks)
                       0.05, 0.15, 0.25, // emission RGB (Ke)
                       0.3,              // sheen (Ps)
                       0.35,             // clearcoat (Pc)
                       0.45,             // clearcoat roughness (Pcr)
                       0.55,             // anisotropic (aniso)
                       0.65,             // anisotropic rotation (anisor)
                       0.9}};            // transmission (Tf)

    arma::mat vlo;
    arma::umat fio;
    quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                         &vlo, &fio, nullptr, nullptr, &bsdf);
    REQUIRE(std::filesystem::exists("cube.mtl"));

    arma::mat mtl_prop, bsdf_rd;
    arma::uvec mtl_ind_rd;
    std::vector<std::string> mtl_names_rd;
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &mtl_prop, nullptr, nullptr,
                                        nullptr, &mtl_ind_rd, nullptr, &mtl_names_rd, &bsdf_rd);

    REQUIRE(bsdf_rd.n_rows == 1);
    REQUIRE(bsdf_rd.n_cols == 17);
    CHECK(arma::approx_equal(bsdf_rd.row(0), bsdf.row(0), "absdiff", 1e-9));

    std::remove("cube.obj");
    std::remove("cube.mtl");
}

TEST_CASE("Test OBJ File Write - Multiple objects")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();

    arma::mat meshA = make_mesh(V, F);
    arma::mat meshB = meshA;
    meshB.col(0) += 10.0; // shift x of all three triangle corners -> disjoint second cube
    meshB.col(3) += 10.0;
    meshB.col(6) += 10.0;

    arma::mat mesh = arma::join_cols(meshA, meshB); // [24, 9]
    arma::uvec obj_ind = arma::join_cols(arma::ones<arma::uvec>(12), 2 * arma::ones<arma::uvec>(12));
    std::vector<std::string> obj_names = {"CubeA", "CubeB"};

    arma::mat vlo;
    arma::umat fio;
    quadriga_lib::obj_file_write<double>("cubes.obj", &mesh, &obj_ind, nullptr, &obj_names, nullptr, &vlo, &fio);

    // No cross-object merging -> 8 + 8 vertices
    CHECK(vlo.n_rows == 16);

    arma::mat vert_list_rd;
    arma::umat face_ind_rd;
    arma::uvec obj_ind_rd;
    std::vector<std::string> obj_names_rd;
    auto n_faces = quadriga_lib::obj_file_read<double>("cubes.obj", nullptr, nullptr, &vert_list_rd,
                                                       &face_ind_rd, &obj_ind_rd, nullptr, &obj_names_rd);

    CHECK(n_faces == 24ULL);
    CHECK(vert_list_rd.n_rows == 16);

    REQUIRE(obj_names_rd.size() == 2);
    CHECK(obj_names_rd[0] == "CubeA");
    CHECK(obj_names_rd[1] == "CubeB");

    CHECK(arma::all(obj_ind_rd.subvec(0, 11) == 1U));
    CHECK(arma::all(obj_ind_rd.subvec(12, 23) == 2U));

    CHECK(arma::approx_equal(make_mesh(vert_list_rd, face_ind_rd), mesh, "absdiff", 1e-12));

    std::remove("cubes.obj");
}

TEST_CASE("Test OBJ File Write - Error handling")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::mat vlo;
    arma::umat fio;

    // Both mesh and vert_list/face_ind given
    CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, nullptr, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, &V, &F),
                    std::invalid_argument);

    // Neither geometry source given
    CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, nullptr, nullptr),
                    std::invalid_argument);

    // vert_list without face_ind
    CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                                         nullptr, nullptr, &V, nullptr),
                    std::invalid_argument);

    // Non-contiguous obj_ind: {1,1,2,2,1,...} -> object 1 reappears
    {
        arma::uvec obj_bad = arma::ones<arma::uvec>(12);
        obj_bad(2) = 2;
        obj_bad(3) = 2;
        std::vector<std::string> on = {"A", "B"};
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, &obj_bad, nullptr, &on, nullptr,
                                                             nullptr, nullptr),
                        std::invalid_argument);
    }

    // File name does not end in .obj
    CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("cube.txt", &mesh, nullptr, nullptr, nullptr, nullptr,
                                                         &vlo, &fio),
                    std::invalid_argument);

    // obj_names too short for obj_ind
    {
        arma::uvec obj_ind = arma::join_cols(arma::ones<arma::uvec>(6), 2 * arma::ones<arma::uvec>(6));
        std::vector<std::string> on = {"OnlyOne"};
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, &obj_ind, nullptr, &on, nullptr,
                                                             &vlo, &fio),
                        std::invalid_argument);
    }

    // mtl_names too short for mtl_ind
    {
        arma::uvec mtl_ind = arma::join_cols(arma::ones<arma::uvec>(6), 2 * arma::ones<arma::uvec>(6));
        std::vector<std::string> mn = {"OnlyOne"};
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, nullptr, &mtl_ind, nullptr, &mn,
                                                             &vlo, &fio),
                        std::invalid_argument);
    }

    // bsdf given without mtl_ind / mtl_names
    {
        arma::mat bsdf(1, 17, arma::fill::zeros);
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, nullptr, nullptr, nullptr, nullptr,
                                                             &vlo, &fio, nullptr, nullptr, &bsdf),
                        std::invalid_argument);
    }

    // None of the error cases should have produced a file
    CHECK_FALSE(std::filesystem::exists("x.obj"));
    CHECK_FALSE(std::filesystem::exists("cube.txt"));
}

TEST_CASE("Test OBJ File Write - Outputs only (empty filename)")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    // Empty filename: derive vert_list / face_ind from mesh, write no file
    arma::mat vlo;
    arma::umat fio;
    CHECK_NOTHROW(quadriga_lib::obj_file_write<double>("", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio));

    CHECK(vlo.n_rows == 8);
    CHECK(vlo.n_cols == 3);
    CHECK(fio.n_rows == 12);
    CHECK(fio.n_cols == 3);
    CHECK(arma::approx_equal(make_mesh(vlo, fio), mesh, "absdiff", 1e-12));
}