// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdio>
#include <fstream>
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

// Read the property value for material index iM from column 'key' of csv_prop.
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
// {a,b,c,d,att,attB,alpha,alphaB,fRef} row, applying documented defaults for absent columns.
static inline bool em_row_matches(const std::unordered_map<std::string, std::vector<double>> &p,
                                  arma::uword iM, const std::array<double, 9> &e, double tol = 1e-9)
{
    const std::array<std::pair<const char *, double>, 9> cols = {{{"a", 1.0}, {"b", 0.0}, {"c", 0.0}, {"d", 0.0}, {"att", 0.0}, {"attB", 0.0}, {"alpha", 0.0}, {"alphaB", 0.0}, {"fRef", 1.0}}};
    for (size_t k = 0; k < 9; ++k)
    {
        double got = prop_at(p, cols[k].first, iM, cols[k].second);
        double scale = std::max(1.0, std::abs(e[k]));
        if (std::abs(got - e[k]) > tol * scale)
            return false;
    }
    return true;
}

// Collect the text following a line prefix, e.g. "o " -> object names, "newmtl " -> material names
static inline std::vector<std::string> collect_after(const std::string &fn, const std::string &prefix)
{
    std::ifstream f(fn);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(prefix, 0) == 0)
            out.push_back(line.substr(prefix.size()));
    return out;
}

// 1-based index of a material name in the read-back list, 0 if absent
static inline arma::uword name_index(const std::vector<std::string> &names, const std::string &n)
{
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == n)
            return (arma::uword)i + 1;
    return 0;
}

// True if the rows of A are a permutation of the rows of B (small n, O(n^2))
static inline bool rows_match_unordered(const arma::mat &A, const arma::mat &B, double tol = 1e-12)
{
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols)
        return false;

    std::vector<bool> used(B.n_rows, false);
    for (arma::uword i = 0; i < A.n_rows; ++i)
    {
        bool found = false;
        for (arma::uword j = 0; j < B.n_rows && !found; ++j)
            if (!used[j] && arma::approx_equal(A.row(i), B.row(j), "absdiff", tol))
                used[j] = true, found = true;

        if (!found)
            return false;
    }
    return true;
}

TEST_CASE("Test OBJ File Write - Mesh round-trip (geometry only)")
{
    std::remove("cube.mtl"); // clear any stale file from a prior (randomly-ordered) test
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
    arma::mat mesh_rd, vert_list_rd;
    arma::umat face_ind_rd;
    arma::uvec obj_ind_rd, mtl_ind_rd, csv_ind_rd;
    std::vector<std::string> obj_names_rd, mtl_names_rd, csv_names_rd;
    std::unordered_map<std::string, std::vector<double>> csv_prop_rd;

    auto n_faces = quadriga_lib::obj_file_read<double>("cube.obj", &mesh_rd, &vert_list_rd, &face_ind_rd,
                                                       &obj_ind_rd, &obj_names_rd, &mtl_ind_rd, &mtl_names_rd, nullptr,
                                                       "", &csv_ind_rd, &csv_names_rd, &csv_prop_rd);

    CHECK(n_faces == 12ULL);
    CHECK(vert_list_rd.n_rows == 8);
    CHECK(arma::approx_equal(mesh_rd, mesh, "absdiff", 1e-12));
    CHECK(arma::approx_equal(make_mesh(vert_list_rd, face_ind_rd), mesh, "absdiff", 1e-12));

    CHECK(obj_names_rd.size() == 1);
    CHECK(obj_names_rd[0] == "object");

    // No usemtl written -> no materials on read-back (no synthetic "default")
    CHECK(mtl_names_rd.size() == 0);

    // obj_ind 0-based; no materials -> mtl_ind / csv_ind are all 0 (no material)
    CHECK(arma::all(obj_ind_rd == 0U));
    CHECK(arma::all(mtl_ind_rd == 0U));
    CHECK(arma::all(csv_ind_rd == 0U));

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
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, &vert_list_rd, &face_ind_rd);

    arma::mat mesh_expected = make_mesh(V, F);
    CHECK(arma::approx_equal(make_mesh(vert_list_rd, face_ind_rd), mesh_expected, "absdiff", 1e-12));

    std::remove("cube.obj");
}

TEST_CASE("Test OBJ File Write - Materials round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::zeros<arma::uvec>(12); // single object, 0-based
    std::vector<std::string> obj_names = {"Cube"};
    arma::mat vlo;
    arma::umat fio;

    SECTION("Named ITU materials")
    {
        arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
        mtl_ind.subvec(4, 11).fill(2); // faces 0-3 = material 1, 4-11 = material 2 (1-based)
        std::vector<std::string> mtl_names = {"itu_concrete", "itu_wood"};

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names, &vlo, &fio);
        CHECK(std::filesystem::exists("cube.mtl"));

        // Read back; resolve EM properties from the built-in default table (names are ITU materials)
        arma::uvec mtl_ind_rd, csv_ind_rd;
        std::vector<std::string> mtl_names_rd, csv_names_rd;
        std::unordered_map<std::string, std::vector<double>> csv_prop_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            &mtl_ind_rd, &mtl_names_rd, nullptr,
                                            "", &csv_ind_rd, &csv_names_rd, &csv_prop_rd);

        REQUIRE(mtl_names_rd.size() == 2);
        CHECK(mtl_names_rd[0] == "itu_concrete");
        CHECK(mtl_names_rd[1] == "itu_wood");

        CHECK(em_row_matches(csv_prop_rd, csv_ind_rd(0) - 1, {5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0}));
        CHECK(em_row_matches(csv_prop_rd, csv_ind_rd(4) - 1, {1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0}));

        // .mtl side index round-trips 1-based
        CHECK(arma::all(mtl_ind_rd.subvec(0, 3) == 1U));
        CHECK(arma::all(mtl_ind_rd.subvec(4, 11) == 2U));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }

    SECTION("Custom material via CSV")
    {
        arma::uvec mtl_ind = arma::ones<arma::uvec>(12); // 1-based: material 1 = "glass"
        std::vector<std::string> mtl_names = {"glass"};

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names, &vlo, &fio);

        // EM properties come from a CSV, not from the OBJ/MTL
        std::ofstream csv_file("custom_materials.csv");
        REQUIRE(csv_file.is_open());
        csv_file << "name,a,b,c,d,att\n";
        csv_file << "air,1.0,0.0,0.0,0.0,0.0\n";
        csv_file << "glass,6.0,0.0,0.1,1.2,0.0\n";
        csv_file.close();

        arma::uvec mtl_ind_rd, csv_ind_rd;
        std::vector<std::string> mtl_names_rd, csv_names_rd;
        std::unordered_map<std::string, std::vector<double>> csv_prop_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            &mtl_ind_rd, &mtl_names_rd, nullptr,
                                            "custom_materials.csv", &csv_ind_rd, &csv_names_rd, &csv_prop_rd);

        REQUIRE(mtl_names_rd.size() == 1);
        CHECK(mtl_names_rd[0] == "glass");

        CHECK(em_row_matches(csv_prop_rd, csv_ind_rd(0) - 1, {6.0, 0.0, 0.1, 1.2, 0.0, 0.0, 0.0, 0.0, 1.0}));
        CHECK(arma::all(mtl_ind_rd == 1U));

        std::remove("cube.obj");
        std::remove("cube.mtl");
        std::remove("custom_materials.csv");
    }
}

TEST_CASE("Test OBJ File Write - BSDF round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::zeros<arma::uvec>(12);
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12); // 1-based: material 1 = "painted"
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

    arma::mat bsdf_rd;
    arma::uvec mtl_ind_rd;
    std::vector<std::string> mtl_names_rd;
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        &mtl_ind_rd, &mtl_names_rd, &bsdf_rd);

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

    arma::mat mesh = arma::join_cols(meshA, meshB);                                                // [24, 9]
    arma::uvec obj_ind = arma::join_cols(arma::zeros<arma::uvec>(12), arma::ones<arma::uvec>(12)); // 0-based
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
    auto n_faces = quadriga_lib::obj_file_read<double>("cubes.obj", nullptr, &vert_list_rd, &face_ind_rd,
                                                       &obj_ind_rd, &obj_names_rd);

    CHECK(n_faces == 24ULL);
    CHECK(vert_list_rd.n_rows == 16);

    REQUIRE(obj_names_rd.size() == 2);
    CHECK(obj_names_rd[0] == "CubeA");
    CHECK(obj_names_rd[1] == "CubeB");

    CHECK(arma::all(obj_ind_rd.subvec(0, 11) == 0U));
    CHECK(arma::all(obj_ind_rd.subvec(12, 23) == 1U));

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

    // Non-contiguous obj_ind: {0,0,1,1,0,...} -> object 0 reappears
    {
        arma::uvec obj_bad = arma::zeros<arma::uvec>(12);
        obj_bad(2) = 1;
        obj_bad(3) = 1;
        std::vector<std::string> on = {"A", "B"};
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, &obj_bad, nullptr, &on, nullptr,
                                                             nullptr, nullptr),
                        std::invalid_argument);
    }

    // File name does not end in .obj
    CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("cube.txt", &mesh, nullptr, nullptr, nullptr, nullptr,
                                                         &vlo, &fio),
                    std::invalid_argument);

    // obj_names too short for obj_ind (0-based: max index 1 needs 2 names)
    {
        arma::uvec obj_ind = arma::join_cols(arma::zeros<arma::uvec>(6), arma::ones<arma::uvec>(6));
        std::vector<std::string> on = {"OnlyOne"};
        CHECK_THROWS_AS(quadriga_lib::obj_file_write<double>("x.obj", &mesh, &obj_ind, nullptr, &on, nullptr,
                                                             &vlo, &fio),
                        std::invalid_argument);
    }

    // mtl_names too short for mtl_ind (1-based: max index 2 needs 2 names)
    {
        arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
        mtl_ind.subvec(6, 11).fill(2);
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

TEST_CASE("Test OBJ File Write - CSV material table round-trip")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::zeros<arma::uvec>(12);
    std::vector<std::string> obj_names = {"Cube"};

    // 1-based material indices: faces 0-3 -> concrete, faces 4-11 -> wood
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
    mtl_ind.subvec(4, 11).fill(2);
    std::vector<std::string> mtl_names = {"concrete", "wood"};

    // EM/acoustic table; names must match the usemtl names so the read-back resolves by name
    std::vector<std::string> csv_names = {"concrete", "wood"};
    std::unordered_map<std::string, std::vector<double>> csv_prop;
    csv_prop["a"] = {5.24, 1.99};
    csv_prop["c"] = {0.0462, 0.0047};
    csv_prop["d"] = {0.7822, 1.0718};
    csv_prop["fRef"] = {1.0, 1.0};

    // csv_ind is 1-based, validated against csv_names
    arma::uvec csv_ind = arma::ones<arma::uvec>(12);
    csv_ind.subvec(4, 11).fill(2);

    arma::mat vlo;
    arma::umat fio;
    quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                         &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                         &csv_ind, &csv_names, &csv_prop, /*csv_write_defaults=*/false);

    REQUIRE(std::filesystem::exists("cube.obj"));
    REQUIRE(std::filesystem::exists("cube.mtl"));
    REQUIRE(std::filesystem::exists("cube.csv"));

    arma::uvec mtl_ind_rd, csv_ind_rd;
    std::vector<std::string> mtl_names_rd, csv_names_rd;
    std::unordered_map<std::string, std::vector<double>> csv_prop_rd;
    quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                        &mtl_ind_rd, &mtl_names_rd, nullptr,
                                        "cube.csv", &csv_ind_rd, &csv_names_rd, &csv_prop_rd);

    // CSV table preserved (full table, in written order)
    REQUIRE(csv_names_rd.size() == 2);
    CHECK(csv_names_rd[0] == "concrete");
    CHECK(csv_names_rd[1] == "wood");

    // EM properties resolve per face (csv_ind is 1-based -> -1 for the table row)
    CHECK(em_row_matches(csv_prop_rd, csv_ind_rd(0) - 1, {5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0}));
    CHECK(em_row_matches(csv_prop_rd, csv_ind_rd(4) - 1, {1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0}));

    // Visual side round-trips 1-based
    CHECK(arma::all(mtl_ind_rd.subvec(0, 3) == 1U));
    CHECK(arma::all(mtl_ind_rd.subvec(4, 11) == 2U));

    std::remove("cube.obj");
    std::remove("cube.mtl");
    std::remove("cube.csv");
}

TEST_CASE("Test OBJ File Write - CSV columns, defaults and validation")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::zeros<arma::uvec>(12);
    std::vector<std::string> obj_names = {"Cube"};
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
    std::vector<std::string> mtl_names = {"slab"};

    std::vector<std::string> csv_names = {"slab"};
    std::unordered_map<std::string, std::vector<double>> csv_prop;
    csv_prop["c"] = {0.05};  // canonical, present
    csv_prop["tf"] = {2.0};  // canonical, present
    csv_prop["zzz"] = {7.0}; // extra (non-canonical)

    arma::mat vlo;
    arma::umat fio;

    SECTION("csv_write_defaults = false -> only present columns, canonical order then extras")
    {
        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                             nullptr, &csv_names, &csv_prop, /*csv_write_defaults=*/false);
        REQUIRE(std::filesystem::exists("cube.csv"));

        std::ifstream f("cube.csv");
        std::string header, row;
        std::getline(f, header);
        std::getline(f, row);
        f.close();

        CHECK(header == "name,c,tf,zzz"); // c before tf (canonical), zzz last (extra)
        CHECK(row == "slab,0.05,2,7");

        std::remove("cube.obj");
        std::remove("cube.mtl");
        std::remove("cube.csv");
    }

    SECTION("csv_write_defaults = true -> full canonical set with defaults")
    {
        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                             nullptr, &csv_names, &csv_prop, /*csv_write_defaults=*/true);
        REQUIRE(std::filesystem::exists("cube.csv"));

        std::ifstream f("cube.csv");
        std::string header, row;
        std::getline(f, header);
        std::getline(f, row);
        f.close();

        CHECK(header == "name,a,b,c,d,e,f,g,h,att,attB,alpha,alphaB,fRef,m,resF,resQ,resS,coiF,coiQ,coiA,tf,tfB,zzz");
        // a=1, e=1, fRef=1 by default; c=0.05 and tf=2 from csv_prop; rest 0; zzz=7 last
        CHECK(row == "slab,1,0,0.05,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,7");

        std::remove("cube.obj");
        std::remove("cube.mtl");
        std::remove("cube.csv");
    }

    SECTION("Validation: csv_prop column with wrong length throws")
    {
        std::unordered_map<std::string, std::vector<double>> bad_prop;
        bad_prop["a"] = {1.0, 2.0}; // length 2, csv_names has 1 entry
        CHECK_THROWS_AS(
            quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                                 &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                                 nullptr, &csv_names, &bad_prop, false),
            std::invalid_argument);
    }

    SECTION("Validation: csv_ind out of range throws")
    {
        arma::uvec csv_ind_bad = arma::ones<arma::uvec>(12);
        csv_ind_bad(0) = 5; // csv_names has 1 entry -> max valid index is 1
        CHECK_THROWS_AS(
            quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                                 &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                                 &csv_ind_bad, &csv_names, &csv_prop, false),
            std::invalid_argument);
    }

    SECTION("Validation: csv inputs without csv_names throw")
    {
        CHECK_THROWS_AS(
            quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                                 &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                                 nullptr, nullptr, &csv_prop, false),
            std::invalid_argument);
    }
}

TEST_CASE("Test OBJ File Write - Duplicate material names")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();
    arma::mat mesh = make_mesh(V, F);

    arma::uvec obj_ind = arma::zeros<arma::uvec>(12);
    std::vector<std::string> obj_names = {"Cube"};

    // 1-based material index: faces 0-3 -> 1, faces 4-7 -> 2, faces 8-11 -> 3
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);
    mtl_ind.subvec(4, 7).fill(2);
    mtl_ind.subvec(8, 11).fill(3);

    arma::mat vlo;
    arma::umat fio;

    SECTION("No bsdf -> same-named entries collapse into one")
    {
        std::vector<std::string> mtl_names = {"concrete", "concrete", "steel"};

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names, &vlo, &fio);
        REQUIRE(std::filesystem::exists("cube.mtl"));

        // Both "concrete" entries share one .mtl block
        auto written = collect_after("cube.mtl", "newmtl ");
        REQUIRE(written.size() == 2);
        CHECK(written[0] == "concrete");
        CHECK(written[1] == "steel");

        // Merged materials do not emit a redundant usemtl tag
        auto used = collect_after("cube.obj", "usemtl ");
        REQUIRE(used.size() == 2);
        CHECK(used[0] == "concrete");
        CHECK(used[1] == "steel");

        arma::uvec mtl_ind_rd;
        std::vector<std::string> mtl_names_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            &mtl_ind_rd, &mtl_names_rd);

        REQUIRE(mtl_names_rd.size() == 2);
        const arma::uword iC = name_index(mtl_names_rd, "concrete");
        const arma::uword iS = name_index(mtl_names_rd, "steel");
        REQUIRE(iC != 0);
        REQUIRE(iS != 0);

        CHECK(arma::all(mtl_ind_rd.subvec(0, 7) == iC)); // faces 0-3 and 4-7 merged
        CHECK(arma::all(mtl_ind_rd.subvec(8, 11) == iS));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }

    SECTION("Identical bsdf rows -> same-named entries collapse into one")
    {
        std::vector<std::string> mtl_names = {"concrete", "concrete", "steel"};

        arma::mat bsdf(3, 17, arma::fill::zeros);
        bsdf.col(0).fill(0.8), bsdf.col(1).fill(0.8), bsdf.col(2).fill(0.8); // base color
        bsdf.col(3).fill(1.0);                                               // d
        bsdf.col(4).fill(0.5);                                               // Pr
        bsdf.col(6).fill(1.45);                                              // Ni
        bsdf.col(7).fill(0.5);                                               // Ks
        bsdf(2, 4) = 0.9;                                                    // steel differs, rows 0 and 1 identical

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, &bsdf);
        REQUIRE(std::filesystem::exists("cube.mtl"));

        auto written = collect_after("cube.mtl", "newmtl ");
        REQUIRE(written.size() == 2);
        CHECK(written[0] == "concrete");
        CHECK(written[1] == "steel");

        arma::mat bsdf_rd;
        arma::uvec mtl_ind_rd;
        std::vector<std::string> mtl_names_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            &mtl_ind_rd, &mtl_names_rd, &bsdf_rd);

        REQUIRE(mtl_names_rd.size() == 2);
        REQUIRE(bsdf_rd.n_rows == 2);

        const arma::uword iC = name_index(mtl_names_rd, "concrete");
        const arma::uword iS = name_index(mtl_names_rd, "steel");
        REQUIRE(iC != 0);
        REQUIRE(iS != 0);

        CHECK(arma::approx_equal(bsdf_rd.row(iC - 1), bsdf.row(0), "absdiff", 1e-9));
        CHECK(arma::approx_equal(bsdf_rd.row(iS - 1), bsdf.row(2), "absdiff", 1e-9));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }

    SECTION("Differing bsdf rows -> same-named entries are suffixed")
    {
        std::vector<std::string> mtl_names = {"concrete", "concrete", "steel"};

        arma::mat bsdf(3, 17, arma::fill::zeros);
        bsdf.col(0).fill(0.8), bsdf.col(1).fill(0.8), bsdf.col(2).fill(0.8);
        bsdf.col(3).fill(1.0);
        bsdf.col(4).fill(0.5);
        bsdf.col(6).fill(1.45);
        bsdf.col(7).fill(0.5);
        bsdf(1, 4) = 0.2; // second "concrete" has a different roughness
        bsdf(2, 4) = 0.9;

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, &bsdf);
        REQUIRE(std::filesystem::exists("cube.mtl"));

        // Written in material-row order: rows 0, 1, 2
        auto written = collect_after("cube.mtl", "newmtl ");
        REQUIRE(written.size() == 3);
        CHECK(written[0] == "concrete.001");
        CHECK(written[1] == "concrete.002");
        CHECK(written[2] == "steel");

        // "steel" is unique and keeps its bare name
        auto used = collect_after("cube.obj", "usemtl ");
        REQUIRE(used.size() == 3);
        CHECK(used[0] == "concrete.001");
        CHECK(used[1] == "concrete.002");
        CHECK(used[2] == "steel");

        arma::mat bsdf_rd;
        arma::uvec mtl_ind_rd;
        std::vector<std::string> mtl_names_rd;
        quadriga_lib::obj_file_read<double>("cube.obj", nullptr, nullptr, nullptr, nullptr, nullptr,
                                            &mtl_ind_rd, &mtl_names_rd, &bsdf_rd);

        REQUIRE(mtl_names_rd.size() == 3);
        REQUIRE(bsdf_rd.n_rows == 3);

        const arma::uword i1 = name_index(mtl_names_rd, "concrete.001");
        const arma::uword i2 = name_index(mtl_names_rd, "concrete.002");
        const arma::uword iS = name_index(mtl_names_rd, "steel");
        REQUIRE(i1 != 0);
        REQUIRE(i2 != 0);
        REQUIRE(iS != 0);

        // Each variant keeps its own BSDF data
        CHECK(arma::approx_equal(bsdf_rd.row(i1 - 1), bsdf.row(0), "absdiff", 1e-9));
        CHECK(arma::approx_equal(bsdf_rd.row(i2 - 1), bsdf.row(1), "absdiff", 1e-9));
        CHECK(arma::approx_equal(bsdf_rd.row(iS - 1), bsdf.row(2), "absdiff", 1e-9));

        // Faces keep their material assignment
        CHECK(arma::all(mtl_ind_rd.subvec(0, 3) == i1));
        CHECK(arma::all(mtl_ind_rd.subvec(4, 7) == i2));
        CHECK(arma::all(mtl_ind_rd.subvec(8, 11) == iS));

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }

    SECTION("Generated suffix does not collide with an existing name")
    {
        // "concrete.001" already exists, so the split entries must skip it
        std::vector<std::string> mtl_names = {"concrete", "concrete", "concrete.001"};

        arma::mat bsdf(3, 17, arma::fill::zeros);
        bsdf.col(0).fill(0.8), bsdf.col(1).fill(0.8), bsdf.col(2).fill(0.8);
        bsdf.col(3).fill(1.0);
        bsdf.col(4).fill(0.5);
        bsdf.col(6).fill(1.45);
        bsdf.col(7).fill(0.5);
        bsdf(1, 4) = 0.2;
        bsdf(2, 4) = 0.9;

        quadriga_lib::obj_file_write<double>("cube.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, &bsdf);
        REQUIRE(std::filesystem::exists("cube.mtl"));

        auto written = collect_after("cube.mtl", "newmtl ");
        REQUIRE(written.size() == 3);
        CHECK(written[0] == "concrete.002");
        CHECK(written[1] == "concrete.003");
        CHECK(written[2] == "concrete.001");

        // All three names are distinct
        CHECK(written[0] != written[1]);
        CHECK(written[0] != written[2]);
        CHECK(written[1] != written[2]);

        std::remove("cube.obj");
        std::remove("cube.mtl");
    }
}

TEST_CASE("Test OBJ File Write - Separate by loose parts")
{
    arma::mat V = cube_vertices();
    arma::umat F = cube_faces();

    arma::mat meshA = make_mesh(V, F); // cube at the origin
    arma::mat meshB = meshA;           // disjoint cube, shifted along x
    meshB.col(0) += 10.0;
    meshB.col(3) += 10.0;
    meshB.col(6) += 10.0;

    // Interleave the two islands so the parts are NOT contiguous in the input
    arma::mat mesh(24, 9);
    for (arma::uword i = 0; i < 12; ++i)
    {
        mesh.row(2 * i) = meshA.row(i);
        mesh.row(2 * i + 1) = meshB.row(i);
    }

    // Both islands live in a single object
    arma::uvec obj_ind = arma::zeros<arma::uvec>(24);
    std::vector<std::string> obj_names = {"Cube"};

    // Island A -> material 1, island B -> material 2 (1-based)
    arma::uvec mtl_ind = arma::ones<arma::uvec>(24);
    for (arma::uword i = 0; i < 12; ++i)
        mtl_ind(2 * i + 1) = 2;
    std::vector<std::string> mtl_names = {"matA", "matB"};

    arma::mat vlo;
    arma::umat fio;

    SECTION("Disabled -> single object, faces in input order")
    {
        quadriga_lib::obj_file_write<double>("cubes.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                             nullptr, nullptr, nullptr, false, /*split_loose_parts=*/false);

        auto blocks = collect_after("cubes.obj", "o ");
        REQUIRE(blocks.size() == 1);
        CHECK(blocks[0] == "Cube");

        arma::mat mesh_rd;
        auto n_faces = quadriga_lib::obj_file_read<double>("cubes.obj", &mesh_rd);
        CHECK(n_faces == 24ULL);
        CHECK(arma::approx_equal(mesh_rd, mesh, "absdiff", 1e-12));

        std::remove("cubes.obj");
        std::remove("cubes.mtl");
    }

    SECTION("Enabled -> one object per connected component")
    {
        quadriga_lib::obj_file_write<double>("cubes.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names,
                                             &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                             nullptr, nullptr, nullptr, false, /*split_loose_parts=*/true);

        // The split object is renamed; parts are numbered by their first face
        auto blocks = collect_after("cubes.obj", "o ");
        REQUIRE(blocks.size() == 2);
        CHECK(blocks[0] == "Cube.001");
        CHECK(blocks[1] == "Cube.002");

        // Splitting does not duplicate geometry: parts share no vertices by construction
        CHECK(vlo.n_rows == 16);

        arma::mat mesh_rd, vert_list_rd;
        arma::umat face_ind_rd;
        arma::uvec obj_ind_rd, mtl_ind_rd;
        std::vector<std::string> obj_names_rd, mtl_names_rd;

        auto n_faces = quadriga_lib::obj_file_read<double>("cubes.obj", &mesh_rd, &vert_list_rd, &face_ind_rd,
                                                           &obj_ind_rd, &obj_names_rd, &mtl_ind_rd, &mtl_names_rd);

        CHECK(n_faces == 24ULL);
        CHECK(vert_list_rd.n_rows == 16);

        REQUIRE(obj_names_rd.size() == 2);
        CHECK(obj_names_rd[0] == "Cube.001");
        CHECK(obj_names_rd[1] == "Cube.002");

        // 12 faces per part, regrouped out of the interleaved input
        CHECK(arma::all(obj_ind_rd.subvec(0, 11) == 0U));
        CHECK(arma::all(obj_ind_rd.subvec(12, 23) == 1U));

        // Same faces, reordered
        CHECK(rows_match_unordered(mesh_rd, mesh));

        // Materials travel with their faces: part 1 is island A, part 2 is island B
        const arma::uword iA = name_index(mtl_names_rd, "matA");
        const arma::uword iB = name_index(mtl_names_rd, "matB");
        REQUIRE(iA != 0);
        REQUIRE(iB != 0);
        CHECK(arma::all(mtl_ind_rd.subvec(0, 11) == iA));
        CHECK(arma::all(mtl_ind_rd.subvec(12, 23) == iB));

        // Part 1 holds the cube at the origin, part 2 the shifted one
        arma::vec x_part1 = vert_list_rd.submat(0, 0, 7, 0);
        arma::vec x_part2 = vert_list_rd.submat(8, 0, 15, 0);
        CHECK(arma::all(arma::abs(x_part1) < 1.5));
        CHECK(arma::all(x_part2 > 8.5));

        std::remove("cubes.obj");
        std::remove("cubes.mtl");
    }

    SECTION("Objects that do not split keep their name")
    {
        arma::mat meshC = make_mesh(V, F); // third cube, its own object
        meshC.col(0) += 20.0;
        meshC.col(3) += 20.0;
        meshC.col(6) += 20.0;

        arma::mat mesh2 = arma::join_cols(mesh, meshC);
        arma::uvec obj_ind2 = arma::join_cols(arma::zeros<arma::uvec>(24), arma::ones<arma::uvec>(12));
        std::vector<std::string> obj_names2 = {"Multi", "Single"};

        quadriga_lib::obj_file_write<double>("cubes.obj", &mesh2, &obj_ind2, nullptr, &obj_names2, nullptr,
                                             &vlo, &fio, nullptr, nullptr, nullptr, 0.001,
                                             nullptr, nullptr, nullptr, false, /*split_loose_parts=*/true);

        auto blocks = collect_after("cubes.obj", "o ");
        REQUIRE(blocks.size() == 3);
        CHECK(blocks[0] == "Multi.001");
        CHECK(blocks[1] == "Multi.002");
        CHECK(blocks[2] == "Single"); // single part -> unsuffixed

        CHECK(vlo.n_rows == 24);

        arma::mat mesh_rd;
        auto n_faces = quadriga_lib::obj_file_read<double>("cubes.obj", &mesh_rd);
        CHECK(n_faces == 36ULL);
        CHECK(rows_match_unordered(mesh_rd, mesh2));

        std::remove("cubes.obj");
    }
}

TEST_CASE("Test OBJ File Write - Vertex welding")
{
    arma::mat vlo;
    arma::umat fio;

    SECTION("Large shared-vertex lattice welds to the exact vertex count")
    {
        // N x N grid of quads in the XY plane; every interior vertex is shared by several faces
        const arma::uword N = 60;
        arma::mat mesh(2 * N * N, 9);

        arma::uword r = 0;
        for (arma::uword i = 0; i < N; ++i)
            for (arma::uword j = 0; j < N; ++j)
            {
                const double x = (double)i, y = (double)j;
                mesh.row(r++) = arma::rowvec{x, y, 0.0, x + 1.0, y, 0.0, x + 1.0, y + 1.0, 0.0};
                mesh.row(r++) = arma::rowvec{x, y, 0.0, x + 1.0, y + 1.0, 0.0, x, y + 1.0, 0.0};
            }

        quadriga_lib::obj_file_write<double>("", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio);

        CHECK(vlo.n_rows == (N + 1) * (N + 1)); // 3721 unique lattice points
        CHECK(fio.n_rows == 2 * N * N);
        CHECK(arma::approx_equal(make_mesh(vlo, fio), mesh, "absdiff", 1e-12));
    }

    SECTION("Co-located vertices merge across grid cell boundaries")
    {
        // The near-duplicate x values straddle a multiple of the threshold, so a spatial lookup
        // must search neighboring cells, not just the cell the vertex falls into
        arma::mat mesh(2, 9);
        mesh.row(0) = arma::rowvec{0.9995, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0};
        mesh.row(1) = arma::rowvec{1.0004, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0};

        // 0.9 mm apart -> merged at a 1 mm threshold
        quadriga_lib::obj_file_write<double>("", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio, nullptr,
                                             nullptr, nullptr, 0.001);
        CHECK(vlo.n_rows == 3);
        CHECK(fio(0, 0) == fio(1, 0));
        CHECK(vlo(0, 0) == 0.9995); // the first vertex seen wins

        // Same geometry, tighter threshold -> kept apart
        quadriga_lib::obj_file_write<double>("", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio, nullptr,
                                             nullptr, nullptr, 0.0005);
        CHECK(vlo.n_rows == 4);
        CHECK(fio(0, 0) != fio(1, 0));
    }

    SECTION("Merging works at negative coordinates")
    {
        arma::mat mesh(2, 9);
        mesh.row(0) = arma::rowvec{-0.0004, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 1.0, 0.0};
        mesh.row(1) = arma::rowvec{-0.0013, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 1.0, 0.0};

        quadriga_lib::obj_file_write<double>("", &mesh, nullptr, nullptr, nullptr, nullptr, &vlo, &fio, nullptr,
                                             nullptr, nullptr, 0.001);
        CHECK(vlo.n_rows == 3);
        CHECK(fio(0, 0) == fio(1, 0));
    }

    SECTION("No merging across objects")
    {
        // Two identical triangles in different objects stay separate regardless of threshold
        arma::mat mesh(2, 9);
        mesh.row(0) = arma::rowvec{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0};
        mesh.row(1) = mesh.row(0);

        arma::uvec obj_ind = {0, 1};
        std::vector<std::string> obj_names = {"A", "B"};

        quadriga_lib::obj_file_write<double>("", &mesh, &obj_ind, nullptr, &obj_names, nullptr, &vlo, &fio);
        CHECK(vlo.n_rows == 6);
    }
}