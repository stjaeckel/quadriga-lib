// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"
#include "quadriga_lib.hpp"
#include <iostream>

TEST_CASE("Ray-Triangle Intersect - Simple Mode")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},  //  2 South Lower
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, //  3 West Lower
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},  //  4 Bottom NorthWest
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},     //  5 East Lower
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},    //  6 North Lower
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},    //  7 Top SouthWest
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},  //  8 South Upper
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},   //  9 West Upper
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},  // 10 Bottom SouthEast
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},     // 11 East Upper
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};     // 12 North Upper

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    CHECK(no_interact.at(0) == 2);
    CHECK(fbs_ind.at(0) == 9);
    CHECK(sbs_ind.at(0) == 11);
}

TEST_CASE("Ray-Triangle Intersect - Sub-mesh Mode")
{

    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},  //  2 South Lower
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, //  3 West Lower
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},  //  4 Bottom NorthWest
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},     //  5 East Lower
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},    //  6 North Lower
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},    //  7 Top SouthWest
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},  //  8 South Upper
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},   //  9 West Upper
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},  // 10 Bottom SouthEast
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},     // 11 East Upper
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};     // 12 North Upper

    // Subdivide the mesh into smaller chunks
    arma::mat cube_sub = cube;
    quadriga_lib::subdivide_triangles(3, &cube, &cube_sub);

    // Build sub-meshes
    arma::mat cube_re;
    arma::u32_vec cube_index;

    auto n_sub = quadriga_lib::triangle_mesh_segmentation(&cube_sub, &cube_re, &cube_index, 64, 8);
    CHECK(n_sub == 3);

    // Single ray
    arma::mat orig = {{0.9, -10.0, 0.5}};
    arma::mat dest = {{0.9, 10.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    // Run intersect algorithm without sub-mesh
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube_re, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind);

    // Check results
    arma::mat T = {{0.9, -1.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));
    T = {{0.9, 1.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    // Store FBS and SBS indices
    unsigned fbs_0 = fbs_ind.at(0);
    unsigned sbs_0 = sbs_ind.at(0);
    
    // Run intersect algorithm with sub-mesh
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube_re, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind, &cube_index);

    // Check results
    T = {{0.9, -1.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));
    T = {{0.9, 1.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    // Store FBS and SBS indices
    unsigned fbs_1 = fbs_ind.at(0);
    unsigned sbs_1 = sbs_ind.at(0);

    CHECK(fbs_1 == fbs_0);
    CHECK(sbs_1 == sbs_0);
}

TEST_CASE("Ray-Triangle Intersect - Invalid use_kernel")
{
    arma::mat orig = {{0.0, 0.0, 1.0}}, dest = {{0.0, 0.0, -1.0}};
    arma::mat mesh = {{-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0}};
    arma::u32_vec fbs_ind;

    for (int k : {7, -1, 4})
    {
        INFO("use_kernel = " << k);
        CHECK_THROWS_AS(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, nullptr, &fbs_ind,
                                                                     nullptr, nullptr, nullptr, k),
                        std::invalid_argument);
    }

    // 2 = AVX2 throws when AVX2 is not available
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        CHECK_THROWS_AS(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, nullptr, &fbs_ind,
                                                                     nullptr, nullptr, nullptr, 2),
                        std::invalid_argument);

    for (int k : {0, 1})
    {
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, nullptr, &fbs_ind,
                                                     nullptr, nullptr, nullptr, k);
        CHECK(fbs_ind.at(0) == 1);
    }
}

// Kernels to compare: GENERIC always, AVX2 when the library and the CPU have it
static std::vector<int> rti_kernels()
{
    std::vector<int> k = {1};
    if (quadriga_lib::quadriga_lib_has_AVX2())
        k.push_back(2);
    return k;
}

template <typename dtype>
static void rti_equal_distance_hits()
{
    // Unit square of two triangles sharing the diagonal (0,0)-(1,1)
    arma::Mat<dtype> square = {{0, 0, 0, 1, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 0, 0, 1, 0}};

    // Fan of 6 triangles around the vertex (0,0,0), placed at rows 3, 6, 8, 9, 12 and 14 (1-based) of 16, so that
    // they fall into different lanes of two AVX2 blocks; the other rows are far away and never hit
    arma::Mat<dtype> fan(16, 9);
    for (arma::uword i = 0; i < 16; ++i)
        fan.row(i) = arma::Row<dtype>({100, 100, 100, 101, 100, 100, 100, 101, 100});
    const dtype ring[7][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, 0}};
    const arma::uword rows[6] = {2, 5, 7, 8, 11, 13};
    for (int k = 0; k < 6; ++k)
        fan.row(rows[k]) = arma::Row<dtype>({0, 0, 0, ring[k][0], ring[k][1], 0, ring[k + 1][0], ring[k + 1][1], 0});

    for (int kernel : rti_kernels())
    {
        INFO("use_kernel = " << kernel << ", " << (sizeof(dtype) == 4 ? "float" : "double"));
        arma::u32_vec no, fi, si;

        // Through the shared edge and through the shared vertex: the lower index is the FBS
        arma::Mat<dtype> orig = {{0.5, 0.5, 1}, {0, 0, 1}, {1, 1, 1}}, dest = {{0.5, 0.5, -1}, {0, 0, -1}, {1, 1, -1}};
        quadriga_lib::ray_triangle_intersect<dtype>(&orig, &dest, &square, nullptr, nullptr, &no, &fi, &si,
                                                    nullptr, nullptr, kernel);
        for (arma::uword r = 0; r < 3; ++r)
        {
            CHECK(no(r) == 2);
            CHECK(fi(r) == 1);
            CHECK(si(r) == 2);
        }

        // Six hits at the same distance: FBS and SBS are the two lowest indices
        orig = {{0, 0, 1}}, dest = {{0, 0, -1}};
        arma::Mat<dtype> fbs, sbs;
        quadriga_lib::ray_triangle_intersect<dtype>(&orig, &dest, &fan, &fbs, &sbs, &no, &fi, &si, nullptr, nullptr, kernel);
        CHECK(no(0) == 6);
        CHECK(fi(0) == 3);
        CHECK(si(0) == 6);
        CHECK(fbs(0, 2) == dtype(0));
        CHECK(sbs(0, 2) == dtype(0));
    }
}

TEST_CASE("Ray-Triangle Intersect - Hits at equal distance - lowest index first")
{
    rti_equal_distance_hits<float>();
    rti_equal_distance_hits<double>();
}

TEST_CASE("Ray-Triangle Intersect - Rays in a plane of the bounding box")
{
    // Wall in the plane x = 1, bottom edge on z = 0 (= z_min of its AABB); the ray from (0,0,0) to (2,0,0) has
    // dz = 0, lies in the plane z = z_min and hits the midpoint of the bottom edge. Rotated: dy = 0 in y = y_min.
    // Last: the same wall, the ray has dz = -0 (destination z = -0.0).
    arma::mat wall_z = {{1, -1, 0, 1, 1, 0, 1, 0, 2}};
    arma::mat wall_y = {{1, 0, -1, 1, 0, 1, 1, 2, 0}};
    arma::mat big = {{-5, 5, -5, 5, -5, 5}}; // enlarged user AABB
    arma::mat orig = {{0, 0, 0}}, dest = {{2, 0, 0}}, dest_m0 = {{2, 0, -0.0}};

    for (int kernel : rti_kernels())
        for (int c = 0; c < 3; ++c)
        {
            INFO("use_kernel = " << kernel << ", case " << c);
            const arma::mat &mesh = (c == 1) ? wall_y : wall_z;
            const arma::mat &d = (c == 2) ? dest_m0 : dest;
            arma::u32_vec f1, f2;
            quadriga_lib::ray_triangle_intersect<double>(&orig, &d, &mesh, nullptr, nullptr, nullptr, &f1, nullptr,
                                                         nullptr, nullptr, kernel);
            quadriga_lib::ray_triangle_intersect<double>(&orig, &d, &mesh, nullptr, nullptr, nullptr, &f2, nullptr,
                                                         nullptr, &big, kernel);
            CHECK(f1(0) == 1);
            CHECK(f2(0) == 1);

            // Float input, computed AABB
            arma::fmat orig_f = arma::conv_to<arma::fmat>::from(orig), d_f = arma::conv_to<arma::fmat>::from(d);
            arma::fmat mesh_f = arma::conv_to<arma::fmat>::from(mesh);
            quadriga_lib::ray_triangle_intersect<float>(&orig_f, &d_f, &mesh_f, nullptr, nullptr, nullptr, &f1, nullptr,
                                                        nullptr, nullptr, kernel);
            CHECK(f1(0) == 1);
        }
}

TEST_CASE("Ray-Triangle Intersect - Double input far from the origin")
{
    // A 10 m cube and random 30 m rays around it, at the origin and at UTM-like offsets. The AVX2 kernel computes in
    // float; for double input its error must depend on the scene size only, not on the distance from the origin.
    if (!quadriga_lib::quadriga_lib_has_AVX2())
        return;

    arma::arma_rng::set_seed(11);
    arma::mat dir = arma::randn<arma::mat>(2000, 3);
    arma::mat start = arma::randu<arma::mat>(2000, 3) * 30.0 - 15.0;
    double max_err[3] = {0.0, 0.0, 0.0};
    int n_hit[3] = {0, 0, 0};
    const double offsets[3][2] = {{0.0, 0.0}, {5e6, 5e6}, {1e7, 1e7}};

    for (int io = 0; io < 3; ++io)
    {
        double ox = offsets[io][0], oy = offsets[io][1];
        arma::mat mesh = quadriga_lib::cube<double>({5.0}, {}, {ox, oy, 0.0});
        arma::mat orig(2000, 3), dest(2000, 3);
        for (arma::uword i = 0; i < 2000; ++i)
        {
            double l = std::sqrt(dir(i, 0) * dir(i, 0) + dir(i, 1) * dir(i, 1) + dir(i, 2) * dir(i, 2));
            for (arma::uword k = 0; k < 3; ++k)
                orig(i, k) = start(i, k), dest(i, k) = start(i, k) + 30.0 * dir(i, k) / l;
            orig(i, 0) += ox, orig(i, 1) += oy, dest(i, 0) += ox, dest(i, 1) += oy;
        }
        arma::mat fbs1, fbs2;
        arma::u32_vec fi1, fi2;
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, &fbs1, nullptr, nullptr, &fi1, nullptr, nullptr, nullptr, 1);
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, &fbs2, nullptr, nullptr, &fi2, nullptr, nullptr, nullptr, 2);
        for (arma::uword i = 0; i < 2000; ++i)
        {
            if (fi1(i) == 0 || fi2(i) == 0) // Rays at a face boundary may differ, the hit position is checked
                continue;
            ++n_hit[io];
            for (arma::uword k = 0; k < 3; ++k)
                max_err[io] = std::max(max_err[io], std::abs(fbs1(i, k) - fbs2(i, k)));
        }
        CHECK(arma::accu(fi1 != fi2) <= 4);
    }
    for (int io = 0; io < 3; ++io)
    {
        INFO("offset " << offsets[io][0] << ": " << n_hit[io] << " hits, max |FBS AVX2 - FBS GENERIC| = " << max_err[io] << " m");
        CHECK(n_hit[io] > 150);
        CHECK(max_err[io] < 1e-4);
    }
}

TEST_CASE("Ray-Triangle Intersect - Outputs that alias the inputs")
{
    // Planes z = 0 and z = -1; ray (0,0,1) -> (0,0,-2): FBS (0,0,0), SBS (0,0,-1)
    arma::mat mesh = {{-1, -1, 0, 1, -1, 0, 0, 1, 0}, {-1, -1, -1, 1, -1, -1, 0, 1, -1}};
    const arma::mat o0 = {{0, 0, 1}}, d0 = {{0, 0, -2}};

    for (int kernel : rti_kernels())
    {
        INFO("use_kernel = " << kernel);
        arma::mat fbs_ref, sbs_ref;
        quadriga_lib::ray_triangle_intersect<double>(&o0, &d0, &mesh, &fbs_ref, &sbs_ref, nullptr, nullptr, nullptr, nullptr, nullptr, kernel);
        CHECK(arma::approx_equal(fbs_ref, arma::mat({{0, 0, 0}}), "absdiff", 1e-6)); // AVX2 computes in float
        CHECK(arma::approx_equal(sbs_ref, arma::mat({{0, 0, -1}}), "absdiff", 1e-6));

        // fbs = orig, sbs = orig, fbs = dest, sbs = dest, fbs = orig and sbs = dest
        for (int c = 0; c < 5; ++c)
        {
            INFO("case " << c);
            arma::mat o = o0, d = d0, f, s;
            arma::mat *pf = (c == 0 || c == 4) ? &o : (c == 2 ? &d : &f);
            arma::mat *ps = (c == 1) ? &o : ((c == 3 || c == 4) ? &d : &s);
            quadriga_lib::ray_triangle_intersect<double>(&o, &d, &mesh, pf, ps, nullptr, nullptr, nullptr, nullptr, nullptr, kernel);
            CHECK(arma::approx_equal(*pf, fbs_ref, "absdiff", 0.0));
            CHECK(arma::approx_equal(*ps, sbs_ref, "absdiff", 0.0));
        }

        // orig and dest with 4 columns: writing fbs into orig or dest reallocates it ('dest' must have as many rows as
        // 'orig', so the extra column replaces the extra row this test used before)
        arma::mat o = {{0, 0, 1, 7}}, d = {{0, 0, -2, 5}}, s;
        quadriga_lib::ray_triangle_intersect<double>(&o, &d, &mesh, &o, &s, nullptr, nullptr, nullptr, nullptr, nullptr, kernel);
        CHECK(arma::approx_equal(o, fbs_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(s, sbs_ref, "absdiff", 0.0));
        o = {{0, 0, 1, 7}}, d = {{0, 0, -2, 5}};
        quadriga_lib::ray_triangle_intersect<double>(&o, &d, &mesh, &d, &o, nullptr, nullptr, nullptr, nullptr, nullptr, kernel);
        CHECK(arma::approx_equal(d, fbs_ref, "absdiff", 0.0));
        CHECK(arma::approx_equal(o, sbs_ref, "absdiff", 0.0));
    }
}

TEST_CASE("Ray-Triangle Intersect - Mesh with extra columns and row count of dest")
{
    // A mesh with more than 9 columns is accepted with and without 'aabb' (the first 9 columns are used); the
    // internal bounding box used to require exactly 9 columns. 'dest' must have as many rows as 'orig'.
    arma::mat mesh9 = quadriga_lib::cube<double>();
    arma::mat mesh10 = arma::join_rows(mesh9, arma::vec(mesh9.n_rows, arma::fill::value(1.0e6)));
    arma::mat orig = {{-10.0, 0.0, 0.5}, {-10.0, 5.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}, {10.0, 5.0, 0.5}};

    for (int kernel : {1, 2})
    {
        if (kernel == 2 && !quadriga_lib::quadriga_lib_has_AVX2())
            continue;
        INFO("kernel " << kernel);
        arma::mat fbs9, sbs9, fbs10, sbs10;
        arma::u32_vec no9, fi9, si9, no10, fi10, si10;
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh9, &fbs9, &sbs9, &no9, &fi9, &si9, nullptr, nullptr, kernel);
        CHECK(no9.at(0) == 2);
        CHECK(no9.at(1) == 0);

        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh10, &fbs10, &sbs10, &no10, &fi10, &si10, nullptr, nullptr, kernel);
        CHECK(arma::approx_equal(fbs10, fbs9, "absdiff", 0.0));
        CHECK(arma::approx_equal(sbs10, sbs9, "absdiff", 0.0));
        CHECK(arma::all(no10 == no9));
        CHECK(arma::all(fi10 == fi9));
        CHECK(arma::all(si10 == si9));

        arma::mat aabb = quadriga_lib::triangle_mesh_aabb(&mesh10);
        CHECK(arma::approx_equal(aabb, quadriga_lib::triangle_mesh_aabb(&mesh9), "absdiff", 0.0));
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh10, &fbs10, &sbs10, &no10, &fi10, &si10, nullptr, &aabb, kernel);
        CHECK(arma::all(fi10 == fi9));
    }

    arma::mat mesh8 = mesh9.cols(0, 7);
    CHECK_THROWS_AS(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh8), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::triangle_mesh_aabb(&mesh8), std::invalid_argument);

    arma::mat dest3 = arma::join_cols(dest, dest.row(0)), dest1 = dest.row(0);
    CHECK_THROWS_AS(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest3, &mesh9), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest1, &mesh9), std::invalid_argument);
}
