// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_lib.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// --- Helper: Check that the CSR outputs and the split form agree ---
// Verifies the structural contract of ray_point_intersect:
//   hit_offset[0] == 0, monotone, hit_offset[n_point] == hit_index.size()
//   hit_count[i]  == hit_offset[i+1] - hit_offset[i]
//   hits_per_point[i] is exactly the hit_index block of point i
// 'check_ascending' additionally requires the ray indices within one point's
// block to be strictly ascending, which holds for GENERIC and AVX2 but not for
// CUDA, where a point's list is grouped by batch.
static bool csr_is_consistent(const std::vector<unsigned> &hit_index,
                              const arma::u32_vec &hit_offset,
                              const arma::u32_vec &hit_count,
                              const std::vector<arma::u32_vec> &hits_per_point,
                              size_t n_point, size_t n_ray, bool check_ascending)
{
    if (hit_offset.n_elem != n_point + 1)
        return false;
    if (hit_offset.at(0) != 0u)
        return false;
    if ((size_t)hit_offset.at(n_point) != hit_index.size())
        return false;
    if (hit_count.n_elem != n_point)
        return false;
    if (hits_per_point.size() != n_point)
        return false;

    for (size_t i_point = 0; i_point < n_point; ++i_point)
    {
        unsigned first = hit_offset.at(i_point);
        unsigned last = hit_offset.at(i_point + 1);

        if (last < first)
            return false;
        if (hit_count.at(i_point) != last - first)
            return false;
        if ((size_t)hits_per_point[i_point].n_elem != (size_t)(last - first))
            return false;

        for (unsigned k = first; k < last; ++k)
        {
            if (hit_index[k] >= (unsigned)n_ray)
                return false;
            if (hits_per_point[i_point].at(k - first) != hit_index[k])
                return false;
            if (check_ascending && k > first && hit_index[k] <= hit_index[k - 1])
                return false;
        }
    }
    return true;
}

// --- Helper: Build a single parallel ray tube pointing in +z ---
// Equilateral triangle of circumradius R in the z = 0 plane with all three
// vertex rays pointing in +z, so the tube cross-section stays constant.
static void make_parallel_tube(float R, arma::fmat &orig, arma::fmat &trivec, arma::fmat &tridir)
{
    orig.zeros(1, 3);
    trivec.zeros(1, 9);
    tridir.zeros(1, 9);

    const float s32 = R * std::sqrt(3.0f) / 2.0f;

    trivec.at(0, 0) = R;
    trivec.at(0, 3) = -0.5f * R;
    trivec.at(0, 4) = s32;
    trivec.at(0, 6) = -0.5f * R;
    trivec.at(0, 7) = -s32;

    tridir.at(0, 2) = 1.0f;
    tridir.at(0, 5) = 1.0f;
    tridir.at(0, 8) = 1.0f;
}

TEST_CASE("Ray-Point Intersect - Simple Mode")
{
    // Generate set of points
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    // Create a sub-cloud index
    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index, reverse_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8, nullptr, &reverse_index);

    // Generate a set of ray beams
    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);

    // Change the location
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    // Call intersect
    arma::u32_vec hit_count;
    std::vector<arma::u32_vec> ind;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir, nullptr, nullptr, &hit_count, &ind, &sub_cloud_index);
    CHECK(arma::all(hit_count == 1));
}

TEST_CASE("Ray-Point Intersect - Ray Subdivision")
{
    double res = 0.1;
    arma::vec rx_pos = {-10, -10, 0.1}; // Lower left point
    arma::vec rx_xy = {20, 20};         // x-y scale

    // Generate x and y vectors
    arma::vec x = arma::regspace(rx_pos(0), res, rx_pos(0) + rx_xy(0));
    arma::vec y = arma::regspace(rx_pos(1), res, rx_pos(1) + rx_xy(1));

    // Create meshgrid
    arma::mat X(y.n_elem, x.n_elem);
    arma::mat Y(y.n_elem, x.n_elem);

    for (arma::uword i = 0; i < y.n_elem; ++i)
        X.row(i) = x.t();

    for (arma::uword j = 0; j < x.n_elem; ++j)
        Y.col(j) = y;

    // Flatten the meshgrid and create the points matrix
    arma::mat points(X.n_elem, 3);
    points.col(0) = arma::vectorise(X);
    points.col(1) = arma::vectorise(Y);
    points.col(2).fill(rx_pos(2));

    // Generate a set of ray beams
    arma::mat orig, trivec, tridir;
    quadriga_lib::icosphere<double>(12, 1.0, &orig, nullptr, &trivec, &tridir, true);

    // Change the location
    orig.col(0) -= 10.0;
    orig.col(1) -= 20.0;
    orig.col(2) -= 30.0;

    // Call intersect
    arma::u32_vec hit_count;
    std::vector<arma::u32_vec> ind;
    quadriga_lib::ray_point_intersect<double>(points, orig, trivec, tridir, nullptr, nullptr, &hit_count, &ind);
    CHECK(hit_count.n_elem == points.n_rows);
    CHECK(arma::all(hit_count == 1));

    // Subdivide all rays
    arma::mat origN, trivecN, tridirN;
    quadriga_lib::subdivide_rays<double>(orig, trivec, tridir, nullptr, &origN, &trivecN, &tridirN);
    CHECK(origN.n_rows == 4 * orig.n_rows);

    // Call intersect
    hit_count.reset();
    quadriga_lib::ray_point_intersect<double>(points, origN, trivecN, tridirN, nullptr, nullptr, &hit_count);
    CHECK(arma::all(hit_count == 1));

    // Subdivide selected rays
    arma::u32_vec index(points.n_rows);
    for (arma::uword i = 0; i < points.n_rows; ++i)
        index.at(i) = ind[i].at(0);

    index = arma::unique(index);
    CHECK(index.n_elem < 4 * orig.n_rows);

    origN.reset();
    trivecN.reset();
    tridirN.reset();
    quadriga_lib::subdivide_rays<double>(orig, trivec, tridir, nullptr, &origN, &trivecN, &tridirN, nullptr, &index);
    CHECK(origN.n_rows == 4 * index.n_elem);

    // Call intersect
    hit_count.reset();
    quadriga_lib::ray_point_intersect<double>(points, origN, trivecN, tridirN, nullptr, nullptr, &hit_count);
    CHECK(arma::all(hit_count == 1));
}
TEST_CASE("Ray-Point Intersect - CSR Outputs")
{
    // Same geometry as the simple mode test, but requesting all four outputs
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);

    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset, hit_count;
    std::vector<arma::u32_vec> hits_per_point;

    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      &hit_index, &hit_offset, &hit_count, &hits_per_point,
                                      &sub_cloud_index, 1);

    CHECK(csr_is_consistent(hit_index, hit_offset, hit_count, hits_per_point,
                            pointsR.n_rows, orig.n_rows, true));
    CHECK(arma::all(hit_count == 1));
    CHECK(hit_index.size() == pointsR.n_rows);
    CHECK(hit_offset.n_elem == pointsR.n_rows + 1);
}

TEST_CASE("Ray-Point Intersect - Optional Outputs")
{
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);

    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    // Reference run with everything requested
    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset, hit_count;
    std::vector<arma::u32_vec> hits_per_point;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      &hit_index, &hit_offset, &hit_count, &hits_per_point,
                                      &sub_cloud_index, 1);

    // Flat list only
    std::vector<unsigned> idx_only;
    arma::u32_vec off_only;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      &idx_only, &off_only, nullptr, nullptr, &sub_cloud_index, 1);
    CHECK(idx_only == hit_index);
    CHECK(off_only.n_elem == hit_offset.n_elem);
    CHECK(arma::all(off_only == hit_offset));

    // Hit counter only
    arma::u32_vec cnt_only;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      nullptr, nullptr, &cnt_only, nullptr, &sub_cloud_index, 1);
    CHECK(cnt_only.n_elem == hit_count.n_elem);
    CHECK(arma::all(cnt_only == hit_count));

    // Split form only
    std::vector<arma::u32_vec> pp_only;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      nullptr, nullptr, nullptr, &pp_only, &sub_cloud_index, 1);
    REQUIRE(pp_only.size() == hits_per_point.size());
    bool same = true;
    for (size_t i = 0; i < pp_only.size(); ++i)
    {
        same &= (pp_only[i].n_elem == hits_per_point[i].n_elem);
        for (arma::uword k = 0; same && k < pp_only[i].n_elem; ++k)
            same &= (pp_only[i].at(k) == hits_per_point[i].at(k));
    }
    CHECK(same);

    // Reusing a populated output container must not leave stale entries behind
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      &idx_only, &off_only, &cnt_only, &pp_only, &sub_cloud_index, 1);
    CHECK(idx_only == hit_index);
    CHECK(pp_only.size() == hits_per_point.size());

    // Nothing requested at all must still be well-formed
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      nullptr, nullptr, nullptr, nullptr, &sub_cloud_index, 1);
}

TEST_CASE("Ray-Point Intersect - Kernel Equivalence")
{
    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(0.0f, 0.1f, 0.3f);
    points = repmat(points, 2, 1);
    points.submat(4, 0, 7, 0) += 40.0f;
    points = repmat(points, 2, 1);
    points.submat(0, 1, 7, 1) -= 50.0f;
    points.submat(8, 1, 15, 1) += 50.0f;
    points.col(2) += 1.0f;
    points.col(1) *= 0.1f;

    arma::fmat pointsR;
    arma::u32_vec sub_cloud_index;
    quadriga_lib::point_cloud_segmentation(&points, &pointsR, &sub_cloud_index, 4, 8);

    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    std::vector<unsigned> idx_gen;
    arma::u32_vec off_gen, cnt_gen;
    quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                      &idx_gen, &off_gen, &cnt_gen, nullptr, &sub_cloud_index, 1);
    REQUIRE(idx_gen.size() > 0);

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<unsigned> idx_avx2;
        arma::u32_vec off_avx2, cnt_avx2;
        quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                          &idx_avx2, &off_avx2, &cnt_avx2, nullptr, &sub_cloud_index, 2);
        CHECK(idx_avx2 == idx_gen);
        CHECK(arma::all(off_avx2 == off_gen));
        CHECK(arma::all(cnt_avx2 == cnt_gen));
    }

    if (quadriga_lib::quadriga_lib_has_CUDA())
    {
        std::vector<unsigned> idx_cuda;
        arma::u32_vec off_cuda, cnt_cuda;
        quadriga_lib::ray_point_intersect(pointsR, orig, trivec, tridir,
                                          &idx_cuda, &off_cuda, &cnt_cuda, nullptr, &sub_cloud_index, 3);
        CHECK(arma::all(cnt_cuda == cnt_gen));
        CHECK(arma::all(off_cuda == off_gen));

        // CUDA groups a point's ray list by batch, so compare after sorting
        bool same = true;
        for (arma::uword i = 0; i < cnt_gen.n_elem; ++i)
        {
            std::vector<unsigned> a(idx_gen.begin() + off_gen.at(i), idx_gen.begin() + off_gen.at(i + 1));
            std::vector<unsigned> b(idx_cuda.begin() + off_cuda.at(i), idx_cuda.begin() + off_cuda.at(i + 1));
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            same &= (a == b);
        }
        CHECK(same);
    }
}

TEST_CASE("Ray-Point Intersect - Unaligned Point Count")
{
    // 13 points, deliberately not a multiple of the AVX2 vector size, so the
    // wrapper pads the point buffers and the kernel must discard the padding
    arma::fmat points(13, 3);
    for (arma::uword i = 0; i < points.n_rows; ++i)
    {
        points.at(i, 0) = 1.0f + 0.5f * (float)i;
        points.at(i, 1) = -2.0f + 0.3f * (float)i;
        points.at(i, 2) = 3.0f - 0.2f * (float)i;
    }

    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(2, 1.0f, &orig, nullptr, &trivec, &tridir, true);
    orig.col(0) -= 10.0f;
    orig.col(1) -= 20.0f;
    orig.col(2) -= 30.0f;

    std::vector<unsigned> idx_gen;
    arma::u32_vec off_gen, cnt_gen;
    std::vector<arma::u32_vec> pp_gen;
    quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                      &idx_gen, &off_gen, &cnt_gen, &pp_gen, nullptr, 1);

    CHECK(off_gen.n_elem == points.n_rows + 1);
    CHECK(csr_is_consistent(idx_gen, off_gen, cnt_gen, pp_gen, points.n_rows, orig.n_rows, true));
    CHECK(arma::all(cnt_gen >= 1)); // the icosphere tiles the sphere

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<unsigned> idx_avx2;
        arma::u32_vec off_avx2, cnt_avx2;
        std::vector<arma::u32_vec> pp_avx2;
        quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                          &idx_avx2, &off_avx2, &cnt_avx2, &pp_avx2, nullptr, 2);

        CHECK(off_avx2.n_elem == points.n_rows + 1);
        CHECK(csr_is_consistent(idx_avx2, off_avx2, cnt_avx2, pp_avx2, points.n_rows, orig.n_rows, true));
        CHECK(idx_avx2 == idx_gen);
        CHECK(arma::all(off_avx2 == off_gen));
    }
}

TEST_CASE("Ray-Point Intersect - No Hits")
{
    // A narrow tube along +z with the points far off to the side. The AABB
    // pre-filter rejects the whole sub-cloud, so this covers the empty CSR
    // path: an all-zero offset array and empty per-point lists.
    arma::fmat orig, trivec, tridir;
    make_parallel_tube(0.5f, orig, trivec, tridir);

    arma::fmat points(4, 3);
    points.col(0) = arma::regspace<arma::fvec>(1000.0f, 1.0f, 1003.0f);
    points.col(1).fill(1000.0f);
    points.col(2).fill(1000.0f);

    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset, hit_count;
    std::vector<arma::u32_vec> hits_per_point;

    quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                      &hit_index, &hit_offset, &hit_count, &hits_per_point, nullptr, 1);

    CHECK(hit_index.empty());
    CHECK(hit_offset.n_elem == points.n_rows + 1);
    CHECK(arma::all(hit_offset == 0));
    CHECK(arma::all(hit_count == 0));
    REQUIRE(hits_per_point.size() == points.n_rows);
    bool all_empty = true;
    for (arma::uword i = 0; i < points.n_rows; ++i)
        all_empty &= (hits_per_point[i].n_elem == 0);
    CHECK(all_empty);

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<unsigned> idx_avx2;
        arma::u32_vec off_avx2, cnt_avx2;
        quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                          &idx_avx2, &off_avx2, &cnt_avx2, nullptr, nullptr, 2);
        CHECK(idx_avx2.empty());
        CHECK(arma::all(cnt_avx2 == 0));
    }
}

TEST_CASE("Ray-Point Intersect - Padding Discard")
{
    // 13 points inside a wide tube. The wrapper pads the point buffers up to a
    // multiple of the AVX2 vector size and the padding sits at the origin,
    // which is inside this tube, so the padded lanes do produce a geometric
    // hit and the kernel has to drop them. Getting this wrong shows up as
    // extra hits or as a longer offset array.
    arma::fmat orig, trivec, tridir;
    make_parallel_tube(50.0f, orig, trivec, tridir);

    const arma::uword n_point = 13;
    arma::fmat points(n_point, 3);
    for (arma::uword i = 0; i < n_point; ++i)
    {
        points.at(i, 0) = -3.0f + 0.5f * (float)i;
        points.at(i, 1) = -2.0f + 0.3f * (float)i;
        points.at(i, 2) = 1.0f;
    }

    std::vector<unsigned> idx_gen;
    arma::u32_vec off_gen, cnt_gen;
    std::vector<arma::u32_vec> pp_gen;
    quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                      &idx_gen, &off_gen, &cnt_gen, &pp_gen, nullptr, 1);

    CHECK(idx_gen.size() == n_point);
    CHECK(arma::all(cnt_gen == 1));
    CHECK(csr_is_consistent(idx_gen, off_gen, cnt_gen, pp_gen, n_point, orig.n_rows, true));

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<unsigned> idx_avx2;
        arma::u32_vec off_avx2, cnt_avx2;
        std::vector<arma::u32_vec> pp_avx2;
        quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                          &idx_avx2, &off_avx2, &cnt_avx2, &pp_avx2, nullptr, 2);

        CHECK(idx_avx2.size() == n_point);
        CHECK(off_avx2.n_elem == n_point + 1);
        CHECK(arma::all(cnt_avx2 == 1));
        CHECK(idx_avx2 == idx_gen);
        CHECK(csr_is_consistent(idx_avx2, off_avx2, cnt_avx2, pp_avx2, n_point, orig.n_rows, true));
    }
}

TEST_CASE("Ray-Point Intersect - Large Hit Count")
{
    // A wide parallel tube over a dense grid. All hits come from a single ray,
    // so one worker stages them all and the internal staging chunk size is
    // crossed several times. The point count is not a multiple of 8 either,
    // so the AVX2 run additionally exercises the padding discard at scale.
    const float res = 0.05f;
    arma::fvec ax = arma::regspace<arma::fvec>(-5.0f, res, 5.0f);
    const arma::uword n_side = ax.n_elem;

    arma::fmat points(n_side * n_side, 3);
    for (arma::uword j = 0; j < n_side; ++j)
        for (arma::uword i = 0; i < n_side; ++i)
        {
            points.at(j * n_side + i, 0) = ax.at(i);
            points.at(j * n_side + i, 1) = ax.at(j);
            points.at(j * n_side + i, 2) = 1.0f;
        }

    REQUIRE(points.n_rows > 16384); // more than one staging chunk

    arma::fmat orig, trivec, tridir;
    make_parallel_tube(50.0f, orig, trivec, tridir);

    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset, hit_count;
    std::vector<arma::u32_vec> hits_per_point;

    quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                      &hit_index, &hit_offset, &hit_count, &hits_per_point, nullptr, 1);

    CHECK(hit_index.size() == points.n_rows);
    CHECK(arma::all(hit_count == 1));
    CHECK(csr_is_consistent(hit_index, hit_offset, hit_count, hits_per_point,
                            points.n_rows, orig.n_rows, true));

    bool single_ray = true;
    for (size_t i = 0; i < hit_index.size(); ++i)
        single_ray &= (hit_index[i] == 0u);
    CHECK(single_ray);

    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        std::vector<unsigned> idx_avx2;
        arma::u32_vec off_avx2, cnt_avx2;
        quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                          &idx_avx2, &off_avx2, &cnt_avx2, nullptr, nullptr, 2);
        CHECK(idx_avx2.size() == points.n_rows);
        CHECK(idx_avx2 == hit_index);
        CHECK(arma::all(off_avx2 == hit_offset));
    }
}

TEST_CASE("Ray-Point Intersect - Input Validation")
{
    arma::fmat points(8, 3, arma::fill::ones);

    arma::fmat orig, trivec, tridir;
    quadriga_lib::icosphere<float>(1, 1.0f, &orig, nullptr, &trivec, &tridir, true);

    arma::fmat empty;
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(empty, orig, trivec, tridir), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, empty, trivec, tridir), std::invalid_argument);

    arma::fmat bad_cols(8, 2, arma::fill::ones);
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(bad_cols, orig, trivec, tridir), std::invalid_argument);

    arma::fmat bad_trivec(orig.n_rows, 3, arma::fill::ones);
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, bad_trivec, tridir), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, bad_trivec), std::invalid_argument);

    arma::fmat short_trivec = trivec.rows(0, trivec.n_rows - 2);
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, short_trivec, tridir), std::invalid_argument);

    // Sub-cloud index must start at 0, ascend strictly and stay in range
    arma::u32_vec bad_start(2);
    bad_start.at(0) = 1, bad_start.at(1) = 4;
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                                      nullptr, nullptr, nullptr, nullptr, &bad_start, 1),
                    std::invalid_argument);

    arma::u32_vec bad_order(2);
    bad_order.at(0) = 0, bad_order.at(1) = 0;
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                                      nullptr, nullptr, nullptr, nullptr, &bad_order, 1),
                    std::invalid_argument);

    arma::u32_vec too_large(2);
    too_large.at(0) = 0, too_large.at(1) = 99;
    CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                                      nullptr, nullptr, nullptr, nullptr, &too_large, 1),
                    std::invalid_argument);

    // Sub-clouds must be aligned to the SIMD vector size for the AVX2 kernel
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        arma::u32_vec misaligned(2);
        misaligned.at(0) = 0, misaligned.at(1) = 3;
        CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                                          nullptr, nullptr, nullptr, nullptr, &misaligned, 2),
                        std::invalid_argument);
    }

    // Requesting a kernel that is not available must be rejected
    if (!quadriga_lib::quadriga_lib_has_CUDA())
        CHECK_THROWS_AS(quadriga_lib::ray_point_intersect(points, orig, trivec, tridir,
                                                          nullptr, nullptr, nullptr, nullptr, nullptr, 3),
                        std::invalid_argument);
}