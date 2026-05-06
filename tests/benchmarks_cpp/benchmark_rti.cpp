// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 \
//     -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src \
//     -o bench_rti benchmark_rti.cpp /sjc/quadriga-lib/lib/libquadriga.a -lcudart

#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"
#include "quadriga_lib_avx2_functions.hpp"
#include "quadriga_lib_cuda_functions.hpp"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ULP distance between two floats
static int32_t ulp_dist(float a, float b)
{
    if (std::isnan(a) || std::isnan(b))
        return INT32_MAX;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    int32_t d = ia - ib;
    return d < 0 ? -d : d;
}

// ============================================================================
// Scene: holds SoA mesh data + sub-mesh info + AABBs
// ============================================================================
struct Scene
{
    size_t n_mesh;      // total triangles (padded to vec_size)
    size_t n_sub;       // number of sub-meshes
    size_t n_sub_s;     // n_sub rounded up to multiple of 8

    // SoA mesh arrays, length n_mesh
    std::vector<float> Tx, Ty, Tz;
    std::vector<float> E1x, E1y, E1z;
    std::vector<float> E2x, E2y, E2z;

    // Double-precision copies for GENERIC<double>
    std::vector<double> dTx, dTy, dTz;
    std::vector<double> dE1x, dE1y, dE1z;
    std::vector<double> dE2x, dE2y, dE2z;

    // Sub-mesh index, length n_sub
    std::vector<unsigned> SMI;

    // AABB arrays, length n_sub_s (padded)
    std::vector<float> Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    std::vector<double> dXmin, dXmax, dYmin, dYmax, dZmin, dZmax;
};

// Build a scene from multiple icospheres scattered around the origin
// n_spheres:        number of icospheres to place
// n_div:            subdivision level per sphere (faces = 20 * n_div^2)
// sphere_radius:    radius of each icosphere
// placement_radius: distance from origin to sphere centres
// target_size:      target sub-mesh size for segmentation
// vec_size:         SIMD alignment (8 for AVX2)
static Scene build_scene(int n_spheres, arma::uword n_div, float sphere_radius,
                          float placement_radius, arma::uword target_size, arma::uword vec_size)
{
    Scene sc;

    // 1. Generate one template icosphere at the origin
    //    center: [n_faces, 3] - pointing vector from origin to triangle center
    //    vert:   [n_faces, 9] - vectors from center to each of the 3 vertices
    //    Absolute vertex = center + vert
    arma::fmat center_tpl, vert_tpl;
    arma::uword faces_per_sphere = quadriga_lib::icosphere(n_div, sphere_radius, &center_tpl,
                                                            (arma::fvec *)nullptr, &vert_tpl);

    arma::uword total_faces = (arma::uword)n_spheres * faces_per_sphere;
    printf("  %d icospheres, n_div=%lu, %lu faces each, %lu total\n",
           n_spheres, (unsigned long)n_div, (unsigned long)faces_per_sphere, (unsigned long)total_faces);

    // 2. Place spheres at random positions on a shell of radius placement_radius
    //    Use a deterministic seed for reproducibility
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    arma::fmat mesh(total_faces, 9);
    arma::uword row = 0;

    for (int s = 0; s < n_spheres; ++s)
    {
        // Uniform random point on sphere
        float theta = 2.0f * (float)M_PI * u01(rng);
        float phi = std::acos(1.0f - 2.0f * u01(rng));
        float cx = placement_radius * std::sin(phi) * std::cos(theta);
        float cy = placement_radius * std::sin(phi) * std::sin(theta);
        float cz = placement_radius * std::cos(phi);

        // Copy template: absolute vertex = center + vert_offset + placement
        for (arma::uword f = 0; f < faces_per_sphere; ++f, ++row)
        {
            float fc_x = center_tpl(f, 0), fc_y = center_tpl(f, 1), fc_z = center_tpl(f, 2);
            mesh(row, 0) = fc_x + vert_tpl(f, 0) + cx;  // v1x
            mesh(row, 1) = fc_y + vert_tpl(f, 1) + cy;  // v1y
            mesh(row, 2) = fc_z + vert_tpl(f, 2) + cz;  // v1z
            mesh(row, 3) = fc_x + vert_tpl(f, 3) + cx;  // v2x
            mesh(row, 4) = fc_y + vert_tpl(f, 4) + cy;  // v2y
            mesh(row, 5) = fc_z + vert_tpl(f, 5) + cz;  // v2z
            mesh(row, 6) = fc_x + vert_tpl(f, 6) + cx;  // v3x
            mesh(row, 7) = fc_y + vert_tpl(f, 7) + cy;  // v3y
            mesh(row, 8) = fc_z + vert_tpl(f, 8) + cz;  // v3z
        }
    }

    // 3. Segment the combined mesh
    arma::fmat meshR;
    arma::u32_vec sub_mesh_index;
    arma::uword n_sub = quadriga_lib::triangle_mesh_segmentation(&mesh, &meshR, &sub_mesh_index,
                                                                  target_size, vec_size);

    sc.n_mesh = meshR.n_rows;
    sc.n_sub = n_sub;
    sc.n_sub_s = ((n_sub + 7) / 8) * 8; // round up to multiple of 8

    printf("  Segmented: n_meshR=%zu, n_sub=%zu, n_sub_s=%zu\n", sc.n_mesh, sc.n_sub, sc.n_sub_s);

    // 3. Convert meshR rows to SoA (vertex + edge vectors)
    sc.Tx.resize(sc.n_mesh);  sc.Ty.resize(sc.n_mesh);  sc.Tz.resize(sc.n_mesh);
    sc.E1x.resize(sc.n_mesh); sc.E1y.resize(sc.n_mesh); sc.E1z.resize(sc.n_mesh);
    sc.E2x.resize(sc.n_mesh); sc.E2y.resize(sc.n_mesh); sc.E2z.resize(sc.n_mesh);

    for (size_t i = 0; i < sc.n_mesh; ++i)
    {
        float v1x = meshR(i, 0), v1y = meshR(i, 1), v1z = meshR(i, 2);
        float v2x = meshR(i, 3), v2y = meshR(i, 4), v2z = meshR(i, 5);
        float v3x = meshR(i, 6), v3y = meshR(i, 7), v3z = meshR(i, 8);
        sc.Tx[i] = v1x;  sc.Ty[i] = v1y;  sc.Tz[i] = v1z;
        sc.E1x[i] = v2x - v1x; sc.E1y[i] = v2y - v1y; sc.E1z[i] = v2z - v1z;
        sc.E2x[i] = v3x - v1x; sc.E2y[i] = v3y - v1y; sc.E2z[i] = v3z - v1z;
    }

    // 4. Double-precision copies
    sc.dTx.resize(sc.n_mesh);  sc.dTy.resize(sc.n_mesh);  sc.dTz.resize(sc.n_mesh);
    sc.dE1x.resize(sc.n_mesh); sc.dE1y.resize(sc.n_mesh); sc.dE1z.resize(sc.n_mesh);
    sc.dE2x.resize(sc.n_mesh); sc.dE2y.resize(sc.n_mesh); sc.dE2z.resize(sc.n_mesh);
    for (size_t i = 0; i < sc.n_mesh; ++i)
    {
        sc.dTx[i] = sc.Tx[i];  sc.dTy[i] = sc.Ty[i];  sc.dTz[i] = sc.Tz[i];
        sc.dE1x[i] = sc.E1x[i]; sc.dE1y[i] = sc.E1y[i]; sc.dE1z[i] = sc.E1z[i];
        sc.dE2x[i] = sc.E2x[i]; sc.dE2y[i] = sc.E2y[i]; sc.dE2z[i] = sc.E2z[i];
    }

    // 5. Sub-mesh index
    sc.SMI.resize(sc.n_sub);
    for (size_t i = 0; i < sc.n_sub; ++i)
        sc.SMI[i] = sub_mesh_index(i);

    // 6. Compute AABBs per sub-mesh
    sc.Xmin.resize(sc.n_sub_s, 1e20f);  sc.Xmax.resize(sc.n_sub_s, -1e20f);
    sc.Ymin.resize(sc.n_sub_s, 1e20f);  sc.Ymax.resize(sc.n_sub_s, -1e20f);
    sc.Zmin.resize(sc.n_sub_s, 1e20f);  sc.Zmax.resize(sc.n_sub_s, -1e20f);

    for (size_t s = 0; s < sc.n_sub; ++s)
    {
        size_t i_start = sc.SMI[s];
        size_t i_end = (s + 1 < sc.n_sub) ? sc.SMI[s + 1] : sc.n_mesh;

        float xmin = 1e20f, xmax = -1e20f;
        float ymin = 1e20f, ymax = -1e20f;
        float zmin = 1e20f, zmax = -1e20f;

        for (size_t i = i_start; i < i_end; ++i)
        {
            // Three vertices: v1, v1+E1, v1+E2
            float v1x = sc.Tx[i], v1y = sc.Ty[i], v1z = sc.Tz[i];
            float v2x = v1x + sc.E1x[i], v2y = v1y + sc.E1y[i], v2z = v1z + sc.E1z[i];
            float v3x = v1x + sc.E2x[i], v3y = v1y + sc.E2y[i], v3z = v1z + sc.E2z[i];

            xmin = std::min({xmin, v1x, v2x, v3x});
            xmax = std::max({xmax, v1x, v2x, v3x});
            ymin = std::min({ymin, v1y, v2y, v3y});
            ymax = std::max({ymax, v1y, v2y, v3y});
            zmin = std::min({zmin, v1z, v2z, v3z});
            zmax = std::max({zmax, v1z, v2z, v3z});
        }

        sc.Xmin[s] = xmin; sc.Xmax[s] = xmax;
        sc.Ymin[s] = ymin; sc.Ymax[s] = ymax;
        sc.Zmin[s] = zmin; sc.Zmax[s] = zmax;
    }

    // Double-precision AABB copies
    sc.dXmin.resize(sc.n_sub_s); sc.dXmax.resize(sc.n_sub_s);
    sc.dYmin.resize(sc.n_sub_s); sc.dYmax.resize(sc.n_sub_s);
    sc.dZmin.resize(sc.n_sub_s); sc.dZmax.resize(sc.n_sub_s);
    for (size_t i = 0; i < sc.n_sub_s; ++i)
    {
        sc.dXmin[i] = sc.Xmin[i]; sc.dXmax[i] = sc.Xmax[i];
        sc.dYmin[i] = sc.Ymin[i]; sc.dYmax[i] = sc.Ymax[i];
        sc.dZmin[i] = sc.Zmin[i]; sc.dZmax[i] = sc.Zmax[i];
    }

    return sc;
}

// ============================================================================
// Ray set: SoA arrays for origin + direction
// ============================================================================
struct Rays
{
    size_t n_ray;
    std::vector<float> Ox, Oy, Oz, Dx, Dy, Dz;
    std::vector<double> dOx, dOy, dOz, dDx, dDy, dDz;
};

// Generate rays from the origin, shooting in random directions
// dest_distance: how far the ray destination is from origin
static Rays generate_rays(size_t n_ray, float dest_distance, unsigned seed = 42)
{
    Rays r;
    r.n_ray = n_ray;
    r.Ox.resize(n_ray); r.Oy.resize(n_ray); r.Oz.resize(n_ray);
    r.Dx.resize(n_ray); r.Dy.resize(n_ray); r.Dz.resize(n_ray);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (size_t i = 0; i < n_ray; ++i)
    {
        // Origin at (0,0,0)
        r.Ox[i] = 0.0f;
        r.Oy[i] = 0.0f;
        r.Oz[i] = 0.0f;

        // Random direction (uniform on unit sphere)
        float theta = 2.0f * (float)M_PI * u01(rng);
        float phi = std::acos(1.0f - 2.0f * u01(rng));
        r.Dx[i] = dest_distance * std::sin(phi) * std::cos(theta);
        r.Dy[i] = dest_distance * std::sin(phi) * std::sin(theta);
        r.Dz[i] = dest_distance * std::cos(phi);
    }

    // Double copies
    r.dOx.resize(n_ray); r.dOy.resize(n_ray); r.dOz.resize(n_ray);
    r.dDx.resize(n_ray); r.dDy.resize(n_ray); r.dDz.resize(n_ray);
    for (size_t i = 0; i < n_ray; ++i)
    {
        r.dOx[i] = r.Ox[i]; r.dOy[i] = r.Oy[i]; r.dOz[i] = r.Oz[i];
        r.dDx[i] = r.Dx[i]; r.dDy[i] = r.Dy[i]; r.dDz[i] = r.Dz[i];
    }

    return r;
}

// ============================================================================
// Result container
// ============================================================================
struct RTI_Result
{
    size_t n_ray;
    std::vector<float> Wf, Ws;
    std::vector<double> dWf, dWs; // for generic-double
    std::vector<unsigned> If, Is, hit_cnt;

    void alloc_float(size_t n)
    {
        n_ray = n;
        Wf.resize(n); Ws.resize(n);
        If.resize(n); Is.resize(n); hit_cnt.resize(n);
    }
    void alloc_double(size_t n)
    {
        n_ray = n;
        dWf.resize(n); dWs.resize(n);
        If.resize(n); Is.resize(n); hit_cnt.resize(n);
    }
};

// ============================================================================
// PERFORMANCE
// ============================================================================
static void run_performance(const Scene &sc, const Rays &rays, const char *label,
                            bool run_generic, int n_reps)
{
    const size_t n = rays.n_ray;

    printf("%-12s | n_mesh=%7zu, n_sub=%4zu | n_ray=%10zu | ", label, sc.n_mesh, sc.n_sub, n);

    // --- Generic float ---
    double dt_gen_f = 0.0;
    if (run_generic)
    {
        RTI_Result res;
        res.alloc_float(n);
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
            qd_RTI_GENERIC(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                           sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                           sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                           sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                           sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                           rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                           rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                           res.Wf.data(), res.Ws.data(), res.If.data(), res.Is.data(), res.hit_cnt.data());
        dt_gen_f = (now_ms() - t0) / n_reps;
        printf("Gen/f %9.1f ms | ", dt_gen_f);
    }
    else
        printf("Gen/f  %9s   | ", "skip");

    // --- Generic double ---
    double dt_gen_d = 0.0;
    if (run_generic)
    {
        RTI_Result res;
        res.alloc_double(n);
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
            qd_RTI_GENERIC(sc.dTx.data(), sc.dTy.data(), sc.dTz.data(),
                           sc.dE1x.data(), sc.dE1y.data(), sc.dE1z.data(),
                           sc.dE2x.data(), sc.dE2y.data(), sc.dE2z.data(), sc.n_mesh,
                           sc.SMI.data(), sc.dXmin.data(), sc.dXmax.data(),
                           sc.dYmin.data(), sc.dYmax.data(), sc.dZmin.data(), sc.dZmax.data(), sc.n_sub,
                           rays.dOx.data(), rays.dOy.data(), rays.dOz.data(),
                           rays.dDx.data(), rays.dDy.data(), rays.dDz.data(), n,
                           res.dWf.data(), res.dWs.data(), res.If.data(), res.Is.data(), res.hit_cnt.data());
        dt_gen_d = (now_ms() - t0) / n_reps;
        printf("Gen/d %9.1f ms | ", dt_gen_d);
    }
    else
        printf("Gen/d  %9s   | ", "skip");

    // --- AVX2 ---
    double dt_avx = 0.0;
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        RTI_Result res;
        res.alloc_float(n);
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
            qd_RTI_AVX2(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                         sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                         sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                         sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                         sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                         rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                         rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                         res.Wf.data(), res.Ws.data(), res.If.data(), res.Is.data(), res.hit_cnt.data());
        dt_avx = (now_ms() - t0) / n_reps;
        printf("AVX2 %9.1f ms | ", dt_avx);
    }
    else
        printf("AVX2 %9s   | ", "N/A");

    // --- CUDA ---
    double dt_cuda = 0.0;
    if (quadriga_lib::quadriga_lib_has_CUDA())
    {
        RTI_Result res;
        res.alloc_float(n);
        double t0 = now_ms();
        for (int r = 0; r < n_reps; ++r)
            qd_RTI_CUDA(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                         sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                         sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                         sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                         sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                         rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                         rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                         res.Wf.data(), res.Ws.data(), res.If.data(), res.Is.data(), res.hit_cnt.data());
        dt_cuda = (now_ms() - t0) / n_reps;
        printf("CUDA %9.1f ms | ", dt_cuda);
    }
    else
        printf("CUDA %9s   | ", "N/A");

    // --- Speedups vs Generic float (if available) ---
    if (run_generic && dt_gen_f > 0.0)
    {
        if (quadriga_lib::quadriga_lib_has_AVX2() && dt_avx > 0.0)
            printf("AVX2 %5.1fx | ", dt_gen_f / dt_avx);
        if (quadriga_lib::quadriga_lib_has_CUDA() && dt_cuda > 0.0)
            printf("CUDA %5.1fx", dt_gen_f / dt_cuda);
    }

    printf("\n");
}

// ============================================================================
// ACCURACY: AVX2 & CUDA vs Generic-double reference
// ============================================================================
static void run_accuracy(const Scene &sc, const Rays &rays, const char *label)
{
    const size_t n = rays.n_ray;

    // --- Reference: Generic double ---
    RTI_Result ref;
    ref.alloc_double(n);
    qd_RTI_GENERIC(sc.dTx.data(), sc.dTy.data(), sc.dTz.data(),
                   sc.dE1x.data(), sc.dE1y.data(), sc.dE1z.data(),
                   sc.dE2x.data(), sc.dE2y.data(), sc.dE2z.data(), sc.n_mesh,
                   sc.SMI.data(), sc.dXmin.data(), sc.dXmax.data(),
                   sc.dYmin.data(), sc.dYmax.data(), sc.dZmin.data(), sc.dZmax.data(), sc.n_sub,
                   rays.dOx.data(), rays.dOy.data(), rays.dOz.data(),
                   rays.dDx.data(), rays.dDy.data(), rays.dDz.data(), n,
                   ref.dWf.data(), ref.dWs.data(), ref.If.data(), ref.Is.data(), ref.hit_cnt.data());

    // Count hits in reference for reporting
    size_t ref_fbs_hits = 0, ref_sbs_hits = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (ref.If[i] != 0) ++ref_fbs_hits;
        if (ref.Is[i] != 0) ++ref_sbs_hits;
    }

    printf("%-12s | n_mesh=%7zu, n_sub=%4zu | n_ray=%10zu | ref FBS hits: %zu (%.1f%%), SBS hits: %zu (%.1f%%)\n",
           label, sc.n_mesh, sc.n_sub, n,
           ref_fbs_hits, 100.0 * ref_fbs_hits / n,
           ref_sbs_hits, 100.0 * ref_sbs_hits / n);

    // --- Helper lambda to compare a float result against the double reference ---
    auto compare = [&](const char *name, const RTI_Result &res)
    {
        size_t if_match = 0, is_match = 0, hc_match = 0;
        int32_t max_wf_ulp = 0, max_ws_ulp = 0;
        int64_t sum_wf_ulp = 0, sum_ws_ulp = 0;
        size_t n_wf_cmp = 0, n_ws_cmp = 0;

        for (size_t i = 0; i < n; ++i)
        {
            if (res.If[i] == ref.If[i]) ++if_match;
            if (res.Is[i] == ref.Is[i]) ++is_match;
            if (res.hit_cnt[i] == ref.hit_cnt[i]) ++hc_match;

            // ULP on Wf (only when both agree on FBS index)
            if (res.If[i] == ref.If[i] && ref.If[i] != 0)
            {
                int32_t u = ulp_dist(res.Wf[i], (float)ref.dWf[i]);
                if (u > max_wf_ulp) max_wf_ulp = u;
                sum_wf_ulp += u;
                ++n_wf_cmp;
            }
            // ULP on Ws
            if (res.Is[i] == ref.Is[i] && ref.Is[i] != 0)
            {
                int32_t u = ulp_dist(res.Ws[i], (float)ref.dWs[i]);
                if (u > max_ws_ulp) max_ws_ulp = u;
                sum_ws_ulp += u;
                ++n_ws_cmp;
            }
        }

        double avg_wf_ulp = n_wf_cmp > 0 ? (double)sum_wf_ulp / n_wf_cmp : 0.0;
        double avg_ws_ulp = n_ws_cmp > 0 ? (double)sum_ws_ulp / n_ws_cmp : 0.0;

        printf("  %-6s | If match %6.2f%% | Is match %6.2f%% | hit_cnt match %6.2f%% "
               "| Wf ULP avg %6.2f max %7d | Ws ULP avg %6.2f max %7d\n",
               name,
               100.0 * if_match / n,
               100.0 * is_match / n,
               100.0 * hc_match / n,
               avg_wf_ulp, max_wf_ulp,
               avg_ws_ulp, max_ws_ulp);
    };

    // --- AVX2 ---
    if (quadriga_lib::quadriga_lib_has_AVX2())
    {
        RTI_Result avx;
        avx.alloc_float(n);
        qd_RTI_AVX2(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                     sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                     sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                     sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                     sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                     rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                     rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                     avx.Wf.data(), avx.Ws.data(), avx.If.data(), avx.Is.data(), avx.hit_cnt.data());
        compare("AVX2", avx);
    }

    // --- CUDA ---
    if (quadriga_lib::quadriga_lib_has_CUDA())
    {
        RTI_Result cuda;
        cuda.alloc_float(n);
        qd_RTI_CUDA(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                     sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                     sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                     sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                     sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                     rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                     rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                     cuda.Wf.data(), cuda.Ws.data(), cuda.If.data(), cuda.Is.data(), cuda.hit_cnt.data());
        compare("CUDA", cuda);
    }

    // --- Generic float (for float-vs-double comparison) ---
    {
        RTI_Result genf;
        genf.alloc_float(n);
        qd_RTI_GENERIC(sc.Tx.data(), sc.Ty.data(), sc.Tz.data(),
                       sc.E1x.data(), sc.E1y.data(), sc.E1z.data(),
                       sc.E2x.data(), sc.E2y.data(), sc.E2z.data(), sc.n_mesh,
                       sc.SMI.data(), sc.Xmin.data(), sc.Xmax.data(),
                       sc.Ymin.data(), sc.Ymax.data(), sc.Zmin.data(), sc.Zmax.data(), sc.n_sub,
                       rays.Ox.data(), rays.Oy.data(), rays.Oz.data(),
                       rays.Dx.data(), rays.Dy.data(), rays.Dz.data(), n,
                       genf.Wf.data(), genf.Ws.data(), genf.If.data(), genf.Is.data(), genf.hit_cnt.data());
        compare("Gen/f", genf);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main()
{
    printf("quadriga-lib RTI Benchmark\n");
    printf("AVX2: %s, CUDA: %s\n\n",
           quadriga_lib::quadriga_lib_has_AVX2() ? "YES" : "NO",
           quadriga_lib::quadriga_lib_has_CUDA() ? "YES" : "NO");

    // ---- Build scenes ----
    // Small: 42 icospheres × n_div=6 (720 faces each) = 30,240 triangles
    // Large: 50 icospheres × n_div=32 (20,480 faces each) = 1,024,000 triangles
    // Spheres of radius 1.0 placed at distance 20 from origin
    printf("Building scenes...\n");
    Scene sc_small = build_scene(42, 6,  1.0f, 20.0f, 1024, 8);
    Scene sc_large = build_scene(50, 32, 1.0f, 20.0f, 1024, 8);
    printf("\n");

    // ---- Generate ray sets ----
    // Rays from origin, destinations at distance 50 (well past the sphere shell)
    printf("Generating rays...\n");
    Rays rays_10k  = generate_rays(10000,    50.0f, 42);
    Rays rays_1M   = generate_rays(1000000,  50.0f, 123);
    Rays rays_50M  = generate_rays(50000000, 50.0f, 456);
    printf("  10k, 1M, 50M rays generated.\n\n");

    // ---- Performance ----
    const int n_reps = 3;
    printf("============================================================\n");
    printf("=== PERFORMANCE (%d reps, times are per-call averages) ===\n", n_reps);
    printf("============================================================\n\n");

    run_performance(sc_small, rays_10k,  "small/10k",  true,  n_reps);
    run_performance(sc_small, rays_1M,   "small/1M",   true,  n_reps);
    run_performance(sc_small, rays_50M,  "small/50M",  false, n_reps);
    run_performance(sc_large, rays_10k,  "large/10k",  true,  n_reps);
    run_performance(sc_large, rays_1M,   "large/1M",   true,  n_reps);
    run_performance(sc_large, rays_50M,  "large/50M",  false, n_reps);

    printf("\n");

    // ---- Accuracy (small ray sets only) ----
    printf("============================================================\n");
    printf("=== ACCURACY (vs Generic-double reference) ===\n");
    printf("============================================================\n\n");

    run_accuracy(sc_small, rays_10k, "small/10k");
    printf("\n");
    run_accuracy(sc_small, rays_1M,  "small/1M");
    printf("\n");
    run_accuracy(sc_large, rays_10k, "large/10k");
    printf("\n");
    run_accuracy(sc_large, rays_1M,  "large/1M");

    printf("\nDone.\n");
    return 0;
}