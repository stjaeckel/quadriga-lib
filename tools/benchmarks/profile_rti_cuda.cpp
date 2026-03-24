// Compile:
// g++ -std=c++17 -mavx2 -mfma -fopenmp -O2 -I/sjc/quadriga-lib/include -I/sjc/quadriga-lib/external/armadillo-14.2.2/include -I/sjc/quadriga-lib/src -o profile_rti_cuda profile_rti_cuda.cpp /sjc/quadriga-lib/lib/libquadriga.a -L/usr/local/cuda-12.4/targets/x86_64-linux/lib -lcudart
//
// Profile:
// sudo $(which ncu) --set full -o profile_rti_cuda ./profile_rti_cuda
// ncu --import profile_rti_cuda.ncu-rep --csv --print-summary per-kernel > profile_summary.csv

#include "quadriga_lib.hpp"
#include "quadriga_lib_cuda_functions.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

int main()
{
    if (!quadriga_lib::quadriga_lib_has_CUDA())
    {
        printf("CUDA not available.\n");
        return 1;
    }

    // ---- Build large scene: 50 icospheres × n_div=32 at distance 20 ----
    const int n_spheres = 50;
    const arma::uword n_div = 32;
    const float sphere_radius = 1.0f;
    const float placement_radius = 20.0f;

    arma::fmat center_tpl, vert_tpl;
    arma::uword faces_per_sphere = quadriga_lib::icosphere(n_div, sphere_radius, &center_tpl,
                                                            (arma::fvec *)nullptr, &vert_tpl);
    arma::uword total_faces = (arma::uword)n_spheres * faces_per_sphere;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    arma::fmat mesh(total_faces, 9);
    arma::uword row = 0;
    for (int s = 0; s < n_spheres; ++s)
    {
        float theta = 2.0f * (float)M_PI * u01(rng);
        float phi = std::acos(1.0f - 2.0f * u01(rng));
        float cx = placement_radius * std::sin(phi) * std::cos(theta);
        float cy = placement_radius * std::sin(phi) * std::sin(theta);
        float cz = placement_radius * std::cos(phi);

        for (arma::uword f = 0; f < faces_per_sphere; ++f, ++row)
        {
            float fc_x = center_tpl(f, 0), fc_y = center_tpl(f, 1), fc_z = center_tpl(f, 2);
            mesh(row, 0) = fc_x + vert_tpl(f, 0) + cx;
            mesh(row, 1) = fc_y + vert_tpl(f, 1) + cy;
            mesh(row, 2) = fc_z + vert_tpl(f, 2) + cz;
            mesh(row, 3) = fc_x + vert_tpl(f, 3) + cx;
            mesh(row, 4) = fc_y + vert_tpl(f, 4) + cy;
            mesh(row, 5) = fc_z + vert_tpl(f, 5) + cz;
            mesh(row, 6) = fc_x + vert_tpl(f, 6) + cx;
            mesh(row, 7) = fc_y + vert_tpl(f, 7) + cy;
            mesh(row, 8) = fc_z + vert_tpl(f, 8) + cz;
        }
    }

    // Segment
    arma::fmat meshR;
    arma::u32_vec sub_mesh_index;
    arma::uword n_sub = quadriga_lib::triangle_mesh_segmentation(&mesh, &meshR, &sub_mesh_index, 1024, 8);
    size_t n_mesh = meshR.n_rows;
    size_t n_sub_s = ((n_sub + 7) / 8) * 8;

    printf("Scene: n_mesh=%zu, n_sub=%zu\n", n_mesh, (size_t)n_sub);

    // Convert to SoA
    std::vector<float> Tx(n_mesh), Ty(n_mesh), Tz(n_mesh);
    std::vector<float> E1x(n_mesh), E1y(n_mesh), E1z(n_mesh);
    std::vector<float> E2x(n_mesh), E2y(n_mesh), E2z(n_mesh);
    for (size_t i = 0; i < n_mesh; ++i)
    {
        float v1x = meshR(i, 0), v1y = meshR(i, 1), v1z = meshR(i, 2);
        float v2x = meshR(i, 3), v2y = meshR(i, 4), v2z = meshR(i, 5);
        float v3x = meshR(i, 6), v3y = meshR(i, 7), v3z = meshR(i, 8);
        Tx[i] = v1x;  Ty[i] = v1y;  Tz[i] = v1z;
        E1x[i] = v2x - v1x; E1y[i] = v2y - v1y; E1z[i] = v2z - v1z;
        E2x[i] = v3x - v1x; E2y[i] = v3y - v1y; E2z[i] = v3z - v1z;
    }

    // SMI
    std::vector<unsigned> SMI(n_sub);
    for (size_t i = 0; i < n_sub; ++i)
        SMI[i] = sub_mesh_index(i);

    // AABBs
    std::vector<float> Xmin(n_sub_s, 1e20f), Xmax(n_sub_s, -1e20f);
    std::vector<float> Ymin(n_sub_s, 1e20f), Ymax(n_sub_s, -1e20f);
    std::vector<float> Zmin(n_sub_s, 1e20f), Zmax(n_sub_s, -1e20f);
    for (size_t s = 0; s < n_sub; ++s)
    {
        size_t i_start = SMI[s];
        size_t i_end = (s + 1 < n_sub) ? SMI[s + 1] : n_mesh;
        for (size_t i = i_start; i < i_end; ++i)
        {
            float v1x = Tx[i], v1y = Ty[i], v1z = Tz[i];
            float v2x = v1x + E1x[i], v2y = v1y + E1y[i], v2z = v1z + E1z[i];
            float v3x = v1x + E2x[i], v3y = v1y + E2y[i], v3z = v1z + E2z[i];
            Xmin[s] = std::min({Xmin[s], v1x, v2x, v3x}); Xmax[s] = std::max({Xmax[s], v1x, v2x, v3x});
            Ymin[s] = std::min({Ymin[s], v1y, v2y, v3y}); Ymax[s] = std::max({Ymax[s], v1y, v2y, v3y});
            Zmin[s] = std::min({Zmin[s], v1z, v2z, v3z}); Zmax[s] = std::max({Zmax[s], v1z, v2z, v3z});
        }
    }

    // ---- Generate 1M rays from origin ----
    const size_t n_ray = 1000000;
    std::vector<float> Ox(n_ray, 0.0f), Oy(n_ray, 0.0f), Oz(n_ray, 0.0f);
    std::vector<float> Dx(n_ray), Dy(n_ray), Dz(n_ray);

    std::mt19937 rng2(123);
    std::uniform_real_distribution<float> u01b(0.0f, 1.0f);
    for (size_t i = 0; i < n_ray; ++i)
    {
        float theta = 2.0f * (float)M_PI * u01b(rng2);
        float phi = std::acos(1.0f - 2.0f * u01b(rng2));
        Dx[i] = 50.0f * std::sin(phi) * std::cos(theta);
        Dy[i] = 50.0f * std::sin(phi) * std::sin(theta);
        Dz[i] = 50.0f * std::cos(phi);
    }

    printf("Rays: n_ray=%zu\n", n_ray);

    // ---- Output buffers ----
    std::vector<float> Wf(n_ray), Ws(n_ray);
    std::vector<unsigned> If(n_ray), Is(n_ray), hit_cnt(n_ray);

    // ---- Single CUDA call (this is what Nsight Compute will profile) ----
    printf("Running qd_RTI_CUDA...\n");

    qd_RTI_CUDA(Tx.data(), Ty.data(), Tz.data(),
                E1x.data(), E1y.data(), E1z.data(),
                E2x.data(), E2y.data(), E2z.data(), n_mesh,
                SMI.data(), Xmin.data(), Xmax.data(),
                Ymin.data(), Ymax.data(), Zmin.data(), Zmax.data(), n_sub,
                Ox.data(), Oy.data(), Oz.data(),
                Dx.data(), Dy.data(), Dz.data(), n_ray,
                Wf.data(), Ws.data(), If.data(), Is.data(), hit_cnt.data());

    // Quick sanity check
    size_t fbs_hits = 0;
    for (size_t i = 0; i < n_ray; ++i)
        if (If[i] != 0) ++fbs_hits;

    printf("Done. FBS hits: %zu (%.1f%%)\n", fbs_hits, 100.0 * fbs_hits / n_ray);
    return 0;
}
