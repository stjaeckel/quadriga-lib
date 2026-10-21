// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_mesh_interact, wave.

#include "ray_test_helpers.hpp"

TEST_CASE("Ray-Mesh Interact - Air to Air (x-z plane)", "[wave]")
{
    // Blender-style 2x2x2 cube centered at the origin, 12 triangles, outward normals
    arma::mat cube = quadriga_lib::cube<double>();

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;

    // Ray start 10 m west of cube center, pointing east
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir_sph(1, 6);
    tridir_sph.at(4) = 1.0 * deg2rad;
    tridir_sph.at(1) = 1.0 * deg2rad;

    // Convert to Cartesian
    arma::mat tridir_crt(1, 9);
    tridir_crt.col(0) = arma::cos(tridir_sph.col(1)) % arma::cos(tridir_sph.col(0));
    tridir_crt.col(1) = arma::cos(tridir_sph.col(1)) % arma::sin(tridir_sph.col(0));
    tridir_crt.col(2) = arma::sin(tridir_sph.col(1));
    tridir_crt.col(3) = arma::cos(tridir_sph.col(3)) % arma::cos(tridir_sph.col(2));
    tridir_crt.col(4) = arma::cos(tridir_sph.col(3)) % arma::sin(tridir_sph.col(2));
    tridir_crt.col(5) = arma::sin(tridir_sph.col(3));
    tridir_crt.col(6) = arma::cos(tridir_sph.col(5)) % arma::cos(tridir_sph.col(4));
    tridir_crt.col(7) = arma::cos(tridir_sph.col(5)) % arma::sin(tridir_sph.col(4));
    tridir_crt.col(8) = arma::sin(tridir_sph.col(5));

    // Calculate interaction location
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    // Output containers
    arma::mat origN, destN, fbsN, sbsN, xprmatN, trivecN, tridir_sphN, normal_vecN, tridir_crtN;
    arma::vec gainN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    // A cube of air
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // Air
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Reflection with a spherical tridir; fbsN and sbsN are the hit points
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir_sph,
                                    &origN, &destN, &fbsN, &sbsN, &gainN, &xprmatN, &trivecN, &tridir_sphN,
                                    &fbs_angleN, &thicknessN, &edge_lengthN, &normal_vecN);

    CHECK(tridir_sphN.n_cols == 6);

    // fbsN/sbsN reproduce the ray_triangle_intersect points
    CHECK(arma::approx_equal(fbsN, fbs, "absdiff", 1e-6));
    CHECK(arma::approx_equal(sbsN, sbs, "absdiff", 1e-6));

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir_crt,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridir_crtN,
                                    &fbs_angleN, &thicknessN, &edge_lengthN, &normal_vecN);

    CHECK(tridir_crtN.n_cols == 9);

    T.zeros(1, 9);
    T.col(0) = arma::cos(tridir_sphN.col(1)) % arma::cos(tridir_sphN.col(0));
    T.col(1) = arma::cos(tridir_sphN.col(1)) % arma::sin(tridir_sphN.col(0));
    T.col(2) = arma::sin(tridir_sphN.col(1));
    T.col(3) = arma::cos(tridir_sphN.col(3)) % arma::cos(tridir_sphN.col(2));
    T.col(4) = arma::cos(tridir_sphN.col(3)) % arma::sin(tridir_sphN.col(2));
    T.col(5) = arma::sin(tridir_sphN.col(3));
    T.col(6) = arma::cos(tridir_sphN.col(5)) % arma::cos(tridir_sphN.col(4));
    T.col(7) = arma::cos(tridir_sphN.col(5)) % arma::sin(tridir_sphN.col(4));
    T.col(8) = arma::sin(tridir_sphN.col(5));

    CHECK(arma::approx_equal(tridir_crtN, T, "absdiff", 1e-6));

    T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = {{-12.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 5e-6));

    U = {0.0}; // Air does not reflect anything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6)); // xprmat is [8, n_rayN]

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{0.0, -0.1, a, 0.0, -0.1, -0.2, 0.0, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{180.0, 1.0, 180.0, 0.0, 179.0, 0.0}};
    CHECK(arma::approx_equal(tridir_sphN, T * deg2rad, "absdiff", 1e-6));

    U = {90.0 * deg2rad};
    CHECK(arma::approx_equal(fbs_angleN, U, "absdiff", 1e-6));

    U = {2.0};
    CHECK(arma::approx_equal(thicknessN, U, "absdiff", 1e-6));

    U = {std::sqrt(a * a + (a + 0.1) * (a + 0.1))};
    CHECK(arma::approx_equal(edge_lengthN, U, "absdiff", 1e-6));

    T = {{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(normal_vecN, T, "absdiff", 1e-6));

    // Test transmission on air
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir_sph,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridir_sphN);

    T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = dest;
    CHECK(arma::approx_equal(destN, T, "absdiff", 5e-6));

    U = {1.0}; // Air does transmit everything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{0.0, -0.1, a, 0.0, -0.1, -0.2, 0.0, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{0.0, 1.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(tridir_sphN, T * deg2rad, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Air to Dielectric Medium (x-z plane)", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;

    // 45° incidence on dielectric medium
    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{-0.5, 0.0, 1.0}}; // Ceiling of cube
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = {{-2.0, 0.0, 1.5}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // See Jaeckel, S.; Raschkowski, L.; Wu, S.; Thiele, L. & Keusgen, W.
    // An Explicit Ground Reflection Model for mm-Wave Channels Proc. IEEE WCNC Workshops '17, 2017
    double eps = mtl_prop.at(0, 0);
    double cos_th = std::cos(45.0 * deg2rad), sin_th = std::sin(45.0 * deg2rad);
    double Z = std::sqrt(eps - cos_th * cos_th);
    double R_par = (eps * sin_th - Z) / (eps * sin_th + Z); // = R_tm
    double R_per = (sin_th - Z) / (sin_th + Z);             // = R_te

    U = {0.5 * R_par * R_par + 0.5 * R_per * R_per}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{-R_par, 0.0, 0.0, 0.0, 0.0, 0.0, -R_per, 0.0}}; // 180° phase shift for reflection
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{180.0, 45.0, 180.0, 45.0, 180.0, 45.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-6));

    // Test refraction into medium
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    // Refraction angle calculated from Snell's law
    double th2 = std::asin(std::sin(45.0 * deg2rad) / std::sqrt(eps));

    T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    double x = std::cos(th2) * std::sqrt(2) - 1;
    double z = std::sin(th2) * std::sqrt(2) + 0.5;
    T = {{x, 0.0, z}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Transmission coefficients, ITU-R P.2040-1, eq. (32a) and (32b), converted to normal flux.
    // A transmitted coefficient carries its own obliquity factor Re(z2 c2)/Re(z1 c1) for TE and
    // Re(conj(z2) c2)/Re(conj(z1) c1) for TM. Both media are real here, so the two coincide.
    double T_te = 2.0 * cos_th / (cos_th + std::sqrt(eps) * std::cos(th2));
    double T_tm = 2.0 * cos_th / (std::sqrt(eps) * cos_th + std::cos(th2));
    double obl = std::sqrt(eps) * std::cos(th2) / cos_th;
    T_te *= std::sqrt(obl), T_tm *= std::sqrt(obl);

    U = {0.5 * T_tm * T_tm + 0.5 * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // With a lossless incidence medium the flux-converted pair total is identically 1 - R
    double R_avg_ref = calc_transition_gain(0, 45.0, 0.0, 0.0,
                                            std::complex<double>(1.0, 0.0), std::complex<double>(eps, 0.0));
    CHECK(std::abs(U(0) - (1.0 - R_avg_ref)) < 1e-9);

    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{0.0, th2, 0.0, th2, 0.0, th2}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));

    // Test transmission into medium
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = dest;
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Gain
    std::complex<double> eta1(1.0, 0.0);
    std::complex<double> eta2(1.5, 0.0);
    double G = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2); // Reflection
    double H = calc_transition_gain(2, 45.0, 0.0, 0.0, eta1, eta2); // Refraction

    U = {1.0 - G};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // The obliquity factor is exactly (1 - G) / H
    CHECK(std::abs((1.0 - G) / H - obl) < 1e-9);

    // Bending the ray does not change the energy the interface passes, so the coefficients are the
    // same ones the refraction type reported, with no rescale on top
    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    // Sub-rays are parallel to the main ray, so the offsets are unchanged
    T = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    a = 45.0 * deg2rad;
    T = {{0.0, a, 0.0, a, 0.0, a}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float)", "[wave]")
{
    arma::fmat cube = quadriga_lib::cube<float>();

    // Conversions
    float deg2rad = arma::datum::pi / 180.0;

    // 45° incidence, x-y plane
    arma::fmat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::fmat dest = {{2.0, 1.6, 0.0}};

    arma::fmat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::fmat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{2.0, 1.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    arma::fmat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::fmat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::fmat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::fvec gainN, fbs_angleN, thicknessN, edge_lengthN, U;

    arma::fmat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<float>> mtl_map;
    mtl_matrix_to_map<float>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    float cos_th = std::cos(45.0 * deg2rad);
    T = {{1.0f, 0.6f, 0.0f}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = {{0.0f, 1.6f, 0.0f}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Gain
    std::complex<double> eta1(1.5, 0.0);
    std::complex<double> eta2(1.0, 0.0);
    double G = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2);
    U = {(float)G}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    float cos_th2 = std::sqrt(1.0 - 1.5 / 1.0 * cos_th * cos_th);
    float sqrt_eps = std::sqrt(mtl_prop.at(0, 0));
    float R_te = (sqrt_eps * cos_th - cos_th2) / (sqrt_eps * cos_th + cos_th2);
    float R_tm = (cos_th - sqrt_eps * cos_th2) / (cos_th + sqrt_eps * cos_th2);
    T = {{R_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, R_tm, 0.0f}}; // x-y plane swaps base vectors
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0f, -0.1f, 0.2, 0.0f, -0.1f, -0.2f, 0.0f, 0.2f, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{135.0f, 0.0f, 135.0f, 0.0f, 135.0f, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    float th2 = std::acos(cos_th2);
    float x = 1.0f;
    float y = 0.6f;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Transmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b), converted to normal flux
    float T_te = (2.0 * sqrt_eps * cos_th) / (sqrt_eps * cos_th + cos_th2);
    float T_tm = (2.0 * sqrt_eps * cos_th) / (cos_th + sqrt_eps * cos_th2);
    float obl = cos_th2 / (sqrt_eps * cos_th); // Re(z2 c2)/Re(z1 c1), the same in both channels here
    T_te *= std::sqrt(obl), T_tm *= std::sqrt(obl);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // The incidence medium is lossless, so the flux-converted pair total is 1 - R, which is what
    // the undeviated type reports for this same interface
    G = calc_transition_gain(1, 45.0, 0.0, 0.0, eta1, eta2);
    U = {(float)G}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, T_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0f, -0.1f, 0.2f, 0.0f, -0.1f, -0.2f, 0.0f, 0.2f, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{th2, 0.0f, th2, 0.0f, th2, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));

    // Transmission from inside to outside without refraction
    quadriga_lib::ray_mesh_interact(1, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    U = {(float)calc_transition_gain(1, 45.0, 0.0, 0.0, eta1, eta2)};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    U = {0.5f * xprmatN(6) * xprmatN(6) + 0.5f * xprmatN(0) * xprmatN(0)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // Total reflection occurs at eta = 2, theta = 45°
    mtl_prop = {{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    mtl_matrix_to_map<float>(mtl_prop, mtl_ind, mtl_map);

    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {1.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // eta = 2 at 45 deg sits exactly on the critical angle, where ct2 = 0 is a sqrt branch point:
    // a 1e-7 error in cos(theta) moves the TIR phase by ~1e-4 rad. Only |R| = 1 and a near-zero
    // phase are well defined here; the conditioned above-critical case follows below.
    CHECK(std::abs(std::hypot(xprmatN(0), xprmatN(1)) - 1.0f) < 1e-5f);
    CHECK(std::abs(std::hypot(xprmatN(6), xprmatN(7)) - 1.0f) < 1e-5f);
    CHECK(std::abs(std::atan2(xprmatN(1), xprmatN(0))) < 1e-3f);
    CHECK(std::abs(std::atan2(xprmatN(7), xprmatN(6))) < 1e-3f);

    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {0.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    // Above critical (eta = 4 at 45 deg): TIR keeps |R| = 1 but carries a non-zero Fresnel phase
    mtl_prop = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    mtl_matrix_to_map<float>(mtl_prop, mtl_ind, mtl_map);

    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::fmat *)nullptr, (arma::fmat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {1.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // z1 = 2, z2 = 1, ct2 = -i (decaying evanescent wave, exp(-jkz) convention):
    //   R_te = (2c + i)/(2c - i) = 1/3 + i*2*sqrt(2)/3
    //   R_tm = (c + 2i)/(c - 2i) = -7/9 + i*4*sqrt(2)/9,   c = cos(45 deg)
    T = {{0.333333f, 0.942809f, 0.0f, 0.0f, 0.0f, 0.0f, -0.777778f, 0.628539f}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-5));
}

TEST_CASE("Ray-Mesh Interact - Medium to Medium (x-y plane, double)", "[wave]")
{
    // Two adjacent cubes; the second is the first shifted +2 in x (shared wall at x = 1)
    arma::mat msh = quadriga_lib::cube<double>();
    arma::mat tmp = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    arma::mat mtl_tmp = {{1.33, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_tmp = repmat(mtl_tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, mtl_tmp);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;

    // 45° incidence, x-y plane
    arma::mat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::mat dest = {{2.0, 1.6, 0.0}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &msh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{1.0, 0.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    double cos_th = std::cos(45.0 * deg2rad);
    T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = {{0.0, 1.6, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Cosine of refraction angle,  ITU-R P.2040-1, eq. (33) with sin(theta) = cos(theta) @ 45°
    double cos_th2 = std::sqrt(1.0 - 1.5 / 1.33 * cos_th * cos_th);

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    double sqrt_eps1 = std::sqrt(1.5);
    double sqrt_eps2 = std::sqrt(1.33);
    double R_te = (sqrt_eps1 * cos_th - sqrt_eps2 * cos_th2) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    double R_tm = (sqrt_eps2 * cos_th - sqrt_eps1 * cos_th2) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    U = {0.5f * R_tm * R_tm + 0.5f * R_te * R_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{R_te, 0.0, 0.0, 0.0, 0.0, 0.0, R_tm, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{135.0, 0.0, 135.0, 0.0, 135.0, 0.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    // The entered medium carries att = 3 dB, a lumped interface loss on the transmission-class gain; the in-medium
    // distance loss is not in it.
    double transition_gain = std::pow(10.0, -0.1 * 3.0);

    double th2 = std::acos(cos_th2);
    double x = 1.0;
    double y = 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Transmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    double T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    double T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    // Normal-flux conversion; both media are real here, so one obliquity factor serves both channels
    double obl = sqrt_eps2 * cos_th2 / (sqrt_eps1 * cos_th);
    T_te *= std::sqrt(obl * transition_gain);
    T_tm *= std::sqrt(obl * transition_gain);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0, 0.0, 0.0, 0.0, 0.0, T_tm, 0.0}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    T = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{th2, 0.0, th2, 0.0, th2, 0.0}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double)", "[wave]")
{
    // Two adjacent cubes; conductive medium 1, dielectric medium 2, shared wall at x = 1
    arma::mat msh = quadriga_lib::cube<double>();
    arma::mat tmp = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.01, std::log10(0.02 / 0.01), 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    arma::mat mtl_tmp = {{1.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_tmp = repmat(mtl_tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, mtl_tmp);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;

    // 45° incidence, x-y plane
    arma::mat orig = {{0.5, 0.1, 0.0}}; // start inside the cube
    arma::mat dest = {{2.0, 1.6, 0.0}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &msh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{1.0, 0.6, 0.0}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));

    T = {{1.0, 0.6, 0.0}}; // End of path
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    double cos_th = std::cos(45.0 * deg2rad);

    std::complex<double> eta1(1.5, -1.798 * 0.02); // @ 10 GHz
    std::complex<double> eta2(1.33, 0.0);

    // Cosine of refraction angle,  ITU-R P.2040-1, eq. (33) with sin(theta) = cos(theta) @ 45°
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * cos_th * cos_th);

    // Reflection coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    std::complex<double> sqrt_eps1 = std::sqrt(eta1);
    std::complex<double> sqrt_eps2 = std::sqrt(eta2);
    std::complex<double> R_te = (sqrt_eps1 * cos_th - sqrt_eps2 * cos_th2) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    std::complex<double> R_tm = (sqrt_eps2 * cos_th - sqrt_eps1 * cos_th2) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    // The gain and the coefficients carry no in-medium loss; the complex eps of the conductive incidence medium shapes
    // the Fresnel terms.
    U = {0.5 * std::norm(R_tm) + 0.5 * std::norm(R_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{std::real(R_te), std::imag(R_te), 0.0, 0.0, 0.0, 0.0, std::real(R_tm), std::imag(R_tm)}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    // The in-medium conductivity loss is medium_gain's: the ITU-R P.2040-1 conductive loss over the in-slab path.
    double thickness = std::sqrt(2.0 * 0.5 * 0.5) + 0.001;
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * thickness / Delta;        // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double gain_bulk = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale
    CHECK(std::abs(quadriga_lib::medium_gain(mtl_map, 1, thickness, 10.0e9) - gain_bulk) < 1e-9);

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    // Wave direction is not defined precisely due to the complex angles
    std::complex<double> th2 = std::acos(cos_th2);
    double x = 1.0;
    double y = 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::real(cos_th2) * std::sqrt(2.0) + 1.0;
    y = std::real(std::sin(th2)) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-3));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b); M2 is lossless (att = 0).
    // The incidence medium is conductive, so this is the case where the normal-flux transmittance
    // and 1 - R part company: the reference below is the flux form, per channel.
    std::complex<double> T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    std::complex<double> T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);
    double obl_te = std::real(sqrt_eps2 * cos_th2) / std::real(sqrt_eps1 * cos_th);
    double obl_tm = std::real(std::conj(sqrt_eps2) * cos_th2) / std::real(std::conj(sqrt_eps1) * cos_th);
    T_te *= std::sqrt(obl_te), T_tm *= std::sqrt(obl_tm);

    U = {0.5 * std::norm(T_tm) + 0.5 * std::norm(T_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-3));

    // x-y plane swaps base vectors
    T = {{std::real(T_te), std::imag(T_te), 0.0, 0.0, 0.0, 0.0, std::real(T_tm), std::imag(T_tm)}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-3));
}

TEST_CASE("Ray-Mesh Interact - fRef parameterization equivalence", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    double deg2rad = arma::datum::pi / 180.0;

    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &cube, &fbs, &sbs, nullptr, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    // Both parameterisations are identical at every frequency:
    //   ε_r(f) = 2·f        σ(f) = 0.01·f
    //   Att(f) = 1·f [dB]   α(f) = 0.5·f [dB/m]
    // Material A specifies at fRef = 1 GHz, material B at fRef = 2 GHz. The gain and xprmat carry att and the complex
    // Fresnel terms, identical for the two.
    arma::mat mtl_A = {{2.0, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0}};
    arma::mat mtl_B = {{4.0, 1.0, 0.02, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0}};
    mtl_A = repmat(mtl_A, 12, 1);
    mtl_B = repmat(mtl_B, 12, 1);
    arma::uvec mtl_ind_A, mtl_ind_B;
    std::unordered_map<std::string, std::vector<double>> mtl_map_A, mtl_map_B;
    mtl_matrix_to_map<double>(mtl_A, mtl_ind_A, mtl_map_A);
    mtl_matrix_to_map<double>(mtl_B, mtl_ind_B, mtl_map_B);

    arma::mat origNa, destNa, xprmatNa, trivecNa, tridirNa;
    arma::mat origNb, destNb, xprmatNb, trivecNb, tridirNb;
    arma::vec gainNa, gainNb;

    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &mtl_ind_A, &mtl_map_A, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir,
                                    &origNa, &destNa, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainNa, &xprmatNa, &trivecNa, &tridirNa);
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &mtl_ind_B, &mtl_map_B, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir,
                                    &origNb, &destNb, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainNb, &xprmatNb, &trivecNb, &tridirNb);

    CHECK(arma::approx_equal(gainNa, gainNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(xprmatNa, xprmatNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(origNa, origNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(destNa, destNb, "absdiff", 1e-12));
}

TEST_CASE("Ray-Mesh Interact - Penetration loss frequency scaling", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // Normal incidence through the cube
    arma::mat orig = {{-2.0, 0.0, 0.5}};
    arma::mat dest = {{2.0, 0.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    // ε_r = 1, σ = 0 (no reflection, no in-medium loss)
    // att = 6 dB @ 2 GHz, attB = 1  →  at 10 GHz:  Att = 6·(10/2)^1 = 30 dB, on the transmission-class gain
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 2.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    arma::mat trivec = {{0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0}};
    arma::mat tridir = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    arma::mat origN, destN, xprmatN, trivecN, tridirN;
    arma::vec gainN;

    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    // transition_gain = 10^(-0.1 · 30) = 1e-3
    arma::vec U = {1.0e-3};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-12));
}

TEST_CASE("Ray-Mesh Interact - Alpha in-medium absorption (medium_gain)", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    double deg2rad = arma::datum::pi / 180.0;

    // Ray starts INSIDE the cube, 45° in x-y plane, reflects off east wall
    arma::mat orig = {{0.5, 0.1, 0.0}};
    arma::mat dest = {{2.0, 1.6, 0.0}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    // ε_r = 1.5, σ = 0, α = 2 dB/m @ 5 GHz, αB = 1 (linear)
    //   →  at 10 GHz:  α(f) = 2·(10/5)^1 = 4 dB/m
    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 5.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    arma::mat origN, destN, xprmatN, trivecN, tridirN;
    arma::vec gainN;

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);

    // The reflection gain is the lossless Fresnel term; the α loss is medium_gain's.
    std::complex<double> eta1(1.5, 0.0), eta2(1.0, 0.0);
    double refl_gain = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2);
    arma::vec U = {refl_gain};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-9));

    // medium_gain owns the α absorption: with real ε and σ = 0 it is exactly the α term over the in-slab path.
    double thickness = std::sqrt(0.5 * 0.5 + 0.5 * 0.5) + 0.001; // OF_length + ray_offset
    double alpha_10 = 2.0 * std::pow(10.0 / 5.0, 1.0);           // 4 dB/m
    double alpha_loss = std::pow(10.0, -0.1 * thickness * alpha_10);
    CHECK(std::abs(quadriga_lib::medium_gain(mtl_map, 1, thickness, 10.0e9) - alpha_loss) < 1e-9);
}

TEST_CASE("Ray-Mesh Interact - Permeability defaults to 1 (backward compatible)", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;

    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    arma::mat mtl = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl = repmat(mtl, 12, 1);

    arma::uvec ind0, ind1;
    std::unordered_map<std::string, std::vector<double>> map0, map1;
    mtl_matrix_to_map<double>(mtl, ind0, map0); // mu absent -> default
    mtl_matrix_to_map<double>(mtl, ind1, map1);
    set_mu<double>(map1, 1.0, 0.0, 0.0, 0.0); // mu explicitly 1

    arma::mat o0, d0, x0, tv0, td0, o1, d1, x1, tv1, td1;
    arma::vec g0, g1;

    for (int it : {0, 1, 2}) // reflection, transmission, refraction
    {
        quadriga_lib::ray_mesh_interact(it, 10.0e9, &orig, &dest, &cube, &ind0, &map0, &fbs_ind, &sbs_ind,
                                        &trivec, &tridir, &o0, &d0, (arma::mat *)nullptr, (arma::mat *)nullptr, &g0, &x0, &tv0, &td0);
        quadriga_lib::ray_mesh_interact(it, 10.0e9, &orig, &dest, &cube, &ind1, &map1, &fbs_ind, &sbs_ind,
                                        &trivec, &tridir, &o1, &d1, (arma::mat *)nullptr, (arma::mat *)nullptr, &g1, &x1, &tv1, &td1);
        CHECK(arma::approx_equal(g0, g1, "absdiff", 1e-12));
        CHECK(arma::approx_equal(x0, x1, "absdiff", 1e-12)); // xprmat must be bit-stable too
        CHECK(arma::approx_equal(d0, d1, "absdiff", 1e-12));
        CHECK(arma::approx_equal(td0, td1, "absdiff", 1e-12));
    }

    // Anchor mu=1 to the oracle (single-eps reflection)
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &ind1, &map1, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &o1, &d1, (arma::mat *)nullptr, (arma::mat *)nullptr, &g1, &x1, &tv1, &td1);
    arma::vec U = {calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {1.5, 0.0}, {1.0, 0.0}, {1.0, 0.0})};
    CHECK(arma::approx_equal(g1, U, "absdiff", 1e-9));
}

TEST_CASE("Ray-Mesh Interact - Permeability decouples reflection from refraction", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;

    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    // A: eps=2, mu=3   B: eps=6, mu=1 (absent)   -> eps*mu = 6 for both (same index n)
    arma::mat mtlA = {{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::mat mtlB = {{6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtlA = repmat(mtlA, 12, 1);
    mtlB = repmat(mtlB, 12, 1);
    arma::uvec indA, indB;
    std::unordered_map<std::string, std::vector<double>> mapA, mapB;
    mtl_matrix_to_map<double>(mtlA, indA, mapA);
    set_mu<double>(mapA, 3.0, 0.0, 0.0, 0.0);
    mtl_matrix_to_map<double>(mtlB, indB, mapB); // mu defaults to 1

    arma::mat oA, dA, xA, tvA, tdA, oB, dB, xB, tvB, tdB;
    arma::vec gA, gB;

    // Refraction: identical n -> identical bending and path
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oA, &dA, (arma::mat *)nullptr, (arma::mat *)nullptr, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oB, &dB, (arma::mat *)nullptr, (arma::mat *)nullptr, &gB, &xB, &tvB, &tdB);
    CHECK(arma::approx_equal(tdA, tdB, "absdiff", 1e-9));
    CHECK(arma::approx_equal(dA, dB, "absdiff", 1e-9));

    // Reflection: different admittance sqrt(eps/mu) -> different reflected power
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oA, &dA, (arma::mat *)nullptr, (arma::mat *)nullptr, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oB, &dB, (arma::mat *)nullptr, (arma::mat *)nullptr, &gB, &xB, &tvB, &tdB);
    double refA = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {3.0, 0.0});
    double refB = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {6.0, 0.0}, {1.0, 0.0}, {1.0, 0.0});
    CHECK(std::abs(refA - refB) > 0.05); // the two genuinely differ
    arma::vec UA = {refA}, UB = {refB};
    CHECK(arma::approx_equal(gA, UA, "absdiff", 1e-9));
    CHECK(arma::approx_equal(gB, UB, "absdiff", 1e-9));
}

TEST_CASE("Ray-Mesh Interact - Permeability impedance-matches a dense medium", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // Normal incidence on the west wall
    arma::mat orig = {{-2.0, 0.0, 0.5}};
    arma::mat dest = {{2.0, 0.0, 0.5}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0}};
    arma::mat tridir = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    arma::mat mtl = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl = repmat(mtl, 12, 1);

    arma::uvec indM, indU;
    std::unordered_map<std::string, std::vector<double>> mapM, mapU;
    mtl_matrix_to_map<double>(mtl, indM, mapM);
    set_mu<double>(mapM, 4.0, 0.0, 0.0, 0.0);   // matched: eps = mu = 4 -> admittance 1, n = 4
    mtl_matrix_to_map<double>(mtl, indU, mapU); // unmatched: eps = 4, mu = 1

    arma::mat o, d, x, tv, td;
    arma::vec g, U;

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indM, &mapM, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &o, &d, (arma::mat *)nullptr, (arma::mat *)nullptr, &g, &x, &tv, &td);
    U = {0.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // matched -> no reflection

    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &indM, &mapM, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &o, &d, (arma::mat *)nullptr, (arma::mat *)nullptr, &g, &x, &tv, &td);
    U = {1.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // matched -> full transmission

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indU, &mapU, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &o, &d, (arma::mat *)nullptr, (arma::mat *)nullptr, &g, &x, &tv, &td);
    U = {1.0 / 9.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // unmatched -> ((1-2)/(1+2))^2
}

TEST_CASE("Ray-Mesh Interact - eps<->mu swap leaves reflected power invariant", "[wave]")
{
    // Swapping eps and mu preserves n = sqrt(eps*mu) and, at normal incidence, |R|, so the reflected power is
    // identical; a Fresnel term that read sqrt(eps) alone would tell A and B apart.
    arma::mat cube = quadriga_lib::cube<double>();

    arma::mat orig = {{-2.0, 0.0, 0.5}};
    arma::mat dest = {{2.0, 0.0, 0.5}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0}};
    arma::mat tridir = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    arma::mat mtlA = {{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::mat mtlB = {{1.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtlA = repmat(mtlA, 12, 1);
    mtlB = repmat(mtlB, 12, 1);
    arma::uvec indA, indB;
    std::unordered_map<std::string, std::vector<double>> mapA, mapB;
    mtl_matrix_to_map<double>(mtlA, indA, mapA);
    set_mu<double>(mapA, 1.7, 0.0, 0.0, 0.0); // A: eps=2.5, mu=1.7
    mtl_matrix_to_map<double>(mtlB, indB, mapB);
    set_mu<double>(mapB, 2.5, 0.0, 0.0, 0.0); // B: eps=1.7, mu=2.5 (swapped)

    arma::mat oA, dA, xA, tvA, tdA, oB, dB, xB, tvB, tdB;
    arma::vec gA, gB;
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oA, &dA, (arma::mat *)nullptr, (arma::mat *)nullptr, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &oB, &dB, (arma::mat *)nullptr, (arma::mat *)nullptr, &gB, &xB, &tvB, &tdB);

    CHECK(arma::approx_equal(gA, gB, "absdiff", 1e-10));
    CHECK(gA(0) > 1e-6); // non-trivial reflection, so the equality is not vacuous
}

TEST_CASE("Ray-Mesh Interact - Permeability drives in-medium loss (medium_gain)", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;

    // Ray starts INSIDE and reflects off the east wall, so the in-medium path accrues loss.
    // eps is real; all bulk loss lives in mu's imaginary part.
    arma::mat orig = {{0.5, 0.1, 0.0}};
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}};
    tridir = tridir * deg2rad;

    // eps = 1.5 (real), mu = 1 - j*(17.98*0.003/10) at fRef = 1, exponents 0
    arma::mat mtl = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl = repmat(mtl, 12, 1);
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> map;
    mtl_matrix_to_map<double>(mtl, ind, map);
    set_mu<double>(map, 1.0, 0.0, 0.003, 0.0);

    arma::mat o, d, x, tv, td;
    arma::vec g;
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &ind, &map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, &o, &d, (arma::mat *)nullptr, (arma::mat *)nullptr, &g, &x, &tv, &td);

    std::complex<double> eta1(1.5, 0.0), eta2(1.0, 0.0);
    std::complex<double> mu1(1.0, -17.98 * 0.003 / 10.0), mu2(1.0, 0.0);
    double thickness = std::sqrt(0.5 * 0.5 + 0.5 * 0.5) + 0.001; // in-medium path + ray_offset

    double full = calc_transition_gain_mu(0, 45.0, thickness, 0.0, eta1, eta2, mu1, mu2);
    double refl_only = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, eta1, eta2, mu1, mu2);

    // The reflection gain is the loss-free admittance term; the bulk loss from mu is medium_gain's.
    arma::vec U = {refl_only};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9));

    double bulk = full / refl_only; // the loss factor medium_gain carries
    CHECK(std::abs(quadriga_lib::medium_gain(map, 1, thickness, 10.0e9) - bulk) < 1e-9);
    CHECK(bulk < 0.7); // mu attenuates the path; eps alone is real here
}

TEST_CASE("Ray-Mesh Interact - path_dirN direction contract (45° dielectric, x-z plane)", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // 45° incidence on the West face
    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // eps_r = 1.5
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    arma::mat origN, destN, path_dirN;
    arma::u32_vec ray_indN;

    auto run = [&](int type)
    {
        quadriga_lib::ray_mesh_interact(
            type, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
            (arma::mat *)nullptr, (arma::mat *)nullptr,
            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
            (arma::vec *)nullptr, (arma::mat *)nullptr,
            (arma::mat *)nullptr, (arma::mat *)nullptr,
            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
            (arma::mat *)nullptr, (std::vector<uint8_t> *)nullptr,
            &path_dirN, true, &ray_indN);
    };

    // Manual unit-norm helpers (avoid arma::norm / arma::normalise, which pull in BLAS)
    auto vnorm = [](const arma::rowvec &v)
    { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); };
    auto dir = [&](const arma::mat &m) -> arma::rowvec
    { arma::rowvec v = m.row(0); return v / vnorm(v); };

    arma::rowvec incoming = dir(dest - orig); // origin->FBS direction (== origin->dest here)

    // Reflection (type 0): mirror direction == normalized (destN - origN), unit norm
    run(0);
    arma::rowvec pd_refl = path_dirN.row(0);
    CHECK(arma::approx_equal(pd_refl, dir(destN - origN), "absdiff", 1e-6));
    CHECK(std::abs(vnorm(pd_refl) - 1.0) < 1e-6);

    // Refraction (type 2): Snell direction == normalized (destN - origN), unit norm
    run(2);
    arma::rowvec pd_refr = path_dirN.row(0);
    CHECK(arma::approx_equal(pd_refr, dir(destN - origN), "absdiff", 1e-6));
    CHECK(std::abs(vnorm(pd_refr) - 1.0) < 1e-6);

    // Undeviated transmission (type 1): path_dirN is the SNELL direction (same as type 2),
    // NOT the incoming direction. The geometric path (destN - origN) stays along the incoming ray.
    run(1);
    arma::rowvec pd_trans = path_dirN.row(0);
    CHECK(std::abs(vnorm(pd_trans) - 1.0) < 1e-6);
    CHECK(arma::approx_equal(pd_trans, pd_refr, "absdiff", 1e-6));            // == Snell direction
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-6)); // geometry undeviated
    CHECK(vnorm(pd_trans - incoming) > 1e-2);                                 // and differs from incoming

    // Scalar reflection (type 3): mirror direction, identical to type 0
    run(3);
    arma::rowvec pd_srefl = path_dirN.row(0);
    CHECK(arma::approx_equal(pd_srefl, dir(destN - origN), "absdiff", 1e-6));
    CHECK(arma::approx_equal(pd_srefl, pd_refl, "absdiff", 1e-6));
    CHECK(std::abs(vnorm(pd_srefl) - 1.0) < 1e-6);

    // Scalar transmission (type 4): Snell direction, identical to type 2; geometry undeviated
    run(4);
    arma::rowvec pd_strans = path_dirN.row(0);
    CHECK(std::abs(vnorm(pd_strans) - 1.0) < 1e-6);
    CHECK(arma::approx_equal(pd_strans, pd_refr, "absdiff", 1e-6));
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - ray_indN compaction round-trip", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // air (irrelevant to ray_indN)
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    auto interact = [&](const arma::mat &orig, const arma::mat &dest,
                        arma::u32_vec &fbs_ind, arma::u32_vec &ray_indN)
    {
        arma::mat fbs, sbs;
        arma::u32_vec sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat origN, destN, path_dirN;
        quadriga_lib::ray_mesh_interact(
            1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
            (arma::mat *)nullptr, (arma::mat *)nullptr,
            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
            (arma::vec *)nullptr, (arma::mat *)nullptr,
            (arma::mat *)nullptr, (arma::mat *)nullptr,
            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
            (arma::mat *)nullptr, (std::vector<uint8_t> *)nullptr,
            &path_dirN, true, &ray_indN);
    };

    SECTION("mixed hits and misses")
    {
        // Rays 0 and 2 hit the West face; ray 1 is offset in y and misses the cube.
        arma::mat orig = {{-1.5, 0.0, 0.0}, {-1.5, 3.0, 0.0}, {-1.5, 0.5, 0.5}};
        arma::mat dest = {{0.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.5, 0.5}};

        arma::u32_vec fbs_ind, ray_indN;
        interact(orig, dest, fbs_ind, ray_indN);

        // Round-trip oracle: ascending list of surviving (fbs_ind != 0) input indices, 0-based.
        std::vector<unsigned> expected;
        for (arma::uword i = 0; i < fbs_ind.n_elem; ++i)
            if (fbs_ind(i) != 0)
                expected.push_back((unsigned)i);

        REQUIRE(expected.size() == 2); // ray 1 missed
        REQUIRE(ray_indN.n_elem == expected.size());
        for (size_t k = 0; k < expected.size(); ++k)
            CHECK(ray_indN(k) == expected[k]); // inverse map, 0-based, order-preserving
    }

    SECTION("identity when all rays survive")
    {
        arma::mat orig = {{-1.5, 0.0, 0.0}, {-1.5, 0.5, 0.0}, {-1.5, -0.5, 0.0}};
        arma::mat dest = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, -0.5, 0.0}};

        arma::u32_vec fbs_ind, ray_indN;
        interact(orig, dest, fbs_ind, ray_indN);

        REQUIRE(fbs_ind.min() != 0); // all three survive
        REQUIRE(ray_indN.n_elem == 3);
        for (arma::uword k = 0; k < ray_indN.n_elem; ++k)
            CHECK(ray_indN(k) == (unsigned)k);
    }
}

TEST_CASE("Ray-Mesh Interact - Scalar interactions (types 3/4/5) gains, invariant, interface gain", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;
    arma::mat orig = {{0.5, 0.1, 0.0}}; // inside, 45 deg on the East face (dense -> air)
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat origN, destN, xprmatN;
    arma::vec gainN, U;

    auto run = [&](int type)
    {
        quadriga_lib::ray_mesh_interact(type, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr);
    };
    auto vnorm = [](const arma::rowvec &v)
    { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); };
    auto dir = [&](const arma::mat &m) -> arma::rowvec
    { arma::rowvec v = m.row(0); return v / vnorm(v); };
    auto coeff_energy = [&]()
    { return xprmatN(0) * xprmatN(0) + xprmatN(1) * xprmatN(1); };
    arma::rowvec incoming = dir(fbs - orig);

    // Scalar pressure coefficients: R == EM TE coeff (z = sqrt(eps), mu = 1), T = 1 + R
    double cos_th = std::cos(45.0 * deg2rad);
    double cos_th2 = std::sqrt(1.0 - 1.5 * cos_th * cos_th); // 0.5
    double sqrt_eps = std::sqrt(1.5);
    double R = (sqrt_eps * cos_th - cos_th2) / (sqrt_eps * cos_th + cos_th2);

    arma::mat mtl_prop = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Type 3 - scalar reflection: gain = |R|^2, single scalar coefficient; scalar xprmat is [2, n_rayN]
    run(3);
    U = {R * R};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));
    CHECK(std::abs(gainN(0) - coeff_energy()) < 1e-6);
    CHECK(xprmatN.n_rows == 2);
    CHECK(xprmatN.n_cols == 1);

    // Type 4 - scalar transmission (undeviated): gain = 1 - |R|^2
    run(4);
    U = {1.0 - R * R};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));
    CHECK(std::abs(gainN(0) - coeff_energy()) < 1e-6);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5));

    // Type 5 (refraction) must carry the interface penetration loss, same factor as type 4.
    // interface_gain comes from the ENTERED medium, so this has to enter the lossy cube from air.
    arma::mat orig_in = {{-1.5, 0.0, 0.0}}, dest_in = {{0.0, 0.0, 1.5}}; // 45 deg into the West face
    arma::mat fbs_in, sbs_in;
    arma::u32_vec fbs_in_ind, sbs_in_ind;
    quadriga_lib::ray_triangle_intersect(&orig_in, &dest_in, &cube, &fbs_in, &sbs_in, NULL, &fbs_in_ind, &sbs_in_ind);

    auto gain_of = [&](int type, double att)
    {
        arma::mat mp = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, att, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);
        quadriga_lib::ray_mesh_interact(type, 10.0e9, &orig_in, &dest_in, &cube, &mtl_ind, &mtl_map,
                                        &fbs_in_ind, &sbs_in_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr);
        return gainN(0);
    };
    double t4_ratio = gain_of(4, 3.0) / gain_of(4, 0.0);
    double t5_ratio = gain_of(5, 3.0) / gain_of(5, 0.0);
    CHECK(std::abs(t4_ratio - t5_ratio) < 1e-6); // type 5 gets the same interface gain as type 4
    CHECK(t5_ratio < 0.99);                      // and it is actually applied (att attenuates)

    // Genuine spine TIR (eps = 2.5 at 45 deg): scalar refraction (5) collapses to undeviated, gain 0 (tf=0)
    mtl_prop = repmat(arma::mat{{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    run(5);
    CHECK(std::abs(gainN(0)) < 1e-6);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5));
}

TEST_CASE("Ray-Mesh Interact - Transmission factor splits reflection/transmission energy", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    arma::mat orig = {{0.5, 0.1, 0.0}};
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat mtl_prop = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat origN, destN, xprmatN;
    arma::vec gainN;

    double deg2rad = arma::datum::pi / 180.0;
    double cos_th = std::cos(45.0 * deg2rad);
    double cos_th2 = std::sqrt(1.0 - 1.5 * cos_th * cos_th);
    double se = std::sqrt(1.5);
    double R_te = (se * cos_th - cos_th2) / (se * cos_th + cos_th2);
    double R_tm = (cos_th - se * cos_th2) / (cos_th + se * cos_th2);
    double reflectance = 0.5 * (R_te * R_te + R_tm * R_tm);

    auto build = [&](double tf)
    { mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map); set_tf<double>(mtl_map, tf, 0.0); };
    auto run = [&](int type)
    {
        quadriga_lib::ray_mesh_interact(type, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr);
        return gainN(0);
    };

    // tf = 0: baseline, ports complementary
    build(0.0);
    double r0 = run(0), t0 = run(1);
    CHECK(std::abs(r0 - reflectance) < 1e-6);
    CHECK(std::abs(t0 - (1.0 - reflectance)) < 1e-6);
    CHECK(std::abs(r0 + t0 - 1.0) < 1e-6);

    // tf = +0.5: reflected energy leaks forward; still sums to 1
    build(0.5);
    double rP = run(0), tP = run(1);
    CHECK(std::abs(rP - reflectance * 0.5) < 1e-6);
    CHECK(std::abs(tP - (1.0 - reflectance * 0.5)) < 1e-6);
    CHECK(std::abs(rP + tP - 1.0) < 1e-6);
    CHECK(rP < r0);
    CHECK(tP > t0);

    // tf = -0.5: forces extra reflection; still sums to 1
    build(-0.5);
    double R_eff_m = reflectance + (1.0 - reflectance) * 0.5;
    double rM = run(0), tM = run(1);
    CHECK(std::abs(rM - R_eff_m) < 1e-6);
    CHECK(std::abs(tM - (1.0 - R_eff_m)) < 1e-6);
    CHECK(rM > r0);

    // EM reflection carries tf into its coefficients: gain == 1/2 sum|xprmat|^2
    build(0.5);
    run(0);
    double ce = 0.0;
    for (int k = 0; k < 8; ++k)
        ce += xprmatN(k) * xprmatN(k);
    CHECK(std::abs(gainN(0) - 0.5 * ce) < 1e-6);
}

TEST_CASE("Ray-Mesh Interact - Ray-tube TIR straddle forces whole-tube pass-through", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;
    arma::mat orig = {{0.5, 0.1, 0.0}}; // spine 45 deg on East face
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat mtl_prop = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat origN, destN, xprmatN, trivecN, tridirN;
    arma::vec gainN;

    auto vnorm = [](const arma::rowvec &v)
    { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); };
    auto dir = [&](const arma::mat &m) -> arma::rowvec
    { arma::rowvec v = m.row(0); return v / vnorm(v); };
    arma::rowvec incoming = dir(fbs - orig); // (0.707, 0.707, 0)

    auto run = [&](arma::mat &tridir)
    {
        quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map,
                                        &fbs_ind, &sbs_ind, &trivec, &tridir,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);
    };

    // Baseline: all edges parallel to the spine (45 deg) -> clean refraction, spine bends
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    arma::mat tridir_clean = arma::mat{{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}} * deg2rad;
    run(tridir_clean);
    CHECK(gainN(0) > 0.1);
    CHECK(vnorm(dir(destN - origN) - incoming) > 1e-2);
    const double gain_spine = gainN(0);

    // Straddle: one edge at 60 deg (eps=1.5 critical ~54.7 deg) -> whole tube passes through undeviated
    arma::mat tridir_straddle = arma::mat{{60.0, 0.0, 45.0, 0.0, 45.0, 0.0}} * deg2rad;
    run(tridir_straddle);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5)); // undeviated
    CHECK(std::abs(gainN(0) - gain_spine) < 1e-12);                           // the coefficient stays the spine's: the verdict is the tube's

    // Same straddle with tf = 0.5: still undeviated, with the spine's coefficient
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    set_tf<double>(mtl_map, 0.5, 0.0);
    run(tridir_clean);
    const double gain_spine_tf = gainN(0);
    run(tridir_straddle);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5));
    CHECK(std::abs(gainN(0) - gain_spine_tf) < 1e-12);
}

TEST_CASE("Ray-Mesh Interact - path_dirN collapses to undeviated under spine TIR", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    arma::mat orig = {{0.5, 0.1, 0.0}};
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat mtl_prop = repmat(arma::mat{{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1); // TIR at 45 deg
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    arma::mat origN, destN, path_dirN;
    arma::u32_vec ray_indN;
    auto run = [&](int type)
    {
        quadriga_lib::ray_mesh_interact(type, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, (std::vector<uint8_t> *)nullptr, &path_dirN, true, &ray_indN);
    };
    auto vnorm = [](const arma::rowvec &v)
    { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); };
    auto dir = [&](const arma::mat &m) -> arma::rowvec
    { arma::rowvec v = m.row(0); return v / vnorm(v); };
    arma::rowvec incoming = dir(fbs - orig);

    // Reflection (0): mirror, unaffected by TIR
    run(0);
    CHECK(arma::approx_equal(path_dirN.row(0), dir(destN - origN), "absdiff", 1e-6));
    arma::rowvec mirror = path_dirN.row(0);
    CHECK(vnorm(mirror - incoming) > 1e-2);

    // Transmission/refraction (1, 2, 4, 5) under spine TIR: path_dirN == undeviated incoming
    for (int type : {1, 2, 4, 5})
    {
        run(type);
        CHECK(arma::approx_equal(path_dirN.row(0), incoming, "absdiff", 1e-6));
        CHECK(std::abs(vnorm(path_dirN.row(0)) - 1.0) < 1e-6);
    }

    // Scalar reflection (3): still mirror
    run(3);
    CHECK(arma::approx_equal(path_dirN.row(0), mirror, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Spherical ray-tube keeps undeviated directions on pass-through", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();
    double deg2rad = arma::datum::pi / 180.0;
    arma::mat orig = {{0.5, 0.1, 0.0}};
    arma::mat dest = {{2.0, 1.6, 0.0}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = arma::mat{{45.0, 0.0, 45.0, 0.0, 45.0, 0.0}} * deg2rad; // spherical az/el
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat origN, destN, xprmatN, trivecN, tridirN;
    arma::vec gainN;

    auto run = [&](int type)
    {
        quadriga_lib::ray_mesh_interact(type, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map,
                                        &fbs_ind, &sbs_ind, &trivec, &tridir,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr, &gainN, &xprmatN, &trivecN, &tridirN);
    };

    // Undeviated transmission (type 1): vertex az/el unchanged (== input), NOT reset to (0,0)
    arma::mat mtl_prop = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    run(1);
    CHECK(arma::approx_equal(tridirN, tridir, "absdiff", 1e-6));

    // Refraction under genuine spine TIR (eps = 2.5): also passes through with input az/el
    mtl_prop = repmat(arma::mat{{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    run(2);
    CHECK(arma::approx_equal(tridirN, tridir, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain", "[wave]")
{
    // refractive_index = Re(sqrt(eps_r * mu_r)); air (medium 0) is 1
    {
        arma::mat m = repmat(arma::mat{{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> map;
        mtl_matrix_to_map<double>(m, ind, map); // eps = 4, mu = 1 -> n = 2
        CHECK(std::abs(quadriga_lib::refractive_index(map, 1, 10.0e9) - 2.0) < 1e-9);
        CHECK(std::abs(quadriga_lib::refractive_index(map, 0, 10.0e9) - 1.0) < 1e-12); // air
    }
    {
        arma::mat m = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> map;
        mtl_matrix_to_map<double>(m, ind, map); // eps = 1.5, mu = 1 -> n = sqrt(1.5)
        CHECK(std::abs(quadriga_lib::refractive_index(map, 1, 10.0e9) - std::sqrt(1.5)) < 1e-9);
    }
    {
        arma::mat m = repmat(arma::mat{{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> map;
        mtl_matrix_to_map<double>(m, ind, map);
        set_mu<double>(map, 4.0, 0.0, 0.0, 0.0); // eps = 4, mu = 4 -> n = 4 (mu participates)
        CHECK(std::abs(quadriga_lib::refractive_index(map, 1, 10.0e9) - 4.0) < 1e-9);
    }
    {
        // dispersion through b: eps = a (f / fRef)^b; at 2 fRef, eps = 4 * 4 = 16 -> n = 4
        std::unordered_map<std::string, std::vector<double>> map;
        map["a"] = {4.0}, map["b"] = {2.0}, map["fRef"] = {5.0};
        CHECK(std::abs(quadriga_lib::refractive_index(map, 1, 10.0e9) - 4.0) < 1e-9);

        // The coincidence columns leave a wave medium's index and interface gain alone: coiF and coiQ act only on a
        // mass-law plate, and coiA is no column (it is read and ignored)
        std::unordered_map<std::string, std::vector<double>> coi = map;
        coi["coiF"] = {20.0}, coi["coiQ"] = {5.0}, coi["coiA"] = {6.0};
        CHECK(std::abs(quadriga_lib::refractive_index(coi, 1, 10.0e9) - quadriga_lib::refractive_index(map, 1, 10.0e9)) < 1e-12);
        CHECK(quadriga_lib::interface_gain(coi, 1, 20.0e9) == quadriga_lib::interface_gain(map, 1, 20.0e9)); // at coiF
    }

    // interface_gain: att = 6 dB @ 2 GHz, attB = 1 -> 30 dB @ 10 GHz -> 1e-3; air is 1
    {
        arma::mat m = repmat(arma::mat{{1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 2.0}}, 12, 1);
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> map;
        mtl_matrix_to_map<double>(m, ind, map);
        CHECK(std::abs(quadriga_lib::interface_gain(map, 1, 10.0e9) - 1.0e-3) < 1e-9);
        CHECK(std::abs(quadriga_lib::interface_gain(map, 0, 10.0e9) - 1.0) < 1e-12); // air
    }

    // medium_gain: eps = 1, sigma = 0, alpha = 10 dB/m (alphaB = 0), dist = 2 -> 20 dB -> 1e-2
    {
        arma::mat m = repmat(arma::mat{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1.0}}, 12, 1);
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> map;
        mtl_matrix_to_map<double>(m, ind, map);
        CHECK(std::abs(quadriga_lib::medium_gain(map, 1, 2.0, 10.0e9) - 1.0e-2) < 1e-9);
        CHECK(std::abs(quadriga_lib::medium_gain(map, 0, 2.0, 10.0e9) - 1.0) < 1e-12); // air
    }
}

TEST_CASE("Ray-Mesh Interact - out_typeN bit encoding", "[wave]")
{
    arma::mat origN, destN;
    arma::vec gainN;

    // Entry: air -> dielectric, front face hit -> OK|front = 3
    {
        arma::mat cube = quadriga_lib::cube<double>();
        arma::mat orig = {{-1.5, 0.0, 0.0}}; // outside, 45 deg into the West face
        arma::mat dest = {{0.0, 0.0, 1.5}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat mp = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl_map;
        mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);

        std::vector<uint8_t> otype;
        quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::mat *)nullptr, &otype);
        REQUIRE(otype.size() == 1);
        CHECK((int)otype[0] == 3); // OK(1) | front(2)
    }

    // Exit: inside -> air, back face hit, no TIR (eps = 1.5 @ 45 deg) -> OK = 1
    {
        arma::mat cube = quadriga_lib::cube<double>();
        arma::mat orig = {{0.5, 0.1, 0.0}}; // inside, 45 deg on the East face
        arma::mat dest = {{2.0, 1.6, 0.0}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat mp = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl_map;
        mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);

        std::vector<uint8_t> otype;
        quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::mat *)nullptr, &otype);
        REQUIRE(otype.size() == 1);
        CHECK((int)otype[0] == 1);        // OK, back face, no TIR
        CHECK(((int)otype[0] & 32) == 0); // TIR bit clear
    }

    // Media-to-media: co-located shared wall between two cubes -> co-located bit (4) set
    {
        arma::mat msh = quadriga_lib::cube<double>();
        arma::mat tmp = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
        msh = arma::join_cols(msh, tmp);

        arma::mat mp = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::mat mp2 = repmat(arma::mat{{1.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        mp = arma::join_cols(mp, mp2);
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl_map;
        mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);

        arma::mat orig = {{0.5, 0.1, 0.0}}; // inside cube 1, crossing into cube 2 at x = 1
        arma::mat dest = {{2.0, 1.6, 0.0}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &msh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        std::vector<uint8_t> otype;
        quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::mat *)nullptr, &otype);
        REQUIRE(otype.size() == 1);
        CHECK(((int)otype[0] & 4) != 0); // co-located bit set

        bool o57 = ((int)otype[0] == 5) || ((int)otype[0] == 7);
        CHECK(o57); // OK|colocated with either front sense
    }

    // TIR: inside eps = 2.5, reflection at 45 deg -> base exit (1) | TIR (32) = 33
    {
        arma::mat cube = quadriga_lib::cube<double>();
        arma::mat orig = {{0.5, 0.1, 0.0}};
        arma::mat dest = {{2.0, 1.6, 0.0}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat mp = repmat(arma::mat{{2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl_map;
        mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);

        std::vector<uint8_t> otype;
        quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::mat *)nullptr, &otype);
        REQUIRE(otype.size() == 1);
        CHECK(((int)otype[0] & 32) != 0); // TIR bit set
        CHECK((int)otype[0] == 33);       // exit(1) | TIR(32)
    }
}

TEST_CASE("Ray-Mesh Interact - compact = false keeps missed rays in place", "[wave]")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // Rays 0 and 2 hit the West face; ray 1 (offset in y) misses the cube.
    arma::mat orig = {{-1.5, 0.0, 0.0}, {-1.5, 3.0, 0.0}, {-1.5, 0.5, 0.5}};
    arma::mat dest = {{0.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.5, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
    REQUIRE(fbs_ind(1) == 0); // ray 1 really misses

    arma::mat mp = repmat(arma::mat{{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mp, mtl_ind, mtl_map);

    arma::mat origN, destN, xprmatN;
    arma::vec gainN;
    std::vector<uint8_t> otype;
    arma::u32_vec ray_indN;

    // compact = false: all three rays are retained, in input order
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    (arma::mat *)nullptr, (arma::mat *)nullptr,
                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                    &gainN, &xprmatN,
                                    (arma::mat *)nullptr, (arma::mat *)nullptr,
                                    (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                    (arma::mat *)nullptr, &otype,
                                    (arma::mat *)nullptr, false, &ray_indN);

    REQUIRE(gainN.n_elem == 3);
    REQUIRE(otype.size() == 3);
    REQUIRE(xprmatN.n_cols == 3);
    REQUIRE(ray_indN.n_elem == 3);

    // ray_indN is the identity map when nothing is dropped
    for (arma::uword k = 0; k < 3; ++k)
        CHECK(ray_indN(k) == (unsigned)k);

    // The missed ray (column 1) passes through untouched
    CHECK(std::abs(gainN(1) - 1.0) < 1e-9);
    CHECK((int)otype[1] == 0);
    arma::vec identity = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    CHECK(arma::approx_equal(xprmatN.col(1), identity, "absdiff", 1e-9));

    // The surviving rays actually interacted (OK bit set, reflected power < 1)
    CHECK(((int)otype[0] & 1) != 0);
    CHECK(((int)otype[2] & 1) != 0);
    CHECK(gainN(0) < 1.0);
    CHECK(gainN(2) < 1.0);
}

TEST_CASE("Diagnostic - Interface energy split per polarization", "[wave]")
{
    // The averaged reflectance and the channels at a single interface. interact_with builds the gain from 0.5 (|R_TE|^2
    // + |R_TM|^2) and hands over the exact per-channel coefficients, so each channel is its own Fresnel value and the
    // gain is their average.
    //
    // Channel identification: the ray lies in the x-y plane and the normal is along x, so eQ = U x N is parallel to z,
    // and so is eV. The polarization basis is then a clean swap rather than a mixing rotation: VV is the TE channel and
    // HH the TM channel, up to sign.
    const double f_hz = 1.0e9;
    const std::complex<double> eps_s(5.24, -0.63); // lossy concrete-like at 1 GHz
    arma::mat mesh = qd_quad_x(200.0, 0.0);

    arma::uvec mtl_ind(2, arma::fill::value(1));
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl["a"] = {5.24};
    mtl["b"] = {0.0};
    mtl["c"] = {0.63 / 17.98}; // eta_i = -17.98 * c / fGHz = -0.63 at 1 GHz
    mtl["d"] = {0.0};
    mtl["att"] = {0.0};
    mtl["attB"] = {0.0};
    mtl["alpha"] = {0.0};
    mtl["alphaB"] = {0.0};
    mtl["fRef"] = {1.0};

    QD_DIAG("\n# diag: interface energy split per polarization, air -> eps 5.24 - 0.63j, 1 GHz\n");
    QD_DIAG("type,angle,gain,ref_avg,VV2_te,HH2_tm,ref_TE,ref_TM,ratio_code,ratio_ref,d_ratio\n");

    for (int type : {0, 2})
        for (double thd : {0.0, 20.0, 40.0, 60.0, 70.0, 80.0, 85.0})
        {
            arma::mat orig, dest;
            qd_ray_at(thd, orig, dest);
            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

            arma::mat origN, destN, xprmatN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(type, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr);

            REQUIRE(gainN.n_elem == 1);
            double vv2 = xprmatN(0, 0) * xprmatN(0, 0) + xprmatN(1, 0) * xprmatN(1, 0);
            double hh2 = xprmatN(6, 0) * xprmatN(6, 0) + xprmatN(7, 0) * xprmatN(7, 0);

            QdIfaceRef F = qd_iface_ref(std::complex<double>(1.0, 0.0), eps_s, thd * arma::datum::pi / 180.0);
            // Reflection stays in one medium, where the obliquity factor is 1; transmission carries
            // the per-channel normal-flux transmittance, which for this lossless incidence medium is
            // identically 1 - R_X.
            double ref_te = (type == 0) ? F.R_te : F.F_te; // pairs with VV
            double ref_tm = (type == 0) ? F.R_tm : F.F_tm; // pairs with HH
            double ref_avg = (type == 0) ? F.R_avg : F.F_avg;

            double lo_c = std::min(vv2, hh2), hi_c = std::max(vv2, hh2);
            double lo_r = std::min(ref_tm, ref_te), hi_r = std::max(ref_tm, ref_te);
            double rc = (hi_c > 0.0) ? lo_c / hi_c : 0.0;
            double rr = (hi_r > 0.0) ? lo_r / hi_r : 0.0;

            QD_DIAG("%d,%.1f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e\n",
                    type, thd, gainN(0), ref_avg, vv2, hh2, ref_te, ref_tm, rc, rr, rc - rr);

            CHECK(std::isfinite(gainN(0)));
            CHECK(gainN(0) >= 0.0);
            CHECK(std::abs(gainN(0) - 0.5 * (vv2 + hh2)) < 1e-9); // gain is derived from xprmat

            // interact_with hands over exact per-channel coefficients; the averaged reflectance sets only the total, so
            // the channel ratio is the Fresnel ratio (measured to 8e-16).
            INFO("type " << type << ", angle " << thd);
            CHECK(std::abs(gainN(0) - ref_avg) < 1e-12);
            CHECK(std::abs(vv2 - ref_te) < 1e-12);
            CHECK(std::abs(hh2 - ref_tm) < 1e-12);
            CHECK(std::abs(rc - rr) < 1e-12);
        }
}

TEST_CASE("Diagnostic - Transmission convention baseline, single interface", "[wave]")
{
    // The absolute level of every transmission-class type at one interface. Every other table measures a ratio or a
    // composite, and a ratio is nearly blind to the obliquity factor.
    //
    // Two references beside the output:
    //   raw_X  = |t_X|^2, the Fresnel amplitude power with no obliquity factor
    //   flux_X = Re(z2 c2)/Re(z1 c1) |t_TE|^2  and  Re(conj(z2) c2)/Re(conj(z1) c1) |t_TM|^2, the per-channel
    //   normal-flux transmittance
    //
    // Asserted: every type (1, 2, 4, 5) in both directions reports flux_X, total reflection excluded; d_raw_dB is
    // printed, the size of the obliquity factor. The m2a rows are the lossy-incidence case, where flux_X and 1 - R_X
    // separate and flux_X is not bounded by 1; in the a2m rows the two coincide.
    //
    // Channel identification: VV is TE and HH is TM in both hit directions (the back-side hit flips the normal before
    // the basis is built). Scalar types carry one coefficient; VV2 is repeated into HH2 and both references use the TE
    // form, which is the scalar branch's: z is the acoustic admittance and T = 1 + R equals t_TE.
    const double f_hz = 1.0e9;
    const double fGHz = f_hz * 1e-9;

    struct Mat
    {
        const char *name;
        double a, c; // eps_r at fRef, and the sigma column: eta_i = -17.98 * c / fGHz
    };
    const std::vector<Mat> mats = {
        {"lossless eps 1.50", 1.5, 0.0},
        {"lossy eps 1.50 - 0.40j", 1.5, 0.40 / 17.98},
        {"lossy eps 5.24 - 0.63j", 5.24, 0.63 / 17.98},
        {"lossy eps 25.0 - 8.00j", 25.0, 8.00 / 17.98},
    };

    // Air to medium reaches grazing incidence. Medium to air stops at 50 deg, because the denser
    // materials are past their internal critical angle well before that and every row beyond it
    // reports total reflection instead of a transmittance.
    const std::vector<double> ang_a2m = {0.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    const std::vector<double> ang_m2a = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0};

    auto dB_ratio = [](double num, double den) -> double
    {
        if (!(num > 0.0) || !(den > 0.0))
            return std::numeric_limits<double>::quiet_NaN();
        return 10.0 * std::log10(num / den);
    };
    auto ratio_of = [](double a, double b)
    { double lo = std::min(a, b), hi = std::max(a, b); return (hi > 0.0) ? lo / hi : 0.0; };

    QD_DIAG("\n# diag: transmission convention baseline, single interface, 1 GHz\n");
    QD_DIAG("# dir a2m = air -> medium (front-side hit), m2a = medium -> air (back-side hit)\n");
    QD_DIAG("# raw = |t|^2; flux = Re(z2 c2)/Re(z1 c1) |t_TE|^2, Re(z2* c2)/Re(z1* c1) |t_TM|^2\n");
    QD_DIAG("material,dir,type,angle,tir,gain,VV2_te,HH2_tm,raw_te,raw_tm,flux_te,flux_tm,"
            "d_raw_dB,d_flux_dB,ratio_code,ratio_flux\n");

    for (const auto &M : mats)
    {
        std::unordered_map<std::string, std::vector<double>> mtl;
        mtl["a"] = {M.a};
        mtl["b"] = {0.0};
        mtl["c"] = {M.c};
        mtl["d"] = {0.0};
        mtl["att"] = {0.0};
        mtl["attB"] = {0.0};
        mtl["alpha"] = {0.0};
        mtl["alphaB"] = {0.0};
        mtl["fRef"] = {1.0};

        const std::complex<double> eps_s(M.a, -17.98 * M.c / fGHz);
        const std::complex<double> eps_air(1.0, 0.0);

        for (int dir = 0; dir < 2; ++dir)
        {
            arma::mat mesh = (dir == 0) ? qd_quad_x(200.0, 0.0) : qd_quad_x_rev(200.0, 0.0);
            arma::uvec mtl_ind(2, arma::fill::value(1));
            const std::complex<double> e1 = (dir == 0) ? eps_air : eps_s;
            const std::complex<double> e2 = (dir == 0) ? eps_s : eps_air;
            const std::vector<double> &angles = (dir == 0) ? ang_a2m : ang_m2a;

            for (int type : {1, 2, 4, 5})
                for (double thd : angles)
                {
                    arma::mat orig, dest;
                    qd_ray_at(thd, orig, dest);
                    arma::mat fbs, sbs;
                    arma::u32_vec fbs_ind, sbs_ind;
                    quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

                    arma::mat origN, destN, xprmatN;
                    arma::vec gainN;
                    std::vector<uint8_t> otype;
                    quadriga_lib::ray_mesh_interact(type, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                                    &fbs_ind, &sbs_ind,
                                                    (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                    &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                    &gainN, &xprmatN,
                                                    (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                    (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                                    (arma::mat *)nullptr, &otype);

                    REQUIRE(gainN.n_elem == 1);
                    REQUIRE(otype.size() == 1);

                    const bool is_scalar = (type >= 3);
                    const bool tir = ((int)otype[0] & 32) != 0;
                    const bool front = ((int)otype[0] & 2) != 0;

                    // The hit direction has to be the one the mesh winding was chosen for; if it is
                    // not, every reference below is built for the wrong pair of media.
                    REQUIRE(front == (dir == 0));

                    double vv2 = xprmatN(0, 0) * xprmatN(0, 0) + xprmatN(1, 0) * xprmatN(1, 0);
                    double hh2 = is_scalar ? vv2
                                           : xprmatN(6, 0) * xprmatN(6, 0) + xprmatN(7, 0) * xprmatN(7, 0);

                    QdIfaceRef F = qd_iface_ref(e1, e2, thd * arma::datum::pi / 180.0);
                    double raw_te = F.T_te, raw_tm = is_scalar ? F.T_te : F.T_tm;
                    double flux_te = F.F_te, flux_tm = is_scalar ? F.F_te : F.F_tm;
                    double raw_avg = is_scalar ? raw_te : 0.5 * (raw_te + raw_tm);
                    double flux_avg = is_scalar ? flux_te : 0.5 * (flux_te + flux_tm);

                    QD_DIAG("%s,%s,%d,%.1f,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.4f,%.4f,%.9f,%.9f\n",
                            M.name, (dir == 0) ? "a2m" : "m2a", type, thd, tir ? 1 : 0,
                            gainN(0), vv2, hh2, raw_te, raw_tm, flux_te, flux_tm,
                            dB_ratio(gainN(0), raw_avg), dB_ratio(gainN(0), flux_avg),
                            ratio_of(vv2, hh2), ratio_of(flux_te, flux_tm));

                    INFO("material " << M.name << ", dir " << ((dir == 0) ? "a2m" : "m2a")
                                     << ", type " << type << ", angle " << thd);
                    CHECK(std::isfinite(gainN(0)));
                    CHECK(gainN(0) >= 0.0);
                    CHECK(std::abs(gainN(0) - (is_scalar ? vv2 : 0.5 * (vv2 + hh2))) < 1e-9);

                    // Every transmission-class type reports the per-channel normal-flux transmittance of its own
                    // interface, so the four types agree. Total reflection is excluded: there is no forward Fresnel
                    // port, and the coefficient carries a flat tf leak instead.
                    if (!tir)
                    {
                        CHECK(std::abs(vv2 - flux_te) < 1e-9);
                        CHECK(std::abs(hh2 - flux_tm) < 1e-9);
                        CHECK(std::abs(gainN(0) - (is_scalar ? flux_te : flux_avg)) < 1e-9);
                    }
                    else
                        CHECK(gainN(0) < 1e-12); // tf = 0 here, so the leak port is closed
                }
        }
    }
}

TEST_CASE("Diagnostic - Thin sheet composite vs analytic slab", "[wave]")
{
    // The sheet composite is produced in one event and equals the exact analytic slab. The polarization-ratio columns
    // are the sharper check: a power error can hide behind the TE/TM average, a ratio error cannot.
    //
    // The last row is magnetic: it reaches ray_mesh_interact's own lookup of the e, f, g, h columns and its
    // polarization basis, built from a real incidence plane. ray_state_update re-resolves sheet crossings when medium
    // tracking is on; this composite serves callers without state.
    //
    // Both ports are measured on every row, per channel and labelled: the min/max ratio and the average are blind to
    // which channel is which, and a magnetic sheet can order the two either way.
    const double f_hz = 1.0e9;

    struct Mat
    {
        const char *name;
        double a, c, thk;
        double e = 1.0, g = 0.0; // mu = e - j 17.98 g / fGHz
    };
    const std::vector<Mat> mats = {
        {"lossless eps 5.24, 100 mm", 5.24, 0.0, 0.100},
        {"lossy eps 5.24, 100 mm", 5.24, 0.63 / 17.98, 0.100},
        {"low eps 2.0, weak loss, 100 mm", 2.0, 0.25 / 17.98, 0.100},
        {"high eps 25.0, strong loss, 100 mm", 25.0, 2.50 / 17.98, 0.100},
        {"lossy eps 5.24, 6 mm", 5.24, 0.63 / 17.98, 0.006},
        {"magnetic eps 5.24 - 0.63j, mu 2.10 - 0.35j, 6 mm", 5.24, 0.63 / 17.98, 0.006, 2.10, 0.35 / 17.98},
    };

    QD_DIAG("\n# diag: thin sheet composite vs analytic slab, 1 GHz\n");
    QD_DIAG("material,angle,sheet_dB,ana_dB,d_dB,ratio_sheet,ratio_ana,d_ratio,arg_sheet_deg,arg_ana_te_deg,d_arg_deg,pred_k0_thk_cos,"
            "d_T_te,d_T_tm,d_R_te,d_R_tm\n");

    for (const auto &M : mats)
    {
        arma::mat mesh = qd_quad_x(200.0, 0.0);
        arma::uvec mtl_ind(2, arma::fill::value(1));
        std::unordered_map<std::string, std::vector<double>> mtl;
        mtl["a"] = {M.a};
        mtl["b"] = {0.0};
        mtl["c"] = {M.c};
        mtl["d"] = {0.0};
        mtl["att"] = {0.0};
        mtl["attB"] = {0.0};
        mtl["alpha"] = {0.0};
        mtl["alphaB"] = {0.0};
        mtl["fRef"] = {1.0};
        mtl["thk"] = {M.thk};
        const bool magnetic = (M.e != 1.0 || M.g != 0.0);
        if (magnetic) // the dielectric rows keep the map they always had
        {
            mtl["e"] = {M.e};
            mtl["f"] = {0.0};
            mtl["g"] = {M.g};
            mtl["h"] = {0.0};
        }
        std::complex<double> mu_s(M.e, -17.98 * M.g / (f_hz * 1e-9));
        std::complex<double> eps_s(M.a, -17.98 * M.c / (f_hz * 1e-9));

        for (double thd : {0.0, 20.0, 40.0, 60.0, 70.0, 80.0, 85.0, 88.0})
        {
            arma::mat orig, dest;
            qd_ray_at(thd, orig, dest);
            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

            arma::mat origN, destN, xprmatN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(1, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr);

            REQUIRE(gainN.n_elem == 1);

            // The reflect port, same ray, same mesh.
            arma::mat origR, destR, xprmatR;
            arma::vec gainR;
            quadriga_lib::ray_mesh_interact(0, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origR, &destR, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainR, &xprmatR, (arma::mat *)nullptr, (arma::mat *)nullptr);
            REQUIRE(gainR.n_elem == 1);
            double rvv2 = xprmatR(0, 0) * xprmatR(0, 0) + xprmatR(1, 0) * xprmatR(1, 0);
            double rhh2 = xprmatR(6, 0) * xprmatR(6, 0) + xprmatR(7, 0) * xprmatR(7, 0);
            double vv2 = xprmatN(0, 0) * xprmatN(0, 0) + xprmatN(1, 0) * xprmatN(1, 0);
            double hh2 = xprmatN(6, 0) * xprmatN(6, 0) + xprmatN(7, 0) * xprmatN(7, 0);
            double arg_vv = std::atan2(xprmatN(1, 0), xprmatN(0, 0)) * 180.0 / arma::datum::pi;

            QdSlabRef R = qd_slab_ref(eps_s, M.thk, thd * arma::datum::pi / 180.0, f_hz, mu_s);
            double Ra_te = 0.0, Ra_tm = 0.0;
            double Ra = qd_slab_ref_R(eps_s, M.thk, thd * arma::datum::pi / 180.0, f_hz, mu_s, &Ra_te, &Ra_tm);
            double lo_c = std::min(vv2, hh2), hi_c = std::max(vv2, hh2);
            double rc = (hi_c > 0.0) ? lo_c / hi_c : 0.0;
            // VV is the TE channel here (see the note in the interface case), so the phase has to be
            // compared against t_te. d_arg is predicted to be exactly k0 * thk * cos(theta_i): the
            // free-space reference the sheet subtracts and the analytic slab does not.
            double arg_ref = std::arg(R.t_te) * 180.0 / arma::datum::pi;
            double d_arg = arg_vv - arg_ref;
            while (d_arg > 180.0)
                d_arg -= 360.0;
            while (d_arg <= -180.0)
                d_arg += 360.0;

            auto dB = [](double p)
            { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

            double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
            double pred = k0 * M.thk * std::cos(thd * arma::datum::pi / 180.0) * 180.0 / arma::datum::pi;
            while (pred > 180.0)
                pred -= 360.0;
            while (pred <= -180.0)
                pred += 360.0;

            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.9f,%.9f,%.3e,%.2f,%.2f,%.2f,%.2f,%.2e,%.2e,%.2e,%.2e\n",
                    M.name, thd, dB(gainN(0)), dB(R.p_avg), dB(gainN(0)) - dB(R.p_avg),
                    rc, R.ratio, rc - R.ratio, arg_vv, arg_ref, d_arg, pred,
                    vv2 - R.p_te, hh2 - R.p_tm, rvv2 - Ra_te, rhh2 - Ra_tm);

            CHECK(std::isfinite(gainN(0)));
            CHECK(gainN(0) >= 0.0);
            CHECK(gainN(0) <= 1.0);

            // The sheet composite matches the exact analytic slab, and its phase sits one free-space reference term
            // away from it, across five materials and eight angles: the same closed form evaluated the same way, so the
            // bounds are numerical.
            INFO(M.name << ", angle " << thd);
            CHECK(std::abs(dB(gainN(0)) - dB(R.p_avg)) < 1.0e-6);
            CHECK(std::abs(rc - R.ratio) < 1.0e-9);
            CHECK(std::abs(d_arg - pred) < 1.0e-3);

            // Per channel and labelled, both ports. VV is TE and HH is TM in this geometry, for the reflected direction
            // as well as the transmitted one; the magnetic row is held to the same bounds as the dielectric ones.
            CHECK(std::abs(vv2 - R.p_te) < 1.0e-9);
            CHECK(std::abs(hh2 - R.p_tm) < 1.0e-9);
            CHECK(std::abs(rvv2 - Ra_te) < 1.0e-9);
            CHECK(std::abs(rhh2 - Ra_tm) < 1.0e-9);
            CHECK(std::abs(gainR(0) - Ra) < 1.0e-9);

            // The magnetic row has to be one a dielectric reference cannot pass. With mu dropped
            // from the oracle the transmit port sits 0.077 dB to 5.94 dB away. The reflect port is
            // compared per channel: the two averaged reflectances cross near 60 deg, where they
            // agree to 2e-4, so the average is the wrong instrument there and the channels are not.
            if (magnetic)
            {
                QdSlabRef Rd = qd_slab_ref(eps_s, M.thk, thd * arma::datum::pi / 180.0, f_hz);
                double Rd_te = 0.0, Rd_tm = 0.0;
                qd_slab_ref_R(eps_s, M.thk, thd * arma::datum::pi / 180.0, f_hz,
                              std::complex<double>(1.0, 0.0), &Rd_te, &Rd_tm);
                CHECK(std::abs(dB(gainN(0)) - dB(Rd.p_avg)) > 1.0e-2);
                CHECK(std::max(std::abs(rvv2 - Rd_te), std::abs(rhh2 - Rd_tm)) > 1.0e-3);
            }
        }
    }
}

TEST_CASE("Thin sheet - permeability and the lumped columns through ray_mesh_interact", "[wave]")
{
    // Two statements the case above cannot make.
    //
    // The eps <-> mu swap. Swapping holds n = sqrt(eps mu), and so every cosine and the propagation factor, and
    // replaces z by 1/z. With air on both sides that maps every reflection and transmission product of the Airy pair
    // onto the other polarization, so the swap exchanges the TE and TM ports. Here the incidence plane is real, VV is
    // TE and HH is TM: a statement about this function's polarization basis that needs no reference.
    //
    // The lumped columns. att, tf and a permittivity resonance on a sheet reach the composite through this function's
    // own material lookup; each is run against qd_slab_ref_lumped on both ports.
    const double f_hz = 1.0e9;
    const double thk = 0.006;
    arma::mat mesh = qd_quad_x(200.0, 0.0);
    arma::uvec mtl_ind(2, arma::fill::value(1));

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    struct Out
    {
        double vv2, hh2, gain;
        std::complex<double> vv, hh;
    };
    auto run = [&](std::unordered_map<std::string, std::vector<double>> &mtl, double thd, int type)
    {
        arma::mat orig, dest;
        qd_ray_at(thd, orig, dest);
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat origN, destN, xprmatN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(type, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr);
        REQUIRE(gainN.n_elem == 1);
        Out O;
        O.vv = std::complex<double>(xprmatN(0, 0), xprmatN(1, 0));
        O.hh = std::complex<double>(xprmatN(6, 0), xprmatN(7, 0));
        O.vv2 = std::norm(O.vv), O.hh2 = std::norm(O.hh), O.gain = gainN(0);
        return O;
    };

    auto base = [&](double a, double ei)
    {
        std::unordered_map<std::string, std::vector<double>> m;
        m["a"] = {a};
        m["b"] = {0.0};
        m["c"] = {ei / 17.98};
        m["d"] = {0.0};
        m["att"] = {0.0};
        m["attB"] = {0.0};
        m["alpha"] = {0.0};
        m["alphaB"] = {0.0};
        m["fRef"] = {1.0};
        m["thk"] = {thk};
        return m;
    };

    // ---- The swap: eps 5.24 - 0.63j with mu 2.10 - 0.35j, against eps 2.10 - 0.35j with mu 5.24 - 0.63j ----
    QD_DIAG("\n# diag: eps<->mu swap on a sheet through ray_mesh_interact, thk 6 mm\n");
    QD_DIAG("angle,port,d_te_vs_tm,d_tm_vs_te,d_gain,separation\n");

    auto mA = base(5.24, 0.63), mB = base(2.10, 0.35);
    mA["e"] = {2.10}, mA["f"] = {0.0}, mA["g"] = {0.35 / 17.98}, mA["h"] = {0.0};
    mB["e"] = {5.24}, mB["f"] = {0.0}, mB["g"] = {0.63 / 17.98}, mB["h"] = {0.0};

    for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
        for (int type : {1, 0})
        {
            Out A = run(mA, thd, type), B = run(mB, thd, type);
            double d1 = std::abs(B.hh2 - A.vv2), d2 = std::abs(B.vv2 - A.hh2);
            double dg = std::abs(A.gain - B.gain), sep = std::abs(A.vv2 - A.hh2);

            QD_DIAG("%.1f,%s,%.3e,%.3e,%.3e,%.6f\n", thd, type ? "T" : "R", d1, d2, dg, sep);

            INFO("swap, angle " << thd << ", port " << (type ? "T" : "R"));
            CHECK(d1 < 1.0e-9);
            CHECK(d2 < 1.0e-9);
            CHECK(dg < 1.0e-12);

            // Away from normal incidence the two channels are far enough apart that the exchange
            // is a statement and not an identity between two equal numbers.
            if (thd > 0.0)
                CHECK(sep > 1.0e-3);
        }

    // ---- The lumped columns, each on a sheet of eps 5.24 - 0.63j ----
    struct Lump
    {
        const char *name;
        double att, tf, resF, resQ, resS;
    };
    const std::vector<Lump> lumps = {
        {"plain         ", 0.0, 0.0, 0.0, 0.0, 0.0},
        {"att 3 dB      ", 3.0, 0.0, 0.0, 0.0, 0.0},
        {"tf +0.5       ", 0.0, 0.5, 0.0, 0.0, 0.0},
        {"tf -0.5       ", 0.0, -0.5, 0.0, 0.0, 0.0},
        {"resonance     ", 0.0, 0.0, 1.2, 5.0, 0.8},
        {"all three     ", 3.0, 0.5, 1.2, 5.0, 0.8},
    };
    const std::complex<double> eps_s(5.24, -0.63);

    QD_DIAG("\n# diag: lumped columns on a sheet through ray_mesh_interact, eps 5.24 - 0.63j, 6 mm\n");
    QD_DIAG("# moved_dB is the transmit gain against the plain sheet at the same angle\n");
    QD_DIAG("column,angle,T_dB,T_ana_dB,d_T_te,d_T_tm,d_R_te,d_R_tm,moved_dB\n");

    for (const auto &Lm : lumps)
    {
        auto mtl = base(5.24, 0.63);
        mtl["att"] = {Lm.att};
        mtl["tf"] = {Lm.tf};
        mtl["tfB"] = {0.0};
        mtl["resF"] = {Lm.resF};
        mtl["resQ"] = {Lm.resQ};
        mtl["resS"] = {Lm.resS};

        std::complex<double> res(0.0, 0.0);
        if (Lm.resS != 0.0)
            res = (Lm.resS * Lm.resF * Lm.resF) /
                  std::complex<double>(Lm.resF * Lm.resF - 1.0, (Lm.resF / Lm.resQ) * 1.0);

        for (double thd : {0.0, 20.0, 40.0, 60.0, 80.0, 85.0})
        {
            double th = thd * arma::datum::pi / 180.0;
            Out T = run(mtl, thd, 1), R = run(mtl, thd, 0);
            QdLumpRef Q = qd_slab_ref_lumped(eps_s, thk, th, f_hz, Lm.att, Lm.tf, res);
            QdLumpRef Q0 = qd_slab_ref_lumped(eps_s, thk, th, f_hz);
            double gQ = 0.5 * (Q.T_te + Q.T_tm), gQ0 = 0.5 * (Q0.T_te + Q0.T_tm);

            QD_DIAG("%s,%.1f,%.6f,%.6f,%.2e,%.2e,%.2e,%.2e,%+.4f\n", Lm.name, thd, dB(T.gain), dB(gQ),
                    T.vv2 - Q.T_te, T.hh2 - Q.T_tm, R.vv2 - Q.R_te, R.hh2 - Q.R_tm, dB(gQ) - dB(gQ0));

            INFO(Lm.name << ", angle " << thd);
            CHECK(std::abs(dB(T.gain) - dB(gQ)) < 1.0e-6);
            CHECK(std::abs(T.vv2 - Q.T_te) < 1.0e-9);
            CHECK(std::abs(T.hh2 - Q.T_tm) < 1.0e-9);
            CHECK(std::abs(R.vv2 - Q.R_te) < 1.0e-9);
            CHECK(std::abs(R.hh2 - Q.R_tm) < 1.0e-9);

            if (Lm.att == 0.0 && Lm.tf == 0.0 && Lm.resS == 0.0)
            {
                // At its defaults the lumped reference is the plain oracle, on both ports.
                QdSlabRef P = qd_slab_ref(eps_s, thk, th, f_hz);
                double Pr_te = 0.0, Pr_tm = 0.0;
                qd_slab_ref_R(eps_s, thk, th, f_hz, std::complex<double>(1.0, 0.0), &Pr_te, &Pr_tm);
                CHECK(std::abs(Q.t_te - P.t_te) < 1.0e-14);
                CHECK(std::abs(Q.t_tm - P.t_tm) < 1.0e-14);
                CHECK(std::abs(Q.R_te - Pr_te) < 1.0e-14);
                CHECK(std::abs(Q.R_tm - Pr_tm) < 1.0e-14);
            }
            else
            {
                // And every column moves the transmit port by far more than the bound it is
                // asserted to, so none of these rows passes on a column that was never read.
                CHECK(std::abs(dB(gQ) - dB(gQ0)) > 1.0e-2);
            }
        }
    }
}

TEST_CASE("Diagnostic - In-slab cosine projection, two ways", "[wave]")
{
    // The sheet path takes the in-slab cosine as |Re(cos_theta2)|. The traced solid gets it from
    // path_dirN, which qd_refract produced by normalizing a vector built from Re(cos_theta2) -
    // and that normalization is not angle preserving. The two therefore disagree for a lossy
    // slab. The last column converts the disagreement into the traversal-distance error for a
    // 100 mm slab, which is the form in which it reaches the result.
    const double f_hz = 1.0e9;
    const std::complex<double> eps_s(5.24, -0.63);
    arma::mat mesh = qd_quad_x(200.0, 0.0);

    arma::uvec mtl_ind(2, arma::fill::value(1));
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl["a"] = {5.24};
    mtl["b"] = {0.0};
    mtl["c"] = {0.63 / 17.98};
    mtl["d"] = {0.0};
    mtl["att"] = {0.0};
    mtl["attB"] = {0.0};
    mtl["alpha"] = {0.0};
    mtl["alphaB"] = {0.0};
    mtl["fRef"] = {1.0};

    QD_DIAG("\n# diag: in-slab cosine, |Re(ct2)| versus the traced path_dirN projection\n");
    QD_DIAG("angle,ct_sheet,ct_traced,rel_diff,d_dist_mm_per_100mm\n");

    for (double thd : {0.0, 20.0, 40.0, 60.0, 70.0, 80.0, 85.0})
    {
        arma::mat orig, dest;
        qd_ray_at(thd, orig, dest);
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat origN, destN, path_dirN, normal_vecN;
        quadriga_lib::ray_mesh_interact(1, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                        &path_dirN, true, (arma::u32_vec *)nullptr);

        REQUIRE(path_dirN.n_rows == 1);
        double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
        double ct_traced = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);

        double s1 = std::sin(thd * arma::datum::pi / 180.0);
        double ct_sheet = std::abs(std::real(std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1)));
        double rel = (ct_traced > 0.0) ? (ct_sheet - ct_traced) / ct_traced : 0.0;
        double d_dist = 1000.0 * (0.1 / ct_sheet - 0.1 / ct_traced); // mm, for a 100 mm slab

        QD_DIAG("%.1f,%.9f,%.9f,%.3e,%.6f\n", thd, ct_sheet, ct_traced, rel, d_dist);

        CHECK(ct_traced > 0.0);
        CHECK(ct_traced <= 1.0 + 1e-12);

        // The two projections differ by under 1e-3 relative, 0.08 mm of path over a 100 mm slab, below every other term
        // in this suite.
        INFO("angle " << thd << ", ct_sheet " << ct_sheet << ", ct_traced " << ct_traced);
        CHECK(std::abs(rel) < 1.0e-3);
    }
}

TEST_CASE("Thin sheet - grazing incidence does not depend on the face", "[wave]")
{
    // A thin sheet is a laterally infinite layer, so the triangle it was cut from does not enter the composite: on a
    // face with a longest edge of 565.7 m and on one of 0.057 m it is the analytic slab at every angle, up to 89.9999
    // deg, where the lossy sheet is nearly a mirror and both ports stay finite. A floor on the incidence cosine at thk
    // / sqrt(thk^2 + ext^2) would freeze the interface from its onset angle: on the small face a constant -3.315 dB of
    // transmission from 83.95 deg on, 28.8 dB above the slab at 89.9 deg, with the reflected port at 0.473 where the
    // slab reflects 0.997.
    //
    // Both ports: grazing incidence turns a pane into a mirror, so the reflected port is where such a floor costs the
    // most energy.
    const QdSheetFace F;
    const std::complex<double> eps_s(5.24, -0.63);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: thin sheet at grazing incidence, large face versus small face, thk 6 mm\n");
    QD_DIAG("# large face longest edge 565.7 m, small face 0.0566 m\n");
    QD_DIAG("angle,T_large_dB,T_small_dB,T_ana_dB,R_large_dB,R_small_dB,R_ana_dB,sum_large,sum_ana\n");

    for (double thd : {60.0, 80.0, 85.0, 88.0, 89.0, 89.5, 89.9, 89.99, 89.999, 89.9999})
    {
        arma::mat big = qd_quad_x(200.0, 0.0), small = qd_quad_x(0.02, 0.0);
        double Tl = F.run(big, 200.0, thd, 1), Ts = F.run(small, 0.02, thd, 1);
        double Rl = F.run(big, 200.0, thd, 0), Rs = F.run(small, 0.02, thd, 0);

        double th = thd * arma::datum::pi / 180.0;
        QdSlabRef A = qd_slab_ref(eps_s, F.thk, th, F.f_hz);
        double Ra = qd_slab_ref_R(eps_s, F.thk, th, F.f_hz);

        QD_DIAG("%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                thd, dB(Tl), dB(Ts), dB(A.p_avg), dB(Rl), dB(Rs), dB(Ra), Tl + Rl, A.p_avg + Ra);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(Tl));
        REQUIRE(std::isfinite(Ts));
        REQUIRE(std::isfinite(Rl));
        REQUIRE(std::isfinite(Rs));

        // Both faces sit on the analytic slab, in both ports, at every angle.
        CHECK(std::abs(dB(Tl) - dB(A.p_avg)) < 1.0e-6);
        CHECK(std::abs(dB(Ts) - dB(A.p_avg)) < 1.0e-6);
        CHECK(std::abs(dB(Rl) - dB(Ra)) < 1.0e-6);
        CHECK(std::abs(dB(Rs) - dB(Ra)) < 1.0e-6);

        // And they agree with each other to rounding, which is the property the pipeline needs:
        // ray_state_update re-resolves every sheet crossing and never sees the face at all.
        CHECK(std::abs(Tl - Ts) <= 1.0e-9 * std::max(Tl, 1.0e-300));
        CHECK(std::abs(Rl - Rs) <= 1.0e-9 * std::max(Rl, 1.0e-300));
    }
}

TEST_CASE("Resonance - the public index and in-medium gain carry the permittivity pole", "[wave]")
{
    // A Lorentz pole in the permittivity is part of the medium. refractive_index is Re sqrt(eps + pole) and medium_gain
    // charges the pole's loss along the path: a lossy pole is the same medium as a conductivity of that size. A pole
    // strong enough to take Re(eps) below zero gives a finite gain, the rate 8.686 k0 |Im n| dB per meter.
    const double f = 1.0e9, fG = 1.0, k0 = 2.0 * 3.14159265358979323846 * f / 299792458.0, L = 0.1;
    auto pal = [](double a, double c, double rF, double rQ, double rS)
    {
        std::unordered_map<std::string, std::vector<double>> m;
        m["a"] = {a}, m["c"] = {c}, m["fRef"] = {1.0}, m["resF"] = {rF}, m["resQ"] = {rQ}, m["resS"] = {rS};
        return m;
    };
    auto pole = [](double rF, double rQ, double rS, double fq) // the Lorentz pole from the column definition
    { return (rS * rF * rF) / std::complex<double>(rF * rF - fq * fq, (rF / rQ) * fq); };
    struct Row
    {
        const char *name;
        double a, rF, rQ, rS;
    };
    QD_DIAG("\n# diag: the public index and in-medium gain of a resonant medium over 100 mm at 1 GHz. n: refractive_index against\n");
    QD_DIAG("# Re sqrt(eps + pole); g: medium_gain against the conductivity twin where eps + pole has a positive real part, else\n");
    QD_DIAG("# against 10^(-0.8686 k0 |Im n| L)\n");
    QD_DIAG("medium,eps_re,eps_im,n,d_n,g_dB,d_g_dB\n");
    for (const Row &r : {Row{"lossless -2", 4.0, 2.0, 1.0e300, -1.5}, Row{"lossy -2.5j", 4.0, 1.0, 5.0, 0.5}, Row{"strong", 2.5, 0.8, 30.0, 2.0}})
    {
        const std::complex<double> e = r.a + pole(r.rF, r.rQ, r.rS, fG);
        auto M = pal(r.a, 0.0, r.rF, r.rQ, r.rS);
        const double n = quadriga_lib::refractive_index<double>(M, 1, f), g = quadriga_lib::medium_gain<double>(M, 1, L, f);
        const double n_ref = std::real(std::sqrt(e));
        const double g_ref = (std::real(e) > 0.0) ? quadriga_lib::medium_gain<double>(pal(std::real(e), std::max(0.0, -std::imag(e)) / 17.98, 0.0, 0.0, 0.0), 1, L, f)
                                                  : std::pow(10.0, -0.1 * 8.686 * k0 * std::abs(std::imag(std::sqrt(e))) * L);
        QD_DIAG("%s,%.6f,%.6f,%.9f,%.3e,%.6f,%.3e\n", r.name, std::real(e), std::imag(e), n, std::abs(n - n_ref), 10.0 * std::log10(g), std::abs(10.0 * std::log10(g / g_ref)));
        INFO("resonant medium " << r.name);
        CHECK(std::abs(n - n_ref) < 1.0e-12);
        CHECK(std::isfinite(g));
        CHECK(std::abs(10.0 * std::log10(g / g_ref)) < 1.0e-4); // the library takes k0 through the ITU constant 0.0477135
    }
}

TEST_CASE("Coincidence column - a table that carries coiA loads and the column changes nothing", "[wave]")
{
    // coiA is no column. A material table that still carries it loads (the reader is name-keyed and hands every column
    // on) and the column is ignored: a wave medium as a traced solid and as a sheet passes and reflects exactly what it
    // does without the column, bitwise, at the coincidence frequency the column names (3 GHz), EM type 1 (both
    // channels) and scalar types 4 and 5. The mass-law part is in the acoustic cases.
    ub::OneThread one_thread;
    const char *FN = "coiA_table_wave.csv";
    const char *HEADER = "name,a,c,att,fRef,thk,coiF,coiQ,coiA";
    const char *ROWS = "air,1,0,0,1,0,0,0,0\nslab,2.5,0.02,3,1,0,3,5,6\nsheet,2.5,0.02,3,1,0.01,3,5,6\n";

    // a table with a coiA column, read through obj_file_read (table only, name-keyed, schema-blind)
    std::unordered_map<std::string, std::vector<double>> pal;
    {
        std::ofstream csv(FN);
        REQUIRE(csv.is_open());
        csv << HEADER << "\n"
            << ROWS;
        csv.close();
        arma::uvec csv_ind;
        std::vector<std::string> csv_names;
        REQUIRE_NOTHROW(quadriga_lib::obj_file_read<double>("", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                            FN, &csv_ind, &csv_names, &pal));
        std::remove(FN);
        REQUIRE(csv_names.size() == 3);
        REQUIRE(pal.count("coiA") == 1); // the column is read and handed on
    }
    std::unordered_map<std::string, std::vector<double>> bare = pal;
    bare.erase("coiA");

    const double f = 3.0e9;
    CHECK(quadriga_lib::interface_gain(pal, 2, f) == quadriga_lib::interface_gain(bare, 2, f));
    CHECK(quadriga_lib::interface_gain(pal, 3, f) == quadriga_lib::interface_gain(bare, 3, f));
    for (int kind = 0; kind < 2; ++kind) // 0 traced solid 0.2 m, 1 sheet 10 mm
        for (int tr : {1, 4, 5})
            for (double thd : {0.0, 30.0, 60.0})
            {
                arma::uvec mi;
                arma::mat mesh = kind ? ub::build({0.0}, {3}, {false}, mi) : ub::build({0.0, 0.2}, {2, 2}, {false, true}, mi), o, d;
                diag_ray_at(thd, o, d);
                bf::Acc A, B;
                bf::walk(pal, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 60, A);
                bf::walk(bare, mesh, mi, tr, f, thd, 0.0, nullptr, o, d, 1.0, 1.0, 0.0, 0.0, 0, 60, B);
                INFO((kind ? "sheet" : "traced solid") << ", type " << tr << ", " << thd << " deg");
                CHECK(A.nT == B.nT);
                CHECK(A.nR == B.nR);
                CHECK((A.T[0] == B.T[0] && A.T[1] == B.T[1]));
                CHECK((A.R[0] == B.R[0] && A.R[1] == B.R[1]));
                CHECK(std::norm(A.T[0]) > 1.0e-6); // the row is not passing on nothing
            }
}

// Acoustic columns that are all zero change nothing: reflection, transmission, gain, the polarization coefficient and
// the diffracted path are identical bit for bit.
TEST_CASE("Acoustic - convergence with acoustic terms absent", "[wave]")
{
    arma::mat cube_mesh = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    arma::uvec mtl_ind, mtl_ind2;
    std::unordered_map<std::string, std::vector<double>> mtl, mtl2;
    single_material({{"a", 2.5}, {"c", 0.02}, {"fRef", 1.0}}, mtl_ind, mtl);
    single_material({{"a", 2.5}, {"c", 0.02}, {"fRef", 1.0}, {"m", 0.0}, {"resF", 0.0}, {"resQ", 0.0}, {"resS", 0.0}, {"coiF", 0.0}, {"coiQ", 0.0}, {"coiA", 0.0}},
                    mtl_ind2, mtl2);

    double f = 3.0e9;

    // ray_mesh_interact: scalar reflection (3) and transmission (4) at 45 deg on the west face.
    arma::mat orig = {{-1.5, 0.0, 0.0}}, dest = {{0.0, 0.0, 1.5}};
    for (int itype : {3, 4})
    {
        double g1 = rmi_gain(itype, f, orig, dest, cube_mesh, mtl_ind, mtl);
        double g2 = rmi_gain(itype, f, orig, dest, cube_mesh, mtl_ind2, mtl2);
        INFO("interaction_type " << itype);
        CHECK(g1 == Approx(g2).epsilon(1e-12));
    }

    // calc_diffraction_gain (scalar): gain, xprmat and coord must all match.
    arma::mat dorig = {{-10.0, 0.0, 0.5}}, ddest = {{0.5, 0.0, 0.5}};
    arma::vec g1, g2;
    arma::mat x1, x2;
    arma::cube c1, c2;
    quadriga_lib::calc_diffraction_gain<double>(dorig, ddest, cube_mesh, mtl_ind, mtl, f, 3,
                                                &g1, &x1, &c1, 0, nullptr, 0, 0, true, 1.0);
    quadriga_lib::calc_diffraction_gain<double>(dorig, ddest, cube_mesh, mtl_ind2, mtl2, f, 3,
                                                &g2, &x2, &c2, 0, nullptr, 0, 0, true, 1.0);
    CHECK(arma::approx_equal(g1, g2, "absdiff", 1e-12));
    CHECK(arma::approx_equal(x1, x2, "absdiff", 1e-12));
    CHECK(arma::approx_equal(c1, c2, "absdiff", 1e-12));
}

// From a dense medium (air, eps = 1) into a lighter one (eps < 1) the reflection follows the closed-form scalar curve
// across the angle sweep and is total past the critical angle: reflection one, transmission zero.
TEST_CASE("Acoustic - scalar total internal reflection", "[wave]")
{
    arma::mat cube_mesh = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;
    double a_light = 0.25; // critical angle where sin(theta) = sqrt(a) = 0.5 -> 30 deg from normal

    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    single_material({{"a", a_light}, {"fRef", fRef_GHz}}, mtl_ind, mtl); // att = 0 -> interface_gain = 1

    double f = ac2rf(4000.0);
    double deg2rad = arma::datum::pi / 180.0;

    // Build a ray that hits the west face (x = -1) near its center at a chosen incidence angle from
    // the face normal. Direction (cos, sin, 0) gives |dir . normal| = cos(theta).
    auto gains_at = [&](double theta_deg, double &g_refl, double &g_trans)
    {
        double th = theta_deg * deg2rad;
        double dx = std::cos(th), dy = std::sin(th);
        double t_cross = 9.0 / dx; // travel from x = -10 to x = -1
        double y0 = -t_cross * dy; // so the crossing lands at y ~ 0
        arma::mat orig = {{-10.0, y0, 0.0}};
        arma::mat dest = {{-10.0 + 40.0 * dx, y0 + 40.0 * dy, 0.0}};
        g_refl = rmi_gain(3, f, orig, dest, cube_mesh, mtl_ind, mtl);
        g_trans = rmi_gain(4, f, orig, dest, cube_mesh, mtl_ind, mtl);
    };

    for (double th : {0.0, 20.0, 40.0, 60.0})
    {
        double gr, gt;
        gains_at(th, gr, gt);
        double ref_R = ref_scalar_reflection_gain(std::complex<double>(1.0, 0.0),
                                                  std::complex<double>(a_light, 0.0), th);
        INFO("theta = " << th << " deg, ref_R = " << ref_R);
        CHECK(gr == Approx(ref_R).epsilon(1e-4));      // reflected power matches the closed form
        CHECK(gt == Approx(1.0 - ref_R).margin(1e-4)); // energy-conserving transmission
    }

    // Explicit total-internal-reflection band: 60 deg is well past the 30 deg critical angle.
    double gr, gt;
    gains_at(60.0, gr, gt);
    CHECK(gr == Approx(1.0).epsilon(1e-6)); // unit reflection
    CHECK(gt == Approx(0.0).margin(1e-6));  // no transmission

    // Below the critical angle the interface still transmits.
    gains_at(20.0, gr, gt);
    CHECK(gr < 1.0);
    CHECK(gt > 0.0);
}

// A single ray through a thin slab (thin-slab resolution off, eps = 1) equals the single-pass product of entry
// interface, in-medium loss over the thickness and exit interface: calc_diffraction_gain's result tied to the term
// functions.
TEST_CASE("Acoustic - pass-through calibration", "[wave]")
{
    double t = 0.1, L = 20.0, d = 50.0; // thin slab, large faces, distant endpoints (normal ray)
    arma::mat slab = quadriga_lib::cube<double>({t / 2.0, L, L + 0.001});
    double f = ac2rf(4000.0);
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;

    // Staged single-ray transmission: entry (o -> i) * medium(thickness) * exit (i -> o).
    auto rmi_chain = [&](double a, double att, double alpha)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        single_material({{"a", a}, {"att", att}, {"alpha", alpha}, {"fRef", fRef_GHz}}, mtl_ind, mtl);

        arma::mat oA = {{-d, 0.0, 0.0}}, dA = {{d, 0.0, 0.0}};
        double g_entry = rmi_gain(4, f, oA, dA, slab, mtl_ind, mtl);

        arma::mat oB = {{-t / 2.0 + 1e-4, 0.0, 0.0}}, dB = {{d, 0.0, 0.0}};
        double g_exit = rmi_gain(4, f, oB, dB, slab, mtl_ind, mtl);

        double g_med = quadriga_lib::medium_gain<double>(mtl, 1, t, f); // normal incidence -> path = t
        return g_entry * g_med * g_exit;
    };

    // calc_diffraction_gain, single ray (lod = 0), thin-slab resolution disabled (eps = 1) so it
    // matches the naive single-pass chain rather than summing internal multiple reflections.
    auto cdg = [&](double a, double att, double alpha)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        single_material({{"a", a}, {"att", att}, {"alpha", alpha}, {"fRef", fRef_GHz}}, mtl_ind, mtl);
        arma::mat orig = {{-d, 0.0, 0.0}}, dest = {{d, 0.0, 0.0}};
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, slab, mtl_ind, mtl, f, 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, true, 1.0);
        return gain(0);
    };

    struct Case
    {
        const char *name;
        double a, att, alpha;
    };
    std::vector<Case> cases = {
        {"dense absorber", 4.0, 3.0, 20.0},
        {"light reflector", 0.3, 12.0, 8.0},
    };

    for (const auto &C : cases)
    {
        INFO("case: " << C.name);

        double g_chain = rmi_chain(C.a, C.att, C.alpha);
        double g_cdg = cdg(C.a, C.att, C.alpha);
        CHECK(g_cdg > 0.0);
        CHECK(std::abs(10.0 * std::log10(g_cdg / g_chain)) < 0.05); // agree to well under 0.05 dB

        // Excess absorption is applied over the full thickness in both engines: switching alpha off
        // raises the gain by exactly alpha * t [dB].
        double g_chain0 = rmi_chain(C.a, C.att, 0.0);
        double g_cdg0 = cdg(C.a, C.att, 0.0);
        CHECK(-10.0 * std::log10(g_cdg / g_cdg0) == Approx(C.alpha * t).margin(0.05));
        CHECK(-10.0 * std::log10(g_chain / g_chain0) == Approx(C.alpha * t).epsilon(1e-6));

        // Penetration loss is live: enabling it strictly lowers the transmission.
        double g_cdg_noatt = cdg(C.a, 0.0, 0.0);
        CHECK(g_cdg0 < g_cdg_noatt);
    }
}

// A smaller cube inside a larger one of the same permittivity, so that nothing reflects, isolates the penetration loss
// of the embedded material. Crossing it charges that loss exactly once: 12 dB against 6 dB is a factor of 10^0.6 in
// gain.
TEST_CASE("Acoustic - embedded material counted once", "[wave]")
{
    arma::mat A = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat B = quadriga_lib::cube<double>({0.4}, {0.0, 0.0, 0.0}, {0.8, 0.0, 0.001});
    arma::mat mesh = arma::join_vert(A, B);

    arma::uvec mtl_ind(mesh.n_rows);
    mtl_ind.head(A.n_rows).ones();  // outer cube -> material 1
    mtl_ind.tail(B.n_rows).fill(2); // embedded cube -> material 2

    double f = ac2rf(4000.0);
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;

    arma::mat orig = {{-10.0, 0.15, 0.1}}, dest = {{1.5, 0.15, 0.1}};

    auto run_att = [&](double att_B)
    {
        std::unordered_map<std::string, std::vector<double>> mtl;
        mtl["a"] = {4.0, 4.0};     // zero permittivity contrast -> no reflections, att isolated
        mtl["att"] = {0.0, att_B}; // only the embedded cube attenuates
        mtl["fRef"] = {fRef_GHz, fRef_GHz};
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl_ind, mtl, f, 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, true, 1.0);
        return gain(0);
    };

    double g6 = run_att(6.0);
    double g12 = run_att(12.0);
    CHECK(g6 > 0.0);
    CHECK(g6 / g12 == Approx(std::pow(10.0, 0.6)).epsilon(1e-3));
}
