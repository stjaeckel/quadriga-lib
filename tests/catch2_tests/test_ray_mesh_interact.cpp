// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <iostream>
#include <complex>
#include <unordered_map>
#include <vector>
#include <string>

// Function to calculate the gain
static inline double calc_transition_gain(int interaction_type,       // (0) Reflection, (1) Transmission, (2) Refraction
                                          double incidence_angle_deg, // Angle between face normal and ray (as in ITU P.2040-1) (degree)
                                          double dist1,               // Medium 1 travel distance (meters)
                                          double dist2,               // Medium 2 travel distance (meters) OR distance after reflection
                                          std::complex<double> eta1,  // relative permittivity of medium 1
                                          std::complex<double> eta2)  // relative permittivity of medium 2
{
    double deg2rad = arma::datum::pi / 180.0;

    // Calculate gain from ITU-R P.2040:
    double cos_th = std::cos(incidence_angle_deg * deg2rad); // Incidence on boundary
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);        // Trigonometric identity
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * sin_th * sin_th);

    // Medium 1 loss
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * dist1 / Delta;                // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_1_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    // Medium 2 loss
    if (interaction_type != 0) // Use eta1 for reflection
    {
        tan_delta = std::imag(eta2) / std::real(eta2); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
        cos_delta = std::cos(std::atan(tan_delta));
        Delta = 2.0 * cos_delta / (1.0 - cos_delta);
        Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta2)));
    }
    A = 8.686 * dist2 / Delta;                       // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_2_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    eta1 = std::sqrt(eta1);
    eta2 = std::sqrt(eta2);

    // Reflection coefficients
    std::complex<double> R_te = (eta1 * cos_th - eta2 * cos_th2) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> R_tm = (eta2 * cos_th - eta1 * cos_th2) / (eta2 * cos_th + eta1 * cos_th2);

    // Transmission coefficients
    std::complex<double> T_te = (2.0 * eta1 * cos_th) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> T_tm = (2.0 * eta1 * cos_th) / (eta2 * cos_th + eta1 * cos_th2);

    double reflection_gain = 0.5 * (std::norm(R_te) + std::norm(R_tm));
    double refraction_gain = 0.5 * (std::norm(T_te) + std::norm(T_tm));

    double total_gain = 0.0;
    if (interaction_type == 0) // Refection
        total_gain = medium_1_gain * reflection_gain * medium_2_gain;
    else if (interaction_type == 1) // Transmission
        total_gain = medium_1_gain * (1.0 - reflection_gain) * medium_2_gain;
    else if (interaction_type == 2) // Refraction
        total_gain = medium_1_gain * refraction_gain * medium_2_gain;

    return total_gain;
}

// mu-aware reference Fresnel: an independent oracle for the e,f,g,h (permeability) path.
// Mirrors calc_transition_gain but with the index ratio formed from eps*mu, the TE/scalar
// quantity replaced by the admittance sqrt(eps/mu), and the in-medium loss driven by the
// product eps*mu (= n^2). At mu = 1 it reduces exactly to calc_transition_gain.
static inline double calc_transition_gain_mu(int interaction_type,
                                             double incidence_angle_deg,
                                             double dist1, double dist2,
                                             std::complex<double> eta1, std::complex<double> eta2,
                                             std::complex<double> mu1, std::complex<double> mu2)
{
    double deg2rad = arma::datum::pi / 180.0;
    double cos_th = std::cos(incidence_angle_deg * deg2rad);
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);

    std::complex<double> ratio = (eta1 * mu1) / (eta2 * mu2); // (n1/n2)^2
    std::complex<double> cos_th2 = std::sqrt(1.0 - ratio * sin_th * sin_th);

    // In-medium loss from Im(sqrt(eps*mu)) via the ITU attenuation distance (10 GHz test freq)
    auto bulk_gain = [](std::complex<double> em, double dist)
    {
        double tan_delta = std::imag(em) / std::real(em);
        double cos_delta = std::cos(std::atan(tan_delta));
        double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
        Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(em)));
        return std::pow(10.0, -0.1 * (8.686 * dist / Delta));
    };
    std::complex<double> em1 = eta1 * mu1;
    std::complex<double> em2 = (interaction_type != 0) ? eta2 * mu2 : em1;
    double medium_1_gain = bulk_gain(em1, dist1);
    double medium_2_gain = bulk_gain(em2, dist2);

    std::complex<double> Y1 = std::sqrt(eta1 / mu1); // TE/scalar admittance
    std::complex<double> Y2 = std::sqrt(eta2 / mu2);

    std::complex<double> R_te = (Y1 * cos_th - Y2 * cos_th2) / (Y1 * cos_th + Y2 * cos_th2);
    std::complex<double> R_tm = (Y2 * cos_th - Y1 * cos_th2) / (Y2 * cos_th + Y1 * cos_th2);
    std::complex<double> T_te = (2.0 * Y1 * cos_th) / (Y1 * cos_th + Y2 * cos_th2);
    std::complex<double> T_tm = (2.0 * Y1 * cos_th) / (Y2 * cos_th + Y1 * cos_th2);

    double reflection_gain = 0.5 * (std::norm(R_te) + std::norm(R_tm));
    double refraction_gain = 0.5 * (std::norm(T_te) + std::norm(T_tm));

    if (interaction_type == 0)
        return medium_1_gain * reflection_gain * medium_2_gain;
    else if (interaction_type == 1)
        return medium_1_gain * (1.0 - reflection_gain) * medium_2_gain;
    return medium_1_gain * refraction_gain * medium_2_gain; // interaction_type == 2
}

// Shared unit cube (same 12 faces every existing test redefines inline)
static inline arma::mat make_cube()
{
    return arma::mat{{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},
                     {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},
                     {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0},
                     {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
                     {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},
                     {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},
                     {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},
                     {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},
                     {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},
                     {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},
                     {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},
                     {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};
}

// Append constant permeability columns e,f,g,h to an existing material map (assumes a
// uniform mu across the deduplicated materials, which is all the mu tests below need).
template <typename dtype>
static inline void set_mu(std::unordered_map<std::string, std::vector<dtype>> &mtl_prop,
                          dtype e, dtype f, dtype g, dtype h)
{
    size_t nm = mtl_prop.at("a").size();
    mtl_prop["e"] = std::vector<dtype>(nm, e);
    mtl_prop["f"] = std::vector<dtype>(nm, f);
    mtl_prop["g"] = std::vector<dtype>(nm, g);
    mtl_prop["h"] = std::vector<dtype>(nm, h);
}

// Convert a per-face material matrix [n_face, 9] with columns
// {a,b,c,d,att,attB,alpha,alphaB,fRef} into the new (mtl_ind, mtl_prop-map) pair.
// Identical rows are deduplicated, so mtl_ind/mtl_prop match what obj_file_read would emit.
template <typename dtype>
static inline void mtl_matrix_to_map(const arma::Mat<dtype> &M,
                                     arma::uvec &mtl_ind,
                                     std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
{
    static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
    const arma::uword n_face = M.n_rows;

    mtl_ind.set_size(n_face);
    std::vector<arma::uword> uniq; // row index of each distinct material
    for (arma::uword f = 0; f < n_face; ++f)
    {
        arma::uword m = 0;
        bool found = false;
        for (; m < uniq.size(); ++m)
            if (arma::approx_equal(M.row(f), M.row(uniq[m]), "absdiff", (dtype)0))
            {
                found = true;
                break;
            }
        if (!found)
        {
            m = (arma::uword)uniq.size();
            uniq.push_back(f);
        }
        mtl_ind(f) = m + 1;
    }

    mtl_prop.clear();
    for (int c = 0; c < 9; ++c)
    {
        std::vector<dtype> col(uniq.size());
        for (size_t m = 0; m < uniq.size(); ++m)
            col[m] = M.at(uniq[m], c);
        mtl_prop[names[c]] = std::move(col);
    }
}

TEST_CASE("Ray-Mesh Interact - Air to Air (x-z plane)")
{
    // Default cube
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

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // Ray start 10 m west of cube center, pointing east
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};
    arma::vec orig_length = {2.7}; // Assuming 2.7 m  previous length

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
    arma::mat origN, destN, xprmatN, trivecN, tridir_sphN, normal_vecN, tridir_crtN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    // Test case 1 : Cube of air
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // Air
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir_sph, &orig_length,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridir_sphN,
                                    &orig_lengthN, &fbs_angleN, &thicknessN, &edge_lengthN, &normal_vecN);

    CHECK(tridir_sphN.n_cols == 6);

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir_crt, &orig_length,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridir_crtN,
                                    &orig_lengthN, &fbs_angleN, &thicknessN, &edge_lengthN, &normal_vecN);

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

    T = {{-1.001, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = {{-12.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    U = {0.0}; // Air does not reflect anything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{0.001, -0.1, a, 0.001, -0.1, -0.2, 0.001, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{180.0, 1.0, 180.0, 0.0, 179.0, 0.0}};
    CHECK(arma::approx_equal(tridir_sphN, T * deg2rad, "absdiff", 1e-6));

    U = {2.7 + 9.0 + 0.001};
    CHECK(arma::approx_equal(orig_lengthN, U, "absdiff", 1e-6));

    U = {90.0 * deg2rad};
    CHECK(arma::approx_equal(fbs_angleN, U, "absdiff", 1e-6));

    U = {2.0};
    CHECK(arma::approx_equal(thicknessN, U, "absdiff", 1e-6));

    U = {std::sqrt(a * a + (a + 0.1) * (a + 0.1))};
    CHECK(arma::approx_equal(edge_lengthN, U, "absdiff", 1e-6));

    T = {{-1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    CHECK(arma::approx_equal(normal_vecN, T, "absdiff", 1e-6));

    // Test transmission on air
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir_sph, &orig_length,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridir_sphN);

    T = {{-0.999, 0.0, 0.5}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    T = dest;
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    U = {1.0}; // Air does transmit everything
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    a = std::tan(1.0 * deg2rad) * 9.0 + 0.2;
    T = {{-0.001, -0.1, a, -0.001, -0.1, -0.2, -0.001, a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{0.0, 1.0, 0.0, 0.0, 1.0, 0.0}};
    CHECK(arma::approx_equal(tridir_sphN, T * deg2rad, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Air to Dielectric Medium (x-z plane)")
{
    // Default cube
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

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

    // 45° incidence on dielectric medium
    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat T = {{-1.0, 0.0, 0.5}};
    CHECK(arma::approx_equal(fbs, T, "absdiff", 1e-6));
    arma::mat xxx = fbs - T;

    T = {{-0.5, 0.0, 1.0}}; // Ceiling of cube
    CHECK(arma::approx_equal(sbs, T, "absdiff", 1e-6));

    arma::mat trivec = {{0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0}};
    arma::mat tridir = {{0.0, 45.0, 0.0, 45.0, 0.0, 45.0}};
    tridir = tridir * deg2rad;

    // Output containers
    arma::mat origN, destN, xprmatN, trivecN, tridirN, normal_vecN;
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;
    double a;

    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    a = 0.001 * std::cos(45 * deg2rad);
    T = {{-1.0 - a, 0.0, 0.5 + a}};
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
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    T = {{a, -0.1, 0.2 - a, a, -0.1, -0.2 - a, a, 0.2, -a}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{180.0, 45.0, 180.0, 45.0, 180.0, 45.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-6));

    // Test refraction into medium
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // Refraction angle calculated from Snell's law
    double th2 = std::asin(std::sin(45.0 * deg2rad) / std::sqrt(eps));

    double x = std::cos(th2) * 0.001 - 1;
    double z = std::sin(th2) * 0.001 + 0.5;
    T = {{x, 0.0, z}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::cos(th2) * std::sqrt(2) - 1;
    z = std::sin(th2) * std::sqrt(2) + 0.5;
    T = {{x, 0.0, z}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Transmission coefficients, ITU-R P.2040-1, eq. (32a) and (32b)
    double T_te = 2.0 * cos_th / (cos_th + std::sqrt(eps) * std::cos(th2));
    double T_tm = 2.0 * cos_th / (std::sqrt(eps) * cos_th + std::cos(th2));

    U = {0.5 * T_tm * T_tm + 0.5 * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    a = std::cos(th2) * 0.001;
    double b = std::sin(th2) * 0.001;
    T = {{-a, -0.1, 0.2 - b, -a, -0.1, -0.2 - b, -a, 0.2, -b}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{0.0, th2, 0.0, th2, 0.0, th2}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));

    // Test transmission into medium
    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    a = 0.001 * std::cos(45 * deg2rad); // Same as reflection
    T = {{-1.0 + a, 0.0, 0.5 + a}};
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

    T = {{T_tm, 0.0, 0.0, 0.0, 0.0, 0.0, T_te, 0.0}}; // Same as refraction
    T = T * std::sqrt((1 - G) / H);
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    T = {{-a, -0.1, 0.2 - a, -a, -0.1, -0.2 - a, -a, 0.2, -a}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    a = 45.0 * deg2rad;
    T = {{0.0, a, 0.0, a, 0.0, a}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float)")
{
    // Default cube
    arma::fmat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    // Conversions
    float deg2rad = arma::datum::pi / 180.0;
    float rad2deg = 180.0 / arma::datum::pi;

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
    arma::fvec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    arma::fmat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<float>> mtl_map;
    mtl_matrix_to_map<float>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    float cos_th = std::cos(45.0 * deg2rad);
    float a = 0.001 * cos_th;
    T = {{1.0f - a, 0.6f + a, 0.0f}};
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
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    T = {{a, -0.1f - a, 0.2, a, -0.1f - a, -0.2f, a, 0.2f - a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{135.0f, 0.0f, 135.0f, 0.0f, 135.0f, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    float th2 = std::acos(cos_th2);
    float x = std::cos(th2) * 0.001 + 1.0;
    float y = std::sin(th2) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    float T_te = (2.0 * sqrt_eps * cos_th) / (sqrt_eps * cos_th + cos_th2);
    float T_tm = (2.0 * sqrt_eps * cos_th) / (cos_th + sqrt_eps * cos_th2);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    G = calc_transition_gain(2, 45.0, 0.0, 0.0, eta1, eta2);
    U = {(float)G}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, T_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    a = std::cos(th2) * 0.001;
    float b = std::sin(th2) * 0.001;
    T = {{-a, -0.1f - b, 0.2f, -a, -0.1f - b, -0.2f, -a, 0.2f - b, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{th2, 0.0f, th2, 0.0f, th2, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));

    // Transmission from inside to outside without refraction
    quadriga_lib::ray_mesh_interact(1, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    U = {1.0};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    U = {0.5f * xprmatN(6) * xprmatN(6) + 0.5f * xprmatN(0) * xprmatN(0)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // Total reflection occurs at eta = 2, theta = 45°
    mtl_prop = {{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    mtl_matrix_to_map<float>(mtl_prop, mtl_ind, mtl_map);

    quadriga_lib::ray_mesh_interact(0, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {1.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    quadriga_lib::ray_mesh_interact(2, 10.0e9f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::fvec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);
    U = {0.0f};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    T = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Medium to Medium (x-y plane, double)")
{

    // Default cube
    arma::mat msh = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    auto tmp = msh; // Second cube
    tmp.col(0) = tmp.col(0) + 2.0;
    tmp.col(3) = tmp.col(3) + 2.0;
    tmp.col(6) = tmp.col(6) + 2.0;
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    tmp = {{1.33, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0}};
    tmp = repmat(tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, tmp);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

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
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    double cos_th = std::cos(45.0 * deg2rad);
    double a = 0.001 * cos_th;
    T = {{1.0 - a, 0.6 + a, 0.0}};
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
    T = {{R_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, R_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    T = {{a, -0.1 - a, 0.2, a, -0.1 - a, -0.2, a, 0.2 - a, 0.0}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{135.0, 0.0, 135.0, 0.0, 135.0, 0.0}};
    CHECK(arma::approx_equal(tridirN, T * deg2rad, "absdiff", 1e-07));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    double transition_gain = std::pow(10.0, -0.1 * 3.0);

    double th2 = std::acos(cos_th2);
    double x = std::cos(th2) * 0.001 + 1.0;
    double y = std::sin(th2) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::cos(th2) * std::sqrt(2) + 1.0;
    y = std::sin(th2) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-6));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    double T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    double T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    T_te *= std::sqrt(transition_gain);
    T_tm *= std::sqrt(transition_gain);

    U = {0.5f * T_tm * T_tm + 0.5f * T_te * T_te}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{T_te, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, T_tm, 0.0f}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    a = std::cos(th2) * 0.001;
    double b = std::sin(th2) * 0.001;
    T = {{-a, -0.1f - b, 0.2f, -a, -0.1f - b, -0.2f, -a, 0.2f - b, 0.0f}};
    CHECK(arma::approx_equal(trivecN, T, "absdiff", 1e-6));

    T = {{th2, 0.0f, th2, 0.0f, th2, 0.0f}};
    CHECK(arma::approx_equal(tridirN, T, "absdiff", 1e-6));
}

TEST_CASE("Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double)")
{

    // Default cube
    arma::mat msh = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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

    auto tmp = msh; // Second cube
    tmp.col(0) = tmp.col(0) + 2.0;
    tmp.col(3) = tmp.col(3) + 2.0;
    tmp.col(6) = tmp.col(6) + 2.0;
    msh = arma::join_cols(msh, tmp);

    arma::mat mtl_prop = {{1.2, std::log10(1.5 / 1.2), 0.01, std::log10(0.02 / 0.01), 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    tmp = {{1.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    tmp = repmat(tmp, 12, 1);
    mtl_prop = arma::join_cols(mtl_prop, tmp);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

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
    arma::vec gainN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, U;

    // Test reflection
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

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

    // In-medium attenuation
    double thickness = std::sqrt(2.0 * 0.5 * 0.5) + 0.001;
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * thickness / Delta;   // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    R_te *= std::sqrt(gain);
    R_tm *= std::sqrt(gain);
    U = {0.5 * std::abs(R_tm) * std::abs(R_tm) + 0.5 * std::abs(R_te) * std::abs(R_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{std::real(R_te), std::imag(R_te), 0.0, 0.0, 0.0, 0.0, std::real(R_tm), std::imag(R_tm)}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-6));

    // Test refraction
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &msh, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind, &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // Wave direction is not defined precisely due to the complex angles
    std::complex<double> th2 = std::acos(cos_th2);
    double x = std::real(cos_th2) * 0.001 + 1.0;
    double y = std::real(std::sin(th2)) * 0.001 + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(origN, T, "absdiff", 1e-6));

    x = std::real(cos_th2) * std::sqrt(2.0) + 1.0;
    y = std::real(std::sin(th2)) * std::sqrt(2) + 0.6;
    T = {{x, y, 0.0}};
    CHECK(arma::approx_equal(destN, T, "absdiff", 1e-3));

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
    std::complex<double> T_te = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps1 * cos_th + sqrt_eps2 * cos_th2);
    std::complex<double> T_tm = (2.0 * sqrt_eps1 * cos_th) / (sqrt_eps2 * cos_th + sqrt_eps1 * cos_th2);

    T_te *= std::sqrt(gain);
    T_tm *= std::sqrt(gain);
    U = {0.5 * std::abs(T_tm) * std::abs(T_tm) + 0.5 * std::abs(T_te) * std::abs(T_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-3));

    // x-y plane swaps base vectors
    T = {{std::real(T_te), std::imag(T_te), 0.0, 0.0, 0.0, 0.0, std::real(T_tm), std::imag(T_tm)}};
    CHECK(arma::approx_equal(xprmatN, T, "absdiff", 1e-3));
}

TEST_CASE("Ray-Mesh Interact - fRef parameterization equivalence")
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
    // Material A specifies at fRef = 1 GHz; Material B at fRef = 2 GHz
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

    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind_A, &mtl_map_A, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr,
                                    &origNa, &destNa, &gainNa, &xprmatNa, &trivecNa, &tridirNa);
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind_B, &mtl_map_B, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr,
                                    &origNb, &destNb, &gainNb, &xprmatNb, &trivecNb, &tridirNb);

    CHECK(arma::approx_equal(gainNa, gainNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(xprmatNa, xprmatNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(origNa, origNb, "absdiff", 1e-12));
    CHECK(arma::approx_equal(destNa, destNb, "absdiff", 1e-12));
}

TEST_CASE("Ray-Mesh Interact - Penetration loss frequency scaling")
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

    // Normal incidence through the cube
    arma::mat orig = {{-2.0, 0.0, 0.5}};
    arma::mat dest = {{2.0, 0.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    // ε_r = 1, σ = 0 (no reflection, no in-medium loss)
    // att = 6 dB @ 2 GHz, attB = 1  →  at 10 GHz:  Att = 6·(10/2)^1 = 30 dB
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 2.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    arma::mat trivec = {{0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0}};
    arma::mat tridir = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    arma::mat origN, destN, xprmatN, trivecN, tridirN;
    arma::vec gainN;

    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // transition_gain = 10^(-0.1 · 30) = 1e-3
    arma::vec U = {1.0e-3};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-12));
}

TEST_CASE("Ray-Mesh Interact - Alpha in-medium absorption")
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

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr,
                                    &origN, &destN, &gainN, &xprmatN, &trivecN, &tridirN);

    // Expected: lossless-reflection gain × α-loss
    std::complex<double> eta1(1.5, 0.0), eta2(1.0, 0.0);
    double refl_gain = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2);
    double thickness = std::sqrt(0.5 * 0.5 + 0.5 * 0.5) + 0.001; // OF_length + ray_offset
    double alpha_10 = 2.0 * std::pow(10.0 / 5.0, 1.0);           // 4 dB/m
    double alpha_loss = std::pow(10.0, -0.1 * thickness * alpha_10);

    arma::vec U = {refl_gain * alpha_loss};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-9));
}

TEST_CASE("Ray-Mesh Interact - Permeability defaults to 1 (backward compatible)")
{
    arma::mat cube = make_cube();
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
        quadriga_lib::ray_mesh_interact(it, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &ind0, &map0, &fbs_ind, &sbs_ind,
                                        &trivec, &tridir, (arma::vec *)nullptr, &o0, &d0, &g0, &x0, &tv0, &td0);
        quadriga_lib::ray_mesh_interact(it, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &ind1, &map1, &fbs_ind, &sbs_ind,
                                        &trivec, &tridir, (arma::vec *)nullptr, &o1, &d1, &g1, &x1, &tv1, &td1);
        CHECK(arma::approx_equal(g0, g1, "absdiff", 1e-12));
        CHECK(arma::approx_equal(x0, x1, "absdiff", 1e-12)); // xprmat must be bit-stable too
        CHECK(arma::approx_equal(d0, d1, "absdiff", 1e-12));
        CHECK(arma::approx_equal(td0, td1, "absdiff", 1e-12));
    }

    // Anchor mu=1 to the oracle (single-eps reflection)
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &ind1, &map1, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &o1, &d1, &g1, &x1, &tv1, &td1);
    arma::vec U = {calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {1.5, 0.0}, {1.0, 0.0}, {1.0, 0.0})};
    CHECK(arma::approx_equal(g1, U, "absdiff", 1e-9));
}

TEST_CASE("Ray-Mesh Interact - Permeability decouples reflection from refraction")
{
    arma::mat cube = make_cube();
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
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oA, &dA, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(2, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oB, &dB, &gB, &xB, &tvB, &tdB);
    CHECK(arma::approx_equal(tdA, tdB, "absdiff", 1e-9));
    CHECK(arma::approx_equal(dA, dB, "absdiff", 1e-9));

    // Reflection: different admittance sqrt(eps/mu) -> different reflected power
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oA, &dA, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oB, &dB, &gB, &xB, &tvB, &tdB);
    double refA = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}, {3.0, 0.0});
    double refB = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, {1.0, 0.0}, {6.0, 0.0}, {1.0, 0.0}, {1.0, 0.0});
    CHECK(std::abs(refA - refB) > 0.05); // the two genuinely differ
    arma::vec UA = {refA}, UB = {refB};
    CHECK(arma::approx_equal(gA, UA, "absdiff", 1e-9));
    CHECK(arma::approx_equal(gB, UB, "absdiff", 1e-9));
}

TEST_CASE("Ray-Mesh Interact - Permeability impedance-matches a dense medium")
{
    arma::mat cube = make_cube();

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
    mtl_matrix_to_map<double>(mtl, indU, mapU); // unmatched: eps = 4, mu = 1 (pre-mu behavior)

    arma::mat o, d, x, tv, td;
    arma::vec g, U;

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indM, &mapM, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &o, &d, &g, &x, &tv, &td);
    U = {0.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // matched -> no reflection

    quadriga_lib::ray_mesh_interact(1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indM, &mapM, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &o, &d, &g, &x, &tv, &td);
    U = {1.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // matched -> full transmission

    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indU, &mapU, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &o, &d, &g, &x, &tv, &td);
    U = {1.0 / 9.0};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9)); // unmatched -> ((1-2)/(1+2))^2
}

TEST_CASE("Ray-Mesh Interact - eps<->mu swap leaves reflected power invariant")
{
    // Swapping eps and mu preserves n = sqrt(eps*mu) and, at normal incidence, |R|, so the
    // reflected power must be identical. Tripwire: if the Fresnel term reverts to sqrt(eps),
    // A and B diverge.
    arma::mat cube = make_cube();

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
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indA, &mapA, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oA, &dA, &gA, &xA, &tvA, &tdA);
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &indB, &mapB, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &oB, &dB, &gB, &xB, &tvB, &tdB);

    CHECK(arma::approx_equal(gA, gB, "absdiff", 1e-10));
    CHECK(gA(0) > 1e-6); // non-trivial reflection, so the equality is not vacuous
}

TEST_CASE("Ray-Mesh Interact - Permeability drives in-medium loss")
{
    arma::mat cube = make_cube();
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
    quadriga_lib::ray_mesh_interact(0, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &ind, &map, &fbs_ind, &sbs_ind,
                                    &trivec, &tridir, (arma::vec *)nullptr, &o, &d, &g, &x, &tv, &td);

    std::complex<double> eta1(1.5, 0.0), eta2(1.0, 0.0);
    std::complex<double> mu1(1.0, -17.98 * 0.003 / 10.0), mu2(1.0, 0.0);
    double thickness = std::sqrt(0.5 * 0.5 + 0.5 * 0.5) + 0.001; // in-medium path + ray_offset

    double full = calc_transition_gain_mu(0, 45.0, thickness, 0.0, eta1, eta2, mu1, mu2);
    double refl_only = calc_transition_gain_mu(0, 45.0, 0.0, 0.0, eta1, eta2, mu1, mu2);

    arma::vec U = {full};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9));
    CHECK(full < 0.7 * refl_only); // mu attenuates the path; this factor is exactly 1 if the
                                   // loss feed reverts to eps (which is real here)
}

TEST_CASE("Ray-Mesh Interact - path_dirN direction contract (45° dielectric, x-z plane)")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0},
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};

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
            type, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
            (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr,
            &origN, &destN, (arma::vec *)nullptr, (arma::mat *)nullptr,
            (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr,
            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
            (arma::mat *)nullptr, (arma::s32_vec *)nullptr,
            &path_dirN, &ray_indN);
    };

    // Manual unit-norm helpers (avoid arma::norm / arma::normalise, which pull in BLAS)
    auto vnorm = [](const arma::rowvec &v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); };
    auto dir = [&](const arma::mat &m) -> arma::rowvec { arma::rowvec v = m.row(0); return v / vnorm(v); };

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
    CHECK(vnorm(pd_trans - incoming) > 1e-2);                                // and differs from incoming

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

TEST_CASE("Ray-Mesh Interact - ray_indN compaction round-trip")
{
    arma::mat cube = {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},
                      {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0},
                      {-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0},
                      {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0},
                      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0},
                      {-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0},
                      {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0},
                      {1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0},
                      {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0},
                      {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0},
                      {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0},
                      {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}};

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
            1, 10.0e9, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl_map, &fbs_ind, &sbs_ind,
            (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr,
            &origN, &destN, (arma::vec *)nullptr, (arma::mat *)nullptr,
            (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr,
            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
            (arma::mat *)nullptr, (arma::s32_vec *)nullptr,
            &path_dirN, &ray_indN);
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