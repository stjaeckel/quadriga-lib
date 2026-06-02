// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

// Function to calculate the gain
#ifndef calc_transition_gain_HELPER
#define calc_transition_gain_HELPER
static double calc_transition_gain(int interaction_type,       // (0) Reflection, (1) Transmission, (2) Refraction
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
#endif

// Convert a per-face material matrix [n_face, 9] with columns
// {a,b,c,d,att,attB,alpha,alphaB,fRef} into the new (mtl_ind, mtl_prop-map) pair.
// Identical rows are deduplicated, so mtl_ind/mtl_prop match what obj_file_read would emit.
static inline void mtl_matrix_to_map(const arma::mat &M,
                                     arma::uvec &mtl_ind,
                                     std::unordered_map<std::string, std::vector<double>> &mtl_prop)
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
            if (arma::approx_equal(M.row(f), M.row(uniq[m]), "absdiff", 0.0))
            {
                found = true;
                break;
            }
        if (!found)
        {
            m = (arma::uword)uniq.size();
            uniq.push_back(f);
        }
        mtl_ind(f) = m;
    }

    mtl_prop.clear();
    for (int c = 0; c < 9; ++c)
    {
        std::vector<double> col(uniq.size());
        for (size_t m = 0; m < uniq.size(); ++m)
            col[m] = M.at(uniq[m], c);
        mtl_prop[names[c]] = std::move(col);
    }
}

TEST_CASE("Calc Diffraction Gain")
{
    double deg2rad = arma::datum::pi / 180.0;

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

    arma::mat mtl_prop, orig, dest, tm;
    arma::vec gain, tv;
    arma::cube coord, tc;

    mtl_prop = {{1.5, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    // Entire path outside
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{-5.0, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 1.0e9, 1, &gain, &coord, 0);

    tv = {1.0};
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));

    tc.set_size(3, 2, 1);
    tc.slice(0) = {{-8.75, -6.25}, {0.0, 0.0}, {0.5, 0.5}};
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 1.0e9, 2, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 3, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.1464, -7.5, -10.0 + 5.0 * 0.8536}, {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 1.0e9, 3, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 4, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.0955, -10.0 + 5.0 * 0.3455, -10.0 + 5.0 * 0.6545, -10.0 + 5.0 * 0.9045}, {0.0, 0.0, 0.0, 0.0}, {0.5, 0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 1.0e9, 4, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    // Single path outside to inside
    std::complex<double> eta1(1.0, 0.0);                                // Air
    std::complex<double> eta2(mtl_prop(0, 0), -1.798 * mtl_prop(0, 2)); // @ 10 GHz

    double total_gain = calc_transition_gain(1, 0.0, 1.0, 1.5, eta1, eta2);

    tv = {total_gain};
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0, &gain, &coord);

    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 5, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 6, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    // 2 segments, (1) outside to inside, (2) inside
    orig = {{-1.5, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 5, &gain, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));
}

TEST_CASE("Calc Diffraction Gain - Alpha in-medium absorption")
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

    // ε_r = 1 (no Fresnel), σ = 0, α = 4 dB/m, all exponents 0, fRef = 1
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0, &gain);

    // 1.5 m × 4 dB/m = 6 dB  →  gain = 10^-0.6
    arma::vec tv = {std::pow(10.0, -0.1 * 4.0 * 1.5)};
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));
}

TEST_CASE("Calc Diffraction Gain - Penetration loss frequency scaling")
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

    // ε_r = 1 (no Fresnel), σ = 0, α = 0
    // att = 3 dB @ fRef = 2 GHz, attB = 1  →  at 10 GHz:  Att = 3·(10/2)^1 = 15 dB
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0, &gain);

    arma::vec tv = {std::pow(10.0, -1.5)}; // 10^-1.5
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));
}

TEST_CASE("Calc Diffraction Gain - fRef parameterization equivalence")
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

    // At every f:  ε_r = 1.5·f,  σ = 0.001·f,  Att = 2·f dB,  α = 0.5·f dB/m
    arma::mat mat_A = {{1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0}}; // fRef = 1
    arma::mat mat_B = {{3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0}}; // fRef = 2
    mat_A = repmat(mat_A, 12, 1);
    mat_B = repmat(mat_B, 12, 1);
    arma::uvec mtl_ind_A, mtl_ind_B;
    std::unordered_map<std::string, std::vector<double>> mtl_map_A, mtl_map_B;
    mtl_matrix_to_map(mat_A, mtl_ind_A, mtl_map_A);
    mtl_matrix_to_map(mat_B, mtl_ind_B, mtl_map_B);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain_A, gain_B;
    // Use lod=3 to exercise the full ray-state machine with multi-path + multi-hit logic
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind_A, &mtl_map_A, 10.0e9, 3, &gain_A);
    quadriga_lib::calc_diffraction_gain(&orig, &dest, &cube, &mtl_ind_B, &mtl_map_B, 10.0e9, 3, &gain_B);

    CHECK(arma::approx_equal(gain_A, gain_B, "absdiff", 1e-12));
}

TEST_CASE("Calc Diffraction Gain - Scalar mode")
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

    // ε_r = 2 (EM path keeps a Fresnel boundary reflection) + att = 6 dB isolation.
    // Under the partition model 'att' is the only through-wall loss on the scalar path; the
    // EM path additionally loses (1 - |R(θ)|²) at the boundary.
    arma::mat mtl_prop = {{2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0}}; // col 4 = att
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::vec g_em_n, g_sc_n, g_em_o, g_sc_o;

    // Normal incidence: dest INSIDE the cube -> single wall crossing
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0,
                                                &g_em_n, nullptr, 0, nullptr, 0, 0, false);
    quadriga_lib::calc_diffraction_gain<double>(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0,
                                                &g_sc_n, nullptr, 0, nullptr, 0, 0, true);

    // Oblique incidence (~39° off normal at the west wall): dest still INSIDE -> single crossing
    orig = {{-10.0, -8.0, 0.5}};
    dest = {{0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0,
                                                &g_em_o, nullptr, 0, nullptr, 0, 0, false);
    quadriga_lib::calc_diffraction_gain<double>(&orig, &dest, &cube, &mtl_ind, &mtl_map, 10.0e9, 0,
                                                &g_sc_o, nullptr, 0, nullptr, 0, 0, true);

    // Scalar transmission is pure pass-through scaled by the calibrated isolation only:
    // gain = 10^(-att/10) at every angle, no Fresnel boundary loss, no angle dependence.
    double att_lin = std::pow(10.0, -0.6); // -6 dB
    CHECK(std::abs(g_sc_n(0) - att_lin) < 1e-9);
    CHECK(std::abs(g_sc_o(0) - att_lin) < 1e-9);
    CHECK(std::abs(g_sc_n(0) - g_sc_o(0)) < 1e-12); // angle-independent

    // EM carries the same isolation plus the Fresnel boundary loss (1 - |R(θ)|²), so it sits
    // below the scalar value and gets more lossy at oblique incidence.
    CHECK(g_em_n(0) > 0.0);
    CHECK(g_em_n(0) < g_sc_n(0));
    CHECK(g_em_o(0) < g_em_n(0));
}