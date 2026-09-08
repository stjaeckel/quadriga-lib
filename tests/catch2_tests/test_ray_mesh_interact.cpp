// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>

// Diagnostic output for the thin-sheet / slab work. Uncomment to print the tables. The
// assertions in the diagnostic cases run either way and only check sanity, so leaving this
// commented costs nothing but the printing.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

// Exact analytic slab, complex angles throughout, per polarization:
//   T = t12 t23 phi / (1 - r12 r23 phi^2),  phi = exp(-j k0 n d cos(theta_slab))
// Independent of every library internal. This is the oracle both the thin-sheet composite and
// the traced solid are measured against, and it is the only way to tell which of the two moved
// when they disagree.
struct QdSlabRef
{
    double p_te = 0.0, p_tm = 0.0, p_avg = 0.0, ratio = 0.0; // powers and min/max channel ratio
    std::complex<double> t_te, t_tm;                         // complex amplitudes
    std::complex<double> ct_slab;                            // complex in-slab cosine
};

static inline QdSlabRef qd_slab_ref(std::complex<double> eps_s, double thk,
                                    double theta_n_rad, double f_hz)
{
    const double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s), n = std::sqrt(eps_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1);
    const std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * thk * c2);

    QdSlabRef R;
    R.ct_slab = c2;
    for (int pol = 0; pol < 2; ++pol)
    {
        std::complex<double> r, t12, t21;
        if (pol == 0) // TE
            r = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2),
            t12 = (2.0 * z1 * c1) / (z1 * c1 + z2 * c2),
            t21 = (2.0 * z2 * c2) / (z2 * c2 + z1 * c1);
        else // TM
            r = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2),
            t12 = (2.0 * z1 * c1) / (z2 * c1 + z1 * c2),
            t21 = (2.0 * z2 * c2) / (z1 * c2 + z2 * c1);
        std::complex<double> T = t12 * t21 * phi / (1.0 - r * r * phi * phi);
        if (pol == 0)
            R.t_te = T, R.p_te = std::norm(T);
        else
            R.t_tm = T, R.p_tm = std::norm(T);
    }
    R.p_avg = 0.5 * (R.p_te + R.p_tm);
    double lo = std::min(R.p_te, R.p_tm), hi = std::max(R.p_te, R.p_tm);
    R.ratio = (hi > 0.0) ? lo / hi : 0.0;
    return R;
}

// Single-interface Fresnel powers, per polarization. eta1 -> eta2 at theta from the normal.
struct QdIfaceRef
{
    double R_te = 0.0, R_tm = 0.0, T_te = 0.0, T_tm = 0.0;
    double R_avg = 0.0, T_avg = 0.0;
    double obl_te = 0.0, obl_tm = 0.0;          // normal-flux obliquity factor, per polarization
    double F_te = 0.0, F_tm = 0.0, F_avg = 0.0; // per-channel normal-flux transmittance
};

static inline QdIfaceRef qd_iface_ref(std::complex<double> e1, std::complex<double> e2, double theta_n_rad)
{
    const std::complex<double> z1 = std::sqrt(e1), z2 = std::sqrt(e2);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);

    // Past the critical angle the argument is negative real and the principal root is the growing
    // evanescent wave. Force the decaying root, as the library does, so that a medium -> air
    // reference stays comparable above critical. For a complex e2 the argument never lands on the
    // negative real axis and this is inert.
    std::complex<double> ct2_arg = 1.0 - (e1 / e2) * s1 * s1;
    if (std::imag(ct2_arg) == 0.0 && std::real(ct2_arg) < 0.0)
        ct2_arg = std::complex<double>(std::real(ct2_arg), -0.0);
    const std::complex<double> c2 = std::sqrt(ct2_arg);

    QdIfaceRef F;
    F.R_te = std::norm((z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2));
    F.R_tm = std::norm((z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2));
    F.T_te = std::norm((2.0 * z1 * c1) / (z1 * c1 + z2 * c2));
    F.T_tm = std::norm((2.0 * z1 * c1) / (z2 * c1 + z1 * c2));
    F.R_avg = 0.5 * (F.R_te + F.R_tm);
    F.T_avg = 0.5 * (F.T_te + F.T_tm);

    // Normal-flux transmittance. The time-averaged normal Poynting component of a plane wave is
    // proportional to |E|^2 Re(z c) for TE and to |E|^2 Re(conj(z) c) for TM, with z = sqrt(eps/mu),
    // so the obliquity factor in front of |t|^2 is not the same in the two channels once z is
    // complex. F_X equals 1 - R_X identically when the incidence medium is lossless, at any angle
    // and for any complex e2; it does not when the incidence medium is lossy, where it is also not
    // bounded by 1.
    F.obl_te = std::real(z2 * c2) / std::real(z1 * c1);
    F.obl_tm = std::real(std::conj(z2) * c2) / std::real(std::conj(z1) * c1);
    F.F_te = F.obl_te * F.T_te;
    F.F_tm = F.obl_tm * F.T_tm;
    F.F_avg = 0.5 * (F.F_te + F.F_tm);
    return F;
}

// Two coplanar triangles spanning +-half in y and z at x = x0, normal along -x. Built by hand
// rather than through plane() so the face extent - which drives the thin-sheet grazing clamp - is
// set explicitly and independently of the builder's conventions.
//
// The winding matters and is easy to get wrong. ray_mesh_interact forms
// thetaF = acos(OD . N) - pi/2, so a normal pointing ALONG the ray gives thetaF < 0, which is a
// back-side hit: the face material becomes M1 and the interaction is medium -> air, not air ->
// medium. With the rays below travelling towards +x the normal has to point at -x. The shared
// triangulation edge runs along y + z = 0, so a ray in the z = 0 plane must not cross at y = 0.
static inline arma::mat qd_quad_x(double half, double x0)
{
    arma::mat m(2, 9);
    m.row(0) = arma::rowvec{x0, -half, -half, x0, -half, half, x0, half, -half};
    m.row(1) = arma::rowvec{x0, half, -half, x0, -half, half, x0, half, half};
    return m;
}

// The same square as qd_quad_x, meshed as 2 * n^2 triangles instead of 2. Same plane, same normal
// orientation, same material - only the triangulation differs. Every cell is split on the same
// diagonal as the parent quad, so the winding and therefore the normal direction are unchanged.
static inline arma::mat qd_quad_x_sub(double half, double x0, int n)
{
    arma::mat m((arma::uword)(2 * n * n), 9);
    double step = 2.0 * half / (double)n;
    arma::uword row = 0;
    for (int iy = 0; iy < n; ++iy)
        for (int iz = 0; iz < n; ++iz)
        {
            double y0 = -half + step * (double)iy, y1 = y0 + step;
            double z0 = -half + step * (double)iz, z1 = z0 + step;
            m.row(row++) = arma::rowvec{x0, y0, z0, x0, y0, z1, x0, y1, z0};
            m.row(row++) = arma::rowvec{x0, y1, z0, x0, y0, z1, x0, y1, z1};
        }
    return m;
}

// The same square as qd_quad_x with the winding reversed, so the normal points along +x. A ray
// travelling towards +x then produces thetaF < 0, which is a back-side hit: the face material
// becomes M1 and the crossing is medium -> air. That is the same trick ray_mesh_interact uses to
// recover the incidence medium without any state. The triangulation edge is unchanged, so the
// rays from qd_ray_at clear it exactly as they do for the front-side quad.
static inline arma::mat qd_quad_x_rev(double half, double x0)
{
    arma::mat m(2, 9);
    m.row(0) = arma::rowvec{x0, -half, -half, x0, half, -half, x0, -half, half};
    m.row(1) = arma::rowvec{x0, half, -half, x0, half, half, x0, -half, half};
    return m;
}

// TX at (-1, 0, 0), RX placed so the ray crosses the plane x = 0 at the given angle from the
// face normal. The crossing point moves with the angle, which is what keeps it off the shared
// triangulation edge at normal incidence.
static inline void qd_ray_at(double theta_n_deg, arma::mat &orig, arma::mat &dest)
{
    double t = std::tan(theta_n_deg * arma::datum::pi / 180.0);
    orig = arma::mat{{-1.0, 0.37, 0.0}};
    dest = arma::mat{{1.0, 0.37 + 2.0 * t, 0.0}};
}


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

// Append a constant transmission-factor column (tf, tfB) to an existing material map.
template <typename dtype>
static inline void set_tf(std::unordered_map<std::string, std::vector<dtype>> &mtl_prop, dtype tf, dtype tfB = (dtype)0)
{
    size_t nm = mtl_prop.at("a").size();
    mtl_prop["tf"] = std::vector<dtype>(nm, tf);
    mtl_prop["tfB"] = std::vector<dtype>(nm, tfB);
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
    // Blender-style 2x2x2 cube centered at the origin, 12 triangles, outward normals
    arma::mat cube = quadriga_lib::cube<double>();

    // Conversions
    double deg2rad = arma::datum::pi / 180.0;
    double rad2deg = 180.0 / arma::datum::pi;

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

    // Test case 1 : Cube of air
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // Air
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);

    // Test reflection (spherical tridir); also capture fbsN/sbsN (now engine-computed outputs)
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
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6)); // xprmat now [8, n_rayN]

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

TEST_CASE("Ray-Mesh Interact - Air to Dielectric Medium (x-z plane)")
{
    arma::mat cube = quadriga_lib::cube<double>();

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

    // The obliquity factor is exactly the ratio the undeviated type used to apply as a rescale
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

TEST_CASE("Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float)")
{
    arma::fmat cube = quadriga_lib::cube<float>();

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

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b), converted to normal flux
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

TEST_CASE("Ray-Mesh Interact - Medium to Medium (x-y plane, double)")
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

    // Entered medium (M2) carries att = 3 dB as a lumped interface loss (still folded into
    // the transmission-class gain after the refactor; in-medium distance loss is not).
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

    // Tranmission coefficients, ITU-R P.2040-1, eq. (31a) and (31b)
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

TEST_CASE("Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double)")
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

    // Gain and coefficients now exclude in-medium (distance) loss; the complex eta from the
    // conductive incidence medium still shapes the Fresnel terms.
    U = {0.5 * std::norm(R_tm) + 0.5 * std::norm(R_te)}; // Gain without FSPL
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-6));

    // x-y plane swaps base vectors
    T = {{std::real(R_te), std::imag(R_te), 0.0, 0.0, 0.0, 0.0, std::real(R_tm), std::imag(R_tm)}};
    CHECK(arma::approx_equal(xprmatN.t(), T, "absdiff", 1e-6));

    // The in-medium conductivity attenuation the interaction gain used to carry now lives in
    // medium_gain(): verify it reproduces the ITU P.2040-1 conductive loss over the in-slab path.
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

TEST_CASE("Ray-Mesh Interact - fRef parameterization equivalence")
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
    // Material A specifies at fRef = 1 GHz; Material B at fRef = 2 GHz.
    // After the refactor only att and the (complex) Fresnel terms remain in gain/xprmat; both stay
    // identical across the two parameterisations, so the equivalence still holds.
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

TEST_CASE("Ray-Mesh Interact - Penetration loss frequency scaling")
{
    arma::mat cube = quadriga_lib::cube<double>();

    // Normal incidence through the cube
    arma::mat orig = {{-2.0, 0.0, 0.5}};
    arma::mat dest = {{2.0, 0.0, 0.5}};

    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    // ε_r = 1, σ = 0 (no reflection, no in-medium loss)
    // att = 6 dB @ 2 GHz, attB = 1  →  at 10 GHz:  Att = 6·(10/2)^1 = 30 dB
    // Interface (att) loss is still folded into transmission-class gain after the refactor.
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

TEST_CASE("Ray-Mesh Interact - Alpha in-medium absorption (medium_gain)")
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

    // Reflection gain is now the lossless Fresnel term only; the α loss moved to medium_gain().
    std::complex<double> eta1(1.5, 0.0), eta2(1.0, 0.0);
    double refl_gain = calc_transition_gain(0, 45.0, 0.0, 0.0, eta1, eta2);
    arma::vec U = {refl_gain};
    CHECK(arma::approx_equal(gainN, U, "absdiff", 1e-9));

    // α in-medium absorption is now owned by medium_gain(): with real ε and σ = 0 it returns
    // exactly the α term over the in-slab path.
    double thickness = std::sqrt(0.5 * 0.5 + 0.5 * 0.5) + 0.001; // OF_length + ray_offset
    double alpha_10 = 2.0 * std::pow(10.0 / 5.0, 1.0);           // 4 dB/m
    double alpha_loss = std::pow(10.0, -0.1 * thickness * alpha_10);
    CHECK(std::abs(quadriga_lib::medium_gain(mtl_map, 1, thickness, 10.0e9) - alpha_loss) < 1e-9);
}

TEST_CASE("Ray-Mesh Interact - Permeability defaults to 1 (backward compatible)")
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

TEST_CASE("Ray-Mesh Interact - Permeability decouples reflection from refraction")
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

TEST_CASE("Ray-Mesh Interact - Permeability impedance-matches a dense medium")
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
    mtl_matrix_to_map<double>(mtl, indU, mapU); // unmatched: eps = 4, mu = 1 (pre-mu behavior)

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

TEST_CASE("Ray-Mesh Interact - eps<->mu swap leaves reflected power invariant")
{
    // Swapping eps and mu preserves n = sqrt(eps*mu) and, at normal incidence, |R|, so the
    // reflected power must be identical. Tripwire: if the Fresnel term reverts to sqrt(eps),
    // A and B diverge.
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

TEST_CASE("Ray-Mesh Interact - Permeability drives in-medium loss (medium_gain)")
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

    // Reflection gain is now the loss-free admittance term; the mu-driven bulk loss moved to medium_gain().
    arma::vec U = {refl_only};
    CHECK(arma::approx_equal(g, U, "absdiff", 1e-9));

    double bulk = full / refl_only; // loss factor the gain used to carry
    CHECK(std::abs(quadriga_lib::medium_gain(map, 1, thickness, 10.0e9) - bulk) < 1e-9);
    CHECK(bulk < 0.7); // mu attenuates the path; this factor is exactly 1 if the loss feed
                       // reverts to eps (which is real here)
}

TEST_CASE("Ray-Mesh Interact - path_dirN direction contract (45° dielectric, x-z plane)")
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

TEST_CASE("Ray-Mesh Interact - ray_indN compaction round-trip")
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

TEST_CASE("Ray-Mesh Interact - Scalar interactions (types 3/4/5) gains, invariant, interface gain")
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
    double Tt = 1.0 + R;

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

TEST_CASE("Ray-Mesh Interact - Transmission factor splits reflection/transmission energy")
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

TEST_CASE("Ray-Mesh Interact - Ray-tube TIR straddle forces whole-tube pass-through")
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

    // Straddle: one edge at 60 deg (eps=1.5 critical ~54.7 deg) -> whole tube passes through undeviated
    arma::mat tridir_straddle = arma::mat{{60.0, 0.0, 45.0, 0.0, 45.0, 0.0}} * deg2rad;
    run(tridir_straddle);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5)); // undeviated
    CHECK(std::abs(gainN(0)) < 1e-6);                                         // tf=0 -> forced TIR, no power

    // Same straddle with tf = 0.5: still undeviated, now carries 1 - R_eff = 0.5
    mtl_matrix_to_map<double>(mtl_prop, mtl_ind, mtl_map);
    set_tf<double>(mtl_map, 0.5, 0.0);
    run(tridir_straddle);
    CHECK(arma::approx_equal(dir(destN - origN), incoming, "absdiff", 1e-5));
    CHECK(std::abs(gainN(0) - 0.5) < 1e-6);
}

TEST_CASE("Ray-Mesh Interact - path_dirN collapses to undeviated under spine TIR")
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

TEST_CASE("Ray-Mesh Interact - Spherical ray-tube keeps undeviated directions on pass-through")
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

TEST_CASE("Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain")
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

TEST_CASE("Ray-Mesh Interact - out_typeN bit encoding")
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

TEST_CASE("Ray-Mesh Interact - compact = false keeps missed rays in place")
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
// ================================================================================================
// Diagnostics for the thin-sheet / slab work.
//
// These cases exist to produce numbers, not to gate the build. Each prints a table under
// QD_TEST_VERBOSE and asserts only sanity (finite, energy-conserving), so they stay green while
// the divergences they measure are still open.
// ================================================================================================

TEST_CASE("Diagnostic - Interface energy split per polarization")
{
    // Where the averaged reflectance enters. Material::interact_with builds its gain from
    // 0.5*(|R_TE|^2 + |R_TM|^2) and rescales both Fresnel coefficients by one common factor to
    // carry it. This table shows what that does to each channel at a single interface: the
    // per-channel RATIO should survive the common rescale, while the reported gain is the average
    // by construction. If the ratio moves here, the defect is in interact_with itself; if it only
    // moves after a slab resolve, it is the corr / Tsp coupling in qd_slab_resolve.
    //
    // Channel identification. The ray lies in the x-y plane and the normal is along x, so
    // eQ = U x N is parallel to z, and so is eV. qd_polbasis therefore performs a clean swap rather
    // than a mixing rotation: VV is the TE channel and HH is the TM channel, up to sign. That is the
    // reverse of the degenerate collinear case, and getting it backwards inverts every comparison
    // against a per-polarization reference.
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

            // interact_with hands over exact per-channel coefficients. The averaged reflectance sets
            // the total and rescales both channels by one common factor, so it cannot move a ratio -
            // measured to 8e-16. Anything that reintroduces a per-channel rescale here fails.
            INFO("type " << type << ", angle " << thd);
            CHECK(std::abs(gainN(0) - ref_avg) < 1e-12);
            CHECK(std::abs(vv2 - ref_te) < 1e-12);
            CHECK(std::abs(hh2 - ref_tm) < 1e-12);
            CHECK(std::abs(rc - rr) < 1e-12);
        }
}

TEST_CASE("Diagnostic - Transmission convention baseline, single interface")
{
    // Absolute per-type level at one interface, which nothing else in either suite prints. Every
    // other table here measures a ratio or a composite: a ratio is nearly blind to the obliquity
    // factor, and the sheet composite never takes its coefficients from Material::interact_with at
    // all. Without this table a shifted level after the convention change could only be compared
    // against the previous run, not against physics.
    //
    // Two references sit beside the code's output:
    //   raw_X  = |t_X|^2, the Fresnel amplitude power with no obliquity factor
    //   flux_X = Re(z2 c2)/Re(z1 c1) |t_TE|^2  and  Re(conj(z2) c2)/Re(conj(z1) c1) |t_TM|^2,
    //            the per-channel normal-flux transmittance
    //
    // d_raw_dB and d_flux_dB report which of the two each interaction type is currently on. Types
    // 2 and 5 sit on raw and types 1 and 4 on the TE/TM-averaged flux value, so the same physical
    // crossing reports different energy depending only on whether the tracer was asked to bend the
    // ray. That gap, not the channel ratio, is the size of the convention defect.
    //
    // The m2a rows are the lossy-incidence case, where flux_X and 1 - R_X separate and flux_X is
    // no longer bounded by 1. The a2m rows are where the two coincide identically.
    //
    // Channel identification: the ray lies in the x-y plane and the normal is along x, so
    // qd_polbasis performs a clean swap and VV is TE while HH is TM, in both hit directions - the
    // back-side hit flips the normal before the basis is built. Scalar types carry one coefficient;
    // VV2 is repeated into HH2 and both references use the TE form, which is what the scalar branch
    // computes, since z is the acoustic admittance and T = 1 + R equals t_TE algebraically.
    //
    // The acceptance test is |d_flux_dB| < 1e-9 for all four types in both directions, with the
    // total-reflection rows excluded. d_raw_dB is kept as a printed column: it is the size of the
    // obliquity factor and records what the raw-amplitude convention used to report.
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

                    // Acceptance test for the energy convention: every transmission-class type
                    // reports the per-channel normal-flux transmittance of its own interface, and
                    // the four types therefore agree with each other. Total reflection is excluded,
                    // because there is no forward Fresnel port there and the coefficient carries a
                    // flat tf leak instead.
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

TEST_CASE("Diagnostic - Thin sheet composite vs analytic slab")
{
    // The sheet composite is produced in one atomic event and should equal the exact analytic
    // slab. Any residual here is the floor on how close the sheet can get to physics, and the
    // traced solid cannot do better than that floor either. The polarization-ratio columns are
    // the sharper of the two: a power error can hide behind the TE/TM average, a ratio error
    // cannot.
    const double f_hz = 1.0e9;

    struct Mat
    {
        const char *name;
        double a, c, thk;
    };
    const std::vector<Mat> mats = {
        {"lossless eps 5.24, 100 mm", 5.24, 0.0, 0.100},
        {"lossy eps 5.24, 100 mm", 5.24, 0.63 / 17.98, 0.100},
        {"low eps 2.0, weak loss, 100 mm", 2.0, 0.25 / 17.98, 0.100},
        {"high eps 25.0, strong loss, 100 mm", 25.0, 2.50 / 17.98, 0.100},
        {"lossy eps 5.24, 6 mm", 5.24, 0.63 / 17.98, 0.006},
    };

    QD_DIAG("\n# diag: thin sheet composite vs analytic slab, 1 GHz\n");
    QD_DIAG("material,angle,sheet_dB,ana_dB,d_dB,ratio_sheet,ratio_ana,d_ratio,arg_sheet_deg,arg_ana_te_deg,d_arg_deg,pred_k0_thk_cos\n");

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
            double vv2 = xprmatN(0, 0) * xprmatN(0, 0) + xprmatN(1, 0) * xprmatN(1, 0);
            double hh2 = xprmatN(6, 0) * xprmatN(6, 0) + xprmatN(7, 0) * xprmatN(7, 0);
            double arg_vv = std::atan2(xprmatN(1, 0), xprmatN(0, 0)) * 180.0 / arma::datum::pi;

            QdSlabRef R = qd_slab_ref(eps_s, M.thk, thd * arma::datum::pi / 180.0, f_hz);
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

            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.9f,%.9f,%.3e,%.2f,%.2f,%.2f,%.2f\n",
                    M.name, thd, dB(gainN(0)), dB(R.p_avg), dB(gainN(0)) - dB(R.p_avg),
                    rc, R.ratio, rc - R.ratio, arg_vv, arg_ref, d_arg, pred);

            CHECK(std::isfinite(gainN(0)));
            CHECK(gainN(0) >= 0.0);
            CHECK(gainN(0) <= 1.0);

            // The sheet composite matches the exact analytic slab to the amplitude-convention floor,
            // and its phase sits exactly one free-space reference term away from it. Both hold across
            // five materials and eight angles, so both are regression guards rather than targets.
            // With the raw-amplitude interface convention and the exact complex propagation factor
            // both in, nothing separates the sheet composite from the exact analytic slab: it is the
            // same closed form evaluated the same way. The bounds are numerical, not physical, and a
            // failure here means an approximation has been reintroduced somewhere in produce mode.
            INFO(M.name << ", angle " << thd);
            CHECK(std::abs(dB(gainN(0)) - dB(R.p_avg)) < 1.0e-6);
            CHECK(std::abs(rc - R.ratio) < 1.0e-9);
            CHECK(std::abs(d_arg - pred) < 1.0e-3);
        }
    }
}

TEST_CASE("Diagnostic - In-slab cosine projection, two ways")
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

        // The two projections differ by under 1e-3 relative, which is 0.08 mm of path over a 100 mm
        // slab and below every other term in this suite. Guarded rather than fixed.
        INFO("angle " << thd << ", ct_sheet " << ct_sheet << ", ct_traced " << ct_traced);
        CHECK(std::abs(rel) < 1.0e-3);
    }
}

// Shared fixture for the grazing cases below: one 6 mm lossy sheet, one ray per angle, either
// port. The reach shrinks with the angle so the crossing stays on the face at every angle instead
// of walking off a small one and reporting -inf for a geometric reason.
struct QdSheetFace
{
    double f_hz = 1.0e9;
    double thk = 0.006;

    std::unordered_map<std::string, std::vector<double>> mtl() const
    {
        std::unordered_map<std::string, std::vector<double>> m;
        m["a"] = {5.24};
        m["b"] = {0.0};
        m["c"] = {0.63 / 17.98};
        m["d"] = {0.0};
        m["att"] = {0.0};
        m["attB"] = {0.0};
        m["alpha"] = {0.0};
        m["alphaB"] = {0.0};
        m["fRef"] = {1.0};
        m["thk"] = {thk};
        return m;
    }

    double run(arma::mat &mesh, double half, double thd, int type) const
    {
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value(1));
        auto mp = mtl();

        double t = std::tan(thd * arma::datum::pi / 180.0);
        double y0 = 0.3 * half;
        double reach = (t > 1.0e-6) ? 0.4 * half / t : half;
        arma::mat orig = arma::mat{{-reach, y0, 0.0}};
        arma::mat dest = arma::mat{{reach, y0 + 2.0 * reach * t, 0.0}};

        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        if (fbs_ind.n_elem == 0 || fbs_ind(0) == 0)
            return std::numeric_limits<double>::quiet_NaN();

        arma::mat origN, destN, xprmatN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(type, f_hz, &orig, &dest, &mesh, &mtl_ind, &mp,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr);
        return gainN(0);
    }
};

// Composite slab reflectance, TE/TM averaged, air | slab | air. The stack is symmetric, so the
// internal reflection is the same at both faces and R = r (1 - phi^2) / (1 - r^2 phi^2).
static inline double qd_slab_ref_R(std::complex<double> eps_s, double thk,
                                   double theta_n_rad, double f_hz)
{
    const double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s), n = std::sqrt(eps_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1);
    const std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * thk * c2);
    const std::complex<double> e = phi * phi;
    const std::complex<double> r_te = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2);
    const std::complex<double> r_tm = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2);
    const std::complex<double> R_te = r_te * (1.0 - e) / (1.0 - r_te * r_te * e);
    const std::complex<double> R_tm = r_tm * (1.0 - e) / (1.0 - r_tm * r_tm * e);
    return 0.5 * (std::norm(R_te) + std::norm(R_tm));
}

TEST_CASE("Thin sheet - grazing incidence does not depend on the face")
{
    // A thin sheet is a laterally infinite layer, so the triangle it was cut from does not enter the
    // composite and the composite is the analytic slab at every angle up to tangency. This replaces
    // the clamp-threshold case: the grazing clamp used to floor the incidence cosine at
    // thk / sqrt(thk^2 + ext^2), which froze the whole interface at the onset angle. On the 57 mm
    // face below that meant a constant -3.315 dB of transmission from 83.95 deg onward, 28.8 dB
    // above the slab at 89.9 deg, with the reflected port held at 0.473 where the slab reflects
    // 0.997.
    //
    // Both ports are checked. Grazing incidence turns a pane into a mirror, so the reflected port is
    // where the freeze did the most damage in absolute energy, and it is held to the same bound as
    // the transmitted one.
    const QdSheetFace F;
    const std::complex<double> eps_s(5.24, -0.63);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: thin sheet at grazing incidence, large face versus small face, thk 6 mm\n");
    QD_DIAG("# large face longest edge 565.7 m, small face 0.0566 m; the old clamp engaged past\n");
    QD_DIAG("# 89.9994 deg and 83.95 deg respectively\n");
    QD_DIAG("angle,T_large_dB,T_small_dB,T_ana_dB,R_large_dB,R_small_dB,R_ana_dB,sum_large,sum_ana\n");

    for (double thd : {60.0, 80.0, 85.0, 88.0, 89.0, 89.5, 89.9, 89.99, 89.999})
    {
        arma::mat big = qd_quad_x(200.0, 0.0), small = qd_quad_x(0.02, 0.0);
        double Tl = F.run(big, 200.0, thd, 1), Ts = F.run(small, 0.02, thd, 1);
        double Rl = F.run(big, 200.0, thd, 0), Rs = F.run(small, 0.02, thd, 0);

        double th = thd * arma::datum::pi / 180.0;
        QdSlabRef A = qd_slab_ref(eps_s, F.thk, th, F.f_hz);
        double Ra = qd_slab_ref_R(eps_s, F.thk, th, F.f_hz);

        QD_DIAG("%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
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

TEST_CASE("Thin sheet - mesh subdivision invariance")
{
    // The same plane, the same rays, the same material, meshed two ways: as two triangles and as
    // 2 * 65^2. No coefficient may notice. The grazing clamp did notice, because it read the longest
    // edge of the hit triangle: the coarse mesh clamped past 89.9994 deg and the fine mesh past
    // 89.96 deg, so at 89.99 deg the same wall answered 20 dB apart depending on mesh density.
    //
    // 65 cells per side is odd, which puts z = 0 inside a row rather than on a shared edge, and the
    // crossing at y = 140 is not a cell boundary either. The physics does not care - every
    // sub-triangle is coplanar, identically oriented and carries the same material - but a hit
    // landing exactly on a shared edge would make the comparison depend on which of two triangles
    // the intersector happened to return.
    const QdSheetFace F;
    const std::complex<double> eps_s(5.24, -0.63);
    const double half = 200.0;
    const int n_sub = 65;

    arma::mat coarse = qd_quad_x(half, 0.0);
    arma::mat fine = qd_quad_x_sub(half, 0.0, n_sub);
    REQUIRE(fine.n_rows == (arma::uword)(2 * n_sub * n_sub));

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: subdivision invariance, 400 m face as 2 triangles and as 2 * 65^2\n");
    QD_DIAG("# coarse longest edge 565.7 m, fine 8.70 m\n");
    QD_DIAG("angle,T_coarse_dB,T_fine_dB,T_ana_dB,R_coarse_dB,R_fine_dB,R_ana_dB,rel_T,rel_R\n");

    for (double thd : {60.0, 80.0, 88.0, 89.9, 89.99, 89.999})
    {
        double Tc = F.run(coarse, half, thd, 1), Tf = F.run(fine, half, thd, 1);
        double Rc = F.run(coarse, half, thd, 0), Rf = F.run(fine, half, thd, 0);

        double th = thd * arma::datum::pi / 180.0;
        QdSlabRef A = qd_slab_ref(eps_s, F.thk, th, F.f_hz);
        double Ra = qd_slab_ref_R(eps_s, F.thk, th, F.f_hz);

        double rel_T = std::abs(Tc - Tf) / std::max(Tc, 1.0e-300);
        double rel_R = std::abs(Rc - Rf) / std::max(Rc, 1.0e-300);

        QD_DIAG("%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e\n",
                thd, dB(Tc), dB(Tf), dB(A.p_avg), dB(Rc), dB(Rf), dB(Ra), rel_T, rel_R);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(Tc));
        REQUIRE(std::isfinite(Tf));
        REQUIRE(std::isfinite(Rc));
        REQUIRE(std::isfinite(Rf));

        CHECK(rel_T < 1.0e-9);
        CHECK(rel_R < 1.0e-9);
        CHECK(std::abs(dB(Tf) - dB(A.p_avg)) < 1.0e-6);
        CHECK(std::abs(dB(Rf) - dB(Ra)) < 1.0e-6);
    }
}

TEST_CASE("Diagnostic - Derived lengths and the tangency limit")
{
    // What the clamp was said to be guarding against, printed unclamped.
    //
    // The transfer function is written in the normal thickness and the complex in-slab cosine. The
    // normal thickness is thk by construction and the in-slab cosine is bounded from below by the
    // internal critical angle - 0.9011 for eps 5.24 - so neither moves at grazing incidence. The
    // geometric path thk / cos does grow without bound, but it reaches a result only as the mass-law
    // term dist_geo * cos^2 = thk * cos, which shrinks instead. qd_slab_resolve now forms that term
    // as a product and never builds the quotient.
    //
    // The one genuine path length left is thk / cos_slab, read by the alpha column and by the
    // mass-law conductivity term. Both are dB-linear in it, so it saturates and is capped by loss
    // rather than by angle.
    const QdSheetFace F;
    const std::complex<double> eps_s(5.24, -0.63);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: derived lengths, sheet thk 6 mm, eps 5.24 - 0.63j\n");
    QD_DIAG("angle,ct_geo,ct_slab,thk_over_ct_slab_m,thk_normal_m,thk_over_ct_geo_m,mass_path_m\n");
    for (double thd : {80.0, 89.0, 89.9, 89.99, 89.999, 89.9999})
    {
        double th = thd * arma::datum::pi / 180.0;
        double ct_geo = std::cos(th), s1 = std::sin(th);
        double ct_slab = std::abs(std::real(std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1)));
        double d_slab = F.thk / ct_slab, d_geo = F.thk / ct_geo;

        QD_DIAG("%.4f,%.3e,%.9f,%.9f,%.9f,%.6f,%.3e\n",
                thd, ct_geo, ct_slab, d_slab, d_slab * ct_slab, d_geo, d_geo * ct_geo * ct_geo);

        CHECK(std::abs(d_slab * ct_slab - F.thk) < 1e-15);       // clamp-invariant by construction
        CHECK(d_geo * ct_geo * ct_geo <= F.thk + 1e-15);         // shrinks rather than diverging
        CHECK(ct_slab > 0.9);                                    // bounded by the internal critical angle
    }

    // The tangency limit itself. A lossy slab approaches a perfect mirror, so the transmitted port
    // goes to zero and the reflected one to unity, and both stay finite all the way in. This is the
    // behaviour the clamp was introduced to protect and which it in fact prevented.
    arma::mat mesh = qd_quad_x(200.0, 0.0);
    QD_DIAG("\n# diag: tangency limit, unclamped\n");
    QD_DIAG("angle,T_dB,T_ana_dB,R_lin,R_ana_lin\n");
    for (double thd : {89.9, 89.99, 89.999, 89.9999})
    {
        double T = F.run(mesh, 200.0, thd, 1);
        double R = F.run(mesh, 200.0, thd, 0);
        double th = thd * arma::datum::pi / 180.0;
        QdSlabRef A = qd_slab_ref(eps_s, F.thk, th, F.f_hz);
        double Ra = qd_slab_ref_R(eps_s, F.thk, th, F.f_hz);

        QD_DIAG("%.4f,%.6f,%.6f,%.9f,%.9f\n", thd, dB(T), dB(A.p_avg), R, Ra);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(T));
        REQUIRE(std::isfinite(R));
        CHECK(T >= 0.0);
        CHECK(R <= 1.0);
        CHECK(std::abs(dB(T) - dB(A.p_avg)) < 1.0e-6);
        CHECK(std::abs(R - Ra) < 1.0e-9);
    }
}

// Closed-form oracle for the incoherent mass-law composite, air | sheet | air, real lossless
// permittivity, att = alpha = tf = 0, no resonance. Shares no code with the library.
//
// A sheet whose material carries m > 0 takes the incoherent route in qd_slab_resolve produce
// mode: no internal series, no in-slab phase, the near interface evaluated at the incidence
// cosine and the far one at the in-slab cosine, and one lumped mass-law power factor. With tf = 0
// the amplitude rescale is the identity, and with air on both sides the near-to-far flux
// conversion and the lumped interface gain are both unity, so the transmit port is the product of
// the two raw Fresnel transmission amplitudes times the square root of the mass gain.
struct QdMassRef
{
    double base = 0.0; // composite transmitted power with the mass term removed
    double refl = 0.0; // reflect port: the bare near interface, Stokes-reversed
    double cts = 0.0;  // in-slab cosine
};

static inline QdMassRef qd_mass_ref(double eps_s, double thd)
{
    const double th = thd * arma::datum::pi / 180.0;
    const double c1 = std::cos(th), s1 = std::sin(th);
    const double zs = std::sqrt(eps_s);
    const double cts = std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1);

    const double tn_te = 2.0 * c1 / (c1 + zs * cts);
    const double tn_tm = 2.0 * c1 / (zs * c1 + cts);
    const double tf_te = 2.0 * zs * cts / (zs * cts + c1);
    const double tf_tm = 2.0 * zs * cts / (cts + zs * c1);

    const double r_te = (c1 - zs * cts) / (c1 + zs * cts);
    const double r_tm = (zs * c1 - cts) / (zs * c1 + cts);

    QdMassRef R;
    R.cts = cts;
    R.base = 0.5 * (tn_te * tf_te * tn_te * tf_te + tn_tm * tf_tm * tn_tm * tf_tm);
    R.refl = 0.5 * (r_te * r_te + r_tm * r_tm);
    return R;
}

// The mass-law transmission loss in dB. arg is (f/fRef) times the surface-mass surrogate with the
// angle removed, which for a sheet is the normal thickness.
//
// The normal-incidence term A is the material table's own value and the angle carries the mass
// law's shape normalized to it, so the two endpoints are exact: A at normal incidence and zero at
// tangency. The asymptotic angle slope is 20 dB per decade of the cosine, independent of m.
static inline double qd_mass_TL(double m, double arg, double ct)
{
    if (m <= 0.0 || arg <= 0.0 || ct <= 0.0)
        return 0.0;
    double A = m * std::log10(arg);
    if (A <= 0.0)
        return 0.0;
    if (ct >= 1.0)
        return A;
    double X0 = std::pow(arg, 0.05 * m);
    double x = X0 * ct;
    return A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0);
}

// The superseded form, with the angle inside the same logarithm as the frequency and the result
// clipped at zero. Its angle slope is m rather than 20 and it cuts the term off at arg * ct = 1.
// Kept because the two forms must agree exactly at normal incidence, which is asserted below, and
// because the diagnostic table records what the change is worth at each angle.
static inline double qd_mass_TL_clipped(double m, double arg, double ct)
{
    if (m <= 0.0 || arg <= 0.0 || ct <= 0.0)
        return 0.0;
    double v = m * std::log10(arg * ct);
    return (v > 0.0) ? v : 0.0;
}

// One 12.5 mm mass-law sheet, one ray per angle, either port. fRef carries the frequency scale so
// that the mass term lands in a realistic range; every other exponent column is zero, so fRef
// reaches nothing else. The permittivity is real and lossless and alpha is zero, which leaves the
// mass term as the only contribution to medium_gain and lets it be read off the composite.
struct QdMassSheet
{
    double f_hz = 1.0e9;
    double fRef = 1.0e-4; // GHz
    double thk = 0.0125;  // m
    double eps = 5.24;
    double m = 12.0; // dB/decade, deliberately not 20 so the angle slope is distinguishable

    double arg() const { return (f_hz * 1.0e-9 / fRef) * thk; } // 125.0

    std::unordered_map<std::string, std::vector<double>> mtl(double m_val) const
    {
        std::unordered_map<std::string, std::vector<double>> mp;
        mp["a"] = {eps};
        mp["b"] = {0.0};
        mp["c"] = {0.0};
        mp["d"] = {0.0};
        mp["att"] = {0.0};
        mp["attB"] = {0.0};
        mp["alpha"] = {0.0};
        mp["alphaB"] = {0.0};
        mp["fRef"] = {fRef};
        mp["thk"] = {thk};
        mp["m"] = {m_val};
        return mp;
    }

    double run(arma::mat &mesh, double half, double thd, int type, double m_val) const
    {
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value(1));
        auto mp = mtl(m_val);

        double t = std::tan(thd * arma::datum::pi / 180.0);
        double y0 = 0.3 * half;
        double reach = (t > 1.0e-6) ? 0.4 * half / t : half;
        arma::mat orig = arma::mat{{-reach, y0, 0.0}};
        arma::mat dest = arma::mat{{reach, y0 + 2.0 * reach * t, 0.0}};

        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        if (fbs_ind.n_elem == 0 || fbs_ind(0) == 0)
            return std::numeric_limits<double>::quiet_NaN();

        arma::mat origN, destN, xprmatN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(type, f_hz, &orig, &dest, &mesh, &mtl_ind, &mp,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr);
        return gainN(0);
    }
};

TEST_CASE("Thin sheet - mass-law angle dependence")
{
    // The mass term is the only angle-dependent quantity in the sheet composite that is not a
    // Fresnel amplitude. The case divides the measured composite by the hand-built Fresnel product,
    // which leaves the mass gain alone, and compares the result with the closed form.
    //
    // A sheet whose material carries m > 0 takes the incoherent route in qd_slab_resolve produce
    // mode: no internal series, no in-slab phase, the near interface at the incidence cosine and
    // the far one at the in-slab cosine, and one lumped mass-law power factor. With tf = 0 the
    // amplitude rescale is the identity, and with air on both sides the near-to-far flux conversion
    // and the lumped interface gain are both unity, so the transmit port is the product of the two
    // raw Fresnel transmission amplitudes times the square root of the mass gain.
    const QdMassSheet F;
    const double half = 200.0;
    arma::mat mesh = qd_quad_x(half, 0.0);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    const double arg = F.arg();
    const double A0 = F.m * std::log10(arg);           // normal-incidence term
    const double X0 = std::pow(arg, 0.05 * F.m);       // = 10^(A0/20)
    const double ct_knee = 1.0 / X0;                   // x = 1
    const double ct_clip = 1.0 / arg;                  // superseded form: term reaches zero here
    const double th_knee = std::acos(ct_knee) * 180.0 / arma::datum::pi;
    const double th_clip = std::acos(ct_clip) * 180.0 / arma::datum::pi;

    QD_DIAG("\n# diag: mass-law angle dependence, sheet thk 12.5 mm, eps 5.24, m 12 dB/decade\n");
    QD_DIAG("# (f/fRef) * thk = %.4f; knee at %.6f deg; the superseded form cut the term off at %.6f deg\n",
            arg, th_knee, th_clip);
    QD_DIAG("angle,cos,base_dB,T_dB,TL_meas_dB,TL_dB,TL_superseded_dB,d_dB,R_meas,R_ref\n");

    double tl_prev = 1.0e300;
    double tl_at[2] = {0.0, 0.0}; // 30 deg and 60 deg, for the slope check

    const double sweep[] = {0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 78.0, 80.0,
                            85.0, 88.0, 89.0, 89.5, 89.6, 89.75, 89.9};

    for (double thd : sweep)
    {
        double ct = std::cos(thd * arma::datum::pi / 180.0);
        QdMassRef Rf = qd_mass_ref(F.eps, thd);

        double T = F.run(mesh, half, thd, 1, F.m);
        double R = F.run(mesh, half, thd, 0, F.m);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(T));
        REQUIRE(std::isfinite(R));
        CHECK(T >= 0.0);
        CHECK(R >= 0.0);
        CHECK(R <= 1.0);

        double tl_meas = dB(Rf.base) - dB(T); // the mass term, read off the composite
        double tl_want = qd_mass_TL(F.m, arg, ct);
        double tl_old = qd_mass_TL_clipped(F.m, arg, ct);

        QD_DIAG("%.4f,%.6e,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.9f,%.9f\n",
                thd, ct, dB(Rf.base), dB(T), tl_meas, tl_want, tl_old, tl_meas - tl_want, R, Rf.refl);

        // The composite as a whole, which pins the incoherent route and the mass term together.
        CHECK(std::abs(dB(T) - (dB(Rf.base) - tl_want)) < 1.0e-6);

        // The mass term on its own.
        CHECK(std::abs(tl_meas - tl_want) < 1.0e-6);

        // The reflect port of a mass-law sheet is the bare near interface, Stokes-reversed. It
        // carries no mass term and no internal series, so it is a single-interface Fresnel value at
        // every angle.
        CHECK(std::abs(R - Rf.refl) < 1.0e-9);

        // The term never grows with the angle and is never a gain.
        CHECK(tl_meas >= -1.0e-9);
        CHECK(tl_meas <= tl_prev + 1.0e-9);
        tl_prev = tl_meas;

        if (thd == 30.0)
            tl_at[0] = tl_meas;
        if (thd == 60.0)
            tl_at[1] = tl_meas;
    }

    // Angle slope, taken between two rows well inside the mass-controlled region. The cosine enters
    // through the fluid admittance rather than through the panel, so the slope is 20 dB per decade
    // and does not follow the fitted m. It approaches 20 from below by the saturation gap, which is
    // 0.05 dB at 60 deg for this material. The superseded form read exactly m = 12 here.
    double c30 = std::cos(30.0 * arma::datum::pi / 180.0);
    double c60 = std::cos(60.0 * arma::datum::pi / 180.0);
    double slope = (tl_at[0] - tl_at[1]) / (std::log10(c30) - std::log10(c60));
    QD_DIAG("# angle slope, 30 to 60 deg: %.6f dB per decade of cos(theta)\n", slope);
    CHECK(slope > 19.5);
    CHECK(slope < 20.0);

    // Normal incidence is untouched. The angle law is anchored on the material table's own value,
    // so it agrees exactly with the superseded form at theta = 0. This is what keeps the public
    // medium_gain entry point, and every table calibrated through it, bit-identical.
    {
        double T = F.run(mesh, half, 0.0, 1, F.m);
        QdMassRef Rf = qd_mass_ref(F.eps, 0.0);
        double tl = dB(Rf.base) - dB(T);
        CHECK(std::abs(tl - qd_mass_TL_clipped(F.m, arg, 1.0)) < 1.0e-9);
    }

    // There is no cutoff in angle. The knee sits where x = 1, which is fixed by m, the frequency
    // and the surface mass, and by no geometry. The term there is A * log10(2) / log10(1 + X0^2),
    // which is 10 * log10(2) up to the normalization.
    {
        double want = A0 * std::log10(2.0) / std::log10(1.0 + X0 * X0);
        double T = F.run(mesh, half, th_knee, 1, F.m);
        QdMassRef Rf = qd_mass_ref(F.eps, th_knee);
        double tl = dB(Rf.base) - dB(T);
        INFO("knee at " << th_knee << " deg");
        CHECK(std::abs(tl - want) < 1.0e-6);
        CHECK(std::abs(want - 10.0 * std::log10(2.0)) < 0.01);
    }

    // Past the knee the term tapers rather than switching off: still positive at 89.9 deg, and
    // decaying as the square of the cosine.
    {
        double T = F.run(mesh, half, 89.9, 1, F.m);
        QdMassRef Rf = qd_mass_ref(F.eps, 89.9);
        double tl = dB(Rf.base) - dB(T);
        CHECK(tl > 0.0);
        CHECK(tl < 0.05);

        double T2 = F.run(mesh, half, 89.95, 1, F.m);
        QdMassRef Rf2 = qd_mass_ref(F.eps, 89.95);
        double tl2 = dB(Rf2.base) - dB(T2);
        CHECK(tl2 > 0.0);
        CHECK(tl2 < tl);
        CHECK(std::abs(tl / tl2 - 4.0) < 0.05); // cos halves, the term quarters
    }

    // The mass term is a transmission surrogate and must not reach the reflect port. Doubling m
    // leaves the reflected gain bit-identical.
    for (double thd : {30.0, 70.0, 85.0, 89.9})
    {
        double Ra = F.run(mesh, half, thd, 0, F.m);
        double Rb = F.run(mesh, half, thd, 0, 2.0 * F.m);
        INFO("reflect port, angle " << thd);
        CHECK(std::abs(Ra - Rb) < 1.0e-12);
    }
}