// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Blind verification suite for the public ray_state_update API.
//
// Implemented from the design spec (ray_state_update_spec.md, behavior oracle for the
// section-10 dispatch tables and section-9 physics) and the test spec
// (test_ray_state_update_spec.md, API contract and test plan) only. All expected values
// come from independent reference math (ITU-R P.2040 Fresnel and medium loss, the
// closed-form Airy sum), never from library internals.
//
// Conventions assumed from the public contract and the existing test suite:
//  - fbs_angleN uses the ray_mesh_interact convention: the angle between the ray and the
//    surface plane (grazing angle), i.e. pi/2 at perpendicular incidence. The reference
//    Fresnel below is parameterized by the angle from the surface normal (theta_n), so
//    tests feed fbs_angleN = pi/2 - theta_n.
//  - The propagation constant uses c0 = 299792458 m/s.
//  - The scalar (acoustic) Fresnel coefficient is the TE / admittance form, matching the
//    scalar branch verified in test_calc_diffraction_gain.cpp.
//  - Transmission-type interaction amplitudes follow the ray_mesh_interact convention of
//    energy-normalized magnitude, |t|^2 = 1 - |r|^2 (Stokes-consistent, t12*t21 = 1 - r^2).
//  - Extended mtl_prop key names assumed: "m" (mass-law column) and "tf"/"tfB"
//    (transmission factor). Adjust MTL_KEY_MASS / MTL_KEY_TF / MTL_KEY_TFB below if the
//    library uses different names; the affected tests fail loudly (engagement tripwires)
//    if the keys do not take effect.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// mu-aware reference Fresnel: an independent oracle for the e,f,g,h (permeability) path.
// At mu = 1 it reduces exactly to calc_transition_gain.
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
        mtl_ind(f) = m;
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

// Constants of the suite
static const double C0 = 299792458.0;           // speed of light, m/s
static const double QPI = arma::datum::pi;      // pi
static const double OFF = 0.001;                // ray_offset (design spec, section 4)
static const double TOL = 1.0e-9;               // double tolerance (test spec)
static const double FRQ = 1.0e9;                // default test frequency, Hz

// Assumed extended material keys (see file header). Adjust here if the library differs.
static const char *MTL_KEY_MASS = "m";
static const char *MTL_KEY_TF = "tf";
static const char *MTL_KEY_TFB = "tfB";

// Bit-mask state encoding helpers (design spec, section 7). Never arithmetic negation.
static inline short enc(int m, bool flag) { return (short)(flag ? (m | 0x8000) : m); }
static inline int mat_of(short w) { return (int)(w & 0x7FFF); }
static inline bool flag_of(short w) { return (w & 0x8000) != 0; }

// Col<short> from int literals without narrowing warnings
static inline arma::Col<short> sv(std::initializer_list<int> v)
{
    arma::Col<short> out((arma::uword)v.size());
    arma::uword i = 0;
    for (int x : v)
        out(i++) = (short)x;
    return out;
}

// Complex view of an xprmat entry pair k in {0 VV, 1 HV, 2 VH, 3 HH}
template <typename dtype>
static inline std::complex<double> cpx(const arma::Mat<dtype> &X, arma::uword i, int k)
{
    return std::complex<double>((double)X(i, 2 * k), (double)X(i, 2 * k + 1));
}

// Complex TE / scalar Fresnel pair at an interface, parameterized by the angle from the
// surface NORMAL on the incident (eta1) side. t is energy-normalized to |t|^2 = 1 - |r|^2
// with the raw Fresnel phase (the ray_mesh_interact transmission convention), so
// t12 * t21 = 1 - r^2 holds for any lossless pair (Stokes consistency, spec 9.5).
struct FresnelTE
{
    std::complex<double> r, t;
};
static inline FresnelTE fresnel_te(std::complex<double> eta1, std::complex<double> eta2, double theta_n_rad)
{
    double cth1 = std::cos(theta_n_rad), sth1 = std::sin(theta_n_rad);
    std::complex<double> n1 = std::sqrt(eta1), n2 = std::sqrt(eta2);
    std::complex<double> cth2 = std::sqrt(1.0 - (eta1 / eta2) * sth1 * sth1);
    std::complex<double> den = n1 * cth1 + n2 * cth2;
    std::complex<double> r = (n1 * cth1 - n2 * cth2) / den;
    std::complex<double> t = 2.0 * n1 * cth1 / den;
    double tm = std::abs(t);
    if (tm > 0.0)
        t *= std::sqrt(std::max(0.0, 1.0 - std::norm(r))) / tm;
    return {r, t};
}

// One-way in-slab propagation factor phi (design spec, section 9.1):
// |phi| = sqrt(medium_gain(L)), arg(phi) = -(omega/c) * Re(sqrt(eta_slab)) * L
static inline std::complex<double> phi_one_way(double n_real, double medium_gain_L, double f_hz, double L)
{
    double beta_L = 2.0 * QPI * f_hz / C0 * n_real * L;
    return std::sqrt(medium_gain_L) * std::exp(std::complex<double>(0.0, -beta_L));
}

// Airy factor S = 1 / (1 - r_near * r_far * phi^2)
static inline std::complex<double> airy_S(std::complex<double> r_near, std::complex<double> r_far,
                                          std::complex<double> phi)
{
    return 1.0 / (1.0 - r_near * r_far * phi * phi);
}

// Survival-gate quantity rho = sqrt(R_near * R_far * medium_gain(2L)) (spec 9.4)
static inline double gate_rho(std::complex<double> r_near, std::complex<double> r_far, double medium_gain_2L)
{
    return std::sqrt(std::norm(r_near) * std::norm(r_far) * medium_gain_2L);
}

// One-way path of m half-waves inside a medium of (real) index n: 2*k*n*L = 2*pi*m at L = m * half_wave
static inline double half_wave(double n_real, double f_hz)
{
    return C0 / (2.0 * f_hz * n_real);
}

// Material palette (test spec, section 3), 1-based indices, 0 = air:
//   1 FOG1   eta = 1, alpha = 2 dB/m  (no Fresnel mismatch, pure in-medium loss)
//   2 FOG2   identical properties to FOG1 but a distinct index, so same_materials(...) on
//            indices is false while every transition factor TRN(FOG1, FOG2) is exactly 1
//   3 FOG3   eta = 1, alpha = 5 dB/m
//   4 DENSE  eta = 4, lossless, high mismatch (n = 2, inside-face r = 1/3)
//   5 NINE   eta = 9, lossless (n = 3, inside-face r = 1/2, lossless rho = 0.25)
//   6 ABSORB eta = 4, alpha = 30 dB/m (resolve-critical lossy slab)
//   7 WEAK   eta = 1.21, lossless (near-lossless weak cavity)
// All have b = c = d = att = attB = alphaB = 0 and fRef = 1, so eta is real and frequency
// flat and the only medium loss is the alpha power law, alpha * dist dB, exactly
// medg(idx, dist) below. TRN between any eta = 1 pair (or eta = 1 and air) is exactly 1.
template <typename dtype>
static inline std::unordered_map<std::string, std::vector<dtype>> base_palette()
{
    std::unordered_map<std::string, std::vector<dtype>> m;
    m["a"] = {(dtype)1.0, (dtype)1.0, (dtype)1.0, (dtype)4.0, (dtype)9.0, (dtype)4.0, (dtype)1.21};
    m["b"] = std::vector<dtype>(7, (dtype)0.0);
    m["c"] = std::vector<dtype>(7, (dtype)0.0);
    m["d"] = std::vector<dtype>(7, (dtype)0.0);
    m["att"] = std::vector<dtype>(7, (dtype)0.0);
    m["attB"] = std::vector<dtype>(7, (dtype)0.0);
    m["alpha"] = {(dtype)2.0, (dtype)2.0, (dtype)5.0, (dtype)0.0, (dtype)0.0, (dtype)30.0, (dtype)0.0};
    m["alphaB"] = std::vector<dtype>(7, (dtype)0.0);
    m["fRef"] = std::vector<dtype>(7, (dtype)1.0);
    return m;
}

// Medium gain of a palette material over dist meters (alpha-only loss, alphaB = 0)
static inline double medg(int idx, double dist)
{
    static const double alpha[8] = {0.0, 2.0, 2.0, 5.0, 0.0, 0.0, 30.0, 0.0};
    return std::pow(10.0, -0.1 * alpha[idx] * dist);
}

// Real refractive index of a palette material
static inline double n_of(int idx)
{
    static const double a[8] = {1.0, 1.0, 1.0, 1.0, 4.0, 9.0, 4.0, 1.21};
    return std::sqrt(a[idx]);
}

// Call harness. Owns every argument of ray_state_update; has_* switches turn each
// optional (and, for the validation group, required) pointer into nullptr. run() takes a
// snapshot of the incoming gain/xprmat so factor checks can compare out against in.
template <typename dtype>
struct Rsu
{
    int itype = 4;
    dtype freq = (dtype)FRQ;
    double eps = 0.15;

    arma::Mat<dtype> orig, dest, fbs, sbs;
    arma::u32_vec no_interact;
    arma::Col<dtype> fbs_angle;
    arma::s32_vec out_type;
    std::unordered_map<std::string, std::vector<dtype>> mtl;
    arma::Col<short> m1, m2;
    arma::Col<short> prev_in, cur_in, buf_in;
    arma::Mat<dtype> normals;
    arma::Col<short> prev_out, cur_out, buf_out;
    arma::Col<dtype> gain;
    arma::Mat<dtype> xprmat;
    arma::u32_vec ray_ind;

    arma::Col<dtype> g_in; // snapshots taken by run()
    arma::Mat<dtype> x_in;

    bool has_orig = true, has_dest = true, has_fbs = true, has_sbs = true, has_ni = true;
    bool has_angle = true, has_otype = true, has_mtl = true, has_m1 = true, has_m2 = true;
    bool has_prev_in = true, has_cur_in = true, has_buf_in = true, has_normals = true;
    bool has_prev_out = true, has_cur_out = true, has_buf_out = true;
    bool has_gain = true, has_xprmat = true, has_ray_ind = false;

    void run()
    {
        g_in = gain;
        x_in = xprmat;
        quadriga_lib::ray_state_update<dtype>(
            itype, freq,
            has_orig ? &orig : nullptr, has_dest ? &dest : nullptr,
            has_fbs ? &fbs : nullptr, has_sbs ? &sbs : nullptr,
            has_ni ? &no_interact : nullptr,
            has_angle ? &fbs_angle : nullptr, has_otype ? &out_type : nullptr,
            has_mtl ? &mtl : nullptr,
            has_m1 ? &m1 : nullptr, has_m2 ? &m2 : nullptr,
            has_prev_in ? &prev_in : nullptr,
            has_cur_in ? &cur_in : nullptr,
            has_buf_in ? &buf_in : nullptr,
            has_normals ? &normals : nullptr,
            has_prev_out ? &prev_out : nullptr,
            has_cur_out ? &cur_out : nullptr,
            has_buf_out ? &buf_out : nullptr,
            has_gain ? &gain : nullptr,
            has_xprmat ? &xprmat : nullptr,
            has_ray_ind ? &ray_ind : nullptr,
            eps);
    }
};

// Single-ray call builder on the x axis: orig at 0, fbs at d_orig_fbs, sbs at
// d_orig_fbs + d_fbs_sbs, dest at d_orig_fbs + d_fbs_dest. The default normals are the
// antiparallel slab pair (-1,0,0 | 1,0,0). The probe field is `feed` on VV (scalar mode)
// or on VV and HH (EM mode) with gainN consistent under the mode's convention.
template <typename dtype>
static inline Rsu<dtype> make1(const std::unordered_map<std::string, std::vector<dtype>> &mtl,
                               int itype, int otype, unsigned nH, int M1, int M2,
                               int prev, int cur, int buf,
                               double d_orig_fbs, double d_fbs_dest, double d_fbs_sbs,
                               double theta_graz_rad, double eps,
                               std::complex<double> feed = std::complex<double>(0.5, 0.3),
                               double f_hz = FRQ)
{
    Rsu<dtype> C;
    C.itype = itype;
    C.eps = eps;
    C.freq = (dtype)f_hz;
    C.mtl = mtl;

    C.orig.zeros(1, 3);
    C.fbs.zeros(1, 3);
    C.fbs(0, 0) = (dtype)d_orig_fbs;
    C.sbs.zeros(1, 3);
    C.sbs(0, 0) = (dtype)(d_orig_fbs + d_fbs_sbs);
    C.dest.zeros(1, 3);
    C.dest(0, 0) = (dtype)(d_orig_fbs + d_fbs_dest);

    C.no_interact.set_size(1);
    C.no_interact(0) = nH;
    C.fbs_angle.set_size(1);
    C.fbs_angle(0) = (dtype)theta_graz_rad;
    C.out_type.set_size(1);
    C.out_type(0) = otype;

    C.m1 = sv({M1});
    C.m2 = sv({M2});
    C.prev_in = sv({prev});
    C.cur_in = sv({cur});
    C.buf_in = sv({buf});

    C.normals.zeros(1, 6);
    C.normals(0, 0) = (dtype)-1.0;
    C.normals(0, 3) = (dtype)1.0;

    C.prev_out.set_size(1);
    C.cur_out.set_size(1);
    C.buf_out.set_size(1);
    C.prev_out.fill((short)11111); // sentinel: every output word must be written
    C.cur_out.fill((short)11111);
    C.buf_out.fill((short)11111);

    C.xprmat.zeros(1, 8);
    C.xprmat(0, 0) = (dtype)feed.real();
    C.xprmat(0, 1) = (dtype)feed.imag();
    if (itype < 3)
    {
        C.xprmat(0, 6) = (dtype)feed.real();
        C.xprmat(0, 7) = (dtype)feed.imag();
    }
    C.gain.set_size(1);
    C.gain(0) = (dtype)std::norm(feed); // 0.5*(|VV|^2+|HH|^2) in EM, |VV|^2 in scalar

    return C;
}

// State words, exact short comparison
template <typename dtype>
static inline void check_state(const Rsu<dtype> &C, int prev, int cur, int buf, arma::uword i = 0)
{
    CHECK((int)C.prev_out(i) == (int)(short)prev);
    CHECK((int)C.cur_out(i) == (int)(short)cur);
    CHECK((int)C.buf_out(i) == (int)(short)buf);
}

// IG keep: outputs bit-identical to inputs
template <typename dtype>
static inline void check_keep(const Rsu<dtype> &C)
{
    CHECK(arma::approx_equal(C.xprmat, C.x_in, "absdiff", (dtype)0));
    if (C.has_gain)
        CHECK(arma::approx_equal(C.gain, C.g_in, "absdiff", (dtype)0));
}

// KILL: gain and xprmat zero
template <typename dtype>
static inline void check_kill(const Rsu<dtype> &C, arma::uword i = 0)
{
    for (int c = 0; c < 8; ++c)
        CHECK(std::abs((double)C.xprmat(i, c)) == 0.0);
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i)) == 0.0);
}

// IG * f: each complex Jones pair multiplied by f, gain by |f|^2
template <typename dtype>
static inline void check_mult(const Rsu<dtype> &C, std::complex<double> f, double tol, arma::uword i = 0)
{
    for (int k = 0; k < 4; ++k)
    {
        std::complex<double> e = cpx(C.x_in, i, k) * f;
        CHECK(std::abs((double)C.xprmat(i, 2 * k) - e.real()) < tol);
        CHECK(std::abs((double)C.xprmat(i, 2 * k + 1) - e.imag()) < tol);
    }
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i) - (double)C.g_in(i) * std::norm(f)) < tol);
}

// Isotropic replace with power g: sqrt(g) on VV (and HH in EM mode), zeros elsewhere,
// gain = g under both conventions (design spec, section 5)
template <typename dtype>
static inline void check_replace(const Rsu<dtype> &C, double g, double tol, arma::uword i = 0)
{
    double s = std::sqrt(g);
    bool scalar = C.itype >= 3;
    CHECK(std::abs((double)C.xprmat(i, 0) - s) < tol);
    CHECK(std::abs((double)C.xprmat(i, 1)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 2)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 3)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 4)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 5)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 6) - (scalar ? 0.0 : s)) < tol);
    CHECK(std::abs((double)C.xprmat(i, 7)) < tol);
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i) - g) < tol);
}

// Applied complex factor on the VV pair
template <typename dtype>
static inline std::complex<double> vvf(const Rsu<dtype> &C, arma::uword i = 0)
{
    return cpx(C.xprmat, i, 0) / cpx(C.x_in, i, 0);
}

// Standard single-ray dispatch probe: scalar transmission pass, eps = 1.5 (S off),
// normal incidence, default distances d(orig,fbs) = 1, d(fbs,dest) = 2, d(fbs,sbs) = 0.5
static inline Rsu<double> disp(int otype, unsigned nH, int M1, int M2,
                               int prev, int cur, int buf,
                               double dof = 1.0, double dfd = 2.0, double dsbs = 0.5,
                               int itype = 4, double eps = 1.5)
{
    static const std::unordered_map<std::string, std::vector<double>> mtl = base_palette<double>();
    auto C = make1<double>(mtl, itype, otype, nH, M1, M2, prev, cur, buf, dof, dfd, dsbs, QPI / 2.0, eps);
    C.run();
    return C;
}

TEST_CASE("ray_state_update - state word encoding round trip")
{
    // Test spec 5.1, design spec section 7: bit-mask encoding, never arithmetic negation
    CHECK(enc(0, true) == (short)-32768); // resolved air word is 0x8000, not -0
    CHECK(mat_of(enc(0, true)) == 0);
    CHECK(flag_of(enc(0, true)));
    CHECK(enc(5, true) != (short)-5); // 0x8005, not the arithmetic negative
    CHECK(mat_of(enc(5, true)) == 5);
    CHECK(flag_of(enc(5, true)));
    CHECK(mat_of(enc(5, false)) == 5);
    CHECK(!flag_of(enc(5, false)));
    CHECK(mat_of((short)0x7FFF) == 0x7FFF);
}

TEST_CASE("ray_state_update - dispatch: resolved-ray precedence")
{
    // Test spec 4.1, design spec 10.0. The resolved branch runs before topology dispatch.

    SECTION("reflection pass kills a resolved ray, state copies through") // spec 10.0
    {
        auto C = disp(2, 1, 4, 0, 0, enc(4, true), 0, 1.0, 2.0, 0.5, 3, 1.5);
        check_kill(C);
        check_state(C, 0, enc(4, true), 0);
    }
    SECTION("transmission pass, resolved i-o exit clears the state") // spec 10.0
    {
        auto C = disp(2, 1, 4, 0, 0, enc(4, true), 0);
        check_keep(C);
        check_state(C, 0, 0, 0);

        // The resolved exit owns the in-medium loss of its incoming segment (up trip):
        // factor sqrt(MED(cur, d(orig, fbs))), here over the default d(orig, fbs) = 1
        auto D = disp(8, 2, 4, 0, 0, enc(1, true), 0);
        check_mult(D, std::sqrt(medg(1, 1.0)), TOL);
        check_state(D, 0, 0, 0);

        auto E = disp(14, 2, 4, 0, 0, enc(1, true), 0);
        check_mult(E, std::sqrt(medg(1, 1.0)), TOL);
        check_state(E, 0, 0, 0);
    }
    SECTION("transmission pass, resolved i-i crossing keeps the flag and shifts prev") // spec 10.0
    {
        // Resolved i-i also charges its incoming segment: sqrt(MED(cur, d(orig, fbs)))
        auto C = disp(4, 2, 2, 3, 0, enc(1, true), 0);
        check_mult(C, std::sqrt(medg(1, 1.0)), TOL);
        check_state(C, 1, enc(2, true), 0); // iM = M1 for type 4, prev <- old cur material

        auto D = disp(5, 2, 2, 3, 0, enc(1, true), 0);
        check_mult(D, std::sqrt(medg(1, 1.0)), TOL);
        check_state(D, 1, enc(3, true), 0); // iM = M2 for type 5
    }
    SECTION("transmission pass, resolved transparent pass-through elsewhere") // spec 10.0
    {
        auto C = disp(1, 1, 2, 0, 0, enc(1, true), 0); // o-i while resolved
        check_keep(C);
        check_state(C, 0, enc(1, true), 0);

        auto D = disp(10, 2, 2, 2, 0, enc(1, true), 0); // o-i-o edge while resolved
        check_keep(D);
        check_state(D, 0, enc(1, true), 0);

        auto E = disp(11, 2, 2, 2, 0, enc(1, true), 0); // i-o-i edge while resolved
        check_keep(E);
        check_state(E, 0, enc(1, true), 0);
    }
}

TEST_CASE("ray_state_update - dispatch: o-i entry family")
{
    // Test spec 4.2, design spec 10.1. eps = 1.5 keeps the survival gate off.

    SECTION("branch A, clean entry applies the medium gain and sets cur") // spec 10.1
    {
        // type 1, nH = 1, cur = 0: IG * sqrt(MED(M1, d(fbs,dest) - off)), state (0, M1, 0).
        // prev_in = 3 is a sentinel proving the explicit prev_out <- 0 write.
        auto C = disp(1, 1, 1, 0, 3, 0, 0, 1.0, 2.0);
        check_mult(C, std::sqrt(medg(1, 2.0 - OFF)), TOL);
        check_state(C, 0, 1, 0);
    }
    SECTION("branch A, the entry clamp keeps the medium distance non-negative") // spec 10.1
    {
        // d(fbs,dest) = 0.0005 < ray_offset: max(d - off, 0) = 0, the factor must be 1
        auto C = disp(1, 1, 1, 0, 0, 0, 0, 1.0, 0.0005);
        check_mult(C, 1.0, TOL);
        check_state(C, 0, 1, 0);
    }
    SECTION("branch A also serves nH = 2 types 7 and 13") // spec 10.1
    {
        auto C = disp(7, 2, 1, 0, 0, 0, 0, 1.0, 1.5);
        check_mult(C, std::sqrt(medg(1, 1.5 - OFF)), TOL);
        check_state(C, 0, 1, 0);

        auto D = disp(13, 2, 1, 0, 0, 0, 0, 1.0, 1.5);
        check_mult(D, std::sqrt(medg(1, 1.5 - OFF)), TOL);
        check_state(D, 0, 1, 0);
    }
    SECTION("branch A routes identically on the EM and refraction passes") // spec 10.1
    {
        static const auto mtl = base_palette<double>();
        for (int it : {1, 2})
        {
            auto C = make1<double>(mtl, it, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
            C.run();
            check_mult(C, std::sqrt(medg(1, 2.0 - OFF)), TOL);
            check_state(C, 0, 1, 0);
        }
    }
    SECTION("branch A, nested entry replaces with the outer-medium gain and buffers M1") // spec 10.1
    {
        // cur != 0: replace MED(cur, d(orig,dest)), buffer <- M1
        auto C = disp(1, 1, 3, 0, 0, 2, 0, 1.0, 2.0); // d(orig,dest) = 3
        check_replace(C, medg(2, 3.0), TOL);
        check_state(C, 0, 2, 3);
    }
    SECTION("branch B, slab entry from air is gain-neutral") // spec 10.1
    {
        auto C = disp(1, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5);
        check_keep(C);
        check_state(C, 0, 4, 0); // antiparallel normals: non-parallel flag stays clear
    }
    SECTION("branch B, nested slab entry replaces and buffers M1") // spec 10.1
    {
        auto C = disp(1, 2, 1, 1, 0, 3, 0, 1.5, 2.0, 0.5); // replace MED(cur, d(orig,fbs))
        check_replace(C, medg(3, 1.5), TOL);
        check_state(C, 0, 3, 1);
    }
    SECTION("branch C, nH > 2 with cur = 0 and a non-empty buffer is inconsistent") // spec 10.1
    {
        auto C = disp(1, 3, 1, 0, 0, 0, 2);
        check_kill(C);
        check_state(C, 0, 0, 2);
    }
    SECTION("branch C, nH > 2 clean entries set cur (and the buffer for type 13)") // spec 10.1
    {
        auto C = disp(1, 3, 4, 0, 0, 0, 0);
        check_keep(C);
        check_state(C, 0, 4, 0);

        auto D = disp(7, 3, 4, 0, 0, 0, 0);
        check_keep(D);
        check_state(D, 0, 4, 0);

        auto E = disp(13, 3, 1, 2, 0, 0, 0);
        check_keep(E);
        check_state(E, 0, 1, 2);
    }
    SECTION("branch C, nH > 2 nested entry replaces over d(orig,fbs) + off") // spec 10.1
    {
        auto C = disp(7, 3, 3, 0, 0, 1, 0, 2.0, 2.0, 0.5);
        check_replace(C, medg(1, 2.0 + OFF), TOL);
        check_state(C, 0, 1, 3);
    }
}

TEST_CASE("ray_state_update - dispatch: i-o exit family")
{
    // Test spec 4.3, design spec 10.2. eps = 1.5 keeps the survival gate off, so every
    // cavity exit re-emits with the legacy gain untouched.

    SECTION("branch A, cur = 0 means the inside classification was false, pass through") // spec 10.2
    {
        auto C = disp(2, 1, 4, 0, 0, 0, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);

        auto D = disp(8, 2, 4, 0, 0, 0, 0);
        check_keep(D);
        check_state(D, 0, 0, 0);
    }
    SECTION("branch A, cavity exit clears cur (S gated off here)") // spec 10.2
    {
        auto C = disp(2, 1, 4, 0, 0, 4, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);

        auto D = disp(14, 2, 1, 0, 0, 1, 0);
        check_keep(D);
        check_state(D, 0, 0, 0);
    }
    SECTION("branch A, buffered exit into the same material flushes the buffer") // spec 10.2
    {
        // same_materials(buf, M1): replace MED(cur, d(orig,dest)), buffer cleared
        auto C = disp(2, 1, 2, 0, 0, 1, 2, 1.0, 2.0); // d(orig,dest) = 3
        check_replace(C, medg(1, 3.0), TOL);
        check_state(C, 0, 1, 0);
    }
    SECTION("branch A, buffered exit into a different material chains the gains") // spec 10.2
    {
        // replace MED(cur, d(orig,fbs)) * TRN(cur, buf) * MED(buf, d(fbs,dest)); TRN = 1
        // between the eta-1 FOG materials, so the chain is the two medium gains
        auto C = disp(2, 1, 2, 0, 0, 1, 3, 1.2, 0.7);
        check_replace(C, medg(1, 1.2) * medg(3, 0.7), TOL);
        check_state(C, 0, 3, 0);
    }
    SECTION("branch A, type 8 ii-oo crossing leaves both media at once") // spec 10.2
    {
        auto C = disp(8, 2, 2, 0, 0, 1, 2, 0.9, 2.0);
        check_replace(C, medg(1, 0.9), TOL); // MED(cur, d(orig,fbs)) * TRN(cur, 0)
        check_state(C, 0, 0, 0);
    }
    SECTION("branch B, type 2 at nH = 2 with no far material is inconsistent") // spec 10.2
    {
        auto C = disp(2, 2, 1, 0, 0, 1, 0);
        check_kill(C);
        check_state(C, 0, 1, 0);
    }
    SECTION("branch B, air-gap exit survives with cur cleared (S gated off)") // spec 10.2
    {
        auto C = disp(2, 2, 1, 4, 0, 1, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("branch B, buffered same-material exit") // spec 10.2
    {
        auto C = disp(2, 2, 2, 4, 0, 1, 2, 1.4, 2.0);
        check_replace(C, medg(1, 1.4 + OFF), TOL); // MED(cur, d(orig,fbs) + off)
        check_state(C, 0, 1, 0);
    }
    SECTION("branch B, buffered different-material exit") // spec 10.2
    {
        auto C = disp(2, 2, 2, 4, 0, 1, 3, 1.4, 2.0);
        check_replace(C, medg(1, 1.4) * medg(3, OFF), TOL);
        check_state(C, 0, 3, 0);
    }
    SECTION("branch B, a buffer without a current medium is inconsistent") // spec 10.2
    {
        auto C = disp(2, 2, 1, 4, 0, 0, 2);
        check_kill(C);
        check_state(C, 0, 0, 2);
    }
    SECTION("branch C, nH > 2 false-inside and cavity exits") // spec 10.2
    {
        auto C = disp(2, 3, 4, 0, 0, 0, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);

        auto D = disp(14, 3, 4, 0, 0, 4, 0);
        check_keep(D);
        check_state(D, 0, 0, 0);
    }
    SECTION("branch C, nH > 2 buffered exits") // spec 10.2
    {
        auto C = disp(2, 3, 2, 0, 0, 1, 2, 1.1, 2.0); // same material
        check_replace(C, medg(1, 1.1 + OFF), TOL);
        check_state(C, 0, 1, 0);

        auto D = disp(14, 3, 2, 0, 0, 1, 3, 1.1, 2.0); // different material
        check_replace(D, medg(1, 1.1) * medg(3, OFF), TOL);
        check_state(D, 0, 3, 0);
    }
    SECTION("branch D, type 8 at nH > 2") // spec 10.2
    {
        auto C = disp(8, 3, 1, 0, 0, 1, 0); // plain exit
        check_keep(C);
        check_state(C, 0, 0, 0);

        auto D = disp(8, 3, 1, 0, 0, 1, 2, 0.8, 2.0); // buffered ii-oo
        check_replace(D, medg(1, 0.8), TOL);
        check_state(D, 0, 0, 0);

        auto E = disp(8, 3, 1, 0, 0, 0, 0); // cur = 0 is inconsistent here
        check_kill(E);
        check_state(E, 0, 0, 0);
    }
}

TEST_CASE("ray_state_update - dispatch: material-to-material family")
{
    // Test spec 4.4, design spec 10.3, types 4 and 5

    SECTION("an i-i crossing from outside any medium is inconsistent") // spec 10.3
    {
        auto C = disp(4, 2, 1, 2, 0, 0, 0);
        check_kill(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("an i-i crossing with a missing face material is inconsistent") // spec 10.3
    {
        auto C = disp(4, 2, 0, 2, 0, 1, 0);
        check_kill(C);
        check_state(C, 0, 1, 0);

        auto D = disp(4, 2, 1, 0, 0, 1, 0);
        check_kill(D);
        check_state(D, 0, 1, 0);
    }
    SECTION("cavity transition applies the next medium gain and shifts prev") // spec 10.3
    {
        // type 4: iM = M1. IG * sqrt(MED(iM, d(fbs,dest) - off)) without the entry clamp.
        auto C = disp(4, 2, 1, 2, 0, 4, 0, 1.0, 1.2);
        check_mult(C, std::sqrt(medg(1, 1.2 - OFF)), TOL);
        check_state(C, 4, 1, 0);

        // type 5: iM = M2
        auto D = disp(5, 2, 2, 3, 0, 4, 0, 1.0, 1.2);
        check_mult(D, std::sqrt(medg(3, 1.2 - OFF)), TOL);
        check_state(D, 4, 3, 0);
    }
    SECTION("buffered i-i replaces and swaps the buffer to the far face material") // spec 10.3
    {
        // replace MED(cur, d(orig,dest)); buffer <- same_materials(buf, M1) ? M2 : M1
        auto C = disp(4, 2, 2, 3, 0, 1, 2, 1.0, 1.5); // buf == M1 -> buffer <- M2 = 3
        check_replace(C, medg(1, 2.5), TOL);
        check_state(C, 0, 1, 3);

        auto D = disp(4, 2, 3, 2, 0, 1, 2, 1.0, 1.5); // buf != M1 -> buffer <- M1 = 3
        check_replace(D, medg(1, 2.5), TOL);
        check_state(D, 0, 1, 3);
    }
    SECTION("nH > 2 with a buffer only flushes the spurious buffer") // spec 10.3
    {
        auto C = disp(4, 3, 2, 3, 0, 1, 2);
        check_keep(C);
        check_state(C, 0, 1, 0);
    }
    SECTION("nH > 2 cavity transition shifts state without a medium gain") // spec 10.3
    {
        auto C = disp(5, 3, 2, 3, 0, 1, 0, 1.0, 1.2);
        check_keep(C); // no MED term at nH > 2
        check_state(C, 1, 3, 0);
    }
    SECTION("nH > 2 i-i from outside any medium is inconsistent") // spec 10.3
    {
        auto C = disp(4, 3, 1, 2, 0, 0, 0);
        check_kill(C);
        check_state(C, 0, 0, 0);
    }
}

TEST_CASE("ray_state_update - dispatch: o-i-o edge family")
{
    // Test spec 4.5, design spec 10.4, type 10 (no S factor on this topology)

    SECTION("nH = 2 grazing pass outside any medium") // spec 10.4
    {
        auto C = disp(10, 2, 2, 2, 0, 0, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("nH = 2 inside, same material on both faces") // spec 10.4
    {
        auto C = disp(10, 2, 2, 2, 0, 1, 0, 1.0, 1.2); // replace MED(cur, d(orig,dest))
        check_replace(C, medg(1, 2.2), TOL);
        check_state(C, 0, 1, 0);
    }
    SECTION("nH = 2 inside, different materials chain through M1") // spec 10.4
    {
        // replace MED(cur, d(orig,fbs)) * TRN(cur, M1) * MED(M1, d(fbs,dest)); TRN = 1
        auto C = disp(10, 2, 2, 3, 0, 1, 0, 0.8, 1.0);
        check_replace(C, medg(1, 0.8) * medg(2, 1.0), TOL);
        check_state(C, 0, 2, 0);
    }
    SECTION("nH > 2 grazing pass outside any medium") // spec 10.4
    {
        auto C = disp(10, 3, 2, 2, 0, 0, 0);
        check_keep(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("nH > 2 inside without a buffer") // spec 10.4
    {
        auto C = disp(10, 3, 2, 2, 0, 1, 0, 1.3, 2.0); // same material
        check_replace(C, medg(1, 1.3 + OFF), TOL);
        check_state(C, 0, 1, 0);

        auto D = disp(10, 3, 2, 3, 0, 1, 0, 1.3, 2.0); // different material
        check_replace(D, medg(1, 1.3) * medg(2, OFF), TOL);
        check_state(D, 0, 2, 0);
    }
    SECTION("nH > 2 inside with a buffer") // spec 10.4
    {
        auto C = disp(10, 3, 2, 3, 0, 1, 2, 1.3, 2.0); // same_materials(buf, M1)
        check_replace(C, medg(1, 1.3 + OFF), TOL);
        check_state(C, 0, 1, 0);

        auto D = disp(10, 3, 3, 2, 0, 1, 2, 1.3, 2.0); // buf != M1
        check_replace(D, medg(1, 1.3) * medg(2, OFF), TOL);
        check_state(D, 0, 2, 0);
    }
}

TEST_CASE("ray_state_update - dispatch: i-o-i edge family")
{
    // Test spec 4.6, design spec 10.5, type 11

    SECTION("nH = 2 entry through an edge picks up M2 when the hits are distinct") // spec 10.5
    {
        auto C = disp(11, 2, 2, 4, 0, 0, 0, 1.0, 2.0, 0.5); // d(fbs,sbs) = 0.5 > 1e-6
        check_keep(C);
        check_state(C, 0, 4, 0);
    }
    SECTION("nH = 2 degenerate edge with coincident hits stays outside") // spec 10.5
    {
        auto C = disp(11, 2, 2, 4, 0, 0, 0, 1.0, 2.0, 0.0); // sbs == fbs
        check_keep(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("nH = 2 inside, same material") // spec 10.5
    {
        auto C = disp(11, 2, 2, 2, 0, 1, 0, 1.0, 1.0); // replace MED(cur, d(orig,dest))
        check_replace(C, medg(1, 2.0), TOL);
        check_state(C, 0, 1, 0);
    }
    SECTION("nH = 2 inside, different materials re-enter through M2") // spec 10.5
    {
        auto C = disp(11, 2, 2, 3, 0, 1, 0, 1.0, 1.5); // replace MED(M2, d(fbs,dest) - off)
        check_replace(C, medg(3, 1.5 - OFF), TOL);
        check_state(C, 0, 3, 0);
    }
    SECTION("nH > 2 with cur = 0 is inconsistent") // spec 10.5
    {
        auto C = disp(11, 3, 2, 3, 0, 0, 0);
        check_kill(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("nH > 2 inside without a buffer") // spec 10.5
    {
        auto C = disp(11, 3, 2, 2, 0, 1, 0, 0.9, 2.0); // same material
        check_replace(C, medg(1, 0.9 + OFF), TOL);
        check_state(C, 0, 1, 0);

        auto D = disp(11, 3, 2, 3, 0, 1, 0, 0.9, 2.0); // different material
        check_replace(D, medg(1, 0.9) * medg(3, OFF), TOL);
        check_state(D, 0, 3, 0);
    }
    SECTION("nH > 2 with a buffer only flushes the spurious buffer") // spec 10.5
    {
        auto C = disp(11, 3, 2, 3, 0, 1, 2);
        check_keep(C);
        check_state(C, 0, 1, 0);
    }
}

TEST_CASE("ray_state_update - dispatch: total reflection kills the refraction path")
{
    // Test spec 4.7 and 6.9, design spec 10.6. TR codes are 3, 6, 9, 12, 15 and appear on
    // the refraction pass (interaction type 2).

    SECTION("every TR code zeroes the ray and copies the state") // spec 10.6
    {
        for (int ot : {3, 6, 9, 12, 15})
        {
            auto C = disp(ot, ot > 3 ? 2u : 1u, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, 2);
            check_kill(C);
            check_state(C, 0, 1, 0);
        }
    }
    SECTION("a resolved ray with a TR code is also killed, flag intact") // spec 10.0
    {
        auto C = disp(3, 1, 1, 0, 0, enc(1, true), 0, 1.0, 2.0, 0.5, 2);
        check_kill(C);
        check_state(C, 0, enc(1, true), 0);
    }
    SECTION("TR codes on the other passes fall into the global default kill") // spec 10.8
    {
        auto C = disp(3, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, 1);
        check_kill(C);
        check_state(C, 0, 1, 0);

        auto D = disp(6, 2, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, 4);
        check_kill(D);
        check_state(D, 0, 1, 0);
    }
}

TEST_CASE("ray_state_update - dispatch: reflection pass")
{
    // Test spec 4.8 and 6.7, design spec 10.7

    SECTION("front-side reflection outside any medium passes through unchanged") // spec 10.7
    {
        for (int it : {0, 3})
        {
            auto C = disp(1, 1, 4, 0, 0, 0, 0, 1.0, 2.0, 0.5, it, 0.0); // even with eps = 0
            check_keep(C);
            check_state(C, 0, 0, 0);
        }
    }
    SECTION("internal reflection re-emits unflagged when the gate is off") // spec 10.7
    {
        auto C = disp(2, 1, 4, 0, 0, 4, 0, 1.0, 2.0, 0.5, 3, 1.5);
        check_keep(C);
        check_state(C, 0, 4, 0); // no resolved flag
    }
    SECTION("internal reflection of a resolvable slab applies S and flags the ray") // spec 10.7
    {
        // DENSE cavity, one-way path at the m = 2 resonance: S = 1/(1 - (1/3)^2) = 1.125
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        CHECK(std::abs(S - 1.125) < TOL); // oracle self-check at resonance
        check_mult(C, S, TOL);
        check_state(C, 0, enc(4, true), 0);
    }
}

TEST_CASE("ray_state_update - dispatch: global default is a kill, not an exception")
{
    // Test spec 4.9, design spec 10.8. Unhandled (interaction type, out type, state)
    // combinations are data, not programming errors.

    SECTION("transmission pass on a pure-reflection code")
    {
        auto C = disp(0, 1, 1, 0, 0, 1, 0); // out type 0 has no transmission row
        check_kill(C);
        check_state(C, 0, 1, 0);
    }
    SECTION("inconsistent hit count for the code")
    {
        auto C = disp(7, 1, 1, 0, 0, 1, 0); // type 7 is an nH = 2 code
        check_kill(C);
        check_state(C, 0, 1, 0);
    }
    SECTION("an out-of-range out type is data and kills the ray")
    {
        Rsu<double> C;
        CHECK_NOTHROW(C = disp(99, 1, 1, 0, 0, 1, 0));
        check_kill(C);
        check_state(C, 0, 1, 0);
    }
}

TEST_CASE("ray_state_update - flag semantics")
{
    // Test spec 5.2-5.4, design spec sections 7 and 9.3
    static const auto mtl = base_palette<double>();

    SECTION("the resolved flag lands on current_out as a negative short")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        CHECK((int)C.cur_out(0) < 0);
        CHECK(mat_of(C.cur_out(0)) == 4);
        CHECK(flag_of(C.cur_out(0)));
    }
    SECTION("a wedge entry sets the non-parallel flag on prev_out")
    {
        // Slab entry from air (type 1, nH = 2, cur = 0): the wedge test compares the two
        // face normals. |dot| = 0.5 here, well below the parallel threshold.
        auto C = make1<double>(mtl, 4, 1, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.normals(0, 0) = -1.0;
        C.normals(0, 1) = 0.0;
        C.normals(0, 2) = 0.0;
        C.normals(0, 3) = 0.5;
        C.normals(0, 4) = std::sqrt(3.0) / 2.0;
        C.normals(0, 5) = 0.0;
        C.run();
        check_keep(C);
        CHECK((int)C.cur_out(0) == 4);
        CHECK((int)C.prev_out(0) == (int)(short)0x8000); // flag on air: exactly -32768
        CHECK(mat_of(C.prev_out(0)) == 0);
        CHECK(flag_of(C.prev_out(0)));
        CHECK((int)C.buf_out(0) == 0);
    }
    SECTION("antiparallel and parallel faces leave the non-parallel flag clear")
    {
        // Antiparallel pair (default normals)
        auto C = disp(1, 2, 4, 4, 0, 0, 0);
        check_state(C, 0, 4, 0);

        // Parallel pair, same sign: |dot| = 1 also counts as parallel (spec 9.3)
        auto D = make1<double>(mtl, 4, 1, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        D.normals(0, 0) = -1.0;
        D.normals(0, 3) = -1.0;
        D.run();
        check_state(D, 0, 4, 0);
    }
}

TEST_CASE("ray_state_update - ray_ind mapping and read-only inputs")
{
    // Test spec 5.5 and 5.6: full-set arrays are read at g = ray_ind[i], compact arrays
    // at i, and no input array is modified.
    auto mtl = base_palette<double>();

    Rsu<double> C;
    C.itype = 4;
    C.eps = 1.5;
    C.mtl = mtl;
    C.has_ray_ind = true;

    const arma::uword n_ray = 5, n_rayN = 2;
    C.orig.zeros(n_ray, 3);
    C.fbs.zeros(n_ray, 3);
    C.sbs.zeros(n_ray, 3);
    C.dest.zeros(n_ray, 3);
    for (arma::uword g = 0; g < n_ray; ++g)
    {
        C.fbs(g, 0) = 0.4 * (double)(g + 1);
        C.sbs(g, 0) = C.fbs(g, 0) + 10.0;
        C.dest(g, 0) = (double)(g + 1); // d(orig, dest) = g + 1
    }
    C.no_interact.set_size(n_ray);
    C.no_interact.fill(1u);
    C.prev_in = sv({5, 6, 5, 7, 5});
    C.cur_in = sv({7, 3, 7, 1, 7}); // FOG3 at g = 1, FOG1 at g = 3
    C.buf_in = sv({0, 2, 0, 2, 0});

    C.fbs_angle.set_size(n_rayN);
    C.fbs_angle.fill(QPI / 2.0);
    C.out_type.set_size(n_rayN);
    C.out_type.fill(2);
    C.m1 = sv({2, 2});
    C.m2 = sv({0, 0});
    C.ray_ind = {3u, 1u}; // out of order on purpose
    C.normals.zeros(n_rayN, 6);
    C.normals.col(0).fill(-1.0);
    C.normals.col(3).fill(1.0);
    C.prev_out = sv({11111, 11111});
    C.cur_out = sv({11111, 11111});
    C.buf_out = sv({11111, 11111});
    C.gain = {1.0, 1.0};
    C.xprmat.zeros(n_rayN, 8);
    C.xprmat(0, 0) = 1.0;
    C.xprmat(1, 0) = 1.0;

    // Snapshots for the read-only check
    arma::Col<short> p0 = C.prev_in, c0 = C.cur_in, b0 = C.buf_in;
    arma::mat o0 = C.orig, d0 = C.dest, f0 = C.fbs, s0 = C.sbs;

    C.run();

    // Both compact rows are buffered same-material i-o exits (spec 10.2 branch A):
    // replace MED(cur, d(orig,dest)) with the full-set state and geometry of ray g
    check_replace(C, medg(1, 4.0), TOL, 0); // i = 0 reads g = 3: cur = 1, d = 4
    check_replace(C, medg(3, 2.0), TOL, 1); // i = 1 reads g = 1: cur = 3, d = 2
    CHECK((int)C.prev_out(0) == 7);         // prev_in(3)
    CHECK((int)C.prev_out(1) == 6);         // prev_in(1)
    CHECK((int)C.cur_out(0) == 1);
    CHECK((int)C.cur_out(1) == 3);
    CHECK((int)C.buf_out(0) == 0);
    CHECK((int)C.buf_out(1) == 0);

    // Input arrays are read-only
    for (arma::uword g = 0; g < n_ray; ++g)
    {
        CHECK((int)C.prev_in(g) == (int)p0(g));
        CHECK((int)C.cur_in(g) == (int)c0(g));
        CHECK((int)C.buf_in(g) == (int)b0(g));
    }
    CHECK(arma::approx_equal(C.orig, o0, "absdiff", 0.0));
    CHECK(arma::approx_equal(C.dest, d0, "absdiff", 0.0));
    CHECK(arma::approx_equal(C.fbs, f0, "absdiff", 0.0));
    CHECK(arma::approx_equal(C.sbs, s0, "absdiff", 0.0));
}

// Merge single-ray harness objects into one identity-mapped batch (same pass settings)
static Rsu<double> merge_rows(const std::vector<Rsu<double>> &v)
{
    Rsu<double> B = v[0];
    for (size_t i = 1; i < v.size(); ++i)
    {
        const Rsu<double> &r = v[i];
        B.orig = arma::join_vert(B.orig, r.orig);
        B.dest = arma::join_vert(B.dest, r.dest);
        B.fbs = arma::join_vert(B.fbs, r.fbs);
        B.sbs = arma::join_vert(B.sbs, r.sbs);
        B.no_interact = arma::join_vert(B.no_interact, r.no_interact);
        B.fbs_angle = arma::join_vert(B.fbs_angle, r.fbs_angle);
        B.out_type = arma::join_vert(B.out_type, r.out_type);
        B.m1 = arma::join_vert(B.m1, r.m1);
        B.m2 = arma::join_vert(B.m2, r.m2);
        B.prev_in = arma::join_vert(B.prev_in, r.prev_in);
        B.cur_in = arma::join_vert(B.cur_in, r.cur_in);
        B.buf_in = arma::join_vert(B.buf_in, r.buf_in);
        B.normals = arma::join_vert(B.normals, r.normals);
        B.prev_out = arma::join_vert(B.prev_out, r.prev_out);
        B.cur_out = arma::join_vert(B.cur_out, r.cur_out);
        B.buf_out = arma::join_vert(B.buf_out, r.buf_out);
        B.gain = arma::join_vert(B.gain, r.gain);
        B.xprmat = arma::join_vert(B.xprmat, r.xprmat);
    }
    return B;
}

TEST_CASE("ray_state_update - gain consistency and batch determinism")
{
    // Test spec 5.7 and 5.8
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ); // NINE resonance, rho = 0.25 resolves at eps = 0.15

    // A mixed batch: clean entry, resolvable cavity exit, inconsistent kill, nested replace,
    // resolved pass-through, buffered i-i replace. All scalar transmission, eps = 0.15.
    auto rows = [&]()
    {
        std::vector<Rsu<double>> v;
        v.push_back(make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, 2, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, 4, 2, 1, 2, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, 1, 1, 2, 0, 0, enc(1, true), 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, 4, 2, 2, 3, 0, 1, 2, 1.0, 1.5, 0.5, QPI / 2.0, 0.15));
        return v;
    };

    SECTION("scalar mode keeps gainN equal to the xprmat power sum on every row")
    {
        auto v = rows();
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int c = 0; c < 8; ++c)
                p += (double)B.xprmat(i, c) * (double)B.xprmat(i, c);
            CHECK(std::abs((double)B.gain(i) - p) < TOL);
        }
    }
    SECTION("EM mode keeps gainN equal to half the xprmat power sum")
    {
        auto v = rows();
        for (auto &r : v) // same rows on the EM pass: probe HH too
        {
            r.itype = 1;
            r.xprmat(0, 6) = r.xprmat(0, 0);
            r.xprmat(0, 7) = r.xprmat(0, 1);
            r.gain(0) = 0.5 * (r.xprmat(0, 0) * r.xprmat(0, 0) + r.xprmat(0, 1) * r.xprmat(0, 1) +
                               r.xprmat(0, 6) * r.xprmat(0, 6) + r.xprmat(0, 7) * r.xprmat(0, 7));
        }
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int c = 0; c < 8; ++c)
                p += (double)B.xprmat(i, c) * (double)B.xprmat(i, c);
            CHECK(std::abs((double)B.gain(i) - 0.5 * p) < TOL);
        }
    }
    SECTION("a mixed batch equals the same rows run one at a time")
    {
        auto v = rows();
        auto B = merge_rows(v);
        B.run();
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i].run();
            CHECK(arma::approx_equal(B.xprmat.row((arma::uword)i), v[i].xprmat.row(0), "absdiff", 0.0));
            CHECK(std::abs((double)B.gain((arma::uword)i) - (double)v[i].gain(0)) == 0.0);
            CHECK((int)B.prev_out((arma::uword)i) == (int)v[i].prev_out(0));
            CHECK((int)B.cur_out((arma::uword)i) == (int)v[i].cur_out(0));
            CHECK((int)B.buf_out((arma::uword)i) == (int)v[i].buf_out(0));
        }
    }
}

TEST_CASE("ray_state_update - optional arguments")
{
    // Test spec 5.9-5.13
    static const auto mtl = base_palette<double>();

    SECTION("null state inputs behave exactly like explicit zero state")
    {
        auto A = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        auto B = A;
        B.has_prev_in = false;
        B.has_cur_in = false;
        B.has_buf_in = false;
        A.run();
        B.run();
        CHECK(arma::approx_equal(A.xprmat, B.xprmat, "absdiff", 0.0));
        CHECK(std::abs((double)A.gain(0) - (double)B.gain(0)) == 0.0);
        CHECK((int)A.prev_out(0) == (int)B.prev_out(0));
        CHECK((int)A.cur_out(0) == (int)B.cur_out(0));
        CHECK((int)A.buf_out(0) == (int)B.buf_out(0));
        check_state(A, 0, 1, 0);
    }
    SECTION("null state outputs skip the write but not the physics")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

        auto C = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_prev_out = false;
        C.run();
        check_mult(C, S, TOL);
        CHECK((int)C.cur_out(0) == 0);
        CHECK((int)C.buf_out(0) == 0);

        auto D = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        D.has_cur_out = false;
        D.has_buf_out = false;
        D.run();
        check_mult(D, S, TOL);
        CHECK((int)D.prev_out(0) == 0);
    }
    SECTION("gainN and xprmatN can each be omitted")
    {
        double g = medg(1, 2.0 - OFF);

        auto A = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_gain = false;
        A.run();
        check_mult(A, std::sqrt(g), TOL); // xprmat still scaled
        check_state(A, 0, 1, 0);

        auto B = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.has_xprmat = false;
        B.run();
        CHECK(std::abs((double)B.gain(0) - (double)B.g_in(0) * g) < TOL); // gain still scaled
        check_state(B, 0, 1, 0);

        auto C = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_gain = false;
        C.has_xprmat = false;
        CHECK_NOTHROW(C.run()); // pure state update
        check_state(C, 0, 1, 0);
    }
    SECTION("null normal_vecN disables the wedge test, never the gain math")
    {
        // Entry without normals cannot set the non-parallel flag
        auto A = make1<double>(mtl, 4, 1, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_normals = false;
        A.run();
        check_keep(A);
        check_state(A, 0, 4, 0);

        // A later exit still resolves with the full S factor
        double L = 2.0 * half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        auto B = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        B.has_normals = false;
        B.run();
        check_mult(B, S, TOL);
        check_state(B, 0, 0, 0);
    }
    SECTION("with every state pointer null the per-call physics still works")
    {
        auto strip = [](Rsu<double> &C)
        {
            C.has_prev_in = C.has_cur_in = C.has_buf_in = false;
            C.has_prev_out = C.has_cur_out = C.has_buf_out = false;
        };

        // (a) entry medium gain
        auto A = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(A);
        A.run();
        check_mult(A, std::sqrt(medg(1, 2.0 - OFF)), TOL);

        // (b) TR kill
        auto B = make1<double>(mtl, 2, 3, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(B);
        B.run();
        check_kill(B);

        // (c) single-call air-gap S: i-o-i with cur = 0, distinct hits. With
        // d(orig,fbs) = d(fbs,sbs) = G there is no ambiguity in the cavity length.
        double G = half_wave(1.0, FRQ); // air-gap resonance, m = 1
        auto C = make1<double>(mtl, 4, 2, 2, 4, 4, 0, 0, 0, G, G + 1.0, G, QPI / 2.0, 0.0);
        strip(C);
        C.run();
        std::complex<double> r = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0).r; // air side, both faces
        std::complex<double> S = airy_S(r, r, phi_one_way(1.0, 1.0, FRQ, G));
        check_mult(C, S, TOL);

        // (d) a separate exit call sees cur = 0 and passes through, no cross-call S
        auto D = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 0, 0, G, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(D);
        D.run();
        check_keep(D);
    }
}

TEST_CASE("ray_state_update - gain operation semantics and probe handling")
{
    // Test spec 5.14-5.17, design spec section 5
    static const auto mtl = base_palette<double>();

    SECTION("IG keep is bit-identical, including a fully populated EM probe")
    {
        auto C = make1<double>(mtl, 1, 1, 1, 2, 0, 0, enc(1, true), 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        C.xprmat(0, 0) = 0.31;
        C.xprmat(0, 1) = -0.12;
        C.xprmat(0, 2) = 0.05;
        C.xprmat(0, 3) = 0.21;
        C.xprmat(0, 4) = -0.17;
        C.xprmat(0, 5) = 0.02;
        C.xprmat(0, 6) = 0.44;
        C.xprmat(0, 7) = 0.09;
        double p = 0.0;
        for (int c = 0; c < 8; ++c)
            p += C.xprmat(0, c) * C.xprmat(0, c);
        C.gain(0) = 0.5 * p;
        C.run();
        check_keep(C);
        check_state(C, 0, enc(1, true), 0);
    }
    SECTION("IG * S multiplies every Jones pair, off-diagonals included")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

        auto C = make1<double>(mtl, 1, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.xprmat(0, 0) = 0.31;
        C.xprmat(0, 1) = -0.12;
        C.xprmat(0, 2) = 0.05;
        C.xprmat(0, 3) = 0.21;
        C.xprmat(0, 4) = -0.17;
        C.xprmat(0, 5) = 0.02;
        C.xprmat(0, 6) = 0.44;
        C.xprmat(0, 7) = 0.09;
        double p = 0.0;
        for (int c = 0; c < 8; ++c)
            p += C.xprmat(0, c) * C.xprmat(0, c);
        C.gain(0) = 0.5 * p;
        C.run();
        check_mult(C, S, TOL);
        check_state(C, 0, 0, 0); // transmission exit clears cur (spec 10.2)
    }
    SECTION("replace ignores the incoming field entirely")
    {
        auto A = make1<double>(mtl, 4, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3));
        auto B = make1<double>(mtl, 4, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(-0.1, 0.8));
        A.run();
        B.run();
        CHECK(arma::approx_equal(A.xprmat, B.xprmat, "absdiff", 0.0));
        CHECK(std::abs((double)A.gain(0) - (double)B.gain(0)) == 0.0);
        check_replace(A, medg(2, 3.0), TOL);
    }
    SECTION("EM replace writes sqrt(g) on VV and HH, scalar replace on VV only")
    {
        auto A = make1<double>(mtl, 1, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.run();
        check_replace(A, medg(2, 3.0), TOL); // checks the VV + HH layout for itype < 3

        auto B = make1<double>(mtl, 4, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.run();
        check_replace(B, medg(2, 3.0), TOL); // checks the VV-only layout for itype >= 3
        double p = 0.0;
        for (int c = 0; c < 8; ++c)
            p += B.xprmat(0, c) * B.xprmat(0, c);
        CHECK(std::abs((double)B.gain(0) - p) < TOL); // scalar gain has no 0.5 factor
    }
    SECTION("gainN-only callers still receive the |S|^2 power factor")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

        auto C = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_xprmat = false;
        C.run();
        CHECK(std::abs((double)C.gain(0) - (double)C.g_in(0) * std::norm(S)) < TOL);
        check_state(C, 0, 0, 0);
    }
}

TEST_CASE("ray_state_update - Airy factor against the closed form")
{
    // Test spec 6.1, design spec 9.1: complex S on a cavity exit, eps = 0 (always resolve)
    static const auto mtl = base_palette<double>();
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r; // DENSE inside face, 1/3

    SECTION("resonance, antiresonance, and a generic length at 1 GHz")
    {
        double half = half_wave(2.0, FRQ);
        for (double L : {half, 1.5 * half, 0.2718})
        {
            auto C = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
            C.run();
            std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
            check_mult(C, S, TOL);
            check_state(C, 0, 0, 0);
        }

        // Closed-form values at the extremes: 1/(1 - 1/9) and 1/(1 + 1/9)
        std::complex<double> S_res = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, half));
        std::complex<double> S_null = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, 1.5 * half));
        CHECK(std::abs(S_res - 1.125) < TOL);
        CHECK(std::abs(S_null - 0.9) < TOL);
    }
    SECTION("the phase tracks the carrier frequency")
    {
        double f2 = 2.4e9, L = 0.2718;
        auto C = make1<double>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                               std::complex<double>(0.5, 0.3), f2);
        C.run();
        check_mult(C, airy_S(r, r, phi_one_way(2.0, 1.0, f2, L)), TOL);
    }
    SECTION("an impedance-matched medium has no cavity: S = 1")
    {
        // eps = mu = 4: the interface admittance matches air, r = 0, so the Airy sum is 1.
        // Anchored to the mu-aware reference Fresnel.
        CHECK(calc_transition_gain_mu(0, 0.0, 0.0, 0.0, {4.0, 0.0}, {1.0, 0.0}, {4.0, 0.0}, {1.0, 0.0}) < 1e-12);

        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> matched;
        mtl_matrix_to_map<double>(M, ind, matched);
        set_mu<double>(matched, 4.0, 0.0, 0.0, 0.0);

        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = make1<double>(matched, 4, 2, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.run();
        check_mult(C, 1.0, TOL);
        check_state(C, 0, 0, 0);
    }
}

// Stage the five canonical events of one air | slab | air crossing and return the call
// harnesses. The test owns the in-slab one-way phase; the function owns magnitudes.
// theta_n is the OUTSIDE angle from the normal; internal events are fed the refracted
// internal angle. All events share the slab material `mat` and one-way path L (= L_eff).
struct SlabEvents
{
    Rsu<double> entry, exitt, front, internal_r, outcouple;
};
static SlabEvents stage_slab(const std::unordered_map<std::string, std::vector<double>> &mtl,
                             int mat, double n, double L, double theta_n, double eps, bool em,
                             std::complex<double> t_in, std::complex<double> t_out,
                             std::complex<double> r_front, std::complex<double> r_back,
                             std::complex<double> t_oc, int prev = 0)
{
    double theta_i = std::asin(std::sin(theta_n) / n); // internal angle from the normal
    double g_out = QPI / 2.0 - theta_n;                // grazing convention of fbs_angleN
    double g_in = QPI / 2.0 - theta_i;
    int itT = em ? 1 : 4, itR = em ? 0 : 3;

    SlabEvents E;
    // Entry o-i: dest at the far face, so the entry MED runs over L - off + off = L... the
    // clamp distance is d(fbs,dest) - off; place dest so that distance equals L.
    E.entry = make1<double>(mtl, itT, 1, 1, mat, 0, 0, 0, 0, 0.5, L + OFF, 10.0, g_out, eps, t_in);
    // Exit i-o: one-way path d(orig,fbs) = L
    E.exitt = make1<double>(mtl, itT, 2, 1, mat, 0, prev, mat, 0, L, 2.0, 0.5, g_in, eps, t_out);
    // Front reflection, order 0, outside
    E.front = make1<double>(mtl, itR, 1, 1, mat, 0, 0, 0, 0, 0.5, 2.0, 10.0, g_out, eps, r_front);
    // Internal back reflection, same segment geometry as the exit
    E.internal_r = make1<double>(mtl, itR, 2, 1, mat, 0, prev, mat, 0, L, 2.0, 0.5, g_in, eps, r_back);
    // Resolved outcouple through the front face
    E.outcouple = make1<double>(mtl, itT, 2, 1, mat, 0, prev, enc(mat, true), 0, L, 2.0, 0.5, g_in, eps, t_oc);

    E.entry.run();
    E.exitt.run();
    E.front.run();
    E.internal_r.run();
    E.outcouple.run();
    return E;
}

TEST_CASE("ray_state_update - energy conservation of a lossless slab")
{
    // Test spec 6.2, design spec 9.2 and 13: |R|^2 + |T|^2 = 1 with the test owning the
    // in-slab one-way phase and the function owning every magnitude.
    static const auto mtl = base_palette<double>();
    const int mat = 4; // DENSE, eta = 4, lossless
    const double L = 0.275;

    auto run_ledger = [&](double theta_n, bool em)
    {
        double n = n_of(mat);
        double theta_i = std::asin(std::sin(theta_n) / n);
        FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, theta_n); // outside face
        FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i); // inside face

        if (theta_n == 0.0) // anchor the reference Fresnel to the established oracle
        {
            CHECK(std::abs(std::norm(out.r) - calc_transition_gain(0, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);
            CHECK(std::abs(std::norm(out.t) - calc_transition_gain(1, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);
        }

        auto E = stage_slab(mtl, mat, n, L, theta_n, 0.0, em, out.t, ins.t, out.r, ins.r, ins.t);

        // Expected per-call factors
        std::complex<double> phi = phi_one_way(n, 1.0, FRQ, L); // unit magnitude here
        std::complex<double> S = airy_S(ins.r, ins.r, phi);
        check_mult(E.entry, 1.0, TOL); // lossless MED
        check_mult(E.exitt, S, TOL);
        check_mult(E.front, 1.0, TOL);
        check_mult(E.internal_r, S, TOL); // same S on both passes (test spec 7.1)
        check_mult(E.outcouple, 1.0, TOL);
        check_state(E.exitt, 0, 0, 0);
        check_state(E.internal_r, 0, enc(mat, true), 0);
        check_state(E.outcouple, 0, 0, 0);

        // Ledger: assemble the two ports from the measured factors plus the test-owned phase
        std::complex<double> fE = vvf(E.entry), fX = vvf(E.exitt), fF = vvf(E.front);
        std::complex<double> fI = vvf(E.internal_r), fO = vvf(E.outcouple);
        std::complex<double> T = out.t * fE * phi * ins.t * fX;
        std::complex<double> B = out.t * fE * phi * (ins.r * fI) * phi * ins.t * fO;
        std::complex<double> R = out.r * fF + B;
        CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < TOL);
    };

    SECTION("scalar, normal incidence") { run_ledger(0.0, false); }
    SECTION("scalar, 30 degrees") { run_ledger(30.0 * QPI / 180.0, false); }
    SECTION("scalar, 55 degrees") { run_ledger(55.0 * QPI / 180.0, false); }
    SECTION("EM, normal incidence") { run_ledger(0.0, true); }
}

TEST_CASE("ray_state_update - energy safety of a lossy slab")
{
    // Test spec 6.3, design spec 9.1 and 13: ABSORB (eta = 4, alpha = 30 dB/m) driven
    // hard. The measured ports must match the oracle ports; in particular the reflection
    // port catches a missing up-trip medium loss.
    static const auto mtl = base_palette<double>();
    const int mat = 6;
    const double L = 2.0 * half_wave(2.0, FRQ); // resonance phase
    const double g = medg(mat, L);              // one-way medium gain

    FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0);
    FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0);

    auto E = stage_slab(mtl, mat, 2.0, L, 0.0, 0.0, false, out.t, ins.t, out.r, ins.r, ins.t);

    std::complex<double> phi = phi_one_way(2.0, g, FRQ, L);
    std::complex<double> S = airy_S(ins.r, ins.r, phi);

    // The entry owns the down-trip medium loss
    check_mult(E.entry, std::sqrt(g), TOL);
    check_mult(E.exitt, S, TOL);
    check_mult(E.front, 1.0, TOL);

    // Oracle ports of the slab (phi carries the medium loss)
    double T_or = std::norm(out.t * phi * ins.t * S);
    double R_or = std::norm(out.r + out.t * ins.t * ins.r * phi * phi * S);
    double A_or = 1.0 - T_or - R_or;
    CHECK(A_or > 0.3); // the slab is driven hard enough for the test to mean something

    // Measured ports: the test contributes only unit-magnitude phases per one-way trip
    std::complex<double> ph = std::exp(std::complex<double>(0.0, -2.0 * QPI * FRQ / C0 * 2.0 * L));
    std::complex<double> fE = vvf(E.entry), fX = vvf(E.exitt), fF = vvf(E.front);
    std::complex<double> fI = vvf(E.internal_r), fO = vvf(E.outcouple);
    std::complex<double> T_amp = out.t * fE * ph * ins.t * fX;
    std::complex<double> B_amp = out.t * fE * ph * (ins.r * fI) * ph * ins.t * fO;
    double T_meas = std::norm(T_amp);
    double R_meas = std::norm(out.r * fF + B_amp);

    CHECK(std::abs(T_meas - T_or) < TOL);
    CHECK(std::abs(R_meas - R_or) < TOL); // fails if the up trip drops its medium loss
    CHECK(std::abs(R_meas + T_meas + A_or - 1.0) < TOL);
}

TEST_CASE("ray_state_update - mass-law material feeds |phi| from medium_gain")
{
    // Test spec 6.4, design spec 9.1: for a mass-law slab, |phi|^2 must equal the
    // implementation's own medium_gain(L), not an eta-only loss (eta is real here, so an
    // eta-only |phi| would be 1 and S would over-resolve).
    //
    // The mass column has no public closed form in the specs, so the test measures
    // medium_gain through the nested-entry replace row (pure MED over a known distance)
    // and then checks S against the closed form with the measured |phi|.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl[MTL_KEY_MASS] = {10.0}; // assumed mass-law key, see file header

    // The mass law engages only above its clamp region (fGHz * path > 1), so run at
    // 10 GHz with about 0.3 m of path, still on a resonance phase (n = 2)
    const double F10 = 10.0e9;
    const double L = 40.0 * half_wave(2.0, F10);

    // Measure medium_gain over L and 2L with the replace row (spec 10.1 branch A nested):
    // gain out = MED(cur, d(orig, dest))
    auto med_of = [&](double d)
    {
        // nested-entry geometry with d(orig, dest) = d
        auto C = make1<double>(mtl, 4, 1, 1, 1, 0, 0, 1, 0, 0.4 * d, 0.6 * d, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3), F10);
        C.run();
        return (double)C.gain(0);
    };
    double g_L = med_of(L);
    double g_2L = med_of(2.0 * L);

    // Engagement tripwire: if this fails, the assumed key MTL_KEY_MASS is wrong or the
    // mass column is ignored, and the rest of the test is vacuous.
    CHECK(g_L < 0.999);

    // The mass law is logarithmic in distance, so the loss is not multiplicative
    CHECK_NOFAIL(g_2L > g_L * g_L + 1e-9);

    // Cavity exit with eps = 0: S must use the measured medium gain
    auto C = make1<double>(mtl, 4, 2, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                           std::complex<double>(0.5, 0.3), F10);
    C.run();
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S_right = airy_S(r, r, phi_one_way(2.0, g_L, F10, L));
    std::complex<double> S_wrong = airy_S(r, r, phi_one_way(2.0, 1.0, F10, L)); // eta-only
    std::complex<double> S_func = vvf(C);

    CHECK(std::abs(S_func - S_right) < 1e-6);
    if (g_L < 0.99) // only meaningful when the two models are well separated
        CHECK(std::abs(S_func - S_right) < std::abs(S_func - S_wrong));
}

TEST_CASE("ray_state_update - survival gate thresholds on eps")
{
    // Test spec 6.5, design spec 9.4: rho = sqrt(R_near * R_far * medium_gain(2L)).
    // NINE is lossless with inside-face r = 1/2, so rho = 0.25 exactly.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ);
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r;
    CHECK(std::abs(gate_rho(r, r, 1.0) - 0.25) < TOL); // oracle self-check
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));
    CHECK(std::abs(S - 4.0 / 3.0) < TOL); // resonance closed form

    auto run_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, 2, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps clearly below rho resolves")
    {
        auto C = run_at(0.225);
        check_mult(C, S, TOL);
        check_state(C, 0, 0, 0);
    }
    SECTION("eps clearly above rho re-emits with the gain untouched")
    {
        auto C = run_at(0.275);
        check_keep(C);
        check_state(C, 0, 0, 0); // the exit row still clears cur
    }
    SECTION("eps = 0 resolves even a weak near-lossless cavity")
    {
        auto C = run_at(0.0);
        check_mult(C, S, TOL);

        // WEAK, eta = 1.21: rho is about 0.0023, still resolved at eps = 0
        double Lw = 2.0 * half_wave(n_of(7), FRQ);
        std::complex<double> rw = fresnel_te({1.21, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> Sw = airy_S(rw, rw, phi_one_way(n_of(7), 1.0, FRQ, Lw));
        auto D = make1<double>(mtl, 4, 2, 1, 7, 0, 0, 7, 0, Lw, 2.0, 0.5, QPI / 2.0, 0.0);
        D.run();
        check_mult(D, Sw, TOL);
    }
    SECTION("eps >= 1 never resolves")
    {
        for (double eps : {1.0, 1.5})
        {
            auto C = run_at(eps);
            check_keep(C);
            check_state(C, 0, 0, 0);
        }
    }
}

TEST_CASE("ray_state_update - parallelism is a magnitude test")
{
    // Test spec 6.6, design spec 9.3: two staged calls, prev_out of the entry feeds
    // prev_in of the exit. Antiparallel and parallel same-orientation normals both count
    // as a slab; only a genuine wedge sets the flag and blocks the resolve.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

    auto entry_with = [&](double nx2, double ny2)
    {
        auto C = make1<double>(mtl, 4, 1, 2, 4, 4, 0, 0, 0, 1.0, 2.0, L, QPI / 2.0, 0.0);
        C.normals(0, 0) = -1.0;
        C.normals(0, 1) = 0.0;
        C.normals(0, 2) = 0.0;
        C.normals(0, 3) = nx2;
        C.normals(0, 4) = ny2;
        C.normals(0, 5) = 0.0;
        C.run();
        return C;
    };
    auto exit_with = [&](short prev_word)
    {
        auto C = make1<double>(mtl, 4, 2, 1, 4, 0, prev_word, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.run();
        return C;
    };

    SECTION("antiparallel pair: flag clear, exit resolves")
    {
        auto A = entry_with(1.0, 0.0);
        CHECK((int)A.prev_out(0) == 0);
        auto B = exit_with(A.prev_out(0));
        check_mult(B, S, TOL);
    }
    SECTION("parallel same-orientation pair: flag clear, exit resolves")
    {
        auto A = entry_with(-1.0, 0.0);
        CHECK((int)A.prev_out(0) == 0);
        auto B = exit_with(A.prev_out(0));
        check_mult(B, S, TOL);
    }
    SECTION("genuine wedge: flag set, exit re-emits")
    {
        auto A = entry_with(0.5, std::sqrt(3.0) / 2.0); // |dot| = 0.5
        CHECK((int)A.prev_out(0) == (int)(short)0x8000);
        auto B = exit_with(A.prev_out(0));
        check_keep(B);
        CHECK((int)B.cur_out(0) == 0); // the exit row still clears cur
        CHECK((int)B.buf_out(0) == 0);
    }
}

TEST_CASE("ray_state_update - transmission factor folds into the resolved gain")
{
    // Test spec 6.10, design spec 9.5: with tf on the slab material, S and the interface
    // ports must use the same tf-effective coefficient set, and Stokes consistency
    // (t12 * t21 = 1 - r_eff^2) must hold so the lossless ledger still closes.
    //
    // The tf model has no public closed form in the specs, so the test extracts the
    // effective round-trip reflectance from the function's own S at resonance and then
    // demands self-consistency of the full ledger with that extracted value.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl[MTL_KEY_TF] = {0.6}; // assumed transmission-factor keys, see file header
    mtl[MTL_KEY_TFB] = {0.0};

    const double L = 2.0 * half_wave(2.0, FRQ); // resonance: phi^2 = 1

    // Extract r_eff^2 = 1 - 1/S from the cavity exit
    auto C = make1<double>(mtl, 4, 2, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
    C.run();
    std::complex<double> S_func = vvf(C);
    std::complex<double> x = 1.0 - 1.0 / S_func; // r_near * r_far * phi^2, phi^2 = 1

    CHECK(std::abs(x.imag()) < 1e-6);
    CHECK(x.real() > 0.0);
    CHECK(x.real() < 1.0);
    // Engagement tripwire: with tf = 0.6 the effective reflectance must differ from the
    // bare Fresnel value 1/9. If this fails, the assumed MTL_KEY_TF key is wrong.
    CHECK(std::abs(x.real() - 1.0 / 9.0) > 1e-3);

    // Self-consistent lossless ledger with the extracted tf-effective coefficients
    double r_eff = std::sqrt(x.real());
    double t_eff = std::sqrt(1.0 - x.real()); // Stokes: t12 * t21 = 1 - r^2
    auto E = stage_slab(mtl, 1, 2.0, L, 0.0, 0.0, false, t_eff, t_eff, -r_eff, r_eff, t_eff);

    std::complex<double> phi = phi_one_way(2.0, 1.0, FRQ, L);
    std::complex<double> fE = vvf(E.entry), fX = vvf(E.exitt), fF = vvf(E.front);
    std::complex<double> fI = vvf(E.internal_r), fO = vvf(E.outcouple);
    CHECK(std::abs(fX - fI) < TOL); // same tf-effective S on both passes
    std::complex<double> T = t_eff * fE * phi * t_eff * fX;
    std::complex<double> B = t_eff * fE * phi * (r_eff * fI) * phi * t_eff * fO;
    std::complex<double> R = -r_eff * fF + B;
    CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < TOL);
}

TEST_CASE("ray_state_update - resolve clamp near the Airy pole")
{
    // Test spec 6.11, design spec 9.7: lossless slab near grazing drives r^2 toward 1.
    // At resonance with |1 - r_near*r_far*phi^2| < 1e-2 the resolve re-emits; detuned or
    // at moderate angles it resolves with a large but finite, oracle-exact S.
    static const auto mtl = base_palette<double>();
    const int mat = 4;
    const double n = 2.0;

    auto exit_at = [&](double theta_n, double L)
    {
        double theta_i = std::asin(std::sin(theta_n) / n);
        auto C = make1<double>(mtl, 4, 2, 1, mat, 0, 0, mat, 0, L, 2.0, 0.5, QPI / 2.0 - theta_i, 0.0);
        C.run();
        return C;
    };

    SECTION("at the pole the resolve is refused")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ); // phi^2 = 1
        CHECK(std::abs(1.0 - r * r) < 1e-2); // the configuration is inside the clamp

        auto C = exit_at(theta_n, L);
        check_keep(C);
        check_state(C, 0, 0, 0);
    }
    SECTION("a quarter-wave detune leaves the clamp and resolves")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.5 * half_wave(n, FRQ); // phi^2 = -1: |1 - r*r*phi^2| is about 2

        auto C = exit_at(theta_n, L);
        std::complex<double> S = airy_S(r, r, phi_one_way(n, 1.0, FRQ, L));
        CHECK(std::abs(S) < 1.0); // antiresonant suppression
        check_mult(C, S, TOL);
    }
    SECTION("a strong but sub-clamp resonance resolves finite and bounded")
    {
        double theta_n = 88.0 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ);
        CHECK(std::abs(1.0 - r * r) > 1e-2); // outside the clamp

        auto C = exit_at(theta_n, L);
        std::complex<double> S = airy_S(r, r, phi_one_way(n, 1.0, FRQ, L));
        check_mult(C, S, TOL);
        CHECK(std::abs(S) < 105.0); // bounded by the clamp design margin
        for (int c = 0; c < 8; ++c)
            CHECK(std::isfinite((double)C.xprmat(0, c)));
        CHECK(std::isfinite((double)C.gain(0)));
    }
}

TEST_CASE("ray_state_update - cross-pass invariance of the resolve decision")
{
    // Test spec 7.1, design spec 9.8: the transmission-pass exit and the reflection-pass
    // internal bounce of the same slab segment must make the same resolve-vs-re-emit
    // decision and apply the same S.
    static const auto mtl = base_palette<double>();
    const double L = 0.123; // generic, off resonance
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r; // NINE, rho = 0.25
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));

    auto exit_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, 2, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };
    auto refl_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 3, 2, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps below rho: both passes resolve with the identical S")
    {
        auto T = exit_at(0.2);
        auto R = refl_at(0.2);
        check_mult(T, S, TOL);
        check_mult(R, S, TOL);
        CHECK(std::abs(vvf(T) - vvf(R)) < TOL);
        check_state(T, 0, 0, 0);
        check_state(R, 0, enc(5, true), 0);
    }
    SECTION("eps above rho: both passes re-emit")
    {
        auto T = exit_at(0.3);
        auto R = refl_at(0.3);
        check_keep(T);
        check_keep(R);
        check_state(T, 0, 0, 0);
        check_state(R, 0, 5, 0); // re-emitted, no resolved flag
    }
}

TEST_CASE("ray_state_update - resolved flag persists across internal crossings")
{
    // Test spec 7.2, design spec 10.0: stage current_out into current_in. The flag stays
    // set through an internal i-i crossing and clears only on the exit to air.
    static const auto mtl = base_palette<double>();

    auto A = disp(4, 2, 2, 3, 0, enc(1, true), 0); // resolved i-i crossing
    check_mult(A, std::sqrt(medg(1, 1.0)), TOL);   // charges its incoming segment in FOG1
    CHECK((int)A.cur_out(0) == (int)enc(2, true));
    CHECK((int)A.prev_out(0) == 1);

    auto B = make1<double>(mtl, 4, 2, 1, 2, 0, 1, A.cur_out(0), 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
    B.run();                                     // resolved exit to air
    check_mult(B, std::sqrt(medg(2, 1.0)), TOL); // up-trip loss in FOG2
    check_state(B, 1, 0, 0);
}

TEST_CASE("ray_state_update - stacked slabs do not create energy")
{
    // Test spec 7.3, design spec section 3 (non-goals) and 13: air | A (DENSE) | B (NINE)
    // | air at normal incidence, staged with the real state chain (each call's state
    // outputs feed the next call) and eps = 0. The stack is resolved with first-order
    // inter-slab coupling; the promise under test is energy safety, R + T <= 1 (lossless,
    // so A = 0), never above.
    //
    // The specs leave one point open: whether the M2M cavity transition marks the ray
    // RESOLVED (which makes the second cavity transparent, a strict under-count) or not
    // (both cavities resolve). The test accepts either per-call behavior and asserts the
    // energy bound on whatever the function actually produced.
    static const auto mtl = base_palette<double>();
    const double LA = 0.137, LB = 0.093;

    FresnelTE f0A = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0); // air -> A
    FresnelTE fA0 = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0); // A -> air
    FresnelTE fAB = fresnel_te({4.0, 0.0}, {9.0, 0.0}, 0.0); // A -> B
    FresnelTE fBA = fresnel_te({9.0, 0.0}, {4.0, 0.0}, 0.0); // B -> A
    FresnelTE fB0 = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0); // B -> air

    std::complex<double> phiA = phi_one_way(2.0, 1.0, FRQ, LA);
    std::complex<double> phiB = phi_one_way(3.0, 1.0, FRQ, LB);
    std::complex<double> SA = airy_S(fAB.r, fA0.r, phiA); // cavity A at the A|B face, far = air
    std::complex<double> SB = airy_S(fB0.r, fBA.r, phiB); // cavity B at the far face, far = A

    auto ev = [&](int it, int ot, unsigned nH, int M1, int M2, int prev, int cur,
                  double dof, std::complex<double> feed)
    {
        auto C = make1<double>(mtl, it, ot, nH, M1, M2, prev, cur, 0, dof, 0.3, 0.5, QPI / 2.0, 0.0, feed);
        C.run();
        return C;
    };

    // Transmission chain: entry into A, M2M crossing into B, exit from B
    auto E1 = ev(4, 1, 1, 4, 0, 0, 0, 0.5, f0A.t);
    check_mult(E1, 1.0, TOL);
    check_state(E1, 0, 4, 0);

    auto E2 = ev(4, 4, 2, 5, 4, E1.prev_out(0), E1.cur_out(0), LA, fAB.t); // IG * S_A * MED(NINE)
    check_mult(E2, SA, TOL);
    CHECK(mat_of(E2.cur_out(0)) == 5); // flagged or not, see above
    CHECK((int)E2.prev_out(0) == 4);
    bool resolved_after_m2m = flag_of(E2.cur_out(0));

    auto E3 = ev(4, 2, 1, 5, 0, E2.prev_out(0), E2.cur_out(0), LB, fB0.t);
    std::complex<double> fX = vvf(E3); // S_B, or 1 on a transparent resolved exit
    CHECK(std::min(std::abs(fX - SB), std::abs(fX - 1.0)) < TOL);
    if (resolved_after_m2m)
        CHECK(std::abs(fX - 1.0) < TOL);
    CHECK((int)E3.cur_out(0) == 0);

    // Reflection branch 0: bare front reflection
    auto B0 = ev(3, 1, 1, 4, 0, 0, 0, 0.5, f0A.r);
    check_mult(B0, 1.0, TOL);

    // Reflection branch A: internal bounce at the A|B face, then resolved outcouple
    auto BA1 = ev(3, 4, 2, 5, 4, E1.prev_out(0), E1.cur_out(0), LA, fAB.r);
    check_mult(BA1, SA, TOL);
    check_state(BA1, 0, enc(4, true), 0);
    auto BA2 = ev(4, 2, 1, 4, 0, BA1.prev_out(0), BA1.cur_out(0), LA, fA0.t);
    check_keep(BA2);
    CHECK((int)BA2.cur_out(0) == 0);

    // Reflection branch B: internal bounce at the B|air face. If the M2M marked the ray
    // resolved, the reflection pass kills it (spec 10.0) and the branch carries nothing;
    // otherwise it resolves with S_B and outcouples transparently through A.
    auto BB1 = ev(3, 2, 1, 5, 0, E2.prev_out(0), E2.cur_out(0), LB, fB0.r);
    std::complex<double> bB = 0.0;
    if ((double)BB1.gain(0) == 0.0) // killed
    {
        CHECK(resolved_after_m2m); // a kill is only sanctioned for a resolved ray
    }
    else
    {
        check_mult(BB1, SB, TOL);
        CHECK((int)BB1.cur_out(0) == (int)enc(5, true));
        auto BB2 = ev(4, 4, 2, 4, 5, BB1.prev_out(0), BB1.cur_out(0), LB, fBA.t);
        check_keep(BB2);
        CHECK((int)BB2.cur_out(0) == (int)enc(4, true));
        CHECK((int)BB2.prev_out(0) == 5);
        auto BB3 = ev(4, 2, 1, 4, 0, BB2.prev_out(0), BB2.cur_out(0), LA, fA0.t);
        check_keep(BB3);
        CHECK((int)BB3.cur_out(0) == 0);
        bB = f0A.t * vvf(E1) * phiA * fAB.t * vvf(E2) * phiB * (fB0.r * vvf(BB1)) * phiB *
             fBA.t * vvf(BB2) * phiA * fA0.t * vvf(BB3);
    }

    // Assemble the ports from the function's own factors; the test owns one unit phase
    // per one-way trip
    std::complex<double> T = f0A.t * vvf(E1) * phiA * fAB.t * vvf(E2) * phiB * fB0.t * vvf(E3);
    std::complex<double> bA = f0A.t * vvf(E1) * phiA * (fAB.r * vvf(BA1)) * phiA * fA0.t * vvf(BA2);
    double RT = std::norm(f0A.r * vvf(B0) + bA + bB) + std::norm(T);
    CHECK(RT <= 1.0 + 1e-9); // never energy creation
    CHECK(RT > 0.5);         // and the staging is actually engaged
}

// Type parity: the same scenario set in a second precision. Tolerances per the test
// spec: 1e-9 for double, 1e-5 for float.
template <typename dtype>
static void run_parity(double tol)
{
    const auto mtl = base_palette<dtype>();
    double g2 = QPI / 2.0;

    // Entry medium gain (spec 10.1 branch A)
    {
        auto C = make1<dtype>(mtl, 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, g2, 1.5);
        C.run();
        check_mult(C, std::sqrt(medg(1, 2.0 - OFF)), tol);
        check_state(C, 0, 1, 0);
    }
    // TR kill on the refraction pass
    {
        auto C = make1<dtype>(mtl, 2, 3, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, g2, 1.5);
        C.run();
        check_kill(C);
        check_state(C, 0, 1, 0);
    }
    // Airy factor at the m = 1 resonance
    {
        double L = half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        auto C = make1<dtype>(mtl, 4, 2, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, g2, 0.0);
        C.run();
        check_mult(C, S, tol);
        check_state(C, 0, 0, 0);
    }
    // Replace layouts in both modes (spec 10.1 branch A nested)
    {
        auto A = make1<dtype>(mtl, 1, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, g2, 1.5);
        A.run();
        check_replace(A, medg(2, 3.0), tol);
        auto B = make1<dtype>(mtl, 4, 1, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, g2, 1.5);
        B.run();
        check_replace(B, medg(2, 3.0), tol);
    }
    // Lossless energy closure at normal incidence
    {
        double L = 0.275;
        FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0);
        FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0);
        std::complex<double> phi = phi_one_way(2.0, 1.0, FRQ, L);

        auto ev = [&](int it, int ot, int cur, double dof, double dfd, std::complex<double> feed)
        {
            auto C = make1<dtype>(mtl, it, ot, 1, 4, 0, 0, cur, 0, dof, dfd, 0.5, g2, 0.0, feed);
            C.run();
            return C;
        };
        auto entry = ev(4, 1, 0, 0.5, L + OFF, out.t);
        auto exitt = ev(4, 2, 4, L, 2.0, ins.t);
        auto front = ev(3, 1, 0, 0.5, 2.0, out.r);
        auto internal_r = ev(3, 2, 4, L, 2.0, ins.r);
        auto outcouple = ev(4, 2, enc(4, true), L, 2.0, ins.t);

        std::complex<double> T = out.t * vvf(entry) * phi * ins.t * vvf(exitt);
        std::complex<double> B = out.t * vvf(entry) * phi * (ins.r * vvf(internal_r)) * phi * ins.t * vvf(outcouple);
        std::complex<double> R = out.r * vvf(front) + B;
        CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < tol);
    }
}

TEST_CASE("ray_state_update - type parity between double and float")
{
    // Test spec group 8
    SECTION("double") { run_parity<double>(1.0e-9); }
    SECTION("float") { run_parity<float>(1.0e-5); }
}

// Valid baseline call for the validation group: n_ray = 2 (full set), n_rayN = 1
// (compact set), ray_ind = {1}, a clean o-i entry into FOG1
static Rsu<double> make_valid9()
{
    auto C = make1<double>(base_palette<double>(), 4, 1, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 10.0, QPI / 2.0, 0.15);
    C.orig = arma::join_vert(C.orig, C.orig);
    C.dest = arma::join_vert(C.dest, C.dest);
    C.fbs = arma::join_vert(C.fbs, C.fbs);
    C.sbs = arma::join_vert(C.sbs, C.sbs);
    C.no_interact = arma::join_vert(C.no_interact, C.no_interact);
    C.prev_in = arma::join_vert(C.prev_in, C.prev_in);
    C.cur_in = arma::join_vert(C.cur_in, C.cur_in);
    C.buf_in = arma::join_vert(C.buf_in, C.buf_in);
    C.ray_ind.set_size(1);
    C.ray_ind(0) = 1;
    C.has_ray_ind = true;
    return C;
}

TEST_CASE("ray_state_update - input validation")
{
    // Test spec group 9: every malformed input throws std::invalid_argument; the paired
    // well-formed call succeeds. No exception-message matching (blind suite).

    { // the baseline itself must run in every section
        auto C = make_valid9();
        CHECK_NOTHROW(C.run());
        check_state(C, 0, 1, 0);
    }

    SECTION("interaction type range") // 9.1
    {
        for (int it : {5, -1, 99})
        {
            auto C = make_valid9();
            C.itype = it;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (int it : {0, 1, 2, 3, 4})
        {
            auto C = make_valid9();
            C.itype = it;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("center frequency must be positive and finite") // 9.2
    {
        for (double f : {0.0, -1.0e9, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid9();
            C.freq = f;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
    }
    SECTION("eps must be finite and non-negative") // 9.3
    {
        for (double e : {-0.1, (double)std::numeric_limits<double>::quiet_NaN(),
                         (double)std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid9();
            C.eps = e;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (double e : {0.0, 0.5, 1.0, 2.0})
        {
            auto C = make_valid9();
            C.eps = e;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("geometry arrays need exactly three columns") // 9.4
    {
        for (int which = 0; which < 4; ++which)
            for (arma::uword nc : {2u, 4u})
            {
                auto C = make_valid9();
                arma::mat *A = which == 0 ? &C.orig : which == 1 ? &C.dest
                                                  : which == 2   ? &C.fbs
                                                                 : &C.sbs;
                A->set_size(2, nc);
                A->zeros();
                CHECK_THROWS_AS(C.run(), std::invalid_argument);
            }
    }
    SECTION("normal_vecN needs six columns, xprmatN needs eight") // 9.5
    {
        auto C = make_valid9();
        C.normals.set_size(1, 5);
        C.normals.zeros();
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid9();
        D.xprmat.set_size(1, 7);
        D.xprmat.zeros();
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("full-set arrays must agree on n_ray") // 9.6
    {
        auto C = make_valid9();
        C.orig = arma::join_vert(C.orig, C.orig.row(0)); // 3 rows, others 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid9();
        D.cur_in = sv({0, 0, 0});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("compact arrays must agree on n_rayN") // 9.7
    {
        auto C0 = make_valid9();
        C0.out_type = {1, 1};
        CHECK_THROWS_AS(C0.run(), std::invalid_argument);

        auto C1 = make_valid9();
        C1.fbs_angle = {QPI / 2.0, QPI / 2.0};
        CHECK_THROWS_AS(C1.run(), std::invalid_argument);

        auto C2 = make_valid9();
        C2.m1 = sv({1, 1});
        CHECK_THROWS_AS(C2.run(), std::invalid_argument);

        auto C3 = make_valid9();
        C3.m2 = sv({0, 0});
        CHECK_THROWS_AS(C3.run(), std::invalid_argument);

        auto C4 = make_valid9();
        C4.gain = {0.34, 0.34};
        CHECK_THROWS_AS(C4.run(), std::invalid_argument);

        auto C5 = make_valid9();
        C5.xprmat = arma::join_vert(C5.xprmat, C5.xprmat);
        CHECK_THROWS_AS(C5.run(), std::invalid_argument);

        auto C6 = make_valid9();
        C6.ray_ind = {1u, 0u};
        CHECK_THROWS_AS(C6.run(), std::invalid_argument);

        auto C7 = make_valid9();
        C7.normals = arma::join_vert(C7.normals, C7.normals);
        CHECK_THROWS_AS(C7.run(), std::invalid_argument);
    }
    SECTION("required and optional pointers") // 9.8
    {
        // Unconditionally required: geometry and out_typeN
        for (int k = 0; k < 5; ++k)
        {
            auto C = make_valid9();
            switch (k)
            {
            case 0: C.has_orig = false; break;
            case 1: C.has_dest = false; break;
            case 2: C.has_fbs = false; break;
            case 3: C.has_sbs = false; break;
            case 4: C.has_otype = false; break;
            }
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }

        // mtl_prop is required whenever a nonzero material index is referenced
        {
            auto C = make_valid9();
            C.has_mtl = false; // baseline has M1 = 1
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }

        // Optional with defaults: no_interact (nH = 1), fbs_angleN, mtl_ind_fbs and
        // mtl_ind_sbs (0 = air). The baseline entry then runs against air, gain-neutral.
        for (int k = 0; k < 4; ++k)
        {
            auto C = make_valid9();
            switch (k)
            {
            case 0: C.has_ni = false; break;
            case 1: C.has_angle = false; break;
            case 2: C.has_m1 = false; break;
            case 3: C.has_m2 = false; break;
            }
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("ray_ind bounds and the identity requirement") // 9.9
    {
        auto C = make_valid9();
        C.ray_ind = {5u}; // out of range for n_ray = 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid9();
        D.has_ray_ind = false; // null with n_ray != n_rayN
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("material indices must exist in the map") // 9.10
    {
        auto C = make_valid9();
        C.m1 = sv({8}); // the palette has 7 materials
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid9();
        D.cur_in = sv({8, 8});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);

        auto E = make_valid9();
        E.m1 = sv({0}); // air is always valid
        CHECK_NOTHROW(E.run());

        auto F = make_valid9();
        F.cur_in = sv({enc(7, true), enc(7, true)}); // flagged but in range
        CHECK_NOTHROW(F.run());
    }
    SECTION("the material map may be sparse but must be consistent") // 9.11
    {
        // Missing columns fall back to their defaults (sparse maps are supported)
        auto C = make_valid9();
        C.mtl.erase("a");
        CHECK_NOTHROW(C.run());

        // Columns of unequal length are an error
        auto D = make_valid9();
        D.mtl["alpha"].resize(6); // others have 7 entries
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("in-out array sizes are inputs; pure outputs may throw or resize") // 9.12
    {
        auto C = make_valid9(); // gainN with the wrong length is an input error
        C.gain = {0.34, 0.34, 0.34};
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        // prev_out with the wrong length: accept either documented behavior, a throw or
        // an internal resize to n_rayN
        auto D = make_valid9();
        D.prev_out = sv({11111, 11111, 11111});
        bool threw = false;
        try
        {
            D.run();
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        if (!threw)
        {
            CHECK(D.prev_out.n_elem == 1);
            CHECK((int)D.prev_out(0) == 0);
        }
    }
    SECTION("an empty batch is a no-op, not an error") // 9.13
    {
        Rsu<double> C;
        C.itype = 4;
        C.eps = 0.15;
        C.mtl = base_palette<double>();
        C.orig.set_size(0, 3);
        C.dest.set_size(0, 3);
        C.fbs.set_size(0, 3);
        C.sbs.set_size(0, 3);
        C.no_interact.set_size(0);
        C.fbs_angle.set_size(0);
        C.out_type.set_size(0);
        C.m1.set_size(0);
        C.m2.set_size(0);
        C.prev_in.set_size(0);
        C.cur_in.set_size(0);
        C.buf_in.set_size(0);
        C.normals.set_size(0, 6);
        C.prev_out.set_size(0);
        C.cur_out.set_size(0);
        C.buf_out.set_size(0);
        C.gain.set_size(0);
        C.xprmat.set_size(0, 8);
        CHECK_NOTHROW(C.run());
        CHECK(C.gain.n_elem == 0);
        CHECK(C.xprmat.n_rows == 0);
    }
}