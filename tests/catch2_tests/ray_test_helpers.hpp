#pragma once
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Shared helpers of the ray tests: test_wave_*.cpp and test_acoustic_*.cpp. The wave files hold every
// case in which no material with the mass column (m > 0) is evaluated; the acoustic files hold the
// cases in which one is. The three sections below are the file-scope code of the former
// test_ray_state_update.cpp, test_ray_mesh_interact.cpp and test_ray_progress.cpp.

#include <catch2/catch_test_macros.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "quadriga_lib.hpp"
#include "quadriga_tools.hpp"
#include "bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Diagnostic tables. Comment out to silence them; the assertions of the diagnostic cases run either
// way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

// ================================================================================================
// ray_state_update: verification suite conventions and helpers
// ================================================================================================

// Verification suite for the public ray_state_update API (post-VBS refactor).
//
// All expected values come from independent reference math (ITU-R P.2040 Fresnel and medium
// loss, the closed-form Airy sum), never from library internals. Where the corrected field
// depends on internals that are not part of this translation unit (the polarization-basis
// build qd_polbasis, the Snell/mirror direction updates, and Material::interact_with), the
// affected checks assert physical invariants instead: energy balance |R|^2 + |T|^2 <= 1,
// gain-vs-xprmat power consistency, and cross-pass agreement. The function is then its own
// oracle only for the parts it alone can produce, and the physics is pinned by the invariant.
//
// Model of the refactored function (what this suite assumes and checks):
//  - The interface Fresnel/Jones coefficient is written by ray_mesh_interact. ray_state_update
//    CORRECTS the incoming field: in the geometric (no path_dir) path it is a pure multiply by
//    the in-medium factor (magnitude sqrt(medium_gain), excess phase) and the thin-slab factor
//    S; it OVERWRITES the field only on the VBS path or the EM per-polarization slab resolve.
//  - Consequences used below: an o-i entry defers the medium loss (field kept); a no-crossing
//    pass-through and a same-material transition reset the field to an isotropic unit interface
//    (gain 1); a cavity exit / M2M transition / internal reflection multiplies by close * S.
//  - State: three signed-short words (mat = w & 0x3FFF, resolved/non-parallel flag = w & 0x8000,
//    bit 14 of the previous-medium word: the sheet in it lies inside the current medium).
//  - resolved_typeN: the bit-encoded outcome code (qd::bits<uint8_t>), see the code table below.
//
// Conventions:
//  - fbs_angleN uses the grazing convention (pi/2 at perpendicular incidence). The reference
//    Fresnel below is parameterized by the angle from the surface normal theta_n, so tests feed
//    fbs_angleN = pi/2 - theta_n.
//  - c0 = 299792458 m/s. The scalar (acoustic) Fresnel coefficient is the TE / admittance form.
//  - Transmission amplitudes are energy-normalized, |t|^2 = 1 - |r|^2 (Stokes-consistent).
//  - Extended material keys: "m" (mass-law slope), "tf"/"tfB" (transmission factor),
//    "e"/"f"/"g"/"h" (permeability). These are the current obj_file_read column names.

// Reference gain from ITU-R P.2040 (independent oracle, mu = 1 path).
#ifndef calc_transition_gain_HELPER
#define calc_transition_gain_HELPER
static double calc_transition_gain(int interaction_type,       // (0) Reflection, (1) Transmission, (2) Refraction
                                   double incidence_angle_deg, // Angle between face normal and ray (ITU P.2040-1) (degree)
                                   double dist1,               // Medium 1 travel distance (meters)
                                   double dist2,               // Medium 2 travel distance (meters) OR distance after reflection
                                   std::complex<double> eta1,  // relative permittivity of medium 1
                                   std::complex<double> eta2)  // relative permittivity of medium 2
{
    double deg2rad = arma::datum::pi / 180.0;

    double cos_th = std::cos(incidence_angle_deg * deg2rad); // Incidence on boundary
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);        // Trigonometric identity
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * sin_th * sin_th);

    // Medium 1 loss
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * dist1 / Delta;                // Attenuation in dB/m, ITU-R P.2040-1, eq. (26)
    double medium_1_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    // Medium 2 loss
    if (interaction_type != 0) // Use eta1 for reflection
    {
        tan_delta = std::imag(eta2) / std::real(eta2);
        cos_delta = std::cos(std::atan(tan_delta));
        Delta = 2.0 * cos_delta / (1.0 - cos_delta);
        Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta2)));
    }
    A = 8.686 * dist2 / Delta;
    double medium_2_gain = std::pow(10.0, -0.1 * A);

    eta1 = std::sqrt(eta1);
    eta2 = std::sqrt(eta2);

    std::complex<double> R_te = (eta1 * cos_th - eta2 * cos_th2) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> R_tm = (eta2 * cos_th - eta1 * cos_th2) / (eta2 * cos_th + eta1 * cos_th2);
    std::complex<double> T_te = (2.0 * eta1 * cos_th) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> T_tm = (2.0 * eta1 * cos_th) / (eta2 * cos_th + eta1 * cos_th2);

    double reflection_gain = 0.5 * (std::norm(R_te) + std::norm(R_tm));
    double refraction_gain = 0.5 * (std::norm(T_te) + std::norm(T_tm));

    double total_gain = 0.0;
    if (interaction_type == 0) // Reflection
        total_gain = medium_1_gain * reflection_gain * medium_2_gain;
    else if (interaction_type == 1) // Transmission
        total_gain = medium_1_gain * (1.0 - reflection_gain) * medium_2_gain;
    else if (interaction_type == 2) // Refraction
        total_gain = medium_1_gain * refraction_gain * medium_2_gain;

    return total_gain;
}
#endif

// mu-aware reference Fresnel: independent oracle for the e,f,g,h (permeability) path.
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

// Append constant permeability columns e,f,g,h to a material map.
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

// Convert a per-face matrix [n_face, 9] with columns {a,b,c,d,att,attB,alpha,alphaB,fRef} into
// a deduplicated (mtl_ind, mtl_prop-map) pair, matching what obj_file_read would emit.
template <typename dtype>
static inline void mtl_matrix_to_map0(const arma::Mat<dtype> &M,
                                     arma::uvec &mtl_ind,
                                     std::unordered_map<std::string, std::vector<dtype>> &mtl_prop)
{
    static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
    const arma::uword n_face = M.n_rows;

    mtl_ind.set_size(n_face);
    std::vector<arma::uword> uniq;
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
static const double C0 = 299792458.0;      // speed of light, m/s
static const double QPI = arma::datum::pi; // pi
static const double TOL = 1.0e-9;          // double tolerance
static const double FRQ = 1.0e9;           // default test frequency, Hz

// out_typeN codes produced by ray_mesh_interact (bit-encoded, qd::bits<uint8_t>).
//   bit 0 OK, bit 1 front-side, bit 2 co-located FBS/SBS, bit 3 same-direction normals,
//   bit 4 corner (faces not parallel), bit 5 total reflection. Add 32 for the TIR variant.
static constexpr uint8_t OT_NOHIT = 0;    // No hit / transparent
static constexpr uint8_t OT_EXIT = 1;     // i->o single-hit exit
static constexpr uint8_t OT_ENTRY = 3;    // o->i single-hit entry
static constexpr uint8_t OT_M2M_M1 = 5;   // media-to-media, M1 (back) hit first
static constexpr uint8_t OT_M2M_M2 = 7;   // media-to-media, M2 (front) hit first
static constexpr uint8_t OT_OVL_IIO = 13; // overlapping faces, ii->o
static constexpr uint8_t OT_OVL_OII = 15; // overlapping faces, o->ii
static constexpr uint8_t OT_COR_IOI = 21; // corner, i-o-i
static constexpr uint8_t OT_COR_OIO = 23; // corner, o-i-o
static constexpr uint8_t OT_TIR = 32;     // add to any code for the total-reflection variant

// resolved_typeN outcome codes (bit-encoded). Base values used by the assertions:
//   bit0 ok, bit1 vbs, bit2 resolve, bit3 inside, bit4 fix, bit5 tir, bit6 trans, bit7 refl.
static constexpr uint8_t RT_KILL = 0;          // ray killed
static constexpr uint8_t RT_ENTRY = 9;         // inside, ok            (o-i entry / i-i FBS==VBS)
static constexpr uint8_t RT_ENTRY_DEFER = 8;   // inside                (o-i entry, deferred buffer)
static constexpr uint8_t RT_EXIT = 1;          // ok                    (i-o exit)
static constexpr uint8_t RT_EXIT_SLAB = 5;     // resolve, ok           (i-o exit + slab series)
static constexpr uint8_t RT_SAMEPASS = 73;     // trans, inside, ok     (same-medium pass)
static constexpr uint8_t RT_NESTED_DEFER = 72; // trans, inside         (nested pass, deferred buffer)
static constexpr uint8_t RT_ADVANCE = 65;      // trans, ok             (advance to destination)
static constexpr uint8_t RT_R0 = 129;          // refl, ok              (front reflection R0)
static constexpr uint8_t RT_RINT = 137;        // refl, inside, ok      (internal back reflection)
static constexpr uint8_t RT_RINT_SLAB = 141;   // refl, inside, resolve, ok

// Read one bit of a resolved_type / out_type code
static inline bool rt_bit(uint8_t code, unsigned b) { return qd::bits<uint8_t>(code).test(b); }

// State-word encoding (bit 15 = resolved / non-parallel flag, bits 0..14 = material index).
// Uses qd::bits so the intent is explicit; never arithmetic negation.
static inline short enc(int m, bool flag)
{
    qd::bits<short> w = (short)m;
    if (flag)
        w.set(15);
    return w.get();
}
static inline int mat_of(short w) { return (int)(unsigned short)qd::bits<short>(w).tail(14); }
static inline bool flag_of(short w) { return qd::bits<short>(w).test(15); }

// Col<short> from int literals without narrowing warnings
static inline arma::Col<short> sv(std::initializer_list<int> v)
{
    arma::Col<short> out((arma::uword)v.size());
    arma::uword i = 0;
    for (int x : v)
        out(i++) = (short)x;
    return out;
}

// Complex view of xprmat entry pair k in {0 VV, 1 HV, 2 VH, 3 HH} for ray i.
// Layout is [nXPR, n_rayN] column-major: entry k of ray i sits at rows (2k, 2k+1), column i.
template <typename dtype>
static inline std::complex<double> cpx(const arma::Mat<dtype> &X, arma::uword i, int k)
{
    return std::complex<double>((double)X(2 * k, i), (double)X(2 * k + 1, i));
}

// Complex TE / scalar Fresnel pair at an interface, parameterized by the angle from the
// surface NORMAL on the incident (eta1) side. t is energy-normalized to |t|^2 = 1 - |r|^2.
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

// One-way in-slab propagation factor phi: |phi| = sqrt(medium_gain(L)),
// arg(phi) = -(omega/c) * n_real * L (normal incidence).
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

// Survival-gate quantity rho = sqrt(R_near * R_far * medium_gain(2L))
static inline double gate_rho(std::complex<double> r_near, std::complex<double> r_far, double medium_gain_2L)
{
    return std::sqrt(std::norm(r_near) * std::norm(r_far) * medium_gain_2L);
}

// One-way path of m half-waves inside a medium of (real) index n
static inline double half_wave(double n_real, double f_hz)
{
    return C0 / (2.0 * f_hz * n_real);
}

// Material palette, 1-based indices, 0 = air:
//   1 FOG1   eta = 1, alpha = 2 dB/m  (no Fresnel mismatch, pure in-medium loss)
//   2 FOG2   identical to FOG1 but a distinct index (SAME(...) false, TRN exactly 1)
//   3 FOG3   eta = 1, alpha = 5 dB/m
//   4 DENSE  eta = 4, lossless (n = 2, inside-face r = 1/3)
//   5 NINE   eta = 9, lossless (n = 3, inside-face r = 1/2, lossless rho = 0.25)
//   6 ABSORB eta = 4, alpha = 30 dB/m
//   7 WEAK   eta = 1.21, lossless
// b = c = d = att = attB = alphaB = 0, fRef = 1. eta is real and frequency-flat; the only
// medium loss is alpha * dist dB, exactly medg(idx, dist).
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

// Medium gain of a palette material over dist meters (alpha-only loss)
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

// Complex relative permittivity of a palette material (0 = air = 1)
static inline std::complex<double> et(int idx)
{
    static const double a[8] = {1.0, 1.0, 1.0, 1.0, 4.0, 9.0, 4.0, 1.21};
    return std::complex<double>(idx < 1 ? 1.0 : a[idx], 0.0);
}

// In-medium close factor for a single crossing at normal incidence (VBS distance = geometric
// distance): amp = sqrt(medg(m, L)), excess phase = k0 * (n_of(m) - 1) * L.
static inline std::complex<double> close_at(int m, double L, double f = FRQ)
{
    double k0 = 2.0 * QPI * f / C0;
    double excess = k0 * (n_of(m) - 1.0) * L;
    return std::sqrt(medg(m, L)) * std::exp(std::complex<double>(0.0, -excess));
}

// Call harness. Owns every argument of ray_state_update; has_* switches turn each optional
// (and, for the validation group, required) pointer into nullptr. run() snapshots the incoming
// gain / xprmat so factor checks can compare out against in.
//   Full ray set  [n_ray]  : orig, dest, no_interact, prev_in, cur_in, buf_in, path_dir_prev,
//                            acc_dist_geometric, acc_dist_refracted
//   Compact set   [n_rayN] : fbs, sbs, fbs_angle, normals, out_type, m1, m2, and every output
template <typename dtype>
struct Rsu
{
    int itype = 4;
    dtype freq = (dtype)FRQ;
    double eps = 0.15;

    arma::Mat<dtype> orig, dest;   // full  [n_ray, 3]
    arma::Mat<dtype> fbs, sbs;     // compact [n_rayN, 3]
    arma::u32_vec no_interact;     // full  [n_ray]
    arma::Col<dtype> fbs_angle;    // compact [n_rayN]
    arma::Mat<dtype> normals;      // compact [n_rayN, 6]
    std::vector<uint8_t> out_type; // compact [n_rayN]
    std::unordered_map<std::string, std::vector<dtype>> mtl;
    arma::Col<short> m1, m2;                  // compact [n_rayN]
    arma::Col<short> prev_in, cur_in, buf_in; // full [n_ray]
    arma::Mat<dtype> path_dir_prev;           // full  [n_ray, 3]
    arma::Col<dtype> acc_geo_in;              // full  [n_ray]  geometric in-medium distance
    arma::Col<dtype> acc_ref_in;              // full  [n_ray]  refracted (VBS) in-medium distance

    arma::Col<short> prev_out, cur_out, buf_out; // compact [n_rayN]
    arma::Col<dtype> gain;                       // compact [n_rayN]
    arma::Mat<dtype> xprmat;                     // compact [nXPR, n_rayN]
    arma::Mat<dtype> path_dirN;                  // compact [n_rayN, 3]
    arma::Col<dtype> acc_geo_out;                // compact [n_rayN]
    arma::Col<dtype> acc_ref_out;                // compact [n_rayN]
    std::vector<uint8_t> resolved_type;          // compact [n_rayN]
    arma::u32_vec ray_ind;                       // compact [n_rayN]

    arma::Col<dtype> g_in; // snapshots taken by run()
    arma::Mat<dtype> x_in;

    bool has_orig = true, has_dest = true, has_fbs = true, has_sbs = true, has_ni = true;
    bool has_angle = true, has_normals = true, has_otype = true, has_mtl = true;
    bool has_m1 = true, has_m2 = true;
    bool has_prev_in = true, has_cur_in = true, has_buf_in = true;
    bool has_path_dir = false;
    bool has_acc_geo_in = false, has_acc_ref_in = false;
    bool has_prev_out = true, has_cur_out = true, has_buf_out = true;
    bool has_gain = true, has_xprmat = true;
    bool has_path_dirN = false, has_rtype = true, has_ray_ind = false;
    bool has_acc_geo_out = false, has_acc_ref_out = false;

    int nXPR() const { return itype >= 3 ? 2 : 8; }

    void run()
    {
        g_in = gain;
        x_in = xprmat;
        quadriga_lib::ray_state_update<dtype>(
            itype, freq,
            has_orig ? &orig : nullptr,
            has_dest ? &dest : nullptr,
            has_fbs ? &fbs : nullptr,
            has_sbs ? &sbs : nullptr,
            has_ni ? &no_interact : nullptr,
            has_angle ? &fbs_angle : nullptr,
            has_normals ? &normals : nullptr,
            has_otype ? &out_type : nullptr,
            has_mtl ? &mtl : nullptr,
            has_m1 ? &m1 : nullptr,
            has_m2 ? &m2 : nullptr,
            has_prev_in ? &prev_in : nullptr,
            has_cur_in ? &cur_in : nullptr,
            has_buf_in ? &buf_in : nullptr,
            has_path_dir ? &path_dir_prev : nullptr,
            has_acc_geo_in ? &acc_geo_in : nullptr,
            has_acc_ref_in ? &acc_ref_in : nullptr,
            has_prev_out ? &prev_out : nullptr,
            has_cur_out ? &cur_out : nullptr,
            has_buf_out ? &buf_out : nullptr,
            has_gain ? &gain : nullptr,
            has_xprmat ? &xprmat : nullptr,
            has_path_dirN ? &path_dirN : nullptr,
            has_acc_geo_out ? &acc_geo_out : nullptr,
            has_acc_ref_out ? &acc_ref_out : nullptr,
            has_rtype ? &resolved_type : nullptr,
            has_ray_ind ? &ray_ind : nullptr,
            eps);
    }
};

// Single-ray call builder on the x axis: orig at 0, fbs at d_orig_fbs, sbs at
// d_orig_fbs + d_fbs_sbs, dest at d_orig_fbs + d_fbs_dest. Default normals are the antiparallel
// slab pair (-1,0,0 | 1,0,0). The probe field is `feed` on VV (scalar) or VV and HH (EM).
template <typename dtype>
static inline Rsu<dtype> make1(const std::unordered_map<std::string, std::vector<dtype>> &mtl,
                               int itype, uint8_t otype, unsigned nH, int M1, int M2,
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
    int nX = C.nXPR();

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
    C.out_type.assign(1, otype);

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
    C.resolved_type.assign(1, (uint8_t)222); // sentinel

    C.xprmat.zeros(nX, 1);
    C.xprmat(0, 0) = (dtype)feed.real(); // VV re
    C.xprmat(1, 0) = (dtype)feed.imag(); // VV im
    if (itype < 3)                       // EM: probe HH as well
    {
        C.xprmat(6, 0) = (dtype)feed.real();
        C.xprmat(7, 0) = (dtype)feed.imag();
    }
    C.gain.set_size(1);
    C.gain(0) = (dtype)std::norm(feed); // |VV|^2 (scalar) = 0.5(|VV|^2+|HH|^2) (EM)

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

// Resolved interaction-type code, exact comparison
template <typename dtype>
static inline void check_rtype(const Rsu<dtype> &C, uint8_t code, arma::uword i = 0)
{
    CHECK((int)C.resolved_type[i] == (int)code);
}

// Keep: outputs bit-identical to inputs
template <typename dtype>
static inline void check_keep(const Rsu<dtype> &C)
{
    CHECK(arma::approx_equal(C.xprmat, C.x_in, "absdiff", (dtype)0));
    if (C.has_gain)
        CHECK(arma::approx_equal(C.gain, C.g_in, "absdiff", (dtype)0));
}

// Kill: gain and xprmat zero
template <typename dtype>
static inline void check_kill(const Rsu<dtype> &C, arma::uword i = 0)
{
    for (int r = 0; r < C.nXPR(); ++r)
        CHECK(std::abs((double)C.xprmat(r, i)) == 0.0);
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i)) == 0.0);
}

// Multiply: each complex channel multiplied by f, gain by |f|^2
template <typename dtype>
static inline void check_mult(const Rsu<dtype> &C, std::complex<double> f, double tol, arma::uword i = 0)
{
    for (int k = 0; k < C.nXPR() / 2; ++k)
    {
        std::complex<double> e = cpx(C.x_in, i, k) * f;
        CHECK(std::abs((double)C.xprmat(2 * k, i) - e.real()) < tol);
        CHECK(std::abs((double)C.xprmat(2 * k + 1, i) - e.imag()) < tol);
    }
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i) - (double)C.g_in(i) * std::norm(f)) < tol);
}

// Isotropic replace with power g: sqrt(g) on VV (and HH in EM mode), zeros elsewhere, gain = g
template <typename dtype>
static inline void check_replace(const Rsu<dtype> &C, double g, double tol, arma::uword i = 0)
{
    double s = std::sqrt(g);
    bool scalar = C.itype >= 3;
    CHECK(std::abs((double)C.xprmat(0, i) - s) < tol); // VV re
    CHECK(std::abs((double)C.xprmat(1, i)) < tol);     // VV im
    if (!scalar)
    {
        for (int r = 2; r < 6; ++r)
            CHECK(std::abs((double)C.xprmat(r, i)) < tol); // HV, VH
        CHECK(std::abs((double)C.xprmat(6, i) - s) < tol); // HH re
        CHECK(std::abs((double)C.xprmat(7, i)) < tol);     // HH im
    }
    if (C.has_gain)
        CHECK(std::abs((double)C.gain(i) - g) < tol);
}

// Applied complex factor on the VV pair (out / in)
template <typename dtype>
static inline std::complex<double> vvf(const Rsu<dtype> &C, arma::uword i = 0)
{
    return cpx(C.xprmat, i, 0) / cpx(C.x_in, i, 0);
}

// A pure scale leaves the gain equal to |VV factor|^2 times the incoming gain. Used as a class
// check when the exact factor is not asserted (e.g. the VBS / EM paths).
template <typename dtype>
static inline void check_scale_consistent(const Rsu<dtype> &C, double tol, arma::uword i = 0)
{
    double f2 = std::norm(vvf(C, i));
    CHECK(std::abs((double)C.gain(i) - (double)C.g_in(i) * f2) < tol);
}

// Standard single-ray dispatch probe: scalar transmission, eps = 1.5 (slab off), normal
// incidence, default distances d(orig,fbs) = 1, d(fbs,dest) = 2, d(fbs,sbs) = 0.5.
static inline Rsu<double> disp(uint8_t otype, unsigned nH, int M1, int M2,
                               int prev, int cur, int buf,
                               double dof = 1.0, double dfd = 2.0, double dsbs = 0.5,
                               int itype = 4, double eps = 1.5)
{
    static const std::unordered_map<std::string, std::vector<double>> mtl = base_palette<double>();
    auto C = make1<double>(mtl, itype, otype, nH, M1, M2, prev, cur, buf, dof, dfd, dsbs, QPI / 2.0, eps);
    C.run();
    return C;
}

// Merge single-ray harness objects into one identity-mapped batch (same pass settings).
// xprmat rows are the polarization entries, so rays join along columns.
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
        B.normals = arma::join_vert(B.normals, r.normals);
        B.out_type.insert(B.out_type.end(), r.out_type.begin(), r.out_type.end());
        B.m1 = arma::join_vert(B.m1, r.m1);
        B.m2 = arma::join_vert(B.m2, r.m2);
        B.prev_in = arma::join_vert(B.prev_in, r.prev_in);
        B.cur_in = arma::join_vert(B.cur_in, r.cur_in);
        B.buf_in = arma::join_vert(B.buf_in, r.buf_in);
        B.prev_out = arma::join_vert(B.prev_out, r.prev_out);
        B.cur_out = arma::join_vert(B.cur_out, r.cur_out);
        B.buf_out = arma::join_vert(B.buf_out, r.buf_out);
        B.resolved_type.insert(B.resolved_type.end(), r.resolved_type.begin(), r.resolved_type.end());
        B.gain = arma::join_vert(B.gain, r.gain);
        B.xprmat = arma::join_horiz(B.xprmat, r.xprmat);
    }
    return B;
}

// Type parity: the same scenario set in a second precision. Tolerances: 1e-9 for double, a
// looser bound for float (the VBS / phase math accumulates more rounding than the legacy path).
template <typename dtype>
static void run_parity(double tol)
{
    const auto mtl = base_palette<dtype>();
    double g2 = QPI / 2.0;

    { // entry keeps the field
        auto C = make1<dtype>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, g2, 1.5);
        C.run();
        check_keep(C);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_ENTRY);
    }
    { // cavity exit multiplies by the medium close
        auto C = make1<dtype>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, 1.2, 2.0, 0.5, g2, 1.5);
        C.run();
        check_mult(C, close_at(1, 1.2), tol);
        check_state(C, 1, 0, 0);
    }
    { // unmatched code kills
        auto C = make1<dtype>(mtl, 4, (uint8_t)17, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, g2, 1.5);
        C.run();
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
    { // Airy factor at the m = 1 resonance
        double L = half_wave(2.0, FRQ);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        auto C = make1<dtype>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, g2, 0.0);
        C.run();
        check_mult(C, close_at(4, L) * S, tol);
        check_state(C, 4, 0, 0);
    }
    { // replace layouts in both modes
        auto A = make1<dtype>(mtl, 1, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, g2, 1.5);
        A.run();
        check_replace(A, 1.0, tol);
        auto B = make1<dtype>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, g2, 1.5);
        B.run();
        check_replace(B, 1.0, tol);
    }
    { // lossless energy closure at normal incidence
        double L = 0.275;
        FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0);
        FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0);
        double k0 = 2.0 * QPI * FRQ / C0;
        auto avv = [](const Rsu<dtype> &C)
        { return cpx(C.xprmat, 0, 0); };
        auto ev = [&](int it, uint8_t ot, int prev, int cur, double dof, std::complex<double> feed)
        {
            auto C = make1<dtype>(mtl, it, ot, 1, 4, 0, prev, cur, 0, dof, 2.0, 0.5, g2, 0.0, feed);
            C.run();
            return C;
        };
        auto E = ev(4, OT_ENTRY, 0, 0, 0.5, out.t);
        auto X = ev(4, OT_EXIT, 0, 4, L, ins.t);
        auto F = ev(3, OT_ENTRY, 0, 0, 0.5, out.r);
        auto I = ev(3, OT_EXIT, 0, 4, L, ins.r);
        auto O = ev(4, OT_EXIT, I.prev_out(0), I.cur_out(0), L, ins.t);
        std::complex<double> vac = std::exp(std::complex<double>(0.0, -k0 * L));
        std::complex<double> T = out.t * vac * avv(X);
        std::complex<double> B = out.t * vac * vac * avv(I) * avv(O);
        std::complex<double> R = avv(F) + B;
        CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < tol);
    }
}

// Valid baseline for the validation group: n_ray = 2 (full set), n_rayN = 1 (compact set),
// ray_ind = {1}, a clean o-i entry into FOG1.
static Rsu<double> make_valid()
{
    auto C = make1<double>(base_palette<double>(), 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 10.0, QPI / 2.0, 0.15);
    C.orig = arma::join_vert(C.orig, C.orig); // full set -> 2 rows
    C.dest = arma::join_vert(C.dest, C.dest);
    C.no_interact = arma::join_vert(C.no_interact, C.no_interact);
    C.prev_in = arma::join_vert(C.prev_in, C.prev_in);
    C.cur_in = arma::join_vert(C.cur_in, C.cur_in);
    C.buf_in = arma::join_vert(C.buf_in, C.buf_in);
    C.ray_ind.set_size(1);
    C.ray_ind(0) = 1;
    C.has_ray_ind = true;
    return C;
}

// Palette for the diagnostics, 1-based, 0 = air. Conductive so that the in-slab refraction
// cosine is genuinely complex, which is the condition every open item below depends on.
//   1 LOSSY   eps = 5.24 - 0.63j at 1 GHz, no alpha
//   2 CLEAN   eps = 5.24, lossless (control: every complex-angle effect vanishes)
//   3 SHEET   electrically identical to LOSSY, plus thk = 6 mm (a distinct material index)
//   4 DENSER  eps = 9.0 - 1.20j at 1 GHz
//   5 LO      eps = 1.50 - 0.40j at 1 GHz
//   6 LO_CLEAN eps = 1.50, lossless
//   7 LO_SHEET electrically identical to LO, plus thk = 6 mm
//   8 LO_CLEAN_TWIN electrically identical to LO_CLEAN, plus thk = 6 mm
//   9 LO_NEAR  eps = 1.5015 - 0.40j at 1 GHz, plus thk = 6 mm
//
// LO_NEAR is a sheet NEAR-matched to LO rather than matched to it. The 1e-3 relative offset in the
// real part is large enough that TRANSPARENT_TO fails, so SHEET_EVENT does not merge the host layer
// and the sheet is written into the previous-medium word, and small enough that the host | sheet
// reflection is 2.5e-4 in amplitude, so the wave passes through it almost unimpeded. It is the case
// section 3.1 names as the one merge_host cannot catch: correction-mode Fresnel reads r = 0 at that
// boundary and drops the host layer's internal series, when physically the sheet is a nearly
// transparent 6 mm layer through which the wave still sees the real boundary behind it.
//
// The twins (3 with 1, 7 with 5, 8 with 6) are distinct material indices presenting no boundary to a
// wave. Placed as the previous medium they make the far Fresnel coefficient vanish, so the Airy
// denominator is 1 and the series factor is exactly 1 with everything else about the run untouched -
// which is how a diagnostic measures its own reference instead of reconstructing it.
//
// The LO group exists because an i-o exit from eps 5.24 into air total-reflects above
// asin(1/2.289) = 25.9 deg, which turns every row past 25 deg of an in-slab sweep into a
// forced-TIR result carrying no information about the slab. eps 1.50 puts the critical angle at
// asin(1/1.2247) = 54.7 deg, so a sweep to 50 deg stays entirely below it while keeping a loss
// tangent of 0.27 - enough that the in-slab refraction cosine is genuinely complex, which is the
// condition every open item depends on.
template <typename dtype>
static inline std::unordered_map<std::string, std::vector<dtype>> diag_palette()
{
    std::unordered_map<std::string, std::vector<dtype>> m;
    m["a"] = {(dtype)5.24, (dtype)5.24, (dtype)5.24, (dtype)9.0,
              (dtype)1.50, (dtype)1.50, (dtype)1.50, (dtype)1.50, (dtype)1.5015};
    m["b"] = std::vector<dtype>(9, (dtype)0.0);
    m["c"] = {(dtype)(0.63 / 17.98), (dtype)0.0, (dtype)(0.63 / 17.98), (dtype)(1.20 / 17.98),
              (dtype)(0.40 / 17.98), (dtype)0.0, (dtype)(0.40 / 17.98), (dtype)0.0,
              (dtype)(0.40 / 17.98)};
    m["d"] = std::vector<dtype>(9, (dtype)0.0);
    m["att"] = std::vector<dtype>(9, (dtype)0.0);
    m["attB"] = std::vector<dtype>(9, (dtype)0.0);
    m["alpha"] = std::vector<dtype>(9, (dtype)0.0);
    m["alphaB"] = std::vector<dtype>(9, (dtype)0.0);
    m["fRef"] = std::vector<dtype>(9, (dtype)1.0);
    m["thk"] = {(dtype)0.0, (dtype)0.0, (dtype)0.006, (dtype)0.0,
                (dtype)0.0, (dtype)0.0, (dtype)0.006, (dtype)0.006, (dtype)0.006};
    return m;
}

// Complex relative permittivity of a diag_palette entry at f_hz
// Index 0 is air, and the tables run to the palette's last material.
static inline std::complex<double> diag_eps(int idx, double f_hz)
{
    static const double a[10] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50, 1.50, 1.50, 1.5015};
    static const double c[10] = {0.0, 0.63 / 17.98, 0.0, 0.63 / 17.98, 1.20 / 17.98,
                                 0.40 / 17.98, 0.0, 0.40 / 17.98, 0.0, 0.40 / 17.98};
    if (idx < 1 || idx > 9)
        return std::complex<double>(1.0, 0.0);
    return std::complex<double>(a[idx], -17.98 * c[idx] / (f_hz * 1e-9));
}

// In-medium correction the function applies over an accumulated path L at fbs angle theta_n, with
// no path_dir supplied. Two things this has to get right, both of which the first run got wrong:
// the harness feeds theta_n straight in as fbs_angle, so abs_cos_theta_t is cos(theta_n) and NOT a
// refracted cosine; and medium_gain is evaluated over dist_refract = L, the accumulated geometric
// length, with no 1/cos factor. The excess phase then follows the walk-off form
// k0 * (n - 1) * L * cos^2(theta_n), which the diagnostics confirmed to 0.000 deg.
static inline std::complex<double> diag_close(std::complex<double> eps, double L,
                                              double theta_n_rad, double f_hz)
{
    double ct = std::cos(theta_n_rad);
    double n_re = std::real(std::sqrt(eps));
    double k0 = 2.0 * QPI * f_hz / C0;
    double er = std::real(eps), td = std::imag(eps) / er;
    double cd = 1.0 / std::sqrt(1.0 + td * td);
    double Del = std::sqrt(2.0 * cd / (1.0 - cd)) * 0.0477135 / ((f_hz * 1e-9) * std::sqrt(er));
    double gL = std::pow(10.0, -0.1 * (L * 8.686 / Del));
    double excess = k0 * (n_re - 1.0) * L * ct * ct;
    return std::sqrt(gL) * std::exp(std::complex<double>(0.0, -excess));
}

// One-way in-slab propagation factor over the same path, same conventions as diag_close.
static inline std::complex<double> diag_phi(std::complex<double> eps, double L,
                                            double theta_n_rad, double f_hz)
{
    double ct = std::cos(theta_n_rad);
    double n_re = std::real(std::sqrt(eps));
    double k0 = 2.0 * QPI * f_hz / C0;
    double er = std::real(eps), td = std::imag(eps) / er;
    double cd = 1.0 / std::sqrt(1.0 + td * td);
    double Del = std::sqrt(2.0 * cd / (1.0 - cd)) * 0.0477135 / ((f_hz * 1e-9) * std::sqrt(er));
    double gL = std::pow(10.0, -0.1 * (L * 8.686 / Del));
    return std::sqrt(gL) * std::exp(std::complex<double>(0.0, -k0 * n_re * L * ct * ct));
}

// The qd_refract map itself: incidence angle in air -> the real in-slab cosine a tracer is left
// holding. qd_refract normalizes eta*u + (eta*cos_in - Re(ct2))*n, so the traced cosine is NOT
// Re(ct2) but Re(ct2)/sqrt(eta^2 sin^2 + Re(ct2)^2). Reproduced here so the fixture can hand the
// function the same direction a tracer would have produced, and so the reference for the recovery is
// the map that was actually applied rather than real-index Snell, which is a different map.
static inline double diag_traced_ct(std::complex<double> eps_slab, double theta_ext_rad,
                                    std::complex<double> mu_slab = std::complex<double>(1.0, 0.0))
{
    std::complex<double> eps_r = 1.0 / (eps_slab * mu_slab); // entry medium is air
    double s = std::sin(theta_ext_rad);
    std::complex<double> ct2 = std::sqrt(1.0 - eps_r * s * s);
    double eta = std::sqrt(std::abs(eps_r)), r = std::real(ct2);
    return r / std::sqrt((eta * s) * (eta * s) + r * r);
}

// The exact complex in-slab cosine the projection above stands for
static inline std::complex<double> diag_ct_exact(std::complex<double> eps_slab, double theta_ext_rad,
                                                 std::complex<double> mu_slab = std::complex<double>(1.0, 0.0))
{
    double s = std::sin(theta_ext_rad);
    return std::sqrt(1.0 - (1.0 / (eps_slab * mu_slab)) * s * s);
}

// Conductivity-only medium gain over a path, ITU-R P.2040 form (the palette carries no alpha)
static inline double diag_medium_gain(std::complex<double> eps, double dist, double f_hz)
{
    double er = std::real(eps), td = std::imag(eps) / er;
    double cd = 1.0 / std::sqrt(1.0 + td * td);
    double Del = std::sqrt(2.0 * cd / (1.0 - cd)) * 0.0477135 / ((f_hz * 1e-9) * std::sqrt(er));
    return std::pow(10.0, -0.1 * (dist * 8.686 / Del));
}

// Slab-side Fresnel reflection at a slab|adjacent boundary, TE / scalar form, parameterized by
// the cosine used on the slab side. Passing the real projection reproduces what fresnel_r does;
// passing the exact complex in-slab cosine gives the value the entry boundary actually applied.
static inline std::complex<double> diag_r_slabside(std::complex<double> eps_s, std::complex<double> eps_a,
                                                   std::complex<double> cos_slab)
{
    std::complex<double> z1 = std::sqrt(eps_s), z2 = std::sqrt(eps_a);
    std::complex<double> sin2 = 1.0 - cos_slab * cos_slab;
    std::complex<double> c2 = std::sqrt(1.0 - (eps_s / eps_a) * sin2);
    return (z1 * cos_slab - z2 * c2) / (z1 * cos_slab + z2 * c2);
}

// Exact analytic slab, TE/TM averaged power, air | slab | air. Independent of every library
// internal; the same closed form the ray_mesh_interact suite uses as its oracle.
// Exact analytic slab, per polarization. Same closed form as diag_slab_p_avg, which is the
// average of the two outputs; split out because a scalar chain measures one channel.
static inline void diag_slab_p_pol(std::complex<double> eps_s, double thk,
                                   double theta_n_rad, double f_hz,
                                   double &p_te, double &p_tm,
                                   std::complex<double> mu_s = std::complex<double>(1.0, 0.0))
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s / mu_s),
                               n = std::sqrt(eps_s * mu_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / (eps_s * mu_s)) * s1 * s1);
    const std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * thk * c2);
    double p[2] = {0.0, 0.0};
    for (int pol = 0; pol < 2; ++pol)
    {
        std::complex<double> r, t12, t21;
        if (pol == 0)
            r = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2),
            t12 = (2.0 * z1 * c1) / (z1 * c1 + z2 * c2),
            t21 = (2.0 * z2 * c2) / (z2 * c2 + z1 * c1);
        else
            r = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2),
            t12 = (2.0 * z1 * c1) / (z2 * c1 + z1 * c2),
            t21 = (2.0 * z2 * c2) / (z1 * c2 + z2 * c1);
        p[pol] = std::norm(t12 * t21 * phi / (1.0 - r * r * phi * phi));
    }
    p_te = p[0], p_tm = p[1];
}

// The transmission port Material::interact_with writes for a crossing e1 -> e2, at an incidence
// cosine that may be complex: the raw Fresnel amplitude times the square root of its own
// per-channel normal-flux obliquity factor, Re(z2 c2)/Re(z1 c1) for TE and the conjugated form for
// TM. The diagnostic palette carries no tf and no att, so neither of those two scales appears.
//
// Passing a real cosine reproduces an ordinary face hit; passing the exact complex in-slab cosine
// reproduces the far face of a traced solid.
static inline void diag_port(std::complex<double> e1, std::complex<double> e2,
                             std::complex<double> ct1, bool scalar,
                             std::complex<double> &T_te, std::complex<double> &T_tm)
{
    std::complex<double> z1 = std::sqrt(e1), z2 = std::sqrt(e2);
    std::complex<double> a = 1.0 - (e1 / e2) * (1.0 - ct1 * ct1);
    if (std::imag(a) == 0.0 && std::real(a) < 0.0)
        a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
    std::complex<double> ct2 = std::sqrt(a);

    double den_te = std::real(z1 * ct1), den_tm = std::real(std::conj(z1) * ct1);
    double num_te = std::real(z2 * ct2), num_tm = std::real(std::conj(z2) * ct2);
    double s_te = (den_te > 0.0 && num_te > 0.0) ? std::sqrt(num_te / den_te) : 0.0;
    double s_tm = (den_tm > 0.0 && num_tm > 0.0) ? std::sqrt(num_tm / den_tm) : 0.0;

    if (scalar) // pressure form, T = 1 + R, which is the TE amplitude
    {
        std::complex<double> R = (z1 * ct1 - z2 * ct2) / (z1 * ct1 + z2 * ct2);
        T_te = (1.0 + R) * s_te;
        T_tm = T_te;
        return;
    }
    T_te = ((2.0 * z1 * ct1) / (z1 * ct1 + z2 * ct2)) * s_te;
    T_tm = ((2.0 * z1 * ct1) / (z2 * ct1 + z1 * ct2)) * s_tm;
}

// Correction-mode series factor S = 1 / (1 - rn rf phi^2), with both boundaries and the
// propagation factor written at one in-slab cosine. Reflections are taken from the slab side, so
// the near and far media enter only through their own refraction cosines.
static inline void diag_series(std::complex<double> es, std::complex<double> en,
                               std::complex<double> ef, std::complex<double> cs,
                               double thk_normal, double f_hz, bool scalar,
                               std::complex<double> &S_te, std::complex<double> &S_tm)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> zs = std::sqrt(es);
    const std::complex<double> sin2 = 1.0 - cs * cs;

    auto refl = [&](std::complex<double> ea, std::complex<double> &te, std::complex<double> &tm)
    {
        std::complex<double> za = std::sqrt(ea);
        std::complex<double> a = 1.0 - (es / ea) * sin2;
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0);
        std::complex<double> c2 = std::sqrt(a);
        te = (zs * cs - za * c2) / (zs * cs + za * c2);
        tm = scalar ? te : (za * cs - zs * c2) / (za * cs + zs * c2);
    };

    std::complex<double> rn_te, rn_tm, rf_te, rf_tm;
    refl(en, rn_te, rn_tm);
    refl(ef, rf_te, rf_tm);
    std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(es) * thk_normal * cs);
    S_te = 1.0 / (1.0 - rn_te * rf_te * phi * phi);
    S_tm = scalar ? S_te : 1.0 / (1.0 - rn_tm * rf_tm * phi * phi);
}

// In-medium traversal factor on a leg the tracer refracted into, exact form:
//   exp(-j k0 (n cs d + q s)) / exp(-j k0 dist_geo)
// with d the normal thickness and s the transverse displacement, both pure geometry off the traced
// direction, and q the transverse wavenumber the medium carries, q^2 = n^2 (1 - cs^2). The
// division is the free-space reference the distance ledger already carries.
//
// For a real index the two terms recombine into n * dist_refract, so this reduces to the walk-off
// form rather than replacing it. The palette carries no alpha, so there is no excess-gain factor.
//
// That is the phase at the point where the refracted ray leaves. With ct_geo, the cosine of the
// segment, it is carried to the point where the polyline leaves, whose lateral walk is
// dist_geo * sin_geo: a phase of Re(q) times the lateral distance between the two points.
static inline std::complex<double> diag_medium_vbs(std::complex<double> es, double dist_refract,
                                                   double ct_traced, double dist_geo,
                                                   std::complex<double> cs, double f_hz, double ct_geo = -1.0)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> n = std::sqrt(es);
    double d = dist_refract * ct_traced;
    double s2 = 1.0 - ct_traced * ct_traced;
    double s = dist_refract * std::sqrt((s2 > 0.0) ? s2 : 0.0);

    std::complex<double> qa = n * n * (1.0 - cs * cs);
    if (std::imag(qa) == 0.0 && std::real(qa) < 0.0)
        qa = std::complex<double>(std::real(qa), -0.0);
    std::complex<double> q = std::sqrt(qa);

    double s_poly = s;
    if (ct_geo >= 0.0)
        s_poly = dist_geo * std::sqrt((1.0 - ct_geo * ct_geo > 0.0) ? 1.0 - ct_geo * ct_geo : 0.0);

    return std::exp(std::complex<double>(0.0, -1.0) * k0 * (n * cs * d + q * s + std::real(q) * (s_poly - s))) *
           std::exp(std::complex<double>(0.0, 1.0) * k0 * dist_geo);
}

// In-medium traversal factor with no VBS correction available: magnitude from the ITU-R P.2040
// path form, phase from the real index over the same path, both against the free-space reference.
static inline std::complex<double> diag_medium_geo(std::complex<double> es, double dist,
                                                   double dist_geo, double f_hz)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    double n_re = std::real(std::sqrt(es));
    return std::sqrt(diag_medium_gain(es, dist, f_hz)) *
           std::exp(std::complex<double>(0.0, -k0 * (n_re * dist - dist_geo)));
}

// In-medium traversal on the walk-off branch, in its general form. SCALE_BY_MEDIUM takes this
// branch whenever walk_off is true, which on a sheet crossing means the reflection pass:
//   sqrt(medium_gain(dist_refract)) * exp(-j k0 (n_re dist_refract ct_t^2 - dist_geo ct^2))
// diag_medium_geo is the ct_t = ct = 1 special case of it and cannot be used where the traced and
// geometric cosines differ. The palette carries no alpha and no mass term, so medium_gain is the
// ITU-R P.2040 path form over dist_refract alone.
static inline std::complex<double> diag_medium_walk(std::complex<double> es, double dist_refract,
                                                    double ct_traced, double dist_geo,
                                                    double ct_geo, double f_hz)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    double n_re = std::real(std::sqrt(es));
    double excess = k0 * (n_re * dist_refract * ct_traced * ct_traced - dist_geo * ct_geo * ct_geo);
    return std::sqrt(diag_medium_gain(es, dist_refract, f_hz)) *
           std::exp(std::complex<double>(0.0, -excess));
}

// Two coplanar triangles spanning +-half in y and z at x = x0, normal along -x, so a ray
// travelling towards +x makes a front-side hit and the crossing is air -> medium. Same winding as
// the builder in the ray_mesh_interact suite; the shared triangulation edge runs along y + z = 0,
// so a ray in the z = 0 plane must not cross at y = 0.
static inline arma::mat diag_quad_x(double half, double x0)
{
    arma::mat m(2, 9);
    m.row(0) = arma::rowvec{x0, -half, -half, x0, -half, half, x0, half, -half};
    m.row(1) = arma::rowvec{x0, half, -half, x0, -half, half, x0, half, half};
    return m;
}

// TX at (-1, 0.37, 0), RX placed so the ray crosses x = 0 at the given angle from the face normal.
// The offset in y keeps the crossing off the triangulation edge at normal incidence.
static inline void diag_ray_at(double theta_n_deg, arma::mat &orig, arma::mat &dest)
{
    double t = std::tan(theta_n_deg * QPI / 180.0);
    orig = arma::mat{{-1.0, 0.37, 0.0}};
    dest = arma::mat{{1.0, 0.37 + 2.0 * t, 0.0}};
}

// Exact analytic slab, TE/TM averaged power, air | slab | air. Independent of every library
// internal; the same closed form the ray_mesh_interact suite uses as its oracle. Delegates to the
// per-polarization form below so there is one copy of the closed form, not two.
static inline double diag_slab_p_avg(std::complex<double> eps_s, double thk,
                                     double theta_n_rad, double f_hz,
                                     std::complex<double> mu_s = std::complex<double>(1.0, 0.0))
{
    double p_te = 0.0, p_tm = 0.0;
    diag_slab_p_pol(eps_s, thk, theta_n_rad, f_hz, p_te, p_tm, mu_s);
    return 0.5 * (p_te + p_tm);
}

// Hand-built thin-sheet composite for a near | sheet | far stack, per polarization, in the form
// produce mode returns it. Shares no code with the library.
//
// The stack is three media and two interfaces. In raw amplitudes its transfer and its front
// reflection are the plane-parallel Airy pair
//
//   A_X = t_ns,X t_sf,X phi / (1 - r_sn,X r_sf,X phi^2)
//   R_X = r_ns,X + t_ns,X t_sn,X r_sf,X phi^2 / (1 - r_sn,X r_sf,X phi^2)
//
// An amplitude is not what a ray tube carries. The normal-flux transmittance of the pair the
// composite spans is |A_X|^2 Re(z_f c_f) / Re(z_n c_n), so the transmit port owes the square root
// of that ratio once, and only once: the flux inside the sheet cancels between the two internal
// crossings and never enters. That factor is fx_X below. The reflect port turns around in the near
// medium, where the ratio is Re(z_n c_n) / Re(z_n c_n) = 1, so it carries no such factor.
//
// Cosines, one per medium, each named where it is used:
//   c_n = ct, the real incidence cosine in the near medium
//   c_s = sqrt(1 - (e_n/e_s)(1 - ct^2)), the cosine in the sheet
//   c_f = sqrt(1 - (e_n/e_f)(1 - ct^2)), the cosine in the far medium
// c_f is written from the near-side sine directly rather than by chaining through the sheet. Snell
// is transitive - e_n sin^2_n = e_s sin^2_s = e_f sin^2_f - so the sheet drops out of it. That is
// what makes c_f, and therefore fx, the same quantity a bare near|far crossing would have used,
// and it is why a glued composite can be measured against an unglued reference at all.
//
// The transmit port also divides out the propagation of the medium the sheet displaces: a sheet has
// no world-space thickness, so the tracer walked thk of the host and the rest of the path already
// carries it. The host is the side of a glued pair that is not the solid - the near medium on an
// entry, the far medium on an exit.
//
// Permeability enters in exactly three places and nowhere else: the admittance becomes
// z = sqrt(eps/mu), every Snell ratio becomes a ratio of the products eps*mu, and the propagation
// index becomes n = sqrt(eps*mu). The Fresnel groupings, the Airy assembly, the Stokes sign and the
// flux conversion are already written in z and c and need no structural change. The three arguments
// default to 1, so every dielectric call site reads exactly as before.
//
// Three lumped columns are carried through the trailing DiagLump, which see: the interface gains,
// the sheet's transmission factor and a permittivity resonance on any of the three media. At its
// defaults - which is what diag_palette supplies - none of them executes anything.
//
// Preconditions that remain: alpha = 0 so the only in-sheet loss is the conductivity the complex
// index already carries, m = 0 so the mass-law route is not taken, and the displacement budget is
// uncapped so the full thickness is replaced. Under those conditions the in-slab cosine the caller
// passes is inert in produce mode - it reaches only excess_gain, which is 1 - so this reference
// does not take one.
struct DiagSheet
{
    std::complex<double> T_te, T_tm, R_te, R_tm;
    double fx_te = 1.0, fx_tm = 1.0;    // near -> far normal-flux conversion, transmit port only
    std::complex<double> c_slab, c_far; // the cosines the sheet and far terms are evaluated at
};

// Permittivity resonance of a material at f_hz, written from the column definition: a complex
// Lorentz pole resS * resF^2 / (resF^2 - f^2 + j (resF / resQ) f), all frequencies in GHz. Zero
// where any of the three columns disables it.
static inline std::complex<double> diag_res(double resF, double resQ, double resS, double f_hz)
{
    if (resF <= 0.0 || resQ <= 0.0 || resS == 0.0)
        return std::complex<double>(0.0, 0.0);
    double f = f_hz * 1e-9;
    return (resS * resF * resF) / std::complex<double>(resF * resF - f * f, (resF / resQ) * f);
}

// The lumped material columns a sheet composite can carry, as the reference takes them. Every
// field defaults to the value at which it is inert, and at those defaults both builders below
// execute the expressions they executed before this struct existed.
//
//   g_slab, g_far   Interface power gains, 10^(-att/10) with the coincidence term folded in. They
//                   multiply the transmit port only, as sqrt(g_slab * g_far). g_far is NOT gated
//                   inside the builder: the row states what the far medium is owed - its own gain
//                   where the wave newly enters it, 1 where the wave was already in it or leaves
//                   into air - so the gate the library applies is measured against a statement of
//                   the physics rather than against a copy of the test it is written as.
//
//   tf              The SHEET's transmission factor. tf is not a boundary admittance: it moves
//                   energy between the reflected and the transmitted port of a boundary after the
//                   Fresnel problem has been solved, so it has no place in the field-continuity
//                   form and none in a characteristic matrix. It does have one in the Airy form,
//                   which is written in per-boundary r and t, and that is the form this builder
//                   is in. Per boundary and per channel, with R0 = |r|^2 the raw reflectance:
//                       R' = R0 (1 - tf)             for tf >= 0, a leak out of the reflection
//                       R' = R0 + (1 - R0) (-tf)     for tf <  0, a forced reflection
//                       r' = r sqrt(R' / R0)         t' = t sqrt((1 - R') / (1 - R0))
//                   Each face is owed by the PAIR of media that form it, symmetrically: tf_near
//                   and tf_far are the neighbours' columns, and diag_tf_port combines them with
//                   the sheet's. With both at zero this is the sheet's own tf on both faces.
//                   Preconditions: 0 < R0 < 1 on every boundary and channel wherever tf < 0, so
//                   no exact Brewster angle, no matched boundary and no internal total reflection.
//
//   res_*           Permittivity resonances, as increments. A resonance is part of the
//                   permittivity: the admittance sqrt((eps + res)/mu), every cosine and the
//                   propagation index sqrt((eps + res) mu) all read it, one medium at its faces
//                   and inside it.
struct DiagLump
{
    double g_slab = 1.0, g_far = 1.0;
    double tf = 0.0;
    std::complex<double> res_near{0.0, 0.0}, res_slab{0.0, 0.0}, res_far{0.0, 0.0};
    double tf_near = 0.0, tf_far = 0.0; // the neighbours' tf: each face is owed by the PAIR of media forming it
};

// tf redistribution at one boundary, one channel: rescales the reflection in place and returns the
// factor the two transmission amplitudes of that boundary take.
//
// The rule is the symmetric two-medium one: a boundary is formed by two media and both act on it.
// With tf+ = max(tf, 0) and tf- = max(-tf, 0) on either side,
//     R_leak = R0 (1 - tfA+) (1 - tfB+)        R' = R_leak + (1 - R_leak) max(tfA-, tfB-)
// which is the single-medium rule whenever the other side carries no tf - bit for bit, since the
// extra factor is 1.0 and the extra term is a product with 0.0.
//
// A boundary with no forward Fresnel port, 1 - R0 below 1e-9, has nothing to rescale and returns
// 0 for the transmission factor; a caller that wants the flat leak of such a face reads it from
// diag_tf_pair directly.
static inline double diag_tf_pair(double R0, double tf_a, double tf_b)
{
    double ap = (tf_a > 0.0) ? tf_a : 0.0, am = (tf_a < 0.0) ? -tf_a : 0.0;
    double bp = (tf_b > 0.0) ? tf_b : 0.0, bm = (tf_b < 0.0) ? -tf_b : 0.0;
    double Rl = R0 * (1.0 - ap) * (1.0 - bp);
    return Rl + (1.0 - Rl) * ((am > bm) ? am : bm);
}
static inline double diag_tf_port(std::complex<double> &r, double tf, double tf_other = 0.0)
{
    double R0 = std::norm(r);
    double Rt = diag_tf_pair(R0, tf, tf_other);
    r = (R0 > 0.0) ? r * std::sqrt(Rt / R0) : std::complex<double>(0.0, 0.0);
    return (1.0 - R0 > 1.0e-9) ? std::sqrt((1.0 - Rt) / (1.0 - R0)) : 0.0;
}

// tf at the face of a coherent sheet: an ideal transformer of real ratio k = sqrt(nu) in front of
// the far side of the boundary. With beta = (1 - r) / (1 + r) the admittance ratio of the two
// sides, the face is the Fresnel face of the ratio nu beta, r' = (1 - nu beta) / (1 + nu beta), and
// both crossings take k (1 + r') / (1 + r). nu comes from the pair rule applied to the real
// reflection of the modulus of beta; for a real beta that is the pair rule on r itself, so the
// factor is diag_tf_port's. The flux across the face is continuous for any two media. Rescales the
// reflection in place and returns the factor of the two transmission amplitudes. A face with r of
// exactly +1 or -1 is left as it is.
static inline std::complex<double> diag_tf_face(std::complex<double> &r, double tf, double tf_other = 0.0)
{
    const std::complex<double> r0 = r, beta = (1.0 - r) / (1.0 + r);
    const double x0 = std::abs(beta);
    if (!(x0 > 0.0) || std::isinf(x0))
        return 1.0;
    const double rm = (1.0 - x0) / (1.0 + x0), Rm = diag_tf_pair(rm * rm, tf, tf_other), s = (rm < 0.0) ? -std::sqrt(Rm) : std::sqrt(Rm);
    if (!(1.0 + s > 0.0)) // forced shut towards the denser side
    {
        r = -1.0;
        return 0.0;
    }
    const double nu = ((1.0 - s) / (1.0 + s)) / x0;
    r = (1.0 - nu * beta) / (1.0 + nu * beta);
    return std::sqrt(nu) * (1.0 + r) / (1.0 + r0);
}

// tf at a closed face: a matched resistive pad in front of the reactance behind the face, formed on
// the modulus of the immittance y of the side the wave is on - z c for TE and the scalar type,
// conj(z) c for TM, the form r is written in. With b the far side over |y|, read as a reactance,
// the load is |y| (nu + b) / (1 + nu b), nu = (1 - s) / (1 + s), s^2 the pair rule on a
// reflectance of one. Rescales the reflection in place and returns the amplitude the face passes
// per unit forward amplitude, in flux. A pair that leaks nothing leaves the face raw.
static inline double diag_closed_face(std::complex<double> &r, std::complex<double> y, double tf, double tf_other = 0.0)
{
    const double s = std::sqrt(diag_tf_pair(1.0, tf, tf_other)), nu = (1.0 - s) / (1.0 + s), a = std::abs(y);
    if (!(nu > 0.0) || !(a > 0.0))
        return 0.0;
    const std::complex<double> e = y / a, b(0.0, std::imag(e * (1.0 - r) / (1.0 + r))), load = (nu + b) / (1.0 + nu * b);
    r = (e - load) / (e + load);
    return std::abs(1.0 + r) * std::sqrt(std::real(load) * a);
}

static inline DiagSheet diag_sheet_composite(std::complex<double> e_near, std::complex<double> e_slab,
                                             std::complex<double> e_far, double thk, double ct,
                                             double f_hz, bool disp_far, bool scalar,
                                             std::complex<double> mu_near = std::complex<double>(1.0, 0.0),
                                             std::complex<double> mu_slab = std::complex<double>(1.0, 0.0),
                                             std::complex<double> mu_far = std::complex<double>(1.0, 0.0),
                                             const DiagLump &lump = DiagLump())
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    auto root = [](std::complex<double> a)
    {
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
        return std::sqrt(a);
    };

    // The permittivities with their resonance, at the faces and in propagation.
    const std::complex<double> i_near = e_near + lump.res_near, i_slab = e_slab + lump.res_slab,
                               i_far = e_far + lump.res_far;

    const std::complex<double> zn = std::sqrt(i_near / mu_near), zs = std::sqrt(i_slab / mu_slab),
                               zf = std::sqrt(i_far / mu_far);
    const std::complex<double> w_near = i_near * mu_near; // the transverse invariant's anchor
    const double s2 = 1.0 - ct * ct;
    const std::complex<double> c1(ct, 0.0);
    const std::complex<double> cs = root(1.0 - (w_near / (i_slab * mu_slab)) * s2);
    const std::complex<double> cf = root(1.0 - (w_near / (i_far * mu_far)) * s2);

    DiagSheet D;
    D.c_slab = cs, D.c_far = cf;

    // near | sheet, at c1 in the near medium and cs in the sheet
    std::complex<double> r_ns_te = (zn * c1 - zs * cs) / (zn * c1 + zs * cs);
    std::complex<double> t_ns_te = (2.0 * zn * c1) / (zn * c1 + zs * cs);
    std::complex<double> t_sn_te = (2.0 * zs * cs) / (zs * cs + zn * c1);
    std::complex<double> r_ns_tm = scalar ? r_ns_te : (zs * c1 - zn * cs) / (zs * c1 + zn * cs);
    std::complex<double> t_ns_tm = scalar ? t_ns_te : (2.0 * zn * c1) / (zs * c1 + zn * cs);
    std::complex<double> t_sn_tm = scalar ? t_sn_te : (2.0 * zs * cs) / (zn * cs + zs * c1);

    // sheet | far, at cs in the sheet and cf in the far medium
    std::complex<double> r_sf_te = (zs * cs - zf * cf) / (zs * cs + zf * cf);
    std::complex<double> t_sf_te = (2.0 * zs * cs) / (zs * cs + zf * cf);
    std::complex<double> r_sf_tm = scalar ? r_sf_te : (zf * cs - zs * cf) / (zf * cs + zs * cf);
    std::complex<double> t_sf_tm = scalar ? t_sf_te : (2.0 * zs * cs) / (zf * cs + zs * cf);

    // One-way propagation across the sheet, at cs, and the host propagation it replaces, taken at
    // the host's own cosine over the same normal thickness.
    //
    const std::complex<double> phi =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(i_slab * mu_slab) * thk * cs);
    const std::complex<double> i_disp = disp_far ? i_far : i_near;
    const std::complex<double> mu_disp = disp_far ? mu_far : mu_near;
    const std::complex<double> c_disp = root(1.0 - (w_near / (i_disp * mu_disp)) * s2);
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(i_disp * mu_disp) * thk * c_disp);
    const std::complex<double> phi_t = phi / phi_disp;

    // The sheet's tf, on both of its faces. Skipped outright at tf = 0, so that case executes
    // nothing it did not execute before.
    //
    // Each face whose pair carries tf is a transformer, propagating or tunnelling. A far face with
    // no wave behind it - a lossless far medium past the critical angle of the outer pair - is the
    // pad of a closed face, at the Fresnel magnitude of its reflection, as in the library.
    if (lump.tf != 0.0 || lump.tf_near != 0.0 || lump.tf_far != 0.0)
    {
        const bool n = lump.tf != 0.0 || lump.tf_near != 0.0, fr = lump.tf != 0.0 || lump.tf_far != 0.0;
        const bool closed = std::imag(i_far * mu_far) == 0.0 && std::abs((w_near / (i_far * mu_far)) * s2) >= 1.0;
        const std::complex<double> one(1.0, 0.0);
        std::complex<double> kn_te = n ? diag_tf_face(r_ns_te, lump.tf, lump.tf_near) : one, kn_tm = n ? diag_tf_face(r_ns_tm, lump.tf, lump.tf_near) : one;
        std::complex<double> kf_te = one, kf_tm = one;
        if (fr && closed)
        {
            diag_closed_face(r_sf_te, zs * cs, lump.tf, lump.tf_far), diag_closed_face(r_sf_tm, scalar ? zs * cs : std::conj(zs) * cs, lump.tf, lump.tf_far);
        }
        else if (fr)
            kf_te = diag_tf_face(r_sf_te, lump.tf, lump.tf_far), kf_tm = diag_tf_face(r_sf_tm, lump.tf, lump.tf_far);
        t_ns_te *= kn_te, t_sn_te *= kn_te, t_sf_te *= kf_te;
        t_ns_tm *= kn_tm, t_sn_tm *= kn_tm, t_sf_tm *= kf_tm;
    }

    // Near -> far normal flux, per channel. Both denominators collapse to Re(z_n) * ct for a real
    // incidence cosine, so with a real near side the two channels can separate only through the far
    // medium. The gate is near == far, which is why an isolated sheet and a sheet embedded in a
    // medium both take the factor as exactly 1.
    if (e_far != e_near || mu_far != mu_near || lump.res_far != lump.res_near)
    {
        double den_te = std::real(zn * c1), den_tm = std::real(std::conj(zn) * c1);
        double num_te = std::real(zf * cf), num_tm = std::real(std::conj(zf) * cf);
        D.fx_te = (den_te > 0.0 && num_te > 0.0) ? std::sqrt(num_te / den_te) : 0.0;
        D.fx_tm = (den_tm > 0.0 && num_tm > 0.0) ? std::sqrt(num_tm / den_tm) : 0.0;
        if (scalar)
            D.fx_tm = D.fx_te;
    }

    // The internal series runs between the two sheet-side reflections, and r_sn = -r_ns.
    std::complex<double> p2 = phi * phi;
    std::complex<double> den_te = 1.0 + r_ns_te * r_sf_te * p2;
    std::complex<double> den_tm = 1.0 + r_ns_tm * r_sf_tm * p2;

    // The reflect port is handed to a ray that turns around at the mesh plane, so it owes the
    // round trip through the part of the layer in front of that plane: half of it where one
    // medium is on both sides, all of it where the layer lies in the near medium in front of a
    // solid, none where it lies beyond. And what entered the sheet and came back paid its gain.
    const bool same = !(e_far != e_near || mu_far != mu_near || lump.res_far != lump.res_near);
    const std::complex<double> front = same ? 1.0 / phi_disp : (disp_far ? std::complex<double>(1.0, 0.0) : 1.0 / (phi_disp * phi_disp));
    const double ge = std::sqrt(lump.g_slab);

    D.T_te = D.fx_te * t_ns_te * t_sf_te * phi_t / den_te;
    D.T_tm = D.fx_tm * t_ns_tm * t_sf_tm * phi_t / den_tm;
    D.R_te = front * (r_ns_te + ge * t_ns_te * t_sn_te * r_sf_te * p2 / den_te);
    D.R_tm = front * (r_ns_tm + ge * t_ns_tm * t_sn_tm * r_sf_tm * p2 / den_tm);

    // Lumped interface gains on the transmit port, once per medium entered.
    if (lump.g_slab != 1.0 || lump.g_far != 1.0)
    {
        double ig = std::sqrt(lump.g_slab * lump.g_far);
        D.T_te *= ig, D.T_tm *= ig;
    }
    return D;
}

// The same composite written at a complex incidence cosine.
//
// diag_sheet_composite takes a real ct because every stack it was built for is reached from air or
// from a medium the harness enters at the geometric angle. A sheet reached from inside a traced
// solid has no real incidence cosine: the propagation cosine there is complex and the traced
// direction carries only its real projection. The form is unchanged and every expression below is
// analytic in ct, so passing a real cosine reproduces the function above term for term; the fixture
// asserts that on every row rather than assuming it.
//
// Snell transitivity is what makes this well posed. The transverse invariant is e mu (1 - c^2),
// conserved across every interface, so writing c_slab and c_far from the near-side cosine gives the
// same two numbers the entry medium's own cosine would have given. With the exact complex c_near
// they are the cosines a bare air | slab and air | far crossing would use; with the real projection
// they are not, and that is the whole of what this reference measures.
//
// Same preconditions as diag_sheet_composite, and the same three permeability substitutions. Neither
// s2 nor c1 reads mu, so the generalization is identical in the two builders and the statement that
// this one reproduces the other at a real cosine is unaffected by it.
//
// Same preconditions otherwise: no resonance, att = tf = alpha = 0, m = 0,
// and an uncapped displacement budget.
static inline DiagSheet diag_sheet_composite_cx(std::complex<double> e_near, std::complex<double> e_slab,
                                                std::complex<double> e_far, double thk,
                                                std::complex<double> ct, double f_hz,
                                                bool disp_far, bool scalar,
                                                std::complex<double> mu_near = std::complex<double>(1.0, 0.0),
                                                std::complex<double> mu_slab = std::complex<double>(1.0, 0.0),
                                                std::complex<double> mu_far = std::complex<double>(1.0, 0.0))
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    auto root = [](std::complex<double> a)
    {
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
        return std::sqrt(a);
    };

    const std::complex<double> zn = std::sqrt(e_near / mu_near), zs = std::sqrt(e_slab / mu_slab),
                               zf = std::sqrt(e_far / mu_far);
    const std::complex<double> w_near = e_near * mu_near; // the transverse invariant's anchor
    const std::complex<double> c1 = ct;
    const std::complex<double> s2 = 1.0 - c1 * c1;
    const std::complex<double> cs = root(1.0 - (w_near / (e_slab * mu_slab)) * s2);
    const std::complex<double> cf = root(1.0 - (w_near / (e_far * mu_far)) * s2);

    DiagSheet D;
    D.c_slab = cs, D.c_far = cf;

    std::complex<double> r_ns_te = (zn * c1 - zs * cs) / (zn * c1 + zs * cs);
    std::complex<double> t_ns_te = (2.0 * zn * c1) / (zn * c1 + zs * cs);
    std::complex<double> t_sn_te = (2.0 * zs * cs) / (zs * cs + zn * c1);
    std::complex<double> r_ns_tm = scalar ? r_ns_te : (zs * c1 - zn * cs) / (zs * c1 + zn * cs);
    std::complex<double> t_ns_tm = scalar ? t_ns_te : (2.0 * zn * c1) / (zs * c1 + zn * cs);
    std::complex<double> t_sn_tm = scalar ? t_sn_te : (2.0 * zs * cs) / (zn * cs + zs * c1);

    std::complex<double> r_sf_te = (zs * cs - zf * cf) / (zs * cs + zf * cf);
    std::complex<double> t_sf_te = (2.0 * zs * cs) / (zs * cs + zf * cf);
    std::complex<double> r_sf_tm = scalar ? r_sf_te : (zf * cs - zs * cf) / (zf * cs + zs * cf);
    std::complex<double> t_sf_tm = scalar ? t_sf_te : (2.0 * zs * cs) / (zf * cs + zs * cf);

    const std::complex<double> phi =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_slab * mu_slab) * thk * cs);
    const std::complex<double> e_disp = disp_far ? e_far : e_near;
    const std::complex<double> mu_disp = disp_far ? mu_far : mu_near;
    const std::complex<double> c_disp = root(1.0 - (w_near / (e_disp * mu_disp)) * s2);
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_disp * mu_disp) * thk * c_disp);
    const std::complex<double> phi_t = phi / phi_disp;

    // Per-channel normal flux, near to far. With a complex incidence cosine the two denominators
    // Re(z_n c_n) and Re(conj(z_n) c_n) no longer collapse onto Re(z_n) ct, so the channels separate
    // through the near medium as well as through the far one.
    if (e_far != e_near || mu_far != mu_near)
    {
        double den_te = std::real(zn * c1), den_tm = std::real(std::conj(zn) * c1);
        double num_te = std::real(zf * cf), num_tm = std::real(std::conj(zf) * cf);
        D.fx_te = (den_te > 0.0 && num_te > 0.0) ? std::sqrt(num_te / den_te) : 0.0;
        D.fx_tm = (den_tm > 0.0 && num_tm > 0.0) ? std::sqrt(num_tm / den_tm) : 0.0;
        if (scalar)
            D.fx_tm = D.fx_te;
    }

    std::complex<double> p2 = phi * phi;
    std::complex<double> den_te = 1.0 + r_ns_te * r_sf_te * p2;
    std::complex<double> den_tm = 1.0 + r_ns_tm * r_sf_tm * p2;

    // The round trip through the part of the layer in front of the mesh plane, as above
    const bool same = !(e_far != e_near || mu_far != mu_near);
    const std::complex<double> front = same ? 1.0 / phi_disp : (disp_far ? std::complex<double>(1.0, 0.0) : 1.0 / (phi_disp * phi_disp));

    D.T_te = D.fx_te * t_ns_te * t_sf_te * phi_t / den_te;
    D.T_tm = D.fx_tm * t_ns_tm * t_sf_tm * phi_t / den_tm;
    D.R_te = front * (r_ns_te + t_ns_te * t_sn_te * r_sf_te * p2 / den_te);
    D.R_tm = front * (r_ns_tm + t_ns_tm * t_sn_tm * r_sf_tm * p2 / den_tm);
    return D;
}

// One layer of a stratified stack: a complex permittivity and permeability pair and a normal
// thickness. Permeability is carried from the start even though nothing supplies it yet, because
// issue 3 generalizes the two builders above the same way and the core is meant to inherit a
// reference that already has it.
struct DiagLayer
{
    std::complex<double> e{1.0, 0.0};
    std::complex<double> mu{1.0, 0.0};
    double d = 0.0;                  // normal thickness, m
    std::complex<double> res{0.0, 0.0}; // permittivity resonance, part of the permittivity, see DiagLump
    double alpha = 0.0;              // in-medium absorption, dB per meter of path
    double tf = 0.0;                 // transmission factor; read by diag_sc_stack only
    double m = 0.0;                  // mass-law slope, dB/decade; read by diag_sc_stack only
    double fRef = 1.0;               // reference frequency of the mass-law argument, GHz
    double g = 1.0;                  // interface power gain, on the reflect port's echo; diag_sc_stack only
};

// Hand-built 2x2 characteristic-matrix reference for an ordered stack of layers between two
// half-spaces. Shares no code with the library and no code with the two single-layer builders
// above, which it has to reproduce.
//
// The variables. Per medium the admittance is z = sqrt(eps/mu) and the index is n = sqrt(eps mu),
// and every cosine comes from the transverse invariant eps mu (1 - c^2), which is conserved across
// every boundary and is fixed by the incidence cosine in the near medium. Per polarization the
// tilted admittance is
//
//     eta_TE = z c        eta_TM = z / c
//
// which are the groupings the Fresnel expressions in the library are already written in: the TE
// reflection is (eta1 - eta2)/(eta1 + eta2) exactly, and dividing the TM reflection through by
// c1 c2 turns it into (eta2 - eta1)/(eta2 + eta1), so the library's TM reflection is minus the
// admittance-form one and its TM transmission carries an extra c1/c2. Both conversions are applied
// at the ends below.
//
// What propagates. Tangential E and tangential H are continuous across a boundary; normal flux is
// not, since it decays inside a lossy layer. So the state that the matrix product carries is the
// amplitude pair, and the conversion to per-channel normal flux is applied once, at the outer pair,
// on the transmit port. Inside the product the intermediate admittances cancel identically, which
// is the same statement as the library's "the intermediate slab flux cancels between the two
// crossings and never appears in the composite", here for any layer count.
//
// The layer matrix for thickness d and phase delta = k0 n c d is
//
//     [ cos delta            (i / eta) sin delta ]
//     [ i eta sin delta       cos delta          ]
//
// and with M the product in propagation order, [B; C] = M [1; eta_far],
//
//     r = (eta_near B - C) / (eta_near B + C)      t = 2 eta_near / (eta_near B + C)
//
// At one layer this reduces to the Airy pair term for term, which is asserted against both
// builders above rather than assumed.
//
// The displaced host is not a matrix quantity. A sheet has no world-space thickness, so the
// transmit port divides out the propagation of the medium the stack displaces over the total
// normal thickness, capped by disp_budget exactly as produce mode caps it. Pass disp_budget = 0.0
// where the reference is wanted undisplaced, which is what the reflect port always is.
//
// A permittivity resonance is carried, per layer through DiagLayer::res and on the two half-spaces
// through the trailing arguments, as part of the permittivity, as in DiagLump. The interface gains are an end quantity on
// the transmit port and are left to the caller, who multiplies by sqrt of their product.
//
// tf has no entry here and cannot have one. This form states the boundary conditions - continuity
// of the tangential field pair - and has no per-boundary reflection or transmission to rescale; a
// boundary that has been through tf no longer satisfies 1 + r = t, so it is not a boundary this
// matrix can describe. The mass-law column is not carried either.
//
// alpha is carried, per layer. It is a real one-way amplitude factor 10^(-alpha d / (20 |Re c|))
// over the path through the layer, and the layer phase is where a one-way factor lives: the
// matrix is cos and sin of delta, and the one-way propagation is exp(-j delta). The admittance
// does not read it.
static inline DiagSheet diag_cm_stack(std::complex<double> e_near, std::complex<double> mu_near,
                                      const std::vector<DiagLayer> &layers,
                                      std::complex<double> e_far, std::complex<double> mu_far,
                                      std::complex<double> ct, double f_hz,
                                      bool disp_far, bool scalar, double disp_budget = -1.0,
                                      std::complex<double> res_near = std::complex<double>(0.0, 0.0),
                                      std::complex<double> res_far = std::complex<double>(0.0, 0.0))
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    auto root = [](std::complex<double> a)
    {
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
        return std::sqrt(a);
    };

    // The permittivities with their resonance
    const std::complex<double> i_near = e_near + res_near, i_far = e_far + res_far;

    const std::complex<double> w_near = i_near * mu_near; // the transverse invariant's anchor
    const std::complex<double> s2 = 1.0 - ct * ct;
    const std::complex<double> zn = std::sqrt(i_near / mu_near), zf = std::sqrt(i_far / mu_far);
    const std::complex<double> cf = root(1.0 - (w_near / (i_far * mu_far)) * s2);

    DiagSheet D;
    D.c_far = cf;
    D.c_slab = layers.empty() ? ct
                              : root(1.0 - (w_near / ((layers[0].e + layers[0].res) * layers[0].mu)) * s2);

    std::complex<double> r_pol[2], t_pol[2];
    for (int pol = 0; pol < 2; ++pol)
    {
        auto adm = [&](std::complex<double> z, std::complex<double> c)
        { return (pol == 0) ? (z * c) : (z / c); };

        std::complex<double> M11(1.0, 0.0), M12(0.0, 0.0), M21(0.0, 0.0), M22(1.0, 0.0);
        for (const DiagLayer &L : layers)
        {
            std::complex<double> ij = L.e + L.res;
            std::complex<double> cj = root(1.0 - (w_near / (ij * L.mu)) * s2);
            std::complex<double> nj = std::sqrt(ij * L.mu);
            std::complex<double> dl = k0 * nj * cj * L.d;
            if (L.alpha != 0.0) // nepers of amplitude over the path d / |Re c|, into the layer phase
                dl -= std::complex<double>(0.0, (std::log(10.0) / 20.0) * L.alpha * L.d / std::abs(std::real(cj)));
            std::complex<double> ej = adm(std::sqrt(ij / L.mu), cj);

            std::complex<double> A = std::cos(dl);
            std::complex<double> B = std::complex<double>(0.0, 1.0) * std::sin(dl) / ej;
            std::complex<double> Cc = std::complex<double>(0.0, 1.0) * ej * std::sin(dl);
            std::complex<double> Dd = std::cos(dl);

            std::complex<double> n11 = M11 * A + M12 * Cc, n12 = M11 * B + M12 * Dd;
            std::complex<double> n21 = M21 * A + M22 * Cc, n22 = M21 * B + M22 * Dd;
            M11 = n11, M12 = n12, M21 = n21, M22 = n22;
        }

        std::complex<double> y0 = adm(zn, ct), ys = adm(zf, cf);
        std::complex<double> B = M11 + M12 * ys, Cc = M21 + M22 * ys;
        std::complex<double> den = y0 * B + Cc;
        r_pol[pol] = (y0 * B - Cc) / den;
        t_pol[pol] = 2.0 * y0 / den;
    }

    // Into the library's amplitude convention: the TM reflection sign, and the TM cosine ratio,
    // which telescopes over the stack because the intermediate cosines cancel in the product.
    D.R_te = r_pol[0];
    D.R_tm = scalar ? r_pol[0] : -r_pol[1];
    std::complex<double> t_te = t_pol[0];
    std::complex<double> t_tm = scalar ? t_pol[0] : (ct / cf) * t_pol[1];

    // The one conversion to per-channel normal flux, at the outer pair, transmit port only.
    if (e_far != e_near || mu_far != mu_near || res_far != res_near)
    {
        double den_te = std::real(zn * ct), den_tm = std::real(std::conj(zn) * ct);
        double num_te = std::real(zf * cf), num_tm = std::real(std::conj(zf) * cf);
        D.fx_te = (den_te > 0.0 && num_te > 0.0) ? std::sqrt(num_te / den_te) : 0.0;
        D.fx_tm = (den_tm > 0.0 && num_tm > 0.0) ? std::sqrt(num_tm / den_tm) : 0.0;
        if (scalar)
            D.fx_tm = D.fx_te;
    }

    double t_tot = 0.0;
    for (const DiagLayer &L : layers)
        t_tot += L.d;
    double t_disp = (disp_budget >= 0.0 && disp_budget < t_tot) ? disp_budget : t_tot;
    if (t_disp < 0.0)
        t_disp = 0.0;

    const std::complex<double> i_d = disp_far ? i_far : i_near;
    const std::complex<double> mu_d = disp_far ? mu_far : mu_near;
    const std::complex<double> c_disp = root(1.0 - (w_near / (i_d * mu_d)) * s2);
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(i_d * mu_d) * t_disp * c_disp);

    D.T_te = D.fx_te * t_te / phi_disp;
    D.T_tm = D.fx_tm * t_tm / phi_disp;

    // The reflect port: the round trip through the part of the stack in front of the mesh plane
    const bool same = !(e_far != e_near || mu_far != mu_near || res_far != res_near);
    const std::complex<double> front = same ? 1.0 / phi_disp : (disp_far ? std::complex<double>(1.0, 0.0) : 1.0 / (phi_disp * phi_disp));
    D.R_te *= front, D.R_tm *= front;
    return D;
}

// The mass-law term of Material::medium_gain, in dB, written from the column definition:
//     A  = m log10((f / fRef) M)        X0 = ((f / fRef) M)^(m / 20)
//     TL = A log10(1 + (X0 c)^2) / log10(1 + X0^2)
// with M the normal thickness and c the real incidence cosine. Zero below an argument of one.
static inline double diag_mass_TL(double m, double fRef_GHz, double f_hz, double thk, double cosv)
{
    double arg = (f_hz * 1e-9 / fRef_GHz) * thk;
    double A = (m > 0.0 && arg > 0.0) ? m * std::log10(arg) : 0.0;
    if (!(A > 0.0) || !(cosv > 0.0))
        return 0.0;
    if (cosv >= 1.0)
        return A;
    double X0 = std::pow(arg, 0.05 * m), x = X0 * cosv;
    return A * std::log10(1.0 + x * x) / std::log10(1.0 + X0 * X0);
}

// Scattering-form reference for an ordered stack, carrying a tf per medium and the mass-law column.
// Shares no code with the library and none with diag_cm_stack, which it has to reproduce at tf = 0.
//
// A characteristic matrix states field continuity and has no per-boundary reflection to rescale.
// This form does. Each boundary a|b is a two-port in the library's own amplitude convention,
//     r, t_ab, t_ba, and r_ba = -r,
// rescaled by diag_tf_port under the pair rule with the tf of the two media that form it, and each
// layer is a one-way factor phi = exp(-j k0 n d c) times its alpha term. The stack is the product
// of the wave-transfer matrices B_1 P_1 B_2 ... B_{N+1}, evaluated here by recursion from the far
// end so that a closed boundary (t = 0 at tf = -1) and an opaque layer need no inverse:
//     G_j = r_j + t_j tb_j G_{j+1} phi^2 / (1 + r_j G_{j+1} phi^2)
//     T_j = t_j phi T_{j+1} / (1 + r_j G_{j+1} phi^2)
// At one layer that is the Airy pair of diag_sheet_composite term for term.
//
// A mass-law layer, m > 0, is not a coherent layer: no series and no in-layer phase. It ends the
// coherent segment in front of it as a half-space of its own admittance, contributes one real
// amplitude factor - the conductivity loss and its alpha over d / mass_cos, the path the held wave
// takes, and the mass term at the real incidence cosine mass_cos - and starts the next segment as
// its incidence half-space. Nothing returns across it, so the reflect port is that of the first
// segment alone. Mass-law layers with no coherent layer between them are one plate (the plate rule,
// md/acoustic.md 10 N1): its faces are the transformer, the faces inside it pass what reaches them and
// reflect nothing, its first face passes a (its flux transmittance), and its last face puts the whole
// right so that the plate passes a tau / (a + tau - a tau), a the smaller of what its two outer faces
// pass, tau the product of its layers' factors. With no coherent layer at all this is the mass-law
// route of a single sheet.
//
// End quantities as in diag_cm_stack: one flux conversion at the outer pair, and the displaced
// host over the COHERENT thickness only, since a mass-law sheet displaces nothing. Interface gains
// are left to the caller.
static inline DiagSheet diag_sc_stack(std::complex<double> e_near, std::complex<double> mu_near, double tf_near,
                                      const std::vector<DiagLayer> &layers,
                                      std::complex<double> e_far, std::complex<double> mu_far, double tf_far,
                                      std::complex<double> ct, double f_hz,
                                      bool disp_far, bool scalar, double disp_budget = -1.0,
                                      std::complex<double> res_near = std::complex<double>(0.0, 0.0),
                                      std::complex<double> res_far = std::complex<double>(0.0, 0.0),
                                      double mass_cos = -1.0)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    auto root = [](std::complex<double> a)
    {
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
        return std::sqrt(a);
    };
    const size_t N = layers.size();
    if (mass_cos < 0.0)
        mass_cos = std::abs(std::real(ct));

    // Media 0 .. N+1: interface admittance, cosine, tf
    std::vector<std::complex<double>> z(N + 2), c(N + 2);
    std::vector<double> tfm(N + 2);
    const std::complex<double> i_near = e_near + res_near, i_far = e_far + res_far;
    const std::complex<double> w_near = i_near * mu_near, s2 = 1.0 - ct * ct;
    z[0] = std::sqrt(i_near / mu_near), c[0] = ct, tfm[0] = tf_near;
    for (size_t j = 0; j < N; ++j)
    {
        std::complex<double> ij = layers[j].e + layers[j].res;
        z[j + 1] = std::sqrt(ij / layers[j].mu);
        c[j + 1] = root(1.0 - (w_near / (ij * layers[j].mu)) * s2);
        tfm[j + 1] = layers[j].tf;
    }
    z[N + 1] = std::sqrt(i_far / mu_far), c[N + 1] = root(1.0 - (w_near / (i_far * mu_far)) * s2);
    tfm[N + 1] = tf_far;

    DiagSheet D;
    D.c_far = c[N + 1], D.c_slab = c[(N > 0) ? 1 : 0];

    std::complex<double> Tp[2], Rp[2];
    for (int pol = 0; pol < 2; ++pol)
    {
        bool tm = (pol == 1) && !scalar;
        double a_entry = -1.0, a_run = -1.0, tau_run = 1.0; // what the plate's first face passed, and its layers' factors
        auto bnd = [&](size_t j, std::complex<double> &r, std::complex<double> &t, std::complex<double> &tb)
        {
            std::complex<double> a = z[j] * c[j], b = z[j + 1] * c[j + 1];
            if (!tm)
                r = (a - b) / (a + b), t = 2.0 * a / (a + b), tb = 2.0 * b / (a + b);
            else
            {
                std::complex<double> p = z[j + 1] * c[j], q = z[j] * c[j + 1];
                r = (p - q) / (p + q), t = 2.0 * a / (p + q), tb = 2.0 * b / (p + q);
            }
            const bool mA = j >= 1 && layers[j - 1].m > 0.0, mB = j < N && layers[j].m > 0.0;
            const std::complex<double> ya = tm ? std::conj(z[j]) * c[j] : a, yb = tm ? std::conj(z[j + 1]) * c[j + 1] : b;
            if (mA && mB) // inside one plate: what reaches the face passes, nothing is reflected
            {
                const double k = std::sqrt(std::real(ya) / std::real(yb));
                r = 0.0, t = std::polar(k, std::arg(t)), tb = std::polar(1.0 / k, std::arg(tb));
                return;
            }
            if (tfm[j] != 0.0 || tfm[j + 1] != 0.0)
            {
                std::complex<double> k = diag_tf_face(r, tfm[j], tfm[j + 1]);
                t *= k, tb *= k;
            }
            if (mB) // the plate's first face: what it passes into the plate
                a_entry = std::norm(t) * std::real(yb) / std::real(ya);
            else if (mA) // its last face: what reached it passes, put right by the plate rule
            {
                const double k = (std::real(yb) > 0.0) ? std::sqrt(std::real(ya) / std::real(yb)) : 0.0;
                const double a_out = std::norm(tb) * std::real(ya) / std::real(yb), am = std::min(a_run, a_out);
                const double fix = (a_run > 0.0 && tau_run > 0.0) ? std::sqrt((am * tau_run / (am + tau_run - am * tau_run)) / (a_run * tau_run)) : 1.0;
                t = std::polar(k * fix, std::arg(t));
            }
        };
        // Coherent segment between media p and q, layers p+1 .. q-1
        // G is the reflection the series runs on; Ge the one handed out, in which what entered a
        // layer and came back paid that layer's interface gain once
        auto seg = [&](size_t p, size_t q, std::complex<double> &Gout, std::complex<double> &T)
        {
            std::complex<double> r, t, tb, G, Ge;
            bnd(q - 1, G, T, tb);
            Ge = G;
            for (size_t j = q - 1; j-- > p;)
            {
                const DiagLayer &L = layers[j]; // the layer behind boundary j is medium j+1
                std::complex<double> phi =
                    std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt((L.e + L.res) * L.mu) * L.d * c[j + 1]);
                if (L.alpha != 0.0)
                    phi *= std::pow(10.0, -L.alpha * L.d / (20.0 * std::abs(std::real(c[j + 1]))));
                bnd(j, r, t, tb);
                std::complex<double> den = 1.0 + r * G * phi * phi;
                T = t * phi * T / den;
                Ge = r + std::sqrt(L.g) * t * tb * Ge * phi * phi / den;
                G = r + t * tb * G * phi * phi / den;
            }
            Gout = Ge;
        };

        // Mass-law layers with no coherent layer between them are bonded and are one plate: its
        // impedance ratio is the sum of theirs, X0 = ((f / fRef) d)^(m / 20) each, and the plate
        // pays the single-body term of that sum once. A lone layer keeps diag_mass_TL.
        auto plate_TL = [&](double X)
        {
            if (!(X > 1.0) || !(mass_cos > 0.0))
                return 0.0;
            double A = 20.0 * std::log10(X), x = X * mass_cos;
            return (mass_cos >= 1.0) ? A : A * std::log10(1.0 + x * x) / std::log10(1.0 + X * X);
        };
        std::complex<double> Tt(1.0, 0.0), Rt(0.0, 0.0), G, T;
        size_t p = 0, run = 0; // run: mass-law layers in the plate so far
        double X_plate = 0.0, paid_dB = 0.0;
        bool first = true;
        for (size_t q = 1; q <= N + 1; ++q)
        {
            bool mass = (q <= N) && layers[q - 1].m > 0.0;
            if (q <= N && !mass)
                continue;
            seg(p, q, G, T);
            Tt *= T;
            if (first)
                Rt = G, first = false;
            if (mass && (q - p > 1 || p == 0 || !(layers[p - 1].m > 0.0))) // a plate begins
                a_run = a_entry, tau_run = 1.0;
            if (mass)
            {
                const DiagLayer &L = layers[q - 1];
                double dist = L.d / mass_cos; // the held path
                if (q - p > 1) // a coherent layer in between: a new plate
                    run = 0, X_plate = 0.0, paid_dB = 0.0;
                double mass_dB;
                if (run == 0)
                    mass_dB = diag_mass_TL(L.m, L.fRef, f_hz, L.d, mass_cos);
                else
                    mass_dB = plate_TL(X_plate + std::pow((f_hz * 1e-9 / L.fRef) * L.d, 0.05 * L.m)) - paid_dB;
                X_plate += std::pow((f_hz * 1e-9 / L.fRef) * L.d, 0.05 * L.m), paid_dB += mass_dB, ++run;
                double loss_dB = L.alpha * dist + mass_dB;
                const double g = diag_medium_gain((L.e + L.res) * L.mu, dist, f_hz) * std::pow(10.0, -0.1 * loss_dB);
                tau_run *= g;
                Tt *= std::sqrt(g);
            }
            p = q;
        }
        Tp[pol] = Tt, Rp[pol] = Rt;
        if (scalar)
        {
            Tp[1] = Tp[0], Rp[1] = Rp[0];
            break;
        }
    }

    if (e_far != e_near || mu_far != mu_near || res_far != res_near)
    {
        double den_te = std::real(z[0] * ct), den_tm = std::real(std::conj(z[0]) * ct);
        double num_te = std::real(z[N + 1] * c[N + 1]), num_tm = std::real(std::conj(z[N + 1]) * c[N + 1]);
        D.fx_te = (den_te > 0.0 && num_te > 0.0) ? std::sqrt(num_te / den_te) : 0.0;
        D.fx_tm = (den_tm > 0.0 && num_tm > 0.0) ? std::sqrt(num_tm / den_tm) : 0.0;
        if (scalar)
            D.fx_tm = D.fx_te;
    }

    double t_coh = 0.0;
    for (const DiagLayer &L : layers)
        if (!(L.m > 0.0))
            t_coh += L.d;
    double t_disp = (disp_budget >= 0.0 && disp_budget < t_coh) ? disp_budget : t_coh;
    const std::complex<double> e_d = disp_far ? i_far : i_near;
    const std::complex<double> mu_d = disp_far ? mu_far : mu_near;
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_d * mu_d) * t_disp * c[disp_far ? N + 1 : 0]);

    const bool same = !(e_far != e_near || mu_far != mu_near || res_far != res_near);
    const std::complex<double> front = same ? 1.0 / phi_disp : (disp_far ? std::complex<double>(1.0, 0.0) : 1.0 / (phi_disp * phi_disp));

    D.T_te = D.fx_te * Tp[0] / phi_disp, D.T_tm = D.fx_tm * Tp[1] / phi_disp;
    D.R_te = front * Rp[0], D.R_tm = front * Rp[1];
    return D;
}

// One row of a palette that carries the lumped columns. `ei` is the magnitude of the imaginary
// part of eps at 1 GHz, so eps = a - j ei, the same convention diag_palette uses.
struct LumpMtl
{
    double a = 1.0, ei = 0.0, thk = 0.0;
    double att = 0.0, tf = 0.0;
    double resF = 0.0, resQ = 0.0, resS = 0.0;
    double alpha = 0.0, m = 0.0;
    double fRef = 1.0; // GHz; sets the mass-law argument (f / fRef) * thk and nothing else here
};

static inline std::unordered_map<std::string, std::vector<double>> lump_palette(const std::vector<LumpMtl> &v)
{
    std::unordered_map<std::string, std::vector<double>> m;
    for (const char *k : {"a", "b", "c", "d", "thk", "att", "attB", "alpha", "alphaB", "fRef",
                          "tf", "tfB", "resF", "resQ", "resS", "m"})
        m[k] = std::vector<double>(v.size(), 0.0);
    for (size_t i = 0; i < v.size(); ++i)
    {
        m["a"][i] = v[i].a, m["c"][i] = v[i].ei / 17.98, m["thk"][i] = v[i].thk;
        m["att"][i] = v[i].att, m["tf"][i] = v[i].tf;
        m["resF"][i] = v[i].resF, m["resQ"][i] = v[i].resQ, m["resS"][i] = v[i].resS;
        m["alpha"][i] = v[i].alpha, m["m"][i] = v[i].m, m["fRef"][i] = v[i].fRef;
    }
    return m;
}

// Permittivity of a LumpMtl row without its resonance, and the resonance on its own
static inline std::complex<double> lump_eps(const LumpMtl &M, double f_hz)
{
    return std::complex<double>(M.a, -17.98 * (M.ei / 17.98) / (f_hz * 1e-9));
}
static inline std::complex<double> lump_res(const LumpMtl &M, double f_hz)
{
    return diag_res(M.resF, M.resQ, M.resS, f_hz);
}

// DiagLayer from a LumpMtl row
static inline DiagLayer lump_layer(const LumpMtl &M, double f_hz)
{
    DiagLayer L;
    L.e = lump_eps(M, f_hz), L.d = M.thk, L.res = lump_res(M, f_hz), L.alpha = M.alpha;
    L.tf = M.tf, L.m = M.m, L.fRef = M.fRef;
    return L;
}

// Transmit port of a coherent sheet whose outer pair is past its critical angle: the near crossing
// of the composite (the transformer of a face with a wave behind it, in raw amplitude), the
// sheet's one-way factor, the internal series the reflect port carries, and what the pad of the
// closed far face passes of the incident flux. The displaced medium is air on every row that
// calls this, taken out at the incidence cosine because the far side carries no wave.
// S_te / S_tm are the series factors in power, so a row can say what the series alone is worth.
struct DiagLeak
{
    std::complex<double> T_te, T_tm;
    double S_te = 1.0, S_tm = 1.0;
};
static inline DiagLeak diag_leak_series(std::complex<double> e_near, std::complex<double> e_slab,
                                        std::complex<double> e_far, double tf_near, double tf_slab,
                                        double tf_far, double thk, double ct, double f_hz, bool scalar)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    auto root = [](std::complex<double> a)
    {
        if (std::imag(a) == 0.0 && std::real(a) < 0.0)
            a = std::complex<double>(std::real(a), -0.0); // decaying evanescent root
        return std::sqrt(a);
    };
    double s2 = 1.0 - ct * ct;
    std::complex<double> zn = std::sqrt(e_near), zs = std::sqrt(e_slab), zf = std::sqrt(e_far);
    std::complex<double> cs = root(1.0 - (e_near / e_slab) * s2), cf = root(1.0 - (e_near / e_far) * s2);

    // near crossing, flux-normalized; the face has a wave behind it and tf is a transformer, per channel
    std::complex<double> r_te = (zn * ct - zs * cs) / (zn * ct + zs * cs);
    std::complex<double> r_tm = scalar ? r_te : (zs * ct - zn * cs) / (zs * ct + zn * cs);
    std::complex<double> t_te = 2.0 * zn * ct / (zn * ct + zs * cs);
    std::complex<double> t_tm = scalar ? t_te : 2.0 * zn * ct / (zs * ct + zn * cs);
    std::complex<double> q_te = r_te, q_tm = r_tm;
    const bool near_tf = tf_near != 0.0 || tf_slab != 0.0;
    std::complex<double> c_te = near_tf ? diag_tf_face(q_te, tf_near, tf_slab) : 1.0, c_tm = near_tf ? diag_tf_face(q_tm, tf_near, tf_slab) : 1.0;
    std::complex<double> a_te = t_te * c_te, a_tm = t_tm * c_tm;

    // far face from inside the sheet: closed, so the pad, in flux over the incident flux; real in EM,
    // at arg(1 + R) of the raw face in scalar mode
    std::complex<double> f_te = (zs * cs - zf * cf) / (zs * cs + zf * cf);
    std::complex<double> f_tm = scalar ? f_te : (zf * cs - zs * cf) / (zf * cs + zs * cf);
    const double ph = scalar ? std::arg(1.0 + f_te) : 0.0;
    std::complex<double> b_te = std::polar(diag_closed_face(f_te, zs * cs, tf_slab, tf_far) / std::sqrt(std::real(zn * ct)), ph);
    std::complex<double> b_tm = scalar ? b_te : std::complex<double>(diag_closed_face(f_tm, std::conj(zs) * cs, tf_slab, tf_far) / std::sqrt(std::real(std::conj(zn) * ct)), 0.0);
    if (scalar)
        f_tm = f_te;

    std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_slab) * thk * cs);
    std::complex<double> phi_t = phi / std::exp(std::complex<double>(0.0, -k0 * thk * ct));

    auto den = [&](std::complex<double> r, std::complex<double> f)
    {
        if (tf_slab != 0.0 || tf_near != 0.0)
            diag_tf_face(r, tf_slab, tf_near); // the near face of the composite: a transformer
        std::complex<double> rn = -r;
        return 1.0 - rn * f * phi * phi; // f: the pad's reflection, at its own magnitude
    };
    DiagLeak D;
    std::complex<double> d_te = den(r_te, f_te), d_tm = den(r_tm, f_tm);
    D.T_te = a_te * b_te * phi_t / d_te, D.T_tm = a_tm * b_tm * phi_t / d_tm;
    D.S_te = 1.0 / std::norm(d_te), D.S_tm = 1.0 / std::norm(d_tm);
    return D;
}

// ================================================================================================
// Branches that had a guard and no row: the displacement budget, float, a corner with a sheet,
// the core's loss cap, ray_mesh_interact's own composite, a mixed stack, the followed mark and the
// resolved flag across abutting, nested and overlapping bodies, a verdict handed in, and a sheet
// without the mass column in a mass-law host. Every chain runs through ray_triangle_intersect,
// ray_mesh_interact and ray_state_update, one ray per call.
// ================================================================================================
namespace ub
{
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, xpr0, fbs, sbs, dir0, tridirN; // xpr0, dir0: what ray_mesh_interact wrote
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0, vout = 0;
        unsigned nh = 0;
    };
    struct Opt
    {
        double eps = 1.5;
        bool relaunch = false;                       // hand origN and destN to ray_state_update
        const std::vector<uint8_t> *vin = nullptr;   // verdict_inN
        const arma::mat *trivec = nullptr, *tridir = nullptr;
    };
    struct OneThread
    {
#ifdef _OPENMP
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
#endif
    };

    // One event at dtype T; everything is handed in and returned in double
    template <typename T>
    static Ev traced(const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                     const arma::mat &orig, const arma::mat &dest, const Ev *in, const Opt &opt = Opt())
    {
        typedef arma::Mat<T> M;
        typedef arma::Col<T> V;
        std::unordered_map<std::string, std::vector<T>> p;
        for (const auto &kv : pal)
            p[kv.first] = std::vector<T>(kv.second.begin(), kv.second.end());
        M o = arma::conv_to<M>::from(orig), d = arma::conv_to<M>::from(dest), ms = arma::conv_to<M>::from(mesh);
        M fbs, sbs, oN, dN, fN, sN, xp, nv, pd, tv, td, tvN, tdN;
        V gain, fang;
        arma::u32_vec nh, fi, si;
        std::vector<uint8_t> ot, rt, vo;
        quadriga_lib::ray_triangle_intersect<T>(&o, &d, &ms, &fbs, &sbs, &nh, &fi, &si);
        const bool tube = opt.trivec && opt.tridir;
        if (tube)
            tv = arma::conv_to<M>::from(*opt.trivec), td = arma::conv_to<M>::from(*opt.tridir);
        quadriga_lib::ray_mesh_interact<T>(itype, (T)f, &o, &d, &ms, &mtl_ind, &p, &fi, &si, tube ? &tv : nullptr, tube ? &td : nullptr,
                                           &oN, &dN, &fN, &sN, &gain, &xp, tube ? &tvN : nullptr, tube ? &tdN : nullptr, &fang, nullptr,
                                           nullptr, &nv, &ot, &pd, false, nullptr);
        Ev E;
        E.xpr0 = arma::conv_to<arma::mat>::from(xp), E.dir0 = arma::conv_to<arma::mat>::from(pd), E.ot = ot[0], E.nh = nh(0);
        arma::Col<short> mf(1), m2(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, m2(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        M pin;
        V ag, ar, ago, aro;
        if (in)
            pin = arma::conv_to<M>::from(in->path_dirN), ag = arma::conv_to<V>::from(in->accg), ar = arma::conv_to<V>::from(in->accr);
        quadriga_lib::ray_state_update<T>(itype, (T)f, &o, &d, &fN, &sN, &nh, &fang, &nv, &ot, &p, &mf, &m2,
                                          in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                          in ? &pin : nullptr, in ? &ag : nullptr, in ? &ar : nullptr,
                                          &E.prev, &E.cur, &E.buf, &gain, &xp, &pd, &ago, &aro, &rt, nullptr, opt.eps,
                                          opt.relaunch ? &oN : nullptr, opt.relaunch ? &dN : nullptr,
                                          (opt.relaunch && tube) ? &td : nullptr, (opt.relaunch && tube) ? &tdN : nullptr, opt.vin, &vo);
        E.origN = arma::conv_to<arma::mat>::from(oN), E.destN = arma::conv_to<arma::mat>::from(dN);
        E.fbs = arma::conv_to<arma::mat>::from(fN), E.sbs = arma::conv_to<arma::mat>::from(sN);
        E.xpr = arma::conv_to<arma::mat>::from(xp), E.path_dirN = arma::conv_to<arma::mat>::from(pd);
        E.accg = arma::conv_to<arma::vec>::from(ago), E.accr = arma::conv_to<arma::vec>::from(aro);
        if (tube)
            E.tridirN = arma::conv_to<arma::mat>::from(tdN);
        E.rt = rt[0], E.vout = vo.empty() ? (uint8_t)0 : vo[0];
        return E;
    }

    // A face in the plane x = const, its normal towards -x, or towards +x when flipped
    static arma::mat face(double x, bool flip)
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    }
    // A face through (px, py, 0) whose normal is turned by alpha about z from the x axis
    static arma::mat tilted(double px, double py, double alpha_deg, bool flip)
    {
        double a = alpha_deg * QPI / 180.0, tx = -std::sin(a), ty = std::cos(a), h = 200.0;
        arma::rowvec A = {px - h * tx, py - h * ty, -h}, B = {px - h * tx, py - h * ty, h}, C = {px + h * tx, py + h * ty, -h}, D = {px + h * tx, py + h * ty, h};
        arma::mat m(2, 9);
        if (!flip)
            m.row(0) = arma::join_rows(arma::join_rows(A, B), C), m.row(1) = arma::join_rows(arma::join_rows(C, B), D);
        else
            m.row(0) = arma::join_rows(arma::join_rows(A, C), B), m.row(1) = arma::join_rows(arma::join_rows(C, D), B);
        return m;
    }
    static arma::mat build(const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    }
    static double sumsq(const arma::mat &a);
    // The next segment: along the one the event wrote, far enough to cross every face that is left
    static double onward(const Ev &E, arma::mat &orig, arma::mat &dest)
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
        return d(0, 0);
    }
    // A path: one event per entry of seq, each on the segment the one before wrote
    template <typename T>
    static std::vector<Ev> chain_t(const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, const std::vector<int> &seq, double f, double thd,
                                   const std::vector<double> &eps, const Ev *start = nullptr)
    {
        std::vector<Ev> out;
        arma::mat o, d;
        diag_ray_at(thd, o, d);
        for (size_t q = 0; q < seq.size(); ++q)
        {
            Opt op;
            op.eps = eps[(q < eps.size()) ? q : eps.size() - 1];
            op.relaunch = seq[q] != 3 && seq[q] != 0;
            out.push_back(traced<T>(pal, mesh, mi, seq[q], f, o, d, q ? &out[q - 1] : start, op));
            if (out.back().nh == 0 || out.back().rt == 0)
                break;
            onward(out.back(), o, d);
        }
        return out;
    }
    static std::vector<Ev> chain(const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, const std::vector<int> &seq, double f, double thd,
                                 const std::vector<double> &eps, const Ev *start = nullptr)
    {
        return chain_t<double>(pal, mesh, mi, seq, f, thd, eps, start);
    }
    // The field of a path at the point its polyline ends at, referenced to the point it began at
    static std::complex<double> at_point(const std::vector<Ev> &c, double f, double thd)
    {
        std::complex<double> v = 1.0;
        double Lin = 0.0;
        for (size_t q = 0; q < c.size(); ++q)
        {
            v *= cpx(c[q].xpr, 0, 0);
            if (q)
                Lin += std::sqrt(ub::sumsq(c[q].fbs - c[q - 1].fbs));
        }
        const double k = 2.0 * QPI * f / C0, dy = c.back().fbs(0, 1) - c.front().fbs(0, 1);
        return v * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * std::sin(thd * QPI / 180.0) * dy));
    }
    static double sumsq(const arma::mat &a)
    {
        double v = 0.0;
        for (arma::uword i = 0; i < a.n_elem; ++i)
            v += a(i) * a(i);
        return v;
    }
    static bool same_bits(const arma::mat &a, const arma::mat &b)
    {
        return a.n_elem == b.n_elem && std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
    }
    static double dB(std::complex<double> z) { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; }
    static double deg(std::complex<double> z) { return std::arg(z) * 180.0 / QPI; }
    static unsigned w(short s) { return (unsigned)(unsigned short)s; }
}

// Every path of one ray through a wall: both passes at every event, through ray_triangle_intersect,
// ray_mesh_interact and ray_state_update, each event on the segment the one before wrote, collected
// where the ray is back in air and summed at its polyline's point, in front of the wall into R and
// behind it into T. eps 0 sums every series, eps 1.5 follows every bounce. inc: the sum of the path
// powers, which no interference enters.
namespace bf
{
    struct Acc
    {
        std::complex<double> T[2], R[2]; // scalar or VV, HH
        int nT = 0, nR = 0;
        double drop = 0.0, inc[2] = {0.0, 0.0};
    };
    static void walk(const ub::Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double thd, double eps, const ub::Ev *in,
                     const arma::mat &o, const arma::mat &d, std::complex<double> a0, std::complex<double> a1, double Lin, double y0, int depth,
                     int maxdepth, Acc &A)
    {
        const double k = 2.0 * QPI * f / C0;
        for (int it : {tr, (tr > 2) ? 3 : 0})
        {
            ub::Opt op;
            op.eps = eps, op.relaunch = (it == tr);
            ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, in, op);
            if (E.nh == 0 || E.rt == 0)
                continue;
            std::complex<double> c0 = cpx(E.xpr, 0, 0), c1 = (tr > 2) ? c0 : cpx(E.xpr, 0, 3);
            std::complex<double> b0 = a0 * c0, b1 = a1 * c1;
            double L2 = Lin + (in ? std::sqrt(ub::sumsq(E.fbs - in->fbs)) : 0.0), yy0 = in ? y0 : E.fbs(0, 1);
            if ((E.cur(0) & 0x3FFF) == 0)
            {
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -k * L2)) *
                                          std::exp(std::complex<double>(0.0, k * std::sin(thd * QPI / 180.0) * (E.fbs(0, 1) - yy0)));
                const bool fwd = (E.destN(0, 0) - E.origN(0, 0)) > 0.0;
                (fwd ? A.T[0] : A.R[0]) += b0 * ph, (fwd ? A.T[1] : A.R[1]) += b1 * ph, ++(fwd ? A.nT : A.nR);
                A.inc[0] += std::norm(b0), A.inc[1] += std::norm(b1);
                continue;
            }
            if (depth + 1 >= maxdepth || std::max(std::norm(b0), std::norm(b1)) < 1.0e-24)
            {
                A.drop += std::max(std::norm(b0), std::norm(b1));
                continue;
            }
            arma::mat o2, d2;
            ub::onward(E, o2, d2);
            walk(pal, mesh, mi, tr, f, thd, eps, &E, o2, d2, b0, b1, L2, yy0, depth + 1, maxdepth, A);
        }
    }
}

namespace cf
{
    struct Acc
    {
        std::complex<double> T[2], R[2]; // scalar or VV, HH
        int nT = 0, nR = 0;
    };
    // bf::walk with a half-space: a ray is collected where it is in air or has entered the sink
    // medium, into T where it runs towards +x. The first event of a ray launched inside a medium
    // forms no series there. kt: the transverse wavenumber of the launch
    static void walk(const ub::Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double kt, double eps, int sink, const ub::Ev *in,
                     bool first, const arma::mat &o, const arma::mat &d, std::complex<double> a0, std::complex<double> a1, double Lin, double y0,
                     int depth, int maxdepth, Acc &A)
    {
        const double k = 2.0 * QPI * f / C0;
        for (int it : {tr, (tr > 2) ? 3 : 0})
        {
            ub::Opt op;
            op.eps = (first && in) ? 1.5 : eps, op.relaunch = (it == tr);
            ub::Ev E = ub::traced<double>(pal, mesh, mi, it, f, o, d, in, op);
            if (E.nh == 0 || E.rt == 0)
                continue;
            std::complex<double> c0 = cpx(E.xpr, 0, 0), c1 = (tr > 2) ? c0 : cpx(E.xpr, 0, 3);
            std::complex<double> b0 = a0 * c0, b1 = a1 * c1;
            double L2 = Lin + (first ? 0.0 : std::sqrt(ub::sumsq(E.fbs - in->fbs))), yy0 = first ? E.fbs(0, 1) : y0;
            const int cur = E.cur(0) & 0x3FFF;
            if (cur == 0 || cur == sink)
            {
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -k * L2)) * std::exp(std::complex<double>(0.0, kt * (E.fbs(0, 1) - yy0)));
                const bool fwd = (E.destN(0, 0) - E.origN(0, 0)) > 0.0;
                (fwd ? A.T[0] : A.R[0]) += b0 * ph, (fwd ? A.T[1] : A.R[1]) += b1 * ph, ++(fwd ? A.nT : A.nR);
                continue;
            }
            if (depth + 1 >= maxdepth || std::max(std::norm(b0), std::norm(b1)) < 1.0e-24)
                continue;
            arma::mat o2, d2;
            ub::onward(E, o2, d2);
            walk(pal, mesh, mi, tr, f, kt, eps, sink, &E, false, o2, d2, b0, b1, L2, yy0, depth + 1, maxdepth, A);
        }
    }
}

// ================================================================================================
// ray_mesh_interact: reference oracles and mesh helpers
// ================================================================================================

// Exact analytic slab, complex angles throughout, per polarization:
//   T = t12 t23 phi / (1 - r12 r23 phi^2),  phi = exp(-j k0 n d cos(theta_slab))
// Independent of every library internal. This is the oracle both the thin-sheet composite and
// the traced solid are measured against, and it is the only way to tell which of the two moved
// when they disagree.
//
// Permeability enters through three substitutions and nowhere else: the admittance is
// z = sqrt(eps/mu), the Snell ratio is a ratio of eps*mu products, and the propagation index is
// n = sqrt(eps*mu). Air is on both sides, so the near-side product is 1 and z1 does not move. The
// Fresnel groupings and the Airy assembly are already written in z and c. mu_s defaults to 1, where
// all three reduce to the dielectric expressions they replace.
struct QdSlabRef
{
    double p_te = 0.0, p_tm = 0.0, p_avg = 0.0, ratio = 0.0; // powers and min/max channel ratio
    std::complex<double> t_te, t_tm;                         // complex amplitudes
    std::complex<double> ct_slab;                            // complex in-slab cosine
};

static inline QdSlabRef qd_slab_ref(std::complex<double> eps_s, double thk,
                                    double theta_n_rad, double f_hz,
                                    std::complex<double> mu_s = std::complex<double>(1.0, 0.0))
{
    const double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s / mu_s), n = std::sqrt(eps_s * mu_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / (eps_s * mu_s)) * s1 * s1);
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

// Composite slab reflectance, TE/TM averaged, air | slab | air. The stack is symmetric, so the
// internal reflection is the same at both faces and R = r (1 - phi^2) / (1 - r^2 phi^2).
//
// Carries permeability through the same three substitutions as qd_slab_ref, and returns the two
// channels separately where asked: the average cannot tell TE from TM, and for a magnetic sheet
// which of the two is the larger is exactly what is under test.
static inline double qd_slab_ref_R(std::complex<double> eps_s, double thk,
                                   double theta_n_rad, double f_hz,
                                   std::complex<double> mu_s = std::complex<double>(1.0, 0.0),
                                   double *R_te_out = nullptr, double *R_tm_out = nullptr)
{
    const double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s / mu_s), n = std::sqrt(eps_s * mu_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / (eps_s * mu_s)) * s1 * s1);
    const std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * thk * c2);
    const std::complex<double> e = phi * phi;
    const std::complex<double> r_te = (z1 * c1 - z2 * c2) / (z1 * c1 + z2 * c2);
    const std::complex<double> r_tm = (z2 * c1 - z1 * c2) / (z2 * c1 + z1 * c2);
    const std::complex<double> R_te = r_te * (1.0 - e) / (1.0 - r_te * r_te * e);
    const std::complex<double> R_tm = r_tm * (1.0 - e) / (1.0 - r_tm * r_tm * e);
    if (R_te_out)
        *R_te_out = std::norm(R_te);
    if (R_tm_out)
        *R_tm_out = std::norm(R_tm);
    return 0.5 * (std::norm(R_te) + std::norm(R_tm));
}

// The analytic slab with the lumped material columns a sheet can carry, air | sheet | air, both
// ports, per polarization. Written separately from qd_slab_ref so that the oracle everything else
// is measured against stays the plain Airy pair; at its defaults this reproduces it, which the
// fixture asserts rather than assumes.
//
//   att_dB   Penetration loss of the sheet. One power factor 10^(-att/10) on the transmit port,
//            nothing on the reflect port.
//   tf       Transmission factor of the sheet, on both of its faces: an ideal transformer of real
//            ratio at each face, which for a symmetric stack is the same sheet with its tilted
//            admittance scaled by nu and its phase thickness untouched. With x0 the modulus of the
//            admittance ratio sheet / air and rm = (1 - x0) / (1 + x0), the pair rule
//            R' = rm^2 (1 - tf) for tf >= 0 and rm^2 + (1 - rm^2)(-tf) for tf < 0 gives the modulus
//            the ratio is moved to, x = (1 - s) / (1 + s) with s = sqrt(R') at the sign of rm, and
//            nu = x / x0. For a lossless sheet below its critical angle that is the pair rule on
//            the reflectance of each face.
//   res      Permittivity resonance, as an increment. It is part of the permittivity: the
//            admittance, the cosine and the propagation index all read it.
struct QdLumpRef
{
    double T_te = 0.0, T_tm = 0.0, R_te = 0.0, R_tm = 0.0; // channel powers
    std::complex<double> t_te, t_tm;                       // complex transmit amplitudes
};

static inline QdLumpRef qd_slab_ref_lumped(std::complex<double> eps_s, double thk,
                                           double theta_n_rad, double f_hz,
                                           double att_dB = 0.0, double tf = 0.0,
                                           std::complex<double> res = std::complex<double>(0.0, 0.0))
{
    const double k0 = 2.0 * arma::datum::pi * f_hz / 299792458.0;
    const std::complex<double> e_if = eps_s + res;
    const std::complex<double> z2 = std::sqrt(e_if), n = std::sqrt(e_if);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / e_if) * s1 * s1);
    const std::complex<double> p2 = std::exp(std::complex<double>(0.0, -2.0) * k0 * n * thk * c2);
    const std::complex<double> p1 = std::exp(std::complex<double>(0.0, -1.0) * k0 * n * thk * c2);
    const double g = std::pow(10.0, -0.1 * att_dB);

    QdLumpRef Q;
    for (int pol = 0; pol < 2; ++pol)
    {
        std::complex<double> r, t12, t21;
        if (pol == 0)
            r = (c1 - z2 * c2) / (c1 + z2 * c2), t12 = (2.0 * c1) / (c1 + z2 * c2),
            t21 = (2.0 * z2 * c2) / (z2 * c2 + c1);
        else
            r = (z2 * c1 - c2) / (z2 * c1 + c2), t12 = (2.0 * c1) / (z2 * c1 + c2),
            t21 = (2.0 * z2 * c2) / (c2 + z2 * c1);

        std::complex<double> rr = r, kap2(1.0, 0.0); // the reflection of either face, and the factor on the t12 t21 product
        if (tf != 0.0)
        {
            const std::complex<double> beta = (1.0 - r) / (1.0 + r);
            const double x0 = std::abs(beta), rm = (1.0 - x0) / (1.0 + x0);
            const double Rt = (tf >= 0.0) ? rm * rm * (1.0 - tf) : rm * rm + (1.0 - rm * rm) * (-tf);
            const double s = (rm < 0.0) ? -std::sqrt(Rt) : std::sqrt(Rt);
            if (1.0 + s > 0.0)
            {
                const double nu = ((1.0 - s) / (1.0 + s)) / x0;
                rr = (1.0 - nu * beta) / (1.0 + nu * beta);
                kap2 = (1.0 - rr * rr) / (1.0 - r * r);
            }
            else
                rr = -1.0, kap2 = 0.0;
        }

        std::complex<double> den = 1.0 - rr * rr * p2;
        std::complex<double> T = std::sqrt(g) * kap2 * t12 * t21 * p1 / den;
        std::complex<double> R = rr - std::sqrt(g) * kap2 * t12 * t21 * rr * p2 / den; // the echo entered the sheet and paid
        if (pol == 0)
            Q.t_te = T, Q.T_te = std::norm(T), Q.R_te = std::norm(R);
        else
            Q.t_tm = T, Q.T_tm = std::norm(T), Q.R_tm = std::norm(R);
    }
    return Q;
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

// ================================================================================================
// ray_progress: scenes and launch configurations
// ================================================================================================

// Verification suite for the public ray_progress API.
//
// ray_progress is an orchestrator, not a physics kernel: the Fresnel/Jones math is pinned by the
// mesh-interact cases, the state machine by the state-update cases, and the beam split by the
// subdivide_rays tests. The ray_progress cases therefore assert the things only ray_progress can get wrong,
// namely the bookkeeping between three index spaces:
//
//   full set     [n_ray]      -> the launch configuration handed in
//   compact set  [n_interact] -> rays that hit the mesh (gated on no_interact != 0)
//   per-stage    [n_subdiv], [n_reflect], [n_transmit] -> subsets addressed through index arrays
//
// What is checked: the per-stage return counts, the layout and column counts of every returned
// block, the ordering of the new launch configuration (sub-beams, then reflections, then
// transmissions), the survival gates (gain, interaction / reflection / transmission / subdivision
// limits), the medium-state hand-off across a slab, and — most importantly — that the two sides of
// the single compaction is a pure permutation of per-ray results.
//
// Oracles: geometry is exact by construction (axis-aligned 2x2x2 cubes, rays aimed at known
// points), so first-bounce coordinates are asserted directly. Where the value depends on the
// physics kernels, the assertion is an invariant instead (finiteness, monotonicity, cross-branch
// agreement, energy <= 1) so this file does not duplicate — or drift from — the kernel suites.
//
// Ray identity: the tests borrow quadriga_lib::path::iC as a stable per-ray handle. ray_progress
// never writes it (only the downstream commit stage assigns real channel IDs), and both
// path::extend and path copy-assignment preserve it, so seeding it in the builder gives a stable
// handle from an output row back to the input ray it came from. That is what makes the
// cross-branch comparison possible.
//
// Delegated intersection: the optional no_interact_in / fbs_ind_in / sbs_ind_in inputs let a caller
// hand in a ray_triangle_intersect result instead of having ray_progress compute it. Those cases
// assert bit-identical output against the internal path rather than mere equivalence, since the two
// routes feed the same arrays into the same per-ray kernels in the same order; anything less would
// let a remapping bug hide behind a tolerance.
//
// Conventions:
//  - All cubes are the 2x2x2 Blender-style cube, translated by the constructor argument. A cube
//    centred at (5,0,0) therefore has faces at x = 4 (near) and x = 6 (far).
//  - The point source sits at the coordinate origin; rays are launched from a small launch sphere
//    of radius r0 = 0.1 m, matching what ray_init produces.
//  - Material columns are the obj_file_read names {a,b,c,d,att,attB,alpha,alphaB,fRef}.

namespace
{
    constexpr float PROG_FRQ = 10.0e9f;      // Reference frequency
    constexpr float R0 = 0.1f;          // Launch sphere radius
    constexpr float WIDE_DEG = 1.0f;    // Beam half-angle that subdivides at SUB_TOL
    constexpr float NARROW_DEG = 0.01f; // Beam half-angle that never subdivides
    constexpr float SUB_TOL = 0.05f;    // Subdivision tolerance used by the beam tests

    // Euclidean length of a 3-element row. arma::norm dispatches to BLAS snrm2, which is not
    // linked for single precision here, so do it directly.
    float nrm3(const arma::frowvec &v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); }

    // ---------------------------------------------------------------------------------------
    // Materials
    // ---------------------------------------------------------------------------------------

    // Lossless dielectric, eps_r = 5, frequency independent
    arma::frowvec mtl_dielectric() { return {5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    // ITU-style concrete: eps_r = 5.24, sigma = 0.0462 * f^0.7822 -> frequency dependent
    arma::frowvec mtl_concrete() { return {5.24f, 0.0f, 0.0462f, 0.7822f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; }

    // ---------------------------------------------------------------------------------------
    // Scene
    // ---------------------------------------------------------------------------------------

    struct Scene
    {
        arma::fmat mesh;
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<float>> mtl_prop;
    };

    // One 2x2x2 cube centred at (cx,cy,cz), all 12 faces of the same material
    Scene one_cube(float cx, float cy, float cz, const arma::frowvec &mtl = mtl_dielectric())
    {
        Scene S;
        S.mesh = quadriga_lib::cube<float>({}, {}, {cx, cy, cz});
        arma::fmat M = arma::repmat(mtl, (arma::uword)S.mesh.n_rows, 1);
        mtl_matrix_to_map(M, S.mtl_ind, S.mtl_prop);
        return S;
    }

    // Two 2x2x2 cubes, one at (5,0,0) and one at (9,0,0), same material. Used for the sub-mesh
    // partitioning test: faces 0..11 are the first cube, 12..23 the second.
    Scene two_cubes()
    {
        Scene S;
        arma::fmat a = quadriga_lib::cube<float>({}, {}, {5.0f, 0.0f, 0.0f});
        arma::fmat b = quadriga_lib::cube<float>({}, {}, {9.0f, 0.0f, 0.0f});
        S.mesh = arma::join_cols(a, b);
        arma::fmat M = arma::repmat(mtl_dielectric(), (arma::uword)S.mesh.n_rows, 1);
        mtl_matrix_to_map(M, S.mtl_ind, S.mtl_prop);
        return S;
    }

    // ---------------------------------------------------------------------------------------
    // Launch configuration
    // ---------------------------------------------------------------------------------------

    // The optional block of ray_progress, so a scenario reads as one call
    struct Opt
    {
        uint8_t max_int = 20;
        uint8_t max_ref = 10;
        uint8_t max_tra = 10;
        uint8_t max_sub = 2;
        float min_gain_dB = -140.0f;
        float sub_tol = 3.0f;
        float slab = 0.15f;
        bool refraction = true;
        const arma::u32_vec *smi = nullptr;
        const arma::fmat *aabb = nullptr;

        // Delegated ray-mesh intersection (all three or none)
        const arma::u32_vec *ni_in = nullptr;
        const arma::u32_vec *fi_in = nullptr;
        const arma::u32_vec *si_in = nullptr;

        // Delegated subdivision flag
        const std::vector<bool> *sf_in = nullptr;
    };

    struct Cfg
    {
        float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
        arma::fvec freq;
        bool scalar = false;
        bool beam = false;

        arma::fmat orig, dest, acc_dist, trivec, tridir;
        arma::fcube path_dir; // [n_ray, 3, n_freq], one direction slice per frequency
        arma::Col<short> prev, cur, buf;
        std::vector<quadriga_lib::path> paths;

        arma::uword n() const { return orig.n_rows; }
        arma::uword nf() const { return freq.n_elem; }

        std::array<unsigned, 4> step(const Scene &S, const Opt &o = Opt())
        {
            return quadriga_lib::ray_progress(S.mesh, S.mtl_ind, S.mtl_prop, freq,
                                              Ox, Oy, Oz,
                                              orig, dest, prev, cur, buf, path_dir, acc_dist, paths,
                                              beam ? &trivec : nullptr,
                                              beam ? &tridir : nullptr,
                                              o.smi, o.aabb,
                                              o.max_int, o.max_ref, o.max_tra, o.max_sub,
                                              o.min_gain_dB, o.sub_tol, o.slab,
                                              o.refraction, scalar,
                                              o.ni_in, o.fi_in, o.si_in, o.sf_in);
        }
    };

    // Build a launch configuration: n rays from the source along the rows of "dirs" (need not be
    // normalized), each reaching "len" metres past the launch sphere. If "half_deg" has one entry
    // per ray the configuration is a beam set, with a triangular tube of that half-angle whose
    // vertices sit on the launch sphere.
    Cfg make_cfg(const arma::fmat &dirs_in, float len,
                 arma::uword n_freq = 1, bool scalar = false,
                 const arma::fvec &half_deg = arma::fvec())
    {
        const arma::uword n = dirs_in.n_rows;

        Cfg C;
        C.scalar = scalar;
        C.beam = (n != 0 && half_deg.n_elem == n);

        C.freq.set_size(n_freq);
        for (arma::uword f = 0; f < n_freq; ++f)
            C.freq(f) = PROG_FRQ * (1.0f + 0.5f * (float)f); // 10, 15, 20 GHz, ...

        arma::fmat d = dirs_in;
        for (arma::uword i = 0; i < n; ++i)
            d.row(i) /= std::sqrt(arma::accu(d.row(i) % d.row(i)));

        C.orig = d * R0;
        C.dest = d * (R0 + len);

        // A launched ray is outside a medium, so every frequency starts from the same direction
        C.path_dir.set_size(n, 3, n_freq);
        for (arma::uword f = 0; f < n_freq; ++f)
            C.path_dir.slice(f) = d;

        C.prev.zeros(n);
        C.cur.zeros(n);
        C.buf.zeros(n);
        C.acc_dist.zeros(n, 1 + n_freq);

        C.paths.resize(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            C.paths[i].init(0, n_freq, scalar);
            C.paths[i].set_length(R0);   // source -> ray origin
            C.paths[i].iC = (unsigned)i; // stable identity, preserved by copy and extend
        }

        if (C.beam)
        {
            C.trivec.zeros(n, 9);
            C.tridir.zeros(n, 9);
            for (arma::uword i = 0; i < n; ++i)
            {
                arma::frowvec k = d.row(i);
                arma::frowvec a = (std::abs(k(2)) < 0.9f) ? arma::frowvec{0.0f, 0.0f, 1.0f}
                                                          : arma::frowvec{1.0f, 0.0f, 0.0f};
                arma::frowvec u = arma::cross(a, k);
                u /= nrm3(u);
                arma::frowvec v = arma::cross(k, u);

                const float ha = half_deg(i) * (float)(arma::datum::pi / 180.0);
                for (int j = 0; j < 3; ++j)
                {
                    const float phi = (float)(2.0 * arma::datum::pi * (double)j / 3.0);
                    arma::frowvec vj = std::cos(ha) * k +
                                       std::sin(ha) * (std::cos(phi) * u + std::sin(phi) * v);
                    vj /= nrm3(vj);

                    arma::frowvec tv = R0 * vj - R0 * k; // vertex on the launch sphere, rel. to orig
                    for (int c = 0; c < 3; ++c)
                    {
                        C.tridir(i, 3 * j + c) = vj(c);
                        C.trivec(i, 3 * j + c) = tv(c);
                    }
                }
            }
        }
        return C;
    }

    // A single ray along +x
    Cfg one_ray(float len = 10.0f, arma::uword n_freq = 1, bool scalar = false, float half_deg = -1.0f)
    {
        arma::fmat d = {{1.0f, 0.0f, 0.0f}};
        arma::fvec h;
        if (half_deg > 0.0f)
            h = arma::fvec{half_deg};
        return make_cfg(d, len, n_freq, scalar, h);
    }

    // Deterministic fan of rays that all strike the near face (x = 4) of a cube centred at
    // (5,0,0): ray i points at (4, y_i, z_i) with |y_i|, |z_i| <= 0.6. The first n_wide rays get
    // WIDE_DEG (subdivides at SUB_TOL), the rest NARROW_DEG (never subdivides). The narrow
    // directions depend only on i, so two fans of different length share their common prefix —
    // which is what lets the compaction test compare the same rays under both branches.
    void beam_fan(arma::uword n_total, arma::uword n_wide, arma::fmat &dirs, arma::fvec &half)
    {
        dirs.set_size(n_total, 3);
        half.set_size(n_total);
        for (arma::uword i = 0; i < n_total; ++i)
        {
            const double t = (double)(i + 1);
            const double u = std::fmod(0.7548776662466927 * t, 1.0); // frac(i / phi)
            const double v = std::fmod(0.5698402909980532 * t, 1.0); // frac(i / phi^2)
            dirs(i, 0) = 4.0f;
            dirs(i, 1) = (float)(1.2 * u - 0.6);
            dirs(i, 2) = (float)(1.2 * v - 0.6);
            half(i) = (i < n_wide) ? WIDE_DEG : NARROW_DEG;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Delegated ray-mesh intersection
    // ---------------------------------------------------------------------------------------

    // Intersection results, computed exactly the way a caller delegating the work would
    struct Hits
    {
        arma::u32_vec no_interact, fbs_ind, sbs_ind;
    };

    Hits intersect(const Cfg &C, const Scene &S, const arma::u32_vec *smi = nullptr)
    {
        Hits H;
        quadriga_lib::ray_triangle_intersect<float>(&C.orig, &C.dest, &S.mesh, nullptr, nullptr,
                                                    &H.no_interact, &H.fbs_ind, &H.sbs_ind,
                                                    smi, nullptr);
        return H;
    }

    // Bind a delegated intersection into an option block
    Opt with_hits(const Opt &base, const Hits &H)
    {
        Opt o = base;
        o.ni_in = &H.no_interact;
        o.fi_in = &H.fbs_ind;
        o.si_in = &H.sbs_ind;
        return o;
    }

    // Subdivision flags, computed the way a shading pass would: from the launch configuration as it
    // stands, before ray_progress touches it, using the same limits ray_progress will apply.
    std::vector<bool> subdiv_flags(const Cfg &C, const Scene &S, const Hits &H, const Opt &o)
    {
        return quadriga_lib::ray_subdivide_flag(S.mesh, C.orig, C.dest, H.fbs_ind, C.trivec, C.tridir,
                                                C.paths, C.cur, o.max_int, o.max_sub, o.sub_tol);
    }

    // Bind a delegated subdivision flag into an option block
    Opt with_flag(const Opt &base, const std::vector<bool> &f)
    {
        Opt o = base;
        o.sf_in = &f;
        return o;
    }

    // Number of set flags
    arma::uword count_true(const std::vector<bool> &f)
    {
        arma::uword n = 0;
        for (size_t i = 0; i < f.size(); ++i)
            n += f[i] ? 1u : 0u;
        return n;
    }

    // ---------------------------------------------------------------------------------------
    // Assertions
    // ---------------------------------------------------------------------------------------

    // Every returned block matches n_out, keeps its column count, and is finite
    void check_shapes(const Cfg &C, arma::uword n_out)
    {
        CHECK(C.orig.n_rows == n_out);
        CHECK(C.orig.n_cols == 3);
        CHECK(C.dest.n_rows == n_out);
        CHECK(C.dest.n_cols == 3);
        CHECK(C.path_dir.n_rows == n_out);
        CHECK(C.path_dir.n_cols == 3);
        CHECK(C.path_dir.n_slices == C.nf());
        CHECK(C.acc_dist.n_rows == n_out);
        CHECK(C.acc_dist.n_cols == 1 + C.nf());
        CHECK(C.prev.n_elem == n_out);
        CHECK(C.cur.n_elem == n_out);
        CHECK(C.buf.n_elem == n_out);
        CHECK(C.paths.size() == n_out);

        if (C.beam)
        {
            CHECK(C.trivec.n_rows == n_out);
            CHECK(C.trivec.n_cols == 9);
            CHECK(C.tridir.n_rows == n_out);
            CHECK(C.tridir.n_cols == 9);
        }

        if (n_out == 0)
            return;

        CHECK(C.orig.is_finite());
        CHECK(C.dest.is_finite());
        CHECK(C.path_dir.is_finite());
        CHECK(C.acc_dist.is_finite());
        CHECK(C.acc_dist.min() >= 0.0f); // ray_progress rejects a negative acc_dist on entry
        if (C.beam)
        {
            CHECK(C.trivec.is_finite());
            CHECK(C.tridir.is_finite());
        }
        for (const auto &p : C.paths)
        {
            CHECK(std::isfinite(p.length()));
            CHECK(p.calc_gain(0.0f, 0) >= 0.0f);
        }
    }

    // n_out follows from the per-stage counts
    arma::uword expected_out(const std::array<unsigned, 4> &s)
    {
        return 4ull * s[1] + s[2] + s[3];
    }

    // Map input ray id (path::iC) -> output row. Requires the ids to be unique, which holds when
    // only one continuation class is launched (e.g. reflections with max_no_transmissions = 0).
    std::map<unsigned, arma::uword> id_map(const Cfg &C)
    {
        std::map<unsigned, arma::uword> m;
        for (arma::uword i = 0; i < C.paths.size(); ++i)
            m.emplace(C.paths[i].iC, i);
        return m;
    }

    // Compare two runs row by row for every ray id present in both. Ids below min_id are
    // skipped, which is how a run with subdivision is compared against one without: the
    // subdivided ids describe different objects in the two runs.
    void check_same_rays(const Cfg &A, const Cfg &B, unsigned min_id = 0u, float tol = 1e-5f)
    {
        auto ma = id_map(A), mb = id_map(B);
        arma::uword matched = 0;
        for (const auto &kv : ma)
        {
            if (kv.first < min_id)
                continue;
            auto it = mb.find(kv.first);
            if (it == mb.end())
                continue;
            const arma::uword ia = kv.second, ib = it->second;
            ++matched;

            for (arma::uword c = 0; c < 3; ++c)
            {
                CHECK(std::abs(A.orig(ia, c) - B.orig(ib, c)) < tol);
                CHECK(std::abs(A.dest(ia, c) - B.dest(ib, c)) < tol);
                for (arma::uword f = 0; f < A.path_dir.n_slices; ++f)
                    CHECK(std::abs(A.path_dir(ia, c, f) - B.path_dir(ib, c, f)) < tol);
            }
            for (arma::uword c = 0; c < A.acc_dist.n_cols; ++c)
                CHECK(std::abs(A.acc_dist(ia, c) - B.acc_dist(ib, c)) < tol);
            CHECK((int)A.prev(ia) == (int)B.prev(ib));
            CHECK((int)A.cur(ia) == (int)B.cur(ib));
            CHECK((int)A.buf(ia) == (int)B.buf(ib));
            CHECK(A.paths[ia].n_seg() == B.paths[ib].n_seg());
            CHECK((int)A.paths[ia].nREF == (int)B.paths[ib].nREF);
            CHECK((int)A.paths[ia].nTRA == (int)B.paths[ib].nTRA);
            CHECK(std::abs(A.paths[ia].calc_gain(0.0f, 0) - B.paths[ib].calc_gain(0.0f, 0)) < tol);

            if (A.beam && B.beam)
                for (arma::uword c = 0; c < 9; ++c)
                {
                    CHECK(std::abs(A.trivec(ia, c) - B.trivec(ib, c)) < tol);
                    CHECK(std::abs(A.tridir(ia, c) - B.tridir(ib, c)) < tol);
                }
        }
        CHECK(matched > 0); // a vacuous comparison is a failed comparison
    }

    // Two runs that must agree row for row, not merely ray for ray
    void check_identical(const Cfg &A, const Cfg &B)
    {
        REQUIRE(A.n() == B.n());
        REQUIRE(A.paths.size() == B.paths.size());

        if (A.n() != 0)
        {
            CHECK(arma::approx_equal(A.orig, B.orig, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.dest, B.dest, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.path_dir, B.path_dir, "absdiff", 0.0f));
            CHECK(arma::approx_equal(A.acc_dist, B.acc_dist, "absdiff", 0.0f));
            CHECK(arma::all(A.prev == B.prev));
            CHECK(arma::all(A.cur == B.cur));
            CHECK(arma::all(A.buf == B.buf));
            if (A.beam && B.beam)
            {
                CHECK(arma::approx_equal(A.trivec, B.trivec, "absdiff", 0.0f));
                CHECK(arma::approx_equal(A.tridir, B.tridir, "absdiff", 0.0f));
            }
        }

        for (size_t i = 0; i < A.paths.size(); ++i)
        {
            CHECK(A.paths[i].iC == B.paths[i].iC);
            CHECK(A.paths[i].n_seg() == B.paths[i].n_seg());
            CHECK((int)A.paths[i].nREF == (int)B.paths[i].nREF);
            CHECK((int)A.paths[i].nTRA == (int)B.paths[i].nTRA);
            CHECK((int)A.paths[i].nSUB == (int)B.paths[i].nSUB);
            CHECK(A.paths[i].length() == B.paths[i].length());
            CHECK(A.paths[i].calc_gain(0.0f, 0) == B.paths[i].calc_gain(0.0f, 0));
        }
    }

    // Distance from a row of M to the coordinate origin
    float radius(const arma::fmat &M, arma::uword row)
    {
        return std::sqrt(M(row, 0) * M(row, 0) + M(row, 1) * M(row, 1) + M(row, 2) * M(row, 2));
    }
}

// ================================================================================================
// Acoustic frequency mapping and single-material helpers (former test_acoustic_interactions.cpp)
// ================================================================================================

#include <utility>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

// Frequency mapping constants: acoustic Hz -> mapped radio Hz.
static constexpr double C_LIGHT = 299792458.0;
static constexpr double C_SOUND = 342.77;
static constexpr double AC2RF = C_LIGHT / C_SOUND; // acoustic Hz -> radio Hz

static inline double ac2rf(double f_acoustic_Hz) { return f_acoustic_Hz * AC2RF; }

// Build a one-material (mtl_ind, mtl_prop) pair from a list of named columns, broadcast over all
// 12 cube faces. Columns not listed are simply absent from the map so the consumer applies its
// default, which mirrors how obj_file_read emits a schema-blind table.
static inline void single_material(const std::vector<std::pair<std::string, double>> &cols,
                                   arma::uvec &mtl_ind,
                                   std::unordered_map<std::string, std::vector<double>> &mtl_prop)
{
    mtl_ind.set_size(12);
    mtl_ind.ones(); // all faces share material 1
    mtl_prop.clear();
    for (const auto &kv : cols)
        mtl_prop[kv.first] = std::vector<double>(1, kv.second);
}

// Reference scalar (TE-only) reflected-power coefficient for a single interface, energy-conserving.
// eta1 = incoming medium, eta2 = outgoing medium, incidence angle measured from the surface normal
// (0 deg = normal incidence). Mirrors the scalar branch of ray_mesh_interact when mu_r = 1, including
// total internal reflection (|R|^2 -> 1) when eta1 > eta2 past the critical angle.
static inline double ref_scalar_reflection_gain(std::complex<double> eta1, std::complex<double> eta2,
                                                double incidence_angle_deg)
{
    double deg2rad = arma::datum::pi / 180.0;
    double cos_th = std::cos(incidence_angle_deg * deg2rad);
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * sin_th * sin_th);
    std::complex<double> s1 = std::sqrt(eta1), s2 = std::sqrt(eta2);
    std::complex<double> R_te = (s1 * cos_th - s2 * cos_th2) / (s1 * cos_th + s2 * cos_th2);
    return std::norm(R_te);
}

// Single-interface gain from ray_mesh_interact. Computes the ray-triangle indices, runs the
// interaction, and returns the per-ray gain. interaction_type: 3 = scalar reflection,
// 4 = scalar transmission. Under the new API ray_mesh_interact recomputes the intersection points
// internally, so only the indices are forwarded.
static double rmi_gain(int interaction_type, double freq,
                       const arma::mat &orig, const arma::mat &dest,
                       const arma::mat &mesh, const arma::uvec &mtl_ind,
                       const std::unordered_map<std::string, std::vector<double>> &mtl)
{
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::vec gainN;
    quadriga_lib::ray_mesh_interact<double>(interaction_type, freq, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind,
                                            nullptr, nullptr, // trivec, tridir,
                                            nullptr, nullptr, // origN, destN
                                            nullptr, nullptr, // fbsN, sbsN
                                            &gainN);          // gainN
    return gainN.n_elem ? gainN(0) : 0.0;
}

// ================================================================================================
// Twins with every digit, asserted against stored values
// ================================================================================================
// The wave tables that print a coefficient with every digit also assert it against the value stored
// in tests/data/wave_twins.tsv, one row per table row: table, key, values, tab-separated. A value
// passes within a relative 1e-12, so a change that moves a twin fails an assertion instead of only
// showing in a diff of the output. With the environment variable QD_TWIN_WRITE set to a file name the
// values are appended to that file instead and nothing is asserted; that is how the stored file is
// regenerated after a change that is meant to move them. The suite runs from the repository root.
#include <cstdarg>
#include <fstream>
#include <sstream>

static inline std::string twin_key(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

static inline const std::map<std::string, std::vector<double>> &twin_store()
{
    static std::map<std::string, std::vector<double>> S;
    static bool loaded = false;
    if (!loaded)
    {
        loaded = true;
        std::ifstream in("tests/data/wave_twins.tsv");
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream ls(line);
            std::string table, key, v;
            if (!std::getline(ls, table, '\t') || !std::getline(ls, key, '\t'))
                continue;
            std::vector<double> &x = S[table + '\t' + key];
            while (std::getline(ls, v, '\t'))
                x.push_back(std::strtod(v.c_str(), nullptr));
        }
    }
    return S;
}

static inline void twin(const char *table, const std::string &key, const std::vector<double> &v)
{
    const std::string id = std::string(table) + '\t' + key;
    if (const char *w = std::getenv("QD_TWIN_WRITE"))
    {
        if (FILE *fp = std::fopen(w, "a"))
        {
            std::fprintf(fp, "%s", id.c_str());
            for (double x : v)
                std::fprintf(fp, "\t%.17g", x);
            std::fprintf(fp, "\n");
            std::fclose(fp);
        }
        return;
    }
    const auto &S = twin_store();
    const auto it = S.find(id);
    INFO("twin " << table << ", " << key);
    bool ok = it != S.end() && it->second.size() == v.size();
    for (size_t i = 0; ok && i < v.size(); ++i)
        ok = std::abs(v[i] - it->second[i]) <= 1.0e-12 * std::max(std::abs(v[i]), std::abs(it->second[i]));
    CHECK(ok);
}
