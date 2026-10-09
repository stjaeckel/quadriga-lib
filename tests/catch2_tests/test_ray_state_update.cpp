// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

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

#include <catch2/catch_test_macros.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "quadriga_tools.hpp"
#include "bits.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
static inline void mtl_matrix_to_map(const arma::Mat<dtype> &M,
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
static const double OFF = 0.001;           // ray_offset used inside the function
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
static constexpr uint8_t OT_COR_IIO = 29; // corner, ii->o
static constexpr uint8_t OT_COR_OII = 31; // corner, o->ii
static constexpr uint8_t OT_TIR = 32;     // add to any code for the total-reflection variant

// resolved_typeN outcome codes (bit-encoded). Base values used by the assertions:
//   bit0 ok, bit1 vbs, bit2 resolve, bit3 inside, bit4 fix, bit5 tir, bit6 trans, bit7 refl.
static constexpr uint8_t RT_KILL = 0;          // ray killed
static constexpr uint8_t RT_ENTRY = 9;         // inside, ok            (o-i entry / i-i FBS==VBS)
static constexpr uint8_t RT_ENTRY_DEFER = 8;   // inside                (o-i entry, deferred buffer)
static constexpr uint8_t RT_II_SLAB = 13;      // inside, resolve, ok   (i-i + slab series)
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

TEST_CASE("ray_state_update - state and code encoding")
{
    // State words: bit 15 flag, bit 14 a second flag, bits 0..13 material. Never arithmetic negation.
    CHECK(enc(0, true) == (short)-32768); // resolved air word is 0x8000, not -0
    CHECK(mat_of(enc(0, true)) == 0);
    CHECK(flag_of(enc(0, true)));
    CHECK(enc(5, true) != (short)-5); // 0x8005, not the arithmetic negative
    CHECK(mat_of(enc(5, true)) == 5);
    CHECK(flag_of(enc(5, true)));
    CHECK(mat_of(enc(5, false)) == 5);
    CHECK(!flag_of(enc(5, false)));
    CHECK(mat_of((short)0x7FFF) == 0x3FFF);

    // out_type / resolved_type bit reads
    CHECK(rt_bit(OT_ENTRY, 0));                    // ok
    CHECK(rt_bit(OT_ENTRY, 1));                    // front-side
    CHECK(!rt_bit(OT_EXIT, 1));                    // back-side
    CHECK(rt_bit(OT_M2M_M2, 2));                   // co-located
    CHECK(rt_bit((uint8_t)(OT_EXIT | OT_TIR), 5)); // total-reflection variant
    CHECK(rt_bit(RT_R0, 7));                       // reflection flag
    CHECK(rt_bit(RT_RINT, 3));                     // inside flag
    CHECK(rt_bit(RT_EXIT_SLAB, 2));                // resolve flag
}

TEST_CASE("ray_state_update - dispatch: o-i entry family")
{
    // eps = 1.5 keeps the slab factor off. An entry defers its medium loss, so the field is
    // kept and only the state advances.

    SECTION("clean entry from air keeps the field, sets cur, resets prev")
    {
        auto C = disp(OT_ENTRY, 1, 1, 0, 3, 0, 0, 1.0, 2.0); // prev_in = 3 is a sentinel
        check_keep(C);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("entry into a parallel slab at nH = 2 keeps prev clear")
    {
        auto C = disp(OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5);
        check_keep(C);
        check_state(C, 0, 4, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("nested entry (already inside) resets to a unit interface and buffers M1")
    {
        auto C = disp(OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0); // inside FOG2, hits FOG3 face
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 2, 3);
        check_rtype(C, RT_NESTED_DEFER);
    }
    SECTION("degenerate o->ii entry buffers M2 and clears the OK bit")
    {
        auto C = disp(OT_OVL_OII, 2, 4, 5, 0, 0, 0, 1.0, 2.0, 0.5);
        check_keep(C);
        check_state(C, 0, 4, 5);
        check_rtype(C, RT_ENTRY_DEFER);
    }
}

TEST_CASE("ray_state_update - dispatch: i-o exit family")
{
    // eps = 1.5, so a cavity exit re-emits with the in-medium close applied and cur cleared.
    // The just-exited material lands on prev_out.

    SECTION("cavity exit multiplies by the medium close and clears cur")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, 1.0, 2.0); // DENSE, lossless
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("a lossy cavity exit carries the medium magnitude")
    {
        auto C = disp(OT_EXIT, 1, 1, 0, 0, 1, 0, 1.2, 2.0); // FOG1, alpha = 2 dB/m over 1.2 m
        check_mult(C, close_at(1, 1.2), TOL);
        check_state(C, 1, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("buffered exit into the same material flushes the buffer to a unit interface")
    {
        auto C = disp(OT_EXIT, 2, 2, 0, 0, 1, 2, 1.0, 2.0); // buf(2) == M1(2): stay in cur
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_SAMEPASS);
    }
    SECTION("buffered exit into a distinct material applies the entered interface")
    {
        auto C = disp(OT_EXIT, 2, 4, 0, 0, 1, 3, 1.2, 2.0);
        // Commit into buffer FOG3: the entered interface interact_with(FOG1, FOG3) is applied (unit
        // here - both eta = 1, att = 0) and the carried field is reset, then the FOG1 medium close
        // applies. So the field becomes the current medium close on a reset (unit) field.
        std::complex<double> c = close_at(1, 1.2);
        CHECK(std::abs((double)C.xprmat(0, 0) - c.real()) < TOL);
        CHECK(std::abs((double)C.xprmat(1, 0) - c.imag()) < TOL);
        CHECK(std::abs((double)C.gain(0) - std::norm(c)) < TOL);
        check_state(C, 1, 3, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("ii-o overlap exit leaves the medium and clears cur")
    {
        auto C = disp(OT_OVL_IIO, 2, 4, 0, 0, 4, 2, 0.9, 2.0); // buf set -> ii-o exit to air
        check_mult(C, close_at(4, 0.9), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
}

TEST_CASE("ray_state_update - dispatch: material-to-material family")
{
    // types 5 (M1 hit first) and 7 (M2 hit first). A cavity transition applies the current
    // medium close and shifts prev <- current.

    SECTION("M1-first crossing enters M2, applies the current medium close")
    {
        auto C = disp(OT_M2M_M1, 2, 4, 5, 0, 4, 0, 1.0, 1.2); // DENSE -> NINE
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 5, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("M2-first crossing enters M1")
    {
        auto C = disp(OT_M2M_M2, 2, 5, 4, 0, 4, 0, 1.0, 1.2); // travelling in DENSE (M2), into NINE (M1)
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 4, 5, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("a crossing from outside any medium sets the fix flag")
    {
        auto C = disp(OT_M2M_M1, 2, 4, 5, 0, 0, 0, 1.0, 1.2); // cur = 0: exit/entry mismatch
        CHECK(rt_bit(C.resolved_type[0], 4));                 // fix flag
        CHECK((int)C.cur_out(0) == 5);
    }
    SECTION("buffered i-i stays in the current material and swaps the buffer")
    {
        auto C = disp(OT_M2M_M1, 2, 2, 3, 0, 1, 2, 1.0, 1.5); // buf(2)==M1(2) -> swap to M2(3)
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 1, 3);
        check_rtype(C, RT_NESTED_DEFER);
    }
}

TEST_CASE("ray_state_update - dispatch: corner families")
{
    // o-i-o (23) and i-o-i (21).

    SECTION("o-i-o outside any medium is a pass-through")
    {
        auto C = disp(OT_COR_OIO, 2, 2, 2, 0, 0, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 0, 0);
        check_rtype(C, RT_ADVANCE);
    }
    SECTION("o-i-o inside a medium exits to air")
    {
        auto C = disp(OT_COR_OIO, 2, 4, 4, 0, 4, 0, 0.8, 2.0);
        check_mult(C, close_at(4, 0.8), TOL);
        check_state(C, 4, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("i-o-i air gap bounded by two faces is treated as an i-i transition")
    {
        // cur = M1 = FOG1, M2 = FOG3: cross into M2 through the air gap
        auto C = disp(OT_COR_IOI, 2, 1, 3, 0, 1, 0, 1.0, 1.5);
        check_mult(C, close_at(1, 1.0), TOL);
        check_state(C, 1, 3, 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("i-o-i without a far material and no buffer terminates")
    {
        auto C = disp(OT_COR_IOI, 2, 1, 0, 0, 1, 0);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
    SECTION("i-o-i with a buffer but no current medium terminates")
    {
        auto C = disp(OT_COR_IOI, 2, 1, 3, 0, 0, 2);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - dispatch: no crossing and unmatched codes")
{
    SECTION("a no-hit ray advances with an identity interface")
    {
        auto C = disp(OT_NOHIT, 1, 0, 0, 0, 0, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 0, 0);
        check_rtype(C, RT_ADVANCE);
    }
    SECTION("a no-hit ray inside a medium keeps the inside flag")
    {
        auto C = disp(OT_NOHIT, 1, 0, 0, 0, 4, 0);
        check_replace(C, 1.0, TOL);
        check_state(C, 0, 4, 0);
        check_rtype(C, RT_SAMEPASS);
    }
    SECTION("an unmatched but OK-flagged code kills the ray")
    {
        auto C = disp((uint8_t)17, 1, 1, 0, 0, 1, 0); // bits ok + corner, not a valid topology
        check_kill(C);
        check_state(C, 0, 1, 0);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - dispatch: reflection pass")
{
    SECTION("front reflection outside any medium keeps the field")
    {
        for (int it : {0, 3}) // EM / scalar reflection
        {
            auto C = disp(OT_ENTRY, 1, 4, 0, 0, 0, 0, 1.0, 2.0, 0.5, it, 0.0);
            check_keep(C);
            check_state(C, 0, 0, 0);
            check_rtype(C, RT_R0);
        }
    }
    SECTION("internal reflection re-emits with the medium close when the gate is off")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, 1.0, 2.0, 0.5, 3, 1.5);
        check_mult(C, close_at(4, 1.0), TOL);
        check_state(C, 0, 4, 0x4000); // no resolved flag; followed
        check_rtype(C, RT_RINT);
    }
    SECTION("internal reflection of a resolvable slab applies S and flags the ray")
    {
        // DENSE cavity at the m = 2 resonance: S = 1/(1 - (1/3)^2) = 1.125
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
        CHECK(std::abs(S - 1.125) < TOL); // oracle self-check at resonance
        check_mult(C, close_at(4, L) * S, TOL);
        check_state(C, 0, enc(4, true), 0);
        check_rtype(C, RT_RINT_SLAB);
    }
    SECTION("a resolved ray is killed on the reflection pass")
    {
        auto C = disp(OT_EXIT, 1, 4, 0, 0, enc(4, true), 0, 1.0, 2.0, 0.5, 3, 0.0);
        check_kill(C);
        check_rtype(C, RT_KILL);
    }
}

TEST_CASE("ray_state_update - flag semantics")
{
    static const auto mtl = base_palette<double>();

    SECTION("the resolved flag lands on current_out as a negative short")
    {
        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0);
        CHECK((int)C.cur_out(0) < 0);
        CHECK(mat_of(C.cur_out(0)) == 4);
        CHECK(flag_of(C.cur_out(0)));
        check_rtype(C, RT_RINT_SLAB);
    }
    SECTION("a wedge entry sets the non-parallel flag on prev_out")
    {
        auto C = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.normals(0, 3) = 0.5; // SBS normal at 60 deg to FBS: |dot| = 0.5
        C.normals(0, 4) = std::sqrt(3.0) / 2.0;
        C.normals(0, 5) = 0.0;
        C.run();
        check_keep(C);
        CHECK((int)C.cur_out(0) == 4);
        CHECK((int)C.prev_out(0) == (int)(short)0x8000); // flag on air: exactly -32768
        CHECK(mat_of(C.prev_out(0)) == 0);
        CHECK(flag_of(C.prev_out(0)));
        CHECK((int)C.buf_out(0) == 0);
        check_rtype(C, RT_ENTRY);
    }
    SECTION("antiparallel and parallel faces leave the non-parallel flag clear")
    {
        auto C = disp(OT_ENTRY, 2, 4, 4, 0, 0, 0); // antiparallel default
        check_state(C, 0, 4, 0);

        auto D = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        D.normals(0, 3) = -1.0; // parallel same-orientation: |dot| = 1
        D.run();
        check_state(D, 0, 4, 0);
    }
    SECTION("the input TIR flag is re-evaluated at the interface angle")
    {
        // out_type carries TIR, but at normal incidence there is no total reflection, so the
        // resolved type must not carry it (the interface re-evaluation overrides the input).
        auto C = disp((uint8_t)(OT_EXIT | OT_TIR), 1, 4, 0, 0, 4, 0);
        CHECK(!rt_bit(C.resolved_type[0], 5));
        check_state(C, 4, 0, 0);
    }
}

TEST_CASE("ray_state_update - Airy factor against the closed form")
{
    // Complex S on a cavity exit, eps = 0 (always resolve). The exit factor is close * S.
    static const auto mtl = base_palette<double>();
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r; // DENSE inside face, 1/3

    SECTION("resonance, antiresonance, and a generic length at 1 GHz")
    {
        double half = half_wave(2.0, FRQ);
        for (double L : {half, 1.5 * half, 0.2718})
        {
            auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
            C.run();
            std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));
            check_mult(C, close_at(4, L) * S, TOL);
            check_state(C, 4, 0, 0);
            check_rtype(C, RT_EXIT_SLAB);
        }
        std::complex<double> S_res = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, half));
        std::complex<double> S_null = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, 1.5 * half));
        CHECK(std::abs(S_res - 1.125) < TOL);
        CHECK(std::abs(S_null - 0.9) < TOL);
    }
    SECTION("the phase tracks the carrier frequency")
    {
        double f2 = 2.4e9, L = 0.2718;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                               std::complex<double>(0.5, 0.3), f2);
        C.run();
        std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, f2, L));
        check_mult(C, close_at(4, L, f2) * S, TOL);
    }
    SECTION("an impedance-matched medium has no cavity: S = 1")
    {
        // eps = mu = 4: the interface admittance matches air, r = 0, so the Airy sum is 1.
        CHECK(calc_transition_gain_mu(0, 0.0, 0.0, 0.0, {4.0, 0.0}, {1.0, 0.0}, {4.0, 0.0}, {1.0, 0.0}) < 1e-12);

        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> matched;
        mtl_matrix_to_map<double>(M, ind, matched);
        set_mu<double>(matched, 4.0, 0.0, 0.0, 0.0);

        double L = 2.0 * half_wave(2.0, FRQ);
        auto C = make1<double>(matched, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.run();
        // n = sqrt(eps * mu) = 4; lossless, so the factor is a pure excess phase, S = 1
        double k0 = 2.0 * QPI * FRQ / C0;
        std::complex<double> closem = std::exp(std::complex<double>(0.0, -k0 * (4.0 - 1.0) * L));
        check_mult(C, closem, TOL);
        check_state(C, 1, 0, 0);
    }
}

TEST_CASE("ray_state_update - survival gate thresholds on eps")
{
    // rho = sqrt(R_near * R_far * medium_gain(2L)). NINE is lossless with inside-face r = 1/2,
    // so rho = 0.25 exactly.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ);
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r;
    CHECK(std::abs(gate_rho(r, r, 1.0) - 0.25) < TOL);
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));
    CHECK(std::abs(S - 4.0 / 3.0) < TOL);

    auto run_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps clearly below rho resolves")
    {
        auto C = run_at(0.225);
        check_mult(C, close_at(5, L) * S, TOL);
        check_state(C, 5, 0, 0);
        check_rtype(C, RT_EXIT_SLAB);
    }
    SECTION("eps clearly above rho re-emits with the gain untouched")
    {
        auto C = run_at(0.275);
        check_mult(C, close_at(5, L), TOL); // no S
        check_state(C, 5, 0, 0);
        check_rtype(C, RT_EXIT);
    }
    SECTION("eps = 0 resolves even a weak near-lossless cavity")
    {
        auto C = run_at(0.0);
        check_mult(C, close_at(5, L) * S, TOL);

        double Lw = 2.0 * half_wave(n_of(7), FRQ);
        std::complex<double> rw = fresnel_te({1.21, 0.0}, {1.0, 0.0}, 0.0).r;
        std::complex<double> Sw = airy_S(rw, rw, phi_one_way(n_of(7), 1.0, FRQ, Lw));
        auto D = make1<double>(mtl, 4, OT_EXIT, 1, 7, 0, 0, 7, 0, Lw, 2.0, 0.5, QPI / 2.0, 0.0);
        D.run();
        check_mult(D, close_at(7, Lw) * Sw, TOL);
    }
    SECTION("eps >= 1 never resolves")
    {
        for (double eps : {1.0, 1.5})
        {
            auto C = run_at(eps);
            check_mult(C, close_at(5, L), TOL);
            check_state(C, 5, 0, 0);
        }
    }
}

TEST_CASE("ray_state_update - lossy medium feeds |phi| from medium_gain")
{
    // For a slab with power-law absorption |phi|^2 must equal the implementation's own
    // medium_gain(L), not an eta-only loss. The test measures medium_gain through a cavity-exit
    // magnitude and then checks S against the closed form. (The mass column cannot be used here:
    // m > 0 marks a lumped-transmission material and is excluded from slab resolution.)
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl["alpha"] = {5.0}; // power-law absorption, dB/m (alphaB defaults to 0)

    const double F10 = 10.0e9; // mass law engages above its clamp (fGHz * dist > 1)
    const double L = 40.0 * half_wave(2.0, F10);

    // Measure medium_gain over d by the exit magnitude: |factor|^2 = medium_gain(d) (eps high, no S)
    auto med_of = [&](double d)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, d, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3), F10);
        C.run();
        return std::norm(vvf(C));
    };
    double g_L = med_of(L);
    double g_2L = med_of(2.0 * L);

    CHECK(g_L < 0.999); // engagement tripwire: the mass column took effect
    // WARN("mass-law g_L = " << g_L << ", g_2L = " << g_2L << " (log-in-distance, not multiplicative)");

    // Cavity exit with eps = 0: extract S from the exit factor and compare with the measured |phi|
    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                           std::complex<double>(0.5, 0.3), F10);
    C.run();
    double k0 = 2.0 * QPI * F10 / C0;
    std::complex<double> close = std::sqrt(g_L) * std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L));
    std::complex<double> S_func = vvf(C) / close;

    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S_right = airy_S(r, r, phi_one_way(2.0, g_L, F10, L));
    std::complex<double> S_wrong = airy_S(r, r, phi_one_way(2.0, 1.0, F10, L)); // eta-only

    CHECK(std::abs(S_func - S_right) < 1e-6);
    if (g_L < 0.99)
        CHECK(std::abs(S_func - S_right) < std::abs(S_func - S_wrong));
}

TEST_CASE("ray_state_update - mass-law material is excluded from slab resolution")
{
    // A mass-law slab carries a lumped transmission surrogate: eta/mu are calibrated for the
    // surface impedance only, so the in-slab phase of phi is not physical and the internal series
    // must not be resummed. This is a material property, not an accuracy threshold, so the gate
    // holds even at eps = 0.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl["m"] = {10.0};

    const double F10 = 10.0e9;
    const double L = 40.0 * half_wave(2.0, F10);

    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0,
                           std::complex<double>(0.5, 0.3), F10);
    C.run();

    // Measure medium_gain over L the same way, with eps high so no S can appear
    auto D = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 1.5,
                           std::complex<double>(0.5, 0.3), F10);
    D.run();
    double g_L = std::norm(vvf(D));

    double k0 = 2.0 * QPI * F10 / C0;
    std::complex<double> close = std::sqrt(g_L) * std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L));

    CHECK(std::abs(vvf(C) / close - 1.0) < 1e-6); // bare close-out, S = 1
    // Not RT_EXIT_SLAB. The exit carries the undeviated bit: no refracted direction was supplied,
    // so the wave is held at its geometric angle inside a mass-law medium and the face is charged
    // from the far side rather than judged from inside it.
    check_rtype(C, RT_EXIT + OT_TIR);

    // Teeth: without the gate this slab would have resolved to a clearly non-unit S
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    CHECK(std::abs(airy_S(r, r, phi_one_way(2.0, g_L, F10, L)) - 1.0) > 1e-2);
}

TEST_CASE("ray_state_update - transmission factor folds into the resolved gain")
{
    // With tf on the slab material, S must use the tf-effective reflectance. The tf model has no
    // public closed form, so the test extracts r_eff^2 = 1 - 1/S from the function's own S at
    // resonance and demands it differ from the bare Fresnel value.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl["tf"] = {0.6};
    mtl["tfB"] = {0.0};

    const double L = 2.0 * half_wave(2.0, FRQ); // resonance: phi^2 = 1

    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
    C.run();
    double k0 = 2.0 * QPI * FRQ / C0;
    std::complex<double> close = std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L)); // lossless
    std::complex<double> S_func = vvf(C) / close;
    std::complex<double> x = 1.0 - 1.0 / S_func; // r_near * r_far * phi^2, phi^2 = 1

    CHECK(std::abs(x.imag()) < 1e-6);
    CHECK(x.real() > 0.0);
    CHECK(x.real() < 1.0);
    // Engagement tripwire: with tf = 0.6 the effective reflectance must differ from 1/9.
    CHECK(std::abs(x.real() - 1.0 / 9.0) > 1e-3);
}

TEST_CASE("ray_state_update - resolve clamp near the Airy pole")
{
    // A lossless slab near grazing drives r^2 toward 1. At resonance with the Airy denominator
    // below 1e-2 the resolve is refused; detuned or at moderate angles it resolves finite.
    static const auto mtl = base_palette<double>();
    const int mat = 4;
    const double n = 2.0;

    auto exit_at = [&](double theta_n, double L)
    {
        double theta_i = std::asin(std::sin(theta_n) / n);
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, mat, 0, 0, mat, 0, L, 2.0, 0.5, QPI / 2.0 - theta_i, 0.0);
        C.run();
        return C;
    };

    SECTION("at the pole the resolve is clamped, not divergent")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ);
        CHECK(std::abs(1.0 - r * r) < 1e-2); // Airy denominator inside the clamp band
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB); // still resolved: the denominator is floored, not refused
        check_state(C, 4, 0, 0);
        check_scale_consistent(C, TOL);
        for (int rr = 0; rr < C.nXPR(); ++rr)
            CHECK(std::isfinite((double)C.xprmat(rr, 0)));
        CHECK(std::isfinite((double)C.gain(0)));
        CHECK(std::abs(vvf(C)) < 1000.0); // floored denominator keeps the factor bounded
    }
    SECTION("a quarter-wave detune leaves the clamp and resolves finite")
    {
        double theta_n = 89.8 * QPI / 180.0;
        double L = 2.5 * half_wave(n, FRQ); // phi^2 = -1: denominator ~ 2
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB);
        check_scale_consistent(C, TOL);
        for (int r = 0; r < C.nXPR(); ++r)
            CHECK(std::isfinite((double)C.xprmat(r, 0)));
        CHECK(std::isfinite((double)C.gain(0)));
    }
    SECTION("a strong but sub-clamp resonance resolves bounded")
    {
        double theta_n = 88.0 * QPI / 180.0;
        double theta_i = std::asin(std::sin(theta_n) / n);
        std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, theta_i).r;
        double L = 2.0 * half_wave(n, FRQ);
        CHECK(std::abs(1.0 - r * r) > 1e-2); // outside the clamp
        auto C = exit_at(theta_n, L);
        check_rtype(C, RT_EXIT_SLAB);
        check_scale_consistent(C, TOL);
        CHECK(std::abs(vvf(C)) < 105.0); // bounded by the clamp design margin
    }
}

TEST_CASE("ray_state_update - cross-pass invariance of the resolve decision")
{
    // The transmission-pass exit and the reflection-pass internal bounce of the same segment
    // must make the same resolve-vs-re-emit decision and apply the same S.
    static const auto mtl = base_palette<double>();
    const double L = 0.123;                                             // generic, off resonance
    std::complex<double> r = fresnel_te({9.0, 0.0}, {1.0, 0.0}, 0.0).r; // NINE, rho = 0.25
    std::complex<double> S = airy_S(r, r, phi_one_way(3.0, 1.0, FRQ, L));

    auto exit_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };
    auto refl_at = [&](double eps)
    {
        auto C = make1<double>(mtl, 3, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, eps);
        C.run();
        return C;
    };

    SECTION("eps below rho: both passes resolve with the identical S")
    {
        auto T = exit_at(0.2);
        auto R = refl_at(0.2);
        check_mult(T, close_at(5, L) * S, TOL);
        check_mult(R, close_at(5, L) * S, TOL);
        check_state(T, 5, 0, 0);
        check_state(R, 0, enc(5, true), 0);
        check_rtype(T, RT_EXIT_SLAB);
        check_rtype(R, RT_RINT_SLAB);
    }
    SECTION("eps above rho: both passes re-emit")
    {
        auto T = exit_at(0.3);
        auto R = refl_at(0.3);
        check_mult(T, close_at(5, L), TOL);
        check_mult(R, close_at(5, L), TOL);
        check_state(T, 5, 0, 0);
        check_state(R, 0, 5, 0x4000); // re-emitted, no resolved flag; followed
        check_rtype(T, RT_EXIT);
        check_rtype(R, RT_RINT);
    }
}

TEST_CASE("ray_state_update - a resolved ray passes an exit transparently")
{
    // Stage current_out into current_in. A reflection resolves the slab and sets the flag; the
    // subsequent transmission exit of the same segment adds only the medium close and clears cur.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);

    auto A = disp(OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, 3, 0.0); // reflection resolves
    CHECK(flag_of(A.cur_out(0)));

    auto B = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, A.cur_out(0), 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
    B.run();
    check_mult(B, close_at(4, L), TOL); // no extra S: the series was already summed
    check_state(B, 4, 0, 0);
    check_rtype(B, RT_EXIT);
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

TEST_CASE("ray_state_update - gain consistency and batch determinism")
{
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(3.0, FRQ); // NINE resonance, rho = 0.25 resolves at eps = 0.15

    // A mixed batch: clean entry, resolvable cavity exit, inconsistent kill, nested replace,
    // resolved pass-through, buffered i-i replace. All scalar, eps = 0.15.
    auto scalar_rows = [&]()
    {
        std::vector<Rsu<double>> v;
        v.push_back(make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, (uint8_t)17, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, enc(4, true), 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 4, OT_M2M_M1, 2, 2, 3, 0, 1, 2, 1.0, 1.5, 0.5, QPI / 2.0, 0.15));
        return v;
    };

    SECTION("scalar mode keeps gainN equal to the xprmat power sum on every row")
    {
        auto v = scalar_rows();
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int rr = 0; rr < 2; ++rr)
                p += (double)B.xprmat(rr, i) * (double)B.xprmat(rr, i);
            CHECK(std::abs((double)B.gain(i) - p) < TOL);
        }
    }
    SECTION("EM mode keeps gainN equal to half the xprmat power sum")
    {
        // Independent EM rows: entry (kept) and cavity exit (resolves). qd_polbasis sets
        // gain = 0.5 * sum by construction, so the invariant holds without knowing the layout.
        std::vector<Rsu<double>> v;
        v.push_back(make1<double>(mtl, 1, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15));
        v.push_back(make1<double>(mtl, 1, OT_EXIT, 1, 5, 0, 0, 5, 0, L, 2.0, 0.5, QPI / 2.0, 0.15));
        auto B = merge_rows(v);
        B.run();
        for (arma::uword i = 0; i < B.gain.n_elem; ++i)
        {
            double p = 0.0;
            for (int rr = 0; rr < 8; ++rr)
                p += (double)B.xprmat(rr, i) * (double)B.xprmat(rr, i);
            CHECK(std::abs((double)B.gain(i) - 0.5 * p) < TOL);
        }
    }
    SECTION("a mixed batch equals the same rows run one at a time")
    {
        auto v = scalar_rows();
        auto B = merge_rows(v);
        B.run();
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i].run();
            CHECK(arma::approx_equal(B.xprmat.col((arma::uword)i), v[i].xprmat.col(0), "absdiff", 0.0));
            CHECK(std::abs((double)B.gain((arma::uword)i) - (double)v[i].gain(0)) == 0.0);
            CHECK((int)B.prev_out((arma::uword)i) == (int)v[i].prev_out(0));
            CHECK((int)B.cur_out((arma::uword)i) == (int)v[i].cur_out(0));
            CHECK((int)B.buf_out((arma::uword)i) == (int)v[i].buf_out(0));
            CHECK((int)B.resolved_type[i] == (int)v[i].resolved_type[0]);
        }
    }
}

TEST_CASE("ray_state_update - optional arguments")
{
    static const auto mtl = base_palette<double>();

    SECTION("null state inputs behave exactly like explicit zero state")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
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

        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_prev_out = false;
        C.run();
        check_mult(C, close_at(4, L) * S, TOL);
        CHECK((int)C.cur_out(0) == 0);
        CHECK((int)C.buf_out(0) == 0);
    }
    SECTION("gainN and xprmatN can each be omitted")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_gain = false;
        A.run();
        check_keep(A); // entry defers the medium loss; xprmat unchanged
        check_state(A, 0, 1, 0);

        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.has_xprmat = false;
        B.run();
        CHECK(std::abs((double)B.gain(0) - (double)B.g_in(0)) < TOL); // gain unchanged at entry
        check_state(B, 0, 1, 0);

        auto C = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_gain = false;
        C.has_xprmat = false;
        CHECK_NOTHROW(C.run()); // pure state update
        check_state(C, 0, 1, 0);
    }
    SECTION("null normal_vecN throws (required VBS plane normal)")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.has_normals = false;
        CHECK_THROWS_AS(A.run(), std::invalid_argument);
    }
    SECTION("with every state pointer null the per-call physics still works")
    {
        auto strip = [](Rsu<double> &C)
        {
            C.has_prev_in = C.has_cur_in = C.has_buf_in = false;
            C.has_prev_out = C.has_cur_out = C.has_buf_out = false;
        };

        // (a) entry defers the medium loss and keeps the field
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(A);
        A.run();
        check_keep(A);

        // (b) unmatched code kills, regardless of state tracking
        auto B = make1<double>(mtl, 4, (uint8_t)17, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(B);
        B.run();
        check_kill(B);

        // (c) no-crossing pass-through resets to an identity interface
        auto C = make1<double>(mtl, 4, OT_NOHIT, 1, 0, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.0);
        strip(C);
        C.run();
        check_replace(C, 1.0, TOL);
    }
}

TEST_CASE("ray_state_update - gain operation semantics and probe handling")
{
    static const auto mtl = base_palette<double>();

    SECTION("keep is bit-identical, including a fully populated EM probe")
    {
        auto C = make1<double>(mtl, 1, OT_ENTRY, 1, 2, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        C.xprmat(0, 0) = 0.31;
        C.xprmat(1, 0) = -0.12;
        C.xprmat(2, 0) = 0.05;
        C.xprmat(3, 0) = 0.21;
        C.xprmat(4, 0) = -0.17;
        C.xprmat(5, 0) = 0.02;
        C.xprmat(6, 0) = 0.44;
        C.xprmat(7, 0) = 0.09;
        double p = 0.0;
        for (int rr = 0; rr < 8; ++rr)
            p += C.xprmat(rr, 0) * C.xprmat(rr, 0);
        C.gain(0) = 0.5 * p;
        C.run();
        check_keep(C);
        check_state(C, 0, 2, 0);
    }
    SECTION("replace ignores the incoming field entirely")
    {
        auto A = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(0.5, 0.3));
        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5,
                               std::complex<double>(-0.1, 0.8));
        A.run();
        B.run();
        CHECK(arma::approx_equal(A.xprmat, B.xprmat, "absdiff", 0.0));
        CHECK(std::abs((double)A.gain(0) - (double)B.gain(0)) == 0.0);
        check_replace(A, 1.0, TOL);
    }
    SECTION("EM replace writes VV and HH, scalar replace writes VV only")
    {
        auto A = make1<double>(mtl, 1, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        A.run();
        check_replace(A, 1.0, TOL); // VV + HH layout for itype < 3

        auto B = make1<double>(mtl, 4, OT_ENTRY, 1, 3, 0, 0, 2, 0, 1.0, 2.0, 0.5, QPI / 2.0, 1.5);
        B.run();
        check_replace(B, 1.0, TOL); // VV-only layout for itype >= 3
        double p = (double)B.xprmat(0, 0) * (double)B.xprmat(0, 0) +
                   (double)B.xprmat(1, 0) * (double)B.xprmat(1, 0);
        CHECK(std::abs((double)B.gain(0) - p) < TOL); // scalar gain has no 0.5 factor
    }
}

TEST_CASE("ray_state_update - energy conservation of a lossless slab")
{
    // air | slab | air at normal incidence. The function owns every magnitude (interface
    // coefficient fed in, then corrected by close * S); the test owns only the vacuum phase per
    // one-way trip and assembles the ports. |R|^2 + |T|^2 = 1.
    static const auto mtl = base_palette<double>();
    const int mat = 4; // DENSE, lossless
    const double L = 0.275;
    const double k0 = 2.0 * QPI * FRQ / C0;

    FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0); // air -> DENSE (r01, t01)
    FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0); // DENSE -> air (r10, t10)

    CHECK(std::abs(std::norm(out.r) - calc_transition_gain(0, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);
    CHECK(std::abs(std::norm(out.t) - calc_transition_gain(1, 0.0, 0.0, 0.0, {1.0, 0.0}, {4.0, 0.0})) < 1e-12);

    auto avv = [](const Rsu<double> &C)
    { return cpx(C.xprmat, 0, 0); };
    auto ev = [&](int it, uint8_t ot, int prev, int cur, double dof, std::complex<double> feed)
    {
        auto C = make1<double>(mtl, it, ot, 1, mat, 0, prev, cur, 0, dof, 2.0, 0.5, QPI / 2.0, 0.0, feed);
        C.run();
        return C;
    };

    auto E = ev(4, OT_ENTRY, 0, 0, 0.5, out.t);                     // entry: keeps t01, sets cur
    auto X = ev(4, OT_EXIT, 0, mat, L, ins.t);                      // exit: t10 * close * S
    auto F = ev(3, OT_ENTRY, 0, 0, 0.5, out.r);                     // front reflection: keeps r01
    auto I = ev(3, OT_EXIT, 0, mat, L, ins.r);                      // internal bounce: r10 * close * S
    auto O = ev(4, OT_EXIT, I.prev_out(0), I.cur_out(0), L, ins.t); // resolved outcouple: t10 * close

    check_keep(E);
    check_state(X, mat, 0, 0);
    check_state(I, 0, enc(mat, true), 0);
    check_state(O, mat, 0, 0);

    // phi_one_way = vac * close; the test supplies one vacuum phase per one-way trip.
    std::complex<double> vac = std::exp(std::complex<double>(0.0, -k0 * L));
    std::complex<double> T = out.t * vac * avv(X);                // t01 * phi * (t10 S)
    std::complex<double> B = out.t * vac * vac * avv(I) * avv(O); // t01 * phi^2 * (r10 S) * t10
    std::complex<double> R = avv(F) + B;                          // r01 + t01 t10 r10 phi^2 S
    CHECK(std::abs(std::norm(R) + std::norm(T) - 1.0) < TOL);
}

TEST_CASE("ray_state_update - energy safety of a lossy slab")
{
    // ABSORB (eta = 4, alpha = 30 dB/m) at resonance. The measured ports must match the oracle
    // ports; the reflection port catches a missing up-trip medium loss.
    static const auto mtl = base_palette<double>();
    const int mat = 6;
    const double L = 2.0 * half_wave(2.0, FRQ);
    const double g = medg(mat, L);
    const double k0 = 2.0 * QPI * FRQ / C0;

    FresnelTE out = fresnel_te({1.0, 0.0}, {4.0, 0.0}, 0.0);
    FresnelTE ins = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0);

    auto avv = [](const Rsu<double> &C)
    { return cpx(C.xprmat, 0, 0); };
    auto ev = [&](int it, uint8_t ot, int prev, int cur, double dof, std::complex<double> feed)
    {
        auto C = make1<double>(mtl, it, ot, 1, mat, 0, prev, cur, 0, dof, 2.0, 0.5, QPI / 2.0, 0.0, feed);
        C.run();
        return C;
    };
    auto X = ev(4, OT_EXIT, 0, mat, L, ins.t);
    auto F = ev(3, OT_ENTRY, 0, 0, 0.5, out.r);
    auto I = ev(3, OT_EXIT, 0, mat, L, ins.r);
    auto O = ev(4, OT_EXIT, I.prev_out(0), I.cur_out(0), L, ins.t);

    std::complex<double> phi = phi_one_way(2.0, g, FRQ, L);
    std::complex<double> S = airy_S(ins.r, ins.r, phi);
    double T_or = std::norm(out.t * ins.t * phi * S);
    double R_or = std::norm(out.r + out.t * ins.t * ins.r * phi * phi * S);
    double A_or = 1.0 - T_or - R_or;
    CHECK(A_or > 0.3); // driven hard enough for the test to mean something

    std::complex<double> vac = std::exp(std::complex<double>(0.0, -k0 * L));
    std::complex<double> T = out.t * vac * avv(X);
    std::complex<double> B = out.t * vac * vac * avv(I) * avv(O);
    std::complex<double> R = avv(F) + B;
    CHECK(std::abs(std::norm(T) - T_or) < TOL);
    CHECK(std::abs(std::norm(R) - R_or) < TOL); // fails if the up trip drops its medium loss
    CHECK(std::norm(R) + std::norm(T) < 1.0 + 1e-9);
}

TEST_CASE("ray_state_update - scalar refraction (interaction type 5)")
{
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);
    std::complex<double> r = fresnel_te({4.0, 0.0}, {1.0, 0.0}, 0.0).r;
    std::complex<double> S = airy_S(r, r, phi_one_way(2.0, 1.0, FRQ, L));

    SECTION("type 5 is accepted and matches scalar transmission at a normal-incidence exit")
    {
        auto C4 = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        auto C5 = make1<double>(mtl, 5, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C4.run();
        C5.run();
        check_mult(C4, close_at(4, L) * S, TOL);
        check_mult(C5, close_at(4, L) * S, TOL);
        check_state(C5, 4, 0, 0);
        check_rtype(C5, RT_EXIT_SLAB);
        CHECK(C5.xprmat.n_rows == 2); // scalar mode: two-row xprmat
    }
    SECTION("type 5 uses the same o-i entry routing as type 4")
    {
        auto C = make1<double>(mtl, 5, OT_ENTRY, 1, 1, 0, 0, 0, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        C.run();
        check_keep(C);
        check_state(C, 0, 1 | 0x4000, 0); // the refracting entry marks the leg as bent
        check_rtype(C, RT_ENTRY);
    }
}

TEST_CASE("ray_state_update - path direction correction (VBS)")
{
    // path_dir_prev supplied at a real angle triggers the VBS construction, which corrects the
    // continuation direction. The reflection mirror is convention-independent.
    static const auto mtl = base_palette<double>();
    double L = 2.0 * half_wave(2.0, FRQ);
    double alpha = 25.0 * QPI / 180.0; // incoming direction 25 deg off the x axis

    SECTION("internal reflection mirrors the incoming direction about the face normal")
    {
        auto C = make1<double>(mtl, 3, OT_EXIT, 1, 4, 0, 0, 4, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        C.has_path_dir = true;
        C.path_dir_prev.set_size(1, 3);
        C.path_dir_prev(0, 0) = std::cos(alpha);
        C.path_dir_prev(0, 1) = std::sin(alpha);
        C.path_dir_prev(0, 2) = 0.0;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.run();
        // mirror about N = (-1, 0, 0): the x component flips, the tangential part is unchanged
        CHECK(std::abs((double)C.path_dirN(0, 0) - (-std::cos(alpha))) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 1) - std::sin(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 2)) < 1e-9);
        CHECK(rt_bit(C.resolved_type[0], 1)); // VBS correction flag
    }
    SECTION("a transparent transition carries the incoming direction through")
    {
        auto C = make1<double>(mtl, 4, OT_M2M_M1, 2, 1, 2, 0, 1, 0, 1.0, 1.5, 0.5, QPI / 2.0, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.set_size(1, 3);
        C.path_dir_prev(0, 0) = std::cos(alpha);
        C.path_dir_prev(0, 1) = std::sin(alpha);
        C.path_dir_prev(0, 2) = 0.0;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.run();
        CHECK(std::abs((double)C.path_dirN(0, 0) - std::cos(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 1) - std::sin(alpha)) < 1e-9);
        CHECK(std::abs((double)C.path_dirN(0, 2)) < 1e-9);
    }
}

TEST_CASE("ray_state_update - accumulated distance ledger")
{
    static const auto mtl = base_palette<double>();

    SECTION("a cavity exit preloads the outgoing leg into both distance columns")
    {
        double dof = 1.0, dfd = 2.0;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 4, 0, 0, 4, 0, dof, dfd, 0.5, QPI / 2.0, 1.5);
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        C.acc_geo_out.zeros(1);
        C.acc_ref_out.zeros(1);
        C.run();
        CHECK(std::abs((double)C.acc_ref_out(0) - dfd) < 1e-9); // refracted (post-exit leg is unrefracted)
        CHECK(std::abs((double)C.acc_geo_out(0) - dfd) < 1e-9); // geometric
    }
    SECTION("a non-terminal entry restarts the in-medium distance at zero")
    {
        auto C = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.5, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        C.acc_geo_out.zeros(1);
        C.acc_ref_out.zeros(1);
        C.run();
        CHECK(std::abs((double)C.acc_ref_out(0)) < 1e-9);
        CHECK(std::abs((double)C.acc_geo_out(0)) < 1e-9);
    }
    SECTION("the accumulated input distance threads into the exit medium loss")
    {
        double dof = 0.7, preload = 0.5;
        double L = dof + preload;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5); // ABSORB
        C.has_acc_geo_in = true, C.has_acc_ref_in = true;
        C.acc_geo_in.zeros(1);
        C.acc_ref_in.zeros(1);
        C.acc_ref_in(0) = preload; // refracted
        C.acc_geo_in(0) = preload; // geometric
        C.run();
        CHECK(std::abs(std::norm(vvf(C)) - medg(6, L)) < 1e-9); // |factor|^2 = medium_gain(acc_in + d(orig,fbs))
    }
    SECTION("the geometric and refracted columns are independent on both sides")
    {
        // A caller that shares one geometric accumulator across frequencies passes the refracted
        // column alone on all but the first, so each side must accept one column without the other.
        double dof = 0.7, preload = 0.5;
        double L = dof + preload;

        const double sentinel = 7.5; // a NULL side must leave the caller's vector exactly as it was

        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5); // ABSORB
        C.has_acc_ref_in = true; // geometric input omitted, reads as 0
        C.acc_ref_in.zeros(1);
        C.acc_ref_in(0) = preload;
        C.has_acc_ref_out = true; // geometric output omitted, write skipped
        C.acc_ref_out.zeros(1);
        C.acc_geo_out.ones(1);
        C.acc_geo_out(0) = sentinel;
        CHECK_NOTHROW(C.run());
        CHECK(std::abs(std::norm(vvf(C)) - medg(6, L)) < 1e-9); // the refracted column still drives the loss
        CHECK(std::abs((double)C.acc_geo_out(0) - sentinel) < 1e-9); // untouched by the call

        auto D = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5);
        D.has_acc_geo_in = true; // refracted input omitted, reads as 0
        D.acc_geo_in.zeros(1);
        D.acc_geo_in(0) = preload;
        D.has_acc_geo_out = true; // refracted output omitted, write skipped
        D.acc_geo_out.zeros(1);
        D.acc_ref_out.ones(1);
        D.acc_ref_out(0) = sentinel;
        CHECK_NOTHROW(D.run());
        CHECK(std::abs(std::norm(vvf(D)) - medg(6, dof)) < 1e-9); // refracted preload absent -> loss over d(orig,fbs) only
        CHECK(std::abs((double)D.acc_ref_out(0) - sentinel) < 1e-9); // untouched by the call
    }
}

TEST_CASE("ray_state_update - permeability affects the refractive index")
{
    // eta = 1, mu = 4 -> n = sqrt(eta*mu) = 2. The exit excess phase must use n, not sqrt(eta).
    arma::mat M = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    set_mu<double>(mtl, 4.0, 0.0, 0.0, 0.0);

    double L = 0.2718;
    auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 1.5); // eps high, no S
    C.run();
    double k0 = 2.0 * QPI * FRQ / C0;
    std::complex<double> expect = std::exp(std::complex<double>(0.0, -k0 * (2.0 - 1.0) * L)); // lossless, n = 2
    check_mult(C, expect, TOL);
}

TEST_CASE("ray_state_update - resonance and coincidence material columns")
{
    // Engagement tripwires for the new material features.
    SECTION("a permittivity resonance shifts the resolved slab factor")
    {
        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> base, res;
        mtl_matrix_to_map<double>(M, ind, base);
        res = base;
        res["resF"] = {1.0}; // resonance at the test frequency
        res["resQ"] = {5.0};
        res["resS"] = {0.5};

        double L = 2.0 * half_wave(2.0, FRQ);
        auto Cb = make1<double>(base, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        auto Cr = make1<double>(res, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, L, 2.0, 0.5, QPI / 2.0, 0.0);
        Cb.run();
        Cr.run();
        CHECK(std::abs(vvf(Cb) - vvf(Cr)) > 1e-3);
    }
    SECTION("coincidence columns are accepted without error")
    {
        arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
        arma::uvec ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        mtl_matrix_to_map<double>(M, ind, mtl);
        mtl["coiF"] = {1.0};
        mtl["coiQ"] = {3.0};
        mtl["coiA"] = {6.0};
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 1, 0, 0, 1, 0, 1.0, 2.0, 0.5, QPI / 2.0, 0.15);
        CHECK_NOTHROW(C.run());
    }
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

TEST_CASE("ray_state_update - type parity between double and float")
{
    SECTION("double") { run_parity<double>(1.0e-9); }
    SECTION("float") { run_parity<float>(1.0e-4); }
}

TEST_CASE("ray_state_update - ray_ind mapping and read-only inputs")
{
    // Full-set arrays (orig, dest, no_interact, state) are read at g = ray_ind[i]; compact-set
    // arrays (fbs, sbs, angle, normals, materials, out_type) at i. No input array is modified.
    auto mtl = base_palette<double>();

    Rsu<double> C;
    C.itype = 4;
    C.eps = 1.5;
    C.mtl = mtl;
    C.has_ray_ind = true;

    const arma::uword n_ray = 5, n_rayN = 2;
    C.orig.zeros(n_ray, 3);
    C.dest.zeros(n_ray, 3);
    for (arma::uword g = 0; g < n_ray; ++g)
        C.dest(g, 0) = (double)(g + 1);
    C.no_interact.set_size(n_ray);
    C.no_interact.fill(1u);
    C.prev_in = sv({5, 6, 5, 7, 5});
    C.cur_in = sv({7, 3, 7, 1, 7}); // FOG3 at g = 1, FOG1 at g = 3
    C.buf_in = sv({0, 0, 0, 0, 0});

    // Compact-set arrays, read at i
    C.fbs.zeros(n_rayN, 3);
    C.sbs.zeros(n_rayN, 3);
    for (arma::uword i = 0; i < n_rayN; ++i)
    {
        C.fbs(i, 0) = 0.4 * (double)(i + 1); // d(orig, fbs) = 0.4, 0.8
        C.sbs(i, 0) = C.fbs(i, 0) + 10.0;
    }
    C.fbs_angle.set_size(n_rayN);
    C.fbs_angle.fill(QPI / 2.0);
    C.out_type.assign(n_rayN, OT_EXIT);
    C.m1 = sv({1, 3}); // exit face materials (match cur at the mapped g)
    C.m2 = sv({0, 0});
    C.ray_ind = {3u, 1u}; // out of order on purpose
    C.normals.zeros(n_rayN, 6);
    C.normals.col(0).fill(-1.0);
    C.normals.col(3).fill(1.0);
    C.prev_out = sv({11111, 11111});
    C.cur_out = sv({11111, 11111});
    C.buf_out = sv({11111, 11111});
    C.resolved_type.assign(n_rayN, (uint8_t)222);
    C.gain = {1.0, 1.0};
    C.xprmat.zeros(2, n_rayN);
    C.xprmat(0, 0) = 1.0;
    C.xprmat(0, 1) = 1.0;

    arma::Col<short> p0 = C.prev_in, c0 = C.cur_in, b0 = C.buf_in;
    arma::mat o0 = C.orig, d0 = C.dest;

    C.run();

    // Row 0 -> g = 3: cur = FOG1 (1), clean cavity exit over d(orig, fbs) = 0.4.
    // Row 1 -> g = 1: cur = FOG3 (3), clean cavity exit over d(orig, fbs) = 0.8.
    check_mult(C, close_at(1, 0.4), TOL, 0);
    check_mult(C, close_at(3, 0.8), TOL, 1);
    CHECK((int)C.prev_out(0) == 1); // just-exited material
    CHECK((int)C.prev_out(1) == 3);
    CHECK((int)C.cur_out(0) == 0);
    CHECK((int)C.cur_out(1) == 0);
    check_rtype(C, RT_EXIT, 0);
    check_rtype(C, RT_EXIT, 1);

    // Input arrays are read-only
    for (arma::uword g = 0; g < n_ray; ++g)
    {
        CHECK((int)C.prev_in(g) == (int)p0(g));
        CHECK((int)C.cur_in(g) == (int)c0(g));
        CHECK((int)C.buf_in(g) == (int)b0(g));
    }
    CHECK(arma::approx_equal(C.orig, o0, "absdiff", 0.0));
    CHECK(arma::approx_equal(C.dest, d0, "absdiff", 0.0));
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

TEST_CASE("ray_state_update - input validation")
{
    // Every malformed input throws std::invalid_argument; the paired well-formed call succeeds.

    { // the baseline must run
        auto C = make_valid();
        CHECK_NOTHROW(C.run());
        check_state(C, 0, 1, 0);
    }

    SECTION("interaction type range")
    {
        for (int it : {6, -1, 99})
        {
            auto C = make_valid();
            C.itype = it;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (int it : {0, 1, 2, 3, 4, 5}) // 5 (scalar refraction) is now valid
        {
            auto C = make_valid();
            C.itype = it;
            int nX = it >= 3 ? 2 : 8; // xprmat width tracks EM (8) vs scalar (2)
            C.xprmat.zeros(nX, 1);
            C.xprmat(0, 0) = 0.5;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("center frequency must be positive and finite")
    {
        for (double f : {0.0, -1.0e9, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid();
            C.freq = f;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
    }
    SECTION("eps must be finite and non-negative")
    {
        for (double e : {-0.1, (double)std::numeric_limits<double>::quiet_NaN(),
                         (double)std::numeric_limits<double>::infinity()})
        {
            auto C = make_valid();
            C.eps = e;
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
        for (double e : {0.0, 0.5, 1.0, 2.0})
        {
            auto C = make_valid();
            C.eps = e;
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("full-set geometry needs three columns")
    {
        for (arma::uword nc : {2u, 4u})
        {
            auto C = make_valid();
            C.orig.set_size(2, nc);
            C.orig.zeros();
            CHECK_THROWS_AS(C.run(), std::invalid_argument);

            auto D = make_valid();
            D.dest.set_size(2, nc);
            D.dest.zeros();
            CHECK_THROWS_AS(D.run(), std::invalid_argument);
        }
    }
    SECTION("compact-set geometry needs three columns")
    {
        for (arma::uword nc : {2u, 4u})
        {
            auto C = make_valid();
            C.fbs.set_size(1, nc);
            C.fbs.zeros();
            CHECK_THROWS_AS(C.run(), std::invalid_argument);

            auto D = make_valid();
            D.sbs.set_size(1, nc);
            D.sbs.zeros();
            CHECK_THROWS_AS(D.run(), std::invalid_argument);
        }
    }
    SECTION("normal_vecN needs six columns, xprmatN needs nXPR rows")
    {
        auto C = make_valid();
        C.normals.set_size(1, 5);
        C.normals.zeros();
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid(); // scalar mode: xprmat must be [2, n_rayN]
        D.xprmat.set_size(8, 1);
        D.xprmat.zeros();
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("full-set arrays must agree on n_ray")
    {
        auto C = make_valid();
        C.dest = arma::join_vert(C.dest, C.dest.row(0)); // 3 rows, orig has 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.cur_in = sv({0, 0, 0});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("the accumulated-distance inputs are sized against n_ray")
    {
        auto C = make_valid();
        C.has_acc_geo_in = true;
        C.acc_geo_in.zeros(3); // n_ray is 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.has_acc_ref_in = true;
        D.acc_ref_in.zeros(3);
        CHECK_THROWS_AS(D.run(), std::invalid_argument);

        auto E = make_valid(); // correctly sized, either column alone
        E.has_acc_geo_in = true;
        E.acc_geo_in.zeros(2);
        CHECK_NOTHROW(E.run());

        auto F = make_valid();
        F.has_acc_ref_in = true;
        F.acc_ref_in.zeros(2);
        CHECK_NOTHROW(F.run());
    }
    SECTION("the accumulated-distance outputs are resized on demand")
    {
        auto C = make_valid(); // n_rayN is 1, both start empty
        C.has_acc_geo_out = true, C.has_acc_ref_out = true;
        CHECK_NOTHROW(C.run());
        CHECK(C.acc_geo_out.n_elem == 1);
        CHECK(C.acc_ref_out.n_elem == 1);

        auto D = make_valid(); // wrong size is corrected, not rejected
        D.has_acc_geo_out = true, D.has_acc_ref_out = true;
        D.acc_geo_out.zeros(7);
        D.acc_ref_out.zeros(7);
        CHECK_NOTHROW(D.run());
        CHECK(D.acc_geo_out.n_elem == 1);
        CHECK(D.acc_ref_out.n_elem == 1);
    }
    SECTION("compact arrays must agree on n_rayN")
    {
        auto C0 = make_valid();
        C0.out_type.assign(2, OT_ENTRY);
        CHECK_THROWS_AS(C0.run(), std::invalid_argument);

        auto C1 = make_valid();
        C1.m1 = sv({1, 1});
        CHECK_THROWS_AS(C1.run(), std::invalid_argument);

        auto C2 = make_valid();
        C2.m2 = sv({0, 0});
        CHECK_THROWS_AS(C2.run(), std::invalid_argument);

        auto C3 = make_valid();
        C3.xprmat = arma::join_horiz(C3.xprmat, C3.xprmat);
        CHECK_THROWS_AS(C3.run(), std::invalid_argument);

        auto C4 = make_valid();
        C4.normals = arma::join_vert(C4.normals, C4.normals);
        CHECK_THROWS_AS(C4.run(), std::invalid_argument);

        auto C5 = make_valid();
        C5.sbs = arma::join_vert(C5.sbs, C5.sbs);
        CHECK_THROWS_AS(C5.run(), std::invalid_argument);
    }
    SECTION("unconditionally required pointers")
    {
        for (int k = 0; k < 9; ++k)
        {
            auto C = make_valid();
            switch (k)
            {
            case 0:
                C.has_orig = false;
                break;
            case 1:
                C.has_dest = false;
                break;
            case 2:
                C.has_fbs = false;
                break;
            case 3:
                C.has_sbs = false;
                break;
            case 4:
                C.has_ni = false;
                break;
            case 5:
                C.has_normals = false;
                break;
            case 6:
                C.has_otype = false;
                break;
            case 7:
                C.has_m1 = false;
                break;
            case 8:
                C.has_m2 = false;
                break;
            }
            CHECK_THROWS_AS(C.run(), std::invalid_argument);
        }
    }
    SECTION("optional pointers may be null")
    {
        for (int k = 0; k < 6; ++k)
        {
            auto C = make_valid();
            switch (k)
            {
            case 0:
                C.has_prev_in = false;
                break;
            case 1:
                C.has_cur_in = false;
                break;
            case 2:
                C.has_buf_in = false;
                break;
            case 3:
                C.has_prev_out = false;
                break;
            case 4:
                C.has_gain = false;
                break;
            case 5:
                C.has_xprmat = false;
                break;
            }
            CHECK_NOTHROW(C.run());
        }
    }
    SECTION("ray_ind bounds and the identity requirement")
    {
        auto C = make_valid();
        C.ray_ind = {5u}; // out of range for n_ray = 2
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.has_ray_ind = false; // null with n_ray != n_rayN
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("material indices must exist in the map")
    {
        auto C = make_valid();
        C.m1 = sv({8}); // the palette has 7 materials
        CHECK_THROWS_AS(C.run(), std::invalid_argument);

        auto D = make_valid();
        D.cur_in = sv({8, 8});
        CHECK_THROWS_AS(D.run(), std::invalid_argument);

        auto E = make_valid();
        E.m1 = sv({0}); // air is always valid
        CHECK_NOTHROW(E.run());

        auto F = make_valid();
        F.cur_in = sv({enc(7, true), enc(7, true)}); // flagged but in range
        CHECK_NOTHROW(F.run());
    }
    SECTION("the material map may be sparse but must be consistent")
    {
        auto C = make_valid();
        C.mtl.erase("a"); // missing columns fall back to defaults
        CHECK_NOTHROW(C.run());

        auto D = make_valid();
        D.mtl["alpha"].resize(6); // others have 7 entries
        CHECK_THROWS_AS(D.run(), std::invalid_argument);
    }
    SECTION("an empty batch is a no-op, not an error")
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
        C.normals.set_size(0, 6);
        C.out_type.clear();
        C.m1.set_size(0);
        C.m2.set_size(0);
        C.prev_in.set_size(0);
        C.cur_in.set_size(0);
        C.buf_in.set_size(0);
        C.prev_out.set_size(0);
        C.cur_out.set_size(0);
        C.buf_out.set_size(0);
        C.resolved_type.clear();
        C.gain.set_size(0);
        C.xprmat.set_size(2, 0);
        CHECK_NOTHROW(C.run());
        CHECK(C.gain.n_elem == 0);
        CHECK(C.xprmat.n_cols == 0);
    }
}
// ================================================================================================
// Diagnostics for the thin-sheet / slab work.
//
// These cases produce numbers rather than gate the build. Each prints a table under
// QD_TEST_VERBOSE and asserts only sanity, so they stay green while the divergences they measure
// are still open.
//
// All of them use scalar mode (itype 4). In scalar mode SLAB_AIRY_FACTOR scales the stored
// xprmat in place and SCALE_BY_MEDIUM does the same, so the applied correction is recoverable
// exactly as out/in - a single complex number, with no polarization basis rebuild in the way.
// The one case that needs both channels (the polarization ratio) runs in EM mode and relies on
// the harness geometry being degenerate: orig, fbs and dest are collinear with the normal, so
// qd_polbasis takes its identity branch and VV maps to TM, HH maps to TE with no mixing.
// ================================================================================================

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
// Index 0 is air, and the tables run to the palette's last material. They used to stop at 8
// entries while the palette carried 8 materials, so a lookup for material 8 read one past the end;
// no case called it, but adding material 9 makes the range explicit rather than latent.
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

TEST_CASE("Diagnostic - Slab factor per polarization")
{
    // The traced solid's TE/TM ratio, and the acceptance test for removing the Tsp * corr pair from
    // qd_slab_resolve's correction mode.
    //
    // Three references are printed. ratio_pre is what the code produced BEFORE that change:
    // |T_X|^2 * |S_X|^2 * Tsp_X, the port's own per-channel transmission times a second copy of it.
    // ratio_post is what it must produce after, with the series factor alone: |T_X|^2 * |S_X|^2.
    // ratio_ref is the per-channel energy target (1 - R_X) * |S_X|^2; the gap between ratio_post and
    // ratio_ref is the TE/TM obliquity asymmetry of a lossy interface, which belongs to the sqrt(1-R)
    // amplitude convention and is settled separately.
    //
    // The assertion is on ratio_post. It is exactly 1 at normal incidence for either version, so a
    // sweep is required to see the change at all.
    //
    // Two corrections over the first run. The probe field is symmetric, because the function
    // REBUILDS the interaction coefficient from cTE/cTM on this path rather than scaling what it
    // was handed - feeding an asymmetric probe measured the rebuild, not the physics. And the slab
    // is eps 1.50, whose critical angle is 54.7 deg, so the sweep stays below total reflection
    // instead of collapsing past 25.9 deg as eps 5.24 did.
    //
    // Channel identification: make1 puts orig, fbs and dest on the x axis with the normal along x,
    // so the incidence plane is degenerate, qd_polbasis takes its identity branch, and VV is TM
    // while HH is TE. That is the opposite of the oblique fixture in the ray_mesh_interact suite.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SLAB = 5; // eps 1.50 - 0.40j

    QD_DIAG("\n# diag: slab factor per polarization, eps 1.50 - 0.40j, L = 100 mm, i-o exit\n");
    QD_DIAG("# critical angle asin(1/n) = %.2f deg; the sweep stays below it\n",
            std::asin(1.0 / std::real(std::sqrt(diag_eps(SLAB, f_hz)))) * 180.0 / QPI);
    QD_DIAG("theta_n,gain_out,VV2_tm,HH2_te,ratio_out,ratio_post,ratio_pre,ratio_ref,"
            "d_vs_post,d_vs_ref,rtype\n");

    for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_EXIT, 1u,
                               /*M1*/ SLAB, /*M2*/ 0, /*prev*/ 0, /*cur*/ SLAB, /*buf*/ 0,
                               /*d_orig_fbs*/ L, /*d_fbs_dest*/ 0.5, /*d_fbs_sbs*/ 0.5,
                               /*grazing angle*/ QPI / 2.0 - theta_n, /*eps: always resolve*/ 0.0);
        C.run();

        double vv2 = C.xprmat(0, 0) * C.xprmat(0, 0) + C.xprmat(1, 0) * C.xprmat(1, 0);
        double hh2 = C.xprmat(6, 0) * C.xprmat(6, 0) + C.xprmat(7, 0) * C.xprmat(7, 0);
        double lo = std::min(vv2, hh2), hi = std::max(vv2, hh2);
        double ratio_out = (hi > 0.0) ? lo / hi : 0.0;

        // Slab-side exit interface at the same angle the function used, per channel.
        std::complex<double> es = diag_eps(SLAB, f_hz), ea(1.0, 0.0);
        double ct = std::cos(theta_n);
        std::complex<double> z1 = std::sqrt(es), z2 = std::sqrt(ea);
        std::complex<double> c2 = std::sqrt(1.0 - (es / ea) * (1.0 - ct * ct));
        std::complex<double> r_te = (z1 * ct - z2 * c2) / (z1 * ct + z2 * c2);
        std::complex<double> r_tm = (z2 * ct - z1 * c2) / (z2 * ct + z1 * c2);
        std::complex<double> t_te = (2.0 * z1 * ct) / (z1 * ct + z2 * c2);
        std::complex<double> t_tm = (2.0 * z1 * ct) / (z2 * ct + z1 * c2);
        std::complex<double> phi = diag_phi(es, L, theta_n, f_hz);
        std::complex<double> S_te = 1.0 / (1.0 - r_te * r_te * phi * phi);
        std::complex<double> S_tm = 1.0 / (1.0 - r_tm * r_tm * phi * phi);
        double R_te = std::norm(r_te), R_tm = std::norm(r_tm);

        auto ratio_of = [](double a, double b)
        { double lo = std::min(a, b), hi = std::max(a, b); return (hi > 0.0) ? lo / hi : 0.0; };

        // Post-fix: port times series, nothing else. Common factors drop out of a ratio.
        double ratio_post = ratio_of(std::norm(t_te) * std::norm(S_te),
                                     std::norm(t_tm) * std::norm(S_tm));
        // Pre-fix: the same with the extra per-channel single-pass transmittance. Both boundaries of
        // this fixture are slab|air at the same angle, so Tsp_X = (1 - R_X)^2.
        double ratio_pre = ratio_of(std::norm(t_te) * std::norm(S_te) * (1.0 - R_te) * (1.0 - R_te),
                                    std::norm(t_tm) * std::norm(S_tm) * (1.0 - R_tm) * (1.0 - R_tm));
        // Energy target, for the amplitude-convention residual.
        double ratio_ref = ratio_of((1.0 - R_te) * std::norm(S_te), (1.0 - R_tm) * std::norm(S_tm));

        QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u\n",
                thd, C.gain(0), vv2, hh2, ratio_out, ratio_post, ratio_pre, ratio_ref,
                ratio_out - ratio_post, ratio_out - ratio_ref,
                (unsigned)C.resolved_type[0]);

        CHECK(std::isfinite(C.gain(0)));
        CHECK(C.gain(0) >= 0.0);
        CHECK(std::abs(C.gain(0) - 0.5 * (vv2 + hh2)) < 1e-9);

        // The correction-mode series factor is S alone. Anything else here is the Tsp * corr pair
        // coming back.
        INFO("theta_n " << thd << ", ratio_out " << ratio_out << ", ratio_post " << ratio_post);
        CHECK(std::abs(ratio_out - ratio_post) < 1e-9);
    }
}

TEST_CASE("Diagnostic - Entry boundary evaluated at two angles")
{
    // One slab crossing, evaluated two ways. ray_mesh_interact applied the entry boundary at the true
    // incidence angle with the complex refraction cosine kept; qd_slab_resolve then re-derived that
    // same boundary from the real projection the tracer was left holding, so the Airy denominator
    // summed a series against a coefficient the path never carried.
    //
    // The fixture hands the function what a tracer would actually have produced: fbs_angle carries the
    // geometric external angle, path_dir_prev carries the direction qd_refract wrote, and the in-slab
    // cosine the function sees is that direction's real projection. Supplying path_dir_prev is what
    // arms the recovery - without it nothing was ever projected and the gate correctly declines.
    //
    // ISOLATION. This case used to run each row twice, dividing a run with air behind the slab by one
    // with an electrically identical twin behind it, whose far Fresnel coefficient vanishes so the
    // series factor is exactly 1. Everything other than the series was common to the two runs and
    // divided out. That construction no longer holds: the twin makes the entry and slab permittivities
    // equal, which is one of the recovery's own decline conditions, so the twin run evaluates its exit
    // port at the real projection while the air run evaluates it at the recovered cosine. The ratio
    // then carries T_exit(cs) / T_exit(ct_traced) on top of the series - up to 4.1 dB at 85 deg - and
    // measures the two things at once.
    //
    // So the series is now isolated against a closed form instead. That was tried once before and was
    // wrong by 4.6 dB, because the exit port was evaluated at an angle no closed form could name; with
    // the port on the recovered cosine the whole leg is analytic and the reconstruction is exact. The
    // lossless control pins it: for a real permittivity the normalization inside qd_refract is inert,
    // both cosines coincide, and the two references must agree exactly.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();

    QD_DIAG("\n# diag: entry boundary, recovered complex angle versus the real projection\n");
    QD_DIAG("theta_ext,ct_traced,|S_applied|,arg_S_applied,|S_exact|,arg_S_exact,|S_traced|,"
            "arg_S_traced,d_vs_exact_dB,d_traced_vs_exact_dB,rtype\n");

    for (int m : {5, 6}) // 5 = lossy eps 1.50 - 0.40j, 6 = lossless control
    {
        QD_DIAG("# material %d (%s)\n", m, (m == 5) ? "lossy eps 1.50 - 0.40j" : "lossless eps 1.50");
        for (double thd : {0.0, 20.0, 40.0, 60.0, 75.0})
        {
            std::complex<double> es = diag_eps(m, f_hz), ea(1.0, 0.0);
            double th_ext = thd * QPI / 180.0;
            double ct_tr = diag_traced_ct(es, th_ext);
            std::complex<double> ct_ex = diag_ct_exact(es, th_ext);

            auto C = make1<double>(mtl, /*scalar transmission*/ 4, OT_EXIT, 1u,
                                   /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                   L, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th_ext,
                                   /*always resolve*/ 0.0);
            // The refracted direction the tracer left behind: |dir . N| is the traced in-slab cosine
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = ct_tr;
            C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct_tr * ct_tr);
            C.run();

            std::complex<double> out = cpx(C.xprmat, 0, 0);
            std::complex<double> probe = cpx(C.x_in, 0, 0);
            unsigned rtype_seen = (unsigned)C.resolved_type[0];

            // Whether the recovery fires at all is decided by use_vbs, not by the angle: at exactly
            // normal incidence the traced direction and the geometric one coincide, nothing was ever
            // projected, and the function correctly declines. It then keeps BOTH the hybrid
            // medium_gain propagation factor and, because the VBS correction is a no-op there, the
            // interface coefficient ray_mesh_interact already wrote - so the probe field survives
            // into the output instead of being rebuilt. The reference has to follow it on both
            // counts, or the row reports the gap between the ITU-R P.2040 Delta form and
            // exp(-k0 Im(n) L) as if it were a defect in the recovery.
            bool recovery_fires = std::abs(ct_tr - std::cos(th_ext)) > 1e-6;

            double dr = L / ct_tr;
            double thk_n = dr * ct_tr; // normal thickness, pure geometry
            const double k0 = 2.0 * QPI * f_hz / C0;
            double n_re = std::real(std::sqrt(es));

            // The carrier: everything the exit leg applies apart from the series.
            std::complex<double> carrier, S_ex;
            if (recovery_fires)
            {
                std::complex<double> T_te, T_tm, S_te, S_tm;
                diag_port(es, ea, ct_ex, true, T_te, T_tm);
                diag_series(es, ea, ea, ct_ex, thk_n, f_hz, true, S_te, S_tm);
                carrier = T_te * diag_medium_vbs(es, dr, ct_tr, L, ct_ex, f_hz, std::cos(th_ext));
                S_ex = S_te;
            }
            else
            {
                std::complex<double> phi_tr =
                    std::sqrt(diag_medium_gain(es, dr, f_hz)) *
                    std::exp(std::complex<double>(0.0, -k0 * n_re * dr * ct_tr * ct_tr));
                std::complex<double> r_tr = diag_r_slabside(es, ea, std::complex<double>(ct_tr, 0.0));
                S_ex = 1.0 / (1.0 - r_tr * r_tr * phi_tr * phi_tr);
                carrier = probe * diag_medium_geo(es, dr, L, f_hz);
            }

            std::complex<double> S_applied = (std::abs(carrier) > 0.0)
                                                 ? out / carrier
                                                 : std::complex<double>(0.0, 0.0);

            // The reference the code would produce if both boundaries and the propagation factor
            // were built from the real projection instead of the recovered cosine. Printed only.
            std::complex<double> phi_tr =
                std::sqrt(diag_medium_gain(es, dr, f_hz)) *
                std::exp(std::complex<double>(0.0, -k0 * n_re * dr * ct_tr * ct_tr));
            std::complex<double> r_tr = diag_r_slabside(es, ea, std::complex<double>(ct_tr, 0.0));
            std::complex<double> S_tr = 1.0 / (1.0 - r_tr * r_tr * phi_tr * phi_tr);

            auto dB = [](std::complex<double> x, std::complex<double> y)
            { return 20.0 * std::log10(std::max(1e-300, std::abs(x)) / std::max(1e-300, std::abs(y))); };

            QD_DIAG("%.1f,%.9f,%.9f,%.3f,%.9f,%.3f,%.9f,%.3f,%.6f,%.6f,%u\n",
                    thd, ct_tr, std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                    std::abs(S_ex), std::arg(S_ex) * 180.0 / QPI,
                    std::abs(S_tr), std::arg(S_tr) * 180.0 / QPI,
                    dB(S_applied, S_ex), dB(S_tr, S_ex), rtype_seen);

            // The series is built from the recovered angle, not its projection.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(S_applied - S_ex) < 1e-9);

            // The lossless control pins the reference itself: there the normalization inside
            // qd_refract is inert, so both cosines coincide and the two references must agree
            // exactly. A gap here means diag_traced_ct no longer matches the map qd_refract applies.
            if (m == 6)
                CHECK(std::abs(S_tr - S_ex) < 1e-12);
        }
    }
}

TEST_CASE("Diagnostic - Series collapse when the previous medium matches the slab")
{
    // What happens when the host layer's Airy denominator is evaluated with a matched sheet as one
    // of its bounding media: r = 0 there and the denominator collapses to exactly 1, removing the
    // internal series.
    //
    // This characterises SLAB_AIRY_FACTOR, and the prev = matched-sheet state is now unreachable by
    // construction - SHEET_EVENT no longer writes a matched sheet into the previous-medium word,
    // because it no longer interrupts the layer there. The rows are kept because the same collapse
    // still occurs for a sheet whose impedance merely happens to be close to the host's, which the
    // merge does not catch and only the transfer-matrix core fixes.
    //
    // Material 7 is electrically identical to material 5 and differs only by carrying thk. The
    // first row of each pair is the ordinary case (previous medium = air, series present); the
    // second replaces it with the matched sheet. |S| and arg(S) should both stay put.
    const double f_hz = FRQ;
    auto mtl = diag_palette<double>();
    const int SLAB = 5, SHEET = 7;

    QD_DIAG("\n# diag: series collapse with a matched sheet as the previous medium\n");
    QD_DIAG("theta_n,L_mm,prev,|S_applied|,arg_S_applied,|S_ref_air|,arg_S_ref_air,d_dB,resolve_bit,rtype\n");

    for (double L : {0.05, 0.10})
        for (double thd : {0.0, 20.0, 40.0, 50.0})
            for (int prev : {0, SHEET})
            {
                double theta_n = thd * QPI / 180.0;
                auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                       /*M1*/ SLAB, /*M2*/ 0, /*prev*/ prev, /*cur*/ SLAB, /*buf*/ 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 0.0);
                C.run();

                std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
                std::complex<double> applied = (std::abs(in) > 0.0) ? out / in : std::complex<double>(0.0, 0.0);
                std::complex<double> es = diag_eps(SLAB, f_hz), ea(1.0, 0.0);
                std::complex<double> close = diag_close(es, L, theta_n, f_hz);
                std::complex<double> S_applied = (std::abs(close) > 0.0) ? applied / close
                                                                         : std::complex<double>(0.0, 0.0);

                std::complex<double> phi = diag_phi(es, L, theta_n, f_hz);
                std::complex<double> r = diag_r_slabside(es, ea, std::complex<double>(std::cos(theta_n), 0.0));
                std::complex<double> S_ref = 1.0 / (1.0 - r * r * phi * phi);
                double d_dB = 20.0 * std::log10(std::max(1e-300, std::abs(S_applied)) /
                                                std::max(1e-300, std::abs(S_ref)));

                QD_DIAG("%.1f,%.1f,%d,%.9f,%.3f,%.9f,%.3f,%.6f,%d,%u\n",
                        thd, L * 1000.0, prev, std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                        std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI, d_dB,
                        (int)rt_bit(C.resolved_type[0], 2), (unsigned)C.resolved_type[0]);

                CHECK(std::isfinite(C.gain(0)));
                CHECK(C.gain(0) >= 0.0);
            }
}

TEST_CASE("Diagnostic - Host layer carried through a matched sheet")
{
    // A sheet crossing where the sheet is electrically identical to the host it sits in. The sheet is
    // not a boundary, so the layer must not be cut there: the accumulated distance has to keep
    // running, the medium words have to stay put, and the series must be left to resolve once at the
    // far side over the whole path.
    //
    // Materials 5 and 7 of the palette differ only in thk, which is exactly the column that does not
    // affect a boundary - so Material::same_as separates them while the wave cannot.
    //
    // The control uses material 3, whose permittivity differs from the host, and must still split the
    // layer in the old way. It has to be an actual sheet: material 4 was tried first and every row
    // came back with resolved_type 0, because its thk column is zero, so is_sheet is false, the case
    // re-derivation returns "not a sheet", and out_type 67 then falls through to the legacy unmatched
    // branch and kills the ray. Correct defensive behaviour, wrong fixture.
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const uint8_t OT_SHEET_FRONT = 67; // isolated sheet, front-side hit
    const int HOST = 5, MATCHED = 7, MISMATCHED = 3;

    QD_DIAG("\n# diag: host layer carried through a matched sheet\n");
    QD_DIAG("sheet_mtl,theta_n,prev_out,cur_out,acc_geo_out,acc_ref_out,gain_out,merged,rtype\n");

    for (int sheet_mtl : {MATCHED, MISMATCHED})
        for (double thd : {0.0, 30.0})
        {
            double theta_n = thd * QPI / 180.0;
            auto C = make1<double>(mtl, 4, OT_SHEET_FRONT, 1u,
                                   /*M1 = the sheet face*/ sheet_mtl, /*M2*/ 0,
                                   /*prev*/ 0, /*cur = host*/ HOST, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 0.0);
            C.has_acc_geo_out = true;
            C.has_acc_ref_out = true;
            C.run();

            bool merged = (mat_of(C.prev_out(0)) == 0) && ((double)C.acc_geo_out(0) > 0.5 * L);
            QD_DIAG("%d,%.1f,%d,%d,%.6f,%.6f,%.9f,%d,%u\n",
                    sheet_mtl, thd, mat_of(C.prev_out(0)), mat_of(C.cur_out(0)),
                    (double)C.acc_geo_out(0), (double)C.acc_ref_out(0), (double)C.gain(0),
                    (int)merged, (unsigned)C.resolved_type[0]);

            CHECK(std::isfinite(C.gain(0)));
            CHECK(C.gain(0) >= 0.0);

            INFO("sheet material " << sheet_mtl << ", theta_n " << thd);
            if (sheet_mtl == MATCHED)
            {
                // The layer continues: nothing about the medium state may change, and the distance
                // accumulated up to the sheet has to survive rather than being reset.
                CHECK(mat_of(C.prev_out(0)) == 0);
                CHECK(mat_of(C.cur_out(0)) == HOST);
                CHECK(std::abs((double)C.acc_geo_out(0) - L) < 1e-9);

                // And the crossing itself is a no-op. The sheet occupies thk of the medium it sits
                // in, so a sheet made of that same medium replaces it with itself: both interfaces
                // are transparent, the series is empty, and the propagation factor cancels against
                // the displacement of an identical medium over the same thickness. Gain 1, exactly.
                CHECK(std::abs((double)C.gain(0) - 1.0) < 1e-9);
            }
            else
            {
                // A real boundary still ends the layer and records the sheet as the medium behind.
                CHECK(mat_of(C.prev_out(0)) == MISMATCHED);
                CHECK(mat_of(C.cur_out(0)) == HOST);
            }
        }
}

TEST_CASE("Diagnostic - Excess-phase walk-off convention")
{
    // SCALE_BY_MEDIUM applies k0 * (n * dist_refract * cos2 - dist_geo * cos2_geo), where the cos^2
    // factors are present only when the tracer did NOT trace the bend (walk_off) and drop to 1 when
    // it did. A thin sheet is never displaced and carries the cos^2 form unconditionally, so the two
    // paths describe different geometries and their phases differ.
    //
    // Both terms sit on the thickness axis. Grouping each into a normal thickness, d_t =
    // dist_refract * ct_t and d_i = dist_geo * ct_i, the applied argument is
    // -k0 * (n * d_t * ct_t - d_i * ct_i), which is the tilted-slab excess phase
    // k0 * d * (n * cos_t - cos_i) referenced to the emergent wavefront rather than to the ray path.
    // An undeviated tracer holds no cos_t, so the branch substitutes cos_i and the expression
    // collapses to k0 * d * cos_i * (n - 1) = k0 * L * cos_i^2 * (n - 1). That is what the cos^2
    // standing in for the untraced lateral shift amounts to.
    //
    // The case isolates that factor. With no path_dir supplied use_vbs is false, so REPLACE_BY_VBS
    // takes its fast return and writes no coefficient; the buffer branch above that return needs
    // mtl_buffer != 0 and the fixture passes 0. eps = 1.5 exceeds any reachable rho, which is
    // bounded by 1, so SLAB_AIRY_FACTOR declines on every row. out / in is the medium factor alone.
    // The lossless material keeps the in-slab angle real, which isolates the convention from every
    // complex-angle effect and puts medium_gain at exactly 1.
    //
    // What this fixture cannot separate. use_vbs false forces walk_off true whatever refl_pass and
    // resolved hold, dist_refract equals dist_geo, and ct_t equals ct_i, so the two cosines and the
    // two length ledgers are one quantity each here; any degree-2 split on the n-term agrees. The
    // recovery also declines on its no-direction gate, so the exact branch is unreachable. The
    // reference below is therefore called in its general form with those equalities passed in as
    // arguments rather than written into the expression, so the row states what it rests on. A
    // fixture that separates them needs path_dir_prev at an angle to orig -> FBS on a leg that still
    // takes this branch, which is issue 7's and not this one's.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SLAB = 6; // lossless eps 1.50

    QD_DIAG("\n# diag: excess-phase walk-off convention, lossless eps 1.50, L = 100 mm\n");
    QD_DIAG("theta_n,arg_applied,arg_cos2_fbs,arg_raw_path,d_vs_cos2_fbs,d_vs_raw\n");

    for (double thd : {0.0, 20.0, 40.0, 60.0, 70.0, 80.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                               /*M1*/ SLAB, /*M2*/ 0, /*prev*/ 0, /*cur*/ SLAB, /*buf*/ 0,
                               L, 0.5, 0.5, QPI / 2.0 - theta_n, /*eps: never resolve*/ 1.5);
        C.run();

        std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
        std::complex<double> applied = (std::abs(in) > 0.0) ? out / in : std::complex<double>(0.0, 0.0);

        double n_re = std::real(std::sqrt(diag_eps(SLAB, f_hz)));
        double k0 = 2.0 * QPI * f_hz / C0;
        double ct = std::cos(theta_n);
        double phase_cos2 = -k0 * (n_re - 1.0) * L * ct * ct; // walk-off form, fbs cosine
        double phase_raw = -k0 * (n_re - 1.0) * L;            // raw path length, no walk-off

        auto wrap = [](double d)
        {
            while (d > 180.0)
                d -= 360.0;
            while (d <= -180.0)
                d += 360.0;
            return d;
        };
        double a = std::arg(applied) * 180.0 / QPI;
        double c2 = phase_cos2 * 180.0 / QPI, rw = phase_raw * 180.0 / QPI;

        QD_DIAG("%.1f,%.3f,%.3f,%.3f,%.3f,%.3f\n", thd, wrap(a), wrap(c2), wrap(rw),
                wrap(a - c2), wrap(a - rw));

        CHECK(std::isfinite(C.gain(0)));
        CHECK(C.gain(0) >= 0.0);

        // Promoted. The applied factor against the general walk-off traversal form, with the
        // fixture's own degeneracies (ct_t = ct_i = ct, dist_refract = dist_geo = L) supplied as
        // arguments. Written this way the row survives issue 7 in either direction: narrowing
        // walk_off leaves this leg on the branch, routing recovered-cosine legs to the exact form
        // leaves it alone because no cosine is recovered here, and a change to which cosine the
        // walk-off form reads lands in diag_medium_walk rather than in this expression.
        //
        // The magnitude is asserted with the phase so the object pinned is the factor and not its
        // argument. The palette carries no alpha and the material is lossless, so medium_gain is 1
        // and this also catches an interface coefficient or an undeclined series reaching this path.
        std::complex<double> ref = diag_medium_walk(diag_eps(SLAB, f_hz), L, ct, L, ct, f_hz);

        INFO("angle " << thd);
        CHECK(std::abs(wrap(a - std::arg(ref) * 180.0 / QPI)) < 1.0e-2);
        CHECK(std::abs(std::abs(applied) - std::abs(ref)) < 1.0e-12);

        // The raw-path convention, the other column the table prints, is excluded by a margin that
        // grows with the angle. The equality above says nothing about whether the row can tell the
        // two conventions apart; this records that it can. The two coincide at normal incidence by
        // construction, so the margin is claimed only past it.
        if (thd > 0.0)
            CHECK(std::abs(wrap(a - rw)) > 1.0);
    }
}

TEST_CASE("Thin sheet - the redo sits on the analytic slab at grazing incidence")
{
    // ray_state_update re-resolves every sheet crossing when medium tracking is on, so in any run
    // with state this function's composite is the one that reaches the output and whatever
    // ray_mesh_interact computed is discarded. It also has no mesh: the fixture supplies none, and
    // the function has no way to learn anything about the face.
    //
    // That is why the grazing clamp here was a fixed extent-to-thickness ratio rather than a
    // geometry, and why it froze the interface past 89.943 deg for every sheet at every thickness.
    // With it gone the composite is the analytic slab all the way to tangency.
    //
    // The sheet is isolated in air, so mtl_current is 0: no host leg is closed and no medium factor
    // is applied, and the gain the function writes is the composite by itself. It is written, not
    // scaled - SHEET_RESOLVE rebuilds xprmat from cTE/cTM with unit amplitude - so the incoming
    // probe field sets the polarization basis and not the level.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;               // eps 5.24 - 0.63j, thk 6 mm
    const uint8_t OT_SHEET_FRONT = 67; // isolated sheet, front-side hit
    const std::complex<double> eps_s = diag_eps(SHEET, f_hz);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    QD_DIAG("\n# diag: sheet redo at grazing incidence, thk 6 mm, eps 5.24 - 0.63j\n");
    QD_DIAG("# the old placeholder clamped past 89.94270 deg; there is no clamp now\n");
    QD_DIAG("angle,gain_out,gain_dB,ana_dB,d_dB,rtype\n");

    for (double thd : {80.0, 85.0, 88.0, 89.5, 89.9, 89.94, 89.96, 89.99, 89.999})
    {
        double theta_n = thd * QPI / 180.0;
        auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_FRONT, 1u,
                               /*M1 = the sheet face*/ SHEET, /*M2*/ 0,
                               /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                               /*d_orig_fbs*/ 1.0, /*d_fbs_dest*/ 0.5, /*d_fbs_sbs*/ 0.5,
                               QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
        C.run();

        double g = (double)C.gain(0);
        double ana = diag_slab_p_avg(eps_s, thk, theta_n, f_hz);

        QD_DIAG("%.4f,%.9e,%.6f,%.6f,%.6f,%u\n",
                thd, g, dB(g), dB(ana), dB(g) - dB(ana), (unsigned)C.resolved_type[0]);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(g));
        CHECK(g >= 0.0);
        CHECK(std::abs(dB(g) - dB(ana)) < 1.0e-6);
    }
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
//   res_*           Permittivity resonances, as increments. A resonance is carried by every
//                   INTERFACE quantity - the admittance z = sqrt((eps + res)/mu) and every cosine,
//                   since the transverse invariant is (eps + res) mu (1 - c^2) - and by no
//                   propagation INDEX, which stays n = sqrt(eps mu). The propagation factor
//                   exp(-j k0 n d c) therefore moves with a resonance through its cosine and not
//                   through its index, and is unmoved only at normal incidence.
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

    // Interface permittivities. They carry the resonance; the propagation indices below do not.
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
    // Both indices are resonance-excluded and both cosines are not: a resonant sheet, and a
    // resonant host under it, propagate at sqrt(eps mu) along a cosine set by eps + res.
    const std::complex<double> phi =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_slab * mu_slab) * thk * cs);
    const std::complex<double> e_disp = disp_far ? e_far : e_near;
    const std::complex<double> i_disp = disp_far ? i_far : i_near;
    const std::complex<double> mu_disp = disp_far ? mu_far : mu_near;
    const std::complex<double> c_disp = root(1.0 - (w_near / (i_disp * mu_disp)) * s2);
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_disp * mu_disp) * thk * c_disp);
    const std::complex<double> phi_t = phi / phi_disp;

    // The sheet's tf, on both of its faces. Skipped outright at tf = 0, so that case executes
    // nothing it did not execute before.
    if (lump.tf != 0.0 || lump.tf_near != 0.0 || lump.tf_far != 0.0)
    {
        double kn_te = diag_tf_port(r_ns_te, lump.tf, lump.tf_near),
               kn_tm = diag_tf_port(r_ns_tm, lump.tf, lump.tf_near);
        double kf_te = diag_tf_port(r_sf_te, lump.tf, lump.tf_far),
               kf_tm = diag_tf_port(r_sf_tm, lump.tf, lump.tf_far);
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
    std::complex<double> res{0.0, 0.0}; // permittivity resonance: interface terms only, see DiagLump
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
// through the trailing arguments, with the same split as DiagLump: the admittances and the cosines
// read eps + res, the layer phase's index reads eps. The interface gains are an end quantity on
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

    // Interface permittivities carry the resonance; the layer phase's index does not.
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
            std::complex<double> nj = std::sqrt(L.e * L.mu); // resonance-excluded
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

    const std::complex<double> e_d = disp_far ? e_far : e_near;
    const std::complex<double> i_d = disp_far ? i_far : i_near;
    const std::complex<double> mu_d = disp_far ? mu_far : mu_near;
    const std::complex<double> c_disp = root(1.0 - (w_near / (i_d * mu_d)) * s2);
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_d * mu_d) * t_disp * c_disp);

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
// amplitude factor - the conductivity loss over d / |Re c|, its alpha, and the mass term at the
// real incidence cosine mass_cos - and starts the next segment as its incidence half-space.
// Nothing returns across it, so the reflect port is that of the first segment alone. With no
// coherent layer at all this is the mass-law route of a single sheet.
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
            if (tfm[j] != 0.0 || tfm[j + 1] != 0.0)
            {
                double k = diag_tf_port(r, tfm[j], tfm[j + 1]);
                t *= k, tb *= k;
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
                    std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(L.e * L.mu) * L.d * c[j + 1]);
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
            if (mass)
            {
                const DiagLayer &L = layers[q - 1];
                double dist = L.d / std::abs(std::real(c[q]));
                if (q - p > 1) // a coherent layer in between: a new plate
                    run = 0, X_plate = 0.0, paid_dB = 0.0;
                double mass_dB;
                if (run == 0)
                    mass_dB = diag_mass_TL(L.m, L.fRef, f_hz, L.d, mass_cos);
                else
                    mass_dB = plate_TL(X_plate + std::pow((f_hz * 1e-9 / L.fRef) * L.d, 0.05 * L.m)) - paid_dB;
                X_plate += std::pow((f_hz * 1e-9 / L.fRef) * L.d, 0.05 * L.m), paid_dB += mass_dB, ++run;
                double loss_dB = L.alpha * dist + mass_dB;
                Tt *= std::sqrt(diag_medium_gain(L.e * L.mu, dist, f_hz) * std::pow(10.0, -0.1 * loss_dB));
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
    const std::complex<double> e_d = disp_far ? e_far : e_near;
    const std::complex<double> mu_d = disp_far ? mu_far : mu_near;
    const std::complex<double> phi_disp =
        std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_d * mu_d) * t_disp * c[disp_far ? N + 1 : 0]);

    const bool same = !(e_far != e_near || mu_far != mu_near || res_far != res_near);
    const std::complex<double> front = same ? 1.0 / phi_disp : (disp_far ? std::complex<double>(1.0, 0.0) : 1.0 / (phi_disp * phi_disp));

    D.T_te = D.fx_te * Tp[0] / phi_disp, D.T_tm = D.fx_tm * Tp[1] / phi_disp;
    D.R_te = front * Rp[0], D.R_tm = front * Rp[1];
    return D;
}

TEST_CASE("Glued sheet - the near-to-far flux conversion")
{
    // qd_slab_resolve produce mode applies the near-to-far obliquity factor once, on the transmit
    // port, so that a glued sheet reports the same energy as the equivalent traced entry. The branch
    // is gated on near != far, so only sheet_case 2 (glued entry, air | sheet | solid) and
    // sheet_case 3 (glued exit, solid | sheet | air) reach it. It shipped unmeasured: every existing
    // sheet case is air | sheet | air or has near == far on every row.
    //
    // Six stacks. Two exercise the trivial branch and four the non-trivial one. The embedded row is
    // there so the gate is pinned as near == far rather than as "far is air": it is a sheet inside a
    // medium, where the factor must still be exactly 1.
    //
    // ISOLATION. The composite is what is measured, so nothing else may multiply it.
    //   - Entry rows run with mtl_current = 0, so SHEET_EVENT applies no host factors at all and the
    //     written field is the composite outright. Those rows are asserted complex.
    //   - Rows inside a medium close out a host leg. eps = 1.5 puts the survival gate above any
    //     reachable rho, so the correction-mode series declines and SLAB_AIRY_FACTOR applies
    //     nothing; SCALE_BY_MEDIUM then contributes sqrt(medium_gain) with a phase. Those rows are
    //     asserted per-channel in power, where the phase drops out. That is the right target anyway:
    //     fx is a real positive scale and cannot move a phase, and the composite phase is already
    //     pinned by the air | sheet | air cases.
    //
    // Channel identification: make1 puts orig, fbs and dest on the x axis with the normal along x,
    // so the incidence plane is degenerate, qd_polbasis takes its identity branch, and VV is TM
    // while HH is TE, for the reflected direction as well as the transmitted one.
    //
    // The exit sweeps stop at 50 deg. The internal critical angle of eps 1.50 against air is
    // 54.7 deg, and past it the slab-side TM reflection magnitude exceeds 1 for a lossy sheet, which
    // apply_tf clamps; that is a separate question from this one and is kept out of the measurement.
    const double f_hz = FRQ;
    const double thk = 0.006; // the sheet material's own thickness column
    const double L = 0.10;    // host path up to the sheet, only read on the rows inside a medium
    auto mtl = diag_palette<double>();

    const uint8_t OT_SHEET_ISO = 67;   // ok | sheet, no co-located face
    const uint8_t OT_GLUED_ENTRY = 71; // ok | front | co-located | sheet -> sheet_case 2
    const uint8_t OT_GLUED_EXIT = 69;  // ok | co-located | sheet      -> sheet_case 3
    const int SHEET = 3, LOSSY = 1, LO = 5, LO_CLEAN = 6;

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2;       // the co-located non-sheet face, 0 when there is none
        int cur;      // medium the ray is travelling in
        int e_near;   // palette index of the near medium (0 = air)
        int e_far;    // palette index of the far medium
        bool disp_far;
        double lim;   // highest angle swept
        bool trivial; // near == far, so the factor must be exactly 1
    };
    const std::vector<Row> rows = {
        {"isolated   air | sheet | air", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 85.0, true},
        {"embedded   LO_CLEAN | sheet | LO_CLEAN", OT_SHEET_ISO, 1u, 0, LO_CLEAN, LO_CLEAN, LO_CLEAN, false, 50.0, true},
        {"entry      air | sheet | LOSSY", OT_GLUED_ENTRY, 2u, LOSSY, 0, 0, LOSSY, false, 85.0, false},
        {"entry      air | sheet | LO", OT_GLUED_ENTRY, 2u, LO, 0, 0, LO, false, 85.0, false},
        {"exit       LO_CLEAN | sheet | air", OT_GLUED_EXIT, 2u, LO_CLEAN, LO_CLEAN, LO_CLEAN, 0, true, 50.0, false},
        {"exit       LO | sheet | air", OT_GLUED_EXIT, 2u, LO, LO, LO, 0, true, 50.0, false},
    };

    QD_DIAG("\n# diag: glued-sheet near-to-far flux conversion, sheet eps 5.24 - 0.63j, thk 6 mm\n");
    QD_DIAG("# fx_dB is what the composite carries against one built with the factor omitted\n");
    QD_DIAG("stack,theta_n,fx_te,fx_tm,fx_te_dB,fx_tm_dB,P_te_meas,P_te_ref,P_tm_meas,P_tm_ref,"
            "d_te,d_tm,rtype\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto C = make1<double>(mtl, /*EM transmission*/ 1, R.otype, R.nH,
                                   /*M1 = the sheet face*/ SHEET, /*M2*/ R.m2,
                                   /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n,
                                   /*eps: the host series must not fire*/ 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(diag_eps(R.e_near, f_hz), diag_eps(SHEET, f_hz),
                                               diag_eps(R.e_far, f_hz), thk, ct, f_hz,
                                               R.disp_far, false);

            // The host leg the crossing closes out. Lossless hosts contribute exactly 1 in
            // magnitude; the lossy one contributes the ITU-R P.2040 path form over L, which is what
            // SCALE_BY_MEDIUM applies with no path_dir supplied.
            double host_g = (R.cur == 0) ? 1.0 : diag_medium_gain(diag_eps(R.cur, f_hz), L, f_hz);

            double vv2 = C.xprmat(0, 0) * C.xprmat(0, 0) + C.xprmat(1, 0) * C.xprmat(1, 0); // TM
            double hh2 = C.xprmat(6, 0) * C.xprmat(6, 0) + C.xprmat(7, 0) * C.xprmat(7, 0); // TE
            double P_te = std::norm(D.T_te) * host_g, P_tm = std::norm(D.T_tm) * host_g;

            QD_DIAG("%s,%.1f,%.6f,%.6f,%+.4f,%+.4f,%.9e,%.9e,%.9e,%.9e,%.2e,%.2e,%u\n",
                    R.name, thd, D.fx_te, D.fx_tm,
                    20.0 * std::log10(D.fx_te > 0.0 ? D.fx_te : 1.0),
                    20.0 * std::log10(D.fx_tm > 0.0 ? D.fx_tm : 1.0),
                    hh2, P_te, vv2, P_tm, hh2 - P_te, vv2 - P_tm,
                    (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0); // the ray must survive: case 4 kills, and so does a
                                              // hit code that fails the sheet re-derivation

            // The composite, per channel, against the hand-built stack.
            CHECK(std::abs(hh2 - P_te) < 1.0e-9);
            CHECK(std::abs(vv2 - P_tm) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (P_te + P_tm)) < 1.0e-9);

            // Where nothing else multiplies the composite, pin the phase too.
            if (R.cur == 0)
            {
                CHECK(std::abs(cpx(C.xprmat, 0, 0) - D.T_tm) < 1.0e-9);
                CHECK(std::abs(cpx(C.xprmat, 0, 3) - D.T_te) < 1.0e-9);
            }

            if (R.trivial)
            {
                // near == far: the factor is identically 1, not merely close to it. This is what
                // separates the two branches, and it holds for a sheet embedded in a medium as well
                // as for one in air.
                CHECK(D.fx_te == 1.0);
                CHECK(D.fx_tm == 1.0);
            }
            else
            {
                // And the non-trivial branch is worth enough that the assertion above cannot pass
                // with the factor removed. It is 3.6 dB at normal incidence for the lossy entry, so
                // this does not rely on grazing angles.
                CHECK(std::abs(20.0 * std::log10(D.fx_te)) > 0.5);
                CHECK(std::abs(hh2 - P_te / (D.fx_te * D.fx_te)) > 1.0e-6);
                CHECK(std::abs(vv2 - P_tm / (D.fx_tm * D.fx_tm)) > 1.0e-6);
            }
        }

    // The reflect port carries no flux conversion, because a reflection turns around in the near
    // medium where the obliquity ratio is 1. Measured on the entry rows only, where mtl_current is 0
    // and the reflection pass applies nothing beyond the composite itself.
    QD_DIAG("\n# diag: glued-sheet reflect port, which carries no flux conversion\n");
    QD_DIAG("stack,theta_n,R_te_meas,R_te_ref,R_tm_meas,R_tm_ref,d_te,d_tm,rtype\n");

    for (const auto &R : rows)
    {
        if (R.cur != 0)
            continue;
        for (double thd : {0.0, 20.0, 40.0, 60.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto C = make1<double>(mtl, /*EM reflection*/ 0, R.otype, R.nH,
                                   SHEET, R.m2, /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(diag_eps(R.e_near, f_hz), diag_eps(SHEET, f_hz),
                                               diag_eps(R.e_far, f_hz), thk, ct, f_hz,
                                               R.disp_far, false);

            std::complex<double> vv = cpx(C.xprmat, 0, 0), hh = cpx(C.xprmat, 0, 3);
            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%u\n",
                    R.name, thd, std::abs(hh), std::abs(D.R_te), std::abs(vv), std::abs(D.R_tm),
                    std::abs(hh - D.R_te), std::abs(vv - D.R_tm), (unsigned)C.resolved_type[0]);

            INFO(R.name << " reflect, theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            CHECK(std::abs(hh - D.R_te) < 1.0e-9);
            CHECK(std::abs(vv - D.R_tm) < 1.0e-9);
        }
    }
}

TEST_CASE("Sheet in a medium - the composite at the exact in-medium cosine")
{
    // Issue 2. SHEET_RESOLVE hands qd_slab_resolve produce mode ct_geo = abs_cos_theta_t as the
    // incidence cosine, and produce mode builds every term of the composite from it on the
    // assumption that it is a real incidence angle in mat_near. When mat_near is a traced solid it
    // is not: the propagation cosine there is complex and abs_cos_theta_t is the real projection
    // qd_refract left behind. This case measures the gap.
    //
    // What separates it from the glued-sheet case above. That one supplies no path_dir, so
    // abs_cos_theta_t equals the fbs angle and the cosine handed to produce mode is a genuine real
    // incidence cosine; the reference is then built at the same value and the case is a check on the
    // conversion's form rather than on its argument. Here path_dir_prev carries the refracted
    // direction a tracer would have written, so the two separate and the reference is built at the
    // exact complex cosine instead.
    //
    // Geometry, the same one the exit-boundary case uses. A ray crosses air | LO at the external
    // angle, the tracer writes the refracted direction, and the sheet is met after 100 mm of normal
    // thickness. fbs_angle stays the external angle because the geometric ray was never bent.
    //
    // Two stacks:
    //   embedded    LO | sheet | LO,  sheet_case 1, near == far so the flux conversion is 1
    //   glued exit  LO | sheet | air, sheet_case 3, near != far so the conversion is live
    // and both ports of each, so the reflect port is covered as well as the transmit port.
    //
    // Isolation. eps = 1.5 puts the correction-mode survival gate above any reachable rho, so
    // SLAB_AIRY_FACTOR declines and the only thing multiplying the composite is the host leg
    // SCALE_BY_MEDIUM closes out. That leg is reproduced from the closed forms already used by the
    // exit-boundary case: the exact inhomogeneous form on the transmission pass, where the recovery
    // is armed, and the walk-off form on the reflection pass, where it is not.
    //
    // Modes. The scalar run pins the complex coefficient outright. The EM run pins the gain, which
    // is the Frobenius norm of the Jones matrix and therefore invariant under the polarization
    // basis, so it measures both channels without depending on which matrix element each lands in.
    // The refracted direction has a transverse component, so the basis is not the degenerate one the
    // glued-sheet case relies on.
    const double f_hz = FRQ;
    const double thk = 0.006; // the sheet material's own thickness column
    const double L = 0.10;    // normal thickness of the host layer up to the sheet
    auto mtl = diag_palette<double>();

    const uint8_t OT_SHEET_ISO = 67;  // ok | sheet, no co-located face   -> sheet_case 1
    const uint8_t OT_GLUED_EXIT = 69; // ok | co-located | sheet          -> sheet_case 3
    const int SHEET = 3, LO = 5;

    const std::complex<double> e_h = diag_eps(LO, f_hz);    // host, eps 1.50 - 0.40j
    const std::complex<double> e_s = diag_eps(SHEET, f_hz); // sheet, eps 5.24 - 0.63j
    const std::complex<double> e_air(1.0, 0.0);

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2;                    // the co-located non-sheet face, 0 when there is none
        std::complex<double> e_far;
        bool disp_far;             // the sheet sits on the side that is not the solid
        bool trivial;              // near == far, so the flux conversion is identically 1
    };
    const std::vector<Row> rows = {
        {"embedded   LO | sheet | LO", OT_SHEET_ISO, 1u, 0, e_h, false, true},
        {"glued exit LO | sheet | air", OT_GLUED_EXIT, 2u, LO, e_air, true, false},
    };

    QD_DIAG("\n# diag: sheet reached from inside a traced solid, host eps 1.50 - 0.40j over 100 mm,\n");
    QD_DIAG("# sheet eps 5.24 - 0.63j, thk 6 mm, 1 GHz. The composite is built at ct_traced; the\n");
    QD_DIAG("# reference is the same composite at the exact complex in-medium cosine.\n");
    QD_DIAG("# A is the composite amplitude moving with the cosine, B is the flux conversion failing\n");
    QD_DIAG("# to cancel against the entry that produced it, and A + B is the total on the TE port.\n");
    QD_DIAG("stack,theta_ext,ct_traced,ct_exact_re,ct_exact_im,A_te_dB,B_te_dB,tot_te_dB,tot_tm_dB,"
            "R_te_dB,R_tm_dB,d_T_scalar,d_T_gain,d_R_scalar,d_R_gain,d_host_T,d_host_R,rtype_T,rtype_R\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(e_h, th);              // the real projection the tracer holds
            std::complex<double> cx = diag_ct_exact(e_h, th); // the exact complex in-medium cosine
            double dr = L / ct;                              // refracted path up to the sheet

            // The gate use_vbs applies. At normal incidence the projection and the exact value
            // coincide, the recovery declines and the traversal keeps its walk-off form, so that row
            // is a control: it must read zero before the fix as well as after.
            bool armed = std::abs(ct - std::cos(th)) > 1.0e-6;

            std::complex<double> Mt = armed ? diag_medium_vbs(e_h, dr, ct, L, cx, f_hz, std::cos(th))
                                            : diag_medium_geo(e_h, L, L, f_hz);
            // The reflect pass closes a bent leg with the same form; the walk-off form is left
            // to a wave that was not bent, which here is the normal-incidence control
            std::complex<double> Mr = armed ? Mt : diag_medium_walk(e_h, dr, ct, L, std::cos(th), f_hz);

            // The reference, at the exact cosine, and the same composite at the projection, which is
            // what the code built before the recovery was handed to the sheet path. The second is
            // printed and used to record the size of the defect; it is never the target.
            DiagSheet Xs = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cx, f_hz, R.disp_far, true);
            DiagSheet Xe = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cx, f_hz, R.disp_far, false);
            std::complex<double> cp(ct, 0.0);
            DiagSheet Pe = diag_sheet_composite_cx(e_h, e_s, R.e_far, thk, cp, f_hz, R.disp_far, false);

            auto run = [&](int itype, double host_len)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH,
                                       /*M1 = the sheet face*/ SHEET, /*M2*/ R.m2,
                                       /*prev = what the host was entered through*/ 0,
                                       /*cur*/ LO, /*buf*/ 0,
                                       host_len, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th,
                                       /*eps: the host series must not fire*/ 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };

            auto Ct = run(4, L); // scalar transmission
            auto Ce = run(1, L); // EM transmission
            auto Cr = run(3, L); // scalar reflection
            auto Cx = run(0, L); // EM reflection

            // The same two crossings over twice the host path. Only the traversal changes: the
            // composite is a function of the cosine and the materials and is identical on both, so
            // it cancels out of the ratio below.
            auto Ct2 = run(4, 2.0 * L);
            auto Cr2 = run(3, 2.0 * L);

            std::complex<double> refTs = Xs.T_te * Mt, refRs = Xs.R_te * Mr;
            double refTg = 0.5 * (std::norm(Xe.T_te * Mt) + std::norm(Xe.T_tm * Mt));
            double refRg = 0.5 * (std::norm(Xe.R_te * Mr) + std::norm(Xe.R_tm * Mr));

            double d_Ts = std::abs(cpx(Ct.xprmat, 0, 0) - refTs);
            double d_Tg = std::abs((double)Ce.gain(0) - refTg);
            double d_Rs = std::abs(cpx(Cr.xprmat, 0, 0) - refRs);
            double d_Rg = std::abs((double)Cx.gain(0) - refRg);

            // The host leg on its own, isolated from the composite entirely.
            //
            // Dividing the written field by a hand-built composite would only work while the code
            // and that composite are built at the same cosine, so it cannot survive a change to
            // which cosine the composite uses. Two runs that differ only in host path length can:
            // the composite is identical on both and cancels in the ratio, whatever it is. Both
            // traversal forms are pure exponentials in the path length, so the ratio over L and 2L
            // is the factor over L itself.
            //
            // diag_medium_vbs is also asserted by the exit-boundary case; diag_medium_walk is not
            // asserted anywhere else, which is why the reflect pass gets its own row.
            double d_hT = std::abs(cpx(Ct2.xprmat, 0, 0) / cpx(Ct.xprmat, 0, 0) - Mt);
            double d_hR = std::abs(cpx(Cr2.xprmat, 0, 0) / cpx(Cr.xprmat, 0, 0) - Mr);

            auto dB = [](std::complex<double> a, std::complex<double> b)
            { return 20.0 * std::log10(std::max(1e-300, std::abs(a)) / std::max(1e-300, std::abs(b))); };

            double B_te = (Xe.fx_te > 0.0 && Pe.fx_te > 0.0) ? 20.0 * std::log10(Pe.fx_te / Xe.fx_te) : 0.0;
            double tot_te = dB(Pe.T_te, Xe.T_te), tot_tm = dB(Pe.T_tm, Xe.T_tm);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,"
                    "%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%u,%u\n",
                    R.name, thd, ct, std::real(cx), std::imag(cx),
                    tot_te - B_te, B_te, tot_te, tot_tm,
                    dB(Pe.R_te, Xe.R_te), dB(Pe.R_tm, Xe.R_tm),
                    d_Ts, d_Tg, d_Rs, d_Rg, d_hT, d_hR,
                    (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO(R.name << ", theta_ext " << thd);

            // The ray must survive: sheet_case 4 kills, and so does a hit code that fails the sheet
            // re-derivation.
            REQUIRE(Ct.resolved_type[0] != 0);
            REQUIRE(Cr.resolved_type[0] != 0);
            REQUIRE(std::isfinite(Ct.gain(0)));
            REQUIRE(std::isfinite(Cr.gain(0)));

            // The complex builder reproduces the measured one at a real cosine, term for term. This
            // is what carries the 1.67e-15 bound of the glued-sheet case over to this reference.
            DiagSheet Pd = diag_sheet_composite(e_h, e_s, R.e_far, thk, ct, f_hz, R.disp_far, false);
            CHECK(std::abs(Pd.T_te - Pe.T_te) < 1.0e-15);
            CHECK(std::abs(Pd.T_tm - Pe.T_tm) < 1.0e-15);
            CHECK(std::abs(Pd.R_te - Pe.R_te) < 1.0e-15);
            CHECK(std::abs(Pd.R_tm - Pe.R_tm) < 1.0e-15);

            // Two conditions the measurement depends on, asserted so that a failure of either shows
            // up as itself rather than as an unexplained residual.
            //
            // The apply_tf clamp of section 5 must not engage. It caps the slab-side TM reflection at
            // unity past the internal critical angle of the near | far pair, and this sweep stays
            // below that angle at every row because the in-medium traced angle reaches only 51.7 deg
            // at 85 deg of external incidence.
            std::complex<double> zs = std::sqrt(e_s), zf = std::sqrt(R.e_far);
            std::complex<double> r_slab_p = (zf * Pe.c_slab - zs * Pe.c_far) / (zf * Pe.c_slab + zs * Pe.c_far);
            std::complex<double> r_slab_x = (zf * Xe.c_slab - zs * Xe.c_far) / (zf * Xe.c_slab + zs * Xe.c_far);
            CHECK(std::abs(r_slab_p) < 1.0);
            CHECK(std::abs(r_slab_x) < 1.0);

            // The displacement budget must not cap, so the reference may divide out the full
            // thickness. Air costs nothing to displace and is uncapped, so only the embedded stack
            // needs it: the budget is 80 percent of the host path the crossing can see.
            if (!R.disp_far)
                CHECK(0.8 * (dr + 0.5) * ct > thk);

            // The two cosines differ, so the case can fail. This stays true after the fix and is not
            // an assertion about the defect.
            if (thd > 0.0)
                CHECK(std::abs(cx - cp) > 1.0e-6);

            // The flux conversion is identically 1 wherever near == far, at a complex cosine as well
            // as at a real one. The gate is near == far, not "far is air".
            if (R.trivial)
            {
                CHECK(Xe.fx_te == 1.0);
                CHECK(Xe.fx_tm == 1.0);
            }

            // The host leg the crossing closes out, on its own. Asserted first, so that a red
            // fixture is attributable: these two are independent of the composite by construction
            // and a failure here is a statement about the traversal reference, not about the
            // incidence cosine.
            CHECK(d_hT < 1.0e-12);
            CHECK(d_hR < 1.0e-12);

            // The two references are far enough apart that the composite assertions below cannot
            // pass at both cosines. This is a statement about the references, not about the code,
            // so it holds whichever one the code is built at and guards against a revert.
            if (thd >= 20.0)
            {
                CHECK(std::abs(Pe.T_te - Xe.T_te) > 1.0e-6);
                CHECK(std::abs(Pe.R_tm - Xe.R_tm) > 1.0e-6);
            }

            // The composite at the exact in-medium cosine, both ports, both modes.
            CHECK(d_Ts < 1.0e-9);
            CHECK(d_Tg < 1.0e-9);
            CHECK(d_Rs < 1.0e-9);
            CHECK(d_Rg < 1.0e-9);
        }
}

TEST_CASE("Exit boundary - the port at the recovered angle")
{
    // The far face of a traced solid, measured against a hand-built reference at the exact
    // recovered cosine rather than at the real projection the tracer was left holding.
    //
    // The whole exit leg is asserted, not just the port: ray_state_update writes
    // T_exit * S * M, with S the series factor and M the in-medium traversal, and all three are
    // reproduced here from closed forms that share no code with the library. The port alone is
    // printed as out / (S M) so the table shows the quantity the fix moves.
    //
    // Nothing here divides one run by another. REPLACE_BY_VBS REBUILDS the field from the
    // interface coefficient rather than scaling what it was handed, so the written value is
    // absolute and the probe cancels out of the measurement by construction.
    //
    // Two modes. The scalar run pins the complex coefficient outright. The EM run pins the gain,
    // which is the Frobenius norm of the Jones matrix and is therefore invariant under the
    // polarization basis - so it measures the TE and TM ports without depending on which matrix
    // element each one lands in for this geometry.
    const double f_hz = FRQ;
    const double L = 0.10; // normal thickness of the traversed layer
    auto mtl = diag_palette<double>();

    QD_DIAG("\n# diag: exit port at the recovered complex cosine, i-o exit, normal thickness 100 mm\n");
    QD_DIAG("# port_ref is the exit interface at the exact angle; port_traced is the same interface\n");
    QD_DIAG("# at the real projection, which is what the exit used to be evaluated at\n");
    QD_DIAG("mtl,theta_ext,ct_traced,|port_meas|,|port_ref|,|port_traced|,d_port_dB,"
            "d_amp,d_gain_em,rtype\n");

    for (int m : {5, 1, 4, 6})
    {
        QD_DIAG("# material %d, eps %.2f %+.2fj\n", m,
                std::real(diag_eps(m, f_hz)), std::imag(diag_eps(m, f_hz)));
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
        {
            std::complex<double> es = diag_eps(m, f_hz), ea(1.0, 0.0);
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(es, th);
            std::complex<double> cs = diag_ct_exact(es, th);
            double dr = L / ct; // refracted path the tracer walked

            // Every angle in this sweep bends the direction, so the recovery is armed on all of
            // them. Normal incidence is excluded deliberately: there is nothing projected there and
            // the exit keeps the geometric path, which the entry-boundary case covers instead.
            REQUIRE(std::abs(ct - std::cos(th)) > 1e-6);

            std::complex<double> Ts_te, Ts_tm, Te_te, Te_tm;
            diag_port(es, ea, cs, true, Ts_te, Ts_tm);  // scalar port at the exact cosine
            diag_port(es, ea, cs, false, Te_te, Te_tm); // EM port, per channel

            std::complex<double> Ss_te, Ss_tm, Se_te, Se_tm;
            diag_series(es, ea, ea, cs, L, f_hz, true, Ss_te, Ss_tm);
            diag_series(es, ea, ea, cs, L, f_hz, false, Se_te, Se_tm);

            std::complex<double> M = diag_medium_vbs(es, dr, ct, L, cs, f_hz, std::cos(th));

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_EXIT, 1u,
                                       /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                       L, 0.5, 0.5, /*fbs angle = geometric*/ QPI / 2.0 - th,
                                       /*always resolve*/ 0.0);
                // The refracted direction the tracer left behind
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };

            auto Cs = run(4); // scalar transmission
            auto Ce = run(1); // EM transmission

            std::complex<double> out = cpx(Cs.xprmat, 0, 0);
            std::complex<double> ref = Ts_te * Ss_te * M;
            std::complex<double> carrier = Ss_te * M;
            std::complex<double> port_meas = (std::abs(carrier) > 0.0)
                                                 ? out / carrier
                                                 : std::complex<double>(0.0, 0.0);

            // The same port at the real projection, which is what the code applied before.
            std::complex<double> Tt_te, Tt_tm;
            diag_port(es, ea, std::complex<double>(ct, 0.0), true, Tt_te, Tt_tm);

            double g_ref = 0.5 * (std::norm(Te_te * Se_te * M) + std::norm(Te_tm * Se_tm * M));

            auto dB = [](double x, double y)
            { return 20.0 * std::log10(std::max(1e-300, x) / std::max(1e-300, y)); };

            QD_DIAG("%d,%.1f,%.9f,%.9f,%.9f,%.9f,%.6f,%.3e,%.3e,%u\n",
                    m, thd, ct, std::abs(port_meas), std::abs(Ts_te), std::abs(Tt_te),
                    dB(std::abs(Tt_te), std::abs(Ts_te)),
                    std::abs(out - ref), std::abs(Ce.gain(0) - g_ref),
                    (unsigned)Cs.resolved_type[0]);

            // The exit leg is the port at the recovered angle, the series and the traversal, and
            // nothing else.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(out - ref) < 1e-9);

            // Same statement per channel, read through a basis-invariant quantity so the EM run
            // pins the TM port as well.
            CHECK(std::abs(Ce.gain(0) - g_ref) < 1e-9);

            // Sanity: the port actually moved, so the case can fail. The recovery is a provable
            // no-op for a real permittivity, which is why the lossless control is exempt.
            if (m != 6 && thd >= 40.0)
                CHECK(std::abs(Tt_te - Ts_te) > 1e-6);
        }
    }
}

TEST_CASE("Traced solid - air | slab | air chain against the analytic slab")
{
    // The end-to-end statement, and the case that would have caught the exit-angle defect.
    //
    // The chain is assembled from the ports the two public functions actually write, with nothing
    // reconstructed in between: ray_mesh_interact supplies the entry coefficient and the refracted
    // direction, and ray_state_update supplies the exit leg - exit port, internal series and
    // in-medium traversal - from that direction. Their product is compared against the analytic
    // slab, which shares no code with either.
    //
    // The tracer is undeviated (type 4), which is the configuration the whole VBS machinery exists
    // for: the geometric ray continues straight, so the far face is met at the external angle,
    // while path_dirN carries the Snell direction and the two disagree. A refracting tracer
    // (type 5) bends the geometry itself, the two angles coincide, and there is nothing to correct.
    //
    // Why the entry-boundary case cannot see what this one sees: it divides the exit factor out by
    // construction, so an exit evaluated at the wrong cosine cancels against itself. Here it does
    // not cancel against anything.
    const double f_hz = FRQ;
    const double L = 0.10; // normal thickness of the slab
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);

    QD_DIAG("\n# diag: traced chain, entry port from ray_mesh_interact, exit leg from ray_state_update\n");
    QD_DIAG("# air | slab | air, normal thickness 100 mm, 1 GHz; oracle is the analytic slab, TE channel\n");
    QD_DIAG("mtl,theta_ext,ct_mesh,ct_map,d_ct,chain_dB,ana_dB,d_dB,rtype\n");

    for (int m : {5, 1, 4, 6})
    {
        QD_DIAG("# material %d, eps %.2f %+.2fj\n", m,
                std::real(diag_eps(m, f_hz)), std::imag(diag_eps(m, f_hz)));
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
        {
            std::complex<double> es = diag_eps(m, f_hz);
            double th = thd * QPI / 180.0;

            // Entry: one face, air -> slab, undeviated transmission.
            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));

            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL,
                                                 &fbs_ind, &sbs_ind);
            REQUIRE(fbs_ind.n_elem == 1);
            REQUIRE(fbs_ind(0) != 0);

            arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                            &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                            &path_dirN, true, (arma::u32_vec *)nullptr);

            REQUIRE(path_dirN.n_rows == 1);
            std::complex<double> a_entry(xprmatN(0, 0), xprmatN(1, 0));

            double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
            double ct_mesh = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
            double ct_map = diag_traced_ct(es, th); // the same map, evaluated by hand

            // Exit: the far face of the layer, reached along the direction the mesh call wrote.
            // fbs_angle stays the external angle - the geometric ray was never bent - and the
            // normal thickness enters as the origin-to-face distance.
            auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                   /*M1*/ m, /*M2*/ 0, /*prev*/ 0, /*cur*/ m, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - th, /*always resolve*/ 0.0);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = path_dirN(0, 0);
            C.path_dir_prev(0, 1) = path_dirN(0, 1);
            C.path_dir_prev(0, 2) = path_dirN(0, 2);
            C.run();

            // Absolute, because the exit rebuilds the field rather than scaling it.
            std::complex<double> a_exit = cpx(C.xprmat, 0, 0);
            std::complex<double> chain = a_entry * a_exit;

            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);

            double chain_dB = 10.0 * std::log10(std::max(1e-300, std::norm(chain)));
            double ana_dB = 10.0 * std::log10(std::max(1e-300, p_te));

            QD_DIAG("%d,%.1f,%.9f,%.9f,%.3e,%.6f,%.6f,%.6f,%u\n",
                    m, thd, ct_mesh, ct_map, std::abs(ct_mesh - ct_map),
                    chain_dB, ana_dB, chain_dB - ana_dB, (unsigned)C.resolved_type[0]);

            // The direction the tracer wrote is the map the recovery inverts. A gap here means the
            // two have drifted apart and every reference below is measuring the wrong thing.
            INFO("material " << m << ", theta_ext " << thd);
            CHECK(std::abs(ct_mesh - ct_map) < 1e-9);

            // Entry port, traversal, exit port and series, telescoping onto the analytic slab.
            CHECK(std::abs(chain_dB - ana_dB) < 1e-6);

            // The oracle is the average of the two channels; the scalar chain measures TE, so the
            // average is printed only where the two coincide. Kept as a guard that the per-channel
            // split of the reference has not been swapped.
            CHECK(std::abs(diag_slab_p_avg(es, L, th, f_hz) - 0.5 * (p_te + p_tm)) < 1e-15);
        }
    }
}

TEST_CASE("Permeability - the traced solid and a magnetic layer in a stack")
{
    // The sheet, glued and single-layer stack paths are covered elsewhere and the library was found
    // correct on them without a source change. Two statements were left open by that work and are
    // made here: a traced solid with mu != 1, and a magnetic layer inside an ordered stack.
    //
    // The analytic-slab oracle could not express either. diag_slab_p_pol formed both the admittance
    // and the index as sqrt(eps), so a magnetic slab had no reference at all; it now takes the same
    // three substitutions the sheet builders took, z = sqrt(eps/mu), the Snell ratio as a ratio of
    // eps*mu products, and n = sqrt(eps*mu). diag_traced_ct and diag_ct_exact take the third of
    // those, since the map qd_refract applies is built from the product.
    //
    // What is distinct about a magnetic traced solid, over and above a magnetic sheet:
    //
    //   The telescoping property does not read mu. The entry port carries
    //   sqrt(Re(z_s c_s)/Re(z_a c_a)) and the exit port its reciprocal, and the pair cancels as a
    //   ratio of one and the same real number whatever z_s is. So mu cannot break it; what a
    //   magnetic run checks is that both crossings still read the SAME z_s and the same c_s, which
    //   is the property the recovered cosine exists to provide.
    //
    //   The swap leaves the geometry bit-identical. qd_refract's snell, its refraction root, the
    //   total-reflection test, EXACT_IN_SLAB_COSINE's eps_r, medium_gain's loss rate and
    //   SCALE_BY_MEDIUM's index are all eps*mu products, so an eps<->mu swap moves nothing about
    //   the traced path, the recovered cosine or the traversal - only the two interface channels,
    //   which it exchanges. That is a stronger statement than the sheet version of it, because it
    //   also asserts that none of the geometry reads the admittance.
    const double f_hz = FRQ;
    const double L = 0.10;   // normal thickness of the traced solid
    const double thk = 0.006; // sheet thickness in the stack block
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    struct MuRow
    {
        double a, c, e, g, thk;
    };
    const int MSOL = 1, MSWP = 2, MMAT = 3, MLAY = 4, DLAY = 5;
    const std::vector<MuRow> pal = {
        {1.50, 0.40 / 17.98, 1.20, 0.0, 0.0},                  // 1 MSOL, magnetic lossy solid
        {1.20, 0.0, 1.50, 0.40 / 17.98, 0.0},                  // 2 MSWP, its eps<->mu swap
        {2.00, 0.0, 2.00, 0.0, 0.0},                           // 3 MMAT, matched: z = 1, n = 2
        {5.24, 0.63 / 17.98, 2.10, 0.35 / 17.98, 0.006},       // 4 MLAY, magnetic sheet layer
        {1.50, 0.40 / 17.98, 1.00, 0.0, 0.006},                // 5 DLAY, dielectric sheet layer
    };
    const std::complex<double> e_sol(1.50, -0.40), mu_sol(1.20, 0.0);
    const std::complex<double> e_swp(1.20, 0.0), mu_swp(1.50, -0.40);
    const std::complex<double> e_mat(2.0, 0.0), mu_mat(2.0, 0.0);
    const std::complex<double> e_lay(5.24, -0.63), mu_lay(2.10, -0.35);
    const std::complex<double> e_dly(1.50, -0.40);

    std::unordered_map<std::string, std::vector<double>> mtl;
    for (const char *k : {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB",
                          "alpha", "alphaB", "fRef"})
        mtl[k] = std::vector<double>(pal.size(), 0.0);
    for (size_t i = 0; i < pal.size(); ++i)
    {
        mtl["a"][i] = pal[i].a, mtl["c"][i] = pal[i].c;
        mtl["e"][i] = pal[i].e, mtl["g"][i] = pal[i].g;
        mtl["thk"][i] = pal[i].thk, mtl["fRef"][i] = 1.0;
    }

    // ------------------------------------------------------------------------------------------
    // A traced air | slab | air chain through a magnetic solid, assembled from the ports the two
    // public functions write, exactly as the dielectric chain case assembles it: the entry
    // coefficient and the refracted direction from ray_mesh_interact, the exit leg from
    // ray_state_update, nothing reconstructed in between.
    //
    // The sweep stops at 75 deg. The internal critical angle of eps*mu = 1.80 - 0.48j is 47.6 deg
    // and the traced internal angle reaches 45.5 deg at 75 deg of external incidence, so the whole
    // sweep stays below it with about two degrees to spare and no row turns into a forced mirror.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability, traced air | slab | air chain, normal thickness 100 mm\n");
    QD_DIAG("# MSOL is eps 1.50 - 0.40j with mu 1.20; MSWP is the eps<->mu swap of it, so the two\n");
    QD_DIAG("# share every cosine and every phase and differ only in the admittance.\n");
    QD_DIAG("mtl,theta_ext,ct_mesh,ct_map,d_ct,chain_dB,ana_dB,d_dB,rtype\n");

    struct Solid
    {
        const char *name;
        int idx;
        std::complex<double> e, mu;
    };
    const std::vector<Solid> solids = {
        {"MSOL", MSOL, e_sol, mu_sol},
        {"MSWP", MSWP, e_swp, mu_swp},
        {"MMAT", MMAT, e_mat, mu_mat},
    };

    // The sweep starts at 10 deg, as the dielectric chain case does. At exactly normal incidence
    // the traced and geometric cosines coincide, ray_state_update drops use_vbs on its own 1e-6
    // re-test, and the traversal falls back to the hybrid path form, which sits 2.5e-5 dB from the
    // exact one and would not clear the bound below. That gap is its own open item and is nothing
    // to do with permeability, so it is kept out of the measurement rather than absorbed into it.
    for (const auto &S : solids)
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0;

            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)S.idx));

            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL,
                                                 &fbs_ind, &sbs_ind);
            REQUIRE(fbs_ind.n_elem == 1);
            REQUIRE(fbs_ind(0) != 0);

            arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                            &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                            &path_dirN, true, (arma::u32_vec *)nullptr);

            REQUIRE(path_dirN.n_rows == 1);
            std::complex<double> a_entry(xprmatN(0, 0), xprmatN(1, 0));

            double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
            double ct_mesh = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
            double ct_map = diag_traced_ct(S.e, th, S.mu);

            auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                   /*M1*/ S.idx, /*M2*/ 0, /*prev*/ 0, /*cur*/ S.idx, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - th, /*always resolve*/ 0.0);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = path_dirN(0, 0);
            C.path_dir_prev(0, 1) = path_dirN(0, 1);
            C.path_dir_prev(0, 2) = path_dirN(0, 2);
            C.run();

            std::complex<double> chain = a_entry * cpx(C.xprmat, 0, 0);

            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(S.e, L, th, f_hz, p_te, p_tm, S.mu);

            double chain_dB = 10.0 * std::log10(std::max(1e-300, std::norm(chain)));
            double ana_dB = 10.0 * std::log10(std::max(1e-300, p_te));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.3e,%.6f,%.6f,%.6f,%u\n",
                    S.name, thd, ct_mesh, ct_map, std::abs(ct_mesh - ct_map),
                    chain_dB, ana_dB, chain_dB - ana_dB, (unsigned)C.resolved_type[0]);

            INFO(S.name << ", theta_ext " << thd);

            // The direction the tracer wrote is the map the recovery inverts, with the product
            // eps*mu in both. A gap here means the map and the reference have drifted apart.
            CHECK(std::abs(ct_mesh - ct_map) < 1.0e-9);

            // Entry port, traversal, exit port and series, telescoping onto the analytic slab.
            CHECK(std::abs(chain_dB - ana_dB) < 1.0e-6);
            CHECK(std::abs(diag_slab_p_avg(S.e, L, th, f_hz, S.mu) - 0.5 * (p_te + p_tm)) < 1.0e-15);

            // The matched solid, eps = mu = 2, has unit admittance, so neither face reflects at
            // normal incidence and the lossless slab is fully transparent there. At any angle its
            // two channels stay equal, because equal admittances collapse the TE and TM Fresnel
            // expressions onto the same function of the two cosines. No permittivity reaches
            // either statement.
            if (S.idx == MMAT)
                CHECK(std::abs(p_te - p_tm) < 1.0e-14);
        }

    // The matched solid at normal incidence, on the reference alone, where the chain cannot be used
    // for the reason above. With z = 1 on both sides neither face reflects and a lossless slab is
    // fully transparent whatever its index; the transmitted field is the propagation across it and
    // nothing else. No permittivity reaches this, because one that moves n also moves z.
    {
        double p_te = 0.0, p_tm = 0.0;
        diag_slab_p_pol(e_mat, L, 0.0, f_hz, p_te, p_tm, mu_mat);
        INFO("matched solid at normal incidence");
        CHECK(std::abs(p_te - 1.0) < 1.0e-12);
        CHECK(std::abs(p_tm - 1.0) < 1.0e-12);
    }

    // The swap, read off the two sweeps above. Scalar mode carries the TE channel only, so the
    // exchange shows up as the swapped run's TE equalling the original run's TM. The traced cosine
    // is asserted identical between the two, which is the geometry half of the statement.
    QD_DIAG("\n# diag: permeability, eps<->mu swap on a traced solid\n");
    QD_DIAG("theta_ext,d_ct,p_te_swp,p_tm_sol,d_exchange,separation_dB\n");

    for (double thd : {0.0, 10.0, 20.0, 40.0, 60.0, 75.0})
    {
        double th = thd * QPI / 180.0;
        double a_te = 0.0, a_tm = 0.0, b_te = 0.0, b_tm = 0.0;
        diag_slab_p_pol(e_sol, L, th, f_hz, a_te, a_tm, mu_sol);
        diag_slab_p_pol(e_swp, L, th, f_hz, b_te, b_tm, mu_swp);

        double ct_a = diag_traced_ct(e_sol, th, mu_sol), ct_b = diag_traced_ct(e_swp, th, mu_swp);
        double sep = 10.0 * std::log10(a_te / a_tm);

        QD_DIAG("%.1f,%.3e,%.9e,%.9e,%.3e,%+.4f\n", thd, std::abs(ct_a - ct_b),
                b_te, a_tm, std::abs(b_te - a_tm), sep);

        INFO("swap, theta_ext " << thd);
        CHECK(std::abs(ct_a - ct_b) < 1.0e-12); // the traced path does not move
        CHECK(std::abs(b_te - a_tm) < 1.0e-14); // and the two channels exchange
        CHECK(std::abs(b_tm - a_te) < 1.0e-14);

        // Past normal incidence the two channels are far enough apart that the exchange is a
        // statement rather than an identity between two equal numbers.
        if (thd >= 40.0)
            CHECK(std::abs(sep) > 1.0e-2);
    }

    // ------------------------------------------------------------------------------------------
    // A magnetic layer inside an ordered two-sheet stack, air | 6 mm | 6 mm | air. diag_cm_stack
    // already carries (eps, mu) per layer, so this is a fixture row rather than a reference change.
    // What it adds over a magnetic single sheet is the layer product: the matrix multiplies two
    // layers whose admittances and indices are now independent of each other, and the order
    // statement has to survive that.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability, ordered stack with one magnetic layer\n");
    QD_DIAG("# air | MLAY (eps 5.24 - 0.63j, mu 2.10 - 0.35j) | DLAY (eps 1.50 - 0.40j) | air\n");
    QD_DIAG("order,theta_n,ref_|T|,ref_|R|,out_|T|,out_|R|,dT,dR,rtype_T,rtype_R\n");

    const uint8_t OT_STACK_FBS = 71; // FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // SBS layer traversed first

    struct MuOrd
    {
        const char *name;
        uint8_t otype;
        std::complex<double> e1, mu1, e2, mu2;
    };
    const std::vector<MuOrd> orders = {
        {"M|D", OT_STACK_FBS, e_lay, mu_lay, e_dly, U},
        {"D|M", OT_STACK_SBS, e_dly, U, e_lay, mu_lay},
    };

    for (const auto &O : orders)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);

            DiagSheet Xs = diag_cm_stack(e_air, U, {{O.e1, O.mu1, thk}, {O.e2, O.mu2, thk}},
                                         e_air, U, ci, f_hz, false, /*scalar*/ true);
            DiagSheet Xe = diag_cm_stack(e_air, U, {{O.e1, O.mu1, thk}, {O.e2, O.mu2, thk}},
                                         e_air, U, ci, f_hz, false, /*EM*/ false);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, O.otype, 2u,
                                       /*M1*/ MLAY, /*M2*/ DLAY,
                                       /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };

            auto Ct = run(4), Ce = run(1), Cr = run(3), Cx = run(0);

            double gT = 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
            double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));
            double dT = std::abs(cpx(Ct.xprmat, 0, 0) - Xs.T_te);
            double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u,%u\n",
                    O.name, thd, std::abs(Xs.T_te), std::abs(Xs.R_te),
                    std::abs(cpx(Ct.xprmat, 0, 0)), std::abs(cpx(Cr.xprmat, 0, 0)),
                    dT, dR, (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO("magnetic stack " << O.name << ", theta_n " << thd);
            CHECK(gT > 1.0e-3);
            CHECK(gR > 1.0e-3);
            CHECK(Ct.resolved_type[0] != 0);
            CHECK(Cr.resolved_type[0] != 0);
            CHECK(dT < 1.0e-9);
            CHECK(dR < 1.0e-9);
            CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
            CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);
        }

    // Reciprocity survives a magnetic layer: the transmit port of the stack does not depend on
    // which layer comes first and the reflect port does, with the layers now differing in their
    // admittance as well as their index.
    for (double thd : {0.0, 45.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet MD = diag_cm_stack(e_air, U, {{e_lay, mu_lay, thk}, {e_dly, U, thk}},
                                     e_air, U, ci, f_hz, false, false);
        DiagSheet DM = diag_cm_stack(e_air, U, {{e_dly, U, thk}, {e_lay, mu_lay, thk}},
                                     e_air, U, ci, f_hz, false, false);
        INFO("magnetic stack order, theta_n " << thd);
        CHECK(std::abs(MD.T_te - DM.T_te) < 1.0e-14);
        CHECK(std::abs(MD.T_tm - DM.T_tm) < 1.0e-14);
        CHECK(std::abs(MD.R_te - DM.R_te) > 1.0e-3);
    }
}
// ================================================================================================
// Layered stacks. Issue 1 of the fix plan: there is no transfer-matrix core, so a stack of more
// than one layer is either approximated or dropped. Three symptoms, measured separately below.
//
// The reference for all three is diag_cm_stack, the hand-built characteristic matrix defined
// above. The first case pins that reference against the two single-layer builders the plan names
// as its regression oracles; the second and third then use it on stacks the solver cannot express.
// ================================================================================================

TEST_CASE("Layered stack - the characteristic-matrix reference against the single-layer oracles")
{
    // diag_cm_stack has to reproduce diag_sheet_composite and diag_sheet_composite_cx exactly at
    // one layer, since those are what the shipped composite is already asserted against. Nothing
    // here touches the library: it is a check that two independently written closed forms agree,
    // so that a failure in the two cases below is attributable to the solver rather than to the
    // reference.
    //
    // Four further properties of the matrix product are asserted, each of which would catch a
    // different transcription error:
    //   a layer of zero thickness is inert, which pins the identity element,
    //   splitting one layer into two halves is inert, which pins the product and the phase,
    //   the transmit port is invariant under layer order, which is reciprocity (det M = 1),
    //   the reflect port is not, which is the whole reason sheet_case 4 carries an order at all.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0);

    const std::complex<double> e_air(1.0, 0.0);
    const std::complex<double> e_s = diag_eps(3, f_hz);  // sheet, eps 5.24 - 0.63j
    const std::complex<double> e_lo = diag_eps(5, f_hz); // LO, eps 1.50 - 0.40j
    const std::complex<double> e_d = diag_eps(4, f_hz);  // DENSER, eps 9.0 - 1.20j

    struct Stack
    {
        const char *name;
        std::complex<double> e_near, e_far;
        bool disp_far;
    };
    const std::vector<Stack> stacks = {
        {"air | sheet | air", e_air, e_air, false},
        {"air | sheet | LO ", e_air, e_lo, false},
        {"LO  | sheet | air", e_lo, e_air, true},
        {"LO  | sheet | LO ", e_lo, e_lo, false},
        {"DEN | sheet | LO ", e_d, e_lo, false},
    };

    QD_DIAG("\n# diag: characteristic-matrix reference against the single-layer builders\n");
    QD_DIAG("stack,mode,theta_n,d_real_builder,d_complex_builder\n");

    double worst_real = 0.0, worst_cx = 0.0;
    for (const auto &S : stacks)
        for (int mode = 0; mode < 2; ++mode) // 0 = EM, 1 = scalar
            for (double thd : {0.0, 10.0, 30.0, 50.0, 70.0, 85.0, 89.9})
            {
                bool scalar = (mode == 1);
                double th = thd * QPI / 180.0, ct = std::cos(th);

                DiagSheet A = diag_sheet_composite(S.e_near, e_s, S.e_far, thk, ct, f_hz,
                                                   S.disp_far, scalar);
                DiagSheet B = diag_cm_stack(S.e_near, U, {{e_s, U, thk}}, S.e_far, U,
                                            std::complex<double>(ct, 0.0), f_hz, S.disp_far, scalar);

                double d_re = std::max(std::max(std::abs(A.T_te - B.T_te), std::abs(A.T_tm - B.T_tm)),
                                       std::max(std::abs(A.R_te - B.R_te), std::abs(A.R_tm - B.R_tm)));

                // The same statement at a complex incidence cosine, which is the argument the core
                // will actually be handed wherever the near medium is a traced solid.
                std::complex<double> cx = diag_ct_exact(e_lo, th);
                DiagSheet Ac = diag_sheet_composite_cx(e_lo, e_s, S.e_far, thk, cx, f_hz,
                                                       S.disp_far, scalar);
                DiagSheet Bc = diag_cm_stack(e_lo, U, {{e_s, U, thk}}, S.e_far, U, cx, f_hz,
                                             S.disp_far, scalar);
                double d_cx = std::max(std::max(std::abs(Ac.T_te - Bc.T_te), std::abs(Ac.T_tm - Bc.T_tm)),
                                       std::max(std::abs(Ac.R_te - Bc.R_te), std::abs(Ac.R_tm - Bc.R_tm)));

                worst_real = std::max(worst_real, d_re);
                worst_cx = std::max(worst_cx, d_cx);

                QD_DIAG("%s,%s,%.1f,%.3e,%.3e\n", S.name, scalar ? "scalar" : "EM    ", thd, d_re, d_cx);

                INFO(S.name << ", " << (scalar ? "scalar" : "EM") << ", theta_n " << thd);
                CHECK(d_re < 1.0e-14);
                CHECK(d_cx < 1.0e-14);
            }

    QD_DIAG("# worst against the real-cosine builder %.3e, against the complex one %.3e\n",
            worst_real, worst_cx);

    // Structural properties of the product, on a stack the two cases below both use.
    for (double thd : {0.0, 40.0, 80.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);

        DiagSheet one = diag_cm_stack(e_air, U, {{e_s, U, thk}}, e_air, U, ci, f_hz, false, false);
        DiagSheet zero = diag_cm_stack(e_air, U, {{e_s, U, thk}, {e_d, U, 0.0}}, e_air, U, ci, f_hz,
                                       false, false);
        DiagSheet half = diag_cm_stack(e_air, U, {{e_s, U, 0.5 * thk}, {e_s, U, 0.5 * thk}},
                                       e_air, U, ci, f_hz, false, false);

        DiagSheet AB = diag_cm_stack(e_air, U, {{e_s, U, thk}, {e_lo, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        DiagSheet BA = diag_cm_stack(e_air, U, {{e_lo, U, thk}, {e_s, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);

        INFO("theta_n " << thd);

        // A layer of zero thickness contributes the identity.
        CHECK(std::abs(one.T_te - zero.T_te) < 1.0e-14);
        CHECK(std::abs(one.R_tm - zero.R_tm) < 1.0e-14);

        // One layer and two halves of it are the same layer.
        CHECK(std::abs(one.T_te - half.T_te) < 1.0e-14);
        CHECK(std::abs(one.R_tm - half.R_tm) < 1.0e-14);

        // Reciprocity: the transmit port of a stack does not depend on which layer comes first.
        CHECK(std::abs(AB.T_te - BA.T_te) < 1.0e-14);
        CHECK(std::abs(AB.T_tm - BA.T_tm) < 1.0e-14);

        // The reflect port does. This is the only thing the ordering machinery of sheet_case 4
        // exists to serve, and it is asserted so that a reference that lost the order shows up here
        // rather than as a silent pass in the third case below.
        CHECK(std::abs(AB.R_te - BA.R_te) > 1.0e-3);
    }
}

TEST_CASE("Permeability - the sheet, stack and glued paths")
{
    // The e, f, g, h columns are live in every expression the sheet and stack paths form, and no
    // sheet, stack, glued pair or traced chain has ever run with a non-unit mu. This case supplies
    // the coverage. No source change accompanies it.
    //
    // Where mu enters. Every site is one of two forms and never a third. Through the PRODUCT
    // eps*mu: the index n = sqrt(eps*mu) in every propagation factor, every Snell ratio and every
    // refraction root, the total-reflection test, medium_gain's loss rate, qd_refract's snell,
    // EXACT_IN_SLAB_COSINE's eps_r, the displaced-host cosine and index, and SCALE_BY_MEDIUM, which
    // never sees mu on its own at all. Through the QUOTIENT eps/mu: the admittance z in every
    // Fresnel amplitude, the core's tilted admittances z*c and z/c, and flux_amp's Re(z c) and
    // Re(conj(z) c).
    //
    // So there is exactly one magnetic degree of freedom to test: n and z stop being the same
    // number. A single permittivity fixes both, which is why no dielectric palette can reach any of
    // this, and it is why an eps<->mu swap is the sharpest available probe - it holds n, and
    // therefore every cosine and every phase, fixed while inverting z.
    //
    // What the six existing permeability cases cover is the product-and-quotient split at a single
    // interface and in medium_gain. None of them involves a layer, so none of them can see a
    // propagation phase built at the wrong index or an internal series built at the wrong
    // admittance.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    // A local palette, so diag_palette and diag_eps stay dielectric and every existing residual
    // stays where it is. Same column convention: eps = a - j*17.98*c/fGHz, mu = e - j*17.98*g/fGHz.
    struct MuRow
    {
        double a, c, e, g, thk;
    };
    const std::complex<double> e_air(1.0, 0.0), U(1.0, 0.0);
    const std::complex<double> e_ms(5.24, -0.63), mu_ms(2.10, -0.35); // magnetic sheet
    const std::complex<double> e_mh(1.50, -0.40), mu_mh(1.60, 0.0);   // magnetic host
    const std::complex<double> e_k(4.0, 0.0), mu_k(4.0, 0.0);         // matched dense sheet

    const int MSHEET = 1, MSWAP = 2, MHOST = 3, MATCHED = 4;
    const std::vector<MuRow> pal = {
        {5.24, 0.63 / 17.98, 2.10, 0.35 / 17.98, thk}, // 1 MSHEET
        {2.10, 0.35 / 17.98, 5.24, 0.63 / 17.98, thk}, // 2 MSWAP, the eps<->mu swap of MSHEET
        {1.50, 0.40 / 17.98, 1.60, 0.0, 0.0},          // 3 MHOST, a magnetic solid
        {4.00, 0.0, 4.00, 0.0, thk},                   // 4 MATCHED, eps = mu so z = 1, n = 4
    };

    std::unordered_map<std::string, std::vector<double>> mtl;
    for (const char *k : {"a", "b", "c", "d", "e", "f", "g", "h", "thk", "att", "attB",
                          "alpha", "alphaB", "fRef"})
        mtl[k] = std::vector<double>(pal.size(), 0.0);
    for (size_t i = 0; i < pal.size(); ++i)
    {
        mtl["a"][i] = pal[i].a, mtl["c"][i] = pal[i].c;
        mtl["e"][i] = pal[i].e, mtl["g"][i] = pal[i].g;
        mtl["thk"][i] = pal[i].thk, mtl["fRef"][i] = 1.0;
    }

    const uint8_t OT_SHEET_ISO = 67;   // ok | sheet, no co-located face
    const uint8_t OT_GLUED_ENTRY = 71; // ok | front | co-located | sheet -> sheet_case 2

    // Channel identification, as in the flux-conversion case: make1 puts orig, fbs and dest on the
    // x axis with the normal along x, so the incidence plane is degenerate, qd_polbasis takes its
    // identity branch, and VV is TM while HH is TE.
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ------------------------------------------------------------------------------------------
    // The generalized builders against diag_cm_stack, at mu != 1. Nothing here touches the
    // library. diag_cm_stack was written with (eps, mu) pairs from the start and shares no code
    // with the two single-layer builders, so this is what establishes that the three substitutions
    // above are the right three before any of them is used as an oracle below.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability, generalized single-layer builders against diag_cm_stack\n");
    QD_DIAG("stack,mode,theta_n,d_real_builder,d_complex_builder\n");

    struct MuStack
    {
        const char *name;
        std::complex<double> e_near, mu_near, e_far, mu_far;
        bool disp_far;
    };
    const std::vector<MuStack> stacks = {
        {"air  | Msheet | air ", e_air, U, e_air, U, false},
        {"air  | Msheet | Mhst", e_air, U, e_mh, mu_mh, false},
        {"Mhst | Msheet | air ", e_mh, mu_mh, e_air, U, true},
        {"Mhst | Msheet | Mhst", e_mh, mu_mh, e_mh, mu_mh, false},
    };

    double worst_re = 0.0, worst_cx = 0.0;
    for (const auto &S : stacks)
        for (int mode = 0; mode < 2; ++mode)
            for (double thd : {0.0, 10.0, 30.0, 50.0, 70.0, 85.0, 89.9})
            {
                bool scalar = (mode == 1);
                double th = thd * QPI / 180.0, ct = std::cos(th);

                DiagSheet A = diag_sheet_composite(S.e_near, e_ms, S.e_far, thk, ct, f_hz,
                                                   S.disp_far, scalar, S.mu_near, mu_ms, S.mu_far);
                DiagSheet B = diag_cm_stack(S.e_near, S.mu_near, {{e_ms, mu_ms, thk}},
                                            S.e_far, S.mu_far, std::complex<double>(ct, 0.0),
                                            f_hz, S.disp_far, scalar);
                double d_re = std::max(std::max(std::abs(A.T_te - B.T_te), std::abs(A.T_tm - B.T_tm)),
                                       std::max(std::abs(A.R_te - B.R_te), std::abs(A.R_tm - B.R_tm)));

                // The same at a complex incidence cosine, which is what a sheet reached from inside
                // a magnetic solid is handed. The cosine is built from the eps*mu product, so this
                // also pins that the transverse invariant carries mu.
                std::complex<double> cx =
                    std::sqrt(1.0 - ((e_air * U) / (e_mh * mu_mh)) * (1.0 - ct * ct));
                DiagSheet Ac = diag_sheet_composite_cx(e_mh, e_ms, S.e_far, thk, cx, f_hz,
                                                       S.disp_far, scalar, mu_mh, mu_ms, S.mu_far);
                DiagSheet Bc = diag_cm_stack(e_mh, mu_mh, {{e_ms, mu_ms, thk}}, S.e_far, S.mu_far,
                                             cx, f_hz, S.disp_far, scalar);
                double d_cx = std::max(std::max(std::abs(Ac.T_te - Bc.T_te), std::abs(Ac.T_tm - Bc.T_tm)),
                                       std::max(std::abs(Ac.R_te - Bc.R_te), std::abs(Ac.R_tm - Bc.R_tm)));

                worst_re = std::max(worst_re, d_re), worst_cx = std::max(worst_cx, d_cx);
                QD_DIAG("%s,%s,%.1f,%.3e,%.3e\n", S.name, scalar ? "scalar" : "EM    ", thd, d_re, d_cx);

                INFO(S.name << ", " << (scalar ? "scalar" : "EM") << ", theta_n " << thd);
                CHECK(d_re < 1.0e-14);
                CHECK(d_cx < 1.0e-14);

                // And the complex-cosine builder still reproduces the real-cosine one at a real
                // cosine with mu live, which is the invariant the sheet-in-a-medium case rests on.
                DiagSheet Ar = diag_sheet_composite_cx(S.e_near, e_ms, S.e_far, thk,
                                                       std::complex<double>(ct, 0.0), f_hz,
                                                       S.disp_far, scalar, S.mu_near, mu_ms, S.mu_far);
                CHECK(std::abs(A.T_te - Ar.T_te) < 1.0e-14);
                CHECK(std::abs(A.R_tm - Ar.R_tm) < 1.0e-14);
            }

    QD_DIAG("# worst against the real-cosine builder %.3e, against the complex one %.3e\n",
            worst_re, worst_cx);

    // ------------------------------------------------------------------------------------------
    // An isolated magnetic sheet in air, and a glued entry into a magnetic solid, both against the
    // generalized reference. The isolated rows run with mtl_current = 0, so SHEET_EVENT applies no
    // host factor and the written field is the composite outright; those are asserted complex.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability through the library, sheet eps 5.24 - 0.63j, mu 2.10 - 0.35j\n");
    QD_DIAG("stack,theta_n,T_te_meas,T_te_ref,T_tm_meas,T_tm_ref,d_te,d_tm,rtype\n");

    struct LibRow
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2;
        int cur;
        std::complex<double> e_far, mu_far;
        bool disp_far;
    };
    const std::vector<LibRow> lrows = {
        {"isolated air | Msheet | air ", OT_SHEET_ISO, 1u, 0, 0, e_air, U, false},
        {"entry    air | Msheet | Mhst", OT_GLUED_ENTRY, 2u, MHOST, 0, e_mh, mu_mh, false},
    };

    for (const auto &R : lrows)
        for (double thd : {0.0, 20.0, 40.0, 60.0, 80.0, 85.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto C = make1<double>(mtl, /*EM transmission*/ 1, R.otype, R.nH,
                                   /*M1 = the sheet face*/ MSHEET, /*M2*/ R.m2,
                                   /*prev*/ 0, /*cur*/ R.cur, /*buf*/ 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n,
                                   /*eps: the host series must not fire*/ 1.5);
            C.run();

            DiagSheet D = diag_sheet_composite(e_air, e_ms, R.e_far, thk, ct, f_hz,
                                               R.disp_far, false, U, mu_ms, R.mu_far);

            std::complex<double> t_te = TE(C.xprmat), t_tm = TM(C.xprmat);
            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%u\n", R.name, thd,
                    std::abs(t_te), std::abs(D.T_te), std::abs(t_tm), std::abs(D.T_tm),
                    std::abs(t_te - D.T_te), std::abs(t_tm - D.T_tm), (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0);
            CHECK(std::abs(t_te - D.T_te) < 1.0e-9);
            CHECK(std::abs(t_tm - D.T_tm) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm))) < 1.0e-9);
        }

    // ------------------------------------------------------------------------------------------
    // The eps<->mu swap, on the sheet only, in air. Swapping holds n = sqrt(eps*mu) and therefore
    // every cosine and the propagation factor exactly fixed, and replaces z by 1/z. With both hosts
    // at unit admittance every reflection and every transmission product in the Airy pair maps onto
    // its opposite polarization, so the swap exchanges the TE and TM ports exactly, on both ports.
    //
    // The power-averaged gain is invariant under that exchange, so this statement is invisible to
    // gain and to scalar mode and can only be made per channel. That is the point: a solver that
    // reverted the admittance to sqrt(eps) would still report the same total.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability, eps<->mu swap on an isolated sheet in air\n");
    QD_DIAG("theta_n,d_te_vs_tm,d_tm_vs_te,d_gain,separation\n");

    for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
    {
        double theta_n = thd * QPI / 180.0;
        auto mk = [&](int idx)
        {
            auto C = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_ISO, 1u,
                                   idx, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();
            return C;
        };
        auto A = mk(MSHEET), B = mk(MSWAP);

        std::complex<double> a_te = TE(A.xprmat), a_tm = TM(A.xprmat);
        std::complex<double> b_te = TE(B.xprmat), b_tm = TM(B.xprmat);
        double d1 = std::abs(b_te - a_tm), d2 = std::abs(b_tm - a_te);
        double dg = std::abs((double)A.gain(0) - (double)B.gain(0));
        double sep = std::abs(a_te - a_tm);

        QD_DIAG("%.1f,%.3e,%.3e,%.3e,%.6f\n", thd, d1, d2, dg, sep);

        INFO("swap, theta_n " << thd);
        REQUIRE(A.resolved_type[0] != 0);
        REQUIRE(B.resolved_type[0] != 0);
        CHECK(d1 < 1.0e-9);
        CHECK(d2 < 1.0e-9);
        CHECK(dg < 1.0e-12);

        // Past normal incidence the two channels are far enough apart that the exchange above is a
        // statement rather than an identity between two equal numbers.
        if (thd > 0.0)
            CHECK(sep > 1.0e-3);
    }

    // ------------------------------------------------------------------------------------------
    // An impedance-matched dense sheet, eps = mu = 4, so z = 1 and n = 4. At normal incidence both
    // interfaces vanish and the sheet is invisible except for its phase: the composite is exactly
    // the propagation it carries over the host it displaces, exp(-j k0 thk (n - 1)), with unit
    // magnitude and no reflection at all. No dielectric can do that, because a single permittivity
    // that moves n also moves z.
    //
    // At oblique incidence it does reflect, and the reflection is polarization-independent, since
    // with z equal on both sides the TE and TM Fresnel expressions collapse onto the same function
    // of the two cosines. That is the second thing only a magnetic material produces.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: permeability, impedance-matched dense sheet, eps = mu = 4, thk 6 mm\n");
    QD_DIAG("theta_n,|T|,arg_T_deg,pred_deg,|R_te|,|R_tm|,d_pol\n");

    const double k0 = 2.0 * QPI * f_hz / C0;
    for (double thd : {0.0, 20.0, 45.0, 70.0})
    {
        double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

        auto Ct = make1<double>(mtl, /*EM transmission*/ 1, OT_SHEET_ISO, 1u,
                                MATCHED, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
        Ct.run();
        auto Cr = make1<double>(mtl, /*EM reflection*/ 0, OT_SHEET_ISO, 1u,
                                MATCHED, 0, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
        Cr.run();

        DiagSheet D = diag_sheet_composite(e_air, e_k, e_air, thk, ct, f_hz, false, false,
                                           U, mu_k, U);

        std::complex<double> t_te = TE(Ct.xprmat);
        double pred = -k0 * thk * (4.0 - 1.0) * 180.0 / QPI; // normal incidence only
        double arg_t = std::arg(t_te) * 180.0 / QPI;
        double r_te = std::abs(TE(Cr.xprmat)), r_tm = std::abs(TM(Cr.xprmat));

        QD_DIAG("%.1f,%.9f,%.6f,%.6f,%.9f,%.9f,%.2e\n", thd, std::abs(t_te), arg_t,
                (thd == 0.0) ? pred : 0.0, r_te, r_tm, std::abs(r_te - r_tm));

        INFO("matched sheet, theta_n " << thd);
        CHECK(std::abs(t_te - D.T_te) < 1.0e-9);
        CHECK(std::abs(TM(Ct.xprmat) - D.T_tm) < 1.0e-9);

        // The reflection carries no polarization dependence at any angle.
        CHECK(std::abs(std::abs(D.R_te) - std::abs(D.R_tm)) < 1.0e-14);
        CHECK(std::abs(r_te - r_tm) < 1.0e-9);

        if (thd == 0.0)
        {
            // Invisible except for the phase, exactly.
            CHECK(std::abs(D.R_te) < 1.0e-15);
            CHECK(std::abs(std::abs(t_te) - 1.0) < 1.0e-9);
            CHECK(std::abs(arg_t - pred) < 1.0e-3);
            CHECK((double)Cr.gain(0) < 1.0e-12);
        }
        else
        {
            // And it is a real reflection, so the normal-incidence statement is not the whole story.
            CHECK(r_te > 1.0e-3);
        }
    }
}

TEST_CASE("Layered stack - a host layer bounded by a sheet")
{
    // Symptom one and symptom two of issue 1, on the same geometry.
    //
    // An i-o exit out of a host layer whose previous-medium word holds a sheet. APPLY_TRANSITION
    // calls SLAB_AIRY_FACTOR(mtl_current, mtl_next, mtl_prev), so correction mode is handed the
    // sheet as mat_far and treats it as a half-space: no thickness, no internal structure, and
    // nothing behind it. The host layer's Airy denominator then reflects off an object that does
    // not exist.
    //
    // Two sheets are swept. SHEET (eps 5.24 - 0.63j) is a real boundary and measures the
    // cross-layer coupling: the denominator gets a reflection coefficient of roughly the right size
    // but the wrong value. LO_NEAR (eps 1.5015 - 0.40j) is near-matched to the host and measures
    // the series collapse: the half-space reflection is 2.5e-4, the denominator collapses onto 1,
    // and the host loses its internal series entirely, when physically the wave passes through the
    // sheet almost unimpeded and still sees the boundary behind it.
    //
    // The reference is the two-layer stack the crossing actually is. The quantity the harness can
    // isolate is the series factor, so the reference is written as
    //     S = 1 / (1 - r_ahead * r_behind * phi^2)
    // with r_behind taken from the characteristic matrix rather than from a half-space Fresnel
    // value. That factorization is exact: substituting the matrix value for r_behind reproduces the
    // full two-layer stack, which is what makes a series-factor assertion a statement about the
    // stack and not about the shape of the correction.
    //
    // Everything else in the row is held at the values the shipped diagnostic uses. No path_dir is
    // supplied, so abs_cos_theta_t is the fbs cosine, the propagation factor is the hybrid form
    // diag_phi reproduces, and both boundaries are taken from the slab side at that same real
    // cosine. The reference is built at the same cosine for the same reason.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9;

    const std::complex<double> e_h = diag_eps(HOST, f_hz), e_air(1.0, 0.0);

    struct Row
    {
        const char *name;
        int prev;
        bool near_matched;
    };
    const std::vector<Row> rows = {
        {"SHEET   eps 5.24 - 0.63j", SHEET, false},
        {"LO_NEAR eps 1.5015- 0.40j", NEAR, true},
    };

    QD_DIAG("\n# diag: host layer bounded by a sheet, host eps 1.50 - 0.40j, sheet thk 6 mm, 1 GHz.\n");
    QD_DIAG("# S_applied is the series SLAB_AIRY_FACTOR produces; S_ref is the same series with the\n");
    QD_DIAG("# sheet resolved as a layer backed by air, from the characteristic matrix. r_hs is the\n");
    QD_DIAG("# half-space reflection the solver used before the core landed and r_cm is the layer\n");
    QD_DIAG("# value it uses now; the gap between them is what the core closed.\n");
    QD_DIAG("sheet,L_mm,theta_n,|S_applied|,arg_S_applied,|S_ref|,arg_S_ref,|dS|,d_dB,|r_hs|,|r_cm|,rtype\n");

    for (const auto &R : rows)
        for (double L : {0.05, 0.10})
            for (double thd : {0.0, 20.0, 40.0, 50.0})
            {
                double theta_n = thd * QPI / 180.0;
                const std::complex<double> e_s = diag_eps(R.prev, f_hz);

                auto C = make1<double>(mtl, 4, OT_EXIT, 1u,
                                       /*M1*/ HOST, /*M2*/ 0, /*prev*/ R.prev, /*cur*/ HOST, /*buf*/ 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, /*eps: always resolve*/ 0.0);
                C.run();

                std::complex<double> out = cpx(C.xprmat, 0, 0), in = cpx(C.x_in, 0, 0);
                std::complex<double> applied = (std::abs(in) > 0.0) ? out / in
                                                                    : std::complex<double>(0.0, 0.0);
                std::complex<double> close = diag_close(e_h, L, theta_n, f_hz);
                std::complex<double> S_applied = (std::abs(close) > 0.0)
                                                     ? applied / close
                                                     : std::complex<double>(0.0, 0.0);

                // The reference. cs is the cosine correction mode uses on both boundaries, and the
                // matrix is anchored in the host so that its transverse invariant is the same one.
                std::complex<double> cs(std::cos(theta_n), 0.0);
                std::complex<double> phi = diag_phi(e_h, L, theta_n, f_hz);
                std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> r_hs = diag_r_slabside(e_h, e_s, cs);

                DiagSheet B = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz,
                                            false, true, /*undisplaced*/ 0.0);
                std::complex<double> S_ref = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);

                double dS = std::abs(S_applied - S_ref);
                double d_dB = 20.0 * std::log10(std::max(1e-300, std::abs(S_applied)) /
                                                std::max(1e-300, std::abs(S_ref)));

                QD_DIAG("%s,%.1f,%.1f,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%+.4f,%.3e,%.3e,%u\n",
                        R.name, L * 1000.0, thd,
                        std::abs(S_applied), std::arg(S_applied) * 180.0 / QPI,
                        std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI,
                        dS, d_dB, std::abs(r_hs), std::abs(B.R_te),
                        (unsigned)C.resolved_type[0]);

                INFO(R.name << ", L " << L * 1000.0 << " mm, theta_n " << thd);

                // The row has to be a resolve, or there is no series to measure.
                REQUIRE(C.resolved_type[0] != 0);
                REQUIRE(rt_bit(C.resolved_type[0], 2));
                REQUIRE(std::isfinite(C.gain(0)));

                // The two far-side reflections are far enough apart that the assertion below
                // discriminates. A statement about the references, true whichever one the solver is
                // built against, so it guards against a revert as well as against a bad reference.
                CHECK(std::abs(r_hs - B.R_te) > 1.0e-3);

                // A near-matched sheet is nearly not a boundary, so the layer it stands for is
                // nearly six more millimeters of host. At normal incidence the two descriptions of
                // that stack are the same object and the reference converges onto the extended-path
                // form linearly in the mismatch. Only at normal incidence: at an angle the two
                // carry different propagation conventions, the layer using k0 n thk cos and the
                // extended path the walk-off cos^2 of issue 9, and the gap between those two is not
                // a statement about this item.
                if (R.near_matched && thd == 0.0)
                {
                    std::complex<double> phi2 = diag_phi(e_h, L + thk, theta_n, f_hz);
                    std::complex<double> S_ext = 1.0 / (1.0 - r_ahead * r_ahead * phi2 * phi2);
                    CHECK(std::abs(S_ref - S_ext) < 1.0e-4);
                }

                // The host layer's series resolves against the sheet as a layer, magnitude and
                // phase together.
                CHECK(std::abs(std::abs(S_applied) - std::abs(S_ref)) < 1.0e-9);
                CHECK(std::abs(S_applied - S_ref) < 1.0e-9);
            }
}

TEST_CASE("Layered stack - sheet_case 4 resolves rather than killing the ray")
{
    // Symptom three. Two co-located sheets of different materials are an ordered stack. Detection
    // and the type bits are correct in both functions, but no sheet physics runs for it:
    // SHEET_EVENT calls KILL_RAY and the ray disappears.
    //
    // The stack here is air | A | B | air with both sheets 6 mm, so mtl_current bounds both sides,
    // near == far, the flux conversion is identically 1, and the displaced host is 12 mm of air.
    // That makes the target composite a pure two-layer characteristic matrix with nothing else
    // multiplying it: with mtl_current at 0, SHEET_EVENT closes out no host leg and returns without
    // touching the state words, so the written field IS the composite.
    //
    // Both orders are run. Bit 1 of the hit code carries the traversal order that ray_mesh_interact
    // derived from the face normals, the layer hit on its front side being traversed first, with
    // bit 3 recording that the order fell back to the geometric FBS -> SBS sequence. Note that
    // ray_state_update's case re-derivation currently keeps only sheet_mtl = M1 and leaves
    // sheet_solid at 0, so the second material index and the order are both discarded before
    // SHEET_EVENT sees them; the core needs them carried.
    //
    // Transmission is order-independent by reciprocity and only the reflect port separates the two,
    // which is asserted here on the reference as well as on the solver so that the order can be
    // seen to matter at all.
    const double f_hz = FRQ;
    const double thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0);
    const std::complex<double> e_air(1.0, 0.0);

    const uint8_t OT_STACK_FBS = 71; // ok | front | co-located | sheet: FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // ok | back  | co-located | sheet: SBS layer traversed first

    const int A = 3, B = 7; // SHEET eps 5.24 - 0.63j and LO_SHEET eps 1.50 - 0.40j, both 6 mm
    const std::complex<double> e_A = diag_eps(A, f_hz), e_B = diag_eps(B, f_hz);

    struct Ord
    {
        const char *name;
        uint8_t otype;
        int m1, m2;
        std::complex<double> first, second;
    };
    const std::vector<Ord> orders = {
        {"A|B", OT_STACK_FBS, A, B, e_A, e_B},
        {"B|A", OT_STACK_SBS, A, B, e_B, e_A},
    };

    QD_DIAG("\n# diag: sheet_case 4, air | 6 mm | 6 mm | air, sheets eps 5.24 - 0.63j and\n");
    QD_DIAG("# eps 1.50 - 0.40j, 1 GHz. ref_* is the two-layer characteristic matrix, out_* is what\n");
    QD_DIAG("# the stack resolve wrote. This case killed the ray outright before the core landed.\n");
    QD_DIAG("order,theta_n,ref_|T|,ref_|R|,out_|T|,out_gain_T,out_gain_R,dT,dR,rtype_T,rtype_R\n");

    for (const auto &O : orders)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);

            // near == far == air, so the conversion is 1 and the displaced host is air over the
            // whole 12 mm, which is what SHEET_RESOLVE leaves uncapped for an air host.
            DiagSheet Xs = diag_cm_stack(e_air, U, {{O.first, U, thk}, {O.second, U, thk}},
                                         e_air, U, ci, f_hz, false, /*scalar*/ true);
            DiagSheet Xe = diag_cm_stack(e_air, U, {{O.first, U, thk}, {O.second, U, thk}},
                                         e_air, U, ci, f_hz, false, /*EM*/ false);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, O.otype, 2u,
                                       /*M1*/ O.m1, /*M2*/ O.m2,
                                       /*prev*/ 0, /*cur = air*/ 0, /*buf*/ 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };

            auto Ct = run(4); // scalar transmission
            auto Ce = run(1); // EM transmission
            auto Cr = run(3); // scalar reflection
            auto Cx = run(0); // EM reflection

            double gT = 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
            double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));

            double dT = std::abs(cpx(Ct.xprmat, 0, 0) - Xs.T_te);
            double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%u,%u\n",
                    O.name, thd, std::abs(Xs.T_te), std::abs(Xs.R_te),
                    std::abs(cpx(Ct.xprmat, 0, 0)), (double)Ce.gain(0), (double)Cx.gain(0),
                    dT, dR, (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

            INFO("order " << O.name << ", theta_n " << thd);

            // The reference carries real energy on both ports, so a zero output is a dropped ray
            // and not a genuinely dark stack.
            CHECK(gT > 1.0e-3);
            CHECK(gR > 1.0e-3);

            // The stack resolves, both ports, scalar and EM. This case killed the ray outright
            // before the core landed, so the first two lines are the whole of what changed for it.
            CHECK(Ct.resolved_type[0] != 0);
            CHECK(Cr.resolved_type[0] != 0);
            CHECK(dT < 1.0e-9);
            CHECK(dR < 1.0e-9);
            CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
            CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);
        }

    // The order matters, and only in reflection. Asserted on the reference alone, so a core that
    // lost the traversal order fails here rather than passing a transmission-only check.
    for (double thd : {0.0, 45.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet AB = diag_cm_stack(e_air, U, {{e_A, U, thk}, {e_B, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        DiagSheet BA = diag_cm_stack(e_air, U, {{e_B, U, thk}, {e_A, U, thk}}, e_air, U, ci, f_hz,
                                     false, false);
        INFO("theta_n " << thd);
        CHECK(std::abs(AB.T_te - BA.T_te) < 1.0e-14);
        CHECK(std::abs(AB.T_tm - BA.T_tm) < 1.0e-14);
        CHECK(std::abs(AB.R_te - BA.R_te) > 1.0e-3);
    }
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

TEST_CASE("Lumped columns - interface gain on the sheet, glued and embedded paths")
{
    // ig = sqrt(slab.interface_gain * (far_is_new ? far.interface_gain : 1)) in produce mode, with
    // far_is_new = !far_is_air && !far.same_as(near). No sheet test has ever set att, so the branch
    // has only ever evaluated sqrt(1 * 1).
    //
    // What it has to say, in the quantities it holds. An interface gain is a penetration loss owed
    // once per medium ENTERED, which is what Material::interact_with charges for a single
    // transmission-class crossing: the entered medium's, never the one being left. A sheet crossing
    // enters the sheet, so it owes the sheet's. It enters the far medium as well only if the wave
    // was not already in it:
    //
    //   isolated   air  | sheet | air     G_sheet             air owes nothing
    //   entry      air  | sheet | wall    G_sheet * G_wall    the wall is entered here and nowhere else
    //   embedded   host | sheet | host    G_sheet             the host was entered earlier and paid then
    //   exit       host | sheet | air     G_sheet             leaving, and air owes nothing
    //
    // The embedded row is the one far_is_new was written for. Testing for air instead of for
    // near == far would charge the host a second time at every sheet inside it, and the row is
    // built so that this costs the full 5 dB of the host's column.
    //
    // The reference does not copy the gate. DiagLump::g_far is set by the row to what the far
    // medium is owed, so what is measured is the gate against the table above.
    //
    // The reflect port owes nothing: nothing was penetrated by the part that turns around at the
    // near face, and produce mode extends that to the whole port, including the series that did
    // enter the sheet and came back out. That second half is a convention rather than a
    // consequence - a traced solid charges the same echo its entry gain - and it is asserted here
    // as what the composite does, exactly, not as what a solid would do.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    const int SH = 1, SH0 = 2, WALL = 3, HOST = 4, HOST0 = 5;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk, /*att*/ 3.0}, // 1 SH     the sheet, 3 dB
        {5.24, 0.63, thk, 0.0},         // 2 SH0    its twin without the column
        {1.50, 0.40, 0.0, 5.0},         // 3 WALL   a lossy solid, 5 dB
        {1.50, 0.00, 0.0, 5.0},         // 4 HOST   a lossless solid, 5 dB
        {1.50, 0.00, 0.0, 0.0},         // 5 HOST0  its twin without the column
    };
    auto mtl = lump_palette(pal);
    auto E = [&](int idx)
    { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };

    const double G_sh = std::pow(10.0, -0.1 * 3.0), G_wall = std::pow(10.0, -0.1 * 5.0);

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2, cur, near, far;
        bool disp_far;
        double g_far; // what the far medium is owed
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air  | SH | air ", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 1.0, 85.0},
        {"entry     air  | SH | WALL", OT_GLUED_ENTRY, 2u, WALL, 0, 0, WALL, false, G_wall, 85.0},
        {"embedded  HOST | SH | HOST", OT_SHEET_ISO, 1u, 0, HOST, HOST, HOST, false, 1.0, 50.0},
        {"exit      HOST | SH | air ", OT_GLUED_EXIT, 2u, HOST, HOST, HOST, 0, true, 1.0, 50.0},
    };

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };
    auto P = [](std::complex<double> v)
    { return std::norm(v); };

    QD_DIAG("\n# diag: lumped interface gain, sheet eps 5.24 - 0.63j att 3 dB, wall and host att 5 dB\n");
    QD_DIAG("# paid_dB is the gain the composite carries against the same stack with the sheet's att\n");
    QD_DIAG("# removed; owed_dB is what the table in the case header says the crossing owes in total\n");
    QD_DIAG("stack,theta_n,P_te_meas,P_te_ref,P_tm_meas,P_tm_ref,d_te,d_tm,paid_dB,owed_dB,rtype\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 20.0, 40.0, 50.0, 70.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto run = [&](int sheet, int m2, int cur, int itype)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH, sheet, m2, 0, cur, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto C = run(SH, R.m2, R.cur, 1);
            auto C0 = run(SH0, R.m2, R.cur, 1); // the same crossing, sheet without the column

            DiagLump Lp;
            Lp.g_slab = G_sh, Lp.g_far = R.g_far;
            DiagSheet D = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz, R.disp_far,
                                               false, 1.0, 1.0, 1.0, Lp);

            // Hosts are lossless here, so the leg a row inside a medium closes out contributes a
            // pure phase and the channel powers are the composite's own.
            double p_te = P(TE(C.xprmat)), p_tm = P(TM(C.xprmat));
            double paid = 10.0 * std::log10(p_te / P(TE(C0.xprmat)));
            double owed = 10.0 * std::log10(G_sh * R.g_far);

            QD_DIAG("%s,%.1f,%.9e,%.9e,%.9e,%.9e,%.2e,%.2e,%+.6f,%+.6f,%u\n", R.name, thd,
                    p_te, P(D.T_te), p_tm, P(D.T_tm), p_te - P(D.T_te), p_tm - P(D.T_tm),
                    paid, owed, (unsigned)C.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(C.gain(0)));
            REQUIRE(C.resolved_type[0] != 0);

            // Each gain owed is paid, once.
            CHECK(std::abs(p_te - P(D.T_te)) < 1.0e-9);
            CHECK(std::abs(p_tm - P(D.T_tm)) < 1.0e-9);
            CHECK(std::abs((double)C.gain(0) - 0.5 * (P(D.T_te) + P(D.T_tm))) < 1.0e-9);
            if (R.cur == 0) // nothing else multiplies the composite: pin it complex
            {
                CHECK(std::abs(TE(C.xprmat) - D.T_te) < 1.0e-9);
                CHECK(std::abs(TM(C.xprmat) - D.T_tm) < 1.0e-9);
            }

            // Exact by construction: the sheet's gain is one real factor on the transmit port, so
            // the field with and without it differ by sqrt(G_sheet) and by nothing else.
            CHECK(std::abs(p_te / P(TE(C0.xprmat)) - G_sh) < 1.0e-14);
            CHECK(std::abs(p_tm / P(TM(C0.xprmat)) - G_sh) < 1.0e-14);

            // And the assertion above cannot pass with a gain dropped or paid twice: each is worth
            // dB, not rounding.
            CHECK(std::abs(p_te - P(D.T_te) / G_sh) > 1.0e-3 * p_te);
            CHECK(std::abs(p_te - P(D.T_te) * G_sh) > 1.0e-3 * p_te);

            if (R.near == HOST && R.far == HOST)
            {
                // The embedded sheet pays the host's gain not at all. Stated twice. Against the
                // reference: charging it would cost the full 5 dB. And against the host's twin
                // without the column, where it is exact by construction: with near == far the
                // host's att column is never read, so the two runs are the same arithmetic.
                auto Ch = run(SH, 0, HOST0, 1);
                CHECK(std::abs(p_te - P(D.T_te) * G_wall) > 0.5 * p_te);
                CHECK(p_te == P(TE(Ch.xprmat)));
                CHECK(p_tm == P(TM(Ch.xprmat)));
            }
            if (R.far == WALL)
            {
                // The glued entry pays the wall's, which no other event on the path will: without
                // it the wall's penetration loss disappears for every wall that carries a coating.
                CHECK(std::abs(p_te - P(D.T_te) / G_wall) > 0.5 * p_te);
            }

            // The reflect port carries the sheet's gain on the part that entered the sheet, once,
            // and never the wall's. Complex on the rows where the composite is the whole of what
            // is written; and told apart from the sheet without the column.
            if (R.cur == 0)
            {
                auto Cr = run(SH, R.m2, R.cur, 0);
                auto Cr0 = run(SH0, R.m2, R.cur, 0);
                CHECK(std::abs(TE(Cr.xprmat) - D.R_te) < 1.0e-9);
                CHECK(std::abs(TM(Cr.xprmat) - D.R_tm) < 1.0e-9);
                CHECK(std::abs(TE(Cr.xprmat) - TE(Cr0.xprmat)) > 1.0e-3 * std::abs(TE(Cr0.xprmat)));
                CHECK(std::abs(TM(Cr.xprmat) - TM(Cr0.xprmat)) > 1.0e-3 * std::abs(TM(Cr0.xprmat)));
            }
        }
}

TEST_CASE("Lumped columns - the transmission factor on a sheet, both ports")
{
    // The AMP lambda in produce mode, and the apply_tf call in the two boundary lambdas that feeds
    // it. Per boundary and per channel, with R0 the raw Fresnel reflectance and R' = apply_tf(R0):
    //
    //     r' = sqrt(R') exp(j arg r)        t' = t sqrt((1 - R') / (1 - R0))
    //
    // so the reflectance is replaced, the transmission keeps its Fresnel phase, and its power is
    // moved by exactly what the reflection gave up or took. tf > 0 leaks, R' = R0 (1 - tf), which
    // scales |r| by sqrt(1 - tf) whatever the angle but scales t by an amount that grows with R0
    // and is unbounded towards grazing; tf < 0 forces, R' = R0 + (1 - R0)|tf|, which scales t by
    // sqrt(1 - |tf|) whatever the angle. Both faces take the SHEET's tf.
    //
    // tf is not a boundary admittance, so the characteristic matrix cannot hold it. The Airy pair
    // can, because it is written in exactly the per-boundary r and t that tf rescales, and
    // diag_sheet_composite is in that form. So the reference expresses the rule in closed form and
    // the assertion is the ordinary one. What it cannot be is independent of the rule: it is the
    // same convention written a second time. Two statements below are independent of it.
    //
    //   Unitarity. In flux-normalized amplitudes a lossless boundary below its critical angle is
    //   [[r, tau], [tau, -r]] with r and tau real and r^2 + tau^2 = 1. The rescale keeps both
    //   phases and keeps the sum, so the boundary is still unitary, and a lossless sheet between
    //   two unitary boundaries conserves energy per channel at any tf. Nothing in the reference is
    //   needed to say that.
    //
    //   The end points. tf = +1 removes the reflection outright, so the reflect port is exactly
    //   zero and a lossless sheet is a pure phase; tf = -1 makes both faces mirrors, so the
    //   transmit port is exactly zero.
    //
    // A tf-bearing PAIR, sheet and solid both, is the pair rule: each face is owed by the two
    // media that form it. The last block is the table that used to record a wall's tf vanishing
    // under a coating, now asserted against the reference carrying the wall's column; the full
    // sweep is in the neighbour-tf case below.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SH = 1, WALL = 2, HOST = 3, SHL = 4;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int m2, cur, near, far;
        bool disp_far;
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air  | SH | air ", OT_SHEET_ISO, 1u, 0, 0, 0, 0, false, 85.0},
        {"entry     air  | SH | WALL", OT_GLUED_ENTRY, 2u, WALL, 0, 0, WALL, false, 85.0},
        {"exit      HOST | SH | air ", OT_GLUED_EXIT, 2u, HOST, HOST, HOST, 0, true, 50.0},
    };

    QD_DIAG("\n# diag: transmission factor on a 6 mm sheet, eps 5.24 - 0.63j, both ports\n");
    QD_DIAG("# dT_dB and dR_dB are against the same crossing at tf = 0, from the reference\n");
    QD_DIAG("stack,tf,theta_n,|T_te|,|T_tm|,|R_te|,|R_tm|,d_T,d_R,d_scalar,dT_dB,dR_dB\n");

    double worst_T = 0.0, worst_R = 0.0, worst_E = 0.0;
    for (double tf : {-1.0, -0.5, -0.25, 0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, tf}, // 1 SH    the sheet
            {1.50, 0.40, 0.0},          // 2 WALL  a lossy solid
            {1.50, 0.00, 0.0},          // 3 HOST  a lossless solid
            {5.24, 0.00, thk, 0.0, tf}, // 4 SHL   a lossless sheet with the same tf
        };
        auto mtl = lump_palette(pal);
        auto E = [&](int idx)
        { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };

        DiagLump Lp;
        Lp.tf = tf;

        for (const auto &R : rows)
            for (double thd : {0.0, 20.0, 45.0, 50.0, 70.0, 85.0})
            {
                if (thd > R.lim)
                    continue;
                double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

                auto run = [&](int itype)
                {
                    auto C = make1<double>(mtl, itype, R.otype, R.nH, SH, R.m2, 0, R.cur, 0,
                                           L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                    C.run();
                    return C;
                };
                auto Ct = run(1), Cr = run(0);

                DiagSheet D = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                   R.disp_far, false, 1.0, 1.0, 1.0, Lp);
                DiagSheet D0 = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                    R.disp_far, false);

                std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);
                std::complex<double> r_te = TE(Cr.xprmat), r_tm = TM(Cr.xprmat);

                // In air nothing else multiplies either port and both are compared complex. Inside
                // a host the transmit port picks up the phase of the leg it closes and the reflect
                // port the host's own bookkeeping, so that row is compared in transmitted power.
                double d_T, d_R = 0.0, d_S = 0.0;
                if (R.cur == 0)
                {
                    d_T = std::max(std::abs(t_te - D.T_te), std::abs(t_tm - D.T_tm));
                    d_R = std::max(std::abs(r_te - D.R_te), std::abs(r_tm - D.R_tm));

                    // Scalar mode carries one channel, the TE / admittance form.
                    DiagSheet Ds = diag_sheet_composite(E(R.near), E(SH), E(R.far), thk, ct, f_hz,
                                                        R.disp_far, true, 1.0, 1.0, 1.0, Lp);
                    auto St = run(4), Sr = run(3);
                    d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te),
                                   std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
                }
                else
                    d_T = std::max(std::abs(std::norm(t_te) - std::norm(D.T_te)),
                                   std::abs(std::norm(t_tm) - std::norm(D.T_tm)));

                worst_T = std::max(worst_T, d_T), worst_R = std::max(worst_R, std::max(d_R, d_S));

                double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
                double gT0 = 0.5 * (std::norm(D0.T_te) + std::norm(D0.T_tm));
                double gR = 0.5 * (std::norm(D.R_te) + std::norm(D.R_tm));
                double gR0 = 0.5 * (std::norm(D0.R_te) + std::norm(D0.R_tm));

                QD_DIAG("%s,%+.2f,%.1f,%.9f,%.9f,%.9f,%.9f,%.2e,%.2e,%.2e,%+.4f,%+.4f\n",
                        R.name, tf, thd, std::abs(t_te), std::abs(t_tm), std::abs(r_te), std::abs(r_tm),
                        d_T, d_R, d_S,
                        (gT > 0.0) ? 10.0 * std::log10(gT / gT0) : -999.0,
                        (gR > 0.0) ? 10.0 * std::log10(gR / gR0) : -999.0);

                INFO(R.name << ", tf " << tf << ", theta_n " << thd);
                REQUIRE(std::isfinite(Ct.gain(0)));
                REQUIRE(std::isfinite(Cr.gain(0)));

                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(d_S < 1.0e-9);
                CHECK(std::abs((double)Ct.gain(0) - gT) < 1.0e-9);

                // tf moves both ports by far more than they are asserted to, so the assertions
                // above are not passing on a rescale that never ran. The transmit port moves least
                // for a weak leak into a glued solid, 0.076 dB at tf = 0.25, and the reflect port
                // least for a weak forced reflection at grazing incidence, 0.87 dB at tf = -0.25.
                CHECK(std::abs(gT - gT0) > 1.0e-3 * gT0);
                CHECK(std::abs(gR - gR0) > 1.0e-3 * gR0);

                if (R.cur == 0)
                {
                    // The end points, exact by construction: sqrt(0) is 0 and every term of the
                    // port that closes carries it.
                    if (tf == 1.0)
                        CHECK((double)Cr.gain(0) == 0.0);
                    if (tf == -1.0)
                        CHECK((double)Ct.gain(0) == 0.0);
                }
            }

        // Unitarity, on a lossless sheet in air. No reference enters: this is a statement about
        // the rule itself, and it is the reason the rescale has the form it has.
        for (double thd : {0.0, 30.0, 60.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0;
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_SHEET_ISO, 1u, SHL, 0, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), Cr = run(0);
            double e_te = std::norm(TE(Ct.xprmat)) + std::norm(TE(Cr.xprmat));
            double e_tm = std::norm(TM(Ct.xprmat)) + std::norm(TM(Cr.xprmat));
            worst_E = std::max(worst_E, std::max(std::abs(e_te - 1.0), std::abs(e_tm - 1.0)));

            INFO("lossless sheet, tf " << tf << ", theta_n " << thd);
            CHECK(std::abs(e_te - 1.0) < 1.0e-12);
            CHECK(std::abs(e_tm - 1.0) < 1.0e-12);
            if (tf == 1.0) // no reflection at either face: the sheet is a pure phase
                CHECK(std::abs(std::abs(TE(Ct.xprmat)) - 1.0) < 1.0e-12);
        }
    }
    QD_DIAG("# worst: transmit %.3e, reflect and scalar %.3e, energy of a lossless sheet %.3e\n",
            worst_T, worst_R, worst_E);

    // The far medium's tf. A glued entry into a wall that carries one, with a sheet that does not,
    // against the same entry with the wall's column removed. The wall's boundary is now
    // sheet | wall rather than air | wall, so the column is not worth what it is worth on the bare
    // wall, and a forced far face adds a series against the sheet's front face. Asserted against
    // the reference. bare_dB is what the same column is worth on the uncoated wall, from the
    // rule Material::interact_with applies to a single crossing: the TE/TM-averaged reflectance R
    // becomes R (1 - tf) for a leak and R + (1 - R)|tf| for a forced reflection, and the
    // transmitted power moves by (1 - R') / (1 - R).
    QD_DIAG("\n# diag: the far medium's tf on a glued entry, air | sheet (tf 0) | wall (tf)\n");
    QD_DIAG("# composite_dB is what the wall's column moves the coated entry by, bare_dB what it\n");
    QD_DIAG("# moves the uncoated wall by\n");
    QD_DIAG("wall_tf,theta_n,composite_dB,bare_dB\n");
    for (double wtf : {0.5, -0.5})
        for (double thd : {0.0, 45.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            double g[2];
            for (int k = 0; k < 2; ++k)
            {
                const std::vector<LumpMtl> pal = {
                    {5.24, 0.63, thk},                       // 1 the sheet, no tf
                    {1.50, 0.40, 0.0, 0.0, (k ? wtf : 0.0)}, // 2 the wall
                };
                auto mtl = lump_palette(pal);
                auto C = make1<double>(mtl, 1, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                g[k] = (double)C.gain(0);
            }

            std::complex<double> ew(1.50, -0.40), zw = std::sqrt(ew);
            std::complex<double> cw = std::sqrt(1.0 - (1.0 / ew) * (1.0 - ct * ct));
            double Rb = 0.5 * (std::norm((ct - zw * cw) / (ct + zw * cw)) +
                               std::norm((zw * ct - cw) / (zw * ct + cw)));
            double Rb_tf = (wtf >= 0.0) ? Rb * (1.0 - wtf) : Rb + (1.0 - Rb) * (-wtf);

            QD_DIAG("%+.1f,%.1f,%+.6f,%+.6f\n", wtf, thd, 10.0 * std::log10(g[1] / g[0]),
                    10.0 * std::log10((1.0 - Rb_tf) / (1.0 - Rb)));
            CHECK(std::isfinite(g[1]));
            CHECK(g[1] > 0.0);

            double gr[2];
            for (int k = 0; k < 2; ++k)
            {
                DiagLump Lw;
                Lw.tf_far = k ? wtf : 0.0;
                DiagSheet Dw = diag_sheet_composite(1.0, std::complex<double>(5.24, -0.63), ew, thk, ct, f_hz,
                                                    false, false, 1.0, 1.0, 1.0, Lw);
                gr[k] = 0.5 * (std::norm(Dw.T_te) + std::norm(Dw.T_tm));
            }
            INFO("coated wall, wall tf " << wtf << ", theta_n " << thd);
            CHECK(std::abs(10.0 * std::log10(g[1] / g[0]) - 10.0 * std::log10(gr[1] / gr[0])) < 1.0e-6);
            CHECK(std::abs(g[1] - g[0]) > 1.0e-3 * g[0]); // and the wall's column is read at all
        }
}

TEST_CASE("Lumped columns - a permittivity resonance is an interface term, not an index")
{
    // eta_resonance is added to the permittivity wherever an INTERFACE quantity is formed - every
    // admittance sqrt((eps + res)/mu), and every Snell ratio and cosine, since those are built from
    // the same interface permittivity - and is left out of every propagation INDEX, which stays
    // sqrt(eps mu). Produce mode makes that split at five sites: the in-sheet cosine, the near
    // boundary, the far boundary, the near-to-far flux conversion and the displaced host. One test
    // in the suite sets resF and it is not a sheet, so the split has never been evaluated with a
    // non-zero resonance on any sheet path.
    //
    // The split is not "the interface terms move and the propagation factor does not". The
    // propagation factor is exp(-j k0 n thk c) and its cosine is an interface quantity, so at
    // oblique incidence it does move with the resonance, through c and not through n. That is what
    // the code does and what the reference does, and the planned assertion that the propagation
    // factor is unmoved is exact only at normal incidence. It is withdrawn in that form and made in
    // the form the code can express, below.
    //
    // Three kinds of statement.
    //
    //   Against the reference carrying the split, 1e-9, complex wherever nothing else multiplies
    //   the composite: a resonant sheet in air, a resonant sheet glued to a plain solid, a plain
    //   sheet glued to a resonant solid, and a resonant sheet inside a resonant host. The last one
    //   is what reaches the displaced-host site, where the index is the host's sqrt(eps mu) and the
    //   cosine is the host's interface cosine.
    //
    //   Against the two ways of getting the split wrong, as margins: the resonance dropped, and the
    //   resonance folded into the index as well. Either would otherwise pass a check built on a
    //   reference with the same mistake in it. The second margin exists only where a resonant
    //   medium is propagated through - the sheet, and the host it displaces - so the row with a
    //   plain sheet on a resonant solid carries the first margin alone.
    //
    //   Without a reference at all. A lossless sheet of eps 4 whose resonance takes its interface
    //   permittivity to exactly 1 presents no boundary to a wave in air: both reflections vanish at
    //   every angle and in both channels, the cosine inside it is the air cosine, and what is left
    //   of the sheet is its propagation at the index the resonance did not touch,
    //   exp(-j k0 thk cos(theta) (2 - 1)). The interface terms followed the resonance all the way to
    //   zero and the index did not move, which is the split in one measurement. It is also not a
    //   wave any medium supports - an index of 2 carried along the cosine of an index of 1 - which
    //   is worth knowing about a column that is meant for interfaces.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const double L = 0.10;
    const double k0 = 2.0 * QPI * f_hz / C0;

    const int SHR = 1, WALL = 2, WALLR = 3, HOSTR = 4, SH = 5, MATCH = 6;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk, 0.0, 0.0, 1.2, 5.0, 0.8},     // 1 SHR    resonant sheet
        {1.50, 0.40, 0.0},                              // 2 WALL   plain lossy solid
        {1.50, 0.40, 0.0, 0.0, 0.0, 0.8, 4.0, 0.5},     // 3 WALLR  resonant lossy solid
        {1.50, 0.00, 0.0, 0.0, 0.0, 0.8, 4.0, 0.5},     // 4 HOSTR  resonant solid, lossless index
        {5.24, 0.63, thk},                              // 5 SH     plain sheet
        {4.00, 0.00, thk, 0.0, 0.0, 2.0, 1.0e300, -2.25}, // 6 MATCH  eps 4, interface eps exactly 1
    };
    auto mtl = lump_palette(pal);
    const std::complex<double> one(1.0, 0.0), zero(0.0, 0.0);
    auto E = [&](int idx)
    { return (idx == 0) ? one : lump_eps(pal[(size_t)idx - 1], f_hz); };
    auto Rz = [&](int idx)
    { return (idx == 0) ? zero : lump_res(pal[(size_t)idx - 1], f_hz); };

    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Row
    {
        const char *name;
        uint8_t otype;
        unsigned nH;
        int sheet, m2, cur, near, far;
        double lim;
    };
    const std::vector<Row> rows = {
        {"isolated  air   | SHR | air  ", OT_SHEET_ISO, 1u, SHR, 0, 0, 0, 0, 85.0},
        {"entry     air   | SHR | WALL ", OT_GLUED_ENTRY, 2u, SHR, WALL, 0, 0, WALL, 85.0},
        {"entry     air   | SH  | WALLR", OT_GLUED_ENTRY, 2u, SH, WALLR, 0, 0, WALLR, 85.0},
        {"embedded  HOSTR | SHR | HOSTR", OT_SHEET_ISO, 1u, SHR, 0, HOSTR, HOSTR, HOSTR, 50.0},
    };

    QD_DIAG("\n# diag: permittivity resonance on the sheet paths. sheet res (%.4f, %.4f), solid res\n",
            std::real(Rz(SHR)), std::imag(Rz(SHR)));
    QD_DIAG("# (%.4f, %.4f) at 1 GHz. d_split is against the reference carrying the interface /\n",
            std::real(Rz(WALLR)), std::imag(Rz(WALLR)));
    QD_DIAG("# index split; sep_dropped and sep_folded are how far the same field sits from a\n");
    QD_DIAG("# reference with the resonance removed, and with it folded into the index as well\n");
    QD_DIAG("stack,theta_n,|T_te|,|T_tm|,d_split,d_refl,d_scalar,sep_dropped,sep_folded,rtype\n");

    double worst = 0.0;
    for (const auto &R : rows)
        for (double thd : {0.0, 20.0, 40.0, 50.0, 70.0, 85.0})
        {
            if (thd > R.lim)
                continue;
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);

            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, R.otype, R.nH, R.sheet, R.m2, 0, R.cur, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1);

            DiagLump Lp;
            Lp.res_near = Rz(R.near), Lp.res_slab = Rz(R.sheet), Lp.res_far = Rz(R.far);

            // The split, and the two ways of getting it wrong.
            DiagSheet D = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz, false,
                                               false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Dd = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz, false,
                                                false);
            DiagSheet Df = diag_sheet_composite(E(R.near) + Rz(R.near), E(R.sheet) + Rz(R.sheet),
                                                E(R.far) + Rz(R.far), thk, ct, f_hz, false, false);

            std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);

            // The host of the embedded row propagates at its resonance-excluded index, which is
            // lossless, so the leg it closes contributes a pure phase. Compared in power there.
            auto dist = [&](const DiagSheet &X)
            {
                if (R.cur == 0)
                    return std::max(std::abs(t_te - X.T_te), std::abs(t_tm - X.T_tm));
                return std::max(std::abs(std::norm(t_te) - std::norm(X.T_te)),
                                std::abs(std::norm(t_tm) - std::norm(X.T_tm)));
            };
            double d_split = dist(D), sep_d = dist(Dd), sep_f = dist(Df);

            double d_refl = 0.0, d_scalar = 0.0;
            if (R.cur == 0)
            {
                auto Cr = run(0);
                d_refl = std::max(std::abs(TE(Cr.xprmat) - D.R_te), std::abs(TM(Cr.xprmat) - D.R_tm));

                DiagSheet Ds = diag_sheet_composite(E(R.near), E(R.sheet), E(R.far), thk, ct, f_hz,
                                                    false, true, 1.0, 1.0, 1.0, Lp);
                auto St = run(4), Sr = run(3);
                d_scalar = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te),
                                    std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
            }
            worst = std::max(worst, std::max(d_split, std::max(d_refl, d_scalar)));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.2e,%.2e,%.2e,%.3e,%.3e,%u\n", R.name, thd,
                    std::abs(t_te), std::abs(t_tm), d_split, d_refl, d_scalar, sep_d, sep_f,
                    (unsigned)Ct.resolved_type[0]);

            INFO(R.name << ", theta_n " << thd);
            REQUIRE(std::isfinite(Ct.gain(0)));
            REQUIRE(Ct.resolved_type[0] != 0);
            CHECK(d_split < 1.0e-9);
            CHECK(d_refl < 1.0e-9);
            CHECK(d_scalar < 1.0e-9);
            CHECK(std::abs((double)Ct.gain(0) - 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm))) < 1.0e-9);

            // The resonance dropped is excluded by a margin six orders above the bound, on every
            // row. The resonance folded into the index is excluded wherever a resonant medium's
            // index is in the composite at all, which is the sheet and the host it displaces. A
            // resonant FAR half-space is not such a row: nothing propagates through a half-space,
            // so its index appears nowhere and the two references are the same expression. That
            // row reaches the far boundary and the flux conversion and cannot reach the split.
            CHECK(sep_d > 1.0e-3);
            if (R.sheet == SHR)
                CHECK(sep_f > 1.0e-3);
            else
                CHECK(std::max(std::abs(D.T_te - Df.T_te), std::abs(D.T_tm - Df.T_tm)) < 1.0e-14);
        }
    QD_DIAG("# worst against the split reference %.3e\n", worst);

    // The sheet whose interfaces the resonance removes.
    QD_DIAG("\n# diag: resonance-matched sheet, eps 4, interface eps (%.17g, %.1e), thk 6 mm.\n",
            std::real(E(MATCH) + Rz(MATCH)), std::imag(E(MATCH) + Rz(MATCH)));
    QD_DIAG("# pred is -k0 thk cos(theta) (2 - 1); unmoved is what the phase would be if the cosine\n");
    QD_DIAG("# inside the sheet were the one eps 4 gives, i.e. if the propagation factor did not\n");
    QD_DIAG("# move with the resonance at all\n");
    QD_DIAG("theta_n,|R_te|,|R_tm|,|T|-1,arg_T_deg,pred_deg,d_arg_deg,unmoved_deg\n");

    for (double thd : {0.0, 30.0, 60.0, 85.0})
    {
        double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
        auto run = [&](int itype)
        {
            auto C = make1<double>(mtl, itype, OT_SHEET_ISO, 1u, MATCH, 0, 0, 0, 0,
                                   L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
            C.run();
            return C;
        };
        auto Ct = run(1), Cr = run(0);

        std::complex<double> t_te = TE(Ct.xprmat), t_tm = TM(Ct.xprmat);
        double pred = -k0 * thk * ct * (2.0 - 1.0);
        double c4 = std::sqrt(1.0 - (1.0 - ct * ct) / 4.0);
        double unmoved = -k0 * thk * (2.0 * c4 - ct);

        QD_DIAG("%.1f,%.3e,%.3e,%.3e,%.9f,%.9f,%.3e,%.9f\n", thd,
                std::abs(TE(Cr.xprmat)), std::abs(TM(Cr.xprmat)), std::abs(t_te) - 1.0,
                std::arg(t_te) * 180.0 / QPI, pred * 180.0 / QPI,
                (std::arg(t_te) - pred) * 180.0 / QPI, unmoved * 180.0 / QPI);

        INFO("resonance-matched sheet, theta_n " << thd);

        // No boundary, at any angle, in either channel.
        CHECK(std::abs(TE(Cr.xprmat)) < 1.0e-12);
        CHECK(std::abs(TM(Cr.xprmat)) < 1.0e-12);

        // What is left is propagation at the untouched index along the interface cosine.
        CHECK(std::abs(std::abs(t_te) - 1.0) < 1.0e-12);
        CHECK(std::abs(t_te - std::polar(1.0, pred)) < 1.0e-12);
        CHECK(std::abs(t_tm - t_te) < 1.0e-12);

        // At normal incidence the cosine is 1 whatever the permittivity, so there the propagation
        // factor is unmoved outright. Away from it the two readings separate, by 5.8 deg at 60 deg.
        if (thd == 0.0)
            CHECK(std::abs(pred - unmoved) == 0.0);
        else
            CHECK(std::abs(pred - unmoved) * 180.0 / QPI > 1.0);
    }
}

TEST_CASE("Lumped columns - an ordered stack: what the core carries, by matrix and by scattering")
{
    // The characteristic-matrix core generalizes produce mode's two expressible columns to a
    // stack, and declines on the three it has no expression for.
    //
    // Carried. The interface gains become a product, one per layer entered plus the far medium's
    // where it is newly entered, on the transmit port only. The resonance split is made per layer:
    // the tilted admittance and the cosine read eps + res, the layer phase's index reads eps.
    //
    // alpha is carried as well, folded into the layer phase as nepers over the path thk / |Re c|.
    // It has two statements here. Against the reference, which carries it the same way. And
    // against produce mode, with no reference at all: a sheet of eps 5.24 - 0.63j with 100 dB/m
    // stacked on its twin without the column is 12 mm of one permittivity with no boundary in the
    // middle, so every wave crosses both halves and the stack is the single 12 mm sheet at
    // 50 dB/m, on both ports. One is resolved by the core and the other by produce mode.
    //
    // tf and m have no matrix expression and used to make qd_stack_resolve return false, which
    // for an ordered stack killed the ray. They are resolved in the scattering form now - tf per
    // boundary by the pair rule, a mass-law layer as the end of a coherent segment - and the last
    // block asserts the two rows that used to be killed against that reference. The mass-law row
    // here has a zero mass term, its argument being below one, so it measures the route alone.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    const int A = 1, B = 2, A_ATT = 3, B_ATT = 4, A_RES = 5, B_ALPHA = 6, B_TF = 7, B_M = 8;
    const int A_ALPHA = 9, A_THICK = 10;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk},                              // 1 A
        {1.50, 0.40, thk},                              // 2 B
        {5.24, 0.63, thk, 3.0},                         // 3 A_ATT    3 dB
        {1.50, 0.40, thk, 2.0},                         // 4 B_ATT    2 dB
        {5.24, 0.63, thk, 0.0, 0.0, 1.2, 5.0, 0.8},     // 5 A_RES    resonant
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0}, // 6 B_ALPHA  50 dB/m
        {1.50, 0.40, thk, 0.0, 0.25},                   // 7 B_TF     tf 0.25
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0}, // 8 B_M  mass law
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0},       // 9 A_ALPHA  100 dB/m
        {5.24, 0.63, 2.0 * thk, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0},  // 10 A_THICK 12 mm at 50 dB/m
    };
    auto mtl = lump_palette(pal);
    auto E = [&](int idx)
    { return lump_eps(pal[(size_t)idx - 1], f_hz); };
    auto Rz = [&](int idx)
    { return lump_res(pal[(size_t)idx - 1], f_hz); };
    auto Al = [&](int idx)
    { return pal[(size_t)idx - 1].alpha; };

    const uint8_t OT_STACK_FBS = 71; // FBS layer traversed first
    const uint8_t OT_STACK_SBS = 69; // SBS layer traversed first

    auto run = [&](int itype, uint8_t otype, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0,
                               0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // Carried: att on both layers, a resonance in one, and alpha in one.
    // ------------------------------------------------------------------------------------------
    struct Carry
    {
        const char *name;
        int m1, m2;  // the two faces, FBS and SBS
        double g;    // product of the layer gains
    };
    const double G_A = std::pow(10.0, -0.1 * 3.0), G_B = std::pow(10.0, -0.1 * 2.0);
    const std::vector<Carry> carried = {
        {"att 3 dB | att 2 dB", A_ATT, B_ATT, G_A * G_B},
        {"resonant | plain   ", A_RES, B, 1.0},
        {"plain | alpha 50   ", A, B_ALPHA, 1.0},
    };

    QD_DIAG("\n# diag: ordered stack carrying att, a resonance and alpha, air | 6 mm | 6 mm | air.\n");
    QD_DIAG("# paid_dB is the EM transmit gain against the same stack with neither column\n");
    QD_DIAG("stack,order,theta_n,ref_|T|,ref_|R|,dT,dR,d_gain_T,d_gain_R,paid_dB\n");

    for (const auto &S : carried)
        for (int ord = 0; ord < 2; ++ord)
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> ci(std::cos(th), 0.0);
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;

                std::vector<DiagLayer> lay = {{E(first), U, thk, Rz(first), Al(first)},
                                              {E(second), U, thk, Rz(second), Al(second)}};
                DiagSheet Xs = diag_cm_stack(e_air, U, lay, e_air, U, ci, f_hz, false, true);
                DiagSheet Xe = diag_cm_stack(e_air, U, lay, e_air, U, ci, f_hz, false, false);
                double ig = std::sqrt(S.g);

                auto Ct = run(4, ot, S.m1, S.m2, th), Ce = run(1, ot, S.m1, S.m2, th);
                auto Cr = run(3, ot, S.m1, S.m2, th), Cx = run(0, ot, S.m1, S.m2, th);
                auto C0 = run(1, ot, A, B, th); // neither column

                // The reflect port with att: what entered a layer and came back paid that layer's
                // gain once. A matrix has no per-layer echo to charge, so that row takes its
                // reflect port from the scattering reference, which carries a gain per layer.
                if (S.g != 1.0)
                {
                    std::vector<DiagLayer> lg = lay;
                    lg[0].g = (first == A_ATT) ? G_A : G_B, lg[1].g = (second == A_ATT) ? G_A : G_B;
                    DiagSheet Ys = diag_sc_stack(e_air, U, 0.0, lg, e_air, U, 0.0, ci, f_hz, false, true);
                    DiagSheet Ye = diag_sc_stack(e_air, U, 0.0, lg, e_air, U, 0.0, ci, f_hz, false, false);
                    CHECK(std::abs(Ys.R_te - Xs.R_te) > 1.0e-3); // and the gain is worth something on it
                    Xs.R_te = Ys.R_te, Xe.R_te = Ye.R_te, Xe.R_tm = Ye.R_tm;
                }

                double gT = S.g * 0.5 * (std::norm(Xe.T_te) + std::norm(Xe.T_tm));
                double gR = 0.5 * (std::norm(Xe.R_te) + std::norm(Xe.R_tm));
                double dT = std::abs(cpx(Ct.xprmat, 0, 0) - ig * Xs.T_te);
                double dR = std::abs(cpx(Cr.xprmat, 0, 0) - Xs.R_te);

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e,%.3e,%+.6f\n", S.name, ord ? "2|1" : "1|2",
                        thd, std::abs(ig * Xs.T_te), std::abs(Xs.R_te), dT, dR,
                        std::abs((double)Ce.gain(0) - gT), std::abs((double)Cx.gain(0) - gR),
                        10.0 * std::log10((double)Ce.gain(0) / (double)C0.gain(0)));

                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                REQUIRE(Ct.resolved_type[0] != 0);
                REQUIRE(Cr.resolved_type[0] != 0);

                // Each layer's gain once on the transmit port, once on what came back out of it
                // on the reflect port, and the resonance split per layer, scalar and EM.
                CHECK(dT < 1.0e-9);
                CHECK(dR < 1.0e-9);
                CHECK(std::abs((double)Ce.gain(0) - gT) < 1.0e-9);
                CHECK(std::abs((double)Cx.gain(0) - gR) < 1.0e-9);

                if (S.g != 1.0)
                {
                    // The gains are one real factor on the transmit port. A stack with att takes
                    // the scattering form and its twin without the column the matrix, two forms
                    // of one product, so the ratio is at the rounding floor and not exact.
                    CHECK(std::abs((double)Ce.gain(0) / (double)C0.gain(0) - S.g) < 1.0e-12);
                    // A gain dropped, or one layer's paid twice, is worth dB.
                    CHECK(std::abs((double)Ce.gain(0) - gT / G_A) > 1.0e-3 * gT);
                    CHECK(std::abs((double)Ce.gain(0) - gT * G_B) > 1.0e-3 * gT);
                }
                else
                {
                    // The column moved the stack: the plain reference is excluded by a margin.
                    DiagSheet Xp = diag_cm_stack(e_air, U, {{E(first), U, thk}, {E(second), U, thk}},
                                                 e_air, U, ci, f_hz, false, true);
                    CHECK(std::abs(cpx(Ct.xprmat, 0, 0) - Xp.T_te) > 1.0e-3);
                }
            }

    // ------------------------------------------------------------------------------------------
    // alpha against produce mode. The core's two-layer stack and produce mode's single sheet are
    // the same object, so the two solvers are measured against each other through the library.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: alpha, the core against produce mode. stack 6 mm at 100 dB/m | 6 mm at 0,\n");
    QD_DIAG("# eps 5.24 - 0.63j, against one 12 mm sheet at 50 dB/m. moved_dB is what the column\n");
    QD_DIAG("# is worth on the stack's transmit gain\n");
    QD_DIAG("order,theta_n,|T_te|,|R_te|,d_T,d_R,moved_dB\n");
    for (int ord = 0; ord < 2; ++ord)
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto one = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, 67, 1u, A_THICK, 0, 0, 0, 0,
                                       0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto St = run(1, ot, A_ALPHA, A, th), Sr = run(0, ot, A_ALPHA, A, th);
            auto Pt = one(1), Pr = one(0);
            auto S0 = run(1, ot, A, B, th), Sa = run(1, ot, A, B_ALPHA, th);

            double d_T = std::max(std::abs(cpx(St.xprmat, 0, 0) - cpx(Pt.xprmat, 0, 0)),
                                  std::abs(cpx(St.xprmat, 0, 3) - cpx(Pt.xprmat, 0, 3)));
            double d_R = std::max(std::abs(cpx(Sr.xprmat, 0, 0) - cpx(Pr.xprmat, 0, 0)),
                                  std::abs(cpx(Sr.xprmat, 0, 3) - cpx(Pr.xprmat, 0, 3)));

            QD_DIAG("%s,%.1f,%.9f,%.9f,%.3e,%.3e,%+.6f\n", ord ? "2|1" : "1|2", thd,
                    std::abs(cpx(St.xprmat, 0, 3)), std::abs(cpx(Sr.xprmat, 0, 3)), d_T, d_R,
                    10.0 * std::log10((double)Sa.gain(0) / (double)S0.gain(0)));

            INFO("alpha against produce mode, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            REQUIRE(St.resolved_type[0] != 0);
            REQUIRE(Pt.resolved_type[0] != 0);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);

            // And the column is worth tenths of a dB on a stack, so neither side is passing on an
            // alpha that was never read.
            CHECK((double)Sa.gain(0) < 0.99 * (double)S0.gain(0));
        }

    // ------------------------------------------------------------------------------------------
    // The two columns the matrix cannot hold, on the second layer, both orders, both passes.
    // ------------------------------------------------------------------------------------------
    struct Decl
    {
        const char *name;
        int m2;
    };
    const std::vector<Decl> declined = {
        {"tf 0.25     ", B_TF},
        {"m 10        ", B_M},
    };

    QD_DIAG("\n# diag: ordered stack whose second sheet carries tf or m, resolved in the scattering\n");
    QD_DIAG("# form. gain_plain is the same stack without the column\n");
    QD_DIAG("column,order,theta_n,gain_plain_T,gain_T,gain_R,d_T,d_R,d_scalar,rtype_T,rtype_R\n");

    for (const auto &S : declined)
        for (int ord = 0; ord < 2; ++ord)
            for (double thd : {0.0, 45.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> ci(std::cos(th), 0.0);
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : A, second = ord ? A : S.m2;

                auto C0 = run(1, ot, A, B, th);
                auto Ct = run(1, ot, A, S.m2, th), Cr = run(0, ot, A, S.m2, th);
                auto St = run(4, ot, A, S.m2, th), Sr = run(3, ot, A, S.m2, th);

                std::vector<DiagLayer> lay = {lump_layer(pal[(size_t)first - 1], f_hz),
                                              lump_layer(pal[(size_t)second - 1], f_hz)};
                DiagSheet Xe = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, false);
                DiagSheet Xs = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, true);
                double d_T = std::max(std::abs(cpx(Ct.xprmat, 0, 3) - Xe.T_te), std::abs(cpx(Ct.xprmat, 0, 0) - Xe.T_tm));
                double d_R = std::max(std::abs(cpx(Cr.xprmat, 0, 3) - Xe.R_te), std::abs(cpx(Cr.xprmat, 0, 0) - Xe.R_tm));
                double d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Xs.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Xs.R_te));

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.9f,%.3e,%.3e,%.3e,%u,%u\n", S.name, ord ? "2|1" : "1|2", thd,
                        (double)C0.gain(0), (double)Ct.gain(0), (double)Cr.gain(0), d_T, d_R, d_S,
                        (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);

                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                REQUIRE(C0.resolved_type[0] != 0);
                CHECK((double)C0.gain(0) > 1.0e-3);

                // Resolved on both passes and in both modes, and on the reference. The mass law is
                // an acoustic column, so its EM row is the scattering form's two channels and no
                // statement about a use case.
                CHECK(Ct.resolved_type[0] != 0);
                CHECK(Cr.resolved_type[0] != 0);
                CHECK(St.resolved_type[0] != 0);
                CHECK((double)Ct.gain(0) > 1.0e-3);
                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(d_S < 1.0e-9);
            }
}

TEST_CASE("Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space")
{
    // The other consequence of a declining core, and the quieter one. A host layer whose
    // previous-medium word holds a sheet resolves its internal series against that sheet as a
    // LAYER, through qd_stack_resolve. Where the sheet carries m or tf the core returns false and
    // fresnel_r keeps the half-space reflection it computed first: no thickness, no structure,
    // nothing behind it. That is the treatment the core was added to replace, and the ray is not
    // killed, so nothing marks the row.
    //
    // What it costs is the gap between two columns of the backing table in the plan, and it is not
    // uniformly a loss. The half-space is exact when whatever is behind the sheet has the sheet's
    // own permittivity and is 1.560 dB off in |S| when it is air; the air-backed layer is the
    // other way round. So a declining sheet moves a coating with air behind it from 0 to 1.560 dB
    // of error, and a sheet with a matching solid behind it from 1.560 dB to 0. The first is the
    // motivating geometry.
    //
    // The fallback is a step: the decline tests the column for being non-zero and not for
    // mattering. alpha used to be a third such column, and 1e-6 dB/m of it - 6e-9 dB across this
    // sheet - moved the series by the full amount. It is carried by the core now, so the alpha
    // rows here are the guard on that: a negligible alpha has to leave the sheet a layer, on the
    // layer series to the same bound as the sheet without it.
    //
    // tf is carried now as well: the sheet is a layer, with tf by the pair rule on both of its
    // faces, and the row is asserted against the layer series built from the scattering
    // reference. The half-space column still records what the decline used to cost.
    //
    // A mass-law sheet keeps the half-space, and that is not a fallback. Its reflect port is its
    // near interface and nothing else, by the mass-law convention, so the half-space series IS
    // its series. Asserted against the half-space, to 1e-9, with the half-space and layer series
    // far enough apart that neither assertion could pass by accident.
    const double f_hz = FRQ;
    const double thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);

    const int HOST = 1;
    const std::vector<LumpMtl> pal = {
        {1.50, 0.40, 0.0},                                         // 1 HOST
        {5.24, 0.63, thk},                                         // 2 SHEET        resolves as a layer
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6},        // 3 SHEET alpha  1e-6 dB/m, a layer too
        {5.24, 0.63, thk, 0.0, 0.25},                              // 4 SHEET tf     0.25
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0},     // 5 SHEET m      10
        {1.5015, 0.40, thk},                                       // 6 NEAR         resolves as a layer
        {1.5015, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6},      // 7 NEAR alpha   1e-6 dB/m, a layer too
    };
    auto mtl = lump_palette(pal);
    const std::complex<double> e_h = lump_eps(pal[0], f_hz);

    struct Row
    {
        const char *name;
        int prev;
        bool declines;
    };
    const std::vector<Row> rows = {
        {"SHEET   plain        ", 2, false},
        {"SHEET   alpha 1e-6   ", 3, false},
        {"SHEET   tf 0.25      ", 4, false},
        {"SHEET   m 10         ", 5, true},
        {"LO_NEAR plain        ", 6, false},
        {"LO_NEAR alpha 1e-6   ", 7, false},
    };

    QD_DIAG("\n# diag: host layer bounded by a sheet the core declines on, host eps 1.50 - 0.40j.\n");
    QD_DIAG("# S_layer is the series with the sheet resolved as a layer backed by air, S_hs the\n");
    QD_DIAG("# series against the sheet as a half-space. cost_dB is |S_hs| against |S_layer|, which\n");
    QD_DIAG("# is what the decline moves the host's series by when air is what is behind the sheet\n");
    QD_DIAG("sheet,L_mm,theta_n,|S_applied|,|S_layer|,|S_hs|,d_layer,d_hs,cost_dB,rtype\n");

    double worst_cost = 0.0;
    for (const auto &R : rows)
        for (double L : {0.05, 0.10})
            for (double thd : {0.0, 20.0, 40.0, 50.0})
            {
                double theta_n = thd * QPI / 180.0;
                const std::complex<double> e_s = lump_eps(pal[(size_t)R.prev - 1], f_hz);

                auto C = make1<double>(mtl, 4, OT_EXIT, 1u, HOST, 0, R.prev, HOST, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
                C.run();

                std::complex<double> S_applied = (cpx(C.xprmat, 0, 0) / cpx(C.x_in, 0, 0)) /
                                                 diag_close(e_h, L, theta_n, f_hz);

                std::complex<double> cs(std::cos(theta_n), 0.0);
                std::complex<double> phi = diag_phi(e_h, L, theta_n, f_hz);
                std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> r_hs = diag_r_slabside(e_h, e_s, cs);
                // The layer, with the sheet's tf where it has one; at tf = 0 this is diag_cm_stack
                DiagLayer Ls{e_s, U, thk};
                Ls.tf = pal[(size_t)R.prev - 1].tf;
                DiagSheet B = diag_sc_stack(e_h, U, 0.0, {Ls}, e_air, U, 0.0, cs, f_hz,
                                            false, true, /*undisplaced*/ 0.0);

                std::complex<double> S_layer = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);
                std::complex<double> S_hs = 1.0 / (1.0 - r_ahead * r_hs * phi * phi);
                double d_layer = std::abs(S_applied - S_layer), d_hs = std::abs(S_applied - S_hs);
                double cost = 20.0 * std::log10(std::abs(S_hs) / std::abs(S_layer));
                if (R.declines)
                    worst_cost = std::max(worst_cost, std::abs(cost));

                QD_DIAG("%s,%.1f,%.1f,%.9f,%.9f,%.9f,%.3e,%.3e,%+.4f,%u\n", R.name, L * 1000.0, thd,
                        std::abs(S_applied), std::abs(S_layer), std::abs(S_hs), d_layer, d_hs, cost,
                        (unsigned)C.resolved_type[0]);

                INFO(R.name << ", L " << L * 1000.0 << " mm, theta_n " << thd);
                REQUIRE(C.resolved_type[0] != 0);
                REQUIRE(rt_bit(C.resolved_type[0], 2)); // the host series did resolve
                REQUIRE(std::isfinite(C.gain(0)));

                // The two series are far enough apart that landing on one excludes the other.
                CHECK(std::abs(S_hs - S_layer) > 1.0e-3);

                if (R.declines)
                    CHECK(d_hs < 1.0e-9); // a mass-law sheet: its reflect port is its near interface
                else
                    CHECK(d_layer < 1.0e-9); // a layer: no column, or one the core carries
            }
    QD_DIAG("# worst |cost| over the half-space rows %.4f dB\n", worst_cost);

    // The step is at least the plan's figure for this geometry, 1.560 dB over 0 to 50 deg at 50 mm.
    CHECK(worst_cost > 1.5);
}

// ================================================================================================
// Fix plan v14, items 2 and 3: tf per boundary by the pair rule, tf and the mass-law column in a
// stack, and the sheet under total reflection of its outer pair.
//
// Every case below carries control rows that hold before the source change - a neighbour without
// tf, a stack without the column, an angle below the critical one - so a row that fails is
// failing on the property and not on the fixture.
// ================================================================================================

TEST_CASE("Neighbour tf - each face of a sheet is owed by the pair of media that form it")
{
    // Item 3. Produce mode gave both faces of a sheet to the sheet and read the neighbours' tf
    // nowhere, while Material::interact_with combines the two media of a bare boundary through
    // apply_tf_pair. The rule asserted here is the second one applied to each face of the sheet
    // with its own neighbour, per channel:
    //
    //     near face   pair(tf_near, tf_sheet)        far face   pair(tf_sheet, tf_far)
    //
    // It reduces to the sheet's own tf, bit for bit, wherever the neighbour carries none, which is
    // every tf row that existed before; those are the control rows here.
    //
    // The limit. A sheet whose permittivity is its host's is not a boundary on the host side, so
    // an air-like sheet on a tf-bearing wall has to be the bare tf-bearing wall. That is asserted
    // against the bare boundary written from Material::interact_with's rule: complex in scalar
    // mode, complex per channel at normal incidence in EM, and on the TE/TM mean in EM at oblique
    // incidence. Per channel at oblique incidence the two cannot agree and are not asked to: the
    // bare boundary rescales by the TE/TM-averaged reflectance and the composite per channel, and
    // both conventions are settled.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    const double k0 = 2.0 * QPI * f_hz / C0;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SH = 1, WALL = 2, HOST = 3;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    struct Combo
    {
        double tf_s, tf_n;
    };
    const std::vector<Combo> combos = {{0.0, 0.0}, {0.25, 0.0}, {-0.5, 0.0}, {0.0, 0.5}, {0.0, -0.5},
                                       {0.25, 0.5}, {-0.25, 0.5}, {0.25, -0.5}, {-0.5, -0.25}};

    QD_DIAG("\n# diag: neighbour tf by the pair rule, 6 mm sheet eps 5.24 - 0.63j. entry is\n");
    QD_DIAG("# air | SH | WALL (eps 1.50 - 0.40j, tf_n), exit is HOST (eps 1.50, tf_n) | SH | air.\n");
    QD_DIAG("# nb_dB is what the neighbour's column is worth on the transmit gain, from the reference\n");
    QD_DIAG("stack,tf_s,tf_n,theta_n,d_T,d_R,d_scalar,nb_dB\n");

    double worst_ctl = 0.0, worst_nb = 0.0;
    for (const auto &K : combos)
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, K.tf_s}, // 1 SH
            {1.50, 0.40, 0.0, 0.0, K.tf_n}, // 2 WALL  lossy solid carrying the neighbour's tf
            {1.50, 0.00, 0.0, 0.0, K.tf_n}, // 3 HOST  lossless solid carrying the neighbour's tf
        };
        auto mtl = lump_palette(pal);
        auto E = [&](int idx)
        { return (idx == 0) ? std::complex<double>(1.0, 0.0) : lump_eps(pal[(size_t)idx - 1], f_hz); };
        bool control = (K.tf_n == 0.0);

        // Glued entry, complex on both ports: in air nothing else multiplies the composite.
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, SH, WALL, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            DiagLump Lp, Lo;
            Lp.tf = K.tf_s, Lp.tf_far = K.tf_n;
            Lo.tf = K.tf_s; // the sheet as sole owner
            DiagSheet D = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Ds = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, true, 1.0, 1.0, 1.0, Lp);
            DiagSheet Do = diag_sheet_composite(1.0, E(SH), E(WALL), thk, ct, f_hz, false, false, 1.0, 1.0, 1.0, Lo);

            auto Ct = run(1), Cr = run(0), St = run(4), Sr = run(3);
            double d_T = std::max(std::abs(TE(Ct.xprmat) - D.T_te), std::abs(TM(Ct.xprmat) - D.T_tm));
            double d_R = std::max(std::abs(TE(Cr.xprmat) - D.R_te), std::abs(TM(Cr.xprmat) - D.R_tm));
            double d_S = std::max(std::abs(cpx(St.xprmat, 0, 0) - Ds.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Ds.R_te));
            double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
            double gO = 0.5 * (std::norm(Do.T_te) + std::norm(Do.T_tm));
            double worst = std::max(d_T, std::max(d_R, d_S));
            (control ? worst_ctl : worst_nb) = std::max(control ? worst_ctl : worst_nb, worst);

            QD_DIAG("entry,%+.2f,%+.2f,%.1f,%.2e,%.2e,%.2e,%+.4f\n", K.tf_s, K.tf_n, thd, d_T, d_R, d_S,
                    (gT > 0.0 && gO > 0.0) ? 10.0 * std::log10(gT / gO) : -999.0);
            INFO("entry, tf_s " << K.tf_s << ", tf_n " << K.tf_n << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
            CHECK(d_S < 1.0e-9);
        }

        // Glued exit, reached from inside a lossless host. The host leg multiplies both ports by a
        // pure phase, so they are compared in power per channel.
        for (double thd : {0.0, 20.0, 45.0, 50.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SH, HOST, 0, HOST, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            DiagLump Lp, Lo;
            Lp.tf = K.tf_s, Lp.tf_near = K.tf_n;
            Lo.tf = K.tf_s;
            DiagSheet D = diag_sheet_composite(E(HOST), E(SH), 1.0, thk, ct, f_hz, true, false, 1.0, 1.0, 1.0, Lp);
            DiagSheet Do = diag_sheet_composite(E(HOST), E(SH), 1.0, thk, ct, f_hz, true, false, 1.0, 1.0, 1.0, Lo);

            auto Ct = run(1), Cr = run(0);
            double d_T = std::max(std::abs(std::norm(TE(Ct.xprmat)) - std::norm(D.T_te)),
                                  std::abs(std::norm(TM(Ct.xprmat)) - std::norm(D.T_tm)));
            double d_R = std::max(std::abs(std::norm(TE(Cr.xprmat)) - std::norm(D.R_te)),
                                  std::abs(std::norm(TM(Cr.xprmat)) - std::norm(D.R_tm)));
            double gT = 0.5 * (std::norm(D.T_te) + std::norm(D.T_tm));
            double gO = 0.5 * (std::norm(Do.T_te) + std::norm(Do.T_tm));
            (control ? worst_ctl : worst_nb) = std::max(control ? worst_ctl : worst_nb, std::max(d_T, d_R));

            QD_DIAG("exit ,%+.2f,%+.2f,%.1f,%.2e,%.2e,%.2e,%+.4f\n", K.tf_s, K.tf_n, thd, d_T, d_R, 0.0,
                    (gT > 0.0 && gO > 0.0) ? 10.0 * std::log10(gT / gO) : -999.0);
            INFO("exit, tf_s " << K.tf_s << ", tf_n " << K.tf_n << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
        }
    }
    QD_DIAG("# worst: control rows %.3e, rows with a tf-bearing neighbour %.3e\n", worst_ctl, worst_nb);

    // The limit: an air-like sheet on a tf-bearing wall against the bare tf-bearing wall.
    QD_DIAG("\n# diag: air | sheet eps 1 (6 mm) | wall eps 1.50 - 0.40j (tf) against the bare wall\n");
    QD_DIAG("wall_tf,theta_n,d_scalar_T,d_scalar_R,d_gain_T,d_gain_R,d_chan,moved_dB\n");
    for (double wtf : {0.5, -0.5})
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            const std::vector<LumpMtl> pal = {
                {1.00, 0.00, thk},            // 1 a sheet of the host's own permittivity
                {1.50, 0.40, 0.0, 0.0, wtf},  // 2 the wall
                {1.50, 0.40, 0.0, 0.0, 0.0},  // 3 the wall without its column
            };
            auto mtl = lump_palette(pal);
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype, int wall)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, 1, wall, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };

            // The bare boundary air | wall, from the rule Material::interact_with applies
            std::complex<double> ew = lump_eps(pal[1], f_hz), zw = std::sqrt(ew);
            std::complex<double> cw = std::sqrt(1.0 - (1.0 / ew) * (1.0 - ct * ct));
            std::complex<double> r_te = (ct - zw * cw) / (ct + zw * cw), r_tm = (zw * ct - cw) / (zw * ct + cw);
            std::complex<double> t_te = 2.0 * ct / (ct + zw * cw), t_tm = 2.0 * ct / (zw * ct + cw);
            double s_te = std::sqrt(std::real(zw * cw) / ct), s_tm = std::sqrt(std::real(std::conj(zw) * cw) / ct);
            double R0 = std::norm(r_te), Rp = diag_tf_pair(R0, 0.0, wtf);
            double Rbar = 0.5 * (std::norm(r_te) + std::norm(r_tm)), Rbar_p = diag_tf_pair(Rbar, 0.0, wtf);

            // The layer lies in the air in front of the wall, so the wall is on the mesh plane and
            // the reflect port is the bare wall's, phase included
            std::complex<double> back(1.0, 0.0);
            std::complex<double> Ts_ref = (1.0 + r_te) * s_te * std::sqrt((1.0 - Rp) / (1.0 - R0));
            std::complex<double> Rs_ref = std::polar(std::sqrt(Rp), std::arg(r_te)) * back;

            auto St = run(4, 2), Sr = run(3, 2), Ct = run(1, 2), Cr = run(0, 2), C0 = run(1, 3);
            double d_sT = std::abs(cpx(St.xprmat, 0, 0) - Ts_ref), d_sR = std::abs(cpx(Sr.xprmat, 0, 0) - Rs_ref);
            double d_gT = std::abs((double)Ct.gain(0) - (1.0 - Rbar_p)), d_gR = std::abs((double)Cr.gain(0) - Rbar_p);
            double d_ch = 0.0;
            if (thd == 0.0) // one reflectance, so the mean and the channel rule are the same statement
            {
                double k = std::sqrt((1.0 - Rp) / (1.0 - R0));
                d_ch = std::max(std::max(std::abs(TE(Ct.xprmat) - t_te * s_te * k), std::abs(TM(Ct.xprmat) - t_tm * s_tm * k)),
                                std::max(std::abs(TE(Cr.xprmat) - std::polar(std::sqrt(Rp), std::arg(r_te)) * back),
                                         std::abs(TM(Cr.xprmat) - std::polar(std::sqrt(Rp), std::arg(r_tm)) * back)));
            }
            QD_DIAG("%+.1f,%.1f,%.2e,%.2e,%.2e,%.2e,%.2e,%+.6f\n", wtf, thd, d_sT, d_sR, d_gT, d_gR, d_ch,
                    10.0 * std::log10((double)Ct.gain(0) / (double)C0.gain(0)));
            INFO("limit, wall tf " << wtf << ", theta_n " << thd);
            CHECK(d_sT < 1.0e-9);
            CHECK(d_sR < 1.0e-9);
            CHECK(d_gT < 1.0e-9);
            CHECK(d_gR < 1.0e-9);
            CHECK(d_ch < 1.0e-9);
        }
}

TEST_CASE("Stack with tf - the scattering form against its reference, produce mode and itself")
{
    // Item 2, the tf half. A characteristic matrix has no per-boundary r and t to rescale, so the
    // core declined on tf: an ordered stack was killed and a bounding sheet fell back to a
    // half-space. The scattering form carries it, and what is asserted is
    //
    //   the reference against the two it has to reduce to - diag_cm_stack at tf = 0, and
    //   diag_sheet_composite with tf at one layer;
    //   the library's ordered stack against that reference, both orders, both ports, scalar and EM;
    //   three statements that need no reference - a lossless stack conserves energy per channel
    //   at any tf, the transmit port is invariant under layer order, and tf = +1 / -1 on every
    //   layer closes the reflect / transmit port exactly;
    //   the core against produce mode - two 6 mm sheets of one permittivity and one tf > 0 are a
    //   single 12 mm sheet, because a matched boundary has no reflection to leak;
    //   a bounding sheet with tf resolved as a layer, with a negligible tf leaving the series
    //   where the tf-free layer puts it.
    const double f_hz = FRQ, thk = 0.006;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const uint8_t OT_STACK_FBS = 71, OT_STACK_SBS = 69;

    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ------------------------------------------------------------------------------------------
    // The reference against the forms it reduces to. No library call.
    // ------------------------------------------------------------------------------------------
    {
        const std::complex<double> eA(5.24, -0.63), eB(1.50, -0.40);
        double w_cm = 0.0, w_one = 0.0;
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
            for (int sc = 0; sc < 2; ++sc)
            {
                std::complex<double> ci(std::cos(thd * QPI / 180.0), 0.0);
                std::vector<DiagLayer> lay = {{eA, U, thk, std::complex<double>(0.3, -0.1), 0.0},
                                              {eB, U, thk, std::complex<double>(0.0, 0.0), 50.0}};
                for (int far = 0; far < 2; ++far)
                {
                    std::complex<double> ef = far ? eB : e_air;
                    DiagSheet X = diag_cm_stack(e_air, U, lay, ef, U, ci, f_hz, false, sc != 0);
                    DiagSheet Y = diag_sc_stack(e_air, U, 0.0, lay, ef, U, 0.0, ci, f_hz, false, sc != 0);
                    w_cm = std::max(w_cm, std::max(std::max(std::abs(X.T_te - Y.T_te), std::abs(X.T_tm - Y.T_tm)),
                                                   std::max(std::abs(X.R_te - Y.R_te), std::abs(X.R_tm - Y.R_tm))));
                }
                for (double tfs : {-0.5, 0.25, 1.0})
                    for (double tfn : {0.0, 0.5, -0.25})
                    {
                        DiagLump Lp;
                        Lp.tf = tfs, Lp.tf_far = tfn;
                        DiagSheet X = diag_sheet_composite(e_air, eA, eB, thk, std::real(ci), f_hz, false, sc != 0,
                                                           1.0, 1.0, 1.0, Lp);
                        std::vector<DiagLayer> one = {{eA, U, thk, std::complex<double>(0.0, 0.0), 0.0, tfs}};
                        DiagSheet Y = diag_sc_stack(e_air, U, 0.0, one, eB, U, tfn, ci, f_hz, false, sc != 0);
                        w_one = std::max(w_one, std::max(std::max(std::abs(X.T_te - Y.T_te), std::abs(X.T_tm - Y.T_tm)),
                                                         std::max(std::abs(X.R_te - Y.R_te), std::abs(X.R_tm - Y.R_tm))));
                    }
            }
        QD_DIAG("\n# diag: scattering-form reference. against diag_cm_stack at tf = 0, two layers with a\n");
        QD_DIAG("# resonance and alpha: %.3e. against diag_sheet_composite with tf, one layer: %.3e\n", w_cm, w_one);
        CHECK(w_cm < 1.0e-13);
        CHECK(w_one < 1.0e-14);
    }

    // ------------------------------------------------------------------------------------------
    // The library's ordered stack against the reference
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: ordered stack with tf, air | 6 mm | 6 mm | air. A eps 5.24 - 0.63j,\n");
    QD_DIAG("# B eps 1.50 - 0.40j; AL and BL are their lossless twins\n");
    QD_DIAG("stack,tf,order,theta_n,ref_|T_te|,ref_|R_te|,d_scalar,d_EM_T,d_EM_R,rtype_T,rtype_R\n");

    double worst_ctl = 0.0, worst_tf = 0.0, worst_E = 0.0, worst_ord = 0.0;
    for (double tf : {0.0, -1.0, -0.5, 0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk},          // 1 A
            {1.50, 0.40, thk, 0.0, tf}, // 2 B_TF
            {5.24, 0.63, thk, 0.0, tf}, // 3 A_TF
            {5.24, 0.00, thk, 0.0, tf}, // 4 AL_TF
            {1.50, 0.00, thk, 0.0, tf}, // 5 BL_TF
        };
        auto mtl = lump_palette(pal);
        auto run = [&](int itype, uint8_t otype, int m1, int m2, double th)
        {
            auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
            C.run();
            return C;
        };
        struct St
        {
            const char *name;
            int m1, m2;
            bool all; // every layer carries the tf
        };
        const std::vector<St> stacks = {{"A | B_TF   ", 1, 2, false}, {"A_TF | B_TF", 3, 2, true}};

        for (const auto &S : stacks)
            for (int ord = 0; ord < 2; ++ord)
                for (double thd : {0.0, 20.0, 45.0, 70.0})
                {
                    double th = thd * QPI / 180.0;
                    std::complex<double> ci(std::cos(th), 0.0);
                    uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                    int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;
                    std::vector<DiagLayer> lay = {lump_layer(pal[(size_t)first - 1], f_hz),
                                                  lump_layer(pal[(size_t)second - 1], f_hz)};
                    DiagSheet Xs = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, true);
                    DiagSheet Xe = diag_sc_stack(e_air, U, 0.0, lay, e_air, U, 0.0, ci, f_hz, false, false);

                    auto St_ = run(4, ot, S.m1, S.m2, th), Sr = run(3, ot, S.m1, S.m2, th);
                    auto Et = run(1, ot, S.m1, S.m2, th), Er = run(0, ot, S.m1, S.m2, th);
                    double d_S = std::max(std::abs(cpx(St_.xprmat, 0, 0) - Xs.T_te), std::abs(cpx(Sr.xprmat, 0, 0) - Xs.R_te));
                    double d_T = std::max(std::abs(TE(Et.xprmat) - Xe.T_te), std::abs(TM(Et.xprmat) - Xe.T_tm));
                    double d_R = std::max(std::abs(TE(Er.xprmat) - Xe.R_te), std::abs(TM(Er.xprmat) - Xe.R_tm));
                    double w = std::max(d_S, std::max(d_T, d_R));
                    ((tf == 0.0) ? worst_ctl : worst_tf) = std::max((tf == 0.0) ? worst_ctl : worst_tf, w);

                    QD_DIAG("%s,%+.2f,%s,%.1f,%.9f,%.9f,%.2e,%.2e,%.2e,%u,%u\n", S.name, tf, ord ? "2|1" : "1|2", thd,
                            std::abs(Xe.T_te), std::abs(Xe.R_te), d_S, d_T, d_R,
                            (unsigned)Et.resolved_type[0], (unsigned)Er.resolved_type[0]);
                    INFO(S.name << ", tf " << tf << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                    CHECK(Et.resolved_type[0] != 0); // resolved, not killed
                    CHECK(Er.resolved_type[0] != 0);
                    CHECK(d_S < 1.0e-9);
                    CHECK(d_T < 1.0e-9);
                    CHECK(d_R < 1.0e-9);

                    // The end points, exact by construction: a factor sqrt(0) on every term of the port
                    if (tf == -1.0)
                        CHECK((double)Et.gain(0) == 0.0);
                    if (tf == 1.0 && S.all)
                        CHECK((double)Er.gain(0) == 0.0);
                }

        // Lossless stack: energy per channel, and the transmit port under layer order
        for (double thd : {0.0, 30.0, 60.0, 80.0})
        {
            double th = thd * QPI / 180.0;
            auto T1 = run(1, OT_STACK_FBS, 4, 5, th), R1 = run(0, OT_STACK_FBS, 4, 5, th);
            auto T2 = run(1, OT_STACK_SBS, 4, 5, th), R2 = run(0, OT_STACK_SBS, 4, 5, th);
            double e_te = std::norm(TE(T1.xprmat)) + std::norm(TE(R1.xprmat));
            double e_tm = std::norm(TM(T1.xprmat)) + std::norm(TM(R1.xprmat));
            double d_o = std::max(std::abs(TE(T1.xprmat) - TE(T2.xprmat)), std::abs(TM(T1.xprmat) - TM(T2.xprmat)));
            double d_r = std::abs(TE(R1.xprmat) - TE(R2.xprmat));
            if (tf != 0.0)
                worst_E = std::max(worst_E, std::max(std::abs(e_te - 1.0), std::abs(e_tm - 1.0))),
                worst_ord = std::max(worst_ord, d_o);
            INFO("lossless stack, tf " << tf << ", theta_n " << thd);
            CHECK(std::abs(e_te - 1.0) < 1.0e-12);
            CHECK(std::abs(e_tm - 1.0) < 1.0e-12);
            CHECK(d_o < 1.0e-14);
            if (tf > -1.0 && tf < 1.0) // at tf = +1 both orders reflect nothing
                CHECK(d_r > 1.0e-3);
        }
    }
    QD_DIAG("# worst: tf = 0 rows %.3e, tf rows %.3e, energy of a lossless stack %.3e, order %.3e\n",
            worst_ctl, worst_tf, worst_E, worst_ord);

    // ------------------------------------------------------------------------------------------
    // The core against produce mode. Two sheets of one permittivity and one tf > 0 share a
    // matched boundary, R0 = 0, which has nothing to leak; the stack is the 12 mm sheet.
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: tf, the core against produce mode. 6 mm | 6 mm of eps 5.24 - 0.63j, one tf,\n");
    QD_DIAG("# against one 12 mm sheet with that tf\ntf,theta_n,|T_te|,|R_te|,d_T,d_R,d_scalar\n");
    for (double tf : {0.25, 0.5, 1.0})
    {
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk, 0.0, tf},                // 1
            {5.24, 0.63, thk, 0.0, tf, 0.0, 1.0, 0.0}, // 2 the same sheet; resQ alone is inert and keeps it distinct
            {5.24, 0.63, 2.0 * thk, 0.0, tf},          // 3 12 mm
        };
        auto mtl = lump_palette(pal);
        for (double thd : {0.0, 20.0, 45.0, 70.0, 85.0})
        {
            double th = thd * QPI / 180.0;
            auto stk = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_STACK_FBS, 2u, 1, 2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto one = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, 67, 1u, 3, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
                C.run();
                return C;
            };
            auto St = stk(1), Sr = stk(0), Pt = one(1), Pr = one(0);
            auto Ss = stk(4), Sq = stk(3), Ps = one(4), Pq = one(3);
            double d_T = std::max(std::abs(TE(St.xprmat) - TE(Pt.xprmat)), std::abs(TM(St.xprmat) - TM(Pt.xprmat)));
            double d_R = std::max(std::abs(TE(Sr.xprmat) - TE(Pr.xprmat)), std::abs(TM(Sr.xprmat) - TM(Pr.xprmat)));
            double d_S = std::max(std::abs(cpx(Ss.xprmat, 0, 0) - cpx(Ps.xprmat, 0, 0)),
                                  std::abs(cpx(Sq.xprmat, 0, 0) - cpx(Pq.xprmat, 0, 0)));
            QD_DIAG("%+.2f,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e\n", tf, thd, std::abs(TE(Pt.xprmat)), std::abs(TE(Pr.xprmat)),
                    d_T, d_R, d_S);
            INFO("tf against produce mode, tf " << tf << ", theta_n " << thd);
            CHECK(St.resolved_type[0] != 0);
            CHECK(d_T < 1.0e-9);
            CHECK(d_R < 1.0e-9);
            CHECK(d_S < 1.0e-9);
        }
    }

    // ------------------------------------------------------------------------------------------
    // A bounding sheet with tf, resolved as a layer. The host's own tf belongs to the host | sheet
    // face under the same rule, not to the layer's composite reflectance.
    // ------------------------------------------------------------------------------------------
    {
        const std::vector<LumpMtl> pal = {
            {1.50, 0.40, 0.0},               // 1 HOST
            {1.50, 0.40, 0.0, 0.0, 0.3},     // 2 HOST with tf 0.3
            {5.24, 0.63, thk},               // 3 SHEET
            {5.24, 0.63, thk, 0.0, 1.0e-9},  // 4 SHEET tf 1e-9
            {5.24, 0.63, thk, 0.0, 0.25},    // 5 SHEET tf 0.25
            {5.24, 0.63, thk, 0.0, -0.25},   // 6 SHEET tf -0.25
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int host, prev;
        };
        const std::vector<Row> rows = {
            {"plain host | SHEET        ", 1, 3}, {"plain host | SHEET tf 1e-9", 1, 4},
            {"plain host | SHEET tf 0.25", 1, 5}, {"plain host | SHEET tf -.25", 1, 6},
            {"host tf .3 | SHEET        ", 2, 3}, {"host tf .3 | SHEET tf 0.25", 2, 5},
        };
        QD_DIAG("\n# diag: host layer bounded by a sheet carrying tf, host eps 1.50 - 0.40j. S_ref has the\n");
        QD_DIAG("# sheet as a layer backed by air with tf by the pair rule; d_plain is against the same\n");
        QD_DIAG("# layer with no tf anywhere\nrow,L_mm,theta_n,|S_applied|,|S_ref|,d_ref,d_plain\n");
        for (const auto &R : rows)
            for (double Lh : {0.05, 0.10})
                for (double thd : {0.0, 20.0, 40.0, 50.0})
                {
                    double theta_n = thd * QPI / 180.0;
                    const LumpMtl &H = pal[(size_t)R.host - 1], &Sm = pal[(size_t)R.prev - 1];
                    const std::complex<double> e_h = lump_eps(H, f_hz), e_s = lump_eps(Sm, f_hz);

                    auto C = make1<double>(mtl, 4, OT_EXIT, 1u, R.host, 0, R.prev, R.host, 0,
                                           Lh, 0.5, 0.5, QPI / 2.0 - theta_n, /*always resolve*/ 0.0);
                    C.run();
                    std::complex<double> S_applied = (cpx(C.xprmat, 0, 0) / cpx(C.x_in, 0, 0)) /
                                                     diag_close(e_h, Lh, theta_n, f_hz);

                    std::complex<double> cs(std::cos(theta_n), 0.0);
                    std::complex<double> phi = diag_phi(e_h, Lh, theta_n, f_hz);
                    std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs), r_ahead0 = r_ahead;
                    diag_tf_port(r_ahead, H.tf, 0.0);
                    DiagSheet B = diag_sc_stack(e_h, U, H.tf, {lump_layer(Sm, f_hz)}, e_air, U, 0.0, cs, f_hz,
                                                false, true, /*undisplaced*/ 0.0);
                    DiagSheet B0 = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0);
                    std::complex<double> S_ref = 1.0 / (1.0 - r_ahead * B.R_te * phi * phi);
                    std::complex<double> S_plain = 1.0 / (1.0 - r_ahead0 * B0.R_te * phi * phi);
                    double d_ref = std::abs(S_applied - S_ref), d_plain = std::abs(S_applied - S_plain);

                    QD_DIAG("%s,%.1f,%.1f,%.9f,%.9f,%.3e,%.3e\n", R.name, Lh * 1000.0, thd,
                            std::abs(S_applied), std::abs(S_ref), d_ref, d_plain);
                    INFO(R.name << ", L " << Lh * 1000.0 << " mm, theta_n " << thd);
                    CHECK(rt_bit(C.resolved_type[0], 2));
                    CHECK(d_ref < 1.0e-9);
                    if (Sm.tf == 1.0e-9 && H.tf == 0.0) // no step: a negligible tf is a negligible change
                        CHECK(d_plain < 1.0e-8);
                }
    }
}

TEST_CASE("Stack with a mass-law layer - resolved by segments, not killed")
{
    // Item 2, the mass-law half. A mass-law layer is not a coherent layer and is not made one. It
    // ends the coherent segment in front of it as a half-space of its own admittance - which is
    // what its eta and mu are calibrated for - contributes one real factor, and starts the next
    // segment. Nothing returns across it. With no coherent layer at all that is produce mode's
    // mass-law route for a single sheet, which is the first thing asserted, on the reference and
    // through the library, and it holds before the source change.
    //
    // There is no exact reference for a mass-law layer inside a coherent stack, so beyond the
    // convention written a second time the assertions are the ones that need none:
    //
    //   the transmit port is invariant under layer order;
    //   the reflect port does not read m, and not what lies behind the mass-law layer;
    //   the mass term is one real factor on the transmit port, so two slopes differ by exactly
    //   the difference of their terms;
    //   a lossless-permittivity stack reflects and transmits no more than it received;
    //   an air-like coherent sheet in front of a mass-law sheet is that mass-law sheet.
    //
    // Scalar only: the mass law is an acoustic column. Two mass-law layers with no coherent layer
    // between them are bonded and pay as one plate, X0 summed; that rule has its own case.
    const double f_hz = FRQ, thk = 0.006;
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const uint8_t OT_STACK_FBS = 71, OT_STACK_SBS = 69;

    const int A = 1, M = 2, M2 = 3, AIRS = 4, AL = 5, ML = 6, M_TF = 7, MB = 8, B = 9;
    const std::vector<LumpMtl> pal = {
        {5.24, 0.63, thk},                                                // 1 A      coherent
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 2 M      m 20, argument 6
        {1.50, 0.40, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.001},     // 3 M2     m 10
        {1.00, 0.00, thk},                                                // 4 AIRS   an air-like coherent sheet
        {4.00, 0.00, thk},                                                // 5 AL     lossless coherent
        {2.25, 0.00, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 6 ML     lossless, m 20
        {1.50, 0.40, thk, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},     // 7 M_TF   m 20 and tf 0.3
        {5.24, 0.63, thk, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.001},     // 8 MB     a second mass-law sheet
        {1.50, 0.40, thk},                                                // 9 B      M without its column
    };
    auto mtl = lump_palette(pal);
    auto Lay = [&](int idx)
    { return lump_layer(pal[(size_t)idx - 1], f_hz); };
    auto stk = [&](int itype, uint8_t otype, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, otype, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };
    auto one = [&](int itype, int m1, double th)
    {
        auto C = make1<double>(mtl, itype, 67, 1u, m1, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // The reference at one mass-law layer against produce mode's own route. Holds today.
    QD_DIAG("\n# diag: mass-law reference at one layer against a single mass-law sheet, eps 1.50 - 0.40j,\n");
    QD_DIAG("# 6 mm, m 20, argument 6\ntheta_n,TL_dB,|T|,|R|,d_T,d_R\n");
    for (double thd : {0.0, 20.0, 45.0, 70.0})
    {
        double th = thd * QPI / 180.0;
        std::complex<double> ci(std::cos(th), 0.0);
        DiagSheet X = diag_sc_stack(e_air, U, 0.0, {Lay(M)}, e_air, U, 0.0, ci, f_hz, false, true);
        auto Ct = one(4, M, th), Cr = one(3, M, th);
        double d_T = std::abs(cpx(Ct.xprmat, 0, 0) - X.T_te), d_R = std::abs(cpx(Cr.xprmat, 0, 0) - X.R_te);
        QD_DIAG("%.1f,%.4f,%.9f,%.9f,%.3e,%.3e\n", thd, diag_mass_TL(20.0, 0.001, f_hz, thk, std::cos(th)),
                std::abs(X.T_te), std::abs(X.R_te), d_T, d_R);
        INFO("single mass-law sheet, theta_n " << thd);
        CHECK(d_T < 1.0e-9);
        CHECK(d_R < 1.0e-9);
    }

    struct St
    {
        const char *name;
        int m1, m2;
    };
    const std::vector<St> stacks = {
        {"A | B (control) ", A, B}, {"A | M           ", A, M}, {"A | M_TF        ", A, M_TF},
        {"AL | ML lossless", AL, ML}, {"M | MB both mass", M, MB},
    };

    QD_DIAG("\n# diag: ordered stack with a mass-law layer, air | 6 mm | 6 mm | air, scalar.\n");
    QD_DIAG("# energy is |R|^2 + |T|^2 of the library's two ports\n");
    QD_DIAG("stack,order,theta_n,ref_|T|,ref_|R|,d_T,d_R,d_order,energy,rtype_T,rtype_R\n");
    for (const auto &S : stacks)
        for (double thd : {0.0, 20.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> ci(std::cos(th), 0.0);
            std::complex<double> T_ord[2];
            for (int ord = 0; ord < 2; ++ord)
            {
                uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
                int first = ord ? S.m2 : S.m1, second = ord ? S.m1 : S.m2;
                DiagSheet X = diag_sc_stack(e_air, U, 0.0, {Lay(first), Lay(second)}, e_air, U, 0.0, ci, f_hz,
                                            false, true, -1.0, 0.0, 0.0, std::cos(th));
                auto Ct = stk(4, ot, S.m1, S.m2, th), Cr = stk(3, ot, S.m1, S.m2, th);
                std::complex<double> t = cpx(Ct.xprmat, 0, 0), r = cpx(Cr.xprmat, 0, 0);
                T_ord[ord] = t;
                double d_T = std::abs(t - X.T_te), d_R = std::abs(r - X.R_te);
                double d_o = ord ? std::abs(T_ord[0] - T_ord[1]) : 0.0;
                double en = std::norm(t) + std::norm(r);

                QD_DIAG("%s,%s,%.1f,%.9f,%.9f,%.3e,%.3e,%.3e,%.9f,%u,%u\n", S.name, ord ? "2|1" : "1|2", thd,
                        std::abs(X.T_te), std::abs(X.R_te), d_T, d_R, d_o, en,
                        (unsigned)Ct.resolved_type[0], (unsigned)Cr.resolved_type[0]);
                INFO(S.name << ", order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
                CHECK(Ct.resolved_type[0] != 0); // survives
                CHECK(Cr.resolved_type[0] != 0);
                CHECK(d_T < 1.0e-9);
                CHECK(d_R < 1.0e-9);
                CHECK(std::abs(t) > 1.0e-4); // and carries energy
                if (ord)
                    CHECK(d_o < 1.0e-14); // reciprocity of the transmit port
                if (S.m1 == AL)
                    CHECK(en <= 1.0 + 1.0e-12); // lossless permittivities: nothing is created
            }
        }

    // Statements about the mass term alone
    QD_DIAG("\n# diag: the mass term in a stack. m 20 against m 10 on the same layer, and an air-like\n");
    QD_DIAG("# sheet in front of the mass-law sheet against that sheet on its own\n");
    QD_DIAG("theta_n,order,dTL_dB,paid_dB,d_R_m,d_null_T,d_null_R\n");
    for (double thd : {0.0, 20.0, 45.0, 70.0})
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto T20 = stk(4, ot, A, M, th), T10 = stk(4, ot, A, M2, th);
            auto R20 = stk(3, ot, A, M, th), R10 = stk(3, ot, A, M2, th);
            double dTL = diag_mass_TL(20.0, 0.001, f_hz, thk, ct) - diag_mass_TL(10.0, 0.001, f_hz, thk, ct);
            double paid = -10.0 * std::log10((double)T20.gain(0) / (double)T10.gain(0));
            double d_Rm = std::abs(cpx(R20.xprmat, 0, 0) - cpx(R10.xprmat, 0, 0));

            // air-like coherent sheet | M against M alone. The stack is centred on the mesh plane
            // over its coherent thickness, so M stands half of thk behind the plane when the
            // air-like sheet is in front and half of thk in front of it otherwise.
            auto Nt = stk(4, ot, AIRS, M, th), Nr = stk(3, ot, AIRS, M, th);
            auto Pt = one(4, M, th), Pr = one(3, M, th);
            std::complex<double> back = std::exp(std::complex<double>(0.0, (ord ? 1.0 : -1.0) * k0 * thk * ct));
            double d_nT = std::abs(cpx(Nt.xprmat, 0, 0) - cpx(Pt.xprmat, 0, 0));
            double d_nR = std::abs(cpx(Nr.xprmat, 0, 0) - cpx(Pr.xprmat, 0, 0) * back);

            QD_DIAG("%.1f,%s,%.6f,%.6f,%.3e,%.3e,%.3e\n", thd, ord ? "2|1" : "1|2", dTL, paid, d_Rm, d_nT, d_nR);
            INFO("mass term, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(std::abs(paid - dTL) < 1.0e-9); // one real factor on the transmit port
            CHECK(dTL > 1.0);                     // and worth dB, so the row is not passing on a zero term
            CHECK(d_Rm < 1.0e-15);                // the reflect port never reads m
            CHECK(std::abs(cpx(R20.xprmat, 0, 0)) > 1.0e-3);
            CHECK(d_nT < 1.0e-12);
            CHECK(d_nR < 1.0e-12);
        }
}

// Transmit port of a sheet whose outer pair is past its critical angle, where the wave entered the
// sheet as an ordinary Fresnel wave: the near crossing as Material::interact_with gives it, the far
// face as the flat tf leak at the Snell cosine the wave has in the sheet, the sheet's one-way
// factor, and the internal series the reflect port already carries. The displaced medium is air on
// every row that calls this, taken out at the incidence cosine because the leak leaves undeviated.
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

    // near crossing, flux-normalized, tf on the TE/TM mean in EM and on the one channel in scalar mode
    std::complex<double> r_te = (zn * ct - zs * cs) / (zn * ct + zs * cs);
    std::complex<double> r_tm = scalar ? r_te : (zs * ct - zn * cs) / (zs * ct + zn * cs);
    std::complex<double> t_te = 2.0 * zn * ct / (zn * ct + zs * cs);
    std::complex<double> t_tm = scalar ? t_te : 2.0 * zn * ct / (zs * ct + zn * cs);
    double R0 = scalar ? std::norm(r_te) : 0.5 * (std::norm(r_te) + std::norm(r_tm));
    double sc = std::sqrt((1.0 - diag_tf_pair(R0, tf_near, tf_slab)) / (1.0 - R0));
    double fl_te = std::sqrt(std::real(zs * cs) / std::real(zn * ct));
    double fl_tm = scalar ? fl_te : std::sqrt(std::real(std::conj(zs) * cs) / std::real(std::conj(zn) * ct));
    std::complex<double> a_te = t_te * fl_te * sc, a_tm = t_tm * fl_tm * sc;

    // far face from inside the sheet: closed, so the flat leak; real in EM, at arg(1 + R) in scalar mode
    std::complex<double> f_te = (zs * cs - zf * cf) / (zs * cs + zf * cf);
    std::complex<double> f_tm = scalar ? f_te : (zf * cs - zs * cf) / (zf * cs + zs * cf);
    double leak = 1.0 - diag_tf_pair(1.0, tf_slab, tf_far);
    std::complex<double> b = std::polar(std::sqrt(leak), scalar ? std::arg(1.0 + f_te) : 0.0);

    std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e_slab) * thk * cs);
    std::complex<double> phi_t = phi / std::exp(std::complex<double>(0.0, -k0 * thk * ct));

    auto den = [&](std::complex<double> r, std::complex<double> f)
    {
        double Rf0 = std::norm(f);
        std::complex<double> rn = std::polar(std::sqrt(diag_tf_pair(std::norm(r), tf_slab, tf_near)), std::arg(-r));
        std::complex<double> rf = std::polar(std::sqrt(diag_tf_pair((Rf0 > 1.0) ? 1.0 : Rf0, tf_slab, tf_far)), std::arg(f));
        return 1.0 - rn * rf * phi * phi;
    };
    DiagLeak D;
    std::complex<double> d_te = den(r_te, f_te), d_tm = den(r_tm, f_tm);
    D.T_te = a_te * b * phi_t / d_te, D.T_tm = a_tm * b * phi_t / d_tm;
    D.S_te = 1.0 / std::norm(d_te), D.S_tm = 1.0 / std::norm(d_tm);
    return D;
}

TEST_CASE("Total reflection of the outer pair - the sheet stays in the result")
{
    // Where the near | far pair is past its critical angle SHEET_RESOLVE discarded the sheet and
    // wrote the bare near | far boundary: a mirror on the reflect pass, the pair's tf leak on the
    // transmit pass. Total reflection of the outer pair says one thing only, that no Fresnel wave
    // propagates in the far medium. It says nothing about the sheet.
    //
    // In EM this is a coated solid at a steep in-medium angle. Under the acoustic mapping,
    // eps = (Z_air / Z)^2, it is every sheet on a rigid wall at every angle: air | concrete has a
    // critical angle of 0.003 deg.
    //
    // Reflect port. The wave enters the sheet, turns around at the far face and comes back, which
    // is the ordinary composite with a totally reflecting far face - an absorber on a rigid
    // backing. Asserted against diag_sheet_composite, whose reflect port never needed a forward
    // port in the far medium.
    //
    // Transmit port. The Fresnel part is zero; what crosses is the tf leak, and a leak that
    // crosses the sheet pays what the sheet charges. The statement is the one a traced solid of
    // the same material would make, three events in a row at the incidence angle the leak keeps:
    //
    //     near | sheet      Material::interact_with's transmission, the sheet's interface gain
    //     the sheet         its one-way loss; for a mass-law sheet the mass term
    //     sheet | far       Material::interact_with's transmission, the far medium's interface gain
    //
    // with no internal series. Each crossing keeps the phase interact_with gives it, a coherent
    // sheet contributes its one-way phase, a mass-law sheet none, and the displaced host is taken
    // out at the incidence cosine because a leak is undeviated. Asserted in power and complex,
    // on lossless permittivities.
    //
    // Each event is evaluated at the cosine the wave has there: the incidence cosine after a first
    // crossing that was itself a leak, the Snell cosine after an ordinary one. A leak inside a
    // mass-law sheet is not judged from inside it at the far face and is charged what the other
    // direction charges, and a wave that entered by an ordinary crossing carries the sheet's
    // internal series. Both have their own case; the rows here follow the same rules.
    const double f_hz = FRQ;
    const double L = 0.10;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };

    // ------------------------------------------------------------------------------------------
    // Reflect port, acoustic: a lossy sheet on a rigid wall, air | sheet | rigid
    // ------------------------------------------------------------------------------------------
    {
        const std::vector<LumpMtl> pal = {
            {0.31, 0.20, 0.010},                 // 1 a porous layer, rarer than air
            {2.50, 0.80, 0.010},                 // 2 a dense lossy layer
            {2.401e-9, 0.0, 0.0},                // 3 rigid wall
            {2.401e-9, 0.0, 0.0, 0.0, 0.3},      // 4 rigid wall leaking, tf 0.3
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int sh, wall;
        };
        const std::vector<Row> rows = {{"porous | rigid      ", 1, 3}, {"dense  | rigid      ", 2, 3},
                                       {"porous | rigid tf .3", 1, 4}, {"dense  | rigid tf .3", 2, 4}};
        QD_DIAG("\n# diag: reflect port of a 10 mm lossy sheet on a rigid wall, scalar, air | sheet | rigid.\n");
        QD_DIAG("# alpha_ref is the absorbed fraction 1 - |R|^2 of the reference; 0 deg is below the\n");
        QD_DIAG("# critical angle and is the control row\nstack,theta_n,|R|^2,ref_|R|^2,alpha_ref,d_R,R0_far\n");
        for (const auto &R : rows)
            for (double thd : {0.0, 20.0, 45.0, 70.0})
            {
                double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
                const LumpMtl &Sm = pal[(size_t)R.sh - 1], &Wm = pal[(size_t)R.wall - 1];
                auto C = make1<double>(mtl, 3, OT_GLUED_ENTRY, 2u, R.sh, R.wall, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                DiagLump Lp;
                Lp.tf_far = Wm.tf;
                DiagSheet D = diag_sheet_composite(1.0, lump_eps(Sm, f_hz), lump_eps(Wm, f_hz), Sm.thk, ct, f_hz,
                                                   false, true, 1.0, 1.0, 1.0, Lp);
                // raw slab-side reflectance of the far face: the known clamp in apply_tf engages above 1
                std::complex<double> zs = std::sqrt(lump_eps(Sm, f_hz)), zf = std::sqrt(lump_eps(Wm, f_hz));
                double R0f = std::norm((zs * D.c_slab - zf * D.c_far) / (zs * D.c_slab + zf * D.c_far));
                double d_R = std::abs(cpx(C.xprmat, 0, 0) - D.R_te);

                QD_DIAG("%s,%.1f,%.9f,%.9f,%.6f,%.3e,%.6f\n", R.name, thd, (double)C.gain(0), std::norm(D.R_te),
                        1.0 - std::norm(D.R_te), d_R, R0f);
                INFO(R.name << ", theta_n " << thd);
                CHECK(R0f <= 1.0);                   // the row is outside the clamp
                CHECK(std::norm(D.R_te) < 0.99);     // and the sheet absorbs, so a bare mirror cannot pass
                CHECK(d_R < 1.0e-9);
            }
    }

    // ------------------------------------------------------------------------------------------
    // Reflect port, EM and scalar: a coated solid past its critical angle, HOST | SH | air.
    // The TM channel of a lossy sheet is inside the clamp recorded in section 5 and is printed.
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.006;
        const std::vector<LumpMtl> pal = {
            {5.24, 0.63, thk}, // 1 SH
            {1.50, 0.00, 0.0}, // 2 HOST, lossless: the host leg is a pure phase
        };
        auto mtl = lump_palette(pal);
        const std::complex<double> e_s = lump_eps(pal[0], f_hz), e_h(1.50, 0.0);
        QD_DIAG("\n# diag: reflect port of a coated solid, eps 1.50 | 6 mm eps 5.24 - 0.63j | air, critical\n");
        QD_DIAG("# angle 54.7 deg. loss_dB is the reference TE reflectance, 0 for a bare mirror\n");
        QD_DIAG("theta_n,|R_te|^2,ref_|R_te|^2,loss_dB,d_te,d_scalar,|R_tm|^2,ref_|R_tm|^2\n");
        for (double thd : {40.0, 50.0, 60.0, 70.0, 80.0})
        {
            double theta_n = thd * QPI / 180.0, ct = std::cos(theta_n);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - theta_n, 1.5);
                C.run();
                return C;
            };
            auto Cr = run(0), Sr = run(3);
            DiagSheet D = diag_sheet_composite(e_h, e_s, 1.0, thk, ct, f_hz, true, false);
            double d_te = std::abs(std::norm(TE(Cr.xprmat)) - std::norm(D.R_te));
            double d_sc = std::abs(std::norm(cpx(Sr.xprmat, 0, 0)) - std::norm(D.R_te));
            QD_DIAG("%.1f,%.9f,%.9f,%+.4f,%.3e,%.3e,%.9f,%.9f\n", thd, std::norm(TE(Cr.xprmat)), std::norm(D.R_te),
                    10.0 * std::log10(std::norm(D.R_te)), d_te, d_sc, std::norm(TM(Cr.xprmat)), std::norm(D.R_tm));
            INFO("coated solid, theta_n " << thd);
            CHECK(d_te < 1.0e-9);
            CHECK(d_sc < 1.0e-9);
            // TM past the critical angle is inside the clamp of section 5: apply_tf caps a
            // slab-side reflectance above one, measured at 4.0e-2 in amplitude. Guarded at that
            // size until produce mode is routed through the core, which does not clamp.
            CHECK(std::abs(std::abs(TM(Cr.xprmat)) - std::abs(D.R_tm)) < ((thd > 54.7) ? 5.0e-2 : 1.0e-9));
        }
    }

    // ------------------------------------------------------------------------------------------
    // Both ports, acoustic: a wood panel on a concrete wall, both mass-law, both with tf
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.020;
        const double tf_w = 0.2, tf_c = 0.1, att_w = 1.0, att_c = 2.0;
        const std::vector<LumpMtl> pal = {
            {2.1609e-8, 0.0, thk, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 WOOD   m 20, argument 20
            {2.401e-9, 0.0, 0.0, att_c, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},  // 2 CONC   as a wall
            {2.401e-9, 0.0, 0.0, att_c, tf_c},                                   // 3 CONC0  without its mass law
        };
        auto mtl = lump_palette(pal);
        const double e_w = 2.1609e-8, e_c = 2.401e-9;
        const double G_w = std::pow(10.0, -0.1 * att_w), G_c = std::pow(10.0, -0.1 * att_c);

        // Transmitted power fraction of one lossless crossing e1 -> e2 at incidence angle th, by
        // the rule of Material::interact_with: the flat leak 1 - R' past the critical angle, and
        // 1 - R' below it, with R' the pair rule on the Fresnel reflectance.
        auto cross = [](double e1, double e2, double tf1, double tf2, double th)
        {
            double s = std::sin(th), c = std::cos(th), q = (e1 / e2) * s * s;
            if (std::sqrt(e1 / e2) * s >= 1.0)
                return 1.0 - diag_tf_pair(1.0, tf1, tf2);
            double z1 = std::sqrt(e1), z2 = std::sqrt(e2), c2 = std::sqrt(1.0 - q);
            double r = (z1 * c - z2 * c2) / (z1 * c + z2 * c2);
            return 1.0 - diag_tf_pair(r * r, tf1, tf2);
        };

        QD_DIAG("\n# diag: wood panel (20 mm, tf 0.2, att 1 dB, m 20) on concrete (tf 0.1, att 2 dB), scalar.\n");
        QD_DIAG("# entry is air | wood | concrete, exit is concrete | wood | air. *_dB are power gains;\n");
        QD_DIAG("# TL_w is the panel's mass term\n");
        QD_DIAG("theta_n,TL_w,entry_T_dB,ref_entry_T_dB,entry_R,ref_entry_R,exit_T_dB,ref_exit_T_dB,exit_R,ref_exit_R\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            double TL = diag_mass_TL(20.0, 0.001, f_hz, thk, ct), gM = std::pow(10.0, -0.1 * TL);
            auto dB = [](double g)
            { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

            // Entry, from the room
            auto Et = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Er = make1<double>(mtl, 3, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Et.run(), Er.run();
            // Past the critical angle of air | wood the wave in the panel is a leak, and the panel
            // is mass-law, so the wood | concrete face is the crossing from the concrete side at
            // the same angle. At normal incidence everything is an ordinary crossing.
            bool leak = std::sqrt(1.0 / e_w) * std::sin(th) >= 1.0;
            double x1 = cross(1.0, e_w, 0.0, tf_w, th);
            double x2 = leak ? cross(e_c, e_w, tf_c, tf_w, th) : cross(e_w, e_c, tf_w, tf_c, th);
            double ref_eT = x1 * G_w * gM * x2 * G_c, ref_eR = 1.0 - x1;

            // Phase of the entry leak. Each crossing keeps the phase interact_with gives it,
            // arg(1 + R) of its own Fresnel reflection with the decaying root past the critical
            // angle, and a mass-law sheet adds none of its own.
            auto arg_T = [](double e1, double e2, double th_)
            {
                double s_ = std::sin(th_), c_ = std::cos(th_), a_ = 1.0 - (e1 / e2) * s_ * s_;
                std::complex<double> c2 = (a_ < 0.0) ? std::complex<double>(0.0, -std::sqrt(-a_))
                                                     : std::complex<double>(std::sqrt(a_), 0.0);
                std::complex<double> z1 = std::sqrt(e1), z2 = std::sqrt(e2);
                return std::arg(1.0 + (z1 * c_ - z2 * c2) / (z1 * c_ + z2 * c2));
            };
            std::complex<double> ref_eT_cx = std::polar(std::sqrt(ref_eT), arg_T(1.0, e_w, th) +
                                                                              (leak ? arg_T(e_c, e_w, th) : arg_T(e_w, e_c, th)));
            CHECK(std::abs(cpx(Et.xprmat, 0, 0) - ref_eT_cx) < 1.0e-9 * std::sqrt(ref_eT));

            // Exit, from inside the wall. Concrete without its mass law, so the leg it closes is a
            // pure phase and the written power is the crossing alone. The outer pair is not past
            // its critical angle in this direction, so this is produce mode's mass-law route with
            // the pair rule on the concrete | wood face.
            auto Xt = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 1, 3, 0, 3, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Xr = make1<double>(mtl, 3, OT_GLUED_EXIT, 2u, 1, 3, 0, 3, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Xt.run(), Xr.run();
            double s2 = 1.0 - ct * ct;
            double zc = std::sqrt(e_c), zw = std::sqrt(e_w);
            double cw = std::sqrt(1.0 - (e_c / e_w) * s2), ca = std::sqrt(1.0 - e_c * s2);
            double rn = (zc * ct - zw * cw) / (zc * ct + zw * cw), rf = (zw * cw - ca) / (zw * cw + ca);
            double Rn = diag_tf_pair(rn * rn, tf_c, tf_w), Rf = diag_tf_pair(rf * rf, tf_w, 0.0);
            double ref_xT = (1.0 - Rn) * G_w * gM * (1.0 - Rf), ref_xR = Rn;

            QD_DIAG("%.1f,%.4f,%.4f,%.4f,%.9f,%.9f,%.4f,%.4f,%.9f,%.9f\n", thd, TL, dB((double)Et.gain(0)), dB(ref_eT),
                    (double)Er.gain(0), ref_eR, dB((double)Xt.gain(0)), dB(ref_xT), (double)Xr.gain(0), ref_xR);
            INFO("wood on concrete, theta_n " << thd);
            CHECK(std::abs((double)Et.gain(0) - ref_eT) < 1.0e-9 * ref_eT + 1.0e-15);
            CHECK(std::abs((double)Er.gain(0) - ref_eR) < 1.0e-9);
            CHECK(std::abs((double)Xt.gain(0) - ref_xT) < 1.0e-9 * ref_xT + 1.0e-15);
            CHECK(std::abs((double)Xr.gain(0) - ref_xR) < 1.0e-9);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Transmit leak through a coherent sheet, EM and scalar: HOST | SH (tf) | air past critical
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {
            {5.24, 0.00, thk, 0.0, tf_s}, // 1 lossless sheet with a leak
            {1.50, 0.00, 0.0},            // 2 HOST
        };
        auto mtl = lump_palette(pal);
        QD_DIAG("\n# diag: tf leak through a coated solid past its critical angle, eps 1.50 | 6 mm eps 5.24\n");
        QD_DIAG("# (tf 0.5) | air, lossless throughout\ntheta_n,gain_EM,ref_EM,gain_scalar,ref_scalar,d_complex\n");
        for (double thd : {60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), St = run(4);
            // The wave enters the sheet by an ordinary crossing, so it meets the closed far face at
            // the Snell cosine and carries the sheet's internal series; diag_leak_series is that
            // statement. The host leg the crossing closes is a pure phase, removed with diag_close.
            DiagLeak De = diag_leak_series(1.5, 5.24, 1.0, 0.0, tf_s, 0.0, thk, ct, f_hz, false);
            DiagLeak Ds = diag_leak_series(1.5, 5.24, 1.0, 0.0, tf_s, 0.0, thk, ct, f_hz, true);
            double ref_em = 0.5 * (std::norm(De.T_te) + std::norm(De.T_tm)), ref_sc = std::norm(Ds.T_te);
            std::complex<double> close = diag_close(std::complex<double>(1.5, 0.0), L, th, f_hz);
            double d_cx = std::max(std::max(std::abs(TE(Ct.xprmat) / close - De.T_te), std::abs(TM(Ct.xprmat) / close - De.T_tm)),
                                   std::abs(cpx(St.xprmat, 0, 0) / close - Ds.T_te));

            QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.3e\n", thd, (double)Ct.gain(0), ref_em, (double)St.gain(0), ref_sc, d_cx);
            INFO("leak through a coated solid, theta_n " << thd);
            CHECK(std::abs((double)Ct.gain(0) - ref_em) < 1.0e-9);
            CHECK(std::abs((double)St.gain(0) - ref_sc) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
        }
    }
}

TEST_CASE("Reciprocity - glued entry against glued exit")
{
    // One three-medium stack resolved from both sides: air | sheet | solid entered from the air at
    // theta_ext, and solid | sheet | air left from inside the solid along the direction a tracer
    // holds after refracting in at that same theta_ext. The two are the same pair of waves
    // reversed, so in per-channel normal-flux powers they owe the same number.
    //
    // Lossless solid. Exact: the raw amplitudes of a reciprocal stack differ by the ratio of the two
    // outer tilted admittances, and the two flux conversions are that ratio and its inverse. This is
    // the row that guards the conversion - with it removed the two would sit 1.76 dB apart at normal
    // incidence.
    //
    // Lossy solid. Not exact, and the residual has a closed form. With eta = z c for TE and z / c
    // for TM in the solid, exit over entry is |eta|^2 / Re(eta)^2 = sec^2(arg eta) per channel: the
    // incident and reflected waves in a lossy medium interfere, so splitting its flux into two
    // independent parts is not exact, and the flux form is the one that composes across a chain.
    // That is the settled energy convention, so the residual is asserted against its closed form
    // and recorded against a bound, never asserted to zero. The normal-incidence row is called out:
    // no cosine enters there and the residual is sec^2(arg z) alone, 0.073946 dB for eps 1.50 - 0.40j.
    //
    // The host leg the exit closes is isolated by running it over L and 2 L and taking the ratio,
    // which holds whatever cosine the composite is built at.
    const double f_hz = FRQ;
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const int SHEET = 3, LO = 5, LO_CLEAN = 6;

    struct Row
    {
        const char *name;
        int host;
        bool lossless;
    };
    const std::vector<Row> rows = {{"lossless eps 1.50       ", LO_CLEAN, true}, {"lossy eps 1.50 - 0.40j  ", LO, false}};

    QD_DIAG("\n# diag: glued entry / exit reciprocity, air | 6 mm eps 5.24 - 0.63j | solid, 1 GHz.\n");
    QD_DIAG("# x_over_e_* is exit over entry in dB; ref_* is the closed form sec^2(arg eta) of the solid,\n");
    QD_DIAG("# zero for a lossless one. d_* is the relative distance of the measured ratio from it\n");
    QD_DIAG("solid,theta_ext,P_entry_scalar,P_exit_scalar,x_over_e_scalar_dB,ref_te_dB,ref_tm_dB,"
            "x_over_e_gain_dB,ref_gain_dB,d_scalar,d_gain\n");

    for (const auto &R : rows)
        for (double thd : {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0;
            std::complex<double> e_h = diag_eps(R.host, f_hz);
            double ct = diag_traced_ct(e_h, th);              // the real projection the tracer holds
            std::complex<double> cx = diag_ct_exact(e_h, th); // the exact complex in-solid cosine

            auto enter = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, SHEET, R.host, 0, 0, 0,
                                       L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto leave = [&](int itype, double host_len)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SHEET, R.host, 0, R.host, 0,
                                       host_len, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return C;
            };
            auto Es = enter(4), Ee = enter(1);
            auto Xs = leave(4, L), Xs2 = leave(4, 2.0 * L), Xe = leave(1, L);
            REQUIRE(Es.resolved_type[0] != 0);
            REQUIRE(Xs.resolved_type[0] != 0);

            double host2 = std::norm(cpx(Xs2.xprmat, 0, 0) / cpx(Xs.xprmat, 0, 0)); // host leg over L, power
            double Pe_s = std::norm(cpx(Es.xprmat, 0, 0)), Px_s = std::norm(cpx(Xs.xprmat, 0, 0)) / host2;
            double Pe_te = std::norm(cpx(Ee.xprmat, 0, 3)), Pe_tm = std::norm(cpx(Ee.xprmat, 0, 0));
            double Ge = (double)Ee.gain(0), Gx = (double)Xe.gain(0) / host2;

            std::complex<double> z = std::sqrt(e_h);
            double c_te = std::cos(std::arg(z * cx)), c_tm = std::cos(std::arg(z / cx));
            double k_te = 1.0 / (c_te * c_te), k_tm = 1.0 / (c_tm * c_tm);
            double ref_G = 0.5 * (Pe_te * k_te + Pe_tm * k_tm);
            double d_s = std::abs(Px_s - Pe_s * k_te) / Pe_s, d_g = std::abs(Gx - ref_G) / Ge;
            auto dB = [](double g)
            { return 10.0 * std::log10(g); };

            QD_DIAG("%s,%.1f,%.9f,%.9f,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f,%.3e,%.3e\n", R.name, thd, Pe_s, Px_s,
                    dB(Px_s / Pe_s), dB(k_te), dB(k_tm), dB(Gx / Ge), dB(ref_G / Ge), d_s, d_g);

            INFO(R.name << ", theta_ext " << thd);
            CHECK(std::abs(Ge - 0.5 * (Pe_te + Pe_tm)) < 1.0e-12); // the entry gain is the channel mean
            CHECK(d_s < 1.0e-9);
            CHECK(d_g < 1.0e-9);
            if (R.lossless)
            {
                CHECK(k_te == 1.0); // the closed form is identically 1, so the two rows above read zero
                CHECK(k_tm == 1.0);
            }
            else
            {
                // The flux-splitting residual, recorded against a bound. TE grows towards grazing
                // and TM falls, so the bound is on the larger channel and on the mean.
                CHECK(dB(Px_s / Pe_s) > 0.0);
                CHECK(dB(Px_s / Pe_s) < 0.50);
                CHECK(std::abs(dB(Gx / Ge)) < 0.25);
                if (thd == 0.0) // no cosine enters: sec^2(arg z) and nothing else
                {
                    CHECK(std::abs(dB(Px_s / Pe_s) - 0.073946) < 1.0e-6);
                    CHECK(std::abs(dB(Gx / Ge) - 0.073946) < 1.0e-6);
                }
            }
        }
}

TEST_CASE("Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids")
{
    // A tf leak keeps its direction, so the reverse of a leak that crossed a face at theta is the
    // crossing of that face from the other side at the same theta. Wood (eps 2.1609e-8) on concrete
    // (eps 2.401e-9) has a Snell ratio of 3: from the wood the face is past its critical angle
    // beyond 19.47 deg and passes only the flat leak, from the concrete it never is and is an
    // ordinary crossing. Same face, same two media, same angle, a factor of three.
    //
    // The angle 19.47 deg is formed from the index of a mass-law material, which is calibrated for
    // surface impedance only and is not a propagation index. So a face is not judged from inside a
    // mass-law medium by a wave that is held undeviated: it is charged what the other direction
    // charges at that angle, with the interface gain of the medium entered.
    //
    // Owed between the two directions, and taken out here: the interface gain of the medium newly
    // entered. Recorded, not owed and not removed: on the sheet rows the air | wood face is the
    // settled flat leak on entry (R0 = 1) and a Fresnel crossing on exit (R0 = 1 - 4 z cos), worth
    // 0.010 dB. Everything else has to vanish.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9;
    const double tf_w = 0.2, tf_c = 0.1, att_w = 1.0, att_c = 2.0;
    const double G_w = std::pow(10.0, -0.1 * att_w), G_c = std::pow(10.0, -0.1 * att_c);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    // ------------------------------------------------------------------------------------------
    // As a sheet: air | wood | concrete against concrete | wood | air
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.020, L = 0.10;
        const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
        const std::vector<LumpMtl> pal = {
            {e_w, 0.0, thk, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001}, // 1 WOOD   panel, m 20
            {e_c, 0.0, 0.0, att_c, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001}, // 2 CONC   as a wall
            {e_c, 0.0, 0.0, att_c, tf_c},                                  // 3 CONC0  without its mass law
        };
        auto mtl = lump_palette(pal);
        QD_DIAG("\n# diag: leak reciprocity as a sheet, wood panel (20 mm, tf 0.2, att 1 dB, m 20) on concrete\n");
        QD_DIAG("# (tf 0.1, att 2 dB), scalar. e_minus_x is entry minus exit with the wall's interface gain\n");
        QD_DIAG("# taken out; near_face is the flat first face against the Fresnel exit, the part recorded\n");
        QD_DIAG("theta_n,entry_T_dB,exit_T_dB,e_minus_x_dB,near_face_dB,rest_dB\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th), s2 = 1.0 - ct * ct;
            auto Et = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Xt = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 1, 3, 0, 3, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Et.run(), Xt.run();
            double d = dB((double)Et.gain(0) / G_c) - dB((double)Xt.gain(0));

            // The air | wood face: flat on entry past its critical angle, Fresnel on exit at the
            // cosine the exit holds in the wood. At normal incidence both directions are Fresnel.
            double zw = std::sqrt(e_w), cw = std::sqrt(1.0 - (e_c / e_w) * s2), ca = std::sqrt(1.0 - e_c * s2);
            double r = (zw * cw - ca) / (zw * cw + ca);
            double near_face = (thd > 0.0) ? dB((1.0 - diag_tf_pair(1.0, 0.0, tf_w)) / (1.0 - diag_tf_pair(r * r, tf_w, 0.0))) : 0.0;

            QD_DIAG("%.1f,%.4f,%.4f,%+.6f,%+.6f,%+.3e\n", thd, dB((double)Et.gain(0)), dB((double)Xt.gain(0)), d, near_face, d - near_face);
            INFO("wood on concrete as a sheet, theta_n " << thd);
            CHECK(std::abs(d) < 0.02);           // the stated bound, the recorded near-face term inside it
            CHECK(std::abs(d - near_face) < 1.0e-9);  // and nothing else: the wood | concrete face is one number
        }
    }

    // ------------------------------------------------------------------------------------------
    // As two traced solids: the wood | concrete face through ray_mesh_interact, both directions
    // ------------------------------------------------------------------------------------------
    {
        // Two adjacent cubes sharing the wall at x = 1. The coefficient of an undeviated crossing
        // is the one ray_mesh_interact writes; ray_state_update replaces it only where the tracer
        // supplied a refracted direction, which a leak does not have.
        const std::vector<LumpMtl> pal = {
            {e_w, 0.0, 0.0, att_w, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0}, // 1 WOOD as a solid
            {e_c, 0.0, 0.0, att_c, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0}, // 2 CONC as a solid
        };
        auto mtl = lump_palette(pal);
        arma::mat mesh = quadriga_lib::cube<double>();
        arma::mat cube2 = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
        mesh = arma::join_cols(mesh, cube2);
        arma::uvec mtl_ind(mesh.n_rows);
        for (arma::uword i = 0; i < mesh.n_rows; ++i)
            mtl_ind(i) = (i < mesh.n_rows / 2) ? 1 : 2;

        auto face = [&](bool wood_to_conc, double th)
        {
            double dy = 0.2 * std::tan(th), xa = wood_to_conc ? 0.9 : 1.1, xb = wood_to_conc ? 1.1 : 0.9;
            arma::mat orig = {{xa, -0.4, 0.1}}, dest = {{xb, -0.4 + dy, 0.1}};
            arma::mat fbs, sbs;
            arma::u32_vec fbs_ind, sbs_ind;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
            arma::mat origN, destN, xprmatN;
            arma::vec gainN;
            quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                            &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                            &gainN, &xprmatN);
            REQUIRE(gainN.n_elem == 1);
            return (double)gainN(0);
        };

        QD_DIAG("\n# diag: leak reciprocity as two traced solids, the wood | concrete face through\n");
        QD_DIAG("# ray_mesh_interact, scalar. Each direction with the interface gain of the medium it enters\n");
        QD_DIAG("# taken out; w2c_minus_c2w has to vanish\n");
        QD_DIAG("theta_n,wood_to_conc,conc_to_wood,w2c_minus_c2w_dB\n");
        for (double thd : {0.0, 10.0, 30.0, 45.0, 70.0})
        {
            double th = thd * QPI / 180.0;
            double wc = face(true, th) / G_c, cw = face(false, th) / G_w;
            QD_DIAG("%.1f,%.9f,%.9f,%+.6f\n", thd, wc, cw, dB(wc) - dB(cw));
            INFO("wood | concrete as two traced solids, theta_n " << thd);
            CHECK(cw > 0.5);                               // the face is open from the concrete at every angle
            CHECK(std::abs(dB(wc) - dB(cw)) < 1.0e-9);     // exact: one crossing, read from either side
        }
    }

    // ------------------------------------------------------------------------------------------
    // Guard: a wave the tracer refracted into a mass-law solid is a Fresnel wave and is left alone
    // ------------------------------------------------------------------------------------------
    {
        const double L = 0.10;
        const std::vector<LumpMtl> pal = {
            {4.0, 0.0, 0.0},                                             // 1 plain solid
            {4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0},    // 2 the same with a mass law, term zero at this path
        };
        auto mtl = lump_palette(pal);
        for (double thd : {30.0, 60.0})
        {
            double th = thd * QPI / 180.0, ct = diag_traced_ct(std::complex<double>(4.0, 0.0), th);
            auto run = [&](int M)
            {
                auto C = make1<double>(mtl, 4, OT_EXIT, 1, M, 0, 0, M, 0, L, 2.0, 0.5, QPI / 2.0 - th, 1.5);
                C.has_path_dir = true;
                C.path_dir_prev.zeros(1, 3);
                C.path_dir_prev(0, 0) = ct;
                C.path_dir_prev(0, 1) = std::sqrt(1.0 - ct * ct);
                C.run();
                return (double)C.gain(0);
            };
            INFO("refracted wave in a mass-law solid, theta_ext " << thd);
            double g_plain = run(1), g_mass = run(2);
            CHECK(g_plain > 0.1);
            CHECK(std::abs(g_plain - g_mass) < 1.0e-12);
        }
    }
}

TEST_CASE("Leak chain - each face at the cosine the wave has and the series on a closed far face")
{
    // The transmit port of a sheet whose outer pair is past its critical angle is three events. Two
    // things about them, and one about what follows them.
    //
    // The cosine. A wave that entered the sheet by an ordinary Fresnel crossing was refracted and
    // meets the far face at the Snell cosine; by transitivity that face is then closed whenever the
    // outer pair is, and passes the flat tf leak or nothing. Only a wave that entered as a leak
    // kept the incidence cosine, and its one-way factor through the sheet is taken at that cosine
    // too, not at the Snell one, which for a leak is evanescent.
    //
    // The series. A sheet has no interior the tracer could follow a bounce into, so what the far
    // face reflects has to be summed here or it is lost. Where the wave is a Fresnel wave inside
    // the sheet the series is the one the reflect port already carries. A wave that entered as a
    // leak has no phase the media define for its bounces and keeps the three events alone, as the
    // mass-law route does at every angle.
    const double f_hz = FRQ;
    const double L = 0.10;
    const double k0 = 2.0 * QPI * f_hz / C0;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    auto TM = [](const arma::mat &X)
    { return cpx(X, 0, 0); };
    auto TE = [](const arma::mat &X)
    { return cpx(X, 0, 3); };
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    // ------------------------------------------------------------------------------------------
    // No tf anywhere: a closed outer pair transmits nothing, whatever the sheet is
    // ------------------------------------------------------------------------------------------
    {
        auto mtl = diag_palette<double>(); // 2 = eps 5.24 solid, 8 = eps 1.50 sheet, both lossless
        QD_DIAG("\n# diag: closed outer pair with no tf, eps 5.24 | 6 mm eps 1.50 | air. The pair closes at\n");
        QD_DIAG("# 25.9 deg, the host | sheet face at 32.3 deg; the exact transmit port is zero throughout\n");
        QD_DIAG("theta_n,gain_EM,gain_scalar\n");
        for (double thd : {26.0, 28.0, 30.0, 32.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            auto Ce = make1<double>(mtl, 1, OT_GLUED_EXIT, 2u, 8, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            auto Cs = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, 8, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            Ce.run(), Cs.run();
            QD_DIAG("%.1f,%.9f,%.9f\n", thd, (double)Ce.gain(0), (double)Cs.gain(0));
            INFO("closed pair, no tf, EM palette, theta_n " << thd);
            CHECK((double)Ce.gain(0) == 0.0);
            CHECK((double)Cs.gain(0) == 0.0);
        }

        const std::vector<LumpMtl> pal = {{0.31, 0.0, 0.010}, {0.05, 0.0, 0.0}};
        auto mta = lump_palette(pal);
        QD_DIAG("\n# diag: the same in the acoustic direction, air | 10 mm eps 0.31 | eps 0.05, scalar. The pair\n");
        QD_DIAG("# closes at 12.9 deg; the sheet | far face judged at the incidence cosine only at 23.7 deg\n");
        QD_DIAG("theta_n,gain\n");
        for (double thd : {13.0, 15.0, 20.0, 23.0, 30.0})
        {
            double th = thd * QPI / 180.0;
            auto C = make1<double>(mta, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.run();
            QD_DIAG("%.1f,%.9f\n", thd, (double)C.gain(0));
            INFO("closed pair, no tf, acoustic palette, theta_n " << thd);
            CHECK((double)C.gain(0) == 0.0);
        }
    }

    // ------------------------------------------------------------------------------------------
    // A coherent sheet entered by a leak: three events, the one-way factor at the incidence cosine
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.010, e_s = 0.31, e_c = 2.401e-9, tf_s = 0.3, tf_c = 0.1;
        const std::vector<LumpMtl> pal = {{e_s, 0.0, thk, 0.0, tf_s}, {e_c, 0.0, 0.0, 0.0, tf_c}};
        auto mtl = lump_palette(pal);
        QD_DIAG("\n# diag: coherent lossless sheet entered by a leak, air | 10 mm eps 0.31 (tf 0.3) | rigid\n");
        QD_DIAG("# (tf 0.1), scalar. ref is the two flat leaks; evan_dB is the Snell-cosine decay a leak\n");
        QD_DIAG("# does not take\ntheta_n,gain,ref,gain_minus_ref_dB,evan_dB,d_complex\n");
        for (double thd : {40.0, 45.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th), s2 = 1.0 - ct * ct;
            auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.run();
            double ref = (1.0 - diag_tf_pair(1.0, 0.0, tf_s)) * (1.0 - diag_tf_pair(1.0, tf_s, tf_c));
            double zs = std::sqrt(e_s), zc = std::sqrt(e_c);
            std::complex<double> c2n(0.0, -std::sqrt(s2 / e_s - 1.0)), c2f(0.0, -std::sqrt((e_s / e_c) * s2 - 1.0));
            double arg_n = std::arg(1.0 + (ct - zs * c2n) / (ct + zs * c2n));
            double arg_f = std::arg(1.0 + (zs * ct - zc * c2f) / (zs * ct + zc * c2f));
            std::complex<double> ref_cx = std::polar(std::sqrt(ref), arg_n + arg_f - k0 * (zs - 1.0) * thk * ct);
            double evan = dB(std::exp(-2.0 * k0 * thk * std::sqrt(s2 - e_s)));
            double d_cx = std::abs(cpx(C.xprmat, 0, 0) - ref_cx);
            QD_DIAG("%.1f,%.9f,%.9f,%+.4f,%+.4f,%.3e\n", thd, (double)C.gain(0), ref, dB((double)C.gain(0) / ref), evan, d_cx);
            INFO("coherent sheet entered by a leak, theta_n " << thd);
            CHECK(std::abs((double)C.gain(0) - ref) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
        }
    }

    // ------------------------------------------------------------------------------------------
    // The series on a closed far face, EM and scalar: eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air
    // ------------------------------------------------------------------------------------------
    {
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {{5.24, 0.00, thk, 0.0, tf_s}, {1.50, 0.00, 0.0}};
        auto mtl = lump_palette(pal);
        const std::complex<double> e_h(1.50, 0.0), e_s(5.24, 0.0), e_a(1.0, 0.0);
        QD_DIAG("\n# diag: leak through a coated solid with its series, eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air,\n");
        QD_DIAG("# lossless. S_dB is the series factor of the scalar channel; energy is |R|^2 + |T|^2 in scalar\n");
        QD_DIAG("# mode, which a lossless stack owes exactly\n");
        QD_DIAG("theta_n,gain_EM,ref_EM,gain_scalar,ref_scalar,S_dB,d_complex,energy_scalar\n");
        for (double thd : {60.0, 70.0, 80.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto run = [&](int itype)
            {
                auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                return C;
            };
            auto Ct = run(1), St = run(4), Sr = run(3);
            DiagLeak De = diag_leak_series(e_h, e_s, e_a, 0.0, tf_s, 0.0, thk, ct, f_hz, false);
            DiagLeak Ds = diag_leak_series(e_h, e_s, e_a, 0.0, tf_s, 0.0, thk, ct, f_hz, true);
            std::complex<double> close = diag_close(e_h, L, th, f_hz); // the host leg, a pure phase
            double ref_em = 0.5 * (std::norm(De.T_te) + std::norm(De.T_tm)), ref_sc = std::norm(Ds.T_te);
            double d_cx = std::max(std::max(std::abs(TE(Ct.xprmat) / close - De.T_te), std::abs(TM(Ct.xprmat) / close - De.T_tm)),
                                   std::abs(cpx(St.xprmat, 0, 0) / close - Ds.T_te));
            double energy = (double)St.gain(0) + (double)Sr.gain(0);
            QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%+.4f,%.3e,%.12f\n", thd, (double)Ct.gain(0), ref_em,
                    (double)St.gain(0), ref_sc, dB(Ds.S_te), d_cx, energy);
            INFO("leak through a coated solid with its series, theta_n " << thd);
            CHECK(std::abs((double)Ct.gain(0) - ref_em) < 1.0e-9);
            CHECK(std::abs((double)St.gain(0) - ref_sc) < 1.0e-9);
            CHECK(d_cx < 1.0e-9);
            CHECK(std::abs(energy - 1.0) < 1.0e-12); // needs no reference
        }
    }

    // ------------------------------------------------------------------------------------------
    // The series on a closed far face, acoustic: an absorber without tf on a leaking rigid wall
    // ------------------------------------------------------------------------------------------
    {
        const double tf_c = 0.3;
        const std::vector<LumpMtl> pal = {
            {2.50, 0.80, 0.010},            // 1 a dense lossy layer
            {0.31, 0.20, 0.010},            // 2 a porous layer, rarer than air
            {2.401e-9, 0.0, 0.0, 0.0, tf_c}, // 3 rigid wall leaking, tf 0.3
        };
        auto mtl = lump_palette(pal);
        struct Row
        {
            const char *name;
            int sh;
            double lim; // the air | sheet face must be an ordinary crossing on the row
        };
        const std::vector<Row> rows = {{"dense  | rigid tf .3", 1, 70.0}, {"porous | rigid tf .3", 2, 20.0}};
        QD_DIAG("\n# diag: leak into a rigid wall (tf 0.3) through a 10 mm absorber without tf, scalar,\n");
        QD_DIAG("# air | sheet | rigid. S_dB is the series factor, the part a three-event chain leaves out\n");
        QD_DIAG("stack,theta_n,T_dB,ref_T_dB,S_dB,d_complex\n");
        for (const auto &R : rows)
            for (double thd : {10.0, 20.0, 45.0, 70.0})
            {
                if (thd > R.lim)
                    continue;
                double th = thd * QPI / 180.0, ct = std::cos(th);
                const LumpMtl &Sm = pal[(size_t)R.sh - 1];
                auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, R.sh, 3, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                C.run();
                DiagLeak D = diag_leak_series(1.0, lump_eps(Sm, f_hz), lump_eps(pal[2], f_hz), 0.0, 0.0, tf_c,
                                              Sm.thk, ct, f_hz, true);
                double d_cx = std::abs(cpx(C.xprmat, 0, 0) - D.T_te);
                QD_DIAG("%s,%.1f,%.4f,%.4f,%+.4f,%.3e\n", R.name, thd, dB((double)C.gain(0)), dB(std::norm(D.T_te)), dB(D.S_te), d_cx);
                INFO(R.name << ", theta_n " << thd);
                CHECK(d_cx < 1.0e-9 * std::abs(D.T_te) + 1.0e-12);
            }
    }

    // ------------------------------------------------------------------------------------------
    // The step at the critical angle of the outer pair, recorded
    // ------------------------------------------------------------------------------------------
    {
        // Just below, the composite; just above, the chain. With the series on both sides and the
        // far face at one cosine on both sides, the scalar step closes. In EM a per-channel step
        // stays: the near crossing takes tf on the TE/TM mean in the chain and per channel in the
        // composite, which is the settled channel rule.
        const double thk = 0.006, tf_s = 0.5;
        const std::vector<LumpMtl> pal = {{5.24, 0.00, thk, 0.0, tf_s}, {1.50, 0.00, 0.0}};
        auto mtl = lump_palette(pal);
        const double thc = std::asin(std::sqrt(1.0 / 1.5)) * 180.0 / QPI, dth = 1.0e-5;
        auto run = [&](int itype, double thd)
        {
            auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, 1, 2, 0, 2, 0, L, 0.5, 0.5, QPI / 2.0 - thd * QPI / 180.0, 1.5);
            C.run();
            return C;
        };
        auto Sb = run(4, thc - dth), Sa = run(4, thc + dth), Eb = run(1, thc - dth), Ea = run(1, thc + dth);
        double st_s = dB((double)Sa.gain(0)) - dB((double)Sb.gain(0));
        double st_te = dB(std::norm(TE(Ea.xprmat))) - dB(std::norm(TE(Eb.xprmat)));
        double st_tm = dB(std::norm(TM(Ea.xprmat))) - dB(std::norm(TM(Eb.xprmat)));
        QD_DIAG("\n# diag: step from the composite to the leak chain at the critical angle of the outer pair,\n");
        QD_DIAG("# above minus below in dB, +- 1e-5 deg\nstack,theta_c,step_scalar_dB,step_te_dB,step_tm_dB\n");
        QD_DIAG("eps 1.50 | 6 mm eps 5.24 (tf 0.5) | air,%.4f,%+.4f,%+.4f,%+.4f\n", thc, st_s, st_te, st_tm);
        CHECK(std::abs(st_s) < 0.01);
        CHECK(std::abs(st_te) < 0.5);
        CHECK(std::abs(st_tm) < 0.5);

        // Wood on concrete. The outer pair closes at asin(z_conc) = 0.0028 deg and the air | wood
        // face at asin(z_wood) = 0.0084 deg. Between them the wave in the wood is a refracted one
        // near grazing and the far face is closed for it, so the first angle is continuous. At the
        // second the wave becomes a leak and its angle in the wood drops from grazing to the
        // incidence angle; that step is what a leak keeping its direction means and is recorded.
        const double e_w = 2.1609e-8, e_c = 2.401e-9;
        const std::vector<LumpMtl> pw = {
            {e_w, 0.0, 0.020, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 0.001},
            {e_c, 0.0, 0.0, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 30.0, 0.001},
        };
        auto mw = lump_palette(pw);
        auto wood = [&](double thd)
        {
            auto C = make1<double>(mw, 4, OT_GLUED_ENTRY, 2u, 1, 2, 0, 0, 0, L, 0.5, 0.5, QPI / 2.0 - thd * QPI / 180.0, 1.5);
            C.run();
            return dB((double)C.gain(0));
        };
        const double t_outer = std::asin(std::sqrt(e_c)) * 180.0 / QPI, t_near = std::asin(std::sqrt(e_w)) * 180.0 / QPI;
        double st_o = wood(t_outer * (1.0 + 1.0e-6)) - wood(t_outer * (1.0 - 1.0e-6));
        double st_n = wood(t_near * (1.0 + 1.0e-6)) - wood(t_near * (1.0 - 1.0e-6));
        QD_DIAG("wood panel on concrete at the outer pair,%.6f,%+.4f,,\n", t_outer, st_o);
        QD_DIAG("wood panel on concrete at the air | wood face,%.6f,%+.4f,,\n", t_near, st_n);
        CHECK(std::abs(st_o) < 0.05);
        CHECK(std::abs(st_n) < 5.0); // recorded: 10 log10(0.82 / 0.28) = 4.67 dB, the wave becoming a leak
    }
}

// One body of a bonded mass-law plate, for diag_bonded_TL: the slope, the reference frequency and
// the normal thickness, exactly the three quantities Material::medium_gain forms its term from.
struct DiagMass
{
    double m, fRef, thk;
};

// Mass term of a bonded plate, in dB, written from the column definitions. Bonded bodies move with
// one velocity, so their transfer impedances are in series and what adds is the impedance ratio
//     X0 = ((f / fRef) M)^(m / 20)
// of each body, not its term in dB. The plate is then one body: A = 20 log10(sum X0), with the
// single-body angle shape. A body contributes its X0 unclipped, so the hard zero sits on the sum;
// a body without the column (m = 0) is absent.
static inline double diag_bonded_TL(const std::vector<DiagMass> &bodies, double f_hz, double cosv)
{
    double X = 0.0;
    for (const auto &B : bodies)
    {
        double arg = (f_hz * 1e-9 / B.fRef) * B.thk;
        if (B.m > 0.0 && arg > 0.0)
            X += std::pow(arg, 0.05 * B.m);
    }
    if (!(X > 1.0) || !(cosv > 0.0))
        return 0.0;
    double A = 20.0 * std::log10(X);
    if (cosv >= 1.0)
        return A;
    double x = X * cosv;
    return A * std::log10(1.0 + x * x) / std::log10(1.0 + X * X);
}

TEST_CASE("Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate")
{
    // Item 1. Two mass-law bodies within the co-location distance have no air gap, so they move as
    // one plate and their transfer impedances add: TL = 20 log10(X01 + X02), the pair being one
    // body in every other respect. Each body used to pay its own term, so the terms added in dB,
    // which is the decoupled double leaf.
    //
    // Charging is telescoped. The first body crossed pays its own term, every later one pays
    // TL(X_run + X_own) - TL(X_run), so no event carries a factor above one and the product over
    // the plate is the pair. Where the second body is a traced solid it needs the first one's
    // surface mass at its own close-out: a sheet's is its thk column, a solid's is the normal
    // thickness of the leg just closed, which travels as a logarithmic code in the buffer word
    // under its flag bit. That code is quantized, so the solid | solid rows are bounded at
    // 0.01 dB and everything else at 1e-9.
    //
    // Two instruments. Where both bodies are one material with m = 20, X0 is linear in thickness
    // and "one body of the summed surface mass" is a statement that does not use the rule: the
    // pair is compared with that body through the library. Everywhere else the mass paid is
    // isolated as a ratio against the same bodies with their mass argument taken to nothing
    // (fRef 1e9), which leaves every face, tf and interface gain in place, and is compared with
    // diag_bonded_TL.
    //
    // The ray runs on the x axis and the angle comes from fbs_angle, so a solid of normal
    // thickness d is given the path d / cos.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9, tf_w = 0.2, tf_c = 0.1;
    const double tw = 0.020, Lw = 0.10, Lc = 0.10, fR = 0.001, fZ = 1.0e9;
    const uint8_t OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69, OT_STACK_FBS = 71, OT_STACK_SBS = 69;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    const int P8 = 1, P12 = 2, P20 = 3, P20Z = 4, WS = 5, WSF = 6, WS2 = 7, SB = 8, CONC = 9, CONCZ = 10,
              P8Z = 11, SBZ = 12, CARPET = 13, WSZ = 14;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                  // 1  P8     wood sheet, X0 8
        {e_w, 0.0, 0.012, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                  // 2  P12    X0 12
        {e_w, 0.0, tw, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                     // 3  P20    X0 20
        {e_w, 0.0, tw, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                     // 4  P20Z   P20 without its mass
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                    // 5  WS     wood solid
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR * Lw / (Lw + tw)},   // 6  WSF    WS with P20's mass folded in
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},                    // 7  WS2    WS under a second index
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},                  // 8  SB     a rigid board, m 30
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},                    // 9  CONC   concrete solid, m 30
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},                    // 10 CONCZ  without its mass
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                  // 11 P8Z
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},                  // 12 SBZ
        {2.50, 0.80, 0.010},                                                         // 13 CARPET a lossy lining, m 0
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                    // 14 WSZ
        {e_w, 0.0, 0.012, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},                  // 15 P12Z
    };
    const int P12Z = 15;
    auto mtl = lump_palette(pal);
    mtl["attB"][(size_t)WS2 - 1] = 1.0; // att is zero, so this separates WS2 from WS and does nothing else

    const std::vector<double> angles = {0.0, 30.0, 60.0, 75.0};
    const DiagMass b_p8{20.0, fR, 0.008}, b_p20{20.0, fR, tw}, b_sb{30.0, fR, 0.012};
    const DiagMass b_ws{20.0, fR, Lw}, b_conc{30.0, fR, Lc};

    // The reference at one body is the single-body term
    for (double thd : angles)
    {
        double ct = std::cos(thd * QPI / 180.0);
        CHECK(std::abs(diag_bonded_TL({b_p20}, f_hz, ct) - diag_mass_TL(20.0, fR, f_hz, tw, ct)) < 1.0e-12);
        CHECK(std::abs(diag_bonded_TL({b_conc}, f_hz, ct) - diag_mass_TL(30.0, fR, f_hz, Lc, ct)) < 1.0e-12);
    }

    auto stk = [&](int itype, uint8_t ot, int m1, int m2, double th)
    {
        auto C = make1<double>(mtl, itype, ot, 2u, m1, m2, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };
    auto one = [&](int itype, int m1, double th)
    {
        auto C = make1<double>(mtl, itype, 67, 1u, m1, 0, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 0.0);
        C.run();
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // Sheet on sheet
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: bonded mass, sheet on sheet. Wood 8 mm | wood 12 mm against one 20 mm sheet of the\n");
    QD_DIAG("# same wood (m 20, tf 0.2), scalar, both orders. over_dB is what the stack pays above the\n");
    QD_DIAG("# single sheet; sum_minus_pair is what adding in dB would cost\n");
    QD_DIAG("theta_n,order,T_one_dB,T_stack_dB,over_dB,sum_minus_pair,d_T,d_R\n");
    for (double thd : angles)
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto St = stk(4, ot, P8, P12, th), Sr = stk(3, ot, P8, P12, th);
            auto Ot = one(4, P20, th), Or = one(3, P20, th);
            double over = dB((double)Ot.gain(0)) - dB((double)St.gain(0));
            double add = diag_mass_TL(20.0, fR, f_hz, 0.008, ct) + diag_mass_TL(20.0, fR, f_hz, 0.012, ct) -
                         diag_mass_TL(20.0, fR, f_hz, tw, ct);
            double d_T = std::abs(cpx(St.xprmat, 0, 0) - cpx(Ot.xprmat, 0, 0)) / std::abs(cpx(Ot.xprmat, 0, 0));
            double d_R = std::abs(cpx(Sr.xprmat, 0, 0) - cpx(Or.xprmat, 0, 0));
            QD_DIAG("%.1f,%s,%.4f,%.4f,%+.6f,%+.6f,%.3e,%.3e\n", thd, ord ? "2|1" : "1|2", dB((double)Ot.gain(0)),
                    dB((double)St.gain(0)), over, add, d_T, d_R);
            INFO("sheet on sheet, one material, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9); // one sheet of the summed surface mass, complex
            CHECK(d_R < 1.0e-9); // the reflect port is the first face and never read the mass
        }

    QD_DIAG("\n# diag: bonded mass, sheet on sheet, two materials. Wood 8 mm (m 20, X0 8) | board 12 mm\n");
    QD_DIAG("# (m 30, X0 41.6), tf on both. paid is the mass the stack pays, against the plate and\n");
    QD_DIAG("# against the terms added in dB\n");
    QD_DIAG("theta_n,order,paid_dB,plate_dB,sum_dB,paid_minus_plate\n");
    for (double thd : angles)
        for (int ord = 0; ord < 2; ++ord)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            uint8_t ot = ord ? OT_STACK_SBS : OT_STACK_FBS;
            auto St = stk(4, ot, P8, SB, th), Sz = stk(4, ot, P8Z, SBZ, th);
            double paid = dB((double)Sz.gain(0)) - dB((double)St.gain(0));
            double plate = diag_bonded_TL({b_p8, b_sb}, f_hz, ct);
            double sum = diag_mass_TL(20.0, fR, f_hz, 0.008, ct) + diag_mass_TL(30.0, fR, f_hz, 0.012, ct);
            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%+.3e\n", thd, ord ? "2|1" : "1|2", paid, plate, sum, paid - plate);
            INFO("sheet on sheet, two materials, order " << (ord ? "2|1" : "1|2") << ", theta_n " << thd);
            CHECK(std::abs(paid - plate) < 1.0e-9);
            CHECK(std::abs(sum - plate) > 0.1); // the row can tell the two rules apart
        }

    // ------------------------------------------------------------------------------------------
    // Sheet on a traced solid, both directions
    // ------------------------------------------------------------------------------------------
    // Entry is two events: the sheet crossing, then the close-out of the solid's leg at its far
    // face, fed the three state words the crossing wrote. Exit is one event: the host leg closes
    // and the sheet is crossed together.
    struct Chain
    {
        std::complex<double> T; // sheet composite times the solid's close-out factor
        double g;
        int buf;
    };
    auto entry = [&](int sheet, int solid, double L, double th)
    {
        auto E1 = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, solid, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        E1.run();
        auto E2 = make1<double>(mtl, 4, OT_EXIT, 1, solid, 0, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                (int)E1.buf_out(0), L / std::cos(th), 2.0, 0.5, QPI / 2.0 - th, 1.5);
        E2.run();
        Chain R;
        R.T = cpx(E1.xprmat, 0, 0) * vvf(E2);
        R.g = std::norm(R.T), R.buf = (int)E1.buf_out(0);
        return R;
    };
    auto exitc = [&](int sheet, int solid, double L, double th)
    {
        auto X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, sheet, solid, 0, solid, 0, L / std::cos(th), 0.5, 0.5,
                               QPI / 2.0 - th, 1.5);
        X.run();
        Chain R;
        R.T = cpx(X.xprmat, 0, 0), R.g = (double)X.gain(0), R.buf = (int)X.buf_out(0);
        return R;
    };

    QD_DIAG("\n# diag: bonded mass, sheet on solid, one material. Wood 20 mm sheet on 100 mm of the same wood\n");
    QD_DIAG("# against that solid with the sheet's mass folded into its fRef (m 20), scalar. Entry is\n");
    QD_DIAG("# air | sheet | solid and the solid's close-out; exit is solid | sheet | air\n");
    QD_DIAG("theta_n,dir,T_folded_dB,T_pair_dB,over_dB,sum_minus_pair,d_T\n");
    for (double thd : angles)
        for (int dir = 0; dir < 2; ++dir)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Chain P = dir ? exitc(P20, WS, Lw, th) : entry(P20, WS, Lw, th);
            Chain F = dir ? exitc(P20Z, WSF, Lw, th) : entry(P20Z, WSF, Lw, th);
            double add = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(20.0, fR, f_hz, Lw, ct) -
                         diag_mass_TL(20.0, fR, f_hz, tw + Lw, ct);
            double d_T = std::abs(P.T - F.T) / std::abs(F.T);
            QD_DIAG("%.1f,%s,%.4f,%.4f,%+.6f,%+.6f,%.3e\n", thd, dir ? "exit " : "entry", dB(F.g), dB(P.g),
                    dB(F.g) - dB(P.g), add, d_T);
            INFO("sheet on solid, one material, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(d_T < 1.0e-9);
        }

    QD_DIAG("\n# diag: bonded mass, wood panel (20 mm, m 20, tf 0.2) on concrete (100 mm, m 30, tf 0.1,\n");
    QD_DIAG("# att 2 dB), scalar. paid is the mass the pair pays in that direction\n");
    QD_DIAG("theta_n,dir,paid_dB,plate_dB,sum_dB,paid_minus_plate\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_p20, b_conc}, f_hz, ct);
        double sum = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(30.0, fR, f_hz, Lc, ct);
        double paid_dir[2];
        for (int dir = 0; dir < 2; ++dir)
        {
            Chain P = dir ? exitc(P20, CONC, Lc, th) : entry(P20, CONC, Lc, th);
            Chain Z = dir ? exitc(P20Z, CONCZ, Lc, th) : entry(P20Z, CONCZ, Lc, th);
            double paid = dB(Z.g) - dB(P.g);
            paid_dir[dir] = paid;
            QD_DIAG("%.1f,%s,%.6f,%.6f,%.6f,%+.3e\n", thd, dir ? "exit " : "entry", paid, plate, sum, paid - plate);
            INFO("wood panel on concrete, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(std::abs(paid - plate) < 1.0e-9);
        }
        CHECK(std::abs(paid_dir[0] - paid_dir[1]) < 1.0e-9); // the two directions through the pair
        CHECK(sum - plate > 10.0);
    }

    // A lining without the column is absent from the plate: the concrete pays its own term
    QD_DIAG("\n# diag: an m = 0 lining on concrete (lossy 10 mm, eps 2.50 - 0.80j). The concrete's mass,\n");
    QD_DIAG("# paid against its own term\ntheta_n,dir,paid_dB,own_dB,d\n");
    for (double thd : angles)
        for (int dir = 0; dir < 2; ++dir)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Chain P = dir ? exitc(CARPET, CONC, Lc, th) : entry(CARPET, CONC, Lc, th);
            Chain Z = dir ? exitc(CARPET, CONCZ, Lc, th) : entry(CARPET, CONCZ, Lc, th);
            double paid = dB(Z.g) - dB(P.g), own = diag_mass_TL(30.0, fR, f_hz, Lc, ct);
            QD_DIAG("%.1f,%s,%.6f,%.6f,%+.3e\n", thd, dir ? "exit " : "entry", paid, own, paid - own);
            INFO("m = 0 lining on concrete, " << (dir ? "exit" : "entry") << ", theta_n " << thd);
            CHECK(std::abs(paid - own) < 1.0e-9);
        }

    // ------------------------------------------------------------------------------------------
    // Solid on solid, both directions, and the leg that begins at a reflection
    // ------------------------------------------------------------------------------------------
    // No interface coefficient is written on these events - no refracted direction is supplied,
    // so the one ray_mesh_interact wrote stands - and the permittivities are lossless, so the
    // applied factor is the mass term and a phase.
    struct Leg
    {
        double paid; // dB, this event
        int prev, cur, buf;
    };
    auto m2m = [&](int a, int b, double d, double th, int prev, int buf)
    {
        auto C = make1<double>(mtl, 4, OT_M2M_M1, 2u, a, b, prev, a, buf, d / std::cos(th), 2.0, 0.0, QPI / 2.0 - th, 1.5);
        C.run();
        return Leg{-dB(std::norm(vvf(C))), (int)C.prev_out(0), (int)C.cur_out(0), (int)C.buf_out(0)};
    };
    auto far_face = [&](int itype, int b, double d, double th, int prev, int buf)
    {
        auto C = make1<double>(mtl, itype, OT_EXIT, 1, b, 0, prev, b, buf, d / std::cos(th), 2.0, 0.5, QPI / 2.0 - th, 1.5);
        C.run();
        return Leg{-dB(std::norm(vvf(C))), (int)C.prev_out(0), (int)C.cur_out(0), (int)C.buf_out(0)};
    };

    QD_DIAG("\n# diag: bonded mass, solid on solid. Wood 100 mm (m 20, X0 100) and concrete 100 mm (m 30,\n");
    QD_DIAG("# X0 1000), and wood 60 mm | wood 40 mm against one 100 mm solid. first and second are what\n");
    QD_DIAG("# the two close-outs pay; flag is bit 15 of the buffer word after the shared face\n");
    QD_DIAG("pair,theta_n,first_dB,second_dB,paid_dB,plate_dB,sum_dB,paid_minus_plate,flag\n");
    struct SS
    {
        const char *name;
        int a, b;
        double da, db;
        DiagMass ma, mb;
    };
    const std::vector<SS> pairs = {
        {"wood | concrete", WS, CONC, Lw, Lc, b_ws, b_conc},
        {"concrete | wood", CONC, WS, Lc, Lw, b_conc, b_ws},
        {"wood 60 | wood 40", WS, WS2, 0.06, 0.04, {20.0, fR, 0.06}, {20.0, fR, 0.04}},
        {"wood 40 | wood 60", WS2, WS, 0.04, 0.06, {20.0, fR, 0.04}, {20.0, fR, 0.06}},
    };
    for (const auto &S : pairs)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            Leg A = m2m(S.a, S.b, S.da, th, 0, 0);
            Leg B = far_face(4, S.b, S.db, th, A.prev, A.buf);
            double paid = A.paid + B.paid, plate = diag_bonded_TL({S.ma, S.mb}, f_hz, ct);
            double own_a = diag_bonded_TL({S.ma}, f_hz, ct), own_b = diag_bonded_TL({S.mb}, f_hz, ct);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.3e,%d\n", S.name, thd, A.paid, B.paid, paid, plate,
                    own_a + own_b, paid - plate, (A.buf & 0x8000) ? 1 : 0);
            INFO(S.name << ", theta_n " << thd);
            CHECK(std::abs(A.paid - own_a) < 1.0e-9); // the first body crossed pays its own term
            CHECK(std::abs(paid - plate) < 0.01);     // the thickness code is quantized
            CHECK((A.buf & 0x8000) != 0);             // and travels under the flag
            if (S.a == WS && S.b == WS2)              // one material: one solid of the summed thickness
                CHECK(std::abs(paid - diag_mass_TL(20.0, fR, f_hz, 0.10, ct)) < 0.01);
        }

    QD_DIAG("\n# diag: a bounce inside the second body, wood | concrete. The reflection at the concrete's far\n");
    QD_DIAG("# face closes the first traversal and clears the code; the way back pays the pair once more,\n");
    QD_DIAG("# the concrete its own term and the wood the increment\n");
    QD_DIAG("theta_n,fwd_dB,refl_buf,back_conc_dB,conc_own_dB,back_wood_dB,back_dB,plate_dB\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_ws, b_conc}, f_hz, ct), own_c = diag_bonded_TL({b_conc}, f_hz, ct);
        Leg A = m2m(WS, CONC, Lw, th, 0, 0);
        Leg R = far_face(3, CONC, Lc, th, A.prev, A.buf);  // reflect at the far face
        Leg C = m2m(CONC, WS, Lc, th, R.prev, R.buf);      // back through the concrete
        Leg W = far_face(4, WS, Lw, th, C.prev, C.buf);    // and out through the wood
        QD_DIAG("%.1f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", thd, A.paid + R.paid, R.buf, C.paid, own_c, W.paid,
                C.paid + W.paid, plate);
        INFO("bounce in the second body, theta_n " << thd);
        CHECK(std::abs(A.paid + R.paid - plate) < 0.01); // the reflection pass closes the leg as the exit does
        CHECK(R.buf == 0x4000);                          // followed, and no mass flag: nothing bonded was crossed on the way back yet
        CHECK(std::abs(C.paid - own_c) < 1.0e-9);        // so the stale previous medium is not read
        CHECK(std::abs(C.paid + W.paid - plate) < 0.01);
    }

    // ------------------------------------------------------------------------------------------
    // A solid between two sheets, and a bounce returning to the glued sheet it entered by
    // ------------------------------------------------------------------------------------------
    // Both leave the solid through a glued sheet with a sheet in the previous-medium word. That
    // word is not a medium there: what lies beyond the sheet ahead is air, and the event leaves
    // the ray in air. The mass rule needs nothing new: the host leg hands the sheet ahead its own
    // X0 plus what it was entered through.
    //
    // The mass paid is a ratio of two runs through the same faces, so it cannot see the far
    // medium. The level can: the one-material sandwich is compared with the solid carrying all
    // three masses, entered through a massless sheet and left through a massless sheet with a
    // clean previous-medium word, and the returning bounce with the same exit entered through
    // nothing.
    auto exit_from = [&](int sheet, int solid, double L, double th, int prev, int cur, int buf)
    {
        auto X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, sheet, solid, prev, cur, buf, L / std::cos(th), 0.5, 0.5,
                               QPI / 2.0 - th, 1.5);
        X.run();
        return X;
    };
    auto entry_ev = [&](int sheet, int solid, double th)
    {
        auto E = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, solid, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        E.run();
        return E;
    };

    QD_DIAG("\n# diag: bonded mass, a solid between two sheets, one material. Wood 8 mm | wood 100 mm | wood\n");
    QD_DIAG("# 12 mm against the solid with both sheets' mass folded into its fRef (m 20), scalar, two\n");
    QD_DIAG("# events. The reference leaves through its second sheet with a clean previous-medium word.\n");
    QD_DIAG("# prev, cur and buf are the state words after the exit\n");
    QD_DIAG("theta_n,T_folded_dB,T_pair_dB,over_dB,d_T,prev,cur,buf\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0;
        auto E1 = entry_ev(P8, WS, th), F1 = entry_ev(P8Z, WSF, th);
        auto E2 = exit_from(P12, WS, Lw, th, (int)E1.prev_out(0), (int)E1.cur_out(0), (int)E1.buf_out(0));
        auto F2 = exit_from(P12Z, WSF, Lw, th, 0, WSF, 0);
        std::complex<double> Tp = cpx(E1.xprmat, 0, 0) * cpx(E2.xprmat, 0, 0);
        std::complex<double> Tf = cpx(F1.xprmat, 0, 0) * cpx(F2.xprmat, 0, 0);
        double d_T = std::abs(Tp - Tf) / std::abs(Tf);
        QD_DIAG("%.1f,%.4f,%.4f,%+.6f,%.3e,%d,%d,%d\n", thd, dB(std::norm(Tf)), dB(std::norm(Tp)),
                dB(std::norm(Tf)) - dB(std::norm(Tp)), d_T, mat_of(E2.prev_out(0)), mat_of(E2.cur_out(0)),
                (int)E2.buf_out(0));
        INFO("solid between two sheets, one material, theta_n " << thd);
        check_state(E1, P8, WS, (int)(short)0x8000); // the sheet was crossed, and says so
        CHECK(d_T < 1.0e-9);                         // one body of the summed surface mass, with air beyond
        check_state(E2, P12, 0, 0);                  // and the ray is in air
    }

    QD_DIAG("\n# diag: bonded mass, wood panel 20 mm | concrete 100 mm | rigid board 12 mm (m 20, 30, 30).\n");
    QD_DIAG("# paid is the mass the two events pay, against the plate of all three\n");
    QD_DIAG("theta_n,first_dB,second_dB,paid_dB,plate_dB,sum_dB,paid_minus_plate,cur\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        auto E1 = entry_ev(P20, CONC, th), Z1 = entry_ev(P20Z, CONCZ, th);
        auto E2 = exit_from(SB, CONC, Lc, th, (int)E1.prev_out(0), (int)E1.cur_out(0), (int)E1.buf_out(0));
        auto Z2 = exit_from(SBZ, CONCZ, Lc, th, (int)Z1.prev_out(0), (int)Z1.cur_out(0), (int)Z1.buf_out(0));
        double first = dB((double)Z1.gain(0)) - dB((double)E1.gain(0));
        double second = dB((double)Z2.gain(0)) - dB((double)E2.gain(0));
        double plate = diag_bonded_TL({b_p20, b_conc, b_sb}, f_hz, ct);
        double sum = diag_mass_TL(20.0, fR, f_hz, tw, ct) + diag_mass_TL(30.0, fR, f_hz, Lc, ct) +
                     diag_mass_TL(30.0, fR, f_hz, 0.012, ct);
        QD_DIAG("%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.3e,%d\n", thd, first, second, first + second, plate, sum,
                first + second - plate, mat_of(E2.cur_out(0)));
        INFO("solid between two sheets, three bodies, theta_n " << thd);
        CHECK(std::abs(first + second - plate) < 1.0e-9);
        CHECK(sum - plate > 10.0);
        check_state(E2, SB, 0, 0);
    }

    QD_DIAG("\n# diag: a bounce returning to the glued sheet it entered by. Wood panel 20 mm on concrete\n");
    QD_DIAG("# 100 mm: in through the panel, reflected at the bare far face, out through the panel. fwd\n");
    QD_DIAG("# and back are the mass each traversal pays; d_bare compares the return event with the same\n");
    QD_DIAG("# exit entered with a clean previous-medium word\n");
    QD_DIAG("theta_n,fwd_dB,back_dB,plate_dB,refl_prev,refl_buf,back_minus_plate,over_bare_dB,d_bare,cur\n");
    for (double thd : angles)
    {
        double th = thd * QPI / 180.0, ct = std::cos(th);
        double plate = diag_bonded_TL({b_p20, b_conc}, f_hz, ct);
        auto E1 = entry_ev(P20, CONC, th), Z1 = entry_ev(P20Z, CONCZ, th);
        Leg R = far_face(3, CONC, Lc, th, (int)E1.prev_out(0), (int)E1.buf_out(0));
        Leg Rz = far_face(3, CONCZ, Lc, th, (int)Z1.prev_out(0), (int)Z1.buf_out(0));
        auto X = exit_from(P20, CONC, Lc, th, R.prev, R.cur, R.buf);
        auto Xz = exit_from(P20Z, CONCZ, Lc, th, Rz.prev, Rz.cur, Rz.buf);
        auto B = exit_from(P20, CONC, Lc, th, 0, CONC, 0);
        double fwd = dB((double)Z1.gain(0)) - dB((double)E1.gain(0)) + R.paid - Rz.paid;
        double back = dB((double)Xz.gain(0)) - dB((double)X.gain(0));
        double d_bare = std::abs(cpx(X.xprmat, 0, 0) - cpx(B.xprmat, 0, 0)) / std::abs(cpx(B.xprmat, 0, 0));
        QD_DIAG("%.1f,%.6f,%.6f,%.6f,%d,%d,%+.3e,%+.6f,%.3e,%d\n", thd, fwd, back, plate, R.prev, R.buf, back - plate,
                dB((double)X.gain(0)) - dB((double)B.gain(0)), d_bare, mat_of(X.cur_out(0)));
        INFO("bounce returning to the glued sheet, theta_n " << thd);
        CHECK(std::abs(fwd - plate) < 1.0e-9);  // the way in pays the plate
        CHECK(R.prev == P20);                   // the stale sheet is still in the word
        CHECK(R.buf == 0x4000);                 // but it was not crossed on this traversal: followed, no mass flag
        CHECK(std::abs(back - plate) < 1.0e-9); // so the way back pays the plate once
        CHECK(d_bare < 1.0e-12);                // through the same exit a bare-entered solid takes
        check_state(X, P20, 0, 0);
    }
}

TEST_CASE("Ports of a mass-law face from inside - the reflect port is the other direction's")
{
    // Item 3. A wave held undeviated inside a mass-law medium, at a face towards the lower index,
    // is transmitted as the other direction transmits. Its reflection was still judged from
    // inside, with the critical angle the transmit rule exists to avoid, so the two ports of one
    // face added to more than one: 1.50 at 30 deg for wood | concrete.
    //
    // The reflect port is the reflection the other direction sees, sign reversed, which is
    // Stokes' relation for one face read from its two sides at the cosine the wave holds. For a
    // lossless pair it conserves, in scalar mode and on the TE/TM mean in EM; per channel in EM
    // the bare boundary applies tf on the mean and does not conserve from either side.
    //
    // Through ray_mesh_interact, whose incidence angle is the geometric one, so the wave is
    // always held undeviated there. No interface gain, so the ports are the face alone.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 20.0, 1.0}, // 1 WOOD as a solid
        {e_c, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 30.0, 1.0}, // 2 CONC as a solid
    };
    auto mtl = lump_palette(pal);
    arma::mat mesh = quadriga_lib::cube<double>();
    arma::mat cube2 = quadriga_lib::cube<double>(arma::vec{1.0}, arma::vec{0.0, 0.0, 0.0}, arma::vec{2.0, 0.0, 0.0});
    mesh = arma::join_cols(mesh, cube2);
    arma::uvec mtl_ind(mesh.n_rows);
    for (arma::uword i = 0; i < mesh.n_rows; ++i)
        mtl_ind(i) = (i < mesh.n_rows / 2) ? 1 : 2;

    struct Port
    {
        double g;
        std::complex<double> c;
    };
    auto face = [&](int itype, bool wood_to_conc, double th)
    {
        double dy = 0.2 * std::tan(th), xa = wood_to_conc ? 0.9 : 1.1, xb = wood_to_conc ? 1.1 : 0.9;
        arma::mat orig = {{xa, -0.4, 0.1}}, dest = {{xb, -0.4 + dy, 0.1}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN);
        REQUIRE(gainN.n_elem == 1);
        return Port{(double)gainN(0), std::complex<double>(xprmatN(0, 0), xprmatN(1, 0))};
    };

    QD_DIAG("\n# diag: ports of the wood | concrete face from inside the wood (tf 0.2 | 0.1, lossless, no att),\n");
    QD_DIAG("# through ray_mesh_interact. T_in and R_in are the scalar ports from the wood, R_other the\n");
    QD_DIAG("# reflection from the concrete; d_stokes is |r_in + r_other|; sum_EM the EM gains from the wood\n");
    QD_DIAG("theta_n,T_in,R_in,sum,R_other,d_R,d_stokes,sum_EM\n");
    for (double thd : {0.0, 10.0, 19.0, 30.0, 45.0, 70.0})
    {
        double th = thd * QPI / 180.0;
        Port T = face(4, true, th), R = face(3, true, th), Ro = face(3, false, th), To = face(4, false, th);
        Port Te = face(1, true, th), Re = face(0, true, th);
        double d_st = std::abs(R.c + Ro.c);
        QD_DIAG("%.1f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%.9f\n", thd, T.g, R.g, T.g + R.g, Ro.g, std::abs(R.g - Ro.g),
                d_st, Te.g + Re.g);
        INFO("wood | concrete from inside the wood, theta_n " << thd);
        CHECK(std::abs(T.g - To.g) < 1.0e-12);        // settled: the transmit port is the other direction's
        CHECK(std::abs(Ro.g + To.g - 1.0) < 1.0e-12); // the concrete side conserves, today and after
        CHECK(std::abs(T.g + R.g - 1.0) < 1.0e-12);   // and so does the wood side
        CHECK(std::abs(R.g - Ro.g) < 1.0e-12);        // because it is the same reflectance
        CHECK(d_st < 1.0e-12);                        // with the sign a face has from its other side
        CHECK(std::abs(Te.g + Re.g - 1.0) < 1.0e-12); // EM, on the TE/TM mean
    }
}

TEST_CASE("Lined on both faces - a glued exit behind a sheet exits into air")
{
    // Item 1. A glued exit, solid | sheet | whatever encloses the host, took its far medium from
    // the previous-medium word. After a sheet crossing that word holds a sheet, so a solid entered
    // through a sheet and left through a sheet resolved its second sheet against a half-space of
    // the first sheet's material and left the event with that sheet as the current medium.
    //
    // A sheet in that word is not a medium. Beyond the sheet ahead is air, which is what
    // ray_mesh_interact's own composite assumes for every glued exit, and the exit leaves the ray
    // in air with the sheet it crossed as the previous medium.
    //
    // No path_dir is supplied, so every boundary is at the fbs cosine and the host leg closes in
    // the hybrid form diag_close reproduces. The composite is measured with the host's series
    // declined (eps 1.5) and the series as the ratio of the run that resolves (eps 0) over it.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9, CLEAN = 6;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const std::complex<double> e_h = diag_eps(HOST, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Row
    {
        const char *name;
        int s1, s2;
    };
    const std::vector<Row> rows = {
        {"S | host | S", SHEET, SHEET},
        {"N | host | S", NEAR, SHEET},
        {"S | host | N", SHEET, NEAR},
    };
    const std::vector<double> angles = {0.0, 20.0, 40.0};

    auto entry = [&](int itype, int s1, double th)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_ENTRY, 2u, s1, HOST, 0, 0, 0, 0.5, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.run();
        return C;
    };
    auto exitev = [&](int itype, int s2, int host, int prev, int cur, double th, double eps)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, s2, host, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.run();
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // The composite and the state words, both ports, scalar
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: lined on both faces, air | S1 | host | S2 | air as two events, host eps 1.50 - 0.40j\n");
    QD_DIAG("# over 100 mm, S eps 5.24 - 0.63j and N eps 1.5015 - 0.40j, 6 mm, scalar, host series declined.\n");
    QD_DIAG("# The second event against the reference with air beyond (d_ref) and against the same exit\n");
    QD_DIAG("# with a clean previous-medium word (d_bare); c_ref is that control run against the reference\n");
    QD_DIAG("stack,theta_n,port,meas_dB,ref_dB,d_ref,d_bare,c_ref,prev,cur,resolved,buf,rtype\n");
    for (const auto &R : rows)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            auto E1 = entry(4, R.s1, th);
            INFO(R.name << ", theta_n " << thd);
            check_state(E1, R.s1, HOST, 0);

            DiagSheet D = diag_sheet_composite(e_h, diag_eps(R.s2, f_hz), e_air, thk, ct, f_hz, true, true);
            std::complex<double> close = diag_close(e_h, L, th, f_hz);
            for (int port = 0; port < 2; ++port)
            {
                int itype = port ? 3 : 4;
                auto A = exitev(itype, R.s2, HOST, (int)E1.prev_out(0), (int)E1.cur_out(0), th, 1.5);
                auto B = exitev(itype, R.s2, HOST, 0, HOST, th, 1.5);
                std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
                std::complex<double> ref = (port ? D.R_te : D.T_te) * close;
                double d_ref = std::abs(a - ref) / std::abs(ref), d_bare = std::abs(a - b) / std::abs(b);
                double c_ref = std::abs(b - ref) / std::abs(ref);
                QD_DIAG("%s,%.1f,%s,%.4f,%.4f,%.3e,%.3e,%.3e,%d,%d,%d,%d,%u\n", R.name, thd, port ? "R" : "T",
                        dB(std::norm(a)), dB(std::norm(ref)), d_ref, d_bare, c_ref, mat_of(A.prev_out(0)),
                        mat_of(A.cur_out(0)), (int)flag_of(A.cur_out(0)), (int)A.buf_out(0),
                        (unsigned)A.resolved_type[0]);
                INFO((port ? "reflect" : "transmit"));
                CHECK(c_ref < 1.0e-9); // the reference is the one the bare-entered exit sits on
                CHECK(d_ref < 1.0e-9); // air beyond the second sheet
                CHECK(a == b);         // and the exit does not know how the host was entered
                if (port)
                {
                    check_state(A, R.s2, enc(HOST, false), 0x4000); // a reflection stays in the host; its series was declined, so it is not resolved but followed
                    check_rtype(A, 207);
                }
                else
                {
                    check_state(A, R.s2, 0, 0); // the ray is in air, behind the sheet it crossed
                    check_state(B, R.s2, 0, 0); // as it is after any glued exit
                    check_rtype(A, 71);
                }
            }
        }

    // ------------------------------------------------------------------------------------------
    // The same in EM, per entry of the matrix, against the bare-entered exit
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: lined on both faces, EM. Gain of the second event against the same exit with a clean\n");
    QD_DIAG("# previous-medium word; d_x is the largest difference over the eight matrix entries\n");
    QD_DIAG("stack,theta_n,port,gain_dB,bare_dB,d_gain_dB,d_x,cur\n");
    for (const auto &R : rows)
        for (double thd : angles)
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                auto E1 = entry(1, R.s1, th);
                auto A = exitev(port ? 0 : 1, R.s2, HOST, (int)E1.prev_out(0), (int)E1.cur_out(0), th, 1.5);
                auto B = exitev(port ? 0 : 1, R.s2, HOST, 0, HOST, th, 1.5);
                double d_x = 0.0;
                for (arma::uword k = 0; k < 8; ++k)
                    d_x = std::max(d_x, std::abs((double)A.xprmat(k, 0) - (double)B.xprmat(k, 0)));
                QD_DIAG("%s,%.1f,%s,%.4f,%.4f,%+.4f,%.3e,%d\n", R.name, thd, port ? "R" : "T", dB((double)A.gain(0)),
                        dB((double)B.gain(0)), dB((double)A.gain(0)) - dB((double)B.gain(0)), d_x,
                        mat_of(A.cur_out(0)));
                INFO(R.name << ", EM, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                CHECK(d_x == 0.0);
                CHECK((double)A.gain(0) == (double)B.gain(0));
                CHECK(mat_of(A.cur_out(0)) == (port ? HOST : 0));
            }

    // ------------------------------------------------------------------------------------------
    // The host's series between two sheets
    // ------------------------------------------------------------------------------------------
    // Its mirrors are the sheet ahead and the sheet behind, each a layer of its own thickness
    // with air beyond. The bare-entered row has a half-space of air behind and is the control.
    QD_DIAG("\n# diag: lined on both faces, the host's series. S_applied is the resolving run over the\n");
    QD_DIAG("# declined one; S_ref has both sheets as layers backed by air. behind = 0 is the bare-entered\n");
    QD_DIAG("# control, with a half-space of air behind\n");
    QD_DIAG("ahead,behind,theta_n,port,|S_applied|,arg_S_applied,|S_ref|,arg_S_ref,|dS|,d_dB\n");
    for (const auto &R : rows)
        for (int ctrl = 0; ctrl < 2; ++ctrl)
            for (double thd : angles)
                for (int port = 0; port < 2; ++port)
                {
                    double th = thd * QPI / 180.0;
                    int itype = port ? 3 : 4, prev = ctrl ? 0 : R.s1;
                    auto A1 = exitev(itype, R.s2, HOST, prev, HOST, th, 0.0);
                    auto A0 = exitev(itype, R.s2, HOST, prev, HOST, th, 1.5);
                    std::complex<double> S = cpx(A1.xprmat, 0, 0) / cpx(A0.xprmat, 0, 0);

                    std::complex<double> cs(std::cos(th), 0.0);
                    std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                    DiagSheet Ba = diag_cm_stack(e_h, U, {{diag_eps(R.s2, f_hz), U, thk}}, e_air, U, cs, f_hz,
                                                 false, true, 0.0);
                    std::complex<double> r_behind = diag_r_slabside(e_h, e_air, cs);
                    if (!ctrl)
                        r_behind = diag_cm_stack(e_h, U, {{diag_eps(R.s1, f_hz), U, thk}}, e_air, U, cs, f_hz,
                                                 false, true, 0.0)
                                       .R_te;
                    std::complex<double> S_ref = 1.0 / (1.0 - Ba.R_te * r_behind * phi * phi);
                    QD_DIAG("%d,%d,%.1f,%s,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%+.4f\n", R.s2, prev, thd, port ? "R" : "T",
                            std::abs(S), std::arg(S) * 180.0 / QPI, std::abs(S_ref), std::arg(S_ref) * 180.0 / QPI,
                            std::abs(S - S_ref), 20.0 * std::log10(std::abs(S) / std::abs(S_ref)));
                    INFO(R.name << (ctrl ? ", bare-entered control" : "") << ", series, theta_n " << thd
                                << (port ? ", reflect" : ", transmit"));
                    REQUIRE(rt_bit(A1.resolved_type[0], 2));
                    CHECK(std::abs(S - S_ref) < 1.0e-9);
                }

    // ------------------------------------------------------------------------------------------
    // The continuation direction
    // ------------------------------------------------------------------------------------------
    // The outer pair decides it, and the outer pair is host | air. The ray is tilted in the xy
    // plane so the direction the segment has and the angle the hit reports agree.
    QD_DIAG("\n# diag: lined on both faces, the direction the transmitted ray leaves in, degrees from the\n");
    QD_DIAG("# normal, against the same exit with a clean previous-medium word\n");
    QD_DIAG("stack,theta_n,out_deg,bare_deg,d_dir\n");
    for (const auto &R : rows)
        for (double thd : {10.0, 20.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            Rsu<double> C[2];
            for (int k = 0; k < 2; ++k)
            {
                C[k] = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, R.s2, HOST, k ? 0 : R.s1, HOST, 0, L / std::cos(th), 0.5,
                                     0.0, QPI / 2.0 - th, 1.5);
                C[k].orig(0, 0) = C[k].fbs(0, 0) - L;
                C[k].orig(0, 1) = -L * std::tan(th);
                C[k].dest(0, 0) = C[k].fbs(0, 0) + 0.5 * std::cos(th);
                C[k].dest(0, 1) = 0.5 * std::sin(th);
                C[k].has_path_dirN = true;
                C[k].path_dirN.zeros(1, 3);
                C[k].path_dirN(0, 0) = std::cos(th), C[k].path_dirN(0, 1) = std::sin(th);
                C[k].run();
            }
            auto deg = [&](const Rsu<double> &X)
            { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
            double d_dir = 0.0;
            for (arma::uword k = 0; k < 3; ++k)
                d_dir = std::max(d_dir, std::abs((double)C[0].path_dirN(0, k) - (double)C[1].path_dirN(0, k)));
            QD_DIAG("%s,%.1f,%.4f,%.4f,%.3e\n", R.name, thd, deg(C[0]), deg(C[1]), d_dir);
            INFO(R.name << ", direction, theta_n " << thd);
            CHECK(d_dir == 0.0);
            CHECK(deg(C[1]) > thd + 1.0); // host to air bends away from the normal
        }

    // ------------------------------------------------------------------------------------------
    // A resolved ray at a sheet
    // ------------------------------------------------------------------------------------------
    // A reflection that collapsed the host's bounce series flags the ray as resolved, and the
    // close-out of that leg must not apply the series a second time. A bare exit does not; a
    // sheet event did. Measured as the resolving run over the declined one, with the flag set and
    // without it, for a glued exit, an embedded sheet and an ordered stack.
    QD_DIAG("\n# diag: a resolved ray at a sheet. S is what the host's series applies at the event; it has\n");
    QD_DIAG("# to be one for a resolved ray, whatever it is for an unresolved one\n");
    QD_DIAG("event,host,prev,theta_n,|S_unresolved|,|S_resolved|,|S_resolved - 1|,cur,rtype\n");
    struct Ev
    {
        const char *name;
        uint8_t ot;
        unsigned nH;
        int m1, m2, prev;
    };
    for (int host : {HOST, CLEAN})
    {
        const std::vector<Ev> evs = {
            {"glued exit", OT_GLUED_EXIT, 2u, SHEET, host, 0},
            {"glued exit", OT_GLUED_EXIT, 2u, SHEET, host, SHEET},
            {"embedded  ", OT_SHEET_ISO, 1u, SHEET, 0, 0},
            {"stack     ", OT_GLUED_ENTRY, 2u, SHEET, NEAR, 0},
        };
        for (const auto &E : evs)
            for (double thd : {0.0, 40.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> S[2];
                bool same[2] = {false, false}; // the two runs wrote the same field, bit for bit
                Rsu<double> last;
                for (int res = 0; res < 2; ++res)
                {
                    std::complex<double> v[2];
                    for (int k = 0; k < 2; ++k)
                    {
                        auto C = make1<double>(mtl, 4, E.ot, E.nH, E.m1, E.m2, E.prev, (int)enc(host, res != 0), 0, L,
                                               0.5, 0.5, QPI / 2.0 - th, k ? 0.0 : 1.5);
                        C.run();
                        v[k] = cpx(C.xprmat, 0, 0);
                        last = C;
                    }
                    S[res] = v[1] / v[0];
                    same[res] = (v[1] == v[0]);
                }
                QD_DIAG("%s,%d,%d,%.1f,%.9f,%.9f,%.3e,%d,%u\n", E.name, host, E.prev, thd, std::abs(S[0]), std::abs(S[1]),
                        std::abs(S[1] - 1.0), mat_of(last.cur_out(0)), (unsigned)last.resolved_type[0]);
                INFO(E.name << ", host " << host << ", prev " << E.prev << ", theta_n " << thd);
                CHECK(std::abs(S[0] - 1.0) > 1.0e-3); // the row has a series to apply
                CHECK(same[1]);                       // and a resolved ray does not get it
            }
    }
}

TEST_CASE("Layered wall - a glued exit behind a bare-entered second layer exits into air")
{
    // A glued exit, solid | sheet | air, took what lies beyond its sheet from the previous-medium
    // word wherever that word names a medium. For a wall of two solid layers with a coating on
    // its far side, air | A | B | coating | air, the leg in B was entered from A, so the event
    // resolved B | coating | A and left the ray "in A" while it is in air.
    //
    // Beyond the sheet of a glued exit is air. The medium the host was entered from is behind
    // the ray and is the mirror behind it in the host's series, nothing else.
    //
    // Four instruments. Per event, with no path_dir: the exit with A in the previous-medium word,
    // with and without bit 14, against the composite with air beyond and against the same exit
    // with a clean word, bitwise, both ports, scalar and EM, the state words and the direction.
    // The host's series as the ratio of the resolving run over the declined one, with the coating
    // backed by air and by A as two references. The chain, entry from ray_mesh_interact, the
    // transition A -> B and the glued exit fed the words the transition wrote, with path_dir:
    // against the single pass through A and B times the composite, and, with B's series
    // resolved, against the characteristic matrix of {B, coating} between the half-spaces A and
    // air. And the next wall after the exit, which has to be entered from air.
    //
    // The matrix of the whole wall, air | A | B | coating | air, is recorded and not asserted:
    // two traced solids are resolved as two series, and the mirror behind B is A as a half-space
    // where the whole wall has A as a layer backed by air.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, CLEAN = 6, SHEET = 3, WALL = 4;
    const uint8_t OT_GLUED_EXIT = 69;
    const std::complex<double> e_h = diag_eps(HOST, f_hz), e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<int> firsts = {2, 1, 4}; // eps 5.24, eps 5.24 - 0.63j, eps 9.0 - 1.20j

    auto exitev = [&](int itype, int prev, double th, double eps)
    {
        auto C = make1<double>(mtl, itype, OT_GLUED_EXIT, 2u, SHEET, HOST, prev, HOST, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.run();
        return C;
    };

    // ------------------------------------------------------------------------------------------
    // The composite and the state words, both ports, scalar
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: layered wall, the glued exit of B | coating | air with A in the previous-medium word.\n");
    QD_DIAG("# B eps 1.50 - 0.40j over 100 mm, coating eps 5.24 - 0.63j 6 mm, scalar, no path_dir, host series\n");
    QD_DIAG("# declined. A is 2 (eps 5.24), 1 (eps 5.24 - 0.63j) or 4 (eps 9.0 - 1.20j); flag is bit 14 of\n");
    QD_DIAG("# the word. d_ref against the composite with air beyond, d_bare against the same exit with a\n");
    QD_DIAG("# clean previous-medium word, c_ref that control against the reference\n");
    QD_DIAG("A,flag,theta_n,port,meas_dB,ref_dB,d_ref,d_bare,c_ref,prev,cur,resolved,buf,rtype\n");
    for (int first : firsts)
        for (int flag = 0; flag < 2; ++flag)
            for (double thd : {0.0, 20.0, 40.0})
            {
                double th = thd * QPI / 180.0, ct = std::cos(th);
                int pw = first | (flag ? 0x4000 : 0);
                DiagSheet D = diag_sheet_composite(e_h, e_s, e_air, thk, ct, f_hz, true, true);
                std::complex<double> close = diag_close(e_h, L, th, f_hz);
                for (int port = 0; port < 2; ++port)
                {
                    int itype = port ? 3 : 4;
                    auto A = exitev(itype, pw, th, 1.5);
                    auto B = exitev(itype, 0, th, 1.5);
                    std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
                    std::complex<double> ref = (port ? D.R_te : D.T_te) * close;
                    double d_ref = std::abs(a - ref) / std::abs(ref), d_bare = std::abs(a - b) / std::abs(b);
                    double c_ref = std::abs(b - ref) / std::abs(ref);
                    QD_DIAG("%d,%d,%.1f,%s,%.4f,%.4f,%.3e,%.3e,%.3e,%d,%d,%d,%d,%u\n", first, flag, thd, port ? "R" : "T",
                            dB(std::norm(a)), dB(std::norm(ref)), d_ref, d_bare, c_ref, mat_of(A.prev_out(0)),
                            mat_of(A.cur_out(0)), (int)flag_of(A.cur_out(0)), (int)A.buf_out(0),
                            (unsigned)A.resolved_type[0]);
                    INFO("A " << first << ", flag " << flag << ", theta_n " << thd << (port ? ", reflect" : ", transmit"));
                    CHECK(c_ref < 1.0e-9); // the reference is the one the bare-entered exit sits on
                    CHECK(d_ref < 1.0e-9); // air beyond the coating
                    CHECK(a == b);         // and the exit does not know what the host was entered from
                    if (port)
                    {
                        check_state(A, SHEET, enc(HOST, false), 0x4000); // a reflection stays in the host; its series was declined, so it is not resolved but followed
                        check_rtype(A, 207);
                    }
                    else
                    {
                        check_state(A, SHEET, 0, 0); // the ray is in air, behind the sheet it crossed
                        check_rtype(A, 71);
                    }
                }
            }

    // ------------------------------------------------------------------------------------------
    // The same in EM, per entry of the matrix
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: layered wall, EM. Gain of the glued exit with A in the previous-medium word against\n");
    QD_DIAG("# the same exit with a clean word; d_x is the largest difference over the eight matrix entries\n");
    QD_DIAG("A,theta_n,port,gain_dB,bare_dB,d_gain_dB,d_x,cur\n");
    for (int first : firsts)
        for (double thd : {0.0, 20.0, 40.0})
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                auto A = exitev(port ? 0 : 1, first, th, 1.5);
                auto B = exitev(port ? 0 : 1, 0, th, 1.5);
                double d_x = 0.0;
                for (arma::uword k = 0; k < 8; ++k)
                    d_x = std::max(d_x, std::abs((double)A.xprmat(k, 0) - (double)B.xprmat(k, 0)));
                QD_DIAG("%d,%.1f,%s,%.4f,%.4f,%+.4f,%.3e,%d\n", first, thd, port ? "R" : "T", dB((double)A.gain(0)),
                        dB((double)B.gain(0)), dB((double)A.gain(0)) - dB((double)B.gain(0)), d_x, mat_of(A.cur_out(0)));
                INFO("A " << first << ", EM, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                CHECK(d_x == 0.0);
                CHECK((double)A.gain(0) == (double)B.gain(0));
                CHECK(mat_of(A.cur_out(0)) == (port ? HOST : 0));
            }

    // ------------------------------------------------------------------------------------------
    // The host's series: the coating ahead backed by air, A behind as a half-space
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: layered wall, the host's series at the glued exit. S_applied is the resolving run over\n");
    QD_DIAG("# the declined one. S_air has the coating as a layer backed by air and A behind as a half-space;\n");
    QD_DIAG("# S_A backs the coating with A\n");
    QD_DIAG("A,theta_n,port,|S_applied|,arg_S_applied,|S_air|,arg_S_air,|dS_air|,|dS_A|,sep\n");
    for (int first : firsts)
        for (double thd : {0.0, 20.0, 40.0})
            for (int port = 0; port < 2; ++port)
            {
                double th = thd * QPI / 180.0;
                int itype = port ? 3 : 4;
                auto A1 = exitev(itype, first, th, 0.0);
                auto A0 = exitev(itype, first, th, 1.5);
                std::complex<double> S = cpx(A1.xprmat, 0, 0) / cpx(A0.xprmat, 0, 0);
                std::complex<double> cs(std::cos(th), 0.0), e_a = diag_eps(first, f_hz);
                std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                std::complex<double> r_behind = diag_r_slabside(e_h, e_a, cs);
                std::complex<double> r_air = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_A = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_a, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> S_air = 1.0 / (1.0 - r_air * r_behind * phi * phi);
                std::complex<double> S_A = 1.0 / (1.0 - r_A * r_behind * phi * phi);
                QD_DIAG("%d,%.1f,%s,%.9f,%+7.3f,%.9f,%+7.3f,%.3e,%.3e,%.3e\n", first, thd, port ? "R" : "T", std::abs(S),
                        std::arg(S) * 180.0 / QPI, std::abs(S_air), std::arg(S_air) * 180.0 / QPI, std::abs(S - S_air),
                        std::abs(S - S_A), std::abs(S_air - S_A));
                INFO("A " << first << ", series, theta_n " << thd << (port ? ", reflect" : ", transmit"));
                REQUIRE(rt_bit(A1.resolved_type[0], 2));
                CHECK(std::abs(S_air - S_A) > 1.0e-3); // the two backings can be told apart
                CHECK(std::abs(S - S_air) < 1.0e-9);
            }

    // ------------------------------------------------------------------------------------------
    // The continuation direction, no recovery involved
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: layered wall, the direction the transmitted ray leaves in, degrees from the normal,\n");
    QD_DIAG("# against the same exit with a clean previous-medium word\n");
    QD_DIAG("A,theta_n,out_deg,bare_deg,d_dir\n");
    for (int first : firsts)
        for (double thd : {10.0, 20.0, 40.0})
        {
            double th = thd * QPI / 180.0;
            Rsu<double> C[2];
            for (int k = 0; k < 2; ++k)
            {
                C[k] = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, SHEET, HOST, k ? 0 : first, HOST, 0, L / std::cos(th), 0.5,
                                     0.0, QPI / 2.0 - th, 1.5);
                C[k].orig(0, 0) = C[k].fbs(0, 0) - L;
                C[k].orig(0, 1) = -L * std::tan(th);
                C[k].dest(0, 0) = C[k].fbs(0, 0) + 0.5 * std::cos(th);
                C[k].dest(0, 1) = 0.5 * std::sin(th);
                C[k].has_path_dirN = true;
                C[k].path_dirN.zeros(1, 3);
                C[k].path_dirN(0, 0) = std::cos(th), C[k].path_dirN(0, 1) = std::sin(th);
                C[k].run();
            }
            auto deg = [&](const Rsu<double> &X)
            { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
            double d_dir = 0.0;
            for (arma::uword k = 0; k < 3; ++k)
                d_dir = std::max(d_dir, std::abs((double)C[0].path_dirN(0, k) - (double)C[1].path_dirN(0, k)));
            QD_DIAG("%d,%.1f,%.4f,%.4f,%.3e\n", first, thd, deg(C[0]), deg(C[1]), d_dir);
            INFO("A " << first << ", direction, theta_n " << thd);
            CHECK(d_dir == 0.0);
            CHECK(deg(C[1]) > thd + 1.0); // host to air bends away from the normal
        }

    // ------------------------------------------------------------------------------------------
    // The chain, air | A | B | coating | air, with path_dir
    // ------------------------------------------------------------------------------------------
    struct Ent
    {
        std::complex<double> te;
        double c, s; // traced cosine and sine in the first solid
    };
    auto entry = [&](int first, double thd) -> Ent
    {
        arma::mat orig, dest, fbs, sbs;
        diag_ray_at(thd, orig, dest);
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                        &path_dirN, true, (arma::u32_vec *)nullptr);
        REQUIRE(path_dirN.n_rows == 1);
        Ent E;
        E.te = std::complex<double>(xprmatN(0, 0), xprmatN(1, 0));
        double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
        E.c = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
        E.s = std::sqrt(std::max(0.0, 1.0 - E.c * E.c));
        return E;
    };
    // One event closing a leg of normal thickness Ln along (dx, dy). A media-to-media event takes
    // its plane normal from the second face, reversed to oppose the ray.
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double Ln,
                     double th, double eps, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, Ln, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        if (ot == OT_M2M_M1)
            C.normals(0, 3) = -1.0;
        C.run();
        return C;
    };
    auto port_p = [](std::complex<double> ea, std::complex<double> eb, std::complex<double> ca)
    {
        std::complex<double> te, tm;
        diag_port(ea, eb, ca, true, te, tm);
        return std::norm(te);
    };
    auto decay = [&](std::complex<double> e, std::complex<double> c, double Ln)
    { return std::exp(2.0 * k0 * std::imag(std::sqrt(e) * c) * Ln); };
    auto clampr = [](std::complex<double> r)
    { return (std::abs(r) > 1.0) ? r / std::abs(r) : r; };

    QD_DIAG("\n# diag: layered wall, the chain air | A 50 mm | B 100 mm | coating 6 mm | air, scalar, path_dir\n");
    QD_DIAG("# supplied, the glued exit fed the words the transition A -> B wrote. Series declined: d_T and\n");
    QD_DIAG("# d_R against the single pass through A and B times the composite B | coating | air at the exact\n");
    QD_DIAG("# cosine, c_T the same for the exit with a clean previous-medium word, d_twin the two exits\n");
    QD_DIAG("# against each other. out_deg is the direction the ray leaves in and d_dir its distance from\n");
    QD_DIAG("# the clean-word exit's. prev, cur, buf, rtype after the transmit pass\n");
    QD_DIAG("A,B,theta_ext,ref_T_dB,d_T,d_R,c_T,d_twin,out_deg,d_dir,prev,cur,buf,rtype\n");
    const std::vector<double> ext = {10.0, 40.0, 60.0, 75.0, 85.0};
    for (int host : {HOST, CLEAN})
        for (int first : {2, 1})
            for (double thd : ext)
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(first, thd);
                auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, 1.5, E.c, E.s);
                double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                int pv = (int)T.prev_out(0), cu = (int)T.cur_out(0), bu = (int)T.buf_out(0);
                auto Xt = event(4, OT_GLUED_EXIT, 2u, SHEET, host, pv, cu, bu, L2, th, 1.5, dx, dy);
                auto Xr = event(3, OT_GLUED_EXIT, 2u, SHEET, host, pv, cu, bu, L2, th, 1.5, dx, dy);
                auto Ct = event(4, OT_GLUED_EXIT, 2u, SHEET, host, 0, host, 0, L2, th, 1.5, dx, dy);

                DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e_air, thk, c2, f_hz, true, true);
                double front = port_p(e_air, e1, c0) * port_p(e1, e2, c1) * decay(e1, c1, L1);
                double pre = std::norm(E.te * cpx(T.xprmat, 0, 0));
                double ref_T = front * decay(e2, c2, L2) * std::norm(Ref.T_te);
                // A reflect pass at a sheet closes the host leg as the transmit pass does
                double ref_R = front * decay(e2, c2, L2) * std::norm(Ref.R_te);
                double ch_T = pre * std::norm(cpx(Xt.xprmat, 0, 0)), ch_R = pre * std::norm(cpx(Xr.xprmat, 0, 0));
                double cl_T = pre * std::norm(cpx(Ct.xprmat, 0, 0));
                double d_dir = 0.0;
                for (arma::uword k = 0; k < 3; ++k)
                    d_dir = std::max(d_dir, std::abs((double)Xt.path_dirN(0, k) - (double)Ct.path_dirN(0, k)));
                double out_deg = std::atan2(std::abs((double)Xt.path_dirN(0, 1)), std::abs((double)Xt.path_dirN(0, 0))) * 180.0 / QPI;
                QD_DIAG("%d,%d,%.1f,%.6f,%+.6f,%+.6f,%+.6f,%+.6f,%.4f,%.3e,%d,%d,%d,%u\n", first, host, thd, dB(ref_T),
                        dB(ch_T) - dB(ref_T), dB(ch_R) - dB(ref_R), dB(cl_T) - dB(ref_T), dB(ch_T) - dB(cl_T), out_deg,
                        d_dir, mat_of(Xt.prev_out(0)), mat_of(Xt.cur_out(0)), (int)Xt.buf_out(0),
                        (unsigned)Xt.resolved_type[0]);
                INFO("chain, A " << first << ", B " << host << ", theta_ext " << thd);
                REQUIRE(mat_of(T.prev_out(0)) == first);
                CHECK(std::abs(dB(cl_T) - dB(ref_T)) < 1.0e-6); // the control: the clean-word exit on the reference
                CHECK(std::abs(dB(ch_T) - dB(ref_T)) < 1.0e-6);
                CHECK(std::abs(dB(ch_R) - dB(ref_R)) < 1.0e-6);
                CHECK(std::abs(dB(ch_T) - dB(cl_T)) < 1.0e-9);
                CHECK(d_dir < 1.0e-9); // the direction is that of B -> air
                check_state(Xt, SHEET, 0, 0);
                check_rtype(Xt, 71);
                check_state(Xr, SHEET, enc(host, false), 0x4000); // declined: the reflected ray is the tracer's, followed
                check_rtype(Xr, 207);
            }

    QD_DIAG("\n# diag: layered wall, the same chain with B's series resolved, against the characteristic\n");
    QD_DIAG("# matrix of {B, coating} between the half-spaces A and air times the single pass through A.\n");
    QD_DIAG("# clamp_dB is what the clamp on a mirror's reflectance costs. d_wall is recorded, not asserted:\n");
    QD_DIAG("# both series resolved against the matrix of the whole wall, air | A | B | coating | air\n");
    QD_DIAG("A,B,theta_ext,chain_dB,ref_dB,d_dB,|r_sheet|,clamp_dB,wall_dB,d_wall,rtype\n");
    for (int host : {HOST, CLEAN})
        for (int first : {2, 1})
            for (double thd : ext)
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(first, thd);
                double chain[2];
                Rsu<double> X;
                for (int k = 0; k < 2; ++k) // k = 1 resolves A's series as well
                {
                    auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, k ? 0.0 : 1.5, E.c, E.s);
                    X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, (int)T.prev_out(0), (int)T.cur_out(0), (int)T.buf_out(0),
                              L2, th, 0.0, (double)T.path_dirN(0, 0), (double)T.path_dirN(0, 1));
                    chain[k] = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                }
                DiagSheet M = diag_cm_stack(e1, U, {{e2, U, L2}, {e_s, U, thk}}, e_air, U, c1, f_hz, false, true, 0.0);
                double ref = port_p(e_air, e1, c0) * decay(e1, c1, L1) * std::norm(M.T_te);
                DiagSheet W = diag_cm_stack(e_air, U, {{e1, U, L1}, {e2, U, L2}, {e_s, U, thk}}, e_air, U, c0, f_hz, false,
                                            true, 0.0);

                std::complex<double> r_s = diag_cm_stack(e2, U, {{e_s, U, thk}}, e_air, U, c2, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_b = diag_r_slabside(e2, e1, c2);
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -1.0) * k0 * std::sqrt(e2) * c2 * L2);
                double clamp_dB = 20.0 * std::log10(std::abs(1.0 - r_b * r_s * ph * ph) /
                                                    std::abs(1.0 - clampr(r_b) * clampr(r_s) * ph * ph));
                QD_DIAG("%d,%d,%.1f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%.6f,%+.6f,%u\n", first, host, thd, dB(chain[0]), dB(ref),
                        dB(chain[0]) - dB(ref), std::abs(r_s), clamp_dB, dB(std::norm(W.T_te)),
                        dB(chain[1]) - dB(std::norm(W.T_te)), (unsigned)X.resolved_type[0]);
                INFO("chain resolved, A " << first << ", B " << host << ", theta_ext " << thd);
                CHECK(std::abs(dB(chain[0]) - dB(ref) - clamp_dB) < 1.0e-6);
                if (!(host == HOST && thd == 85.0))
                    CHECK(clamp_dB == 0.0); // the clamp is engaged on the lossy host at 85 deg only
            }

    // ------------------------------------------------------------------------------------------
    // The next wall after the exit
    // ------------------------------------------------------------------------------------------
    // The ray is in air. The next wall reflects off its front face, is entered through it and is
    // left through its back face exactly as by a ray with clean state words.
    QD_DIAG("\n# diag: layered wall, the next wall (eps 9.0 - 1.20j, 100 mm) after the glued exit, fed the words\n");
    QD_DIAG("# the exit wrote, against the same three events with clean words. Words as prev/cur/buf\n");
    QD_DIAG("A,theta_n,event,gain_dB,clean_dB,words_in,words_out,clean_out,rtype,clean_rtype\n");
    for (int first : firsts)
        for (double thd : {0.0, 10.0})
        {
            double th = thd * QPI / 180.0;
            auto X = exitev(4, first, th, 1.5);
            int w[3] = {(int)X.prev_out(0), (int)X.cur_out(0), (int)X.buf_out(0)};
            int z[3] = {0, 0, 0};
            const char *names[3] = {"reflect", "enter  ", "leave  "};
            for (int ev = 0; ev < 3; ++ev)
            {
                Rsu<double> R[2];
                int in[2][3];
                for (int k = 0; k < 2; ++k)
                {
                    int *s = k ? z : w;
                    in[k][0] = s[0], in[k][1] = s[1], in[k][2] = s[2];
                    if (ev == 2)
                        R[k] = make1<double>(mtl, 4, OT_EXIT, 1u, WALL, 0, s[0], s[1], s[2], L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                    else
                        R[k] = make1<double>(mtl, ev ? 4 : 3, OT_ENTRY, 1u, WALL, 0, s[0], s[1], s[2], 0.5, L, 0.5,
                                             QPI / 2.0 - th, 1.5);
                    R[k].run();
                }
                QD_DIAG("%d,%.1f,%s,%.4f,%.4f,%d/%d/%d,%d/%d/%d,%d/%d/%d,%u,%u\n", first, thd, names[ev],
                        dB((double)R[0].gain(0)), dB((double)R[1].gain(0)), in[0][0], in[0][1], in[0][2],
                        (int)R[0].prev_out(0), (int)R[0].cur_out(0), (int)R[0].buf_out(0), (int)R[1].prev_out(0),
                        (int)R[1].cur_out(0), (int)R[1].buf_out(0), (unsigned)R[0].resolved_type[0],
                        (unsigned)R[1].resolved_type[0]);
                INFO("next wall, A " << first << ", theta_n " << thd << ", " << names[ev]);
                CHECK(cpx(R[0].xprmat, 0, 0) == cpx(R[1].xprmat, 0, 0));
                CHECK((int)R[0].resolved_type[0] == (int)R[1].resolved_type[0]);
                CHECK((int)R[0].cur_out(0) == (int)R[1].cur_out(0));
                CHECK((int)R[0].buf_out(0) == (int)R[1].buf_out(0));
                if (ev == 1) // the words the entry wrote feed the exit
                {
                    w[0] = (int)R[0].prev_out(0), w[1] = (int)R[0].cur_out(0), w[2] = (int)R[0].buf_out(0);
                    z[0] = (int)R[1].prev_out(0), z[1] = (int)R[1].cur_out(0), z[2] = (int)R[1].buf_out(0);
                }
            }
        }
}

TEST_CASE("Recovery behind a sheet - the leg after a sheet crossing at the exact in-medium cosine")
{
    // A traced ray holds a real direction inside a lossy host while the propagation cosine there
    // is complex, and the recovery returns the complex value by inverting the refraction map of
    // the pair previous medium -> current medium. After a sheet crossing the previous-medium word
    // holds the sheet, and the recovery declines: the exit interface, the host's series, the
    // traversal and a sheet composite on that leg are then built at the real projection, while
    // the entry was evaluated at the exact cosine.
    //
    // Two instruments. End to end, the chain air | S | host | air and air | S | host | S | air,
    // entered through a glued sheet and assembled from what ray_state_update writes, against the
    // characteristic matrix of the whole stack, in power: the displaced air and the free-space
    // reference are pure phases. And per event, the same leg with a sheet in the previous-medium
    // word against the same leg with a clean word, host series declined, which isolates what the
    // decline costs on one event and needs no reference.
    //
    // A lossless host has a real cosine and nothing to recover, so its rows sit on the reference
    // whatever the recovery does and are the control on the chain and on the reference.
    const double f_hz = FRQ, thk = 0.006, L = 0.10;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int LOSSY = 5, CLEAN = 6, SHEET = 3;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71, OT_GLUED_EXIT = 69;
    const std::complex<double> e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    // Glued entry from air at the external angle, tilted in the xy plane so the segment and the
    // reported angle agree. Returns the event; path_dirN is the direction the tracer then holds.
    auto glued_entry = [&](int host, double th)
    {
        auto C = make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, SHEET, host, 0, 0, 0, 0.5, 0.5, 0.0, QPI / 2.0 - th, 1.5);
        C.orig(0, 0) = C.fbs(0, 0) - 0.5 * std::cos(th);
        C.orig(0, 1) = -0.5 * std::sin(th);
        C.dest(0, 0) = C.fbs(0, 0) + 0.5 * std::cos(th);
        C.dest(0, 1) = 0.5 * std::sin(th);
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        C.run();
        return C;
    };
    // A leg of normal thickness L inside the host, closed by the given event, along a direction
    auto leg = [&](uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double th, double eps,
                   double dx, double dy)
    {
        auto C = make1<double>(mtl, 4, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.run();
        return C;
    };

    QD_DIAG("\n# diag: recovery behind a sheet, end to end. air | S | host 100 mm | air (coated) and\n");
    QD_DIAG("# air | S | host 100 mm | S | air (lined), S eps 5.24 - 0.63j 6 mm, scalar, path_dir supplied,\n");
    QD_DIAG("# against the characteristic matrix of the whole stack. Host 6 is lossless and the control.\n");
    QD_DIAG("# r_sheet is the reflection of a sheet backed by air seen from the host; where it exceeds\n");
    QD_DIAG("# one the series clamps it, and clamp_dB is what that costs\n");
    QD_DIAG("chain,host,theta_ext,ct_event,ct_map,chain_dB,ref_dB,d_dB,|r_sheet|,clamp_dB,rtype\n");
    for (int host : {CLEAN, LOSSY})
        for (int lined = 0; lined < 2; ++lined)
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                const std::complex<double> e_h = diag_eps(host, f_hz);
                auto E1 = glued_entry(host, th);
                double dx = (double)E1.path_dirN(0, 0), dy = (double)E1.path_dirN(0, 1);
                auto E2 = lined ? leg(OT_GLUED_EXIT, 2u, SHEET, host, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                      (int)E1.buf_out(0), th, 0.0, dx, dy)
                                : leg(OT_EXIT, 1u, host, 0, (int)E1.prev_out(0), (int)E1.cur_out(0),
                                      (int)E1.buf_out(0), th, 0.0, dx, dy);
                std::complex<double> chain = cpx(E1.xprmat, 0, 0) * cpx(E2.xprmat, 0, 0);

                std::vector<DiagLayer> lay = {{e_s, U, thk}, {e_h, U, L}};
                if (lined)
                    lay.push_back({e_s, U, thk});
                DiagSheet Ref = diag_cm_stack(e_air, U, lay, e_air, U, std::complex<double>(std::cos(th), 0.0), f_hz,
                                              false, true, 0.0);
                double chain_dB = dB(std::norm(chain)), ref_dB = dB(std::norm(Ref.T_te));
                double ct_map = diag_traced_ct(e_h, th);

                // The host's series clamps a mirror's reflectance into [0, 1]. Seen from a lossy
                // host close to its critical angle a sheet backed by air reflects more than that,
                // and the clamp is then the whole of what separates the chain from the matrix.
                // What it costs is formed from the factorized series, exact against clamped.
                std::complex<double> cs = diag_ct_exact(e_h, th);
                std::complex<double> r_s = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, false, true, 0.0).R_te;
                std::complex<double> r_c = (std::abs(r_s) > 1.0) ? r_s / std::abs(r_s) : r_s;
                std::complex<double> r_a = diag_r_slabside(e_h, e_air, cs);
                std::complex<double> ph = std::exp(std::complex<double>(0.0, -1.0) * (2.0 * QPI * f_hz / C0) *
                                                   std::sqrt(e_h) * cs * L);
                double clamp_dB = lined ? 20.0 * std::log10(std::abs(1.0 - r_s * r_s * ph * ph) / std::abs(1.0 - r_c * r_c * ph * ph))
                                        : 20.0 * std::log10(std::abs(1.0 - r_a * r_s * ph * ph) / std::abs(1.0 - r_a * r_c * ph * ph));

                QD_DIAG("%s,%d,%.1f,%.9f,%.9f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%u\n", lined ? "lined " : "coated", host, thd,
                        std::abs(dx), ct_map, chain_dB, ref_dB, chain_dB - ref_dB, std::abs(r_s), clamp_dB,
                        (unsigned)E2.resolved_type[0]);
                INFO((lined ? "lined" : "coated") << ", host " << host << ", theta_ext " << thd);
                CHECK(std::abs(std::abs(dx) - ct_map) < 1.0e-9); // the entry wrote the map the recovery inverts
                CHECK(std::abs(chain_dB - ref_dB - clamp_dB) < 1.0e-6);
                if (!(host == LOSSY && thd == 85.0))
                    CHECK(clamp_dB == 0.0); // the clamp is engaged on the one row and nowhere else
            }

    QD_DIAG("\n# diag: recovery behind a sheet, per event. One leg of the lossy host (eps 1.50 - 0.40j,\n");
    QD_DIAG("# 100 mm) along the direction a tracer holds after entering from air, series declined, with\n");
    QD_DIAG("# a sheet in the previous-medium word against the same leg with a clean word. The bare\n");
    QD_DIAG("# exit is the far face of a coated wall and of a wall with a sheet inside it\n");
    QD_DIAG("event,theta_ext,sheet_dB,clean_dB,d_dB,d_rel\n");
    struct Ev
    {
        const char *name;
        uint8_t ot;
        unsigned nH;
        int m1, m2;
    };
    const std::vector<Ev> evs = {
        {"bare exit ", OT_EXIT, 1u, LOSSY, 0},
        {"glued exit", OT_GLUED_EXIT, 2u, SHEET, LOSSY},
        {"embedded  ", OT_SHEET_ISO, 1u, SHEET, 0},
    };
    const std::complex<double> e_l = diag_eps(LOSSY, f_hz);
    for (const auto &E : evs)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            double ct = diag_traced_ct(e_l, th), st = std::sqrt(1.0 - ct * ct);
            auto A = leg(E.ot, E.nH, E.m1, E.m2, SHEET, LOSSY, 0, th, 1.5, ct, st);
            auto B = leg(E.ot, E.nH, E.m1, E.m2, 0, LOSSY, 0, th, 1.5, ct, st);
            std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
            double d_rel = std::abs(a - b) / std::abs(b);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%+.6f,%.3e\n", E.name, thd, dB(std::norm(a)), dB(std::norm(b)),
                    dB(std::norm(a)) - dB(std::norm(b)), d_rel);
            INFO(E.name << ", theta_ext " << thd);
            CHECK(d_rel < 1.0e-12);
        }
    // The recovered cosine does not depend on which lossless medium the wave came from: the
    // equation reads the entry medium only through the real transverse invariant. A leg entered
    // from a dense lossless solid, named in the previous-medium word, against the same leg with
    // a sheet there, where the recovery cannot name the entry medium. The last two rows are past
    // what air could launch, invariant above one. Such a wave cannot leave into air either, so
    // those two are measured on a face that admits it, the transition into the dense solid. The
    // total reflection at the air face is not used: at a recovered cosine its evanescent root
    // has an argument that is real and negative up to rounding, and the branch it takes is not
    // defined by the media.
    QD_DIAG("\n# diag: recovery behind a sheet, the entry medium. A bare exit out of the lossy host along\n");
    QD_DIAG("# the direction a tracer holds after entering from a lossless solid, eps 5.24, with that solid\n");
    QD_DIAG("# in the previous-medium word against a sheet there. p is the transverse invariant; past one\n");
    QD_DIAG("# the face is host | dense solid\n");
    QD_DIAG("p,theta_entry,face,ct_traced,named_dB,named_arg,sheet_dB,sheet_arg,d_rel\n");
    const int DENSE = 2;
    const std::complex<double> e_d = diag_eps(DENSE, f_hz);
    for (double pinv : {0.2, 0.8, 1.2, 1.45})
    {
        double s1 = std::sqrt(pinv / std::real(e_d)), th1 = std::asin(s1);
        std::complex<double> eps_r = e_d / e_l;
        std::complex<double> c2 = std::sqrt(1.0 - eps_r * s1 * s1);
        double eta = std::sqrt(std::abs(eps_r)), r = std::real(c2);
        double ct = r / std::sqrt((eta * s1) * (eta * s1) + r * r), st = std::sqrt(1.0 - ct * ct);
        bool m2m = pinv > 1.0;
        auto A = m2m ? leg(OT_M2M_M1, 2u, LOSSY, DENSE, SHEET, LOSSY, 0, th1, 1.5, ct, st)
                     : leg(OT_EXIT, 1u, LOSSY, 0, SHEET, LOSSY, 0, th1, 1.5, ct, st);
        auto B = m2m ? leg(OT_M2M_M1, 2u, LOSSY, DENSE, DENSE, LOSSY, 0, th1, 1.5, ct, st)
                     : leg(OT_EXIT, 1u, LOSSY, 0, DENSE, LOSSY, 0, th1, 1.5, ct, st);
        std::complex<double> a = cpx(A.xprmat, 0, 0), b = cpx(B.xprmat, 0, 0);
        double d_rel = std::abs(a - b) / std::abs(b);
        QD_DIAG("%.2f,%.3f,%s,%.9f,%.6f,%+.4f,%.6f,%+.4f,%.3e\n", pinv, th1 * 180.0 / QPI, m2m ? "dense" : "air", ct,
                dB(std::norm(b)), std::arg(b) * 180.0 / QPI, dB(std::norm(a)), std::arg(a) * 180.0 / QPI, d_rel);
        INFO("entry medium, p " << pinv);
        REQUIRE(std::abs(b) > 0.0);
        CHECK(d_rel < 1.0e-12);
    }
}

TEST_CASE("Measurement - a host entered from a lossy solid across a bare face")
{
    // A measurement, taken before any decision. The recovery inverts the refraction map of the
    // pair previous medium -> current medium, and every fixture that pins it enters from air or
    // from a lossless solid. Here the host is entered from a LOSSY solid across a bare shared
    // face, air | first | host | air, so the previous-medium word names a lossy medium.
    //
    // A ray that starts in air keeps a real transverse wavenumber across parallel faces however
    // lossy the layers are, so the exact cosine in the host is sqrt(1 - sin^2(theta_ext) / eps_h)
    // and the first solid drops out of it. The reference is the single pass through the stack at
    // those cosines, in power, with every series declined (eps 1.5) so that nothing but the three
    // ports and the two traversals is in the chain.
    //
    // The last event is run twice along the same direction: with the state words the transition
    // wrote, which name the first solid, and with a clean previous-medium word, where the
    // recovery solves for a real invariant with no medium named.
    const double f_hz = FRQ, L1 = 0.05, L2 = 0.10, k0 = 2.0 * QPI * f_hz / C0;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    QD_DIAG("\n# diag: measurement, air | first 50 mm | host 100 mm | air across bare faces, scalar, every\n");
    QD_DIAG("# series declined, against the single pass at the exact cosines. named: the last event with the\n");
    QD_DIAG("# words the transition wrote; clean: the same event with a clean previous-medium word.\n");
    QD_DIAG("# ct_host is the traced cosine in the host, ct_one what one refraction from air would give\n");
    QD_DIAG("first,host,theta_ext,ct_host,ct_one,ref_dB,named_dB,d_named,clean_dB,d_clean,prev,rtype\n");
    // first, host. The last three put the solid with the large loss tangent in front, where the
    // projection of the first leg is furthest from a unit vector.
    const int pairs[9][2] = {{2, 5}, {1, 5}, {4, 5}, {2, 6}, {1, 6}, {4, 6}, {5, 1}, {5, 4}, {5, 6}};
    for (const auto &pr : pairs)
            for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                const int first = pr[0], host = pr[1];
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);

                // Entry air -> first, from ray_mesh_interact, as the traced-chain case does
                arma::mat orig, dest, fbs, sbs;
                diag_ray_at(thd, orig, dest);
                arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
                arma::u32_vec fbs_ind, sbs_ind;
                quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
                arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
                arma::vec gainN;
                quadriga_lib::ray_mesh_interact(4, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                                &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                                (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                                &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                                &path_dirN, true, (arma::u32_vec *)nullptr);
                REQUIRE(path_dirN.n_rows == 1);
                std::complex<double> a0(xprmatN(0, 0), xprmatN(1, 0));
                double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
                double c1t = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
                double s1t = std::sqrt(std::max(0.0, 1.0 - c1t * c1t));

                // Transition first -> host, in the frame make1 uses (normal along x)
                auto T = make1<double>(mtl, 4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                T.has_path_dir = true;
                T.path_dir_prev.zeros(1, 3);
                T.path_dir_prev(0, 0) = c1t, T.path_dir_prev(0, 1) = s1t;
                T.has_path_dirN = true;
                T.path_dirN.zeros(1, 3);
                T.path_dirN(0, 0) = std::cos(th), T.path_dirN(0, 1) = std::sin(th);
                T.normals(0, 3) = -1.0; // the host's front face: its normal opposes the ray
                T.run();
                double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);

                // Exit host -> air along the direction the transition wrote
                std::complex<double> a2[2];
                Rsu<double> X;
                for (int k = 0; k < 2; ++k)
                {
                    X = make1<double>(mtl, 4, OT_EXIT, 1u, host, 0, k ? 0 : (int)T.prev_out(0), (int)T.cur_out(0),
                                      k ? 0 : (int)T.buf_out(0), L2, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                    X.has_path_dir = true;
                    X.path_dir_prev.zeros(1, 3);
                    X.path_dir_prev(0, 0) = dx, X.path_dir_prev(0, 1) = dy;
                    X.run();
                    a2[k] = cpx(X.xprmat, 0, 0);
                }
                std::complex<double> a1 = cpx(T.xprmat, 0, 0);

                // Single pass at the exact cosines, written from the media alone
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                std::complex<double> z0(1.0, 0.0), z1 = std::sqrt(e1), z2 = std::sqrt(e2);
                std::complex<double> t01 = 2.0 * z0 * c0 / (z0 * c0 + z1 * c1), t12 = 2.0 * z1 * c1 / (z1 * c1 + z2 * c2),
                                     t20 = 2.0 * z2 * c2 / (z2 * c2 + z0 * c0);
                double ref = std::norm(t01 * t12 * t20) *
                             std::exp(2.0 * k0 * (std::imag(z1 * c1) * L1 + std::imag(z2 * c2) * L2));
                double named = std::norm(a0 * a1 * a2[0]), clean = std::norm(a0 * a1 * a2[1]);
                double ct_one = diag_traced_ct(e2, th);

                QD_DIAG("%d,%d,%.1f,%.9f,%.9f,%.6f,%.6f,%+.6f,%.6f,%+.6f,%d,%u\n", first, host, thd, std::abs(dx), ct_one,
                        dB(ref), dB(named), dB(named) - dB(ref), dB(clean), dB(clean) - dB(ref),
                        mat_of(T.prev_out(0)), (unsigned)X.resolved_type[0]);
                INFO("first " << first << ", host " << host << ", theta_ext " << thd);
                REQUIRE(mat_of(T.prev_out(0)) == first);
                REQUIRE(std::isfinite(named));
                // Every row, the lossless first solid being the control on the chain and on the
                // reference: the ray is not killed, the word names nothing the recovery reads, the
                // host direction is the one a single refraction from air writes, and the chain is
                // on the single pass.
                CHECK(named > 0.0);
                CHECK(std::abs(dB(named) - dB(clean)) < 1.0e-9);
                CHECK(std::abs(std::abs(dx) - ct_one) < 1.0e-9);
                CHECK(std::abs(dB(named) - dB(ref)) < 1.0e-6);
                // Bit 14 of the word the transition wrote: set out of a lossy solid whose face
                // was evaluated at the recovered cosine, clear out of a lossless one
                CHECK((((int)T.prev_out(0) & 0x4000) != 0) == (std::imag(e1) != 0.0));
            }
}

TEST_CASE("Recovery behind a lossy solid - EM, a second transition and a sheet inside the host")
{
    // The consumers of the recovered cosine the bare-face measurement does not reach. All three
    // chains start air | first | host across bare faces with every series declined (eps 1.5), the
    // entry from ray_mesh_interact, and are compared with the single pass at the exact cosines
    // sqrt(1 - sin^2(theta_ext) / eps), which no layer in front changes.
    //
    //   EM       the exit port per channel, VV being TE and HH being TM in this frame
    //   3 solids air | A | B | C | air: the port and the written direction of a transition out of
    //            a leg that was itself entered from a solid, and a second leg behind a named solid
    //   sheet    air | first | host | sheet | host | air: the composite of a sheet reached from
    //            inside the host, both ports, and the leg behind it
    //
    // A lossless first solid is the control on each chain and on its reference.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, thk = 0.006;
    auto mtl = diag_palette<double>();
    arma::mat mesh = diag_quad_x(200.0, 0.0);
    const uint8_t OT_SHEET_ISO = 67;
    const int SHEET = 3;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Ent
    {
        std::complex<double> te, tm; // VV and HH of the entry; scalar: both the one coefficient
        double c, s;                 // traced cosine and sine in the first solid
    };
    auto entry = [&](int itype, int first, double thd) -> Ent
    {
        arma::mat orig, dest, fbs, sbs;
        diag_ray_at(thd, orig, dest);
        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)first));
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        arma::mat origN, destN, xprmatN, path_dirN, normal_vecN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl,
                                        &fbs_ind, &sbs_ind, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &origN, &destN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        &gainN, &xprmatN, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::vec *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        &normal_vecN, (std::vector<uint8_t> *)nullptr,
                                        &path_dirN, true, (arma::u32_vec *)nullptr);
        REQUIRE(path_dirN.n_rows == 1);
        Ent E;
        E.te = std::complex<double>(xprmatN(0, 0), xprmatN(1, 0));
        E.tm = (itype < 3) ? std::complex<double>(xprmatN(6, 0), xprmatN(7, 0)) : E.te;
        double nx = normal_vecN(0, 0), ny = normal_vecN(0, 1), nz = normal_vecN(0, 2);
        E.c = std::abs(path_dirN(0, 0) * nx + path_dirN(0, 1) * ny + path_dirN(0, 2) * nz);
        E.s = std::sqrt(std::max(0.0, 1.0 - E.c * E.c));
        return E;
    };
    // One event of ray_state_update closing a leg of normal thickness L along (dx, dy). A
    // media-to-media event takes its plane normal from the second face, reversed to oppose the ray.
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double L,
                     double th, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        if (ot == OT_M2M_M1)
            C.normals(0, 3) = -1.0;
        C.run();
        return C;
    };
    // Power of one port at an exact cosine, per channel, and of one normal-thickness decay
    auto port = [](std::complex<double> ea, std::complex<double> eb, std::complex<double> ca, bool scalar, double &pte, double &ptm)
    {
        std::complex<double> te, tm;
        diag_port(ea, eb, ca, scalar, te, tm);
        pte = std::norm(te), ptm = std::norm(tm);
    };
    auto decay = [&](std::complex<double> e, std::complex<double> c, double L)
    { return std::exp(2.0 * k0 * std::imag(std::sqrt(e) * c) * L); };
    const std::complex<double> e0(1.0, 0.0);

    // ---- EM ---------------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, EM. air | first 50 mm | host 100 mm | air, per channel,\n");
    QD_DIAG("# chain minus the single pass at the exact cosines, dB\n");
    QD_DIAG("first,host,theta_ext,ref_te_dB,ref_tm_dB,d_te,d_tm,rtype\n");
    for (int host : {5, 6})
        for (int first : {2, 1})
            for (double thd : {40.0, 60.0, 75.0, 85.0})
            {
                const double L1 = 0.05, L2 = 0.10;
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                Ent E = entry(1, first, thd);
                auto T = event(1, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, E.c, E.s);
                auto X = event(1, OT_EXIT, 1u, host, 0, (int)T.prev_out(0), (int)T.cur_out(0), (int)T.buf_out(0), L2, th,
                               (double)T.path_dirN(0, 0), (double)T.path_dirN(0, 1));
                double ch_te = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                double ch_tm = std::norm(E.tm * cpx(T.xprmat, 0, 3) * cpx(X.xprmat, 0, 3));
                double a[2], b[2], c[2];
                port(e0, e1, c0, false, a[0], a[1]), port(e1, e2, c1, false, b[0], b[1]), port(e2, e0, c2, false, c[0], c[1]);
                double dec = decay(e1, c1, L1) * decay(e2, c2, L2);
                double r_te = a[0] * b[0] * c[0] * dec, r_tm = a[1] * b[1] * c[1] * dec;
                QD_DIAG("%d,%d,%.1f,%.6f,%.6f,%+.6f,%+.6f,%u\n", first, host, thd, dB(r_te), dB(r_tm), dB(ch_te) - dB(r_te),
                        dB(ch_tm) - dB(r_tm), (unsigned)X.resolved_type[0]);
                INFO("EM, first " << first << ", host " << host << ", theta_ext " << thd);
                CHECK(ch_te > 0.0);
                CHECK(std::abs(dB(ch_te) - dB(r_te)) < 1.0e-6);
                CHECK(std::abs(dB(ch_tm) - dB(r_tm)) < 1.0e-6);
            }

    // ---- Three solids -------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, a second transition. air | A 50 mm | B 100 mm | C 100 mm | air,\n");
    QD_DIAG("# scalar. d_T2 is the transition B -> C alone against its port and its leg; ct_C the traced\n");
    QD_DIAG("# cosine in C and ct_one what one refraction from air gives; d_named and d_clean the chain\n");
    QD_DIAG("# with the last event fed the words the transition wrote and a clean previous-medium word\n");
    QD_DIAG("A,B,C,theta_ext,ct_C,ct_one,ref_dB,d_T2,d_named,d_clean,rtype\n");
    const int triples[4][3] = {{2, 6, 2}, {1, 5, 4}, {5, 1, 6}, {4, 5, 1}};
    for (const auto &tr : triples)
        for (double thd : {40.0, 60.0, 75.0, 85.0})
        {
            const int A = tr[0], B = tr[1], C = tr[2];
            const double LA = 0.05, LB = 0.10, LC = 0.10;
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            std::complex<double> eA = diag_eps(A, f_hz), eB = diag_eps(B, f_hz), eC = diag_eps(C, f_hz);
            std::complex<double> c0(std::cos(th), 0.0), cA = std::sqrt(1.0 - s2 / eA), cB = std::sqrt(1.0 - s2 / eB),
                                 cC = std::sqrt(1.0 - s2 / eC);
            Ent E = entry(4, A, thd);
            auto T1 = event(4, OT_M2M_M1, 2u, A, B, 0, A, 0, LA, th, E.c, E.s);
            auto T2 = event(4, OT_M2M_M1, 2u, B, C, (int)T1.prev_out(0), (int)T1.cur_out(0), (int)T1.buf_out(0), LB, th,
                            (double)T1.path_dirN(0, 0), (double)T1.path_dirN(0, 1));
            double dx = (double)T2.path_dirN(0, 0), dy = (double)T2.path_dirN(0, 1);
            double pw[2];
            Rsu<double> X;
            for (int k = 0; k < 2; ++k)
            {
                X = event(4, OT_EXIT, 1u, C, 0, k ? 0 : (int)T2.prev_out(0), (int)T2.cur_out(0), k ? 0 : (int)T2.buf_out(0),
                          LC, th, dx, dy);
                pw[k] = std::norm(E.te * cpx(T1.xprmat, 0, 0) * cpx(T2.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
            }
            double p01, pAB, pBC, pC0, u;
            port(e0, eA, c0, true, p01, u), port(eA, eB, cA, true, pAB, u), port(eB, eC, cB, true, pBC, u), port(eC, e0, cC, true, pC0, u);
            double ref = p01 * pAB * pBC * pC0 * decay(eA, cA, LA) * decay(eB, cB, LB) * decay(eC, cC, LC);
            double d_T2 = dB(std::norm(cpx(T2.xprmat, 0, 0))) - dB(pBC * decay(eB, cB, LB));
            double ct_one = diag_traced_ct(eC, th);
            QD_DIAG("%d,%d,%d,%.1f,%.9f,%.9f,%.6f,%+.6f,%+.6f,%+.6f,%u\n", A, B, C, thd, std::abs(dx), ct_one, dB(ref), d_T2,
                    dB(pw[0]) - dB(ref), dB(pw[1]) - dB(ref), (unsigned)X.resolved_type[0]);
            INFO("three solids " << A << " | " << B << " | " << C << ", theta_ext " << thd);
            REQUIRE(mat_of(T2.prev_out(0)) == B);
            CHECK(pw[0] > 0.0);
            CHECK(std::abs(d_T2) < 1.0e-6);
            CHECK(std::abs(std::abs(dx) - ct_one) < 1.0e-9);
            CHECK(std::abs(dB(pw[0]) - dB(pw[1])) < 1.0e-9);
            CHECK(std::abs(dB(pw[0]) - dB(ref)) < 1.0e-6);
        }

    // ---- A sheet inside the host --------------------------------------------------------------
    QD_DIAG("\n# diag: recovery behind a lossy solid, a sheet inside the host. air | first 50 mm | host 50 mm |\n");
    QD_DIAG("# sheet 6 mm eps 5.24 - 0.63j | host 50 mm | air, scalar. d_ev is the sheet event against the\n");
    QD_DIAG("# composite at the exact cosine and the host leg in front of it; d_twin the same event against\n");
    QD_DIAG("# its twin with a clean previous-medium word; d_chain the transmitted chain end to end\n");
    QD_DIAG("first,host,theta_ext,port,ev_dB,d_ev,d_twin,d_chain\n");
    const std::complex<double> e_s = diag_eps(SHEET, f_hz);
    for (int host : {5, 6})
        for (int first : {2, 1})
            for (double thd : {40.0, 60.0, 75.0, 85.0})
                for (int rp = 0; rp < 2; ++rp)
                {
                    const double L1 = 0.05, La = 0.05, Lb = 0.05;
                    double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    std::complex<double> e1 = diag_eps(first, f_hz), e2 = diag_eps(host, f_hz);
                    std::complex<double> c0(std::cos(th), 0.0), c1 = std::sqrt(1.0 - s2 / e1), c2 = std::sqrt(1.0 - s2 / e2);
                    Ent E = entry(4, first, thd);
                    auto T = event(4, OT_M2M_M1, 2u, first, host, 0, first, 0, L1, th, E.c, E.s);
                    double dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                    auto S = event(rp ? 3 : 4, OT_SHEET_ISO, 1u, SHEET, 0, (int)T.prev_out(0), (int)T.cur_out(0),
                                   (int)T.buf_out(0), La, th, dx, dy);
                    auto S0 = event(rp ? 3 : 4, OT_SHEET_ISO, 1u, SHEET, 0, 0, (int)T.cur_out(0), 0, La, th, dx, dy);
                    DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e2, thk, c2, f_hz, false, true);
                    double ev = std::norm(cpx(S.xprmat, 0, 0)), tw = std::norm(cpx(S0.xprmat, 0, 0));
                    // Both passes close the host leg with the normal-thickness decay
                    double leg = decay(e2, c2, La);
                    double ev_ref = std::norm(rp ? Ref.R_te : Ref.T_te) * leg;
                    double d_chain = 0.0;
                    if (!rp)
                    {
                        auto X = event(4, OT_EXIT, 1u, host, 0, (int)S.prev_out(0), (int)S.cur_out(0), (int)S.buf_out(0), Lb,
                                       th, (double)S.path_dirN(0, 0), (double)S.path_dirN(0, 1));
                        double p01, p12, p20, u;
                        port(e0, e1, c0, true, p01, u), port(e1, e2, c1, true, p12, u), port(e2, e0, c2, true, p20, u);
                        double ref = p01 * p12 * std::norm(Ref.T_te) * p20 * decay(e1, c1, L1) * decay(e2, c2, La + Lb);
                        double ch = std::norm(E.te * cpx(T.xprmat, 0, 0) * cpx(S.xprmat, 0, 0) * cpx(X.xprmat, 0, 0));
                        d_chain = dB(ch) - dB(ref);
                    }
                    QD_DIAG("%d,%d,%.1f,%s,%.6f,%+.6f,%+.6f,%+.6f\n", first, host, thd, rp ? "R" : "T", dB(ev), dB(ev) - dB(ev_ref),
                            dB(ev) - dB(tw), d_chain);
                    INFO("sheet inside the host, first " << first << ", host " << host << ", theta_ext " << thd << ", port " << rp);
                    REQUIRE(mat_of(S.prev_out(0)) == SHEET);
                    CHECK(std::abs(dB(ev) - dB(tw)) < 1.0e-9);
                    CHECK(std::abs(dB(ev) - dB(ev_ref)) < 1.0e-6);
                    CHECK(std::abs(d_chain) < 1.0e-6);
                }
}

TEST_CASE("Recovery behind a lossy solid - a face that was evaluated at a real angle keeps the solid named")
{
    // Where the recovery declined on the leg in the first
    // solid, the shared face was evaluated at a real angle in that solid, and a real angle in a
    // lossy medium is a complex transverse invariant, e1 sin^2(theta_1). The cosine in the host is
    // then sqrt(1 - (e1 / e2) sin^2(theta_1)), which is what the solve with the first solid named
    // returns and what a solve for a real invariant does not.
    //
    //   wedge     the first solid entered from air with its faces known not parallel, so its own
    //             recovery declines; the transition is ray_state_update's own and writes the
    //             direction. theta is the external angle.
    //   launched  a wave that starts inside the first solid at the real angle theta there. No
    //             direction reaches the transition, so the direction in the host is written here
    //             as ray_mesh_interact writes it, from qd_refract at the real angle.
    //
    // The last event, host | air, is compared with its port at that cosine and its traversal, with
    // the first solid in the previous-medium word (named) and with a clean word (clean).
    const double f_hz = FRQ, L1 = 0.05, L2 = 0.10;
    auto mtl = diag_palette<double>();
    const int FIRST = 1;
    const std::complex<double> e1 = diag_eps(FIRST, f_hz), e0(1.0, 0.0);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    QD_DIAG("\n# diag: measurement, a host entered across a face evaluated at a real angle in the lossy first\n");
    QD_DIAG("# solid, eps 5.24 - 0.63j, scalar. theta is external for the wedge and inside the first solid\n");
    QD_DIAG("# for the launched wave; |p| is the magnitude of the transverse invariant. The last event,\n");
    QD_DIAG("# host 100 mm | air, against its port and traversal at the cosine the face was evaluated at\n");
    QD_DIAG("route,host,theta,theta_1,|p|,ct_host,ref_dB,d_named,d_clean,prev,rtype\n");
    for (int route = 0; route < 2; ++route)
        for (int host : {5, 6})
            for (double thd : (route ? std::vector<double>{10.0, 20.0, 23.0, 25.0} : std::vector<double>{40.0, 60.0, 75.0, 85.0}))
            {
                double th = thd * QPI / 180.0;
                std::complex<double> e2 = diag_eps(host, f_hz);
                double c1r, s1r, dx, dy;
                int prev_named = FIRST;
                if (route == 0)
                {
                    // traced direction in the first solid, one refraction from air
                    c1r = diag_traced_ct(e1, th), s1r = std::sqrt(1.0 - c1r * c1r);
                    auto T = make1<double>(mtl, 4, OT_M2M_M1, 2u, FIRST, host, (int)enc(0, true), FIRST, 0, L1, 0.5, 0.5,
                                           QPI / 2.0 - th, 1.5);
                    T.has_path_dir = true;
                    T.path_dir_prev.zeros(1, 3);
                    T.path_dir_prev(0, 0) = c1r, T.path_dir_prev(0, 1) = s1r;
                    T.has_path_dirN = true;
                    T.path_dirN.zeros(1, 3);
                    T.path_dirN(0, 0) = std::cos(th), T.path_dirN(0, 1) = std::sin(th);
                    T.normals(0, 3) = -1.0;
                    T.run();
                    dx = (double)T.path_dirN(0, 0), dy = (double)T.path_dirN(0, 1);
                    prev_named = (int)T.prev_out(0);
                    REQUIRE(mat_of(T.prev_out(0)) == FIRST);
                }
                else
                    c1r = std::cos(th), s1r = std::sin(th);

                // The cosine the face was evaluated at, and the direction qd_refract writes from it
                std::complex<double> eps_r = e1 / e2, c2w = std::sqrt(1.0 - eps_r * s1r * s1r);
                double eta = std::sqrt(std::abs(eps_r)), rr = std::real(c2w);
                double ct_map = rr / std::sqrt((eta * s1r) * (eta * s1r) + rr * rr);
                if (route == 1)
                    dx = ct_map, dy = std::sqrt(1.0 - ct_map * ct_map);

                double pw[2];
                Rsu<double> X;
                for (int k = 0; k < 2; ++k)
                {
                    X = make1<double>(mtl, 4, OT_EXIT, 1u, host, 0, k ? 0 : prev_named, host, 0, L2, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                    X.has_path_dir = true;
                    X.path_dir_prev.zeros(1, 3);
                    X.path_dir_prev(0, 0) = dx, X.path_dir_prev(0, 1) = dy;
                    X.run();
                    pw[k] = std::norm(cpx(X.xprmat, 0, 0));
                }
                std::complex<double> te, tm;
                diag_port(e2, e0, c2w, true, te, tm);
                double ct_h = std::abs(dx);
                double ref = std::norm(te) * std::norm(diag_medium_vbs(e2, L2 / ct_h, ct_h, L2, c2w, f_hz));
                QD_DIAG("%s,%d,%.1f,%.3f,%.4f,%.9f,%.6f,%+.6f,%+.6f,%d,%u\n", route ? "launched" : "wedge   ", host, thd,
                        std::acos(c1r) * 180.0 / QPI, std::abs(e1) * s1r * s1r, ct_h, dB(ref), dB(pw[0]) - dB(ref),
                        dB(pw[1]) - dB(ref), mat_of((short)prev_named), (unsigned)X.resolved_type[0]);
                INFO((route ? "launched" : "wedge") << ", host " << host << ", theta " << thd);
                CHECK(std::abs(ct_h - ct_map) < 1.0e-9); // the direction is the real-angle map
                REQUIRE(std::isfinite(pw[0]));
                REQUIRE(std::isfinite(pw[1]));
                // The word carries no flag, so the recovery keeps the solid named and the event
                // sits on the cosine the face was evaluated at. The clean column is recorded: it
                // is what a solve for a real invariant makes of this wave.
                CHECK((prev_named & 0x4000) == 0);
                CHECK(pw[0] > 0.0);
                CHECK(std::abs(dB(pw[0]) - dB(ref)) < 1.0e-6);
            }
}

TEST_CASE("Sheet inside a host - the mirror behind the ray is backed by the host")
{
    // The state half of the backing question. After a sheet is crossed inside a host the
    // previous-medium word holds the sheet, and the next close-out of the host resolves it as a
    // layer backed by air. Behind a sheet inside a host is the host. Two events: the crossing,
    // host | sheet | host, and the bare far face of the host fed the three state words the
    // crossing wrote. The series is the resolving run (eps 0) over the declined one (eps 1.5) and
    // is compared with the factorized form, the sheet a layer from the characteristic matrix,
    // once backed by the host and once by air.
    //
    // No path_dir is supplied, so every boundary is at the fbs cosine and the propagation factor
    // is the hybrid form diag_phi reproduces. A coating, air | sheet | host, is the control: its
    // sheet is backed by air and has to stay there.
    const double f_hz = FRQ, thk = 0.006;
    auto mtl = diag_palette<double>();
    const std::complex<double> U(1.0, 0.0), e_air(1.0, 0.0);
    const int HOST = 5, SHEET = 3, NEAR = 9;
    const uint8_t OT_SHEET_ISO = 67, OT_GLUED_ENTRY = 71;
    const std::complex<double> e_h = diag_eps(HOST, f_hz);

    QD_DIAG("\n# diag: sheet inside a host, the host's series on the leg behind it. Host eps 1.50 - 0.40j,\n");
    QD_DIAG("# sheet 6 mm, scalar. S_applied is the resolving run over the declined one; S_host has the\n");
    QD_DIAG("# sheet behind the ray backed by the host, S_air by air. kind 0 is the sheet inside the host,\n");
    QD_DIAG("# kind 1 the coating, whose sheet is backed by air\n");
    QD_DIAG("kind,sheet,L_mm,theta_n,port,|S_applied|,|S_host|,|S_air|,d_host_dB,d_air_dB,|dS_host|,|dS_air|,prev\n");
    for (int kind = 0; kind < 2; ++kind)
        for (int sheet : {SHEET, NEAR})
            for (double L : {0.05, 0.10})
                for (double thd : {0.0, 20.0, 40.0, 50.0})
                    for (int port = 0; port < 2; ++port)
                    {
                        double th = thd * QPI / 180.0;
                        const std::complex<double> e_s = diag_eps(sheet, f_hz);
                        auto A = kind ? make1<double>(mtl, 4, OT_GLUED_ENTRY, 2u, sheet, HOST, 0, 0, 0, 0.5, 0.5, 0.5,
                                                      QPI / 2.0 - th, 1.5)
                                      : make1<double>(mtl, 4, OT_SHEET_ISO, 1u, sheet, 0, 0, HOST, 0, L, 0.5, 0.5,
                                                      QPI / 2.0 - th, 1.5);
                        A.run();
                        INFO("kind " << kind << ", sheet " << sheet << ", L " << L << ", theta_n " << thd << ", port " << port);
                        REQUIRE(mat_of(A.cur_out(0)) == HOST);

                        std::complex<double> v[2];
                        uint8_t rt = 0;
                        for (int k = 0; k < 2; ++k)
                        {
                            auto B = make1<double>(mtl, port ? 3 : 4, OT_EXIT, 1u, HOST, 0, (int)A.prev_out(0),
                                                   (int)A.cur_out(0), (int)A.buf_out(0), L, 0.5, 0.5, QPI / 2.0 - th,
                                                   k ? 0.0 : 1.5);
                            B.run();
                            v[k] = cpx(B.xprmat, 0, 0);
                            rt = B.resolved_type[0];
                        }
                        REQUIRE(rt_bit(rt, 2));
                        std::complex<double> S = v[1] / v[0];

                        std::complex<double> cs(std::cos(th), 0.0);
                        std::complex<double> phi = diag_phi(e_h, L, th, f_hz);
                        std::complex<double> r_ahead = diag_r_slabside(e_h, e_air, cs);
                        // inside the host the sheet is centred on its face; a lining lies in the air beyond
                        std::complex<double> r_host = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_h, U, cs, f_hz, false, true).R_te;
                        std::complex<double> r_air = diag_cm_stack(e_h, U, {{e_s, U, thk}}, e_air, U, cs, f_hz, true, true).R_te;
                        std::complex<double> S_host = 1.0 / (1.0 - r_ahead * r_host * phi * phi);
                        std::complex<double> S_air = 1.0 / (1.0 - r_ahead * r_air * phi * phi);
                        QD_DIAG("%d,%d,%.0f,%.1f,%s,%.9f,%.9f,%.9f,%+.4f,%+.4f,%.3e,%.3e,%d\n", kind, sheet, L * 1000.0, thd,
                                port ? "R" : "T", std::abs(S), std::abs(S_host), std::abs(S_air),
                                20.0 * std::log10(std::abs(S) / std::abs(S_host)),
                                20.0 * std::log10(std::abs(S) / std::abs(S_air)), std::abs(S - S_host),
                                std::abs(S - S_air), mat_of(A.prev_out(0)));
                        CHECK(std::abs(S_host - S_air) > 1.0e-3); // the row can tell the two backings apart
                        if (kind)
                            CHECK(std::abs(S - S_air) < 1.0e-9); // a coating is backed by air
                        else
                            CHECK(std::abs(S - S_host) < 1.0e-9); // a sheet inside a host by the host
                    }
}

TEST_CASE("Bonded mass - a mass-law sheet inside a mass-law wall")
{
    // Plate depth, the one case that reaches a result. A wall in two halves with a sheet between
    // them is one plate of three bodies. The first half pays its own term and the sheet the
    // increment, both at the crossing; the second half then knows only the sheet, so it pays
    // almost its whole term again. What it needs is the first half's thickness, which is its own
    // material's and travels as the thickness code.
    //
    // Two events: the crossing inside the wall, and the wall's bare far face fed the three state
    // words the crossing wrote. The mass paid is the ratio against the same bodies with their
    // mass argument taken to nothing. A sheet without the column separates the halves, by the
    // settled rule, and is the control: each half pays its own term. Every sheet differs from its
    // wall in permittivity; one that does not is merged into the host leg and is another case.
    const double f_hz = FRQ;
    const double e_w = 2.1609e-8, e_c = 2.401e-9, tf_w = 0.2, tf_c = 0.1, fR = 0.001, fZ = 1.0e9;
    const uint8_t OT_SHEET_ISO = 67;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const int WS = 3, WSZ = 4, SB = 5, SBZ = 6, CONC = 7, CONCZ = 8, P20 = 9, P20Z = 10, CARPET = 11;
    const std::vector<LumpMtl> pal = {
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 1  P8     wood sheet
        {e_w, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 2  P8Z
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR},   // 3  WS     wood solid
        {e_w, 0.0, 0.0, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ},   // 4  WSZ
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR}, // 5  SB     rigid board
        {e_c, 0.0, 0.012, 0.5, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ}, // 6  SBZ
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fR},   // 7  CONC   concrete solid
        {e_c, 0.0, 0.0, 2.0, tf_c, 0.0, 0.0, 0.0, 0.0, 30.0, fZ},   // 8  CONCZ
        {e_w, 0.0, 0.020, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 9  P20    wood sheet
        {e_w, 0.0, 0.020, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 10 P20Z
        {2.50, 0.80, 0.010},                                        // 11 CARPET a lossy layer, m 0
        {1.0e-8, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fR}, // 12 D8   a denser wood sheet, m 20
        {1.0e-8, 0.0, 0.008, 0.0, tf_w, 0.0, 0.0, 0.0, 0.0, 20.0, fZ}, // 13 D8Z
    };
    const int D8 = 12, D8Z = 13;
    auto mtl = lump_palette(pal);

    struct Row
    {
        const char *name;
        int sheet, sheetz, wall, wallz;
        double d1, d2;
        DiagMass ms, mw; // sheet, and the wall per meter of thickness set below
    };
    const std::vector<Row> rows = {
        {"wood 60 | dense 8 | wood 40   ", D8, D8Z, WS, WSZ, 0.06, 0.04, {20.0, fR, 0.008}, {20.0, fR, 0.0}},
        {"wood 50 | board 12 | wood 50  ", SB, SBZ, WS, WSZ, 0.05, 0.05, {30.0, fR, 0.012}, {20.0, fR, 0.0}},
        {"conc 50 | wood 20 | conc 50   ", P20, P20Z, CONC, CONCZ, 0.05, 0.05, {20.0, fR, 0.020}, {30.0, fR, 0.0}},
        {"conc 30 | wood 20 | conc 70   ", P20, P20Z, CONC, CONCZ, 0.03, 0.07, {20.0, fR, 0.020}, {30.0, fR, 0.0}},
        {"conc 50 | m=0 layer | conc 50 ", CARPET, CARPET, CONC, CONCZ, 0.05, 0.05, {0.0, fR, 0.010}, {30.0, fR, 0.0}},
    };

    QD_DIAG("\n# diag: bonded mass, a sheet inside a wall. first is what the crossing pays (the first half\n");
    QD_DIAG("# and the sheet), second what the far face pays (the second half). plate is the three bodies\n");
    QD_DIAG("# as one; the last row has a sheet without the column, which separates the halves\n");
    QD_DIAG("stack,theta_n,first_dB,second_dB,paid_dB,plate_dB,sum_dB,paid_minus_plate,buf_flag\n");
    for (const auto &R : rows)
        for (double thd : {0.0, 30.0, 60.0, 75.0})
        {
            double th = thd * QPI / 180.0, ct = std::cos(th);
            double paid_ev[2] = {0.0, 0.0};
            int buf = 0;
            Rsu<double> A[2];
            for (int z = 0; z < 2; ++z)
            {
                int sh = z ? R.sheetz : R.sheet, wl = z ? R.wallz : R.wall;
                A[z] = make1<double>(mtl, 4, OT_SHEET_ISO, 1u, sh, 0, 0, wl, 0, R.d1 / ct, 0.5, 0.5, QPI / 2.0 - th, 1.5);
                A[z].run();
                auto B = make1<double>(mtl, 4, OT_EXIT, 1u, wl, 0, (int)A[z].prev_out(0), (int)A[z].cur_out(0),
                                       (int)A[z].buf_out(0), R.d2 / ct, 2.0, 0.5, QPI / 2.0 - th, 1.5);
                B.run();
                double sgn = z ? 1.0 : -1.0;
                paid_ev[0] += sgn * dB(std::norm(cpx(A[z].xprmat, 0, 0)));
                paid_ev[1] += sgn * dB(std::norm(vvf(B)));
                if (!z)
                    buf = (int)A[z].buf_out(0);
            }
            DiagMass h1 = R.mw, h2 = R.mw;
            h1.thk = R.d1, h2.thk = R.d2;
            double paid = paid_ev[0] + paid_ev[1];
            double own1 = diag_bonded_TL({h1}, f_hz, ct), own2 = diag_bonded_TL({h2}, f_hz, ct);
            bool massy = R.ms.m > 0.0;
            double plate = massy ? diag_bonded_TL({h1, R.ms, h2}, f_hz, ct) : own1 + own2;
            double sum = own1 + own2 + (massy ? diag_bonded_TL({R.ms}, f_hz, ct) : 0.0);
            QD_DIAG("%s,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%+.6f,%d\n", R.name, thd, paid_ev[0], paid_ev[1], paid, plate, sum,
                    paid - plate, (buf & 0x8000) ? 1 : 0);
            INFO(R.name << ", theta_n " << thd);
            REQUIRE(mat_of(A[0].prev_out(0)) == R.sheet);
            if (massy)
            {
                CHECK(std::abs(paid_ev[0] - diag_bonded_TL({h1, R.ms}, f_hz, ct)) < 1.0e-9); // the crossing pays two bodies
                CHECK(std::abs(paid - plate) < 0.01);                                         // and the far face the third
                CHECK((buf & 0x8000) != 0);
            }
            else
                CHECK(std::abs(paid - plate) < 1.0e-9); // separated: each half its own term
        }
}

TEST_CASE("ray_state_update - the material index is 14 bits")
{
    // Bit 14 of each state word is a flag, so a material index runs to 16383.
    auto pal = diag_palette<double>();
    auto big = [&](size_t n)
    {
        auto m = pal;
        for (auto &kv : m)
            kv.second.resize(n, kv.second[5]); // pad with the lossless solid, material 6
        return m;
    };
    {
        // The largest index, as the current medium, with the resolved flag beside it
        auto m = big(16383);
        auto C = make1<double>(m, 4, OT_EXIT, 1u, 16383, 0, 0, 16383, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(C.run());
        CHECK((int)C.prev_out(0) == 16383);
        auto D = make1<double>(m, 4, OT_EXIT, 1u, 16383, 0, 0, (int)enc(16383, true), 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(D.run());
        CHECK((int)D.prev_out(0) == 16383);
    }
    {
        // One material more does not fit the index
        auto m = big(16384);
        auto C = make1<double>(m, 4, OT_EXIT, 1u, 6, 0, 0, 6, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_THROWS_AS(C.run(), std::invalid_argument);
    }
    {
        // Bit 14 of the previous-medium word is a flag and not part of the index
        auto C = make1<double>(pal, 4, OT_EXIT, 1u, 5, 0, 0x4000 | 3, 5, 0, 0.1, 0.5, 0.5, QPI / 2.0, 1.5);
        CHECK_NOTHROW(C.run());
    }
}

TEST_CASE("Recovery gate - the hit code a traced run carries")
{
    // Every chain fixture above hands ray_state_update a synthetic hit code, 1 for a bare exit and
    // 69 for a glued one. A traced run does not: with the undeviated tracer a ray inside a solid
    // travels at its external angle, ray_mesh_interact judges the exit face at that angle, and from
    // sqrt|eps| sin(theta_ext) >= 1 its code carries the total-reflection bit, 33 and 101. That bit
    // is a statement about the face ahead at an angle the wave does not have. The recovery inverts
    // the bend at the face behind, and declines on it.
    //
    // Four instruments. The code ray_mesh_interact writes for the exit, asserted. The twin: the
    // same event with that code and with the bit cleared, bitwise, which needs no reference. The
    // absolute level with the real code, against the analytic slab and the composite at the exact
    // cosine. And the two-event run through the intersector with nothing synthetic in it, for the
    // undeviated and for the refracting tracer; the lossy rows of the refracting tracer are
    // recorded and not asserted.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, LOSSY = 5, DENSE = 2;
    const std::complex<double> e_air(1.0, 0.0), e_s = diag_eps(SHEET, f_hz);
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    // The code ray_mesh_interact writes for an exit met along the external direction: the exit
    // face is the back of a quad at x = 0, with a sheet face 10 um behind it for a glued exit
    auto exit_code = [&](int itype, int host, bool glued, double thd) -> uint8_t
    {
        arma::mat mesh = diag_quad_x(200.0, 0.0);
        arma::uvec mtl_ind(2, arma::fill::value((arma::uword)host));
        if (glued)
        {
            mesh = arma::join_cols(mesh, diag_quad_x(200.0, -1.0e-5));
            mtl_ind = arma::uvec{(arma::uword)host, (arma::uword)host, (arma::uword)SHEET, (arma::uword)SHEET};
        }
        double t = std::tan(thd * QPI / 180.0);
        arma::mat orig{{1.0, 0.37, 0.0}}, dest{{-1.0, 0.37 + 2.0 * t, 0.0}}, fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        std::vector<uint8_t> ot;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, &ot, (arma::mat *)nullptr, true,
                                        (arma::u32_vec *)nullptr);
        REQUIRE(ot.size() == 1);
        return ot[0];
    };
    // One event closing a leg of normal thickness L along (dx, dy)
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, double th, double eps,
                     double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, eps);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.run();
        return C;
    };
    auto same = [](const Rsu<double> &A, const Rsu<double> &B)
    {
        for (arma::uword k = 0; k < A.xprmat.n_rows; ++k)
            if ((double)A.xprmat(k, 0) != (double)B.xprmat(k, 0))
                return false;
        return (double)A.gain(0) == (double)B.gain(0);
    };

    // ------------------------------------------------------------------------------------------
    // Bare exit
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery gate, bare exit. air | slab 100 mm | air, scalar, the exit event along the\n");
    QD_DIAG("# direction a tracer holds, series resolved. code is what ray_mesh_interact writes for that exit\n");
    QD_DIAG("# met at the external angle. d_real and d_syn are the chain against the analytic slab with\n");
    QD_DIAG("# that code and with code 1; twin_T, twin_R, twin_EM say whether the two runs are bit-identical\n");
    QD_DIAG("mtl,theta_ext,code,d_real_dB,d_syn_dB,twin_T,twin_R,twin_EM,rtype\n");
    for (int m : {5, 1, 4, 6})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            std::complex<double> es = diag_eps(m, f_hz);
            double ct = diag_traced_ct(es, th), st = std::sqrt(1.0 - ct * ct);
            uint8_t code = exit_code(4, m, false, thd);
            bool past = std::sqrt(std::abs(es)) * std::sin(th) >= 1.0;
            std::complex<double> t_in, u;
            diag_port(e_air, es, std::complex<double>(std::cos(th), 0.0), true, t_in, u);
            auto Xr = event(4, code, 1u, m, 0, 0, m, th, 0.0, ct, st);
            auto Xs = event(4, OT_EXIT, 1u, m, 0, 0, m, th, 0.0, ct, st);
            auto Rr = event(3, exit_code(3, m, false, thd), 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Rs = event(3, OT_EXIT, 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Er = event(1, exit_code(1, m, false, thd), 1u, m, 0, 0, m, th, 1.5, ct, st);
            auto Es = event(1, OT_EXIT, 1u, m, 0, 0, m, th, 1.5, ct, st);
            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);
            double d_real = dB(std::norm(t_in * cpx(Xr.xprmat, 0, 0))) - dB(p_te);
            double d_syn = dB(std::norm(t_in * cpx(Xs.xprmat, 0, 0))) - dB(p_te);
            QD_DIAG("%d,%.1f,%u,%+.6f,%+.6f,%d,%d,%d,%u\n", m, thd, (unsigned)code, d_real, d_syn, (int)same(Xr, Xs),
                    (int)same(Rr, Rs), (int)same(Er, Es), (unsigned)Xr.resolved_type[0]);
            INFO("bare exit, material " << m << ", theta_ext " << thd);
            CHECK((int)code == (past ? 33 : 1)); // the bit is the geometric-angle verdict
            CHECK(std::abs(d_syn) < 1.0e-6);     // the control: the synthetic code on the slab
            CHECK(std::abs(d_real) < 1.0e-6);
            CHECK(same(Xr, Xs));
            CHECK(same(Rr, Rs));
            CHECK(same(Er, Es));
        }

    // ------------------------------------------------------------------------------------------
    // Glued exit
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery gate, glued exit. host 100 mm | coating 6 mm eps 5.24 - 0.63j | air, scalar,\n");
    QD_DIAG("# host series declined. d_T and d_R: the event with the code ray_mesh_interact writes against\n");
    QD_DIAG("# the composite at the exact cosine times the host leg; s_T the same with code 69\n");
    QD_DIAG("host,theta_ext,code,d_T_dB,d_R_dB,s_T_dB,twin_T,twin_R,twin_EM,rtype\n");
    for (int host : {5, 6})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
            std::complex<double> e2 = diag_eps(host, f_hz), c2 = std::sqrt(1.0 - s2 / e2);
            double ct = diag_traced_ct(e2, th), st = std::sqrt(1.0 - ct * ct);
            uint8_t code = exit_code(4, host, true, thd);
            bool past = std::sqrt(std::abs(e2)) * std::sin(th) >= 1.0;
            auto Tr = event(4, code, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Ts = event(4, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Rr = event(3, exit_code(3, host, true, thd), 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Rs = event(3, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Er = event(1, exit_code(1, host, true, thd), 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            auto Es = event(1, 69, 2u, SHEET, host, 0, host, th, 1.5, ct, st);
            DiagSheet Ref = diag_sheet_composite_cx(e2, e_s, e_air, thk, c2, f_hz, true, true);
            // A sheet event replaces the field, so the probe feed is not in it
            double ref_T = std::exp(2.0 * k0 * std::imag(std::sqrt(e2) * c2) * L) * std::norm(Ref.T_te);
            // A reflect pass at a sheet closes the host leg as the transmit pass does
            double ref_R = std::exp(2.0 * k0 * std::imag(std::sqrt(e2) * c2) * L) * std::norm(Ref.R_te);
            double d_T = dB(std::norm(cpx(Tr.xprmat, 0, 0))) - dB(ref_T), d_R = dB(std::norm(cpx(Rr.xprmat, 0, 0))) - dB(ref_R);
            double s_T = dB(std::norm(cpx(Ts.xprmat, 0, 0))) - dB(ref_T);
            QD_DIAG("%d,%.1f,%u,%+.6f,%+.6f,%+.6f,%d,%d,%d,%u\n", host, thd, (unsigned)code, d_T, d_R, s_T, (int)same(Tr, Ts),
                    (int)same(Rr, Rs), (int)same(Er, Es), (unsigned)Tr.resolved_type[0]);
            INFO("glued exit, host " << host << ", theta_ext " << thd);
            CHECK((int)code == (past ? 101 : 69));
            CHECK(std::abs(s_T) < 1.0e-6); // the control
            CHECK(std::abs(d_T) < 1.0e-6);
            CHECK(std::abs(d_R) < 1.0e-6);
            CHECK(same(Tr, Ts));
            CHECK(same(Rr, Rs));
            CHECK(same(Er, Es));
        }

    // ------------------------------------------------------------------------------------------
    // A face that is closed at the recovered cosine
    // ------------------------------------------------------------------------------------------
    // The lossy host entered from a lossless dense solid with a transverse invariant above one:
    // the face to air is closed, the bit is right, and the reflection is evaluated at the
    // recovered cosine or, where the recovery declines, at the projection. The far medium's root
    // has an argument that is real and negative up to rounding.
    QD_DIAG("\n# diag: recovery gate, a closed face. Reflection at host | air inside eps 1.50 - 0.40j entered\n");
    QD_DIAG("# from a lossless eps 5.24 solid, invariant p above one, scalar. arg with code 1 and with code\n");
    QD_DIAG("# 33, with the dense solid named and with a sheet in the previous-medium word; step is the\n");
    QD_DIAG("# change of arg_1_named from the row above\n");
    QD_DIAG("p,ct_traced,dB_1,arg_1_named,arg_1_sheet,arg_33_named,arg_33_sheet,step,rtype\n");
    {
        const std::complex<double> e_d = diag_eps(DENSE, f_hz), e_l = diag_eps(LOSSY, f_hz);
        double prev_arg = 0.0;
        bool first = true;
        for (double pinv = 1.10; pinv < 1.46; pinv += 0.025)
        {
            double s1 = std::sqrt(pinv / std::real(e_d)), th1 = std::asin(s1);
            std::complex<double> eps_r = e_d / e_l, c2 = std::sqrt(1.0 - eps_r * s1 * s1);
            double eta = std::sqrt(std::abs(eps_r)), r = std::real(c2);
            double ct = r / std::sqrt((eta * s1) * (eta * s1) + r * r), st = std::sqrt(1.0 - ct * ct);
            auto A1 = event(3, OT_EXIT, 1u, LOSSY, 0, DENSE, LOSSY, th1, 1.5, ct, st);
            auto B1 = event(3, OT_EXIT, 1u, LOSSY, 0, SHEET, LOSSY, th1, 1.5, ct, st);
            auto A3 = event(3, 33, 1u, LOSSY, 0, DENSE, LOSSY, th1, 1.5, ct, st);
            auto B3 = event(3, 33, 1u, LOSSY, 0, SHEET, LOSSY, th1, 1.5, ct, st);
            auto deg = [&](const Rsu<double> &X)
            { return std::arg(cpx(X.xprmat, 0, 0)) * 180.0 / QPI; };
            double step = deg(A1) - prev_arg;
            step -= 360.0 * std::round(step / 360.0);
            QD_DIAG("%.3f,%.9f,%.6f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%u\n", pinv, ct, dB(std::norm(cpx(A1.xprmat, 0, 0))), deg(A1),
                    deg(B1), deg(A3), deg(B3), first ? 0.0 : step, (unsigned)A1.resolved_type[0]);
            INFO("closed face, p " << pinv);
            CHECK(std::abs(cpx(A1.xprmat, 0, 0) - cpx(B1.xprmat, 0, 0)) < 1.0e-9 * std::abs(cpx(A1.xprmat, 0, 0)));
            CHECK(same(A1, A3)); // the bit does not move the event
            CHECK(same(B1, B3));
            if (!first)
                CHECK(std::abs(step) < 45.0); // no branch flip between neighbours
            prev_arg = deg(A1), first = false;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Nothing synthetic: two events through the intersector
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: recovery gate, the traced run. air | slab 100 mm | air through ray_triangle_intersect,\n");
    QD_DIAG("# ray_mesh_interact and ray_state_update, two events, scalar, series resolved, against the\n");
    QD_DIAG("# analytic slab. itype 4 is the undeviated tracer, 5 the refracting one. out_deg is the\n");
    QD_DIAG("# direction written at the exit\n");
    QD_DIAG("itype,mtl,theta_ext,code,rtype,out_deg,chain_dB,slab_dB,d_dB\n");
    {
        arma::mat mesh = arma::join_cols(diag_quad_x(200.0, 0.0), diag_quad_x(200.0, L));
        mesh.row(2) = arma::rowvec{L, -200.0, -200.0, L, 200.0, -200.0, L, -200.0, 200.0}; // outward normal
        mesh.row(3) = arma::rowvec{L, 200.0, -200.0, L, 200.0, 200.0, L, -200.0, 200.0};
        struct Ev
        {
            arma::mat origN, destN, path_dirN, xpr;
            arma::Col<short> prev, cur, buf;
            arma::vec accg, accr;
            uint8_t ot = 0, rt = 0;
        };
        auto traced = [&](int itype, int m, const arma::mat &orig, const arma::mat &dest, const Ev *in) -> Ev
        {
            arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));
            arma::mat fbs, sbs, fbsN, sbsN, nv;
            arma::u32_vec nh, fi, si;
            quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
            Ev E;
            arma::vec gain, fang;
            std::vector<uint8_t> ot, rt;
            quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fi, &si, (arma::mat *)nullptr,
                                            (arma::mat *)nullptr, &E.origN, &E.destN, &fbsN, &sbsN, &gain, &E.xpr,
                                            (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                            (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
            arma::Col<short> mf(1), ms(1);
            mf(0) = fi(0) ? (short)m : (short)0, ms(0) = si(0) ? (short)m : (short)0;
            quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &fbsN, &sbsN, &nh, &fang, &nv, &ot, &mtl, &mf, &ms,
                                           in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                           in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                           &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                           (arma::u32_vec *)nullptr, 0.0);
            E.ot = ot[0], E.rt = rt[0];
            return E;
        };
        for (int itype : {4, 5})
            for (int m : {5, 6})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0;
                    std::complex<double> es = diag_eps(m, f_hz);
                    arma::mat orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev A = traced(itype, m, orig, dest, nullptr);
                    arma::mat d = A.destN - A.origN;
                    d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                    arma::mat dest2 = A.origN + 2.0 * d;
                    Ev B = traced(itype, m, A.origN, dest2, &A);
                    double ch = std::norm(cpx(A.xpr, 0, 0) * cpx(B.xpr, 0, 0));
                    double p_te = 0.0, p_tm = 0.0;
                    diag_slab_p_pol(es, L, th, f_hz, p_te, p_tm);
                    double out_deg = std::atan2(std::abs(B.path_dirN(0, 1)), std::abs(B.path_dirN(0, 0))) * 180.0 / QPI;
                    QD_DIAG("%d,%d,%.1f,%u,%u,%.4f,%.6f,%.6f,%+.6f\n", itype, m, thd, (unsigned)B.ot, (unsigned)B.rt, out_deg,
                            dB(ch), dB(p_te), dB(ch) - dB(p_te));
                    INFO("traced run, itype " << itype << ", material " << m << ", theta_ext " << thd);
                    REQUIRE((B.ot & 0x1F) == 1);
                    // The relaunch offset moves the leg by a few float ulp, so the bound is not
                    // the 1e-6 dB of a hand-built event. The refracting tracer's lossy rows are
                    // recorded: the recovery does not run where traced and geometric cosines agree
                    if (itype == 4 || m == 6)
                        CHECK(std::abs(dB(ch) - dB(p_te)) < 1.0e-3);
                }
    }
}

TEST_CASE("Direction - a ray leaves a parallel slab parallel to how it arrived")
{
    // Across parallel faces the transverse wavenumber of a wave that started in air is real and
    // conserved, so behind any stack of layers the direction in air has tangential part sqrt(p)
    // and normal part sqrt(1 - p) with p = sin^2(theta_ext): the direction the ray came in at.
    //
    // Measured at every writer of a direction into air, with the undeviated tracer, which is the
    // mode in which ray_state_update writes one: a glued exit, a glued exit behind a sheet and
    // behind an ordered stack inside the host, a bare exit and a bare exit of a resolved ray, the
    // last two through the intersector because a bare exit keeps what ray_mesh_interact wrote
    // unless ray_state_update replaces it. Lossless host as the control.
    //
    // d_t and d_n are the tangential and the normal component of the written unit vector minus
    // those of the incoming one. map_t and map_n are what the real-angle map makes of the traced
    // direction before normalization, minus sqrt(p) and sqrt(1 - p): the tangential shortfall
    // 1 / N of the projection, and the far cosine formed at a real angle in a lossy host.
    //
    // Three kinds of row keep the direction they had and are pinned to it: a host whose recovery
    // declines (no direction supplied, or faces known not parallel), which takes the real-angle
    // map; a closed outer pair, whose leak is undeviated; and a leak through a mass-law face,
    // at every face of the body. The refracting tracer's bare-exit rows are recorded.
    const double f_hz = FRQ, L = 0.10;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, SHEET2 = 8, DENSE = 2;
    const uint8_t OT_SHEET_ISO = 67, OT_STACK = 71, OT_GLUED_EXIT = 69;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};

    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, int buf, double th, double dx,
                     double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, buf, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th);
        C.run();
        return C;
    };
    // The real-angle map out of a medium into air at the real in-medium angle (c, s), unnormalized
    auto real_map = [](std::complex<double> e, double s, double &a, double &b)
    {
        a = std::sqrt(std::abs(e)) * s;
        b = std::real(std::sqrt(1.0 - e * s * s));
    };

    // ------------------------------------------------------------------------------------------
    // Glued exit: entered through a bare face, behind a sheet inside the host, behind a stack
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: direction, glued exit. air | host 100 mm | coating 6 mm | air, undeviated tracer, the\n");
    QD_DIAG("# exit along the direction a tracer holds. route: bare-entered, behind a sheet inside the\n");
    QD_DIAG("# host, behind an ordered stack inside the host. d_pass is the direction the sheet or stack\n");
    QD_DIAG("# event hands on against the one it received\n");
    QD_DIAG("route,host,theta_ext,out_deg,d_t,d_n,map_t,map_n,d_pass\n");
    for (int host : {5, 6})
        for (int route = 0; route < 3; ++route)
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                std::complex<double> eh = diag_eps(host, f_hz);
                double ct = diag_traced_ct(eh, th), st = std::sqrt(1.0 - ct * ct);
                double dx = ct, dy = st, d_pass = 0.0;
                int prev = 0, cur = host, buf = 0;
                if (route > 0)
                {
                    auto S = route == 1 ? event(4, OT_SHEET_ISO, 1u, SHEET, 0, 0, host, 0, th, ct, st)
                                        : event(4, OT_STACK, 2u, SHEET, SHEET2, 0, host, 0, th, ct, st);
                    REQUIRE(S.resolved_type[0] != 0);
                    dx = (double)S.path_dirN(0, 0), dy = (double)S.path_dirN(0, 1);
                    d_pass = std::max(std::abs(dx - ct), std::abs(dy - st));
                    prev = (int)S.prev_out(0), cur = (int)S.cur_out(0), buf = (int)S.buf_out(0);
                }
                auto X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, prev, cur, buf, th, dx, dy);
                double ox = std::abs((double)X.path_dirN(0, 0)), oy = std::abs((double)X.path_dirN(0, 1));
                double a, b;
                real_map(eh, st, a, b);
                QD_DIAG("%s,%d,%.1f,%.4f,%+.3e,%+.3e,%+.3e,%+.3e,%.3e\n", route == 0 ? "bare " : (route == 1 ? "sheet" : "stack"),
                        host, thd, std::atan2(oy, ox) * 180.0 / QPI, oy - std::sin(th), ox - std::cos(th), a - std::sin(th),
                        b - std::cos(th), d_pass);
                INFO("glued exit, route " << route << ", host " << host << ", theta_ext " << thd);
                CHECK(d_pass < 1.0e-12); // near == far: the event hands the direction on
                CHECK(std::abs(oy - std::sin(th)) < 1.0e-9);
                CHECK(std::abs(ox - std::cos(th)) < 1.0e-9);
            }

    // ------------------------------------------------------------------------------------------
    // Rows that keep the direction they have
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: direction, glued exits that keep it. nodir: no direction supplied, the reported angle\n");
    QD_DIAG("# is the in-medium one. wedge: faces known not parallel. Both take the real-angle map, d_map\n");
    QD_DIAG("# against it. closed: the outer pair is closed, the leak is undeviated, d_map against dir_prev\n");
    QD_DIAG("kind,host,theta,out_deg,d_map\n");
    for (int host : {5, 6})
        for (int kind = 0; kind < 2; ++kind)
            for (double thd : {10.0, 20.0, 40.0})
            {
                double th = thd * QPI / 180.0;
                std::complex<double> eh = diag_eps(host, f_hz);
                double c_in = std::cos(th), s_in = std::sin(th); // the real in-medium angle the map reads
                Rsu<double> X;
                if (kind == 0)
                {
                    X = make1<double>(mtl, 4, OT_GLUED_EXIT, 2u, SHEET, host, 0, host, 0, L / c_in, 0.5, 0.0, QPI / 2.0 - th, 1.5);
                    X.orig(0, 0) = X.fbs(0, 0) - L, X.orig(0, 1) = -L * std::tan(th);
                    X.dest(0, 0) = X.fbs(0, 0) + 0.5 * c_in, X.dest(0, 1) = 0.5 * s_in;
                    X.has_path_dirN = true;
                    X.path_dirN.zeros(1, 3);
                    X.path_dirN(0, 0) = c_in, X.path_dirN(0, 1) = s_in;
                    X.run();
                }
                else
                {
                    c_in = diag_traced_ct(eh, th), s_in = std::sqrt(1.0 - c_in * c_in);
                    X = event(4, OT_GLUED_EXIT, 2u, SHEET, host, (int)enc(0, true), host, 0, th, c_in, s_in);
                }
                double a, b;
                real_map(eh, s_in, a, b);
                double n = std::sqrt(a * a + b * b);
                double ox = std::abs((double)X.path_dirN(0, 0)), oy = std::abs((double)X.path_dirN(0, 1));
                double d_map = std::max(std::abs(ox - b / n), std::abs(oy - a / n));
                QD_DIAG("%s,%d,%.1f,%.4f,%.3e\n", kind ? "wedge " : "nodir ", host, thd, std::atan2(oy, ox) * 180.0 / QPI, d_map);
                INFO("kept, kind " << kind << ", host " << host << ", theta " << thd);
                CHECK(d_map < 1.0e-9);
            }
    for (double thd : {60.0, 70.0})
    {
        // Launched in the lossless host past its critical angle, the dense solid named
        double th = thd * QPI / 180.0;
        auto X = event(4, OT_GLUED_EXIT, 2u, SHEET, 6, DENSE, 6, 0, th, std::cos(th), std::sin(th));
        double d_map = std::max(std::abs((double)X.path_dirN(0, 0) - std::cos(th)), std::abs((double)X.path_dirN(0, 1) - std::sin(th)));
        QD_DIAG("closed,6,%.1f,%.4f,%.3e\n", thd,
                std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI, d_map);
        INFO("closed outer pair, theta " << thd);
        CHECK(rt_bit(X.resolved_type[0], 5));
        CHECK(d_map < 1.0e-12);
    }

    // ------------------------------------------------------------------------------------------
    // Bare exit, through the intersector
    // ------------------------------------------------------------------------------------------
    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, const arma::mat &orig, const arma::mat &dest, const Ev *in,
                      double eps) -> Ev
    {
        arma::mat fbs, sbs, fbsN, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &fbsN, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &fbsN, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    // The next segment: from where the event relaunched, 2 m along the direction it relaunched in
    auto onward = [](const Ev &E, arma::mat &orig, arma::mat &dest)
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + 2.0 * d;
    };
    // A face at x with its normal along -x (flip false) or +x (flip true)
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };

    QD_DIAG("\n# diag: direction, bare exit. air | slab 100 mm | air through the intersector, two events, and\n");
    QD_DIAG("# three for the resolved ray: enter, reflect at the far face with the series collapsed, leave\n");
    QD_DIAG("# through the front face, where the ray has to leave along the mirror of how it arrived.\n");
    QD_DIAG("# itype 4 is the undeviated tracer, 5 the refracting one, whose rows are recorded\n");
    QD_DIAG("itype,route,mtl,theta_ext,code,rtype,out_deg,d_t,d_n\n");
    {
        arma::mat mesh = arma::join_cols(face(0.0, false), face(L, true));
        for (int itype : {4, 5})
            for (int route = 0; route < 2; ++route)
                for (int m : {5, 6})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0;
                        arma::uvec mtl_ind(mesh.n_rows, arma::fill::value((arma::uword)m));
                        arma::mat orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev A = traced(mtl, mesh, mtl_ind, itype, orig, dest, nullptr, 0.0);
                        onward(A, orig, dest);
                        Ev B = traced(mtl, mesh, mtl_ind, route ? 3 : itype, orig, dest, &A, 0.0);
                        if (route)
                        {
                            REQUIRE(rt_bit(B.rt, 2)); // the reflection collapsed the series
                            onward(B, orig, dest);
                            B = traced(mtl, mesh, mtl_ind, itype, orig, dest, &B, 0.0);
                        }
                        double ox = std::abs(B.path_dirN(0, 0)), oy = std::abs(B.path_dirN(0, 1));
                        QD_DIAG("%d,%s,%d,%.1f,%u,%u,%.4f,%+.3e,%+.3e\n", itype, route ? "resolved" : "through ", m, thd,
                                (unsigned)B.ot, (unsigned)B.rt, std::atan2(oy, ox) * 180.0 / QPI, oy - std::sin(th), ox - std::cos(th));
                        INFO("bare exit, itype " << itype << ", route " << route << ", material " << m << ", theta_ext " << thd);
                        REQUIRE((B.ot & 0x1F) == 1);
                        REQUIRE(mat_of(B.cur(0)) == 0);
                        CHECK((B.path_dirN(0, 0) > 0.0) == (route == 0)); // onward, or back out of the front face
                        if (itype == 4)
                        {
                            CHECK(std::abs(oy - std::sin(th)) < 1.0e-9);
                            CHECK(std::abs(ox - std::cos(th)) < 1.0e-9);
                        }
                    }
    }

    // ------------------------------------------------------------------------------------------
    // A leak through a mass-law face
    // ------------------------------------------------------------------------------------------
    // Rigid materials, eps = (Z_air / Z)^2: a wave enters as a tf leak and is held undeviated
    // at every face of the body, towards either index: as two traced solids, through a sheet
    // whose outer pair is closed, and at every exit into air.
    QD_DIAG("\n# diag: direction, a leak. Wood eps 2.1609e-8, concrete eps 2.401e-9, both with tf and the mass\n");
    QD_DIAG("# column, 30 deg in, undeviated tracer. One row per event; kept says whether the row is\n");
    QD_DIAG("# asserted undeviated (1)\n");
    QD_DIAG("stack,event,code,rtype,out_deg,d_deg,kept\n");
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8};
        pal["fRef"] = {0.001, 0.001, 0.001};
        pal["tf"] = {0.2, 0.1, 0.2};
        pal["m"] = {20.0, 30.0, 20.0};
        pal["thk"] = {0.0, 0.0, 0.02};
        const int WOOD = 1, CONC = 2, PANEL = 3;
        const double g = 1.0e-5, thd = 30.0;
        struct Stack
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
            std::vector<int> kept; // per event
        };
        const std::vector<Stack> stacks = {
            {"air | wood 50 | conc 50 | air  ", {0.0, 0.05, 0.05 + g, 0.10}, {WOOD, WOOD, CONC, CONC}, {false, true, false, true}, {1, 1, 1}},
            {"air | conc 50 | wood 50 | air  ", {0.0, 0.05, 0.05 + g, 0.10}, {CONC, CONC, WOOD, WOOD}, {false, true, false, true}, {1, 1, 1}},
            {"air | panel | conc 100 | air   ", {0.0, g, 0.10}, {PANEL, CONC, CONC}, {false, false, true}, {1, 1}},
            {"air | conc 100 | panel | air   ", {0.0, 0.10, 0.10 + g}, {CONC, CONC, PANEL}, {false, true, true}, {1, 1}},
            {"air | panel | air              ", {0.0}, {PANEL}, {false}, {1}},
        };
        for (const auto &S : stacks)
        {
            arma::mat mesh;
            arma::uvec mtl_ind(2 * S.x.size());
            for (size_t k = 0; k < S.x.size(); ++k)
            {
                mesh = k ? arma::join_cols(mesh, face(S.x[k], S.flip[k])) : face(S.x[k], S.flip[k]);
                mtl_ind(2 * k) = (arma::uword)S.mat[k], mtl_ind(2 * k + 1) = (arma::uword)S.mat[k];
            }
            arma::mat orig, dest;
            diag_ray_at(thd, orig, dest);
            Ev P;
            const Ev *in = nullptr;
            for (size_t k = 0; k < S.kept.size(); ++k)
            {
                Ev E = traced(pal, mesh, mtl_ind, 4, orig, dest, in, 1.5);
                REQUIRE(E.nh != 0);
                double out = std::atan2(std::abs(E.path_dirN(0, 1)), std::abs(E.path_dirN(0, 0))) * 180.0 / QPI;
                QD_DIAG("%s,%d,%u,%u,%.4f,%+.4f,%d\n", S.name, (int)k, (unsigned)E.ot, (unsigned)E.rt, out, out - thd, S.kept[k]);
                INFO(S.name << ", event " << k);
                if (S.kept[k])
                    CHECK(std::abs(out - thd) < 1.0e-9);
                onward(E, orig, dest);
                P = E, in = &P;
            }
        }
    }
}

TEST_CASE("Refracting tracer - a traced bend is known from the state and the leg is exact")
{
    // With types 2 and 5 ray_mesh_interact bends the segment itself, so inside a solid the traced
    // and the geometric cosine coincide and nothing can be read off their difference. That the
    // leg follows a bend is a fact of the polyline, and the event that entered the medium knows
    // it: bit 14 of the current-medium word, set where a type 2 or 5 event refracted into the
    // medium, clear where it entered under total reflection, carried while the ray stays in the
    // medium and zero in air. Every consumer of the recovered cosine then runs on that leg as it
    // does for the undeviated tracer, and the traversal takes no walk-off, since the lateral shift
    // is in the polyline.
    //
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update.
    // The quantity compared is the absolute transfer between the first and the last hit point,
    // the chain coefficient with the free-space reference over the in-wall polyline put back:
    //   - against the analytic slab at the polyline's exit point, power and phase;
    //   - tracer 5 against tracer 4. Each traversal carries the phase at the point its own
    //     polyline leaves at, so each absolute transfer is taken there, times
    //     exp(+j k0 sin(theta) dy) with dy that polyline's lateral walk, and the two are one number.
    // Recorded and not asserted: the trajectory (destN - origN) out of a lossy wall and a mass-law
    // solid the wave is refracted into.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, gap = 1.0e-5;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const short BENT = (short)0x4000;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    auto dB = [](double g)
    { return 10.0 * std::log10(g); };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    // relaunch: hand ray_state_update the geometry ray_mesh_interact built, so that the segment out of
    // a lossy medium follows the direction it writes. Off for every table but the last one.
    bool relaunch = false;
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, double f, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, const Ev *in_state, double eps) -> Ev
    {
        // in: direction and refracted ledger; in_state: the state words and the geometric ledger,
        // which in a multi-frequency run are the base frequency's
        if (!in_state)
            in_state = in;
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in_state->prev : nullptr, in ? &in_state->cur : nullptr, in ? &in_state->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in_state->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        std::complex<double> vv = 1.0, hh = 1.0; // chain coefficient, VV (TE, scalar) and HH (TM)
        double Lin = 0.0, dy = 0.0;              // in-wall polyline length, lateral walk first to last hit
        double out_x = 0.0, out_y = 0.0, traj = 0.0;
        std::vector<Ev> ev;
        bool ok = true;
        std::complex<double> abs_vv(double k) const { return vv * std::exp(std::complex<double>(0.0, -k * Lin)); }
    };
    // seq: one type repeated until the ray is out, or one type per event. f_base != 0: geometry
    // and state words at f_base, coefficient and direction at f, as ray_progress runs a band.
    auto run = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh, const arma::uvec &mtl_ind,
                   std::vector<int> seq, double thd, double eps, double f, double f_base) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P, Pb;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            Ev B = traced(pal, mesh, mtl_ind, itype, f_base != 0.0 ? f_base : f, orig, dest, have ? &Pb : nullptr, nullptr, eps);
            if (B.nh == 0)
                break;
            Ev E = f_base != 0.0 ? traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, have ? &Pb : nullptr, eps) : B;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0), C.hh *= cpx(E.xpr, 0, (E.xpr.n_rows == 8) ? 3 : 0);
            double fx = B.fbs(0, 0), fy = B.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(B.cur(0)) != 0;
            C.out_x = std::abs(E.path_dirN(0, 0)), C.out_y = std::abs(E.path_dirN(0, 1));
            arma::mat d = B.destN - B.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            C.traj = std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI;
            orig = B.origN, dest = B.origN + 2.0 * d;
            P = E, Pb = B, have = true;
            C.ev.push_back(E);
        }
        return C;
    };
    // The analytic slab, TE, complex: transmit, and the part of the reflect port that entered
    // The thickness is the one the polyline crossed: ray_mesh_interact relaunches a few float ulp
    // of the coordinate magnitude past the face, along the outgoing direction, so the leg is short
    // by the normal part of that, 1e-6 m for the undeviated tracer and up to 1e-5 m for the
    // refracting one at 85 deg, which the series of a dense slab turns into 4e-3 dB.
    auto slab = [&](std::complex<double> es, double th, double f, bool series, double Lx, std::complex<double> &t, std::complex<double> &r_in)
    {
        std::complex<double> c1 = std::sqrt(1.0 - std::sin(th) * std::sin(th) / es), n = std::sqrt(es);
        std::complex<double> a = std::cos(th), b = n * c1, r10 = (b - a) / (b + a);
        std::complex<double> t01 = 2.0 * a / (a + b), t10 = 2.0 * b / (a + b);
        std::complex<double> phi = std::exp(std::complex<double>(0.0, -1.0) * (2.0 * QPI * f / C0) * n * c1 * Lx);
        std::complex<double> S = series ? 1.0 / (1.0 - r10 * r10 * phi * phi) : std::complex<double>(1.0, 0.0);
        t = t01 * t10 * phi * S, r_in = t01 * r10 * t10 * phi * phi * S;
    };

    // ------------------------------------------------------------------------------------------
    // Bare slab: the state, the chain against the analytic slab, the two tracers, the direction
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: refracting tracer, bare slab. air | 100 mm | air through the intersector, series resolved.\n");
    QD_DIAG("# d_dB, ph: tracer 5 against the analytic slab at the polyline's exit point. r54: tracer 5 over 4,\n");
    QD_DIAG("# absolute transfer. tm_dB: tracer 2, HH against the TM slab. cur: current word after the entry,\n");
    QD_DIAG("# hex. out: path_dirN at the exit, traj: destN - origN there, deg, recorded\n");
    QD_DIAG("mtl,theta,cur4,cur5,rt5,d_dB,ph_deg,r54_dB,r54_deg,te2_dB,tm_dB,out_deg,traj_deg\n");
    for (int m : {5, 6, 1, 4})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            arma::uvec mi;
            arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
            Chain c4 = run(mtl, mesh, mi, {4}, thd, 0.0, f_hz, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, 0.0, f_hz, 0.0);
            Chain c2 = run(mtl, mesh, mi, {2}, thd, 0.0, f_hz, 0.0);
            REQUIRE((c4.ok && c5.ok && c2.ok));
            REQUIRE((c4.ev.size() == 2 && c5.ev.size() == 2 && c2.ev.size() == 2));
            std::complex<double> t, t4, r_in;
            slab(diag_eps(m, f_hz), th, f_hz, true, L - c5.ev[0].origN(0, 0), t, r_in);
            slab(diag_eps(m, f_hz), th, f_hz, true, L - c4.ev[0].origN(0, 0), t4, r_in);
            std::complex<double> q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t;
            // each over the slab it crossed, each at the point its own polyline leaves at
            std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t) /
                                       (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy)) / t4);
            double p_te = 0.0, p_tm = 0.0;
            diag_slab_p_pol(diag_eps(m, f_hz), L - c2.ev[0].origN(0, 0), th, f_hz, p_te, p_tm);
            QD_DIAG("%d,%.1f,%04x,%04x,%u,%+.6f,%+.4f,%+.6f,%+.4f,%+.6f,%+.6f,%.4f,%.4f\n", m, thd, (unsigned)(unsigned short)c4.ev[0].cur(0),
                    (unsigned)(unsigned short)c5.ev[0].cur(0), (unsigned)c5.ev[1].rt, dB(std::norm(q)), deg(q), dB(std::norm(r54)), deg(r54),
                    dB(std::norm(c2.vv) / p_te), dB(std::norm(c2.hh) / p_tm), std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI, c5.traj);
            INFO("bare slab, material " << m << ", theta_ext " << thd);
            // The state: the bent leg says so, with either refracting type, and the undeviated one never
            CHECK(c5.ev[0].cur(0) == (short)(m | BENT));
            CHECK(c2.ev[0].cur(0) == (short)(m | BENT));
            CHECK(c4.ev[0].cur(0) == (short)m);
            CHECK(c5.ev[1].cur(0) == (short)0); // zero in air
            CHECK(c5.ev[1].prev(0) == c4.ev[1].prev(0));
            CHECK(c5.ev[1].rt == 5); // the VBS bit keeps its meaning: the VBS is the FBS here
            // The chain
            CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
            CHECK(std::abs(deg(q)) < 0.05);
            CHECK(std::abs(dB(std::norm(r54))) < 1.0e-3);
            CHECK(std::abs(deg(r54)) < 0.05);
            CHECK(std::abs(dB(std::norm(c2.vv) / p_te)) < 1.0e-3);
            CHECK(std::abs(dB(std::norm(c2.hh) / p_tm)) < 1.0e-3);
            // The direction written at the exit is the one the ray arrived in
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }

    // ------------------------------------------------------------------------------------------
    // Sheets on and in the host, and two solids: tracer 5 against tracer 4
    // ------------------------------------------------------------------------------------------
    struct Stack
    {
        const char *name;
        std::vector<double> x;
        std::vector<int> mat;
        std::vector<bool> flip;
        double eps;
        bool phase; // the phase is asserted
        int sheet_ev; // event after which the ray is behind a sheet inside the host, or -1
    };
    std::vector<Stack> stacks;
    for (int h : {5, 6})
    {
        stacks.push_back({"glued_entry ", {0.0, gap, L}, {SHEET, h, h}, {false, false, true}, 0.0, true, -1});
        stacks.push_back({"glued_exit  ", {0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, 0.0, true, -1});
        stacks.push_back({"lined       ", {0.0, gap, L, L + gap}, {SHEET, h, h, SHEET}, {false, false, true, true}, 0.0, true, -1});
        stacks.push_back({"sheet_inside", {0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, 0.0, true, 1});
    }
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {5, 1}, {2, 5}, {2, 6}, {1, 6}})
        stacks.push_back({"two_solids  ", {0.0, 0.05, 0.05 + gap, 0.15 + gap}, {P.first, P.first, P.second, P.second},
                          {false, true, false, true}, 1.5, true, -1});
    QD_DIAG("\n# diag: refracting tracer, layered walls. Tracer 5 over tracer 4, absolute transfer, scalar. Sheets:\n");
    QD_DIAG("# series resolved. two_solids: series declined. Each tracer at the point its polyline leaves at.\n");
    QD_DIAG("# words: the last in-wall state words of tracer 5, hex\n");
    QD_DIAG("stack,mats,theta,rt4,rt5,r54_dB,r54_deg,prev5,cur5,out_deg,traj_deg\n");
    for (const auto &S : stacks)
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0;
            arma::uvec mi;
            arma::mat mesh = build(S.x, S.mat, S.flip, mi);
            Chain c4 = run(mtl, mesh, mi, {4}, thd, S.eps, f_hz, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, S.eps, f_hz, 0.0);
            REQUIRE((c4.ok && c5.ok));
            REQUIRE(c4.ev.size() == c5.ev.size());
            size_t n = c5.ev.size();
            std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy))) /
                                       (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy)));
            std::string mats, r4, r5;
            for (int v : S.mat)
                mats += std::to_string(v);
            for (size_t k = 0; k < n; ++k)
                r4 += std::to_string((unsigned)c4.ev[k].rt) + "/", r5 += std::to_string((unsigned)c5.ev[k].rt) + "/";
            QD_DIAG("%s,%s,%.1f,%s,%s,%+.6f,%+.4f,%04x,%04x,%.4f,%.4f\n", S.name, mats.c_str(), thd, r4.c_str(), r5.c_str(), dB(std::norm(r54)),
                    deg(r54), (unsigned)(unsigned short)c5.ev[n - 2].prev(0), (unsigned)(unsigned short)c5.ev[n - 2].cur(0),
                    std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI, c5.traj);
            INFO(S.name << " " << mats << ", theta_ext " << thd);
            // No reference here, so the two polylines are compared as they are, each short by its
            // own relaunch offsets: 1.1e-3 dB on the lined lossless wall at 85 deg, below 6e-4 elsewhere
            CHECK(std::abs(dB(std::norm(r54))) < 2.0e-3);
            if (S.phase)
                CHECK(std::abs(deg(r54)) < 0.05);
            for (size_t k = 0; k < n; ++k)
            {
                // The words of the two tracers differ in the one bit and nowhere else, and the bit
                // is set on every in-wall leg: carried through a sheet inside the host, written
                // again by a transition that refracted
                bool in_wall = mat_of(c5.ev[k].cur(0)) != 0;
                CHECK(c5.ev[k].prev(0) == c4.ev[k].prev(0));
                CHECK(c5.ev[k].buf(0) == c4.ev[k].buf(0));
                CHECK(c5.ev[k].cur(0) == (short)(c4.ev[k].cur(0) | (in_wall ? BENT : (short)0)));
                CHECK((c4.ev[k].cur(0) & BENT) == 0);
            }
            if (S.sheet_ev >= 0)
                CHECK((c5.ev[(size_t)S.sheet_ev].cur(0) & BENT) != 0);
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }

    // ------------------------------------------------------------------------------------------
    // Back reflection: the reflect pass does not name the tracer, the word does
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: refracting tracer, back reflection. Enter (5), reflect at the far face (3), leave by the\n");
    QD_DIAG("# front face (5), against t01 r10 t10 phi^2 [S] at the polyline's exit point. eps 0: the reflection\n");
    QD_DIAG("# collapses the series and the ray leaves resolved. cur: the word after the reflection, hex\n");
    QD_DIAG("mtl,eps,theta,rt5,cur5,d_dB,ph_deg\n");
    for (int m : {5, 6})
        for (double eps : {1.5, 0.0})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0;
                arma::uvec mi;
                arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                Chain c5 = run(mtl, mesh, mi, {5, 3, 5}, thd, eps, f_hz, 0.0);
                REQUIRE((c5.ok && c5.ev.size() == 3));
                std::complex<double> t, r_in;
                slab(diag_eps(m, f_hz), th, f_hz, eps == 0.0, L - c5.ev[0].origN(0, 0), t, r_in);
                std::complex<double> q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / r_in;
                std::string r5;
                for (const auto &E : c5.ev)
                    r5 += std::to_string((unsigned)E.rt) + "/";
                QD_DIAG("%d,%.1f,%.1f,%s,%04x,%+.6f,%+.4f\n", m, eps, thd, r5.c_str(), (unsigned)(unsigned short)c5.ev[1].cur(0), dB(std::norm(q)), deg(q));
                INFO("back reflection, material " << m << ", eps " << eps << ", theta_ext " << thd);
                CHECK((c5.ev[1].cur(0) & BENT) != 0); // a reflection keeps the leg's bit
                CHECK(mat_of(c5.ev[1].cur(0)) == m);
                CHECK(flag_of(c5.ev[1].cur(0)) == (eps == 0.0));
                CHECK(c5.ev[2].cur(0) == (short)0);
                CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
                CHECK(std::abs(deg(q)) < 0.05);
                CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9); // leaves along the mirror of how it arrived
                CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
            }

    // ------------------------------------------------------------------------------------------
    // A leak is not a bend: rigid bodies, entered under total reflection
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: refracting tracer, a leak. Wood eps 2.1609e-8, concrete eps 2.401e-9, tf and the mass column,\n");
    QD_DIAG("# 30 deg. The entry is a total reflection, the wave is held at every face, the bit stays clear and\n");
    QD_DIAG("# every event is the undeviated tracer's. Recorded: DENSE, eps 1.50 with m 20, a mass-law solid\n");
    QD_DIAG("# the wave is refracted into\n");
    QD_DIAG("stack,tracer,event,code,rtype,gain_dB,out_deg,prev,cur,buf\n");
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8, 1.5};
        pal["fRef"] = {0.001, 0.001, 0.001, 0.001};
        pal["tf"] = {0.2, 0.1, 0.2, 0.0};
        pal["m"] = {20.0, 30.0, 20.0, 20.0};
        pal["thk"] = {0.0, 0.0, 0.02, 0.0};
        struct St
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
            bool rigid;
        };
        const std::vector<St> sts = {
            {"air | wood 50 | conc 50 | air", {0.0, 0.05, 0.05 + gap, L}, {1, 1, 2, 2}, {false, true, false, true}, true},
            {"air | conc 50 | wood 50 | air", {0.0, 0.05, 0.05 + gap, L}, {2, 2, 1, 1}, {false, true, false, true}, true},
            {"air | panel | conc 100 | air ", {0.0, gap, L}, {3, 2, 2}, {false, false, true}, true},
            {"air | conc 100 | panel | air ", {0.0, L, L + gap}, {2, 2, 3}, {false, true, true}, true},
            {"air | DENSE m20 100 | air    ", {0.0, L}, {4, 4}, {false, true}, false},
        };
        for (const auto &S : sts)
        {
            arma::uvec mi;
            arma::mat mesh = build(S.x, S.mat, S.flip, mi);
            Chain c4 = run(pal, mesh, mi, {4}, 30.0, 1.5, f_hz, 0.0), c5 = run(pal, mesh, mi, {5}, 30.0, 1.5, f_hz, 0.0);
            REQUIRE(c4.ev.size() == c5.ev.size());
            for (int t = 0; t < 2; ++t)
            {
                const Chain &C = t ? c5 : c4;
                for (size_t k = 0; k < C.ev.size(); ++k)
                    QD_DIAG("%s,%d,%d,%u,%u,%+.6f,%.4f,%04x,%04x,%04x\n", S.name, t ? 5 : 4, (int)k, (unsigned)C.ev[k].ot, (unsigned)C.ev[k].rt,
                            dB(std::norm(cpx(C.ev[k].xpr, 0, 0))),
                            std::atan2(std::abs(C.ev[k].path_dirN(0, 1)), std::abs(C.ev[k].path_dirN(0, 0))) * 180.0 / QPI,
                            (unsigned)(unsigned short)C.ev[k].prev(0), (unsigned)(unsigned short)C.ev[k].cur(0),
                            (unsigned)(unsigned short)C.ev[k].buf(0));
            }
            if (!S.rigid)
                continue;
            INFO(S.name);
            for (size_t k = 0; k < c5.ev.size(); ++k)
            {
                CHECK(c5.ev[k].cur(0) == c4.ev[k].cur(0)); // no bit: the wave was not bent into the body
                CHECK(c5.ev[k].prev(0) == c4.ev[k].prev(0));
                CHECK(c5.ev[k].buf(0) == c4.ev[k].buf(0));
                CHECK(c5.ev[k].rt == c4.ev[k].rt);
                CHECK(cpx(c5.ev[k].xpr, 0, 0) == cpx(c4.ev[k].xpr, 0, 0));
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // A second frequency of a run, recorded
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: refracting tracer, a second frequency. Geometry and state words at 1 GHz, coefficient and\n");
    QD_DIAG("# direction at f1, as ray_progress runs a band. eps 1.50 - 0.40j at 1 GHz, 100 mm, against the slab at\n");
    QD_DIAG("# f1 at the polyline's exit point, power and phase\n");
    QD_DIAG("f1_GHz,theta,d_dB,ph_deg,out_deg\n");
    for (double f1 : {0.5e9, 2.0e9})
        for (double thd : angles)
        {
            double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0;
            arma::uvec mi;
            arma::mat mesh = build({0.0, L}, {5, 5}, {false, true}, mi);
            Chain c5 = run(mtl, mesh, mi, {5}, thd, 0.0, f1, f_hz);
            REQUIRE((c5.ok && c5.ev.size() == 2));
            std::complex<double> t, r_in;
            slab(diag_eps(5, f1), th, f1, true, L - c5.ev[0].origN(0, 0), t, r_in);
            std::complex<double> q = c5.abs_vv(k1) * std::exp(std::complex<double>(0.0, k1 * std::sin(th) * c5.dy)) / t;
            QD_DIAG("%.1f,%.1f,%+.6f,%+.4f,%.4f\n", f1 * 1e-9, thd, dB(std::norm(q)), deg(q), std::atan2(c5.out_y, c5.out_x) * 180.0 / QPI);
            INFO("second frequency " << f1 << ", theta_ext " << thd);
            CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
            CHECK(std::abs(deg(q)) < 0.05);
            CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
            CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
        }
    // ------------------------------------------------------------------------------------------
    // The trajectory: the segment out of a lossy medium is the direction written
    // ------------------------------------------------------------------------------------------
    // ray_state_update is handed origN and destN and puts them on one line through the point the
    // ray leaves at, along the direction it writes, wherever that direction is the one of a wave
    // out of a lossy medium whose cosine was recovered. Everything else of the event is what it
    // was, so the chain is the one of the tables above, and the leg in a second solid now runs
    // where the transition says the wave runs.
    relaunch = true;
    auto lossy = [&](int m)
    { return std::imag(diag_eps(m, f_hz)) != 0.0; };
    auto seg_err = [](const Ev &E) -> double // largest component of the unit segment against path_dirN
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        double e = 0.0;
        for (int c = 0; c < 3; ++c)
            e = std::max(e, std::abs(d(0, c) - E.path_dirN(0, c)));
        return e;
    };
    QD_DIAG("\n# diag: refracting tracer, the trajectory. As the tables above, with origN and destN handed to\n");
    QD_DIAG("# ray_state_update. traj: destN - origN behind the wall, deg. d_out: that segment against path_dirN,\n");
    QD_DIAG("# largest component. d_in: the same for the worst event that leaves a lossy medium, the transition\n");
    QD_DIAG("# of two solids included. d_dB, ph: the bare slab against the analytic one. r54: tracer 5 over 4.\n");
    QD_DIAG("# rt: the resolved types, with the geometry and without\n");
    QD_DIAG("stack,mats,theta,tracer,rt,rt_without,traj_deg,d_out,d_in,d_dB,ph_deg,r54_dB,r54_deg\n");
    {
        std::vector<Stack> all;
        for (int m : {5, 6, 1, 4})
            all.push_back({"bare        ", {0.0, L}, {m, m}, {false, true}, 0.0, true, -1});
        for (const auto &S : stacks)
            all.push_back(S);
        for (const auto &S : all)
            for (double thd : angles)
                for (int itype : {5, 2})
                {
                    bool bare = S.x.size() == 2;
                    if (itype == 2 && !bare)
                        continue;
                    double th = thd * QPI / 180.0;
                    arma::uvec mi;
                    arma::mat mesh = build(S.x, S.mat, S.flip, mi);
                    Chain c5 = run(mtl, mesh, mi, {itype}, thd, S.eps, f_hz, 0.0);
                    relaunch = false;
                    Chain c4 = run(mtl, mesh, mi, {4}, thd, S.eps, f_hz, 0.0), c0 = run(mtl, mesh, mi, {itype}, thd, S.eps, f_hz, 0.0);
                    relaunch = true;
                    REQUIRE((c4.ok && c5.ok && c0.ok));
                    REQUIRE((c4.ev.size() == c5.ev.size() && c0.ev.size() == c5.ev.size()));
                    size_t n = c5.ev.size();
                    double d_in = 0.0;
                    for (size_t k = 1; k < n; ++k)
                    {
                        int left = mat_of(c5.ev[k - 1].cur(0));
                        if (lossy(left) && mat_of(c5.ev[k].cur(0)) != left)
                            d_in = std::max(d_in, seg_err(c5.ev[k]));
                    }
                    std::complex<double> r54 = (c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy))) /
                                               (c4.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c4.dy))),
                                         q(1.0, 0.0);
                    if (bare)
                    {
                        std::complex<double> t, t4, r_in;
                        slab(diag_eps(S.mat[0], f_hz), th, f_hz, true, L - c5.ev[0].origN(0, 0), t, r_in);
                        slab(diag_eps(S.mat[0], f_hz), th, f_hz, true, L - c4.ev[0].origN(0, 0), t4, r_in);
                        q = c5.abs_vv(k0) * std::exp(std::complex<double>(0.0, k0 * std::sin(th) * c5.dy)) / t;
                        r54 *= t4 / t; // each over the slab it crossed
                    }
                    std::string mats, r5, r0;
                    for (int v : S.mat)
                        mats += std::to_string(v);
                    for (size_t k = 0; k < n; ++k)
                        r5 += std::to_string((unsigned)c5.ev[k].rt) + "/", r0 += std::to_string((unsigned)c0.ev[k].rt) + "/";
                    QD_DIAG("%s,%s,%.1f,%d,%s,%s,%.4f,%.3e,%.3e,%+.6f,%+.4f,%+.6f,%+.4f\n", S.name, mats.c_str(), thd, itype, r5.c_str(), r0.c_str(), c5.traj,
                            seg_err(c5.ev[n - 1]), d_in, dB(std::norm(q)), deg(q), dB(std::norm(r54)), deg(r54));
                    INFO("trajectory, " << S.name << " " << mats << ", theta_ext " << thd << ", type " << itype);
                    // The segment behind the wall is the direction written, and that is the one the ray arrived in
                    CHECK(std::abs(c5.out_y - std::sin(th)) < 1.0e-9);
                    CHECK(std::abs(c5.out_x - std::cos(th)) < 1.0e-9);
                    CHECK(d_in < 1.0e-9);
                    if (lossy(mat_of(c5.ev[n - 2].cur(0))))
                        CHECK(seg_err(c5.ev[n - 1]) < 1.0e-9);
                    // Nothing else of an event moves: the words and the type codes. One bit follows the
                    // geometry by its own definition: the VBS bit says that the direction the wave has
                    // differs from the segment, which behind a lossy first solid it did and no longer does
                    for (size_t k = 0; k < n; ++k)
                    {
                        CHECK(c5.ev[k].prev(0) == c0.ev[k].prev(0));
                        CHECK(c5.ev[k].cur(0) == c0.ev[k].cur(0));
                        CHECK(c5.ev[k].buf(0) == c0.ev[k].buf(0));
                        CHECK((c5.ev[k].rt & 0xFD) == (c0.ev[k].rt & 0xFD));
                        CHECK((c5.ev[k].rt & 0x02) <= (c0.ev[k].rt & 0x02));
                        if (k > 0 && lossy(mat_of(c5.ev[k - 1].prev(0))) && mat_of(c5.ev[k - 1].prev(0)) != SHEET && mat_of(c5.ev[k - 1].cur(0)) != 0)
                            CHECK((c5.ev[k].rt & 0x02) == 0); // the leg behind a lossy solid: the VBS is the FBS
                    }
                    // The chain
                    if (itype == 5)
                    {
                        CHECK(std::abs(dB(std::norm(r54))) < 2.0e-3);
                        if (S.phase)
                            CHECK(std::abs(deg(r54)) < 0.05);
                    }
                    if (bare)
                    {
                        CHECK(std::abs(dB(std::norm(q))) < 1.0e-3);
                        CHECK(std::abs(deg(q)) < 0.05);
                        // One wall: the events are the ones of the run without the geometry, bit for bit
                        for (size_t k = 0; k < n; ++k)
                            CHECK(cpx(c5.ev[k].xpr, 0, 0) == cpx(c0.ev[k].xpr, 0, 0));
                    }
                }
    }
    relaunch = false;
}

TEST_CASE("Phase reference - a traversal carries the phase at the point the polyline leaves at")
{
    // The tracer commits a polyline, and the path length, the delay and the free-space phase of a
    // path are taken from it. A coefficient is therefore owed the field at the point where the
    // polyline leaves a body. For a plane wave that field is the stack's transfer on the thickness
    // axis times exp(-j k0 sin(theta) dy), with dy the lateral walk of the polyline, whatever the
    // polyline is: one path, and per frequency the value the field has where that path leaves. The
    // geometry is never moved for it. Everything runs through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update.
    //
    // Two points are compared on every row:
    //   ray:  the point where the refracted ray of that frequency leaves, lateral walk s
    //   poly: the point where the polyline leaves, lateral walk dy
    // ph_x = arg( coefficient * exp(-j k0 L_in) * exp(+j k0 sin(theta) lateral_x) / reference ).
    // A band is run as ray_progress runs it: geometry, state words and the geometric ledger from
    // the base-frequency event, with origN and destN handed in on the base-frequency transmit call,
    // and the coefficient, the direction and the refracted ledger from the event at that frequency.
    const double L = 0.10;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    auto dB = [](double g)
    { return 10.0 * std::log10(g); };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };
    const std::complex<double> J(0.0, 1.0);

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh,
                      const arma::uvec &mtl_ind, int itype, double f, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, const Ev *in_state, double eps, bool relaunch) -> Ev
    {
        if (!in_state)
            in_state = in;
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in_state->prev : nullptr, in ? &in_state->cur : nullptr, in ? &in_state->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in_state->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev, evb;    // the events at f, and at the base frequency (geometry, words)
        // per event, for the leg it closes: polyline length and lateral walk between the two hit
        // points, the length the ledger was given (relaunch point to hit point), its normal part,
        // and the lateral offset of the relaunch point from the hit point it left
        std::vector<double> len, lat, led, ledx, ledy, offy;
        bool ok = true;
        std::complex<double> abs_vv(double k) const { return vv * std::exp(std::complex<double>(0.0, -k * Lin)); }
    };
    auto run = [&](const std::unordered_map<std::string, std::vector<double>> &pal, const arma::mat &mesh, const arma::uvec &mtl_ind,
                   std::vector<int> seq, double thd, double eps, double f, double f_base) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P, Pb;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            bool tx = itype != 0 && itype != 3;
            Ev B = traced(pal, mesh, mtl_ind, itype, f_base != 0.0 ? f_base : f, orig, dest, have ? &Pb : nullptr, nullptr, eps, tx);
            if (B.nh == 0)
                break;
            Ev E = f_base != 0.0 ? traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, have ? &Pb : nullptr, eps, false) : B;
            if (E.rt == 0 || B.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = B.fbs(0, 0), fy = B.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            double seg = std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            double ox = fx - orig(0, 0), oy = fy - orig(0, 1);
            C.len.push_back(inside ? seg : 0.0), C.lat.push_back(inside ? std::abs(fy - yp) : 0.0);
            C.led.push_back(inside ? std::sqrt(ox * ox + oy * oy) : 0.0), C.ledx.push_back(inside ? std::abs(ox) : 0.0), C.ledy.push_back(inside ? std::abs(oy) : 0.0);
            C.offy.push_back(std::abs(B.origN(0, 1) - fy));
            if (inside)
                C.Lin += seg;
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(B.cur(0)) != 0;
            arma::mat d = B.destN - B.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = B.origN, dest = B.origN + 2.0 * d;
            P = E, Pb = B, have = true;
            C.ev.push_back(E), C.evb.push_back(B);
        }
        return C;
    };
    auto tan_of = [](const Ev &E) -> double // tangent of the written direction against the x axis
    { return std::abs(E.path_dirN(0, 1)) / std::abs(E.path_dirN(0, 0)); };

    // One face pair air | medium, scalar (TE), in admittances: z c, with z = sqrt(eps / mu)
    struct Sl
    {
        std::complex<double> t01, t10, r10, n, c1, nc; // nc = n c1, the normal wavenumber over k0
    };
    auto slabp = [&](std::complex<double> es, std::complex<double> mus, double th) -> Sl
    {
        Sl s;
        s.c1 = std::sqrt(1.0 - std::sin(th) * std::sin(th) / (es * mus)), s.n = std::sqrt(es * mus), s.nc = s.n * s.c1;
        std::complex<double> a = std::cos(th), b = std::sqrt(es / mus) * s.c1;
        s.r10 = (b - a) / (b + a), s.t01 = 2.0 * a / (a + b), s.t10 = 2.0 * b / (a + b);
        return s;
    };
    auto phi_of = [&](const Sl &s, double f, double Lx)
    { return std::exp(-J * (2.0 * QPI * f / C0) * s.nc * Lx); };

    // ------------------------------------------------------------------------------------------
    // A. A transmitted chain, both tracers, the base frequency and two more of a band
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: phase reference, transmitted. air | 100 mm | air through the intersector, series resolved, scalar,\n");
    QD_DIAG("# against the analytic slab. Base frequency 1 GHz; 0.5 and 2 GHz are run as frequencies of a band.\n");
    QD_DIAG("# ph_ray: at the point where the refracted ray of that frequency leaves. ph_poly: at the polyline's exit\n");
    QD_DIAG("# point. s, dy: the lateral walk of the ray and of the polyline, mm\n");
    QD_DIAG("tracer,mtl,f_GHz,theta,d_dB,ph_ray,ph_poly,s_mm,dy_mm\n");
    for (int tr : {4, 5})
        for (int m : {5, 6, 1})
            for (double f1 : {1.0e9, 0.5e9, 2.0e9})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                    Chain c = run(mtl, mesh, mi, {tr}, thd, 0.0, f1, f1 == FRQ ? 0.0 : FRQ);
                    REQUIRE((c.ok && c.ev.size() == 2));
                    double Lx = L - c.evb[0].origN(0, 0);
                    Sl s = slabp(diag_eps(m, f1), 1.0, th);
                    std::complex<double> phi = phi_of(s, f1, Lx), t = s.t01 * s.t10 * phi / (1.0 - s.r10 * s.r10 * phi * phi);
                    double s_ray = Lx * tan_of(c.ev[0]) + c.offy[0];
                    std::complex<double> a = c.abs_vv(k1) / t;
                    std::complex<double> q_ray = a * std::exp(J * k1 * std::sin(th) * s_ray), q_poly = a * std::exp(J * k1 * std::sin(th) * c.dy);
                    QD_DIAG("%d,%d,%.1f,%.1f,%+.6f,%+.4f,%+.4f,%.4f,%.4f\n", tr, m, f1 * 1e-9, thd, dB(std::norm(a)), deg(q_ray), deg(q_poly),
                            s_ray * 1e3, c.dy * 1e3);
                    INFO("transmitted, tracer " << tr << ", material " << m << ", f " << f1 << ", theta_ext " << thd);
                    CHECK(std::abs(dB(std::norm(a))) < 1.0e-3); // every power is where it is
                    CHECK(std::abs(deg(q_poly)) < 0.05);
                }

    // ------------------------------------------------------------------------------------------
    // B. A back reflection, the two legs apart
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: phase reference, back reflection. Enter, reflect at the far face (type 3), leave by the front\n");
    QD_DIAG("# face, against t01 r10 t10 phi^2 [S]. eps 1.5: series declined; eps 0: the reflection collapses it\n");
    QD_DIAG("# and the ray leaves resolved. Per leg: the event that closes it against r10 phi [S] (leg 1, closed\n");
    QD_DIAG("# on the reflect pass) and t10 phi (leg 2, closed at the exit), phase; leg1_dB: leg 1 in power\n");
    QD_DIAG("tracer,mtl,f_GHz,eps,theta,rt,d_dB,ph_ray,ph_poly,leg1_ray,leg1_poly,leg2_ray,leg2_poly,leg1_dB\n");
    for (int tr : {4, 5})
        for (int m : {5, 6})
            for (double f1 : {1.0e9, 2.0e9})
                for (double eps : {1.5, 0.0})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0, st = std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = build({0.0, L}, {m, m}, {false, true}, mi);
                        Chain c = run(mtl, mesh, mi, {tr, 3, tr}, thd, eps, f1, f1 == FRQ ? 0.0 : FRQ);
                        std::string rt;
                        for (const auto &E : c.ev)
                            rt += std::to_string((unsigned)E.rt) + "/";
                        if (!c.ok || c.ev.size() != 3 || std::norm(c.vv) == 0.0)
                        {
                            QD_DIAG("%d,%d,%.1f,%.1f,%.1f,%s,dead\n", tr, m, f1 * 1e-9, eps, thd, rt.c_str());
                            INFO("back reflection, tracer " << tr << ", material " << m << ", f " << f1 << ", eps " << eps << ", theta_ext " << thd);
                            CHECK(false); // no ray dies on this path: the wave crosses the front face
                            continue;
                        }
                        double Lx1 = L - c.evb[0].origN(0, 0), Lx2 = c.evb[1].origN(0, 0), tt = tan_of(c.ev[0]);
                        Sl s = slabp(diag_eps(m, f1), 1.0, th);
                        std::complex<double> p1 = phi_of(s, f1, Lx1), p2 = phi_of(s, f1, Lx2);
                        std::complex<double> S = (eps == 0.0) ? 1.0 / (1.0 - s.r10 * s.r10 * p1 * p1) : std::complex<double>(1.0, 0.0);
                        std::complex<double> a = c.abs_vv(k1) / (s.t01 * s.r10 * s.t10 * p1 * p2 * S);
                        double s_ray = (Lx1 + Lx2) * tt + c.offy[0] + c.offy[1];
                        std::complex<double> a1 = cpx(c.ev[1].xpr, 0, 0) * std::exp(-J * k1 * c.len[1]) / (s.r10 * p1 * S);
                        std::complex<double> a2 = cpx(c.ev[2].xpr, 0, 0) * std::exp(-J * k1 * c.len[2]) / (s.t10 * p2);
                        double s1 = Lx1 * tt + c.offy[0], s2 = Lx2 * tt + c.offy[1];
                        QD_DIAG("%d,%d,%.1f,%.1f,%.1f,%s,%+.6f,%+.4f,%+.4f", tr, m, f1 * 1e-9, eps, thd, rt.c_str(), dB(std::norm(a)),
                                deg(a * std::exp(J * k1 * st * s_ray)), deg(a * std::exp(J * k1 * st * c.dy)));
                        QD_DIAG(",%+.4f,%+.4f,%+.4f,%+.4f,%+.6f\n", deg(a1 * std::exp(J * k1 * st * s1)), deg(a1 * std::exp(J * k1 * st * c.lat[1])),
                                deg(a2 * std::exp(J * k1 * st * s2)), deg(a2 * std::exp(J * k1 * st * c.lat[2])), dB(std::norm(a1)));
                        INFO("back reflection, tracer " << tr << ", material " << m << ", f " << f1 << ", eps " << eps << ", theta_ext " << thd);
                        CHECK(std::isfinite(std::norm(a)));
                        // Leg 1, closed on the reflect pass: the exact form, with the series where it collapsed
                        CHECK(std::abs(dB(std::norm(a1))) < 1.0e-3);
                        CHECK(std::abs(deg(a1 * std::exp(J * k1 * st * c.lat[1]))) < 0.05);
                        // The whole path, a resolved ray of the undeviated tracer included
                        CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                        CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05);
                        CHECK(std::abs(deg(a2 * std::exp(J * k1 * st * c.lat[2]))) < 0.05);
                    }

    // ------------------------------------------------------------------------------------------
    // C. A sheet inside a host: which point the composite and its displaced-host term sit on
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: phase reference, a sheet inside a host. air | host 50 | sheet 6 mm eps 5.24 - 0.63j | host 50 | air,\n");
    QD_DIAG("# host series declined, scalar, 1 GHz, against the single pass through host (100 - 6 mm) and the sheet's\n");
    QD_DIAG("# Airy pair. Four points: poly, the polyline's; host, where a ray refracted in the host alone leaves,\n");
    QD_DIAG("# the sheet taken as host; true, where the ray leaves that is also refracted in the sheet; less, the\n");
    QD_DIAG("# host ray over 100 - 6 mm, which is where a transverse term in the displaced-host division would put it\n");
    QD_DIAG("tracer,host,theta,d_dB,ph_poly,ph_host,ph_true,ph_less\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th), thk = 0.006;
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, mi);
                Chain c = run(mtl, mesh, mi, {tr}, thd, 1.5, FRQ, 0.0);
                REQUIRE((c.ok && c.ev.size() == 3));
                double Dw = (0.05 - c.evb[0].origN(0, 0)) + (L - c.evb[1].origN(0, 0)), tth = tan_of(c.ev[0]);
                Sl sh = slabp(diag_eps(h, FRQ), 1.0, th), ss = slabp(diag_eps(SHEET, FRQ), 1.0, th);
                std::complex<double> bh = sh.nc, bs = ss.nc, r_sh = (bs - bh) / (bs + bh);
                std::complex<double> ps = phi_of(ss, FRQ, thk), Ts = (2.0 * bh / (bh + bs)) * (2.0 * bs / (bh + bs)) * ps / (1.0 - r_sh * r_sh * ps * ps);
                std::complex<double> a = c.abs_vv(k1) / (sh.t01 * phi_of(sh, FRQ, Dw - thk) * Ts * sh.t10);
                double tts = st / (std::sqrt(std::abs(diag_eps(SHEET, FRQ))) * std::real(ss.c1)); // qd_refract's map from air
                double off = c.offy[0] + c.offy[1];
                QD_DIAG("%d,%d,%.1f,%+.6f,%+.4f,%+.4f,%+.4f,%+.4f\n", tr, h, thd, dB(std::norm(a)), deg(a * std::exp(J * k1 * st * c.dy)),
                        deg(a * std::exp(J * k1 * st * (Dw * tth + off))), deg(a * std::exp(J * k1 * st * ((Dw - thk) * tth + thk * tts + off))),
                        deg(a * std::exp(J * k1 * st * ((Dw - thk) * tth + off))));
                INFO("sheet inside a host, tracer " << tr << ", host " << h << ", theta_ext " << thd);
                CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05); // the composite and the host legs on one point
            }

    // ------------------------------------------------------------------------------------------
    // D. An acoustic absorber as a traced solid, over a band
    // ------------------------------------------------------------------------------------------
    // Mineral wool at about 12 kPa s/m^2, the two-column-pair fit of the material model, 50 mm; and a
    // rigid backing, eps 2.401e-9. Frequencies are acoustic, scaled by c0 / 343 m/s. The index is
    // about 1.5 - 0.75j at 1 kHz, so the wave is refracted into the layer at every angle.
    {
        const double W = 0.05, fscale = C0 / 343.0, gapw = 1.0e-5;
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {1.16, 2.401e-9}, pal["b"] = {-0.04, 0.0}, pal["c"] = {0.0038, 0.0}, pal["d"] = {1.46, 0.0};
        pal["e"] = {1.66, 1.0}, pal["f"] = {-0.19, 0.0}, pal["g"] = {0.091, 0.0}, pal["h"] = {0.13, 0.0};
        pal["fRef"] = {0.874, 0.874};
        auto wool = [&](double f_hz, std::complex<double> &es, std::complex<double> &mus)
        {
            double fG = f_hz * 1e-9, fr = fG / 0.874;
            es = std::complex<double>(1.16 * std::pow(fr, -0.04), -17.98 * 0.0038 * std::pow(fr, 1.46) / fG);
            mus = std::complex<double>(1.66 * std::pow(fr, -0.19), -17.98 * 0.091 * std::pow(fr, 0.13) / fG);
        };
        const std::vector<double> band = {250.0, 500.0, 1000.0, 2000.0, 4000.0}, ang = {20.0, 40.0, 60.0, 75.0};
        const double f_base = 1000.0 * fscale;
        QD_DIAG("\n# diag: phase reference, an acoustic absorber as a traced solid. Mineral wool 50 mm, scalar, base\n");
        QD_DIAG("# frequency 1 kHz, the others run as frequencies of a band. through: air | wool | air, series\n");
        QD_DIAG("# resolved. backed: air | wool | rigid wall, enter, reflect at the wall, leave, series declined.\n");
        QD_DIAG("# n: the index of the wool at that frequency. ph_ray, ph_poly, s, dy as above. The wall is taken at\n");
        QD_DIAG("# unit reflectance and the Fresnel phase of wool | eps 2.401e-9, decaying root; r12: that Fresnel\n");
        QD_DIAG("# coefficient itself, power in dB and phase, recorded\n");
        QD_DIAG("chain,tracer,f_Hz,n_re,n_im,theta,rt,d_dB,ph_ray,ph_poly,s_mm,dy_mm,r12_dB,r12_deg\n");
        for (int back = 0; back < 2; ++back)
            for (int tr : {4, 5})
                for (double fa : band)
                    for (double thd : ang)
                    {
                        double f1 = fa * fscale, th = thd * QPI / 180.0, k1 = 2.0 * QPI * f1 / C0, st = std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = back ? build({0.0, W, W + gapw, W + gapw + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, mi)
                                              : build({0.0, W}, {1, 1}, {false, true}, mi);
                        Chain c = run(pal, mesh, mi, back ? std::vector<int>{tr, 3, tr} : std::vector<int>{tr}, thd, back ? 1.5 : 0.0, f1,
                                      fa == 1000.0 ? 0.0 : f_base);
                        std::string rt;
                        for (const auto &E : c.ev)
                            rt += std::to_string((unsigned)E.rt) + "/";
                        REQUIRE((c.ok && c.ev.size() == (back ? 3u : 2u)));
                        std::complex<double> es, mus;
                        wool(f1, es, mus);
                        Sl s = slabp(es, mus, th);
                        double Lx1 = W - c.evb[0].origN(0, 0), Lx2 = back ? c.evb[1].origN(0, 0) : 0.0, tt = tan_of(c.ev[0]);
                        std::complex<double> p1 = phi_of(s, f1, Lx1), ref;
                        double r12_dB = 0.0, r12_deg = 0.0;
                        if (back)
                        {
                            // The wall from inside the wool: the decaying root in the rigid body
                            std::complex<double> e2(2.401e-9, 0.0), c2 = -J * std::sqrt(st * st / e2 - 1.0);
                            std::complex<double> b1 = std::sqrt(es / mus) * s.c1, b2 = std::sqrt(e2) * c2, r12 = (b1 - b2) / (b1 + b2);
                            // ray_state_update holds a closed face at unit reflectance and the Fresnel phase
                            r12_dB = dB(std::norm(r12)), r12_deg = deg(r12);
                            ref = s.t01 * (r12 / std::abs(r12)) * s.t10 * p1 * phi_of(s, f1, Lx2);
                        }
                        else
                            ref = s.t01 * s.t10 * p1 / (1.0 - s.r10 * s.r10 * p1 * p1);
                        std::complex<double> a = c.abs_vv(k1) / ref;
                        double s_ray = (Lx1 + Lx2) * tt + c.offy[0] + (back ? c.offy[1] : 0.0);
                        QD_DIAG("%s,%d,%.0f,%.4f,%.4f,%.1f,%s,%+.6f,%+.4f,%+.4f,%.4f,%.4f,%+.4f,%+.3f\n", back ? "backed " : "through", tr, fa, std::real(s.n),
                                std::imag(s.n), thd, rt.c_str(), dB(std::norm(a)), deg(a * std::exp(J * k1 * st * s_ray)),
                                deg(a * std::exp(J * k1 * st * c.dy)), s_ray * 1e3, c.dy * 1e3, r12_dB, r12_deg);
                        INFO("absorber, " << (back ? "backed" : "through") << ", tracer " << tr << ", f " << fa << ", theta_ext " << thd);
                        CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                        CHECK(std::abs(deg(a * std::exp(J * k1 * st * c.dy))) < 0.05);
                    }
    }

    // ------------------------------------------------------------------------------------------
    // E. Two solids across a bare face: where the phase between the two tracers comes from
    // ------------------------------------------------------------------------------------------
    // The absolute transfer puts the free-space reference back over the polyline between hit points.
    // A coefficient carries that reference over the ledger only, which starts at the relaunch point
    // and ends at the hit point. What lies between a hit point and the next relaunch point - the
    // relaunch offset, and across a co-located pair the gap between its two faces - is in the
    // polyline and in no ledger, and is charged as free space along the polyline. At the polyline's
    // point what is left of such a piece of length u and lateral walk v is u - sin(theta) v, summed
    // to U. And a leg is as thick as its ledger is long along the normal, d. Both differ between the
    // two tracers, and the phase that follows from the geometry alone is
    // -k0 (U5 - U4) - k0 sum Re(n c) (d5 - d4).
    QD_DIAG("\n# diag: phase reference, two solids across a bare face. air | A 50 | B 100 | air, series declined, scalar,\n");
    QD_DIAG("# 1 GHz. r54: tracer 5 over tracer 4, each at its polyline's exit point. U: what no ledger holds, um. pU,\n");
    QD_DIAG("# pd: the phase that follows from U and from the thickness the ledgers hold, deg. res: r54_deg - pU - pd\n");
    QD_DIAG("pair,gap_um,theta,r54_dB,r54_deg,U4_um,U5_um,pU_deg,pd_deg,res_deg\n");
    for (auto P : std::vector<std::pair<int, int>>{{1, 5}, {2, 6}})
        for (double g : {1.0e-5, 1.0e-6, 5.0e-5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, 0.05 + g, 0.15 + g}, {P.first, P.first, P.second, P.second}, {false, true, false, true}, mi);
                Chain c4 = run(mtl, mesh, mi, {4}, thd, 1.5, FRQ, 0.0), c5 = run(mtl, mesh, mi, {5}, thd, 1.5, FRQ, 0.0);
                REQUIRE((c4.ok && c5.ok && c4.ev.size() == 3 && c5.ev.size() == 3));
                std::complex<double> r54 = (c5.abs_vv(k1) * std::exp(J * k1 * st * c5.dy)) / (c4.abs_vv(k1) * std::exp(J * k1 * st * c4.dy));
                double U4 = 0.0, U5 = 0.0, pd = 0.0;
                for (size_t k = 1; k < 3; ++k)
                {
                    U4 += c4.len[k] - c4.led[k] - st * (c4.lat[k] - c4.ledy[k]), U5 += c5.len[k] - c5.led[k] - st * (c5.lat[k] - c5.ledy[k]);
                    Sl s = slabp(diag_eps(k == 1 ? P.first : P.second, FRQ), 1.0, th);
                    pd -= k1 * std::real(s.nc) * (c5.ledx[k] - c4.ledx[k]);
                }
                double pU = -k1 * (U5 - U4);
                QD_DIAG("%d|%d,%.0f,%.1f,%+.6f,%+.4f,%.3f,%.3f,%+.4f,%+.4f,%+.4f\n", P.first, P.second, g * 1e6, thd, dB(std::norm(r54)), deg(r54),
                        U4 * 1e6, U5 * 1e6, pU * 180.0 / QPI, pd * 180.0 / QPI, deg(r54) - (pU + pd) * 180.0 / QPI);
                INFO("two solids " << P.first << "|" << P.second << ", gap " << g << ", theta_ext " << thd);
                CHECK(std::abs(dB(std::norm(r54))) < 2.0e-2); // 2e-3 at the 10 um of the refracting-tracer case
                CHECK(std::abs(deg(r54) - (pU + pd) * 180.0 / QPI) < 0.005); // nothing but the geometry
                CHECK(std::abs(deg(r54)) < 0.05);
            }

    // ------------------------------------------------------------------------------------------
    // F. The legs the exact form does not take: a mass-law solid, refracted into and entered as a leak
    // ------------------------------------------------------------------------------------------
    {
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.1609e-8, 1.5};
        pal["fRef"] = {0.001, 0.001};
        pal["tf"] = {0.2, 0.0};
        pal["m"] = {20.0, 20.0};
        QD_DIAG("\n# diag: phase reference, a mass-law solid, scalar, 1 GHz. The traversal alone, as the ratio of the\n");
        QD_DIAG("# absolute transfer through 100 mm over that through 50 mm. dense: eps 1.50, m 20, refracted into,\n");
        QD_DIAG("# against exp(-j k0 n cos(theta_t) d). rigid: wood eps 2.1609e-8, tf 0.2, m 20, entered as a leak,\n");
        QD_DIAG("# against exp(-j k0 n cos(theta) d), the walk-off form. bits: tracer 5 against tracer 4, every event\n");
        QD_DIAG("body,tracer,theta,cur,ph_ray,ph_poly,bits\n");
        for (int m : {2, 1})
            for (int tr : {4, 5})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, k1 = 2.0 * QPI * FRQ / C0, st = std::sin(th);
                    arma::uvec mi;
                    arma::mat mesh1 = build({0.0, 0.5 * L}, {m, m}, {false, true}, mi), mesh2 = build({0.0, L}, {m, m}, {false, true}, mi);
                    Chain c1 = run(pal, mesh1, mi, {tr}, thd, 1.5, FRQ, 0.0), c2 = run(pal, mesh2, mi, {tr}, thd, 1.5, FRQ, 0.0);
                    Chain d1 = run(pal, mesh1, mi, {4}, thd, 1.5, FRQ, 0.0);
                    REQUIRE((c1.ok && c2.ok && c1.ev.size() == 2 && c2.ev.size() == 2 && d1.ev.size() == 2));
                    double n = std::sqrt(m == 2 ? 1.5 : 2.1609e-8), tt = tan_of(c1.ev[0]);
                    double ct = (m == 2) ? std::sqrt(1.0 - st * st / 1.5) : std::cos(th); // refracted, or held undeviated
                    std::complex<double> a = (c2.abs_vv(k1) / c1.abs_vv(k1)) / std::exp(-J * k1 * n * ct * 0.5 * L);
                    bool same = true;
                    for (size_t k = 0; k < 2; ++k)
                        same = same && cpx(c1.ev[k].xpr, 0, 0) == cpx(d1.ev[k].xpr, 0, 0) && c1.ev[k].rt == d1.ev[k].rt &&
                               c1.ev[k].cur(0) == d1.ev[k].cur(0);
                    QD_DIAG("%s,%d,%.1f,%04x,%+.4f,%+.4f,%s\n", m == 2 ? "dense" : "rigid", tr, thd, (unsigned)(unsigned short)c1.ev[0].cur(0),
                            deg(a * std::exp(J * k1 * st * 0.5 * L * tt)), deg(a * std::exp(J * k1 * st * (c2.dy - c1.dy))), same ? "same" : "differ");
                    INFO("mass-law solid " << m << ", tracer " << tr << ", theta_ext " << thd);
                    CHECK(std::abs(std::abs(c2.abs_vv(k1) / c1.abs_vv(k1)) - 1.0) < 1.0); // a traversal attenuates or holds
                    CHECK(std::abs(deg(a * std::exp(J * k1 * st * (c2.dy - c1.dy)))) < 0.05);
                    if (m == 1)
                        CHECK(same); // a leak is the undeviated tracer's with either type, bit for bit
                }
    }

    // ------------------------------------------------------------------------------------------
    // G. Outside the item, recorded: the phase a rigid body reflects with, seen from air
    // ------------------------------------------------------------------------------------------
    // The backed rows above take the wall at the Fresnel phase of the permittivity it is mapped to.
    // Seen from air that phase is measured here, for the one-parameter mapping (eps = (Z_air / Z)^2,
    // mu = 1) and for a body with the same impedance and an index of 0.1 (eps = z n, mu = n / z).
    {
        const double zr = std::sqrt(2.401e-9), nr = 0.1;
        std::unordered_map<std::string, std::vector<double>> pal;
        pal["a"] = {2.401e-9, zr * nr};
        pal["e"] = {1.0, nr / zr};
        QD_DIAG("\n# diag: outside the item, recorded. Reflection off a rigid half-space from air, type 3, 1 GHz,\n");
        QD_DIAG("# through the intersector. one: eps 2.401e-9, mu 1. two: the same impedance with an index of 0.1\n");
        QD_DIAG("mapping,theta,refl_dB,refl_deg\n");
        for (int m : {1, 2})
            for (double thd : {0.0, 10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.5}, {m, m}, {false, true}, mi), orig, dest;
                diag_ray_at(thd, orig, dest);
                Ev E = traced(pal, mesh, mi, 3, FRQ, orig, dest, nullptr, nullptr, 1.5, false);
                std::complex<double> r = cpx(E.xpr, 0, 0);
                QD_DIAG("%s,%.1f,%+.6f,%+.4f\n", m == 1 ? "one" : "two", thd, dB(std::norm(r)), deg(r));
                INFO("rigid half-space, mapping " << m << ", theta " << thd);
                CHECK(std::abs(std::norm(r) - 1.0) < 1.0e-3);
            }
    }
}

TEST_CASE("Exits and verdicts - every exit at the cosine the wave has there")
{
    // Every exit and every verdict is formed at the cosine the wave has at that face:
    //   A. a resolved ray of the undeviated tracer by two more routes: reflected off a coating on
    //      the far side (always resolved, whatever eps is), and returning across a bare face into
    //      a first solid;
    //   B. a bare exit behind a wedge against the glued one, through the intersector;
    //   C. the verdict and the direction at a bare and at a glued exit, invariant p around one;
    //   D. the reflect port of a coated lossy solid past its critical angle.
    const double f_hz = FRQ, k0 = 2.0 * QPI * f_hz / C0, L = 0.10, thk = 0.006, gap = 1.0e-5;
    auto mtl = diag_palette<double>();
    const int SHEET = 3, LOSSY = 5, DENSE = 2;
    const std::complex<double> J(0.0, 1.0);
    typedef std::complex<double> cd;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](cd z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr, gain;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, const arma::mat &orig, const arma::mat &dest,
                      const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &E.gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &mtl, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &E.gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        cd vv = 1.0;
        double Lin = 0.0, dy = 0.0;
        std::vector<Ev> ev;
        std::vector<double> offy;
        bool ok = true;
        cd abs_vv(double k) const { return vv * std::exp(cd(0.0, -k * Lin)); }
    };
    auto run = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, std::vector<int> seq, double thd, double eps) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : (k < (int)seq.size() ? seq[k] : -1);
            if (itype < 0)
                break;
            Ev E = traced(mesh, mtl_ind, itype, orig, dest, have ? &P : nullptr, eps, itype != 0 && itype != 3);
            if (E.nh == 0)
                break;
            C.ev.push_back(E);
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            C.offy.push_back(std::abs(E.origN(0, 1) - fy));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = E.origN, dest = E.origN + 2.0 * d;
            P = E, have = true;
        }
        return C;
    };
    auto rts = [](const Chain &c)
    {
        std::string s;
        for (const auto &E : c.ev)
            s += std::to_string((unsigned)E.rt) + "/";
        return s;
    };

    // Everything from the transverse invariant p. Admittances z c (TE) and z / c (TM); the far
    // side is air, with the decaying root above p = 1. R is the reflect port of near | sheet | air
    // or, without a sheet, of near | air; pT the per-channel normal-flux transmittance (TE only).
    struct Ref
    {
        cd c_near, R;
        double pT = 0.0, leg = 1.0; // leg: power decay of the host over a normal thickness
    };
    auto ref_of = [&](cd e_near, bool sheet, double p, bool tm, double d_host) -> Ref
    {
        const cd e_s = diag_eps(SHEET, f_hz);
        Ref r;
        r.c_near = std::sqrt(1.0 - p / e_near);
        cd cs = std::sqrt(1.0 - p / e_s), cf = (p < 1.0) ? cd(std::sqrt(1.0 - p), 0.0) : cd(0.0, -std::sqrt(p - 1.0));
        cd zn = std::sqrt(e_near), zs = std::sqrt(e_s);
        cd an = tm ? zn / r.c_near : zn * r.c_near, as = tm ? zs / cs : zs * cs, af = tm ? 1.0 / cf : cf;
        cd t;
        if (sheet)
        {
            cd r1 = (an - as) / (an + as), r2 = (as - af) / (as + af), ph = std::exp(-J * k0 * zs * cs * thk);
            r.R = (r1 + r2 * ph * ph) / (1.0 + r1 * r2 * ph * ph);
            t = (1.0 + r1) * (1.0 + r2) * ph / (1.0 + r1 * r2 * ph * ph);
        }
        else
            r.R = (an - af) / (an + af), t = 1.0 + r.R;
        r.pT = (p < 1.0) ? std::real(af) / std::real(an) * std::norm(t) : 0.0;
        r.leg = std::exp(2.0 * k0 * std::imag(zn * r.c_near) * d_host);
        return r;
    };

    // ------------------------------------------------------------------------------------------
    // A1. A resolved ray that was reflected off a coating on the far side
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: exits and verdicts, a ray reflected off a far-side coating. air | host 100 mm | coating 6 mm eps 5.24 - 0.63j |\n");
    QD_DIAG("# air through the intersector: enter, reflect off the coating (type 3), leave by the front face. Host\n");
    QD_DIAG("# series declined (eps 1.5); the reflection off a sheet flags the ray resolved all the same. Scalar,\n");
    QD_DIAG("# against t01 R t10 phi^2 with R the reflect port of host | coating | air. ph_poly at the polyline's point\n");
    QD_DIAG("tracer,host,theta,rt,d_dB,ph_poly\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0, 85.0})
            {
                double th = thd * QPI / 180.0, st = std::sin(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                Chain c = run(mesh, mi, {tr, 3, tr}, thd, 1.5);
                INFO("far-side coating, tracer " << tr << ", host " << h << ", theta_ext " << thd);
                if (!c.ok || c.ev.size() != 3 || std::norm(c.vv) == 0.0)
                {
                    QD_DIAG("%d,%d,%.1f,%s,dead\n", tr, h, thd, rts(c).c_str());
                    CHECK(false); // the wave crosses the front face
                    continue;
                }
                cd eh = diag_eps(h, f_hz), zh = std::sqrt(eh);
                Ref r = ref_of(eh, true, st * st, false, 0.0);
                cd a0 = std::cos(th), b0 = zh * r.c_near, t01 = 2.0 * a0 / (a0 + b0), t10 = 2.0 * b0 / (a0 + b0);
                double Lx = (L - c.ev[0].origN(0, 0)) + c.ev[1].origN(0, 0);
                cd a = c.abs_vv(k0) / (t01 * r.R * t10 * std::exp(-J * k0 * zh * r.c_near * Lx));
                QD_DIAG("%d,%d,%.1f,%s,%+.6f,%+.4f\n", tr, h, thd, rts(c).c_str(), dB(std::norm(a)), deg(a * std::exp(J * k0 * st * c.dy)));
                CHECK(std::abs(dB(std::norm(a))) < 1.0e-3);
                CHECK(std::abs(deg(a * std::exp(J * k0 * st * c.dy))) < 0.05);
            }

    // ------------------------------------------------------------------------------------------
    // A2. A resolved ray returning across a bare face into the first solid
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: exits and verdicts, a resolved ray across a bare face into a solid. air | A 50 mm | B 100 mm | air through the\n");
    QD_DIAG("# intersector, every series resolved (eps 0): enter A, cross into B, reflect at B | air, cross back into\n");
    QD_DIAG("# A, leave by the front face. No reference: the undeviated tracer over the refracting one, each at its\n");
    QD_DIAG("# polyline's exit point. rt4, rt5: the resolved types of the five events\n");
    QD_DIAG("pair,theta,rt4,rt5,r45_dB,r45_deg\n");
    for (auto P : std::vector<std::pair<int, int>>{{2, 5}, {1, 5}, {2, 6}, {5, 2}})
        for (double thd : {10.0, 20.0, 40.0, 60.0, 75.0})
        {
            double st = std::sin(thd * QPI / 180.0);
            arma::uvec mi;
            arma::mat mesh = build({0.0, 0.05, 0.05 + gap, 0.15 + gap}, {P.first, P.first, P.second, P.second}, {false, true, false, true}, mi);
            Chain c4 = run(mesh, mi, {4, 4, 3, 4, 4}, thd, 0.0), c5 = run(mesh, mi, {5, 5, 3, 5, 5}, thd, 0.0);
            INFO("two solids, resolved, " << P.first << "|" << P.second << ", theta_ext " << thd);
            bool dead4 = !c4.ok || c4.ev.size() != 5 || std::norm(c4.vv) == 0.0, dead5 = !c5.ok || c5.ev.size() != 5 || std::norm(c5.vv) == 0.0;
            if (dead4 || dead5)
            {
                QD_DIAG("%d|%d,%.1f,%s,%s,%s\n", P.first, P.second, thd, rts(c4).c_str(), rts(c5).c_str(), dead5 ? "dead5" : "dead4");
                CHECK(false); // no ray dies on this path with either tracer
                continue;
            }
            cd r45 = (c4.abs_vv(k0) * std::exp(J * k0 * st * c4.dy)) / (c5.abs_vv(k0) * std::exp(J * k0 * st * c5.dy));
            QD_DIAG("%d|%d,%.1f,%s,%s,%+.6f,%+.4f\n", P.first, P.second, thd, rts(c4).c_str(), rts(c5).c_str(), dB(std::norm(r45)), deg(r45));
            CHECK(std::abs(dB(std::norm(r45))) < 2.0e-3); // two polylines, each short by its own offsets
            CHECK(std::abs(deg(r45)) < 0.05);
        }

    // ------------------------------------------------------------------------------------------
    // B. A wedge: the bare exit against the glued one
    // ------------------------------------------------------------------------------------------
    // Entry face at x = 0, exit face through (L, 0.37) with its outward normal turned by 10 deg
    // towards +y, so the entry event sets the non-parallel flag itself and the recovery declines
    // on the leg. Angles are against the exit normal. in: the direction the wave has; seg_in: the
    // segment; out: the direction written; seg_out: the relaunched segment; map: the real-angle
    // map applied to the direction the wave has; rmi: the same map applied to the segment.
    QD_DIAG("\n# diag: exits and verdicts, a wedge. air | host | air, exit face tilted by 10 deg, through the intersector, scalar,\n");
    QD_DIAG("# series declined. bare: a bare exit; glued: the same exit behind a 6 mm coating. Angles in deg against\n");
    QD_DIAG("# the exit normal: in, the direction the wave has; seg_in, the segment; out, the direction written;\n");
    QD_DIAG("# seg_out, the relaunched segment; map, the real-angle map of in; rmi, the real-angle map of seg_in.\n");
    QD_DIAG("# prev: the previous-medium word the entry wrote. gain: the exit event, dB\n");
    QD_DIAG("tracer,host,theta,exit,prev,rt,in,seg_in,out,seg_out,map,rmi,gain_dB\n");
    {
        const double a = 10.0 * QPI / 180.0, ca = std::cos(a), sa = std::sin(a), y0 = 0.37;
        auto tilt = [&](arma::mat m, double shift)
        {
            for (arma::uword r = 0; r < m.n_rows; ++r)
                for (arma::uword v = 0; v < 3; ++v)
                {
                    double x = m(r, 3 * v) - L, y = m(r, 3 * v + 1) - y0;
                    m(r, 3 * v) = L + x * ca - y * sa + shift * ca, m(r, 3 * v + 1) = y0 + x * sa + y * ca + shift * sa;
                }
            return m;
        };
        auto ang = [&](double dx, double dy) // against the exit normal
        { return std::atan2(std::abs(dy * ca - dx * sa), std::abs(dx * ca + dy * sa)) * 180.0 / QPI; };
        auto map_of = [&](cd e, double in_deg)
        {
            double s = std::sin(in_deg * QPI / 180.0);
            return std::atan2(std::sqrt(std::abs(e)) * s, std::real(std::sqrt(1.0 - e * s * s))) * 180.0 / QPI;
        };
        double bare_x = 0.0, bare_y = 0.0;
        for (int tr : {4, 5})
            for (int h : {5, 6})
                for (double thd : {10.0, 20.0, 40.0, 60.0})
                    for (int glued = 0; glued < 2; ++glued)
                    {
                        double t = std::tan(thd * QPI / 180.0);
                        arma::mat mesh = arma::join_cols(face(0.0, false), tilt(face(L, true), 0.0));
                        arma::uvec mi(4, arma::fill::value((arma::uword)h));
                        if (glued)
                        {
                            mesh = arma::join_cols(mesh, tilt(face(L, true), gap));
                            mi = arma::uvec{(arma::uword)h, (arma::uword)h, (arma::uword)h, (arma::uword)h, (arma::uword)SHEET, (arma::uword)SHEET};
                        }
                        arma::mat orig{{-1.0, y0 - t, 0.0}}, dest{{1.0, y0 + t, 0.0}};
                        Ev A = traced(mesh, mi, tr, orig, dest, nullptr, 1.5, true);
                        arma::mat d = A.destN - A.origN;
                        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                        arma::mat dest2 = A.origN + 2.0 * d;
                        Ev B = traced(mesh, mi, tr, A.origN, dest2, &A, 1.5, true);
                        arma::mat d2 = B.destN - B.origN;
                        cd eh = diag_eps(h, f_hz);
                        double in_deg = ang(A.path_dirN(0, 0), A.path_dirN(0, 1)), seg_deg = ang(d(0, 0), d(0, 1));
                        QD_DIAG("%d,%d,%.1f,%s,%04x,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%+.4f\n", tr, h, thd, glued ? "glued" : "bare ",
                                (unsigned)(unsigned short)A.prev(0), (unsigned)B.rt, in_deg, seg_deg, ang(B.path_dirN(0, 0), B.path_dirN(0, 1)),
                                ang(d2(0, 0), d2(0, 1)), map_of(eh, in_deg), map_of(eh, seg_deg), dB(std::norm(cpx(B.xpr, 0, 0))));
                        INFO("wedge, tracer " << tr << ", host " << h << ", theta_ext " << thd << ", glued " << glued);
                        CHECK(B.nh > 0);
                        CHECK(((unsigned short)A.prev(0) & 0x8000u) != 0u); // the entry saw the wedge
                        if (!glued)
                            bare_x = B.path_dirN(0, 0), bare_y = B.path_dirN(0, 1);
                        else // a bare and a glued exit write one direction
                            CHECK(std::max(std::abs(B.path_dirN(0, 0) - bare_x), std::abs(B.path_dirN(0, 1) - bare_y)) < 1.0e-9);
                    }
    }

    // ------------------------------------------------------------------------------------------
    // C and D. The lossy host entered from a lossless dense solid, invariant p around and above one
    // ------------------------------------------------------------------------------------------
    // The code ray_mesh_interact writes for the exit met along the segment, which with the
    // undeviated tracer runs at the angle the wave had in the dense solid
    auto exit_code = [&](int itype, int host, bool glued, double th) -> uint8_t
    {
        arma::mat mesh = diag_quad_x(200.0, 0.0);
        arma::uvec mtl_ind(2, arma::fill::value((arma::uword)host));
        if (glued)
        {
            mesh = arma::join_cols(mesh, diag_quad_x(200.0, -1.0e-5));
            mtl_ind = arma::uvec{(arma::uword)host, (arma::uword)host, (arma::uword)SHEET, (arma::uword)SHEET};
        }
        double t = std::tan(th);
        arma::mat orig{{1.0, 0.37, 0.0}}, dest{{-1.0, 0.37 + 2.0 * t, 0.0}}, fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);
        std::vector<uint8_t> ot;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &mtl, &fbs_ind, &sbs_ind,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, (arma::vec *)nullptr, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, (arma::mat *)nullptr, &ot, (arma::mat *)nullptr, true,
                                        (arma::u32_vec *)nullptr);
        REQUIRE(ot.size() == 1);
        return ot[0];
    };
    auto event = [&](int itype, uint8_t ot, unsigned nH, int m1, int m2, int prev, int cur, double th, double dx, double dy)
    {
        auto C = make1<double>(mtl, itype, ot, nH, m1, m2, prev, cur, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
        C.has_path_dir = true;
        C.path_dir_prev.zeros(1, 3);
        C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
        C.has_path_dirN = true;
        C.path_dirN.zeros(1, 3);
        C.path_dirN(0, 0) = std::cos(th), C.path_dirN(0, 1) = std::sin(th); // the undeviated tracer's
        C.run();
        return C;
    };
    const cd e_d = diag_eps(DENSE, f_hz), e_l = diag_eps(LOSSY, f_hz);
    auto dir_of = [&](double p, double &th1, double &ct, double &st, double &N2)
    {
        double s1 = std::sqrt(p / std::real(e_d));
        th1 = std::asin(s1);
        cd c2 = std::sqrt(1.0 - p / e_l);
        double eta = std::sqrt(std::abs(e_d / e_l)), r = std::real(c2);
        N2 = (eta * s1) * (eta * s1) + r * r; // the squared length the projection lost
        ct = r / std::sqrt(N2), st = std::sqrt(1.0 - ct * ct);
    };

    QD_DIAG("\n# diag: exits and verdicts, the verdict at an exit, invariant p around one. Host eps 1.50 - 0.40j 100 mm entered from a\n");
    QD_DIAG("# lossless eps 5.24 solid, named in the previous-medium word, undeviated tracer, scalar, series declined.\n");
    QD_DIAG("# The wave is closed towards air from p = 1. N2: the squared length the projection lost; p/N2: what the\n");
    QD_DIAG("# real-angle test reads. Per exit: hit code, resolved type, tr = its total-reflection bit, the event\n");
    QD_DIAG("# against the reference in dB (ref 0 above p = 1: gain printed as is), and the direction written, deg\n");
    QD_DIAG("p,N2,p/N2,b_code,b_rt,b_tr,b_gain_dB,b_ref_dB,b_out,g_code,g_rt,g_tr,g_gain_dB,g_ref_dB,g_out\n");
    for (double p : {0.50, 0.90, 0.95, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10, 1.20, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        uint8_t cb = exit_code(4, LOSSY, false, th1), cg = exit_code(4, LOSSY, true, th1);
        auto Xb = event(4, cb, 1u, LOSSY, 0, DENSE, LOSSY, th1, ct, st);
        auto Xg = event(4, cg, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        Ref rb = ref_of(e_l, false, p, false, L), rg = ref_of(e_l, true, p, false, L);
        auto out = [](const Rsu<double> &X)
        { return std::atan2(std::abs((double)X.path_dirN(0, 1)), std::abs((double)X.path_dirN(0, 0))) * 180.0 / QPI; };
        double gb = std::norm(cpx(Xb.xprmat, 0, 0)), gg = std::norm(cpx(Xg.xprmat, 0, 0));
        QD_DIAG("%.2f,%.6f,%.6f,%u,%u,%d,%+.6f,%+.6f,%.4f,%u,%u,%d,%+.6f,%+.6f,%.4f\n", p, N2, p / N2,
                (unsigned)cb, (unsigned)Xb.resolved_type[0], (int)rt_bit(Xb.resolved_type[0], 5), dB(gb), dB(rb.pT * rb.leg), out(Xb),
                (unsigned)cg, (unsigned)Xg.resolved_type[0], (int)rt_bit(Xg.resolved_type[0], 5), dB(gg), dB(rg.pT * rg.leg), out(Xg));
        INFO("verdict, p " << p);
        // Closed exactly where the invariant exceeds the far medium's, bare and glued alike
        CHECK(rt_bit(Xb.resolved_type[0], 5) == (p > 1.0));
        CHECK(rt_bit(Xg.resolved_type[0], 5) == (p > 1.0));
        if (p > 1.0) // a closed pair hands on the direction the wave has, and passes nothing without tf
        {
            CHECK(std::max(std::abs((double)Xg.path_dirN(0, 0) - ct), std::abs((double)Xg.path_dirN(0, 1) - st)) < 1.0e-12);
            CHECK(gb == 0.0);
            CHECK(gg == 0.0);
        }
        else
        {
            CHECK(std::abs(dB(gb) - dB(rb.pT * rb.leg)) < 1.0e-6);
            CHECK(std::abs(dB(gg) - dB(rg.pT * rg.leg)) < 1.0e-6);
        }
    }

    QD_DIAG("\n# diag: exits and verdicts, the reflect port of a coated lossy solid. The same host and entry, host | coating 6 mm\n");
    QD_DIAG("# eps 5.24 - 0.63j | air, reflect pass, series declined, against the composite at the exact cosine times\n");
    QD_DIAG("# the host leg. sc: scalar (type 3). vv, hh: the two EM channels (type 0) against the TE and the TM\n");
    QD_DIAG("# reference; amplitudes, difference of magnitudes. bare: the same without the coating, scalar\n");
    QD_DIAG("p,code,rt,|R_sc|,ref_te,d_sc,|vv|,d_vv_te,|hh|,ref_tm,d_hh_tm,|R_bare|,ref_bare,d_bare\n");
    for (double p : {0.50, 0.90, 1.02, 1.04, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        uint8_t c3 = exit_code(3, LOSSY, true, th1), c0 = exit_code(0, LOSSY, true, th1);
        auto S = event(3, c3, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        auto E = event(0, c0, 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        auto B = event(3, exit_code(3, LOSSY, false, th1), 1u, LOSSY, 0, DENSE, LOSSY, th1, ct, st);
        Ref rte = ref_of(e_l, true, p, false, L), rtm = ref_of(e_l, true, p, true, L), rb = ref_of(e_l, false, p, false, L);
        double a = std::sqrt(rte.leg);
        double sc = std::abs(cpx(S.xprmat, 0, 0)), vv = std::abs(cpx(E.xprmat, 0, 0)), hh = std::abs(cpx(E.xprmat, 0, 3)), bb = std::abs(cpx(B.xprmat, 0, 0));
        QD_DIAG("%.2f,%u,%u,%.9f,%.9f,%+.3e,%.9f,%+.3e,%.9f,%.9f,%+.3e,%.9f,%.9f,%+.3e\n", p, (unsigned)c3, (unsigned)S.resolved_type[0],
                sc, a * std::abs(rte.R), sc - a * std::abs(rte.R), vv, vv - a * std::abs(rte.R), hh, a * std::abs(rtm.R),
                hh - a * std::abs(rtm.R), bb, a * std::abs(rb.R), bb - a * std::abs(rb.R));
        INFO("reflect port, p " << p);
        // Scalar and TE at the exact cosine. Above p = 1.35 the three Newton steps of the recovery
        // leave 1e-9 and more, which the table below attributes; those rows are recorded. TM past
        // the critical angle is inside the clamp of section 5
        if (p < 1.36)
        {
            CHECK(std::abs(sc - a * std::abs(rte.R)) < 1.0e-9);
            CHECK(std::abs(vv - a * std::abs(rte.R)) < 1.0e-9);
        }
        CHECK(std::abs(hh - a * std::abs(rtm.R)) < ((p > 1.0) ? 5.0e-2 : 1.0e-9));
    }

    // What is left on the scalar column above p = 1.2. The recovery inverts qd_refract's map with
    // three Newton steps from the real-index start. Written a second time here, with three steps
    // and with twelve: u is the recovered sin^2 in the dense solid, and the event is compared with
    // the reference at the invariant each of them gives.
    QD_DIAG("\n# diag: exits and verdicts, the recovery above p = 1. The map inverted a second time in the fixture, three Newton\n");
    QD_DIAG("# steps and twelve. du3, du12: the recovered invariant over the true one, minus one. d_ref: the scalar\n");
    QD_DIAG("# reflect port of the coated host against the reference at p; d_3: against the reference at the\n");
    QD_DIAG("# invariant three steps give\n");
    QD_DIAG("p,du3,du12,d_ref,d_3\n");
    for (double p : {0.50, 0.90, 0.99, 1.10, 1.20, 1.30, 1.40, 1.45})
    {
        double th1, ct, st, N2;
        dir_of(p, th1, ct, st, N2);
        cd eps_r = e_d / e_l;
        double eta2 = std::abs(eps_r), t = st / ct;
        auto solve = [&](int n)
        {
            double u = t * t / (eta2 * (1.0 + t * t));
            u = (u > 1.0) ? 1.0 : u;
            for (int k = 0; k < n; ++k)
            {
                cd w = std::sqrt(1.0 - eps_r * u);
                double fu = eta2 * u - t * t * std::real(w) * std::real(w), dfu = eta2 + t * t * std::real(w) * std::real(eps_r / w);
                u -= fu / dfu;
                u = (u < 0.0) ? 0.0 : ((u > 1.0) ? 1.0 : u);
            }
            return u;
        };
        double u3 = solve(3), u12 = solve(12), u0 = p / std::real(e_d);
        auto S = event(3, exit_code(3, LOSSY, true, th1), 2u, SHEET, LOSSY, DENSE, LOSSY, th1, ct, st);
        Ref r0 = ref_of(e_l, true, p, false, L), r3 = ref_of(e_l, true, u3 * std::real(e_d), false, L);
        double sc = std::abs(cpx(S.xprmat, 0, 0));
        QD_DIAG("%.2f,%+.3e,%+.3e,%+.3e,%+.3e\n", p, u3 / u0 - 1.0, u12 / u0 - 1.0, sc - std::sqrt(r0.leg) * std::abs(r0.R),
                sc - std::sqrt(r3.leg) * std::abs(r3.R));
        INFO("recovery above one, p " << p);
        CHECK(std::abs(sc - std::sqrt(r3.leg) * std::abs(r3.R)) < 1.0e-12); // nothing but the three steps
        CHECK(std::abs(u12 / u0 - 1.0) < 1.0e-12);
    }

    // The window with a transmission factor on the pair: what the transmit port of the glued exit
    // is where the verdict reads open on a closed wave, against where it reads closed.
    QD_DIAG("\n# diag: exits and verdicts, the verdict window with tf. The glued exit of the table above, transmit port, with tf 0.5\n");
    QD_DIAG("# on the coating, on the host, and on neither. tr: the total-reflection bit of the resolved type\n");
    QD_DIAG("p,p/N2,tr,none_dB,tf_sheet_dB,tf_host_dB\n");
    {
        auto pal_s = mtl, pal_h = mtl;
        pal_s["tf"] = std::vector<double>(9, 0.0), pal_h["tf"] = std::vector<double>(9, 0.0);
        pal_s["tf"][SHEET - 1] = 0.5, pal_h["tf"][LOSSY - 1] = 0.5;
        auto ev_pal = [&](const std::unordered_map<std::string, std::vector<double>> &pal, uint8_t ot, double th, double dx, double dy)
        {
            auto C = make1<double>(pal, 4, ot, 2u, SHEET, LOSSY, DENSE, LOSSY, 0, L, 0.5, 0.5, QPI / 2.0 - th, 1.5);
            C.has_path_dir = true;
            C.path_dir_prev.zeros(1, 3);
            C.path_dir_prev(0, 0) = dx, C.path_dir_prev(0, 1) = dy;
            C.run();
            return C;
        };
        double leak_prev = 0.0;
        for (double p : {0.95, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.10, 1.20})
        {
            double th1, ct, st, N2;
            dir_of(p, th1, ct, st, N2);
            uint8_t cg = exit_code(4, LOSSY, true, th1);
            auto X0 = ev_pal(mtl, cg, th1, ct, st), Xs = ev_pal(pal_s, cg, th1, ct, st), Xh = ev_pal(pal_h, cg, th1, ct, st);
            QD_DIAG("%.2f,%.6f,%d,%+.4f,%+.4f,%+.4f\n", p, p / N2, (int)rt_bit(Xs.resolved_type[0], 5), dB(std::norm(cpx(X0.xprmat, 0, 0))),
                    dB(std::norm(cpx(Xs.xprmat, 0, 0))), dB(std::norm(cpx(Xh.xprmat, 0, 0))));
            INFO("verdict window with tf, p " << p);
            double leak = dB(std::norm(cpx(Xs.xprmat, 0, 0)));
            CHECK(rt_bit(Xs.resolved_type[0], 5) == (p > 1.0));
            if (p > 1.0)
            {
                CHECK(leak > -10.0); // the leak of the coating crosses from p = 1 on
                if (p > 1.015)
                    CHECK(std::abs(leak - leak_prev) < 0.5); // and has no hole
                CHECK(std::norm(cpx(X0.xprmat, 0, 0)) == 0.0);
                CHECK(std::norm(cpx(Xh.xprmat, 0, 0)) == 0.0);
            }
            leak_prev = leak;
        }
    }
}

TEST_CASE("Leak direction - a wave held in a mass-law body keeps its direction and a closed face hands on the wave's")
{
    // A mass-law body is a lumped plate with no wave inside. A wave that entered it as a tf leak
    // is held undeviated, and a face that is charged the same from both sides at one angle is a
    // face without a bend: the wave keeps its direction at every face, towards either index,
    // in the segment and in path_dirN, with either tracer. A wall then transmits parallel to the
    // direction the wave arrived in, a layered wall has one angle throughout, and the leg behind
    // a shared face is not a bent one.
    //
    // A. Rigid walls through the intersector, both tracers. var now is the code. hold is the same
    //    run with the fixture doing the holding: after an event that leaves a mass-law medium it
    //    hands on the direction the wave came in with, relaunches along it where the tracer bends
    //    (type 5), and clears the bent flag of the leg that begins. hold_air does so only where
    //    the event leaves into air, and is recorded: it is the option that was not taken.
    //    Asserted: every direction and every segment on the incidence angle, to 1e-9; no bent
    //    flag; every event of now on hold; the refracting tracer on the undeviated one, in
    //    every word and type code, and in every gain exactly on one wall. Scalar, so the
    //    coefficient of an event does not read the direction it writes.
    // B. A face under total reflection at the corrected angle hands on the direction the wave
    //    has: a bare exit and a transition into another solid as a sheet's closed outer pair
    //    does. The relaunched segment of the refracting tracer is recorded.
    const double f_hz = FRQ, gap = 1.0e-5, L = 0.10;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    typedef std::unordered_map<std::string, std::vector<double>> Pal;

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, const arma::mat &orig,
                      const arma::mat &dest, const Ev *in, double eps) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f_hz, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f_hz, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, &E.origN, &E.destN);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    auto unit = [](arma::mat d) -> arma::mat
    {
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        return d;
    };
    auto ang = [](const arma::mat &d) -> double // against the x axis, which every face normal is on
    { return std::atan2(std::abs(d(0, 1)), std::abs(d(0, 0))) * 180.0 / QPI; };

    struct Row
    {
        std::vector<double> seg, dir, g, out;
        std::vector<unsigned> code, rt, cur;
        double tot = 0.0, dy = 0.0;
        bool ok = true;
    };
    auto chain = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int itype, arma::mat orig, arma::mat dest,
                     const Ev *seed, int hold) -> Row
    {
        Row R;
        const std::vector<double> &mcol = pal.at("m");
        Ev P;
        bool have = seed != nullptr;
        if (seed)
            P = *seed;
        double y0 = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            arma::mat sd = unit(dest - orig), wd = have ? unit(P.path_dirN) : sd;
            Ev E = traced(pal, mesh, mi, itype, orig, dest, have ? &P : nullptr, 1.5);
            if (E.nh == 0)
                break;
            int mb = have ? mat_of(P.cur(0)) : 0, ma = mat_of(E.cur(0));
            double g = dB(std::norm(cpx(E.xpr, 0, 0)));
            R.seg.push_back(ang(sd)), R.dir.push_back(ang(wd)), R.g.push_back(g), R.code.push_back(E.ot), R.rt.push_back(E.rt);
            if (E.rt == 0)
            {
                R.ok = false;
                break;
            }
            R.tot += g;
            if (k == 0)
                y0 = E.fbs(0, 1);
            R.dy = E.fbs(0, 1) - y0;
            arma::mat d = unit(E.destN - E.origN);
            bool held = mb != 0 && mcol[(size_t)mb - 1] > 0.0 && (hold == 1 || (hold == 2 && ma == 0));
            if (held)
            {
                E.path_dirN = wd;
                if (itype == 5)
                    d = wd;
                if (ma != 0)
                    E.cur(0) = (short)(E.cur(0) & ~(short)0x4000);
            }
            R.out.push_back(ang(E.path_dirN)), R.cur.push_back((unsigned)(unsigned short)E.cur(0));
            // Long enough to cross every face that is left: a segment that ends inside a body
            // preloads the ledger for a ray that ends there, and the next event would add to it
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            P = E, have = true;
        }
        return R;
    };

    // ------------------------------------------------------------------------------------------
    // A. Gains per event, the later faces at the bent and at the incidence angle
    // ------------------------------------------------------------------------------------------
    {
        const int WOOD = 1, CONC = 2, PANEL = 3;
        Pal heavy;
        heavy["a"] = {2.1609e-8, 2.401e-9, 2.1609e-8};
        heavy["fRef"] = {0.001, 0.001, 0.001};
        heavy["tf"] = {0.2, 0.1, 0.2};
        heavy["m"] = {20.0, 30.0, 20.0};
        heavy["thk"] = {0.0, 0.0, 0.02};
        Pal light = heavy;
        light["fRef"] = {0.02, 0.02, 0.02};
        struct St
        {
            const char *name;
            std::vector<double> x;
            std::vector<int> mat;
            std::vector<bool> flip;
        };
        const std::vector<St> sts = {
            {"wood 50 | conc 50              ", {0.0, 0.05, 0.05 + gap, 0.10}, {WOOD, WOOD, CONC, CONC}, {false, true, false, true}},
            {"conc 50 | wood 50              ", {0.0, 0.05, 0.05 + gap, 0.10}, {CONC, CONC, WOOD, WOOD}, {false, true, false, true}},
            {"panel | conc 100               ", {0.0, gap, 0.10}, {PANEL, CONC, CONC}, {false, false, true}},
            {"conc 100 | panel               ", {0.0, 0.10, 0.10 + gap}, {CONC, CONC, PANEL}, {false, true, true}},
            {"conc 100 / air 500 / conc 100 ", {0.0, 0.10, 0.60, 0.70}, {CONC, CONC, CONC, CONC}, {false, true, false, true}},
            {"conc 100 / air 500 / panel|conc", {0.0, 0.10, 0.60, 0.60 + gap, 0.70}, {CONC, CONC, PANEL, CONC, CONC}, {false, true, false, false, true}},
            {"conc 100 / air 500 / panel     ", {0.0, 0.10, 0.60}, {CONC, CONC, PANEL}, {false, true, false}},
            {"conc|wood / air 500 / wood 50  ", {0.0, 0.05, 0.05 + gap, 0.10, 0.60, 0.65}, {CONC, CONC, WOOD, WOOD, WOOD, WOOD}, {false, true, false, true, false, true}},
        };
        QD_DIAG("\n# diag: leak direction, rigid walls. Wood eps 2.1609e-8 tf 0.2 m 20, concrete eps 2.401e-9 tf 0.1 m 30,\n");
        QD_DIAG("# panel: the wood as a 20 mm sheet. pal heavy: fRef 0.001 (X0 50 and 354 at 50 mm); light: fRef 0.02 (2.5 and 3.95).\n");
        QD_DIAG("# Through the intersector, scalar, tracer 4 undeviated and 5 refracting. var now: the code. hold: every face out of\n");
        QD_DIAG("# a mass-law medium hands on the direction the wave came in with. hold_air: only an exit into air does.\n");
        QD_DIAG("# Per event k: dk the angle of the direction the wave is handed in with, deg; sk the angle of the segment; gk the\n");
        QD_DIAG("# gain in dB. tot: the sum. out: the direction after the last event. dy: lateral walk first to last hit, m\n");
        QD_DIAG("pal,stack,tracer,var,theta,d0,s0,g0,d1,s1,g1,d2,s2,g2,d3,s3,g3,d4,s4,g4,tot_dB,out_deg,dy_m,codes,rts,curs\n");
        double d54 = 0.0; // largest difference of an event's gain between the two tracers, dB
        for (int ip = 0; ip < 2; ++ip)
            for (const auto &S : sts)
                for (double thd : {30.0, 60.0, 75.0})
                {
                    Row RR[2][3];
                    for (int it = 0; it < 2; ++it)
                        for (int hold = 0; hold < 3; ++hold)
                        {
                            int itype = it ? 5 : 4;
                            arma::uvec mi;
                            arma::mat mesh = build(S.x, S.mat, S.flip, mi), orig, dest;
                            diag_ray_at(thd, orig, dest);
                            RR[it][hold] = chain(ip ? light : heavy, mesh, mi, itype, orig, dest, nullptr, hold);
                            const Row &R = RR[it][hold];
                            INFO(S.name << ", tracer " << itype << ", theta " << thd << ", variant " << hold);
                            REQUIRE(R.ok);
                            QD_DIAG("%s,%s,%d,%s,%.0f", ip ? "light" : "heavy", S.name, itype, hold == 0 ? "now     " : (hold == 1 ? "hold    " : "hold_air"), thd);
                            for (size_t k = 0; k < 5; ++k)
                                if (k < R.g.size())
                                    QD_DIAG(",%.4f,%.4f,%+.4f", R.dir[k], R.seg[k], R.g[k]);
                                else
                                    QD_DIAG(",,,");
                            QD_DIAG(",%+.4f,%.4f,%+.5f,", R.tot, R.out.empty() ? 0.0 : R.out.back(), R.dy);
                            for (unsigned c : R.code)
                                QD_DIAG("%u/", c);
                            QD_DIAG(",");
                            for (unsigned c : R.rt)
                                QD_DIAG("%u/", c);
                            QD_DIAG(",");
                            for (unsigned c : R.cur)
                                QD_DIAG("%04x/", c);
                            QD_DIAG("\n");
                        }
                    const double rad = QPI / 180.0;
                    for (int it = 0; it < 2; ++it)
                    {
                        const Row &N = RR[it][0], &H = RR[it][1];
                        INFO((ip ? "light, " : "heavy, ") << S.name << ", tracer " << (it ? 5 : 4) << ", theta " << thd);
                        REQUIRE(N.g.size() == H.g.size());
                        for (size_t k = 0; k < N.g.size(); ++k)
                        {
                            INFO("event " << k);
                            CHECK(std::abs(N.out[k] - thd) * rad < 1.0e-9); // the direction written
                            CHECK(std::abs(N.seg[k] - thd) * rad < 1.0e-9); // the segment the event was met on
                            CHECK(std::abs(N.dir[k] - thd) * rad < 1.0e-9); // the direction handed in
                            CHECK((N.cur[k] & 0x4000u) == 0u);              // no leg is a bent one
                            CHECK(std::abs(N.g[k] - H.g[k]) < 1.0e-9);      // every face at the incidence angle
                        }
                    }
                    {
                        // The refracting tracer is the undeviated one on a rigid stack
                        const Row &N4 = RR[0][0], &N5 = RR[1][0];
                        const bool gapped = std::string(S.name).find('/') != std::string::npos;
                        INFO((ip ? "light, " : "heavy, ") << S.name << ", tracer 5 on tracer 4, theta " << thd);
                        REQUIRE(N4.g.size() == N5.g.size());
                        for (size_t k = 0; k < N4.g.size(); ++k)
                        {
                            INFO("event " << k);
                            CHECK(N5.code[k] == N4.code[k]);
                            CHECK(N5.rt[k] == N4.rt[k]);
                            CHECK(N5.cur[k] == N4.cur[k]);
                            // Exact on one wall. Behind an air gap the two tracers relaunch from
                            // points built differently and the last digit can differ: recorded
                            if (!gapped)
                                CHECK(N5.g[k] == N4.g[k]);
                            d54 = std::max(d54, std::abs(N5.g[k] - N4.g[k]));
                        }
                        CHECK(std::abs(N5.dy - N4.dy) < 1.0e-4); // the same place, up to the relaunch offsets
                    }
                }
        QD_DIAG("# largest difference of one event between tracer 5 and tracer 4, dB: %.1e\n", d54);
    }

    // ------------------------------------------------------------------------------------------
    // B. The direction a closed face writes
    // ------------------------------------------------------------------------------------------
    {
        // diag_palette with a tenth material, RARE: eps 1, lossless, a solid. The host eps
        // 1.50 - 0.40j is entered from the lossless eps 5.24 solid the ray is launched in, with a
        // transverse invariant p; from p = 1 the wave is closed towards air and towards RARE.
        const int DENSE = 2, SHEET = 3, HOST = 5, RARE = 10;
        Pal base = diag_palette<double>();
        for (auto &kv : base)
            kv.second.push_back(kv.first == "a" ? 1.0 : (kv.first == "fRef" ? 1.0 : 0.0));
        Pal with_tf = base;
        with_tf["tf"] = std::vector<double>(10, 0.0);
        with_tf["tf"][(size_t)SHEET - 1] = 0.5, with_tf["tf"][(size_t)RARE - 1] = 0.5;
        const double e_d = 5.24;

        QD_DIAG("\n# diag: leak direction, the direction a closed face writes. Launched in a lossless eps 5.24 solid, into host eps\n");
        QD_DIAG("# 1.50 - 0.40j 100 mm, invariant p; then bare: host | air. glued: host | coating 6 mm | air. trans: host | RARE\n");
        QD_DIAG("# eps 1, a solid. Through the intersector, scalar, series declined. tf: 0.5 on the coating and on RARE.\n");
        QD_DIAG("# in: the direction the wave has in the host; seg: the segment; out: the direction written; traj: the\n");
        QD_DIAG("# relaunched segment, all deg against the normal. tr: the total-reflection bit of the resolved type\n");
        QD_DIAG("p,tracer,face,tf,code,rt,tr,in_deg,seg_deg,out_deg,traj_deg,gain_dB,cur\n");
        for (double p : {0.90, 1.01, 1.04, 1.10, 1.20, 1.45})
            for (int itype : {4, 5})
                for (int kind = 0; kind < 3; ++kind)
                    for (int tf = 0; tf < 2; ++tf)
                    {
                        arma::uvec mi;
                        arma::mat mesh = kind == 0   ? build({0.0, gap, L}, {DENSE, HOST, HOST}, {true, false, true}, mi)
                                         : kind == 1 ? build({0.0, gap, L, L + gap}, {DENSE, HOST, HOST, SHEET}, {true, false, true, true}, mi)
                                                     : build({0.0, gap, L, L + gap, L + 0.05}, {DENSE, HOST, HOST, RARE, RARE}, {true, false, true, false, true}, mi);
                        double th1 = std::asin(std::sqrt(p / e_d));
                        arma::mat orig{{-0.05, 0.37, 0.0}}, dest{{-0.05 + 2.0 * std::cos(th1), 0.37 + 2.0 * std::sin(th1), 0.0}};
                        Ev seed;
                        seed.prev = arma::Col<short>{(short)0}, seed.cur = arma::Col<short>{(short)DENSE}, seed.buf = arma::Col<short>{(short)0};
                        seed.path_dirN = unit(dest - orig), seed.accg = arma::vec{0.0}, seed.accr = arma::vec{0.0};
                        const Pal &pal = tf ? with_tf : base;

                        // Two events by hand, so that the second can be read in full
                        Ev A = traced(pal, mesh, mi, itype, orig, dest, &seed, 1.5);
                        REQUIRE(A.nh != 0);
                        REQUIRE(mat_of(A.cur(0)) == HOST);
                        arma::mat d = unit(A.destN - A.origN), o2 = A.origN, d2 = A.origN + 2.0 * d;
                        Ev B = traced(pal, mesh, mi, itype, o2, d2, &A, 1.5);
                        REQUIRE(B.nh != 0);
                        QD_DIAG("%.2f,%d,%s,%d,%u,%u,%d,%.4f,%.4f,%.4f,%.4f,%+.4f,%04x\n", p, itype, kind == 0 ? "bare " : (kind == 1 ? "glued" : "trans"), tf,
                                (unsigned)B.ot, (unsigned)B.rt, (int)rt_bit(B.rt, 5), ang(A.path_dirN), ang(d), ang(B.path_dirN),
                                ang(B.destN - B.origN), dB(std::norm(cpx(B.xpr, 0, 0))), (unsigned)(unsigned short)B.cur(0));
                        INFO("closed face, p " << p << ", tracer " << itype << ", face " << kind << ", tf " << tf);
                        CHECK(rt_bit(B.rt, 5) == (p > 1.0));
                        if (p > 1.0) // closed: the leak travels undeviated, whatever the face is
                        {
                            arma::mat w = unit(A.path_dirN), o = unit(B.path_dirN);
                            CHECK(std::max(std::abs(o(0, 0) - w(0, 0)), std::abs(o(0, 1) - w(0, 1))) < 1.0e-12);
                        }
                    }
    }
}

TEST_CASE("Measurement - the mirror behind a second solid and behind a sheet inside a wall")
{
    // Plan item 1, measured and nothing asserted about the code beyond sanity. Two traced solids
    // that share a face are two series, and the second one's takes the first as a half-space
    // where the wall has it as a layer backed by air; a sheet inside a wall is the same limit.
    // Everything runs through ray_triangle_intersect, ray_mesh_interact and ray_state_update,
    // scalar, both tracers. The references are written here as a TE characteristic matrix in
    // the admittance b = sqrt(eps - sin^2), sharing no code with the library.
    //
    // A. The series of each solid as a ratio, the resolving run over the declined one, against
    //    the factorized form with the mirror behind as a half-space and as a layer backed by air.
    // B. The transmitted field as a sum over the paths a tracer follows, every series resolved:
    //    the straight path, then the paths that reflect at the far face, return into the first
    //    solid, reflect at the front face and come forward again, each at its polyline's point.
    // C. The same two for a wall with a sheet inside it.
    // D. Three solids: what each candidate mirror would leave on the third solid's series.
    // E. The thickness code: the code the library writes, and what its step costs the mirror.
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev;
        std::vector<double> fx, ox; // per event: x of the hit point, x of the relaunch point
        bool ok = true;
        std::string rts, words;
        std::complex<double> at_point(double k, double st) const // the field at the polyline's exit point
        { return vv * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * st * dy)); }
    };
    // seq: one type per event, or one type repeated until the ray is out. epsv likewise
    auto run = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq,
                   const std::vector<double> &epsv, double thd, double f) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        const int n_max = seq.size() == 1 ? 16 : (int)seq.size();
        for (int k = 0; k < n_max; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : seq[(size_t)k];
            double eps = epsv.size() == 1 ? epsv[0] : epsv[(size_t)k];
            bool tx = itype != 0 && itype != 3;
            Ev E = traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, eps, tx);
            if (E.nh == 0)
            {
                C.ok = seq.size() == 1;
                break;
            }
            C.rts += std::to_string((unsigned)E.rt) + "/";
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            // long enough to cross every face that is left
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            C.fx.push_back(fx), C.ox.push_back(E.origN(0, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        if (seq.size() > 1 && C.ev.size() != seq.size())
            C.ok = false;
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto tint = [&](std::complex<double> ea, std::complex<double> eb, double s2)
    { return 2.0 * bof(ea, s2) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };

    const double L1 = 0.05, L2 = 0.10;
    const double k1 = 2.0 * QPI * FRQ / C0;

    // ------------------------------------------------------------------------------------------
    // A. The series of each solid
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mirror, the series of each solid. air | A 50 mm | B 100 mm | air through the intersector, scalar, 1 GHz.\n");
    QD_DIAG("# S_B: B's exit, resolving over declined. d_half: |S_B - S| with A behind as a half-space; d_layer: with A as a\n");
    QD_DIAG("# layer of its thickness backed by air; sep: the two references apart; worth_dB: 20 log |S_layer / S_half|.\n");
    QD_DIAG("# |R_h|, |R_l|: the mirror behind, half-space and layer. d_A: A's own series against air | B half-space.\n");
    QD_DIAG("# d_wall: the straight path, every series resolved, against the matrix of the whole wall, dB and deg at the\n");
    QD_DIAG("# polyline's point. resid: the same with S_B replaced by S_layer. fact: the factorized form against the matrix\n");
    QD_DIAG("tracer,A,B,theta,|S_B|,arg_S_B,d_half,d_layer,sep,worth_dB,|R_h|,|R_l|,d_A,d_wall_dB,d_wall_deg,resid_dB,resid_deg,fact\n");
    for (int tr : {4, 5})
        for (int B : {5, 6})
            for (int A : {2, 1})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {A, A, B, B}, {false, true, false, true}, mi);
                    Chain cD = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 1.5}, thd, FRQ);
                    Chain cB = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 0.0}, thd, FRQ);
                    Chain cA = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 0.0, 1.5}, thd, FRQ);
                    Chain c0 = run(mtl, mesh, mi, {tr, tr, tr}, {0.0, 0.0, 0.0}, thd, FRQ);
                    INFO("series, tracer " << tr << ", A " << A << ", B " << B << ", theta " << thd);
                    REQUIRE((cD.ok && cB.ok && cA.ok && c0.ok));
                    double LA = cD.fx[1] - cD.ox[0], LB = cD.fx[2] - cD.ox[1]; // the thickness each leg crossed
                    std::complex<double> eA = diag_eps(A, FRQ), eB = diag_eps(B, FRQ);
                    std::complex<double> S_B = cpx(cB.ev[2].xpr, 0, 0) / cpx(cD.ev[2].xpr, 0, 0);
                    std::complex<double> S_A = cpx(cA.ev[1].xpr, 0, 0) / cpx(cD.ev[1].xpr, 0, 0);
                    std::complex<double> pA = phi(eA, s2, k1, LA), pB = phi(eB, s2, k1, LB);
                    std::complex<double> R_h = rint(eB, eA, s2), R_l = cm(eB, {{eA, LA}}, e_air, s2, k1).r, r_out = rint(eB, e_air, s2);
                    std::complex<double> S_half = 1.0 / (1.0 - R_h * r_out * pB * pB), S_layer = 1.0 / (1.0 - R_l * r_out * pB * pB);
                    std::complex<double> S_A_ref = 1.0 / (1.0 - rint(eA, e_air, s2) * rint(eA, eB, s2) * pA * pA);
                    std::complex<double> W = cm(e_air, {{eA, LA}, {eB, LB}}, e_air, s2, k1).t;
                    std::complex<double> fact = cm(e_air, {{eA, LA}}, eB, s2, k1).t * pB * tint(eB, e_air, s2) * S_layer;
                    std::complex<double> F0 = c0.at_point(k1, st) / W, F0c = F0 * S_layer / S_B;
                    QD_DIAG("%d,%d,%d,%.0f,%.6f,%+.3f,%.3e,%.3e,%.3e,%+.4f,%.4f,%.4f,%.3e,%+.4f,%+.3f,%+.6f,%+.4f,%.1e\n", tr, A, B, thd,
                            std::abs(S_B), deg(S_B), std::abs(S_B - S_half), std::abs(S_B - S_layer), std::abs(S_half - S_layer),
                            2.0 * dB(std::abs(S_layer / S_half)), std::abs(R_h), std::abs(R_l), std::abs(S_A - S_A_ref),
                            dB(std::norm(F0)), deg(F0), dB(std::norm(F0c)), deg(F0c), std::abs(fact - W) / std::abs(W));
                    CHECK(std::abs(fact - W) < 1.0e-12 * std::abs(W)); // the factorization is the wall, reference against reference
                    // Each series is the one of its own body between half-spaces: that is the
                    // decomposition, and the paths below carry what the half-spaces do not return
                    CHECK(std::abs(S_B - S_half) < 1.0e-9);
                    CHECK(std::abs(S_A - S_A_ref) < 1.0e-9);
                    CHECK(std::abs(S_half - S_layer) > 1.0e-3);    // the two mirrors can be told apart
                    CHECK(std::abs(dB(std::norm(F0c))) < 1.0e-3); // the first path is off the wall by the mirror and nothing else
                }

    // ------------------------------------------------------------------------------------------
    // B. The paths a tracer follows, summed
    // ------------------------------------------------------------------------------------------
    auto seq_of = [](int tr, int n) // the path that couples the two series n times
    {
        std::vector<int> s = {tr, tr};
        for (int i = 0; i < n; ++i)
            s.push_back(3), s.push_back(tr), s.push_back(3), s.push_back(tr);
        s.push_back(tr);
        return s;
    };
    QD_DIAG("\n# diag: mirror, the paths a tracer follows. The same wall, every series resolved (eps 0). Path n reflects n times\n");
    QD_DIAG("# at B's far face, returns into A, reflects at A's front face and comes forward again: 3 + 4 n events. Each path\n");
    QD_DIAG("# at its polyline's exit point; d0, d01, d012: the sum of the first one, two and three paths against the matrix\n");
    QD_DIAG("# of the whole wall, dB; ph: its phase, deg. p1_dB: path 1 over path 0. geo: |F2/F1 - F1/F0|, the paths a\n");
    QD_DIAG("# geometric sequence. dinf: its limit F0 / (1 - F1/F0) against the matrix, dB and deg. rts1: the resolved types of path 1\n");
    QD_DIAG("tracer,A,B,theta,d0_dB,d01_dB,d012_dB,ph0,ph01,ph012,p1_dB,p2_dB,geo,dinf_dB,dinf_deg,rts1\n");
    for (int tr : {4, 5})
        for (int B : {5, 6})
            for (int A : {2, 1})
                for (double thd : angles)
                {
                    double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {A, A, B, B}, {false, true, false, true}, mi);
                    Chain c[3];
                    std::complex<double> F[3] = {0.0, 0.0, 0.0};
                    for (int n = 0; n < 3; ++n)
                    {
                        c[n] = run(mtl, mesh, mi, seq_of(tr, n), {0.0}, thd, FRQ);
                        if (c[n].ok)
                            F[n] = c[n].at_point(k1, st);
                    }
                    INFO("paths, tracer " << tr << ", A " << A << ", B " << B << ", theta " << thd);
                    REQUIRE(c[0].ok);
                    double LA = c[0].fx[1] - c[0].ox[0], LB = c[0].fx[2] - c[0].ox[1];
                    std::complex<double> W = cm(e_air, {{diag_eps(A, FRQ), LA}, {diag_eps(B, FRQ), LB}}, e_air, s2, k1).t;
                    std::complex<double> s0 = F[0] / W, s1 = (F[0] + F[1]) / W, s2c = (F[0] + F[1] + F[2]) / W;
                    std::complex<double> q = F[1] / F[0], sinf = s0 / (1.0 - q);
                    QD_DIAG("%d,%d,%d,%.0f,%+.4f,%+.4f,%+.4f,%+.3f,%+.3f,%+.3f,%+.2f,%+.2f,%.1e,%+.6f,%+.4f,%s%s\n", tr, A, B, thd, dB(std::norm(s0)),
                            dB(std::norm(s1)), dB(std::norm(s2c)), deg(s0), deg(s1), deg(s2c), dB(std::norm(F[1] / F[0])),
                            dB(std::norm(F[2] / F[0])), std::abs(F[2] / F[1] - q), dB(std::norm(sinf)), deg(sinf), c[1].rts.c_str(),
                            c[1].ok ? "" : " dead");
                    // The paths the tracer follows are a geometric sequence and their sum is the wall
                    REQUIRE((c[1].ok && c[2].ok));
                    CHECK(std::abs(F[2] / F[1] - q) < 1.0e-4);
                    // Every path is short by its own relaunch offsets, which the reference cannot
                    // hold: up to 1e-5 m per leg with the refracting tracer at 85 deg, and a lossless
                    // second layer there turns that into 5e-3 dB on the limit
                    CHECK(std::abs(dB(std::norm(sinf))) < ((tr == 5 && B == 6 && thd == 85.0) ? 1.0e-2 : 1.0e-3));
                    CHECK(std::abs(deg(sinf)) < 0.05);
                }
    {
        arma::uvec mi;
        arma::mat mesh = build({0.0, L1, L1 + gap, L1 + gap + L2}, {2, 2, 5, 5}, {false, true, false, true}, mi);
        Chain c = run(mtl, mesh, mi, seq_of(4, 1), {0.0}, 40.0, FRQ);
        QD_DIAG("# the state words after each event of path 1, A 2, B 5, 40 deg, tracer 4, as prev.cur.buf: %s\n", c.words.c_str());
    }

    // ------------------------------------------------------------------------------------------
    // C. A wall with a sheet inside it
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mirror, a sheet inside a wall. air | host 50 | sheet 6 mm eps 5.24 - 0.63j | host 50 | air, scalar, 1 GHz.\n");
    QD_DIAG("# S_2: the second half's series at the exit; d_host: against the sheet behind backed by a half-space of the host;\n");
    QD_DIAG("# d_wall: against the sheet and the first half backed by air; worth_dB: 20 log of the two. d_1: the first half's\n");
    QD_DIAG("# series at the sheet event against air | sheet backed by a half-space of the host. Paths as above, the sheet\n");
    QD_DIAG("# crossed in place of the shared face, against the matrix of host | sheet | host with 6 mm of host divided out\n");
    QD_DIAG("# (W). dinf: the limit of the paths against W; dinf2: against host 47 | sheet 6 | host 47, nothing divided out;\n");
    QD_DIAG("# dq: the path ratio F1/F0 against the one of W, r^2 Ts^2 pa^2 pb^2 S_1 S_host, and dq_d: with Ts over the 6 mm of host\n");
    QD_DIAG("tracer,host,theta,|S_2|,arg_S_2,d_host,d_wall,sep,worth_dB,|R_h|,|R_w|,d_1,d0_dB,d01_dB,d012_dB,ph0,ph01,ph012,p1_dB,fact,geo,dinf_dB,dinf_deg,dinf2_dB,dinf2_deg,dq,dq_d,rts1\n");
    for (int tr : {4, 5})
        for (int h : {5, 6})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, 0.05, 0.10}, {h, SHEET, h}, {false, false, true}, mi);
                Chain cD = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 1.5}, thd, FRQ);
                Chain c2 = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 1.5, 0.0}, thd, FRQ);
                Chain c1 = run(mtl, mesh, mi, {tr, tr, tr}, {1.5, 0.0, 1.5}, thd, FRQ);
                INFO("sheet wall, tracer " << tr << ", host " << h << ", theta " << thd);
                REQUIRE((cD.ok && c2.ok && c1.ok));
                double La = cD.fx[1] - cD.ox[0], Lb = cD.fx[2] - cD.ox[1];
                std::complex<double> eH = diag_eps(h, FRQ), eS = diag_eps(SHEET, FRQ);
                std::complex<double> S_2 = cpx(c2.ev[2].xpr, 0, 0) / cpx(cD.ev[2].xpr, 0, 0);
                std::complex<double> S_1 = cpx(c1.ev[1].xpr, 0, 0) / cpx(cD.ev[1].xpr, 0, 0);
                std::complex<double> pa = phi(eH, s2, k1, La), pb = phi(eH, s2, k1, Lb), r_out = rint(eH, e_air, s2);
                RT Sh = cm(eH, {{eS, thk}}, eH, s2, k1);
                std::complex<double> R_h = Sh.r, R_w = cm(eH, {{eS, thk}, {eH, La}}, e_air, s2, k1).r;
                std::complex<double> S_host = 1.0 / (1.0 - R_h * r_out * pb * pb), S_wall = 1.0 / (1.0 - R_w * r_out * pb * pb);
                std::complex<double> S_1_ref = 1.0 / (1.0 - r_out * R_h * pa * pa);
                std::complex<double> W = cm(e_air, {{eH, La}, {eS, thk}, {eH, Lb}}, e_air, s2, k1).t / phi(eH, s2, k1, thk);
                std::complex<double> fact = tint(e_air, eH, s2) * pa * S_1_ref * (Sh.t / phi(eH, s2, k1, thk)) * pb * S_wall * tint(eH, e_air, s2);
                Chain c[3];
                std::complex<double> F[3] = {0.0, 0.0, 0.0};
                for (int n = 0; n < 3; ++n)
                {
                    c[n] = run(mtl, mesh, mi, seq_of(tr, n), {0.0}, thd, FRQ);
                    if (c[n].ok)
                        F[n] = c[n].at_point(k1, st);
                }
                REQUIRE(c[0].ok);
                std::complex<double> s0 = F[0] / W, s1 = (F[0] + F[1]) / W, s2c = (F[0] + F[1] + F[2]) / W;
                std::complex<double> q = F[1] / F[0], sinf = s0 / (1.0 - q);
                std::complex<double> W2 = cm(e_air, {{eH, La - 0.5 * thk}, {eS, thk}, {eH, Lb - 0.5 * thk}}, e_air, s2, k1).t;
                std::complex<double> sinf2 = sinf * W / W2;
                std::complex<double> q_W = r_out * r_out * Sh.t * Sh.t * pa * pa * pb * pb * S_1_ref * S_host;
                std::complex<double> q_d = q_W / (phi(eH, s2, k1, thk) * phi(eH, s2, k1, thk));
                QD_DIAG("%d,%d,%.0f,%.6f,%+.3f,%.3e,%.3e,%.3e,%+.4f,%.4f,%.4f,%.3e,%+.4f,%+.4f,%+.4f,%+.3f,%+.3f,%+.3f,%+.2f,%.1e,%.1e,%+.6f,%+.4f,%+.6f,%+.4f,%.2e,%.2e,%s%s\n", tr, h,
                        thd, std::abs(S_2), deg(S_2), std::abs(S_2 - S_host), std::abs(S_2 - S_wall), std::abs(S_host - S_wall),
                        2.0 * dB(std::abs(S_wall / S_host)), std::abs(R_h), std::abs(R_w), std::abs(S_1 - S_1_ref), dB(std::norm(s0)),
                        dB(std::norm(s1)), dB(std::norm(s2c)), deg(s0), deg(s1), deg(s2c), dB(std::norm(F[1] / F[0])),
                        std::abs(fact - W) / std::abs(W), std::abs(F[2] / F[1] - q), dB(std::norm(sinf)), deg(sinf), dB(std::norm(sinf2)),
                        deg(sinf2), std::abs(q - q_W) / std::abs(q_W), std::abs(q - q_d) / std::abs(q_d), c[1].rts.c_str(), c[1].ok ? "" : " dead");
                CHECK(std::abs(fact - W) < 1.0e-12 * std::abs(W));
                CHECK(std::isfinite(std::abs(S_2)));
            }
    {
        arma::uvec mi;
        arma::mat mesh = build({0.0, 0.05, 0.10}, {5, SHEET, 5}, {false, false, true}, mi);
        Chain c = run(mtl, mesh, mi, seq_of(4, 1), {0.0}, 40.0, FRQ);
        QD_DIAG("# the state words after each event of path 1, host 5, 40 deg, tracer 4, as prev.cur.buf: %s\n", c.words.c_str());
    }

    // ------------------------------------------------------------------------------------------
    // D. Three solids
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mirror, three solids. air | A 50 | B 100 | C 50 | air, every series resolved, tracer 4. S_C: C's series\n");
    QD_DIAG("# at the exit. d_half: against B behind as a half-space. In dB on the path, 20 log |S_x / S_exact| with S_exact\n");
    QD_DIAG("# the mirror {B, A} backed by air: half, B a half-space; B_air, B a layer backed by air; B_A, B a layer backed\n");
    QD_DIAG("# by a half-space of A. d_wall: the straight path against the matrix of the whole wall\n");
    QD_DIAG("A,B,C,theta,|S_C|,d_half,half_dB,B_air_dB,B_A_dB,d_wall_dB\n");
    for (int trip = 0; trip < 3; ++trip)
        for (double thd : angles)
        {
            const int A = trip == 0 ? 2 : (trip == 1 ? 1 : 2), B = trip == 0 ? 6 : 5, Cm = trip == 0 ? 2 : (trip == 1 ? 4 : 6);
            const double x1 = L1, x2 = x1 + gap, x3 = x2 + L2, x4 = x3 + gap, x5 = x4 + L1;
            double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
            arma::uvec mi;
            arma::mat mesh = build({0.0, x1, x2, x3, x4, x5}, {A, A, B, B, Cm, Cm}, {false, true, false, true, false, true}, mi);
            Chain cD = run(mtl, mesh, mi, {4, 4, 4, 4}, {0.0, 0.0, 0.0, 1.5}, thd, FRQ);
            Chain c0 = run(mtl, mesh, mi, {4, 4, 4, 4}, {0.0, 0.0, 0.0, 0.0}, thd, FRQ);
            INFO("three solids, " << A << " " << B << " " << Cm << ", theta " << thd);
            REQUIRE((cD.ok && c0.ok));
            double LA = c0.fx[1] - c0.ox[0], LB = c0.fx[2] - c0.ox[1], LC = c0.fx[3] - c0.ox[2];
            std::complex<double> eA = diag_eps(A, FRQ), eB = diag_eps(B, FRQ), eC = diag_eps(Cm, FRQ);
            std::complex<double> S_C = cpx(c0.ev[3].xpr, 0, 0) / cpx(cD.ev[3].xpr, 0, 0), pC = phi(eC, s2, k1, LC), r_out = rint(eC, e_air, s2);
            auto S_of = [&](std::complex<double> R)
            { return 1.0 / (1.0 - R * r_out * pC * pC); };
            std::complex<double> S_half = S_of(rint(eC, eB, s2)), S_Bair = S_of(cm(eC, {{eB, LB}}, e_air, s2, k1).r);
            std::complex<double> S_BA = S_of(cm(eC, {{eB, LB}}, eA, s2, k1).r), S_ex = S_of(cm(eC, {{eB, LB}, {eA, LA}}, e_air, s2, k1).r);
            std::complex<double> W = cm(e_air, {{eA, LA}, {eB, LB}, {eC, LC}}, e_air, s2, k1).t;
            QD_DIAG("%d,%d,%d,%.0f,%.6f,%.3e,%+.4f,%+.4f,%+.4f,%+.4f\n", A, B, Cm, thd, std::abs(S_C), std::abs(S_C - S_half),
                    2.0 * dB(std::abs(S_half / S_ex)), 2.0 * dB(std::abs(S_Bair / S_ex)), 2.0 * dB(std::abs(S_BA / S_ex)),
                    dB(std::norm(c0.at_point(k1, st) / W)));
            CHECK(std::isfinite(std::abs(S_C)));
        }

    // ------------------------------------------------------------------------------------------
    // E. The thickness code
    // ------------------------------------------------------------------------------------------
    // The library writes the code only out of a mass-law solid, so A carries the mass column
    // here and nothing else of the row is read: the buffer word after the transition A -> B.
    QD_DIAG("\n# diag: mirror, the thickness code. air | A d | B 100 mm | air with the mass column on A (eps 5.24), tracer 4:\n");
    QD_DIAG("# word: the buffer word after A -> B; k: its code; k_fix: the code formula applied to the thickness the leg\n");
    QD_DIAG("# crossed; d_q: the decoded thickness; rel: d_q over the crossed thickness less one; off: the crossed thickness\n");
    QD_DIAG("# over the meshed one less one, the relaunch offset. ph: what d_q - d is on a round trip through A, deg, at 1\n");
    QD_DIAG("# and 60 GHz. dS: 20 log |S_layer(d_q) / S_layer(d)| for B eps 1.50 lossless behind A, dB, at 1 and 60 GHz\n");
    QD_DIAG("d_mm,theta,word,k,k_fix,d_q_mm,rel,off,ph_1GHz,ph_60GHz,dS_1GHz,dS_60GHz\n");
    {
        Pal pm = mtl;
        pm["m"] = std::vector<double>(9, 0.0);
        pm["m"][1] = 20.0;
        for (double d : {0.05, 0.10, 0.0731, 0.0123, 0.20})
            for (double thd : {10.0, 60.0, 85.0})
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, d, d + gap, d + gap + L2}, {2, 2, 6, 6}, {false, true, false, true}, mi);
                Chain c = run(pm, mesh, mi, {4, 4}, {1.5, 1.5}, thd, FRQ);
                INFO("thickness code, d " << d << ", theta " << thd);
                REQUIRE(c.ok);
                unsigned word = (unsigned)(unsigned short)c.ev[1].buf(0);
                int k = (int)(word & 0x7FFFu);
                double LA = c.fx[1] - c.ox[0];
                double kf = 1.0 + std::round((std::log10(LA) + 4.0) * (32766.0 / 6.0));
                double d_q = std::pow(10.0, -4.0 + 6.0 * (double)(k - 1) / 32766.0);
                double ph[2], dS[2];
                for (int i = 0; i < 2; ++i)
                {
                    double f = i ? 60.0e9 : 1.0e9, kk = 2.0 * QPI * f / C0;
                    std::complex<double> eA = diag_eps(2, f), eB = diag_eps(6, f), pB = phi(eB, s2, kk, L2), r_out = rint(eB, e_air, s2);
                    ph[i] = 2.0 * kk * std::real(bof(eA, s2)) * (d_q - LA) * 180.0 / QPI;
                    std::complex<double> Sq = 1.0 / (1.0 - cm(eB, {{eA, d_q}}, e_air, s2, kk).r * r_out * pB * pB);
                    std::complex<double> Sx = 1.0 / (1.0 - cm(eB, {{eA, LA}}, e_air, s2, kk).r * r_out * pB * pB);
                    dS[i] = 2.0 * dB(std::abs(Sq / Sx));
                }
                QD_DIAG("%.2f,%.0f,%04x,%d,%.0f,%.5f,%+.2e,%+.2e,%+.4f,%+.3f,%+.5f,%+.4f\n", d * 1e3, thd, word, k, kf, d_q * 1e3, d_q / LA - 1.0,
                        LA / d - 1.0, ph[0], ph[1], dS[0], dS[1]);
                CHECK((word & 0x8000u) != 0u); // the code is there to be read
            }
    }
}

TEST_CASE("Sheet reflection - a declined series is followed and not dropped")
{
    // A host leg that closes at a sheet either has its bounce series summed or leaves the bounces
    // to the tracer. A reflection off the sheet flagged the ray resolved either way, so with the
    // series declined the next reflection of that ray was killed: neither summed nor followed.
    // The flag belongs to a ray whose series was summed and to no other.
    //
    // Through ray_triangle_intersect, ray_mesh_interact and ray_state_update, scalar, both
    // tracers. Walls: HS, air | host 100 | coating 6 mm | air; HSH, air | host 50 | sheet | host 50
    // | air. eps 1.5 declines every series, eps 0 sums every one.
    // A. The current-medium word after the reflection off the sheet.
    // B. HS, declined: the transmitted paths, one more round trip each, and the reflected ones,
    //    as geometric sequences whose limits are the wall. Control: eps 0 on one path.
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    const std::vector<double> angles = {10.0, 20.0, 40.0, 60.0, 75.0, 85.0};
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        std::complex<double> vv = 1.0;
        double Lin = 0.0, dy = 0.0; // in-wall polyline length, lateral walk first to last hit
        std::vector<Ev> ev;
        std::vector<double> fx, ox; // per event: x of the hit point, x of the relaunch point
        bool ok = true;
        std::string rts, words;
        std::complex<double> at_point(double k, double st) const // the field at the polyline's exit point
        { return vv * std::exp(std::complex<double>(0.0, -k * Lin)) * std::exp(std::complex<double>(0.0, k * st * dy)); }
    };
    // seq: one type per event, or one type repeated until the ray is out. epsv likewise
    auto run = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq,
                   const std::vector<double> &epsv, double thd, double f) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false, inside = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0;
        const int n_max = seq.size() == 1 ? 16 : (int)seq.size();
        for (int k = 0; k < n_max; ++k)
        {
            int itype = seq.size() == 1 ? seq[0] : seq[(size_t)k];
            double eps = epsv.size() == 1 ? epsv[0] : epsv[(size_t)k];
            bool tx = itype != 0 && itype != 3;
            Ev E = traced(pal, mesh, mtl_ind, itype, f, orig, dest, have ? &P : nullptr, eps, tx);
            if (E.nh == 0)
            {
                C.ok = seq.size() == 1;
                break;
            }
            C.rts += std::to_string((unsigned)E.rt) + "/";
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            if (E.rt == 0)
            {
                C.ok = false;
                break;
            }
            C.vv *= cpx(E.xpr, 0, 0);
            double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
            if (k == 0)
                y0 = fy;
            if (inside)
                C.Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
            xp = fx, yp = fy, C.dy = fy - y0;
            inside = mat_of(E.cur(0)) != 0;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            // long enough to cross every face that is left
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
            C.fx.push_back(fx), C.ox.push_back(E.origN(0, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        if (seq.size() > 1 && C.ev.size() != seq.size())
            C.ok = false;
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto tint = [&](std::complex<double> ea, std::complex<double> eb, double s2)
    { return 2.0 * bof(ea, s2) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };


    const double L = 0.10, k1 = 2.0 * QPI * FRQ / C0;

    // ------------------------------------------------------------------------------------------
    // A. The state word
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: sheet reflection, the word. Enter, then reflect off the sheet (type 3). wall HS: a coating on the far\n");
    QD_DIAG("# face; HSH: a sheet inside the host. S: the reflect event at this eps over the one at eps 1.5. res: the\n");
    QD_DIAG("# resolved flag of the current-medium word after it. words as prev.cur.buf\n");
    QD_DIAG("wall,host,tracer,eps,theta,|S|,res,rt,words\n");
    for (int wall = 0; wall < 2; ++wall)
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (double eps : {1.5, 0.0})
                    for (double thd : angles)
                    {
                        arma::uvec mi;
                        arma::mat mesh = wall ? build({0.0, 0.05, L}, {h, SHEET, h}, {false, false, true}, mi)
                                              : build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                        Chain c = run(mtl, mesh, mi, {tr, 3}, {eps}, thd, FRQ), d = run(mtl, mesh, mi, {tr, 3}, {1.5}, thd, FRQ);
                        INFO("word, wall " << wall << ", host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                        REQUIRE((c.ok && d.ok));
                        std::complex<double> S = cpx(c.ev[1].xpr, 0, 0) / cpx(d.ev[1].xpr, 0, 0);
                        bool res = (c.ev[1].cur(0) & (short)0x8000) != 0, summed = std::abs(S - 1.0) > 1.0e-12;
                        QD_DIAG("%s,%d,%d,%.1f,%.0f,%.6f,%d,%u,%s\n", wall ? "HSH" : "HS ", h, tr, eps, thd, std::abs(S), (int)res, (unsigned)c.ev[1].rt,
                                c.words.c_str());
                        CHECK(mat_of(c.ev[1].cur(0)) == h);
                        CHECK(res == summed); // resolved exactly where the series was summed
                        if (eps == 0.0)
                            CHECK(summed);
                    }

    // ------------------------------------------------------------------------------------------
    // B. The paths of a wall coated on its far side
    // ------------------------------------------------------------------------------------------
    auto seqT = [](int tr, int n)
    {
        std::vector<int> s = {tr};
        for (int i = 0; i < n; ++i)
            s.push_back(3), s.push_back(3);
        s.push_back(tr);
        return s;
    };
    auto seqR = [](int tr, int n) // n >= 1: n reflections off the coating, n - 1 off the front face
    {
        std::vector<int> s = {tr, 3};
        for (int i = 1; i < n; ++i)
            s.push_back(3), s.push_back(3);
        s.push_back(tr);
        return s;
    };
    QD_DIAG("\n# diag: sheet reflection, the paths of air | host 100 | coating 6 mm | air with every series declined (eps 1.5).\n");
    QD_DIAG("# T path n: enter, n round trips (off the coating, off the front face), leave through the coating. R path 0: off\n");
    QD_DIAG("# the front face; R path n: enter, off the coating, n - 1 round trips, leave by the front face. Each at its\n");
    QD_DIAG("# polyline's point. T0, T01: the first one and two T paths against the matrix of the wall, dB; Tinf: the limit\n");
    QD_DIAG("# T0 / (1 - T1/T0); geo: |T2/T1 - T1/T0|. R01, Rinf and its phase likewise, Rinf = R0 + R1 / (1 - R2/R1).\n");
    QD_DIAG("# c0: the control, eps 0 on one path; clamp: what the clamp of a mirror above one costs that control\n");
    QD_DIAG("# dR: |Rinf - R| over the front-face term; canc: the wall's reflection over the front-face term, dB\n");
    QD_DIAG("host,tracer,theta,dR,canc_dB,T0_dB,T01_dB,Tinf_dB,geo,R01_dB,Rinf_dB,Rinf_deg,c0_dB,clamp_dB,rtsT1,rtsR2\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, L, L + gap}, {h, h, SHEET}, {false, true, true}, mi);
                Chain t[3], r[3], r0 = run(mtl, mesh, mi, {3}, {1.5}, thd, FRQ), c0 = run(mtl, mesh, mi, {tr, tr}, {0.0}, thd, FRQ);
                std::complex<double> FT[3] = {0.0, 0.0, 0.0}, FR[4] = {0.0, 0.0, 0.0, 0.0};
                bool all = r0.ok && c0.ok;
                for (int n = 0; n < 3; ++n)
                {
                    t[n] = run(mtl, mesh, mi, seqT(tr, n), {1.5}, thd, FRQ), r[n] = run(mtl, mesh, mi, seqR(tr, n + 1), {1.5}, thd, FRQ);
                    if (t[n].ok)
                        FT[n] = t[n].at_point(k1, st);
                    if (r[n].ok)
                        FR[n + 1] = r[n].at_point(k1, st);
                    all = all && t[n].ok && r[n].ok;
                }
                INFO("paths, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((t[0].ok && r0.ok && c0.ok));
                FR[0] = r0.at_point(k1, st);
                double Lx = t[0].fx[1] - t[0].ox[0];
                std::complex<double> eH = diag_eps(h, FRQ), eS = diag_eps(SHEET, FRQ);
                RT W = cm(e_air, {{eH, Lx}, {eS, thk}}, e_air, s2, k1);
                // the clamp of section 5: the coating seen from the host, above one in magnitude
                std::complex<double> Rs = cm(eH, {{eS, thk}}, e_air, s2, k1).r, r_fr = rint(eH, e_air, s2), pp = phi(eH, s2, k1, Lx);
                std::complex<double> Rc = (std::abs(Rs) > 1.0) ? Rs / std::abs(Rs) : Rs;
                double clamp_dB = 2.0 * dB(std::abs((1.0 - r_fr * Rs * pp * pp) / (1.0 - r_fr * Rc * pp * pp)));
                std::complex<double> qT = FT[1] / FT[0], Tinf = FT[0] / (1.0 - qT), qR = FR[2] / FR[1], Rinf = FR[0] + FR[1] / (1.0 - qR);
                double c0_dB = dB(std::norm(c0.at_point(k1, st)) / std::norm(W.t));
                if (all)
                    QD_DIAG("%d,%d,%.0f,%.1e,%+.2f,", h, tr, thd, std::abs(Rinf - W.r) / std::abs(FR[0]), dB(std::norm(W.r / FR[0])));
                if (all)
                    QD_DIAG("%+.4f,%+.4f,%+.6f,%.1e,%+.4f,%+.6f,%+.4f,%+.6f,%+.6f,%s,%s\n", dB(std::norm(FT[0] / W.t)),
                            dB(std::norm((FT[0] + FT[1]) / W.t)), dB(std::norm(Tinf / W.t)), std::abs(FT[2] / FT[1] - qT),
                            dB(std::norm((FR[0] + FR[1]) / W.r)), dB(std::norm(Rinf / W.r)), deg(Rinf / W.r), c0_dB, clamp_dB, t[1].rts.c_str(),
                            r[1].rts.c_str());
                else
                    QD_DIAG("%d,%d,%.0f,dead,dead,%+.4f,dead,dead,dead,%+.4f,dead,dead,%+.6f,%+.6f,%s,%s\n", h, tr, thd, dB(std::norm(FT[0] / W.t)),
                            dB(std::norm((FR[0] + FR[1]) / W.r)), c0_dB, clamp_dB, t[1].rts.c_str(), r[1].rts.c_str());
                CHECK(std::abs(c0_dB - clamp_dB) < 1.0e-3); // the control: summed, one path is the wall
                CHECK(all);                                  // no bounce of a declined series dies
                if (all)
                {
                    const double bound = (tr == 5 && thd == 85.0) ? 1.0e-2 : 1.0e-3; // the relaunch offsets, as for two solids
                    CHECK(std::abs(FT[2] / FT[1] - qT) < 1.0e-4);
                    CHECK(std::abs(dB(std::norm(Tinf / W.t))) < bound);
                    // The reflected limit against the front-face term, which is what the relaunch
                    // offsets act on: where the wall reflects 10 dB less than its front face, the
                    // same distance is more in dB (1.3e-3 dB at 16 dB of cancellation)
                    CHECK(std::abs(Rinf - W.r) < 2.0e-4 * std::abs(FR[0]));
                    CHECK(std::abs(dB(std::norm(Rinf / W.r))) < ((std::norm(W.r / FR[0]) < 0.1) ? 5.0e-3 : bound));
                    CHECK(std::abs(deg(Rinf / W.r)) < 0.05);
                }
            }
}

TEST_CASE("Measurement - the mirror behind the ray after a back reflection")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // Plan item 1, measured. A series is formed from the face ahead of the ray and the one behind
    // it, and the one behind is read from the previous-medium word. A back reflection at a bare
    // face does not write that word. Through ray_triangle_intersect, ray_mesh_interact and
    // ray_state_update, scalar, double, 1 GHz, both tracers. Nothing is asserted beyond sanity.
    //
    // Walls, all standing in air:
    //   H    host 100                          HS   host 100 | coating 6 mm
    //   SH   coating 6 mm | host 100           AB   A 50 | B 100, two solids across a bare face
    //   HSH  host 50 | sheet 6 mm | host 50
    //
    // A. One leg at a time. The event that closes a leg which began at a back reflection at a
    //    bare face, at eps 0 over the same at eps 1.5: the series it forms, against the form with
    //    the mirror that is behind the ray and against the form with what the word holds. And the
    //    threshold at which that event stops summing, found by bisection: the rho the library
    //    formed, next to the one of the first leg in the same body.
    // B. Every path. One ray, every reflected and transmitted ray followed until it is in air,
    //    dead or 20 events old, summed at the polyline's point. Four runs per row: the library at
    //    eps 0.15; the same with eps 1.5 handed to every event on a leg that began at a declined
    //    reflection (the verdict inherited); the same with the mirror written into the word by
    //    the fixture after a declined reflection at a bare face (the word made true); and eps 0.
    const double gap = 1.0e-5, thk = 0.006;
    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const std::complex<double> J(0.0, 1.0), e_air(1.0, 0.0);
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](double g)
    { return (g > 0.0) ? 10.0 * std::log10(g) : -999.0; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, fbs;
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch,
                      const std::vector<uint8_t> *vin = nullptr, std::vector<uint8_t> *vout = nullptr) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr,
                                       (arma::mat *)nullptr, (arma::mat *)nullptr, vin, vout);
        E.ot = ot[0], E.rt = rt[0], E.nh = nh(0);
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    struct Chain
    {
        std::vector<Ev> ev;
        std::vector<double> fx, ox;
        bool ok = true;
        std::string words;
    };
    auto run = [&](const arma::mat &mesh, const arma::uvec &mtl_ind, const std::vector<int> &seq, const std::vector<double> &epsv, double thd) -> Chain
    {
        Chain C;
        arma::mat orig, dest;
        diag_ray_at(thd, orig, dest);
        Ev P;
        bool have = false;
        for (size_t k = 0; k < seq.size(); ++k)
        {
            bool tx = seq[k] != 0 && seq[k] != 3;
            Ev E = traced(mtl, mesh, mtl_ind, seq[k], FRQ, orig, dest, have ? &P : nullptr, epsv.size() == 1 ? epsv[0] : epsv[k], tx);
            if (E.nh == 0 || E.rt == 0)
            {
                C.ok = false;
                break;
            }
            char w[40];
            std::snprintf(w, sizeof(w), "%04x.%04x.%04x/", (unsigned)(unsigned short)E.prev(0), (unsigned)(unsigned short)E.cur(0),
                          (unsigned)(unsigned short)E.buf(0));
            C.words += w;
            arma::mat d = E.destN - E.origN;
            d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
            orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d; // long enough to cross every face that is left
            C.fx.push_back(E.fbs(0, 0)), C.ox.push_back(E.origN(0, 0));
            P = E, have = true;
            C.ev.push_back(E);
        }
        return C;
    };

    // The reference: TE, mu = 1, admittance b = sqrt(eps - sin^2), layer phase k b d
    struct Lay
    {
        std::complex<double> e;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, double s2)
    { return std::sqrt(e - s2); };
    auto cm = [&](std::complex<double> en, const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, s2), dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / b, a21 = J * b * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(en, s2), hf = bof(ef, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    auto rint = [&](std::complex<double> ea, std::complex<double> eb, double s2) // seen from a, towards b
    { return (bof(ea, s2) - bof(eb, s2)) / (bof(ea, s2) + bof(eb, s2)); };
    auto phi = [&](std::complex<double> e, double s2, double k, double d)
    { return std::exp(-J * k * bof(e, s2) * d); };
    auto unit = [](std::complex<double> r) // the clamp of section 5 on a mirror above one
    { return (std::abs(r) > 1.0) ? r / std::abs(r) : r; };
    const double k1 = 2.0 * QPI * FRQ / C0;
    std::vector<double> angles;
    for (double a = 10.0; a < 86.0; a += 5.0)
        angles.push_back(a);

    // The walls. leg: 0 = the only body, or the first one; 1 = the second
    struct Wall
    {
        const char *name;
        int a, b; // host, or first and second solid
    };
    auto mesh_of = [&](const std::string &w, int a, int b, arma::uvec &mi) -> arma::mat
    {
        if (w == "H")
            return build({0.0, 0.10}, {a, a}, {false, true}, mi);
        if (w == "HS")
            return build({0.0, 0.10, 0.10 + gap}, {a, a, SHEET}, {false, true, true}, mi);
        if (w == "SH")
            return build({0.0, gap, 0.10 + gap}, {SHEET, a, a}, {false, false, true}, mi);
        if (w == "AB")
            return build({0.0, 0.05, 0.05 + gap, 0.15 + gap}, {a, a, b, b}, {false, true, false, true}, mi);
        return build({0.0, 0.05, 0.10}, {a, SHEET, a}, {false, false, true}, mi); // HSH
    };

    // ------------------------------------------------------------------------------------------
    // A. One leg at a time
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: mirror behind, one leg. The event that closes a leg which began at a back reflection at a bare face, at\n");
    QD_DIAG("# eps 0 over the same at eps 1.5: the series S it forms. leg: the body and the path, t a transmission, r a\n");
    QD_DIAG("# reflection. d_true: |S - S| with the mirror that is behind the ray; d_word: with what the previous-medium word\n");
    QD_DIAG("# holds. rho_true, rho_word: |r_ahead r_behind| |phi|^2 of the two, from the reference. rho_lib: the eps at which\n");
    QD_DIAG("# that event stops summing, by bisection; rho_1st: the same for the first leg in that body, whose word is true.\n");
    QD_DIAG("# worth_dB: 20 log |S_word / S_true|. win: 1 where rho_true < 0.15 <= rho_word, -1 for the reverse\n");
    QD_DIAG("wall,a,b,leg,tracer,theta,|S|,d_true,d_word,rho_true,rho_word,rho_lib,rho_1st,worth_dB,win,words\n");
    struct LegCase
    {
        const char *wall, *leg;
        int a, b;
        std::string seq, seq1; // t/r per event; the last event of each is the one measured
        int body;              // which solid the leg is in: 0 = a, 1 = b
        int ahead, behind, word; // 0 air, 1 the other solid, 2 the sheet backed by air, 3 the sheet backed by the host
    };
    std::vector<LegCase> legs;
    for (int h : {5, 6})
    {
        legs.push_back({"HS", "host:trrt", h, 0, "trrt", "tt", 0, 2, 0, 2});
        legs.push_back({"SH", "host:trt", h, 0, "trt", "tt", 0, 2, 0, 2});
        legs.push_back({"HSH", "half1:trrt", h, 0, "trrt", "tt", 0, 3, 0, 3});
        legs.push_back({"HSH", "half2:ttrt", h, 0, "ttrt", "ttt", 0, 3, 0, 3});
    }
    for (int b : {5, 6})
    {
        legs.push_back({"AB", "A:trt", 2, b, "trt", "tt", 0, 0, 1, 0});
        legs.push_back({"AB", "B:ttrt", 2, b, "ttrt", "ttt", 1, 1, 0, 1});
    }
    for (const LegCase &lc : legs)
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                arma::uvec mi;
                arma::mat mesh = mesh_of(lc.wall, lc.a, lc.b, mi);
                auto seq_of = [&](const std::string &s)
                {
                    std::vector<int> q;
                    for (char c : s)
                        q.push_back(c == 't' ? tr : 3);
                    return q;
                };
                auto last_at = [&](const std::string &s, double e, std::complex<double> &x, Chain *keep = nullptr) -> bool
                {
                    std::vector<double> ev(s.size(), 1.5);
                    ev.back() = e;
                    Chain c = run(mesh, mi, seq_of(s), ev, thd);
                    if (!c.ok)
                        return false;
                    x = cpx(c.ev.back().xpr, 0, 0);
                    if (keep)
                        *keep = c;
                    return true;
                };
                auto rho_lib = [&](const std::string &s) -> double // the eps at which the last event stops summing
                {
                    std::complex<double> x0, x;
                    if (!last_at(s, 1.5, x0) || !last_at(s, 1.0e-9, x) || std::abs(x - x0) <= 1.0e-12 * std::abs(x0))
                        return -1.0;
                    double lo = 1.0e-9, hi = 1.5;
                    for (int it = 0; it < 30; ++it)
                    {
                        double mid = 0.5 * (lo + hi);
                        last_at(s, mid, x);
                        if (std::abs(x - x0) > 1.0e-12 * std::abs(x0))
                            lo = mid;
                        else
                            hi = mid;
                    }
                    return 0.5 * (lo + hi);
                };
                Chain c;
                std::complex<double> x0, x1;
                INFO("leg, wall " << lc.wall << ", " << lc.leg << ", a " << lc.a << ", b " << lc.b << ", tracer " << tr << ", theta " << thd);
                bool ok = last_at(lc.seq, 0.0, x1, &c) && last_at(lc.seq, 1.5, x0);
                if (!ok)
                {
                    QD_DIAG("%s,%d,%d,%s,%d,%.0f,dead\n", lc.wall, lc.a, lc.b, lc.leg, tr, thd);
                    continue;
                }
                std::complex<double> S = x1 / x0;
                size_t n = c.ev.size();
                std::complex<double> eM = diag_eps(lc.body ? lc.b : lc.a, FRQ), eO = diag_eps(lc.body ? lc.a : lc.b, FRQ), eS = diag_eps(SHEET, FRQ);
                auto mirror = [&](int kind) -> std::complex<double>
                {
                    if (kind == 0)
                        return rint(eM, e_air, s2);
                    if (kind == 1)
                        return rint(eM, eO, s2);
                    return unit(cm(eM, {{eS, thk}}, kind == 2 ? e_air : eM, s2, k1).r);
                };
                std::complex<double> pp = phi(eM, s2, k1, std::abs(c.fx[n - 1] - c.ox[n - 2]));
                std::complex<double> ra = mirror(lc.ahead), rb = mirror(lc.behind), rw = mirror(lc.word);
                std::complex<double> S_true = 1.0 / (1.0 - ra * rb * pp * pp), S_word = 1.0 / (1.0 - ra * rw * pp * pp);
                double rho_t = std::abs(ra * rb) * std::norm(pp), rho_w = std::abs(ra * rw) * std::norm(pp);
                QD_DIAG("%s,%d,%d,%s,%d,%.0f,%.6f,%.3e,%.3e,%.4f,%.4f,%.4f,%.4f,%+.4f,%d,%s\n", lc.wall, lc.a, lc.b, lc.leg, tr, thd, std::abs(S),
                        std::abs(S - S_true), std::abs(S - S_word), rho_t, rho_w, rho_lib(lc.seq), rho_lib(lc.seq1), 2.0 * dB(std::abs(S_word / S_true)),
                        (rho_t < 0.15 && rho_w >= 0.15) ? 1 : ((rho_w < 0.15 && rho_t >= 0.15) ? -1 : 0), c.words.c_str());
                CHECK(std::isfinite(std::abs(S)));
                // The verdict is inherited: a leg that begins at a declined reflection forms no
                // series, whatever the threshold of the event that closes it
                CHECK(x1 == x0);
                for (size_t k = 0; k + 1 < n; ++k)
                    if (lc.seq[k] == 'r')
                    {
                        // followed: the mark in the buffer word, and the ray not resolved
                        CHECK((unsigned short)c.ev[k].buf(0) == 0x4000u);
                        CHECK((c.ev[k].cur(0) & (short)0x8000) == 0);
                        // a bare face leaves the previous-medium word as it was, so the recovery
                        // reads what it read before the reflection
                        if ((c.ev[k].rt & 0x40) == 0)
                            CHECK(c.ev[k].prev(0) == c.ev[k - 1].prev(0));
                    }
            }

    // ------------------------------------------------------------------------------------------
    // C. The reflection of a wall coated on its near side, every series summed
    // ------------------------------------------------------------------------------------------
    // At eps 0 the wall reflects on two paths: off the coating, and enter, off the bare far face
    // with the host's series, leave through the coating. Both at the polyline's point.
    auto field = [&](const Chain &c, double st) -> std::complex<double>
    {
        std::complex<double> v = 1.0;
        double Lin = 0.0;
        for (size_t k = 0; k < c.ev.size(); ++k)
        {
            v *= cpx(c.ev[k].xpr, 0, 0);
            if (k && mat_of(c.ev[k - 1].cur(0)) != 0)
                Lin += std::sqrt(std::pow(c.ev[k].fbs(0, 0) - c.ev[k - 1].fbs(0, 0), 2) + std::pow(c.ev[k].fbs(0, 1) - c.ev[k - 1].fbs(0, 1), 2));
        }
        return v * std::exp(-J * k1 * Lin) * std::exp(J * k1 * st * (c.ev.back().fbs(0, 1) - c.ev[0].fbs(0, 1)));
    };
    QD_DIAG("\n# diag: mirror behind, the reflection of a wall coated on its near side. air | coating 6 mm | host 100 | air, eps 0.\n");
    QD_DIAG("# R0: off the coating; R1: enter, off the far face, leave through the coating. R_dB, R_deg: R0 + R1 against the\n");
    QD_DIAG("# matrix of coating | host with the coating in front of the mesh plane, where the reflect port of a glued entry\n");
    QD_DIAG("# puts it. T_dB: enter, leave by the far face, against the matrix. R1/R0: dB\n");
    QD_DIAG("host,tracer,theta,R1/R0_dB,R_dB,R_deg,T_dB,words_R1\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = mesh_of("SH", h, 0, mi);
                Chain c0 = run(mesh, mi, {3}, {0.0}, thd), c1 = run(mesh, mi, {tr, 3, tr}, {0.0}, thd), ct = run(mesh, mi, {tr, tr}, {0.0}, thd);
                INFO("near coating, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((c0.ok && c1.ok && ct.ok));
                std::complex<double> R0 = field(c0, st), R1 = field(c1, st), T = field(ct, st);
                RT W = cm(e_air, {{diag_eps(SHEET, FRQ), thk}, {diag_eps(h, FRQ), std::abs(ct.fx[1] - ct.ox[0])}}, e_air, s2, k1);
                std::complex<double> Ra = R0 + R1, Rw = W.r * std::exp(2.0 * J * k1 * std::cos(th) * thk);
                QD_DIAG("%d,%d,%.0f,%+.2f,%+.6f,%+.4f,%+.6f,%s\n", h, tr, thd, dB(std::norm(R1 / R0)), dB(std::norm(Ra / Rw)),
                        std::arg(Ra / Rw) * 180.0 / QPI, dB(std::norm(T / W.t)), c1.words.c_str());
                CHECK(std::isfinite(std::abs(Ra)));
            }

    // ------------------------------------------------------------------------------------------
    // B. Every path
    // ------------------------------------------------------------------------------------------
    struct Out
    {
        std::complex<double> T = 0.0, R = 0.0;
        int nT = 0, nR = 0, n_sum = 0, n_cut = 0; // n_sum: events on a leg that began at a declined reflection and summed a series
        double worst = 0.0;                       // the largest |20 log |S|| among them
    };
    struct Node
    {
        arma::mat orig, dest;
        Ev P;
        bool have = false, inside = false, followed = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, Lin = 0.0;
        std::complex<double> vv = 1.0;
        int depth = 0;
    };
    // mode 0: the library; 1: eps 1.5 on a leg that began at a declined reflection; 2: the mirror
    // written into the previous-medium word after a declined reflection at a bare face
    auto walk = [&](const arma::mat &mesh, const arma::uvec &mi, int tr, double eps, int mode, double thd, int max_ev, double floor_dB,
                    double x_back, int A, int B, bool two_solids) -> Out
    {
        Out O;
        const double st = std::sin(thd * QPI / 180.0), floor_lin = std::pow(10.0, 0.1 * floor_dB);
        std::vector<Node> stack(1);
        diag_ray_at(thd, stack[0].orig, stack[0].dest);
        while (!stack.empty())
        {
            Node Nn = stack.back();
            stack.pop_back();
            for (int itype : {3, tr})
            {
                const bool tx = itype != 3;
                const double e_ev = (mode == 1 && Nn.followed) ? 1.5 : eps;
                Ev E = traced(mtl, mesh, mi, itype, FRQ, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, e_ev, tx);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                if (Nn.followed && mode != 1) // did this event sum a series?
                {
                    Ev D = traced(mtl, mesh, mi, itype, FRQ, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, 1.5, tx);
                    std::complex<double> S = cpx(E.xpr, 0, 0) / cpx(D.xpr, 0, 0);
                    if (D.rt != 0 && std::abs(S - 1.0) > 1.0e-12)
                        ++O.n_sum, O.worst = std::max(O.worst, std::abs(2.0 * dB(std::abs(S))));
                }
                Node M;
                M.vv = Nn.vv * cpx(E.xpr, 0, 0);
                double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                M.y0 = Nn.depth ? Nn.y0 : fy;
                M.Lin = Nn.Lin + (Nn.inside ? std::sqrt((fx - Nn.xp) * (fx - Nn.xp) + (fy - Nn.yp) * (fy - Nn.yp)) : 0.0);
                M.xp = fx, M.yp = fy, M.depth = Nn.depth + 1;
                const int cur = mat_of(E.cur(0));
                M.inside = cur != 0;
                arma::mat d = E.destN - E.origN;
                d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                if (!M.inside) // in air: collected
                {
                    std::complex<double> F = M.vv * std::exp(-J * k1 * M.Lin) * std::exp(J * k1 * st * (fy - M.y0));
                    if (d(0, 0) > 0.0 && fx > x_back - 1.0e-3)
                        O.T += F, ++O.nT;
                    else if (d(0, 0) < 0.0 && fx < 1.0e-3)
                        O.R += F, ++O.nR;
                    continue;
                }
                if (M.depth >= max_ev || std::norm(M.vv) < floor_lin)
                {
                    ++O.n_cut;
                    continue;
                }
                const bool declined_refl = !tx && (E.cur(0) & (short)0x8000) == 0;
                M.followed = declined_refl;
                M.P = E, M.have = true;
                if (mode == 2 && declined_refl && (E.rt & 0x40) == 0) // a bare face: the fixture writes what the ray was reflected off
                {
                    int m = (two_solids && std::abs(fx - 0.05) < 1.0e-3) ? (cur == A ? B : A) : 0;
                    M.P.prev(0) = (short)((E.prev(0) & (short)0x8000) | m | (m ? 0x4000 : 0));
                }
                M.orig = E.origN, M.dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
                stack.push_back(M);
            }
        }
        return O;
    };

    QD_DIAG("\n# diag: mirror behind, every path. One ray, every reflected and transmitted ray followed until it is in air, dead\n");
    QD_DIAG("# or 20 events old, summed at the polyline's point, against the matrix of the wall (HSH: against the run at\n");
    QD_DIAG("# eps 0 with 60 events), dB. lib: the library at eps 0.15; inh: eps 1.5 on every leg that began at a declined\n");
    QD_DIAG("# reflection; true: the mirror written into the word after a declined reflection at a bare face; res: eps 0.\n");
    QD_DIAG("# n: the paths behind and in front of the wall, lib. sum: the events on a leg that began at a declined reflection\n");
    QD_DIAG("# and summed a series, lib; worth: the largest of them, dB; sum_true: the same count with the word made true.\n");
    QD_DIAG("# cut: the rays stopped at 20 events, lib\n");
    QD_DIAG("wall,a,b,tracer,theta,nT,nR,cut,sum,worth,sum_true,T_lib,T_inh,T_true,T_res,R_lib,R_inh,R_true,R_res,T_inh-true,R_inh-true\n");
    std::vector<Wall> walls = {{"H", 5, 0}, {"H", 6, 0}, {"HS", 5, 0}, {"HS", 6, 0}, {"SH", 5, 0}, {"SH", 6, 0},
                               {"AB", 2, 5}, {"AB", 2, 6}, {"HSH", 5, 0}, {"HSH", 6, 0}};
    for (const Wall &w : walls)
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                const std::string nm = w.name;
                double s2 = std::sin(thd * QPI / 180.0) * std::sin(thd * QPI / 180.0);
                arma::uvec mi;
                arma::mat mesh = mesh_of(nm, w.a, w.b, mi);
                const double x_back = (nm == "AB") ? 0.15 : 0.10;
                const bool two = nm == "AB";
                Out L = walk(mesh, mi, tr, 0.15, 0, thd, 20, -400.0, x_back, w.a, w.b, two);
                Out I = walk(mesh, mi, tr, 0.15, 1, thd, 20, -400.0, x_back, w.a, w.b, two);
                Out U = walk(mesh, mi, tr, 0.15, 2, thd, 20, -400.0, x_back, w.a, w.b, two);
                Out Z = walk(mesh, mi, tr, 0.0, 0, thd, 20, -400.0, x_back, w.a, w.b, two);
                std::complex<double> eA = diag_eps(w.a, FRQ), eB = diag_eps(w.b, FRQ), eS = diag_eps(SHEET, FRQ);
                RT W;
                if (nm == "H")
                    W = cm(e_air, {{eA, 0.10}}, e_air, s2, k1);
                else if (nm == "HS")
                    W = cm(e_air, {{eA, 0.10}, {eS, thk}}, e_air, s2, k1);
                else if (nm == "SH")
                    W = cm(e_air, {{eS, thk}, {eA, 0.10}}, e_air, s2, k1);
                else if (nm == "AB")
                    W = cm(e_air, {{eA, 0.05}, {eB, 0.10}}, e_air, s2, k1);
                else
                {
                    Out Z60 = walk(mesh, mi, tr, 0.0, 0, thd, 60, -400.0, x_back, w.a, w.b, two);
                    W.t = Z60.T, W.r = Z60.R;
                }
                INFO("paths, wall " << nm << ", a " << w.a << ", b " << w.b << ", tracer " << tr << ", theta " << thd);
                CHECK(L.nT >= 1);
                QD_DIAG("%s,%d,%d,%d,%.0f,%d,%d,%d,%d,%.3f,%d,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.4f,%+.1e,%+.1e\n", w.name, w.a, w.b, tr, thd,
                        L.nT, L.nR, L.n_cut, L.n_sum, L.worst, U.n_sum, dB(std::norm(L.T / W.t)), dB(std::norm(I.T / W.t)), dB(std::norm(U.T / W.t)),
                        dB(std::norm(Z.T / W.t)), dB(std::norm(L.R / W.r)), dB(std::norm(I.R / W.r)), dB(std::norm(U.R / W.r)), dB(std::norm(Z.R / W.r)),
                        dB(std::norm(I.T / U.T)), dB(std::norm(I.R / U.R)));
            }

    // ------------------------------------------------------------------------------------------
    // D. A verdict that is handed in
    // ------------------------------------------------------------------------------------------
    // The event that closes the first leg in a body, both passes: with the verdict "sum" handed
    // in it is the event at eps 0 whatever eps says, with "do not sum" the event at eps 1.5, and
    // the verdict handed out says which of the two a call was.
    for (const char *wn : {"H", "HS", "AB"})
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (int itype : {3, tr})
                    for (double thd : {10.0, 40.0, 60.0, 75.0, 85.0})
                    {
                        arma::uvec mi;
                        arma::mat mesh = mesh_of(wn, std::string(wn) == "AB" ? 2 : h, h, mi), orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev E1 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 1.5, true);
                        arma::mat d = E1.destN - E1.origN;
                        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
                        orig = E1.origN, dest = E1.origN + (1.2 / std::abs(d(0, 0))) * d;
                        const bool tx = itype != 3;
                        const std::vector<uint8_t> yes = {1}, no = {0};
                        std::vector<uint8_t> v0, v15, vy, vn;
                        Ev S0 = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 0.0, tx, nullptr, &v0);
                        Ev S15 = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 1.5, tx, nullptr, &v15);
                        Ev Sy = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 1.5, tx, &yes, &vy);
                        Ev Sn = traced(mtl, mesh, mi, itype, FRQ, orig, dest, &E1, 0.0, tx, &no, &vn);
                        INFO("verdict, wall " << wn << ", host " << h << ", tracer " << tr << ", type " << itype << ", theta " << thd);
                        REQUIRE((S0.rt != 0 && S15.rt != 0));
                        CHECK(cpx(S0.xpr, 0, 0) != cpx(S15.xpr, 0, 0)); // the two are told apart
                        CHECK(cpx(Sy.xpr, 0, 0) == cpx(S0.xpr, 0, 0));
                        CHECK(cpx(Sn.xpr, 0, 0) == cpx(S15.xpr, 0, 0));
                        CHECK((Sy.cur(0) == S0.cur(0) && Sy.buf(0) == S0.buf(0) && Sy.prev(0) == S0.prev(0)));
                        CHECK((Sn.cur(0) == S15.cur(0) && Sn.buf(0) == S15.buf(0) && Sn.prev(0) == S15.prev(0)));
                        CHECK((v0[0] == 1 && v15[0] == 0 && vy[0] == 1 && vn[0] == 0));
                    }
}

TEST_CASE("Sheet reflect port - where the layer lies and what the echo pays")
{
#ifdef _OPENMP
    // One ray per call, many thousand calls: a thread team per call costs twenty times the work
    struct OneThread
    {
        int n = omp_get_max_threads();
        OneThread() { omp_set_num_threads(1); }
        ~OneThread() { omp_set_num_threads(n); }
    } one_thread;
#endif
    // A sheet has no thickness in the mesh. Its composite stands for a layer of its thickness
    // that lies around the mesh plane, and the tracer turns a reflected ray around at that plane,
    // so the reflect port owes the round trip through the part of the layer in front of it:
    //   glued entry, air | sheet | solid     the layer lies in the air in front, all of it
    //   one medium on both sides             the mesh plane is the mid-plane, half of it
    //   glued exit, solid | sheet | air      the layer lies in the air beyond, none
    // And the part of the reflection that entered the sheet pays the sheet's interface gain
    // once, as the echo of a traced solid does. Through ray_triangle_intersect,
    // ray_mesh_interact and ray_state_update, scalar, double.
    //
    // Every reflected sum is the plane-wave reflection referenced to the plane x = 0 and every
    // transmitted one the transfer from the plane the polyline enters at to the one it leaves at.
    //
    // A. A wall coated on its near side through ray_mesh_interact alone, no state
    // B. A wall coated on its near side and a wall lined on both faces, every path
    // C. An absorber on a rigid wall, as a sheet and as a traced solid
    // D. A sheet in front of a wall across an air gap
    // E. A sheet inside a wall
    // F. A sheet and a traced solid of one material and thickness, with att
    const double gap = 1.0e-5;
    const std::complex<double> J(0.0, 1.0), one(1.0, 0.0);
    typedef std::unordered_map<std::string, std::vector<double>> Pal;
    auto dB = [](std::complex<double> z)
    { return (std::norm(z) > 0.0) ? 10.0 * std::log10(std::norm(z)) : -999.0; };
    auto deg = [](std::complex<double> z)
    { return std::arg(z) * 180.0 / QPI; };

    struct Ev
    {
        arma::mat origN, destN, path_dirN, xpr, xpr0, fbs; // xpr0: what ray_mesh_interact wrote
        arma::Col<short> prev, cur, buf;
        arma::vec accg, accr;
        uint8_t ot = 0, rt = 0;
        unsigned nh = 0;
    };
    auto traced = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mtl_ind, int itype, double f,
                      const arma::mat &orig, const arma::mat &dest, const Ev *in, double eps, bool relaunch, bool stateless = false) -> Ev
    {
        arma::mat fbs, sbs, sbsN, nv;
        arma::u32_vec nh, fi, si;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &mesh, &fbs, &sbs, &nh, &fi, &si);
        Ev E;
        arma::vec gain, fang;
        std::vector<uint8_t> ot, rt;
        quadriga_lib::ray_mesh_interact(itype, f, &orig, &dest, &mesh, &mtl_ind, &pal, &fi, &si, (arma::mat *)nullptr,
                                        (arma::mat *)nullptr, &E.origN, &E.destN, &E.fbs, &sbsN, &gain, &E.xpr,
                                        (arma::mat *)nullptr, (arma::mat *)nullptr, &fang, (arma::vec *)nullptr,
                                        (arma::vec *)nullptr, &nv, &ot, &E.path_dirN, false, (arma::u32_vec *)nullptr);
        E.xpr0 = E.xpr, E.ot = ot[0], E.nh = nh(0);
        if (stateless)
            return E;
        arma::Col<short> mf(1), ms(1);
        mf(0) = fi(0) ? (short)mtl_ind(fi(0) - 1) : (short)0, ms(0) = si(0) ? (short)mtl_ind(si(0) - 1) : (short)0;
        quadriga_lib::ray_state_update(itype, f, &orig, &dest, &E.fbs, &sbsN, &nh, &fang, &nv, &ot, &pal, &mf, &ms,
                                       in ? &in->prev : nullptr, in ? &in->cur : nullptr, in ? &in->buf : nullptr,
                                       in ? &in->path_dirN : nullptr, in ? &in->accg : nullptr, in ? &in->accr : nullptr,
                                       &E.prev, &E.cur, &E.buf, &gain, &E.xpr, &E.path_dirN, &E.accg, &E.accr, &rt,
                                       (arma::u32_vec *)nullptr, eps, relaunch ? &E.origN : nullptr, relaunch ? &E.destN : nullptr);
        E.rt = rt[0];
        return E;
    };
    auto face = [](double x, bool flip) -> arma::mat
    {
        arma::mat m = diag_quad_x(200.0, x);
        if (flip)
        {
            m.row(0) = arma::rowvec{x, -200.0, -200.0, x, 200.0, -200.0, x, -200.0, 200.0};
            m.row(1) = arma::rowvec{x, 200.0, -200.0, x, 200.0, 200.0, x, -200.0, 200.0};
        }
        return m;
    };
    auto build = [&](const std::vector<double> &x, const std::vector<int> &mat, const std::vector<bool> &flip, arma::uvec &mtl_ind)
    {
        arma::mat mesh;
        mtl_ind.set_size(2 * x.size());
        for (size_t k = 0; k < x.size(); ++k)
        {
            mesh = k ? arma::join_cols(mesh, face(x[k], flip[k])) : face(x[k], flip[k]);
            mtl_ind(2 * k) = (arma::uword)mat[k], mtl_ind(2 * k + 1) = (arma::uword)mat[k];
        }
        return mesh;
    };
    auto onward = [](const Ev &E, arma::mat &orig, arma::mat &dest) // long enough to cross every face that is left
    {
        arma::mat d = E.destN - E.origN;
        d /= std::sqrt(d(0, 0) * d(0, 0) + d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2));
        orig = E.origN, dest = E.origN + (1.2 / std::abs(d(0, 0))) * d;
        return d(0, 0);
    };

    // Every path. One ray, both passes at every event, every ray followed until it has left the
    // structure, is dead, max_ev events old or below the floor. A ray in air between two bodies
    // is followed like any other.
    struct Out
    {
        std::complex<double> T = 0.0, R = 0.0, R0 = 0.0; // R0: the path that is reflected at the first event
        int nT = 0, nR = 0, n_cut = 0;
    };
    struct Node
    {
        arma::mat orig, dest;
        Ev P;
        bool have = false;
        double xp = 0.0, yp = 0.0, y0 = 0.0, Lin = 0.0, dirx = 1.0;
        std::complex<double> vv = 1.0;
        int depth = 0;
    };
    auto walk = [&](const Pal &pal, const arma::mat &mesh, const arma::uvec &mi, int tr, double f, double eps, double thd, int max_ev,
                    double floor_dB, double x_front, double x_back) -> Out
    {
        Out O;
        const double st = std::sin(thd * QPI / 180.0), floor_lin = std::pow(10.0, 0.1 * floor_dB), k = 2.0 * QPI * f / C0;
        std::vector<Node> stack(1);
        diag_ray_at(thd, stack[0].orig, stack[0].dest);
        while (!stack.empty())
        {
            Node Nn = stack.back();
            stack.pop_back();
            for (int itype : {3, tr})
            {
                const bool tx = itype != 3;
                Ev E = traced(pal, mesh, mi, itype, f, Nn.orig, Nn.dest, Nn.have ? &Nn.P : nullptr, eps, tx);
                if (E.nh == 0 || E.rt == 0)
                    continue;
                Node M;
                M.vv = Nn.vv * cpx(E.xpr, 0, 0);
                double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                M.y0 = Nn.depth ? Nn.y0 : fy;
                M.Lin = Nn.Lin + (Nn.depth ? std::sqrt((fx - Nn.xp) * (fx - Nn.xp) + (fy - Nn.yp) * (fy - Nn.yp)) : 0.0);
                M.xp = fx, M.yp = fy, M.depth = Nn.depth + 1;
                M.dirx = onward(E, M.orig, M.dest);
                if (mat_of(E.cur(0)) == 0 && ((M.dirx < 0.0 && fx < x_front + 1.0e-3) || (M.dirx > 0.0 && fx > x_back - 1.0e-3)))
                {
                    std::complex<double> F = M.vv * std::exp(-J * k * M.Lin) * std::exp(J * k * st * (fy - M.y0));
                    if (M.dirx > 0.0)
                        O.T += F, ++O.nT;
                    else
                    {
                        O.R += F, ++O.nR;
                        if (M.depth == 1)
                            O.R0 = F;
                    }
                    continue;
                }
                if (M.depth >= max_ev || std::norm(M.vv) < floor_lin)
                {
                    ++O.n_cut;
                    continue;
                }
                M.P = E, M.have = true;
                stack.push_back(M);
            }
        }
        return O;
    };

    // The reference: the scalar channel, admittance b / mu with b = sqrt(eps mu - sin^2), the
    // decaying root, layer phase k b d. It shares nothing with the library.
    struct Lay
    {
        std::complex<double> e, mu;
        double d;
    };
    struct RT
    {
        std::complex<double> r, t;
    };
    auto bof = [](std::complex<double> e, std::complex<double> mu, double s2)
    {
        std::complex<double> b = std::sqrt(e * mu - s2);
        return (std::imag(b) > 0.0) ? -b : b;
    };
    auto cm = [&](const std::vector<Lay> &ls, std::complex<double> ef, double s2, double k) -> RT // from air
    {
        std::complex<double> m11 = 1.0, m12 = 0.0, m21 = 0.0, m22 = 1.0;
        for (const Lay &l : ls)
        {
            std::complex<double> b = bof(l.e, l.mu, s2), y = b / l.mu, dl = k * b * l.d, c = std::cos(dl), s = std::sin(dl);
            std::complex<double> a12 = J * s / y, a21 = J * y * s;
            std::complex<double> n11 = m11 * c + m12 * a21, n12 = m11 * a12 + m12 * c, n21 = m21 * c + m22 * a21, n22 = m21 * a12 + m22 * c;
            m11 = n11, m12 = n12, m21 = n21, m22 = n22;
        }
        std::complex<double> hn = bof(one, one, s2), hf = bof(ef, one, s2), B = m11 + m12 * hf, Cc = m21 + m22 * hf;
        RT o;
        o.r = (hn * B - Cc) / (hn * B + Cc), o.t = 2.0 * hn / (hn * B + Cc);
        return o;
    };
    const std::vector<double> angles = {10.0, 25.0, 40.0, 55.0, 70.0, 85.0};

    auto mtl = diag_palette<double>();
    const int SHEET = 3;
    const double thk = 0.006, k1 = 2.0 * QPI * FRQ / C0;
    const std::complex<double> eS = diag_eps(SHEET, FRQ);

    // A reflected sum against its wall. Where the wall reflects 10 dB less than the path off its
    // first face, the two nearly cancel and the same distance reads larger in dB; such a row,
    // and a row that carries the clamp of section 5, is held against the first-face term.
    auto on_wall = [&](std::complex<double> R, std::complex<double> ref, std::complex<double> R0, bool clamp) -> bool
    {
        if (clamp || std::abs(ref) < 0.316 * std::abs(R0))
            return std::abs(R - ref) < 5.0e-3 * std::abs(R0);
        return std::abs(dB(R / ref)) < 5.0e-3 && std::abs(deg(R / ref)) < 0.05;
    };

    // ------------------------------------------------------------------------------------------
    // A. A wall coated on its near side through ray_mesh_interact alone
    // ------------------------------------------------------------------------------------------
    // What a caller without state gets. First the glued entry itself, both ports, against the
    // redo of ray_state_update. Then, on the lossless host with the refracting tracer, whose
    // segments are the wave, the paths of the wall from ray_mesh_interact's coefficients alone,
    // the host legs as exp(-j k0 n L) over the polyline: reflected off the coating; in, off the
    // far face, out; and the same with one more bounce. The three give the limit of the sequence.
    QD_DIAG("\n# diag: reflect port, a wall coated on its near side, ray_mesh_interact alone. air | coating 6 mm | host 100 | air.\n");
    QD_DIAG("# dR_own, dT_own: the glued entry of ray_mesh_interact against the redo of ray_state_update, complex difference.\n");
    QD_DIAG("# Host 6 and tracer 5 only, no state: R_dB, R_deg: the limit of the reflected paths against the matrix of\n");
    QD_DIAG("# coating | host with the coating in front of the mesh plane; T_dB: the limit of the transmitted paths against the\n");
    QD_DIAG("# matrix. R1/R0: dB\n");
    QD_DIAG("host,tracer,theta,dR_own,dT_own,R1/R0_dB,R_dB,R_deg,T_dB\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : angles)
            {
                double th = thd * QPI / 180.0, st = std::sin(th), s2 = st * st;
                arma::uvec mi;
                arma::mat mesh = build({0.0, gap, 0.10 + gap}, {SHEET, h, h}, {false, false, true}, mi), orig, dest;
                diag_ray_at(thd, orig, dest);
                Ev er = traced(mtl, mesh, mi, 3, FRQ, orig, dest, nullptr, 0.0, false), et = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 0.0, true);
                INFO("own composite, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((er.nh != 0 && et.nh != 0));
                double dR = std::abs(cpx(er.xpr0, 0, 0) - cpx(er.xpr, 0, 0)), dT = std::abs(cpx(et.xpr0, 0, 0) - cpx(et.xpr, 0, 0));
                CHECK(dR < 1.0e-12); // the two functions form one composite for an entry from air
                CHECK(dT < 1.0e-12);
                if (h != 6 || tr != 5)
                {
                    QD_DIAG("%d,%d,%.0f,%.3e,%.3e\n", h, tr, thd, dR, dT);
                    continue;
                }
                const double n_host = std::real(std::sqrt(diag_eps(h, FRQ)));
                auto stateless = [&](const std::vector<int> &seq, bool &ok) -> std::complex<double>
                {
                    arma::mat o, d;
                    diag_ray_at(thd, o, d);
                    std::complex<double> v = 1.0;
                    double Lin = 0.0, xp = 0.0, yp = 0.0, y0 = 0.0;
                    bool inside = false;
                    ok = true;
                    for (size_t q = 0; q < seq.size(); ++q)
                    {
                        Ev E = traced(mtl, mesh, mi, seq[q], FRQ, o, d, nullptr, 0.0, false, true);
                        if (E.nh == 0)
                        {
                            ok = false;
                            break;
                        }
                        v *= cpx(E.xpr0, 0, 0);
                        double fx = E.fbs(0, 0), fy = E.fbs(0, 1);
                        if (q == 0)
                            y0 = fy;
                        else if (inside)
                            Lin += std::sqrt((fx - xp) * (fx - xp) + (fy - yp) * (fy - yp));
                        if (seq[q] != 3)
                            inside = !inside;
                        xp = fx, yp = fy;
                        onward(E, o, d);
                    }
                    return v * std::exp(-J * k1 * n_host * Lin) * std::exp(J * k1 * st * (yp - y0));
                };
                bool ok0, ok1, ok2, ok3, ok4;
                std::complex<double> R0 = stateless({3}, ok0), R1 = stateless({tr, 3, tr}, ok1), R2 = stateless({tr, 3, 3, 3, tr}, ok2);
                std::complex<double> T1 = stateless({tr, tr}, ok3), T2 = stateless({tr, 3, 3, tr}, ok4);
                REQUIRE((ok0 && ok1 && ok2 && ok3 && ok4));
                RT W = cm({{eS, one, thk}, {diag_eps(h, FRQ), one, 0.10 + gap}}, one, s2, k1);
                std::complex<double> ref = W.r * std::exp(2.0 * J * k1 * std::cos(th) * thk); // the coating in front of the plane
                std::complex<double> Ra = R0 + R1 / (1.0 - R2 / R1), T = T1 / (1.0 - T2 / T1);
                QD_DIAG("%d,%d,%.0f,%.3e,%.3e,%+.2f,%+.6f,%+.4f,%+.6f\n", h, tr, thd, dR, dT, dB(R1 / R0), dB(Ra / ref), deg(Ra / ref), dB(T / W.t));
                CHECK(on_wall(Ra, ref, R0, false));
                CHECK(std::abs(dB(T / W.t)) < 5.0e-3);
            }

    // ------------------------------------------------------------------------------------------
    // B. A wall coated on its near side and a wall lined on both faces, every path
    // ------------------------------------------------------------------------------------------
    QD_DIAG("\n# diag: reflect port, a wall coated on its near side (SH) and lined on both faces (SHS), every path. coating 6 mm,\n");
    QD_DIAG("# host 100. eps 0 and 0.15, 30 events. R_dB, R_deg: the reflected sum against the matrix of the wall with the near\n");
    QD_DIAG("# coating in front of the mesh plane; R_ff: its distance from that wall over the path off the first coating.\n");
    QD_DIAG("# T_dB: the transmitted sum against the matrix; Td_deg: its phase with exp(-j k0 cos(theta) thk) per coating taken\n");
    QD_DIAG("# out. The rows at 85 deg on host 5 carry the clamp of section 5\n");
    QD_DIAG("wall,host,tracer,eps,theta,nT,nR,R_dB,R_deg,R_ff,T_dB,Td_deg\n");
    for (int lined = 0; lined < 2; ++lined)
        for (int h : {5, 6})
            for (int tr : {4, 5})
                for (double eps : {0.0, 0.15})
                    for (double thd : angles)
                    {
                        double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                        arma::uvec mi;
                        arma::mat mesh = lined ? build({0.0, gap, 0.10 + gap, 0.10 + 2.0 * gap}, {SHEET, h, h, SHEET}, {false, false, true, true}, mi)
                                               : build({0.0, gap, 0.10 + gap}, {SHEET, h, h}, {false, false, true}, mi);
                        arma::mat orig, dest;
                        diag_ray_at(thd, orig, dest);
                        Ev e0 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, nullptr, 1.5, true);
                        onward(e0, orig, dest);
                        Ev e1 = traced(mtl, mesh, mi, tr, FRQ, orig, dest, &e0, 1.5, true);
                        const double d_host = std::abs(e1.fbs(0, 0) - e0.origN(0, 0)); // the thickness the leg crossed
                        Out O = walk(mtl, mesh, mi, tr, FRQ, eps, thd, 30, -300.0, 0.0, 0.10);
                        std::vector<Lay> ls = {{eS, one, thk}, {diag_eps(h, FRQ), one, d_host}};
                        if (lined)
                            ls.push_back({eS, one, thk});
                        RT W = cm(ls, one, s2, k1);
                        std::complex<double> D1 = std::exp(-J * k1 * std::cos(th) * thk), ref = W.r / (D1 * D1);
                        const bool clamp = h == 5 && thd == 85.0;
                        INFO("lined " << lined << ", host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                        CHECK((O.nT >= 1 && O.nR >= 2));
                        QD_DIAG("%s,%d,%d,%.2f,%.0f,%d,%d,%+.6f,%+.4f,%.2e,%+.6f,%+.4f\n", lined ? "SHS" : "SH", h, tr, eps, thd, O.nT, O.nR, dB(O.R / ref),
                                deg(O.R / ref), std::abs(O.R - ref) / std::abs(O.R0), dB(O.T / W.t), deg(O.T * (lined ? D1 * D1 : D1) / W.t));
                        CHECK(on_wall(O.R, ref, O.R0, clamp));
                        if (!clamp)
                            CHECK(std::abs(dB(O.T / W.t)) < 5.0e-3);
                    }

    // ------------------------------------------------------------------------------------------
    // C. An absorber on a rigid wall, as a sheet and as a traced solid
    // ------------------------------------------------------------------------------------------
    // Mineral wool 50 mm, the two-column-pair fit of the phase-reference case, on a rigid wall
    // eps 2.401e-9 whose face stands at x = 0 in both meshes. As a sheet the wool is a face at
    // x = 0, co-located with the wall; as a traced solid it stands in front of the wall, from
    // x = -50 mm. The two are one wall. The reference is the layer on that wall with its front
    // face at x = -50 mm, so referenced to x = 0 it carries exp(+2j k0 cos(theta) W). Its mirror
    // wool | rigid is capped at one where the Fresnel value is above and left where it is below,
    // which is what a sheet does; a traced solid holds a closed face at one (section 5), so the
    // two rows where the Fresnel value is below one are recorded against each other.
    {
        const double W = 0.05, fscale = C0 / 343.0;
        Pal pal; // 1: wool, a solid; 2: the rigid wall; 3: wool, a sheet of 50 mm
        pal["a"] = {1.16, 2.401e-9, 1.16}, pal["b"] = {-0.04, 0.0, -0.04}, pal["c"] = {0.0038, 0.0, 0.0038}, pal["d"] = {1.46, 0.0, 1.46};
        pal["e"] = {1.66, 1.0, 1.66}, pal["f"] = {-0.19, 0.0, -0.19}, pal["g"] = {0.091, 0.0, 0.091}, pal["h"] = {0.13, 0.0, 0.13};
        pal["fRef"] = {0.874, 0.874, 0.874}, pal["thk"] = {0.0, 0.0, W};
        QD_DIAG("\n# diag: reflect port, an absorber on a rigid wall. air | mineral wool 50 mm | rigid, scalar, the wall face at x = 0.\n");
        QD_DIAG("# |R|_dB: the reflect port of the sheet. own: ray_mesh_interact's against the redo, complex difference. sheet:\n");
        QD_DIAG("# the sheet against the reference, complex distance. s/s_dB, s/s_deg: the sheet over the summed paths of the\n");
        QD_DIAG("# traced solid. owed_deg: 2 k0 cos(theta) W, not wrapped. r12_dB: the Fresnel reflection wool | rigid\n");
        QD_DIAG("tracer,f_Hz,theta,nR,|R|_dB,own,sheet,s/s_dB,s/s_deg,owed_deg,r12_dB\n");
        for (int tr : {4, 5})
            for (double fa : {250.0, 500.0, 1000.0, 2000.0, 4000.0})
                for (double thd : {20.0, 40.0, 60.0, 75.0})
                {
                    const double f1 = fa * fscale, fG = f1 * 1e-9, fr = fG / 0.874, th = thd * QPI / 180.0, k = 2.0 * QPI * f1 / C0;
                    const double st = std::sin(th), s2 = st * st;
                    std::complex<double> es(1.16 * std::pow(fr, -0.04), -17.98 * 0.0038 * std::pow(fr, 1.46) / fG);
                    std::complex<double> mus(1.66 * std::pow(fr, -0.19), -17.98 * 0.091 * std::pow(fr, 0.13) / fG);
                    arma::uvec miS, miB;
                    arma::mat meshS = build({0.0, gap, gap + 0.2}, {3, 2, 2}, {false, false, true}, miS);
                    arma::mat meshB = build({-W, 0.0, gap, gap + 0.2}, {1, 1, 2, 2}, {false, true, false, true}, miB), orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev er = traced(pal, meshS, miS, 3, f1, orig, dest, nullptr, 0.0, false);
                    Out O = walk(pal, meshB, miB, tr, f1, 0.0, thd, 30, -200.0, -W, gap + 0.2);
                    INFO("absorber, tracer " << tr << ", f " << fa << ", theta " << thd);
                    REQUIRE((er.nh != 0 && er.rt != 0 && O.nR >= 1));
                    std::complex<double> owed = std::exp(2.0 * J * k * std::cos(th) * W);
                    std::complex<double> A_sheet = cpx(er.xpr, 0, 0), A_solid = O.R * owed; // both referenced to x = 0
                    std::complex<double> y0 = bof(one, one, s2), b1 = bof(es, mus, s2), y1 = b1 / mus, y2 = bof(std::complex<double>(2.401e-9, 0.0), one, s2);
                    std::complex<double> r01 = (y0 - y1) / (y0 + y1), r12 = (y1 - y2) / (y1 + y2), p2 = std::exp(-2.0 * J * k * b1 * W);
                    const double r12_dB = dB(r12), own = std::abs(cpx(er.xpr0, 0, 0) - cpx(er.xpr, 0, 0));
                    if (std::abs(r12) > 1.0)
                        r12 /= std::abs(r12);
                    std::complex<double> ref = (r01 + r12 * p2) / (1.0 + r01 * r12 * p2) * owed;
                    QD_DIAG("%d,%.0f,%.0f,%d,%+.4f,%.3e,%.3e,%+.6f,%+.4f,%.3f,%+.3f\n", tr, fa, thd, O.nR, dB(A_sheet), own, std::abs(A_sheet / ref - 1.0),
                            dB(A_sheet / A_solid), deg(A_sheet / A_solid), 2.0 * k * std::cos(th) * W * 180.0 / QPI, r12_dB);
                    CHECK(own < 1.0e-12); // a glued entry under a closed outer pair, both functions
                    CHECK(std::abs(A_sheet / ref - 1.0) < 1.0e-9);
                    if (r12_dB >= 0.0) // the sheet and the traced solid are one wall
                    {
                        CHECK(std::abs(dB(A_sheet / A_solid)) < 5.0e-3);
                        CHECK(std::abs(deg(A_sheet / A_solid)) < 0.05);
                    }
                }
    }

    // ------------------------------------------------------------------------------------------
    // D. A sheet in front of a wall across an air gap
    // ------------------------------------------------------------------------------------------
    // air | sheet 6 mm | air 30 mm | host 100 | air, the sheet's face at x = 0 and the wall's at
    // x = 30 mm. The layer of the sheet occupies 6 mm of that air, [-u, 6 mm - u]. Three walls,
    // each referenced to x = 0: u = 0 (behind the face, in the gap), 3 mm (centred) and 6 mm (in
    // front). The mesh plane is the mid-plane, so the paths sum to the centred wall on both
    // ports, and the other two are recorded to show they are told apart.
    QD_DIAG("\n# diag: reflect port, a sheet in front of a wall across an air gap. air | sheet 6 mm | air 30 mm | host 100 | air,\n");
    QD_DIAG("# eps 0, every path, floor -120 dB. The summed paths against the wall whose layer lies at [-u, thk - u]: uh centred,\n");
    QD_DIAG("# asserted; u0 behind the face and ut in front, recorded. dB and deg. cut: rays stopped at the floor\n");
    QD_DIAG("host,tracer,theta,nT,nR,cut,R_uh,Rdeg_uh,T_uh,Tdeg_uh,R_u0,Rdeg_u0,T_u0,R_ut,Rdeg_ut,T_ut\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : {10.0, 40.0, 70.0})
            {
                const double g = 0.03, th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th), ck = k1 * std::cos(th);
                arma::uvec mi;
                arma::mat mesh = build({0.0, g, g + 0.10}, {SHEET, h, h}, {false, false, true}, mi);
                Out L = walk(mtl, mesh, mi, tr, FRQ, 0.0, thd, 200, -120.0, 0.0, g + 0.10);
                std::complex<double> rr[3], tt[3];
                const double uu[3] = {0.0, 0.5 * thk, thk};
                for (int q = 0; q < 3; ++q)
                {
                    RT Wu = cm({{eS, one, thk}, {one, one, g - thk + uu[q]}, {diag_eps(h, FRQ), one, 0.10}}, one, s2, k1);
                    rr[q] = Wu.r * std::exp(2.0 * J * ck * uu[q]), tt[q] = Wu.t * std::exp(J * ck * uu[q]);
                }
                INFO("air gap, host " << h << ", tracer " << tr << ", theta " << thd);
                CHECK((L.nT >= 1 && L.nR >= 2));
                QD_DIAG("%d,%d,%.0f,%d,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.4f,%+.3f,%+.4f\n", h, tr, thd, L.nT, L.nR, L.n_cut, dB(L.R / rr[1]),
                        deg(L.R / rr[1]), dB(L.T / tt[1]), deg(L.T / tt[1]), dB(L.R / rr[0]), deg(L.R / rr[0]), dB(L.T / tt[0]), dB(L.R / rr[2]),
                        deg(L.R / rr[2]), dB(L.T / tt[2]));
                CHECK(std::abs(dB(L.R / rr[1])) < 5.0e-3);
                CHECK(std::abs(deg(L.R / rr[1])) < 0.05);
                CHECK(std::abs(dB(L.T / tt[1])) < 5.0e-3);
                CHECK(std::abs(deg(L.T / tt[1])) < 0.05);
                CHECK(std::abs(L.R / rr[0] - 1.0) > 1.0e-2); // the three walls are told apart
                CHECK(std::abs(L.R / rr[2] - 1.0) > 1.0e-2);
            }

    // ------------------------------------------------------------------------------------------
    // E. A sheet inside a wall
    // ------------------------------------------------------------------------------------------
    // air | host 50 | sheet 6 mm | host 50 | air. The sheet displaces 3 mm of host on either side
    // of its face, so the wall is host 47 | sheet | host 47, on both ports, whether every series
    // is summed, every bounce followed, or the threshold falls between. w44: the same sums
    // against host 50 | sheet | host 44, recorded.
    QD_DIAG("\n# diag: reflect port, a sheet inside a wall. air | host 50 | sheet 6 mm | host 50 | air, every path, floor -100 dB;\n");
    QD_DIAG("# eps 1.5 with 20 events, eps 0 and 0.15 with 60. The summed paths against host 47 | sheet | host 47, dB and deg;\n");
    QD_DIAG("# w44: against host 50 | sheet | host 44, dB\n");
    QD_DIAG("host,tracer,eps,theta,nT,nR,cut,R_dB,R_deg,T_dB,T_deg,R_w44,T_w44\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double eps : {1.5, 0.0, 0.15})
                for (double thd : {10.0, 40.0, 60.0})
                {
                    const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    const std::complex<double> eH = diag_eps(h, FRQ);
                    arma::uvec mi;
                    arma::mat mesh = build({0.0, 0.05, 0.10}, {h, SHEET, h}, {false, false, true}, mi);
                    Out L = walk(mtl, mesh, mi, tr, FRQ, eps, thd, eps > 1.0 ? 20 : 60, -100.0, 0.0, 0.10);
                    RT W = cm({{eH, one, 0.047}, {eS, one, thk}, {eH, one, 0.047}}, one, s2, k1), W44 = cm({{eH, one, 0.05}, {eS, one, thk}, {eH, one, 0.044}}, one, s2, k1);
                    INFO("sheet inside, host " << h << ", tracer " << tr << ", eps " << eps << ", theta " << thd);
                    CHECK((L.nT >= 1 && L.nR >= 2));
                    QD_DIAG("%d,%d,%.2f,%.0f,%d,%d,%d,%+.4f,%+.3f,%+.4f,%+.3f,%+.4f,%+.4f\n", h, tr, eps, thd, L.nT, L.nR, L.n_cut, dB(L.R / W.r), deg(L.R / W.r),
                            dB(L.T / W.t), deg(L.T / W.t), dB(L.R / W44.r), dB(L.T / W44.t));
                    CHECK(std::abs(dB(L.R / W.r)) < 1.0e-2);
                    CHECK(std::abs(deg(L.R / W.r)) < 0.05);
                    CHECK(std::abs(dB(L.T / W.t)) < 1.0e-2);
                    CHECK(std::abs(deg(L.T / W.t)) < 0.05);
                }

    // The same port without a reference: a wall with the sheet off its middle, host 30 | sheet |
    // host 70, and its mirror image, host 70 | sheet | host 30. The transmission through a layered
    // wall between two half-spaces of one medium is the same from either side.
    QD_DIAG("\n# diag: reflect port, a sheet off the middle of a wall, from either side. air | host 30 | sheet 6 mm | host 70 | air\n");
    QD_DIAG("# over its mirror image, the transmitted sums, eps 1.5, 20 events, floor -100 dB, dB and deg\n");
    QD_DIAG("host,tracer,theta,T_dB,T_deg\n");
    for (int h : {5, 6})
        for (int tr : {4, 5})
            for (double thd : {10.0, 40.0, 60.0})
            {
                arma::uvec mi, mj;
                arma::mat m37 = build({0.0, 0.03, 0.10}, {h, SHEET, h}, {false, false, true}, mi), m73 = build({0.0, 0.07, 0.10}, {h, SHEET, h}, {false, false, true}, mj);
                Out A = walk(mtl, m37, mi, tr, FRQ, 1.5, thd, 20, -100.0, 0.0, 0.10), B = walk(mtl, m73, mj, tr, FRQ, 1.5, thd, 20, -100.0, 0.0, 0.10);
                INFO("either side, host " << h << ", tracer " << tr << ", theta " << thd);
                REQUIRE((A.nT >= 1 && B.nT >= 1));
                QD_DIAG("%d,%d,%.0f,%+.4f,%+.3f\n", h, tr, thd, dB(A.T / B.T), deg(A.T / B.T));
                CHECK(std::abs(dB(A.T / B.T)) < 1.0e-3);
                CHECK(std::abs(deg(A.T / B.T)) < 0.01);
            }

    // ------------------------------------------------------------------------------------------
    // F. A sheet and a traced solid of one material and thickness, with att
    // ------------------------------------------------------------------------------------------
    // 20 mm of eps 5.24 - 0.63j and of eps 5.24, in air, with att 0, 3 and 10 dB. The sheet is
    // one face and one event, its layer centred on the face; the solid is two faces 20 mm apart
    // with its series summed, its front face on the mesh plane. So the sheet's reflection is the
    // solid's times exp(+j k0 cos(theta) d). once: the Airy reflection with the echo, the part
    // that entered the layer, multiplied by sqrt(G). A traced leg is short by the relaunch
    // offset, so the two representations meet at 1e-3 dB and the sheet meets the form at 1e-9.
    QD_DIAG("\n# diag: reflect port, a sheet against a traced solid with att. 20 mm in air, scalar, 1 GHz. E/r: the echo over the\n");
    QD_DIAG("# front-face term of the Airy reflection, dB. sheet: the sheet's reflect port against the form whose echo carries\n");
    QD_DIAG("# sqrt(G), at the mid-plane, complex distance. solid_dB, solid_deg: the summed reflection of the solid at eps 0\n");
    QD_DIAG("# against that form at its front face. s/s: the sheet over the solid, both at the solid's front face. own:\n");
    QD_DIAG("# ray_mesh_interact's reflect port against the redo. T_s/s: the transmit ports, sheet over solid, dB. fol: the\n");
    QD_DIAG("# solid's reflected sum with every bounce followed over the summed one. none_dB: the sheet against the Airy\n");
    QD_DIAG("# reflection without the gain, which is what att is worth on the port\n");
    QD_DIAG("eps_im,att,tracer,theta,E/r_dB,sheet,solid_dB,solid_deg,s/s_dB,s/s_deg,own,T_s/s_dB,fol_dB,none_dB\n");
    for (double ei : {0.63, 0.0})
        for (double att : {0.0, 3.0, 10.0})
        {
            const double d = 0.02;
            LumpMtl solid, sheet;
            solid.a = 5.24, solid.ei = ei, solid.att = att;
            sheet = solid, sheet.thk = d;
            Pal pal = lump_palette({solid, sheet});
            const std::complex<double> e = lump_eps(solid, FRQ);
            const double sg = std::pow(10.0, -0.05 * att);
            for (int tr : {4, 5})
                for (double thd : {0.0, 20.0, 40.0, 60.0, 75.0, 85.0})
                {
                    const double th = thd * QPI / 180.0, s2 = std::sin(th) * std::sin(th);
                    arma::uvec miS, miB;
                    arma::mat meshS = build({0.0}, {2}, {false}, miS), meshB = build({0.0, d}, {1, 1}, {false, true}, miB), orig, dest;
                    diag_ray_at(thd, orig, dest);
                    Ev er = traced(pal, meshS, miS, 3, FRQ, orig, dest, nullptr, 0.0, false), et = traced(pal, meshS, miS, tr, FRQ, orig, dest, nullptr, 0.0, true);
                    Out O = walk(pal, meshB, miB, tr, FRQ, 0.0, thd, 30, -300.0, 0.0, d), Fo = walk(pal, meshB, miB, tr, FRQ, 1.5, thd, 60, -300.0, 0.0, d);
                    INFO("att, eps_im " << ei << ", att " << att << ", tracer " << tr << ", theta " << thd);
                    REQUIRE((er.nh != 0 && er.rt != 0 && et.rt != 0 && O.nR >= 2 && O.nT >= 1));
                    std::complex<double> y0 = bof(one, one, s2), y1 = bof(e, one, s2), r01 = (y0 - y1) / (y0 + y1), p2 = std::exp(-2.0 * J * k1 * y1 * d);
                    std::complex<double> echo = -(1.0 - r01 * r01) * r01 * p2 / (1.0 - r01 * r01 * p2), none = r01 + echo, once = r01 + sg * echo;
                    std::complex<double> half = std::exp(J * k1 * std::cos(th) * d), Rs = cpx(er.xpr, 0, 0), own = cpx(er.xpr0, 0, 0);
                    QD_DIAG("%.2f,%.0f,%d,%.0f,%+.2f,%.3e,%+.6f,%+.4f,%+.6f,%+.4f,%.3e,%+.6f,%+.6f,%+.4f\n", ei, att, tr, thd, dB(echo / r01),
                            std::abs(Rs / (once * half) - 1.0), dB(O.R / once), deg(O.R / once), dB(Rs / (O.R * half)), deg(Rs / (O.R * half)),
                            std::abs(own - Rs), dB(cpx(et.xpr, 0, 0) / O.T), dB(Fo.R / O.R), dB(Rs / none));
                    CHECK(std::abs(Rs / (once * half) - 1.0) < 1.0e-9);
                    CHECK(std::abs(own - Rs) < 1.0e-12);
                    CHECK(std::abs(dB(O.R / once)) < 1.0e-3); // the traced solid pays the gain once
                    CHECK(std::abs(dB(Rs / (O.R * half))) < 1.0e-3); // and the two representations are one wall
                    CHECK(std::abs(deg(Rs / (O.R * half))) < 0.05);
                    CHECK(std::abs(dB(cpx(et.xpr, 0, 0) / O.T)) < 5.0e-3); // a traced leg is short by the relaunch offset
                }
        }
}
