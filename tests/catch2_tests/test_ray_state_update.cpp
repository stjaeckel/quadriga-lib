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
//  - State: three signed-short words (mat = w & 0x7FFF, resolved/non-parallel flag = w & 0x8000).
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
static inline int mat_of(short w) { return (int)(unsigned short)qd::bits<short>(w).tail(15); }
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
    // State words: bit 15 flag, bits 0..14 material. Never arithmetic negation.
    CHECK(enc(0, true) == (short)-32768); // resolved air word is 0x8000, not -0
    CHECK(mat_of(enc(0, true)) == 0);
    CHECK(flag_of(enc(0, true)));
    CHECK(enc(5, true) != (short)-5); // 0x8005, not the arithmetic negative
    CHECK(mat_of(enc(5, true)) == 5);
    CHECK(flag_of(enc(5, true)));
    CHECK(mat_of(enc(5, false)) == 5);
    CHECK(!flag_of(enc(5, false)));
    CHECK(mat_of((short)0x7FFF) == 0x7FFF);

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
        check_state(C, 0, 4, 0); // no resolved flag
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
    check_rtype(C, RT_EXIT);                      // not RT_EXIT_SLAB

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
        check_state(R, 0, 5, 0); // re-emitted, no resolved flag
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
        check_state(C, 0, 1, 0);
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
              (dtype)1.50, (dtype)1.50, (dtype)1.50, (dtype)1.50};
    m["b"] = std::vector<dtype>(8, (dtype)0.0);
    m["c"] = {(dtype)(0.63 / 17.98), (dtype)0.0, (dtype)(0.63 / 17.98), (dtype)(1.20 / 17.98),
              (dtype)(0.40 / 17.98), (dtype)0.0, (dtype)(0.40 / 17.98), (dtype)0.0};
    m["d"] = std::vector<dtype>(8, (dtype)0.0);
    m["att"] = std::vector<dtype>(8, (dtype)0.0);
    m["attB"] = std::vector<dtype>(8, (dtype)0.0);
    m["alpha"] = std::vector<dtype>(8, (dtype)0.0);
    m["alphaB"] = std::vector<dtype>(8, (dtype)0.0);
    m["fRef"] = std::vector<dtype>(8, (dtype)1.0);
    m["thk"] = {(dtype)0.0, (dtype)0.0, (dtype)0.006, (dtype)0.0,
                (dtype)0.0, (dtype)0.0, (dtype)0.006, (dtype)0.006};
    return m;
}

// Complex relative permittivity of a diag_palette entry at f_hz
static inline std::complex<double> diag_eps(int idx, double f_hz)
{
    static const double a[8] = {1.0, 5.24, 5.24, 5.24, 9.0, 1.50, 1.50, 1.50};
    static const double c[8] = {0.0, 0.63 / 17.98, 0.0, 0.63 / 17.98, 1.20 / 17.98,
                                0.40 / 17.98, 0.0, 0.40 / 17.98};
    if (idx < 1)
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
static inline double diag_traced_ct(std::complex<double> eps_slab, double theta_ext_rad)
{
    std::complex<double> eps_r = 1.0 / eps_slab; // entry medium is air
    double s = std::sin(theta_ext_rad);
    std::complex<double> ct2 = std::sqrt(1.0 - eps_r * s * s);
    double eta = std::sqrt(std::abs(eps_r)), r = std::real(ct2);
    return r / std::sqrt((eta * s) * (eta * s) + r * r);
}

// The exact complex in-slab cosine the projection above stands for
static inline std::complex<double> diag_ct_exact(std::complex<double> eps_slab, double theta_ext_rad)
{
    double s = std::sin(theta_ext_rad);
    return std::sqrt(1.0 - (1.0 / eps_slab) * s * s);
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
                                   double &p_te, double &p_tm)
{
    const double k0 = 2.0 * QPI * f_hz / C0;
    const std::complex<double> z1(1.0, 0.0), z2 = std::sqrt(eps_s), n = std::sqrt(eps_s);
    const double c1 = std::cos(theta_n_rad), s1 = std::sin(theta_n_rad);
    const std::complex<double> c2 = std::sqrt(1.0 - (1.0 / eps_s) * s1 * s1);
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
static inline std::complex<double> diag_medium_vbs(std::complex<double> es, double dist_refract,
                                                   double ct_traced, double dist_geo,
                                                   std::complex<double> cs, double f_hz)
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

    return std::exp(std::complex<double>(0.0, -1.0) * k0 * (n * cs * d + q * s)) *
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
                                     double theta_n_rad, double f_hz)
{
    double p_te = 0.0, p_tm = 0.0;
    diag_slab_p_pol(eps_s, thk, theta_n_rad, f_hz, p_te, p_tm);
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
                carrier = T_te * diag_medium_vbs(es, dr, ct_tr, L, ct_ex, f_hz);
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
    // The first run pinned the applied form exactly: with no path_dir supplied it is
    // -k0 * (n - 1) * L * cos^2(theta_fbs), matching to 0.000 deg at every angle. Both candidate
    // conventions are printed against it. The lossless material keeps the in-slab angle real, which
    // isolates the convention from every complex-angle effect. eps = 1.5 for the resolve gate means
    // the series never fires, so only the medium factor acts.
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

            std::complex<double> M = diag_medium_vbs(es, dr, ct, L, cs, f_hz);

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