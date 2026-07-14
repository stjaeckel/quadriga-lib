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
//   Full ray set  [n_ray]  : orig, dest, no_interact, prev_in, cur_in, buf_in, path_dir_prev, acc_dist_in
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
    arma::Mat<dtype> acc_dist_in;             // full  [n_ray, 2]  (col 0 refracted, col 1 geometric)

    arma::Col<short> prev_out, cur_out, buf_out; // compact [n_rayN]
    arma::Col<dtype> gain;                       // compact [n_rayN]
    arma::Mat<dtype> xprmat;                     // compact [nXPR, n_rayN]
    arma::Mat<dtype> path_dirN;                  // compact [n_rayN, 3]
    arma::Mat<dtype> acc_dist_out;               // compact [n_rayN, 2]
    std::vector<uint8_t> resolved_type;          // compact [n_rayN]
    arma::u32_vec ray_ind;                       // compact [n_rayN]

    arma::Col<dtype> g_in; // snapshots taken by run()
    arma::Mat<dtype> x_in;

    bool has_orig = true, has_dest = true, has_fbs = true, has_sbs = true, has_ni = true;
    bool has_angle = true, has_normals = true, has_otype = true, has_mtl = true;
    bool has_m1 = true, has_m2 = true;
    bool has_prev_in = true, has_cur_in = true, has_buf_in = true;
    bool has_path_dir = false, has_acc_in = false;
    bool has_prev_out = true, has_cur_out = true, has_buf_out = true;
    bool has_gain = true, has_xprmat = true;
    bool has_path_dirN = false, has_acc_out = false, has_rtype = true, has_ray_ind = false;

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
            has_acc_in ? &acc_dist_in : nullptr,
            has_prev_out ? &prev_out : nullptr,
            has_cur_out ? &cur_out : nullptr,
            has_buf_out ? &buf_out : nullptr,
            has_gain ? &gain : nullptr,
            has_xprmat ? &xprmat : nullptr,
            has_path_dirN ? &path_dirN : nullptr,
            has_acc_out ? &acc_dist_out : nullptr,
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

TEST_CASE("ray_state_update - mass-law material feeds |phi| from medium_gain")
{
    // For a mass-law slab |phi|^2 must equal the implementation's own medium_gain(L), not an
    // eta-only loss. The mass column has no public closed form, so the test measures
    // medium_gain through a cavity-exit magnitude and then checks S against the closed form.
    arma::mat M = {{4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    mtl_matrix_to_map<double>(M, ind, mtl);
    mtl["m"] = {10.0}; // mass-law slope, dB/decade

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
        C.has_acc_out = true;
        C.acc_dist_out.zeros(1, 2);
        C.run();
        CHECK(std::abs((double)C.acc_dist_out(0, 0) - dfd) < 1e-9); // refracted (post-exit leg is unrefracted)
        CHECK(std::abs((double)C.acc_dist_out(0, 1) - dfd) < 1e-9); // geometric
    }
    SECTION("a non-terminal entry restarts the in-medium distance at zero")
    {
        auto C = make1<double>(mtl, 4, OT_ENTRY, 2, 4, 4, 0, 0, 0, 1.5, 2.0, 0.5, QPI / 2.0, 1.5);
        C.has_acc_out = true;
        C.acc_dist_out.zeros(1, 2);
        C.run();
        CHECK(std::abs((double)C.acc_dist_out(0, 0)) < 1e-9);
        CHECK(std::abs((double)C.acc_dist_out(0, 1)) < 1e-9);
    }
    SECTION("acc_dist_in threads into the exit medium loss")
    {
        double dof = 0.7, preload = 0.5;
        double L = dof + preload;
        auto C = make1<double>(mtl, 4, OT_EXIT, 1, 6, 0, 0, 6, 0, dof, 2.0, 0.5, QPI / 2.0, 1.5); // ABSORB
        C.has_acc_in = true;
        C.acc_dist_in.zeros(1, 2);
        C.acc_dist_in(0, 0) = preload; // refracted
        C.acc_dist_in(0, 1) = preload; // geometric
        C.run();
        CHECK(std::abs(std::norm(vvf(C)) - medg(6, L)) < 1e-9); // |factor|^2 = medium_gain(acc_in + d(orig,fbs))
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