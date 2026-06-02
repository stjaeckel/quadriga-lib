// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx; // so the bare `Approx(...)` calls compile unchanged

#include "quadriga_tools.hpp"

#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>
#include <string>

// Acoustic-domain interaction tests for the material model extensions.
//
// The acoustic interpretation reuses the electromagnetic ray tracer by mapping every acoustic
// frequency to an equivalent radio frequency, f_radio = f_acoustic * (c_light / c_sound), so that
// the acoustic wavelength equals the radio wavelength the EM Fresnel/medium math expects. All
// material reference frequencies (fRef) are set on the same mapped scale, so f/fRef collapses to
// the plain acoustic frequency ratio and the expected values stay closed-form. Tests run with the
// scalar interaction types (3 = reflection, 4 = transmission), which is the acoustic mode.
//
// What is tested, and why it matters in practice:
//
//   1. Frequency mapping round-trip — Hz -> GHz and back. Underpins every acoustic call: the ray
//      tracer is frequency-agnostic, so an off-by-a-constant mapping silently shifts every
//      resonance and coincidence feature. Asserts c_light/c_sound and the inverse are consistent.
//
//   2. Mass-law transmission (m) — partition isolation rises with frequency and thickness
//      (+6 dB/octave and +6 dB per doubling at m = 20). This is the dominant slope of real walls,
//      floors and partitions. Exercised via a two-call slab traverse (enter, then exit) so the
//      in-medium distance feeding the mass term is exactly the slab thickness.
//
//   3. Coincidence dip (coiF/coiQ/coiA) — thin stiff panels (glass, drywall, metal sheet) lose
//      isolation in a narrow band around the coincidence frequency. Modeled as a Lorentzian added
//      to the lumped per-entry transmission loss; a negative coiA is a transmission dip.
//
//   4. Permittivity resonance (resF/resQ/resS) — Helmholtz / membrane / micro-perforated absorbers
//      show an absorption peak at their tuning frequency. Modeled as a Lorentz pole in the Fresnel
//      permittivity, so the absorption peak appears through the reflection branch as 1 - |R|^2.
//
//   5. EM <-> acoustic convergence at the edges. The acoustic terms must be exact no-ops where they
//      are inactive, so the acoustic path reduces to the plain scalar-EM path:
//        (a) with m = res* = coi* = 0 (terms structurally absent), and
//        (b) far from resF / coiF (terms numerically negligible).
//      Checked for both ray_mesh_interact and calc_diffraction_gain.
//
//   6. Cross-method pass-through calibration — ray_mesh_interact vs calc_diffraction_gain. A wave
//      crossing a partition (wall, floor, panel) pays entry + exit interface losses plus the
//      in-medium traversal. ray_mesh_interact computes this for a single ray (two calls); calc_-
//      diffraction_gain computes it for a bundle of rays on the Fresnel arc. When a large, thin
//      slab fully obstructs the bundle there is no edge diffraction, so the diffraction gain must
//      reduce to the plane-wave transmission a single ray sees. This is the calibration anchor
//      between the two engines, and it is the test that catches any disagreement on the single
//      dense->light crossing every slab has — EM agrees regardless (both override the exit),
//      scalar agrees only if transition_gain_linear's guard matches ray_mesh_interact's gating.

// Speed of light and a fixed speed of sound used for the acoustic frequency mapping.
// c_sound = 342.77 m/s reproduces the format_materials.md constant (c_light/c_sound ~= 874636),
// i.e. 1 kHz acoustic <-> 0.875 GHz radio.
static constexpr double C_LIGHT = 299792458.0;
static constexpr double C_SOUND = 342.77;
static constexpr double AC2RF = C_LIGHT / C_SOUND; // acoustic Hz -> radio Hz

// Acoustic Hz -> mapped radio Hz, and the inverse.
static inline double ac2rf(double f_acoustic_Hz) { return f_acoustic_Hz * AC2RF; }
static inline double rf2ac(double f_radio_Hz) { return f_radio_Hz / AC2RF; }

// Build a one-material (mtl_ind, mtl_prop) pair from a list of named columns, broadcast over all
// 12 cube faces. Columns not listed are simply absent from the map (consumer applies the default),
// which mirrors how obj_file_read emits a schema-blind table.
static inline void single_material(const std::vector<std::pair<std::string, double>> &cols,
                                   arma::uvec &mtl_ind,
                                   std::unordered_map<std::string, std::vector<double>> &mtl_prop)
{
    mtl_ind.set_size(12);
    mtl_ind.zeros(); // all faces share material 0
    mtl_prop.clear();
    for (const auto &kv : cols)
        mtl_prop[kv.first] = std::vector<double>(1, kv.second);
}

// Reference Lorentz permittivity pole, mirroring eta_resonance() in the library.
static inline std::complex<double> ref_eta_resonance(double resF, double resQ, double resS, double fGHz)
{
    if (resF <= 0.0 || resQ <= 0.0 || resS == 0.0)
        return std::complex<double>(0.0, 0.0);
    double resF2 = resF * resF;
    std::complex<double> denom(resF2 - fGHz * fGHz, (resF / resQ) * fGHz);
    return (resS * resF2) / denom;
}

// Reference scalar (TE-only) reflected-power coefficient for a single interface, energy-conserving.
// eta1 = incoming medium, eta2 = outgoing medium, incidence angle measured from the surface (P.2040).
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

// Standard cube fixture (side 2, centered at origin), outward normals.
static inline arma::mat make_cube()
{
    return {{-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0},      //  1 Top NorthEast
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
}

// =====================================================================================
// 1. Frequency mapping round-trip
// =====================================================================================
TEST_CASE("Acoustic - Frequency mapping round-trip")
{
    // 1 kHz acoustic maps to ~0.875 GHz radio (the format_materials.md reference point).
    double f_ac = 1.0e3;
    double f_rf = ac2rf(f_ac);
    CHECK(f_rf / 1.0e9 == Approx(0.875).epsilon(1e-3)); // ~0.875 GHz

    // Hz -> GHz -> Hz must be exact to floating-point round-off.
    for (double f : {20.0, 125.0, 500.0, 1000.0, 2000.0, 8000.0, 16000.0})
    {
        CHECK(rf2ac(ac2rf(f)) == Approx(f).epsilon(1e-12));
        CHECK(ac2rf(rf2ac(f)) == Approx(f).epsilon(1e-12));
    }

    // The mapping constant itself.
    CHECK(AC2RF == Approx(874636.0).epsilon(1e-4));
}

// =====================================================================================
// 2. Mass-law transmission (m) — two-call slab traverse, ray_mesh_interact
// =====================================================================================
//
// A 2 m thick slab (the cube) at normal incidence. The wave enters at the west face and exits at
// the east face; the in-medium distance for the mass term is the 2 m slab traversal applied on the
// exit call (ray starts inside). We use a transparent-but-rigid-ish material so the only frequency-
// and-thickness dependent loss is the mass term, then verify the +6 dB/octave and +6 dB/doubling
// slopes that define the acoustic mass law.
TEST_CASE("Acoustic - Mass-law slab traverse (ray_mesh_interact)")
{
    arma::mat cube = make_cube();
    double fRef_rf = ac2rf(1000.0); // reference at 1 kHz acoustic

    // Material: eps = 1 (no Fresnel loss), sigma = 0, alpha = 0, m = 20 (canonical mass law),
    // fRef on the mapped scale. With eps = 1 there is no interface reflection, so the entire
    // transmitted gain is the mass-law in-medium loss over the traversed distance.
    auto run_slab = [&](double f_ac, double m_slope, double &gain_enter, double &gain_exit, double &in_medium_dist)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        single_material({{"a", 1.0}, {"m", m_slope}, {"fRef", fRef_rf / 1.0e9}}, mtl_ind, mtl);
        double f_rf = ac2rf(f_ac);

        // --- Call 1: enter the slab at the west face (outside -> inside). No in-medium path yet.
        arma::mat orig = {{-10.0, 0.0, 0.5}};
        arma::mat dest = {{10.0, 0.0, 0.5}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat origN, destN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact<double>(4, f_rf, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl,
                                                &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr,
                                                &origN, &destN, &gainN);
        gain_enter = gainN(0);

        // --- Call 2: start from a point clearly INSIDE the cube and travel out through the east
        // face. The ray starts inside, so ray_mesh_interact applies the in-medium loss over the
        // traversal to the FBS. Use an explicit interior origin (not the chained origN, whose
        // ray_offset step direction is convention-dependent) so the geometry is unambiguous.
        arma::mat orig2 = {{-0.5, 0.0, 0.5}}; // inside the cube (half-width 1)
        arma::mat dest2 = {{10.0, 0.0, 0.5}}; // exits east face at x = +1
        arma::mat fbs2, sbs2;
        arma::u32_vec fbs_ind2, sbs_ind2;
        quadriga_lib::ray_triangle_intersect(&orig2, &dest2, &cube, &fbs2, &sbs2, NULL, &fbs_ind2, &sbs_ind2);

        arma::mat origN2, destN2;
        arma::vec gainN2, orig_lengthN2;
        quadriga_lib::ray_mesh_interact<double>(4, f_rf, &orig2, &dest2, &fbs2, &sbs2, &cube, &mtl_ind, &mtl,
                                                &fbs_ind2, &sbs_ind2, nullptr, nullptr, nullptr,
                                                &origN2, &destN2, &gainN2, nullptr, nullptr, nullptr,
                                                &orig_lengthN2);
        gain_exit = gainN2(0);

        // The loss was applied over OF_length; orig_lengthN returns OF_length + ray_offset.
        in_medium_dist = orig_lengthN2(0) - 0.001; // ray_offset = 1 mm
    };

    // Reference mass-law dB over a 2 m path: max(0, m * log10((f/fRef) * dist)).
    auto ref_mass_gain = [&](double f_ac, double m_slope, double dist)
    {
        double f_rel = ac2rf(f_ac) / fRef_rf; // == f_ac / 1000
        double arg = f_rel * dist;
        double dB = (arg > 1.0) ? m_slope * std::log10(arg) : 0.0;
        return std::pow(10.0, -0.1 * dB);
    };

    double dist = 2.0;

    // Single frequency check at 4 kHz: (f/fRef)*dist = 4 * 2 = 8 -> m*log10(8).
    double g_enter, g_exit, Lm;
    run_slab(4000.0, 20.0, g_enter, g_exit, Lm);
    CHECK(std::abs(g_enter - 1.0) < 1e-6);
    CHECK(std::abs(g_exit - ref_mass_gain(4000.0, 20.0, Lm)) < 1e-5); // Lm ≈ 1.999

    // +6 dB/octave: doubling the frequency adds m*log10(2) dB = 6.02 dB at m = 20.
    double gA_enter, gA_exit, gB_enter, gB_exit, LmA, LmB;
    run_slab(2000.0, 20.0, gA_enter, gA_exit, LmA);
    run_slab(4000.0, 20.0, gB_enter, gB_exit, LmB);
    double dB_octave = -10.0 * std::log10(gB_exit / gA_exit);
    CHECK(std::abs(dB_octave - 6.0206) < 1e-3);

    // Mass-law term is exactly zero when (f/fRef)*dist <= 1 (clamp). At 250 Hz: 0.25 * 2 = 0.5 < 1.
    double gZ_enter, gZ_exit;
    run_slab(250.0, 20.0, gZ_enter, gZ_exit, Lm);
    CHECK(gZ_exit == Approx(1.0).epsilon(1e-6));
}

// =====================================================================================
// 3. Coincidence dip (coiF/coiQ/coiA) — interface transmission, calc_diffraction_gain
// =====================================================================================
//
// A thin stiff panel loses transmission isolation in a narrow band around the coincidence
// frequency. We model only the lumped interface transmission loss (att + coincidence): eps = 1, so
// there is no Fresnel reflection and the only transmission loss is the per-entry interface term.
// The single-interface o->i diffraction path (lod = 0) isolates exactly one application of it.
TEST_CASE("Acoustic - Coincidence dip (calc_diffraction_gain)")
{
    arma::mat cube = make_cube();
    double fRef_rf = ac2rf(1000.0);

    // att = 10 dB baseline penetration loss, attB = 0 (flat), coincidence dip of -6 dB at 2 kHz
    // with quality factor 5. Negative coiA = dip (more transmission at coincidence).
    double att = 10.0, coiQ = 5.0, coiA = -6.0;
    double coiF_ac = 2000.0;
    double coiF_rf_GHz = ac2rf(coiF_ac) / 1.0e9;

    auto run = [&](double f_ac)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        single_material({{"a", 1.0},
                         {"att", att},
                         {"coiF", coiF_rf_GHz},
                         {"coiQ", coiQ},
                         {"coiA", coiA},
                         {"fRef", fRef_rf / 1.0e9}},
                        mtl_ind, mtl);

        arma::mat orig = {{-10.0, 0.0, 0.5}};
        arma::mat dest = {{0.5, 0.0, 0.5}}; // ends inside -> single o->i interface
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(&orig, &dest, &cube, &mtl_ind, &mtl, ac2rf(f_ac),
                                                    0, &gain, nullptr, 0, nullptr, 0, 0, true);
        return gain(0);
    };

    // Reference: att(f) = att*(f/fRef)^0 + coiA / (1 + (coiQ*(f-coiF)/coiF)^2), clamped >= 0,
    // expressed in the mapped GHz domain exactly as interface_loss_dB sees it.
    auto ref_interface_gain = [&](double f_ac)
    {
        double fGHz = ac2rf(f_ac) / 1.0e9;
        double dB = att; // attB = 0
        double x = coiQ * (fGHz - coiF_rf_GHz) / coiF_rf_GHz;
        dB += coiA / (1.0 + x * x);
        if (dB < 0.0)
            dB = 0.0;
        return std::pow(10.0, -0.1 * dB);
    };

    // At the coincidence frequency the dip is at full depth: att + coiA = 10 - 6 = 4 dB.
    CHECK(run(2000.0) == Approx(std::pow(10.0, -0.1 * 4.0)).epsilon(1e-5));
    CHECK(run(2000.0) == Approx(ref_interface_gain(2000.0)).epsilon(1e-5));

    // Off the dip the loss returns to the baseline; transmission must be lower than at coincidence.
    CHECK(run(1000.0) == Approx(ref_interface_gain(1000.0)).epsilon(1e-5));
    CHECK(run(4000.0) == Approx(ref_interface_gain(4000.0)).epsilon(1e-5));
    CHECK(run(2000.0) > run(1000.0)); // dip raises transmission at coincidence
    CHECK(run(2000.0) > run(4000.0));
}

// =====================================================================================
// 4. Permittivity resonance (resF/resQ/resS) — absorption peak via reflection
// =====================================================================================
//
// A resonant absorber shows an absorption peak at its tuning frequency. The Lorentz pole lives in
// the Fresnel permittivity, so the peak appears in the reflection branch as 1 - |R|^2. We probe the
// reflected power directly with scalar reflection (type 3) at normal incidence and confirm that
// absorption is higher at resF than off resonance, matching the closed-form Fresnel value.
TEST_CASE("Acoustic - Permittivity resonance absorption peak (ray_mesh_interact)")
{
    arma::mat cube = make_cube();
    double fRef_rf = ac2rf(1000.0);

    // Rigid-ish baseline (eps small) so off-resonance reflection is high (low absorption); the
    // resonance adds a complex pole that drops |R| near resF. resF at 500 Hz acoustic.
    double a_base = 0.05;
    double resF_ac = 500.0;
    double resF_GHz = ac2rf(resF_ac) / 1.0e9;
    double resQ = 8.0, resS = 0.4;

    // Probe |R|^2 with the resonance active vs. an otherwise identical material with res* removed,
    // through the same code path and geometry. Tests the physical claim directly: the Lorentz pole
    // adds loss near resF, so reflected power drops and absorption rises there.
    auto reflect_gain = [&](double f_ac, bool with_resonance)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        std::vector<std::pair<std::string, double>> cols = {{"a", a_base}, {"fRef", fRef_rf / 1.0e9}};
        if (with_resonance)
        {
            cols.push_back({"resF", resF_GHz});
            cols.push_back({"resQ", resQ});
            cols.push_back({"resS", resS});
        }
        single_material(cols, mtl_ind, mtl);

        arma::mat orig = {{-10.0, 0.0, 0.5}};
        arma::mat dest = {{10.0, 0.0, 0.5}};
        arma::mat fbs, sbs;
        arma::u32_vec fbs_ind, sbs_ind;
        quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

        arma::mat origN, destN;
        arma::vec gainN;
        quadriga_lib::ray_mesh_interact<double>(3, ac2rf(f_ac), &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl,
                                                &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr,
                                                &origN, &destN, &gainN);
        return gainN(0); // |R|^2
    };

    double R_on_res = reflect_gain(resF_ac, true);   // at resF, resonance active
    double R_off_res = reflect_gain(resF_ac, false); // at resF, no resonance (baseline)

    // 1) At resonance the pole adds loss -> reflected power drops -> absorption rises.
    CHECK(R_on_res < R_off_res);
    CHECK((1.0 - R_on_res) > (1.0 - R_off_res));

    // 2) Convergence edge is ABOVE resF: the pole decays as ~1/f^2 there, so the resonant material
    //    reverts to the baseline. (Below resF the pole tends to the static offset resS and does NOT
    //    vanish, so the low-frequency side is not a convergence edge.)
    double f_far = 20000.0; // 40x above resF = 500 Hz; pole tail ~ (resF/f)^2 is negligible
    CHECK(std::abs(reflect_gain(f_far, true) - reflect_gain(f_far, false)) < 1e-3);
}

// =====================================================================================
// 5a. EM <-> acoustic convergence: terms structurally absent (m = res* = coi* = 0)
// =====================================================================================
//
// With none of the acoustic columns present, an acoustic-domain call must reproduce the plain
// scalar-EM result bit-for-bit (the new terms are exact no-ops). Checked for both functions.
TEST_CASE("Acoustic - Convergence to scalar EM when acoustic terms absent")
{
    arma::mat cube = make_cube();

    // Plain lossy dielectric, scalar mode, no acoustic columns at all.
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    single_material({{"a", 2.5}, {"c", 0.02}, {"fRef", 1.0}}, mtl_ind, mtl);

    // Same material, but with the acoustic columns present and set to their inert defaults.
    arma::uvec mtl_ind2;
    std::unordered_map<std::string, std::vector<double>> mtl2;
    single_material({{"a", 2.5}, {"c", 0.02}, {"fRef", 1.0}, {"m", 0.0}, {"resF", 0.0}, {"resQ", 0.0}, {"resS", 0.0}, {"coiF", 0.0}, {"coiQ", 0.0}, {"coiA", 0.0}},
                    mtl_ind2, mtl2);

    double f = 3.0e9;

    // --- ray_mesh_interact: reflection (3) and transmission (4) must match between the two maps.
    arma::mat orig = {{-1.5, 0.0, 0.0}};
    arma::mat dest = {{0.0, 0.0, 1.5}}; // 45 deg incidence on the west face
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    for (int itype : {3, 4})
    {
        arma::mat oN1, dN1, oN2, dN2;
        arma::vec gN1, gN2;
        quadriga_lib::ray_mesh_interact<double>(itype, f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind, &mtl,
                                                &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &oN1, &dN1, &gN1);
        quadriga_lib::ray_mesh_interact<double>(itype, f, &orig, &dest, &fbs, &sbs, &cube, &mtl_ind2, &mtl2,
                                                &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &oN2, &dN2, &gN2);
        CHECK(arma::approx_equal(gN1, gN2, "absdiff", 1e-12));
    }

    // --- calc_diffraction_gain (scalar): gain and coord must match between the two maps.
    arma::mat dorig = {{-10.0, 0.0, 0.5}};
    arma::mat ddest = {{0.5, 0.0, 0.5}};
    arma::vec g1, g2;
    arma::cube c1, c2;
    quadriga_lib::calc_diffraction_gain(&dorig, &ddest, &cube, &mtl_ind, &mtl, f, 3, &g1, &c1, 0, nullptr, 0, 0, true);
    quadriga_lib::calc_diffraction_gain(&dorig, &ddest, &cube, &mtl_ind2, &mtl2, f, 3, &g2, &c2, 0, nullptr, 0, 0, true);
    CHECK(arma::approx_equal(g1, g2, "absdiff", 1e-12));
    CHECK(arma::approx_equal(c1, c2, "absdiff", 1e-12));
}

// =====================================================================================
// 5b. EM <-> acoustic convergence: terms present but evaluated far from their features
// =====================================================================================
//
// The resonance and coincidence Lorentzians decay as 1/Q^2-scaled distance from their centers. Far
// enough away, an acoustic material with active res*/coi* columns must converge to the same result
// as the bare baseline material (terms numerically negligible). The mass term is also kept inert
// here by staying in the clamp region.
TEST_CASE("Acoustic - Convergence away from resonance and coincidence")
{
    arma::mat cube = make_cube();
    double fRef_rf = ac2rf(1000.0);

    // Baseline acoustic material: rigid-ish, modest penetration loss, no acoustic features.
    arma::uvec mtl_base_i;
    std::unordered_map<std::string, std::vector<double>> mtl_base;
    single_material({{"a", 0.05}, {"att", 8.0}, {"fRef", fRef_rf / 1.0e9}}, mtl_base_i, mtl_base);

    // Same baseline, plus a sharp resonance and a sharp coincidence dip, both centered low (200 Hz)
    // and high (300 Hz), with high quality factors so they are localized.
    arma::uvec mtl_feat_i;
    std::unordered_map<std::string, std::vector<double>> mtl_feat;
    single_material({{"a", 0.05}, {"att", 8.0}, {"fRef", fRef_rf / 1.0e9}, {"resF", ac2rf(200.0) / 1.0e9}, {"resQ", 50.0}, {"resS", 0.3}, {"coiF", ac2rf(300.0) / 1.0e9}, {"coiQ", 50.0}, {"coiA", -5.0}},
                    mtl_feat_i, mtl_feat);

    // Probe far above both features (12 kHz): both Lorentzians are deep in their tails.
    double f_probe = ac2rf(12000.0);

    // --- Reflection (type 3): resonance feature affects |R|^2; must converge to baseline.
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};
    arma::mat fbs, sbs;
    arma::u32_vec fbs_ind, sbs_ind;
    quadriga_lib::ray_triangle_intersect(&orig, &dest, &cube, &fbs, &sbs, NULL, &fbs_ind, &sbs_ind);

    arma::mat oB, dB_, oF, dF;
    arma::vec gB, gF;
    quadriga_lib::ray_mesh_interact<double>(3, f_probe, &orig, &dest, &fbs, &sbs, &cube, &mtl_base_i, &mtl_base,
                                            &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &oB, &dB_, &gB);
    quadriga_lib::ray_mesh_interact<double>(3, f_probe, &orig, &dest, &fbs, &sbs, &cube, &mtl_feat_i, &mtl_feat,
                                            &fbs_ind, &sbs_ind, nullptr, nullptr, nullptr, &oF, &dF, &gF);
    // Far from resF (Q = 50), the pole contribution is tiny but not exactly zero.
    CHECK(arma::approx_equal(gB, gF, "absdiff", 1e-3));

    // --- Transmission (type 4) via diffraction: coincidence dip affects the interface loss.
    arma::mat dorig = {{-10.0, 0.0, 0.5}};
    arma::mat ddest = {{0.5, 0.0, 0.5}};
    arma::vec gdB, gdF;
    quadriga_lib::calc_diffraction_gain<double>(&dorig, &ddest, &cube, &mtl_base_i, &mtl_base, f_probe, 0, &gdB, nullptr, 0, nullptr, 0, 0, true);
    quadriga_lib::calc_diffraction_gain<double>(&dorig, &ddest, &cube, &mtl_feat_i, &mtl_feat, f_probe, 0, &gdF, nullptr, 0, nullptr, 0, 0, true);
    CHECK(arma::approx_equal(gdB, gdF, "absdiff", 1e-3));
}

// =====================================================================================
// 6. Cross-method calibration: full pass-through via ray_mesh_interact vs calc_diffraction_gain
// =====================================================================================
//
// A ray passing through a slab pays entry + exit interface losses and the in-medium traversal.
// calc_diffraction_gain does the same for a bundle of rays on the Fresnel arc. If the slab is large
// enough to fully obstruct the bundle (no edge diffraction) and thin/flat (all rays ~normal, same
// thickness), both methods must yield the same pass-through gain. This also validates that the two
// reflection paths agree on the single dense->light crossing every slab has: EM passes regardless
// (both override the exit), scalar passes only if transition_gain_linear has the `|| scalar_mode`
// guard matching ray_mesh_interact's gated override.
TEST_CASE("Acoustic - Pass-through calibration: ray_mesh_interact vs calc_diffraction_gain")
{
    double t = 0.1;  // slab thickness [m] (thin)
    double L = 20.0; // slab half-extent in y,z [m] (large: >> Fresnel radius at the slab)
    double d = 50.0; // orig/dest distance from the slab center [m] (large: keeps the bundle tight)

    // Thin slab = cube scaled by t/2 in x and L in y,z (winding/normals inherited from the cube).
    arma::mat slab = make_cube();
    for (arma::uword c = 0; c < 9; ++c)
        slab.col(c) *= (c % 3 == 0) ? (t / 2.0) : L; // x-cols 0,3,6 -> t/2; y,z -> L

    double f_ac = 4000.0;
    double f_rf = ac2rf(f_ac);
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;

    // Moderate dielectric (eps = 2.5) with some penetration loss and in-medium loss.
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl;
    single_material({{"a", 2.5}, {"att", 3.0}, {"alpha", 2.0}, {"fRef", fRef_GHz}}, mtl_ind, mtl);

    for (bool scalar : {false, true})
    {
        int itype = scalar ? 4 : 1;

        // --- Reference: single-ray pass-through = entry call * exit call. ---
        // Call A: entry (outside -> front face). ray_starts_inside = false -> no medium loss.
        arma::mat origA = {{-d, 0.0, 0.0}}, destA = {{d, 0.0, 0.0}};
        arma::mat fbsA, sbsA;
        arma::u32_vec fiA, siA;
        quadriga_lib::ray_triangle_intersect(&origA, &destA, &slab, &fbsA, &sbsA, NULL, &fiA, &siA);
        arma::mat oA, dA;
        arma::vec gA;
        quadriga_lib::ray_mesh_interact<double>(itype, f_rf, &origA, &destA, &fbsA, &sbsA, &slab,
                                                &mtl_ind, &mtl, &fiA, &siA, nullptr, nullptr, nullptr,
                                                &oA, &dA, &gA);
        double gain_entry = gA(0);

        // Call B: exit (just inside front face -> back face). ray_starts_inside = true -> medium
        // loss over the full thickness.
        arma::mat origB = {{-t / 2.0 + 1e-4, 0.0, 0.0}}, destB = {{d, 0.0, 0.0}};
        arma::mat fbsB, sbsB;
        arma::u32_vec fiB, siB;
        quadriga_lib::ray_triangle_intersect(&origB, &destB, &slab, &fbsB, &sbsB, NULL, &fiB, &siB);
        arma::mat oB, dB_;
        arma::vec gB, olenB;
        quadriga_lib::ray_mesh_interact<double>(itype, f_rf, &origB, &destB, &fbsB, &sbsB, &slab,
                                                &mtl_ind, &mtl, &fiB, &siB, nullptr, nullptr, nullptr,
                                                &oB, &dB_, &gB, nullptr, nullptr, nullptr, &olenB);
        double gain_exit = gB(0);
        CHECK(std::abs(olenB(0) - 1e-4 - t) < 1e-3); // traversed ~ full thickness

        double gain_rmi = gain_entry * gain_exit;

        // --- Full bundle through the same slab. ---
        arma::mat origD = {{-d, 0.0, 0.0}}, destD = {{d, 0.0, 0.0}};
        arma::vec gD;
        quadriga_lib::calc_diffraction_gain<double>(&origD, &destD, &slab, &mtl_ind, &mtl, f_rf,
                                                    3, &gD, nullptr, 0, nullptr, 0, 0, scalar);
        double gain_diff = gD(0);

        // Same physical pass-through; residual is the bundle's small angle/path spread. Compare in dB.
        double dB_rmi = -10.0 * std::log10(gain_rmi);
        double dB_diff = -10.0 * std::log10(gain_diff);
        INFO("mode " << (scalar ? "scalar" : "EM") << ": rmi=" << dB_rmi << " dB, diff=" << dB_diff << " dB");
        CHECK(std::abs(dB_rmi - dB_diff) < 0.02);
    }
}