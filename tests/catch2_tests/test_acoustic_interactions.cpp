// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib - see LICENSE for terms.

// Acoustic physics validation for the material-interaction stack.
//
// The acoustic solver reuses the electromagnetic ray tracer under a fixed frequency mapping
// (acoustic Hz -> radio Hz via the speed-of-light / speed-of-sound ratio). This suite validates the
// acoustic physics only, on three surfaces:
//
//   - Isolated material terms are checked against closed-form references through the public
//     medium_gain (in-medium loss: conductivity, excess absorption, mass law), interface_gain
//     (penetration loss and coincidence), and refractive_index (bulk index).
//   - Single-interface reflection and transmission are checked through ray_mesh_interact
//     (scalar types 3 and 4).
//   - Integrated slab and partition behavior is checked through calc_diffraction_gain, which drives
//     ray_state_update internally, and is cross-checked against an explicit
//     interface x medium x interface composition.
//
// State-machine mechanics of ray_state_update (bit encoding, re-emit gating) are covered by their
// own dedicated spec tests; here only the resolved, observable acoustic result is validated.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;

#include "quadriga_tools.hpp"

#include <cmath>
#include <complex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Frequency mapping constants: acoustic Hz -> mapped radio Hz.
static constexpr double C_LIGHT = 299792458.0;
static constexpr double C_SOUND = 342.77;
static constexpr double AC2RF = C_LIGHT / C_SOUND; // acoustic Hz -> radio Hz

static inline double ac2rf(double f_acoustic_Hz) { return f_acoustic_Hz * AC2RF; }
static inline double rf2ac(double f_radio_Hz) { return f_radio_Hz / AC2RF; }

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
                                            nullptr, nullptr, nullptr, // trivec, tridir, orig_length
                                            nullptr, nullptr,          // origN, destN
                                            nullptr, nullptr,          // fbsN, sbsN
                                            &gainN);                   // gainN
    return gainN.n_elem ? gainN(0) : 0.0;
}

// Test 1 - Frequency mapping round-trip
// The acoustic-to-radio mapping is a pure scalar; the inverse must recover the input exactly.
TEST_CASE("Acoustic - frequency mapping round-trip")
{
    // 1 kHz acoustic maps to ~0.875 GHz radio (the format_materials.md reference point).
    CHECK(ac2rf(1000.0) / 1.0e9 == Approx(0.875).epsilon(1e-3));

    // Hz -> GHz -> Hz must be exact to floating-point round-off, both directions.
    for (double f : {20.0, 125.0, 500.0, 1000.0, 2000.0, 8000.0, 16000.0})
    {
        CHECK(rf2ac(ac2rf(f)) == Approx(f).epsilon(1e-12));
        CHECK(ac2rf(rf2ac(f)) == Approx(f).epsilon(1e-12));
    }

    // The mapping constant itself.
    CHECK(AC2RF == Approx(874636.0).epsilon(1e-4));
}

// Test 2 - refractive_index uses the base permittivity only
// The bulk index is n = Re(sqrt(eta_base * mu)); resonance and coincidence are surface effects and
// must not leak into it.
TEST_CASE("Acoustic - refractive_index base permittivity only")
{
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;

    // Index 0 selects air regardless of the table -> n = 1.
    std::unordered_map<std::string, std::vector<double>> mtl_air;
    mtl_air["a"] = {4.0};
    mtl_air["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::refractive_index<double>(mtl_air, 0, ac2rf(1000.0)) == Approx(1.0).epsilon(1e-12));

    // Non-dispersive dielectric: n = sqrt(a).
    std::unordered_map<std::string, std::vector<double>> mtl4;
    mtl4["a"] = {4.0};
    mtl4["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::refractive_index<double>(mtl4, 1, ac2rf(1000.0)) == Approx(2.0).epsilon(1e-9));

    // Dispersion via b: eps = a*(f/fRef)^b -> at 2*fRef, eps = 4*4 = 16, n = 4.
    std::unordered_map<std::string, std::vector<double>> mtl_disp;
    mtl_disp["a"] = {4.0};
    mtl_disp["b"] = {2.0};
    mtl_disp["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::refractive_index<double>(mtl_disp, 1, ac2rf(2000.0)) == Approx(4.0).epsilon(1e-9));

    // Adding resonance and coincidence poles must not change the index.
    std::unordered_map<std::string, std::vector<double>> mtl_res = mtl4;
    mtl_res["resF"] = {ac2rf(1000.0) / 1.0e9};
    mtl_res["resQ"] = {8.0};
    mtl_res["resS"] = {0.5};
    mtl_res["coiF"] = {ac2rf(2000.0) / 1.0e9};
    mtl_res["coiQ"] = {5.0};
    mtl_res["coiA"] = {-6.0};
    CHECK(quadriga_lib::refractive_index<double>(mtl_res, 1, ac2rf(1000.0)) ==
          Approx(quadriga_lib::refractive_index<double>(mtl4, 1, ac2rf(1000.0))).epsilon(1e-12));
}

// Test 3 - medium_gain in-medium loss (mass law, excess absorption)
// medium_gain is the standalone in-medium term. It applies mass-law transmission and distance-linear
// absorption, and must not include the interface penetration loss att.
TEST_CASE("Acoustic - medium_gain in-medium loss")
{
    double fRef_rf = ac2rf(1000.0);
    double fRef_GHz = fRef_rf / 1.0e9;

    // Closed-form mass-law reference at normal incidence (geometric path equals refracted path).
    auto ref_mass_gain = [&](double f_ac, double m_slope, double dist)
    {
        double f_rel = ac2rf(f_ac) / fRef_rf; // equals f_ac / 1000
        double arg = f_rel * dist;
        double dB = (arg > 1.0) ? m_slope * std::log10(arg) : 0.0;
        return std::pow(10.0, -0.1 * dB);
    };

    // Mass-law-only material: eps = 1 (no Fresnel, no conductivity), no excess absorption.
    std::unordered_map<std::string, std::vector<double>> mtl_mass;
    mtl_mass["a"] = {1.0};
    mtl_mass["m"] = {20.0};
    mtl_mass["fRef"] = {fRef_GHz};

    double dist = 2.0;

    // Point value: at 4 kHz, (f/fRef)*dist = 8, so loss = 20*log10(8).
    CHECK(quadriga_lib::medium_gain<double>(mtl_mass, 1, dist, ac2rf(4000.0)) ==
          Approx(ref_mass_gain(4000.0, 20.0, dist)).epsilon(1e-9));

    // +6.02 dB/octave: doubling frequency adds 20*log10(2).
    double g2k = quadriga_lib::medium_gain<double>(mtl_mass, 1, dist, ac2rf(2000.0));
    double g4k = quadriga_lib::medium_gain<double>(mtl_mass, 1, dist, ac2rf(4000.0));
    CHECK(-10.0 * std::log10(g4k / g2k) == Approx(6.0206).epsilon(1e-4));

    // Clamp region: (f/fRef)*dist <= 1 yields no mass loss. At 250 Hz, 0.25*2 = 0.5.
    CHECK(quadriga_lib::medium_gain<double>(mtl_mass, 1, dist, ac2rf(250.0)) == Approx(1.0).epsilon(1e-9));

    // Excess absorption (alpha, dB/m) is linear in distance and independent of att.
    std::unordered_map<std::string, std::vector<double>> mtl_abs;
    mtl_abs["a"] = {1.0};
    mtl_abs["alpha"] = {5.0}; // 5 dB/m at fRef, alphaB = 0 -> frequency-flat
    mtl_abs["att"] = {10.0};  // interface term, must not enter medium_gain
    mtl_abs["fRef"] = {fRef_GHz};

    double gd = quadriga_lib::medium_gain<double>(mtl_abs, 1, 1.0, ac2rf(4000.0));
    double g2d = quadriga_lib::medium_gain<double>(mtl_abs, 1, 2.0, ac2rf(4000.0));
    CHECK(-10.0 * std::log10(gd) == Approx(5.0).epsilon(1e-6));   // 5 dB over 1 m
    CHECK(-10.0 * std::log10(g2d) == Approx(10.0).epsilon(1e-6)); // exactly double over 2 m

    // Dropping att changes nothing: it is not an in-medium term.
    std::unordered_map<std::string, std::vector<double>> mtl_noatt = mtl_abs;
    mtl_noatt["att"] = {0.0};
    CHECK(quadriga_lib::medium_gain<double>(mtl_abs, 1, 1.0, ac2rf(4000.0)) ==
          Approx(quadriga_lib::medium_gain<double>(mtl_noatt, 1, 1.0, ac2rf(4000.0))).epsilon(1e-12));
}

// Test 4 - interface_gain penetration loss and coincidence
// interface_gain is the standalone interface term: a frequency-scaled penetration loss att plus an
// optional coincidence Lorentzian, clamped so it never becomes gain.
TEST_CASE("Acoustic - interface_gain penetration and coincidence")
{
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;
    double coiF_GHz = ac2rf(2000.0) / 1.0e9;

    // Closed-form interface reference (att power-law + coincidence Lorentzian, clamped at >= 0 dB).
    auto ref_if_gain = [&](double f_ac, double att, double attB, double coiQ, double coiA)
    {
        double fGHz = ac2rf(f_ac) / 1.0e9;
        double dB = att * std::pow(fGHz / fRef_GHz, attB);
        if (coiA != 0.0)
        {
            double x = coiQ * (fGHz - coiF_GHz) / coiF_GHz;
            dB += coiA / (1.0 + x * x);
        }
        return dB < 0.0 ? 1.0 : std::pow(10.0, -0.1 * dB);
    };

    // Coincidence dip: 10 dB baseline with a -6 dB dip at 2 kHz, Q = 5.
    std::unordered_map<std::string, std::vector<double>> mtl_dip;
    mtl_dip["att"] = {10.0};
    mtl_dip["coiF"] = {coiF_GHz};
    mtl_dip["coiQ"] = {5.0};
    mtl_dip["coiA"] = {-6.0};
    mtl_dip["fRef"] = {fRef_GHz};

    // At coincidence the net loss is 10 - 6 = 4 dB.
    CHECK(quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(2000.0)) ==
          Approx(std::pow(10.0, -0.1 * 4.0)).epsilon(1e-6));
    CHECK(quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(2000.0)) ==
          Approx(ref_if_gain(2000.0, 10.0, 0.0, 5.0, -6.0)).epsilon(1e-6));

    // The dip raises transmission at coincidence relative to off-band.
    CHECK(quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(2000.0)) >
          quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(1000.0)));
    CHECK(quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(2000.0)) >
          quadriga_lib::interface_gain<double>(mtl_dip, 1, ac2rf(4000.0)));

    // Coincidence stop-band: positive coiA increases loss at coiF.
    std::unordered_map<std::string, std::vector<double>> mtl_stop;
    mtl_stop["att"] = {3.0};
    mtl_stop["coiF"] = {coiF_GHz};
    mtl_stop["coiQ"] = {5.0};
    mtl_stop["coiA"] = {6.0};
    mtl_stop["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::interface_gain<double>(mtl_stop, 1, ac2rf(2000.0)) <
          quadriga_lib::interface_gain<double>(mtl_stop, 1, ac2rf(1000.0)));

    // Clamp: a dip deeper than the baseline cannot turn into gain -> clamps to unity.
    std::unordered_map<std::string, std::vector<double>> mtl_clamp;
    mtl_clamp["att"] = {2.0};
    mtl_clamp["coiF"] = {coiF_GHz};
    mtl_clamp["coiQ"] = {5.0};
    mtl_clamp["coiA"] = {-6.0};
    mtl_clamp["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::interface_gain<double>(mtl_clamp, 1, ac2rf(2000.0)) == Approx(1.0).epsilon(1e-9));

    // Penetration loss scales with frequency via attB: att ~ (f/fRef)^1 doubles at 2*fRef.
    std::unordered_map<std::string, std::vector<double>> mtl_attB;
    mtl_attB["att"] = {6.0};
    mtl_attB["attB"] = {1.0};
    mtl_attB["fRef"] = {fRef_GHz};
    CHECK(quadriga_lib::interface_gain<double>(mtl_attB, 1, ac2rf(2000.0)) ==
          Approx(std::pow(10.0, -0.1 * 12.0)).epsilon(1e-6));
}

// Test 5 - permittivity resonance absorption peak (ray_mesh_interact type 3)
// A Lorentz permittivity pole adds loss on resonance, lowering the reflected power, and decays well
// above the resonance frequency.
TEST_CASE("Acoustic - permittivity resonance absorption peak")
{
    arma::mat cube_mesh = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;
    double resF_ac = 500.0;
    double resF_GHz = ac2rf(resF_ac) / 1.0e9;
    double a_base = 0.05; // rigid-ish: high baseline reflection, so added absorption is visible

    auto reflect_gain = [&](double f_ac, bool with_res)
    {
        std::vector<std::pair<std::string, double>> cols = {{"a", a_base}, {"fRef", fRef_GHz}};
        if (with_res)
        {
            cols.push_back({"resF", resF_GHz});
            cols.push_back({"resQ", 8.0});
            cols.push_back({"resS", 0.4});
        }
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl;
        single_material(cols, mtl_ind, mtl);
        arma::mat orig = {{-10.0, 0.0, 0.5}}, dest = {{10.0, 0.0, 0.5}};
        return rmi_gain(3, ac2rf(f_ac), orig, dest, cube_mesh, mtl_ind, mtl);
    };

    double R_on = reflect_gain(resF_ac, true);
    double R_off = reflect_gain(resF_ac, false);
    CHECK(R_on < R_off);                 // resonance adds loss -> less reflected power
    CHECK((1.0 - R_on) > (1.0 - R_off)); // -> more absorption on resonance

    // Well above resF the pole decays as ~(resF/f)^2 and reverts to the baseline.
    CHECK(std::abs(reflect_gain(20000.0, true) - reflect_gain(20000.0, false)) < 1e-3);
}

// Test 6 - convergence to the scalar reference when acoustic terms are absent
// Adding acoustic columns that are all zero must be a structural no-op: reflection, transmission,
// gain, polarization coefficient, and diffracted path must be identical bit-for-bit.
TEST_CASE("Acoustic - convergence with acoustic terms absent")
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

// Test 7 - convergence away from resonance and coincidence
// Sharp permittivity and coincidence features probed far above their center frequencies must be
// negligible: both reflection and transmission revert to the featureless baseline.
TEST_CASE("Acoustic - convergence away from spectral features")
{
    arma::mat cube_mesh = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;

    arma::uvec mtl_base_i, mtl_feat_i;
    std::unordered_map<std::string, std::vector<double>> mtl_base, mtl_feat;
    single_material({{"a", 0.05}, {"att", 8.0}, {"fRef", fRef_GHz}}, mtl_base_i, mtl_base);
    single_material({{"a", 0.05}, {"att", 8.0}, {"fRef", fRef_GHz}, {"resF", ac2rf(200.0) / 1.0e9}, {"resQ", 50.0}, {"resS", 0.3}, {"coiF", ac2rf(300.0) / 1.0e9}, {"coiQ", 50.0}, {"coiA", -5.0}},
                    mtl_feat_i, mtl_feat);

    double f_probe = ac2rf(12000.0); // far above both features

    // Reflection (type 3): the resonance tail is negligible.
    arma::mat orig = {{-10.0, 0.0, 0.5}}, dest = {{10.0, 0.0, 0.5}};
    double gB = rmi_gain(3, f_probe, orig, dest, cube_mesh, mtl_base_i, mtl_base);
    double gF = rmi_gain(3, f_probe, orig, dest, cube_mesh, mtl_feat_i, mtl_feat);
    CHECK(std::abs(gB - gF) < 1e-3);

    // Transmission via calc_diffraction_gain: the coincidence tail is negligible.
    arma::mat dorig = {{-10.0, 0.0, 0.5}}, ddest = {{0.5, 0.0, 0.5}};
    arma::vec gdB, gdF;
    quadriga_lib::calc_diffraction_gain<double>(dorig, ddest, cube_mesh, mtl_base_i, mtl_base, f_probe, 0,
                                                &gdB, nullptr, nullptr, 0, nullptr, 0, 0, true, 1.0);
    quadriga_lib::calc_diffraction_gain<double>(dorig, ddest, cube_mesh, mtl_feat_i, mtl_feat, f_probe, 0,
                                                &gdF, nullptr, nullptr, 0, nullptr, 0, 0, true, 1.0);
    CHECK(arma::approx_equal(gdB, gdF, "absdiff", 1e-3));
}

// Test 8 - scalar total internal reflection at oblique incidence
// Going from a dense medium (air, eps = 1) into a lighter one (eps < 1), reflection follows the
// closed-form scalar curve across the angle sweep and becomes total past the critical angle:
// reflection saturates at unity and transmission collapses to zero.
TEST_CASE("Acoustic - scalar total internal reflection")
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

// Test 9 - pass-through calibration against an explicit interface x medium x interface composition
// A single ray through a thin slab (thin-slab resolution disabled, eps = 1) equals the naive
// single-pass product of the entry interface, the in-medium loss over the thickness, and the exit
// interface. This ties calc_diffraction_gain's integrated result to the isolated term functions.
TEST_CASE("Acoustic - pass-through calibration")
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

// Test 10 - embedded material counted once (calc_diffraction_gain)
// A smaller cube interpenetrating a larger one, sharing zero permittivity contrast so no reflections
// occur, isolates the penetration loss of the embedded material. Crossing it must charge that loss
// exactly once: 12 dB versus 6 dB is a factor of 10^0.6 in gain.
TEST_CASE("Acoustic - embedded material counted once")
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

// Test 11 - thin-slab Fabry-Perot resolution (calc_diffraction_gain)
// Resolving a thin slab (eps = 0) sums the internal multiple reflections into an Airy transmission.
// With no interface contrast there is nothing to resonate, so resolving reduces to the plain
// in-medium result. With reflective, dispersionless, lossless faces the resolved transmission stays
// physical and ripples with frequency (the Fabry-Perot signature), while the naive single pass,
// having no phase term, is frequency-flat.
TEST_CASE("Acoustic - thin-slab Fabry-Perot resolution")
{
    double t = 0.1, L = 20.0, d = 50.0;
    arma::mat slab = quadriga_lib::cube<double>({t / 2.0, L, L + 0.001});
    double fRef_GHz = ac2rf(1000.0) / 1.0e9;
    arma::mat orig = {{-d, 0.0, 0.0}}, dest = {{d, 0.0, 0.0}};

    auto transmit = [&](const std::vector<std::pair<std::string, double>> &cols, double f_ac, double eps)
    {
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> m;
        single_material(cols, mtl_ind, m);
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, slab, mtl_ind, m, ac2rf(f_ac), 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, true, eps);
        return gain(0);
    };

    // Zero-contrast, lossless slab: no interface to resonate, so resolving is a no-op and both
    // branches reduce to the unit in-medium transmission.
    {
        double g_res = transmit({{"a", 1.0}, {"fRef", fRef_GHz}}, 4000.0, 0.0);
        double g_naive = transmit({{"a", 1.0}, {"fRef", fRef_GHz}}, 4000.0, 1.0);
        CHECK(g_res == Approx(g_naive).epsilon(1e-9));
        CHECK(g_res == Approx(1.0).epsilon(1e-9));
    }

    // Reflective, dispersionless, lossless slab (n = 2). Sweeping frequency changes the optical
    // thickness, so the resolved transmission ripples; the naive single pass does not.
    {
        std::vector<std::pair<std::string, double>> cols = {{"a", 4.0}, {"b", 0.0}, {"fRef", fRef_GHz}};
        std::vector<double> freqs = {3000.0, 3500.0, 4000.0, 4500.0, 5000.0};

        double res_min = 2.0, res_max = -1.0, naive_min = 2.0, naive_max = -1.0;
        for (double fa : freqs)
        {
            double g_res = transmit(cols, fa, 0.0);
            double g_naive = transmit(cols, fa, 1.0);
            CHECK(g_res > 0.0);
            CHECK(g_res <= 1.0 + 1e-9);
            res_min = std::min(res_min, g_res);
            res_max = std::max(res_max, g_res);
            naive_min = std::min(naive_min, g_naive);
            naive_max = std::max(naive_max, g_naive);
        }
        CHECK((res_max - res_min) > 0.05);     // resolved transmission ripples with frequency
        CHECK((naive_max - naive_min) < 1e-6); // naive single pass is frequency-flat
    }
}