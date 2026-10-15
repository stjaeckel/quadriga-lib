// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// ray_mesh_interact, acoustic: cases that evaluate a material with the mass column (m > 0).

#include "ray_test_helpers.hpp"

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

TEST_CASE("Thin sheet - mass-law angle dependence", "[acoustic]")
{
    // The mass term is the only angle-dependent quantity in the sheet composite that is not a
    // Fresnel amplitude. A mass-law sheet is one plate: its first face passes what it absorbs, the
    // flux transmittance a = 1 - r^2 of air | sheet, and the plate passes the incoherent series of
    // the face and its term, a tau / (a + tau - a tau) (the plate rule, md/acoustic.md 10 N1). Scalar,
    // the acoustic types. The case inverts the rule, tau = T a / (a - T + a T), which leaves the mass
    // term alone, and compares it with the closed form.
    const QdMassSheet F;
    const double half = 200.0;
    arma::mat mesh = qd_quad_x(half, 0.0);

    auto dB = [](double p)
    { return (p > 0.0) ? 10.0 * std::log10(p) : -std::numeric_limits<double>::infinity(); };

    const double arg = F.arg();
    const double A0 = F.m * std::log10(arg);           // normal-incidence term
    const double X0 = std::pow(arg, 0.05 * F.m);       // = 10^(A0/20)
    const double ct_knee = 1.0 / X0;                   // x = 1
    const double th_knee = std::acos(ct_knee) * 180.0 / arma::datum::pi;

    QD_DIAG("\n# diag: mass-law angle dependence, sheet thk 12.5 mm, eps 5.24, m 12 dB/decade\n");
    QD_DIAG("# (f/fRef) * thk = %.4f; knee at %.6f deg\n", arg, th_knee);
    QD_DIAG("angle,cos,a,T_dB,TL_meas_dB,TL_dB,d_dB,R_meas,R_ref\n");

    // the face of air | sheet, scalar (TE form): its reflectance, and the mass term read off a
    // transmit port through the plate rule
    auto face_R = [&](double thd)
    {
        QdMassRef Rf = qd_mass_ref(F.eps, thd);
        double ct = std::cos(thd * arma::datum::pi / 180.0), zc = std::sqrt(F.eps) * Rf.cts, r = (ct - zc) / (ct + zc);
        return r * r;
    };
    auto tl_of = [&](double T, double thd)
    {
        double a = 1.0 - face_R(thd);
        return -dB(T * a / (a - T + a * T));
    };
    double tl_prev = 1.0e300;
    double tl_at[2] = {0.0, 0.0}; // 30 deg and 60 deg, for the slope check

    const double sweep[] = {0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 78.0, 80.0,
                            85.0, 88.0, 89.0, 89.5, 89.6, 89.75, 89.9};

    for (double thd : sweep)
    {
        double ct = std::cos(thd * arma::datum::pi / 180.0);
        const double Rface = face_R(thd), a = 1.0 - Rface;

        double T = F.run(mesh, half, thd, 4, F.m);
        double R = F.run(mesh, half, thd, 3, F.m);

        INFO("angle " << thd);
        REQUIRE(std::isfinite(T));
        REQUIRE(std::isfinite(R));
        CHECK(T >= 0.0);
        CHECK(R >= 0.0);
        CHECK(R <= 1.0);

        double tl_meas = tl_of(T, thd); // the mass term, read off the composite through the plate rule
        double tl_want = qd_mass_TL(F.m, arg, ct), tau_want = std::pow(10.0, -0.1 * tl_want);

        QD_DIAG("%.4f,%.6e,%.6f,%.6f,%.6f,%.6f,%.3e,%.9f,%.9f\n",
                thd, ct, a, dB(T), tl_meas, tl_want, tl_meas - tl_want, R, Rface);

        // The composite as a whole, which pins the plate rule and the mass term together.
        CHECK(std::abs(dB(T) - dB(a * tau_want / (a + tau_want - a * tau_want))) < 1.0e-6);

        // The mass term on its own.
        CHECK(std::abs(tl_meas - tl_want) < 1.0e-6);

        // The reflect port of a mass-law sheet is the bare near interface. It carries no mass term
        // and no internal series, so it is a single-interface Fresnel value at every angle.
        CHECK(std::abs(R - Rface) < 1.0e-9);

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
    // 0.05 dB at 60 deg for this material.
    double c30 = std::cos(30.0 * arma::datum::pi / 180.0);
    double c60 = std::cos(60.0 * arma::datum::pi / 180.0);
    double slope = (tl_at[0] - tl_at[1]) / (std::log10(c30) - std::log10(c60));
    QD_DIAG("# angle slope, 30 to 60 deg: %.6f dB per decade of cos(theta)\n", slope);
    CHECK(slope > 19.5);
    CHECK(slope < 20.0);

    // Normal incidence is the material table's own value, m log10((f / fRef) thk), clipped at zero.
    // This is what keeps the public medium_gain entry point, and every table calibrated through it,
    // on the column definition.
    {
        double T = F.run(mesh, half, 0.0, 4, F.m);
        double tl = tl_of(T, 0.0);
        CHECK(std::abs(tl - std::max(0.0, F.m * std::log10(arg))) < 1.0e-9);
    }

    // There is no cutoff in angle. The knee sits where x = 1, which is fixed by m, the frequency
    // and the surface mass, and by no geometry. The term there is A * log10(2) / log10(1 + X0^2),
    // which is 10 * log10(2) up to the normalization.
    {
        double want = A0 * std::log10(2.0) / std::log10(1.0 + X0 * X0);
        double T = F.run(mesh, half, th_knee, 4, F.m);
        double tl = tl_of(T, th_knee);
        INFO("knee at " << th_knee << " deg");
        CHECK(std::abs(tl - want) < 1.0e-6);
        CHECK(std::abs(want - 10.0 * std::log10(2.0)) < 0.01);
    }

    // Past the knee the term tapers rather than switching off: still positive at 89.9 deg, and
    // decaying as the square of the cosine.
    {
        double T = F.run(mesh, half, 89.9, 4, F.m);
        double tl = tl_of(T, 89.9);
        CHECK(tl > 0.0);
        CHECK(tl < 0.05);

        double T2 = F.run(mesh, half, 89.95, 4, F.m);
        double tl2 = tl_of(T2, 89.95);
        CHECK(tl2 > 0.0);
        CHECK(tl2 < tl);
        CHECK(std::abs(tl / tl2 - 4.0) < 0.05); // cos halves, the term quarters
    }

    // The mass term is a transmission surrogate and must not reach the reflect port. Doubling m
    // leaves the reflected gain bit-identical.
    for (double thd : {30.0, 70.0, 85.0, 89.9})
    {
        double Ra = F.run(mesh, half, thd, 3, F.m);
        double Rb = F.run(mesh, half, thd, 3, 2.0 * F.m);
        INFO("reflect port, angle " << thd);
        CHECK(std::abs(Ra - Rb) < 1.0e-12);
    }
}

// Test 3 - medium_gain in-medium loss (mass law, excess absorption)
// medium_gain is the standalone in-medium term. It applies mass-law transmission and distance-linear
// absorption, and must not include the interface penetration loss att.
TEST_CASE("Acoustic - medium_gain in-medium loss", "[acoustic]")
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
